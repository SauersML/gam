//! Regression for #803: multinomial predict must score **label-free** new data.
//!
//! `predict_multinomial_formula` replayed the saved termspec directly against
//! the predict frame's raw values. The termspec was frozen against the training
//! table `[response, features...]`, so its feature-column indices are absolute
//! positions in *that* layout. A label-free predict frame `[features...]` puts
//! the features at different indices, so the design rebuild dereferenced a stale
//! index and failed with `feature column N out of bounds`. Multinomial was the
//! lone family that could not predict on feature-only data — the one thing
//! prediction is for — and the only workaround was re-supplying the (unknown, at
//! predict time) response column in its exact training position.
//!
//! The fix realigns the saved indices onto the predict frame's columns *by name*
//! (via `TermCollectionSpec::remap_feature_columns` + the model's
//! `training_headers`), so prediction no longer cares about column order and the
//! response column may be absent.
//!
//! This test fits a 3-class `y ~ x` with the response **first** in the training
//! table (`[y, x]`, the exact layout from the issue, so `x` sits at index 1),
//! then predicts on three differently-shaped frames and checks they all agree:
//!   * label-free `[x]` (the case that used to crash),
//!   * feature/response reordered `[x, y]` (x moved to index 0),
//!   * the original `[y, x]` layout (the previously-only-working case).
//! All three must yield identical, valid `(N, 3)` probability matrices.

use csv::StringRecord;
use gam::families::multinomial::{
    MultinomialFitRequest, fit_penalized_multinomial_formula, predict_multinomial_formula,
};
use gam::{FitConfig, encode_recordswith_inferred_schema, init_parallelism};
use rand::SeedableRng;
use rand::rngs::StdRng;
use rand_distr::{Distribution, Uniform};

const CLASSES: [&str; 3] = ["a", "b", "c"];
const N: usize = 300;

/// Softmax class probabilities at `x` for a well-posed, clearly *non-separable*
/// 3-class problem over bounded `x ∈ [-1.5, 1.5]`: moderate slopes with
/// intercepts so all three classes stay well represented everywhere (no class
/// exceeds ~80% at the edges). This is the regime the penalized softmax solver
/// is happy with — the bug under test is predict-time column realignment, not
/// fit robustness, so the data is deliberately easy to fit. Reference class "c"
/// is pinned to η = 0.
fn class_probs(x: f64) -> [f64; 3] {
    let eta = [0.3 + 0.9 * x, -0.2 - 0.7 * x, 0.0];
    let m = eta.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
    let exps = eta.map(|e| (e - m).exp());
    let z: f64 = exps.iter().sum();
    exps.map(|e| e / z)
}

/// The x grid (and per-row sampled labels) shared by training and prediction.
fn samples() -> Vec<(f64, &'static str)> {
    let mut rng = StdRng::seed_from_u64(0xC0FFEE);
    let ux = Uniform::new(-1.5_f64, 1.5_f64).expect("uniform x");
    let udraw = Uniform::new(0.0_f64, 1.0_f64).expect("uniform draw");
    (0..N)
        .map(|_| {
            let x = ux.sample(&mut rng);
            let p = class_probs(x);
            let u = udraw.sample(&mut rng);
            let label = if u < p[0] {
                CLASSES[0]
            } else if u < p[0] + p[1] {
                CLASSES[1]
            } else {
                CLASSES[2]
            };
            (x, label)
        })
        .collect()
}

/// Training table with the **response column first**: headers `[y, x]`, so the
/// feature `x` is at absolute index 1 — the layout that exposes the bug (the
/// stored termspec carries `x` at index 1, but a label-free predict frame puts
/// it at index 0).
fn training_dataset(rows: &[(f64, &'static str)]) -> gam::data::EncodedDataset {
    let headers = vec!["y".to_string(), "x".to_string()];
    let records = rows
        .iter()
        .map(|(x, label)| StringRecord::from(vec![label.to_string(), x.to_string()]))
        .collect::<Vec<_>>();
    encode_recordswith_inferred_schema(headers, records).expect("encode training dataset")
}

/// Build a predict frame from `(header, column-builder)` pairs so we can place
/// `x` at any index and optionally include / drop the response column.
fn predict_frame(
    xs: &[f64],
    columns: &[(&str, Box<dyn Fn(f64) -> String>)],
) -> gam::data::EncodedDataset {
    let headers = columns
        .iter()
        .map(|(h, _)| h.to_string())
        .collect::<Vec<_>>();
    let records = xs
        .iter()
        .map(|&x| StringRecord::from(columns.iter().map(|(_, f)| f(x)).collect::<Vec<_>>()))
        .collect::<Vec<_>>();
    encode_recordswith_inferred_schema(headers, records).expect("encode predict frame")
}

fn assert_valid_probability_matrix(probs: &ndarray::Array2<f64>, n: usize) {
    assert_eq!(probs.nrows(), n, "one probability row per input row");
    assert_eq!(probs.ncols(), 3, "three classes → (N, 3) probabilities");
    for i in 0..probs.nrows() {
        let mut row_sum = 0.0;
        for k in 0..probs.ncols() {
            let p = probs[[i, k]];
            assert!(
                p.is_finite() && (0.0..=1.0).contains(&p),
                "probability p[{i},{k}] = {p} must lie in [0, 1]"
            );
            row_sum += p;
        }
        assert!(
            (row_sum - 1.0).abs() <= 1e-9,
            "row {i} probabilities must sum to 1 (got {row_sum})"
        );
    }
}

#[test]
fn multinomial_predicts_on_label_free_and_reordered_frames() {
    init_parallelism();
    let rows = samples();
    let train = training_dataset(&rows);
    let config = FitConfig::default();
    let model = fit_penalized_multinomial_formula(&MultinomialFitRequest {
        tol: 1.0e-8,
        ..MultinomialFitRequest::new(&train, "y ~ x", &config)
    })
    .expect("multinomial formula fit must succeed");
    let xs = rows.iter().map(|(x, _)| *x).collect::<Vec<_>>();
    let n = xs.len();

    // (1) Label-free frame: only the feature column `x`. This is what prediction
    // is for, and exactly what used to fail with "feature column 1 out of
    // bounds for 1 columns".
    let label_free = predict_frame(&xs, &[("x", Box::new(|x| x.to_string()))]);
    let probs_label_free =
        predict_multinomial_formula(&model, &label_free).expect("predict on label-free frame");
    assert_valid_probability_matrix(&probs_label_free, n);

    // (2) Feature/response reordered relative to training (`[x, y]` vs `[y, x]`),
    // so `x` is now at index 0. By-name realignment must follow it.
    let reordered = predict_frame(
        &xs,
        &[
            ("x", Box::new(|x| x.to_string())),
            ("y", Box::new(|_x| "a".to_string())),
        ],
    );
    let probs_reordered =
        predict_multinomial_formula(&model, &reordered).expect("predict on reordered frame");
    assert_valid_probability_matrix(&probs_reordered, n);

    // (3) The original `[y, x]` training layout (the previously-only-working
    // case) — must still work and serves as the correctness baseline.
    let with_label = predict_frame(
        &xs,
        &[
            ("y", Box::new(|_x| "a".to_string())),
            ("x", Box::new(|x| x.to_string())),
        ],
    );
    let probs_with_label =
        predict_multinomial_formula(&model, &with_label).expect("predict on full-layout frame");
    assert_valid_probability_matrix(&probs_with_label, n);

    // Column order must not change the prediction: all three frames carry the
    // same feature values, so realignment-by-name must make them identical.
    for i in 0..n {
        for k in 0..3 {
            let base = probs_with_label[[i, k]];
            assert!(
                (probs_label_free[[i, k]] - base).abs() <= 1e-12,
                "label-free prediction differs from full-layout at [{i},{k}]: \
                 {} vs {base}",
                probs_label_free[[i, k]]
            );
            assert!(
                (probs_reordered[[i, k]] - base).abs() <= 1e-12,
                "reordered prediction differs from full-layout at [{i},{k}]: \
                 {} vs {base}",
                probs_reordered[[i, k]]
            );
        }
    }
}

#[test]
fn multinomial_predict_reports_a_genuinely_missing_feature_by_name() {
    init_parallelism();
    let rows = samples();
    let train = training_dataset(&rows);
    let config = FitConfig::default();
    let model = fit_penalized_multinomial_formula(&MultinomialFitRequest {
        tol: 1.0e-8,
        ..MultinomialFitRequest::new(&train, "y ~ x", &config)
    })
    .expect("multinomial formula fit must succeed");
    let xs = rows.iter().map(|(x, _)| *x).collect::<Vec<_>>();

    // A frame that carries neither the feature `x` nor the response: realignment
    // by name must fail with a clear, named diagnostic — not a positional
    // out-of-bounds panic and not a silent wrong-column read.
    let wrong = predict_frame(&xs, &[("not_x", Box::new(|x| x.to_string()))]);
    let err = predict_multinomial_formula(&model, &wrong)
        .expect_err("predict must fail when the fitted feature column is absent");
    let msg = err.to_string();
    assert!(
        msg.contains('x'),
        "error should name the missing feature column 'x', got: {msg}"
    );
}
