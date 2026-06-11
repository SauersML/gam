//! Regression (different angle from `bug_hunt_predict_linear_term_clamped_to_training_range`):
//! the predict-time clamp of continuous covariates to the training range did
//! not only flatten the *point* prediction of a parametric linear term — it
//! also froze the prediction **standard error** at the training-hull boundary.
//!
//! The SE of a linear-predictor value is `sqrt(xᵀ Var(β) x)`. For a model with
//! an intercept and a linear slope, `x = [1, x]`, so the SE is a U-shaped
//! function of `x` that *grows without bound* as `x` moves away from the data
//! centroid. If the covariate is clamped to `[x_min, x_max]` before `xᵀ Var(β) x`
//! is formed, every out-of-hull probe reuses the boundary `x`, so the reported
//! SE plateaus at the boundary value and credible intervals stop widening —
//! over-confident exactly where the model is least supported.
//!
//! This test fits `y ~ x` with a little noise (so `Var(β)` is non-degenerate),
//! predicts with `--uncertainty` at probes reaching well past both training
//! boundaries, and asserts the SE keeps growing as the probe moves outward
//! instead of saturating. It complements the point-prediction test by pinning
//! the *second* symptom of the same root cause
//! (`FittedModel::axis_clip_to_training_ranges` clamping linear-term columns).
//! Once the clamp skips linear-term columns, both pass.

use gam::test_support::cli_harness::{run_or_panic, write_predict_csv_rows};
use std::path::Path;
use std::process::Command;

const SLOPE: f64 = 1.25;
const INTERCEPT: f64 = 0.5;
const TRAIN_LO: f64 = -2.0;
const TRAIN_HI: f64 = 2.0;

fn write_training_csv(path: &Path) {
    // A tiny deterministic zig-zag residual keeps Var(β) non-degenerate without
    // an RNG dependency: the fit is still essentially the line y = 0.5 + 1.25x
    // but the posterior covariance of (β0, β1) is non-trivial.
    let mut writer = csv::Writer::from_path(path).expect("create training csv");
    writer.write_record(["x", "y"]).expect("write header");
    let n = 81usize;
    for i in 0..n {
        let x = TRAIN_LO + (TRAIN_HI - TRAIN_LO) * (i as f64) / ((n - 1) as f64);
        let wobble = if i % 2 == 0 { 0.02 } else { -0.02 };
        let y = INTERCEPT + SLOPE * x + wobble;
        writer
            .write_record([format!("{x:.12}"), format!("{y:.12}")])
            .expect("write training row");
    }
    writer.flush().expect("flush training csv");
}

fn read_column(path: &Path, name: &str) -> Vec<f64> {
    let mut reader = csv::Reader::from_path(path).expect("open predictions csv");
    let headers = reader.headers().expect("predict csv headers").clone();
    let idx = headers
        .iter()
        .position(|h| h == name)
        .unwrap_or_else(|| panic!("predict csv missing `{name}` column: {headers:?}"));
    reader
        .records()
        .map(|rec| {
            let rec = rec.expect("predict csv row");
            rec[idx]
                .parse::<f64>()
                .unwrap_or_else(|_| panic!("non-numeric `{name}`: {:?}", &rec[idx]))
        })
        .collect()
}

#[test]
fn linear_term_prediction_se_grows_outside_training_range() {
    let dir = tempfile::tempdir().expect("create tempdir");
    let train_path = dir.path().join("train.csv");
    let predict_path = dir.path().join("predict.csv");
    let model_path = dir.path().join("model.json");
    let out_path = dir.path().join("pred_out.csv");

    write_training_csv(&train_path);

    // Symmetric probes marching outward from the training centroid (0.0).
    // Index 0 is the boundary; each subsequent probe is strictly farther out.
    let probes_pos: [f64; 5] = [2.0, 3.0, 5.0, 10.0, 20.0];
    let probes_neg: [f64; 5] = [-2.0, -3.0, -5.0, -10.0, -20.0];
    let mut probes = Vec::new();
    probes.extend_from_slice(&probes_pos);
    probes.extend_from_slice(&probes_neg);
    write_predict_csv_rows(
        &predict_path,
        ["x", "y"],
        probes
            .iter()
            .map(|&x| [format!("{x:.12}"), "0.0".to_string()]),
    );

    let mut fit_cmd = Command::new(gam::gam_binary!());
    fit_cmd
        .arg("fit")
        .arg(&train_path)
        .arg("y ~ x")
        .args(["--family", "gaussian"])
        .arg("--out")
        .arg(&model_path);
    run_or_panic(fit_cmd, "gam fit y ~ x (gaussian)");

    let mut predict_cmd = Command::new(gam::gam_binary!());
    predict_cmd
        .arg("predict")
        .arg(&model_path)
        .arg(&predict_path)
        .arg("--uncertainty")
        .arg("--out")
        .arg(&out_path);
    run_or_panic(predict_cmd, "gam predict --uncertainty");

    let se = read_column(&out_path, "std_error");
    let mean = read_column(&out_path, "mean");
    assert_eq!(se.len(), probes.len(), "one SE per probe");

    // Point prediction must extrapolate (sanity: the slope is alive out there).
    for (&x, &m) in probes.iter().zip(mean.iter()) {
        if x.abs() <= TRAIN_HI {
            continue;
        }
        let expected = INTERCEPT + SLOPE * x;
        assert!(
            (m - expected).abs() < 0.1,
            "linear point prediction did not extrapolate: x={x} pred={m} expected≈{expected}"
        );
    }

    // The SE must strictly increase as each probe marches outward from the
    // boundary. A clamped covariate makes the SE identical for every
    // out-of-hull probe (it reuses the boundary x), so any non-increase is the
    // clamp bug resurfacing. Check both tails independently.
    for tail in [&probes_pos[..], &probes_neg[..]] {
        let idxs: Vec<usize> = tail
            .iter()
            .map(|&t| probes.iter().position(|&p| p == t).unwrap())
            .collect();
        for w in idxs.windows(2) {
            let (a, b) = (w[0], w[1]);
            assert!(
                se[b] > se[a] * (1.0 + 1e-6),
                "prediction SE did not grow moving outward: \
                 x={:+.1} SE={:.3e} -> x={:+.1} SE={:.3e} \
                 (covariate looks clamped to the training range before xᵀVar(β)x)",
                probes[a],
                se[a],
                probes[b],
                se[b],
            );
        }
    }

    // And the far-tail SE must dwarf the boundary SE — not merely tick up.
    let se_boundary = se[probes.iter().position(|&p| p == 2.0).unwrap()];
    let se_far = se[probes.iter().position(|&p| p == 20.0).unwrap()];
    assert!(
        se_far > 3.0 * se_boundary,
        "far-extrapolation SE ({se_far:.3e}) should be much larger than the \
         boundary SE ({se_boundary:.3e}); a near-equal value means the input \
         was clamped to the training hull"
    );
}
