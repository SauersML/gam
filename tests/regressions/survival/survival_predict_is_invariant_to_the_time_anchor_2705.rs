//! Regression for #2705: a saved survival model's predicted surface must not
//! depend on the baseline time ANCHOR.
//!
//! # The contract
//!
//! `center_survival_time_designs_at_anchor` subtracts the time-basis row at the
//! anchor from every entry and exit design row, so the anchor sets the origin
//! of the baseline reparameterization. Its own documentation calls that "an
//! exact affine reparameterization of the baseline offset", and the CLI flag
//! `--survival-time-anchor` is documented as a conditioning knob: it may change
//! the coefficients, and it must not change the model.
//!
//! The FIT honours that. Measured on this fixture's shape, the maximised
//! log-likelihood is identical to seven digits across anchors spanning five
//! decades. **Prediction did not.** `predict_survival` re-centered the rebuilt
//! time design only for an enumerated list of likelihood modes —
//! `LocationScale | MarginalSlope`, plus bare `Weibull` — and the list omitted
//! `Transformation`, the Royston-Parmar default. Centered coefficients
//! evaluated against an uncentered basis shift every reported `log Λ(t)` by the
//! CONSTANT `X(anchor)ᵀγ`: measured `+2.859821842` at every one of six
//! (time, covariate) pairs, a factor `e^2.86 = 17.5` on the reported cumulative
//! hazard.
//!
//! It is invisible on ordinary right-censored data because the default anchor
//! there is the earliest entry — the time origin — where `I_k(left) = 0`
//! exactly and the shift is zero. It appears the moment the anchor moves: on
//! any genuinely left-truncated dataset, which takes the robust interior anchor
//! by rule (#751/#1790/#2631), and on any explicit `--survival-time-anchor`.
//!
//! # What is asserted
//!
//! Two fits of the SAME data differing only in `--survival-time-anchor` must
//! produce the same predicted survival surface through BOTH public prediction
//! routes: the in-process library and `gam predict`. The non-vacuity check is
//! the other half of the same contract: the two fits' coefficients must DIFFER,
//! because that is what makes the agreement of the surfaces a statement about
//! the reparameterization rather than about two identical models.

use std::path::Path;
use std::process::Command;

use csv::StringRecord;
use gam::encode_recordswith_inferred_schema;
use gam::families::survival::predict::{
    SurvivalPredictRequest, SurvivalPredictionCovarianceMode, predict_survival,
};
use gam::inference::data::EncodedDataset;
use gam::inference::model::FittedModel;
use gam::test_support::cli_harness::run_or_panic;
use ndarray::{Array1, Array2};

const N: usize = 600;
const GRID: [f64; 5] = [0.25, 0.5, 1.0, 2.0, 4.0];

fn build_dataset() -> (Vec<f64>, Vec<f64>, Vec<f64>) {
    let mut state: u64 = 0x2705_2705_2705_2705;
    let mut next_u01 = || {
        state = state
            .wrapping_mul(6_364_136_223_846_793_005)
            .wrapping_add(1_442_695_040_888_963_407);
        (((state >> 11) as f64) / ((1u64 << 53) as f64)).clamp(1.0e-12, 1.0 - 1.0e-12)
    };
    let mut x = Vec::with_capacity(N);
    let mut exit = Vec::with_capacity(N);
    let mut event = Vec::with_capacity(N);
    for _ in 0..N {
        let xi = -1.0 + 2.0 * next_u01();
        let lam = 0.5 * (0.8 * xi).exp();
        let t_event = -next_u01().ln() / lam;
        let t_cens = -next_u01().ln() * 5.0;
        let observed = t_event.min(t_cens).max(0.05);
        x.push(xi);
        exit.push(observed);
        event.push(if t_event <= t_cens { 1.0 } else { 0.0 });
    }
    (x, exit, event)
}

fn write_training_csv(path: &Path, x: &[f64], exit: &[f64], event: &[f64]) {
    let mut writer = csv::Writer::from_path(path).expect("create training csv");
    writer
        .write_record(["entry", "exit", "event", "x"])
        .expect("write header");
    for i in 0..x.len() {
        writer
            .write_record([
                "0.0".to_string(),
                format!("{:.12}", exit[i]),
                format!("{}", event[i] as i64),
                format!("{:.12}", x[i]),
            ])
            .expect("write training row");
    }
    writer.flush().expect("flush training csv");
}

fn predict_rows() -> EncodedDataset {
    let headers = vec![
        "entry".to_string(),
        "exit".to_string(),
        "event".to_string(),
        "x".to_string(),
    ];
    let rows: Vec<StringRecord> = [-0.7_f64, 0.0, 0.7]
        .iter()
        .map(|x| {
            StringRecord::from(vec![
                "0.0".to_string(),
                "9.0".to_string(),
                "1".to_string(),
                format!("{x:.12}"),
            ])
        })
        .collect();
    encode_recordswith_inferred_schema(headers, rows).expect("encode predict rows")
}

fn write_cli_predict_rows(path: &Path) {
    let mut writer = csv::Writer::from_path(path).expect("create CLI predict csv");
    writer
        .write_record(["entry", "exit", "event", "x"])
        .expect("write CLI predict header");
    for x in [-0.7_f64, 0.0, 0.7] {
        for time in GRID {
            writer
                .write_record([
                    "0.0".to_string(),
                    format!("{time:.12}"),
                    "1".to_string(),
                    format!("{x:.12}"),
                ])
                .expect("write CLI predict row");
        }
    }
    writer.flush().expect("flush CLI predict csv");
}

fn read_cli_survival(path: &Path) -> Array2<f64> {
    let mut reader = csv::Reader::from_path(path).expect("open CLI prediction csv");
    let headers = reader.headers().expect("CLI prediction headers").clone();
    let survival_column = headers
        .iter()
        .position(|name| name == "survival_prob")
        .unwrap_or_else(|| panic!("CLI prediction is missing survival_prob: {headers:?}"));
    let values = reader
        .records()
        .map(|record| {
            let record = record.expect("CLI prediction row");
            record[survival_column]
                .parse::<f64>()
                .expect("numeric CLI survival probability")
        })
        .collect::<Vec<_>>();
    Array2::from_shape_vec((3, GRID.len()), values)
        .expect("CLI surface has one row per covariate/time pair")
}

/// `(library surface, CLI surface, fitted coefficient vector)` for one anchor.
fn fit_at_anchor(
    train_path: &Path,
    dir: &Path,
    anchor: f64,
    tag: &str,
) -> (Array2<f64>, Array2<f64>, Vec<f64>) {
    let model_path = dir.join(format!("model_{tag}.json"));
    let mut fit_cmd = Command::new(gam::gam_binary!());
    fit_cmd
        .arg("fit")
        .arg(train_path)
        .arg("Surv(entry, exit, event) ~ s(x)")
        .args(["--survival-time-anchor", &format!("{anchor}")])
        .arg("--out")
        .arg(&model_path);
    run_or_panic(fit_cmd, "gam fit Surv(entry, exit, event) ~ s(x)");
    assert!(model_path.is_file(), "gam fit did not write {model_path:?}");

    let model = FittedModel::load_from_path(&model_path).expect("load saved survival model");
    // The unified coefficient vector, which the transformation family carries
    // as one block (`survival_beta_time` is populated only by the families that
    // split their coefficients into named channels).
    let beta = model
        .payload()
        .unified
        .as_ref()
        .or(model.payload().fit_result.as_ref())
        .map(|fit| fit.beta.to_vec())
        .expect("saved model must carry its fitted coefficients");
    let saved_anchor = model
        .payload()
        .survival_time_anchor
        .expect("saved model must carry its time anchor");
    assert!(
        (saved_anchor - anchor).abs() <= 1.0e-9,
        "the fit did not honour --survival-time-anchor: asked {anchor}, saved {saved_anchor}"
    );

    let dataset = predict_rows();
    let col_map = dataset.column_map();
    let payload = model.payload();
    let training_headers = payload.training_headers.as_ref();
    let rows = dataset.values.nrows();
    let primary_offset = Array1::<f64>::zeros(rows);
    let noise_offset = Array1::<f64>::zeros(rows);
    let grid = GRID.to_vec();
    let request = SurvivalPredictRequest {
        model: &model,
        data: dataset.values.view(),
        col_map: &col_map,
        training_headers,
        primary_offset: &primary_offset,
        noise_offset: &noise_offset,
        time_grid: Some(&grid),
        with_uncertainty: false,
        estimand: gam::families::survival::predict::SurvivalPredictEstimand::Plugin,
    };
    let result = predict_survival(request, SurvivalPredictionCovarianceMode::Conditional)
        .expect("library survival predict");

    // Exercise the independent CLI replay. Before the second #2705 repair the
    // library path centered every non-empty time design, while this path used a
    // likelihood-mode allow-list that omitted Transformation. The two surfaces
    // then differed by the constant `X(anchor)^T gamma` on the interior arm.
    let cli_data_path = dir.join(format!("predict_{tag}.csv"));
    let cli_output_path = dir.join(format!("prediction_{tag}.csv"));
    write_cli_predict_rows(&cli_data_path);
    let mut predict_cmd = Command::new(gam::gam_binary!());
    predict_cmd
        .arg("predict")
        .arg(&model_path)
        .arg(&cli_data_path)
        .args(["--mode", "map"])
        .arg("--out")
        .arg(&cli_output_path);
    run_or_panic(predict_cmd, "gam predict Royston-Parmar anchor replay");
    let cli_surface = read_cli_survival(&cli_output_path);

    (result.survival, cli_surface, beta)
}

fn worst_surface_gap(left: &Array2<f64>, right: &Array2<f64>) -> (f64, (usize, usize)) {
    assert_eq!(left.dim(), right.dim());
    let mut worst = 0.0_f64;
    let mut worst_at = (0usize, 0usize);
    for row in 0..left.nrows() {
        for column in 0..left.ncols() {
            let gap = (left[[row, column]] - right[[row, column]]).abs();
            if gap > worst {
                worst = gap;
                worst_at = (row, column);
            }
        }
    }
    (worst, worst_at)
}

#[test]
fn transformation_survival_prediction_does_not_depend_on_the_time_anchor_2705() {
    let (x, exit, event) = build_dataset();
    let dir = tempfile::tempdir().expect("create tempdir");
    let train_path = dir.path().join("train.csv");
    write_training_csv(&train_path, &x, &exit, &event);

    // `1e-7` is below the first knot, so its anchor row is the anchored zero
    // row and centering is a no-op — the regime every right-censored fit is in
    // by default, and the one the omission was invisible in. `1.0` is interior.
    let (library_origin, cli_origin, beta_origin) =
        fit_at_anchor(train_path.as_path(), dir.path(), 1.0e-7, "origin");
    let (library_interior, cli_interior, beta_interior) =
        fit_at_anchor(train_path.as_path(), dir.path(), 1.0, "interior");

    assert_eq!(
        beta_origin.len(),
        beta_interior.len(),
        "the two fits must share a coefficient layout"
    );
    let coefficient_gap = beta_origin
        .iter()
        .zip(beta_interior.iter())
        .fold(0.0_f64, |worst, (a, b)| worst.max((a - b).abs()));
    assert!(
        coefficient_gap > 1.0e-6,
        "non-vacuity: the two anchors must produce DIFFERENT coefficients, otherwise the \
         surfaces agreeing says nothing about the reparameterization. max|Δβ| = \
         {coefficient_gap:.3e}"
    );

    let (worst, worst_at) = worst_surface_gap(&library_origin, &library_interior);
    assert!(
        worst <= 1.0e-4,
        "the LIBRARY survival surface moved with the time ANCHOR, which is a \
         reparameterization and not a model: max|ΔS| = {worst:.6e} at row {} time {} \
         (S = {:.9} vs {:.9}). Centered coefficients evaluated against an uncentered basis \
         shift every log Λ by the constant X(anchor)ᵀγ (gam#2705).",
        worst_at.0,
        GRID[worst_at.1],
        library_origin[[worst_at.0, worst_at.1]],
        library_interior[[worst_at.0, worst_at.1]]
    );

    let (worst, worst_at) = worst_surface_gap(&cli_origin, &cli_interior);
    assert!(
        worst <= 1.0e-4,
        "the CLI survival surface moved with the time ANCHOR: max|ΔS| = {worst:.6e} at \
         row {} time {} (S = {:.9} vs {:.9}). `gam predict` must replay the same \
         anchor-centered time design the fit used (gam#2705).",
        worst_at.0,
        GRID[worst_at.1],
        cli_origin[[worst_at.0, worst_at.1]],
        cli_interior[[worst_at.0, worst_at.1]]
    );

    for (label, library, cli) in [
        ("origin", &library_origin, &cli_origin),
        ("interior", &library_interior, &cli_interior),
    ] {
        let (worst, worst_at) = worst_surface_gap(library, cli);
        assert!(
            worst <= 1.0e-6,
            "{label}-anchor CLI/library survival parity failed: max|ΔS| = {worst:.6e} at \
             row {} time {} (library = {:.9}, CLI = {:.9})",
            worst_at.0,
            GRID[worst_at.1],
            library[[worst_at.0, worst_at.1]],
            cli[[worst_at.0, worst_at.1]]
        );
    }
}
