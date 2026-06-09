//! Regression guard for the first-factor-only linear-interaction design bug
//! fixed in commit f7bc5eb7.
//!
//! `build_term_collection_design_inner` (`src/terms/smooth.rs`) used to
//! materialize every linear term from the SINGULAR `LinearTermSpec::feature_col`
//! (the first factor only), so a Wilkinson-Rogers `:` interaction such as
//! `x0:x1` was silently built as just `x0` — the product with the second factor
//! was dropped. That `_inner` builder backs `build_term_collection_design`,
//! which is the design path the real `FittedModel` predict pipeline runs on new
//! data (`build_predict_input_for_model_inner` → `build_term_collection_design`,
//! reached by both the `gam predict` CLI and `gamfit.predict`). The standard
//! GLM/GAM *fitting* path instead uses the incremental realizer
//! (`build_term_collection_fixed_blocks`), which built the product correctly —
//! so the two builders disagreed and the bug surfaced only at PREDICT time,
//! where a model with a genuine interaction coefficient produced predictions as
//! if the interaction column were a bare `x0` main effect.
//!
//! This test fits a Gaussian model with a known nonzero `x0:x1` interaction and
//! then asks the real predict pipeline (the `gam` CLI) for predictions on a
//! FRESH grid that was not in training. It asserts the interaction is recovered
//! from the PREDICTIONS — via the four-corner Hadamard contrast
//! `0.25·(p(-1,-1) − p(-1,1) − p(1,-1) + p(1,1))`, which equals the interaction
//! slope for an additive `b0 + b1·x0 + b2·x1 + g·x0·x1` mean — and that the grid
//! RMSE versus the known DGP is small. Under the old first-factor-only bug the
//! interaction column at predict time is `x0` (whose four-corner Hadamard
//! contrast is identically 0), so this contrast collapses to ≈0 and the test
//! fails. With the product correctly materialized it recovers `g`.
//!
//! The prediction grid lives strictly INSIDE the training hull (`|x| ≤ 1` while
//! training spans `[-2, 2]`), so the unrelated predict-time input clamp
//! (`FittedModel::axis_clip_to_training_ranges`, see
//! `bug_hunt_predict_linear_term_clamped_to_training_range.rs`) never fires and
//! cannot confound this guard.

use std::path::{Path, PathBuf};
use std::process::Command;

fn gam_binary() -> PathBuf {
    option_env!("CARGO_BIN_EXE_gam")
        .map(PathBuf::from)
        .unwrap_or_else(|| PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("target/debug/gam"))
}

// Known generating coefficients for the additive-plus-interaction Gaussian mean
// `mu = B0 + B1*x0 + B2*x1 + G*x0*x1`.
const B0: f64 = 0.3;
const B1: f64 = 0.8;
const B2: f64 = -0.5;
const G: f64 = 1.5;
const NOISE_SD: f64 = 0.2;

// Training covariates span [-2, 2]; the prediction grid stays within [-1, 1] so
// the predict-time clamp is never engaged.
const TRAIN_HALF_RANGE: f64 = 2.0;

fn truth_mean(x0: f64, x1: f64) -> f64 {
    B0 + B1 * x0 + B2 * x1 + G * x0 * x1
}

/// Deterministic, dependency-free pseudo-random stream in `[0, 1)`. A simple
/// SplitMix64 keeps the dataset reproducible across runs without pulling in an
/// RNG crate, and the two covariates are drawn from independent sub-streams so
/// `x0`, `x1` (and hence `x0·x1`) are genuinely identifiable.
struct SplitMix64 {
    state: u64,
}

impl SplitMix64 {
    fn new(seed: u64) -> Self {
        Self { state: seed }
    }

    fn next_u64(&mut self) -> u64 {
        self.state = self.state.wrapping_add(0x9E37_79B9_7F4A_7C15);
        let mut z = self.state;
        z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
        z ^ (z >> 31)
    }

    /// Uniform in `[0, 1)`.
    fn next_unit(&mut self) -> f64 {
        // 53-bit mantissa fraction.
        (self.next_u64() >> 11) as f64 / (1u64 << 53) as f64
    }

    /// Standard normal via Box-Muller (one of the pair).
    fn next_normal(&mut self) -> f64 {
        let u1 = self.next_unit().max(f64::MIN_POSITIVE);
        let u2 = self.next_unit();
        (-2.0 * u1.ln()).sqrt() * (std::f64::consts::TAU * u2).cos()
    }
}

/// Write the training CSV: `n` rows of `(x0, x1, y)` with
/// `y = truth_mean(x0, x1) + N(0, NOISE_SD²)`.
fn write_training_csv(path: &Path, n: usize) {
    let mut rng = SplitMix64::new(0xC0FF_EE12_3456_789A);
    let mut writer = csv::Writer::from_path(path).expect("create training csv");
    writer
        .write_record(["x0", "x1", "y"])
        .expect("write header");
    for _ in 0..n {
        let x0 = (2.0 * rng.next_unit() - 1.0) * TRAIN_HALF_RANGE;
        let x1 = (2.0 * rng.next_unit() - 1.0) * TRAIN_HALF_RANGE;
        let y = truth_mean(x0, x1) + NOISE_SD * rng.next_normal();
        writer
            .write_record([format!("{x0:.12}"), format!("{x1:.12}"), format!("{y:.12}")])
            .expect("write training row");
    }
    writer.flush().expect("flush training csv");
}

/// Write the prediction CSV at fresh `(x0, x1)` points (not in training). `y` is
/// a placeholder the predict path ignores; the schema just needs the column.
fn write_predict_csv(path: &Path, grid: &[(f64, f64)]) {
    let mut writer = csv::Writer::from_path(path).expect("create predict csv");
    writer
        .write_record(["x0", "x1", "y"])
        .expect("write header");
    for &(x0, x1) in grid {
        writer
            .write_record([format!("{x0:.12}"), format!("{x1:.12}"), "0.0".to_string()])
            .expect("write predict row");
    }
    writer.flush().expect("flush predict csv");
}

fn run_or_panic(mut command: Command, label: &str) {
    let output = command
        .output()
        .unwrap_or_else(|err| panic!("failed to spawn `{label}`: {err}"));
    assert!(
        output.status.success(),
        "`{label}` failed with status {}\n--- stdout ---\n{}\n--- stderr ---\n{}",
        output.status,
        String::from_utf8_lossy(&output.stdout),
        String::from_utf8_lossy(&output.stderr),
    );
}

/// Read the `mean` (or `linear_predictor`) column from a `gam predict --out` CSV.
fn read_predictions(path: &Path) -> Vec<f64> {
    let mut reader = csv::Reader::from_path(path).expect("open predictions csv");
    let headers = reader.headers().expect("predict csv headers").clone();
    let mean_idx = headers
        .iter()
        .position(|h| h == "mean")
        .or_else(|| headers.iter().position(|h| h == "linear_predictor"))
        .unwrap_or_else(|| {
            panic!("predict csv has neither `mean` nor `linear_predictor` column: {headers:?}")
        });
    reader
        .records()
        .map(|rec| {
            let rec = rec.expect("predict csv row");
            rec[mean_idx]
                .parse::<f64>()
                .unwrap_or_else(|_| panic!("non-numeric prediction: {:?}", &rec[mean_idx]))
        })
        .collect()
}

#[test]
fn predict_recovers_known_linear_interaction_on_a_fresh_grid() {
    let dir = tempfile::tempdir().expect("create tempdir");
    let train_path = dir.path().join("train.csv");
    let predict_path = dir.path().join("predict.csv");
    let model_path = dir.path().join("model.json");
    let out_path = dir.path().join("pred_out.csv");

    write_training_csv(&train_path, 400);

    // Fresh evaluation grid: the four corners drive the Hadamard interaction
    // contrast; the interior points round out a grid RMSE check. All within the
    // training hull (|x| ≤ 1 ⊂ [-2, 2]) so the predict-time input clamp cannot
    // fire and confound the result.
    let corners: [(f64, f64); 4] = [(-1.0, -1.0), (-1.0, 1.0), (1.0, -1.0), (1.0, 1.0)];
    let interior: [(f64, f64); 5] = [
        (0.0, 0.0),
        (0.5, -0.5),
        (-0.5, 0.5),
        (0.75, 0.25),
        (-0.25, -0.75),
    ];
    let mut grid: Vec<(f64, f64)> = Vec::new();
    grid.extend_from_slice(&corners);
    grid.extend_from_slice(&interior);
    write_predict_csv(&predict_path, &grid);

    let mut fit_cmd = Command::new(gam_binary());
    fit_cmd
        .arg("fit")
        .arg(&train_path)
        .arg("y ~ x0 + x1 + x0:x1")
        .args(["--family", "gaussian"])
        .arg("--out")
        .arg(&model_path);
    run_or_panic(fit_cmd, "gam fit y ~ x0 + x1 + x0:x1 (gaussian)");
    assert!(model_path.is_file(), "gam fit did not write {model_path:?}");

    let mut predict_cmd = Command::new(gam_binary());
    predict_cmd
        .arg("predict")
        .arg(&model_path)
        .arg(&predict_path)
        .arg("--out")
        .arg(&out_path);
    run_or_panic(predict_cmd, "gam predict (linear interaction grid)");

    let preds = read_predictions(&out_path);
    assert_eq!(
        preds.len(),
        grid.len(),
        "expected one prediction per grid row"
    );

    // The four corners come first; recover the interaction slope from the
    // PREDICTIONS via the Hadamard contrast. For an additive
    // `b0 + b1·x0 + b2·x1 + g·x0·x1` mean evaluated at (∓1, ∓1) this contrast
    // isolates `g` exactly (the intercept and both main effects cancel). Under
    // the old first-factor-only bug the interaction column predicts as `x0`,
    // whose corner Hadamard contrast is 0, so `predicted_g ≈ 0`.
    let p_mm = preds[0]; // (-1, -1)
    let p_mp = preds[1]; // (-1, +1)
    let p_pm = preds[2]; // (+1, -1)
    let p_pp = preds[3]; // (+1, +1)
    let predicted_g = 0.25 * (p_mm - p_mp - p_pm + p_pp);

    assert!(
        (predicted_g - G).abs() <= 0.1,
        "interaction not recovered from predictions: predicted_g={predicted_g:.4} \
         vs truth g={G:.4} (corner means: (-1,-1)={p_mm:.4} (-1,1)={p_mp:.4} \
         (1,-1)={p_pm:.4} (1,1)={p_pp:.4}). A value near 0 means the predict-time \
         design built `x0:x1` as the first factor `x0` only \
         (build_term_collection_design_inner first-factor-only bug)."
    );

    // Overall predicted-vs-truth accuracy on the fresh grid: with a real
    // interaction column the additive surface is recovered cleanly; a dropped
    // product factor distorts the whole surface, not just the corner contrast.
    let mse = grid
        .iter()
        .zip(preds.iter())
        .map(|(&(x0, x1), &p)| {
            let err = p - truth_mean(x0, x1);
            err * err
        })
        .sum::<f64>()
        / grid.len() as f64;
    let rmse = mse.sqrt();
    assert!(
        rmse <= 0.1,
        "grid predictions are far from the known DGP: rmse={rmse:.4} \
         (truth mean b0={B0} b1={B1} b2={B2} g={G}); a large RMSE indicates the \
         interaction product term was not materialized at predict time."
    );
}
