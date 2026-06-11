//! Regression guard for the tensor-product smooth PREDICT path.
//!
//! A `te(x, z)` tensor-product smooth carries non-trivial fit-time structure
//! that MUST be replayed faithfully at predict time on a fresh `(x, z)` grid:
//!
//!   * each margin is realized as a 1-D B-spline whose augmented knot vector,
//!     degree, and (possible) periodicity are frozen into the model
//!     (`BasisMetadata::TensorBSpline { knots, degrees, periods, .. }`) and
//!     restored as `BSplineKnotSpec::Provided(knots[i])` at predict so the raw
//!     marginal column counts match exactly;
//!   * the sum-to-zero identifiability reparametrization `Z` (for `te`, a single
//!     constraint over the *joint* tensor design) is data-dependent — it is
//!     computed once at fit from the FIT design's column sums and frozen as a
//!     `FrozenTransform`. Predict must re-apply that exact `Z` to the freshly
//!     rebuilt raw tensor `B₀ ⊗ B₁` (`dense.dot(z)`), never re-center on the
//!     prediction grid's own column means.
//!
//! Any margin-misalignment, column-drop, knot/degree desync, or
//! re-centering-at-predict bug in that rebuild corrupts the recovered surface
//! while leaving the fit itself intact — so it only surfaces at PREDICT on data
//! not seen during fitting. (Precedent: the tensor *periodic-margin* off-by-one
//! in `bug_hunt_tensor_periodic_margin_predict_offbyone.rs`.)
//!
//! This test drives the real predict pipeline end to end via the `gam` CLI:
//! fit `y ~ te(x, z)` (Gaussian) on a known NON-separable surface, save the
//! model, reload it, and predict on a FRESH interior grid of `(x, z)` points
//! that were not in training. It asserts the predictions track the TRUE
//! generating surface within a tight RMSE bar (gam-internal truth recovery —
//! no external tool needed). A misaligned margin or a re-centered transform
//! would distort the surface and blow past the bar.
//!
//! The truth surface `f(x, z) = sin(pi*x)*z + 0.5*cos(pi*z)*x` is genuinely
//! bivariate (the `x`-shape depends on `z` and vice versa), so a tensor product
//! is required to represent it — an additive `s(x)+s(z)` or a margin-swapped
//! rebuild cannot. The prediction grid lies strictly INSIDE the training hull
//! (`[-0.8, 0.8]²` ⊂ training `[-1, 1]²`) so the unrelated predict-time input
//! clamp (`FittedModel::axis_clip_to_training_ranges`) never fires.

use gam::test_support::cli_harness::{fit_then_predict_gaussian, write_predict_csv_rows};
use std::path::Path;

const NOISE_SD: f64 = 0.05;
// Training covariates span [-1, 1]²; the prediction grid stays within
// [-0.8, 0.8]² so the predict-time clamp is never engaged.
const TRAIN_HALF_RANGE: f64 = 1.0;
const GRID_HALF_RANGE: f64 = 0.8;

/// Genuinely bivariate truth surface (NOT additively separable): the per-`x`
/// shape is modulated by `z` and vice versa, so the tensor interaction is
/// load-bearing.
fn truth_surface(x: f64, z: f64) -> f64 {
    (std::f64::consts::PI * x).sin() * z + 0.5 * (std::f64::consts::PI * z).cos() * x
}

/// Deterministic, dependency-free pseudo-random stream in `[0, 1)` (SplitMix64),
/// matching the sibling predict-on-grid regression tests.
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
        (self.next_u64() >> 11) as f64 / (1u64 << 53) as f64
    }

    /// Standard normal via Box-Muller (one of the pair).
    fn next_normal(&mut self) -> f64 {
        let u1 = self.next_unit().max(f64::MIN_POSITIVE);
        let u2 = self.next_unit();
        (-2.0 * u1.ln()).sqrt() * (std::f64::consts::TAU * u2).cos()
    }
}

/// Write the training CSV: `n` rows of `(x, z, y)` with
/// `y = truth_surface(x, z) + N(0, NOISE_SD²)`, both covariates uniform on
/// `[-TRAIN_HALF_RANGE, TRAIN_HALF_RANGE]`.
fn write_training_csv(path: &Path, n: usize) {
    let mut rng = SplitMix64::new(0x5EED_7E50_4ABC_1234);
    let mut writer = csv::Writer::from_path(path).expect("create training csv");
    writer.write_record(["x", "z", "y"]).expect("write header");
    for _ in 0..n {
        let x = (2.0 * rng.next_unit() - 1.0) * TRAIN_HALF_RANGE;
        let z = (2.0 * rng.next_unit() - 1.0) * TRAIN_HALF_RANGE;
        let y = truth_surface(x, z) + NOISE_SD * rng.next_normal();
        writer
            .write_record([format!("{x:.12}"), format!("{z:.12}"), format!("{y:.12}")])
            .expect("write training row");
    }
    writer.flush().expect("flush training csv");
}

/// A regular `g × g` grid of `(x, z)` points over `[-GRID_HALF_RANGE, +]²`,
/// strictly inside the training hull.
fn fresh_grid(g: usize) -> Vec<(f64, f64)> {
    let mut grid = Vec::with_capacity(g * g);
    for i in 0..g {
        // Offset off the exact endpoints/integer ratios so the grid points are
        // genuinely fresh (not coincidentally training rows) and never land on
        // a knot.
        let fx = (i as f64 + 0.5) / g as f64; // in (0, 1)
        let x = (2.0 * fx - 1.0) * GRID_HALF_RANGE;
        for j in 0..g {
            let fz = (j as f64 + 0.5) / g as f64;
            let z = (2.0 * fz - 1.0) * GRID_HALF_RANGE;
            grid.push((x, z));
        }
    }
    grid
}

#[test]
fn predict_recovers_known_tensor_te_surface_on_a_fresh_grid() {
    let dir = tempfile::tempdir().expect("create tempdir");
    let train_path = dir.path().join("train.csv");
    let predict_path = dir.path().join("predict.csv");
    let model_path = dir.path().join("model.json");
    let out_path = dir.path().join("pred_out.csv");

    write_training_csv(&train_path, 900);

    let grid = fresh_grid(11);
    // `y` is a placeholder (predict ignores it); the schema just needs the column.
    write_predict_csv_rows(
        &predict_path,
        ["x", "z", "y"],
        grid.iter()
            .map(|&(x, z)| [format!("{x:.12}"), format!("{z:.12}"), "0.0".to_string()]),
    );

    let preds = fit_then_predict_gaussian(
        &train_path,
        "y ~ te(x, z, k=8)",
        &model_path,
        &predict_path,
        &out_path,
    );
    assert_eq!(
        preds.len(),
        grid.len(),
        "expected one prediction per grid row"
    );
    assert!(
        preds.iter().all(|p| p.is_finite()),
        "predict produced non-finite values on the fresh tensor grid: {preds:?}"
    );

    // Truth recovery on the fresh grid: a faithful margin replay + frozen
    // sum-to-zero transform reproduces the bivariate surface up to fit noise;
    // a margin-misalignment / column-drop / re-centering bug distorts it.
    let mse = grid
        .iter()
        .zip(preds.iter())
        .map(|(&(x, z), &p)| {
            let err = p - truth_surface(x, z);
            err * err
        })
        .sum::<f64>()
        / grid.len() as f64;
    let rmse = mse.sqrt();
    assert!(
        rmse <= 0.06,
        "te(x,z) predict-on-grid failed to recover the known bivariate surface: \
         rmse_vs_truth={rmse:.5} (noise_sd={NOISE_SD}). A large RMSE indicates the \
         tensor predict rebuild misaligned a margin, dropped a column, desynced a \
         knot/degree, or re-centered the sum-to-zero transform on the prediction \
         grid instead of replaying the frozen fit-time transform."
    );

    // Range sanity: the recovered surface must actually vary across the grid
    // (a collapsed/constant prediction would also nominally fail RMSE for some
    // truths, but assert it explicitly so a degenerate flat surface is named).
    let pmin = preds.iter().cloned().fold(f64::INFINITY, f64::min);
    let pmax = preds.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
    assert!(
        pmax - pmin > 0.5,
        "te(x,z) predictions are nearly constant across the grid \
         (range={:.4}); the tensor surface collapsed at predict time",
        pmax - pmin
    );
}
