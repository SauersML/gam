//! Bug hunt (#1214 / #1215): a 1-D penalized smooth must be invariant to an
//! affine rescaling of its covariate, `x → a·x` — the fitted function (matched
//! at corresponding covariate points) is unchanged and the smoothing parameter
//! co-transforms equivariantly. Two bases were not:
//!
//! * `s(x, bs="cr")` — the single-penalty (`double_penalty=false`) cubic
//!   regression spline routes to the exact O(n) state-space spline scan. The
//!   scan's `log λ` search bracket was a fixed *absolute* interval, so it did
//!   not track the covariate's own length scale. The order-`m` integrated-Wiener
//!   process noise is `Q(δ) ∝ q·δ^{2m−1}` in the abscissa gap `δ`, so under
//!   `x → a·x` (all gaps `δ → a·δ`) the posterior `f(x)` is *exactly* invariant
//!   iff `q → q / a^{2m−1}`, i.e. `log λ → log λ + (2m−1)·log a`. With a fixed
//!   bracket the equivariant optimum railed out of range at small/large
//!   covariate scale, drifting `λ̂` ~4 orders of magnitude and under-smoothing
//!   the curve (#1214). Fix: anchor the `log λ` bracket to the abscissa span so
//!   the search runs in scale-free units (`src/solver/spline_scan.rs`).
//!
//! * `s(x, bs="tp")` — the thin-plate kernel's length-scale `ℓ` was seeded from
//!   the *raw* covariate magnitude and the `ψ = log κ = −log ℓ` REML optimizer
//!   ran in raw covariate units, landing in a scale-dependent basin (a bimodal
//!   step across `|a| ⋛ 1`, ~2e-2 drift, #1215). 1-D spatial inputs were never
//!   standardized (`compute_spatial_input_scales` bailed at `d ≤ 1`), so the raw
//!   magnitude leaked into the optimizer. Fix: standardize the single covariate
//!   axis to unit spread the same way `d > 1` axes already are, so the kernel
//!   and its `ψ`-optimizer operate in scale-free coordinates
//!   (`src/terms/smooth/input_standardization.rs`); the frozen scale replays at
//!   predict.
//!
//! Both are siblings of the response-axis REML non-equivariance family (#1000 /
//! #1127): a `λ̂` drifting under a transform the underlying problem is invariant
//! to. The principled fix lives in the seeding / normalization, not a tolerance.

use csv::StringRecord;
use gam::matrix::LinearOperator;
use gam::smooth::build_term_collection_design;
use gam::{
    FitConfig, FitResult, encode_recordswith_inferred_schema, fit_from_formula, init_parallelism,
};
use ndarray::Array2;
use rand::SeedableRng;
use rand::rngs::StdRng;
use rand_distr::{Distribution, Normal, Uniform};

/// Fixed synthetic Gaussian dataset `y = sin(2π x) + N(0, 0.2)` on `x ∈ [0, 1]`,
/// with the covariate column linearly rescaled to `a·x`. The response and the
/// noise realization are identical for every `a` (same seed), so the only
/// difference between two datasets is the multiplicative covariate scale.
fn dataset_with_covariate_scale(a: f64, n: usize) -> gam::data::EncodedDataset {
    let mut rng = StdRng::seed_from_u64(20240616);
    let u = Uniform::new(0.0_f64, 1.0).expect("uniform");
    let noise = Normal::new(0.0, 0.2).expect("normal");
    let rows: Vec<StringRecord> = (0..n)
        .map(|_| {
            let x = u.sample(&mut rng);
            let y = (2.0 * std::f64::consts::PI * x).sin() + noise.sample(&mut rng);
            StringRecord::from(vec![(a * x).to_string(), y.to_string()])
        })
        .collect();
    let headers = ["x", "y"].into_iter().map(String::from).collect();
    encode_recordswith_inferred_schema(headers, rows).expect("encode")
}

/// Fit `formula` (Gaussian) and predict the fitted function at a fixed grid of
/// *relative* covariate points scaled into the same `a·x` coordinate, so the
/// predictions of two scale-`a` fits are directly comparable as functions.
fn fit_grid_predictions(formula: &str, a: f64, n: usize) -> Vec<f64> {
    let data = dataset_with_covariate_scale(a, n);
    let cfg = FitConfig {
        family: Some("gaussian".to_string()),
        ..FitConfig::default()
    };
    let result = fit_from_formula(formula, &data, &cfg).expect("fit ok");
    let probes: Vec<f64> = (0..15).map(|i| 0.05 + 0.90 * (i as f64) / 14.0).collect();
    match result {
        FitResult::Standard(fit) => {
            let mut grid = Array2::<f64>::zeros((probes.len(), 2));
            for (i, &v) in probes.iter().enumerate() {
                grid[[i, 0]] = a * v;
                grid[[i, 1]] = 0.0;
            }
            let design =
                build_term_collection_design(grid.view(), &fit.resolvedspec).expect("design");
            design.design.apply(&fit.fit.beta).to_vec()
        }
        FitResult::SplineScan(scan) => probes
            .iter()
            .map(|&v| scan.predict(a * v).expect("predict").0)
            .collect(),
        _ => panic!("unexpected fit result variant for {formula}"),
    }
}

/// Selected `log λ` for the exact spline-scan (`bs="cr"`) path.
fn fit_scan_log_lambda(a: f64, n: usize) -> f64 {
    let data = dataset_with_covariate_scale(a, n);
    let cfg = FitConfig {
        family: Some("gaussian".to_string()),
        ..FitConfig::default()
    };
    match fit_from_formula("y ~ s(x, bs=\"cr\")", &data, &cfg).expect("fit ok") {
        FitResult::SplineScan(scan) => scan.log_lambda,
        _ => panic!("expected cr default to route to the exact spline scan"),
    }
}

/// Same fixed synthetic dataset as [`dataset_with_covariate_scale`], but the
/// covariate is *translated* by a constant `b` (`x → x + b`) instead of scaled.
/// The response and noise realization are identical for every `b`, so two
/// datasets differ only by the additive covariate offset.
fn dataset_with_covariate_offset(b: f64, n: usize) -> gam::data::EncodedDataset {
    let mut rng = StdRng::seed_from_u64(20240616);
    let u = Uniform::new(0.0_f64, 1.0).expect("uniform");
    let noise = Normal::new(0.0, 0.2).expect("normal");
    let rows: Vec<StringRecord> = (0..n)
        .map(|_| {
            let x = u.sample(&mut rng);
            let y = (2.0 * std::f64::consts::PI * x).sin() + noise.sample(&mut rng);
            StringRecord::from(vec![(x + b).to_string(), y.to_string()])
        })
        .collect();
    let headers = ["x", "y"].into_iter().map(String::from).collect();
    encode_recordswith_inferred_schema(headers, rows).expect("encode")
}

/// Fit `formula` (Gaussian) on the `x → x + b` translated covariate and predict
/// at a fixed grid of *relative* points mapped into the same offset coordinate,
/// so two offset-`b` fits are directly comparable as functions.
fn fit_grid_predictions_offset(formula: &str, b: f64, n: usize) -> Vec<f64> {
    let data = dataset_with_covariate_offset(b, n);
    let cfg = FitConfig {
        family: Some("gaussian".to_string()),
        ..FitConfig::default()
    };
    let result = fit_from_formula(formula, &data, &cfg).expect("fit ok");
    let probes: Vec<f64> = (0..15).map(|i| 0.05 + 0.90 * (i as f64) / 14.0).collect();
    match result {
        FitResult::Standard(fit) => {
            let mut grid = Array2::<f64>::zeros((probes.len(), 2));
            for (i, &v) in probes.iter().enumerate() {
                grid[[i, 0]] = v + b;
                grid[[i, 1]] = 0.0;
            }
            let design =
                build_term_collection_design(grid.view(), &fit.resolvedspec).expect("design");
            design.design.apply(&fit.fit.beta).to_vec()
        }
        FitResult::SplineScan(scan) => probes
            .iter()
            .map(|&v| scan.predict(v + b).expect("predict").0)
            .collect(),
        _ => panic!("unexpected fit result variant for {formula}"),
    }
}

fn centered(v: &[f64]) -> Vec<f64> {
    let mean = v.iter().sum::<f64>() / v.len() as f64;
    v.iter().map(|x| x - mean).collect()
}

fn max_shape_drift(base: &[f64], other: &[f64]) -> f64 {
    centered(base)
        .iter()
        .zip(&centered(other))
        .map(|(b, s)| (b - s).abs())
        .fold(0.0, f64::max)
}

#[test]
fn cr_smooth_is_invariant_to_covariate_rescaling() {
    init_parallelism();
    let formula = "y ~ s(x, bs=\"cr\")";
    let n = 400;
    let base = fit_grid_predictions(formula, 1.0, n);

    // The fitted function (matched at corresponding covariate points) must be
    // identical across covariate scales spanning many orders of magnitude.
    for &a in &[1.0e3, 1.0e-3, 1.0e6, 1.0e-6] {
        let drift = max_shape_drift(&base, &fit_grid_predictions(formula, a, n));
        assert!(
            drift < 1.0e-6,
            "s(x, bs=\"cr\") fitted function changed under covariate rescale a={a:.0e}: \
             max |shape(1) − shape(a)| = {drift:.3e} over a signal of range ~2 \
             (must be ~1e-12 — the design + difference penalty are scale-free)."
        );
    }

    // λ̂ must co-transform: for the order-m = 2 (cubic) spline scan,
    // log λ(a·x) = log λ(x) + (2m−1)·log a = log λ(x) + 3·log a.
    let base_ll = fit_scan_log_lambda(1.0, n);
    for &a in &[1.0e3, 1.0e-3] {
        let got = fit_scan_log_lambda(a, n);
        let expected = base_ll + 3.0 * a.ln();
        assert!(
            (got - expected).abs() < 1.0e-3,
            "s(x, bs=\"cr\") log λ̂ not equivariant under covariate rescale a={a:.0e}: \
             got {got:.6}, expected {expected:.6} (= log λ̂(1) + 3·ln a)."
        );
    }
}

#[test]
fn tp_smooth_is_invariant_to_covariate_translation() {
    init_parallelism();
    let formula = "y ~ s(x, bs=\"tp\")";
    let n = 400;
    let base = fit_grid_predictions_offset(formula, 0.0, n);

    // A thin-plate spline is an exactly translation-EQUIVARIANT functional of the
    // data: both the radial kernel `η(x_i − x_j)` and the `∫(f'')²` penalty depend
    // only on coordinate differences, and the polynomial null space `{1, x}` is
    // shift-closed (`{1, x + b}` spans the same space). So fitting on `x` and on
    // `x + b` and predicting at correspondingly shifted points must return
    // identical curves. Before #1269 the null-space block was assembled at the
    // *absolute* coordinate, so a large offset near-collinearized `{1, x}`,
    // ill-conditioned the design, and drifted REML λ̂ — moving the fit ~1.4% of
    // signal range (saturating with |b|). The fix builds the basis in the knot
    // cloud's centred frame, so the fit is location-free like `bs="cr"`/`"ps"`.
    // Offsets span the issue's reported table (drift saturated at ~2.8e-2 for
    // b ≥ 50). Capped at 100: building the centred frame from a knot mean ≈ b
    // incurs an unavoidable `b·ε` cancellation when subtracting it back off the
    // raw coordinate, so the achievable floor grows with |b| (~2e-14 at b=100);
    // a 1e-10 ceiling stays ~6 orders below the bug while above that floor.
    for &b in &[1.0, 10.0, 50.0, 100.0] {
        let drift = max_shape_drift(&base, &fit_grid_predictions_offset(formula, b, n));
        assert!(
            drift < 1.0e-10,
            "s(x, bs=\"tp\") fitted function changed under covariate translation b={b:.0e}: \
             max |shape(0) − shape(b)| = {drift:.3e} over a signal of range ~2 \
             (must be ~1e-13 — the kernel and penalty are translation-invariant; \
             observed ~3e-2 before the polynomial null space was built in a centred frame)."
        );
    }
}

#[test]
fn tp_smooth_is_invariant_to_covariate_rescaling() {
    init_parallelism();
    let formula = "y ~ s(x, bs=\"tp\")";
    let n = 400;
    let base = fit_grid_predictions(formula, 1.0, n);

    // The thin-plate fitted function must be invariant to `x → a·x`: the
    // length-scale is now seeded and optimized in scale-free (standardized)
    // covariate coordinates. Observed pre-fix drift was a bimodal ~2e-2 step;
    // post-fix the fit lands in one scale-free basin.
    for &a in &[1.0e1, 1.0e2, 1.0e3, 1.0e-2, 1.0e-3] {
        let drift = max_shape_drift(&base, &fit_grid_predictions(formula, a, n));
        assert!(
            drift < 1.0e-4,
            "s(x, bs=\"tp\") fitted function changed under covariate rescale a={a:.0e}: \
             max |shape(1) − shape(a)| = {drift:.3e} over a signal of range ~2 \
             (must be scale-free; observed ~2e-2 before the length-scale seed was \
             normalized by the covariate spread)."
        );
    }
}
