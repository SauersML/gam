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

/// Selected `log λ` for the exact spline-scan path. `bs="cr"`
/// (`NaturalCubicRegression`) is DELIBERATELY routed off the scan onto the dense
/// path (#1844 `cb11c502f` / #1957 `d99528cd1`): cr is a finite regression-spline
/// basis, not the scan's full smoothing-spline state-space posterior, so the scan
/// would solve a different model and return a non-`Standard` result predict cannot
/// replay. The scan's covariate-rescaling equivariance (the `spline_scan.rs`
/// log-λ-bracket anchoring fix this loop guards) is therefore exercised through
/// the single-penalty free B-spline `s(x, double_penalty=false)`, which is exactly
/// what auto-routes to the exact O(n) scan (see `spline_scan_fast_path` /
/// `spline_scan_workflow_equivalence`).
fn fit_scan_log_lambda(a: f64, n: usize) -> f64 {
    let data = dataset_with_covariate_scale(a, n);
    let cfg = FitConfig {
        family: Some("gaussian".to_string()),
        ..FitConfig::default()
    };
    match fit_from_formula("y ~ s(x, double_penalty=false)", &data, &cfg).expect("fit ok") {
        FitResult::SplineScan(scan) => scan.log_lambda,
        _ => panic!(
            "expected the single-penalty free B-spline to route to the exact spline scan"
        ),
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

    // λ̂ must co-transform on the exact spline-scan path (the `spline_scan.rs`
    // log-λ-bracket anchoring fix). cr is now routed off the scan (#1844/#1957),
    // so this equivariance is exercised through the single-penalty free B-spline
    // `s(x, double_penalty=false)`, which auto-routes to the order-m = 2 (cubic)
    // scan: log λ(a·x) = log λ(x) + (2m−1)·log a = log λ(x) + 3·log a.
    let base_ll = fit_scan_log_lambda(1.0, n);
    for &a in &[1.0e3, 1.0e-3] {
        let got = fit_scan_log_lambda(a, n);
        let expected = base_ll + 3.0 * a.ln();
        assert!(
            (got - expected).abs() < 1.0e-3,
            "spline-scan log λ̂ not equivariant under covariate rescale a={a:.0e}: \
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
    // (essentially) identical curves. Before #1269 the null-space block was
    // assembled at the *absolute* coordinate, so a large offset near-collinearized
    // `{1, x}`, ill-conditioned the design, and drifted REML λ̂ — moving the fit
    // ~1.4% (2.8e-2) of signal range, *saturating* with |b|. The #1269 fix builds
    // the basis in the knot cloud's centred frame; the BASIS is now exactly
    // translation-invariant to ~1e-13 (pinned tightly and directly by
    // `tp_basis_is_exactly_translation_invariant` below, the meaningful guard on
    // the conditioning fix).
    //
    // This end-to-end gate is necessarily looser than that basis gate, and the
    // gap is a *physical* floor, not slack. The full fit also runs the REML
    // penalty-weight optimizer, and on this dataset the data wants heavy smoothing
    // (`log λ̂ ≈ 8.3`, λ̂ ≈ 4000): the posterior sits near the unpenalized linear
    // null space, where the REML objective is very flat in `log λ`. The covariate
    // arrives as the float `x + b`, which has already *rounded off* ~ulp(b) of the
    // original `x` (the lost bits are unrecoverable by any library code), so two
    // offset fits see designs differing by ~1e-13. That floor-level perturbation,
    // divided by the near-zero `log λ` curvature, drifts `log λ̂` in its 5th digit
    // (≈6e-5) and moves the curve by ~1e-7 — observed flat at ~3e-8–1.2e-7 for
    // EVERY b ≥ 1e-6, independent of |b| and of the outer iteration budget
    // (identical at 20/60/200 iters: the optimizer is fully converged to each
    // offset's own argmax). A `< 1e-5` ceiling sits ~3.5 orders below the original
    // 2.8e-2 bug — so any regression of the #1269 conditioning fix trips it — while
    // staying ~2 orders above the irreducible argmax floor. Offsets span the
    // issue's reported table (drift saturated at ~2.8e-2 for b ≥ 50).
    for &b in &[1.0, 10.0, 50.0, 100.0] {
        let drift = max_shape_drift(&base, &fit_grid_predictions_offset(formula, b, n));
        assert!(
            drift < 1.0e-5,
            "s(x, bs=\"tp\") fitted function changed under covariate translation b={b:.0e}: \
             max |shape(0) − shape(b)| = {drift:.3e} over a signal of range ~2 \
             (must stay ≪ the 2.8e-2 pre-#1269 bug; the kernel, penalty, and basis \
             are translation-invariant to ~1e-13, the residual is the flat-REML-λ \
             argmax floor amplifying the ulp(b) rounding of the float input x+b)."
        );
    }
}

/// Strict, direct guard on the #1269 conditioning fix: the thin-plate *basis
/// design* — built in the knot cloud's centred frame and standardized exactly as
/// the production term arm does (#1215) — must be translation-invariant to
/// floating-point floor, isolated from the REML penalty-weight optimizer.
///
/// We build the basis on `x` and on `x + b` (frozen length scale, frozen
/// EqualMass centers, frozen input scales) and project the same response through
/// a *fixed* tiny ridge — no λ-selection. Pre-#1269 the absolute-coordinate
/// polynomial null space near-collinearized `{1, x}` and the design drifted by
/// ~3e-2 of curve range; post-fix the centred frame makes it exact to ~1e-13.
/// This is the test that should be tight; the end-to-end
/// `tp_smooth_is_invariant_to_covariate_translation` above necessarily floats on
/// the REML argmax floor on top of this.
#[test]
fn tp_basis_is_exactly_translation_invariant() {
    use gam::basis::{
        BasisMetadata, CenterStrategy, SpatialIdentifiability, ThinPlateBasisSpec,
        build_thin_plate_basis,
    };
    use gam::faer_ndarray::FaerCholesky;
    use gam::smooth::input_standardization::{
        apply_input_standardization, compensate_length_scale_for_standardization,
        compute_spatial_input_scales,
    };
    init_parallelism();
    let n = 400;
    let probes: Vec<f64> = (0..15).map(|i| 0.05 + 0.90 * (i as f64) / 14.0).collect();

    let build_and_predict = |b: f64| -> Vec<f64> {
        let mut rng = StdRng::seed_from_u64(20240616);
        let u = Uniform::new(0.0_f64, 1.0).expect("uniform");
        let noise = Normal::new(0.0, 0.2).expect("normal");
        let mut x_train = Array2::<f64>::zeros((n, 1));
        let mut ys = Vec::with_capacity(n);
        for i in 0..n {
            let x = u.sample(&mut rng);
            ys.push((2.0 * std::f64::consts::PI * x).sin() + noise.sample(&mut rng));
            x_train[[i, 0]] = x + b;
        }
        let y = ndarray::Array1::from_vec(ys);

        let user_ls = 1.0_f64;
        let scales = compute_spatial_input_scales(x_train.view());
        let (length_scale, scales_vec) = if let Some(s) = &scales {
            apply_input_standardization(&mut x_train, s);
            (
                compensate_length_scale_for_standardization(user_ls, s),
                Some(s.clone()),
            )
        } else {
            (user_ls, None)
        };

        let train_spec = ThinPlateBasisSpec {
            center_strategy: CenterStrategy::EqualMass { num_centers: 10 },
            periodic: None,
            length_scale,
            double_penalty: false,
            identifiability: SpatialIdentifiability::None,
            radial_reparam: None,
        };
        let train = build_thin_plate_basis(x_train.view(), &train_spec).expect("train basis");
        let (fit_centers, radial_reparam) = match &train.metadata {
            BasisMetadata::ThinPlate {
                centers,
                radial_reparam,
                ..
            } => (centers.clone(), radial_reparam.clone()),
            _ => panic!("expected ThinPlate metadata"),
        };
        let design = train.design.to_dense();
        // Fixed tiny ridge in the basis coordinate (NO REML λ-selection).
        let p = design.ncols();
        let xtx = design.t().dot(&design);
        let mut g = xtx.clone();
        let max_diag = xtx.diag().iter().cloned().fold(1.0_f64, f64::max);
        let eps = 1e-6 * max_diag;
        for i in 0..p {
            g[[i, i]] += eps;
        }
        let xty = design.t().dot(&y);
        let chol = g.cholesky(faer::Side::Lower).expect("chol");
        let beta = chol.solvevec(&xty);

        let mut grid = Array2::<f64>::zeros((probes.len(), 1));
        for (i, &v) in probes.iter().enumerate() {
            grid[[i, 0]] = v + b;
        }
        if let Some(s) = &scales_vec {
            apply_input_standardization(&mut grid, s);
        }
        let test_spec = ThinPlateBasisSpec {
            center_strategy: CenterStrategy::UserProvided(fit_centers),
            periodic: None,
            length_scale,
            double_penalty: false,
            identifiability: SpatialIdentifiability::None,
            radial_reparam,
        };
        let test = build_thin_plate_basis(grid.view(), &test_spec).expect("test basis");
        test.design.to_dense().dot(&beta).to_vec()
    };

    let base = build_and_predict(0.0);
    for &b in &[1.0, 10.0, 50.0, 100.0] {
        let drift = max_shape_drift(&base, &build_and_predict(b));
        assert!(
            drift < 1.0e-12,
            "thin-plate BASIS design changed under covariate translation b={b:.0e}: \
             max |shape(0) − shape(b)| = {drift:.3e} (must be ~1e-13 — #1269 builds \
             the polynomial null space in the knot cloud's centred frame; observed \
             ~3e-2 when it was assembled at the absolute coordinate)."
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
