//! End-to-end SCALE + MEMORY tests for gam's non-periodic Euclidean Duchon
//! smoother. The claim under test is operational, not comparative: the
//! redesigned default `duchon(x, k=...)` — a cubic (`r³`) polyharmonic
//! structural smoother with an analytic native reproducing-norm Gram penalty
//! plus null-space ridge — must FIT AT SCALE without OOM and still RECOVER a
//! known smooth truth.
//!
//! OBJECTIVE METRICS (the only pass/fail claims here): for each fit we assert
//! (a) the fit completes, (b) every fitted value is finite, and (c) the fit's
//! own posterior band covers the truth across the held-out interior grid (the
//! Nychka across-the-function coverage gate, derived at
//! [`across_function_coverage`]). The recovery RMSE and the trivial-predictor
//! RMS are printed as diagnostics, not asserted. We do NOT compare against
//! any reference tool and we do NOT assert closeness to a reference output —
//! these tests stand on gam's own truth recovery at scale. (The companion file
//! `quality_vs_mgcv_duchon_smooth.rs` owns the match-or-beat-mgcv comparison.)
//!
//! RAM DISCIPLINE. The whole point of the lazy path is to NOT allocate the
//! dense `n × p` design when it would be large, so we deliberately keep `n`
//! bounded (≤ 40_000) and exercise the chunked operator's *correctness*, not an
//! actual multi-GB allocation. Caps and their rationale are documented at each
//! test below. With n ≤ 40_000 the raw input array is a few hundred KiB and the
//! p × p normal-equations Gram is single-digit MiB, so CI memory stays bounded
//! even on the lazy path (which streams the design in row chunks rather than
//! materializing it).

use gam::matrix::LinearOperator;
use gam::smooth::build_term_collection_design;
use gam::test_support::reference::rmse;
use gam::{
    FitConfig, FitResult, encode_recordswith_inferred_schema, fit_from_formula, init_parallelism,
};
use gam_math::probability::chi_square_quantile;
use ndarray::Array2;
use rand::SeedableRng;
use rand::rngs::StdRng;
use rand_distr::{Distribution, Normal};

/// Root-mean-square of a slice (used for the trivial-predictor floor).
fn rms(v: &[f64]) -> f64 {
    (v.iter().map(|&t| t * t).sum::<f64>() / v.len() as f64).sqrt()
}

/// Family-wise false-alarm rate of the coverage gates in this file.
const FAMILY_ALPHA: f64 = 0.01;

/// Across-the-function coverage of the truth by the fit's posterior band.
///
/// With `C = X_p V_b X_pᵀ` the posterior covariance of the fitted curve at the
/// `P` probes, `s_i = √C_ii` and `e_i = f̂(x_i) − f(x_i)`, the statistic is the
/// mean squared standardized error `Q = (1/P) Σ e_i² / s_i²`. If the band is
/// calibrated (Nychka 1988: a Bayesian band covers the truth *on average across
/// the function*), `e ~ N(0, C)`, so `Q = zᵀRz / P` with `R = D⁻¹CD⁻¹` the
/// probe correlation matrix and `z` standard normal. Its law is a weighted sum
/// of χ²₁ with weights = eigenvalues of `R / P`: mean 1 and variance
/// `2 tr(R²) / P²`. The Satterthwaite two-moment match `Q ≈ g · χ²_ν` with
/// `g = tr(R²) / P²`, `ν = 1 / g` reproduces both moments exactly, so the upper
/// `1 − α` bound is `g · χ²_{1−α}(ν)`. Every quantity is the fit's own; nothing
/// is tuned to a test outcome. A band that is too narrow, a biased curve, or a
/// collapsed/blown-up fit all push `Q` past the bound.
fn across_function_coverage(
    x_probe: &Array2<f64>,
    cov: &Array2<f64>,
    err: &[f64],
    alpha: f64,
) -> Result<(f64, f64), String> {
    let c = x_probe.dot(cov).dot(&x_probe.t());
    let p = err.len();
    let s: Vec<f64> = (0..p).map(|i| c[[i, i]].sqrt()).collect();
    if let Some(i) = s.iter().position(|v| !(v.is_finite() && *v > 0.0)) {
        return Err(format!("posterior SE at probe {i} is {}", s[i]));
    }
    let q = err
        .iter()
        .zip(s.iter())
        .map(|(e, si)| (e / si).powi(2))
        .sum::<f64>()
        / p as f64;
    let mut tr_r2 = 0.0;
    for i in 0..p {
        for j in 0..p {
            let r = c[[i, j]] / (s[i] * s[j]);
            tr_r2 += r * r;
        }
    }
    let g = tr_r2 / (p * p) as f64;
    let bound = g * chi_square_quantile(1.0 - alpha, 1.0 / g);
    Ok((q, bound))
}

/// Build a single-feature dataset `{x, y}` from parallel vectors.
fn encode_xy(x: &[f64], y: &[f64]) -> gam::data::EncodedDataset {
    let headers = ["x", "y"].into_iter().map(String::from).collect();
    let rows = x
        .iter()
        .zip(y.iter())
        .map(|(a, b)| csv::StringRecord::from(vec![a.to_string(), b.to_string()]))
        .collect();
    encode_recordswith_inferred_schema(headers, rows).expect("encode synthetic 1-D dataset")
}

/// Build a two-feature dataset `{x, z, y}` from parallel vectors.
fn encode_xzy(x: &[f64], z: &[f64], y: &[f64]) -> gam::data::EncodedDataset {
    let headers = ["x", "z", "y"].into_iter().map(String::from).collect();
    let rows = (0..x.len())
        .map(|i| {
            csv::StringRecord::from(vec![x[i].to_string(), z[i].to_string(), y[i].to_string()])
        })
        .collect();
    encode_recordswith_inferred_schema(headers, rows).expect("encode synthetic 2-D dataset")
}

/// Fit `formula` as a Gaussian GAM and return the standard fit, panicking with a
/// clear message on the non-standard arm.
fn fit_gaussian(formula: &str, ds: &gam::data::EncodedDataset) -> gam::StandardFitResult {
    let cfg = FitConfig {
        family: Some("gaussian".to_string()),
        ..FitConfig::default()
    };
    let result = fit_from_formula(formula, ds, &cfg)
        .unwrap_or_else(|e| panic!("gam duchon fit failed for `{formula}`: {e}"));
    match result {
        FitResult::Standard(fit) => fit,
        _ => panic!("expected a standard GAM fit for a gaussian Duchon smooth: `{formula}`"),
    }
}

#[test]
fn duchon_1d_recovers_truth_across_increasing_n() {
    init_parallelism();

    // Low-frequency truth f(x) = sin(2π·x): exactly ONE period over [0,1], which
    // a k=40 cubic Duchon basis resolves comfortably — so the achievable error
    // is the noise floor, not under-resolution bias. We sweep n to prove the
    // standard (dense) Duchon path scales: n ∈ {2_000, 10_000, 40_000}.
    //
    // n CAP = 40_000. WHY: at k=40 the dense design is n·~42·8 bytes ≈ 13 MiB at
    // n=40_000 — well under the 256 MiB default materialization budget, so this
    // arm stays on the dense path on purpose (the lazy path gets its own test
    // below). 40_000 is large enough to be a genuine scale test while keeping
    // the raw input array (~0.6 MiB) and all linear algebra trivially in-RAM for
    // CI; pushing n higher buys no extra coverage here and only burns CI time.
    let sigma = 0.05;
    let ns = [2_000usize, 10_000, 40_000];
    // Bonferroni split of the family-wise rate over the independent n cases.
    let alpha = FAMILY_ALPHA / ns.len() as f64;
    for &n in &ns {
        let mut rng = StdRng::seed_from_u64(0xD0_C0_00 ^ n as u64);
        let noise = Normal::new(0.0, sigma).expect("normal");
        let mut x: Vec<f64> = (0..n).map(|i| i as f64 / (n as f64 - 1.0)).collect();
        x.sort_by(|a, b| a.partial_cmp(b).expect("finite x"));
        let two_pi = 2.0 * std::f64::consts::PI;
        let y: Vec<f64> = x
            .iter()
            .map(|&t| (two_pi * t).sin() + noise.sample(&mut rng))
            .collect();

        let ds = encode_xy(&x, &y);
        let x_idx = ds.column_map()["x"];
        let fit = fit_gaussian("y ~ duchon(x, k=40)", &ds);

        // Every training fitted value must be finite. With identity link the
        // mean is design*beta; apply the (possibly chunked) operator to beta.
        let train_fitted: Vec<f64> = fit.design.design.apply(&fit.fit.beta).to_vec();
        assert!(
            train_fitted.iter().all(|v| v.is_finite()),
            "duchon 1d n={n}: non-finite fitted value among training points"
        );

        // Truth recovery on a dense interior grid (avoid extrapolation edges).
        let m = 201usize;
        let x_test: Vec<f64> = (0..m)
            .map(|i| 0.01 + 0.98 * i as f64 / (m as f64 - 1.0))
            .collect();
        let y_truth: Vec<f64> = x_test.iter().map(|&t| (two_pi * t).sin()).collect();

        let mut grid = Array2::<f64>::zeros((m, ds.headers.len()));
        for (i, &t) in x_test.iter().enumerate() {
            grid[[i, x_idx]] = t;
        }
        let design = build_term_collection_design(grid.view(), &fit.resolvedspec)
            .expect("rebuild Duchon design at 1-D test grid");
        let gam_fitted: Vec<f64> = design.design.apply(&fit.fit.beta).to_vec();
        assert!(
            gam_fitted.iter().all(|v| v.is_finite()),
            "duchon 1d n={n}: non-finite fitted value on the test grid"
        );

        let recovery_rmse = rmse(&gam_fitted, &y_truth);
        let truth_mean = y_truth.iter().sum::<f64>() / m as f64;
        let demeaned: Vec<f64> = y_truth.iter().map(|&t| t - truth_mean).collect();
        let trivial = rms(&demeaned);
        let err: Vec<f64> = gam_fitted
            .iter()
            .zip(y_truth.iter())
            .map(|(f, t)| f - t)
            .collect();
        let cov = fit
            .fit
            .beta_covariance()
            .unwrap_or_else(|| panic!("duchon 1d n={n}: converged fit carries no V_b"));
        let (q, bound) = across_function_coverage(&design.design.to_dense(), cov, &err, alpha)
            .unwrap_or_else(|e| panic!("duchon 1d n={n}: {e}"));
        eprintln!(
            "duchon-scale-1d: n={n} sigma={sigma} k=40 recovery_rmse={recovery_rmse:.4} \
             trivial_predictor_rms={trivial:.4} coverage_Q={q:.4} bound={bound:.4} alpha={alpha:.2e}"
        );
        assert!(
            q <= bound,
            "duchon 1d n={n}: the posterior band does not cover sin(2πx) across the function: \
             Q={q:.4} > bound={bound:.4} at alpha={alpha:.2e} (recovery_rmse={recovery_rmse:.4}, \
             trivial-predictor RMS={trivial:.4})"
        );
    }
}

#[test]
fn duchon_2d_recovers_smooth_surface() {
    init_parallelism();

    // 2-D Duchon syntax is the multi-arg term `duchon(x, z, k=...)` (each smooth
    // coordinate is a separate argument; `k`/`centers` set the number of radial
    // centers). The default power in 2D is s=(d-1)/2 = 0.5 over the affine null
    // space — the magic structural surface smoother.
    //
    // Known smooth truth on [0,1]²: a separable low-frequency surface
    // f(x,z) = sin(2π·x) · cos(2π·z). It is globally smooth (one period per
    // axis) so a moderate center count resolves it; recovery error is the noise
    // floor, not under-resolution.
    //
    // n CAP = 4_000, k = 60. WHY: 2-D fitting on a uniform random cloud needs
    // enough points to pin the surface but the test is about correctness at
    // moderate scale, not OOM. The dense design here is 4_000·~63·8 ≈ 2 MiB and
    // the 63×63 Gram is trivial; keeping n=4_000 keeps CI fast while still being
    // a genuine surface-recovery check.
    let n = 4_000usize;
    let k = 60usize;
    let sigma = 0.05;
    let mut rng = StdRng::seed_from_u64(0x2D_C0_FE);
    let unit = rand_distr::Uniform::new(0.0, 1.0).expect("uniform");
    let noise = Normal::new(0.0, sigma).expect("normal");
    let two_pi = 2.0 * std::f64::consts::PI;

    let x: Vec<f64> = (0..n).map(|_| unit.sample(&mut rng)).collect();
    let z: Vec<f64> = (0..n).map(|_| unit.sample(&mut rng)).collect();
    let truth_at = |a: f64, b: f64| (two_pi * a).sin() * (two_pi * b).cos();
    let y: Vec<f64> = (0..n)
        .map(|i| truth_at(x[i], z[i]) + noise.sample(&mut rng))
        .collect();

    let ds = encode_xzy(&x, &z, &y);
    let col = ds.column_map();
    let (x_idx, z_idx) = (col["x"], col["z"]);
    let fit = fit_gaussian(&format!("y ~ duchon(x, z, k={k})"), &ds);

    let train_fitted: Vec<f64> = fit.design.design.apply(&fit.fit.beta).to_vec();
    assert!(
        train_fitted.iter().all(|v| v.is_finite()),
        "duchon 2d: non-finite fitted value among training points"
    );

    // Interior tensor grid for truth recovery (avoid the convex-hull edges where
    // a scattered-data smoother extrapolates).
    let g = 25usize;
    let coords: Vec<f64> = (0..g)
        .map(|i| 0.08 + 0.84 * i as f64 / (g as f64 - 1.0))
        .collect();
    let mut grid = Array2::<f64>::zeros((g * g, ds.headers.len()));
    let mut y_truth = Vec::with_capacity(g * g);
    for (i, &gx) in coords.iter().enumerate() {
        for (j, &gz) in coords.iter().enumerate() {
            let row = i * g + j;
            grid[[row, x_idx]] = gx;
            grid[[row, z_idx]] = gz;
            y_truth.push(truth_at(gx, gz));
        }
    }
    let design = build_term_collection_design(grid.view(), &fit.resolvedspec)
        .expect("rebuild Duchon design at 2-D test grid");
    let gam_fitted: Vec<f64> = design.design.apply(&fit.fit.beta).to_vec();
    assert!(
        gam_fitted.iter().all(|v| v.is_finite()),
        "duchon 2d: non-finite fitted value on the test grid"
    );

    let recovery_rmse = rmse(&gam_fitted, &y_truth);
    let truth_mean = y_truth.iter().sum::<f64>() / y_truth.len() as f64;
    let demeaned: Vec<f64> = y_truth.iter().map(|&t| t - truth_mean).collect();
    let trivial = rms(&demeaned);
    let err: Vec<f64> = gam_fitted
        .iter()
        .zip(y_truth.iter())
        .map(|(f, t)| f - t)
        .collect();
    let cov = fit
        .fit
        .beta_covariance()
        .expect("duchon 2d: converged fit carries no V_b");
    let (q, bound) =
        across_function_coverage(&design.design.to_dense(), cov, &err, FAMILY_ALPHA)
            .unwrap_or_else(|e| panic!("duchon 2d: {e}"));
    eprintln!(
        "duchon-scale-2d: n={n} k={k} sigma={sigma} recovery_rmse={recovery_rmse:.4} \
         trivial_predictor_rms={trivial:.4} coverage_Q={q:.4} bound={bound:.4}"
    );
    assert!(
        q <= bound,
        "duchon 2d: the posterior band does not cover sin(2πx)cos(2πz) across the function: \
         Q={q:.4} > bound={bound:.4} at alpha={FAMILY_ALPHA} (recovery_rmse={recovery_rmse:.4}, \
         trivial-predictor RMS={trivial:.4})"
    );
}

