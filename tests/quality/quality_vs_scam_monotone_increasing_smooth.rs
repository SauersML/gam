//! End-to-end quality: gam's **monotone-increasing shape-constrained smooth**
//! `s(x, shape=monotone_increasing)` must (a) RECOVER a known strictly-increasing
//! nonlinear truth from noisy data, (b) produce a fitted curve that is itself
//! genuinely NON-DECREASING (gam's own constraint property, not a peer tool's),
//! and (c) match-or-beat R `scam`'s monotone P-spline on truth recovery — not
//! merely reproduce scam's fitted output.
//!
//! Why this combination earns its own test. Shape-constrained smoothing is a
//! distinct capability from an unconstrained `s(x)`: the penalised basis is
//! reparameterised so the fitted function is provably monotone for ANY noise
//! realization, trading a little flexibility for a hard structural guarantee.
//! The canonical mature reference is the R `scam` package (Pya & Wood), whose
//! monotone-increasing P-spline is `bs="mpi"` (monotone P-spline, increasing).
//! scam is demoted here to a match-or-beat accuracy baseline on the SAME
//! truth-recovery objective; it is never an output to imitate. gam's constraint
//! (assertion 2) is checked against the analytic definition of monotonicity, so
//! it stands on its own even if scam were absent.
//!
//! Data (seed=20260621, n=300): x ~ U(0,1) sorted; truth is a logistic ramp
//! `f(x) = 1/(1+exp(-10*(x-0.5)))`, a smooth STRICTLY-increasing nonlinear S
//! curve on [0,1] with range ≈ [0.0067, 0.993] (signal range ≈ 0.987). Gaussian
//! noise sigma = 0.20 ≈ 20% of the signal range is added. A fixed seed feeds the
//! BYTE-IDENTICAL (x, y) to both gam and scam.
//!
//! Asserts:
//!   1. PRIMARY truth recovery: gam's RMSE-to-truth on a dense grid is below the
//!      noise floor sigma — gam recovers the monotone truth despite the noise.
//!   2. MONOTONICITY of gam's OWN fitted curve on a dense sorted grid: successive
//!      fitted values are non-decreasing to within a tiny numerical epsilon
//!      (1e-6). A broken constraint reparameterisation would dip here regardless
//!      of any reference tool.
//!   3. MATCH-OR-BEAT: gam's truth-recovery RMSE <= scam's RMSE * 1.15.

use gam::matrix::LinearOperator;
use gam::smooth::build_term_collection_design;
use gam::test_support::reference::{Column, rmse, run_r};
use gam::{
    FitConfig, FitResult, encode_recordswith_inferred_schema, fit_from_formula, init_parallelism,
};
use ndarray::Array2;
use rand::SeedableRng;
use rand::rngs::StdRng;
use rand_distr::{Distribution, Normal, Uniform};

const N: usize = 300;
const SIGMA: f64 = 0.20;

/// The strictly-increasing nonlinear truth: a logistic ramp on [0,1]. Its first
/// derivative `10*f*(1-f)` is strictly positive everywhere, so the function is
/// genuinely monotone increasing (not merely non-decreasing).
fn truth(x: f64) -> f64 {
    1.0 / (1.0 + (-10.0 * (x - 0.5)).exp())
}

#[test]
fn gam_monotone_increasing_smooth_recovers_truth_and_matches_scam() {
    init_parallelism();

    // ---- synthetic monotone-increasing truth on [0,1] ---------------------
    // x sorted so the monotonicity check on the fitted training-row order is
    // meaningful; y = f(x) + N(0, sigma). A fixed seed feeds identical draws to
    // gam and scam.
    let mut rng = StdRng::seed_from_u64(20260621);
    let unif = Uniform::new(0.0_f64, 1.0).expect("uniform [0,1]");
    let noise = Normal::new(0.0, SIGMA).expect("gaussian noise");

    let mut x: Vec<f64> = (0..N).map(|_| unif.sample(&mut rng)).collect();
    x.sort_by(|a, b| a.partial_cmp(b).expect("finite x"));
    let y: Vec<f64> = x.iter().map(|&xi| truth(xi) + noise.sample(&mut rng)).collect();

    // Signal range of the truth; the noise floor (sigma) is the recovery bar.
    let signal = {
        let lo = x.iter().cloned().map(truth).fold(f64::INFINITY, f64::min);
        let hi = x.iter().cloned().map(truth).fold(f64::NEG_INFINITY, f64::max);
        hi - lo
    };

    // ---- fit with gam: shape-constrained monotone-increasing smooth -------
    let headers: Vec<String> = ["x", "y"].into_iter().map(String::from).collect();
    let rows: Vec<csv::StringRecord> = (0..N)
        .map(|i| csv::StringRecord::from(vec![x[i].to_string(), y[i].to_string()]))
        .collect();
    let ds = encode_recordswith_inferred_schema(headers, rows).expect("encode monotone dataset");
    let x_idx = ds.column_map()["x"];
    let width = ds.headers.len();

    let cfg = FitConfig {
        family: Some("gaussian".to_string()),
        ..FitConfig::default()
    };
    let result = fit_from_formula("y ~ s(x, k=12, shape=monotone_increasing)", &ds, &cfg)
        .expect("gam monotone-increasing shape-constrained fit");
    let FitResult::Standard(fit) = result else {
        panic!("Gaussian identity-link smooth => expected FitResult::Standard");
    };

    // Dense sorted evaluation grid spanning the support. Used both for
    // truth-recovery (vs analytic f) and for the monotonicity check.
    const G: usize = 200;
    let grid: Vec<f64> = (0..G).map(|i| i as f64 / (G as f64 - 1.0)).collect();
    let truth_grid: Vec<f64> = grid.iter().map(|&xg| truth(xg)).collect();

    let mut pts = Array2::<f64>::zeros((G, width));
    for (r, &xg) in grid.iter().enumerate() {
        pts[[r, x_idx]] = xg;
    }
    let design = build_term_collection_design(pts.view(), &fit.resolvedspec)
        .expect("rebuild monotone design on grid");
    // Gaussian identity link => the linear predictor IS the fitted mean.
    let gam_grid: Vec<f64> = design.design.apply(&fit.fit.beta).to_vec();

    // gam's OWN constraint property: maximum downward step across the sorted grid
    // (a positive number means the fitted curve DECREASED there). For a correct
    // monotone-increasing constraint this is ~0 (numerical noise only).
    let max_violation = gam_grid
        .windows(2)
        .map(|w| w[0] - w[1]) // >0 iff value decreased
        .fold(0.0_f64, f64::max);

    // ---- fit the SAME data with scam (the mature reference) ---------------
    // scam(y ~ s(x, bs="mpi")) is the monotone-increasing P-spline; gaussian().
    let r = run_r(
        &[Column::new("x", &x), Column::new("y", &y)],
        &format!(
            r#"
            suppressPackageStartupMessages(library(scam))
            m <- scam(y ~ s(x, k = 12, bs = "mpi"), data = df, family = gaussian())
            xg <- seq(0, 1, length.out = {ng})
            emit("pred", as.numeric(predict(m, newdata = data.frame(x = xg))))
            "#,
            ng = grid.len(),
        ),
    );
    let scam_grid = r.vector("pred");
    assert_eq!(scam_grid.len(), grid.len(), "scam prediction length mismatch");

    // ---- OBJECTIVE METRICS -------------------------------------------------
    let gam_err = rmse(&gam_grid, &truth_grid);
    let scam_err = rmse(scam_grid, &truth_grid);

    eprintln!(
        "[scam-monotone] n={N} sigma={SIGMA} signal_range={signal:.4} \
         gam_rmse={gam_err:.5} scam_rmse={scam_err:.5} \
         max_monotone_violation={max_violation:.3e} ratio={:.3}",
        gam_err / scam_err.max(1e-12)
    );

    // PRIMARY: gam recovers the monotone truth to better than the noise floor.
    // A broken constraint basis or a mis-smoothed fit would smear the S curve
    // well past sigma.
    assert!(
        gam_err < SIGMA,
        "gam should recover the monotone-increasing truth below the noise floor: \
         rmse={gam_err:.5} (bar = sigma {SIGMA})"
    );

    // CONSTRAINT (gam's own property): the fitted curve is non-decreasing on the
    // dense sorted grid to within numerical epsilon. This stands independent of
    // any reference tool — it is the defining guarantee of the capability.
    assert!(
        max_violation < 1e-6,
        "gam's monotone-increasing fitted curve must be non-decreasing: \
         max downward step {max_violation:.3e} exceeds 1e-6"
    );

    // MATCH-OR-BEAT: gam's truth-recovery RMSE is no worse than scam's by more
    // than 15%, holding scam's monotone P-spline as an accuracy baseline.
    assert!(
        gam_err <= scam_err * 1.15,
        "gam's monotone truth-recovery must match-or-beat scam: \
         gam_rmse={gam_err:.5} vs scam*1.15={:.5}",
        scam_err * 1.15
    );
}
