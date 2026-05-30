//! End-to-end quality: gam's Gaussian *location-scale* fit with MULTIPLE
//! additive thin-plate smooths in BOTH the mean and the log-sigma blocks must
//! match `gamlss::gamlss(family = NO())` — the reference GAMLSS implementation
//! for distributional regression and the de-facto standard for location-scale
//! models with smooth mean and smooth log-variance in R. To compare like with
//! like, the gamlss side uses the SAME thin-plate basis as gam via
//! `ga(~ s(x, bs="tp"))` (the `gamlss.add` bridge to `mgcv`'s `s()`), not the
//! P-spline `pb()` default, so any divergence is in the solver, not the basis.
//!
//! This is the cross-feature combination that single-smooth location-scale
//! tests never exercise: family (Gaussian) x TWO additive smooths per block
//! (mu = s(x1) + s(x2), log-sigma = s(x1) + s(x2)) fit jointly by penalized
//! blockwise PIRLS. With more than one penalized term in each block, the design
//! is the concatenation of per-term sub-bases and the penalty is a block-
//! diagonal concatenation of per-term penalties; recovering each contribution
//! correctly requires gam's penalty-block alignment and blockwise Jacobian to
//! keep every term's column range and penalty in register across BOTH active
//! blocks. A bug that mis-aligns a penalty block or leaks one term's columns
//! into another's would distort the recovered additive surface — invisible to a
//! single-smooth test, caught here.
//!
//! We synthesize an additive heteroscedastic recipe, feed the *identical*
//! (x1, x2, y) rows to both engines, and compare the recovered surfaces — not
//! held-out predictions — on a dense 10x10 grid over [0,1]^2:
//!   * the fitted mean mu(x1, x2), and
//!   * the fitted log standard deviation log(sigma(x1, x2)).
//!
//! Both engines maximize the same Gaussian location-scale (penalized) joint
//! log-likelihood `ell = -1/2 sum (y-mu)^2 / sigma^2 - sum log sigma`, so the
//! recovered additive mean and log-sigma surfaces should converge to nearly
//! identical shapes up to basis-convention / numerical-integration tolerance.
//! A genuine divergence here is a real bug in gam's blockwise location-scale
//! solver or its multi-term penalty concatenation.
//!
//! Notes on the gam side that this test pins down by reading the source:
//!   * `fit_from_formula(..., FitConfig{ noise_formula: Some(...), .. })` routes
//!     through `materialize_location_scale` -> `FitRequest::GaussianLocationScale`.
//!     This in-Rust path does NOT rescale `y`, so the reconstructed mu / sigma
//!     are already in raw response units.
//!   * gam's noise (sigma) link is `sigma = LOGB_SIGMA_FLOOR + exp(eta_scale)`
//!     with `LOGB_SIGMA_FLOOR = 0.01` (see `families::sigma_link`); the location
//!     block carries role `BlockRole::Location`, the log-sigma block role
//!     `BlockRole::Scale`.
//!   * The spec's `linkwiggle(...)` term is a *binomial-only* link correction
//!     (`reject_explicit_linkwiggle_for_nonbinomial` rejects it for a Gaussian
//!     response); it is meaningless here, so the gam formula is the pair of
//!     two-smooth additive blocks without it.

use gam::estimate::BlockRole;
use gam::gamlss::GaussianLocationScaleFitResult;
use gam::matrix::LinearOperator;
use gam::smooth::build_term_collection_design;
use gam::test_support::reference::{Column, relative_l2, run_r};
use gam::{
    FitConfig, FitResult, encode_recordswith_inferred_schema, fit_from_formula, init_parallelism,
};
use ndarray::Array2;

/// gam's location-scale noise link floor: sigma = 0.01 + exp(eta_scale).
/// Mirrors `families::sigma_link::LOGB_SIGMA_FLOOR`.
const LOGB_SIGMA_FLOOR: f64 = 0.01;

#[test]
fn gam_gaussian_multi_smooth_matches_gamlss() {
    init_parallelism();

    // ---- synthetic additive heteroscedastic recipe (fed IDENTICALLY) -------
    // n=250, x1 ~ Uniform(0,1), x2 ~ Uniform(0,1),
    // mu(x1,x2)    = sin(2*pi*x1) + cos(2*pi*x2),
    // sigma(x1,x2) = 0.1 + 0.1*sin(pi*x1) + 0.05*x2,
    // y ~ N(mu, sigma^2), seed=999. A deterministic seeded LCG draws both the
    // uniforms and the standard normals so the exact same data is reproducible
    // in pure Rust and sent verbatim to gamlss.
    let n = 250usize;
    let pi = std::f64::consts::PI;
    let two_pi = 2.0 * pi;

    let mut state: u64 = 999;
    let mut next_unit = || -> f64 {
        // Numerical Recipes LCG; take the high bits for a uniform in [0,1).
        state = state
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        ((state >> 11) as f64) / ((1u64 << 53) as f64)
    };

    let x1: Vec<f64> = (0..n).map(|_| next_unit()).collect();
    let x2: Vec<f64> = (0..n).map(|_| next_unit()).collect();

    // Box-Muller standard normals from the same LCG stream (seed continues).
    let mut zvals: Vec<f64> = Vec::with_capacity(n);
    while zvals.len() < n {
        let u1 = next_unit().max(1e-300);
        let u2 = next_unit();
        let r = (-2.0 * u1.ln()).sqrt();
        zvals.push(r * (two_pi * u2).cos());
        if zvals.len() < n {
            zvals.push(r * (two_pi * u2).sin());
        }
    }

    let mu_true = |a: f64, b: f64| (two_pi * a).sin() + (two_pi * b).cos();
    let sigma_true = |a: f64, b: f64| 0.1 + 0.1 * (pi * a).sin() + 0.05 * b;
    let y: Vec<f64> = (0..n)
        .map(|i| mu_true(x1[i], x2[i]) + sigma_true(x1[i], x2[i]) * zvals[i])
        .collect();

    // ---- build the dataset (columns: x1, x2, y) ----------------------------
    let headers: Vec<String> = vec!["x1".to_string(), "x2".to_string(), "y".to_string()];
    let rows: Vec<csv::StringRecord> = (0..n)
        .map(|i| {
            csv::StringRecord::from(vec![
                format!("{:.17e}", x1[i]),
                format!("{:.17e}", x2[i]),
                format!("{:.17e}", y[i]),
            ])
        })
        .collect();
    let ds = encode_recordswith_inferred_schema(headers, rows).expect("encode multi-smooth data");
    let col = ds.column_map();
    let x1_idx = col["x1"];
    let x2_idx = col["x2"];
    let ncols = ds.headers.len();

    // ---- fit with gam: TWO smooths in each block ---------------------------
    // mu       ~ s(x1, bs='tp') + s(x2, bs='tp')
    // log-sigma ~ s(x1, bs='tp') + s(x2, bs='tp')
    let cfg = FitConfig {
        family: Some("gaussian".to_string()),
        noise_formula: Some("s(x1, bs='tp') + s(x2, bs='tp')".to_string()),
        ..FitConfig::default()
    };
    let result = fit_from_formula("y ~ s(x1, bs='tp') + s(x2, bs='tp')", &ds, &cfg)
        .expect("gam multi-smooth location-scale fit");
    let FitResult::GaussianLocationScale(GaussianLocationScaleFitResult { fit, .. }) = result
    else {
        panic!("expected a Gaussian location-scale fit");
    };

    let beta_location = fit
        .fit
        .block_by_role(BlockRole::Location)
        .expect("location (mean) block present")
        .beta
        .clone();
    let beta_scale = fit
        .fit
        .block_by_role(BlockRole::Scale)
        .expect("scale (log-sigma) block present")
        .beta
        .clone();

    // ---- evaluate gam's surfaces on a dense 10x10 grid over [0,1]^2 --------
    let g = 10usize;
    let grid_n = g * g;
    let axis: Vec<f64> = (0..g).map(|i| i as f64 / (g as f64 - 1.0)).collect();
    let mut grid_x1: Vec<f64> = Vec::with_capacity(grid_n);
    let mut grid_x2: Vec<f64> = Vec::with_capacity(grid_n);
    for &a in &axis {
        for &b in &axis {
            grid_x1.push(a);
            grid_x2.push(b);
        }
    }
    let mut grid = Array2::<f64>::zeros((grid_n, ncols));
    for i in 0..grid_n {
        grid[[i, x1_idx]] = grid_x1[i];
        grid[[i, x2_idx]] = grid_x2[i];
    }

    // Rebuild the SAME frozen mean / log-sigma designs at the grid points and
    // apply each block's coefficients. mu = X_mean*beta_location;
    // sigma = LOGB_SIGMA_FLOOR + exp(X_scale*beta_scale).
    let mean_design_grid = build_term_collection_design(grid.view(), &fit.meanspec_resolved)
        .expect("rebuild mean design at grid");
    let scale_design_grid = build_term_collection_design(grid.view(), &fit.noisespec_resolved)
        .expect("rebuild log-sigma design at grid");

    let gam_mu: Vec<f64> = mean_design_grid.design.apply(&beta_location).to_vec();
    let gam_eta_sigma: Vec<f64> = scale_design_grid.design.apply(&beta_scale).to_vec();
    let gam_sigma: Vec<f64> = gam_eta_sigma
        .iter()
        .map(|&e| LOGB_SIGMA_FLOOR + e.exp())
        .collect();
    let gam_log_sigma: Vec<f64> = gam_sigma.iter().map(|&s| s.ln()).collect();

    assert_eq!(gam_mu.len(), grid_n);
    assert_eq!(gam_sigma.len(), grid_n);

    // ---- fit the SAME model with gamlss (the mature GAMLSS reference) ------
    // family = NO() (Gaussian with mu + log-sigma); two penalized thin-plate
    // smooths ga(~ s(x1, bs="tp")) + ga(~ s(x2, bs="tp")) in BOTH mu.formula
    // and sigma.formula — the SAME tp basis gam uses — predicted on the
    // identical 10x10 grid. `predictAll(..., data = df)` re-supplies the fitting
    // frame the ga()/mgcv smoother needs to evaluate at new points and returns
    // mu and sigma on the response scale in one call.
    let grid_x1_csv = grid_x1
        .iter()
        .map(|t| format!("{t:.17e}"))
        .collect::<Vec<_>>()
        .join(",");
    let grid_x2_csv = grid_x2
        .iter()
        .map(|t| format!("{t:.17e}"))
        .collect::<Vec<_>>()
        .join(",");
    let body = format!(
        r#"
        suppressPackageStartupMessages(library(gamlss))
        suppressPackageStartupMessages(library(gamlss.add))
        m <- gamlss(y ~ ga(~ s(x1, bs = "tp")) + ga(~ s(x2, bs = "tp")),
                    sigma.formula = ~ ga(~ s(x1, bs = "tp")) + ga(~ s(x2, bs = "tp")),
                    family = NO(), data = df,
                    control = gamlss.control(trace = FALSE))
        gx1 <- as.numeric(strsplit("{grid_x1_csv}", ",")[[1]])
        gx2 <- as.numeric(strsplit("{grid_x2_csv}", ",")[[1]])
        nd <- data.frame(x1 = gx1, x2 = gx2)
        pa <- predictAll(m, newdata = nd, data = df, type = "response")
        emit("mu", as.numeric(pa$mu))
        emit("sigma", as.numeric(pa$sigma))
        "#
    );
    let r = run_r(
        &[
            Column::new("x1", &x1),
            Column::new("x2", &x2),
            Column::new("y", &y),
        ],
        &body,
    );
    let gamlss_mu = r.vector("mu");
    let gamlss_sigma = r.vector("sigma");
    assert_eq!(gamlss_mu.len(), grid_n, "gamlss mu grid length mismatch");
    assert_eq!(
        gamlss_sigma.len(),
        grid_n,
        "gamlss sigma grid length mismatch"
    );
    let gamlss_log_sigma: Vec<f64> = gamlss_sigma.iter().map(|&s| s.ln()).collect();

    // ---- compare the recovered surfaces on the grid ------------------------
    let rel_mu = relative_l2(&gam_mu, gamlss_mu);
    let rel_log_sigma = relative_l2(&gam_log_sigma, &gamlss_log_sigma);

    eprintln!(
        "gaussian multi-smooth location-scale vs gamlss NO(): n={n} grid={grid_n} \
         rel_l2(mu)={rel_mu:.5} rel_l2(log sigma)={rel_log_sigma:.5}"
    );

    // Both engines maximize the same penalized Gaussian location-scale joint
    // log-likelihood with two additive penalized thin-plate smooths per block,
    // so the recovered additive mean and log-sigma surfaces must coincide up to
    // numerical tolerance. The mean is the better-determined parameter
    // (variance-stabilized by the same 1/sigma^2 weights both engines use),
    // hence the tighter 0.02 bound — the same bar the single-smooth mgcv and
    // gamlss benchmarks hit. The log-sigma surface is a noisier second-moment
    // quantity AND uses a different noise link (gam's floored sigma = 0.01 +
    // exp(eta) vs gamlss NO()'s log link sigma = exp(eta)); since sigma_true in
    // [0.1, 0.25] keeps the 0.01 floor at most ~10% of sigma at its smallest,
    // that reparametrization adds a small pointwise bias on log sigma, so it
    // gets the looser 0.04 bound (matching the by-group gamlss benchmark). With
    // two smooths active per block, exceeding either bound flags a penalty-
    // block-alignment or blockwise-Jacobian bug invisible to a single-smooth fit.
    assert!(
        rel_mu < 0.02,
        "fitted mean surface diverges from gamlss: rel_l2(mu)={rel_mu:.5}"
    );
    assert!(
        rel_log_sigma < 0.04,
        "fitted log-sigma surface diverges from gamlss: rel_l2(log sigma)={rel_log_sigma:.5}"
    );
}
