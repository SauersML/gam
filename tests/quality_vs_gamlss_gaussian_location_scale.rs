//! End-to-end quality: gam's Gaussian *location-scale* fit (a smooth mean
//! AND a smooth log-sigma, fit jointly by penalized blockwise PIRLS) must
//! RECOVER THE KNOWN GENERATING FUNCTIONS of a heteroscedastic synthetic
//! dataset. This is the cross-feature combination single-parameter GAM tests
//! never exercise: family (Gaussian) x TWO smooths (mean + scale) fit jointly.
//!
//! OBJECTIVE METRIC (truth recovery, NOT closeness to a reference tool):
//!   * the data are drawn from a KNOWN mean mu_true(x) = sin(2*pi*x) and a
//!     KNOWN noise standard deviation s_true(x) = |0.1 + 0.2*sin(2*pi*x)|;
//!   * the PRIMARY pass/fail assertions are that gam's recovered mean smooth
//!     and recovered log-sigma smooth track those TRUE functions:
//!       - RMSE(gam_mu, mu_true) <= a fraction of the mean noise level, and
//!       - RMSE(gam_log_sigma, log s_true) <= a small absolute bar in log units
//!         (a constant +d in log-sigma is an exp(d) multiplicative factor in
//!         sigma), with strong Pearson correlation against the true envelope.
//!
//! `gamlss::gamlss(family = NO())` — the de-facto standard GAMLSS engine for
//! distributional regression in R — is fit on the IDENTICAL rows and used only
//! as a BASELINE-TO-MATCH-OR-BEAT on the same truth-recovery error: gam's error
//! must not exceed gamlss's by more than 10%. "We reproduce gamlss's fitted
//! output" is explicitly NOT the claim; recovering the truth at least as
//! accurately as the mature tool is.
//!
//! Notes on the gam side that this test pins down by reading the source:
//!   * `fit_from_formula(..., FitConfig{ noise_formula: Some(...), .. })` routes
//!     through `materialize_location_scale` -> `FitRequest::GaussianLocationScale`.
//!     Unlike the CLI in `main.rs`, this in-Rust path does NOT rescale `y` by its
//!     sample std, so the fitted coefficients and reconstructed mu / sigma are
//!     already in raw response units.
//!   * gam's noise (sigma) link is `sigma = LOGB_SIGMA_FLOOR + exp(eta_scale)`
//!     with `LOGB_SIGMA_FLOOR = 0.01` (see `families::sigma_link`), the same
//!     soft floor mgcv's `gaulss(b=0.01)` uses; the location block carries role
//!     `BlockRole::Location`, the log-sigma block role `BlockRole::Scale`.
//!   * The spec's `linkwiggle(...)` term is a *binomial-only* link correction
//!     (`reject_explicit_linkwiggle_for_nonbinomial` rejects it for a Gaussian
//!     response); it is meaningless for a Gaussian location-scale fit, so the
//!     gam formula is the smooth-mean / smooth-log-sigma pair without it.

use gam::estimate::BlockRole;
use gam::gamlss::GaussianLocationScaleFitResult;
use gam::matrix::LinearOperator;
use gam::smooth::build_term_collection_design;
use gam::test_support::reference::{Column, pearson, relative_l2, rmse, run_r};
use gam::{
    FitConfig, FitResult, encode_recordswith_inferred_schema, fit_from_formula, init_parallelism,
};
use ndarray::Array2;

/// gam's location-scale noise link floor: sigma = 0.01 + exp(eta_scale).
/// Mirrors `families::sigma_link::LOGB_SIGMA_FLOOR` (and mgcv `gaulss(b=0.01)`).
const LOGB_SIGMA_FLOOR: f64 = 0.01;

#[test]
fn gam_gaussian_location_scale_matches_gamlss() {
    init_parallelism();

    // ---- synthetic heteroscedastic recipe (fed IDENTICALLY to both engines) ----
    // n=200, x ~ Uniform(0,1), sigma(x) = 0.1 + 0.2*sin(2*pi*x),
    // y ~ N(sin(2*pi*x), sigma(x)^2), seed=42. A deterministic seeded LCG draws
    // the standard normals so the exact same y is reproducible in pure Rust and
    // sent verbatim to gamlss. (sigma(x) can dip negative for some x; as the
    // multiplier of a standard normal its sign is irrelevant to the draw, and
    // both engines see the same y, which is all that matters for agreement.)
    let n = 200usize;
    let two_pi = 2.0 * std::f64::consts::PI;

    // Sorted, evenly spread x in (0,1) via a fixed van der Corput-like seed-42
    // LCG, then sort, so the design is identical across runs and engines.
    let mut state: u64 = 42;
    let mut next_unit = || -> f64 {
        // Numerical Recipes LCG; take the high bits for a uniform in [0,1).
        state = state
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        ((state >> 11) as f64) / ((1u64 << 53) as f64)
    };
    let mut x: Vec<f64> = (0..n).map(|_| next_unit()).collect();
    x.sort_by(|a, b| a.partial_cmp(b).unwrap());

    // Box-Muller standard normals from the same LCG stream (seed continues).
    let mut z: Vec<f64> = Vec::with_capacity(n);
    while z.len() < n {
        let u1 = next_unit().max(1e-300);
        let u2 = next_unit();
        let r = (-2.0 * u1.ln()).sqrt();
        z.push(r * (two_pi * u2).cos());
        if z.len() < n {
            z.push(r * (two_pi * u2).sin());
        }
    }

    let mu_true = |t: f64| (two_pi * t).sin();
    let sigma_true = |t: f64| 0.1 + 0.2 * (two_pi * t).sin();
    let y: Vec<f64> = (0..n)
        .map(|i| mu_true(x[i]) + sigma_true(x[i]) * z[i])
        .collect();

    // ---- build the dataset (column 0 = x, column 1 = y) --------------------
    let headers: Vec<String> = vec!["x".to_string(), "y".to_string()];
    let rows: Vec<csv::StringRecord> = (0..n)
        .map(|i| csv::StringRecord::from(vec![format!("{:.17e}", x[i]), format!("{:.17e}", y[i])]))
        .collect();
    let ds = encode_recordswith_inferred_schema(headers, rows).expect("encode location-scale data");
    let col = ds.column_map();
    let x_idx = col["x"];
    let ncols = ds.headers.len();

    // ---- fit with gam: mu ~ s(x, bs='tp'), log-sigma ~ 1 + s(x, bs='tp') ----
    let cfg = FitConfig {
        family: Some("gaussian".to_string()),
        noise_formula: Some("1 + s(x, bs='tp')".to_string()),
        ..FitConfig::default()
    };
    let result = fit_from_formula("y ~ s(x, bs='tp')", &ds, &cfg).expect("gam location-scale fit");
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

    // ---- evaluate gam's smooths at the TRAINING x (the n fitted points) ----
    // We compare the two engines at the exact training abscissae rather than a
    // synthetic dense grid. This (a) keeps the comparison strictly inside the
    // interpolation region (the training x are Uniform(0,1) and never reach the
    // open boundaries, so neither thin-plate nor P-spline basis extrapolates),
    // and (b) lets us read gamlss's recovered smooths off `fitted(m, "mu")` /
    // `fitted(m, "sigma")` — its fitted values ON the training data — instead of
    // `predict(newdata=)`, whose smoother-refit path in `predict.gamlss` is
    // fragile and was erroring here. Both engines are thus scored on the same
    // x, with each engine's own native evaluation of its fitted smooth.
    let grid_n = n;
    let grid_x: Vec<f64> = x.clone();
    let mut grid = Array2::<f64>::zeros((grid_n, ncols));
    for (i, &t) in grid_x.iter().enumerate() {
        grid[[i, x_idx]] = t;
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
    // family = NO() (Gaussian with mu + log-sigma), smooth mean and smooth
    // log-sigma via pb() penalized B-splines. We read the recovered smooths off
    // the FITTED VALUES on the training data — `fitted(m, "mu")` returns mu(x_i)
    // on the response scale and `fitted(m, "sigma")` returns sigma(x_i) directly
    // (the identity-then-log link inside NO() is undone for us) — instead of
    // `predict(m, newdata=)`, whose smoother-refit machinery in `predict.gamlss`
    // is fragile (it dereferences the model's `data` slot by name and errored
    // with "object of type 'closure' is not subsettable" here). The rows of df
    // are in the SAME order gam sees them, so fitted[i] aligns with x[i].
    let body = r#"
        suppressPackageStartupMessages(library(gamlss))
        m <- gamlss(y ~ pb(x), sigma.formula = ~ pb(x), family = NO(),
                    data = df, control = gamlss.control(trace = FALSE))
        mu <- fitted(m, "mu")
        sigma <- fitted(m, "sigma")
        emit("mu", as.numeric(mu))
        emit("sigma", as.numeric(sigma))
        "#
    .to_string();
    let r = run_r(&[Column::new("x", &x), Column::new("y", &y)], &body);
    let gamlss_mu = r.vector("mu");
    let gamlss_sigma = r.vector("sigma");
    assert_eq!(gamlss_mu.len(), grid_n, "gamlss mu grid length mismatch");
    assert_eq!(
        gamlss_sigma.len(),
        grid_n,
        "gamlss sigma grid length mismatch"
    );
    let gamlss_log_sigma: Vec<f64> = gamlss_sigma.iter().map(|&s| s.ln()).collect();

    // ---- TRUTH on the SAME grid (the known generating functions) -----------
    // The data were drawn as y_i = mu_true(x_i) + s_true(x_i) * z_i with z_i
    // standard normal. The recoverable scale function is the standard deviation
    // of that noise, |s_true(x)| (the multiplier's sign is invisible to a
    // symmetric standard normal), so the truth for the log-sigma smooth is
    // log|s_true(x)|. These are the ground-truth targets both engines aim at.
    let true_mu: Vec<f64> = grid_x.iter().map(|&t| mu_true(t)).collect();
    let true_log_sigma: Vec<f64> = grid_x.iter().map(|&t| sigma_true(t).abs().ln()).collect();

    // ---- PRIMARY objective metric: recovery of the KNOWN functions ---------
    let mean_noise_level = {
        // Average true noise sd over the grid; the natural scale for the mean
        // smooth's reconstruction error (a fit cannot reasonably beat the noise
        // floor it is averaging over).
        let s: f64 = grid_x.iter().map(|&t| sigma_true(t).abs()).sum();
        s / grid_n as f64
    };
    let gam_rmse_mu = rmse(&gam_mu, &true_mu);
    let gam_rmse_log_sigma = rmse(&gam_log_sigma, &true_log_sigma);
    let gam_corr_log_sigma = pearson(&gam_log_sigma, &true_log_sigma);

    // ---- gamlss as a BASELINE-TO-MATCH-OR-BEAT on the SAME truth ------------
    let gamlss_rmse_mu = rmse(gamlss_mu, &true_mu);
    let gamlss_rmse_log_sigma = rmse(&gamlss_log_sigma, &true_log_sigma);

    // Context only: how close the two fitted outputs happen to be. NOT asserted.
    let rel_mu_vs_gamlss = relative_l2(&gam_mu, gamlss_mu);

    eprintln!(
        "gaussian location-scale truth recovery: n={n} grid={grid_n} \
         mean_noise_level={mean_noise_level:.4} \
         | gam: rmse(mu->truth)={gam_rmse_mu:.5} rmse(log sigma->truth)={gam_rmse_log_sigma:.5} \
         pearson(log sigma,truth)={gam_corr_log_sigma:.5} \
         | gamlss: rmse(mu->truth)={gamlss_rmse_mu:.5} rmse(log sigma->truth)={gamlss_rmse_log_sigma:.5} \
         | (context) rel_l2(gam mu, gamlss mu)={rel_mu_vs_gamlss:.5}"
    );

    // PRIMARY claim #1: gam recovers the TRUE mean. The mean is variance-
    // stabilized by the shared 1/sigma^2 weights and is the better-determined
    // parameter; its reconstruction error must sit comfortably below the mean
    // noise standard deviation it is averaging through.
    assert!(
        gam_rmse_mu < 0.5 * mean_noise_level,
        "gam mean smooth does not recover the truth: rmse(mu->truth)={gam_rmse_mu:.5} \
         (bar = 0.5*mean_noise_level = {:.5})",
        0.5 * mean_noise_level
    );

    // PRIMARY claim #2: gam recovers the TRUE log-sigma envelope. log-sigma is a
    // second-moment quantity from n=200 squared residuals, so it is genuinely
    // noisier; we require the recovered shape to be strongly correlated with the
    // true heteroscedastic envelope AND the level to land within ~0.30 in log
    // units of the truth (better than a ~35% multiplicative error in sigma(x)).
    assert!(
        gam_corr_log_sigma > 0.85,
        "gam log-sigma smooth does not trace the true envelope: \
         pearson(log sigma, truth)={gam_corr_log_sigma:.5}"
    );
    assert!(
        gam_rmse_log_sigma < 0.30,
        "gam log-sigma smooth does not recover the true level: \
         rmse(log sigma->truth)={gam_rmse_log_sigma:.5}"
    );

    // BASELINE claim: gam must recover the truth AT LEAST AS WELL as the mature
    // GAMLSS engine (matching the noisy fitted output of gamlss would prove
    // nothing — beating it on TRUTH-RECOVERY error does). Allow a 10% slack so
    // basis/lambda-selector differences alone never flip the test.
    assert!(
        gam_rmse_mu <= 1.10 * gamlss_rmse_mu,
        "gam mean recovery worse than gamlss: gam={gam_rmse_mu:.5} > 1.10*gamlss={:.5}",
        1.10 * gamlss_rmse_mu
    );
    assert!(
        gam_rmse_log_sigma <= 1.10 * gamlss_rmse_log_sigma,
        "gam log-sigma recovery worse than gamlss: gam={gam_rmse_log_sigma:.5} \
         > 1.10*gamlss={:.5}",
        1.10 * gamlss_rmse_log_sigma
    );
}
