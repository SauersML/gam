//! End-to-end quality: gam's Gaussian *location-scale* fit (a smooth mean
//! AND a smooth log-sigma, fit jointly by penalized blockwise PIRLS) must
//! RECOVER THE KNOWN GENERATING FUNCTIONS of a heteroscedastic synthetic
//! dataset, and must predict the joint (mean, scale) density at least as well
//! as the mature boosted-GAMLSS engine `gamboostLSS` on the IDENTICAL rows.
//!
//! Capability under test: family (Gaussian) x TWO smooths (mean + scale) fit
//! jointly — gam's gaulss-style distributional regression.
//!
//! COMPARATOR (mature, match-or-beat baseline — NOT truth): R `gamboostLSS`
//! (component-wise boosting for GAMLSS, Mayr/Fenske/Hofner/Schmid/Greven), with
//! `mboost` P-spline base-learners `bbs(x)` for both the `mu` and `sigma`
//! sub-models and the Gaussian location-scale family `GaussianLSS()`. It is fit
//! on byte-identical (x, y) and used ONLY as a baseline to match-or-beat on a
//! predictive score that rewards correct scale estimation (mean Gaussian NLL).
//! "We reproduce gamboostLSS's fitted output" is explicitly NOT the claim.
//!
//! OBJECTIVE metrics asserted:
//!   PRIMARY (truth recovery, tool-free): the data are drawn from a KNOWN mean
//!     `mu_true(x) = sin(2*pi*x)` and a KNOWN, genuinely varying noise sd
//!     `sigma_true(x) = exp(-0.5 + 0.8*x)`. gam's recovered mean smooth must sit
//!     below the average noise level, and gam's recovered sigma smooth must
//!     track the true heteroscedastic envelope (strong Pearson correlation in
//!     log-sigma, small log-sigma RMSE).
//!   MATCH-OR-BEAT (predictive): the mean per-observation Gaussian negative
//!     log-likelihood — `0.5*log(2*pi*sigma^2) + (y-mu)^2/(2*sigma^2)`, the
//!     score that rewards getting BOTH moments right — for gam must be no worse
//!     than gamboostLSS's by more than a small slack.
//!
//! gam-side API (copied verbatim from the sibling
//! `quality_vs_gamlss_gaussian_location_scale.rs`, which compiles):
//!   * `fit_from_formula(.., FitConfig{ noise_formula: Some(..), .. })` routes
//!     through `materialize_location_scale` -> `FitResult::GaussianLocationScale`;
//!     the in-Rust path does NOT rescale `y`, so reconstructed mu/sigma are in
//!     raw response units.
//!   * the location block carries `BlockRole::Location`, the log-sigma block
//!     `BlockRole::Scale`; resolved designs live in `fit.meanspec_resolved` /
//!     `fit.noisespec_resolved`.
//!   * the noise link is `sigma = LOGB_SIGMA_FLOOR + exp(eta_scale)` with
//!     `LOGB_SIGMA_FLOOR = 0.01` (mirrors `families::sigma_link`, mgcv `gaulss(b=0.01)`).

use gam::estimate::BlockRole;
use gam::gamlss::GaussianLocationScaleFitResult;
use gam::matrix::LinearOperator;
use gam::smooth::build_term_collection_design;
use gam::test_support::reference::{Column, QualityPair, pearson, rmse, run_r};
use gam::{
    FitConfig, FitResult, encode_recordswith_inferred_schema, fit_from_formula, init_parallelism,
};
use ndarray::Array2;

/// gam's location-scale noise link floor: sigma = 0.01 + exp(eta_scale).
/// Mirrors `families::sigma_link::LOGB_SIGMA_FLOOR` (and mgcv `gaulss(b=0.01)`).
const LOGB_SIGMA_FLOOR: f64 = 0.01;

/// Mean per-observation Gaussian negative log-likelihood. The natural objective
/// score for a *location-scale* fit because it rewards BOTH the predicted mean
/// AND the predicted sigma at each point:
///   0.5*log(2*pi) + log(sigma_i) + 0.5*((y_i - mu_i)/sigma_i)^2.
fn mean_gaussian_nll(mu: &[f64], sigma: &[f64], y: &[f64]) -> f64 {
    assert_eq!(mu.len(), sigma.len(), "nll mu/sigma length mismatch");
    assert_eq!(mu.len(), y.len(), "nll mu/y length mismatch");
    let half_log_2pi = 0.5 * (2.0 * std::f64::consts::PI).ln();
    let n = mu.len() as f64;
    let s: f64 = (0..mu.len())
        .map(|i| {
            let sd = sigma[i].max(1e-12);
            let z = (y[i] - mu[i]) / sd;
            half_log_2pi + sd.ln() + 0.5 * z * z
        })
        .sum();
    s / n
}

#[test]
fn gam_gaussian_location_scale_matches_gamboostlss() {
    init_parallelism();

    // ---- synthetic heteroscedastic recipe (fed IDENTICALLY to both engines) ----
    // n=400, x ~ Uniform(0,1), sigma(x) = exp(-0.5 + 0.8*x) (a smoothly varying,
    // strictly-positive noise sd that genuinely changes scale across x: from
    // ~0.61 at x=0 to ~1.35 at x=1), y ~ N(sin(2*pi*x), sigma(x)^2), seed=42. A
    // deterministic seeded LCG draws the uniforms and the Box-Muller normals so
    // the exact same y is reproducible in pure Rust and sent verbatim to R.
    let n = 400usize;
    let two_pi = 2.0 * std::f64::consts::PI;

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
    // Strictly-positive, smoothly varying scale: log sigma is linear in x, so
    // log|sigma_true| has NO cusps (unlike the sibling 0.1+0.2*sin recipe) and
    // the achievable log-sigma correlation ceiling is high.
    let sigma_true = |t: f64| (-0.5 + 0.8 * t).exp();
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
    // We score both engines at the exact training abscissae (Uniform(0,1), so
    // strictly inside the interpolation region — no basis extrapolates) and read
    // gamboostLSS's recovered smooths off its FITTED values on the same rows.
    let grid_n = n;
    let grid_x: Vec<f64> = x.clone();
    let mut grid = Array2::<f64>::zeros((grid_n, ncols));
    for (i, &t) in grid_x.iter().enumerate() {
        grid[[i, x_idx]] = t;
    }

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

    // ---- fit the SAME model with gamboostLSS (the mature boosted-GAMLSS ref) ----
    // Component-wise boosting with mboost P-spline base-learners bbs(x) for both
    // the mu and sigma sub-models, family GaussianLSS() (mu link identity,
    // sigma link log). We read the recovered smooths off the FITTED values on the
    // training data: fitted(m, parameter="mu") returns mu(x_i) on the response
    // scale and fitted(m, parameter="sigma") returns sigma(x_i) directly (the log
    // link is undone for us). The rows of df are in the SAME order gam sees them,
    // so fitted[i] aligns with x[i]. nu = 0.1 is the standard small boosting step
    // length; mstop is chosen large enough for both sub-models to converge on this
    // smooth low-dimensional problem.
    let body = r#"
        suppressPackageStartupMessages(library(gamboostLSS))
        suppressPackageStartupMessages(library(mboost))
        m <- gamboostLSS(y ~ bbs(x, df = 4), data = df, families = GaussianLSS(),
                         control = boost_control(mstop = c(mu = 600, sigma = 600),
                                                 nu = 0.1))
        mu <- as.numeric(fitted(m, parameter = "mu"))
        sigma <- as.numeric(fitted(m, parameter = "sigma"))
        emit("mu", mu)
        emit("sigma", sigma)
        "#
    .to_string();
    let r = run_r(&[Column::new("x", &x), Column::new("y", &y)], &body);
    let boost_mu = r.vector("mu");
    let boost_sigma = r.vector("sigma");
    assert_eq!(
        boost_mu.len(),
        grid_n,
        "gamboostLSS mu grid length mismatch"
    );
    assert_eq!(
        boost_sigma.len(),
        grid_n,
        "gamboostLSS sigma grid length mismatch"
    );
    let boost_log_sigma: Vec<f64> = boost_sigma.iter().map(|&s| s.ln()).collect();

    // ---- TRUTH on the SAME grid (the known generating functions) -----------
    let true_mu: Vec<f64> = grid_x.iter().map(|&t| mu_true(t)).collect();
    let true_log_sigma: Vec<f64> = grid_x.iter().map(|&t| sigma_true(t).ln()).collect();

    // ---- PRIMARY objective metric: recovery of the KNOWN functions ---------
    let mean_noise_level = {
        let s: f64 = grid_x.iter().map(|&t| sigma_true(t)).sum();
        s / grid_n as f64
    };
    let gam_rmse_mu = rmse(&gam_mu, &true_mu);
    let gam_rmse_log_sigma = rmse(&gam_log_sigma, &true_log_sigma);
    let gam_corr_log_sigma = pearson(&gam_log_sigma, &true_log_sigma);

    // ---- predictive (joint) score: mean Gaussian NLL of each engine --------
    let gam_nll = mean_gaussian_nll(&gam_mu, &gam_sigma, &y);
    let boost_nll = mean_gaussian_nll(boost_mu, boost_sigma, &y);

    // gamboostLSS as a BASELINE-TO-MATCH-OR-BEAT on the SAME truth (context).
    let boost_rmse_mu = rmse(boost_mu, &true_mu);
    let boost_corr_log_sigma = pearson(&boost_log_sigma, &true_log_sigma);

    eprintln!(
        "[gamboostLSS-LS] n={n} grid={grid_n} mean_noise_level={mean_noise_level:.4} \
         | gam: rmse(mu->truth)={gam_rmse_mu:.5} rmse(log sigma->truth)={gam_rmse_log_sigma:.5} \
         pearson(log sigma,truth)={gam_corr_log_sigma:.5} nll={gam_nll:.5} \
         | gamboostLSS: rmse(mu->truth)={boost_rmse_mu:.5} \
         pearson(log sigma,truth)={boost_corr_log_sigma:.5} nll={boost_nll:.5}"
    );
    eprintln!(
        "{}",
        QualityPair::error(
            "families",
            "quality_vs_gamboostlss_gaussian_location_scale::mu",
            "mu_rmse_to_truth",
            gam_rmse_mu,
            "gamboostlss",
            boost_rmse_mu,
        )
        .line()
    );

    // PRIMARY claim #1: gam recovers the TRUE mean. The mean is variance-
    // stabilized by the shared 1/sigma^2 weights and is the better-determined
    // parameter; its reconstruction error must sit comfortably below the average
    // noise standard deviation it is averaging through.
    assert!(
        gam_rmse_mu < 0.5 * mean_noise_level,
        "gam mean smooth does not recover the truth: rmse(mu->truth)={gam_rmse_mu:.5} \
         (bar = 0.5*mean_noise_level = {:.5})",
        0.5 * mean_noise_level
    );

    // PRIMARY claim #2: gam recovers the TRUE log-sigma envelope. With a smooth,
    // cusp-free log-sigma truth (linear in x), a correctly fit scale smooth must
    // be strongly correlated with the true heteroscedastic envelope. The
    // over-smoothed-to-nullspace failure mode (#686: scale shrunk to its penalty
    // null space) would NOT clear this floor.
    assert!(
        gam_corr_log_sigma > 0.70,
        "gam log-sigma smooth does not trace the true envelope: \
         pearson(log sigma, truth)={gam_corr_log_sigma:.5} (floor 0.70)"
    );

    // PRIMARY claim #3: the recovered log-sigma LEVEL is close. Linear-in-x log
    // sigma is cusp-free, so the level error is genuinely small (no -inf spikes).
    assert!(
        gam_rmse_log_sigma < 0.40,
        "gam log-sigma smooth does not recover the true level: \
         rmse(log sigma->truth)={gam_rmse_log_sigma:.5} (bound 0.40)"
    );

    // MATCH-OR-BEAT claim: on the predictive score that rewards getting BOTH
    // moments right (mean Gaussian NLL), gam must be no worse than the mature
    // gamboostLSS engine by more than a small slack. Matching its fitted output
    // would prove nothing; predicting the joint density at least as well does.
    assert!(
        gam_nll <= boost_nll + 0.05,
        "gam joint location-scale NLL worse than gamboostLSS by >0.05: \
         gam={gam_nll:.5} gamboostLSS={boost_nll:.5}"
    );
}
