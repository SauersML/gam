//! End-to-end quality: gam's Gaussian *location-scale* fit (a smooth mean
//! AND a smooth log-sigma, fit jointly by penalized blockwise PIRLS) must
//! match `gamlss::gamlss(family = NO())` — the reference GAMLSS implementation
//! for distributional regression and the de-facto standard for location-scale
//! models with both a smooth mean and a smooth log-variance in R.
//!
//! This is the cross-feature combination that single-parameter GAM tests never
//! exercise: family (Gaussian) x TWO smooths (mean + scale) fit jointly. We
//! synthesize a sin signal with a heteroscedastic noise envelope, feed the
//! *identical* (x, y) rows to both engines, and compare the recovered smooth
//! shapes — not predictions of held-out data — on a dense grid:
//!   * the fitted mean mu(x), and
//!   * the fitted log standard deviation log(sigma(x)).
//!
//! Both engines maximize the same Gaussian location-scale (penalized) joint
//! log-likelihood `ell = -1/2 sum (y-mu)^2 / sigma^2 - sum log sigma`, so the
//! recovered mean and log-sigma smooths should converge to nearly identical
//! shapes up to numerical-integration / basis-convention tolerance. A genuine
//! divergence here is a real bug in gam's blockwise location-scale solver.
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
use gam::{FitConfig, FitResult, encode_recordswith_inferred_schema, fit_from_formula, init_parallelism};
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
        state = state.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
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
    let y: Vec<f64> = (0..n).map(|i| mu_true(x[i]) + sigma_true(x[i]) * z[i]).collect();

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
    let result =
        fit_from_formula("y ~ s(x, bs='tp')", &ds, &cfg).expect("gam location-scale fit");
    let FitResult::GaussianLocationScale(GaussianLocationScaleFitResult { fit, .. }) = result else {
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

    // ---- evaluate gam's smooths on a dense grid at 100 points --------------
    // Span the EMPIRICAL data range [min(x), max(x)] rather than [0,1]: the
    // training x are Uniform(0,1) and never reach the open boundaries, so a
    // [0,1] grid would force both smoothers to *extrapolate* at the ends, where
    // thin-plate (gam) and P-spline (gamlss) bases disagree for reasons that
    // have nothing to do with gam's solver. Restricting to the interpolation
    // region makes the comparison test the fitted smooth, not boundary policy.
    let x_lo = x.first().copied().expect("x is non-empty");
    let x_hi = x.last().copied().expect("x is non-empty");
    let grid_n = 100usize;
    let grid_x: Vec<f64> = (0..grid_n)
        .map(|i| x_lo + (x_hi - x_lo) * (i as f64) / (grid_n as f64 - 1.0))
        .collect();
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
    // log-sigma via pb() penalized B-splines, predicted on the identical grid.
    let grid_csv = grid_x
        .iter()
        .map(|t| format!("{t:.17e}"))
        .collect::<Vec<_>>()
        .join(",");
    let body = format!(
        r#"
        suppressPackageStartupMessages(library(gamlss))
        m <- gamlss(y ~ pb(x), sigma.formula = ~ pb(x), family = NO(),
                    data = df, control = gamlss.control(trace = FALSE))
        gx <- as.numeric(strsplit("{grid_csv}", ",")[[1]])
        nd <- data.frame(x = gx)
        mu <- predict(m, what = "mu", newdata = nd, type = "response")
        sigma <- predict(m, what = "sigma", newdata = nd, type = "response")
        emit("mu", as.numeric(mu))
        emit("sigma", as.numeric(sigma))
        "#
    );
    let r = run_r(&[Column::new("x", &x), Column::new("y", &y)], &body);
    let gamlss_mu = r.vector("mu");
    let gamlss_sigma = r.vector("sigma");
    assert_eq!(gamlss_mu.len(), grid_n, "gamlss mu grid length mismatch");
    assert_eq!(gamlss_sigma.len(), grid_n, "gamlss sigma grid length mismatch");
    let gamlss_log_sigma: Vec<f64> = gamlss_sigma.iter().map(|&s| s.ln()).collect();

    // ---- compare the recovered smooth shapes on the grid -------------------
    // mu(x) is a genuine sine that crosses zero, so its grid vector's norm is
    // the signal itself (no large additive offset); relative L2 is the right,
    // scale-free element-wise agreement metric there.
    let rel_mu = relative_l2(&gam_mu, gamlss_mu);

    // log-sigma(x) sits at a large negative LEVEL (sigma ~ 0.01..0.3 here, so
    // log-sigma ~ -4.6..-1.2). A relative-L2 on log-sigma would be dominated by
    // that constant offset and would barely move with the smooth's *shape* — it
    // would assert almost nothing. The meaningful, well-conditioned metrics are
    // (a) RMSE in log units, which is exactly the multiplicative error in sigma
    // (a constant +d in log-sigma is a constant exp(d) factor in sigma), and
    // (b) the Pearson correlation of the recovered shape across the grid.
    let rmse_log_sigma = rmse(&gam_log_sigma, &gamlss_log_sigma);
    let corr_log_sigma = pearson(&gam_log_sigma, &gamlss_log_sigma);

    eprintln!(
        "gaussian location-scale vs gamlss NO(): n={n} grid={grid_n} \
         rel_l2(mu)={rel_mu:.5} rmse(log sigma)={rmse_log_sigma:.5} \
         pearson(log sigma)={corr_log_sigma:.5}"
    );

    // Both engines maximize the same Gaussian location-scale (penalized) joint
    // log-likelihood, so they must recover the same mean and log-sigma smooths.
    // But this is a CROSS-PACKAGE comparison: gam uses thin-plate smooths with
    // REML smoothing selection, gamlss `pb()` uses P-splines selected by its RS
    // local-ML criterion. Different bases + different lambda selectors mean the
    // two penalized fits agree to a few percent, not to mgcv-vs-gam precision,
    // so the bounds are deliberately looser than the same-engine smooth test.
    //
    // mu is the better-determined parameter (variance-stabilized by the shared
    // 1/sigma^2 weights), so 5% relative L2 is a tight-but-fair bar that would
    // still catch any real mean-block divergence.
    assert!(
        rel_mu < 0.05,
        "fitted mean smooth diverges from gamlss: rel_l2(mu)={rel_mu:.5}"
    );
    // log-sigma is a second-moment quantity estimated from n=200 squared
    // residuals across two different P-spline/thin-plate bases, so it is
    // genuinely noisier. We require the shapes to be strongly correlated
    // (pearson > 0.9 — both must trace the same heteroscedastic envelope) AND
    // the level to agree to within ~0.20 in log units, i.e. better than a ~22%
    // multiplicative discrepancy in the recovered sigma(x). Either bound being
    // exceeded is a real divergence of gam's blockwise solver from GAMLSS.
    assert!(
        corr_log_sigma > 0.9,
        "log-sigma smooth shape uncorrelated with gamlss: pearson={corr_log_sigma:.5}"
    );
    assert!(
        rmse_log_sigma < 0.20,
        "fitted log-sigma level diverges from gamlss: rmse(log sigma)={rmse_log_sigma:.5}"
    );
}
