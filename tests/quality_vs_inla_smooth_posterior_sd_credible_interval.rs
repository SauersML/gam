//! End-to-end quality: gam's *uncertainty quantification* — the pointwise
//! posterior standard deviation and 95% credible-interval width derived from its
//! Laplace covariance `V_b` (`covariance_conditional`) — must agree with
//! **R-INLA**, the mature approximate-Bayesian latent-Gaussian engine.
//!
//! This is the credible-interval counterpart to the mgcv point-estimate test.
//! gam stores `V_b = H⁻¹·φ̂` (the Bayesian/conditional coefficient covariance)
//! and turns it into a pointwise posterior SD on the linear predictor via
//! `sqrt(diag(D · V_b · Dᵀ))`, where `D` is the design matrix evaluated at the
//! training points. INLA, by contrast, integrates the latent field numerically
//! and reports the marginal posterior mean / SD / quantiles of the linear
//! predictor at every observation in `summary.linear.predictor`. Comparing the
//! two directly tests whether gam's Hessian-based Laplace approximation tracks a
//! fully marginalized posterior on the *same* additive model.
//!
//! Model (identical data to both engines):
//!   `price ~ s(h_rain) + s(s_temp)` — two univariate smooths, Gaussian/REML.
//!
//! INLA has no `bs="ps"`/`bs="tp"` basis; its mature analog for a smooth
//! function of a continuous covariate is a second-order random walk
//! (`f(x, model="rw2")`), the intrinsic-GMRF spline that INLA was built around
//! and the correct head-to-head for a penalized cubic/P-spline. Both engines
//! therefore fit a doubly-penalized (second-difference) smooth selected by an
//! empirical-Bayes criterion (REML for gam, the INLA hyperparameter posterior
//! mode for INLA), so close agreement on posterior SD is the right expectation
//! and a large gap is a real divergence in gam's uncertainty quantification.
//!
//! Data: `bench/datasets/wine.csv` (the Bordeaux vintage dataset). The spec's
//! nominal column names (`alcohol`/`fixed acidity`/`pH`) belong to a different
//! "wine" table; this repository's `wine.csv` carries vintage price and weather
//! covariates, so we use `price` as the Gaussian response and the two clean,
//! NA-free weather covariates `h_rain` (harvest rain) and `s_temp` (summer
//! temperature) as the two univariate smooth terms — exactly two smooths over
//! two continuous covariates, as the capability requires.

use gam::matrix::LinearOperator;
use gam::smooth::build_term_collection_design;
use gam::test_support::reference::{Column, relative_l2, rmse, run_r};
use gam::{FitConfig, FitResult, fit_from_formula, init_parallelism, load_csvwith_inferred_schema};
use ndarray::Array2;
use std::path::Path;

const WINE_CSV: &str = concat!(env!("CARGO_MANIFEST_DIR"), "/bench/datasets/wine.csv");

#[test]
fn gam_posterior_sd_and_credible_interval_match_inla() {
    init_parallelism();

    // ---- load the Bordeaux wine dataset -----------------------------------
    let ds = load_csvwith_inferred_schema(Path::new(WINE_CSV)).expect("load wine.csv");
    let col = ds.column_map();
    let price_idx = col["price"];
    let h_rain_idx = col["h_rain"];
    let s_temp_idx = col["s_temp"];
    let price: Vec<f64> = ds.values.column(price_idx).to_vec();
    let h_rain: Vec<f64> = ds.values.column(h_rain_idx).to_vec();
    let s_temp: Vec<f64> = ds.values.column(s_temp_idx).to_vec();
    let n = price.len();
    assert!(n > 40, "wine dataset should have ~47 rows, got {n}");
    // The three columns we use must be fully finite (no NA round-trips into
    // either engine); the unused `parker` column carries the NAs.
    for (name, v) in [("price", &price), ("h_rain", &h_rain), ("s_temp", &s_temp)] {
        assert!(
            v.iter().all(|x| x.is_finite()),
            "column {name} has non-finite entries; cannot feed identical data to both engines"
        );
    }

    // ---- fit with gam: price ~ s(h_rain) + s(s_temp), Gaussian/REML --------
    let cfg = FitConfig {
        family: Some("gaussian".to_string()),
        ..FitConfig::default()
    };
    let result = fit_from_formula(
        "price ~ s(h_rain, bs='ps', k=20) + s(s_temp, bs='tp', k=15)",
        &ds,
        &cfg,
    )
    .expect("gam fit");
    let FitResult::Standard(fit) = result else {
        panic!("expected a standard GAM fit");
    };

    // Vb = H⁻¹·φ̂ : gam's Bayesian/conditional coefficient covariance. This is
    // exactly the matrix the spec targets; pull it directly off the fit.
    let vb = fit
        .fit
        .covariance_conditional
        .as_ref()
        .expect("gam fit reports the conditional (Bayesian) covariance V_b");
    let beta = &fit.fit.beta;
    let p = beta.len();
    assert_eq!(vb.nrows(), p, "V_b row dim must match coefficient count");
    assert_eq!(vb.ncols(), p, "V_b col dim must match coefficient count");

    // Rebuild the (dense) design at the training points from the frozen spec.
    // Identity link ⇒ row i of D maps β to the fitted mean at observation i,
    // and the pointwise posterior variance is dᵢᵀ V_b dᵢ.
    let mut grid = Array2::<f64>::zeros((n, ds.headers.len()));
    for i in 0..n {
        grid[[i, h_rain_idx]] = h_rain[i];
        grid[[i, s_temp_idx]] = s_temp[i];
    }
    let design = build_term_collection_design(grid.view(), &fit.resolvedspec)
        .expect("rebuild design at training points");
    let d_dense = design.design.to_dense();
    assert_eq!(d_dense.nrows(), n, "design row count must match n");
    assert_eq!(d_dense.ncols(), p, "design col count must match coefficient count");

    // gam posterior mean, pointwise SD, and 95% CI width at each training point.
    let gam_mean: Vec<f64> = design.design.apply(beta).to_vec();
    let mut gam_sd = vec![0.0f64; n];
    let mut gam_ci_width = vec![0.0f64; n];
    let mut gam_lo = vec![0.0f64; n];
    let mut gam_hi = vec![0.0f64; n];
    // z for a two-sided 95% interval.
    const Z95: f64 = 1.959963984540054;
    for i in 0..n {
        let di = d_dense.row(i);
        // var_i = dᵢᵀ V_b dᵢ  (quadratic form against the full covariance).
        let mut var_i = 0.0;
        for a in 0..p {
            let dia = di[a];
            if dia == 0.0 {
                continue;
            }
            let mut acc = 0.0;
            for b in 0..p {
                acc += vb[[a, b]] * di[b];
            }
            var_i += dia * acc;
        }
        assert!(
            var_i.is_finite() && var_i >= 0.0,
            "posterior variance at point {i} is not a valid (finite, non-negative) value: {var_i}"
        );
        let sd = var_i.sqrt();
        gam_sd[i] = sd;
        gam_ci_width[i] = 2.0 * Z95 * sd;
        gam_lo[i] = gam_mean[i] - Z95 * sd;
        gam_hi[i] = gam_mean[i] + Z95 * sd;
    }

    // ---- fit the SAME model with R-INLA (the mature reference) ------------
    // Two independent rw2 (second-order random-walk) smooths — INLA's intrinsic
    // GMRF spline, the mature analog of a doubly-penalized P-spline / thin-plate
    // smooth. rw2 needs an integer location index per distinct covariate value,
    // so we pass the raw covariates and let R map them to ranked indices; INLA's
    // summary.linear.predictor then reports the marginal posterior of η at each
    // observation, grid-aligned row-for-row with gam's training points.
    let r = run_r(
        &[
            Column::new("price", &price),
            Column::new("h_rain", &h_rain),
            Column::new("s_temp", &s_temp),
        ],
        r#"
        suppressPackageStartupMessages(library(INLA))
        # rw2 requires an integer location column; map each covariate to the
        # rank of its (sorted, unique) value so equal covariate values share a
        # latent node, exactly as a smooth basis would tie them.
        df$ih <- match(df$h_rain, sort(unique(df$h_rain)))
        df$is <- match(df$s_temp, sort(unique(df$s_temp)))
        form <- price ~ f(ih, model = "rw2", scale.model = TRUE) +
                        f(is, model = "rw2", scale.model = TRUE)
        m <- inla(
            form,
            data = df,
            family = "gaussian",
            control.predictor = list(compute = TRUE),
            control.compute = list(config = TRUE)
        )
        lp <- m$summary.linear.predictor
        emit("mean", as.numeric(lp[["mean"]]))
        emit("sd", as.numeric(lp[["sd"]]))
        emit("lo", as.numeric(lp[["0.025quant"]]))
        emit("hi", as.numeric(lp[["0.975quant"]]))
        "#,
    );
    let inla_mean = r.vector("mean");
    let inla_sd = r.vector("sd");
    let inla_lo = r.vector("lo");
    let inla_hi = r.vector("hi");
    assert_eq!(inla_sd.len(), n, "INLA posterior-SD length must equal n");
    assert_eq!(inla_mean.len(), n, "INLA posterior-mean length must equal n");

    let inla_ci_width: Vec<f64> = inla_hi
        .iter()
        .zip(inla_lo.iter())
        .map(|(h, l)| h - l)
        .collect();

    // ---- metric 1: posterior SD agreement ---------------------------------
    let mean_inla_sd = inla_sd.iter().sum::<f64>() / inla_sd.len() as f64;
    let sd_rmse = rmse(&gam_sd, inla_sd);
    let sd_rmse_rel = sd_rmse / mean_inla_sd;

    // ---- metric 2: 95% credible-interval width agreement ------------------
    let ci_rel_l2 = relative_l2(&gam_ci_width, &inla_ci_width);

    // ---- metric 3: coverage of INLA posterior means by gam's intervals ----
    let mut covered = 0usize;
    for i in 0..n {
        if inla_mean[i] >= gam_lo[i] && inla_mean[i] <= gam_hi[i] {
            covered += 1;
        }
    }
    let coverage = covered as f64 / n as f64;

    eprintln!(
        "wine price ~ s(h_rain)+s(s_temp): n={n} p={p} \
         mean_inla_sd={mean_inla_sd:.4} sd_rmse={sd_rmse:.4} \
         sd_rmse_rel={sd_rmse_rel:.4} ci_rel_l2={ci_rel_l2:.4} coverage={coverage:.3}"
    );

    // ---- principled bounds ------------------------------------------------
    // gam's Laplace `V_b` and INLA's fully marginalized posterior solve the same
    // empirical-Bayes additive model with second-difference penalties; their
    // pointwise posterior SDs should track each other to well within the basis-
    // and hyperparameter-marginalization differences. The spec's bounds:
    //  (1) SD RMSE under 12% of the mean INLA SD,
    //  (2) CI-width relative-L2 under 8%,
    //  (3) at least 85% of INLA's posterior means inside gam's 95% intervals
    //      (relaxed to absorb Laplace-vs-numerical-integration drift).
    assert!(
        sd_rmse_rel < 0.12,
        "gam posterior SD diverges from INLA: rmse={sd_rmse:.4} is {:.1}% of mean INLA SD {mean_inla_sd:.4}",
        100.0 * sd_rmse_rel
    );
    assert!(
        ci_rel_l2 < 0.08,
        "gam 95% credible-interval width diverges from INLA: relative_l2={ci_rel_l2:.4}"
    );
    assert!(
        coverage >= 0.85,
        "gam 95% credible intervals cover too few INLA posterior means: coverage={coverage:.3}"
    );
}
