//! End-to-end quality: gam's REML p-spline smooth (`bs='ps', m=2`) vs **R-INLA**
//! — the mature, best-in-class approximate-Bayesian latent-Gaussian engine — on
//! its own home turf: a *structured second-order-difference prior*.
//!
//! ## Why INLA, and why this is a distinct comparison
//!
//! mgcv tests in this suite check gam's p-spline against a *frequentist* REML
//! engine that targets the identical penalized objective. INLA is a different
//! animal: it does approximate fully-Bayesian inference on a **latent Gaussian
//! field** with a *known* precision structure. Its `f(x, model="rw2")` puts a
//! second-order random-walk (RW2) prior on the ordered latent values — exactly a
//! discrete second-order difference penalty, `tau * sum (Δ² u)²`. gam's
//! `s(x, bs='ps', m=2)` builds a B-spline basis with a second-order difference
//! penalty `lambda * βᵀSβ`. Both are the *same structured prior*; the engines
//! differ in how they (a) select the hyperparameter (gam: REML / marginal
//! likelihood over `lambda`; INLA: marginal posterior mode of `log tau` from its
//! Laplace evidence) and (b) propagate uncertainty (gam: Bayesian `Vb`; INLA:
//! marginal posterior SD). Agreement here means the two engines' *evidence*
//! calculations and *posterior uncertainty* coincide — the canonical
//! latent-Gaussian cross-check, not available against mgcv.
//!
//! ## Data
//!
//! The SPEC nominated a bone-mineral-density `bone.csv (n=459, age->spnbmd)`.
//! That ElemStatLearn dataset is **not** present in this repo — the shipped
//! `bench/datasets/bone.csv` is an unrelated n=23 bone-marrow-transplant survival
//! table (columns `t,d,trt`), which has no continuous covariate->continuous
//! response structure and cannot exercise an RW2/p-spline smooth. Rather than
//! fabricate the missing columns, this test uses the canonical real univariate
//! smoothing benchmark already wired into the suite, `lidar` (`range->logratio`,
//! n=221). It is fed **identically** to gam and to INLA, which is what the
//! comparison requires; the substitution is documented here as the honest
//! finding (the spec's nominal dataset is absent).
//!
//! ## What is asserted (all three SPEC metrics)
//!
//!   1. **Fitted values**: `relative_l2(gam_fitted, inla_fitted) < 0.08` at the
//!      training points. Looser than the mgcv p-spline bound because INLA fits a
//!      *discrete* RW2 on grouped locations (not a continuous B-spline) and
//!      integrates over `tau`, so the posterior-mean curve is a genuinely
//!      different estimator — 8% is tight enough that a real structural
//!      divergence fails, loose enough to admit the two priors' legitimate gap.
//!   2. **Penalty (hyperparameter) equivalence**: after the σ² gauge that maps
//!      INLA's precision `tau` to gam's penalty weight `lambda` (`lambda ≈
//!      tau * σ²` for the identity-Gaussian model — both standardize the
//!      structure matrix, gam via REML scale, INLA via `scale.model=TRUE`),
//!      `|log lambda_gam − log lambda_inla| / |log lambda_gam| < 0.3`. This is
//!      the heart of the test: do the two evidence calculations select the same
//!      smoothing level on the log scale.
//!   3. **Posterior SD**: `rmse(gam_sd, inla_sd) < 0.1 * mean(inla_sd)` at the
//!      training points, where gam's SD is `sqrt(diag(X Vb Xᵀ))` from the
//!      Bayesian covariance and INLA's is the fitted-value marginal SD.
//!
//! A failing assertion because gam genuinely diverges from INLA is a real
//! finding, not a reason to weaken the bound.

use gam::matrix::LinearOperator;
use gam::smooth::build_term_collection_design;
use gam::test_support::reference::{Column, relative_l2, rmse, run_r};
use gam::{FitConfig, FitResult, fit_from_formula, init_parallelism, load_csvwith_inferred_schema};
use ndarray::{Array1, Array2};
use std::path::Path;

const LIDAR_CSV: &str = concat!(env!("CARGO_MANIFEST_DIR"), "/bench/datasets/lidar.csv");

#[test]
fn gam_rw2_pspline_matches_inla_latent_gaussian() {
    init_parallelism();

    // ---- load the canonical lidar dataset (range -> logratio) -------------
    let ds = load_csvwith_inferred_schema(Path::new(LIDAR_CSV)).expect("load lidar.csv");
    let col = ds.column_map();
    let range_idx = col["range"];
    let logratio_idx = col["logratio"];
    let range: Vec<f64> = ds.values.column(range_idx).to_vec();
    let logratio: Vec<f64> = ds.values.column(logratio_idx).to_vec();
    let n = range.len();
    assert!(n > 100, "lidar should have ~221 rows, got {n}");

    // ---- fit with gam: logratio ~ s(range, bs='ps', penalty_order=2), REML -
    // bs='ps' with a SECOND-order difference penalty (penalty_order=2 is gam's
    // spelling of mgcv's `m=2`; it is also the bspline-builder default, set
    // explicitly here to make the RW2 intent unambiguous) -> the discrete analog
    // of INLA's rw2 latent prior. REML selects lambda by marginal likelihood,
    // the frequentist twin of INLA's evidence.
    let cfg = FitConfig {
        family: Some("gaussian".to_string()),
        ..FitConfig::default()
    };
    let result = fit_from_formula("logratio ~ s(range, bs='ps', penalty_order=2)", &ds, &cfg)
        .expect("gam RW2 p-spline fit");
    let FitResult::Standard(fit) = result else {
        panic!("expected a standard GAM fit for a gaussian RW2 p-spline smooth");
    };

    // The single smooth block's penalty weight on the log scale. (Intercept is
    // unpenalized, so the lone penalty entry is the s(range) smoothing param.)
    let log_lambdas = &fit.fit.log_lambdas;
    assert_eq!(
        log_lambdas.len(),
        1,
        "expected exactly one penalized smooth block, got {}",
        log_lambdas.len()
    );
    let gam_log_lambda = log_lambdas[0];
    // gam's residual scale sigma (identity Gaussian stores residual SD here).
    let gam_sigma = fit.fit.standard_deviation;
    let gam_sigma2 = gam_sigma * gam_sigma;

    // ---- gam fitted values + posterior SD at the training points ----------
    // Rebuild the (dense) design from the frozen spec at the observed `range`;
    // for the identity link the fitted mean is X*beta and the Bayesian
    // covariance of the fitted values is X Vb Xᵀ.
    let mut grid = Array2::<f64>::zeros((n, ds.headers.len()));
    for (i, &r) in range.iter().enumerate() {
        grid[[i, range_idx]] = r;
    }
    let design = build_term_collection_design(grid.view(), &fit.resolvedspec)
        .expect("rebuild design at training points");
    let gam_fitted: Vec<f64> = design.design.apply(&fit.fit.beta).to_vec();

    let xdense: Array2<f64> = design.design.to_dense();
    assert_eq!(xdense.nrows(), n, "dense design row count mismatch");
    let vb = fit
        .fit
        .covariance_conditional
        .as_ref()
        .expect("standard Gaussian fit reports the Bayesian covariance Vb");
    assert_eq!(
        vb.nrows(),
        xdense.ncols(),
        "Vb dimension must match the design column count"
    );
    // sd_i = sqrt( x_iᵀ Vb x_i ),  x_i = i-th row of X.
    let mut gam_sd: Vec<f64> = Vec::with_capacity(n);
    for i in 0..n {
        let xi: Array1<f64> = xdense.row(i).to_owned();
        let vbxi: Array1<f64> = vb.dot(&xi);
        let var = xi.dot(&vbxi);
        gam_sd.push(var.max(0.0).sqrt());
    }

    // ---- fit the SAME data with R-INLA: rw2 latent-Gaussian model ---------
    // f(idx, model="rw2", scale.model=TRUE): a second-order random walk on the
    // ordered, grouped covariate locations. inla.group bins `range` onto the
    // ordered lattice the rw2 prior requires; scale.model=TRUE standardizes the
    // structure matrix (matching gam's REML-scaled penalty). We read back: the
    // posterior-mean fitted values, their marginal SDs, the posterior mode of
    // log(tau) for the latent field, and the Gaussian observation precision so
    // we can recover sigma^2 for the lambda<->tau gauge.
    let r = run_r(
        &[
            Column::new("range", &range),
            Column::new("logratio", &logratio),
        ],
        r#"
        suppressPackageStartupMessages(library(INLA))
        # Order rows by the covariate so rw2's ordered-lattice assumption holds,
        # then bin onto the rw2 location index. Restore original order at the end.
        df$.row <- seq_len(nrow(df))
        df$xg <- inla.group(df$range, n = 50, method = "quantile")
        form <- logratio ~ -1 + intercept + f(xg, model = "rw2",
                                               scale.model = TRUE,
                                               constr = TRUE)
        df$intercept <- 1
        m <- inla(form, data = df, family = "gaussian",
                  control.predictor = list(compute = TRUE),
                  control.compute = list(config = TRUE),
                  control.inla = list(int.strategy = "grid"))
        # Posterior-mean fitted values + marginal SD at each training row, in the
        # ORIGINAL row order.
        fit_mean <- m$summary.fitted.values$mean[df$.row]
        fit_sd   <- m$summary.fitted.values$sd[df$.row]
        emit("fitted", as.numeric(fit_mean))
        emit("sd", as.numeric(fit_sd))
        # Posterior mode of the latent field's log-precision (log tau for rw2).
        hp <- m$summary.hyperpar
        rw2_row <- grep("rw2|xg", rownames(hp), ignore.case = TRUE)[1]
        log_tau <- log(hp$mode[rw2_row])
        emit("log_tau", as.numeric(log_tau))
        # Gaussian observation precision -> sigma^2 = 1 / precision (posterior mode).
        prec_row <- grep("Precision for the Gaussian observations",
                         rownames(hp), fixed = TRUE)
        sigma2 <- 1.0 / hp$mode[prec_row]
        emit("sigma2", as.numeric(sigma2))
        "#,
    );
    let inla_fitted = r.vector("fitted");
    let inla_sd = r.vector("sd");
    let inla_log_tau = r.scalar("log_tau");
    let inla_sigma2 = r.scalar("sigma2");

    assert_eq!(inla_fitted.len(), n, "INLA fitted length mismatch");
    assert_eq!(inla_sd.len(), n, "INLA SD length mismatch");

    // ---- gauge: map INLA precision tau -> gam penalty weight lambda -------
    // Identity-Gaussian penalized objective is  (1/sigma^2)||y-Xb||^2 + lambda bᵀSb;
    // INLA's latent prior is  tau bᵀRb  with R == S after both standardize the
    // structure matrix. The observation noise enters gam's REML scale, so the
    // dimensionless penalty weight comparable to gam's lambda is  tau * sigma^2.
    // Use INLA's own sigma^2 for INLA's side and gam's sigma^2 for gam's side;
    // both are already folded into their respective log-lambda definitions, so
    // we compare  log(lambda_gam)  against  log(tau) + log(sigma2_inla).
    let inla_log_lambda = inla_log_tau + inla_sigma2.ln();

    // ---- compare ----------------------------------------------------------
    let rel_fit = relative_l2(&gam_fitted, inla_fitted);
    let sd_rmse = rmse(&gam_sd, inla_sd);
    let inla_sd_mean = inla_sd.iter().sum::<f64>() / inla_sd.len() as f64;
    let sd_tol = 0.1 * inla_sd_mean;
    let lambda_rel = (gam_log_lambda - inla_log_lambda).abs() / gam_log_lambda.abs().max(1e-12);

    eprintln!(
        "lidar RW2 gam vs INLA: n={n} \
         rel_l2(fitted)={rel_fit:.4} \
         gam_log_lambda={gam_log_lambda:.4} inla_log_tau={inla_log_tau:.4} \
         inla_sigma2={inla_sigma2:.4} gam_sigma2={gam_sigma2:.4} \
         inla_log_lambda={inla_log_lambda:.4} lambda_rel={lambda_rel:.4} \
         sd_rmse={sd_rmse:.5} inla_sd_mean={inla_sd_mean:.5} sd_tol={sd_tol:.5}"
    );

    // Metric 1: fitted curves agree. Two different estimators of the same
    // structured prior (continuous B-spline RW2 penalty vs discrete grouped RW2
    // latent field, marginalized over tau) -> 8% relative L2 catches a real
    // structural divergence while admitting their legitimate methodological gap.
    assert!(
        rel_fit < 0.08,
        "gam RW2 fitted values diverge from INLA: rel_l2={rel_fit:.4}"
    );

    // Metric 2: both engines' evidence calculations select the same smoothing
    // level on the log scale (after the sigma^2 gauge). 30% relative on the log
    // penalty weight is principled: log lambda is the quantity both engines
    // optimize, and a 30% gap on the log scale would already flag a real
    // disagreement in the marginal-likelihood / evidence machinery.
    assert!(
        lambda_rel < 0.3,
        "gam and INLA disagree on the selected smoothing level: \
         log_lambda_gam={gam_log_lambda:.4} log_lambda_inla={inla_log_lambda:.4} \
         (rel={lambda_rel:.4})"
    );

    // Metric 3: posterior uncertainty agrees. RMSE of the fitted-value SDs must
    // be within 10% of INLA's mean SD — i.e. the two posterior-uncertainty
    // bands coincide to a tenth of their own scale.
    assert!(
        sd_rmse < sd_tol,
        "gam and INLA posterior SDs disagree: rmse={sd_rmse:.5} \
         exceeds 0.1*mean(inla_sd)={sd_tol:.5}"
    );
}
