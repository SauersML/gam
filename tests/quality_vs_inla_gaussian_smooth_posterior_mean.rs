//! End-to-end quality: gam's REML/Laplace latent-Gaussian smoothing must match
//! **R-INLA** — the reference scalable approximate-Bayesian engine for
//! latent-Gaussian models — on real data, for both the posterior *mean* and the
//! posterior *uncertainty* (SD) of the fitted smooth.
//!
//! Why INLA is the right comparator. gam's penalized smooth `y ~ s(x)` with a
//! Gaussian response is exactly a latent-Gaussian model: a smooth function
//! `f(x)` with a Gaussian-process / penalized prior (here a 2nd-derivative
//! penalty, the thin-plate analogue), Gaussian observation noise, and a
//! data-driven smoothing hyperparameter. INLA fits the *same structural model*
//! — `y_i = f(x_i) + eps_i` with `f` an `rw2` (second-order random walk, the
//! canonical discrete analogue of a 2nd-derivative penalized smooth) — but by a
//! different route: gam uses a single Laplace approximation at the REML-selected
//! smoothing parameter, whereas INLA integrates nested Laplace approximations
//! over the hyperparameter posterior. Because they target the same latent
//! posterior, the fitted posterior means must nearly coincide and the posterior
//! SDs (credible-interval widths) must be very close. A real divergence is a
//! real modelling bug, not a basis-convention artefact.
//!
//! NOTE on licensing: R-INLA is distributed under a non-commercial license
//! (academic/research use). This is a scientific benchmarking test that fits a
//! published smoothing dataset to validate gam's approximate-Bayesian inference;
//! it is not a production or commercial comparison.
//!
//! Data: the canonical `lidar` smoothing benchmark (`logratio ~ s(range)`,
//! n = 221), fed identically to both engines.
//!
//! Bounds (principled, un-weakened):
//!   1. Posterior mean: relative_l2(gam, INLA) < 0.05 — both recover the same
//!      latent function; 5% L2 is tight on a smooth signal-dominated fit yet
//!      tolerates the rw2-vs-thin-plate basis difference at the boundary knots.
//!   2. Posterior SD width: max_abs_diff(gam_sd, inla_sd) / mean(inla_sd) < 0.15
//!      — credible-band widths must agree to ~15% of the typical width; this is
//!      strict enough to catch a mis-scaled covariance yet absorbs the
//!      single-Laplace-vs-INLA hyperparameter-integration gap.
//!   3. Pearson correlation of posterior means > 0.995 — the two fitted curves
//!      must be essentially the same shape.

use gam::matrix::LinearOperator;
use gam::smooth::build_term_collection_design;
use gam::test_support::reference::{Column, max_abs_diff, pearson, relative_l2, run_r};
use gam::{FitConfig, FitResult, fit_from_formula, init_parallelism, load_csvwith_inferred_schema};
use ndarray::{Array1, Array2};
use std::path::Path;

const LIDAR_CSV: &str = concat!(env!("CARGO_MANIFEST_DIR"), "/bench/datasets/lidar.csv");

#[test]
fn gam_smooth_posterior_matches_inla_on_lidar() {
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

    // ---- fit with gam: logratio ~ s(range, bs='tp'), REML/Laplace ---------
    let cfg = FitConfig {
        family: Some("gaussian".to_string()),
        ..FitConfig::default()
    };
    let result = fit_from_formula("logratio ~ s(range, bs='tp')", &ds, &cfg).expect("gam fit");
    let FitResult::Standard(fit) = result else {
        panic!("expected a standard GAM fit");
    };

    // gam posterior mean of the fitted smooth at the observed points: rebuild
    // the design from the frozen spec (identity link => design*beta = mean).
    let mut grid = Array2::<f64>::zeros((n, ds.headers.len()));
    for (i, &r) in range.iter().enumerate() {
        grid[[i, range_idx]] = r;
    }
    let design = build_term_collection_design(grid.view(), &fit.resolvedspec)
        .expect("rebuild design at training points");
    let gam_mean: Vec<f64> = design.design.apply(&fit.fit.beta).to_vec();

    // gam posterior SD of the fitted smooth: SD_i = sqrt(X_i Vp X_i^T), where
    // Vp is the Bayesian covariance WITH the smoothing-parameter uncertainty
    // correction — the analogue of INLA integrating over the hyperparameter
    // posterior, so it is the right object to compare against INLA's marginals.
    let vp = fit
        .fit
        .covariance_corrected
        .as_ref()
        .expect("Gaussian REML fit should expose the corrected (Vp) covariance");
    let p = fit.fit.beta.len();
    assert_eq!(vp.nrows(), p, "Vp must be p x p");
    // Materialise the design rows densely (n small, p small) by transposing
    // one-hot row selectors through the operator: row i = X^T e_i.
    let mut x_dense = Array2::<f64>::zeros((n, p));
    for i in 0..n {
        let mut e = Array1::<f64>::zeros(n);
        e[i] = 1.0;
        let row = design.design.apply_transpose(&e);
        for j in 0..p {
            x_dense[[i, j]] = row[j];
        }
    }
    let mut gam_sd = vec![0.0f64; n];
    for i in 0..n {
        let xi = x_dense.row(i).to_owned();
        let vxi: Array1<f64> = vp.dot(&xi);
        let var: f64 = xi.iter().zip(vxi.iter()).map(|(a, b)| a * b).sum();
        gam_sd[i] = var.max(0.0).sqrt();
    }

    // ---- fit the SAME latent-Gaussian model with R-INLA -------------------
    // f(idx, model="rw2") is the second-order random walk: the discrete
    // analogue of a 2nd-derivative penalized smooth (gam's thin-plate target).
    // We index the smooth by the rank of `range` (lidar's range values are
    // distinct and ordered), giving INLA the identical covariate ordering.
    // summary.fitted.values gives the posterior marginal (mean + sd) of the
    // linear predictor at each observation = the fitted smooth for an
    // identity-link Gaussian, directly comparable to gam's mean/SD above.
    let r = run_r(
        &[
            Column::new("range", &range),
            Column::new("logratio", &logratio),
        ],
        r#"
        suppressPackageStartupMessages(library(INLA))
        idx <- rank(df$range, ties.method = "first")
        dat <- data.frame(y = df$logratio, idx = idx)
        m <- inla(
            y ~ f(idx, model = "rw2", scale.model = TRUE,
                  constr = TRUE),
            family = "gaussian",
            data = dat,
            control.predictor = list(compute = TRUE),
            control.compute = list(config = FALSE)
        )
        # summary.fitted.values is in the data-frame row order (one row per
        # observation), so it already aligns element-wise with gam's gam_mean /
        # gam_sd, which are in the same input row order. `idx` only sets the
        # rw2 ordering of the latent field; it does not permute the fitted
        # values relative to the input rows.
        fv <- m$summary.fitted.values
        emit("mean", fv$mean[seq_len(nrow(df))])
        emit("sd",   fv$sd[seq_len(nrow(df))])
        "#,
    );
    let inla_mean = r.vector("mean");
    let inla_sd = r.vector("sd");

    assert_eq!(inla_mean.len(), n, "INLA fitted-mean length mismatch");
    assert_eq!(inla_sd.len(), n, "INLA fitted-sd length mismatch");

    // ---- compare ----------------------------------------------------------
    let rel = relative_l2(&gam_mean, inla_mean);
    let corr = pearson(&gam_mean, inla_mean);
    let mean_inla_sd = inla_sd.iter().sum::<f64>() / (n as f64);
    let sd_width_rel = max_abs_diff(&gam_sd, inla_sd) / mean_inla_sd.max(1e-300);
    let mean_gam_sd = gam_sd.iter().sum::<f64>() / (n as f64);

    eprintln!(
        "lidar s(range) gam-vs-INLA: n={n} p={p} \
         rel_l2(mean)={rel:.4} pearson(mean)={corr:.5} \
         mean_gam_sd={mean_gam_sd:.4} mean_inla_sd={mean_inla_sd:.4} \
         sd_width_rel={sd_width_rel:.4}"
    );

    // (1) Posterior means recover the same latent function.
    assert!(
        rel < 0.05,
        "posterior means diverge from INLA: rel_l2={rel:.4} (bound 0.05)"
    );
    // (3) Same curve shape.
    assert!(
        corr > 0.995,
        "posterior means are not the same shape as INLA: pearson={corr:.5} (bound 0.995)"
    );
    // (2) Credible-band widths agree to ~15% of the typical width.
    assert!(
        sd_width_rel < 0.15,
        "posterior SD widths diverge from INLA: max_abs_diff/mean_sd={sd_width_rel:.4} (bound 0.15)"
    );
}
