//! End-to-end quality: gam's Gaussian *location-scale* fit with a TENSOR-PRODUCT
//! smooth in BOTH the mean and the log-sigma block must match `mgcv::gam(family
//! = gaulss())` — the mature, standard implementation of Gaussian location-scale
//! GAMs with multidimensional `te(...)` tensor smooths in R.
//!
//! This is a cross-feature combination that the univariate location-scale test
//! (`quality_vs_gamlss_gaussian_location_scale.rs`) never exercises: family
//! (Gaussian) x location-scale (mean + log-sigma) x TENSOR-PRODUCT smooth (each
//! block depends on two covariates through a Kronecker-structured basis with a
//! shared knot layout and Kronecker-sum penalty). Tensor smooths inside gaulss
//! are far less tested than univariate smooths, so we compare the recovered
//! SMOOTH SHAPES (not held-out predictions) on a dense 2-D grid:
//!   * the fitted mean surface mu(x1, x2), and
//!   * the fitted log standard-deviation surface log(sigma(x1, x2)).
//!
//! Both engines maximize the same penalized Gaussian location-scale joint
//! log-likelihood `ell = -1/2 sum (y-mu)^2 / sigma^2 - sum log sigma` over a
//! tensor-product basis, so the recovered surfaces should converge to nearly
//! identical shapes up to basis-convention / numerical tolerance. A divergence
//! here is a real bug in how gam builds and applies tensor penalties (Kronecker
//! products of marginal penalties, shared multi-dimensional knot structure) in a
//! multi-block location-scale setting.
//!
//! Notes pinned by reading the source:
//!   * `fit_from_formula(..., FitConfig{ noise_formula: Some(...), .. })` routes
//!     through `materialize_location_scale` -> `FitRequest::GaussianLocationScale`,
//!     in raw response units (no y-rescaling on the in-Rust path).
//!   * gam's noise (sigma) link is `sigma = LOGB_SIGMA_FLOOR + exp(eta_scale)`
//!     with `LOGB_SIGMA_FLOOR = 0.01` (see `families::sigma_link`), the same soft
//!     floor mgcv's `gaulss(b=0.01)` uses. The location block carries role
//!     `BlockRole::Location`, the log-sigma block role `BlockRole::Scale`.
//!   * `te(x1, x2, bs=c('tp','tp'))` parses to a `SmoothKind::Te` tensor smooth.
//!     `linkwiggle(...)` is a binomial-only link correction and is rejected for a
//!     Gaussian response, so it is intentionally absent from the gam formula.

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
/// Mirrors `families::sigma_link::LOGB_SIGMA_FLOOR` (and mgcv `gaulss(b=0.01)`).
const LOGB_SIGMA_FLOOR: f64 = 0.01;

#[test]
fn gam_gaulss_tensor_product_matches_mgcv() {
    init_parallelism();

    // ---- synthetic 2-D heteroscedastic recipe (fed IDENTICALLY to both engines) ----
    // n=200, x1 ~ Uniform(-1,1), x2 ~ Uniform(-1,1),
    // mu(x1,x2)   = sin(pi*x1)*cos(pi*x2),
    // sigma(x1,x2)= 0.1 + 0.15*|x1| + 0.1*|x2|  (strictly positive everywhere),
    // y ~ N(mu, sigma^2), seed=654. A deterministic seeded LCG draws the uniforms
    // and the standard normals so the exact same (x1, x2, y) rows are reproducible
    // in pure Rust and sent verbatim to mgcv.
    let n = 200usize;
    let pi = std::f64::consts::PI;
    let two_pi = 2.0 * pi;

    let mut state: u64 = 654;
    let mut next_unit = || -> f64 {
        // Numerical Recipes LCG; take the high bits for a uniform in [0,1).
        state = state
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        ((state >> 11) as f64) / ((1u64 << 53) as f64)
    };

    // Draw the two covariates on [-1,1].
    let x1: Vec<f64> = (0..n).map(|_| 2.0 * next_unit() - 1.0).collect();
    let x2: Vec<f64> = (0..n).map(|_| 2.0 * next_unit() - 1.0).collect();

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

    let mu_true = |a: f64, b: f64| (pi * a).sin() * (pi * b).cos();
    let sigma_true = |a: f64, b: f64| 0.1 + 0.15 * a.abs() + 0.1 * b.abs();
    let y: Vec<f64> = (0..n)
        .map(|i| mu_true(x1[i], x2[i]) + sigma_true(x1[i], x2[i]) * z[i])
        .collect();

    // ---- build the dataset (columns x1, x2, y) -----------------------------
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
    let ds =
        encode_recordswith_inferred_schema(headers, rows).expect("encode gaulss-tensor data");
    let col = ds.column_map();
    let x1_idx = col["x1"];
    let x2_idx = col["x2"];
    let ncols = ds.headers.len();

    // ---- fit with gam: mu ~ te(x1,x2), log-sigma ~ 1 + te(x1,x2) -----------
    let cfg = FitConfig {
        family: Some("gaussian".to_string()),
        noise_formula: Some("1 + te(x1, x2, bs=c('tp','tp'))".to_string()),
        ..FitConfig::default()
    };
    let result = fit_from_formula("y ~ te(x1, x2, bs=c('tp','tp'))", &ds, &cfg)
        .expect("gam gaulss tensor-product fit");
    let FitResult::GaussianLocationScale(GaussianLocationScaleFitResult { fit, .. }) = result
    else {
        panic!("expected a Gaussian location-scale fit for a smooth noise_formula model");
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

    // ---- evaluate gam's tensor surfaces on a dense 10x10 grid over [-1,1]^2 -
    let grid_side = 10usize;
    let grid_n = grid_side * grid_side;
    let mut grid_x1: Vec<f64> = Vec::with_capacity(grid_n);
    let mut grid_x2: Vec<f64> = Vec::with_capacity(grid_n);
    for i in 0..grid_side {
        for j in 0..grid_side {
            let a = -1.0 + 2.0 * (i as f64) / (grid_side as f64 - 1.0);
            let b = -1.0 + 2.0 * (j as f64) / (grid_side as f64 - 1.0);
            grid_x1.push(a);
            grid_x2.push(b);
        }
    }
    let mut grid = Array2::<f64>::zeros((grid_n, ncols));
    for k in 0..grid_n {
        grid[[k, x1_idx]] = grid_x1[k];
        grid[[k, x2_idx]] = grid_x2[k];
    }

    // Rebuild the SAME frozen mean / log-sigma tensor designs at the grid points
    // and apply each block's coefficients. mu = X_mean*beta_location;
    // sigma = LOGB_SIGMA_FLOOR + exp(X_scale*beta_scale).
    let mean_design_grid = build_term_collection_design(grid.view(), &fit.meanspec_resolved)
        .expect("rebuild mean tensor design at grid");
    let scale_design_grid = build_term_collection_design(grid.view(), &fit.noisespec_resolved)
        .expect("rebuild log-sigma tensor design at grid");

    let gam_mu: Vec<f64> = mean_design_grid.design.apply(&beta_location).to_vec();
    let gam_eta_sigma: Vec<f64> = scale_design_grid.design.apply(&beta_scale).to_vec();
    let gam_sigma: Vec<f64> = gam_eta_sigma
        .iter()
        .map(|&e| LOGB_SIGMA_FLOOR + e.exp())
        .collect();
    let gam_log_sigma: Vec<f64> = gam_sigma.iter().map(|&s| s.ln()).collect();

    assert_eq!(gam_mu.len(), grid_n);
    assert_eq!(gam_sigma.len(), grid_n);

    // ---- fit the SAME model with mgcv gaulss (the mature reference) --------
    // family = gaulss() expects a two-part formula list: the mean linear
    // predictor and the log-sigma linear predictor. Both carry a te(x1,x2)
    // tensor-product smooth with thin-plate marginals, fit by REML, and are
    // predicted on the identical grid. gaulss's second linear predictor is the
    // log-sigma scale; predict(type="response") returns 1/sigma for the scale
    // part, so we recover sigma = 1 / that response.
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
        suppressPackageStartupMessages(library(mgcv))
        m <- gam(list(y ~ te(x1, x2, bs = c("tp", "tp")),
                         ~ te(x1, x2, bs = c("tp", "tp"))),
                 family = gaulss(), data = df, method = "REML")
        gx1 <- as.numeric(strsplit("{grid_x1_csv}", ",")[[1]])
        gx2 <- as.numeric(strsplit("{grid_x2_csv}", ",")[[1]])
        nd <- data.frame(x1 = gx1, x2 = gx2)
        # gaulss: column 1 = mu (link=identity), column 2 = scale (link=logb).
        # type="response" maps the second predictor to 1/sigma (gaulss b=0.01).
        pr <- predict(m, newdata = nd, type = "response")
        mu <- pr[, 1]
        sigma <- 1.0 / pr[, 2]
        emit("mu", as.numeric(mu))
        emit("sigma", as.numeric(sigma))
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
    let mgcv_mu = r.vector("mu");
    let mgcv_sigma = r.vector("sigma");
    assert_eq!(mgcv_mu.len(), grid_n, "mgcv mu grid length mismatch");
    assert_eq!(mgcv_sigma.len(), grid_n, "mgcv sigma grid length mismatch");
    let mgcv_log_sigma: Vec<f64> = mgcv_sigma.iter().map(|&s| s.ln()).collect();

    // ---- compare the recovered smooth surfaces on the grid -----------------
    let rel_mu = relative_l2(&gam_mu, mgcv_mu);
    let rel_log_sigma = relative_l2(&gam_log_sigma, &mgcv_log_sigma);

    eprintln!(
        "gaulss tensor-product vs mgcv gaulss(): n={n} grid={grid_n} \
         rel_l2(mu)={rel_mu:.5} rel_l2(log sigma)={rel_log_sigma:.5}"
    );

    // Both engines maximize the same penalized Gaussian location-scale joint
    // log-likelihood over a tensor-product basis (Kronecker products of marginal
    // penalties, shared multi-dimensional knot layout), fit by REML. The mean
    // surface is the better-determined parameter (variance-stabilized by the same
    // 1/sigma^2 weights both engines use), hence the tighter 0.025 bound; the
    // log-sigma surface is a second-moment quantity estimated from squared
    // residuals over a 2-D basis and so is allowed a looser 0.05. Either bound
    // being exceeded is a genuine divergence of gam's tensor-penalty construction
    // in a location-scale context from the mgcv gaulss standard.
    assert!(
        rel_mu < 0.025,
        "fitted mean tensor surface diverges from mgcv gaulss: rel_l2(mu)={rel_mu:.5}"
    );
    assert!(
        rel_log_sigma < 0.05,
        "fitted log-sigma tensor surface diverges from mgcv gaulss: rel_l2(log sigma)={rel_log_sigma:.5}"
    );
}
