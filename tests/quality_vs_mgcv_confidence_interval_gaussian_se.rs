//! End-to-end quality: gam's pointwise standard errors on a Gaussian smooth
//! must match `mgcv` — the mature, standard GAM implementation — element by
//! element on the same data.
//!
//! Benchmarked tool: `mgcv::predict.gam(..., se.fit = TRUE, type = "link")`.
//! mgcv is the reference for penalized GAMs; its `predict(se.fit=TRUE)` returns
//! `sqrt(diag(X Vp Xᵀ))` on the linear-predictor scale, where `Vp` is the
//! Bayesian posterior covariance of the coefficients (`H⁻¹ φ̂`, conditional on
//! the REML-estimated smoothing parameters). This is the atomic unit behind
//! every confidence-interval half-width.
//!
//! gam exposes the same Bayesian covariance as `fit.vb_covariance()` (Wood's
//! `Vb`/`Vp`), so gam's linear-predictor standard error at a query row `x` is
//! `sqrt(xᵀ Vb x)`. Both engines REML-fit `logratio ~ s(range)` on the
//! identical lidar data, so the *standard error of the fitted linear predictor*
//! at each training `range` must coincide — it is a basis-invariant property of
//! the fit (Var of η̂(x)), not an artifact of either basis. A real divergence
//! here is a real bug in the covariance computation or the design reconstruction.
//!
//! We evaluate at the training `range` points to isolate the design/covariance
//! path from any boundary/OOD correction factors.

use gam::matrix::LinearOperator;
use gam::smooth::build_term_collection_design;
use gam::test_support::reference::{Column, max_abs_diff, pearson, relative_l2, run_r};
use gam::{FitConfig, FitResult, fit_from_formula, init_parallelism, load_csvwith_inferred_schema};
use ndarray::{Array1, Array2};
use std::path::Path;

const LIDAR_CSV: &str = concat!(env!("CARGO_MANIFEST_DIR"), "/bench/datasets/lidar.csv");

#[test]
fn gam_pointwise_eta_se_matches_mgcv_on_lidar() {
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

    // ---- fit with gam: logratio ~ s(range), REML --------------------------
    let cfg = FitConfig {
        family: Some("gaussian".to_string()),
        ..FitConfig::default()
    };
    let result = fit_from_formula("logratio ~ s(range)", &ds, &cfg).expect("gam fit");
    let FitResult::Standard(fit) = result else {
        panic!("expected a standard GAM fit");
    };

    // Bayesian coefficient covariance Vb = H⁻¹ φ̂ (mgcv's Vp; the matrix that
    // predict(se.fit=TRUE) uses by default). Must be available for a fitted
    // Gaussian GAM; its absence would itself be the failure.
    let vb = fit
        .fit
        .vb_covariance()
        .expect("gam reports Bayesian coefficient covariance Vb")
        .clone();
    let p = fit.fit.beta.len();
    assert_eq!(vb.nrows(), p, "Vb dimension must match coefficient count");
    assert_eq!(vb.ncols(), p, "Vb must be square in coefficient space");

    // Rebuild the frozen design at the training `range` points (identity link
    // => the design row is exactly the linear-predictor sensitivity x = ∂η/∂β).
    let mut grid = Array2::<f64>::zeros((n, ds.headers.len()));
    for (i, &r) in range.iter().enumerate() {
        grid[[i, range_idx]] = r;
    }
    let design = build_term_collection_design(grid.view(), &fit.resolvedspec)
        .expect("rebuild design at training points");
    assert_eq!(
        design.design.ncols(),
        p,
        "design column count must match coefficient dimension"
    );
    assert_eq!(design.design.nrows(), n, "design row count must match data");

    // Materialize the dense design X by applying the operator to each unit
    // basis vector: X·e_j is column j. p is small (~10), n=221, so this is cheap
    // and keeps the comparison on the exact operator the solver used.
    let mut x_dense = Array2::<f64>::zeros((n, p));
    for j in 0..p {
        let mut e_j = Array1::<f64>::zeros(p);
        e_j[j] = 1.0;
        let col_j = design.design.apply(&e_j);
        for i in 0..n {
            x_dense[[i, j]] = col_j[i];
        }
    }

    // Pointwise SE of the linear predictor: se_i = sqrt(x_iᵀ Vb x_i).
    let gam_eta_se: Vec<f64> = (0..n)
        .map(|i| {
            let xi = x_dense.row(i);
            let vbxi = vb.dot(&xi);
            let var = xi.dot(&vbxi);
            var.max(0.0).sqrt()
        })
        .collect();

    // ---- fit the SAME model with mgcv and ask for predict(se.fit=TRUE) ----
    let r = run_r(
        &[
            Column::new("range", &range),
            Column::new("logratio", &logratio),
        ],
        r#"
        suppressPackageStartupMessages(library(mgcv))
        m <- gam(logratio ~ s(range), data = df, method = "REML")
        # type="link" => standard error of the fitted linear predictor (== response
        # scale here, identity link). Uses Vp = H^{-1} * scale by default, exactly
        # the conditional Bayesian covariance gam exposes as Vb.
        pr <- predict(m, newdata = df, se.fit = TRUE, type = "link")
        emit("se", as.numeric(pr$se.fit))
        "#,
    );
    let mgcv_se = r.vector("se");
    assert_eq!(mgcv_se.len(), n, "mgcv se.fit length mismatch");

    // ---- compare element-wise on the linear-predictor SE ------------------
    let mad = max_abs_diff(&gam_eta_se, mgcv_se);
    let rel = relative_l2(&gam_eta_se, mgcv_se);
    let corr = pearson(&gam_eta_se, mgcv_se);

    eprintln!(
        "lidar s(range) eta-SE vs mgcv: n={n} p={p} max_abs_diff={mad:.6} \
         rel_l2={rel:.5} pearson={corr:.6}"
    );

    // Both engines fit the same Gaussian GAM by REML and compute the SE of the
    // fitted linear predictor as sqrt(diag(X Vb Xᵀ)). The SE is basis-invariant,
    // so the only sources of disagreement are (a) a genuinely different smoothing
    // parameter / scale estimate or (b) a bug in gam's covariance or design path.
    // mgcv's lidar SE ranges ~0.0143–0.0290 (mean ~0.0154) for this REML fit; the
    // bounds below are tight enough that any real divergence in the covariance
    // computation trips them (a 5% relative-L2 break is ~0.0008 on this scale, and
    // the max-abs bound of 0.003 is ~20% of the typical SE — a gross covariance or
    // design bug distorts the SE far beyond that), while still allowing for the
    // small basis/null-space-convention differences between gam's default
    // thin-plate construction and mgcv's k=10 default.
    assert!(
        corr > 0.9999,
        "pointwise eta SE shape must track mgcv near-exactly: pearson={corr:.6}"
    );
    assert!(
        rel < 0.05,
        "pointwise eta SE diverges from mgcv in relative L2: rel_l2={rel:.5}"
    );
    assert!(
        mad < 0.003,
        "pointwise eta SE max abs diff from mgcv too large: max_abs_diff={mad:.6}"
    );
}
