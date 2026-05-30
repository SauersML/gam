//! End-to-end quality: gam's tensor *interaction* smooth `ti(x, z, k=6)` must
//! match mgcv — the mature, standard GAM implementation — on identical data.
//!
//! Reference: `mgcv::gam(y ~ ti(x, z, bs="ps", m=list(c(2,2),c(2,2)), k=6),
//! method="REML")`.
//!
//! `ti()` is mgcv's mechanism for an interaction-ONLY tensor smooth: each
//! marginal B-spline basis is sum-to-zero centered *before* the tensor product
//! is formed, so the resulting column space contains no function of a single
//! variable alone — the marginal main effects are excluded and only the pure
//! `x×z` interaction survives. This is algebraically distinct from `te()`'s
//! single global centering constraint: `te(x,z,k=6)` keeps `k*k - 1 = 35`
//! coefficients (one global sum-to-zero drop), whereas `ti(x,z,k=6)` carries
//! exactly `(k-1)^2 = 25` — the Kronecker product `Z₀ ⊗ Z₁` of the two
//! per-margin null-space bases, each margin contributing `(k-1)=5` dimensions.
//!
//! The marginal basis MUST be matched to gam's tensor margin to make the
//! fitted-surface comparison an apples-to-apples test of the SAME penalized
//! objective. gam's `ti(x,z,k=6)` margins are cubic B-splines (`degree=3`) with
//! a second-order DIFFERENCE penalty (`penalty_order=2`) and 6 basis columns
//! per margin (`num_internal_knots = k-(degree+1) = 2`). That is precisely
//! mgcv's P-spline `bs="ps"` with `m=c(2,2)` (cubic B-spline + 2nd-order
//! difference penalty), so the R reference uses `bs="ps", m=list(c(2,2),c(2,2))`
//! rather than the mgcv `ti` default `bs="cr"` (cubic regression spline with a
//! curvature/derivative penalty), which would be a genuinely different smoother
//! and would not justify a tight pointwise bound.
//!
//! This test benchmarks two things gam must replicate exactly:
//!   1. The fitted interaction surface, pointwise on the training grid, must
//!      agree with mgcv's (both REML-fit the SAME penalized objective, so close
//!      agreement is the correct expectation; a real divergence is a real bug).
//!   2. The coefficient count of the `ti` block must be `(k-1)^2`, NOT
//!      `(k-1)*k` or `k*k-1`. A count mismatch is direct proof that per-margin
//!      centering-before-tensor (the load-bearing `ti` identifiability
//!      contract) is broken — gam would then be silently fitting a `te`-style
//!      smooth or leaking main-effect dimensions into the interaction block.
//!
//! Data: a fixed, deterministic 18×18 grid on [0,1]² (n=324, no RNG) with truth
//! f(x,z) = sin(3x)·cos(3z), a product structure that lives largely in the pure
//! interaction subspace. Identical rows are handed to gam and to mgcv.

use csv::StringRecord;
use gam::matrix::LinearOperator;
use gam::smooth::build_term_collection_design;
use gam::test_support::reference::{Column, pearson, relative_l2, run_r};
use gam::{
    FitConfig, FitResult, encode_recordswith_inferred_schema, fit_from_formula, init_parallelism,
};
use ndarray::Array2;

// k as passed to ti(...); per-margin centering => (k-1)^2 interaction coefs.
const K: usize = 6;
// 18x18 deterministic grid on [0,1]^2 => n = 324, no RNG (identical to mgcv).
const GRID: usize = 18;

#[test]
fn gam_ti_2d_interaction_matches_mgcv() {
    init_parallelism();

    // ---- build the deterministic interaction-only grid --------------------
    // f(x,z) = sin(3x)*cos(3z) on a regular 18x18 grid; n = 324, fixed seed-free.
    let n = GRID * GRID;
    let mut x = Vec::with_capacity(n);
    let mut z = Vec::with_capacity(n);
    let mut y = Vec::with_capacity(n);
    for i in 0..GRID {
        let xi = i as f64 / (GRID as f64 - 1.0); // in [0,1]
        for j in 0..GRID {
            let zj = j as f64 / (GRID as f64 - 1.0); // in [0,1]
            x.push(xi);
            z.push(zj);
            y.push((3.0 * xi).sin() * (3.0 * zj).cos());
        }
    }

    // ---- encode for gam ---------------------------------------------------
    let headers = ["x", "z", "y"].into_iter().map(String::from).collect();
    let rows: Vec<StringRecord> = (0..n)
        .map(|r| StringRecord::from(vec![x[r].to_string(), z[r].to_string(), y[r].to_string()]))
        .collect();
    let ds = encode_recordswith_inferred_schema(headers, rows).expect("encode ti grid");
    let col = ds.column_map();
    let x_idx = col["x"];
    let z_idx = col["z"];

    // ---- fit with gam: y ~ ti(x, z, k=6), REML ----------------------------
    let cfg = FitConfig {
        family: Some("gaussian".to_string()),
        ..FitConfig::default()
    };
    let result = fit_from_formula("y ~ ti(x, z, k=6)", &ds, &cfg).expect("gam ti fit succeeded");
    let FitResult::Standard(fit) = result else {
        panic!("expected a standard GAM fit for y ~ ti(x, z, k=6)");
    };

    // gam fitted values at the training grid: rebuild the frozen design at the
    // observed (x, z) (identity link => design*beta = mean).
    let mut grid = Array2::<f64>::zeros((n, ds.headers.len()));
    for r in 0..n {
        grid[[r, x_idx]] = x[r];
        grid[[r, z_idx]] = z[r];
    }
    let design = build_term_collection_design(grid.view(), &fit.resolvedspec)
        .expect("rebuild ti design at training points");
    let gam_fitted: Vec<f64> = design.design.apply(&fit.fit.beta).to_vec();

    // Coefficient count of the ti block: total design columns minus the single
    // intercept column. Per-margin centering-before-tensor must yield (k-1)^2.
    let total_cols = design.design.ncols();
    let intercept_cols = design.intercept_range.len();
    let ti_coeff_count = total_cols - intercept_cols;

    // ---- fit the SAME model with mgcv (the mature reference) --------------
    let r = run_r(
        &[
            Column::new("x", &x),
            Column::new("z", &z),
            Column::new("y", &y),
        ],
        r#"
        suppressPackageStartupMessages(library(mgcv))
        # bs="ps", m=c(2,2) per margin == gam's tensor margin: cubic B-spline +
        # 2nd-order difference penalty, 6 columns/margin for k=6. This matches
        # gam's penalized objective so the fitted surfaces are comparable.
        m <- gam(y ~ ti(x, z, bs = "ps", m = list(c(2, 2), c(2, 2)), k = 6),
                 data = df, method = "REML")
        emit("fitted", as.numeric(fitted(m)))
        # number of coefficients in the ti smooth block (excludes intercept):
        # mgcv stores per-smooth coefficient ranges in m$smooth[[1]]$first.para..last.para
        s1 <- m$smooth[[1]]
        emit("ti_ncoef", as.numeric(s1$last.para - s1$first.para + 1))
        "#,
    );
    let mgcv_fitted = r.vector("fitted");
    let mgcv_ti_ncoef = r.scalar("ti_ncoef").round() as usize;

    assert_eq!(mgcv_fitted.len(), n, "mgcv fitted length mismatch");

    // ---- compare ----------------------------------------------------------
    let rel = relative_l2(&gam_fitted, mgcv_fitted);
    let corr = pearson(&gam_fitted, mgcv_fitted);
    let expected_count = (K - 1) * (K - 1);

    eprintln!(
        "ti(x,z,k=6) gaussian: n={n} rel_l2={rel:.4} pearson={corr:.5} \
         gam_ti_ncoef={ti_coeff_count} mgcv_ti_ncoef={mgcv_ti_ncoef} \
         expected=(k-1)^2={expected_count}"
    );

    // (1) LOAD-BEARING: gam's ti block must carry exactly (k-1)^2 coefficients.
    //     A mismatch proves per-margin centering-before-tensor is broken (would
    //     show (k-1)*k = 30 if only one margin centered, or k*k-1 = 35 if it
    //     fell back to te()'s single global constraint).
    assert_eq!(
        ti_coeff_count, expected_count,
        "gam ti coefficient count {ti_coeff_count} != (k-1)^2 = {expected_count}: \
         per-margin sum-to-zero identifiability is broken"
    );
    // mgcv must agree on the same count — it is the canonical ti contract.
    assert_eq!(
        mgcv_ti_ncoef, expected_count,
        "mgcv ti coefficient count {mgcv_ti_ncoef} != (k-1)^2 = {expected_count}"
    );

    // (2) Both engines REML-fit the SAME interaction-only objective on identical
    //     data with a matched marginal basis (cubic P-spline + 2nd-order
    //     difference penalty, 6 cols/margin), so their fitted surfaces must
    //     essentially coincide. 0.025 is a tight relative-L2 bound: the only
    //     residual difference is internal-knot placement (gam vs mgcv quantile
    //     vs even spacing), which on this uniform grid is negligible, yet the
    //     bound still catches any real divergence in the interaction smoother.
    assert!(
        corr > 0.999,
        "ti fitted surfaces should be near-identical: pearson={corr:.5}"
    );
    assert!(
        rel < 0.025,
        "ti fitted surface diverges from mgcv: rel_l2={rel:.4}"
    );
}
