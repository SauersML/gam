//! End-to-end OBJECTIVE quality of gam's tensor *interaction* smooth
//! `ti(x, z, k=6)` on a noiseless, known surface.
//!
//! OBJECTIVE METRIC (the pass/fail claim):
//!   1. TRUTH RECOVERY. The data are generated *without noise* from a known
//!      function `f(x,z) = sin(3x)·cos(3z)`. An interaction-only `ti` smooth can
//!      only represent the *pure-interaction* part of any surface, so the
//!      well-defined recovery target is the two-way functional-ANOVA interaction
//!      component of the truth, `f_int = f − f̄ − r(x) − c(z)` (grand mean, plus
//!      x- and z-marginal means removed). We assert
//!      `RMSE(gam_fit_int, f_int) <= 0.025 · range(f_int)` — gam reconstructs
//!      the true interaction surface to a small fraction of its own amplitude.
//!      This is an absolute statement about gam's accuracy against ground truth,
//!      not about any other tool's output.
//!
//!      The bar is 2.5% (not the interpolation floor): the data are noiseless,
//!      but REML's restricted-likelihood Occam term forbids `λ → 0`, so a
//!      faithful REML smoother does NOT interpolate even noiseless data — it
//!      settles at a smoothed fixed point. On THIS surface the achievable floor
//!      is ~2.1–2.3% of amplitude for any correct REML engine (mgcv itself lands
//!      at 2.28%; gam at 2.07%), well above the 0.06% the basis could reach at
//!      `λ → 0`. A bar tighter than the REML floor cannot be met by any faithful
//!      smoother and is therefore not the right test; 2.5% comfortably exceeds
//!      what every correct engine achieves here while still catching gross
//!      over-smoothing (the pre-fix gam, whose contaminated penalty null space
//!      drove it to 4.2% of amplitude, fails this bar). The MATCH-OR-BEAT check
//!      below is the binding quality gate.
//!   2. STRUCTURE / IDENTIFIABILITY. `ti` is interaction-ONLY: per-margin
//!      sum-to-zero centering before the tensor product must purge all main
//!      effects. We assert this directly on gam's own fitted surface — its
//!      x-marginal means and z-marginal means over the regular grid must each
//!      vanish *once the model intercept (grand mean) is removed* — a leaked
//!      main effect is a margin-index-dependent mean, whereas the intercept
//!      merely shifts every margin equally and is legitimate (max
//!      |grand-mean-removed marginal mean| <= 1e-6 · range) — and that the smooth block
//!      carries exactly `(k-1)^2` coefficients (the Kronecker null-space
//!      dimension `Z₀ ⊗ Z₁`). A non-zero marginal mean or a wrong coefficient
//!      count is direct proof the `ti` contract is broken (it would have leaked
//!      main effects or fallen back to a `te`-style single global constraint at
//!      `k*k-1 = 35`, or centered only one margin at `(k-1)*k = 30`).
//!
//! BASELINE TO MATCH-OR-BEAT: mgcv (`bs="ps", m=list(c(2,2),c(2,2)), k=6`,
//! method="REML") fits the SAME interaction-only penalized objective with the
//! SAME marginal basis (cubic P-spline + 2nd-order difference penalty, 6
//! cols/margin). We fit mgcv on identical rows, ANOVA-center its fitted surface
//! the same way, and require gam's recovery error to be no worse than mgcv's by
//! more than 10% (`rmse_gam <= 1.10 · rmse_mgcv`). mgcv is the bar to match-or-
//! beat on accuracy, NOT the thing gam must reproduce: the primary claim is
//! truth recovery; the relative-L2 of the two fits is printed for context only.
//!
//! Data: a fixed, deterministic 18×18 grid on [0,1]² (n=324, no RNG). Identical
//! rows are handed to gam and to mgcv.

use csv::StringRecord;
use gam::matrix::LinearOperator;
use gam::smooth::build_term_collection_design;
use gam::test_support::reference::{Column, relative_l2, rmse, run_r};
use gam::{
    FitConfig, FitResult, encode_recordswith_inferred_schema, fit_from_formula, init_parallelism,
};
use ndarray::Array2;

// k as passed to ti(...); per-margin centering => (k-1)^2 interaction coefs.
const K: usize = 6;
// 18x18 deterministic grid on [0,1]^2 => n = 324, no RNG (identical to mgcv).
const GRID: usize = 18;

/// Two-way functional-ANOVA interaction component of a surface sampled on a
/// regular `rows x cols` grid stored row-major (`v[i*cols + j]`): subtract the
/// grand mean, each row mean, and each column mean. The result is the pure
/// interaction part — exactly what an interaction-only `ti` smooth represents —
/// and by construction has zero row means and zero column means.
fn anova_interaction(v: &[f64], rows: usize, cols: usize) -> Vec<f64> {
    assert_eq!(v.len(), rows * cols, "anova_interaction grid size mismatch");
    let n = (rows * cols) as f64;
    let grand: f64 = v.iter().sum::<f64>() / n;
    let mut row_mean = vec![0.0_f64; rows];
    let mut col_mean = vec![0.0_f64; cols];
    for i in 0..rows {
        for j in 0..cols {
            let val = v[i * cols + j];
            row_mean[i] += val;
            col_mean[j] += val;
        }
    }
    for rm in row_mean.iter_mut() {
        *rm /= cols as f64;
    }
    for cm in col_mean.iter_mut() {
        *cm /= rows as f64;
    }
    let mut out = vec![0.0_f64; rows * cols];
    for i in 0..rows {
        for j in 0..cols {
            out[i * cols + j] = v[i * cols + j] - row_mean[i] - col_mean[j] + grand;
        }
    }
    out
}

/// Max absolute x-marginal (row) mean and z-marginal (column) mean of a
/// row-major `rows x cols` grid, AFTER removing the overall (grand) mean. A
/// leaked main effect shows up as a margin-index-dependent mean — i.e. a
/// systematic row-to-row or column-to-column variation. The model INTERCEPT,
/// by contrast, shifts every marginal mean by the same constant and is NOT a
/// main-effect leak, so it must be subtracted out before the check (otherwise
/// the metric measures the grand mean of the surface, not an interaction
/// contract violation). For a genuine interaction-only surface every
/// grand-mean-removed marginal mean must be ~0 at every margin index.
fn max_marginal_mean(v: &[f64], rows: usize, cols: usize) -> f64 {
    let n = (rows * cols) as f64;
    let grand: f64 = v.iter().sum::<f64>() / n;
    let mut worst = 0.0_f64;
    for i in 0..rows {
        let m: f64 = (0..cols).map(|j| v[i * cols + j]).sum::<f64>() / cols as f64 - grand;
        worst = worst.max(m.abs());
    }
    for j in 0..cols {
        let m: f64 = (0..rows).map(|i| v[i * cols + j]).sum::<f64>() / rows as f64 - grand;
        worst = worst.max(m.abs());
    }
    worst
}

fn range_of(v: &[f64]) -> f64 {
    let (mut lo, mut hi) = (f64::INFINITY, f64::NEG_INFINITY);
    for &x in v {
        lo = lo.min(x);
        hi = hi.max(x);
    }
    hi - lo
}

#[test]
fn gam_ti_2d_interaction_recovers_truth() {
    init_parallelism();

    // ---- build the deterministic interaction-only grid --------------------
    // f(x,z) = sin(3x)*cos(3z) on a regular 18x18 grid; n = 324, no RNG.
    // Stored row-major: row i fixes x_i, column j fixes z_j.
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

    // The well-defined recovery target for an interaction-only smooth: the
    // two-way ANOVA interaction component of the noiseless truth.
    let truth_int = anova_interaction(&y, GRID, GRID);
    let truth_int_range = range_of(&truth_int);
    assert!(
        truth_int_range > 0.1,
        "degenerate test: truth interaction component has range {truth_int_range:.4}; \
         f(x,z) must carry real x*z interaction"
    );

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

    // gam's pure interaction surface (the intercept is the only non-ti column,
    // so ANOVA-centering both fit and truth the same way is the apples-to-apples
    // comparison and removes that constant).
    let gam_int = anova_interaction(&gam_fitted, GRID, GRID);

    // ---- fit the SAME model with mgcv (BASELINE to match-or-beat) ---------
    let r = run_r(
        &[
            Column::new("x", &x),
            Column::new("z", &z),
            Column::new("y", &y),
        ],
        r#"
        suppressPackageStartupMessages(library(mgcv))
        # bs="ps", m=c(2,2) per margin == gam's tensor margin: cubic B-spline +
        # 2nd-order difference penalty, 6 columns/margin for k=6.
        m <- gam(y ~ ti(x, z, bs = "ps", m = list(c(2, 2), c(2, 2)), k = 6),
                 data = df, method = "REML")
        emit("fitted", as.numeric(fitted(m)))
        s1 <- m$smooth[[1]]
        emit("ti_ncoef", as.numeric(s1$last.para - s1$first.para + 1))
        "#,
    );
    let mgcv_fitted = r.vector("fitted");
    let mgcv_ti_ncoef = r.scalar("ti_ncoef").round() as usize;
    assert_eq!(mgcv_fitted.len(), n, "mgcv fitted length mismatch");
    let mgcv_int = anova_interaction(mgcv_fitted, GRID, GRID);

    // ---- objective metrics ------------------------------------------------
    let rmse_gam = rmse(&gam_int, &truth_int);
    let rmse_mgcv = rmse(&mgcv_int, &truth_int);
    let rel_gam = relative_l2(&gam_int, &truth_int);
    let rel_vs_mgcv = relative_l2(&gam_int, &mgcv_int); // context only
    let gam_marginal = max_marginal_mean(&gam_fitted, GRID, GRID);
    let expected_count = (K - 1) * (K - 1);
    // 2.5% of amplitude: the REML-achievable recovery floor on this noiseless
    // surface (~2.1–2.3% for any faithful engine; REML's Occam term forbids the
    // λ→0 interpolation that would reach ~0.06%). See the module docstring.
    let recovery_bar = 0.025 * truth_int_range;

    eprintln!(
        "ti(x,z,k=6) gaussian truth-recovery: n={n} \
         rmse_gam_vs_truth={rmse_gam:.5} rmse_mgcv_vs_truth={rmse_mgcv:.5} \
         bar={recovery_bar:.5} rel_gam_vs_truth={rel_gam:.4} \
         rel_gam_vs_mgcv={rel_vs_mgcv:.4}(ctx) \
         gam_max_marginal_mean={gam_marginal:.2e} \
         truth_int_range={truth_int_range:.4} \
         gam_ti_ncoef={ti_coeff_count} mgcv_ti_ncoef={mgcv_ti_ncoef} \
         expected=(k-1)^2={expected_count}"
    );

    // (1) STRUCTURE: gam's ti block carries exactly (k-1)^2 coefficients.
    //     A mismatch proves per-margin centering-before-tensor is broken
    //     ((k-1)*k = 30 if only one margin centered; k*k-1 = 35 if it fell back
    //     to te()'s single global constraint).
    assert_eq!(
        ti_coeff_count, expected_count,
        "gam ti coefficient count {ti_coeff_count} != (k-1)^2 = {expected_count}: \
         per-margin sum-to-zero identifiability is broken"
    );

    // (2) STRUCTURE: the fitted ti surface is genuinely interaction-only — every
    //     x-marginal mean and z-marginal mean must vanish ONCE the model
    //     intercept (grand mean) is removed. A leaked main effect is a
    //     margin-index-dependent mean; the intercept shifts every marginal mean
    //     by the same constant and is legitimate, so `max_marginal_mean`
    //     subtracts the grand mean first. This is the load-bearing `ti`
    //     contract, asserted directly on gam's own output.
    assert!(
        gam_marginal <= 1e-6 * truth_int_range.max(1.0),
        "gam ti surface leaks a main effect: max |marginal mean| = {gam_marginal:.3e} \
         (must be ~0 for an interaction-only smooth)"
    );

    // (3) TRUTH RECOVERY (PRIMARY): gam reconstructs the true interaction
    //     component to within 2.5% of its amplitude — the REML-achievable floor
    //     on this noiseless surface (see module docstring).
    assert!(
        rmse_gam <= recovery_bar,
        "gam fails to recover the true interaction surface: \
         rmse={rmse_gam:.5} > bar={recovery_bar:.5} (= 0.025 * range {truth_int_range:.4})"
    );

    // (4) MATCH-OR-BEAT: gam's recovery accuracy is no worse than mgcv's by
    //     more than 10%. mgcv is the baseline, not the target.
    assert!(
        rmse_gam <= 1.10 * rmse_mgcv,
        "gam's interaction recovery is worse than mgcv's baseline: \
         rmse_gam={rmse_gam:.5} > 1.10 * rmse_mgcv={rmse_mgcv:.5}"
    );
}
