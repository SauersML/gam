//! #1378 regression: the DEFAULT univariate thin-plate smooth `s(x, bs="tp")`
//! (no explicit `k`) must be row-permutation invariant.
//!
//! A GAM fit is a functional of the SET of (x, y) observations, so reordering
//! rows of byte-identical data must reproduce a byte-identical fit (up to
//! round-off). The reporter observed that the DEFAULT 1-D `bs="tp"` smooth was
//! NOT invariant: permuting rows moved the fitted curve by ~0.076 (≈3% of the
//! signal range), while `bs="cr"` / `bs="ps"` were ≤1e-14 on the same data.
//!
//! ROOT CAUSE: the generic spatial center heuristic over-sized the default 1-D
//! tp basis (≈62 coefficients at n=300 vs mgcv's modest k≈10). The two penalty
//! blocks (radial kernel + {1,x} nullspace) then left the REML ρ-surface with a
//! weakly-identified flat valley; the outer optimizer stalled on it at a point
//! that depended on the row order of the training data, so a pure permutation
//! moved λ̂ and thus the curve. Sizing the default 1-D tp basis down to a
//! modest, well-identified value restores an identified ρ-surface (the fix lives
//! in `term_builder.rs` `THIN_PLATE_1D_DEFAULT_BASIS_DIM`).
//!
//! This is the FIT-LEVEL invariance check (the companion unit test
//! `default_univariate_thinplate_basis_dim_is_modest` pins the sizing property
//! that this depends on). It fails on the oversized default and passes on the
//! modest default.

use gam::matrix::LinearOperator;
use gam::smooth::build_term_collection_design;
use gam::{
    FitConfig, FitResult, fit_from_formula, init_parallelism, load_csvwith_inferred_schema,
};
use ndarray::Array2;
use std::io::Write;

/// Deterministic LCG + Box-Muller so the test is bit-reproducible with no RNG
/// dev-dependency (same pattern as the #1271 tp test).
fn make_dataset(n: usize) -> (Vec<f64>, Vec<f64>) {
    let mut state: u64 = 0x1378_1378_dead_beef;
    let mut next_unit = || {
        state = state
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        ((state >> 11) as f64) / ((1u64 << 53) as f64)
    };
    let mut normal = move || {
        let u1 = next_unit().max(1e-12);
        let u2 = next_unit();
        (-2.0 * u1.ln()).sqrt() * (std::f64::consts::TAU * u2).cos()
    };
    let mut xs = Vec::with_capacity(n);
    let mut ys = Vec::with_capacity(n);
    for i in 0..n {
        let x = -3.0 + 6.0 * (i as f64) / ((n - 1) as f64);
        // A genuinely nonlinear signal so the smooth has real curvature to fit.
        let y = (1.5 * x).sin() + 0.3 * x + 0.20 * normal();
        xs.push(x);
        ys.push(y);
    }
    (xs, ys)
}

/// Write `(x, y)` rows (in the supplied order) to a temp CSV and return its path.
fn write_csv(tag: &str, xs: &[f64], ys: &[f64]) -> std::path::PathBuf {
    let path = std::env::temp_dir().join(format!("gam_1378_{tag}.csv"));
    let mut f = std::fs::File::create(&path).expect("create csv");
    writeln!(f, "x,y").unwrap();
    for (x, y) in xs.iter().zip(ys.iter()) {
        writeln!(f, "{x:.17e},{y:.17e}").unwrap();
    }
    path
}

/// Fit `formula` on the CSV at `path` and predict on `grid` (1-D x values).
/// Identity link ⇒ design·β is the predicted mean.
fn fit_and_predict(formula: &str, path: &std::path::Path, grid: &[f64]) -> Vec<f64> {
    let ds = load_csvwith_inferred_schema(path).expect("load csv");
    let col = ds.column_map();
    let x_idx = col["x"];
    let cfg = FitConfig {
        family: Some("gaussian".to_string()),
        ..FitConfig::default()
    };
    let result = fit_from_formula(formula, &ds, &cfg).expect("gam fit");
    let FitResult::Standard(fit) = result else {
        panic!("expected a standard GAM fit");
    };
    let mut design_grid = Array2::<f64>::zeros((grid.len(), ds.headers.len()));
    for (i, &x) in grid.iter().enumerate() {
        design_grid[[i, x_idx]] = x;
    }
    let design = build_term_collection_design(design_grid.view(), &fit.resolvedspec)
        .expect("rebuild design at prediction grid");
    design.design.apply(&fit.fit.beta).to_vec()
}

/// Maximum absolute difference between two equal-length prediction vectors.
fn max_abs_diff(a: &[f64], b: &[f64]) -> f64 {
    assert_eq!(a.len(), b.len(), "prediction length mismatch");
    a.iter()
        .zip(b.iter())
        .map(|(p, q)| (p - q).abs())
        .fold(0.0_f64, f64::max)
}

#[test]
fn default_univariate_tp_fit_is_row_permutation_invariant_1378() {
    init_parallelism();

    let n = 300usize;
    let (xs, ys) = make_dataset(n);

    // Signal range (for a relative tolerance on the prediction difference).
    let y_min = ys.iter().copied().fold(f64::INFINITY, f64::min);
    let y_max = ys.iter().copied().fold(f64::NEG_INFINITY, f64::max);
    let signal_range = (y_max - y_min).max(1e-300);

    // A fixed, deterministic permutation of the row indices (a "reversal-stride"
    // shuffle — no RNG). The permuted dataset is the SAME multiset of (x, y)
    // pairs, only reordered, so any correct GAM fit must reproduce the curve.
    let mut perm: Vec<usize> = (0..n).collect();
    perm.reverse();
    // Add a stride interleave on top of the reversal so the permutation is not a
    // trivial flip (defends against any accidental symmetry in the solver path).
    perm.rotate_left(37);
    let xs_perm: Vec<f64> = perm.iter().map(|&i| xs[i]).collect();
    let ys_perm: Vec<f64> = perm.iter().map(|&i| ys[i]).collect();

    let orig_csv = write_csv("orig", &xs, &ys);
    let perm_csv = write_csv("perm", &xs_perm, &ys_perm);

    // Common prediction grid spanning the data.
    let grid: Vec<f64> = (0..200)
        .map(|i| -3.0 + 6.0 * (i as f64) / 199.0)
        .collect();

    // DEFAULT univariate tp: NO explicit k. This is the path #1378 fixes.
    let tp = "y ~ s(x, bs=\"tp\")";
    let pred_orig = fit_and_predict(tp, &orig_csv, &grid);
    let pred_perm = fit_and_predict(tp, &perm_csv, &grid);
    let tp_drift = max_abs_diff(&pred_orig, &pred_perm);
    let tp_rel = tp_drift / signal_range;

    // Controls: cr / ps are exactly (or near-exactly) row-permutation invariant.
    let cr = "y ~ s(x, bs=\"cr\")";
    let cr_orig = fit_and_predict(cr, &orig_csv, &grid);
    let cr_perm = fit_and_predict(cr, &perm_csv, &grid);
    let cr_rel = max_abs_diff(&cr_orig, &cr_perm) / signal_range;

    eprintln!(
        "[#1378] default tp drift={tp_drift:.3e} (rel={tp_rel:.3e})  \
         cr control rel={cr_rel:.3e}  signal_range={signal_range:.4}"
    );

    // The cr control confirms the harness/data round-trip is itself invariant.
    assert!(
        cr_rel <= 1e-4,
        "cr control should be row-permutation invariant: rel drift={cr_rel:.3e}"
    );

    // PRIMARY claim: the DEFAULT univariate tp fit is row-permutation invariant.
    // Before the fix (oversized n-scaled default basis) this drift was ~3% of the
    // signal range; with the modest mgcv-sized default basis it collapses to
    // round-off. Tolerance: ≤ 1e-4 of the signal range.
    assert!(
        tp_rel <= 1e-4,
        "default univariate s(x, bs=\"tp\") fit is NOT row-permutation invariant: \
         max prediction drift {tp_drift:.3e} = {tp_rel:.3e} of signal range \
         (need ≤ 1e-4). The oversized default tp basis left a weakly-identified \
         flat REML ρ-valley whose optimizer landing point depends on row order (#1378)."
    );
}
