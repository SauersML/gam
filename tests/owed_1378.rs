//! Owed-work regression gate for issue #1378 — the default univariate thin-plate
//! smooth `s(x, bs="tp")` was NOT invariant to a pure row permutation of the
//! training data.
//!
//! ## The defect (now fixed)
//!
//! A GAM fit is a functional of the *unordered set* of `(x, y)` observations, so
//! reordering the rows of an identical dataset must reproduce the identical fit
//! (up to round-off). The local bases `bs="cr"` and `bs="ps"` honour this
//! bit-for-bit. The default `bs="tp"` did not: permuting the rows moved the
//! fitted curve by ~3% of the signal range (the issue measured a worst-case
//! prediction drift of 0.0756, ~1.5× the fit's own RMSE-to-truth).
//!
//! Two mechanisms, both fixed on `main`:
//!
//!   (a) `select_thin_plate_knots` (`src/terms/basis/duchon_thinplate.rs`) broke
//!       the greedy maximin tie (equal `min_dist2` to the chosen set, common on
//!       near-regular 1-D data) by ROW INDEX. A pure permutation then selected a
//!       different tied knot, yielding a different basis, conditioning, and REML
//!       λ̂. Fixed (`93f938970`) by breaking ties value-lexicographically — a
//!       pure function of the unordered data value set (the `value_less` closure,
//!       used for both the seed and every maximin step).
//!
//!   (b) The default 1-D tp basis dimension was n-scaled and oversized; the
//!       resulting REML ARC sub-problem could cost-stall NON-converged, leaving
//!       the selected λ̂ order-dependent. Fixed (`980725fc2`) by sizing the
//!       default univariate tp basis to the modest mgcv-style ceiling
//!       `THIN_PLATE_1D_DEFAULT_BASIS_DIM = 10` (`src/terms/term_builder.rs`).
//!
//! The knot-set invariance is pinned in-crate by
//! `terms::basis::duchon_thinplate::tests::knot_set_is_row_permutation_invariant_gh1378`.
//! This file is the end-to-end complement: it drives the PUBLIC fit path on the
//! issue's own reproducer geometry and asserts the fitted curve (predictions on
//! a fixed grid) is bit-stable under pure row permutations.

use csv::StringRecord;
use gam::matrix::LinearOperator;
use gam::smooth::build_term_collection_design;
use gam::{
    FitConfig, FitResult, encode_recordswith_inferred_schema, fit_from_formula, init_parallelism,
};
use ndarray::Array2;
use rand::SeedableRng;
use rand::rngs::StdRng;
use rand::seq::SliceRandom;
use rand_distr::{Distribution, Normal, Uniform};

/// Build the issue-#1378 dataset: `y = sin(2x) + 0.3x + N(0, 0.2)` on `n` points
/// with `x` drawn uniform on `[-2, 2]` and SORTED ascending (so the canonical row
/// order is the value order — exactly the regime where an index-based tie-break
/// silently masquerades as a value-based one until the rows are permuted).
fn make_data(n: usize, seed: u64) -> (Vec<f64>, Vec<f64>) {
    let mut rng = StdRng::seed_from_u64(seed);
    let unif = Uniform::new(-2.0_f64, 2.0).expect("uniform");
    let noise = Normal::new(0.0, 0.2).expect("normal");
    let mut x: Vec<f64> = (0..n).map(|_| unif.sample(&mut rng)).collect();
    x.sort_by(|a, b| a.partial_cmp(b).unwrap());
    let y: Vec<f64> = x
        .iter()
        .map(|&xi| (2.0 * xi).sin() + 0.3 * xi + noise.sample(&mut rng))
        .collect();
    (x, y)
}

/// Apply a row permutation to `(x, y)` and return the reordered columns.
fn permute(x: &[f64], y: &[f64], perm: &[usize]) -> (Vec<f64>, Vec<f64>) {
    (
        perm.iter().map(|&i| x[i]).collect(),
        perm.iter().map(|&i| y[i]).collect(),
    )
}

/// Fit `y ~ s(x, bs="<bs>")` on `(x, y)` and return the predicted curve on a
/// fixed grid in `[-1.8, 1.8]` (the issue's prediction grid). With identity link
/// the prediction is `design(grid) · beta`, so a row-order-invariant fit returns
/// bit-identical predictions for every permutation of the same data.
fn fit_predict(bs: &str, x: &[f64], y: &[f64], grid: &[f64]) -> Vec<f64> {
    let n = x.len();
    let headers: Vec<String> = ["x", "y"].into_iter().map(String::from).collect();
    let rows: Vec<StringRecord> = (0..n)
        .map(|i| StringRecord::from(vec![x[i].to_string(), y[i].to_string()]))
        .collect();
    let data = encode_recordswith_inferred_schema(headers, rows).expect("encode");
    let x_idx = data.column_map()["x"];

    let cfg = FitConfig {
        family: Some("gaussian".to_string()),
        ..FitConfig::default()
    };
    let formula = format!("y ~ s(x, bs=\"{bs}\")");
    let result = fit_from_formula(&formula, &data, &cfg).expect("fit");

    match result {
        FitResult::Standard(fit) => {
            // Identity-link Gaussian: prediction = design(grid) · beta.
            let mut grid_design = Array2::<f64>::zeros((grid.len(), data.headers.len()));
            for (row, &g) in grid.iter().enumerate() {
                grid_design[[row, x_idx]] = g;
            }
            let dm = build_term_collection_design(grid_design.view(), &fit.resolvedspec)
                .expect("rebuild design at grid");
            dm.design.apply(&fit.fit.beta).to_vec()
        }
        // A 1-D cubic Gaussian smooth (e.g. bs="cr") routes through the exact
        // O(n) state-space smoothing-spline scan, which carries its own exact
        // per-abscissa posterior rather than a dense design + beta. Predict the
        // posterior mean directly so the control base is exercised on the
        // representation the fit actually produced.
        FitResult::SplineScan(scan) => grid
            .iter()
            .map(|&g| scan.predict(g).expect("spline scan predict").0)
            .collect(),
        FitResult::ResidualCascade(cascade) => grid
            .iter()
            .map(|&g| cascade.predict(&[g]).expect("residual cascade predict").0)
            .collect(),
        _ => panic!("unexpected fit variant for {formula}"),
    }
}

/// The worst |prediction drift| of `bs` under a battery of pure row permutations,
/// reported as both an absolute value and a fraction of the baseline curve's
/// signal range.
fn worst_permutation_drift(bs: &str) -> (f64, f64) {
    let n = 300usize;
    let (x, y) = make_data(n, 7);
    let grid: Vec<f64> = (0..40).map(|i| -1.8 + 3.6 * i as f64 / 39.0).collect();

    let base = fit_predict(bs, &x, &y, &grid);
    let signal_range = {
        let (mut lo, mut hi) = (f64::INFINITY, f64::NEG_INFINITY);
        for &v in &base {
            lo = lo.min(v);
            hi = hi.max(v);
        }
        (hi - lo).max(1e-12)
    };

    let mut worst = 0.0_f64;
    for s in 0..6u64 {
        // A pure shuffle of the row order; the (x, y) value set is untouched.
        let mut perm: Vec<usize> = (0..n).collect();
        let mut rng = StdRng::seed_from_u64(200 + s);
        perm.shuffle(&mut rng);
        let (xp, yp) = permute(&x, &y, &perm);
        let pred = fit_predict(bs, &xp, &yp, &grid);
        let dp = base
            .iter()
            .zip(&pred)
            .fold(0.0_f64, |mx, (a, c)| mx.max((a - c).abs()));
        worst = worst.max(dp);
    }
    (worst, signal_range)
}

/// #1378: the default `s(x, bs="tp")` fit must be invariant to a pure row
/// permutation of the training data. A correct fit is a functional of the
/// unordered observation set, so the fitted curve must be bit-stable (up to
/// round-off) under any reordering of the rows.
#[test]
fn default_thin_plate_fit_is_row_permutation_invariant_1378() {
    init_parallelism();

    let (worst_tp, range_tp) = worst_permutation_drift("tp");
    eprintln!(
        "#1378 bs=tp row-permutation drift = {worst_tp:.3e} ({:.4}% of signal range {range_tp:.3e})",
        100.0 * worst_tp / range_tp
    );

    // Anchors: the value-based local bases were ALWAYS row-order invariant
    // (the issue measured cr = 0, ps ≤ 1e-14). They guard against a regression
    // in the fit/predict harness itself masking the tp result.
    let (worst_cr, _) = worst_permutation_drift("cr");
    let (worst_ps, _) = worst_permutation_drift("ps");
    eprintln!("#1378 bs=cr row-permutation drift = {worst_cr:.3e}");
    eprintln!("#1378 bs=ps row-permutation drift = {worst_ps:.3e}");
    assert!(
        worst_cr < 1e-10,
        "cr is value-anchored and must be row-permutation invariant; harness drift {worst_cr:.3e}"
    );
    assert!(
        worst_ps < 1e-10,
        "ps is value-anchored and must be row-permutation invariant; harness drift {worst_ps:.3e}"
    );

    // The fix: tp must now match cr/ps. Before #1378 this drift was ~0.0756
    // (~3% of the signal range); a value-lexicographic knot tie-break plus the
    // mgcv-sized default basis make the selected knot set, basis, and REML λ̂ a
    // pure function of the unordered data, so the curve is bit-stable. The bound
    // is the same machine-precision floor the local bases meet — NOT a loosened
    // "small enough" tolerance — because an order-invariant pipeline produces a
    // bit-identical λ̂ and hence a bit-identical curve.
    assert!(
        worst_tp < 1e-10 * range_tp.max(1.0),
        "default s(x, bs=\"tp\") is NOT row-permutation invariant: worst prediction drift \
         {worst_tp:.3e} ({:.4}% of signal range). The greedy maximin knot tie-break must break \
         ties value-lexicographically (duchon_thinplate.rs select_thin_plate_knots `value_less`, \
         #93f938970) and the default 1-D tp basis must be sized to \
         THIN_PLATE_1D_DEFAULT_BASIS_DIM=10 (term_builder.rs, #980725fc2) so REML λ̂ does not \
         drift with row order.",
        100.0 * worst_tp / range_tp
    );
}
