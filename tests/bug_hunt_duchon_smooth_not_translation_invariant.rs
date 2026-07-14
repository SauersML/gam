//! Bug hunt: the 2-D Duchon spatial smooth (`duchon(x, z)`, mgcv `bs="ds"`) is
//! NOT invariant to a pure covariate translation `(x, z) -> (x + b, z + b)`.
//!
//! A Duchon / thin-plate-equivalent radial smooth is built from a radial kernel
//! that reads only coordinate *differences* `data - center`, so the kernel block
//! is translation-invariant by construction. The polynomial NULL SPACE, however
//! -- the affine block `P = {1, x, z}` and the side-condition `P(centers)' a = 0`
//! -- is assembled at the *absolute* coordinate (see
//! `build_duchon_basis_uncached` -> `polynomial_block_from_order(data, ...)` in
//! `src/terms/basis/duchon_thinplate.rs:208,262`, and the raw monomials in
//! `monomial_basis_block` / `polynomial_block_from_order`, which use
//! `points[[row, axis]]` with no centering).
//!
//! The model space `span{1, x, z}` equals `span{1, x+b, z+b}`, so a correct fit
//! is mathematically identical under translation. But at a large offset the
//! `{1, x, z}` columns become near-collinear (a constant plus a huge near-constant
//! ramp), the design ill-conditions, and REML lambda-selection lands in a
//! different basin -- exactly the defect class fixed for the 1-D / general
//! `bs="tp"` ThinPlate path in #1269 (which now subtracts the knot-cloud mean in
//! `create_thin_plate_spline_basis_scaledwithworkspace`,
//! `duchon_thinplate.rs:1279`). The Duchon `bs="ds"` path was left UNCENTERED and
//! still leaks the absolute coordinate.
//!
//! This test fits the SAME data on `(x, z)` vs `(x + b, z + b)` and asserts the
//! recovered surface (predictions on a shifted-back grid) and the smoothing
//! amount (EDF) match to rounding. They do not.

use csv::StringRecord;
use gam::matrix::LinearOperator;
use gam::smooth::build_term_collection_design;
use gam::{
    FitConfig, FitResult, encode_recordswith_inferred_schema, fit_from_formula, init_parallelism,
};
use ndarray::Array2;
use rand::SeedableRng;
use rand::rngs::StdRng;
use rand_distr::{Distribution, Normal, Uniform};

/// Known smooth 2-D surface: a single broad Gaussian bump over the unit square.
fn truth_surface(x: f64, z: f64) -> f64 {
    let d2 = (x - 0.5).powi(2) + (z - 0.5).powi(2);
    1.5 * (-d2 / (2.0 * 0.25_f64 * 0.25)).exp()
}

/// Fit `y ~ duchon(x, z, k=49)` on covariates offset by `b` in BOTH axes and
/// return (EDF of the smooth, predictions on the interior grid mapped through
/// the same offset). The truth is a function of the *unshifted* coordinate, so
/// the data `y` is identical across offsets -- only the covariate labels move.
fn fit_offset(b: f64, seed: u64) -> (f64, Vec<f64>) {
    let n = 300usize;
    let mut rng = StdRng::seed_from_u64(seed);
    let unit = Uniform::new(0.0_f64, 1.0).expect("uniform");
    let noise = Normal::new(0.0, 0.10).expect("normal");

    let mut x = Vec::with_capacity(n);
    let mut z = Vec::with_capacity(n);
    let mut y = Vec::with_capacity(n);
    for _ in 0..n {
        let xi = unit.sample(&mut rng);
        let zi = unit.sample(&mut rng);
        x.push(xi);
        z.push(zi);
        y.push(truth_surface(xi, zi) + noise.sample(&mut rng));
    }

    let headers: Vec<String> = ["x", "z", "y"].into_iter().map(String::from).collect();
    let rows: Vec<StringRecord> = (0..n)
        .map(|i| {
            StringRecord::from(vec![
                (x[i] + b).to_string(),
                (z[i] + b).to_string(),
                y[i].to_string(),
            ])
        })
        .collect();
    let data = encode_recordswith_inferred_schema(headers, rows).expect("encode");
    let col = data.column_map();
    let x_idx = col["x"];
    let z_idx = col["z"];

    let cfg = FitConfig {
        family: Some("gaussian".to_string()),
        ..FitConfig::default()
    };
    let result = fit_from_formula("y ~ duchon(x, z, k=49)", &data, &cfg).expect("duchon fit");
    let FitResult::Standard(fit) = result else {
        panic!("expected standard fit");
    };

    // EDF of the (single) smooth term.
    let unified = &fit.fit;
    let design = &fit.design;
    let mut penalty_cursor = 0usize;
    for (_n, _r) in &design.random_effect_ranges {
        penalty_cursor += 1;
    }
    let mut edf = f64::NAN;
    for term in &design.smooth.terms {
        let k = term.active_penalties.len();
        edf = unified.per_term_edf(term.coeff_range.clone(), penalty_cursor, k);
        penalty_cursor += k;
    }

    // Predictions on a fixed interior grid of the UNSHIFTED coordinate, mapped
    // through the same `+b` offset used to fit. With identity link the surface
    // is design * beta; a translation-invariant smooth must return the same
    // surface for every `b`.
    let g = 20usize;
    let coord = |i: usize| 0.05 + 0.90 * i as f64 / (g as f64 - 1.0);
    let m = g * g;
    let mut grid = Array2::<f64>::zeros((m, data.headers.len()));
    let mut t = 0usize;
    for i in 0..g {
        for j in 0..g {
            grid[[t, x_idx]] = coord(i) + b;
            grid[[t, z_idx]] = coord(j) + b;
            t += 1;
        }
    }
    let dm = build_term_collection_design(grid.view(), &fit.resolvedspec)
        .expect("rebuild duchon design at grid");
    let preds: Vec<f64> = dm.design.apply(&fit.fit.beta).to_vec();
    (edf, preds)
}

#[test]
fn duchon_smooth_is_translation_invariant() {
    init_parallelism();
    let seed = 7u64;
    let (edf0, pred0) = fit_offset(0.0, seed);

    // Translations that leave the model space unchanged but blow up the absolute
    // coordinate of the polynomial null-space block.
    let offsets = [10.0_f64, 100.0, 1000.0];

    let mut worst_pred = 0.0_f64;
    let mut worst_edf = 0.0_f64;
    let mut worst_off = 0.0_f64;
    let signal_range = {
        let (mut lo, mut hi) = (f64::INFINITY, f64::NEG_INFINITY);
        for &v in &pred0 {
            lo = lo.min(v);
            hi = hi.max(v);
        }
        (hi - lo).max(1e-12)
    };

    for &b in &offsets {
        let (edf_b, pred_b) = fit_offset(b, seed);
        let dp = pred0
            .iter()
            .zip(&pred_b)
            .fold(0.0_f64, |mx, (a, c)| mx.max((a - c).abs()));
        let de = (edf_b - edf0).abs();
        if dp > worst_pred {
            worst_pred = dp;
            worst_off = b;
        }
        worst_edf = worst_edf.max(de);
        eprintln!(
            "duchon translation b={b}: EDF {edf0:.4} -> {edf_b:.4} (|d|={de:.3e}), \
             worst |pred drift| = {dp:.3e} ({:.3}% of signal range)",
            100.0 * dp / signal_range
        );
    }

    eprintln!(
        "duchon translation-invariance: worst |pred drift| = {worst_pred:.3e} \
         ({:.3}% of signal range, at offset {worst_off}), worst |EDF drift| = {worst_edf:.3e}",
        100.0 * worst_pred / signal_range
    );

    // A translation-invariant 2-D Duchon smooth would match to rounding. Anything
    // approaching a percent of the signal range means the absolute coordinate is
    // leaking into the polynomial null-space block and reshaping the fit.
    assert!(
        worst_pred < 1e-6 * signal_range.max(1.0),
        "Duchon bs=ds smooth is NOT translation-invariant: worst prediction drift {worst_pred:.3e} \
         ({:.3}% of signal range) at offset {worst_off}; the polynomial null-space block is built \
         at absolute coordinates (duchon_thinplate.rs polynomial_block_from_order) and never \
         centered like the #1269 ThinPlate fix, so REML lambda drifts under x -> x + b.",
        100.0 * worst_pred / signal_range
    );
    assert!(
        worst_edf < 1e-4,
        "Duchon bs=ds EDF is NOT translation-invariant: worst EDF drift {worst_edf:.3e}."
    );
}
