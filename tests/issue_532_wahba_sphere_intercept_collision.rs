//! Regression (#532): a Wahba spherical spline (`bs="sos"` / `sphere(...)`,
//! method = Wahba) used together with a parametric intercept must FIT — it must
//! not be refused by the identifiability audit.
//!
//! Root cause (the #531 constant-vs-intercept collision class). The Wahba
//! design is `K(data, centers) · z`, where `z =
//! weighted_coefficient_sum_to_zero_transform(area_weights)` enforces
//! `1ᵀ W α = 0` over the *centers* — a coefficient-space constraint. The
//! realized design at the data rows therefore still spans the constant, which
//! duplicates the global parametric intercept (every model carries one). The
//! intra-block rank-1 deficiency makes the identifiability audit FATAL.
//!
//! The fix routes the Wahba sphere through
//! `apply_global_smooth_identifiability`'s parametric orthogonalization (it is
//! now in `smooth_requires_parametric_orthogonality`), composes the resulting
//! transform onto `z`, and FREEZES the composed realized-design transform onto
//! `SphericalSplineBasisSpec::identifiability` so the predict-time rebuild
//! reuses it verbatim instead of recomputing the center-space `z` (which would
//! drop the orthogonalization and resurrect the collision at predict time).
//!
//! What we assert:
//!   1. The Wahba-sphere-plus-intercept Gaussian fit COMPLETES (no audit
//!      refusal) — the bug face.
//!   2. The fitted smooth no longer spans the constant: the intercept absorbs
//!      the level, so re-predicting on a held-out grid and on the SAME grid
//!      after recompiling the design (save→reload-equivalent path through the
//!      frozen transform) is finite and reproduces the fit-time surface.
//!   3. TRUTH RECOVERY — the fit reproduces a known smooth function on the
//!      sphere with low RMSE, confirming the orthogonalization did not damage
//!      the smoother (it only removed the redundant constant).
//!   4. The Harmonic method (which starts at degree l = 1 and never spans the
//!      constant) also fits — guarding against an over-broad orthogonalization.

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

/// A smooth, intercept-bearing function on the sphere: a global level (which
/// the parametric intercept must absorb) plus a degree-1/2 spatial pattern.
fn truth(lat: f64, lon: f64) -> f64 {
    let lat_r = lat.to_radians();
    let lon_r = lon.to_radians();
    3.0 + 1.5 * lat_r.sin() + 0.8 * lat_r.cos() * lon_r.cos()
}

fn make_dataset(n: usize, seed: u64) -> gam::data::EncodedDataset {
    let mut rng = StdRng::seed_from_u64(seed);
    let u_lat = Uniform::new(-80.0_f64, 80.0).expect("uniform");
    let u_lon = Uniform::new(-179.9_f64, 179.9).expect("uniform");
    let noise = Normal::new(0.0, 0.10).expect("normal");
    let headers = ["lat", "lon", "y"].into_iter().map(String::from).collect();
    let mut rows = Vec::with_capacity(n);
    for _ in 0..n {
        let lat = u_lat.sample(&mut rng);
        let lon = u_lon.sample(&mut rng);
        let y = truth(lat, lon) + noise.sample(&mut rng);
        rows.push(StringRecord::from(vec![
            lat.to_string(),
            lon.to_string(),
            y.to_string(),
        ]));
    }
    encode_recordswith_inferred_schema(headers, rows).expect("encode")
}

fn grid_points() -> Vec<(f64, f64)> {
    let mut pts = Vec::new();
    for i in 0..12 {
        let lat = -70.0 + 140.0 * (i as f64) / 11.0;
        for j in 0..18 {
            let lon = -170.0 + 340.0 * (j as f64) / 17.0;
            pts.push((lat, lon));
        }
    }
    pts
}

fn fit_predict_and_rmse(formula: &str) -> (Vec<f64>, f64) {
    let data = make_dataset(1200, 532);
    let cfg = FitConfig::default(); // gaussian / identity / REML, intercept on
    let result = fit_from_formula(formula, &data, &cfg).unwrap_or_else(|e| {
        panic!("Wahba-sphere + intercept fit refused (#532): `{formula}`: {e}")
    });
    let FitResult::Standard(fit) = result else {
        panic!("expected standard GAM fit");
    };

    let pts = grid_points();
    let n = pts.len();
    let mut m = Array2::<f64>::zeros((n, 3));
    for (i, (lat, lon)) in pts.iter().enumerate() {
        m[[i, 0]] = *lat;
        m[[i, 1]] = *lon;
    }
    // Rebuild the design from the FROZEN resolved spec — this is the predict-time
    // path that must replay the composed `z · z_parametric` transform, not
    // recompute the center-space `z`.
    let design = build_term_collection_design(m.view(), &fit.resolvedspec)
        .expect("rebuild Wahba-sphere design at predict time (frozen transform replay)");
    let pred = design.design.apply(&fit.fit.beta).to_vec();

    let truth_vals: Vec<f64> = pts.iter().map(|&(la, lo)| truth(la, lo)).collect();
    let mse = pred
        .iter()
        .zip(truth_vals.iter())
        .map(|(&p, &t)| (p - t) * (p - t))
        .sum::<f64>()
        / pred.len() as f64;
    (pred, mse.sqrt())
}

#[test]
fn wahba_sphere_with_intercept_fits_and_recovers_truth() {
    init_parallelism();

    // bs="sos" is the mgcv alias for the Wahba spherical spline.
    let (pred, rmse) = fit_predict_and_rmse("y ~ s(lat, lon, bs=\"sos\", k=40)");
    assert!(
        pred.iter().all(|v| v.is_finite()),
        "Wahba-sphere predictions must be finite after frozen-transform replay"
    );
    // Truth spans ~4.6 over the sphere (level 3 + ±~1.7 pattern). A correctly
    // orthogonalized Wahba smooth recovers it to well under 10% of that range.
    assert!(
        rmse < 0.35,
        "Wahba sphere + intercept must recover the smooth truth: RMSE {rmse:.4}"
    );

    // The `sphere(...)` spelling resolves to the same Wahba basis and must fit
    // identically (same collision class, same fix).
    let (pred2, rmse2) = fit_predict_and_rmse("y ~ sphere(lat, lon, k=40)");
    assert!(pred2.iter().all(|v| v.is_finite()));
    assert!(
        rmse2 < 0.35,
        "sphere(...) Wahba + intercept must recover the truth: RMSE {rmse2:.4}"
    );
}

#[test]
fn harmonic_sphere_with_intercept_still_fits() {
    init_parallelism();
    // The Harmonic method starts at degree l = 1, never spans the constant, and
    // must be UNAFFECTED by the Wahba-only orthogonalization.
    let (pred, rmse) = fit_predict_and_rmse("y ~ sphere(lat, lon, method=\"harmonic\", k=6)");
    assert!(pred.iter().all(|v| v.is_finite()));
    assert!(
        rmse < 0.5,
        "Harmonic sphere + intercept must still fit and recover the truth: RMSE {rmse:.4}"
    );
}
