//! Bug hunt (#2127): the DEFAULT `sphere(lat, lon)` smooth (`method=sobolev`,
//! Wahba's reproducing kernel on S²) is NOT invariant to the arbitrary choice of
//! longitude origin — i.e. it is not rotation-invariant about the pole.
//!
//! Wahba's reproducing kernel is a function of the GEODESIC ANGLE between two
//! points on S² alone (`k(cos γ)`, with `cos γ` = the dot product of the two
//! unit vectors), so it is intrinsically invariant under any rotation of S²,
//! including a rigid longitude shift `lon -> lon + Δ` (a rotation about the
//! pole). The `spherical_wahba_kernel_matrix_cpu` evaluation is indeed built
//! from `cos γ` and is invariant. The defect is upstream, in how the basis
//! CENTERS are chosen: `select_spherical_farthest_point_centers`
//! (`crates/gam-terms/src/basis/workspace_cache.rs`) laid down a fixed
//! golden-angle (Fibonacci) lattice whose longitudes are `i · golden_angle`,
//! i.e. a FRAME-ANCHORED lattice with the longitude origin baked in. The centers
//! did not rotate with the data, so a common longitude shift applied to BOTH the
//! training data and the evaluation grid moved the data relative to the
//! STATIONARY centers, changed every data-to-center geodesic angle, and reshaped
//! the fitted surface — even though the smooth is advertised as rotation
//! invariant with no pole artefacts.
//!
//! The `method=harmonic` control (spherical-harmonic basis, no data-dependent
//! centers, Laplace-Beltrami penalty) spans an SO(3)-closed function space and
//! IS invariant; it is exercised as a positive control below.
//!
//! FIX: anchor the golden-angle spiral's longitude origin to the data's circular
//! mean longitude, so the whole center cloud rotates WITH the data. Because the
//! circular mean obeys `mean(lon + Δ) = mean(lon) + Δ`, every center longitude
//! shifts by the same Δ under a pole rotation, the `cos γ` arguments the kernel
//! sees are unchanged, and predictions become invariant to the longitude origin.
//!
//! This test fits the SAME responses on `lon` vs `lon + Δ` (with the evaluation
//! grid shifted by the same Δ) for the DEFAULT sobolev smooth and asserts the
//! recovered surface matches to rounding — the positive control that the
//! harmonic smooth already satisfies. Before the fix the sobolev arm drifts by a
//! large fraction of the signal range; after the fix it matches to fp noise.

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

/// Wrap a longitude in degrees into (-180, 180].
fn wrap_lon(lon: f64) -> f64 {
    let mut v = (lon + 180.0).rem_euclid(360.0) - 180.0;
    if v <= -180.0 {
        v += 360.0;
    }
    v
}

/// Known smooth scalar field on S², a genuine function of both latitude and
/// longitude so the fit is non-degenerate and longitude-sensitive. The response
/// is attached to the physical point; a longitude relabel `lon -> lon + Δ` does
/// not change `y`, it only moves the coordinate label — exactly a rotation of
/// the frame about the pole.
fn truth_surface(lat_deg: f64, lon_deg: f64) -> f64 {
    let lat = lat_deg.to_radians();
    let lon = lon_deg.to_radians();
    lat.sin() * (2.0 * lon).cos() + 0.5 * lat.cos() * lon.sin()
}

/// Fit `y ~ <spec>` on data whose longitudes are all shifted by `shift_deg`
/// degrees (a rotation about the pole), and return predictions on a fixed
/// interior grid whose longitudes are shifted by the SAME amount. The truth is a
/// function of the UNSHIFTED coordinate, so `y` is identical across shifts —
/// only the longitude labels move. A rotation-invariant smooth must return the
/// same surface for every shift.
fn fit_shifted(formula: &str, shift_deg: f64, seed: u64) -> Vec<f64> {
    let n = 240usize;
    let mut rng = StdRng::seed_from_u64(seed);
    // Uniform on the sphere: z ~ U(-1, 1) -> lat = asin(z); lon ~ U(-180, 180).
    let u_z = Uniform::new_inclusive(-1.0_f64, 1.0).expect("uniform z");
    let u_lon = Uniform::new(-180.0_f64, 180.0_f64).expect("uniform lon");
    let noise = Normal::new(0.0, 0.05).expect("normal");

    let mut lat = Vec::with_capacity(n);
    let mut lon = Vec::with_capacity(n);
    let mut y = Vec::with_capacity(n);
    for _ in 0..n {
        let z: f64 = u_z.sample(&mut rng);
        let lat_deg = z.asin().to_degrees();
        let lon_deg: f64 = u_lon.sample(&mut rng);
        lat.push(lat_deg);
        lon.push(lon_deg);
        y.push(truth_surface(lat_deg, lon_deg) + noise.sample(&mut rng));
    }

    let headers: Vec<String> = ["lat", "lon", "y"].into_iter().map(String::from).collect();
    let rows: Vec<StringRecord> = (0..n)
        .map(|i| {
            StringRecord::from(vec![
                lat[i].to_string(),
                wrap_lon(lon[i] + shift_deg).to_string(),
                y[i].to_string(),
            ])
        })
        .collect();
    let data = encode_recordswith_inferred_schema(headers, rows).expect("encode");
    let col = data.column_map();
    let lat_idx = col["lat"];
    let lon_idx = col["lon"];

    let cfg = FitConfig {
        family: Some("gaussian".to_string()),
        ..FitConfig::default()
    };
    let result = fit_from_formula(formula, &data, &cfg).expect("sphere fit");
    let FitResult::Standard(fit) = result else {
        panic!("expected standard fit");
    };

    // Predictions on a fixed interior grid (away from the poles), longitudes
    // shifted by the SAME amount used to fit. With the identity link the fitted
    // surface is design * beta; a rotation-invariant smooth must return the same
    // values for every shift.
    let g = 16usize;
    let lat_at = |i: usize| -70.0 + 140.0 * i as f64 / (g as f64 - 1.0);
    let lon_at = |j: usize| -170.0 + 340.0 * j as f64 / (g as f64 - 1.0);
    let m = g * g;
    let mut grid = Array2::<f64>::zeros((m, data.headers.len()));
    let mut t = 0usize;
    for i in 0..g {
        for j in 0..g {
            grid[[t, lat_idx]] = lat_at(i);
            grid[[t, lon_idx]] = wrap_lon(lon_at(j) + shift_deg);
            t += 1;
        }
    }
    let dm = build_term_collection_design(grid.view(), &fit.resolvedspec)
        .expect("rebuild sphere design at grid");
    dm.design.apply(&fit.fit.beta).to_vec()
}

fn signal_range(v: &[f64]) -> f64 {
    let (mut lo, mut hi) = (f64::INFINITY, f64::NEG_INFINITY);
    for &x in v {
        lo = lo.min(x);
        hi = hi.max(x);
    }
    (hi - lo).max(1e-12)
}

fn worst_drift(base: &[f64], shifted: &[f64]) -> f64 {
    base.iter()
        .zip(shifted)
        .fold(0.0_f64, |mx, (a, c)| mx.max((a - c).abs()))
}

/// PRIMARY: the default sobolev sphere smooth must be invariant to the choice of
/// longitude origin (a rotation about the pole).
#[test]
fn default_sphere_sobolev_smooth_is_rotation_invariant() {
    init_parallelism();
    let seed = 11u64;
    let base = fit_shifted("y ~ sphere(lat, lon)", 0.0, seed);
    let range = signal_range(&base);

    // Longitude shifts that leave the intrinsic geometry unchanged but move the
    // data relative to a frame-anchored center lattice.
    let shifts = [37.0_f64, 90.0, 143.0, 211.0];
    let mut worst = 0.0_f64;
    let mut worst_shift = 0.0_f64;
    for &d in &shifts {
        let shifted = fit_shifted("y ~ sphere(lat, lon)", d, seed);
        let drift = worst_drift(&base, &shifted);
        eprintln!(
            "[sobolev] longitude shift Δ={d}°: worst |pred drift| = {drift:.3e} \
             ({:.3}% of signal range)",
            100.0 * drift / range
        );
        if drift > worst {
            worst = drift;
            worst_shift = d;
        }
    }

    eprintln!(
        "[sobolev] rotation-invariance: worst |pred drift| = {worst:.3e} \
         ({:.3}% of signal range, at Δ={worst_shift}°)",
        100.0 * worst / range
    );

    assert!(
        worst < 1e-6 * range.max(1.0),
        "default sphere(sobolev) smooth is NOT rotation-invariant about the pole: \
         worst prediction drift {worst:.3e} ({:.3}% of signal range) at longitude shift \
         Δ={worst_shift}°. Wahba's kernel depends only on cos(geodesic angle) and is \
         intrinsically rotation invariant, so the leak is the frame-anchored golden-angle \
         center lattice (select_spherical_farthest_point_centers), whose longitudes must be \
         anchored to the data's circular mean so the centers rotate with the data.",
        100.0 * worst / range
    );
}

/// POSITIVE CONTROL: the harmonic sphere smooth already spans an SO(3)-closed
/// space and is invariant to the longitude origin; it must pass both before and
/// after the sobolev fix.
#[test]
fn harmonic_sphere_smooth_is_rotation_invariant_control() {
    init_parallelism();
    let seed = 11u64;
    let formula = "y ~ sphere(lat, lon, method=harmonic, max_degree=4)";
    let base = fit_shifted(formula, 0.0, seed);
    let range = signal_range(&base);

    let shifts = [37.0_f64, 90.0, 143.0, 211.0];
    let mut worst = 0.0_f64;
    for &d in &shifts {
        let shifted = fit_shifted(formula, d, seed);
        let drift = worst_drift(&base, &shifted);
        eprintln!(
            "[harmonic-control] longitude shift Δ={d}°: worst |pred drift| = {drift:.3e} \
             ({:.3}% of signal range)",
            100.0 * drift / range
        );
        worst = worst.max(drift);
    }

    // The harmonic refit is an independent REML run over a bit-different rotated
    // design, so the residual is bounded by REML convergence, not f64 alone.
    assert!(
        worst < 5e-3 * range.max(1.0),
        "harmonic sphere smooth (positive control) should be rotation-invariant: \
         worst prediction drift {worst:.3e}"
    );
}
