//! Bug hunt (#2127, deeper): the DEFAULT `sphere(lat, lon)` smooth
//! (`method=sobolev`, Wahba's reproducing kernel on S²) must be invariant under
//! *any* rotation of the sphere — the full rotation group SO(3), not merely the
//! one-parameter subgroup of rotations about the pole (longitude shifts).
//!
//! Wahba's reproducing kernel is a function of the geodesic angle alone
//! (`k(cos γ)`, `cos γ = uᵢ·u_c` for unit vectors), so the continuous estimator
//! is exactly SO(3)-invariant. The finite-center discretization inherits that
//! invariance **iff** the basis CENTERS rotate rigidly with the data. The first
//! `#2127` fix anchored a golden-angle (Fibonacci) center lattice's *longitude
//! origin* to the data's circular-mean longitude, which restores invariance to
//! a rotation about the pole but NOT to a general rotation: the lattice
//! latitudes stay pinned in the (lat, lon) frame, so a tilt (a rotation about an
//! in-plane axis) still slides the data across the stationary latitude rings and
//! reshapes the fit.
//!
//! The root-cause fix is genuine geodesic farthest-point sampling of the data:
//! the centers are a well-spread subset of the actual data rows, so under ANY
//! rotation R the SAME physical rows are selected, the centers rotate with the
//! data, every kernel entry `k(uᵢ·u_c)` is preserved, and the fit and every
//! prediction are invariant.
//!
//! This test applies a genuine TILT (a rotation about the x-axis) — and a
//! compound tilt+spin — to both the training data and the evaluation grid, and
//! asserts the default sobolev surface is unchanged to rounding. The
//! `method=harmonic` control (an SO(3)-closed spherical-harmonic space) is
//! exercised as the positive control.

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

type Vec3 = [f64; 3];

fn to_unit(lat_deg: f64, lon_deg: f64) -> Vec3 {
    let lat = lat_deg.to_radians();
    let lon = lon_deg.to_radians();
    let cos_lat = lat.cos();
    [cos_lat * lon.cos(), cos_lat * lon.sin(), lat.sin()]
}

/// Convert a unit vector back to (lat_deg, lon_deg) with lon in (-180, 180].
fn to_latlon_deg(u: Vec3) -> (f64, f64) {
    let lat = u[2].clamp(-1.0, 1.0).asin().to_degrees();
    let lon = u[1].atan2(u[0]).to_degrees();
    (lat, lon)
}

fn matvec(r: &[[f64; 3]; 3], u: Vec3) -> Vec3 {
    [
        r[0][0] * u[0] + r[0][1] * u[1] + r[0][2] * u[2],
        r[1][0] * u[0] + r[1][1] * u[1] + r[1][2] * u[2],
        r[2][0] * u[0] + r[2][1] * u[1] + r[2][2] * u[2],
    ]
}

/// Rotation about the x-axis by `t` radians — a genuine TILT of the poles that a
/// longitude-only fix cannot absorb.
fn rot_x(t: f64) -> [[f64; 3]; 3] {
    let (s, c) = t.sin_cos();
    [[1.0, 0.0, 0.0], [0.0, c, -s], [0.0, s, c]]
}

fn rot_z(t: f64) -> [[f64; 3]; 3] {
    let (s, c) = t.sin_cos();
    [[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]]
}

fn matmul(a: &[[f64; 3]; 3], b: &[[f64; 3]; 3]) -> [[f64; 3]; 3] {
    let mut out = [[0.0; 3]; 3];
    for i in 0..3 {
        for j in 0..3 {
            out[i][j] = (0..3).map(|k| a[i][k] * b[k][j]).sum();
        }
    }
    out
}

/// Known smooth scalar field on S², a genuine function of the physical point
/// (its unit vector), so the response is attached to the physical point and a
/// rotation of the frame does not change `y` — it only moves the coordinate
/// label.
fn truth_surface(u: Vec3) -> f64 {
    // Low-order harmonic-like field: `x·y + ½ z² - ⅓`, smooth on the whole sphere.
    u[0] * u[1] + 0.5 * u[2] * u[2] - 1.0 / 3.0
}

/// Fit `y ~ <formula>` on data rotated by `r` (a rigid rotation of the physical
/// points, relabeled into (lat, lon)) and return predictions on a fixed physical
/// interior grid rotated by the SAME `r`. A rotation-invariant smooth must return
/// the same values for every `r`.
fn fit_rotated(formula: &str, r: &[[f64; 3]; 3], seed: u64) -> Vec<f64> {
    let n = 300usize;
    let mut rng = StdRng::seed_from_u64(seed);
    let u_z = Uniform::new_inclusive(-1.0_f64, 1.0).expect("uniform z");
    let u_lon = Uniform::new(-180.0_f64, 180.0_f64).expect("uniform lon");
    let noise = Normal::new(0.0, 0.05).expect("normal");

    let mut lat = Vec::with_capacity(n);
    let mut lon = Vec::with_capacity(n);
    let mut y = Vec::with_capacity(n);
    for _ in 0..n {
        // Physical point (uniform on S²), its response, then rotate the LABEL.
        let z: f64 = u_z.sample(&mut rng);
        let lon_deg: f64 = u_lon.sample(&mut rng);
        let lat_deg = z.asin().to_degrees();
        let p = to_unit(lat_deg, lon_deg);
        let resp = truth_surface(p) + noise.sample(&mut rng);
        let (rlat, rlon) = to_latlon_deg(matvec(r, p));
        lat.push(rlat);
        lon.push(rlon);
        y.push(resp);
    }

    let headers: Vec<String> = ["lat", "lon", "y"].into_iter().map(String::from).collect();
    let rows: Vec<StringRecord> = (0..n)
        .map(|i| {
            StringRecord::from(vec![
                lat[i].to_string(),
                lon[i].to_string(),
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

    // Fixed physical interior grid (away from the poles); rotate every grid point
    // by the SAME `r` so we predict at the rotated image of a fixed physical set.
    let g = 14usize;
    let lat_at = |i: usize| -70.0 + 140.0 * i as f64 / (g as f64 - 1.0);
    let lon_at = |j: usize| -170.0 + 340.0 * j as f64 / (g as f64 - 1.0);
    let m = g * g;
    let mut grid = Array2::<f64>::zeros((m, data.headers.len()));
    let mut t = 0usize;
    for i in 0..g {
        for j in 0..g {
            let p = to_unit(lat_at(i), lon_at(j));
            let (rlat, rlon) = to_latlon_deg(matvec(r, p));
            grid[[t, lat_idx]] = rlat;
            grid[[t, lon_idx]] = rlon;
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

fn worst_drift(base: &[f64], rotated: &[f64]) -> f64 {
    base.iter()
        .zip(rotated)
        .fold(0.0_f64, |mx, (a, c)| mx.max((a - c).abs()))
}

/// PRIMARY: the default sobolev sphere smooth must be invariant under a general
/// SO(3) rotation, including a tilt about an in-plane axis (not just a rotation
/// about the pole).
#[test]
fn default_sphere_sobolev_smooth_is_general_rotation_invariant() {
    init_parallelism();
    let seed = 7u64;
    let identity = [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]];
    let base = fit_rotated("y ~ sphere(lat, lon)", &identity, seed);
    let range = signal_range(&base);

    // Genuine tilts (rotation about the x-axis) and compound tilt+spin. None of
    // these is a pure rotation about the pole, so a longitude-anchored center
    // lattice cannot absorb them.
    let rotations: Vec<(&str, [[f64; 3]; 3])> = vec![
        ("tilt_x(0.6)", rot_x(0.6)),
        ("tilt_x(1.2)", rot_x(1.2)),
        ("spin_z(1.0)*tilt_x(0.9)", matmul(&rot_z(1.0), &rot_x(0.9))),
        (
            "tilt_x(-0.7)*spin_z(2.0)",
            matmul(&rot_x(-0.7), &rot_z(2.0)),
        ),
    ];

    let mut worst = 0.0_f64;
    let mut worst_name = "";
    for (name, r) in &rotations {
        let rotated = fit_rotated("y ~ sphere(lat, lon)", r, seed);
        let drift = worst_drift(&base, &rotated);
        eprintln!(
            "[sobolev] rotation {name}: worst |pred drift| = {drift:.3e} \
             ({:.3}% of signal range)",
            100.0 * drift / range
        );
        if drift > worst {
            worst = drift;
            worst_name = name;
        }
    }

    eprintln!(
        "[sobolev] general-rotation invariance: worst |pred drift| = {worst:.3e} \
         ({:.3}% of signal range, at {worst_name})",
        100.0 * worst / range
    );

    assert!(
        worst < 1e-6 * range.max(1.0),
        "default sphere(sobolev) smooth is NOT invariant under a general SO(3) rotation: \
         worst prediction drift {worst:.3e} ({:.3}% of signal range) at {worst_name}. Wahba's \
         kernel depends only on cos(geodesic angle) and is intrinsically rotation invariant, so \
         the leak is a frame-anchored center set: the centers must be a rotation-equivariant \
         function of the data (geodesic farthest-point sampling) so they rotate WITH the data \
         under any rotation, not just a rotation about the pole.",
        100.0 * worst / range
    );
}

/// POSITIVE CONTROL: the harmonic sphere smooth spans an SO(3)-closed space and
/// is already invariant under a general rotation.
#[test]
fn harmonic_sphere_smooth_is_general_rotation_invariant_control() {
    init_parallelism();
    let seed = 7u64;
    let identity = [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]];
    let formula = "y ~ sphere(lat, lon, method=harmonic, max_degree=4)";
    let base = fit_rotated(formula, &identity, seed);
    let range = signal_range(&base);

    let rotations: Vec<(&str, [[f64; 3]; 3])> = vec![
        ("tilt_x(0.6)", rot_x(0.6)),
        ("tilt_x(1.2)", rot_x(1.2)),
        ("spin_z(1.0)*tilt_x(0.9)", matmul(&rot_z(1.0), &rot_x(0.9))),
    ];
    let mut worst = 0.0_f64;
    for (name, r) in &rotations {
        let rotated = fit_rotated(formula, r, seed);
        let drift = worst_drift(&base, &rotated);
        eprintln!(
            "[harmonic-control] rotation {name}: worst |pred drift| = {drift:.3e} \
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
