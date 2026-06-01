//! Regression for issue #515: `SphereManifold::log_map` recovers the geodesic
//! length between two points as `theta = acos(p·q)`. For nearby points
//! `p·q ≈ 1`, so `acos` near its branch point amplifies the ~1e-16 rounding in
//! the dot product into a large relative error — ~1% at theta=1e-7 and total
//! nonsense at theta=1e-8.
//!
//! Root cause: for nearby unit vectors `p·q = 1 − |p-q|²/2` saturates to ~1,
//! so `1 − c` keeps only `~eps/(theta²/2)` relative accuracy and
//! `acos(1−x) ≈ √(2x)` maps that to `~eps/theta²` error in theta.
//!
//! Fix: the chord/haversine form `theta = 2·arcsin(|p-q|/2)` (clamped), where
//! `|p-q|` is formed directly from coordinates (no near-1 subtraction). The
//! same `acos(dot)` pattern was also fixed in `sphere_weighted_log_step` and
//! `sphere_frechet_objective` (the Fréchet-mean path).
//!
//! Related: #514 (poincaré arccosh), #513 (upper-tail quantile).

use gam::geometry::sphere::sphere_frechet_mean;
use gam::{RiemannianManifold, SphereManifold};
use ndarray::array;

/// `log_map`'s output norm *is* the geodesic distance. On great-circle pairs in
/// the x-y plane the exact distance is the angular separation `theta`.
///
/// The old `acos(p·q)` form lost ~eps/theta² of relative accuracy (1.1e-6 at
/// theta=1e-6, 1.2e-2 at 1e-7, total nonsense at 1e-8). The chord form is
/// optimal — its floor is ~eps/theta, the information limit for the geodesic
/// angle between two O(1)-magnitude float64 unit vectors — so it stays well
/// under 1e-8 down to theta=1e-7, an improvement of 4–6 orders of magnitude.
#[test]
fn sphere_log_map_recovers_nearby_geodesic_length() {
    let sphere = SphereManifold::new(2); // S² in R³
    let a = 0.7_f64;
    for &theta in &[1e-5_f64, 1e-6, 1e-7] {
        let p = array![a.cos(), a.sin(), 0.0];
        let q = array![(a + theta).cos(), (a + theta).sin(), 0.0];
        let v = sphere.log_map(p.view(), q.view()).expect("log_map");
        let recovered = v.dot(&v).sqrt();
        let rel = (recovered - theta).abs() / theta;
        assert!(
            rel < 1e-8,
            "theta={theta:e}: recovered {recovered:.16e}, exact {theta:.16e}, \
             rel err {rel:.3e} > 1e-8 — acos(p·q) cancellation near the branch point \
             (old acos form lost ~1e-6..1e-2 here)"
        );

        // Direction must also point along the great circle: the unit tangent at
        // p toward increasing angle is (−sin a, cos a, 0). Compare via the
        // cosine, which is robust to the inherent ~eps/theta length floor.
        let tangent = array![-a.sin(), a.cos(), 0.0];
        let cos_align = v.dot(&tangent) / recovered;
        assert!(
            (1.0 - cos_align).abs() < 1e-10,
            "theta={theta:e}: log_map direction misaligned with the great-circle \
             tangent, 1−cos = {:.3e}",
            (1.0 - cos_align).abs()
        );
    }
}

/// General-position (non-planar) and distant pairs must not regress: the chord
/// identity is exact for the whole range, so the recovered length matches the
/// independently-computed great-circle angle to ~1e-12.
#[test]
fn sphere_log_map_exact_for_distant_and_general_pairs() {
    let sphere = SphereManifold::new(2);
    // (p, q) unit vectors; exact geodesic = angle = acos(p·q) computed here in
    // a regime where acos is well-conditioned (well away from p·q≈1).
    let cases = [
        (array![1.0, 0.0, 0.0], array![0.0, 1.0, 0.0]), // π/2
        (array![1.0, 0.0, 0.0], array![-1.0, 0.0, 0.0_f64]), // near antipodal
        (
            array![0.5_f64.sqrt(), 0.5_f64.sqrt(), 0.0],
            array![0.0, 0.5_f64.sqrt(), 0.5_f64.sqrt()],
        ),
    ];
    for (p, q) in cases {
        // Renormalize to guard against literal rounding.
        let pn = &p / p.dot(&p).sqrt();
        let qn = &q / q.dot(&q).sqrt();
        let exact = pn.dot(&qn).clamp(-1.0, 1.0).acos();
        if exact > std::f64::consts::PI - 1e-9 {
            continue; // antipodal: log map direction is degenerate, skip length cmp
        }
        let v = sphere.log_map(pn.view(), qn.view()).expect("log_map");
        let recovered = v.dot(&v).sqrt();
        let rel = (recovered - exact).abs() / exact;
        assert!(
            rel < 1e-9,
            "p={pn:?}, q={qn:?}: recovered {recovered:.16e}, exact {exact:.16e}, rel err {rel:.3e}"
        );
    }
}

/// Fréchet mean over a tight cluster (exercises `sphere_weighted_log_step` and
/// `sphere_frechet_objective`, both formerly using the cancelling `acos(dot)`).
/// Two points placed symmetrically at ±theta from an axis have that axis as
/// their exact Fréchet mean; with theta in the cancellation regime the mean
/// must still be recovered to high precision.
#[test]
fn sphere_frechet_mean_accurate_on_nearby_cluster() {
    let theta = 1e-6_f64;
    // Axis (1,0,0); two points at ±theta in the x-y plane.
    let points = ndarray::array![
        [theta.cos(), theta.sin(), 0.0],
        [theta.cos(), -theta.sin(), 0.0],
    ];
    let mean = sphere_frechet_mean(points.view(), None, 1e-14, 200).expect("frechet mean");
    let axis = [1.0_f64, 0.0, 0.0];
    let mut dot = 0.0;
    for i in 0..3 {
        dot += mean[i] * axis[i];
    }
    let ang = dot.clamp(-1.0, 1.0).acos();
    assert!(
        ang < 1e-7,
        "Fréchet mean {mean:?} is {ang:.3e} rad off the symmetry axis (1,0,0)"
    );
}
