//! Measurement suite for #2489: the geodesic separation on S² was carried as
//! a dot product.
//!
//! Every number in the issue's tables is re-measured here against the shipped
//! code, plus the legacy route kept alongside so the comparison is not against
//! a remembered figure. The pre-registration in the issue thread fixed the
//! metric, the offsets and the pass bars before the fix was written; this file
//! is that pre-registration executed.
//!
//! Three claims:
//!
//! 1. `u = sin²(γ/2)` from the haversine/chord form holds full relative
//!    accuracy down to sub-nanodegree separations, where the dot-product form
//!    loses every digit and finally returns exactly zero.
//! 2. The Gram diagonal of a zonal kernel is ONE number, because `u = 0` at
//!    coincidence is now a theorem about bit patterns rather than a rounding
//!    accident that depends on the coordinates of the center.
//! 3. The jet's `u`-chain is the same derivative as the `cos γ` chain at
//!    ordinary separations, and `∂u` is exactly zero at a center.

use super::sphere_half_angle::{
    HalfAngleSeparation, SphereTrig, ambient_half_angle_separation, half_angle_partials,
    half_angle_separation_scalar,
};
use super::sphere_kernels::{wahba_sphere_kernel_derivative_dhav_kind, wahba_sphere_kernel_kind};
use super::sphere_spec::SphereWahbaKernel;
use ndarray::{Array2, array};

const DEG: f64 = std::f64::consts::PI / 180.0;

/// The route that shipped before this fix: form `cos γ` from the precomputed
/// trigonometry by angle-subtraction, then recover `u` from it.
fn legacy_u(a: SphereTrig<f64>, b: SphereTrig<f64>) -> f64 {
    let dlon_cos = a.cos_lon * b.cos_lon + a.sin_lon * b.sin_lon;
    let cos_gamma = a.sin_lat * b.sin_lat + a.cos_lat * b.cos_lat * dlon_cos;
    (1.0 - cos_gamma.clamp(-1.0, 1.0)) * 0.5
}

/// `u = sin²(γ/2)` for two points separated in latitude only, in closed form.
///
/// With `ψ = ψ_c` the haversine collapses to `u = sin²(Δφ/2)` exactly, and
/// `Δφ` is available to full relative precision because the two latitudes are
/// subtracted in the raw units they were written in. This is the reference the
/// two production routes are measured against; it shares no arithmetic with
/// either (no `cos γ`, no chord, no trig of the individual latitudes).
fn reference_u_lat_offset(offset_deg: f64) -> f64 {
    let half = 0.5 * offset_deg * DEG;
    half.sin().powi(2)
}

// ---------------------------------------------------------------------------
// 1. The separation itself.
// ---------------------------------------------------------------------------

#[test]
fn zz_measure_2489_haversine_u_survives_where_the_dot_product_collapses() {
    // The bar is derived, not chosen. Both routes reconstruct `u` from stored
    // `sin`/`cos` values that carry an absolute error of order `ε`, so the
    // question is only how that error is amplified:
    //
    //   * the chord form subtracts those values to make a chord of length
    //     `≈ γ`, then squares it, giving relative error `O(ε/γ)`;
    //   * the dot product forms `cos γ = 1 − O(γ²)` and takes the difference
    //     against `1`, so the same absolute `ε` lands on a quantity of size
    //     `γ²` — relative error `O(ε/γ²)`, one whole factor of `γ` worse.
    //
    // So the test is: the chord form stays inside `8ε/γ` at every offset, and
    // the dot product is outside it at every offset. No tolerance constant
    // appears that is not `f64::EPSILON` and the separation itself.
    let base_lat = 37.7749_f64;
    let lon = -122.4194_f64;
    let offsets = [1e-2, 1e-3, 1e-4, 1e-5, 1e-6, 1e-7, 1e-8, 1e-9, 1e-10];

    println!(
        "\n{:>10} {:>24} {:>24} {:>11} {:>24} {:>11} {:>11}",
        "offset(°)", "reference u", "chord u", "rel", "dot-product u", "rel", "bound 8ε/γ"
    );
    for offset in offsets {
        let a = SphereTrig::from_radians(base_lat * DEG, lon * DEG);
        let b = SphereTrig::from_radians((base_lat + offset) * DEG, lon * DEG);
        let want = reference_u_lat_offset(offset);
        let chord = half_angle_separation_scalar(a, b).u;
        let dot = legacy_u(a, b);
        let rel_chord = (chord - want).abs() / want;
        let rel_dot = (dot - want).abs() / want;
        let bound = 8.0 * f64::EPSILON / (offset * DEG);
        println!(
            "{offset:>10.0e} {want:>24.16e} {chord:>24.16e} {rel_chord:>11.2e} \
             {dot:>24.16e} {rel_dot:>11.2e} {bound:>11.2e}"
        );
        assert!(
            rel_chord <= bound,
            "chord-form u is {rel_chord:.3e} off at {offset:.0e}°, outside the \
             O(ε/γ) bound {bound:.3e} its conditioning allows"
        );
        assert!(
            rel_dot > bound,
            "the dot-product route came in at {rel_dot:.3e} at {offset:.0e}°, \
             inside the {bound:.3e} bound it is not supposed to reach — either \
             the legacy reconstruction here no longer matches what shipped, or \
             the premise of #2489 has changed"
        );
    }

    // The endpoint the issue reported: below ~1e-8 degrees the dot product has
    // nothing left at all. Assert it directly so this file records WHY the
    // change was made, not just that the new route is accurate.
    let a = SphereTrig::from_radians(base_lat * DEG, lon * DEG);
    let b = SphereTrig::from_radians((base_lat + 1e-10) * DEG, lon * DEG);
    assert_eq!(
        legacy_u(a, b),
        0.0,
        "the dot-product route is supposed to report two points 1e-10° apart as \
         exactly coincident — if it no longer does, this measurement has drifted"
    );
    assert!(
        half_angle_separation_scalar(a, b).u > 0.0,
        "the chord form must separate them"
    );
}

#[test]
fn zz_measure_2489_coincidence_is_exact_and_the_pair_sums_to_one() {
    // `u = 0` iff bitwise-equal coordinates, at latitudes and longitudes chosen
    // to be nothing special — the point is that no coordinate is special.
    let coords = [
        (37.7749_f64, -122.4194_f64),
        (0.0, 0.0),
        (90.0, 0.0),
        (-90.0, 180.0),
        (-33.8688, 151.2093),
        (1e-9, 1e-9),
        (89.999_999, -0.000_001),
    ];
    for (lat, lon) in coords {
        let t = SphereTrig::from_radians(lat * DEG, lon * DEG);
        let sep = half_angle_separation_scalar(t, t);
        assert_eq!(
            sep.u, 0.0,
            "self-separation at ({lat}, {lon}) must be exactly 0, not {:.3e}",
            sep.u
        );
        // u + v = 1 analytically; the two are computed independently, so their
        // sum is a check on both.
        let closure = (sep.u + sep.v - 1.0).abs();
        assert!(
            closure < 4.0 * f64::EPSILON,
            "u + v = 1 violated by {closure:.3e} at ({lat}, {lon})"
        );
    }

    // Antipodal pairs are the mirror image: v, not u, must vanish exactly.
    for (lat, lon) in [(0.0_f64, 0.0_f64), (37.5, 12.25), (-64.0, -90.0)] {
        let t = SphereTrig::from_radians(lat * DEG, lon * DEG);
        let anti = SphereTrig::from_radians(-lat * DEG, (lon + 180.0) * DEG);
        let sep = half_angle_separation_scalar(t, anti);
        assert!(
            sep.v < 1e-30,
            "antipodal v at ({lat}, {lon}) is {:.3e}, not ~0",
            sep.v
        );
    }
}

#[test]
fn zz_measure_2489_gram_diagonal_is_one_number_across_centers() {
    // The rotation-invariance failure the issue reports: a zonal kernel's
    // diagonal is K(0) for EVERY point, but the dot-product route made it a
    // function of where the point happens to sit, because whether
    // sin²φ + cos²φ·(cos²ψ + sin²ψ) rounds to 1.0 depends on φ and ψ. Over a
    // spread of centers the issue measured three distinct diagonal values in
    // one Gram matrix. The Sobolev `m = 2` closed form needs `Li₂(v)`, and `v`
    // is `1` at coincidence only up to rounding, so the kernel takes `Li₂(v)`
    // through the exact `u` (`polylog::dilog_of_complement`); reading
    // `dilog_unit(v)` directly gave three distinct diagonals here.
    let mut centers = Vec::<(f64, f64)>::new();
    for i in 0..24 {
        let t = i as f64;
        centers.push((
            -85.0 + 170.0 * (t * 0.041_666_7).fract(),
            -180.0 + 360.0 * (t * 0.137).fract(),
        ));
    }

    let mut legacy_diag = std::collections::BTreeSet::<u64>::new();
    let mut shipped_diag = std::collections::BTreeSet::<u64>::new();
    for &(lat, lon) in &centers {
        let t = SphereTrig::from_radians(lat * DEG, lon * DEG);
        // Legacy: u from the dot product, then the kernel.
        let legacy_sep = HalfAngleSeparation {
            u: legacy_u(t, t),
            v: 1.0 - legacy_u(t, t),
        };
        let legacy = wahba_sphere_kernel_kind(legacy_sep, 2, SphereWahbaKernel::Sobolev)
            .expect("sobolev m=2 is finite at coincidence");
        let shipped = wahba_sphere_kernel_kind(
            half_angle_separation_scalar(t, t),
            2,
            SphereWahbaKernel::Sobolev,
        )
        .expect("sobolev m=2 is finite at coincidence");
        legacy_diag.insert(legacy.to_bits());
        shipped_diag.insert(shipped.to_bits());
    }
    println!(
        "\n  distinct Gram diagonal values over {} centers: legacy {}, shipped {}\n",
        centers.len(),
        legacy_diag.len(),
        shipped_diag.len()
    );
    assert_eq!(
        shipped_diag.len(),
        1,
        "a zonal kernel's Gram diagonal must be ONE number; got {} distinct \
         values across {} centers",
        shipped_diag.len(),
        centers.len()
    );
}

// ---------------------------------------------------------------------------
// 2. The jet — the arm with the worst symptom.
// ---------------------------------------------------------------------------

fn shipped_jet_dphi(
    a: SphereTrig<f64>,
    b: SphereTrig<f64>,
    m: usize,
    kind: SphereWahbaKernel,
) -> f64 {
    let sep = half_angle_separation_scalar(a, b);
    if sep.u <= 0.0 {
        return 0.0;
    }
    let (du_dphi, _) = half_angle_partials(a, b);
    wahba_sphere_kernel_derivative_dhav_kind(sep, m, kind) * du_dphi * DEG
}

#[test]
fn zz_measure_2489_half_angle_partials_vanish_at_a_center() {
    // AT a center. Reachable in every fit, because farthest-point selection
    // picks centers from the data rows. `∂u` is exactly zero there, so a cusp
    // is resolved by the caller rather than reached as `∞ · 0`.
    let t = SphereTrig::from_radians(12.5 * DEG, 44.25 * DEG);
    let (du_dphi, du_dpsi) = half_angle_partials(t, t);
    assert_eq!(
        (du_dphi, du_dpsi),
        (0.0, 0.0),
        "∂u/∂φ and ∂u/∂ψ at a center must be exactly zero, so the cusp is \
         resolved by the caller rather than reached as ∞·0"
    );
}

#[test]
fn zz_measure_2489_smooth_jet_matches_a_finite_difference() {
    // Non-regression at ordinary separations, and a check that the u-chain is
    // the same derivative as before rather than merely a better-conditioned
    // number. Central differences on the forward kernel, at offsets where both
    // routes are healthy.
    let lon = -3.75_f64;
    for &(kind, m) in &[
        (SphereWahbaKernel::Sobolev, 2usize),
        (SphereWahbaKernel::Sobolev, 3),
        (SphereWahbaKernel::SobolevTruncated { lmax: 64 }, 2),
    ] {
        for &(row_lat, center_lat) in &[(20.0_f64, 35.0_f64), (-5.0, 5.0), (60.0, 61.5)] {
            let center = SphereTrig::from_radians(center_lat * DEG, lon * DEG);
            let h = 1e-5_f64;
            let k_at = |lat: f64| -> f64 {
                let row = SphereTrig::from_radians(lat * DEG, lon * DEG);
                wahba_sphere_kernel_kind(half_angle_separation_scalar(row, center), m, kind)
                    .expect("finite away from coincidence")
            };
            let fd = (k_at(row_lat + h) - k_at(row_lat - h)) / (2.0 * h);
            let row = SphereTrig::from_radians(row_lat * DEG, lon * DEG);
            let analytic = shipped_jet_dphi(row, center, m, kind);
            let rel = (analytic - fd).abs() / fd.abs().max(1e-300);
            assert!(
                rel < 1e-6,
                "{kind:?} m={m}: analytic ∂K/∂φ {analytic:.9e} vs central \
                 difference {fd:.9e} (rel {rel:.3e}) at row {row_lat}, center \
                 {center_lat}"
            );
        }
    }
}

// ---------------------------------------------------------------------------
// 3. The ambient N-D twin.
// ---------------------------------------------------------------------------

#[test]
fn zz_measure_2489_ambient_separation_is_exact_at_coincidence() {
    // `sphere_first_derivative_nd` had the same defect in xyz coordinates,
    // where `u = |t − c|²/4` is an identity for unit vectors rather than a
    // trigonometric rearrangement.
    let points: Array2<f64> = array![
        [1.0, 0.0, 0.0],
        [0.0, 1.0, 0.0],
        [
            0.577_350_269_189_625_7,
            0.577_350_269_189_625_7,
            0.577_350_269_189_625_7
        ],
    ];
    for i in 0..points.nrows() {
        let sep = ambient_half_angle_separation(points.row(i), points.row(i));
        assert_eq!(sep.u, 0.0, "ambient self-separation must be exactly 0");
        assert!((sep.u + sep.v - 1.0).abs() < 4.0 * f64::EPSILON);
    }
    // Orthogonal vectors: γ = π/2, u = v = ½.
    let sep = ambient_half_angle_separation(points.row(0), points.row(1));
    assert!((sep.u - 0.5).abs() < 4.0 * f64::EPSILON);
    assert!((sep.v - 0.5).abs() < 4.0 * f64::EPSILON);

    // A near-coincident pair the dot product cannot resolve: perturb one
    // component by 1e-12 and renormalize implicitly (the separation is
    // scale-free in the two norms).
    let a = array![1.0, 0.0, 0.0];
    let b = array![1.0, 1e-12, 0.0];
    let sep = ambient_half_angle_separation(a.view(), b.view());
    // |t − c|² = 1e-24, norms ~ 1 and ~1, so u ≈ 1e-24/4.
    let want = 1e-24 / 4.0;
    let rel = (sep.u - want).abs() / want;
    let legacy = (1.0 - (a[0] * b[0] + a[1] * b[1] + a[2] * b[2])) * 0.5;
    println!(
        "\n  ambient 1e-12 perturbation: chord u = {:.6e} (rel {rel:.2e}), \
         dot-product u = {legacy:.6e}\n",
        sep.u
    );
    assert!(rel < 1e-6, "ambient chord u is {rel:.3e} off");
    assert_eq!(
        legacy, 0.0,
        "the dot-product route is supposed to lose this pair entirely"
    );
}
