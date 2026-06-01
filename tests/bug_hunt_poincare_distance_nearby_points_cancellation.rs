//! Regression for issue #514: `poincare_distance` — the public, Python-exposed
//! Poincaré-ball geodesic distance — loses precision catastrophically for
//! nearby points, reaching ~4% relative error at separation 1e-8.
//!
//! Root cause: it evaluated the textbook form
//!   d = (1/√k)·arccosh(1 + 2δ),  δ = k|a-b|²/((1-k|a|²)(1-k|b|²)).
//! For nearby points δ is tiny, so `1 + 2δ` rounds away everything in δ below
//! ~1e-16 relative to 1 — the classic cancellation at the arccosh branch point.
//! Since arccosh(1+2δ) ≈ 2√δ, a relative perturbation eps/(2δ) in the argument
//! becomes a ~eps/δ ~ eps/sep² relative error in the distance.
//!
//! Fix: the cosh half-angle identity arccosh(1 + 2δ) = 2·arcsinh(√δ). √δ is
//! formed straight from |a-b|² (faithful to ~eps) and arcsinh(small) does not
//! cancel, so the whole range is accurate; arcsinh(0)=0 keeps identical points
//! at exact zero.
//!
//! Related: #513 (upper-tail quantile), #515 (sphere acos).

use gam::geometry::poincare::poincare_distance;
use ndarray::array;

/// Exact geodesic distance between two collinear points `a = r1·e`, `b = r2·e`
/// on a ray through the origin, curvature `c` (`k = -c`). The geodesic is the
/// diameter, and the artanh addition formula collapses
/// `2·artanh(√k·r2) − 2·artanh(√k·r1)` to the cancellation-free
///   d = (2/√k)·artanh(√k·(r2 − r1)/(1 − k·r1·r2)),
/// which is formed from `(r2 − r1)` directly (no near-1 subtraction) and is
/// therefore faithful to ~1e-15 at every separation — an independent ground
/// truth for the library's general-position formula.
fn collinear_exact(r1: f64, r2: f64, curvature: f64) -> f64 {
    let k = -curvature;
    let sk = k.sqrt();
    (2.0 / sk) * (sk * (r2 - r1) / (1.0 - k * r1 * r2)).atanh()
}

#[test]
fn poincare_distance_accurate_for_nearby_points() {
    let r1 = 0.3_f64;
    let curvature = -1.0_f64;
    // Separations spanning the cancellation regime; the old arccosh(1+2δ) form
    // hit rel err 9e-8, 2e-6, 1e-3, 4e-2 respectively here.
    for &sep in &[1e-5_f64, 1e-6, 1e-7, 1e-8, 1e-9] {
        let r2 = r1 + sep;
        let a = array![r1];
        let b = array![r2];
        let got = poincare_distance(a.view(), b.view(), curvature).expect("distance");
        let exact = collinear_exact(r1, r2, curvature);
        let rel = (got - exact).abs() / exact;
        assert!(
            rel < 1e-9,
            "sep={sep:e} apart: got {got:.16e}, exact {exact:.16e}, rel err {rel:.3e} > 1e-9 \
             — arccosh(1+2δ) cancellation near the branch point"
        );
    }
}

/// The cure must not regress general-position / high-curvature pairs: the
/// half-angle identity is exact for all δ ≥ 0, so distant collinear pairs and
/// a non-unit curvature must still match the artanh ground truth tightly.
#[test]
fn poincare_distance_exact_for_distant_and_high_curvature_pairs() {
    let cases = [
        (0.1_f64, 0.8_f64, -1.0_f64),          // far apart, c = -1
        (-0.6_f64, 0.7_f64, -1.0_f64),         // straddling the origin
        (0.05_f64, 0.2_f64, -8.0_f64),         // high curvature, nearby
        (0.05_f64, 0.05_f64 + 1e-7, -8.0_f64), // high curvature, very nearby
    ];
    for (r1, r2, c) in cases {
        let a = array![r1];
        let b = array![r2];
        let got = poincare_distance(a.view(), b.view(), c).expect("distance");
        let exact = collinear_exact(r1, r2, c);
        let rel = (got - exact).abs() / exact.abs().max(1e-300);
        assert!(
            rel < 1e-9,
            "r1={r1}, r2={r2}, c={c}: got {got:.16e}, exact {exact:.16e}, rel err {rel:.3e} > 1e-9"
        );
    }
}

/// Identical points must still yield exactly 0.0 (arcsinh(0)=0), preserving the
/// #239 self-distance contract that the previous exact-1.0 clamp protected.
#[test]
fn poincare_distance_identical_points_exact_zero() {
    let a = array![0.3_f64, -0.1, 0.2];
    let d = poincare_distance(a.view(), a.view(), -1.0).expect("distance");
    assert_eq!(d, 0.0, "self-distance must be exactly 0.0, got {d:e}");
}
