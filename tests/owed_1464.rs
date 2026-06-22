//! #1464 regression — the constant-curvature DATA-FIT term must be κ-FAIR so the
//! curvature SIGN is identifiable on a GENERIC radial signal.
//!
//! The headline bug: on a center-peaked radial signal `μ = 2·exp(−d_{κ⋆}) − 1`
//! (NOT a kernel column inside the κ⋆ RKHS — the weakness of the older
//! `constant_curvature_recovers_curvature_sign_1404` oracle, whose planted signal
//! is literally `B(κ⋆)·β` and so is trivially best-fit at κ⋆), the +κ chart's
//! geodesic-distance COMPRESSION makes the design a uniformly better interpolator
//! of any radial peak regardless of the true sign. The raw profiled-REML data-fit
//! term therefore decreases monotonically toward the +chart bound for BOTH
//! spherical (κ⋆ = +2) and hyperbolic (κ⋆ = −2) truth, so κ̂ rails to ~+0.5/max‖x‖²
//! for both and hyperbolic data is mis-recovered as spherical.
//!
//! The fix (`constant_curvature_kappa_fair_sign_score`) subtracts the design's
//! GENERIC radial-peak-fitting power — measured on a bank of κ-independent
//! Euclidean-radial reference signals — from the data's profiled REML. The generic
//! +κ advantage cancels (it lifts the data fit and the reference fit equally),
//! leaving only the genuine curvature-shape signal, so the κ-fair score's argmin
//! lands on the correct SIDE of 0 for both signs.
//!
//! Two datasets that are exact mirror images under the curvature sign — one
//! spherical, one hyperbolic — are generated with gam's OWN geodesic-distance
//! convention (`ConstantCurvature::distance`), so the planted signal is gam's own
//! truth, never another tool's output. A correct estimator MUST distinguish them.

use gam::geometry::constant_curvature::ConstantCurvature;
use gam::terms::basis::{
    CenterStrategy, ConstantCurvatureBasisSpec, ConstantCurvatureIdentifiability,
    constant_curvature_kappa_fair_sign_score,
};
use ndarray::{Array1, Array2};

// --- deterministic RNG (splitmix64 → unit / gaussian), no external deps -------
fn splitmix64(state: &mut u64) -> u64 {
    *state = state.wrapping_add(0x9E37_79B9_7F4A_7C15);
    let mut z = *state;
    z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
    z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
    z ^ (z >> 31)
}
fn next_unit(state: &mut u64) -> f64 {
    (splitmix64(state) >> 11) as f64 / (1u64 << 53) as f64
}
fn next_gauss(state: &mut u64) -> f64 {
    let u1 = next_unit(state).max(1.0e-12);
    let u2 = next_unit(state);
    (-2.0 * u1.ln()).sqrt() * (std::f64::consts::TAU * u2).cos()
}

/// `n` chart points uniformly in a disk of radius `radius`, with a GENERIC radial
/// Gaussian response `μ = 2·exp(−d_{κ⋆}) − 1` of the `M_{κ⋆}` geodesic distance to
/// the origin (gam's OWN `ConstantCurvature::distance`). This is NOT a kernel
/// column in the κ⋆ RKHS, so identifying κ⋆ from it exercises the genuine
/// data-fit κ-fairness, not a trivial best-fit-at-truth tautology.
fn curved_dataset(kappa_star: f64, seed: u64) -> (Array2<f64>, Array1<f64>) {
    let radius = 0.68_f64;
    let noise = 0.02_f64;
    let n = 600usize;
    let manifold = ConstantCurvature::new(2, kappa_star);
    let origin = ndarray::array![0.0_f64, 0.0_f64];
    let mut st = seed;
    let mut x = Array2::<f64>::zeros((n, 2));
    let mut y = Array1::<f64>::zeros(n);
    let mut filled = 0usize;
    while filled < n {
        let a = 2.0 * next_unit(&mut st) - 1.0;
        let b = 2.0 * next_unit(&mut st) - 1.0;
        if a * a + b * b > 1.0 {
            continue;
        }
        let x1 = a * radius;
        let x2 = b * radius;
        let pt = ndarray::array![x1, x2];
        let d = manifold
            .distance(pt.view(), origin.view())
            .expect("in-chart geodesic distance");
        x[(filled, 0)] = x1;
        x[(filled, 1)] = x2;
        y[filled] = 2.0 * (-d).exp() - 1.0 + noise * next_gauss(&mut st);
        filled += 1;
    }
    (x, y)
}

fn base_spec(kappa: f64) -> ConstantCurvatureBasisSpec {
    ConstantCurvatureBasisSpec {
        center_strategy: CenterStrategy::FarthestPoint { num_centers: 10 },
        kappa,
        length_scale: 0.0,
        double_penalty: false,
        identifiability: ConstantCurvatureIdentifiability::CenterSumToZero,
    }
}

/// argmin of the κ-fair sign score over the chart window `[−half, half]`, with
/// `half = 0.5 / max‖x‖²` (the production κ window).
fn kappa_fair_argmin(x: &Array2<f64>, y: &Array1<f64>) -> f64 {
    let max_r2 = x
        .outer_iter()
        .map(|r| r.dot(&r))
        .fold(1e-8_f64, f64::max);
    let half = 0.5 / max_r2;
    let steps = 24usize;
    let mut best_k = f64::NAN;
    let mut best_v = f64::INFINITY;
    for i in 0..=steps {
        let kappa = -half + 2.0 * half * (i as f64) / (steps as f64);
        let v = constant_curvature_kappa_fair_sign_score(x.view(), y.view(), &base_spec(kappa))
            .expect("κ-fair score evaluates on in-chart data");
        if v < best_v {
            best_v = v;
            best_k = kappa;
        }
    }
    best_k
}

/// The κ-fair data-fit score identifies the curvature SIGN on a generic radial
/// signal: spherical truth → κ̂ > 0, hyperbolic truth → κ̂ < 0, and the two mirror
/// datasets are decisively distinguished. The raw profiled-REML data-fit term
/// (used for the κ̂ magnitude/CI) rails both to the +chart bound — this asserts
/// the κ-fairness correction restores sign identifiability.
#[test]
fn kappa_fair_data_fit_identifies_curvature_sign_on_generic_radial_signal() {
    // Two seeds each so the assertion is not a single-realization fluke.
    for &(seed_sph, seed_hyp) in &[(0x5151_0001_u64, 0x5151_0003_u64), (0x7777_0002, 0x7777_0004)]
    {
        let (x_sph, y_sph) = curved_dataset(2.0, seed_sph);
        let (x_hyp, y_hyp) = curved_dataset(-2.0, seed_hyp);

        let k_sph = kappa_fair_argmin(&x_sph, &y_sph);
        let k_hyp = kappa_fair_argmin(&x_hyp, &y_hyp);

        eprintln!(
            "[#1464] κ-fair argmin: spherical(κ⋆=+2)={k_sph:+.4}  hyperbolic(κ⋆=−2)={k_hyp:+.4}"
        );

        // (a) Control — spherical truth recovers POSITIVE curvature.
        assert!(
            k_sph > 0.0,
            "spherical truth (κ⋆=+2) must recover POSITIVE κ̂ from the κ-fair score; got {k_sph}"
        );
        // (b) Headline — hyperbolic truth recovers NEGATIVE curvature (NOT the
        //     +chart-bound rail the sign-blind raw data-fit term produces).
        assert!(
            k_hyp < 0.0,
            "hyperbolic truth (κ⋆=−2) must recover NEGATIVE κ̂ from the κ-fair score; got {k_hyp} \
             (the #1464 bug rails this to the +chart bound, calling it spherical)"
        );
        // (c) The two mirror datasets are decisively distinguished.
        assert!(
            (k_sph - k_hyp).abs() > 0.1,
            "spherical and hyperbolic mirror datasets must yield materially DIFFERENT κ̂: \
             spherical {k_sph}, hyperbolic {k_hyp}"
        );
    }
}
