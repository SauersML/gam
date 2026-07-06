//! #1464 regression (different angle — basis-level criterion unit test): the
//! constant-curvature curvature criterion that the production fit and the
//! CI/flatness oracle now consume — the HONEST fixed-κ profiled REML
//! (`constant_curvature_honest_profiled_reml_score`) — must be SIGN-IDENTIFYING:
//! its argmin over the chart window is at NEGATIVE κ for genuinely hyperbolic
//! data and POSITIVE κ for genuinely spherical data.
//!
//! This guards the root-cause fix at the criterion level, below the full-fit
//! pipeline the headline contract (`bug_hunt_1464_curv_sign_identifiable`) and
//! the criterion-vs-solver test exercise. The #1464 bug was that the production
//! full-fit `reml_score` — which heavily SMOOTHS this RKHS kernel — is monotone
//! toward the +chart bound for EVERY truth (under heavy smoothing the +κ
//! geodesic-distance compression makes the collapsed kernel fit the over-smoothed
//! target better regardless of the true sign). The honest profiled REML keeps the
//! curvature-shape signal in the data fit, so its argmin tracks the planted sign.
//! A regression that re-routed the production criterion back through the
//! over-smoothed score would flip the hyperbolic argmin positive and trip this.
//!
//! Mirror datasets are generated with gam's OWN `ConstantCurvature::distance`, so
//! the planted geometry is the engine's own truth.

use gam::basis::{
    CenterStrategy, ConstantCurvatureBasisSpec, ConstantCurvatureIdentifiability,
    constant_curvature_honest_profiled_reml_score,
};
use gam::geometry::constant_curvature::ConstantCurvature;
use ndarray::{Array1, Array2};

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

fn curved_dataset(kappa_star: f64, seed: u64) -> (Array2<f64>, Array1<f64>) {
    let radius = 0.68_f64;
    let noise = 0.02_f64;
    let n = 600usize;
    let manifold = ConstantCurvature::new(2, kappa_star);
    let origin = ndarray::array![0.0_f64, 0.0_f64];
    let mut st = seed;
    let mut feats = Array2::<f64>::zeros((n, 2));
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
        let d = manifold.distance(pt.view(), origin.view()).expect("dist");
        feats[(filled, 0)] = x1;
        feats[(filled, 1)] = x2;
        y[filled] = 2.0 * (-d).exp() - 1.0 + noise * next_gauss(&mut st);
        filled += 1;
    }
    (feats, y)
}

/// argmin κ of the honest profiled-REML criterion over the chart-bounded grid.
fn honest_argmin(feats: &Array2<f64>, y: &Array1<f64>) -> f64 {
    // Chart window for radius 0.68: max ‖x‖² ≈ 0.46 ⇒ |κ| ≲ 1.08.
    let max_r2 = feats
        .outer_iter()
        .map(|r| r.dot(&r))
        .fold(0.0_f64, f64::max);
    let bound = 0.5 / max_r2;
    let mut best = (f64::INFINITY, f64::NAN);
    for i in 0..=40 {
        let kappa = -bound + (i as f64 / 40.0) * (2.0 * bound);
        let spec = ConstantCurvatureBasisSpec {
            center_strategy: CenterStrategy::FarthestPoint { num_centers: 10 },
            kappa,
            kappa_fixed: false,
            length_scale: 0.0,
            double_penalty: false,
            identifiability: ConstantCurvatureIdentifiability::CenterSumToZero,
        };
        if let Ok(v) = constant_curvature_honest_profiled_reml_score(feats.view(), y.view(), &spec)
            && v < best.0
        {
            best = (v, kappa);
        }
    }
    best.1
}

#[test]
fn honest_profiled_reml_criterion_identifies_curvature_sign() {
    // Spherical mirror (κ⋆ = +2): argmin κ must be POSITIVE.
    let (sph_x, sph_y) = curved_dataset(2.0, 0x5151_0001);
    let sph_argmin = honest_argmin(&sph_x, &sph_y);
    // Hyperbolic mirror (κ⋆ = −2): argmin κ must be NEGATIVE.
    let (hyp_x, hyp_y) = curved_dataset(-2.0, 0x5151_0003);
    let hyp_argmin = honest_argmin(&hyp_x, &hyp_y);

    eprintln!(
        "[#1464] honest criterion argmin: spherical κ={sph_argmin:+.4}, hyperbolic κ={hyp_argmin:+.4}"
    );

    assert!(
        sph_argmin > 0.0,
        "honest profiled-REML criterion must be minimised at κ>0 for spherical truth; got {sph_argmin}"
    );
    assert!(
        hyp_argmin < 0.0,
        "honest profiled-REML criterion must be minimised at κ<0 for hyperbolic truth; got {hyp_argmin} \
         (a sign-blind/over-smoothed criterion rails this positive — the #1464 root cause)"
    );
    // The two mirror datasets must be decisively separated, not the bit-identical
    // chart-bound value the bug returns for both signs.
    assert!(
        (sph_argmin - hyp_argmin) > 0.2,
        "spherical and hyperbolic argmins must be materially separated: \
         spherical {sph_argmin}, hyperbolic {hyp_argmin}"
    );
}
