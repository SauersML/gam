//! Regression pin for #1095 — a circle (K=1, d=2 periodic) SAE atom must accept
//! at least one seed and fit on a SMALL activation bank (N=180), the same size
//! as the OLMo L44 color bank where every candidate seed was rejected at the
//! curvature-homotopy entry validation stage ("all 13 seeds rejected").
//!
//! The issue signature: on N=180 the joint Hessian at the early curvature spine
//! has a sub-floor pivot (`min pivot 3.865e-9`) that the entry walk classified
//! as a NON-gauge branch bifurcation, refusing the seed; the SAME settings on
//! N=635 converge. The circle's rotation gauge null must be recognised so a
//! small bank fits instead of erroring with `RemlConvergenceError`.
//!
//! This drives the fit exactly the way production does (`OuterProblem::run`
//! around `SaeManifoldOuterObjective`) and asserts the cascade COMPLETES with a
//! finite criterion (not the all-seeds-rejected startup error).

use gam::solver::rho_optimizer::OuterProblem;
use gam::terms::latent::LatentManifold;
use gam::terms::{
    AssignmentMode, PeriodicHarmonicEvaluator, SaeAssignment, SaeAtomBasisKind, SaeBasisEvaluator,
    SaeManifoldAtom, SaeManifoldOuterObjective, SaeManifoldRho, SaeManifoldTerm,
};
use ndarray::{Array1, Array2};
use std::sync::Arc;

const N: usize = 180; // the L44 color-bank size from #1095
const P: usize = 24; // PCA ambient (issue uses 32; 24 keeps the test cheap)
const M: usize = 3; // const + 1 harmonic (sin, cos) -> circle
const TAU: f64 = 0.5;
const ALPHA: f64 = 1.0;
const INNER_MAX_ITER: usize = 50;
const LEARNING_RATE: f64 = 1.0;
const RIDGE_EXT_COORD: f64 = 1.0e-6;
const RIDGE_BETA: f64 = 1.0e-6;

/// Deterministic Lehmer-style uniform in [0,1) keyed by index (no clock).
fn idx_uniform(seed: u64) -> f64 {
    let mut state = seed
        .wrapping_mul(6364136223846793005)
        .wrapping_add(1442695040888963407);
    state = state
        .wrapping_mul(6364136223846793005)
        .wrapping_add(1442695040888963407);
    ((state >> 11) as f64) * f64::from_bits(0x3CA0000000000000)
}

fn idx_normal(seed: u64) -> f64 {
    let u1 = idx_uniform(seed).max(1.0e-12);
    let u2 = idx_uniform(seed.wrapping_add(0x9E3779B97F4A7C15));
    (-2.0 * u1.ln()).sqrt() * (std::f64::consts::TAU * u2).cos()
}

/// A circle planted in a 2-plane of `R^P`, plus small ambient noise. This is a
/// genuinely-circular bank (so a fit exists); the bug is purely the small-N
/// seed-acceptance gate, not representability.
fn planted_small_circle() -> Array2<f64> {
    let mut z = Array2::<f64>::zeros((N, P));
    for i in 0..N {
        let theta = std::f64::consts::TAU * ((i as f64) * 0.061_803 + 0.13).rem_euclid(1.0);
        // plane spanned by ambient axes 0 and 1, mild signal on a few more.
        z[[i, 0]] = theta.cos();
        z[[i, 1]] = theta.sin();
        z[[i, 2]] = 0.4 * (2.0 * theta).cos();
        for col in 0..P {
            z[[i, col]] += 0.03 * idx_normal((i as u64) * 31 + col as u64);
        }
    }
    z
}

/// Cold K=1 circle term: latent angles seeded near the planted angles, periodic
/// harmonic basis, zero decoder (the engine refits it). Mirrors the cold term
/// the production `sae_manifold_fit` hands the outer cascade.
fn build_cold_circle_term(z: &Array2<f64>) -> SaeManifoldTerm {
    let evaluator = Arc::new(PeriodicHarmonicEvaluator::new(M).unwrap());
    // Seed each row's angle from its ambient (axis-0, axis-1) projection — the
    // cold PCA-plane angle the production seeder would recover — normalised to
    // the unit period the Circle manifold expects.
    let coords = Array2::from_shape_fn((N, 1), |(i, _)| {
        let angle = z[[i, 1]].atan2(z[[i, 0]]);
        (angle / std::f64::consts::TAU).rem_euclid(1.0)
    });
    let (phi, jet) = evaluator.evaluate(coords.view()).unwrap();
    let atom = SaeManifoldAtom::new(
        "circle_0".to_string(),
        SaeAtomBasisKind::Periodic,
        1,
        phi,
        jet,
        Array2::<f64>::zeros((M, P)),
        Array2::<f64>::eye(M),
    )
    .unwrap()
    .with_basis_evaluator(evaluator);
    let assignment = SaeAssignment::from_blocks_with_mode_and_manifolds(
        Array2::<f64>::zeros((N, 1)),
        vec![coords],
        vec![LatentManifold::Circle { period: 1.0 }],
        AssignmentMode::ibp_map(TAU, ALPHA, false),
    )
    .unwrap();
    SaeManifoldTerm::new(vec![atom], assignment).unwrap()
}

fn reconstruction_r2(fitted: &Array2<f64>, z: &Array2<f64>) -> f64 {
    let mut zbar = 0.0;
    for v in z.iter() {
        zbar += *v;
    }
    zbar /= (N * P) as f64;
    let mut ssr = 0.0;
    let mut sst = 0.0;
    for (fi, zi) in fitted.iter().zip(z.iter()) {
        ssr += (fi - zi) * (fi - zi);
    }
    for v in z.iter() {
        sst += (v - zbar) * (v - zbar);
    }
    1.0 - ssr / sst.max(1.0e-300)
}

#[test]
fn sae_manifold_small_n_circle_accepts_a_seed_and_fits() {
    let z = planted_small_circle();
    let term = build_cold_circle_term(&z);
    let init_rho = SaeManifoldRho::new(0.0, 0.0, vec![Array1::<f64>::zeros(0); 1]);
    let init_rho_flat = init_rho.to_flat();
    let n_params = init_rho_flat.len();
    let mut objective = SaeManifoldOuterObjective::new(
        term,
        z.clone(),
        None,
        init_rho,
        INNER_MAX_ITER,
        LEARNING_RATE,
        RIDGE_EXT_COORD,
        RIDGE_BETA,
    );
    let problem = OuterProblem::new(n_params).with_initial_rho(init_rho_flat);
    let result = problem
        .run(&mut objective, "SAE small-N circle seed accept (#1095)")
        .expect(
            "outer cascade must complete on a small-N circle bank — \
             all seeds rejected reproduces #1095",
        );
    let (fitted_term, _rho, _loss) = objective.into_fitted();
    let fitted = fitted_term.fitted();
    let r2 = reconstruction_r2(&fitted, &z);
    println!(
        "[#1095] small-N circle fit: final_value={:.6e} recon_R2={:.6}",
        result.final_value, r2
    );
    assert!(
        result.final_value.is_finite() && result.final_value < 1.0e11,
        "small-N circle fit terminated at the infeasible sentinel \
         (final_value={:.6e})",
        result.final_value
    );
    assert!(
        r2 > 0.9,
        "small-N circle reconstruction R²={r2:.6} < 0.9 — the circle was not recovered"
    );
}
