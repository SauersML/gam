//! Regression pin for #2228 / #1095 / #2226: a K=1, d=1 IBP SAE must
//! converge at fixed rho and reconstruct an exactly representable curve.
//!
//! Ordered IBP shrinkage belongs only to the empirical-Bayes assignment prior.
//! The forward posterior-mean gate is sigmoid(logit / temperature), without a
//! second multiplication by the ordered prior mean. Decoder magnitude stays in
//! the physical coefficient block B, so this fixture directly catches either a
//! reintroduced prior cap or a convergence path that returns a cold seed.
//! `RemlConvergenceError` (inner solve stalls) or an `R² ≈ 0.4375` pristine-seed
//! fallback. Drives the public outer engine at ridge-0 exactly as production does.

use gam::linalg::faer_ndarray::{FaerCholesky, fast_ata, fast_atb};
use gam::solver::rho_optimizer::OuterProblem;
use gam::terms::latent::LatentManifold;
use gam::terms::sae::manifold::EuclideanPatchEvaluator;
use gam::terms::{
    sae::manifold::AssignmentMode, sae::manifold::SaeAssignment, sae::manifold::SaeAtomBasisKind,
    sae::manifold::SaeBasisEvaluator, sae::manifold::SaeManifoldAtom,
    sae::manifold::SaeManifoldOuterObjective, sae::manifold::SaeManifoldRho,
    sae::manifold::SaeManifoldTerm,
};
use ndarray::{Array1, Array2};
use std::sync::Arc;

use faer::Side as FaerSide;

const N: usize = 160;
const P: usize = 12; // ambient output dim
const D: usize = 1; // the #2228 configuration: intrinsic 1-D atom
const MAX_DEGREE: usize = 2; // degree-2 patch {1, t, t²} — spans the planted curve
const TAU: f64 = 0.5;
// K=1 ⇒ the ordered prior mean is 0.5. It prices the empirical-Bayes assignment
// prior, but must not multiply the forward posterior-mean gate a second time.
const ALPHA: f64 = 1.0;
const INNER_MAX_ITER: usize = 50;
const LEARNING_RATE: f64 = 1.0;
const RIDGE_EXT_COORD: f64 = 0.0; // the ridge-0 public path
const RIDGE_BETA: f64 = 1.0e-6;

fn idx_noise(seed: u64) -> f64 {
    let mut s = seed
        .wrapping_mul(6364136223846793005)
        .wrapping_add(1442695040888963407);
    s = s
        .wrapping_mul(6364136223846793005)
        .wrapping_add(1442695040888963407);
    let u = ((s >> 11) as f64) * f64::from_bits(0x3CA0000000000000);
    (u - 0.5) * 2.0
}

/// The intrinsic 1-D latent coordinate for row `i`, spread over `[-1, 1]`.
fn latent_t(i: usize) -> f64 {
    -1.0 + 2.0 * (i as f64) / ((N - 1) as f64)
}

/// A 1-D curve embedded in `R^P` as an EXACT degree-2 polynomial image of its
/// latent coordinate `t`: `z[i, c] = A[0, c] + A[1, c]·t_i + A[2, c]·t_i²` plus
/// small ambient noise. A degree-2 Euclidean patch `{1, t, t²}` reconstructs this
/// exactly, so any reconstruction shortfall is the gate co-collapse, not the
/// basis. The per-channel coefficients are deterministic (seeded `idx_noise`).
fn planted_curve() -> Array2<f64> {
    let mut z = Array2::<f64>::zeros((N, P));
    for i in 0..N {
        let t = latent_t(i);
        for c in 0..P {
            let a0 = idx_noise((c as u64) * 3 + 1);
            let a1 = idx_noise((c as u64) * 3 + 2);
            let a2 = idx_noise((c as u64) * 3 + 3);
            z[[i, c]] = a0 + a1 * t + a2 * t * t + 0.02 * idx_noise((i as u64) * 97 + c as u64);
        }
    }
    z
}

/// Latent coords seeded at the TRUE 1-D coordinate `t_i`, so the chart is
/// correct and the only remaining fit is the (gate-compensating) decoder.
fn seed_coords() -> Array2<f64> {
    Array2::from_shape_fn((N, D), |(i, _)| latent_t(i))
}

fn decoder_lsq_init(phi: &Array2<f64>, z: &Array2<f64>) -> Array2<f64> {
    let m = phi.ncols();
    let mut gram = fast_ata(phi);
    for i in 0..m {
        gram[[i, i]] += 1.0e-8;
    }
    let rhs = fast_atb(phi, &z.to_owned());
    gram.cholesky(FaerSide::Lower)
        .expect("decoder LSQ Cholesky")
        .solve_mat(&rhs)
}

fn build_cold_d1_term(z: &Array2<f64>) -> SaeManifoldTerm {
    let evaluator = Arc::new(EuclideanPatchEvaluator::new(D, MAX_DEGREE).unwrap());
    let n_basis = evaluator.basis_size();
    let coords = seed_coords();
    let (phi, jet) = evaluator.evaluate(coords.view()).unwrap();
    let decoder = decoder_lsq_init(&phi, z);
    let atom = SaeManifoldAtom::new(
        "curve_d1".to_string(),
        SaeAtomBasisKind::EuclideanPatch,
        D,
        phi,
        jet,
        decoder,
        Array2::<f64>::eye(n_basis),
    )
    .unwrap()
    .with_basis_second_jet(evaluator.clone());
    let assignment = SaeAssignment::from_blocks_with_mode_and_manifolds(
        Array2::<f64>::zeros((N, 1)),
        vec![coords],
        vec![LatentManifold::Euclidean],
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
fn sae_manifold_d1_ibp_gate_cocollapse() {
    let z = planted_curve();
    let term = build_cold_d1_term(&z);
    let init_rho = SaeManifoldRho::new(0.0, 0.0, vec![Array1::<f64>::zeros(D); 1]);
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
    // A `GamError` here (inner solve stalls at fixed ρ) is the #2228 refusal.
    let result = problem
        .run(
            &mut objective,
            "SAE d=1 K=1 IBP-gate co-collapse (#2228/#1095)",
        )
        .expect(
            "outer cascade must complete on a K=1 d=1 atom whose degree-2 patch spans its \
             planted curve — a RemlConvergenceError (inner solve stalls at fixed ρ) reproduces \
             the #2228 IBP-gate/decoder co-collapse",
        );
    objective
        .certify_outer_result(&result)
        .expect("IBP co-collapse outer result must certify the installed state");
    let fitted = objective.into_fitted().expect("outer fit was evaluated");
    let fitted_out = fitted.term.fitted();
    let r2 = reconstruction_r2(&fitted_out, &z);
    let converged_via = result
        .converged_via
        .expect("certified result carries its convergence verdict")
        .as_str();
    let criterion_certificate = result
        .criterion_certificate
        .as_ref()
        .expect("certified result carries its analytic criterion certificate")
        .summary();
    println!(
        "[#2228/#1095] d=1 K=1 IBP co-collapse: converged_via={converged_via} \
         iterations={} final_grad_norm={:?} certificate={criterion_certificate} \
         final_value={:.6e} recon_R2={:.6}",
        result.iterations, result.final_grad_norm, result.final_value, r2
    );
    assert!(
        result.final_value.is_finite() && result.final_value < 1.0e11,
        "d=1 K=1 fit terminated at the infeasible sentinel (final_value={:.6e}) — the inner \
         solve stalled under the gate co-collapse (#2228/#1095)",
        result.final_value
    );
    // The degree-2 patch spans the planted curve exactly. Reintroducing the old
    // double prior application multiplies the zero-logit posterior gate 0.5 by
    // the ordered prior mean 0.5, producing a 0.25-scaled pristine reconstruction:
    // `R² ≈ 0.4375 = 1 − (1 − 0.25)²`.
    assert!(
        r2 > 0.9,
        "d=1 K=1 reconstruction R²={r2:.6} < 0.9 — R²≈0.4375 identifies the old \
         double-prior gate scale and pristine-seed co-collapse (#2228/#1095)"
    );
}
