//! Regression pin for #1095 / #2228 — an over-parametrized circle atom (a d=2
//! Euclidean chart carrying intrinsically 1-D circular data) must FIT through the
//! production outer engine at ridge-0, not error out.
//!
//! Root cause (perf1689): the inner `reml_criterion` accepts the converged inner
//! `(t, β)` optimum through the stall-acceptance certificates (KKT grad-norm and
//! the #2226 affine Newton-decrement ½λ²), but both were computed ONLY from the
//! ridge-0 undamped factorization of the stationary system. A d=2 chart on
//! intrinsically 1-D data plants a rank-1 RADIAL null in every per-row `H_tt`
//! (all rows sit on the unit circle in the (t₁, t₂) latent plane, so scaling the
//! radius is unpopulated), so that undamped per-row Cholesky is non-PD BY
//! CONSTRUCTION — the factorization errored, every acceptance certificate was
//! skipped, and the fit was refused to the non-convergence sentinel → the public
//! `sae_manifold_fit` K=1 circle returned a `GamError` at every N.
//!
//! The fix routes the stall-acceptance factorization through the SAME per-row
//! spectral-deflation the evidence log-det uses (the radial null is unit-stiffness
//! deflated → `log 1 = 0`, ρ-independent), so acceptance is REACHABLE and the
//! affine ½λ² is measured on the identifiable (tangent) subspace. An
//! over-parametrized latent is a legitimate configuration: the fit must land the
//! best incumbent on the deflated subspace, not abort.
//!
//! This uses `RIDGE_EXT_COORD = 0.0` deliberately — the sibling circle pins mask
//! this path with a hand ridge of `1e-6`. It drives the fit exactly the way
//! production does (`OuterProblem::run` around `SaeManifoldOuterObjective`) and
//! asserts the cascade COMPLETES with a finite criterion (a `GamError` / infeasible
//! sentinel reproduces #1095/#2228), then re-evaluates the criterion directly at
//! the converged ρ to pin the criterion PATH finite too.

use gam::linalg::faer_ndarray::{FaerCholesky, fast_ata, fast_atb};
use gam::solver::rho_optimizer::OuterProblem;
use gam::terms::latent::LatentManifold;
use gam::terms::sae::manifold::EuclideanPatchEvaluator;
use gam::terms::{
    AssignmentMode, SaeAssignment, SaeAtomBasisKind, SaeBasisEvaluator, SaeManifoldAtom,
    SaeManifoldOuterObjective, SaeManifoldRho, SaeManifoldTerm,
};
use ndarray::{Array1, Array2};
use std::sync::Arc;

use faer::Side as FaerSide;

const N: usize = 180; // the #1095 L44 color-bank size
const P: usize = 24; // PCA ambient
const D: usize = 2; // OVER-PARAMETRIZED chart: 2-D latent on intrinsic 1-D data
const MAX_DEGREE: usize = 2; // degree-2 patch {1, t₁, t₂, t₁², t₁t₂, t₂²}
const TAU: f64 = 0.5;
const ALPHA: f64 = 1.0;
const INNER_MAX_ITER: usize = 50;
const LEARNING_RATE: f64 = 1.0;
const RIDGE_EXT_COORD: f64 = 0.0; // the ridge-0 path the sibling pins mask with 1e-6
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

/// A circle planted in the (axis 0, axis 1) 2-plane of `R^P`, plus small ambient
/// noise. Genuinely circular (a fit exists); the difficulty is purely the d=2
/// chart's radial null, not representability.
fn planted_circle() -> Array2<f64> {
    let mut z = Array2::<f64>::zeros((N, P));
    for i in 0..N {
        let theta = std::f64::consts::TAU * ((i as f64) * 0.061_803 + 0.13).rem_euclid(1.0);
        z[[i, 0]] = theta.cos();
        z[[i, 1]] = theta.sin();
        z[[i, 2]] = 0.4 * (2.0 * theta).cos();
        for col in 0..P {
            z[[i, col]] += 0.03 * idx_noise((i as u64) * 31 + col as u64);
        }
    }
    z
}

/// Latent coords seeded as the 2-D circle embedding `(cos θ, sin θ)` recovered
/// from each row's own (axis 0, axis 1) projection. Every row lands on the unit
/// circle in the latent plane, so the RADIAL direction is unpopulated — exactly
/// the rank-1 per-row `H_tt` null the fix must deflate.
fn embedded_circle_coords(z: &Array2<f64>) -> Array2<f64> {
    Array2::from_shape_fn((N, D), |(i, axis)| {
        let theta = z[[i, 1]].atan2(z[[i, 0]]);
        if axis == 0 { theta.cos() } else { theta.sin() }
    })
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

fn build_cold_circle_d2_term(z: &Array2<f64>) -> SaeManifoldTerm {
    let evaluator = Arc::new(EuclideanPatchEvaluator::new(D, MAX_DEGREE).unwrap());
    let n_basis = evaluator.basis_size();
    let coords = embedded_circle_coords(z);
    let (phi, jet) = evaluator.evaluate(coords.view()).unwrap();
    let decoder = decoder_lsq_init(&phi, z);
    let atom = SaeManifoldAtom::new(
        "circle_d2".to_string(),
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
fn sae_manifold_circle_d2_ridge0_fits() {
    let z = planted_circle();
    let term = build_cold_circle_d2_term(&z);
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
    // A `GamError` here (not a finite fit) is the #1095/#2228 refusal.
    let result = problem
        .run(&mut objective, "SAE d=2 circle ridge-0 fits (#1095/#2228)")
        .expect(
            "outer cascade must complete on an over-parametrized (d=2) circle chart at ridge-0 \
             — the rank-1 radial-null refusal reproduces #1095/#2228",
        );
    let fitted = objective.into_fitted();
    let mut fitted_term = fitted.term;
    let fitted_out = fitted_term.fitted();
    let r2 = reconstruction_r2(&fitted_out, &z);
    // Re-evaluate the criterion directly at the converged ρ: the criterion PATH
    // itself (not just the optimizer's bookkeeping) must return a finite score for
    // this feasible over-parametrized fit.
    let (recheck_cost, _loss) = fitted_term
        .reml_criterion(
            z.view(),
            &fitted.rho,
            None,
            INNER_MAX_ITER,
            LEARNING_RATE,
            RIDGE_EXT_COORD,
            RIDGE_BETA,
        )
        .expect("outer criterion must EVALUATE (not refuse) at the converged d=2 circle optimum");
    println!(
        "[#1095/#2228] d=2 circle ridge-0 fit: final_value={:.6e} recheck_criterion={:.6e} recon_R2={:.6}",
        result.final_value, recheck_cost, r2
    );
    assert!(
        result.final_value.is_finite() && result.final_value < 1.0e11,
        "d=2 circle ridge-0 fit terminated at the infeasible sentinel (final_value={:.6e}) — \
         the rank-1 radial-null factorization refusal (#1095/#2228)",
        result.final_value
    );
    assert!(
        recheck_cost.is_finite() && recheck_cost < 1.0e11,
        "d=2 circle ridge-0 outer criterion re-evaluated to the infeasible sentinel at the \
         converged optimum (recheck_criterion={recheck_cost:.6e}) — a feasible over-parametrized \
         chart reporting as infeasible (#1095/#2228)"
    );
    // The circle lies exactly in a 2-plane, so even the over-parametrized d=2
    // chart reconstructs it once the radial null is deflated rather than refused.
    assert!(
        r2 > 0.9,
        "d=2 circle reconstruction R²={r2:.6} < 0.9 — the circle was not recovered on the \
         deflated identifiable subspace"
    );
}

/// zz_ loop discriminator (#1095/#2228): is the failing R²≈0.4347 the gate-scaled
/// PRISTINE SEED reconstruction rather than any fitted state? For K=1 with α=1 the
/// ordered-IBP prior mass is π₀=(α/(α+1))¹=0.5, and the seed logits are 0, so the
/// gate is a₁=σ(0)·π₀=0.5·0.5=0.25. `decoder_lsq_init` fits the decoder UNGATED
/// (z≈Φ·B), so the gated reconstruction is 0.25·Φ·B≈0.25·z on the signal columns →
/// R²≈1−(1−0.25)²=0.4375, which matches the measured 0.434688. If `into_fitted`
/// returns the pristine seed (because the settled curved fit reconstructs even
/// worse — the IBP-gate/decoder co-collapse), the reported R² is this FIXED seed
/// value, byte-identical across builds and INDEPENDENT of any inner-solve
/// coordinate fix. This prints the cold-seed R², the returned R², and which
/// into_fitted fallback fired so the loop can confirm the binding mechanism is the
/// gate/decoder co-collapse, not the radial-coordinate drift. Pure measurement —
/// no assertion (never gates CI).
#[test]
fn zz_1095_2228_measure_seed_vs_settled_r2() {
    let z = planted_circle();
    // Cold seed, NO outer fit: gate a₁=σ(0)·π₀=0.25 applied to the ungated LSQ decoder.
    let seed_term = build_cold_circle_d2_term(&z);
    let seed_r2 = reconstruction_r2(&seed_term.fitted(), &z);

    let term = build_cold_circle_d2_term(&z);
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
    let result = problem
        .run(&mut objective, "SAE d=2 circle ridge-0 measure (#1095/#2228)")
        .expect("outer cascade completes");
    let fitted = objective.into_fitted();
    let used_pristine = fitted.used_pristine_seed_fallback;
    let used_seed_basin = fitted.used_seed_basin_fallback;
    let charts_canonicalized = fitted.charts_canonicalized;
    let mut fitted_term = fitted.term;
    let returned_r2 = reconstruction_r2(&fitted_term.fitted(), &z);
    println!(
        "[#1095/#2228 measure] cold_seed_R2={seed_r2:.6} returned_R2={returned_r2:.6} \
         final_value={:.6e} used_pristine_seed_fallback={used_pristine} \
         used_seed_basin_fallback={used_seed_basin} charts_canonicalized={charts_canonicalized} \
         (a1=σ(0)·π0=0.25 → predicted gated-seed R2≈0.4375)",
        result.final_value
    );
}
