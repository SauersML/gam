//! Honest-red repro pin for #2228 / #1095 / #2226 (K=1 d_atom=1 IBP-gate ↔
//! decoder co-collapse) — the ACTUAL issue-body configuration.
//!
//! Root cause (verified by code-trace on HEAD; see the fleet report): the IBP-MAP
//! gate is `a_k = σ(l_k/τ)·π_k` with the ordered stick-breaking prior mean
//! `π_k = (α/(α+1))^{k+1}` (`assignment.rs` `ibp_map_row` / `ordered_prior_means`).
//! For K=1 at the production `α = 1`, `π_1 = 0.5`, so the gate is HARD-CAPPED at
//! `a_1 = σ(l_1/τ)·0.5 ≤ 0.5` even with the logit driven to `+∞`; the repro
//! (`ibp_map(TAU, ALPHA, /*learnable_alpha=*/false)`) freezes `α`, pinning the cap
//! at `0.5`. By the gate's OWN design contract ("magnitude lives in the decoder,
//! the gate stays in [0, 1)", `assignment.rs` `jumprelu_row`), the decoder `B` must
//! carry the compensating `1/a_1 ≥ 2×` magnitude so the gated reconstruction
//! `a_1·Φ·B` reaches `z`.
//!
//! The bug is a gate-weighting ASYMMETRY between the two decoder terms the joint
//! solve assembles:
//!   * the decoder DATA-fit β-Hessian is gate-weighted `a_k²` — the design is
//!     `D_k = diag(a_·k)·Φ_k` (`construction_arrow_schur_assembly.rs`,
//!     `w = a_k·φ·…`), so the β Gram is `Σ_i a_{ik}² φ_iφ_iᵀ`;
//!   * the decoder SMOOTHNESS penalty `S_k` is scaled by `λ_smooth[k]` ONLY, NOT
//!     by the gate (`construction_arrow_schur_assembly.rs` `scaled_s = λ·S`).
//! So the effective decoder shrinkage is `λ/a_k²`, which EXPLODES as `a_k → 0.25`:
//! the penalized decoder argmin is over-shrunk ≥ 16× relative to its data weight,
//! the gated reconstruction cannot reach `z`, and the inner Newton KKT residual
//! cannot be driven to tolerance (the reported `‖g‖ ≈ 1e1–1e2` stall →
//! `RemlConvergenceError`). `into_fitted`'s pristine-seed guard then returns the
//! cold seed, whose gated reconstruction is exactly `a_1·z = 0.25·z`, giving the
//! `R² ≈ 1 − (1 − 0.25)² = 0.4375` co-collapse signature. `gauge.rs`
//! `optimize_log_amplitudes_closed_form` documents the same mechanism from the
//! amplitude side ("s DRIFTS to collapse under the penalty's residual shape
//! gradient … instead of co-vanishing") but its retraction is a `k < 2` early-out
//! and is gated behind the default-OFF `quotient_scale`, so the K=1 default fit
//! path never compensates the gate.
//!
//! This pin plants a 1-D curve that a degree-2 Euclidean patch spans EXACTLY, so
//! the ONLY obstacle to `R² ≈ 1` is the K=1 gate co-collapse (the basis is not a
//! confound, unlike a circle a `d = 1` patch cannot represent). A healthy fit
//! compensates the `0.5` gate cap and recovers the curve; the bug reproduces as a
//! `RemlConvergenceError` (inner solve stalls) or an `R² ≈ 0.4375` pristine-seed
//! fallback. Drives the public outer engine at ridge-0 exactly as production does.

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

const N: usize = 160;
const P: usize = 12; // ambient output dim
const D: usize = 1; // the #2228 configuration: intrinsic 1-D atom
const MAX_DEGREE: usize = 2; // degree-2 patch {1, t, t²} — spans the planted curve
const TAU: f64 = 0.5;
const ALPHA: f64 = 1.0; // K=1 ⇒ π_1 = (α/(α+1))^1 = 0.5, so the gate caps at 0.5
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
        .run(&mut objective, "SAE d=1 K=1 IBP-gate co-collapse (#2228/#1095)")
        .expect(
            "outer cascade must complete on a K=1 d=1 atom whose degree-2 patch spans its \
             planted curve — a RemlConvergenceError (inner solve stalls at fixed ρ) reproduces \
             the #2228 IBP-gate/decoder co-collapse",
        );
    let fitted = objective.into_fitted();
    let fitted_out = fitted.term.fitted();
    let r2 = reconstruction_r2(&fitted_out, &z);
    println!(
        "[#2228/#1095] d=1 K=1 IBP co-collapse: final_value={:.6e} recon_R2={:.6} \
         used_pristine_seed_fallback={}",
        result.final_value, r2, fitted.used_pristine_seed_fallback
    );
    assert!(
        result.final_value.is_finite() && result.final_value < 1.0e11,
        "d=1 K=1 fit terminated at the infeasible sentinel (final_value={:.6e}) — the inner \
         solve stalled under the gate co-collapse (#2228/#1095)",
        result.final_value
    );
    // The degree-2 patch spans the planted curve exactly, so a fit that COMPENSATES
    // the 0.5 gate cap recovers it. `R² ≈ 0.4375 = 1 − (1 − 0.25)²` is the
    // gate-scaled pristine-seed signature: the decoder co-collapsed rather than
    // scaling to `1/a_1` to undo the gate.
    assert!(
        r2 > 0.9,
        "d=1 K=1 reconstruction R²={r2:.6} < 0.9 — the decoder did not compensate the capped \
         IBP gate (R²≈0.4375 is the gate-scaled pristine seed; #2228/#1095 co-collapse)"
    );
}
