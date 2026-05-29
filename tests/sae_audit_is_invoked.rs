//! Verify that the per-atom decoder identifiability audit fires inside
//! `SaeManifoldTerm::run_joint_fit_arrow_schur`.
//!
//! # What is tested
//!
//! Two scenarios:
//!
//! 1. **Well-posed atoms** — two decoder atoms whose per-atom weighted designs
//!    `D_k = diag(a_·k)·Φ_k` are each full column rank.  The audit passes
//!    cleanly and `run_joint_fit_arrow_schur` returns `Ok`.
//!
//! 2. **Rank-0 atom** — one atom whose basis evaluations are identically zero,
//!    so its weighted design `D_k` is rank 0.  The Arrow-Schur Newton system
//!    for that decoder block is singular, and the pre-fit audit surfaces this
//!    as an identifiability error before any Newton step.
//!
//! # Why this is the correct check
//!
//! The SAE decoder Hessian for atom `k` is `H_data = G_k ⊗ I_p` with
//! `G_k = D_kᵀ D_k`, so decoder identifiability is fully determined by the
//! per-atom `(n, M_k)` design `D_k` — the `p`-fold output replication carries
//! no extra structural information.  The audit therefore runs the pivoted-QR
//! rank check directly on each `D_k`, never materialising the (mis-specified)
//! `(n·p, M_k·p)` channel-replicated block that previously broadcast-panicked
//! when routed through the cross-block flat audit.

use gam::terms::sae_manifold::{
    AssignmentMode, SaeAssignment, SaeAtomBasisKind, SaeManifoldAtom, SaeManifoldRho,
    SaeManifoldTerm,
};
use ndarray::{Array1, Array2, Array3};

const N: usize = 20;
const M: usize = 3;
const P: usize = 2;
const LATENT_DIM: usize = 1;

fn make_atom(name: &str, phi: Array2<f64>) -> SaeManifoldAtom {
    let m = phi.ncols();
    let mut b = Array2::<f64>::zeros((m, P));
    // Give each atom a distinct, non-zero decoder so the Jacobian columns
    // are non-trivial. The atoms differ only in their basis evaluations.
    for mm in 0..m {
        for pp in 0..P {
            b[[mm, pp]] = (mm as f64 + 1.0) + (pp as f64) * 0.1;
        }
    }
    let penalty = Array2::<f64>::eye(m);
    let jet = Array3::<f64>::zeros((N, m, LATENT_DIM));
    SaeManifoldAtom::new(
        name,
        SaeAtomBasisKind::EuclideanPatch,
        LATENT_DIM,
        phi,
        jet,
        b,
        penalty,
    )
    .unwrap()
}

/// Build uniform-weight assignment (softmax logits = 0) for K atoms.
fn make_assignment(k_atoms: usize) -> SaeAssignment {
    let logits = Array2::<f64>::zeros((N, k_atoms));
    let coords: Vec<Array2<f64>> = (0..k_atoms)
        .map(|_| Array2::<f64>::zeros((N, LATENT_DIM)))
        .collect();
    SaeAssignment::from_blocks_with_mode(logits, coords, AssignmentMode::softmax(1.0)).unwrap()
}

fn make_rho(k_atoms: usize) -> SaeManifoldRho {
    let log_ard = (0..k_atoms)
        .map(|_| Array1::<f64>::zeros(LATENT_DIM))
        .collect();
    SaeManifoldRho::new(-2.0_f64.ln(), -2.0_f64.ln(), log_ard)
}

/// Distinct polynomial-like basis: atom 0 uses {1, t, t²}, atom 1 uses {1, 1-t, (1-t)²}.
/// These are each full column rank over N=20 generic points, so the per-atom
/// audit passes for both atoms.
fn distinct_phi(atom_idx: usize) -> Array2<f64> {
    let mut phi = Array2::<f64>::zeros((N, M));
    for i in 0..N {
        let t = if atom_idx == 0 {
            (i as f64 + 1.0) / (N as f64)
        } else {
            1.0 - (i as f64 + 1.0) / (N as f64)
        };
        phi[[i, 0]] = 1.0;
        phi[[i, 1]] = t;
        phi[[i, 2]] = t * t;
    }
    phi
}

/// A trivial target matrix (all zeros). The audit runs before any Newton step,
/// so the target only affects the loss, not the audit outcome.
fn zero_target() -> Array2<f64> {
    Array2::<f64>::zeros((N, P))
}

#[test]
fn run_joint_fit_passes_with_full_rank_atoms() {
    // Two atoms with distinct, full-column-rank weighted designs — the per-atom
    // audit passes cleanly and the fit returns Ok.
    let phi0 = distinct_phi(0);
    let phi1 = distinct_phi(1);
    let atom0 = make_atom("atom_a", phi0);
    let atom1 = make_atom("atom_b", phi1);
    let assignment = make_assignment(2);
    let mut term = SaeManifoldTerm::new(vec![atom0, atom1], assignment).unwrap();
    let mut rho = make_rho(2);
    let target = zero_target();

    let result = term.run_joint_fit_arrow_schur(
        target.view(),
        &mut rho,
        None,
        1,   // max_iter = 1 — we only need to confirm the audit fires
        1.0, // step_size
        1.0e-3,
        1.0e-3,
    );
    assert!(
        result.is_ok(),
        "run_joint_fit_arrow_schur must succeed with full-rank atoms; got: {:?}",
        result,
    );
}

#[test]
fn run_joint_fit_fails_with_rank_zero_atom() {
    // One atom whose basis evaluations are identically zero — its weighted
    // design D_k is rank 0, so the decoder block Hessian G_k ⊗ I_p is singular.
    // The pre-fit per-atom audit must catch this and the fit must error before
    // any Newton step.
    let phi0 = distinct_phi(0);
    let phi_zero = Array2::<f64>::zeros((N, M));
    let atom0 = make_atom("atom_ok", phi0);
    let atom1 = make_atom("atom_degenerate", phi_zero);
    let assignment = make_assignment(2);
    let mut term = SaeManifoldTerm::new(vec![atom0, atom1], assignment).unwrap();
    let mut rho = make_rho(2);
    let target = zero_target();

    let result =
        term.run_joint_fit_arrow_schur(target.view(), &mut rho, None, 1, 1.0, 1.0e-3, 1.0e-3);
    assert!(
        result.is_err(),
        "run_joint_fit_arrow_schur must fail when an atom has a rank-0 weighted design; got Ok(…)",
    );
    let msg = result.unwrap_err();
    assert!(
        msg.contains("identifiability"),
        "error message must mention identifiability; got: {msg}",
    );
}
