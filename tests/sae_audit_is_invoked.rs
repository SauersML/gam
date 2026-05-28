//! Verify that the channel-aware identifiability audit fires inside
//! `SaeManifoldTerm::run_joint_fit_arrow_schur`.
//!
//! # What is tested
//!
//! Two scenarios:
//!
//! 1. **Aliased atoms** — two decoder atoms with intentionally identical
//!    basis evaluations on the training rows.  Because the two atoms
//!    map to the same `(n·p, M·p)` effective Jacobian columns, the
//!    channel-aware audit detects the alias and `run_joint_fit_arrow_schur`
//!    returns an `Err` whose message mentions the identifiability failure.
//!
//! 2. **Non-aliased atoms** — two atoms with distinct, linearly independent
//!    basis evaluations.  The audit passes cleanly and the fit returns `Ok`.
//!
//! # Why this confirms the callback is load-bearing
//!
//! `decoder_parameter_block_specs()` is called inside
//! `run_joint_fit_arrow_schur` before the Newton loop.  The specs carry
//! `SaeDecoderBlockJacobian` callbacks (n_outputs = p), which makes
//! `canonicalize_for_identifiability` route through
//! `audit_identifiability_channel_aware`.  An alias that the flat audit
//! would have missed (due to orthogonal output channels) is correctly
//! classified; a true structural alias (same Jacobian column in the same
//! channel) is caught and surfaces as a fit error.

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
/// These are linearly independent over N=20 generic points, so the audit passes.
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

/// Aliased basis: both atoms return the identical polynomial basis {1, t, t²}.
/// In the channel-aware Jacobian (n·p, M·p), the two blocks produce the same
/// weighted columns when the assignments are equal, making the joint design
/// rank-deficient. The audit should detect this and the fit should fail.
fn aliased_phi() -> Array2<f64> {
    let mut phi = Array2::<f64>::zeros((N, M));
    for i in 0..N {
        let t = (i as f64 + 1.0) / (N as f64);
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
fn run_joint_fit_passes_with_distinct_atoms() {
    // Two atoms with distinct, linearly independent bases — audit should pass
    // cleanly and the fit should return Ok.
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
        "run_joint_fit_arrow_schur must succeed with distinct atoms; got: {:?}",
        result,
    );
}

#[test]
fn run_joint_fit_fails_with_aliased_atoms() {
    // Two atoms with identical basis evaluations — the channel-aware audit
    // detects the alias (both atoms produce the same Jacobian columns in each
    // output channel when the assignments are equal) and the fit must error.
    let phi = aliased_phi();
    let atom0 = make_atom("atom_alias_0", phi.clone());
    let atom1 = make_atom("atom_alias_1", phi);
    let assignment = make_assignment(2);
    let mut term = SaeManifoldTerm::new(vec![atom0, atom1], assignment).unwrap();
    let mut rho = make_rho(2);
    let target = zero_target();

    let result = term.run_joint_fit_arrow_schur(
        target.view(),
        &mut rho,
        None,
        1,
        1.0,
        1.0e-3,
        1.0e-3,
    );
    assert!(
        result.is_err(),
        "run_joint_fit_arrow_schur must fail with aliased atoms (identical bases, equal assignments); \
         got Ok(…)",
    );
    let msg = result.unwrap_err();
    assert!(
        msg.contains("identifiability"),
        "error message must mention identifiability; got: {msg}",
    );
}
