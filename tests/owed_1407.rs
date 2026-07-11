//! Owed-work regression gate for GitHub issue #1407.
//!
//! Fix landed in commit 5c4a7c3b3 ("fix(#1407): fixed-decoder SAE encode
//! assembles only per-row htt+gt, skips the decoder tier"). The fixed-decoder
//! encode path (`SaeManifoldTerm::run_fixed_decoder_arrow_schur`) freezes the
//! decoder and updates only the per-row latent ext-coordinates via the per-row
//! `(htt, gt)` block-diagonal. Before the fix it called the FULL joint
//! assembler every iteration, materialising and discarding the entire
//! K-dependent decoder β tier (`G`/`g_β`/`H_tβ`/dense `h_ββ`/β-penalties) it
//! never reads.
//!
//! This test pins the user-observable contract of the fix through the public
//! API only (so it survives internal refactors of the private assembler): the
//! fixed-decoder encode must run to completion with a finite loss and leave the
//! decoder coefficients EXACTLY frozen. A regression that wires the β tier back
//! into this path (recomputing/updating the decoder) would perturb
//! `flatten_beta()` and fail here.
//!
//! No `let _`, no `#[allow(...)]`, no env vars, no `#[cfg(feature=...)]`.

use std::sync::Arc;

use ndarray::{Array1, Array2, Array3, array};

use gam::terms::latent::LatentManifold;
use gam::terms::sae::assignment::{AssignmentMode, SaeAssignment};
use gam::terms::sae::basis::PeriodicHarmonicEvaluator;
use gam::terms::sae::manifold::{
    SaeAtomBasisKind, SaeManifoldAtom, SaeManifoldRho, SaeManifoldTerm,
};

/// The rank-1 periodic harmonic basis `[1, sin(2πx), cos(2πx)]` matching the
/// `PeriodicHarmonicEvaluator(3)` width, with its first jet in the single
/// latent axis. This is the same basis shape the crate's own SAE fixtures use.
fn periodic_basis(coords: &Array2<f64>) -> (Array2<f64>, Array3<f64>) {
    let n = coords.nrows();
    let mut phi = Array2::<f64>::zeros((n, 3));
    let mut jet = Array3::<f64>::zeros((n, 3, 1));
    for row in 0..n {
        let x = coords[[row, 0]].rem_euclid(1.0);
        let angle = 2.0 * std::f64::consts::PI * x;
        phi[[row, 0]] = 1.0;
        phi[[row, 1]] = angle.sin();
        phi[[row, 2]] = angle.cos();
        jet[[row, 1, 0]] = 2.0 * std::f64::consts::PI * angle.cos();
        jet[[row, 2, 0]] = -2.0 * std::f64::consts::PI * angle.sin();
    }
    (phi, jet)
}

/// Build a small two-atom periodic SAE term over `p = 1` output channel using
/// only the public crate API.
fn two_atom_periodic_term() -> (SaeManifoldTerm, Array2<f64>, SaeManifoldRho) {
    let coords0 = array![[0.05], [0.20], [0.55], [0.80], [0.35]];
    let coords1 = array![[0.15], [0.30], [0.65], [0.90], [0.45]];
    let (phi0, jet0) = periodic_basis(&coords0);
    let (phi1, jet1) = periodic_basis(&coords1);

    let atom0 = SaeManifoldAtom::new_with_provided_function_gram(
        "periodic0",
        SaeAtomBasisKind::Periodic,
        1,
        phi0,
        jet0,
        array![[0.25], [-0.35], [0.15]],
        Array2::<f64>::eye(3),
    )
    .expect("atom0 constructs")
    .with_basis_evaluator(Arc::new(
        PeriodicHarmonicEvaluator::new(3).expect("periodic evaluator width 3"),
    ));
    let atom1 = SaeManifoldAtom::new_with_provided_function_gram(
        "periodic1",
        SaeAtomBasisKind::Periodic,
        1,
        phi1,
        jet1,
        array![[-0.10], [0.20], [0.30]],
        Array2::<f64>::eye(3),
    )
    .expect("atom1 constructs")
    .with_basis_evaluator(Arc::new(
        PeriodicHarmonicEvaluator::new(3).expect("periodic evaluator width 3"),
    ));

    let logits = array![
        [0.7, -0.2],
        [0.1, 0.4],
        [-0.3, 0.5],
        [0.6, -0.1],
        [0.2, 0.3]
    ];
    let assignment = SaeAssignment::from_blocks_with_mode_and_manifolds(
        logits,
        vec![coords0, coords1],
        vec![
            LatentManifold::Circle { period: 1.0 },
            LatentManifold::Circle { period: 1.0 },
        ],
        AssignmentMode::softmax(0.8),
    )
    .expect("assignment constructs");

    let term = SaeManifoldTerm::new(vec![atom0, atom1], assignment).expect("term constructs");
    let target = array![[0.12], [-0.03], [0.08], [0.20], [-0.11]];
    let rho = SaeManifoldRho::new(
        -0.3_f64,
        0.7_f64.ln(),
        vec![array![0.9_f64.ln()], array![1.1_f64.ln()]],
    );
    (term, target, rho)
}

/// #1407: the fixed-decoder encode path must keep the decoder EXACTLY frozen.
///
/// The whole point of the fix is that `run_fixed_decoder_arrow_schur` updates
/// only the per-row latent ext-coordinates and never touches the decoder β
/// tier. So `flatten_beta()` (the flat decoder-coefficient vector) must be
/// byte-identical before and after the encode, while the encode still produces
/// a finite loss. A regression that re-engages the β tier and updates the
/// decoder on this path would move `flatten_beta()` and fail.
#[test]
fn fixed_decoder_encode_keeps_decoder_frozen_1407() {
    let (mut term, target, mut rho) = two_atom_periodic_term();

    let beta_before: Array1<f64> = term.flatten_beta();
    assert!(
        beta_before.iter().all(|v| v.is_finite()) && !beta_before.is_empty(),
        "the seeded decoder tier must be finite and non-empty"
    );

    let loss = term
        .run_fixed_decoder_arrow_schur(target.view(), &mut rho, None, 8, 1.0, 1.0e-6)
        .expect("fixed-decoder encode runs to completion");
    assert!(
        loss.total().is_finite(),
        "fixed-decoder encode must yield a finite loss; got {}",
        loss.total()
    );

    let beta_after: Array1<f64> = term.flatten_beta();
    assert_eq!(
        beta_after.len(),
        beta_before.len(),
        "decoder dimension must not change across the fixed-decoder encode"
    );
    // The decoder is FROZEN: every coefficient is bit-identical. Any β-tier
    // update wired back into the fixed-decoder path (the #1407 defect) would
    // perturb at least one of these.
    for (idx, (a, b)) in beta_after.iter().zip(beta_before.iter()).enumerate() {
        assert_eq!(
            a, b,
            "decoder coefficient {idx} changed during fixed-decoder encode \
             ({b} -> {a}); the decoder must stay frozen (#1407)"
        );
    }
}
