//! Regression gate for GitHub issue #1407 — "Fixed-decoder SAE encoding
//! assembles and discards full decoder-parameter tier".
//!
//! `SaeManifoldTerm::run_fixed_decoder_arrow_schur` freezes the decoder and
//! updates only the per-row latent ext-coordinates via the per-row `(htt, gt)`
//! block-diagonal. Before the #1407 fix it called the FULL joint assembler
//! every iteration, materialising and discarding the entire `K`-dependent
//! decoder β tier (`G`/`g_β`/`H_tβ`/dense `h_ββ`/β-penalties) it never reads.
//!
//! The fix routes that path through a LEAN fixed-decoder assembler that builds
//! ONLY the per-row `htt`/`gt` blocks (zero shared-border / cross-block /
//! decoder-tier work). This is a perf/memory fix: the job of THIS test is to
//! PROVE the lean path is NUMERICALLY EQUIVALENT to the old full-assembler
//! path, so a future refactor cannot silently diverge the two.
//!
//! Strategy: assemble the system both ways at the same term/ρ and compute the
//! fixed-decoder latent step from each. The fixed-decoder step reads only the
//! per-row `htt`/`gt`, which the lean assembler builds identically to the full
//! one, so the two steps must be BIT-IDENTICAL. We also drive the full
//! `run_fixed_decoder_arrow_schur` encode (which now uses the lean assembler)
//! and pin that it runs to a finite loss with the decoder left exactly frozen.
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

/// #1407 core equivalence: the LEAN fixed-decoder assembler (per-row htt/gt
/// only, β tier elided) must produce a fixed-decoder latent step that is
/// BIT-IDENTICAL to the one produced via the FULL joint assembler (which also
/// builds — and the fixed-decoder step discards — the entire decoder β tier).
///
/// This is the perf-fix's correctness contract: we are only skipping wasted
/// decoder-tier work, not changing the math. Identical steps ⇒ the lean path
/// is a faithful drop-in for the full path on the encode walk.
#[test]
fn fixed_decoder_lean_step_equals_full_step_1407() {
    let (mut term, target, rho) = two_atom_periodic_term();

    let (lean_step, full_step) = term
        .fixed_decoder_step_lean_vs_full_1407(target.view(), &rho, None, 1.0e-6)
        .expect("both fixed-decoder assemblies + steps succeed");

    assert_eq!(
        lean_step.len(),
        full_step.len(),
        "lean and full fixed-decoder steps must have the same dimension"
    );
    assert!(
        !lean_step.is_empty(),
        "the fixed-decoder latent step must be non-empty for this fixture"
    );
    assert!(
        lean_step.iter().all(|v| v.is_finite()) && full_step.iter().all(|v| v.is_finite()),
        "both fixed-decoder steps must be finite"
    );
    // Bit-identical: the lean assembler builds htt/gt exactly as the full path
    // does; it only elides the β decoder tier the fixed-decoder step never
    // reads. Any divergence here is a #1407 regression (β-tier work leaking
    // into the per-row latent blocks).
    for (idx, (lean, full)) in lean_step.iter().zip(full_step.iter()).enumerate() {
        assert_eq!(
            lean, full,
            "lean vs full fixed-decoder step component {idx} diverged \
             (lean {lean} != full {full}); the lean assembler must be \
             numerically identical to the full one on htt/gt (#1407)"
        );
    }
}

/// #1407 end-to-end: the fixed-decoder encode (now driven through the lean
/// assembler) runs to a finite loss and leaves the decoder EXACTLY frozen.
/// A regression that wires the β tier back into this path (recomputing /
/// updating the decoder) would perturb `flatten_beta()` and fail here.
#[test]
fn fixed_decoder_encode_via_lean_assembler_keeps_decoder_frozen_1407() {
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
    for (idx, (a, b)) in beta_after.iter().zip(beta_before.iter()).enumerate() {
        assert_eq!(
            a, b,
            "decoder coefficient {idx} changed during fixed-decoder encode \
             ({b} -> {a}); the decoder must stay frozen (#1407)"
        );
    }
}
