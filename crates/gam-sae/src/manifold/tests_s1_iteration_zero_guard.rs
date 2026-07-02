//! `co_collapse_ev_arm_is_disarmed_at_iteration_zero_s1`, split out of `tests.rs`
//! to keep that tracked file under the #780 10k-line gate. Declared as a sibling
//! `#[cfg(test)] mod` in `mod.rs`; the shared `periodic_basis` fixture helper and
//! the `TestPeriodicEvaluator` are sourced from the sibling `tests` module.

use super::*;
use super::tests::{TestPeriodicEvaluator, periodic_basis};
use ndarray::{Array2, Array3, array};
use std::sync::Arc;

/// S1 (guard surgery) — the EV co-collapse arm must be DISARMED at iteration 0.
/// The entry (iteration-0) guard call evaluates the cold SEED; a cold seed sits
/// below any bar by definition, so evaluating the EV arm there made every real
/// K≥2 fit open by burning the reseed budget and recording Terminal collapse
/// events before the first Newton step (the "co-collapse" opening log). This pins
/// the fix: a genuinely co-collapsed (output-vanished, no relative-norm breach)
/// dictionary is left UNTOUCHED at iteration 0, and the SAME state reseeds at
/// iteration 1 — checking the seed against a bar checks coldness, not health.
#[test]
pub(crate) fn co_collapse_ev_arm_is_disarmed_at_iteration_zero_s1() {
    // Two periodic atoms with TINY-but-nonzero EQUAL decoders: median decoder norm
    // is positive (so the `median == 0` cold-seed early return does NOT fire) and
    // no atom is relatively behind its peer (no relative-norm breach), so the
    // dictionary reaches the absolute-EV co-collapse arm — the arm the iteration-0
    // gate protects. The output is ≈ 0, and the target has zero column means, so
    // both the EV and the output energy sit at the null floor.
    let coords0 = array![[0.05_f64], [0.20], [0.55], [0.80], [0.35], [0.65]];
    let coords1 = array![[0.15_f64], [0.30], [0.65], [0.90], [0.45], [0.10]];
    let (phi0, jet0) = periodic_basis(&coords0);
    let (phi1, jet1) = periodic_basis(&coords1);
    let make_atom = |name: &str, phi: Array2<f64>, jet: Array3<f64>, scale: f64| {
        SaeManifoldAtom::new(
            name,
            SaeAtomBasisKind::Periodic,
            1,
            phi,
            jet,
            Array2::<f64>::from_elem((3, 3), scale),
            Array2::<f64>::eye(3),
        )
        .unwrap()
        .with_basis_evaluator(Arc::new(TestPeriodicEvaluator))
    };
    let atom0 = make_atom("periodic0", phi0, jet0, 1.0e-5);
    let atom1 = make_atom("periodic1", phi1, jet1, 1.1e-5);
    let logits = array![
        [0.6, -0.2],
        [0.1, 0.4],
        [-0.3, 0.5],
        [0.4, 0.1],
        [0.2, 0.3],
        [-0.1, 0.4]
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
    .unwrap();
    let mut term = SaeManifoldTerm::new(vec![atom0, atom1], assignment).unwrap();
    // Zero-column-mean target (p=3, three distinct residual PCs for the disjoint-PC
    // reseed): a vanished dictionary (fitted ≈ 0 ≈ mean) then has ≈ zero EV AND
    // ≈ zero output energy, both at or below the null floor `q / n`.
    let target = array![
        [0.40, -0.10, 0.05],
        [-0.20, 0.35, -0.15],
        [0.10, 0.05, 0.30],
        [0.25, -0.30, -0.05],
        [-0.15, 0.20, 0.18],
        [-0.40, -0.20, -0.33]
    ];
    let rho = SaeManifoldRho::new(
        (-0.3_f64).exp().ln(),
        0.7_f64.ln(),
        vec![array![0.9_f64.ln()], array![1.1_f64.ln()]],
    );

    // Precondition: genuinely co-collapsed (output-vanished) at the null floor.
    let ev = term
        .dictionary_reconstruction_ev(target.view(), &rho)
        .expect("EV evaluates");
    let out_ratio = term
        .dictionary_reconstruction_output_energy_ratio(target.view(), &rho)
        .expect("output-energy ratio evaluates");
    let q = crate::manifold::outer_objective::reachable_dictionary_rank(
        &term.atoms,
        term.n_obs(),
        target.ncols(),
    );
    let floor =
        crate::manifold::outer_objective::absolute_degeneracy_ev_floor(target.view(), q);
    assert!(
        ev <= floor && out_ratio <= floor,
        "precondition: co-collapsed state must sit at the null floor \
         (EV={ev:.4}, out_ratio={out_ratio:.4}, floor={floor:.4})"
    );

    let before: Vec<Array2<f64>> = term
        .atoms
        .iter()
        .map(|a| a.decoder_coefficients.clone())
        .collect();

    // ── iteration 0: the EV arm is DISARMED — no reseed, no event, state frozen ──
    term.enforce_decoder_norm_guard(target.view(), 0, &rho)
        .expect("guard must not error at iteration 0");
    assert!(
        term.collapse_events().is_empty(),
        "iteration-0 EV arm must record NO collapse event on a cold co-collapsed seed; \
         events: {:?}",
        term.collapse_events()
    );
    for (atom, b) in before.iter().enumerate() {
        assert_eq!(
            &term.atoms[atom].decoder_coefficients, b,
            "iteration-0 guard must leave atom {atom}'s decoder untouched"
        );
    }

    // ── iteration 1: the SAME state now reseeds (a post-entry stall is genuine) ──
    term.enforce_decoder_norm_guard(target.view(), 1, &rho)
        .expect("guard must recover at iteration 1");
    for atom in 0..2 {
        let reseeded = term
            .collapse_events()
            .iter()
            .any(|e| e.atom == atom && e.action == CollapseAction::Reseeded);
        assert!(
            reseeded,
            "at iteration 1 the genuine co-collapse must reseed atom {atom}; events: {:?}",
            term.collapse_events()
        );
    }
}
