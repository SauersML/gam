//! gam#2731 — the structural-coherence scan's contribution cosine is read
//! from each atom's two SMALL factors, `⟨W_jᵀW_k, B_jB_kᵀ⟩_F` with the gated
//! basis `W_k = diag(a_·k)·Φ_k`, instead of the materialized `n × p`
//! contributions `Y_k = diag(a_·k)·Φ_k·B_k` the scan used to form for every
//! candidate atom. Same identity, same verdict: this pins the scan's reported
//! cosine against the materialized one over a sweep that crosses the bar in
//! both directions, so neither arm can pass vacuously.

#![cfg(test)]

use super::tests::{TestPeriodicEvaluator, periodic_basis};
use super::*;
use ndarray::{Array2, Array3, array};

/// Two periodic atoms sharing one decoder, their charts offset by `offset`.
/// At offset 0 the two gated contributions differ only through the gates
/// (cosine ≈ 1, a duplicate); at a large offset they reconstruct different
/// rows (cosine at the independence null, benign).
fn two_atom_term(offset: f64) -> SaeManifoldTerm {
    let base = array![[0.05], [0.20], [0.55], [0.80], [0.35], [0.65], [0.12], [0.93]];
    let coords0 = base.clone();
    let coords1 = base.mapv(|t: f64| (t + offset).rem_euclid(1.0));
    let (phi0, jet0) = periodic_basis(&coords0);
    let (phi1, jet1) = periodic_basis(&coords1);
    let m = phi0.ncols();
    let p = 5usize;
    // One full-rank decoder shared by both atoms, so their output frames
    // coincide and the PASS-1 frame prune admits the pair at every offset —
    // the contribution verdict is what decides, which is the point of the pin.
    let decoder = Array2::from_shape_fn((m, p), |(a, b)| {
        0.3 + 0.17 * a as f64 - 0.09 * b as f64 + 0.05 * ((a * 7 + b * 3) % 5) as f64
    });
    let make_atom = |name: &str, phi: Array2<f64>, jet: Array3<f64>| {
        SaeManifoldAtom::new_with_provided_function_gram(
            name,
            SaeAtomBasisKind::Periodic,
            1,
            phi,
            jet,
            decoder.clone(),
            Array2::<f64>::eye(m),
        )
        .unwrap()
        .with_basis_evaluator(Arc::new(TestPeriodicEvaluator))
    };
    let atom0 = make_atom("periodic0", phi0, jet0);
    let atom1 = make_atom("periodic1", phi1, jet1);
    let logits = array![
        [0.10, -0.05],
        [-0.08, 0.12],
        [0.15, 0.02],
        [-0.03, -0.14],
        [0.07, 0.09],
        [-0.11, 0.04],
        [0.13, -0.10],
        [0.01, 0.16]
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
    SaeManifoldTerm::new(vec![atom0, atom1], assignment).unwrap()
}

/// The materialized reference: form both `Y_k` in full and take their
/// Frobenius cosine, exactly as the scan did before gam#2731.
fn materialized_contribution_cosine(term: &SaeManifoldTerm) -> f64 {
    let gates = term.assignment.assignments();
    let contribution = |k: usize| -> Array2<f64> {
        let mut y = term.atoms[k]
            .basis_values
            .dot(term.atoms[k].decoder_coefficients());
        for (row, mut y_row) in y.rows_mut().into_iter().enumerate() {
            y_row.mapv_inplace(|value| value * gates[[row, k]]);
        }
        y
    };
    let (y0, y1) = (contribution(0), contribution(1));
    let dot: f64 = y0.iter().zip(y1.iter()).map(|(a, b)| a * b).sum();
    let norm0: f64 = y0.iter().map(|a| a * a).sum();
    let norm1: f64 = y1.iter().map(|b| b * b).sum();
    (dot / (norm0 * norm1).sqrt()).abs()
}

#[test]
fn the_scan_reports_the_materialized_contribution_cosine_2731() {
    // The derived bar for this shape: `D = min(M_j r_j, M_k r_k)` with a
    // 3-function periodic basis and a rank-3 shared output frame.
    let d_eff = 9.0_f64;
    let bar = 0.5 * ((2.0 / (std::f64::consts::PI * d_eff)).sqrt().min(1.0) + 1.0);

    let mut flagged = 0usize;
    let mut cleared = 0usize;
    for offset in [0.0_f64, 0.02, 0.15, 0.30] {
        let term = two_atom_term(offset);
        let reference = materialized_contribution_cosine(&term);
        assert!(
            reference.is_finite(),
            "offset {offset}: the materialized cosine must be finite"
        );
        let detected = term
            .structural_coherence_collapse_detected()
            .expect("the scan runs on a complete fitted state");
        if reference > bar {
            let (j, kk, coherence) = detected.unwrap_or_else(|| {
                panic!(
                    "offset {offset}: the materialized cosine {reference} is above the bar {bar}, \
                     so the pair must be reported"
                )
            });
            assert_eq!((j, kk), (0, 1), "offset {offset}: the only pair is (0, 1)");
            assert!(
                (coherence - reference).abs() <= 1e-12,
                "offset {offset}: small-Gram cosine {coherence} must equal the materialized \
                 cosine {reference}"
            );
            flagged += 1;
        } else {
            assert!(
                detected.is_none(),
                "offset {offset}: the materialized cosine {reference} is below the bar {bar}, \
                 so nothing may be reported; got {detected:?}"
            );
            cleared += 1;
        }
    }
    // Non-vacuity: the sweep crossed the bar, so the equality above was
    // exercised on a REPORTED pair and the clearing arm on a real one.
    assert!(
        flagged > 0 && cleared > 0,
        "the sweep must cross the bar in both directions (flagged {flagged}, cleared {cleared})"
    );
}
