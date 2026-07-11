//! #2134 (walls 1+2) / #1893 — the CHUNKED-SEED cold start for the overcomplete
//! (`K > P`) hard-TopK curved lane must produce the SAME per-atom decoder seed
//! as the dense full-height build, and the front door must now ADMIT the shape
//! it once refused.
//!
//! [`SaeManifoldTerm::seed_cold_start_disjoint_charts_streaming`] places the
//! identical charts as the dense
//! [`SaeManifoldTerm::seed_cold_start_disjoint_charts`] (shared
//! `seed_atom_chart_coords`) and fits each atom's decoder from the row-chunked
//! normal equations instead of a resident `(N × M_k)` design. This pins the
//! end-to-end parity of the two seeds on a `K > P` TopK term — the size where
//! both run — so the streaming lane the front door now admits is trusted to
//! reproduce the dense seed.

use super::*;
use ndarray::Array1;
use std::sync::Arc;

/// Build a `K > P` hard-TopK manifold term over `EuclideanPatch` atoms, with a
/// per-row routing window that selects a moving support so every atom fires on
/// enough rows to have a full-rank decoder design. Returns the freshly
/// constructed term (decoders zero), the target `Z`, and a ρ.
fn build_topk_term(
    n: usize,
    p: usize,
    k: usize,
    support_k: usize,
) -> (SaeManifoldTerm, Array2<f64>, SaeManifoldRho) {
    let evaluator = Arc::new(EuclideanPatchEvaluator::new(1, 2).expect("evaluator"));
    let mut atoms = Vec::with_capacity(k);
    let mut coord_blocks = Vec::with_capacity(k);
    let mut manifolds = Vec::with_capacity(k);
    for atom_idx in 0..k {
        // A distinct, deterministic 1-D chart per atom.
        let mut coords = Array2::<f64>::zeros((n, 1));
        for row in 0..n {
            let t = row as f64 / n as f64;
            coords[[row, 0]] = (t - 0.5) * 2.0 + 0.13 * atom_idx as f64;
        }
        let (phi, jet) = evaluator.evaluate(coords.view()).expect("basis");
        let m = phi.ncols();
        let smooth_penalty =
            gam_terms::basis::create_difference_penalty_matrix(m, 2, None).expect("penalty");
        let atom = SaeManifoldAtom::new(
            "topk-patch",
            SaeAtomBasisKind::EuclideanPatch,
            1,
            phi,
            jet,
            Array2::<f64>::zeros((m, p)),
            smooth_penalty,
        )
        .expect("atom")
        .with_basis_second_jet(evaluator.clone());
        atoms.push(atom);
        coord_blocks.push(coords);
        manifolds.push(LatentManifold::Euclidean);
    }

    // Routing logits: favour the atoms nearest a per-row moving centre, so the
    // hard TopK support rotates across all K atoms over the corpus.
    let mut logits = Array2::<f64>::zeros((n, k));
    for row in 0..n {
        let centre = (row as f64 / n as f64) * k as f64;
        for atom in 0..k {
            logits[[row, atom]] = -(atom as f64 - centre).abs();
        }
    }

    // Target with structure across the P output channels.
    let mut z = Array2::<f64>::zeros((n, p));
    for row in 0..n {
        let t = row as f64 / n as f64;
        for c in 0..p {
            z[[row, c]] = ((c as f64 + 1.0) * (t * 3.0)).sin() + 0.3 * (t - 0.5) * (c as f64 + 1.0);
        }
    }

    let assignment = SaeAssignment::from_blocks_with_mode_and_manifolds(
        logits,
        coord_blocks,
        manifolds,
        AssignmentMode::TopK { k: support_k },
    )
    .expect("assignment");
    let term = SaeManifoldTerm::new(atoms, assignment).expect("term");
    let rho = SaeManifoldRho::new(0.0, (0.01_f64).ln(), vec![Array1::<f64>::zeros(1); k]);
    (term, z, rho)
}

/// The chunked-seed cold start matches the dense full-height seed on every
/// atom's decoder, at several chunk widths — including chunks far smaller than
/// `N`. This is the end-to-end dense/chunked parity the front-door admission of
/// the streaming lane rests on.
#[test]
fn streaming_seed_matches_dense_seed_on_topk_overcomplete() {
    let (n, p, k, support_k) = (64usize, 3usize, 5usize, 2usize);
    assert!(k > p, "the fixture must be overcomplete (K > P)");

    // Dense reference seed.
    let (mut dense_term, z, _rho_unused) = build_topk_term(n, p, k, support_k);
    dense_term
        .seed_cold_start_disjoint_charts(z.view())
        .expect("dense seed");

    for &chunk in &[7usize, 16, 31, 64, 4096] {
        let (mut stream_term, z2, _rho_unused) = build_topk_term(n, p, k, support_k);
        stream_term
            .seed_cold_start_disjoint_charts_streaming(z2.view(), chunk)
            .expect("streaming seed");

        // Every atom's decoder must agree to tolerance (the chunked normal
        // equations reproduce the dense thin-SVD solve; the per-atom residual
        // deflation is identical, so the small solver differences do not
        // accumulate beyond f64 LSQ tolerance).
        let mut max_abs = 0.0_f64;
        for atom in 0..k {
            let a = &dense_term.atoms[atom].decoder_coefficients;
            let b = &stream_term.atoms[atom].decoder_coefficients;
            assert_eq!(
                a.dim(),
                b.dim(),
                "atom {atom} decoder shape differs between dense and chunked seed"
            );
            for (x, y) in a.iter().zip(b.iter()) {
                max_abs = max_abs.max((x - y).abs());
            }
        }
        assert!(
            max_abs <= 1.0e-6,
            "chunked seed (chunk {chunk}) disagrees with dense seed by {max_abs:.3e}"
        );
    }
}

/// The front door ADMITS this overcomplete TopK shape to the curved streaming
/// lane (it is `K > P`, so it would have been demoted to the linear sparse-code
/// lane through the plain `admit_sae_fit`), and the chunked-seed driver names
/// the same sanctioned chunk width the streaming seed consumes.
#[test]
fn front_door_admits_overcomplete_topk_to_curved_streaming() {
    let (n, p, k, support_k, d_max) = (64usize, 3usize, 5usize, 2usize, 1usize);
    // K > P through the plain admission would be the sparse-code lane.
    assert_eq!(
        crate::front_door::admit_sae_fit(n, p, k).unwrap().lane,
        crate::front_door::SaeFitLane::SparseCodes
    );
    // Through the TopK front door it is the CURVED lane instead (a tiny resident
    // seed here, so it is admitted outright — the streaming region is exercised
    // by the front_door ledger tests).
    let admission = crate::front_door::admit_topk_manifold(n, p, k, d_max, support_k)
        .expect("overcomplete TopK admits to the curved lane");
    assert_eq!(
        admission.lane,
        crate::front_door::SaeFitLane::CurvedStreaming
    );

    // The admission ledger's chunk width is the one the streaming seed consumes.
    let lane = crate::manifold::admit_topk_curved_lane(n, p, k, d_max, support_k)
        .expect("curved lane admits");
    assert!(lane.seed_chunk_rows() >= 1);
    assert!(lane.seed_chunk_rows() <= n);
}
