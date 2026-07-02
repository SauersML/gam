//! #1801 (Bug A) — streaming `SaeManifoldTerm::materialize_chunk` must be able
//! to re-evaluate `Φ(t)` at each chunk's coordinates even for atoms built from
//! precomputed design matrices (`SaeManifoldAtom::new`, which leaves
//! `basis_evaluator = None`). Split into its own module to keep `tests.rs`
//! under the 10k-line ban gate (#780). Uses the crate-level manifold items via
//! `super::*`.

use super::*;
use ndarray::array;

/// Issue #1801 (Bug A): a streaming `materialize_chunk` must succeed for an
/// `EuclideanPatch` atom built from a precomputed design matrix via
/// `SaeManifoldAtom::new` (which leaves `basis_evaluator = None`). Before the
/// fix `materialize_chunk` hard-required a carried evaluator and returned
/// `Err("... has no basis evaluator ...")`; after the fix it synthesizes the
/// deterministic monomial-patch evaluator from the atom geometry and
/// re-evaluates `Φ(t)` at the chunk coordinates.
#[test]
fn materialize_chunk_synthesizes_euclidean_patch_evaluator() -> Result<(), String> {
    // Degree-2 patch in latent_dim = 1 => Φ columns {1, t, t²}, basis_size 3.
    let evaluator = EuclideanPatchEvaluator::new(1, 2)?;
    let train_coords = array![[-1.0_f64], [-0.25], [0.4], [0.9], [1.5]];
    let (phi, jet) = evaluator.evaluate(train_coords.view())?;
    let decoder = array![[0.2, -0.5], [1.1, 0.35], [-0.15, 0.6]];
    // Built via the general constructor WITHOUT `with_basis_evaluator`, so the
    // atom carries no evaluator — exactly the streaming-unfriendly shape.
    let atom = SaeManifoldAtom::new(
        "euclidean_patch",
        SaeAtomBasisKind::EuclideanPatch,
        1,
        phi,
        jet,
        decoder,
        Array2::<f64>::eye(evaluator.basis_size()),
    )?;
    assert!(
        atom.basis_evaluator.is_none(),
        "precondition: precomputed atom must carry no basis evaluator"
    );
    let assignment = SaeAssignment::from_blocks_with_mode_and_manifolds(
        Array2::<f64>::zeros((train_coords.nrows(), 1)),
        vec![train_coords],
        vec![LatentManifold::Euclidean],
        AssignmentMode::softmax(1.0),
    )?;
    let term = SaeManifoldTerm::new(vec![atom], assignment)?;

    // A small chunk with a different row count than the training term.
    let chunk_coords = array![[-0.6_f64], [0.1], [0.75]];
    let n_chunk = chunk_coords.nrows();
    let chunk_logits = Array2::<f64>::zeros((n_chunk, 1));
    let chunk_term = term.materialize_chunk(chunk_logits, vec![chunk_coords.clone()])?;

    assert_eq!(chunk_term.atoms.len(), 1);
    let chunk_atom = &chunk_term.atoms[0];
    // Φ was re-evaluated at the chunk coordinates: (n_chunk, basis_size).
    assert_eq!(chunk_atom.n_obs(), n_chunk);
    assert_eq!(chunk_atom.basis_size(), 3);
    // The synthesized evaluator is carried onto the chunk atom so downstream
    // streaming assembly re-evaluates Φ(t) exactly like the non-precomputed path.
    assert!(
        chunk_atom.basis_evaluator.is_some(),
        "chunk atom must carry the synthesized monomial-patch evaluator"
    );
    // The re-evaluated Φ must equal a direct evaluation of the monomial design.
    let (expected_phi, _) = evaluator.evaluate(chunk_coords.view())?;
    let max_abs = chunk_atom
        .basis_values
        .iter()
        .zip(expected_phi.iter())
        .fold(0.0_f64, |acc, (&a, &b)| acc.max((a - b).abs()));
    assert!(
        max_abs <= 1.0e-12,
        "synthesized Φ disagrees with direct monomial evaluation by {max_abs:.3e}"
    );
    Ok(())
}
