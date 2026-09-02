//! Canonical dictionary artifacts (#2018).
//!
//! The serialized object is the dictionary orbit representative, not the raw
//! fitted parameters: each atom frame is put in a deterministic finite-gauge
//! convention, scaled to `||B_k||_F = 1`, and hashed from canonical bytes.  The
//! residual finite chart group is recorded explicitly so callers can distinguish
//! byte equality from certified equivalence modulo the remaining group action.

use ndarray::Array2;

use crate::identifiability::AtomTopology;

#[derive(Debug, Clone, PartialEq)]
pub struct CanonicalAtomArtifact {
    pub name: String,
    pub topology: AtomTopology,
    pub decoder_block: Array2<f64>,
    pub frobenius_norm_before_gauge: f64,
    pub residual_finite_gauge: String,
}

#[derive(Debug, Clone, PartialEq)]
pub struct CanonicalDictionaryArtifact {
    pub atoms: Vec<CanonicalAtomArtifact>,
    pub gauge_certificate: String,
    pub content_hash: String,
}

#[derive(Debug, Clone, PartialEq)]
pub struct AtomDiff {
    pub left_atom: usize,
    pub right_atom: usize,
    pub frame_alignment: f64,
    pub decoder_residual: f64,
    pub hash_equal: bool,
    pub certified_equivalent: bool,
}

#[derive(Debug, Clone, PartialEq)]
pub struct DictionaryDiff {
    pub atom_diffs: Vec<AtomDiff>,
    pub max_decoder_residual: f64,
    pub substantive_differences: usize,
    pub hash_equal_after_alignment: bool,
    pub subspace_agreement: f64,
    /// Left-side atom indices with no counterpart in `right` (no atom of the
    /// same topology/shape was left to pair with). Each such atom is an
    /// atom-level removal and counts toward [`Self::substantive_differences`].
    pub unmatched_left_atoms: Vec<usize>,
    /// Right-side atom indices with no counterpart in `left` — atom-level
    /// additions, also counted in [`Self::substantive_differences`].
    pub unmatched_right_atoms: Vec<usize>,
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::identifiability::{FittedAtom, FittedSaeManifold};
    use gam_problem::RowMetric;
    use ndarray::array;

    fn model(frames: Vec<Array2<f64>>) -> FittedSaeManifold {
        let atoms = frames
            .into_iter()
            .enumerate()
            .map(|(i, frame)| FittedAtom {
                name: format!("a{i}"),
                topology: AtomTopology::Circle,
                frame,
                ard_variances: None,
                lowering_error: 0.0,
                chart_canonicalized: true,
                inner_fit: None,
            })
            .collect();
        FittedSaeManifold {
            atoms,
            jacobian_rows: Vec::new(),
            isometry_penalty_root: Array2::zeros((0, 0)),
            metric: RowMetric::euclidean(0, 0).unwrap(),
        }
    }

    #[test]
    fn canonical_hash_ignores_atom_order_scale_and_reflection() {
        let a = model(vec![array![[2.0], [0.0]], array![[0.0], [3.0]]]);
        let b = model(vec![array![[0.0], [-9.0]], array![[-4.0], [0.0]]]);
        let ca = canonical_dictionary_artifact(&a).unwrap();
        let cb = canonical_dictionary_artifact(&b).unwrap();
        let d = diff_dictionaries(&ca, &cb, 1e-12);
        assert_eq!(d.substantive_differences, 0);
        assert!(d.hash_equal_after_alignment);
    }

    #[test]
    fn diff_flags_unmatched_atoms_at_equal_count() {
        // Both dictionaries carry two atoms, and one atom pair is byte-identical,
        // but the second atoms cannot align (different decoder-block shape). The
        // aligned-equality claim must be FALSE and the leftover atoms must be
        // reported as an atom-level removal + addition — not silently dropped
        // into a spurious "no differences" verdict at equal total count.
        let left =
            canonical_dictionary_artifact(&model(vec![array![[1.0], [0.0]], array![[0.0], [1.0]]]))
                .unwrap();
        let right = canonical_dictionary_artifact(&model(vec![
            array![[1.0], [0.0]],
            array![[1.0], [0.0], [0.0]],
        ]))
        .unwrap();
        let d = diff_dictionaries(&left, &right, 1e-12);
        assert!(
            !d.hash_equal_after_alignment,
            "dictionaries with different atom sets must not claim aligned equality"
        );
        assert_eq!(
            d.unmatched_left_atoms.len(),
            1,
            "left[1] has no counterpart"
        );
        assert_eq!(
            d.unmatched_right_atoms.len(),
            1,
            "the 3-row right atom has no counterpart"
        );
        assert!(
            d.substantive_differences >= 2,
            "each unmatched atom is a substantive difference, got {}",
            d.substantive_differences
        );
    }

    #[test]
    fn diff_localizes_decoder_row_perturbation() {
        let a =
            canonical_dictionary_artifact(&model(vec![array![[1.0], [0.0]], array![[0.0], [1.0]]]))
                .unwrap();
        let b =
            canonical_dictionary_artifact(&model(vec![array![[1.0], [0.2]], array![[0.0], [1.0]]]))
                .unwrap();
        let d = diff_dictionaries(&a, &b, 1e-6);
        assert_eq!(d.substantive_differences, 1);
        assert!(d.max_decoder_residual > 0.05);
    }
}
