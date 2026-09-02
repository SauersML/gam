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

