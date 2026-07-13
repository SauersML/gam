//! Canonical dictionary artifacts (#2018).
//!
//! The serialized object is the dictionary orbit representative, not the raw
//! fitted parameters: each atom frame is put in a deterministic finite-gauge
//! convention, scaled to `||B_k||_F = 1`, and hashed from canonical bytes.  The
//! residual finite chart group is recorded explicitly so callers can distinguish
//! byte equality from certified equivalence modulo the remaining group action.

use ndarray::{Array2, ArrayView2};
use sha2::{Digest, Sha256};

use crate::identifiability::{AtomTopology, FittedSaeManifold, residual_gauge};

const HASH_VERSION: &[u8] = b"gam-sae-dictionary-artifact-v1";
const EPS: f64 = 1.0e-12;

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

pub fn canonical_dictionary_artifact(
    model: &FittedSaeManifold,
) -> Result<CanonicalDictionaryArtifact, String> {
    let gauge_certificate = residual_gauge(model)
        .map(|r| r.group_signature().to_string())
        .unwrap_or_else(|e| format!("residual-gauge unavailable: {e}"));
    let mut atoms = Vec::with_capacity(model.atoms.len());
    for atom in &model.atoms {
        let (decoder_block, norm) = canonical_decoder_block(atom.frame.view());
        atoms.push(CanonicalAtomArtifact {
            name: atom.name.clone(),
            topology: atom.topology.clone(),
            decoder_block,
            frobenius_norm_before_gauge: norm,
            residual_finite_gauge: residual_finite_gauge(&atom.topology, atom.chart_canonicalized),
        });
    }
    atoms.sort_by(|a, b| atom_sort_key(a).cmp(&atom_sort_key(b)));
    let content_hash = hash_atoms(&atoms, &gauge_certificate);
    Ok(CanonicalDictionaryArtifact {
        atoms,
        gauge_certificate,
        content_hash,
    })
}

pub fn diff_dictionaries(
    left: &CanonicalDictionaryArtifact,
    right: &CanonicalDictionaryArtifact,
    tol: f64,
) -> DictionaryDiff {
    let pairs = align_dictionaries(left, right);
    let mut atom_diffs = Vec::with_capacity(pairs.len());
    let mut max_decoder_residual = 0.0_f64;
    let mut substantive_differences = 0_usize;
    let mut agreement_sum = 0.0_f64;
    // Track which atoms found a partner so unmatched atoms on either side are
    // reported as atom-level additions/removals rather than silently dropped.
    // Without this, `{circle, circle}` vs `{circle, sphere}` — equal counts, one
    // aligned pair — would report zero substantive differences and a spurious
    // `hash_equal_after_alignment = true`.
    let mut left_matched = vec![false; left.atoms.len()];
    let mut right_matched = vec![false; right.atoms.len()];
    for &(li, ri) in &pairs {
        left_matched[li] = true;
        right_matched[ri] = true;
    }
    for (li, ri) in pairs {
        let l = &left.atoms[li];
        let r = &right.atoms[ri];
        let aligned_r = orient_to_reference(r.decoder_block.view(), l.decoder_block.view());
        let decoder_residual = relative_frobenius_diff(l.decoder_block.view(), aligned_r.view());
        let frame_alignment = frame_alignment(l.decoder_block.view(), aligned_r.view());
        let hash_equal = atom_hash(l) == atom_hash_with_block(r, aligned_r.view());
        let certified_equivalent = decoder_residual <= tol && l.topology == r.topology;
        if !certified_equivalent {
            substantive_differences += 1;
        }
        max_decoder_residual = max_decoder_residual.max(decoder_residual);
        agreement_sum += frame_alignment;
        atom_diffs.push(AtomDiff {
            left_atom: li,
            right_atom: ri,
            frame_alignment,
            decoder_residual,
            hash_equal,
            certified_equivalent,
        });
    }
    let subspace_agreement = if atom_diffs.is_empty() {
        1.0
    } else {
        agreement_sum / atom_diffs.len() as f64
    };
    // Unmatched atoms are genuine structural differences (an atom present on one
    // side with no counterpart on the other). Count each toward the substantive
    // total so `hash_equal_after_alignment` can never claim equivalence when the
    // dictionaries carry different atom sets, even at equal total count.
    let unmatched_left_atoms: Vec<usize> = left_matched
        .iter()
        .enumerate()
        .filter_map(|(i, &m)| (!m).then_some(i))
        .collect();
    let unmatched_right_atoms: Vec<usize> = right_matched
        .iter()
        .enumerate()
        .filter_map(|(i, &m)| (!m).then_some(i))
        .collect();
    substantive_differences += unmatched_left_atoms.len() + unmatched_right_atoms.len();
    DictionaryDiff {
        atom_diffs,
        max_decoder_residual,
        // `substantive_differences == 0` now already implies every atom on both
        // sides was matched and certified-equivalent (any unmatched atom bumped
        // the count above), so equal counts follow and need no separate guard.
        hash_equal_after_alignment: substantive_differences == 0,
        substantive_differences,
        subspace_agreement,
        unmatched_left_atoms,
        unmatched_right_atoms,
    }
}

/// Merge two certified-compatible artifacts by taking all left atoms and only
/// non-equivalent right atoms.  The result is re-hashed from canonical bytes.
pub fn merge_dictionaries(
    left: &CanonicalDictionaryArtifact,
    right: &CanonicalDictionaryArtifact,
    tol: f64,
) -> CanonicalDictionaryArtifact {
    let diff = diff_dictionaries(left, right, tol);
    let mut atoms = left.atoms.clone();
    for (ri, atom) in right.atoms.iter().enumerate() {
        let matched = diff
            .atom_diffs
            .iter()
            .any(|d| d.right_atom == ri && d.certified_equivalent);
        if !matched {
            atoms.push(atom.clone());
        }
    }
    atoms.sort_by(|a, b| atom_sort_key(a).cmp(&atom_sort_key(b)));
    let gauge_certificate = format!(
        "merge({}, {})",
        left.gauge_certificate, right.gauge_certificate
    );
    let content_hash = hash_atoms(&atoms, &gauge_certificate);
    CanonicalDictionaryArtifact {
        atoms,
        gauge_certificate,
        content_hash,
    }
}

fn canonical_decoder_block(frame: ArrayView2<'_, f64>) -> (Array2<f64>, f64) {
    let norm = frame.iter().map(|v| v * v).sum::<f64>().sqrt();
    let scale = if norm > 0.0 && norm.is_finite() {
        1.0 / norm
    } else {
        1.0
    };
    let mut out = frame.to_owned();
    out.mapv_inplace(|v| canonical_zero(v * scale));
    orient_in_place(&mut out);
    (out, norm)
}

fn orient_to_reference(block: ArrayView2<'_, f64>, reference: ArrayView2<'_, f64>) -> Array2<f64> {
    let dot: f64 = block.iter().zip(reference.iter()).map(|(a, b)| a * b).sum();
    let sign = if dot < 0.0 { -1.0 } else { 1.0 };
    block.mapv(|v| canonical_zero(sign * v))
}

fn orient_in_place(block: &mut Array2<f64>) {
    if let Some((_, &v)) = block
        .iter()
        .enumerate()
        .max_by(|(_, a), (_, b)| a.abs().total_cmp(&b.abs()))
    {
        if v < 0.0 {
            block.mapv_inplace(|x| -x);
        }
    }
}

fn canonical_zero(v: f64) -> f64 {
    if v.abs() < EPS { 0.0 } else { v }
}

fn residual_finite_gauge(topology: &AtomTopology, chart_canonicalized: bool) -> String {
    if !chart_canonicalized {
        return "continuous chart gauge not canonicalized".to_string();
    }
    match topology {
        AtomTopology::Circle => "O(2): origin rotation + reflection".to_string(),
        AtomTopology::Torus { .. } => {
            "U(1)^d ⋊ GL(d,Z): origin translations + lattice/reflection symmetries".to_string()
        }
        AtomTopology::EuclideanPatch { .. } => {
            "flat isometry residual: reflection/translation convention-fixed".to_string()
        }
        AtomTopology::Sphere => "O(3): round-sphere isometry residual".to_string(),
        AtomTopology::ProjectivePlane => {
            "PO(3): round-RP2 isometry residual + antipodal deck".to_string()
        }
        AtomTopology::KleinBottle => {
            "U(1)_theta x Z2: axial translation + Klein deck".to_string()
        }
    }
}

fn align_dictionaries(
    left: &CanonicalDictionaryArtifact,
    right: &CanonicalDictionaryArtifact,
) -> Vec<(usize, usize)> {
    let mut used = vec![false; right.atoms.len()];
    let mut pairs = Vec::new();
    for (li, la) in left.atoms.iter().enumerate() {
        let mut best = None;
        for (ri, ra) in right.atoms.iter().enumerate() {
            if used[ri]
                || la.topology != ra.topology
                || la.decoder_block.dim() != ra.decoder_block.dim()
            {
                continue;
            }
            let score = frame_alignment(la.decoder_block.view(), ra.decoder_block.view()).max(
                frame_alignment(
                    la.decoder_block.view(),
                    ra.decoder_block.mapv(|v| -v).view(),
                ),
            );
            if best.is_none_or(|(_, s)| score > s) {
                best = Some((ri, score));
            }
        }
        if let Some((ri, _)) = best {
            used[ri] = true;
            pairs.push((li, ri));
        }
    }
    pairs
}

fn frame_alignment(a: ArrayView2<'_, f64>, b: ArrayView2<'_, f64>) -> f64 {
    let dot = a
        .iter()
        .zip(b.iter())
        .map(|(x, y)| x * y)
        .sum::<f64>()
        .abs();
    let na = a.iter().map(|v| v * v).sum::<f64>().sqrt();
    let nb = b.iter().map(|v| v * v).sum::<f64>().sqrt();
    if na == 0.0 || nb == 0.0 {
        0.0
    } else {
        (dot / (na * nb)).min(1.0)
    }
}

fn relative_frobenius_diff(a: ArrayView2<'_, f64>, b: ArrayView2<'_, f64>) -> f64 {
    let num = a
        .iter()
        .zip(b.iter())
        .map(|(x, y)| (x - y) * (x - y))
        .sum::<f64>()
        .sqrt();
    let den = a.iter().map(|v| v * v).sum::<f64>().sqrt().max(EPS);
    num / den
}

fn hash_atoms(atoms: &[CanonicalAtomArtifact], cert: &str) -> String {
    let mut h = Sha256::new();
    h.update(HASH_VERSION);
    h.update(cert.as_bytes());
    for atom in atoms {
        hash_atom_into(&mut h, atom, atom.decoder_block.view());
    }
    hex(&h.finalize())
}
fn atom_hash(atom: &CanonicalAtomArtifact) -> String {
    atom_hash_with_block(atom, atom.decoder_block.view())
}
fn atom_hash_with_block(atom: &CanonicalAtomArtifact, block: ArrayView2<'_, f64>) -> String {
    let mut h = Sha256::new();
    hash_atom_into(&mut h, atom, block);
    hex(&h.finalize())
}
fn hash_atom_into(h: &mut Sha256, atom: &CanonicalAtomArtifact, block: ArrayView2<'_, f64>) {
    h.update(format!("{:?}|{}|{}|", atom.topology, block.nrows(), block.ncols()).as_bytes());
    for &v in block {
        h.update(canonical_zero(v).to_le_bytes());
    }
    h.update(atom.residual_finite_gauge.as_bytes());
}
fn hex(bytes: &[u8]) -> String {
    bytes.iter().map(|b| format!("{b:02x}")).collect()
}
fn atom_sort_key(atom: &CanonicalAtomArtifact) -> String {
    atom_hash(atom)
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
