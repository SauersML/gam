//! Per-point sparse atom codes for multi-manifold reconstruction.
//!
//! This module owns the storage of per-observation soft assignments over a
//! library of `K` candidate manifold-atoms (see [`crate::terms::atom_selection`]
//! for the surrounding selection layer). The two key types are:
//!
//! * [`BitVec`] — a minimal dependency-free bitset used to record the *active
//!   support* `S_n ⊆ {0, …, K−1}` of each observation. We avoid pulling in
//!   the external `bitvec` crate to keep this module aligned with the rest of
//!   `gam`'s "no extra deps for new primitives" policy.
//! * [`SparseAtomCode`] — the per-point pair `(active_mask, weights)` whose
//!   semantics are documented on the type. Reconstruction at point `n` is
//!
//!   ```text
//!   Ẑ_n  =  Σ_{k ∈ S_n}  w_{n,k}  ·  decoder_k(t_{n,k})
//!   ```
//!
//!   so `weights[k]` is meaningful only when `active_mask.get(k) == true`.
//!   We store `weights` densely (`Vec<f64>` of length `K`) rather than
//!   sparsely; for the typical SAE workload `K` is small (tens to low
//!   hundreds), and the dense layout lets us reuse [`ndarray`] views and
//!   simple BLAS-shaped loops downstream. The mask carries the discrete
//!   active-set information; the weights carry the soft amplitudes.
//!
//! ## Per-point block locality (arrow structure)
//!
//! Each [`SparseAtomCode`] is the per-row ψ-block for observation `n`
//! restricted to the `K` atoms. Combined with the per-atom on-manifold
//! coordinate `t_{n,k} ∈ ℝ^{d_k}` (held in
//! [`crate::terms::atom_selection::AtomLibrary`]'s per-atom
//! `LatentCoordValues`), the row-local ψ-vector is
//!
//! ```text
//!   ψ_n  =  ( a_{n,1..K}  ;  t_{n,1,·}  ;  …  ;  t_{n,K,·} )
//! ```
//!
//! whose interaction graph with the shared decoder coefficients `B_1..B_K`
//! is exactly the arrow / bordered-Hessian pattern from `latent_coord.md`
//! §2.2. The Schur complement that Piece 1 uses to eliminate β before the
//! per-row solve generalises here with one change: the row-`n` block now
//! couples to *only the active subset* `S_n` of decoder borders, not to all
//! K of them. That is the structural fact this module records.

use ndarray::{Array1, ArrayView1};

/// Minimal bit-vector. Backing storage is `Vec<u64>` words.
///
/// We expose only the operations the atom-selection layer needs: construction,
/// `get`, `set`, `count_ones`, and iteration of set indices. This is
/// deliberately tiny — adding the external `bitvec` crate would be overkill
/// for a few hundred bits per observation.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct BitVec {
    words: Vec<u64>,
    len: usize,
}

impl BitVec {
    /// All-zero bitset of length `len`.
    pub fn zeros(len: usize) -> Self {
        let words = vec![0u64; (len + 63) / 64];
        Self { words, len }
    }

    /// All-ones bitset of length `len`.
    pub fn ones(len: usize) -> Self {
        let mut bv = Self::zeros(len);
        for i in 0..len {
            bv.set(i, true);
        }
        bv
    }

    pub fn len(&self) -> usize {
        self.len
    }

    pub fn is_empty(&self) -> bool {
        self.len == 0
    }

    #[inline]
    pub fn get(&self, i: usize) -> bool {
        debug_assert!(i < self.len, "BitVec::get index {i} out of bounds {}", self.len);
        let (w, b) = (i / 64, i % 64);
        (self.words[w] >> b) & 1 == 1
    }

    #[inline]
    pub fn set(&mut self, i: usize, v: bool) {
        debug_assert!(i < self.len, "BitVec::set index {i} out of bounds {}", self.len);
        let (w, b) = (i / 64, i % 64);
        if v {
            self.words[w] |= 1u64 << b;
        } else {
            self.words[w] &= !(1u64 << b);
        }
    }

    /// Number of set bits.
    pub fn count_ones(&self) -> usize {
        self.words.iter().map(|w| w.count_ones() as usize).sum()
    }

    /// Iterator over set indices in ascending order.
    pub fn iter_ones(&self) -> impl Iterator<Item = usize> + '_ {
        (0..self.len).filter(move |&i| self.get(i))
    }

    /// Zero all bits in place.
    pub fn clear(&mut self) {
        for w in self.words.iter_mut() {
            *w = 0;
        }
    }
}

/// Per-point sparse code over `K` candidate atoms.
///
/// Invariants (checked in debug builds):
///
/// * `active_mask.len() == weights.len() == K`.
/// * For any `k` with `active_mask.get(k) == false`, the value `weights[k]`
///   is a nuisance — it must not influence reconstruction. Selection
///   strategies that lower a weight to zero (e.g. [`crate::terms::atom_selection::AtomSelectionStrategy`]'s
///   `L1Relaxed` after thresholding) are responsible for clearing the
///   corresponding mask bit *and* zeroing `weights[k]`.
///
/// We do not require `weights[k] >= 0`; some strategies (entropic softmax,
/// TopK projection) keep the simplex, while others (L¹-relaxed) only enforce
/// non-negativity at the active-set step. The owning
/// [`crate::terms::atom_selection::AtomSelectionStrategy`] documents which
/// invariant it maintains.
#[derive(Debug, Clone)]
pub struct SparseAtomCode {
    /// Length-`K` bitmask of active atoms for this point.
    pub active_mask: BitVec,
    /// Length-`K` dense weight vector. Only entries at active indices are
    /// semantically meaningful.
    pub weights: Vec<f64>,
}

impl SparseAtomCode {
    /// Cold-start: no atoms active, all weights zero.
    pub fn empty(k_atoms: usize) -> Self {
        Self {
            active_mask: BitVec::zeros(k_atoms),
            weights: vec![0.0; k_atoms],
        }
    }

    /// Total number of candidate atoms `K` this code is sized for.
    pub fn k_atoms(&self) -> usize {
        self.weights.len()
    }

    /// Cardinality of the active support `|S_n|`.
    pub fn n_active(&self) -> usize {
        self.active_mask.count_ones()
    }

    /// Sum of active weights. For simplex-projected codes this should be ≈ 1.
    pub fn active_weight_sum(&self) -> f64 {
        self.active_mask.iter_ones().map(|k| self.weights[k]).sum()
    }

    /// Set the weight for atom `k` and mark it active.
    pub fn assign(&mut self, k: usize, w: f64) {
        debug_assert!(k < self.k_atoms());
        self.active_mask.set(k, true);
        self.weights[k] = w;
    }

    /// Deactivate atom `k` and zero its stored weight.
    pub fn deactivate(&mut self, k: usize) {
        debug_assert!(k < self.k_atoms());
        self.active_mask.set(k, false);
        self.weights[k] = 0.0;
    }

    /// View the dense weight vector as an `ArrayView1`.
    pub fn weights_view(&self) -> ArrayView1<'_, f64> {
        ArrayView1::from(&self.weights[..])
    }

    /// Materialize the *effective* weight vector (zeros at inactive indices)
    /// as an owned `Array1`. Useful for matmul-shaped downstream code.
    pub fn effective_weights(&self) -> Array1<f64> {
        let mut out = Array1::<f64>::zeros(self.k_atoms());
        for k in self.active_mask.iter_ones() {
            out[k] = self.weights[k];
        }
        out
    }
}

/// Storage for the per-row codes of all `N` observations.
///
/// Held column-of-structs rather than struct-of-columns: each row's
/// `(active_mask, weights)` lives together because the atom-selection
/// strategies all touch a single row at a time. Cross-row vectorization
/// happens through ndarray views built on demand.
#[derive(Debug, Clone)]
pub struct SparseAtomCodes {
    codes: Vec<SparseAtomCode>,
    k_atoms: usize,
}

impl SparseAtomCodes {
    /// Allocate `n_obs` empty codes, each sized for `k_atoms`.
    pub fn empty(n_obs: usize, k_atoms: usize) -> Self {
        let codes = (0..n_obs).map(|_| SparseAtomCode::empty(k_atoms)).collect();
        Self { codes, k_atoms }
    }

    pub fn n_obs(&self) -> usize {
        self.codes.len()
    }

    pub fn k_atoms(&self) -> usize {
        self.k_atoms
    }

    pub fn row(&self, n: usize) -> &SparseAtomCode {
        &self.codes[n]
    }

    pub fn row_mut(&mut self, n: usize) -> &mut SparseAtomCode {
        &mut self.codes[n]
    }

    pub fn iter(&self) -> impl Iterator<Item = &SparseAtomCode> {
        self.codes.iter()
    }

    pub fn iter_mut(&mut self) -> impl Iterator<Item = &mut SparseAtomCode> {
        self.codes.iter_mut()
    }

    /// Flatten weights into a single `(N, K)` array, with zeros where the
    /// mask is unset. Allocates; intended for diagnostic / post-fit use.
    pub fn weights_matrix(&self) -> ndarray::Array2<f64> {
        let n = self.n_obs();
        let k = self.k_atoms();
        let mut out = ndarray::Array2::<f64>::zeros((n, k));
        for n_idx in 0..n {
            let code = &self.codes[n_idx];
            for kk in code.active_mask.iter_ones() {
                out[[n_idx, kk]] = code.weights[kk];
            }
        }
        out
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn bitvec_basic() {
        let mut bv = BitVec::zeros(70);
        assert_eq!(bv.len(), 70);
        assert!(!bv.get(5));
        bv.set(5, true);
        bv.set(64, true);
        assert!(bv.get(5));
        assert!(bv.get(64));
        assert_eq!(bv.count_ones(), 2);
        let ones: Vec<usize> = bv.iter_ones().collect();
        assert_eq!(ones, vec![5, 64]);
        bv.set(5, false);
        assert_eq!(bv.count_ones(), 1);
    }

    #[test]
    fn sparse_code_assign() {
        let mut c = SparseAtomCode::empty(8);
        c.assign(2, 0.7);
        c.assign(5, 0.3);
        assert_eq!(c.n_active(), 2);
        assert!((c.active_weight_sum() - 1.0).abs() < 1e-12);
        c.deactivate(2);
        assert_eq!(c.n_active(), 1);
        assert_eq!(c.weights[2], 0.0);
    }

    #[test]
    fn codes_matrix_roundtrip() {
        let mut codes = SparseAtomCodes::empty(3, 4);
        codes.row_mut(0).assign(1, 0.5);
        codes.row_mut(2).assign(3, 0.9);
        let m = codes.weights_matrix();
        assert_eq!(m[[0, 1]], 0.5);
        assert_eq!(m[[2, 3]], 0.9);
        assert_eq!(m[[1, 0]], 0.0);
    }
}
