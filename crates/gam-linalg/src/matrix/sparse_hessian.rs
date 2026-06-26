//! Sparse Hessian accumulator: a banded upper-triangle CSC assembler for
//! `Xᵀ diag(w) X` on local-support (B-spline / banded) designs.
//!
//! The symbolic sparsity pattern depends only on the design `X` (not on the
//! weights), so it is built once and shared across parallel workers via `Arc`.
//! Each worker owns only a cheap flat values buffer and accumulates with
//! `O(nnz)` memory instead of the dense `O(p²)` representation.

use faer::sparse::{SparseColMat, SparseRowMat};
use std::sync::Arc;

/// Immutable symbolic sparsity pattern for a banded upper-triangle CSC Hessian.
///
/// Shared via `Arc` across all parallel workers so that only the mutable values
/// buffer needs to be cloned per worker.
struct SparseHessianSymbolic {
    dim: usize,
    nnz: usize,
    /// CSC column pointers, length `dim + 1`.
    col_ptrs: Vec<usize>,
    /// CSC row indices (upper triangle: row ≤ col), length `nnz`.
    row_indices: Vec<usize>,
    /// First row index in each column.  For banded patterns the rows within a
    /// column are contiguous, so `offset = col_ptrs[c] + (r - first_row[c])`.
    /// Columns with zero entries store `usize::MAX`.
    first_row: Vec<usize>,
    /// Whether every column has strictly contiguous row indices (true for
    /// B-spline bases).  When true, `add` is O(1) arithmetic instead of a
    /// linear scan.
    contiguous: bool,
}

impl SparseHessianSymbolic {
    fn build(csrs: &[&SparseRowMat<usize, f64>], dim: usize) -> Self {
        use std::collections::BTreeSet;

        let n = csrs[0].nrows();
        let mut rows_by_col = vec![BTreeSet::<usize>::new(); dim];

        let mut cols = Vec::with_capacity(32);
        for i in 0..n {
            cols.clear();
            for csr in csrs {
                let sym = csr.symbolic();
                let rp = sym.row_ptr();
                let ci = sym.col_idx();
                for p in rp[i]..rp[i + 1] {
                    cols.push(ci[p]);
                }
            }
            cols.sort_unstable();
            cols.dedup();
            for (ai, &ca) in cols.iter().enumerate() {
                assert!(
                    ca < dim,
                    "SparseHessianSymbolic::build: column index {ca} out of Hessian dimension {dim}"
                );
                for &cb in &cols[ai..] {
                    assert!(
                        cb < dim,
                        "SparseHessianSymbolic::build: column index {cb} out of Hessian dimension {dim}"
                    );
                    rows_by_col[cb].insert(ca);
                }
            }
        }

        // Convert column buckets to CSC.
        let nnz = rows_by_col.iter().map(BTreeSet::len).sum();
        let mut col_ptrs = Vec::with_capacity(dim + 1);
        let mut row_indices = Vec::with_capacity(nnz);
        col_ptrs.push(0);
        for rows in rows_by_col {
            row_indices.extend(rows);
            col_ptrs.push(row_indices.len());
        }

        // Detect contiguity and record first_row per column.
        let mut first_row = vec![usize::MAX; dim];
        let mut contiguous = true;
        for c in 0..dim {
            let start = col_ptrs[c];
            let end = col_ptrs[c + 1];
            if start == end {
                continue;
            }
            first_row[c] = row_indices[start];
            // Check rows are first_row, first_row+1, ..., first_row+(end-start-1).
            for (off, &ri) in row_indices[start..end].iter().enumerate() {
                if ri != first_row[c] + off {
                    contiguous = false;
                    break;
                }
            }
            if !contiguous {
                break;
            }
        }

        SparseHessianSymbolic {
            dim,
            nnz,
            col_ptrs,
            row_indices,
            first_row,
            contiguous,
        }
    }
}

/// Pre-computed upper-triangle CSC sparsity pattern + flat values buffer for
/// assembling `X^T diag(w) X` directly in sparse form.
///
/// Designed for B-spline / local-support bases where the Hessian is banded and
/// the dense `p×p` representation wastes most of its storage.  The pattern is
/// computed once from the design matrices; each parallel worker then owns a
/// cheap values buffer (the symbolic structure is `Arc`-shared) and accumulates
/// with `O(nnz)` memory instead of `O(p²)`.
pub struct SparseHessianAccumulator {
    sym: Arc<SparseHessianSymbolic>,
    /// Values buffer, length `sym.nnz`. Crate-visible for reductions; callers
    /// must not resize it because unchecked accumulation relies on this invariant.
    pub values: Vec<f64>,
}

// Manual Clone: only the values buffer is duplicated; the symbolic pattern is
// Arc-shared.
impl Clone for SparseHessianAccumulator {
    fn clone(&self) -> Self {
        SparseHessianAccumulator {
            sym: Arc::clone(&self.sym),
            values: self.values.clone(),
        }
    }
}

impl SparseHessianAccumulator {
    // ── pattern builders ─────────────────────────────────────────────

    /// Build the symbolic upper-triangle pattern of `X^T X` from a single
    /// sparse CSR design matrix.
    pub fn from_single_csr(csr: &SparseRowMat<usize, f64>, dim: usize) -> Self {
        Self::from_multi_csr(&[csr], dim)
    }

    /// Build the symbolic upper-triangle pattern of the block Hessian produced
    /// by multiple sparse CSR designs that share the same column space.
    pub fn from_multi_csr(csrs: &[&SparseRowMat<usize, f64>], dim: usize) -> Self {
        let sym = Arc::new(SparseHessianSymbolic::build(csrs, dim));
        let nnz = sym.nnz;
        SparseHessianAccumulator {
            sym,
            values: vec![0.0; nnz],
        }
    }

    // ── accumulation ─────────────────────────────────────────────────

    /// Add `val` to the upper-triangle entry `(r, c)`.
    ///
    /// **Caller must ensure `r <= c`.**  This method does NOT canonicalize —
    /// it is the caller's responsibility to only emit upper-triangle pairs.
    /// This avoids the double-counting bug that arises when both `(ca, cb)`
    /// and `(cb, ca)` are mapped to the same upper-triangle slot.
    #[inline(always)]
    pub fn add_upper(&mut self, r: usize, c: usize, val: f64) {
        assert!(r <= c, "add_upper requires r <= c, got ({r}, {c})");
        let s = &*self.sym;
        if s.contiguous {
            // O(1) direct-index path: rows within each column are contiguous
            // integers starting at first_row[c], so the offset is arithmetic.
            let start = s.col_ptrs[c];
            let end = s.col_ptrs[c + 1];
            let offset = r.wrapping_sub(s.first_row[c]);
            assert!(
                r >= s.first_row[c] && offset < end - start,
                "add_upper contiguous OOB"
            );
            let idx = start + offset;
            // SAFETY: the assert! immediately above proves idx is in
            // start..end, and SparseHessianAccumulator preserves values.len()
            // == s.nnz with end <= s.nnz.
            unsafe {
                *self.values.get_unchecked_mut(idx) += val;
            }
        } else {
            // Fallback linear scan for non-contiguous patterns.
            let start = s.col_ptrs[c];
            let end = s.col_ptrs[c + 1];
            let slice = &s.row_indices[start..end];
            for (off, &ri) in slice.iter().enumerate() {
                if ri == r {
                    // SAFETY: off comes from row_indices[start..end], so start+off < end.
                    // The values.len() == s.nnz invariant and end <= s.nnz make the slot valid.
                    unsafe {
                        *self.values.get_unchecked_mut(start + off) += val;
                    }
                    return;
                }
            }
            assert!(
                false,
                "SparseHessianAccumulator::add_upper: ({r}, {c}) not in pattern"
            );
        }
    }

    /// Element-wise `self.values += other`.
    #[inline]
    pub fn add_values(&mut self, other: &[f64]) {
        assert_eq!(self.values.len(), other.len());
        for (a, &b) in self.values.iter_mut().zip(other.iter()) {
            *a += b;
        }
    }

    /// Create a zero-valued copy sharing the same symbolic structure.
    pub fn empty_clone(&self) -> Self {
        SparseHessianAccumulator {
            sym: Arc::clone(&self.sym),
            values: vec![0.0; self.values.len()],
        }
    }

    // ── finalization ─────────────────────────────────────────────────

    /// Consume the accumulator and return a `SparseColMat` (upper-triangle).
    ///
    /// Constructs the CSC matrix directly from the symbolic structure — no
    /// triplet roundtrip.  If this is the last reference to the symbolic
    /// pattern, it is moved rather than cloned.
    pub fn into_sparse_col_mat(self) -> SparseColMat<usize, f64> {
        use faer::sparse::SymbolicSparseColMat;

        // Try to take ownership of the symbolic data (avoids clone when the
        // parallel reduction has already discarded all other Arcs).
        let (col_ptrs, row_indices, dim) = match Arc::try_unwrap(self.sym) {
            Ok(owned) => (owned.col_ptrs, owned.row_indices, owned.dim),
            Err(shared) => (
                shared.col_ptrs.clone(),
                shared.row_indices.clone(),
                shared.dim,
            ),
        };
        let symbolic = {
            // col_ptrs is dim+1 monotone 0..row_indices.len(); ca/cb assert row<dim.
            // BTreeSet iter ⇒ each column sorted + duplicate-free.
            // SAFETY: invariants enforced by the assertions above.
            unsafe { SymbolicSparseColMat::new_unchecked(dim, dim, col_ptrs, None, row_indices) }
        };
        SparseColMat::new(symbolic, self.values)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use faer::sparse::Triplet;

    #[test]
    fn sparse_hessian_pattern_is_column_major_csc() {
        let sparse = SparseColMat::try_new_from_triplets(
            1,
            3,
            &[
                Triplet::new(0, 0, 1.0),
                Triplet::new(0, 1, 1.0),
                Triplet::new(0, 2, 1.0),
            ],
        )
        .expect("sparse column matrix");
        let csr = sparse.to_row_major().expect("csr conversion");
        let accumulator = SparseHessianAccumulator::from_single_csr(&csr, 3);

        assert_eq!(accumulator.sym.col_ptrs, vec![0, 1, 3, 6]);
        assert_eq!(accumulator.sym.row_indices, vec![0, 0, 1, 0, 1, 2]);
    }
}
