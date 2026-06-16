//! Sparse-exact REML penalty block assembly.
//!
//! This module owns the solver-level representation that pairs canonical
//! penalties with the sparse-exact REML path. Generic sparse factorization and
//! solve routines stay in `linalg::sparse_exact`.

use crate::construction::CanonicalPenalty;
use crate::estimate::EstimationError;
use crate::types::PenaltyIdx;
use faer::sparse::{SparseColMat, SymbolicSparseColMat};
use ndarray::Array2;
use rayon::prelude::*;
use std::sync::Arc;

const SPARSE_PENALTY_ZERO_TOL: f64 = 1e-12;
const PARALLEL_SPARSE_FILL_COLUMN_THRESHOLD: usize = 64;

#[derive(Clone)]
pub(crate) struct SparsePenaltyBlock {
    pub(crate) penalty_idx: PenaltyIdx,
    pub(crate) p_start: usize,
    pub(crate) p_end: usize,
    pub(crate) positive_eigenvalues: Arc<Vec<f64>>,
    pub(crate) block_support_strict: bool,
    pub(crate) s_k_sparse: SparseColMat<usize, f64>,
    pub(crate) s_k_block_dense: Arc<Array2<f64>>,
    pub(crate) s_k_block_upper_entries: Arc<Vec<(usize, usize, f64)>>,
}

/// Build sparse penalty blocks from canonical penalties, avoiding redundant
/// block-range scanning and eigendecomposition. Uses the pre-computed col_range
/// and positive_eigenvalues from `CanonicalPenalty`.
pub(crate) fn build_sparse_penalty_blocks_from_canonical(
    penalties: &[CanonicalPenalty],
    p: usize,
) -> Result<Option<Vec<SparsePenaltyBlock>>, EstimationError> {
    if penalties.is_empty() {
        return Ok(Some(Vec::new()));
    }

    // Check for overlapping ranges.
    let mut sorted_ranges: Vec<(usize, usize, usize)> = penalties
        .iter()
        .enumerate()
        .map(|(i, cp)| (i, cp.col_range.start, cp.col_range.end))
        .collect();
    sorted_ranges.sort_by_key(|&(_, start, _)| start);
    for pair in sorted_ranges.windows(2) {
        let (_, _, end_left) = pair[0];
        let (_, start_right, _) = pair[1];
        if end_left > start_right {
            return Ok(None);
        }
    }

    // `par_iter()` over a slice is indexed, and Rayon preserves the input
    // sequence when collecting into a `Vec`, so the final blocks remain in
    // canonical penalty order even though each block is built independently.
    let block_results: Vec<Result<SparsePenaltyBlock, EstimationError>> = penalties
        .par_iter()
        .enumerate()
        .map(|(penalty_ordinal, cp)| {
            let p_start = cp.col_range.start;
            let p_end = cp.col_range.end;
            let s_k_block_dense = cp.local_penalty();
            let s_k_sparse = embed_dense_block_to_sparse_symmetric_upper(
                &s_k_block_dense,
                p_start,
                p,
                SPARSE_PENALTY_ZERO_TOL,
            )?;

            let mut s_k_block_upper_entries = Vec::<(usize, usize, f64)>::new();
            for col in 0..s_k_block_dense.ncols() {
                for row in 0..=col {
                    let value = s_k_block_dense[[row, col]];
                    if value.abs() > SPARSE_PENALTY_ZERO_TOL {
                        s_k_block_upper_entries.push((row, col, value));
                    }
                }
            }

            Ok(SparsePenaltyBlock {
                penalty_idx: PenaltyIdx::new(penalty_ordinal),
                p_start,
                p_end,
                positive_eigenvalues: Arc::new(cp.positive_eigenvalues.clone()),
                block_support_strict: true,
                s_k_sparse,
                s_k_block_dense: Arc::new(s_k_block_dense),
                s_k_block_upper_entries: Arc::new(s_k_block_upper_entries),
            })
        })
        .collect();

    let blocks = block_results.into_iter().collect::<Result<Vec<_>, _>>()?;
    Ok(Some(blocks))
}

fn embed_dense_block_to_sparse_symmetric_upper(
    local: &Array2<f64>,
    offset: usize,
    total_dim: usize,
    tol: f64,
) -> Result<SparseColMat<usize, f64>, EstimationError> {
    let block_n = local.nrows();
    if local.ncols() != block_n {
        crate::bail_invalid_estim!(
            "embed_dense_block_to_sparse_symmetric_upper requires a square block"
        );
    }
    if offset + block_n > total_dim {
        crate::bail_invalid_estim!(
            "embed_dense_block_to_sparse_symmetric_upper offset+block exceeds total_dim"
                .to_string(),
        );
    }

    let counts: Vec<usize> = (0..block_n)
        .into_par_iter()
        .map(|c| {
            let mut count = 0usize;
            for r in 0..=c {
                if local[[r, c]].abs() > tol {
                    count += 1;
                }
            }
            count
        })
        .collect();
    let block_col_ptr = prefix_sum_counts(&counts);
    let mut col_ptr = vec![0usize; total_dim + 1];
    for c in 0..block_n {
        col_ptr[offset + c + 1] = block_col_ptr[c + 1];
    }
    let nnz_in_block_end = block_col_ptr[block_n];
    for c in (offset + block_n)..total_dim {
        col_ptr[c + 1] = nnz_in_block_end;
    }

    let nnz = col_ptr[total_dim];
    let mut row_idx = vec![0usize; nnz];
    let mut values = vec![0.0; nnz];
    fill_embedded_symmetric_upper_columns(
        local,
        offset,
        tol,
        0,
        block_n,
        &block_col_ptr,
        &mut row_idx,
        &mut values,
    );
    let symbolic =
        SymbolicSparseColMat::<usize>::new_checked(total_dim, total_dim, col_ptr, None, row_idx);
    Ok(SparseColMat::<usize, f64>::new(symbolic, values))
}

fn prefix_sum_counts(counts: &[usize]) -> Vec<usize> {
    let mut col_ptr = Vec::with_capacity(counts.len() + 1);
    col_ptr.push(0);
    let mut running = 0usize;
    for &count in counts {
        running += count;
        col_ptr.push(running);
    }
    col_ptr
}

fn fill_embedded_symmetric_upper_columns(
    local: &Array2<f64>,
    offset: usize,
    tol: f64,
    col_start: usize,
    col_end: usize,
    col_ptr: &[usize],
    row_idx: &mut [usize],
    values: &mut [f64],
) {
    if col_end - col_start <= PARALLEL_SPARSE_FILL_COLUMN_THRESHOLD {
        let base = col_ptr[col_start];
        for col in col_start..col_end {
            let mut write = col_ptr[col] - base;
            for row in 0..=col {
                let value = local[[row, col]];
                if value.abs() > tol {
                    row_idx[write] = offset + row;
                    values[write] = value;
                    write += 1;
                }
            }
        }
        return;
    }

    let mid = col_start + (col_end - col_start) / 2;
    let split = col_ptr[mid] - col_ptr[col_start];
    let (left_rows, right_rows) = row_idx.split_at_mut(split);
    let (left_values, right_values) = values.split_at_mut(split);
    rayon::join(
        || {
            fill_embedded_symmetric_upper_columns(
                local,
                offset,
                tol,
                col_start,
                mid,
                col_ptr,
                left_rows,
                left_values,
            );
        },
        || {
            fill_embedded_symmetric_upper_columns(
                local,
                offset,
                tol,
                mid,
                col_end,
                col_ptr,
                right_rows,
                right_values,
            );
        },
    );
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::{Array1, array};

    fn canonical_penalty(
        col_range: std::ops::Range<usize>,
        local: Array2<f64>,
        positive_eigenvalues: Vec<f64>,
        total_dim: usize,
    ) -> CanonicalPenalty {
        let block_dim = col_range.len();
        CanonicalPenalty {
            root: Array2::<f64>::zeros((0, block_dim)),
            col_range,
            total_dim,
            nullity: 0,
            local,
            prior_mean: Array1::zeros(block_dim),
            positive_eigenvalues,
            op: None,
        }
    }

    #[test]
    fn canonical_sparse_penalty_blocks_preserve_input_order() {
        let penalties = vec![
            canonical_penalty(2..4, array![[2.0, 0.5], [0.5, 3.0]], vec![2.0, 3.0], 5),
            canonical_penalty(0..1, array![[7.0]], vec![7.0], 5),
            canonical_penalty(4..5, array![[11.0]], vec![11.0], 5),
        ];

        let blocks = build_sparse_penalty_blocks_from_canonical(&penalties, 5)
            .unwrap()
            .expect("non-overlapping canonical blocks should be sparse-block compatible");

        let observed: Vec<(usize, usize, usize)> = blocks
            .iter()
            .map(|block| (block.penalty_idx.get(), block.p_start, block.p_end))
            .collect();
        assert_eq!(observed, vec![(0, 2, 4), (1, 0, 1), (2, 4, 5)]);
        assert_eq!(&*blocks[0].positive_eigenvalues, &vec![2.0, 3.0]);
        assert_eq!(&*blocks[1].positive_eigenvalues, &vec![7.0]);
        assert_eq!(&*blocks[2].positive_eigenvalues, &vec![11.0]);
    }
}
