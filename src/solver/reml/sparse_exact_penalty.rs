//! Sparse-exact REML penalty block assembly.
//!
//! This module owns the solver-level representation that pairs canonical
//! penalties with the sparse-exact REML path. Generic sparse factorization and
//! solve routines stay in `linalg::sparse_exact`.

use crate::construction::CanonicalPenalty;
use crate::estimate::EstimationError;
use crate::types::PenaltyIdx;
use rayon::prelude::*;
use std::sync::Arc;

#[derive(Clone)]
pub(crate) struct SparsePenaltyBlock {
    pub(crate) penalty_idx: PenaltyIdx,
    pub(crate) positive_eigenvalues: Arc<Vec<f64>>,
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
    for &(penalty_ordinal, start, end) in &sorted_ranges {
        if start > end || end > p {
            crate::bail_invalid_estim!(
                "canonical penalty {penalty_ordinal} has invalid column range {start}..{end} for p={p}"
            );
        }
    }
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
            Ok(SparsePenaltyBlock {
                penalty_idx: PenaltyIdx::new(penalty_ordinal),
                positive_eigenvalues: Arc::new(cp.positive_eigenvalues.clone()),
            })
        })
        .collect();

    let blocks = block_results.into_iter().collect::<Result<Vec<_>, _>>()?;
    Ok(Some(blocks))
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::{Array1, Array2, array};

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

        let observed: Vec<usize> = blocks.iter().map(|block| block.penalty_idx.get()).collect();
        assert_eq!(observed, vec![0, 1, 2]);
        assert_eq!(&*blocks[0].positive_eigenvalues, &vec![2.0, 3.0]);
        assert_eq!(&*blocks[1].positive_eigenvalues, &vec![7.0]);
        assert_eq!(&*blocks[2].positive_eigenvalues, &vec![11.0]);
    }
}
