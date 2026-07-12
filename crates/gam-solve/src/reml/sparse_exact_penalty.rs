//! Sparse-exact REML penalty block assembly.
//!
//! This module owns the solver-level representation that pairs canonical
//! penalties with the sparse-exact REML path. Generic sparse factorization and
//! solve routines stay in `linalg::sparse_exact`.

use gam_terms::construction::CanonicalPenalty;
use crate::estimate::EstimationError;

/// Return the count of block-separable canonical penalties eligible for the
/// sparse-exact REML path. `None` means at least two penalties overlap in
/// coefficient range, so the sparse backend must not use block-local assembly.
pub(crate) fn sparse_penalty_block_count_from_canonical(
    penalties: &[CanonicalPenalty],
    p: usize,
) -> Result<Option<usize>, EstimationError> {
    if penalties.is_empty() {
        return Ok(Some(0));
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

    Ok(Some(penalties.len()))
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
    fn canonical_sparse_penalty_block_count_accepts_non_overlapping_ranges() {
        let penalties = vec![
            canonical_penalty(2..4, array![[2.0, 0.5], [0.5, 3.0]], vec![2.0, 3.0], 5),
            canonical_penalty(0..1, array![[7.0]], vec![7.0], 5),
            canonical_penalty(4..5, array![[11.0]], vec![11.0], 5),
        ];

        let block_count = sparse_penalty_block_count_from_canonical(&penalties, 5)
            .unwrap()
            .expect("non-overlapping canonical blocks should be sparse-block compatible");

        assert_eq!(block_count, 3);
    }
}
