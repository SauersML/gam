//! Sparse-exact REML penalty block assembly.
//!
//! This module owns the solver-level representation that pairs canonical
//! penalties with the sparse-exact REML path. Generic sparse factorization and
//! solve routines stay in `linalg::sparse_exact`.

use crate::estimate::EstimationError;
use gam_terms::construction::CanonicalPenalty;

/// Return the count of canonical penalties the sparse-exact REML path carries,
/// after checking that every penalty's coefficient range lies inside `0..p`.
///
/// Penalties may share or overlap coefficient ranges (a smooth's double
/// penalty, `by`-factor smooths): the sparse path assembles `S_λ` from every
/// component, takes `log|S_λ|₊` and its ρ-derivatives from the merged
/// eigenspace blocks of [`super::penalty_logdet::PenaltyPseudologdet`], and
/// reads each `tr(H⁻¹S_k)` from that penalty's own block, so no layout
/// restriction beyond valid ranges applies.
pub(crate) fn sparse_penalty_block_count_from_canonical(
    penalties: &[CanonicalPenalty],
    p: usize,
) -> Result<usize, EstimationError> {
    for (penalty_ordinal, cp) in penalties.iter().enumerate() {
        let (start, end) = (cp.col_range.start, cp.col_range.end);
        if start > end || end > p {
            crate::bail_invalid_estim!(
                "canonical penalty {penalty_ordinal} has invalid column range {start}..{end} for p={p}"
            );
        }
    }
    Ok(penalties.len())
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
            .expect("non-overlapping canonical blocks should be sparse-exact compatible");

        assert_eq!(block_count, 3);
    }

    /// A smooth's double penalty puts two components on one coefficient
    /// range. The sparse-exact path handles shared ranges exactly, so such a
    /// layout must not be refused (it used to route every `s(x) + group(g)`
    /// fit to the dense p×p backend).
    #[test]
    fn canonical_sparse_penalty_block_count_accepts_shared_ranges() {
        let penalties = vec![
            canonical_penalty(0..2, array![[2.0, 0.5], [0.5, 3.0]], vec![2.0, 3.0], 5),
            canonical_penalty(0..2, array![[1.0, 0.0], [0.0, 0.0]], vec![1.0], 5),
            canonical_penalty(2..5, Array2::eye(3), vec![1.0, 1.0, 1.0], 5),
        ];

        let block_count = sparse_penalty_block_count_from_canonical(&penalties, 5)
            .expect("shared canonical ranges should be sparse-exact compatible");

        assert_eq!(block_count, 3);
    }

    #[test]
    fn canonical_sparse_penalty_block_count_rejects_out_of_range_penalty() {
        let penalties = vec![canonical_penalty(3..6, Array2::eye(3), vec![1.0; 3], 6)];
        assert!(sparse_penalty_block_count_from_canonical(&penalties, 5).is_err());
    }
}
