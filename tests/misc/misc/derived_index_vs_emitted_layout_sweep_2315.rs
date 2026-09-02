//! Gap 2 of #2315: a GENERALIZED derived-index-vs-emitted-position CROSS-CHECK
//! sweep over a spec zoo.
//!
//! The pre-existing discriminating regressions each pin one site:
//!   * `dropped_candidate_cannot_shift_atomic_active_penalty_identity_2315`
//!     (`basis/bspline_build.rs`) — a single 3-candidate `filter_penalty_candidates`
//!     case, and
//!   * `spatial_penalty_ranges_follow_realized_global_layout_2287`
//!     (`fit_orchestration/drivers/adaptive_bounded_duchon_tests.rs`) — one
//!     spatial-penalty-range case.
//! Both encode the SAME invariant — a derived penalty/coefficient index must
//! equal the position the penalty actually occupies once the realized global
//! layout is emitted — but only at one point each. This harness sweeps that
//! invariant across a zoo, driving two independent fully-public production
//! index-derivation paths:
//!   1. `gam::families::custom_family::realize_coefficient_groups_for_custom_family`
//!      — the composed layout builder that assigns each physical penalty piece an
//!      optimizer (outer) coordinate; the derived outer index must equal the
//!      first-emitted-position anchor of its precision label. This is the reach
//!      of the layout class behind #2287.
//!   2. `gam::terms::basis::filter_penalty_candidates` — the atomic penalty
//!      canonicalizer that partitions candidates into active identities and
//!      dropped diagnostics; every retained `ActivePenalty::info.original_index`
//!      must equal its ORIGINAL input position, so a candidate dropped earlier can
//!      never shift a later active penalty's identity. This is the reach of the
//!      atomic-active-position class the bspline regression pins.

use gam::terms::basis::{
    ConstructiveQuadratic, PenaltyCandidate, PenaltyDropReason, PenaltySource,
    filter_penalty_candidates,
};
use ndarray::Array2;

// ---------------------------------------------------------------------------
// Path 1: composed layout — derived outer index == emitted-position anchor.
// ---------------------------------------------------------------------------

// ---------------------------------------------------------------------------
// Path 2: atomic penalty canonicalization — active/dropped original_index law.
// ---------------------------------------------------------------------------

/// Diagonal input with `diag` entries; a zero diagonal is a rank-0 candidate
/// that must be dropped, a nonzero diagonal is a retained active penalty whose
/// effective rank is its number of nonzero entries.
fn candidate(diag: &[f64], scale: f64, tag: &str) -> (PenaltyCandidate, bool, usize, Array2<f64>) {
    let p = diag.len();
    let mut dense = Array2::<f64>::zeros((p, p));
    for (i, &d) in diag.iter().enumerate() {
        dense[[i, i]] = d;
    }
    let rank = diag.iter().filter(|&&d| d != 0.0).count();
    let is_zero = rank == 0;
    let quad = if is_zero {
        ConstructiveQuadratic::zero(p)
    } else {
        ConstructiveQuadratic::try_from_dense_psd(dense.clone(), tag)
            .expect("diagonal PSD candidate is constructive")
    };
    let cand = PenaltyCandidate {
        matrix: quad,
        source: PenaltySource::Other(tag.to_string()),
        normalization_scale: scale,
        kronecker_factors: None,
        op: None,
    };
    (cand, is_zero, rank, dense)
}

fn assert_matrix_roundoff_equal(actual: &Array2<f64>, expected: &Array2<f64>, context: &str) {
    assert_eq!(actual.dim(), expected.dim(), "{context}: shape mismatch");
    let scale = expected
        .iter()
        .fold(1.0_f64, |current, value| current.max(value.abs()));
    let tolerance = 32.0 * f64::EPSILON * scale;
    let max_error = actual
        .iter()
        .zip(expected.iter())
        .map(|(left, right)| (left - right).abs())
        .fold(0.0_f64, f64::max);
    assert!(
        max_error <= tolerance,
        "{context}: canonical PSD reconstruction changed a penalty beyond roundoff: max error {max_error:e}, tolerance {tolerance:e}"
    );
}

struct FilterCase {
    name: &'static str,
    /// Each entry: (diagonal spectrum, normalization_scale, tag).
    rows: Vec<(Vec<f64>, f64, &'static str)>,
}

fn filter_zoo() -> Vec<FilterCase> {
    vec![
        FilterCase {
            name: "leading_drop",
            rows: vec![
                (vec![0.0, 0.0, 0.0], 11.0, "z0"),
                (vec![4.0, 0.0, 0.0], 13.0, "a1"),
                (vec![0.0, 3.0, 2.0], 17.0, "a2"),
            ],
        },
        FilterCase {
            name: "trailing_drop",
            rows: vec![
                (vec![1.0, 2.0], 5.0, "a0"),
                (vec![0.0, 3.0], 6.0, "a1"),
                (vec![0.0, 0.0], 7.0, "z2"),
            ],
        },
        FilterCase {
            name: "interleaved_drops",
            rows: vec![
                (vec![0.0, 0.0], 1.0, "z0"),
                (vec![2.0, 0.0], 2.0, "a1"),
                (vec![0.0, 0.0], 3.0, "z2"),
                (vec![1.0, 1.0], 4.0, "a3"),
                (vec![0.0, 0.0], 5.0, "z4"),
            ],
        },
        FilterCase {
            name: "no_drops",
            rows: vec![
                (vec![1.0, 0.0, 0.0], 9.0, "a0"),
                (vec![2.0, 3.0, 0.0], 10.0, "a1"),
                (vec![1.0, 1.0, 1.0], 12.0, "a2"),
            ],
        },
        FilterCase {
            name: "all_but_one_dropped",
            rows: vec![
                (vec![0.0, 0.0], 21.0, "z0"),
                (vec![0.0, 0.0], 22.0, "z1"),
                (vec![5.0, 0.0], 23.0, "a2"),
                (vec![0.0, 0.0], 24.0, "z3"),
            ],
        },
    ]
}

#[test]
fn dropped_candidates_never_shift_active_penalty_original_index_sweep_2315() {
    let zoo = filter_zoo();
    assert!(
        zoo.len() >= 5,
        "the filter zoo must sweep several drop patterns"
    );

    for case in &zoo {
        // Build candidates and record the ground-truth per-position expectation.
        let mut candidates = Vec::new();
        let mut expected_active_indices = Vec::new();
        let mut expected_active_ranks = Vec::new();
        let mut expected_active_matrices = Vec::new();
        let mut expected_dropped_indices = Vec::new();
        for (index, (diag, scale, tag)) in case.rows.iter().enumerate() {
            let (cand, is_zero, rank, dense) = candidate(diag, *scale, tag);
            if is_zero {
                expected_dropped_indices.push((index, *scale, *tag));
            } else {
                expected_active_indices.push((index, *scale, *tag));
                expected_active_ranks.push(rank);
                expected_active_matrices.push(dense);
            }
            candidates.push(cand);
        }
        assert!(
            !expected_active_indices.is_empty(),
            "[{}] a meaningful case must retain at least one active penalty",
            case.name
        );

        let filtered = filter_penalty_candidates(candidates)
            .unwrap_or_else(|e| panic!("[{}] canonical filtering must succeed: {e}", case.name));

        // Partition is total and non-overlapping.
        assert_eq!(
            filtered.active.len() + filtered.dropped.len(),
            case.rows.len(),
            "[{}] every candidate must be either active or dropped exactly once",
            case.name
        );

        // Active identities: their ORIGINAL indices are the non-dropped input
        // positions, in order — a dropped-before-active candidate never shifts
        // them. Rank, source, scale, and matrix travel with the same record.
        assert_eq!(
            filtered.active.len(),
            expected_active_indices.len(),
            "[{}] active count mismatch",
            case.name
        );
        for (k, active) in filtered.active.iter().enumerate() {
            let (orig_index, scale, tag) = expected_active_indices[k];
            assert_eq!(
                active.info.original_index, orig_index,
                "[{}] active penalty {k} must keep its original input position {orig_index}",
                case.name
            );
            assert_eq!(
                active.info.effective_rank, expected_active_ranks[k],
                "[{}] active penalty at original index {orig_index} has the wrong effective rank",
                case.name
            );
            assert_eq!(
                active.info.normalization_scale, scale,
                "[{}] active penalty at original index {orig_index} lost its normalization scale",
                case.name
            );
            assert_eq!(
                active.info.source,
                PenaltySource::Other(tag.to_string()),
                "[{}] active penalty at original index {orig_index} lost its source identity",
                case.name
            );
            assert_matrix_roundoff_equal(
                &active.matrix,
                &expected_active_matrices[k],
                &format!(
                    "[{}] active penalty at original index {orig_index}",
                    case.name
                ),
            );
            let p = active.matrix.nrows();
            assert_eq!(
                active.nullity,
                p - expected_active_ranks[k],
                "[{}] active penalty at original index {orig_index} has an inconsistent nullity",
                case.name
            );
        }

        // Dropped diagnostics: their ORIGINAL indices are the rank-0 input
        // positions, typed as ZeroMatrix, and retain their normalization scale.
        assert_eq!(
            filtered.dropped.len(),
            expected_dropped_indices.len(),
            "[{}] dropped count mismatch",
            case.name
        );
        for (k, dropped) in filtered.dropped.iter().enumerate() {
            let (orig_index, scale, tag) = expected_dropped_indices[k];
            assert_eq!(
                dropped.original_index, orig_index,
                "[{}] dropped diagnostic {k} must keep its original input position {orig_index}",
                case.name
            );
            assert_eq!(
                dropped.reason,
                PenaltyDropReason::ZeroMatrix,
                "[{}] dropped diagnostic at original index {orig_index} must be a ZeroMatrix drop",
                case.name
            );
            assert_eq!(
                dropped.normalization_scale, scale,
                "[{}] dropped diagnostic at original index {orig_index} lost its normalization scale",
                case.name
            );
            assert_eq!(
                dropped.source,
                PenaltySource::Other(tag.to_string()),
                "[{}] dropped diagnostic at original index {orig_index} lost its source identity",
                case.name
            );
        }
    }
}
