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

use std::collections::BTreeMap;

use gam::families::custom_family::{
    CoefficientGroupSpec, ParameterBlockSpec, PenaltyMatrix,
    coefficient_label, realize_coefficient_groups_for_custom_family,
};
use gam::linalg::matrix::{DenseDesignMatrix, DesignMatrix};
use gam::terms::basis::{
    ConstructiveQuadratic, PenaltyCandidate, PenaltyDropReason, PenaltySource,
    filter_penalty_candidates,
};
use gam_problem::RhoPrior;
use gam_spec::CoefficientGroupPrior;
use ndarray::{Array1, Array2};

// ---------------------------------------------------------------------------
// Path 1: composed layout — derived outer index == emitted-position anchor.
// ---------------------------------------------------------------------------

fn diag_penalty(diag: &[f64]) -> PenaltyMatrix {
    let p = diag.len();
    let mut m = Array2::<f64>::zeros((p, p));
    for (i, &d) in diag.iter().enumerate() {
        m[[i, i]] = d;
    }
    PenaltyMatrix::Dense(m)
}

fn plain(p: usize) -> PenaltyMatrix {
    diag_penalty(&vec![1.0; p])
}

fn labeled(p: usize, label: &str) -> PenaltyMatrix {
    plain(p).with_precision_label(label)
}

fn fixed(p: usize, log_lambda: f64) -> PenaltyMatrix {
    plain(p).with_fixed_log_lambda(log_lambda)
}

fn block(name: &str, p: usize, penalties: Vec<PenaltyMatrix>, base_lambda: f64) -> ParameterBlockSpec {
    let k = penalties.len();
    let initial_log_lambdas = Array1::from_iter((0..k).map(|j| base_lambda + j as f64 * 0.25));
    ParameterBlockSpec {
        name: name.to_string(),
        design: DesignMatrix::Dense(DenseDesignMatrix::from(Array2::<f64>::eye(p))),
        offset: Array1::zeros(p),
        penalties,
        nullspace_dims: Vec::new(),
        initial_log_lambdas,
        ..ParameterBlockSpec::defaults()
    }
}

fn with_initial_log_lambdas(
    mut spec: ParameterBlockSpec,
    values: &[f64],
) -> ParameterBlockSpec {
    assert_eq!(spec.penalties.len(), values.len());
    spec.initial_log_lambdas = Array1::from_vec(values.to_vec());
    spec
}

fn flat_group(label: &str, coords: Vec<(&str, usize)>, mean: f64) -> CoefficientGroupSpec {
    let mut g = CoefficientGroupSpec::new(
        label,
        coords.into_iter().map(|(b, c)| coefficient_label(b, c)).collect(),
    )
    .with_prior(CoefficientGroupPrior::NormalLogPrecision { mean, sd: 2.0 });
    g.initial_log_precision = Some(mean / 10.0);
    g
}

struct LayoutCase {
    name: &'static str,
    specs: Vec<ParameterBlockSpec>,
    groups: Vec<CoefficientGroupSpec>,
    base_prior: RhoPrior,
}

/// A broad zoo of composed specs mixing plain/tied/cross-block-tied/fixed base
/// penalties with and without coefficient-group layers.
fn layout_zoo() -> Vec<LayoutCase> {
    let scalar = || RhoPrior::Normal { mean: 0.0, sd: 2.0 };
    vec![
        LayoutCase {
            name: "single_block_two_plain",
            specs: vec![block("m", 2, vec![plain(2), plain(2)], -1.0)],
            groups: vec![],
            base_prior: scalar(),
        },
        LayoutCase {
            name: "two_blocks_no_groups",
            specs: vec![
                block("m", 3, vec![plain(3), plain(3)], -1.0),
                block("s", 3, vec![plain(3)], 0.5),
            ],
            groups: vec![],
            base_prior: scalar(),
        },
        LayoutCase {
            name: "within_block_tie",
            specs: vec![with_initial_log_lambdas(
                block(
                    "m",
                    2,
                    vec![labeled(2, "t"), labeled(2, "t"), plain(2)],
                    -1.0,
                ),
                &[-1.0, -1.0, -0.5],
            )],
            groups: vec![],
            base_prior: scalar(),
        },
        LayoutCase {
            name: "cross_block_tie",
            specs: vec![
                block("a", 2, vec![plain(2), labeled(2, "shared")], -1.0),
                with_initial_log_lambdas(
                    block("b", 2, vec![labeled(2, "shared"), plain(2)], 0.0),
                    &[-0.75, 0.25],
                ),
            ],
            groups: vec![],
            base_prior: scalar(),
        },
        LayoutCase {
            name: "fixed_penalty_between_optimized",
            specs: vec![block(
                "m",
                2,
                vec![plain(2), fixed(2, 1.0), plain(2)],
                -1.0,
            )],
            groups: vec![],
            base_prior: scalar(),
        },
        LayoutCase {
            name: "tie_and_fixed_mixed",
            specs: vec![
                block("a", 2, vec![labeled(2, "t"), fixed(2, 2.0)], -1.0),
                with_initial_log_lambdas(
                    block("b", 2, vec![labeled(2, "t"), plain(2)], 0.0),
                    &[-1.0, 0.25],
                ),
            ],
            groups: vec![],
            base_prior: scalar(),
        },
        LayoutCase {
            name: "single_flat_group",
            specs: vec![
                block("m", 3, vec![plain(3), plain(3)], -1.0),
                block("s", 3, vec![plain(3)], 0.0),
            ],
            groups: vec![flat_group("g0", vec![("m", 0)], 30.0)],
            base_prior: scalar(),
        },
        LayoutCase {
            name: "two_flat_groups_one_cross_block",
            specs: vec![
                block("m", 3, vec![plain(3), labeled(3, "b")], -1.0),
                block("s", 3, vec![plain(3), plain(3)], 0.0),
            ],
            groups: vec![
                flat_group("g0", vec![("m", 0)], 30.0),
                flat_group("g1", vec![("m", 1), ("s", 0)], 40.0),
            ],
            base_prior: scalar(),
        },
        LayoutCase {
            name: "nested_pair_with_fixed_base",
            specs: vec![
                block("m", 3, vec![plain(3), fixed(3, 1.0)], -1.0),
                block("s", 3, vec![plain(3)], 0.0),
            ],
            // A nested (depth-2) pair: `child` -> `parent`. The interior node's
            // coordinates must equal the union of its children, so both select
            // exactly (m, 0).
            groups: vec![
                {
                    let mut child = flat_group("child", vec![("m", 0)], 10.0);
                    child.parent = Some("parent".to_string());
                    child
                },
                flat_group("parent", vec![("m", 0)], 20.0),
            ],
            base_prior: scalar(),
        },
    ]
}

/// For a realized layout, independently compute the expected physical→outer
/// coordinate map (`None` for fixed pieces; otherwise the first-occurrence index
/// of the precision label) and the outer labels, and assert both agree with the
/// production result — i.e. the DERIVED index equals the EMITTED position.
#[test]
fn derived_penalty_outer_index_equals_emitted_position_across_spec_zoo_2315() {
    let zoo = layout_zoo();
    assert!(zoo.len() >= 8, "the layout zoo must be broad");

    for case in &zoo {
        let realized = realize_coefficient_groups_for_custom_family(
            &case.specs,
            &case.groups,
            case.base_prior.clone(),
        )
        .unwrap_or_else(|e| panic!("[{}] spec must realize: {e}", case.name));

        // Independent derivation over the flattened emission walk.
        let mut derived_physical_to_outer = Vec::<Option<usize>>::new();
        let mut derived_outer_labels = Vec::<String>::new();
        let mut derived_penalty_labels = Vec::<String>::new();
        let mut first_index_of = BTreeMap::<String, usize>::new();
        for (b, spec) in realized.specs.iter().enumerate() {
            for (j, penalty) in spec.penalties.iter().enumerate() {
                let label = penalty
                    .precision_label()
                    .map(str::to_owned)
                    .unwrap_or_else(|| format!("__block_{b}_penalty_{j}"));
                derived_penalty_labels.push(label.clone());
                if penalty.fixed_log_lambda().is_some() {
                    derived_physical_to_outer.push(None);
                    continue;
                }
                let outer = *first_index_of.entry(label.clone()).or_insert_with(|| {
                    let idx = derived_outer_labels.len();
                    derived_outer_labels.push(label.clone());
                    idx
                });
                derived_physical_to_outer.push(Some(outer));
            }
        }

        // The realized emitted penalty labels and outer labels must match the
        // independent derivation exactly.
        assert_eq!(
            realized.penalty_labels, derived_penalty_labels,
            "[{}] emitted penalty labels disagree with derived emission order",
            case.name
        );
        assert_eq!(
            realized.outer_labels, derived_outer_labels,
            "[{}] realized outer coordinates disagree with derived first-occurrence indices",
            case.name
        );

        // Emitted-position anchoring: outer coordinate i is owned by the FIRST
        // emitted non-fixed physical piece that maps to i.
        let mut anchored = vec![false; derived_outer_labels.len()];
        for (physical, outer) in derived_physical_to_outer.iter().enumerate() {
            if let Some(i) = outer {
                if !anchored[*i] {
                    anchored[*i] = true;
                    assert_eq!(
                        derived_penalty_labels[physical], realized.outer_labels[*i],
                        "[{}] outer coordinate {i} is not anchored by its first emitted piece",
                        case.name
                    );
                }
            }
        }
        assert!(
            anchored.iter().all(|&a| a),
            "[{}] every optimizer coordinate must be anchored by an emitted penalty",
            case.name
        );

        // One prior per derived coordinate.
        let RhoPrior::Independent(priors) = &realized.rho_prior else {
            panic!("[{}] realized prior must be coordinate-wise", case.name);
        };
        assert_eq!(
            priors.len(),
            derived_outer_labels.len(),
            "[{}] prior count must equal the number of derived optimizer coordinates",
            case.name
        );
    }
}

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

fn assert_matrix_roundoff_equal(
    actual: &Array2<f64>,
    expected: &Array2<f64>,
    context: &str,
) {
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
    assert!(zoo.len() >= 5, "the filter zoo must sweep several drop patterns");

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
