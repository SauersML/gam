//! Hard-scenario coverage for arbitrary cross-term coefficient groups with
//! hierarchical nesting.
//!
//! These tests focus on the runtime quadratic
//!
//!     P_g(β) = β' S_p β / 2
//!
//! for the `PenaltySpec::Dense(S_p)` produced by
//! `design.realize_coefficient_groups`.
//!
//! We exercise penalty correctness directly against the realized `S_p`
//! rather than running a full REML fit so that the assertions are
//! bitwise-deterministic and immune to optimizer tolerance drift.

use gam::estimate::PenaltySpec;
use gam::smooth::{
    CoefficientGroupSpec, CoefficientSelector, LinearTermSpec, RandomEffectTermSpec,
    TermCollectionSpec, build_term_collection_design,
};
use gam::types::{CoefficientGroupPrior, RhoPrior};
use ndarray::{Array1, Array2};

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

/// A small synthetic data matrix with two linear-feature columns (col 0, col 1)
/// and one categorical column (col 2) suitable for a random-effect term.
fn mixed_design_data(n: usize) -> Array2<f64> {
    let mut x = Array2::<f64>::zeros((n, 3));
    for i in 0..n {
        x[[i, 0]] = (i as f64 - (n as f64 - 1.0) / 2.0) / (n as f64);
        x[[i, 1]] = if i % 2 == 0 { -0.4 } else { 0.6 };
        // Categorical feature: 4 levels for the random-effect term.
        x[[i, 2]] = (i % 4) as f64;
    }
    x
}

fn mixed_term_spec() -> TermCollectionSpec {
    TermCollectionSpec {
        linear_terms: vec![
            LinearTermSpec {
                name: "lin_a".to_string(),
                feature_col: 0,
                feature_cols: vec![0],
                categorical_levels: vec![],
                double_penalty: false,
                coefficient_geometry: Default::default(),
                coefficient_min: None,
                coefficient_max: None,
                frozen_function_mass: None,
            },
            LinearTermSpec {
                name: "lin_b".to_string(),
                feature_col: 1,
                feature_cols: vec![1],
                categorical_levels: vec![],
                double_penalty: false,
                coefficient_geometry: Default::default(),
                coefficient_min: None,
                coefficient_max: None,
                frozen_function_mass: None,
            },
        ],
        random_effect_terms: vec![RandomEffectTermSpec {
            name: "re_g".to_string(),
            feature_col: 2,
            frozen_levels: None,
            lenient_unseen: true,
        }],
        smooth_terms: Vec::new(),
        level: Default::default(),
    }
}

/// Closed-form evaluation of `β' S β / 2` for a single penalty.
fn penalty_quadratic(matrix: &Array2<f64>, beta: &Array1<f64>) -> f64 {
    0.5 * beta.dot(&matrix.dot(beta))
}

// ---------------------------------------------------------------------------
// 1. Cross-term group spans a linear term + a random-effect term.
// ---------------------------------------------------------------------------

#[test]
fn cross_term_group_linear_plus_random_effect_penalty_quadratic_matches_analytic() {
    let x = mixed_design_data(16);
    let spec = mixed_term_spec();
    let design = build_term_collection_design(x.view(), &spec).expect("design");
    let p = design.design.ncols();

    let realized = design
        .realize_coefficient_groups(
            &[CoefficientGroupSpec {
                name: "lin_a_plus_re_g".to_string(),
                selectors: vec![
                    CoefficientSelector::LinearTerm("lin_a".to_string()),
                    CoefficientSelector::RandomEffectTerm("re_g".to_string()),
                ],
                parent: None,
                prior: Some(CoefficientGroupPrior::GammaPrecision {
                    shape: 2.0,
                    rate: 1.0,
                }),
            }],
            &RhoPrior::Flat,
        )
        .expect("cross-term group realizes");

    // The realized group penalty must be the LAST entry in penalty_specs;
    // it must be a Dense penalty with identity on the active columns.
    let group_penalty = realized
        .penalty_specs
        .last()
        .expect("group penalty present");
    let matrix = match group_penalty {
        PenaltySpec::Dense(matrix) => matrix.clone(),
        other => panic!("expected Dense, got {other:?}"),
    };

    // The active columns are the union of lin_a's columns and the re_g block.
    let active_cols: Vec<usize> = realized
        .group_column_indices
        .iter()
        .find(|(name, _)| name == "lin_a_plus_re_g")
        .map(|(_, cols)| cols.clone())
        .expect("group columns recorded");
    assert!(
        active_cols.len() >= 2,
        "must span linear + at least one RE coef"
    );

    // Build a deterministic β with all entries nonzero.
    let mut beta = Array1::<f64>::zeros(p);
    for (i, b) in beta.iter_mut().enumerate() {
        *b = 0.1 + 0.07 * (i as f64);
    }

    // Analytic quadratic: sum of β_j^2 for active columns.
    let analytic = 0.5 * active_cols.iter().map(|&j| beta[j] * beta[j]).sum::<f64>();
    let matrix_quad = penalty_quadratic(&matrix, &beta);
    assert!(
        (matrix_quad - analytic).abs() < 1e-12,
        "S_p quadratic mismatch: matrix={matrix_quad} analytic={analytic}"
    );
}

// ---------------------------------------------------------------------------
// 2. Two-level nested hierarchy: outer = inner_a ∪ inner_b.
// ---------------------------------------------------------------------------

#[test]
fn nested_two_level_hierarchy_penalty_contributions_compose_additively() {
    let x = mixed_design_data(16);
    let spec = mixed_term_spec();
    let design = build_term_collection_design(x.view(), &spec).expect("design");
    let p = design.design.ncols();

    let realized = design
        .realize_coefficient_groups(
            &[
                CoefficientGroupSpec {
                    name: "outer".to_string(),
                    selectors: vec![
                        CoefficientSelector::LinearTerm("lin_a".to_string()),
                        CoefficientSelector::LinearTerm("lin_b".to_string()),
                    ],
                    parent: None,
                    prior: None,
                },
                CoefficientGroupSpec {
                    name: "inner_a".to_string(),
                    selectors: vec![CoefficientSelector::LinearTerm("lin_a".to_string())],
                    parent: Some("outer".to_string()),
                    prior: None,
                },
                CoefficientGroupSpec {
                    name: "inner_b".to_string(),
                    selectors: vec![CoefficientSelector::LinearTerm("lin_b".to_string())],
                    parent: Some("outer".to_string()),
                    prior: None,
                },
            ],
            &RhoPrior::Flat,
        )
        .expect("nested hierarchy realizes");

    // The three group penalty_specs are appended after the base penalties.
    let base_count = realized.penalty_specs.len() - 3;
    let outer = &realized.penalty_specs[base_count];
    let inner_a = &realized.penalty_specs[base_count + 1];
    let inner_b = &realized.penalty_specs[base_count + 2];

    let unwrap_matrix = |spec: &PenaltySpec| -> Array2<f64> {
        match spec {
            PenaltySpec::Dense(m) => m.clone(),
            other => panic!("unexpected spec variant: {other:?}"),
        }
    };
    let outer_m = unwrap_matrix(outer);
    let inner_a_m = unwrap_matrix(inner_a);
    let inner_b_m = unwrap_matrix(inner_b);

    // The outer dense S equals the sum of the two inner dense Ss because the
    // outer materializes the concatenated child penalty (see comment block
    // around src/terms/smooth.rs:1576).
    let sum_inner = &inner_a_m + &inner_b_m;
    for ((i, j), &v) in sum_inner.indexed_iter() {
        assert!(
            (v - outer_m[[i, j]]).abs() < 1e-12,
            "outer != inner_a + inner_b at ({i},{j}): outer={} sum={}",
            outer_m[[i, j]],
            v
        );
    }

    // Pick a deterministic β.
    let mut beta = Array1::<f64>::zeros(p);
    for (i, b) in beta.iter_mut().enumerate() {
        *b = -0.3 + 0.11 * (i as f64);
    }
    let q_outer = penalty_quadratic(&outer_m, &beta);
    let q_inner_a = penalty_quadratic(&inner_a_m, &beta);
    let q_inner_b = penalty_quadratic(&inner_b_m, &beta);
    assert!(
        (q_outer - (q_inner_a + q_inner_b)).abs() < 1e-12,
        "additive composition broken: outer={q_outer} inner_a+inner_b={}",
        q_inner_a + q_inner_b
    );
}

// ---------------------------------------------------------------------------
// 3. Unknown-label rejection.
// ---------------------------------------------------------------------------

#[test]
fn unknown_label_in_selector_is_rejected_with_label_in_error_message() {
    let x = mixed_design_data(8);
    let spec = mixed_term_spec();
    let design = build_term_collection_design(x.view(), &spec).expect("design");

    let bad_label = "no_such_term_zzz";
    let err = design
        .realize_coefficient_groups(
            &[CoefficientGroupSpec {
                name: "typo_group".to_string(),
                selectors: vec![CoefficientSelector::LinearTerm(bad_label.to_string())],
                parent: None,
                prior: None,
            }],
            &RhoPrior::Flat,
        )
        .expect_err("unknown label must be rejected");
    let msg = err.to_string();
    assert!(
        msg.contains(bad_label),
        "error must mention bad label '{bad_label}', got: {msg}"
    );
}

// ---------------------------------------------------------------------------
// 4. Empty-group rejection (no selectors).
// ---------------------------------------------------------------------------

#[test]
fn empty_coefficient_group_is_rejected() {
    let x = mixed_design_data(8);
    let spec = mixed_term_spec();
    let design = build_term_collection_design(x.view(), &spec).expect("design");

    let err = design
        .realize_coefficient_groups(
            &[CoefficientGroupSpec {
                name: "empty".to_string(),
                selectors: vec![],
                parent: None,
                prior: None,
            }],
            &RhoPrior::Flat,
        )
        .expect_err("empty group must be rejected");
    let msg = err.to_string();
    assert!(
        msg.contains("empty") || msg.contains("no selectors") || msg.contains("no coefficients"),
        "error must indicate emptiness, got: {msg}"
    );
}

// ---------------------------------------------------------------------------
// 5. Disjointness: non-nested overlap is documented to be ALLOWED.
//
// See `overlapping_coefficient_groups_are_distinct_precision_coordinates` in
// tests/coefficient_groups.rs — overlap without parent/child relationship is
// permitted and produces two independent precision coordinates.  Verify the
// penalty contribution is computed twice (once per coordinate).
// ---------------------------------------------------------------------------

#[test]
fn overlapping_non_nested_groups_apply_penalty_twice() {
    let x = mixed_design_data(8);
    let spec = mixed_term_spec();
    let design = build_term_collection_design(x.view(), &spec).expect("design");
    let p = design.design.ncols();

    let realized = design
        .realize_coefficient_groups(
            &[
                CoefficientGroupSpec {
                    name: "g1".to_string(),
                    selectors: vec![CoefficientSelector::LinearTerm("lin_a".to_string())],
                    parent: None,
                    prior: Some(CoefficientGroupPrior::GammaPrecision {
                        shape: 2.0,
                        rate: 1.0,
                    }),
                },
                CoefficientGroupSpec {
                    name: "g2".to_string(),
                    selectors: vec![CoefficientSelector::LinearTerm("lin_a".to_string())],
                    parent: None,
                    prior: Some(CoefficientGroupPrior::GammaPrecision {
                        shape: 5.0,
                        rate: 1.0,
                    }),
                },
            ],
            &RhoPrior::Flat,
        )
        .expect("overlapping non-nested groups realize");

    // Two coordinates appended after base.
    let base_count = realized.penalty_specs.len() - 2;
    let s1 = match &realized.penalty_specs[base_count] {
        PenaltySpec::Dense(matrix) => matrix.clone(),
        other => panic!("{other:?}"),
    };
    let s2 = match &realized.penalty_specs[base_count + 1] {
        PenaltySpec::Dense(matrix) => matrix.clone(),
        other => panic!("{other:?}"),
    };

    let mut beta = Array1::<f64>::zeros(p);
    for (i, b) in beta.iter_mut().enumerate() {
        *b = 0.2 + 0.05 * (i as f64);
    }
    let q1 = penalty_quadratic(&s1, &beta);
    let q2 = penalty_quadratic(&s2, &beta);
    assert!(q1 > 0.0 && q2 > 0.0, "both penalties must be active");
    // The matrices must be EQUAL (same selector → same identity-on-cols S).
    for ((i, j), &v) in s1.indexed_iter() {
        assert!(
            (v - s2[[i, j]]).abs() < 1e-12,
            "overlapping groups should share the same S at ({i},{j}): s1={v} s2={}",
            s2[[i, j]]
        );
    }
    // Each coordinate gets its own λ in REML; the total prior energy at fixed
    // β with λ_1=λ_2=1 would be q1+q2 = 2*q1 — i.e. computed twice.
    assert!(
        (q1 + q2 - 2.0 * q1).abs() < 1e-12,
        "overlap must double-count when both λ=1"
    );
}

// ---------------------------------------------------------------------------
// 6. Hierarchy spanning K=3 components (uses three linear terms in a wider spec).
// ---------------------------------------------------------------------------

fn three_linear_spec() -> (TermCollectionSpec, Array2<f64>) {
    let n = 24;
    let mut x = Array2::<f64>::zeros((n, 3));
    for i in 0..n {
        x[[i, 0]] = (i as f64) / (n as f64);
        x[[i, 1]] = ((i % 5) as f64) - 2.0;
        x[[i, 2]] = if i % 3 == 0 { 0.7 } else { -0.4 };
    }
    let spec = TermCollectionSpec {
        linear_terms: vec![
            LinearTermSpec {
                name: "z1".to_string(),
                feature_col: 0,
                feature_cols: vec![0],
                categorical_levels: vec![],
                double_penalty: false,
                coefficient_geometry: Default::default(),
                coefficient_min: None,
                coefficient_max: None,
                frozen_function_mass: None,
            },
            LinearTermSpec {
                name: "z2".to_string(),
                feature_col: 1,
                feature_cols: vec![1],
                categorical_levels: vec![],
                double_penalty: false,
                coefficient_geometry: Default::default(),
                coefficient_min: None,
                coefficient_max: None,
                frozen_function_mass: None,
            },
            LinearTermSpec {
                name: "z3".to_string(),
                feature_col: 2,
                feature_cols: vec![2],
                categorical_levels: vec![],
                double_penalty: false,
                coefficient_geometry: Default::default(),
                coefficient_min: None,
                coefficient_max: None,
                frozen_function_mass: None,
            },
        ],
        random_effect_terms: Vec::new(),
        smooth_terms: Vec::new(),
        level: Default::default(),
    };
    (spec, x)
}

#[test]
fn three_way_hierarchy_per_group_penalty_decomposes_correctly() {
    let (spec, x) = three_linear_spec();
    let design = build_term_collection_design(x.view(), &spec).expect("design");
    let p = design.design.ncols();

    let realized = design
        .realize_coefficient_groups(
            &[
                CoefficientGroupSpec {
                    name: "outer_all".to_string(),
                    selectors: vec![
                        CoefficientSelector::LinearTerm("z1".to_string()),
                        CoefficientSelector::LinearTerm("z2".to_string()),
                        CoefficientSelector::LinearTerm("z3".to_string()),
                    ],
                    parent: None,
                    prior: None,
                },
                CoefficientGroupSpec {
                    name: "group_A".to_string(),
                    selectors: vec![CoefficientSelector::LinearTerm("z1".to_string())],
                    parent: Some("outer_all".to_string()),
                    prior: None,
                },
                CoefficientGroupSpec {
                    name: "group_B".to_string(),
                    selectors: vec![CoefficientSelector::LinearTerm("z2".to_string())],
                    parent: Some("outer_all".to_string()),
                    prior: None,
                },
                CoefficientGroupSpec {
                    name: "group_C".to_string(),
                    selectors: vec![CoefficientSelector::LinearTerm("z3".to_string())],
                    parent: Some("outer_all".to_string()),
                    prior: None,
                },
            ],
            &RhoPrior::Flat,
        )
        .expect("3-way hierarchy realizes");

    // Last 4 specs correspond to outer_all, group_A, group_B, group_C (in spec order).
    let base = realized.penalty_specs.len() - 4;
    let outer_m = match &realized.penalty_specs[base] {
        PenaltySpec::Dense(matrix) => matrix.clone(),
        other => panic!("{other:?}"),
    };
    let leaf_ms: Vec<Array2<f64>> = (1..=3)
        .map(|k| match &realized.penalty_specs[base + k] {
            PenaltySpec::Dense(matrix) => matrix.clone(),
            other => panic!("{other:?}"),
        })
        .collect();

    // outer_all matrix equals the sum of the three leaf matrices.
    let mut summed = Array2::<f64>::zeros((p, p));
    for m in &leaf_ms {
        summed = &summed + m;
    }
    for ((i, j), &v) in summed.indexed_iter() {
        assert!(
            (v - outer_m[[i, j]]).abs() < 1e-12,
            "outer != sum_of_leaves at ({i},{j}): outer={} sum={}",
            outer_m[[i, j]],
            v
        );
    }
}

// ---------------------------------------------------------------------------
// 7. Permutation invariance of the selector index set.
// ---------------------------------------------------------------------------

#[test]
fn selector_permutation_yields_identical_penalty_quadratic() {
    let x = mixed_design_data(12);
    let spec = mixed_term_spec();
    let design = build_term_collection_design(x.view(), &spec).expect("design");
    let p = design.design.ncols();

    let make_spec = |selectors: Vec<CoefficientSelector>| CoefficientGroupSpec {
        name: "perm_group".to_string(),
        selectors,
        parent: None,
        prior: Some(CoefficientGroupPrior::GammaPrecision {
            shape: 2.0,
            rate: 1.0,
        }),
    };

    let r_ab = design
        .realize_coefficient_groups(
            &[make_spec(vec![
                CoefficientSelector::LinearTerm("lin_a".to_string()),
                CoefficientSelector::LinearTerm("lin_b".to_string()),
            ])],
            &RhoPrior::Flat,
        )
        .expect("ab order");
    let r_ba = design
        .realize_coefficient_groups(
            &[make_spec(vec![
                CoefficientSelector::LinearTerm("lin_b".to_string()),
                CoefficientSelector::LinearTerm("lin_a".to_string()),
            ])],
            &RhoPrior::Flat,
        )
        .expect("ba order");

    let s_ab = match r_ab.penalty_specs.last().unwrap() {
        PenaltySpec::Dense(matrix) => matrix.clone(),
        other => panic!("{other:?}"),
    };
    let s_ba = match r_ba.penalty_specs.last().unwrap() {
        PenaltySpec::Dense(matrix) => matrix.clone(),
        other => panic!("{other:?}"),
    };

    // The two penalty matrices must be bitwise identical.
    for ((i, j), &v) in s_ab.indexed_iter() {
        assert_eq!(
            v.to_bits(),
            s_ba[[i, j]].to_bits(),
            "permutation should not alter S at ({i},{j})"
        );
    }

    // And the quadratic on any β must agree exactly.
    let mut beta = Array1::<f64>::zeros(p);
    for (i, b) in beta.iter_mut().enumerate() {
        *b = -0.13 + 0.21 * (i as f64);
    }
    let q_ab = penalty_quadratic(&s_ab, &beta);
    let q_ba = penalty_quadratic(&s_ba, &beta);
    assert_eq!(
        q_ab.to_bits(),
        q_ba.to_bits(),
        "quadratic must be bitwise-invariant under selector permutation"
    );
}
