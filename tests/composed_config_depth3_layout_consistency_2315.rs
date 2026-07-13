//! Gap 1 of #2315: a single GENERALIZED "2-of-everything, depth-3"
//! composed-configuration standing harness.
//!
//! The pre-existing per-site regressions (`composed_score_link_influence_rho_
//! layout_advances_by_emitted_counts_2315` in `bms/block_specs.rs`,
//! `multi_block_group_priors_follow_realized_penalty_order_2315` in
//! `coefficient_groups.rs`, and the penalty-label regressions in
//! `penalty_labels.rs`) each pin ONE composition site. None of them build a
//! model that stacks two of every composable ingredient at once and asserts the
//! whole realized layout stays self-consistent. That "silent-wrong-answer under
//! composition" class is exactly what shipped as #2287–#2292: a penalty piece
//! that received the right-looking slice but the wrong optimizer coordinate.
//!
//! This harness drives the fully-public production composition entry point
//!     gam::families::custom_family::realize_coefficient_groups_for_custom_family
//! which is the layout builder the fit itself calls. Internally it fans out to
//! the exact helpers the per-site regressions cover:
//!   * `validate_blockspecs` / `validate_blockspec_consistency` (block topology),
//!   * `penalty_label_layout_with_joint` (physical→outer coordinate law),
//!   * `resolved_physical_penalty_label` (the `__block_i_penalty_j` convention).
//! Coefficient groups are the public composable "influence / score-link layer"
//! analogue: each is a separate independent Gaussian prior factor layered onto
//! the base smooth prior, exactly like the BMS score_warp / link_dev / influence
//! absorbers whose `push_deviation_aux_blockspecs` sibling is `pub(crate)`.
//!
//! Every zoo entry is a full 2-of-everything, depth-3 configuration:
//!   * >= 2 parameter blocks (smooth terms),
//!   * >= 2 blocks carrying >= 2 base penalties,
//!   * >= 2 coefficient groups (composable prior-factor layers),
//!   * a coefficient-group hierarchy of nesting depth >= 3.
//! For each we re-derive the emitted penalty labels, the optimizer-coordinate
//! (outer) labels, and the per-coordinate priors independently from the realized
//! block specs and assert they match production coordinate-for-coordinate.

use std::collections::BTreeMap;

use gam::families::custom_family::{
    CoefficientGroupSpec, ParameterBlockSpec, PenaltyMatrix,
    coefficient_label, realize_coefficient_groups_for_custom_family,
};
use gam::linalg::matrix::{DenseDesignMatrix, DesignMatrix};
use gam_problem::RhoPrior;
use gam_spec::CoefficientGroupPrior;
use ndarray::{Array1, Array2};

/// A diagonal PSD penalty with a caller-chosen spectrum (all entries >= 0), so
/// its rank and symmetry are known and `validate_blockspec_consistency` accepts
/// it. `None` in `diag` means a zero on that coordinate (still PSD).
fn diag_penalty(diag: &[f64]) -> PenaltyMatrix {
    let p = diag.len();
    let mut m = Array2::<f64>::zeros((p, p));
    for (i, &d) in diag.iter().enumerate() {
        m[[i, i]] = d;
    }
    PenaltyMatrix::Dense(m)
}

/// A plain (unlabeled, optimized) full-rank penalty.
fn plain(p: usize) -> PenaltyMatrix {
    diag_penalty(&vec![1.0; p])
}

/// A penalty tied to a shared precision `label`; pieces carrying the same label
/// collapse onto ONE optimizer coordinate (the first physical occurrence owns
/// it). Cross-block ties are the sharpest test of the first-occurrence law.
fn labeled(p: usize, label: &str) -> PenaltyMatrix {
    plain(p).with_precision_label(label)
}

/// A penalty with a frozen log-lambda; it consumes NO optimizer coordinate.
fn fixed(p: usize, log_lambda: f64) -> PenaltyMatrix {
    plain(p).with_fixed_log_lambda(log_lambda)
}

/// Build a `p`-column block whose penalties carry distinct sentinel initial
/// log-lambdas (so any coordinate mis-indexing shifts an observable value).
fn block(name: &str, p: usize, penalties: Vec<PenaltyMatrix>, base_lambda: f64) -> ParameterBlockSpec {
    let k = penalties.len();
    let initial_log_lambdas =
        Array1::from_iter((0..k).map(|j| base_lambda + j as f64 * 0.5));
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

struct GroupSeed {
    label: &'static str,
    coords: Vec<(&'static str, usize)>,
    parent: Option<&'static str>,
    mean: f64,
    sd: f64,
    init_log_precision: f64,
}

fn group(seed: &GroupSeed) -> CoefficientGroupSpec {
    let mut g = CoefficientGroupSpec::new(
        seed.label,
        seed.coords
            .iter()
            .map(|(b, c)| coefficient_label(*b, *c))
            .collect(),
    )
    .with_prior(CoefficientGroupPrior::NormalLogPrecision {
        mean: seed.mean,
        sd: seed.sd,
    });
    if let Some(parent) = seed.parent {
        g = g.with_parent(parent);
    }
    g.initial_log_precision = Some(seed.init_log_precision);
    g
}

struct ComposedSpec {
    name: &'static str,
    specs: Vec<ParameterBlockSpec>,
    groups: Vec<GroupSeed>,
    base_prior: RhoPrior,
}

/// The zoo: four distinct 2-of-everything, depth-3 composed configurations.
fn composed_zoo() -> Vec<ComposedSpec> {
    vec![
        // A: two blocks, 2 penalties each; a branching depth-3 hierarchy
        // (grandparent -> parent -> {leaf_a, leaf_b}) plus a shared-label base
        // penalty, all on the mean block.
        ComposedSpec {
            name: "branching_depth3_single_block_groups",
            specs: vec![
                block("mean", 3, vec![plain(3), labeled(3, "shared")], -1.0),
                block("scale", 3, vec![plain(3), plain(3)], 2.0),
            ],
            groups: vec![
                GroupSeed { label: "leaf_a", coords: vec![("mean", 0)], parent: Some("parent_g"), mean: 10.0, sd: 1.0, init_log_precision: 4.0 },
                GroupSeed { label: "leaf_b", coords: vec![("mean", 1)], parent: Some("parent_g"), mean: 20.0, sd: 2.0, init_log_precision: 5.0 },
                GroupSeed { label: "parent_g", coords: vec![("mean", 0), ("mean", 1)], parent: Some("grand_g"), mean: 30.0, sd: 3.0, init_log_precision: 6.0 },
                GroupSeed { label: "grand_g", coords: vec![("mean", 0), ("mean", 1)], parent: None, mean: 40.0, sd: 4.0, init_log_precision: 7.0 },
            ],
            base_prior: RhoPrior::Normal { mean: 0.0, sd: 2.0 },
        },
        // B: cross-block tie ("tied" appears in both blocks) + a fixed base
        // penalty (no optimizer coordinate) + a linear depth-3 chain.
        ComposedSpec {
            name: "cross_block_tie_fixed_penalty_linear_depth3",
            specs: vec![
                block("loc", 2, vec![plain(2), labeled(2, "tied")], -2.0),
                block("disp", 2, vec![labeled(2, "tied"), fixed(2, 1.5)], 0.5),
            ],
            groups: vec![
                GroupSeed { label: "child_g", coords: vec![("loc", 0)], parent: Some("parent_g"), mean: 11.0, sd: 1.0, init_log_precision: 3.0 },
                GroupSeed { label: "parent_g", coords: vec![("loc", 0)], parent: Some("grand_g"), mean: 22.0, sd: 2.0, init_log_precision: 4.0 },
                GroupSeed { label: "grand_g", coords: vec![("loc", 0)], parent: None, mean: 33.0, sd: 3.0, init_log_precision: 5.0 },
            ],
            base_prior: RhoPrior::Normal { mean: -1.0, sd: 1.5 },
        },
        // C: two independent hierarchies (a branching depth-3 on block "a" and a
        // depth-2 chain on block "b"), 3 base penalties per block including a
        // labeled base penalty.
        ComposedSpec {
            name: "two_independent_hierarchies_wide_blocks",
            specs: vec![
                block("a", 4, vec![plain(4), plain(4), labeled(4, "ta")], -1.0),
                block("b", 4, vec![plain(4), plain(4), plain(4)], 1.0),
            ],
            groups: vec![
                GroupSeed { label: "la", coords: vec![("a", 0)], parent: Some("pa"), mean: 10.0, sd: 1.0, init_log_precision: 2.0 },
                GroupSeed { label: "lb", coords: vec![("a", 1)], parent: Some("pa"), mean: 12.0, sd: 1.5, init_log_precision: 2.5 },
                GroupSeed { label: "pa", coords: vec![("a", 0), ("a", 1)], parent: Some("ga"), mean: 14.0, sd: 2.0, init_log_precision: 3.0 },
                GroupSeed { label: "ga", coords: vec![("a", 0), ("a", 1)], parent: None, mean: 16.0, sd: 2.5, init_log_precision: 3.5 },
                GroupSeed { label: "lc", coords: vec![("b", 0)], parent: Some("pb"), mean: 20.0, sd: 3.0, init_log_precision: 4.0 },
                GroupSeed { label: "pb", coords: vec![("b", 0)], parent: None, mean: 22.0, sd: 3.5, init_log_precision: 4.5 },
            ],
            base_prior: RhoPrior::Normal { mean: 0.5, sd: 2.0 },
        },
        // D: a cross-block-spanning group hierarchy (a group whose coordinates
        // straddle two blocks emits one piece per block per component) with an
        // explicit coordinate-wise Independent base prior.
        ComposedSpec {
            name: "cross_block_spanning_group_independent_base_prior",
            specs: vec![
                block("first", 2, vec![plain(2), plain(2)], -1.0),
                block("second", 2, vec![plain(2), plain(2)], 0.0),
            ],
            groups: vec![
                GroupSeed { label: "span_leaf", coords: vec![("first", 0), ("second", 0)], parent: Some("span_parent"), mean: 30.0, sd: 2.0, init_log_precision: 6.0 },
                GroupSeed { label: "span_parent", coords: vec![("first", 0), ("second", 0)], parent: Some("span_grand"), mean: 31.0, sd: 2.0, init_log_precision: 6.5 },
                GroupSeed { label: "span_grand", coords: vec![("first", 0), ("second", 0)], parent: None, mean: 32.0, sd: 2.0, init_log_precision: 7.0 },
            ],
            // Four base optimizer coordinates: first{p0,p1}, second{p0,p1}.
            base_prior: RhoPrior::Independent(vec![
                RhoPrior::Normal { mean: 1.0, sd: 1.0 },
                RhoPrior::Normal { mean: 2.0, sd: 1.0 },
                RhoPrior::Normal { mean: 3.0, sd: 1.0 },
                RhoPrior::Normal { mean: 4.0, sd: 1.0 },
            ]),
        },
    ]
}

/// Independently reproduce the production physical→outer emission walk (the body
/// of `penalty_label_layout_with_joint` / the label loop in
/// `realize_coefficient_groups_for_custom_family`) directly from the realized
/// block specs, so a mismatch means the realized layout disagrees with the
/// coordinate law it claims to follow.
///
/// Returns (physical penalty labels in flattened emission order, outer labels in
/// first-occurrence order over the non-fixed pieces).
fn rederive_layout(specs: &[ParameterBlockSpec]) -> (Vec<String>, Vec<String>) {
    let mut penalty_labels = Vec::new();
    let mut outer_labels = Vec::new();
    let mut seen = BTreeMap::<String, usize>::new();
    for (b, spec) in specs.iter().enumerate() {
        assert_eq!(
            spec.penalties.len(),
            spec.initial_log_lambdas.len(),
            "realized block {b} must carry one initial log-lambda per emitted penalty",
        );
        for (j, penalty) in spec.penalties.iter().enumerate() {
            let label = penalty
                .precision_label()
                .map(str::to_owned)
                .unwrap_or_else(|| format!("__block_{b}_penalty_{j}"));
            penalty_labels.push(label.clone());
            if penalty.fixed_log_lambda().is_some() {
                continue;
            }
            if !seen.contains_key(&label) {
                seen.insert(label.clone(), outer_labels.len());
                outer_labels.push(label);
            }
        }
    }
    (penalty_labels, outer_labels)
}

/// Nesting depth of the deepest coefficient-group chain (root = depth 1).
fn max_group_depth(groups: &[GroupSeed]) -> usize {
    let parent: BTreeMap<&str, Option<&str>> =
        groups.iter().map(|g| (g.label, g.parent)).collect();
    fn depth(label: &str, parent: &BTreeMap<&str, Option<&str>>) -> usize {
        match parent.get(label).and_then(|p| *p) {
            Some(p) => 1 + depth(p, parent),
            None => 1,
        }
    }
    groups.iter().map(|g| depth(g.label, &parent)).max().unwrap_or(0)
}

#[test]
fn composed_two_of_everything_depth3_layout_stays_consistent_2315() {
    let zoo = composed_zoo();
    assert!(zoo.len() >= 4, "the composed zoo must exercise several configurations");

    for case in &zoo {
        // Precondition: every entry really is a 2-of-everything, depth-3 config.
        assert!(
            case.specs.len() >= 2,
            "[{}] composition requires >= 2 parameter blocks",
            case.name
        );
        let blocks_with_two_penalties = case
            .specs
            .iter()
            .filter(|s| s.penalties.len() >= 2)
            .count();
        assert!(
            blocks_with_two_penalties >= 2,
            "[{}] composition requires >= 2 blocks carrying >= 2 base penalties, found {}",
            case.name,
            blocks_with_two_penalties
        );
        assert!(
            case.groups.len() >= 2,
            "[{}] composition requires >= 2 composable prior-factor layers (groups)",
            case.name
        );
        assert!(
            max_group_depth(&case.groups) >= 3,
            "[{}] composition requires a coefficient-group hierarchy of depth >= 3, got {}",
            case.name,
            max_group_depth(&case.groups)
        );

        let groups: Vec<CoefficientGroupSpec> = case.groups.iter().map(group).collect();
        let realized = realize_coefficient_groups_for_custom_family(
            &case.specs,
            &groups,
            case.base_prior.clone(),
        )
        .unwrap_or_else(|e| panic!("[{}] composed configuration must realize: {e}", case.name));

        // 1. Emitted physical labels and the derived outer coordinate labels must
        //    match the independent re-derivation from the realized specs.
        let (derived_penalty_labels, derived_outer_labels) = rederive_layout(&realized.specs);
        assert_eq!(
            realized.penalty_labels, derived_penalty_labels,
            "[{}] emitted penalty labels disagree with the flattened block emission walk",
            case.name
        );
        assert_eq!(
            realized.outer_labels, derived_outer_labels,
            "[{}] realized outer (optimizer) coordinates disagree with the first-occurrence law",
            case.name
        );

        // 2. Emitted-count consistency: one label per emitted penalty, across all
        //    blocks, with matching per-block lambda counts.
        let total_emitted: usize = realized.specs.iter().map(|s| s.penalties.len()).sum();
        assert_eq!(
            realized.penalty_labels.len(),
            total_emitted,
            "[{}] every emitted penalty must contribute exactly one physical label",
            case.name
        );

        // 3. One prior per optimizer coordinate, and outer labels are unique.
        let RhoPrior::Independent(priors) = &realized.rho_prior else {
            panic!("[{}] realized coefficient-group prior must be coordinate-wise", case.name);
        };
        assert_eq!(
            priors.len(),
            realized.outer_labels.len(),
            "[{}] each optimizer coordinate must own exactly one prior",
            case.name
        );
        let mut unique = std::collections::BTreeSet::new();
        for label in &realized.outer_labels {
            assert!(
                unique.insert(label.clone()),
                "[{}] outer label '{label}' owns more than one optimizer coordinate",
                case.name
            );
        }

        // 4. Every non-fixed emitted penalty resolves to a real optimizer
        //    coordinate; fixed pieces resolve to none.
        for (b, spec) in realized.specs.iter().enumerate() {
            for (j, penalty) in spec.penalties.iter().enumerate() {
                let label = penalty
                    .precision_label()
                    .map(str::to_owned)
                    .unwrap_or_else(|| format!("__block_{b}_penalty_{j}"));
                if penalty.fixed_log_lambda().is_some() {
                    continue;
                }
                assert!(
                    realized.outer_labels.contains(&label),
                    "[{}] optimized penalty '{label}' (block {b}, piece {j}) has no optimizer coordinate",
                    case.name
                );
            }
        }
    }
}

#[test]
fn composed_depth3_group_priors_land_on_their_own_outer_coordinate_2315() {
    for case in &composed_zoo() {
        let groups: Vec<CoefficientGroupSpec> = case.groups.iter().map(group).collect();
        let realized = realize_coefficient_groups_for_custom_family(
            &case.specs,
            &groups,
            case.base_prior.clone(),
        )
        .unwrap_or_else(|e| panic!("[{}] composed configuration must realize: {e}", case.name));

        let RhoPrior::Independent(priors) = &realized.rho_prior else {
            panic!("[{}] realized prior must be coordinate-wise", case.name);
        };

        // Every declared group is an INDEPENDENT prior factor, so its label must
        // appear exactly once among the optimizer coordinates AND in
        // `independent_prior_factor_labels`, and the prior at that coordinate must
        // be the group's own NormalLogPrecision mapped to a Normal rho prior.
        assert_eq!(
            realized.independent_prior_factor_labels,
            case.groups.iter().map(|g| g.label.to_string()).collect::<Vec<_>>(),
            "[{}] independent prior-factor labels must be the declared groups in order",
            case.name
        );

        for seed in &case.groups {
            let hits = realized
                .outer_labels
                .iter()
                .filter(|l| l.as_str() == seed.label)
                .count();
            assert_eq!(
                hits, 1,
                "[{}] group '{}' must own exactly one optimizer coordinate, found {hits}",
                case.name, seed.label
            );
            let idx = realized
                .outer_labels
                .iter()
                .position(|l| l.as_str() == seed.label)
                .expect("group label present among outer coordinates");
            assert_eq!(
                priors[idx],
                RhoPrior::Normal { mean: seed.mean, sd: seed.sd },
                "[{}] group '{}' prior must land on its own optimizer coordinate",
                case.name, seed.label
            );

            // The group's emitted physical pieces all carry its label, and its
            // first physical occurrence carries its declared initial log-precision.
            let piece_count = realized
                .penalty_labels
                .iter()
                .filter(|l| l.as_str() == seed.label)
                .count();
            assert!(
                piece_count >= 1,
                "[{}] group '{}' must emit at least one physical penalty piece",
                case.name, seed.label
            );
            let first_lambda = realized
                .specs
                .iter()
                .flat_map(|spec| spec.penalties.iter().zip(spec.initial_log_lambdas.iter()))
                .find_map(|(penalty, lam)| {
                    (penalty.precision_label() == Some(seed.label)).then_some(*lam)
                })
                .expect("group label present among emitted penalties");
            assert_eq!(
                first_lambda, seed.init_log_precision,
                "[{}] group '{}' first emitted piece must carry its declared initial log-precision",
                case.name, seed.label
            );
        }
    }
}
