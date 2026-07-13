//! Coefficient-group realization for the custom-family blockwise carrier:
//! resolve declared `(block, column)` group selectors into penalty pieces +
//! one tied Gamma-precision rho coordinate per group, with hierarchical
//! parent/child concatenation.
//!
//! The realizer emits each physical penalty together with its nullity, initial
//! precision, label, and optimizer-coordinate prior, then materializes every
//! public side vector from that single ordered sequence. The public realizer is
//! re-exported by the parent module.

use super::{
    CoefficientBlockSelector, CoefficientGroupSpec, CustomFamilyError, ParameterBlockSpec,
    PenaltyMatrix, RealizedCoefficientGroup, RealizedCoefficientGroupSpecs, validate_blockspecs,
};
use ndarray::{Array1, Array2};
use std::collections::{BTreeMap, BTreeSet};

/// One physical penalty emission in its final block-local order. Keeping the
/// matrix, nullity mode, initial precision, label, and prior in one record makes
/// it impossible for a group penalty to be inserted into a block while its
/// metadata is appended to a differently ordered side vector.
#[derive(Clone)]
struct RealizedPenaltyEmission {
    penalty: PenaltyMatrix,
    nullspace_dim: Option<usize>,
    initial_log_lambda: f64,
    label: String,
    /// Prior for this penalty's optimizer coordinate. Fixed penalties have no
    /// optimizer coordinate; tied physical pieces carry the same prior.
    prior: Option<gam_problem::RhoPrior>,
}

pub(crate) fn coefficient_group_block_index(
    specs: &[ParameterBlockSpec],
    selector: &CoefficientBlockSelector,
) -> Result<usize, String> {
    match selector {
        CoefficientBlockSelector::Index(index) => {
            if *index >= specs.len() {
                Err(format!(
                    "coefficient group references block index {index}, but only {} blocks exist",
                    specs.len()
                ))
            } else {
                Ok(*index)
            }
        }
        CoefficientBlockSelector::Name(name) => specs
            .iter()
            .position(|spec| spec.name == *name)
            .ok_or_else(|| format!("coefficient group references unknown block '{name}'")),
    }
}

pub(crate) fn validate_group_rho_prior_coordinate(
    prior: &gam_problem::RhoPrior,
    context: &str,
) -> Result<(), String> {
    match prior {
        gam_problem::RhoPrior::Flat => Ok(()),
        gam_problem::RhoPrior::Normal { mean, sd } => {
            if !mean.is_finite() {
                return Err(format!(
                    "{context} Normal log-precision prior requires finite mean, got {mean}"
                ));
            }
            if !sd.is_finite() || *sd <= 0.0 {
                return Err(format!(
                    "{context} Normal log-precision prior requires sd > 0, got {sd}"
                ));
            }
            Ok(())
        }
        gam_problem::RhoPrior::GammaPrecision { shape, rate } => {
            if !shape.is_finite() || *shape <= 0.0 {
                return Err(CustomFamilyError::DimensionMismatch {
                    reason: format!(
                        "{context} Gamma precision prior requires shape > 0, got {shape}"
                    ),
                }
                .into());
            }
            if !rate.is_finite() || *rate < 0.0 {
                return Err(format!(
                    "{context} Gamma precision prior requires rate >= 0, got {rate}"
                ));
            }
            Ok(())
        }
        gam_problem::RhoPrior::PenalizedComplexity { upper, tail_prob } => {
            validate_penalized_complexity_prior(context, *upper, *tail_prob)
        }
        gam_problem::RhoPrior::Independent(_) => Err(CustomFamilyError::ConstraintViolation {
            reason: format!("{context} must be a scalar rho prior, not a nested Independent prior"),
        }
        .into()),
    }
}

/// Shared validation of penalized-complexity hyperparameters: `upper` finite and
/// strictly positive, `tail_prob` a probability in the open interval `(0, 1)`.
pub(crate) fn validate_penalized_complexity_prior(
    context: &str,
    upper: f64,
    tail_prob: f64,
) -> Result<(), String> {
    if !upper.is_finite() || upper <= 0.0 {
        return Err(format!(
            "{context} penalized-complexity prior requires upper > 0, got {upper}"
        ));
    }
    if !tail_prob.is_finite() || tail_prob <= 0.0 || tail_prob >= 1.0 {
        return Err(format!(
            "{context} penalized-complexity prior requires tail probability in (0, 1), got {tail_prob}"
        ));
    }
    Ok(())
}

pub(crate) fn expand_custom_group_base_prior(
    base_prior: &gam_problem::RhoPrior,
    base_outer_count: usize,
    context: &str,
) -> Result<Vec<gam_problem::RhoPrior>, String> {
    match base_prior {
        gam_problem::RhoPrior::Independent(priors) => {
            if priors.len() != base_outer_count {
                return Err(CustomFamilyError::DimensionMismatch { reason: format!(
                    "{context} base Independent rho prior length mismatch: got {}, expected {base_outer_count}",
                    priors.len()
                ) }.into());
            }
            for (idx, prior) in priors.iter().enumerate() {
                validate_group_rho_prior_coordinate(prior, &format!("{context} base prior {idx}"))?;
            }
            Ok(priors.clone())
        }
        prior => {
            validate_group_rho_prior_coordinate(prior, context)?;
            Ok((0..base_outer_count).map(|_| prior.clone()).collect())
        }
    }
}

pub fn realize_coefficient_groups_for_custom_family(
    specs: &[ParameterBlockSpec],
    groups: &[CoefficientGroupSpec],
    base_prior: gam_problem::RhoPrior,
) -> Result<RealizedCoefficientGroupSpecs, String> {
    use gam_terms::structure::coefficient_group_resolver::{ResolvedGroup, ResolvedGroupHierarchy};

    let base_penalty_counts = validate_blockspecs(specs)?;
    // Carrier-specific validation. The prior and the custom-only
    // `initial_log_precision` field are validated here because they have no
    // analogue on the standard-term carrier; label, duplicate, empty-set, and
    // hierarchy checks are delegated to the shared resolver below.
    for group in groups {
        if let Some(prior) = group.prior.as_ref() {
            prior.validate(&format!("coefficient group '{}'", group.label))?;
        }
        if let Some(initial) = group.initial_log_precision
            && !initial.is_finite()
        {
            return Err(CustomFamilyError::DimensionMismatch {
                reason: format!(
                    "coefficient group '{}' initial log precision must be finite, got {initial}",
                    group.label
                ),
            }
            .into());
        }
    }

    // Carrier = `(block_idx, column)` coordinates of the parameter blocks.
    // Resolve every declared label into its coordinate set, then hand the
    // carrier-agnostic policy (labels, hierarchy, subsets, child unions) to the
    // shared resolver.
    let resolved_groups = groups
        .iter()
        .map(|group| {
            let mut coordinates = BTreeSet::<(usize, usize)>::new();
            for label in &group.coefficients {
                let block_idx = coefficient_group_block_index(specs, &label.block)?;
                let p = specs[block_idx].design.ncols();
                if label.column >= p {
                    return Err(format!(
                        "coefficient group '{}' references column {} in block '{}' (index {block_idx}), but the block has {p} columns",
                        group.label, label.column, specs[block_idx].name
                    ));
                }
                coordinates.insert((block_idx, label.column));
            }
            Ok(ResolvedGroup {
                label: group.label.clone(),
                parent: group.parent.clone(),
                coordinates,
            })
        })
        .collect::<Result<Vec<_>, String>>()?;
    let hierarchy = ResolvedGroupHierarchy::build(resolved_groups)?;

    let realized_groups = groups
        .iter()
        .zip(hierarchy.groups())
        .map(|(group, resolved)| RealizedCoefficientGroup {
            label: group.label.clone(),
            parent: group.parent.clone(),
            coefficients: resolved.coordinates.iter().copied().collect(),
            prior: group.prior.clone(),
            initial_log_precision: group.initial_log_precision.unwrap_or(0.0),
        })
        .collect::<Vec<_>>();

    // Resolve the pre-group optimizer topology once with the same authoritative
    // layout builder used by the fit. `Independent` priors are indexed by
    // optimizer coordinates, not physical matrices: tied pieces share a prior
    // and fixed pieces consume no prior coordinate.
    let base_layout = penalty_label_layout_with_joint(specs, base_penalty_counts, Vec::new())?;
    let base_priors = expand_custom_group_base_prior(
        &base_prior,
        base_layout.initial_rho.len(),
        "coefficient groups",
    )?;
    let mut physical_idx = 0usize;
    let mut base_labels = BTreeSet::<String>::new();
    let mut infer_nullity_by_block = Vec::with_capacity(specs.len());
    let mut emissions_by_block = Vec::<Vec<RealizedPenaltyEmission>>::with_capacity(specs.len());
    for (block_idx, spec) in specs.iter().enumerate() {
        if !spec.nullspace_dims.is_empty() && spec.nullspace_dims.len() != spec.penalties.len() {
            return Err(CustomFamilyError::DimensionMismatch {
                reason: format!(
                    "coefficient-group block '{}' has {} penalties but {} structural nullities; nullity metadata must be either complete or absent",
                    spec.name,
                    spec.penalties.len(),
                    spec.nullspace_dims.len()
                ),
            }
            .into());
        }
        let infer_nullity = !spec.penalties.is_empty() && spec.nullspace_dims.is_empty();
        infer_nullity_by_block.push(infer_nullity);
        let mut emissions = Vec::with_capacity(spec.penalties.len());
        for penalty_idx in 0..spec.penalties.len() {
            let penalty = spec.penalties[penalty_idx].clone();
            let label = resolved_physical_penalty_label(&penalty, block_idx, penalty_idx);
            base_labels.insert(label.clone());
            let prior = base_layout.physical_to_outer[physical_idx]
                .map(|outer_idx| base_priors[outer_idx].clone());
            emissions.push(RealizedPenaltyEmission {
                penalty,
                nullspace_dim: (!infer_nullity).then(|| spec.nullspace_dims[penalty_idx]),
                initial_log_lambda: spec.initial_log_lambdas[penalty_idx],
                label,
                prior,
            });
            physical_idx += 1;
        }
        emissions_by_block.push(emissions);
    }
    assert_eq!(physical_idx, base_layout.physical_count());

    // Group labels classify penalty pieces as independent Gaussian factors in
    // assembly, so silently reusing a base label would change the meaning of an
    // unrelated base penalty. Labels therefore have one unambiguous owner.
    for group in &realized_groups {
        if base_labels.contains(&group.label) {
            return Err(CustomFamilyError::ConstraintViolation {
                reason: format!(
                    "coefficient group label '{}' collides with an existing base penalty label",
                    group.label
                ),
            }
            .into());
        }
    }

    for group in &realized_groups {
        let group_prior = match group.prior.as_ref() {
            Some(prior) => prior.to_rho_prior(),
            None => match &base_prior {
                gam_problem::RhoPrior::Independent(_) => {
                    return Err(CustomFamilyError::ConstraintViolation { reason: format!(
                        "coefficient group '{}' must declare a prior when base_prior is Independent",
                        group.label
                    ) }.into());
                }
                prior => prior.clone(),
            },
        };

        // Hierarchical Gamma precision update.
        //
        // For one Gaussian coefficient group with fixed beta and precision
        // lambda,
        //
        //   p(beta_g | lambda) p(lambda)
        //     ∝ lambda^{|g|/2} exp[-lambda q_g/2]
        //       lambda^{a_g-1} exp[-b_g lambda],
        //   q_g = (beta_g - mu_g)' S_g (beta_g - mu_g).
        //
        // Maximizing the log posterior in lambda gives
        //
        //   lambda* = (a_g + |g|/2 - 1) / (b_g + q_g/2).
        //
        // If a node has children, beta_g is the concatenation of the child
        // coefficient vectors.  The parent density is therefore the product
        // of those child Gaussian factors under one lambda_g: replace |g| and
        // q_g by sums over the child components, expanding recursively when a
        // child is itself an interior node.  We preserve that identity by
        // emitting one physical penalty piece per concatenated child component
        // and tying those pieces with the parent's precision label.  This is
        // not a block-sum shortcut: overlapping children remain separate
        // factors, so their log normalizers and quadratic contributions both
        // add.
        let penalty_components = hierarchy.concatenated_penalty_components(&group.label);
        for component in penalty_components {
            let mut by_block = BTreeMap::<usize, Vec<usize>>::new();
            for &(block_idx, column) in &component {
                by_block.entry(block_idx).or_default().push(column);
            }
            for (block_idx, columns) in by_block {
                let p = specs[block_idx].design.ncols();
                let mut matrix = Array2::<f64>::zeros((p, p));
                for column in &columns {
                    matrix[[*column, *column]] = 1.0;
                }
                emissions_by_block[block_idx].push(RealizedPenaltyEmission {
                    penalty: PenaltyMatrix::Dense(matrix).with_precision_label(group.label.clone()),
                    // `ParameterBlockSpec` currently represents inferred nullities
                    // with one block-wide empty vector. Preserve that mode for the
                    // whole block instead of creating a dangerous partial vector by
                    // appending only the new group's known nullity.
                    nullspace_dim: (!infer_nullity_by_block[block_idx])
                        .then_some(p.saturating_sub(columns.len())),
                    initial_log_lambda: group.initial_log_precision,
                    label: group.label.clone(),
                    prior: Some(group_prior.clone()),
                });
            }
        }
    }

    let mut realized_specs = specs.to_vec();
    for ((spec, emissions), &infer_nullity) in realized_specs
        .iter_mut()
        .zip(emissions_by_block.iter())
        .zip(infer_nullity_by_block.iter())
    {
        spec.penalties = emissions
            .iter()
            .map(|emission| emission.penalty.clone())
            .collect();
        spec.initial_log_lambdas =
            Array1::from_iter(emissions.iter().map(|emission| emission.initial_log_lambda));
        spec.nullspace_dims = if infer_nullity {
            Vec::new()
        } else {
            emissions
                .iter()
                .map(|emission| {
                    emission
                        .nullspace_dim
                        .expect("structural-nullity blocks emit complete metadata")
                })
                .collect()
        };
    }

    // Derive both physical and outer metadata from the exact same final
    // block-flattened emission walk consumed by `penalty_label_layout_with_joint`.
    // The first physical occurrence of a label owns its outer coordinate; tied
    // group pieces reuse it and must agree on prior and initial precision.
    let mut penalty_labels = Vec::<String>::new();
    let mut outer_labels = Vec::<String>::new();
    let mut priors = Vec::<gam_problem::RhoPrior>::new();
    let mut outer_initial_log_lambdas = Vec::<f64>::new();
    let mut outer_by_label = BTreeMap::<String, usize>::new();
    for emissions in &emissions_by_block {
        for emission in emissions {
            penalty_labels.push(emission.label.clone());
            if emission.penalty.fixed_log_lambda().is_some() {
                continue;
            }
            let prior = emission
                .prior
                .as_ref()
                .expect("every optimized penalty emission owns a rho prior");
            if let Some(&outer_idx) = outer_by_label.get(&emission.label) {
                if priors[outer_idx] != *prior {
                    return Err(CustomFamilyError::ConstraintViolation {
                        reason: format!(
                            "precision label '{}' carries inconsistent rho priors across physical penalty pieces",
                            emission.label
                        ),
                    }
                    .into());
                }
                let first = outer_initial_log_lambdas[outer_idx];
                if (first - emission.initial_log_lambda).abs() > 1e-10 {
                    return Err(CustomFamilyError::ConstraintViolation {
                        reason: format!(
                            "precision label '{}' has inconsistent initial log-precisions: {first} and {}",
                            emission.label, emission.initial_log_lambda
                        ),
                    }
                    .into());
                }
                continue;
            }
            let outer_idx = outer_labels.len();
            outer_by_label.insert(emission.label.clone(), outer_idx);
            outer_labels.push(emission.label.clone());
            priors.push(prior.clone());
            outer_initial_log_lambdas.push(emission.initial_log_lambda);
        }
    }

    // Every group is a separate Gaussian factor in the prior product (the
    // hierarchical-Gamma identity above), so its label must be carried into
    // `BlockwiseFitOptions::independent_prior_factor_labels`: the evidence
    // normalizer for these pieces is per-factor `½(rank Sₖ·log λₖ +
    // log|Sₖ|₊)`, and coalescing them into one block pseudo-logdet would
    // lose `½ log λ` for every dimension shared by overlapping groups.
    let independent_prior_factor_labels = realized_groups
        .iter()
        .map(|group| group.label.clone())
        .collect();

    Ok(RealizedCoefficientGroupSpecs {
        specs: realized_specs,
        groups: realized_groups,
        penalty_labels,
        rho_prior: gam_problem::RhoPrior::Independent(priors),
        outer_labels,
        independent_prior_factor_labels,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use gam_spec::CoefficientGroupPrior;

    fn one_penalty_block(
        name: &str,
        initial_log_lambda: f64,
        nullspace_dims: Vec<usize>,
    ) -> ParameterBlockSpec {
        ParameterBlockSpec {
            name: name.to_string(),
            design: crate::DesignMatrix::from(Array2::<f64>::eye(2)),
            offset: Array1::zeros(2),
            penalties: vec![PenaltyMatrix::Dense(Array2::<f64>::eye(2))],
            nullspace_dims,
            initial_log_lambdas: Array1::from_vec(vec![initial_log_lambda]),
            initial_beta: None,
            gauge_priority: 100,
            jacobian_callback: None,
            stacked_design: None,
            stacked_offset: None,
        }
    }

    #[test]
    fn multi_block_group_priors_follow_realized_penalty_order_2315() {
        let specs = vec![
            // Empty means infer every nullity in this block. Appending the group
            // must keep it empty rather than manufacturing a partial vector.
            one_penalty_block("early", -1.0, Vec::new()),
            one_penalty_block("late", -2.0, vec![0]),
        ];
        let mut group =
            CoefficientGroupSpec::new("early_group", vec![crate::coefficient_label("early", 0)])
                .with_prior(CoefficientGroupPrior::NormalLogPrecision {
                    mean: 30.0,
                    sd: 3.0,
                });
        group.initial_log_precision = Some(7.0);
        let base_prior = gam_problem::RhoPrior::Independent(vec![
            gam_problem::RhoPrior::Normal {
                mean: 10.0,
                sd: 1.0,
            },
            gam_problem::RhoPrior::Normal {
                mean: 20.0,
                sd: 2.0,
            },
        ]);

        let realized = realize_coefficient_groups_for_custom_family(
            &specs,
            std::slice::from_ref(&group),
            base_prior,
        )
        .expect("two-block coefficient-group layout must realize");

        let expected_labels = vec![
            "__block_0_penalty_0".to_string(),
            "early_group".to_string(),
            "__block_1_penalty_0".to_string(),
        ];
        assert_eq!(realized.penalty_labels, expected_labels);
        assert_eq!(realized.outer_labels, expected_labels);
        assert_eq!(realized.specs[0].penalties.len(), 2);
        assert_eq!(realized.specs[0].nullspace_dims, Vec::<usize>::new());
        assert_eq!(
            realized.specs[0].initial_log_lambdas.as_slice(),
            Some(&[-1.0, 7.0][..])
        );
        assert_eq!(realized.specs[1].penalties.len(), 1);
        assert_eq!(realized.specs[1].nullspace_dims, vec![0]);

        let gam_problem::RhoPrior::Independent(priors) = &realized.rho_prior else {
            panic!("realized coefficient-group prior must be coordinate-wise")
        };
        assert_eq!(
            priors,
            &vec![
                gam_problem::RhoPrior::Normal {
                    mean: 10.0,
                    sd: 1.0,
                },
                gam_problem::RhoPrior::Normal {
                    mean: 30.0,
                    sd: 3.0,
                },
                gam_problem::RhoPrior::Normal {
                    mean: 20.0,
                    sd: 2.0,
                },
            ]
        );

        let layout =
            crate::penalty_label_layout_with_joint(&realized.specs, vec![2, 1], Vec::new())
                .expect("realized specs must reproduce the exported outer layout");
        assert_eq!(layout.initial_rho.as_slice(), Some(&[-1.0, 7.0, -2.0][..]));
        assert_eq!(layout.physical_to_outer, vec![Some(0), Some(1), Some(2)]);
    }

    #[test]
    fn tied_and_fixed_base_penalties_use_optimizer_coordinate_priors_2315() {
        let mut early = one_penalty_block("early", -1.0, vec![0]);
        early.penalties = vec![
            PenaltyMatrix::Dense(Array2::<f64>::eye(2)).with_precision_label("tied_base"),
            PenaltyMatrix::Dense(Array2::<f64>::eye(2)).with_precision_label("tied_base"),
            PenaltyMatrix::Dense(Array2::<f64>::eye(2)).with_fixed_log_lambda(4.0),
        ];
        early.nullspace_dims = vec![0, 0, 0];
        early.initial_log_lambdas = Array1::from_vec(vec![-1.0, -1.0, 3.0]);
        let late = one_penalty_block("late", -2.0, vec![0]);

        let mut group =
            CoefficientGroupSpec::new("group", vec![crate::coefficient_label("early", 0)])
                .with_prior(CoefficientGroupPrior::NormalLogPrecision {
                    mean: 30.0,
                    sd: 3.0,
                });
        group.initial_log_precision = Some(7.0);
        let base_prior = gam_problem::RhoPrior::Independent(vec![
            gam_problem::RhoPrior::Normal {
                mean: 10.0,
                sd: 1.0,
            },
            gam_problem::RhoPrior::Normal {
                mean: 20.0,
                sd: 2.0,
            },
        ]);

        let realized =
            realize_coefficient_groups_for_custom_family(&[early, late], &[group], base_prior)
                .expect("base priors must follow pre-group optimizer coordinates");

        assert_eq!(
            realized.penalty_labels,
            vec![
                "tied_base".to_string(),
                "tied_base".to_string(),
                "__block_0_penalty_2".to_string(),
                "group".to_string(),
                "__block_1_penalty_0".to_string(),
            ]
        );
        assert_eq!(
            realized.outer_labels,
            vec![
                "tied_base".to_string(),
                "group".to_string(),
                "__block_1_penalty_0".to_string(),
            ]
        );
        let gam_problem::RhoPrior::Independent(priors) = &realized.rho_prior else {
            panic!("realized coefficient-group prior must be coordinate-wise")
        };
        assert_eq!(
            priors,
            &vec![
                gam_problem::RhoPrior::Normal {
                    mean: 10.0,
                    sd: 1.0,
                },
                gam_problem::RhoPrior::Normal {
                    mean: 30.0,
                    sd: 3.0,
                },
                gam_problem::RhoPrior::Normal {
                    mean: 20.0,
                    sd: 2.0,
                },
            ]
        );

        let layout =
            crate::penalty_label_layout_with_joint(&realized.specs, vec![4, 1], Vec::new())
                .expect("realized specs must preserve tied and fixed base topology");
        assert_eq!(layout.initial_rho.as_slice(), Some(&[-1.0, 7.0, -2.0][..]));
        assert_eq!(
            layout.physical_to_outer,
            vec![Some(0), Some(0), None, Some(1), Some(2)]
        );
    }

    #[test]
    fn coefficient_group_labels_cannot_reclassify_base_penalties_2315() {
        let cases = vec![
            (
                PenaltyMatrix::Dense(Array2::<f64>::eye(2)).with_precision_label("declared_base"),
                "declared_base",
            ),
            (
                PenaltyMatrix::Dense(Array2::<f64>::eye(2)),
                "__block_0_penalty_0",
            ),
        ];

        for (penalty, colliding_label) in cases {
            let mut spec = one_penalty_block("base", -1.0, vec![0]);
            spec.penalties[0] = penalty;
            let group = CoefficientGroupSpec::new(
                colliding_label,
                vec![crate::coefficient_label("base", 0)],
            );
            let error = realize_coefficient_groups_for_custom_family(
                &[spec],
                &[group],
                gam_problem::RhoPrior::Flat,
            )
            .expect_err("group and base penalty labels must have distinct owners");
            assert!(
                error.contains("collides with an existing base penalty label"),
                "unexpected collision error: {error}"
            );
        }
    }
}
