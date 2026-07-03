//! Coefficient-group realization for the custom-family blockwise carrier:
//! resolve declared `(block, column)` group selectors into penalty pieces +
//! one tied Gamma-precision rho coordinate per group, with hierarchical
//! parent/child concatenation.
//!
//! Pure relocation from `custom_family.rs` (issue #780 decomposition): the
//! block-index selector resolution, the per-coordinate rho-prior validation +
//! penalized-complexity validator, the base-prior expansion, and the main
//! `realize_coefficient_groups_for_custom_family`. No behavior change — bodies
//! are byte-identical; the public realizer is re-exported by the parent.

use super::{
    CoefficientBlockSelector, CoefficientGroupSpec, CustomFamilyError, ParameterBlockSpec,
    PenaltyMatrix, RealizedCoefficientGroup, RealizedCoefficientGroupSpecs, validate_blockspecs,
};
use ndarray::{Array1, Array2, s};
use std::collections::{BTreeMap, BTreeSet};

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
    base_count: usize,
    context: &str,
) -> Result<Vec<gam_problem::RhoPrior>, String> {
    match base_prior {
        gam_problem::RhoPrior::Independent(priors) => {
            if priors.len() != base_count {
                return Err(CustomFamilyError::DimensionMismatch { reason: format!(
                    "{context} base Independent rho prior length mismatch: got {}, expected {base_count}",
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
            Ok((0..base_count).map(|_| prior.clone()).collect())
        }
    }
}

pub fn realize_coefficient_groups_for_custom_family(
    specs: &[ParameterBlockSpec],
    groups: &[CoefficientGroupSpec],
    base_prior: gam_problem::RhoPrior,
) -> Result<RealizedCoefficientGroupSpecs, String> {
    use gam_terms::structure::coefficient_group_resolver::{ResolvedGroup, ResolvedGroupHierarchy};

    validate_blockspecs(specs)?;
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

    let mut realized_specs = specs.to_vec();
    let mut penalty_labels = Vec::<String>::new();
    let mut outer_labels = Vec::<String>::new();
    let base_count = specs.iter().map(|spec| spec.penalties.len()).sum::<usize>();
    let mut priors = expand_custom_group_base_prior(&base_prior, base_count, "coefficient groups")?;
    let mut base_prior_idx = 0usize;

    for (block_idx, spec) in specs.iter().enumerate() {
        for penalty_idx in 0..spec.penalties.len() {
            let label = format!("__block_{block_idx}_penalty_{penalty_idx}");
            penalty_labels.push(label.clone());
            outer_labels.push(label);
            base_prior_idx += 1;
        }
    }
    assert_eq!(base_prior_idx, base_count);

    for group in &realized_groups {
        outer_labels.push(group.label.clone());
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
        priors.push(group_prior);

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
                let p = realized_specs[block_idx].design.ncols();
                let mut matrix = Array2::<f64>::zeros((p, p));
                for column in &columns {
                    matrix[[*column, *column]] = 1.0;
                }
                realized_specs[block_idx]
                    .penalties
                    .push(PenaltyMatrix::Dense(matrix).with_precision_label(group.label.clone()));
                realized_specs[block_idx]
                    .nullspace_dims
                    .push(p.saturating_sub(columns.len()));
                let mut rho =
                    Array1::<f64>::zeros(realized_specs[block_idx].initial_log_lambdas.len() + 1);
                if !realized_specs[block_idx].initial_log_lambdas.is_empty() {
                    let old_len = realized_specs[block_idx].initial_log_lambdas.len();
                    rho.slice_mut(s![..old_len])
                        .assign(&realized_specs[block_idx].initial_log_lambdas);
                }
                let last = rho.len() - 1;
                rho[last] = group.initial_log_precision;
                realized_specs[block_idx].initial_log_lambdas = rho;
                penalty_labels.push(group.label.clone());
            }
        }
    }

    Ok(RealizedCoefficientGroupSpecs {
        specs: realized_specs,
        groups: realized_groups,
        penalty_labels,
        rho_prior: gam_problem::RhoPrior::Independent(priors),
        outer_labels,
    })
}
