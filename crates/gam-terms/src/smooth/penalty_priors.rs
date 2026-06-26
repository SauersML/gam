//! Penalty-block Gamma hyperprior realization and coefficient-group resolution
//! for the term collection.
//!
//! Pure relocation from `smooth.rs` (issue #780 decomposition): the
//! penalty-block label/metadata derivation, the Gamma-precision /
//! penalized-complexity prior validators, the keyed and callback-driven
//! penalty-block Gamma prior realizers, the per-coordinate rho-prior validation
//! + base-prior expansion + group-prior combination, and the coefficient-group
//! column resolution + hierarchical penalty realization. No behavior change —
//! bodies are byte-identical and the entry points consumed elsewhere in
//! `smooth.rs` are re-imported by the parent so every call site is unchanged.

use super::{PenaltyBlockInfo, TermCollectionDesign};
use crate::basis::BasisError;
use crate::PenaltySpec;
use gam_spec::CoefficientGroupPrior;
use ndarray::{Array1, Array2};
use serde::{Deserialize, Serialize};
use std::collections::{BTreeMap, BTreeSet};
use std::ops::Range;

/// Selector that maps a declared coefficient group onto columns of the realized
/// term-collection design matrix.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum CoefficientSelector {
    /// Explicit global coefficient indices in the realized design matrix.
    GlobalColumns(Vec<usize>),
    /// A half-open global coefficient range.
    GlobalRange(Range<usize>),
    LinearTerm(String),
    RandomEffectTerm(String),
    SmoothTerm(String),
    /// Selected basis columns within one smooth term.
    SmoothTermColumns {
        term: String,
        columns: Vec<usize>,
    },
}

/// A declared coefficient group: a named selector set plus optional parent and
/// prior, used to realize hierarchical penalties on the design.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CoefficientGroupSpec {
    pub name: String,
    pub selectors: Vec<CoefficientSelector>,
    pub parent: Option<String>,
    pub prior: Option<CoefficientGroupPrior>,
    #[serde(skip, default)]
    pub prior_mean: gam_problem::CoefficientPriorMean,
}

/// The penalties / null-space dims / rho-prior realized from a coefficient
/// group hierarchy against a concrete design.
#[derive(Debug, Clone)]
pub struct RealizedCoefficientGroups {
    pub penalty_specs: Vec<PenaltySpec>,
    pub nullspace_dims: Vec<usize>,
    pub rho_prior: gam_spec::RhoPrior,
    pub group_column_indices: Vec<(String, Vec<usize>)>,
}

#[derive(Debug, Clone)]
pub struct PenaltyBlockGammaPriorMetadata<'a> {
    pub label: String,
    pub global_index: usize,
    pub termname: Option<&'a str>,
    pub source: String,
    pub effective_rank: usize,
    pub nullspace_dim_hint: usize,
}

fn penalty_block_label_candidates(info: &PenaltyBlockInfo) -> Vec<String> {
    let mut labels = Vec::<String>::new();
    labels.push(format!("penalty:{}", info.global_index));
    labels.push(info.global_index.to_string());
    if let Some(termname) = info.termname.as_ref() {
        labels.push(termname.clone());
        labels.push(format!("{termname}:{}", info.penalty.original_index));
    }
    if let crate::basis::PenaltySource::Other(label) = &info.penalty.source {
        labels.push(label.clone());
    }
    labels.push(format!("{:?}", info.penalty.source));
    labels.sort();
    labels.dedup();
    labels
}

fn penalty_block_metadata(info: &PenaltyBlockInfo) -> PenaltyBlockGammaPriorMetadata<'_> {
    PenaltyBlockGammaPriorMetadata {
        label: info
            .termname
            .clone()
            .unwrap_or_else(|| format!("penalty:{}", info.global_index)),
        global_index: info.global_index,
        termname: info.termname.as_deref(),
        source: format!("{:?}", info.penalty.source),
        effective_rank: info.penalty.effective_rank,
        nullspace_dim_hint: info.penalty.nullspace_dim_hint,
    }
}

pub(super) fn validate_gamma_precision_prior(
    label: &str,
    shape: f64,
    rate: f64,
) -> Result<(), BasisError> {
    if !shape.is_finite() || shape <= 0.0 {
        crate::bail_invalid_basis!(
            "Gamma precision hyperprior for penalty block '{label}' requires shape > 0, got {shape}"
        );
    }
    if !rate.is_finite() || rate < 0.0 {
        crate::bail_invalid_basis!(
            "Gamma precision hyperprior for penalty block '{label}' requires rate >= 0, got {rate}"
        );
    }
    Ok::<(), _>(())
}

pub(super) fn validate_penalized_complexity_prior(
    label: &str,
    upper: f64,
    tail_prob: f64,
) -> Result<(), BasisError> {
    if !upper.is_finite() || upper <= 0.0 {
        crate::bail_invalid_basis!(
            "Penalized-complexity hyperprior for '{label}' requires upper > 0, got {upper}"
        );
    }
    if !tail_prob.is_finite() || tail_prob <= 0.0 || tail_prob >= 1.0 {
        crate::bail_invalid_basis!(
            "Penalized-complexity hyperprior for '{label}' requires tail probability in (0, 1), got {tail_prob}"
        );
    }
    Ok::<(), _>(())
}

pub(super) fn realize_penalty_block_gamma_priors<F>(
    design: &TermCollectionDesign,
    mut callback: F,
) -> Result<gam_spec::RhoPrior, BasisError>
where
    F: FnMut(&PenaltyBlockGammaPriorMetadata<'_>) -> Option<(f64, f64)>,
{
    let mut priors = Vec::<gam_spec::RhoPrior>::with_capacity(design.penaltyinfo.len());
    for info in &design.penaltyinfo {
        let metadata = penalty_block_metadata(info);
        if let Some((shape, rate)) = callback(&metadata) {
            validate_gamma_precision_prior(&metadata.label, shape, rate)?;
            priors.push(gam_spec::RhoPrior::GammaPrecision { shape, rate });
        } else {
            priors.push(gam_spec::RhoPrior::Flat);
        }
    }
    Ok(gam_spec::RhoPrior::Independent(priors))
}

pub(super) fn realize_keyed_penalty_block_gamma_priors(
    design: &TermCollectionDesign,
    priors: &[(String, f64, f64)],
) -> Result<gam_spec::RhoPrior, BasisError> {
    let mut keyed = BTreeMap::<String, (f64, f64)>::new();
    for (label, shape, rate) in priors {
        if keyed.insert(label.clone(), (*shape, *rate)).is_some() {
            crate::bail_invalid_basis!(
                "duplicate Gamma precision hyperprior for penalty block label '{label}'"
            );
        }
    }
    let mut consumed = BTreeSet::<String>::new();
    let prior = realize_penalty_block_gamma_priors(design, |metadata| {
        let info = design
            .penaltyinfo
            .iter()
            .find(|info| info.global_index == metadata.global_index)
            .expect("metadata global index should match penaltyinfo");
        for label in penalty_block_label_candidates(info) {
            if let Some(value) = keyed.get(&label) {
                consumed.insert(label);
                return Some(*value);
            }
        }
        None
    })?;
    let unknown: Vec<String> = keyed
        .keys()
        .filter(|label| !consumed.contains(*label))
        .cloned()
        .collect();
    if !unknown.is_empty() {
        let available = design
            .penaltyinfo
            .iter()
            .flat_map(penalty_block_label_candidates)
            .collect::<BTreeSet<_>>()
            .into_iter()
            .collect::<Vec<_>>()
            .join(", ");
        crate::bail_invalid_basis!(
            "unknown Gamma precision hyperprior penalty block label(s): {}; available labels: {available}",
            unknown.join(", ")
        );
    }
    Ok(prior)
}

fn validate_rho_prior_coordinate(
    prior: &gam_spec::RhoPrior,
    context: &str,
) -> Result<(), BasisError> {
    match prior {
        gam_spec::RhoPrior::Flat => Ok(()),
        gam_spec::RhoPrior::Normal { mean, sd } => {
            if !mean.is_finite() {
                crate::bail_invalid_basis!(
                    "{context} Normal log-precision prior requires finite mean, got {mean}"
                );
            }
            if !sd.is_finite() || *sd <= 0.0 {
                crate::bail_invalid_basis!(
                    "{context} Normal log-precision prior requires sd > 0, got {sd}"
                );
            }
            Ok(())
        }
        gam_spec::RhoPrior::GammaPrecision { shape, rate } => {
            validate_gamma_precision_prior(context, *shape, *rate)
        }
        gam_spec::RhoPrior::PenalizedComplexity { upper, tail_prob } => {
            validate_penalized_complexity_prior(context, *upper, *tail_prob)
        }
        gam_spec::RhoPrior::Independent(_) => Err(BasisError::InvalidInput(format!(
            "{context} must be a scalar rho prior, not a nested Independent prior"
        ))),
    }
}

fn expand_base_rho_prior(
    base_prior: &gam_spec::RhoPrior,
    base_count: usize,
    context: &str,
) -> Result<Vec<gam_spec::RhoPrior>, BasisError> {
    match base_prior {
        gam_spec::RhoPrior::Independent(priors) => {
            if priors.len() != base_count {
                crate::bail_invalid_basis!(
                    "{context} base Independent rho prior length mismatch: got {}, expected {base_count}",
                    priors.len()
                );
            }
            for (idx, prior) in priors.iter().enumerate() {
                validate_rho_prior_coordinate(prior, &format!("{context} base prior {idx}"))?;
            }
            Ok(priors.clone())
        }
        prior => {
            validate_rho_prior_coordinate(prior, context)?;
            Ok((0..base_count).map(|_| prior.clone()).collect())
        }
    }
}

fn combine_group_rho_prior(
    base_prior: &gam_spec::RhoPrior,
    base_count: usize,
    groups: &[CoefficientGroupSpec],
) -> Result<gam_spec::RhoPrior, BasisError> {
    let mut priors = Vec::with_capacity(base_count + groups.len());
    priors.extend(expand_base_rho_prior(
        base_prior,
        base_count,
        "coefficient groups",
    )?);
    for group in groups {
        let context = format!("coefficient group '{}'", group.name);
        let prior = match group.prior.as_ref() {
            Some(prior) => {
                prior.validate(&context).map_err(BasisError::InvalidInput)?;
                prior.to_rho_prior()
            }
            None => {
                validate_rho_prior_coordinate(base_prior, &context)?;
                base_prior.clone()
            }
        };
        priors.push(prior);
    }
    Ok(gam_spec::RhoPrior::Independent(priors))
}

fn insert_range(
    cols: &mut BTreeSet<usize>,
    range: Range<usize>,
    p: usize,
    context: &str,
) -> Result<(), BasisError> {
    if range.end > p {
        crate::bail_dim_basis!(
            "{context} coefficient range {}..{} exceeds design width {p}",
            range.start,
            range.end
        );
    }
    cols.extend(range);
    Ok(())
}

fn resolve_group_columns(
    design: &TermCollectionDesign,
    group: &CoefficientGroupSpec,
) -> Result<BTreeSet<usize>, BasisError> {
    let p = design.design.ncols();
    let mut cols = BTreeSet::<usize>::new();
    for selector in &group.selectors {
        match selector {
            CoefficientSelector::GlobalColumns(indices) => {
                for &idx in indices {
                    if idx >= p {
                        crate::bail_dim_basis!(
                            "coefficient group '{}' references global column {idx}, but design width is {p}",
                            group.name
                        );
                    }
                    cols.insert(idx);
                }
            }
            CoefficientSelector::GlobalRange(range) => insert_range(
                &mut cols,
                range.clone(),
                p,
                &format!("coefficient group '{}'", group.name),
            )?,
            CoefficientSelector::LinearTerm(name) => {
                let (_, range) = design
                    .linear_ranges
                    .iter()
                    .find(|(term, _)| term == name)
                    .ok_or_else(|| {
                        BasisError::InvalidInput(format!(
                            "coefficient group '{}' references unknown linear term '{name}'",
                            group.name
                        ))
                    })?;
                insert_range(&mut cols, range.clone(), p, &group.name)?;
            }
            CoefficientSelector::RandomEffectTerm(name) => {
                let (_, range) = design
                    .random_effect_ranges
                    .iter()
                    .find(|(term, _)| term == name)
                    .ok_or_else(|| {
                        BasisError::InvalidInput(format!(
                            "coefficient group '{}' references unknown random-effect term '{name}'",
                            group.name
                        ))
                    })?;
                insert_range(&mut cols, range.clone(), p, &group.name)?;
            }
            CoefficientSelector::SmoothTerm(name) => {
                let term = design
                    .smooth
                    .terms
                    .iter()
                    .find(|term| &term.name == name)
                    .ok_or_else(|| {
                        BasisError::InvalidInput(format!(
                            "coefficient group '{}' references unknown smooth term '{name}'",
                            group.name
                        ))
                    })?;
                let start = p - design.smooth.total_smooth_cols() + term.coeff_range.start;
                insert_range(
                    &mut cols,
                    start..(start + term.coeff_range.len()),
                    p,
                    &group.name,
                )?;
            }
            CoefficientSelector::SmoothTermColumns { term, columns } => {
                let smooth_term = design
                    .smooth
                    .terms
                    .iter()
                    .find(|smooth_term| &smooth_term.name == term)
                    .ok_or_else(|| {
                        BasisError::InvalidInput(format!(
                            "coefficient group '{}' references unknown smooth term '{term}'",
                            group.name
                        ))
                    })?;
                let smooth_start = p - design.smooth.total_smooth_cols();
                for &local_col in columns {
                    if local_col >= smooth_term.coeff_range.len() {
                        crate::bail_dim_basis!(
                            "coefficient group '{}' references smooth term '{term}' local column {local_col}, but the term has {} columns",
                            group.name,
                            smooth_term.coeff_range.len()
                        );
                    }
                    cols.insert(smooth_start + smooth_term.coeff_range.start + local_col);
                }
            }
        }
    }
    if cols.is_empty() {
        crate::bail_invalid_basis!(
            "coefficient group '{}' contains no coefficients",
            group.name
        );
    }
    Ok(cols)
}

pub(super) fn realize_coefficient_groups(
    design: &TermCollectionDesign,
    groups: &[CoefficientGroupSpec],
    base_prior: &gam_spec::RhoPrior,
) -> Result<RealizedCoefficientGroups, BasisError> {
    use crate::structure::coefficient_group_resolver::{ResolvedGroup, ResolvedGroupHierarchy};

    let p = design.design.ncols();
    // Carrier-specific validation and selector resolution. The standard-term
    // carrier is columns of the realized design matrix; `resolve_group_columns`
    // turns each declared selector into a `BTreeSet<usize>` and rejects empty
    // selector lists. The prior is validated here because its diagnostic
    // context uses the standard-term label.
    for group in groups {
        if group.selectors.is_empty() {
            crate::bail_invalid_basis!("coefficient group '{}' contains no selectors", group.name);
        }
        if let Some(prior) = group.prior.as_ref() {
            prior
                .validate(&format!("coefficient group '{}'", group.name))
                .map_err(BasisError::InvalidInput)?;
        }
    }

    let resolved_groups = groups
        .iter()
        .map(|group| {
            Ok(ResolvedGroup {
                label: group.name.clone(),
                parent: group.parent.clone(),
                coordinates: resolve_group_columns(design, group)?,
            })
        })
        .collect::<Result<Vec<_>, BasisError>>()?;
    // Carrier-agnostic policy: unique non-empty labels, acyclic hierarchy,
    // child ⊆ parent, interior == union of children.
    let hierarchy =
        ResolvedGroupHierarchy::build(resolved_groups).map_err(BasisError::InvalidInput)?;

    let mut penalty_specs: Vec<PenaltySpec> = design
        .penalties
        .iter()
        .map(PenaltySpec::from_blockwise_ref)
        .collect();
    let mut nullspace_dims = design.nullspace_dims.clone();
    let mut group_column_indices = Vec::<(String, Vec<usize>)>::with_capacity(groups.len());
    for (group, resolved) in groups.iter().zip(hierarchy.groups()) {
        let cols = &resolved.coordinates;
        let mut penalty = Array2::<f64>::zeros((p, p));
        let penalty_components = hierarchy.concatenated_penalty_components(&group.name);
        let active_cols = penalty_components
            .iter()
            .flat_map(|component| component.iter().copied())
            .collect::<BTreeSet<_>>();
        let local_mean = group
            .prior_mean
            .evaluate(
                active_cols.len(),
                &format!("coefficient group '{}'", group.name),
            )
            .map_err(|err| BasisError::InvalidInput(err.to_string()))?;
        let mut prior_mean = Array1::<f64>::zeros(p);
        // Hierarchical Gamma precision update.
        //
        // For a leaf group,
        //
        //   p(beta_g | lambda_g) p(lambda_g)
        //     ∝ lambda_g^{|g|/2}
        //       exp[-lambda_g (beta_g - mu_g)' S_g (beta_g - mu_g) / 2]
        //       lambda_g^{a_g-1} exp[-b_g lambda_g],
        //
        // so fixed-beta MAP gives
        //
        //   lambda_g* = (a_g + |g|/2 - 1)
        //               / (b_g + (beta_g - mu_g)' S_g (beta_g - mu_g) / 2).
        //
        // Interior nodes use the same identity with beta_g formed by
        // concatenating child beta vectors.  Equivalently, |g| and the
        // quadratic are sums over the recursively expanded child factors.  In
        // the standard term-collection path there is one rho coordinate per
        // group, so we materialize that summed child penalty into the group's
        // dense S_g.  Leaves reduce to the ordinary identity penalty.
        for component in &penalty_components {
            for &col in component {
                penalty[[col, col]] += 1.0;
            }
        }
        for (mean_idx, &col) in active_cols.iter().enumerate() {
            prior_mean[col] = local_mean[mean_idx];
        }
        penalty_specs.push(PenaltySpec::DenseWithMean {
            matrix: penalty,
            prior_mean: gam_problem::CoefficientPriorMean::constant(prior_mean),
        });
        nullspace_dims.push(p.saturating_sub(active_cols.len()));
        group_column_indices.push((group.name.clone(), cols.iter().copied().collect()));
    }

    Ok(RealizedCoefficientGroups {
        penalty_specs,
        nullspace_dims,
        rho_prior: combine_group_rho_prior(base_prior, design.penalties.len(), groups)?,
        group_column_indices,
    })
}
