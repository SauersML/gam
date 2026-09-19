use super::*;

/// Canonical termspec lowering path: formula DSL builds the initial
/// `SmoothBasisSpec`, then any `gamfit.fit(..., smooths={...})` Python
/// override registry entry whose `feature_cols` match the term's column set
/// replaces the spec's kind-specific tunables in place (explicit center
/// coordinate matrices, knot vectors, kernel hyperparameters). When all
/// descriptor fields default to the same values the DSL would auto-pick, the
/// override is a no-op and the spec is bit-identical to the formula-only
/// path. Callers that don't have overrides simply pass `smooth_overrides =
/// None`.
pub(crate) fn build_termspec_with_geometry_and_overrides(
    terms: &[ParsedTerm],
    data: &Dataset,
    col_map: &HashMap<String, usize>,
    inference_notes: &mut FitNotes,
    scale_dimensions: bool,
    smooth_overrides: Option<&JsonValue>,
    spatial_center_counts: Option<&[Option<usize>]>,
) -> Result<TermCollectionSpec, WorkflowError> {
    let mut spec = build_termspec(terms, data, col_map, inference_notes)?;
    if scale_dimensions {
        enable_scale_dimensions(&mut spec);
    }
    // The standard formula path starts auto-sized multivariate radial smooths at
    // their structural minimum. Univariate radial smooths retain the canonical
    // formula resolution because that resolution is derived from the competing
    // univariate spline basis. Per-term evidence-backed expansions return through
    // this same materializer. Explicit formula counts are not `Auto`, and Python
    // overrides apply afterward, so both remain authoritative.
    if let Some(counts) = spatial_center_counts {
        apply_adaptive_spatial_center_counts(&mut spec, data, counts)?;
    }
    if let Some(overrides) = smooth_overrides {
        gam_terms::smooth_overrides::apply_smooth_overrides(
            &mut spec,
            overrides,
            data,
            inference_notes,
        )
        .map_err(|reason| WorkflowError::InvalidConfig { reason })?;
    }
    Ok(spec)
}

/// Apply the standard workflow's per-term adaptive resolution plan.
///
/// Eligibility (never clobber a pinned basis):
/// * spatial radial families with a validated saturation contract (thin-plate /
///   Duchon / constant-curvature / measure-jet); Matérn has a separately learned
///   kernel range whose basin and numerical rank both move with the center count,
///   so it retains its established full/default count until a Matérn-specific
///   saturation proof exists;
/// * the formula-default 1-D `s(x)` B-spline, whose knot count nobody chose
///   (`BSplineKnotSpec::Automatic { adaptive: true, .. }`); an explicit `k=` /
///   `knots=` is a fixed spec and is untouched;
/// * the current strategy must retain [`CenterStrategy::Auto`] provenance;
///   every explicit formula/programmatic strategy is therefore left alone;
/// Python `smooths={...}` overrides are applied by the caller AFTER this, so they
/// override the escalated value unconditionally.
fn apply_adaptive_spatial_center_counts(
    spec: &mut TermCollectionSpec,
    data: &Dataset,
    requested_counts: &[Option<usize>],
) -> Result<(), WorkflowError> {
    use gam_terms::basis::{
        center_strategy_is_auto, center_strategy_with_num_centers, starting_num_centers,
    };
    let n = data.values.nrows();
    if n == 0 {
        return Ok(());
    }
    for (term_index, term) in spec.smooth_terms.iter_mut().enumerate() {
        if let gam_terms::smooth::SmoothBasisSpec::BSpline1D {
            spec:
                gam_terms::basis::BSplineBasisSpec {
                    knotspec:
                        gam_terms::basis::BSplineKnotSpec::Automatic {
                            num_internal_knots,
                            adaptive: true,
                            ..
                        },
                    ..
                },
            ..
        } = &mut term.basis
        {
            // The formula default `s(x)` keeps its `adaptive` provenance while
            // taking the knot count this loop proposed, so every refit is still
            // owned (and measured) by the same loop.
            if let Some(proposed) = requested_counts.get(term_index).copied().flatten() {
                *num_internal_knots = proposed;
            }
            continue;
        }
        let structural_minimum = gam_terms::smooth::spatial_term_min_center_count(term)
            .saturating_add(1)
            .min(n);
        let Some((strategy, feature_cols)) = spatial_center_strategy_mut(&mut term.basis) else {
            continue;
        };
        let d = feature_cols.len();
        if d == 0 {
            continue;
        }
        if !center_strategy_is_auto(strategy) {
            continue;
        }
        let proposed = requested_counts.get(term_index).copied().flatten();
        let target = proposed
            .unwrap_or_else(|| {
                if d == 1 {
                    strategy.planned_num_centers(d)
                } else {
                    starting_num_centers(n, d)
                }
            })
            .max(structural_minimum)
            .min(n);
        *strategy = center_strategy_with_num_centers(strategy, target, d).map_err(|error| {
            WorkflowError::InvalidConfig {
                reason: format!(
                    "failed to set adaptive center count for spatial term '{}': {error}",
                    term.name
                ),
            }
        })?;
    }
    Ok(())
}

/// Mutable `(center_strategy, feature_cols)` for a spatial radial smooth, peeling
/// the `ByVariable`/`FactorSumToZero` row-gating envelopes; `None` for any
/// non-spatial or non-radial basis (B-spline, tensor, sphere, PCA, …).
fn spatial_center_strategy_mut(
    basis: &mut gam_terms::smooth::SmoothBasisSpec,
) -> Option<(&mut gam_terms::basis::CenterStrategy, Vec<usize>)> {
    use gam_terms::smooth::SmoothBasisSpec as B;
    match basis {
        B::ByVariable { inner, .. } | B::FactorSumToZero { inner, .. } => {
            spatial_center_strategy_mut(inner)
        }
        B::BySmooth { smooth, .. } => spatial_center_strategy_mut(smooth),
        B::ThinPlate {
            feature_cols,
            spec,
            input_scale: _,
        } => {
            let cols = feature_cols.clone();
            Some((&mut spec.center_strategy, cols))
        }
        B::Duchon {
            feature_cols,
            spec,
            input_scale: _,
        } => {
            let cols = feature_cols.clone();
            Some((&mut spec.center_strategy, cols))
        }
        B::ConstantCurvature { feature_cols, spec } => {
            let cols = feature_cols.clone();
            Some((&mut spec.center_strategy, cols))
        }
        B::MeasureJet {
            feature_cols,
            spec,
            input_scale: _,
        } => {
            let cols = feature_cols.clone();
            Some((&mut spec.center_strategy, cols))
        }
        _ => None,
    }
}

fn linear_term_training_column(
    data: &Dataset,
    term: &LinearTermSpec,
) -> Result<Array1<f64>, WorkflowError> {
    // Single shared realizer: numeric product gated by any categorical-level
    // indicators (factor-aware `:` interaction). Same column the design
    // assembly emits, so the marginal-slope rank check sees the realized cell
    // columns rather than the raw categorical codes.
    term.realized_design_column(data.values.view())
        .map_err(|reason| WorkflowError::SchemaMismatch { reason })
}

fn residualize_against_orthonormal_basis(
    column: &Array1<f64>,
    basis: &[Array1<f64>],
) -> Array1<f64> {
    let mut residual = column.clone();
    for q in basis {
        let coeff = residual.dot(q);
        residual.scaled_add(-coeff, q);
    }
    residual
}

fn l2_norm(column: &Array1<f64>) -> f64 {
    column.iter().map(|v| v * v).sum::<f64>().sqrt()
}

pub(crate) fn prune_unidentified_linear_terms_for_marginal_slope(
    spec: &mut TermCollectionSpec,
    data: &Dataset,
    label: &str,
    inference_notes: &mut FitNotes,
) -> Result<Vec<UnidentifiedScalarTerm>, WorkflowError> {
    if spec.linear_terms.is_empty() {
        return Ok(Vec::new());
    }

    let n = data.values.nrows();
    if n == 0 {
        return Err(WorkflowError::InvalidConfig {
            reason: format!("{label}: cannot rank-check scalar terms on zero rows"),
        });
    }

    let mut basis = Vec::<Array1<f64>>::new();
    let intercept = Array1::<f64>::ones(n);
    let intercept_norm = l2_norm(&intercept);
    if intercept_norm == 0.0 || !intercept_norm.is_finite() {
        return Err(WorkflowError::InvalidConfig {
            reason: format!("{label}: implicit intercept has invalid norm {intercept_norm}"),
        });
    }
    basis.push(intercept.mapv(|v| v / intercept_norm));

    let rank_alpha = gam_linalg::faer_ndarray::default_rrqr_rank_alpha();
    let mut scale = intercept_norm.max(1.0);
    let mut kept = Vec::<LinearTermSpec>::with_capacity(spec.linear_terms.len());
    let mut dropped = Vec::<UnidentifiedScalarTerm>::new();

    for term in &spec.linear_terms {
        let column = linear_term_training_column(data, term)?;
        let norm = l2_norm(&column);
        if !norm.is_finite() {
            return Err(WorkflowError::InvalidConfig {
                reason: format!("{label}: linear term '{}' has non-finite norm", term.name),
            });
        }
        scale = scale.max(norm.max(1.0));
        let residual = residualize_against_orthonormal_basis(&column, &basis);
        let residual_norm = l2_norm(&residual);
        let tol = rank_alpha * f64::EPSILON * ((n + basis.len() + 1).max(1) as f64) * scale;
        let is_data_redundant = residual_norm <= tol;
        let has_constraints = term.coefficient_min.is_some() || term.coefficient_max.is_some();
        if is_data_redundant {
            if has_constraints {
                return Err(WorkflowError::InvalidConfig {
                    reason: format!(
                        "{label}: constrained linear term '{}' is redundant with the implicit \
                         intercept or earlier scalar terms; remove the constraint or the \
                         redundant term",
                        term.name
                    ),
                });
            }
            // Every formula linear effect carries the null-recovery ridge by default
            // (b7b874a2a). A ridge adds no identifiable data direction, so a
            // redundant column is pruned whether or not it carries one.
            dropped.push(UnidentifiedScalarTerm {
                formula: label.to_string(),
                term: term.name.clone(),
                residual_norm,
                tolerance: tol,
            });
            continue;
        }
        if residual_norm > tol {
            basis.push(residual.mapv(|v| v / residual_norm));
        }
        kept.push(term.clone());
    }

    if !dropped.is_empty() {
        inference_notes.advise(format!(
            "{label}: removed {} scalar term(s) that add no identifiable \
             direction beyond the implicit intercept and earlier scalar terms: {}",
            dropped.len(),
            dropped
                .iter()
                .map(|removed| format!(
                    "{} (residual_norm={:.3e}, tol={:.3e})",
                    removed.term, removed.residual_norm, removed.tolerance
                ))
                .collect::<Vec<_>>()
                .join(", ")
        ));
        spec.linear_terms = kept;
    }
    Ok(dropped)
}

