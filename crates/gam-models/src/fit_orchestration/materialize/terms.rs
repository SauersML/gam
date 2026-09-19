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
    adaptive_resolution: Option<&[Option<gam_terms::smooth::AdaptiveResolution>]>,
) -> Result<TermCollectionSpec, WorkflowError> {
    let mut spec = build_termspec(terms, data, col_map, inference_notes)?;
    if scale_dimensions {
        enable_scale_dimensions(&mut spec);
    }
    // The standard formula path starts every formula-default smooth at its
    // data-derived pilot resolution; auto-sized multivariate radial smooths
    // start at their penalized-resolution count. Per-term evidence-backed
    // refinements return through this same materializer. Explicit sizes carry
    // no adaptive provenance, and Python overrides apply afterward, so both
    // remain authoritative.
    if let Some(plan) = adaptive_resolution {
        apply_adaptive_resolution_plan(&mut spec, data, plan)?;
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
/// Only a smooth whose size nobody chose carries an adaptive resolution
/// ([`gam_terms::smooth::adaptive_resolution_of`]); every explicit
/// formula/programmatic size is left alone. A missing plan entry keeps the
/// formula pilot, except that an auto-sized multivariate radial smooth starts
/// at [`gam_terms::basis::starting_num_centers`]. Center counts are held
/// between the term's structural minimum and one center per row. Python
/// `smooths={...}` overrides are applied by the caller AFTER this, so they
/// override the refined value unconditionally.
fn apply_adaptive_resolution_plan(
    spec: &mut TermCollectionSpec,
    data: &Dataset,
    plan: &[Option<gam_terms::smooth::AdaptiveResolution>],
) -> Result<(), WorkflowError> {
    use gam_terms::basis::starting_num_centers;
    use gam_terms::smooth::{AdaptiveResolution, adaptive_resolution_of, apply_adaptive_resolution};
    let n = data.values.nrows();
    if n == 0 {
        return Ok(());
    }
    for (term_index, term) in spec.smooth_terms.iter_mut().enumerate() {
        let Some(current) = adaptive_resolution_of(&term.basis) else {
            continue;
        };
        let proposed = plan.get(term_index).cloned().flatten();
        let target = if let AdaptiveResolution::Centers(planned) = current {
            let nullspace_dim = gam_terms::smooth::spatial_term_min_center_count(term);
            let structural_minimum = nullspace_dim.saturating_add(1).min(n);
            let radial_dim = spatial_center_strategy_mut(&mut term.basis).map(|(_, cols)| cols.len());
            let count = match (&proposed, radial_dim) {
                (Some(AdaptiveResolution::Centers(requested)), _) => *requested,
                (_, Some(d)) if d > 1 => starting_num_centers(n, d, nullspace_dim),
                _ => planned,
            };
            AdaptiveResolution::Centers(count.max(structural_minimum).min(n))
        } else {
            proposed.unwrap_or_else(|| current.clone())
        };
        if adaptive_resolution_of(&term.basis).as_ref() == Some(&target) {
            continue;
        }
        apply_adaptive_resolution(&mut term.basis, &target).map_err(
            |error| WorkflowError::InvalidConfig {
                reason: format!(
                    "failed to set the adaptive resolution of smooth term '{}': {error}",
                    term.name
                ),
            },
        )?;
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

