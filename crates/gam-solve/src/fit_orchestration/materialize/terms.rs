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
    inference_notes: &mut Vec<String>,
    scale_dimensions: bool,
    policy: &gam_runtime::resource::ResourcePolicy,
    smooth_overrides: Option<&JsonValue>,
) -> Result<TermCollectionSpec, WorkflowError> {
    let mut spec = build_termspec(terms, data, col_map, inference_notes, policy)?;
    if scale_dimensions {
        enable_scale_dimensions(&mut spec);
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
    inference_notes: &mut Vec<String>,
) -> Result<(), WorkflowError> {
    if spec.linear_terms.is_empty() {
        return Ok(());
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
    let mut dropped = Vec::<String>::new();

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
            if term.double_penalty {
                return Err(WorkflowError::InvalidConfig {
                    reason: format!(
                        "{label}: explicitly penalized linear term '{}' is redundant with the \
                         implicit intercept or earlier scalar terms; remove the redundant term \
                         instead of relying on a ridge to identify a duplicate data direction",
                        term.name
                    ),
                });
            }
            dropped.push(format!(
                "{} (residual_norm={:.3e}, tol={:.3e})",
                term.name, residual_norm, tol
            ));
            continue;
        }
        if residual_norm > tol {
            basis.push(residual.mapv(|v| v / residual_norm));
        }
        kept.push(term.clone());
    }

    if !dropped.is_empty() {
        inference_notes.push(format!(
            "{label}: removed {} scalar term(s) that add no identifiable \
             direction beyond the implicit intercept and earlier scalar terms: {}",
            dropped.len(),
            dropped.join(", ")
        ));
        spec.linear_terms = kept;
    }
    Ok(())
}

pub(super) fn standard_adaptive_regularization_options(
    config: &FitConfig,
) -> Option<AdaptiveRegularizationOptions> {
    let enabled = config.adaptive_regularization.unwrap_or(false);
    enabled.then(|| AdaptiveRegularizationOptions {
        enabled: true,
        ..AdaptiveRegularizationOptions::default()
    })
}
