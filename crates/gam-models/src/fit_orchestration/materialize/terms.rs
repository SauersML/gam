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
    // The one place a formula default becomes its pilot: only a route whose
    // resolution loop grows the fit carries a plan, and it starts every basis
    // that loop grows at `gam_terms::smooth::starting_resolution`. Every other
    // route keeps the provisioned formula default, adequate without growth
    // (#3149). Per-term evidence-backed refinements return through this same
    // materializer. Explicit sizes carry no adaptive provenance, and Python
    // overrides apply afterward, so both remain authoritative.
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
/// formula/programmatic size is left alone. A plan entry is the loop's
/// evidence-backed refinement of that smooth. A missing entry starts it at its
/// pilot ([`gam_terms::smooth::starting_resolution`]), but only when the loop
/// can grow it, meaning its refinement nests
/// ([`gam_terms::smooth::adaptive_refinement_can_nest`], #3331). A basis whose
/// refinement does not nest is never grown, so it keeps its provisioned
/// default; started at the pilot, it would stay there (#3149). The pilot is
/// also the root every refinement is nested over, so it is written with the
/// target. Python `smooths={...}` overrides are applied by the caller AFTER
/// this, so they override the refined value unconditionally.
fn apply_adaptive_resolution_plan(
    spec: &mut TermCollectionSpec,
    data: &Dataset,
    plan: &[Option<gam_terms::smooth::AdaptiveResolution>],
) -> Result<(), WorkflowError> {
    use gam_terms::smooth::{
        adaptive_refinement_can_nest, adaptive_resolution_of, apply_adaptive_resolution,
        starting_resolution,
    };
    if data.values.nrows() == 0 {
        return Ok(());
    }
    for (term_index, term) in spec.smooth_terms.iter_mut().enumerate() {
        if adaptive_resolution_of(&term.basis).is_none() {
            continue;
        }
        let requested = plan.get(term_index).cloned().flatten();
        // A basis whose refinement does not nest is never grown by the loop
        // (#3331), so it keeps its provisioned default: started at the pilot it
        // could never leave, it would only lose resolution (#3149).
        if requested.is_none() && !adaptive_refinement_can_nest(&term.basis) {
            continue;
        }
        let start = starting_resolution(&term.basis, data.values.view()).ok_or_else(|| {
            WorkflowError::InvalidConfig {
                reason: format!(
                    "adaptive smooth term '{}' has no starting resolution on these data",
                    term.name
                ),
            }
        })?;
        let target = requested.unwrap_or_else(|| start.clone());
        // Applied even when the count already equals the target: the start is
        // the root the refinement chain grows from, and the spec records it.
        apply_adaptive_resolution(&mut term.basis, &start, &target).map_err(
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
        let residual = residualize_against_orthonormal_basis(&column, &basis);
        let residual_norm = l2_norm(&residual);
        // A scalar coefficient absorbs any rescaling of its column, so whether
        // the column adds a direction is a question about its angle to the span
        // so far: the residual is measured against the column's own norm. A
        // scale shared across columns would make the answer depend on the
        // units of other terms and on their order in the formula.
        //
        // The size of that measurement is the QR's columnwise backward error.
        // `residualize_against_orthonormal_basis` is modified Gram-Schmidt,
        // numerically Householder QR on `[0; A]` (Bjorck & Paige 1992), and QR
        // perturbs each column by at most `γ_K‖a_j‖` (Higham, ASNA 2nd ed.,
        // Thm 19.4). `Q` is exactly orthogonal in that model, so a column truly
        // in the span leaves a computed residual no larger than that (#4045).
        let columns = basis.len() + 1;
        let tol = gam_linalg::roundoff::householder_qr_backward_band(n + columns, columns, norm);
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

#[cfg(test)]
mod scalar_rank_scale_invariance_tests {
    use super::*;
    use gam_data::{ColumnKindTag, DataSchema, SchemaColumn};
    use ndarray::Array2;

    fn linear(name: &str, column: usize) -> LinearTermSpec {
        LinearTermSpec {
            name: name.to_string(),
            feature_col: column,
            feature_cols: vec![column],
            categorical_levels: vec![],
            double_penalty: true,
            coefficient_geometry: gam_terms::smooth::LinearCoefficientGeometry::Unconstrained,
            coefficient_min: None,
            coefficient_max: None,
            frozen_function_mass: None,
        }
    }

    fn kept_after_prune(order: &[(&str, usize)], data: &Dataset) -> Vec<String> {
        let mut spec = TermCollectionSpec {
            linear_terms: order.iter().map(|&(name, column)| linear(name, column)).collect(),
            random_effect_terms: vec![],
            smooth_terms: vec![],
            level: Default::default(),
        };
        let mut notes = FitNotes::default();
        prune_unidentified_linear_terms_for_marginal_slope(&mut spec, data, "test", &mut notes)
            .expect("rank check runs");
        let mut kept: Vec<String> = spec.linear_terms.iter().map(|t| t.name.clone()).collect();
        kept.sort();
        kept
    }

    /// A timestamp-scale column must not make a later small-unit covariate look
    /// redundant: both orders keep both identified columns, and an exact
    /// multiple of the timestamp is still pruned.
    #[test]
    fn identified_scalar_terms_survive_other_terms_units_and_order() {
        let n = 2000;
        let headers: Vec<String> =
            ["t", "x", "t2"].iter().map(|name| name.to_string()).collect();
        let values = Array2::from_shape_fn((n, 3), |(row, column)| {
            let t = 1.7e9 + row as f64 * 1.0e4;
            match column {
                0 => t,
                1 => 0.01 * (row as f64 * 0.7).sin(),
                _ => 2.0 * t,
            }
        });
        let data = Dataset {
            headers: headers.clone(),
            values,
            schema: DataSchema {
                columns: headers
                    .iter()
                    .map(|name| SchemaColumn {
                        name: name.clone(),
                        kind: ColumnKindTag::Continuous,
                        levels: vec![],
                    })
                    .collect(),
            },
            column_kinds: vec![ColumnKindTag::Continuous; 3],
        };
        let both = vec!["t".to_string(), "x".to_string()];
        assert_eq!(kept_after_prune(&[("t", 0), ("x", 1)], &data), both);
        assert_eq!(kept_after_prune(&[("x", 1), ("t", 0)], &data), both);
        assert_eq!(kept_after_prune(&[("t", 0), ("x", 1), ("t2", 2)], &data), both);
    }
}

