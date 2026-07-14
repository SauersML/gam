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
    spatial_center_counts: Option<&[Option<usize>]>,
) -> Result<TermCollectionSpec, WorkflowError> {
    let mut spec = build_termspec(terms, data, col_map, inference_notes, policy)?;
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

/// Apply the standard workflow's per-term adaptive spatial resolution plan.
///
/// Eligibility (never clobber a pinned basis):
/// * spatial radial families with a validated saturation contract (thin-plate /
///   Duchon / constant-curvature / measure-jet); Matérn has a separately learned
///   kernel range whose basin and numerical rank both move with the center count,
///   so it retains its established full/default count until a Matérn-specific
///   saturation proof exists; ordinary 1-D `s(x)` P-splines are also untouched;
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

/// Drop the Duchon *operator* penalties (the collocation-Gram mass `Σ(f−f̄)²`
/// and tension `Σ‖∇f‖²` lower-order blocks) for any fit that is NOT
/// Gaussian-identity, leaving only the exact RKHS-curvature `Primary` Gram and
/// the polynomial-nullspace trend ridge.
///
/// WHY (#1074). The Duchon default penalty is a Hilbert scale: curvature
/// (`Primary`) plus the lower-order mass/tension operator dials, each carrying
/// its own smoothing parameter that REML/LAML is meant to deselect when the
/// data don't support it. That deselection is faithful only in the
/// **ProfiledGaussian** REML arm, where the dispersion `φ̂` is profiled out and
/// the penalty's contribution to the criterion is scale-flat: an unsupported
/// operator block simply rails its `λ→∞` and drops out at no cost. In every
/// **fixed-dispersion** GLM arm (`φ=1`: Poisson/log, Binomial/logit, …) the
/// penalized-likelihood term enters at full weight, so the near-full-rank
/// operator-Gram blocks are *rewarded* by the LAML criterion for over-shrinking
/// the fit — a genuine criterion optimum, not an optimizer miss (same
/// ProfiledGaussian-vs-fixed-φ asymmetry as #1373). The result is systematic
/// under-recovery of the true mean (the `duchon(x,k)` Poisson regime missed
/// mgcv's `bs="ds"` recovery by 1.36×).
///
/// mgcv's `bs="ds"` carries a *single* curvature penalty — it never ships the
/// mass/tension operator overlay — so dropping those blocks for the GLM path
/// makes gam's default penalty structurally match the mature reference exactly,
/// removing only a block the fixed-φ criterion mis-rewards. The Gaussian path
/// is untouched: it keeps the full Hilbert scale and correctly deselects the
/// lower orders via profiled REML (and the cyclic-duchon / tps Gaussian fits
/// stay bit-identical). No new knob, REML/LAML stays always-on, no FD.
pub fn gate_duchon_operator_penalties_for_family(
    spec: &mut TermCollectionSpec,
    family: &gam_spec::LikelihoodSpec,
) {
    if family.is_gaussian_identity() {
        return;
    }
    for term in spec.smooth_terms.iter_mut() {
        disable_duchon_operator_penalties_in_basis(&mut term.basis);
    }
}

fn disable_duchon_operator_penalties_in_basis(basis: &mut gam_terms::smooth::SmoothBasisSpec) {
    use gam_terms::smooth::SmoothBasisSpec;
    match basis {
        SmoothBasisSpec::Duchon { spec, .. } => {
            // Keep `Primary` curvature + nullspace ridge (the mgcv `bs="ds"`
            // structure); silence the collocation-Gram lower orders.
            spec.operator_penalties = gam_terms::basis::DuchonOperatorPenaltySpec::all_disabled();
        }
        SmoothBasisSpec::ByVariable { inner, .. }
        | SmoothBasisSpec::FactorSumToZero { inner, .. } => {
            disable_duchon_operator_penalties_in_basis(inner);
        }
        SmoothBasisSpec::BySmooth { smooth, .. } => {
            disable_duchon_operator_penalties_in_basis(smooth);
        }
        _ => {}
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
