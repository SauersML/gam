use super::*;

pub(crate) fn sas_log_deltaridgeweight() -> f64 {
    // Weak fixed stabilization for the SAS tail parameter to avoid
    // boundary/flat-region pathologies in outer optimization.
    1e-4
}

#[inline]
pub(crate) fn sas_log_delta_edge_barrierweight() -> f64 {
    // Keep SAS raw log-delta away from tanh-saturation edges where
    // link sensitivities collapse and outer gradients become uninformative.
    1e-2
}

#[inline]
pub(crate) fn sas_log_delta_bound() -> f64 {
    crate::mixture_link::SAS_LOG_DELTA_BOUND
}

#[inline]
pub(crate) fn sas_log_delta_edge_barriercostgrad(raw_log_delta: f64) -> (f64, f64) {
    let w = sas_log_delta_edge_barrierweight();
    if w <= 0.0 || !raw_log_delta.is_finite() {
        return (0.0, 0.0);
    }
    let b = sas_log_delta_bound().max(f64::EPSILON);
    let t = (raw_log_delta / b).tanh();
    let one_minus_t2 = (1.0 - t * t).max(1e-12);
    let cost = -w * one_minus_t2.ln();
    // d/draw[-w log(1-t^2)] = (2w/B) * t.
    let grad = (2.0 * w / b) * t;
    (cost, grad)
}

#[inline]
pub(crate) fn sas_epsilon_bound() -> f64 {
    // Fixed smooth bound on raw SAS epsilon during outer optimization.
    8.0
}

#[inline]
pub(crate) fn sas_effective_epsilon(raw_epsilon: f64) -> (f64, f64) {
    let bound = sas_epsilon_bound().max(f64::EPSILON);
    let t = (raw_epsilon / bound).tanh();
    let epsilon = bound * t;
    let d_epsilon_d_raw = 1.0 - t * t;
    (epsilon, d_epsilon_d_raw)
}

#[inline]
pub(crate) fn sas_effective_epsilon_second(raw_epsilon: f64) -> (f64, f64, f64) {
    let bound = sas_epsilon_bound().max(f64::EPSILON);
    let t = (raw_epsilon / bound).tanh();
    let first = 1.0 - t * t;
    let second = -2.0 * t * first / bound;
    (bound * t, first, second)
}

#[inline]
pub(crate) fn sas_log_delta_edge_barriercostgradhess(raw_log_delta: f64) -> (f64, f64, f64) {
    let w = sas_log_delta_edge_barrierweight();
    if w <= 0.0 || !raw_log_delta.is_finite() {
        return (0.0, 0.0, 0.0);
    }
    let b = sas_log_delta_bound().max(f64::EPSILON);
    let t = (raw_log_delta / b).tanh();
    let one_minus_t2 = (1.0 - t * t).max(1e-12);
    let cost = -w * one_minus_t2.ln();
    let grad = (2.0 * w / b) * t;
    let hess = (2.0 * w / (b * b)) * one_minus_t2;
    (cost, grad, hess)
}

pub(crate) fn materialize_link_outer_hessian(
    hessian: gam_problem::HessianResult,
    theta_dim: usize,
) -> Result<Array2<f64>, EstimationError> {
    match hessian.materialize_dense() {
        Ok(Some(h)) => {
            if h.nrows() != theta_dim || h.ncols() != theta_dim {
                crate::bail_invalid_estim!(
                    "unified evaluator Hessian shape {}x{} != theta_dim {}",
                    h.nrows(),
                    h.ncols(),
                    theta_dim
                );
            }
            Ok(h)
        }
        Ok(None) => Err(EstimationError::InvalidInput(
            "unified evaluator returned no analytic Hessian in ValueGradientHessian mode"
                .to_string(),
        )),
        Err(err) => Err(EstimationError::InvalidInput(format!(
            "failed to materialize analytic link Hessian: {err}"
        ))),
    }
}

/// Evaluate the analytic gradient of the external REML objective.
pub fn evaluate_externalgradient<X>(
    y: ArrayView1<'_, f64>,
    w: ArrayView1<'_, f64>,
    x: X,
    offset: ArrayView1<'_, f64>,
    s_list: &[BlockwisePenalty],
    opts: &ExternalOptimOptions,
    rho: &Array1<f64>,
) -> Result<Array1<f64>, EstimationError>
where
    X: Into<DesignMatrix>,
{
    let specs: Vec<PenaltySpec> = s_list.iter().map(PenaltySpec::from_blockwise_ref).collect();
    let x = x.into();
    if let Some(message) = row_mismatch_message(y.len(), w.len(), x.nrows(), offset.len()) {
        crate::bail_invalid_estim!("{}", message);
    }

    let p = x.ncols();
    validate_penalty_specs(&specs, p, "evaluate_externalgradient")?;
    let (canonical, active_nullspace_dims) = gam_terms::construction::canonicalize_penalty_specs(
        &specs,
        &opts.nullspace_dims,
        p,
        "evaluate_externalgradient",
    )?;
    if rho.len() != active_nullspace_dims.len() {
        crate::bail_invalid_estim!(
            "rho dimension mismatch: rho_dim={}, active_penalties={}",
            rho.len(),
            active_nullspace_dims.len()
        );
    }

    let (cfg, _) = resolved_external_config(opts)?;

    let y_o = y.to_owned();
    let w_o = w.to_owned();
    let offset_o = offset.to_owned();
    let conditioning = ParametricColumnConditioning::infer_from_penalty_specs(&x, &specs);
    let x_fit = conditioning.apply_to_design(&x);
    let fit_linear_constraints =
        conditioning.transform_linear_constraints_to_internal(opts.linear_constraints.clone());

    let mut reml_state = RemlState::newwith_offset(
        y_o.view(),
        x_fit,
        w_o.view(),
        offset_o.view(),
        canonical,
        p,
        &cfg,
        Some(active_nullspace_dims),
        None,
        fit_linear_constraints,
    )?;
    reml_state.set_penalty_shrinkage_floor(opts.penalty_shrinkage_floor);
    reml_state.set_rho_prior(opts.rho_prior.clone());
    reml_state.set_link_states(
        cfg.link_kind.mixture_state().cloned(),
        cfg.link_kind.sas_state().copied(),
    );

    reml_state.compute_gradient(rho)
}

fn gaussian_identity_inner_residual_norm(
    y: ArrayView1<'_, f64>,
    w: ArrayView1<'_, f64>,
    x: &DesignMatrix,
    offset: ArrayView1<'_, f64>,
    canonical_penalties: &[gam_terms::construction::CanonicalPenalty],
    rho: &Array1<f64>,
    beta: &Array1<f64>,
) -> Result<f64, EstimationError> {
    if beta.len() != x.ncols() {
        crate::bail_invalid_estim!(
            "beta dimension mismatch: beta_dim={}, x_cols={}",
            beta.len(),
            x.ncols()
        );
    }
    if rho.len() != canonical_penalties.len() {
        crate::bail_invalid_estim!(
            "rho dimension mismatch: rho_dim={}, active_penalties={}",
            rho.len(),
            canonical_penalties.len()
        );
    }

    let mut residual = x.apply(beta);
    residual += &offset;
    residual -= &y;
    residual *= &w;
    let mut gradient = x.apply_transpose(&residual);

    for (k, cp) in canonical_penalties.iter().enumerate() {
        let lambda = rho[k].exp();
        if lambda == 0.0 || cp.rank() == 0 {
            continue;
        }
        let r = cp.col_range.clone();
        let centered = &beta.slice(s![r.start..r.end]) - &cp.prior_mean;
        let penalty_grad = cp.local.dot(&centered) * lambda;
        gradient
            .slice_mut(s![r.start..r.end])
            .scaled_add(1.0, &penalty_grad);
    }

    Ok(gradient.iter().map(|v| v * v).sum::<f64>().sqrt())
}

/// Evaluate IFT and flat warm-start inner residuals at `rho + delta_rho`.
///
/// Computes the inner-KKT residual norm at the IFT-predicted coefficient
/// `β_pred(ρ+Δρ)` obtained by linearizing the inner solution around the
/// converged `β̂(ρ)`, alongside the residual norm for the "flat" warm start
/// `β̂(ρ)` (the same coefficient without any IFT correction). The pair lets
/// callers verify that the IFT predictor reduces the inner residual to the
/// expected second-order remainder in `‖Δρ‖`.
///
/// # Math
///
/// Let `β̂(ρ)` minimize the penalized inner objective and `v_j = ∂β̂/∂ρ_j`
/// be the IFT sensitivity vectors at `ρ`. The first-order predictor is
///
/// ```text
///   β_pred(ρ + Δρ) = β̂(ρ) − Σ_j Δρ_j · v_j.
/// ```
///
/// Writing `r(β, ρ) = ∇_β L(β, ρ)` for the inner-KKT residual, the test
/// invariant exercised by callers is
///
/// ```text
///   ‖ r( β_pred(ρ+Δρ),  ρ + Δρ ) ‖ = O( ‖Δρ‖² ).
/// ```
///
/// The flat baseline `‖ r( β̂(ρ), ρ + Δρ ) ‖` is `O(‖Δρ‖)` for comparison.
///
/// # Arguments
///
/// * `y`, `w`, `x`, `offset` — full-data response, weights, design, offset.
/// * `s_list` — blockwise penalty specifications matching `rho`.
/// * `opts` — external optimization options; must be `GaussianIdentity`
///   with no linear constraints.
/// * `rho` — base log-smoothing parameter vector at which the IFT
///   sensitivities are taken.
/// * `delta_rho` — perturbation applied to `rho` for the residual probe.
///
/// # Returns
///
/// `(ift_residual_norm, flat_residual_norm)` — the L2 norm of the inner
/// KKT residual at `β_pred(ρ+Δρ)` and at the flat warm start `β̂(ρ)`,
/// both evaluated at `ρ + Δρ`.
///
/// # Used by
///
/// Tests that exercise the IFT predictor's residual-order property; not
/// part of the production solver hot path.
pub fn evaluate_external_ift_residual_at_perturbed_rho<X>(
    y: ArrayView1<'_, f64>,
    w: ArrayView1<'_, f64>,
    x: X,
    offset: ArrayView1<'_, f64>,
    s_list: &[BlockwisePenalty],
    opts: &ExternalOptimOptions,
    rho: &Array1<f64>,
    delta_rho: ArrayView1<'_, f64>,
) -> Result<(f64, f64), EstimationError>
where
    X: Into<DesignMatrix>,
{
    if !opts.family.is_gaussian_identity() {
        crate::bail_invalid_estim!(
            "evaluate_external_ift_residual_at_perturbed_rho currently supports GaussianIdentity"
                .to_string(),
        );
    }
    if opts.linear_constraints.is_some() {
        crate::bail_invalid_estim!(
            "evaluate_external_ift_residual_at_perturbed_rho does not support constrained fits"
                .to_string(),
        );
    }

    let specs: Vec<PenaltySpec> = s_list.iter().map(PenaltySpec::from_blockwise_ref).collect();
    let x = x.into();
    if let Some(message) = row_mismatch_message(y.len(), w.len(), x.nrows(), offset.len()) {
        crate::bail_invalid_estim!("{}", message);
    }

    let p = x.ncols();
    validate_penalty_specs(&specs, p, "evaluate_external_ift_residual_at_perturbed_rho")?;
    let (canonical, active_nullspace_dims) = gam_terms::construction::canonicalize_penalty_specs(
        &specs,
        &opts.nullspace_dims,
        p,
        "evaluate_external_ift_residual_at_perturbed_rho",
    )?;
    if rho.len() != active_nullspace_dims.len() {
        crate::bail_invalid_estim!(
            "rho dimension mismatch: rho_dim={}, active_penalties={}",
            rho.len(),
            active_nullspace_dims.len()
        );
    }
    if delta_rho.len() != rho.len() {
        crate::bail_invalid_estim!(
            "delta_rho dimension mismatch: delta_dim={}, rho_dim={}",
            delta_rho.len(),
            rho.len()
        );
    }

    let mut tight_opts = opts.clone();
    tight_opts.tol = 1e-12;
    let (cfg, _) = resolved_external_config(&tight_opts)?;

    let y_o = y.to_owned();
    let w_o = w.to_owned();
    let offset_o = offset.to_owned();
    let conditioning = ParametricColumnConditioning::infer_from_penalty_specs(&x, &specs);
    let x_fit = conditioning.apply_to_design(&x);
    let fit_linear_constraints =
        conditioning.transform_linear_constraints_to_internal(tight_opts.linear_constraints);

    let mut reml_state = RemlState::newwith_offset(
        y_o.view(),
        x_fit.clone(),
        w_o.view(),
        offset_o.view(),
        canonical.clone(),
        p,
        &cfg,
        Some(active_nullspace_dims),
        None,
        fit_linear_constraints,
    )?;
    reml_state.set_penalty_shrinkage_floor(tight_opts.penalty_shrinkage_floor);
    reml_state.set_rho_prior(tight_opts.rho_prior.clone());
    reml_state.set_link_states(
        cfg.link_kind.mixture_state().cloned(),
        cfg.link_kind.sas_state().copied(),
    );

    reml_state.compute_gradient(rho)?;
    let beta_hat = reml_state
        .warm_start_beta
        .read()
        .unwrap()
        .as_ref()
        .map(|beta| beta.0.clone())
        .ok_or_else(|| {
            EstimationError::InvalidInput(
                "PIRLS solve did not populate the warm-start beta cache".to_string(),
            )
        })?;

    let rho_perturbed = rho + &delta_rho.to_owned();
    let beta_pred = reml_state
        .predict_warm_start_beta_ift_with_outcome(&rho_perturbed)
        .map(|(beta, _)| beta.as_ref().clone())
        .ok_or_else(|| {
            EstimationError::InvalidInput(
                "IFT warm-start predictor rejected the perturbed rho".to_string(),
            )
        })?;

    let ift_residual = gaussian_identity_inner_residual_norm(
        y_o.view(),
        w_o.view(),
        &x_fit,
        offset_o.view(),
        &canonical,
        &rho_perturbed,
        &beta_pred,
    )?;
    let flat_residual = gaussian_identity_inner_residual_norm(
        y_o.view(),
        w_o.view(),
        &x_fit,
        offset_o.view(),
        &canonical,
        &rho_perturbed,
        &beta_hat,
    )?;

    Ok((ift_residual, flat_residual))
}

/// Evaluate the external cost and report the stabilization ridge used.
/// This is a diagnostic helper for tests that need to detect ridge jitter.
pub fn evaluate_externalcost_andridge<X>(
    y: ArrayView1<'_, f64>,
    w: ArrayView1<'_, f64>,
    x: X,
    offset: ArrayView1<'_, f64>,
    s_list: &[BlockwisePenalty],
    opts: &ExternalOptimOptions,
    rho: &Array1<f64>,
) -> Result<(f64, f64), EstimationError>
where
    X: Into<DesignMatrix>,
{
    let specs: Vec<PenaltySpec> = s_list.iter().map(PenaltySpec::from_blockwise_ref).collect();
    let x = x.into();
    if let Some(message) = row_mismatch_message(y.len(), w.len(), x.nrows(), offset.len()) {
        crate::bail_invalid_estim!("{}", message);
    }

    let p = x.ncols();
    validate_penalty_specs(&specs, p, "evaluate_externalcost_andridge")?;
    let (canonical, active_nullspace_dims) = gam_terms::construction::canonicalize_penalty_specs(
        &specs,
        &opts.nullspace_dims,
        p,
        "evaluate_externalcost_andridge",
    )?;
    if rho.len() != active_nullspace_dims.len() {
        crate::bail_invalid_estim!(
            "rho dimension mismatch: rho_dim={}, active_penalties={}",
            rho.len(),
            active_nullspace_dims.len()
        );
    }

    let (cfg, _) = resolved_external_config(opts)?;

    let y_o = y.to_owned();
    let w_o = w.to_owned();
    let offset_o = offset.to_owned();
    let conditioning = ParametricColumnConditioning::infer_from_penalty_specs(&x, &specs);
    let x_fit = conditioning.apply_to_design(&x);
    let fit_linear_constraints =
        conditioning.transform_linear_constraints_to_internal(opts.linear_constraints.clone());

    let mut reml_state = RemlState::newwith_offset(
        y_o.view(),
        x_fit,
        w_o.view(),
        offset_o.view(),
        canonical,
        p,
        &cfg,
        Some(active_nullspace_dims),
        None,
        fit_linear_constraints,
    )?;
    reml_state.set_penalty_shrinkage_floor(opts.penalty_shrinkage_floor);
    reml_state.set_rho_prior(opts.rho_prior.clone());
    reml_state.set_link_states(
        cfg.link_kind.mixture_state().cloned(),
        cfg.link_kind.sas_state().copied(),
    );

    let cost = reml_state.compute_cost(rho)?;
    let ridge = reml_state.last_ridge_used().unwrap_or(0.0);
    Ok((cost, ridge))
}
