use super::*;

pub(crate) fn materialize_link_outer_hessian(
    hessian: gam_problem::HessianValue,
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
    with_external_reml_state(
        y,
        w,
        x,
        offset,
        s_list,
        opts,
        rho,
        "evaluate_externalgradient",
        |state| state.compute_gradient(rho),
    )
}

/// Evaluate the external outer REML/LAML cost at `rho`.
pub fn evaluate_externalcost<X>(
    y: ArrayView1<'_, f64>,
    w: ArrayView1<'_, f64>,
    x: X,
    offset: ArrayView1<'_, f64>,
    s_list: &[BlockwisePenalty],
    opts: &ExternalOptimOptions,
    rho: &Array1<f64>,
) -> Result<f64, EstimationError>
where
    X: Into<DesignMatrix>,
{
    with_external_reml_state(
        y,
        w,
        x,
        offset,
        s_list,
        opts,
        rho,
        "evaluate_externalcost",
        |state| state.compute_cost(rho),
    )
}

/// Evaluate the exact analytic outer ρ-Hessian of the external REML/LAML
/// criterion at `rho`, the Hessian the outer search and the smoothing-corrected
/// covariance consume. A criterion that declares no analytic Hessian is a typed
/// error naming why.
pub fn evaluate_externalhessian<X>(
    y: ArrayView1<'_, f64>,
    w: ArrayView1<'_, f64>,
    x: X,
    offset: ArrayView1<'_, f64>,
    s_list: &[BlockwisePenalty],
    opts: &ExternalOptimOptions,
    rho: &Array1<f64>,
) -> Result<Array2<f64>, EstimationError>
where
    X: Into<DesignMatrix>,
{
    with_external_reml_state(
        y,
        w,
        x,
        offset,
        s_list,
        opts,
        rho,
        "evaluate_externalhessian",
        |state| state.compute_lamlhessian_consistent(rho),
    )
}

/// Build the external REML state exactly as the fitting entry does and run
/// `evaluate` on it at `rho`.
fn with_external_reml_state<X, R>(
    y: ArrayView1<'_, f64>,
    w: ArrayView1<'_, f64>,
    x: X,
    offset: ArrayView1<'_, f64>,
    s_list: &[BlockwisePenalty],
    opts: &ExternalOptimOptions,
    rho: &Array1<f64>,
    context: &str,
    evaluate: impl FnOnce(&RemlState<'_>) -> Result<R, EstimationError>,
) -> Result<R, EstimationError>
where
    X: Into<DesignMatrix>,
{
    let specs: Vec<PenaltySpec> = s_list.iter().map(PenaltySpec::from_blockwise_ref).collect();
    let x = x.into();
    if let Some(message) = row_mismatch_message(y.len(), w.len(), x.nrows(), offset.len()) {
        crate::bail_invalid_estim!("{}", message);
    }

    let p = x.ncols();
    validate_penalty_specs(&specs, p, context)?;
    let (canonical, active_nullspace_dims) = gam_terms::construction::canonicalize_penalty_specs(
        &specs,
        &opts.nullspace_dims,
        p,
        context,
    )?;
    if rho.len() != active_nullspace_dims.len() {
        crate::bail_invalid_estim!(
            "rho dimension mismatch: rho_dim={}, active_penalties={}",
            rho.len(),
            active_nullspace_dims.len()
        );
    }

    let (mut cfg, _) = resolved_external_config(opts)?;

    let y_o = y.to_owned();
    let w_o = w.to_owned();
    let offset_o = offset.to_owned();
    let conditioning = ParametricColumnConditioning::infer_from_penalty_specs(&x, &specs);
    let x_fit = conditioning.apply_to_design(&x);
    let fit_linear_constraints =
        conditioning.transform_linear_constraints_to_internal(opts.linear_constraints.clone());

    // Decide the binomial prior before any inner solve, as the fitting entry
    // does: a separated unpenalized design has no finite flat-prior mode
    // (#2469), so the objective evaluated here is the Jeffreys-prior one the
    // fit optimizes on that design (#3129).
    crate::estimate::prefit::arm_jeffreys_on_prefit_binomial_separation(
        &mut cfg, opts, y, w, &x_fit, &canonical,
    )?;
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
    reml_state.set_rho_prior(opts.rho_prior.clone());
    reml_state.set_link_states(
        cfg.link_kind.mixture_state().cloned(),
        cfg.link_kind.sas_state().copied(),
    );

    evaluate(&reml_state)
}
