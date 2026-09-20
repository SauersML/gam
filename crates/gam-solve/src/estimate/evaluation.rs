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
    let b = sas_log_delta_bound();
    let u = raw_log_delta / b;
    let t = u.tanh();
    // `−w·ln(1 − t²) = 2w·ln cosh u`, in the form that never forms `1 − t²`
    // (see `ln_cosh`); the former `(1 − t²).max(1e-12)` capped the barrier at
    // `27.6·w` once `tanh` had rounded to `±1` (#2469).
    let cost = 2.0 * w * ln_cosh(u);
    // d/draw[-w log(1-t^2)] = (2w/B) * t.
    let grad = (2.0 * w / b) * t;
    (cost, grad)
}

/// `ln cosh u` without forming `cosh u` (overflows past `|u| ≈ 710`) or
/// `1 − tanh²u` (cancels to exactly zero past `|u| ≈ 19`):
/// `ln cosh u = |u| + ln(1 + e^{−2|u|}) − ln 2`, exact for every finite `u`.
#[inline]
fn ln_cosh(u: f64) -> f64 {
    let a = u.abs();
    a + (-2.0 * a).exp().ln_1p() - std::f64::consts::LN_2
}

#[inline]
pub(crate) fn sas_epsilon_bound() -> f64 {
    // Fixed smooth bound on raw SAS epsilon during outer optimization.
    8.0
}

#[inline]
pub(crate) fn sas_effective_epsilon(raw_epsilon: f64) -> (f64, f64) {
    let bound = sas_epsilon_bound();
    let t = (raw_epsilon / bound).tanh();
    let epsilon = bound * t;
    let d_epsilon_d_raw = 1.0 - t * t;
    (epsilon, d_epsilon_d_raw)
}

#[inline]
pub(crate) fn sas_effective_epsilon_second(raw_epsilon: f64) -> (f64, f64, f64) {
    let bound = sas_epsilon_bound();
    let t = (raw_epsilon / bound).tanh();
    let first = 1.0 - t * t;
    let second = -2.0 * t * first / bound;
    (bound * t, first, second)
}

/// The raw-ε interval the `sas_epsilon_bound()·tanh(raw/bound)` chart resolves
/// (#2902 row 8). The chart's slope `sech²(raw/bound)` falls to `√ε` at
/// `|raw| = bound·acosh(ε^{−1/4})`; past that a unit raw step moves ε by less
/// than `√ε`, the resolution every derived ρ-domain edge is placed at
/// ([`log_gradient_resolution`](crate::estimate::rho_domain::log_gradient_resolution)).
pub(crate) fn sas_epsilon_domain() -> (f64, f64) {
    let bound = sas_epsilon_bound();
    let edge = bound
        * (-0.5 * crate::estimate::rho_domain::log_gradient_resolution())
            .exp()
            .acosh();
    (-edge, edge)
}

#[inline]
pub(crate) fn sas_log_delta_edge_barriercostgradhess(raw_log_delta: f64) -> (f64, f64, f64) {
    let w = sas_log_delta_edge_barrierweight();
    if w <= 0.0 || !raw_log_delta.is_finite() {
        return (0.0, 0.0, 0.0);
    }
    let b = sas_log_delta_bound();
    let u = raw_log_delta / b;
    let t = u.tanh();
    let ln_cosh_u = ln_cosh(u);
    let cost = 2.0 * w * ln_cosh_u;
    let grad = (2.0 * w / b) * t;
    // `1 − t² = sech²u = e^{−2·ln cosh u}`: underflows to an honest zero far
    // past the bound instead of being floored.
    let one_minus_t2 = (-2.0 * ln_cosh_u).exp();
    let hess = (2.0 * w / (b * b)) * one_minus_t2;
    (cost, grad, hess)
}

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

    let (cfg, _) = resolved_external_config(opts)?;

    let y_o = y.to_owned();
    let w_o = w.to_owned();
    let offset_o = offset.to_owned();
    let conditioning = ParametricColumnConditioning::infer_from_penalty_specs(&x, &specs);
    let x_fit = conditioning.apply_to_design(&x);
    let fit_linear_constraints =
        conditioning.transform_linear_constraints_to_internal(opts.linear_constraints.clone());

    // Certify binomial separation before any inner solve, as the fitting entry
    // does: a separated unpenalized design has no finite mode (#2469).
    crate::estimate::prefit::reject_prefit_binomial_separation(&cfg, y, w, &x_fit, &canonical)?;
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

#[cfg(test)]
mod sas_epsilon_domain_tests {
    use super::*;

    /// #2902 row 8: the outer box on raw SAS ε ends where its tanh chart's slope
    /// falls to `√ε`, so the search box is the range the chart resolves.
    #[test]
    fn the_sas_epsilon_domain_ends_where_the_chart_slope_falls_to_root_epsilon_2902() {
        let root_epsilon = f64::EPSILON.sqrt();
        let (lower, upper) = sas_epsilon_domain();
        assert_eq!(lower, -upper);
        let slope = |raw: f64| sas_effective_epsilon(raw).1;
        let edge = slope(upper);
        assert!(
            (edge - root_epsilon).abs() <= 1.0e-6 * root_epsilon,
            "the chart slope at the domain edge must be √ε, got {edge:.6e}"
        );
        assert!(
            slope(0.99 * upper) > root_epsilon && slope(1.01 * upper) < root_epsilon,
            "the chart slope must cross √ε at the domain edge"
        );
    }
}
