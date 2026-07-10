/// Preserve the core SAE fit error taxonomy at the Python boundary.
///
/// In particular, a completed-but-nonstationary outer search is not an input
/// error: it is a typed REML convergence failure whose full `OuterResult`
/// remains the resume/evidence payload.  Keep that distinction instead of
/// flattening `SaeFitError` through `Display` into `GamError`.
fn sae_fit_error_to_pyerr(
    py: Python<'_>,
    err: gam::terms::sae::manifold::SaeFitError,
) -> PyErr {
    use gam::terms::sae::manifold::SaeFitError;

    let message = err.to_string();
    match err {
        SaeFitError::Fit(_) => py_value_error(message),
        SaeFitError::OuterRun { stage, source } => {
            let exc = estimation_error_to_pyerr(source);
            let bound = exc.value(py);
            if let Err(attach_err) = bound.setattr("stage", stage.to_string()) {
                attach_err.write_unraisable(py, Some(&bound));
            }
            exc
        }
        SaeFitError::OuterDidNotConverge { stage, result } => {
            let exc = RemlConvergenceError::new_err(message);
            let bound = exc.value(py);
            let attach_result: PyResult<()> = (|| {
                bound.setattr("stage", stage.to_string())?;
                bound.setattr("converged", false)?;
                bound.setattr("rho_checkpoint", result.rho.clone().into_pyarray(py))?;
                bound.setattr("final_value", result.final_value)?;
                bound.setattr("iterations", result.iterations)?;
                match result.final_grad_norm {
                    Some(value) => bound.setattr("final_grad_norm", value)?,
                    None => bound.setattr("final_grad_norm", py.None())?,
                }
                match result.final_gradient.as_ref() {
                    Some(value) => {
                        bound.setattr("final_gradient", value.clone().into_pyarray(py))?
                    }
                    None => bound.setattr("final_gradient", py.None())?,
                }
                match result.final_hessian.as_ref() {
                    Some(value) => {
                        bound.setattr("final_hessian", value.clone().into_pyarray(py))?
                    }
                    None => bound.setattr("final_hessian", py.None())?,
                }
                bound.setattr("plan", result.plan_used.to_string())?;
                bound.setattr(
                    "operator_stop_reason",
                    result
                        .operator_stop_reason
                        .as_ref()
                        .map(|reason| format!("{reason:?}")),
                )?;
                match result.operator_trust_radius {
                    Some(value) => bound.setattr("operator_trust_radius", value)?,
                    None => bound.setattr("operator_trust_radius", py.None())?,
                }
                bound.setattr(
                    "criterion_certificate",
                    result
                        .criterion_certificate
                        .as_ref()
                        .map(|certificate| format!("{certificate:?}")),
                )?;
                Ok(())
            })();
            if let Err(attach_err) = attach_result {
                attach_err.write_unraisable(py, Some(&bound));
            }
            exc
        }
    }
}

#[pyfunction(signature = (
    t,
    y,
    n_obs,
    latent_dim,
    centers,
    penalty,
    m = 2,
    weights = None,
    fisher_w = None,
    init_lambda = None,
    aux_u = None,
    aux_family = "ridge".to_string(),
    aux_strength = None,
    dim_selection_log_precision = None,
    basis_kind = "duchon".to_string(),
    tensor_knots_concat = None,
    tensor_knot_offsets = None,
    tensor_degrees = None,
    manifold = "euclidean".to_string(),
    sigma_eff_mode = "profiled".to_string(),
    max_iter = 200,
    grad_tol = 1.0e-8,
    stationarity_reference = None,
    trust_radius = 1.0,
    max_radius = 1.0e6,
    n_restarts = 1,
    restart_scale = 0.25,
    seed = 0,
    init = "spectral".to_string(),
    seed_neighbors = 10,
))]
fn gaussian_reml_optimize_latent<'py>(
    py: Python<'py>,
    t: PyReadonlyArray1<'py, f64>,
    y: PyReadonlyArray2<'py, f64>,
    n_obs: usize,
    latent_dim: usize,
    centers: PyReadonlyArray2<'py, f64>,
    penalty: PyReadonlyArray2<'py, f64>,
    m: usize,
    weights: Option<PyReadonlyArray1<'py, f64>>,
    fisher_w: Option<PyReadonlyArray3<'py, f64>>,
    init_lambda: Option<f64>,
    aux_u: Option<PyReadonlyArray2<'py, f64>>,
    aux_family: String,
    aux_strength: Option<f64>,
    dim_selection_log_precision: Option<PyReadonlyArray1<'py, f64>>,
    basis_kind: String,
    tensor_knots_concat: Option<PyReadonlyArray1<'py, f64>>,
    tensor_knot_offsets: Option<Vec<usize>>,
    tensor_degrees: Option<Vec<usize>>,
    manifold: String,
    sigma_eff_mode: String,
    max_iter: usize,
    grad_tol: f64,
    stationarity_reference: Option<f64>,
    trust_radius: f64,
    max_radius: f64,
    n_restarts: usize,
    restart_scale: f64,
    seed: u64,
    init: String,
    seed_neighbors: usize,
) -> PyResult<Py<PyDict>> {
    use rand::SeedableRng;
    use rand_distr::Distribution;

    let family = match aux_family.to_ascii_lowercase().as_str() {
        "ridge" => AuxPriorFamily::Ridge,
        "linear" => AuxPriorFamily::Linear,
        other => {
            return Err(py_value_error(format!(
                "aux_family must be 'ridge' or 'linear'; got {other:?}"
            )));
        }
    };
    let sigma_eff_mode = SigmaEffMode::parse(&sigma_eff_mode).map_err(py_value_error)?;
    if n_restarts == 0 {
        return Err(py_value_error("n_restarts must be at least 1".to_string()));
    }
    if !(grad_tol.is_finite() && grad_tol > 0.0) {
        return Err(py_value_error(format!(
            "grad_tol must be finite and positive; got {grad_tol}"
        )));
    }
    if let Some(reference) = stationarity_reference {
        if !(reference.is_finite() && reference >= 0.0) {
            return Err(py_value_error(format!(
                "stationarity_reference must be finite and non-negative; got {reference}"
            )));
        }
    }
    if !(trust_radius.is_finite() && trust_radius > 0.0) {
        return Err(py_value_error(format!(
            "trust_radius must be finite and positive; got {trust_radius}"
        )));
    }
    if !(max_radius.is_finite() && max_radius >= trust_radius) {
        return Err(py_value_error(format!(
            "max_radius must be finite and at least trust_radius ({trust_radius}); got {max_radius}"
        )));
    }
    if !(restart_scale.is_finite() && restart_scale > 0.0) {
        return Err(py_value_error(format!(
            "restart_scale must be finite and positive; got {restart_scale}"
        )));
    }
    let expected = n_obs
        .checked_mul(latent_dim)
        .ok_or_else(|| py_value_error("n_obs * latent_dim overflows usize".to_string()))?;
    let t_values = t.as_array().to_owned();
    if t_values.len() != expected {
        return Err(py_value_error(format!(
            "t length {} must equal n_obs * latent_dim = {expected}",
            t_values.len()
        )));
    }
    // Choose the base start for restart 0 (further restarts perturb it). A
    // spectral seed escapes the random-init local optimum that leaves the outer
    // optimizer stuck (#627); `"caller"` keeps the passed-in `t` unchanged for
    // callers that already have a good warm start or want a pure local solve.
    let base_start = match init.to_ascii_lowercase().as_str() {
        "caller" | "warm" | "passthrough" => t_values.clone(),
        "spectral" | "laplacian" | "eigenmap" => latent_spectral_seed_start(
            y.as_array(),
            centers.as_array(),
            &manifold,
            n_obs,
            latent_dim,
            seed_neighbors,
            t_values.view(),
        )
        .map_err(py_value_error)?,
        other => {
            return Err(py_value_error(format!(
                "init must be 'spectral' or 'caller'; got {other:?}"
            )));
        }
    };
    let weight_values = weights.as_ref().map(|w| w.as_array().to_owned());
    let fisher_values = fisher_w.as_ref().map(|w| w.as_array().to_owned());
    let effective_weights = latent_scalar_weights_with_fisher(
        n_obs,
        weight_values.as_ref().map(|w| w.view()),
        fisher_values.as_ref().map(|w| w.view()),
    )
    .map_err(py_value_error)?;
    let dim_selection_values = dim_selection_log_precision
        .as_ref()
        .map(|a| a.as_array().to_owned());
    let tensor_knots_values = tensor_knots_concat
        .as_ref()
        .map(|a| a.as_array().to_owned());

    // Derive the periodic chart descriptor ONCE from the manifold; it drives the
    // periodic Duchon decoder both during optimization (`try_value_and_grad`) and
    // for the final reported fit so the OPTIMIZED basis == the FINAL basis. Only
    // the Duchon decoder consumes it; matern/sphere/tensor branches ignore it.
    let latent_periodic = latent_manifold_periodic_descriptor(&manifold, latent_dim);
    let problem = LatentOuterProblem {
        y: y.as_array().to_owned(),
        centers: centers.as_array().to_owned(),
        penalty: penalty.as_array().to_owned(),
        weights: effective_weights,
        aux_u: aux_u.as_ref().map(|a| a.as_array().to_owned()),
        dim_selection: dim_selection_values,
        family,
        aux_strength,
        init_lambda,
        sigma_eff_mode,
        n_obs,
        latent_dim,
        m,
        basis_kind,
        tensor_knots: tensor_knots_values,
        tensor_knot_offsets,
        tensor_degrees,
        periodic: latent_periodic,
    };

    let manifold_box =
        build_latent_outer_manifold(&manifold, n_obs, latent_dim).map_err(py_value_error)?;
    let trust_region = gam::geometry::RiemannianTrustRegion {
        radius: trust_radius,
        max_radius,
        max_iter,
        grad_tol,
    };

    // Restart 0 starts from `base_start` (the spectral seed, or the caller's `t`
    // when `init="caller"`); further restarts perturb it in the tangent space and
    // retract back onto the manifold, then we keep the lowest-score latent.
    let (best_t, best_value, best_start_grad_norm, best_restart) = py
        .detach(|| -> Result<(Array1<f64>, f64, f64, usize), String> {
            let manifold_ref: &dyn gam::geometry::RiemannianManifold = manifold_box.as_ref();
            let normal =
                rand_distr::Normal::new(0.0, restart_scale).map_err(|err| err.to_string())?;
            let mut rng = rand::rngs::StdRng::seed_from_u64(seed);
            let mut best: Option<(Array1<f64>, f64, f64, usize)> = None;
            for restart in 0..n_restarts {
                let start = if restart == 0 {
                    base_start.clone()
                } else {
                    let noise =
                        Array1::from_shape_fn(base_start.len(), |_| normal.sample(&mut rng));
                    let tangent = manifold_ref
                        .project_tangent(base_start.view(), noise.view())
                        .map_err(|err| err.to_string())?;
                    manifold_ref
                        .retract(base_start.view(), tangent.view())
                        .map_err(|err| err.to_string())?
                };
                // Every manifold accepted by `build_latent_outer_manifold`
                // carries the induced ambient metric. Its Riemannian gradient
                // is therefore the tangent projection and its norm is the
                // Euclidean norm in ambient coordinates. Record the scale at
                // THIS restart's initial point: the winning restart must be
                // certified against the same reference its optimizer used.
                let (_, start_gradient) = problem.value_and_grad(start.view(), true);
                let start_gradient = start_gradient.ok_or_else(|| {
                    format!("restart {restart} did not produce an initial latent gradient")
                })?;
                let start_projected = manifold_ref
                    .riemannian_gradient(start.view(), start_gradient.view())
                    .map_err(|err| err.to_string())?;
                let start_grad_norm = start_projected
                    .iter()
                    .map(|value| value * value)
                    .sum::<f64>()
                    .sqrt();
                let mut objective = LatentOuterObjective { problem: &problem };
                let optimized = trust_region
                    .minimize(manifold_ref, &mut objective, start.view())
                    .map_err(|err| err.to_string())?;
                let (value, _) = problem.value_and_grad(optimized.view(), false);
                let improved = best
                    .as_ref()
                    .map(|(_, incumbent, _, _)| value < *incumbent)
                    .unwrap_or(true);
                if improved {
                    best = Some((optimized, value, start_grad_norm, restart));
                }
            }
            best.ok_or_else(|| "no restart produced a latent".to_string())
        })
        .map_err(py_value_error)?;

    // Final gradient norm at the chosen latent, as a convergence diagnostic.
    // Report the PROJECTED (Riemannian) gradient — the quantity the trust region
    // actually tests against `grad_tol` (`g_norm` in optimizer.rs) — not the raw
    // ambient gradient. On the circle/torus the ambient gradient carries a
    // normal component the optimizer never sees; reporting it inflated the norm
    // and made `converged` disagree with the optimizer's own stopping test
    // (issue #879). On a Euclidean manifold the tangent projection is the
    // identity, so this leaves that path byte-identical.
    let (_, final_grad) = problem.value_and_grad(best_t.view(), true);
    let grad_t_norm = match final_grad.as_ref() {
        Some(gradient) => {
            let riemannian = manifold_box
                .as_ref()
                .riemannian_gradient(best_t.view(), gradient.view())
                .map_err(|err| py_value_error(err.to_string()))?;
            riemannian.iter().map(|value| value * value).sum::<f64>().sqrt()
        }
        None => f64::INFINITY,
    };
    // The winning restart carries its own initial-gradient scale. A resumed
    // solve may supply the original scale explicitly so the convergence test
    // remains the same test across process/wall boundaries instead of silently
    // renormalizing at the checkpoint.
    let grad0_norm = stationarity_reference.unwrap_or(best_start_grad_norm);
    // Latent spread: a genuine collapse (all rows retract to one latent
    // coordinate, the issue #876 failure mode) leaves `latent_t_std ≈ 0`, which
    // distinguishes it from a healthy fit whose latent gradient merely failed to
    // reach `grad_tol`.
    let latent_t_std = {
        let n = best_t.len();
        if n == 0 {
            0.0
        } else {
            let mean = best_t.iter().sum::<f64>() / n as f64;
            (best_t.iter().map(|&v| (v - mean) * (v - mean)).sum::<f64>() / n as f64).sqrt()
        }
    };

    // Relative-gradient stationarity measure for the profiled-scale latent
    // objective (issue #879). The latent objective is the *profiled* Gaussian
    // REML score `n·log σ̂²(t) + ½·log|Hλ| + …`. Near interpolation the profiled
    // scale `σ̂²` collapses toward zero, which steepens that `n·log σ̂²` term and
    // leaves the raw latent gradient `‖∇ₜ f‖` at an O(n) magnitude *even at a
    // genuine stationary point* (R²≈1). So the bare absolute test
    // `‖∇ₜ f‖ ≤ grad_tol` is mis-calibrated: it flags an excellent,
    // near-stationary fit as non-converged.
    //
    // We use the SAME shift-invariant relative-gradient test the optimizer's own
    // stopping rule uses (`relative_stationarity` in optimizer.rs, issue #954):
    //
    //   rel = ‖∇ₜ f(t̂)‖_g / max(‖∇ₜ f(t₀)‖_g, 1)
    //
    // where `t₀` is the initial iterate. The denominator carries the SAME
    // multiplicative scale `f` and `∇f` share (`f → c·f ⇒ ∇f → c·∇f`), so a
    // fixed `grad_tol` still reads as a *relative* tolerance — and, crucially,
    // because the profiled objective's gradient is O(n) at the seed too, the
    // O(n) magnitude divides out (#879). The `max(·, 1)` floor reduces this to
    // the absolute test `‖∇ₜ f‖ ≤ grad_tol` on a unit-scale objective.
    //
    // This REPLACES the earlier `‖∇ₜ f‖ · ‖t‖_typ / max(|f|, 1)`, which was
    // *not* shift-invariant: minimization is invariant under `f → f + C`
    // (minimizer, gradient, Hessian, model reduction all unchanged), yet the old
    // `max(|f|, 1)` denominator grows with an additive constant `C` and could
    // falsely certify a non-stationary latent as converged (issue #954). The
    // `‖t‖_typ` factor was also non-intrinsic — the ambient latent magnitude is
    // not chart/translation invariant on a manifold (the circle/torus charts
    // wrap to `[-π, π)`), so it does not belong in a Riemannian stationarity
    // test. Anchoring to `‖∇ₜ f(t₀)‖` is both shift-invariant (the gradient does
    // not depend on `C`) and scale-invariant.
    let grad_t_norm_scaled = latent_relative_stationarity(grad_t_norm, grad0_norm);
    if !(grad_t_norm_scaled.is_finite() && grad_t_norm_scaled <= grad_tol) {
        // SPEC 20 — a fit object must only ever come from a converged
        // optimization. The chosen latent failed the shift-invariant relative
        // stationarity test, so no fit is rebuilt or returned: the caller gets
        // a typed `RemlConvergenceError` carrying the available numerical
        // evidence plus the best latent as a one-dimensional resume checkpoint,
        // exactly matching the public API's `t` input.
        let err = RemlConvergenceError::new_err(format!(
            "gaussian_reml_optimize_latent did not reach latent stationarity: relative \
             gradient {grad_t_norm_scaled:.6e} did not satisfy grad_tol {grad_tol:.6e} with a \
             budget of {max_iter} iteration(s) x {n_restarts} restart(s) (projected gradient {grad_t_norm:.6e}, \
             seed gradient {grad0_norm:.6e}, objective {best_value:.9e}, latent spread \
             {latent_t_std:.6e}). No fit is minted from a non-converged optimization; \
             resume from the exception's `checkpoint_t` and \
             `checkpoint_stationarity_reference` attributes with `init=\"caller\"`, or loosen \
             grad_tol if this stationarity precision is not required."
        ));
        // Attach the structured evidence + checkpoint as instance attributes
        // (same best-effort pattern as `ColumnNotFoundError` in ffi_errors.rs):
        // the typed class + message remain the primary contract if a setattr
        // ever fails.
        let bound = err.value(py);
        let attach_result: PyResult<()> = (|| {
            bound.setattr("grad_t_norm", grad_t_norm)?;
            bound.setattr("grad_t_norm_init", grad0_norm)?;
            bound.setattr("grad_t_norm_scaled", grad_t_norm_scaled)?;
            bound.setattr("grad_tol", grad_tol)?;
            bound.setattr("latent_t_std", latent_t_std)?;
            bound.setattr("objective_value", best_value)?;
            bound.setattr("max_iter", max_iter)?;
            bound.setattr("n_restarts", n_restarts)?;
            bound.setattr("restart_index", best_restart)?;
            bound.setattr("init", init.as_str())?;
            bound.setattr("checkpoint_t", best_t.into_pyarray(py))?;
            bound.setattr("checkpoint_shape", (n_obs, latent_dim))?;
            bound.setattr("checkpoint_stationarity_reference", grad0_norm)?;
            Ok(())
        })();
        if let Err(attach_err) = attach_result {
            attach_err.write_unraisable(py, Some(&bound));
        }
        return Err(err);
    }

    // Rebuild the full fit dictionary at the converged latent so callers get the
    // identical schema [`gaussian_reml_fit_latent`] returns, then echo `t`. The
    // detached fit closure must be `'static`, so move owned copies in (the
    // problem's array buffers are no longer needed on this thread afterwards).
    let latent_payload = serde_json::json!({"t": {"name": "t", "n": n_obs, "d": latent_dim}});
    let LatentOuterProblem {
        y,
        centers,
        penalty,
        weights,
        aux_u,
        dim_selection,
        basis_kind,
        tensor_knots,
        tensor_knot_offsets,
        tensor_degrees,
        periodic: latent_periodic_final,
        ..
    } = problem;
    let best_t_for_fit = best_t.clone();
    // Retain the response for the #879 reconstruction-quality diagnostic; `y` is
    // moved into the `move` fit closure below.
    let y_for_diag = y.clone();
    let (fit, _design, aux_strength_state) =
        detach_py_result(py, "gaussian_reml_optimize_latent", move || {
            let registry = build_analytic_penalty_registry_from_json(Some(&latent_payload), None)?;
            gaussian_reml_fit_latent_impl(
                best_t_for_fit.view(),
                y.view(),
                n_obs,
                latent_dim,
                centers.view(),
                m,
                &basis_kind,
                tensor_knots.as_ref().map(|a| a.view()),
                tensor_knot_offsets.as_deref(),
                tensor_degrees.as_deref(),
                penalty.view(),
                weights.as_ref().map(|w| w.view()),
                init_lambda,
                aux_u.as_ref().map(|a| a.view()),
                family,
                aux_strength,
                dim_selection.as_ref().map(|a| a.view()),
                Some(&registry),
                // Final reported fit MUST use the SAME manifold-derived periodic
                // Duchon decoder the optimizer used (so OPTIMIZED basis == FINAL
                // basis); `None` for Euclidean / sphere keeps those byte-identical.
                latent_periodic_final.as_deref(),
            )
        })?;

    // Reconstruction quality of the decoder against the response, reported next
    // to `converged` so model selection can distinguish a good decoder fit whose
    // latent gradient simply did not reach `grad_tol` (near-interpolation the
    // profiled scale stiffens the latent objective, so ‖∇ₜ‖ stays O(n) even at
    // R²≈1 — issue #879) from a genuinely failed/collapsed fit. Computed over all
    // (row, output) entries of the response and the fitted decoder image.
    let (residual_ss, total_ss) = {
        let fitted = &fit.fitted;
        let mean = if y_for_diag.is_empty() {
            0.0
        } else {
            y_for_diag.iter().sum::<f64>() / y_for_diag.len() as f64
        };
        let mut rss = 0.0;
        let mut tss = 0.0;
        for (&yi, &fi) in y_for_diag.iter().zip(fitted.iter()) {
            rss += (yi - fi) * (yi - fi);
            tss += (yi - mean) * (yi - mean);
        }
        (rss, tss)
    };
    let response_residual_norm = residual_ss.sqrt();
    // R² = 1 − RSS/TSS; a degenerate (constant) response has TSS = 0, in which
    // case a zero residual is a perfect fit (1.0) and any residual is reported as
    // 0.0 rather than a spurious −∞.
    let response_r2 = if total_ss > 0.0 {
        1.0 - residual_ss / total_ss
    } else if residual_ss == 0.0 {
        1.0
    } else {
        0.0
    };

    let out = PyDict::new(py);
    set_ok_gaussian_reml_items(py, &out, fit)?;
    set_aux_strength_items(py, &out, aux_strength_state)?;
    let t_matrix = best_t
        .clone()
        .into_shape_with_order((n_obs, latent_dim))
        .map_err(shape_error_to_pyerr)?;
    out.set_item("t", t_matrix.clone().into_pyarray(py))?;
    out.set_item("latent", t_matrix.into_pyarray(py))?;
    out.set_item("t_flat", best_t.into_pyarray(py))?;
    out.set_item("grad_t_norm", grad_t_norm)?;
    out.set_item("grad_t_norm_init", grad0_norm)?;
    out.set_item("grad_t_norm_scaled", grad_t_norm_scaled)?;
    out.set_item("latent_t_std", latent_t_std)?;
    out.set_item("response_r2", response_r2)?;
    out.set_item("response_residual_norm", response_residual_norm)?;
    out.set_item("objective_value", best_value)?;
    out.set_item("n_restarts", n_restarts)?;
    out.set_item("init", init)?;
    Ok(out.unbind())
}

fn latent_glm_family_from_str(
    value: &str,
    tweedie_p: f64,
    negbin_theta: f64,
    beta_phi: f64,
) -> Result<LikelihoodSpec, String> {
    match value.to_ascii_lowercase().replace('_', "-").as_str() {
        "gaussian" | "gaussian-identity" => Ok(LikelihoodSpec::new(
            ResponseFamily::Gaussian,
            InverseLink::Standard(StandardLink::Identity),
        )),
        "binomial" | "binomial-logit" | "logistic" => Ok(LikelihoodSpec::new(
            ResponseFamily::Binomial,
            InverseLink::Standard(StandardLink::Logit),
        )),
        "binomial-probit" | "probit" => Ok(LikelihoodSpec::new(
            ResponseFamily::Binomial,
            InverseLink::Standard(StandardLink::Probit),
        )),
        "binomial-cloglog" | "cloglog" => Ok(LikelihoodSpec::new(
            ResponseFamily::Binomial,
            InverseLink::Standard(StandardLink::CLogLog),
        )),
        "poisson" | "poisson-log" => Ok(LikelihoodSpec::new(
            ResponseFamily::Poisson,
            InverseLink::Standard(StandardLink::Log),
        )),
        "tweedie" | "tweedie-log" => {
            // The compound-Poisson-Gamma Tweedie variance power must lie
            // strictly in (1, 2); outside that range the unit-deviance and
            // working-response formulas are undefined and the downstream
            // solver bails with a NaN deviance. Validate eagerly here (mirror
            // of the negbin/beta nuisance-parameter checks below) so the user
            // sees an actionable error instead of an opaque fit failure.
            if !gam::types::is_valid_tweedie_power(tweedie_p) {
                return Err(format!(
                    "tweedie_p must be finite and strictly between 1 and 2; got {tweedie_p}"
                ));
            }
            Ok(LikelihoodSpec::new(
                ResponseFamily::Tweedie { p: tweedie_p },
                InverseLink::Standard(StandardLink::Log),
            ))
        }
        "negbin" | "negbin-log" | "negative-binomial" | "negative-binomial-log" => {
            if !(negbin_theta.is_finite() && negbin_theta > 0.0) {
                return Err(format!(
                    "negbin_theta must be finite and > 0; got {negbin_theta}"
                ));
            }
            Ok(LikelihoodSpec::new(
                ResponseFamily::NegativeBinomial {
                    theta: negbin_theta,
                    theta_fixed: false,
                },
                InverseLink::Standard(StandardLink::Log),
            ))
        }
        "beta" | "beta-logit" | "beta-regression" | "beta-regression-logit" => {
            if !(beta_phi.is_finite() && beta_phi > 0.0) {
                return Err(format!("beta_phi must be finite and > 0; got {beta_phi}"));
            }
            Ok(LikelihoodSpec::new(
                ResponseFamily::Beta { phi: beta_phi },
                InverseLink::Standard(StandardLink::Logit),
            ))
        }
        "gamma-log" => Ok(LikelihoodSpec::new(
            ResponseFamily::Gamma,
            InverseLink::Standard(StandardLink::Log),
        )),
        other => Err(format!(
            "unsupported latent GLM family {other:?}; supported families are gaussian-identity, binomial-logit, binomial-probit, binomial-cloglog, poisson-log, tweedie-log, negbin-log, beta-regression-logit, gamma-log"
        )),
    }
}

fn glm_reml_fit_latent_impl(
    t_flat: ArrayView1<'_, f64>,
    y: ArrayView2<'_, f64>,
    n_obs: usize,
    latent_dim: usize,
    centers: ArrayView2<'_, f64>,
    m: usize,
    penalty: ArrayView2<'_, f64>,
    weights: Option<ArrayView1<'_, f64>>,
    init_lambda: Option<f64>,
    family: LikelihoodSpec,
    aux_u: Option<ArrayView2<'_, f64>>,
    aux_family: AuxPriorFamily,
    aux_strength: Option<f64>,
    dim_selection_precision: Option<ArrayView1<'_, f64>>,
    analytic_penalties: Option<&AnalyticPenaltyRegistry>,
) -> Result<
    (
        gam::solver::estimate::ExternalOptimResult,
        Array2<f64>,
        Array2<f64>,
        Option<LatentAuxStrengthState>,
    ),
    String,
> {
    if y.ncols() != 1 {
        return Err(format!(
            "glm_reml_fit_latent requires y with one column; got {}",
            y.ncols()
        ));
    }
    // GLM standalone latent fit: no manifold/chart concept here, so the latent
    // Duchon decoder stays the open Euclidean basis (byte-identical).
    let (design, t_mat) = build_latent_duchon_design(t_flat, n_obs, latent_dim, centers, m, None)?;
    if penalty.dim() != (design.ncols(), design.ncols()) {
        return Err(format!(
            "penalty shape mismatch: expected {}x{}, got {}x{}",
            design.ncols(),
            design.ncols(),
            penalty.nrows(),
            penalty.ncols()
        ));
    }
    let y_vec = y.column(0).to_owned();
    let weights_owned = match weights {
        Some(w) => gaussian_reml_weight_vector_local(n_obs, Some(w))?,
        None => Array1::ones(n_obs),
    };
    let offset = Array1::<f64>::zeros(n_obs);
    let penalty_block = BlockwisePenalty::new(0..design.ncols(), penalty.to_owned());
    let opts = ExternalOptimOptions {
        family,
        latent_cloglog: None,
        mixture_link: None,
        optimize_mixture: false,
        sas_link: None,
        optimize_sas: false,
        compute_inference: false,
        skip_rho_posterior_inference: false,
        max_iter: 100,
        tol: 1e-7,
        nullspace_dims: vec![0],
        linear_constraints: None,
        firth_bias_reduction: Some(false),
        penalty_shrinkage_floor: None,
        rho_prior: RhoPrior::Flat,
        persist_warm_start_disk: false,
        kronecker_penalty_system: None,
        kronecker_factored: None,
    };
    let heuristic_lambda = init_lambda.map(|lambda| [lambda]);
    let mut fit = optimize_external_designwith_heuristic_lambdas(
        y_vec.view(),
        weights_owned.view(),
        design.clone(),
        offset.view(),
        vec![penalty_block],
        heuristic_lambda.as_ref().map(|values| values.as_slice()),
        &opts,
    )
    .map_err(|err| err.to_string())?;
    let mut latent_prior_score = 0.0_f64;
    let mut aux_strength_state = None;
    if let Some(u_view) = aux_u {
        let stats = latent_aux_prior_stats(t_mat.view(), u_view, aux_family, aux_strength)?;
        latent_prior_score += stats.score;
        aux_strength_state = Some(stats.strength);
    }
    if let Some(log_prec) = dim_selection_precision {
        if log_prec.len() != latent_dim {
            return Err(format!(
                "dim_selection_log_precision length {} must equal latent_dim {}",
                log_prec.len(),
                latent_dim
            ));
        }
        for a in 0..latent_dim {
            let log_alpha = log_prec[a];
            let alpha = log_alpha.exp();
            if !(alpha.is_finite() && alpha > 0.0) {
                return Err(format!(
                    "dim_selection_log_precision[{a}] must exponentiate to a finite positive precision"
                ));
            }
            let mut sq = 0.0_f64;
            for n in 0..n_obs {
                let v = t_mat[[n, a]];
                sq += v * v;
            }
            latent_prior_score += 0.5 * alpha * sq - 0.5 * (n_obs as f64) * log_alpha;
        }
    }
    if let Some(registry) = analytic_penalties {
        latent_prior_score +=
            analytic_penalty_value_for_targets(registry, t_flat, Some(fit.beta.view()))?;
    }
    fit.reml_score += latent_prior_score;
    Ok((fit, design, t_mat, aux_strength_state))
}

fn set_ok_glm_latent_items<'py>(
    py: Python<'py>,
    out: &Bound<'py, PyDict>,
    fit: gam::solver::estimate::ExternalOptimResult,
    n_obs: usize,
    p: usize,
    aux_strength_state: Option<LatentAuxStrengthState>,
) -> PyResult<()> {
    let pirls = fit.artifacts.pirls.as_ref().ok_or_else(|| {
        py_value_error("latent GLM fit did not return PIRLS artifacts".to_string())
    })?;
    let lambda = fit.lambdas.get(0).copied().unwrap_or(f64::NAN);
    let rho = lambda.ln();
    let mut coefficients = Array2::<f64>::zeros((p, 1));
    for row in 0..p.min(fit.beta.len()) {
        coefficients[[row, 0]] = fit.beta[row];
    }
    let mut fitted = Array2::<f64>::zeros((n_obs, 1));
    for row in 0..n_obs.min(pirls.finalmu.len()) {
        fitted[[row, 0]] = pirls.finalmu[row];
    }
    out.set_item(
        "status",
        if fit.outer_converged {
            "ok"
        } else {
            "not_converged"
        },
    )?;
    out.set_item("lambda", lambda)?;
    out.set_item("rho", rho)?;
    out.set_item("reml_score", fit.reml_score)?;
    out.set_item("reml_grad_lambda", f64::NAN)?;
    out.set_item("reml_hess_lambda", f64::NAN)?;
    out.set_item("reml_grad_rho", f64::NAN)?;
    out.set_item("reml_hess_rho", f64::NAN)?;
    out.set_item("edf", pirls.edf)?;
    out.set_item("coefficients", coefficients.into_pyarray(py))?;
    out.set_item("fitted", fitted.into_pyarray(py))?;
    out.set_item(
        "sigma2",
        Array1::from_vec(vec![fit.standard_deviation * fit.standard_deviation]).into_pyarray(py),
    )?;
    out.set_item(
        "cache_penalty_eigenvalues",
        Array1::<f64>::zeros(0).into_pyarray(py),
    )?;
    out.set_item(
        "cache_eigenvectors",
        Array2::<f64>::zeros((0, 0)).into_pyarray(py),
    )?;
    out.set_item(
        "cache_coefficient_basis",
        Array2::<f64>::zeros((0, 0)).into_pyarray(py),
    )?;
    out.set_item("cache_xtwx_fingerprint", 0_u64)?;
    out.set_item("cache_penalty_fingerprint", 0_u64)?;
    out.set_item("cache_logdet_xtwx", f64::NAN)?;
    out.set_item("cache_logdet_penalty_positive", f64::NAN)?;
    out.set_item("cache_penalty_rank", 0_usize)?;
    out.set_item("cache_nullity", 0_usize)?;
    set_aux_strength_items(py, out, aux_strength_state)?;
    Ok(())
}

#[pyfunction(signature = (
    t,
    y,
    n_obs,
    latent_dim,
    centers,
    penalty,
    family,
    tweedie_p = 1.5,
    negbin_theta = 1.0,
    beta_phi = 1.0,
    m = 2,
    weights = None,
    fisher_w = None,
    init_lambda = None,
    aux_u = None,
    aux_family = "ridge".to_string(),
    aux_strength = None,
    dim_selection_log_precision = None,
    analytic_penalties = None,
))]
fn glm_reml_fit_latent<'py>(
    py: Python<'py>,
    t: PyReadonlyArray1<'py, f64>,
    y: PyReadonlyArray2<'py, f64>,
    n_obs: usize,
    latent_dim: usize,
    centers: PyReadonlyArray2<'py, f64>,
    penalty: PyReadonlyArray2<'py, f64>,
    family: String,
    tweedie_p: f64,
    negbin_theta: f64,
    beta_phi: f64,
    m: usize,
    weights: Option<PyReadonlyArray1<'py, f64>>,
    fisher_w: Option<PyReadonlyArray3<'py, f64>>,
    init_lambda: Option<f64>,
    aux_u: Option<PyReadonlyArray2<'py, f64>>,
    aux_family: String,
    aux_strength: Option<f64>,
    dim_selection_log_precision: Option<PyReadonlyArray1<'py, f64>>,
    analytic_penalties: Option<String>,
) -> PyResult<Py<PyDict>> {
    let analytic_penalties: Option<serde_json::Value> = match analytic_penalties {
        Some(s) => Some(serde_json::from_str(&s).map_err(serde_json_error_to_pyerr)?),
        None => None,
    };
    let latent_payload = serde_json::json!({"t": {"name": "t", "n": n_obs, "d": latent_dim}});
    let registry = build_analytic_penalty_registry_from_json(
        Some(&latent_payload),
        analytic_penalties.as_ref(),
    )
    .map_err(py_value_error)?;
    let family_name = family.clone();
    let family_normalized = family_name.to_ascii_lowercase().replace('_', "-");
    let aux_family = match aux_family.to_ascii_lowercase().as_str() {
        "ridge" => AuxPriorFamily::Ridge,
        "linear" => AuxPriorFamily::Linear,
        other => {
            return Err(py_value_error(format!(
                "aux_family must be 'ridge' or 'linear'; got {other:?}"
            )));
        }
    };
    let t_values = t.as_array().to_owned();
    let y_values = y.as_array().to_owned();
    let centers_values = centers.as_array().to_owned();
    let penalty_values = penalty.as_array().to_owned();
    let weight_values = weights.as_ref().map(|w| w.as_array().to_owned());
    let fisher_values = fisher_w.as_ref().map(|w| w.as_array().to_owned());
    let aux_u_values = aux_u.as_ref().map(|a| a.as_array().to_owned());
    let dim_selection_values = dim_selection_log_precision
        .as_ref()
        .map(|a| a.as_array().to_owned());
    if y_values.ncols() > 1
        || matches!(
            family_normalized.as_str(),
            "multinomial" | "multinomial-logit" | "softmax" | "categorical-logit"
        )
    {
        // Per-row Fisher-block override is now threaded through the
        // multi-output canonical fitters (issue #349): the multinomial path
        // consumes the active `(N, K-1, K-1)` leading sub-block and the
        // binomial-multi path the diagonal of each `(N, K, K)` block.
        let (design, t_mat) = build_latent_duchon_design(
            t_values.view(),
            n_obs,
            latent_dim,
            centers_values.view(),
            m,
            // GLM standalone latent entrypoint: no manifold/chart, open Euclidean.
            None,
        )
        .map_err(py_value_error)?;
        let (prior_score, aux_strength_state) = latent_prior_score_and_aux_state_for_t(
            t_mat.view(),
            aux_u_values.as_ref().map(|a| a.view()),
            aux_family,
            aux_strength,
            dim_selection_values.as_ref().map(|a| a.view()),
        )
        .map_err(py_value_error)?;
        let analytic_score = analytic_penalty_value_for_targets(&registry, t_values.view(), None)
            .map_err(py_value_error)?;
        return latent_multi_output_fit_to_pydict(
            py,
            design.view(),
            y_values.view(),
            penalty_values.view(),
            weight_values.as_ref().map(|w| w.view()),
            fisher_values.as_ref().map(|w| w.view()),
            init_lambda,
            &family_name,
            prior_score + analytic_score,
            aux_strength_state,
        );
    }
    let family = latent_glm_family_from_str(&family, tweedie_p, negbin_theta, beta_phi)
        .map_err(py_value_error)?;
    let analytic_penalties_for_thread = analytic_penalties.clone();
    let latent_payload_for_thread = latent_payload.clone();
    let (fit, design, _, aux_strength_state) =
        detach_py_result(py, "glm_reml_fit_latent", move || {
            let registry = build_analytic_penalty_registry_from_json(
                Some(&latent_payload_for_thread),
                analytic_penalties_for_thread.as_ref(),
            )?;
            let effective_weights = latent_scalar_weights_with_fisher(
                n_obs,
                weight_values.as_ref().map(|w| w.view()),
                fisher_values.as_ref().map(|w| w.view()),
            )?;
            glm_reml_fit_latent_impl(
                t_values.view(),
                y_values.view(),
                n_obs,
                latent_dim,
                centers_values.view(),
                m,
                penalty_values.view(),
                effective_weights.as_ref().map(|w| w.view()),
                init_lambda,
                family,
                aux_u_values.as_ref().map(|a| a.view()),
                aux_family,
                aux_strength,
                dim_selection_values.as_ref().map(|a| a.view()),
                Some(&registry),
            )
        })?;
    let out = PyDict::new(py);
    set_ok_glm_latent_items(py, &out, fit, n_obs, design.ncols(), aux_strength_state)?;
    Ok(out.unbind())
}

#[pyfunction(signature = (
    t,
    y,
    n_obs,
    latent_dim,
    centers,
    penalty,
    family,
    grad_reml_score = 1.0,
    m = 2,
    weights = None,
    fisher_w = None,
    init_lambda = None,
    aux_u = None,
    aux_family = "ridge".to_string(),
    aux_strength = None,
    dim_selection_log_precision = None,
    tweedie_p = 1.5,
    negbin_theta = 1.0,
    beta_phi = 1.0,
    basis_kind = "duchon".to_string(),
))]
fn glm_reml_fit_latent_backward<'py>(
    py: Python<'py>,
    t: PyReadonlyArray1<'py, f64>,
    y: PyReadonlyArray2<'py, f64>,
    n_obs: usize,
    latent_dim: usize,
    centers: PyReadonlyArray2<'py, f64>,
    penalty: PyReadonlyArray2<'py, f64>,
    family: String,
    grad_reml_score: f64,
    m: usize,
    weights: Option<PyReadonlyArray1<'py, f64>>,
    fisher_w: Option<PyReadonlyArray3<'py, f64>>,
    init_lambda: Option<f64>,
    aux_u: Option<PyReadonlyArray2<'py, f64>>,
    aux_family: String,
    aux_strength: Option<f64>,
    dim_selection_log_precision: Option<PyReadonlyArray1<'py, f64>>,
    tweedie_p: f64,
    negbin_theta: f64,
    beta_phi: f64,
    basis_kind: String,
) -> PyResult<Py<PyDict>> {
    let family = latent_glm_family_from_str(&family, tweedie_p, negbin_theta, beta_phi)
        .map_err(py_value_error)?;
    let aux_family = match aux_family.to_ascii_lowercase().as_str() {
        "ridge" => AuxPriorFamily::Ridge,
        "linear" => AuxPriorFamily::Linear,
        other => {
            return Err(py_value_error(format!(
                "aux_family must be 'ridge' or 'linear'; got {other:?}"
            )));
        }
    };
    let basis_kind_normalized = latent_basis_kind(&basis_kind).map_err(py_value_error)?;
    if basis_kind_normalized != "duchon" {
        return Err(PyNotImplementedError::new_err(format!(
            "glm_reml_fit_latent_backward currently builds only Duchon latent designs; derivative hook exists for {basis_kind_normalized:?}"
        )));
    }
    let effective_weights = latent_scalar_weights_with_fisher(
        n_obs,
        weights.as_ref().map(|w| w.as_array()),
        fisher_w.as_ref().map(|w| w.as_array()),
    )
    .map_err(py_value_error)?;
    let (fit, design, t_mat, _) = glm_reml_fit_latent_impl(
        t.as_array(),
        y.as_array(),
        n_obs,
        latent_dim,
        centers.as_array(),
        m,
        penalty.as_array(),
        effective_weights.as_ref().map(|w| w.view()),
        init_lambda,
        family,
        aux_u.as_ref().map(|a| a.as_array()),
        aux_family,
        aux_strength,
        dim_selection_log_precision.as_ref().map(|a| a.as_array()),
        None,
    )
    .map_err(py_value_error)?;
    let pirls = fit.artifacts.pirls.as_ref().ok_or_else(|| {
        py_value_error("latent GLM fit did not return PIRLS artifacts".to_string())
    })?;
    let h = pirls
        .dense_stabilizedhessian_transformed("latent GLM backward")
        .map_err(|err| py_value_error(err.to_string()))?;
    let factor = factorize_symmetricwith_fallback(
        gam::linalg::faer_ndarray::FaerArrayView::new(&h).as_ref(),
        Side::Lower,
    )
    .map_err(|err| py_value_error(format!("latent GLM Hessian factorization failed: {err}")))?;
    let qs = &pirls.reparam_result.qs;
    let beta_t = pirls.beta_transformed.as_ref();
    let q = beta_t.len();
    let p = design.ncols();
    let jet = latent_input_location_jet(
        basis_kind_normalized,
        t_mat.view(),
        centers.as_array(),
        m,
        None,
        None,
        None,
    )
    .map_err(py_value_error)?;
    if jet.shape()[0] != n_obs || jet.shape()[1] != p || jet.shape()[2] != latent_dim {
        return Err(py_value_error(format!(
            "latent input-location jet shape mismatch: expected {}x{}x{}, got {}x{}x{}",
            n_obs,
            p,
            latent_dim,
            jet.shape()[0],
            jet.shape()[1],
            jet.shape()[2],
        )));
    }
    let mut grad_t = Array1::<f64>::zeros(n_obs * latent_dim);
    let mut rhs = Array2::<f64>::zeros((q, 1));
    let mut direction_t = Array1::<f64>::zeros(q);
    let mut direction_orig = Array1::<f64>::zeros(p);
    for n in 0..n_obs {
        for col in 0..q {
            let mut value = 0.0_f64;
            for k in 0..p {
                value += design[[n, k]] * qs[[k, col]];
            }
            rhs[[col, 0]] = value;
        }
        {
            let mut rhs_view = array2_to_matmut(&mut rhs);
            factor.solve_in_place(rhs_view.as_mut());
        }
        let score_eta =
            pirls.solveweights[n] * (pirls.final_eta[n] - pirls.solveworking_response[n]);
        for col in 0..q {
            direction_t[col] = score_eta * beta_t[col] + pirls.finalweights[n] * rhs[[col, 0]];
        }
        direction_orig.fill(0.0);
        for k in 0..p {
            let mut value = 0.0_f64;
            for col in 0..q {
                value += qs[[k, col]] * direction_t[col];
            }
            direction_orig[k] = value;
        }
        for a in 0..latent_dim {
            let mut acc = 0.0_f64;
            for k in 0..p {
                acc += direction_orig[k] * jet[[n, k, a]];
            }
            grad_t[n * latent_dim + a] += grad_reml_score * acc;
        }
    }

    let mut grad_aux_log_strength: Option<f64> = None;
    if let Some(u_arr) = aux_u.as_ref() {
        let u_view = u_arr.as_array();
        let stats = latent_aux_prior_stats(t_mat.view(), u_view, aux_family, aux_strength)
            .map_err(py_value_error)?;
        let residual = &t_mat - &stats.targets;
        let projected_residual =
            aux_prior_targets(residual.view(), u_view, aux_family).map_err(py_value_error)?;
        let grad_base = residual - projected_residual;
        for n in 0..n_obs {
            for a in 0..latent_dim {
                grad_t[n * latent_dim + a] +=
                    grad_reml_score * stats.strength.mu * grad_base[[n, a]];
            }
        }
        grad_aux_log_strength = Some(
            grad_reml_score
                * (0.5 * stats.strength.mu * stats.residual_sq
                    - 0.5 * (n_obs * latent_dim) as f64),
        );
    }
    if let Some(log_prec) = dim_selection_log_precision.as_ref() {
        let lp = log_prec.as_array();
        if lp.len() != latent_dim {
            return Err(py_value_error(format!(
                "dim_selection_log_precision length {} must equal latent_dim {}",
                lp.len(),
                latent_dim
            )));
        }
        for n in 0..n_obs {
            for a in 0..latent_dim {
                let prec = lp[a].exp();
                if !(prec.is_finite() && prec > 0.0) {
                    return Err(py_value_error(format!(
                        "dim_selection_log_precision[{a}] must exponentiate to a finite positive precision"
                    )));
                }
                grad_t[n * latent_dim + a] += grad_reml_score * prec * t_mat[[n, a]];
            }
        }
    }
    let mut grad_t_matrix = Array2::<f64>::zeros((n_obs, latent_dim));
    for n in 0..n_obs {
        for a in 0..latent_dim {
            grad_t_matrix[[n, a]] = grad_t[n * latent_dim + a];
        }
    }
    let out = PyDict::new(py);
    out.set_item("grad_t", grad_t_matrix.into_pyarray(py))?;
    if let Some(grad) = grad_aux_log_strength {
        out.set_item("grad_aux_log_strength", grad)?;
        out.set_item("grad_log_mu", grad)?;
    } else {
        out.set_item("grad_aux_log_strength", py.None())?;
        out.set_item("grad_log_mu", py.None())?;
    }
    Ok(out.unbind())
}

fn set_ok_gaussian_reml_items<'py>(
    py: Python<'py>,
    out: &Bound<'py, PyDict>,
    fit: gam::solver::gaussian_reml::GaussianRemlMultiResult,
) -> PyResult<()> {
    let status = if fit.lambda.is_finite() && fit.reml_score.is_finite() {
        "ok"
    } else {
        "diverged"
    };
    out.set_item("status", status)?;
    out.set_item("lambda", fit.lambda)?;
    out.set_item("rho", fit.rho)?;
    out.set_item("reml_score", fit.reml_score)?;
    out.set_item("reml_grad_lambda", fit.reml_grad_lambda)?;
    out.set_item("reml_hess_lambda", fit.reml_hess_lambda)?;
    out.set_item("reml_grad_rho", fit.reml_grad_rho)?;
    out.set_item("reml_hess_rho", fit.reml_hess_rho)?;
    out.set_item("edf", fit.edf)?;
    out.set_item("coefficients", fit.coefficients.into_pyarray(py))?;
    out.set_item("fitted", fit.fitted.into_pyarray(py))?;
    out.set_item("sigma2", fit.sigma2.into_pyarray(py))?;
    out.set_item(
        "cache_penalty_eigenvalues",
        fit.cache.penalty_eigenvalues.into_pyarray(py),
    )?;
    out.set_item(
        "cache_eigenvectors",
        fit.cache.eigenvectors.into_pyarray(py),
    )?;
    out.set_item(
        "cache_coefficient_basis",
        fit.cache.coefficient_basis.into_pyarray(py),
    )?;
    out.set_item("cache_xtwx_fingerprint", fit.cache.xtwx_fingerprint)?;
    out.set_item("cache_penalty_fingerprint", fit.cache.penalty_fingerprint)?;
    out.set_item("cache_logdet_xtwx", fit.cache.logdet_xtwx)?;
    out.set_item(
        "cache_logdet_penalty_positive",
        fit.cache.logdet_penalty_positive,
    )?;
    out.set_item("cache_penalty_rank", fit.cache.penalty_rank)?;
    out.set_item("cache_nullity", fit.cache.nullity)?;
    Ok(())
}

fn gaussian_reml_fit_state_from_pydict(
    state: &Bound<'_, PyDict>,
) -> Result<gam::solver::gaussian_reml::GaussianRemlMultiResult, String> {
    fn get<'py>(state: &'py Bound<'py, PyDict>, key: &str) -> Result<Bound<'py, PyAny>, String> {
        state
            .get_item(key)
            .map_err(|err| err.to_string())?
            .ok_or_else(|| format!("forward_state is missing key {key:?}"))
    }

    let penalty_eigenvalues = get(state, "cache_penalty_eigenvalues")?
        .extract::<PyReadonlyArray1<'_, f64>>()
        .map_err(|err| err.to_string())?
        .as_array()
        .to_owned();
    let eigenvectors = get(state, "cache_eigenvectors")?
        .extract::<PyReadonlyArray2<'_, f64>>()
        .map_err(|err| err.to_string())?
        .as_array()
        .to_owned();
    let coefficient_basis = get(state, "cache_coefficient_basis")?
        .extract::<PyReadonlyArray2<'_, f64>>()
        .map_err(|err| err.to_string())?
        .as_array()
        .to_owned();
    let coefficients = get(state, "coefficients")?
        .extract::<PyReadonlyArray2<'_, f64>>()
        .map_err(|err| err.to_string())?
        .as_array()
        .to_owned();
    let fitted = get(state, "fitted")?
        .extract::<PyReadonlyArray2<'_, f64>>()
        .map_err(|err| err.to_string())?
        .as_array()
        .to_owned();
    let sigma2 = get(state, "sigma2")?
        .extract::<PyReadonlyArray1<'_, f64>>()
        .map_err(|err| err.to_string())?
        .as_array()
        .to_owned();

    Ok(gam::solver::gaussian_reml::GaussianRemlMultiResult {
        lambda: get(state, "lambda")?
            .extract::<f64>()
            .map_err(|err| err.to_string())?,
        rho: get(state, "rho")?
            .extract::<f64>()
            .map_err(|err| err.to_string())?,
        coefficients,
        fitted,
        reml_score: get(state, "reml_score")?
            .extract::<f64>()
            .map_err(|err| err.to_string())?,
        reml_grad_lambda: get(state, "reml_grad_lambda")?
            .extract::<f64>()
            .map_err(|err| err.to_string())?,
        reml_hess_lambda: get(state, "reml_hess_lambda")?
            .extract::<f64>()
            .map_err(|err| err.to_string())?,
        reml_grad_rho: get(state, "reml_grad_rho")?
            .extract::<f64>()
            .map_err(|err| err.to_string())?,
        reml_hess_rho: get(state, "reml_hess_rho")?
            .extract::<f64>()
            .map_err(|err| err.to_string())?,
        edf: get(state, "edf")?
            .extract::<f64>()
            .map_err(|err| err.to_string())?,
        sigma2,
        cache: gam::solver::gaussian_reml::GaussianRemlEigenCache {
            penalty_eigenvalues,
            eigenvectors,
            coefficient_basis,
            xtwx_fingerprint: get(state, "cache_xtwx_fingerprint")?
                .extract::<u64>()
                .map_err(|err| err.to_string())?,
            penalty_fingerprint: get(state, "cache_penalty_fingerprint")?
                .extract::<u64>()
                .map_err(|err| err.to_string())?,
            logdet_xtwx: get(state, "cache_logdet_xtwx")?
                .extract::<f64>()
                .map_err(|err| err.to_string())?,
            logdet_penalty_positive: get(state, "cache_logdet_penalty_positive")?
                .extract::<f64>()
                .map_err(|err| err.to_string())?,
            penalty_rank: get(state, "cache_penalty_rank")?
                .extract::<usize>()
                .map_err(|err| err.to_string())?,
            nullity: get(state, "cache_nullity")?
                .extract::<usize>()
                .map_err(|err| err.to_string())?,
        },
    })
}

fn batched_gaussian_reml_fits_from_pydict(
    state: &Bound<'_, PyDict>,
    row_offsets: ArrayView1<'_, usize>,
) -> Result<Vec<Option<gam::solver::gaussian_reml::GaussianRemlMultiResult>>, String> {
    fn get<'py>(state: &'py Bound<'py, PyDict>, key: &str) -> Result<Bound<'py, PyAny>, String> {
        state
            .get_item(key)
            .map_err(|err| err.to_string())?
            .ok_or_else(|| format!("forward_state is missing key {key:?}"))
    }
    if row_offsets.len() < 2 {
        return Err("row_offsets must contain at least [0, n]".to_string());
    }
    let batch = row_offsets.len() - 1;

    let statuses = get(state, "status")?
        .extract::<Vec<String>>()
        .map_err(|err| err.to_string())?;
    if statuses.len() != batch {
        return Err(format!(
            "forward_state[\"status\"] length mismatch: expected {batch}, got {}",
            statuses.len()
        ));
    }
    let lambdas = get(state, "lambda")?
        .extract::<PyReadonlyArray1<'_, f64>>()
        .map_err(|err| err.to_string())?;
    let lambdas = lambdas.as_array();
    let rhos = get(state, "rho")?
        .extract::<PyReadonlyArray1<'_, f64>>()
        .map_err(|err| err.to_string())?;
    let rhos = rhos.as_array();
    let reml_scores = get(state, "reml_score")?
        .extract::<PyReadonlyArray1<'_, f64>>()
        .map_err(|err| err.to_string())?;
    let reml_scores = reml_scores.as_array();
    let reml_grad_lambdas = get(state, "reml_grad_lambda")?
        .extract::<PyReadonlyArray1<'_, f64>>()
        .map_err(|err| err.to_string())?;
    let reml_grad_lambdas = reml_grad_lambdas.as_array();
    let reml_hess_lambdas = get(state, "reml_hess_lambda")?
        .extract::<PyReadonlyArray1<'_, f64>>()
        .map_err(|err| err.to_string())?;
    let reml_hess_lambdas = reml_hess_lambdas.as_array();
    let reml_grad_rhos = get(state, "reml_grad_rho")?
        .extract::<PyReadonlyArray1<'_, f64>>()
        .map_err(|err| err.to_string())?;
    let reml_grad_rhos = reml_grad_rhos.as_array();
    let reml_hess_rhos = get(state, "reml_hess_rho")?
        .extract::<PyReadonlyArray1<'_, f64>>()
        .map_err(|err| err.to_string())?;
    let reml_hess_rhos = reml_hess_rhos.as_array();
    let edf = get(state, "edf")?
        .extract::<PyReadonlyArray1<'_, f64>>()
        .map_err(|err| err.to_string())?;
    let edf = edf.as_array();
    let coefficients = get(state, "coefficients")?
        .extract::<PyReadonlyArray3<'_, f64>>()
        .map_err(|err| err.to_string())?;
    let coefficients = coefficients.as_array();
    let fitted = get(state, "fitted")?
        .extract::<PyReadonlyArray2<'_, f64>>()
        .map_err(|err| err.to_string())?;
    let fitted = fitted.as_array();
    let sigma2 = get(state, "sigma2")?
        .extract::<PyReadonlyArray2<'_, f64>>()
        .map_err(|err| err.to_string())?;
    let sigma2 = sigma2.as_array();
    let cache_penalty_eigenvalues = get(state, "cache_penalty_eigenvalues")?
        .extract::<PyReadonlyArray2<'_, f64>>()
        .map_err(|err| err.to_string())?;
    let cache_penalty_eigenvalues = cache_penalty_eigenvalues.as_array();
    let cache_eigenvectors = get(state, "cache_eigenvectors")?
        .extract::<PyReadonlyArray3<'_, f64>>()
        .map_err(|err| err.to_string())?;
    let cache_eigenvectors = cache_eigenvectors.as_array();
    let cache_coefficient_basis = get(state, "cache_coefficient_basis")?
        .extract::<PyReadonlyArray3<'_, f64>>()
        .map_err(|err| err.to_string())?;
    let cache_coefficient_basis = cache_coefficient_basis.as_array();
    let cache_xtwx_fingerprints = get(state, "cache_xtwx_fingerprints")?
        .extract::<PyReadonlyArray1<'_, u64>>()
        .map_err(|err| err.to_string())?;
    let cache_xtwx_fingerprints = cache_xtwx_fingerprints.as_array();
    let cache_penalty_fingerprints = get(state, "cache_penalty_fingerprints")?
        .extract::<PyReadonlyArray1<'_, u64>>()
        .map_err(|err| err.to_string())?;
    let cache_penalty_fingerprints = cache_penalty_fingerprints.as_array();
    let cache_logdet_xtwx = get(state, "cache_logdet_xtwx")?
        .extract::<PyReadonlyArray1<'_, f64>>()
        .map_err(|err| err.to_string())?;
    let cache_logdet_xtwx = cache_logdet_xtwx.as_array();
    let cache_logdet_penalty_positive = get(state, "cache_logdet_penalty_positive")?
        .extract::<PyReadonlyArray1<'_, f64>>()
        .map_err(|err| err.to_string())?;
    let cache_logdet_penalty_positive = cache_logdet_penalty_positive.as_array();
    let cache_penalty_ranks = get(state, "cache_penalty_ranks")?
        .extract::<PyReadonlyArray1<'_, i64>>()
        .map_err(|err| err.to_string())?;
    let cache_penalty_ranks = cache_penalty_ranks.as_array();
    let cache_nullities = get(state, "cache_nullities")?
        .extract::<PyReadonlyArray1<'_, i64>>()
        .map_err(|err| err.to_string())?;
    let cache_nullities = cache_nullities.as_array();

    let mut fits = Vec::with_capacity(batch);
    for b in 0..batch {
        if statuses[b] != "ok" {
            fits.push(None);
            continue;
        }
        let rank = cache_penalty_ranks[b];
        let nullity = cache_nullities[b];
        if rank < 0 || nullity < 0 {
            return Err(format!(
                "forward_state cache_penalty_ranks[{b}]={rank} or cache_nullities[{b}]={nullity} must be non-negative"
            ));
        }
        let start = row_offsets[b];
        let end = row_offsets[b + 1];
        if start > end || end > fitted.nrows() {
            return Err(format!(
                "row_offsets[{b}..{}]=({start},{end}) outside fitted shape {}",
                b + 1,
                fitted.nrows()
            ));
        }
        fits.push(Some(gam::solver::gaussian_reml::GaussianRemlMultiResult {
            lambda: lambdas[b],
            rho: rhos[b],
            coefficients: coefficients.slice(s![b, .., ..]).to_owned(),
            fitted: fitted.slice(s![start..end, ..]).to_owned(),
            reml_score: reml_scores[b],
            reml_grad_lambda: reml_grad_lambdas[b],
            reml_hess_lambda: reml_hess_lambdas[b],
            reml_grad_rho: reml_grad_rhos[b],
            reml_hess_rho: reml_hess_rhos[b],
            edf: edf[b],
            sigma2: sigma2.slice(s![b, ..]).to_owned(),
            cache: gam::solver::gaussian_reml::GaussianRemlEigenCache {
                penalty_eigenvalues: cache_penalty_eigenvalues.slice(s![b, ..]).to_owned(),
                eigenvectors: cache_eigenvectors.slice(s![b, .., ..]).to_owned(),
                coefficient_basis: cache_coefficient_basis.slice(s![b, .., ..]).to_owned(),
                xtwx_fingerprint: cache_xtwx_fingerprints[b],
                penalty_fingerprint: cache_penalty_fingerprints[b],
                logdet_xtwx: cache_logdet_xtwx[b],
                logdet_penalty_positive: cache_logdet_penalty_positive[b],
                penalty_rank: rank as usize,
                nullity: nullity as usize,
            },
        }));
    }
    Ok(fits)
}

fn set_degenerate_gaussian_reml_items<'py>(
    py: Python<'py>,
    out: &Bound<'py, PyDict>,
    n_rows: usize,
    n_outputs: usize,
    n_coefficients: usize,
) -> PyResult<()> {
    out.set_item("status", "degenerate")?;
    out.set_item("lambda", f64::NAN)?;
    out.set_item("rho", f64::NAN)?;
    out.set_item("reml_score", f64::NAN)?;
    out.set_item("reml_grad_lambda", f64::NAN)?;
    out.set_item("reml_hess_lambda", f64::NAN)?;
    out.set_item("reml_grad_rho", f64::NAN)?;
    out.set_item("reml_hess_rho", f64::NAN)?;
    out.set_item("edf", 0.0)?;
    out.set_item(
        "coefficients",
        Array2::<f64>::zeros((n_coefficients, n_outputs)).into_pyarray(py),
    )?;
    out.set_item(
        "fitted",
        Array2::<f64>::zeros((n_rows, n_outputs)).into_pyarray(py),
    )?;
    out.set_item(
        "sigma2",
        Array1::<f64>::from_elem(n_outputs, f64::NAN).into_pyarray(py),
    )?;
    Ok(())
}

struct BatchedGaussianRemlResult {
    statuses: Vec<String>,
    lambdas: Array1<f64>,
    rhos: Array1<f64>,
    reml_scores: Array1<f64>,
    reml_grad_lambdas: Array1<f64>,
    reml_hess_lambdas: Array1<f64>,
    reml_grad_rhos: Array1<f64>,
    reml_hess_rhos: Array1<f64>,
    edf: Array1<f64>,
    coefficients: Array3<f64>,
    fitted: Array2<f64>,
    sigma2: Array2<f64>,
    // Per-fit cache stacked across the batch — populated for fits whose status
    // is "ok". Backward callers re-bind these into `GaussianRemlMultiResult`
    // and route to `_from_fit`, skipping the redundant ρ-search + cache build
    // that would otherwise double per-step cost at training-loop frequency.
    cache_penalty_eigenvalues: Array2<f64>,
    cache_eigenvectors: Array3<f64>,
    cache_coefficient_basis: Array3<f64>,
    cache_xtwx_fingerprints: Array1<u64>,
    cache_penalty_fingerprints: Array1<u64>,
    cache_logdet_xtwx: Array1<f64>,
    cache_logdet_penalty_positive: Array1<f64>,
    cache_penalty_ranks: Array1<i64>,
    cache_nullities: Array1<i64>,
}

struct BatchedGaussianRemlBackwardResult {
    statuses: Vec<String>,
    grad_x: Array2<f64>,
    grad_y: Array2<f64>,
    grad_penalty: Array2<f64>,
    grad_weights: Array1<f64>,
}

struct PositionGaussianRemlBackwardResult {
    grad_t: Array1<f64>,
    grad_y: Array2<f64>,
    grad_penalty: Array2<f64>,
    grad_weights: Array1<f64>,
    grad_by: Option<Array1<f64>>,
}

struct BatchedPositionGaussianRemlBackwardResult {
    statuses: Vec<String>,
    grad_t: Array1<f64>,
    grad_y: Array2<f64>,
    grad_penalty: Array2<f64>,
    grad_weights: Array1<f64>,
    grad_by: Option<Array1<f64>>,
}

/// Shared-tangent multi-output Gaussian REML fit.
///
/// One full Gaussian GAM is fitted per tangent coordinate (matching the
/// documented `response_geometry` contract: "one scalar Gaussian GAM is fitted
/// for each tangent coordinate"), but all coordinates are estimated jointly
/// under one smoothing parameter per formula smooth shared across every
/// coordinate, and an optional cross-coordinate Fisher-Rao precision metric
/// couples their residuals. The numerical engine is
/// `gam::families::response_geometry::fit_shared_tangent_reml`; this FFI layer
/// only marshals the formula artifacts into its typed request.
struct TangentRemlMultiResult {
    /// Per-output coefficients, shape `(K, D)`.
    coefficients: Array2<f64>,
    /// Per-output fitted tangent values `X · β_d`, shape `(N, D)`.
    fitted: Array2<f64>,
    /// Pooled isotropic residual variance, length 1. Isotropic tangent noise
    /// (`Cov = σ²·I_D`) is the rotation-invariant noise model; a per-coordinate
    /// scale would itself break frame equivariance.
    sigma2: Array1<f64>,
    /// Shared per-smooth fitted smoothing parameters, length `M`. One λ per
    /// formula smooth, common to every tangent output coordinate.
    lambdas: Array1<f64>,
    /// Shared per-smooth effective degrees of freedom, length `M` (the full
    /// effective df of the `Sᵇ ⊗ I_D` block across all `D` outputs).
    edf: Array1<f64>,
    /// Joint REML score.
    reml_score: f64,
}

/// Marshaling-vs-engine error split for the shared-tangent formula fit. The
/// engine's typed `EstimationError` must reach `estimation_error_to_pyerr`
/// unflattened so a non-converged outer search keeps its
/// `RemlConvergenceError` class identity and structured resume evidence.
enum SharedTangentFfiError {
    Spec(String),
    Engine(EstimationError),
}

fn gaussian_reml_fit_formula_dataset_impl(
    dataset: EncodedDataset,
    formula: String,
    y: ArrayView2<'_, f64>,
    config_json: Option<&str>,
    fisher_rao_w: Option<ArrayView3<'_, f64>>,
) -> Result<TangentRemlMultiResult, SharedTangentFfiError> {
    let mut fit_config = parse_fit_config(config_json).map_err(SharedTangentFfiError::Spec)?;
    fit_config.family = Some("gaussian".to_string());
    fit_config.link = Some("identity".to_string());
    let materialized = materialize(&formula, &dataset, &fit_config)
        .map_err(|err| SharedTangentFfiError::Spec(err.to_string()))?;
    let standard = match materialized.request {
        FitRequest::Standard(request) => request,
        _ => {
            return Err(SharedTangentFfiError::Spec(
                "shared-tangent Gaussian REML fitting requires a standard Gaussian formula"
                    .to_string(),
            ));
        }
    };
    if !standard.family.is_gaussian_identity() {
        return Err(SharedTangentFfiError::Spec(
            "shared-tangent Gaussian REML fitting requires Gaussian identity".to_string(),
        ));
    }
    if standard.wiggle.is_some() {
        return Err(SharedTangentFfiError::Spec(
            "shared-tangent Gaussian REML fitting does not support link wiggle".to_string(),
        ));
    }
    if standard.offset.iter().any(|value| value.abs() > 0.0) {
        return Err(SharedTangentFfiError::Spec(
            "shared-tangent Gaussian REML fitting does not support offsets".to_string(),
        ));
    }
    // Build the formula design and hand the joint problem to the core
    // shared-tangent REML engine. The engine consumes the design through
    // bounded row chunks and streams exact joint sufficient statistics, so the
    // stacked `(N·D) × (K·D)` system and the `Sᵇ ⊗ I_D` Kronecker penalties are
    // never materialized on this side of the boundary; all remaining input
    // validation (shapes, finiteness, weight signs, metric shape/PD) is owned
    // by the engine's typed request preparation.
    let design =
        gam::terms::smooth::build_term_collection_design(standard.data.view(), &standard.spec)
            .map_err(|err| {
                SharedTangentFfiError::Spec(format!("failed to build formula design matrix: {err}"))
            })?;
    let penalties: Vec<gam::families::response_geometry::SharedTangentPenalty> = design
        .penalties
        .iter()
        .map(|penalty| {
            gam::families::response_geometry::SharedTangentPenalty::new(
                penalty.col_range.start,
                penalty.local.clone(),
            )
        })
        .collect();
    let request = gam::families::response_geometry::SharedTangentRemlRequest::new(
        design.design,
        y.to_owned(),
        standard.weights.clone(),
        fisher_rao_w.map(|metric| metric.to_owned()),
        penalties,
    );
    let fit = gam::families::response_geometry::fit_shared_tangent_reml(request)
        .map_err(SharedTangentFfiError::Engine)?;
    Ok(TangentRemlMultiResult {
        sigma2: Array1::from_elem(1, fit.sigma2),
        coefficients: fit.coefficients,
        fitted: fit.fitted,
        lambdas: fit.lambdas,
        edf: fit.edf_by_penalty,
        reml_score: fit.reml_score,
    })
}




fn gaussian_reml_fit_batched_impl(
    x: ArrayView2<'_, f64>,
    y: ArrayView2<'_, f64>,
    row_offsets: ArrayView1<'_, usize>,
    penalty: ArrayView2<'_, f64>,
    weights: Option<ArrayView1<'_, f64>>,
    init_lambda: Option<f64>,
) -> Result<BatchedGaussianRemlResult, String> {
    use rayon::iter::{IntoParallelIterator, ParallelIterator};

    if row_offsets.len() < 2 {
        return Err("row_offsets must contain at least [0, n]".to_string());
    }
    if row_offsets[0] != 0 || row_offsets[row_offsets.len() - 1] != x.nrows() {
        return Err(format!(
            "row_offsets must start at 0 and end at X.nrows(); got start={}, end={}, n={}",
            row_offsets[0],
            row_offsets[row_offsets.len() - 1],
            x.nrows()
        ));
    }
    for idx in 0..row_offsets.len() - 1 {
        if row_offsets[idx] > row_offsets[idx + 1] {
            return Err("row_offsets must be non-decreasing".to_string());
        }
    }
    if y.nrows() != x.nrows() {
        return Err(format!(
            "batched Gaussian REML row mismatch: X has {} rows but Y has {}",
            x.nrows(),
            y.nrows()
        ));
    }
    if x.ncols() == 0 || y.ncols() == 0 {
        return Err("batched Gaussian REML requires non-empty X and Y columns".to_string());
    }
    if penalty.nrows() != x.ncols() || penalty.ncols() != x.ncols() {
        return Err(format!(
            "penalty shape mismatch: expected {}x{}, got {}x{}",
            x.ncols(),
            x.ncols(),
            penalty.nrows(),
            penalty.ncols()
        ));
    }
    if let Some(weights) = weights {
        if weights.len() != x.nrows() {
            return Err(format!(
                "weights length mismatch: expected {}, got {}",
                x.nrows(),
                weights.len()
            ));
        }
        if weights
            .iter()
            .any(|value| !value.is_finite() || *value < 0.0)
        {
            return Err(
                "batched Gaussian REML weights must be finite non-negative values".to_string(),
            );
        }
    }
    if x.iter()
        .chain(y.iter())
        .chain(penalty.iter())
        .any(|value| !value.is_finite())
    {
        return Err("batched Gaussian REML inputs must be finite".to_string());
    }
    if let Some(lambda) = init_lambda {
        if !lambda.is_finite() || lambda <= 0.0 {
            return Err(format!(
                "init_lambda must be finite and positive when provided; got {lambda}"
            ));
        }
    }

    let batch = row_offsets.len() - 1;
    let p = x.ncols();
    let d = y.ncols();

    // Phase A: compute X'WX per fit in parallel (CPU or per-fit GPU dispatch
    // via `fast_xt_diag_x`). Per-fit X'WX cost is `O(n_b · p²)`; this phase
    // produces K p×p matrices that feed the batched Cholesky in Phase B.
    let xtwx_phase: Vec<Option<Array2<f64>>> = (0..batch)
        .into_par_iter()
        .map(|b| {
            let start = row_offsets[b];
            let end = row_offsets[b + 1];
            if start == end {
                return None;
            }
            let x_slice = x.slice(s![start..end, ..]);
            let owned_weight: Array1<f64> = match weights.as_ref() {
                Some(w) => w.slice(s![start..end]).to_owned(),
                None => Array1::ones(end - start),
            };
            Some(gam::linalg::faer_ndarray::fast_xt_diag_x(
                &x_slice,
                &owned_weight,
            ))
        })
        .collect();

    // Phase B: assemble live X'WX matrices and run the batched cache build.
    // The Cholesky step inside collapses to a single `cusolverDnDpotrfBatched`
    // call when policy + uniform shape allow; otherwise falls back to
    // per-fit Cholesky inside the helper. The remaining whitened-penalty
    // eigh stays per-fit (cuSOLVER has no batched symmetric eigensolver).
    let mut live_indices: Vec<usize> = Vec::with_capacity(batch);
    let mut live_xtwx: Vec<Array2<f64>> = Vec::with_capacity(batch);
    for (b, slot) in xtwx_phase.into_iter().enumerate() {
        if let Some(xtwx) = slot {
            live_indices.push(b);
            live_xtwx.push(xtwx);
        }
    }
    let batched_caches = build_gaussian_reml_eigen_cache_batched(live_xtwx, penalty, None);
    let mut prebuilt_caches: Vec<Option<gam::solver::gaussian_reml::GaussianRemlEigenCache>> =
        (0..batch).map(|_| None).collect();
    for (i, cache_result) in batched_caches.into_iter().enumerate() {
        if let Ok(cache) = cache_result {
            prebuilt_caches[live_indices[i]] = Some(cache);
        }
    }

    // Phase C: per-fit completion. Each fit either uses the prebuilt cache
    // (skipping its chol + eigh in `prepare_gaussian_reml`) or falls through
    // to a fresh build when the batched cache build dropped that element.
    let fit_results: Vec<
        Result<
            (
                usize,
                Option<gam::solver::gaussian_reml::GaussianRemlMultiResult>,
            ),
            String,
        >,
    > = (0..batch)
        .into_par_iter()
        .map(|b| {
            let start = row_offsets[b];
            let end = row_offsets[b + 1];
            if start == end {
                return Ok((b, None));
            }
            let weight_slice = weights.as_ref().map(|w| w.slice(s![start..end]));
            let cache_ref = prebuilt_caches[b].as_ref();
            match gaussian_reml_multi_closed_form_with_cache(
                x.slice(s![start..end, ..]),
                y.slice(s![start..end, ..]),
                penalty,
                weight_slice,
                init_lambda,
                cache_ref,
            ) {
                Ok(result) => Ok((b, Some(result))),
                Err(EstimationError::ModelIsIllConditioned { .. }) => Ok((b, None)),
                Err(err) => Err(format!("batched Gaussian REML fit {b} failed: {err}")),
            }
        })
        .collect();

    let mut lambdas = Array1::<f64>::from_elem(batch, f64::NAN);
    let mut rhos = Array1::<f64>::from_elem(batch, f64::NAN);
    let mut reml_scores = Array1::<f64>::from_elem(batch, f64::NAN);
    let mut reml_grad_lambdas = Array1::<f64>::from_elem(batch, f64::NAN);
    let mut reml_hess_lambdas = Array1::<f64>::from_elem(batch, f64::NAN);
    let mut reml_grad_rhos = Array1::<f64>::from_elem(batch, f64::NAN);
    let mut reml_hess_rhos = Array1::<f64>::from_elem(batch, f64::NAN);
    let mut edf = Array1::<f64>::zeros(batch);
    let mut coefficients = Array3::<f64>::zeros((batch, p, d));
    let mut fitted = Array2::<f64>::zeros((x.nrows(), d));
    let mut sigma2 = Array2::<f64>::from_elem((batch, d), f64::NAN);
    let mut statuses = vec!["degenerate".to_string(); batch];
    let mut cache_penalty_eigenvalues = Array2::<f64>::zeros((batch, p));
    let mut cache_eigenvectors = Array3::<f64>::zeros((batch, p, p));
    let mut cache_coefficient_basis = Array3::<f64>::zeros((batch, p, p));
    let mut cache_xtwx_fingerprints = Array1::<u64>::zeros(batch);
    let mut cache_penalty_fingerprints = Array1::<u64>::zeros(batch);
    let mut cache_logdet_xtwx = Array1::<f64>::zeros(batch);
    let mut cache_logdet_penalty_positive = Array1::<f64>::zeros(batch);
    let mut cache_penalty_ranks = Array1::<i64>::zeros(batch);
    let mut cache_nullities = Array1::<i64>::zeros(batch);

    for result in fit_results {
        let (b, fit) = result?;
        if let Some(fit) = fit {
            let start = row_offsets[b];
            let end = row_offsets[b + 1];
            statuses[b] = if fit.lambda.is_finite() && fit.reml_score.is_finite() {
                "ok".to_string()
            } else {
                "diverged".to_string()
            };
            lambdas[b] = fit.lambda;
            rhos[b] = fit.rho;
            reml_scores[b] = fit.reml_score;
            reml_grad_lambdas[b] = fit.reml_grad_lambda;
            reml_hess_lambdas[b] = fit.reml_hess_lambda;
            reml_grad_rhos[b] = fit.reml_grad_rho;
            reml_hess_rhos[b] = fit.reml_hess_rho;
            edf[b] = fit.edf;
            coefficients
                .slice_mut(s![b, .., ..])
                .assign(&fit.coefficients);
            fitted.slice_mut(s![start..end, ..]).assign(&fit.fitted);
            sigma2.slice_mut(s![b, ..]).assign(&fit.sigma2);
            cache_penalty_eigenvalues
                .slice_mut(s![b, ..])
                .assign(&fit.cache.penalty_eigenvalues);
            cache_eigenvectors
                .slice_mut(s![b, .., ..])
                .assign(&fit.cache.eigenvectors);
            cache_coefficient_basis
                .slice_mut(s![b, .., ..])
                .assign(&fit.cache.coefficient_basis);
            cache_xtwx_fingerprints[b] = fit.cache.xtwx_fingerprint;
            cache_penalty_fingerprints[b] = fit.cache.penalty_fingerprint;
            cache_logdet_xtwx[b] = fit.cache.logdet_xtwx;
            cache_logdet_penalty_positive[b] = fit.cache.logdet_penalty_positive;
            cache_penalty_ranks[b] = fit.cache.penalty_rank as i64;
            cache_nullities[b] = fit.cache.nullity as i64;
        }
    }

    Ok(BatchedGaussianRemlResult {
        statuses,
        lambdas,
        rhos,
        reml_scores,
        reml_grad_lambdas,
        reml_hess_lambdas,
        reml_grad_rhos,
        reml_hess_rhos,
        edf,
        coefficients,
        fitted,
        sigma2,
        cache_penalty_eigenvalues,
        cache_eigenvectors,
        cache_coefficient_basis,
        cache_xtwx_fingerprints,
        cache_penalty_fingerprints,
        cache_logdet_xtwx,
        cache_logdet_penalty_positive,
        cache_penalty_ranks,
        cache_nullities,
    })
}

fn gaussian_reml_fit_batched_backward_impl(
    x: ArrayView2<'_, f64>,
    y: ArrayView2<'_, f64>,
    row_offsets: ArrayView1<'_, usize>,
    penalty: ArrayView2<'_, f64>,
    weights: Option<ArrayView1<'_, f64>>,
    init_lambda: Option<f64>,
    grad_lambda: Option<ArrayView1<'_, f64>>,
    grad_coefficients: Option<ArrayView3<'_, f64>>,
    grad_fitted: Option<ArrayView2<'_, f64>>,
    grad_reml_score: Option<ArrayView1<'_, f64>>,
    grad_edf: Option<ArrayView1<'_, f64>>,
    forward_fits: Option<&[Option<gam::solver::gaussian_reml::GaussianRemlMultiResult>]>,
) -> Result<BatchedGaussianRemlBackwardResult, String> {
    use rayon::iter::{IntoParallelIterator, ParallelIterator};

    validate_batched_reml_common(x, y, row_offsets, penalty, weights, init_lambda)?;
    let batch = row_offsets.len() - 1;
    let p = x.ncols();
    let d = y.ncols();
    validate_batched_reml_upstreams(
        batch,
        p,
        d,
        y.nrows(),
        grad_lambda,
        grad_coefficients,
        grad_fitted,
        grad_reml_score,
        grad_edf,
    )?;
    if let Some(fits) = forward_fits {
        if fits.len() != batch {
            return Err(format!(
                "forward_state fit count mismatch: expected {batch}, got {}",
                fits.len()
            ));
        }
    }

    // When the caller supplied forward state, every active fit gets its
    // own `GaussianRemlMultiBackwardProblem` and the K-way batched entry
    // point handles the K-aggregate inverse-Hessian computation via
    // cuBLAS strided-batched gemm (one device call instead of K). Without
    // state, each fit refits internally; that path keeps the existing
    // per-fit par_iter shape.
    let results: Vec<
        Result<
            (
                usize,
                Option<gam::solver::gaussian_reml::GaussianRemlBackwardResult>,
            ),
            String,
        >,
    > = if let Some(fits) = forward_fits {
        let mut active: Vec<(usize, GaussianRemlMultiBackwardProblem<'_>)> =
            Vec::with_capacity(batch);
        for b in 0..batch {
            let start = row_offsets[b];
            let end = row_offsets[b + 1];
            if start == end {
                continue;
            }
            let Some(fit) = fits[b].as_ref() else {
                continue;
            };
            active.push((
                b,
                GaussianRemlMultiBackwardProblem {
                    x: x.slice(s![start..end, ..]),
                    y: y.slice(s![start..end, ..]),
                    weights: weights.as_ref().map(|w| w.slice(s![start..end])),
                    fit,
                    grad_lambda: grad_lambda.as_ref().map_or(0.0, |g| g[b]),
                    grad_coefficients: grad_coefficients.as_ref().map(|g| g.slice(s![b, .., ..])),
                    grad_fitted: grad_fitted.as_ref().map(|g| g.slice(s![start..end, ..])),
                    grad_reml_score: grad_reml_score.as_ref().map_or(0.0, |g| g[b]),
                    grad_edf: grad_edf.as_ref().map_or(0.0, |g| g[b]),
                },
            ));
        }
        let (indices, problems): (Vec<usize>, Vec<GaussianRemlMultiBackwardProblem<'_>>) =
            active.into_iter().unzip();
        let batch_results = gaussian_reml_multi_closed_form_backward_batch(&problems, penalty);
        batch_results
            .into_iter()
            .zip(indices.into_iter())
            .map(|(result, b)| match result {
                Ok(backward) => Ok((b, Some(backward))),
                Err(EstimationError::ModelIsIllConditioned { .. }) => Ok((b, None)),
                Err(err) => Err(format!("batched Gaussian REML backward {b} failed: {err}")),
            })
            .collect()
    } else {
        (0..batch)
            .into_par_iter()
            .map(|b| {
                let start = row_offsets[b];
                let end = row_offsets[b + 1];
                if start == end {
                    return Ok((b, None));
                }
                let weight_slice = weights.as_ref().map(|w| w.slice(s![start..end]));
                let upstream_lambda = grad_lambda.as_ref().map_or(0.0, |g| g[b]);
                let upstream_reml_score = grad_reml_score.as_ref().map_or(0.0, |g| g[b]);
                let upstream_edf = grad_edf.as_ref().map_or(0.0, |g| g[b]);
                let upstream_coefficients =
                    grad_coefficients.as_ref().map(|g| g.slice(s![b, .., ..]));
                let upstream_fitted = grad_fitted.as_ref().map(|g| g.slice(s![start..end, ..]));
                let x_slice = x.slice(s![start..end, ..]);
                let y_slice = y.slice(s![start..end, ..]);
                let backward_result = gaussian_reml_multi_closed_form_backward(
                    x_slice,
                    y_slice,
                    penalty,
                    weight_slice,
                    init_lambda,
                    upstream_lambda,
                    upstream_coefficients,
                    upstream_fitted,
                    upstream_reml_score,
                    upstream_edf,
                );
                match backward_result {
                    Ok(backward) => Ok((b, Some(backward))),
                    Err(EstimationError::ModelIsIllConditioned { .. }) => Ok((b, None)),
                    Err(err) => Err(format!("batched Gaussian REML backward {b} failed: {err}")),
                }
            })
            .collect()
    };

    let mut statuses = vec!["degenerate".to_string(); batch];
    let mut grad_x = Array2::<f64>::zeros(x.dim());
    let mut grad_y = Array2::<f64>::zeros(y.dim());
    let mut grad_penalty = Array2::<f64>::zeros(penalty.dim());
    let mut grad_weights = Array1::<f64>::zeros(x.nrows());
    for result in results {
        let (b, backward) = result?;
        if let Some(backward) = backward {
            let start = row_offsets[b];
            let end = row_offsets[b + 1];
            statuses[b] = "ok".to_string();
            grad_x
                .slice_mut(s![start..end, ..])
                .assign(&backward.grad_x);
            grad_y
                .slice_mut(s![start..end, ..])
                .assign(&backward.grad_y);
            grad_penalty += &backward.grad_penalty;
            grad_weights
                .slice_mut(s![start..end])
                .assign(&backward.grad_weights);
        }
    }

    Ok(BatchedGaussianRemlBackwardResult {
        statuses,
        grad_x,
        grad_y,
        grad_penalty,
        grad_weights,
    })
}

fn validate_batched_reml_common(
    x: ArrayView2<'_, f64>,
    y: ArrayView2<'_, f64>,
    row_offsets: ArrayView1<'_, usize>,
    penalty: ArrayView2<'_, f64>,
    weights: Option<ArrayView1<'_, f64>>,
    init_lambda: Option<f64>,
) -> Result<(), String> {
    if row_offsets.len() < 2 {
        return Err("row_offsets must contain at least [0, n]".to_string());
    }
    if row_offsets[0] != 0 || row_offsets[row_offsets.len() - 1] != x.nrows() {
        return Err(format!(
            "row_offsets must start at 0 and end at X.nrows(); got start={}, end={}, n={}",
            row_offsets[0],
            row_offsets[row_offsets.len() - 1],
            x.nrows()
        ));
    }
    for idx in 0..row_offsets.len() - 1 {
        if row_offsets[idx] > row_offsets[idx + 1] {
            return Err("row_offsets must be non-decreasing".to_string());
        }
    }
    if y.nrows() != x.nrows() {
        return Err(format!(
            "batched Gaussian REML row mismatch: X has {} rows but Y has {}",
            x.nrows(),
            y.nrows()
        ));
    }
    if x.ncols() == 0 || y.ncols() == 0 {
        return Err("batched Gaussian REML requires non-empty X and Y columns".to_string());
    }
    if penalty.nrows() != x.ncols() || penalty.ncols() != x.ncols() {
        return Err(format!(
            "penalty shape mismatch: expected {}x{}, got {}x{}",
            x.ncols(),
            x.ncols(),
            penalty.nrows(),
            penalty.ncols()
        ));
    }
    if let Some(weights) = weights {
        if weights.len() != x.nrows() {
            return Err(format!(
                "weights length mismatch: expected {}, got {}",
                x.nrows(),
                weights.len()
            ));
        }
        if weights
            .iter()
            .any(|value| !value.is_finite() || *value < 0.0)
        {
            return Err(
                "batched Gaussian REML weights must be finite non-negative values".to_string(),
            );
        }
    }
    if x.iter()
        .chain(y.iter())
        .chain(penalty.iter())
        .any(|value| !value.is_finite())
    {
        return Err("batched Gaussian REML inputs must be finite".to_string());
    }
    if let Some(lambda) = init_lambda {
        if !lambda.is_finite() || lambda <= 0.0 {
            return Err(format!(
                "init_lambda must be finite and positive when provided; got {lambda}"
            ));
        }
    }
    Ok(())
}

fn validate_batched_reml_upstreams(
    batch: usize,
    p: usize,
    d: usize,
    n: usize,
    grad_lambda: Option<ArrayView1<'_, f64>>,
    grad_coefficients: Option<ArrayView3<'_, f64>>,
    grad_fitted: Option<ArrayView2<'_, f64>>,
    grad_reml_score: Option<ArrayView1<'_, f64>>,
    grad_edf: Option<ArrayView1<'_, f64>>,
) -> Result<(), String> {
    if let Some(grad_lambda) = grad_lambda {
        if grad_lambda.len() != batch {
            return Err(format!(
                "grad_lambda length mismatch: expected {batch}, got {}",
                grad_lambda.len()
            ));
        }
        if grad_lambda.iter().any(|value| !value.is_finite()) {
            return Err("grad_lambda must contain only finite values".to_string());
        }
    }
    if let Some(grad_reml_score) = grad_reml_score {
        if grad_reml_score.len() != batch {
            return Err(format!(
                "grad_reml_score length mismatch: expected {batch}, got {}",
                grad_reml_score.len()
            ));
        }
        if grad_reml_score.iter().any(|value| !value.is_finite()) {
            return Err("grad_reml_score must contain only finite values".to_string());
        }
    }
    if let Some(grad_edf) = grad_edf {
        if grad_edf.len() != batch {
            return Err(format!(
                "grad_edf length mismatch: expected {batch}, got {}",
                grad_edf.len()
            ));
        }
        if grad_edf.iter().any(|value| !value.is_finite()) {
            return Err("grad_edf must contain only finite values".to_string());
        }
    }
    if let Some(grad_coefficients) = grad_coefficients {
        if grad_coefficients.dim() != (batch, p, d) {
            let (got_b, got_p, got_d) = grad_coefficients.dim();
            return Err(format!(
                "grad_coefficients shape mismatch: expected {batch}x{p}x{d}, got {got_b}x{got_p}x{got_d}"
            ));
        }
        if grad_coefficients.iter().any(|value| !value.is_finite()) {
            return Err("grad_coefficients must contain only finite values".to_string());
        }
    }
    if let Some(grad_fitted) = grad_fitted {
        if grad_fitted.dim() != (n, d) {
            return Err(format!(
                "grad_fitted shape mismatch: expected {n}x{d}, got {}x{}",
                grad_fitted.nrows(),
                grad_fitted.ncols()
            ));
        }
        if grad_fitted.iter().any(|value| !value.is_finite()) {
            return Err("grad_fitted must contain only finite values".to_string());
        }
    }
    Ok(())
}

fn gaussian_reml_fit_positions_backward_impl(
    t: ArrayView1<'_, f64>,
    y: ArrayView2<'_, f64>,
    knots_or_centers: ArrayView1<'_, f64>,
    basis_kind: &str,
    basis_order: usize,
    periodic: bool,
    period: Option<f64>,
    penalty: ArrayView2<'_, f64>,
    weights: Option<ArrayView1<'_, f64>>,
    init_lambda: Option<f64>,
    grad_lambda: f64,
    grad_coefficients: Option<ArrayView2<'_, f64>>,
    grad_fitted: Option<ArrayView2<'_, f64>>,
    grad_reml_score: f64,
    grad_edf: f64,
    by: Option<ArrayView1<'_, f64>>,
    by_start_col: usize,
    forward_fit: Option<&gam::solver::gaussian_reml::GaussianRemlMultiResult>,
) -> Result<PositionGaussianRemlBackwardResult, String> {
    let x = position_basis_design(
        t,
        knots_or_centers,
        basis_kind,
        basis_order,
        periodic,
        period,
    )?;
    let basis_derivative = position_basis_derivative(
        t,
        knots_or_centers,
        basis_kind,
        basis_order,
        periodic,
        period,
    )?;
    let gated_x = gate_design_for_forward(x.view(), by, by_start_col)?;
    let fit_x = gated_x.as_ref().map_or(x.view(), |g| g.view());
    let gated_weights = gate_weights_for_forward(weights, by, x.nrows())?;
    let backward = if let Some(fit) = forward_fit {
        gaussian_reml_multi_closed_form_backward_from_fit(
            fit_x,
            y,
            penalty,
            gated_weights.as_ref().map(|w| w.view()),
            fit,
            grad_lambda,
            grad_coefficients,
            grad_fitted,
            grad_reml_score,
            grad_edf,
        )
    } else {
        gaussian_reml_multi_closed_form_backward(
            fit_x,
            y,
            penalty,
            gated_weights.as_ref().map(|w| w.view()),
            init_lambda,
            grad_lambda,
            grad_coefficients,
            grad_fitted,
            grad_reml_score,
            grad_edf,
        )
    }
    .map_err(|err| err.to_string())?;
    let (grad_x, grad_by) = ungate_design_gradient(x.view(), by, by_start_col, backward.grad_x)?;
    let grad_t = contract_position_gradient(grad_x.view(), basis_derivative.view())?;
    Ok(PositionGaussianRemlBackwardResult {
        grad_t,
        grad_y: backward.grad_y,
        grad_penalty: backward.grad_penalty,
        grad_weights: ungate_weight_gradient(by, backward.grad_weights),
        grad_by,
    })
}

fn gaussian_reml_fit_positions_batched_impl(
    t: ArrayView1<'_, f64>,
    y: ArrayView2<'_, f64>,
    row_offsets: ArrayView1<'_, usize>,
    knots_or_centers: ArrayView1<'_, f64>,
    basis_kind: &str,
    basis_order: usize,
    periodic: bool,
    period: Option<f64>,
    penalty: ArrayView2<'_, f64>,
    weights: Option<ArrayView1<'_, f64>>,
    init_lambda: Option<f64>,
    by: Option<ArrayView1<'_, f64>>,
    by_start_col: usize,
) -> Result<BatchedGaussianRemlResult, String> {
    gaussian_reml_fit_positions_batched_streaming_impl(
        t,
        y,
        row_offsets,
        knots_or_centers,
        basis_kind,
        basis_order,
        periodic,
        period,
        penalty,
        weights,
        init_lambda,
        by,
        by_start_col,
    )
}

fn validate_position_batched_reml_common(
    t: ArrayView1<'_, f64>,
    y: ArrayView2<'_, f64>,
    row_offsets: ArrayView1<'_, usize>,
    knots_or_centers: ArrayView1<'_, f64>,
    penalty: ArrayView2<'_, f64>,
    weights: Option<ArrayView1<'_, f64>>,
    init_lambda: Option<f64>,
    by: Option<ArrayView1<'_, f64>>,
    by_start_col: usize,
) -> Result<(usize, usize, usize), String> {
    validate_vector("t", t)?;
    validate_vector("knots_or_centers", knots_or_centers)?;
    if row_offsets.len() < 2 {
        return Err("row_offsets must contain at least [0, n]".to_string());
    }
    if row_offsets[0] != 0 || row_offsets[row_offsets.len() - 1] != t.len() {
        return Err(format!(
            "row_offsets must start at 0 and end at t.len(); got start={}, end={}, n={}",
            row_offsets[0],
            row_offsets[row_offsets.len() - 1],
            t.len()
        ));
    }
    for idx in 0..row_offsets.len() - 1 {
        if row_offsets[idx] > row_offsets[idx + 1] {
            return Err("row_offsets must be non-decreasing".to_string());
        }
    }
    if y.nrows() != t.len() {
        return Err(format!(
            "position batched Gaussian REML row mismatch: t has {} rows but Y has {}",
            t.len(),
            y.nrows()
        ));
    }
    if y.ncols() == 0 {
        return Err("batched Gaussian REML requires non-empty Y columns".to_string());
    }
    if penalty.nrows() == 0 || penalty.ncols() != penalty.nrows() {
        return Err(format!(
            "penalty must be a non-empty square matrix; got {}x{}",
            penalty.nrows(),
            penalty.ncols()
        ));
    }
    if by_start_col > penalty.nrows() {
        return Err(format!(
            "by_start_col must be <= number of columns; got {by_start_col} for {} columns",
            penalty.nrows()
        ));
    }
    if y.iter()
        .chain(penalty.iter())
        .any(|value| !value.is_finite())
    {
        return Err("batched Gaussian REML inputs must be finite".to_string());
    }
    if let Some(weights) = weights {
        if weights.len() != t.len() {
            return Err(format!(
                "weights length mismatch: expected {}, got {}",
                t.len(),
                weights.len()
            ));
        }
        if weights
            .iter()
            .any(|value| !value.is_finite() || *value < 0.0)
        {
            return Err(
                "batched Gaussian REML weights must be finite non-negative values".to_string(),
            );
        }
    }
    if let Some(lambda) = init_lambda {
        if !lambda.is_finite() || lambda <= 0.0 {
            return Err(format!(
                "init_lambda must be finite and positive when provided; got {lambda}"
            ));
        }
    }
    if let Some(by) = by {
        if by.len() != t.len() {
            return Err(format!(
                "by gate length mismatch: expected {}, got {}",
                t.len(),
                by.len()
            ));
        }
        if by.iter().any(|value| !value.is_finite()) {
            return Err("by gate must contain only finite values".to_string());
        }
    }
    Ok((row_offsets.len() - 1, penalty.nrows(), y.ncols()))
}

fn position_fit_design_for_slice(
    t: ArrayView1<'_, f64>,
    knots_or_centers: ArrayView1<'_, f64>,
    basis_kind: &str,
    basis_order: usize,
    periodic: bool,
    period: Option<f64>,
    by: Option<ArrayView1<'_, f64>>,
    by_start_col: usize,
    expected_cols: usize,
    batch_index: usize,
) -> Result<Array2<f64>, String> {
    let x = position_basis_design(
        t,
        knots_or_centers,
        basis_kind,
        basis_order,
        periodic,
        period,
    )?;
    if x.ncols() != expected_cols {
        return Err(format!(
            "position basis width mismatch in batch {batch_index}: penalty expects {expected_cols}, basis produced {}",
            x.ncols()
        ));
    }
    if let Some(by_values) = by {
        apply_by_gate(x.view(), by_values, by_start_col)
    } else {
        Ok(x)
    }
}

fn gaussian_reml_fit_positions_batched_streaming_impl(
    t: ArrayView1<'_, f64>,
    y: ArrayView2<'_, f64>,
    row_offsets: ArrayView1<'_, usize>,
    knots_or_centers: ArrayView1<'_, f64>,
    basis_kind: &str,
    basis_order: usize,
    periodic: bool,
    period: Option<f64>,
    penalty: ArrayView2<'_, f64>,
    weights: Option<ArrayView1<'_, f64>>,
    init_lambda: Option<f64>,
    by: Option<ArrayView1<'_, f64>>,
    by_start_col: usize,
) -> Result<BatchedGaussianRemlResult, String> {
    use rayon::iter::{IntoParallelIterator, ParallelIterator};

    let (batch, p, d) = validate_position_batched_reml_common(
        t,
        y,
        row_offsets,
        knots_or_centers,
        penalty,
        weights,
        init_lambda,
        by,
        by_start_col,
    )?;

    // Fold the `by` gate into the prior weights ONCE for the whole batch, then
    // slice the gated array per segment. Zero-`by` rows get weight 0, so they
    // drop out of both the segment `XᵀWX` (cache build) and the segment fit
    // consistently — the cache fingerprint check requires the same weights in
    // both — and out of `ywy` / the effective-DoF `ν` in the solver (#2031).
    let gated_weights = gate_weights_for_forward(weights, by, t.len())?;
    let weights = gated_weights.as_ref().map(|w| w.view());

    // Build each ragged segment's basis independently. This makes it
    // impossible for the position-batched API to materialize the concatenated
    // n_total x p design, which is exactly the shape that becomes
    // operator-backed for large Duchon batches.
    let xtwx_phase: Vec<Result<Option<Array2<f64>>, String>> = (0..batch)
        .into_par_iter()
        .map(|b| {
            let start = row_offsets[b];
            let end = row_offsets[b + 1];
            if start == end {
                return Ok(None);
            }
            let x = position_fit_design_for_slice(
                t.slice(s![start..end]),
                knots_or_centers,
                basis_kind,
                basis_order,
                periodic,
                period,
                by.as_ref().map(|values| values.slice(s![start..end])),
                by_start_col,
                p,
                b,
            )?;
            let owned_weight: Array1<f64> = match weights.as_ref() {
                Some(w) => w.slice(s![start..end]).to_owned(),
                None => Array1::ones(end - start),
            };
            Ok(Some(gam::linalg::faer_ndarray::fast_xt_diag_x(
                &x.view(),
                &owned_weight,
            )))
        })
        .collect();

    let mut live_indices: Vec<usize> = Vec::with_capacity(batch);
    let mut live_xtwx: Vec<Array2<f64>> = Vec::with_capacity(batch);
    for (b, slot) in xtwx_phase.into_iter().enumerate() {
        if let Some(xtwx) = slot? {
            live_indices.push(b);
            live_xtwx.push(xtwx);
        }
    }
    let batched_caches = build_gaussian_reml_eigen_cache_batched(live_xtwx, penalty, None);
    let mut prebuilt_caches: Vec<Option<gam::solver::gaussian_reml::GaussianRemlEigenCache>> =
        (0..batch).map(|_| None).collect();
    for (i, cache_result) in batched_caches.into_iter().enumerate() {
        if let Ok(cache) = cache_result {
            prebuilt_caches[live_indices[i]] = Some(cache);
        }
    }

    let fit_results: Vec<
        Result<
            (
                usize,
                Option<gam::solver::gaussian_reml::GaussianRemlMultiResult>,
            ),
            String,
        >,
    > = (0..batch)
        .into_par_iter()
        .map(|b| {
            let start = row_offsets[b];
            let end = row_offsets[b + 1];
            if start == end {
                return Ok((b, None));
            }
            let x = position_fit_design_for_slice(
                t.slice(s![start..end]),
                knots_or_centers,
                basis_kind,
                basis_order,
                periodic,
                period,
                by.as_ref().map(|values| values.slice(s![start..end])),
                by_start_col,
                p,
                b,
            )?;
            let weight_slice = weights.as_ref().map(|w| w.slice(s![start..end]));
            let cache_ref = prebuilt_caches[b].as_ref();
            match gaussian_reml_multi_closed_form_with_cache(
                x.view(),
                y.slice(s![start..end, ..]),
                penalty,
                weight_slice,
                init_lambda,
                cache_ref,
            ) {
                Ok(result) => Ok((b, Some(result))),
                Err(EstimationError::ModelIsIllConditioned { .. }) => Ok((b, None)),
                Err(err) => Err(format!(
                    "batched position Gaussian REML fit {b} failed: {err}"
                )),
            }
        })
        .collect();

    let mut lambdas = Array1::<f64>::from_elem(batch, f64::NAN);
    let mut rhos = Array1::<f64>::from_elem(batch, f64::NAN);
    let mut reml_scores = Array1::<f64>::from_elem(batch, f64::NAN);
    let mut reml_grad_lambdas = Array1::<f64>::from_elem(batch, f64::NAN);
    let mut reml_hess_lambdas = Array1::<f64>::from_elem(batch, f64::NAN);
    let mut reml_grad_rhos = Array1::<f64>::from_elem(batch, f64::NAN);
    let mut reml_hess_rhos = Array1::<f64>::from_elem(batch, f64::NAN);
    let mut edf = Array1::<f64>::zeros(batch);
    let mut coefficients = Array3::<f64>::zeros((batch, p, d));
    let mut fitted = Array2::<f64>::zeros((t.len(), d));
    let mut sigma2 = Array2::<f64>::from_elem((batch, d), f64::NAN);
    let mut statuses = vec!["degenerate".to_string(); batch];
    let mut cache_penalty_eigenvalues = Array2::<f64>::zeros((batch, p));
    let mut cache_eigenvectors = Array3::<f64>::zeros((batch, p, p));
    let mut cache_coefficient_basis = Array3::<f64>::zeros((batch, p, p));
    let mut cache_xtwx_fingerprints = Array1::<u64>::zeros(batch);
    let mut cache_penalty_fingerprints = Array1::<u64>::zeros(batch);
    let mut cache_logdet_xtwx = Array1::<f64>::zeros(batch);
    let mut cache_logdet_penalty_positive = Array1::<f64>::zeros(batch);
    let mut cache_penalty_ranks = Array1::<i64>::zeros(batch);
    let mut cache_nullities = Array1::<i64>::zeros(batch);

    for result in fit_results {
        let (b, fit) = result?;
        if let Some(fit) = fit {
            let start = row_offsets[b];
            let end = row_offsets[b + 1];
            statuses[b] = if fit.lambda.is_finite() && fit.reml_score.is_finite() {
                "ok".to_string()
            } else {
                "diverged".to_string()
            };
            lambdas[b] = fit.lambda;
            rhos[b] = fit.rho;
            reml_scores[b] = fit.reml_score;
            reml_grad_lambdas[b] = fit.reml_grad_lambda;
            reml_hess_lambdas[b] = fit.reml_hess_lambda;
            reml_grad_rhos[b] = fit.reml_grad_rho;
            reml_hess_rhos[b] = fit.reml_hess_rho;
            edf[b] = fit.edf;
            coefficients
                .slice_mut(s![b, .., ..])
                .assign(&fit.coefficients);
            fitted.slice_mut(s![start..end, ..]).assign(&fit.fitted);
            sigma2.slice_mut(s![b, ..]).assign(&fit.sigma2);
            cache_penalty_eigenvalues
                .slice_mut(s![b, ..])
                .assign(&fit.cache.penalty_eigenvalues);
            cache_eigenvectors
                .slice_mut(s![b, .., ..])
                .assign(&fit.cache.eigenvectors);
            cache_coefficient_basis
                .slice_mut(s![b, .., ..])
                .assign(&fit.cache.coefficient_basis);
            cache_xtwx_fingerprints[b] = fit.cache.xtwx_fingerprint;
            cache_penalty_fingerprints[b] = fit.cache.penalty_fingerprint;
            cache_logdet_xtwx[b] = fit.cache.logdet_xtwx;
            cache_logdet_penalty_positive[b] = fit.cache.logdet_penalty_positive;
            cache_penalty_ranks[b] = fit.cache.penalty_rank as i64;
            cache_nullities[b] = fit.cache.nullity as i64;
        }
    }

    Ok(BatchedGaussianRemlResult {
        statuses,
        lambdas,
        rhos,
        reml_scores,
        reml_grad_lambdas,
        reml_hess_lambdas,
        reml_grad_rhos,
        reml_hess_rhos,
        edf,
        coefficients,
        fitted,
        sigma2,
        cache_penalty_eigenvalues,
        cache_eigenvectors,
        cache_coefficient_basis,
        cache_xtwx_fingerprints,
        cache_penalty_fingerprints,
        cache_logdet_xtwx,
        cache_logdet_penalty_positive,
        cache_penalty_ranks,
        cache_nullities,
    })
}

fn gaussian_reml_fit_positions_batched_backward_impl(
    t: ArrayView1<'_, f64>,
    y: ArrayView2<'_, f64>,
    row_offsets: ArrayView1<'_, usize>,
    knots_or_centers: ArrayView1<'_, f64>,
    basis_kind: &str,
    basis_order: usize,
    periodic: bool,
    period: Option<f64>,
    penalty: ArrayView2<'_, f64>,
    weights: Option<ArrayView1<'_, f64>>,
    init_lambda: Option<f64>,
    grad_lambda: Option<ArrayView1<'_, f64>>,
    grad_coefficients: Option<ArrayView3<'_, f64>>,
    grad_fitted: Option<ArrayView2<'_, f64>>,
    grad_reml_score: Option<ArrayView1<'_, f64>>,
    grad_edf: Option<ArrayView1<'_, f64>>,
    by: Option<ArrayView1<'_, f64>>,
    by_start_col: usize,
    forward_fits: Option<&[Option<gam::solver::gaussian_reml::GaussianRemlMultiResult>]>,
) -> Result<BatchedPositionGaussianRemlBackwardResult, String> {
    use rayon::iter::{IntoParallelIterator, ParallelIterator};

    let (batch, p, d) = validate_position_batched_reml_common(
        t,
        y,
        row_offsets,
        knots_or_centers,
        penalty,
        weights,
        init_lambda,
        by,
        by_start_col,
    )?;
    validate_batched_reml_upstreams(
        batch,
        p,
        d,
        t.len(),
        grad_lambda,
        grad_coefficients,
        grad_fitted,
        grad_reml_score,
        grad_edf,
    )?;

    if let Some(fits) = forward_fits {
        if fits.len() != batch {
            return Err(format!(
                "forward_state fit count mismatch: expected {batch}, got {}",
                fits.len()
            ));
        }
    }
    type PositionBackwardParts = (
        Array1<f64>,
        Array2<f64>,
        Array2<f64>,
        Array1<f64>,
        Option<Array1<f64>>,
    );
    let results: Vec<Result<(usize, Option<PositionBackwardParts>), String>> = (0..batch)
        .into_par_iter()
        .map(|b| {
            let start = row_offsets[b];
            let end = row_offsets[b + 1];
            if start == end {
                return Ok((b, None));
            }
            let weight_slice = weights.as_ref().map(|w| w.slice(s![start..end]));
            let upstream_lambda = grad_lambda.as_ref().map_or(0.0, |g| g[b]);
            let upstream_reml_score = grad_reml_score.as_ref().map_or(0.0, |g| g[b]);
            let upstream_edf = grad_edf.as_ref().map_or(0.0, |g| g[b]);
            let upstream_coefficients = grad_coefficients.as_ref().map(|g| g.slice(s![b, .., ..]));
            let upstream_fitted = grad_fitted.as_ref().map(|g| g.slice(s![start..end, ..]));
            let x_base = position_basis_design(
                t.slice(s![start..end]),
                knots_or_centers,
                basis_kind,
                basis_order,
                periodic,
                period,
            )?;
            if x_base.ncols() != p {
                return Err(format!(
                    "position basis width mismatch in batch {b}: penalty expects {p}, basis produced {}",
                    x_base.ncols()
                ));
            }
            let basis_derivative = position_basis_derivative(
                t.slice(s![start..end]),
                knots_or_centers,
                basis_kind,
                basis_order,
                periodic,
                period,
            )?;
            if basis_derivative.dim() != x_base.dim() {
                return Err(format!(
                    "basis derivative shape mismatch in batch {b}: basis is {}x{} but dX/dt is {}x{}",
                    x_base.nrows(),
                    x_base.ncols(),
                    basis_derivative.nrows(),
                    basis_derivative.ncols()
                ));
            }
            let by_slice = by.as_ref().map(|values| values.slice(s![start..end]));
            let gated_x = gate_design_for_forward(x_base.view(), by_slice, by_start_col)?;
            let x_slice = gated_x.as_ref().map_or(x_base.view(), |g| g.view());
            // Fold the `by` gate into this segment's weights so the backward's
            // effective sample size matches the forward's (#2031).
            let gated_weight_slice = gate_weights_for_forward(weight_slice, by_slice, end - start)?;
            let y_slice = y.slice(s![start..end, ..]);
            let backward_result = if let Some(fits) = forward_fits {
                match fits[b].as_ref() {
                    Some(fit) => gaussian_reml_multi_closed_form_backward_from_fit(
                        x_slice,
                        y_slice,
                        penalty,
                        gated_weight_slice.as_ref().map(|w| w.view()),
                        fit,
                        upstream_lambda,
                        upstream_coefficients,
                        upstream_fitted,
                        upstream_reml_score,
                        upstream_edf,
                    ),
                    None => return Ok((b, None)),
                }
            } else {
                gaussian_reml_multi_closed_form_backward(
                    x_slice,
                    y_slice,
                    penalty,
                    gated_weight_slice.as_ref().map(|w| w.view()),
                    init_lambda,
                    upstream_lambda,
                    upstream_coefficients,
                    upstream_fitted,
                    upstream_reml_score,
                    upstream_edf,
                )
            };
            match backward_result {
                Ok(backward) => {
                    let (grad_x, grad_by) =
                        ungate_design_gradient(x_base.view(), by_slice, by_start_col, backward.grad_x)?;
                    let grad_t = contract_position_gradient(grad_x.view(), basis_derivative.view())?;
                    Ok((
                        b,
                        Some((
                            grad_t,
                            backward.grad_y,
                            backward.grad_penalty,
                            ungate_weight_gradient(by_slice, backward.grad_weights),
                            grad_by,
                        )),
                    ))
                }
                Err(EstimationError::ModelIsIllConditioned { .. }) => Ok((b, None)),
                Err(err) => Err(format!(
                    "batched position Gaussian REML backward {b} failed: {err}"
                )),
            }
        })
        .collect();

    let mut statuses = vec!["degenerate".to_string(); batch];
    let mut grad_t = Array1::<f64>::zeros(t.len());
    let mut grad_y = Array2::<f64>::zeros(y.dim());
    let mut grad_penalty = Array2::<f64>::zeros(penalty.dim());
    let mut grad_weights = Array1::<f64>::zeros(t.len());
    let mut grad_by = by.map(|_| Array1::<f64>::zeros(t.len()));
    for result in results {
        let (b, backward) = result?;
        if let Some((
            batch_grad_t,
            batch_grad_y,
            batch_grad_penalty,
            batch_grad_weights,
            batch_grad_by,
        )) = backward
        {
            let start = row_offsets[b];
            let end = row_offsets[b + 1];
            statuses[b] = "ok".to_string();
            grad_t.slice_mut(s![start..end]).assign(&batch_grad_t);
            grad_y.slice_mut(s![start..end, ..]).assign(&batch_grad_y);
            grad_penalty += &batch_grad_penalty;
            grad_weights
                .slice_mut(s![start..end])
                .assign(&batch_grad_weights);
            if let (Some(target), Some(source)) = (grad_by.as_mut(), batch_grad_by.as_ref()) {
                target.slice_mut(s![start..end]).assign(source);
            }
        }
    }

    Ok(BatchedPositionGaussianRemlBackwardResult {
        statuses,
        grad_t,
        grad_y,
        grad_penalty,
        grad_weights,
        grad_by,
    })
}

fn position_basis_design(
    t: ArrayView1<'_, f64>,
    knots_or_centers: ArrayView1<'_, f64>,
    basis_kind: &str,
    basis_order: usize,
    periodic: bool,
    period: Option<f64>,
) -> Result<Array2<f64>, String> {
    match normalized_position_basis_kind(basis_kind)?.as_str() {
        "bspline" => {
            bspline_position_basis_impl(t, knots_or_centers, basis_order, periodic, period)
        }
        "duchon" => {
            validate_position_period("duchon", knots_or_centers, periodic, period)?;
            duchon_basis_1d_impl(t, knots_or_centers, basis_order, periodic, period)
        }
        other => Err(format!(
            "normalized_position_basis_kind returned an unsupported basis name: {other}"
        )),
    }
}

fn position_basis_derivative(
    t: ArrayView1<'_, f64>,
    knots_or_centers: ArrayView1<'_, f64>,
    basis_kind: &str,
    basis_order: usize,
    periodic: bool,
    period: Option<f64>,
) -> Result<Array2<f64>, String> {
    match normalized_position_basis_kind(basis_kind)?.as_str() {
        "bspline" => {
            bspline_position_derivative_impl(t, knots_or_centers, basis_order, 1, periodic, period)
        }
        "duchon" => {
            validate_position_period("duchon", knots_or_centers, periodic, period)?;
            duchon_basis_1d_derivative_impl(t, knots_or_centers, basis_order, 1, periodic, period)
        }
        other => Err(format!(
            "normalized_position_basis_kind returned an unsupported basis name: {other}"
        )),
    }
}

fn bspline_position_basis_impl(
    t: ArrayView1<'_, f64>,
    knots: ArrayView1<'_, f64>,
    degree: usize,
    periodic: bool,
    period: Option<f64>,
) -> Result<Array2<f64>, String> {
    if !periodic {
        validate_position_period("B-spline", knots, periodic, period)?;
        return bspline_basis_impl(t, knots, degree, false);
    }
    let (left, right, num_basis) = periodic_position_domain(knots, period)?;
    validate_vector("t", t)?;
    periodic_bspline_basis_dense_via_spec(t, (left, right), degree, num_basis)
}

fn bspline_position_derivative_impl(
    t: ArrayView1<'_, f64>,
    knots: ArrayView1<'_, f64>,
    degree: usize,
    order: usize,
    periodic: bool,
    period: Option<f64>,
) -> Result<Array2<f64>, String> {
    if !periodic {
        validate_position_period("B-spline", knots, periodic, period)?;
        return bspline_basis_derivative_impl(t, knots, degree, order, false);
    }
    let (left, right, num_basis) = periodic_position_domain(knots, period)?;
    validate_vector("t", t)?;
    periodic_bspline_derivative_dense(t, (left, right), degree, num_basis, order)
}

fn periodic_position_domain(
    knots_or_centers: ArrayView1<'_, f64>,
    period: Option<f64>,
) -> Result<(f64, f64, usize), String> {
    validate_vector("knots_or_centers", knots_or_centers)?;
    if knots_or_centers.len() < 2 {
        return Err("periodic position basis requires at least two knots or centers".to_string());
    }
    let Some(period) = period else {
        return Err(
            "periodic position basis requires an explicit finite positive period".to_string(),
        );
    };
    if !period.is_finite() || period <= 0.0 {
        return Err(format!(
            "periodic position basis period must be finite and positive; got {period}"
        ));
    }
    let left = knots_or_centers[0];
    let right = left + period;
    Ok((left, right, knots_or_centers.len() - 1))
}

fn validate_position_period(
    label: &str,
    knots_or_centers: ArrayView1<'_, f64>,
    periodic: bool,
    period: Option<f64>,
) -> Result<(), String> {
    if periodic {
        let left = knots_or_centers
            .iter()
            .fold(f64::INFINITY, |a, &b| a.min(b));
        let right = knots_or_centers
            .iter()
            .fold(f64::NEG_INFINITY, |a, &b| a.max(b));
        if !left.is_finite() || !right.is_finite() || left >= right {
            return Err(format!(
                "{label} periodic support must have increasing finite endpoints"
            ));
        }
        let implied = right - left;
        if let Some(period) = period {
            if !period.is_finite() || period <= 0.0 {
                return Err(format!(
                    "{label} period must be finite and positive; got {period}"
                ));
            }
            // The period is the domain WRAP, not the sample/center span. Centers
            // on a half-open grid [start, start+period) (e.g. linspace(0,1,K,
            // endpoint=False) with period 1.0) span only `period − one_spacing`,
            // so requiring span == period rejected every legitimate explicit
            // period (gam#580). The only real constraint is that every center
            // fits inside a single period, i.e. `period >= span`.
            if period < implied - 1.0e-10 * implied.max(1.0) {
                return Err(format!(
                    "{label} explicit period ({period}) is smaller than the center span \
                     ({implied}); every center must lie within a single period"
                ));
            }
        } else if label != "duchon" {
            return Err(format!(
                "{label} periodic position basis requires an explicit period"
            ));
        }
    } else if period.is_some() {
        return Err(format!("{label} period is only valid when periodic=true"));
    }
    Ok(())
}

fn normalized_position_basis_kind(basis_kind: &str) -> Result<String, String> {
    let normalized = basis_kind
        .trim()
        .to_ascii_lowercase()
        .replace(['_', '-'], "");
    match normalized.as_str() {
        "bspline" | "spline" => Ok("bspline".to_string()),
        "duchon" | "duchonspline" => Ok("duchon".to_string()),
        _ => Err(format!(
            "basis_kind must be 'bspline' or 'duchon'; got {basis_kind:?}"
        )),
    }
}

fn contract_position_gradient(
    grad_x: ArrayView2<'_, f64>,
    basis_derivative: ArrayView2<'_, f64>,
) -> Result<Array1<f64>, String> {
    if grad_x.dim() != basis_derivative.dim() {
        return Err(format!(
            "basis derivative shape mismatch: grad_x is {}x{} but dX/dt is {}x{}",
            grad_x.nrows(),
            grad_x.ncols(),
            basis_derivative.nrows(),
            basis_derivative.ncols()
        ));
    }
    let mut grad_t = Array1::<f64>::zeros(grad_x.nrows());
    for row in 0..grad_x.nrows() {
        grad_t[row] = grad_x.row(row).dot(&basis_derivative.row(row));
    }
    Ok(grad_t)
}

fn apply_by_gate(
    x: ArrayView2<'_, f64>,
    by: ArrayView1<'_, f64>,
    start_col: usize,
) -> Result<Array2<f64>, String> {
    if by.len() != x.nrows() {
        return Err(format!(
            "by gate length mismatch: expected {}, got {}",
            x.nrows(),
            by.len()
        ));
    }
    if start_col > x.ncols() {
        return Err(format!(
            "by_start_col must be <= number of columns; got {start_col} for {} columns",
            x.ncols()
        ));
    }
    if by.iter().any(|value| !value.is_finite()) {
        return Err("by gate must contain only finite values".to_string());
    }
    let mut out = x.to_owned();
    for row in 0..out.nrows() {
        let gate = by[row];
        for col in start_col..out.ncols() {
            out[[row, col]] *= gate;
        }
    }
    Ok(out)
}

fn apply_by_gate_backward(
    x: ArrayView2<'_, f64>,
    by: ArrayView1<'_, f64>,
    start_col: usize,
    grad_gated_x: ArrayView2<'_, f64>,
) -> Result<(Array2<f64>, Array1<f64>), String> {
    if grad_gated_x.dim() != x.dim() {
        return Err(format!(
            "by gate backward gradient shape mismatch: expected {}x{}, got {}x{}",
            x.nrows(),
            x.ncols(),
            grad_gated_x.nrows(),
            grad_gated_x.ncols()
        ));
    }
    if by.len() != x.nrows() {
        return Err(format!(
            "by gate length mismatch: expected {}, got {}",
            x.nrows(),
            by.len()
        ));
    }
    if start_col > x.ncols() {
        return Err(format!(
            "by_start_col must be <= number of columns; got {start_col} for {} columns",
            x.ncols()
        ));
    }
    let mut grad_x = grad_gated_x.to_owned();
    let mut grad_by = Array1::<f64>::zeros(x.nrows());
    for row in 0..x.nrows() {
        let gate = by[row];
        for col in start_col..x.ncols() {
            grad_x[[row, col]] = grad_gated_x[[row, col]] * gate;
            grad_by[row] += grad_gated_x[[row, col]] * x[[row, col]];
        }
    }
    Ok((grad_x, grad_by))
}

/// Materialize the `by`-gated design for a forward solve, if a gate is present.
///
/// Returns `None` when there is no gate so the caller fits against the raw
/// `x` view directly; returns `Some(gated)` otherwise. Centralizing this here
/// keeps every Gaussian REML forward `#[pyfunction]` from re-implementing the
/// `let gated; let fit_x = if let Some(by) = ... else x.view()` lifetime
/// dance. The owned array is held by the caller so the fit view borrows from
/// a binding that outlives the solve. The error type is `String`; the typed
/// (`EstimationError`) call sites wrap it with `EstimationError::InvalidInput`.
fn gate_design_for_forward(
    x: ArrayView2<'_, f64>,
    by: Option<ArrayView1<'_, f64>>,
    by_start_col: usize,
) -> Result<Option<Array2<f64>>, String> {
    match by {
        Some(by_arr) => Ok(Some(apply_by_gate(x, by_arr, by_start_col)?)),
        None => Ok(None),
    }
}

/// Fold the `by` gate into the REML row weights for a forward solve.
///
/// The `by` gate scales the modulated design columns by `by[row]`, so a row
/// with `by[row] == 0` has an all-zero modulated design and cannot move the
/// coefficients at a fixed λ. But the response energy `Σ w·y²` and the residual
/// degrees of freedom `ν` counted such rows, letting a gated-off row's response
/// leak into `σ²`, `λ`, and — through λ — the coefficients, contradicting the
/// documented contract that zero-`by` rows are inert (#2031). Honoring "the
/// REML fit is also weighted by `by`", we zero the prior weight of every
/// `by == 0` row: the response energy and the effective-sample-size DoF
/// (`effective_observation_count` in `gam-solve`) then both exclude it, so a
/// zero-`by` row is a complete no-op.
///
/// Rows with a nonzero `by` keep their prior weight unchanged, so a fit whose
/// `by` is entirely nonzero is byte-identical to manually gating the design and
/// passing the raw weights (the pre-existing, tested contract). When there is
/// no gate, or the gate has no zeros, the caller's original weights are
/// returned verbatim (preserving `None`) so those paths are untouched.
///
/// The owned array is held by the caller so the solver's weight view borrows a
/// binding that outlives the solve, mirroring [`gate_design_for_forward`]. The
/// error type is `String`; typed call sites wrap it with
/// `EstimationError::InvalidInput`.
fn gate_weights_for_forward(
    weights: Option<ArrayView1<'_, f64>>,
    by: Option<ArrayView1<'_, f64>>,
    n_rows: usize,
) -> Result<Option<Array1<f64>>, String> {
    let Some(by_arr) = by else {
        return Ok(weights.map(|w| w.to_owned()));
    };
    if by_arr.len() != n_rows {
        return Err(format!(
            "by gate length mismatch: expected {n_rows}, got {}",
            by_arr.len()
        ));
    }
    if !by_arr.iter().any(|&gate| gate == 0.0) {
        // No gated-off rows: the effective weights equal the prior weights, so
        // preserve the caller's exact argument (including `None`) for
        // byte-identical behavior with the pre-#2031 path.
        return Ok(weights.map(|w| w.to_owned()));
    }
    let mut gated = match weights {
        Some(w) => {
            if w.len() != n_rows {
                return Err(format!(
                    "weights length mismatch: expected {n_rows}, got {}",
                    w.len()
                ));
            }
            w.to_owned()
        }
        None => Array1::ones(n_rows),
    };
    for (row, &gate) in by_arr.iter().enumerate() {
        if gate == 0.0 {
            gated[row] = 0.0;
        }
    }
    Ok(Some(gated))
}

/// Route the effective-weight gradient back to the raw prior weights.
///
/// The forward folds the `by` gate into the weights as `w_eff = w · [by ≠ 0]`
/// (see [`gate_weights_for_forward`]), so `∂L/∂w = (∂L/∂w_eff) · [by ≠ 0]`: a
/// zero-`by` row's prior weight has no effect on the fit and therefore zero
/// gradient. The mask is an indicator with zero derivative almost everywhere,
/// so it contributes nothing to `grad_by` (which the design gate produces in
/// full). With no gate the gradient passes through unchanged.
fn ungate_weight_gradient(
    by: Option<ArrayView1<'_, f64>>,
    mut grad_weights: Array1<f64>,
) -> Array1<f64> {
    if let Some(by_arr) = by {
        let limit = by_arr.len().min(grad_weights.len());
        for row in 0..limit {
            if by_arr[row] == 0.0 {
                grad_weights[row] = 0.0;
            }
        }
    }
    grad_weights
}

/// Route a raw design-gradient back through the `by` gate, if one was applied.
///
/// `grad_x_raw` is the gradient w.r.t. the (possibly gated) design that went
/// into the solver. With no gate this is already the gradient w.r.t. the raw
/// design and `grad_by` is absent; with a gate this splits the cotangent into
/// `(grad_x, grad_by)` via the gate's analytic backward. Collapses the
/// repeated `(grad_x, grad_by)` reconstruction across every Gaussian REML
/// backward `#[pyfunction]`. The error type is `String`; typed call sites
/// wrap it with `EstimationError::InvalidInput`.
fn ungate_design_gradient(
    x: ArrayView2<'_, f64>,
    by: Option<ArrayView1<'_, f64>>,
    by_start_col: usize,
    grad_x_raw: Array2<f64>,
) -> Result<(Array2<f64>, Option<Array1<f64>>), String> {
    match by {
        Some(by_arr) => {
            let (grad_x, grad_by) =
                apply_by_gate_backward(x, by_arr, by_start_col, grad_x_raw.view())?;
            Ok((grad_x, Some(grad_by)))
        }
        None => Ok((grad_x_raw, None)),
    }
}

fn owned_row_major_f64(values: ArrayView2<'_, f64>) -> Array2<f64> {
    Array2::from_shape_fn(values.dim(), |index| values[index])
}

fn posterior_bands_payload_to_py(
    py: Python<'_>,
    payload: PosteriorPredictBandsPayload,
) -> PyResult<Py<PyDict>> {
    let out = PyDict::new(py);
    out.set_item(
        "linear_predictor",
        payload.linear_predictor.into_pyarray(py),
    )?;
    out.set_item(
        "linear_predictor_lower",
        payload.linear_predictor_lower.into_pyarray(py),
    )?;
    out.set_item(
        "linear_predictor_upper",
        payload.linear_predictor_upper.into_pyarray(py),
    )?;
    out.set_item("mean", payload.mean.into_pyarray(py))?;
    out.set_item("mean_lower", payload.mean_lower.into_pyarray(py))?;
    out.set_item("mean_upper", payload.mean_upper.into_pyarray(py))?;
    out.set_item("model_class", payload.model_class)?;
    out.set_item("family_kind", payload.family_kind)?;
    Ok(out.unbind())
}

fn posterior_predict_result_to_py(
    py: Python<'_>,
    result: PosteriorPredictResult,
) -> PyResult<Py<PyDict>> {
    let out = PyDict::new(py);
    out.set_item("eta", result.eta.into_pyarray(py))?;
    out.set_item("mean", result.mean.into_pyarray(py))?;
    out.set_item("model_class", result.model_class)?;
    out.set_item("family_kind", result.family_kind)?;
    out.set_item("link_spec", result.link_spec)?;
    Ok(out.unbind())
}

#[pyfunction]
fn posterior_predict_table(
    py: Python<'_>,
    model_bytes: Vec<u8>,
    headers: Vec<String>,
    rows: PyRef<'_, PyEncodedTable>,
    samples: PyReadonlyArray2<'_, f64>,
) -> PyResult<Py<PyDict>> {
    rows.require_headers(&headers).map_err(py_value_error)?;
    let dataset = rows.dataset.clone();
    let samples = owned_row_major_f64(samples.as_array());
    let result = detach_py_result(py, "posterior_predict_table", move || {
        posterior_predict_encoded_table_impl(&model_bytes, dataset, samples)
    })?;
    posterior_predict_result_to_py(py, result)
}

#[pyfunction]
fn posterior_predict_bands_table(
    py: Python<'_>,
    model_bytes: Vec<u8>,
    headers: Vec<String>,
    rows: PyRef<'_, PyEncodedTable>,
    samples: PyReadonlyArray2<'_, f64>,
    level: f64,
) -> PyResult<Py<PyDict>> {
    rows.require_headers(&headers).map_err(py_value_error)?;
    let dataset = rows.dataset.clone();
    let samples = owned_row_major_f64(samples.as_array());
    let payload = detach_py_result(py, "posterior_predict_bands_table", move || {
        posterior_predict_bands_encoded_table_impl(
            &model_bytes,
            dataset,
            samples,
            level,
        )
    })?;
    posterior_bands_payload_to_py(py, payload)
}

#[pyfunction]
fn posterior_draw_bands(
    py: Python<'_>,
    eta: PyReadonlyArray2<'_, f64>,
    mean: PyReadonlyArray2<'_, f64>,
    level: f64,
) -> PyResult<Py<PyDict>> {
    let eta = owned_row_major_f64(eta.as_array());
    let mean = owned_row_major_f64(mean.as_array());
    let payload = detach_py_result(py, "posterior_draw_bands", move || {
        posterior_draw_bands_impl(eta, mean, level)
    })?;
    posterior_bands_payload_to_py(py, payload)
}

#[pyfunction]
#[pyo3(signature = (eta, family_kind, level, link_spec=None))]
fn posterior_eta_bands(
    py: Python<'_>,
    eta: PyReadonlyArray2<'_, f64>,
    family_kind: String,
    level: f64,
    link_spec: Option<String>,
) -> PyResult<Py<PyDict>> {
    let eta = owned_row_major_f64(eta.as_array());
    let payload = detach_py_result(py, "posterior_eta_bands", move || {
        posterior_eta_bands_impl(eta, &family_kind, level, link_spec.as_deref())
    })?;
    posterior_bands_payload_to_py(py, payload)
}

#[pyfunction]
fn posterior_credible_interval(
    py: Python<'_>,
    samples: PyReadonlyArray2<'_, f64>,
    level: f64,
) -> PyResult<Py<PyArray2<f64>>> {
    let samples = owned_row_major_f64(samples.as_array());
    let intervals = detach_py_result(py, "posterior_credible_interval", move || {
        posterior_credible_interval_impl(samples, level)
    })?;
    Ok(intervals.into_pyarray(py).unbind())
}

#[pyfunction]
fn posterior_coefficient_names_json(request_json: &str) -> PyResult<String> {
    posterior_coefficient_names_json_impl(request_json).map_err(py_value_error)
}

#[pyfunction]
fn posterior_trace_selection_json(request_json: &str) -> PyResult<String> {
    posterior_trace_selection_json_impl(request_json).map_err(py_value_error)
}

#[pyfunction]
#[pyo3(signature = (eta, family_kind, link_spec=None))]
fn apply_inverse_link_array(
    py: Python<'_>,
    eta: PyReadonlyArray2<'_, f64>,
    family_kind: String,
    link_spec: Option<String>,
) -> PyResult<Py<PyArray2<f64>>> {
    let eta = owned_row_major_f64(eta.as_array());
    let shape = eta.dim();
    let values = eta.into_raw_vec_and_offset().0;
    let transformed = detach_py_result(py, "apply_inverse_link_array", move || {
        apply_inverse_link_with_optional_spec(&values, &family_kind, link_spec.as_deref())
    })?;
    let result = Array2::from_shape_vec(shape, transformed)
        .map_err(|err| py_value_error(format!("failed to shape inverse-link result: {err}")))?;
    Ok(result.into_pyarray(py).unbind())
}

/// Apply the inverse link to `eta`, preferring the typed `link_spec` (a
/// serialized [`InverseLink`]) when present. The parameterized links (`Sas`,
/// `Mixture`, `LatentCLogLog`, `BetaLogistic`) carry per-fit state the bare
/// `family_kind` tag drops, so they are only fully evaluable via the spec; when
/// no spec is supplied the call falls back to the string-tag path, which still
/// covers every `Standard` link (issue #1133).
fn apply_inverse_link_with_optional_spec(
    eta: &[f64],
    family_kind: &str,
    link_spec: Option<&str>,
) -> Result<Vec<f64>, String> {
    if let Some(spec_json) = link_spec {
        let link: InverseLink = serde_json::from_str(spec_json)
            .map_err(|err| format!("failed to parse link_spec for inverse link: {err}"))?;
        return apply_inverse_link_spec_vec(eta, &link);
    }
    apply_inverse_link_vec(eta, family_kind)
}

#[pyfunction]
fn summary_json(py: Python<'_>, model_bytes: Vec<u8>) -> PyResult<String> {
    detach_py_result(py, "summary_json", move || summary_json_impl(&model_bytes))
}

/// #944 curvature-as-an-estimand report for every `curv(...)` constant-curvature
/// smooth in a fitted model: κ̂, its profile-likelihood CI, the interior κ = 0
/// likelihood-ratio flatness test, and the sign-of-CI geometry verdict.
///
/// Unlike `summary_json` (κ̂ only — a pure read), the CI and flatness test
/// re-profile the criterion `V_p(κ)` and so need the model's training data
/// (`headers`/`rows` for its training formula). The fitted κ̂ comes from the
/// model's saved (frozen) `resolved_termspec`; the data only supply the
/// responses/weights/offset the per-κ profile refits need.
#[pyfunction]
fn curvature_inference_json(
    py: Python<'_>,
    model_bytes: Vec<u8>,
    headers: Vec<String>,
    rows: PyRef<'_, PyEncodedTable>,
    level: Option<f64>,
) -> PyResult<String> {
    rows.require_headers(&headers).map_err(py_value_error)?;
    let dataset = rows.dataset.clone();
    detach_py_result(py, "curvature_inference_json", move || {
        curvature_inference_dataset_json_impl(&model_bytes, dataset, level.unwrap_or(0.95))
    })
}

/// #1063 per-term LR significance report for every penalized smooth term:
/// `statistic_lr`, `ref_df`, `bartlett_factor`,
/// `bartlett_factor_conditional`, `rho_variation_shift`,
/// `statistic_corrected`, `p_value_uncorrected`, `p_value_corrected`, and
/// `correction_provenance` (`"lawley_lr_estimated_lambda"` |
/// `"lawley_lr_fixed_lambda"` | `"none"`).
///
/// Unlike `summary_json` (Wood rank-truncated **Wald** χ²), this computes a
/// genuine **likelihood-ratio** statistic by a constrained refit dropping each
/// smooth term, then magic-Bartlett-corrects it with the exact Lawley factor.
/// Needs the model's training data (`headers`/`rows`) for the per-term null
/// refits, exactly as `curvature_inference_json` does.
#[pyfunction]
fn smooth_term_lr_inference_json(
    py: Python<'_>,
    model_bytes: Vec<u8>,
    headers: Vec<String>,
    rows: PyRef<'_, PyEncodedTable>,
) -> PyResult<String> {
    rows.require_headers(&headers).map_err(py_value_error)?;
    let dataset = rows.dataset.clone();
    detach_py_result(py, "smooth_term_lr_inference_json", move || {
        smooth_term_lr_inference_dataset_json_impl(&model_bytes, dataset)
    })
}

/// #1055 Riesz-representer debiased / Neyman-orthogonal estimate of a smooth
/// functional of a fitted standard GAM model. Requires training `data` to
/// compute per-row score contributions; eligible for Gaussian-identity and any
/// model whose saved fit carries a dense penalized Hessian and weighted Gram.
///
/// `target_spec_json` is a JSON object with:
/// * `"target"` — one of `"point"`, `"contrast"`, `"average_derivative"`,
///   `"average_value"`, `"linear"`.
/// * For `"point"` / `"linear"`: `"x0"` — the query row dict (same column
///   names as training data); the design row at `x0` is evaluated from the
///   saved model's design map.
/// * For `"contrast"`: `"x0"` and `"x1"` — two query row dicts; the
///   functional is `m(x0) − m(x1)`.
/// * For `"average_value"`: uses the value design matrix at the training data
///   rows (already materialized from `headers`/`rows`).
/// * For `"average_derivative"`: uses the basis-DERIVATIVE design `∂φ_j/∂x` at
///   the training rows (built analytically by differentiating each smooth term's
///   basis in closed form and replaying the same frozen identifiability chart),
///   differentiating with respect to the model's single smooth covariate, or the
///   column named by an optional `"deriv_var"` key when the model has smooths
///   over more than one covariate.
/// * Both averaged targets accept an optional `"weights"` key: a list of
///   per-row weights.
///
/// Returns `{"target", "theta_plugin", "theta_debiased", "se",
/// "penalty_bias", "ci_lower", "ci_upper", "ci_level"}`.
#[pyfunction]
fn model_debiased_functional_json(
    py: Python<'_>,
    model_bytes: Vec<u8>,
    headers: Vec<String>,
    rows: PyRef<'_, PyEncodedTable>,
    target_spec_json: String,
) -> PyResult<String> {
    rows.require_headers(&headers).map_err(py_value_error)?;
    let dataset = rows.dataset.clone();
    detach_py_result(py, "model_debiased_functional_json", move || {
        model_debiased_functional_dataset_json_impl(&model_bytes, dataset, &target_spec_json)
    })
}

/// Build the design row `X(x0)` for a single `point`/`contrast` query under the
/// model's FULL training-schema column layout — the same order as the training
/// design `standard.data` that the saved [`TermCollectionSpec`] indexes into.
///
/// The saved spec resolves each term's feature column by its TRAINING-schema
/// offset (a smooth over the 2nd training column reads data column 1), so the
/// query design must be laid out against that full schema, NOT just the keys of
/// `x0`. Encoding only the `x0` keys put the predictor at position 0 of a
/// 1-column frame, so a model trained from a `[y, x]` frame asked for column 1
/// of 1 column and aborted with "feature column out of bounds" whenever the
/// response column preceded the predictor (#1621) — a silent, column-order
/// dependent failure of `point`/`contrast` for the natural `{y, x}` dataframe.
///
/// `training_headers` is the exact column order the training rows were ingested
/// in (which `standard.data` inherits verbatim). Columns named in `x0` take
/// their query value; every other column — the response and any unreferenced
/// bookkeeping column — is filled with a neutral placeholder that never enters
/// the mean design (the design reads only predictor / `by=` columns; the family
/// is already validated Gaussian/identity, so a continuous-response placeholder
/// always encodes). Every required prediction column must be supplied in `x0`,
/// mirroring `predict`'s input contract, so the plug-in equals `predict(x0)`.
fn debiased_query_design_full_schema(
    model: &FittedModel,
    training_headers: &[String],
    x0: &serde_json::Map<String, serde_json::Value>,
    spec: &TermCollectionSpec,
    label: &str,
) -> Result<ndarray::Array1<f64>, String> {
    // A partial `x0` cannot define m(x0); it would silently evaluate the
    // unspecified smooths at a placeholder. Reject it up front with a clear
    // message instead of the old out-of-bounds abort.
    let required = required_prediction_columns(model)?;
    let mut missing: Vec<String> = required
        .iter()
        .filter(|name| !x0.contains_key(name.as_str()))
        .cloned()
        .collect();
    if !missing.is_empty() {
        missing.sort();
        return Err(format!(
            "debiased_functional: {label} must specify a value for every model predictor \
             column; missing {missing:?}"
        ));
    }
    let field_for = |value: &serde_json::Value| -> String {
        value
            .as_f64()
            .map(|v| format!("{v}"))
            .or_else(|| value.as_str().map(|s| s.to_string()))
            .unwrap_or_default()
    };
    // One row in the FULL training-header order; "0" placeholder for any column
    // `x0` does not name (response + any extra column). The placeholder never
    // reaches the mean design.
    let row: Vec<String> = training_headers
        .iter()
        .map(|h| x0.get(h).map(field_for).unwrap_or_else(|| "0".to_string()))
        .collect();
    let headers_vec = training_headers.to_vec();
    // Encode against the SAVED schema (correct categorical level codes) WITHOUT
    // the predict-time column projection, so the encoded width and order match
    // `standard.data` exactly — the invariant `build_term_collection_design`
    // relies on.
    let records = string_records_from_rows(&headers_vec, std::slice::from_ref(&row))?;
    let schema = model.require_data_schema()?;
    // Lenient (unseen-OK) encoding for every column that did NOT receive a real
    // query value — i.e. every column not in `required`, which is exactly the
    // set carrying the neutral "0" placeholder (the response + any unreferenced
    // bookkeeping column). Without this, a training frame that carried an
    // unrelated CATEGORICAL bookkeeping column (e.g. a `group`/`color`/`id`
    // label fit under `y ~ s(x)`) would strict-encode that column's "0"
    // placeholder against the saved levels and abort with `unseen level '0'`,
    // re-introducing the very #840 foot-gun `predict` avoids by projecting the
    // frame to the model's columns (`project_frame_to_model_columns`). The
    // placeholder never reaches the mean design, so encoding it as an unknown
    // level is harmless and order-preserving. Random-effect group columns stay
    // lenient too (they may be required yet still want the held-out-group
    // policy), matching the predict-time encode.
    let mut lenient: std::collections::HashSet<String> = training_headers
        .iter()
        .filter(|h| !required.contains(h.as_str()))
        .cloned()
        .collect();
    lenient.extend(model.random_effect_group_columns());
    let policy = UnseenCategoryPolicy::encode_unknown_for_columns(lenient);
    let q_dataset = encode_recordswith_schema(headers_vec, records, schema, policy)?;
    let q_design = build_term_collection_design(q_dataset.values.view(), spec)
        .map_err(|e| format!("debiased_functional: {label} design failed: {e}"))?;
    let qx = q_design
        .design
        .try_to_dense_arc(&format!("debiased_functional {label}"))
        .map_err(|e| format!("debiased_functional: {label} densification failed: {e}"))?;
    if qx.nrows() != 1 {
        return Err(format!(
            "debiased_functional: {label} query produced {} design rows, expected 1",
            qx.nrows()
        ));
    }
    Ok(qx.row(0).to_owned())
}

fn model_debiased_functional_dataset_json_impl(
    model_bytes: &[u8],
    dataset: EncodedDataset,
    target_spec_json: &str,
) -> Result<String, String> {
    use gam::inference::riesz::{RieszInput, SmoothFunctional, debias_with_dense_hessian};

    let model = load_model_impl(model_bytes)?;
    let formula = model.payload().formula.clone();

    // Only standard (non-survival, non-marginal-slope) models supported: they
    // carry a dense penalized Hessian + weighted Gram + coefficient vector.
    if !matches!(model.predict_model_class(), PredictModelClass::Standard) {
        return Err(format!(
            "debiased_functional: only standard GAM models are supported; got '{}'",
            prediction_model_class_label(&model)
        ));
    }

    let spec = model
        .payload()
        .resolved_termspec
        .as_ref()
        .ok_or_else(|| {
            "debiased_functional: model is missing resolved_termspec; refit to enable".to_string()
        })?
        .clone();

    // Preserve the exact training-frame column order before the dataset is moved
    // into the encoder. `standard.data` (below) inherits this order verbatim
    // (`StandardFitRequest::data == dataset.values`, no reordering), and the
    // saved `TermCollectionSpec` resolves every term's feature column by its
    // offset in THIS layout. The `point`/`contrast` query design must be built
    // against the same full layout, not just the columns named in `x0` (#1621).
    let training_headers = dataset.headers.clone();
    let (fit_config, _) = parse_fit_config(None)?;
    let materialized = materialize(&formula, &dataset, &fit_config).map_err(|e| format!("{e}"))?;
    let standard = match materialized.request {
        FitRequest::Standard(req) => req,
        _ => {
            return Err(
                "debiased_functional: formula materialized to a non-standard fit path".to_string(),
            );
        }
    };

    // Rebuild the design from the saved termspec + training data so we get the
    // same coefficient-space columns the fit used.
    let design_built = build_term_collection_design(standard.data.view(), &spec)
        .map_err(|e| format!("debiased_functional: design rebuild failed: {e}"))?;
    let x = design_built
        .design
        .try_to_dense_arc("debiased_functional design")
        .map_err(|e| format!("debiased_functional: design densification failed: {e}"))?;

    // Recover fitted beta, H, and X'WX from the saved model.
    let saved_fit =
        gam::families::survival::predict::fit_result_from_saved_model_for_prediction(&model)
            .map_err(|e| format!("debiased_functional: {e}"))?;
    let h = saved_fit.penalized_hessian().ok_or_else(|| {
        "debiased_functional: model does not carry a dense penalized Hessian; \
         refit with a smaller basis (dense fits only)"
            .to_string()
    })?;
    // Gaussian/identity is the only supported family (enforced below for the
    // score chain). Hoist that check here because the weighted-Gram fallback
    // (#1622) is only valid for the profiled-Gaussian weight convention.
    let family = model.likelihood();
    let is_gaussian_identity = matches!(family.response, ResponseFamily::Gaussian)
        && matches!(family.link, InverseLink::Standard(StandardLink::Identity));
    // Fast path: the weighted Gram X'WX was stored by the REML posterior block.
    // Fallback (#1622): the parametric-term fit path leaves `weighted_gram` at
    // its `None` default, so reconstruct X'WX from the already-rebuilt dense
    // design. For a profiled Gaussian/identity fit the stored penalized Hessian
    // is H = XᵀWX + S(λ) with W = diag(prior weights) (scale-free; the penalty
    // is added UNSCALED — see optimizer.rs `cov_scale` contract), so
    // X'WX = Xᵀ diag(w) X exactly and S(λ) = H − X'WX is recovered consistently
    // (for an unpenalized `y ~ x` this gives S(λ)=0, i.e. X'WX == H).
    let xwx_owned: ndarray::Array2<f64> = match saved_fit.weighted_gram() {
        Some(g) => g.clone(),
        None => {
            if !is_gaussian_identity {
                return Err(format!(
                    "debiased_functional: model does not carry the weighted Gram X'WX and \
                     it can only be reconstructed for Gaussian/identity models; this model \
                     uses family='{}'",
                    family.pretty_name()
                ));
            }
            let xref = x.as_ref();
            let w = standard.weights.view();
            if w.len() != xref.nrows() {
                return Err(format!(
                    "debiased_functional: prior-weight length {} does not match design rows {}",
                    w.len(),
                    xref.nrows()
                ));
            }
            // Xᵀ diag(w) X = (diag(w) X)ᵀ X.
            let mut wx = xref.to_owned();
            for (mut row, &wi) in wx.outer_iter_mut().zip(w.iter()) {
                row.mapv_inplace(|v| v * wi);
            }
            wx.t().dot(xref)
        }
    };
    let xwx = &xwx_owned;
    let beta = saved_fit.beta_flat();
    if beta.len() != x.ncols() {
        return Err(format!(
            "debiased_functional: beta length {} does not match design width {}",
            beta.len(),
            x.ncols()
        ));
    }

    // Penalty gradient S_lambda × beta = (H − X'WX) × beta.
    let s_lambda = h.clone() - xwx.clone();
    let penalty_beta = s_lambda.dot(&beta);

    // Per-row score contributions ∂nll_i/∂β.
    // For a Gaussian identity model: ∂nll_i/∂β = x_i · (η_i − y_i).
    // Other families need their own derivative chain; currently restricted to
    // Gaussian/identity where the score is exact and the debiasing is cleanest.
    let y = standard.y.view();
    let n = x.nrows();
    let p = x.ncols();
    if !is_gaussian_identity {
        return Err(format!(
            "debiased_functional: currently only supported for Gaussian/identity models; \
             this model uses family='{}'. Supply pre-computed row_scores via the low-level \
             gamfit._rust.debiased_functional() call for other families.",
            family.pretty_name()
        ));
    }
    let eta = x.as_ref().dot(&beta);
    let mut row_scores = ndarray::Array2::<f64>::zeros((n, p));
    for i in 0..n {
        let residual = eta[i] - y[i]; // ∂nll_i/∂η = η_i − y_i for Gaussian/identity
        let x_row = x.row(i);
        for j in 0..p {
            row_scores[[i, j]] = x_row[j] * residual;
        }
    }

    // Parse target spec.
    let spec_val: serde_json::Value = serde_json::from_str(target_spec_json)
        .map_err(|e| format!("debiased_functional: invalid target_spec_json: {e}"))?;
    let target = spec_val
        .get("target")
        .and_then(|v| v.as_str())
        .ok_or_else(|| {
            "debiased_functional: target_spec_json must contain \"target\"".to_string()
        })?;

    // Build the functional gradient g = dθ/dβ from the spec.
    let gradient: ndarray::Array1<f64> = match target {
        "point" | "linear" => {
            // Requires an "x0" row dict → evaluate the design at x0, built under
            // the FULL training schema so the saved spec's feature-column offsets
            // resolve correctly regardless of where the response sits (#1621).
            let x0_obj = spec_val
                .get("x0")
                .ok_or_else(|| {
                    format!("debiased_functional: target \"{target}\" requires \"x0\" in spec")
                })?
                .as_object()
                .ok_or_else(|| "debiased_functional: \"x0\" must be an object".to_string())?;
            debiased_query_design_full_schema(&model, &training_headers, x0_obj, &spec, "x0")?
        }
        "contrast" => {
            let get_row = |key: &str| -> Result<ndarray::Array1<f64>, String> {
                let row_obj = spec_val
                    .get(key)
                    .ok_or_else(|| {
                        format!(
                            "debiased_functional: target \"contrast\" requires \"{key}\" in spec"
                        )
                    })?
                    .as_object()
                    .ok_or_else(|| format!("debiased_functional: \"{key}\" must be an object"))?;
                debiased_query_design_full_schema(&model, &training_headers, row_obj, &spec, key)
            };
            let row_a = get_row("x0")?;
            let row_b = get_row("x1")?;
            SmoothFunctional::Contrast {
                design_row_a: row_a.view(),
                design_row_b: row_b.view(),
            }
            .gradient()
            .map_err(|e| format!("debiased_functional: contrast gradient: {e}"))?
        }
        "average_derivative" | "average_value" => {
            // Uses the full training design; optional per-row weights from spec.
            let weights: Option<ndarray::Array1<f64>> = spec_val
                .get("weights")
                .and_then(|v| v.as_array())
                .map(|arr| {
                    arr.iter()
                        .map(|v| v.as_f64().unwrap_or(1.0))
                        .collect::<Vec<_>>()
                })
                .map(ndarray::Array1::from);
            let x_ref = x.as_ref();
            if target == "average_value" {
                SmoothFunctional::AverageValue {
                    value_design: x_ref.view(),
                    weights: weights.as_ref().map(|w| w.view()),
                }
                .gradient()
                .map_err(|e| format!("debiased_functional: average_value gradient: {e}"))?
            } else {
                // average_derivative needs rows of basis-function DERIVATIVES
                // ∂φ_j/∂x(x_i), NOT the value design φ_j(x_i). Feeding the value
                // design returns mean_i w_i·m(x_i) (the average value) instead of
                // mean_i w_i·m'(x_i) (#1120). Build the EXACT analytic derivative
                // design: each smooth term's basis FIRST DERIVATIVE pushed through
                // the same frozen identifiability chart the value design uses, so
                // its columns align with beta.
                let deriv_col = resolve_average_derivative_column(&spec, &dataset, &spec_val)?;
                let dx = build_term_collection_derivative_design(
                    standard.data.view(),
                    &spec,
                    deriv_col,
                )
                .map_err(|e| {
                    format!("debiased_functional: average_derivative design build failed: {e}")
                })?;
                if dx.ncols() != x.ncols() {
                    return Err(format!(
                        "debiased_functional: average_derivative design width {} does not \
                         match fitted coefficient width {}",
                        dx.ncols(),
                        x.ncols()
                    ));
                }
                SmoothFunctional::AverageDerivative {
                    derivative_design: dx.view(),
                    weights: weights.as_ref().map(|w| w.view()),
                }
                .gradient()
                .map_err(|e| format!("debiased_functional: average_derivative gradient: {e}"))?
            }
        }
        other => {
            return Err(format!(
                "debiased_functional: unknown target {other:?}; expected one of \
                 \"point\", \"contrast\", \"average_derivative\", \"average_value\", \"linear\""
            ));
        }
    };

    let input = RieszInput {
        beta: beta.view(),
        functional_gradient: gradient.view(),
        row_scores: row_scores.view(),
        penalty_beta: penalty_beta.view(),
        leverage: None,
    };
    let report = debias_with_dense_hessian(&input, h.view())
        .map_err(|e| format!("debiased_functional: Riesz engine error: {e}"))?;

    let half_width = 1.959_963_984_540_054 * report.se;
    let out = serde_json::json!({
        "target": target,
        "theta_plugin": report.theta_plugin,
        "theta_debiased": report.theta_onestep,
        "se": report.se,
        "penalty_bias": report.penalty_bias,
        "ci_lower": report.theta_onestep - half_width,
        "ci_upper": report.theta_onestep + half_width,
        "ci_level": 0.95_f64,
    });
    serde_json::to_string(&out)
        .map_err(|e| format!("debiased_functional: serialization failed: {e}"))
}

/// Resolve the covariate column index to differentiate for an
/// `average_derivative` functional (#1120 / #1097).
///
/// If the target spec carries an explicit `"deriv_var"` column name, it is
/// resolved against the materialized dataset headers. Otherwise the column is
/// auto-selected (magic-by-default) as the single feature column shared by all
/// smooth terms in the model; if the model has smooths over more than one
/// covariate the caller must disambiguate via `"deriv_var"`.
fn resolve_average_derivative_column(
    spec: &TermCollectionSpec,
    dataset: &EncodedDataset,
    spec_val: &serde_json::Value,
) -> Result<usize, String> {
    if let Some(name) = spec_val.get("deriv_var").and_then(|v| v.as_str()) {
        return dataset.column_map().get(name).copied().ok_or_else(|| {
            format!(
                "debiased_functional: average_derivative \"deriv_var\" '{name}' \
                     is not a column of the training data"
            )
        });
    }
    let mut cols: Vec<usize> = spec
        .smooth_terms
        .iter()
        .flat_map(smooth_term_feature_cols)
        .collect();
    cols.sort_unstable();
    cols.dedup();
    match cols.as_slice() {
        [single] => Ok(*single),
        [] => Err(
            "debiased_functional: average_derivative requires at least one smooth term \
             to differentiate; the model has no smooths"
                .to_string(),
        ),
        _ => Err(
            "debiased_functional: average_derivative is ambiguous because the model has \
             smooths over more than one covariate; specify the covariate via the \
             \"deriv_var\" key in the target spec"
                .to_string(),
        ),
    }
}

#[pyfunction]
fn summary_payload_from_model(py: Python<'_>, model_bytes: Vec<u8>) -> PyResult<PyObject> {
    let payload = detach_py_result(py, "summary_payload_from_model", move || {
        summary_payload_value_from_model_bytes(&model_bytes)
    })?;
    json_object_to_py_dict(py, payload)
}

#[pyfunction]
fn smoothing_parameters_from_model(py: Python<'_>, model_bytes: Vec<u8>) -> PyResult<PyObject> {
    let payload = summary_payload_from_model_bytes(&model_bytes)?;
    let out = PyDict::new(py);
    let Some(lambdas) = payload.get("lambdas").and_then(serde_json::Value::as_array) else {
        return Ok(out.unbind().into_any());
    };
    for (idx, value) in lambdas.iter().enumerate() {
        let lambda = value.as_f64().ok_or_else(|| {
            py_value_error(format!("summary lambdas[{idx}] must be a JSON number"))
        })?;
        out.set_item(idx, lambda)?;
    }
    Ok(out.unbind().into_any())
}

#[pyfunction]
fn model_group_metadata(py: Python<'_>, model_bytes: Vec<u8>) -> PyResult<PyObject> {
    let payload = summary_payload_from_model_bytes(&model_bytes)?;
    match payload.get("group_metadata") {
        Some(value @ serde_json::Value::Object(_)) => json_value_to_py(py, value.clone()),
        _ => Ok(py.None()),
    }
}

#[pyfunction]
fn model_deployment_extensions(py: Python<'_>, model_bytes: Vec<u8>) -> PyResult<PyObject> {
    let payload = summary_payload_from_model_bytes(&model_bytes)?;
    let out = PyList::empty(py);
    let Some(extensions) = payload
        .get("deployment_extensions")
        .and_then(serde_json::Value::as_array)
    else {
        return Ok(out.unbind().into_any());
    };
    for extension in extensions {
        if matches!(extension, serde_json::Value::Object(_)) {
            let py_value = json_value_to_py(py, extension.clone())?;
            out.append(py_value.bind(py))?;
        }
    }
    Ok(out.unbind().into_any())
}

#[pyfunction]
fn model_evidence(model_bytes: Vec<u8>) -> PyResult<f64> {
    let payload = summary_payload_from_model_bytes(&model_bytes)?;
    // Report the SAME Occam-penalised conditional-AIC ranking score that
    // `gamfit.compare_models` ranks on (`-2·loglik + 2·edf`), not the raw
    // REML/LAML evidence headline, so `Model.evidence` ordering agrees with the
    // winner `compare_models` declares. Lower is still better (issue #2079).
    ranking_score_from_summary_payload(&payload)
}

fn summary_payload_value_from_model_bytes(model_bytes: &[u8]) -> Result<serde_json::Value, String> {
    let summary_json = summary_json_impl(model_bytes)?;
    serde_json::from_str(&summary_json).map_err(|err| format!("invalid model summary JSON: {err}"))
}

fn json_object_to_py_dict(py: Python<'_>, value: serde_json::Value) -> PyResult<PyObject> {
    let serde_json::Value::Object(items) = value else {
        return Err(py_value_error(
            "model summary payload must be a JSON object".to_string(),
        ));
    };
    let out = PyDict::new(py);
    for (key, value) in items {
        let py_value = json_value_to_py(py, value)?;
        out.set_item(key, py_value.bind(py))?;
    }
    Ok(out.unbind().into_any())
}

fn json_value_to_py(py: Python<'_>, value: serde_json::Value) -> PyResult<PyObject> {
    match value {
        serde_json::Value::Null => Ok(py.None()),
        serde_json::Value::Bool(value) => value.into_py_any(py),
        serde_json::Value::Number(value) => {
            if let Some(value) = value.as_i64() {
                return value.into_py_any(py);
            }
            if let Some(value) = value.as_u64() {
                return value.into_py_any(py);
            }
            let value = value
                .as_f64()
                .ok_or_else(|| py_value_error("JSON number is not representable".to_string()))?;
            value.into_py_any(py)
        }
        serde_json::Value::String(value) => value.into_py_any(py),
        serde_json::Value::Array(values) => {
            let out = PyList::empty(py);
            for value in values {
                let py_value = json_value_to_py(py, value)?;
                out.append(py_value.bind(py))?;
            }
            Ok(out.unbind().into_any())
        }
        serde_json::Value::Object(values) => {
            let out = PyDict::new(py);
            for (key, value) in values {
                let py_value = json_value_to_py(py, value)?;
                out.set_item(key, py_value.bind(py))?;
            }
            Ok(out.unbind().into_any())
        }
    }
}

#[pyfunction]
fn summary_repr(payload: &Bound<'_, PyDict>) -> PyResult<String> {
    let mut fields = Vec::new();
    for key in [
        "formula",
        "family_name",
        "model_class",
        "deviance",
        "reml_score",
    ] {
        if let Some(value) = payload.get_item(key)? {
            let repr = value.repr()?.extract::<String>()?;
            fields.push(format!("{key}={repr}"));
        }
    }
    Ok(format!("Summary({})", fields.join(", ")))
}

#[pyfunction]
fn summary_html(payload: &Bound<'_, PyDict>) -> PyResult<String> {
    // Pure presentation layer; no math.
    let mut rows = String::new();
    for (key, value) in payload.iter() {
        let key_text = key.str()?.extract::<String>()?;
        if key_text == "coefficients" || key_text == "covariance_flat" {
            continue;
        }
        rows.push_str("<tr>");
        rows.push_str("<th style='text-align:left;padding:0.25rem 0.75rem 0.25rem 0;'>");
        rows.push_str(&summary_html_escape(&key_text));
        rows.push_str("</th><td style='padding:0.25rem 0;'>");
        rows.push_str(&summary_html_escape(&summary_render_value(&value)?));
        rows.push_str("</td></tr>");
    }
    let coefficient_table = summary_render_coefficients_html(payload)?;
    Ok(format!(
        "<div style='font-family: ui-sans-serif, system-ui, sans-serif;'>\
         <h3 style='margin:0 0 0.5rem 0;'>Model Summary</h3>\
         <table style='border-collapse:collapse;'>{rows}</table>\
         {coefficient_table}\
         </div>"
    ))
}

#[pyfunction]
fn coefficient_state_json(py: Python<'_>, model_bytes: Vec<u8>) -> PyResult<String> {
    detach_py_result(py, "coefficient_state_json", move || {
        coefficient_state_json_impl(&model_bytes)
    })
}

#[pyfunction]
fn term_blocks_for_model(model_bytes: Vec<u8>) -> PyResult<Vec<(String, String, usize, usize)>> {
    term_blocks_for_model_impl(&model_bytes).map_err(PyValueError::new_err)
}

#[pyfunction]
fn difference_smooth_json(
    py: Python<'_>,
    model_bytes: Vec<u8>,
    request_json: String,
) -> PyResult<String> {
    detach_py_result(py, "difference_smooth_json", move || {
        difference_smooth_json_impl(&model_bytes, &request_json)
    })
}

#[pyfunction]
fn difference_smooth_rows(
    py: Python<'_>,
    model_bytes: Vec<u8>,
    request_json: String,
) -> PyResult<PyObject> {
    let rows = detach_py_result(py, "difference_smooth_rows", move || {
        let raw = difference_smooth_json_impl(&model_bytes, &request_json)?;
        serde_json::from_str::<Vec<serde_json::Map<String, serde_json::Value>>>(&raw)
            .map_err(|err| format!("failed to parse difference_smooth rows json: {err}"))
    })?;
    difference_smooth_rows_to_py(py, rows)
}

fn difference_smooth_rows_to_py(
    py: Python<'_>,
    rows: Vec<serde_json::Map<String, serde_json::Value>>,
) -> PyResult<PyObject> {
    let out = PyList::empty(py);
    for row in rows {
        let py_row = PyDict::new(py);
        for (key, value) in row {
            py_dict_set_json_value(py, &py_row, &key, value)?;
        }
        out.append(py_row)?;
    }
    Ok(out.unbind().into_any())
}

fn py_dict_set_json_value(
    py: Python<'_>,
    dict: &Bound<'_, PyDict>,
    key: &str,
    value: serde_json::Value,
) -> PyResult<()> {
    match value {
        serde_json::Value::Null => dict.set_item(key, py.None()),
        serde_json::Value::Bool(value) => dict.set_item(key, value),
        serde_json::Value::Number(value) => py_dict_set_json_number(dict, key, value),
        serde_json::Value::String(value) => dict.set_item(key, value),
        serde_json::Value::Array(values) => {
            let list = PyList::empty(py);
            for value in values {
                py_list_append_json_value(py, &list, value)?;
            }
            dict.set_item(key, list)
        }
        serde_json::Value::Object(values) => {
            let nested = PyDict::new(py);
            for (nested_key, nested_value) in values {
                py_dict_set_json_value(py, &nested, &nested_key, nested_value)?;
            }
            dict.set_item(key, nested)
        }
    }
}

fn py_dict_set_json_number(
    dict: &Bound<'_, PyDict>,
    key: &str,
    value: serde_json::Number,
) -> PyResult<()> {
    if let Some(value) = value.as_i64() {
        dict.set_item(key, value)
    } else if let Some(value) = value.as_u64() {
        dict.set_item(key, value)
    } else if let Some(value) = value.as_f64() {
        dict.set_item(key, value)
    } else {
        Err(PyValueError::new_err(
            "difference_smooth row contains an unsupported JSON number",
        ))
    }
}

fn py_list_append_json_value(
    py: Python<'_>,
    list: &Bound<'_, PyList>,
    value: serde_json::Value,
) -> PyResult<()> {
    match value {
        serde_json::Value::Null => list.append(py.None()),
        serde_json::Value::Bool(value) => list.append(value),
        serde_json::Value::Number(value) => py_list_append_json_number(list, value),
        serde_json::Value::String(value) => list.append(value),
        serde_json::Value::Array(values) => {
            let nested = PyList::empty(py);
            for value in values {
                py_list_append_json_value(py, &nested, value)?;
            }
            list.append(nested)
        }
        serde_json::Value::Object(values) => {
            let nested = PyDict::new(py);
            for (nested_key, nested_value) in values {
                py_dict_set_json_value(py, &nested, &nested_key, nested_value)?;
            }
            list.append(nested)
        }
    }
}

fn py_list_append_json_number(list: &Bound<'_, PyList>, value: serde_json::Number) -> PyResult<()> {
    if let Some(value) = value.as_i64() {
        list.append(value)
    } else if let Some(value) = value.as_u64() {
        list.append(value)
    } else if let Some(value) = value.as_f64() {
        list.append(value)
    } else {
        Err(PyValueError::new_err(
            "difference_smooth row contains an unsupported JSON number",
        ))
    }
}

#[pyfunction]
fn cross_fit_shared_precision_groups_json(
    py: Python<'_>,
    request_json: String,
) -> PyResult<String> {
    detach_py_result(py, "cross_fit_shared_precision_groups_json", move || {
        cross_fit_shared_precision_groups_json_impl(&request_json)
    })
}

#[pyfunction]
fn check_json(
    py: Python<'_>,
    model_bytes: Vec<u8>,
    headers: Vec<String>,
    rows: PyRef<'_, PyEncodedTable>,
) -> PyResult<String> {
    rows.require_headers(&headers).map_err(py_value_error)?;
    let dataset = rows.dataset.clone();
    detach_py_result(py, "check_json", move || {
        check_dataset_json_impl(&model_bytes, dataset)
    })
}

#[pyfunction]
fn check_payload_from_model(
    py: Python<'_>,
    model_bytes: Vec<u8>,
    headers: Vec<String>,
    rows: PyRef<'_, PyEncodedTable>,
) -> PyResult<PyObject> {
    rows.require_headers(&headers).map_err(py_value_error)?;
    let dataset = rows.dataset.clone();
    let payload = detach_py_result(py, "check_payload_from_model", move || {
        let check_json = check_dataset_json_impl(&model_bytes, dataset)?;
        serde_json::from_str::<serde_json::Value>(&check_json)
            .map_err(|err| format!("invalid schema check JSON: {err}"))
    })?;
    json_object_to_py_dict(py, payload)
}

#[pyfunction]
fn report_html(py: Python<'_>, model_bytes: Vec<u8>) -> PyResult<String> {
    detach_py_result(py, "report_html", move || report_html_impl(&model_bytes))
}

#[pyfunction]
fn compute_residuals<'py>(
    py: Python<'py>,
    observed: PyReadonlyArray1<'py, f64>,
    predicted_mean: PyReadonlyArray1<'py, f64>,
) -> PyResult<Py<PyArray1<f64>>> {
    let observed_values = observed.as_array();
    let predicted_values = predicted_mean.as_array();
    if observed_values.len() != predicted_values.len() {
        return Err(py_value_error(format!(
            "compute_residuals length mismatch: observed has {} values but predicted mean has {}",
            observed_values.len(),
            predicted_values.len()
        )));
    }

    let residuals = observed_values
        .iter()
        .zip(predicted_values.iter())
        .map(|(obs, pred)| *obs - *pred)
        .collect::<Vec<_>>();
    Ok(Array1::from_vec(residuals).into_pyarray(py).unbind())
}

#[pyfunction]
fn diagnostics_from_predictions(
    py: Python<'_>,
    observed: Vec<f64>,
    predicted_mean: Vec<f64>,
) -> PyResult<Py<PyDict>> {
    let diagnostics =
        gam::inference::diagnostics::diagnostics_from_predictions(&observed, &predicted_mean)
            .map_err(py_value_error)?;

    let metrics = PyDict::new(py);
    metrics.set_item("n_obs", diagnostics.n_obs)?;
    metrics.set_item("mae", diagnostics.mae)?;
    metrics.set_item("rmse", diagnostics.rmse)?;
    metrics.set_item("bias", diagnostics.bias)?;
    if let Some(r_squared) = diagnostics.r_squared {
        metrics.set_item("r_squared", r_squared)?;
    }

    let out = PyDict::new(py);
    out.set_item("residuals", PyList::new(py, diagnostics.residuals)?)?;
    out.set_item("metrics", metrics)?;
    Ok(out.unbind())
}

#[pyfunction]
fn auc_from_predictions(observed: Vec<f64>, predicted_mean: Vec<f64>) -> PyResult<f64> {
    gam::inference::diagnostics::auc_from_predictions(&observed, &predicted_mean)
        .map_err(py_value_error)
}

#[pyfunction]
fn weighted_auc_from_predictions(
    observed: Vec<f64>,
    predicted_mean: Vec<f64>,
    weights: Vec<f64>,
) -> PyResult<f64> {
    gam::inference::diagnostics::weighted_auc_from_predictions(
        &observed,
        &predicted_mean,
        Some(&weights),
    )
    .map_err(py_value_error)
}

#[pyfunction]
fn brier_from_predictions(observed: Vec<f64>, predicted_mean: Vec<f64>) -> PyResult<f64> {
    gam::inference::diagnostics::brier_from_predictions(&observed, &predicted_mean)
        .map_err(py_value_error)
}

#[pyfunction(signature = (observed, predicted_mean, eps = 1e-12))]
fn log_loss_from_predictions(
    observed: Vec<f64>,
    predicted_mean: Vec<f64>,
    eps: f64,
) -> PyResult<f64> {
    gam::inference::diagnostics::binary_log_loss_from_predictions(
        &observed,
        &predicted_mean,
        eps,
    )
    .map_err(py_value_error)
}

#[pyfunction(signature = (observed, predicted_mean, null_mean, eps = 1e-12))]
fn nagelkerke_r2_from_predictions(
    observed: Vec<f64>,
    predicted_mean: Vec<f64>,
    null_mean: f64,
    eps: f64,
) -> PyResult<Option<f64>> {
    gam::inference::diagnostics::nagelkerke_r_squared_from_predictions(
        &observed,
        &predicted_mean,
        null_mean,
        eps,
    )
    .map_err(py_value_error)
}

#[pyfunction(signature = (y, n_splits, seed, stratified))]
fn make_folds_indices(
    y: Vec<f64>,
    n_splits: usize,
    seed: u64,
    stratified: bool,
) -> PyResult<Vec<(Vec<usize>, Vec<usize>)>> {
    use rand::SeedableRng;
    use rand::rngs::StdRng;
    use rand::seq::SliceRandom;
    use std::collections::BTreeMap;

    if n_splits == 0 {
        return Err(py_value_error(
            "make_folds_indices: n_splits must be >= 1".to_string(),
        ));
    }
    let n = y.len();
    if n == 0 {
        return Err(py_value_error(
            "make_folds_indices: y must have at least one observation".to_string(),
        ));
    }
    for (i, v) in y.iter().enumerate() {
        if !v.is_finite() {
            return Err(py_value_error(format!(
                "make_folds_indices: y[{i}] is not finite ({v}); CV labels must be finite"
            )));
        }
    }
    if n_splits >= 2 && n < n_splits {
        return Err(py_value_error(format!(
            "make_folds_indices: n_splits={n_splits} cannot exceed n_observations={n}"
        )));
    }

    let mut rng = StdRng::seed_from_u64(seed);

    // Group row indices either into a single bucket (unstratified) or into
    // one bucket per numeric label value (stratified). The key is the bit
    // pattern with both IEEE zeros canonicalized to +0.0 — `-0.0 == +0.0`,
    // so they are one class level, not two; NaN is rejected above.
    let mut buckets: Vec<Vec<usize>> = if stratified {
        let label_key = |v: f64| if v == 0.0 { 0.0_f64.to_bits() } else { v.to_bits() };
        let mut by_label: BTreeMap<u64, Vec<usize>> = BTreeMap::new();
        for (i, v) in y.iter().enumerate() {
            by_label.entry(label_key(*v)).or_default().push(i);
        }
        by_label.into_values().collect()
    } else {
        vec![(0..n).collect()]
    };

    if n_splits >= 2 {
        // K-fold: round-robin assignment per bucket guarantees that every
        // row lands in exactly one test fold and that strata are balanced
        // across folds.
        let mut fold_of_row = vec![usize::MAX; n];
        for bucket in buckets.iter_mut() {
            bucket.shuffle(&mut rng);
            for (k, &idx) in bucket.iter().enumerate() {
                fold_of_row[idx] = k % n_splits;
            }
        }
        let mut folds: Vec<(Vec<usize>, Vec<usize>)> =
            (0..n_splits).map(|_| (Vec::new(), Vec::new())).collect();
        for (idx, &f) in fold_of_row.iter().enumerate() {
            for (k, fold) in folds.iter_mut().enumerate() {
                if k == f {
                    fold.1.push(idx);
                } else {
                    fold.0.push(idx);
                }
            }
        }
        for (k, (train, test)) in folds.iter().enumerate() {
            if test.is_empty() {
                return Err(py_value_error(format!(
                    "make_folds_indices: fold {k}/{n_splits} has empty test set \
                     (n={n}); reduce n_splits or supply more observations"
                )));
            }
            if train.is_empty() {
                return Err(py_value_error(format!(
                    "make_folds_indices: fold {k}/{n_splits} has empty train set \
                     (n={n}); reduce n_splits or supply more observations"
                )));
            }
        }
        Ok(folds)
    } else {
        // n_splits == 1: a single deterministic holdout. Hold out 1/5 of
        // each stratum, matching the proportion of one fold from the
        // default 5-fold split so binary/survival validation that demands
        // ≥1 row of each class in both train and test is satisfied as
        // long as each class has ≥2 observations (the contract the Python
        // wrapper documents).
        const HOLDOUT_DENOMINATOR: usize = 5;
        let mut train: Vec<usize> = Vec::new();
        let mut test: Vec<usize> = Vec::new();
        for bucket in buckets.iter_mut() {
            bucket.shuffle(&mut rng);
            let m = bucket.len();
            // ≥1 in test, ≥1 in train for every stratum of size ≥ 2.
            // Strata of size 1 (which only the unstratified single-bucket
            // path with n == 1 can produce — already rejected above for
            // stratified=true via empty-fold check below) collapse to
            // train, which the empty-test sanity check will catch.
            let mut n_test = m / HOLDOUT_DENOMINATOR;
            if n_test == 0 && m >= 2 {
                n_test = 1;
            }
            if n_test >= m && m >= 2 {
                n_test = m - 1;
            }
            for (i, &idx) in bucket.iter().enumerate() {
                if i < n_test {
                    test.push(idx);
                } else {
                    train.push(idx);
                }
            }
        }
        if test.is_empty() {
            return Err(py_value_error(format!(
                "make_folds_indices: holdout split has empty test set (n={n}); \
                 supply at least 2 observations (and ≥2 per class when stratified)"
            )));
        }
        if train.is_empty() {
            return Err(py_value_error(format!(
                "make_folds_indices: holdout split has empty train set (n={n}); \
                 supply at least 2 observations (and ≥2 per class when stratified)"
            )));
        }
        train.sort_unstable();
        test.sort_unstable();
        Ok(vec![(train, test)])
    }
}

fn gaussian_log_loss_value(
    observed: &[f64],
    predicted_mean: &[f64],
    sigma: &[f64],
    eps: f64,
) -> PyResult<f64> {
    gam::inference::diagnostics::gaussian_log_loss_from_predictions(
        observed,
        predicted_mean,
        sigma,
        eps,
    )
    .map_err(py_value_error)
}

#[pyfunction(signature = (observed, predicted_mean, sigma, eps = 1e-12))]
fn gaussian_log_loss_from_predictions(
    observed: Vec<f64>,
    predicted_mean: Vec<f64>,
    sigma: Vec<f64>,
    eps: f64,
) -> PyResult<f64> {
    gaussian_log_loss_value(&observed, &predicted_mean, &sigma, eps)
}

#[pyfunction]
fn gaussian_prediction_scores_from_predictions(
    py: Python<'_>,
    observed: Vec<f64>,
    predicted_mean: Vec<f64>,
    sigma: Vec<f64>,
) -> PyResult<Py<PyDict>> {
    let diag =
        gam::inference::diagnostics::diagnostics_from_predictions(&observed, &predicted_mean)
            .map_err(py_value_error)?;
    let logloss = gaussian_log_loss_value(
        &observed,
        &predicted_mean,
        &sigma,
        gam::inference::diagnostics::DEFAULT_GAUSSIAN_SCALE_FLOOR,
    )?;

    let out = PyDict::new(py);
    out.set_item("n_obs", diag.n_obs)?;
    out.set_item("rmse", diag.rmse)?;
    out.set_item("mse", diag.rmse * diag.rmse)?;
    out.set_item("mae", diag.mae)?;
    out.set_item("bias", diag.bias)?;
    out.set_item("logloss", logloss)?;
    match diag.r_squared {
        Some(r2) => out.set_item("r2", r2)?,
        None => out.set_item("r2", py.None())?,
    }
    Ok(out.unbind())
}

#[pyfunction]
fn zscore_train_test_arrays<'py>(
    py: Python<'py>,
    train: PyReadonlyArray2<'py, f64>,
    test: PyReadonlyArray2<'py, f64>,
) -> PyResult<(Py<PyArray2<f64>>, Py<PyArray2<f64>>)> {
    let tr = train.as_array();
    let te = test.as_array();
    let p = tr.ncols();
    if te.ncols() != p {
        return Err(py_value_error(format!(
            "zscore_train_test_arrays: train has {p} columns but test has {}",
            te.ncols()
        )));
    }
    let n_train = tr.nrows();
    if n_train == 0 {
        return Err(py_value_error(
            "zscore_train_test_arrays: train has zero rows; cannot estimate column statistics"
                .to_string(),
        ));
    }
    let n_train_f = n_train as f64;
    for (j, col) in tr.axis_iter(Axis(1)).enumerate() {
        for (i, v) in col.iter().enumerate() {
            if !v.is_finite() {
                return Err(py_value_error(format!(
                    "zscore_train_test_arrays: train[{i}, {j}] is not finite ({v})"
                )));
            }
        }
    }
    for (j, col) in te.axis_iter(Axis(1)).enumerate() {
        for (i, v) in col.iter().enumerate() {
            if !v.is_finite() {
                return Err(py_value_error(format!(
                    "zscore_train_test_arrays: test[{i}, {j}] is not finite ({v})"
                )));
            }
        }
    }

    let mut means = vec![0.0_f64; p];
    let mut stds = vec![1.0_f64; p];
    for j in 0..p {
        let col = tr.column(j);
        // Welford one-pass moments: the running mean never leaves the data's
        // range, so finite inputs near f64::MAX cannot overflow the way a
        // naive `Σx / n` does (Σx saturates to +inf and the centred output
        // turns NaN).
        let mut mean = 0.0_f64;
        let mut m2 = 0.0_f64;
        for (k, v) in col.iter().enumerate() {
            let delta = v - mean;
            mean += delta / (k + 1) as f64;
            m2 += delta * (v - mean);
        }
        // Population standard deviation matches sklearn StandardScaler and
        // pandas (ddof=0). A constant column collapses to std=0 below; we
        // pass the centred zero through unchanged by leaving the divisor
        // at 1.0 in that branch (column has no variation to scale away).
        let std = (m2 / n_train_f).sqrt();
        if !mean.is_finite() || !std.is_finite() {
            return Err(py_value_error(format!(
                "zscore_train_test_arrays: column {j} moments are not representable in f64 \
                 (mean={mean}, std={std}); rescale the inputs before standardizing"
            )));
        }
        means[j] = mean;
        stds[j] = if std > 0.0 { std } else { 1.0 };
    }

    let mut tr_out = Array2::<f64>::zeros(tr.raw_dim());
    let mut te_out = Array2::<f64>::zeros(te.raw_dim());
    for j in 0..p {
        let mean = means[j];
        let std = stds[j];
        for i in 0..tr.nrows() {
            tr_out[[i, j]] = (tr[[i, j]] - mean) / std;
        }
        for i in 0..te.nrows() {
            te_out[[i, j]] = (te[[i, j]] - mean) / std;
        }
    }
    Ok((
        tr_out.into_pyarray(py).unbind(),
        te_out.into_pyarray(py).unbind(),
    ))
}

#[pyfunction]
fn classification_metrics(
    py: Python<'_>,
    observed: Vec<f64>,
    predicted_mean: Vec<f64>,
    train_prev: f64,
) -> PyResult<Py<PyDict>> {
    let metrics = gam::inference::diagnostics::classification_metrics_from_predictions(
        &observed,
        &predicted_mean,
        train_prev,
    )
    .map_err(py_value_error)?;
    let out = PyDict::new(py);
    out.set_item("auc", metrics.auc)?;
    out.set_item("pr_auc", metrics.precision_recall_auc)?;
    out.set_item("brier", metrics.brier)?;
    out.set_item("logloss", metrics.log_loss)?;
    match metrics.nagelkerke_r_squared {
        Some(value) => out.set_item("nagelkerke_r2", value)?,
        None => out.set_item("nagelkerke_r2", py.None())?,
    }
    out.set_item("ece", metrics.expected_calibration_error)?;
    Ok(out.unbind())
}

#[pyfunction]
fn survival_concordance(
    event_times: Vec<f64>,
    risk_score: Vec<f64>,
    events: Vec<f64>,
) -> PyResult<Option<f64>> {
    if event_times.len() != risk_score.len() || event_times.len() != events.len() {
        return Err(PyValueError::new_err(format!(
            "survival_concordance length mismatch: times={} risk={} events={}",
            event_times.len(),
            risk_score.len(),
            events.len()
        )));
    }
    // Delegate to the single source of truth for Harrell's C-index in
    // gam-models (`survival::predict::harrell_concordance`). The core counts
    // tied event times as a comparable half-credit pair and returns None when
    // there are no comparable pairs at all (e.g. every row censored); the old
    // hand-rolled pair loop here dropped tied-time pairs entirely and returned
    // a silent 0.5 sentinel. Where the two disagreed the core wins — a None
    // degenerate result is surfaced as Python None, matching how the
    // neighboring metric pyfunctions report an undefined score.
    Ok(gam::families::survival::predict::harrell_concordance(
        &event_times,
        &events,
        &risk_score,
    ))
}

#[pyfunction]
fn survival_score_grid_from_times<'py>(
    py: Python<'py>,
    train_times: Vec<f64>,
) -> PyResult<Py<PyArray1<f64>>> {
    let mut times: Vec<f64> = train_times
        .into_iter()
        .filter(|value| value.is_finite() && *value > 0.0)
        .collect();
    if times.is_empty() {
        return Ok(Array1::from_vec(vec![0.0, 1.0]).into_pyarray(py).unbind());
    }
    times.sort_by(|a, b| a.total_cmp(b));
    let max_t = *times.last().unwrap();
    // Data-driven, quantile-spaced evaluation grid spanning [0, max(t)]. The old
    // grid was a fixed {0,1,2,5,10,median} set whose magic constants only made
    // sense for O(1)–O(10) survival times; on any other time scale it either ran
    // far past the data (so an integrated Brier was dominated by an empty
    // extrapolation tail) or never reached it. Concentrating the interior knots
    // at empirical quantiles resolves the event-dense region well for both the
    // survival-matrix evaluation and the integrated IPCW Brier integration.
    const INTERIOR: usize = 24;
    let mut grid: Vec<f64> = Vec::with_capacity(INTERIOR + 2);
    grid.push(0.0);
    for j in 1..=INTERIOR {
        let p = j as f64 / (INTERIOR as f64 + 1.0);
        grid.push(quantile_of_sorted(&times, p));
    }
    grid.push(max_t);
    grid.sort_by(|a, b| a.total_cmp(b));
    // Strictly increasing: drop points that collapse onto their predecessor
    // (heavy ties pull many quantiles to the same value).
    grid.dedup_by(|a, b| (*a - *b).abs() <= f64::EPSILON * a.abs().max(*b).max(1.0));
    grid[0] = 0.0;
    if grid.len() < 2 {
        grid = vec![0.0, max_t.max(1.0)];
    }
    Ok(Array1::from_vec(grid).into_pyarray(py).unbind())
}

/// Type-7 (NumPy default) linear-interpolation quantile of an ascending slice.
fn quantile_of_sorted(sorted: &[f64], p: f64) -> f64 {
    match sorted.len() {
        0 => f64::NAN,
        1 => sorted[0],
        n => {
            let h = (n as f64 - 1.0) * p.clamp(0.0, 1.0);
            let lo = h.floor() as usize;
            let hi = (lo + 1).min(n - 1);
            let frac = h - lo as f64;
            sorted[lo] + frac * (sorted[hi] - sorted[lo])
        }
    }
}

#[pyfunction]
fn repeat_survival_curve<'py>(
    py: Python<'py>,
    survival: Vec<f64>,
    n_rows: usize,
) -> PyResult<Py<PyArray2<f64>>> {
    let mut out = Array2::<f64>::zeros((n_rows, survival.len()));
    for mut row in out.rows_mut() {
        for (dst, src) in row.iter_mut().zip(survival.iter()) {
            *dst = *src;
        }
    }
    Ok(out.into_pyarray(py).unbind())
}

/// Non-parametric Kaplan-Meier survival estimate `S(t) = ∏_{t_i ≤ t} (1 − d_i/n_i)`
/// (right-continuous step function), evaluated at each point in `grid`.
/// Non-finite and non-positive times are dropped before fitting. Rows tied at
/// the same event time are pooled into a single risk-set update, matching the
/// standard product-limit definition.
fn benchmark_km_curve(times: &[f64], events: &[f64], grid: &[f64]) -> Vec<f64> {
    let mut rows: Vec<(f64, f64)> = times
        .iter()
        .zip(events.iter())
        .filter(|(&t, _)| t.is_finite() && t > 0.0)
        .map(|(&t, &e)| (t, e))
        .collect();
    if rows.is_empty() {
        return vec![1.0; grid.len()];
    }
    rows.sort_by(|a, b| a.0.total_cmp(&b.0));
    let n_total = rows.len();
    let mut event_step_times = Vec::<f64>::new();
    let mut survival_after_step = Vec::<f64>::new();
    let mut survivor = 1.0;
    let mut i = 0usize;
    while i < rows.len() {
        let t = rows[i].0;
        let mut j = i;
        let mut deaths = 0usize;
        while j < rows.len() && rows[j].0 == t {
            if rows[j].1 > 0.5 {
                deaths += 1;
            }
            j += 1;
        }
        if deaths > 0 {
            let at_risk = (n_total - i) as f64;
            survivor *= 1.0 - deaths as f64 / at_risk;
            event_step_times.push(t);
            survival_after_step.push(survivor);
        }
        i = j;
    }
    grid.iter()
        .map(|&g| {
            let idx = event_step_times.partition_point(|&t| t <= g);
            if idx == 0 {
                1.0
            } else {
                survival_after_step[idx - 1]
            }
        })
        .collect()
}

/// The marginal (covariate-free) Kaplan-Meier survival curve fit on the
/// training sample, evaluated at `grid`. This is the "null model" baseline
/// [`survival_lifted_metrics_from_predictions`] scores a model's calibrated
/// survival matrix against.
#[pyfunction]
fn survival_null_curve_from_train<'py>(
    py: Python<'py>,
    train_times: Vec<f64>,
    train_events: Vec<f64>,
    grid: Vec<f64>,
) -> PyResult<Py<PyArray1<f64>>> {
    if train_times.len() != train_events.len() {
        return Err(PyValueError::new_err(format!(
            "survival_null_curve_from_train length mismatch: times={} events={}",
            train_times.len(),
            train_events.len()
        )));
    }
    let curve = benchmark_km_curve(&train_times, &train_events, &grid);
    Ok(Array1::from_vec(curve).into_pyarray(py).unbind())
}

fn benchmark_survival_matrix_from_risk(
    train_times: &[f64],
    train_events: &[f64],
    train_risk: &[f64],
    test_risk: &[f64],
    grid: &[f64],
) -> PyResult<Array2<f64>> {
    if train_times.len() != train_events.len() || train_times.len() != train_risk.len() {
        return Err(PyValueError::new_err(format!(
            "survival calibration length mismatch: times={} events={} risk={}",
            train_times.len(),
            train_events.len(),
            train_risk.len()
        )));
    }
    let mut rows: Vec<(f64, f64, f64)> = train_times
        .iter()
        .zip(train_events.iter())
        .zip(train_risk.iter())
        .filter_map(|((&time, &event), &risk)| {
            (time.is_finite() && event.is_finite() && risk.is_finite() && time > 0.0)
                .then_some((time, event, risk))
        })
        .collect();
    if rows.is_empty() {
        return Ok(Array2::<f64>::ones((test_risk.len(), grid.len())));
    }
    let risk_mean = rows.iter().map(|row| row.2).sum::<f64>() / rows.len() as f64;
    let risk_sd = (rows
        .iter()
        .map(|row| {
            let d = row.2 - risk_mean;
            d * d
        })
        .sum::<f64>()
        / rows.len() as f64)
        .sqrt();
    if rows.len() < 2 || risk_sd < 1.0e-12 {
        let times: Vec<f64> = rows.iter().map(|row| row.0).collect();
        let events: Vec<f64> = rows.iter().map(|row| row.1).collect();
        let curve = benchmark_km_curve(&times, &events, grid);
        let mut out = Array2::<f64>::zeros((test_risk.len(), grid.len()));
        for mut row in out.rows_mut() {
            for (dst, src) in row.iter_mut().zip(curve.iter()) {
                *dst = *src;
            }
        }
        return Ok(out);
    }
    for row in &mut rows {
        row.2 -= risk_mean;
    }
    rows.sort_by(|a, b| a.0.total_cmp(&b.0));
    let mut event_times = Vec::<(f64, usize, f64)>::new();
    let mut i = 0usize;
    while i < rows.len() {
        let time = rows[i].0;
        let mut j = i + 1;
        let mut d = usize::from(rows[i].1 > 0.5);
        let mut event_x = if rows[i].1 > 0.5 { rows[i].2 } else { 0.0 };
        while j < rows.len() && rows[j].0 == time {
            if rows[j].1 > 0.5 {
                d += 1;
                event_x += rows[j].2;
            }
            j += 1;
        }
        if d > 0 {
            event_times.push((time, d, event_x));
        }
        i = j;
    }
    let mut beta = 0.0;
    let ridge = 1.0e-8;
    for _ in 0..50 {
        let mut score = -ridge * beta;
        let mut info = ridge;
        for &(time, d, event_x) in &event_times {
            let mut s0 = 0.0;
            let mut s1 = 0.0;
            let mut s2 = 0.0;
            for &(row_time, _, x) in &rows {
                if row_time >= time {
                    let w = (beta * x).clamp(-50.0, 50.0).exp();
                    s0 += w;
                    s1 += w * x;
                    s2 += w * x * x;
                }
            }
            if s0 > 0.0 {
                let mean = s1 / s0;
                score += event_x - d as f64 * mean;
                info += d as f64 * (s2 / s0 - mean * mean).max(0.0);
            }
        }
        if !info.is_finite() || info <= 0.0 {
            break;
        }
        let step = (score / info).clamp(-2.0, 2.0);
        beta += step;
        if step.abs() < 1.0e-10 {
            break;
        }
    }
    let mut baseline = Vec::<(f64, f64)>::new();
    let mut cumulative = 0.0;
    for &(time, d, _) in &event_times {
        let mut s0 = 0.0;
        for &(row_time, _, x) in &rows {
            if row_time >= time {
                s0 += (beta * x).clamp(-50.0, 50.0).exp();
            }
        }
        if s0 > 0.0 {
            cumulative += d as f64 / s0;
            baseline.push((time, cumulative));
        }
    }
    let mut out = Array2::<f64>::zeros((test_risk.len(), grid.len()));
    for (row_idx, &risk) in test_risk.iter().enumerate() {
        let x = if risk.is_finite() {
            risk - risk_mean
        } else {
            0.0
        };
        let mult = (beta * x).clamp(-50.0, 50.0).exp();
        let mut step_idx = 0usize;
        let mut h0 = 0.0;
        let mut prev = 1.0;
        for (col_idx, &time) in grid.iter().enumerate() {
            while step_idx < baseline.len() && baseline[step_idx].0 <= time {
                h0 = baseline[step_idx].1;
                step_idx += 1;
            }
            let value = if col_idx == 0 {
                1.0
            } else {
                (-(h0 * mult)).exp().clamp(1.0e-12, 1.0).min(prev)
            };
            out[[row_idx, col_idx]] = value;
            prev = value;
        }
    }
    Ok(out)
}

#[pyfunction]
fn survival_matrix_from_risk_calibration<'py>(
    py: Python<'py>,
    train_times: Vec<f64>,
    train_events: Vec<f64>,
    train_risk: Vec<f64>,
    test_risk: Vec<f64>,
    grid: Vec<f64>,
) -> PyResult<Py<PyArray2<f64>>> {
    Ok(benchmark_survival_matrix_from_risk(
        &train_times,
        &train_events,
        &train_risk,
        &test_risk,
        &grid,
    )?
    .into_pyarray(py)
    .unbind())
}

#[pyfunction(signature = (event_times, events, grid, survival_matrix, null_survival_matrix = None, eps = 1e-12))]
fn survival_lifted_metrics_from_predictions<'py>(
    py: Python<'py>,
    event_times: Vec<f64>,
    events: Vec<f64>,
    grid: Vec<f64>,
    survival_matrix: PyReadonlyArray2<'py, f64>,
    null_survival_matrix: Option<PyReadonlyArray2<'py, f64>>,
    eps: f64,
) -> PyResult<Py<PyDict>> {
    let out = PyDict::new(py);
    let none_result = |out: &Bound<'_, PyDict>| -> PyResult<Py<PyDict>> {
        out.set_item("brier", py.None())?;
        out.set_item("hazard_quadratic_score", py.None())?;
        out.set_item("logloss", py.None())?;
        out.set_item("lifted_brier", py.None())?;
        out.set_item("lifted_hazard_quadratic_score", py.None())?;
        out.set_item("lifted_logloss", py.None())?;
        out.set_item("nagelkerke_r2", py.None())?;
        Ok(out.clone().unbind())
    };
    let obs: Vec<bool> = events.iter().map(|value| *value > 0.5).collect();
    let mut surv = survival_matrix.as_array().to_owned();
    if event_times.len() != obs.len()
        || surv.nrows() != event_times.len()
        || surv.ncols() != grid.len()
        || grid.len() < 2
        || grid.windows(2).any(|pair| pair[1] <= pair[0])
    {
        return none_result(&out);
    }
    for mut row in surv.rows_mut() {
        row[0] = 1.0;
        let mut prev = 1.0;
        for value in row.iter_mut() {
            *value = value.clamp(eps, 1.0).min(prev);
            prev = *value;
        }
    }
    let dt: Vec<f64> = grid.windows(2).map(|pair| pair[1] - pair[0]).collect();
    let cumhaz = surv.mapv(|value| -value.clamp(eps, 1.0).ln());
    let mut haz = Array2::<f64>::zeros((surv.nrows(), surv.ncols() - 1));
    for row in 0..surv.nrows() {
        for col in 0..surv.ncols() - 1 {
            haz[[row, col]] = ((cumhaz[[row, col + 1]] - cumhaz[[row, col]]) / dt[col]).max(0.0);
        }
    }
    let mut haz_sq_prefix = Array2::<f64>::zeros((surv.nrows(), surv.ncols()));
    for row in 0..surv.nrows() {
        for col in 0..haz.ncols() {
            haz_sq_prefix[[row, col + 1]] =
                haz_sq_prefix[[row, col]] + haz[[row, col]] * haz[[row, col]] * dt[col];
        }
    }
    let mut log_losses = vec![0.0; event_times.len()];
    // The "hazard quadratic" per-subject score 0.5∫h² − δ·h(T): a proper score
    // for the hazard model, but NOT the (IPCW) Brier score — see #1563.
    let mut hazard_quadratic_losses = vec![0.0; event_times.len()];
    for (row, &time) in event_times.iter().enumerate() {
        if !time.is_finite() || time <= 0.0 {
            return none_result(&out);
        }
        let mut j = grid.partition_point(|value| *value < time);
        if j >= grid.len() {
            j = grid.len() - 1;
        }
        let interval_idx = j.saturating_sub(1);
        let (h_z, h2_int, hcum_z) = if (grid[j] - time).abs() <= 1.0e-12 {
            (
                haz[[row, interval_idx]],
                haz_sq_prefix[[row, j]],
                cumhaz[[row, j]],
            )
        } else {
            let elapsed = time - grid[interval_idx];
            let h = haz[[row, interval_idx]];
            (
                h,
                haz_sq_prefix[[row, interval_idx]] + h * h * elapsed,
                cumhaz[[row, interval_idx]] + h * elapsed,
            )
        };
        log_losses[row] = hcum_z - if obs[row] { h_z.max(eps).ln() } else { 0.0 };
        hazard_quadratic_losses[row] = 0.5 * h2_int - if obs[row] { h_z } else { 0.0 };
    }
    let logloss = log_losses.iter().sum::<f64>() / log_losses.len() as f64;
    let hazard_quadratic =
        hazard_quadratic_losses.iter().sum::<f64>() / hazard_quadratic_losses.len() as f64;

    // Genuine integrated IPCW Brier score (Graf et al. 1999), comparable to
    // scikit-survival's `integrated_brier_score`, pec, and `survival::brier`.
    // This is what `brier` now reports; the hazard quadratic score above (which
    // this field previously mis-reported as `brier`) is exposed honestly under
    // `hazard_quadratic_score`. The censoring distribution G(t) is estimated by
    // Kaplan–Meier on the evaluation set itself, so the metric is self-contained
    // and identical for every model scored against the same fold (fair ranking).
    // Integration is capped at the largest observed time to avoid the
    // extrapolation tail where the IPCW weights blow up.
    let censoring_km =
        gam::families::survival::predict::KaplanMeier::fit_censoring(&event_times, &events);
    let horizon = event_times
        .iter()
        .copied()
        .filter(|value| value.is_finite())
        .fold(f64::NEG_INFINITY, f64::max);
    let ibs = gam::families::survival::predict::integrated_ipcw_brier_score(
        surv.view(),
        &event_times,
        &events,
        &grid,
        horizon,
        |t| censoring_km.at(t),
    );
    match ibs {
        Some(value) => out.set_item("brier", value)?,
        None => out.set_item("brier", py.None())?,
    }
    out.set_item("hazard_quadratic_score", hazard_quadratic)?;
    out.set_item("logloss", logloss)?;
    out.set_item("lifted_brier", py.None())?;
    out.set_item("lifted_hazard_quadratic_score", py.None())?;
    out.set_item("lifted_logloss", py.None())?;

    let mut nagelkerke = None;
    if let Some(null_matrix) = null_survival_matrix {
        let mut null_surv = null_matrix.as_array().to_owned();
        if null_surv.dim() == surv.dim() {
            for mut row in null_surv.rows_mut() {
                row[0] = 1.0;
                let mut prev = 1.0;
                for value in row.iter_mut() {
                    *value = value.clamp(eps, 1.0).min(prev);
                    prev = *value;
                }
            }
            let null_cumhaz = null_surv.mapv(|value| -value.clamp(eps, 1.0).ln());
            let mut null_haz = Array2::<f64>::zeros((null_surv.nrows(), null_surv.ncols() - 1));
            for row in 0..null_surv.nrows() {
                for col in 0..null_surv.ncols() - 1 {
                    null_haz[[row, col]] =
                        ((null_cumhaz[[row, col + 1]] - null_cumhaz[[row, col]]) / dt[col])
                            .max(0.0);
                }
            }
            let mut null_haz_sq_prefix =
                Array2::<f64>::zeros((null_surv.nrows(), null_surv.ncols()));
            for row in 0..null_surv.nrows() {
                for col in 0..null_haz.ncols() {
                    null_haz_sq_prefix[[row, col + 1]] = null_haz_sq_prefix[[row, col]]
                        + null_haz[[row, col]] * null_haz[[row, col]] * dt[col];
                }
            }
            let mut null_log_losses = vec![0.0; event_times.len()];
            let mut null_hazard_quadratic_losses = vec![0.0; event_times.len()];
            for (row, &time) in event_times.iter().enumerate() {
                let mut j = grid.partition_point(|value| *value < time);
                if j >= grid.len() {
                    j = grid.len() - 1;
                }
                let interval_idx = j.saturating_sub(1);
                let (h_z, hcum_z, h2_int) = if (grid[j] - time).abs() <= 1.0e-12 {
                    (
                        null_haz[[row, interval_idx]],
                        null_cumhaz[[row, j]],
                        null_haz_sq_prefix[[row, j]],
                    )
                } else {
                    let elapsed = time - grid[interval_idx];
                    let h = null_haz[[row, interval_idx]];
                    (
                        h,
                        null_cumhaz[[row, interval_idx]] + h * elapsed,
                        null_haz_sq_prefix[[row, interval_idx]] + h * h * elapsed,
                    )
                };
                null_log_losses[row] = hcum_z - if obs[row] { h_z.max(eps).ln() } else { 0.0 };
                null_hazard_quadratic_losses[row] =
                    0.5 * h2_int - if obs[row] { h_z } else { 0.0 };
            }
            let null_logloss = null_log_losses.iter().sum::<f64>() / null_log_losses.len() as f64;
            let null_hazard_quadratic = null_hazard_quadratic_losses.iter().sum::<f64>()
                / null_hazard_quadratic_losses.len() as f64;
            // `lifted_brier` is the relative IPCW-Brier skill of the model over
            // the Kaplan–Meier null curve — consistent with `brier` now being a
            // genuine Brier score. The hazard-quadratic relative skill (what
            // this used to report) is preserved as `lifted_hazard_quadratic_score`.
            let null_ibs = gam::families::survival::predict::integrated_ipcw_brier_score(
                null_surv.view(),
                &event_times,
                &events,
                &grid,
                horizon,
                |t| censoring_km.at(t),
            );
            if let (Some(model_ibs), Some(null_ibs)) = (ibs, null_ibs) {
                out.set_item(
                    "lifted_brier",
                    (null_ibs - model_ibs) / null_ibs.abs().max(eps),
                )?;
            }
            out.set_item(
                "lifted_hazard_quadratic_score",
                (null_hazard_quadratic - hazard_quadratic) / null_hazard_quadratic.abs().max(eps),
            )?;
            out.set_item(
                "lifted_logloss",
                (null_logloss - logloss) / null_logloss.abs().max(eps),
            )?;
            let ll_model = -log_losses.iter().sum::<f64>();
            let ll_null = -null_log_losses.iter().sum::<f64>();
            nagelkerke =
                gam::inference::diagnostics::nagelkerke_r_squared_from_log_likelihoods(
                    ll_model,
                    ll_null,
                    event_times.len(),
                );
        }
    }
    match nagelkerke {
        Some(value) => out.set_item("nagelkerke_r2", value)?,
        None => out.set_item("nagelkerke_r2", py.None())?,
    }
    Ok(out.unbind())
}

fn sigmoid(value: f64) -> f64 {
    1.0 / (1.0 + (-value).exp())
}

/// Deterministic synthetic-fixture RNG for the Python-facing `synthetic_*`
/// generators below. Built on the canonical SplitMix64 stream
/// ([`gam::utils::splitmix64`]) rather than a bespoke LCG so every seeded draw
/// here traces back to the one hash primitive the rest of the codebase uses.
struct SplitMixNormalRng {
    state: u64,
    spare_normal: Option<f64>,
}

impl SplitMixNormalRng {
    fn new(seed: u64) -> Self {
        Self {
            state: seed,
            spare_normal: None,
        }
    }

    /// A uniform draw on the open interval `(0, 1)`.
    fn uniform_open01(&mut self) -> f64 {
        let bits = gam::utils::splitmix64(&mut self.state) >> 11;
        (bits as f64 + 0.5) / ((1_u64 << 53) as f64)
    }

    fn uniform_range(&mut self, lo: f64, hi: f64) -> f64 {
        lo + (hi - lo) * self.uniform_open01()
    }

    /// A standard-normal draw via the Box-Muller transform.
    fn standard_normal(&mut self) -> f64 {
        if let Some(value) = self.spare_normal.take() {
            return value;
        }
        let radius = (-2.0 * self.uniform_open01().ln()).sqrt();
        let angle = std::f64::consts::TAU * self.uniform_open01();
        let first = radius * angle.cos();
        self.spare_normal = Some(radius * angle.sin());
        first
    }

    fn normal(&mut self, mean: f64, sd: f64) -> f64 {
        mean + sd * self.standard_normal()
    }

    fn bernoulli(&mut self, p: f64) -> bool {
        self.uniform_open01() < p
    }
}

fn standardize_vector(values: &mut [f64]) {
    if values.is_empty() {
        return;
    }
    let mean = values.iter().sum::<f64>() / values.len() as f64;
    let sd = (values
        .iter()
        .map(|value| {
            let d = *value - mean;
            d * d
        })
        .sum::<f64>()
        / values.len() as f64)
        .sqrt();
    let denom = if sd.is_finite() && sd >= 1.0e-12 {
        sd
    } else {
        1.0
    };
    for value in values {
        *value = (*value - mean) / denom;
    }
}

#[pyfunction]
fn synthetic_binomial_columns<'py>(
    py: Python<'py>,
    n: usize,
    p: usize,
    seed: u64,
) -> PyResult<Py<PyDict>> {
    let n = n.max(1);
    let p = p.max(3);
    let mut rng = SplitMixNormalRng::new(seed);
    let mut x = Array2::<f64>::zeros((n, p - 1));
    let mut y = Array1::<f64>::zeros(n);
    for i in 0..n {
        for j in 0..p - 1 {
            x[[i, j]] = rng.standard_normal();
        }
        let eta = -0.25 + 1.1 * x[[i, 0]] - 0.9 * x[[i, 1]] + 0.2 * x[[i, 1]].sin();
        y[i] = if rng.bernoulli(sigmoid(eta)) {
            1.0
        } else {
            0.0
        };
    }
    let out = PyDict::new(py);
    out.set_item("x", x.into_pyarray(py))?;
    out.set_item("y", y.into_pyarray(py))?;
    Ok(out.unbind())
}

#[pyfunction]
fn synthetic_geo_disease_columns<'py>(
    py: Python<'py>,
    n: usize,
    seed: u64,
) -> PyResult<Py<PyDict>> {
    let n = n.max(500);
    let mut rng = SplitMixNormalRng::new(seed);
    let mut x = Array2::<f64>::zeros((n, 16));
    let mut y = Array1::<f64>::zeros(n);
    for i in 0..n {
        let lat = rng.uniform_range(-1.0, 1.0);
        let lon = rng.uniform_range(-1.0, 1.0);
        let equator = 1.0 - lat.abs();
        let geo_signal = -1.0
            + 2.20 * equator
            + 0.55 * (std::f64::consts::PI * lon).sin()
            + 0.35 * (2.25 * std::f64::consts::PI * lon).cos()
            + 0.30 * (2.0 * std::f64::consts::PI * equator * lon).sin();
        let southness = (-lat).clamp(0.0, 1.0);
        let eta = geo_signal + rng.normal(0.0, 0.20 + 0.85 * southness.powf(1.35));
        y[i] = if rng.bernoulli(sigmoid(eta)) {
            1.0
        } else {
            0.0
        };
        for j in 0..16 {
            let jf = j as f64;
            let a = 0.95 - 0.045 * jf;
            let b = 0.25 + 0.035 * jf;
            let c = if j % 2 == 0 { 1.0 } else { -1.0 } * (0.10 + 0.01 * jf);
            let noise_sd = 0.15 + 0.015 * jf;
            x[[i, j]] = a * lat + b * lon + c * lat * lon + rng.normal(0.0, noise_sd);
        }
    }
    let out = PyDict::new(py);
    out.set_item("x", x.into_pyarray(py))?;
    out.set_item("y", y.into_pyarray(py))?;
    Ok(out.unbind())
}

#[pyfunction(signature = (mode, n, seed, true_nu = None, true_kappa2 = None))]
fn synthetic_continuous_order_columns<'py>(
    py: Python<'py>,
    mode: String,
    n: usize,
    seed: u64,
    true_nu: Option<f64>,
    true_kappa2: Option<f64>,
) -> PyResult<Py<PyDict>> {
    let n = n.max(128);
    let mut rng = SplitMixNormalRng::new(seed);
    let mut x = Array1::<f64>::zeros(n);
    let mut y = Array1::<f64>::zeros(n);
    let mut latent = vec![0.0; n];
    for i in 0..n {
        x[i] = -1.0 + 2.0 * i as f64 / (n.saturating_sub(1)).max(1) as f64;
    }
    match mode.as_str() {
        "fractional" => {
            let nu = true_nu.unwrap_or(1.8).max(0.1);
            let k2 = true_kappa2.unwrap_or(0.7).max(1.0e-9);
            let harmonics = 32usize.min(n / 2).max(4);
            for h in 1..=harmonics {
                let freq = h as f64;
                let amp = (k2 + freq * freq).powf(-(nu + 0.5) * 0.5);
                let phase = rng.uniform_range(0.0, std::f64::consts::TAU);
                let weight = rng.standard_normal() * amp;
                for i in 0..n {
                    latent[i] += weight
                        * (std::f64::consts::TAU * freq * (i as f64 / n as f64) + phase).sin();
                }
            }
            standardize_vector(&mut latent);
            for i in 0..n {
                y[i] = latent[i] + rng.normal(0.0, 0.20);
            }
        }
        "rough" => {
            let mut acc = 0.0;
            for value in &mut latent {
                acc += rng.standard_normal();
                *value = acc;
            }
            standardize_vector(&mut latent);
            for i in 0..n {
                y[i] = latent[i] + rng.normal(0.0, 0.30);
            }
        }
        "smooth" => {
            for i in 0..n {
                latent[i] = 1.4 * (2.0 * std::f64::consts::PI * (x[i] + 0.1)).sin()
                    + 0.8 * (0.5 * std::f64::consts::PI * (x[i] - 0.2)).cos();
            }
            standardize_vector(&mut latent);
            for i in 0..n {
                y[i] = latent[i] + rng.normal(0.0, 0.03);
            }
        }
        other => {
            return Err(PyValueError::new_err(format!(
                "unsupported continuous-order synthetic mode '{other}'"
            )));
        }
    }
    let out = PyDict::new(py);
    out.set_item("x", x.into_pyarray(py))?;
    out.set_item("y", y.into_pyarray(py))?;
    Ok(out.unbind())
}

#[pyfunction]
fn synthetic_thread3_admixture_cliff_columns<'py>(
    py: Python<'py>,
    n: usize,
    seed: u64,
) -> PyResult<Py<PyDict>> {
    let n = n.max(500);
    let mut rng = SplitMixNormalRng::new(seed);
    let mut x = Array2::<f64>::zeros((n, 16));
    let mut y = Array1::<f64>::zeros(n);
    let coeffs = [1.0, 0.35, -0.20, 0.10];
    for i in 0..n {
        let z0 = rng.standard_normal();
        let z1 = rng.standard_normal();
        let z2 = rng.standard_normal();
        let z3 = rng.standard_normal();
        let pc1 = z0;
        let pc2 = 0.52 * z0 + (1.0_f64 - 0.52_f64 * 0.52_f64).sqrt() * z1;
        let pc3 = -0.18 * z0 + 0.22 * z1 + 0.96 * z2;
        let pc4 = 0.10 * z0 - 0.15 * z1 + 0.35 * z2 + 0.92 * z3;
        let pcs = [pc1, pc2, pc3, pc4];
        for j in 0..4 {
            x[[i, j]] = pcs[j];
        }
        let cliff_axis = coeffs
            .iter()
            .zip(pcs.iter())
            .map(|(a, b)| a * b)
            .sum::<f64>();
        let eta = -1.15 + 3.8 * (16.0 * cliff_axis).tanh() + rng.normal(0.0, 0.15);
        y[i] = if rng.bernoulli(sigmoid(eta)) {
            1.0
        } else {
            0.0
        };
        for j in 0..12 {
            let jf = j as f64;
            let a = 0.45 - 0.02 * jf;
            let b = if j % 2 == 0 { 1.0 } else { -1.0 } * (0.18 + 0.01 * jf);
            let c = 0.12 + 0.02 * (j % 4) as f64;
            x[[i, j + 4]] = a * pc1 + b * pc2 + c * pc4 + rng.normal(0.0, 0.18 + 0.02 * jf);
        }
    }
    let out = PyDict::new(py);
    out.set_item("x", x.into_pyarray(py))?;
    out.set_item("y", y.into_pyarray(py))?;
    Ok(out.unbind())
}

#[pyfunction]
fn synthetic_geo_disease_eas_columns<'py>(
    py: Python<'py>,
    n: usize,
    seed: u64,
    n_pcs: usize,
) -> PyResult<Py<PyDict>> {
    let n = n.max(5);
    let n_pcs = n_pcs.max(3);
    let mut rng = SplitMixNormalRng::new(seed);
    let mut x = Array2::<f64>::zeros((n, n_pcs));
    let mut y = Array1::<f64>::zeros(n);
    for i in 0..n {
        let eas = rng.bernoulli(0.23);
        let lat = if eas {
            rng.uniform_range(15.0, 52.0)
        } else {
            rng.uniform_range(-55.0, 70.0)
        };
        let lon = if eas {
            rng.uniform_range(95.0, 145.0)
        } else {
            rng.uniform_range(-175.0, 175.0)
        };
        let mut eta = if eas {
            (0.10_f64 / 0.90_f64).ln()
        } else {
            (0.02_f64 / 0.98_f64).ln()
        };
        if eas {
            let lat_e = (lat - 33.5) / 11.0;
            let lon_e = (lon - 120.0) / 10.0;
            eta += 3.25 * (1.35 * lat_e).sin() - 2.85 * (1.55 * lon_e).cos()
                + 2.50 * (1.10 * lat_e * lon_e).sin()
                + 1.90 * (1.60 * lat_e + 0.45 * lon_e).cos()
                + rng.normal(0.0, 0.20);
        } else {
            eta += rng.normal(0.0, 0.08);
        }
        y[i] = if rng.bernoulli(sigmoid(eta)) {
            1.0
        } else {
            0.0
        };
        let lat_s = lat / 90.0;
        let lon_s = lon / 180.0;
        for j in 0..n_pcs {
            let jf = j as f64;
            let a = 0.98 - 0.05 * jf;
            let b = 0.26 + 0.03 * jf;
            let c = if j % 2 == 0 { 1.0 } else { -1.0 } * (0.12 + 0.01 * jf);
            let d = if j >= 8 { 0.22 } else { 0.06 };
            x[[i, j]] = a * lat_s
                + b * lon_s
                + c * lat_s * lon_s
                + d * (if eas { 1.0 } else { 0.0 })
                + rng.normal(0.0, 0.13 + 0.018 * jf);
        }
    }
    let out = PyDict::new(py);
    out.set_item("x", x.into_pyarray(py))?;
    out.set_item("y", y.into_pyarray(py))?;
    Ok(out.unbind())
}

#[pyfunction]
fn synthetic_papuan_oce_columns<'py>(
    py: Python<'py>,
    n: usize,
    seed: u64,
    n_pcs: usize,
) -> PyResult<Py<PyDict>> {
    let n = n.max(1200);
    let n_pcs = n_pcs.max(3);
    let mut rng = SplitMixNormalRng::new(seed);
    let centers = [
        [2.7, -0.8, 1.7, 0.7],
        [3.0, -0.6, 1.5, 0.8],
        [2.9, -0.3, 1.9, 0.4],
        [3.1, -0.5, 1.6, 0.8],
        [2.5, -1.0, 1.8, 0.5],
        [0.4, 2.0, -0.1, 0.2],
        [0.7, 1.8, -0.2, 0.1],
        [-2.1, 0.4, 0.1, -0.2],
        [-1.9, -2.2, 0.5, 0.0],
        [-0.3, -0.9, -1.6, 0.4],
    ];
    let probs = [0.12, 0.08, 0.06, 0.06, 0.08, 0.17, 0.10, 0.14, 0.11, 0.08];
    let mut x = Array2::<f64>::zeros((n, n_pcs));
    let mut y = Array1::<f64>::zeros(n);
    for i in 0..n {
        let draw = rng.uniform_open01();
        let mut cum = 0.0;
        let mut group = 0usize;
        for (idx, prob) in probs.iter().enumerate() {
            cum += prob;
            if draw <= cum {
                group = idx;
                break;
            }
        }
        let mut z = [0.0; 4];
        for d in 0..4 {
            z[d] = centers[group][d] + rng.normal(0.0, 0.55);
        }
        for j in 0..n_pcs {
            let d = j % 4;
            let mix = z[d] + 0.35 * z[(d + 1) % 4] - 0.18 * z[(d + 2) % 4];
            x[[i, j]] = mix + rng.normal(0.0, 0.20 + 0.02 * j as f64);
        }
        let high_risk = group <= 4;
        y[i] = if rng.bernoulli(if high_risk { 0.40 } else { 0.02 }) {
            1.0
        } else {
            0.0
        };
    }
    for j in 0..n_pcs {
        let mut col: Vec<f64> = (0..n).map(|i| x[[i, j]]).collect();
        standardize_vector(&mut col);
        for i in 0..n {
            x[[i, j]] = col[i];
        }
    }
    let out = PyDict::new(py);
    out.set_item("x", x.into_pyarray(py))?;
    out.set_item("y", y.into_pyarray(py))?;
    Ok(out.unbind())
}

#[pyfunction]
fn synthetic_hgdp_pc_panel_columns<'py>(py: Python<'py>, seed: u64) -> PyResult<Py<PyDict>> {
    let mut rng = SplitMixNormalRng::new(seed);
    let specs = [
        ("AFR", 2.0, 20.0),
        ("EUR", 50.0, 15.0),
        ("EAS", 35.0, 115.0),
        ("SAS", 20.0, 78.0),
        ("AMR", -12.0, -72.0),
        ("OCE", -8.0, 145.0),
    ];
    let rows_per_subpop = 40usize;
    let n = specs.len() * 4 * rows_per_subpop;
    let mut pc = Array2::<f64>::zeros((n, 16));
    let mut sample_ids = Vec::with_capacity(n);
    let mut superpops = Vec::with_capacity(n);
    let mut subpops = Vec::with_capacity(n);
    let mut latitudes = Vec::with_capacity(n);
    let mut longitudes = Vec::with_capacity(n);
    let mut row_idx = 0usize;
    for &(name, base_lat, base_lon) in &specs {
        let mut super_shift = [0.0; 16];
        for value in &mut super_shift {
            *value = rng.normal(0.0, 0.85);
        }
        for sub_idx in 0..4 {
            let sub_name = format!("{name}_SUB{:02}", sub_idx + 1);
            let sub_lat = base_lat + rng.normal(0.0, 5.0);
            let sub_lon = base_lon + rng.normal(0.0, 7.5);
            let mut sub_shift = [0.0; 16];
            for value in &mut sub_shift {
                *value = rng.normal(0.0, 0.30);
            }
            for _ in 0..rows_per_subpop {
                let sample_lat = (sub_lat + rng.normal(0.0, 1.2)).clamp(-58.0, 72.0);
                let sample_lon =
                    ((sub_lon + rng.normal(0.0, 1.8) + 180.0).rem_euclid(360.0)) - 180.0;
                let lat_norm = sample_lat / 90.0;
                let lon_norm = sample_lon / 180.0;
                let geo_terms = [
                    lat_norm,
                    lon_norm,
                    lat_norm * lon_norm,
                    (std::f64::consts::PI * lat_norm).sin(),
                    (std::f64::consts::PI * lon_norm).cos(),
                    (std::f64::consts::PI * (lat_norm + lon_norm) / 2.0).sin(),
                    lat_norm.powi(2),
                    lon_norm.powi(2),
                    (std::f64::consts::PI * lat_norm).cos(),
                    (std::f64::consts::PI * lon_norm).sin(),
                    lat_norm - lon_norm,
                    lat_norm + lon_norm,
                    (std::f64::consts::PI * lat_norm * lon_norm).sin(),
                    (std::f64::consts::PI * (lat_norm - lon_norm) / 2.0).cos(),
                    lat_norm.powi(3),
                    lon_norm.powi(3),
                ];
                for j in 0..16 {
                    pc[[row_idx, j]] = 1.55 * super_shift[j]
                        + 0.90 * sub_shift[j]
                        + 1.20 * geo_terms[j]
                        + rng.normal(0.0, 0.18);
                }
                sample_ids.push(format!("sample_{row_idx:05}"));
                superpops.push(name.to_string());
                subpops.push(sub_name.clone());
                latitudes.push(sample_lat);
                longitudes.push(sample_lon);
                row_idx += 1;
            }
        }
    }
    let out = PyDict::new(py);
    out.set_item("sample_id", sample_ids)?;
    out.set_item("Superpopulation", superpops)?;
    out.set_item("Subpopulation", subpops)?;
    out.set_item("Latitude", latitudes)?;
    out.set_item("Longitude", longitudes)?;
    out.set_item("pc", pc.into_pyarray(py))?;
    Ok(out.unbind())
}

#[pyfunction]
fn synthetic_geo_subpop_response<'py>(
    py: Python<'py>,
    subpop_codes: Vec<usize>,
    seed: u64,
    prevalence_min: f64,
    prevalence_max: f64,
    noise_scale_min: f64,
    noise_scale_max: f64,
    random_scale: bool,
) -> PyResult<Py<PyArray1<f64>>> {
    let mut rng = SplitMixNormalRng::new(seed);
    let n_groups = subpop_codes.iter().copied().max().unwrap_or(0) + 1;
    let mut prevalence = vec![0.0; n_groups];
    let mut noise_scale = vec![0.0; n_groups];
    for group in 0..n_groups {
        prevalence[group] = rng
            .uniform_range(prevalence_min, prevalence_max)
            .clamp(1.0e-5, 1.0 - 1.0e-5);
        noise_scale[group] = if random_scale {
            rng.uniform_range(noise_scale_min, noise_scale_max)
        } else {
            rng.uniform_range(0.25, 0.85)
        };
    }
    let mut y = Array1::<f64>::zeros(subpop_codes.len());
    for (i, &group) in subpop_codes.iter().enumerate() {
        let p = prevalence[group];
        let eta = (p / (1.0 - p)).ln() + rng.normal(0.0, noise_scale[group]);
        y[i] = if rng.bernoulli(sigmoid(eta)) {
            1.0
        } else {
            0.0
        };
    }
    Ok(y.into_pyarray(py).unbind())
}

#[pyfunction]
fn synthetic_geo_latlon_response<'py>(
    py: Python<'py>,
    mode_code: String,
    superpop_codes: Vec<usize>,
    latitudes: Vec<f64>,
    longitudes: Vec<f64>,
    seed: u64,
    prevalence_min: f64,
    prevalence_max: f64,
) -> PyResult<Py<PyArray1<f64>>> {
    if latitudes.len() != longitudes.len() || latitudes.len() != superpop_codes.len() {
        return Err(PyValueError::new_err(
            "geo lat/lon response length mismatch",
        ));
    }
    let mut rng = SplitMixNormalRng::new(seed);
    let n_super = superpop_codes.iter().copied().max().unwrap_or(0) + 1;
    let mut super_noise = vec![0.0; n_super];
    for value in &mut super_noise {
        *value = rng.uniform_range(0.10, 0.90);
    }
    let mut y = Array1::<f64>::zeros(latitudes.len());
    for i in 0..latitudes.len() {
        let lat_norm = (latitudes[i].abs() / 90.0).clamp(0.0, 1.0);
        let lon_norm = ((longitudes[i] + 180.0) / 360.0).clamp(0.0, 1.0);
        let (base_prev, noise_sd) = match mode_code.as_str() {
            "superpopnoise" => {
                let risk = (0.68 * lat_norm + 0.32 * (1.0 - lon_norm)).clamp(0.0, 1.0);
                (
                    prevalence_min + (prevalence_max - prevalence_min) * risk,
                    super_noise[superpop_codes[i]],
                )
            }
            "equatornoise" => {
                let edge_risk = (longitudes[i].abs() / 180.0).clamp(0.0, 1.0);
                (
                    prevalence_min + (prevalence_max - prevalence_min) * edge_risk,
                    0.05 + 1.25 * (1.0 - lat_norm).clamp(0.0, 1.0),
                )
            }
            other => {
                return Err(PyValueError::new_err(format!(
                    "unsupported geo_latlon mode: {other}"
                )));
            }
        };
        let p = base_prev.clamp(1.0e-5, 1.0 - 1.0e-5);
        let eta = (p / (1.0 - p)).ln() + rng.normal(0.0, noise_sd);
        y[i] = if rng.bernoulli(sigmoid(eta)) {
            1.0
        } else {
            0.0
        };
    }
    Ok(y.into_pyarray(py).unbind())
}

#[pyfunction]
fn thread3_cliff_gradient_magnitude<'py>(
    py: Python<'py>,
    collocation_points: PyReadonlyArray2<'py, f64>,
    coefficients: Vec<f64>,
    jump: f64,
    sharpness: f64,
) -> PyResult<Option<Py<PyArray1<f64>>>> {
    let points = collocation_points.as_array();
    if points.nrows() == 0 || points.ncols() != coefficients.len() {
        return Ok(None);
    }
    let coeff_norm = coefficients
        .iter()
        .map(|value| value * value)
        .sum::<f64>()
        .sqrt();
    if coeff_norm < 1.0e-12 {
        return Ok(None);
    }
    let mut out = Array1::<f64>::zeros(points.nrows());
    for row in 0..points.nrows() {
        let mut z = 0.0;
        for col in 0..points.ncols() {
            z += points[[row, col]] * coefficients[col];
        }
        let az = (sharpness * z).abs().clamp(0.0, 50.0);
        let sech2 = 1.0 / az.cosh().powi(2);
        out[row] = (jump * sharpness).abs() * sech2 * coeff_norm;
    }
    Ok(Some(out.into_pyarray(py).unbind()))
}

// =========================================================================
// LatentCoord input-location derivative helpers (thin pyffi wrappers).
//
// Each function here is a thin shim over a `*_first_derivative_nd` helper
// in `gam::terms::basis`. The helpers are analytic, closed-form chain rules
// of the underlying basis with respect to the *first* kernel argument
// (the latent coordinate `t`). They share the same analytic machinery the
// kernel-parameter chain (`SpatialLogKappaCoords`) uses, re-pointed at
// `t` instead of at the anisotropy / log-kappa.
// =========================================================================

/// `φ'(r_{n,k})` for the Matérn kernel at every `(row, center)` pair.
/// Returns `(n_rows, n_centers)`. `nu` accepted: `"1/2"`, `"3/2"`, `"5/2"`,
/// `"7/2"`, `"9/2"` (any whitespace stripped).
fn parse_matern_nu_py(context: &str, nu: &str) -> PyResult<MaternNu> {
    match nu.replace(char::is_whitespace, "").as_str() {
        "1/2" | "0.5" => Ok(MaternNu::Half),
        "3/2" | "1.5" => Ok(MaternNu::ThreeHalves),
        "5/2" | "2.5" => Ok(MaternNu::FiveHalves),
        "7/2" | "3.5" => Ok(MaternNu::SevenHalves),
        "9/2" | "4.5" => Ok(MaternNu::NineHalves),
        other => Err(py_value_error(format!(
            "{context}: nu must be one of '1/2','3/2','5/2','7/2','9/2'; got {other:?}"
        ))),
    }
}

/// Full input-location first jet `∂Φ/∂t` of the Matérn kernel under the
/// anisotropic metric `r_A = √(Σ_a w_a (t_a − c_a)²)`.
///
/// Returns a `(n_rows, n_centers, dim)` tensor. `aniso_log_scales` is threaded
/// through the *same* centred-contrast transform the forward `matern_basis`
/// applies, so this jet exactly differentiates the forward design value even
/// when anisotropy is active (issue #437). With `aniso_log_scales = None` the
/// metric is isotropic.
#[pyfunction(signature = (t, centers, length_scale, nu = "3/2", aniso_log_scales = None))]
fn matern_input_location_first_jet<'py>(
    py: Python<'py>,
    t: PyReadonlyArray2<'py, f64>,
    centers: PyReadonlyArray2<'py, f64>,
    length_scale: f64,
    nu: &str,
    aniso_log_scales: Option<PyReadonlyArray1<'py, f64>>,
) -> PyResult<Py<PyArray3<f64>>> {
    let nu_parsed = parse_matern_nu_py("matern_input_location_first_jet", nu)?;
    let aniso_vec = aniso_log_scales
        .as_ref()
        .map(|values| values.as_slice())
        .transpose()
        .map_err(|err| py_value_error(format!("aniso_log_scales must be contiguous: {err}")))?
        .map(|slice| slice.to_vec());
    let out = matern_input_location_jet_nd(
        t.as_array(),
        centers.as_array(),
        length_scale,
        nu_parsed,
        aniso_vec.as_deref(),
    )
    .map_err(basis_error_to_pyerr)?;
    Ok(out.into_pyarray(py).unbind())
}

/// Full input-location Hessian `∂²Φ/∂t∂tᵀ` of the Matérn kernel under the
/// anisotropic metric. Companion to `matern_input_location_first_jet`;
/// returns a `(n_rows, n_centers, dim, dim)` tensor exactly matching the
/// second derivative of the forward kernel value. `aniso_log_scales` follows
/// the same forward centred-contrast transform.
#[pyfunction(signature = (t, centers, length_scale, nu = "3/2", aniso_log_scales = None))]
fn matern_input_location_hessian<'py>(
    py: Python<'py>,
    t: PyReadonlyArray2<'py, f64>,
    centers: PyReadonlyArray2<'py, f64>,
    length_scale: f64,
    nu: &str,
    aniso_log_scales: Option<PyReadonlyArray1<'py, f64>>,
) -> PyResult<Py<PyArray4<f64>>> {
    let nu_parsed = parse_matern_nu_py("matern_input_location_hessian", nu)?;
    let aniso_vec = aniso_log_scales
        .as_ref()
        .map(|values| values.as_slice())
        .transpose()
        .map_err(|err| py_value_error(format!("aniso_log_scales must be contiguous: {err}")))?
        .map(|slice| slice.to_vec());
    let out = matern_input_location_hessian_nd(
        t.as_array(),
        centers.as_array(),
        length_scale,
        nu_parsed,
        aniso_vec.as_deref(),
    )
    .map_err(basis_error_to_pyerr)?;
    Ok(out.into_pyarray(py).unbind())
}

/// Streaming closed-form `dK/dtheta` for Matérn basis values.
///
/// `target="log_kappa"` returns the global scale derivative. `target="aniso"`
/// requires `axis` and returns the derivative for that axis' log-scale.
#[pyfunction(signature = (
    data,
    centers,
    length_scale,
    nu = "3/2",
    target = "log_kappa",
    axis = None,
    aniso_log_scales = None,
    chunk_size = None
))]
fn matern_basis_gradient_streaming<'py>(
    py: Python<'py>,
    data: PyReadonlyArray2<'py, f64>,
    centers: PyReadonlyArray2<'py, f64>,
    length_scale: f64,
    nu: &str,
    target: &str,
    axis: Option<usize>,
    aniso_log_scales: Option<PyReadonlyArray1<'py, f64>>,
    chunk_size: Option<usize>,
) -> PyResult<Py<PyArray2<f64>>> {
    let nu_parsed = parse_matern_nu_py("matern_basis_gradient_streaming", nu)?;
    let target = match target
        .trim()
        .to_ascii_lowercase()
        .replace('-', "_")
        .as_str()
    {
        "log_kappa" | "kappa" | "scale" => MaternBasisGradientTarget::LogKappa,
        "aniso" | "aniso_log_scale" | "axis" => {
            MaternBasisGradientTarget::AnisoLogScale(axis.ok_or_else(|| {
                py_value_error(
                    "matern_basis_gradient_streaming: axis is required for target='aniso'"
                        .to_string(),
                )
            })?)
        }
        other => {
            return Err(py_value_error(format!(
                "matern_basis_gradient_streaming: target must be 'log_kappa' or 'aniso'; got {other:?}"
            )));
        }
    };
    let aniso = aniso_log_scales
        .as_ref()
        .map(|values| values.as_slice())
        .transpose()
        .map_err(|err| py_value_error(format!("aniso_log_scales must be contiguous: {err}")))?;
    let evaluator = StreamingMaternBasisGradientEvaluator::new(
        centers.as_array(),
        length_scale,
        nu_parsed,
        aniso,
        chunk_size,
    )
    .map_err(basis_error_to_pyerr)?;
    let out = evaluator
        .evaluate(data.as_array(), target)
        .map_err(basis_error_to_pyerr)?;
    Ok(out.into_pyarray(py).unbind())
}

/// Closed-form `(N, n_centers, dim)` jet of the Sobolev sphere kernel
/// w.r.t. ambient coordinates `t_n ∈ S^{dim-1}`.
///
/// The returned jet is tangent-projected by default for the intrinsic
/// Riemannian gradient (`g − (g · t) t`) used by embedded sphere updates.
/// Pass `project_to_tangent = false` to receive the ambient gradient.
#[pyfunction(signature = (points, centers, penalty_order = 2, project_to_tangent = true))]
fn sphere_input_location_first_derivative<'py>(
    py: Python<'py>,
    points: PyReadonlyArray2<'py, f64>,
    centers: PyReadonlyArray2<'py, f64>,
    penalty_order: usize,
    project_to_tangent: bool,
) -> PyResult<Py<PyArray3<f64>>> {
    let jet = sphere_first_derivative_nd(
        points.as_array(),
        centers.as_array(),
        penalty_order,
        project_to_tangent,
    )
    .map_err(basis_error_to_pyerr)?;
    Ok(jet.into_pyarray(py).unbind())
}

/// Closed-form `(N, num_basis, 1)` jet of the cyclic-B-spline basis w.r.t.
/// the scalar latent coordinate. `t` is `(N, 1)`.
#[pyfunction]
fn periodic_bspline_input_location_first_derivative<'py>(
    py: Python<'py>,
    t: PyReadonlyArray2<'py, f64>,
    data_range_left: f64,
    data_range_right: f64,
    degree: usize,
    num_basis: usize,
) -> PyResult<Py<PyArray3<f64>>> {
    let jet = periodic_bspline_first_derivative_nd(
        t.as_array(),
        (data_range_left, data_range_right),
        degree,
        num_basis,
    )
    .map_err(basis_error_to_pyerr)?;
    Ok(jet.into_pyarray(py).unbind())
}

/// Closed-form `(N, K_total, n_axes)` tensor-product B-spline derivative
/// jet, via the product rule. `knots_concat` is the per-axis knot vectors
/// flattened together; `knot_offsets` is `[0, len_axis_0, len_axis_0 +
/// len_axis_1, ...]` (length `n_axes + 1`) — i.e. the standard CSR-style
/// concat. `degrees` is per-axis.
#[pyfunction]
fn bspline_tensor_input_location_first_derivative<'py>(
    py: Python<'py>,
    t: PyReadonlyArray2<'py, f64>,
    knots_concat: PyReadonlyArray1<'py, f64>,
    knot_offsets: Vec<usize>,
    degrees: Vec<usize>,
) -> PyResult<Py<PyArray3<f64>>> {
    let n_axes = degrees.len();
    if knot_offsets.len() != n_axes + 1 {
        return Err(py_value_error(format!(
            "bspline_tensor_input_location_first_derivative: knot_offsets must have \
             length n_axes + 1 = {}, got {}",
            n_axes + 1,
            knot_offsets.len()
        )));
    }
    if t.as_array().ncols() != n_axes {
        return Err(py_value_error(format!(
            "bspline_tensor_input_location_first_derivative: t has {} cols but \
             degrees has {} entries",
            t.as_array().ncols(),
            n_axes
        )));
    }
    let knots_concat_view = knots_concat.as_array();
    // Slice the concatenated knot vector into per-axis views.
    let mut per_axis_views: Vec<ArrayView1<'_, f64>> = Vec::with_capacity(n_axes);
    for a in 0..n_axes {
        let lo = knot_offsets[a];
        let hi = knot_offsets[a + 1];
        if hi > knots_concat_view.len() || lo > hi {
            return Err(py_value_error(format!(
                "bspline_tensor_input_location_first_derivative: knot_offsets axis {a} \
                 out of range (lo={lo}, hi={hi}, total={})",
                knots_concat_view.len()
            )));
        }
        per_axis_views.push(knots_concat_view.slice(s![lo..hi]));
    }
    let jet = bspline_tensor_first_derivative(t.as_array(), &per_axis_views, &degrees)
        .map_err(basis_error_to_pyerr)?;
    Ok(jet.into_pyarray(py).unbind())
}

#[pyfunction(signature = (group, w, g, z, weight, ard_weight, log_bandwidth = None))]
fn equivariant_penalty_value<'py>(
    group: String,
    w: PyReadonlyArray3<'py, f64>,
    g: PyReadonlyArrayDyn<'py, f64>,
    z: PyReadonlyArray2<'py, f64>,
    weight: f64,
    ard_weight: f64,
    log_bandwidth: Option<PyReadonlyArray1<'py, f64>>,
) -> PyResult<f64> {
    let log_bandwidth = log_bandwidth.as_ref().map(|values| values.as_array());
    gam::terms::analytic_penalties::equivariant_penalty::equivariant_penalty_value(
        group.as_str(),
        w.as_array(),
        g.as_array(),
        z.as_array(),
        weight,
        ard_weight,
        log_bandwidth,
    )
    .map_err(py_value_error)
}

// ===========================================================================
// Response-geometry transforms (simplex + sphere) and equivariant rho/jvp +
// gauge-companion loss — rustified from gamfit/_response_geometry.py and
// gamfit/_equivariant.py. Each pyfunction is a thin marshalled entrypoint
// that delegates to a pure-Rust impl on ndarray views.
// ===========================================================================

/// Project a spherical base point onto the unit sphere, rejecting a zero-norm
/// input. Shared by the explicit `response_geometry_sphere_normalize_base` FFI
/// and the consolidated log-map dispatch.
fn rg_normalize_sphere_base(base: ArrayView1<'_, f64>) -> Result<Array1<f64>, String> {
    let norm = base.iter().fold(0.0_f64, |acc, value| acc.hypot(*value));
    if !norm.is_finite() || norm <= 0.0 {
        return Err("spherical base point must have non-zero norm".to_string());
    }
    Ok(base.mapv(|v| v / norm))
}

/// Resolve the simplex coordinate label exactly as the response-geometry log/exp
/// maps require: an explicit `coordinates` request wins (lower-cased), otherwise
/// `alr` for an `alr` geometry and `clr` for everything else.
fn rg_resolve_simplex_coord_label(kind: &str, coordinates: Option<&str>) -> String {
    match coordinates {
        Some(c) => c.to_ascii_lowercase(),
        None => {
            if kind == "alr" {
                "alr".to_string()
            } else {
                "clr".to_string()
            }
        }
    }
}

/// Consolidated response-geometry log map: pick the base point (intrinsic
/// Fréchet mean when none is supplied, else the projected/closed input base),
/// dispatch to the sphere or simplex log map, and report the resolved
/// coordinate label. This owns the geometry-kind routing, coordinate
/// resolution, and base-point selection that previously lived in the Python
/// wrapper.
///
/// `weights` are the per-observation prior weights used ONLY to choose the
/// intrinsic base point when `base` is `None`. A weighted response-geometry fit
/// linearizes every response in the tangent space at the base point and runs a
/// weighted tangent regression there; if the base point ignored the weights the
/// chart would be expanded around where the *unweighted* data balances rather
/// than where the weighted mass lives — a biased linearization whenever the
/// weighted and unweighted intrinsic means differ (#2125). Threading the same
/// weights the tangent regression uses into the Fréchet mean puts the chart
/// origin at the weighted mean. `None` recovers the uniform intrinsic mean.
fn rg_log_map_dispatch(
    values: ArrayView2<'_, f64>,
    geometry: &str,
    base: Option<ArrayView1<'_, f64>>,
    coordinates: Option<&str>,
    reference: isize,
    weights: Option<ArrayView1<'_, f64>>,
) -> Result<(Array2<f64>, Array1<f64>, String), String> {
    let kind = geometry.to_ascii_lowercase();
    match kind.as_str() {
        "spherical" | "sphere" => {
            let base_point = match base {
                None => Array1::from(gam::geometry::sphere::sphere_frechet_mean(
                    values, weights, 1.0e-12, 256,
                )?),
                Some(b) => rg_normalize_sphere_base(b)?,
            };
            let tangent =
                gam::geometry::sphere::response_sphere_log_map(values, base_point.view())?;
            Ok((tangent, base_point, "spherical".to_string()))
        }
        "simplex" | "clr" | "alr" => {
            let coord_label = rg_resolve_simplex_coord_label(&kind, coordinates);
            let coord = gam::geometry::simplex::parse_simplex_coord(&coord_label)?;
            let base_point = match base {
                None => Array1::from(simplex_frechet_mean(values, weights)?),
                Some(b) => {
                    let b2 = Array2::from_shape_fn((1, b.len()), |(_, j)| b[j]);
                    simplex_closure(b2.view())?.row(0).to_owned()
                }
            };
            let tangent = gam::geometry::simplex::simplex_log_map(
                values,
                base_point.view(),
                coord,
                reference,
            )?;
            Ok((tangent, base_point, coord_label))
        }
        // Curved matrix / hyperbolic response geometries (#1061): the math is in
        // `gam::geometry` and the label-parsing + intrinsic-mean + batched-map
        // routing lives in `response_geometry`. `reference` does not apply.
        _ => gam::geometry::response_geometry::dispatch_log_map(values, &kind, base, weights),
    }
}

/// Consolidated response-geometry exponential map: dispatch tangent coordinates
/// back to the response manifold given the geometry kind and (already resolved)
/// coordinate label.
fn rg_exp_map_dispatch(
    tangent: ArrayView2<'_, f64>,
    geometry: &str,
    base: ArrayView1<'_, f64>,
    coordinates: Option<&str>,
    reference: isize,
) -> Result<Array2<f64>, String> {
    let kind = geometry.to_ascii_lowercase();
    match kind.as_str() {
        "spherical" | "sphere" => gam::geometry::sphere::response_sphere_exp_map(tangent, base),
        "simplex" | "clr" | "alr" => {
            let coord_label = rg_resolve_simplex_coord_label(&kind, coordinates);
            let coord = gam::geometry::simplex::parse_simplex_coord(&coord_label)?;
            gam::geometry::simplex::simplex_exp_map(tangent, base, coord, reference)
        }
        // Curved matrix / hyperbolic response geometries (#1061).
        _ => gam::geometry::response_geometry::dispatch_exp_map(tangent, &kind, base),
    }
}

#[pyfunction]
fn response_geometry_closure<'py>(
    py: Python<'py>,
    values: PyReadonlyArray2<'py, f64>,
) -> PyResult<Py<PyArray2<f64>>> {
    let arr = values.as_array().to_owned();
    let out = detach_py_result(py, "response_geometry_closure", move || {
        simplex_closure(arr.view())
    })?;
    Ok(out.into_pyarray(py).unbind())
}

#[pyfunction]
fn response_geometry_clr<'py>(
    py: Python<'py>,
    values: PyReadonlyArray2<'py, f64>,
) -> PyResult<Py<PyArray2<f64>>> {
    let arr = values.as_array().to_owned();
    let out = detach_py_result(py, "response_geometry_clr", move || {
        gam::geometry::simplex::clr(arr.view())
    })?;
    Ok(out.into_pyarray(py).unbind())
}

#[pyfunction(signature = (values, reference = -1))]
fn response_geometry_alr<'py>(
    py: Python<'py>,
    values: PyReadonlyArray2<'py, f64>,
    reference: isize,
) -> PyResult<Py<PyArray2<f64>>> {
    let arr = values.as_array().to_owned();
    let out = detach_py_result(py, "response_geometry_alr", move || {
        gam::geometry::simplex::alr(arr.view(), reference)
    })?;
    Ok(out.into_pyarray(py).unbind())
}

#[pyfunction(signature = (coords, reference = -1))]
fn response_geometry_inverse_alr<'py>(
    py: Python<'py>,
    coords: PyReadonlyArray2<'py, f64>,
    reference: isize,
) -> PyResult<Py<PyArray2<f64>>> {
    let arr = coords.as_array().to_owned();
    let out = detach_py_result(py, "response_geometry_inverse_alr", move || {
        gam::geometry::simplex::inverse_alr(arr.view(), reference)
    })?;
    Ok(out.into_pyarray(py).unbind())
}

#[pyfunction(signature = (values, weights = None))]
fn response_geometry_simplex_frechet_mean<'py>(
    py: Python<'py>,
    values: PyReadonlyArray2<'py, f64>,
    weights: Option<PyReadonlyArray1<'py, f64>>,
) -> PyResult<Py<PyArray1<f64>>> {
    let arr = values.as_array().to_owned();
    let w_owned = weights.as_ref().map(|w| w.as_array().to_owned());
    let out = detach_py_result(py, "response_geometry_simplex_frechet_mean", move || {
        simplex_frechet_mean(arr.view(), w_owned.as_ref().map(|w| w.view()))
    })?;
    Ok(Array1::from_vec(out).into_pyarray(py).unbind())
}

// ------------------------------------------------------------------
// Poincaré ball / Lorentz model primitives.
// ------------------------------------------------------------------

#[pyfunction]
fn poincare_mobius_add<'py>(
    py: Python<'py>,
    u: PyReadonlyArray1<'py, f64>,
    v: PyReadonlyArray1<'py, f64>,
    curvature: f64,
) -> PyResult<Py<PyArray1<f64>>> {
    let u_owned = u.as_array().to_owned();
    let v_owned = v.as_array().to_owned();
    let out = detach_geometry_result(py, "poincare_mobius_add", move || {
        poincare_mobius_add_impl(u_owned.view(), v_owned.view(), curvature)
    })?;
    Ok(out.into_pyarray(py).unbind())
}

#[cfg(test)]
mod latent_glm_family_validation_tests {
    use super::latent_glm_family_from_str;
    use gam::types::ResponseFamily;

    /// The Tweedie compound-Poisson-Gamma variance power must lie strictly in
    /// (1, 2). Previously `latent_glm_family_from_str` only rejected non-finite
    /// `tweedie_p`, so an out-of-range power (e.g. 2.5, 1.0, 0.5) constructed a
    /// `ResponseFamily::Tweedie { p }` whose deviance/working-response formulas
    /// are undefined — surfacing only later as an opaque NaN-deviance fit
    /// failure. This now mirrors the eager negbin/beta nuisance-parameter
    /// checks and rejects the bad power up front with an actionable message.
    #[test]
    fn tweedie_rejects_out_of_range_variance_power() {
        for &bad in &[0.5_f64, 1.0, 2.0, 2.5, -1.0, f64::INFINITY, f64::NAN] {
            let result = latent_glm_family_from_str("tweedie", bad, 1.0, 1.0);
            assert!(
                result.is_err(),
                "tweedie_p={bad} is outside (1, 2) and must be rejected"
            );
            let msg = result.err().unwrap();
            assert!(
                msg.contains("between 1 and 2"),
                "error message must explain the (1, 2) constraint; got {msg:?}"
            );
        }
    }

    /// A valid in-range Tweedie power is still accepted and yields the expected
    /// `ResponseFamily::Tweedie { p }` (the validation does not over-reject).
    #[test]
    fn tweedie_accepts_canonical_variance_power() {
        for &good in &[1.1_f64, 1.5, 1.9] {
            let spec = latent_glm_family_from_str("tweedie", good, 1.0, 1.0)
                .expect("in-range Tweedie power must be accepted");
            match spec.response {
                ResponseFamily::Tweedie { p } => assert_eq!(p, good),
                other => panic!("expected Tweedie family, got {other:?}"),
            }
        }
    }
}

// #1388/#2138 — SAE joint-fit worker driver (used by `sae_manifold_fit_inner` in
// the sibling `latent_basis_and_sae_ffi.rs` fragment; both are `include!`d into
// the same crate module, so this item is visible there). 512 MiB stack: the
// outer-ρ per-row jet loop's multi-megabyte `Tower4<16>` frames overflow Python's
// calling-thread stack → SIGSEGV; mirror the native CLI (`CLI_WORKER_STACK_SIZE`
// in `src/main.rs`).
const SAE_FIT_WORKER_STACK_SIZE: usize = 512 << 20;

/// Run an owned SAE fit closure on a 512 MiB worker thread with the GIL RELEASED
/// (#2138), so a multi-minute Rust solve no longer holds the interpreter lock for
/// its whole duration. The calling thread waits on the result slot in short
/// GIL-dropped windows and, between them, reacquires the GIL to run any pending
/// Python signal handler ([`Python::check_signals`]): a `KeyboardInterrupt` /
/// `signal.alarm` RAISES here mid-solve instead of only after the fit returns. On
/// interrupt `cancel` is set — the fit objective shares it and bails out of its
/// next outer eval so the detached worker stops — then the worker is abandoned;
/// it owns every input (`'static`) and drops on its own thread. `f` MUST NOT
/// touch a Python object.
fn run_sae_fit_interruptible<T, F>(
    py: Python<'_>,
    thread_name: &str,
    cancel: &std::sync::Arc<std::sync::atomic::AtomicBool>,
    f: F,
) -> PyResult<T>
where
    T: Send + 'static,
    F: FnOnce() -> T + Send + 'static,
{
    // Sync result slot: an mpsc `Receiver` is `!Sync` (so `&Receiver` is `!Ungil`
    // and cannot cross `detach`); `Arc<(Mutex, Condvar)>` is `Send + Sync`.
    // Poison is recovered, never panicked.
    let slot: std::sync::Arc<(std::sync::Mutex<Option<T>>, std::sync::Condvar)> =
        std::sync::Arc::new((std::sync::Mutex::new(None), std::sync::Condvar::new()));
    let worker_slot = std::sync::Arc::clone(&slot);
    std::thread::Builder::new()
        .name(thread_name.to_string())
        .stack_size(SAE_FIT_WORKER_STACK_SIZE)
        .spawn(move || {
            let out = f();
            let (lock, cvar) = &*worker_slot;
            let mut guard = lock.lock().unwrap_or_else(|poisoned| poisoned.into_inner());
            *guard = Some(out);
            cvar.notify_all();
        })
        .map_err(|err| {
            py_value_error(format!("sae_manifold_fit: spawn fit worker thread: {err}"))
        })?;
    let (lock, cvar) = &*slot;
    loop {
        // Wait up to 50 ms with the GIL released, then reacquire it to service
        // signals. The guard never escapes the closure, so the GIL is never
        // reacquired while the mutex is held.
        let taken = py.detach(|| {
            let guard = lock.lock().unwrap_or_else(|poisoned| poisoned.into_inner());
            let mut guard = cvar
                .wait_timeout(guard, std::time::Duration::from_millis(50))
                .unwrap_or_else(|poisoned| poisoned.into_inner())
                .0;
            guard.take()
        });
        if let Some(value) = taken {
            return Ok(value);
        }
        // On interrupt request cooperative cancellation so the abandoned worker's
        // next outer eval bails, then propagate the Python error.
        if let Err(err) = py.check_signals() {
            cancel.store(true, std::sync::atomic::Ordering::Relaxed);
            return Err(err);
        }
    }
}
