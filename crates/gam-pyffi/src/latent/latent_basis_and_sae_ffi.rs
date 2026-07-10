fn latent_basis_kind(value: &str) -> Result<&'static str, String> {
    match value.to_ascii_lowercase().replace(['_', '-'], "").as_str() {
        "duchon" | "duchonspline" => Ok("duchon"),
        // Dispatch hooks for the in-flight non-Duchon derivative helpers.
        // The call sites below are intentionally shaped around
        // `InputLocationDerivative::{Radial, Jet}` so Matérn can plug into
        // the radial path and sphere / tensor / periodic bases can plug into
        // the pre-computed-jet path without changing the contraction code.
        "matern" | "maternradial" => Ok("matern"),
        "sphere" | "spherical" => Ok("sphere"),
        "bsplinetensor" | "tensorbspline" => Ok("bspline_tensor"),
        "periodicbspline" | "periodicspline" => Ok("periodic_bspline"),
        other => Err(format!("unsupported latent basis_kind {other:?}")),
    }
}

fn radial_input_location_jet(
    t_mat: ArrayView2<'_, f64>,
    centers: ArrayView2<'_, f64>,
    phi_r: ArrayView2<'_, f64>,
) -> Result<Array3<f64>, String> {
    if phi_r.dim() != (t_mat.nrows(), centers.nrows()) {
        return Err(format!(
            "radial derivative shape {:?} does not match t/centers ({}, {})",
            phi_r.dim(),
            t_mat.nrows(),
            centers.nrows()
        ));
    }
    if t_mat.ncols() != centers.ncols() {
        return Err(format!(
            "radial derivative dimension mismatch: t has {} cols, centers has {}",
            t_mat.ncols(),
            centers.ncols()
        ));
    }
    let mut out = Array3::<f64>::zeros((t_mat.nrows(), centers.nrows(), t_mat.ncols()));
    for n in 0..t_mat.nrows() {
        for k in 0..centers.nrows() {
            let mut r2 = 0.0;
            for a in 0..t_mat.ncols() {
                let delta = t_mat[[n, a]] - centers[[k, a]];
                r2 += delta * delta;
            }
            let r = r2.sqrt();
            if r <= 1.0e-12 {
                continue;
            }
            let scale = phi_r[[n, k]] / r;
            for a in 0..t_mat.ncols() {
                out[[n, k, a]] = scale * (t_mat[[n, a]] - centers[[k, a]]);
            }
        }
    }
    Ok(out)
}

fn latent_input_location_jet(
    basis_kind: &str,
    t_mat: ArrayView2<'_, f64>,
    centers: ArrayView2<'_, f64>,
    m: usize,
    tensor_knots_concat: Option<ArrayView1<'_, f64>>,
    tensor_knot_offsets: Option<&[usize]>,
    tensor_degrees: Option<&[usize]>,
) -> Result<Array3<f64>, String> {
    match latent_basis_kind(basis_kind)? {
        "duchon" => {
            // Mirror the column layout used by `build_latent_duchon_design` /
            // `build_duchon_basis`: the forward design is the radial block
            // projected through the kernel-constraint nullspace Z
            // (p_constrained = n_centers − n_poly cols) concatenated with the
            // polynomial nullspace block (n_poly cols). The derivative jet
            // must use the same effective nullspace order and the same Z
            // projection so its column count matches the design exactly.
            // Resolve the SAME admissible (nullspace_order, power) pair the
            // forward `build_latent_duchon_design` resolves for this ambient
            // dimension (issue #875): the pure polyharmonic kernel exists only
            // when 2(p + s) > d, so `resolve_duchon_orders` may lift the
            // spectral power s (and null-space order) above the m-derived
            // request. The jet must differentiate the *resolved* forward kernel
            // — same power, same null-space — or its column count and scaling
            // would diverge from the design.
            let dim_ambient = centers.ncols();
            let (resolved_nullspace, resolved_power) =
                resolve_duchon_orders(dim_ambient, duchon_nullspace_from_m(m), 0, None);
            let effective_nullspace =
                pyffi_duchon_effective_nullspace_order(centers, resolved_nullspace);
            let radial_transform =
                pyffi_duchon_kernel_constraint_nullspace(centers, effective_nullspace)?;
            // The radial derivative differentiates the exact forward Green's
            // function: same scale-free pure Duchon spectrum (`length_scale =
            // None`, the resolved spectral power `s`), not a hard-coded
            // surrogate (issue #440).
            let phi_r = duchon_radial_first_derivative_nd(
                t_mat,
                centers,
                None,
                effective_nullspace,
                resolved_power,
            )
            .map_err(|err| err.to_string())?;
            let radial_jet = radial_input_location_jet(t_mat, centers, phi_r.view())?;
            let poly_jet = duchon_polynomial_first_derivative_nd(t_mat, effective_nullspace);

            let n_rows = radial_jet.shape()[0];
            let dim = radial_jet.shape()[2];
            let n_kernel = radial_transform.ncols();
            let n_poly = poly_jet.shape()[1];
            if poly_jet.shape()[0] != n_rows || poly_jet.shape()[2] != dim {
                return Err(format!(
                    "Duchon polynomial derivative shape mismatch: radial jet is \
                     {}x{}x{}, polynomial jet is {}x{}x{}",
                    n_rows,
                    radial_jet.shape()[1],
                    dim,
                    poly_jet.shape()[0],
                    n_poly,
                    poly_jet.shape()[2],
                ));
            }
            // Scalar kernel amplification `α` the forward
            // `build_latent_duchon_design` (→ `build_duchon_basis` with the
            // resolved `power`, `length_scale = None`) applies to the kernel
            // block `α·K(t,C)·Z`. The input-location derivative is
            // `α·K'(t,C)·Z`, so the raw radial jet must carry the same `α`
            // computed against the same resolved spectral power; the appended
            // polynomial columns are un-amplified, matching the forward.
            let kernel_amp = duchon_pure_kernel_amplification(
                centers,
                resolved_nullspace,
                resolved_power as f64,
            );
            let mut jet = Array3::<f64>::zeros((n_rows, n_kernel + n_poly, dim));
            for axis in 0..dim {
                let projected = radial_jet.index_axis(Axis(2), axis).dot(&radial_transform);
                let mut block = jet.slice_mut(s![.., ..n_kernel, axis]);
                block.assign(&projected);
                block *= kernel_amp;
            }
            jet.slice_mut(s![.., n_kernel.., ..]).assign(&poly_jet);
            Ok(jet)
        }
        "matern" => {
            // Fixes audit-revised claim that non-Duchon latent input-location
            // derivatives must use the closed-form helper instead of stubbing.
            let phi_r =
                matern_radial_first_derivative_nd(t_mat, centers, 1.0, MaternNu::ThreeHalves)
                    .map_err(|err| err.to_string())?;
            radial_input_location_jet(t_mat, centers, phi_r.view())
        }
        "sphere" => {
            // Fixes audit-revised claim that sphere latent derivatives are
            // analytic jets, not unsupported hooks.
            let jet = sphere_first_derivative_nd(t_mat, centers, m, true)
                .map_err(|err| err.to_string())?;
            Ok(jet)
        }
        "bspline_tensor" => {
            let knots = tensor_knots_concat.ok_or_else(|| {
                "tensor B-spline latent derivative requires knots_concat".to_string()
            })?;
            let offsets = tensor_knot_offsets.ok_or_else(|| {
                "tensor B-spline latent derivative requires knot_offsets".to_string()
            })?;
            let degrees = tensor_degrees
                .ok_or_else(|| "tensor B-spline latent derivative requires degrees".to_string())?;
            let per_axis = split_tensor_knots_owned(knots, offsets, t_mat.ncols())?;
            let per_axis_views = per_axis
                .iter()
                .map(|axis_knots| axis_knots.view())
                .collect::<Vec<_>>();
            let jet = bspline_tensor_first_derivative(t_mat, &per_axis_views, degrees)
                .map_err(|err| err.to_string())?;
            Ok(jet)
        }
        "periodic_bspline" => {
            // Fixes audit-revised claim that periodic latent derivatives are
            // analytic jets. The latent pyffi path carries only centers today,
            // so infer the period from the first center column.
            if centers.ncols() != 1 || centers.nrows() == 0 {
                return Err(
                    "periodic B-spline latent derivative requires one-column centers".to_string(),
                );
            }
            let mut lo = f64::INFINITY;
            let mut hi = f64::NEG_INFINITY;
            for &value in centers.column(0).iter() {
                lo = lo.min(value);
                hi = hi.max(value);
            }
            if !(lo.is_finite() && hi.is_finite() && hi > lo) {
                return Err("periodic B-spline centers must define a finite range".to_string());
            }
            let jet = periodic_bspline_first_derivative_nd(t_mat, (lo, hi), m, centers.nrows())
                .map_err(|err| err.to_string())?;
            Ok(jet)
        }
        other => Err(format!(
            "latent_basis_kind returned an unknown normalized kind: {other}"
        )),
    }
}

fn gaussian_reml_weight_vector_local(
    n_obs: usize,
    weights: Option<ArrayView1<'_, f64>>,
) -> Result<Array1<f64>, String> {
    match weights {
        Some(w) => {
            if w.len() != n_obs {
                return Err(format!(
                    "Gaussian REML weights length mismatch: expected {n_obs}, got {}",
                    w.len()
                ));
            }
            if w.iter().any(|value| !value.is_finite() || *value < 0.0) {
                return Err("Gaussian REML weights must be finite and non-negative".to_string());
            }
            Ok(w.to_owned())
        }
        None => Ok(Array1::ones(n_obs)),
    }
}

fn latent_scalar_weights_with_fisher(
    n_obs: usize,
    weights: Option<ArrayView1<'_, f64>>,
    fisher_w: Option<ArrayView3<'_, f64>>,
) -> Result<Option<Array1<f64>>, String> {
    let Some(fw) = fisher_w else {
        return Ok(weights.map(|w| w.to_owned()));
    };
    if fw.shape() != [n_obs, 1, 1] {
        return Err(format!(
            "fisher_w currently accepts scalar blocks of shape ({n_obs}, 1, 1) on this latent entry point; got {:?}",
            fw.shape()
        ));
    }
    let mut out = match weights {
        Some(w) => gaussian_reml_weight_vector_local(n_obs, Some(w))?,
        None => Array1::ones(n_obs),
    };
    for n in 0..n_obs {
        let v = fw[[n, 0, 0]];
        if !(v.is_finite() && v >= 0.0) {
            return Err(format!(
                "fisher_w[{n},0,0] must be finite and non-negative; got {v}"
            ));
        }
        out[n] *= v;
    }
    Ok(Some(out))
}

fn latent_row_weights(
    n_obs: usize,
    weights: Option<ArrayView1<'_, f64>>,
) -> Result<Array1<f64>, String> {
    match weights {
        Some(w) => gaussian_reml_weight_vector_local(n_obs, Some(w)),
        None => Ok(Array1::ones(n_obs)),
    }
}

fn validate_dense_fisher_w(
    n_obs: usize,
    n_outputs: usize,
    fisher_w: ArrayView3<'_, f64>,
) -> Result<(), String> {
    if fisher_w.shape() != [n_obs, n_outputs, n_outputs] {
        return Err(format!(
            "fisher_w dense blocks must have shape ({n_obs}, {n_outputs}, {n_outputs}); got {:?}",
            fisher_w.shape()
        ));
    }
    for n in 0..n_obs {
        for a in 0..n_outputs {
            for b in 0..n_outputs {
                let v = fisher_w[[n, a, b]];
                if !v.is_finite() {
                    return Err(format!("fisher_w[{n},{a},{b}] must be finite; got {v}"));
                }
            }
            if fisher_w[[n, a, a]] < 0.0 {
                return Err(format!(
                    "fisher_w[{n},{a},{a}] must be non-negative; got {}",
                    fisher_w[[n, a, a]]
                ));
            }
        }
    }
    Ok(())
}

#[derive(Clone, Copy, Debug)]
struct LatentAuxStrengthState {
    log_mu: f64,
    mu: f64,
    auto: bool,
}

struct LatentAuxPriorStats {
    targets: Array2<f64>,
    residual_sq: f64,
    strength: LatentAuxStrengthState,
    score: f64,
}

fn latent_aux_prior_stats(
    t_mat: ArrayView2<'_, f64>,
    u_view: ArrayView2<'_, f64>,
    aux_family: AuxPriorFamily,
    aux_strength: Option<f64>,
) -> Result<LatentAuxPriorStats, String> {
    let targets = aux_prior_targets(t_mat, u_view, aux_family)?;
    // The closed-form auxiliary-prior REML statistics (residual norm + the
    // log_mu optimum at fixed t + the prior score) live in core; this packs
    // them into the FFI latent-fit plumbing struct.
    let stats = gam::terms::latent::aux_prior_reml_stats(t_mat, targets.view(), aux_strength)?;
    Ok(LatentAuxPriorStats {
        targets,
        residual_sq: stats.residual_sq,
        strength: LatentAuxStrengthState {
            log_mu: stats.log_mu,
            mu: stats.mu,
            auto: stats.auto,
        },
        score: stats.score,
    })
}

/// Honestly surface the SAE manifold loss score under `primary_key` (#1231).
///
/// The score is `−(data_fit + assignment_sparsity + smoothness + ard)` — the
/// NEGATIVE penalized loss of the four loss components — NOT a REML / marginal
/// likelihood: it omits the Hessian log-determinant, the Occam log-λ term, any
/// extra analytic penalties, the co-training fold, the top-k projection effect,
/// and hybrid-collapse effects. `primary_key` must therefore be an honest name
/// (`"penalized_loss_score"` on the in-sample fit, `"oos_penalized_loss"` on the
/// fixed-decoder OOS path). The full component breakdown is written under
/// `"penalized_loss_breakdown"` so a consumer can see exactly what the score is
/// (and is not).
fn sae_set_penalized_loss_items(
    out: &Bound<'_, PyDict>,
    loss: &gam::terms::sae::manifold::SaeManifoldLoss,
    primary_key: &str,
) -> PyResult<()> {
    let b = loss.breakdown();
    out.set_item(primary_key, b.penalized_loss_score)?;
    let breakdown = PyDict::new(out.py());
    breakdown.set_item("penalized_loss_score", b.penalized_loss_score)?;
    breakdown.set_item("total_penalized_loss", b.total_penalized_loss)?;
    breakdown.set_item("data_fit", b.data_fit)?;
    breakdown.set_item("assignment_sparsity", b.assignment_sparsity)?;
    breakdown.set_item("smoothness", b.smoothness)?;
    breakdown.set_item("ard", b.ard)?;
    breakdown.set_item(
        "evidence_gauge_deflated_directions",
        b.evidence_gauge_deflated_directions,
    )?;
    // Honesty markers: this score is NOT a REML criterion; these evidence pieces
    // are deliberately absent (#1231). Surfaced as `None` so a consumer that
    // wants real evidence sees that it was not computed here, rather than reading
    // the penalized loss as if it were the marginal likelihood.
    for missing in [
        "logdet_hessian",
        "occam_log_lambda",
        "extra_penalty_energy",
        "amortization_consistency",
        "collapse_penalty",
    ] {
        breakdown.set_item(missing, out.py().None())?;
    }
    breakdown.set_item("is_reml", false)?;
    out.set_item("penalized_loss_breakdown", breakdown)?;
    Ok(())
}

fn set_aux_strength_items<'py>(
    py: Python<'py>,
    out: &Bound<'py, PyDict>,
    state: Option<LatentAuxStrengthState>,
) -> PyResult<()> {
    if let Some(state) = state {
        out.set_item("aux_strength", state.mu)?;
        out.set_item("aux_log_strength", state.log_mu)?;
        out.set_item("log_mu", state.log_mu)?;
        out.set_item(
            "aux_strength_mode",
            if state.auto { "auto" } else { "fixed" },
        )?;
    } else {
        out.set_item("aux_strength", py.None())?;
        out.set_item("aux_log_strength", py.None())?;
        out.set_item("log_mu", py.None())?;
        out.set_item("aux_strength_mode", py.None())?;
    }
    Ok(())
}

fn latent_prior_score_and_aux_state_for_t(
    t_mat: ArrayView2<'_, f64>,
    aux_u: Option<ArrayView2<'_, f64>>,
    aux_family: AuxPriorFamily,
    aux_strength: Option<f64>,
    dim_selection_precision: Option<ArrayView1<'_, f64>>,
) -> Result<(f64, Option<LatentAuxStrengthState>), String> {
    let n_obs = t_mat.nrows();
    let latent_dim = t_mat.ncols();
    let mut latent_prior_score = 0.0_f64;
    let mut aux_strength_state = None;
    if let Some(u_view) = aux_u {
        let stats = latent_aux_prior_stats(t_mat, u_view, aux_family, aux_strength)?;
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
    Ok((latent_prior_score, aux_strength_state))
}

fn dense_fisher_gaussian_fit_to_pydict<'py>(
    py: Python<'py>,
    design: ArrayView2<'_, f64>,
    y: ArrayView2<'_, f64>,
    penalty: ArrayView2<'_, f64>,
    weights: Option<ArrayView1<'_, f64>>,
    fisher_w: ArrayView3<'_, f64>,
    init_lambda: Option<f64>,
    latent_prior_score: f64,
    aux_strength_state: Option<LatentAuxStrengthState>,
) -> PyResult<Py<PyDict>> {
    let n_obs = design.nrows();
    let k = design.ncols();
    let n_outputs = y.ncols();
    validate_dense_fisher_w(n_obs, n_outputs, fisher_w).map_err(py_value_error)?;
    if y.nrows() != n_obs {
        return Err(py_value_error(format!(
            "dense Fisher Gaussian row mismatch: X has {n_obs}, y has {}",
            y.nrows()
        )));
    }
    let row_weights = latent_row_weights(n_obs, weights).map_err(py_value_error)?;
    let lambda = init_lambda.unwrap_or(1.0);
    if !(lambda.is_finite() && lambda > 0.0) {
        return Err(py_value_error(format!(
            "init_lambda must be finite and positive for dense Fisher fit; got {lambda}"
        )));
    }
    // Closed-form fixed-λ multi-output Gaussian fit under the dense Fisher-Rao
    // metric lives in core; this shim validates inputs and packs the PyDict.
    let fit = gam::solver::gaussian_reml::dense_fisher_gaussian_fit(
        design,
        y,
        penalty,
        row_weights.view(),
        fisher_w,
        lambda,
        latent_prior_score,
    )
    .map_err(|err| py_value_error(err.to_string()))?;
    let coefficients = fit.coefficients;
    let fitted = fit.fitted;
    let sigma2 = fit.sigma2;
    let objective = fit.objective;
    let out = PyDict::new(py);
    out.set_item("status", "ok")?;
    out.set_item("lambda", lambda)?;
    out.set_item("rho", lambda.ln())?;
    out.set_item("reml_score", objective)?;
    out.set_item("reml_grad_lambda", f64::NAN)?;
    out.set_item("reml_hess_lambda", f64::NAN)?;
    out.set_item("reml_grad_rho", f64::NAN)?;
    out.set_item("reml_hess_rho", f64::NAN)?;
    out.set_item("edf", (k * n_outputs) as f64)?;
    out.set_item("coefficients", coefficients.into_pyarray(py))?;
    out.set_item("fitted", fitted.into_pyarray(py))?;
    out.set_item("sigma2", sigma2.into_pyarray(py))?;
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
    set_aux_strength_items(py, &out, aux_strength_state)?;
    Ok(out.unbind())
}

/// Numerically stable logistic CDF used by `numerics_sigmoid_stable` (a
/// scalar Python helper exposed to the gamfit numerics module). The
/// multi-output penalised fitter no longer needs this helper directly —
/// it routes through `gam::families::binomial_multi` whose internal
/// `sigmoid_stable` lives in the Rust core.
fn sigmoid_stable(eta: f64) -> f64 {
    if eta >= 0.0 {
        let e = (-eta).exp();
        1.0 / (1.0 + e)
    } else {
        let e = eta.exp();
        e / (1.0 + e)
    }
}

/// FFI shim wrapping the canonical multi-output penalized fitters at fixed λ.
///
/// Dispatches to either
/// [`gam::families::multinomial::fit_penalized_multinomial`] (softmax
/// reference-coded multinomial-logit, `K − 1` active classes coupled by the
/// dense per-row Fisher block `μ_a δ_{ab} − μ_a μ_b`) or
/// [`gam::families::binomial_multi::fit_penalized_binomial_multi`] (`K`
/// independent binomial-logit columns with row-diagonal Fisher
/// `μ_a (1 − μ_a)`). REML / LAML λ selection is not performed on this
/// latent-coordinate path; `init_lambda` is taken as the fixed smoothing
/// parameter and the `reml_grad_*` / `reml_hess_*` diagnostic fields stay
/// NaN, matching the prior `dense_fisher_glm_fit_to_pydict` dict-shape
/// contract that downstream Python callers consume.
fn latent_multi_output_fit_to_pydict<'py>(
    py: Python<'py>,
    design: ArrayView2<'_, f64>,
    y: ArrayView2<'_, f64>,
    penalty: ArrayView2<'_, f64>,
    weights: Option<ArrayView1<'_, f64>>,
    fisher_w: Option<ArrayView3<'_, f64>>,
    init_lambda: Option<f64>,
    family_name: &str,
    latent_prior_score: f64,
    aux_strength_state: Option<LatentAuxStrengthState>,
) -> PyResult<Py<PyDict>> {
    let n_obs = design.nrows();
    let p = design.ncols();
    let n_outputs = y.ncols();
    if y.nrows() != n_obs || n_outputs == 0 {
        return Err(py_value_error(format!(
            "multi-output GLM latent response shape mismatch: X has {n_obs} rows, y is {}x{}",
            y.nrows(),
            y.ncols()
        )));
    }
    // Both supported families (binomial-logit and softmax multinomial-logit)
    // consume probability-valued targets: the per-entry binomial term
    // y log μ + (1 − y) log(1 − μ) and the softmax cross-entropy − Σ y_a log μ_a
    // are bounded log-likelihoods only when every observed entry lies in [0, 1].
    // A finite-but-out-of-range entry (e.g. y = 2) makes the objective unbounded
    // in η, so reject it before calling the solver.
    for ((n, a), &v) in y.indexed_iter() {
        if !(v.is_finite() && (0.0..=1.0).contains(&v)) {
            return Err(py_value_error(format!(
                "multi-output GLM latent response must be a probability in [0,1]; \
                 got y[{n},{a}] = {v}"
            )));
        }
    }
    let row_weights = latent_row_weights(n_obs, weights).map_err(py_value_error)?;
    let lambda = init_lambda.unwrap_or(1.0);
    if !(lambda.is_finite() && lambda > 0.0) {
        return Err(py_value_error(format!(
            "init_lambda must be finite and positive for multi-output GLM latent fit; got {lambda}"
        )));
    }
    let normalized = family_name.to_ascii_lowercase().replace('_', "-");
    let multinomial = matches!(
        normalized.as_str(),
        "multinomial" | "multinomial-logit" | "softmax" | "categorical-logit"
    );
    let binomial_multi = matches!(
        normalized.as_str(),
        "binomial" | "binomial-logit" | "logistic"
    );
    if multinomial && n_outputs < 2 {
        return Err(py_value_error(
            "multinomial-logit requires at least two response columns".to_string(),
        ));
    }
    if multinomial {
        // The softmax cross-entropy −Σ_c y_c log p_c has residual gradient
        // y_a − p_a and Fisher block p_a δ_ab − p_a p_b only when each row is a
        // point on the probability simplex (Σ_c y_c = 1; nonnegativity is
        // already enforced by the per-entry [0,1] loop above). A row with mass
        // s ≠ 1 has true gradient y_a − s p_a, so accepting it would fit with a
        // curvature that disagrees with the objective. Reject before solving.
        // (Binomial-multi treats the K columns as independent Bernoulli draws,
        // for which the per-entry [0,1] constraint alone is correct.)
        for n in 0..n_obs {
            let mut row_sum = 0.0_f64;
            for a in 0..n_outputs {
                row_sum += y[[n, a]];
            }
            if (row_sum - 1.0).abs() > 1.0e-9 {
                return Err(py_value_error(format!(
                    "multinomial-logit response rows must sum to 1 (one-hot for hard \
                     labels, or a label-smoothed probability vector); row {n} sums to \
                     {row_sum}"
                )));
            }
        }
    }
    if !(multinomial || binomial_multi) {
        return Err(py_value_error(format!(
            "multi-output GLM latent supports binomial-logit and multinomial-logit; got {family_name:?}"
        )));
    }

    // Per-class smoothing pulled from `init_lambda`: the latent path is a
    // fixed-λ inner solve, so every active class shares the same λ. REML
    // selection per class is a separate slice (issue #349 follow-up).
    let active_outputs = if multinomial {
        n_outputs - 1
    } else {
        n_outputs
    };
    let lambdas_vec = Array1::<f64>::from_elem(active_outputs, lambda);

    let max_iter = 50usize;
    let tol = 1.0e-7_f64;

    // Per-row Fisher-block override (issue #349). The wire-level array is
    // `(N, K, K)`; validate finiteness + non-negative diagonal here, then adapt
    // to each branch's curvature gauge:
    //   - multinomial: the fitter consumes the active `(N, K-1, K-1)` leading
    //     sub-block (the reference class K-1 is dropped); require each per-row
    //     active block to be symmetric.
    //   - binomial-multi: the K columns are fit independently, so off-diagonal
    //     cross terms cannot be represented — require them to be zero — and the
    //     full `(N, K, K)` array is forwarded for its diagonal.
    let multinomial_fisher_active: Option<Array3<f64>> = match fisher_w.as_ref() {
        Some(fw) => {
            validate_dense_fisher_w(n_obs, n_outputs, *fw).map_err(py_value_error)?;
            if multinomial {
                let mut active = Array3::<f64>::zeros((n_obs, active_outputs, active_outputs));
                for n in 0..n_obs {
                    for a in 0..active_outputs {
                        for b in 0..active_outputs {
                            active[[n, a, b]] = fw[[n, a, b]];
                        }
                    }
                    for a in 0..active_outputs {
                        for b in (a + 1)..active_outputs {
                            if (fw[[n, a, b]] - fw[[n, b, a]]).abs() > 1.0e-9 {
                                return Err(py_value_error(format!(
                                    "fisher_w active block[{n}] must be symmetric for the \
                                     multinomial path; entries [{a},{b}] and [{b},{a}] differ"
                                )));
                            }
                        }
                    }
                }
                Some(active)
            } else {
                for n in 0..n_obs {
                    for a in 0..n_outputs {
                        for b in 0..n_outputs {
                            if a != b && fw[[n, a, b]] != 0.0 {
                                return Err(py_value_error(format!(
                                    "fisher_w block[{n}] off-diagonal [{a},{b}] must be zero for \
                                     the binomial-multi path (each response column is fit \
                                     independently); got {}",
                                    fw[[n, a, b]]
                                )));
                            }
                        }
                    }
                }
                None
            }
        }
        None => None,
    };

    // Coefficients matrix `(P, K)` with reference-class column zeroed for
    // the multinomial branch (so the wire-level shape matches the prior
    // `dense_fisher_glm_fit_to_pydict` contract).
    let mut coefficients = Array2::<f64>::zeros((p, n_outputs));
    let fitted: Array2<f64>;
    let iterations: usize;
    let penalized_neg_log_likelihood: f64;

    if multinomial {
        let outputs = gam::families::multinomial::fit_penalized_multinomial(
            gam::families::multinomial::MultinomialFitInputs {
                design,
                y_one_hot: y,
                penalty,
                lambdas: lambdas_vec.view(),
                row_weights: Some(row_weights.view()),
                fisher_w_override: multinomial_fisher_active.as_ref().map(|a| a.view()),
                max_iter,
                tol,
                resume_from: None,
            },
        )
        .map_err(|err| py_value_error(err.to_string()))?;
        // Pack the (P, K-1) active coefficients into the (P, K) wire shape;
        // reference class column K-1 stays zero by construction.
        for a in 0..active_outputs {
            for col in 0..p {
                coefficients[[col, a]] = outputs.coefficients_active[[col, a]];
            }
        }
        fitted = outputs.fitted_probabilities;
        iterations = outputs.iterations;
        penalized_neg_log_likelihood = outputs.penalized_neg_log_likelihood;
    } else {
        let outputs = gam::families::binomial_multi::fit_penalized_binomial_multi(
            gam::families::binomial_multi::BinomialMultiFitInputs {
                design,
                y,
                penalty,
                lambdas: lambdas_vec.view(),
                row_weights: Some(row_weights.view()),
                fisher_w_override: fisher_w,
                max_iter,
                tol,
            },
        )
        .map_err(|err| py_value_error(err.to_string()))?;
        coefficients = outputs.coefficients;
        fitted = outputs.fitted_probabilities;
        iterations = outputs.iterations;
        penalized_neg_log_likelihood = outputs.penalized_neg_log_likelihood;
    }

    // `reml_score` on this fixed-λ path is the penalized negative
    // log-likelihood plus the latent prior / analytic penalty contribution
    // pre-summed in `latent_prior_score`. The full REML log-marginal would
    // additionally include the log|H| / log|S| Laplace terms; those are
    // not computed here (consistent with the prior contract, which left
    // every `reml_grad_*` / `reml_hess_*` field NaN).
    let reml_score = penalized_neg_log_likelihood + latent_prior_score;

    let out = PyDict::new(py);
    // Both core adapters are convergence-only result boundaries: a stalled
    // vector solve is a typed error above and can never reach this dictionary.
    out.set_item("status", "ok")?;
    out.set_item("lambda", lambda)?;
    out.set_item("rho", lambda.ln())?;
    out.set_item("reml_score", reml_score)?;
    out.set_item("reml_grad_lambda", f64::NAN)?;
    out.set_item("reml_hess_lambda", f64::NAN)?;
    out.set_item("reml_grad_rho", f64::NAN)?;
    out.set_item("reml_hess_rho", f64::NAN)?;
    out.set_item("edf", (p * active_outputs) as f64)?;
    out.set_item("coefficients", coefficients.into_pyarray(py))?;
    out.set_item("fitted", fitted.into_pyarray(py))?;
    out.set_item("sigma2", Array1::<f64>::ones(n_outputs).into_pyarray(py))?;
    out.set_item("iterations", iterations)?;
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
    set_aux_strength_items(py, &out, aux_strength_state)?;
    Ok(out.unbind())
}

/// Penalized multinomial-logit GAM fit at fixed λ.
///
/// Reference-coded softmax with class `K - 1` as the implicit reference
/// (`η_{K-1} ≡ 0`). The penalty matrix `S` is replicated per active class
/// with per-class smoothing `lambdas[a]`; the joint penalized Hessian is
/// `Xᵀ W X + diag_a(λ_a) ⊗ S` in output-major coefficient ordering
/// `β = [β_0; β_1; …; β_{K-2}]`.
///
/// Routes through `gam::families::multinomial::fit_penalized_multinomial`,
/// which uses `MultinomialLogitLikelihood: VectorLikelihood` (canonical
/// observed = Fisher per-row dense `(K-1)×(K-1)` block
/// `p_a δ_{ab} − p_a p_b`) and `gam::solver::pirls::dense_block_xtwx` for the
/// stacked Newton solve.
///
/// REML / LAML λ selection is a separate slice (the multinomial CustomFamily
/// outer that lifts this driver into the ρ loop). Until that lands, callers
/// pass an explicit `lambdas` vector (length `K - 1`).
#[pyfunction(signature = (
    design,
    y_one_hot,
    penalty,
    lambdas,
    row_weights = None,
    max_iter = 50,
    tol = 1.0e-7,
))]
fn fit_penalized_multinomial_pyfunc<'py>(
    py: Python<'py>,
    design: PyReadonlyArray2<'py, f64>,
    y_one_hot: PyReadonlyArray2<'py, f64>,
    penalty: PyReadonlyArray2<'py, f64>,
    lambdas: PyReadonlyArray1<'py, f64>,
    row_weights: Option<PyReadonlyArray1<'py, f64>>,
    max_iter: usize,
    tol: f64,
) -> PyResult<Py<PyDict>> {
    let design_view = design.as_array();
    let y_view = y_one_hot.as_array();
    let penalty_view = penalty.as_array();
    let lambdas_view = lambdas.as_array();
    let row_weights_view = row_weights.as_ref().map(|w| w.as_array());

    let outputs = gam::families::multinomial::fit_penalized_multinomial(
        gam::families::multinomial::MultinomialFitInputs {
            design: design_view,
            y_one_hot: y_view,
            penalty: penalty_view,
            lambdas: lambdas_view,
            row_weights: row_weights_view,
            fisher_w_override: None,
            max_iter,
            tol,
            resume_from: None,
        },
    )
    .map_err(|err| py_value_error(err.to_string()))?;

    let out = PyDict::new(py);
    // Non-convergence is a typed error from the core entry point (SPEC: a fit
    // only ever comes from a converged optimization), so reaching here means
    // the solve converged.
    out.set_item("status", "ok")?;
    out.set_item("iterations", outputs.iterations)?;
    out.set_item(
        "coefficients_active",
        outputs.coefficients_active.into_pyarray(py),
    )?;
    out.set_item(
        "fitted_probabilities",
        outputs.fitted_probabilities.into_pyarray(py),
    )?;
    out.set_item(
        "penalized_neg_log_likelihood",
        outputs.penalized_neg_log_likelihood,
    )?;
    out.set_item("deviance", outputs.deviance)?;
    Ok(out.unbind())
}

// ---------------------------------------------------------------------------
// Formula-driven multinomial pipeline (Slice A of #328)
// ---------------------------------------------------------------------------
//
// The high-level `gamfit.fit(data, formula, family='multinomial')` Python
// entry routes through `fit_multinomial_formula_pyfunc` below. The Rust core
// in `gam::families::multinomial::fit_penalized_multinomial_formula` parses
// the formula, materialises the term-collection design + penalty blocks the
// same way the standard workflow does, one-hot-encodes the categorical
// response, and runs `fit_penalized_multinomial` at a uniform initial
// smoothing parameter. REML / LAML λ selection is the next slice.
//
// Round-trip format: `serde_json::to_vec(&MultinomialModelEnvelope { ... })`
// keyed by `model_class = "multinomial"` so the Python `_model.load` path
// can dispatch by inspecting the JSON discriminator without deserialising
// the whole `FittedModel` struct.

// The multinomial persistence envelope (the `model_class`-discriminated wrapper
// that lets the Python load path tell a `MultinomialSavedModel` apart from the
// scalar `FittedModel` payload) is defined once in the core so the CLI and the
// FFI share one on-disk contract.
use gam::families::multinomial::MultinomialModelEnvelope;

/// Fit a penalized multinomial-logit GAM from a Wilkinson formula against
/// a `headers + rows` table. Returns the bincode-free, serde-JSON model
/// payload that `gamfit.MultinomialModel` deserialises and stores under
/// `Model._model_bytes`.
///
/// `config_json` is the same canonical fit-config document every formula
/// family consumes (`gam::config_resolve`). The typed core request honors
/// `weights` as per-row case weights and rejects fields the softmax family
/// cannot consume (offsets, noise formulas, manual Firth, ...) instead of
/// silently dropping them.
#[pyfunction(signature = (
    headers,
    rows,
    formula,
    config_json = None,
    init_lambda = 1.0,
    max_iter = 50,
    tol = 1.0e-7,
))]
fn fit_multinomial_formula_pyfunc<'py>(
    py: Python<'py>,
    headers: Vec<String>,
    rows: PyRef<'py, PyEncodedTable>,
    formula: String,
    config_json: Option<String>,
    init_lambda: f64,
    max_iter: usize,
    tol: f64,
) -> PyResult<Py<PyBytes>> {
    rows.require_headers(&headers).map_err(py_value_error)?;
    let dataset = rows.dataset.clone();
    let bytes = detach_pyresult(py, "fit_multinomial_formula", move || {
        let fit_config = gam::config_resolve::parse_fit_config_json(config_json.as_deref())
            .map_err(py_value_error)?;
        // Typed engine path: `EstimationError` → matching `gamfit.*Error`
        // subclass via `estimation_error_to_pyerr` (issue #343).
        let saved = gam::families::multinomial::fit_penalized_multinomial_formula(
            &gam::families::multinomial::MultinomialFitRequest {
                init_lambda,
                max_iter,
                tol,
                ..gam::families::multinomial::MultinomialFitRequest::new(
                    &dataset,
                    &formula,
                    &fit_config,
                )
            },
        )
        .map_err(estimation_error_to_pyerr)?;
        MultinomialModelEnvelope::new(saved)
            .map_err(estimation_error_to_pyerr)?
            .to_json_bytes()
            .map_err(estimation_error_to_pyerr)
    })?;
    Ok(PyBytes::new(py, &bytes).unbind())
}

/// Predict class probabilities for a saved multinomial model. The returned
/// `(N_new, K)` numpy array column-aligns with `MultinomialSavedModel.class_levels`.
#[pyfunction(signature = (model_bytes, headers, rows))]
fn predict_multinomial_formula_pyfunc<'py>(
    py: Python<'py>,
    model_bytes: Vec<u8>,
    headers: Vec<String>,
    rows: PyRef<'py, PyEncodedTable>,
) -> PyResult<Py<PyArray2<f64>>> {
    rows.require_headers(&headers).map_err(py_value_error)?;
    let dataset = rows.dataset.clone();
    let probs = detach_pyresult(py, "predict_multinomial_formula", move || {
        let envelope =
            MultinomialModelEnvelope::from_json_bytes(&model_bytes).map_err(estimation_error_to_pyerr)?;
        // Typed engine path: `EstimationError` → matching `gamfit.*Error`
        // subclass via `estimation_error_to_pyerr` (issue #343).
        gam::families::multinomial::predict_multinomial_formula(&envelope.saved, &dataset)
            .map_err(estimation_error_to_pyerr)
    })?;
    Ok(probs.into_pyarray(py).unbind())
}

/// Predict posterior-mean class probabilities WITH integrated per-class
/// standard errors and simplex-clamped confidence bounds for a saved
/// multinomial model (#1101). Returns a dict with `probs`, `prob_se`,
/// `mean_lower`, `mean_upper` (all `(N_new, K)` arrays, columns aligned with
/// `class_levels`) plus the `level` used. Center and spread both come from the
/// same deterministic logistic-normal posterior integral (SPEC 3: the estimand
/// is `E[softmax(η) | data]`, never the plug-in `softmax(E[η])`); a model
/// without stored posterior covariance is a typed error.
#[pyfunction(signature = (model_bytes, headers, rows, level = 0.95))]
fn predict_multinomial_intervals_pyfunc<'py>(
    py: Python<'py>,
    model_bytes: Vec<u8>,
    headers: Vec<String>,
    rows: PyRef<'py, PyEncodedTable>,
    level: f64,
) -> PyResult<Py<PyDict>> {
    rows.require_headers(&headers).map_err(py_value_error)?;
    let dataset = rows.dataset.clone();
    let intervals = detach_pyresult(py, "predict_multinomial_intervals", move || {
        let envelope =
            MultinomialModelEnvelope::from_json_bytes(&model_bytes).map_err(estimation_error_to_pyerr)?;
        gam::families::multinomial::predict_multinomial_formula_with_intervals(
            &envelope.saved,
            &dataset,
            level,
        )
        .map_err(estimation_error_to_pyerr)
    })?;
    let out = PyDict::new(py);
    out.set_item("level", intervals.level)?;
    out.set_item("probs", intervals.mean.into_pyarray(py))?;
    out.set_item("prob_se", intervals.standard_error.into_pyarray(py))?;
    out.set_item("mean_lower", intervals.mean_lower.into_pyarray(py))?;
    out.set_item("mean_upper", intervals.mean_upper.into_pyarray(py))?;
    Ok(out.unbind())
}

/// Draw posterior-predictive replicate class assignments for a saved
/// multinomial model on fresh rows (#1101). Each of `n_draws` replicates samples
/// every row's class from `Categorical(softmax(X·β̂))` — the plug-in predictive
/// (categorical observation noise around the fitted mean). Returns a dict with
/// `draws` (an `(n_draws, N_new)` int64 array of class INDICES `0..K`) and
/// `class_levels` (the label list those indices index). Deterministic in `seed`.
#[pyfunction(signature = (model_bytes, headers, rows, n_draws, seed = 0))]
fn posterior_predict_multinomial_pyfunc<'py>(
    py: Python<'py>,
    model_bytes: Vec<u8>,
    headers: Vec<String>,
    rows: PyRef<'py, PyEncodedTable>,
    n_draws: usize,
    seed: u64,
) -> PyResult<Py<PyDict>> {
    rows.require_headers(&headers).map_err(py_value_error)?;
    let dataset = rows.dataset.clone();
    let (draws, class_levels) = detach_pyresult(py, "posterior_predict_multinomial", move || {
        let envelope =
            MultinomialModelEnvelope::from_json_bytes(&model_bytes).map_err(estimation_error_to_pyerr)?;
        let draws = gam::families::multinomial::posterior_predict_multinomial_formula(
            &envelope.saved,
            &dataset,
            n_draws,
            seed,
        )
        .map_err(estimation_error_to_pyerr)?;
        // Map u32 class indices to i64 for a natural numpy integer dtype.
        let draws_i64 = draws.mapv(|v| v as i64);
        Ok((draws_i64, envelope.saved.class_levels.clone()))
    })?;
    let out = PyDict::new(py);
    out.set_item("draws", draws.into_pyarray(py))?;
    out.set_item("class_levels", class_levels)?;
    Ok(out.unbind())
}

/// Wood rank-truncated Wald smooth-significance table for a saved multinomial
/// model (#1101). Returns a list of dicts, one per `(active class, smooth term)`:
/// `class`, `term`, `edf`, `ref_df`, `statistic`, `p_value`. Empty when the
/// model has no smooth terms or no stored covariance.
#[pyfunction(signature = (model_bytes))]
fn multinomial_smooth_significance_pyfunc<'py>(
    py: Python<'py>,
    model_bytes: Vec<u8>,
) -> PyResult<Py<pyo3::types::PyList>> {
    let envelope = MultinomialModelEnvelope::from_json_bytes(&model_bytes)
        .map_err(estimation_error_to_pyerr)?;
    let rows = envelope.saved.smooth_significance();
    let list = pyo3::types::PyList::empty(py);
    for r in rows {
        let row = PyDict::new(py);
        row.set_item("class", r.class_label)?;
        row.set_item("term", r.term_label)?;
        row.set_item("edf", r.edf)?;
        row.set_item("ref_df", r.ref_df)?;
        row.set_item("statistic", r.statistic)?;
        row.set_item("p_value", r.p_value)?;
        list.append(row)?;
    }
    Ok(list.unbind())
}

/// Inspect a multinomial saved-model byte blob and return the class-level
/// metadata needed by `MultinomialModel.summary()` and `.classes_`. Keeping
/// this on the FFI side avoids re-encoding the serde envelope in Python.
#[pyfunction(signature = (model_bytes))]
fn multinomial_model_metadata_pyfunc<'py>(
    py: Python<'py>,
    model_bytes: Vec<u8>,
) -> PyResult<Py<PyDict>> {
    let envelope = MultinomialModelEnvelope::from_json_bytes(&model_bytes)
        .map_err(estimation_error_to_pyerr)?;
    let out = PyDict::new(py);
    out.set_item("formula", &envelope.saved.formula)?;
    out.set_item("class_levels", envelope.saved.class_levels.clone())?;
    out.set_item(
        "reference_class_index",
        envelope.saved.reference_class_index,
    )?;
    out.set_item("p_per_class", envelope.saved.p_per_class)?;
    out.set_item("n_active_classes", envelope.saved.n_active_classes)?;
    out.set_item("training_headers", envelope.saved.training_headers.clone())?;
    out.set_item("lambdas", envelope.saved.lambdas.clone())?;
    out.set_item(
        "lambdas_per_block",
        envelope.saved.lambdas_per_block.clone(),
    )?;
    let smooth_term_labels: Vec<String> = envelope
        .saved
        .smooth_term_spans
        .iter()
        .map(|span| span.label.clone())
        .collect();
    out.set_item("smooth_term_labels", smooth_term_labels)?;
    // Per-penalty-component λ labels, parallel to a single class block's λ slice
    // (#1544). With the Marra–Wood double penalty a smooth term emits two
    // components (primary wiggliness + null-space shrinkage), so this is NOT 1:1
    // with `smooth_term_labels`; the summary renderer pairs each λ with the
    // label here component-for-component instead of assuming one λ per term.
    out.set_item("lambda_labels", envelope.saved.lambda_labels.clone())?;
    out.set_item("iterations", envelope.saved.iterations)?;
    out.set_item(
        "penalized_neg_log_likelihood",
        envelope.saved.penalized_neg_log_likelihood,
    )?;
    out.set_item("deviance", envelope.saved.deviance)?;
    out.set_item(
        "coefficients_flat",
        envelope.saved.coefficients_flat.clone(),
    )?;
    if let Some(edf) = envelope.saved.edf_per_class.as_ref() {
        out.set_item("edf_per_class", edf.clone())?;
    }
    if let Some(edf_pen) = envelope.saved.edf_per_penalty.as_ref() {
        out.set_item("edf_per_penalty", edf_pen.clone())?;
    }
    Ok(out.unbind())
}

fn latent_augmented_hessian_factor(
    design: ArrayView2<'_, f64>,
    penalty: ArrayView2<'_, f64>,
    weights: ArrayView1<'_, f64>,
    lambda: f64,
) -> Result<gam::linalg::faer_ndarray::FaerSymmetricFactor, String> {
    if penalty.dim() != (design.ncols(), design.ncols()) {
        return Err(format!(
            "penalty shape mismatch for latent Hessian: expected {}x{}, got {}x{}",
            design.ncols(),
            design.ncols(),
            penalty.nrows(),
            penalty.ncols()
        ));
    }
    let mut hessian = fast_xt_diag_x(&design, &weights);
    for row in 0..hessian.nrows() {
        for col in 0..hessian.ncols() {
            let s_sym = 0.5 * (penalty[[row, col]] + penalty[[col, row]]);
            hessian[[row, col]] += lambda * s_sym;
        }
    }
    factorize_symmetricwith_fallback(
        gam::linalg::faer_ndarray::FaerArrayView::new(&hessian).as_ref(),
        Side::Lower,
    )
    .map_err(|err| format!("latent REML Hessian factorization failed: {err}"))
}

/// Add the analytic outer REML score contribution to `grad_t`.
///
/// Implements `/tmp/codex_outer_analytic.md`:
///
/// ```text
/// ∇_{t_i} V = w_i J_i^T [K_H x_i - (r_i / σ_eff²) β]
/// ```
///
/// This is the Occam-factor contribution called out in the
/// `composition_engine.md` §7 audit revisions. The previous
/// `gaussian_reml_fit_latent_backward` path contracted only the data-fit
/// design gradient for the latent row; without the `K_H x_i` term, the
/// returned `grad_t` omitted the REML log-determinant correction. `K_H` is
/// never materialized here: the row vector `u_i = K_H x_i` is computed by one
/// solve against the shared factor of `H = X^T W X + λS` per row and then
/// reused across all latent coordinates for that row.
///
/// Future `S(ρ, t)` penalties belong after the row-local contraction below:
/// add the standard `β^T S_{i,a} β`, `tr(K_H S_{i,a})`, and
/// `tr(S_+ S_{i,a})` terms from the derivation when coefficient penalties
/// become input-location dependent.
fn add_latent_outer_reml_score_gradient(
    grad_t: &mut Array1<f64>,
    scale: f64,
    design: ArrayView2<'_, f64>,
    y: ArrayView2<'_, f64>,
    t_mat: ArrayView2<'_, f64>,
    jet: &Array3<f64>,
    penalty: ArrayView2<'_, f64>,
    weights: Option<ArrayView1<'_, f64>>,
    fit: &gam::solver::gaussian_reml::GaussianRemlMultiResult,
    sigma_eff_mode: SigmaEffMode,
) -> Result<(), String> {
    if scale == 0.0 {
        return Ok(());
    }
    let n_obs = design.nrows();
    let p = design.ncols();
    let latent_dim = t_mat.ncols();
    let n_outputs = fit.coefficients.ncols();
    if grad_t.len() != n_obs * latent_dim {
        return Err(format!(
            "latent grad_t shape mismatch: expected {}, got {}",
            n_obs * latent_dim,
            grad_t.len()
        ));
    }
    if fit.coefficients.nrows() != p || fit.fitted.nrows() != n_obs {
        return Err("latent REML fit shape mismatch".to_string());
    }
    if y.dim() != (n_obs, n_outputs) {
        return Err(format!(
            "latent REML response shape mismatch: expected {}x{}, got {}x{}",
            n_obs,
            n_outputs,
            y.nrows(),
            y.ncols()
        ));
    }
    let weight = gaussian_reml_weight_vector_local(n_obs, weights)?;
    let factor = latent_augmented_hessian_factor(design, penalty, weight.view(), fit.lambda)?;
    if jet.shape() != [n_obs, p, latent_dim] {
        return Err(format!(
            "latent REML jet shape mismatch: expected {}x{}x{}, got {}x{}x{}",
            n_obs,
            p,
            latent_dim,
            jet.shape()[0],
            jet.shape()[1],
            jet.shape()[2],
        ));
    }
    let mut rhs = Array2::<f64>::zeros((p, 1));
    let mut direction = Array1::<f64>::zeros(p);
    for n in 0..n_obs {
        for col in 0..p {
            rhs[[col, 0]] = design[[n, col]];
        }
        {
            let mut rhs_view = array2_to_matmut(&mut rhs);
            factor.solve_in_place(rhs_view.as_mut());
        }
        for col in 0..p {
            direction[col] = (n_outputs as f64) * rhs[[col, 0]];
        }
        for output in 0..n_outputs {
            let sigma_eff2 = match sigma_eff_mode {
                SigmaEffMode::Profiled => fit.sigma2[output],
                // Fixed-dispersion latent REML is wired as an explicit mode so
                // the Python/Rust boundary is stable. The current closed-form
                // forward does not accept an external σ² yet, so the fixed path
                // uses the fit's σ² slot until that forward parameter lands.
                SigmaEffMode::Fixed => fit.sigma2[output],
            };
            if !(sigma_eff2.is_finite() && sigma_eff2 > 0.0) {
                return Err(format!(
                    "sigma_eff2 must be finite and positive; got {sigma_eff2}"
                ));
            }
            let residual = y[[n, output]] - fit.fitted[[n, output]];
            let data_scale = residual / sigma_eff2;
            for col in 0..p {
                direction[col] -= data_scale * fit.coefficients[[col, output]];
            }
        }
        let row_scale = scale * weight[n];
        for col in 0..p {
            let d_col = direction[col];
            if d_col == 0.0 {
                continue;
            }
            let col_scale = row_scale * d_col;
            for a in 0..latent_dim {
                grad_t[n * latent_dim + a] += col_scale * jet[[n, col, a]];
            }
        }
    }
    Ok(())
}

fn analytic_penalty_value_for_targets(
    registry: &AnalyticPenaltyRegistry,
    target_t: ArrayView1<'_, f64>,
    target_beta: Option<ArrayView1<'_, f64>>,
) -> Result<f64, String> {
    let rho = Array1::<f64>::zeros(registry.total_rho_count());
    let mut value = 0.0_f64;
    for (penalty, (rho_slice, tier, _name)) in registry.penalties.iter().zip(registry.rho_layout())
    {
        let rho_local = rho.slice(s![rho_slice]);
        let target = match (tier, penalty) {
            (PenaltyTier::Psi, _) => target_t.view(),
            (PenaltyTier::Beta, _) => {
                let Some(target_beta) = target_beta.as_ref() else {
                    continue;
                };
                target_beta.view()
            }
            (PenaltyTier::Rho, _) => continue,
        };
        value += penalty.value(target, rho_local);
    }
    Ok(value)
}

fn gaussian_reml_fit_latent_impl(
    t_flat: ArrayView1<'_, f64>,
    y: ArrayView2<'_, f64>,
    n_obs: usize,
    latent_dim: usize,
    centers: ArrayView2<'_, f64>,
    m: usize,
    basis_kind: &str,
    tensor_knots_concat: Option<ArrayView1<'_, f64>>,
    tensor_knot_offsets: Option<&[usize]>,
    tensor_degrees: Option<&[usize]>,
    penalty: ArrayView2<'_, f64>,
    weights: Option<ArrayView1<'_, f64>>,
    init_lambda: Option<f64>,
    aux_u: Option<ArrayView2<'_, f64>>,
    aux_family: AuxPriorFamily,
    aux_strength: Option<f64>,
    dim_selection_precision: Option<ArrayView1<'_, f64>>,
    analytic_penalties: Option<&AnalyticPenaltyRegistry>,
    periodic: Option<&[Option<f64>]>,
) -> Result<
    (
        gam::solver::gaussian_reml::GaussianRemlMultiResult,
        Array2<f64>,
        Option<LatentAuxStrengthState>,
    ),
    String,
> {
    let (design, t_mat, _jet) = build_latent_forward_design(
        basis_kind,
        t_flat,
        n_obs,
        latent_dim,
        centers,
        m,
        tensor_knots_concat,
        tensor_knot_offsets,
        tensor_degrees,
        periodic,
    )?;
    // Build the (optionally) augmented Y/X stack carrying the identifiability
    // penalty. The penalty `½ μ ‖t − t_ref‖²` is *not* on the design Φ; it
    // acts on t directly. Because t enters Φ nonlinearly, we cannot fold it
    // into the inner Gaussian-closed-form solve without changing the solver.
    // We therefore evaluate the *penalty contribution* here and return it
    // for the caller to expose; the inner ridge stays unchanged.
    //
    // The forward path's responsibility is to produce a self-consistent fit
    // at the current t; the outer loop owns the gauge enforcement (it adds
    // ∂R_id/∂t to grad_t and walks t under that combined gradient).
    let mut fit = gaussian_reml_multi_closed_form_with_cache(
        design.view(),
        y,
        penalty,
        weights,
        init_lambda,
        None,
    )
    .map_err(|err| err.to_string())?;
    // Fixes audit-revised claim that ARD / aux-prior REML selection requires
    // normalized priors, not raw quadratic corrections alone.
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
        latent_prior_score += analytic_penalty_value_for_targets(registry, t_flat, None)?;
    }
    fit.reml_score += latent_prior_score;
    Ok((fit, design, aux_strength_state))
}

/// Forward fit: build the latent design at the current latent `t`,
/// solve the Gaussian REML inner problem, and return the standard
/// REML fit dictionary plus the materialized design (for warm-starts).
///
/// `t` is a flat row-major `(n_obs * latent_dim)` array; `n_obs` and
/// `latent_dim` must be passed explicitly because the flat vector cannot
/// carry shape.
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
    analytic_penalties = None,
    basis_kind = "duchon".to_string(),
    tensor_knots_concat = None,
    tensor_knot_offsets = None,
    tensor_degrees = None,
))]
fn gaussian_reml_fit_latent<'py>(
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
    analytic_penalties: Option<String>,
    basis_kind: String,
    tensor_knots_concat: Option<PyReadonlyArray1<'py, f64>>,
    tensor_knot_offsets: Option<Vec<usize>>,
    tensor_degrees: Option<Vec<usize>>,
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
    let family = match aux_family.to_ascii_lowercase().as_str() {
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
    let tensor_knots_values = tensor_knots_concat
        .as_ref()
        .map(|a| a.as_array().to_owned());
    if let Some(fw) = fisher_values.as_ref() {
        let n_outputs = y_values.ncols();
        if fw.shape() == [n_obs, n_outputs, n_outputs] && n_outputs > 1 {
            let (design, t_mat, _jet) = build_latent_forward_design(
                &basis_kind,
                t_values.view(),
                n_obs,
                latent_dim,
                centers_values.view(),
                m,
                tensor_knots_values.as_ref().map(|a| a.view()),
                tensor_knot_offsets.as_deref(),
                tensor_degrees.as_deref(),
                // Standalone fit entrypoint: no manifold/chart concept, so the
                // latent design stays the open Euclidean basis (unchanged).
                None,
            )
            .map_err(py_value_error)?;
            let (prior_score, aux_strength_state) = latent_prior_score_and_aux_state_for_t(
                t_mat.view(),
                aux_u_values.as_ref().map(|a| a.view()),
                family,
                aux_strength,
                dim_selection_values.as_ref().map(|a| a.view()),
            )
            .map_err(py_value_error)?;
            let analytic_score =
                analytic_penalty_value_for_targets(&registry, t_values.view(), None)
                    .map_err(py_value_error)?;
            return dense_fisher_gaussian_fit_to_pydict(
                py,
                design.view(),
                y_values.view(),
                penalty_values.view(),
                weight_values.as_ref().map(|w| w.view()),
                fw.view(),
                init_lambda,
                prior_score + analytic_score,
                aux_strength_state,
            );
        }
    }
    let analytic_penalties_for_thread = analytic_penalties.clone();
    let latent_payload_for_thread = latent_payload.clone();
    let t_for_echo = t_values.clone();
    let (fit, _design, aux_strength_state) =
        detach_py_result(py, "gaussian_reml_fit_latent", move || {
            let registry = build_analytic_penalty_registry_from_json(
                Some(&latent_payload_for_thread),
                analytic_penalties_for_thread.as_ref(),
            )?;
            let effective_weights = latent_scalar_weights_with_fisher(
                n_obs,
                weight_values.as_ref().map(|w| w.view()),
                fisher_values.as_ref().map(|w| w.view()),
            )?;
            gaussian_reml_fit_latent_impl(
                t_values.view(),
                y_values.view(),
                n_obs,
                latent_dim,
                centers_values.view(),
                m,
                &basis_kind,
                tensor_knots_values.as_ref().map(|a| a.view()),
                tensor_knot_offsets.as_deref(),
                tensor_degrees.as_deref(),
                penalty_values.view(),
                effective_weights.as_ref().map(|w| w.view()),
                init_lambda,
                aux_u_values.as_ref().map(|a| a.view()),
                family,
                aux_strength,
                dim_selection_values.as_ref().map(|a| a.view()),
                Some(&registry),
                // Standalone fit entrypoint: no manifold/chart, open Euclidean.
                None,
            )
        })?;
    let out = PyDict::new(py);
    set_ok_gaussian_reml_items(py, &out, fit)?;
    set_aux_strength_items(py, &out, aux_strength_state)?;
    // Echo the latent this fit was evaluated at. The forward primitive solves a
    // single `β | t`; it does not move `t`, so the returned `t` equals the input.
    // Callers driving the outer loop (or `gaussian_reml_optimize_latent`) read
    // it back rather than having to thread the input through themselves.
    let t_matrix = t_for_echo
        .into_shape_with_order((n_obs, latent_dim))
        .map_err(shape_error_to_pyerr)?;
    out.set_item("t", t_matrix.into_pyarray(py))?;
    Ok(out.unbind())
}

/// Canonicalize every Python-facing assignment token through the Rust-owned
/// schema shared with the public facade and distilled encoder.
fn canonicalize_assignment_kind(kind: &str) -> Result<String, String> {
    crate::manifold::manifold_sae_coercion::canonical_assignment_kind(kind).map(str::to_string)
}

// The OOS assignment-logit seeders moved to the library
// (`gam_sae::manifold::oos_logit_seed`, methods on `SaeManifoldTerm`) so the CLI
// and Rust callers seed identically to python — issue #2236. The binding calls
// `term.seed_oos_softmax_logits_from_projection_residuals(..)` /
// `term.seed_oos_ibp_logits_from_projected_decoder_lsq(..)` directly.

fn gumbel_temperature_schedule_from_pydict(
    schedule: Option<&Bound<'_, PyDict>>,
) -> Result<Option<GumbelTemperatureSchedule>, String> {
    fn get<'py>(state: &'py Bound<'py, PyDict>, key: &str) -> Result<Bound<'py, PyAny>, String> {
        state
            .get_item(key)
            .map_err(|err| err.to_string())?
            .ok_or_else(|| format!("gumbel_schedule is missing key {key:?}"))
    }

    let Some(schedule) = schedule else {
        return Ok(None);
    };
    let decay_name = get(schedule, "decay")?
        .extract::<String>()
        .map_err(|err| err.to_string())?
        .to_ascii_lowercase()
        .replace('-', "_");
    let tau_start = get(schedule, "tau_start")?
        .extract::<f64>()
        .map_err(|err| err.to_string())?;
    let tau_min = match schedule
        .get_item("tau_min")
        .map_err(|err| err.to_string())?
    {
        Some(value) => value.extract::<f64>().map_err(|err| err.to_string())?,
        None => return Err("gumbel_schedule is missing key \"tau_min\"".to_string()),
    };
    let decay = match decay_name.as_str() {
        "geometric" => {
            // Prefer the (tau_start, tau_min, steps) endpoints spec and derive
            // the rate here; fall back to an explicit `rate` (default 0.9).
            let steps = match schedule.get_item("steps").map_err(|err| err.to_string())? {
                Some(value) => Some(value.extract::<usize>().map_err(|err| err.to_string())?),
                None => None,
            };
            let rate = match steps {
                Some(steps) => ScheduleKind::geometric_rate_from_steps(tau_start, tau_min, steps),
                None => match schedule.get_item("rate").map_err(|err| err.to_string())? {
                    Some(value) => value.extract::<f64>().map_err(|err| err.to_string())?,
                    None => 0.9,
                },
            };
            ScheduleKind::Geometric { rate }
        }
        "linear" => {
            let steps = get(schedule, "steps")?
                .extract::<usize>()
                .map_err(|err| err.to_string())?;
            ScheduleKind::Linear { steps }
        }
        "reciprocal_iter" => ScheduleKind::ReciprocalIter,
        other => {
            return Err(format!(
                "gumbel_schedule decay must be 'geometric', 'linear', or 'reciprocal_iter'; got {other:?}"
            ));
        }
    };
    let mut schedule_out = GumbelTemperatureSchedule::new(tau_start, tau_min, decay)?;
    if let Some(iter_count) = schedule
        .get_item("iter_count")
        .map_err(|err| err.to_string())?
    {
        schedule_out.iter_count = iter_count
            .extract::<usize>()
            .map_err(|err| err.to_string())?;
        schedule_out.validate()?;
    }
    Ok(Some(schedule_out))
}

fn structured_residual_pass_diagnostics_dict<'py>(
    py: Python<'py>,
    diagnostics: &[gam::terms::sae::manifold::StructuredResidualPassDiagnostic],
) -> PyResult<Bound<'py, PyList>> {
    let out = PyList::empty(py);
    for d in diagnostics {
        let item = PyDict::new(py);
        item.set_item("pass", d.pass)?;
        item.set_item("gamma", d.gamma)?;
        item.set_item("factor_rank", d.factor_rank)?;
        item.set_item("log_evidence", d.log_evidence)?;
        item.set_item("factor_energy", d.factor_energy)?;
        item.set_item("diagonal_mean", d.diagonal_mean)?;
        item.set_item("dispersion_before", d.dispersion_before)?;
        item.set_item("dispersion_after", d.dispersion_after)?;
        item.set_item(
            "log_lambda_smooth_before",
            d.log_lambda_smooth_before.clone(),
        )?;
        item.set_item("log_lambda_smooth_after", d.log_lambda_smooth_after.clone())?;
        out.append(item)?;
    }
    Ok(out)
}

/// #2071 residual-factor promotion alignment gate. For two independent unit
/// directions in an r-dimensional subspace, cos²θ follows
/// Beta(1/2, (r−1)/2); use its 95th percentile as the per-pass false-merge
/// threshold. Rank one has no informative angle, so the threshold is exactly
/// one. The fit engine receives this function because the quantile routine lives
/// above `gam-sae` in the crate graph.
fn sae_promotion_align_min(factor_rank: usize) -> f64 {
    if factor_rank <= 1 {
        return 1.0;
    }
    let rank = factor_rank as f64;
    let cos2 = gam::inference::probability::beta_quantile(0.95, 0.5, (rank - 1.0) / 2.0);
    if cos2.is_finite() {
        cos2.sqrt().clamp(0.0, 1.0)
    } else {
        1.0
    }
}

/// Fit a SAE-manifold term end-to-end in Rust: up to `max_iter` Newton steps
/// per λ_smooth candidate, refreshing `Phi` and `dPhi/dt` between steps via
/// the per-atom [`SaeBasisSecondJet`] (analytic harmonic for `Periodic`
/// `latent_dim == 1`, chart for `Sphere`, tensor harmonic for `Torus`; the
/// radial+polynomial Duchon and monomial Euclidean evaluators refresh from the
/// auto path that threads their metadata). This precomputed-basis entry point
/// carries no Duchon centers, so a Duchon/precomputed atom here errors instead
/// of freezing the seed snapshot. The smoothing / sparsity / ARD strengths
/// (ρ) are selected automatically by the generic outer cascade driving the
/// term's REML criterion — no per-call λ-grid.
#[pyfunction(signature = (
    z,
    atom_basis,
    atom_dim,
    basis_values,
    basis_jacobian,
    basis_sizes,
    decoder_coefficients,
    smooth_penalties,
    initial_logits,
    initial_coords,
    alpha,
    tau,
    learnable_alpha,
    assignment_kind,
    sparsity_strength = 1.0,
    smoothness = 1.0,
    max_iter = 12,
    learning_rate = 1.0,
    ridge_ext_coord = 1.0e-6,
    ridge_beta = 1.0e-6,
    gumbel_schedule = None,
    analytic_penalties = None,
    top_k = None,
    jumprelu_threshold = 0.0,
    fisher_factors = None,
    fisher_mass_residual = None,
    fisher_provenance = None,
    row_loss_weights = None,
    separation_barrier_strength_override = None,
    ibp_alpha_override = None,
    // #2239 magic-by-default: evidence-certified residual structure is promoted
    // to the primary tier by default (the certificate gates the birth; the
    // alternation self-extends its pass budget only while lineages are live).
    promote_from_residual = true,
    run_structure_search = true,
    run_outer_rho_search = true,
    data_row_reseed = false,
))]
fn sae_manifold_fit<'py>(
    py: Python<'py>,
    z: PyReadonlyArray2<'py, f64>,
    atom_basis: Vec<String>,
    atom_dim: Vec<usize>,
    basis_values: PyReadonlyArray3<'py, f64>,
    basis_jacobian: PyReadonlyArray4<'py, f64>,
    basis_sizes: Vec<usize>,
    decoder_coefficients: PyReadonlyArray3<'py, f64>,
    smooth_penalties: PyReadonlyArray3<'py, f64>,
    initial_logits: PyReadonlyArray2<'py, f64>,
    initial_coords: PyReadonlyArray3<'py, f64>,
    alpha: f64,
    tau: f64,
    learnable_alpha: bool,
    assignment_kind: String,
    sparsity_strength: f64,
    smoothness: f64,
    max_iter: usize,
    learning_rate: f64,
    ridge_ext_coord: f64,
    ridge_beta: f64,
    gumbel_schedule: Option<&Bound<'py, PyDict>>,
    analytic_penalties: Option<String>,
    top_k: Option<usize>,
    jumprelu_threshold: f64,
    // WP-D output-Fisher shard (#980). `fisher_factors` is `(n, p, r)` f64 (the
    // harvest shard's `U`); its presence activates `RowMetric::OutputFisher`. No
    // flag — magic-by-default. `fisher_mass_residual` is the optional `(n,)`
    // truncation diagnostic that rides into the report.
    fisher_factors: Option<PyReadonlyArray3<'py, f64>>,
    fisher_mass_residual: Option<PyReadonlyArray1<'py, f64>>,
    // The harvest shard's provenance tag (#980): `"output_fisher"` (same-position,
    // the default) or `"output_fisher_downstream"` (KV-path aggregate over future
    // positions). Selects which output-Fisher `RowMetric` is installed; the gauge
    // / lens / dose consume either unchanged. Ignored when no shard is supplied.
    fisher_provenance: Option<String>,
    // Per-row design-honesty reconstruction weights (#977); `(n,)` √w. Absent ⇒
    // unweighted path. Installed on the term before the joint fit / ρ selection.
    row_loss_weights: Option<PyReadonlyArray1<'py, f64>>,
    // Per-fit config (separation-barrier strength / IBP-α). `Some` pins this
    // term's value; `None` selects the canonical data-derived or mode default.
    separation_barrier_strength_override: Option<f64>,
    ibp_alpha_override: Option<f64>,
    promote_from_residual: bool,
    run_structure_search: bool,
    run_outer_rho_search: bool,
    data_row_reseed: bool,
) -> PyResult<Py<PyDict>> {
    // The precomputed-basis entry point carries no Duchon centers / kernel
    // metadata, so any basis kind whose refresh needs them cannot re-evaluate
    // `Phi(t)` at updated coordinates. Pass empty per-atom centers; the
    // evaluator builder errors for such kinds (rather than silently freezing
    // the seed snapshot). Kinds with an analytic, centers-free basis
    // (periodic, sphere, torus) refresh as usual.
    let atom_centers: Vec<Option<Array2<f64>>> = vec![None; atom_basis.len()];
    // Validate the strict canonical assignment token before the inner driver parses it.
    let assignment_kind = canonicalize_assignment_kind(&assignment_kind).map_err(py_value_error)?;
    let fisher_u = fisher_factors.as_ref().map(|f| f.as_array());
    let fisher_mr = fisher_mass_residual.as_ref().map(|m| m.as_array());
    let row_w = row_loss_weights.as_ref().map(|w| w.as_array());
    sae_manifold_fit_inner(
        py,
        z.as_array(),
        &atom_basis,
        atom_dim,
        &atom_centers,
        basis_values.as_array(),
        basis_jacobian.as_array(),
        basis_sizes,
        decoder_coefficients.as_array(),
        smooth_penalties.as_array(),
        initial_logits.as_array(),
        initial_coords.as_array(),
        alpha,
        tau,
        learnable_alpha,
        assignment_kind,
        sparsity_strength,
        smoothness,
        max_iter,
        learning_rate,
        ridge_ext_coord,
        ridge_beta,
        gumbel_schedule,
        analytic_penalties,
        top_k,
        jumprelu_threshold,
        true,
        // This precomputed-basis entry point hands the term verbatim seeds;
        // routing-seed refinement is owned by the higher-level `fit_minimal`
        // auto path, so do not re-seed here (random_state unused when off).
        false,
        0,
        fisher_u,
        fisher_mr,
        fisher_provenance.as_deref(),
        row_w,
        separation_barrier_strength_override,
        ibp_alpha_override,
        promote_from_residual,
        run_structure_search,
        run_outer_rho_search,
        data_row_reseed,
    )
}

/// Structural string tag for a fitted atom's basis kind (the discrete topology
/// identity the SAC payload exposes). Exhaustive over [`SaeAtomBasisKind`] so a
/// new topology surfaces as a compile error rather than a silent mislabel.
fn stagewise_basis_kind_tag(kind: &SaeAtomBasisKind) -> &'static str {
    match kind {
        SaeAtomBasisKind::Duchon => "duchon",
        SaeAtomBasisKind::Periodic => "periodic",
        SaeAtomBasisKind::Sphere => "sphere",
        SaeAtomBasisKind::Torus => "torus",
        SaeAtomBasisKind::Cylinder => "cylinder",
        SaeAtomBasisKind::Mobius => "mobius",
        SaeAtomBasisKind::Linear => "linear",
        SaeAtomBasisKind::EuclideanPatch => "euclidean_patch",
        SaeAtomBasisKind::Poincare => "poincare",
        SaeAtomBasisKind::FiniteSet => "finite_set",
        SaeAtomBasisKind::Precomputed(_) => "precomputed",
    }
}

fn stagewise_birth_kind_tag(kind: gam::terms::sae::manifold::BirthKind) -> &'static str {
    match kind {
        gam::terms::sae::manifold::BirthKind::NewAtom => "new_atom",
        gam::terms::sae::manifold::BirthKind::ChartExtension => "chart_extension",
    }
}

fn stagewise_event_kind_tag(kind: gam::terms::sae::manifold::StagewiseEventKind) -> &'static str {
    match kind {
        gam::terms::sae::manifold::StagewiseEventKind::SeedReady => "seed_ready",
        gam::terms::sae::manifold::StagewiseEventKind::BirthRoundStarted => "birth_round_started",
        gam::terms::sae::manifold::StagewiseEventKind::ResidualModelStarted => {
            "residual_model_started"
        }
        gam::terms::sae::manifold::StagewiseEventKind::ResidualModelFitted => {
            "residual_model_fitted"
        }
        gam::terms::sae::manifold::StagewiseEventKind::CurrentEvidenceStarted => {
            "current_evidence_started"
        }
        gam::terms::sae::manifold::StagewiseEventKind::CurrentEvidenceFinished => {
            "current_evidence_finished"
        }
        gam::terms::sae::manifold::StagewiseEventKind::CandidateStarted => "candidate_started",
        gam::terms::sae::manifold::StagewiseEventKind::CandidateFinished => "candidate_finished",
        gam::terms::sae::manifold::StagewiseEventKind::BirthAccepted => "birth_accepted",
        gam::terms::sae::manifold::StagewiseEventKind::BirthRejected => "birth_rejected",
        gam::terms::sae::manifold::StagewiseEventKind::BackfitSweepStarted => {
            "backfit_sweep_started"
        }
        gam::terms::sae::manifold::StagewiseEventKind::BackfitSweepAccepted => {
            "backfit_sweep_accepted"
        }
        gam::terms::sae::manifold::StagewiseEventKind::BackfitSweepRejected => {
            "backfit_sweep_rejected"
        }
        gam::terms::sae::manifold::StagewiseEventKind::TerminalEvidenceCompleted => {
            "terminal_evidence_completed"
        }
    }
}

fn stagewise_atoms_py<'py>(
    py: Python<'py>,
    term: &gam::terms::sae::manifold::SaeManifoldTerm,
) -> PyResult<Bound<'py, PyList>> {
    let assignments = term.assignment.assignments();
    let atoms_py = PyList::empty(py);
    for atom_idx in 0..term.k_atoms() {
        let atom = &term.atoms[atom_idx];
        let atom_dict = PyDict::new(py);
        atom_dict.set_item(
            // #2135/#2237 — persistence consumes BOTH fit-internal gauges: expand
            // the reduced frame (`Q B̃`) and materialize the scale quotient
            // (`exp(s_k) Q B̃`). A reloaded checkpoint therefore receives the
            // physical `M × p` decoder and can set `log_amplitude = 0` without
            // changing the fitted atom.
            "decoder_B",
            atom.physical_full_width_decoder().into_pyarray(py),
        )?;
        atom_dict.set_item("basis_kind", stagewise_basis_kind_tag(&atom.basis_kind))?;
        atom_dict.set_item("latent_dim", atom.latent_dim)?;
        atom_dict.set_item(
            "on_atom_coords_t",
            term.assignment.coords[atom_idx]
                .as_matrix()
                .into_pyarray(py),
        )?;
        atom_dict.set_item(
            "assignments_z",
            assignments.column(atom_idx).to_owned().into_pyarray(py),
        )?;
        atoms_py.append(atom_dict)?;
    }
    Ok(atoms_py)
}

fn stagewise_checkpoint_py<'py>(
    py: Python<'py>,
    term: &gam::terms::sae::manifold::SaeManifoldTerm,
    rho: &SaeManifoldRho,
) -> PyResult<Bound<'py, PyDict>> {
    let checkpoint = PyDict::new(py);
    checkpoint.set_item("k_final", term.k_atoms())?;
    checkpoint.set_item("atoms", stagewise_atoms_py(py, term)?)?;
    checkpoint.set_item("logits", term.assignment.logits.clone().into_pyarray(py))?;
    checkpoint.set_item("log_lambda_sparse", rho.log_lambda_sparse)?;
    checkpoint.set_item(
        "log_lambda_smooth",
        Array1::from(rho.log_lambda_smooth.clone()).into_pyarray(py),
    )?;
    let log_ard_py = PyList::empty(py);
    for atom_log_ard in &rho.log_ard {
        log_ard_py.append(atom_log_ard.clone().into_pyarray(py))?;
    }
    checkpoint.set_item("log_ard", log_ard_py)?;
    Ok(checkpoint)
}

fn stagewise_progress_py<'py>(
    py: Python<'py>,
    event: &gam::terms::sae::manifold::StagewiseProgress<'_>,
) -> PyResult<Py<PyDict>> {
    let out = PyDict::new(py);
    out.set_item("event", stagewise_event_kind_tag(event.event))?;
    out.set_item("birth_round", event.birth_round)?;
    out.set_item("backfit_sweep", event.backfit_sweep)?;
    out.set_item("candidate", event.candidate.map(stagewise_birth_kind_tag))?;
    out.set_item("accepted", event.accepted)?;
    out.set_item("checkpoint_available", event.checkpoint)?;
    out.set_item("k", event.k_atoms)?;
    out.set_item("births_accepted", event.births_accepted)?;
    out.set_item("births_rejected", event.births_rejected)?;
    out.set_item("ev", event.ev)?;
    out.set_item("factor_energy", event.factor_energy)?;
    out.set_item("joint_reml_before", event.joint_reml_before)?;
    out.set_item("joint_reml_after", event.joint_reml_after)?;
    out.set_item("terminal_joint_reml", event.terminal_joint_reml)?;
    if event.checkpoint {
        out.set_item(
            "checkpoint",
            stagewise_checkpoint_py(py, event.term, event.rho)?,
        )?;
    } else {
        out.set_item("checkpoint", py.None())?;
    }
    Ok(out.unbind())
}

/// SAC — Sequential Atom Composition entry. Grows a curved dictionary from a
/// single K=1 seed atom by forward-stagewise births + backfitting, then reports
/// the terminal frozen joint evidence. This is the productionised
/// [`fit_stagewise`](gam::terms::sae::manifold::fit_stagewise) driver behind a
/// thin FFI: the seed arrays are the K=1 slice of [`sae_manifold_fit`]'s
/// precomputed-basis inputs (every array leads with a length-1 atom axis). The
/// returned payload is compact — the composed per-atom dictionary, the
/// by-construction-monotone EV traces, the birth ledger, and the terminal joint
/// REML — enough for the SAC prototype to reconstruct and score. ρ is fixed at the
/// dispersion-scaled seed (the outer EFS cascade is the caller's job); the
/// stagewise driver is monotone at that ρ by construction.
#[pyfunction(signature = (
    z,
    atom_basis,
    atom_dim,
    basis_values,
    basis_jacobian,
    basis_sizes,
    decoder_coefficients,
    smooth_penalties,
    initial_logits,
    initial_coords,
    alpha,
    tau,
    learnable_alpha,
    assignment_kind,
    sparsity_strength = 0.0,
    smoothness = 1.0,
    max_iter = 64,
    learning_rate = 1.0,
    ridge_ext_coord = 1.0e-6,
    ridge_beta = 1.0e-6,
    max_births = 32,
    max_backfit_sweeps = 4,
    min_effect_ev = 0.0,
    max_factor_rank = 4,
    structured_whitening = true,
    row_loss_weights = None,
    progress_callback = None,
    // #1939 — appended LAST so the signature stays strictly additive: existing
    // positional/kwarg callers (which never pass this) are byte-for-byte unaffected.
    cone_atom_recovery = false,
    // Rung 1 (B4) — the harvest-emitted output-Fisher factor stack `(n, p, r)` and
    // its provenance tag. Appended LAST so the signature stays strictly additive.
    // Presence installs the metric on the seed term (carried across every birth /
    // backfit clone); `"behavioral_fisher"` makes it the GLS reconstruction
    // likelihood weight (nats). Conflicts with `structured_whitening` (which refits
    // its own Σ⁻¹ metric per birth) — supply one or the other, not both.
    fisher_factors = None,
    fisher_provenance = None,
))]
fn sae_manifold_fit_stagewise<'py>(
    py: Python<'py>,
    z: PyReadonlyArray2<'py, f64>,
    atom_basis: Vec<String>,
    atom_dim: Vec<usize>,
    basis_values: PyReadonlyArray3<'py, f64>,
    basis_jacobian: PyReadonlyArray4<'py, f64>,
    basis_sizes: Vec<usize>,
    decoder_coefficients: PyReadonlyArray3<'py, f64>,
    smooth_penalties: PyReadonlyArray3<'py, f64>,
    initial_logits: PyReadonlyArray2<'py, f64>,
    initial_coords: PyReadonlyArray3<'py, f64>,
    alpha: f64,
    tau: f64,
    learnable_alpha: bool,
    assignment_kind: String,
    sparsity_strength: f64,
    smoothness: f64,
    max_iter: usize,
    learning_rate: f64,
    ridge_ext_coord: f64,
    ridge_beta: f64,
    max_births: usize,
    max_backfit_sweeps: usize,
    min_effect_ev: f64,
    max_factor_rank: usize,
    structured_whitening: bool,
    row_loss_weights: Option<PyReadonlyArray1<'py, f64>>,
    progress_callback: Option<PyObject>,
    cone_atom_recovery: bool,
    fisher_factors: Option<PyReadonlyArray3<'py, f64>>,
    fisher_provenance: Option<String>,
) -> PyResult<Py<PyDict>> {
    let assignment_kind = canonicalize_assignment_kind(&assignment_kind).map_err(py_value_error)?;
    let assignment = SaeFitAssignmentKind::from_tag(&assignment_kind).map_err(py_value_error)?;
    let z_view = z.as_array();
    let fisher_metric = match fisher_factors.as_ref() {
        Some(factors) => Some(
            SaeFisherRowMetricRequest::from_tag(
                factors.as_array(),
                z_view.nrows(),
                z_view.ncols(),
                fisher_provenance.as_deref(),
                None,
            )
            .map_err(py_value_error)?,
        ),
        None => None,
    };
    let seed = build_sae_stagewise_seed(SaeStagewiseSeedRequest {
        target: z_view,
        atom_basis: &atom_basis,
        atom_dim: &atom_dim,
        basis_values: basis_values.as_array(),
        basis_jacobian: basis_jacobian.as_array(),
        basis_sizes: &basis_sizes,
        decoder_coefficients: decoder_coefficients.as_array(),
        smooth_penalties: smooth_penalties.as_array(),
        initial_logits: initial_logits.as_array(),
        initial_coords: initial_coords.as_array(),
        alpha,
        tau,
        learnable_alpha,
        assignment_kind: assignment,
        sparsity_strength,
        smoothness,
        max_iter,
        learning_rate,
        ridge_ext_coord,
        ridge_beta,
        cone_atom_recovery,
        structured_whitening,
        fisher_metric,
    })
    .map_err(py_value_error)?;
    let SaeStagewiseSeedReport {
        base_term,
        initial_rho: init_rho,
    } = seed;

    let weights: Option<Vec<f64>> = row_loss_weights.as_ref().map(|w| w.as_array().to_vec());
    let config = gam::terms::sae::manifold::StagewiseConfig {
        inner_max_iter: max_iter,
        learning_rate,
        ridge_ext_coord,
        ridge_beta,
        max_births,
        max_backfit_sweeps,
        min_effect_ev,
        max_factor_rank,
        structured_whitening,
    };
    let mut progress_callback = progress_callback.map(|callback| {
        move |event: gam::terms::sae::manifold::StagewiseProgress<'_>| -> Result<(), String> {
            Python::attach(|py| {
                let payload = stagewise_progress_py(py, &event).map_err(|err| err.to_string())?;
                callback
                    .call1(py, (payload,))
                    .map_err(|err| err.to_string())?;
                Ok(())
            })
        }
    });
    let progress_hook: Option<&mut gam::terms::sae::manifold::StagewiseProgressCallback<'_>> =
        progress_callback.as_mut().map(|callback| {
            callback as &mut gam::terms::sae::manifold::StagewiseProgressCallback<'_>
        });
    // #2138 — with no progress callback, run the whole multi-minute forward-birth
    // + backfit solve through the GIL-releasing interruptible driver: on a Python
    // interrupt it sets `cancel_flag`, which the stagewise loop polls per birth /
    // per backfit sweep to stop the abandoned compose worker promptly. A callback
    // keeps the direct call — the `&mut dyn` hook is not Ungil (can't cross
    // `detach`) and already re-enters Python (raising there) each event.
    let cancel_flag = std::sync::Arc::new(std::sync::atomic::AtomicBool::new(false));
    let result = match progress_hook {
        Some(hook) => gam::terms::sae::manifold::fit_stagewise(
            base_term,
            init_rho,
            z_view,
            None,
            weights.as_deref(),
            &config,
            Some(hook),
            None,
        ),
        None => {
            // The worker thread is `'static`, but `z_view` borrows the (Python)
            // numpy buffer — own the target so nothing non-`'static` crosses in
            // (the primary fit path owns its target the same way).
            let target_owned = z_view.to_owned();
            let worker_cancel = std::sync::Arc::clone(&cancel_flag);
            run_sae_fit_interruptible(py, "gam-sae-stagewise", &cancel_flag, move || {
                gam::terms::sae::manifold::fit_stagewise(
                    base_term,
                    init_rho,
                    target_owned.view(),
                    None,
                    weights.as_deref(),
                    &config,
                    None,
                    Some(&*worker_cancel),
                )
            })?
        }
    }
    .map_err(py_value_error)?;

    let term = result.term;
    let rho = result.rho;
    let report = result.report;
    let k_final = term.k_atoms();

    let atoms_py = stagewise_atoms_py(py, &term)?;

    let births_py = PyList::empty(py);
    for rec in &report.birth_records {
        let d = PyDict::new(py);
        d.set_item("kind", stagewise_birth_kind_tag(rec.kind))?;
        d.set_item("delta_ev", rec.delta_ev)?;
        d.set_item("factor_energy", rec.factor_energy)?;
        d.set_item("joint_reml_before", rec.joint_reml_before)?;
        d.set_item("joint_reml_after", rec.joint_reml_after)?;
        d.set_item("accepted", rec.accepted)?;
        births_py.append(d)?;
    }

    let stopped_reason = match report.stopped_reason {
        gam::terms::sae::manifold::StagewiseStop::TwoConsecutiveRejections => {
            "two_consecutive_rejections"
        }
        gam::terms::sae::manifold::StagewiseStop::MaxBirths => "max_births",
        gam::terms::sae::manifold::StagewiseStop::NoResidualStructure => "no_residual_structure",
        gam::terms::sae::manifold::StagewiseStop::Cancelled => "cancelled",
    };

    let log_ard_py = PyList::empty(py);
    for atom_log_ard in &rho.log_ard {
        log_ard_py.append(atom_log_ard.clone().into_pyarray(py))?;
    }

    let out = PyDict::new(py);
    out.set_item("k_final", k_final)?;
    // #1939 — echo the resolved cone-atom RECOVERY opt-in so a harness can VERIFY
    // the kwarg engaged (no silent no-op): the value the fit actually ran with.
    out.set_item("cone_atom_recovery_used", cone_atom_recovery)?;
    out.set_item("births_accepted", report.births_accepted)?;
    out.set_item("births_rejected", report.births_rejected)?;
    out.set_item("stopped_reason", stopped_reason)?;
    out.set_item("terminal_joint_reml", report.terminal_joint_reml)?;
    out.set_item("terminal_data_fit", report.terminal_joint_loss.data_fit)?;
    out.set_item(
        "ev_trace",
        Array1::from(report.ev_trace.clone()).into_pyarray(py),
    )?;
    out.set_item(
        "backfit_ev_trace",
        Array1::from(report.backfit_ev_trace.clone()).into_pyarray(py),
    )?;
    out.set_item("logits", term.assignment.logits.clone().into_pyarray(py))?;
    out.set_item("atoms", atoms_py)?;
    out.set_item("birth_records", births_py)?;
    out.set_item("log_ard", log_ard_py)?;
    out.set_item(
        "log_lambda_smooth",
        Array1::from(rho.log_lambda_smooth.clone()).into_pyarray(py),
    )?;
    out.set_item("log_lambda_sparse", rho.log_lambda_sparse)?;
    Ok(out.unbind())
}

fn sae_manifold_fit_inner<'py>(
    py: Python<'py>,
    z_view: ArrayView2<'_, f64>,
    atom_basis: &[String],
    atom_dim: Vec<usize>,
    atom_centers: &[Option<Array2<f64>>],
    basis_values: ArrayView3<'_, f64>,
    basis_jacobian: ArrayView4<'_, f64>,
    basis_sizes: Vec<usize>,
    decoder_coefficients: ArrayView3<'_, f64>,
    smooth_penalties: ArrayView3<'_, f64>,
    initial_logits: ArrayView2<'_, f64>,
    initial_coords: ArrayView3<'_, f64>,
    alpha: f64,
    tau: f64,
    learnable_alpha: bool,
    assignment_kind: String,
    sparsity_strength: f64,
    smoothness: f64,
    max_iter: usize,
    learning_rate: f64,
    ridge_ext_coord: f64,
    ridge_beta: f64,
    gumbel_schedule: Option<&Bound<'py, PyDict>>,
    analytic_penalties: Option<String>,
    top_k: Option<usize>,
    jumprelu_threshold: f64,
    native_ard_enabled: bool,
    seed_refine_routing: bool,
    seed_refine_random_state: u64,
    // WP-D output-Fisher shard (#980). Magic-by-default: the *presence* of
    // `fisher_u` activates `RowMetric::OutputFisher` — there is no flag. `fisher_u`
    // is `(n_obs, p_out, rank)` row-major (`U[n, i, k]`), exactly the harvest
    // shard's `U`; `fisher_mass_residual` is the per-row truncation diagnostic
    // `trace(G_n) − Σ_{k≤r} λ_k` that rides into the report so a too-small rank is
    // visible, not silent. Absent ⇒ the bit-identical Euclidean / isotropic path.
    fisher_u: Option<ArrayView3<'_, f64>>,
    fisher_mass_residual: Option<ArrayView1<'_, f64>>,
    // Harvest provenance (#980): which output-Fisher `RowMetric` to install when
    // `fisher_u` is present — same-position `"output_fisher"` (default) or
    // forward-looking `"output_fisher_downstream"`. Gauge/lens consume either
    // unchanged.
    fisher_provenance: Option<&str>,
    // Per-row design-honesty reconstruction weights (#977). When present, the
    // length-`n_obs` `√w` reweighting installed via `set_row_loss_weights`
    // scales every per-row reconstruction loss before the inner joint fit and
    // outer ρ selection. Uniform / absent ⇒ the bit-identical unweighted path.
    row_loss_weights: Option<ArrayView1<'_, f64>>,
    // Per-fit config. `Some(x)` pins this term's separation-barrier strength /
    // IBP-α to `x` via `SaeManifoldTerm::set_fit_config`; `None` selects the
    // canonical data-derived or mode default.
    separation_barrier_strength_override: Option<f64>,
    ibp_alpha_override: Option<f64>,
    promote_from_residual: bool,
    run_structure_search: bool,
    run_outer_rho_search: bool,
    data_row_reseed: bool,
) -> PyResult<Py<PyDict>> {
    let analytic_penalties: Option<serde_json::Value> = match analytic_penalties {
        Some(s) => Some(serde_json::from_str(&s).map_err(serde_json_error_to_pyerr)?),
        None => None,
    };
    let (n_obs, p_out) = z_view.dim();
    let k_atoms = atom_dim.len();
    // Preserve the seed dictionary size for variable-K payload plan recovery.
    let seed_k_atoms = k_atoms;

    // Registry descriptor parsing remains boundary marshalling because the JSON
    // descriptor builder lives above gam-sae. Every seed decision after this
    // point is owned by the typed gam-sae entry.
    let total_basis: usize = basis_sizes.iter().copied().sum();
    let mut latent_blocks = serde_json::Map::new();
    latent_blocks.insert(
        "t".into(),
        serde_json::json!({"name": "t", "n": n_obs, "d": atom_dim.iter().copied().max().unwrap_or(1)}),
    );
    latent_blocks.insert(
        "beta".into(),
        serde_json::json!({"name": "beta", "n": p_out, "d": total_basis}),
    );
    let latent_payload = serde_json::Value::Object(latent_blocks);
    let registry = build_analytic_penalty_registry_from_json(
        Some(&latent_payload),
        analytic_penalties.as_ref(),
    )
    .map_err(py_value_error)?;

    let assignment = SaeFitAssignmentKind::from_tag(&assignment_kind).map_err(py_value_error)?;
    let temperature_schedule =
        gumbel_temperature_schedule_from_pydict(gumbel_schedule).map_err(py_value_error)?;
    let fisher_metric = match fisher_u {
        Some(factors) => Some(
            SaeFisherRowMetricRequest::from_tag(
                factors,
                n_obs,
                p_out,
                fisher_provenance,
                fisher_mass_residual.as_ref().map(|mass| mass.view()),
            )
            .map_err(py_value_error)?,
        ),
        None => None,
    };
    let seed = build_sae_fit_seed(SaeFitSeedRequest {
        target: z_view,
        atom_basis,
        atom_dim: &atom_dim,
        atom_centers,
        basis_values,
        basis_jacobian,
        basis_sizes: &basis_sizes,
        decoder_coefficients,
        smooth_penalties,
        initial_logits,
        initial_coords,
        alpha,
        tau,
        learnable_alpha,
        assignment_kind: assignment,
        sparsity_strength,
        smoothness,
        max_iter,
        learning_rate,
        ridge_ext_coord,
        ridge_beta,
        top_k,
        threshold: jumprelu_threshold,
        native_ard_enabled,
        seed_refine_routing,
        seed_refine_random_state,
        data_row_reseed,
        fit_config: gam::terms::sae::manifold::SaeFitConfig {
            separation_barrier_strength_override,
            ibp_alpha_override,
        },
        temperature_schedule,
        fisher_metric,
        row_loss_weights,
        registry: &registry,
    })
    .map_err(py_value_error)?;
    let SaeFitSeedReport {
        base_term,
        initial_rho: init_rho,
        isometry_pin_active,
        metric_provenance,
    } = seed;

    // The typed gam-sae seed entry above owns construction and validation. Every
    // fit, structured-residual pass, evidence-guarded structure move,
    // certificate, diagnostic, and top-k projection below belongs to gam-sae's
    // single typed orchestration entry (#2236).
    let cancel_flag = std::sync::Arc::new(std::sync::atomic::AtomicBool::new(false));
    let request = gam::terms::sae::manifold::SaeFitRequest {
        base_term,
        target: z_view.to_owned(),
        registry,
        initial_rho: init_rho,
        max_iter,
        learning_rate,
        ridge_ext_coord,
        ridge_beta,
        assignment_kind: assignment_kind.clone(),
        alpha,
        top_k,
        isometry_pin_active,
        metric_provenance,
        promote_from_residual,
        run_structure_search,
        run_outer_rho_search,
        align_min_from_rank: sae_promotion_align_min,
        cancel: Some(std::sync::Arc::clone(&cancel_flag)),
    };
    let report = run_sae_fit_interruptible(py, "gam-sae-fit", &cancel_flag, move || {
        gam::terms::sae::manifold::run_sae_manifold_fit(request)
    })?
    .map_err(|err| sae_fit_error_to_pyerr(py, err))?;
    let gam::terms::sae::manifold::SaeFitReport {
        term,
        rho,
        loss,
        post_topk_loss,
        assignments,
        fitted,
        active_mask,
        reconstruction_r2,
        outer_termination,
        shape_uncertainty,
        metric_provenance,
        structured_residual_diagnostics,
        trust_diagnostics,
        fit_diagnostics,
        structure_search_json,
        structure_certificate_json,
        top_k_will_project,
        pre_topk_assignments,
        pre_topk_fitted,
        reported_log_alpha,
    } = report;

    // #977 variable-K boundary: structure search may grow or shrink the seed
    // dictionary, so every per-atom payload vector is re-derived from the
    // returned term rather than the caller's now-stale seed metadata.
    let k_atoms = term.k_atoms();
    let atom_basis: Vec<String> = term
        .atoms
        .iter()
        .map(|atom| sae_atom_basis_kind_name(&atom.basis_kind))
        .collect();
    let atom_dim: Vec<usize> = term.atoms.iter().map(|atom| atom.latent_dim).collect();
    let log_ard_py = PyList::empty(py);
    for atom_log_ard in &rho.log_ard {
        log_ard_py.append(atom_log_ard.clone().into_pyarray(py))?;
    }
    let atoms_py = PyList::empty(py);
    for atom_idx in 0..k_atoms {
        let atom = &term.atoms[atom_idx];
        let atom_dict = PyDict::new(py);
        atom_dict.set_item(
            // #2135/#2237 — emit the PHYSICAL full-width decoder
            // `exp(s_k) Q B̃` (`M × p`). The rank-reduced frame `Q` and the
            // quotient-scale coordinate `s_k` are both fit-internal; OOS, steering,
            // reconstruction, and reload consumers rebuild the standard `M`-column
            // basis with zero log-amplitude, so this boundary must materialize both.
            "decoder_B",
            atom.physical_full_width_decoder().into_pyarray(py),
        )?;
        atom_dict.set_item("basis_kind", atom_basis[atom_idx].clone())?;
        atom_dict.set_item("basis_centers", py.None())?;
        atom_dict.set_item(
            "on_atom_coords_t",
            term.assignment.coords[atom_idx]
                .as_matrix()
                .into_pyarray(py),
        )?;
        // #2081 — the honest arc-length coordinate `u_arc = s(t_i)/L` alongside
        // the raw (gauge-arbitrary) `on_atom_coords_t`. Pulled from the
        // coordinate-fidelity certificate the fit diagnostics just built, so it
        // reflects the FINAL reported chart regardless of whether the mutating
        // canonicalization committed. `None` for atoms without a d=1 chart or a
        // degenerate chart (verdict `degenerate`) — a coordinate consumer must
        // then read the certificate `verdict` and refuse rather than substitute
        // the raw `t` as if it were an angle.
        match fit_diagnostics
            .coordinate_fidelity
            .get(atom_idx)
            .and_then(|entry| entry.as_ref())
            .and_then(|fid| fid.coords_u_arc.as_ref())
        {
            Some(u_arc) => {
                atom_dict.set_item("on_atom_coords_u_arc", u_arc.clone().into_pyarray(py))?
            }
            None => atom_dict.set_item("on_atom_coords_u_arc", py.None())?,
        }
        atom_dict.set_item(
            "assignments_z",
            assignments.column(atom_idx).to_owned().into_pyarray(py),
        )?;
        atom_dict.set_item("active_dim", atom_dim[atom_idx])?;
        // Posterior shape uncertainty for this atom: φ-scaled decoder
        // covariance Cov(β_k) and the closed-form ambient band (coords / mean /
        // per-channel sd) along the atom's on-atom coordinates.
        //
        // #977 variable-K: `shape_uncertainty` was assembled from the PRE-search
        // joint-Hessian Schur factor (indexed by the SEED dictionary), then
        // COMPLETED above (`complete_born_atom_shape_bands`) so every post-search
        // atom — born atoms included — carries a band: seed atoms keep their exact
        // joint-Hessian band, born atoms get the per-atom Laplace band from their
        // own fitted inner Hessian, and a genuinely-degenerate atom keeps an
        // honest NaN band. `decoder_covariance` is still present only for the
        // Schur-assembled atoms (the dense lift is omitted for born atoms and at
        // LLM-scale `p`); the python reader treats it as optional (`_opt_arr`). The
        // atom's decoder, coordinates, assignments, basis kind, and active dim (all
        // read from the fitted term above) are exact regardless.
        if let Some(unc) = shape_uncertainty.atoms.get(atom_idx) {
            // Omitted (not set) above the SAE_DECODER_COV_PAYLOAD_MAX_ENTRIES
            // budget — the python reader treats the key as optional and the band
            // quantities below remain exact.
            if let Some(cov) = &unc.decoder_covariance {
                // #2135 — the emitted decoder is the FULL-width `M × p` block, so
                // its covariance must live in the same `M`-frame. For a #1117
                // rank-reduced atom the assembled covariance is the reduced
                // `(r·p)²`; lift it by the `(Q ⊗ I_p)` congruence to the exact
                // `(M·p)²` posterior of `Q B̃`. Identity clone for un-reduced atoms.
                let cov_full = atom
                    .lift_reduced_decoder_covariance(cov, p_out)
                    .map_err(py_value_error)?;
                atom_dict.set_item("decoder_covariance", cov_full.into_pyarray(py))?;
            }
            atom_dict.set_item(
                "shape_band_coords",
                unc.band_coords.clone().into_pyarray(py),
            )?;
            atom_dict.set_item("shape_band_mean", unc.band_mean.clone().into_pyarray(py))?;
            atom_dict.set_item("shape_band_sd", unc.band_sd.clone().into_pyarray(py))?;
        }
        atoms_py.append(atom_dict)?;
    }

    let out = PyDict::new(py);
    out.set_item("atoms", atoms_py)?;
    out.set_item("assignments_z", assignments.into_pyarray(py))?;
    out.set_item("logits", term.assignment.logits.clone().into_pyarray(py))?;
    out.set_item("atom_active_mask", active_mask)?;
    out.set_item("fitted", fitted.into_pyarray(py))?;
    out.set_item("reconstruction_r2", reconstruction_r2)?;
    // #2235 — outer-search accounting for the CONVERGED fit (the only kind
    // that exists: non-convergence raises a typed error before a fit is
    // minted — the forcing-function contract, see SPEC).
    {
        let termination = PyDict::new(py);
        // #2235/#2241 — which certificate concluded the outer search; the wire
        // names are owned by the Rust enums (SaeOuterVerdict/OuterConvergedVia
        // as_str) — the binding marshals, it does not map.
        termination.set_item("verdict", outer_termination.verdict.as_str())?;
        termination.set_item("evals", outer_termination.evals)?;
        termination.set_item(
            "evals_since_improvement",
            outer_termination.evals_since_improvement,
        )?;
        termination.set_item("wall_seconds", outer_termination.wall.as_secs_f64())?;
        out.set_item("termination", termination)?;
    }
    // #1231 / #1232 — in-sample fit: the score is the negative penalized loss,
    // surfaced honestly as `penalized_loss_score` (NOT `reml_score`) with a
    // breakdown. The top-level payload describes ONE coherent model: when a hard
    // top-k gate is applied the top-level `assignments`/`fitted`/`diagnostics`
    // AND this score all describe the PROJECTED model (the score's data-fit is
    // recomputed on the projected reconstruction; the decoder/ρ penalties are
    // unchanged by the gate). The unprojected smooth-optimization model — its
    // assignments, fitted, and score — is carried in full under `pre_topk`, with
    // a `top_k_projection` descriptor naming the split so no reader ever conflates
    // the two layers.
    let top_level_loss = post_topk_loss.as_ref().unwrap_or(&loss);
    sae_set_penalized_loss_items(&out, top_level_loss, "penalized_loss_score")?;
    if top_k_will_project {
        out.set_item("top_k_projection_applied", true)?;
        // Self-describing split descriptor (#1232): every top-level model field
        // describes the post-top-k projected model; the pre-projection model is
        // under `pre_topk`.
        let descriptor = PyDict::new(py);
        descriptor.set_item("applied", true)?;
        if let Some(k_top) = top_k {
            descriptor.set_item("top_k", k_top)?;
        }
        descriptor.set_item("top_level_model", "post_topk")?;
        descriptor.set_item("pre_projection_model", "pre_topk")?;
        out.set_item("top_k_projection", descriptor)?;
        if let (Some(pre_assignments), Some(pre_fitted)) = (pre_topk_assignments, pre_topk_fitted) {
            let pre_topk = PyDict::new(py);
            pre_topk.set_item("assignments_z", pre_assignments.into_pyarray(py))?;
            pre_topk.set_item("fitted", pre_fitted.into_pyarray(py))?;
            // The smooth-model score (unprojected data-fit + the same penalties).
            sae_set_penalized_loss_items(&pre_topk, &loss, "penalized_loss_score")?;
            out.set_item("pre_topk", pre_topk)?;
        }
    }
    out.set_item("log_alpha", reported_log_alpha)?;
    // Clone, do NOT move the field out: `rho` is still owned and is borrowed
    // again below for the post-fit amortized-encoder diagnostic
    // (`term.amortized_encoder_consistency(.., &rho)`). Moving the non-`Copy`
    // `log_lambda_smooth: Vec<f64>` out here would partially move `rho` and make
    // that later `&rho` borrow a borrow-after-move (E0382), which broke the
    // gam-pyffi build (#1559). The predict path (`sae_manifold_predict_oos`) can
    // move the field by value because its `rho` dies immediately after; here it
    // lives on, so we clone the K-length vector once at result emission — a
    // negligible allocation that matches every other still-live-owner field
    // emission in this file (`lambdas.clone()`, `class_levels.clone()`, …).
    out.set_item("log_lambda_smooth", rho.log_lambda_smooth.clone())?;
    // #2132 — the terminal REML-selected sparsity strength, alongside the
    // per-atom `log_lambda_smooth` / `log_ard` above. The OOS fixed-decoder
    // encode (`sae_manifold_predict_oos`) must optimize the SAME penalized
    // objective the training state converged under; without this key Python
    // could only feed the INITIAL `sparsity_strength` back in, so the OOS
    // Newton solve descended a different objective and walked the warm-started
    // trained optimum away from itself (the cold re-encode collapse).
    out.set_item("log_lambda_sparse", rho.log_lambda_sparse)?;
    out.set_item("log_ard", log_ard_py)?;
    out.set_item("assignment_prior", assignment_kind)?;
    out.set_item(
        "solver_plan",
        sae_streaming_plan_to_pydict(py, term.streaming_plan())?,
    )?;
    out.set_item(
        "diagnostics",
        sae_trust_diagnostics_dict(py, &trust_diagnostics)?,
    )?;
    // Gaussian reconstruction scale φ̂ used to scale every per-atom decoder
    // covariance (Cov(β_k) = φ̂·S_β⁻¹[block]).
    out.set_item("dispersion", shape_uncertainty.dispersion)?;
    // Provenance of the per-row inner product the fit installed (#980). Object 4
    // reads this to certify which metric the gauge pulled back through:
    // "Euclidean" (no shard, bit-identical isotropic path) or "OutputFisher"
    // (a WP-D shard was supplied and `RowMetric::OutputFisher` was installed).
    out.set_item("metric_provenance", metric_provenance)?;
    out.set_item(
        "structured_residual_diagnostics",
        structured_residual_pass_diagnostics_dict(py, &structured_residual_diagnostics)?,
    )?;
    // Truncation diagnostic: per-row output-Fisher mass `trace(G_n) − Σ_{k≤r} λ_k`
    // that fell off the captured rank-r subspace. Surfaced so a too-small rank is
    // visible, not silent. Present only when a Fisher shard with a mass_residual
    // was supplied.
    if let Some(mr) = fisher_mass_residual {
        out.set_item("fisher_mass_residual", mr.to_owned().into_pyarray(py))?;
    }
    // Additive post-fit diagnostics (#980): the two-score per-atom lens and the
    // residual-gauge certificate. Both are read through the same single metric;
    // coupling is `None` (NaN array entries) under a Euclidean / no-harvest
    // provenance, and the gauge is then certified under Euclidean provenance.
    out.set_item(
        "atom_two_lens",
        sae_atom_two_lens_dict(py, &fit_diagnostics.atom_two_lens)?,
    )?;
    out.set_item(
        "residual_gauge",
        sae_residual_gauge_dict(py, &fit_diagnostics.residual_gauge)?,
    )?;
    // #1026 — the load-bearing curved-vs-linear hybrid-split verdict: per d=1
    // atom, whether its fitted curved image beats its straight (linear
    // special-case) secant on the common Laplace evidence scale. Present
    // whenever the post-fit pass adjudicated at least one eligible atom; absent
    // for dictionaries with no eligible d=1 atom (nothing to split).
    if let Some(report) = term.hybrid_split_report() {
        out.set_item("hybrid_split", sae_hybrid_split_dict(py, report)?)?;
    }
    // #1154 — post-fit amortized-encoder diagnostics. The outer ρ-cascade ranks
    // the pure REML criterion; encoder consistency is deliberately excluded from
    // fitting/ranking because it has no matching analytic ρ gradient. Here we
    // measure, at the settled dictionary, how faithfully the cheap initializer
    // approximates the exact encode-by-inner-solve map:
    //   * `recon_consistency` — mean per-element squared gap between the amortized
    //     reconstruction and the exact encode-by-inner-solve reconstruction (0 ⇒
    //     the IFT predictor is an exact first-order model of the encode map);
    //   * `unconverged_fraction` — share of shared-residual row solves that did
    //     not meet the first-order stationarity tolerance.
    // Best-effort: a degenerate atlas (no usable charts) yields no report rather
    // than aborting the fit payload; this diagnostic never changes the fit.
    if let Ok(consistency) = term.amortized_encoder_consistency(z_view.view(), &rho) {
        let cotrain = PyDict::new(py);
        cotrain.set_item("recon_consistency", consistency.recon_consistency)?;
        cotrain.set_item("unconverged_fraction", consistency.unconverged_fraction)?;
        cotrain.set_item("n_unconverged", consistency.n_unconverged)?;
        cotrain.set_item("n_encodes", consistency.n_encodes)?;
        out.set_item("cotrain", cotrain)?;
    }
    if let Some(report) = &fit_diagnostics.incoherence_report {
        out.set_item("curvature_report", sae_curvature_report_dict(py, report)?)?;
        out.set_item(
            "incoherence_report",
            sae_incoherence_report_dict(py, report)?,
        )?;
    }
    // #1097 / #1103 — per-atom Riesz-debiased smooth-functional inference
    // (peak-vs-mode contrast, on-fit decoder-variation norm, data-averaged value;
    // each a plug-in + one-step-debiased POINT estimate with the removed penalty
    // bias — #1115 dropped the coverage-claiming SE/CI) and an any-n-valid
    // split-LRT e-value for smooth significance. One entry per fitted atom.
    out.set_item(
        "atom_inference",
        sae_atom_inference_list(py, &fit_diagnostics.atom_inference)?,
    )?;
    // #2081 — per-atom chart coordinate-fidelity certificate: the circular
    // coordinate-uniformity statistic (Watson U² + closed-form p-value) against
    // the atom's invariant measure, and the arc-length (unit-speed) defect of the
    // chart parameterization. Always present (one entry per atom); reconstruction
    // EV does not certify coordinate quality.
    out.set_item(
        "coordinate_fidelity",
        sae_coordinate_fidelity_dict(py, &fit_diagnostics.coordinate_fidelity)?,
    )?;
    out.set_item(
        "topology_persistence",
        topology_persistence_ffi::sae_topology_persistence_dict(
            py,
            &fit_diagnostics.topology_persistence,
        )?,
    )?;
    // #16 — ONE coherent certificate ledger. Every certificate this fit produced
    // implements the shared claim+evidence+conservative-verdict contract
    // ([`gam::inference::certificates::Certificate`]); the ledger folds them into
    // a single inspectable block keyed by claim id, with a top-level conservative
    // `overall` roll-up (the weakest member). The detailed feature reports above
    // remain first-class because they carry per-atom coordinates, generators,
    // and measurements that the generic ledger intentionally does not encode;
    // the ledger is the canonical cross-certificate verdict surface. Verdicts
    // cannot read stronger than their evidence: an absent or below-margin
    // certificate is `unavailable` / `insufficient`, never a silent pass.
    {
        use gam::inference::certificates::CertificateLedger;
        let mut ledger = CertificateLedger::new();
        ledger.record(&fit_diagnostics.residual_gauge);
        let coordinate_fidelity_certificate =
            gam::terms::sae::manifold::CoordinateFidelityCertificate::new(
                &fit_diagnostics.coordinate_fidelity,
            );
        ledger.record(&coordinate_fidelity_certificate);
        let topology_persistence_certificate =
            gam::terms::sae::manifold::TopologyPersistenceCertificate::new(
                &fit_diagnostics.topology_persistence,
            );
        ledger.record(&topology_persistence_certificate);
        if let Some(report) = &fit_diagnostics.incoherence_report {
            ledger.record(report);
        }
        out.set_item("certificates", certificate_ledger_dict(py, &ledger)?)?;
    }
    // #977 — VARIABLE-K `atom_plans`, emitted from the POST-search dictionary so
    // its length matches the returned `atoms` exactly (the python `from_payload`
    // boundary zips `payload["atoms"]` with `plans[atom_idx]`, so a length
    // mismatch from a grown K would IndexError). Each plan is re-derived from the
    // fitted atom: `kind` + `latent_dim` straight off the atom, `basis_size` from
    // its Φ width, `n_harmonics` recovered from the basis size for the harmonic
    // families, and `duchon_centers` inherited from the seed template (a
    // structure-search-born atom clones atom 0's basis, so it carries the seed
    // template's centers). This REPLACES the old post-hoc seed-K plan list that
    // `sae_manifold_fit_minimal` attached after the fact — that list was pinned
    // at the input K and was exactly the plumbing constraint the #997 comment
    // cited for disabling births; emitting the plans here makes variable K
    // first-class with no truncation or padding.
    let atom_plans_py = PyList::empty(py);
    for (atom_idx, atom) in term.atoms.iter().enumerate() {
        let entry = PyDict::new(py);
        let kind = &atom.basis_kind;
        let latent_dim = atom.latent_dim;
        // #2135 — report the FULL inner-basis width `M` (== `basis_size` for an
        // un-reduced atom, `Q.nrows()` for a #1117 rank-reduced one). This MUST
        // match the re-expanded `M × p` decoder emitted above and the standard
        // inner basis the OOS / reconstruct paths rebuild; emitting the reduced
        // `r` here (with the full decoder) would desync the `n_harmonics` /
        // `basis_size` round-trip from the decoder width.
        let basis_size = atom.full_basis_size();
        let kind_name = sae_atom_basis_kind_name(kind);
        // Recover the per-(axis-)harmonic order from the fitted basis width for
        // the harmonic families; 0 for the non-harmonic kinds (the python reader
        // only consults `n_harmonics` for periodic/torus atoms). The harmonic
        // order is floored at 1 for the harmonic kinds: a periodic/torus atom is
        // by construction at least one harmonic (basis width 2H+1 with H>=1), and
        // a degenerate collapse to a constant-only width (basis_size <= 2) would
        // otherwise re-emit n_harmonics=0 — which the round-trip rebuild
        // (`sae_build_periodic_atom`, `sae_build_torus_atom`) rejects as a
        // non-positive harmonic order, breaking K>=4 reload/OOS reconstruct.
        let n_harmonics = match kind {
            SaeAtomBasisKind::Periodic => (basis_size.saturating_sub(1) / 2).max(1),
            SaeAtomBasisKind::Torus => {
                // basis_size = (2H+1)^latent_dim; recover the per-axis 2H+1, then H.
                match sae_torus_axis_basis_size(basis_size, latent_dim.max(1)) {
                    Ok(axis_m) => (axis_m.saturating_sub(1) / 2).max(1),
                    Err(_) => 1,
                }
            }
            _ => 0,
        };
        entry.set_item("kind", kind_name)?;
        entry.set_item("latent_dim", latent_dim)?;
        entry.set_item("n_harmonics", n_harmonics)?;
        entry.set_item("basis_size", basis_size)?;
        // Born atoms (index ≥ seed K) inherit atom 0's basis template, so they
        // carry the seed template's Duchon centers (None for a periodic template).
        let center_src = atom_idx.min(seed_k_atoms.saturating_sub(1));
        match atom_centers.get(center_src).and_then(|c| c.as_ref()) {
            Some(centers) => {
                entry.set_item("duchon_centers", centers.clone().into_pyarray(py))?;
            }
            None => {
                // #357 — a structure-search BIRTH races a heterogeneous topology
                // (`race_birth_topology`): a circle seed can grow a EuclideanPatch
                // (monomial-patch line) / Linear / Poincaré atom. When the SEED
                // template carried no Duchon centers (periodic/sphere/torus seed),
                // that born monomial atom would inherit `None` here and then the OOS
                // round-trip (`sae_manifold_predict_oos`) would reject it with "atom
                // N (Duchon-like) needs duchon_centers", breaking `converged_latents`,
                // `project`, and `reconstruct` on any held-out matrix.
                //
                // A monomial-patch atom (EuclideanPatch / Linear / Poincaré) is
                // CENTERLESS for the OOS rebuild: that path reads only
                // `centers.ncols()` (the build dimension, equal to `latent_dim`) to
                // recover the monomial degree from the trained decoder width — the
                // center COORDINATES are never consulted (cf.
                // `build_sae_basis_evaluators`, where only the genuine `Duchon` kernel
                // reads center content). But the SAME `duchon_centers` field also
                // feeds the python `_functional_basis_params` diagnostic, which builds
                // a Duchon KERNEL design (`duchon_basis_with_jet`) that needs enough
                // real center ROWS to leave a non-empty radial block — a 1-row
                // placeholder degenerates it. So derive the centers from the atom's
                // OWN converged on-atom coordinates (the honest analogue of the seed
                // path, which subsamples the PCA-seeded coords): correct `ncols` for
                // the OOS rebuild AND real rows for the kernel diagnostic. Births
                // never produce a genuine Duchon kernel atom, so a Duchon-kind atom
                // here keeps `None` and its missing-centers error still surfaces.
                let needs_centerless_patch = matches!(
                    kind,
                    SaeAtomBasisKind::EuclideanPatch
                        | SaeAtomBasisKind::Linear
                        | SaeAtomBasisKind::Poincare
                );
                if needs_centerless_patch {
                    let coords = term.assignment.coords[atom_idx].as_matrix();
                    let n_rows = coords.nrows();
                    let dim = latent_dim.max(1);
                    // Mirror the seed-path center budget (`sae_build_atom_plans`):
                    // a handful of rows is plenty for the monomial patch (ncols is
                    // what the OOS rebuild reads) and gives the kernel diagnostic a
                    // non-degenerate radial block. Deterministic subsample keyed by
                    // the atom index so reloads are reproducible.
                    let center_floor = (dim + 2).max(8);
                    let n_centers = n_rows.min(center_floor.max(8)).max(dim + 1).min(n_rows);
                    // No `random_state` in this entry point's scope (the precomputed-
                    // basis fit takes its seed jitter elsewhere); key the deterministic
                    // subsample purely on the atom index so a reload is reproducible.
                    let idx = sae_pick_duchon_center_indices(n_rows, n_centers, atom_idx as u64);
                    let mut centers = Array2::<f64>::zeros((idx.len().max(1), dim));
                    for (out_row, src_row) in idx.iter().copied().enumerate() {
                        for col in 0..dim.min(coords.ncols()) {
                            centers[[out_row, col]] = coords[[src_row, col]];
                        }
                    }
                    entry.set_item("duchon_centers", centers.into_pyarray(py))?;
                } else {
                    entry.set_item("duchon_centers", py.None())?;
                }
            }
        }
        atom_plans_py.append(entry)?;
    }
    out.set_item("atom_plans", atom_plans_py)?;

    // Contract keys the python `ManifoldSAE.from_payload` boundary reads
    // unconditionally (tightened in 23db2c80a, which rejected stale payload
    // shapes python-side without adding the producer side): the fitted atom
    // count, and whether OOS encode projects each row onto its single
    // top-mass atom (true exactly for `top_k == 1`, mirroring the fit-time
    // assignment-support projection the payload's `fitted` was computed with).
    out.set_item("chosen_k", k_atoms)?;
    out.set_item("oos_projection_top1", top_k == Some(1))?;
    // #997 — the evidence-guarded structure-search honesty surface: the per-round
    // SearchLedger (every harvested move in canonical order with its e-gate
    // verdict) plus the joint fit's collapse events, serialized as JSON. Present
    // whenever the structure search ran (it runs on every fit); the value is the
    // certificate of which dictionary moves the held-out data does and does not
    // support — an all-contested ledger is the common, conservative outcome.
    if let Some(json) = structure_search_json {
        out.set_item("structure_search", json)?;
    }
    // Anytime-valid structure certificate (#1058 / #984): the e-BH certificate
    // over the ledger's per-claim e-processes at the search FDR level α = 0.05.
    // Serialized onto the payload (alongside the raw `structure_search` rounds)
    // so the post-fit `ManifoldSAE.structure_certificate()` can surface which
    // discovered atoms / bindings / geometries the held-out data confirmed vs
    // left contested — without re-running any fitting. Valid at this (or any)
    // data-dependent stopping time because each claim is an e-process.
    out.set_item("structure_certificate", structure_certificate_json)?;
    Ok(out.unbind())
}

/// Build the result-dict entry for the honest SAE trust diagnostics (#1005).
fn sae_trust_diagnostics_dict<'py>(
    py: Python<'py>,
    report: &gam::terms::sae::manifold::SaeTrustDiagnostics,
) -> PyResult<Bound<'py, PyDict>> {
    let d = PyDict::new(py);
    let atoms = PyList::empty(py);
    for atom in &report.atoms {
        let atom_dict = PyDict::new(py);
        atom_dict.set_item("trust_score", atom.trust_score)?;
        atom_dict.set_item("sigma_min_tangent", atom.sigma_min_tangent)?;
        atom_dict.set_item("sigma_max_tangent", atom.sigma_max_tangent)?;
        atom_dict.set_item("tangent_condition_score", atom.tangent_condition_score)?;
        atom_dict.set_item("coverage", atom.coverage)?;
        atom_dict.set_item("activation_frequency", atom.activation_frequency)?;
        atom_dict.set_item("untyped", atom.untyped)?;
        atom_dict.set_item("active_token_count", atom.active_token_count)?;
        atoms.append(atom_dict)?;
    }
    d.set_item(
        "atom_trust",
        Array1::from_vec(report.atom_trust.clone()).into_pyarray(py),
    )?;
    d.set_item("atoms", atoms)?;
    Ok(d)
}

/// Build the result-dict entry for the two-score per-atom lens
/// ([`gam::inference::atom_lens::AtomTwoLensReport`]). Per-atom presence /
/// coupling / discrepancy arrays plus the coupling provenance string. Coupling /
/// coupling_normalized / discrepancy are `NaN` for atoms whose behavioral axis is
/// unavailable (Euclidean / no-harvest provenance), mirroring the Rust `None`.
fn sae_atom_two_lens_dict<'py>(
    py: Python<'py>,
    report: &gam::inference::atom_lens::AtomTwoLensReport,
) -> PyResult<Bound<'py, PyDict>> {
    let d = PyDict::new(py);
    let names = PyList::empty(py);
    let mut presence = Vec::with_capacity(report.atoms.len());
    let mut presence_norm = Vec::with_capacity(report.atoms.len());
    let mut coupling = Vec::with_capacity(report.atoms.len());
    let mut coupling_norm = Vec::with_capacity(report.atoms.len());
    let mut discrepancy = Vec::with_capacity(report.atoms.len());
    for atom in &report.atoms {
        names.append(atom.name.clone())?;
        presence.push(atom.presence);
        presence_norm.push(atom.presence_normalized);
        coupling.push(atom.coupling.unwrap_or(f64::NAN));
        coupling_norm.push(atom.coupling_normalized.unwrap_or(f64::NAN));
        discrepancy.push(atom.discrepancy.unwrap_or(f64::NAN));
    }
    d.set_item("names", names)?;
    d.set_item("presence", Array1::from_vec(presence).into_pyarray(py))?;
    d.set_item(
        "presence_normalized",
        Array1::from_vec(presence_norm).into_pyarray(py),
    )?;
    d.set_item("coupling", Array1::from_vec(coupling).into_pyarray(py))?;
    d.set_item(
        "coupling_normalized",
        Array1::from_vec(coupling_norm).into_pyarray(py),
    )?;
    d.set_item(
        "discrepancy",
        Array1::from_vec(discrepancy).into_pyarray(py),
    )?;
    d.set_item("coupling_available", report.coupling_available())?;
    d.set_item(
        "coupling_provenance",
        report.coupling_provenance.map(|p| format!("{p:?}")),
    )?;
    Ok(d)
}

/// Render a single [`gam::inference::certificates::EvidenceValue`] to a Python
/// object (scalar / int / bool / str / list-of-float).
fn certificate_evidence_value<'py>(
    py: Python<'py>,
    value: &gam::inference::certificates::EvidenceValue,
) -> PyResult<pyo3::Bound<'py, pyo3::PyAny>> {
    use gam::inference::certificates::EvidenceValue;
    Ok(match value {
        EvidenceValue::Scalar(v) => (*v).into_bound_py_any(py)?,
        EvidenceValue::Integer(v) => (*v).into_bound_py_any(py)?,
        EvidenceValue::Flag(v) => (*v).into_bound_py_any(py)?,
        EvidenceValue::Text(v) => v.as_str().into_bound_py_any(py)?,
        EvidenceValue::Vector(v) => v.clone().into_bound_py_any(py)?,
    })
}

/// Render a [`gam::inference::certificates::CertificateLedger`] as ONE coherent
/// `certificates` payload block (task #16): a dict keyed by claim id, each entry
/// `{claim, verdict, certified, evidence{...}}`, plus a top-level `overall`
/// conservative roll-up verdict (the weakest member). This is the program's
/// single inspectable cross-certificate verdict artifact. Detailed feature
/// reports remain separate because their rich per-atom payloads are not aliases
/// of this generic claim/evidence schema.
fn certificate_ledger_dict<'py>(
    py: Python<'py>,
    ledger: &gam::inference::certificates::CertificateLedger,
) -> PyResult<Bound<'py, PyDict>> {
    let out = PyDict::new(py);
    // The conservative roll-up: weakest verdict across every recorded claim
    // (Unavailable on an empty ledger). One number answering "did everything
    // this fit could certify, certify?" — never stronger than its weakest claim.
    let overall = ledger.overall();
    out.set_item("overall", overall.label())?;
    out.set_item("overall_certified", overall.is_certified())?;
    let claims = PyDict::new(py);
    for entry in ledger.entries() {
        let claim_dict = PyDict::new(py);
        claim_dict.set_item("claim", &entry.claim.statement)?;
        claim_dict.set_item("verdict", entry.verdict.label())?;
        claim_dict.set_item("certified", entry.verdict.is_certified())?;
        let evidence = PyDict::new(py);
        for (key, value) in &entry.evidence {
            evidence.set_item(*key, certificate_evidence_value(py, value)?)?;
        }
        claim_dict.set_item("evidence", evidence)?;
        claims.set_item(entry.claim.id, claim_dict)?;
    }
    out.set_item("claims", claims)?;
    Ok(out)
}

/// Build the result-dict entry for the #1026 hybrid curved-vs-linear split
/// ([`gam::terms::sae::hybrid_split::SaeHybridSplitReport`]). One per-atom
/// verdict (curved vs linear, the evidence margin, the fitted turning Θ that
/// decided it) plus the dictionary-level aggregates the EV-vs-Θ frontier reports
/// against. Honest by construction: a curved atom that does not beat its straight
/// secant on evidence is reported as collapsed to the linear tail.
fn sae_hybrid_split_dict<'py>(
    py: Python<'py>,
    report: &gam::terms::sae::hybrid_split::SaeHybridSplitReport,
) -> PyResult<Bound<'py, PyDict>> {
    use gam::solver::evidence::HybridAtomParam;
    let d = PyDict::new(py);
    d.set_item("curved_atom_count", report.selection.curved_atom_count)?;
    d.set_item("linear_atom_count", report.selection.linear_atom_count())?;
    d.set_item(
        "total_negative_log_evidence",
        report.selection.total_negative_log_evidence,
    )?;
    d.set_item("total_parameters", report.selection.total_parameters)?;
    d.set_item("is_pure_linear", report.selection.is_pure_linear())?;
    d.set_item("is_pure_curved", report.selection.is_pure_curved())?;
    d.set_item(
        "envelope_note",
        "chart_efficiency_eta = EV_curved / EV_lin(top-M). Eta near 1 means \
         the K=1 chart is at the top-M \
         linear information ceiling; much below 1 means convergence headroom. \
         Curvature-is-identifiability is a claim about sparse sums of charts \
         over strata, so K=1 on the raw stream is the wrong test; compare \
         curved vs linear at matched parameter count within strata, as in the \
         wall-closure experiment.",
    )?;
    let atoms = PyList::empty(py);
    for verdict in &report.verdicts {
        let a = PyDict::new(py);
        a.set_item("atom", &verdict.atom_name)?;
        a.set_item("kept_curved", verdict.kept_curved)?;
        let param = match verdict.choice.param {
            HybridAtomParam::Linear => "linear".to_string(),
            HybridAtomParam::Curved { latent_dim } => format!("curved(d={latent_dim})"),
        };
        a.set_item("parameterization", param)?;
        a.set_item(
            "negative_log_evidence",
            verdict.choice.negative_log_evidence,
        )?;
        a.set_item("num_parameters", verdict.choice.num_parameters)?;
        a.set_item(
            "curved_evidence_margin",
            verdict.choice.curved_evidence_margin,
        )?;
        match verdict.choice.curved_turning {
            Some(theta) => a.set_item("fitted_turning", theta)?,
            None => a.set_item("fitted_turning", py.None())?,
        }
        // Per-atom IN-SAMPLE leave-one-atom-out explained-variance contribution
        // `ΔEV_k = EV(full) − EV(full∖{k})` — the EV axis of the #1026 (Θ, ΔEV)
        // frontier. Computed during fit on the TRAINING reconstruction target
        // (`per_atom_loao_explained_variance(target, rho)` in
        // `compute_hybrid_split_report`), so it is an in-sample LOAO ΔEV, NOT a
        // held-out generalization number (#1226).
        match verdict.train_loao_delta_ev {
            Some(dev) => a.set_item("train_loao_delta_ev", dev)?,
            None => a.set_item("train_loao_delta_ev", py.None())?,
        }
        match verdict.curved_ev {
            Some(ev) => a.set_item("curved_ev", ev)?,
            None => a.set_item("curved_ev", py.None())?,
        }
        match verdict.topm_linear_ev {
            Some(ev) => a.set_item("topm_linear_ev", ev)?,
            None => a.set_item("topm_linear_ev", py.None())?,
        }
        match verdict.curved_vs_envelope_ratio {
            Some(ratio) => a.set_item("curved_vs_envelope_ratio", ratio)?,
            None => a.set_item("curved_vs_envelope_ratio", py.None())?,
        }
        match verdict.chart_efficiency_eta {
            Some(eta) => a.set_item("chart_efficiency_eta", eta)?,
            None => a.set_item("chart_efficiency_eta", py.None())?,
        }
        // #1228 — the fitted straight sub-model `b₀ + (t − t̄)·b₁` for a slot the
        // verdict collapsed to LINEAR. Serialized so a held-out reconstruction
        // (`sae_manifold_predict_oos`) can decode this slot by the SAME straight
        // image the training reconstruction used, instead of the original curved
        // decoder — otherwise train and OOS EV would score different models.
        // `None` for slots that kept the curved parameterization.
        match verdict.linear_image.as_ref() {
            Some(img) => {
                let li = PyDict::new(py);
                li.set_item("atom_idx", img.atom_idx)?;
                li.set_item("t_bar", img.t_bar)?;
                li.set_item("b0", img.b0.clone().into_pyarray(py))?;
                li.set_item("b1", img.b1.clone().into_pyarray(py))?;
                // #1777 — the collapse-rescue projection direction `v` (length p,
                // unit norm), present iff this is a collapse-rescued image. Serialized
                // so a held-out (OOS) term can recompute any row's coordinate as
                // `u_i = <y_i - Σ_{j≠k} f_j(x_i), v>` — the SAME math the train
                // per-row codes were built with — making a rescued atom reconstruct
                // identically train vs held-out. `None` for the ordinary straight
                // image (decoded at the atom's own coordinate).
                match img.v.as_ref() {
                    Some(v) => li.set_item("v", v.clone().into_pyarray(py))?,
                    None => li.set_item("v", py.None())?,
                }
                a.set_item("linear_image", li)?;
            }
            None => a.set_item("linear_image", py.None())?,
        }
        atoms.append(a)?;
    }
    d.set_item("atoms", atoms)?;
    Ok(d)
}

/// Serialize one SAE atom decoder-functional POINT summary (#1097, narrowed
/// under #1115): the plug-in value, the one-step penalty-debiased value, and the
/// removed penalty bias. Deliberately carries NO standard error and NO
/// confidence interval — the conditional-on-generated-regressors variance
/// channel is unmodelled, so any SE would under-cover. Use the atom's
/// `smooth_significance` (any-n-valid e-value) for an honest structure test.
fn sae_atom_functional_estimate_dict<'py>(
    py: Python<'py>,
    estimate: &gam::terms::sae::identifiability::AtomFunctionalEstimate,
) -> PyResult<Bound<'py, PyDict>> {
    let d = PyDict::new(py);
    d.set_item("theta_plugin", estimate.theta_plugin)?;
    d.set_item("theta_onestep", estimate.theta_onestep)?;
    d.set_item("penalty_bias", estimate.penalty_bias)?;
    Ok(d)
}

/// Build the result-dict list of per-atom post-PIRLS inference reports (#1097
/// penalty-debiased functional point summaries + #1103 split-LRT smooth
/// significance). One entry per fitted atom; `functionals` /
/// `smooth_significance` are `None` (Python `None`) for atoms whose
/// inner-decoder smooth was not harvestable.
fn sae_atom_inference_list<'py>(
    py: Python<'py>,
    reports: &[gam::terms::sae::identifiability::AtomInferenceReport],
) -> PyResult<Bound<'py, PyList>> {
    let list = PyList::empty(py);
    for report in reports {
        let a = PyDict::new(py);
        a.set_item("atom_index", report.atom_index)?;
        a.set_item("atom_name", &report.atom_name)?;
        match &report.functionals {
            Some(f) => {
                let fd = PyDict::new(py);
                match &f.peak_contrast {
                    Some(r) => {
                        fd.set_item("peak_contrast", sae_atom_functional_estimate_dict(py, r)?)?
                    }
                    None => fd.set_item("peak_contrast", py.None())?,
                }
                match &f.decoder_variation_norm {
                    Some(r) => fd.set_item(
                        "decoder_variation_norm",
                        sae_atom_functional_estimate_dict(py, r)?,
                    )?,
                    None => fd.set_item("decoder_variation_norm", py.None())?,
                }
                match &f.average_value {
                    Some(r) => {
                        fd.set_item("average_value", sae_atom_functional_estimate_dict(py, r)?)?
                    }
                    None => fd.set_item("average_value", py.None())?,
                }
                a.set_item("functionals", fd)?;
            }
            None => a.set_item("functionals", py.None())?,
        }
        match &report.smooth_significance {
            Some(s) => {
                let sd = PyDict::new(py);
                // #1103: any-n-valid split-LRT e-value for "atom smooth is
                // non-constant" (null = constant). log E, with E_{H0}[E] ≤ 1.
                match s.log_e_nonconstant {
                    Some(log_e) => sd.set_item("log_e_nonconstant", log_e)?,
                    None => sd.set_item("log_e_nonconstant", py.None())?,
                }
                a.set_item("smooth_significance", sd)?;
            }
            None => a.set_item("smooth_significance", py.None())?,
        }
        list.append(a)?;
    }
    Ok(list)
}

/// Build the result-dict entry for the residual-gauge certificate
/// ([`gam::terms::sae::identifiability::ResidualGaugeReport`]). Group signature +
/// per-generator pinned/unpinned verdicts + the metric provenance the
/// certificate was computed in.
fn sae_residual_gauge_dict<'py>(
    py: Python<'py>,
    report: &gam::terms::sae::identifiability::ResidualGaugeReport,
) -> PyResult<Bound<'py, PyDict>> {
    let d = PyDict::new(py);
    d.set_item("group_signature", report.group_signature())?;
    d.set_item(
        "metric_provenance",
        format!("{:?}", report.metric_provenance),
    )?;
    d.set_item("pinning_rank", report.pinning_rank)?;
    d.set_item("residual_gauge_dim", report.residual_gauge_dim)?;
    d.set_item("diffeomorphism_unpinned", report.diffeomorphism_unpinned)?;
    d.set_item(
        "sym_f_trivial_under_output_fisher",
        report.sym_f_trivial_under_output_fisher,
    )?;
    d.set_item("summary", report.summary.clone())?;
    let generators = PyList::empty(py);
    for g in &report.generators {
        let gd = PyDict::new(py);
        gd.set_item("family", format!("{:?}", g.family))?;
        gd.set_item("description", g.description.clone())?;
        gd.set_item("unpinned", g.unpinned)?;
        gd.set_item("generator_norm", g.generator_norm)?;
        gd.set_item("pinned_energy_fraction", g.pinned_energy_fraction)?;
        gd.set_item("lowering_error_scale", g.lowering_error_scale)?;
        generators.append(gd)?;
    }
    d.set_item("generators", generators)?;
    Ok(d)
}

/// Build the per-atom SAE curvature report (#1099, rescoped under #1115). Each
/// `kappa_hat` is the fitted empirical second-fundamental-form sup-norm bound
/// already computed for the curved-dictionary certificate
/// (`CertificateInputs::per_atom_kappa_hat`, the one source of truth shared with
/// `dictionary_report`). It is a descriptive plug-in geometry summary, not an
/// estimand with a profiled criterion: a sup-norm curvature BOUND has no
/// confidence interval, and the delta-method SE that #1099 first shipped was
/// conditioned on the generated latent coordinates as if known (omitting the
/// generated-regressor channel) and under-covered, so #1115 removed it. The
/// schema therefore carries the point bound only — no SE/CI/flatness fields.
fn sae_curvature_report_dict<'py>(
    py: Python<'py>,
    report: &gam::terms::sae::manifold::CertificateInputs,
) -> PyResult<Bound<'py, PyDict>> {
    let d = PyDict::new(py);
    d.set_item(
        "note",
        "SAE per-atom kappa_hat is the fitted empirical second-fundamental-form \
         sup-norm bound (a descriptive plug-in geometry summary). It is not an \
         estimand with a confidence interval: a curvature bound has no profiled \
         criterion, so no SE/CI is reported.",
    )?;
    let atoms = PyList::empty(py);
    for (atom_idx, &kappa_hat) in report.per_atom_kappa_hat.iter().enumerate() {
        let a = PyDict::new(py);
        a.set_item("atom", atom_idx)?;
        a.set_item("kappa_hat", kappa_hat)?;
        atoms.append(a)?;
    }
    d.set_item("atoms", atoms)?;
    Ok(d)
}

/// #2081 — build the result-dict entry for the per-atom coordinate-fidelity
/// certificate: the circular coordinate-uniformity statistic (Watson U² +
/// closed-form asymptotic p-value) against the atom's invariant measure, and the
/// arc-length (unit-speed) defect of the chart parameterization. One per-atom
/// entry in atom order; atoms without a `d = 1` circle/interval chart carry a
/// null `topology`.
fn sae_coordinate_fidelity_dict<'py>(
    py: Python<'py>,
    fidelity: &[Option<gam::terms::sae::manifold::AtomCoordinateFidelity>],
) -> PyResult<Bound<'py, PyDict>> {
    let d = PyDict::new(py);
    d.set_item(
        "note",
        "SAE per-atom coordinate-fidelity certificate (#2081). `uniformity_statistic` \
         is Watson's U² of the fitted d=1 coordinate against the atom's uniform \
         invariant measure (rotation/reflection invariant); `uniformity_p_value` is its \
         closed-form asymptotic null p-value (small => coordinates flagged non-uniform). \
         `arclength_defect` is the speed coefficient-of-variation of the decoded curve on \
         a uniform latent grid (0 => arc-length / unit-speed chart; positive => the \
         parameterization squishes arc length, which reconstruction EV cannot see). \
         `verdict` is the certified reading rule: `arclength_honest` (read the raw \
         `on_atom_coords_t`), `recoverable_via_arclength` (read `coords_u_arc` instead — \
         the raw chart squishes arc length but the honest coordinate is recoverable), or \
         `degenerate` (the chart collapses; refuse — no faithful coordinate exists). \
         `certified` is `verdict != degenerate`. `coords_u_arc` is the honest arc-length \
         coordinate s(t_i)/L in [0,1) for every fitted row (a pure read, computed \
         regardless of whether the decoder-mutating canonicalization committed); \
         `raw_arclength_defect_{rms,max}` is the (circular) raw-vs-`u_arc` distance at the \
         data rows after best O(2) gauge alignment; `min_speed_over_mean` / \
         `max_speed_over_mean` / `log_speed_rms` summarize the decoder-speed profile. \
         Entries with null `topology` carry no d=1 chart.",
    )?;
    let atoms = PyList::empty(py);
    for (atom_idx, entry) in fidelity.iter().enumerate() {
        let a = PyDict::new(py);
        a.set_item("atom", atom_idx)?;
        match entry {
            Some(fid) => {
                a.set_item("topology", fid.topology)?;
                a.set_item("uniformity_statistic", fid.uniformity_statistic)?;
                a.set_item("uniformity_p_value", fid.uniformity_p_value)?;
                a.set_item("arclength_defect", fid.arclength_defect)?;
                a.set_item("n_coords", fid.n_coords)?;
                a.set_item("verdict", fid.verdict.label())?;
                a.set_item("certified", fid.certified)?;
                match &fid.coords_u_arc {
                    Some(u) => a.set_item("coords_u_arc", u.clone().into_pyarray(py))?,
                    None => a.set_item("coords_u_arc", py.None())?,
                }
                a.set_item("raw_arclength_defect_rms", fid.raw_arclength_defect_rms)?;
                a.set_item("raw_arclength_defect_max", fid.raw_arclength_defect_max)?;
                a.set_item("min_speed_over_mean", fid.min_speed_over_mean)?;
                a.set_item("max_speed_over_mean", fid.max_speed_over_mean)?;
                a.set_item("log_speed_rms", fid.log_speed_rms)?;
                // F2 two-part split: chart honesty vs occupancy law.
                a.set_item("chart_honest", fid.chart_honest)?;
                a.set_item("occupancy", fid.occupancy)?;
                a.set_item("occupancy_anchors", fid.occupancy_anchors)?;
                a.set_item("occupancy_d_eff", fid.occupancy_d_eff)?;
            }
            None => {
                a.set_item("topology", py.None())?;
            }
        }
        atoms.append(a)?;
    }
    d.set_item("atoms", atoms)?;
    Ok(d)
}

/// Build the result-dict entry for the curved-dictionary incoherence/curvature
/// certificate inputs (#1008). This is a measurement payload only; it deliberately
/// does not include a certified/uncertified verdict.
fn sae_incoherence_report_dict<'py>(
    py: Python<'py>,
    report: &gam::terms::sae::manifold::CertificateInputs,
) -> PyResult<Bound<'py, PyDict>> {
    let d = PyDict::new(py);
    d.set_item("mu_hat", report.mu_hat)?;
    d.set_item(
        "per_atom_kappa_hat",
        Array1::from_vec(report.per_atom_kappa_hat.clone()).into_pyarray(py),
    )?;
    d.set_item(
        "per_atom_mean_activity",
        Array1::from_vec(report.per_atom_mean_activity.clone()).into_pyarray(py),
    )?;
    d.set_item(
        "per_atom_peak_activity",
        Array1::from_vec(report.per_atom_peak_activity.clone()).into_pyarray(py),
    )?;
    d.set_item("mean_activity_floor", report.mean_activity_floor)?;
    d.set_item("peak_activity_floor", report.peak_activity_floor)?;
    d.set_item("snr_proxy", report.snr_proxy)?;
    d.set_item("dispersion", report.dispersion)?;
    // The #1008 global-optimality verdict: a string label + signed margin so a
    // consumer can read both the decision and how far it is from the threshold.
    let (verdict_label, margin) = match report.global_optimality {
        gam::terms::sae::manifold::GlobalOptimalityVerdict::CertifiedGlobal { margin } => {
            ("certified_global", margin)
        }
        gam::terms::sae::manifold::GlobalOptimalityVerdict::Uncertified { margin } => {
            ("uncertified", margin)
        }
    };
    d.set_item("global_optimality", verdict_label)?;
    d.set_item(
        "global_optimality_certified",
        report.global_optimality.is_certified(),
    )?;
    d.set_item("global_optimality_margin", margin)?;
    d.set_item("note", report.note.clone())?;
    Ok(d)
}

#[pyfunction(signature = (
    z,
    atom_basis,
    atom_dim,
    basis_values,
    basis_jacobian,
    basis_sizes,
    decoder_coefficients,
    smooth_penalties,
    initial_logits,
    initial_coords,
    alpha,
    tau,
    learnable_alpha,
    sparsity_strength = 1.0,
    smoothness = 1.0,
    max_iter = 12,
    learning_rate = 1.0,
    ridge_ext_coord = 1.0e-6,
    ridge_beta = 1.0e-6,
    gumbel_schedule = None,
    analytic_penalties = None,
))]
fn sae_manifold_fit_ibp<'py>(
    py: Python<'py>,
    z: PyReadonlyArray2<'py, f64>,
    atom_basis: Vec<String>,
    atom_dim: Vec<usize>,
    basis_values: PyReadonlyArray3<'py, f64>,
    basis_jacobian: PyReadonlyArray4<'py, f64>,
    basis_sizes: Vec<usize>,
    decoder_coefficients: PyReadonlyArray3<'py, f64>,
    smooth_penalties: PyReadonlyArray3<'py, f64>,
    initial_logits: PyReadonlyArray2<'py, f64>,
    initial_coords: PyReadonlyArray3<'py, f64>,
    alpha: f64,
    tau: f64,
    learnable_alpha: bool,
    sparsity_strength: f64,
    smoothness: f64,
    max_iter: usize,
    learning_rate: f64,
    ridge_ext_coord: f64,
    ridge_beta: f64,
    gumbel_schedule: Option<&Bound<'py, PyDict>>,
    analytic_penalties: Option<String>,
) -> PyResult<Py<PyDict>> {
    sae_manifold_fit(
        py,
        z,
        atom_basis,
        atom_dim,
        basis_values,
        basis_jacobian,
        basis_sizes,
        decoder_coefficients,
        smooth_penalties,
        initial_logits,
        initial_coords,
        alpha,
        tau,
        learnable_alpha,
        "ibp_map".to_string(),
        sparsity_strength,
        smoothness,
        max_iter,
        learning_rate,
        ridge_ext_coord,
        ridge_beta,
        gumbel_schedule,
        analytic_penalties,
        None,
        // IBP-MAP never reaches the JumpReLU dispatch; threshold is inert here.
        0.0,
        // No output-Fisher shard on this convenience IBP entry point; the
        // metric stays Euclidean (the precomputed-basis `sae_manifold_fit` and
        // the auto `sae_manifold_fit_minimal` entry points carry the shard).
        None,
        None,
        None,
        // No per-row design-honesty weights on this convenience IBP entry point.
        None,
        // No explicit separation-barrier / IBP-α values on this convenience IBP
        // entry point: use their canonical data-derived and assignment defaults.
        None,
        None,
        // Residual promotion stays disabled for this narrow convenience entry;
        // canonical structured whitening, scale quotienting, and rank charge are
        // unconditional in the shared core.
        false,
        true,
        true,
        false,
    )
}

///
/// The full-batch arrow-Schur path retains, for all `n_obs` rows, the basis
/// `(N × M_total)`, the basis jacobian `(N × M_total × d_max)`, and the gating
/// logits `(N × K)`. The dominant per-row working set is therefore
/// `w = (M_total · (1 + d_max) + K) · 8` bytes. When `n_obs · w` exceeds the
/// in-core budget — the GPU memory budget when a device is present, else a
/// conservative host-RAM working-set fraction — the fit streams in row chunks.
///
/// The chunk size keeps one chunk's per-row working set inside a cache/budget
/// window: on GPU, a modest fraction of the device memory budget; on CPU, a
/// multiple of the L2-cache estimate so a chunk's basis stays hot. The chunk is
/// clamped to `[1, n_obs]` and to at least a small floor so the per-chunk
/// reduced-Schur amortizes its `O(M² p + K³_reduced)` overhead.
///
/// Exposed to Python as a read-only diagnostic so callers (and the LLM-scale
/// streaming demo) can inspect the exact dispatch decision and chunk size the
/// fit will follow for a given `(n_obs, total_basis, k_atoms, d_max)` without
/// running it. It carries no tunable knobs — the plan is fully derived from the
/// problem size and the `GpuRuntime` memory budget.
#[pyfunction(signature = (n_obs, total_basis, k_atoms, d_max, border_dim = None))]
fn sae_streaming_plan(
    py: Python<'_>,
    n_obs: usize,
    total_basis: usize,
    k_atoms: usize,
    d_max: usize,
    border_dim: Option<usize>,
) -> PyResult<Py<PyDict>> {
    let border_dim = border_dim.unwrap_or(total_basis);
    let plan = gam::terms::sae::manifold::sae_streaming_plan_for_shape(
        n_obs,
        total_basis,
        k_atoms,
        d_max,
        border_dim,
    );
    let out = PyDict::new(py);
    out.set_item("streaming", plan.streaming)?;
    out.set_item("chunk_size", plan.chunk_size)?;
    out.set_item(
        "estimated_full_batch_bytes",
        plan.estimated_full_batch_bytes,
    )?;
    out.set_item(
        "estimated_dense_schur_bytes",
        plan.estimated_dense_schur_bytes,
    )?;
    out.set_item("estimated_row_cross_bytes", plan.estimated_row_cross_bytes)?;
    out.set_item(
        "estimated_direct_peak_bytes",
        plan.estimated_direct_peak_bytes,
    )?;
    out.set_item(
        "estimated_matrix_free_peak_bytes",
        plan.estimated_matrix_free_peak_bytes,
    )?;
    out.set_item("in_core_budget_bytes", plan.in_core_budget_bytes)?;
    out.set_item("host_available_bytes", plan.host_available_bytes)?;
    out.set_item("direct_admitted", plan.direct_admitted)?;
    out.set_item("matrix_free_admitted", plan.matrix_free_admitted)?;
    out.set_item(
        "route",
        if plan.direct_admitted {
            "direct"
        } else if plan.matrix_free_admitted {
            "matrix_free_streaming"
        } else {
            "refuse"
        },
    )?;
    Ok(out.unbind())
}

fn sae_streaming_plan_to_pydict<'py>(
    py: Python<'py>,
    plan: gam::terms::sae::manifold::SaeStreamingPlan,
) -> PyResult<Bound<'py, PyDict>> {
    let out = PyDict::new(py);
    out.set_item("streaming", plan.streaming)?;
    out.set_item("chunk_size", plan.chunk_size)?;
    out.set_item(
        "estimated_full_batch_bytes",
        plan.estimated_full_batch_bytes,
    )?;
    out.set_item(
        "estimated_dense_schur_bytes",
        plan.estimated_dense_schur_bytes,
    )?;
    out.set_item("estimated_row_cross_bytes", plan.estimated_row_cross_bytes)?;
    out.set_item(
        "estimated_direct_peak_bytes",
        plan.estimated_direct_peak_bytes,
    )?;
    out.set_item(
        "estimated_matrix_free_peak_bytes",
        plan.estimated_matrix_free_peak_bytes,
    )?;
    out.set_item("in_core_budget_bytes", plan.in_core_budget_bytes)?;
    out.set_item("host_available_bytes", plan.host_available_bytes)?;
    out.set_item("direct_admitted", plan.direct_admitted)?;
    out.set_item("matrix_free_admitted", plan.matrix_free_admitted)?;
    out.set_item(
        "route",
        if plan.direct_admitted {
            "direct"
        } else if plan.matrix_free_admitted {
            "matrix_free_streaming"
        } else {
            "refuse"
        },
    )?;
    Ok(out)
}

/// Parse a chart-topology name into the engine enum. `"circle"` charts carry
/// radian coordinates mod 2π; `"interval"` charts derive their bounds from
/// the observed coordinate range (magic by default — no extra knobs).
fn parse_chart_topology(
    name: &str,
    coords: ndarray::ArrayView1<'_, f64>,
) -> PyResult<gam::inference::layer_transport::ChartTopology> {
    use gam::inference::layer_transport::ChartTopology;
    match name {
        "circle" => Ok(ChartTopology::Circle),
        "interval" => {
            let lo = coords.iter().copied().fold(f64::INFINITY, f64::min);
            let hi = coords.iter().copied().fold(f64::NEG_INFINITY, f64::max);
            if !(lo.is_finite() && hi.is_finite()) || hi <= lo {
                return Err(PyValueError::new_err(format!(
                    "interval chart needs a non-degenerate finite coordinate range, got [{lo}, {hi}]"
                )));
            }
            Ok(ChartTopology::Interval { lo, hi })
        }
        other => Err(PyValueError::new_err(format!(
            "unknown chart topology {other:?}; expected \"circle\" or \"interval\""
        ))),
    }
}

fn layer_transport_report_to_pydict<'py>(
    py: Python<'py>,
    report: &gam::inference::layer_transport::LayerTransportReport,
) -> PyResult<Bound<'py, PyDict>> {
    let out = PyDict::new(py);
    out.set_item("layer_from", report.layer_from)?;
    out.set_item("layer_to", report.layer_to)?;
    out.set_item("topology_from", report.topology_from.name())?;
    out.set_item("topology_to", report.topology_to.name())?;
    out.set_item("topology_preserved", report.topology_preserved)?;
    out.set_item("degree", report.degree)?;
    out.set_item("degree_concentration", report.degree_concentration)?;
    out.set_item("rotation_offset", report.rotation_offset)?;
    out.set_item("isometry_defect", report.isometry_defect)?;
    out.set_item("isometry_defect_se", report.isometry_defect_se)?;
    out.set_item(
        "min_directional_derivative",
        report.min_directional_derivative,
    )?;
    out.set_item("transport_edf", report.transport_edf)?;
    out.set_item("smoothing_lambda", report.smoothing_lambda)?;
    out.set_item("noise_variance", report.noise_variance)?;
    out.set_item("residual_rms", report.residual_rms)?;
    out.set_item("n_obs", report.n_obs)?;
    out.set_item("composition_defect", report.composition_defect)?;
    out.set_item(
        "composition_max_studentized",
        report.composition_max_studentized,
    )?;
    out.set_item("composition_p_value", report.composition_p_value)?;
    out.set_item(
        "composition_gauge_reflected",
        report.composition_gauge_reflected,
    )?;
    Ok(out)
}

/// Fit one inter-layer concept transport map `t_to = h(t_from)` with the
/// engine's REML machinery (issue #1013) and return the evidence payload:
/// winding degree (circle→circle), topology-preservation verdict, the
/// data-density-weighted isometry defect with its delta-method SE, EDF, and
/// the selected smoothing level. See
/// `gam::inference::layer_transport` for the estimator and gauge discipline.
#[pyfunction(signature = (coords_from, coords_to, topology_from = "circle", topology_to = "circle", layer_from = 0, layer_to = 1))]
fn layer_transport_fit(
    py: Python<'_>,
    coords_from: PyReadonlyArray1<'_, f64>,
    coords_to: PyReadonlyArray1<'_, f64>,
    topology_from: &str,
    topology_to: &str,
    layer_from: usize,
    layer_to: usize,
) -> PyResult<Py<PyDict>> {
    let from = coords_from.as_array();
    let to = coords_to.as_array();
    let topo_from = parse_chart_topology(topology_from, from)?;
    let topo_to = parse_chart_topology(topology_to, to)?;
    let report = gam::inference::layer_transport::fit_layer_transport(
        layer_from, layer_to, from, to, topo_from, topo_to,
    )
    .map_err(PyValueError::new_err)?;
    Ok(layer_transport_report_to_pydict(py, &report)?.unbind())
}

/// A fitted inter-chart transport map `h: t_from ↦ t_to`, exposed as a live,
/// **invertible** object. Where [`layer_transport_fit`] returns only the
/// evidence dict, this carries the map itself: callers can `eval`,
/// `derivative`, `eval_with_variance`, and `invert` it — the transport algebra
/// needed to form `g_B ∘ g_A⁻¹` from two fitted transports. Construct with
/// [`fit_transport`]; the summary properties mirror the dict keys.
#[pyclass(module = "gamfit._rust", name = "FittedTransport")]
pub struct PyFittedTransport {
    inner: gam::inference::layer_transport::FittedTransport,
}

#[pymethods]
impl PyFittedTransport {
    /// Evaluate `h(t)` (wrapped to `[0, 2π)` on circle targets).
    fn eval<'py>(
        &self,
        py: Python<'py>,
        t: PyReadonlyArray1<'py, f64>,
    ) -> PyResult<Py<PyArray1<f64>>> {
        let out = self
            .inner
            .eval(t.as_array())
            .map_err(PyValueError::new_err)?;
        Ok(out.into_pyarray(py).unbind())
    }

    /// Evaluate the chart-coordinate derivative `h′(t)`.
    fn derivative<'py>(
        &self,
        py: Python<'py>,
        t: PyReadonlyArray1<'py, f64>,
    ) -> PyResult<Py<PyArray1<f64>>> {
        let out = self
            .inner
            .derivative(t.as_array())
            .map_err(PyValueError::new_err)?;
        Ok(out.into_pyarray(py).unbind())
    }

    /// Evaluate `h(t)` and its pointwise delta-method variance as `(values,
    /// variances)`.
    fn eval_with_variance<'py>(
        &self,
        py: Python<'py>,
        t: PyReadonlyArray1<'py, f64>,
    ) -> PyResult<(Py<PyArray1<f64>>, Py<PyArray1<f64>>)> {
        let (values, variances) = self
            .inner
            .eval_with_variance(t.as_array())
            .map_err(PyValueError::new_err)?;
        Ok((
            values.into_pyarray(py).unbind(),
            variances.into_pyarray(py).unbind(),
        ))
    }

    /// Invert: for each target coordinate `y`, the source coordinate `t` with
    /// `eval([t]) == y`. Requires a topology-preserving (fold-free) map and
    /// rejects out-of-image interval targets.
    fn invert<'py>(
        &self,
        py: Python<'py>,
        y: PyReadonlyArray1<'py, f64>,
    ) -> PyResult<Py<PyArray1<f64>>> {
        let out = self
            .inner
            .invert(y.as_array())
            .map_err(PyValueError::new_err)?;
        Ok(out.into_pyarray(py).unbind())
    }

    /// The full evidence payload — same keys as [`layer_transport_fit`]'s dict.
    #[pyo3(signature = (layer_from = 0, layer_to = 1))]
    fn report(&self, py: Python<'_>, layer_from: usize, layer_to: usize) -> PyResult<Py<PyDict>> {
        let report = self.inner.report(layer_from, layer_to);
        Ok(layer_transport_report_to_pydict(py, &report)?.unbind())
    }

    #[getter]
    fn topology_from(&self) -> &'static str {
        self.inner.topology_from.name()
    }
    #[getter]
    fn topology_to(&self) -> &'static str {
        self.inner.topology_to.name()
    }
    #[getter]
    fn topology_preserved(&self) -> bool {
        self.inner.topology_preserved
    }
    #[getter]
    fn degree(&self) -> Option<i32> {
        self.inner.degree
    }
    #[getter]
    fn isometry_defect(&self) -> f64 {
        self.inner.isometry_defect
    }
}

/// Fit an inter-chart transport map `t_to = h(t_from)` and return it as a live,
/// invertible [`PyFittedTransport`] (rather than the summary dict that
/// [`layer_transport_fit`] returns). `coords_from[i]` and `coords_to[i]`
/// coordinatize the same observation in the source and target charts;
/// topologies are `"circle"` or `"interval"`.
#[pyfunction(signature = (coords_from, coords_to, topology_from = "circle", topology_to = "circle"))]
fn fit_transport(
    coords_from: PyReadonlyArray1<'_, f64>,
    coords_to: PyReadonlyArray1<'_, f64>,
    topology_from: &str,
    topology_to: &str,
) -> PyResult<PyFittedTransport> {
    let from = coords_from.as_array();
    let to = coords_to.as_array();
    let topo_from = parse_chart_topology(topology_from, from)?;
    let topo_to = parse_chart_topology(topology_to, to)?;
    let inner = gam::inference::layer_transport::fit_transport_map(from, to, topo_from, topo_to)
        .map_err(PyValueError::new_err)?;
    Ok(PyFittedTransport { inner })
}

/// Fit a whole ladder of layer charts: every adjacent transport map plus
/// every two-hop map with the composition-law test
/// `h_{l→l+2} ≟ h_{l+1→l+2} ∘ h_{l→l+1}` attached (issue #1013). `coords` is
/// a list of equal-length 1-D coordinate arrays (one per layer, same rows);
/// `topology` applies to every chart; `layers` are optional labels
/// (defaulting to `0..len`). Returns `{"adjacent": [...], "two_hop": [...]}`.
#[pyfunction(signature = (coords, topology = "circle", layers = None))]
fn layer_transport_ladder(
    py: Python<'_>,
    coords: &Bound<'_, PyList>,
    topology: &str,
    layers: Option<Vec<usize>>,
) -> PyResult<Py<PyDict>> {
    use gam::inference::layer_transport::transport_ladder;
    let mut coord_vecs: Vec<ndarray::Array1<f64>> = Vec::with_capacity(coords.len());
    for item in coords.iter() {
        let arr: PyReadonlyArray1<'_, f64> = item.extract()?;
        coord_vecs.push(arr.as_array().to_owned());
    }
    let layer_labels: Vec<usize> = match layers {
        Some(labels) => {
            if labels.len() != coord_vecs.len() {
                return Err(PyValueError::new_err(format!(
                    "layers has {} entries for {} coordinate arrays",
                    labels.len(),
                    coord_vecs.len()
                )));
            }
            labels
        }
        None => (0..coord_vecs.len()).collect(),
    };
    let mut topologies = Vec::with_capacity(coord_vecs.len());
    for coord in &coord_vecs {
        topologies.push(parse_chart_topology(topology, coord.view())?);
    }
    let ladder =
        transport_ladder(&layer_labels, &coord_vecs, &topologies).map_err(PyValueError::new_err)?;
    let out = PyDict::new(py);
    let adjacent = PyList::empty(py);
    for report in &ladder.adjacent {
        adjacent.append(layer_transport_report_to_pydict(py, report)?)?;
    }
    let two_hop = PyList::empty(py);
    for report in &ladder.two_hop {
        two_hop.append(layer_transport_report_to_pydict(py, report)?)?;
    }
    out.set_item("adjacent", adjacent)?;
    out.set_item("two_hop", two_hop)?;
    Ok(out.unbind())
}

/// Pulled-back chart-to-chart transfer operators `A_kj` for a frozen model
/// component (the D2 "equation of motion" for a fitted atom across layers).
///
/// The torch lane owns the ambient component JVP `J_F(x)·v`; this entry point
/// owns the chart-frame algebra and density aggregation
/// `A_kj(x) = (J_kᵀ J_k)⁻¹ J_kᵀ J_F(x) J_j(x)`, per token, then the
/// density-weighted mean/variance across tokens.
///
/// `output_chart_jets` is `[n_tokens, ambient, d_out]` — the output atom's
/// decoder frame `J_k(x)` (for a fitted circle, the 2-D plane frame) at each
/// token. `ambient_jvps` is `[n_tokens, ambient, d_in]` — the model Jacobian
/// already applied to the input atom's decoder frame, `J_F(x)·J_j(x)`. Optional
/// non-negative `weights` `[n_tokens]` supply the empirical token density.
///
/// Returns `{"mean": [d_out, d_in], "variance": [d_out, d_in], "effective_n":
/// f, "token_operators": [n_tokens, d_out, d_in]}`. `mean` is the transfer
/// operator; `variance` is the density-weighted token spread (the band);
/// `effective_n` is Kish's effective sample size under the weights.
#[pyfunction(signature = (output_chart_jets, ambient_jvps, weights = None))]
fn chart_transfer_operator<'py>(
    py: Python<'py>,
    output_chart_jets: PyReadonlyArray3<'py, f64>,
    ambient_jvps: PyReadonlyArray3<'py, f64>,
    weights: Option<PyReadonlyArray1<'py, f64>>,
) -> PyResult<Py<PyDict>> {
    let jets = output_chart_jets.as_array();
    let jvps = ambient_jvps.as_array();
    let weights_arr = weights.as_ref().map(|w| w.as_array());
    let report = gam::terms::sae::chart_transfer::aggregate_pulled_back_operators(
        jets,
        jvps,
        weights_arr.as_ref().map(|w| w.view()),
    )
    .map_err(PyValueError::new_err)?;
    let out = PyDict::new(py);
    // For square 2x2 operators, the circle-atom readout: the density-weighted
    // circular mean of the per-token SO(2) polar angles with its SE (the
    // rotation band), plus the mean operator's own polar angle. `None` when the
    // operators are not 2x2 or any token reflects/collapses the chart — the
    // caller must then report the orientation flip rather than an angle.
    if report.token_operators.dim().1 == 2 && report.token_operators.dim().2 == 2 {
        match gam::terms::sae::chart_transfer::rotation_angle_band(
            report.token_operators.view(),
            weights_arr.as_ref().map(|w| w.view()),
        ) {
            Ok((mean_angle, se)) => {
                out.set_item("rotation_angle_mean", mean_angle)?;
                out.set_item("rotation_angle_se", se)?;
            }
            Err(_) => {
                out.set_item("rotation_angle_mean", py.None())?;
                out.set_item("rotation_angle_se", py.None())?;
            }
        }
        match gam::terms::sae::chart_transfer::so2_polar_angle(report.mean.view()) {
            Ok(angle) => out.set_item("rotation_angle_of_mean", angle)?,
            Err(_) => out.set_item("rotation_angle_of_mean", py.None())?,
        }
    } else {
        out.set_item("rotation_angle_mean", py.None())?;
        out.set_item("rotation_angle_se", py.None())?;
        out.set_item("rotation_angle_of_mean", py.None())?;
    }
    out.set_item("mean", report.mean.into_pyarray(py))?;
    out.set_item("variance", report.variance.into_pyarray(py))?;
    out.set_item("effective_n", report.effective_n)?;
    out.set_item("token_operators", report.token_operators.into_pyarray(py))?;
    Ok(out.unbind())
}

/// Transport / Lie-equivariance certificate for one square transfer operator
/// `A` (the D2 classifier: near-isometry ⇒ the layer *carries* the atom;
/// rotation/rescale ⇒ the layer *computes* on it).
///
/// `transport_defect = ‖AᵀA − I‖_F` (zero ⇒ isometric transport). The circle's
/// infinitesimal-rotation generator `G = [[0,−1],[1,0]]` supplies the
/// equivariance test: `equivariance_defect = ‖A·G_in − G_out·A‖_F` (zero ⇒ `A`
/// commutes with the rotation action, so the cyclic structure is preserved).
/// `input_generator` / `output_generator` are the `[d, d]` generators of the
/// input and output charts. Returns `{"transport_defect": f,
/// "equivariance_defect": f, "rotation_angle": f | None}` (the SO(2)
/// polar-factor angle of `A`; `None` unless `A` is 2x2 with det > 0).
#[pyfunction(signature = (operator, input_generator, output_generator))]
fn certify_chart_transfer(
    py: Python<'_>,
    operator: PyReadonlyArray2<'_, f64>,
    input_generator: PyReadonlyArray2<'_, f64>,
    output_generator: PyReadonlyArray2<'_, f64>,
) -> PyResult<Py<PyDict>> {
    let cert = gam::terms::sae::chart_transfer::certify_square_transfer(
        operator.as_array(),
        input_generator.as_array(),
        output_generator.as_array(),
    )
    .map_err(PyValueError::new_err)?;
    let out = PyDict::new(py);
    out.set_item("transport_defect", cert.transport_defect)?;
    out.set_item("equivariance_defect", cert.equivariance_defect)?;
    // The certified operator's SO(2) polar angle (2x2, det > 0 only): the
    // "rotates by Z radians" readout next to the defects. `None` for other
    // dimensions or an orientation-flipping operator.
    match gam::terms::sae::chart_transfer::so2_polar_angle(operator.as_array()) {
        Ok(angle) => out.set_item("rotation_angle", angle)?,
        Err(_) => out.set_item("rotation_angle", py.None())?,
    }
    Ok(out.unbind())
}

/// Cross-checkpoint descriptive atom-trajectory dynamics (issue #1102).
///
/// `decoder_grid` is `[n_checkpoints, n_atoms, n_grid, ambient_dim]`: every
/// atom's decoder curve sampled on the shared `latent_grid` at every checkpoint.
/// `checkpoint_ids` / `atom_names` label the checkpoint and atom axes (lengths
/// must match the corresponding grid axes). For each atom this walks consecutive
/// checkpoints and, per step `c → c+1`, fits the inter-checkpoint transport map
/// (checkpoint axis reused as the transport "layer" axis) and records the
/// DESCRIPTIVE decoder change of that step: the displacement vector at the
/// most-moved latent coordinate, its L2 norm, and the grid RMS / max L2 change.
///
/// Returns `{"trajectories": [ {atom_name, descriptive_step_changes:[...],
/// transports:[...transport...]} ]}`. `descriptive_step_changes` are plain
/// measured decoder changes — NO e-values, NO penalty-debiased Riesz contrasts,
/// and NO coverage claim: the anytime-valid change e-process and the Riesz
/// contrast estimator were removed because a bare decoder grid does not supply
/// the inputs a coverage-valid change certificate requires. See
/// `gam::inference::checkpoint_dynamics`.
#[pyfunction(signature = (decoder_grid, checkpoint_ids, atom_names, latent_grid))]
fn sae_checkpoint_dynamics(
    py: Python<'_>,
    decoder_grid: PyReadonlyArray4<'_, f64>,
    checkpoint_ids: Vec<String>,
    atom_names: Vec<String>,
    latent_grid: PyReadonlyArray1<'_, f64>,
) -> PyResult<Py<PyDict>> {
    use gam::inference::checkpoint_dynamics::{CheckpointDynamicsInput, checkpoint_atom_dynamics};
    let grid = decoder_grid.as_array();
    let lat = latent_grid.as_array();
    let input = CheckpointDynamicsInput {
        decoder_grid: grid,
        checkpoint_ids: &checkpoint_ids,
        atom_names: &atom_names,
        latent_grid: lat,
    };
    let trajectories = checkpoint_atom_dynamics(&input).map_err(PyValueError::new_err)?;

    let out = PyDict::new(py);
    let traj_list = PyList::empty(py);
    for traj in &trajectories {
        let t = PyDict::new(py);
        t.set_item("atom_name", &traj.atom_name)?;
        // Descriptive per-step decoder changes: measured displacement only, with
        // no e-values, debiased contrasts, or coverage claim (the change
        // estimator a bare decoder grid cannot support was removed).
        let steps = PyList::empty(py);
        for change in &traj.descriptive_step_changes {
            let c = PyDict::new(py);
            c.set_item("checkpoint_from", &change.checkpoint_from)?;
            c.set_item("checkpoint_to", &change.checkpoint_to)?;
            c.set_item("latent_coordinate", change.latent_coordinate)?;
            c.set_item(
                "displacement_at_mode",
                change.displacement_at_mode.clone().into_pyarray(py),
            )?;
            c.set_item("l2_at_mode", change.l2_at_mode)?;
            c.set_item("grid_rms_l2", change.grid_rms_l2)?;
            c.set_item("grid_max_l2", change.grid_max_l2)?;
            steps.append(c)?;
        }
        t.set_item("descriptive_step_changes", steps)?;
        let transports = PyList::empty(py);
        for report in &traj.transports {
            transports.append(layer_transport_report_to_pydict(py, report)?)?;
        }
        t.set_item("transports", transports)?;
        traj_list.append(t)?;
    }
    out.set_item("trajectories", traj_list)?;
    Ok(out.unbind())
}

/// Rust-owned Rung-3 calibration design.  Python receives prepared fit and
/// prediction frames from this object and returns only the fitted linear
/// predictors; split/floor/screening/log/gauge/diagnostic math never crosses
/// the FFI boundary.
#[pyclass(module = "gamfit._rust", name = "_InterventionCalibrationPlan")]
struct PyInterventionCalibrationPlan {
    inner: gam::terms::sae::inference::intervention_shard::InterventionCalibrationPlan,
}

fn intervention_atom_labels(atoms: &[i64]) -> Vec<String> {
    atoms.iter().map(i64::to_string).collect()
}

#[pymethods]
impl PyInterventionCalibrationPlan {
    #[getter]
    fn formula(&self) -> &'static str {
        gam::terms::sae::inference::intervention_shard::CHART_CALIBRATION_FORMULA
    }

    fn constraints(&self, py: Python<'_>) -> PyResult<Py<PyDict>> {
        let out = PyDict::new(py);
        out.set_item(
            gam::terms::sae::inference::intervention_shard::CHART_CALIBRATION_SMOOTH_TERM,
            gam::terms::sae::inference::intervention_shard::CHART_CALIBRATION_SMOOTH_CONSTRAINT,
        )?;
        Ok(out.unbind())
    }

    fn fit_frame(&self, py: Python<'_>) -> PyResult<Py<PyDict>> {
        let out = PyDict::new(py);
        out.set_item("log_nu", self.inner.train_log_nu.clone())?;
        out.set_item("log_nu_hat", self.inner.train_log_nu_hat.clone())?;
        out.set_item("atom", intervention_atom_labels(&self.inner.train_atom))?;
        Ok(out.unbind())
    }

    fn reference_frame(&self, py: Python<'_>) -> PyResult<Py<PyDict>> {
        let out = PyDict::new(py);
        out.set_item(
            "log_nu_hat",
            vec![self.inner.reference_log_nu_hat; self.inner.measurable_atoms.len()],
        )?;
        out.set_item(
            "atom",
            intervention_atom_labels(&self.inner.measurable_atoms),
        )?;
        Ok(out.unbind())
    }

    fn eval_frame(&self, py: Python<'_>) -> PyResult<Py<PyDict>> {
        let out = PyDict::new(py);
        out.set_item("log_nu_hat", self.inner.eval_log_nu_hat.clone())?;
        out.set_item("atom", intervention_atom_labels(&self.inner.eval_atom))?;
        Ok(out.unbind())
    }

    #[getter]
    fn n_eval(&self) -> usize {
        self.inner.eval_log_nu.len()
    }

    fn finish(
        &self,
        py: Python<'_>,
        reference_eta: Vec<f64>,
        eval_eta: Vec<f64>,
    ) -> PyResult<Py<PyDict>> {
        let result = self
            .inner
            .finish(&reference_eta, &eval_eta)
            .map_err(py_value_error)?;
        let respeed = PyDict::new(py);
        for (atom, value) in result.respeed {
            respeed.set_item(atom, value)?;
        }
        let out = PyDict::new(py);
        out.set_item("respeed", respeed)?;
        out.set_item(
            "below_measurement_floor",
            result.below_measurement_floor,
        )?;
        out.set_item(
            "no_training_intervention",
            result.no_training_intervention,
        )?;
        out.set_item("floor_nats", result.floor_nats)?;
        out.set_item("heldout_rmse_lognats", result.heldout_rmse_lognats)?;
        out.set_item("n_train", result.n_train)?;
        out.set_item("n_eval", result.n_eval)?;
        Ok(out.unbind())
    }
}

/// Validate a complete intervention shard and prepare the single Rust-owned
/// Rung-3 calibration design.  `prediction` is deliberately a closed choice;
/// requesting Rung 2 without a Rung-2 channel is a typed core error.
#[pyfunction]
fn intervention_calibration_plan<'py>(
    row_id: PyReadonlyArray1<'py, i64>,
    atom: PyReadonlyArray1<'py, i64>,
    dose: PyReadonlyArray2<'py, f64>,
    nu_hat_1: PyReadonlyArray1<'py, f64>,
    nu_hat_2: Option<PyReadonlyArray1<'py, f64>>,
    nu_measured: PyReadonlyArray1<'py, f64>,
    group: PyReadonlyArray1<'py, i64>,
    is_control: PyReadonlyArray1<'py, bool>,
    layer: i64,
    seed: u64,
    prediction: &str,
    split_seed: u64,
    floor_quantile: f64,
) -> PyResult<PyInterventionCalibrationPlan> {
    use gam::terms::sae::inference::intervention_shard::{
        InterventionCalibrationSpec, InterventionShard, PredictedNats,
        prepare_intervention_calibration,
    };

    let prediction = match prediction {
        "rung1" => PredictedNats::Rung1,
        "rung2" => PredictedNats::Rung2,
        other => {
            return Err(py_value_error(format!(
                "intervention calibration prediction must be 'rung1' or 'rung2'; got {other:?}"
            )));
        }
    };
    let dose_view = dose.as_array();
    let shard = InterventionShard {
        row_id: row_id.as_array().iter().copied().collect(),
        atom: atom.as_array().iter().copied().collect(),
        dose: dose_view.iter().copied().collect(),
        d_dose: dose_view.ncols(),
        nu_hat_1: nu_hat_1.as_array().iter().copied().collect(),
        nu_hat_2: nu_hat_2.map(|values| values.as_array().iter().copied().collect()),
        nu_measured: nu_measured.as_array().iter().copied().collect(),
        group: group.as_array().iter().copied().collect(),
        is_control: is_control.as_array().iter().copied().collect(),
        layer,
        seed,
    };
    let spec = InterventionCalibrationSpec {
        prediction,
        split_seed,
        floor_quantile,
    };
    let inner = prepare_intervention_calibration(&shard, spec).map_err(py_value_error)?;
    Ok(PyInterventionCalibrationPlan { inner })
}
