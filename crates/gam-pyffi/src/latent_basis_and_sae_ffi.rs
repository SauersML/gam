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
    let converged: bool;
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
        converged = outputs.converged;
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
        converged = outputs.converged;
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
    out.set_item("status", if converged { "ok" } else { "not_converged" })?;
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
        },
    )
    .map_err(|err| py_value_error(err.to_string()))?;

    let out = PyDict::new(py);
    out.set_item(
        "status",
        if outputs.converged {
            "ok"
        } else {
            "not_converged"
        },
    )?;
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

/// Envelope used for both the fit and predict pyfunctions. The
/// `model_class` discriminator is required so the Python load path can
/// tell a `MultinomialSavedModel` apart from the scalar `FittedModel`
/// payload that `fit_table` produces.
#[derive(serde::Serialize, serde::Deserialize)]
struct MultinomialModelEnvelope {
    model_class: String,
    saved: gam::families::multinomial::MultinomialSavedModel,
}

/// Fit a penalized multinomial-logit GAM from a Wilkinson formula against
/// a `headers + rows` table. Returns the bincode-free, serde-JSON model
/// payload that `gamfit.MultinomialModel` deserialises and stores under
/// `Model._model_bytes`.
#[pyfunction(signature = (
    headers,
    rows,
    formula,
    init_lambda = 1.0,
    max_iter = 50,
    tol = 1.0e-7,
))]
fn fit_multinomial_formula_pyfunc<'py>(
    py: Python<'py>,
    headers: Vec<String>,
    rows: Vec<Vec<String>>,
    formula: String,
    init_lambda: f64,
    max_iter: usize,
    tol: f64,
) -> PyResult<Py<PyBytes>> {
    let bytes = detach_pyresult(py, "fit_multinomial_formula", move || {
        let dataset = dataset_with_inferred_schema(headers, rows).map_err(py_value_error)?;
        // Typed engine path: `EstimationError` → matching `gamfit.*Error`
        // subclass via `estimation_error_to_pyerr` (issue #343).
        let saved = gam::families::multinomial::fit_penalized_multinomial_formula(
            &dataset,
            &formula,
            &gam::solver::fit_orchestration::FitConfig::default(),
            init_lambda,
            max_iter,
            tol,
        )
        .map_err(estimation_error_to_pyerr)?;
        let envelope = MultinomialModelEnvelope {
            model_class: "multinomial".to_string(),
            saved,
        };
        serde_json::to_vec(&envelope)
            .map_err(|err| py_value_error(format!("failed to serialize multinomial model: {err}")))
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
    rows: Vec<Vec<String>>,
) -> PyResult<Py<PyArray2<f64>>> {
    let probs = detach_pyresult(py, "predict_multinomial_formula", move || {
        let envelope: MultinomialModelEnvelope =
            serde_json::from_slice(&model_bytes).map_err(|err| {
                py_value_error(format!("failed to deserialize multinomial model: {err}"))
            })?;
        if envelope.model_class != "multinomial" {
            return Err(py_value_error(format!(
                "predict_multinomial_formula: model_class = {:?}, expected 'multinomial'",
                envelope.model_class
            )));
        }
        let dataset = dataset_with_inferred_schema(headers, rows).map_err(py_value_error)?;
        // Typed engine path: `EstimationError` → matching `gamfit.*Error`
        // subclass via `estimation_error_to_pyerr` (issue #343).
        gam::families::multinomial::predict_multinomial_formula(&envelope.saved, &dataset)
            .map_err(estimation_error_to_pyerr)
    })?;
    Ok(probs.into_pyarray(py).unbind())
}

/// Predict class probabilities WITH delta-method per-class probability standard
/// errors and z-scaled confidence bounds for a saved multinomial model (#1101).
/// Returns a dict with `probs`, `prob_se`, `mean_lower`, `mean_upper` (all
/// `(N_new, K)` arrays, columns aligned with `class_levels`) plus the `z`
/// multiplier used. The bounds are the simplex-clamped delta band
/// `p_c ± z·SE(p_c)`. `prob_se`/bounds are NaN-free only when the saved model
/// carries covariance (REML-fitted models do); a legacy payload yields
/// `prob_se = None`.
#[pyfunction(signature = (model_bytes, headers, rows, z = 1.959963984540054))]
fn predict_multinomial_intervals_pyfunc<'py>(
    py: Python<'py>,
    model_bytes: Vec<u8>,
    headers: Vec<String>,
    rows: Vec<Vec<String>>,
    z: f64,
) -> PyResult<Py<PyDict>> {
    let (probs, prob_se) = detach_pyresult(py, "predict_multinomial_intervals", move || {
        let envelope: MultinomialModelEnvelope =
            serde_json::from_slice(&model_bytes).map_err(|err| {
                py_value_error(format!("failed to deserialize multinomial model: {err}"))
            })?;
        if envelope.model_class != "multinomial" {
            return Err(py_value_error(format!(
                "predict_multinomial_intervals: model_class = {:?}, expected 'multinomial'",
                envelope.model_class
            )));
        }
        let dataset = dataset_with_inferred_schema(headers, rows).map_err(py_value_error)?;
        gam::families::multinomial::predict_multinomial_formula_with_se(&envelope.saved, &dataset)
            .map_err(estimation_error_to_pyerr)
    })?;
    let out = PyDict::new(py);
    out.set_item("z", z)?;
    out.set_item("probs", probs.clone().into_pyarray(py))?;
    match prob_se {
        Some(se) => {
            // Simplex-clamped delta band p_c ± z·SE(p_c).
            let mut lower = probs.clone();
            let mut upper = probs.clone();
            for ((r, c), &s) in se.indexed_iter() {
                let p = probs[[r, c]];
                lower[[r, c]] = (p - z * s).clamp(0.0, 1.0);
                upper[[r, c]] = (p + z * s).clamp(0.0, 1.0);
            }
            out.set_item("prob_se", se.into_pyarray(py))?;
            out.set_item("mean_lower", lower.into_pyarray(py))?;
            out.set_item("mean_upper", upper.into_pyarray(py))?;
        }
        None => {
            out.set_item("prob_se", py.None())?;
            out.set_item("mean_lower", py.None())?;
            out.set_item("mean_upper", py.None())?;
        }
    }
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
    let envelope: MultinomialModelEnvelope = serde_json::from_slice(&model_bytes)
        .map_err(|err| py_value_error(format!("failed to deserialize multinomial model: {err}")))?;
    if envelope.model_class != "multinomial" {
        return Err(py_value_error(format!(
            "multinomial_smooth_significance: model_class = {:?}, expected 'multinomial'",
            envelope.model_class
        )));
    }
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
    let envelope: MultinomialModelEnvelope = serde_json::from_slice(&model_bytes)
        .map_err(|err| py_value_error(format!("failed to deserialize multinomial model: {err}")))?;
    if envelope.model_class != "multinomial" {
        return Err(py_value_error(format!(
            "multinomial_model_metadata: model_class = {:?}, expected 'multinomial'",
            envelope.model_class
        )));
    }
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
    out.set_item("iterations", envelope.saved.iterations)?;
    out.set_item("converged", envelope.saved.converged)?;
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

fn sae_atom_basis_kind_from_str(value: &str) -> SaeAtomBasisKind {
    match value.to_ascii_lowercase().replace('-', "_").as_str() {
        "duchon" => SaeAtomBasisKind::Duchon,
        "periodic" | "periodic_spline" | "circle" => SaeAtomBasisKind::Periodic,
        "sphere" => SaeAtomBasisKind::Sphere,
        "torus" => SaeAtomBasisKind::Torus,
        "euclidean" | "euclidean_patch" => SaeAtomBasisKind::EuclideanPatch,
        "poincare" | "poincare_patch" | "hyperbolic" => SaeAtomBasisKind::Poincare,
        "cylinder" => SaeAtomBasisKind::Cylinder,
        other => SaeAtomBasisKind::Precomputed(other.to_string()),
    }
}

/// The canonical lowercase string name of an atom basis kind — the inverse of
/// [`sae_atom_basis_kind_from_str`] for the round-trippable kinds, and the
/// string the python `from_payload` boundary reads under each atom's
/// `"basis_kind"` key and each plan's `"kind"` key. Derived from the FITTED
/// atom (not the user's input metadata) so a structure-search-grown / shrunk
/// dictionary serializes its DISCOVERED per-atom topology, not the input one.
fn sae_atom_basis_kind_name(kind: &SaeAtomBasisKind) -> String {
    match kind {
        SaeAtomBasisKind::Periodic => "periodic".to_string(),
        SaeAtomBasisKind::Duchon => "duchon".to_string(),
        SaeAtomBasisKind::Sphere => "sphere".to_string(),
        SaeAtomBasisKind::Torus => "torus".to_string(),
        SaeAtomBasisKind::EuclideanPatch => "euclidean_patch".to_string(),
        SaeAtomBasisKind::Poincare => "poincare".to_string(),
        SaeAtomBasisKind::Cylinder => "cylinder".to_string(),
        SaeAtomBasisKind::Precomputed(name) => name.clone(),
    }
}

/// Default per-axis harmonic order for a torus atom (Φ has `(2H+1)^d`
/// columns). Three harmonics per axis gives a 7-column 1-D factor and a
/// 49-column tensor basis at `d=2`, which is the smallest expansion that
/// reliably resolves a non-trivial signal on `T^2` without exploding the
/// design.
const SAE_DEFAULT_TORUS_HARMONICS: usize = 3;
/// Sphere chart basis size (lat/lon ⇒ `[1, x, y, z, xy, yz, xz]`).
const SAE_SPHERE_BASIS_SIZE: usize = 7;
/// Per-(compact)-axis resolution of the global decoder-projection coordinate
/// seed used by the fixed-decoder OOS solve. 256 phases on the circle spaces
/// adjacent grid points ≈0.004 apart, comfortably inside the basin of the
/// analytic Newton refinement for a small harmonic order, while staying cheap
/// (`O(K·N·256·p)` for periodic atoms). See
/// [`gam::terms::sae::manifold::SaeManifoldTerm::seed_coords_by_decoder_projection`].
const SAE_OOS_PROJECTION_GRID_RESOLUTION: usize = 256;

fn seed_oos_softmax_logits_from_projection_residuals(
    term: &mut gam::terms::sae::manifold::SaeManifoldTerm,
    target: ArrayView2<'_, f64>,
    tau: f64,
) {
    let (n_obs, p_out) = target.dim();
    let k_atoms = term.k_atoms();
    let mut seeded_logits = Array2::<f64>::zeros((n_obs, k_atoms));
    let mut decoded = vec![0.0_f64; p_out];
    for row in 0..n_obs {
        for atom_idx in 0..k_atoms {
            term.atoms[atom_idx].fill_decoded_row(row, &mut decoded);
            let mut err = 0.0_f64;
            for out_col in 0..p_out {
                let diff = target[[row, out_col]] - decoded[out_col];
                err += diff * diff;
            }
            seeded_logits[[row, atom_idx]] = -err / tau;
        }
        let reference = seeded_logits[[row, k_atoms - 1]];
        for atom_idx in 0..k_atoms {
            seeded_logits[[row, atom_idx]] -= reference;
        }
    }
    term.assignment.logits.assign(&seeded_logits);
}

fn seed_oos_ibp_logits_from_projected_decoder_lsq(
    term: &mut gam::terms::sae::manifold::SaeManifoldTerm,
    target: ArrayView2<'_, f64>,
    tau: f64,
    alpha: f64,
) {
    let (n_obs, p_out) = target.dim();
    let k_atoms = term.k_atoms();
    let ratio = alpha / (alpha + 1.0);
    let mut prior = Vec::with_capacity(k_atoms);
    for atom_idx in 0..k_atoms {
        prior.push(ratio.powi(atom_idx as i32).max(f64::MIN_POSITIVE));
    }
    let mut decoded = vec![vec![0.0_f64; p_out]; k_atoms];
    let mut norm_sq = vec![0.0_f64; k_atoms];
    let mut gates = vec![0.0_f64; k_atoms];
    let mut fitted = vec![0.0_f64; p_out];
    let mut seeded_logits = Array2::<f64>::zeros((n_obs, k_atoms));
    const OOS_IBP_BOX_LSQ_SWEEPS: usize = 12;
    const OOS_IBP_GATE_EPS: f64 = 1.0e-6;
    for row in 0..n_obs {
        for atom_idx in 0..k_atoms {
            term.atoms[atom_idx].fill_decoded_row(row, &mut decoded[atom_idx]);
            norm_sq[atom_idx] = decoded[atom_idx]
                .iter()
                .map(|v| v * v)
                .sum::<f64>()
                .max(1.0e-12);
            gates[atom_idx] = 0.0;
        }
        fitted.fill(0.0);
        for _ in 0..OOS_IBP_BOX_LSQ_SWEEPS {
            for atom_idx in 0..k_atoms {
                let old_gate = gates[atom_idx];
                let g_row = &decoded[atom_idx];
                let mut numerator = 0.0_f64;
                for out_col in 0..p_out {
                    let residual_without_atom =
                        target[[row, out_col]] - fitted[out_col] + old_gate * g_row[out_col];
                    numerator += g_row[out_col] * residual_without_atom;
                }
                let upper = prior[atom_idx] * (1.0 - OOS_IBP_GATE_EPS);
                let new_gate = (numerator / norm_sq[atom_idx]).clamp(0.0, upper);
                if new_gate != old_gate {
                    let delta = new_gate - old_gate;
                    for out_col in 0..p_out {
                        fitted[out_col] += delta * g_row[out_col];
                    }
                    gates[atom_idx] = new_gate;
                }
            }
        }
        for atom_idx in 0..k_atoms {
            let q =
                (gates[atom_idx] / prior[atom_idx]).clamp(OOS_IBP_GATE_EPS, 1.0 - OOS_IBP_GATE_EPS);
            seeded_logits[[row, atom_idx]] = tau * (q / (1.0 - q)).ln();
        }
    }
    term.assignment.logits.assign(&seeded_logits);
}

/// Duchon nullspace knob `m` for a SAE-manifold atom of latent dimension
/// `dim`, sized so the native reproducing-norm Gram (`PenaltySource::Primary`)
/// on the scale-free polyharmonic basis is well-posed in every dimension.
///
/// `m` maps to the polynomial-nullspace order via [`duchon_nullspace_from_m`]
/// (`m = 1 → Zero/p=1`, `m = 2 → Linear/p=2`, `m = k+1 → Degree(k)/p=k+1`), so
/// `p_order == m`. The pure-polyharmonic basis the `DuchonCoordinateEvaluator`
/// evaluates uses `s = 0`; sizing the null space by `2·m > dim + 2` keeps the
/// kernel CPD/null-space adequacy comfortable across dimensions, whose smallest
/// integer solution is `m = ⌊dim / 2⌋ + 2`.
///
/// For `dim == 1` this reproduces the historical `m = 2` (constant + linear
/// null space). For `dim ≥ 2` it grows the null space with the latent
/// dimension — the previous hard-coded `m = 2` left the resolved power/order
/// inconsistent with the power-0 evaluator. The seed build and every
/// [`DuchonCoordinateEvaluator`] refresh read this same derived `m`, so the
/// design `Φ`, its jet, and the penalty stay column-consistent (the issue-247
/// invariant).
fn sae_duchon_atom_m(dim: usize) -> usize {
    dim / 2 + 2
}
/// Maximum total monomial degree for a Euclidean tangent-patch SAE atom
/// (`{1, t_a, t_a t_b}` at degree 2).
const SAE_EUCLIDEAN_PATCH_MAX_DEGREE: usize = 2;

/// Flat-line polynomial degree of a Cylinder `S¹ × ℝ` atom's line axis (axis 1).
/// Mirrors the Euclidean-patch degree so the cylinder's flat factor matches the
/// patch candidate it races against; `Ml = SAE_CYLINDER_LINE_DEGREE + 1`.
const SAE_CYLINDER_LINE_DEGREE: usize = 2;

/// Recover the cylinder evaluator's `(circle_harmonics H, line_degree D)` from
/// its fitted product basis width `m = (2H + 1)·(D + 1)` and the production
/// line-degree convention (`D = SAE_CYLINDER_LINE_DEGREE`). The line factor width
/// `Ml = D + 1` must divide `m`, and the recovered circle width `Mc = m / Ml`
/// must be odd (`= 2H + 1`); otherwise the basis width is inconsistent with a
/// cylinder product and the error is surfaced rather than guessed.
fn sae_cylinder_harmonics_degree(m: usize) -> Result<(usize, usize), String> {
    let d_line = SAE_CYLINDER_LINE_DEGREE;
    let ml = d_line + 1;
    if ml == 0 || m == 0 || m % ml != 0 {
        return Err(format!(
            "sae_cylinder_harmonics_degree: basis size {m} is not (2H+1)·{ml} for a cylinder \
             with line degree {d_line}"
        ));
    }
    let mc = m / ml;
    if mc < 3 || mc % 2 == 0 {
        return Err(format!(
            "sae_cylinder_harmonics_degree: recovered circle width {mc} (= {m}/{ml}) is not an \
             odd 2H+1 ≥ 3 for a cylinder"
        ));
    }
    Ok(((mc - 1) / 2, d_line))
}

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
            let rate = match schedule.get_item("rate").map_err(|err| err.to_string())? {
                Some(value) => value.extract::<f64>().map_err(|err| err.to_string())?,
                None => 0.9,
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

/// Build per-atom Rust basis evaluators so the Newton loop can refresh
/// `Phi_k` and `dPhi_k/dt` between steps without bouncing back to Python.
///
/// Every supported kind has a concrete analytic evaluator: `Periodic`
/// (`latent_dim == 1`) → harmonic, `Sphere` (`latent_dim == 2`) → chart,
/// `Torus` → tensor harmonic, `Duchon` → radial+polynomial coordinate
/// evaluator (requires the atom's centers in `atom_centers`), and
/// `EuclideanPatch` → monomial patch. A `Precomputed` atom — or a kind whose
/// refresh metadata is missing (e.g. a Duchon atom with no centers) — has no
/// way to re-evaluate `Phi(t)` at updated coordinates, so construction errors
/// rather than freezing a stale snapshot.
fn build_sae_basis_evaluators(
    basis_kinds: &[SaeAtomBasisKind],
    basis_sizes: &[usize],
    atom_dim: &[usize],
    coord_blocks: &[Array2<f64>],
    atom_centers: &[Option<Array2<f64>>],
) -> Result<Vec<Option<Arc<dyn SaeBasisSecondJet>>>, String> {
    let k_atoms = basis_kinds.len();
    if atom_dim.len() != k_atoms
        || basis_sizes.len() != k_atoms
        || coord_blocks.len() != k_atoms
        || atom_centers.len() != k_atoms
    {
        return Err(format!(
            "build_sae_basis_evaluators: K-length metadata mismatch (kinds={k_atoms}, dims={}, sizes={}, coords={}, centers={})",
            atom_dim.len(),
            basis_sizes.len(),
            coord_blocks.len(),
            atom_centers.len()
        ));
    }
    let mut out: Vec<Option<Arc<dyn SaeBasisSecondJet>>> = Vec::with_capacity(k_atoms);
    for k in 0..k_atoms {
        let m = basis_sizes[k];
        let d = atom_dim[k];
        // Every production atom evaluator implements `SaeBasisSecondJet` (it
        // exposes the analytic basis Hessian). Returning the second-jet trait
        // object lets the term builder install it through
        // `with_basis_second_jet`, which is the slot the #1117 rank-revealing
        // reduction reads to reparametrize a rank-deficient decoder.
        let evaluator: Arc<dyn SaeBasisSecondJet> = match &basis_kinds[k] {
            SaeAtomBasisKind::Periodic if d == 1 && m % 2 == 1 => {
                Arc::new(PeriodicHarmonicEvaluator::new(m)?)
            }
            SaeAtomBasisKind::Sphere if d == 2 && m == SAE_SPHERE_BASIS_SIZE => {
                Arc::new(SphereChartEvaluator)
            }
            SaeAtomBasisKind::Torus if d >= 1 => {
                // Recover the per-axis harmonic count `H` from `m = (2H+1)^d`.
                let axis_m = sae_torus_axis_basis_size(m, d)?;
                let h = (axis_m - 1) / 2;
                Arc::new(TorusHarmonicEvaluator::new(d, h)?)
            }
            SaeAtomBasisKind::Duchon => {
                let centers = atom_centers[k].as_ref().ok_or_else(|| {
                    format!(
                        "build_sae_basis_evaluators: Duchon atom {k} cannot refresh its basis without centers; \
                         build the atom through the SAE auto path so its Duchon centers are threaded in"
                    )
                })?;
                // Same dimension-aware `m` the seed build
                // (`sae_build_duchon_atom`) used: both read `centers.ncols()`,
                // so the refreshed `Φ`/jet stays column-consistent with the
                // seed design and its native (Primary) penalty (issue-247 invariant).
                Arc::new(DuchonCoordinateEvaluator::new(
                    centers.clone(),
                    sae_duchon_atom_m(centers.ncols()),
                )?)
            }
            SaeAtomBasisKind::Cylinder if d == 2 => {
                // Cylinder `S¹ × ℝ`: recover the circle harmonic count `H` and
                // line degree `D` from the fitted product width
                // `m = (2H+1)·(D+1)` (the production line-degree convention), so
                // the refreshed `Φ`/jet stays column-consistent with the seed
                // design. An inconsistent width is surfaced, not guessed.
                let (h, d_line) = sae_cylinder_harmonics_degree(m)?;
                Arc::new(CylinderHarmonicEvaluator::new(h, d_line)?)
            }
            SaeAtomBasisKind::EuclideanPatch | SaeAtomBasisKind::Poincare => Arc::new(
                EuclideanPatchEvaluator::new(d, SAE_EUCLIDEAN_PATCH_MAX_DEGREE)?,
            ),
            SaeAtomBasisKind::Cylinder => {
                return Err(format!(
                    "build_sae_basis_evaluators: Cylinder atom {k} requires latent_dim == 2; got dim={d}, m={m}"
                ));
            }
            SaeAtomBasisKind::Precomputed(label) => {
                return Err(format!(
                    "build_sae_basis_evaluators: atom {k} basis {label:?} is precomputed and has no \
                     analytic refresh routine; the inner Newton latent update requires a basis kind \
                     that can re-evaluate Phi(t)/dPhi/dt at updated coordinates"
                ));
            }
            SaeAtomBasisKind::Periodic => {
                return Err(format!(
                    "build_sae_basis_evaluators: Periodic atom {k} requires latent_dim == 1 and odd basis size; got dim={d}, m={m}"
                ));
            }
            SaeAtomBasisKind::Sphere => {
                return Err(format!(
                    "build_sae_basis_evaluators: Sphere atom {k} requires latent_dim == 2 and basis size {SAE_SPHERE_BASIS_SIZE}; got dim={d}, m={m}"
                ));
            }
            SaeAtomBasisKind::Torus => {
                return Err(format!(
                    "build_sae_basis_evaluators: Torus atom {k} requires latent_dim >= 1; got dim={d}, m={m}"
                ));
            }
        };
        out.push(Some(evaluator));
    }
    Ok(out)
}

/// Build the per-row output-Fisher `RowMetric` from a flattened factor stack and
/// the harvest shard's provenance tag (#980).
///
/// `"output_fisher"` (the default, and the only tag a same-position shard or a
/// raw factor array carries) installs the same-position
/// [`RowMetric::output_fisher`](gam::inference::row_metric::RowMetric::output_fisher);
/// `"output_fisher_downstream"` (the KV-path aggregate over future positions)
/// installs
/// [`RowMetric::output_fisher_downstream`](gam::inference::row_metric::RowMetric::output_fisher_downstream).
/// Both are gauge-only and consumed identically by the lens/gauge/enrichment;
/// only the tag (and the science it certifies) differs. An unknown tag errors —
/// a silent fall-through to the same-position metric would mislabel the
/// certificate.
fn row_metric_from_fisher_provenance(
    u_flat: Array2<f64>,
    p_out: usize,
    rank: usize,
    provenance: Option<&str>,
) -> PyResult<gam::inference::row_metric::RowMetric> {
    let u = std::sync::Arc::new(u_flat);
    match provenance.unwrap_or("output_fisher") {
        "output_fisher" => gam::inference::row_metric::RowMetric::output_fisher(u, p_out, rank),
        "output_fisher_downstream" => {
            gam::inference::row_metric::RowMetric::output_fisher_downstream(u, p_out, rank)
        }
        other => Err(format!(
            "fisher_provenance must be 'output_fisher' or 'output_fisher_downstream'; \
             got {other:?}"
        )),
    }
    .map_err(py_value_error)
}

/// The Python-facing label for a [`MetricProvenance`], shared across every FFI
/// site that surfaces `metric_provenance` (#980). Centralized so a new
/// provenance variant is labelled in exactly one place.
fn metric_provenance_label(
    provenance: gam::inference::row_metric::MetricProvenance,
) -> &'static str {
    use gam::inference::row_metric::MetricProvenance;
    match provenance {
        MetricProvenance::Euclidean => "Euclidean",
        MetricProvenance::OutputFisher { .. } => "OutputFisher",
        MetricProvenance::OutputFisherDownstream { .. } => "OutputFisherDownstream",
        MetricProvenance::WhitenedStructured { .. } => "WhitenedStructured",
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
))]
#[allow(clippy::too_many_arguments)]
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
) -> PyResult<Py<PyDict>> {
    // The precomputed-basis entry point carries no Duchon centers / kernel
    // metadata, so any basis kind whose refresh needs them cannot re-evaluate
    // `Phi(t)` at updated coordinates. Pass empty per-atom centers; the
    // evaluator builder errors for such kinds (rather than silently freezing
    // the seed snapshot). Kinds with an analytic, centers-free basis
    // (periodic, sphere, torus) refresh as usual.
    let atom_centers: Vec<Option<Array2<f64>>> = vec![None; atom_basis.len()];
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
    )
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
) -> PyResult<Py<PyDict>> {
    let analytic_penalties: Option<serde_json::Value> = match analytic_penalties {
        Some(s) => Some(serde_json::from_str(&s).map_err(serde_json_error_to_pyerr)?),
        None => None,
    };
    let (n_obs, p_out) = z_view.dim();
    if n_obs == 0 || p_out == 0 {
        return Err(py_value_error(
            "sae_manifold_fit requires a non-empty (N, p) response".to_string(),
        ));
    }
    let k_atoms = atom_dim.len();
    // The SEED dictionary size, captured before the structure search may grow or
    // shrink it (#977). Used to map a structure-search-BORN atom (index ≥
    // `seed_k_atoms`) back to its template plan when serializing the variable-K
    // `atom_plans`: births clone atom 0's basis, so a born atom's build plan is
    // the seed template's, re-derived from the fitted atom below.
    let seed_k_atoms = k_atoms;
    if k_atoms == 0 {
        return Err(py_value_error(
            "sae_manifold_fit requires at least one atom".to_string(),
        ));
    }
    if atom_basis.len() != k_atoms || basis_sizes.len() != k_atoms {
        return Err(py_value_error(format!(
            "sae_manifold_fit metadata lengths must equal K={k_atoms}; got atom_basis={}, basis_sizes={}",
            atom_basis.len(),
            basis_sizes.len()
        )));
    }
    // The "t" block addresses the per-row latent coordinates (n_obs × d_max,
    // tier=Psi) — ARD, Isometry, BlockOrthogonality, etc. target this. The
    // "beta" block addresses the decoder coefficient matrix. `flatten_beta`
    // concatenates per-atom (M_k, p_out) blocks; the total decoder vector has
    // length `(Σ_k M_k) · p_out`. We register one "beta" block covering the
    // full vector with `d = Σ_k M_k` so the descriptor builder constructs a
    // valid MechanismSparsityPenalty (target length = d · p_out, feature
    // groups partitioning p_out). For multi-atom fits the penalty cannot
    // group-lasso the whole concatenation as a single (Σ M_k, p_out) matrix —
    // instead `add_sae_beta_penalty` in src/terms/sae/manifold.rs dispatches
    // the penalty per atom, rebuilding its target range/latent_dim to each
    // atom's (M_k, p_out) block. For k_atoms == 1 the per-atom dispatch
    // collapses to exactly this single block, so both paths agree (#240).
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
    if max_iter < 1 {
        return Err(py_value_error(format!(
            "sae_manifold_fit requires max_iter >= 1; got {max_iter}"
        )));
    }
    if let Some(k_top) = top_k {
        if k_top == 0 || k_top > k_atoms {
            return Err(py_value_error(format!(
                "top_k must satisfy 1 <= top_k <= k_atoms={k_atoms}; got {k_top}"
            )));
        }
    }
    if initial_logits.dim() != (n_obs, k_atoms) {
        return Err(py_value_error(format!(
            "initial_logits must be ({n_obs}, {k_atoms}); got {:?}",
            initial_logits.dim()
        )));
    }
    for (name, value) in [
        ("alpha", alpha),
        ("tau", tau),
        ("smoothness", smoothness),
        ("learning_rate", learning_rate),
        ("ridge_ext_coord", ridge_ext_coord),
        ("ridge_beta", ridge_beta),
    ] {
        if !value.is_finite() || value <= 0.0 {
            return Err(py_value_error(format!(
                "{name} must be finite and positive; got {value}"
            )));
        }
    }
    // `sparsity_strength == 0.0` is the canonical "no sparsity" baseline
    // (issue #184). Accept it and floor to a tiny positive sentinel before
    // taking the log so the log-rho parametrisation stays finite. The
    // resulting `lambda_sparse = exp(log(SPARSITY_DISABLED_FLOOR))` is
    // numerically equivalent to zero relative to the data-fit Hessian for
    // any realistic problem scale. Negatives, infinities, and NaN remain
    // rejected.
    if !sparsity_strength.is_finite() || sparsity_strength < 0.0 {
        return Err(py_value_error(format!(
            "sparsity_strength must be finite and non-negative; got {sparsity_strength}"
        )));
    }
    const SPARSITY_DISABLED_FLOOR: f64 = 1.0e-300;
    let sparsity_strength = if sparsity_strength == 0.0 {
        SPARSITY_DISABLED_FLOOR
    } else {
        sparsity_strength
    };

    let basis_values_shape = basis_values.shape().to_vec();
    if basis_values_shape[0] != k_atoms || basis_values_shape[1] != n_obs {
        return Err(py_value_error(format!(
            "basis_values must start with (K, N)=({k_atoms}, {n_obs}); got {:?}",
            basis_values_shape
        )));
    }
    let basis_jacobian_shape = basis_jacobian.shape().to_vec();
    if basis_jacobian_shape[0] != k_atoms || basis_jacobian_shape[1] != n_obs {
        return Err(py_value_error(format!(
            "basis_jacobian must start with (K, N)=({k_atoms}, {n_obs}); got {:?}",
            basis_jacobian_shape
        )));
    }
    let decoder_shape = decoder_coefficients.shape().to_vec();
    if decoder_shape[0] != k_atoms || decoder_shape[2] != p_out {
        return Err(py_value_error(format!(
            "decoder_coefficients must have shape (K, M_max, p)=({k_atoms}, M_max, {p_out}); got {:?}",
            decoder_shape
        )));
    }
    let smooth_shape = smooth_penalties.shape().to_vec();
    if smooth_shape[0] != k_atoms || smooth_shape[1] != smooth_shape[2] {
        return Err(py_value_error(format!(
            "smooth_penalties must have shape (K, M_max, M_max); got {:?}",
            smooth_shape
        )));
    }
    let coords_view = initial_coords;
    let coords_shape = coords_view.shape().to_vec();
    if coords_shape[0] != k_atoms || coords_shape[1] != n_obs {
        return Err(py_value_error(format!(
            "initial_coords must start with (K, N)=({k_atoms}, {n_obs}); got {:?}",
            coords_shape
        )));
    }
    let max_dim = coords_view.shape().get(2).copied().ok_or_else(|| {
        py_value_error("initial_coords must be a rank-3 (K, N, D_max) array".to_string())
    })?;
    let mut coord_blocks = Vec::with_capacity(k_atoms);
    for atom_idx in 0..k_atoms {
        let d = atom_dim[atom_idx];
        let m = basis_sizes[atom_idx];
        if m > basis_values_shape[2]
            || m > basis_jacobian_shape[2]
            || m > decoder_shape[1]
            || m > smooth_shape[1]
        {
            return Err(py_value_error(format!(
                "basis_sizes[{atom_idx}]={m} exceeds one of the padded M_max dimensions"
            )));
        }
        if d > max_dim {
            return Err(py_value_error(format!(
                "atom_dim[{atom_idx}]={d} exceeds initial_coords D_max={max_dim}"
            )));
        }
        if d > basis_jacobian_shape[3] {
            return Err(py_value_error(format!(
                "atom_dim[{atom_idx}]={d} exceeds basis_jacobian D_max={}",
                basis_jacobian_shape[3]
            )));
        }
        coord_blocks.push(coords_view.slice(s![atom_idx, 0..n_obs, 0..d]).to_owned());
    }

    let basis_kinds: Vec<SaeAtomBasisKind> = atom_basis
        .iter()
        .map(|kind| sae_atom_basis_kind_from_str(kind))
        .collect();
    let mode = match assignment_kind.as_str() {
        "softmax" => AssignmentMode::softmax(tau),
        "ibp_map" => AssignmentMode::ibp_map(tau, alpha, learnable_alpha),
        // The JumpReLU gate is a hard-thresholded bounded sigmoid: an atom is
        // active when its raw logit clears `jumprelu_threshold`, and the gate
        // value is the sigmoid (in [0, 1]) — the reconstruction *magnitude*
        // lives in the decoder, not the gate. `tau` is the sigmoid temperature
        // on the same logits; the threshold is the activation cut. The
        // threshold is caller-configurable (default 0.0); the cold-start seed
        // (see `sae_manifold_fit_minimal`) starts every logit above it so the
        // data-fit JVP, the sparsity prior gradient, and the assignment-weighted
        // decoder gradient are all non-zero at step 0.
        "jumprelu" => AssignmentMode::jumprelu(tau, jumprelu_threshold),
        _ => {
            return Err(py_value_error(format!(
                "assignment_kind must be one of 'softmax', 'ibp_map', or 'jumprelu'; got {assignment_kind}"
            )));
        }
    };
    if atom_centers.len() != k_atoms {
        return Err(py_value_error(format!(
            "sae_manifold_fit: atom_centers length {} must equal K={k_atoms}",
            atom_centers.len()
        )));
    }
    let evaluators = build_sae_basis_evaluators(
        &basis_kinds,
        &basis_sizes,
        &atom_dim,
        &coord_blocks,
        atom_centers,
    )
    .map_err(py_value_error)?;
    let mut base_term = term_from_padded_blocks_with_mode(
        n_obs,
        p_out,
        &basis_kinds,
        basis_values,
        basis_jacobian,
        &basis_sizes,
        &atom_dim,
        decoder_coefficients,
        smooth_penalties,
        initial_logits,
        &coord_blocks,
        mode,
        &evaluators,
    )
    .map_err(py_value_error)?;
    if let Some(schedule) =
        gumbel_temperature_schedule_from_pydict(gumbel_schedule).map_err(py_value_error)?
    {
        base_term
            .set_temperature_schedule(schedule)
            .map_err(py_value_error)?;
    }

    // Cold-start routing seed refinement (#629, #630). The cold residual-logit
    // seed is computed at the cold (shared-across-atoms) coordinates and cannot
    // separate planted disjoint atoms, so the joint solve starts in the
    // near-uniform routing saddle. Alternate the closed-form coordinate
    // projection (the #628 OOS mechanism) and the weighted LSQ decoder refit a
    // few times to place each row in the correct atom basin before the joint
    // Arrow-Schur fit. Gated to cold multi-atom softmax / IBP-MAP fits by the
    // caller; warm starts and JumpReLU keep their supplied seed.
    if seed_refine_routing
        && k_atoms > 1
        && matches!(assignment_kind.as_str(), "softmax" | "ibp_map")
    {
        sae_em_refine_routing_seed(
            &mut base_term,
            z_view,
            &basis_sizes,
            assignment_kind.as_str(),
            alpha,
            tau,
            jumprelu_threshold,
            seed_refine_random_state,
        )
        .map_err(py_value_error)?;
    }

    // WP-D → fit wiring (#980): if a per-row output-Fisher shard was supplied,
    // build the single `RowMetric::OutputFisher` and install it on the term via
    // `set_row_metric` *before* the objective consumes the term. This is the only
    // way the gauge acquires a non-identity weight, and it flips the provenance to
    // `OutputFisher` (the likelihood stays untouched — the inner product only
    // drives the isometry gauge, per `RowMetric::whitens_likelihood` == false for
    // `OutputFisher`). Magic-by-default: presence of `fisher_u` activates it; no
    // flag. The factor stack is `(n_obs, p_out, rank)` row-major, reshaped to the
    // `(n_obs, p_out * rank)` layout `RowMetric::output_fisher` expects
    // (`u[n, i * rank + k] = U[n, i, k]`). Shapes are validated at the boundary.
    let metric_provenance: &'static str = if let Some(u3) = fisher_u {
        let u_shape = u3.shape();
        if u_shape[0] != n_obs || u_shape[1] != p_out {
            return Err(py_value_error(format!(
                "sae_manifold_fit: fisher_factors U must be (n, p, r)=({n_obs}, {p_out}, r); \
                 got leading dims ({}, {})",
                u_shape[0], u_shape[1]
            )));
        }
        let rank = u_shape[2];
        if rank == 0 {
            return Err(py_value_error(
                "sae_manifold_fit: fisher_factors U rank (last axis) must be >= 1".to_string(),
            ));
        }
        if rank > p_out {
            return Err(py_value_error(format!(
                "sae_manifold_fit: fisher_factors U rank {rank} exceeds output dim p={p_out}"
            )));
        }
        if let Some(mr) = fisher_mass_residual.as_ref() {
            if mr.len() != n_obs {
                return Err(py_value_error(format!(
                    "sae_manifold_fit: fisher_factors mass_residual must be (n,)=({n_obs},); got \
                     length {}",
                    mr.len()
                )));
            }
        }
        // Flatten (n, p, r) row-major -> (n, p*r) with u[n, i*r + k] = U[n, i, k].
        let mut u_flat = Array2::<f64>::zeros((n_obs, p_out * rank));
        for row in 0..n_obs {
            for i in 0..p_out {
                for k in 0..rank {
                    u_flat[[row, i * rank + k]] = u3[[row, i, k]];
                }
            }
        }
        let metric =
            row_metric_from_fisher_provenance(u_flat, p_out, rank, fisher_provenance.as_deref())?;
        let label = metric_provenance_label(metric.provenance());
        base_term.set_row_metric(metric).map_err(py_value_error)?;
        label
    } else {
        "Euclidean"
    };

    // Per-row design-honesty reconstruction weights (#977). Installed on the
    // term before it is moved into the outer objective so the √w reweighting
    // is honored by both the inner joint fit and the outer ρ criterion. The
    // length / positivity contract is enforced inside `set_row_loss_weights`;
    // a uniform (or absent) vector self-normalizes to the unweighted path.
    if let Some(weights) = row_loss_weights {
        if weights.len() != n_obs {
            return Err(py_value_error(format!(
                "sae_manifold_fit: weights length {} must equal the {n_obs} response rows",
                weights.len()
            )));
        }
        base_term
            .set_row_loss_weights(weights.to_vec())
            .map_err(py_value_error)?;
    }

    let log_ard: Vec<Array1<f64>> = atom_dim
        .iter()
        .map(|&d| {
            if native_ard_enabled {
                Array1::<f64>::zeros(d)
            } else {
                Array1::<f64>::zeros(0)
            }
        })
        .collect();
    // Drive ρ (sparsity / smoothing λ's + per-atom ARD precisions) through the
    // one generic outer cascade — the same engine the GAM REML path uses
    // (`OuterProblem::run` → `plan()` → derivative-free / FD outer strategy).
    // `SaeManifoldOuterObjective::eval_cost` evaluates the term's true REML
    // criterion at each candidate ρ via an inner Arrow-Schur joint fit; the
    // engine selects ρ. No hand-rolled λ-grid, no manual `evidence_proxy` max:
    // the criterion + cascade subsume both, so smoothness/sparsity selection is
    // automatic (magic by default — no per-call grid kwarg).
    //
    // `init_rho` packs `[log_lambda_sparse, log_lambda_smooth, per-atom ARD]`;
    // its `to_flat()` defines the outer ρ vector the engine optimizes, and its
    // length is the objective's declared `n_params`. For learnable-alpha IBP,
    // the first coordinate is a dimensionless log-alpha offset rather than a
    // response-scale penalty strength, so assignment-aware seed scaling leaves it
    // unshifted while still scaling smoothness and ARD.
    let seed_dispersion = base_term
        .seed_reconstruction_dispersion(z_view)
        .map_err(py_value_error)?;
    let init_rho = SaeManifoldRho::new(sparsity_strength.ln(), smoothness.ln(), log_ard)
        .seed_scaled_by_dispersion_for_assignment(seed_dispersion, mode)
        .map_err(py_value_error)?;
    let init_rho_flat = init_rho.to_flat();
    let n_params = init_rho_flat.len();
    // Whether an isometry gauge penalty is installed on this fit. Read here,
    // before the registry is moved into the objective, and threaded into the
    // residual-gauge certificate below: an inactive isometry pin escalates the
    // certificate to the `diffeomorphism-unpinned` verdict.
    let isometry_pin_active = registry.penalties.iter().any(|p| {
        matches!(
            p,
            gam::terms::analytic_penalties::AnalyticPenaltyKind::Isometry(_)
        )
    });
    // Route every problem size through the full-batch objective on the owned
    // `target`: the inner Arrow-Schur fit materializes the `(N × M_total)`
    // basis, `(N × M_total × d)` jacobian, and `(N × K)` logit buffers in full,
    // so the outer-cascade entry point owns the full target verbatim.
    // `sae_streaming_plan` is exposed separately as a standalone diagnostic
    // pyfunction.
    let mut objective = gam::terms::sae::manifold::SaeManifoldOuterObjective::new(
        base_term,
        z_view.to_owned(),
        Some(registry),
        init_rho,
        max_iter,
        learning_rate,
        ridge_ext_coord,
        ridge_beta,
    );
    let problem =
        gam::solver::rho_optimizer::OuterProblem::new(n_params).with_initial_rho(init_rho_flat);
    problem
        .run(&mut objective, "SAE manifold")
        .map_err(estimation_error_to_pyerr)?;
    // Posterior shape uncertainty: per-atom φ-scaled decoder covariance and
    // ambient bands, read off the converged joint-Hessian Schur factor at the
    // settled ρ. Computed before `into_fitted` consumes the objective; reflects
    // the fitted (smooth) decoder shape, independent of any top-k assignment
    // gate applied below.
    let mut shape_uncertainty = objective
        .decoder_shape_uncertainty()
        .map_err(py_value_error)?;
    let (mut term, mut rho, loss) = objective.into_fitted();
    {
        let assignments = term.assignment.assignments();
        let fitted = term.fitted();
        term.record_fit_data_collapse_if_needed(
            z_view.view(),
            fitted.view(),
            assignments.view(),
            max_iter,
        )
        .map_err(py_value_error)?;
    }

    // #977 / #997 — evidence-guarded structure search around the production fit:
    // the genuine dictionary learner. Harvest deaths (diverged ARD ∪ terminal
    // collapse), fusions (co-activation), fission audits (absorption asymmetry),
    // and BIRTHS (whitened residual-factor subspace), then run the e-gated move
    // engine over a held-out estimation/evaluation row split. Every move — and in
    // particular every birth/fission that GROWS K — must pass the #984 held-out
    // e-value gate ([`run_atom_birth_gate`], invoked inside [`search`] for births,
    // fissions, and fusions) before it lands; deaths demote-never-reject. So K is
    // DISCOVERED from the data (grown by certified births, shrunk by demoted
    // deaths / fused atoms) rather than pinned at the user's input K, and the
    // returned `atoms`/`logits`/`assignments`/`atom_plans` reflect the discovered
    // K (the variable-K payload boundary below threads the post-search shape
    // through `from_payload`). The SearchLedger (+ the joint fit's collapse
    // events) is serialized onto the payload as the honesty surface — never a
    // silent restructure. Conservative by construction: only evidence-earning
    // atoms are born; the gates rarely certify, so the common all-contested case
    // returns the fit unchanged.
    //
    // Move budgets are magic-by-default — derived from the fitted dictionary, not
    // surfaced as user flags. The per-round breadth scales with the current atom
    // count so a small dictionary proposes few candidates and a large one a few
    // more, while `max_moves` (below) caps how many of those land per round and
    // `max_rounds` bounds the harvest→gate→refit loop. A genuine residual factor
    // earns its atom across rounds; a spurious one fails the held-out gate and is
    // recorded contested.
    let mut structure_ledger = gam::inference::structure_evidence::StructureLedger::new();
    let structure_search_json = {
        // Per-round harvest breadth derived from the fitted K (magic-by-default):
        // propose at most a handful of each move kind, scaled gently with the
        // dictionary size, with a small fixed floor so even a K=1 fit can grow
        // (the #1117 rank-revealing basis makes K>1 fits converge, so births no
        // longer hit the old K>1 wall). The e-gate, not these caps, decides what
        // lands; the caps only keep the proposal stream finite.
        let k_now = term.k_atoms().max(1);
        let births_per_round = (k_now + 1).min(4);
        let fissions_per_round = k_now.min(4);
        let harvest_params = gam::solver::structure_harvest::HarvestParams {
            max_fusions: 4,
            max_fissions: fissions_per_round,
            max_births: births_per_round,
        };
        let refit_params = gam::solver::structure_harvest::ProductionRefitParams {
            inner_max_iter: max_iter,
            learning_rate,
            ridge_ext_coord,
            ridge_beta,
        };
        // Moves that may LAND this round (accepted births/fissions/fusions +
        // demoted deaths); remaining proposals are recorded `Deferred` and
        // replayed next round. Sized to the current dictionary plus headroom for
        // the new birth/fission proposals so a single round can both prune a dead
        // atom and grow a genuinely-supported one. Magic-by-default — a function
        // of the fitted K, never a user flag.
        let max_moves = k_now + births_per_round + fissions_per_round;
        let budget = gam::solver::structure_search::MoveBudget {
            max_moves,
            alpha: 0.05,
        };
        let config = gam::solver::structure_harvest::RoundDriverConfig {
            n_shards: 4,
            budget,
            max_rounds: 3,
            harvest_params,
        };
        match gam::solver::structure_harvest::run_production_structure_search(
            term,
            rho,
            z_view.view(),
            config,
            refit_params,
            &mut structure_ledger,
        ) {
            Ok(result) => {
                term = result.term;
                rho = result.rho;
                gam::solver::structure_harvest::rounds_to_json(&result.rounds).ok()
            }
            Err(e) => {
                // Structure search is a post-fit audit pass; a failure must not
                // silently corrupt the fit — surface it loudly.
                return Err(py_value_error(format!(
                    "structure search around SAE fit failed: {e}"
                )));
            }
        }
    };

    // Clear any per-row estimation mask the structure-search refit left on the
    // adopted term so the returned `fitted` / dispersion / diagnostics are
    // computed over ALL rows (the mask is an internal split device, not a
    // property of the returned fit).
    term.clear_row_loss_weights();

    // #977 — VARIABLE-K boundary. The structure search above may have GROWN K
    // (certified births / fissions) or shrunk it (demoted deaths fold to ~0
    // routing — they keep their index but carry no mass). The returned payload
    // must reflect the DISCOVERED dictionary, not the user's input K, so every
    // downstream per-atom field is re-derived FROM THE FITTED TERM here. The
    // input `k_atoms` / `atom_basis` / `atom_dim` described the seed dictionary;
    // they are stale the moment a birth lands. `term.k_atoms()` is the source of
    // truth from this point on, and the per-atom basis-kind / active-dim vectors
    // are read off each fitted atom (born atoms inherit a template basis at
    // birth; their topology is then adjudicated by the post-fit curved-vs-linear
    // hybrid-split pass — see `set_atom_inner_fits` below — and reported on the
    // payload's `hybrid_split` key, so the dictionary is genuinely heterogeneous
    // rather than all-circle).
    let k_atoms = term.k_atoms();
    let atom_basis: Vec<String> = term
        .atoms
        .iter()
        .map(|atom| sae_atom_basis_kind_name(&atom.basis_kind))
        .collect();
    let atom_dim: Vec<usize> = term.atoms.iter().map(|atom| atom.latent_dim).collect();

    term.set_certificate_dispersion(shape_uncertainty.dispersion)
        .map_err(py_value_error)?;

    // #1097 / #1103: harvest each atom's fixed inner-decoder-smooth snapshot
    // (design, derivative design, penalized inner Hessian, per-row Gaussian
    // scores, roughness Gram, peak/mode design rows) at the settled state, so
    // the diagnostics report can produce per-atom Riesz-debiased functionals and
    // the split-LRT smooth-structure e-value. Needs the reconstruction target `Z` (dropped
    // from the objective at fit end) and the fitted dispersion, both available
    // here. A degenerate atom (no active rows / non-SPD inner Hessian) yields a
    // `None` slot inside; a structural shape inconsistency surfaces loudly.
    term.set_atom_inner_fits(z_view.view(), &rho, shape_uncertainty.dispersion)
        .map_err(py_value_error)?;

    // #977 — COMPLETE the per-atom shape band for any structure-search-BORN atom
    // (index ≥ the seed K the Schur factor was assembled at). `shape_uncertainty`
    // above was read off the PRE-search joint-Hessian Schur factor, so a born atom
    // has no Schur block and would be reported with NO uncertainty band — a silent
    // gap. This fills that atom's band from its OWN fitted penalized inner Hessian
    // `H_k = Φ_kᵀ W_k Φ_k + S̃_k` (harvested by `set_atom_inner_fits` for every
    // atom): the principled per-atom Laplace band `Var_c(t) = φ · Φ_k(t)ᵀ H_k⁻¹
    // Φ_k(t)`. Seed atoms keep their exact joint-Hessian band (untouched); only
    // missing bands are filled, and a degenerate born atom (no active rows /
    // non-SPD inner Hessian) keeps an honest NaN band rather than a fabricated one.
    // After this, NO post-search atom is reported without an uncertainty band.
    term.complete_born_atom_shape_bands(&mut shape_uncertainty)
        .map_err(py_value_error)?;

    // Additive post-fit diagnostics (#980): the two-score per-atom lens
    // (presence / behavioral coupling / discrepancy) and the residual-gauge
    // certificate. Both read the fitted term + its single per-row metric; under a
    // Euclidean / no-harvest provenance the lens coupling is `None` and the gauge
    // is certified under Euclidean provenance — never an error, never flag-gated.
    // Per-atom ARD variances (∝ exp(−log_precision); equality-preserving for the
    // certificate's equal-ARD-rotation detection) are threaded in when native ARD
    // was enabled, else `None` per atom.
    let ard_variances: Vec<Option<Array1<f64>>> = rho
        .log_ard
        .iter()
        .map(|log_prec| {
            if log_prec.is_empty() {
                None
            } else {
                Some(log_prec.mapv(|lp| (-lp).exp()))
            }
        })
        .collect();
    let fit_diagnostics = term
        .fit_diagnostics_report(
            Some(&ard_variances),
            isometry_pin_active,
            Some(shape_uncertainty.dispersion),
        )
        .map_err(py_value_error)?;

    let mut assignments = term.assignment.assignments();
    let mut fitted = term.fitted();
    // Apply hard top-k projection per row, then recompute `fitted` from the
    // projected assignments so the returned `assignments` and `fitted` stay
    // mutually consistent (i.e. `fitted == sum_k a_k * decoder_k @ basis_k`).
    // Smooth softmax (or IBP/JumpReLU) drives optimisation; the hard top-k
    // gate is applied at inference time. For softmax mode the kept entries
    // are renormalised so the resulting per-row distribution sums to 1; for
    // the other modes the kept entries retain their unnormalised values
    // (those modes' assignments are not probability distributions).
    if let Some(k_top) = top_k {
        if k_top < k_atoms {
            let n_obs_local = z_view.nrows();
            let p_out_local = z_view.ncols();
            let renormalise = assignment_kind == "softmax";
            for row in 0..n_obs_local {
                // Collect (value, atom_idx) pairs; pick the indices of the
                // largest k_top values. Ties broken by lower atom index.
                let mut paired: Vec<(f64, usize)> = (0..k_atoms)
                    .map(|atom_idx| (assignments[[row, atom_idx]], atom_idx))
                    .collect();
                paired.sort_by(|a, b| {
                    b.0.partial_cmp(&a.0)
                        .unwrap_or(std::cmp::Ordering::Equal)
                        .then(a.1.cmp(&b.1))
                });
                let mut keep = vec![false; k_atoms];
                for &(_, atom_idx) in paired.iter().take(k_top) {
                    keep[atom_idx] = true;
                }
                if renormalise {
                    let mut kept_sum = 0.0_f64;
                    for atom_idx in 0..k_atoms {
                        if keep[atom_idx] {
                            kept_sum += assignments[[row, atom_idx]];
                        }
                    }
                    if kept_sum > 0.0 {
                        for atom_idx in 0..k_atoms {
                            assignments[[row, atom_idx]] = if keep[atom_idx] {
                                assignments[[row, atom_idx]] / kept_sum
                            } else {
                                0.0
                            };
                        }
                    } else {
                        // Pathological case: all kept entries are zero. Fall
                        // back to uniform mass over the kept indices so the
                        // contract `assignments.sum(axis=1) == 1` still holds.
                        let inv = 1.0 / (k_top as f64);
                        for atom_idx in 0..k_atoms {
                            assignments[[row, atom_idx]] = if keep[atom_idx] { inv } else { 0.0 };
                        }
                    }
                } else {
                    for atom_idx in 0..k_atoms {
                        if !keep[atom_idx] {
                            assignments[[row, atom_idx]] = 0.0;
                        }
                    }
                }
            }
            // Recompute `fitted` from the projected assignments. Mirrors the
            // body of `SaeManifoldTerm::try_fitted` but uses our projected
            // matrix instead of the logit-derived row distribution.
            fitted = Array2::<f64>::zeros((n_obs_local, p_out_local));
            let mut g_buf = vec![0.0_f64; p_out_local];
            for row in 0..n_obs_local {
                for atom_idx in 0..k_atoms {
                    let a_k = assignments[[row, atom_idx]];
                    if a_k == 0.0 {
                        continue;
                    }
                    term.atoms[atom_idx].fill_decoded_row(row, &mut g_buf);
                    let mut out_row = fitted.row_mut(row);
                    for out_col in 0..p_out_local {
                        out_row[out_col] += a_k * g_buf[out_col];
                    }
                }
            }
        }
    }
    term.record_fit_data_collapse_if_needed(
        z_view.view(),
        fitted.view(),
        assignments.view(),
        max_iter,
    )
    .map_err(py_value_error)?;
    let trust_diagnostics = term
        .trust_diagnostics_report(assignments.view())
        .map_err(py_value_error)?;
    let log_ard_py = PyList::empty(py);
    for atom_log_ard in &rho.log_ard {
        log_ard_py.append(atom_log_ard.clone().into_pyarray(py))?;
    }
    let atoms_py = PyList::empty(py);
    for atom_idx in 0..k_atoms {
        let atom = &term.atoms[atom_idx];
        let atom_dict = PyDict::new(py);
        atom_dict.set_item(
            "decoder_B",
            atom.decoder_coefficients.clone().into_pyarray(py),
        )?;
        atom_dict.set_item("basis_kind", atom_basis[atom_idx].clone())?;
        atom_dict.set_item("basis_centers", py.None())?;
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
                atom_dict.set_item("decoder_covariance", cov.clone().into_pyarray(py))?;
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

    let active_mask: Vec<bool> = (0..k_atoms)
        .map(|atom_idx| assignments.column(atom_idx).sum() > 1.0e-8)
        .collect();
    let out = PyDict::new(py);
    out.set_item("atoms", atoms_py)?;
    out.set_item("assignments_z", assignments.into_pyarray(py))?;
    out.set_item("logits", term.assignment.logits.clone().into_pyarray(py))?;
    out.set_item("atom_active_mask", active_mask)?;
    out.set_item("fitted", fitted.into_pyarray(py))?;
    out.set_item("reml_score", loss.evidence_proxy())?;
    let reported_log_alpha = match term.assignment.mode {
        gam::terms::sae::manifold::AssignmentMode::IBPMap { alpha, .. } => alpha.ln(),
        _ => alpha.ln(),
    };
    out.set_item("log_alpha", reported_log_alpha)?;
    out.set_item("log_lambda_smooth", rho.log_lambda_smooth)?;
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
    // #1154 — the co-trained amortized-encoder report. The outer ρ-cascade ranked
    // ρ by the co-trained criterion (REML + amortized-encoder consistency), and
    // the inner solve was warm-started from the amortized encoder built on the
    // running dictionary (Design A). Here we surface, at the settled dictionary,
    // how faithfully and certifiably the cheap one-mat-vec encoder inverts it:
    //   * `recon_consistency` — mean per-element squared gap between the amortized
    //     reconstruction and the exact encode-by-inner-solve reconstruction (0 ⇒
    //     the IFT predictor is an exact first-order model of the encode map);
    //   * `uncertified_fraction` — share of (row, atom) amortized encodes whose
    //     Kantorovich certificate failed and fell back to the exact Newton (the
    //     encoder's certifiable coverage of the fitted dictionary).
    // Best-effort: a degenerate atlas (no usable charts) yields no report rather
    // than aborting the fit payload (the co-training fold itself is advisory).
    if let Ok(consistency) = term.amortized_encoder_consistency(z_view.view(), &rho) {
        let cotrain = PyDict::new(py);
        cotrain.set_item("recon_consistency", consistency.recon_consistency)?;
        cotrain.set_item("uncertified_fraction", consistency.uncertified_fraction)?;
        cotrain.set_item("n_uncertified", consistency.n_uncertified)?;
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
    // #16 — ONE coherent certificate ledger. Every certificate this fit produced
    // implements the shared claim+evidence+conservative-verdict contract
    // ([`gam::inference::certificates::Certificate`]); the ledger folds them into
    // a single inspectable block keyed by claim id, with a top-level conservative
    // `overall` roll-up (the weakest member). This consolidates the scattered
    // per-feature reads — the bespoke `residual_gauge` / `incoherence_report`
    // keys above are KEPT working (same values) for back-compat, and the ledger
    // is the additive, canonical surface. Verdicts cannot read stronger than
    // their evidence: an absent or below-margin certificate is `unavailable` /
    // `insufficient`, never a silent pass.
    {
        use gam::inference::certificates::CertificateLedger;
        let mut ledger = CertificateLedger::new();
        ledger.record(&fit_diagnostics.residual_gauge);
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
        let basis_size = atom.basis_size();
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
                entry.set_item("duchon_centers", py.None())?;
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
    let structure_certificate = structure_ledger.certify(0.05);
    let certificate_json =
        serde_json::to_string(&structure_certificate).map_err(serde_json_error_to_pyerr)?;
    out.set_item("structure_certificate", certificate_json)?;
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
/// single inspectable certificate artifact; it replaces reading the scattered
/// per-feature keys (which are kept working alongside it for back-compat).
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
        // Per-atom held-out leave-one-atom-out explained-variance contribution
        // `ΔEV_k = EV(full) − EV(full∖{k})` — the EV axis of the #1026 (Θ, ΔEV)
        // frontier. Computed during fit (target supplied) but previously not
        // surfaced to Python; exposing it makes the discriminating measurement
        // queryable as structured data rather than a transient log line.
        match verdict.held_out_delta_ev {
            Some(dev) => a.set_item("held_out_delta_ev", dev)?,
            None => a.set_item("held_out_delta_ev", py.None())?,
        }
        atoms.append(a)?;
    }
    d.set_item("atoms", atoms)?;
    Ok(d)
}

/// Serialize one Riesz-debiased smooth-functional report (#1097): the plug-in
/// estimate, the one-step penalty-debiased estimate, its influence-based SE, the
/// removed penalty bias, and a 95% Wald interval on the debiased estimate.
fn sae_riesz_report_dict<'py>(
    py: Python<'py>,
    report: &gam::inference::riesz::RieszDebiasReport,
) -> PyResult<Bound<'py, PyDict>> {
    let d = PyDict::new(py);
    d.set_item("theta_plugin", report.theta_plugin)?;
    d.set_item("theta_onestep", report.theta_onestep)?;
    d.set_item("se", report.se)?;
    d.set_item("penalty_bias", report.penalty_bias)?;
    // 95% Wald interval on the debiased estimate (z = 1.959963985 for 0.975).
    let z = 1.959_963_984_540_054_f64;
    d.set_item("ci_lo", report.theta_onestep - z * report.se)?;
    d.set_item("ci_hi", report.theta_onestep + z * report.se)?;
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
    )
}

/// Per-atom basis spec used by [`sae_manifold_build_inputs`] to assemble the
/// padded `(K, N, M_max)` design plus jacobian, smoothness penalty stack, and
/// per-atom `basis_sizes`. The Python wrapper passes only `(z, atom_basis,
/// atom_dim)`; this Rust helper picks `n_harmonics` for periodic atoms and
/// samples Duchon centers deterministically from the PCA seed.
#[derive(Debug, Clone)]
struct SaeAtomBuildPlan {
    kind: SaeAtomBasisKind,
    latent_dim: usize,
    n_harmonics: usize,
    duchon_centers: Option<Array2<f64>>,
    basis_size: usize,
}

fn sae_atom_basis_size(plan: &SaeAtomBuildPlan) -> usize {
    plan.basis_size
}

const SAE_MAX_PERIODIC_HARMONICS: usize = 4096;

fn sae_periodic_basis_size(n_harmonics: usize) -> Result<usize, String> {
    if n_harmonics > SAE_MAX_PERIODIC_HARMONICS {
        return Err(format!(
            "sae_build_periodic_atom: n_harmonics={n_harmonics} exceeds dense limit {SAE_MAX_PERIODIC_HARMONICS}"
        ));
    }
    n_harmonics
        .checked_mul(2)
        .and_then(|twice| twice.checked_add(1))
        .ok_or_else(|| {
            format!("sae_build_periodic_atom: basis size overflows for n_harmonics={n_harmonics}")
        })
}

/// Auto-derive whether the SAE joint fit should stream, and the row-chunk size.
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

/// Cross-checkpoint Riesz-debiased atom-trajectory dynamics (issue #1102).
///
/// `decoder_grid` is `[n_checkpoints, n_atoms, n_grid, ambient_dim]`: every
/// atom's decoder curve sampled on the shared `latent_grid` at every checkpoint.
/// `checkpoint_ids` / `atom_names` label the checkpoint and atom axes (lengths
/// must match the corresponding grid axes). For each atom this walks consecutive
/// checkpoints and, per step `c → c+1`:
///   1. fits the inter-checkpoint transport map (checkpoint axis reused as the
///      transport "layer" axis);
///   2. evaluates the Riesz penalty-debiased decoder-displacement contrast
///      `‖g^{(c+1)}(t_mode) − g^{(c)}(t_mode)‖` with a delta-method SE;
///   3. accumulates an anytime-valid e-process under the no-change null.
///
/// Returns `{"trajectories": [ {atom_name,
/// conditional_step_contrasts:[...riesz...], transports:[...transport...],
/// change_evidence:{...certificate...}} ]}`. The PRIMARY, coverage-valid
/// deliverable is `change_evidence` (the anytime-valid change e-process);
/// `conditional_step_contrasts` is a descriptive readout CONDITIONAL ON THE
/// FITTED COORDINATES (its SE conditions away the generated-regressor
/// uncertainty in t̂/â — issue #1115), not a coverage-valid CI. See
/// `gam::inference::checkpoint_dynamics` for the estimator and the honest
/// accounting of which Riesz inputs a bare decoder grid supports.
#[pyfunction(signature = (decoder_grid, checkpoint_ids, atom_names, latent_grid, alpha = 0.05))]
fn sae_checkpoint_dynamics(
    py: Python<'_>,
    decoder_grid: PyReadonlyArray4<'_, f64>,
    checkpoint_ids: Vec<String>,
    atom_names: Vec<String>,
    latent_grid: PyReadonlyArray1<'_, f64>,
    alpha: f64,
) -> PyResult<Py<PyDict>> {
    use gam::inference::checkpoint_dynamics::{CheckpointDynamicsInput, checkpoint_atom_dynamics};
    if !(alpha > 0.0 && alpha < 1.0) {
        return Err(PyValueError::new_err(format!(
            "sae_checkpoint_dynamics: alpha must be in (0, 1), got {alpha}"
        )));
    }
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
        let steps = PyList::empty(py);
        for report in &traj.conditional_step_contrasts {
            steps.append(sae_riesz_report_dict(py, report)?)?;
        }
        t.set_item("conditional_step_contrasts", steps)?;
        let transports = PyList::empty(py);
        for report in &traj.transports {
            transports.append(layer_transport_report_to_pydict(py, report)?)?;
        }
        t.set_item("transports", transports)?;
        // Anytime-valid change evidence: the e-BH certificate of the atom's
        // change e-process at level `alpha`, one entry per consecutive-step
        // claim with its accumulated log-evidence and confirmation verdict.
        let certificate = traj.change_evidence.certify(alpha);
        let ev = PyDict::new(py);
        ev.set_item("alpha", certificate.alpha)?;
        let entries = PyList::empty(py);
        for entry in &certificate.entries {
            let e = PyDict::new(py);
            let label = match &entry.kind {
                gam::inference::structure_evidence::ClaimKind::Custom { label } => label.clone(),
                other => format!("{other:?}"),
            };
            e.set_item("claim", label)?;
            e.set_item("log_e", entry.log_e)?;
            e.set_item("steps", entry.steps)?;
            e.set_item("confirmed", entry.confirmed)?;
            entries.append(e)?;
        }
        ev.set_item("entries", entries)?;
        t.set_item("change_evidence", ev)?;
        traj_list.append(t)?;
    }
    out.set_item("trajectories", traj_list)?;
    Ok(out.unbind())
}

/// PCA seed: returns coords with shape `(k_atoms, n_obs, d_max)`. For periodic
/// atoms, column 0 is `atan2(Z·v2, Z·v1) / (2π)` (per-atom v2 picked from PCs)
/// and remaining columns are min-max normalized projections onto subsequent
/// PCs. For non-periodic atoms, the first `d_k` columns are min-max normalized
/// `U·diag(s)` from the thin SVD of the centered response.
/// Seed each atom's decoder coefficient block via a joint ridge-regularized
/// least-squares projection of `Z` onto the atom design `[a_init * Phi_1, ...,
/// a_init * Phi_K]`, where `a_init` is the assignment map that the inner Newton
/// driver will produce at iteration 0 from the supplied `initial_logits`.
/// IBP-MAP uses the base `alpha` because learnable-alpha fits start with
/// `rho0 = 0`, and JumpReLU uses the configured hard threshold.
///
/// Zero-initialised decoder coefficients leave the joint-fit Arrow-Schur
/// system in a degenerate fixed point on multi-atom configurations: the
/// data-fit Jacobian, the assignment-weighted decoder gradient, and the
/// sparsity-prior gradient cannot all be zero simultaneously, but the
/// assignment prior (IBP-MAP stick-breaking or softmax entropy) is the only
/// term with a non-zero gradient at iter 0. The optimizer then collapses the
/// assignments to zero before any data signal has accumulated, even on
/// trivially-separable signals such as the K=2 periodic torus reproducer in
/// issue #174. Seeding with a closed-form LSQ projection eliminates the
/// degeneracy: at iter 0 the residual already carries the data information
/// the atoms need, so the assignment update has both a sparsity-prior pull
/// and a data-fit push to balance against.
///
/// Returns the padded `(K, M_max, p_out)` decoder array directly.
///
/// Build a data-driven asymmetric assignment-logit seed for a cold start
/// (issue #629). A uniform logit seed (`Array2::zeros`) is an exact symmetric
/// saddle of the joint objective whenever the atoms are exchangeable under the
/// assignment forward map: every atom carries identical responsibility, the
/// LSQ decoder init projects the same target onto every atom, and the
/// assignment update has no gradient to break the tie, so the fit never routes.
/// The tiny `random_state` jitter is too weak to escape on conditioned data.
///
/// This helper runs one EM-style M-then-E step on the seed geometry: it fits
/// each atom's decoder independently against the *full* response (each atom's
/// own seed coordinates already give it a distinct `Phi_k`), measures the
/// per-row reconstruction residual under that fit, and emits mean-centred logits
/// that prefer the atom which best explains each row. Rows that every atom
/// explains equally well land at exactly zero logits (the residual ties centre
/// to the neutral state), so the existing jitter still breaks those rare ties;
/// rows with a clear best atom get a decisive — but bounded, hence escapable by
/// the Newton refinement — head start. The mean-centring is translation-identity
/// for softmax and keeps the IBP-MAP `sigmoid(logit/τ)` gate neutral (0.5) on
/// ties instead of slamming both gates shut, so the seed is safe for both
/// assignment maps. The result is a proper responsibility seed rather than a
/// saddle.
fn sae_residual_seed_logits(
    basis_values: ArrayView3<'_, f64>,
    basis_sizes: &[usize],
    z: ArrayView2<'_, f64>,
    gain: f64,
) -> Result<Array2<f64>, String> {
    let k_atoms = basis_sizes.len();
    let (n_obs, p_out) = z.dim();
    let mut logits = Array2::<f64>::zeros((n_obs, k_atoms));
    if n_obs == 0 || p_out == 0 || k_atoms <= 1 {
        return Ok(logits);
    }
    if basis_values.shape()[0] != k_atoms || basis_values.shape()[1] != n_obs {
        return Err(format!(
            "sae_residual_seed_logits: basis_values must start with (K, N)=({k_atoms}, {n_obs}); got {:?}",
            basis_values.shape()
        ));
    }
    let z_owned = z.to_owned();
    // Per-row residual energy after fitting each atom independently.
    let mut resid = Array2::<f64>::zeros((n_obs, k_atoms));
    for atom_idx in 0..k_atoms {
        let m_k = basis_sizes[atom_idx];
        if m_k == 0 {
            // No basis columns: the atom predicts zero, so its residual is the
            // full row energy. Leave `resid` column at that value below.
            for row in 0..n_obs {
                let mut e = 0.0_f64;
                for col in 0..p_out {
                    e += z[[row, col]] * z[[row, col]];
                }
                resid[[row, atom_idx]] = e;
            }
            continue;
        }
        // Phi_k = basis_values[atom_idx, :, :m_k]  (N, m_k).
        let mut phi = Array2::<f64>::zeros((n_obs, m_k));
        for row in 0..n_obs {
            for c in 0..m_k {
                phi[[row, c]] = basis_values[[atom_idx, row, c]];
            }
        }
        let mut gram = fast_ata(&phi);
        let mut trace = 0.0_f64;
        for i in 0..m_k {
            trace += gram[[i, i]];
        }
        let jitter = (trace / m_k as f64).max(1.0).max(1.0e-12) * 1.0e-8;
        for i in 0..m_k {
            gram[[i, i]] += jitter;
        }
        let rhs = fast_atb(&phi, &z_owned);
        let factor = gram
            .cholesky(Side::Lower)
            .map_err(|err| format!("sae_residual_seed_logits: Cholesky failed: {err:?}"))?;
        let b_k = factor.solve_mat(&rhs); // (m_k, p_out)
        if !b_k.iter().all(|v| v.is_finite()) {
            return Err("sae_residual_seed_logits: non-finite LSQ solution".to_string());
        }
        let fitted = phi.dot(&b_k); // (N, p_out)
        for row in 0..n_obs {
            let mut e = 0.0_f64;
            for col in 0..p_out {
                let d = z[[row, col]] - fitted[[row, col]];
                e += d * d;
            }
            resid[[row, atom_idx]] = e;
        }
    }
    // Convert per-row residuals to mean-centred logits relative to each row's
    // own scale so the head start is dimensionless. The best atom (lowest
    // residual) gets a positive logit, the worst a negative one, and a row whose
    // atoms all explain it equally well lands at exactly zero. Normalise by the
    // row's mean residual across atoms (with a floor relative to the dataset to
    // keep near-zero-energy rows well posed) so the spread is O(gain) regardless
    // of output magnitude.
    //
    // The mean-centring is what keeps the seed safe across assignment maps.
    // Softmax is translation-invariant, so subtracting the per-row mean leaves
    // it bit-identical to the raw `-gain·r/m` form. IBP-MAP, by contrast, maps
    // each logit through an unnormalised `sigmoid(logit/τ)`: an *uncentred*
    // negative-only seed would push *every* gate below 0.5 and slam a tied row
    // shut (`sigmoid(-gain/τ)≈0`), which is worse than the neutral 0.5/0.5 state
    // the uniform saddle held. Centring restores `logit=0 ⇒ gate=0.5` on ties
    // and opens the gate (`logit>0`) only for atoms that beat the row mean.
    let mut global_mean = 0.0_f64;
    for row in 0..n_obs {
        for k in 0..k_atoms {
            global_mean += resid[[row, k]];
        }
    }
    global_mean /= (n_obs * k_atoms) as f64;
    let floor = (global_mean * 1.0e-6).max(1.0e-12);
    for row in 0..n_obs {
        let mut row_mean = 0.0_f64;
        for k in 0..k_atoms {
            row_mean += resid[[row, k]];
        }
        row_mean = (row_mean / k_atoms as f64).max(floor);
        for k in 0..k_atoms {
            logits[[row, k]] = -gain * (resid[[row, k]] - row_mean) / row_mean;
        }
    }
    Ok(logits)
}

fn sae_output_energy_cluster_labels(z: ArrayView2<'_, f64>, k_atoms: usize) -> Vec<usize> {
    let (n_obs, p_out) = z.dim();
    let mut labels = vec![0usize; n_obs];
    if n_obs == 0 || p_out == 0 || k_atoms <= 1 {
        return labels;
    }
    let mut features = Array2::<f64>::zeros((n_obs, p_out));
    let mut row_energy = vec![0.0_f64; n_obs];
    for row in 0..n_obs {
        let mut energy = 0.0_f64;
        for col in 0..p_out {
            let value = z[[row, col]];
            energy += value * value;
        }
        row_energy[row] = energy;
        let denom = energy.max(1.0e-12);
        for col in 0..p_out {
            let value = z[[row, col]];
            features[[row, col]] = value * value / denom;
        }
    }

    let mut centers = Array2::<f64>::zeros((k_atoms, p_out));
    let first = row_energy
        .iter()
        .enumerate()
        .max_by(|a, b| a.1.partial_cmp(b.1).unwrap_or(std::cmp::Ordering::Equal))
        .map(|(idx, _)| idx)
        .unwrap_or(0);
    centers.row_mut(0).assign(&features.row(first));
    let mut min_dist = vec![0.0_f64; n_obs];
    for row in 0..n_obs {
        let mut dist = 0.0_f64;
        for col in 0..p_out {
            let diff = features[[row, col]] - centers[[0, col]];
            dist += diff * diff;
        }
        min_dist[row] = dist;
    }
    for atom_idx in 1..k_atoms {
        let next = min_dist
            .iter()
            .enumerate()
            .max_by(|a, b| a.1.partial_cmp(b.1).unwrap_or(std::cmp::Ordering::Equal))
            .map(|(idx, _)| idx)
            .unwrap_or(0);
        centers.row_mut(atom_idx).assign(&features.row(next));
        for row in 0..n_obs {
            let mut dist = 0.0_f64;
            for col in 0..p_out {
                let diff = features[[row, col]] - centers[[atom_idx, col]];
                dist += diff * diff;
            }
            if dist < min_dist[row] {
                min_dist[row] = dist;
            }
        }
    }

    for _ in 0..20 {
        let mut changed = false;
        for row in 0..n_obs {
            let mut best_atom = 0usize;
            let mut best_dist = f64::INFINITY;
            for atom_idx in 0..k_atoms {
                let mut dist = 0.0_f64;
                for col in 0..p_out {
                    let diff = features[[row, col]] - centers[[atom_idx, col]];
                    dist += diff * diff;
                }
                if dist < best_dist {
                    best_dist = dist;
                    best_atom = atom_idx;
                }
            }
            if labels[row] != best_atom {
                labels[row] = best_atom;
                changed = true;
            }
        }
        if !changed {
            break;
        }

        centers.fill(0.0);
        let mut counts = vec![0usize; k_atoms];
        for row in 0..n_obs {
            let atom_idx = labels[row];
            counts[atom_idx] += 1;
            for col in 0..p_out {
                centers[[atom_idx, col]] += features[[row, col]];
            }
        }
        for atom_idx in 0..k_atoms {
            if counts[atom_idx] == 0 {
                let row = atom_idx % n_obs;
                centers.row_mut(atom_idx).assign(&features.row(row));
                continue;
            }
            let inv = 1.0 / counts[atom_idx] as f64;
            for col in 0..p_out {
                centers[[atom_idx, col]] *= inv;
            }
        }
    }
    labels
}

fn sae_refine_periodic_seed_coords_by_cluster(
    z: ArrayView2<'_, f64>,
    plans: &[SaeAtomBuildPlan],
    labels: &[usize],
    seed_coords: &mut Array3<f64>,
) -> Result<(), String> {
    let (n_obs, p_out) = z.dim();
    let k_atoms = plans.len();
    if labels.len() != n_obs {
        return Err(format!(
            "sae_refine_periodic_seed_coords_by_cluster: labels length {} must equal n_obs={n_obs}",
            labels.len()
        ));
    }
    if n_obs < 2 || p_out < 2 || k_atoms <= 1 {
        return Ok(());
    }
    for (atom_idx, plan) in plans.iter().enumerate() {
        if !matches!(plan.kind, SaeAtomBasisKind::Periodic) {
            continue;
        }
        let rows: Vec<usize> = (0..n_obs).filter(|&row| labels[row] == atom_idx).collect();
        if rows.len() < 2 {
            continue;
        }
        let mut mean = Array1::<f64>::zeros(p_out);
        for &row in &rows {
            for col in 0..p_out {
                mean[col] += z[[row, col]];
            }
        }
        let inv_count = 1.0 / rows.len() as f64;
        for col in 0..p_out {
            mean[col] *= inv_count;
        }

        let mut local = Array2::<f64>::zeros((rows.len(), p_out));
        for (out_row, &src_row) in rows.iter().enumerate() {
            for col in 0..p_out {
                local[[out_row, col]] = z[[src_row, col]] - mean[col];
            }
        }
        let (_u_opt, _s_vals, vt_opt) = local.svd(false, true).map_err(|err| {
            format!("sae_refine_periodic_seed_coords_by_cluster: SVD failed: {err:?}")
        })?;
        let vt = vt_opt.ok_or_else(|| {
            "sae_refine_periodic_seed_coords_by_cluster: SVD returned no Vt".to_string()
        })?;
        if vt.nrows() < 2 {
            continue;
        }
        let pc1 = vt.row(0);
        let pc2 = vt.row(1);
        let two_pi = std::f64::consts::TAU;
        for row in 0..n_obs {
            let mut a = 0.0_f64;
            let mut b = 0.0_f64;
            for col in 0..p_out {
                let centered = z[[row, col]] - mean[col];
                a += centered * pc1[col];
                b += centered * pc2[col];
            }
            let phase = b.atan2(a) / two_pi;
            seed_coords[[atom_idx, row, 0]] = phase - phase.floor();
        }
    }
    Ok(())
}

fn sae_decoder_lsq_init(
    basis_values: ArrayView3<'_, f64>,
    basis_sizes: &[usize],
    z: ArrayView2<'_, f64>,
    initial_logits: ArrayView2<'_, f64>,
    assignment_kind: &str,
    alpha: f64,
    tau: f64,
    jumprelu_threshold: f64,
) -> Result<Array3<f64>, String> {
    let k_atoms = basis_sizes.len();
    let (n_obs, p_out) = z.dim();
    let m_max = basis_sizes.iter().copied().max().unwrap_or(1).max(1);
    let mut out = Array3::<f64>::zeros((k_atoms, m_max, p_out));
    if n_obs == 0 || p_out == 0 || k_atoms == 0 {
        return Ok(out);
    }
    if basis_values.shape()[0] != k_atoms || basis_values.shape()[1] != n_obs {
        return Err(format!(
            "sae_decoder_lsq_init: basis_values must start with (K, N)=({k_atoms}, {n_obs}); got {:?}",
            basis_values.shape()
        ));
    }
    if initial_logits.dim() != (n_obs, k_atoms) {
        return Err(format!(
            "sae_decoder_lsq_init: initial_logits must be ({n_obs}, {k_atoms}); got {:?}",
            initial_logits.dim()
        ));
    }
    if !tau.is_finite() || tau <= 0.0 {
        return Err(format!(
            "sae_decoder_lsq_init: tau must be finite and positive; got {tau}"
        ));
    }
    // Compute per-row, per-atom assignment weight a_init that matches the
    // forward map of `assignment_kind` evaluated at `initial_logits`.
    let mut a_init = Array2::<f64>::zeros((n_obs, k_atoms));
    match assignment_kind {
        "softmax" => {
            let inv_tau = 1.0 / tau;
            for row in 0..n_obs {
                let mut max_logit = f64::NEG_INFINITY;
                for k in 0..k_atoms {
                    let v = initial_logits[[row, k]];
                    if v > max_logit {
                        max_logit = v;
                    }
                }
                let mut sum = 0.0_f64;
                let mut buf = vec![0.0_f64; k_atoms];
                for k in 0..k_atoms {
                    let v = ((initial_logits[[row, k]] - max_logit) * inv_tau).exp();
                    buf[k] = v;
                    sum += v;
                }
                if sum > 0.0 && sum.is_finite() {
                    for k in 0..k_atoms {
                        a_init[[row, k]] = buf[k] / sum;
                    }
                }
            }
        }
        "ibp_map" => {
            if !alpha.is_finite() || alpha <= 0.0 {
                return Err(format!(
                    "sae_decoder_lsq_init: alpha must be finite and positive for IBP-MAP; got {alpha}"
                ));
            }
            // Use the base alpha here. In learnable-alpha fits the first rho
            // coordinate starts at zero, so alpha_eff = alpha at initialization.
            for row in 0..n_obs {
                let weights =
                    gam::terms::sae::manifold::ibp_map_row(initial_logits.row(row), tau, alpha);
                for k in 0..k_atoms {
                    a_init[[row, k]] = weights[k];
                }
            }
        }
        "jumprelu" => {
            if !jumprelu_threshold.is_finite() {
                return Err(format!(
                    "sae_decoder_lsq_init: jumprelu_threshold must be finite; got {jumprelu_threshold}"
                ));
            }
            for row in 0..n_obs {
                let weights = gam::terms::sae::manifold::jumprelu_row(
                    initial_logits.row(row),
                    tau,
                    jumprelu_threshold,
                );
                for k in 0..k_atoms {
                    a_init[[row, k]] = weights[k];
                }
            }
        }
        other => {
            return Err(format!(
                "sae_decoder_lsq_init: unsupported assignment_kind {other:?}"
            ));
        }
    }
    // Build joint design X = [a_init[:,0] * Phi_1 | ... | a_init[:,K-1] * Phi_K]
    // with column count M_total = sum_k basis_sizes[k]. If every atom has zero
    // weight on a row, that row contributes nothing — but with all the
    // supported initial logits we use, a_init has at least one non-zero
    // column per row. Solve (X^T X + ridge I) B = X^T Z, then split.
    let offsets: Vec<usize> = {
        let mut acc = 0usize;
        let mut v = Vec::with_capacity(k_atoms + 1);
        v.push(0);
        for &m in basis_sizes {
            acc += m;
            v.push(acc);
        }
        v
    };
    let m_total = offsets[k_atoms];
    if m_total == 0 {
        return Ok(out);
    }
    let mut x = Array2::<f64>::zeros((n_obs, m_total));
    for atom_idx in 0..k_atoms {
        let m_k = basis_sizes[atom_idx];
        let off = offsets[atom_idx];
        for row in 0..n_obs {
            let w = a_init[[row, atom_idx]];
            if w == 0.0 {
                continue;
            }
            for basis_col in 0..m_k {
                x[[row, off + basis_col]] = w * basis_values[[atom_idx, row, basis_col]];
            }
        }
    }
    // Symmetric normal-equations matrix and rhs.
    let mut xtx = fast_ata(&x);
    // Diagonal Tikhonov ridge for the seed projection (issue #671 multi-atom
    // conditioning). The cold multi-atom seed places near-identical coordinates
    // on every atom (the periodic seed shares the leading principal component
    // across atoms), so the joint design's per-atom column blocks are nearly
    // collinear and `X^T X` is severely ill-conditioned. A tiny mean-relative
    // ridge (the historical `mean_diag * 1e-8`) leaves the near-null directions
    // unregularized, producing decoder coefficients of order 1e5; the
    // DecoderIncoherence penalty's gradient is cubic in `B`, so those seeds blow
    // the joint solver up by ~1e15. We instead anchor the ridge to the SPECTRAL
    // scale (the maximum diagonal, an upper bound on the largest eigenvalue)
    // with a larger relative floor. This bounds the seed solution norm by
    // roughly `||X^T Z|| / ridge` while leaving well-conditioned designs
    // essentially unchanged (the ridge stays negligible against the signal
    // eigenvalues there). Conditioning the seed is correct here: the inner
    // data-fit Newton step refines `B` from a sane, bounded starting point
    // rather than a pathological one.
    let mut trace = 0.0_f64;
    let mut max_diag = 0.0_f64;
    for i in 0..m_total {
        let d = xtx[[i, i]];
        trace += d;
        if d > max_diag {
            max_diag = d;
        }
    }
    let mean_diag = (trace / m_total as f64).max(0.0);
    // Spectral-scale ridge: tie the floor to the largest diagonal so collinear
    // column blocks (small eigenvalues) are damped relative to the design's
    // dominant scale, not its average. `1e-4` is large enough to keep the seed
    // coefficient norm bounded under near-duplicate atoms yet small enough that
    // a well-conditioned design recovers essentially the unregularized LSQ fit.
    let spectral_scale = max_diag.max(mean_diag).max(1.0e-12);
    let jitter = spectral_scale * 1.0e-4;
    for i in 0..m_total {
        xtx[[i, i]] += jitter;
    }
    let xtz = fast_atb(&x, &z.to_owned());
    let factor = xtx
        .cholesky(Side::Lower)
        .map_err(|err| format!("sae_decoder_lsq_init: Cholesky failed: {err:?}"))?;
    let b_joint = factor.solve_mat(&xtz);
    if !b_joint.iter().all(|v| v.is_finite()) {
        return Err("sae_decoder_lsq_init: non-finite LSQ solution".to_string());
    }
    for atom_idx in 0..k_atoms {
        let m_k = basis_sizes[atom_idx];
        let off = offsets[atom_idx];
        for basis_col in 0..m_k {
            for out_col in 0..p_out {
                out[[atom_idx, basis_col, out_col]] = b_joint[[off + basis_col, out_col]];
            }
        }
    }
    Ok(out)
}

/// EM-style seed refinement that resolves the cold-start routing collapse of
/// the training fit (issues #629, #630) before the joint Arrow-Schur solve.
///
/// The cold residual-logit seed ([`sae_residual_seed_logits`]) is computed at
/// the cold latent coordinates (the per-atom PCA/atan2 seed). Those coordinates
/// are *shared* across atoms (the seed places the same leading component on
/// every atom), so each atom's independent LSQ fit against the full response is
/// equally mediocre on every row: the per-row residual barely separates the
/// atoms and the logit seed stays near the symmetric saddle the random jitter
/// cannot escape. The joint solver then never routes (the planted disjoint
/// atoms collapse to a near-uniform mixture, negative R²).
///
/// This is the exact dual of the frozen-decoder OOS fix (#628): there, each row
/// is placed in the correct latent basin by projecting it onto every atom's
/// *known* decoder over a dense manifold grid
/// ([`SaeManifoldTerm::seed_coords_by_decoder_projection`]). Here the decoder is
/// being *learned*, so we alternate the two cheap closed-form steps the OOS path
/// and the existing seed already provide:
///
/// 1. **Coordinate E-step** — project each row's per-atom latent onto the
///    current decoder over the manifold grid. This separates the atoms'
///    geometries: a row generated from atom `k` snaps to the coordinate where
///    atom `k` reconstructs it well, while the off-atoms snap to wherever their
///    current decoder is least wrong on that row.
/// 2. **Decoder M-step** — refit every atom's decoder by the same weighted joint
///    LSQ used for the cold init ([`sae_decoder_lsq_init`]), now at the
///    *separated* coordinates, so each atom's block specializes toward the rows
///    it actually explains.
/// 3. **Routing seed** — recompute the mean-centred residual logits
///    ([`sae_residual_seed_logits`]) at the refined geometry. With the atoms now
///    geometrically distinct, the per-row residual is decisive and the seed is
///    one-hot for the planted disjoint oracle.
///
/// A handful of rounds converges this alternation for separable atoms while
/// leaving an already-routed warm fit at its fixed point (the projection finds
/// the same basin, the LSQ recovers the same decoder). Atoms whose evaluator has
/// no projection grid (Duchon / Euclidean patch) are left untouched by step 1,
/// so the refinement degrades gracefully to the existing cold seed for them.
///
/// Only invoked for cold-start multi-atom softmax / IBP-MAP fits; JumpReLU keeps
/// its margin-above-threshold gate seed and warm starts are respected verbatim.
fn sae_em_refine_routing_seed(
    term: &mut gam::terms::sae::manifold::SaeManifoldTerm,
    z: ArrayView2<'_, f64>,
    basis_sizes: &[usize],
    assignment_kind: &str,
    alpha: f64,
    tau: f64,
    jumprelu_threshold: f64,
    random_state: u64,
) -> Result<(), String> {
    const SAE_SEED_REFINE_ROUNDS: usize = 4;
    const SAE_RESIDUAL_SEED_GAIN: f64 = 4.0;
    // Same tiny seed-keyed logit jitter the cold-start path applies (issue
    // #178): the refined residual logits are decisive (O(gain)), so this 1e-3
    // perturbation does not change which atom wins, but it keeps distinct
    // `random_state` values on distinct inner Newton trajectories and fixed
    // seeds bit-identical. Without it, the deterministic EM seed would erase
    // the seed-dependence the cold-start jitter installed upstream.
    const SAE_RANDOM_STATE_LOGIT_JITTER: f64 = 1.0e-3;
    let k_atoms = basis_sizes.len();
    let n_obs = z.nrows();
    if k_atoms <= 1 || n_obs == 0 {
        return Ok(());
    }
    let m_max = basis_sizes.iter().copied().max().unwrap_or(0);
    if m_max == 0 {
        return Ok(());
    }
    for _ in 0..SAE_SEED_REFINE_ROUNDS {
        // 1. Coordinate E-step: project each row onto the current decoder.
        term.seed_coords_by_decoder_projection(z, SAE_OOS_PROJECTION_GRID_RESOLUTION)?;
        // Snapshot the refreshed per-atom basis `Φ_k(t_k)` as a padded
        // (K, N, m_max) stack for the closed-form seed helpers.
        let mut basis3 = Array3::<f64>::zeros((k_atoms, n_obs, m_max));
        for atom_idx in 0..k_atoms {
            let phi = &term.atoms[atom_idx].basis_values;
            let m_k = basis_sizes[atom_idx];
            if phi.dim() != (n_obs, m_k) {
                return Err(format!(
                    "sae_em_refine_routing_seed: atom {atom_idx} basis is {:?}, expected ({n_obs}, {m_k})",
                    phi.dim()
                ));
            }
            for row in 0..n_obs {
                for c in 0..m_k {
                    basis3[[atom_idx, row, c]] = phi[[row, c]];
                }
            }
        }
        // 2. Decoder M-step: weighted joint LSQ at the refined coordinates,
        //    using the current routing as the responsibility weights.
        let decoder = sae_decoder_lsq_init(
            basis3.view(),
            basis_sizes,
            z,
            term.assignment.logits.view(),
            assignment_kind,
            alpha,
            tau,
            jumprelu_threshold,
        )?;
        for atom_idx in 0..k_atoms {
            let m_k = basis_sizes[atom_idx];
            let p_out = term.atoms[atom_idx].decoder_coefficients.ncols();
            let dst = &mut term.atoms[atom_idx].decoder_coefficients;
            for c in 0..m_k {
                for out_col in 0..p_out {
                    dst[[c, out_col]] = decoder[[atom_idx, c, out_col]];
                }
            }
        }
        // 3. Routing seed: mean-centred residual logits at the refined geometry.
        let logits =
            sae_residual_seed_logits(basis3.view(), basis_sizes, z, SAE_RESIDUAL_SEED_GAIN)?;
        term.assignment.logits.assign(&logits);
    }
    // Re-apply the seed-keyed jitter the deterministic refinement above erased,
    // so `random_state` keeps perturbing the inner Newton trajectory (#178).
    let mut state = random_state
        .wrapping_mul(6364136223846793005)
        .wrapping_add(1442695040888963407);
    for row in 0..n_obs {
        for atom_idx in 0..k_atoms {
            state = state
                .wrapping_mul(6364136223846793005)
                .wrapping_add(1442695040888963407);
            // Map top 53 bits to a double in [0, 1), then to [-1, 1).
            let u = ((state >> 11) as f64) * f64::from_bits(0x3CA0000000000000);
            let signed = 2.0 * u - 1.0;
            term.assignment.logits[[row, atom_idx]] += SAE_RANDOM_STATE_LOGIT_JITTER * signed;
        }
    }
    Ok(())
}

/// Build (phi, jet, penalty) for a periodic 1-D atom — same math as
/// `periodic_basis_with_jet`, but plain Rust so the helper can be reused by
/// [`sae_manifold_build_inputs`] without Python in the loop.
fn sae_build_periodic_atom(
    t: ArrayView1<'_, f64>,
    n_harmonics: usize,
) -> Result<(Array2<f64>, Array3<f64>, Array2<f64>), String> {
    let (phi, jet, penalty) =
        build_wrapped_periodic_harmonic_basis_with_jet(t, n_harmonics, "sae_build_periodic_atom")?;
    let expected_cols = sae_periodic_basis_size(n_harmonics)?;
    if phi.ncols() != expected_cols {
        return Err(format!(
            "sae_build_periodic_atom: basis width {} disagrees with declared width {expected_cols}",
            phi.ncols()
        ));
    }
    Ok((phi, jet, penalty))
}

/// Compute the per-axis basis size `axis_m = (m)^(1/d)` for a torus atom and
/// verify that `m = axis_m^d` with `axis_m` odd (i.e. `2H+1`). Returns
/// `axis_m`.
fn sae_torus_axis_basis_size(m: usize, d: usize) -> Result<usize, String> {
    if d == 0 {
        return Err("sae_torus_axis_basis_size: d must be >= 1".to_string());
    }
    if m == 0 {
        return Err("sae_torus_axis_basis_size: m must be >= 1".to_string());
    }
    // Integer d-th root via search; m is small (`<= 4096^d` in practice).
    let mut axis_m: usize = 1;
    loop {
        let mut prod: usize = 1;
        let mut overflow = false;
        for _ in 0..d {
            match prod.checked_mul(axis_m) {
                Some(p) => prod = p,
                None => {
                    overflow = true;
                    break;
                }
            }
        }
        if overflow || prod > m {
            return Err(format!(
                "sae_torus_axis_basis_size: m={m} is not a perfect d-th power for d={d}"
            ));
        }
        if prod == m {
            if axis_m % 2 == 0 {
                return Err(format!(
                    "sae_torus_axis_basis_size: m={m} = {axis_m}^{d} but axis size must be odd (2H+1)"
                ));
            }
            return Ok(axis_m);
        }
        axis_m += 1;
    }
}

/// Build (phi, jet, penalty) for a sphere atom via the (lat, lon) chart
/// evaluator. The penalty is identity on the six non-constant basis
/// functions (the constant column gets a 1e-8 floor so the (M,M) block stays
/// strictly positive on the constant subspace).
fn sae_build_sphere_atom(
    coords: ArrayView2<'_, f64>,
) -> Result<(Array2<f64>, Array3<f64>, Array2<f64>), String> {
    let (phi, jet) = SphereChartEvaluator.evaluate(coords)?;
    let m = phi.ncols();
    let mut penalty = Array2::<f64>::zeros((m, m));
    penalty[[0, 0]] = 1.0e-8;
    for i in 1..m {
        penalty[[i, i]] = 1.0;
    }
    Ok((phi, jet, penalty))
}

/// Build (phi, jet, penalty) for a torus atom via the tensor-product
/// periodic harmonic evaluator. Penalty diagonal encodes the squared
/// Laplace–Beltrami eigenvalue
/// `((2π)^2 · Σ_a h_a^2)^2` so the smoothness term penalises high-frequency
/// modes — same shape as the 1-D periodic harmonic penalty
/// (`(2π·h)^4` reduces to `h^4` up to a constant), generalised to T^d.
fn sae_build_torus_atom(
    coords: ArrayView2<'_, f64>,
    latent_dim: usize,
    num_harmonics: usize,
) -> Result<(Array2<f64>, Array3<f64>, Array2<f64>), String> {
    let evaluator = TorusHarmonicEvaluator::new(latent_dim, num_harmonics)?;
    let (phi, jet) = evaluator.evaluate(coords)?;
    let axis_m = evaluator.axis_basis_size();
    let m = phi.ncols();
    let mut penalty = Array2::<f64>::zeros((m, m));
    // Decode axis index `idx_axis ∈ {0..axis_m}` → harmonic number `h`:
    // 0 → 0 (constant), 1 → 1 (sin), 2 → 1 (cos), 3 → 2, 4 → 2, …
    let axis_harmonic = |idx_axis: usize| -> usize {
        if idx_axis == 0 {
            0
        } else {
            idx_axis.div_ceil(2)
        }
    };
    let mut idx = vec![0usize; latent_dim];
    for flat in 0..m {
        let mut h_sum_sq: usize = 0;
        for axis in 0..latent_dim {
            let h = axis_harmonic(idx[axis]);
            h_sum_sq += h * h;
        }
        let lambda = if h_sum_sq == 0 {
            1.0e-8
        } else {
            (h_sum_sq as f64).powi(2)
        };
        penalty[[flat, flat]] = lambda;
        for axis in (0..latent_dim).rev() {
            idx[axis] += 1;
            if idx[axis] < axis_m {
                break;
            }
            idx[axis] = 0;
        }
    }
    Ok((phi, jet, penalty))
}

/// Build (phi, jet, penalty) for a Duchon atom. Mirrors the pure-Rust path
/// inside `duchon_basis_with_jet` but accepts in-Rust types and returns
/// owned `(Array2, Array3, Array2)`.
fn sae_build_duchon_atom(
    pts: ArrayView2<'_, f64>,
    centers: ArrayView2<'_, f64>,
) -> Result<(Array2<f64>, Array3<f64>, Array2<f64>), String> {
    // The `DuchonCoordinateEvaluator` evaluates the pure polyharmonic basis
    // (`length_scale = None`, `power = 0`, scale-free) at the resolved nullspace
    // order, so the matching smoothness penalty is the native reproducing-norm
    // Gram `ω = α²·Zᵀ K_CC Z`, which the redesigned `build_duchon_basis` emits
    // as `PenaltySource::Primary` (the structural roughness energy on the same
    // `[K(·,C)·Z | P]` columns the evaluator produces). The redesign no longer
    // ships the old mass/tension/stiffness operator triplet for the Euclidean
    // path, so `operator_penalties` is fully disabled here — the native penalty
    // path ignores it, and leaving a stale `Active` stiffness would only mislead.
    //
    // The nullspace order and power MUST match the evaluator (`power = 0`,
    // order = `duchon_nullspace_from_m(m)`); with `m = sae_duchon_atom_m(d)` the
    // polynomial nullspace is large enough that the native Gram is well-posed.
    let dim = centers.ncols();
    let m: usize = sae_duchon_atom_m(dim);
    let spec = DuchonBasisSpec {
        center_strategy: CenterStrategy::UserProvided(centers.to_owned()),
        length_scale: None,
        power: 0.0,
        nullspace_order: duchon_nullspace_from_m(m),
        identifiability: SpatialIdentifiability::None,
        aniso_log_scales: None,
        operator_penalties: DuchonOperatorPenaltySpec {
            mass: OperatorPenaltySpec::Disabled,
            tension: OperatorPenaltySpec::Disabled,
            stiffness: OperatorPenaltySpec::Disabled,
        },
        periodic: None,
        boundary: OneDimensionalBoundary::Open,
    };
    // The penalty matrix is the only piece sourced from the full builder; the
    // design `Phi` and its input-location jet come from the single
    // amplification-consistent core entry point so the seed atom matches the
    // `DuchonCoordinateEvaluator` refresh bit-for-bit (issue #247).
    let built = build_duchon_basis(centers, &spec).map_err(|err| err.to_string())?;
    // The structural roughness penalty is the native reproducing-norm Gram,
    // emitted as the `PenaltySource::Primary` block. The redesigned Euclidean
    // Duchon basis always emits exactly one `Primary` candidate, so assert it as
    // a structural invariant rather than silently returning `None`.
    let primary_idx = built
        .penaltyinfo
        .iter()
        .position(|info| matches!(info.source, gam::terms::basis::PenaltySource::Primary))
        .ok_or_else(|| {
            format!(
                "sae_build_duchon_atom: native (Primary) smoothness penalty was not built for \
                 dim={dim}, m={m}, nullspace_order={:?}; the Euclidean Duchon basis must always \
                 emit a Primary native-norm Gram",
                duchon_nullspace_from_m(m),
            )
        })?;
    let penalty = built
        .penalties
        .get(primary_idx)
        .ok_or_else(|| {
            format!(
                "sae_build_duchon_atom: native penalty index {primary_idx} exceeds {} penalty matrices",
                built.penalties.len()
            )
        })?
        .clone();
    let evaluator = DuchonCoordinateEvaluator::new(centers.to_owned(), m)?;
    let (phi, jet) = if pts.nrows() == 0 {
        let probe = Array2::<f64>::zeros((1, pts.ncols()));
        let (probe_phi, _probe_jet) = evaluator.evaluate(probe.view())?;
        let cols = probe_phi.ncols();
        (
            Array2::<f64>::zeros((0, cols)),
            Array3::<f64>::zeros((0, cols, dim)),
        )
    } else {
        evaluator.evaluate(pts)?
    };
    if phi.ncols() != jet.shape()[1] {
        return Err(format!(
            "sae_build_duchon_atom: phi/jet column mismatch {} vs {}",
            phi.ncols(),
            jet.shape()[1]
        ));
    }
    Ok((phi, jet, penalty))
}

/// Build (phi, jet, penalty) for a Euclidean tangent-patch atom.
///
/// A Euclidean atom is a *flat* (zero-curvature) polynomial expansion in the
/// atom's latent coordinates — distinct from the thin-plate Duchon kernel.
/// The basis is the set of monomials of total degree ≤ `EUCLIDEAN_PATCH_MAX_DEGREE`,
/// the jet is the first derivative of those monomials, and the penalty is an
/// identity ridge over the non-constant monomials (the constant term is left
/// unpenalized to preserve the affine-equivariance of the patch).
///
/// `centers` is accepted for API symmetry with the Duchon path; its row count
/// determines the random-state matching seam (issue #246), but it is not
/// otherwise used: a polynomial atom has no center-based locality.
fn sae_euclidean_degree_for_basis_size(dim: usize, basis_size: usize) -> Result<usize, String> {
    for degree in 0..=SAE_EUCLIDEAN_PATCH_MAX_DEGREE {
        if monomial_exponents(dim, degree).len() == basis_size {
            return Ok(degree);
        }
    }
    Err(format!(
        "euclidean patch basis size {basis_size} is not a valid monomial width for latent_dim={dim} with max_degree<={SAE_EUCLIDEAN_PATCH_MAX_DEGREE}"
    ))
}

fn sae_build_euclidean_atom_with_degree(
    pts: ArrayView2<'_, f64>,
    centers: ArrayView2<'_, f64>,
    max_degree: usize,
) -> Result<(Array2<f64>, Array3<f64>, Array2<f64>), String> {
    let dim = centers.ncols();
    let exponents = monomial_exponents(dim, max_degree);
    let n_basis = exponents.len();
    // The design `Phi` and its jet come from the same evaluator the inner
    // Newton loop refreshes against, so the seed atom and every refresh share
    // one monomial layout.
    let evaluator = EuclideanPatchEvaluator::new(dim, max_degree)?;
    let (phi, jet) = if pts.nrows() == 0 {
        (
            Array2::<f64>::zeros((0, n_basis)),
            Array3::<f64>::zeros((0, n_basis, dim)),
        )
    } else {
        evaluator.evaluate(pts)?
    };
    if jet.shape()[1] != n_basis {
        return Err(format!(
            "sae_build_euclidean_atom: monomial/jet column mismatch {} vs {}",
            n_basis,
            jet.shape()[1]
        ));
    }
    // Identity ridge with the constant term (alpha == zeros) unpenalized so the
    // patch can absorb a global offset without paying a penalty.
    let mut penalty = Array2::<f64>::zeros((n_basis, n_basis));
    for (col, alpha) in exponents.iter().enumerate() {
        let is_constant = alpha.iter().all(|&e| e == 0);
        if !is_constant {
            penalty[[col, col]] = 1.0;
        }
    }
    Ok((phi, jet, penalty))
}

fn sae_build_euclidean_atom(
    pts: ArrayView2<'_, f64>,
    centers: ArrayView2<'_, f64>,
) -> Result<(Array2<f64>, Array3<f64>, Array2<f64>), String> {
    sae_build_euclidean_atom_with_degree(pts, centers, SAE_EUCLIDEAN_PATCH_MAX_DEGREE)
}

/// Deterministically pick Duchon centers from the PCA-seeded coordinates.
/// Uses a Lehmer (LCG) walk over `0..n_obs` keyed by `random_state` so the
/// result is reproducible without a heavy RNG dependency.
fn sae_pick_duchon_center_indices(n_obs: usize, n_centers: usize, random_state: u64) -> Vec<usize> {
    let want = n_centers.min(n_obs);
    if want == 0 || n_obs == 0 {
        return Vec::new();
    }
    if want >= n_obs {
        return (0..n_obs).collect();
    }
    let mut chosen: Vec<usize> = (0..n_obs).collect();
    let mut state = random_state
        .wrapping_mul(6364136223846793005)
        .wrapping_add(1442695040888963407);
    for i in (1..n_obs).rev() {
        state = state
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        let j = (state >> 33) as usize % (i + 1);
        chosen.swap(i, j);
    }
    chosen.truncate(want);
    chosen.sort_unstable();
    chosen
}

/// Build the padded `(phi_stack, jet_stack, penalty_stack)` arrays plus
/// per-atom `basis_sizes` for the given atom plans and seed coords. Returns
/// `(basis_values, basis_jacobian, smooth_penalties, basis_sizes, coord_blocks)`.
fn sae_build_padded_basis_stacks(
    plans: &[SaeAtomBuildPlan],
    seed_coords: ArrayView3<'_, f64>,
    n_obs: usize,
) -> Result<
    (
        Array3<f64>,
        Array4<f64>,
        Array3<f64>,
        Vec<usize>,
        Vec<Array2<f64>>,
    ),
    String,
> {
    let k_atoms = plans.len();
    let seed_shape = seed_coords.shape();
    if seed_shape[0] != k_atoms || seed_shape[1] < n_obs {
        return Err(format!(
            "sae_build_padded_basis_stacks: seed_coords must have shape (K, N_seed, D_max) with K={k_atoms} and N_seed >= {n_obs}; got {:?}",
            seed_shape
        ));
    }
    for (atom_idx, plan) in plans.iter().enumerate() {
        if plan.latent_dim == 0 {
            return Err(format!(
                "sae_build_padded_basis_stacks: atom {atom_idx} latent_dim must be positive"
            ));
        }
        if plan.latent_dim > seed_shape[2] {
            return Err(format!(
                "sae_build_padded_basis_stacks: atom {atom_idx} latent_dim {} exceeds seed_coords D_max={}",
                plan.latent_dim, seed_shape[2]
            ));
        }
    }
    let basis_sizes: Vec<usize> = plans.iter().map(sae_atom_basis_size).collect();
    let m_max = basis_sizes.iter().copied().max().unwrap_or(1).max(1);
    let d_max = plans.iter().map(|p| p.latent_dim).max().unwrap_or(1).max(1);
    let mut phi_stack = Array3::<f64>::zeros((k_atoms, n_obs, m_max));
    let mut jet_stack = Array4::<f64>::zeros((k_atoms, n_obs, m_max, d_max));
    let mut penalty_stack = Array3::<f64>::zeros((k_atoms, m_max, m_max));
    let mut coord_blocks: Vec<Array2<f64>> = Vec::with_capacity(k_atoms);
    for (atom_idx, plan) in plans.iter().enumerate() {
        let d = plan.latent_dim;
        let coords = seed_coords.slice(s![atom_idx, 0..n_obs, 0..d]).to_owned();
        match &plan.kind {
            SaeAtomBasisKind::Periodic => {
                let t = if d >= 1 {
                    coords.column(0).to_owned()
                } else {
                    Array1::<f64>::zeros(n_obs)
                };
                let (phi, jet, penalty) = sae_build_periodic_atom(t.view(), plan.n_harmonics)?;
                let m = phi.ncols();
                if phi.nrows() != n_obs || m != basis_sizes[atom_idx] {
                    return Err(format!(
                        "sae_build_padded_basis_stacks: atom {atom_idx} periodic basis shape {:?} disagrees with N={n_obs}, declared M={}",
                        phi.dim(),
                        basis_sizes[atom_idx]
                    ));
                }
                if jet.shape() != &[n_obs, m, 1] {
                    return Err(format!(
                        "sae_build_padded_basis_stacks: atom {atom_idx} periodic jet shape {:?} disagrees with expected ({n_obs}, {m}, 1)",
                        jet.shape()
                    ));
                }
                if penalty.dim() != (m, m) {
                    return Err(format!(
                        "sae_build_padded_basis_stacks: atom {atom_idx} periodic penalty shape {:?} disagrees with M={m}",
                        penalty.dim()
                    ));
                }
                phi_stack
                    .slice_mut(s![atom_idx, 0..n_obs, 0..m])
                    .assign(&phi);
                let jet_d = jet.shape()[2].min(d_max);
                jet_stack
                    .slice_mut(s![atom_idx, 0..n_obs, 0..m, 0..jet_d])
                    .assign(&jet.slice(s![.., .., 0..jet_d]));
                penalty_stack
                    .slice_mut(s![atom_idx, 0..m, 0..m])
                    .assign(&penalty);
            }
            SaeAtomBasisKind::Sphere => {
                if d != 2 {
                    return Err(format!(
                        "sae_build_padded_basis_stacks: atom {atom_idx} Sphere requires latent_dim == 2, got {d}"
                    ));
                }
                let (phi, jet, penalty) = sae_build_sphere_atom(coords.view())?;
                let m = phi.ncols();
                if m != basis_sizes[atom_idx] {
                    return Err(format!(
                        "sae_build_padded_basis_stacks: atom {atom_idx} Sphere basis size {m} disagrees with declared M={}",
                        basis_sizes[atom_idx]
                    ));
                }
                phi_stack
                    .slice_mut(s![atom_idx, 0..n_obs, 0..m])
                    .assign(&phi);
                let jet_d = jet.shape()[2].min(d_max);
                jet_stack
                    .slice_mut(s![atom_idx, 0..n_obs, 0..m, 0..jet_d])
                    .assign(&jet.slice(s![.., .., 0..jet_d]));
                penalty_stack
                    .slice_mut(s![atom_idx, 0..m, 0..m])
                    .assign(&penalty);
            }
            SaeAtomBasisKind::Torus => {
                let h = plan.n_harmonics.max(1);
                let (phi, jet, penalty) = sae_build_torus_atom(coords.view(), d, h)?;
                let m = phi.ncols();
                if m != basis_sizes[atom_idx] {
                    return Err(format!(
                        "sae_build_padded_basis_stacks: atom {atom_idx} Torus basis size {m} disagrees with declared M={}",
                        basis_sizes[atom_idx]
                    ));
                }
                phi_stack
                    .slice_mut(s![atom_idx, 0..n_obs, 0..m])
                    .assign(&phi);
                let jet_d = jet.shape()[2].min(d_max);
                jet_stack
                    .slice_mut(s![atom_idx, 0..n_obs, 0..m, 0..jet_d])
                    .assign(&jet.slice(s![.., .., 0..jet_d]));
                penalty_stack
                    .slice_mut(s![atom_idx, 0..m, 0..m])
                    .assign(&penalty);
            }
            SaeAtomBasisKind::Duchon
            | SaeAtomBasisKind::EuclideanPatch
            | SaeAtomBasisKind::Poincare => {
                let centers = plan
                    .duchon_centers
                    .as_ref()
                    .ok_or_else(|| {
                        format!(
                            "sae_build_padded_basis_stacks: atom {atom_idx} non-periodic atom requires centers"
                        )
                    })?;
                if centers.ncols() != d {
                    return Err(format!(
                        "sae_build_padded_basis_stacks: atom {atom_idx} centers have dim {} but plan latent_dim is {d}",
                        centers.ncols()
                    ));
                }
                let (phi, jet, penalty) = match plan.kind {
                    SaeAtomBasisKind::EuclideanPatch | SaeAtomBasisKind::Poincare => {
                        let degree = sae_euclidean_degree_for_basis_size(d, basis_sizes[atom_idx])?;
                        sae_build_euclidean_atom_with_degree(coords.view(), centers.view(), degree)?
                    }
                    _ => sae_build_duchon_atom(coords.view(), centers.view())?,
                };
                let m = phi.ncols();
                if phi.nrows() != n_obs || m != basis_sizes[atom_idx] {
                    return Err(format!(
                        "sae_build_padded_basis_stacks: atom {atom_idx} Duchon basis shape {:?} disagrees with N={n_obs}, declared M={}",
                        phi.dim(),
                        basis_sizes[atom_idx]
                    ));
                }
                if jet.shape() != &[n_obs, m, d] {
                    return Err(format!(
                        "sae_build_padded_basis_stacks: atom {atom_idx} Duchon jet shape {:?} disagrees with expected ({n_obs}, {m}, {d})",
                        jet.shape()
                    ));
                }
                if penalty.dim() != (m, m) {
                    return Err(format!(
                        "sae_build_padded_basis_stacks: atom {atom_idx} Duchon penalty shape {:?} disagrees with M={m}",
                        penalty.dim()
                    ));
                }
                phi_stack
                    .slice_mut(s![atom_idx, 0..n_obs, 0..m])
                    .assign(&phi);
                let jet_d = jet.shape()[2].min(d_max);
                jet_stack
                    .slice_mut(s![atom_idx, 0..n_obs, 0..m, 0..jet_d])
                    .assign(&jet.slice(s![.., .., 0..jet_d]));
                penalty_stack
                    .slice_mut(s![atom_idx, 0..m, 0..m])
                    .assign(&penalty);
            }
            SaeAtomBasisKind::Cylinder => {
                // Cylinder is a birth-discovered topology, never built through the
                // seed-plan stack path (`sae_build_atom_plans` rejects it above), so
                // it cannot reach here from a `SaeAtomBuildPlan`. Surfaced loudly if
                // it ever does, rather than mis-built.
                return Err(format!(
                    "sae_build_padded_basis_stacks: atom {atom_idx} 'cylinder' is a birth-discovered \
                     topology, not a seed-plan stack kind; it has no padded seed basis to build"
                ));
            }
            SaeAtomBasisKind::Precomputed(name) => {
                return Err(format!(
                    "sae_build_padded_basis_stacks: unsupported atom {atom_idx} basis {:?}; precomputed atoms require caller-supplied padded basis arrays",
                    name
                ));
            }
        }
        coord_blocks.push(coords);
    }
    Ok((
        phi_stack,
        jet_stack,
        penalty_stack,
        basis_sizes,
        coord_blocks,
    ))
}

/// Build [`SaeAtomBuildPlan`]s from `(z, atom_basis, atom_dim)` + per-atom
/// PCA seed. Periodic atoms get `n_harmonics = max(1, d_atom)`; Duchon atoms
/// get deterministic center indices from the PCA seed.
fn sae_build_atom_plans(
    z: ArrayView2<'_, f64>,
    atom_basis: &[String],
    atom_dim: &[usize],
    seed_coords: ArrayView3<'_, f64>,
    random_state: u64,
) -> Result<Vec<SaeAtomBuildPlan>, String> {
    let k_atoms = atom_basis.len();
    let n_obs = z.nrows();
    let seed_shape = seed_coords.shape();
    if atom_dim.len() != k_atoms {
        return Err(format!(
            "sae_build_atom_plans: atom_dim length {} must equal atom_basis length {k_atoms}",
            atom_dim.len()
        ));
    }
    if seed_shape[0] != k_atoms || seed_shape[1] != n_obs {
        return Err(format!(
            "sae_build_atom_plans: seed_coords must start with (K, N)=({k_atoms}, {n_obs}); got {:?}",
            seed_shape
        ));
    }
    let mut plans: Vec<SaeAtomBuildPlan> = Vec::with_capacity(k_atoms);
    for atom_idx in 0..k_atoms {
        let kind = sae_atom_basis_kind_from_str(&atom_basis[atom_idx]);
        let d = atom_dim[atom_idx];
        if d == 0 {
            return Err(format!(
                "sae_build_atom_plans: atom_dim[{atom_idx}] must be positive"
            ));
        }
        if d > seed_shape[2] {
            return Err(format!(
                "sae_build_atom_plans: atom_dim[{atom_idx}]={d} exceeds seed_coords D_max={}",
                seed_shape[2]
            ));
        }
        match &kind {
            SaeAtomBasisKind::Periodic => {
                // A periodic atom parameterises a circle, which is intrinsically
                // 1-dimensional: the latent coordinate is a single phase angle
                // `t ∈ [0, 1)`. The user-facing `atom_dim` (a.k.a. `d_atom`) for
                // a periodic basis selects the *number of harmonics* in the
                // truncated Fourier expansion (basis size `2·n_harmonics + 1`),
                // not a latent-space dimensionality. Setting
                // `latent_dim = atom_dim` would make
                // `build_sae_basis_evaluators` reject the atom (the analytic
                // `PeriodicHarmonicEvaluator` requires `latent_dim == 1`),
                // since there is no longer a frozen-snapshot fallback. Bind the
                // optimizer-visible latent dimension to 1 and route the user's
                // `d_atom` into the harmonic count.
                let n_harmonics = d.max(1);
                let basis_size = sae_periodic_basis_size(n_harmonics)?;
                plans.push(SaeAtomBuildPlan {
                    kind: SaeAtomBasisKind::Periodic,
                    latent_dim: 1,
                    n_harmonics,
                    duchon_centers: None,
                    basis_size,
                });
            }
            SaeAtomBasisKind::Sphere => {
                // The (lat, lon) chart fixes latent_dim = 2 and basis_size = 7
                // regardless of the user-supplied `atom_dim` — the chart
                // already captures the embedded S² geometry. Reject any
                // d_atom other than 2 to keep the contract honest.
                if d != 2 {
                    return Err(format!(
                        "sae_build_atom_plans: atom {atom_idx} basis 'sphere' requires atom_dim == 2, got {d}"
                    ));
                }
                plans.push(SaeAtomBuildPlan {
                    kind: SaeAtomBasisKind::Sphere,
                    latent_dim: 2,
                    n_harmonics: 0,
                    duchon_centers: None,
                    basis_size: SAE_SPHERE_BASIS_SIZE,
                });
            }
            SaeAtomBasisKind::Torus => {
                // Torus of dim `d` uses a tensor-product periodic harmonic
                // basis of size `(2H+1)^d`. The user's `atom_dim` selects
                // the latent dimension; `n_harmonics` defaults to
                // `SAE_DEFAULT_TORUS_HARMONICS`. The design grows
                // exponentially in `d`, so reject runaway combinations.
                let h = SAE_DEFAULT_TORUS_HARMONICS;
                let evaluator = TorusHarmonicEvaluator::new(d, h)?;
                let basis_size = evaluator.basis_size();
                if basis_size > SAE_MAX_PERIODIC_HARMONICS * 4 {
                    return Err(format!(
                        "sae_build_atom_plans: atom {atom_idx} torus basis size {basis_size} = (2*{h}+1)^{d} exceeds the dense limit; reduce atom_dim or harmonics"
                    ));
                }
                plans.push(SaeAtomBuildPlan {
                    kind: SaeAtomBasisKind::Torus,
                    latent_dim: d,
                    n_harmonics: h,
                    duchon_centers: None,
                    basis_size,
                });
            }
            SaeAtomBasisKind::Duchon
            | SaeAtomBasisKind::EuclideanPatch
            | SaeAtomBasisKind::Poincare => {
                // A Duchon atom's curvature penalty degrades (and ultimately
                // fails its D2 collocation) when the center count does not
                // exceed the polynomial nullspace dimension of its resolved
                // order. Pick enough centers to clear that dimension with a
                // margin (so a positive-rank kernel block survives), bounded
                // above by `n_obs` and the dense cap. The Euclidean patch
                // ignores centers, so this lower bound is harmless there.
                let duchon_m = sae_duchon_atom_m(d);
                let poly_nullspace_dim = duchon_nullspace_dimension(d, duchon_m.saturating_sub(1));
                let center_floor = (poly_nullspace_dim + d + 1).max(8);
                let center_ceiling = center_floor.max(32);
                let lo = center_floor.min(n_obs);
                let hi = center_ceiling.min(n_obs);
                let n_centers = n_obs.min(hi).max(lo);
                let idx = sae_pick_duchon_center_indices(
                    n_obs,
                    n_centers,
                    random_state.wrapping_add(atom_idx as u64),
                );
                let mut centers = Array2::<f64>::zeros((idx.len(), d));
                for (out_row, src_row) in idx.iter().copied().enumerate() {
                    for col in 0..d {
                        centers[[out_row, col]] = seed_coords[[atom_idx, src_row, col]];
                    }
                }
                // Probe one build to learn the final basis size. Euclidean
                // patches use the polynomial atom; everything else (Duchon)
                // uses the thin-plate kernel.
                let probe_pts = Array2::<f64>::zeros((1, d));
                let (phi, _jet, _penalty) = match kind {
                    SaeAtomBasisKind::EuclideanPatch | SaeAtomBasisKind::Poincare => {
                        sae_build_euclidean_atom(probe_pts.view(), centers.view())?
                    }
                    _ => sae_build_duchon_atom(probe_pts.view(), centers.view())?,
                };
                let basis_size = phi.ncols();
                plans.push(SaeAtomBuildPlan {
                    kind,
                    latent_dim: d,
                    n_harmonics: 0,
                    duchon_centers: Some(centers),
                    basis_size,
                });
            }
            SaeAtomBasisKind::Cylinder => {
                // A cylinder atom is not SEEDED through `sae_manifold_fit_minimal`:
                // it arises only by EVIDENCE, when the #977 birth topology race
                // selects `S¹ × ℝ` for a residual factor (the born atom's evaluator
                // is built directly by `race_birth_topology`, and OOS refresh reads
                // it back through `build_sae_basis_evaluators`). There is no
                // user-facing cylinder seed geometry to derive a plan from here, so
                // a cylinder in the seed dictionary is a caller error, surfaced
                // loudly rather than mis-built as a torus / patch.
                return Err(
                    "sae_build_atom_plans: 'cylinder' is a birth-discovered topology, not a seed \
                     dictionary kind; seed with periodic, duchon, sphere, torus, or \
                     euclidean_patch and let the structure search grow a cylinder by evidence"
                        .to_string(),
                );
            }
            SaeAtomBasisKind::Precomputed(name) => {
                return Err(format!(
                    "sae_build_atom_plans: unsupported atom_basis {:?}; sae_manifold_fit_minimal can build only periodic, duchon, sphere, torus, or euclidean_patch atoms",
                    name
                ));
            }
        }
    }
    Ok(plans)
}

/// One-shot SAE-manifold fit driver: takes only `(z, atom_basis, atom_dim,
/// ...scalar hyperparams)` and assembles the full basis + jacobian + penalty
/// stack + PCA seed coords + zero-init decoder + zero-init logits internally
/// before delegating to the same end-to-end Rust Newton loop as
/// [`sae_manifold_fit`]. Returns the same payload dict with one extra key,
/// `"atom_plans"`, holding the per-atom basis spec so OOS prediction can
/// rebuild the design without going through Python.
///
/// Warm starts (issue #357): `initial_logits` (N, K) and `initial_coords`
/// (K, N, D_max) are optional caller-supplied seeds for the assignment logits
/// and the per-atom on-manifold coordinates. When supplied they replace the
/// internal PCA seed coords / zero-jitter logit init, so an amortized encoder
/// can predict `(a_init, t_init)` and have the joint solver refine them for a
/// bounded `max_iter` steps. The basis *design* (Duchon centers, harmonic
/// counts) is still derived from the PCA seed so the warm coordinates are
/// evaluated against the same atom geometry the unconstrained fit would build.
#[pyfunction(signature = (
    z,
    atom_basis,
    atom_dim,
    alpha,
    tau,
    learnable_alpha,
    assignment_kind,
    sparsity_strength = 1.0,
    smoothness = 1.0,
    max_iter = 50,
    learning_rate = 0.05,
    ridge_ext_coord = 1.0e-6,
    ridge_beta = 1.0e-6,
    gumbel_schedule = None,
    analytic_penalties = None,
    random_state = 0,
    top_k = None,
    initial_logits = None,
    initial_coords = None,
    jumprelu_threshold = 0.0,
    native_ard_enabled = true,
    fisher_factors = None,
    fisher_mass_residual = None,
    fisher_provenance = None,
    row_loss_weights = None,
))]
#[allow(clippy::too_many_arguments)]
fn sae_manifold_fit_minimal<'py>(
    py: Python<'py>,
    z: PyReadonlyArray2<'py, f64>,
    atom_basis: Vec<String>,
    atom_dim: Vec<usize>,
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
    random_state: u64,
    top_k: Option<usize>,
    initial_logits: Option<PyReadonlyArray2<'py, f64>>,
    initial_coords: Option<PyReadonlyArray3<'py, f64>>,
    jumprelu_threshold: f64,
    native_ard_enabled: bool,
    // WP-D output-Fisher shard (#980). `(n, p, r)` f64 factors; presence activates
    // `RowMetric::OutputFisher`. This is the entry point the high-level Python
    // `sae_manifold_fit` facade routes through, so it carries the shard the same
    // magic-by-default way as the precomputed-basis `sae_manifold_fit`.
    fisher_factors: Option<PyReadonlyArray3<'py, f64>>,
    fisher_mass_residual: Option<PyReadonlyArray1<'py, f64>>,
    // Harvest provenance tag (#980): same-position `"output_fisher"` (default) or
    // forward-looking `"output_fisher_downstream"`. Routed to the matching
    // `RowMetric` constructor; gauge/lens/dose consume either unchanged.
    fisher_provenance: Option<String>,
    // Per-row design-honesty reconstruction weights (#977); `(n,)` √w. Absent ⇒
    // unweighted path. Installed on the term before the joint fit / ρ selection.
    row_loss_weights: Option<PyReadonlyArray1<'py, f64>>,
) -> PyResult<Py<PyDict>> {
    let z_view = z.as_array();
    let (n_obs, _p_out) = z_view.dim();
    let k_atoms = atom_basis.len();
    if n_obs == 0 || z_view.ncols() == 0 {
        return Err(py_value_error(format!(
            "sae_manifold_fit_minimal: z must be non-empty; got shape ({}, {})",
            n_obs,
            z_view.ncols()
        )));
    }
    if k_atoms == 0 {
        return Err(py_value_error(
            "sae_manifold_fit_minimal: atom_basis must be non-empty".into(),
        ));
    }
    if !z_view.iter().all(|v| v.is_finite()) {
        return Err(py_value_error(
            "sae_manifold_fit_minimal: z contains non-finite values".into(),
        ));
    }
    if atom_dim.len() != k_atoms {
        return Err(py_value_error(format!(
            "sae_manifold_fit_minimal: atom_dim length {} must equal atom_basis length {k_atoms}",
            atom_dim.len()
        )));
    }
    let basis_kinds: Vec<SaeAtomBasisKind> = atom_basis
        .iter()
        .map(|kind| sae_atom_basis_kind_from_str(kind))
        .collect();
    let seed_coords =
        gam::terms::sae::manifold::sae_pca_seed_initial_coords(z_view, &basis_kinds, &atom_dim)
            .map_err(py_value_error)?;
    let plans = sae_build_atom_plans(
        z_view,
        &atom_basis,
        &atom_dim,
        seed_coords.view(),
        random_state,
    )
    .map_err(py_value_error)?;
    // The optimizer's per-atom latent dimension is `plan.latent_dim`, not the
    // user-supplied `atom_dim` (periodic atoms carry a harmonic count there).
    let plan_latent_dim: Vec<usize> = plans.iter().map(|plan| plan.latent_dim).collect();
    // Warm-start coordinates (issue #357). When the caller supplies
    // `initial_coords` (an amortized encoder's predicted on-manifold `t`), use
    // them as the Newton start and as the point at which the basis stacks are
    // first evaluated; otherwise fall back to the PCA seed. The basis *design*
    // (`plans`) is always built from the PCA seed so the atom geometry matches
    // the cold-start fit. Shape must be `(K, N, D_max)` with `D_max` covering
    // every atom's `plan.latent_dim`.
    let mut start_coords: Array3<f64> = match &initial_coords {
        Some(arr) => {
            let view = arr.as_array();
            let shape = view.shape();
            if shape.len() != 3 {
                return Err(py_value_error(
                    "sae_manifold_fit_minimal: initial_coords must be a rank-3 (K, N, D_max) array"
                        .to_string(),
                ));
            }
            if shape[0] != k_atoms || shape[1] != n_obs {
                return Err(py_value_error(format!(
                    "sae_manifold_fit_minimal: initial_coords must start with (K, N)=({k_atoms}, {n_obs}); got {shape:?}"
                )));
            }
            let max_dim = *shape.get(2).unwrap_or(&0);
            for (atom_idx, &d) in plan_latent_dim.iter().enumerate() {
                if d > max_dim {
                    return Err(py_value_error(format!(
                        "sae_manifold_fit_minimal: initial_coords D_max={max_dim} is too small for atom {atom_idx} latent_dim={d}"
                    )));
                }
            }
            if !view.iter().all(|v| v.is_finite()) {
                return Err(py_value_error(
                    "sae_manifold_fit_minimal: initial_coords contains non-finite values".into(),
                ));
            }
            view.to_owned()
        }
        None => seed_coords.clone(),
    };
    if initial_coords.is_none()
        && k_atoms > 1
        && matches!(assignment_kind.as_str(), "softmax" | "ibp_map")
    {
        let labels = sae_output_energy_cluster_labels(z_view, k_atoms);
        sae_refine_periodic_seed_coords_by_cluster(z_view, &plans, &labels, &mut start_coords)
            .map_err(py_value_error)?;
    }
    let (basis_values, basis_jacobian, smooth_penalties, basis_sizes, _coord_blocks) =
        sae_build_padded_basis_stacks(&plans, start_coords.view(), n_obs)
            .map_err(py_value_error)?;
    // The JumpReLU gate activates only when a logit clears
    // `jumprelu_threshold` (caller-configurable, default 0.0). Seeding the
    // logits at or below the threshold would leave every gate closed at step
    // 0, making the data-fit Jacobian, the sparsity prior gradient, and the
    // assignment-weighted decoder gradient all zero simultaneously — the fit
    // cannot escape that fixed point. Seed JumpReLU runs a fixed margin
    // ABOVE the configured threshold so every atom starts active relative to
    // its cut and the fit can learn which atoms to prune. Softmax
    // (translation-invariant) remains neutral at zero, while IBP-MAP uses the
    // zero seed except for its degenerate K=1 gate handled below.
    // Warm-start logits (issue #357): a caller-supplied `(N, K)` assignment
    // logit seed (from an amortized encoder) replaces the cold-start init.
    // When absent we fall back to the documented zero / JumpReLU-positive init
    // plus the seed-keyed jitter below.
    let warm_logits: Option<Array2<f64>> = match &initial_logits {
        Some(arr) => {
            let view = arr.as_array();
            if view.dim() != (n_obs, k_atoms) {
                return Err(py_value_error(format!(
                    "sae_manifold_fit_minimal: initial_logits must be ({n_obs}, {k_atoms}); got {:?}",
                    view.dim()
                )));
            }
            if !view.iter().all(|v| v.is_finite()) {
                return Err(py_value_error(
                    "sae_manifold_fit_minimal: initial_logits contains non-finite values".into(),
                ));
            }
            Some(view.to_owned())
        }
        None => None,
    };
    let logits_are_cold = warm_logits.is_none();
    let mut initial_logits = match warm_logits {
        Some(logits) => logits,
        None if assignment_kind == "jumprelu" => {
            // Start every atom one full margin above its activation threshold.
            const SAE_JUMPRELU_SEED_MARGIN: f64 = 1.0;
            Array2::<f64>::from_elem(
                (n_obs, k_atoms),
                jumprelu_threshold + SAE_JUMPRELU_SEED_MARGIN,
            )
        }
        None if k_atoms == 1 && assignment_kind == "ibp_map" => {
            // At K=1 the IBP stick-breaking prior is degenerate (pi_0 == 1), so the
            // gate zeta = sigma(logit/tau) is a free multiplicative scalar on the
            // reconstruction with no competing atom and no sparsity pressure. A zero
            // seed starts it at sigma(0)=0.5 -- a 50% radial seed contraction the
            // joint fit must climb back from against a vanishing sigmoid gradient,
            // landing the ring inside the data (#1023). Seed the single atom
            // "present" so zeta starts ~1; the gate stays free to fall if the atom is
            // genuinely vacuous (the post-fit EV collapse guard, not zeta->0, flags
            // that, so part-3 collapse detection is unaffected). Temperature-robust:
            // seed logit = c*tau so zeta = sigma(c) is independent of tau.
            const SAE_IBP_K1_PRESENT_GATE_LOGIT: f64 = 6.0;
            Array2::<f64>::from_elem((n_obs, k_atoms), SAE_IBP_K1_PRESENT_GATE_LOGIT * tau)
        }
        None => Array2::<f64>::zeros((n_obs, k_atoms)),
    };
    // Data-driven asymmetric cold-start seed (issue #629). A uniform logit
    // init is an exact symmetric saddle for K>=2 exchangeable atoms under
    // softmax / IBP-MAP, so the fit never routes and the decoder overfits
    // through the frozen uniform mixture. Replace the saddle with one EM-style
    // M-then-E step on the seed geometry: prefer the atom that best
    // reconstructs each row. JumpReLU keeps its margin-above-threshold seed
    // (its degeneracy is a closed gate, not a routing tie), and warm-started
    // logits are left exactly as supplied.
    if logits_are_cold && k_atoms > 1 && matches!(assignment_kind.as_str(), "softmax" | "ibp_map") {
        const SAE_RESIDUAL_SEED_GAIN: f64 = 4.0;
        let residual_logits = sae_residual_seed_logits(
            basis_values.view(),
            &basis_sizes,
            z_view,
            SAE_RESIDUAL_SEED_GAIN,
        )
        .map_err(py_value_error)?;
        initial_logits = residual_logits;
    }
    // Wire `random_state` into the optimizer init: jitter the initial
    // assignment logits with a tiny, seed-keyed deterministic perturbation
    // so different seeds explore different Newton trajectories (issue #178).
    // The jitter is uniform in `±SAE_RANDOM_STATE_LOGIT_JITTER` and uses the
    // same Lehmer LCG pattern as the Duchon-center picker, keyed by
    // `random_state`. Fixed seeds still produce bit-identical fits. A
    // warm-started logit seed is left untouched: the encoder's prediction is
    // the requested starting point and the bounded refinement must begin
    // exactly there.
    if n_obs > 0 && k_atoms > 0 && logits_are_cold {
        const SAE_RANDOM_STATE_LOGIT_JITTER: f64 = 1.0e-3;
        let mut state = random_state
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        for row in 0..n_obs {
            for atom_idx in 0..k_atoms {
                state = state
                    .wrapping_mul(6364136223846793005)
                    .wrapping_add(1442695040888963407);
                // Map top 53 bits to a double in [0, 1), then to [-1, 1).
                let u = ((state >> 11) as f64) * f64::from_bits(0x3CA0000000000000);
                let signed = 2.0 * u - 1.0;
                initial_logits[[row, atom_idx]] += SAE_RANDOM_STATE_LOGIT_JITTER * signed;
            }
        }
    }
    // Seed each atom's decoder block by least-squares projection of Z onto
    // the joint atom design weighted by the iter-0 soft assignments. Without
    // this, multi-atom fits collapse to A=0 because the assignment prior
    // dominates a zero-decoder fit — see [`sae_decoder_lsq_init`] for the
    // full diagnosis (issue #174).
    let decoder_coefficients = sae_decoder_lsq_init(
        basis_values.view(),
        &basis_sizes,
        z_view,
        initial_logits.view(),
        assignment_kind.as_str(),
        alpha,
        tau,
        jumprelu_threshold,
    )
    .map_err(py_value_error)?;
    // `plan_latent_dim` (computed above) is the optimizer's per-atom latent
    // dimension — `plan.latent_dim`, not the user-supplied `atom_dim` (periodic
    // atoms carry a harmonic count there; their `latent_dim == 1`).
    let effective_atom_dim: Vec<usize> = plan_latent_dim.clone();
    // Thread the per-atom Duchon centers into the inner driver so its
    // `DuchonCoordinateEvaluator` can re-evaluate `Phi(t)` / `dPhi/dt` at each
    // updated latent coordinate instead of freezing the seed snapshot.
    let atom_centers: Vec<Option<Array2<f64>>> = plans
        .iter()
        .map(|plan| plan.duchon_centers.clone())
        .collect();
    let fisher_u = fisher_factors.as_ref().map(|f| f.as_array());
    let fisher_mr = fisher_mass_residual.as_ref().map(|m| m.as_array());
    let row_w = row_loss_weights.as_ref().map(|w| w.as_array());
    let result_dict = sae_manifold_fit_inner(
        py,
        z_view,
        &atom_basis,
        effective_atom_dim,
        &atom_centers,
        basis_values.view(),
        basis_jacobian.view(),
        basis_sizes.clone(),
        decoder_coefficients.view(),
        smooth_penalties.view(),
        initial_logits.view(),
        start_coords.view(),
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
        native_ard_enabled,
        // Refine the cold routing seed (alternating coordinate projection +
        // weighted decoder LSQ) only when BOTH the logits and the coordinates
        // are cold — i.e. the auto seed is in control. A user-supplied warm
        // start (amortized encoder, #357) is respected verbatim.
        logits_are_cold && initial_coords.is_none(),
        random_state,
        // WP-D → fit wiring (#980): thread the optional output-Fisher shard
        // through so the auto facade installs `RowMetric::OutputFisher` the same
        // magic-by-default way as the precomputed-basis `sae_manifold_fit`.
        // Absent ⇒ the bit-identical Euclidean path.
        fisher_u,
        fisher_mr,
        fisher_provenance.as_deref(),
        row_w,
    )?;
    // #977 — the per-atom `atom_plans` are now emitted by `sae_manifold_fit_inner`
    // FROM THE POST-SEARCH dictionary (variable K), so OOS predict can rebuild the
    // design for the DISCOVERED atoms (births included) without Python. The old
    // post-hoc attachment here re-derived plans from the seed `plans` (input K),
    // which would shadow the grown dictionary with a too-short list and break the
    // `from_payload` zip the moment a birth landed — exactly the plumbing
    // constraint #997 cited. The seed `plans` still build the cold-start design
    // and the per-atom Duchon centers threaded into the inner driver above; they
    // are no longer the serialized plan surface.
    Ok(result_dict)
}

/// Out-of-sample inference: same Newton driver as the fit path, with the
/// trained decoder blocks held frozen across iterations. `decoder_blocks` is
/// a per-atom list of `(M_k, p)` arrays; `duchon_centers` is `Some` only for
/// non-periodic atoms; `n_harmonics_list` is `Some` only for periodic atoms.
///
/// Returns the same full payload dict as the fit path (issue #357): the
/// converged per-token assignments `assignments_z` (N, K), per-atom
/// on-manifold coordinates `on_atom_coords_t`, gating logits, and the
/// reconstruction `fitted`. Downstream supervised heads consume the OOS
/// assignments directly, and the amortized-encoder loop reads the converged
/// coordinates as distillation targets. `initial_logits` (N, K) and
/// `initial_coords` (K, N, D_max) optionally warm-start the OOS refinement
/// from an encoder's per-token prediction.
#[pyfunction(signature = (
    x_new,
    atom_basis,
    atom_dim,
    decoder_blocks,
    duchon_centers,
    n_harmonics_list,
    basis_size_list,
    alpha,
    tau,
    assignment_kind,
    sparsity_strength = 1.0,
    smoothness = 1.0,
    max_iter = 50,
    learning_rate = 0.04,
    ridge_ext_coord = 1.0e-6,
    ridge_beta = 1.0e-6,
    initial_logits = None,
    initial_coords = None,
    jumprelu_threshold = 0.0,
    top_k = None,
))]
fn sae_manifold_predict_oos<'py>(
    py: Python<'py>,
    x_new: PyReadonlyArray2<'py, f64>,
    atom_basis: Vec<String>,
    atom_dim: Vec<usize>,
    decoder_blocks: Vec<PyReadonlyArray2<'py, f64>>,
    duchon_centers: Vec<Option<PyReadonlyArray2<'py, f64>>>,
    n_harmonics_list: Vec<Option<usize>>,
    basis_size_list: Vec<usize>,
    alpha: f64,
    tau: f64,
    assignment_kind: String,
    sparsity_strength: f64,
    smoothness: f64,
    max_iter: usize,
    learning_rate: f64,
    ridge_ext_coord: f64,
    ridge_beta: f64,
    initial_logits: Option<PyReadonlyArray2<'py, f64>>,
    initial_coords: Option<PyReadonlyArray3<'py, f64>>,
    jumprelu_threshold: f64,
    top_k: Option<usize>,
) -> PyResult<Py<PyDict>> {
    let x_view = x_new.as_array();
    let (n_obs, p_out) = x_view.dim();
    let k_atoms = atom_basis.len();
    if n_obs == 0 || p_out == 0 {
        return Err(py_value_error(format!(
            "sae_manifold_predict_oos: x_new must be non-empty; got shape ({n_obs}, {p_out})"
        )));
    }
    if k_atoms == 0 {
        return Err(py_value_error(
            "sae_manifold_predict_oos: atom_basis must be non-empty".into(),
        ));
    }
    if !x_view.iter().all(|v| v.is_finite()) {
        return Err(py_value_error(
            "sae_manifold_predict_oos: x_new contains non-finite values".into(),
        ));
    }
    if atom_dim.len() != k_atoms
        || decoder_blocks.len() != k_atoms
        || duchon_centers.len() != k_atoms
        || n_harmonics_list.len() != k_atoms
        || basis_size_list.len() != k_atoms
    {
        return Err(py_value_error(format!(
            "sae_manifold_predict_oos: per-atom metadata lengths must equal K={k_atoms}"
        )));
    }
    let basis_kinds: Vec<SaeAtomBasisKind> = atom_basis
        .iter()
        .map(|kind| sae_atom_basis_kind_from_str(kind))
        .collect();
    let seed_coords =
        gam::terms::sae::manifold::sae_pca_seed_initial_coords(x_view, &basis_kinds, &atom_dim)
            .map_err(py_value_error)?;
    let mut plans: Vec<SaeAtomBuildPlan> = Vec::with_capacity(k_atoms);
    for atom_idx in 0..k_atoms {
        let kind = basis_kinds[atom_idx].clone();
        let d = atom_dim[atom_idx];
        match kind {
            SaeAtomBasisKind::Periodic => {
                // Periodic atoms are intrinsically 1-D (see comment in
                // `sae_build_atom_plans`); the user-supplied `atom_dim` only
                // ever selects the harmonic count when no explicit
                // `n_harmonics_list[k]` is supplied. Force the optimizer
                // latent dimension to 1 so the analytic harmonic evaluator
                // remains live during OOS prediction.
                let n_harmonics = n_harmonics_list[atom_idx].unwrap_or(d.max(1));
                let basis_size = sae_periodic_basis_size(n_harmonics).map_err(py_value_error)?;
                plans.push(SaeAtomBuildPlan {
                    kind: SaeAtomBasisKind::Periodic,
                    latent_dim: 1,
                    n_harmonics,
                    duchon_centers: None,
                    basis_size,
                });
            }
            SaeAtomBasisKind::Sphere => {
                if d != 2 {
                    return Err(py_value_error(format!(
                        "sae_manifold_predict_oos: atom {atom_idx} basis 'sphere' requires atom_dim == 2, got {d}"
                    )));
                }
                plans.push(SaeAtomBuildPlan {
                    kind: SaeAtomBasisKind::Sphere,
                    latent_dim: 2,
                    n_harmonics: 0,
                    duchon_centers: None,
                    basis_size: SAE_SPHERE_BASIS_SIZE,
                });
            }
            SaeAtomBasisKind::Torus => {
                let h = n_harmonics_list[atom_idx].unwrap_or(SAE_DEFAULT_TORUS_HARMONICS);
                let evaluator = TorusHarmonicEvaluator::new(d, h).map_err(py_value_error)?;
                let basis_size = evaluator.basis_size();
                plans.push(SaeAtomBuildPlan {
                    kind: SaeAtomBasisKind::Torus,
                    latent_dim: d,
                    n_harmonics: h,
                    duchon_centers: None,
                    basis_size,
                });
            }
            _ => {
                let centers = duchon_centers[atom_idx]
                    .as_ref()
                    .ok_or_else(|| {
                        py_value_error(format!(
                            "sae_manifold_predict_oos: atom {atom_idx} (Duchon-like) needs duchon_centers"
                        ))
                    })?
                    .as_array()
                    .to_owned();
                let probe_pts = Array2::<f64>::zeros((1, d.max(1)));
                let (phi, _jet, _penalty) = match kind {
                    SaeAtomBasisKind::EuclideanPatch => {
                        // #1132 bug 3: the monomial WIDTH of a Euclidean patch is
                        // `monomial_exponents(dim, degree).len()` where `dim` is
                        // the build dimension `sae_build_euclidean_atom_with_degree`
                        // reads off `centers.ncols()` — NOT the user-facing
                        // `atom_dim`. Recovering the degree from `(atom_dim,
                        // basis_size_list)` while the build uses `centers.ncols()`
                        // re-emits a different width when those two disagree (the
                        // "decoder_blocks[0] has M=1 but rebuilt basis has M=3"
                        // euclidean OOS reconstruct failure). Anchor the recovery
                        // to the TRAINED decoder block's row count `M` against the
                        // centers' own dimension so the rebuilt basis width matches
                        // the decoder bit-for-bit and the M-consistency guard below
                        // passes.
                        let euclidean_dim = centers.ncols();
                        let trained_m = decoder_blocks[atom_idx].as_array().nrows();
                        let degree = sae_euclidean_degree_for_basis_size(euclidean_dim, trained_m)
                            .map_err(py_value_error)?;
                        sae_build_euclidean_atom_with_degree(
                            probe_pts.view(),
                            centers.view(),
                            degree,
                        )
                        .map_err(py_value_error)?
                    }
                    _ => sae_build_duchon_atom(probe_pts.view(), centers.view())
                        .map_err(py_value_error)?,
                };
                let basis_size = phi.ncols();
                plans.push(SaeAtomBuildPlan {
                    kind,
                    latent_dim: d,
                    n_harmonics: 0,
                    duchon_centers: Some(centers),
                    basis_size,
                });
            }
        }
    }
    // Warm-start coordinates (issue #357): an amortized encoder's predicted
    // per-token on-manifold `t` seeds the OOS Newton refinement. When absent
    // the PCA seed of `x_new` is the cold start. Layout `(K, N, D_max)` with
    // `D_max` covering every atom's `atom_dim`.
    let start_coords: Array3<f64> = match &initial_coords {
        Some(arr) => {
            let view = arr.as_array();
            let shape = view.shape();
            if shape.len() != 3 {
                return Err(py_value_error(
                    "sae_manifold_predict_oos: initial_coords must be a rank-3 (K, N, D_max) array"
                        .to_string(),
                ));
            }
            if shape[0] != k_atoms || shape[1] != n_obs {
                return Err(py_value_error(format!(
                    "sae_manifold_predict_oos: initial_coords must start with (K, N)=({k_atoms}, {n_obs}); got {shape:?}"
                )));
            }
            let max_dim = *shape.get(2).unwrap_or(&0);
            for (atom_idx, &d) in atom_dim.iter().enumerate() {
                if d > max_dim {
                    return Err(py_value_error(format!(
                        "sae_manifold_predict_oos: initial_coords D_max={max_dim} is too small for atom {atom_idx} atom_dim={d}"
                    )));
                }
            }
            if !view.iter().all(|v| v.is_finite()) {
                return Err(py_value_error(
                    "sae_manifold_predict_oos: initial_coords contains non-finite values".into(),
                ));
            }
            view.to_owned()
        }
        None => seed_coords.clone(),
    };
    let (basis_values, basis_jacobian, smooth_penalties, basis_sizes, _coord_blocks) =
        sae_build_padded_basis_stacks(&plans, start_coords.view(), n_obs)
            .map_err(py_value_error)?;
    let m_max = basis_sizes.iter().copied().max().unwrap_or(1).max(1);
    // Pad trained decoder blocks into (K, M_max, p)
    let mut decoder_coefficients = Array3::<f64>::zeros((k_atoms, m_max, p_out));
    for atom_idx in 0..k_atoms {
        let block = decoder_blocks[atom_idx].as_array();
        if block.ncols() != p_out {
            return Err(py_value_error(format!(
                "sae_manifold_predict_oos: decoder_blocks[{atom_idx}] has p={} but x_new has p={p_out}",
                block.ncols()
            )));
        }
        if block.nrows() != basis_sizes[atom_idx] {
            return Err(py_value_error(format!(
                "sae_manifold_predict_oos: decoder_blocks[{atom_idx}] has M={} but rebuilt basis has M={}; \
                 atom_basis / atom_dim / n_harmonics_list / duchon_centers must match the trained design",
                block.nrows(),
                basis_sizes[atom_idx]
            )));
        }
        if !block.iter().all(|v| v.is_finite()) {
            return Err(py_value_error(format!(
                "sae_manifold_predict_oos: decoder_blocks[{atom_idx}] contains non-finite values"
            )));
        }
        let m_k = basis_sizes[atom_idx];
        decoder_coefficients
            .slice_mut(s![atom_idx, 0..m_k, 0..p_out])
            .assign(&block.slice(s![0..m_k, 0..p_out]));
    }
    let logits_are_warm = initial_logits.is_some();
    let initial_logits = match &initial_logits {
        Some(arr) => {
            let view = arr.as_array();
            if view.dim() != (n_obs, k_atoms) {
                return Err(py_value_error(format!(
                    "sae_manifold_predict_oos: initial_logits must be ({n_obs}, {k_atoms}); got {:?}",
                    view.dim()
                )));
            }
            if !view.iter().all(|v| v.is_finite()) {
                return Err(py_value_error(
                    "sae_manifold_predict_oos: initial_logits contains non-finite values".into(),
                ));
            }
            view.to_owned()
        }
        None => {
            let mut logits = Array2::<f64>::zeros((n_obs, k_atoms));
            if k_atoms == 1 && assignment_kind == "ibp_map" {
                for row in 0..n_obs {
                    logits[[row, 0]] = 4.0;
                }
            }
            logits
        }
    };
    if let Some(k_top) = top_k {
        if k_top == 0 || k_top > k_atoms {
            return Err(py_value_error(format!(
                "sae_manifold_predict_oos: top_k must satisfy 1 <= top_k <= k_atoms={k_atoms}; got {k_top}"
            )));
        }
    }
    for (name, value) in [
        ("alpha", alpha),
        ("tau", tau),
        ("smoothness", smoothness),
        ("learning_rate", learning_rate),
        ("ridge_ext_coord", ridge_ext_coord),
        ("ridge_beta", ridge_beta),
    ] {
        if !value.is_finite() || value <= 0.0 {
            return Err(py_value_error(format!(
                "sae_manifold_predict_oos: {name} must be finite and positive; got {value}"
            )));
        }
    }
    if !sparsity_strength.is_finite() || sparsity_strength < 0.0 {
        return Err(py_value_error(format!(
            "sae_manifold_predict_oos: sparsity_strength must be finite and non-negative; got {sparsity_strength}"
        )));
    }
    const SPARSITY_DISABLED_FLOOR: f64 = 1.0e-300;
    let sparsity_strength = if sparsity_strength == 0.0 {
        SPARSITY_DISABLED_FLOOR
    } else {
        sparsity_strength
    };

    let mode = match assignment_kind.as_str() {
        "softmax" => AssignmentMode::softmax(tau),
        "ibp_map" => AssignmentMode::ibp_map(tau, alpha, false),
        "jumprelu" => AssignmentMode::jumprelu(tau, jumprelu_threshold),
        _ => {
            return Err(py_value_error(format!(
                "sae_manifold_predict_oos: assignment_kind must be one of 'softmax', 'ibp_map', or 'jumprelu'; got {assignment_kind}"
            )));
        }
    };
    let effective_atom_dim: Vec<usize> = plans.iter().map(|plan| plan.latent_dim).collect();
    let mut coord_blocks = Vec::with_capacity(k_atoms);
    for atom_idx in 0..k_atoms {
        let d = effective_atom_dim[atom_idx];
        coord_blocks.push(start_coords.slice(s![atom_idx, 0..n_obs, 0..d]).to_owned());
    }
    let atom_centers: Vec<Option<Array2<f64>>> = plans
        .iter()
        .map(|plan| plan.duchon_centers.clone())
        .collect();
    let evaluators = build_sae_basis_evaluators(
        &basis_kinds,
        &basis_sizes,
        &effective_atom_dim,
        &coord_blocks,
        &atom_centers,
    )
    .map_err(py_value_error)?;
    let mut term = term_from_padded_blocks_with_mode(
        n_obs,
        p_out,
        &basis_kinds,
        basis_values.view(),
        basis_jacobian.view(),
        &basis_sizes,
        &effective_atom_dim,
        decoder_coefficients.view(),
        smooth_penalties.view(),
        initial_logits.view(),
        &coord_blocks,
        mode,
        &evaluators,
    )
    .map_err(py_value_error)?;
    // Global decoder-projection coordinate seed (cold start only). With the
    // decoder frozen, the exact OOS latent of each row is its projection onto
    // the atom's image manifold, `argmin_t ‖x − Φ(t)·B‖²`. That objective is
    // non-convex on the compact periodic / torus / sphere latents, so the
    // PCA-`atan2` seed plus a few Newton steps lands many rows in the wrong
    // basin and mis-routes them (negative R², near-uniform assignments). Seeding
    // from a dense manifold grid puts every row in the correct basin before the
    // residual-based softmax logit seed and the Newton refinement below. An
    // explicit warm `initial_coords` (e.g. from an amortized encoder, issue
    // #357) is respected as-is and not overwritten.
    if initial_coords.is_none() {
        term.seed_coords_by_decoder_projection(x_view, SAE_OOS_PROJECTION_GRID_RESOLUTION)
            .map_err(py_value_error)?;
    }
    if !logits_are_warm && assignment_kind == "softmax" {
        seed_oos_softmax_logits_from_projection_residuals(&mut term, x_view, tau);
    } else if !logits_are_warm && assignment_kind == "ibp_map" {
        seed_oos_ibp_logits_from_projected_decoder_lsq(&mut term, x_view, tau, alpha);
    }
    let log_ard: Vec<Array1<f64>> = effective_atom_dim
        .iter()
        .map(|&d| Array1::<f64>::zeros(d))
        .collect();
    let mut rho = SaeManifoldRho::new(sparsity_strength.ln(), smoothness.ln(), log_ard);
    let loss = term
        .run_fixed_decoder_arrow_schur(
            x_view,
            &mut rho,
            None,
            max_iter,
            learning_rate,
            ridge_ext_coord,
        )
        .map_err(py_value_error)?;

    if !logits_are_warm && assignment_kind == "softmax" {
        seed_oos_softmax_logits_from_projection_residuals(&mut term, x_view, tau);
    } else if !logits_are_warm && assignment_kind == "ibp_map" {
        seed_oos_ibp_logits_from_projected_decoder_lsq(&mut term, x_view, tau, alpha);
    }
    let mut assignments = term.assignment.assignments();
    let mut fitted = term.fitted();
    if let Some(k_top) = top_k {
        if k_top < k_atoms {
            let renormalise = assignment_kind == "softmax";
            for row in 0..n_obs {
                let mut paired: Vec<(f64, usize)> = (0..k_atoms)
                    .map(|atom_idx| (assignments[[row, atom_idx]], atom_idx))
                    .collect();
                paired.sort_by(|a, b| {
                    b.0.partial_cmp(&a.0)
                        .unwrap_or(std::cmp::Ordering::Equal)
                        .then(a.1.cmp(&b.1))
                });
                let mut keep = vec![false; k_atoms];
                for &(_, atom_idx) in paired.iter().take(k_top) {
                    keep[atom_idx] = true;
                }
                if renormalise {
                    let kept_sum: f64 = (0..k_atoms)
                        .filter(|&atom_idx| keep[atom_idx])
                        .map(|atom_idx| assignments[[row, atom_idx]])
                        .sum();
                    if !(kept_sum.is_finite() && kept_sum > 0.0) {
                        return Err(py_value_error(format!(
                            "sae_manifold_predict_oos: top_k softmax projection has non-positive kept mass on row {row}"
                        )));
                    }
                    for atom_idx in 0..k_atoms {
                        assignments[[row, atom_idx]] = if keep[atom_idx] {
                            assignments[[row, atom_idx]] / kept_sum
                        } else {
                            0.0
                        };
                    }
                } else {
                    for atom_idx in 0..k_atoms {
                        if !keep[atom_idx] {
                            assignments[[row, atom_idx]] = 0.0;
                        }
                    }
                }
            }
            fitted = Array2::<f64>::zeros((n_obs, p_out));
            let mut g_buf = vec![0.0_f64; p_out];
            for row in 0..n_obs {
                for atom_idx in 0..k_atoms {
                    let a_k = assignments[[row, atom_idx]];
                    if a_k == 0.0 {
                        continue;
                    }
                    term.atoms[atom_idx].fill_decoded_row(row, &mut g_buf);
                    let mut out_row = fitted.row_mut(row);
                    for out_col in 0..p_out {
                        out_row[out_col] += a_k * g_buf[out_col];
                    }
                }
            }
        }
    }

    let log_ard_py = PyList::empty(py);
    for atom_log_ard in &rho.log_ard {
        log_ard_py.append(atom_log_ard.clone().into_pyarray(py))?;
    }
    let atoms_py = PyList::empty(py);
    for atom_idx in 0..k_atoms {
        let atom = &term.atoms[atom_idx];
        let atom_dict = PyDict::new(py);
        atom_dict.set_item(
            "decoder_B",
            atom.decoder_coefficients.clone().into_pyarray(py),
        )?;
        atom_dict.set_item("basis_kind", atom_basis[atom_idx].clone())?;
        atom_dict.set_item("basis_centers", py.None())?;
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
        atom_dict.set_item("active_dim", effective_atom_dim[atom_idx])?;
        atoms_py.append(atom_dict)?;
    }

    let active_mask: Vec<bool> = (0..k_atoms)
        .map(|atom_idx| assignments.column(atom_idx).sum() > 1.0e-8)
        .collect();
    let out = PyDict::new(py);
    out.set_item("atoms", atoms_py)?;
    out.set_item("assignments_z", assignments.into_pyarray(py))?;
    out.set_item("logits", term.assignment.logits.clone().into_pyarray(py))?;
    out.set_item("atom_active_mask", active_mask)?;
    out.set_item("fitted", fitted.into_pyarray(py))?;
    out.set_item("reml_score", loss.evidence_proxy())?;
    out.set_item("log_alpha", alpha.ln())?;
    out.set_item("log_lambda_smooth", rho.log_lambda_smooth)?;
    out.set_item("log_ard", log_ard_py)?;
    out.set_item("assignment_prior", assignment_kind)?;
    out.set_item(
        "solver_plan",
        sae_streaming_plan_to_pydict(py, term.streaming_plan())?,
    )?;
    out.set_item("chosen_k", k_atoms)?;
    Ok(out.unbind())
}

/// Compute a steering plan with output dosimetry for a fitted SAE-manifold atom
/// ([`gam::inference::steering::steer_delta`]).
///
/// This is the FFI surface for the steering primitive: it rebuilds the fitted
/// [`gam::terms::sae::manifold::SaeManifoldTerm`] from the trained decoder blocks
/// + basis metadata, seeds it with the *trained* on-atom coordinates and routing
/// logits (no re-solve — the model is fixed), optionally installs the WP-D
/// per-row output-Fisher metric ([`gam::inference::row_metric::RowMetric::output_fisher`])
/// from `fisher_factors` (the same shard the fit used), and calls `steer_delta`
/// to drive atom `atom_k` from `t_from` to `t_to`. It returns the
/// [`gam::inference::steering::SteerPlan`] fields as a dict: the activation-space
/// `delta`, the path-integrated `predicted_nats` dose, the `validity_radius`,
/// the `off_manifold_norm` self-check, and the `metric_provenance`.
///
/// The term rebuild mirrors [`sae_manifold_predict_oos`] (same plan/evaluator
/// machinery), but where `predict_oos` runs the frozen-decoder Newton solve on a
/// *new* `X`, this seeds the term directly from the trained latents/logits so the
/// dose is measured through the model as fitted. `coords` is one `(N, d_k)` array
/// per atom (the trained `on_atom_coords_t`); `logits` is `(N, K)` (the trained
/// routing logits) so the per-atom amplitude / measured-row selection inside
/// `steer_delta` sees the fitted assignments. `fisher_factors` is the `(n, p, r)`
/// harvest shard `U`; its presence installs `RowMetric::OutputFisher` (and makes
/// `predicted_nats` / `validity_radius` available), exactly as in the fit.
#[pyfunction(signature = (
    atom_k,
    t_from,
    t_to,
    n_obs,
    p_out,
    atom_basis,
    atom_dim,
    decoder_blocks,
    duchon_centers,
    n_harmonics_list,
    basis_size_list,
    coords,
    logits,
    assignment_kind,
    tau,
    alpha = 1.0,
    jumprelu_threshold = 0.0,
    fisher_factors = None,
    fisher_provenance = None,
))]
#[allow(clippy::too_many_arguments)]
fn sae_steer_delta<'py>(
    py: Python<'py>,
    atom_k: usize,
    t_from: PyReadonlyArray1<'py, f64>,
    t_to: PyReadonlyArray1<'py, f64>,
    n_obs: usize,
    p_out: usize,
    atom_basis: Vec<String>,
    atom_dim: Vec<usize>,
    decoder_blocks: Vec<PyReadonlyArray2<'py, f64>>,
    duchon_centers: Vec<Option<PyReadonlyArray2<'py, f64>>>,
    n_harmonics_list: Vec<Option<usize>>,
    basis_size_list: Vec<usize>,
    coords: Vec<PyReadonlyArray2<'py, f64>>,
    logits: PyReadonlyArray2<'py, f64>,
    assignment_kind: String,
    tau: f64,
    alpha: f64,
    jumprelu_threshold: f64,
    fisher_factors: Option<PyReadonlyArray3<'py, f64>>,
    // Harvest provenance tag (#980): same-position `"output_fisher"` (default) or
    // forward-looking `"output_fisher_downstream"`. Selects the re-installed
    // output-Fisher `RowMetric` the dose is measured through.
    fisher_provenance: Option<String>,
) -> PyResult<Py<PyDict>> {
    let k_atoms = atom_basis.len();
    if n_obs == 0 || p_out == 0 {
        return Err(py_value_error(format!(
            "sae_steer_delta: n_obs and p_out must be positive; got ({n_obs}, {p_out})"
        )));
    }
    if k_atoms == 0 {
        return Err(py_value_error(
            "sae_steer_delta: atom_basis must be non-empty".into(),
        ));
    }
    if atom_k >= k_atoms {
        return Err(py_value_error(format!(
            "sae_steer_delta: atom_k={atom_k} out of range for K={k_atoms} atoms"
        )));
    }
    if atom_dim.len() != k_atoms
        || decoder_blocks.len() != k_atoms
        || duchon_centers.len() != k_atoms
        || n_harmonics_list.len() != k_atoms
        || basis_size_list.len() != k_atoms
        || coords.len() != k_atoms
    {
        return Err(py_value_error(format!(
            "sae_steer_delta: per-atom metadata lengths must equal K={k_atoms}"
        )));
    }
    let basis_kinds: Vec<SaeAtomBasisKind> = atom_basis
        .iter()
        .map(|kind| sae_atom_basis_kind_from_str(kind))
        .collect();

    // Build the per-atom plans (latent dim / basis size / centers), reusing the
    // exact rebuild path `sae_manifold_predict_oos` uses so the rebuilt design
    // matches the trained one.
    let mut plans: Vec<SaeAtomBuildPlan> = Vec::with_capacity(k_atoms);
    for atom_idx in 0..k_atoms {
        let kind = basis_kinds[atom_idx].clone();
        let d = atom_dim[atom_idx];
        match kind {
            SaeAtomBasisKind::Periodic => {
                let n_harmonics = n_harmonics_list[atom_idx].unwrap_or(d.max(1));
                let basis_size = sae_periodic_basis_size(n_harmonics).map_err(py_value_error)?;
                plans.push(SaeAtomBuildPlan {
                    kind: SaeAtomBasisKind::Periodic,
                    latent_dim: 1,
                    n_harmonics,
                    duchon_centers: None,
                    basis_size,
                });
            }
            SaeAtomBasisKind::Sphere => {
                if d != 2 {
                    return Err(py_value_error(format!(
                        "sae_steer_delta: atom {atom_idx} basis 'sphere' requires atom_dim == 2, got {d}"
                    )));
                }
                plans.push(SaeAtomBuildPlan {
                    kind: SaeAtomBasisKind::Sphere,
                    latent_dim: 2,
                    n_harmonics: 0,
                    duchon_centers: None,
                    basis_size: SAE_SPHERE_BASIS_SIZE,
                });
            }
            SaeAtomBasisKind::Torus => {
                let h = n_harmonics_list[atom_idx].unwrap_or(SAE_DEFAULT_TORUS_HARMONICS);
                let evaluator = TorusHarmonicEvaluator::new(d, h).map_err(py_value_error)?;
                let basis_size = evaluator.basis_size();
                plans.push(SaeAtomBuildPlan {
                    kind: SaeAtomBasisKind::Torus,
                    latent_dim: d,
                    n_harmonics: h,
                    duchon_centers: None,
                    basis_size,
                });
            }
            _ => {
                let centers = duchon_centers[atom_idx]
                    .as_ref()
                    .ok_or_else(|| {
                        py_value_error(format!(
                            "sae_steer_delta: atom {atom_idx} (Duchon-like) needs duchon_centers"
                        ))
                    })?
                    .as_array()
                    .to_owned();
                let probe_pts = Array2::<f64>::zeros((1, d.max(1)));
                let (phi, _jet, _penalty) = match kind {
                    SaeAtomBasisKind::EuclideanPatch => {
                        // #1132 bug 3: recover the Euclidean patch degree from the
                        // TRAINED decoder block's `M` against the centers' own
                        // dimension (the build dim is `centers.ncols()`), not from
                        // `(atom_dim, basis_size_list)` which can disagree and
                        // re-emit a wrong width. Keeps the steer rebuild's basis
                        // width matched to the decoder, same as predict_oos.
                        let euclidean_dim = centers.ncols();
                        let trained_m = decoder_blocks[atom_idx].as_array().nrows();
                        let degree = sae_euclidean_degree_for_basis_size(euclidean_dim, trained_m)
                            .map_err(py_value_error)?;
                        sae_build_euclidean_atom_with_degree(
                            probe_pts.view(),
                            centers.view(),
                            degree,
                        )
                        .map_err(py_value_error)?
                    }
                    _ => sae_build_duchon_atom(probe_pts.view(), centers.view())
                        .map_err(py_value_error)?,
                };
                let basis_size = phi.ncols();
                plans.push(SaeAtomBuildPlan {
                    kind,
                    latent_dim: d,
                    n_harmonics: 0,
                    duchon_centers: Some(centers),
                    basis_size,
                });
            }
        }
    }
    let effective_atom_dim: Vec<usize> = plans.iter().map(|plan| plan.latent_dim).collect();

    // Pack the trained per-atom coordinates into the padded (K, N, D_max) buffer
    // the basis-stack builder expects, validating each block against the rebuilt
    // latent dim. These are the *trained* coords — no solve.
    let d_max = effective_atom_dim.iter().copied().max().unwrap_or(1).max(1);
    let mut start_coords = Array3::<f64>::zeros((k_atoms, n_obs, d_max));
    for atom_idx in 0..k_atoms {
        let block = coords[atom_idx].as_array();
        let d = effective_atom_dim[atom_idx];
        if block.nrows() != n_obs || block.ncols() != d {
            return Err(py_value_error(format!(
                "sae_steer_delta: coords[{atom_idx}] must be (N, d)=({n_obs}, {d}); got {:?}",
                block.dim()
            )));
        }
        if !block.iter().all(|v| v.is_finite()) {
            return Err(py_value_error(format!(
                "sae_steer_delta: coords[{atom_idx}] contains non-finite values"
            )));
        }
        start_coords
            .slice_mut(s![atom_idx, 0..n_obs, 0..d])
            .assign(&block);
    }

    let (basis_values, basis_jacobian, smooth_penalties, basis_sizes, _coord_blocks) =
        sae_build_padded_basis_stacks(&plans, start_coords.view(), n_obs)
            .map_err(py_value_error)?;
    let m_max = basis_sizes.iter().copied().max().unwrap_or(1).max(1);
    let mut decoder_coefficients = Array3::<f64>::zeros((k_atoms, m_max, p_out));
    for atom_idx in 0..k_atoms {
        let block = decoder_blocks[atom_idx].as_array();
        if block.ncols() != p_out {
            return Err(py_value_error(format!(
                "sae_steer_delta: decoder_blocks[{atom_idx}] has p={} but p_out={p_out}",
                block.ncols()
            )));
        }
        if block.nrows() != basis_sizes[atom_idx] {
            return Err(py_value_error(format!(
                "sae_steer_delta: decoder_blocks[{atom_idx}] has M={} but rebuilt basis has M={}; \
                 atom_basis / atom_dim / n_harmonics_list / duchon_centers must match the trained design",
                block.nrows(),
                basis_sizes[atom_idx]
            )));
        }
        if !block.iter().all(|v| v.is_finite()) {
            return Err(py_value_error(format!(
                "sae_steer_delta: decoder_blocks[{atom_idx}] contains non-finite values"
            )));
        }
        let m_k = basis_sizes[atom_idx];
        decoder_coefficients
            .slice_mut(s![atom_idx, 0..m_k, 0..p_out])
            .assign(&block.slice(s![0..m_k, 0..p_out]));
    }

    let logits_view = logits.as_array();
    if logits_view.dim() != (n_obs, k_atoms) {
        return Err(py_value_error(format!(
            "sae_steer_delta: logits must be ({n_obs}, {k_atoms}); got {:?}",
            logits_view.dim()
        )));
    }
    if !logits_view.iter().all(|v| v.is_finite()) {
        return Err(py_value_error(
            "sae_steer_delta: logits contains non-finite values".into(),
        ));
    }
    let initial_logits = logits_view.to_owned();

    for (name, value) in [("alpha", alpha), ("tau", tau)] {
        if !value.is_finite() || value <= 0.0 {
            return Err(py_value_error(format!(
                "sae_steer_delta: {name} must be finite and positive; got {value}"
            )));
        }
    }
    let mode = match assignment_kind.as_str() {
        "softmax" => AssignmentMode::softmax(tau),
        "ibp_map" => AssignmentMode::ibp_map(tau, alpha, false),
        "jumprelu" => AssignmentMode::jumprelu(tau, jumprelu_threshold),
        _ => {
            return Err(py_value_error(format!(
                "sae_steer_delta: assignment_kind must be one of 'softmax', 'ibp_map', or 'jumprelu'; got {assignment_kind}"
            )));
        }
    };
    let mut coord_blocks = Vec::with_capacity(k_atoms);
    for atom_idx in 0..k_atoms {
        let d = effective_atom_dim[atom_idx];
        coord_blocks.push(start_coords.slice(s![atom_idx, 0..n_obs, 0..d]).to_owned());
    }
    let atom_centers: Vec<Option<Array2<f64>>> = plans
        .iter()
        .map(|plan| plan.duchon_centers.clone())
        .collect();
    let evaluators = build_sae_basis_evaluators(
        &basis_kinds,
        &basis_sizes,
        &effective_atom_dim,
        &coord_blocks,
        &atom_centers,
    )
    .map_err(py_value_error)?;
    let mut term = term_from_padded_blocks_with_mode(
        n_obs,
        p_out,
        &basis_kinds,
        basis_values.view(),
        basis_jacobian.view(),
        &basis_sizes,
        &effective_atom_dim,
        decoder_coefficients.view(),
        smooth_penalties.view(),
        initial_logits.view(),
        &coord_blocks,
        mode,
        &evaluators,
    )
    .map_err(py_value_error)?;

    // Install the WP-D per-row output-Fisher metric if a shard was supplied — the
    // same `(n, p, r)` → `(n, p*r)` flatten the fit uses. Presence activates the
    // OutputFisher provenance so `predicted_nats` / `validity_radius` are
    // available; absence keeps the Euclidean geometry-only path (dose = None).
    if let Some(u3) = fisher_factors.as_ref().map(|f| f.as_array()) {
        let u_shape = u3.shape();
        if u_shape[0] != n_obs || u_shape[1] != p_out {
            return Err(py_value_error(format!(
                "sae_steer_delta: fisher_factors U must be (n, p, r)=({n_obs}, {p_out}, r); \
                 got leading dims ({}, {})",
                u_shape[0], u_shape[1]
            )));
        }
        let rank = u_shape[2];
        if rank == 0 {
            return Err(py_value_error(
                "sae_steer_delta: fisher_factors U rank (last axis) must be >= 1".to_string(),
            ));
        }
        if rank > p_out {
            return Err(py_value_error(format!(
                "sae_steer_delta: fisher_factors U rank {rank} exceeds output dim p={p_out}"
            )));
        }
        if !u3.iter().all(|v| v.is_finite()) {
            return Err(py_value_error(
                "sae_steer_delta: fisher_factors U contains non-finite values".into(),
            ));
        }
        let mut u_flat = Array2::<f64>::zeros((n_obs, p_out * rank));
        for row in 0..n_obs {
            for i in 0..p_out {
                for k in 0..rank {
                    u_flat[[row, i * rank + k]] = u3[[row, i, k]];
                }
            }
        }
        let metric =
            row_metric_from_fisher_provenance(u_flat, p_out, rank, fisher_provenance.as_deref())?;
        term.set_row_metric(metric).map_err(py_value_error)?;
    }

    // The metric the dose is measured through: the installed per-row metric, or a
    // bit-identical Euclidean metric (geometry-only; dose degrades to None).
    let euclidean =
        gam::inference::row_metric::RowMetric::euclidean(n_obs, p_out).map_err(py_value_error)?;
    let metric = term.row_metric().unwrap_or(&euclidean);

    let t_from_vec = t_from.as_array().to_vec();
    let t_to_vec = t_to.as_array().to_vec();
    let plan = gam::inference::steering::steer_delta(&term, metric, atom_k, &t_from_vec, &t_to_vec)
        .map_err(py_value_error)?;

    let provenance_str = metric_provenance_label(plan.metric_provenance);

    let out = PyDict::new(py);
    out.set_item("atom", plan.atom)?;
    out.set_item("atom_name", plan.atom_name)?;
    out.set_item("t_from", plan.t_from)?;
    out.set_item("t_to", plan.t_to)?;
    out.set_item("amplitude", plan.amplitude)?;
    out.set_item("measured_row", plan.measured_row)?;
    out.set_item("delta", plan.delta.into_pyarray(py))?;
    out.set_item("predicted_nats", plan.predicted_nats)?;
    out.set_item("validity_radius", plan.validity_radius)?;
    out.set_item("off_manifold_norm", plan.off_manifold_norm)?;
    out.set_item("metric_provenance", provenance_str)?;
    Ok(out.unbind())
}

/// Global coefficient of determination
/// R^2 = 1 - (Σ_ij (y_ij - ŷ_ij)²) / (Σ_ij (y_ij - mean_j)²)
/// for a fitted SAE-manifold reconstruction, where `mean_j` is the per-column
/// mean of the observed matrix. Both SSR and SST are summed across all rows and
/// columns, so this returns a single scalar (a global metric), not a vector of
/// per-column R² values. Pure-Rust closed-form so the Python wrapper is one FFI
/// call.
#[pyfunction]
fn sae_manifold_reconstruction_r2(
    observed: PyReadonlyArray2<'_, f64>,
    fitted: PyReadonlyArray2<'_, f64>,
) -> PyResult<f64> {
    let observed = observed.as_array();
    let fitted = fitted.as_array();
    if observed.dim() != fitted.dim() {
        return Err(py_value_error(format!(
            "sae_manifold_reconstruction_r2: shape mismatch observed={:?} fitted={:?}",
            observed.dim(),
            fitted.dim(),
        )));
    }
    let n_rows = observed.nrows();
    let n_cols = observed.ncols();
    if n_rows == 0 || n_cols == 0 {
        return Err(py_value_error(
            "sae_manifold_reconstruction_r2: observed and fitted must be non-empty".into(),
        ));
    }
    if !observed.iter().all(|v| v.is_finite()) {
        return Err(py_value_error(
            "sae_manifold_reconstruction_r2: observed contains non-finite values".into(),
        ));
    }
    if !fitted.iter().all(|v| v.is_finite()) {
        return Err(py_value_error(
            "sae_manifold_reconstruction_r2: fitted contains non-finite values".into(),
        ));
    }
    let mut col_means = vec![0.0_f64; n_cols];
    for col in 0..n_cols {
        let mut acc = 0.0;
        for row in 0..n_rows {
            acc += observed[[row, col]];
        }
        col_means[col] = acc / n_rows as f64;
    }
    let mut ssr = 0.0_f64;
    let mut sst = 0.0_f64;
    for row in 0..n_rows {
        for col in 0..n_cols {
            let d = observed[[row, col]] - fitted[[row, col]];
            ssr += d * d;
            let dm = observed[[row, col]] - col_means[col];
            sst += dm * dm;
        }
    }
    if !ssr.is_finite() || !sst.is_finite() {
        return Err(py_value_error(
            "sae_manifold_reconstruction_r2: SSR/SST overflowed; inputs have extreme magnitudes"
                .into(),
        ));
    }
    if sst == 0.0 {
        return Ok(f64::NAN);
    }
    Ok(1.0 - ssr / sst)
}

/// Sparsity summary stats for an `(n_rows, K)` assignment matrix returned by
/// `sae_manifold_fit*`. Returns `(avg_active_atoms, mean_assignment_mass)` where
/// "active" is `assignment >= threshold`.
#[pyfunction]
fn sae_manifold_assignment_summary(
    assignments: PyReadonlyArray2<'_, f64>,
    threshold: f64,
) -> PyResult<(f64, f64)> {
    if !threshold.is_finite() {
        return Err(py_value_error(
            "sae_manifold_assignment_summary: threshold must be finite".into(),
        ));
    }
    let a = assignments.as_array();
    let (n_rows, k) = a.dim();
    if n_rows == 0 || k == 0 {
        return Err(py_value_error(
            "sae_manifold_assignment_summary: assignments must be non-empty".into(),
        ));
    }
    let n_entries = n_rows.checked_mul(k).ok_or_else(|| {
        py_value_error("sae_manifold_assignment_summary: assignments shape is too large".into())
    })?;
    let mut active_total = 0_usize;
    let mut mass_total = 0.0_f64;
    for row in 0..n_rows {
        for col in 0..k {
            let assignment = a[[row, col]];
            if !assignment.is_finite() {
                return Err(py_value_error(format!(
                    "sae_manifold_assignment_summary: non-finite assignment at ({row}, {col})"
                )));
            }
            mass_total += assignment;
            if !mass_total.is_finite() {
                return Err(py_value_error(
                    "sae_manifold_assignment_summary: assignment mass overflowed".into(),
                ));
            }
            if assignment >= threshold {
                active_total += 1;
            }
        }
    }
    let avg_active = active_total as f64 / n_rows as f64;
    let mean_mass = mass_total / n_entries as f64;
    Ok((avg_active, mean_mass))
}

#[pyfunction(signature = (x, w_gate, w_amp))]
fn gated_sae_decode<'py>(
    py: Python<'py>,
    x: PyReadonlyArray2<'py, f64>,
    w_gate: PyReadonlyArray2<'py, f64>,
    w_amp: PyReadonlyArray2<'py, f64>,
) -> PyResult<Py<PyArray2<f64>>> {
    let decoder = GatedSAEDecoder::new(w_gate.as_array().to_owned(), w_amp.as_array().to_owned())
        .map_err(py_value_error)?;
    let out = decoder.decode_batch(x.as_array()).map_err(py_value_error)?;
    Ok(out.into_pyarray(py).unbind())
}

/// Forward of the per-feature scalar-gate decoder (no swap).
///
/// Returns `X̂[i, d] = Σ_f gate[f] · z[i, f] · weights[d, f] + bias[d]`.
/// `bias` may be `None`. The forward and its analytic gradients are shared
/// across the Rust library, the CLI, and the PyTorch bridge via the
/// `gam::terms::decoders::interchange_decoder` primitive.
#[pyfunction(signature = (z, weights, gate, bias = None))]
fn interchange_decode_forward<'py>(
    py: Python<'py>,
    z: PyReadonlyArray2<'py, f64>,
    weights: PyReadonlyArray2<'py, f64>,
    gate: PyReadonlyArray1<'py, f64>,
    bias: Option<PyReadonlyArray1<'py, f64>>,
) -> PyResult<Py<PyArray2<f64>>> {
    let bias_view = bias.as_ref().map(|b| b.as_array());
    let out = core_interchange_decode_forward(CoreInterchangeDecodeForward {
        z: z.as_array(),
        weights: weights.as_array(),
        gate: gate.as_array(),
        bias: bias_view,
    })
    .map_err(py_value_error)?;
    Ok(out.into_pyarray(py).unbind())
}

/// Backward of the per-feature scalar-gate decoder.
///
/// Returns `(grad_z, grad_weights, grad_gate, grad_bias_or_none)`.
#[pyfunction(signature = (z, weights, gate, grad_out, with_bias))]
fn interchange_decode_backward<'py>(
    py: Python<'py>,
    z: PyReadonlyArray2<'py, f64>,
    weights: PyReadonlyArray2<'py, f64>,
    gate: PyReadonlyArray1<'py, f64>,
    grad_out: PyReadonlyArray2<'py, f64>,
    with_bias: bool,
) -> PyResult<(
    Py<PyArray2<f64>>,
    Py<PyArray2<f64>>,
    Py<PyArray1<f64>>,
    Option<Py<PyArray1<f64>>>,
)> {
    let adjoint = core_interchange_decode_backward(
        z.as_array(),
        weights.as_array(),
        gate.as_array(),
        grad_out.as_array(),
        with_bias,
    )
    .map_err(py_value_error)?;
    let grad_bias = adjoint.grad_bias.map(|b| b.into_pyarray(py).unbind());
    Ok((
        adjoint.grad_z.into_pyarray(py).unbind(),
        adjoint.grad_weights.into_pyarray(py).unbind(),
        adjoint.grad_gate.into_pyarray(py).unbind(),
        grad_bias,
    ))
}

/// Forward of the masked-swap interchange decoder.
///
/// `mask` is a 1-D bool array of length F. For atoms with `mask[f] == true`
/// the corresponding column of `z_a` is used; otherwise the column of `z_b`.
/// Reconstruction weights and gate are shared.
#[pyfunction(signature = (z_a, z_b, mask, weights, gate, bias = None))]
fn interchange_swap_forward<'py>(
    py: Python<'py>,
    z_a: PyReadonlyArray2<'py, f64>,
    z_b: PyReadonlyArray2<'py, f64>,
    mask: PyReadonlyArray1<'py, bool>,
    weights: PyReadonlyArray2<'py, f64>,
    gate: PyReadonlyArray1<'py, f64>,
    bias: Option<PyReadonlyArray1<'py, f64>>,
) -> PyResult<Py<PyArray2<f64>>> {
    let bias_view = bias.as_ref().map(|b| b.as_array());
    let out = core_interchange_swap_forward(CoreInterchangeSwapForward {
        z_a: z_a.as_array(),
        z_b: z_b.as_array(),
        mask: mask.as_array(),
        weights: weights.as_array(),
        gate: gate.as_array(),
        bias: bias_view,
    })
    .map_err(py_value_error)?;
    Ok(out.into_pyarray(py).unbind())
}

/// Backward of the masked-swap interchange decoder.
///
/// Returns `(grad_z_a, grad_z_b, grad_weights, grad_gate, grad_bias_or_none)`.
#[pyfunction(signature = (z_a, z_b, mask, weights, gate, grad_out, with_bias))]
fn interchange_swap_backward<'py>(
    py: Python<'py>,
    z_a: PyReadonlyArray2<'py, f64>,
    z_b: PyReadonlyArray2<'py, f64>,
    mask: PyReadonlyArray1<'py, bool>,
    weights: PyReadonlyArray2<'py, f64>,
    gate: PyReadonlyArray1<'py, f64>,
    grad_out: PyReadonlyArray2<'py, f64>,
    with_bias: bool,
) -> PyResult<(
    Py<PyArray2<f64>>,
    Py<PyArray2<f64>>,
    Py<PyArray2<f64>>,
    Py<PyArray1<f64>>,
    Option<Py<PyArray1<f64>>>,
)> {
    let adjoint = core_interchange_swap_backward(
        z_a.as_array(),
        z_b.as_array(),
        mask.as_array(),
        weights.as_array(),
        gate.as_array(),
        grad_out.as_array(),
        with_bias,
    )
    .map_err(py_value_error)?;
    let grad_bias = adjoint.grad_bias.map(|b| b.into_pyarray(py).unbind());
    Ok((
        adjoint.grad_z_a.into_pyarray(py).unbind(),
        adjoint.grad_z_b.into_pyarray(py).unbind(),
        adjoint.grad_weights.into_pyarray(py).unbind(),
        adjoint.grad_gate.into_pyarray(py).unbind(),
        grad_bias,
    ))
}

/// Backward pass: compute `grad_t` and the standard REML adjoint
/// gradients at the current latent `t`.
///
/// The construction mirrors `gaussian_reml_fit_positions_backward`:
/// the inner adjoint produces `grad_x` (= ∂L/∂Φ); we then contract
/// against the N-D radial derivative jet to obtain
/// `grad_t ∈ ℝ^{n_obs × latent_dim}`.
///
/// For `grad_reml_score`, the latent contraction uses the explicit outer
/// REML formula from `/tmp/codex_outer_analytic.md` so the REML Occam
/// correction `J_i^T K_H x_i` is included with one shared solve per row.
///
/// Identifiability-mode contributions to `grad_t`:
///   * `AuxPrior`: the projected pullback of `μ · (t − ĥ(u))`;
///   * `DimSelection`: `+ Λ · t` with diagonal precision per axis.
///
/// These additive terms are computed here from the supplied auxiliary /
/// precision arrays and folded into the returned `grad_t`. The outer
/// REML loop sees a *unique* minimum because the inner Hessian on t is
/// now bounded below by `μI` (auxiliary). Fixes the audit-revised claim:
/// dim-selection/ARD alone is not a rotation-gauge fix and must be paired
/// with AuxPrior or Isometry for identifiability.
#[pyfunction(signature = (
    t,
    y,
    n_obs,
    latent_dim,
    centers,
    penalty,
    grad_lambda = 0.0,
    grad_coefficients = None,
    grad_fitted = None,
    grad_reml_score = 0.0,
    grad_edf = 0.0,
    m = 2,
    weights = None,
    fisher_w = None,
    init_lambda = None,
    aux_u = None,
    aux_family = "ridge".to_string(),
    aux_strength = None,
    dim_selection_log_precision = None,
    basis_kind = "duchon".to_string(),
    sigma_eff_mode = "profiled".to_string(),
    tensor_knots_concat = None,
    tensor_knot_offsets = None,
    tensor_degrees = None,
))]
fn gaussian_reml_fit_latent_backward<'py>(
    py: Python<'py>,
    t: PyReadonlyArray1<'py, f64>,
    y: PyReadonlyArray2<'py, f64>,
    n_obs: usize,
    latent_dim: usize,
    centers: PyReadonlyArray2<'py, f64>,
    penalty: PyReadonlyArray2<'py, f64>,
    grad_lambda: f64,
    grad_coefficients: Option<PyReadonlyArray2<'py, f64>>,
    grad_fitted: Option<PyReadonlyArray2<'py, f64>>,
    grad_reml_score: f64,
    grad_edf: f64,
    m: usize,
    weights: Option<PyReadonlyArray1<'py, f64>>,
    fisher_w: Option<PyReadonlyArray3<'py, f64>>,
    init_lambda: Option<f64>,
    aux_u: Option<PyReadonlyArray2<'py, f64>>,
    aux_family: String,
    aux_strength: Option<f64>,
    dim_selection_log_precision: Option<PyReadonlyArray1<'py, f64>>,
    basis_kind: String,
    sigma_eff_mode: String,
    tensor_knots_concat: Option<PyReadonlyArray1<'py, f64>>,
    tensor_knot_offsets: Option<Vec<usize>>,
    tensor_degrees: Option<Vec<usize>>,
) -> PyResult<Py<PyDict>> {
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
    let basis_kind_normalized = latent_basis_kind(&basis_kind).map_err(py_value_error)?;
    let centers_view = centers.as_array();
    let t_view = t.as_array();
    let y_view = y.as_array();
    let penalty_view = penalty.as_array();
    let effective_weights = latent_scalar_weights_with_fisher(
        n_obs,
        weights.as_ref().map(|w| w.as_array()),
        fisher_w.as_ref().map(|w| w.as_array()),
    )
    .map_err(py_value_error)?;
    let weights_view = effective_weights.as_ref().map(|w| w.view());

    // Forward design (Φ), t-matrix, and input-location jet share one dispatcher.
    let (design, t_mat, jet) = build_latent_forward_design(
        basis_kind_normalized,
        t_view,
        n_obs,
        latent_dim,
        centers_view,
        m,
        tensor_knots_concat.as_ref().map(|a| a.as_array()),
        tensor_knot_offsets.as_deref(),
        tensor_degrees.as_deref(),
        // Standalone Python backward/gradient entrypoint: no manifold/chart
        // concept here (the Rust outer optimizer routes through
        // `LatentOuterProblem`), so the latent design stays the open Euclidean
        // basis — byte-identical to prior behavior.
        None,
    )
    .map_err(py_value_error)?;
    let fit = gaussian_reml_multi_closed_form_with_cache(
        design.view(),
        y_view,
        penalty_view,
        weights_view,
        init_lambda,
        None,
    )
    .map_err(|err| py_value_error(err.to_string()))?;
    // Inner adjoint for the returned standard gradients. This still follows
    // the generic REML VJP path for y/S/w, but grad_t below replaces the
    // score-design component with the row-shared analytic latent formula.
    let backward = gaussian_reml_multi_closed_form_backward_from_fit(
        design.view(),
        y_view,
        penalty_view,
        weights_view,
        &fit,
        grad_lambda,
        grad_coefficients.as_ref().map(|g| g.as_array()),
        grad_fitted.as_ref().map(|g| g.as_array()),
        grad_reml_score,
        grad_edf,
    )
    .map_err(|err| py_value_error(err.to_string()))?;
    let backward_for_t = if grad_reml_score != 0.0 {
        gaussian_reml_multi_closed_form_backward_from_fit(
            design.view(),
            y_view,
            penalty_view,
            weights_view,
            &fit,
            grad_lambda,
            grad_coefficients.as_ref().map(|g| g.as_array()),
            grad_fitted.as_ref().map(|g| g.as_array()),
            0.0,
            grad_edf,
        )
        .map_err(|err| py_value_error(err.to_string()))?
    } else {
        backward.clone()
    };
    let grad_x = &backward_for_t.grad_x;
    let mut grad_t =
        contract_input_loc_gradient(grad_x.view(), &jet).map_err(basis_error_to_pyerr)?;
    if grad_reml_score != 0.0 {
        add_latent_outer_reml_score_gradient(
            &mut grad_t,
            grad_reml_score,
            design.view(),
            y_view,
            t_mat.view(),
            &jet,
            penalty_view,
            weights_view,
            &fit,
            sigma_eff_mode,
        )
        .map_err(py_value_error)?;
    }
    // Identifiability-mode additive contributions to grad_t plus log-normalizer
    // adjoints. Fixes audit-revised claim that REML ARD/AuxPrior selection
    // needs the normalized prior terms, not only raw quadratic gradients.
    let mut grad_aux_log_strength: Option<f64> = None;
    let mut grad_dim_selection_log_precision: Option<Array1<f64>> = None;
    if let Some(u_arr) = aux_u.as_ref() {
        let u_view = u_arr.as_array();
        let stats = latent_aux_prior_stats(t_mat.view(), u_view, family, aux_strength)
            .map_err(py_value_error)?;
        let residual = &t_mat - &stats.targets;
        let projected_residual =
            aux_prior_targets(residual.view(), u_view, family).map_err(py_value_error)?;
        let grad_base = residual - projected_residual;
        for n in 0..n_obs {
            for a in 0..latent_dim {
                if grad_reml_score != 0.0 {
                    grad_t[n * latent_dim + a] +=
                        grad_reml_score * stats.strength.mu * grad_base[[n, a]];
                }
            }
        }
        grad_aux_log_strength = Some(
            grad_reml_score * (0.5 * stats.strength.mu * stats.residual_sq - 0.5 * n_obs as f64),
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
        let mut grad_log_prec = Array1::<f64>::zeros(latent_dim);
        for n in 0..n_obs {
            for a in 0..latent_dim {
                let prec = lp[a].exp();
                if grad_reml_score != 0.0 {
                    grad_t[n * latent_dim + a] += grad_reml_score * prec * t_mat[[n, a]];
                }
            }
        }
        for a in 0..latent_dim {
            let log_alpha = lp[a];
            let prec = log_alpha.exp();
            if !(prec.is_finite() && prec > 0.0) {
                return Err(py_value_error(format!(
                    "dim_selection_log_precision[{a}] must exponentiate to a finite positive precision"
                )));
            }
            let mut sq = 0.0_f64;
            for n in 0..n_obs {
                let v = t_mat[[n, a]];
                sq += v * v;
            }
            grad_log_prec[a] = grad_reml_score * (0.5 * prec * sq - 0.5 * (n_obs as f64));
        }
        grad_dim_selection_log_precision = Some(grad_log_prec);
    }
    let mut grad_t_matrix = Array2::<f64>::zeros((n_obs, latent_dim));
    for n in 0..n_obs {
        for a in 0..latent_dim {
            grad_t_matrix[[n, a]] = grad_t[n * latent_dim + a];
        }
    }
    let out = PyDict::new(py);
    out.set_item("grad_t", grad_t_matrix.into_pyarray(py))?;
    out.set_item("grad_y", backward.grad_y.into_pyarray(py))?;
    out.set_item("grad_penalty", backward.grad_penalty.into_pyarray(py))?;
    out.set_item("grad_weights", backward.grad_weights.into_pyarray(py))?;
    if let Some(grad) = grad_aux_log_strength {
        out.set_item("grad_aux_log_strength", grad)?;
        out.set_item("grad_log_mu", grad)?;
    } else {
        out.set_item("grad_aux_log_strength", py.None())?;
        out.set_item("grad_log_mu", py.None())?;
    }
    if let Some(grad) = grad_dim_selection_log_precision {
        out.set_item("grad_dim_selection_log_precision", grad.into_pyarray(py))?;
    } else {
        out.set_item("grad_dim_selection_log_precision", py.None())?;
    }
    Ok(out.unbind())
}

/// Owned inputs for the latent outer-optimization objective.
///
/// Bundles the data the value/gradient evaluation needs so a single struct can
/// be reused across trust-region iterations and restarts without re-copying
/// from Python.
struct LatentOuterProblem {
    y: Array2<f64>,
    centers: Array2<f64>,
    penalty: Array2<f64>,
    weights: Option<Array1<f64>>,
    aux_u: Option<Array2<f64>>,
    dim_selection: Option<Array1<f64>>,
    family: AuxPriorFamily,
    aux_strength: Option<f64>,
    init_lambda: Option<f64>,
    sigma_eff_mode: SigmaEffMode,
    n_obs: usize,
    latent_dim: usize,
    m: usize,
    basis_kind: String,
    tensor_knots: Option<Array1<f64>>,
    tensor_knot_offsets: Option<Vec<usize>>,
    tensor_degrees: Option<Vec<usize>>,
    /// Per-axis chart period of the optimizer's manifold (radians) for the
    /// Duchon decoder; `None` on Euclidean / sphere so the decoder stays the
    /// open Euclidean basis. Derived from the `manifold` string in
    /// `gaussian_reml_optimize_latent` via `latent_manifold_periodic_descriptor`.
    periodic: Option<Vec<Option<f64>>>,
}

impl LatentOuterProblem {
    /// REML score (inner Gaussian REML plus aux/dim identifiability priors) and,
    /// when `want_grad`, the outer latent gradient `∂(reml_score)/∂t`.
    ///
    /// The value reproduces [`gaussian_reml_fit_latent`]'s `reml_score` and the
    /// gradient reproduces [`gaussian_reml_fit_latent_backward`]'s `grad_t` at
    /// `grad_reml_score = 1`, so the optimizer descends exactly the quantity the
    /// forward primitive reports. A non-finite or unsolvable configuration maps
    /// to `+∞` with no gradient, which the trust region rejects rather than
    /// propagating a NaN into the inner adjoint.
    fn value_and_grad(
        &self,
        t_flat: ArrayView1<'_, f64>,
        want_grad: bool,
    ) -> (f64, Option<Array1<f64>>) {
        match self.try_value_and_grad(t_flat, want_grad) {
            Ok(pair) => pair,
            Err(_) => (f64::INFINITY, None),
        }
    }

    fn try_value_and_grad(
        &self,
        t_flat: ArrayView1<'_, f64>,
        want_grad: bool,
    ) -> Result<(f64, Option<Array1<f64>>), String> {
        let (design, t_mat, jet) = build_latent_forward_design(
            &self.basis_kind,
            t_flat,
            self.n_obs,
            self.latent_dim,
            self.centers.view(),
            self.m,
            self.tensor_knots.as_ref().map(|a| a.view()),
            self.tensor_knot_offsets.as_deref(),
            self.tensor_degrees.as_deref(),
            self.periodic.as_deref(),
        )?;
        let weights_view = self.weights.as_ref().map(|w| w.view());
        let fit = gaussian_reml_multi_closed_form_with_cache(
            design.view(),
            self.y.view(),
            self.penalty.view(),
            weights_view,
            self.init_lambda,
            None,
        )
        .map_err(|err| err.to_string())?;
        let (prior_score, _aux_state) = latent_prior_score_and_aux_state_for_t(
            t_mat.view(),
            self.aux_u.as_ref().map(|a| a.view()),
            self.family,
            self.aux_strength,
            self.dim_selection.as_ref().map(|a| a.view()),
        )?;
        let value = fit.reml_score + prior_score;
        if !value.is_finite() {
            return Ok((f64::INFINITY, None));
        }
        if !want_grad {
            return Ok((value, None));
        }
        let mut grad_t = Array1::<f64>::zeros(self.n_obs * self.latent_dim);
        add_latent_outer_reml_score_gradient(
            &mut grad_t,
            1.0,
            design.view(),
            self.y.view(),
            t_mat.view(),
            &jet,
            self.penalty.view(),
            weights_view,
            &fit,
            self.sigma_eff_mode,
        )?;
        // Identifiability-prior contributions, identical to the backward path's
        // grad_t assembly at `grad_reml_score = 1`.
        if let Some(u_arr) = self.aux_u.as_ref() {
            let u_view = u_arr.view();
            let stats =
                latent_aux_prior_stats(t_mat.view(), u_view, self.family, self.aux_strength)?;
            let residual = &t_mat - &stats.targets;
            let projected_residual = aux_prior_targets(residual.view(), u_view, self.family)?;
            let grad_base = residual - projected_residual;
            for n in 0..self.n_obs {
                for a in 0..self.latent_dim {
                    grad_t[n * self.latent_dim + a] += stats.strength.mu * grad_base[[n, a]];
                }
            }
        }
        if let Some(log_prec) = self.dim_selection.as_ref() {
            for n in 0..self.n_obs {
                for a in 0..self.latent_dim {
                    let prec = log_prec[a].exp();
                    grad_t[n * self.latent_dim + a] += prec * t_mat[[n, a]];
                }
            }
        }
        if !grad_t.iter().all(|value| value.is_finite()) {
            return Ok((f64::INFINITY, None));
        }
        Ok((value, Some(grad_t)))
    }
}

/// Adapter exposing [`LatentOuterProblem`] to the Riemannian trust region.
struct LatentOuterObjective<'a> {
    problem: &'a LatentOuterProblem,
}

impl gam::geometry::RiemannianObjective for LatentOuterObjective<'_> {
    fn value_gradient(
        &mut self,
        point: ArrayView1<'_, f64>,
    ) -> gam::geometry::GeometryResult<(f64, Array1<f64>)> {
        // A degenerate point yields `+∞` and a zero gradient: the trust region
        // reads a zero gradient at the start as "stationary" (it stops at the
        // finite init) and a `+∞` trial value as a rejected step (it shrinks).
        match self.problem.value_and_grad(point, true) {
            (value, Some(grad)) => Ok((value, grad)),
            (_, None) => Ok((f64::INFINITY, Array1::<f64>::zeros(point.len()))),
        }
    }
}

/// Build the manifold the outer optimizer walks `t` on. `manifold` names the
/// per-observation geometry; the full latent lives on the `n_obs`-fold product.
/// Per-axis chart period for the latent decoder, derived from the optimizer's
/// manifold so the Duchon decoder is a genuine function ON that manifold.
///
/// The circle manifold (`src/geometry/circle.rs`) wraps each coordinate to
/// `[-π, π)`, i.e. period `2π = TAU` radians; the torus is its `d`-fold product.
/// The optimizer retracts the latent in radians on these charts, and the
/// periodic eigenmap seed (`latent_periodic_seed_start`) also produces radians,
/// so the decoder kernel distance must be measured modulo `TAU` per circular
/// axis and satisfy `Φ(θ) = Φ(θ + TAU)`. A non-periodic axis is `None`.
///
/// Euclidean / sphere return `None` (no axis is a circle): those latent fits
/// stay byte-identical to the open Euclidean Duchon basis. (`sphere` is `S^{d-1}`
/// embedded in `R^d` with NO periodic chart axis here — the spherical structure
/// is carried by the retraction, not by a per-axis wrap.)
fn latent_manifold_periodic_descriptor(
    manifold: &str,
    latent_dim: usize,
) -> Option<Vec<Option<f64>>> {
    match manifold.to_ascii_lowercase().replace('-', "_").as_str() {
        "circle" | "s1" if latent_dim == 1 => Some(vec![Some(std::f64::consts::TAU)]),
        "torus" => Some(vec![Some(std::f64::consts::TAU); latent_dim]),
        _ => None,
    }
}

fn build_latent_outer_manifold(
    manifold: &str,
    n_obs: usize,
    latent_dim: usize,
) -> Result<Box<dyn gam::geometry::RiemannianManifold>, String> {
    let per_point = match manifold.to_ascii_lowercase().replace('-', "_").as_str() {
        "euclidean" | "rn" => {
            // One flat Euclidean block over the whole latent is equivalent to
            // the product and avoids the per-observation slicing overhead.
            return Ok(Box::new(gam::geometry::EuclideanManifold::new(
                n_obs * latent_dim,
            )));
        }
        "circle" | "s1" => {
            if latent_dim != 1 {
                return Err(format!(
                    "circle latent manifold requires latent_dim == 1; got {latent_dim}"
                ));
            }
            gam::geometry::ManifoldSpec::Circle
        }
        "sphere" => {
            if latent_dim < 2 {
                return Err(format!(
                    "sphere latent manifold requires latent_dim >= 2 (S^{{d-1}} embeds in R^d); got {latent_dim}"
                ));
            }
            gam::geometry::ManifoldSpec::Sphere {
                intrinsic_dim: latent_dim - 1,
            }
        }
        "torus" => gam::geometry::ManifoldSpec::Torus { dim: latent_dim },
        other => {
            return Err(format!(
                "unknown latent manifold {other:?}; expected one of euclidean|circle|sphere|torus"
            ));
        }
    };
    let parts = std::iter::repeat_with(|| per_point.clone())
        .take(n_obs)
        .collect();
    gam::geometry::ManifoldSpec::Product(parts)
        .build()
        .map_err(|err| err.to_string())
}

/// Build the restart-0 start for the latent outer optimizer from a spectral
/// (Laplacian-eigenmaps) embedding of the responses `y`.
///
/// The embedding recovers the intrinsic coordinate up to monotone/rotation
/// gauge; each axis is then affinely mapped from `[0, 1]` onto the span of the
/// decoder `centers` for that axis so the seed lands where the basis `Φ` is
/// well-conditioned. On a *periodic* latent manifold (circle/torus) the natural
/// seed is the circular coordinate recovered from the leading Laplacian modes
/// (see [`latent_periodic_seed_start`]); the sphere has no closed-form spectral
/// seed here, so the caller's `t` is used unchanged.
///
/// A spread seed is essential, not optional: the outer optimizer's REML
/// objective is degenerate at a *collapsed* latent (all rows at the same
/// coordinate give identical decoder rows → a rank-deficient inner solve and no
/// usable descent direction). The default caller start is the all-zero vector,
/// which on a periodic manifold is exactly the collapsed configuration; without
/// a spread seed the circle/torus optimizer can never escape it (issue #876).
fn latent_spectral_seed_start(
    y: ArrayView2<'_, f64>,
    centers: ArrayView2<'_, f64>,
    manifold: &str,
    n_obs: usize,
    latent_dim: usize,
    seed_neighbors: usize,
    caller_t: ArrayView1<'_, f64>,
) -> Result<Array1<f64>, String> {
    let manifold_norm = manifold.to_ascii_lowercase().replace('-', "_");
    if matches!(manifold_norm.as_str(), "circle" | "s1" | "torus") {
        return latent_periodic_seed_start(y, n_obs, latent_dim, seed_neighbors, caller_t);
    }
    if !matches!(manifold_norm.as_str(), "euclidean" | "rn") {
        return Ok(caller_t.to_owned());
    }
    if y.nrows() != n_obs {
        return Err(format!(
            "spectral seed: y has {} rows but n_obs = {n_obs}",
            y.nrows()
        ));
    }
    // Too few rows to expose `latent_dim` non-trivial modes: fall back to the
    // caller's start rather than failing the whole optimize call.
    if n_obs < latent_dim + 2 {
        return Ok(caller_t.to_owned());
    }
    let coords = gam::geometry::laplacian_eigenmap_coords(y, latent_dim, seed_neighbors)?;
    // Per-axis target span from the decoder centers; fall back to [0, 1] when an
    // axis has no corresponding center column or a degenerate span.
    let mut start = Array1::<f64>::zeros(n_obs * latent_dim);
    for a in 0..latent_dim {
        let (lo, hi) = if a < centers.ncols() {
            let col = centers.column(a);
            let lo = col.iter().cloned().fold(f64::INFINITY, f64::min);
            let hi = col.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
            if lo.is_finite() && hi.is_finite() && hi > lo {
                (lo, hi)
            } else {
                (0.0, 1.0)
            }
        } else {
            (0.0, 1.0)
        };
        for n in 0..n_obs {
            start[n * latent_dim + a] = lo + coords[[n, a]] * (hi - lo);
        }
    }
    Ok(start)
}

/// Spectral seed for a *periodic* latent (circle / torus), returning each row's
/// angle in `[-π, π)` per axis.
///
/// On a circle the two leading non-trivial Laplacian-eigenmap modes of the
/// responses are (up to rotation/reflection — exactly the circle's gauge) the
/// `cos θ` / `sin θ` pair of the intrinsic angle, so `θ = atan2(sin-mode,
/// cos-mode)` recovers the circular coordinate directly. A torus of dimension
/// `d` is `d` independent circles; we recover one angle per axis from its own
/// pair of modes, requesting `2·d` modes from the embedding and pairing them in
/// order. The recovered angle is a *seed* — correct up to the periodic gauge the
/// decoder is free in — that the Riemannian outer optimizer then polishes.
///
/// Crucially this seed is *spread* around the circle, breaking the collapsed
/// all-zero start (issue #876). When there are too few rows to expose `2·d`
/// non-trivial modes the embedding cannot run; rather than start collapsed we
/// fall back to a deterministic equispaced angular sweep on each axis, which is
/// still non-degenerate (distinct decoder rows) so the optimizer has a usable
/// gradient.
fn latent_periodic_seed_start(
    y: ArrayView2<'_, f64>,
    n_obs: usize,
    latent_dim: usize,
    seed_neighbors: usize,
    caller_t: ArrayView1<'_, f64>,
) -> Result<Array1<f64>, String> {
    use std::f64::consts::TAU;

    if latent_dim == 0 {
        return Ok(caller_t.to_owned());
    }
    if y.nrows() != n_obs {
        return Err(format!(
            "periodic spectral seed: y has {} rows but n_obs = {n_obs}",
            y.nrows()
        ));
    }
    // A circle/torus axis needs two embedding modes (cos/sin); recover one angle
    // per axis. If the caller already supplied a *spread* warm start (not the
    // collapsed default), keep it — the optimizer can polish a good start, but it
    // can never escape a collapsed one. "Spread" is measured per axis by the
    // angular range the wrapped coordinates cover.
    let caller_spread = caller_t.len() == n_obs * latent_dim
        && (0..latent_dim).any(|a| {
            let mut lo = f64::INFINITY;
            let mut hi = f64::NEG_INFINITY;
            for n in 0..n_obs {
                let v = wrap_to_pi(caller_t[n * latent_dim + a]);
                lo = lo.min(v);
                hi = hi.max(v);
            }
            (hi - lo) > 1.0e-6
        });
    if caller_spread {
        return Ok(caller_t.to_owned());
    }

    let modes = 2 * latent_dim;
    // `laplacian_eigenmap_coords` needs `n >= modes + 2` rows to expose `modes`
    // non-trivial eigenvectors. With fewer rows, sweep angles deterministically.
    if n_obs < modes + 2 {
        let mut start = Array1::<f64>::zeros(n_obs * latent_dim);
        for a in 0..latent_dim {
            for n in 0..n_obs {
                let frac = if n_obs > 0 {
                    n as f64 / n_obs as f64
                } else {
                    0.0
                };
                start[n * latent_dim + a] = wrap_to_pi(frac * TAU);
            }
        }
        return Ok(start);
    }

    // The raw (un-rescaled) generalized eigenvectors are what carry the cos/sin
    // structure; `laplacian_eigenmap_coords` already rescales each axis to
    // [0, 1], which destroys the relative sign/scale needed for atan2. We
    // instead read `2·d` modes and undo the per-axis affine map by recentering
    // each mode to zero mean before pairing — the rescale is affine per mode, so
    // recentering recovers the angle up to the same rotation gauge.
    let coords = gam::geometry::laplacian_eigenmap_coords(y, modes, seed_neighbors)?;
    let mut mode_mean = vec![0.0f64; modes];
    for a in 0..modes {
        let mut sum = 0.0;
        for n in 0..n_obs {
            sum += coords[[n, a]];
        }
        mode_mean[a] = sum / n_obs as f64;
    }

    let mut start = Array1::<f64>::zeros(n_obs * latent_dim);
    for axis in 0..latent_dim {
        let cos_mode = 2 * axis;
        let sin_mode = 2 * axis + 1;
        for n in 0..n_obs {
            let c = coords[[n, cos_mode]] - mode_mean[cos_mode];
            let s = coords[[n, sin_mode]] - mode_mean[sin_mode];
            let angle = if c == 0.0 && s == 0.0 {
                // Degenerate row (both modes vanish): place it deterministically
                // around the circle so it does not coincide with its neighbours.
                wrap_to_pi((n as f64 / n_obs as f64) * TAU)
            } else {
                s.atan2(c)
            };
            start[n * latent_dim + axis] = angle;
        }
        // Guard against a collapsed axis (both modes constant → all angles
        // equal): fall back to an equispaced sweep on that axis only.
        let mut lo = f64::INFINITY;
        let mut hi = f64::NEG_INFINITY;
        for n in 0..n_obs {
            let v = start[n * latent_dim + axis];
            lo = lo.min(v);
            hi = hi.max(v);
        }
        if !(hi - lo > 1.0e-6) {
            for n in 0..n_obs {
                let frac = if n_obs > 0 {
                    n as f64 / n_obs as f64
                } else {
                    0.0
                };
                start[n * latent_dim + axis] = wrap_to_pi(frac * TAU);
            }
        }
    }
    Ok(start)
}

/// Wrap an angle to the half-open interval `[-π, π)`.
fn wrap_to_pi(angle: f64) -> f64 {
    use std::f64::consts::{PI, TAU};
    let mut a = angle % TAU;
    if a >= PI {
        a -= TAU;
    } else if a < -PI {
        a += TAU;
    }
    a
}

/// Optimize the latent coordinate `t` against the Gaussian-REML objective.
///
/// Unlike [`gaussian_reml_fit_latent`], which performs a single `β | t` inner
/// solve at a fixed `t`, this routine runs the *outer* latent optimization: it
/// minimizes the REML score over `t` with a Riemannian trust region driven by
/// the analytic `∂(reml_score)/∂t` (the same gradient
/// [`gaussian_reml_fit_latent_backward`] returns), retracting each accepted
/// step onto `manifold`. It returns the full REML fit dictionary *at the
/// converged latent* plus the optimized `t`/`latent` arrays.
///
/// The latent REML objective is non-convex (a GP-LVM-style coordinate problem),
/// so a single cold random start may settle in a poor local optimum. By default
/// (`init="spectral"`) restart 0 starts from a Laplacian-eigenmaps embedding of
/// the responses, which recovers the intrinsic coordinate up to gauge and lets
/// the optimizer polish it to the global fit instead of sorting rows from
/// scratch; the passed-in `t` is then only a fallback (too few rows, or a
/// non-Euclidean `manifold`). Pass `init="caller"` to start from `t` unchanged
/// (a pure local solve / explicit warm start), and `n_restarts > 1` to also
/// optimize from perturbed starts and keep the lowest-score result.
///
/// Shift-invariant relative-gradient stationarity measure for the latent outer
/// solve: `‖∇ₜ f(t̂)‖_g / max(‖∇ₜ f(t₀)‖_g, 1)`, comparing the projected
/// Riemannian gradient norm at the chosen latent to the gradient norm at the
/// INITIAL iterate `t₀`. This is the FFI analogue of `relative_stationarity` in
/// `src/geometry/optimizer.rs` (issue #954), kept byte-for-byte identical to it
/// so the diagnostic `converged` flag agrees with the optimizer's own stopping
/// rule:
///
/// * **Shift-invariant** — the objective value `f` does not enter at all, so an
///   additive shift `f → f + C` (which leaves the minimizer, gradient, Hessian,
///   and model reduction unchanged) cannot move the measure. The earlier
///   `‖∇ₜ f‖·‖t‖_typ / max(|f|, 1)` divided by the objective magnitude, so a
///   large `C` inflated the denominator and could falsely certify a
///   non-stationary latent as converged (#954).
/// * **Scale-invariant** — under `f → c·f` both `‖∇ₜ f(t̂)‖` and `‖∇ₜ f(t₀)‖`
///   scale by `c`, so the ratio is unchanged and a fixed `grad_tol` reads as a
///   true *relative* tolerance.
/// * **#879 O(n) calibration** — the profiled REML objective leaves `‖∇ₜ f‖` at
///   an O(n) magnitude even at a genuine stationary point near interpolation;
///   anchoring to `‖∇ₜ f(t₀)‖` (itself O(n)) divides that magnitude out, while
///   the `max(·, 1)` floor reduces the test to the bare absolute
///   `‖∇ₜ f‖ ≤ grad_tol` on a unit-scale objective.
/// * **Non-finite** — a blown-up iterate (`‖∇ₜ f‖` or `‖∇ₜ f(t₀)‖` not finite)
///   maps to `+∞`, so it is never reported stationary.
fn latent_relative_stationarity(grad_norm: f64, grad0_norm: f64) -> f64 {
    if !grad_norm.is_finite() || !grad0_norm.is_finite() {
        return f64::INFINITY;
    }
    grad_norm / grad0_norm.max(1.0)
}

#[cfg(test)]
mod sae_euclidean_oos_rebuild_tests {
    use super::{monomial_exponents, sae_euclidean_degree_for_basis_size};

    /// #1132 bug 3: the OOS basis rebuild for a Euclidean (linear) atom must
    /// re-emit a basis whose width `M` equals the TRAINED decoder block's row
    /// count. The width is `monomial_exponents(dim, degree).len()` where `dim`
    /// is the build dimension (`centers.ncols()`). Recovering the degree from
    /// `(dim, trained_M)` and rebuilding against the same `dim` must therefore
    /// reproduce `trained_M` exactly. The regression case is a 1-D linear atom
    /// whose trained decoder has `M = 1` (degree 0): the recovery must yield
    /// degree 0 and width 1, NOT re-expand to width 3 (degree 2) — the
    /// "decoder_blocks[0] has M=1 but rebuilt basis has M=3" OOS failure.
    fn rebuilt_m_for(dim: usize, trained_m: usize) -> usize {
        let degree = sae_euclidean_degree_for_basis_size(dim, trained_m)
            .expect("degree must be recoverable from the trained decoder width");
        monomial_exponents(dim, degree).len()
    }

    #[test]
    fn euclidean_oos_rebuild_m_matches_trained_decoder_m() {
        // The exact #1132 regression: dim = 1, trained M = 1 (constant-only).
        assert_eq!(
            rebuilt_m_for(1, 1),
            1,
            "1-D linear atom with decoder M=1 must rebuild to M=1, not M=3"
        );
        // The recovered width must equal the trained M across the supported
        // degrees and dimensions (self-consistency of the decoder-anchored
        // recovery the OOS / steer paths now use).
        for dim in 1..=2usize {
            for degree in 0..=2usize {
                let trained_m = monomial_exponents(dim, degree).len();
                assert_eq!(
                    rebuilt_m_for(dim, trained_m),
                    trained_m,
                    "dim={dim}, degree={degree}: rebuilt M must equal trained M"
                );
            }
        }
    }
}
