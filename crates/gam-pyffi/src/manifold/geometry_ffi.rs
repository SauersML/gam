#[pyfunction]
fn poincare_distance<'py>(
    py: Python<'py>,
    a: PyReadonlyArray1<'py, f64>,
    b: PyReadonlyArray1<'py, f64>,
    curvature: f64,
) -> PyResult<f64> {
    let a_owned = a.as_array().to_owned();
    let b_owned = b.as_array().to_owned();
    detach_geometry_result(py, "poincare_distance", move || {
        poincare_distance_impl(a_owned.view(), b_owned.view(), curvature)
    })
}

/// K-aware default IBP concentration `α` (#1784) from the Rust source of truth
/// `assignment::default_ibp_concentration_for_k_atoms`. Exposed so the Python
/// facade calls the core formula (`α = max(1, 1/(exp(1/K) − 1))`) instead of
/// mirroring it — the thin-wrapper SPEC: no numeric policy lives in Python.
#[pyfunction]
fn sae_default_ibp_concentration_for_k_atoms(k_atoms: usize) -> f64 {
    gam::terms::sae::assignment::default_ibp_concentration_for_k_atoms(k_atoms)
}

/// Default large-K active cap from the data-per-atom ratio, from the Rust source
/// of truth `assignment::default_top_k_for_large_dictionary`. Returns Python
/// `None` when the dense softmax path is admitted (`N/K ≥ K`, or `K ≤ 1`), and
/// otherwise the per-row cap `clamp(ceil(N/K), 1, K−1)`.
#[pyfunction]
fn sae_default_top_k_for_large_dictionary(n_obs: usize, k_atoms: usize) -> Option<usize> {
    gam::terms::sae::assignment::default_top_k_for_large_dictionary(n_obs, k_atoms)
}

#[pyfunction]
#[pyo3(signature = (n_obs, output_dim, n_atoms, d_max=1, topk_support=None))]
fn sae_fit_admission<'py>(
    py: Python<'py>,
    n_obs: usize,
    output_dim: usize,
    n_atoms: usize,
    d_max: usize,
    topk_support: Option<usize>,
) -> PyResult<Py<PyDict>> {
    // Mode-aware: a hard top-k support request is admitted by the CONCRETE
    // in-core memory budget (the TRUE manifold engine at any K > P within
    // budget; a typed refusal over budget — never a silent linear
    // substitution). Penalty-gated modes keep the architectural K ≤ P rule.
    let admission = match topk_support {
        Some(support) => gam::terms::sae::front_door::admit_topk_manifold(
            n_obs, output_dim, n_atoms, d_max, support,
        )
        .map_err(py_value_error)?,
        None => gam::terms::sae::front_door::admit_sae_fit(n_obs, output_dim, n_atoms)
            .map_err(py_value_error)?,
    };
    let lane = match admission.lane {
        gam::terms::sae::front_door::SaeFitLane::DenseCertification => "dense_certification",
        gam::terms::sae::front_door::SaeFitLane::SparseCodes => "sparse_codes",
        gam::terms::sae::front_door::SaeFitLane::CurvedStreaming => "curved_streaming",
    };
    let out = PyDict::new(py);
    out.set_item("lane", lane)?;
    out.set_item("n_obs", admission.n_obs)?;
    out.set_item("output_dim", admission.output_dim)?;
    out.set_item("n_atoms", admission.n_atoms)?;
    out.set_item("dense_assignment_cells", admission.dense_assignment_cells)?;
    out.set_item("response_cells", admission.response_cells)?;
    Ok(out.unbind())
}

#[pyfunction(signature = (points, mode = "kneedle", knee_slope_fraction = 0.10, complexity_penalty = 0.05, flat_span_tol = 1.0e-6))]
fn sae_select_k(
    py: Python<'_>,
    points: Vec<(usize, f64)>,
    mode: &str,
    knee_slope_fraction: f64,
    complexity_penalty: f64,
    flat_span_tol: f64,
) -> PyResult<PyObject> {
    let curve = gam::terms::sae::k_selection::curve_from_pairs(&points).map_err(py_value_error)?;
    let config = gam::terms::sae::k_selection::KSelectionConfig {
        mode: gam::terms::sae::k_selection::KSelectionMode::parse(mode).map_err(py_value_error)?,
        knee_slope_fraction,
        complexity_penalty,
        flat_span_tol,
        // This scalar-curve FFI carries no fit-measured coding ingredients; a
        // `MeasuredMdl` mode string falls back to Kneedle when this is `None`.
        measured_coding: None,
    };
    let selected = gam::terms::sae::k_selection::select_k(&curve, &config);
    let out = PyDict::new(py);
    out.set_item("k", selected.k)?;
    out.set_item("ev", selected.ev)?;
    out.set_item("flag", selected.flag.as_str())?;
    out.set_item("is_knee", selected.flag.is_knee())?;
    out.set_item("score", selected.score)?;
    Ok(out.into())
}

#[pyfunction(signature = (manifold_points, linear_points, manifold_params_per_atom, linear_params_per_atom, mode = "kneedle", knee_slope_fraction = 0.10, complexity_penalty = 0.05, flat_span_tol = 1.0e-6))]
fn sae_auto_k_recommendation(
    py: Python<'_>,
    manifold_points: Vec<(usize, f64)>,
    linear_points: Vec<(usize, f64)>,
    manifold_params_per_atom: f64,
    linear_params_per_atom: f64,
    mode: &str,
    knee_slope_fraction: f64,
    complexity_penalty: f64,
    flat_span_tol: f64,
) -> PyResult<PyObject> {
    let manifold =
        gam::terms::sae::k_selection::curve_from_pairs(&manifold_points).map_err(py_value_error)?;
    let linear =
        gam::terms::sae::k_selection::curve_from_pairs(&linear_points).map_err(py_value_error)?;
    let config = gam::terms::sae::k_selection::KSelectionConfig {
        mode: gam::terms::sae::k_selection::KSelectionMode::parse(mode).map_err(py_value_error)?,
        knee_slope_fraction,
        complexity_penalty,
        flat_span_tol,
        // This scalar-curve FFI carries no fit-measured coding ingredients; a
        // `MeasuredMdl` mode string falls back to Kneedle when this is `None`.
        measured_coding: None,
    };
    // The manifold-vs-linear advantage is now measured in DECODER PARAMETERS, not
    // atom count: a manifold atom stores `basis_size·p` scalars, a linear atom
    // `p`, so `efficiency_ratio` is the parameter ratio `linear_params /
    // manifold_params` and only exceeds 1 when the manifold reaches the target EV
    // with fewer stored scalars.
    let rec = gam::terms::sae::k_selection::recommend_auto_k(
        &manifold,
        &linear,
        &config,
        manifold_params_per_atom,
        linear_params_per_atom,
    );
    let out = PyDict::new(py);
    out.set_item("k", rec.selection.k)?;
    out.set_item("ev", rec.selection.ev)?;
    out.set_item("flag", rec.selection.flag.as_str())?;
    out.set_item("is_knee", rec.selection.flag.is_knee())?;
    out.set_item("score", rec.selection.score)?;
    out.set_item("target_ev", rec.advantage.target_ev)?;
    out.set_item("manifold_k", rec.advantage.k_manifold)?;
    out.set_item("linear_k", rec.advantage.k_linear)?;
    out.set_item("manifold_params", rec.advantage.manifold_params)?;
    out.set_item("linear_params", rec.advantage.linear_params)?;
    out.set_item("efficiency_ratio", rec.advantage.compression_ratio)?;
    out.set_item("confirmed", rec.advantage.manifold_dominates())?;
    Ok(out.into())
}

/// Per-atom coordinate signal-variance spectrum: the eigenvalues of the coded
/// chart coordinates' population covariance. This is the Gaussian-source variance
/// spectrum the reverse-water-filling code rate is defined over, mirroring the
/// block-chart MDL scorer's `coordinate_spectrum` (same mean-centred covariance,
/// same `jacobi_eigh`, zeros clamped, descending). One eigenvalue per coded
/// coordinate axis.
fn coordinate_variance_spectrum(coords: ndarray::ArrayView2<'_, f64>) -> Vec<f64> {
    let n = coords.nrows();
    let d = coords.ncols();
    if d == 0 || n == 0 {
        return vec![0.0; d];
    }
    let mut means = vec![0.0f64; d];
    for j in 0..d {
        for i in 0..n {
            means[j] += coords[[i, j]];
        }
        means[j] /= n as f64;
    }
    let mut cov = vec![0.0f64; d * d];
    for i in 0..n {
        for a in 0..d {
            let va = coords[[i, a]] - means[a];
            for b in 0..d {
                cov[a * d + b] += va * (coords[[i, b]] - means[b]);
            }
        }
    }
    for v in &mut cov {
        *v /= n as f64;
    }
    let mut vals = vec![0.0f64; d];
    let mut vecs = vec![0.0f64; d * d];
    gam::terms::sae::gpu_kernels::sae_encode_resident::jacobi_eigh(&cov, d, &mut vals, &mut vecs);
    let mut spectrum: Vec<f64> = vals.into_iter().map(|v| v.max(0.0)).collect();
    spectrum.sort_by(|a, b| b.partial_cmp(a).unwrap_or(std::cmp::Ordering::Equal));
    spectrum
}

/// Fit-level bits/token description length of a manifold-SAE reconstruction.
///
/// The headline currency the results.md postmortem argues for: the whole fit's
/// code length per token, decomposed into code / selection / dictionary bits.
/// The valid accounting reads the fit's own empirical byproducts rather than
/// pre-summarised scalars: `assignments` `(N, K)` is binarised into the empirical
/// support matrix (an atom is coded for a token when its gate magnitude clears
/// `active_threshold`), so SELECTION is priced by the support-entropy universal
/// code and a maximally-spread row pays its true high support cost; `coords` is
/// the per-atom `(N, d_k)` chart coordinates, whose per-atom covariance
/// eigen-spectra are the per-coordinate variances the latent CODE rate reverse-
/// water-fills at the achieved distortion `(1 − ev)·Σ var`; the code term is
/// firing-weighted by each atom's coded dim `d_k`. Every math term is owned by
/// `description_length::manifold_fit_description_length`; this marshals the
/// support/spectrum in and the decomposition out.
#[pyfunction(signature = (assignments, coords, ev, n_params, l_param_bits = None, active_threshold = 1.0e-8))]
fn sae_manifold_description_length<'py>(
    py: Python<'py>,
    assignments: PyReadonlyArray2<'py, f64>,
    coords: Vec<PyReadonlyArray2<'py, f64>>,
    ev: f64,
    n_params: i64,
    l_param_bits: Option<f64>,
    active_threshold: f64,
) -> PyResult<PyObject> {
    let assignments = assignments.as_array();
    let (n_obs, k_atoms) = assignments.dim();
    if coords.len() != k_atoms {
        return Err(py_value_error(format!(
            "sae_manifold_description_length: expected {k_atoms} coordinate blocks \
             (one per atom), got {}",
            coords.len()
        )));
    }
    // Per-coordinate signal-variance spectrum + coded dim per atom, from the
    // fitted chart coordinates.
    let mut coord_variances: Vec<f64> = Vec::new();
    let mut atom_coord_dims: Vec<f64> = Vec::with_capacity(k_atoms);
    for (k, block) in coords.iter().enumerate() {
        let c = block.as_array();
        if c.nrows() != n_obs {
            return Err(py_value_error(format!(
                "sae_manifold_description_length: coords[{k}] has {} rows but assignments \
                 have {n_obs}",
                c.nrows()
            )));
        }
        atom_coord_dims.push(c.ncols() as f64);
        coord_variances.extend(coordinate_variance_spectrum(c));
    }
    // Achieved coordinate coding distortion: the fit's residual fraction of the
    // total coordinate signal variance `(1 − ev)·Σ var`, matching the
    // per-featurizer `Featurizer::residual`. `1 − ev` is floored away from zero so
    // a saturated fit reports a large finite rate, not +∞.
    let total_var: f64 = coord_variances.iter().sum();
    let delta2 = (1.0 - ev).max(1.0e-12) * total_var;
    // Empirical binary support matrix: an atom is coded for a token when its gate
    // magnitude clears the numerical-dust floor. Reads the TRUE recorded support
    // per token (sparse gate families stay sparse; a maximally-spread softmax row
    // is priced at its true high support cost) instead of a rounded-mean count.
    let mut codes = gam::terms::sae::atom_codes::SparseAtomCodes::empty(n_obs, k_atoms);
    for n in 0..n_obs {
        for k in 0..k_atoms {
            let gate = assignments[[n, k]];
            if gate.is_finite() && gate.abs() > active_threshold {
                codes.row_mut(n).assign(k, gate);
            }
        }
    }
    let dl = gam::terms::sae::description_length::manifold_fit_description_length(
        &codes,
        &coord_variances,
        delta2,
        &atom_coord_dims,
        ev,
        n_params,
        l_param_bits,
    );
    let out = PyDict::new(py);
    out.set_item("bits_per_token", dl.bits_per_token)?;
    out.set_item("total_bits", dl.total_bits)?;
    out.set_item("code_bits", dl.code_bits)?;
    out.set_item("selection_bits", dl.selection_bits)?;
    out.set_item("dict_bits", dl.dict_bits)?;
    out.set_item("code_bits_per_token", dl.code_bits_per_token)?;
    out.set_item("selection_bits_per_token", dl.selection_bits_per_token)?;
    out.set_item("dict_bits_per_token", dl.dict_bits_per_token)?;
    out.set_item("coordinate_rate_bits", dl.coordinate_rate_bits)?;
    out.set_item("l_param_bits", dl.l_param_bits)?;
    out.set_item("ev", dl.ev)?;
    out.set_item("n_tokens", dl.n_tokens)?;
    out.set_item("k_active", dl.k_active)?;
    out.set_item("coord_dim", dl.coord_dim)?;
    out.set_item("g_dict", dl.g_dict)?;
    out.set_item("n_params", dl.n_params)?;
    Ok(out.into())
}

/// Format a float the way Python's `f"{x:g}"` does (default precision 6), so the
/// per-target dict keys (`bits_at_r2_0.9`, …) are byte-identical to the NumPy
/// scorer's. Fixed notation for `|x| ∈ [1e-4, 1e6)`, exponential otherwise, with
/// trailing zeros (and a bare trailing point) stripped.
fn format_g(x: f64) -> String {
    if x == 0.0 {
        return "0".to_string();
    }
    let strip = |s: String| -> String {
        if s.contains('.') {
            let trimmed = s.trim_end_matches('0');
            trimmed.trim_end_matches('.').to_string()
        } else {
            s
        }
    };
    let exp = x.abs().log10().floor() as i32;
    if !(-4..6).contains(&exp) {
        // Exponential form with 6 significant digits (5 after the point).
        let s = format!("{x:.5e}");
        // Rust emits `e-5`/`e0`; Python `%g` emits `e-05`/`e+00`. Normalize.
        if let Some((mantissa, exp_part)) = s.split_once('e') {
            let mantissa = strip(mantissa.to_string());
            let (sign, digits) = match exp_part.strip_prefix('-') {
                Some(rest) => ('-', rest),
                None => ('+', exp_part.trim_start_matches('+')),
            };
            format!("{mantissa}e{sign}{:0>2}", digits)
        } else {
            strip(s)
        }
    } else {
        let precision = (5 - exp).max(0) as usize;
        strip(format!("{x:.precision$}"))
    }
}

/// Eq. 4 fixed-distortion description length of one fitted featurizer.
///
/// The single Rust home for the `gamfit._description_length` scorer: it prices a
/// featurizer's reconstruction of `test_x` at each R² target into support / code
/// / residual / dictionary bits. Every numeric term (the combinatorial `lgamma`
/// support cost, the residual covariance eigendecomposition, each atom's SVD
/// coordinate spectrum, and the joint firing-weighted reverse-water-filling)
/// lives in `eq4_description_length`; this only marshals the arrays and drives
/// the Python `atom_contribution` callback that materialises each atom's firing
/// rows. Returns the same dict shape the NumPy scorer returned (`support_bits`,
/// `achieved_block_l0`, `bits_at_r2_{g}` / `code_bits_at_r2_{g}` /
/// `resid_bits_at_r2_{g}` per target, and `native_bits_per_token` when given).
#[pyfunction]
#[pyo3(signature = (
    test_x, recon, gate, code_dims, dictionary_params, atom_contribution,
    r2_targets = None, native_bits_per_token = None,
))]
#[allow(clippy::too_many_arguments)]
fn sae_eq4_description_length<'py>(
    py: Python<'py>,
    test_x: PyReadonlyArray2<'py, f64>,
    recon: PyReadonlyArray2<'py, f64>,
    gate: PyReadonlyArray2<'py, f64>,
    code_dims: PyReadonlyArray1<'py, i64>,
    dictionary_params: i64,
    atom_contribution: Bound<'py, PyAny>,
    r2_targets: Option<Vec<f64>>,
    native_bits_per_token: Option<f64>,
) -> PyResult<Py<PyDict>> {
    let test_x = test_x.as_array();
    let recon = recon.as_array();
    let gate = gate.as_array();
    let code_dims = code_dims.as_array();
    let code_dims: Vec<i64> = code_dims.iter().copied().collect();
    let targets = r2_targets.unwrap_or_else(|| vec![0.99, 0.95, 0.90, 0.80]);

    // Captures the real Python exception raised inside the callback so it
    // propagates with its original type instead of being flattened to a
    // ValueError; the closure returns the message the core threads back.
    let callback_err: std::cell::RefCell<Option<PyErr>> = std::cell::RefCell::new(None);
    let fetch = |atom: usize, take: &[usize]| -> Result<Array2<f64>, String> {
        let take_arr = take
            .iter()
            .map(|&i| i as i64)
            .collect::<Vec<i64>>()
            .into_pyarray(py);
        let result = atom_contribution.call1((atom, take_arr)).map_err(|e| {
            let message = e.to_string();
            *callback_err.borrow_mut() = Some(e);
            message
        })?;
        let array = result.extract::<PyReadonlyArray2<f64>>().map_err(|e| {
            let message = format!("atom {atom} contribution must be a float64 matrix: {e}");
            *callback_err.borrow_mut() = Some(e);
            message
        })?;
        Ok(array.as_array().to_owned())
    };

    let dl = gam::terms::sae::eq4_description_length::eq4_fixed_distortion_description_length(
        test_x,
        recon,
        gate,
        &code_dims,
        dictionary_params,
        &targets,
        native_bits_per_token,
        fetch,
    )
    .map_err(|message| callback_err.borrow_mut().take().unwrap_or_else(|| py_value_error(message)))?;

    let out = PyDict::new(py);
    out.set_item("support_bits", dl.support_bits)?;
    out.set_item("achieved_block_l0", dl.achieved_block_l0)?;
    for row in &dl.per_target {
        let suffix = format_g(row.target);
        out.set_item(format!("bits_at_r2_{suffix}"), row.bits)?;
        out.set_item(format!("code_bits_at_r2_{suffix}"), row.code_bits)?;
        out.set_item(format!("resid_bits_at_r2_{suffix}"), row.resid_bits)?;
    }
    if let Some(native) = dl.native_bits_per_token {
        out.set_item("native_bits_per_token", native)?;
    }
    Ok(out.into())
}

/// Joint firing-weighted reverse-water-filling of Gaussian spectra to a fixed
/// total distortion — the standalone core the Eq. 4 code/residual split uses.
///
/// `components` is a sequence of `(weight, spectrum)` pairs; returns one rate
/// (bits) per component. The whole allocation is owned by
/// `eq4_description_length::water_fill_component_bits`.
#[pyfunction]
fn sae_eq4_water_fill_component_bits(
    components: Vec<(f64, Vec<f64>)>,
    total_distortion: f64,
) -> PyResult<Vec<f64>> {
    gam::terms::sae::eq4_description_length::water_fill_component_bits(
        &components,
        total_distortion,
    )
    .map_err(py_value_error)
}

#[pyfunction]
fn poincare_project_into_ball<'py>(
    py: Python<'py>,
    point: PyReadonlyArray1<'py, f64>,
    curvature: f64,
) -> PyResult<Py<PyArray1<f64>>> {
    let owned = point.as_array().to_owned();
    let out = detach_geometry_result(py, "poincare_project_into_ball", move || {
        poincare_project_into_ball_impl(owned.view(), curvature)
    })?;
    Ok(out.into_pyarray(py).unbind())
}

#[pyfunction]
fn poincare_log_origin<'py>(
    py: Python<'py>,
    y: PyReadonlyArray1<'py, f64>,
    curvature: f64,
) -> PyResult<Py<PyArray1<f64>>> {
    let owned = y.as_array().to_owned();
    let out = detach_geometry_result(py, "poincare_log_origin", move || {
        poincare_log_origin_impl(owned.view(), curvature)
    })?;
    Ok(out.into_pyarray(py).unbind())
}

#[pyfunction]
fn poincare_exp_origin<'py>(
    py: Python<'py>,
    v: PyReadonlyArray1<'py, f64>,
    curvature: f64,
) -> PyResult<Py<PyArray1<f64>>> {
    let owned = v.as_array().to_owned();
    let out = detach_geometry_result(py, "poincare_exp_origin", move || {
        poincare_exp_origin_impl(owned.view(), curvature)
    })?;
    Ok(out.into_pyarray(py).unbind())
}

#[pyfunction]
fn poincare_conformal_factor<'py>(
    py: Python<'py>,
    p: PyReadonlyArray1<'py, f64>,
    curvature: f64,
) -> PyResult<f64> {
    let p_owned = p.as_array().to_owned();
    detach_geometry_result(py, "poincare_conformal_factor", move || {
        poincare_conformal_factor_impl(p_owned.view(), curvature)
    })
}

#[pyfunction]
fn poincare_exp_map<'py>(
    py: Python<'py>,
    p: PyReadonlyArray1<'py, f64>,
    v: PyReadonlyArray1<'py, f64>,
    curvature: f64,
) -> PyResult<Py<PyArray1<f64>>> {
    let p_owned = p.as_array().to_owned();
    let v_owned = v.as_array().to_owned();
    let out = detach_geometry_result(py, "poincare_exp_map", move || {
        poincare_exp_map_impl(p_owned.view(), v_owned.view(), curvature)
    })?;
    Ok(out.into_pyarray(py).unbind())
}

#[pyfunction]
fn poincare_log_map<'py>(
    py: Python<'py>,
    p: PyReadonlyArray1<'py, f64>,
    q: PyReadonlyArray1<'py, f64>,
    curvature: f64,
) -> PyResult<Py<PyArray1<f64>>> {
    let p_owned = p.as_array().to_owned();
    let q_owned = q.as_array().to_owned();
    let out = detach_geometry_result(py, "poincare_log_map", move || {
        poincare_log_map_impl(p_owned.view(), q_owned.view(), curvature)
    })?;
    Ok(out.into_pyarray(py).unbind())
}

#[pyfunction]
fn poincare_to_lorentz<'py>(
    py: Python<'py>,
    y: PyReadonlyArray1<'py, f64>,
    curvature: f64,
) -> PyResult<Py<PyArray1<f64>>> {
    let owned = y.as_array().to_owned();
    let out = detach_geometry_result(py, "poincare_to_lorentz", move || {
        poincare_to_lorentz_impl(owned.view(), curvature)
    })?;
    Ok(out.into_pyarray(py).unbind())
}

#[pyfunction]
fn poincare_from_lorentz<'py>(
    py: Python<'py>,
    x: PyReadonlyArray1<'py, f64>,
    curvature: f64,
) -> PyResult<Py<PyArray1<f64>>> {
    let owned = x.as_array().to_owned();
    let out = detach_geometry_result(py, "poincare_from_lorentz", move || {
        poincare_from_lorentz_impl(owned.view(), curvature)
    })?;
    Ok(out.into_pyarray(py).unbind())
}

#[pyfunction]
fn poincare_lorentz_log_origin<'py>(
    py: Python<'py>,
    x: PyReadonlyArray1<'py, f64>,
    curvature: f64,
) -> PyResult<Py<PyArray1<f64>>> {
    let owned = x.as_array().to_owned();
    let out = detach_geometry_result(py, "poincare_lorentz_log_origin", move || {
        poincare_lorentz_log_origin_impl(owned.view(), curvature)
    })?;
    Ok(out.into_pyarray(py).unbind())
}

#[pyfunction]
fn poincare_lorentz_exp_origin<'py>(
    py: Python<'py>,
    v_spatial: PyReadonlyArray1<'py, f64>,
    curvature: f64,
) -> PyResult<Py<PyArray1<f64>>> {
    let owned = v_spatial.as_array().to_owned();
    let out = detach_geometry_result(py, "poincare_lorentz_exp_origin", move || {
        poincare_lorentz_exp_origin_impl(owned.view(), curvature)
    })?;
    Ok(out.into_pyarray(py).unbind())
}

/// Forward pass for the Poincaré tangent-space-at-origin decoder.
///
/// Returns `(x_hat, atoms_projected, v, tangents, proj_scale)` so the Python
/// autograd.Function can stash them and use them on the backward pass without
/// recomputing. `proj_scale` (shape `(F,)`) is the per-atom radial
/// ball-projection factor needed by the backward to chain gradients through
/// the projection back to the raw atom storage.
#[pyfunction]
fn poincare_tangent_decode_forward<'py>(
    py: Python<'py>,
    atoms: PyReadonlyArray2<'py, f64>,
    gates: PyReadonlyArray2<'py, f64>,
    curvature: f64,
) -> PyResult<(
    Py<PyArray2<f64>>,
    Py<PyArray2<f64>>,
    Py<PyArray2<f64>>,
    Py<PyArray2<f64>>,
    Py<PyArray1<f64>>,
)> {
    let atoms_owned = atoms.as_array().to_owned();
    let gates_owned = gates.as_array().to_owned();
    let (x_hat, cache) =
        detach_geometry_result(py, "poincare_tangent_decode_forward", move || {
            poincare_tangent_decode_forward_impl(atoms_owned.view(), gates_owned.view(), curvature)
        })?;
    Ok((
        x_hat.into_pyarray(py).unbind(),
        cache.atoms_projected.into_pyarray(py).unbind(),
        cache.v.into_pyarray(py).unbind(),
        cache.tangents.into_pyarray(py).unbind(),
        cache.proj_scale.into_pyarray(py).unbind(),
    ))
}

/// Backward pass that consumes the cached state returned by the forward.
///
/// Returns `(grad_gates, grad_atoms)`.
#[pyfunction]
fn poincare_tangent_decode_backward<'py>(
    py: Python<'py>,
    atoms_projected: PyReadonlyArray2<'py, f64>,
    gates: PyReadonlyArray2<'py, f64>,
    v: PyReadonlyArray2<'py, f64>,
    tangents: PyReadonlyArray2<'py, f64>,
    proj_scale: PyReadonlyArray1<'py, f64>,
    grad_x_hat: PyReadonlyArray2<'py, f64>,
    curvature: f64,
) -> PyResult<(Py<PyArray2<f64>>, Py<PyArray2<f64>>)> {
    let atoms_p = atoms_projected.as_array().to_owned();
    let gates_owned = gates.as_array().to_owned();
    let v_owned = v.as_array().to_owned();
    let tangents_owned = tangents.as_array().to_owned();
    let proj_scale_owned = proj_scale.as_array().to_owned();
    let grad_owned = grad_x_hat.as_array().to_owned();
    let (gg, ga) = detach_geometry_result(py, "poincare_tangent_decode_backward", move || {
        let cache = gam::geometry::poincare::TangentDecodeCache {
            atoms_projected: atoms_p,
            gates: gates_owned,
            v: v_owned,
            tangents: tangents_owned,
            proj_scale: proj_scale_owned,
            curvature,
        };
        poincare_tangent_decode_backward_impl(&cache, grad_owned.view())
    })?;
    Ok((gg.into_pyarray(py).unbind(), ga.into_pyarray(py).unbind()))
}

#[pyfunction]
fn poincare_lorentz_decode_forward<'py>(
    py: Python<'py>,
    atoms: PyReadonlyArray2<'py, f64>,
    gates: PyReadonlyArray2<'py, f64>,
    curvature: f64,
) -> PyResult<Py<PyArray2<f64>>> {
    let atoms_owned = atoms.as_array().to_owned();
    let gates_owned = gates.as_array().to_owned();
    let out = detach_geometry_result(py, "poincare_lorentz_decode_forward", move || {
        poincare_lorentz_decode_forward_impl(atoms_owned.view(), gates_owned.view(), curvature)
    })?;
    Ok(out.into_pyarray(py).unbind())
}

/// Backward pass for the Lorentz-path decoder. Consumes the cache returned
/// by `poincare_tangent_decode_forward` (the Poincaré and Lorentz tangent
/// decoders are algebraically the same function of the inputs, so the
/// Poincaré cache is the right state to differentiate from). Returns
/// `(grad_gates, grad_atoms)`.
#[pyfunction]
fn poincare_lorentz_decode_backward<'py>(
    py: Python<'py>,
    atoms_projected: PyReadonlyArray2<'py, f64>,
    gates: PyReadonlyArray2<'py, f64>,
    v: PyReadonlyArray2<'py, f64>,
    tangents: PyReadonlyArray2<'py, f64>,
    proj_scale: PyReadonlyArray1<'py, f64>,
    grad_x_hat: PyReadonlyArray2<'py, f64>,
    curvature: f64,
) -> PyResult<(Py<PyArray2<f64>>, Py<PyArray2<f64>>)> {
    let atoms_p = atoms_projected.as_array().to_owned();
    let gates_owned = gates.as_array().to_owned();
    let v_owned = v.as_array().to_owned();
    let tangents_owned = tangents.as_array().to_owned();
    let proj_scale_owned = proj_scale.as_array().to_owned();
    let grad_owned = grad_x_hat.as_array().to_owned();
    let (gg, ga) = detach_geometry_result(py, "poincare_lorentz_decode_backward", move || {
        let cache = gam::geometry::poincare::TangentDecodeCache {
            atoms_projected: atoms_p,
            gates: gates_owned,
            v: v_owned,
            tangents: tangents_owned,
            proj_scale: proj_scale_owned,
            curvature,
        };
        poincare_lorentz_decode_backward_impl(&cache, grad_owned.view())
    })?;
    Ok((gg.into_pyarray(py).unbind(), ga.into_pyarray(py).unbind()))
}

#[pyfunction(signature = (values, base, coordinates, reference = -1))]
fn response_geometry_simplex_log_map<'py>(
    py: Python<'py>,
    values: PyReadonlyArray2<'py, f64>,
    base: PyReadonlyArray1<'py, f64>,
    coordinates: String,
    reference: isize,
) -> PyResult<Py<PyArray2<f64>>> {
    let arr = values.as_array().to_owned();
    let base_owned = base.as_array().to_owned();
    let out = detach_py_result(py, "response_geometry_simplex_log_map", move || {
        let coord = gam::geometry::simplex::parse_simplex_coord(&coordinates)?;
        gam::geometry::simplex::simplex_log_map(arr.view(), base_owned.view(), coord, reference)
    })?;
    Ok(out.into_pyarray(py).unbind())
}

#[pyfunction(signature = (tangent, base, coordinates, reference = -1))]
fn response_geometry_simplex_exp_map<'py>(
    py: Python<'py>,
    tangent: PyReadonlyArray2<'py, f64>,
    base: PyReadonlyArray1<'py, f64>,
    coordinates: String,
    reference: isize,
) -> PyResult<Py<PyArray2<f64>>> {
    let t_owned = tangent.as_array().to_owned();
    let base_owned = base.as_array().to_owned();
    let out = detach_py_result(py, "response_geometry_simplex_exp_map", move || {
        let coord = gam::geometry::simplex::parse_simplex_coord(&coordinates)?;
        gam::geometry::simplex::simplex_exp_map(t_owned.view(), base_owned.view(), coord, reference)
    })?;
    Ok(out.into_pyarray(py).unbind())
}

#[pyfunction]
fn response_geometry_sphere_log_map<'py>(
    py: Python<'py>,
    values: PyReadonlyArray2<'py, f64>,
    base: PyReadonlyArray1<'py, f64>,
) -> PyResult<Py<PyArray2<f64>>> {
    let arr = values.as_array().to_owned();
    let base_owned = base.as_array().to_owned();
    let out = detach_py_result(py, "response_geometry_sphere_log_map", move || {
        gam::geometry::sphere::response_sphere_log_map(arr.view(), base_owned.view())
    })?;
    Ok(out.into_pyarray(py).unbind())
}

#[pyfunction]
fn response_geometry_sphere_exp_map<'py>(
    py: Python<'py>,
    tangent: PyReadonlyArray2<'py, f64>,
    base: PyReadonlyArray1<'py, f64>,
) -> PyResult<Py<PyArray2<f64>>> {
    let t_owned = tangent.as_array().to_owned();
    let base_owned = base.as_array().to_owned();
    let out = detach_py_result(py, "response_geometry_sphere_exp_map", move || {
        gam::geometry::sphere::response_sphere_exp_map(t_owned.view(), base_owned.view())
    })?;
    Ok(out.into_pyarray(py).unbind())
}

#[pyfunction]
fn response_geometry_sphere_normalize_base<'py>(
    py: Python<'py>,
    base: PyReadonlyArray1<'py, f64>,
) -> PyResult<Py<PyArray1<f64>>> {
    let owned = base.as_array().to_owned();
    let normalized = py.detach(move || rg_normalize_sphere_base(owned.view()));
    let normalized = normalized.map_err(PyValueError::new_err)?;
    Ok(normalized.into_pyarray(py).unbind())
}

// ── Torch compositional-geometry surface (ILR chart + autograd jets) ─────────
//
// These back `gamfit.torch.geometry`. The ILR chart, its inverse, and the ALR
// Aitchison Gram have no NumPy twin (the `_response_geometry` surface stops at
// CLR/ALR); the `*_jet` variants return the per-row analytic Jacobian the torch
// autograd `Function`s contract against so gradients survive the FFI round-trip.
// All math lives in `gam::geometry::manifolds::aitchison_ilr`.

#[pyfunction]
fn response_geometry_ilr<'py>(
    py: Python<'py>,
    values: PyReadonlyArray2<'py, f64>,
) -> PyResult<Py<PyArray2<f64>>> {
    let arr = values.as_array().to_owned();
    let out = detach_py_result(py, "response_geometry_ilr", move || {
        gam::geometry::manifolds::aitchison_ilr::ilr(arr.view())
    })?;
    Ok(out.into_pyarray(py).unbind())
}

#[pyfunction]
fn response_geometry_inverse_ilr<'py>(
    py: Python<'py>,
    coords: PyReadonlyArray2<'py, f64>,
) -> PyResult<Py<PyArray2<f64>>> {
    let arr = coords.as_array().to_owned();
    let out = detach_py_result(py, "response_geometry_inverse_ilr", move || {
        gam::geometry::manifolds::aitchison_ilr::inverse_ilr(arr.view())
    })?;
    Ok(out.into_pyarray(py).unbind())
}

#[pyfunction]
fn response_geometry_aitchison_metric<'py>(
    py: Python<'py>,
    parts: usize,
) -> PyResult<Py<PyArray2<f64>>> {
    let out = detach_py_result(py, "response_geometry_aitchison_metric", move || {
        gam::geometry::manifolds::aitchison_ilr::aitchison_metric(parts)
    })?;
    Ok(out.into_pyarray(py).unbind())
}

/// CLR value and its per-row Jacobian `J[n,j,i] = ∂clr(x)_{n,j}/∂x_{n,i}` for the
/// torch autograd `Function`.
#[pyfunction]
fn response_geometry_clr_jet<'py>(
    py: Python<'py>,
    values: PyReadonlyArray2<'py, f64>,
) -> PyResult<(Py<PyArray2<f64>>, Py<PyArray3<f64>>)> {
    let arr = values.as_array().to_owned();
    let (value, jac) = detach_py_result(py, "response_geometry_clr_jet", move || {
        gam::geometry::manifolds::aitchison_ilr::clr_jet(arr.view())
    })?;
    Ok((
        value.into_pyarray(py).unbind(),
        jac.into_pyarray(py).unbind(),
    ))
}

/// Simplex log map value and its per-row Jacobian w.r.t. `values` (base held
/// constant). `coordinates` is `"ilr"`/`"simplex"`, `"clr"`, or `"alr"`.
#[pyfunction(signature = (values, base, coordinates, reference = -1))]
fn response_geometry_simplex_log_map_jet<'py>(
    py: Python<'py>,
    values: PyReadonlyArray2<'py, f64>,
    base: PyReadonlyArray1<'py, f64>,
    coordinates: String,
    reference: isize,
) -> PyResult<(Py<PyArray2<f64>>, Py<PyArray3<f64>>)> {
    let arr = values.as_array().to_owned();
    let base_owned = base.as_array().to_owned();
    let (value, jac) = detach_py_result(py, "response_geometry_simplex_log_map_jet", move || {
        let coord = gam::geometry::manifolds::aitchison_ilr::parse_log_ratio_coord(&coordinates)?;
        gam::geometry::manifolds::aitchison_ilr::simplex_log_map_jet(
            arr.view(),
            base_owned.view(),
            coord,
            reference,
        )
    })?;
    Ok((
        value.into_pyarray(py).unbind(),
        jac.into_pyarray(py).unbind(),
    ))
}

/// Simplex exp map value and its per-row Jacobian w.r.t. `tangent` (base held
/// constant), inverting [`response_geometry_simplex_log_map_jet`].
#[pyfunction(signature = (tangent, base, coordinates, reference = -1))]
fn response_geometry_simplex_exp_map_jet<'py>(
    py: Python<'py>,
    tangent: PyReadonlyArray2<'py, f64>,
    base: PyReadonlyArray1<'py, f64>,
    coordinates: String,
    reference: isize,
) -> PyResult<(Py<PyArray2<f64>>, Py<PyArray3<f64>>)> {
    let t_owned = tangent.as_array().to_owned();
    let base_owned = base.as_array().to_owned();
    let (value, jac) = detach_py_result(py, "response_geometry_simplex_exp_map_jet", move || {
        let coord = gam::geometry::manifolds::aitchison_ilr::parse_log_ratio_coord(&coordinates)?;
        gam::geometry::manifolds::aitchison_ilr::simplex_exp_map_jet(
            t_owned.view(),
            base_owned.view(),
            coord,
            reference,
        )
    })?;
    Ok((
        value.into_pyarray(py).unbind(),
        jac.into_pyarray(py).unbind(),
    ))
}

/// Sphere exp map value and its per-row Jacobian w.r.t. `tangent` (base held
/// constant) for the torch autograd `Function`.
#[pyfunction]
fn response_geometry_sphere_exp_map_jet<'py>(
    py: Python<'py>,
    tangent: PyReadonlyArray2<'py, f64>,
    base: PyReadonlyArray1<'py, f64>,
) -> PyResult<(Py<PyArray2<f64>>, Py<PyArray3<f64>>)> {
    let t_owned = tangent.as_array().to_owned();
    let base_owned = base.as_array().to_owned();
    let (value, jac) = detach_py_result(py, "response_geometry_sphere_exp_map_jet", move || {
        gam::geometry::manifolds::aitchison_ilr::sphere_exp_map_jet(
            t_owned.view(),
            base_owned.view(),
        )
    })?;
    Ok((
        value.into_pyarray(py).unbind(),
        jac.into_pyarray(py).unbind(),
    ))
}

/// Consolidated response-geometry log map. Owns geometry-kind routing,
/// coordinate resolution, and base-point selection (intrinsic Fréchet mean when
/// `base` is `None`) so the Python wrapper marshals arrays only. Returns
/// `(tangent, base_point, resolved_coordinate_label)`.
#[pyfunction(signature = (values, geometry, base=None, coordinates=None, reference=-1, weights=None))]
fn response_geometry_log_map<'py>(
    py: Python<'py>,
    values: PyReadonlyArray2<'py, f64>,
    geometry: String,
    base: Option<PyReadonlyArray1<'py, f64>>,
    coordinates: Option<String>,
    reference: isize,
    weights: Option<PyReadonlyArray1<'py, f64>>,
) -> PyResult<(Py<PyArray2<f64>>, Py<PyArray1<f64>>, String)> {
    let arr = values.as_array().to_owned();
    let base_owned = base.as_ref().map(|b| b.as_array().to_owned());
    let weights_owned = weights.as_ref().map(|w| w.as_array().to_owned());
    let (tangent, base_point, coord_label) =
        detach_py_result(py, "response_geometry_log_map", move || {
            rg_log_map_dispatch(
                arr.view(),
                &geometry,
                base_owned.as_ref().map(|b| b.view()),
                coordinates.as_deref(),
                reference,
                weights_owned.as_ref().map(|w| w.view()),
            )
        })?;
    Ok((
        tangent.into_pyarray(py).unbind(),
        base_point.into_pyarray(py).unbind(),
        coord_label,
    ))
}

/// Consolidated response-geometry exponential map. Dispatches tangent
/// coordinates back to the response manifold given the geometry kind and the
/// (already resolved) coordinate label.
#[pyfunction(signature = (tangent, geometry, base, coordinates=None, reference=-1))]
fn response_geometry_exp_map<'py>(
    py: Python<'py>,
    tangent: PyReadonlyArray2<'py, f64>,
    geometry: String,
    base: PyReadonlyArray1<'py, f64>,
    coordinates: Option<String>,
    reference: isize,
) -> PyResult<Py<PyArray2<f64>>> {
    let t_owned = tangent.as_array().to_owned();
    let base_owned = base.as_array().to_owned();
    let out = detach_py_result(py, "response_geometry_exp_map", move || {
        rg_exp_map_dispatch(
            t_owned.view(),
            &geometry,
            base_owned.view(),
            coordinates.as_deref(),
            reference,
        )
    })?;
    Ok(out.into_pyarray(py).unbind())
}

/// Fit curvature as an estimand on a constant-curvature response geometry
/// (#944 stage 4 / #1104). The user requests `response_geometry="constant_
/// curvature(dim=d)"`; κ is NOT supplied — it is estimated from the manifold-
/// valued responses by the REML/evidence outer loop (the profiled Fréchet-
/// dispersion criterion), and the fit reports κ̂ with a profile-likelihood CI,
/// the geometry verdict, and the interior-point Wilks flatness test of κ = 0.
///
/// Returns the summary tuple
/// `(kappa_hat, ci_lo, ci_hi, lo_at_bound, hi_at_bound, verdict, lr_stat,
///   p_value, railed_at_resolution_limit, kappa_r2, characteristic_radius,
///   base_point)` for the response-geometry fit summary.
///
/// `railed_at_resolution_limit` (#1104): `true` when the cloud is curved BEYOND
/// what its spread can resolve (it fills the sphere), so κ̂ railed to the chart
/// conjugate cap — κ̂/`ci_hi` are then a LOWER BOUND on |κ|, an honest "curvature
/// exceeds chart-resolvable range at this scale", NOT a resolved point estimate.
/// `kappa_r2 = κ̂·r²` is the scale-FREE invariant the cloud determines (invariant
/// under `y ↦ αy`); `characteristic_radius = r` is the κ=0 doubled-gauge spread it
/// is dimensionless to. Read these (not the scale-dependent κ̂ alone) when the data
/// have been arbitrarily rescaled — e.g. unit-normalised OLMo activations.
#[pyfunction(signature = (values, geometry, level=0.95))]
fn response_geometry_fit_curvature<'py>(
    py: Python<'py>,
    values: PyReadonlyArray2<'py, f64>,
    geometry: String,
    level: f64,
) -> PyResult<(
    f64,
    f64,
    f64,
    bool,
    bool,
    String,
    f64,
    f64,
    bool,
    f64,
    f64,
    Py<PyArray1<f64>>,
)> {
    let arr = values.as_array().to_owned();
    let fit = detach_py_result(py, "response_geometry_fit_curvature", move || {
        let manifold =
            gam::geometry::response_geometry::ResponseManifold::parse(&geometry, arr.ncols())?;
        let dim = match manifold {
            gam::geometry::response_geometry::ResponseManifold::ConstantCurvature {
                dim, ..
            } => dim,
            other => {
                return Err(format!(
                    "response_geometry_fit_curvature requires a constant_curvature geometry; \
                     got {}",
                    other.canonical_label()
                ));
            }
        };
        gam::geometry::response_geometry::fit_response_curvature(
            arr.view(),
            dim,
            level,
            1.0e-12,
            256,
        )
    })?;
    let verdict = match fit.profile_ci.verdict {
        gam::geometry::CurvatureVerdict::Spherical => "spherical",
        gam::geometry::CurvatureVerdict::Hyperbolic => "hyperbolic",
        gam::geometry::CurvatureVerdict::Flat => "flat",
    }
    .to_string();
    Ok((
        fit.kappa_hat,
        fit.profile_ci.ci_lo,
        fit.profile_ci.ci_hi,
        fit.profile_ci.lo_at_bound,
        fit.profile_ci.hi_at_bound,
        verdict,
        fit.flatness.lr_stat,
        fit.flatness.p_value,
        fit.railed_at_resolution_limit,
        fit.kappa_r2,
        fit.characteristic_radius,
        fit.base.into_pyarray(py).unbind(),
    ))
}

#[pyfunction]
fn sae_duchon_centers_nd<'py>(
    py: Python<'py>,
    centers_1d: PyReadonlyArray1<'py, f64>,
    d: usize,
) -> PyResult<Py<PyArray2<f64>>> {
    let centers_owned = centers_1d.as_array().to_owned();
    let out = py.detach(move || sae_duchon_centers_nd_impl(centers_owned.view(), d));
    Ok(out.into_pyarray(py).unbind())
}

/// Sinkhorn per-atom log-bias potentials that balance atom usage across the
/// batch (torch-lane routing helper for `ManifoldSAE`). Given the row-wise
/// log-responsibilities `log_scores[n, k]`, returns the additive potential
/// `b_k`; the Python caller forms the balanced responsibilities as
/// `log_scores + b`, keeping `log_scores` on the autograd tape while `b` is a
/// detached constant. See `gam::geometry::sae_routing::sinkhorn_balance_bias`.
#[pyfunction]
fn sae_sinkhorn_balance_bias<'py>(
    py: Python<'py>,
    log_scores: PyReadonlyArray2<'py, f64>,
    iters: usize,
) -> PyResult<Py<PyArray1<f64>>> {
    let scores_owned = log_scores.as_array().to_owned();
    let out = py.detach(move || sae_sinkhorn_balance_bias_impl(scores_owned.view(), iters));
    Ok(out.into_pyarray(py).unbind())
}

/// Exponential-moving-average blend of the per-row assignment accumulator for
/// the torch `softmax_topk` routing lane (issue #1282). Given the current
/// accumulator `prev (N, F)`, the freshly observed assignment signal
/// `signal (N, F)`, and the decay `beta`, returns `beta*prev + (1-beta)*signal`.
/// Both inputs are detached routing state; the Python `_update_assign_ema` keeps
/// the stateful orchestration (lazy sizing, reset on a row-count change, the
/// training-only guard) and delegates only this numeric recurrence so the EMA
/// math is single-sourced. See `gam::geometry::sae_routing::assign_ema_update`.
#[pyfunction]
fn sae_assign_ema_update<'py>(
    py: Python<'py>,
    prev: PyReadonlyArray2<'py, f64>,
    signal: PyReadonlyArray2<'py, f64>,
    beta: f64,
) -> PyResult<Py<PyArray2<f64>>> {
    let prev_owned = prev.as_array().to_owned();
    let signal_owned = signal.as_array().to_owned();
    let out =
        py.detach(move || sae_assign_ema_update_impl(prev_owned.view(), signal_owned.view(), beta));
    Ok(out.into_pyarray(py).unbind())
}

/// Residual-EM routing scores for the torch `softmax_topk` lane (issue #1282).
/// Given the input rows `x (N, D)` and the per-atom decoded curves
/// `per_atom_recon (N, F, D)`, solves the best scalar code against each atom's
/// curve and scores the atom by the scale-free relative residual it leaves,
/// returning `(code (N, F), relative_residual (N, F))`. `nonneg` selects the
/// `target_k == 1` non-negative code convention vs. the `target_k > 1` signed
/// one. This is the criterion math the Python gate used to compute inline; the
/// paired VJP `sae_residual_em_score_vjp` keeps the autograd tape continuous.
/// See `gam::terms::sae::criterion_atoms::residual_em_score`.
#[pyfunction]
fn sae_residual_em_score<'py>(
    py: Python<'py>,
    x: PyReadonlyArray2<'py, f64>,
    per_atom_recon: PyReadonlyArray3<'py, f64>,
    nonneg: bool,
) -> PyResult<(Py<PyArray2<f64>>, Py<PyArray2<f64>>)> {
    let x_owned = x.as_array().to_owned();
    let recon_owned = per_atom_recon.as_array().to_owned();
    let (code, relres) = py.detach(move || {
        gam::terms::sae::criterion_atoms::residual_em_score(
            x_owned.view(),
            recon_owned.view(),
            nonneg,
        )
    });
    Ok((
        code.into_pyarray(py).unbind(),
        relres.into_pyarray(py).unbind(),
    ))
}

/// Analytic backward (VJP) for `sae_residual_em_score`, w.r.t. `per_atom_recon`.
/// Given the upstream cotangents `g_code (N, F)` and `g_relative_residual
/// (N, F)`, returns `grad_per_atom_recon (N, F, D)`. `x` is a tape constant (the
/// activation batch never requires grad), so no `∂/∂x` channel is produced. See
/// `gam::terms::sae::criterion_atoms::residual_em_score_vjp`.
#[pyfunction]
fn sae_residual_em_score_vjp<'py>(
    py: Python<'py>,
    x: PyReadonlyArray2<'py, f64>,
    per_atom_recon: PyReadonlyArray3<'py, f64>,
    nonneg: bool,
    g_code: PyReadonlyArray2<'py, f64>,
    g_relative_residual: PyReadonlyArray2<'py, f64>,
) -> PyResult<Py<PyArray3<f64>>> {
    let x_owned = x.as_array().to_owned();
    let recon_owned = per_atom_recon.as_array().to_owned();
    let gc_owned = g_code.as_array().to_owned();
    let gq_owned = g_relative_residual.as_array().to_owned();
    let grad = py.detach(move || {
        gam::terms::sae::criterion_atoms::residual_em_score_vjp(
            x_owned.view(),
            recon_owned.view(),
            nonneg,
            gc_owned.view(),
            gq_owned.view(),
        )
    });
    Ok(grad.into_pyarray(py).unbind())
}

/// Deterministic line-clustering routing anchor for the torch `softmax_topk`
/// lane (issue #1282). Given the `(N, D)` input rows, returns
/// `(onehot (N, atoms), valid, confident)`: `valid` is false (with an empty
/// `onehot`) for the torch `(None, False)` cases (too few rows, `< 2` atoms, or
/// an eigensolver failure); otherwise `onehot` is the per-row cluster one-hot
/// and `confident` reports the balance/margin gate. See
/// `gam::geometry::sae_routing::direction_cluster_anchor`.
#[pyfunction]
fn sae_direction_cluster_anchor<'py>(
    py: Python<'py>,
    x: PyReadonlyArray2<'py, f64>,
    n_atoms: usize,
    iters: usize,
) -> PyResult<(Py<PyArray2<f64>>, bool, bool)> {
    let x_owned = x.as_array().to_owned();
    let result =
        py.detach(move || sae_direction_cluster_anchor_impl(x_owned.view(), n_atoms, iters));
    match result {
        Some((onehot, confident)) => Ok((onehot.into_pyarray(py).unbind(), true, confident)),
        None => Ok((
            Array2::<f64>::zeros((0, 0)).into_pyarray(py).unbind(),
            false,
            false,
        )),
    }
}

/// Balanced quadratic union-of-subspaces routing anchor for the torch
/// `softmax_topk` lane (issue #1282). Returns
/// `(onehot (N, 2), confident, i, j, threshold)`: when `confident` is true the
/// `onehot` is the accepted two-cluster split and `(i, j, threshold)` is the
/// transferable decision rule; otherwise `confident` is false with an empty
/// `onehot` and zeroed rule. See
/// `gam::geometry::sae_routing::quadratic_subspace_anchor`.
#[pyfunction]
fn sae_quadratic_subspace_anchor<'py>(
    py: Python<'py>,
    x: PyReadonlyArray2<'py, f64>,
    subspace_dim: usize,
) -> PyResult<(Py<PyArray2<f64>>, bool, usize, usize, f64)> {
    let x_owned = x.as_array().to_owned();
    let result =
        py.detach(move || sae_quadratic_subspace_anchor_impl(x_owned.view(), subspace_dim));
    match result {
        Some((onehot, i, j, threshold)) => {
            Ok((onehot.into_pyarray(py).unbind(), true, i, j, threshold))
        }
        None => Ok((
            Array2::<f64>::zeros((0, 0)).into_pyarray(py).unbind(),
            false,
            0,
            0,
            0.0,
        )),
    }
}

/// Apply a cached quadratic-subspace decision rule to route an arbitrary batch
/// (torch `softmax_topk` out-of-sample routing, issue #1282). Returns the
/// `(N, 2)` one-hot from the `(i, j, threshold)` split of `x_i·x_j` on the
/// L2-normalized rows. See `gam::geometry::sae_routing::apply_anchor_rule`.
#[pyfunction]
fn sae_apply_anchor_rule<'py>(
    py: Python<'py>,
    x: PyReadonlyArray2<'py, f64>,
    i: usize,
    j: usize,
    threshold: f64,
) -> PyResult<Py<PyArray2<f64>>> {
    let x_owned = x.as_array().to_owned();
    let out = py.detach(move || sae_apply_anchor_rule_impl(x_owned.view(), i, j, threshold));
    Ok(out.into_pyarray(py).unbind())
}

/// Residual-PC matching-pursuit commitment one-hot for the early training window
/// of the torch `softmax_topk` lane (issue #1282). Given `x (N, D)`, the per-atom
/// reconstructions `per_atom_recon (N, F, D)`, and the current non-negative codes
/// `code (N, F)`, returns `(onehot (N, F), valid)`. `valid` is false (empty
/// `onehot`) for the non-committing configuration (`step ≥ commit_steps`) or an
/// eigensolver failure — the torch `None` cases. See
/// `gam::geometry::sae_routing::matching_pursuit_commit`.
#[pyfunction]
fn sae_matching_pursuit_commit<'py>(
    py: Python<'py>,
    x: PyReadonlyArray2<'py, f64>,
    per_atom_recon: PyReadonlyArray3<'py, f64>,
    code: PyReadonlyArray2<'py, f64>,
    step: usize,
    commit_steps: usize,
    n_atoms: usize,
) -> PyResult<(Py<PyArray2<f64>>, bool)> {
    let x_owned = x.as_array().to_owned();
    let recon_owned = per_atom_recon.as_array().to_owned();
    let code_owned = code.as_array().to_owned();
    let result = py.detach(move || {
        sae_matching_pursuit_commit_impl(
            x_owned.view(),
            recon_owned.view(),
            code_owned.view(),
            step,
            commit_steps,
            n_atoms,
        )
    });
    match result {
        Some(onehot) => Ok((onehot.into_pyarray(py).unbind(), true)),
        None => Ok((
            Array2::<f64>::zeros((0, 0)).into_pyarray(py).unbind(),
            false,
        )),
    }
}

/// Batched rank-1 chart-coordinate E-step solver for the torch `ManifoldSAE`
/// `softmax_topk` lane (#2011-style Python→Rust trainer-math migration). For
/// every `(row, atom)` pair it solves the on-manifold coordinate by projecting
/// the target onto the atom's CURRENT decoded curve, amplitude profiled out,
/// over a grid of `8·K` points derived from the basis width. The solved
/// coordinates are E-step constants on the torch tape; the decoder gradient
/// still flows through the basis evaluation at those coordinates.
///
/// * `x` — `(N, D)` observations.
/// * `decoders` — `(F, K, D)` decoder blocks.
/// * `n_harmonics` — periodic (Fourier) basis harmonic count; the basis width
///   is `2·n_harmonics + 1` and must equal the decoder `K`.
/// * `prev_positions` / `gate_weights` — optional `(N, F)` matrices; when BOTH
///   are supplied the solver targets the leave-one-out residual
///   `x − Σ_{g≠f} gate_g · m_g(t_g)` (second sweep). When either is omitted the
///   plain target `x` is used (first sweep).
///
/// Returns solved coordinates `(N, F)` in `[0, 1)`. See
/// `gam_sae::chart_coordinate_solve::solve_chart_coordinates`.
#[pyfunction]
#[pyo3(signature = (x, decoders, n_harmonics, prev_positions = None, gate_weights = None))]
fn sae_solve_chart_coordinates<'py>(
    py: Python<'py>,
    x: PyReadonlyArray2<'py, f64>,
    decoders: PyReadonlyArray3<'py, f64>,
    n_harmonics: usize,
    prev_positions: Option<PyReadonlyArray2<'py, f64>>,
    gate_weights: Option<PyReadonlyArray2<'py, f64>>,
) -> PyResult<Py<PyArray2<f64>>> {
    let x_owned = x.as_array().to_owned();
    let dec_owned = decoders.as_array().to_owned();
    let prev_owned = prev_positions.map(|p| p.as_array().to_owned());
    let gate_owned = gate_weights.map(|w| w.as_array().to_owned());
    let out = py
        .detach(move || {
            let basis =
                gam::terms::sae::chart_coordinate_solve::ChartBasisKind::Periodic { n_harmonics };
            gam::terms::sae::chart_coordinate_solve::solve_chart_coordinates(
                x_owned.view(),
                dec_owned.view(),
                basis,
                prev_owned.as_ref().map(|p| p.view()),
                gate_owned.as_ref().map(|w| w.view()),
            )
        })
        .map_err(PyValueError::new_err)?;
    Ok(out.into_pyarray(py).unbind())
}

/// Per-row / per-atom SAE trust scores from an assignment matrix and the
/// per-atom trust vector. Returns `(row, per_atom)` where `row` is `(N,)` and
/// `per_atom` is `(N, K)`. See `gam_sae::trust_scores::row_trust_scores`.
#[pyfunction]
fn sae_row_trust_scores<'py>(
    py: Python<'py>,
    assignments: PyReadonlyArray2<'py, f64>,
    atom_trust: PyReadonlyArray1<'py, f64>,
) -> PyResult<(Py<PyArray1<f64>>, Py<PyArray2<f64>>)> {
    let assignments_owned = assignments.as_array().to_owned();
    let atom_trust_owned = atom_trust.as_array().to_owned();
    let (row, per_atom) = py
        .detach(move || {
            gam::terms::sae::trust_scores::row_trust_scores(
                assignments_owned.view(),
                atom_trust_owned.view(),
            )
        })
        .map_err(PyValueError::new_err)?;
    Ok((
        row.into_pyarray(py).unbind(),
        per_atom.into_pyarray(py).unbind(),
    ))
}

/// Value and encoder-gradient of the period-1 coordinate alignment penalty for
/// the torch manifold-SAE trainer. `encoder` and `solved` are `(N, F)`; returns
/// `(value, grad)` where `grad` is `∂value/∂encoder` `(N, F)`. `solved` is a
/// constant (the detached E-step solve). See
/// `gam_sae::chart_coordinate_solve::position_alignment_penalty`.
#[pyfunction]
fn sae_position_alignment_penalty<'py>(
    py: Python<'py>,
    encoder: PyReadonlyArray2<'py, f64>,
    solved: PyReadonlyArray2<'py, f64>,
) -> PyResult<(f64, Py<PyArray2<f64>>)> {
    let encoder_owned = encoder.as_array().to_owned();
    let solved_owned = solved.as_array().to_owned();
    let (value, grad) = py
        .detach(move || {
            gam::terms::sae::chart_coordinate_solve::position_alignment_penalty(
                encoder_owned.view(),
                solved_owned.view(),
            )
        })
        .map_err(PyValueError::new_err)?;
    Ok((value, grad.into_pyarray(py).unbind()))
}

/// Device-resident periodic basis+jet for the torch manifold-SAE lane.
///
/// `t_dev_ptr`, `phi_dev_ptr`, `jet_dev_ptr` are RAW CUDA device addresses of
/// caller-owned (torch) float64 tensors on device `ordinal`: `t` is `(n,)` (or
/// any contiguous layout of `n` doubles), `phi` and `jet` are contiguous
/// `(n, 2*n_harmonics + 1)` buffers the kernel fills in place. The caller must
/// have synchronized the producing stream (`torch.cuda.synchronize()`); the
/// kernel's stream is synchronized before this returns. Mirrors the CPU
/// `basis_with_jet("periodic", ...)` exactly — the d = 1 jet `(n, m, 1)` is
/// contiguous as `(n, m)`. See `gam_sae::basis_gpu`.
#[pyfunction]
fn sae_periodic_basis_with_jet_cuda(
    py: Python<'_>,
    ordinal: usize,
    t_dev_ptr: u64,
    n: usize,
    n_harmonics: usize,
    phi_dev_ptr: u64,
    jet_dev_ptr: u64,
) -> PyResult<()> {
    py.detach(move || {
        gam::terms::sae::basis_gpu::sae_periodic_basis_with_jet_device(
            ordinal,
            t_dev_ptr,
            n,
            n_harmonics,
            phi_dev_ptr,
            jet_dev_ptr,
        )
    })
    .map_err(PyValueError::new_err)
}

/// Width `(n_kernel + n_poly)` of the device Duchon basis for these centers —
/// the torch bridge allocates its output tensors with exactly this width
/// before calling [`sae_duchon_basis_with_jet_cuda`]. Also warms the device
/// cache (uploads centers/Z once).
#[pyfunction]
fn sae_duchon_device_basis_width(
    py: Python<'_>,
    ordinal: usize,
    centers: PyReadonlyArray2<'_, f64>,
    m: usize,
) -> PyResult<usize> {
    let centers_owned = centers.as_array().to_owned();
    py.detach(move || {
        gam::terms::sae::basis_gpu::sae_duchon_device_basis_width(ordinal, centers_owned.view(), m)
    })
    .map_err(PyValueError::new_err)
}

/// Device-resident Duchon basis+jet for the torch manifold-SAE lane — the
/// multi-`d` sibling of [`sae_periodic_basis_with_jet_cuda`] with the same
/// pointer and synchronization contract. `t` is `(n, dim)` doubles on device
/// `ordinal`; `phi` is `(n, width)` and `jet` is `(n, width, dim)` with
/// `width` from [`sae_duchon_device_basis_width`]. Center-derived state
/// (`Z`, amplification, polyharmonic coefficient, monomial exponents) is
/// cached on device per (centers, m). Returns the width actually written.
#[pyfunction]
fn sae_duchon_basis_with_jet_cuda(
    py: Python<'_>,
    ordinal: usize,
    device_ptrs: (u64, u64, u64),
    n: usize,
    centers: PyReadonlyArray2<'_, f64>,
    m: usize,
) -> PyResult<usize> {
    let (t_dev_ptr, phi_dev_ptr, jet_dev_ptr) = device_ptrs;
    let centers_owned = centers.as_array().to_owned();
    py.detach(move || {
        gam::terms::sae::basis_gpu::sae_duchon_basis_with_jet_device(
            ordinal,
            t_dev_ptr,
            n,
            centers_owned.view(),
            m,
            phi_dev_ptr,
            jet_dev_ptr,
        )
    })
    .map_err(PyValueError::new_err)
}

#[pyfunction]
fn sinkhorn_circular_cost<'py>(py: Python<'py>, m: usize) -> PyResult<Py<PyArray2<f64>>> {
    let out = py.detach(move || sinkhorn_circular_cost_impl(m));
    Ok(out.into_pyarray(py).unbind())
}

#[pyfunction]
fn sinkhorn_euclidean_cost<'py>(
    py: Python<'py>,
    points: PyReadonlyArray2<'py, f64>,
) -> PyResult<Py<PyArray2<f64>>> {
    let owned = points.as_array().to_owned();
    let out = detach_py_result(py, "sinkhorn_euclidean_cost", move || {
        sinkhorn_euclidean_cost_impl(owned.view())
    })?;
    Ok(out.into_pyarray(py).unbind())
}

#[pyfunction]
fn sinkhorn_geodesic_sphere_cost<'py>(
    py: Python<'py>,
    directions: PyReadonlyArray2<'py, f64>,
) -> PyResult<Py<PyArray2<f64>>> {
    let owned = directions.as_array().to_owned();
    let out = detach_py_result(py, "sinkhorn_geodesic_sphere_cost", move || {
        sinkhorn_geodesic_sphere_cost_impl(owned.view())
    })?;
    Ok(out.into_pyarray(py).unbind())
}

#[pyfunction]
fn sinkhorn_barycenter_forward<'py>(
    py: Python<'py>,
    atoms: PyReadonlyArray2<'py, f64>,
    weights: PyReadonlyArray1<'py, f64>,
    cost: PyReadonlyArray2<'py, f64>,
    eps: f64,
    n_iter: usize,
) -> PyResult<Py<PyArray1<f64>>> {
    let atoms_owned = atoms.as_array().to_owned();
    let weights_owned = weights.as_array().to_owned();
    let cost_owned = cost.as_array().to_owned();
    let out = detach_py_result(py, "sinkhorn_barycenter_forward", move || {
        sinkhorn_barycenter_impl(
            atoms_owned.view(),
            weights_owned.view(),
            cost_owned.view(),
            eps,
            n_iter,
        )
    })?;
    Ok(out.into_pyarray(py).unbind())
}

#[pyfunction]
fn sinkhorn_barycenter_vjp<'py>(
    py: Python<'py>,
    atoms: PyReadonlyArray2<'py, f64>,
    weights: PyReadonlyArray1<'py, f64>,
    cost: PyReadonlyArray2<'py, f64>,
    eps: f64,
    n_iter: usize,
    cotangent: PyReadonlyArray1<'py, f64>,
) -> PyResult<(Py<PyArray2<f64>>, Py<PyArray1<f64>>)> {
    let atoms_owned = atoms.as_array().to_owned();
    let weights_owned = weights.as_array().to_owned();
    let cost_owned = cost.as_array().to_owned();
    let cot_owned = cotangent.as_array().to_owned();
    let result = detach_py_result(py, "sinkhorn_barycenter_vjp", move || {
        sinkhorn_barycenter_vjp_impl(
            atoms_owned.view(),
            weights_owned.view(),
            cost_owned.view(),
            eps,
            n_iter,
            cot_owned.view(),
        )
    })?;
    let d_atoms = result.d_atoms.into_pyarray(py).unbind();
    let d_weights = result.d_weights.into_pyarray(py).unbind();
    Ok((d_atoms, d_weights))
}

#[pyfunction]
fn numerics_sigmoid_stable<'py>(
    py: Python<'py>,
    x: PyReadonlyArrayDyn<'py, f64>,
) -> PyResult<Py<PyArrayDyn<f64>>> {
    let owned = x.as_array().to_owned();
    let out = py.detach(move || owned.mapv(sigmoid_stable));
    Ok(out.into_pyarray(py).unbind())
}

#[pyfunction]
fn numerics_inverse_softplus<'py>(
    py: Python<'py>,
    x: PyReadonlyArrayDyn<'py, f64>,
) -> PyResult<Py<PyArrayDyn<f64>>> {
    let owned = x.as_array().to_owned();
    let out = py.detach(move || owned.mapv(inverse_softplus_scalar));
    Ok(out.into_pyarray(py).unbind())
}

#[pyfunction]
fn response_geometry_normalize_fisher_rao<'py>(
    py: Python<'py>,
    value: PyReadonlyArrayDyn<'py, f64>,
    n_rows: usize,
    dim: usize,
) -> PyResult<Py<PyArray3<f64>>> {
    let arr = value.as_array().to_owned();
    let out = detach_py_result(py, "response_geometry_normalize_fisher_rao", move || {
        gam::inference::fisher_rao::normalize_fisher_rao_blocks(arr.view(), n_rows, dim)
    })?;
    Ok(out.into_pyarray(py).unbind())
}

#[pyfunction(signature = (values, weights = None, tol = 1.0e-12, max_iter = 256))]
fn sphere_frechet_mean<'py>(
    py: Python<'py>,
    values: PyReadonlyArray2<'py, f64>,
    weights: Option<PyReadonlyArray1<'py, f64>>,
    tol: f64,
    max_iter: usize,
) -> PyResult<Py<PyArray1<f64>>> {
    let arr = values.as_array().to_owned();
    let w_owned = weights.as_ref().map(|w| w.as_array().to_owned());
    let mean = detach_py_result(py, "sphere_frechet_mean", move || {
        gam::geometry::sphere::sphere_frechet_mean(
            arr.view(),
            w_owned.as_ref().map(|w| w.view()),
            tol,
            max_iter,
        )
    })?;
    Ok(Array1::from(mean).into_pyarray(py).unbind())
}

#[pyfunction]
fn equivariant_rho<'py>(
    py: Python<'py>,
    group: String,
    g: PyReadonlyArrayDyn<'py, f64>,
) -> PyResult<Py<PyArrayDyn<f64>>> {
    let g_owned = g.as_array().to_owned();
    let out = detach_py_result(py, "equivariant_rho", move || {
        gam::geometry::lie_so::rho(group.as_str(), g_owned.view())
    })?;
    Ok(out.into_pyarray(py).unbind())
}

#[pyfunction]
fn equivariant_aux_enabled(aux: Option<String>) -> bool {
    aux.is_some()
}

#[pyfunction]
fn equivariant_rho_so2<'py>(
    py: Python<'py>,
    theta: PyReadonlyArray1<'py, f64>,
) -> PyResult<Py<PyArray3<f64>>> {
    let theta_owned = theta.as_array().to_owned();
    let out = detach_py_result(py, "equivariant_rho_so2", move || {
        Ok(gam::geometry::lie_so::rho_so2(theta_owned.view()))
    })?;
    Ok(out.into_pyarray(py).unbind())
}

#[pyfunction]
fn equivariant_rho_so2_jvp<'py>(
    py: Python<'py>,
    theta: PyReadonlyArray1<'py, f64>,
) -> PyResult<Py<PyArray3<f64>>> {
    let theta_owned = theta.as_array().to_owned();
    let out = detach_py_result(py, "equivariant_rho_so2_jvp", move || {
        Ok(gam::geometry::lie_so::rho_so2_jvp(theta_owned.view()))
    })?;
    Ok(out.into_pyarray(py).unbind())
}

#[pyfunction]
fn equivariant_rho_so3<'py>(
    py: Python<'py>,
    omega: PyReadonlyArray2<'py, f64>,
) -> PyResult<Py<PyArray3<f64>>> {
    let omega_owned = omega.as_array().to_owned();
    let out = detach_py_result(py, "equivariant_rho_so3", move || {
        gam::geometry::lie_so::rho_so3(omega_owned.view())
    })?;
    Ok(out.into_pyarray(py).unbind())
}

#[pyfunction]
fn equivariant_rho_so3_jvp<'py>(
    py: Python<'py>,
    omega: PyReadonlyArray2<'py, f64>,
    domega: PyReadonlyArray2<'py, f64>,
) -> PyResult<Py<PyArray3<f64>>> {
    let o_owned = omega.as_array().to_owned();
    let d_owned = domega.as_array().to_owned();
    let out = detach_py_result(py, "equivariant_rho_so3_jvp", move || {
        gam::geometry::lie_so::rho_so3_jvp(o_owned.view(), d_owned.view())
    })?;
    Ok(out.into_pyarray(py).unbind())
}

#[pyfunction(signature = (aux_values, theta, d_aux, weight = 1.0))]
fn equivariant_gauge_companion_loss<'py>(
    py: Python<'py>,
    aux_values: Option<PyReadonlyArray2<'py, f64>>,
    theta: PyReadonlyArray2<'py, f64>,
    d_aux: usize,
    weight: f64,
) -> PyResult<f64> {
    let Some(aux_values) = aux_values else {
        return Ok(0.0);
    };
    let aux_owned = aux_values.as_array().to_owned();
    let theta_owned = theta.as_array().to_owned();
    detach_py_result(py, "equivariant_gauge_companion_loss", move || {
        gam::terms::analytic_penalties::equivariant_penalty::gauge_companion_loss(
            aux_owned.view(),
            theta_owned.view(),
            d_aux,
            weight,
        )
    })
}

#[pyclass(module = "gam_pyffi._rust", name = "SparsityPenalty")]
struct SparsityPenalty {
    #[pyo3(get, set)]
    target: PyObject,
    #[pyo3(get, set)]
    kind: String,
    #[pyo3(get, set)]
    weight: PyObject,
    #[pyo3(get, set)]
    eps: f64,
    #[pyo3(get, set)]
    eps_weight: String,
    #[pyo3(get)]
    weight_schedule: Option<PyObject>,
}

#[pymethods]
impl SparsityPenalty {
    #[new]
    #[pyo3(signature = (kind = "smooth_l1".to_string(), weight = None, eps = 1.0e-3, eps_weight = "fixed".to_string(), *, target = None, weight_schedule = None))]
    fn new(
        py: Python<'_>,
        kind: String,
        weight: Option<&Bound<'_, PyAny>>,
        eps: f64,
        eps_weight: String,
        target: Option<&Bound<'_, PyAny>>,
        weight_schedule: Option<PyObject>,
    ) -> PyResult<Self> {
        let weight = py_object_or_string_default(py, weight, "auto");
        let target = py_object_or_string_default(py, target, "t");
        validate_sparsity_weight(py, weight.bind(py), "SparsityPenalty")?;
        if !matches!(kind.as_str(), "smooth_l1" | "hoyer" | "log") {
            return Err(PyValueError::new_err(format!(
                "SparsityPenalty.kind must be one of 'smooth_l1' | 'hoyer' | 'log', got {kind:?}"
            )));
        }
        if kind == "hoyer" && eps_weight == "auto" {
            return Err(PyValueError::new_err(
                "SparsityPenalty(kind='hoyer'): Hoyer has no smoothing scale, so eps_weight='auto' is not meaningful.",
            ));
        }
        if eps <= 0.0 {
            return Err(PyValueError::new_err(format!(
                "SparsityPenalty.eps must be > 0, got {eps}"
            )));
        }
        if !matches!(eps_weight.as_str(), "auto" | "fixed") {
            return Err(PyValueError::new_err(format!(
                "SparsityPenalty.eps_weight must be 'auto' or 'fixed', got {eps_weight:?}"
            )));
        }
        Ok(Self {
            target,
            kind,
            weight,
            eps,
            eps_weight,
            weight_schedule,
        })
    }

    #[classattr]
    const KIND_TAG: &'static str = "sparsity";

    fn to_rust_descriptor(&self, py: Python<'_>) -> PyResult<PyObject> {
        let payload = PyDict::new(py);
        payload.set_item("kind", Self::KIND_TAG)?;
        payload.set_item("target", target_descriptor(py, self.target.bind(py))?)?;
        payload.set_item("sparsity_kind", &self.kind)?;
        payload.set_item("weight", self.weight.bind(py))?;
        payload.set_item("eps", self.eps)?;
        payload.set_item("eps_weight", &self.eps_weight)?;
        if let Some(schedule) = topk_weight_schedule_descriptor(py, &self.weight_schedule)? {
            payload.set_item("weight_schedule", schedule)?;
        }
        Ok(payload.into())
    }

    fn set_weight_schedule<'py>(
        mut slf: PyRefMut<'py, Self>,
        schedule: PyObject,
    ) -> PyRefMut<'py, Self> {
        slf.weight_schedule = Some(schedule);
        slf
    }

    fn __repr__(&self, py: Python<'_>) -> PyResult<String> {
        Ok(format!(
            "SparsityPenalty(kind={}, weight={}, eps={}, eps_weight={}, target={}, weight_schedule={})",
            self.kind.as_str(),
            py_repr(py, self.weight.bind(py))?,
            self.eps,
            self.eps_weight.as_str(),
            py_repr(py, self.target.bind(py))?,
            match &self.weight_schedule {
                Some(schedule) => py_repr(py, schedule.bind(py))?,
                None => "None".to_string(),
            }
        ))
    }
}

fn validate_sparsity_weight(py: Python<'_>, weight: &Bound<'_, PyAny>, name: &str) -> PyResult<()> {
    if let Ok(value) = weight.extract::<String>() {
        if value == "auto" {
            return Ok(());
        }
        return Err(PyValueError::new_err(format!(
            "{name}.weight: only 'auto' is accepted as a string, got {value:?}"
        )));
    }
    if let Ok(value) = weight.extract::<f64>() {
        if value > 0.0 {
            return Ok(());
        }
        return Err(PyValueError::new_err(format!(
            "{name}.weight must be > 0, got {}",
            py_repr(py, weight)?
        )));
    }
    Err(PyTypeError::new_err(format!(
        "{name}.weight must be 'auto' or a positive float, got {}",
        weight.get_type().name()?
    )))
}

fn py_object_or_string_default(
    py: Python<'_>,
    value: Option<&Bound<'_, PyAny>>,
    default: &str,
) -> PyObject {
    match value {
        Some(value) => value.clone().unbind(),
        None => pyo3::types::PyString::new(py, default).into_any().unbind(),
    }
}

fn target_descriptor(py: Python<'_>, target: &Bound<'_, PyAny>) -> PyResult<PyObject> {
    if target.is_instance_of::<PyString>() || target.extract::<isize>().is_ok() {
        return Ok(target.clone().unbind());
    }
    if let Ok(name) = target.getattr("name") {
        return Ok(name.str()?.into_pyobject(py)?.unbind().into());
    }
    Err(PyTypeError::new_err(
        "analytic penalty target must be a latent block name, latent block index, or object exposing a 'name' attribute",
    ))
}

/// Eagerly validate that an analytic-penalty `target` keyword argument is well
/// formed at construction time, rather than deferring the check until
/// `to_rust_descriptor` is invoked. This lets users see configuration errors
/// at the moment they create the penalty object, with the offending class
/// named in the message so the source of the bad target is obvious.
fn validate_target_eager(
    py: Python<'_>,
    class_name: &str,
    target: Option<&Bound<'_, PyAny>>,
) -> PyResult<()> {
    let Some(target) = target else {
        return Ok(());
    };
    if target.is_none() {
        return Ok(());
    }
    if target.is_instance_of::<PyString>() || target.extract::<isize>().is_ok() {
        return Ok(());
    }
    if let Ok(name) = target.getattr("name") {
        // Require that `.name` actually resolves to a string-coercible value.
        name.str()?.into_pyobject(py)?;
        return Ok(());
    }
    Err(PyTypeError::new_err(format!(
        "{class_name}.target must be a latent block name (str), index (int), or object exposing a 'name' attribute; got {}",
        target.get_type().name()?
    )))
}

fn py_repr(_py: Python<'_>, value: &Bound<'_, PyAny>) -> PyResult<String> {
    value.repr()?.extract()
}

#[pyclass(module = "gam_pyffi._rust", name = "ARDPenalty")]
struct ARDPenalty {
    #[pyo3(get, set)]
    target: PyObject,
    #[pyo3(get)]
    weight_schedule: Option<PyObject>,
}

#[pymethods]
impl ARDPenalty {
    #[new]
    #[pyo3(signature = (*, target = None, weight_schedule = None))]
    fn new(
        py: Python<'_>,
        target: Option<&Bound<'_, PyAny>>,
        weight_schedule: Option<PyObject>,
    ) -> PyResult<Self> {
        validate_target_eager(py, "ARDPenalty", target)?;
        Ok(Self {
            target: py_object_or_string_default(py, target, "t"),
            weight_schedule,
        })
    }

    #[classattr]
    const KIND_TAG: &'static str = "ard";

    fn to_rust_descriptor(&self, py: Python<'_>) -> PyResult<PyObject> {
        let payload = PyDict::new(py);
        payload.set_item("kind", Self::KIND_TAG)?;
        payload.set_item("target", target_descriptor(py, self.target.bind(py))?)?;
        if let Some(schedule) = topk_weight_schedule_descriptor(py, &self.weight_schedule)? {
            payload.set_item("weight_schedule", schedule)?;
        }
        Ok(payload.into())
    }

    fn set_weight_schedule<'py>(
        mut slf: PyRefMut<'py, Self>,
        schedule: PyObject,
    ) -> PyRefMut<'py, Self> {
        slf.weight_schedule = Some(schedule);
        slf
    }

    fn __repr__(&self, py: Python<'_>) -> PyResult<String> {
        Ok(format!(
            "ARDPenalty(target={}, weight_schedule={})",
            py_repr(py, self.target.bind(py))?,
            match &self.weight_schedule {
                Some(schedule) => py_repr(py, schedule.bind(py))?,
                None => "None".to_string(),
            }
        ))
    }
}

#[pyclass(module = "gam_pyffi._rust", name = "TopKActivationPenalty")]
struct PyTopKActivationPenalty {
    #[pyo3(get, set)]
    target: PyObject,
    #[pyo3(get, set)]
    k: i64,
    #[pyo3(get, set)]
    weight: f64,
    #[pyo3(get)]
    weight_schedule: Option<PyObject>,
}

#[pymethods]
impl PyTopKActivationPenalty {
    #[new]
    #[pyo3(signature = (k, weight = 1.0, *, target = None, weight_schedule = None))]
    fn new(
        py: Python<'_>,
        k: &Bound<'_, PyAny>,
        weight: f64,
        target: Option<&Bound<'_, PyAny>>,
        weight_schedule: Option<PyObject>,
    ) -> PyResult<Self> {
        let k = k.call_method0("__index__")?.extract::<i64>()?;
        if k <= 0 {
            return Err(PyValueError::new_err(format!(
                "TopKActivationPenalty.k must be > 0, got {k}"
            )));
        }
        if !(weight > 0.0) {
            return Err(PyValueError::new_err(format!(
                "TopKActivationPenalty.weight must be finite and > 0, got {weight}"
            )));
        }
        Ok(Self {
            target: py_object_or_string_default(py, target, "t"),
            k,
            weight,
            weight_schedule,
        })
    }

    #[classattr]
    const KIND_TAG: &'static str = "topk_activation";

    fn to_rust_descriptor(&self, py: Python<'_>) -> PyResult<PyObject> {
        let payload = PyDict::new(py);
        payload.set_item("kind", Self::KIND_TAG)?;
        payload.set_item("target", target_descriptor(py, self.target.bind(py))?)?;
        payload.set_item("k", self.k)?;
        payload.set_item("weight", self.weight)?;
        if let Some(schedule) = topk_weight_schedule_descriptor(py, &self.weight_schedule)? {
            payload.set_item("weight_schedule", schedule)?;
        }
        Ok(payload.into())
    }

    fn set_weight_schedule<'py>(
        mut slf: PyRefMut<'py, Self>,
        schedule: PyObject,
    ) -> PyRefMut<'py, Self> {
        slf.weight_schedule = Some(schedule);
        slf
    }

    fn __repr__(&self, py: Python<'_>) -> PyResult<String> {
        Ok(format!(
            "TopKActivationPenalty(k={}, weight={}, target={}, weight_schedule={})",
            self.k,
            self.weight,
            py_repr(py, self.target.bind(py))?,
            match &self.weight_schedule {
                Some(schedule) => py_repr(py, schedule.bind(py))?,
                None => "None".to_string(),
            }
        ))
    }
}

#[pyclass(module = "gam_pyffi._rust", name = "JumpReLUPenalty")]
struct JumpReLUPenalty {
    #[pyo3(get, set)]
    target: PyObject,
    #[pyo3(get, set)]
    thresholds: Vec<f64>,
    #[pyo3(get, set)]
    weight: f64,
    #[pyo3(get, set)]
    smoothing_eps: f64,
    #[pyo3(get)]
    weight_schedule: Option<PyObject>,
}

#[pymethods]
impl JumpReLUPenalty {
    #[new]
    #[pyo3(signature = (thresholds, weight = 1.0, smoothing_eps = 1.0e-3, *, target = None, weight_schedule = None))]
    fn new(
        py: Python<'_>,
        thresholds: &Bound<'_, PyAny>,
        weight: f64,
        smoothing_eps: f64,
        target: Option<&Bound<'_, PyAny>>,
        weight_schedule: Option<PyObject>,
    ) -> PyResult<Self> {
        let numpy = py.import("numpy")?;
        let kwargs = PyDict::new(py);
        kwargs.set_item("dtype", "float")?;
        let thresholds_array = numpy
            .call_method("asarray", (thresholds,), Some(&kwargs))?
            .call_method1("reshape", (-1,))?;
        let thresholds = thresholds_array
            .call_method0("tolist")?
            .extract::<Vec<f64>>()?;
        if thresholds.is_empty() {
            return Err(PyValueError::new_err(
                "JumpReLUPenalty.thresholds must be non-empty",
            ));
        }
        if thresholds
            .iter()
            .any(|threshold| !threshold.is_finite() || *threshold <= 0.0)
        {
            return Err(PyValueError::new_err(
                "JumpReLUPenalty.thresholds must be finite and > 0",
            ));
        }
        if !(weight > 0.0) {
            return Err(PyValueError::new_err(format!(
                "JumpReLUPenalty.weight must be finite and > 0, got {weight}"
            )));
        }
        if !(smoothing_eps > 0.0) {
            return Err(PyValueError::new_err(format!(
                "JumpReLUPenalty.smoothing_eps must be finite and > 0, got {smoothing_eps}"
            )));
        }
        Ok(Self {
            target: py_object_or_string_default(py, target, "t"),
            thresholds,
            weight,
            smoothing_eps,
            weight_schedule,
        })
    }

    #[classattr]
    const KIND_TAG: &'static str = "jumprelu";

    fn to_rust_descriptor(&self, py: Python<'_>) -> PyResult<PyObject> {
        let payload = PyDict::new(py);
        payload.set_item("kind", Self::KIND_TAG)?;
        payload.set_item("target", target_descriptor(py, self.target.bind(py))?)?;
        payload.set_item("thresholds", self.thresholds.clone())?;
        payload.set_item("weight", self.weight)?;
        payload.set_item("smoothing_eps", self.smoothing_eps)?;
        if let Some(schedule) = topk_weight_schedule_descriptor(py, &self.weight_schedule)? {
            payload.set_item("weight_schedule", schedule)?;
        }
        Ok(payload.into())
    }

    fn set_weight_schedule<'py>(
        mut slf: PyRefMut<'py, Self>,
        schedule: PyObject,
    ) -> PyRefMut<'py, Self> {
        slf.weight_schedule = Some(schedule);
        slf
    }

    fn __repr__(&self, py: Python<'_>) -> PyResult<String> {
        Ok(format!(
            "JumpReLUPenalty(thresholds={:?}, weight={}, smoothing_eps={}, target={}, weight_schedule={})",
            self.thresholds,
            self.weight,
            self.smoothing_eps,
            py_repr(py, self.target.bind(py))?,
            match &self.weight_schedule {
                Some(schedule) => py_repr(py, schedule.bind(py))?,
                None => "None".to_string(),
            }
        ))
    }
}

fn topk_weight_schedule_descriptor(
    _py: Python<'_>,
    schedule: &Option<PyObject>,
) -> PyResult<Option<PyObject>> {
    let Some(schedule) = schedule else {
        return Ok(None);
    };
    let bound = schedule.bind(_py);
    if bound.hasattr("to_rust_descriptor")? {
        return Ok(Some(
            bound.call_method0("to_rust_descriptor")?.unbind().into(),
        ));
    }
    if let Ok(mapping) = bound.cast::<PyDict>() {
        return Ok(Some(mapping.copy()?.unbind().into()));
    }
    Err(PyTypeError::new_err(
        "weight_schedule must be ScalarWeightSchedule, a mapping, or None",
    ))
}

fn aux_conditional_prior_float_array<'py>(
    py: Python<'py>,
    value: &Bound<'py, PyAny>,
) -> PyResult<Bound<'py, PyAny>> {
    let numpy = py.import("numpy")?;
    let kwargs = PyDict::new(py);
    kwargs.set_item("dtype", "float")?;
    numpy.call_method("asarray", (value,), Some(&kwargs))
}

fn validate_aux_conditional_prior_lambda(
    lambda_per_row: &Bound<'_, PyAny>,
    weight: f64,
    n_eff: i64,
) -> PyResult<()> {
    if !(weight > 0.0) {
        return Err(PyValueError::new_err(format!(
            "AuxConditionalPriorPenalty.weight must be > 0, got {weight}"
        )));
    }
    if n_eff <= 0 {
        return Err(PyValueError::new_err(format!(
            "AuxConditionalPriorPenalty.n_eff must be > 0, got {n_eff}"
        )));
    }

    let array = lambda_per_row.extract::<PyReadonlyArrayDyn<'_, f64>>()?;
    let view = array.as_array();
    let shape = view.shape();
    if view.ndim() != 3 {
        return Err(PyValueError::new_err(format!(
            "AuxConditionalPriorPenalty.lambda_per_row must have shape (N, d, d), got ndim={}",
            view.ndim()
        )));
    }

    let n_obs = shape[0];
    let rows = shape[1];
    let cols = shape[2];
    if n_obs as i64 != n_eff {
        return Err(PyValueError::new_err(format!(
            "AuxConditionalPriorPenalty.lambda_per_row first dimension must equal n_eff={n_eff}, got {n_obs}"
        )));
    }
    if rows == 0 || cols == 0 || rows != cols {
        return Err(PyValueError::new_err(format!(
            "AuxConditionalPriorPenalty.lambda_per_row must have square non-empty row matrices, got shape ({n_obs}, {rows}, {cols})"
        )));
    }
    if !view.iter().all(|value| value.is_finite()) {
        return Err(PyValueError::new_err(
            "AuxConditionalPriorPenalty.lambda_per_row must be finite",
        ));
    }

    let mut max_asym = 0.0_f64;
    for obs in 0..n_obs {
        for row in 0..rows {
            for col in 0..cols {
                let asym = (view[IxDyn(&[obs, row, col])] - view[IxDyn(&[obs, col, row])]).abs();
                if asym > max_asym {
                    max_asym = asym;
                }
            }
        }
    }
    if max_asym >= 1.0e-10 {
        return Err(PyValueError::new_err(format!(
            "AuxConditionalPriorPenalty.lambda_per_row matrices must be symmetric within 1e-10; max asymmetry is {max_asym:.3e}"
        )));
    }
    Ok(())
}

#[pyclass(module = "gam_pyffi._rust", name = "BlockSparsityPenalty")]
struct BlockSparsityPenalty {
    #[pyo3(get, set)]
    target: PyObject,
    #[pyo3(get, set)]
    groups: Vec<Vec<usize>>,
    #[pyo3(get, set)]
    weight: f64,
    #[pyo3(get, set)]
    n_eff: i64,
    #[pyo3(get, set)]
    smoothing_eps: f64,
    #[pyo3(get, set)]
    learnable: bool,
    #[pyo3(get)]
    weight_schedule: Option<PyObject>,
}

#[pymethods]
impl BlockSparsityPenalty {
    #[new]
    #[pyo3(signature = (groups, weight, n_eff, smoothing_eps = 1e-6, learnable = false, *, target = None))]
    fn new(
        py: Python<'_>,
        groups: &Bound<'_, PyAny>,
        weight: f64,
        n_eff: i64,
        smoothing_eps: f64,
        learnable: bool,
        target: Option<&Bound<'_, PyAny>>,
    ) -> PyResult<Self> {
        let groups = block_sparsity_coerce_groups(groups)?;
        if weight <= 0.0 {
            return Err(PyValueError::new_err(format!(
                "BlockSparsityPenalty.weight must be > 0, got {weight}"
            )));
        }
        if n_eff <= 0 {
            return Err(PyValueError::new_err(format!(
                "BlockSparsityPenalty.n_eff must be > 0, got {n_eff}"
            )));
        }
        if smoothing_eps <= 0.0 {
            return Err(PyValueError::new_err(format!(
                "BlockSparsityPenalty.smoothing_eps must be > 0, got {smoothing_eps}"
            )));
        }
        Ok(Self {
            target: py_object_or_string_default(py, target, "t"),
            groups,
            weight,
            n_eff,
            smoothing_eps,
            learnable,
            weight_schedule: None,
        })
    }

    #[classattr]
    const KIND_TAG: &'static str = "block_sparsity";

    fn to_rust_descriptor(&self, py: Python<'_>) -> PyResult<PyObject> {
        let payload = PyDict::new(py);
        payload.set_item("kind", Self::KIND_TAG)?;
        payload.set_item("target", target_descriptor(py, self.target.bind(py))?)?;
        payload.set_item("groups", self.groups.clone())?;
        payload.set_item("weight", self.weight)?;
        payload.set_item("n_eff", self.n_eff)?;
        payload.set_item("smoothing_eps", self.smoothing_eps)?;
        payload.set_item("learnable", self.learnable)?;
        if let Some(schedule) = topk_weight_schedule_descriptor(py, &self.weight_schedule)? {
            payload.set_item("weight_schedule", schedule)?;
        }
        Ok(payload.into())
    }

    fn set_weight_schedule<'py>(
        mut slf: PyRefMut<'py, Self>,
        schedule: PyObject,
    ) -> PyRefMut<'py, Self> {
        slf.weight_schedule = Some(schedule);
        slf
    }

    fn __repr__(&self, py: Python<'_>) -> PyResult<String> {
        Ok(format!(
            "BlockSparsityPenalty(groups={:?}, weight={}, n_eff={}, smoothing_eps={}, learnable={}, target={}, weight_schedule={})",
            self.groups,
            self.weight,
            self.n_eff,
            self.smoothing_eps,
            self.learnable,
            py_repr(py, self.target.bind(py))?,
            match &self.weight_schedule {
                Some(schedule) => py_repr(py, schedule.bind(py))?,
                None => "None".to_string(),
            }
        ))
    }
}

fn block_sparsity_groups_type_error() -> PyErr {
    PyTypeError::new_err("BlockSparsityPenalty.groups must be a sequence of integer sequences")
}

fn block_sparsity_coerce_groups(groups: &Bound<'_, PyAny>) -> PyResult<Vec<Vec<usize>>> {
    let group_iter = groups
        .try_iter()
        .map_err(|_| block_sparsity_groups_type_error())?;
    let mut coerced = Vec::new();
    for group in group_iter {
        let group = group.map_err(|_| block_sparsity_groups_type_error())?;
        let axis_iter = group
            .try_iter()
            .map_err(|_| block_sparsity_groups_type_error())?;
        let mut axes = Vec::new();
        for axis in axis_iter {
            let axis = axis.map_err(|_| block_sparsity_groups_type_error())?;
            axes.push(
                axis.call_method0("__index__")
                    .and_then(|value| value.extract::<isize>())
                    .map_err(|_| block_sparsity_groups_type_error())?,
            );
        }
        coerced.push(axes);
    }
    if coerced.is_empty() {
        return Err(PyValueError::new_err(
            "BlockSparsityPenalty.groups must not be empty",
        ));
    }

    let mut seen = BTreeSet::new();
    let mut out = Vec::with_capacity(coerced.len());
    for (group_idx, group) in coerced.into_iter().enumerate() {
        if group.is_empty() {
            return Err(PyValueError::new_err(format!(
                "BlockSparsityPenalty.groups[{group_idx}] must not be empty"
            )));
        }
        let mut out_group = Vec::with_capacity(group.len());
        for axis in group {
            if axis < 0 {
                return Err(PyValueError::new_err(format!(
                    "BlockSparsityPenalty.groups entries must be non-negative; got {axis}"
                )));
            }
            let axis = axis as usize;
            if !seen.insert(axis) {
                return Err(PyValueError::new_err(format!(
                    "BlockSparsityPenalty.groups axis {axis} appears more than once"
                )));
            }
            out_group.push(axis);
        }
        out.push(out_group);
    }
    let max_axis = seen.iter().next_back().copied().unwrap_or(0);
    for axis in 0..=max_axis {
        if !seen.contains(&axis) {
            return Err(PyValueError::new_err(format!(
                "BlockSparsityPenalty.groups must partition contiguous axes from 0; missing axis {axis}"
            )));
        }
    }
    Ok(out)
}

#[pyclass(module = "gam_pyffi._rust", name = "SoftmaxAssignmentSparsityPenalty")]
struct SoftmaxAssignmentSparsityPenalty {
    #[pyo3(get, set)]
    target: PyObject,
    #[pyo3(get, set)]
    k_atoms: i64,
    #[pyo3(get, set)]
    temperature: f64,
    #[pyo3(get)]
    weight_schedule: Option<PyObject>,
}

#[pymethods]
impl SoftmaxAssignmentSparsityPenalty {
    #[new]
    #[pyo3(signature = (k_atoms, temperature = 1.0, *, target = None, weight_schedule = None))]
    fn new(
        py: Python<'_>,
        k_atoms: &Bound<'_, PyAny>,
        temperature: f64,
        target: Option<&Bound<'_, PyAny>>,
        weight_schedule: Option<PyObject>,
    ) -> PyResult<Self> {
        let builtins = PyModule::import(py, "builtins")?;
        let k_atoms = builtins
            .getattr("int")?
            .call1((k_atoms,))?
            .extract::<i64>()?;
        if k_atoms <= 0 {
            return Err(PyValueError::new_err(format!(
                "SoftmaxAssignmentSparsityPenalty.k_atoms must be > 0, got {k_atoms}"
            )));
        }
        if temperature <= 0.0 {
            return Err(PyValueError::new_err(format!(
                "SoftmaxAssignmentSparsityPenalty.temperature must be > 0, got {temperature}"
            )));
        }
        Ok(Self {
            target: py_object_or_string_default(py, target, "t"),
            k_atoms,
            temperature,
            weight_schedule,
        })
    }

    #[classattr]
    const KIND_TAG: &'static str = "softmax_assignment_sparsity";

    fn to_rust_descriptor(&self, py: Python<'_>) -> PyResult<PyObject> {
        let payload = PyDict::new(py);
        payload.set_item("kind", Self::KIND_TAG)?;
        payload.set_item("target", target_descriptor(py, self.target.bind(py))?)?;
        payload.set_item("k_atoms", self.k_atoms)?;
        payload.set_item("temperature", self.temperature)?;
        if let Some(schedule) = topk_weight_schedule_descriptor(py, &self.weight_schedule)? {
            payload.set_item("weight_schedule", schedule)?;
        }
        Ok(payload.into())
    }

    fn _to_rust_payload(&self, py: Python<'_>) -> PyResult<PyObject> {
        self.to_rust_descriptor(py)
    }

    fn set_weight_schedule<'py>(
        mut slf: PyRefMut<'py, Self>,
        schedule: PyObject,
    ) -> PyRefMut<'py, Self> {
        slf.weight_schedule = Some(schedule);
        slf
    }

    fn __repr__(&self, py: Python<'_>) -> PyResult<String> {
        Ok(format!(
            "SoftmaxAssignmentSparsityPenalty(target={}, k_atoms={}, temperature={}, weight_schedule={})",
            py_repr(py, self.target.bind(py))?,
            self.k_atoms,
            self.temperature,
            match &self.weight_schedule {
                Some(schedule) => py_repr(py, schedule.bind(py))?,
                None => "None".to_string(),
            }
        ))
    }
}

#[pyclass(module = "gam_pyffi._rust", name = "IsometryPenalty")]
struct IsometryPenalty {
    #[pyo3(get, set)]
    target: PyObject,
    #[pyo3(get, set)]
    weight: PyObject,
    #[pyo3(get)]
    weight_schedule: Option<PyObject>,
}

#[pymethods]
impl IsometryPenalty {
    #[new]
    #[pyo3(signature = (weight = None, *, target = None, weight_schedule = None))]
    fn new(
        py: Python<'_>,
        weight: Option<&Bound<'_, PyAny>>,
        target: Option<&Bound<'_, PyAny>>,
        weight_schedule: Option<PyObject>,
    ) -> PyResult<Self> {
        let weight = py_object_or_string_default(py, weight, "auto");
        let target = py_object_or_string_default(py, target, "t");
        validate_sparsity_weight(py, weight.bind(py), "IsometryPenalty")?;
        Ok(Self {
            target,
            weight,
            weight_schedule,
        })
    }

    #[classattr]
    const KIND_TAG: &'static str = "isometry";

    fn to_rust_descriptor(&self, py: Python<'_>) -> PyResult<PyObject> {
        let payload = PyDict::new(py);
        payload.set_item("kind", Self::KIND_TAG)?;
        payload.set_item("target", target_descriptor(py, self.target.bind(py))?)?;
        payload.set_item("weight", self.weight.bind(py))?;
        if let Some(schedule) = &self.weight_schedule {
            payload.set_item("weight_schedule", schedule.bind(py))?;
        }
        Ok(payload.into())
    }

    fn set_weight_schedule<'py>(
        mut slf: PyRefMut<'py, Self>,
        schedule: PyObject,
    ) -> PyRefMut<'py, Self> {
        slf.weight_schedule = Some(schedule);
        slf
    }

    fn __repr__(&self, py: Python<'_>) -> PyResult<String> {
        Ok(format!(
            "IsometryPenalty(weight={}, target={}, weight_schedule={})",
            py_repr(py, self.weight.bind(py))?,
            py_repr(py, self.target.bind(py))?,
            match &self.weight_schedule {
                Some(schedule) => py_repr(py, schedule.bind(py))?,
                None => "None".to_string(),
            }
        ))
    }
}

#[pyclass(module = "gam_pyffi._rust", name = "IBPAssignmentPenalty")]
struct PyIBPAssignmentPenalty {
    #[pyo3(get, set)]
    target: PyObject,
    #[pyo3(get, set)]
    k_max: i64,
    #[pyo3(get, set)]
    alpha: f64,
    #[pyo3(get, set)]
    tau: f64,
    #[pyo3(get, set)]
    learnable: bool,
    #[pyo3(get, set)]
    temperature_schedule: Option<PyObject>,
    #[pyo3(get)]
    weight_schedule: Option<PyObject>,
}

#[pymethods]
impl PyIBPAssignmentPenalty {
    #[new]
    #[pyo3(signature = (k_max, alpha = 1.0, tau = 1.0, learnable = false, *, target = None, temperature_schedule = None, weight_schedule = None))]
    fn new(
        py: Python<'_>,
        k_max: &Bound<'_, PyAny>,
        alpha: f64,
        tau: f64,
        learnable: bool,
        target: Option<&Bound<'_, PyAny>>,
        temperature_schedule: Option<PyObject>,
        weight_schedule: Option<PyObject>,
    ) -> PyResult<Self> {
        let builtins = PyModule::import(py, "builtins")?;
        let k_max = builtins.getattr("int")?.call1((k_max,))?.extract::<i64>()?;
        if k_max <= 0 {
            return Err(PyValueError::new_err(format!(
                "IBPAssignmentPenalty.k_max must be > 0, got {k_max}"
            )));
        }
        if alpha <= 0.0 {
            return Err(PyValueError::new_err(format!(
                "IBPAssignmentPenalty.alpha must be > 0, got {alpha}"
            )));
        }
        if tau <= 0.0 {
            return Err(PyValueError::new_err(format!(
                "IBPAssignmentPenalty.tau must be > 0, got {tau}"
            )));
        }
        Ok(Self {
            target: py_object_or_string_default(py, target, "t"),
            k_max,
            alpha,
            tau,
            learnable,
            temperature_schedule,
            weight_schedule,
        })
    }

    #[classattr]
    const KIND_TAG: &'static str = "ibp_assignment";

    fn to_rust_descriptor(&self, py: Python<'_>) -> PyResult<PyObject> {
        let payload = PyDict::new(py);
        payload.set_item("kind", Self::KIND_TAG)?;
        payload.set_item("target", target_descriptor(py, self.target.bind(py))?)?;
        payload.set_item("k_max", self.k_max)?;
        payload.set_item("alpha", self.alpha)?;
        payload.set_item("tau", self.tau)?;
        payload.set_item("learnable", self.learnable)?;
        if let Some(schedule) = &self.temperature_schedule {
            payload.set_item("temperature_schedule", schedule.bind(py))?;
        }
        if let Some(schedule) = &self.weight_schedule {
            payload.set_item("weight_schedule", schedule.bind(py))?;
        }
        Ok(payload.into())
    }

    fn _to_rust_payload(&self, py: Python<'_>) -> PyResult<PyObject> {
        self.to_rust_descriptor(py)
    }

    fn set_weight_schedule<'py>(
        mut slf: PyRefMut<'py, Self>,
        schedule: PyObject,
    ) -> PyRefMut<'py, Self> {
        slf.weight_schedule = Some(schedule);
        slf
    }

    fn __repr__(&self, py: Python<'_>) -> PyResult<String> {
        Ok(format!(
            "IBPAssignmentPenalty(target={}, k_max={}, alpha={}, tau={}, learnable={}, temperature_schedule={}, weight_schedule={})",
            py_repr(py, self.target.bind(py))?,
            self.k_max,
            self.alpha,
            self.tau,
            self.learnable,
            match &self.temperature_schedule {
                Some(schedule) => py_repr(py, schedule.bind(py))?,
                None => "None".to_string(),
            },
            match &self.weight_schedule {
                Some(schedule) => py_repr(py, schedule.bind(py))?,
                None => "None".to_string(),
            }
        ))
    }
}

fn total_variation_difference_op_type_error() -> PyErr {
    PyTypeError::new_err(
        "TotalVariationPenalty.difference_op must be 'forward_1d' or a sequence of (from_row, to_row) edges",
    )
}

fn total_variation_coerce_edges(edges: &Bound<'_, PyAny>) -> PyResult<Vec<(i64, i64)>> {
    let edge_iter = edges
        .try_iter()
        .map_err(|_| total_variation_difference_op_type_error())?;
    let mut out = Vec::new();
    for edge in edge_iter {
        let edge = edge.map_err(|_| total_variation_difference_op_type_error())?;
        let mut endpoints = edge
            .try_iter()
            .map_err(|_| total_variation_difference_op_type_error())?;
        let a = endpoints
            .next()
            .ok_or_else(total_variation_difference_op_type_error)?
            .map_err(|_| total_variation_difference_op_type_error())?
            .call_method0("__index__")
            .and_then(|value| value.extract::<i64>())
            .map_err(|_| total_variation_difference_op_type_error())?;
        let b = endpoints
            .next()
            .ok_or_else(total_variation_difference_op_type_error)?
            .map_err(|_| total_variation_difference_op_type_error())?
            .call_method0("__index__")
            .and_then(|value| value.extract::<i64>())
            .map_err(|_| total_variation_difference_op_type_error())?;
        if endpoints.next().is_some() {
            return Err(total_variation_difference_op_type_error());
        }
        out.push((a, b));
    }
    Ok(out)
}

#[pyclass(module = "gam_pyffi._rust", name = "TotalVariationPenalty")]
struct TotalVariationPenalty {
    #[pyo3(get, set)]
    target: PyObject,
    #[pyo3(get, set)]
    weight: f64,
    #[pyo3(get, set)]
    n_eff: i64,
    #[pyo3(get, set)]
    difference_op: PyObject,
    #[pyo3(get, set)]
    smoothing_eps: f64,
    #[pyo3(get, set)]
    learnable: bool,
    #[pyo3(get, set)]
    _edges: Option<Vec<(i64, i64)>>,
    #[pyo3(get)]
    weight_schedule: Option<PyObject>,
}

#[pymethods]
impl TotalVariationPenalty {
    #[new]
    #[pyo3(signature = (weight, n_eff, difference_op = None, smoothing_eps = 1.0e-6, learnable = false, *, target = None))]
    fn new(
        py: Python<'_>,
        weight: f64,
        n_eff: i64,
        difference_op: Option<&Bound<'_, PyAny>>,
        smoothing_eps: f64,
        learnable: bool,
        target: Option<&Bound<'_, PyAny>>,
    ) -> PyResult<Self> {
        if weight <= 0.0 {
            return Err(PyValueError::new_err(format!(
                "TotalVariationPenalty.weight must be > 0, got {weight}"
            )));
        }
        if n_eff <= 0 {
            return Err(PyValueError::new_err(format!(
                "TotalVariationPenalty.n_eff must be > 0, got {n_eff}"
            )));
        }
        if smoothing_eps <= 0.0 {
            return Err(PyValueError::new_err(format!(
                "TotalVariationPenalty.smoothing_eps must be > 0, got {smoothing_eps}"
            )));
        }

        let difference_op = py_object_or_string_default(py, difference_op, "forward_1d");
        let target = py_object_or_string_default(py, target, "t");
        let bound_difference_op = difference_op.bind(py);
        let edges = if let Ok(op) = bound_difference_op.extract::<String>() {
            if op != "forward_1d" {
                return Err(PyValueError::new_err(format!(
                    "TotalVariationPenalty.difference_op string must be 'forward_1d', got {op:?}"
                )));
            }
            None
        } else {
            let parsed_edges = total_variation_coerce_edges(bound_difference_op)?;
            if parsed_edges.is_empty() {
                return Err(PyValueError::new_err(
                    "TotalVariationPenalty graph edges must not be empty",
                ));
            }
            for (a, b) in &parsed_edges {
                if *a < 0 || *b < 0 || *a >= n_eff || *b >= n_eff {
                    return Err(PyValueError::new_err(format!(
                        "TotalVariationPenalty graph edges must be within [0, n_eff); got ({a}, {b}) for n_eff={n_eff}"
                    )));
                }
                if a == b {
                    return Err(PyValueError::new_err(format!(
                        "TotalVariationPenalty graph edge ({a}, {b}) is self-referential"
                    )));
                }
            }
            Some(parsed_edges)
        };

        Ok(Self {
            target,
            weight,
            n_eff,
            difference_op,
            smoothing_eps,
            learnable,
            _edges: edges,
            weight_schedule: None,
        })
    }

    #[classattr]
    const KIND_TAG: &'static str = "total_variation";

    fn to_rust_descriptor(&self, py: Python<'_>) -> PyResult<PyObject> {
        let payload = PyDict::new(py);
        payload.set_item("kind", Self::KIND_TAG)?;
        payload.set_item("target", target_descriptor(py, self.target.bind(py))?)?;
        payload.set_item("weight", self.weight)?;
        payload.set_item("n_eff", self.n_eff)?;
        payload.set_item("smoothing_eps", self.smoothing_eps)?;
        payload.set_item("learnable", self.learnable)?;
        if let Some(edges) = &self._edges {
            payload.set_item("difference_op", "graph_edges")?;
            payload.set_item("edges", edges)?;
        } else {
            payload.set_item("difference_op", "forward_1d")?;
        }
        if let Some(schedule) = topk_weight_schedule_descriptor(py, &self.weight_schedule)? {
            payload.set_item("weight_schedule", schedule)?;
        }
        Ok(payload.into())
    }

    fn set_weight_schedule<'py>(
        mut slf: PyRefMut<'py, Self>,
        schedule: PyObject,
    ) -> PyRefMut<'py, Self> {
        slf.weight_schedule = Some(schedule);
        slf
    }

    fn __repr__(&self, py: Python<'_>) -> PyResult<String> {
        Ok(format!(
            "TotalVariationPenalty(weight={}, n_eff={}, difference_op={}, smoothing_eps={}, learnable={}, target={}, weight_schedule={})",
            self.weight,
            self.n_eff,
            py_repr(py, self.difference_op.bind(py))?,
            self.smoothing_eps,
            self.learnable,
            py_repr(py, self.target.bind(py))?,
            match &self.weight_schedule {
                Some(schedule) => py_repr(py, schedule.bind(py))?,
                None => "None".to_string(),
            }
        ))
    }
}

fn validate_parametric_aux_conditional_prior(
    aux: &Bound<'_, PyAny>,
    alpha_init: &Bound<'_, PyAny>,
    beta_init: &Bound<'_, PyAny>,
    mu_init: &Bound<'_, PyAny>,
    weight: f64,
    n_eff: i64,
) -> PyResult<()> {
    if weight <= 0.0 {
        return Err(PyValueError::new_err(format!(
            "ParametricAuxConditionalPriorPenalty.weight must be > 0, got {weight}"
        )));
    }
    if n_eff <= 0 {
        return Err(PyValueError::new_err(format!(
            "ParametricAuxConditionalPriorPenalty.n_eff must be > 0, got {n_eff}"
        )));
    }

    let aux_array = aux.extract::<PyReadonlyArrayDyn<'_, f64>>()?;
    let alpha_array = alpha_init.extract::<PyReadonlyArrayDyn<'_, f64>>()?;
    let beta_array = beta_init.extract::<PyReadonlyArrayDyn<'_, f64>>()?;
    let mu_array = mu_init.extract::<PyReadonlyArrayDyn<'_, f64>>()?;
    let aux_view = aux_array.as_array();
    let alpha_view = alpha_array.as_array();
    let beta_view = beta_array.as_array();
    let mu_view = mu_array.as_array();

    if aux_view.ndim() != 2 {
        return Err(PyValueError::new_err(format!(
            "ParametricAuxConditionalPriorPenalty.aux must have shape (N, du), got ndim={}",
            aux_view.ndim()
        )));
    }
    if alpha_view.ndim() != 1 {
        return Err(PyValueError::new_err(format!(
            "ParametricAuxConditionalPriorPenalty.alpha_init must have shape (d,), got ndim={}",
            alpha_view.ndim()
        )));
    }
    if beta_view.ndim() != 1 {
        return Err(PyValueError::new_err(format!(
            "ParametricAuxConditionalPriorPenalty.beta_init must have shape (d,), got ndim={}",
            beta_view.ndim()
        )));
    }
    if mu_view.ndim() != 2 {
        return Err(PyValueError::new_err(format!(
            "ParametricAuxConditionalPriorPenalty.mu_init must have shape (d, du), got ndim={}",
            mu_view.ndim()
        )));
    }

    let aux_shape = aux_view.shape();
    let alpha_shape = alpha_view.shape();
    let beta_shape = beta_view.shape();
    let mu_shape = mu_view.shape();
    let n_obs = aux_shape[0];
    let aux_dim = aux_shape[1];
    let latent_dim = alpha_shape[0];
    if n_obs as i64 != n_eff {
        return Err(PyValueError::new_err(format!(
            "ParametricAuxConditionalPriorPenalty.aux first dimension must equal n_eff={n_eff}, got {n_obs}"
        )));
    }
    if aux_dim == 0 {
        return Err(PyValueError::new_err(
            "ParametricAuxConditionalPriorPenalty.aux must have du > 0",
        ));
    }
    if latent_dim == 0 {
        return Err(PyValueError::new_err(
            "ParametricAuxConditionalPriorPenalty.alpha_init must be non-empty",
        ));
    }
    if beta_shape[0] != latent_dim {
        return Err(PyValueError::new_err(format!(
            "ParametricAuxConditionalPriorPenalty.beta_init shape must match alpha_init shape ({latent_dim},), got ({},)",
            beta_shape[0]
        )));
    }
    if mu_shape[0] != latent_dim || mu_shape[1] != aux_dim {
        return Err(PyValueError::new_err(format!(
            "ParametricAuxConditionalPriorPenalty.mu_init must have shape ({latent_dim}, {aux_dim}), got ({}, {})",
            mu_shape[0], mu_shape[1]
        )));
    }
    if !aux_view.iter().all(|value| value.is_finite()) {
        return Err(PyValueError::new_err(
            "ParametricAuxConditionalPriorPenalty.aux must be finite",
        ));
    }
    if !alpha_view.iter().all(|value| value.is_finite()) {
        return Err(PyValueError::new_err(
            "ParametricAuxConditionalPriorPenalty.alpha_init must be finite",
        ));
    }
    if !beta_view.iter().all(|value| value.is_finite()) {
        return Err(PyValueError::new_err(
            "ParametricAuxConditionalPriorPenalty.beta_init must be finite",
        ));
    }
    if !mu_view.iter().all(|value| value.is_finite()) {
        return Err(PyValueError::new_err(
            "ParametricAuxConditionalPriorPenalty.mu_init must be finite",
        ));
    }
    if !alpha_view.iter().all(|value| *value > 0.0) {
        return Err(PyValueError::new_err(
            "ParametricAuxConditionalPriorPenalty.alpha_init must be > 0",
        ));
    }
    if !beta_view.iter().all(|value| *value > 0.0) {
        return Err(PyValueError::new_err(
            "ParametricAuxConditionalPriorPenalty.beta_init must be > 0",
        ));
    }
    Ok(())
}

// Softplus⁻¹ reparameterization helper: the single definition lives in the core
// `gam-sae` crate next to the forward softplus link (`terms::sae::assignment`),
// so no numeric policy lives in this FFI shim (SPEC: pyffi thin, single source
// of truth). Aliased to the original local name so the call sites below stay
// unchanged.
use gam::terms::sae::assignment::inverse_softplus as inverse_softplus_scalar;

#[pyclass(
    module = "gam_pyffi._rust",
    name = "ParametricAuxConditionalPriorPenalty"
)]
struct ParametricAuxConditionalPriorPenalty {
    #[pyo3(get, set)]
    target: PyObject,
    #[pyo3(get, set)]
    aux: PyObject,
    #[pyo3(get, set)]
    alpha_init: PyObject,
    #[pyo3(get, set)]
    beta_init: PyObject,
    #[pyo3(get, set)]
    mu_init: PyObject,
    #[pyo3(get, set)]
    weight: f64,
    #[pyo3(get, set)]
    n_eff: i64,
    #[pyo3(get, set)]
    learnable: bool,
    #[pyo3(get)]
    weight_schedule: Option<PyObject>,
}

#[pymethods]
impl ParametricAuxConditionalPriorPenalty {
    #[new]
    #[pyo3(signature = (aux, alpha_init, beta_init, mu_init, weight, n_eff, learnable = false, *, target = None))]
    fn new(
        py: Python<'_>,
        aux: &Bound<'_, PyAny>,
        alpha_init: &Bound<'_, PyAny>,
        beta_init: &Bound<'_, PyAny>,
        mu_init: &Bound<'_, PyAny>,
        weight: f64,
        n_eff: i64,
        learnable: bool,
        target: Option<&Bound<'_, PyAny>>,
    ) -> PyResult<Self> {
        let aux = aux_conditional_prior_float_array(py, aux)?;
        let alpha_init = aux_conditional_prior_float_array(py, alpha_init)?;
        let beta_init = aux_conditional_prior_float_array(py, beta_init)?;
        let mu_init = aux_conditional_prior_float_array(py, mu_init)?;
        validate_parametric_aux_conditional_prior(
            &aux,
            &alpha_init,
            &beta_init,
            &mu_init,
            weight,
            n_eff,
        )?;
        Ok(Self {
            target: py_object_or_string_default(py, target, "t"),
            aux: aux.unbind(),
            alpha_init: alpha_init.unbind(),
            beta_init: beta_init.unbind(),
            mu_init: mu_init.unbind(),
            weight,
            n_eff,
            learnable,
            weight_schedule: None,
        })
    }

    #[classattr]
    const KIND_TAG: &'static str = "parametric_aux_conditional_prior";

    fn to_rust_descriptor(&self, py: Python<'_>) -> PyResult<PyObject> {
        let aux_array = self.aux.bind(py).extract::<PyReadonlyArrayDyn<'_, f64>>()?;
        let alpha_array = self
            .alpha_init
            .bind(py)
            .extract::<PyReadonlyArrayDyn<'_, f64>>()?;
        let beta_array = self
            .beta_init
            .bind(py)
            .extract::<PyReadonlyArrayDyn<'_, f64>>()?;
        let mu_array = self
            .mu_init
            .bind(py)
            .extract::<PyReadonlyArrayDyn<'_, f64>>()?;
        let aux_view = aux_array.as_array();
        let alpha_view = alpha_array.as_array();
        let beta_view = beta_array.as_array();
        let mu_view = mu_array.as_array();

        let payload = PyDict::new(py);
        payload.set_item("kind", Self::KIND_TAG)?;
        payload.set_item("target", target_descriptor(py, self.target.bind(py))?)?;
        payload.set_item(
            "aux",
            PyList::new(py, aux_view.iter().copied().collect::<Vec<_>>())?,
        )?;
        payload.set_item("aux_shape", PyList::new(py, aux_view.shape())?)?;
        payload.set_item(
            "log_alpha",
            PyList::new(
                py,
                alpha_view
                    .iter()
                    .map(|value| value.ln())
                    .collect::<Vec<_>>(),
            )?,
        )?;
        payload.set_item(
            "raw_beta",
            PyList::new(
                py,
                beta_view
                    .iter()
                    .map(|value| inverse_softplus_scalar(*value))
                    .collect::<Vec<_>>(),
            )?,
        )?;
        payload.set_item(
            "mu",
            PyList::new(py, mu_view.iter().copied().collect::<Vec<_>>())?,
        )?;
        payload.set_item("mu_shape", PyList::new(py, mu_view.shape())?)?;
        payload.set_item("weight", self.weight)?;
        payload.set_item("n_eff", self.n_eff)?;
        payload.set_item("learnable", self.learnable)?;
        if let Some(schedule) = topk_weight_schedule_descriptor(py, &self.weight_schedule)? {
            payload.set_item("weight_schedule", schedule)?;
        }
        Ok(payload.into())
    }

    fn set_weight_schedule<'py>(
        mut slf: PyRefMut<'py, Self>,
        schedule: PyObject,
    ) -> PyRefMut<'py, Self> {
        slf.weight_schedule = Some(schedule);
        slf
    }

    fn __repr__(&self, py: Python<'_>) -> PyResult<String> {
        Ok(format!(
            "ParametricAuxConditionalPriorPenalty(aux={}, alpha_init={}, beta_init={}, mu_init={}, weight={}, n_eff={}, learnable={}, target={}, weight_schedule={})",
            py_repr(py, self.aux.bind(py))?,
            py_repr(py, self.alpha_init.bind(py))?,
            py_repr(py, self.beta_init.bind(py))?,
            py_repr(py, self.mu_init.bind(py))?,
            self.weight,
            self.n_eff,
            self.learnable,
            py_repr(py, self.target.bind(py))?,
            match &self.weight_schedule {
                Some(schedule) => py_repr(py, schedule.bind(py))?,
                None => "None".to_string(),
            }
        ))
    }

    /// Evaluate the parametric iVAE-style row-precision prior value and
    /// gradient at latent block `t` of shape `(n_eff, latent_dim)`. Returns
    /// `(value, grad)` with `grad` having the same shape as `t`.
    #[pyo3(signature = (t))]
    fn value_grad<'py>(
        &self,
        py: Python<'py>,
        t: PyReadonlyArray2<'py, f64>,
    ) -> PyResult<(f64, Py<PyArray2<f64>>)> {
        let view = t.as_array();
        let (n_rows, latent_dim) = view.dim();
        if n_rows as i64 != self.n_eff {
            return Err(PyValueError::new_err(format!(
                "ParametricAuxConditionalPriorPenalty.value_grad: t first dim {n_rows} must match n_eff={}",
                self.n_eff
            )));
        }
        if latent_dim == 0 {
            return Err(PyValueError::new_err(
                "ParametricAuxConditionalPriorPenalty.value_grad: t must have latent_dim > 0",
            ));
        }
        let aux_array = self.aux.bind(py).extract::<PyReadonlyArray2<'_, f64>>()?;
        let aux_owned: Array2<f64> = aux_array.as_array().to_owned();
        let alpha_array = self
            .alpha_init
            .bind(py)
            .extract::<PyReadonlyArray1<'_, f64>>()?;
        let beta_array = self
            .beta_init
            .bind(py)
            .extract::<PyReadonlyArray1<'_, f64>>()?;
        let mu_array = self
            .mu_init
            .bind(py)
            .extract::<PyReadonlyArray2<'_, f64>>()?;
        let log_alpha: Array1<f64> = alpha_array.as_array().iter().map(|v| v.ln()).collect();
        let raw_beta: Array1<f64> = beta_array
            .as_array()
            .iter()
            .map(|v| inverse_softplus_scalar(*v))
            .collect();
        let mu_owned: Array2<f64> = mu_array.as_array().to_owned();
        let slice = PsiSlice::full(n_rows * latent_dim, Some(latent_dim));
        let penalty = ParametricRowPrecisionPriorPenalty::new(
            slice,
            aux_owned,
            log_alpha,
            raw_beta,
            mu_owned,
            self.weight,
            n_rows,
            false,
        )
        .map_err(py_value_error)?;
        let flat: Array1<f64> = view.iter().copied().collect();
        let rho = Array1::<f64>::zeros(0);
        let value = penalty.value(flat.view(), rho.view());
        let grad_flat = penalty.grad_target(flat.view(), rho.view());
        let grad = grad_flat
            .into_shape_with_order((n_rows, latent_dim))
            .map_err(|err| {
                py_value_error(format!(
                    "ParametricAuxConditionalPriorPenalty.value_grad: gradient reshape failed: {err}"
                ))
            })?;
        Ok((value, grad.into_pyarray(py).unbind()))
    }
}

#[pyclass(module = "gam_pyffi._rust", name = "OrthogonalityPenalty")]
struct OrthogonalityPenalty {
    #[pyo3(get, set)]
    target: PyObject,
    #[pyo3(get, set)]
    weight: f64,
    #[pyo3(get, set)]
    n_eff: i64,
    #[pyo3(get, set)]
    learnable: bool,
    #[pyo3(get)]
    weight_schedule: Option<PyObject>,
}

#[pymethods]
impl OrthogonalityPenalty {
    #[new]
    #[pyo3(signature = (weight, n_eff, learnable = false, *, target = None))]
    fn new(
        py: Python<'_>,
        weight: f64,
        n_eff: i64,
        learnable: bool,
        target: Option<&Bound<'_, PyAny>>,
    ) -> PyResult<Self> {
        if weight <= 0.0 {
            return Err(PyValueError::new_err(format!(
                "OrthogonalityPenalty.weight must be > 0, got {weight:?}"
            )));
        }
        if n_eff <= 0 {
            return Err(PyValueError::new_err(format!(
                "OrthogonalityPenalty.n_eff must be > 0, got {n_eff}"
            )));
        }
        Ok(Self {
            target: py_object_or_string_default(py, target, "t"),
            weight,
            n_eff,
            learnable,
            weight_schedule: None,
        })
    }

    #[classattr]
    const KIND_TAG: &'static str = "orthogonality";

    fn to_rust_descriptor(&self, py: Python<'_>) -> PyResult<PyObject> {
        let payload = PyDict::new(py);
        payload.set_item("kind", Self::KIND_TAG)?;
        payload.set_item("target", target_descriptor(py, self.target.bind(py))?)?;
        payload.set_item("weight", self.weight)?;
        payload.set_item("n_eff", self.n_eff)?;
        payload.set_item("learnable", self.learnable)?;
        if let Some(schedule) = &self.weight_schedule {
            payload.set_item("weight_schedule", schedule.bind(py))?;
        }
        Ok(payload.into())
    }

    fn set_weight_schedule<'py>(
        mut slf: PyRefMut<'py, Self>,
        schedule: PyObject,
    ) -> PyRefMut<'py, Self> {
        slf.weight_schedule = Some(schedule);
        slf
    }

    fn __repr__(&self, py: Python<'_>) -> PyResult<String> {
        Ok(format!(
            "OrthogonalityPenalty(weight={:?}, n_eff={}, learnable={}, target={}, weight_schedule={})",
            self.weight,
            self.n_eff,
            if self.learnable { "True" } else { "False" },
            py_repr(py, self.target.bind(py))?,
            match &self.weight_schedule {
                Some(schedule) => py_repr(py, schedule.bind(py))?,
                None => "None".to_string(),
            }
        ))
    }
}

#[pyclass(module = "gam_pyffi._rust", name = "ScadMcpPenalty")]
struct ScadMcpPenalty {
    #[pyo3(get, set)]
    target: PyObject,
    #[pyo3(get, set)]
    weight: f64,
    #[pyo3(get, set)]
    n_eff: i64,
    #[pyo3(get, set)]
    gamma: f64,
    #[pyo3(get, set)]
    variant: String,
    #[pyo3(get, set)]
    smoothing_eps: f64,
    #[pyo3(get, set)]
    learnable: bool,
    #[pyo3(get)]
    weight_schedule: Option<PyObject>,
}

#[pymethods]
impl ScadMcpPenalty {
    #[new]
    #[pyo3(signature = (weight, n_eff, gamma = None, variant = "mcp".to_string(), smoothing_eps = 1.0e-6, learnable = false, *, target = None))]
    fn new(
        py: Python<'_>,
        weight: f64,
        n_eff: i64,
        gamma: Option<f64>,
        variant: String,
        smoothing_eps: f64,
        learnable: bool,
        target: Option<&Bound<'_, PyAny>>,
    ) -> PyResult<Self> {
        let gamma = match gamma {
            Some(value) => value,
            None => match variant.as_str() {
                "mcp" => 2.5,
                "scad" => 3.7,
                _ => {
                    return Err(PyValueError::new_err(format!(
                        "ScadMcpPenalty.variant must be 'mcp' or 'scad', got {variant:?}"
                    )));
                }
            },
        };
        let penalty = Self {
            target: py_object_or_string_default(py, target, "t"),
            weight,
            n_eff,
            gamma,
            variant,
            smoothing_eps,
            learnable,
            weight_schedule: None,
        };
        penalty.validate()?;
        Ok(penalty)
    }

    #[classattr]
    const KIND_TAG: &'static str = "scad_mcp";

    fn to_rust_descriptor(&self, py: Python<'_>) -> PyResult<PyObject> {
        let payload = PyDict::new(py);
        payload.set_item("kind", Self::KIND_TAG)?;
        payload.set_item("target", target_descriptor(py, self.target.bind(py))?)?;
        payload.set_item("weight", self.weight)?;
        payload.set_item("n_eff", self.n_eff)?;
        payload.set_item("gamma", self.gamma)?;
        payload.set_item("variant", &self.variant)?;
        payload.set_item("smoothing_eps", self.smoothing_eps)?;
        payload.set_item("learnable", self.learnable)?;
        if let Some(schedule) = &self.weight_schedule {
            payload.set_item("weight_schedule", schedule.bind(py))?;
        }
        Ok(payload.into())
    }

    fn _to_rust_payload(&self, py: Python<'_>) -> PyResult<PyObject> {
        self.to_rust_descriptor(py)
    }

    fn set_weight_schedule<'py>(
        mut slf: PyRefMut<'py, Self>,
        schedule: PyObject,
    ) -> PyRefMut<'py, Self> {
        slf.weight_schedule = Some(schedule);
        slf
    }

    fn __repr__(&self, py: Python<'_>) -> PyResult<String> {
        Ok(format!(
            "ScadMcpPenalty(weight={}, n_eff={}, gamma={}, variant={}, smoothing_eps={}, learnable={}, target={}, weight_schedule={})",
            self.weight,
            self.n_eff,
            self.gamma,
            self.variant.as_str(),
            self.smoothing_eps,
            self.learnable,
            py_repr(py, self.target.bind(py))?,
            match &self.weight_schedule {
                Some(schedule) => py_repr(py, schedule.bind(py))?,
                None => "None".to_string(),
            }
        ))
    }
}

impl ScadMcpPenalty {
    fn validate(&self) -> PyResult<()> {
        if !self.weight.is_finite() || self.weight <= 0.0 {
            return Err(PyValueError::new_err(format!(
                "ScadMcpPenalty.weight must be > 0, got {}",
                self.weight
            )));
        }
        if self.n_eff <= 0 {
            return Err(PyValueError::new_err(format!(
                "ScadMcpPenalty.n_eff must be > 0, got {}",
                self.n_eff
            )));
        }
        if !self.gamma.is_finite() || self.gamma <= 1.0 {
            return Err(PyValueError::new_err(format!(
                "ScadMcpPenalty.gamma must be > 1, got {}",
                self.gamma
            )));
        }
        if !matches!(self.variant.as_str(), "mcp" | "scad") {
            return Err(PyValueError::new_err(format!(
                "ScadMcpPenalty.variant must be 'mcp' or 'scad', got {:?}",
                self.variant
            )));
        }
        if !self.smoothing_eps.is_finite() || self.smoothing_eps <= 0.0 {
            return Err(PyValueError::new_err(format!(
                "ScadMcpPenalty.smoothing_eps must be > 0, got {}",
                self.smoothing_eps
            )));
        }
        Ok(())
    }
}

#[pyclass(module = "gam_pyffi._rust", name = "IvaeRidgeMeanGauge")]
struct IvaeRidgeMeanGauge {
    #[pyo3(get, set)]
    target: PyObject,
    #[pyo3(get, set)]
    aux: PyObject,
    #[pyo3(get, set)]
    ridge_eps: f64,
    #[pyo3(get, set)]
    weight: f64,
    #[pyo3(get, set)]
    n_eff: i64,
    #[pyo3(get, set)]
    learnable: bool,
    #[pyo3(get)]
    weight_schedule: Option<PyObject>,
}

#[pymethods]
impl IvaeRidgeMeanGauge {
    #[new]
    #[pyo3(signature = (aux, weight, n_eff, ridge_eps = 1.0e-6, learnable = false, *, target = None))]
    fn new(
        py: Python<'_>,
        aux: &Bound<'_, PyAny>,
        weight: &Bound<'_, PyAny>,
        n_eff: &Bound<'_, PyAny>,
        ridge_eps: f64,
        learnable: bool,
        target: Option<&Bound<'_, PyAny>>,
    ) -> PyResult<Self> {
        let builtins = py.import("builtins")?;
        let weight = builtins
            .getattr("float")?
            .call1((weight,))?
            .extract::<f64>()?;
        let n_eff = builtins.getattr("int")?.call1((n_eff,))?.extract::<i64>()?;
        let aux = aux_conditional_prior_float_array(py, aux)?;
        validate_ivae_ridge_mean_gauge_aux(&aux, weight, n_eff, ridge_eps)?;
        Ok(Self {
            target: py_object_or_string_default(py, target, "t"),
            aux: aux.unbind(),
            ridge_eps,
            weight,
            n_eff,
            learnable,
            weight_schedule: None,
        })
    }

    #[classattr]
    const KIND_TAG: &'static str = "ivae_ridge_mean_gauge";

    fn to_rust_descriptor(&self, py: Python<'_>) -> PyResult<PyObject> {
        let payload = PyDict::new(py);
        payload.set_item("kind", Self::KIND_TAG)?;
        payload.set_item("target", target_descriptor(py, self.target.bind(py))?)?;

        let aux = aux_conditional_prior_float_array(py, self.aux.bind(py))?;
        let array = aux.extract::<PyReadonlyArrayDyn<'_, f64>>()?;
        let view = array.as_array();
        payload.set_item(
            "aux",
            PyList::new(py, view.iter().copied().collect::<Vec<_>>())?,
        )?;
        payload.set_item("aux_shape", PyList::new(py, view.shape())?)?;
        payload.set_item("ridge_eps", self.ridge_eps)?;
        payload.set_item("weight", self.weight)?;
        payload.set_item("n_eff", self.n_eff)?;
        payload.set_item("learnable", self.learnable)?;
        if let Some(schedule) = topk_weight_schedule_descriptor(py, &self.weight_schedule)? {
            payload.set_item("weight_schedule", schedule)?;
        }
        Ok(payload.into())
    }

    fn set_weight_schedule<'py>(
        mut slf: PyRefMut<'py, Self>,
        schedule: PyObject,
    ) -> PyRefMut<'py, Self> {
        slf.weight_schedule = Some(schedule);
        slf
    }

    fn __repr__(&self, py: Python<'_>) -> PyResult<String> {
        Ok(format!(
            "IvaeRidgeMeanGauge(aux={}, weight={}, n_eff={}, ridge_eps={}, learnable={}, target={}, weight_schedule={})",
            py_repr(py, self.aux.bind(py))?,
            self.weight,
            self.n_eff,
            self.ridge_eps,
            self.learnable,
            py_repr(py, self.target.bind(py))?,
            match &self.weight_schedule {
                Some(schedule) => py_repr(py, schedule.bind(py))?,
                None => "None".to_string(),
            }
        ))
    }

    /// Evaluate the iVAE ridge conditional-mean gauge penalty value and
    /// gradient at latent block `t` of shape `(n_eff, latent_dim)`. Returns
    /// `(value, grad)` with `grad` having the same shape as `t`.
    #[pyo3(signature = (t))]
    fn value_grad<'py>(
        &self,
        py: Python<'py>,
        t: PyReadonlyArray2<'py, f64>,
    ) -> PyResult<(f64, Py<PyArray2<f64>>)> {
        let view = t.as_array();
        let (n_rows, latent_dim) = view.dim();
        if n_rows as i64 != self.n_eff {
            return Err(PyValueError::new_err(format!(
                "IvaeRidgeMeanGauge.value_grad: t first dim {n_rows} must match n_eff={}",
                self.n_eff
            )));
        }
        if latent_dim == 0 {
            return Err(PyValueError::new_err(
                "IvaeRidgeMeanGauge.value_grad: t must have latent_dim > 0",
            ));
        }
        let aux_bound = aux_conditional_prior_float_array(py, self.aux.bind(py))?;
        let aux_array = aux_bound.extract::<PyReadonlyArray2<'_, f64>>()?;
        let aux_owned: Array2<f64> = aux_array.as_array().to_owned();
        let slice = PsiSlice::full(n_rows * latent_dim, Some(latent_dim));
        let penalty = IvaeRidgeMeanGaugePenalty::new(
            slice,
            aux_owned,
            self.ridge_eps,
            self.weight,
            n_rows,
            false,
        )
        .map_err(py_value_error)?;
        let flat: Array1<f64> = view.iter().copied().collect();
        let rho = Array1::<f64>::zeros(0);
        let value = penalty.value(flat.view(), rho.view());
        let grad_flat = penalty.grad_target(flat.view(), rho.view());
        let grad = grad_flat
            .into_shape_with_order((n_rows, latent_dim))
            .map_err(|err| {
                py_value_error(format!(
                    "IvaeRidgeMeanGauge.value_grad: gradient reshape failed: {err}"
                ))
            })?;
        Ok((value, grad.into_pyarray(py).unbind()))
    }
}

fn validate_ivae_ridge_mean_gauge_aux(
    aux: &Bound<'_, PyAny>,
    weight: f64,
    n_eff: i64,
    ridge_eps: f64,
) -> PyResult<()> {
    if !(weight > 0.0) {
        return Err(PyValueError::new_err(format!(
            "IvaeRidgeMeanGauge.weight must be > 0, got {weight}"
        )));
    }
    if n_eff <= 0 {
        return Err(PyValueError::new_err(format!(
            "IvaeRidgeMeanGauge.n_eff must be > 0, got {n_eff}"
        )));
    }
    if !ridge_eps.is_finite() || ridge_eps <= 0.0 {
        return Err(PyValueError::new_err(format!(
            "IvaeRidgeMeanGauge.ridge_eps must be finite and > 0, got {ridge_eps}"
        )));
    }

    let array = aux.extract::<PyReadonlyArrayDyn<'_, f64>>()?;
    let view = array.as_array();
    let shape = view.shape();
    if view.ndim() != 2 {
        return Err(PyValueError::new_err(format!(
            "IvaeRidgeMeanGauge.aux must have shape (N, q), got ndim={}",
            view.ndim()
        )));
    }

    let n_obs = shape[0];
    let aux_dim = shape[1];
    if n_obs as i64 != n_eff {
        return Err(PyValueError::new_err(format!(
            "IvaeRidgeMeanGauge.aux first dimension must equal n_eff={n_eff}, got {n_obs}"
        )));
    }
    if aux_dim == 0 {
        return Err(PyValueError::new_err(
            "IvaeRidgeMeanGauge.aux must have q > 0",
        ));
    }
    if !view.iter().all(|value| value.is_finite()) {
        return Err(PyValueError::new_err(
            "IvaeRidgeMeanGauge.aux must be finite",
        ));
    }
    Ok(())
}

#[pyclass(module = "gam_pyffi._rust", name = "MechanismSparsityPenalty")]
struct MechanismSparsityPenalty {
    #[pyo3(get, set)]
    target: PyObject,
    #[pyo3(get, set)]
    feature_groups: Vec<Vec<usize>>,
    #[pyo3(get, set)]
    weight: f64,
    #[pyo3(get, set)]
    smoothing_eps: f64,
    #[pyo3(get, set)]
    n_eff: f64,
    #[pyo3(get, set)]
    learnable: bool,
    #[pyo3(get)]
    weight_schedule: Option<PyObject>,
}

#[pymethods]
impl MechanismSparsityPenalty {
    #[new]
    #[pyo3(
        signature = (
            feature_groups,
            weight,
            n_eff,
            smoothing_eps = 1.0e-6,
            learnable = false,
            *,
            target = None
        ),
        text_signature = "(feature_groups, weight, n_eff, smoothing_eps=1e-06, learnable=False, *, target=None)"
    )]
    fn new(
        py: Python<'_>,
        feature_groups: &Bound<'_, PyAny>,
        weight: f64,
        n_eff: f64,
        smoothing_eps: f64,
        learnable: bool,
        target: Option<&Bound<'_, PyAny>>,
    ) -> PyResult<Self> {
        if weight <= 0.0 {
            return Err(PyValueError::new_err(format!(
                "MechanismSparsityPenalty.weight must be > 0, got {weight}"
            )));
        }
        if smoothing_eps <= 0.0 {
            return Err(PyValueError::new_err(format!(
                "MechanismSparsityPenalty.smoothing_eps must be > 0, got {smoothing_eps}"
            )));
        }
        if !n_eff.is_finite() || n_eff <= 0.0 {
            return Err(PyValueError::new_err(format!(
                "MechanismSparsityPenalty.n_eff must be > 0, got {n_eff}"
            )));
        }
        Ok(Self {
            target: py_object_or_string_default(py, target, "t"),
            feature_groups: coerce_mechanism_feature_groups(feature_groups)?,
            weight,
            smoothing_eps,
            n_eff,
            learnable,
            weight_schedule: None,
        })
    }

    #[classattr]
    const KIND_TAG: &'static str = "mechanism_sparsity";

    fn to_rust_descriptor(&self, py: Python<'_>) -> PyResult<PyObject> {
        let payload = PyDict::new(py);
        payload.set_item("kind", Self::KIND_TAG)?;
        set_mechanism_target_descriptor(py, &payload, &self.target)?;
        payload.set_item("feature_groups", self.feature_groups.clone())?;
        payload.set_item("weight", self.weight)?;
        payload.set_item("smoothing_eps", self.smoothing_eps)?;
        payload.set_item("n_eff", self.n_eff)?;
        payload.set_item("learnable", self.learnable)?;
        if let Some(schedule) = &self.weight_schedule {
            let schedule = mechanism_weight_schedule_descriptor(py, schedule)?;
            payload.set_item("weight_schedule", schedule)?;
        }
        Ok(payload.unbind().into_any())
    }

    fn set_weight_schedule<'py>(
        mut slf: PyRefMut<'py, Self>,
        schedule: PyObject,
    ) -> PyRefMut<'py, Self> {
        slf.weight_schedule = Some(schedule);
        slf
    }

    fn __repr__(&self, py: Python<'_>) -> PyResult<String> {
        let target_repr = self.target.bind(py).repr()?.extract::<String>()?;
        let schedule_repr = match &self.weight_schedule {
            Some(schedule) => schedule.bind(py).repr()?.extract::<String>()?,
            None => "None".to_string(),
        };
        let learnable = if self.learnable { "True" } else { "False" };
        Ok(format!(
            "MechanismSparsityPenalty(target={target_repr}, feature_groups={:?}, weight={}, smoothing_eps={}, n_eff={}, learnable={learnable}, weight_schedule={schedule_repr})",
            self.feature_groups, self.weight, self.smoothing_eps, self.n_eff
        ))
    }

    /// Evaluate the group-lasso mechanism-sparsity penalty value and gradient
    /// at decoder weight matrix `t` of shape `(d_latent, p_features)`.
    /// Returns `(value, grad)` with `grad` having the same shape as `t`.
    #[pyo3(signature = (t))]
    fn value_grad<'py>(
        &self,
        py: Python<'py>,
        t: PyReadonlyArray2<'py, f64>,
    ) -> PyResult<(f64, Py<PyArray2<f64>>)> {
        let view = t.as_array();
        let (d_latent, p_features) = view.dim();
        if d_latent == 0 || p_features == 0 {
            return Err(PyValueError::new_err(
                "MechanismSparsityPenalty.value_grad: t must have non-zero dimensions",
            ));
        }
        let total = d_latent * p_features;
        let slice = PsiSlice::full(total, Some(d_latent));
        let penalty = CoreMechanismSparsityPenalty::new(
            slice,
            self.feature_groups.clone(),
            self.weight,
            self.smoothing_eps,
            self.n_eff,
            false,
        )
        .map_err(py_value_error)?;
        let flat: Array1<f64> = view.iter().copied().collect();
        let rho = Array1::<f64>::zeros(0);
        let value = penalty.value(flat.view(), rho.view());
        let grad_flat = penalty.grad_target(flat.view(), rho.view());
        let grad = grad_flat
            .into_shape_with_order((d_latent, p_features))
            .map_err(|err| {
                py_value_error(format!(
                    "MechanismSparsityPenalty.value_grad: gradient reshape failed: {err}"
                ))
            })?;
        Ok((value, grad.into_pyarray(py).unbind()))
    }
}

fn coerce_mechanism_feature_groups(feature_groups: &Bound<'_, PyAny>) -> PyResult<Vec<Vec<usize>>> {
    let groups_iter = feature_groups.try_iter().map_err(|_| {
        PyTypeError::new_err(
            "MechanismSparsityPenalty.feature_groups must be a sequence of integer sequences",
        )
    })?;
    let mut out = Vec::new();
    let mut seen = BTreeSet::new();
    let mut max_feature = None::<usize>;
    for (group_idx, raw_group) in groups_iter.enumerate() {
        let raw_group = raw_group?;
        let features_iter = raw_group.try_iter().map_err(|_| {
            PyTypeError::new_err(
                "MechanismSparsityPenalty.feature_groups must be a sequence of integer sequences",
            )
        })?;
        let mut group = Vec::new();
        for raw_feature in features_iter {
            let raw_feature = raw_feature?;
            let indexed = raw_feature.call_method0("__index__").map_err(|_| {
                PyTypeError::new_err(
                    "MechanismSparsityPenalty.feature_groups must be a sequence of integer sequences",
                )
            })?;
            let feature = indexed.extract::<isize>().map_err(|_| {
                PyTypeError::new_err(
                    "MechanismSparsityPenalty.feature_groups must be a sequence of integer sequences",
                )
            })?;
            if feature < 0 {
                return Err(PyValueError::new_err(format!(
                    "MechanismSparsityPenalty.feature_groups entries must be non-negative; got {feature}"
                )));
            }
            let feature = usize::try_from(feature).map_err(|_| {
                PyValueError::new_err(format!(
                    "MechanismSparsityPenalty.feature_groups entries must be non-negative; got {feature}"
                ))
            })?;
            if !seen.insert(feature) {
                return Err(PyValueError::new_err(format!(
                    "MechanismSparsityPenalty.feature_groups feature {feature} appears more than once"
                )));
            }
            max_feature = Some(max_feature.map_or(feature, |current| current.max(feature)));
            group.push(feature);
        }
        if group.is_empty() {
            return Err(PyValueError::new_err(format!(
                "MechanismSparsityPenalty.feature_groups[{group_idx}] must not be empty"
            )));
        }
        out.push(group);
    }
    if out.is_empty() {
        return Err(PyValueError::new_err(
            "MechanismSparsityPenalty.feature_groups must not be empty",
        ));
    }
    if let Some(max_feature) = max_feature {
        for feature in 0..=max_feature {
            if !seen.contains(&feature) {
                return Err(PyValueError::new_err(format!(
                    "MechanismSparsityPenalty.feature_groups must partition contiguous features from 0; missing feature {feature}"
                )));
            }
        }
    }
    Ok(out)
}

fn set_mechanism_target_descriptor(
    py: Python<'_>,
    payload: &Bound<'_, PyDict>,
    target: &PyObject,
) -> PyResult<()> {
    let target = target.bind(py);
    if let Ok(target_name) = target.extract::<String>() {
        payload.set_item("target", target_name)?;
        return Ok(());
    }
    if let Ok(target_index) = target.extract::<isize>() {
        payload.set_item("target", target_index)?;
        return Ok(());
    }
    if let Ok(name) = target.getattr("name") {
        payload.set_item("target", name.str()?.to_str()?)?;
        return Ok(());
    }
    Err(PyTypeError::new_err(
        "analytic penalty target must be a latent block name, latent block index, or object exposing a 'name' attribute",
    ))
}

fn mechanism_weight_schedule_descriptor(py: Python<'_>, schedule: &PyObject) -> PyResult<PyObject> {
    let schedule = schedule.bind(py);
    if schedule.hasattr("to_rust_descriptor")? {
        return Ok(schedule.call_method0("to_rust_descriptor")?.unbind());
    }
    if schedule.cast::<PyDict>().is_ok() {
        return Ok(schedule.clone().unbind());
    }
    Err(PyTypeError::new_err(
        "weight_schedule must be ScalarWeightSchedule, a mapping, or None",
    ))
}

#[pyclass(module = "gam_pyffi._rust", name = "BlockOrthogonalityPenalty")]
struct BlockOrthogonalityPenalty {
    #[pyo3(get, set)]
    target: PyObject,
    #[pyo3(get, set)]
    groups: Vec<Vec<i64>>,
    #[pyo3(get, set)]
    weight: f64,
    #[pyo3(get, set)]
    n_eff: i64,
    #[pyo3(get, set)]
    learnable: bool,
    #[pyo3(get)]
    weight_schedule: Option<PyObject>,
}

fn block_orthogonality_groups(groups: &Bound<'_, PyAny>) -> PyResult<Vec<Vec<i64>>> {
    let group_iter = groups.try_iter().map_err(|_| {
        PyTypeError::new_err(
            "BlockOrthogonalityPenalty.groups must be a sequence of integer sequences",
        )
    })?;
    let mut coerced = Vec::new();
    for group in group_iter {
        let group = group.map_err(|_| {
            PyTypeError::new_err(
                "BlockOrthogonalityPenalty.groups must be a sequence of integer sequences",
            )
        })?;
        let axis_iter = group.try_iter().map_err(|_| {
            PyTypeError::new_err(
                "BlockOrthogonalityPenalty.groups must be a sequence of integer sequences",
            )
        })?;
        let mut axes = Vec::new();
        for axis in axis_iter {
            let axis = axis.map_err(|_| {
                PyTypeError::new_err(
                    "BlockOrthogonalityPenalty.groups must be a sequence of integer sequences",
                )
            })?;
            axes.push(axis.call_method0("__index__")?.extract::<i64>()?);
        }
        coerced.push(axes);
    }
    if coerced.len() < 2 {
        return Err(PyValueError::new_err(
            "BlockOrthogonalityPenalty.groups must contain at least two groups",
        ));
    }
    let mut seen = BTreeSet::new();
    for (group_idx, group) in coerced.iter().enumerate() {
        if group.is_empty() {
            return Err(PyValueError::new_err(format!(
                "BlockOrthogonalityPenalty.groups[{group_idx}] must not be empty"
            )));
        }
        for &axis in group {
            if axis < 0 {
                return Err(PyValueError::new_err(format!(
                    "BlockOrthogonalityPenalty.groups entries must be non-negative; got {axis}"
                )));
            }
            if !seen.insert(axis) {
                return Err(PyValueError::new_err(format!(
                    "BlockOrthogonalityPenalty.groups axis {axis} appears more than once"
                )));
            }
        }
    }
    let max_axis = seen.iter().next_back().copied().unwrap_or(0);
    for axis in 0..=max_axis {
        if !seen.contains(&axis) {
            return Err(PyValueError::new_err(format!(
                "BlockOrthogonalityPenalty.groups must partition contiguous axes from 0; missing axis {axis}"
            )));
        }
    }
    Ok(coerced)
}

fn block_orthogonality_target_descriptor(target: &Bound<'_, PyAny>) -> PyResult<PyObject> {
    if target.is_instance_of::<PyString>() || target.extract::<isize>().is_ok() {
        return Ok(target.clone().unbind());
    }
    if target.hasattr("to_rust_descriptor")? {
        return Ok(target.call_method0("to_rust_descriptor")?.unbind().into());
    }
    Err(PyTypeError::new_err(
        "target must be a string, integer, or object with to_rust_descriptor()",
    ))
}

#[pymethods]
impl BlockOrthogonalityPenalty {
    #[new]
    #[pyo3(signature = (groups, weight, n_eff, learnable = false, *, target = None))]
    fn new(
        py: Python<'_>,
        groups: &Bound<'_, PyAny>,
        weight: f64,
        n_eff: &Bound<'_, PyAny>,
        learnable: bool,
        target: Option<&Bound<'_, PyAny>>,
    ) -> PyResult<Self> {
        let groups = block_orthogonality_groups(groups)?;
        let n_eff = n_eff.call_method0("__index__")?.extract::<i64>()?;
        if weight <= 0.0 {
            return Err(PyValueError::new_err(format!(
                "BlockOrthogonalityPenalty.weight must be > 0, got {weight}"
            )));
        }
        if n_eff <= 0 {
            return Err(PyValueError::new_err(format!(
                "BlockOrthogonalityPenalty.n_eff must be > 0, got {n_eff}"
            )));
        }
        Ok(Self {
            target: py_object_or_string_default(py, target, "t"),
            groups,
            weight,
            n_eff,
            learnable,
            weight_schedule: None,
        })
    }

    #[classattr]
    const KIND_TAG: &'static str = "block_orthogonality";

    fn to_rust_descriptor(&self, py: Python<'_>) -> PyResult<PyObject> {
        let payload = PyDict::new(py);
        payload.set_item("kind", Self::KIND_TAG)?;
        payload.set_item(
            "target",
            block_orthogonality_target_descriptor(self.target.bind(py))?,
        )?;
        payload.set_item("groups", &self.groups)?;
        payload.set_item("weight", self.weight)?;
        payload.set_item("n_eff", self.n_eff)?;
        payload.set_item("learnable", self.learnable)?;
        if let Some(schedule) = topk_weight_schedule_descriptor(py, &self.weight_schedule)? {
            payload.set_item("weight_schedule", schedule)?;
        }
        Ok(payload.into())
    }

    fn set_weight_schedule<'py>(
        mut slf: PyRefMut<'py, Self>,
        schedule: PyObject,
    ) -> PyRefMut<'py, Self> {
        slf.weight_schedule = Some(schedule);
        slf
    }

    fn __repr__(&self, py: Python<'_>) -> PyResult<String> {
        Ok(format!(
            "BlockOrthogonalityPenalty(target={}, groups={:?}, weight={}, n_eff={}, learnable={}, weight_schedule={})",
            py_repr(py, self.target.bind(py))?,
            self.groups,
            self.weight,
            self.n_eff,
            self.learnable,
            match &self.weight_schedule {
                Some(schedule) => py_repr(py, schedule.bind(py))?,
                None => "None".to_string(),
            }
        ))
    }
}

// ---------------------------------------------------------------------------
// SheafConsistencyPenalty — cellular-sheaf consistency loss.
// ---------------------------------------------------------------------------

#[pyclass(module = "gam_pyffi._rust", name = "SheafConsistencyPenalty")]
struct SheafConsistencyPenalty {
    inner: CoreSheafConsistencyPenalty,
    #[pyo3(get)]
    weight: f64,
    #[pyo3(get)]
    target: PyObject,
}

fn sheaf_extract_edges(edges: &Bound<'_, PyAny>) -> PyResult<Vec<(usize, usize)>> {
    let iter = edges.try_iter().map_err(|_| {
        PyTypeError::new_err(
            "SheafConsistencyPenalty.edges must be a sequence of (u, v) integer pairs",
        )
    })?;
    let mut out = Vec::new();
    for item in iter {
        let item = item.map_err(|_| {
            PyTypeError::new_err(
                "SheafConsistencyPenalty.edges must be a sequence of (u, v) integer pairs",
            )
        })?;
        let pair = item.try_iter().map_err(|_| {
            PyTypeError::new_err(
                "SheafConsistencyPenalty.edges entries must be (u, v) sequences of length 2",
            )
        })?;
        let pair_vec: Vec<i64> = pair
            .map(|x| {
                x.and_then(|v| v.call_method0("__index__")?.extract::<i64>())
                    .map_err(|_| {
                        PyTypeError::new_err(
                            "SheafConsistencyPenalty.edges entries must contain integers",
                        )
                    })
            })
            .collect::<PyResult<_>>()?;
        if pair_vec.len() != 2 {
            return Err(PyValueError::new_err(format!(
                "SheafConsistencyPenalty.edges entries must have length 2, got {}",
                pair_vec.len()
            )));
        }
        if pair_vec[0] < 0 || pair_vec[1] < 0 {
            return Err(PyValueError::new_err(
                "SheafConsistencyPenalty.edges entries must be non-negative",
            ));
        }
        out.push((pair_vec[0] as usize, pair_vec[1] as usize));
    }
    Ok(out)
}

fn sheaf_extract_array2(obj: &Bound<'_, PyAny>, label: &str) -> PyResult<Array2<f64>> {
    let arr = obj.cast::<PyArray2<f64>>().map_err(|_| {
        PyTypeError::new_err(format!(
            "SheafConsistencyPenalty: {label} must be a 2-D numpy.ndarray of float64"
        ))
    })?;
    let readonly = arr.readonly();
    Ok(readonly.as_array().to_owned())
}

fn sheaf_extract_restrictions(
    restrictions: &Bound<'_, PyAny>,
) -> PyResult<Vec<CoreEdgeRestriction>> {
    let iter = restrictions.try_iter().map_err(|_| {
        PyTypeError::new_err("SheafConsistencyPenalty.restriction_ops must be a sequence")
    })?;
    let mut out = Vec::new();
    for (e, item) in iter.enumerate() {
        let item = item.map_err(|_| {
            PyTypeError::new_err(
                "SheafConsistencyPenalty.restriction_ops entries must be numpy arrays or (W_uv, W_vu) tuples",
            )
        })?;
        // Try direct 2-D array first → single-restriction edge.
        if item.cast::<PyArray2<f64>>().is_ok() {
            let r = sheaf_extract_array2(&item, &format!("restriction_ops[{e}]"))?;
            out.push(CoreEdgeRestriction::single(r));
            continue;
        }
        // Otherwise expect a (W_uv, W_vu) pair.
        let pair = item.try_iter().map_err(|_| {
            PyTypeError::new_err(format!(
                "SheafConsistencyPenalty.restriction_ops[{e}] must be a 2-D ndarray or a (W_uv, W_vu) tuple"
            ))
        })?;
        let parts: Vec<Bound<'_, PyAny>> = pair.collect::<PyResult<_>>()?;
        if parts.len() != 2 {
            return Err(PyValueError::new_err(format!(
                "SheafConsistencyPenalty.restriction_ops[{e}] tuple must have length 2, got {}",
                parts.len()
            )));
        }
        let r_uv = sheaf_extract_array2(&parts[0], &format!("restriction_ops[{e}].W_uv"))?;
        // Allow None for the second slot via Python's None object.
        if parts[1].is_none() {
            out.push(CoreEdgeRestriction::single(r_uv));
        } else {
            let r_vu = sheaf_extract_array2(&parts[1], &format!("restriction_ops[{e}].W_vu"))?;
            out.push(CoreEdgeRestriction::paired(r_uv, r_vu));
        }
    }
    Ok(out)
}

fn sheaf_extract_stalk_dims(stalk_dims: &Bound<'_, PyAny>) -> PyResult<Vec<usize>> {
    let iter = stalk_dims.try_iter().map_err(|_| {
        PyTypeError::new_err(
            "SheafConsistencyPenalty.stalk_dims must be a sequence of positive integers",
        )
    })?;
    let mut out = Vec::new();
    for item in iter {
        let item = item.map_err(|_| {
            PyTypeError::new_err("SheafConsistencyPenalty.stalk_dims entries must be integers")
        })?;
        let d = item.call_method0("__index__")?.extract::<i64>()?;
        if d <= 0 {
            return Err(PyValueError::new_err(format!(
                "SheafConsistencyPenalty.stalk_dims entries must be > 0, got {d}"
            )));
        }
        out.push(d as usize);
    }
    Ok(out)
}

#[pymethods]
impl SheafConsistencyPenalty {
    #[new]
    #[pyo3(signature = (edges, restriction_ops, stalk_dims, weight = 1.0, *, target = None))]
    fn new(
        py: Python<'_>,
        edges: &Bound<'_, PyAny>,
        restriction_ops: &Bound<'_, PyAny>,
        stalk_dims: &Bound<'_, PyAny>,
        weight: f64,
        target: Option<&Bound<'_, PyAny>>,
    ) -> PyResult<Self> {
        if !(weight.is_finite() && weight > 0.0) {
            return Err(PyValueError::new_err(format!(
                "SheafConsistencyPenalty.weight must be > 0, got {weight}"
            )));
        }
        let edges = sheaf_extract_edges(edges)?;
        let restrictions = sheaf_extract_restrictions(restriction_ops)?;
        let stalk_dims = sheaf_extract_stalk_dims(stalk_dims)?;
        let inner = CoreSheafConsistencyPenalty::new(edges, restrictions, weight, stalk_dims)
            .map_err(PyValueError::new_err)?;
        let target_obj = py_object_or_string_default(py, target, "z");
        Ok(Self {
            inner,
            weight,
            target: target_obj,
        })
    }

    #[classattr]
    const KIND_TAG: &'static str = "sheaf_consistency";

    #[getter]
    fn total_dim(&self) -> usize {
        self.inner.total_dim()
    }

    #[getter]
    fn num_edges(&self) -> usize {
        self.inner.num_edges()
    }

    #[getter]
    fn num_vertices(&self) -> usize {
        self.inner.num_vertices()
    }

    #[getter]
    fn stalk_dims(&self) -> Vec<usize> {
        self.inner.stalk_dims().to_vec()
    }

    fn __call__<'py>(&self, s: PyReadonlyArray1<'py, f64>) -> PyResult<f64> {
        let view = s.as_array();
        if view.len() != self.inner.total_dim() {
            return Err(PyValueError::new_err(format!(
                "SheafConsistencyPenalty: input length {} != total stalk dim {}",
                view.len(),
                self.inner.total_dim()
            )));
        }
        Ok(self.inner.value(view))
    }

    fn value<'py>(&self, s: PyReadonlyArray1<'py, f64>) -> PyResult<f64> {
        self.__call__(s)
    }

    fn gradient<'py>(
        &self,
        py: Python<'py>,
        s: PyReadonlyArray1<'py, f64>,
    ) -> PyResult<Bound<'py, PyArray1<f64>>> {
        let view = s.as_array();
        if view.len() != self.inner.total_dim() {
            return Err(PyValueError::new_err(format!(
                "SheafConsistencyPenalty.gradient: input length {} != total stalk dim {}",
                view.len(),
                self.inner.total_dim()
            )));
        }
        let g = self.inner.gradient(view);
        Ok(g.into_pyarray(py))
    }

    fn hessian_diag<'py>(
        &self,
        py: Python<'py>,
        s: PyReadonlyArray1<'py, f64>,
    ) -> PyResult<Bound<'py, PyArray1<f64>>> {
        let view = s.as_array();
        if view.len() != self.inner.total_dim() {
            return Err(PyValueError::new_err(format!(
                "SheafConsistencyPenalty.hessian_diag: input length {} != total stalk dim {}",
                view.len(),
                self.inner.total_dim()
            )));
        }
        let h = self.inner.hessian_diag(view);
        Ok(h.into_pyarray(py))
    }

    fn hvp<'py>(
        &self,
        py: Python<'py>,
        s: PyReadonlyArray1<'py, f64>,
        v: PyReadonlyArray1<'py, f64>,
    ) -> PyResult<Bound<'py, PyArray1<f64>>> {
        let sv = s.as_array();
        let vv = v.as_array();
        if sv.len() != self.inner.total_dim() || vv.len() != self.inner.total_dim() {
            return Err(PyValueError::new_err(format!(
                "SheafConsistencyPenalty.hvp: input lengths ({}, {}) != total stalk dim {}",
                sv.len(),
                vv.len(),
                self.inner.total_dim()
            )));
        }
        let hv = self.inner.hvp(sv, vv);
        Ok(hv.into_pyarray(py))
    }

    #[pyo3(signature = (tol = 1e-8))]
    fn harmonic_modes(&self, tol: f64) -> PyResult<usize> {
        if !(tol.is_finite() && tol >= 0.0) {
            return Err(PyValueError::new_err(format!(
                "SheafConsistencyPenalty.harmonic_modes: tol must be finite and >= 0, got {tol}"
            )));
        }
        Ok(self.inner.harmonic_modes(tol))
    }

    fn __repr__(&self) -> String {
        format!(
            "SheafConsistencyPenalty(K={}, edges={}, total_dim={}, weight={})",
            self.inner.num_vertices(),
            self.inner.num_edges(),
            self.inner.total_dim(),
            self.weight,
        )
    }
}

#[pyclass(module = "gam_pyffi._rust", name = "AuxConditionalPriorPenalty")]
struct AuxConditionalPriorPenalty {
    #[pyo3(get, set)]
    target: PyObject,
    #[pyo3(get, set)]
    lambda_per_row: PyObject,
    #[pyo3(get, set)]
    weight: f64,
    #[pyo3(get, set)]
    n_eff: i64,
    #[pyo3(get, set)]
    learnable: bool,
    #[pyo3(get)]
    weight_schedule: Option<PyObject>,
}

#[pymethods]
impl AuxConditionalPriorPenalty {
    #[new]
    #[pyo3(signature = (lambda_per_row, weight, n_eff, learnable = false, *, target = None))]
    fn new(
        py: Python<'_>,
        lambda_per_row: &Bound<'_, PyAny>,
        weight: &Bound<'_, PyAny>,
        n_eff: &Bound<'_, PyAny>,
        learnable: bool,
        target: Option<&Bound<'_, PyAny>>,
    ) -> PyResult<Self> {
        let builtins = py.import("builtins")?;
        let weight = builtins
            .getattr("float")?
            .call1((weight,))?
            .extract::<f64>()?;
        let n_eff = builtins.getattr("int")?.call1((n_eff,))?.extract::<i64>()?;
        let lambda_per_row = aux_conditional_prior_float_array(py, lambda_per_row)?;
        validate_aux_conditional_prior_lambda(&lambda_per_row, weight, n_eff)?;
        Ok(Self {
            target: py_object_or_string_default(py, target, "t"),
            lambda_per_row: lambda_per_row.unbind(),
            weight,
            n_eff,
            learnable,
            weight_schedule: None,
        })
    }

    #[classattr]
    const KIND_TAG: &'static str = "aux_conditional_prior";

    fn to_rust_descriptor(&self, py: Python<'_>) -> PyResult<PyObject> {
        let payload = PyDict::new(py);
        payload.set_item("kind", Self::KIND_TAG)?;
        payload.set_item("target", target_descriptor(py, self.target.bind(py))?)?;

        let array = self
            .lambda_per_row
            .bind(py)
            .extract::<PyReadonlyArrayDyn<'_, f64>>()?;
        let view = array.as_array();
        let flattened = view.iter().copied().collect::<Vec<_>>();
        payload.set_item("lambda_per_row", PyList::new(py, flattened)?)?;
        payload.set_item("lambda_per_row_shape", PyList::new(py, view.shape())?)?;
        payload.set_item("weight", self.weight)?;
        payload.set_item("n_eff", self.n_eff)?;
        payload.set_item("learnable", self.learnable)?;
        if let Some(schedule) = topk_weight_schedule_descriptor(py, &self.weight_schedule)? {
            payload.set_item("weight_schedule", schedule)?;
        }
        Ok(payload.into())
    }

    fn set_weight_schedule<'py>(
        mut slf: PyRefMut<'py, Self>,
        schedule: PyObject,
    ) -> PyRefMut<'py, Self> {
        slf.weight_schedule = Some(schedule);
        slf
    }

    fn __repr__(&self, py: Python<'_>) -> PyResult<String> {
        Ok(format!(
            "AuxConditionalPriorPenalty(lambda_per_row={}, weight={}, n_eff={}, learnable={}, target={}, weight_schedule={})",
            py_repr(py, self.lambda_per_row.bind(py))?,
            self.weight,
            self.n_eff,
            self.learnable,
            py_repr(py, self.target.bind(py))?,
            match &self.weight_schedule {
                Some(schedule) => py_repr(py, schedule.bind(py))?,
                None => "None".to_string(),
            }
        ))
    }

    /// Evaluate the fixed-precomputed row-precision prior value and gradient
    /// at latent block `t` of shape `(n_eff, latent_dim)`. Returns
    /// `(value, grad)` with `grad` having the same shape as `t`.
    #[pyo3(signature = (t))]
    fn value_grad<'py>(
        &self,
        py: Python<'py>,
        t: PyReadonlyArray2<'py, f64>,
    ) -> PyResult<(f64, Py<PyArray2<f64>>)> {
        let view = t.as_array();
        let (n_rows, latent_dim) = view.dim();
        if n_rows as i64 != self.n_eff {
            return Err(PyValueError::new_err(format!(
                "AuxConditionalPriorPenalty.value_grad: t first dim {n_rows} must match n_eff={}",
                self.n_eff
            )));
        }
        if latent_dim == 0 {
            return Err(PyValueError::new_err(
                "AuxConditionalPriorPenalty.value_grad: t must have latent_dim > 0",
            ));
        }
        let lambda_array = self
            .lambda_per_row
            .bind(py)
            .extract::<PyReadonlyArray3<'_, f64>>()?;
        let lambda_owned: Array3<f64> = lambda_array.as_array().to_owned();
        let slice = PsiSlice::full(n_rows * latent_dim, Some(latent_dim));
        let n_eff_usize = usize::try_from(self.n_eff).map_err(|_| {
            py_value_error(format!(
                "AuxConditionalPriorPenalty.value_grad: n_eff={} does not fit in usize",
                self.n_eff
            ))
        })?;
        let penalty =
            RowPrecisionPriorPenalty::new(slice, lambda_owned, self.weight, n_eff_usize, false)
                .map_err(py_value_error)?;
        let flat: Array1<f64> = view.iter().copied().collect();
        let rho = Array1::<f64>::zeros(0);
        let value = penalty.value(flat.view(), rho.view());
        let grad_flat = penalty.grad_target(flat.view(), rho.view());
        let grad = grad_flat
            .into_shape_with_order((n_rows, latent_dim))
            .map_err(|err| {
                py_value_error(format!(
                    "AuxConditionalPriorPenalty.value_grad: gradient reshape failed: {err}"
                ))
            })?;
        Ok((value, grad.into_pyarray(py).unbind()))
    }
}

#[pyclass(module = "gam_pyffi._rust", name = "NuclearNormPenalty")]
struct NuclearNormPenalty {
    #[pyo3(get, set)]
    target: PyObject,
    #[pyo3(get, set)]
    weight: f64,
    #[pyo3(get, set)]
    n_eff: i64,
    #[pyo3(get, set)]
    smoothing_eps: f64,
    #[pyo3(get, set)]
    max_rank: Option<i64>,
    #[pyo3(get, set)]
    learnable: bool,
    #[pyo3(get)]
    weight_schedule: Option<PyObject>,
}

#[pymethods]
impl NuclearNormPenalty {
    #[new]
    #[pyo3(signature = (weight, n_eff, smoothing_eps = 1.0e-6, max_rank = None, learnable = false, *, target = None))]
    fn new(
        py: Python<'_>,
        weight: f64,
        n_eff: &Bound<'_, PyAny>,
        smoothing_eps: f64,
        max_rank: Option<&Bound<'_, PyAny>>,
        learnable: bool,
        target: Option<&Bound<'_, PyAny>>,
    ) -> PyResult<Self> {
        let n_eff = n_eff.call_method0("__index__")?.extract::<i64>()?;
        let max_rank = match max_rank {
            Some(rank) => Some(rank.call_method0("__index__")?.extract::<i64>()?),
            None => None,
        };
        if !weight.is_finite() || weight <= 0.0 {
            return Err(PyValueError::new_err(format!(
                "NuclearNormPenalty.weight must be > 0, got {weight}"
            )));
        }
        if n_eff <= 0 {
            return Err(PyValueError::new_err(format!(
                "NuclearNormPenalty.n_eff must be > 0, got {n_eff}"
            )));
        }
        if !(smoothing_eps > 0.0) {
            return Err(PyValueError::new_err(format!(
                "NuclearNormPenalty.smoothing_eps must be > 0, got {smoothing_eps}"
            )));
        }
        if let Some(rank) = max_rank {
            if rank <= 0 {
                return Err(PyValueError::new_err(format!(
                    "NuclearNormPenalty.max_rank must be > 0, got {rank}"
                )));
            }
        }
        Ok(Self {
            target: py_object_or_string_default(py, target, "t"),
            weight,
            n_eff,
            smoothing_eps,
            max_rank,
            learnable,
            weight_schedule: None,
        })
    }

    #[classattr]
    const KIND_TAG: &'static str = "nuclear_norm";

    fn to_rust_descriptor(&self, py: Python<'_>) -> PyResult<PyObject> {
        let payload = PyDict::new(py);
        payload.set_item("kind", Self::KIND_TAG)?;
        payload.set_item("target", target_descriptor(py, self.target.bind(py))?)?;
        payload.set_item("weight", self.weight)?;
        payload.set_item("n_eff", self.n_eff)?;
        payload.set_item("smoothing_eps", self.smoothing_eps)?;
        payload.set_item("max_rank", self.max_rank)?;
        payload.set_item("learnable", self.learnable)?;
        if let Some(schedule) = topk_weight_schedule_descriptor(py, &self.weight_schedule)? {
            payload.set_item("weight_schedule", schedule)?;
        }
        Ok(payload.into())
    }

    fn set_weight_schedule<'py>(
        mut slf: PyRefMut<'py, Self>,
        schedule: PyObject,
    ) -> PyRefMut<'py, Self> {
        slf.weight_schedule = Some(schedule);
        slf
    }

    fn __repr__(&self, py: Python<'_>) -> PyResult<String> {
        Ok(format!(
            "NuclearNormPenalty(weight={}, n_eff={}, smoothing_eps={}, max_rank={}, learnable={}, target={}, weight_schedule={})",
            self.weight,
            self.n_eff,
            self.smoothing_eps,
            match self.max_rank {
                Some(rank) => rank.to_string(),
                None => "None".to_string(),
            },
            self.learnable,
            py_repr(py, self.target.bind(py))?,
            match &self.weight_schedule {
                Some(schedule) => py_repr(py, schedule.bind(py))?,
                None => "None".to_string(),
            }
        ))
    }
}

/// Identifiability theorem precondition checks (Principle (f)).
///
/// Caller serialises a `gam::inference::identifiability_precondition_checks::FitSummary` as
/// JSON; Rust does all numerical work (min std, faer-SVD rank, decoder
/// zero-fraction, activation variance bounds) and returns a JSON array of
/// per-theorem results. This is the *single source of truth* — Python's
/// `gamfit.identifiability.check`, the CLI, and any future R/Julia binding
/// all consume this same FFI.
#[pyfunction]
fn identifiability_check_json(input: &str) -> PyResult<String> {
    gam::identifiability::precondition::identifiability_check_json(input).map_err(py_value_error)
}

/// Set the solver's stderr log verbosity at runtime from Python.
///
/// The extension installs the logger at the quiet `warn` default on import
/// (#1688), so the per-iteration solver trace (`[OUTER …]`, `[KAPPA-PHASE …]`,
/// the `[#1271-diag]` REML logdet dump, etc.) is suppressed — that stream also
/// carries real per-evaluation compute (eigendecompositions), not just I/O, so
/// quieting it speeds fits as well as silencing them. Call
/// `gam._rust.set_log_level("info")` (or `debug`/`trace`) to opt back into the
/// full trace; `off`/`error`/`warn` quiet it further. The accepted spellings
/// are the standard `log` level names (case-insensitive); an unrecognized value
/// raises `ValueError`.
#[pyfunction]
fn set_log_level(level: &str) -> PyResult<()> {
    match gam::solver::progress_log::parse_level_directive(level) {
        Some(filter) => {
            gam::solver::progress_log::init_logging_at(filter);
            Ok(())
        }
        None => Err(py_value_error(format!(
            "unrecognized log level {level:?}; expected one of off|error|warn|info|debug|trace"
        ))),
    }
}

#[pymodule(name = "_rust", gil_used = false)]
fn rust_extension(module: &Bound<'_, PyModule>) -> PyResult<()> {
    gam::init_parallelism();
    // Install the same stderr logger used by the CLI so long-running Rust
    // solver phases (including survival marginal-slope joint-Newton cycles)
    // are visible from Python without requiring a separate shell.
    gam::solver::progress_log::init_logging();
    // Background process monitor: emits a `[process-monitor] elapsed=… rss=…`
    // line every 60s for the life of the process, so silent stretches
    // inside long PIRLS line-searches still surface a process-alive
    // signal with current memory footprint. Unconditional — does not
    // depend on any family pushing a tracked scope.
    gam_runtime::process_monitor::start();
    module.add("__doc__", "PyO3 boundary for the gam Rust engine.")?;
    module.add("__version__", env!("CARGO_PKG_VERSION"))?;
    // gamfit exception hierarchy (see the comment block at the
    // `create_exception!` definitions above and issue #343). Registering
    // the classes here makes them addressable as `gam._rust.GamError`,
    // `gam._rust.RemlConvergenceError`, etc.; `gamfit/_exceptions.py`
    // re-exports each one under its public `gamfit.*` name so the class
    // identity caught by `pytest.raises(gamfit.RemlConvergenceError)`
    // and constructed by `RemlConvergenceError::new_err(...)` on the
    // Rust side is exactly the same object.
    module.add("GamError", module.py().get_type::<GamError>())?;
    module.add("FormulaError", module.py().get_type::<FormulaError>())?;
    module.add(
        "ColumnNotFoundError",
        module.py().get_type::<ColumnNotFoundError>(),
    )?;
    module.add(
        "SchemaMismatchError",
        module.py().get_type::<SchemaMismatchError>(),
    )?;
    module.add("PredictionError", module.py().get_type::<PredictionError>())?;
    module.add("BasisError", module.py().get_type::<BasisError>())?;
    module.add(
        "LinearSystemSolveError",
        module.py().get_type::<LinearSystemSolveError>(),
    )?;
    module.add(
        "EigendecompositionError",
        module.py().get_type::<EigendecompositionError>(),
    )?;
    module.add(
        "PenaltySpectrumError",
        module.py().get_type::<PenaltySpectrumError>(),
    )?;
    module.add(
        "ParameterConstraintError",
        module.py().get_type::<ParameterConstraintError>(),
    )?;
    module.add(
        "PirlsConvergenceError",
        module.py().get_type::<PirlsConvergenceError>(),
    )?;
    module.add(
        "PerfectSeparationError",
        module.py().get_type::<PerfectSeparationError>(),
    )?;
    module.add(
        "HessianNotPositiveDefiniteError",
        module.py().get_type::<HessianNotPositiveDefiniteError>(),
    )?;
    module.add(
        "RemlConvergenceError",
        module.py().get_type::<RemlConvergenceError>(),
    )?;
    module.add(
        "GradientUnavailableError",
        module.py().get_type::<GradientUnavailableError>(),
    )?;
    module.add("LayoutError", module.py().get_type::<LayoutError>())?;
    module.add(
        "ModelOverparameterizedError",
        module.py().get_type::<ModelOverparameterizedError>(),
    )?;
    module.add(
        "IllConditionedError",
        module.py().get_type::<IllConditionedError>(),
    )?;
    module.add(
        "InvalidInputError",
        module.py().get_type::<InvalidInputError>(),
    )?;
    module.add(
        "MonotoneRootError",
        module.py().get_type::<MonotoneRootError>(),
    )?;
    module.add("CalibratorError", module.py().get_type::<CalibratorError>())?;
    module.add(
        "InvalidSpecificationError",
        module.py().get_type::<InvalidSpecificationError>(),
    )?;
    // Remaining engine error enum subclasses (issue #343 follow-up).
    module.add("GeometryError", module.py().get_type::<GeometryError>())?;
    module.add(
        "MatrixMaterializationError",
        module.py().get_type::<MatrixMaterializationError>(),
    )?;
    module.add("GpuError", module.py().get_type::<GpuError>())?;
    module.add(
        "LinearAlgebraError",
        module.py().get_type::<LinearAlgebraError>(),
    )?;
    module.add("MatrixError", module.py().get_type::<MatrixError>())?;
    module.add("CacheStoreError", module.py().get_type::<CacheStoreError>())?;
    module.add("SmoothError", module.py().get_type::<SmoothError>())?;
    module.add("ArrowSchurError", module.py().get_type::<ArrowSchurError>())?;
    module.add(
        "OuterStrategyError",
        module.py().get_type::<OuterStrategyError>(),
    )?;
    module.add(
        "TermBuilderError",
        module.py().get_type::<TermBuilderError>(),
    )?;
    module.add(
        "CorrectedCovarianceError",
        module.py().get_type::<CorrectedCovarianceError>(),
    )?;
    module.add(
        "PredictInputError",
        module.py().get_type::<PredictInputError>(),
    )?;
    module.add("HmcError", module.py().get_type::<HmcError>())?;
    module.add("AloError", module.py().get_type::<AloError>())?;
    module.add("SurvivalError", module.py().get_type::<SurvivalError>())?;
    module.add(
        "CubicCellKernelError",
        module.py().get_type::<CubicCellKernelError>(),
    )?;
    module.add(
        "SurvivalConstructionError",
        module.py().get_type::<SurvivalConstructionError>(),
    )?;
    module.add(
        "TransformationNormalError",
        module.py().get_type::<TransformationNormalError>(),
    )?;
    module.add(
        "CustomFamilyError",
        module.py().get_type::<CustomFamilyError>(),
    )?;
    module.add("GamlssError", module.py().get_type::<GamlssError>())?;
    module.add(
        "SurvivalMarginalSlopeError",
        module.py().get_type::<SurvivalMarginalSlopeError>(),
    )?;
    module.add(
        "LatentSurvivalError",
        module.py().get_type::<LatentSurvivalError>(),
    )?;
    module.add(
        "SurvivalPredictError",
        module.py().get_type::<SurvivalPredictError>(),
    )?;
    module.add(
        "DeviationRuntimeError",
        module.py().get_type::<DeviationRuntimeError>(),
    )?;
    module.add("DataError", module.py().get_type::<DataError>())?;
    module.add(
        "FittedModelError",
        module.py().get_type::<FittedModelError>(),
    )?;
    module.add(
        "LognormalKernelError",
        module.py().get_type::<LognormalKernelError>(),
    )?;
    module.add(
        "ScaleDesignError",
        module.py().get_type::<ScaleDesignError>(),
    )?;
    module.add(
        "IdentifiabilityCompilerError",
        module.py().get_type::<IdentifiabilityCompilerError>(),
    )?;
    module.add(
        "JointPenaltyError",
        module.py().get_type::<JointPenaltyError>(),
    )?;
    module.add(
        "SurvivalLocationScaleError",
        module.py().get_type::<SurvivalLocationScaleError>(),
    )?;
    module.add(
        "MapUniquenessError",
        module.py().get_type::<MapUniquenessError>(),
    )?;
    module.add(
        "UnsupportedLinkError",
        module.py().get_type::<UnsupportedLinkError>(),
    )?;
    module.add(
        "InvalidConfigurationError",
        module.py().get_type::<InvalidConfigurationError>(),
    )?;
    module.add(
        "MissingDependencyError",
        module.py().get_type::<MissingDependencyError>(),
    )?;
    module.add(
        "IntegrationError",
        module.py().get_type::<IntegrationError>(),
    )?;

    // #773: `create_exception!` stamps every gamfit exception with
    // `__module__ = "_rust"`, but the compiled extension is importable only as
    // `gamfit._rust` (see `gamfit._binding.rust_module`). Pickle records a class
    // by `(__module__, __qualname__)` and reconstructs it with `import _rust;
    // getattr(...)`, which fails — so any `_rust.*` exception raised inside a
    // `ProcessPoolExecutor` worker is masked by an opaque `PicklingError` that
    // hides the real failure and takes down the whole pool. Repoint each
    // exception class at its true importable module so the type round-trips
    // through pickle. Walking the module dict for `GamError` subclasses keeps
    // this correct as exceptions are added — no parallel name list to drift.
    {
        let gam_error = module.py().get_type::<GamError>();
        for (_name, value) in module.dict().iter() {
            let Ok(ty) = value.cast::<PyType>() else {
                continue;
            };
            if ty.is_subclass(gam_error.as_any())? {
                ty.as_any().setattr("__module__", "gamfit._rust")?;
            }
        }
    }

    module.add_class::<EuclideanManifold>()?;
    module.add_class::<CircleManifold>()?;
    module.add_class::<SphereManifold>()?;
    module.add_class::<TorusManifold>()?;
    module.add_class::<GrassmannManifold>()?;
    module.add_class::<StiefelManifold>()?;
    module.add_class::<SpdManifold>()?;
    module.add_class::<ProductManifold>()?;
    module.add_function(wrap_pyfunction!(fit_penalized_multinomial_pyfunc, module)?)?;
    module.add_function(wrap_pyfunction!(fit_multinomial_formula_pyfunc, module)?)?;
    module.add_function(wrap_pyfunction!(
        predict_multinomial_formula_pyfunc,
        module
    )?)?;
    module.add_function(wrap_pyfunction!(multinomial_model_metadata_pyfunc, module)?)?;
    module.add_function(wrap_pyfunction!(
        predict_multinomial_intervals_pyfunc,
        module
    )?)?;
    module.add_function(wrap_pyfunction!(
        multinomial_smooth_significance_pyfunc,
        module
    )?)?;
    module.add_function(wrap_pyfunction!(
        posterior_predict_multinomial_pyfunc,
        module
    )?)?;
    module.add_function(wrap_pyfunction!(sklearn_fit_metadata, module)?)?;
    module.add_function(wrap_pyfunction!(build_info, module)?)?;
    module.add_function(wrap_pyfunction!(fidelity_loss_recovered, module)?)?;
    module.add_function(wrap_pyfunction!(fidelity_r2_score, module)?)?;
    module.add_function(wrap_pyfunction!(fidelity_kl_categorical_rows, module)?)?;
    module.add_function(wrap_pyfunction!(fidelity_distortion_floor_r2, module)?)?;
    module.add_function(wrap_pyfunction!(identifiability_check_json, module)?)?;
    module.add_function(wrap_pyfunction!(set_log_level, module)?)?;
    module.add_function(wrap_pyfunction!(interpolate_survival_surface, module)?)?;
    module.add_function(wrap_pyfunction!(interpolate_rows, module)?)?;
    module.add_function(wrap_pyfunction!(survival_chunk_iter_collect, module)?)?;
    module.add_function(wrap_pyfunction!(write_survival_csv, module)?)?;
    module.add_function(wrap_pyfunction!(survival_coerce_times, module)?)?;
    module.add_function(wrap_pyfunction!(survival_parameters_matrix, module)?)?;
    module.add_function(wrap_pyfunction!(survival_collect_chunks, module)?)?;
    module.add_function(wrap_pyfunction!(hazard_from_cumulative_knots, module)?)?;
    module.add_function(wrap_pyfunction!(survival_cumulative_from_survival, module)?)?;
    module.add_function(wrap_pyfunction!(survival_block, module)?)?;
    module.add_function(wrap_pyfunction!(survival_block_hazard, module)?)?;
    module.add_function(wrap_pyfunction!(survival_block_cumulative_hazard, module)?)?;
    module.add_function(wrap_pyfunction!(survival_block_failure, module)?)?;
    module.add_function(wrap_pyfunction!(survival_ffi_surface, module)?)?;
    module.add_function(wrap_pyfunction!(numeric_matrix_validate, module)?)?;
    module.add_function(wrap_pyfunction!(numeric_matrix_f64, module)?)?;
    module.add_function(wrap_pyfunction!(marginal_slope_clip_probabilities, module)?)?;
    module.add_function(wrap_pyfunction!(
        transformation_normal_z_from_columns,
        module
    )?)?;
    module.add_function(wrap_pyfunction!(column_stack_f64, module)?)?;
    module.add_function(wrap_pyfunction!(
        survival_prediction_payload_from_json,
        module
    )?)?;
    module.add_function(wrap_pyfunction!(flat_to_matrix_f64, module)?)?;
    module.add_function(wrap_pyfunction!(vec_to_array1_f64, module)?)?;
    module.add_function(wrap_pyfunction!(extract_row_ids, module)?)?;
    module.add_function(wrap_pyfunction!(default_survival_time_grid, module)?)?;
    module.add_function(wrap_pyfunction!(torch_from_fitted, module)?)?;
    module.add_function(wrap_pyfunction!(fit_table, module)?)?;
    module.add_function(wrap_pyfunction!(fit_array, module)?)?;
    module.add_function(wrap_pyfunction!(load_model, module)?)?;
    module.add_function(wrap_pyfunction!(bayes_factor_log_diff, module)?)?;
    module.add_function(wrap_pyfunction!(saved_model_payload_string, module)?)?;
    module.add_function(wrap_pyfunction!(inference_notes_from_model, module)?)?;
    module.add_function(wrap_pyfunction!(
        required_saved_model_payload_string,
        module
    )?)?;
    module.add_function(wrap_pyfunction!(saved_model_predict_class_name, module)?)?;
    module.add_function(wrap_pyfunction!(build_extend_group_payload_json, module)?)?;
    module.add_function(wrap_pyfunction!(extend_model_with_group, module)?)?;
    module.add_function(wrap_pyfunction!(validate_formula_json, module)?)?;
    module.add_function(wrap_pyfunction!(
        apply_shape_constraints_to_formula,
        module
    )?)?;
    module.add_function(wrap_pyfunction!(
        formula_validation_supported_by_python_json,
        module
    )?)?;
    module.add_function(wrap_pyfunction!(formula_validation_repr_json, module)?)?;
    module.add_function(wrap_pyfunction!(formula_validation_html_json, module)?)?;
    module.add_function(wrap_pyfunction!(build_predict_payload_json, module)?)?;
    module.add_function(wrap_pyfunction!(build_model_predict_payload_json, module)?)?;
    module.add_function(wrap_pyfunction!(predict_table, module)?)?;
    module.add_function(wrap_pyfunction!(predict_table_conformal, module)?)?;
    // #1054: exact Gaussian jackknife+ conformal intervals (no calibration fold).
    module.add_function(wrap_pyfunction!(predict_table_jackknife_plus, module)?)?;
    // #1098: exact Gaussian full-conformal prediction set (no calibration fold).
    module.add_function(wrap_pyfunction!(predict_table_full_conformal, module)?)?;
    // #1057: posterior-predictive replicate sampling from the fitted model.
    module.add_function(wrap_pyfunction!(generative_replicates, module)?)?;
    module.add_function(wrap_pyfunction!(predict_array, module)?)?;
    module.add_function(wrap_pyfunction!(competing_risks_cif, module)?)?;
    module.add_function(wrap_pyfunction!(
        competing_risks_prediction_payload_from_json,
        module
    )?)?;
    module.add_function(wrap_pyfunction!(
        competing_risks_cif_from_predictions,
        module
    )?)?;
    module.add_function(wrap_pyfunction!(build_sample_payload_json, module)?)?;
    module.add_function(wrap_pyfunction!(
        build_analytic_penalty_registry_json,
        module
    )?)?;
    module.add_function(wrap_pyfunction!(sample_table, module)?)?;
    module.add_function(wrap_pyfunction!(design_matrix_table, module)?)?;
    module.add_function(wrap_pyfunction!(design_matrix_table_dense, module)?)?;
    module.add_function(wrap_pyfunction!(design_matrix_array, module)?)?;
    module.add_function(wrap_pyfunction!(
        build_difference_smooth_request_json,
        module
    )?)?;
    module.add_function(wrap_pyfunction!(decoder_channel_cov_factors, module)?)?;
    module.add_function(wrap_pyfunction!(decoder_cov_from_channel_factors, module)?)?;
    module.add_function(wrap_pyfunction!(sae_canonical_n_harmonics, module)?)?;
    module.add_function(wrap_pyfunction!(sae_atom_topologies, module)?)?;
    module.add_function(wrap_pyfunction!(sae_manifold_training_mean, module)?)?;
    module.add_function(wrap_pyfunction!(sae_periodic_shape_band_reorder, module)?)?;
    module.add_function(wrap_pyfunction!(sae_coercion_json_roundtrip, module)?)?;
    module.add_function(wrap_pyfunction!(
        sae_manifold_core_from_fit_payload,
        module
    )?)?;
    module.add_function(wrap_pyfunction!(sae_manifold_payload_roundtrip, module)?)?;
    module.add_class::<ManifoldSaeCore>()?;
    module.add_class::<AtomCore>()?;
    module.add_function(wrap_pyfunction!(bspline_basis, module)?)?;
    module.add_function(wrap_pyfunction!(bspline_basis_derivative, module)?)?;
    module.add_function(wrap_pyfunction!(basis_with_jet, module)?)?;
    module.add_function(wrap_pyfunction!(periodic_basis_with_jet, module)?)?;
    module.add_function(wrap_pyfunction!(periodic_spline_curve_basis, module)?)?;
    module.add_function(wrap_pyfunction!(cyclic_difference_penalty, module)?)?;
    module.add_function(wrap_pyfunction!(duchon_basis_with_jet, module)?)?;
    module.add_function(wrap_pyfunction!(duchon_basis_with_jets, module)?)?;
    module.add_function(wrap_pyfunction!(duchon_basis, module)?)?;
    module.add_function(wrap_pyfunction!(matern_basis, module)?)?;
    module.add_function(wrap_pyfunction!(smoothness_penalty, module)?)?;
    module.add_function(wrap_pyfunction!(duchon_function_norm_penalty, module)?)?;
    module.add_function(wrap_pyfunction!(duchon_operator_penalties, module)?)?;
    module.add_function(wrap_pyfunction!(sphere_basis, module)?)?;
    module.add_function(wrap_pyfunction!(sphere_basis_with_centers, module)?)?;
    module.add_function(wrap_pyfunction!(
        sphere_select_farthest_point_centers,
        module
    )?)?;
    module.add_function(wrap_pyfunction!(sphere_chart_basis_with_jet, module)?)?;
    module.add_function(wrap_pyfunction!(sphere_basis_jet, module)?)?;
    module.add_function(wrap_pyfunction!(sphere_basis_jet_with_centers, module)?)?;
    module.add_function(wrap_pyfunction!(thin_plate_penalty, module)?)?;
    module.add_function(wrap_pyfunction!(auto_knots_1d, module)?)?;
    module.add_function(wrap_pyfunction!(auto_centers_1d, module)?)?;
    module.add_function(wrap_pyfunction!(_block_diag, module)?)?;
    module.add_function(wrap_pyfunction!(gaussian_weighted_ridge_array, module)?)?;
    module.add_function(wrap_pyfunction!(gaussian_weighted_ridge_batch, module)?)?;
    module.add_function(wrap_pyfunction!(
        gaussian_weighted_ridge_batch_backward,
        module
    )?)?;
    module.add_function(wrap_pyfunction!(gaussian_reml_score, module)?)?;
    module.add_function(wrap_pyfunction!(skip_transcoder_reml_metrics, module)?)?;
    module.add_function(wrap_pyfunction!(skip_transcoder_select_reml, module)?)?;
    module.add_function(wrap_pyfunction!(tierney_kadane_normalized_score, module)?)?;
    module.add_function(wrap_pyfunction!(topology_bic_score, module)?)?;
    module.add_function(wrap_pyfunction!(torch_smooth_dispatch_key, module)?)?;
    module.add_function(wrap_pyfunction!(assemble_candidate_formula, module)?)?;
    module.add_function(wrap_pyfunction!(ordered_prediction_columns, module)?)?;
    module.add_function(wrap_pyfunction!(rank_topology_candidates, module)?)?;
    module.add_function(wrap_pyfunction!(stacking_weights_from_log_density, module)?)?;
    module.add_function(wrap_pyfunction!(stack_topologies_gaussian, module)?)?;
    module.add_function(wrap_pyfunction!(extract_reml_score, module)?)?;
    module.add_function(wrap_pyfunction!(extract_reml_score_raw, module)?)?;
    module.add_function(wrap_pyfunction!(extract_reml_edf, module)?)?;
    module.add_function(wrap_pyfunction!(compare_reml_fits, module)?)?;
    module.add_function(wrap_pyfunction!(gaussian_reml_fit, module)?)?;
    module.add_function(wrap_pyfunction!(gaussian_reml_fit_backward, module)?)?;
    module.add_function(wrap_pyfunction!(gaussian_reml_fit_formula_table, module)?)?;
    module.add_function(wrap_pyfunction!(gaussian_reml_fit_blocks_forward, module)?)?;
    module.add_function(wrap_pyfunction!(
        gaussian_reml_fit_blocks_orthogonal_forward,
        module
    )?)?;
    module.add_function(wrap_pyfunction!(gaussian_reml_fit_blocks_backward, module)?)?;
    module.add_function(wrap_pyfunction!(
        gaussian_reml_fit_with_constraints_forward,
        module
    )?)?;
    module.add_function(wrap_pyfunction!(
        gaussian_reml_fit_with_constraints_backward,
        module
    )?)?;
    module.add_function(wrap_pyfunction!(gaussian_reml_fit_batched, module)?)?;
    module.add_function(wrap_pyfunction!(
        gaussian_reml_fit_batched_backward,
        module
    )?)?;
    module.add_function(wrap_pyfunction!(gaussian_reml_fit_positions, module)?)?;
    module.add_function(wrap_pyfunction!(
        gaussian_reml_fit_positions_backward,
        module
    )?)?;
    module.add_function(wrap_pyfunction!(
        gaussian_reml_fit_positions_batched,
        module
    )?)?;
    module.add_function(wrap_pyfunction!(
        gaussian_reml_fit_positions_batched_backward,
        module
    )?)?;
    module.add_function(wrap_pyfunction!(gaussian_reml_fit_latent, module)?)?;
    module.add_function(wrap_pyfunction!(gaussian_reml_optimize_latent, module)?)?;
    module.add_function(wrap_pyfunction!(register_analytic_penalties, module)?)?;
    module.add_function(wrap_pyfunction!(analytic_penalty_value_grad, module)?)?;
    module.add_function(wrap_pyfunction!(analytic_penalty_hvp, module)?)?;
    module.add_function(wrap_pyfunction!(
        harmonic_roughness_evidence_weight,
        module
    )?)?;
    module.add_function(wrap_pyfunction!(gumbel_schedule_tau, module)?)?;
    module.add_function(wrap_pyfunction!(sae_ibp_map_value_grad, module)?)?;
    module.add_function(wrap_pyfunction!(sae_jumprelu_row_value_grad, module)?)?;
    module.add_function(wrap_pyfunction!(sae_jumprelu_batch_value_grad, module)?)?;
    module.add_function(wrap_pyfunction!(sae_topk_activation_value_grad, module)?)?;
    module.add_function(wrap_pyfunction!(jumprelu_gate_value_grad, module)?)?;
    module.add_function(wrap_pyfunction!(equivariant_penalty_value, module)?)?;
    module.add_function(wrap_pyfunction!(riemannian_retract, module)?)?;
    module.add_function(wrap_pyfunction!(manifold_exp_map, module)?)?;
    module.add_function(wrap_pyfunction!(manifold_exp_map_vjp, module)?)?;
    module.add_function(wrap_pyfunction!(manifold_log_map, module)?)?;
    module.add_function(wrap_pyfunction!(manifold_metric_tensor, module)?)?;
    module.add_function(wrap_pyfunction!(manifold_dimension, module)?)?;
    module.add_function(wrap_pyfunction!(manifold_ambient_dimension, module)?)?;
    module.add_function(wrap_pyfunction!(periodic_harmonic_basis, module)?)?;
    module.add_function(wrap_pyfunction!(
        periodic_harmonic_basis_derivative,
        module
    )?)?;
    module.add_function(wrap_pyfunction!(sphere_frechet_mean, module)?)?;
    module.add_function(wrap_pyfunction!(response_geometry_closure, module)?)?;
    module.add_function(wrap_pyfunction!(response_geometry_clr, module)?)?;
    module.add_function(wrap_pyfunction!(response_geometry_alr, module)?)?;
    module.add_function(wrap_pyfunction!(response_geometry_inverse_alr, module)?)?;
    module.add_function(wrap_pyfunction!(
        response_geometry_simplex_frechet_mean,
        module
    )?)?;
    module.add_function(wrap_pyfunction!(poincare_mobius_add, module)?)?;
    module.add_function(wrap_pyfunction!(poincare_distance, module)?)?;
    module.add_function(wrap_pyfunction!(poincare_project_into_ball, module)?)?;
    module.add_function(wrap_pyfunction!(poincare_log_origin, module)?)?;
    module.add_function(wrap_pyfunction!(poincare_exp_origin, module)?)?;
    module.add_function(wrap_pyfunction!(poincare_conformal_factor, module)?)?;
    module.add_function(wrap_pyfunction!(poincare_exp_map, module)?)?;
    module.add_function(wrap_pyfunction!(poincare_log_map, module)?)?;
    module.add_function(wrap_pyfunction!(poincare_to_lorentz, module)?)?;
    module.add_function(wrap_pyfunction!(poincare_from_lorentz, module)?)?;
    module.add_function(wrap_pyfunction!(poincare_lorentz_log_origin, module)?)?;
    module.add_function(wrap_pyfunction!(poincare_lorentz_exp_origin, module)?)?;
    module.add_function(wrap_pyfunction!(poincare_tangent_decode_forward, module)?)?;
    module.add_function(wrap_pyfunction!(poincare_tangent_decode_backward, module)?)?;
    module.add_function(wrap_pyfunction!(poincare_lorentz_decode_forward, module)?)?;
    module.add_function(wrap_pyfunction!(poincare_lorentz_decode_backward, module)?)?;
    module.add_function(wrap_pyfunction!(response_geometry_simplex_log_map, module)?)?;
    module.add_function(wrap_pyfunction!(response_geometry_simplex_exp_map, module)?)?;
    module.add_function(wrap_pyfunction!(response_geometry_sphere_log_map, module)?)?;
    module.add_function(wrap_pyfunction!(response_geometry_sphere_exp_map, module)?)?;
    module.add_function(wrap_pyfunction!(response_geometry_ilr, module)?)?;
    module.add_function(wrap_pyfunction!(response_geometry_inverse_ilr, module)?)?;
    module.add_function(wrap_pyfunction!(
        response_geometry_aitchison_metric,
        module
    )?)?;
    module.add_function(wrap_pyfunction!(response_geometry_clr_jet, module)?)?;
    module.add_function(wrap_pyfunction!(
        response_geometry_simplex_log_map_jet,
        module
    )?)?;
    module.add_function(wrap_pyfunction!(
        response_geometry_simplex_exp_map_jet,
        module
    )?)?;
    module.add_function(wrap_pyfunction!(
        response_geometry_sphere_exp_map_jet,
        module
    )?)?;
    module.add_function(wrap_pyfunction!(response_geometry_log_map, module)?)?;
    module.add_function(wrap_pyfunction!(response_geometry_exp_map, module)?)?;
    module.add_function(wrap_pyfunction!(response_geometry_fit_curvature, module)?)?;
    module.add_function(wrap_pyfunction!(
        response_geometry_sphere_normalize_base,
        module
    )?)?;
    module.add_function(wrap_pyfunction!(sae_duchon_centers_nd, module)?)?;
    module.add_function(wrap_pyfunction!(sae_sinkhorn_balance_bias, module)?)?;
    module.add_function(wrap_pyfunction!(sae_assign_ema_update, module)?)?;
    module.add_function(wrap_pyfunction!(sae_residual_em_score, module)?)?;
    module.add_function(wrap_pyfunction!(sae_residual_em_score_vjp, module)?)?;
    module.add_function(wrap_pyfunction!(sae_direction_cluster_anchor, module)?)?;
    module.add_function(wrap_pyfunction!(sae_quadratic_subspace_anchor, module)?)?;
    module.add_function(wrap_pyfunction!(sae_apply_anchor_rule, module)?)?;
    module.add_function(wrap_pyfunction!(sae_matching_pursuit_commit, module)?)?;
    module.add_function(wrap_pyfunction!(sae_solve_chart_coordinates, module)?)?;
    module.add_function(wrap_pyfunction!(sae_row_trust_scores, module)?)?;
    module.add_function(wrap_pyfunction!(sae_position_alignment_penalty, module)?)?;
    module.add_function(wrap_pyfunction!(sae_periodic_basis_with_jet_cuda, module)?)?;
    module.add_function(wrap_pyfunction!(sae_duchon_device_basis_width, module)?)?;
    module.add_function(wrap_pyfunction!(sae_duchon_basis_with_jet_cuda, module)?)?;
    module.add_function(wrap_pyfunction!(sinkhorn_circular_cost, module)?)?;
    module.add_function(wrap_pyfunction!(sinkhorn_euclidean_cost, module)?)?;
    module.add_function(wrap_pyfunction!(sinkhorn_geodesic_sphere_cost, module)?)?;
    module.add_function(wrap_pyfunction!(sinkhorn_barycenter_forward, module)?)?;
    module.add_function(wrap_pyfunction!(sinkhorn_barycenter_vjp, module)?)?;
    module.add_function(wrap_pyfunction!(numerics_sigmoid_stable, module)?)?;
    module.add_function(wrap_pyfunction!(numerics_inverse_softplus, module)?)?;
    module.add_function(wrap_pyfunction!(
        response_geometry_normalize_fisher_rao,
        module
    )?)?;
    module.add_function(wrap_pyfunction!(equivariant_rho, module)?)?;
    module.add_function(wrap_pyfunction!(equivariant_aux_enabled, module)?)?;
    module.add_function(wrap_pyfunction!(equivariant_rho_so2, module)?)?;
    module.add_function(wrap_pyfunction!(equivariant_rho_so2_jvp, module)?)?;
    module.add_function(wrap_pyfunction!(equivariant_rho_so3, module)?)?;
    module.add_function(wrap_pyfunction!(equivariant_rho_so3_jvp, module)?)?;
    module.add_function(wrap_pyfunction!(equivariant_gauge_companion_loss, module)?)?;
    module.add_function(wrap_pyfunction!(
        sae_default_ibp_concentration_for_k_atoms,
        module
    )?)?;
    module.add_function(wrap_pyfunction!(
        sae_default_top_k_for_large_dictionary,
        module
    )?)?;
    module.add_function(wrap_pyfunction!(sae_select_k, module)?)?;
    module.add_function(wrap_pyfunction!(sae_auto_k_recommendation, module)?)?;
    module.add_function(wrap_pyfunction!(sae_manifold_description_length, module)?)?;
    module.add_function(wrap_pyfunction!(sae_eq4_description_length, module)?)?;
    module.add_function(wrap_pyfunction!(sae_eq4_water_fill_component_bits, module)?)?;
    module.add_function(wrap_pyfunction!(sae_manifold_fit, module)?)?;
    module.add_function(wrap_pyfunction!(sae_manifold_fit_stagewise, module)?)?;
    module.add_function(wrap_pyfunction!(sae_manifold_fit_ibp, module)?)?;
    module.add_function(wrap_pyfunction!(sae_manifold_fit_minimal, module)?)?;
    module.add_function(wrap_pyfunction!(sae_manifold_predict_oos, module)?)?;
    module.add_function(wrap_pyfunction!(build_sae_encode_atlas, module)?)?;
    module.add_class::<PySaeEncodeAtlas>()?;
    module.add_class::<PySaeAmortizedEncoder>()?;
    module.add_function(wrap_pyfunction!(sae_steer_delta, module)?)?;
    module.add_function(wrap_pyfunction!(sae_manifold_reconstruction_r2, module)?)?;
    module.add_function(wrap_pyfunction!(sae_streaming_plan, module)?)?;
    module.add_function(wrap_pyfunction!(layer_transport_fit, module)?)?;
    module.add_function(wrap_pyfunction!(layer_transport_ladder, module)?)?;
    module.add_class::<PyFittedTransport>()?;
    module.add_function(wrap_pyfunction!(fit_transport, module)?)?;
    module.add_function(wrap_pyfunction!(chart_transfer_operator, module)?)?;
    module.add_function(wrap_pyfunction!(certify_chart_transfer, module)?)?;
    module.add_function(wrap_pyfunction!(sae_checkpoint_dynamics, module)?)?;
    module.add_function(wrap_pyfunction!(intervention_eval_forever_mask, module)?)?;
    inference_instruments::register(module)?;
    module.add_function(wrap_pyfunction!(sae_manifold_assignment_summary, module)?)?;
    module.add_function(wrap_pyfunction!(gated_sae_decode, module)?)?;
    module.add_function(wrap_pyfunction!(interchange_decode_forward, module)?)?;
    module.add_function(wrap_pyfunction!(interchange_decode_backward, module)?)?;
    module.add_function(wrap_pyfunction!(interchange_swap_forward, module)?)?;
    module.add_function(wrap_pyfunction!(interchange_swap_backward, module)?)?;
    module.add_function(wrap_pyfunction!(gaussian_reml_fit_latent_backward, module)?)?;
    module.add_function(wrap_pyfunction!(glm_reml_fit_latent, module)?)?;
    module.add_function(wrap_pyfunction!(glm_reml_fit_latent_backward, module)?)?;
    module.add_function(wrap_pyfunction!(posterior_predict_table, module)?)?;
    module.add_function(wrap_pyfunction!(posterior_predict_bands_table, module)?)?;
    module.add_function(wrap_pyfunction!(posterior_eta_bands, module)?)?;
    module.add_function(wrap_pyfunction!(posterior_credible_interval, module)?)?;
    module.add_function(wrap_pyfunction!(posterior_coefficient_names_json, module)?)?;
    module.add_function(wrap_pyfunction!(posterior_samples_summary_json, module)?)?;
    module.add_function(wrap_pyfunction!(posterior_trace_selection_json, module)?)?;
    module.add_function(wrap_pyfunction!(apply_inverse_link_array, module)?)?;
    module.add_function(wrap_pyfunction!(summary_json, module)?)?;
    module.add_function(wrap_pyfunction!(curvature_inference_json, module)?)?;
    module.add_function(wrap_pyfunction!(smooth_term_lr_inference_json, module)?)?;
    module.add_function(wrap_pyfunction!(model_debiased_functional_json, module)?)?;
    module.add_function(wrap_pyfunction!(summary_payload_from_model, module)?)?;
    module.add_function(wrap_pyfunction!(smoothing_parameters_from_model, module)?)?;
    module.add_function(wrap_pyfunction!(model_group_metadata, module)?)?;
    module.add_function(wrap_pyfunction!(model_deployment_extensions, module)?)?;
    module.add_function(wrap_pyfunction!(model_evidence, module)?)?;
    module.add_function(wrap_pyfunction!(summary_repr, module)?)?;
    module.add_function(wrap_pyfunction!(summary_html, module)?)?;
    module.add_function(wrap_pyfunction!(coefficient_state_json, module)?)?;
    module.add_function(wrap_pyfunction!(term_blocks_for_model, module)?)?;
    module.add_function(wrap_pyfunction!(model_partial_dependence, module)?)?;
    module.add_function(wrap_pyfunction!(model_variance_share, module)?)?;
    module.add_function(wrap_pyfunction!(difference_smooth_json, module)?)?;
    module.add_function(wrap_pyfunction!(difference_smooth_rows, module)?)?;
    module.add_function(wrap_pyfunction!(
        cross_fit_shared_precision_groups_json,
        module
    )?)?;
    module.add_function(wrap_pyfunction!(check_json, module)?)?;
    module.add_function(wrap_pyfunction!(check_payload_from_model, module)?)?;
    module.add_function(wrap_pyfunction!(report_html, module)?)?;
    module.add_function(wrap_pyfunction!(compute_residuals, module)?)?;
    module.add_function(wrap_pyfunction!(diagnostics_from_predictions, module)?)?;
    module.add_function(wrap_pyfunction!(benchmark_prediction_metrics, module)?)?;
    module.add_function(wrap_pyfunction!(auc_from_predictions, module)?)?;
    module.add_function(wrap_pyfunction!(weighted_auc_from_predictions, module)?)?;
    module.add_function(wrap_pyfunction!(brier_from_predictions, module)?)?;
    module.add_function(wrap_pyfunction!(log_loss_from_predictions, module)?)?;
    module.add_function(wrap_pyfunction!(nagelkerke_r2_from_predictions, module)?)?;
    module.add_function(wrap_pyfunction!(make_folds_indices, module)?)?;
    module.add_function(wrap_pyfunction!(
        gaussian_log_loss_from_predictions,
        module
    )?)?;
    module.add_function(wrap_pyfunction!(
        gaussian_prediction_scores_from_predictions,
        module
    )?)?;
    module.add_function(wrap_pyfunction!(zscore_train_test_arrays, module)?)?;
    module.add_function(wrap_pyfunction!(classification_metrics, module)?)?;
    module.add_function(wrap_pyfunction!(survival_concordance, module)?)?;
    module.add_function(wrap_pyfunction!(survival_score_grid_from_times, module)?)?;
    module.add_function(wrap_pyfunction!(repeat_survival_curve, module)?)?;
    module.add_function(wrap_pyfunction!(survival_null_curve_from_train, module)?)?;
    module.add_function(wrap_pyfunction!(
        survival_matrix_from_risk_calibration,
        module
    )?)?;
    module.add_function(wrap_pyfunction!(
        survival_lifted_metrics_from_predictions,
        module
    )?)?;
    module.add_function(wrap_pyfunction!(synthetic_binomial_columns, module)?)?;
    module.add_function(wrap_pyfunction!(synthetic_geo_disease_columns, module)?)?;
    module.add_function(wrap_pyfunction!(
        synthetic_continuous_order_columns,
        module
    )?)?;
    module.add_function(wrap_pyfunction!(
        synthetic_thread3_admixture_cliff_columns,
        module
    )?)?;
    module.add_function(wrap_pyfunction!(synthetic_geo_disease_eas_columns, module)?)?;
    module.add_function(wrap_pyfunction!(synthetic_papuan_oce_columns, module)?)?;
    module.add_function(wrap_pyfunction!(synthetic_hgdp_pc_panel_columns, module)?)?;
    module.add_function(wrap_pyfunction!(synthetic_geo_subpop_response, module)?)?;
    module.add_function(wrap_pyfunction!(synthetic_geo_latlon_response, module)?)?;
    module.add_function(wrap_pyfunction!(thread3_cliff_gradient_magnitude, module)?)?;
    // LatentCoord input-location derivative helpers (one per basis kind).
    // The Duchon descriptor differentiates the *built* design (kernel-nullspace
    // projection + polynomial columns + amplification) via `duchon_basis_with_jets`,
    // not the raw centerwise radial kernel, so no Duchon raw-radial FFI is
    // registered here. The Matérn descriptor has no nullspace projection /
    // polynomial tail, so its input-location jet and Hessian are the full
    // metric-aware kernel derivatives (anisotropy threaded through, matching the
    // forward design exactly — issue #437); the assembly lives in Rust, Python
    // only contracts the returned tensors with upstream gradients.
    module.add_function(wrap_pyfunction!(matern_input_location_first_jet, module)?)?;
    module.add_function(wrap_pyfunction!(matern_input_location_hessian, module)?)?;
    module.add_function(wrap_pyfunction!(matern_basis_gradient_streaming, module)?)?;
    module.add_function(wrap_pyfunction!(
        sphere_input_location_first_derivative,
        module
    )?)?;
    module.add_function(wrap_pyfunction!(
        periodic_bspline_input_location_first_derivative,
        module
    )?)?;
    module.add_function(wrap_pyfunction!(
        bspline_tensor_input_location_first_derivative,
        module
    )?)?;
    module.add_function(wrap_pyfunction!(mechanism_sparsity_jacobian, module)?)?;
    module.add_function(wrap_pyfunction!(derive_ivae_aux_scale, module)?)?;
    module.add_function(wrap_pyfunction!(conditional_prior_ivae, module)?)?;
    module.add_function(wrap_pyfunction!(diagnostics_aux_richness, module)?)?;
    module.add_function(wrap_pyfunction!(diagnostics_jacobian_sparsity, module)?)?;
    module.add_function(wrap_pyfunction!(diagnostics_anchor_consistency, module)?)?;
    module.add_function(wrap_pyfunction!(diagnostics_concat_decoder_blocks, module)?)?;
    module.add_function(wrap_pyfunction!(partial_supervision_solve, module)?)?;
    module.add_function(wrap_pyfunction!(thin_svd_scores, module)?)?;
    module.add_function(wrap_pyfunction!(linear_dictionary_fit, module)?)?;
    module.add_function(wrap_pyfunction!(linear_dictionary_transform_ffi, module)?)?;
    module.add_function(wrap_pyfunction!(sae_fit_admission, module)?)?;
    module.add_function(wrap_pyfunction!(sparse_dictionary_fit, module)?)?;
    module.add_function(wrap_pyfunction!(sparse_dictionary_transform_ffi, module)?)?;
    module.add_function(wrap_pyfunction!(sparse_dictionary_reconstruct_ffi, module)?)?;
    module.add_function(wrap_pyfunction!(sae_manifold_reconstruct_ffi, module)?)?;
    module.add_function(wrap_pyfunction!(sae_manifold_steer_rows, module)?)?;
    module.add_function(wrap_pyfunction!(block_sparse_dictionary_fit, module)?)?;
    module.add_function(wrap_pyfunction!(
        block_sparse_dictionary_transform_ffi,
        module
    )?)?;
    module.add_function(wrap_pyfunction!(
        block_sparse_dictionary_reconstruct_ffi,
        module
    )?)?;
    module.add_function(wrap_pyfunction!(
        block_sparse_dictionary_block_coords_ffi,
        module
    )?)?;
    module.add_function(wrap_pyfunction!(
        block_sparse_dictionary_lift_block_ffi,
        module
    )?)?;
    module.add_function(wrap_pyfunction!(
        block_sparse_dictionary_project_residual_ffi,
        module
    )?)?;
    module.add_function(wrap_pyfunction!(
        block_sparse_dictionary_firings_ffi,
        module
    )?)?;
    module.add_function(wrap_pyfunction!(
        block_sparse_dictionary_seed_manifest_ffi,
        module
    )?)?;
    module.add_function(wrap_pyfunction!(
        block_coordinate_chart_compose_ffi,
        module
    )?)?;
    module.add_function(wrap_pyfunction!(rank_charge_dof, module)?)?;
    module.add_class::<SparseDictStream>()?;
    module.add_class::<BlockSparseDictStream>()?;
    module.add_function(wrap_pyfunction!(
        identifiable_factor_select_weights_array,
        module
    )?)?;
    module.add_class::<IsometryPenalty>()?;
    module.add_class::<SparsityPenalty>()?;
    module.add_class::<PyTopKActivationPenalty>()?;
    module.add_class::<JumpReLUPenalty>()?;
    module.add_class::<ARDPenalty>()?;
    module.add_class::<AuxConditionalPriorPenalty>()?;
    module.add_class::<IvaeRidgeMeanGauge>()?;
    module.add_class::<ParametricAuxConditionalPriorPenalty>()?;
    module.add_class::<BlockSparsityPenalty>()?;
    module.add_class::<BlockOrthogonalityPenalty>()?;
    module.add_class::<OrthogonalityPenalty>()?;
    module.add_class::<SheafConsistencyPenalty>()?;
    module.add_class::<SoftmaxAssignmentSparsityPenalty>()?;
    module.add_class::<PyIBPAssignmentPenalty>()?;
    module.add_class::<ScadMcpPenalty>()?;
    module.add_class::<TotalVariationPenalty>()?;
    module.add_class::<NuclearNormPenalty>()?;
    module.add_class::<MechanismSparsityPenalty>()?;
    module.add_function(wrap_pyfunction!(dimension_spectrometer, module)?)?;
    module.add_function(wrap_pyfunction!(block_firing_coordinates, module)?)?;
    module.add_function(wrap_pyfunction!(routability_floor, module)?)?;
    module.add_function(wrap_pyfunction!(routability_audit, module)?)?;
    module.add_function(wrap_pyfunction!(sparse_dict_dual_certificate, module)?)?;
    module.add_function(wrap_pyfunction!(audit_sae, module)?)?;
    module.add_function(wrap_pyfunction!(atlas_nerve_diagram, module)?)?;
    module.add_function(wrap_pyfunction!(separation_limit, module)?)?;
    module.add_function(wrap_pyfunction!(recover_spikes, module)?)?;
    module.add_function(wrap_pyfunction!(compose_contracts, module)?)?;
    module.add_function(wrap_pyfunction!(loop_holonomy, module)?)?;
    module.add_function(wrap_pyfunction!(
        conditional_coactivation_influence,
        module
    )?)?;
    module.add_function(wrap_pyfunction!(coupling_robustness_certificate, module)?)?;
    module.add_function(wrap_pyfunction!(effect_weighted_retention, module)?)?;
    module.add_function(wrap_pyfunction!(chart_interp_score, module)?)?;
    module.add_function(wrap_pyfunction!(dose_response_calibration, module)?)?;
    module.add_function(wrap_pyfunction!(
        coordinate_posterior_from_precision,
        module
    )?)?;
    Ok(())
}

/// Smoothed column-2-norm penalty on a decoder weight matrix W of shape
/// `(d_obs, k_latent)`. Returns `(value, grad)` where `grad` has the same
/// shape as `W`.
#[pyfunction(signature = (weight, epsilon, w))]
fn mechanism_sparsity_jacobian<'py>(
    py: Python<'py>,
    weight: f64,
    epsilon: f64,
    w: PyReadonlyArray2<'py, f64>,
) -> PyResult<(f64, Py<PyArray2<f64>>)> {
    let pen = gam::terms::sae::identifiability::MechanismSparsityJacobian::new(weight, epsilon)
        .map_err(py_value_error)?;
    let (value, grad) = pen.value_and_grad(w.as_array());
    Ok((value, grad.into_pyarray(py).unbind()))
}

/// Derive the iVAE auxiliary-conditional scale σ(u) from the auxiliary table.
#[pyfunction(signature = (aux, log_amplitude, frequency_scale))]
fn derive_ivae_aux_scale<'py>(
    py: Python<'py>,
    aux: PyReadonlyArray2<'py, f64>,
    log_amplitude: f64,
    frequency_scale: f64,
) -> PyResult<Py<PyArray2<f64>>> {
    let scale = gam::terms::sae::identifiability::derive_ivae_aux_scale(
        aux.as_array(),
        log_amplitude,
        frequency_scale,
    );
    Ok(scale.into_pyarray(py).unbind())
}

/// iVAE conditional-Gaussian log-prior. Given per-row `(mean, scale)` arrays
/// of shape `(n_rows, latent_dim)`, returns the negative log-prior value and
/// its gradient w.r.t. `t` (same shape as `t`).
#[pyfunction(signature = (weight, t, mean, scale))]
fn conditional_prior_ivae<'py>(
    py: Python<'py>,
    weight: f64,
    t: PyReadonlyArray2<'py, f64>,
    mean: PyReadonlyArray2<'py, f64>,
    scale: PyReadonlyArray2<'py, f64>,
) -> PyResult<(f64, Py<PyArray2<f64>>)> {
    let pen = gam::terms::sae::identifiability::ConditionalPriorIvae::new(
        mean.as_array().to_owned(),
        scale.as_array().to_owned(),
        weight,
    )
    .map_err(py_value_error)?;
    let (value, grad) = pen.value_and_grad(t.as_array());
    Ok((value, grad.into_pyarray(py).unbind()))
}

/// iVAE auxiliary-richness metrics (Khemakhem et al. 2020 Theorem 1).
///
/// Returns a dict with keys:
/// `aux_observed` (bool), `n_nonfinite_aux` (int), `aux_dim` (int),
/// `latent_dim` (int), `n_rows` (int), `constant_columns` (list[int]),
/// `aux_is_discrete` (bool), `n_distinct_levels` (int),
/// `jacobian_rank` (int), `jacobian_rank_estimated` (bool).
#[pyfunction(signature = (aux, latents))]
fn diagnostics_aux_richness<'py>(
    py: Python<'py>,
    aux: PyReadonlyArray2<'py, f64>,
    latents: PyReadonlyArray2<'py, f64>,
) -> PyResult<Py<PyDict>> {
    let m = gam::identifiability::kernel::aux_richness_metrics(aux.as_array(), latents.as_array());
    let dict = PyDict::new(py);
    dict.set_item("aux_observed", m.aux_observed)?;
    dict.set_item("n_nonfinite_aux", m.n_nonfinite_aux)?;
    dict.set_item("aux_dim", m.aux_dim)?;
    dict.set_item("latent_dim", m.latent_dim)?;
    dict.set_item("n_rows", m.n_rows)?;
    dict.set_item("constant_columns", m.constant_columns)?;
    dict.set_item("aux_is_discrete", m.aux_is_discrete)?;
    dict.set_item("n_distinct_levels", m.n_distinct_levels)?;
    // Map usize::MAX sentinel to Python None when rank not estimated.
    if m.jacobian_rank_estimated {
        dict.set_item("jacobian_rank", m.jacobian_rank)?;
    } else {
        dict.set_item("jacobian_rank", py.None())?;
    }
    dict.set_item("jacobian_rank_estimated", m.jacobian_rank_estimated)?;
    Ok(dict.unbind())
}

/// Decoder-Jacobian sparsity metrics (Hyvarinen-Morioka 2017; Lachapelle 2024).
///
/// `jacobians_flat` has shape `(n_samples * P, latent_dim)`. Returns a dict
/// with keys: `n_samples`, `p_features`, `latent_dim`, `mean_sparsity`,
/// `max_abs`, `ranks` (list[int] of length n_samples).
#[pyfunction(signature = (jacobians_flat, n_samples, zero_threshold))]
fn diagnostics_jacobian_sparsity<'py>(
    py: Python<'py>,
    jacobians_flat: PyReadonlyArray2<'py, f64>,
    n_samples: usize,
    zero_threshold: f64,
) -> PyResult<Py<PyDict>> {
    if n_samples == 0 {
        return Err(py_value_error(
            "diagnostics_jacobian_sparsity: n_samples must be >= 1".into(),
        ));
    }
    let (rows, _) = jacobians_flat.as_array().dim();
    if rows % n_samples != 0 {
        return Err(py_value_error(format!(
            "diagnostics_jacobian_sparsity: rows {} not divisible by n_samples {}",
            rows, n_samples
        )));
    }
    let m = gam::identifiability::kernel::jacobian_sparsity_metrics(
        jacobians_flat.as_array(),
        n_samples,
        zero_threshold,
    );
    let dict = PyDict::new(py);
    dict.set_item("n_samples", m.n_samples)?;
    dict.set_item("p_features", m.p_features)?;
    dict.set_item("latent_dim", m.latent_dim)?;
    dict.set_item("mean_sparsity", m.mean_sparsity)?;
    dict.set_item("max_abs", m.max_abs)?;
    dict.set_item("ranks", m.ranks)?;
    Ok(dict.unbind())
}

/// Anchor-consistency metrics for a manifold-SAE assignment matrix.
///
/// `assignments` has shape `(N, K)`. Returns a dict with keys:
/// `n_rows`, `n_atoms`, `n_anchors`, `anchors_per_atom` (list[int] length K).
#[pyfunction(signature = (assignments, anchor_dominance))]
fn diagnostics_anchor_consistency<'py>(
    py: Python<'py>,
    assignments: PyReadonlyArray2<'py, f64>,
    anchor_dominance: f64,
) -> PyResult<Py<PyDict>> {
    if !(anchor_dominance > 0.0 && anchor_dominance <= 1.0) {
        return Err(py_value_error(format!(
            "diagnostics_anchor_consistency: anchor_dominance must be in (0, 1]; got {}",
            anchor_dominance
        )));
    }
    let m = gam::identifiability::kernel::anchor_consistency_metrics(
        assignments.as_array(),
        anchor_dominance,
    );
    let dict = PyDict::new(py);
    dict.set_item("n_rows", m.n_rows)?;
    dict.set_item("n_atoms", m.n_atoms)?;
    dict.set_item("n_anchors", m.n_anchors)?;
    dict.set_item("anchors_per_atom", m.anchors_per_atom)?;
    Ok(dict.unbind())
}

/// Concatenate a list of `(basis_size_k, P)` per-atom decoder blocks into a
/// single `(P, sum_k basis_size_k)` Jacobian matrix. Saves a numpy
/// concatenation in the Python diagnostics dispatcher.
#[pyfunction(signature = (blocks))]
fn diagnostics_concat_decoder_blocks<'py>(
    py: Python<'py>,
    blocks: Vec<PyReadonlyArray2<'py, f64>>,
) -> PyResult<Py<PyArray2<f64>>> {
    let views: Vec<_> = blocks.iter().map(|b| b.as_array()).collect();
    let out =
        gam::identifiability::kernel::concat_decoder_blocks(&views).map_err(py_value_error)?;
    Ok(out.into_pyarray(py).unbind())
}

/// Generic 2D log-λ weight-selection driver.
///
/// Takes a precomputed `(G1, G2)` RSS grid and a matching penalty grid
/// together with the two 1D weight grids, computes the Laplace-style log
/// marginal-likelihood proxy on every cell, and returns the maximising
/// cell with deterministic tie-breaking. Useful for any two-penalty model
/// (identifiable-factor recipe, double-penalty smooths, IBP + sparsity).
///
/// Returns a dict with keys: ``best_i``, ``best_j``, ``best_lam1``,
/// ``best_lam2``, ``best_evidence``, ``evidence_grid``.
#[pyfunction(signature = (rss_grid, penalty_grid, lam1_grid, lam2_grid, n_obs))]
fn identifiable_factor_select_weights_array<'py>(
    py: Python<'py>,
    rss_grid: PyReadonlyArray2<'py, f64>,
    penalty_grid: PyReadonlyArray2<'py, f64>,
    lam1_grid: PyReadonlyArray1<'py, f64>,
    lam2_grid: PyReadonlyArray1<'py, f64>,
    n_obs: i64,
) -> PyResult<Py<PyDict>> {
    if n_obs <= 0 {
        return Err(py_value_error(format!(
            "identifiable_factor_select_weights_array: n_obs must be > 0, got {n_obs}"
        )));
    }
    let res = gam::terms::sae::identifiability::identifiable_factor_select_weights(
        rss_grid.as_array(),
        penalty_grid.as_array(),
        lam1_grid.as_array(),
        lam2_grid.as_array(),
        n_obs as usize,
    )
    .map_err(py_value_error)?;
    let out = PyDict::new(py);
    out.set_item("best_i", res.best_i)?;
    out.set_item("best_j", res.best_j)?;
    out.set_item("best_lam1", res.best_lam1)?;
    out.set_item("best_lam2", res.best_lam2)?;
    out.set_item("best_evidence", res.best_evidence)?;
    out.set_item("evidence_grid", res.evidence_grid.into_pyarray(py))?;
    Ok(out.unbind())
}

/// Partial-supervision gauge-fix solver.
///
/// Single-call FFI for the supervised + free latent-block gauge fix used by
/// `gamfit.examples.partial_supervision` (and reusable from the CLI / R /
/// Julia bindings). All linear-algebra work — orthogonal Procrustes via
/// SVD, anchor least-squares via SVD pseudo-inverse, soft-L2 ridge map via
/// symmetric eigendecomposition with a REML grid, and the orthogonal-
/// complement projection via thin QR — runs in Rust through the faer
/// bridge.
///
/// Parameters
/// ----------
/// t_sup, aux : (N, d_sup) f64 arrays
///     Initial supervised block and the auxiliary target it aligns to.
/// t_free : (N, d_free) f64 array
///     Free block; ``d_free`` may be zero.
/// method : {"procrustes", "anchor", "soft_l2"}
///     Supervised-block alignment method.
/// anchor_idx : Vec<i64>
///     Anchor rows for the ``anchor`` method (ignored otherwise).
/// free_constraint : {"orthogonal_to_sup", "none"}
///     Decorrelation rule for the free block.
///
/// Returns a dict with keys: ``t_supervised``, ``t_free``,
/// ``alignment_score``, ``selected_weight``, ``map_r``, ``map_a``,
/// ``map_b``. Values not produced by the chosen method are returned as
/// Python ``None``.
#[pyfunction(signature = (t_sup, aux, t_free, method, anchor_idx, free_constraint))]
fn partial_supervision_solve<'py>(
    py: Python<'py>,
    t_sup: PyReadonlyArray2<'py, f64>,
    aux: PyReadonlyArray2<'py, f64>,
    t_free: PyReadonlyArray2<'py, f64>,
    method: &str,
    anchor_idx: Vec<i64>,
    free_constraint: &str,
) -> PyResult<Py<PyDict>> {
    let method_enum = match method {
        "procrustes" => gam::terms::sae::identifiability::PartialSupervisionSupMethod::Procrustes,
        "anchor" => gam::terms::sae::identifiability::PartialSupervisionSupMethod::Anchor,
        "soft_l2" => gam::terms::sae::identifiability::PartialSupervisionSupMethod::SoftL2,
        other => {
            return Err(py_value_error(format!(
                "partial_supervision_solve: method must be 'procrustes', 'anchor' or \
                 'soft_l2'; got {other:?}"
            )));
        }
    };
    let free_enum = match free_constraint {
        "orthogonal_to_sup" => {
            gam::terms::sae::identifiability::PartialSupervisionFreeConstraint::OrthogonalToSup
        }
        "none" => gam::terms::sae::identifiability::PartialSupervisionFreeConstraint::None,
        other => {
            return Err(py_value_error(format!(
                "partial_supervision_solve: free_constraint must be 'orthogonal_to_sup' or \
                 'none'; got {other:?}"
            )));
        }
    };
    let mut anchor_usize: Vec<usize> = Vec::with_capacity(anchor_idx.len());
    for idx in &anchor_idx {
        if *idx < 0 {
            return Err(py_value_error(format!(
                "partial_supervision_solve: anchor_idx entries must be non-negative; got {idx}"
            )));
        }
        anchor_usize.push(*idx as usize);
    }
    let result = gam::terms::sae::identifiability::partial_supervision_solve(
        t_sup.as_array(),
        aux.as_array(),
        t_free.as_array(),
        method_enum,
        &anchor_usize,
        free_enum,
    )
    .map_err(py_value_error)?;
    let out = PyDict::new(py);
    out.set_item("t_supervised", result.t_supervised.into_pyarray(py))?;
    out.set_item("t_free", result.t_free.into_pyarray(py))?;
    out.set_item("alignment_score", result.alignment_score)?;
    match result.selected_weight {
        Some(v) => out.set_item("selected_weight", v)?,
        None => out.set_item("selected_weight", py.None())?,
    }
    match result.map_r {
        Some(arr) => out.set_item("map_r", arr.into_pyarray(py))?,
        None => out.set_item("map_r", py.None())?,
    }
    match result.map_a {
        Some(arr) => out.set_item("map_a", arr.into_pyarray(py))?,
        None => out.set_item("map_a", py.None())?,
    }
    match result.map_b {
        Some(arr) => out.set_item("map_b", arr.into_pyarray(py))?,
        None => out.set_item("map_b", py.None())?,
    }
    Ok(out.unbind())
}

/// Column-centred thin-SVD scores for the leading `k` components.
///
/// Returns `U[:, :k] * Σ[:k]` of `X − mean(X, axis=0)`, computed via the
/// faer SVD bridge. Used by the partial-supervision recipe to seed an
/// initial latent block when the caller does not provide one.
#[pyfunction(signature = (x, k))]
fn thin_svd_scores<'py>(
    py: Python<'py>,
    x: PyReadonlyArray2<'py, f64>,
    k: i64,
) -> PyResult<Py<PyArray2<f64>>> {
    if k < 0 {
        return Err(py_value_error(format!(
            "thin_svd_scores: k must be non-negative, got {k}"
        )));
    }
    let out = gam::terms::sae::identifiability::thin_svd_scores(x.as_array(), k as usize)
        .map_err(py_value_error)?;
    Ok(out.into_pyarray(py).unbind())
}

#[pyfunction(signature = (
    x,
    k,
    max_iter = 30,
    top_k = 1,
    assignment = "top_k",
    temperature = 0.25,
    code_ridge = 1.0e-8,
    tolerance = 1.0e-7,
    center_rank_one = false
))]
fn linear_dictionary_fit<'py>(
    py: Python<'py>,
    x: PyReadonlyArray2<'py, f64>,
    k: usize,
    max_iter: usize,
    top_k: usize,
    assignment: &str,
    temperature: f64,
    code_ridge: f64,
    tolerance: f64,
    center_rank_one: bool,
) -> PyResult<Py<PyDict>> {
    let x_values = x.as_array().to_owned();
    let assignment_kind = LinearDictionaryAssignment::parse(assignment).map_err(py_value_error)?;
    let config = LinearDictionaryConfig {
        n_atoms: k,
        max_iter,
        top_k,
        assignment: assignment_kind,
        temperature,
        code_ridge,
        tolerance,
        center_rank_one,
    };
    let fit = detach_py_result(py, "linear_dictionary_fit", move || {
        fit_linear_dictionary(x_values.view(), &config)
    })?;
    let out = PyDict::new(py);
    out.set_item("atoms", fit.atoms.into_pyarray(py))?;
    out.set_item("assignments", fit.assignments.into_pyarray(py))?;
    out.set_item("fitted", fit.fitted.into_pyarray(py))?;
    out.set_item("lambdas", fit.lambdas.into_pyarray(py))?;
    out.set_item("reml_scores", fit.reml_scores.into_pyarray(py))?;
    out.set_item("explained_variance", fit.explained_variance)?;
    out.set_item("iterations", fit.iterations)?;
    out.set_item("converged", fit.converged)?;
    out.set_item("assignment", fit.assignment.as_str())?;
    out.set_item("top_k", fit.top_k)?;
    Ok(out.unbind())
}

/// Out-of-sample encode: route held-out rows `x` (`M x P`) through a fitted
/// linear dictionary `atoms` (`K x P`) via the Rust top-`top_k` ridge solve,
/// returning the `(M, K)` code matrix.
#[pyfunction(signature = (x, atoms, top_k, code_ridge = 1.0e-8))]
fn linear_dictionary_transform_ffi<'py>(
    py: Python<'py>,
    x: PyReadonlyArray2<'py, f64>,
    atoms: PyReadonlyArray2<'py, f64>,
    top_k: usize,
    code_ridge: f64,
) -> PyResult<Py<PyArray2<f64>>> {
    let x_values = x.as_array().to_owned();
    let atoms_values = atoms.as_array().to_owned();
    let codes = detach_py_result(py, "linear_dictionary_transform", move || {
        linear_dictionary_transform(x_values.view(), atoms_values.view(), top_k, code_ridge)
    })?;
    Ok(codes.into_pyarray(py).unbind())
}

/// #1026 collapsed linear lane — fit a fixed-`K` **sparse, minibatched** linear
/// dictionary. Unlike `linear_dictionary_fit` (dense `N×K` assignments, full-`K`
/// scoring) this routes each row against the dictionary in `K`-tiles, keeps only
/// the top-`active` atoms, and returns fixed-width sparse routing
/// (`indices[N, active]`, `codes[N, active]`) so very large `K` stays tractable.
/// All heavy state is FP32. The exact manifold engine is untouched; this is an
/// additive path.
fn score_route_stats_dict<'py>(
    py: Python<'py>,
    stats: gam::terms::sae::sparse_dict::ScoreRouteStats,
) -> PyResult<Bound<'py, PyDict>> {
    let out = PyDict::new(py);
    out.set_item("minibatches", stats.minibatches)?;
    out.set_item("admitted_minibatches", stats.admitted_minibatches)?;
    out.set_item("device_minibatches", stats.device_minibatches)?;
    out.set_item("cpu_minibatches", stats.cpu_minibatches)?;
    out.set_item("score_elements", stats.score_elements.to_string())?;
    out.set_item("score_tiles", stats.score_tiles)?;
    out.set_item("peak_score_bytes", stats.peak_score_bytes)?;
    out.set_item("device_dtoh_bytes", stats.device_dtoh_bytes.to_string())?;
    out.set_item(
        "unfused_score_dtoh_bytes_avoided",
        stats.unfused_score_dtoh_bytes_avoided.to_string(),
    )?;
    out.set_item(
        "dot_flops_lower_bound",
        stats.dot_flops_lower_bound.to_string(),
    )?;
    Ok(out)
}

fn parse_sparse_dict_score_mode(score_mode: &str) -> PyResult<gam::gpu::GpuPolicy> {
    gam::gpu::GpuPolicy::parse(score_mode).ok_or_else(|| {
        py_value_error(format!(
            "sparse dictionary score_mode must be 'auto', 'required', or 'off'; got {score_mode:?}"
        ))
    })
}

/// Cap on the number of top strictly-improving birth candidates a sparse-dict
/// fit's dual certificate reports. A safety bound on the report size, not a
/// tuned selection knob (the birth threshold itself is the derived `1`).
const SPARSE_DICT_DUAL_CERT_MAX_BIRTHS: usize = 16;

// The `DualCertificateReport` → PyDict builder shared by this lane's fits and the
// SAE-spectral diagnostics lane lives once, as `dual_certificate_report_dict` in
// `latent/sae_spectral_ffi.rs`. Both files are `include!`d into the crate root
// (lib.rs) and share one namespace, so the entrypoints below reach it by bare
// name. Keeping a single definition avoids the duplicate crate-root free function
// that would otherwise make the crate (and the gamfit wheel) fail to compile.

#[pyfunction(signature = (
    x,
    k,
    active = 1,
    minibatch = 512,
    max_epochs = 30,
    score_tile = 4096,
    code_ridge = 1.0e-6,
    decoder_ridge = 1.0e-6,
    tolerance = 1.0e-6,
    score_mode = "auto"
))]
fn sparse_dictionary_fit<'py>(
    py: Python<'py>,
    x: PyReadonlyArray2<'py, f32>,
    k: usize,
    active: usize,
    minibatch: usize,
    max_epochs: usize,
    score_tile: usize,
    code_ridge: f32,
    decoder_ridge: f32,
    tolerance: f64,
    score_mode: &str,
) -> PyResult<Py<PyDict>> {
    let score_mode = parse_sparse_dict_score_mode(score_mode)?;
    let x_values = x.as_array().to_owned();
    let admission =
        gam::terms::sae::front_door::admit_sae_fit(x_values.nrows(), x_values.ncols(), k)
            .map_err(py_value_error)?;
    let config = SparseDictConfig {
        n_atoms: k,
        active,
        minibatch,
        max_epochs,
        score_tile,
        code_ridge,
        decoder_ridge,
        tolerance,
        score_mode,
    };
    let (fit, dual_cert) = detach_py_result(py, "sparse_dictionary_fit", move || {
        let fit = fit_sparse_dictionary(x_values.view(), &config)?;
        // Global-optimality dual certificate, computed inside the same detached
        // block from the fitted routing so every lane fit emits it.
        let dual_cert = gam::terms::sae::dual_certificate::sparse_dict_dual_certificate(
            x_values.view(),
            &fit,
            SPARSE_DICT_DUAL_CERT_MAX_BIRTHS,
        )?;
        Ok::<_, String>((fit, dual_cert))
    })?;
    let out = PyDict::new(py);
    let lane = match admission.lane {
        gam::terms::sae::front_door::SaeFitLane::DenseCertification => "dense_certification",
        gam::terms::sae::front_door::SaeFitLane::SparseCodes => "sparse_codes",
        // Unreachable from `admit_sae_fit` (the curved lane is TopK-only), but
        // the label is kept coherent with the admission FFI.
        gam::terms::sae::front_door::SaeFitLane::CurvedStreaming => "curved_streaming",
    };
    out.set_item("front_door_lane", lane)?;
    out.set_item("decoder", fit.decoder.into_pyarray(py))?;
    out.set_item("indices", fit.indices.into_pyarray(py))?;
    out.set_item("codes", fit.codes.into_pyarray(py))?;
    out.set_item("explained_variance", fit.explained_variance)?;
    out.set_item("epochs", fit.epochs)?;
    out.set_item("converged", fit.converged)?;
    out.set_item("active", fit.active)?;
    out.set_item(
        "score_route_stats",
        score_route_stats_dict(py, fit.score_route_stats)?,
    )?;
    out.set_item(
        "dual_certificate",
        dual_certificate_report_dict(py, &dual_cert)?,
    )?;
    Ok(out.unbind())
}

/// Out-of-sample encode: route held-out rows `x` (`M x P`, f32) through a fitted
/// sparse dictionary `decoder` (`K x P`) via the Rust tiled router + active-set
/// ridge solve, returning `(indices, codes)` each `M x active`.
#[pyfunction(signature = (
    x,
    decoder,
    active,
    score_tile = 4096,
    code_ridge = 1.0e-6,
    score_mode = "auto"
))]
fn sparse_dictionary_transform_ffi<'py>(
    py: Python<'py>,
    x: PyReadonlyArray2<'py, f32>,
    decoder: PyReadonlyArray2<'py, f32>,
    active: usize,
    score_tile: usize,
    code_ridge: f32,
    score_mode: &str,
) -> PyResult<Py<PyDict>> {
    let score_mode = parse_sparse_dict_score_mode(score_mode)?;
    let x_values = x.as_array().to_owned();
    let decoder_values = decoder.as_array().to_owned();
    let transform = detach_py_result(py, "sparse_dictionary_transform", move || {
        sparse_dictionary_transform_with_mode(
            x_values.view(),
            decoder_values.view(),
            active,
            score_tile,
            code_ridge,
            score_mode,
        )
    })?;
    let out = PyDict::new(py);
    out.set_item("indices", transform.indices.into_pyarray(py))?;
    out.set_item("codes", transform.codes.into_pyarray(py))?;
    out.set_item(
        "score_route_stats",
        score_route_stats_dict(py, transform.score_route_stats)?,
    )?;
    Ok(out.unbind())
}

#[pyfunction(signature = (decoder, indices, codes))]
fn sparse_dictionary_reconstruct_ffi<'py>(
    py: Python<'py>,
    decoder: PyReadonlyArray2<'py, f32>,
    indices: PyReadonlyArray2<'py, u32>,
    codes: PyReadonlyArray2<'py, f32>,
) -> PyResult<Py<PyArray2<f32>>> {
    let decoder_values = decoder.as_array().to_owned();
    let index_values = indices.as_array().to_owned();
    let code_values = codes.as_array().to_owned();
    let out = detach_py_result(py, "sparse_dictionary_reconstruct", move || {
        reconstruct_sparse_rows(
            decoder_values.view(),
            index_values.view(),
            code_values.view(),
        )
    })?;
    Ok(out.into_pyarray(py).unbind())
}

#[pyfunction(signature = (
    atom_basis,
    atom_dim,
    decoder_blocks,
    coords,
    assignments,
    p_out,
))]
fn sae_manifold_reconstruct_ffi<'py>(
    py: Python<'py>,
    atom_basis: Vec<String>,
    atom_dim: Vec<usize>,
    decoder_blocks: Vec<PyReadonlyArray2<'py, f64>>,
    coords: Vec<PyReadonlyArray2<'py, f64>>,
    assignments: PyReadonlyArray2<'py, f64>,
    p_out: usize,
) -> PyResult<Py<PyArray2<f64>>> {
    let basis_kinds = atom_basis
        .iter()
        .map(|name| sae_atom_basis_kind_from_str(name))
        .collect::<Vec<_>>();
    let decoder_values = decoder_blocks
        .iter()
        .map(|block| block.as_array().to_owned())
        .collect::<Vec<_>>();
    let coord_values = coords
        .iter()
        .map(|coord| coord.as_array().to_owned())
        .collect::<Vec<_>>();
    let assignment_values = assignments.as_array().to_owned();
    let out = detach_py_result(py, "sae_manifold_reconstruct", move || {
        let decoder_views = decoder_values
            .iter()
            .map(|block| block.view())
            .collect::<Vec<_>>();
        let coord_views = coord_values
            .iter()
            .map(|coord| coord.view())
            .collect::<Vec<_>>();
        gam::terms::sae::manifold::reconstruct_persisted_atom_set(
            &basis_kinds,
            &atom_dim,
            &decoder_views,
            &coord_views,
            assignment_values.view(),
            p_out,
        )
    })?;
    Ok(out.into_pyarray(py).unbind())
}

/// gam#2234 — on-manifold causal STEER of a persisted atom set: returns the
/// ambient steering DELTA `a·(Φ(t⊕δ)−Φ(t))·B_k` for the single atom `steer_atom`
/// on every row (shape `(n_rows, p_out)`), which the Python side adds to the
/// residual-stream activation. `delta` is the intrinsic chart-coordinate step
/// (radians / fraction-of-period per the atom's manifold); the group action
/// `⊕` is the atom's own manifold retraction (Circle phase add, Euclidean
/// translate, product blockwise). Thin marshalling around the single-sourced
/// [`gam::terms::sae::manifold::steer_persisted_atom_set`], the stateless
/// counterpart of `SaeManifoldTerm::steer_rows`, mirroring
/// [`sae_manifold_reconstruct_ffi`].
#[pyfunction(signature = (
    atom_basis,
    atom_dim,
    decoder_blocks,
    coords,
    assignments,
    p_out,
    steer_atom,
    delta,
))]
fn sae_manifold_steer_rows<'py>(
    py: Python<'py>,
    atom_basis: Vec<String>,
    atom_dim: Vec<usize>,
    decoder_blocks: Vec<PyReadonlyArray2<'py, f64>>,
    coords: Vec<PyReadonlyArray2<'py, f64>>,
    assignments: PyReadonlyArray2<'py, f64>,
    p_out: usize,
    steer_atom: usize,
    delta: PyReadonlyArray1<'py, f64>,
) -> PyResult<Py<PyArray2<f64>>> {
    let basis_kinds = atom_basis
        .iter()
        .map(|name| sae_atom_basis_kind_from_str(name))
        .collect::<Vec<_>>();
    let decoder_values = decoder_blocks
        .iter()
        .map(|block| block.as_array().to_owned())
        .collect::<Vec<_>>();
    let coord_values = coords
        .iter()
        .map(|coord| coord.as_array().to_owned())
        .collect::<Vec<_>>();
    let assignment_values = assignments.as_array().to_owned();
    let delta_values = delta.as_array().to_owned();
    let out = detach_py_result(py, "sae_manifold_steer_rows", move || {
        let decoder_views = decoder_values
            .iter()
            .map(|block| block.view())
            .collect::<Vec<_>>();
        let coord_views = coord_values
            .iter()
            .map(|coord| coord.view())
            .collect::<Vec<_>>();
        gam::terms::sae::manifold::steer_persisted_atom_set(
            &basis_kinds,
            &atom_dim,
            &decoder_views,
            &coord_views,
            assignment_values.view(),
            p_out,
            steer_atom,
            delta_values.view(),
        )
    })?;
    Ok(out.into_pyarray(py).unbind())
}

/// #1026 block-sparse lane — fit a **block-sparse** dictionary: the `K = G·b`
/// atoms are grouped into `G` blocks of `b` orthonormal atoms, routing selects
/// whole blocks by their group ℓ₂ gate `‖z_g‖₂` (block-TopK, signed codes, no
/// ReLU), and each block is a Stiefel-constrained frame refreshed by polar steps.
/// Presence (`gates`) and amplitude (`codes`) are returned as separate arrays;
/// every selection is invariant to each block's internal `O(b)` gauge. Returns
/// per-block utilisation + stable rank for the MDL lane. Additive path; the atom
/// lane and the exact manifold engine are untouched.
#[pyfunction(signature = (
    x,
    n_blocks,
    block_size = 2,
    block_topk = 1,
    max_epochs = 30,
    minibatch = 512,
    block_tile = 1024,
    frame_ridge = 1.0e-9,
    aux_k = 0,
    matryoshka_prefix = false,
    tolerance = 1.0e-6
))]
fn block_sparse_dictionary_fit<'py>(
    py: Python<'py>,
    x: PyReadonlyArray2<'py, f32>,
    n_blocks: usize,
    block_size: usize,
    block_topk: usize,
    max_epochs: usize,
    minibatch: usize,
    block_tile: usize,
    frame_ridge: f64,
    aux_k: usize,
    matryoshka_prefix: bool,
    tolerance: f64,
) -> PyResult<Py<PyDict>> {
    let x_values = x.as_array().to_owned();
    let config = BlockSparseConfig {
        n_blocks,
        block_size,
        block_topk,
        max_epochs,
        minibatch,
        block_tile,
        frame_ridge,
        aux_k,
        matryoshka_prefix,
        tolerance,
    };
    let (fit, dual_cert) = detach_py_result(py, "block_sparse_dictionary_fit", move || {
        let fit = fit_block_sparse_dictionary(x_values.view(), &config)?;
        // Block-lane global-optimality dual certificate (gate of the residual),
        // computed in the same detached block so every block fit emits it.
        let dual_cert = gam::terms::sae::dual_certificate::block_dual_certificate(
            x_values.view(),
            &fit,
            SPARSE_DICT_DUAL_CERT_MAX_BIRTHS,
        )?;
        Ok::<_, String>((fit, dual_cert))
    })?;
    let fitted = fit.reconstruct();
    let out = PyDict::new(py);
    out.set_item("decoder", fit.decoder.into_pyarray(py))?;
    out.set_item("blocks", fit.blocks.into_pyarray(py))?;
    out.set_item("gates", fit.gates.into_pyarray(py))?;
    out.set_item("codes", fit.codes.into_pyarray(py))?;
    out.set_item("fitted", fitted.into_pyarray(py))?;
    out.set_item("gamma", fit.gamma)?;
    out.set_item("block_utilization", fit.block_utilization)?;
    out.set_item("block_stable_rank", fit.block_stable_rank)?;
    out.set_item("matryoshka_prefix_losses", fit.matryoshka_prefix_losses)?;
    out.set_item("explained_variance", fit.explained_variance)?;
    out.set_item("epochs", fit.epochs)?;
    out.set_item("converged", fit.converged)?;
    out.set_item("block_topk", fit.block_topk)?;
    out.set_item("block_size", fit.block_size)?;
    out.set_item(
        "dual_certificate",
        dual_certificate_report_dict(py, &dual_cert)?,
    )?;
    Ok(out.unbind())
}

/// Out-of-sample BLOCK encode: route held-out rows `x` (`M×P`, f32) through frozen
/// block frames `decoder` (`K×P`, `K = G·b`) with tied scalar `gamma`, returning the
/// fixed-width sparse block routing `(blocks[M,k], gates[M,k], codes[M,k,b])` via the
/// Rust core — the same group-ℓ₂ gate + block-TopK + tied signed codes the trainer
/// uses. Lets the Python `transform` delegate instead of reimplementing the routing.
#[pyfunction(signature = (x, decoder, gamma, block_size, block_topk, block_tile = 1024))]
fn block_sparse_dictionary_transform_ffi<'py>(
    py: Python<'py>,
    x: PyReadonlyArray2<'py, f32>,
    decoder: PyReadonlyArray2<'py, f32>,
    gamma: f32,
    block_size: usize,
    block_topk: usize,
    block_tile: usize,
) -> PyResult<(Py<PyArray2<u32>>, Py<PyArray2<f32>>, Py<PyArray3<f32>>)> {
    let x_values = x.as_array().to_owned();
    let decoder_values = decoder.as_array().to_owned();
    let (blocks, gates, codes) =
        detach_py_result(py, "block_sparse_dictionary_transform", move || {
            block_sparse_dictionary_transform(
                x_values.view(),
                decoder_values.view(),
                gamma,
                block_size,
                block_topk,
                block_tile,
            )
        })?;
    Ok((
        blocks.into_pyarray(py).unbind(),
        gates.into_pyarray(py).unbind(),
        codes.into_pyarray(py).unbind(),
    ))
}

#[pyfunction(signature = (decoder, blocks, codes, block_size))]
fn block_sparse_dictionary_reconstruct_ffi<'py>(
    py: Python<'py>,
    decoder: PyReadonlyArray2<'py, f32>,
    blocks: PyReadonlyArray2<'py, u32>,
    codes: PyReadonlyArray3<'py, f32>,
    block_size: usize,
) -> PyResult<Py<PyArray2<f32>>> {
    let decoder_values = decoder.as_array().to_owned();
    let block_values = blocks.as_array().to_owned();
    let code_values = codes.as_array().to_owned();
    let out = detach_py_result(py, "block_sparse_dictionary_reconstruct", move || {
        reconstruct_block_sparse_rows(
            decoder_values.view(),
            block_values.view(),
            code_values.view(),
            block_size,
        )
    })?;
    Ok(out.into_pyarray(py).unbind())
}

#[pyfunction(signature = (x, decoder, block_size, block))]
fn block_sparse_dictionary_block_coords_ffi<'py>(
    py: Python<'py>,
    x: PyReadonlyArray2<'py, f32>,
    decoder: PyReadonlyArray2<'py, f32>,
    block_size: usize,
    block: usize,
) -> PyResult<Py<PyArray2<f32>>> {
    let x_values = x.as_array().to_owned();
    let decoder_values = decoder.as_array().to_owned();
    let out = detach_py_result(py, "block_sparse_dictionary_block_coords", move || {
        block_sparse_dictionary_block_coords(
            x_values.view(),
            decoder_values.view(),
            block_size,
            block,
        )
    })?;
    Ok(out.into_pyarray(py).unbind())
}

#[pyfunction(signature = (coords, decoder, block_size, block))]
fn block_sparse_dictionary_lift_block_ffi<'py>(
    py: Python<'py>,
    coords: PyReadonlyArray2<'py, f32>,
    decoder: PyReadonlyArray2<'py, f32>,
    block_size: usize,
    block: usize,
) -> PyResult<Py<PyArray2<f32>>> {
    let coord_values = coords.as_array().to_owned();
    let decoder_values = decoder.as_array().to_owned();
    let out = detach_py_result(py, "block_sparse_dictionary_lift_block", move || {
        block_sparse_dictionary_lift_block(
            coord_values.view(),
            decoder_values.view(),
            block_size,
            block,
        )
    })?;
    Ok(out.into_pyarray(py).unbind())
}

#[pyfunction(signature = (x, decoder, gamma, block_size, block_topk, block, block_tile = 1024))]
fn block_sparse_dictionary_project_residual_ffi<'py>(
    py: Python<'py>,
    x: PyReadonlyArray2<'py, f32>,
    decoder: PyReadonlyArray2<'py, f32>,
    gamma: f32,
    block_size: usize,
    block_topk: usize,
    block: usize,
    block_tile: usize,
) -> PyResult<Py<PyArray2<f32>>> {
    let x_values = x.as_array().to_owned();
    let decoder_values = decoder.as_array().to_owned();
    let out = detach_py_result(py, "block_sparse_dictionary_project_residual", move || {
        block_sparse_dictionary_project_residual(
            x_values.view(),
            decoder_values.view(),
            gamma,
            block_size,
            block_topk,
            block_tile,
            block,
        )
    })?;
    Ok(out.into_pyarray(py).unbind())
}

#[pyfunction(signature = (blocks, n_blocks))]
fn block_sparse_dictionary_firings_ffi<'py>(
    py: Python<'py>,
    blocks: PyReadonlyArray2<'py, u32>,
    n_blocks: usize,
) -> PyResult<Vec<usize>> {
    let block_values = blocks.as_array().to_owned();
    detach_py_result(py, "block_sparse_dictionary_firings", move || {
        block_sparse_dictionary_firings(block_values.view(), n_blocks)
    })
}

#[pyfunction(signature = (
    x,
    decoder,
    blocks,
    block_utilization,
    block_stable_rank,
    gamma,
    block_size,
    block_topk,
    explained_variance,
    residual_target = true,
    n_basis_chart = 4,
    include_bases = true,
    name_prefix = "block",
    block_tile = 1024
))]
fn block_sparse_dictionary_seed_manifest_ffi<'py>(
    py: Python<'py>,
    x: PyReadonlyArray2<'py, f32>,
    decoder: PyReadonlyArray2<'py, f32>,
    blocks: PyReadonlyArray2<'py, u32>,
    block_utilization: PyReadonlyArray1<'py, f32>,
    block_stable_rank: PyReadonlyArray1<'py, f32>,
    gamma: f32,
    block_size: usize,
    block_topk: usize,
    explained_variance: f64,
    residual_target: bool,
    n_basis_chart: usize,
    include_bases: bool,
    name_prefix: &str,
    block_tile: usize,
) -> PyResult<Py<PyDict>> {
    let x_values = x.as_array().to_owned();
    let decoder_values = decoder.as_array().to_owned();
    let block_values = blocks.as_array().to_owned();
    let utilization = block_utilization.as_array().to_vec();
    let stable_rank = block_stable_rank.as_array().to_vec();
    let config = BlockSeedManifestConfig {
        block_size,
        block_topk,
        gamma,
        residual_target,
        n_basis_chart,
        include_bases,
        name_prefix: name_prefix.to_string(),
        block_tile,
    };
    let manifest = detach_py_result(py, "block_sparse_dictionary_seed_manifest", move || {
        block_sparse_dictionary_seed_manifest(
            x_values.view(),
            decoder_values.view(),
            block_values.view(),
            &utilization,
            &stable_rank,
            explained_variance,
            &config,
        )
    })?;
    block_seed_manifest_to_py(py, &manifest)
}

#[pyfunction(signature = (
    x,
    decoder,
    blocks,
    codes,
    gamma,
    block_size,
    block_topk,
    residual_target = true,
    min_firings = 64,
    max_blocks = 256,
    crossfit_folds = 2,
    min_effect = 0.0,
    whitening_ridge = 1.0e-8,
    pair_screen = true,
    pair_top_blocks = 64,
    max_pairs = 128,
    pair_min_cofirings = 64,
    pair_min_score = 0.20,
    block_tile = 1024
))]
fn block_coordinate_chart_compose_ffi<'py>(
    py: Python<'py>,
    x: PyReadonlyArray2<'py, f32>,
    decoder: PyReadonlyArray2<'py, f32>,
    blocks: PyReadonlyArray2<'py, u32>,
    codes: PyReadonlyArray3<'py, f32>,
    gamma: f32,
    block_size: usize,
    block_topk: usize,
    residual_target: bool,
    min_firings: usize,
    max_blocks: usize,
    crossfit_folds: usize,
    min_effect: f64,
    whitening_ridge: f64,
    pair_screen: bool,
    pair_top_blocks: usize,
    max_pairs: usize,
    pair_min_cofirings: usize,
    pair_min_score: f64,
    block_tile: usize,
) -> PyResult<Py<PyDict>> {
    let x_values = x.as_array().to_owned();
    let decoder_values = decoder.as_array().to_owned();
    let block_values = blocks.as_array().to_owned();
    let code_values = codes.as_array().to_owned();
    let config = BlockChartComposeConfig {
        block_size,
        block_topk,
        gamma,
        residual_target,
        min_firings,
        max_blocks,
        crossfit_folds,
        min_effect,
        whitening_ridge,
        pair_screen,
        pair_top_blocks,
        max_pairs,
        pair_min_cofirings,
        pair_min_score,
        block_tile,
    };
    let result = detach_py_result(py, "block_coordinate_chart_compose", move || {
        compose_block_coordinate_charts(
            x_values.view(),
            decoder_values.view(),
            block_values.view(),
            code_values.view(),
            &config,
        )
    })?;
    let out = PyDict::new(py);
    out.set_item("reconstructed", result.reconstructed.into_pyarray(py))?;
    out.set_item("selected_blocks", result.selected_blocks)?;
    out.set_item("selected_chart_blocks", result.selected_chart_blocks)?;
    out.set_item("selected_chart_pairs", result.selected_chart_pairs)?;
    out.set_item("blocks", chart_records_to_py(py, &result.block_records)?)?;
    out.set_item("pairs", chart_records_to_py(py, &result.pair_records)?)?;
    Ok(out.unbind())
}

fn chart_records_to_py(py: Python<'_>, records: &[BlockChartRecord]) -> PyResult<Py<PyList>> {
    let rows = PyList::empty(py);
    for record in records {
        let e = &record.evidence;
        let row = PyDict::new(py);
        row.set_item("block0", record.block0)?;
        row.set_item("block1", record.block1)?;
        row.set_item("screen_score", record.screen_score)?;
        row.set_item("n_rows", e.n_rows)?;
        row.set_item("n_effective", e.n_effective)?;
        row.set_item("linear_loss", e.linear_loss)?;
        row.set_item("chart_loss", e.chart_loss)?;
        row.set_item("deviance_gain", e.deviance_gain)?;
        row.set_item("mean_delta", e.mean_delta)?;
        row.set_item("sandwich_se", e.se)?;
        row.set_item("ci_low", e.ci_low)?;
        row.set_item("ci_high", e.ci_high)?;
        row.set_item("charge", e.charge)?;
        row.set_item("margin", e.margin)?;
        row.set_item("selected_by_bic", e.selected_by_bic)?;
        rows.append(row)?;
    }
    Ok(rows.unbind())
}

fn block_seed_manifest_to_py(py: Python<'_>, manifest: &BlockSeedManifest) -> PyResult<Py<PyDict>> {
    let out = PyDict::new(py);
    out.set_item("schema", "block_seed_manifest.v1")?;
    out.set_item("n_blocks", manifest.n_blocks)?;
    out.set_item("block_size", manifest.block_size)?;
    out.set_item("block_topk", manifest.block_topk)?;
    out.set_item("ambient_p", manifest.ambient_p)?;
    out.set_item("gamma", manifest.gamma)?;
    out.set_item("explained_variance", manifest.explained_variance)?;
    out.set_item("residual_target", manifest.residual_target)?;
    out.set_item("n_basis_chart", manifest.n_basis_chart)?;
    out.set_item(
        "blocks",
        block_seed_records_to_py(py, &manifest.blocks, false)?,
    )?;
    out.set_item(
        "block_seeds",
        block_seed_records_to_py(py, &manifest.blocks, true)?,
    )?;
    let featurizers = PyList::empty(py);
    for record in &manifest.blocks {
        featurizers.append(mdl_featurizer_to_py(py, &record.mdl_block)?)?;
        featurizers.append(mdl_featurizer_to_py(py, &record.mdl_chart)?)?;
    }
    out.set_item("mdl_featurizers", featurizers)?;
    Ok(out.unbind())
}

fn block_seed_records_to_py(
    py: Python<'_>,
    records: &[BlockSeedRecord],
    include_mdl: bool,
) -> PyResult<Py<PyList>> {
    let rows = PyList::empty(py);
    for record in records {
        let row = PyDict::new(py);
        row.set_item("block", record.block)?;
        row.set_item("block_dim", record.block_dim)?;
        row.set_item("n_firings", record.n_firings)?;
        row.set_item("utilization", record.utilization)?;
        row.set_item("stable_rank", record.stable_rank)?;
        row.set_item("coded_var", record.coded_var.clone())?;
        row.set_item("total_var", record.total_var)?;
        row.set_item("block_linear_ev", record.block_linear_ev)?;
        if let Some(basis) = &record.basis {
            row.set_item("basis", basis.clone())?;
        }
        if include_mdl {
            row.set_item("mdl_block", mdl_featurizer_to_py(py, &record.mdl_block)?)?;
            row.set_item("mdl_chart", mdl_featurizer_to_py(py, &record.mdl_chart)?)?;
        }
        rows.append(row)?;
    }
    Ok(rows.unbind())
}

fn mdl_featurizer_to_py(py: Python<'_>, row: &MdlFeaturizerRow) -> PyResult<Py<PyDict>> {
    let out = PyDict::new(py);
    out.set_item("name", &row.name)?;
    out.set_item("kind", &row.kind)?;
    out.set_item("total_var", row.total_var)?;
    out.set_item("n_tokens", row.n_tokens)?;
    out.set_item("n_firings", row.n_firings)?;
    out.set_item("n_params", row.n_params)?;
    out.set_item("coded_var", row.coded_var.clone())?;
    out.set_item("g_dict", row.g_dict)?;
    out.set_item("k_active", row.k_active)?;
    if let Some(block_name) = &row.block_name {
        out.set_item("block_name", block_name)?;
    }
    if let Some(chart_name) = &row.chart_name {
        out.set_item("chart_name", chart_name)?;
    }
    Ok(out.unbind())
}

/// Streaming (partial-fit) handle for the collapsed linear lane: a native-side
/// [`SparseDictStreamState`] so a Python loop can stream epochs over shards
/// without round-tripping the `K×P` decoder or any `N×K` object through Python.
///
/// `fit_begin` is the constructor (seed sample + config); `partial_fit(shard)`
/// routes and accumulates one shard; `end_epoch()` refreshes the decoder and
/// revives dead atoms; `finalize()` returns the decoder + metadata. The decoder
/// and dead-atom revival state warm-start across every call.
#[pyclass(module = "gam_pyffi._rust", name = "SparseDictStream")]
struct SparseDictStream {
    inner: SparseDictStreamState,
}

#[pymethods]
impl SparseDictStream {
    #[new]
    #[pyo3(signature = (
        seed,
        k,
        active = 1,
        minibatch = 512,
        max_epochs = 30,
        score_tile = 4096,
        code_ridge = 1.0e-6,
        decoder_ridge = 1.0e-6,
        tolerance = 1.0e-6,
        score_mode = "auto"
    ))]
    fn new(
        py: Python<'_>,
        seed: PyReadonlyArray2<'_, f32>,
        k: usize,
        active: usize,
        minibatch: usize,
        max_epochs: usize,
        score_tile: usize,
        code_ridge: f32,
        decoder_ridge: f32,
        tolerance: f64,
        score_mode: &str,
    ) -> PyResult<Self> {
        let score_mode = parse_sparse_dict_score_mode(score_mode)?;
        let seed_values = seed.as_array().to_owned();
        let config = SparseDictConfig {
            n_atoms: k,
            active,
            minibatch,
            max_epochs,
            score_tile,
            code_ridge,
            decoder_ridge,
            tolerance,
            score_mode,
        };
        let inner = py
            .detach(|| SparseDictStreamState::new(seed_values.view(), &config))
            .map_err(py_value_error)?;
        Ok(Self { inner })
    }

    /// Route + sparse-code one shard against the current decoder and fold its
    /// contributions into the running epoch. Returns `{rows, rss, alive_atoms,
    /// score_route_stats}`.
    fn partial_fit<'py>(
        &mut self,
        py: Python<'py>,
        shard: PyReadonlyArray2<'py, f32>,
    ) -> PyResult<Py<PyDict>> {
        let shard_values = shard.as_array().to_owned();
        let stats = py
            .detach(|| self.inner.partial_fit(shard_values.view()))
            .map_err(py_value_error)?;
        let out = PyDict::new(py);
        out.set_item("rows", stats.rows)?;
        out.set_item("rss", stats.rss)?;
        out.set_item("alive_atoms", stats.alive_atoms)?;
        out.set_item(
            "score_route_stats",
            score_route_stats_dict(py, stats.score_route_stats)?,
        )?;
        Ok(out.unbind())
    }

    /// Refresh the decoder from the epoch's accumulated normal equations, revive
    /// dead atoms onto worst-reconstructed residual rows, and reset the epoch.
    /// Returns `{explained_variance, revived, dead, converged, epoch}`.
    fn end_epoch(&mut self, py: Python<'_>) -> PyResult<Py<PyDict>> {
        let stats = py
            .detach(|| self.inner.end_epoch())
            .map_err(py_value_error)?;
        let out = PyDict::new(py);
        out.set_item("explained_variance", stats.explained_variance)?;
        out.set_item("revived", stats.revived)?;
        out.set_item("dead", stats.dead)?;
        out.set_item("converged", stats.converged)?;
        out.set_item("epoch", stats.epoch)?;
        Ok(out.unbind())
    }

    /// Hand back the trained decoder (`K×P`, unit-norm) plus run metadata and
    /// aggregate score-route telemetry.
    fn finalize(&self, py: Python<'_>) -> PyResult<Py<PyDict>> {
        let artifact = self.inner.finalize();
        let out = PyDict::new(py);
        out.set_item("decoder", artifact.decoder.into_pyarray(py))?;
        out.set_item("active", artifact.active)?;
        out.set_item("epochs", artifact.epochs)?;
        out.set_item("explained_variance", artifact.explained_variance)?;
        out.set_item("converged", artifact.converged)?;
        out.set_item(
            "score_route_stats",
            score_route_stats_dict(py, artifact.score_route_stats)?,
        )?;
        Ok(out.unbind())
    }

    /// A live copy of the current warm-started decoder (`K×P`, unit-norm rows).
    fn decoder(&self, py: Python<'_>) -> Py<PyArray2<f32>> {
        self.inner.decoder().to_owned().into_pyarray(py).unbind()
    }

    /// Active budget `s` in use (`min(active, K)`).
    #[getter]
    fn active(&self) -> usize {
        self.inner.active()
    }

    /// Epochs closed so far.
    #[getter]
    fn epochs_run(&self) -> usize {
        self.inner.epochs_run()
    }
}

/// Streaming (partial-fit) handle for the block-sparse lane: a native-side
/// [`BlockSparseStreamState`] so a Python loop can stream epochs over shards of a
/// sharded corpus without round-tripping the `K×P` frames or any `N×k` object.
/// Mirrors [`SparseDictStream`]: `fit_begin` (seed + config), `partial_fit(shard)`
/// routes + accumulates one shard, `end_epoch()` refreshes γ + frames and revives
/// dead blocks, `finalize()` returns the frames + γ + per-block report. The frames,
/// γ, and revival state warm-start across every call.
#[pyclass(module = "gam_pyffi._rust", name = "BlockSparseDictStream")]
struct BlockSparseDictStream {
    inner: BlockSparseStreamState,
}

#[pymethods]
impl BlockSparseDictStream {
    #[new]
    #[pyo3(signature = (
        seed,
        n_blocks,
        block_size = 2,
        block_topk = 1,
        max_epochs = 30,
        minibatch = 512,
        block_tile = 1024,
        frame_ridge = 1.0e-9,
        aux_k = 0,
        tolerance = 1.0e-6
    ))]
    fn new(
        py: Python<'_>,
        seed: PyReadonlyArray2<'_, f32>,
        n_blocks: usize,
        block_size: usize,
        block_topk: usize,
        max_epochs: usize,
        minibatch: usize,
        block_tile: usize,
        frame_ridge: f64,
        aux_k: usize,
        tolerance: f64,
    ) -> PyResult<Self> {
        let seed_values = seed.as_array().to_owned();
        let config = BlockSparseConfig {
            n_blocks,
            block_size,
            block_topk,
            max_epochs,
            minibatch,
            block_tile,
            frame_ridge,
            aux_k,
            matryoshka_prefix: false,
            tolerance,
        };
        let inner = py
            .detach(|| BlockSparseStreamState::new(seed_values.view(), &config))
            .map_err(py_value_error)?;
        Ok(Self { inner })
    }

    /// Route + tied-code one shard against the current frames and fold its
    /// contributions into the running epoch. Returns `{rows, rss, alive_blocks}`.
    fn partial_fit<'py>(
        &mut self,
        py: Python<'py>,
        shard: PyReadonlyArray2<'py, f32>,
    ) -> PyResult<Py<PyDict>> {
        let shard_values = shard.as_array().to_owned();
        let stats = py
            .detach(|| self.inner.partial_fit(shard_values.view()))
            .map_err(py_value_error)?;
        let out = PyDict::new(py);
        out.set_item("rows", stats.rows)?;
        out.set_item("rss", stats.rss)?;
        out.set_item("alive_blocks", stats.alive_blocks)?;
        Ok(out.unbind())
    }

    /// Refresh γ + block frames from the epoch's accumulators, revive dead blocks
    /// onto worst-reconstructed residual rows, and reset the epoch. Returns
    /// `{explained_variance, revived, dead, gamma, converged, epoch}`.
    fn end_epoch(&mut self, py: Python<'_>) -> PyResult<Py<PyDict>> {
        let stats = py
            .detach(|| self.inner.end_epoch())
            .map_err(py_value_error)?;
        let out = PyDict::new(py);
        out.set_item("explained_variance", stats.explained_variance)?;
        out.set_item("revived", stats.revived)?;
        out.set_item("dead", stats.dead)?;
        out.set_item("gamma", stats.gamma)?;
        out.set_item("converged", stats.converged)?;
        out.set_item("epoch", stats.epoch)?;
        Ok(out.unbind())
    }

    /// Hand back the trained block frames (`K×P`) + γ + per-block report + metadata:
    /// `{decoder, gamma, block_topk, block_size, block_utilization,
    /// block_stable_rank, epochs, explained_variance, converged}`.
    fn finalize(&self, py: Python<'_>) -> PyResult<Py<PyDict>> {
        let artifact = self.inner.finalize();
        let out = PyDict::new(py);
        out.set_item("decoder", artifact.decoder.into_pyarray(py))?;
        out.set_item("gamma", artifact.gamma)?;
        out.set_item("block_topk", artifact.block_topk)?;
        out.set_item("block_size", artifact.block_size)?;
        out.set_item("block_utilization", artifact.block_utilization)?;
        out.set_item("block_stable_rank", artifact.block_stable_rank)?;
        out.set_item("epochs", artifact.epochs)?;
        out.set_item("explained_variance", artifact.explained_variance)?;
        out.set_item("converged", artifact.converged)?;
        Ok(out.unbind())
    }

    /// A live copy of the current warm-started frames (`K×P`, block-orthonormal).
    fn decoder(&self, py: Python<'_>) -> Py<PyArray2<f32>> {
        self.inner.decoder().to_owned().into_pyarray(py).unbind()
    }

    /// Current shared tied scalar γ.
    #[getter]
    fn gamma(&self) -> f32 {
        self.inner.gamma()
    }

    /// Block routing budget `k` in use (`min(block_topk, G)`).
    #[getter]
    fn block_topk(&self) -> usize {
        self.inner.block_topk()
    }

    /// Block size `b`.
    #[getter]
    fn block_size(&self) -> usize {
        self.inner.block_size()
    }

    /// Epochs closed so far.
    #[getter]
    fn epochs_run(&self) -> usize {
        self.inner.epochs_run()
    }

    /// Per-BLOCK honest-charge ledger from the last CLOSED epoch (#23
    /// certification surface): the same `realised_rank_charge_dof` currency the
    /// joint PROMOTE/DEMOTE gates charge, priced in deviance units against
    /// `½·d_eff·ln(n_obs)`. The block is the certification unit (its `b` atoms
    /// share one jointly-fitted frame and one code Gram — atom ids for block
    /// `g` are `g*b .. (g+1)*b`). Returns parallel lists
    /// `{block, n_eff, d_eff, delta_deviance, charge, margin, kept}`; `margin`
    /// doubles as the `log_e_value` an e-BH certificate can consume. Errors if
    /// no epoch has closed yet.
    fn block_rank_charges(&self, py: Python<'_>, n_obs: usize) -> PyResult<Py<PyDict>> {
        let charges = py
            .detach(|| self.inner.block_rank_charges(n_obs))
            .map_err(py_value_error)?;
        let out = PyDict::new(py);
        out.set_item("block", charges.block)?;
        out.set_item("n_eff", charges.n_eff)?;
        out.set_item("d_eff", charges.d_eff)?;
        out.set_item("delta_deviance", charges.delta_deviance)?;
        out.set_item("charge", charges.charge)?;
        out.set_item("margin", charges.margin)?;
        out.set_item("kept", charges.kept)?;
        Ok(out.unbind())
    }
}

/// Exact realised rank-charge DOF — the single evidence currency (PROMOTE /
/// DEMOTE / streaming block ledger), exposed so Python drivers price candidates
/// with the criterion itself instead of re-deriving it (re-derivations drift; a
/// ½-factor mismatch was caught in the first attempt). `gram` is the
/// candidate's `M×M` weighted design Gram, `decoder` its `M×p` decoder block,
/// `n_eff` the effective sample mass, `p_out` the output dimension,
/// `dispersion` the reconstruction φ̂ (MP floor).
#[pyfunction]
#[pyo3(signature = (gram, decoder, n_eff, p_out, dispersion))]
fn rank_charge_dof(
    py: Python<'_>,
    gram: PyReadonlyArray2<'_, f64>,
    decoder: PyReadonlyArray2<'_, f64>,
    n_eff: f64,
    p_out: f64,
    dispersion: f64,
) -> PyResult<f64> {
    let gram_values = gram.as_array().to_owned();
    let decoder_values = decoder.as_array().to_owned();
    py.detach(|| {
        gam::terms::sae::manifold::rank_charge_dof(
            &gram_values,
            &decoder_values,
            n_eff,
            p_out,
            dispersion,
        )
    })
    .map_err(py_value_error)
}

fn inject_scalar_fisher_rao_weight(
    dataset: &mut EncodedDataset,
    fit_config: &mut FitConfig,
    fisher_rao_w: ArrayView3<'_, f64>,
) -> Result<(), String> {
    let n = dataset.values.nrows();
    if fisher_rao_w.shape() != [n, 1, 1] {
        return Err(format!(
            "fit_table fisher_rao_w requires scalar blocks of shape ({n}, 1, 1); got {:?}",
            fisher_rao_w.shape()
        ));
    }
    let weight_name = "__gamfit_fisher_rao_weight".to_string();
    if dataset.headers.iter().any(|name| name == &weight_name) {
        return Err(format!(
            "reserved fisher_rao_w column already exists: {weight_name}"
        ));
    }
    let col_map = dataset.column_map();
    let base_weight_col = fit_config
        .weight_column
        .as_ref()
        .map(|name| {
            col_map
                .get(name)
                .copied()
                .ok_or_else(|| format!("weights column {name:?} is not present"))
        })
        .transpose()?;
    let mut combined = Array1::<f64>::zeros(n);
    for row in 0..n {
        let mut value = fisher_rao_w[[row, 0, 0]];
        if let Some(col) = base_weight_col {
            value *= dataset.values[[row, col]];
        }
        if !(value.is_finite() && value >= 0.0) {
            return Err(format!(
                "fisher_rao_w scalar row {row} produced invalid non-negative weight {value}"
            ));
        }
        combined[row] = value;
    }
    let old_cols = dataset.values.ncols();
    let mut values = Array2::<f64>::zeros((n, old_cols + 1));
    values.slice_mut(s![.., ..old_cols]).assign(&dataset.values);
    values.column_mut(old_cols).assign(&combined);
    dataset.values = values;
    dataset.headers.push(weight_name.clone());
    dataset.schema.columns.push(SchemaColumn {
        name: weight_name.clone(),
        kind: ColumnKindTag::Continuous,
        levels: Vec::new(),
    });
    dataset.column_kinds.push(ColumnKindTag::Continuous);
    fit_config.weight_column = Some(weight_name);
    Ok(())
}

fn fit_dataset_impl(
    mut dataset: EncodedDataset,
    formula: String,
    config_json: Option<&str>,
    fisher_rao_w: Option<ArrayView3<'_, f64>>,
) -> Result<Vec<u8>, WorkflowError> {
    // The stderr `[OUTER step]` log stream (installed by `progress_log::
    // init_logging` at module import) carries solver progress for the Python
    // bindings; the former always-on TUI session lane has been removed.
    let (mut fit_config, training_table_kind) = parse_fit_config(config_json)?;
    if let Some(w) = fisher_rao_w {
        inject_scalar_fisher_rao_weight(&mut dataset, &mut fit_config, w)?;
    }
    // Expectile (Newey–Powell LAWS) family (#1777): the expectile estimator is an
    // OUTER driver that wraps the standard Gaussian-identity GAM with iterative
    // asymmetric reweighting, so it is selected *before* `materialize` (which has
    // no expectile arm) — exactly as the in-process `fit_from_formula` does. We
    // route it through the single shared dispatch seam so the Python API reaches
    // the same estimator the library call does instead of failing with
    // `unknown family 'expectile(τ)'`. The driver returns an ordinary
    // `StandardFitResult`, so the persistence payload is built by the same
    // `build_standard_payload` used for every other standard fit.
    if let Some(expectile_result) = gam::families::fit_orchestration::fit_expectile_if_requested(
        &formula,
        &dataset,
        &fit_config,
    )? {
        let family = expectile_result
            .fit
            .likelihood_family
            .clone()
            .unwrap_or_else(gam::types::LikelihoodSpec::gaussian_identity);
        let mut payload = build_standard_payload(
            formula,
            &dataset,
            &fit_config,
            family,
            &expectile_result.fit,
            &expectile_result.design,
            expectile_result.resolvedspec,
            expectile_result.adaptive_diagnostics,
            expectile_result.wiggle_knots.map(|knots| knots.to_vec()),
            expectile_result.wiggle_degree,
            expectile_result.wiggle_saved_warp_beta,
            // Expectile LAWS is Gaussian-identity; it never engages the binomial
            // frozen-basis de-aliasing, so there is no frozen-index shift (#2141).
            None,
        )?;
        payload.group_metadata = fit_config.group_metadata.clone();
        payload.training_table_kind = training_table_kind;
        // The LAWS driver materializes its inner Gaussian design itself; there are
        // no outer materialize advisories to carry (matches `fit_from_formula`).
        payload.inference_notes = Vec::new();
        let model = FittedModel::from_payload(payload);
        return serde_json::to_vec(&model).map_err(|err| {
            gam::families::fit_orchestration::WorkflowError::IntegrationFailed {
                reason: format!("failed to serialize model: {err}"),
            }
        });
    }
    // Calibrated marginal-slope chain (#461): when a CTN Stage-1 recipe is present
    // (config.ctn_stage1), the marginal-slope materializer cross-fits the CTN and
    // produces the calibrated `z` out-of-fold — no z_column is needed and no
    // Stage-1 pre-fit / synthetic column round-trip is performed here. The recipe
    // rides on fit_config straight into materialize.
    let materialized = materialize(&formula, &dataset, &fit_config)?;
    let request = materialized.request;
    // Advisories produced while materializing (e.g. the mgcv-style "k reduced to
    // the data support" / basis-degradation notes from the cr/cs/sz cap, #1541
    // #1542). The CLI prints these via `print_inference_summary`; the Python
    // path used to drop them on the floor, so a gamfit user whose basis was
    // silently capped got no signal at all (#1543). Carry them into the
    // serialized payload so gamfit can surface them as `GamInferenceWarning`s
    // and via `model.notes`.
    let inference_notes = materialized.inference_notes;

    let mut payload = match request {
        FitRequest::Standard(standard_request) => {
            // Exact O(n) spline-scan fast path (#1030/#1034): a single 1-D
            // Gaussian cubic smooth is the penalized cubic-spline problem the
            // state-space scan solves exactly — route through it and persist
            // the smoother state instead of the dense fit. Detection is
            // structural; every other shape falls through to the dense fit
            // below. Mirrors the CLI run_fit path so CLI and FFI saves agree.
            if let Some(inputs) =
                gam::families::fit_orchestration::spline_scan_fast_path(&standard_request)
            {
                let scan = gam::solver::spline_scan::fit_spline_scan(
                    &inputs.x,
                    &inputs.y,
                    &inputs.w,
                    inputs.order,
                )
                .map_err(|reason| {
                    gam::families::fit_orchestration::WorkflowError::IntegrationFailed { reason }
                })?;
                let feature_col = match &standard_request.spec.smooth_terms[0].basis {
                    gam::terms::smooth::SmoothBasisSpec::BSpline1D { feature_col, .. } => {
                        *feature_col
                    }
                    _ => {
                        return Err(
                            gam::families::fit_orchestration::WorkflowError::SchemaMismatch {
                                reason: "spline-scan detection accepted a non-1D basis".to_string(),
                            },
                        );
                    }
                };
                let feature_column =
                    dataset.headers.get(feature_col).cloned().ok_or_else(|| {
                        gam::families::fit_orchestration::WorkflowError::SchemaMismatch {
                            reason: format!(
                                "spline-scan feature column {feature_col} has no header"
                            ),
                        }
                    })?;
                let mut scan_payload =
                    gam::inference::model_payload_builders::assemble_spline_scan_payload(
                        formula,
                        feature_column,
                        &scan,
                        dataset.schema.clone(),
                        dataset.headers.clone(),
                        dataset.feature_ranges(),
                    );
                scan_payload.group_metadata = fit_config.group_metadata.clone();
                scan_payload.training_table_kind = training_table_kind;
                scan_payload.inference_notes = inference_notes;
                let model = FittedModel::from_payload(scan_payload);
                return serde_json::to_vec(&model).map_err(|err| {
                    gam::families::fit_orchestration::WorkflowError::IntegrationFailed {
                        reason: format!("failed to serialize model: {err}"),
                    }
                });
            }
            let family = standard_request.family.clone();
            let fit_result = fit_model(FitRequest::Standard(standard_request))?;
            let standard_result = match fit_result {
                FitResult::Standard(standard_result) => standard_result,
                _ => {
                    return Err(gam::families::fit_orchestration::WorkflowError::SchemaMismatch {
                        reason: "python binding expected the standard workflow to return a standard fit result"
                            .to_string(),
                    });
                }
            };
            let saved_fit = standard_result.fit.clone();
            build_standard_payload(
                formula,
                &dataset,
                &fit_config,
                family,
                &saved_fit,
                &standard_result.design,
                standard_result.resolvedspec,
                standard_result.adaptive_diagnostics,
                standard_result.wiggle_knots.map(|knots| knots.to_vec()),
                standard_result.wiggle_degree,
                standard_result.wiggle_saved_warp_beta,
                standard_result.wiggle_saved_index_shift,
            )?
        }
        FitRequest::TransformationNormal(tn_request) => {
            let fit_result = fit_model(FitRequest::TransformationNormal(tn_request))?;
            let tn_result = match fit_result {
                FitResult::TransformationNormal(result) => result,
                _ => {
                    return Err(gam::families::fit_orchestration::WorkflowError::SchemaMismatch {
                        reason: "python binding expected the transformation-normal workflow to return a transformation-normal fit result"
                            .to_string(),
                    });
                }
            };
            build_transformation_normal_ffi_payload(formula, &dataset, &fit_config, tn_result)?
        }
        FitRequest::BernoulliMarginalSlope(ms_request) => {
            let base_link = ms_request.spec.base_link.clone();
            let frailty = ms_request.spec.frailty.clone();
            let fit_result = fit_model(FitRequest::BernoulliMarginalSlope(ms_request))?;
            let ms_result = match fit_result {
                FitResult::BernoulliMarginalSlope(result) => result,
                _ => {
                    return Err(gam::families::fit_orchestration::WorkflowError::SchemaMismatch {
                        reason: "python binding expected the bernoulli marginal-slope workflow to return a marginal-slope fit result"
                            .to_string(),
                    });
                }
            };
            build_bernoulli_marginal_slope_ffi_payload(
                formula,
                &dataset,
                &fit_config,
                base_link,
                frailty,
                ms_result,
            )?
        }
        FitRequest::SurvivalMarginalSlope(ms_request) => {
            let frailty = ms_request.spec.frailty.clone();
            let fit_result = fit_model(FitRequest::SurvivalMarginalSlope(ms_request))?;
            let ms_result = match fit_result {
                FitResult::SurvivalMarginalSlope(result) => result,
                _ => {
                    return Err(gam::families::fit_orchestration::WorkflowError::SchemaMismatch {
                        reason: "python binding expected the survival marginal-slope workflow to return a survival marginal-slope fit result"
                            .to_string(),
                    });
                }
            };
            build_survival_marginal_slope_ffi_payload(
                formula,
                &dataset,
                &fit_config,
                frailty,
                ms_result,
            )?
        }
        FitRequest::GaussianLocationScale(ls_request) => {
            let fit_result = fit_model(FitRequest::GaussianLocationScale(ls_request))?;
            let ls_result = match fit_result {
                FitResult::GaussianLocationScale(result) => result,
                _ => {
                    return Err(gam::families::fit_orchestration::WorkflowError::SchemaMismatch {
                        reason: "python binding expected the gaussian location-scale workflow to return a gaussian location-scale fit result"
                            .to_string(),
                    });
                }
            };
            // Persist the response standardization factor the fit applied so
            // prediction reconstructs the σ floor at `response_scale·0.01`,
            // keeping predictive σ response-scale-equivariant (#884). The fit
            // already mapped the log-σ `exp(η)` term to raw units via the
            // `+ln(response_scale)` intercept shift; only the additive floor
            // still needs the factor at reconstruction time.
            let response_scale = ls_result.response_scale;
            build_gaussian_location_scale_ffi_payload(
                formula,
                &dataset,
                &fit_config,
                ls_result,
                response_scale,
            )?
        }
        FitRequest::BinomialLocationScale(ls_request) => {
            let weights = ls_request.spec.weights.clone();
            let link_kind = ls_request.spec.link_kind.clone();
            let fit_result = fit_model(FitRequest::BinomialLocationScale(ls_request))?;
            let ls_result = match fit_result {
                FitResult::BinomialLocationScale(result) => result,
                _ => {
                    return Err(gam::families::fit_orchestration::WorkflowError::SchemaMismatch {
                        reason: "python binding expected the binomial location-scale workflow to return a binomial location-scale fit result"
                            .to_string(),
                    });
                }
            };
            build_binomial_location_scale_ffi_payload(
                formula,
                &dataset,
                &fit_config,
                link_kind,
                &weights,
                ls_result,
            )?
        }
        FitRequest::SurvivalLocationScale(ls_request) => {
            let weights = ls_request.spec.weights.clone();
            let fit_result = fit_model(FitRequest::SurvivalLocationScale(ls_request))?;
            let ls_result = match fit_result {
                FitResult::SurvivalLocationScale(result) => result,
                _ => {
                    return Err(gam::families::fit_orchestration::WorkflowError::SchemaMismatch {
                        reason: "python binding expected the survival location-scale workflow to return a survival location-scale fit result"
                            .to_string(),
                    });
                }
            };
            build_survival_location_scale_ffi_payload(
                formula,
                &dataset,
                &fit_config,
                &weights,
                ls_result,
            )?
        }
        FitRequest::SurvivalTransformation(rp_request) => {
            let fit_result = fit_model(FitRequest::SurvivalTransformation(rp_request))?;
            let rp_result = match fit_result {
                FitResult::SurvivalTransformation(result) => result,
                _ => {
                    return Err(gam::families::fit_orchestration::WorkflowError::SchemaMismatch {
                        reason: "python binding expected the survival transformation workflow to return a survival transformation fit result"
                            .to_string(),
                    });
                }
            };
            build_survival_transformation_ffi_payload(formula, &dataset, &fit_config, rp_result)?
        }
        FitRequest::LatentSurvival(lat_request) => {
            let frailty = lat_request.frailty.clone();
            let fit_result = fit_model(FitRequest::LatentSurvival(lat_request))?;
            let lat_result = match fit_result {
                FitResult::LatentSurvival(result) => result,
                _ => {
                    return Err(gam::families::fit_orchestration::WorkflowError::SchemaMismatch {
                        reason: "python binding expected the latent survival workflow to return a latent survival fit result"
                            .to_string(),
                    });
                }
            };
            build_latent_survival_ffi_payload(formula, &dataset, &fit_config, frailty, lat_result)?
        }
        FitRequest::LatentBinary(lat_request) => {
            let frailty = lat_request.frailty.clone();
            let fit_result = fit_model(FitRequest::LatentBinary(lat_request))?;
            let lat_result = match fit_result {
                FitResult::LatentBinary(result) => result,
                _ => {
                    return Err(gam::families::fit_orchestration::WorkflowError::SchemaMismatch {
                        reason: "python binding expected the latent binary workflow to return a latent binary fit result"
                            .to_string(),
                    });
                }
            };
            build_latent_binary_ffi_payload(formula, &dataset, &fit_config, frailty, lat_result)?
        }
        FitRequest::DispersionLocationScale(ls_request) => {
            // Genuine-dispersion location-scale family (#913): NB / Gamma / Beta
            // / Tweedie mean families whose `noise_formula` models the
            // overdispersion channel. Magic-detected upstream from a
            // `noise_formula` on one of those families; the FFI freezes the mean
            // and log-precision specs and persists them via the same shared
            // location-scale assembler the CLI uses.
            let kind = ls_request.spec.kind;
            let fit_result = fit_model(FitRequest::DispersionLocationScale(ls_request))?;
            let ls_result = match fit_result {
                FitResult::DispersionLocationScale(result) => result,
                _ => {
                    return Err(gam::families::fit_orchestration::WorkflowError::SchemaMismatch {
                        reason: "python binding expected the dispersion location-scale workflow to return a dispersion location-scale fit result"
                            .to_string(),
                    });
                }
            };
            build_dispersion_location_scale_ffi_payload(
                formula,
                &dataset,
                &fit_config,
                kind,
                ls_result,
            )?
        }
    };
    payload.group_metadata = fit_config.group_metadata.clone();
    payload.training_table_kind = training_table_kind;
    payload.inference_notes = inference_notes;
    let model = FittedModel::from_payload(payload);
    serde_json::to_vec(&model).map_err(|err| {
        gam::families::fit_orchestration::WorkflowError::IntegrationFailed {
            reason: format!("failed to serialize model: {err}"),
        }
    })
}

fn load_model_impl(model_bytes: &[u8]) -> Result<FittedModel, String> {
    let model: FittedModel = serde_json::from_slice(model_bytes)
        .map_err(|err| format!("failed to parse model json: {err}"))?;
    model.validate_for_persistence()?;
    model.validate_numeric_finiteness()?;
    Ok(model)
}

fn extend_model_with_group_impl(model_bytes: &[u8], request_json: &str) -> Result<Vec<u8>, String> {
    let mut model = load_model_impl(model_bytes)?;
    if !matches!(model.predict_model_class(), PredictModelClass::Standard) {
        return Err(format!(
            "extend_with_group currently supports standard GAM models only; got '{}'",
            prediction_model_class_label(&model)
        ));
    }
    if model.has_link_wiggle() {
        return Err("extend_with_group does not support link-wiggle models".to_string());
    }
    let request: PyExtendGroupRequest = serde_json::from_str(request_json)
        .map_err(|err| format!("failed to parse extend_with_group request: {err}"))?;
    let PyExtendGroupRequest {
        kind,
        name,
        term,
        column,
        level,
        levels,
        metadata,
        prior,
    } = request;
    let kind = kind
        .as_deref()
        .unwrap_or("random-effect-level")
        .replace('_', "-");
    if kind != "random-effect-level" {
        return Err(format!(
            "extend_with_group supports kind='random-effect-level'; got '{kind}'"
        ));
    }
    let mut levels = levels.unwrap_or_default();
    if let Some(level) = level {
        levels.push(level);
    }
    if levels.is_empty() {
        return Err("extend_with_group requires level or levels".to_string());
    }
    let term = match term.or(column) {
        Some(term) => term,
        None => {
            let payload = model.payload();
            let spec = payload.resolved_termspec.as_ref().ok_or_else(|| {
                "extend_with_group requires saved resolved_termspec; refit".to_string()
            })?;
            if spec.random_effect_terms.len() == 1 {
                spec.random_effect_terms[0].name.clone()
            } else {
                return Err(
                    "extend_with_group requires term when the model has zero or multiple group terms"
                        .to_string(),
                );
            }
        }
    };

    for level in levels {
        extend_model_with_random_effect_level(
            &mut model,
            term.as_str(),
            name.as_deref(),
            level,
            metadata.clone(),
            prior.clone(),
        )?;
    }
    model.validate_for_persistence()?;
    model.validate_numeric_finiteness()?;
    serde_json::to_vec(&model).map_err(|err| format!("failed to serialize extended model: {err}"))
}

fn extend_model_with_random_effect_level(
    model: &mut FittedModel,
    term_name: &str,
    requested_name: Option<&str>,
    level: serde_json::Value,
    metadata: Option<serde_json::Value>,
    prior: Option<serde_json::Value>,
) -> Result<(), String> {
    let payload: &mut FittedModelPayload = &mut *model;
    let (term_idx, feature_col, penalty_index) = {
        let spec = payload.resolved_termspec.as_ref().ok_or_else(|| {
            "extend_with_group requires saved resolved_termspec; refit".to_string()
        })?;
        let term_idx = spec
            .random_effect_terms
            .iter()
            .position(|term| term.name == term_name)
            .ok_or_else(|| format!("extend_with_group unknown random-effect term '{term_name}'"))?;
        (
            term_idx,
            spec.random_effect_terms[term_idx].feature_col,
            random_effect_penalty_index(spec, term_idx),
        )
    };
    let schema = payload
        .data_schema
        .as_mut()
        .ok_or_else(|| "extend_with_group requires saved data_schema; refit".to_string())?;
    let schema_col = schema.columns.get_mut(feature_col).ok_or_else(|| {
        format!(
            "extend_with_group term '{term_name}' feature column {feature_col} out of saved schema bounds"
        )
    })?;
    let (level_bits, encoded_value) = level_bits_for_extension(schema_col, &level)?;
    {
        let spec = payload.resolved_termspec.as_ref().ok_or_else(|| {
            "extend_with_group requires saved resolved_termspec; refit".to_string()
        })?;
        let levels = spec.random_effect_terms[term_idx]
            .frozen_levels
            .as_ref()
            .ok_or_else(|| {
                format!(
                    "extend_with_group term '{term_name}' is not frozen; refit with persisted metadata"
                )
            })?;
        if levels.contains(&level_bits) {
            return Err(format!(
                "extend_with_group level {} already exists for random-effect term '{term_name}'",
                compact_json(&level)
            ));
        }
    }
    if payload.deployment_extensions.iter().any(|extension| {
        extension.kind == "random-effect-level"
            && extension.term == term_name
            && extension.level_bits == level_bits
    }) {
        return Err(format!(
            "extend_with_group level {} is already deployed for random-effect term '{term_name}'",
            compact_json(&level)
        ));
    }
    let coefficient_index = payload
        .fit_result
        .as_ref()
        .ok_or_else(|| "extend_with_group requires saved fit_result; refit".to_string())?
        .beta
        .len();
    let (coefficient_mean, supplied_variance) = extension_prior_parameters(prior.as_ref())?;
    let coefficient_variance = match supplied_variance {
        Some(variance) => variance,
        None => {
            let fit = payload
                .fit_result
                .as_ref()
                .ok_or_else(|| "extend_with_group requires saved fit_result; refit".to_string())?;
            let lambda = fit
                .lambdas
                .get(penalty_index)
                .copied()
                .filter(|lambda| lambda.is_finite() && *lambda > 0.0)
                .ok_or_else(|| {
                    format!(
                        "extend_with_group term '{term_name}' has no finite positive prior lambda"
                    )
                })?;
            // The unseen-level default prior is the fitted random-effect
            // variance component `σ_b² = φ̂ / λ` (mgcv's `λ = φ̂ / σ_b²`
            // convention), NOT the scale-free `1 / λ`. `φ̂` is the residual
            // dispersion that scales every predict-time covariance: `1` for
            // fixed-scale families (Poisson/Binomial — where `φ̂/λ` collapses
            // to the old `1/λ`), but `σ̂²` for Gaussian and the estimated
            // dispersion for Gamma/Tweedie/NB. Omitting `φ̂` made the prior
            // (and any deployment interval built from it) wrong by `1/φ̂` and,
            // for an estimated scale, not response-scale equivariant. See #674.
            let phi = fit.dispersion_phi();
            if !(phi.is_finite() && phi > 0.0) {
                return Err(format!(
                    "extend_with_group term '{term_name}' has a non-finite or non-positive \
                     dispersion (φ̂ = {phi}); cannot form the default prior variance"
                ));
            }
            phi / lambda
        }
    };
    extend_training_feature_range(
        payload.training_feature_ranges.as_mut(),
        feature_col,
        encoded_value,
    );
    insert_coefficient_into_saved_fit(
        payload.fit_result.as_mut(),
        coefficient_index,
        coefficient_mean,
        coefficient_variance,
    )?;
    insert_coefficient_into_saved_fit(
        payload.unified.as_mut(),
        coefficient_index,
        coefficient_mean,
        coefficient_variance,
    )?;

    let extension_name = requested_name
        .map(str::to_string)
        .unwrap_or_else(|| format!("{term_name}:{}", compact_json(&level)));
    if let Some(metadata_value) = metadata.clone() {
        let group_metadata = payload.group_metadata.get_or_insert_with(BTreeMap::new);
        group_metadata.insert(extension_name.clone(), metadata_value.clone());
    }
    payload
        .deployment_extensions
        .push(SavedDeploymentExtension {
            name: extension_name,
            kind: "random-effect-level".to_string(),
            term: term_name.to_string(),
            level,
            level_bits,
            coefficient_index,
            coefficient_mean,
            coefficient_variance,
            metadata,
            prior,
        });
    Ok(())
}

fn level_bits_for_extension(
    schema_col: &mut SchemaColumn,
    level: &serde_json::Value,
) -> Result<(u64, f64), String> {
    match schema_col.kind {
        ColumnKindTag::Categorical => {
            let label = match level {
                serde_json::Value::String(s) => s.clone(),
                other => compact_json(other),
            };
            if schema_col.levels.iter().any(|existing| existing == &label) {
                return Err(format!(
                    "extend_with_group categorical level '{label}' already exists in column '{}'",
                    schema_col.name
                ));
            }
            let encoded = schema_col.levels.len() as f64;
            schema_col.levels.push(label);
            Ok((encoded.to_bits(), encoded))
        }
        ColumnKindTag::Continuous | ColumnKindTag::Binary => {
            let value = json_level_to_f64(level)?;
            Ok((value.to_bits(), value))
        }
    }
}

fn json_level_to_f64(value: &serde_json::Value) -> Result<f64, String> {
    let out = match value {
        serde_json::Value::Number(n) => n
            .as_f64()
            .ok_or_else(|| format!("extend_with_group level {n} is not representable as f64"))?,
        serde_json::Value::String(s) => s
            .parse::<f64>()
            .map_err(|_| format!("extend_with_group level '{s}' is not numeric"))?,
        other => {
            return Err(format!(
                "extend_with_group numeric random-effect levels must be numbers or numeric strings; got {}",
                compact_json(other)
            ));
        }
    };
    if !out.is_finite() {
        return Err(format!(
            "extend_with_group random-effect level must be finite; got {out}"
        ));
    }
    Ok(out)
}

fn compact_json(value: &serde_json::Value) -> String {
    serde_json::to_string(value).unwrap_or_else(|_| "<unserializable>".to_string())
}

fn random_effect_penalty_index(spec: &TermCollectionSpec, term_idx: usize) -> usize {
    usize::from(spec.linear_terms.iter().any(|term| term.double_penalty)) + term_idx
}

fn extension_prior_parameters(
    prior: Option<&serde_json::Value>,
) -> Result<(f64, Option<f64>), String> {
    let Some(value) = prior else {
        return Ok((0.0, None));
    };
    if value.is_null() {
        return Ok((0.0, None));
    }
    let parsed: PyExtensionPrior = serde_json::from_value(value.clone())
        .map_err(|err| format!("failed to parse extend_with_group prior: {err}"))?;
    let mean = parsed.mean.or(parsed.mu).unwrap_or(0.0);
    if !mean.is_finite() {
        return Err(format!(
            "extend_with_group prior mean must be finite; got {mean}"
        ));
    }
    let variance = match (parsed.variance, parsed.precision) {
        (Some(variance), _) => {
            if !(variance.is_finite() && variance > 0.0) {
                return Err(format!(
                    "extend_with_group prior variance must be finite and positive; got {variance}"
                ));
            }
            Some(variance)
        }
        (None, Some(precision)) => {
            if !(precision.is_finite() && precision > 0.0) {
                return Err(format!(
                    "extend_with_group prior precision must be finite and positive; got {precision}"
                ));
            }
            Some(1.0 / precision)
        }
        (None, None) => None,
    };
    Ok((mean, variance))
}

fn extend_training_feature_range(
    ranges: Option<&mut Vec<(f64, f64)>>,
    feature_col: usize,
    value: f64,
) {
    if let Some(ranges) = ranges
        && let Some((lo, hi)) = ranges.get_mut(feature_col)
    {
        if value.is_finite() {
            *lo = (*lo).min(value);
            *hi = (*hi).max(value);
        }
    }
}

fn insert_coefficient_into_saved_fit(
    fit: Option<&mut gam::solver::estimate::UnifiedFitResult>,
    index: usize,
    value: f64,
    variance: f64,
) -> Result<(), String> {
    let Some(fit) = fit else {
        return Ok(());
    };
    if !(variance.is_finite() && variance > 0.0) {
        return Err(format!(
            "extend_with_group coefficient variance must be finite and positive; got {variance}"
        ));
    }
    if index > fit.beta.len() {
        return Err(format!(
            "extend_with_group coefficient index {index} exceeds fit coefficient length {}",
            fit.beta.len()
        ));
    }
    fit.beta = insert_array1(&fit.beta, index, value);
    let block_idx = fit
        .blocks
        .iter()
        .position(|block| block.role == BlockRole::Mean)
        .unwrap_or(0);
    if block_idx >= fit.blocks.len() {
        return Err("extend_with_group saved fit has no coefficient blocks".to_string());
    }
    if index > fit.blocks[block_idx].beta.len() {
        return Err(format!(
            "extend_with_group coefficient index {index} exceeds mean block length {}",
            fit.blocks[block_idx].beta.len()
        ));
    }
    fit.blocks[block_idx].beta = insert_array1(&fit.blocks[block_idx].beta, index, value);
    // No-refit posterior algebra for a deployment-only block:
    //
    // The fitted posterior precision for the original coefficients is H_old.
    // Extending with a new random-effect coefficient b and no likelihood
    // refit contributes only its Gaussian prior,
    //
    //   -log p(b) = 1/2 (b - mu)' (lambda_new S_new) (b - mu) + const.
    //
    // Since no old likelihood rows or old penalties are recomputed, the joint
    // precision is blockdiag(H_old, lambda_new S_new).  Therefore the
    // conditional covariance is blockdiag(V_old, S_new^{-1}/lambda_new).  The
    // current API extends one iid random-effect coordinate at a time, so
    // S_new = [1] and `variance` is exactly 1/lambda_new, or the caller's
    // supplied scalar prior covariance.
    if let Some(cov) = fit.covariance_conditional.as_mut() {
        *cov = insert_symmetric_array2(cov, index, variance)?;
    }
    if let Some(cov) = fit.covariance_corrected.as_mut() {
        *cov = insert_symmetric_array2(cov, index, variance)?;
    }
    let variance_diag = variance;
    let precision_diag = 1.0 / variance_diag;
    if let Some(inference) = fit.inference.as_mut() {
        // Boundary adapter: `penalized_hessian` is the `UnscaledPrecision`
        // newtype; unwrap for the `insert_symmetric_array2` helper and wrap
        // the result back on assignment.
        inference.penalized_hessian = insert_symmetric_array2(
            inference.penalized_hessian.as_array(),
            index,
            precision_diag,
        )?
        .into();
        if let Some(cov) = inference.beta_covariance.as_mut() {
            // `beta_covariance` is the `PhiScaledCovariance` newtype.
            *cov = insert_symmetric_array2(cov.as_array(), index, variance_diag)?.into();
        }
        if let Some(se) = inference.beta_standard_errors.as_mut() {
            *se = insert_array1(se, index, variance_diag.sqrt());
        }
        if let Some(cov) = inference.beta_covariance_corrected.as_mut() {
            *cov = insert_symmetric_array2(cov, index, variance_diag)?;
        }
        if let Some(se) = inference.beta_standard_errors_corrected.as_mut() {
            *se = insert_array1(se, index, variance_diag.sqrt());
        }
        if let Some(cov) = inference.beta_covariance_frequentist.as_mut() {
            *cov = insert_symmetric_array2(cov, index, 0.0)?;
        }
        if let Some(influence) = inference.coefficient_influence.as_mut() {
            *influence = insert_symmetric_array2(influence, index, 0.0)?;
        }
        if let Some(correction) = inference.smoothing_correction.as_mut() {
            *correction = insert_symmetric_array2(correction, index, 0.0)?;
        }
        if let Some(qs) = inference.reparam_qs.as_mut() {
            *qs = insert_symmetric_array2(qs, index, 1.0)?;
        }
        if let Some(bias) = inference.bias_correction_beta.as_mut() {
            *bias = insert_array1(bias, index, 0.0);
        }
    }
    if let Some(geometry) = fit.geometry.as_mut() {
        geometry.penalized_hessian =
            insert_symmetric_array2(geometry.penalized_hessian.as_array(), index, precision_diag)?
                .into();
    }
    Ok(())
}

fn insert_array1(values: &Array1<f64>, index: usize, value: f64) -> Array1<f64> {
    let mut out = Vec::<f64>::with_capacity(values.len() + 1);
    out.extend(values.iter().take(index).copied());
    out.push(value);
    out.extend(values.iter().skip(index).copied());
    Array1::from_vec(out)
}

fn insert_symmetric_array2(
    matrix: &Array2<f64>,
    index: usize,
    diagonal: f64,
) -> Result<Array2<f64>, String> {
    if matrix.nrows() != matrix.ncols() {
        return Err(format!(
            "extend_with_group expected square matrix, got {}x{}",
            matrix.nrows(),
            matrix.ncols()
        ));
    }
    if index > matrix.nrows() {
        return Err(format!(
            "extend_with_group matrix insert index {index} exceeds dimension {}",
            matrix.nrows()
        ));
    }
    let n = matrix.nrows();
    let mut out = Array2::<f64>::zeros((n + 1, n + 1));
    for old_i in 0..n {
        let new_i = if old_i < index { old_i } else { old_i + 1 };
        for old_j in 0..n {
            let new_j = if old_j < index { old_j } else { old_j + 1 };
            out[[new_i, new_j]] = matrix[[old_i, old_j]];
        }
    }
    out[[index, index]] = diagonal;
    Ok(out)
}

fn validate_formula_json_impl(
    headers: Vec<String>,
    rows: Vec<Vec<String>>,
    formula: String,
    config_json: Option<&str>,
) -> Result<String, String> {
    let mut dataset = dataset_with_inferred_schema(headers, rows)?;
    let (mut fit_config, _training_table_kind) = parse_fit_config(config_json)?;
    // Calibrated marginal-slope chain (#461): validation is purely structural and
    // must stay cheap — it must NOT cross-fit Stage-1. When a CTN Stage-1 recipe
    // is present, strip it and stand in a zero-valued placeholder dose column so
    // the Stage-2 marginal-slope structure validates without any Stage-1 fit or
    // cross-fit. The real fit produces `z` out-of-fold and needs no such column.
    if fit_config.ctn_stage1.is_some() {
        const VALIDATION_PLACEHOLDER_Z: &str = "__gam_validation_ctn_stage1_z";
        fit_config.ctn_stage1 = None;
        if dataset
            .headers
            .iter()
            .any(|name| name == VALIDATION_PLACEHOLDER_Z)
        {
            return Err(format!(
                "reserved validation column '{VALIDATION_PLACEHOLDER_Z}' already exists in the \
                 input data; rename it before validating the calibrated chain"
            ));
        }
        let n = dataset.values.nrows();
        let old_cols = dataset.values.ncols();
        let mut values = Array2::<f64>::zeros((n, old_cols + 1));
        values.slice_mut(s![.., ..old_cols]).assign(&dataset.values);
        dataset.values = values;
        dataset.headers.push(VALIDATION_PLACEHOLDER_Z.to_string());
        dataset
            .column_kinds
            .push(gam::data::ColumnKindTag::Continuous);
        dataset.schema.columns.push(gam::data::SchemaColumn {
            name: VALIDATION_PLACEHOLDER_Z.to_string(),
            kind: gam::data::ColumnKindTag::Continuous,
            levels: Vec::new(),
        });
        fit_config.z_column = Some(VALIDATION_PLACEHOLDER_Z.to_string());
    }
    let materialized = materialize(&formula, &dataset, &fit_config)?;
    let (family_name, model_class, supported_by_python) = request_metadata(&materialized.request);
    let response_column = response_column_name(&formula);
    let payload = ValidationPayload {
        formula,
        family_name: family_name.to_string(),
        model_class: model_class.to_string(),
        response_column,
        columns: dataset.headers.clone(),
        n_rows: dataset.values.nrows(),
        n_columns: dataset.values.ncols(),
        supported_by_python,
    };
    serde_json::to_string(&payload)
        .map_err(|err| format!("failed to serialize validation payload: {err}"))
}

fn parse_formula_validation_payload_json(
    payload_json: &str,
) -> Result<serde_json::Map<String, serde_json::Value>, String> {
    let payload: serde_json::Value = serde_json::from_str(payload_json)
        .map_err(|err| format!("failed to parse FormulaValidation payload: {err}"))?;
    match payload {
        serde_json::Value::Object(map) => Ok(map),
        _ => Err("FormulaValidation payload must be a JSON object".to_string()),
    }
}

fn json_payload_truthy(value: &serde_json::Value) -> bool {
    match value {
        serde_json::Value::Null => false,
        serde_json::Value::Bool(value) => *value,
        serde_json::Value::Number(value) => {
            value.as_f64().map(|number| number != 0.0).unwrap_or(true)
        }
        serde_json::Value::String(value) => !value.is_empty(),
        serde_json::Value::Array(value) => !value.is_empty(),
        serde_json::Value::Object(value) => !value.is_empty(),
    }
}

fn python_repr_json_value(value: Option<&serde_json::Value>) -> String {
    match value {
        None | Some(serde_json::Value::Null) => "None".to_string(),
        Some(serde_json::Value::Bool(value)) => {
            if *value {
                "True".to_string()
            } else {
                "False".to_string()
            }
        }
        Some(serde_json::Value::Number(value)) => value.to_string(),
        Some(serde_json::Value::String(value)) => python_repr_string(value),
        Some(value @ serde_json::Value::Array(_)) | Some(value @ serde_json::Value::Object(_)) => {
            python_str_json_value(value)
        }
    }
}

fn python_repr_string(value: &str) -> String {
    let mut out = String::with_capacity(value.len() + 2);
    out.push('\'');
    for ch in value.chars() {
        match ch {
            '\\' => out.push_str("\\\\"),
            '\'' => out.push_str("\\'"),
            '\n' => out.push_str("\\n"),
            '\r' => out.push_str("\\r"),
            '\t' => out.push_str("\\t"),
            ch if ch.is_control() => {
                out.push_str(&format!("\\x{:02x}", ch as u32));
            }
            ch => out.push(ch),
        }
    }
    out.push('\'');
    out
}

fn python_str_json_value(value: &serde_json::Value) -> String {
    match value {
        serde_json::Value::Null => "None".to_string(),
        serde_json::Value::Bool(value) => {
            if *value {
                "True".to_string()
            } else {
                "False".to_string()
            }
        }
        serde_json::Value::Number(value) => value.to_string(),
        serde_json::Value::String(value) => value.clone(),
        serde_json::Value::Array(values) => {
            let inner = values
                .iter()
                .map(|value| match value {
                    serde_json::Value::String(value) => python_repr_string(value),
                    value => python_str_json_value(value),
                })
                .collect::<Vec<_>>()
                .join(", ");
            format!("[{inner}]")
        }
        serde_json::Value::Object(values) => {
            let inner = values
                .iter()
                .map(|(key, value)| {
                    format!(
                        "{}: {}",
                        python_repr_string(key),
                        match value {
                            serde_json::Value::String(value) => python_repr_string(value),
                            value => python_str_json_value(value),
                        }
                    )
                })
                .collect::<Vec<_>>()
                .join(", ");
            format!("{{{inner}}}")
        }
    }
}

fn escape_html(value: &str) -> String {
    let mut out = String::with_capacity(value.len());
    for ch in value.chars() {
        match ch {
            '&' => out.push_str("&amp;"),
            '<' => out.push_str("&lt;"),
            '>' => out.push_str("&gt;"),
            '"' => out.push_str("&quot;"),
            '\'' => out.push_str("&#x27;"),
            ch => out.push(ch),
        }
    }
    out
}

fn predict_table_impl(
    model_bytes: &[u8],
    headers: Vec<String>,
    rows: Vec<Vec<String>>,
    options_json: Option<&str>,
) -> Result<String, PredictError> {
    let model = load_model_impl(model_bytes)?;
    let model_class = model.predict_model_class();
    let dataset = dataset_with_model_schema_typed(&model, &headers, &rows)?;
    drop(rows);
    drop(headers);
    predict_dataset_impl(&model, model_class, dataset, options_json).map_err(PredictError::Other)
}

fn predict_array_impl(
    model_bytes: &[u8],
    x: ArrayView2<'_, f64>,
    options_json: Option<&str>,
) -> Result<Array2<f64>, String> {
    let model = load_model_impl(model_bytes)?;
    let model_class = model.predict_model_class();
    let dataset = dataset_from_x_array_with_model_schema(&model, x)?;
    if matches!(model_class, PredictModelClass::Survival) {
        return Err("predict_array does not support survival prediction payloads".to_string());
    }
    let options = parse_predict_options(options_json)?;
    let columns = predict_columns(&model, dataset, &options)?;
    // Parity with `predict()` (#1537): with no interval requested, return the
    // single response-scale `mean` column as an `(n, 1)` array — the Python
    // wrapper ravels it to the documented 1-D response-scale prediction vector.
    // Returning the full alphabetical `[linear_predictor, mean]` matrix here
    // both breaks shape parity and silently hands a naive `[:, 0]` / `.ravel()`
    // caller the LINK-scale linear predictor on non-identity links. With an
    // interval the full column matrix is retained for the array caller.
    if options.interval.is_none() {
        let mean = columns
            .get("mean")
            .ok_or_else(|| "predict_array: response `mean` column missing".to_string())?;
        return Array2::from_shape_vec((mean.len(), 1), mean.clone())
            .map_err(|err| format!("predict_array: failed to shape mean column: {err}"));
    }
    columns_to_array(columns)
}

fn predict_dataset_impl(
    model: &FittedModel,
    model_class: PredictModelClass,
    dataset: EncodedDataset,
    options_json: Option<&str>,
) -> Result<String, String> {
    let options = parse_predict_options(options_json)?;
    if matches!(model_class, PredictModelClass::Survival) {
        return predict_table_survival(model, &dataset, &options);
    }
    let columns = predict_columns(model, dataset, &options)?;
    serde_json::to_string(&PredictionPayload {
        columns,
        model_class: prediction_model_class_label(model),
        family: family_link_kind(&model_likelihood_spec(model)).to_string(),
        // The plain dataset predict path returns the model-based credible /
        // predictive band (or no interval at all); the jackknife+ provenance
        // tag is only attached by the dedicated full-conformal predict entry.
        interval_method: None,
    })
    .map_err(|err| format!("failed to serialize prediction payload: {err}"))
}

fn predict_columns(
    model: &FittedModel,
    dataset: EncodedDataset,
    options: &PyPredictOptions,
) -> Result<BTreeMap<String, Vec<f64>>, String> {
    let col_map = dataset.column_map();
    // Spline-scan saved model (#1030/#1034): replay the exact Gaussian bridge
    // per row (identity link, η == mean). No design reconstruction, no
    // predictor. SE/intervals come from the exact posterior variance.
    if let Some((feature_column, fit)) = model.saved_spline_scan().map_err(String::from)? {
        let col = *col_map.get(feature_column).ok_or_else(|| {
            format!("prediction data is missing the model's feature column '{feature_column}'")
        })?;
        let n = dataset.values.nrows();
        let mut eta = Vec::with_capacity(n);
        let mut se = Vec::with_capacity(n);
        for (i, &x) in dataset.values.column(col).iter().enumerate() {
            let (m, v) = fit
                .predict(x)
                .map_err(|e| format!("spline-scan predict failed at row {i}: {e}"))?;
            eta.push(m);
            se.push(v.max(0.0).sqrt());
        }
        let mut columns = BTreeMap::<String, Vec<f64>>::new();
        columns.insert("linear_predictor".to_string(), eta.clone());
        columns.insert("mean".to_string(), eta.clone());
        if let Some(confidence_level) = options.interval {
            let z = gam::inference::probability::standard_normal_quantile(
                0.5 + confidence_level * 0.5,
            )?;
            let lower: Vec<f64> = eta.iter().zip(&se).map(|(m, s)| m - z * s).collect();
            let upper: Vec<f64> = eta.iter().zip(&se).map(|(m, s)| m + z * s).collect();
            // Observation (predictive) interval (#1047): the scan IS the exact
            // Gaussian smoothing-spline posterior, so the response-scale
            // predictive variance is `Var(f(x*)) + σ²` where `Var(f(x*)) = se²`
            // is the off-knot posterior variance from the bridge and
            // `fit.sigma2` is the profiled Gaussian observation variance. The
            // confidence band uses `se` alone; the observation band inflates by
            // σ². Identity link means no response-scale transform is needed.
            if options.observation_interval.unwrap_or(false) {
                let obs_se: Vec<f64> = se
                    .iter()
                    .map(|s| (s * s + fit.sigma2).max(0.0).sqrt())
                    .collect();
                let obs_lower: Vec<f64> = eta.iter().zip(&obs_se).map(|(m, s)| m - z * s).collect();
                let obs_upper: Vec<f64> = eta.iter().zip(&obs_se).map(|(m, s)| m + z * s).collect();
                columns.insert("observation_lower".to_string(), obs_lower);
                columns.insert("observation_upper".to_string(), obs_upper);
            }
            columns.insert("std_error".to_string(), se);
            columns.insert("mean_lower".to_string(), lower);
            columns.insert("mean_upper".to_string(), upper);
        }
        return Ok(columns);
    }
    let offset = resolve_offset_column(&dataset, &col_map, model.offset_column.as_deref())?;
    let offset_noise =
        resolve_offset_column(&dataset, &col_map, model.noise_offset_column.as_deref())?;
    // Resolve the analytic prior-weights column from the PREDICTION frame exactly
    // as `generative_replicates_impl`/`sample_replicates` does (#2077). A weighted
    // Gaussian fit has `Var(y_i) = σ²/w_i`, so the analytic observation
    // (prediction) interval's conditional response variance is per-row `σ̂²/w_i`,
    // not the pooled scalar `σ̂²` broadcast to every row. `resolve_weight_column`
    // returns unit weights when the model carried no weight column, and we only
    // forward weights when the model was actually fitted with a weight column that
    // is present in the prediction frame — so unweighted fits stay byte-identical
    // and a missing column degrades to unit weights (the correct default).
    let weight_column = model
        .weight_column
        .as_deref()
        .filter(|name| col_map.contains_key(*name));
    let observation_prior_weights = match weight_column {
        Some(_) => Some(resolve_weight_column(&dataset, &col_map, weight_column)?),
        None => None,
    };
    let predict_input = build_predict_input_for_model(
        &model,
        dataset.values.view(),
        &col_map,
        model.training_headers.as_ref(),
        &offset,
        &offset_noise,
        false,
    )?;
    let predictor = model
        .predictor()
        .ok_or_else(|| "saved model could not construct a predictor".to_string())?;
    let fit = fit_result_from_saved_model_for_prediction(model)?;

    let mut columns = BTreeMap::<String, Vec<f64>>::new();
    // SPEC: the posterior mean `E[g⁻¹(Xβ)]` is *always* the default point
    // estimate (never the plug-in mode `g⁻¹(Xβ̂)`). The integral is only
    // observably distinct from the plug-in when the inverse link is curved
    // over the posterior's support, so `FittedModel::prediction_uses_posterior_mean`
    // — the single predicate shared with the CLI's `gam predict` — selects the
    // posterior-mean path for exactly those models and keeps the cheaper,
    // exact plug-in for effectively-linear ones (identity-link Gaussian, …).
    // Reported point estimates therefore match the CLI default row-for-row.
    let uses_posterior_mean = model.prediction_uses_posterior_mean();
    // Issue #342 — single uncertainty knob: `interval` is the only switch.
    // `Some(level)` means "quantify uncertainty at this coverage and return
    // SE + CI bounds"; `None` means "point predictions only". The earlier
    // overlapping `with_uncertainty` boolean has been removed: it was
    // strictly weaker than `interval` (SE is a strict subset of
    // SE + bounds), and supporting both forced callers to learn two
    // partially-redundant flags. Migration: `with_uncertainty=True` →
    // `interval=0.95`.
    //
    // Issue #398: the point prediction is a property of the model and the
    // inputs, never of whether an interval was requested — `interval` only
    // *adds* std_error/mean_lower/mean_upper columns and never shifts `mean`
    // or `linear_predictor`. That invariant holds on both axes here: the
    // posterior-mean vs plug-in choice is driven solely by the model
    // (`uses_posterior_mean`), and within each axis the interval and
    // no-interval branches report the identical point.
    //
    // User-facing column names follow issue #310: the engine's internal labels
    // ("eta", "effective_se", "effective_variance") are renamed at the FFI
    // boundary to linear_predictor / std_error, and effective_variance is
    // dropped (== std_error ** 2). The internal Rust fields keep their
    // theoretic names because they describe the math object.
    match (options.interval, uses_posterior_mean) {
        (Some(confidence_level), true) => {
            // Curved inverse link + interval: the canonical posterior-mean path
            // returns the η-scale SE and the inverse-link-transformed credible
            // bounds in one pass, on top of the same posterior-mean point as the
            // no-interval branch. The `covariance_mode` and `observation_interval`
            // knobs are threaded through here exactly as the sibling
            // full-uncertainty arm does for the plug-in families: previously this
            // arm ignored both, so binomial intervals silently dropped the
            // observation band (#811) and the smoothing-parameter correction
            // (#812). The posterior-mean *point* stays conditional regardless of
            // mode (issue #398); only the reported uncertainty responds.
            let covariance_mode = parse_covariance_mode(options.covariance_mode.as_deref())?
                .unwrap_or(gam_predict::InferenceCovarianceMode::ConditionalPlusSmoothingPreferred);
            let posterior_options = gam_predict::PosteriorMeanOptions {
                confidence_level: Some(confidence_level),
                covariance_mode,
                include_observation_interval: options.observation_interval.unwrap_or(false),
            };
            let prediction = predictor
                .predict_posterior_mean(&predict_input, &fit, &posterior_options)
                .map_err(|err| {
                    format!("posterior-mean prediction with uncertainty failed: {err}")
                })?;
            let (mean_lower, mean_upper) = prediction
                .mean_lower
                .zip(prediction.mean_upper)
                .ok_or_else(|| {
                    "posterior-mean prediction did not return confidence bounds".to_string()
                })?;
            columns.insert("linear_predictor".to_string(), prediction.eta.to_vec());
            columns.insert("mean".to_string(), prediction.mean.to_vec());
            // `std_error` is documented and laid out as the response-scale SE
            // (beside the response-scale `mean`/`mean_lower`/`mean_upper`), so
            // emit the response-scale `mean_standard_error` — the SE the band is
            // built from — not the link-scale `eta_standard_error` (#1536). The
            // posterior-mean path always populates it once an interval is
            // requested; fall back to the link-scale SE only for the (here
            // unreachable) point-only case.
            let response_se = prediction
                .mean_standard_error
                .clone()
                .unwrap_or_else(|| prediction.eta_standard_error.clone());
            columns.insert("std_error".to_string(), response_se.to_vec());
            columns.insert("mean_lower".to_string(), mean_lower.to_vec());
            columns.insert("mean_upper".to_string(), mean_upper.to_vec());
            // Observation (prediction) interval: present only when the request
            // was made AND the family exposes a conditional response variance
            // (Binomial `p(1−p)`). Emitting separate columns keeps the standard
            // schema untouched when off and never overwrites the credible bounds.
            if let (Some(obs_lower), Some(obs_upper)) =
                (prediction.observation_lower, prediction.observation_upper)
            {
                columns.insert("observation_lower".to_string(), obs_lower.to_vec());
                columns.insert("observation_upper".to_string(), obs_upper.to_vec());
            }
        }
        (Some(confidence_level), false) => {
            // Effectively-linear model + interval: plug-in == posterior mean, so
            // the delta-method full-uncertainty path reports that same point and
            // only widens the interval for smoothing uncertainty.
            // `apply_bias_correction: false` keeps the point equal to the plain
            // plug-in branch: recentring η by X·H⁻¹Sβ̂ would silently shift `mean`
            // the moment an interval was requested (violating issue #398), is
            // empirically worse against truth, and is inconsistent with the
            // link-wiggle path that never bias-corrects; bias-aware coverage is
            // already supplied by the smoothing-corrected covariance.
            //
            // CLI<->Python parity: `covariance_mode` (default smoothing-preferred,
            // matching the prior hardcode) and `observation_interval` are now
            // user-selectable, mirroring `gam predict --covariance-mode` and the
            // engine's `includeobservation_interval` switch.
            let covariance_mode = parse_covariance_mode(options.covariance_mode.as_deref())?
                .unwrap_or(gam_predict::InferenceCovarianceMode::ConditionalPlusSmoothingPreferred);
            let includeobservation_interval = options.observation_interval.unwrap_or(false);
            let uncertainty_options = gam_predict::PredictUncertaintyOptions {
                confidence_level,
                covariance_mode,
                mean_interval_method: gam_predict::MeanIntervalMethod::TransformEta,
                includeobservation_interval,
                apply_bias_correction: false,
                // Weighted-Gaussian observation band is heteroscedastic in the
                // per-row prior weight `Var(y_i)=σ̂²/w_i` (#2077); unweighted fits
                // pass `None` and stay byte-identical.
                observation_prior_weights: observation_prior_weights.clone(),
                ..gam_predict::PredictUncertaintyOptions::default()
            };
            let prediction = predictor
                .predict_full_uncertainty(&predict_input, &fit, &uncertainty_options)
                .map_err(|err| format!("prediction with uncertainty failed: {err}"))?;
            columns.insert("linear_predictor".to_string(), prediction.eta.to_vec());
            columns.insert("mean".to_string(), prediction.mean.to_vec());
            // Response-scale SE beside the response-scale mean/band (#1536):
            // emit `mean_standard_error`, not the link-scale `eta_standard_error`.
            columns.insert(
                "std_error".to_string(),
                prediction.mean_standard_error.to_vec(),
            );
            columns.insert("mean_lower".to_string(), prediction.mean_lower.to_vec());
            columns.insert("mean_upper".to_string(), prediction.mean_upper.to_vec());
            // Observation (prediction) interval: only present when the family
            // and the `observation_interval` request both support it. Emitting
            // it as separate columns keeps the standard schema untouched when
            // off and never overwrites the credible `mean_lower`/`mean_upper`.
            if let (Some(obs_lower), Some(obs_upper)) =
                (prediction.observation_lower, prediction.observation_upper)
            {
                columns.insert("observation_lower".to_string(), obs_lower.to_vec());
                columns.insert("observation_upper".to_string(), obs_upper.to_vec());
            }
        }
        (None, true) => {
            let prediction = predictor
                .predict_posterior_mean(
                    &predict_input,
                    &fit,
                    &gam_predict::PosteriorMeanOptions::point_only(),
                )
                .map_err(|err| format!("posterior-mean prediction failed: {err}"))?;
            columns.insert("linear_predictor".to_string(), prediction.eta.to_vec());
            columns.insert("mean".to_string(), prediction.mean.to_vec());
        }
        (None, false) => {
            let prediction = predictor
                .predict_plugin_response(&predict_input)
                .map_err(|err| format!("prediction failed: {err}"))?;
            columns.insert("linear_predictor".to_string(), prediction.eta.to_vec());
            columns.insert("mean".to_string(), prediction.mean.to_vec());
        }
    }

    // Issue #365 (secondary defect): location-scale / GAMLSS families fit a
    // per-observation distribution scale (e.g. Gaussian `sigma = exp(noise η)`)
    // that was previously unreachable from Python — `predict_columns` only ever
    // emitted the location block (`mean` / `linear_predictor`), so a user could
    // fit a smooth `noise_formula` but never retrieve the scale function it
    // learned. The scale is a model property independent of the uncertainty
    // knob, so it is emitted in both branches. Families without a response-side
    // scale (single-block GLMs, binomial location-scale, survival, …) return
    // `None` from `predict_noise_scale` and contribute no column, leaving the
    // standard prediction schema untouched.
    if let Some(noise_scale) = predictor
        .predict_noise_scale(&predict_input)
        .map_err(|err| format!("noise-scale prediction failed: {err}"))?
    {
        columns.insert("noise_scale".to_string(), noise_scale.to_vec());
    }

    Ok(columns)
}

/// Build the held-out calibration fold needed by the conformal calibrator: a
/// `PredictInput` over the calibration design (so the model's own predict
/// engine produces `μ̂(x_cal)` and `s(x_cal)` at exactly those points,
/// identically to the test path) and the calibration response `y_cal`. The
/// response column is resolved from the saved formula and must be present in
/// the calibration dataset (calibration is *labeled* held-out data, unlike a
/// predict batch). The fold carries its own design and may be of ANY size,
/// independent of the training set — it is never bound to the training rows.
fn conformal_calibration_fold(
    model: &FittedModel,
    fit: &gam::solver::estimate::UnifiedFitResult,
    calibration: EncodedDataset,
) -> Result<(gam_predict::PredictInput, Array1<f64>), String> {
    if !matches!(model.predict_model_class(), PredictModelClass::Standard) {
        return Err(format!(
            "conformal calibration currently supports only standard GAM models; got '{}'",
            prediction_model_class_label(model)
        ));
    }
    let col_map = calibration.column_map();
    let offset = resolve_offset_column(&calibration, &col_map, model.offset_column.as_deref())?;
    let offset_noise =
        resolve_offset_column(&calibration, &col_map, model.noise_offset_column.as_deref())?;
    let response_name = response_column_name(&model.payload().formula).ok_or_else(|| {
        "conformal calibration: could not resolve the response column from the saved formula"
            .to_string()
    })?;
    let response_col = *col_map.get(&response_name).ok_or_else(|| {
        format!(
            "conformal calibration data must contain the response column '{response_name}' \
             (calibration is held-out labeled data)"
        )
    })?;
    let y = calibration.values.column(response_col).to_owned();
    // Build the calibration-fold predict input the same way the test batch is
    // built, so the predict engine yields μ̂(x_cal) and s(x_cal) from exactly
    // the same source used at test time.
    let cal_input = build_predict_input_for_model(
        model,
        calibration.values.view(),
        &col_map,
        model.training_headers.as_ref(),
        &offset,
        &offset_noise,
        false,
    )?;
    let design_cols = cal_input.design.ncols();
    if design_cols != fit.beta.len() {
        return Err(format!(
            "conformal calibration design has {} columns but the fit has {} coefficients",
            design_cols,
            fit.beta.len()
        ));
    }
    Ok((cal_input, y))
}

/// Conformal-calibrated prediction columns. Runs the model-based full-
/// uncertainty predictor on the test `dataset` (honouring `covariance_mode` /
/// `observation_interval`), then replaces the response-scale `mean_lower` /
/// `mean_upper` with the split-conformal interval calibrated from the supplied
/// held-out `calibration` fold at the level in `options.conformal_level`.
fn predict_columns_conformal(
    model: &FittedModel,
    dataset: EncodedDataset,
    calibration: EncodedDataset,
    options: &PyPredictOptions,
) -> Result<BTreeMap<String, Vec<f64>>, String> {
    let Some(level) = options.conformal_level else {
        return Err("conformal prediction requires conformal_level in (0, 1)".to_string());
    };
    if !matches!(model.predict_model_class(), PredictModelClass::Standard) {
        return Err(format!(
            "conformal prediction currently supports only standard GAM models; got '{}'",
            prediction_model_class_label(model)
        ));
    }
    let col_map = dataset.column_map();
    let offset = resolve_offset_column(&dataset, &col_map, model.offset_column.as_deref())?;
    let offset_noise =
        resolve_offset_column(&dataset, &col_map, model.noise_offset_column.as_deref())?;
    let predict_input = build_predict_input_for_model(
        model,
        dataset.values.view(),
        &col_map,
        model.training_headers.as_ref(),
        &offset,
        &offset_noise,
        false,
    )?;
    let predictor = model
        .predictor()
        .ok_or_else(|| "saved model could not construct a predictor".to_string())?;
    let fit = fit_result_from_saved_model_for_prediction(model)?;
    let family = model_likelihood_spec(model);

    let covariance_mode = parse_covariance_mode(options.covariance_mode.as_deref())?
        .unwrap_or(gam_predict::InferenceCovarianceMode::ConditionalPlusSmoothingPreferred);
    let uncertainty_options = gam_predict::PredictUncertaintyOptions {
        confidence_level: level,
        covariance_mode,
        mean_interval_method: gam_predict::MeanIntervalMethod::TransformEta,
        includeobservation_interval: options.observation_interval.unwrap_or(false),
        apply_bias_correction: false,
        conformal_level: Some(level),
        ..gam_predict::PredictUncertaintyOptions::default()
    };

    let (cal_input, cal_y) = conformal_calibration_fold(model, &fit, calibration)?;
    let calibration_fold = gam_predict::ConformalCalibrationFold {
        input: cal_input,
        y: cal_y.view(),
    };
    let prediction = gam_predict::predict_full_uncertainty_conformal(
        predictor.as_ref(),
        &predict_input,
        &fit,
        &family,
        &uncertainty_options,
        &calibration_fold,
    )
    .map_err(|err| format!("conformal prediction failed: {err}"))?;

    let mut columns = BTreeMap::<String, Vec<f64>>::new();
    columns.insert("linear_predictor".to_string(), prediction.eta.to_vec());
    columns.insert("mean".to_string(), prediction.mean.to_vec());
    // Response-scale SE beside the response-scale mean/band (#1536): emit
    // `mean_standard_error`, not the link-scale `eta_standard_error`.
    columns.insert(
        "std_error".to_string(),
        prediction.mean_standard_error.to_vec(),
    );
    // mean_lower / mean_upper now carry the distribution-free conformal bounds.
    columns.insert("mean_lower".to_string(), prediction.mean_lower.to_vec());
    columns.insert("mean_upper".to_string(), prediction.mean_upper.to_vec());
    if let (Some(obs_lower), Some(obs_upper)) =
        (prediction.observation_lower, prediction.observation_upper)
    {
        columns.insert("observation_lower".to_string(), obs_lower.to_vec());
        columns.insert("observation_upper".to_string(), obs_upper.to_vec());
    }
    Ok(columns)
}

fn predict_table_conformal_impl(
    model_bytes: &[u8],
    headers: Vec<String>,
    rows: Vec<Vec<String>>,
    calibration_headers: Vec<String>,
    calibration_rows: Vec<Vec<String>>,
    conformal_level: f64,
    options_json: Option<&str>,
) -> Result<String, String> {
    let model = load_model_impl(model_bytes)?;
    // Split-conformal calibration is built on the dense predictor + fit_result,
    // neither of which a scan-routed model carries. Point and posterior-interval
    // prediction (the scan-aware predict path) work for these models, so direct
    // users there rather than failing with the cryptic missing-resolved_termspec
    // error (#1046).
    if let Some(scan) = scan_introspection(&model)? {
        return Err(format!(
            "{} is fit by the exact O(n) state-space spline scan, which does not \
             carry the dense predictor split-conformal calibration needs. Use \
             predict(..., interval=<level>) for posterior intervals (scan-aware), \
             or refit with double_penalty=true for conformal intervals.",
            scan_smooth_label(&scan)
        ));
    }
    let mut options = parse_predict_options(options_json)?;
    if !(conformal_level.is_finite() && conformal_level > 0.0 && conformal_level < 1.0) {
        return Err(format!(
            "conformal_level must be in (0, 1), got {conformal_level}"
        ));
    }
    options.conformal_level = Some(conformal_level);
    let dataset = dataset_with_model_schema(&model, &headers, &rows)?;
    drop(rows);
    drop(headers);
    let calibration = dataset_with_model_schema(&model, &calibration_headers, &calibration_rows)?;
    drop(calibration_rows);
    drop(calibration_headers);
    let columns = predict_columns_conformal(&model, dataset, calibration, &options)?;
    serde_json::to_string(&PredictionPayload {
        columns,
        model_class: prediction_model_class_label(&model),
        family: family_link_kind(&model_likelihood_spec(&model)).to_string(),
        interval_method: Some(
            "split-conformal (distribution-free, finite-sample marginal coverage)".to_string(),
        ),
    })
    .map_err(|err| format!("failed to serialize conformal prediction payload: {err}"))
}

/// #1054 Exact Gaussian jackknife+ conformal intervals — no calibration fold.
///
/// Reads the `GaussianJackknifePlusStats` precomputed at fit time (only
/// available for Gaussian-identity, unit-weight, offset-free models without a
/// link wiggle), builds the test design from the saved `resolved_termspec`,
/// and calls `stats.interval(x_*, alpha)` per test row. Returns the same
/// column schema as the model-based predict path so Python can shape-route
/// through the existing machinery.
///
/// Falls back with a clear error when the model is ineligible (non-Gaussian
/// family, scan-routed model, link wiggle, weighted training data, or an older
/// serialised payload that pre-dates the jackknife+ precomputation).
fn predict_table_jackknife_plus_impl(
    model_bytes: &[u8],
    headers: Vec<String>,
    rows: Vec<Vec<String>>,
    conformal_level: f64,
) -> Result<String, String> {
    if !(conformal_level > 0.0 && conformal_level < 1.0) {
        return Err(format!(
            "conformal_level must be in (0, 1), got {conformal_level}"
        ));
    }
    let model = load_model_impl(model_bytes)?;
    // Reject the scan path immediately — it never builds a dense design, so the
    // jackknife+ stats are absent and the termspec-based design reconstruction
    // below would not apply.
    if scan_introspection(&model).map_err(String::from)?.is_some() {
        return Err(
            "jackknife+ conformal intervals require a penalised-spline (B-spline) model; \
             this model was fit by the exact O(n) state-space scan. Refit with \
             double_penalty=true to obtain the standard model that carries jackknife+ stats."
                .to_string(),
        );
    }
    let stats = model.gaussian_jackknife_plus.as_ref().ok_or_else(|| {
        "jackknife+ conformal intervals require a Gaussian-identity GLM trained without \
         prior weights, offsets, or a link wiggle. This model is ineligible (non-Gaussian \
         family, weighted data, offset, link wiggle, or an older serialised payload). \
         Use Model.predict_conformal(calibration=...) for split-conformal intervals on \
         arbitrary families."
            .to_string()
    })?;
    if !matches!(model.predict_model_class(), PredictModelClass::Standard) {
        return Err(format!(
            "jackknife+ conformal prediction supports only standard GAM models; got '{}'",
            prediction_model_class_label(&model)
        ));
    }
    // Build test design via the frozen resolved_termspec so column ordering
    // and spline knots are identical to the training design the stats were
    // computed from.
    let dataset = dataset_with_model_schema(&model, &headers, &rows)?;
    let col_map = dataset.column_map();
    let spec = gam::families::survival::predict::resolve_termspec_for_prediction(
        &model.resolved_termspec,
        model.training_headers.as_ref(),
        &col_map,
        "resolved_termspec",
    )?;
    let design = gam::terms::smooth::build_term_collection_design(dataset.values.view(), &spec)
        .map_err(|err| format!("jackknife+ conformal: failed to build test design: {err}"))?;
    let x_test = design
        .design
        .try_to_dense_by_chunks("jackknife+ conformal test design")?;
    let n_test = x_test.nrows();
    if x_test.ncols() != stats.p() {
        return Err(format!(
            "jackknife+ conformal: test design has {} columns but the stored stats have p={}; \
             the model may need to be refit",
            x_test.ncols(),
            stats.p()
        ));
    }
    // Plug-in mean (= beta-hat @ x_star for the Gaussian-identity model); we
    // read it from the stored stats' beta directly to avoid touching the
    // predictor stack.
    // The jackknife+ theorem (Barber et al. 2021) guarantees
    // P(Y_* ∈ Ĉ_α) ≥ 1 − 2α, while the set built at parameter α delivers
    // ≈ 1 − α marginal coverage on exchangeable data in practice. Running at
    // α = (1 − level)/2 to make the worst-case bound read "≥ level" was
    // measured to systematically over-cover at (1 + level)/2 (#1546), so this
    // path TARGETS the requested level with α = 1 − conformal_level — and
    // every claim surface must then state the honest finite-sample floor at
    // this setting, 1 − 2α = 2·level − 1, never "≥ level" (the theorem does
    // not deliver that here).
    let alpha = 1.0 - conformal_level;
    let mut mean_vec = Vec::with_capacity(n_test);
    let mut lower_vec = Vec::with_capacity(n_test);
    let mut upper_vec = Vec::with_capacity(n_test);
    for i in 0..n_test {
        let x_star = x_test.row(i).to_owned();
        let iv = stats
            .interval(&x_star, alpha)
            .map_err(|e| format!("jackknife+ conformal at row {i}: {e}"))?;
        mean_vec.push(x_star.dot(stats.beta()));
        lower_vec.push(iv.lo);
        upper_vec.push(iv.hi);
    }
    let mut columns = BTreeMap::<String, Vec<f64>>::new();
    // Identity link: linear_predictor == mean.
    columns.insert("linear_predictor".to_string(), mean_vec.clone());
    columns.insert("mean".to_string(), mean_vec);
    columns.insert("mean_lower".to_string(), lower_vec);
    columns.insert("mean_upper".to_string(), upper_vec);
    serde_json::to_string(&PredictionPayload {
        columns,
        model_class: prediction_model_class_label(&model),
        family: family_link_kind(&model_likelihood_spec(&model)).to_string(),
        interval_method: Some(format!(
            "jackknife+ targeting {:.0}% coverage (distribution-free finite-sample \
             guarantee ≥{:.0}%; Barber et al. 2021, ≥ 1 − 2α)",
            conformal_level * 100.0,
            (2.0 * conformal_level - 1.0).max(0.0) * 100.0
        )),
    })
    .map_err(|err| format!("failed to serialize jackknife+ prediction payload: {err}"))
}

/// #1098 Gaussian full-conformal prediction set at frozen `Sλ` — no
/// calibration fold.
///
/// Reads the `ExactFullConformalSubstrate` precomputed at fit time (only
/// available for Gaussian-identity, unit-weight, offset-free models without a
/// link wiggle), rebuilds the test design from the saved `resolved_termspec`,
/// and calls `substrate.interval(x_*, alpha)` per test row — one Cholesky each,
/// zero refits. The set is exact *given the frozen penalty*; because the fitted
/// λ̂ was selected from all training responses, the frozen-λ score construction
/// is not permutation symmetric in the n+1 augmented points, so the
/// distribution-free finite-sample coverage theorem applies only where the
/// per-row frozen-ρ certificate accepts (`frozen_rho_certified` = 1.0, under
/// the global-ρ grid-Lipschitz assumption); a 0.0 row is the frozen-λ
/// approximation with no finite-sample guarantee. The exact set is a union of
/// intervals; the returned `mean_lower`/`mean_upper` are its outer envelope (a
/// superset).
///
/// `alpha = 1 − conformal_level` (the full-conformal set `C_α` has marginal
/// coverage `≥ 1 − α`, so `conformal_level = 1 − α` directly; unlike jackknife+
/// there is no factor of two).
///
/// Falls back with a clear error when the model is ineligible (non-Gaussian
/// family, scan-routed model, link wiggle, weighted training data, or an older
/// serialised payload that pre-dates the substrate).
fn predict_table_full_conformal_impl(
    model_bytes: &[u8],
    headers: Vec<String>,
    rows: Vec<Vec<String>>,
    conformal_level: f64,
) -> Result<String, String> {
    if !(conformal_level > 0.0 && conformal_level < 1.0) {
        return Err(format!(
            "conformal_level must be in (0, 1), got {conformal_level}"
        ));
    }
    let model = load_model_impl(model_bytes)?;
    if scan_introspection(&model).map_err(String::from)?.is_some() {
        return Err(
            "exact full-conformal intervals require a penalised-spline (B-spline) model; \
             this model was fit by the exact O(n) state-space scan. Refit with \
             double_penalty=true to obtain the standard model that carries the substrate."
                .to_string(),
        );
    }
    let substrate = model.full_conformal.as_ref().ok_or_else(|| {
        "exact full-conformal intervals require a Gaussian-identity GLM trained without \
         prior weights, offsets, or a link wiggle. This model is ineligible (non-Gaussian \
         family, weighted data, offset, link wiggle, or an older serialised payload). \
         Use Model.predict_conformal(calibration=...) for split-conformal intervals on \
         arbitrary families."
            .to_string()
    })?;
    if !matches!(model.predict_model_class(), PredictModelClass::Standard) {
        return Err(format!(
            "exact full-conformal prediction supports only standard GAM models; got '{}'",
            prediction_model_class_label(&model)
        ));
    }
    let dataset = dataset_with_model_schema(&model, &headers, &rows)?;
    let col_map = dataset.column_map();
    let spec = gam::families::survival::predict::resolve_termspec_for_prediction(
        &model.resolved_termspec,
        model.training_headers.as_ref(),
        &col_map,
        "resolved_termspec",
    )?;
    let design = gam::terms::smooth::build_term_collection_design(dataset.values.view(), &spec)
        .map_err(|err| format!("full conformal: failed to build test design: {err}"))?;
    let x_test = design
        .design
        .try_to_dense_by_chunks("full conformal test design")?;
    let n_test = x_test.nrows();
    if x_test.ncols() != substrate.p() {
        return Err(format!(
            "full conformal: test design has {} columns but the stored substrate has p={}; \
             the model may need to be refit",
            x_test.ncols(),
            substrate.p()
        ));
    }
    // A symmetric full-conformal set C_α covers Y_* with marginal probability
    // ≥ 1 − α, so the user's conformal_level maps directly to
    // α = 1 − conformal_level (no factor-of-two as in the jackknife+
    // ≥ 1 − 2α guarantee). At frozen λ̂ that theorem is conditional on the
    // per-row frozen-ρ certificate — see the function doc.
    let alpha = 1.0 - conformal_level;
    let mut mean_vec = Vec::with_capacity(n_test);
    let mut lower_vec = Vec::with_capacity(n_test);
    let mut upper_vec = Vec::with_capacity(n_test);
    let mut certified_vec = Vec::with_capacity(n_test);
    for i in 0..n_test {
        let x_star = x_test.row(i).to_owned();
        let iv = substrate
            .interval(&x_star, alpha)
            .map_err(|e| format!("full conformal at row {i}: {e}"))?;
        // Report the envelope centre as the point summary when finite; an
        // unbounded honest envelope reports its finite endpoint.
        let point = if iv.lo.is_finite() && iv.hi.is_finite() {
            0.5 * (iv.lo + iv.hi)
        } else if iv.lo.is_finite() {
            iv.lo
        } else {
            iv.hi
        };
        mean_vec.push(point);
        lower_vec.push(iv.lo);
        upper_vec.push(iv.hi);
        certified_vec.push(if iv.frozen_rho_certified { 1.0 } else { 0.0 });
    }
    let mut columns = BTreeMap::<String, Vec<f64>>::new();
    columns.insert("linear_predictor".to_string(), mean_vec.clone());
    columns.insert("mean".to_string(), mean_vec);
    columns.insert("mean_lower".to_string(), lower_vec);
    columns.insert("mean_upper".to_string(), upper_vec);
    columns.insert("frozen_rho_certified".to_string(), certified_vec);
    serde_json::to_string(&PredictionPayload {
        columns,
        model_class: prediction_model_class_label(&model),
        family: family_link_kind(&model_likelihood_spec(&model)).to_string(),
        interval_method: Some(format!(
            "full-conformal at frozen smoothing parameters (exact set given Sλ; the \
             distribution-free finite-sample ≥{:.0}% guarantee needs the symmetric \
             ρ-re-selecting fit and is certified per row only where \
             frozen_rho_certified=1, under the global-ρ grid-Lipschitz assumption)",
            conformal_level * 100.0
        )),
    })
    .map_err(|err| format!("failed to serialize full-conformal prediction payload: {err}"))
}

/// Full-conformal prediction intervals at frozen smoothing parameters — no
/// held-out calibration fold required (#1098 / #942 Layer 1).
///
/// Routes `predict(interval='full_conformal')` for Gaussian-identity models to
/// the `ExactFullConformalSubstrate` precomputed at fit time. The set is exact
/// given the frozen `Sλ`; the distribution-free finite-sample
/// ≥`conformal_level` marginal-coverage theorem additionally requires the
/// symmetric ρ-re-selecting fit and is certified per row only where the
/// returned `frozen_rho_certified` column is 1.0 (Layer-3 certificate, under
/// the global-ρ grid-Lipschitz assumption). Returns the same column JSON as
/// `predict_table` plus that certificate column.
///
/// Raises a descriptive Python exception for ineligible models (non-Gaussian,
/// weighted, scan-routed, …) directing the user to `predict_conformal`.
#[pyfunction(signature = (model_bytes, headers, rows, conformal_level=0.9))]
fn predict_table_full_conformal(
    py: Python<'_>,
    model_bytes: Vec<u8>,
    headers: Vec<String>,
    rows: Vec<Vec<String>>,
    conformal_level: f64,
) -> PyResult<String> {
    detach_py_result(py, "predict_table_full_conformal", move || {
        predict_table_full_conformal_impl(&model_bytes, headers, rows, conformal_level)
    })
}

/// Distribution-free jackknife+ conformal prediction intervals — no held-out
/// calibration fold required (#1054 / #942).
///
/// Auto-routes `predict(interval='conformal')` for Gaussian-identity models to
/// the `GaussianJackknifePlusStats` precomputed at fit time. The interval
/// targets ≈`conformal_level` marginal coverage (α = 1 − level, #1546); the
/// distribution-free finite-sample guarantee at that setting is
/// ≥ `2·conformal_level − 1` (Barber et al. 2021, coverage ≥ 1 − 2α).
/// Returns the same column JSON as `predict_table` (with `linear_predictor`,
/// `mean`, `mean_lower`, `mean_upper`) so the Python shaper is unchanged.
///
/// Raises a descriptive Python exception for ineligible models (non-Gaussian,
/// weighted, scan-routed, …) directing the user to `predict_conformal`.
#[pyfunction(signature = (model_bytes, headers, rows, conformal_level=0.9))]
fn predict_table_jackknife_plus(
    py: Python<'_>,
    model_bytes: Vec<u8>,
    headers: Vec<String>,
    rows: Vec<Vec<String>>,
    conformal_level: f64,
) -> PyResult<String> {
    detach_py_result(py, "predict_table_jackknife_plus", move || {
        predict_table_jackknife_plus_impl(&model_bytes, headers, rows, conformal_level)
    })
}

/// #1057 Posterior-predictive replicate sampling — `model.sample_replicates`.
///
/// Draws `n_draws` synthetic response vectors from the fitted predictive
/// distribution via `gam::inference::generative`:
///   1. Run the plug-in predictor to get `mean` (response scale).
///   2. Derive the `NoiseModel` from the saved `LikelihoodSpec` + fitted
///      dispersion (`standard_deviation` / `likelihood_scale`).
///   3. Call `sampleobservation_replicates` with a seeded `StdRng`.
///
/// Returns a row-major flat `Vec<f64>` of shape `(n_draws, n_rows)` plus the
/// two dimensions, so the Python side can reshape into a numpy array without
/// a copy. Survives any family supported by `NoiseModel::from_likelihood`
/// (Gaussian, Poisson, Bernoulli, Gamma, Beta, NegBin, Tweedie). Survival /
/// transformation-normal / scan-routed models are rejected early with a clear
/// error — they are not Gauss GLMs and `generativespec_from_predict` does not
/// cover them.
#[pyfunction]
fn generative_replicates(
    py: Python<'_>,
    model_bytes: Vec<u8>,
    headers: Vec<String>,
    rows: Vec<Vec<String>>,
    n_draws: usize,
    seed: u64,
) -> PyResult<PyObject> {
    let result =
        py.detach(|| generative_replicates_impl(&model_bytes, headers, rows, n_draws, seed));
    match result {
        Ok((flat, n_rows)) => {
            let arr = ndarray::Array2::<f64>::from_shape_vec((n_draws, n_rows), flat)
                .map_err(|e| py_value_error(format!("generative_replicates reshape: {e}")))?;
            Ok(arr.into_pyarray(py).into_any().unbind())
        }
        Err(e) => Err(py_value_error(e)),
    }
}

fn generative_replicates_impl(
    model_bytes: &[u8],
    headers: Vec<String>,
    rows: Vec<Vec<String>>,
    n_draws: usize,
    seed: u64,
) -> Result<(Vec<f64>, usize), String> {
    use gam::inference::generative::{generativespec_from_predict, sampleobservation_replicates};
    use rand::SeedableRng;
    let model = load_model_impl(model_bytes)?;
    // Only standard GAM models have a dense predictor + UnifiedFitResult.
    if !matches!(model.predict_model_class(), PredictModelClass::Standard) {
        return Err(format!(
            "sample_replicates supports only standard GAM models; got '{}'. \
             Use the appropriate posterior sampling method for this model class.",
            prediction_model_class_label(&model)
        ));
    }
    // Scan-routed models: no dense predictor, no family-based noise model.
    if scan_introspection(&model).map_err(String::from)?.is_some() {
        return Err(
            "sample_replicates is not yet supported for exact O(n) scan models; \
             refit with double_penalty=true to obtain the standard B-spline model."
                .to_string(),
        );
    }
    let family = model_likelihood_spec(&model);
    // Reject families generativespec_from_predict cannot handle (custom /
    // survival-parametric / latent-cloglog). They require a different noise
    // model path not yet covered by the built-in generative engine.
    match &family.response {
        gam::types::ResponseFamily::Gaussian
        | gam::types::ResponseFamily::Binomial
        | gam::types::ResponseFamily::Poisson
        | gam::types::ResponseFamily::NegativeBinomial { .. }
        | gam::types::ResponseFamily::Beta { .. }
        | gam::types::ResponseFamily::Gamma
        | gam::types::ResponseFamily::Tweedie { .. } => {}
        other => {
            return Err(format!(
                "sample_replicates does not yet support the '{}' family; \
                 supported families: gaussian, binomial, poisson, negbin, \
                 beta, gamma, tweedie",
                format!("{other:?}")
            ));
        }
    }
    let dataset = dataset_with_model_schema(&model, &headers, &rows)?;
    let col_map = dataset.column_map();
    let offset = resolve_offset_column(&dataset, &col_map, model.offset_column.as_deref())?;
    let offset_noise =
        resolve_offset_column(&dataset, &col_map, model.noise_offset_column.as_deref())?;
    // Resolve the analytic prior-weights column exactly as the mean/noise offsets
    // are resolved above. A weighted Gaussian fit has `Var(y_i) = sigma^2 / w_i`,
    // so replicate observation noise must be heteroskedastic in `w_i`; dropping
    // the weights here drew every row from the pooled scalar `N(mu_i, sigma_hat^2)`
    // (#2025). `resolve_weight_column` returns unit weights when the model carried
    // no weight column, leaving unweighted fits unchanged.
    //
    // If the model was fitted with weights but the caller's replicate frame does
    // not carry that column, fall back to unit weights rather than erroring: the
    // #2025 heteroskedastic contract (Var(y_i)=sigma_hat^2/w_i) degrades to the
    // pooled scalar `N(mu_i, sigma_hat^2)` when per-row weights are unavailable,
    // which is the correct default in the absence of the column.
    let weight_column = model
        .weight_column
        .as_deref()
        .filter(|name| col_map.contains_key(*name));
    let prior_weights = resolve_weight_column(&dataset, &col_map, weight_column)?;
    let predict_input = build_predict_input_for_model(
        &model,
        dataset.values.view(),
        &col_map,
        model.training_headers.as_ref(),
        &offset,
        &offset_noise,
        false,
    )?;
    let predictor = model
        .predictor()
        .ok_or_else(|| "saved model could not construct a predictor".to_string())?;
    let fit = fit_result_from_saved_model_for_prediction(&model)?;
    let prediction = predictor
        .predict_plugin_response(&predict_input)
        .map_err(|e| format!("generative_replicates: prediction failed: {e}"))?;
    let n_rows = prediction.mean.len();
    // Extract the fitted dispersion through the SINGLE canonical picker that the
    // CLI `gam generate` path also uses. This crate previously carried its own
    // inline copy of the mapping, and its NB arm returned the *seed* theta
    // (`Some(*theta)`) instead of `likelihood_scale.negbin_theta()` — so
    // `Model.sample_replicates` drew Negative-Binomial counts at theta = 1
    // regardless of the fitted overdispersion (the live remnant of #1124 in the
    // Python path). Routing through `gam::inference::generative::family_noise_parameter`
    // keeps the supported families and the interpretation of every dispersion
    // parameter identical across the CLI and Python front-ends — the whole point
    // of unifying the picker — so this class of bug cannot diverge again.
    let gaussian_scale = gam::inference::generative::family_noise_parameter(
        fit.likelihood_scale,
        fit.standard_deviation,
        &family,
    );
    // Build the generative specification (mean + noise model).
    let spec =
        generativespec_from_predict(prediction, family, gaussian_scale, Some(&prior_weights))
            .map_err(|e| format!("generative_replicates: spec error: {e}"))?;
    let mut rng = rand::rngs::StdRng::seed_from_u64(seed);
    let draws = sampleobservation_replicates(&spec, n_draws, &mut rng)
        .map_err(|e| format!("generative_replicates: sampling failed: {e}"))?;
    Ok((draws.into_raw_vec_and_offset().0, n_rows))
}

fn columns_to_array(columns: BTreeMap<String, Vec<f64>>) -> Result<Array2<f64>, String> {
    let ordered = ordered_prediction_column_values(&columns);
    let n_cols = ordered.len();
    let n_rows = ordered.first().map(|values| values.len()).unwrap_or(0);
    let mut out = Array2::<f64>::zeros((n_rows, n_cols));
    for (j, values) in ordered.into_iter().enumerate() {
        if values.len() != n_rows {
            return Err("prediction columns have inconsistent lengths".to_string());
        }
        for (i, value) in values.into_iter().enumerate() {
            out[[i, j]] = value;
        }
    }
    Ok(out)
}

fn ordered_prediction_column_values(columns: &BTreeMap<String, Vec<f64>>) -> Vec<Vec<f64>> {
    // Single source of truth for the user-facing column order; the numpy
    // (Array2) path and the Python dict path (`ordered_prediction_columns`)
    // must agree, so both read `PREFERRED_PREDICTION_COLUMNS`.
    let mut out = Vec::<Vec<f64>>::new();
    let mut seen = BTreeSet::<&str>::new();
    for key in PREFERRED_PREDICTION_COLUMNS.iter().copied() {
        if let Some(values) = columns.get(key) {
            out.push(values.clone());
            seen.insert(key);
        }
    }
    for (key, values) in columns {
        if !seen.contains(key.as_str()) {
            out.push(values.clone());
        }
    }
    out
}

fn parse_sample_options(options_json: Option<&str>) -> Result<PySampleOptions, String> {
    match options_json {
        Some(raw) if !raw.trim().is_empty() => serde_json::from_str(raw)
            .map_err(|err| format!("failed to parse sample options json: {err}")),
        _ => Ok(PySampleOptions::default()),
    }
}

fn resolve_nuts_config(model: &FittedModel, options: PySampleOptions) -> NutsConfig {
    // Mirror the CLI's adaptive sizing so Python users get sensible defaults
    // without having to think about chain/warmup counts: NUTS samples needed
    // grow with the coefficient count, so we anchor on the saved beta length.
    let n_base_params = model
        .fit_result
        .as_ref()
        .map(|fr| fr.beta.len())
        .unwrap_or(0);
    let adaptive = NutsConfig::for_dimension(n_base_params);
    NutsConfig {
        n_samples: options.samples.unwrap_or(adaptive.n_samples),
        nwarmup: options.warmup.unwrap_or(adaptive.nwarmup),
        n_chains: options.chains.unwrap_or(adaptive.n_chains),
        target_accept: options.target_accept.unwrap_or(adaptive.target_accept),
        seed: options.seed.unwrap_or(adaptive.seed),
    }
}

#[derive(Serialize)]
struct DesignMatrixPayload {
    x_flat: Vec<f64>,
    n_rows: usize,
    n_cols: usize,
    family_kind: String,
    model_class: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    beta: Option<Vec<f64>>,
    #[serde(skip_serializing_if = "Option::is_none")]
    covariance_flat: Option<Vec<f64>>,
    #[serde(skip_serializing_if = "Option::is_none")]
    covariance_n: Option<usize>,
}

#[derive(Deserialize)]
struct DesignMatrixDensePayload {
    x_flat: Vec<f64>,
    n_rows: usize,
    n_cols: usize,
}

fn design_matrix_table_impl(
    model_bytes: &[u8],
    headers: Vec<String>,
    rows: Vec<Vec<String>>,
) -> Result<String, String> {
    let model = load_model_impl(model_bytes)?;
    let dataset = dataset_with_model_schema(&model, &headers, &rows)?;
    drop(rows);
    drop(headers);
    design_matrix_dataset_impl(&model, dataset)
}

fn design_matrix_payload_to_dense(raw: &str) -> Result<Array2<f64>, String> {
    let payload: DesignMatrixDensePayload = serde_json::from_str(raw)
        .map_err(|err| format!("failed to parse design matrix payload: {err}"))?;
    let expected_len = payload
        .n_rows
        .checked_mul(payload.n_cols)
        .ok_or_else(|| "design matrix payload shape overflow".to_string())?;
    if payload.x_flat.len() != expected_len {
        return Err(format!(
            "design matrix FFI payload shape mismatch: got {} floats, expected {} * {}",
            payload.x_flat.len(),
            payload.n_rows,
            payload.n_cols
        ));
    }
    Array2::from_shape_vec((payload.n_rows, payload.n_cols), payload.x_flat)
        .map_err(|err| format!("failed to reshape design matrix payload: {err}"))
}

fn design_matrix_array_impl(
    model_bytes: &[u8],
    x: ArrayView2<'_, f64>,
) -> Result<Array2<f64>, String> {
    let model = load_model_impl(model_bytes)?;
    let dataset = dataset_from_x_array_with_model_schema(&model, x)?;
    design_matrix_dense(&model, dataset)
}

/// Population variance (divide by `n`, matching numpy `np.var`'s default).
fn population_variance(values: &[f64]) -> f64 {
    population_covariance(values, values)
}

/// Population covariance (divide by `n`, matching `population_variance`).
fn population_covariance(a: &[f64], b: &[f64]) -> f64 {
    let n = a.len();
    debug_assert_eq!(n, b.len());
    if n == 0 {
        return 0.0;
    }
    let mean_a = a.iter().sum::<f64>() / n as f64;
    let mean_b = b.iter().sum::<f64>() / n as f64;
    a.iter()
        .zip(b.iter())
        .map(|(&va, &vb)| (va - mean_a) * (vb - mean_b))
        .sum::<f64>()
        / n as f64
}

/// Per-term partial dependence on a grid table.
///
/// For the requested `term` this evaluates `f_t(x) = X_t(x) β_t` and the
/// matching delta-method standard error `sqrt(diag(X_t V_t X_tᵀ))`, where
/// `V_t` is the term-block of the fitted coefficient covariance. The design
/// build, `β`, `V`, and the term column ranges are all owned by the Rust
/// core; the caller only supplies the evaluation grid.
fn model_partial_dependence_impl(
    model_bytes: &[u8],
    term: &str,
    headers: Vec<String>,
    rows: Vec<Vec<String>>,
) -> Result<(Vec<f64>, Vec<f64>), String> {
    let raw = design_matrix_table_impl(model_bytes, headers, rows)?;
    let x = design_matrix_payload_to_dense(&raw)?;
    let model = load_model_impl(model_bytes)?;
    let fit = fit_result_from_saved_model_for_prediction(&model)?;
    let beta = &fit.beta;
    let cov = fit
        .beta_covariance_corrected()
        .or_else(|| fit.beta_covariance())
        .ok_or_else(|| {
            "model does not contain coefficient covariance; refit with \
             covariance-saving inference enabled"
                .to_string()
        })?;
    let blocks = term_blocks_for_model_impl(model_bytes)?;
    let (start, end) = blocks
        .iter()
        .find(|(name, _, _, _)| name.as_str() == term)
        .map(|(_, _, s, e)| (*s, *e))
        .ok_or_else(|| {
            let available: Vec<&str> = blocks.iter().map(|(n, _, _, _)| n.as_str()).collect();
            format!("partial_dependence: term {term:?} not found; available: {available:?}")
        })?;
    let n = x.nrows();
    let mut predicted = vec![0.0_f64; n];
    let mut se = vec![0.0_f64; n];
    for i in 0..n {
        let xi = x.row(i);
        let mut f = 0.0_f64;
        for c in start..end {
            f += xi[c] * beta[c];
        }
        predicted[i] = f;
        let mut var = 0.0_f64;
        for a in start..end {
            let xa = xi[a];
            for b in start..end {
                var += xa * cov[[a, b]] * xi[b];
            }
        }
        se[i] = var.max(0.0).sqrt();
    }
    Ok((predicted, se))
}

/// Per-term variance share: `cov(X_t β_t, X β) / var(X β)` for each
/// non-intercept term block (or a single `term` when supplied). Evaluated on
/// the caller's grid table; `β` and the term column ranges come from the Rust
/// core.
///
/// This is a genuine variance decomposition: `var(η) = Σ_t cov(f_t, η)`, so
/// the shares sum to exactly 1 (the intercept contributes a constant with
/// zero covariance). Each term's cross-covariance with every other term is
/// split symmetrically — half to each side — which is the Shapley allocation
/// for a sum of terms. The naive `var(f_t) / var(η)` it replaces dropped all
/// cross terms: for `f_1 = x`, `f_2 = -0.9x` it reported shares 100 and 81
/// against a total of 0.01·var(x). A share can exceed 1 or be negative only
/// when terms genuinely anticorrelate, which is honest rather than a bug.
fn model_variance_share_impl(
    model_bytes: &[u8],
    headers: Vec<String>,
    rows: Vec<Vec<String>>,
    term: Option<String>,
) -> Result<Vec<(String, f64)>, String> {
    let raw = design_matrix_table_impl(model_bytes, headers, rows)?;
    let x = design_matrix_payload_to_dense(&raw)?;
    let model = load_model_impl(model_bytes)?;
    let fit = fit_result_from_saved_model_for_prediction(&model)?;
    let beta = &fit.beta;
    let blocks = term_blocks_for_model_impl(model_bytes)?;
    let n = x.nrows();
    let p = beta.len();
    let mut eta = vec![0.0_f64; n];
    for i in 0..n {
        let xi = x.row(i);
        let mut s = 0.0_f64;
        for c in 0..p {
            s += xi[c] * beta[c];
        }
        eta[i] = s;
    }
    let total_var = population_variance(&eta);
    let mut out: Vec<(String, f64)> = Vec::new();
    for (name, kind, start, end) in &blocks {
        if kind.as_str() == "intercept" {
            continue;
        }
        if let Some(t) = term.as_ref() {
            if name.as_str() != t.as_str() {
                continue;
            }
        }
        let mut contrib = vec![0.0_f64; n];
        for i in 0..n {
            let xi = x.row(i);
            let mut s = 0.0_f64;
            for c in *start..*end {
                s += xi[c] * beta[c];
            }
            contrib[i] = s;
        }
        let share = if total_var > 0.0 {
            population_covariance(&contrib, &eta) / total_var
        } else {
            0.0
        };
        out.push((name.clone(), share));
    }
    Ok(out)
}

#[pyfunction]
fn model_partial_dependence(
    py: Python<'_>,
    model_bytes: Vec<u8>,
    term: String,
    headers: Vec<String>,
    rows: Vec<Vec<String>>,
) -> PyResult<(Py<PyArray1<f64>>, Py<PyArray1<f64>>)> {
    let (predicted, se) = detach_py_result(py, "model_partial_dependence", move || {
        model_partial_dependence_impl(&model_bytes, &term, headers, rows)
    })?;
    Ok((
        predicted.into_pyarray(py).unbind(),
        se.into_pyarray(py).unbind(),
    ))
}

#[pyfunction(signature = (model_bytes, headers, rows, term = None))]
fn model_variance_share(
    py: Python<'_>,
    model_bytes: Vec<u8>,
    headers: Vec<String>,
    rows: Vec<Vec<String>>,
    term: Option<String>,
) -> PyResult<Vec<(String, f64)>> {
    detach_py_result(py, "model_variance_share", move || {
        model_variance_share_impl(&model_bytes, headers, rows, term)
    })
}

fn design_matrix_dataset_impl(
    model: &FittedModel,
    dataset: EncodedDataset,
) -> Result<String, String> {
    let dense = design_matrix_dense(model, dataset)?;
    let n_rows = dense.nrows();
    let n_cols = dense.ncols();
    // Row-major flatten matches numpy's default reshape order.
    let x_flat: Vec<f64> = dense.iter().copied().collect();
    let fit = fit_result_from_saved_model_for_prediction(model)?;
    let covariance = fit
        .beta_covariance_corrected()
        .or_else(|| fit.beta_covariance());
    let covariance_n = covariance.map(|c| c.nrows());
    let covariance_flat = covariance.map(|c| c.iter().copied().collect::<Vec<_>>());
    let payload = DesignMatrixPayload {
        x_flat,
        n_rows,
        n_cols,
        family_kind: family_link_kind(&model_likelihood_spec(&model)).to_string(),
        model_class: prediction_model_class_label(model),
        beta: Some(fit.beta.to_vec()),
        covariance_flat,
        covariance_n,
    };
    serde_json::to_string(&payload)
        .map_err(|err| format!("failed to serialize design matrix payload: {err}"))
}

fn design_matrix_dense(
    model: &FittedModel,
    dataset: EncodedDataset,
) -> Result<Array2<f64>, String> {
    // A scan-routed model never materializes a dense B-spline design — the
    // exact O(n) state-space smoother is the whole point — so there is no model
    // matrix to export. Replace the cryptic "missing resolved_termspec" error
    // with a precise, actionable one (#1046).
    if let Some(scan) = scan_introspection(model)? {
        return Err(format!(
            "{} is fit by the exact O(n) state-space spline scan, which does not \
             build a dense design matrix; design_matrix() is unavailable for it. \
             Refit with double_penalty=true if you need the explicit B-spline \
             model matrix.",
            scan_smooth_label(&scan)
        ));
    }
    if !matches!(model.predict_model_class(), PredictModelClass::Standard) {
        return Err(format!(
            "design_matrix currently supports only standard GAM models; got '{}'. \
             For other classes use Model.predict / posterior.predict, which dispatch \
             through the saved-model predictor.",
            prediction_model_class_label(model)
        ));
    }
    if model.has_link_wiggle() {
        return Err(
            "design_matrix does not yet support link-wiggle models because the \
             linear predictor is q0 + B(q0)·theta, not a single X·beta product. \
             Use Model.predict for these models."
                .to_string(),
        );
    }
    let col_map = dataset.column_map();
    let training_headers = model.training_headers.as_ref();
    let spec = gam::families::survival::predict::resolve_termspec_for_prediction(
        &model.resolved_termspec,
        training_headers,
        &col_map,
        "resolved_termspec",
    )?;
    let design = gam::terms::smooth::build_term_collection_design(dataset.values.view(), &spec)
        .map_err(|err| format!("failed to build design matrix: {err}"))?;
    let dense = design
        .design
        .try_to_dense_by_chunks("design_matrix prediction design")?;
    append_deployment_extension_columns(
        model.payload(),
        dataset.values.view(),
        &col_map,
        training_headers,
        dense,
    )
    .map_err(|err| err.to_string())
}

#[derive(Serialize)]
struct PosteriorPredictPayload {
    eta_flat: Vec<f64>,
    n_draws: usize,
    n_rows: usize,
    model_class: String,
    family_kind: String,
    /// Serialized parameterized `InverseLink` (JSON) carrying the per-fit state
    /// the bare `family_kind` tag drops; consumed by `predict_draws` /
    /// `posterior_predict_bands_table` to route the parameterized links'
    /// response-scale draws (issue #1133). `None` falls back to the tag.
    #[serde(skip_serializing_if = "Option::is_none")]
    link_spec: Option<String>,
}

fn posterior_credible_interval_impl(
    samples_flat: Vec<f64>,
    n_draws: usize,
    n_coeffs: usize,
    level: f64,
) -> Result<Vec<f64>, String> {
    gam::inference::posterior::credible_interval(&samples_flat, n_draws, n_coeffs, level)
}

#[derive(Deserialize)]
struct PosteriorCoefficientNamesRequest {
    coefficient_names: Option<serde_json::Value>,
    n_coeffs: usize,
}

fn coefficient_name_from_json(value: &serde_json::Value) -> String {
    match value {
        serde_json::Value::String(name) => name.clone(),
        other => other.to_string(),
    }
}

fn default_coefficient_names(n_coeffs: usize) -> Vec<String> {
    (0..n_coeffs).map(|j| format!("beta_{j}")).collect()
}
