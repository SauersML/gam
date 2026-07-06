// PyO3 boundary for the SAE spectral / routing-geometry diagnostics landed in
// the `gam-sae` crate (`gam::terms::sae`): the dimension spectrometer, the
// block-firing circle-coordinate readout, the routability floor + audit, the
// sparse-dictionary dual certificate, and the contract-composition / loop
// holonomy calculus.
//
// NB: plain `//` comments, NOT `//!` inner-doc — this file is an `include!`
// fragment textually inlined into `lib.rs` AFTER other items, where an inner
// doc comment is a hard error (E0753: inner attributes/doc must lead the
// enclosing module). Every sibling entrypoint fragment (`model_ffi.rs`, …)
// leads with plain comments for the same reason.
//
// The fragment shares the crate-root namespace with the other entrypoint
// fragments and reaches the shared prelude items (`PyDict`, `PyReadonlyArray2`,
// `detach_py_result`, …) by bare name. As with the other fragments, the engine
// functions are referenced through their fully-qualified `gam::terms::sae::…`
// paths and the Python surface is a thin wrapper (SPEC rule 8): every number is
// computed in Rust, the FFI only marshals arrays and dicts.

/// Reconstruct a linear-lane [`SparseDictFit`] from the columnar arrays the
/// `sparse_dictionary_fit` FFI hands back to Python, so the diagnostics that
/// take a fitted dictionary (`sparse_dict_dual_certificate`) can be driven from
/// the facade without a live Rust handle. The metadata the diagnostics never
/// read (EV/epochs/route-stats) is filled with neutral defaults.
fn sparse_dict_fit_from_arrays(
    decoder: ndarray::Array2<f32>,
    indices: ndarray::Array2<u32>,
    codes: ndarray::Array2<f32>,
) -> gam::terms::sae::sparse_dict::SparseDictFit {
    let active = codes.ncols();
    gam::terms::sae::sparse_dict::SparseDictFit {
        decoder,
        indices,
        codes,
        explained_variance: 0.0,
        epochs: 0,
        converged: false,
        active,
        score_route_stats: gam::terms::sae::sparse_dict::ScoreRouteStats::default(),
    }
}

/// Reconstruct a block-lane [`BlockSparseFit`] from the columnar arrays the
/// `block_sparse_dictionary_fit` FFI hands back, for the per-block firing
/// readout. The per-block utilisation / stable-rank / γ metadata the readout
/// never reads is left neutral.
fn block_sparse_fit_from_arrays(
    decoder: ndarray::Array2<f32>,
    blocks: ndarray::Array2<u32>,
    gates: ndarray::Array2<f32>,
    codes: ndarray::Array3<f32>,
    block_topk: usize,
    block_size: usize,
) -> gam::terms::sae::sparse_dict::BlockSparseFit {
    gam::terms::sae::sparse_dict::BlockSparseFit {
        decoder,
        blocks,
        gates,
        codes,
        gamma: 1.0,
        block_utilization: Vec::new(),
        block_stable_rank: Vec::new(),
        explained_variance: 0.0,
        epochs: 0,
        converged: false,
        block_topk,
        block_size,
    }
}

fn reconstruct_dense_sae_rows(
    decoder: ndarray::ArrayView2<'_, f32>,
    codes: ndarray::ArrayView2<'_, f32>,
) -> Result<ndarray::Array2<f32>, String> {
    if codes.ncols() != decoder.nrows() {
        return Err(format!(
            "audit_sae: codes have K={} columns but decoder has K={} rows",
            codes.ncols(),
            decoder.nrows()
        ));
    }
    let mut fitted = ndarray::Array2::<f32>::zeros((codes.nrows(), decoder.ncols()));
    for i in 0..codes.nrows() {
        for k in 0..codes.ncols() {
            let code = codes[[i, k]];
            if code == 0.0 {
                continue;
            }
            for p in 0..decoder.ncols() {
                fitted[[i, p]] += code * decoder[[k, p]];
            }
        }
    }
    Ok(fitted)
}

fn residuals_from_dense_sae(
    data: ndarray::ArrayView2<'_, f32>,
    decoder: ndarray::ArrayView2<'_, f32>,
    codes: ndarray::ArrayView2<'_, f32>,
) -> Result<ndarray::Array2<f32>, String> {
    if data.ncols() != decoder.ncols() {
        return Err(format!(
            "audit_sae: data have P={} columns but decoder has P={} columns",
            data.ncols(),
            decoder.ncols()
        ));
    }
    let fitted = reconstruct_dense_sae_rows(decoder, codes)?;
    if fitted.nrows() != data.nrows() {
        return Err(format!(
            "audit_sae: data have N={} rows but codes have N={} rows",
            data.nrows(),
            fitted.nrows()
        ));
    }
    let mut residuals = data.to_owned();
    residuals -= &fitted;
    Ok(residuals)
}

fn dense_linear_topk(
    codes: ndarray::ArrayView2<'_, f32>,
    active: Option<usize>,
) -> (ndarray::Array2<u32>, ndarray::Array2<f32>) {
    let n = codes.nrows();
    let k = codes.ncols();
    let active_width = active.unwrap_or_else(|| {
        let mut max_live = 0usize;
        for row in codes.rows() {
            let live = row.iter().filter(|value| **value != 0.0).count();
            max_live = max_live.max(live);
        }
        max_live.max(1)
    });
    let width = active_width.min(k.max(1));
    let mut indices = ndarray::Array2::<u32>::zeros((n, width));
    let mut sparse_codes = ndarray::Array2::<f32>::zeros((n, width));
    for row_idx in 0..n {
        let mut live = Vec::new();
        for atom_idx in 0..k {
            let code = codes[[row_idx, atom_idx]];
            if code != 0.0 {
                live.push((atom_idx, code));
            }
        }
        live.sort_by(|left, right| right.1.abs().total_cmp(&left.1.abs()));
        for (slot, (atom_idx, code)) in live.into_iter().take(width).enumerate() {
            indices[[row_idx, slot]] = atom_idx as u32;
            sparse_codes[[row_idx, slot]] = code;
        }
    }
    (indices, sparse_codes)
}

fn dense_block_topk(
    codes: ndarray::ArrayView2<'_, f32>,
    block_size: usize,
    block_topk: Option<usize>,
) -> Result<(ndarray::Array2<u32>, ndarray::Array2<f32>, ndarray::Array3<f32>, usize), String> {
    if block_size == 0 {
        return Err("audit_sae: block_size must be >= 1".to_string());
    }
    if codes.ncols() % block_size != 0 {
        return Err(format!(
            "audit_sae: codes have K={} columns, not a multiple of block_size {block_size}",
            codes.ncols()
        ));
    }
    let n = codes.nrows();
    let n_blocks = codes.ncols() / block_size;
    let topk = block_topk.unwrap_or_else(|| {
        let mut max_live = 0usize;
        for row_idx in 0..n {
            let mut live = 0usize;
            for block in 0..n_blocks {
                let mut norm2 = 0.0f32;
                for offset in 0..block_size {
                    let value = codes[[row_idx, block * block_size + offset]];
                    norm2 += value * value;
                }
                if norm2 > 0.0 {
                    live += 1;
                }
            }
            max_live = max_live.max(live);
        }
        max_live.max(1)
    });
    let width = topk.min(n_blocks.max(1));
    let mut blocks = ndarray::Array2::<u32>::zeros((n, width));
    let mut gates = ndarray::Array2::<f32>::zeros((n, width));
    let mut block_codes = ndarray::Array3::<f32>::zeros((n, width, block_size));
    for row_idx in 0..n {
        let mut live = Vec::new();
        for block in 0..n_blocks {
            let mut norm2 = 0.0f32;
            for offset in 0..block_size {
                let value = codes[[row_idx, block * block_size + offset]];
                norm2 += value * value;
            }
            if norm2 > 0.0 {
                live.push((block, norm2.sqrt()));
            }
        }
        live.sort_by(|left, right| right.1.total_cmp(&left.1));
        for (slot, (block, gate)) in live.into_iter().take(width).enumerate() {
            blocks[[row_idx, slot]] = block as u32;
            gates[[row_idx, slot]] = gate;
            for offset in 0..block_size {
                block_codes[[row_idx, slot, offset]] = codes[[row_idx, block * block_size + offset]];
            }
        }
    }
    Ok((blocks, gates, block_codes, width))
}

fn dual_certificate_report_dict<'py>(
    py: Python<'py>,
    report: &gam::terms::sae::dual_certificate::DualCertificateReport,
) -> PyResult<Bound<'py, PyDict>> {
    let out = PyDict::new(py);
    out.set_item("n_rows", report.n_rows)?;
    out.set_item("frac_certified", report.frac_certified)?;
    out.set_item(
        "optimality_ratio_quantiles",
        report.optimality_ratio_quantiles.clone(),
    )?;
    out.set_item("birth_candidates", report.birth_candidates.clone())?;
    Ok(out)
}

fn block_coordinate_report_dict<'py>(
    py: Python<'py>,
    report: &gam::terms::sae::sparse_dict::BlockCoordinateReport,
) -> PyResult<Bound<'py, PyDict>> {
    let n = report.firings.len();
    let mut firing_block = Vec::with_capacity(n);
    let mut firing_row = Vec::with_capacity(n);
    let mut firing_t = Vec::with_capacity(n);
    let mut firing_amplitude = Vec::with_capacity(n);
    let mut firing_t_se = Vec::with_capacity(n);
    let mut firing_amplitude_se = Vec::with_capacity(n);
    let mut firing_t_se_clamped = Vec::with_capacity(n);
    for f in &report.firings {
        firing_block.push(f.block as u64);
        firing_row.push(f.row as u64);
        firing_t.push(f.t);
        firing_amplitude.push(f.amplitude);
        firing_t_se.push(f.t_se);
        firing_amplitude_se.push(f.amplitude_se);
        firing_t_se_clamped.push(f.t_se_clamped);
    }

    let out = PyDict::new(py);
    out.set_item("sigma_hat", report.sigma_hat)?;
    out.set_item("mean_radius", report.mean_radius)?;
    out.set_item("n_firings", report.n_firings)?;
    out.set_item("block", ndarray::Array1::from_vec(firing_block).into_pyarray(py))?;
    out.set_item("row", ndarray::Array1::from_vec(firing_row).into_pyarray(py))?;
    out.set_item("t", ndarray::Array1::from_vec(firing_t).into_pyarray(py))?;
    out.set_item(
        "amplitude",
        ndarray::Array1::from_vec(firing_amplitude).into_pyarray(py),
    )?;
    out.set_item("t_se", ndarray::Array1::from_vec(firing_t_se).into_pyarray(py))?;
    out.set_item(
        "amplitude_se",
        ndarray::Array1::from_vec(firing_amplitude_se).into_pyarray(py),
    )?;
    out.set_item("t_se_clamped", firing_t_se_clamped)?;
    Ok(out)
}

fn absorption_audit_dict<'py>(
    py: Python<'py>,
    codes: ndarray::ArrayView2<'_, f32>,
    block_size: usize,
    activation_threshold: f32,
    max_pairs: usize,
) -> PyResult<Bound<'py, PyDict>> {
    let n = codes.nrows();
    let k_units = if block_size == 1 {
        codes.ncols()
    } else {
        codes.ncols() / block_size
    };
    let mut atom_codes = gam::terms::sae::atom_codes::SparseAtomCodes::empty(n, k_units);
    for row_idx in 0..n {
        for unit in 0..k_units {
            let weight = if block_size == 1 {
                codes[[row_idx, unit]].abs() as f64
            } else {
                let mut norm2 = 0.0f64;
                for offset in 0..block_size {
                    let value = codes[[row_idx, unit * block_size + offset]] as f64;
                    norm2 += value * value;
                }
                norm2.sqrt()
            };
            if weight > activation_threshold as f64 {
                atom_codes.row_mut(row_idx).assign(unit, weight);
            }
        }
    }

    let mut pairs = Vec::new();
    for a in 0..k_units {
        for b in (a + 1)..k_units {
            let stats = atom_codes.coactivation(a, b);
            pairs.push((
                stats.absorption_asymmetry(),
                a,
                b,
                stats.n_obs,
                stats.n_a,
                stats.n_b,
                stats.n_joint,
                stats.p_a_given_b,
                stats.p_b_given_a,
                stats.lift,
                stats.weight_correlation,
                stats.dependence(),
                stats.fusion_evidence(),
            ));
        }
    }
    pairs.sort_by(|left, right| right.0.total_cmp(&left.0));

    let pair_list = PyList::empty(py);
    for (
        absorption_asymmetry,
        a,
        b,
        n_obs,
        n_a,
        n_b,
        n_joint,
        p_a_given_b,
        p_b_given_a,
        lift,
        weight_correlation,
        dependence,
        fusion_evidence,
    ) in pairs.into_iter().take(max_pairs)
    {
        let pair = PyDict::new(py);
        pair.set_item("a", a)?;
        pair.set_item("b", b)?;
        pair.set_item("n_obs", n_obs)?;
        pair.set_item("n_a", n_a)?;
        pair.set_item("n_b", n_b)?;
        pair.set_item("n_joint", n_joint)?;
        pair.set_item("p_a_given_b", p_a_given_b)?;
        pair.set_item("p_b_given_a", p_b_given_a)?;
        pair.set_item("lift", lift)?;
        pair.set_item("weight_correlation", weight_correlation)?;
        pair.set_item("dependence", dependence)?;
        pair.set_item("fusion_evidence", fusion_evidence)?;
        pair.set_item("absorption_asymmetry", absorption_asymmetry)?;
        pair_list.append(pair)?;
    }

    let out = PyDict::new(py);
    out.set_item("n_units", k_units)?;
    out.set_item("activation_threshold", activation_threshold)?;
    out.set_item("pairs", pair_list)?;
    Ok(out)
}

fn transport_report_dict<'py>(
    py: Python<'py>,
    report: &gam::terms::sae::inference::transport_class::CircleTransportReport,
) -> PyResult<Bound<'py, PyDict>> {
    let class_name = match report.class {
        gam::terms::sae::inference::transport_class::CircleTransportClass::Shift => "shift",
        gam::terms::sae::inference::transport_class::CircleTransportClass::Reflect => "reflect",
        gam::terms::sae::inference::transport_class::CircleTransportClass::Mixing => "mixing",
    };
    let out = PyDict::new(py);
    out.set_item("layer_from", report.layer_from)?;
    out.set_item("layer_to", report.layer_to)?;
    out.set_item("n_samples", report.n_samples)?;
    out.set_item("winding", report.winding)?;
    out.set_item("phase", report.phase)?;
    out.set_item("phase_degrees", report.phase_degrees())?;
    out.set_item("defect", report.defect)?;
    out.set_item("resultant_shift", report.resultant_shift)?;
    out.set_item("resultant_reflect", report.resultant_reflect)?;
    out.set_item("class", class_name)?;
    Ok(out)
}

/// Dimension spectrometer: fit a single-atom (`s = 1`) sparse dictionary at each
/// rung of the doubling ladder `k_min·2^j`, `j = 0..=n_doublings`, and invert the
/// fitted reconstruction-loss scaling law `L(K) − σ² ∝ K^{-2/d}` into an
/// intrinsic-dimension estimate `d̂ = −2/m` with delta-method standard errors.
/// The forwarded dictionary template mirrors `sparse_dictionary_fit`'s fit
/// knobs; `active` is forced to 1 per rung by the engine regardless.
#[pyfunction(signature = (
    data,
    k_min = 4,
    n_doublings = 6,
    active = 1,
    minibatch = 512,
    max_epochs = 30,
    score_tile = 4096,
    code_ridge = 1.0e-6,
    decoder_ridge = 1.0e-9,
    tolerance = 1.0e-6,
    score_mode = "required"
))]
fn dimension_spectrometer<'py>(
    py: Python<'py>,
    data: PyReadonlyArray2<'py, f32>,
    k_min: usize,
    n_doublings: usize,
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
    let data_values = data.as_array().to_owned();
    let config = gam::terms::sae::spectrometer::SpectrometerConfig {
        k_min,
        n_doublings,
        dict: SparseDictConfig {
            n_atoms: k_min,
            active,
            minibatch,
            max_epochs,
            score_tile,
            code_ridge,
            decoder_ridge,
            tolerance,
            score_mode,
        },
    };
    let report = detach_py_result(py, "dimension_spectrometer", move || {
        gam::terms::sae::spectrometer::dimension_spectrometer(data_values.view(), &config)
    })?;
    let out = PyDict::new(py);
    out.set_item("rungs", report.rungs)?;
    out.set_item("noise_floor", report.noise_floor)?;
    out.set_item("slope", report.slope)?;
    out.set_item("slope_se", report.slope_se)?;
    out.set_item("d_hat", report.d_hat)?;
    out.set_item("d_hat_se", report.d_hat_se)?;
    out.set_item("floor_saturated", report.floor_saturated)?;
    Ok(out.unbind())
}

/// Per-firing circle-coordinate readout for one `b = 2` block of a fitted
/// block-sparse dictionary: phase `t̂ ∈ [0,1)`, amplitude `‖z‖`, and their
/// closed-form SEs (`σ̂` from the block's radial scatter). Takes the block-lane
/// fit as its columnar arrays; the firings are returned as aligned columns.
#[pyfunction(signature = (decoder, blocks, gates, codes, block, block_topk = 1))]
fn block_firing_coordinates<'py>(
    py: Python<'py>,
    decoder: PyReadonlyArray2<'py, f32>,
    blocks: PyReadonlyArray2<'py, u32>,
    gates: PyReadonlyArray2<'py, f32>,
    codes: PyReadonlyArray3<'py, f32>,
    block: usize,
    block_topk: usize,
) -> PyResult<Py<PyDict>> {
    let decoder_values = decoder.as_array().to_owned();
    let block_values = blocks.as_array().to_owned();
    let gate_values = gates.as_array().to_owned();
    let code_values = codes.as_array().to_owned();
    let block_size = code_values.shape()[2];
    let report = detach_py_result(py, "block_firing_coordinates", move || {
        let fit = block_sparse_fit_from_arrays(
            decoder_values,
            block_values,
            gate_values,
            code_values,
            block_topk,
            block_size,
        );
        gam::terms::sae::sparse_dict::block_firing_coordinates(&fit, block)
    })?;

    let n = report.firings.len();
    let mut firing_block = Vec::with_capacity(n);
    let mut firing_row = Vec::with_capacity(n);
    let mut firing_t = Vec::with_capacity(n);
    let mut firing_amplitude = Vec::with_capacity(n);
    let mut firing_t_se = Vec::with_capacity(n);
    let mut firing_amplitude_se = Vec::with_capacity(n);
    let mut firing_t_se_clamped = Vec::with_capacity(n);
    for f in &report.firings {
        firing_block.push(f.block as u64);
        firing_row.push(f.row as u64);
        firing_t.push(f.t);
        firing_amplitude.push(f.amplitude);
        firing_t_se.push(f.t_se);
        firing_amplitude_se.push(f.amplitude_se);
        firing_t_se_clamped.push(f.t_se_clamped);
    }

    let out = PyDict::new(py);
    out.set_item("sigma_hat", report.sigma_hat)?;
    out.set_item("mean_radius", report.mean_radius)?;
    out.set_item("n_firings", report.n_firings)?;
    out.set_item("block", ndarray::Array1::from_vec(firing_block).into_pyarray(py))?;
    out.set_item("row", ndarray::Array1::from_vec(firing_row).into_pyarray(py))?;
    out.set_item("t", ndarray::Array1::from_vec(firing_t).into_pyarray(py))?;
    out.set_item(
        "amplitude",
        ndarray::Array1::from_vec(firing_amplitude).into_pyarray(py),
    )?;
    out.set_item("t_se", ndarray::Array1::from_vec(firing_t_se).into_pyarray(py))?;
    out.set_item(
        "amplitude_se",
        ndarray::Array1::from_vec(firing_amplitude_se).into_pyarray(py),
    )?;
    out.set_item("t_se_clamped", firing_t_se_clamped)?;
    Ok(out.unbind())
}

/// Build the dict form of a [`RoutabilityFloor`].
fn routability_floor_dict<'py>(
    py: Python<'py>,
    floor: &gam::terms::sae::routability::RoutabilityFloor,
) -> PyResult<Bound<'py, PyDict>> {
    let out = PyDict::new(py);
    out.set_item("p", floor.p)?;
    out.set_item("n_blocks", floor.n_blocks)?;
    out.set_item("b_max", floor.b_max)?;
    out.set_item("delta", floor.delta)?;
    out.set_item("floor", floor.floor)?;
    out.set_item(
        "minimum_routable_energy",
        gam::terms::sae::routability::minimum_routable_energy(floor),
    )?;
    Ok(out)
}

/// Closed-form routability floor `√(b_max/p) + √(2·ln(K/δ)/p)` on the routable
/// energy fraction, plus the derived `minimum_routable_energy`. Degenerate
/// configurations (`p`, `n_blocks`, `b_max` zero; `b_max > p`; non-finite or
/// non-positive `delta`) are rejected as `ValueError` before the (asserting)
/// engine call.
#[pyfunction(signature = (p, n_blocks, b_max, delta))]
fn routability_floor(
    py: Python<'_>,
    p: usize,
    n_blocks: usize,
    b_max: usize,
    delta: f64,
) -> PyResult<Py<PyDict>> {
    if p == 0 || n_blocks == 0 || b_max == 0 {
        return Err(PyValueError::new_err(
            "routability_floor requires p >= 1, n_blocks >= 1, b_max >= 1",
        ));
    }
    if b_max > p {
        return Err(PyValueError::new_err(
            "routability_floor requires b_max <= p (a b-frame must fit in R^p)",
        ));
    }
    if !delta.is_finite() || delta <= 0.0 {
        return Err(PyValueError::new_err(
            "routability_floor requires a finite delta > 0",
        ));
    }
    let floor = gam::terms::sae::routability::routability_floor(p, n_blocks, b_max, delta);
    Ok(routability_floor_dict(py, &floor)?.unbind())
}

/// Empirical routability audit: measure a fitted dictionary's max-cross-gate
/// distribution against real residual rows and compare it to the closed-form
/// floor. `decoder` is `K×P`; `block_size` is 1 for the linear atom lane, `b`
/// for the block lane (must divide `K`). Returns the floor (nested), the
/// requested quantiles, and the coherence-excess summary.
#[pyfunction(signature = (decoder, residuals, block_size, delta, quantile_levels))]
fn routability_audit<'py>(
    py: Python<'py>,
    decoder: PyReadonlyArray2<'py, f32>,
    residuals: PyReadonlyArray2<'py, f32>,
    block_size: usize,
    delta: f64,
    quantile_levels: Vec<f64>,
) -> PyResult<Py<PyDict>> {
    let decoder_values = decoder.as_array().to_owned();
    let residual_values = residuals.as_array().to_owned();
    let report = detach_py_result(py, "routability_audit", move || {
        gam::terms::sae::routability::routability_audit(
            decoder_values.view(),
            residual_values.view(),
            block_size,
            delta,
            &quantile_levels,
        )
    })?;
    let out = PyDict::new(py);
    out.set_item("n_rows", report.n_rows)?;
    out.set_item("floor", routability_floor_dict(py, &report.floor)?)?;
    out.set_item("quantiles", report.quantiles)?;
    out.set_item("empirical_mean", report.empirical_mean)?;
    out.set_item("empirical_max", report.empirical_max)?;
    out.set_item("confidence_quantile", report.confidence_quantile)?;
    out.set_item("coherence_excess", report.coherence_excess)?;
    out.set_item("fraction_below_floor", report.fraction_below_floor)?;
    Ok(out.unbind())
}

/// Global-optimality dual certificate for the collapsed linear lane: for each
/// row of `data`, fold the fitted-routing residual's dual value over the whole
/// dictionary and form the scale-free optimality ratio, reporting the certified
/// fraction, the ratio quantiles, and the top strictly-improving birth
/// candidates. The fit is passed as its columnar arrays.
#[pyfunction(signature = (data, decoder, indices, codes, max_candidates = 16))]
fn sparse_dict_dual_certificate<'py>(
    py: Python<'py>,
    data: PyReadonlyArray2<'py, f32>,
    decoder: PyReadonlyArray2<'py, f32>,
    indices: PyReadonlyArray2<'py, u32>,
    codes: PyReadonlyArray2<'py, f32>,
    max_candidates: usize,
) -> PyResult<Py<PyDict>> {
    let data_values = data.as_array().to_owned();
    let decoder_values = decoder.as_array().to_owned();
    let index_values = indices.as_array().to_owned();
    let code_values = codes.as_array().to_owned();
    let report = detach_py_result(py, "sparse_dict_dual_certificate", move || {
        let fit = sparse_dict_fit_from_arrays(decoder_values, index_values, code_values);
        gam::terms::sae::dual_certificate::sparse_dict_dual_certificate(
            data_values.view(),
            &fit,
            max_candidates,
        )
    })?;
    let out = PyDict::new(py);
    out.set_item("n_rows", report.n_rows)?;
    out.set_item("frac_certified", report.frac_certified)?;
    out.set_item(
        "optimality_ratio_quantiles",
        report.optimality_ratio_quantiles,
    )?;
    out.set_item("birth_candidates", report.birth_candidates)?;
    Ok(out.unbind())
}

/// One-shot audit over an externally supplied, frozen SAE dictionary: decoder
/// rows `K×P`, dense activation/codes matrix `N×K`, and source activations
/// `N×P`. No fitting occurs here; the function reconstructs the frozen rows,
/// derives the sparse/block routing views required by the existing Rust
/// diagnostics, and returns a dict of dual-cert, routability, coordinate-SE,
/// absorption, and optional transport diagnostics.
#[pyfunction(signature = (
    decoder,
    codes,
    data,
    active = None,
    block_size = 1,
    block_topk = None,
    delta = 0.05,
    quantile_levels = None,
    max_candidates = 16,
    coordinate_blocks = None,
    activation_threshold = 0.0,
    max_absorption_pairs = 32,
    transport_theta_in = None,
    transport_theta_out = None,
    transport_layer_from = 0,
    transport_layer_to = 1
))]
fn audit_sae<'py>(
    py: Python<'py>,
    decoder: PyReadonlyArray2<'py, f32>,
    codes: PyReadonlyArray2<'py, f32>,
    data: PyReadonlyArray2<'py, f32>,
    active: Option<usize>,
    block_size: usize,
    block_topk: Option<usize>,
    delta: f64,
    quantile_levels: Option<Vec<f64>>,
    max_candidates: usize,
    coordinate_blocks: Option<Vec<usize>>,
    activation_threshold: f32,
    max_absorption_pairs: usize,
    transport_theta_in: Option<PyReadonlyArray1<'py, f64>>,
    transport_theta_out: Option<PyReadonlyArray1<'py, f64>>,
    transport_layer_from: usize,
    transport_layer_to: usize,
) -> PyResult<Py<PyDict>> {
    let decoder_values = decoder.as_array().to_owned();
    let code_values = codes.as_array().to_owned();
    let data_values = data.as_array().to_owned();
    if decoder_values.nrows() == 0 || decoder_values.ncols() == 0 {
        return Err(PyValueError::new_err(
            "audit_sae requires a non-empty decoder matrix",
        ));
    }
    if code_values.ncols() != decoder_values.nrows() {
        return Err(PyValueError::new_err(format!(
            "audit_sae codes/decoder mismatch: codes K={} but decoder K={}",
            code_values.ncols(),
            decoder_values.nrows()
        )));
    }
    if data_values.nrows() != code_values.nrows() || data_values.ncols() != decoder_values.ncols() {
        return Err(PyValueError::new_err(format!(
            "audit_sae data shape {:?} incompatible with codes {:?} and decoder {:?}",
            data_values.dim(),
            code_values.dim(),
            decoder_values.dim()
        )));
    }
    if block_size == 0 {
        return Err(PyValueError::new_err("audit_sae block_size must be >= 1"));
    }
    if active == Some(0) {
        return Err(PyValueError::new_err("audit_sae active must be >= 1"));
    }
    if block_topk == Some(0) {
        return Err(PyValueError::new_err("audit_sae block_topk must be >= 1"));
    }
    if decoder_values.nrows() % block_size != 0 {
        return Err(PyValueError::new_err(format!(
            "audit_sae decoder has K={} rows, not a multiple of block_size {block_size}",
            decoder_values.nrows()
        )));
    }
    if !delta.is_finite() || delta <= 0.0 {
        return Err(PyValueError::new_err(
            "audit_sae requires a finite delta > 0",
        ));
    }
    if activation_threshold < 0.0 || !activation_threshold.is_finite() {
        return Err(PyValueError::new_err(
            "audit_sae activation_threshold must be finite and non-negative",
        ));
    }

    let quantiles = quantile_levels.unwrap_or_else(|| vec![0.5, 0.9, 0.99]);
    let theta_in_values = transport_theta_in.map(|values| values.as_array().to_owned().to_vec());
    let theta_out_values = transport_theta_out.map(|values| values.as_array().to_owned().to_vec());

    let audit = detach_py_result(py, "audit_sae", move || {
        let residuals =
            residuals_from_dense_sae(data_values.view(), decoder_values.view(), code_values.view())?;
        let routability = gam::terms::sae::routability::routability_audit(
            decoder_values.view(),
            residuals.view(),
            block_size,
            delta,
            &quantiles,
        )?;

        let (dual, coordinate_reports) = if block_size == 1 {
            let (indices, sparse_codes) = dense_linear_topk(code_values.view(), active);
            let fit = sparse_dict_fit_from_arrays(decoder_values.clone(), indices, sparse_codes);
            let report = gam::terms::sae::dual_certificate::sparse_dict_dual_certificate(
                data_values.view(),
                &fit,
                max_candidates,
            )?;
            (report, Vec::new())
        } else {
            let (blocks, gates, block_codes, width) =
                dense_block_topk(code_values.view(), block_size, block_topk)?;
            let fit = block_sparse_fit_from_arrays(
                decoder_values.clone(),
                blocks,
                gates,
                block_codes,
                width,
                block_size,
            );
            let report = gam::terms::sae::dual_certificate::block_dual_certificate(
                data_values.view(),
                &fit,
                max_candidates,
            )?;
            let mut coordinates = Vec::new();
            if block_size >= 2 && block_size % 2 == 0 {
                let total_blocks = decoder_values.nrows() / block_size;
                let blocks = coordinate_blocks.unwrap_or_else(|| (0..total_blocks).collect());
                for block in blocks {
                    if block >= total_blocks {
                        return Err(format!(
                            "audit_sae coordinate block {block} out of range 0..{total_blocks}"
                        ));
                    }
                    coordinates.push(
                        gam::terms::sae::sparse_dict::harmonic_firing_coordinates(&fit, block)?,
                    );
                }
            }
            (report, coordinates)
        };

        let transport = match (theta_in_values, theta_out_values) {
            (Some(theta_in), Some(theta_out)) => Some(
                gam::terms::sae::inference::transport_class::classify_circle_transport(
                    &theta_in,
                    &theta_out,
                    transport_layer_from,
                    transport_layer_to,
                )?,
            ),
            (None, None) => None,
            _ => {
                return Err(
                    "audit_sae transport requires both transport_theta_in and transport_theta_out"
                        .to_string(),
                );
            }
        };

        Ok::<_, String>((
            routability,
            dual,
            coordinate_reports,
            transport,
            decoder_values,
            code_values,
        ))
    })?;

    let (routability, dual, coordinate_reports, transport, decoder_values, code_values) = audit;
    let out = PyDict::new(py);
    out.set_item("decoder_shape", (decoder_values.nrows(), decoder_values.ncols()))?;
    out.set_item("codes_shape", (code_values.nrows(), code_values.ncols()))?;

    let route = PyDict::new(py);
    route.set_item("block_size", block_size)?;
    route.set_item("n_blocks", decoder_values.nrows() / block_size)?;
    match active {
        Some(value) => route.set_item("active", value)?,
        None => route.set_item("active", py.None())?,
    }
    match block_topk {
        Some(value) => route.set_item("block_topk", value)?,
        None => route.set_item("block_topk", py.None())?,
    }
    out.set_item("routing", route)?;

    let routability_dict = PyDict::new(py);
    routability_dict.set_item("n_rows", routability.n_rows)?;
    routability_dict.set_item("floor", routability_floor_dict(py, &routability.floor)?)?;
    routability_dict.set_item("quantiles", routability.quantiles)?;
    routability_dict.set_item("empirical_mean", routability.empirical_mean)?;
    routability_dict.set_item("empirical_max", routability.empirical_max)?;
    routability_dict.set_item("confidence_quantile", routability.confidence_quantile)?;
    routability_dict.set_item("coherence_excess", routability.coherence_excess)?;
    routability_dict.set_item("fraction_below_floor", routability.fraction_below_floor)?;
    out.set_item("routability", routability_dict)?;
    out.set_item("dual_certificate", dual_certificate_report_dict(py, &dual)?)?;
    out.set_item(
        "absorption",
        absorption_audit_dict(
            py,
            code_values.view(),
            block_size,
            activation_threshold,
            max_absorption_pairs,
        )?,
    )?;

    let coordinate_list = PyList::empty(py);
    for report in &coordinate_reports {
        coordinate_list.append(block_coordinate_report_dict(py, report)?)?;
    }
    out.set_item("coordinate_se", coordinate_list)?;

    match transport {
        Some(report) => out.set_item("transport", transport_report_dict(py, &report)?)?,
        None => out.set_item("transport", py.None())?,
    }
    Ok(out.unbind())
}

/// Compose a chain of component contracts into one end-to-end shadowing bound.
/// Each contract is `(name, domain_radius, defect, lipschitz)`; returns the
/// total defect, its per-stage additive contributions, and the drift-only
/// domain-feasibility flag.
#[pyfunction(signature = (chain,))]
fn compose_contracts(
    py: Python<'_>,
    chain: Vec<(String, f64, f64, f64)>,
) -> PyResult<Py<PyDict>> {
    let contracts: Vec<gam::terms::sae::inference::contracts::Contract> = chain
        .into_iter()
        .map(
            |(name, domain_radius, defect, lipschitz)| gam::terms::sae::inference::contracts::Contract {
                name,
                domain_radius,
                defect,
                lipschitz,
            },
        )
        .collect();
    let report = gam::terms::sae::inference::contracts::compose_contracts(&contracts);
    let out = PyDict::new(py);
    out.set_item("total_defect", report.total_defect)?;
    out.set_item("per_stage_contribution", report.per_stage_contribution)?;
    out.set_item("domain_ok", report.domain_ok)?;
    Ok(out.unbind())
}

/// Candès–Fernández-Granda super-resolution separation threshold `Δ ≈ 2/H` for a
/// harmonic atom carrying `n_harmonics = H` harmonics: the minimum wrap-around
/// spike separation at which exact convex recovery is guaranteed.
#[pyfunction(signature = (n_harmonics,))]
fn separation_limit(n_harmonics: usize) -> f64 {
    gam::terms::sae::super_resolution::separation_limit(n_harmonics)
}

/// Recover the point masses `{(a_j, t_j)}` underlying a harmonic atom's Fourier
/// coefficients by the matrix-pencil / Prony method. `fourier_coeffs[h] =
/// (c_{h+1}, s_{h+1})` covers harmonics `1..H`; `sigma` is the per-component
/// coefficient-noise SD (`≤ 0` selects the noiseless numerical-rank path).
/// Returns the recovered spikes (position `t` + amplitude columns), the selected
/// model order, the fit residual, and the Hankel singular spectrum.
#[pyfunction(signature = (fourier_coeffs, sigma = 0.0))]
fn recover_spikes(
    py: Python<'_>,
    fourier_coeffs: Vec<(f64, f64)>,
    sigma: f64,
) -> PyResult<Py<PyDict>> {
    let recovery = detach_py_result(py, "recover_spikes", move || {
        gam::terms::sae::super_resolution::recover_spikes(&fourier_coeffs, sigma)
    })?;
    let n = recovery.spikes.len();
    let mut spike_t = Vec::with_capacity(n);
    let mut spike_amplitude = Vec::with_capacity(n);
    for spike in &recovery.spikes {
        spike_t.push(spike.t);
        spike_amplitude.push(spike.amplitude);
    }
    let out = PyDict::new(py);
    out.set_item("t", ndarray::Array1::from_vec(spike_t).into_pyarray(py))?;
    out.set_item(
        "amplitude",
        ndarray::Array1::from_vec(spike_amplitude).into_pyarray(py),
    )?;
    out.set_item("model_order", recovery.model_order)?;
    out.set_item("residual", recovery.residual)?;
    out.set_item(
        "hankel_singular_values",
        ndarray::Array1::from_vec(recovery.hankel_singular_values).into_pyarray(py),
    )?;
    Ok(out.unbind())
}

/// Compose a closed loop of circle isometries and report the net `O(2)` element.
/// `edges` are `(sign, angle)` (`sign = +1` rotation, `−1` reflection); `defects`
/// are the per-edge `O(2)` departures whose sum is the derived trivial-verdict
/// tolerance. Returns the net sign/angle and the measure-don't-latch triviality
/// verdict.
#[pyfunction(signature = (edges, defects))]
fn loop_holonomy(
    py: Python<'_>,
    edges: Vec<(i8, f64)>,
    defects: Vec<f64>,
) -> PyResult<Py<PyDict>> {
    let report = gam::terms::sae::inference::contracts::loop_holonomy(&edges, &defects);
    let out = PyDict::new(py);
    out.set_item("loop_len", report.loop_len)?;
    out.set_item("net_sign", report.net_sign)?;
    out.set_item("net_angle", report.net_angle)?;
    out.set_item("is_trivial", report.is_trivial)?;
    out.set_item("angle_tolerance", report.angle_tolerance)?;
    Ok(out.unbind())
}
