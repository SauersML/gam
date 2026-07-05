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
        gamma: 0.0,
        block_utilization: Vec::new(),
        block_stable_rank: Vec::new(),
        explained_variance: 0.0,
        epochs: 0,
        converged: false,
        block_topk,
        block_size,
    }
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
