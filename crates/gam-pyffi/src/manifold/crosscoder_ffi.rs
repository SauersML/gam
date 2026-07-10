/// Thin Python marshalling for the typed manifold-crosscoder entry (#2231 Inc D).
/// All target stacking, seeding, fitting, block relevance, drift, and transport
/// math lives in `gam-sae`; this boundary only owns Python arrays and dicts.

fn crosscoder_layer_tag(
    layer: gam::terms::sae::manifold::CrosscoderLayer,
    anchor_label: &str,
    block_labels: &[String],
) -> PyResult<String> {
    match layer {
        gam::terms::sae::manifold::CrosscoderLayer::Anchor => Ok(anchor_label.to_string()),
        gam::terms::sae::manifold::CrosscoderLayer::Block(index) => {
            block_labels.get(index).cloned().ok_or_else(|| {
                py_value_error(format!("crosscoder layer block {index} is out of range"))
            })
        }
    }
}

fn crosscoder_transport_dict<'py>(
    py: Python<'py>,
    report: &gam::terms::sae::manifold::AtomTransportReport,
    anchor_label: &str,
    block_labels: &[String],
    law_gap_tolerance: f64,
) -> PyResult<Bound<'py, PyDict>> {
    let out = PyDict::new(py);
    out.set_item("atom", report.atom)?;
    out.set_item(
        "source",
        crosscoder_layer_tag(report.source, anchor_label, block_labels)?,
    )?;
    out.set_item(
        "target",
        crosscoder_layer_tag(report.target, anchor_label, block_labels)?,
    )?;
    out.set_item("grid_resolution", report.grid_resolution)?;
    out.set_item("n_harmonics", report.n_harmonics)?;
    out.set_item("phase_shift", report.phase_shift)?;
    out.set_item("phase_r2", report.phase_r2)?;
    out.set_item("smooth_r2", report.smooth_r2)?;
    out.set_item("law_gap", report.law_gap())?;
    out.set_item("law_holds", report.law_holds(law_gap_tolerance))?;
    out.set_item("deviation_locus", report.deviation_locus())?;
    out.set_item("drift", report.drift)?;
    out.set_item("principal_angles", report.principal_angles.clone())?;
    out.set_item("transport_grid", report.transport_grid.clone())?;
    Ok(out)
}

fn crosscoder_drift_dict<'py>(
    py: Python<'py>,
    status: &gam::terms::sae::manifold::CrosscoderDriftStatus,
    anchor_label: &str,
    block_labels: &[String],
) -> PyResult<Bound<'py, PyDict>> {
    use gam::terms::sae::manifold::CrosscoderDriftStatus;
    let out = PyDict::new(py);
    match status {
        CrosscoderDriftStatus::Measured(report) => {
            out.set_item("status", "measured")?;
            out.set_item("num_atoms", report.num_atoms)?;
            out.set_item("mean_drift", report.mean_drift())?;
            out.set_item("most_drifting_atom", report.most_drifting_atom())?;
            out.set_item("most_stable_atom", report.most_stable_atom())?;
            let chain = report
                .layer_chain
                .iter()
                .map(|&layer| crosscoder_layer_tag(layer, anchor_label, block_labels))
                .collect::<PyResult<Vec<_>>>()?;
            out.set_item("layer_chain", chain)?;
            let steps = PyList::empty(py);
            for step in &report.steps {
                let item = PyDict::new(py);
                item.set_item("atom", step.atom)?;
                item.set_item(
                    "source",
                    crosscoder_layer_tag(step.source, anchor_label, block_labels)?,
                )?;
                item.set_item(
                    "target",
                    crosscoder_layer_tag(step.target, anchor_label, block_labels)?,
                )?;
                item.set_item("drift", step.drift)?;
                item.set_item("principal_angles", step.principal_angles.clone())?;
                item.set_item("max_principal_angle", step.max_principal_angle())?;
                steps.append(item)?;
            }
            out.set_item("steps", steps)?;
        }
        CrosscoderDriftStatus::Undefined { reason } => {
            out.set_item("status", "undefined")?;
            out.set_item("reason", reason)?;
        }
    }
    Ok(out)
}

/// Fit a row-aligned, named multi-layer manifold crosscoder through the unified
/// SAE engine and return honest-unit layer decoders plus drift/transport reports.
///
/// `targets` and `target_labels` are parallel and represent the non-anchor
/// layers.  The public Python facade accepts `targets=[(name, array), ...]` and
/// performs only that tuple split before calling this boundary.
#[pyfunction(signature = (
    anchor,
    anchor_label,
    target_labels,
    targets,
    n_atoms,
    n_harmonics = 3,
    sparsity_strength = 1.0,
    smoothness = 1.0,
    max_iter = 50,
    learning_rate = 0.05,
    ridge_ext_coord = 1.0e-6,
    ridge_beta = 1.0e-6,
    random_state = 0,
    run_outer_rho_search = true,
    grid_resolution = 256,
    law_gap_tolerance = 0.05,
))]
fn sae_crosscoder_fit<'py>(
    py: Python<'py>,
    anchor: PyReadonlyArray2<'py, f64>,
    anchor_label: String,
    target_labels: Vec<String>,
    targets: Vec<PyReadonlyArray2<'py, f64>>,
    n_atoms: usize,
    n_harmonics: usize,
    sparsity_strength: f64,
    smoothness: f64,
    max_iter: usize,
    learning_rate: f64,
    ridge_ext_coord: f64,
    ridge_beta: f64,
    random_state: u64,
    run_outer_rho_search: bool,
    grid_resolution: usize,
    law_gap_tolerance: f64,
) -> PyResult<Py<PyDict>> {
    use gam::terms::analytic_penalties::AnalyticPenaltyRegistry;
    use gam::terms::sae::manifold::{
        NamedCrosscoderTarget, SaeCrosscoderFitRequest, SaeFitAssignmentKind, SaeFitSeedReport,
        SaeFitSeedRequest, SaeMinimalSeedReport, SaeMinimalSeedRequest, build_sae_fit_seed,
        build_sae_minimal_seed, measure_atom_transport_between, run_sae_crosscoder_fit,
        stack_crosscoder_targets,
    };

    if target_labels.len() != targets.len() {
        return Err(py_value_error(format!(
            "sae_crosscoder_fit: target_labels length {} != targets length {}",
            target_labels.len(),
            targets.len()
        )));
    }
    if n_atoms == 0 {
        return Err(py_value_error(
            "sae_crosscoder_fit: n_atoms must be positive".to_string(),
        ));
    }
    if n_harmonics == 0 {
        return Err(py_value_error(
            "sae_crosscoder_fit: n_harmonics must be positive".to_string(),
        ));
    }
    if !law_gap_tolerance.is_finite() || law_gap_tolerance < 0.0 {
        return Err(py_value_error(format!(
            "sae_crosscoder_fit: law_gap_tolerance must be finite and non-negative; got {law_gap_tolerance}"
        )));
    }

    let anchor_owned = anchor.as_array().to_owned();
    let blocks: Vec<NamedCrosscoderTarget> = target_labels
        .iter()
        .cloned()
        .zip(targets.iter())
        .map(|(label, target)| NamedCrosscoderTarget {
            label,
            target: target.as_array().to_owned(),
        })
        .collect();
    let (stacked, _, _) =
        stack_crosscoder_targets(&anchor_label, &anchor_owned, &blocks).map_err(py_value_error)?;

    let assignment = SaeFitAssignmentKind::Softmax;
    let minimal = build_sae_minimal_seed(SaeMinimalSeedRequest {
        target: stacked.view(),
        atom_basis: vec!["periodic".to_string(); n_atoms],
        atom_dim: vec![n_harmonics; n_atoms],
        assignment_kind: assignment,
        alpha: 1.0,
        tau: 1.0,
        threshold: 0.0,
        top_k: None,
        ibp_alpha_override: None,
        random_state,
        initial_logits: None,
        initial_coords: None,
    })
    .map_err(py_value_error)?;
    let SaeMinimalSeedReport {
        atom_basis,
        effective_atom_dim,
        atom_centers,
        basis_values,
        basis_jacobian,
        basis_sizes,
        decoder_coefficients,
        smooth_penalties,
        initial_logits,
        initial_coords,
        refine_routing,
    } = minimal;
    let registry = AnalyticPenaltyRegistry::new();
    let seed = build_sae_fit_seed(SaeFitSeedRequest {
        target: stacked.view(),
        atom_basis: &atom_basis,
        atom_dim: &effective_atom_dim,
        atom_centers: &atom_centers,
        basis_values: basis_values.view(),
        basis_jacobian: basis_jacobian.view(),
        basis_sizes: &basis_sizes,
        decoder_coefficients: decoder_coefficients.view(),
        smooth_penalties: smooth_penalties.view(),
        initial_logits: initial_logits.view(),
        initial_coords: initial_coords.view(),
        alpha: 1.0,
        tau: 1.0,
        learnable_alpha: false,
        assignment_kind: assignment,
        sparsity_strength,
        smoothness,
        max_iter,
        learning_rate,
        ridge_ext_coord,
        ridge_beta,
        top_k: None,
        threshold: 0.0,
        native_ard_enabled: true,
        seed_refine_routing: refine_routing,
        seed_refine_random_state: random_state,
        data_row_reseed: false,
        fit_config: gam::terms::sae::manifold::SaeFitConfig::default(),
        temperature_schedule: None,
        fisher_metric: None,
        row_loss_weights: None,
        registry: &registry,
    })
    .map_err(py_value_error)?;
    let SaeFitSeedReport {
        base_term,
        initial_rho,
        ..
    } = seed;
    let cancel = std::sync::Arc::new(std::sync::atomic::AtomicBool::new(false));
    let request = SaeCrosscoderFitRequest {
        anchor_label: anchor_label.clone(),
        anchor: anchor_owned,
        blocks,
        base_term,
        registry,
        initial_rho,
        max_iter,
        learning_rate,
        ridge_ext_coord,
        ridge_beta,
        run_outer_rho_search,
        cancel: Some(std::sync::Arc::clone(&cancel)),
    };
    let report = run_sae_fit_interruptible(py, "gam-sae-crosscoder-fit", &cancel, move || {
        run_sae_crosscoder_fit(request)
    })?
    .map_err(|error| sae_fit_error_to_pyerr(py, error))?;

    let block_labels = report.layout.labels().to_vec();
    let out = PyDict::new(py);
    let layout = PyDict::new(py);
    layout.set_item("anchor_label", &anchor_label)?;
    layout.set_item("anchor_dim", report.layout.anchor_dim())?;
    layout.set_item("block_dims", report.layout.block_dims().to_vec())?;
    layout.set_item("labels", block_labels.clone())?;
    layout.set_item(
        "log_lambda_block",
        report.layout.block_log_lambda().to_vec(),
    )?;
    out.set_item("layout", layout)?;
    out.set_item("log_lambda_block", report.rho.log_lambda_block.clone())?;
    out.set_item("log_lambda_sparse", report.rho.log_lambda_sparse)?;
    out.set_item("log_lambda_smooth", report.rho.log_lambda_smooth.clone())?;
    out.set_item(
        "assignments",
        report.term.assignment.assignments().into_pyarray(py),
    )?;
    out.set_item(
        "logits",
        report.term.assignment.logits.clone().into_pyarray(py),
    )?;
    let coords = PyList::empty(py);
    for coord in &report.term.assignment.coords {
        coords.append(coord.as_matrix().into_pyarray(py))?;
    }
    out.set_item("coords", coords)?;

    let loss = PyDict::new(py);
    loss.set_item("data_fit", report.loss.data_fit)?;
    loss.set_item("assignment_sparsity", report.loss.assignment_sparsity)?;
    loss.set_item("smoothness", report.loss.smoothness)?;
    loss.set_item("ard", report.loss.ard)?;
    loss.set_item("total_penalized_loss", report.loss.total())?;
    out.set_item("loss", loss)?;
    let termination = PyDict::new(py);
    termination.set_item("verdict", report.termination.verdict.as_str())?;
    termination.set_item("evals", report.termination.evals)?;
    termination.set_item(
        "evals_since_improvement",
        report.termination.evals_since_improvement,
    )?;
    termination.set_item("wall_seconds", report.termination.wall.as_secs_f64())?;
    out.set_item("termination", termination)?;

    let layers = PyList::empty(py);
    for layer_report in &report.layers {
        let layer = PyDict::new(py);
        layer.set_item("label", &layer_report.label)?;
        layer.set_item("fitted", layer_report.fitted.clone().into_pyarray(py))?;
        layer.set_item("reconstruction_r2", layer_report.reconstruction_r2)?;
        let decoders = PyList::empty(py);
        for decoder in &layer_report.decoders {
            decoders.append(decoder.clone().into_pyarray(py))?;
        }
        layer.set_item("decoders", decoders)?;
        layers.append(layer)?;
    }
    out.set_item("layers", layers)?;
    let drift = crosscoder_drift_dict(py, &report.drift, &anchor_label, &block_labels)?;
    out.set_item("drift", &drift)?;

    let transport = PyList::empty(py);
    let chain: Vec<gam::terms::sae::manifold::CrosscoderLayer> =
        std::iter::once(gam::terms::sae::manifold::CrosscoderLayer::Anchor)
            .chain(
                (0..report.layout.num_blocks())
                    .map(gam::terms::sae::manifold::CrosscoderLayer::Block),
            )
            .collect();
    for atom in 0..report.term.k_atoms() {
        for pair in chain.windows(2) {
            let measured = measure_atom_transport_between(
                &report.term,
                &report.layout,
                atom,
                pair[0],
                pair[1],
                grid_resolution,
            )
            .map_err(py_value_error)?;
            transport.append(crosscoder_transport_dict(
                py,
                &measured,
                &anchor_label,
                &block_labels,
                law_gap_tolerance,
            )?)?;
        }
    }
    out.set_item("transport", &transport)?;
    // Same typed shape `ManifoldSaePayload.crosscoder` persists. Keeping the
    // flat convenience keys above makes the live report ergonomic while this
    // nested block is the single serialization unit.
    let crosscoder = PyDict::new(py);
    crosscoder.set_item("anchor_label", &anchor_label)?;
    crosscoder.set_item("anchor_dim", report.layout.anchor_dim())?;
    crosscoder.set_item("block_dims", report.layout.block_dims().to_vec())?;
    crosscoder.set_item("labels", block_labels)?;
    crosscoder.set_item(
        "log_lambda_block",
        report.layout.block_log_lambda().to_vec(),
    )?;
    crosscoder.set_item("drift", drift)?;
    crosscoder.set_item("transport", transport)?;
    out.set_item("crosscoder", crosscoder)?;
    Ok(out.unbind())
}
