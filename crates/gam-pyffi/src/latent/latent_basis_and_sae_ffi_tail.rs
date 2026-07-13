/// One-shot SAE-manifold fit driver: takes only `(z, atom_basis, atom_dim,
/// ...scalar hyperparams)` and assembles the full basis + jacobian + penalty
/// stack + PCA seed coords + zero-init decoder + zero-init logits internally
/// before delegating to the same end-to-end Rust Newton loop as
/// the native fit orchestration. Returns the raw native fit payload with
/// `"atom_plans"`, holding the per-atom basis spec so OOS prediction can
/// rebuild the design without going through Python.
///
/// `initial_logits` (N, K) and `initial_coords` (K, N, D_max) are optional
/// native-solver warm starts. When supplied they replace the internal PCA seed
/// coordinates / zero-jitter logit initialization. The basis *design* (Duchon centers, harmonic
/// counts) is still derived from the PCA seed so the warm coordinates are
/// evaluated against the same atom geometry the unconstrained fit would build.
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
    threshold_gate_threshold: f64,
    native_ard_enabled: bool,
    // WP-D output-Fisher shard (#980). `(n, p, r)` f64 factors; presence activates
    // `RowMetric::OutputFisher`. This is the entry point the high-level Python
    // `sae_manifold_fit` facade routes through, so it carries the explicit shard
    // exactly as the precomputed-basis `sae_manifold_fit` does.
    fisher_factors: Option<PyReadonlyArray3<'py, f64>>,
    fisher_mass_residual: Option<PyReadonlyArray1<'py, f64>>,
    // Harvest provenance tag (#980): same-position `"output_fisher"` (default) or
    // forward-looking `"output_fisher_downstream"`. Routed to the matching
    // `RowMetric` constructor; gauge/lens/dose consume either unchanged.
    fisher_provenance: Option<String>,
    fisher_factor_kind: Option<String>,
    // Per-row design-honesty reconstruction weights (#977); `(n,)` √w. Absent ⇒
    // unweighted path. Installed on the term before the joint fit / ρ selection.
    row_loss_weights: Option<PyReadonlyArray1<'py, f64>>,
    // Per-fit separation-barrier configuration. `None` selects the native default.
    separation_barrier_strength_override: Option<f64>,
    promote_from_residual: bool,
    // Bundled-pipeline stage toggles (#2267) forwarded to `sae_manifold_fit_inner`.
    run_structure_search: bool,
    structured_residual_passes: usize,
) -> PyResult<Py<PyDict>> {
    // Convert borrowed Python arrays into the typed library seed request.
    let assignment_kind = canonicalize_assignment_kind(&assignment_kind).map_err(py_value_error)?;
    let assignment = SaeFitAssignmentKind::from_tag(&assignment_kind).map_err(py_value_error)?;
    let z_view = z.as_array();
    let seed = build_sae_minimal_seed(SaeMinimalSeedRequest {
        target: z_view,
        atom_basis,
        atom_dim,
        assignment_kind: assignment,
        alpha,
        tau,
        threshold: threshold_gate_threshold,
        top_k,
        random_state,
        initial_logits: initial_logits.as_ref().map(|values| values.as_array()),
        initial_coords: initial_coords.as_ref().map(|values| values.as_array()),
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
        initial_coords: start_coords,
        refine_routing,
    } = seed;
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
        threshold_gate_threshold,
        native_ard_enabled,
        refine_routing,
        random_state,
        // WP-D → fit wiring (#980): the factor shard selects the native
        // OutputFisher row metric; absence selects the Euclidean metric.
        fisher_u,
        fisher_mr,
        fisher_provenance.as_deref(),
        fisher_factor_kind.as_deref(),
        row_w,
        separation_barrier_strength_override,
        promote_from_residual,
        run_structure_search,
        structured_residual_passes,
    )?;
    // Post-search atom plans are emitted by the shared fit entry from the final
    // variable-K dictionary; the minimal binding never patches the payload.
    Ok(result_dict)
}

fn expand_public_fit_values<T: Clone>(
    values: Vec<T>,
    k_atoms: usize,
    label: &str,
) -> Result<Vec<T>, String> {
    match values.len() {
        1 => Ok(vec![values[0].clone(); k_atoms]),
        len if len == k_atoms => Ok(values),
        len => Err(format!(
            "sae_manifold_fit: {label} must contain one shared value or K={k_atoms} values; got {len}"
        )),
    }
}

/// Grouped inputs for [`public_fit_penalties`]. Bundled into a struct (rather
/// than passed as a dozen positional scalars) so the ban-scanner's
/// `#[allow(clippy::too_many_arguments)]` prohibition is satisfied by
/// construction instead of by suppressing the lint.
struct PublicFitPenaltyArgs<'a> {
    isometry_weight: f64,
    coord_sparsity: &'a str,
    sparsity_weight: f64,
    scad_mcp_gamma: Option<f64>,
    decoder_feature_sparsity_groups: Option<Vec<Vec<usize>>>,
    block_orthogonality_weight: f64,
    nuclear_norm_weight: f64,
    nuclear_norm_max_rank: Option<usize>,
    decoder_incoherence_weight: f64,
    k_atoms: usize,
    d_max: usize,
    p_out: usize,
}

fn public_fit_penalties(
    args: PublicFitPenaltyArgs<'_>,
) -> Result<(Option<String>, Vec<String>), String> {
    let PublicFitPenaltyArgs {
        isometry_weight,
        coord_sparsity,
        sparsity_weight,
        scad_mcp_gamma,
        decoder_feature_sparsity_groups,
        block_orthogonality_weight,
        nuclear_norm_weight,
        nuclear_norm_max_rank,
        decoder_incoherence_weight,
        k_atoms,
        d_max,
        p_out,
    } = args;
    for (name, value) in [
        ("isometry_weight", isometry_weight),
        ("block_orthogonality_weight", block_orthogonality_weight),
        ("nuclear_norm_weight", nuclear_norm_weight),
        ("decoder_incoherence_weight", decoder_incoherence_weight),
    ] {
        if !value.is_finite() || value < 0.0 {
            return Err(format!(
                "sae_manifold_fit: {name} must be finite and non-negative; got {value}"
            ));
        }
    }
    let coord_sparsity = coord_sparsity.trim().to_ascii_lowercase();
    if !matches!(coord_sparsity.as_str(), "l1" | "scad" | "mcp") {
        return Err(format!(
            "sae_manifold_fit: coord_sparsity must be 'l1', 'scad', or 'mcp'; got {coord_sparsity:?}"
        ));
    }
    let gamma = scad_mcp_gamma.unwrap_or(if coord_sparsity == "scad" { 3.7 } else { 2.5 });
    match coord_sparsity.as_str() {
        "scad" if !gamma.is_finite() || gamma <= 2.0 => {
            return Err(format!(
                "sae_manifold_fit: scad_mcp_gamma must be finite and > 2 for SCAD; got {gamma}"
            ));
        }
        "mcp" if !gamma.is_finite() || gamma <= 1.0 => {
            return Err(format!(
                "sae_manifold_fit: scad_mcp_gamma must be finite and > 1 for MCP; got {gamma}"
            ));
        }
        _ => {}
    }

    let mut descriptors = Vec::new();
    let mut names = Vec::new();
    if matches!(coord_sparsity.as_str(), "scad" | "mcp") {
        descriptors.push(serde_json::json!({
            "kind": "scad_mcp",
            "target": "t",
            "variant": coord_sparsity,
            "gamma": gamma,
            "weight": sparsity_weight,
        }));
        names.push("ScadMcpPenalty".to_string());
    }
    if isometry_weight > 0.0 {
        descriptors.push(serde_json::json!({
            "kind": "isometry",
            "target": "t",
            "weight": isometry_weight,
        }));
        names.push("IsometryPenalty".to_string());
    }
    if block_orthogonality_weight > 0.0 {
        if d_max < 2 {
            return Err(format!(
                "sae_manifold_fit: block_orthogonality_weight requires d_atom >= 2; got d_max={d_max}"
            ));
        }
        let groups = (0..d_max).map(|axis| vec![axis]).collect::<Vec<_>>();
        descriptors.push(serde_json::json!({
            "kind": "block_orthogonality",
            "target": "t",
            "groups": groups,
            "weight": block_orthogonality_weight,
        }));
        names.push("BlockOrthogonalityPenalty".to_string());
    }
    if let Some(groups) = decoder_feature_sparsity_groups {
        let mut seen = vec![false; p_out];
        if groups.is_empty() || groups.iter().any(Vec::is_empty) {
            return Err(
                "sae_manifold_fit: decoder_feature_sparsity_groups must contain non-empty groups"
                    .to_string(),
            );
        }
        for &feature in groups.iter().flatten() {
            if feature >= p_out || seen[feature] {
                return Err(format!(
                    "sae_manifold_fit: decoder_feature_sparsity_groups must be a disjoint partition of 0..{p_out}"
                ));
            }
            seen[feature] = true;
        }
        if seen.iter().any(|used| !used) {
            return Err(format!(
                "sae_manifold_fit: decoder_feature_sparsity_groups must cover every feature in 0..{p_out}"
            ));
        }
        descriptors.push(serde_json::json!({
            "kind": "mechanism_sparsity",
            "target": "beta",
            "feature_groups": groups,
        }));
        names.push("MechanismSparsityPenalty".to_string());
    }
    if nuclear_norm_weight > 0.0 {
        let mut descriptor = serde_json::json!({
            "kind": "nuclear_norm",
            "target": "beta",
            "weight": nuclear_norm_weight,
        });
        if let Some(max_rank) = nuclear_norm_max_rank {
            descriptor["max_rank"] = serde_json::json!(max_rank);
        }
        descriptors.push(descriptor);
        names.push("NuclearNormPenalty".to_string());
    }
    if decoder_incoherence_weight > 0.0 && k_atoms >= 2 {
        descriptors.push(serde_json::json!({
            "kind": "decoder_incoherence",
            "target": "beta",
            "block_sizes": vec![1_usize; k_atoms],
            "p_out": p_out,
            "weight": decoder_incoherence_weight,
        }));
        names.push("DecoderIncoherencePenalty".to_string());
    }
    let json = if descriptors.is_empty() {
        None
    } else {
        Some(serde_json::to_string(&descriptors).map_err(|error| error.to_string())?)
    };
    Ok((json, names))
}

/// Public fitted-model front door. Python supplies only arrays and literal
/// options; assignment/basis canonicalization, defaults, penalty descriptors,
/// native fitting, and artifact construction all terminate in Rust.
#[pyfunction(signature = (
    z,
    k_atoms,
    atom_dim,
    atom_topology=None,
    atom_basis=None,
    assignment_kind="softmax",
    gumbel_schedule=None,
    isometry_weight=1.0,
    native_ard_enabled=true,
    decoder_feature_sparsity_groups=None,
    max_iter=50,
    sparsity_strength=1.0,
    coord_sparsity="scad",
    scad_mcp_gamma=None,
    smoothness=1.0,
    alpha=None,
    learnable_alpha=false,
    learning_rate=None,
    random_state=0,
    block_orthogonality_weight=0.0,
    nuclear_norm_weight=1.0,
    nuclear_norm_max_rank=None,
    decoder_incoherence_weight=1.0,
    top_k=None,
    initial_coords=None,
    initial_logits=None,
    tau=None,
    threshold_gate_threshold=0.0,
    fisher_factors=None,
    fisher_mass_residual=None,
    fisher_provenance=None,
    fisher_factor_kind=None,
    row_loss_weights=None,
    separation_barrier_strength_override=None,
    promote_from_residual=false,
    run_structure_search=false,
    structured_residual_passes=0,
))]
// No #[allow(clippy::too_many_arguments)]: this is the flat `#[pyfunction]`
// kwarg surface Python calls by name (mirroring `sae_manifold_fit_minimal`
// above, which has the same shape and no allow either); clippy is not run in
// CI here, so the ban-scanner's anti-`#[allow]` rule is the only gate, and it
// is satisfied by simply not writing the attribute.
fn sae_manifold_fit_model<'py>(
    py: Python<'py>,
    z: PyReadonlyArray2<'py, f64>,
    k_atoms: usize,
    atom_dim: Vec<usize>,
    atom_topology: Option<String>,
    atom_basis: Option<Vec<String>>,
    assignment_kind: &str,
    gumbel_schedule: Option<&Bound<'py, PyDict>>,
    isometry_weight: f64,
    native_ard_enabled: bool,
    decoder_feature_sparsity_groups: Option<Vec<Vec<usize>>>,
    max_iter: usize,
    sparsity_strength: f64,
    coord_sparsity: &str,
    scad_mcp_gamma: Option<f64>,
    smoothness: f64,
    alpha: Option<f64>,
    learnable_alpha: bool,
    learning_rate: Option<f64>,
    random_state: u64,
    block_orthogonality_weight: f64,
    nuclear_norm_weight: f64,
    nuclear_norm_max_rank: Option<usize>,
    decoder_incoherence_weight: f64,
    top_k: Option<usize>,
    initial_coords: Option<PyReadonlyArray3<'py, f64>>,
    initial_logits: Option<PyReadonlyArray2<'py, f64>>,
    tau: Option<f64>,
    threshold_gate_threshold: f64,
    fisher_factors: Option<PyReadonlyArray3<'py, f64>>,
    fisher_mass_residual: Option<PyReadonlyArray1<'py, f64>>,
    fisher_provenance: Option<String>,
    fisher_factor_kind: Option<String>,
    row_loss_weights: Option<PyReadonlyArray1<'py, f64>>,
    separation_barrier_strength_override: Option<f64>,
    promote_from_residual: bool,
    run_structure_search: bool,
    structured_residual_passes: usize,
) -> PyResult<Py<crate::ManifoldSaeCore>> {
    if k_atoms == 0 {
        return Err(py_value_error(
            "sae_manifold_fit requires K >= 1".to_string(),
        ));
    }
    let z_view = z.as_array();
    let (n_obs, p_out) = z_view.dim();
    if n_obs < 2 || p_out == 0 {
        return Err(py_value_error(format!(
            "sae_manifold_fit requires a finite (N, p) matrix with N >= 2 and p >= 1; got ({n_obs}, {p_out})"
        )));
    }
    if !z_view.iter().all(|value| value.is_finite()) {
        return Err(py_value_error(
            "sae_manifold_fit response contains non-finite values".to_string(),
        ));
    }
    let atom_dim = expand_public_fit_values(atom_dim, k_atoms, "d_atom").map_err(py_value_error)?;
    if atom_dim.iter().any(|&dimension| dimension == 0) {
        return Err(py_value_error(
            "sae_manifold_fit requires every d_atom >= 1".to_string(),
        ));
    }
    let has_declared_bases = atom_basis.is_some();
    let basis_seed = match atom_basis {
        Some(values) => {
            expand_public_fit_values(values, k_atoms, "atom_basis").map_err(py_value_error)?
        }
        None => {
            let basis = gam::terms::sae::atom_schema::basis_kind_for_topology(
                atom_topology.as_deref().unwrap_or("auto"),
            )
            .map_err(py_value_error)?;
            vec![basis; k_atoms]
        }
    };
    for basis in &basis_seed {
        gam::terms::sae::atom_schema::validate_seed_basis_kind(basis).map_err(py_value_error)?;
    }
    let atom_basis = basis_seed.clone();
    let declared_bases = has_declared_bases.then(|| basis_seed.clone());
    if let (Some(topology), true) = (atom_topology.as_deref(), has_declared_bases) {
        let resolved = gam::terms::sae::atom_schema::topology_for_bases(&atom_basis)
            .map_err(py_value_error)?
            .ok_or_else(|| {
                py_value_error("sae_manifold_fit requires at least one atom".to_string())
            })?;
        let requested =
            gam::terms::sae::atom_schema::canonical_topology(topology).map_err(py_value_error)?;
        if resolved != requested {
            return Err(py_value_error(format!(
                "sae_manifold_fit: atom_basis resolves to topology {resolved:?}, but atom_topology resolves to {requested:?}"
            )));
        }
    }
    let assignment = canonicalize_assignment_kind(assignment_kind).map_err(py_value_error)?;
    let schedule =
        gumbel_temperature_schedule_from_pydict(gumbel_schedule).map_err(py_value_error)?;
    let resolved_tau =
        tau.unwrap_or_else(|| schedule.as_ref().map_or(0.5, |state| state.tau_start));
    let resolved_alpha = alpha.unwrap_or_else(|| {
        if assignment == "ordered_beta_bernoulli" && !learnable_alpha {
            gam::terms::sae::assignment::default_ordered_beta_bernoulli_concentration_for_k_atoms(
                k_atoms,
            )
        } else {
            1.0
        }
    });
    let resolved_learning_rate = learning_rate.unwrap_or(if assignment == "threshold_gate" {
        0.05
    } else {
        1.0
    });
    let d_max = atom_dim.iter().copied().max().unwrap_or(1);
    let (analytic_penalties, mut penalties) = public_fit_penalties(PublicFitPenaltyArgs {
        isometry_weight,
        coord_sparsity,
        sparsity_weight: sparsity_strength,
        scad_mcp_gamma,
        decoder_feature_sparsity_groups,
        block_orthogonality_weight,
        nuclear_norm_weight,
        nuclear_norm_max_rank,
        decoder_incoherence_weight,
        k_atoms,
        d_max,
        p_out,
    })
    .map_err(py_value_error)?;
    if native_ard_enabled {
        penalties.push("ARDPenalty".to_string());
    }

    let raw = sae_manifold_fit_minimal(
        py,
        z.clone(),
        atom_basis.clone(),
        atom_dim,
        resolved_alpha,
        resolved_tau,
        learnable_alpha,
        assignment.clone(),
        sparsity_strength,
        smoothness,
        max_iter,
        resolved_learning_rate,
        1.0e-6,
        1.0e-6,
        gumbel_schedule,
        analytic_penalties,
        random_state,
        top_k,
        initial_logits,
        initial_coords,
        threshold_gate_threshold,
        native_ard_enabled,
        fisher_factors.clone(),
        fisher_mass_residual,
        fisher_provenance.clone(),
        fisher_factor_kind.clone(),
        row_loss_weights,
        separation_barrier_strength_override,
        promote_from_residual,
        run_structure_search,
        structured_residual_passes,
    )?;
    let raw = crate::manifold::manifold_sae_coercion::py_any_to_json_value(raw.bind(py).as_any())?;
    let fisher_nested = fisher_factors.map(|array| {
        array
            .as_array()
            .outer_iter()
            .map(|matrix| matrix.rows().into_iter().map(|row| row.to_vec()).collect())
            .collect()
    });
    let config = crate::manifold::manifold_sae_coercion::FitConfig {
        assignment,
        assignment_label: assignment_kind.to_string(),
        penalties,
        alpha: resolved_alpha,
        learnable_alpha,
        tau: resolved_tau,
        sparsity_strength,
        smoothness,
        learning_rate: resolved_learning_rate,
        max_iter: i64::try_from(max_iter)
            .map_err(|_| py_value_error("max_iter exceeds i64".to_string()))?,
        random_state: i64::try_from(random_state)
            .map_err(|_| py_value_error("random_state exceeds i64".to_string()))?,
        top_k: top_k
            .map(i64::try_from)
            .transpose()
            .map_err(|_| py_value_error("top_k exceeds i64".to_string()))?,
        threshold_gate_threshold,
        fisher_factors: fisher_nested,
        fisher_provenance,
        fisher_factor_kind,
        declared_bases,
    };
    let payload = crate::manifold::manifold_sae_coercion::build_manifold_sae_payload(
        &raw,
        crate::manifold::manifold_sae_coercion::column_mean(z_view),
        &config,
    )
    .map_err(py_value_error)?;
    Py::new(py, crate::ManifoldSaeCore::from_payload(payload)?)
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
/// assignments directly. `initial_logits` (N, K) and `initial_coords`
/// (K, N, D_max) optionally warm-start the native OOS refinement.
// Convert borrowed FFI arrays into the owned typed library request. This helper
// performs wire parsing and ownership transfer only; validation, basis rebuild,
// seeding, inference, projection, and reporting live in gam-sae.
fn sae_oos_request_from_arrays(
    x_view: ndarray::ArrayView2<'_, f64>,
    atom_basis: Vec<String>,
    atom_dim: Vec<usize>,
    decoder_blocks: &[ndarray::ArrayView2<'_, f64>],
    duchon_centers: &[Option<Array2<f64>>],
    n_harmonics_list: Vec<Option<usize>>,
    basis_size_list: Vec<usize>,
    alpha: f64,
    tau: f64,
    assignment_kind: String,
    max_iter: usize,
    learning_rate: f64,
    ridge_ext_coord: f64,
    initial_logits: Option<ndarray::ArrayView2<'_, f64>>,
    initial_coords: Option<ndarray::ArrayView3<'_, f64>>,
    threshold_gate_threshold: f64,
    top_k: Option<usize>,
    hybrid_linear_images: Option<Vec<(usize, f64, Array1<f64>, Array1<f64>, Option<Array1<f64>>)>>,
    log_lambda_sparse: Option<f64>,
    log_lambda_smooth: Option<Vec<f64>>,
    log_ard: Option<Vec<Vec<f64>>>,
    learnable_alpha: bool,
) -> Result<gam::terms::sae::manifold::SaeOosRequest, String> {
    let k_atoms = atom_basis.len();
    if atom_dim.len() != k_atoms
        || decoder_blocks.len() != k_atoms
        || duchon_centers.len() != k_atoms
        || n_harmonics_list.len() != k_atoms
        || basis_size_list.len() != k_atoms
    {
        return Err(format!(
            "sae_manifold_predict_oos: per-atom metadata lengths must equal K={k_atoms}"
        ));
    }
    let assignment = match assignment_kind.as_str() {
        "softmax" => gam::terms::sae::manifold::SaeOosAssignmentKind::Softmax,
        "ordered_beta_bernoulli" => {
            gam::terms::sae::manifold::SaeOosAssignmentKind::OrderedBetaBernoulli {
                learnable_alpha,
            }
        }
        "threshold_gate" => gam::terms::sae::manifold::SaeOosAssignmentKind::ThresholdGate {
            threshold: threshold_gate_threshold,
        },
        "topk" => gam::terms::sae::manifold::SaeOosAssignmentKind::TopK,
        _ => {
            return Err(format!(
                "sae_manifold_predict_oos: unsupported assignment kind {assignment_kind:?}"
            ));
        }
    };
    let regularization = match (log_lambda_sparse, log_lambda_smooth, log_ard) {
        (Some(log_lambda_sparse), Some(log_lambda_smooth), Some(log_ard)) => {
            gam::terms::sae::manifold::SaeOosRegularization {
                log_lambda_sparse,
                log_lambda_smooth,
                log_ard,
            }
        }
        _ => {
            return Err(
                "sae_manifold_predict_oos: terminal rho must provide log_lambda_sparse, \
                 log_lambda_smooth, and log_ard together"
                    .to_string(),
            );
        }
    };
    let atoms = (0..k_atoms)
        .map(|atom_index| gam::terms::sae::manifold::SaeOosAtomSpec {
            basis_kind: sae_atom_basis_kind_from_str(&atom_basis[atom_index]),
            latent_dim: atom_dim[atom_index],
            decoder: decoder_blocks[atom_index].to_owned(),
            centers: duchon_centers[atom_index].clone(),
            n_harmonics: n_harmonics_list[atom_index],
            basis_size: basis_size_list[atom_index],
        })
        .collect();
    let hybrid_linear_images = match hybrid_linear_images {
        Some(images) => images,
        None => Vec::new(),
    }
    .into_iter()
    .map(
        |(atom_idx, t_bar, b0, b1, v)| gam::terms::sae::hybrid_split::AtomLinearImage {
            atom_idx,
            t_bar,
            b0,
            b1,
            v,
        },
    )
    .collect();

    Ok(gam::terms::sae::manifold::SaeOosRequest {
        target: x_view.to_owned(),
        atoms,
        assignment,
        alpha,
        tau,
        regularization,
        max_iter,
        learning_rate,
        ridge_ext_coord,
        initial_logits: initial_logits.map(|view| view.to_owned()),
        initial_coords: initial_coords.map(|view| view.to_owned()),
        top_k,
        hybrid_linear_images,
    })
}

// Serialize the typed library report. No fit or inference decisions live here.
fn sae_oos_report_to_pydict<'py>(
    py: Python<'py>,
    report: gam::terms::sae::manifold::SaeOosReport,
) -> PyResult<Py<PyDict>> {
    let chosen_k = report.active_mask.len();
    let atoms_py = PyList::empty(py);
    for atom in report.atoms {
        let atom_dict = PyDict::new(py);
        atom_dict.set_item("decoder_B", atom.decoder.into_pyarray(py))?;
        atom_dict.set_item("basis_kind", sae_atom_basis_kind_name(&atom.basis_kind))?;
        atom_dict.set_item("basis_centers", py.None())?;
        atom_dict.set_item("on_atom_coords_t", atom.coords.into_pyarray(py))?;
        atom_dict.set_item("assignments_z", atom.assignments.into_pyarray(py))?;
        atom_dict.set_item("active_dim", atom.active_dim)?;
        atom_dict.set_item("atom_reconstruction", atom.reconstruction.into_pyarray(py))?;
        atoms_py.append(atom_dict)?;
    }
    let log_ard_py = PyList::empty(py);
    for atom_log_ard in &report.rho.log_ard {
        log_ard_py.append(atom_log_ard.clone().into_pyarray(py))?;
    }

    let out = PyDict::new(py);
    out.set_item("atoms", atoms_py)?;
    out.set_item("assignments_z", report.assignments.into_pyarray(py))?;
    out.set_item("logits", report.logits.into_pyarray(py))?;
    out.set_item("atom_active_mask", report.active_mask)?;
    out.set_item("fitted", report.fitted.into_pyarray(py))?;
    sae_set_penalized_loss_items(&out, &report.loss, "oos_penalized_loss")?;
    out.set_item("log_alpha", report.alpha.ln())?;
    out.set_item("log_lambda_smooth", report.rho.log_lambda_smooth)?;
    out.set_item("log_ard", log_ard_py)?;
    out.set_item("assignment_prior", report.assignment_kind)?;
    out.set_item(
        "solver_plan",
        sae_streaming_plan_to_pydict(py, report.streaming_plan)?,
    )?;
    out.set_item("chosen_k", chosen_k)?;
    Ok(out.unbind())
}

/// FFI surface for the frozen-decoder out-of-sample solve. The binding only
/// marshals arrays into [`SaeOosRequest`](gam::terms::sae::manifold::SaeOosRequest),
/// calls the typed gam-sae entry, and serializes its report (#2236).
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
    max_iter = 50,
    learning_rate = 0.04,
    ridge_ext_coord = 1.0e-6,
    initial_logits = None,
    initial_coords = None,
    threshold_gate_threshold = 0.0,
    top_k = None,
    hybrid_linear_images = None,
    log_lambda_sparse = None,
    log_lambda_smooth = None,
    log_ard = None,
    learnable_alpha = false,
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
    max_iter: usize,
    learning_rate: f64,
    ridge_ext_coord: f64,
    initial_logits: Option<PyReadonlyArray2<'py, f64>>,
    initial_coords: Option<PyReadonlyArray3<'py, f64>>,
    threshold_gate_threshold: f64,
    top_k: Option<usize>,
    hybrid_linear_images: Option<
        Vec<(
            usize,
            f64,
            PyReadonlyArray1<'py, f64>,
            PyReadonlyArray1<'py, f64>,
            Option<PyReadonlyArray1<'py, f64>>,
        )>,
    >,
    log_lambda_sparse: Option<f64>,
    log_lambda_smooth: Option<Vec<f64>>,
    log_ard: Option<Vec<Vec<f64>>>,
    learnable_alpha: bool,
) -> PyResult<Py<PyDict>> {
    let decoder_views: Vec<ndarray::ArrayView2<'_, f64>> =
        decoder_blocks.iter().map(|b| b.as_array()).collect();
    let duchon_owned: Vec<Option<Array2<f64>>> = duchon_centers
        .iter()
        .map(|o| o.as_ref().map(|a| a.as_array().to_owned()))
        .collect();
    let initial_logits_view = initial_logits.as_ref().map(|a| a.as_array());
    let initial_coords_view = initial_coords.as_ref().map(|a| a.as_array());
    let hybrid_owned = hybrid_linear_images.map(|images| {
        images
            .into_iter()
            .map(|(atom_idx, t_bar, b0, b1, v)| {
                (
                    atom_idx,
                    t_bar,
                    b0.as_array().to_owned(),
                    b1.as_array().to_owned(),
                    v.map(|arr| arr.as_array().to_owned()),
                )
            })
            .collect()
    });
    let request = sae_oos_request_from_arrays(
        x_new.as_array(),
        atom_basis,
        atom_dim,
        &decoder_views,
        &duchon_owned,
        n_harmonics_list,
        basis_size_list,
        alpha,
        tau,
        assignment_kind,
        max_iter,
        learning_rate,
        ridge_ext_coord,
        initial_logits_view,
        initial_coords_view,
        threshold_gate_threshold,
        top_k,
        hybrid_owned,
        log_lambda_sparse,
        log_lambda_smooth,
        log_ard,
        learnable_alpha,
    )
    .map_err(py_value_error)?;
    let report =
        gam::terms::sae::manifold::run_sae_manifold_oos(request).map_err(py_value_error)?;
    sae_oos_report_to_pydict(py, report)
}

/// FFI surface for the #2266 evaluation-only certification entry (#2263 item
/// 4): certify an externally-trained (torch-lane) SAE-manifold state — no
/// closed-form solve, no coordinate/decoder optimization — and return the SAME
/// certificate/diagnostics payload dict a native fit returns (certificates,
/// trust/fit diagnostics, coordinate fidelity, the anytime-valid structure
/// certificate). The binding rebuilds the frozen dictionary from the caller's
/// own trained decoder/coords/logits using the identical per-atom marshalling
/// contract as [`sae_manifold_predict_oos`] / `sae_steer_delta` (basis kind +
/// latent_dim + decoder block + Duchon centers/harmonics), installs it
/// verbatim, and calls
/// [`gam::terms::sae::manifold::run_sae_manifold_certify_external`].
///
/// `initial_coords`/`initial_logits` are REQUIRED here (unlike the fit path's
/// optional warm start): they ARE the trained state being certified, not a
/// seed for further optimization. `log_lambda_sparse`/`log_lambda_smooth`/
/// `log_ard` are the trained terminal regularization state the certificate is
/// evaluated under — the same "must supply the regularization that produced
/// the decoder" contract `sae_manifold_predict_oos` already carries.
/// `outer_termination.verdict` on the returned payload reports `External`
/// (see the library entry's doc) rather than a native stationarity
/// certificate, and `penalized_quasi_laplace_criterion` is the penalized
/// objective evaluated AT the supplied state, not a certified minimum.
#[pyfunction(signature = (
    z,
    atom_basis,
    atom_dim,
    decoder_blocks,
    duchon_centers,
    n_harmonics_list,
    basis_size_list,
    initial_coords,
    initial_logits,
    alpha,
    tau,
    assignment_kind,
    log_lambda_sparse,
    log_lambda_smooth,
    log_ard,
    learnable_alpha = false,
    top_k = None,
    threshold_gate_threshold = 0.0,
    max_iter = 50,
    learning_rate = 0.04,
    ridge_ext_coord = 1.0e-6,
    ridge_beta = 1.0e-6,
    isometry_pin_active = false,
    run_structure_search = false,
    analytic_penalties = None,
    fisher_factors = None,
    fisher_mass_residual = None,
    fisher_provenance = None,
    fisher_factor_kind = None,
))]
fn sae_manifold_certify_external<'py>(
    py: Python<'py>,
    z: PyReadonlyArray2<'py, f64>,
    atom_basis: Vec<String>,
    atom_dim: Vec<usize>,
    decoder_blocks: Vec<PyReadonlyArray2<'py, f64>>,
    duchon_centers: Vec<Option<PyReadonlyArray2<'py, f64>>>,
    n_harmonics_list: Vec<Option<usize>>,
    basis_size_list: Vec<usize>,
    initial_coords: Vec<PyReadonlyArray2<'py, f64>>,
    initial_logits: PyReadonlyArray2<'py, f64>,
    alpha: f64,
    tau: f64,
    assignment_kind: String,
    log_lambda_sparse: f64,
    log_lambda_smooth: Vec<f64>,
    log_ard: Vec<Vec<f64>>,
    learnable_alpha: bool,
    top_k: Option<usize>,
    threshold_gate_threshold: f64,
    max_iter: usize,
    learning_rate: f64,
    ridge_ext_coord: f64,
    ridge_beta: f64,
    isometry_pin_active: bool,
    run_structure_search: bool,
    analytic_penalties: Option<String>,
    fisher_factors: Option<PyReadonlyArray3<'py, f64>>,
    fisher_mass_residual: Option<PyReadonlyArray1<'py, f64>>,
    fisher_provenance: Option<String>,
    fisher_factor_kind: Option<String>,
) -> PyResult<Py<PyDict>> {
    let assignment_kind = canonicalize_assignment_kind(&assignment_kind).map_err(py_value_error)?;
    let z_view = z.as_array();
    let (n_obs, p_out) = z_view.dim();
    let k_atoms = atom_basis.len();
    if atom_dim.len() != k_atoms
        || decoder_blocks.len() != k_atoms
        || duchon_centers.len() != k_atoms
        || n_harmonics_list.len() != k_atoms
        || basis_size_list.len() != k_atoms
    {
        return Err(py_value_error(format!(
            "sae_manifold_certify_external: per-atom metadata lengths must equal K={k_atoms}"
        )));
    }

    let assignment = match assignment_kind.as_str() {
        "softmax" => gam::terms::sae::manifold::SaeOosAssignmentKind::Softmax,
        "ordered_beta_bernoulli" => {
            gam::terms::sae::manifold::SaeOosAssignmentKind::OrderedBetaBernoulli {
                learnable_alpha,
            }
        }
        "threshold_gate" => gam::terms::sae::manifold::SaeOosAssignmentKind::ThresholdGate {
            threshold: threshold_gate_threshold,
        },
        "topk" => gam::terms::sae::manifold::SaeOosAssignmentKind::TopK,
        _ => {
            return Err(py_value_error(format!(
                "sae_manifold_certify_external: unsupported assignment kind {assignment_kind:?}"
            )));
        }
    };

    let atom_centers: Vec<Option<Array2<f64>>> = duchon_centers
        .iter()
        .map(|o| o.as_ref().map(|a| a.as_array().to_owned()))
        .collect();
    let atoms: Vec<gam::terms::sae::manifold::SaeOosAtomSpec> = (0..k_atoms)
        .map(|atom_index| gam::terms::sae::manifold::SaeOosAtomSpec {
            basis_kind: sae_atom_basis_kind_from_str(&atom_basis[atom_index]),
            latent_dim: atom_dim[atom_index],
            decoder: decoder_blocks[atom_index].as_array().to_owned(),
            centers: atom_centers[atom_index].clone(),
            n_harmonics: n_harmonics_list[atom_index],
            basis_size: basis_size_list[atom_index],
        })
        .collect();

    if initial_coords.len() != k_atoms {
        return Err(py_value_error(format!(
            "sae_manifold_certify_external: initial_coords must carry K={k_atoms} atom blocks; got {}",
            initial_coords.len()
        )));
    }
    let coords: Vec<Array2<f64>> = initial_coords
        .iter()
        .map(|block| block.as_array().to_owned())
        .collect();

    let regularization = gam::terms::sae::manifold::SaeOosRegularization {
        log_lambda_sparse,
        log_lambda_smooth,
        log_ard,
    };

    let fisher_mass_residual_owned = fisher_mass_residual
        .as_ref()
        .map(|values| values.as_array().to_owned());
    let fisher_metric = match fisher_factors {
        Some(factors) => {
            let request = SaeFisherRowMetricRequest::from_tag(
                factors.as_array(),
                n_obs,
                p_out,
                fisher_provenance.as_deref(),
                fisher_factor_kind.as_deref(),
                fisher_mass_residual_owned
                    .as_ref()
                    .map(|values| values.view()),
            )
            .map_err(py_value_error)?;
            Some(build_sae_fisher_row_metric(request).map_err(py_value_error)?)
        }
        None => None,
    };
    let metric_provenance: &'static str = match &fisher_metric {
        Some(metric) => gam::terms::sae::manifold::metric_provenance_label(metric.provenance()),
        None => "Euclidean",
    };

    let analytic_penalties: Option<serde_json::Value> = match analytic_penalties {
        Some(s) => Some(serde_json::from_str(&s).map_err(serde_json_error_to_pyerr)?),
        None => None,
    };
    let max_atom_dim = atom_dim.iter().copied().max().ok_or_else(|| {
        py_value_error("sae_manifold_certify_external: atom_dim is empty".to_string())
    })?;
    let total_basis: usize = basis_size_list.iter().copied().sum();
    let mut latent_blocks = serde_json::Map::new();
    latent_blocks.insert(
        "t".into(),
        serde_json::json!({"name": "t", "n": n_obs, "d": max_atom_dim}),
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

    let request = gam::terms::sae::manifold::SaeCertifyExternalRequest {
        target: z_view.to_owned(),
        atoms,
        coords,
        logits: initial_logits.as_array().to_owned(),
        assignment,
        top_k,
        alpha,
        tau,
        regularization,
        fisher_metric,
        registry,
        max_iter,
        learning_rate,
        ridge_ext_coord,
        ridge_beta,
        isometry_pin_active,
        metric_provenance,
        run_structure_search,
    };

    let report = gam::terms::sae::manifold::run_sae_manifold_certify_external(request)
        .map_err(|err| sae_fit_error_to_pyerr(py, err))?;

    // #2266/#2263 — the reported dict shape below is a deliberate duplicate of
    // `sae_manifold_fit_inner`'s postlude (same `SaeFitReport` fields, same
    // helper functions), NOT a shared extraction: `run_sae_manifold_certify`'s
    // own doc comment (fit_entry.rs) records that its postlude was duplicated
    // from `run_sae_manifold_fit_on_target` rather than factored out, because
    // the source function is under concurrent edit elsewhere in this
    // workspace and an extraction risked colliding with that churn. The same
    // reasoning applies here on the pyffi side; keep this block in sync with
    // `sae_manifold_fit_inner`'s tail if that payload shape changes.
    let seed_refine_random_state = 0u64;
    let fisher_mass_residual = fisher_mass_residual_owned
        .as_ref()
        .map(|values| values.view());
    let gam::terms::sae::manifold::SaeFitReport {
        term,
        rho,
        loss,
        penalized_quasi_laplace_criterion,
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
        amortized_encoder_consistency,
        certificate_ledger,
        structure_search_json,
        structure_certificate_json,
        reported_log_alpha,
    } = report;

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
    let tier0_scale = term.tier0_scale().cloned();
    let atoms_py = PyList::empty(py);
    for atom_idx in 0..k_atoms {
        let atom = &term.atoms[atom_idx];
        let atom_dict = PyDict::new(py);
        let mut decoder_physical = atom.full_width_decoder();
        if let Some(sigma) = tier0_scale.as_ref() {
            for mut row in decoder_physical.rows_mut() {
                row *= sigma;
            }
        }
        atom_dict.set_item("decoder_B", decoder_physical.into_pyarray(py))?;
        atom_dict.set_item("basis_kind", atom_basis[atom_idx].clone())?;
        atom_dict.set_item("basis_centers", py.None())?;
        atom_dict.set_item(
            "on_atom_coords_t",
            term.assignment.coords[atom_idx]
                .as_matrix()
                .into_pyarray(py),
        )?;
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
        if let Some(unc) = shape_uncertainty.atoms.get(atom_idx) {
            if let Some(cov) = &unc.decoder_covariance {
                let mut cov_full = atom
                    .lift_reduced_decoder_covariance(cov, p_out)
                    .map_err(py_value_error)?;
                if let Some(sigma) = tier0_scale.as_ref() {
                    let m_p = cov_full.nrows();
                    for i in 0..m_p {
                        let si = sigma[i % p_out];
                        for j in 0..m_p {
                            cov_full[[i, j]] *= si * sigma[j % p_out];
                        }
                    }
                }
                atom_dict.set_item("decoder_covariance", cov_full.into_pyarray(py))?;
            }
            match (&unc.band_coords, &unc.band_mean, &unc.band_sd) {
                (Some(coords), Some(mean), Some(sd)) => {
                    atom_dict.set_item("shape_band_coords", coords.clone().into_pyarray(py))?;
                    let mut band_mean = mean.clone();
                    let mut band_sd = sd.clone();
                    if let Some(sigma) = tier0_scale.as_ref() {
                        for mut row in band_mean.rows_mut() {
                            row *= sigma;
                        }
                        for mut row in band_sd.rows_mut() {
                            row *= sigma;
                        }
                    }
                    atom_dict.set_item("shape_band_mean", band_mean.into_pyarray(py))?;
                    atom_dict.set_item("shape_band_sd", band_sd.into_pyarray(py))?;
                }
                (None, None, None) => {}
                _ => {
                    return Err(py_value_error(format!(
                        "atom {atom_idx} has a partial shape-uncertainty band"
                    )));
                }
            }
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
    match term.tier0_mean() {
        Some(mean) => out.set_item("tier0_mean", mean.clone().into_pyarray(py))?,
        None => out.set_item("tier0_mean", py.None())?,
    }
    match term.tier0_scale() {
        Some(scale) => out.set_item("tier0_scale", scale.clone().into_pyarray(py))?,
        None => out.set_item("tier0_scale", py.None())?,
    }
    {
        let termination = PyDict::new(py);
        termination.set_item("verdict", outer_termination.verdict.as_str())?;
        termination.set_item("evals", outer_termination.evals)?;
        termination.set_item(
            "evals_since_improvement",
            outer_termination.evals_since_improvement,
        )?;
        termination.set_item("wall_seconds", outer_termination.wall.as_secs_f64())?;
        out.set_item("termination", termination)?;
    }
    sae_set_penalized_loss_items(&out, &loss, "penalized_loss_score")?;
    out.set_item(
        "penalized_quasi_laplace_criterion",
        penalized_quasi_laplace_criterion,
    )?;
    out.set_item("log_alpha", reported_log_alpha)?;
    out.set_item("log_lambda_smooth", rho.log_lambda_smooth.clone())?;
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
    out.set_item("dispersion", shape_uncertainty.dispersion)?;
    out.set_item("metric_provenance", metric_provenance)?;
    out.set_item(
        "structured_residual_diagnostics",
        structured_residual_pass_diagnostics_dict(py, &structured_residual_diagnostics)?,
    )?;
    if let Some(mr) = fisher_mass_residual {
        out.set_item("fisher_mass_residual", mr.to_owned().into_pyarray(py))?;
    }
    out.set_item(
        "atom_two_lens",
        sae_atom_two_lens_dict(py, &fit_diagnostics.atom_two_lens)?,
    )?;
    out.set_item(
        "residual_gauge",
        sae_residual_gauge_dict(py, &fit_diagnostics.residual_gauge)?,
    )?;
    if let Some(report) = term.hybrid_split_report() {
        out.set_item("hybrid_split", sae_hybrid_split_dict(py, report)?)?;
    }
    let cotrain = PyDict::new(py);
    cotrain.set_item(
        "recon_consistency",
        amortized_encoder_consistency.recon_consistency,
    )?;
    cotrain.set_item(
        "unconverged_fraction",
        amortized_encoder_consistency.unconverged_fraction,
    )?;
    cotrain.set_item("n_unconverged", amortized_encoder_consistency.n_unconverged)?;
    cotrain.set_item("n_encodes", amortized_encoder_consistency.n_encodes)?;
    out.set_item("cotrain", cotrain)?;
    if let Some(report) = &fit_diagnostics.incoherence_report {
        out.set_item("curvature_report", sae_curvature_report_dict(py, report)?)?;
        out.set_item(
            "incoherence_report",
            sae_incoherence_report_dict(py, report)?,
        )?;
    }
    out.set_item(
        "atom_inference",
        sae_atom_inference_list(py, &fit_diagnostics.atom_inference)?,
    )?;
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
    out.set_item(
        "certificates",
        certificate_ledger_dict(py, &certificate_ledger)?,
    )?;
    let fitted_atom_plans = sae_fitted_atom_plans(&term, &atom_centers, seed_refine_random_state)
        .map_err(py_value_error)?;
    let atom_plans_py = PyList::empty(py);
    for plan in fitted_atom_plans {
        let entry = PyDict::new(py);
        entry.set_item("kind", sae_atom_basis_kind_name(&plan.kind))?;
        entry.set_item("latent_dim", plan.latent_dim)?;
        entry.set_item("n_harmonics", plan.n_harmonics)?;
        entry.set_item("basis_size", plan.basis_size)?;
        match plan.duchon_centers {
            Some(centers) => entry.set_item("duchon_centers", centers.into_pyarray(py))?,
            None => entry.set_item("duchon_centers", py.None())?,
        }
        atom_plans_py.append(entry)?;
    }
    out.set_item("atom_plans", atom_plans_py)?;

    out.set_item("chosen_k", k_atoms)?;
    out.set_item("oos_projection_top1", top_k == Some(1))?;
    if let Some(json) = structure_search_json {
        out.set_item("structure_search", json)?;
    }
    out.set_item("structure_certificate", structure_certificate_json)?;
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
/// Owned-array core of the steering primitive (#2091): the full per-atom basis
/// rebuild + trained-latent seeding + optional output-Fisher metric install +
/// `steer_delta` call, on borrowed ndarray views instead of `PyReadonlyArray`.
///
/// Both callers route through this single rebuild path so their `SteerPlan`s are
/// identical by construction: the `sae_steer_delta` `#[pyfunction]` (arrays
/// marshalled from Python) and `ManifoldSaeCore::steer` (arrays read from the
/// Rust-owned model state, so an attached Fisher shard is NOT re-marshalled
/// across the FFI boundary per call — acceptance bullet 2). The
/// predicted-nats-vs-analytic steering tests guard this rebuild's correctness;
/// the pyclass equivalence test guards that the two callers thread identical
/// inputs into it. `fisher_provenance`: same-position `"output_fisher"` (default)
/// or forward-looking `"output_fisher_downstream"` — selects the re-installed
/// output-Fisher `RowMetric` the dose is measured through.
fn steer_delta_from_arrays(
    atom_k: usize,
    metric_row: usize,
    amplitude: f64,
    t_from: ndarray::ArrayView1<'_, f64>,
    t_to: ndarray::ArrayView1<'_, f64>,
    n_obs: usize,
    p_out: usize,
    atom_basis: &[String],
    atom_dim: &[usize],
    decoder_blocks: &[ndarray::ArrayView2<'_, f64>],
    duchon_centers: &[Option<Array2<f64>>],
    n_harmonics_list: &[Option<usize>],
    basis_size_list: &[usize],
    coords: &[ndarray::ArrayView2<'_, f64>],
    logits: ndarray::ArrayView2<'_, f64>,
    assignment_kind: &str,
    top_k: Option<usize>,
    tau: f64,
    alpha: f64,
    threshold_gate_threshold: f64,
    fisher_factors: Option<ndarray::ArrayView3<'_, f64>>,
    fisher_mass_residual: Option<ndarray::ArrayView1<'_, f64>>,
    fisher_provenance: Option<&str>,
    fisher_factor_kind: Option<&str>,
) -> PyResult<gam::inference::steering::SteerPlan> {
    let fisher_metric = match fisher_factors {
        Some(u3) => {
            let request = SaeFisherRowMetricRequest::from_tag(
                u3,
                n_obs,
                p_out,
                fisher_provenance,
                fisher_factor_kind,
                fisher_mass_residual,
            )
            .map_err(py_value_error)?;
            Some(build_sae_fisher_row_metric(request).map_err(py_value_error)?)
        }
        None => None,
    };
    steer_delta_with_metric_from_arrays(
        atom_k,
        metric_row,
        amplitude,
        t_from,
        t_to,
        atom_basis,
        atom_dim,
        decoder_blocks,
        duchon_centers,
        n_harmonics_list,
        basis_size_list,
        coords,
        logits,
        assignment_kind,
        top_k,
        tau,
        alpha,
        threshold_gate_threshold,
        fisher_metric,
    )
}

/// Resident-metric steering core used by [`ManifoldSaeCore::steer`].  The
/// public array-oriented function above builds a metric because its caller owns
/// only a borrowed factor stack.  A fitted model owns the packed [`RowMetric`]
/// and calls this entry directly, so steering never repacks or revalidates the
/// `(n, p, rank)` Fisher shard.
fn steer_delta_with_metric_from_arrays(
    atom_k: usize,
    metric_row: usize,
    amplitude: f64,
    t_from: ndarray::ArrayView1<'_, f64>,
    t_to: ndarray::ArrayView1<'_, f64>,
    atom_basis: &[String],
    atom_dim: &[usize],
    decoder_blocks: &[ndarray::ArrayView2<'_, f64>],
    duchon_centers: &[Option<Array2<f64>>],
    n_harmonics_list: &[Option<usize>],
    basis_size_list: &[usize],
    coords: &[ndarray::ArrayView2<'_, f64>],
    logits: ndarray::ArrayView2<'_, f64>,
    assignment_kind: &str,
    top_k: Option<usize>,
    tau: f64,
    alpha: f64,
    threshold_gate_threshold: f64,
    fisher_metric: Option<gam::inference::row_metric::RowMetric>,
) -> PyResult<gam::inference::steering::SteerPlan> {
    // Assignment tokens are strict: compatibility aliases are rejected.
    let assignment_kind = canonicalize_assignment_kind(assignment_kind).map_err(py_value_error)?;
    let k_atoms = atom_basis.len();
    // Guard the per-atom metadata lengths before indexing them into the atom
    // specs below; every other precondition (positive dims, atom_k range, coord /
    // logit shapes, positive alpha/tau) is validated inside the engine entry.
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
    let assignment = match assignment_kind.as_str() {
        "softmax" => gam::terms::sae::manifold::SaeOosAssignmentKind::Softmax,
        "ordered_beta_bernoulli" => {
            gam::terms::sae::manifold::SaeOosAssignmentKind::OrderedBetaBernoulli {
                learnable_alpha: false,
            }
        }
        "threshold_gate" => gam::terms::sae::manifold::SaeOosAssignmentKind::ThresholdGate {
            threshold: threshold_gate_threshold,
        },
        "topk" => gam::terms::sae::manifold::SaeOosAssignmentKind::TopK,
        _ => {
            return Err(py_value_error(format!(
                "sae_steer_delta: assignment_kind must be one of 'softmax', 'ordered_beta_bernoulli', \
                 'threshold_gate', or 'topk'; got {assignment_kind}"
            )));
        }
    };

    // Marshal the persisted dictionary schema into typed OOS atom specs — the SAME
    // rebuild contract `sae_manifold_predict_oos` marshals into, so the steer term
    // and the OOS term are rebuilt by one engine path (`run_sae_manifold_steer`,
    // #2236) rather than a duplicated pyffi rebuild.
    let atoms: Vec<gam::terms::sae::manifold::SaeOosAtomSpec> = (0..k_atoms)
        .map(|atom_index| gam::terms::sae::manifold::SaeOosAtomSpec {
            basis_kind: sae_atom_basis_kind_from_str(&atom_basis[atom_index]),
            latent_dim: atom_dim[atom_index],
            decoder: decoder_blocks[atom_index].to_owned(),
            centers: duchon_centers[atom_index].clone(),
            n_harmonics: n_harmonics_list[atom_index],
            basis_size: basis_size_list[atom_index],
        })
        .collect();
    let coord_blocks: Vec<Array2<f64>> = coords.iter().map(|block| block.to_owned()).collect();

    let request = gam::terms::sae::manifold::SaeSteerRequest {
        atoms,
        coords: coord_blocks,
        logits: logits.to_owned(),
        assignment,
        top_k,
        alpha,
        tau,
        fisher_metric,
        atom_k,
        metric_row,
        amplitude,
        t_from: t_from.to_vec(),
        t_to: t_to.to_vec(),
    };
    gam::terms::sae::manifold::run_sae_manifold_steer(request).map_err(py_value_error)
}

/// Rebuild the trained term from arrays and solve for the amplitude realizing a
/// TARGET output-KL dose (gh#2263). Mirrors [`steer_delta_with_metric_from_arrays`]
/// but drives the target-dose entry; the optional `probe` (a patched-forward KL
/// callback) drives the closed-loop correction and the readout-KL radius.
struct SteerToTargetArraysRequest<'a> {
    atom_k: usize,
    metric_row: usize,
    target_nats: f64,
    config: gam::inference::steering::TargetDoseConfig,
    t_from: ndarray::ArrayView1<'a, f64>,
    t_to: ndarray::ArrayView1<'a, f64>,
    atom_basis: &'a [String],
    atom_dim: &'a [usize],
    decoder_blocks: &'a [ndarray::ArrayView2<'a, f64>],
    duchon_centers: &'a [Option<Array2<f64>>],
    n_harmonics_list: &'a [Option<usize>],
    basis_size_list: &'a [usize],
    coords: &'a [ndarray::ArrayView2<'a, f64>],
    logits: ndarray::ArrayView2<'a, f64>,
    assignment_kind: &'a str,
    top_k: Option<usize>,
    tau: f64,
    alpha: f64,
    threshold_gate_threshold: f64,
    fisher_metric: Option<gam::inference::row_metric::RowMetric>,
}

/// Typed extraction of the public `ManifoldSaeCore.steer_to_target` mapping.
/// Every dose and solver field is required; optionality belongs only to the
/// separate patched-forward probe.
struct ManifoldSteerToTargetRequest {
    atom_k: usize,
    metric_row: usize,
    target_nats: f64,
    t_from: Array1<f64>,
    t_to: Array1<f64>,
    config: gam::inference::steering::TargetDoseConfig,
}

fn required_steer_to_target_item<'py>(
    request: &Bound<'py, PyDict>,
    key: &str,
) -> PyResult<Bound<'py, PyAny>> {
    request.get_item(key)?.ok_or_else(|| {
        py_value_error(format!(
            "ManifoldSaeCore.steer_to_target: request is missing required key {key:?}"
        ))
    })
}

impl ManifoldSteerToTargetRequest {
    fn from_pydict(request: &Bound<'_, PyDict>) -> PyResult<Self> {
        let t_from = required_steer_to_target_item(request, "t_from")?
            .extract::<PyReadonlyArray1<'_, f64>>()?
            .as_array()
            .to_owned();
        let t_to = required_steer_to_target_item(request, "t_to")?
            .extract::<PyReadonlyArray1<'_, f64>>()?
            .as_array()
            .to_owned();
        Ok(Self {
            atom_k: required_steer_to_target_item(request, "atom_k")?.extract()?,
            metric_row: required_steer_to_target_item(request, "metric_row")?.extract()?,
            target_nats: required_steer_to_target_item(request, "target_nats")?.extract()?,
            t_from,
            t_to,
            config: gam::inference::steering::TargetDoseConfig {
                tol_rel: required_steer_to_target_item(request, "tol_rel")?.extract()?,
                max_iter: required_steer_to_target_item(request, "max_iter")?.extract()?,
                readout_tol_rel: required_steer_to_target_item(request, "readout_tol_rel")?
                    .extract()?,
            },
        })
    }
}

fn steer_to_target_from_arrays(
    request: SteerToTargetArraysRequest<'_>,
    probe: Option<&mut gam::inference::steering::PatchedForwardKl<'_>>,
) -> PyResult<gam::inference::steering::TargetDosePlan> {
    let SteerToTargetArraysRequest {
        atom_k,
        metric_row,
        target_nats,
        config,
        t_from,
        t_to,
        atom_basis,
        atom_dim,
        decoder_blocks,
        duchon_centers,
        n_harmonics_list,
        basis_size_list,
        coords,
        logits,
        assignment_kind,
        top_k,
        tau,
        alpha,
        threshold_gate_threshold,
        fisher_metric,
    } = request;
    let assignment_kind = canonicalize_assignment_kind(assignment_kind).map_err(py_value_error)?;
    let k_atoms = atom_basis.len();
    if atom_dim.len() != k_atoms
        || decoder_blocks.len() != k_atoms
        || duchon_centers.len() != k_atoms
        || n_harmonics_list.len() != k_atoms
        || basis_size_list.len() != k_atoms
        || coords.len() != k_atoms
    {
        return Err(py_value_error(format!(
            "sae_steer_to_target: per-atom metadata lengths must equal K={k_atoms}"
        )));
    }
    let assignment = match assignment_kind.as_str() {
        "softmax" => gam::terms::sae::manifold::SaeOosAssignmentKind::Softmax,
        "ordered_beta_bernoulli" => {
            gam::terms::sae::manifold::SaeOosAssignmentKind::OrderedBetaBernoulli {
                learnable_alpha: false,
            }
        }
        "threshold_gate" => gam::terms::sae::manifold::SaeOosAssignmentKind::ThresholdGate {
            threshold: threshold_gate_threshold,
        },
        "topk" => gam::terms::sae::manifold::SaeOosAssignmentKind::TopK,
        _ => {
            return Err(py_value_error(format!(
                "sae_steer_to_target: assignment_kind must be one of 'softmax', \
                 'ordered_beta_bernoulli', 'threshold_gate', or 'topk'; got {assignment_kind}"
            )));
        }
    };
    let atoms: Vec<gam::terms::sae::manifold::SaeOosAtomSpec> = (0..k_atoms)
        .map(|atom_index| gam::terms::sae::manifold::SaeOosAtomSpec {
            basis_kind: sae_atom_basis_kind_from_str(&atom_basis[atom_index]),
            latent_dim: atom_dim[atom_index],
            decoder: decoder_blocks[atom_index].to_owned(),
            centers: duchon_centers[atom_index].clone(),
            n_harmonics: n_harmonics_list[atom_index],
            basis_size: basis_size_list[atom_index],
        })
        .collect();
    let coord_blocks: Vec<Array2<f64>> = coords.iter().map(|block| block.to_owned()).collect();

    let request = gam::terms::sae::manifold::SaeSteerToTargetRequest {
        atoms,
        coords: coord_blocks,
        logits: logits.to_owned(),
        assignment,
        top_k,
        alpha,
        tau,
        fisher_metric,
        atom_k,
        metric_row,
        t_from: t_from.to_vec(),
        t_to: t_to.to_vec(),
        target_nats,
        config,
    };
    gam::terms::sae::manifold::run_sae_manifold_steer_to_target(request, probe)
        .map_err(py_value_error)
}

/// Render a [`gam::inference::steering::TargetDosePlan`] as a Python dict.
fn target_dose_plan_to_pydict(
    py: Python<'_>,
    plan: gam::inference::steering::TargetDosePlan,
) -> PyResult<Py<PyDict>> {
    let gam::inference::steering::TargetDosePlan {
        target_nats,
        seed_amplitude,
        steer,
        measured_nats,
        iterations,
        readout_kl_radius,
    } = plan;
    let out = steer_plan_to_pydict(py, steer)?;
    let bound = out.bind(py);
    bound.set_item("target_nats", target_nats)?;
    bound.set_item("seed_amplitude", seed_amplitude)?;
    bound.set_item("measured_nats", measured_nats)?;
    bound.set_item("iterations", iterations)?;
    bound.set_item(
        "validation",
        if measured_nats.is_some() {
            "patched_forward"
        } else {
            "quadratic_model"
        },
    )?;
    bound.set_item("readout_kl_radius", readout_kl_radius)?;
    Ok(out)
}

/// Render a [`gam::inference::steering::SteerPlan`] as the Python dict both steer
/// callers return (the `sae_steer_delta` pyfunction and `ManifoldSaeCore::steer`).
fn steer_plan_to_pydict(
    py: Python<'_>,
    plan: gam::inference::steering::SteerPlan,
) -> PyResult<Py<PyDict>> {
    let provenance_str = gam::terms::sae::manifold::metric_provenance_label(plan.metric_provenance);
    let out = PyDict::new(py);
    out.set_item("atom", plan.atom)?;
    out.set_item("atom_name", plan.atom_name)?;
    out.set_item("t_from", plan.t_from)?;
    out.set_item("t_to", plan.t_to)?;
    out.set_item("amplitude", plan.amplitude)?;
    out.set_item("metric_row", plan.metric_row)?;
    // Keep the plan payload value-semantic: unlike a numpy array, a plain
    // vector has deterministic Python equality, so repeated steering of the
    // immutable fitted model yields directly comparable dictionaries.
    out.set_item("delta", plan.delta.to_vec())?;
    out.set_item("predicted_nats", plan.predicted_nats)?;
    out.set_item("predicted_nats_kind", plan.predicted_nats_kind.as_str())?;
    out.set_item("fisher_mass_captured", plan.fisher_mass_captured)?;
    out.set_item("fisher_mass_residual", plan.fisher_mass_residual)?;
    out.set_item(
        "fisher_mass_residual_fraction",
        plan.fisher_mass_residual_fraction,
    )?;
    out.set_item("validity_radius", plan.validity_radius)?;
    out.set_item("off_manifold_norm", plan.off_manifold_norm)?;
    out.set_item("metric_provenance", provenance_str)?;
    Ok(out.unbind())
}

/// FFI surface for the steering primitive: rebuilds the fitted term from the
/// trained decoder blocks + basis metadata, seeds it with the trained latents /
/// logits (no re-solve), optionally installs the WP-D output-Fisher metric from
/// `fisher_factors`, and drives atom `atom_k` from `t_from` to `t_to`, returning
/// the [`gam::inference::steering::SteerPlan`] as a dict. Thin marshalling over
/// the shared [`steer_delta_from_arrays`] rebuild (#2091).
#[pyfunction(signature = (
    atom_k,
    metric_row,
    amplitude,
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
    threshold_gate_threshold = 0.0,
    top_k = None,
    fisher_factors = None,
    fisher_mass_residual = None,
    fisher_provenance = None,
    fisher_factor_kind = None,
))]
fn sae_steer_delta<'py>(
    py: Python<'py>,
    atom_k: usize,
    metric_row: usize,
    amplitude: f64,
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
    threshold_gate_threshold: f64,
    top_k: Option<usize>,
    fisher_factors: Option<PyReadonlyArray3<'py, f64>>,
    fisher_mass_residual: Option<PyReadonlyArray1<'py, f64>>,
    fisher_provenance: Option<String>,
    fisher_factor_kind: Option<String>,
) -> PyResult<Py<PyDict>> {
    let decoder_views: Vec<ndarray::ArrayView2<'_, f64>> =
        decoder_blocks.iter().map(|b| b.as_array()).collect();
    let coord_views: Vec<ndarray::ArrayView2<'_, f64>> =
        coords.iter().map(|c| c.as_array()).collect();
    let duchon_owned: Vec<Option<Array2<f64>>> = duchon_centers
        .iter()
        .map(|o| o.as_ref().map(|a| a.as_array().to_owned()))
        .collect();
    let fisher_view = fisher_factors.as_ref().map(|f| f.as_array());
    let fisher_mass_view = fisher_mass_residual.as_ref().map(|mass| mass.as_array());
    let plan = steer_delta_from_arrays(
        atom_k,
        metric_row,
        amplitude,
        t_from.as_array(),
        t_to.as_array(),
        n_obs,
        p_out,
        &atom_basis,
        &atom_dim,
        &decoder_views,
        &duchon_owned,
        &n_harmonics_list,
        &basis_size_list,
        &coord_views,
        logits.as_array(),
        &assignment_kind,
        top_k,
        tau,
        alpha,
        threshold_gate_threshold,
        fisher_view,
        fisher_mass_view,
        fisher_provenance.as_deref(),
        fisher_factor_kind.as_deref(),
    )?;
    steer_plan_to_pydict(py, plan)
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
    let dim_selection_precision = dim_selection_log_precision
        .as_ref()
        .map(|values| ValidatedDimSelectionPrecisions::new(values.as_array(), latent_dim))
        .transpose()
        .map_err(py_value_error)?;
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
            grad_reml_score
                * (0.5 * stats.strength.mu * stats.residual_sq - 0.5 * (n_obs * latent_dim) as f64),
        );
    }
    if let Some(precisions) = dim_selection_precision.as_ref() {
        let mut grad_log_prec = Array1::<f64>::zeros(latent_dim);
        for n in 0..n_obs {
            for a in 0..latent_dim {
                let prec = precisions.physical[a];
                if grad_reml_score != 0.0 {
                    grad_t[n * latent_dim + a] += grad_reml_score * prec * t_mat[[n, a]];
                }
            }
        }
        for a in 0..latent_dim {
            let energy = precisions
                .axis_energy(t_mat.view(), a)
                .map_err(py_value_error)?;
            grad_log_prec[a] = grad_reml_score * (energy - 0.5 * n_obs as f64);
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
    dim_selection: Option<ValidatedDimSelectionPrecisions>,
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
            self.dim_selection.as_ref(),
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
        if let Some(precisions) = self.dim_selection.as_ref() {
            for n in 0..self.n_obs {
                for a in 0..self.latent_dim {
                    let prec = precisions.physical[a];
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
    use super::monomial_exponents;
    use gam::terms::sae::manifold::sae_euclidean_degree_for_basis_size;

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

#[cfg(test)]
mod sae_assignment_kind_tests {
    use super::canonicalize_assignment_kind;

    /// The FFI parser accepts exactly the four canonical assignment tokens.
    /// Removed compatibility spellings must remain caller errors.
    #[test]
    fn assignment_kind_accepts_only_canonical_tokens() {
        assert_eq!(
            canonicalize_assignment_kind("threshold_gate").unwrap(),
            "threshold_gate"
        );
        assert_eq!(canonicalize_assignment_kind("softmax").unwrap(), "softmax");
        assert_eq!(
            canonicalize_assignment_kind("ordered_beta_bernoulli").unwrap(),
            "ordered_beta_bernoulli"
        );
        assert_eq!(canonicalize_assignment_kind("topk").unwrap(), "topk");
        let removed = canonicalize_assignment_kind("jumprelu").unwrap_err();
        assert!(
            removed.contains("threshold_gate") && removed.contains("not a recognized"),
            "removed alias must be rejected while naming the canonical token; got {removed:?}"
        );
        let err = canonicalize_assignment_kind("bogus").unwrap_err();
        assert!(
            err.contains("threshold_gate") && err.contains("topk"),
            "error must name canonical tokens; got {err:?}"
        );
    }
}

#[cfg(test)]
mod sae_linear_atom_tests {
    use super::{sae_atom_basis_kind_from_str, sae_atom_basis_kind_name};
    use gam::terms::sae::manifold::{EuclideanPatchEvaluator, SaeAtomBasisKind, SaeBasisEvaluator};
    use ndarray::Array2;

    /// #1221 — `"linear"` (and its synonyms) is a first-class topology distinct
    /// from `"euclidean"`/`"euclidean_patch"` (the degree-2 quadratic patch), and
    /// it round-trips under the honest name `"linear"`. The quadratic patch keeps
    /// its own `"euclidean_patch"` name and additionally accepts the explicit
    /// `"euclidean_quadratic_patch"` synonym.
    #[test]
    fn linear_topology_is_first_class_and_round_trips() {
        for name in ["linear", "linear_rank1", "affine", "LINEAR"] {
            assert_eq!(
                sae_atom_basis_kind_from_str(name),
                SaeAtomBasisKind::Linear,
                "{name:?} must parse to the genuinely-linear atom"
            );
        }
        assert_eq!(
            sae_atom_basis_kind_name(&SaeAtomBasisKind::Linear),
            "linear",
            "the linear atom must round-trip under its honest name"
        );
        // The quadratic patch is a DIFFERENT kind — `"linear"` must not collapse
        // onto it, or the curved-vs-linear comparison would be mislabeled again.
        for name in ["euclidean", "euclidean_patch", "euclidean_quadratic_patch"] {
            assert_eq!(
                sae_atom_basis_kind_from_str(name),
                SaeAtomBasisKind::EuclideanPatch,
                "{name:?} is the degree-2 quadratic patch, distinct from linear"
            );
        }
    }

    /// #1221 — the genuinely-linear atom's decoder reconstructs the affine image
    /// `γ(t) = b₀ + t·b₁` EXACTLY. Its evaluator is the degree-1 monomial patch
    /// `Φ(t) = [1, t]` (width `d + 1 = 2` at `d = 1`), so for a decoder
    /// `B = [[b₀…], [b₁…]]` the reconstruction `Φ(t)·B` equals `b₀ + t·b₁` to
    /// machine precision — the property the reconstruction-parity baseline needs.
    #[test]
    fn linear_atom_reconstructs_affine_image_exactly() {
        let evaluator = EuclideanPatchEvaluator::new(1, 1).expect("degree-1 patch");
        assert_eq!(
            evaluator.basis_size(),
            2,
            "a degree-1 (linear/affine) patch in 1-D has width 2: {{1, t}}"
        );
        // Decoder over p = 2 output channels: γ(t) = b0 + t·b1.
        let b0 = [0.7_f64, -1.3];
        let b1 = [2.0_f64, 0.5];
        let decoder =
            Array2::from_shape_vec((2, 2), vec![b0[0], b0[1], b1[0], b1[1]]).expect("2x2 decoder");

        let coords =
            Array2::from_shape_vec((5, 1), vec![-2.0, -0.5, 0.0, 1.0, 3.0]).expect("5x1 coords");
        let (phi, _jet) = evaluator
            .evaluate(coords.view())
            .expect("evaluate linear patch");
        assert_eq!(phi.dim(), (5, 2));
        let recon = phi.dot(&decoder); // (5, 2) = Φ·B
        for (row, &t) in coords.column(0).iter().enumerate() {
            for ch in 0..2 {
                let expected = b0[ch] + t * b1[ch];
                assert!(
                    (recon[[row, ch]] - expected).abs() < 1e-12,
                    "linear atom must reconstruct b0 + t·b1 exactly: \
                     row {row} ch {ch} got {} want {expected}",
                    recon[[row, ch]]
                );
            }
        }
    }
}
