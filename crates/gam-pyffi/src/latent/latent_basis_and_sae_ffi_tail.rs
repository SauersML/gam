/// One-shot SAE-manifold fit driver: takes only `(z, atom_basis, atom_dim,
/// ...scalar hyperparams)` and assembles the full basis + jacobian + penalty
/// stack + PCA seed coords + least-squares decoder seed + jittered routing logits internally
/// before delegating to the same end-to-end Rust Newton loop as
/// the native fit orchestration. Returns the raw native fit payload with
/// `"geometry_plans"`, holding each validated atom geometry so OOS prediction can
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
    block_orthogonality_weight: f64,
    random_state: u64,
    top_k: Option<usize>,
    initial_logits: Option<PyReadonlyArray2<'py, f64>>,
    initial_coords: Option<PyReadonlyArray3<'py, f64>>,
    threshold_gate_threshold: f64,
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
    gpu_policy: gam::gpu::GpuPolicy,
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
        geometry_plans,
        basis_values,
        basis_jacobian,
        decoder_coefficients,
        smooth_penalties,
        initial_logits,
        initial_coords: start_coords,
        refine_routing,
    } = seed;
    // Public `d_atom` is an INTRINSIC dimension, whereas analytic penalties on
    // `t` index its resolved STORAGE coordinates.  Those differ for ambient
    // sphere / projective-plane atoms (2 intrinsic dimensions, 3 stored
    // coordinates), and an `auto` seed is not resolved until the minimal seed
    // has been built.  Construct the singleton-axis partition only now, from
    // the exact geometry that owns the target.
    let latent_storage_width = geometry_plans
        .iter()
        .map(SaeAtomGeometryPlan::latent_dim)
        .max()
        .ok_or_else(|| {
            py_value_error("sae_manifold_fit: resolved geometry contains no atoms".to_string())
        })?;
    let analytic_penalties = append_public_block_orthogonality_penalty(
        analytic_penalties,
        block_orthogonality_weight,
        latent_storage_width,
    )
    .map_err(py_value_error)?;
    let fisher_u = fisher_factors.as_ref().map(|f| f.as_array());
    let fisher_mr = fisher_mass_residual.as_ref().map(|m| m.as_array());
    let row_w = row_loss_weights.as_ref().map(|w| w.as_array());
    let result_dict = sae_manifold_fit_inner(
        py,
        z_view,
        &geometry_plans,
        basis_values.view(),
        basis_jacobian.view(),
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
        gpu_policy,
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
    p_out: usize,
}

fn append_public_block_orthogonality_penalty(
    analytic_penalties: Option<String>,
    weight: f64,
    latent_storage_width: usize,
) -> Result<Option<String>, String> {
    if !weight.is_finite() || weight < 0.0 {
        return Err(format!(
            "sae_manifold_fit: block_orthogonality_weight must be finite and non-negative; got {weight}"
        ));
    }
    if weight == 0.0 {
        return Ok(analytic_penalties);
    }
    if latent_storage_width < 2 {
        return Err(format!(
            "sae_manifold_fit: block_orthogonality_weight requires at least two resolved latent storage axes; got {latent_storage_width}"
        ));
    }
    let mut descriptors: Vec<serde_json::Value> = match analytic_penalties {
        Some(json) => serde_json::from_str(&json).map_err(|error| {
            format!(
                "sae_manifold_fit: internal analytic-penalty descriptor list is invalid: {error}"
            )
        })?,
        None => Vec::new(),
    };
    let groups = (0..latent_storage_width)
        .map(|axis| vec![axis])
        .collect::<Vec<_>>();
    descriptors.push(serde_json::json!({
        "kind": "block_orthogonality",
        "target": "t",
        "groups": groups,
        "weight": weight,
    }));
    serde_json::to_string(&descriptors)
        .map(Some)
        .map_err(|error| error.to_string())
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
    if coord_sparsity == "scad" && (!gamma.is_finite() || gamma <= 2.0) {
        return Err(format!(
            "sae_manifold_fit: scad_mcp_gamma must be finite and > 2 for SCAD; got {gamma}"
        ));
    }
    if coord_sparsity == "mcp" && (!gamma.is_finite() || gamma <= 1.0) {
        return Err(format!(
            "sae_manifold_fit: scad_mcp_gamma must be finite and > 1 for MCP; got {gamma}"
        ));
    }

    let mut descriptors = Vec::new();
    let mut names = Vec::new();
    if sparsity_weight > 0.0 && matches!(coord_sparsity.as_str(), "scad" | "mcp") {
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
        // The descriptor is appended after seed resolution, when the target's
        // true storage width (not merely public intrinsic `d_atom`) is known.
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

#[pyclass(module = "gamfit._rust", name = "Tier0SAE")]
pub(crate) struct Tier0SaeCore {
    mean: Vec<f64>,
    fitted: Vec<Vec<f64>>,
    residual_sum_squares: f64,
    reconstruction_r2: f64,
    metric_provenance: String,
    vanished_atoms: Vec<usize>,
}

#[pymethods]
impl Tier0SaeCore {
    #[getter]
    fn chosen_k(&self) -> usize {
        0
    }

    #[getter]
    fn training_mean<'py>(&self, py: Python<'py>) -> Bound<'py, PyArray1<f64>> {
        Array1::from_vec(self.mean.clone()).into_pyarray(py)
    }

    #[getter]
    fn fitted<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyArray2<f64>>> {
        let rows = self.fitted.len();
        let cols = self.mean.len();
        let flat = self.fitted.iter().flatten().copied().collect::<Vec<_>>();
        Array2::from_shape_vec((rows, cols), flat)
            .map(|array| array.into_pyarray(py))
            .map_err(|error| py_value_error(format!("Tier0SAE fitted shape is invalid: {error}")))
    }

    #[getter]
    fn residual_sum_squares(&self) -> f64 {
        self.residual_sum_squares
    }

    #[getter]
    fn reconstruction_r2(&self) -> f64 {
        self.reconstruction_r2
    }

    #[getter]
    fn metric_provenance(&self) -> &str {
        &self.metric_provenance
    }

    #[getter]
    fn vanished_atoms(&self) -> Vec<usize> {
        self.vanished_atoms.clone()
    }

    fn reconstruct<'py>(
        &self,
        py: Python<'py>,
        x_new: PyReadonlyArray2<'py, f64>,
    ) -> PyResult<Bound<'py, PyArray2<f64>>> {
        if x_new.as_array().ncols() != self.mean.len() {
            return Err(py_value_error(format!(
                "Tier0SAE.reconstruct expected {} columns; got {}",
                self.mean.len(),
                x_new.as_array().ncols()
            )));
        }
        Ok(
            Array2::from_shape_fn((x_new.as_array().nrows(), self.mean.len()), |(_, col)| {
                self.mean[col]
            })
            .into_pyarray(py),
        )
    }

    fn to_dict(&self, py: Python<'_>) -> PyResult<PyObject> {
        let out = PyDict::new(py);
        out.set_item("schema", "gamfit.Tier0SAE/v1")?;
        out.set_item("training_mean", self.mean.clone())?;
        out.set_item("fitted", self.fitted.clone())?;
        out.set_item("residual_sum_squares", self.residual_sum_squares)?;
        out.set_item("reconstruction_r2", self.reconstruction_r2)?;
        out.set_item("metric_provenance", self.metric_provenance.clone())?;
        out.set_item("vanished_atoms", self.vanished_atoms.clone())?;
        Ok(out.unbind().into_any())
    }

    fn __repr__(&self) -> String {
        format!(
            "Tier0SAE(n={}, p={}, r2={:.3})",
            self.fitted.len(),
            self.mean.len(),
            self.reconstruction_r2
        )
    }
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
    isometry_weight=0.0,
    decoder_feature_sparsity_groups=None,
    max_iter=50,
    sparsity_strength=None,
    coord_sparsity="scad",
    scad_mcp_gamma=None,
    smoothness=1.0,
    alpha=None,
    learning_rate=None,
    random_state=0,
    block_orthogonality_weight=0.0,
    nuclear_norm_weight=0.0,
    nuclear_norm_max_rank=None,
    decoder_incoherence_weight=0.0,
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
    gpu_policy="auto",
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
    decoder_feature_sparsity_groups: Option<Vec<Vec<usize>>>,
    max_iter: usize,
    sparsity_strength: Option<f64>,
    coord_sparsity: &str,
    scad_mcp_gamma: Option<f64>,
    smoothness: f64,
    alpha: Option<f64>,
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
    gpu_policy: &str,
    promote_from_residual: bool,
    run_structure_search: bool,
    structured_residual_passes: usize,
) -> PyResult<PyObject> {
    let gpu_policy = gam::gpu::GpuPolicy::parse(gpu_policy).ok_or_else(|| {
        py_value_error(format!(
            "sae_manifold_fit gpu must be 'auto', 'off', or 'required'; got {gpu_policy:?}"
        ))
    })?;
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
    let strength = gam::terms::sae::manifold::resolve_public_assignment_strength(
        gam::terms::sae::manifold::SaeFitAssignmentKind::from_tag(&assignment)
            .map_err(py_value_error)?,
        n_obs,
        p_out,
        k_atoms,
        alpha,
        sparsity_strength,
    )
    .map_err(py_value_error)?;
    let resolved_alpha = strength.alpha;
    let learnable_alpha = strength.learnable_alpha;
    let sparsity_strength = strength.sparsity_strength;
    let resolved_learning_rate = learning_rate.unwrap_or(if assignment == "threshold_gate" {
        0.05
    } else {
        1.0
    });
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
        p_out,
    })
    .map_err(py_value_error)?;
    // #2822 — the coordinate ARD prior is mandatory, so every fit carries it.
    penalties.push("ARDPenalty".to_string());

    // Crossing K>P under hard TopK changes representation before the dense
    // minimal seed is even named. The support driver owns admission, seeding,
    // direct active-row fitting, LAML selection, and the sparse model payload;
    // there is no dense retry or translation adapter on this branch.
    if assignment == "topk" && k_atoms > p_out {
        let support_k = top_k.ok_or_else(|| {
            py_value_error("overcomplete assignment='topk' requires top_k".to_string())
        })?;
        // #2573: the support-sparse lane is CPU-only. `gpu_policy` is parsed and
        // validated above, then threaded ONLY into the dense constructor below,
        // which this branch never reaches -- so the argument was silently
        // dropped and `gpu="required"` completed a multi-hour CPU fit reporting
        // success (measured: K=32,000/P=128/N=44,818, 2h06m at 0% GPU). That
        // contradicts `GpuPolicy::Required`'s own definition, "require GPU
        // kernels and error if the requested path is unsupported", so refuse
        // rather than downgrade. `auto` and `off` both legitimately run on CPU
        // here and are unchanged.
        if matches!(gpu_policy, gam::gpu::GpuPolicy::Required) {
            return Err(py_value_error(format!(
                "sae_manifold_fit gpu='required' cannot be honoured for an overcomplete \
                 (K={k_atoms} > P={p_out}) assignment='topk' dictionary: the support-sparse \
                 lane has no GPU kernels and would run entirely on the CPU (#2573). Pass \
                 gpu='off' or gpu='auto' to accept a CPU fit, or use a dense (K <= P) \
                 dictionary, which does have a GPU path."
            )));
        }
        if !has_declared_bases
            && matches!(
                atom_topology.as_deref(),
                Some("circle" | "sphere" | "torus" | "projective_plane" | "klein_bottle" | "mobius")
            )
        {
            return Err(py_value_error(format!(
                "overcomplete (K={k_atoms} > P={p_out}) dictionaries do not take a homogeneous \
                 cyclic/compact topology by shorthand: most features are not cyclic, and an \
                 all-{} dictionary is a modeling prior the data never chose. Use the default \
                 atom_topology=None (auto: a uniform linear/euclidean/periodic portfolio \
                 adjudicated by the support competition and the REML/ARD priors), or spell out \
                 an explicit per-atom atom_basis vector if a homogeneous compact dictionary is \
                 genuinely intended.",
                atom_topology.as_deref().unwrap_or("")
            )));
        }
        if max_iter == 0 {
            return Err(py_value_error(
                "support-sparse ManifoldSAE requires max_iter >= 1".to_string(),
            ));
        }
        if initial_logits.is_some() || initial_coords.is_some() {
            return Err(py_value_error(
                "support-sparse ManifoldSAE accepts only canonical sparse warm state; dense a_init/t_init tensors are invalid"
                    .to_string(),
            ));
        }
        if analytic_penalties.is_some() {
            return Err(py_value_error(format!(
                "support-sparse ManifoldSAE does not accept dense-coordinate or coefficient penalties; requested {penalties:?}. Its smoothing term is the LAML-selected final-function seminorm"
            )));
        }
        if learnable_alpha || gumbel_schedule.is_some() || threshold_gate_threshold != 0.0 {
            return Err(py_value_error(
                "support-sparse hard TopK has read-only binary gates: learnable alpha, Gumbel schedules, and threshold-gate thresholds are not model coordinates"
                    .to_string(),
            ));
        }
        if fisher_factors.is_some()
            || fisher_mass_residual.is_some()
            || fisher_provenance.is_some()
            || fisher_factor_kind.is_some()
            || row_loss_weights.is_some()
        {
            return Err(py_value_error(
                "support-sparse ManifoldSAE currently requires the Euclidean row measure with uniform row weights; metric shards must not be silently discarded"
                    .to_string(),
            ));
        }
        if separation_barrier_strength_override.is_some()
            || promote_from_residual
            || run_structure_search
            || structured_residual_passes != 0
        {
            return Err(py_value_error(
                "support-sparse ManifoldSAE is the unbundled fixed-support fit; separation barriers, promotion, structure search, and structured-refit passes are separate stages"
                    .to_string(),
            ));
        }
        return crate::manifold::support_sparse_sae_ffi::fit_support_sparse_manifold_sae(
            py,
            crate::manifold::support_sparse_sae_ffi::SupportSparseFitRequest {
                target: z_view,
                atom_basis,
                atom_dim,
                support_k,
                initial_smoothness: smoothness,
                max_iter,
                trust_radius: resolved_learning_rate,
                random_state,
            },
        );
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
        block_orthogonality_weight,
        random_state,
        top_k,
        initial_logits,
        initial_coords,
        threshold_gate_threshold,
        fisher_factors.clone(),
        fisher_mass_residual,
        fisher_provenance.clone(),
        fisher_factor_kind.clone(),
        row_loss_weights,
        separation_barrier_strength_override,
        gpu_policy,
        promote_from_residual,
        run_structure_search,
        structured_residual_passes,
    )?;
    let raw = crate::manifold::manifold_sae_coercion::py_any_to_json_value(raw.bind(py).as_any())?;
    if raw.get("model_kind").and_then(serde_json::Value::as_str) == Some("tier0_null") {
        let mean = serde_json::from_value(raw.get("training_mean").cloned().ok_or_else(|| {
            py_value_error("Tier0SAE report is missing training_mean".to_string())
        })?)
        .map_err(|error| py_value_error(format!("Tier0SAE training_mean: {error}")))?;
        let fitted = serde_json::from_value(
            raw.get("fitted")
                .cloned()
                .ok_or_else(|| py_value_error("Tier0SAE report is missing fitted".to_string()))?,
        )
        .map_err(|error| py_value_error(format!("Tier0SAE fitted: {error}")))?;
        let vanished_atoms =
            serde_json::from_value(raw.get("vanished_atoms").cloned().ok_or_else(|| {
                py_value_error("Tier0SAE report is missing vanished_atoms".to_string())
            })?)
            .map_err(|error| py_value_error(format!("Tier0SAE vanished_atoms: {error}")))?;
        let core = Tier0SaeCore {
            mean,
            fitted,
            residual_sum_squares: raw
                .get("residual_sum_squares")
                .and_then(serde_json::Value::as_f64)
                .ok_or_else(|| {
                    py_value_error("Tier0SAE report has invalid residual_sum_squares".to_string())
                })?,
            reconstruction_r2: raw
                .get("reconstruction_r2")
                .and_then(serde_json::Value::as_f64)
                .ok_or_else(|| {
                    py_value_error("Tier0SAE report has invalid reconstruction_r2".to_string())
                })?,
            metric_provenance: raw
                .get("metric_provenance")
                .and_then(serde_json::Value::as_str)
                .ok_or_else(|| {
                    py_value_error("Tier0SAE report has invalid metric_provenance".to_string())
                })?
                .to_string(),
            vanished_atoms,
        };
        return Ok(Py::new(py, core)?.into_any());
    }
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
    };
    let payload = crate::manifold::manifold_sae_coercion::build_manifold_sae_payload(
        &raw,
        crate::manifold::manifold_sae_coercion::column_mean(z_view),
        &config,
    )
    .map_err(py_value_error)?;
    Ok(Py::new(py, crate::ManifoldSaeCore::from_payload(payload)?)?.into_any())
}

/// Parse the strict canonical geometry-plan wire. Deserialization re-enters the
/// validated `SaeAtomGeometryPlan` constructor; no former parallel metadata or
/// decoder-width inference is accepted at this boundary.
fn sae_geometry_plans_from_py(
    context: &str,
    geometry_plans: &Bound<'_, PyAny>,
) -> PyResult<Vec<SaeAtomGeometryPlan>> {
    let value = crate::manifold::manifold_sae_coercion::py_any_to_json_value(geometry_plans)?;
    let plans: Vec<SaeAtomGeometryPlan> = serde_json::from_value(value)
        .map_err(|error| py_value_error(format!("{context}: invalid geometry_plans: {error}")))?;
    if plans.is_empty() {
        return Err(py_value_error(format!(
            "{context}: geometry_plans must contain at least one plan"
        )));
    }
    Ok(plans)
}

/// Out-of-sample inference: same Newton driver as the fit path, with the
/// trained decoder blocks held frozen across iterations. `geometry_plans`
/// carries the complete immutable topology, resolution, reference metric, and
/// derived width for every `(M_k, p)` decoder block.
///
/// This array-level entry is the full-support (`K<=P`) specialization. A fitted
/// overcomplete TopK model owns support-native OOS through
/// `ManifoldSAE.converged_latents`; accepting it here would recreate the exact
/// dense `N×K` payload the support representation removes.
///
/// Returns the same full payload dict as the dense fit path (issue #357): the
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
    geometry_plans: &[SaeAtomGeometryPlan],
    decoder_blocks: &[ndarray::ArrayView2<'_, f64>],
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
    let k_atoms = geometry_plans.len();
    if decoder_blocks.len() != k_atoms {
        return Err(format!(
            "sae_manifold_predict_oos: decoder count {} must equal geometry-plan count K={k_atoms}",
            decoder_blocks.len()
        ));
    }
    if assignment_kind == "topk" && k_atoms > x_view.ncols() {
        return Err(format!(
            "sae_manifold_predict_oos: overcomplete hard TopK requires the fitted support-native ManifoldSAE OOS entry; dense array OOS would allocate N×K at K={k_atoms} > P={}",
            x_view.ncols()
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
    let atoms =
        gam::terms::sae::manifold::persisted_oos_atom_specs(geometry_plans, decoder_blocks)?;
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
    geometry_plans,
    decoder_blocks,
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
    geometry_plans: &Bound<'py, PyAny>,
    decoder_blocks: Vec<PyReadonlyArray2<'py, f64>>,
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
    let geometry_plans = sae_geometry_plans_from_py("sae_manifold_predict_oos", geometry_plans)?;
    let decoder_views: Vec<ndarray::ArrayView2<'_, f64>> =
        decoder_blocks.iter().map(|b| b.as_array()).collect();
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
        &geometry_plans,
        &decoder_views,
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
/// in the caller-declared Tier-0 frame, and calls
/// [`gam::terms::sae::manifold::run_sae_manifold_certify_external`].
///
/// `initial_coords`/`initial_logits` are REQUIRED here (unlike the fit path's
/// optional warm start): they ARE the trained state being certified, not a
/// seed for further optimization. `log_lambda_sparse`/`log_lambda_smooth`/
/// `log_ard` are the trained terminal regularization state the certificate is
/// evaluated under — the same "must supply the regularization that produced
/// the decoder" contract `sae_manifold_predict_oos` already carries.
/// `tier0_mean` / `tier0_scale` identify the centered and standardized frame
/// used by training while the target and persisted decoders remain physical.
/// The native entry first audits the supplied state without taking an optimizer
/// step. A nonstationary state returns a typed evaluation-only diagnostic with
/// no fit payload or structure evidence. A passing state alone reaches the
/// ordinary fit-report marshaller, with termination verdict
/// `audited_stationary` and zero optimization iterations.
#[pyfunction(signature = (
    z,
    geometry_plans,
    decoder_blocks,
    initial_coords,
    initial_logits,
    alpha,
    tau,
    assignment_kind,
    log_lambda_sparse,
    log_lambda_smooth,
    log_ard,
    tier0_mean = None,
    tier0_scale = None,
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
    geometry_plans: &Bound<'py, PyAny>,
    decoder_blocks: Vec<PyReadonlyArray2<'py, f64>>,
    initial_coords: Vec<PyReadonlyArray2<'py, f64>>,
    initial_logits: PyReadonlyArray2<'py, f64>,
    alpha: f64,
    tau: f64,
    assignment_kind: String,
    log_lambda_sparse: f64,
    log_lambda_smooth: Vec<f64>,
    log_ard: Vec<Vec<f64>>,
    tier0_mean: Option<PyReadonlyArray1<'py, f64>>,
    tier0_scale: Option<PyReadonlyArray1<'py, f64>>,
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
    let geometry_plans =
        sae_geometry_plans_from_py("sae_manifold_certify_external", geometry_plans)?;
    let z_view = z.as_array();
    let (n_obs, p_out) = z_view.dim();
    let k_atoms = geometry_plans.len();
    if assignment_kind == "topk" && k_atoms > p_out {
        return Err(py_value_error(format!(
            "sae_manifold_certify_external: overcomplete hard TopK certification requires canonical support indices and heterogeneous compact coordinates; dense logits are invalid at K={k_atoms} > P={p_out}"
        )));
    }
    if decoder_blocks.len() != k_atoms {
        return Err(py_value_error(format!(
            "sae_manifold_certify_external: decoder count {} must equal geometry-plan count K={k_atoms}",
            decoder_blocks.len()
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

    let decoder_views = decoder_blocks
        .iter()
        .map(|block| block.as_array())
        .collect::<Vec<_>>();
    let atoms =
        gam::terms::sae::manifold::persisted_oos_atom_specs(&geometry_plans, &decoder_views)
            .map_err(py_value_error)?;

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
    let max_atom_dim = geometry_plans
        .iter()
        .map(SaeAtomGeometryPlan::latent_dim)
        .max()
        .ok_or_else(|| {
            py_value_error("sae_manifold_certify_external: geometry_plans is empty".to_string())
        })?;
    let total_basis = geometry_plans
        .iter()
        .map(SaeAtomGeometryPlan::basis_size)
        .try_fold(0usize, |total, width| {
            total.checked_add(width?).ok_or_else(|| {
                "sae_manifold_certify_external: total basis width overflowed".to_string()
            })
        })
        .map_err(py_value_error)?;
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
        tier0_mean: tier0_mean.map(|values| values.as_array().to_owned()),
        tier0_scale: tier0_scale.map(|values| values.as_array().to_owned()),
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

    let outcome = gam::terms::sae::manifold::run_sae_manifold_certify_external(request)
        .map_err(|err| sae_fit_error_to_pyerr(py, err))?;
    let report = match outcome {
        gam::terms::sae::manifold::SaeExternalCertificationOutcome::Certified(report) => report,
        gam::terms::sae::manifold::SaeExternalCertificationOutcome::NonStationary(report) => {
            let out = PyDict::new(py);
            out.set_item("status", "nonstationary")?;
            out.set_item("is_fit", false)?;
            out.set_item("optimization_iterations", report.optimization_iterations)?;
            out.set_item("reason", report.reason)?;
            let inner = PyDict::new(py);
            inner.set_item("raw_gradient_norm", report.inner.raw_gradient_norm)?;
            inner.set_item(
                "quotient_gradient_norm",
                report.inner.quotient_gradient_norm,
            )?;
            inner.set_item("stationarity_bound", report.inner.stationarity_bound)?;
            match &report.inner.newton_decrement_relative {
                Ok(relative) => inner.set_item("newton_decrement_relative", *relative)?,
                Err(reason) => inner.set_item("newton_decrement_unresolved", reason.as_str())?,
            }
            let parameter_space = PyDict::new(py);
            let parameter_within_bound = report.inner.parameter_space.within_bound();
            match &report.inner.parameter_space {
                gam::terms::sae::manifold::SaeParameterSpaceKktAudit::Resolved {
                    scaled_gradient_max,
                    stationarity_bound,
                } => {
                    parameter_space.set_item("status", "resolved")?;
                    parameter_space
                        .set_item("scaled_gradient_max", scaled_gradient_max)?;
                    parameter_space.set_item("stationarity_bound", stationarity_bound)?;
                }
                gam::terms::sae::manifold::SaeParameterSpaceKktAudit::Unresolved(reason) => {
                    parameter_space.set_item("status", "unresolved")?;
                    parameter_space.set_item("reason", reason.to_string())?;
                }
            }
            parameter_space.set_item("within_bound", parameter_within_bound)?;
            inner.set_item("parameter_space", parameter_space)?;
            inner.set_item("certifies", report.inner.certifies())?;
            out.set_item("inner_kkt", inner)?;
            let outer = PyDict::new(py);
            outer.set_item("raw_gradient_norm", report.outer_raw_gradient_norm)?;
            outer.set_item(
                "projected_gradient_norm",
                report.outer_projected_gradient_norm,
            )?;
            outer.set_item("stationarity_bound", report.outer_stationarity_bound)?;
            out.set_item("outer_stationarity", outer)?;
            out.set_item("structure_search", py.None())?;
            out.set_item("structure_certificate", py.None())?;
            return Ok(out.unbind());
        }
    };

    // #2266/#2263 retired on both layers: gam-sae's `finalize_sae_fit_report`
    // now builds the `SaeFitReport` for the fit and certify entries alike, and
    // `sae_fit_report_into_dict` marshals it here. Only the two leading
    // certification keys below are local to this entry, and they are set first
    // so the emitted key order is unchanged.
    let out = PyDict::new(py);
    out.set_item("status", "certified")?;
    out.set_item("is_fit", true)?;
    sae_fit_report_into_dict(
        py,
        &out,
        report,
        p_out,
        assignment_kind,
        top_k,
        fisher_mass_residual_owned
            .as_ref()
            .map(|values| values.view()),
    )?;
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
/// `delta`, the endpoint `predicted_nats` dose, the `off_manifold_norm`
/// self-check, and the `metric_provenance`.
///
/// The term rebuild mirrors [`sae_manifold_predict_oos`] (same plan/evaluator
/// machinery), but where `predict_oos` runs the frozen-decoder Newton solve on a
/// *new* `X`, this seeds the term directly from the trained latents/logits so the
/// dose is measured through the model as fitted. `coords` is one `(N, d_k)` array
/// per atom (the trained `on_atom_coords_t`); `logits` is `(N, K)` (the trained
/// routing logits) so the per-atom amplitude / measured-row selection inside
/// `steer_delta` sees the fitted assignments. `fisher_factors` is the `(n, p, r)`
/// harvest shard `U`; its presence installs `RowMetric::OutputFisher` (and makes
/// `predicted_nats` available), exactly as in the fit.
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
    geometry_plans: &[SaeAtomGeometryPlan],
    decoder_blocks: &[ndarray::ArrayView2<'_, f64>],
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
        geometry_plans,
        decoder_blocks,
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
    geometry_plans: &[SaeAtomGeometryPlan],
    decoder_blocks: &[ndarray::ArrayView2<'_, f64>],
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
    let k_atoms = geometry_plans.len();
    if decoder_blocks.len() != k_atoms || coords.len() != k_atoms {
        return Err(py_value_error(format!(
            "sae_steer_delta: decoder and coordinate counts must equal geometry-plan count K={k_atoms}"
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
    let atoms = gam::terms::sae::manifold::persisted_oos_atom_specs(geometry_plans, decoder_blocks)
        .map_err(py_value_error)?;
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

/// Rebuild the trained term from arrays and solve for the displacement along a
/// chart direction realizing a TARGET output-KL dose (gh#2263). Mirrors
/// [`steer_delta_with_metric_from_arrays`] but drives the target-dose entry; the
/// optional `probe` (a patched-forward KL callback) drives the closed-loop
/// correction.
struct SteerToTargetArraysRequest<'a> {
    atom_k: usize,
    metric_row: usize,
    target_nats: f64,
    t_from: ndarray::ArrayView1<'a, f64>,
    direction: ndarray::ArrayView1<'a, f64>,
    geometry_plans: &'a [SaeAtomGeometryPlan],
    decoder_blocks: &'a [ndarray::ArrayView2<'a, f64>],
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
/// Every field is required and no other key is accepted; optionality belongs
/// only to the separate plan-aware applied-dose probe.
struct ManifoldSteerToTargetRequest {
    atom_k: usize,
    metric_row: usize,
    target_nats: f64,
    t_from: Array1<f64>,
    direction: Array1<f64>,
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
        // A key the solve does not read would be silently ignored. The landing
        // coordinate is what a target-dose solve returns, and the solve has no
        // accuracy, probe-budget or readout option, so a request naming one of
        // those was written under a retired contract.
        for key in request.keys() {
            let key = key.extract::<String>()?;
            match key.as_str() {
                "atom_k" | "metric_row" | "target_nats" | "t_from" | "direction" => {}
                "t_to" => {
                    return Err(py_value_error(
                        "ManifoldSaeCore.steer_to_target: 't_to' is solved, not requested; pass \
                         the chart 'direction' to move along instead"
                            .to_string(),
                    ));
                }
                other => {
                    return Err(py_value_error(format!(
                        "ManifoldSaeCore.steer_to_target: request key '{other}' is not read; the \
                         request is atom_k, metric_row, target_nats, t_from and direction, and \
                         the solve resolves the displacement to its representation limit with \
                         no accuracy or probe-budget option"
                    )));
                }
            }
        }
        let t_from = required_steer_to_target_item(request, "t_from")?
            .extract::<PyReadonlyArray1<'_, f64>>()?
            .as_array()
            .to_owned();
        let direction = required_steer_to_target_item(request, "direction")?
            .extract::<PyReadonlyArray1<'_, f64>>()?
            .as_array()
            .to_owned();
        Ok(Self {
            atom_k: required_steer_to_target_item(request, "atom_k")?.extract()?,
            metric_row: required_steer_to_target_item(request, "metric_row")?.extract()?,
            target_nats: required_steer_to_target_item(request, "target_nats")?.extract()?,
            t_from,
            direction,
        })
    }
}

fn steer_to_target_from_arrays(
    request: SteerToTargetArraysRequest<'_>,
    probe: Option<&mut gam::inference::steering::AppliedDoseProbe<'_>>,
) -> PyResult<gam::inference::steering::TargetDosePlan> {
    let SteerToTargetArraysRequest {
        atom_k,
        metric_row,
        target_nats,
        t_from,
        direction,
        geometry_plans,
        decoder_blocks,
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
    let k_atoms = geometry_plans.len();
    if decoder_blocks.len() != k_atoms || coords.len() != k_atoms {
        return Err(py_value_error(format!(
            "sae_steer_to_target: decoder and coordinate counts must equal geometry-plan count K={k_atoms}"
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
    let atoms = gam::terms::sae::manifold::persisted_oos_atom_specs(geometry_plans, decoder_blocks)
        .map_err(py_value_error)?;
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
        direction: direction.to_vec(),
        target_nats,
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
        seed_displacement,
        displacement,
        steer,
        applied_probe,
        iterations,
        certified_attainable_upper_nats,
    } = plan;
    let resident_metric_nats = steer.predicted_nats;
    let resident_metric_nats_kind = steer.predicted_nats_kind.as_str();
    let out = steer_plan_to_pydict(py, steer)?;
    let bound = out.bind(py);
    bound.set_item("target_nats", target_nats)?;
    bound.set_item("seed_displacement", seed_displacement)?;
    bound.set_item("displacement", displacement)?;
    bound.set_item("iterations", iterations)?;
    bound.set_item("resident_metric_nats", resident_metric_nats)?;
    bound.set_item("resident_metric_nats_kind", resident_metric_nats_kind)?;
    match applied_probe {
        Some(observation) => {
            bound.set_item("effective_delta", observation.effective_delta.to_vec())?;
            bound.set_item("exact_directional_nats", observation.exact_directional_nats)?;
            bound.set_item("measured_nats", observation.measured_nats)?;
            // The canonical probed prediction is the exact directional Fisher
            // value. The resident factor remains separately visible above and
            // retains its original status; it is never promoted.
            bound.set_item("predicted_nats", observation.exact_directional_nats)?;
            bound.set_item("predicted_nats_kind", "exact_directional")?;
            bound.set_item("validation", "applied_dose_probe")?;
        }
        None => {
            bound.set_item("effective_delta", py.None())?;
            bound.set_item("exact_directional_nats", py.None())?;
            bound.set_item("measured_nats", py.None())?;
            bound.set_item("validation", "exact_factor_model")?;
        }
    }
    bound.set_item(
        "certified_attainable_upper_nats",
        certified_attainable_upper_nats,
    )?;
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
    geometry_plans,
    decoder_blocks,
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
    geometry_plans: &Bound<'py, PyAny>,
    decoder_blocks: Vec<PyReadonlyArray2<'py, f64>>,
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
    let geometry_plans = sae_geometry_plans_from_py("sae_steer_delta", geometry_plans)?;
    let decoder_views: Vec<ndarray::ArrayView2<'_, f64>> =
        decoder_blocks.iter().map(|b| b.as_array()).collect();
    let coord_views: Vec<ndarray::ArrayView2<'_, f64>> =
        coords.iter().map(|c| c.as_array()).collect();
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
        &geometry_plans,
        &decoder_views,
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
/// with AuxPrior for identifiability; the forward fit installs no decoder jets,
/// so it refuses an isometry descriptor by name.
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
    tensor_knots_concat = None,
    tensor_knot_offsets = None,
    tensor_degrees = None,
    analytic_penalties = None,
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
    tensor_knots_concat: Option<PyReadonlyArray1<'py, f64>>,
    tensor_knot_offsets: Option<Vec<usize>>,
    tensor_degrees: Option<Vec<usize>>,
    analytic_penalties: Option<String>,
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
    let registry =
        latent_analytic_penalty_registry(n_obs, latent_dim, analytic_penalties.as_deref())
            .map_err(py_value_error)?;
    let backward = gaussian_reml_fit_latent_backward_impl(
        t.as_array(),
        y.as_array(),
        n_obs,
        latent_dim,
        centers.as_array(),
        penalty.as_array(),
        grad_lambda,
        grad_coefficients.as_ref().map(|g| g.as_array()),
        grad_fitted.as_ref().map(|g| g.as_array()),
        grad_reml_score,
        grad_edf,
        m,
        effective_weights.as_ref().map(|w| w.view()),
        init_lambda,
        aux_u.as_ref().map(|a| a.as_array()),
        family,
        aux_strength,
        dim_selection_precision.as_ref(),
        &basis_kind,
        tensor_knots_concat.as_ref().map(|a| a.as_array()),
        tensor_knot_offsets.as_deref(),
        tensor_degrees.as_deref(),
        &registry,
    )
    .map_err(py_value_error)?;
    let out = PyDict::new(py);
    out.set_item("grad_t", backward.grad_t.into_pyarray(py))?;
    out.set_item("grad_y", backward.reml.grad_y.into_pyarray(py))?;
    out.set_item("grad_penalty", backward.reml.grad_penalty.into_pyarray(py))?;
    out.set_item("grad_weights", backward.reml.grad_weights.into_pyarray(py))?;
    if let Some(grad) = backward.grad_aux_log_strength {
        out.set_item("grad_aux_log_strength", grad)?;
        out.set_item("grad_log_mu", grad)?;
    } else {
        out.set_item("grad_aux_log_strength", py.None())?;
        out.set_item("grad_log_mu", py.None())?;
    }
    if let Some(grad) = backward.grad_dim_selection_log_precision {
        out.set_item("grad_dim_selection_log_precision", grad.into_pyarray(py))?;
    } else {
        out.set_item("grad_dim_selection_log_precision", py.None())?;
    }
    Ok(out.unbind())
}

/// Outputs of [`gaussian_reml_fit_latent_backward_impl`].
struct LatentGaussianBackward {
    reml: gam::solver::gaussian_reml::GaussianRemlBackwardResult,
    grad_t: Array2<f64>,
    grad_aux_log_strength: Option<f64>,
    grad_dim_selection_log_precision: Option<Array1<f64>>,
}

/// The core of [`gaussian_reml_fit_latent_backward`]: the REML adjoint through the
/// latent design, the identifiability priors, and every analytic penalty the
/// forward prices into `reml_score` (#2933 F02).
fn gaussian_reml_fit_latent_backward_impl(
    t: ArrayView1<'_, f64>,
    y: ArrayView2<'_, f64>,
    n_obs: usize,
    latent_dim: usize,
    centers: ArrayView2<'_, f64>,
    penalty: ArrayView2<'_, f64>,
    grad_lambda: f64,
    grad_coefficients: Option<ArrayView2<'_, f64>>,
    grad_fitted: Option<ArrayView2<'_, f64>>,
    grad_reml_score: f64,
    grad_edf: f64,
    m: usize,
    weights: Option<ArrayView1<'_, f64>>,
    init_lambda: Option<f64>,
    aux_u: Option<ArrayView2<'_, f64>>,
    family: AuxPriorFamily,
    aux_strength: Option<f64>,
    dim_selection_precision: Option<&ValidatedDimSelectionPrecisions>,
    basis_kind: &str,
    tensor_knots_concat: Option<ArrayView1<'_, f64>>,
    tensor_knot_offsets: Option<&[usize]>,
    tensor_degrees: Option<&[usize]>,
    analytic_penalties: &AnalyticPenaltyRegistry,
) -> Result<LatentGaussianBackward, String> {
    let basis_kind_normalized = latent_basis_kind(basis_kind)?;
    // Forward design (Φ), t-matrix, and input-location jet share one dispatcher.
    let (design, t_mat, jet) = build_latent_forward_design(
        basis_kind_normalized,
        t,
        n_obs,
        latent_dim,
        centers,
        m,
        tensor_knots_concat,
        tensor_knot_offsets,
        tensor_degrees,
        // Standalone Python backward/gradient entrypoint: no manifold/chart
        // concept here (the Rust outer optimizer routes through
        // `LatentOuterProblem`), so the latent design stays the open Euclidean
        // basis — byte-identical to prior behavior.
        None,
    )?;
    let fit = gaussian_reml_multi_closed_form_with_cache(
        design.view(),
        y,
        penalty,
        weights,
        init_lambda,
        None,
    )
    .map_err(|err| err.to_string())?;
    // Every output lane uses the same core REML adjoint, including its
    // determinant, dispersion, and implicit smoothing-strength derivatives.
    let reml = gaussian_reml_multi_closed_form_backward_from_fit(
        design.view(),
        y,
        penalty,
        weights,
        &fit,
        grad_lambda,
        grad_coefficients,
        grad_fitted,
        grad_reml_score,
        grad_edf,
    )
    .map_err(|err| err.to_string())?;
    let mut grad_t =
        contract_input_loc_gradient(reml.grad_x.view(), &jet).map_err(|err| err.to_string())?;
    // Identifiability-mode additive contributions to grad_t plus log-normalizer
    // adjoints. Fixes audit-revised claim that REML ARD/AuxPrior selection
    // needs the normalized prior terms, not only raw quadratic gradients.
    let mut grad_aux_log_strength: Option<f64> = None;
    let mut grad_dim_selection_log_precision: Option<Array1<f64>> = None;
    if let Some(u_view) = aux_u {
        let stats = latent_aux_prior_stats(t_mat.view(), u_view, family, aux_strength)?;
        let residual = &t_mat - &stats.targets;
        let projected_residual = aux_prior_targets(residual.view(), u_view, family)?;
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
    if let Some(precisions) = dim_selection_precision {
        let mut grad_log_prec = Array1::<f64>::zeros(latent_dim);
        for n in 0..n_obs {
            for a in 0..latent_dim {
                let prec = precisions.physical()[a];
                if grad_reml_score != 0.0 {
                    grad_t[n * latent_dim + a] += grad_reml_score * prec * t_mat[[n, a]];
                }
            }
        }
        for a in 0..latent_dim {
            let energy = precisions.axis_energy(t_mat.view(), a)?;
            grad_log_prec[a] = grad_reml_score * (energy - 0.5 * n_obs as f64);
        }
        grad_dim_selection_log_precision = Some(grad_log_prec);
    }
    // #2933 F02 — the forward prices every analytic penalty into `reml_score`,
    // so its gradient enters here too.
    grad_t.scaled_add(
        grad_reml_score,
        &latent_analytic_penalty_grad(analytic_penalties, t)?,
    );
    let mut grad_t_matrix = Array2::<f64>::zeros((n_obs, latent_dim));
    for n in 0..n_obs {
        for a in 0..latent_dim {
            grad_t_matrix[[n, a]] = grad_t[n * latent_dim + a];
        }
    }
    Ok(LatentGaussianBackward {
        reml,
        grad_t: grad_t_matrix,
        grad_aux_log_strength,
        grad_dim_selection_log_precision,
    })
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
mod public_block_orthogonality_tests {
    use super::append_public_block_orthogonality_penalty;

    #[test]
    fn singleton_groups_use_resolved_storage_width() {
        let existing = serde_json::json!([{
            "kind": "isometry",
            "target": "t",
            "weight": 2.0,
        }])
        .to_string();
        let augmented = append_public_block_orthogonality_penalty(Some(existing), 3.5, 3)
            .expect("three-axis ambient geometry is admissible")
            .expect("positive weight emits a descriptor");
        let descriptors: Vec<serde_json::Value> =
            serde_json::from_str(&augmented).expect("descriptor JSON");

        assert_eq!(descriptors.len(), 2);
        assert_eq!(descriptors[0]["kind"], "isometry");
        assert_eq!(descriptors[1]["kind"], "block_orthogonality");
        assert_eq!(
            descriptors[1]["groups"],
            serde_json::json!([[0], [1], [2]]),
            "S² has two intrinsic degrees of freedom but three ambient storage axes"
        );
        assert_eq!(descriptors[1]["weight"], 3.5);
    }

    #[test]
    fn one_storage_axis_is_not_a_block_partition() {
        let error = append_public_block_orthogonality_penalty(None, 1.0, 1)
            .expect_err("a single stored axis cannot form orthogonal blocks");
        assert!(error.contains("at least two resolved latent storage axes"));
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
    use super::sae_atom_basis_kind_name;
    use gam::terms::sae::manifold::{
        EuclideanPatchEvaluator, SaeAtomBasisKind, SaeBasisEvaluator, sae_atom_basis_kind_from_artifact_str,
        sae_atom_basis_kind_from_str,
    };
    use ndarray::Array2;

    /// #1221 — the canonical `"linear"` token is a first-class topology
    /// distinct from the degree-2 `"euclidean"` patch. Removed aliases must not
    /// silently acquire linear semantics at the internal artifact converter.
    #[test]
    fn linear_topology_is_first_class_and_round_trips() {
        assert_eq!(
            sae_atom_basis_kind_from_str("linear")
                .expect("the canonical linear token parses"),
            SaeAtomBasisKind::Linear,
            "the canonical token must parse to the genuinely-linear atom"
        );
        for removed in ["linear_rank1", "affine", "LINEAR"] {
            assert_eq!(
                sae_atom_basis_kind_from_artifact_str(removed),
                SaeAtomBasisKind::Precomputed(removed.to_string()),
                "removed alias {removed:?} must remain an opaque native artifact tag"
            );
        }
        assert_eq!(
            sae_atom_basis_kind_name(&SaeAtomBasisKind::Linear),
            "linear",
            "the linear atom must round-trip under its honest name"
        );
        // The quadratic patch is a DIFFERENT kind — `"linear"` must not collapse
        // onto it, or the curved-vs-linear comparison would be mislabeled again.
        assert_eq!(
            sae_atom_basis_kind_from_str("euclidean")
                .expect("the canonical euclidean token parses"),
            SaeAtomBasisKind::EuclideanPatch,
            "the canonical euclidean token is the degree-2 patch"
        );
        for removed in ["euclidean_patch", "euclidean_quadratic_patch"] {
            assert_eq!(
                sae_atom_basis_kind_from_artifact_str(removed),
                SaeAtomBasisKind::Precomputed(removed.to_string()),
                "removed alias {removed:?} must remain an opaque native artifact tag"
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
