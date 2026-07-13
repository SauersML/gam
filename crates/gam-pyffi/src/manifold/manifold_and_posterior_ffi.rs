fn normalize_coefficient_names_value(
    names: Option<&serde_json::Value>,
    n_coeffs: usize,
) -> Vec<String> {
    match names.and_then(|value| value.as_array()) {
        Some(raw) if raw.len() == n_coeffs => raw.iter().map(coefficient_name_from_json).collect(),
        _ => default_coefficient_names(n_coeffs),
    }
}

fn posterior_coefficient_names_json_impl(request_json: &str) -> Result<String, String> {
    let request: PosteriorCoefficientNamesRequest = serde_json::from_str(request_json)
        .map_err(|err| format!("posterior_coefficient_names_json: parse request: {err}"))?;
    let names =
        normalize_coefficient_names_value(request.coefficient_names.as_ref(), request.n_coeffs);
    serde_json::to_string(&names)
        .map_err(|err| format!("posterior_coefficient_names_json: serialise names: {err}"))
}

#[derive(Deserialize)]
struct PosteriorTraceSelectionRequest {
    coefficient_names: Vec<String>,
    coefficients: Option<serde_json::Value>,
    max_panels: usize,
}

#[derive(Serialize)]
struct PosteriorTraceSelectionPayload {
    indices: Vec<usize>,
    labels: Vec<String>,
}

fn trace_index_from_value(
    value: &serde_json::Value,
    names: &[String],
    n_coeffs: usize,
) -> Result<usize, String> {
    match value {
        serde_json::Value::String(name) => names
            .iter()
            .position(|entry| entry == name)
            .ok_or_else(|| format!("unknown coefficient {name:?}; known: {names:?}")),
        serde_json::Value::Number(number) => {
            let raw = number
                .as_i64()
                .ok_or_else(|| format!("coefficient index must be an integer, got {number}"))?;
            if raw < 0 || raw as usize >= n_coeffs {
                return Err(format!(
                    "coefficient index {raw} out of range for {n_coeffs} coefficients"
                ));
            }
            Ok(raw as usize)
        }
        other => Err(format!(
            "coefficient selector must be a name, index, or list of names/indices; got {other}"
        )),
    }
}

fn posterior_trace_selection_json_impl(request_json: &str) -> Result<String, String> {
    let request: PosteriorTraceSelectionRequest = serde_json::from_str(request_json)
        .map_err(|err| format!("posterior_trace_selection_json: parse request: {err}"))?;
    let n_coeffs = request.coefficient_names.len();
    let indices = match request.coefficients.as_ref() {
        None | Some(serde_json::Value::Null) => (0..request.max_panels.min(n_coeffs)).collect(),
        Some(serde_json::Value::Array(items)) => items
            .iter()
            .map(|item| trace_index_from_value(item, &request.coefficient_names, n_coeffs))
            .collect::<Result<Vec<_>, _>>()?,
        Some(value) => vec![trace_index_from_value(
            value,
            &request.coefficient_names,
            n_coeffs,
        )?],
    };
    if indices.is_empty() {
        return Err("plot_trace: no coefficients selected".to_string());
    }
    let labels = indices
        .iter()
        .map(|index| request.coefficient_names[*index].clone())
        .collect();
    let payload = PosteriorTraceSelectionPayload { indices, labels };
    serde_json::to_string(&payload)
        .map_err(|err| format!("posterior_trace_selection_json: serialise payload: {err}"))
}

fn posterior_eta_bands_impl(
    eta: Array2<f64>,
    family_kind: &str,
    level: f64,
    link_spec: Option<&str>,
) -> Result<PosteriorPredictBandsPayload, String> {
    // Prefer the typed link spec when supplied so the parameterized links
    // (`Sas`, `Mixture`, `LatentCLogLog`, `BetaLogistic`) push their per-fit
    // state through to the response-scale bands; otherwise fall back to the
    // bare string tag (issue #1133).
    let parsed_link: Option<InverseLink> = match link_spec {
        Some(spec_json) => Some(
            serde_json::from_str(spec_json)
                .map_err(|err| format!("failed to parse link_spec for posterior bands: {err}"))?,
        ),
        None => None,
    };
    let selector = match parsed_link.as_ref() {
        Some(link) => posterior_bands::LinkSelector::Spec(link),
        None => posterior_bands::LinkSelector::Tag(family_kind),
    };
    let (n_draws, n_rows) = eta.dim();
    let family_kind = match selector {
        posterior_bands::LinkSelector::Tag(tag) => tag.to_string(),
        posterior_bands::LinkSelector::Spec(spec) => spec.link_function().name().to_string(),
    };
    let (
        linear_predictor,
        linear_predictor_lower,
        linear_predictor_upper,
        mean,
        mean_lower,
        mean_upper,
    ) = posterior_bands::eta_bands_from_matrix_link(eta.view(), selector, level)?;
    Ok(PosteriorPredictBandsPayload {
        linear_predictor,
        linear_predictor_lower,
        linear_predictor_upper,
        mean,
        mean_lower,
        mean_upper,
        n_rows,
        n_draws,
        model_class: String::new(),
        family_kind,
    })
}

fn posterior_draw_bands_impl(
    eta: Array2<f64>,
    mean: Array2<f64>,
    level: f64,
) -> Result<PosteriorPredictBandsPayload, String> {
    let (n_draws, n_rows) = eta.dim();
    let (
        linear_predictor,
        linear_predictor_lower,
        linear_predictor_upper,
        mean,
        mean_lower,
        mean_upper,
    ) = posterior_bands::draw_bands_from_matrices(eta.view(), mean.view(), level)?;
    Ok(PosteriorPredictBandsPayload {
        linear_predictor,
        linear_predictor_lower,
        linear_predictor_upper,
        mean,
        mean_lower,
        mean_upper,
        n_rows,
        n_draws,
        model_class: String::new(),
        family_kind: String::new(),
    })
}

/// Typed result of a posterior-predictive η evaluation. Carried as ndarray /
/// plain fields between the predict and bands paths and converted to numpy at
/// the pyfunction edge; the `(n_draws, n_rows)` matrix never rides JSON.
struct PosteriorPredictResult {
    eta: Array2<f64>,
    mean: Array2<f64>,
    model_class: String,
    family_kind: String,
    /// Serialized parameterized `InverseLink` metadata for callers that inspect
    /// the fitted link. Response draws themselves come from the polymorphic core
    /// predictor and never reconstruct a mean from this metadata.
    link_spec: Option<String>,
}

fn posterior_predict_bands_encoded_table_impl(
    model_bytes: &[u8],
    source: EncodedDataset,
    samples: Array2<f64>,
    level: f64,
) -> Result<PosteriorPredictBandsPayload, String> {
    // Reuse the polymorphic core prediction pipeline, then collapse both its
    // canonical-predictor and response-mean matrices inside Rust so predict()
    // never materializes either draw matrix on the Python side.
    let result = posterior_predict_encoded_table_impl(model_bytes, source, samples)?;
    let (n_draws, n_rows) = result.eta.dim();
    let (eta_mean, eta_lower, eta_upper, mean, mean_lower, mean_upper) =
        posterior_bands::draw_bands_from_matrices(result.eta.view(), result.mean.view(), level)?;
    Ok(PosteriorPredictBandsPayload {
        linear_predictor: eta_mean,
        linear_predictor_lower: eta_lower,
        linear_predictor_upper: eta_upper,
        mean,
        mean_lower,
        mean_upper,
        n_rows,
        n_draws,
        model_class: result.model_class,
        family_kind: result.family_kind,
    })
}

fn posterior_predict_encoded_table_impl(
    model_bytes: &[u8],
    source: EncodedDataset,
    samples: Array2<f64>,
) -> Result<PosteriorPredictResult, String> {
    let model = load_model_impl(model_bytes)?;
    let dataset = dataset_with_model_schema_from_encoded(&model, &source)?;
    let col_map = dataset.column_map();
    let prediction = gam_predict::predict_posterior_draws(
        &model,
        dataset.values.view(),
        &col_map,
        model.training_headers.as_ref(),
        samples.view(),
    )
    .map_err(|error| error.to_string())?;
    let matrices = prediction.into_matrices();
    Ok(PosteriorPredictResult {
        eta: matrices.eta,
        mean: matrices.mean,
        model_class: prediction_model_class_label(&model),
        family_kind: family_link_kind(&model_likelihood_spec(&model)).to_string(),
        link_spec: model_link_spec_json(&model),
    })
}

fn sample_encoded_table_impl(
    model_bytes: &[u8],
    source: EncodedDataset,
    options_json: Option<&str>,
) -> Result<SamplePayload, String> {
    let model = load_model_impl(model_bytes)?;
    let dataset = dataset_with_model_schema_from_encoded(&model, &source)?;
    let options = parse_sample_options(options_json)?;
    let cfg = resolve_nuts_config(&model, options);
    let col_map = dataset.column_map();
    let training_headers = model.training_headers.as_ref();
    let nuts = gam::inference::sample::sample_saved_model(
        &model,
        dataset.values.view(),
        &col_map,
        training_headers,
        &cfg,
    )?;
    Ok(build_sample_payload(&model, nuts, &cfg))
}

/// plain string on the Python side (the Rust `LikelihoodSpec` itself is
/// Returns the inverse-link kind tag emitted to the Python wrapper.
///
/// The tag is intentionally lower-kebab-case so it can be matched as a
/// plain string on the Python side (the Rust `LikelihoodSpec` itself is
/// not part of the FFI surface). Families that don't have a closed-form
/// scalar inverse link (`royston-parmar`, `binomial-sas`,
/// `binomial-beta-logistic`) get their own tags so the Python side can
/// refuse to compute a response-scale prediction by string-compare
/// instead of by guessing.
fn family_link_kind(family: &LikelihoodSpec) -> &'static str {
    match (&family.response, &family.link) {
        (ResponseFamily::RoystonParmar, _) => "royston-parmar",
        (ResponseFamily::Binomial, InverseLink::Sas(_)) => "sas",
        (ResponseFamily::Binomial, InverseLink::BetaLogistic(_)) => "beta-logistic",
        _ => family.link_function().name(),
    }
}

/// Serialize the fully parameterized [`InverseLink`] of a model's likelihood to
/// JSON so the Python wrapper can carry it back into the response-scale
/// transforms (`apply_inverse_link_array`, `posterior_eta_bands`) as a typed
/// `link_spec` rather than a lossy `family_kind` tag. The parameterized links
/// (`Sas`, `Mixture`, `LatentCLogLog`, `BetaLogistic`) carry per-fit state that
/// the bare string tag cannot represent; this is the seam that wires their
/// response-scale draws (issue #1133).
fn model_link_spec_json(model: &FittedModel) -> Option<String> {
    let spec = model_likelihood_spec(model);
    serde_json::to_string(&spec.link).ok()
}

/// Extract the `LikelihoodSpec` carried by the saved model's `family_state`.
/// Latent-survival / latent-binary states have no GLM-style likelihood; they
/// map to `RoystonParmar` with an identity link for the response-scale tag.
fn model_likelihood_spec(model: &FittedModel) -> LikelihoodSpec {
    match &model.payload().family_state {
        gam::inference::model::FittedFamily::Standard { likelihood, .. }
        | gam::inference::model::FittedFamily::LocationScale { likelihood, .. }
        | gam::inference::model::FittedFamily::MarginalSlope { likelihood, .. }
        | gam::inference::model::FittedFamily::Survival { likelihood, .. }
        | gam::inference::model::FittedFamily::TransformationNormal { likelihood, .. } => {
            likelihood.clone()
        }
        gam::inference::model::FittedFamily::LatentSurvival { .. }
        | gam::inference::model::FittedFamily::LatentBinary { .. } => LikelihoodSpec::new(
            ResponseFamily::RoystonParmar,
            InverseLink::Standard(StandardLink::Identity),
        ),
    }
}

/// Did this `NutsResult` come from exact NUTS or the Laplace fallback?
///
/// We badge the payload so the Python wrapper can surface the method
/// to users without re-deriving it from the model class. The fallback
/// produces iid draws and reports `rhat = 1.0` exactly with
/// `ess == n_draws`, which is a stable signature.
fn nuts_method_label(model: &FittedModel) -> &'static str {
    match model.predict_model_class() {
        PredictModelClass::Standard => "nuts",
        PredictModelClass::Survival => {
            // Survival latent / latent-binary / location-scale fall
            // back; everything else uses exact NUTS. Mirror the
            // dispatch in `gam::inference::sample::sample_saved_model`.
            match model.survival_likelihood.as_deref() {
                Some("latent") | Some("latent-binary") | Some("location-scale") => "laplace",
                None
                | Some("marginal-slope")
                | Some("transformation")
                | Some("weibull")
                | Some("royston-parmar")
                | Some(_) => "nuts",
            }
        }
        PredictModelClass::GaussianLocationScale
        | PredictModelClass::BinomialLocationScale
        | PredictModelClass::BernoulliMarginalSlope
        | PredictModelClass::DispersionLocationScale
        | PredictModelClass::TransformationNormal => "laplace",
    }
}

fn build_sample_payload(model: &FittedModel, nuts: NutsResult, cfg: &NutsConfig) -> SamplePayload {
    let n_coeffs = nuts.samples.ncols();
    let coefficient_names: Vec<String> = (0..n_coeffs).map(|j| format!("beta_{j}")).collect();
    SamplePayload {
        samples: nuts.samples,
        coefficient_names,
        posterior_mean: nuts.posterior_mean,
        posterior_std: nuts.posterior_std,
        rhat: nuts.rhat,
        ess: nuts.ess,
        converged: nuts.converged,
        config: SampleConfigPayload {
            n_samples: cfg.n_samples,
            n_warmup: cfg.nwarmup,
            n_chains: cfg.n_chains,
            target_accept: cfg.target_accept,
            seed: cfg.seed,
        },
        model_class: prediction_model_class_label(model),
        family_kind: family_link_kind(&model_likelihood_spec(&model)).to_string(),
        link_spec: model_link_spec_json(model),
        method: nuts_method_label(model).to_string(),
    }
}

#[derive(Serialize)]
struct CoefficientStatePayload {
    beta: Vec<f64>,
    covariance_flat: Vec<f64>,
    covariance_n: usize,
    #[serde(skip_serializing_if = "Option::is_none")]
    covariance_corrected_flat: Option<Vec<f64>>,
    #[serde(skip_serializing_if = "Option::is_none")]
    covariance_corrected_n: Option<usize>,
    schema: Option<DataSchema>,
    training_feature_ranges: Option<Vec<(f64, f64)>>,
    random_column_ranges: Vec<(usize, usize)>,
    coefficient_provenance: Vec<CoefficientProvenancePayload>,
    term_blocks: Vec<TermBlock>,
    #[serde(skip_serializing_if = "Option::is_none")]
    group_metadata: Option<GroupMetadata>,
}

#[derive(Serialize, Deserialize, Clone)]
struct TermBlock {
    name: String,
    kind: String,
    start: usize,
    end: usize,
}

#[derive(Deserialize)]
struct TermBlocksPayload {
    term_blocks: Vec<TermBlock>,
}

#[derive(Serialize)]
struct CoefficientProvenancePayload {
    index: usize,
    label: String,
    source: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    term: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    column: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    level: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    metadata: Option<serde_json::Value>,
}

fn categorical_level_name_for_bits(
    schema: Option<&DataSchema>,
    column_name: &str,
    bits: u64,
) -> Option<String> {
    let value = f64::from_bits(bits);
    if !value.is_finite() {
        return None;
    }
    let idx = value as usize;
    if (idx as f64 - value).abs() > 1e-12 {
        return None;
    }
    schema
        .and_then(|schema| {
            schema
                .columns
                .iter()
                .find(|column| column.name == column_name)
        })
        .and_then(|column| column.levels.get(idx))
        .cloned()
}

fn smooth_basis_kind_label(basis: &gam::terms::smooth::SmoothBasisSpec) -> &'static str {
    use gam::terms::smooth::SmoothBasisSpec as S;
    match basis {
        S::BSpline1D { .. } => "smooth_bspline1d",
        S::TensorBSpline { .. } => "tensor",
        S::ThinPlate { .. } => "thin_plate",
        S::Sphere { .. } => "sphere",
        S::ConstantCurvature { .. } => "constant_curvature",
        S::Matern { .. } => "matern",
        S::Duchon { .. } => "duchon",
        S::Pca { .. } => "pca",
        S::FactorSmooth { .. } => "factor_smooth",
        S::BySmooth { .. } => "by_smooth",
        S::ByVariable { .. } => "by_variable",
        S::FactorSumToZero { .. } => "factor_sum_to_zero",
        S::MeasureJet { .. } => "measurejet",
    }
}

/// Try to derive per-smooth-term column ranges by building a synthetic
/// two-row design from the saved training feature ranges. Returns
/// `(name, range)` pairs in global-column coordinates, or `None` if the
/// build fails (e.g. no training ranges available).
fn smooth_term_column_ranges(
    payload: &FittedModelPayload,
    smooth_start: usize,
) -> Option<Vec<(String, std::ops::Range<usize>)>> {
    let spec = payload.resolved_termspec.as_ref()?;
    if spec.smooth_terms.is_empty() {
        return Some(Vec::new());
    }
    let ranges = payload.training_feature_ranges.as_ref()?;
    let schema = payload.data_schema.as_ref()?;
    let ncols = schema.columns.len();
    if ranges.is_empty() || ncols == 0 {
        return None;
    }
    let mut data = Array2::<f64>::zeros((2, ncols));
    for (col, &(lo, hi)) in ranges.iter().take(ncols).enumerate() {
        let (lo, hi) = if lo.is_finite() && hi.is_finite() {
            (lo, hi)
        } else {
            (0.0, 1.0)
        };
        data[[0, col]] = lo;
        data[[1, col]] = hi;
    }
    let design = build_term_collection_design(data.view(), spec).ok()?;
    let mut out = Vec::with_capacity(design.smooth.terms.len());
    for term in &design.smooth.terms {
        let r = term.coeff_range.clone();
        let global = (smooth_start + r.start)..(smooth_start + r.end);
        out.push((term.name.clone(), global));
    }
    Some(out)
}

fn coefficient_provenance_for_state(
    payload: &FittedModelPayload,
    beta_len: usize,
) -> (Vec<CoefficientProvenancePayload>, Vec<TermBlock>) {
    let mut provenance = (0..beta_len)
        .map(|index| CoefficientProvenancePayload {
            index,
            label: "__global__".to_string(),
            source: "global".to_string(),
            term: None,
            column: None,
            level: None,
            metadata: None,
        })
        .collect::<Vec<_>>();
    let mut blocks: Vec<TermBlock> = Vec::new();

    let Some(spec) = payload.resolved_termspec.as_ref() else {
        return (provenance, blocks);
    };

    if !provenance.is_empty() {
        provenance[0].term = Some("intercept".to_string());
        provenance[0].label = "intercept".to_string();
        blocks.push(TermBlock {
            name: "intercept".to_string(),
            kind: "intercept".to_string(),
            start: 0,
            end: 1,
        });
    }

    for (offset, term) in spec.linear_terms.iter().enumerate() {
        let index = 1 + offset;
        if let Some(entry) = provenance.get_mut(index) {
            entry.term = Some(term.name.clone());
            entry.column = Some(term.name.clone());
            entry.label = term.name.clone();
            entry.source = "linear".to_string();
        }
        blocks.push(TermBlock {
            name: term.name.clone(),
            kind: "linear".to_string(),
            start: index,
            end: index + 1,
        });
    }

    let mut col = 1 + spec.linear_terms.len();
    for term in &spec.random_effect_terms {
        let levels = term.frozen_levels.as_deref().unwrap_or(&[]);
        let block_start = col;
        for (local, bits) in levels.iter().copied().enumerate() {
            let index = col + local;
            let label =
                categorical_level_name_for_bits(payload.data_schema.as_ref(), &term.name, bits)
                    .unwrap_or_else(|| f64::from_bits(bits).to_string());
            if let Some(entry) = provenance.get_mut(index) {
                entry.label = label.clone();
                entry.source = "group".to_string();
                entry.term = Some(term.name.clone());
                entry.column = Some(term.name.clone());
                entry.level = Some(label.clone());
                entry.metadata = payload
                    .group_metadata
                    .as_ref()
                    .and_then(|metadata| metadata.get(&label))
                    .cloned();
            }
        }
        col += levels.len();
        if col > block_start {
            blocks.push(TermBlock {
                name: term.name.clone(),
                kind: "random_effect".to_string(),
                start: block_start,
                end: col,
            });
        }
    }

    // Smooth terms: derive per-term column widths by building a synthetic
    // design from training_feature_ranges. If that fails (older payloads
    // without saved ranges, or unusual basis variants), the columns simply
    // keep their default `__global__` labels.
    if !spec.smooth_terms.is_empty() {
        let smooth_start = col;
        if let Some(smooth_ranges) = smooth_term_column_ranges(payload, smooth_start) {
            for ((name, range), term_spec) in smooth_ranges.iter().zip(spec.smooth_terms.iter()) {
                let kind = smooth_basis_kind_label(&term_spec.basis);
                for idx in range.clone() {
                    if let Some(entry) = provenance.get_mut(idx) {
                        entry.term = Some(name.clone());
                        entry.column = Some(name.clone());
                        entry.label = format!("{}[{}]", name, idx - range.start);
                        entry.source = "smooth".to_string();
                    }
                }
                blocks.push(TermBlock {
                    name: name.clone(),
                    kind: kind.to_string(),
                    start: range.start,
                    end: range.end,
                });
            }
        }
    }

    blocks.sort_by_key(|block| block.start);
    (provenance, blocks)
}

fn coefficient_state_json_impl(model_bytes: &[u8]) -> Result<String, String> {
    let model = load_model_impl(model_bytes)?;
    // A scan-routed model has no dense coefficient covariance to export: the
    // exact O(n) smoother keeps only the per-knot posterior, and its natural
    // parameter count is ~n, so the dense Gram this payload carries is both
    // unavailable and (at O(n²)) antithetical to the representation. The
    // cross-fit precision-group workflow that consumes this is the dense path;
    // surface a precise, actionable error instead of the cryptic
    // missing-fit_result one (#1046). Introspection that does NOT need the dense
    // covariance — summary(), smoothing_parameters(), evidence(), term_blocks()
    // — is served directly from the scan state.
    if let Some(scan) = scan_introspection(&model)? {
        return Err(format!(
            "{} is fit by the exact O(n) state-space spline scan, which retains \
             the per-knot posterior rather than a dense coefficient covariance; \
             the dense coefficient state is unavailable. Use summary(), \
             smoothing_parameters(), evidence(), or term_blocks() for this \
             model's fitted quantities, or refit with double_penalty=true to \
             obtain a dense coefficient state.",
            scan_smooth_label(&scan)
        ));
    }
    let payload = model.payload();
    let fit = fit_result_from_saved_model_for_prediction(&model)?;
    let cov = fit
        .beta_covariance()
        .ok_or_else(|| "model does not contain coefficient covariance; refit with covariance-saving inference enabled".to_string())?;
    let (covariance_corrected_flat, covariance_corrected_n) = match fit.beta_covariance_corrected()
    {
        Some(c) => (Some(c.iter().copied().collect()), Some(c.nrows())),
        None => (None, None),
    };
    let mut random_ranges = Vec::<(usize, usize)>::new();
    if let Some(spec) = payload.resolved_termspec.as_ref() {
        let mut col = 1 + spec.linear_terms.len();
        for re in &spec.random_effect_terms {
            let n = re.frozen_levels.as_ref().map(|v| v.len()).unwrap_or(0);
            random_ranges.push((col, col + n));
            col += n;
        }
    }
    let (coefficient_provenance, term_blocks) =
        coefficient_provenance_for_state(payload, fit.beta.len());
    let out = CoefficientStatePayload {
        beta: fit.beta.to_vec(),
        covariance_flat: cov.iter().copied().collect(),
        covariance_n: cov.nrows(),
        covariance_corrected_flat,
        covariance_corrected_n,
        schema: payload.data_schema.clone(),
        training_feature_ranges: payload.training_feature_ranges.clone(),
        random_column_ranges: random_ranges,
        coefficient_provenance,
        term_blocks,
        group_metadata: payload.group_metadata.clone(),
    };
    serde_json::to_string(&out)
        .map_err(|err| format!("failed to serialize coefficient state: {err}"))
}

fn term_blocks_for_model_impl(
    model_bytes: &[u8],
) -> Result<Vec<(String, String, usize, usize)>, String> {
    // A scan-routed model has a single smooth term occupying the smoother's
    // entire coefficient space (its per-knot function values). Report that one
    // contiguous block directly — without round-tripping through
    // `coefficient_state_json_impl`, which a scan model cannot satisfy (it keeps
    // no dense coefficient covariance) and which would be O(n²) even if it
    // could (#1046).
    {
        let model = load_model_impl(model_bytes)?;
        if let Some(scan) = scan_introspection(&model)? {
            return Ok(vec![(
                scan_smooth_label(&scan),
                "smooth".to_string(),
                0,
                scan.n_knots,
            )]);
        }
    }
    let state_json = coefficient_state_json_impl(model_bytes)?;
    let payload: TermBlocksPayload = serde_json::from_str(&state_json)
        .map_err(|err| format!("failed to parse coefficient state json: {err}"))?;
    let mut blocks = Vec::with_capacity(payload.term_blocks.len());
    for (idx, block) in payload.term_blocks.into_iter().enumerate() {
        if block.end < block.start {
            return Err(format!(
                "term block {idx} has invalid range [{}, {})",
                block.start, block.end
            ));
        }
        blocks.push((block.name, block.kind, block.start, block.end));
    }
    Ok(blocks)
}

#[pyfunction]
fn build_difference_smooth_request_json(
    view: &str,
    group: Option<String>,
    pairs: Option<Vec<(String, String)>>,
    n: usize,
    level: Option<f64>,
    simultaneous: bool,
    n_sim: Option<usize>,
    seed: Option<u64>,
    marginalise_random: bool,
    group_means: bool,
    template: Option<HashMap<String, String>>,
) -> PyResult<String> {
    let payload = gam::inference::difference_smooth::DifferenceSmoothRequest {
        view: view.to_string(),
        group,
        pairs,
        n,
        level,
        simultaneous,
        n_sim,
        seed,
        marginalise_random,
        group_means,
        template: template.map(|values| values.into_iter().collect()),
    };
    serde_json::to_string(&payload).map_err(|err| {
        py_value_error(format!(
            "failed to serialize difference_smooth request json: {err}"
        ))
    })
}

fn difference_smooth_json_impl(model_bytes: &[u8], request_json: &str) -> Result<String, String> {
    let request: gam::inference::difference_smooth::DifferenceSmoothRequest =
        serde_json::from_str(request_json)
            .map_err(|err| format!("failed to parse difference_smooth request json: {err}"))?;
    let model = load_model_impl(model_bytes)?;
    let fit = fit_result_from_saved_model_for_prediction(&model)?;
    let selected_covariance = gam::inference::effects::select_covariance(
        &fit,
        gam::inference::effects::CovarianceSource::SmoothingCorrected,
    )
    .map_err(|error| error.to_string())?;
    let payload = model.payload();
    let schema = payload
        .data_schema
        .as_ref()
        .ok_or_else(|| "difference_smooth requires a saved model schema".to_string())?;
    let training_feature_ranges = payload
        .training_feature_ranges
        .as_deref()
        .ok_or_else(|| "difference_smooth requires saved training feature ranges".to_string())?;
    let termspec = payload.resolved_termspec.as_ref().ok_or_else(|| {
        "difference_smooth requires a saved resolved term specification".to_string()
    })?;
    let rows = gam::inference::difference_smooth::difference_smooth_report(
        gam::inference::difference_smooth::DifferenceSmoothInputs {
            schema,
            training_feature_ranges,
            termspec,
            beta: fit.beta.view(),
            covariance: selected_covariance.matrix,
            covariance_source: selected_covariance.source,
        },
        request,
        |headers, rows| {
            let dataset = dataset_with_model_schema(&model, headers, rows)?;
            standard_mean_design_dense(&model, dataset)
        },
    )?;
    serde_json::to_string(&rows)
        .map_err(|err| format!("failed to serialize difference_smooth rows: {err}"))
}

fn json_f64_vec(value: &serde_json::Value, key: &str) -> Result<Vec<f64>, String> {
    let values = value
        .get(key)
        .and_then(|raw| raw.as_array())
        .ok_or_else(|| format!("model coefficient state does not include {key}"))?;
    values
        .iter()
        .enumerate()
        .map(|(idx, raw)| {
            raw.as_f64()
                .filter(|value| value.is_finite())
                .ok_or_else(|| format!("{key}[{idx}] must be finite numeric"))
        })
        .collect()
}

fn json_label_text(value: &serde_json::Value) -> Option<String> {
    match value {
        serde_json::Value::String(text) => Some(text.clone()),
        serde_json::Value::Number(number) => Some(number.to_string()),
        serde_json::Value::Bool(flag) => Some(flag.to_string()),
        _ => None,
    }
}

fn coefficient_indices_for_precision_label_json(
    state: &serde_json::Value,
    label: &str,
) -> Result<Vec<usize>, String> {
    let provenance = state
        .get("coefficient_provenance")
        .and_then(|raw| raw.as_array())
        .ok_or_else(|| {
            "model coefficient state does not include coefficient provenance".to_string()
        })?;
    let mut matches = Vec::<usize>::new();
    for (fallback_index, item) in provenance.iter().enumerate() {
        let Some(obj) = item.as_object() else {
            continue;
        };
        let index = obj
            .get("index")
            .and_then(|raw| raw.as_u64())
            .map(|raw| raw as usize)
            .unwrap_or(fallback_index);
        let matched = ["term", "column", "label"].into_iter().any(|key| {
            obj.get(key)
                .and_then(json_label_text)
                .is_some_and(|candidate| candidate == label)
        });
        if matched {
            matches.push(index);
        }
    }
    Ok(matches)
}

fn cross_fit_shared_precision_groups_json_impl(request_json: &str) -> Result<String, String> {
    let request: PySharedPrecisionRequest = serde_json::from_str(request_json)
        .map_err(|err| format!("failed to parse shared precision request json: {err}"))?;
    if request.models.is_empty() {
        return Err("at least one model is required".to_string());
    }
    if request.groups.is_empty() {
        return Err("at least one shared precision group is required".to_string());
    }
    let states = request
        .models
        .iter()
        .map(|model| {
            serde_json::from_str::<serde_json::Value>(&model.state_json).map_err(|err| {
                format!(
                    "failed to parse coefficient state for model {}: {err}",
                    model.key
                )
            })
        })
        .collect::<Result<Vec<_>, _>>()?;
    let mut result = serde_json::Map::<String, serde_json::Value>::new();
    for group in request.groups {
        if group.labels.len() != request.models.len() {
            return Err(format!(
                "shared precision group {:?} has {} labels for {} model(s)",
                group.name,
                group.labels.len(),
                request.models.len()
            ));
        }
        if !(group.shape.is_finite() && group.shape > 0.0) {
            return Err(format!(
                "shared precision group {:?} requires finite shape > 0",
                group.name
            ));
        }
        if !(group.rate.is_finite() && group.rate >= 0.0) {
            return Err(format!(
                "shared precision group {:?} requires finite rate >= 0",
                group.name
            ));
        }
        let mut fit_entries = Vec::<serde_json::Value>::new();
        let mut dims = BTreeSet::<usize>::new();
        let mut quadratic_sum = 0.0;
        for ((model, state), label) in request
            .models
            .iter()
            .zip(states.iter())
            .zip(group.labels.iter())
        {
            let indices = coefficient_indices_for_precision_label_json(state, label)?;
            if indices.is_empty() {
                continue;
            }
            let beta = json_f64_vec(state, "beta")?;
            let cov_n = state
                .get("covariance_n")
                .and_then(|raw| raw.as_u64())
                .ok_or_else(|| {
                    "model coefficient state does not include covariance_n".to_string()
                })? as usize;
            let cov_flat = json_f64_vec(state, "covariance_flat")?;
            if beta.len() != cov_n || cov_flat.len() != cov_n * cov_n {
                return Err(format!(
                    "model {} has inconsistent beta/covariance dimensions",
                    model.key
                ));
            }
            if indices.iter().any(|&index| index >= cov_n) {
                return Err(format!(
                    "model {} has coefficient provenance outside covariance bounds",
                    model.key
                ));
            }
            let trace_covariance = indices
                .iter()
                .map(|&index| cov_flat[index * cov_n + index])
                .sum::<f64>();
            let beta_norm_sq = indices
                .iter()
                .map(|&index| beta[index] * beta[index])
                .sum::<f64>();
            let contribution = beta_norm_sq + trace_covariance;
            if !contribution.is_finite() {
                return Err(format!(
                    "shared precision group {:?} has non-finite contribution in model {}",
                    group.name, model.key
                ));
            }
            quadratic_sum += contribution;
            dims.insert(indices.len());
            fit_entries.push(serde_json::json!({
                "model": model.key,
                "label": label,
                "coefficient_indices": indices,
                "dimension": indices.len(),
                "beta_norm_sq": beta_norm_sq,
                "trace_covariance": trace_covariance,
                "quadratic_contribution": contribution,
            }));
        }
        if fit_entries.is_empty() {
            return Err(format!(
                "shared precision group {:?} did not match any model coefficients",
                group.name
            ));
        }
        if dims.len() != 1 {
            return Err(format!(
                "shared precision group {:?} matched inconsistent dimensions: {:?}",
                group.name,
                dims.into_iter().collect::<Vec<_>>()
            ));
        }
        let dimension = *dims.iter().next().expect("dimension checked above");
        let numerator = fit_entries.len() as f64 * dimension as f64 + 2.0 * (group.shape - 1.0);
        let denominator = quadratic_sum + 2.0 * group.rate;
        if numerator <= 0.0 {
            return Err(format!(
                "shared precision group {:?} has non-positive MAP numerator",
                group.name
            ));
        }
        if denominator <= 0.0 || !denominator.is_finite() {
            return Err(format!(
                "shared precision group {:?} has non-positive/non-finite denominator",
                group.name
            ));
        }
        let lambda = numerator / denominator;
        result.insert(
            group.name.clone(),
            serde_json::json!({
                "lambda": lambda,
                "log_lambda": lambda.ln(),
                "shape": group.shape,
                "rate": group.rate,
                "n_fits": fit_entries.len(),
                "dimension": dimension,
                "quadratic_sum": quadratic_sum,
                "numerator": numerator,
                "denominator": denominator,
                "fits": fit_entries,
            }),
        );
    }
    serde_json::to_string(&result)
        .map_err(|err| format!("failed to serialize shared precision result: {err}"))
}

/// Collect every column that the frozen spec treats as a CATEGORICAL FACTOR,
/// together with the exact bit patterns of its frozen levels.
///
/// A factor-smooth design (`fs`/`re` → `FactorSumToZero`/`FactorSmooth`, a
/// factor `by=` → `ByVariable`/`BySmooth`, or a categorical tensor margin) gates
/// its per-level blocks on `canonical_level_bits(value) == level_bits` against
/// the saved levels: a column value that is not a (signed-zero/NaN-canonical)
/// match for a frozen level matches no block. The summary's representative data
/// (axis-spanning midpoints) therefore
/// fabricates illegal factor values, and rebuilding the factor design on it
/// fails — which is why `summary().smooth_terms` came back empty for ANY model
/// containing a factor-smooth, dragging co-fitted `s(x)` rows down with it
/// (#1370). Returning the real frozen levels here lets the caller place valid,
/// bit-exact levels in those columns so the design rebuilds.
///
/// The map is `col -> sorted, de-duplicated level bit patterns`. Only columns
/// with at least one known frozen level are reported; numeric axes are absent.
fn frozen_factor_levels_by_col(
    spec: &gam::terms::smooth::TermCollectionSpec,
) -> std::collections::BTreeMap<usize, Vec<u64>> {
    use gam::terms::smooth::{ByVarKind, ByVariableSpec, SmoothBasisSpec};

    let mut levels: std::collections::BTreeMap<usize, std::collections::BTreeSet<u64>> =
        std::collections::BTreeMap::new();
    let mut record = |col: usize, bits: &[u64]| {
        if bits.is_empty() {
            return;
        }
        levels.entry(col).or_default().extend(bits.iter().copied());
    };

    // Walk the (possibly nested) basis, accumulating factor columns. The DSL
    // wraps the geometric core in `ByVariable`/`FactorSumToZero`/`BySmooth`
    // envelopes, so recurse through them exactly like `smooth_basis_feature_cols`.
    fn walk(basis: &SmoothBasisSpec, record: &mut dyn FnMut(usize, &[u64])) {
        match basis {
            SmoothBasisSpec::FactorSumToZero {
                inner,
                by_col,
                levels,
                ..
            } => {
                record(*by_col, levels);
                walk(inner, record);
            }
            SmoothBasisSpec::ByVariable {
                inner, by_col, by, ..
            } => {
                if let ByVariableSpec::Level { value_bits, .. } = by {
                    record(*by_col, &[*value_bits]);
                }
                walk(inner, record);
            }
            SmoothBasisSpec::BySmooth { smooth, by_kind } => {
                if let ByVarKind::Factor {
                    feature_col,
                    frozen_levels: Some(frozen),
                    ..
                } = by_kind
                {
                    record(*feature_col, frozen);
                }
                walk(smooth, record);
            }
            SmoothBasisSpec::FactorSmooth { spec } => {
                if let Some(frozen) = &spec.group_frozen_levels {
                    record(spec.group_col, frozen);
                }
            }
            _ => {}
        }
    }

    for term in &spec.smooth_terms {
        walk(&term.basis, &mut record);
    }

    levels
        .into_iter()
        .map(|(col, set)| (col, set.into_iter().collect()))
        .collect()
}

/// Synthesize a small representative data matrix from saved per-axis training
/// ranges, used only to rebuild the design *structure* (per-term coefficient
/// ranges, nullspace dimensions, penalty counts) for the summary smooth-term
/// table. The basis layout these fields describe is fixed by the frozen
/// `resolved_termspec`, not by the data values, so axis-spanning midpoints
/// reproduce the training-time block layout deterministically while remaining
/// inside the training bounding box (no extrapolation artefacts).
///
/// Columns named in `factor_levels` are categorical factors (a `by=` factor or a
/// factor-smooth group): their values must be bit-identical to a frozen level or
/// the design rebuild fails (#1370). Those columns are filled by cycling through
/// the exact frozen level bit patterns instead of axis midpoints, so every level
/// (hence every per-level deviation block) is present at least once and the
/// `fs`/`sz` design rebuilds cleanly. The row count grows to cover the widest
/// factor so no level is dropped.
fn representative_data_from_ranges(
    ranges: &[(f64, f64)],
    factor_levels: &std::collections::BTreeMap<usize, Vec<u64>>,
) -> Array2<f64> {
    const REP_ROWS_MIN: usize = 16;
    // Enough rows that the widest factor has every level represented.
    let max_levels = factor_levels.values().map(|lv| lv.len()).max().unwrap_or(0);
    let rep_rows = REP_ROWS_MIN.max(max_levels).max(1);
    let n_cols = ranges.len();
    let mut data = Array2::<f64>::zeros((rep_rows, n_cols));
    for (col, &(lo, hi)) in ranges.iter().enumerate() {
        if let Some(lv) = factor_levels.get(&col) {
            // Categorical column: cycle through the frozen level bit patterns so
            // every level appears and each value is a valid, bit-exact level.
            if !lv.is_empty() {
                for row in 0..rep_rows {
                    data[[row, col]] = f64::from_bits(lv[row % lv.len()]);
                }
                continue;
            }
        }
        let (lo, hi) = if lo.is_finite() && hi.is_finite() && hi >= lo {
            (lo, hi)
        } else {
            (0.0, 1.0)
        };
        for row in 0..rep_rows {
            let frac = if rep_rows > 1 {
                row as f64 / (rep_rows - 1) as f64
            } else {
                0.5
            };
            data[[row, col]] = lo + frac * (hi - lo);
        }
    }
    data
}

/// Dense, space-filling reconstruction of the training inputs for the Wald
/// design-whitening Gram (#2142). Denser than `representative_data_from_ranges`
/// so a high-basis univariate smooth gets a full-rank Gram, and with each
/// continuous column swept in an independent coprime order so multivariate
/// (tensor) margins are not collinear on the diagonal — the shared-ramp
/// representative grid samples only the diagonal line and would make every
/// tensor Gram rank-deficient. `rows` is forced to a power of two so every odd
/// per-column stride is coprime to it and therefore traverses the full
/// evenly-spaced grid. Categorical columns keep cycling their frozen levels.
fn whitening_data_from_ranges(
    ranges: &[(f64, f64)],
    factor_levels: &std::collections::BTreeMap<usize, Vec<u64>>,
    rows: usize,
) -> Array2<f64> {
    let rows = rows.max(2);
    let n_cols = ranges.len();
    let mut data = Array2::<f64>::zeros((rows, n_cols));
    for (col, &(lo, hi)) in ranges.iter().enumerate() {
        if let Some(lv) = factor_levels.get(&col) {
            if !lv.is_empty() {
                for row in 0..rows {
                    data[[row, col]] = f64::from_bits(lv[row % lv.len()]);
                }
                continue;
            }
        }
        let (lo, hi) = if lo.is_finite() && hi.is_finite() && hi >= lo {
            (lo, hi)
        } else {
            (0.0, 1.0)
        };
        // Odd stride is coprime to the power-of-two `rows`, so `(row*stride) %
        // rows` is a full-period permutation of the evenly-spaced grid — a
        // different one per column, breaking the diagonal collinearity.
        let stride = 2 * col + 1;
        for row in 0..rows {
            let idx = row.wrapping_mul(stride) % rows;
            let frac = idx as f64 / (rows - 1) as f64;
            data[[row, col]] = lo + frac * (hi - lo);
        }
    }
    data
}

/// Reconstruct the design-whitening Gram `X'X` for the summary Wald smooth test
/// from the frozen basis (#2142). The persisted summary path drops the fit's
/// inference block, so the exact weighted Gram `X'WX` is gone; mgcv itself
/// whitens the Wood (2013) statistic with the *unweighted* prediction-matrix
/// Gram, so `X'X` at representative inputs is the intended object (and it
/// reduces to `X'WX` for the Gaussian identity case). Returns the full `p×p`
/// Gram in the trained coefficient layout, or `None` when the rebuilt design's
/// column count does not match the trained coefficient count — a stale/mismatched
/// spec, in which case the test falls back to the un-whitened raw covariance.
fn summary_whitening_gram(
    spec: &gam::terms::smooth::TermCollectionSpec,
    ranges: &[(f64, f64)],
    factor_levels: &std::collections::BTreeMap<usize, Vec<u64>>,
    expected_ncols: usize,
) -> Option<Array2<f64>> {
    if expected_ncols == 0 {
        return None;
    }
    let rows = (4 * expected_ncols).max(64).next_power_of_two();
    let data = whitening_data_from_ranges(ranges, factor_levels, rows);
    let design = gam::terms::smooth::build_term_collection_design(data.view(), spec).ok()?;
    let x = design.design.to_dense();
    if x.ncols() != expected_ncols {
        return None;
    }
    // Lower triangle of `X'X` is the true Gram; the whitening eigendecomposition
    // reads only that side, so no explicit symmetrization is needed.
    Some(x.t().dot(&x))
}

#[cfg(test)]
mod whitening_gram_tests {
    //! Direct tests of the #2142 design-whitening-Gram reconstruction grid used
    //! when a summary is built from an inference-stripped (compact) model. The
    //! whitening math itself is covered by `gam-terms` `smooth_test` tests; here
    //! we only verify the reconstruction *inputs* are non-degenerate.
    use super::whitening_data_from_ranges;
    use std::collections::BTreeMap;

    #[test]
    fn dense_grid_spans_range_and_breaks_diagonal_collinearity() {
        let ranges = [(0.0_f64, 1.0_f64), (-2.0, 4.0)];
        let levels = BTreeMap::new();
        let data = whitening_data_from_ranges(&ranges, &levels, 64);
        assert_eq!(data.nrows(), 64);
        assert_eq!(data.ncols(), 2);
        // Each continuous column sweeps its full [lo, hi] range.
        for (c, &(lo, hi)) in ranges.iter().enumerate() {
            let col = data.column(c);
            let cmin = col.iter().cloned().fold(f64::INFINITY, f64::min);
            let cmax = col.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
            assert!((cmin - lo).abs() < 1e-9, "col {c} min {cmin} != {lo}");
            assert!((cmax - hi).abs() < 1e-9, "col {c} max {cmax} != {hi}");
        }
        // The shared-ramp representative grid puts both columns on the same
        // diagonal (Pearson r == 1), collapsing every tensor Gram. The
        // independent coprime sweeps must break that: |r| strictly below 1.
        let c0 = data.column(0);
        let c1 = data.column(1);
        let m0 = c0.mean().unwrap();
        let m1 = c1.mean().unwrap();
        let (mut cov, mut v0, mut v1) = (0.0, 0.0, 0.0);
        for i in 0..64 {
            let (a, b) = (c0[i] - m0, c1[i] - m1);
            cov += a * b;
            v0 += a * a;
            v1 += b * b;
        }
        let r = cov / (v0.sqrt() * v1.sqrt());
        assert!(
            r.abs() < 0.9,
            "columns must not be collinear (diagonal grid), got r={r}"
        );
    }

    #[test]
    fn categorical_columns_cycle_every_frozen_level() {
        let ranges = [(0.0_f64, 1.0_f64), (0.0, 0.0)];
        let mut levels = BTreeMap::new();
        let lv = vec![1.0_f64.to_bits(), 2.0_f64.to_bits(), 3.0_f64.to_bits()];
        levels.insert(1usize, lv.clone());
        let data = whitening_data_from_ranges(&ranges, &levels, 64);
        let allowed: Vec<f64> = lv.iter().map(|&b| f64::from_bits(b)).collect();
        for i in 0..64 {
            let v = data[[i, 1]];
            assert!(
                allowed.iter().any(|&a| a == v),
                "row {i} value {v} is not a frozen level"
            );
        }
        for &a in &allowed {
            assert!(
                (0..64).any(|i| data[[i, 1]] == a),
                "frozen level {a} never appears"
            );
        }
    }
}

/// Build the mgcv-style per-smooth significance table for the FFI summary.
///
/// Mirrors `main.rs::build_model_summary`'s smooth-term loop: random-effect
/// smooths report `edf` only (their boundary variance-component test is not a
/// Wald χ²); penalized smooth terms get the Wood (2013) rank-truncated Wald
/// statistic and p-value from [`gam::inference::smooth_test::wood_smooth_test`].
///
/// Returns an empty vector (rather than erroring) when the design cannot be
/// rebuilt — e.g. a model saved without `resolved_termspec` or training feature
/// ranges — so `summary()` always succeeds and simply omits the table when the
/// information needed to compute it honestly is not available.
fn summary_smooth_terms(
    model: &FittedModel,
    fit: &gam::solver::estimate::UnifiedFitResult,
) -> Vec<SummarySmoothTermRow> {
    use gam::inference::smooth_test::{SmoothTestInput, SmoothTestScale, wood_smooth_test};
    use gam::terms::smooth::ShapeConstraint;

    let payload = model.payload();
    let Some(spec) = payload.resolved_termspec.as_ref() else {
        return Vec::new();
    };
    if spec.validate_frozen("resolved_termspec").is_err() {
        return Vec::new();
    }
    let Some(ranges) = payload.training_feature_ranges.as_ref() else {
        return Vec::new();
    };
    let Some(headers) = payload.training_headers.as_ref() else {
        return Vec::new();
    };
    if ranges.len() != headers.len() {
        return Vec::new();
    }
    // Categorical columns (factor-smooth groups / factor `by=`) must carry valid
    // frozen levels or the design rebuild fails — and that failure was being
    // swallowed to an EMPTY table for the whole model, erasing co-fitted `s(x)`
    // rows too (#1370). Synthesize the representative data with the real frozen
    // levels in those columns so `fs`/`sz` designs replay cleanly.
    let factor_levels = frozen_factor_levels_by_col(spec);
    let data = representative_data_from_ranges(ranges, &factor_levels);
    let Ok(design) = gam::terms::smooth::build_term_collection_design(data.view(), spec) else {
        return Vec::new();
    };

    // The Wald smooth test uses the CONDITIONAL Bayesian covariance
    // `Vb = H⁻¹·φ̂` (mgcv's `Vp`, the covariance mgcv's `testStat` whitens by
    // default), NOT the smoothing-parameter-corrected `Vc`. `Vc` adds the λ̂
    // uncertainty `(∂β/∂ρ)·Cov(ρ)·(∂β/∂ρ)ᵀ`, whose variance concentrates in the
    // wiggle directions (those are the ones λ controls). For a heavily-smoothed,
    // near-linear term that inflation can exceed the linear direction's variance
    // and flip the whitened eigenvalue ordering, so the rank-`round(edf)`
    // truncation keeps a wiggle mode where β̂≈0 and reports the term
    // non-significant even though its linear effect is real (#2142). The
    // conditional covariance is the correct hypothesis-test object; `Vc` is for
    // prediction/credible bands.
    let cov_forwald = fit
        .beta_covariance()
        .or_else(|| fit.beta_covariance_corrected());
    // Wood (2013) design-whitening metric for the Wald smooth test (#2142).
    // Prefer the fit's exact weighted Gram `X'WX` when the inference block
    // survived; on the persisted summary path (inference dropped) reconstruct
    // the unweighted `X'X` from the frozen basis. `None` → un-whitened fallback.
    let reconstructed_gram = if fit.weighted_gram().is_none() {
        summary_whitening_gram(spec, ranges, &factor_levels, design.design.ncols())
    } else {
        None
    };
    let whitening_gram_full: Option<&Array2<f64>> =
        fit.weighted_gram().or(reconstructed_gram.as_ref());
    let family = model.likelihood();
    let scale_is_estimated = matches!(
        family.response,
        ResponseFamily::Gaussian | ResponseFamily::Gamma
    );
    // n (for the F-distribution denominator) comes from the saved working-set
    // geometry when present; the Wald χ² (Known-scale) branch never reads it.
    let n_obs = fit
        .geometry
        .as_ref()
        .map(|geom| geom.working_response.len() as f64);
    let residual_df = n_obs.zip(fit.edf_total()).and_then(|(n, edf)| {
        let value = n - edf;
        (edf.is_finite() && value.is_finite() && value > 0.0).then_some(value)
    });
    let scale = if scale_is_estimated {
        SmoothTestScale::Estimated
    } else {
        SmoothTestScale::Known
    };

    let mut out = Vec::<SummarySmoothTermRow>::new();
    // The fit's GLOBAL penalty layout (and thus `penalty_block_trace`) opens with a
    // single shared `LinearTermRidge` block IFF any linear term has
    // `double_penalty=true` (`design_construction.rs`). Random-effect and smooth
    // penalty blocks follow it. Seeding `penalty_cursor` at 0 ignored that leading
    // block, sliding every per-term trace window off by one whenever a penalized
    // linear term was present; on this persisted / column-conditioned path `F` is
    // nulled, so `per_term_edf` falls back to the `penalty_block_trace` window and
    // the off-by-one corrupts every per-term EDF (#1372). Start the cursor PAST any
    // leading `LinearTermRidge` block by counting it in the recorded global ordering
    // rather than re-deriving it.
    let mut penalty_cursor = design
        .penaltyinfo
        .iter()
        .filter(|info| {
            matches!(
                &info.penalty.source,
                gam::basis::PenaltySource::Other(s) if s == "LinearTermRidge"
            )
        })
        .count();
    for (re_idx, (name, range)) in design.random_effect_ranges.iter().enumerate() {
        // Per-term EDF as the influence-matrix trace over the term's coefficient
        // block (#1219, #1277) — never the legacy per-block-EDF sum, which
        // double-counts shared coefficients and can exceed the model total.
        //
        // Only PENALIZED, non-empty RE blocks own an entry in the flat
        // `lambdas`/`penalty_block_trace`/`edf_by_block` layout: design assembly
        // (`design_construction.rs`) `continue`s its RE-penalty loop on
        // `range.is_empty() || !penalized`. Advancing the cursor by a fixed 1 per
        // RE term (the #1368 defect — fixed on the in-process `model_summary.rs`
        // path but never propagated here) slides `penalty_cursor` one block past
        // every RE/smooth term that follows an UNPENALIZED RE block (e.g. the
        // treatment-coded factor main effect a `by=` smooth injects) or an empty
        // (zero-kept-group) one, so the trailing smooth's `cursor..+k` window runs
        // off the end of `penalty_block_trace`, `per_term_edf` returns 0, the Wood
        // test is skipped, and ref_df/chi_sq/p_value collapse to 0/None on the
        // Python `summary()` path. Mirror BOTH design conditions.
        let penalized = spec
            .random_effect_terms
            .get(re_idx)
            .map(|t| t.penalized)
            .unwrap_or(true);
        let k_pen = usize::from(penalized && !range.is_empty());
        let edf = fit.per_term_edf(range.clone(), penalty_cursor, k_pen);
        penalty_cursor += k_pen;
        // Random-effect smooths are boundary variance-component tests; a naive
        // coefficient Wald χ² is anti-conservative, so only EDF is reported.
        out.push(SummarySmoothTermRow {
            name: name.clone(),
            edf,
            ref_df: edf.max(0.0),
            chi_sq: None,
            p_value: None,
        });
    }
    // `SmoothTerm::coeff_range` is block-local; the global coefficient layout is
    // [intercept | linear | random | smooth], so each block must be shifted by
    // `smooth_start` before indexing the global `fit.beta` / covariance /
    // influence matrix. The rebuilt design replays the frozen basis, so its
    // column counts and `smooth_start` match the trained fit exactly; only this
    // offset (omitted in the #1360 defect) was missing.
    let smooth_start = design
        .design
        .ncols()
        .saturating_sub(design.smooth.total_smooth_cols());
    for term in &design.smooth.terms {
        let k = term.active_penalties.len();
        // Per-term EDF as the influence-matrix trace over the term's coefficient
        // block, NOT the legacy `Σ_kk edf_by_block` per-penalty sum. For a tensor
        // product `te`/`ti` (and anisotropic / adaptive smooths) several penalty
        // blocks span the SAME shared coefficient range, so the block-sum
        // double-counts and reports a per-term EDF exceeding the model total and
        // the design column count (#1219 fixed the in-process summary; #1277 is
        // this persisted-model path the Python API reads via `summary()`).
        let global_range =
            (smooth_start + term.coeff_range.start)..(smooth_start + term.coeff_range.end);
        let edf = fit.per_term_edf(global_range.clone(), penalty_cursor, k);
        penalty_cursor += k;
        let smooth_test = if term.shape == ShapeConstraint::None {
            cov_forwald.and_then(|cov| {
                // The summary table is built from representative inputs reconstructed
                // from saved feature ranges (not the original training rows).
                wood_smooth_test(SmoothTestInput {
                    beta: fit.beta.view(),
                    covariance: cov,
                    influence_matrix: fit.coefficient_influence(),
                    // Wood (2013) design-whitening Gram in the original
                    // coefficient basis (#2142): exact `X'WX` when available, else
                    // the frozen-basis `X'X` reconstruction. Without it the
                    // rank-r truncation keeps the wrong eigen-subspace and a
                    // dominant wiggly smooth reads as non-significant.
                    whitening_gram: whitening_gram_full,
                    coeff_range: global_range.clone(),
                    edf,
                    nullspace_dim: term.wald_unpenalized_dim(),
                    residual_df,
                    scale,
                })
            })
        } else {
            None
        };
        let chi_sq = smooth_test.as_ref().map(|test| test.statistic);
        let ref_df = smooth_test
            .as_ref()
            .map(|test| test.ref_df)
            .unwrap_or(edf.max(0.0));
        let p_value = smooth_test.as_ref().map(|test| test.p_value);
        out.push(SummarySmoothTermRow {
            name: term.name.clone(),
            edf,
            ref_df,
            chi_sq,
            p_value,
        });
    }
    out
}

/// Read the fitted κ̂ off every `curv(...)` constant-curvature smooth in the
/// resolved (fitted) spec (#944). κ̂ is the outer optimiser's argmin of the
/// profiled criterion over κ; it lives in the basis spec the fit wrote back, so
/// surfacing it is a pure read — no refit, no original data needed. The verdict
/// here is the point-estimate sign tag only; the level-α geometry decision
/// (and the κ = 0 flatness p-value) is the profile-CI from
/// `curvature_inference_json`, which re-profiles `V_p(κ)` against the data.
fn summary_curvature_estimands(model: &FittedModel) -> Vec<SummaryCurvatureRow> {
    use gam::terms::smooth::SmoothBasisSpec;
    let payload = model.payload();
    let Some(spec) = payload.resolved_termspec.as_ref() else {
        return Vec::new();
    };
    let mut out = Vec::<SummaryCurvatureRow>::new();
    for (term_idx, term) in spec.smooth_terms.iter().enumerate() {
        let SmoothBasisSpec::ConstantCurvature { spec: cc, .. } = &term.basis else {
            continue;
        };
        if !cc.kappa.is_finite() {
            continue;
        }
        // Sign-of-κ̂ point tag. The flatness band is a fixed, small absolute
        // window on the curvature scale — a screening label only; the
        // statistically-honest "flat vs curved" call is the κ = 0 LR test.
        let geometry = if cc.kappa > 1e-6 {
            "spherical"
        } else if cc.kappa < -1e-6 {
            "hyperbolic"
        } else {
            "flat"
        };
        out.push(SummaryCurvatureRow {
            name: term.name.clone(),
            term_idx,
            kappa_hat: cc.kappa,
            geometry,
        });
    }
    out
}

/// Canonical fitted quantities reconstructed from a spline-scan model's saved
/// `SplineScanFit` (#1046).
///
/// A scan-routed model (the single-1-D-smooth, Gaussian-identity shape that
/// `spline_scan_fast_path` diverts to the exact O(n) state-space smoother)
/// carries **no dense `fit_result`** — by design the smoother keeps only the
/// per-knot posterior, never a dense design/Gram. So the FFI summary surface
/// reconstructs exactly what the smoother retained: the selected smoothing
/// parameter, the effective d.o.f. (tr S), the diffuse REML score, the profiled
/// scale, and the recovered deviance. This mirrors the numbers the CLI fit log
/// already prints from the same state.
///
/// The dense per-coefficient β / covariance is deliberately NOT materialized:
/// the smoother's natural parameters are the per-knot function values, of which
/// there are ~`n` (the scan exists precisely to avoid an O(n) design and an
/// O(n²) covariance). The summary therefore reports the model the way
/// `summary.gam` does — a parametric block (empty here; the smoother absorbs the
/// polynomial null space into the smooth) plus a smooth-terms table keyed on EDF
/// — rather than dumping ~`n` basis coefficients.
struct ScanIntrospection {
    feature_column: String,
    /// Effective degrees of freedom (tr S), strictly between the polynomial
    /// null-space dimension `order` and `n`.
    edf: f64,
    /// Selected smoothing parameter `λ` (always positive).
    lambda: f64,
    /// Diffuse REML expressed as a COST (lower is better): the negative
    /// restricted log marginal likelihood, on the same sign convention as the
    /// dense `reml_score`. It is exact up to the λ-free additive constant the
    /// concentrated criterion drops, which is what makes it an exact criterion
    /// for the smoother's own λ selection. That constant depends on (n, order,
    /// knots), so it does NOT cancel when comparing different fits — treat this
    /// as a within-fit quantity and prefer held-out predictive metrics for
    /// cross-model comparison.
    reml_cost: f64,
    /// Gaussian deviance — the weighted residual sum of squares.
    deviance: f64,
    /// Number of pooled knots (the smoother's natural coefficient count).
    n_knots: usize,
}

/// Reconstruct the canonical fitted quantities for a spline-scan model, or
/// `Ok(None)` for a dense model that should follow the standard `fit_result`
/// path (#1046).
fn scan_introspection(model: &FittedModel) -> Result<Option<ScanIntrospection>, String> {
    let Some((feature_column, fit)) = model.saved_spline_scan().map_err(|e| e.to_string())? else {
        return Ok(None);
    };
    Ok(Some(ScanIntrospection {
        feature_column: feature_column.to_string(),
        edf: fit.edf(),
        lambda: fit.lambda(),
        reml_cost: -fit.restricted_loglik,
        deviance: fit.deviance(),
        n_knots: fit.knots.len(),
    }))
}

/// Display label for the single smooth a scan model carries, e.g. `s(x)`.
fn scan_smooth_label(scan: &ScanIntrospection) -> String {
    format!("s({})", scan.feature_column)
}

/// Build the canonical FFI summary payload for a scan-routed model (#1046):
/// scalar fitted quantities plus a one-row smooth table keyed on EDF. The
/// parametric coefficient block is empty (the smoother absorbs the polynomial
/// null space) and no dense covariance is emitted — keeping `summary()` O(1) in
/// `n` regardless of how many knots the smoother spans.
fn scan_summary_payload(model: &FittedModel, scan: &ScanIntrospection) -> SummaryPayload {
    let smooth_terms = vec![SummarySmoothTermRow {
        name: scan_smooth_label(scan),
        edf: scan.edf,
        ref_df: scan.edf,
        // The rank-truncated Wald smooth test needs the joint coefficient
        // covariance, which the O(n) smoother does not retain; report EDF only,
        // as `summary.gam` does for terms whose Wald test is unavailable.
        chi_sq: None,
        p_value: None,
    }];
    SummaryPayload {
        formula: model.payload().formula.clone(),
        family_name: model.likelihood().pretty_name().to_string(),
        model_class: prediction_model_class_label(model),
        group_metadata: model.payload().group_metadata.clone(),
        deployment_extensions: model.payload().deployment_extensions.clone(),
        deviance: scan.deviance,
        // Scan-routed models do not retain the λ-comparable log-likelihood, so
        // leave `log_likelihood` unset.
        log_likelihood: None,
        // The O(n) smoother does not retain the IRLS working set, and its
        // `reml_cost` is explicitly within-fit (not cross-model comparable), so
        // leave `n_obs` unset — `compare_models` treats it as unconstrained.
        n_obs: None,
        // null_dim is left unset: the scan does not compute the penalized-Hessian
        // null-space logdet the TK normalizer needs, so `comparable_reml_score`
        // returns the raw cost unchanged (and `evidence()` stays well-defined).
        reml_score: scan.reml_cost,
        raw_reml_score: scan.reml_cost,
        null_space_logdet: None,
        null_dim: None,
        iterations: 0,
        edf_total: Some(scan.edf),
        lambdas: vec![scan.lambda],
        coefficients: Vec::new(),
        smooth_terms,
        covariance_kind: None,
        covariance_n: None,
        covariance_flat: None,
        // Scan-routed (O(n) 1D spline) models carry no `curv(...)` curvature
        // smooths, so there are no curvature estimands to report.
        curvature_estimands: Vec::new(),
    }
}

fn summary_json_impl(model_bytes: &[u8]) -> Result<String, String> {
    let model = load_model_impl(model_bytes)?;
    if let Some(scan) = scan_introspection(&model)? {
        let payload = scan_summary_payload(&model, &scan);
        return serde_json::to_string(&payload)
            .map_err(|err| format!("failed to serialize summary: {err}"));
    }
    let fit = fit_result_from_saved_model_for_prediction(&model)?;
    let smooth_terms = summary_smooth_terms(&model, &fit);
    let standard_errors = fit
        .beta_standard_errors_corrected()
        .or_else(|| fit.beta_standard_errors());
    let covariance = fit
        .beta_covariance_corrected()
        .map(|cov| ("corrected".to_string(), cov))
        .or_else(|| {
            fit.beta_covariance()
                .map(|cov| ("conditional".to_string(), cov))
        });
    let coefficients = fit
        .beta
        .iter()
        .enumerate()
        .map(|(index, estimate)| SummaryCoefficientRow {
            index,
            estimate: *estimate,
            std_error: standard_errors.and_then(|values| values.get(index).copied()),
        })
        .collect();
    let raw_reml_score = fit.reml_score;
    let reml_score = comparable_reml_score(
        raw_reml_score,
        fit.artifacts.null_space_dim.map(|dim| dim as f64),
        fit.artifacts.null_space_logdet,
    )
    .map_err(|err| format!("failed to compute comparable REML score: {err}"))?;
    let payload = SummaryPayload {
        formula: model.payload().formula.clone(),
        family_name: model.likelihood().pretty_name().to_string(),
        model_class: prediction_model_class_label(&model),
        group_metadata: model.payload().group_metadata.clone(),
        deployment_extensions: model.payload().deployment_extensions.clone(),
        deviance: fit.deviance,
        log_likelihood: Some(fit.log_likelihood),
        // Observation count from the IRLS working set (same source the Wald χ²
        // n-dependence reads). Lets `compare_models` reject cross-`n` comparisons.
        n_obs: fit
            .geometry
            .as_ref()
            .map(|geom| geom.working_response.len()),
        reml_score,
        raw_reml_score,
        null_space_logdet: fit.artifacts.null_space_logdet,
        null_dim: fit.artifacts.null_space_dim.map(|dim| dim as f64),
        iterations: fit.outer_iterations,
        edf_total: fit.edf_total(),
        lambdas: fit.lambdas.to_vec(),
        coefficients,
        smooth_terms,
        curvature_estimands: summary_curvature_estimands(&model),
        covariance_kind: covariance.as_ref().map(|(kind, _)| kind.clone()),
        covariance_n: covariance.as_ref().map(|(_, cov)| cov.nrows()),
        covariance_flat: covariance.map(|(_, cov)| cov.iter().copied().collect()),
    };
    serde_json::to_string(&payload).map_err(|err| format!("failed to serialize summary: {err}"))
}

/// One `curv(...)` term's #944 report, JSON-serialized for the Python surface.
#[derive(Serialize)]
struct CurvatureInferenceRow {
    name: String,
    term_idx: usize,
    kappa_hat: f64,
    ci_lo: f64,
    ci_hi: f64,
    /// `true` when the CI is left-open at the κ chart bound (profile too flat to
    /// close the lower endpoint).
    lo_at_bound: bool,
    /// `true` when the CI is right-open at the κ chart bound.
    hi_at_bound: bool,
    /// Sign-of-CI geometry verdict: `"spherical"`, `"hyperbolic"`, `"flat"`, or
    /// `"indistinguishable"` (CI straddles 0).
    verdict: &'static str,
    /// LR statistic `2[V_p(0) − V_p(κ̂)] ≥ 0` for the interior κ = 0 test.
    flatness_lr_stat: f64,
    /// p-value of the κ = 0 flatness test against the interior χ²₁ reference
    /// (no half-χ² boundary correction — κ = 0 is interior to S^d ← ℝ^d → H^d).
    flatness_p_value: f64,
}

#[derive(Serialize)]
struct CurvatureInferencePayload {
    level: f64,
    curvature_terms: Vec<CurvatureInferenceRow>,
}

/// One penalized smooth term's #1063 per-term LR significance report,
/// JSON-serialized for the Python surface.
#[derive(Serialize)]
struct SmoothTermLrRow {
    name: String,
    term_idx: usize,
    /// Uncorrected likelihood-ratio statistic `W = 2(ℓ_full − ℓ_null) ≥ 0`.
    statistic_lr: f64,
    /// Reference d.f. `d` (Wood truncation `tr(F)²/tr(F²)`; same as the Wald row).
    ref_df: f64,
    /// Lawley LR Bartlett factor `c = 1 + Δε/d` (1.0 when uncorrected).
    bartlett_factor: f64,
    /// Fixed-λ conditional Lawley factor when the applied factor also includes
    /// estimated-λ rho variation.
    bartlett_factor_conditional: Option<f64>,
    /// Mean-shift increment from ρ̂ sampling variation, when present.
    rho_variation_shift: Option<f64>,
    /// Bartlett-corrected statistic `W* = W / c`.
    statistic_corrected: f64,
    /// Uncorrected p-value `P(χ²_d > W)`.
    p_value_uncorrected: f64,
    /// Corrected p-value `P(χ²_d > W*)` — the magic-by-default reported value.
    p_value_corrected: f64,
    /// `true` when the correction is **material** (#939 deliverable 4): it moves
    /// the Bartlett factor or the p-value by more than 10% — the diagnostic that
    /// `n` is too small for first-order inference on this term. `false` when no
    /// correction was applied.
    material: bool,
    /// `"lawley_lr_estimated_lambda"` when the full estimated-λ Bartlett
    /// correction was applied, `"lawley_lr_fixed_lambda"` for the conditional
    /// fixed-λ factor, else `"none"`.
    correction_provenance: &'static str,
}

#[derive(Serialize)]
struct SmoothTermLrPayload {
    smooth_terms: Vec<SmoothTermLrRow>,
}

fn curvature_verdict_label(v: gam::geometry::CurvatureVerdict) -> &'static str {
    match v {
        gam::geometry::CurvatureVerdict::Spherical => "spherical",
        gam::geometry::CurvatureVerdict::Hyperbolic => "hyperbolic",
        gam::geometry::CurvatureVerdict::Flat => "flat",
    }
}

/// #944: re-profile `V_p(κ)` for every `curv(...)` smooth and emit κ̂ + profile
/// CI + κ = 0 flatness test. The κ̂ lives in the saved (fitted) spec; the data
/// supply the responses/weights/offset the per-κ profile refits read. We
/// materialize a Standard fit request from the model's training formula + data,
/// then swap in the model's fitted spec and family so the profile oracle refits
/// at the EXACT estimand the model was fitted under — only κ moves.
fn curvature_inference_dataset_json_impl(
    model_bytes: &[u8],
    dataset: EncodedDataset,
    level: f64,
) -> Result<String, String> {
    use gam::terms::smooth::SmoothBasisSpec;
    if !(level.is_finite() && level > 0.0 && level < 1.0) {
        return Err(format!(
            "curvature_inference: confidence level must be in (0, 1), got {level}"
        ));
    }
    let model = load_model_impl(model_bytes)?;
    let formula = model.payload().formula.clone();
    let spec = model
        .payload()
        .resolved_termspec
        .as_ref()
        .ok_or_else(|| {
            "curvature_inference requires the saved resolved_termspec (carries κ̂); refit"
                .to_string()
        })?
        .clone();
    // Fast bail: no constant-curvature term ⇒ empty report (no refit, no data
    // materialization needed).
    let has_curv = spec
        .smooth_terms
        .iter()
        .any(|t| matches!(t.basis, SmoothBasisSpec::ConstantCurvature { .. }));
    if !has_curv {
        let payload = CurvatureInferencePayload {
            level,
            curvature_terms: Vec::new(),
        };
        return serde_json::to_string(&payload)
            .map_err(|err| format!("failed to serialize curvature inference: {err}"));
    }

    // Materialize responses/weights/offset/options from the training data under
    // the model's own family. We replace the materialized (default-κ) spec with
    // the FITTED spec so the V_p oracle profiles around κ̂ — only κ moves.
    let fit_config = postfit_standard_materialization_config(&model)?;
    let materialized = materialize(&formula, &dataset, &fit_config)?;
    let standard = match materialized.request {
        FitRequest::Standard(request) => request,
        _ => {
            return Err(
                "curvature_inference: only standard (non marginal-slope / non survival) \
                 models carry a profileable κ; this model uses a specialized fit request"
                    .to_string(),
            );
        }
    };

    let family = model.likelihood();
    let mut terms_out = Vec::<CurvatureInferenceRow>::new();
    for (term_idx, term) in spec.smooth_terms.iter().enumerate() {
        if !matches!(term.basis, SmoothBasisSpec::ConstantCurvature { .. }) {
            continue;
        }
        let report = gam::families::fit_orchestration::drivers::curvature_inference_forspec(
            standard.data.view(),
            standard.y.view(),
            standard.weights.view(),
            standard.offset.view(),
            &spec,
            term_idx,
            family.clone(),
            &standard.options,
            level,
        )
        .map_err(|e| format!("curvature_inference for term {term_idx}: {e}"))?;
        terms_out.push(CurvatureInferenceRow {
            name: term.name.clone(),
            term_idx,
            kappa_hat: report.kappa_hat,
            ci_lo: report.ci.ci_lo,
            ci_hi: report.ci.ci_hi,
            lo_at_bound: report.ci.lo_at_bound,
            hi_at_bound: report.ci.hi_at_bound,
            verdict: curvature_verdict_label(report.ci.verdict),
            flatness_lr_stat: report.flatness.lr_stat,
            flatness_p_value: report.flatness.p_value,
        });
    }

    let payload = CurvatureInferencePayload {
        level,
        curvature_terms: terms_out,
    };
    serde_json::to_string(&payload)
        .map_err(|err| format!("failed to serialize curvature inference: {err}"))
}

/// #1063: per-term likelihood-ratio significance for every penalized smooth term
/// in a fitted model, Bartlett-corrected by default. The summary table reports
/// Wood's rank-truncated **Wald** statistic, which the Lawley LR factor would
/// correct wrongly under penalization; this entry computes a genuine LR
/// statistic by a constrained refit (the smooth dropped) and corrects *that*.
///
/// Like `curvature_inference_json`, the honest LR needs the model's training
/// data: we materialize a Standard fit request from the model's training formula
/// + data, swap in the model's fitted (frozen) spec and family, then run the
/// core LR + Bartlett driver. The fitted spec carries the exact estimand the
/// model was fitted under.
fn smooth_term_lr_inference_dataset_json_impl(
    model_bytes: &[u8],
    dataset: EncodedDataset,
) -> Result<String, String> {
    let model = load_model_impl(model_bytes)?;
    let formula = model.payload().formula.clone();
    let spec = model
        .payload()
        .resolved_termspec
        .as_ref()
        .ok_or_else(|| {
            "smooth_term_lr_inference requires the saved resolved_termspec; refit".to_string()
        })?
        .clone();
    // Fast bail: no smooth terms ⇒ empty report (no refit, no data needed).
    if spec.smooth_terms.is_empty() {
        let payload = SmoothTermLrPayload {
            smooth_terms: Vec::new(),
        };
        return serde_json::to_string(&payload)
            .map_err(|err| format!("failed to serialize smooth-term LR inference: {err}"));
    }

    let fit_config = postfit_standard_materialization_config(&model)?;
    let materialized = materialize(&formula, &dataset, &fit_config)?;
    let standard = match materialized.request {
        FitRequest::Standard(request) => request,
        _ => {
            return Err(
                "smooth_term_lr_inference: only standard (non marginal-slope / non survival) \
                 models carry a per-term central-χ² LR; this model uses a specialized fit request"
                    .to_string(),
            );
        }
    };

    let family = model.likelihood();
    let reports = gam::families::fit_orchestration::drivers::smooth_term_lr_inference_forspec(
        standard.data.view(),
        standard.y.view(),
        standard.weights.view(),
        standard.offset.view(),
        &spec,
        family,
        &standard.options,
    )
    .map_err(|e| format!("smooth_term_lr_inference: {e}"))?;

    let smooth_terms = reports
        .into_iter()
        .map(|r| SmoothTermLrRow {
            name: r.name,
            term_idx: r.term_idx,
            statistic_lr: r.statistic_lr,
            ref_df: r.ref_df,
            bartlett_factor: r.bartlett_factor,
            bartlett_factor_conditional: r.bartlett_factor_conditional,
            rho_variation_shift: r.rho_variation_shift,
            statistic_corrected: r.statistic_corrected,
            p_value_uncorrected: r.p_value_uncorrected,
            p_value_corrected: r.p_value_corrected,
            material: r.material,
            correction_provenance: r.correction.label(),
        })
        .collect::<Vec<_>>();

    let payload = SmoothTermLrPayload { smooth_terms };
    serde_json::to_string(&payload)
        .map_err(|err| format!("failed to serialize smooth-term LR inference: {err}"))
}

fn postfit_standard_materialization_config(model: &FittedModel) -> Result<FitConfig, String> {
    let mut fit_config = parse_fit_config(None)?;
    fit_config.weight_column = model.weight_column.clone();
    fit_config.offset_column = model.offset_column.clone();
    Ok(fit_config)
}

fn check_dataset_json_impl(model_bytes: &[u8], dataset: EncodedDataset) -> Result<String, String> {
    let model = load_model_impl(model_bytes)?;
    let check = schema_check_encoded(&model, &dataset)?;
    serde_json::to_string(&check).map_err(|err| format!("failed to serialize schema check: {err}"))
}

/// Render the HTML report for a scan-routed model (#1046) from its
/// reconstructed scalar quantities and the single smooth's EDF block. No
/// per-coefficient table — see [`scan_summary_payload`] for why the smoother's
/// ~`n` knot values are not a summary artifact.
fn scan_report_html(model: &FittedModel, scan: &ScanIntrospection) -> Result<String, String> {
    let report_input = ReportInput {
        model_path: "<in-memory>".to_string(),
        family_name: model.likelihood().pretty_name().to_string(),
        model_class: prediction_model_class_label(model),
        formula: model.payload().formula.clone(),
        n_obs: None,
        deviance: scan.deviance,
        reml_score: scan.reml_cost,
        iterations: 0,
        convergence_status: "exact (state-space spline scan)".to_string(),
        converged: true,
        outer_gradient_norm: None,
        criterion_certificate: None,
        smoothing_forensics: Vec::new(),
        edf_total: scan.edf,
        r_squared: None,
        coefficients: Vec::new(),
        edf_blocks: vec![EdfBlockRow {
            index: 0,
            edf: scan.edf,
            role: Some("smooth".to_string()),
        }],
        continuous_order: Vec::new(),
        anisotropic_scales: Vec::new(),
        measure_jet_spectra: Vec::new(),
        diagnostics: None,
        smooth_plots: Vec::new(),
        alo: None,
        notes: vec![format!(
            "Exact O(n) state-space spline scan for {}: λ={:.4e}, EDF={:.3}. \
             The smoother retains the per-knot posterior, not a dense \
             design/Gram, so no per-coefficient table is shown.",
            scan_smooth_label(scan),
            scan.lambda,
            scan.edf,
        )],
    };
    render_html(&report_input)
}

fn report_html_impl(model_bytes: &[u8]) -> Result<String, String> {
    let model = load_model_impl(model_bytes)?;
    if let Some(scan) = scan_introspection(&model)? {
        return scan_report_html(&model, &scan);
    }
    let fit = fit_result_from_saved_model_for_prediction(&model)?;
    let standard_errors = fit
        .beta_standard_errors_corrected()
        .or_else(|| fit.beta_standard_errors());
    let coefficients = fit
        .beta
        .iter()
        .enumerate()
        .map(|(index, estimate)| CoefficientRow {
            index,
            estimate: *estimate,
            std_error: standard_errors.and_then(|values| values.get(index).copied()),
        })
        .collect::<Vec<_>>();
    let edf_blocks = fit
        .edf_by_block()
        .iter()
        .enumerate()
        .map(|(index, edf)| EdfBlockRow {
            index,
            edf: *edf,
            role: fit
                .blocks
                .get(index)
                .map(|block| block.role.clone().name().to_string()),
        })
        .collect::<Vec<_>>();
    let report_input = ReportInput {
        model_path: "<in-memory>".to_string(),
        family_name: model.likelihood().pretty_name().to_string(),
        model_class: prediction_model_class_label(&model),
        formula: model.payload().formula.clone(),
        n_obs: None,
        deviance: fit.deviance,
        reml_score: fit.reml_score,
        iterations: fit.outer_iterations,
        convergence_status: fit
            .convergence_evidence()
            .inner_status()
            .label()
            .to_string(),
        converged: true,
        outer_gradient_norm: fit.outer_gradient_norm,
        criterion_certificate: None,
        smoothing_forensics: Vec::new(),
        edf_total: fit.edf_total().unwrap_or(0.0),
        r_squared: None,
        coefficients,
        edf_blocks,
        continuous_order: Vec::new(),
        anisotropic_scales: Vec::new(),
        measure_jet_spectra: Vec::new(),
        diagnostics: None,
        smooth_plots: Vec::new(),
        alo: None,
        notes: vec![
            "Python report currently omits data-dependent diagnostics and smooth plots."
                .to_string(),
        ],
    };
    render_html(&report_input)
}

/// Python-facing wrapper that builds the analytic-penalty registry from a pair
/// of JSON strings (`latents`, `descriptors`) and returns a JSON array where
/// each entry exposes its canonical `kind` tag — the same string returned by
/// `AnalyticPenaltyKind::name()`. This is the round-trip surface exercised by
/// Python tests to confirm that descriptor JSON travels through the Rust side
/// without losing identity.
#[pyfunction]
fn build_analytic_penalty_registry_json(
    latents_json: &str,
    descriptors_json: &str,
) -> PyResult<String> {
    let latents: serde_json::Value = serde_json::from_str(latents_json)
        .map_err(|err| py_value_error(format!("invalid latents json: {err}")))?;
    let descriptors: serde_json::Value = serde_json::from_str(descriptors_json)
        .map_err(|err| py_value_error(format!("invalid descriptors json: {err}")))?;
    let registry = build_analytic_penalty_registry_from_json(Some(&latents), Some(&descriptors))
        .map_err(py_value_error)?;
    let entries: Vec<serde_json::Value> = registry
        .penalties
        .iter()
        .map(|penalty| {
            let mut entry = serde_json::Map::new();
            entry.insert(
                "kind".to_string(),
                serde_json::Value::String(penalty.name().to_string()),
            );
            entry.insert(
                "kind_tag".to_string(),
                serde_json::Value::String(penalty.kind_tag().to_string()),
            );
            serde_json::Value::Object(entry)
        })
        .collect();
    serde_json::to_string(&serde_json::Value::Array(entries)).map_err(|err| {
        py_value_error(format!(
            "failed to serialize analytic penalty registry json: {err}"
        ))
    })
}

/// Convert a JSON-sourced `u64` into a positive `usize`, rejecting zero and
/// values that exceed `usize::MAX`. Used by the geometry-manifold descriptor
/// parsers below.
fn json_positive_u64_to_usize(value: u64, context: &str) -> Result<usize, String> {
    if value == 0 {
        return Err(format!("{context} must be > 0"));
    }
    usize::try_from(value).map_err(|_| format!("{context} exceeds usize::MAX"))
}

#[pyfunction(signature = (latents_json, penalties_json))]
fn register_analytic_penalties(latents_json: &str, penalties_json: &str) -> PyResult<String> {
    let latents: serde_json::Value = serde_json::from_str(latents_json)
        .map_err(|err| py_value_error(format!("invalid latents json: {err}")))?;
    let penalties: serde_json::Value = serde_json::from_str(penalties_json)
        .map_err(|err| py_value_error(format!("invalid penalties json: {err}")))?;
    let registry = build_analytic_penalty_registry_from_json(Some(&latents), Some(&penalties))
        .map_err(py_value_error)?;
    let layout = registry
        .rho_layout()
        .into_iter()
        .map(|(range, tier, name)| {
            serde_json::json!({
                "name": name,
                "tier": format!("{tier:?}"),
                "rho_start": range.start,
                "rho_end": range.end,
            })
        })
        .collect::<Vec<_>>();
    serde_json::to_string(&serde_json::json!({
        "penalty_count": registry.penalties.len(),
        "rho_count": registry.total_rho_count(),
        "layout": layout,
    }))
    .map_err(serde_json_error_to_pyerr)
}

/// Smooth threshold activation value and exact gradients.
///
/// Returns `(value, dphi_dz, dphi_dtau)`, all shaped like `z` `(N, F)`, where
/// `value = z · σ((z − τ)/ε)` and the returned derivatives are those of
/// that same scalar. `tau` holds the per-column effective thresholds.
#[pyfunction(signature = (z, tau, smoothing_eps))]
fn smooth_threshold_gate_value_grad<'py>(
    py: Python<'py>,
    z: PyReadonlyArray2<'py, f64>,
    tau: PyReadonlyArray1<'py, f64>,
    smoothing_eps: f64,
) -> PyResult<(Py<PyArray2<f64>>, Py<PyArray2<f64>>, Py<PyArray2<f64>>)> {
    if !(smoothing_eps.is_finite() && smoothing_eps > 0.0) {
        return Err(py_value_error(format!(
            "smooth_threshold_gate_value_grad: smoothing_eps must be finite and positive; got {smoothing_eps}"
        )));
    }
    let z_view = z.as_array();
    let tau_view = tau.as_array();
    let (n_rows, n_cols) = z_view.dim();
    if tau_view.len() != n_cols {
        return Err(py_value_error(format!(
            "smooth_threshold_gate_value_grad: tau length {} does not match z columns {n_cols}",
            tau_view.len()
        )));
    }
    let mut value = Array2::<f64>::zeros((n_rows, n_cols));
    let mut dphi_dz = Array2::<f64>::zeros((n_rows, n_cols));
    let mut dphi_dtau = Array2::<f64>::zeros((n_rows, n_cols));
    for r in 0..n_rows {
        for c in 0..n_cols {
            let (v, dz, dt) = gam::terms::analytic_penalties::smooth_threshold_gate_value_grad(
                z_view[[r, c]],
                tau_view[c],
                smoothing_eps,
            );
            value[[r, c]] = v;
            dphi_dz[[r, c]] = dz;
            dphi_dtau[[r, c]] = dt;
        }
    }
    Ok((
        value.into_pyarray(py).unbind(),
        dphi_dz.into_pyarray(py).unbind(),
        dphi_dtau.into_pyarray(py).unbind(),
    ))
}

/// Temperature `τ` of the Rust [`GumbelTemperatureSchedule`] at iteration
/// `iter`, so torch's sparsity-layer annealing queries the single Rust schedule
/// instead of duplicating the decay arithmetic in Python. `schedule` is the
/// same descriptor dict the SAE composition surface accepts (keys: `tau_start`,
/// `tau_min`, `decay`, and `rate`/`steps` as required by the decay).
#[pyfunction(signature = (schedule, iter))]
fn gumbel_schedule_tau(schedule: &Bound<'_, PyDict>, iter: usize) -> PyResult<f64> {
    let parsed = gumbel_temperature_schedule_from_pydict(Some(schedule))
        .map_err(py_value_error)?
        .ok_or_else(|| {
            py_value_error("gumbel_schedule_tau requires a schedule descriptor".to_string())
        })?;
    Ok(parsed.current_tau(iter))
}

#[pyfunction(signature = (
    latents_json,
    penalties_json,
    target,
    rho = None,
    isometry_jacobian = None,
    isometry_jacobian_second = None
))]
fn analytic_penalty_value_grad<'py>(
    py: Python<'py>,
    latents_json: &str,
    penalties_json: &str,
    target: PyReadonlyArray1<'py, f64>,
    rho: Option<PyReadonlyArray1<'py, f64>>,
    isometry_jacobian: Option<PyReadonlyArray2<'py, f64>>,
    isometry_jacobian_second: Option<PyReadonlyArray2<'py, f64>>,
) -> PyResult<(
    f64,
    Py<PyArray1<f64>>,
    Py<PyArray1<f64>>,
    Option<Py<PyArray2<f64>>>,
)> {
    let latents: serde_json::Value = serde_json::from_str(latents_json)
        .map_err(|err| py_value_error(format!("invalid latents json: {err}")))?;
    let penalties: serde_json::Value = serde_json::from_str(penalties_json)
        .map_err(|err| py_value_error(format!("invalid penalties json: {err}")))?;
    let mut registry = build_analytic_penalty_registry_from_json(Some(&latents), Some(&penalties))
        .map_err(py_value_error)?;

    let want_jacobian_grad = isometry_jacobian.is_some();
    if let Some(j) = isometry_jacobian {
        let j_cache = Arc::new(j.as_array().to_owned());
        let h_cache = isometry_jacobian_second.map(|h| Arc::new(h.as_array().to_owned()));
        for penalty in &mut registry.penalties {
            if let AnalyticPenaltyKind::Isometry(inner) = penalty {
                let mut cloned = (**inner).clone().with_jacobian_cache(j_cache.clone());
                if let Some(h) = h_cache.as_ref() {
                    cloned = cloned.with_jacobian_second_cache(h.clone());
                }
                *penalty = AnalyticPenaltyKind::Isometry(Arc::new(cloned));
            }
        }
    }

    let target_view = target.as_array();
    let rho_owned = match rho {
        Some(rho) => rho.as_array().to_owned(),
        None => Array1::<f64>::zeros(registry.total_rho_count()),
    };
    if rho_owned.len() != registry.total_rho_count() {
        return Err(py_value_error(format!(
            "rho length {} does not match analytic penalty rho_count {}",
            rho_owned.len(),
            registry.total_rho_count()
        )));
    }
    registry
        .validate_rho(rho_owned.view())
        .map_err(py_value_error)?;

    let mut value = 0.0_f64;
    let mut grad = Array1::<f64>::zeros(target_view.len());
    let mut grad_rho = Array1::<f64>::zeros(rho_owned.len());
    let mut grad_jacobian: Option<Array2<f64>> = None;
    for (penalty, (rho_slice, tier, _name)) in registry.penalties.iter().zip(registry.rho_layout())
    {
        if matches!(tier, PenaltyTier::Rho) {
            continue;
        }
        let rho_local = rho_owned.slice(s![rho_slice.clone()]);
        value += penalty.value(target_view.view(), rho_local);
        grad += &penalty.grad_target(target_view.view(), rho_local);
        let local_grad_rho = penalty.grad_rho(target_view.view(), rho_local);
        for (local, global) in (rho_slice.start..rho_slice.end).enumerate() {
            grad_rho[global] += local_grad_rho[local];
        }
        if want_jacobian_grad {
            if let AnalyticPenaltyKind::Isometry(inner) = penalty {
                let contrib = inner.grad_jacobian(target_view.view(), rho_local);
                match grad_jacobian.as_mut() {
                    Some(acc) => *acc += &contrib,
                    None => grad_jacobian = Some(contrib),
                }
            }
        }
    }
    Ok((
        value,
        grad.into_pyarray(py).unbind(),
        grad_rho.into_pyarray(py).unbind(),
        grad_jacobian.map(|g| g.into_pyarray(py).unbind()),
    ))
}

/// Hessian-vector product `H · v` of the analytic-penalty registry frozen at
/// `(target, rho)`, accumulated across all penalties whose target tier lives
/// on `target` (`PenaltyTier::Psi`). This is the same kernel that `PIRLS`
/// uses when it folds analytic-penalty Hessians into the inner Newton step;
/// the result is `Σ_p (∂²P_p / ∂target²) · v` evaluated at the supplied
/// iterate.
///
/// `rho` defaults to a zero vector of the registry's total ρ-length, which
/// makes the contribution of each penalty equal to its descriptor-pinned
/// weight (no REML shrinkage offset). Pass an explicit `rho` if the caller
/// is mid-REML and wants the corresponding `exp(ρ)` scaling.
#[pyfunction(signature = (
    latents_json,
    penalties_json,
    target,
    v,
    rho = None,
    isometry_jacobian = None,
    isometry_jacobian_second = None
))]
fn analytic_penalty_hvp<'py>(
    py: Python<'py>,
    latents_json: &str,
    penalties_json: &str,
    target: PyReadonlyArray1<'py, f64>,
    v: PyReadonlyArray1<'py, f64>,
    rho: Option<PyReadonlyArray1<'py, f64>>,
    isometry_jacobian: Option<PyReadonlyArray2<'py, f64>>,
    isometry_jacobian_second: Option<PyReadonlyArray2<'py, f64>>,
) -> PyResult<Py<PyArray1<f64>>> {
    let latents: serde_json::Value = serde_json::from_str(latents_json)
        .map_err(|err| py_value_error(format!("invalid latents json: {err}")))?;
    let penalties: serde_json::Value = serde_json::from_str(penalties_json)
        .map_err(|err| py_value_error(format!("invalid penalties json: {err}")))?;
    let mut registry = build_analytic_penalty_registry_from_json(Some(&latents), Some(&penalties))
        .map_err(py_value_error)?;

    if let Some(j) = isometry_jacobian {
        let j_cache = Arc::new(j.as_array().to_owned());
        let h_cache = isometry_jacobian_second.map(|h| Arc::new(h.as_array().to_owned()));
        for penalty in &mut registry.penalties {
            if let AnalyticPenaltyKind::Isometry(inner) = penalty {
                let mut cloned = (**inner).clone().with_jacobian_cache(j_cache.clone());
                if let Some(h) = h_cache.as_ref() {
                    cloned = cloned.with_jacobian_second_cache(h.clone());
                }
                *penalty = AnalyticPenaltyKind::Isometry(Arc::new(cloned));
            }
        }
    }

    let target_view = target.as_array();
    let v_view = v.as_array();
    if v_view.len() != target_view.len() {
        return Err(py_value_error(format!(
            "analytic_penalty_hvp: v length {} does not match target length {}",
            v_view.len(),
            target_view.len()
        )));
    }
    let rho_owned = match rho {
        Some(rho) => rho.as_array().to_owned(),
        None => Array1::<f64>::zeros(registry.total_rho_count()),
    };
    if rho_owned.len() != registry.total_rho_count() {
        return Err(py_value_error(format!(
            "rho length {} does not match analytic penalty rho_count {}",
            rho_owned.len(),
            registry.total_rho_count()
        )));
    }
    registry
        .validate_rho(rho_owned.view())
        .map_err(py_value_error)?;

    let mut out = Array1::<f64>::zeros(target_view.len());
    for (penalty, (rho_slice, tier, _name)) in registry.penalties.iter().zip(registry.rho_layout())
    {
        if matches!(tier, PenaltyTier::Rho) {
            continue;
        }
        let rho_local = rho_owned.slice(s![rho_slice.clone()]);
        let contrib = penalty.hvp(target_view.view(), rho_local, v_view.view());
        out += &contrib;
    }
    Ok(out.into_pyarray(py).unbind())
}

fn parse_manifold_kind(value: &serde_json::Value) -> Result<gam::geometry::ManifoldSpec, String> {
    if let Some(name) = value.as_str() {
        return match name.to_ascii_lowercase().as_str() {
            "circle" | "s1" => Ok(gam::geometry::ManifoldSpec::Circle),
            other => Err(format!("unknown manifold string {other:?}")),
        };
    }
    let obj = value
        .as_object()
        .ok_or_else(|| "manifold must be a string or object".to_string())?;
    let kind = obj
        .get("kind")
        .or_else(|| obj.get("type"))
        .and_then(serde_json::Value::as_str)
        .ok_or_else(|| "manifold.kind is required".to_string())?
        .to_ascii_lowercase();
    match kind.as_str() {
        "euclidean" => {
            let dim = obj
                .get("dim")
                .or_else(|| obj.get("d"))
                .and_then(serde_json::Value::as_u64)
                .ok_or_else(|| "euclidean manifold requires dim".to_string())?;
            Ok(gam::geometry::ManifoldSpec::Euclidean(
                json_positive_u64_to_usize(dim, "euclidean.dim")?,
            ))
        }
        "circle" | "s1" => Ok(gam::geometry::ManifoldSpec::Circle),
        "sphere" => {
            let n = obj
                .get("intrinsic_dim")
                .or_else(|| obj.get("n"))
                .or_else(|| obj.get("dim"))
                .and_then(serde_json::Value::as_u64)
                .ok_or_else(|| "sphere manifold requires intrinsic_dim".to_string())?;
            Ok(gam::geometry::ManifoldSpec::Sphere {
                intrinsic_dim: json_positive_u64_to_usize(n, "sphere.intrinsic_dim")?,
            })
        }
        "torus" => {
            let d = obj
                .get("d")
                .or_else(|| obj.get("dim"))
                .and_then(serde_json::Value::as_u64)
                .ok_or_else(|| "torus manifold requires d".to_string())?;
            Ok(gam::geometry::ManifoldSpec::Torus {
                dim: json_positive_u64_to_usize(d, "torus.d")?,
            })
        }
        "grassmann" => {
            let k = obj
                .get("k")
                .and_then(serde_json::Value::as_u64)
                .ok_or_else(|| "grassmann manifold requires k".to_string())?;
            let n = obj
                .get("n")
                .and_then(serde_json::Value::as_u64)
                .ok_or_else(|| "grassmann manifold requires n".to_string())?;
            let k = json_positive_u64_to_usize(k, "grassmann.k")?;
            let n = json_positive_u64_to_usize(n, "grassmann.n")?;
            if k > n {
                return Err(format!(
                    "grassmann manifold requires k <= n (got k={k}, n={n}): \
                     Gr(k, n) is the set of k-dimensional subspaces of R^n"
                ));
            }
            Ok(gam::geometry::ManifoldSpec::Grassmann { k, n })
        }
        "stiefel" => {
            let k = obj
                .get("k")
                .and_then(serde_json::Value::as_u64)
                .ok_or_else(|| "stiefel manifold requires k".to_string())?;
            let n = obj
                .get("n")
                .and_then(serde_json::Value::as_u64)
                .ok_or_else(|| "stiefel manifold requires n".to_string())?;
            let k = json_positive_u64_to_usize(k, "stiefel.k")?;
            let n = json_positive_u64_to_usize(n, "stiefel.n")?;
            if k > n {
                return Err(format!(
                    "stiefel manifold requires k <= n (got k={k}, n={n}): \
                     St(n, k) is the set of k-frames (orthonormal columns) in R^n"
                ));
            }
            Ok(gam::geometry::ManifoldSpec::Stiefel { k, n })
        }
        "spd" => {
            let n = obj
                .get("n")
                .and_then(serde_json::Value::as_u64)
                .ok_or_else(|| "spd manifold requires n".to_string())?;
            Ok(gam::geometry::ManifoldSpec::Spd {
                n: json_positive_u64_to_usize(n, "spd.n")?,
            })
        }
        "product" => {
            let parts = obj
                .get("components")
                .or_else(|| obj.get("parts"))
                .and_then(serde_json::Value::as_array)
                .ok_or_else(|| "product manifold requires components".to_string())?;
            let mut parsed = Vec::with_capacity(parts.len());
            for part in parts {
                parsed.push(parse_manifold_kind(part)?);
            }
            Ok(gam::geometry::ManifoldSpec::Product(parsed))
        }
        other => Err(format!("unknown manifold kind {other:?}")),
    }
}

/// Take one batched metric-correct Riemannian gradient step.
///
/// `euclidean_grad` is an ambient Euclidean differential, not a tangent step.
/// The geometry implementation raises it through the manifold metric before
/// scaling and retracting, so affine-invariant SPD and canonical Stiefel
/// geometry use their actual Riesz representatives.
#[pyfunction(signature = (manifold_json, points, euclidean_grad, learning_rate))]
fn riemannian_gradient_step<'py>(
    py: Python<'py>,
    manifold_json: &str,
    points: PyReadonlyArray2<'py, f64>,
    euclidean_grad: PyReadonlyArray2<'py, f64>,
    learning_rate: f64,
) -> PyResult<Py<PyArray2<f64>>> {
    let value: serde_json::Value = serde_json::from_str(manifold_json)
        .map_err(|err| py_value_error(format!("invalid manifold json: {err}")))?;
    let kind = parse_manifold_kind(&value).map_err(py_value_error)?;
    let manifold = kind
        .build()
        .map_err(|err| py_value_error(err.to_string()))?;
    let p = points.as_array();
    let e = euclidean_grad.as_array();
    if p.dim() != e.dim() {
        return Err(py_value_error(format!(
            "points shape {:?} does not match euclidean_grad shape {:?}",
            p.dim(),
            e.dim()
        )));
    }
    if p.ncols() != manifold.ambient_dim() {
        return Err(py_value_error(format!(
            "points width {} does not match manifold ambient_dim {}",
            p.ncols(),
            manifold.ambient_dim()
        )));
    }
    let mut out = Array2::<f64>::zeros(p.dim());
    for row in 0..p.nrows() {
        let next = manifold
            .riemannian_gradient_step(p.row(row), e.row(row), learning_rate)
            .map_err(|err| py_value_error(err.to_string()))?;
        out.row_mut(row).assign(&next);
    }
    Ok(out.into_pyarray(py).unbind())
}

/// Batched Riemannian exponential map: for each row, return ``exp_p(v)``.
/// ``points`` has shape ``(N, ambient_dim)`` and ``vecs`` has shape
/// ``(N, ambient_dim)``. The manifold descriptor is parsed from
/// ``manifold_json`` using the same schema as
/// :func:`riemannian_gradient_step`.
#[pyfunction(signature = (manifold_json, points, vecs))]
fn manifold_exp_map<'py>(
    py: Python<'py>,
    manifold_json: &str,
    points: PyReadonlyArray2<'py, f64>,
    vecs: PyReadonlyArray2<'py, f64>,
) -> PyResult<Py<PyArray2<f64>>> {
    let value: serde_json::Value = serde_json::from_str(manifold_json)
        .map_err(|err| py_value_error(format!("invalid manifold json: {err}")))?;
    let kind = parse_manifold_kind(&value).map_err(py_value_error)?;
    let manifold = kind
        .build()
        .map_err(|err| py_value_error(err.to_string()))?;
    let p = points.as_array();
    let v = vecs.as_array();
    if p.dim() != v.dim() {
        return Err(py_value_error(format!(
            "manifold_exp_map: points shape {:?} does not match vecs shape {:?}",
            p.dim(),
            v.dim()
        )));
    }
    if p.ncols() != manifold.ambient_dim() {
        return Err(py_value_error(format!(
            "manifold_exp_map: points width {} does not match manifold ambient_dim {}",
            p.ncols(),
            manifold.ambient_dim()
        )));
    }
    let mut out = Array2::<f64>::zeros(p.dim());
    for row in 0..p.nrows() {
        let next = manifold
            .exp_map(p.row(row), v.row(row))
            .map_err(|err| py_value_error(err.to_string()))?;
        out.row_mut(row).assign(&next);
    }
    Ok(out.into_pyarray(py).unbind())
}

/// Batched vector–Jacobian product of the Riemannian exponential map.
///
/// Given base points ``points`` ``(N, ambient_dim)``, raw tangent inputs
/// ``vecs`` ``(N, ambient_dim)``, and a cotangent ``grad_output``
/// ``(N, ambient_dim)`` w.r.t. the output of :func:`manifold_exp_map`, returns
/// ``(grad_points, grad_vecs)`` — the analytic pullbacks w.r.t. ``points`` and
/// ``vecs``. This is the backward used by the Python ``torch.autograd.Function``
/// wrapping ``manifold_exp_map``; it routes through the canonical Rust
/// ``RiemannianManifold::exp_map_vjp`` so Sphere, Grassmann, Stiefel, SPD, and
/// products of supported components get their true analytic pullbacks instead
/// of a silent straight-through identity.
#[pyfunction(signature = (manifold_json, points, vecs, grad_output))]
fn manifold_exp_map_vjp<'py>(
    py: Python<'py>,
    manifold_json: &str,
    points: PyReadonlyArray2<'py, f64>,
    vecs: PyReadonlyArray2<'py, f64>,
    grad_output: PyReadonlyArray2<'py, f64>,
) -> PyResult<(Py<PyArray2<f64>>, Py<PyArray2<f64>>)> {
    let value: serde_json::Value = serde_json::from_str(manifold_json)
        .map_err(|err| py_value_error(format!("invalid manifold json: {err}")))?;
    let kind = parse_manifold_kind(&value).map_err(py_value_error)?;
    let manifold = kind
        .build()
        .map_err(|err| py_value_error(err.to_string()))?;
    let p = points.as_array();
    let v = vecs.as_array();
    let g = grad_output.as_array();
    if p.dim() != v.dim() {
        return Err(py_value_error(format!(
            "manifold_exp_map_vjp: points shape {:?} does not match vecs shape {:?}",
            p.dim(),
            v.dim()
        )));
    }
    if p.dim() != g.dim() {
        return Err(py_value_error(format!(
            "manifold_exp_map_vjp: points shape {:?} does not match grad_output shape {:?}",
            p.dim(),
            g.dim()
        )));
    }
    if p.ncols() != manifold.ambient_dim() {
        return Err(py_value_error(format!(
            "manifold_exp_map_vjp: points width {} does not match manifold ambient_dim {}",
            p.ncols(),
            manifold.ambient_dim()
        )));
    }
    let mut grad_points = Array2::<f64>::zeros(p.dim());
    let mut grad_vecs = Array2::<f64>::zeros(p.dim());
    for row in 0..p.nrows() {
        let (gp, gv) = manifold
            .exp_map_vjp(p.row(row), v.row(row), g.row(row))
            .map_err(|err| py_value_error(err.to_string()))?;
        grad_points.row_mut(row).assign(&gp);
        grad_vecs.row_mut(row).assign(&gv);
    }
    Ok((
        grad_points.into_pyarray(py).unbind(),
        grad_vecs.into_pyarray(py).unbind(),
    ))
}

/// Batched Riemannian log map: for each row, return ``log_p(q)``.
#[pyfunction(signature = (manifold_json, p_from, p_to))]
fn manifold_log_map<'py>(
    py: Python<'py>,
    manifold_json: &str,
    p_from: PyReadonlyArray2<'py, f64>,
    p_to: PyReadonlyArray2<'py, f64>,
) -> PyResult<Py<PyArray2<f64>>> {
    let value: serde_json::Value = serde_json::from_str(manifold_json)
        .map_err(|err| py_value_error(format!("invalid manifold json: {err}")))?;
    let kind = parse_manifold_kind(&value).map_err(py_value_error)?;
    let manifold = kind
        .build()
        .map_err(|err| py_value_error(err.to_string()))?;
    let p = p_from.as_array();
    let q = p_to.as_array();
    if p.dim() != q.dim() {
        return Err(py_value_error(format!(
            "manifold_log_map: p_from shape {:?} does not match p_to shape {:?}",
            p.dim(),
            q.dim()
        )));
    }
    if p.ncols() != manifold.ambient_dim() {
        return Err(py_value_error(format!(
            "manifold_log_map: point width {} does not match manifold ambient_dim {}",
            p.ncols(),
            manifold.ambient_dim()
        )));
    }
    let mut out = Array2::<f64>::zeros(p.dim());
    for row in 0..p.nrows() {
        let vec = manifold
            .log_map(p.row(row), q.row(row))
            .map_err(|err| py_value_error(err.to_string()))?;
        out.row_mut(row).assign(&vec);
    }
    Ok(out.into_pyarray(py).unbind())
}

/// Metric tensor at a single point: ``(ambient_dim, ambient_dim)`` ndarray.
#[pyfunction(signature = (manifold_json, point))]
fn manifold_metric_tensor<'py>(
    py: Python<'py>,
    manifold_json: &str,
    point: PyReadonlyArray1<'py, f64>,
) -> PyResult<Py<PyArray2<f64>>> {
    let value: serde_json::Value = serde_json::from_str(manifold_json)
        .map_err(|err| py_value_error(format!("invalid manifold json: {err}")))?;
    let kind = parse_manifold_kind(&value).map_err(py_value_error)?;
    let manifold = kind
        .build()
        .map_err(|err| py_value_error(err.to_string()))?;
    let p = point.as_array();
    if p.len() != manifold.ambient_dim() {
        return Err(py_value_error(format!(
            "manifold_metric_tensor: point length {} does not match manifold ambient_dim {}",
            p.len(),
            manifold.ambient_dim()
        )));
    }
    let g = manifold
        .metric_tensor(p)
        .map_err(|err| py_value_error(err.to_string()))?;
    Ok(g.into_pyarray(py).unbind())
}

/// Manifold intrinsic dimension parsed from the JSON descriptor.
#[pyfunction(signature = (manifold_json))]
fn manifold_dimension(manifold_json: &str) -> PyResult<usize> {
    let value: serde_json::Value = serde_json::from_str(manifold_json)
        .map_err(|err| py_value_error(format!("invalid manifold json: {err}")))?;
    let kind = parse_manifold_kind(&value).map_err(py_value_error)?;
    let manifold = kind
        .build()
        .map_err(|err| py_value_error(err.to_string()))?;
    Ok(manifold.dim())
}

/// Manifold ambient dimension parsed from the JSON descriptor.
#[pyfunction(signature = (manifold_json))]
fn manifold_ambient_dimension(manifold_json: &str) -> PyResult<usize> {
    let value: serde_json::Value = serde_json::from_str(manifold_json)
        .map_err(|err| py_value_error(format!("invalid manifold json: {err}")))?;
    let kind = parse_manifold_kind(&value).map_err(py_value_error)?;
    let manifold = kind
        .build()
        .map_err(|err| py_value_error(err.to_string()))?;
    Ok(manifold.ambient_dim())
}

/// Evaluate the periodic harmonic (Fourier) basis at angles ``theta`` (radians).
/// Returns ``(N, 2*harmonics + 1)`` ndarray with columns
/// ``[1, cos(theta), sin(theta), cos(2 theta), sin(2 theta), ...]``.
#[pyfunction(signature = (theta, harmonics))]
fn periodic_harmonic_basis<'py>(
    py: Python<'py>,
    theta: PyReadonlyArray1<'py, f64>,
    harmonics: usize,
) -> PyResult<Py<PyArray2<f64>>> {
    let t = theta.as_array();
    let n = t.len();
    let width = 2 * harmonics + 1;
    let mut out = Array2::<f64>::zeros((n, width));
    for (row, &angle) in t.iter().enumerate() {
        out[(row, 0)] = 1.0;
        for h in 1..=harmonics {
            let arg = (h as f64) * angle;
            out[(row, 2 * h - 1)] = arg.cos();
            out[(row, 2 * h)] = arg.sin();
        }
    }
    Ok(out.into_pyarray(py).unbind())
}

/// Derivative of :func:`periodic_harmonic_basis` with respect to ``theta``.
#[pyfunction(signature = (theta, harmonics))]
fn periodic_harmonic_basis_derivative<'py>(
    py: Python<'py>,
    theta: PyReadonlyArray1<'py, f64>,
    harmonics: usize,
) -> PyResult<Py<PyArray2<f64>>> {
    let t = theta.as_array();
    let n = t.len();
    let width = 2 * harmonics + 1;
    let mut out = Array2::<f64>::zeros((n, width));
    for (row, &angle) in t.iter().enumerate() {
        for h in 1..=harmonics {
            let arg = (h as f64) * angle;
            out[(row, 2 * h - 1)] = -(h as f64) * arg.sin();
            out[(row, 2 * h)] = (h as f64) * arg.cos();
        }
    }
    Ok(out.into_pyarray(py).unbind())
}

fn parse_fit_config(config_json: Option<&str>) -> Result<FitConfig, String> {
    gam::config_resolve::parse_fit_config_json(config_json)
}

fn request_metadata(request: &FitRequest<'_>) -> (&'static str, &'static str, bool) {
    match request {
        FitRequest::Standard(standard_request) => {
            (standard_request.family.pretty_name(), "standard", true)
        }
        FitRequest::GaussianLocationScale(_) => {
            ("Gaussian location-scale", "gaussian location-scale", true)
        }
        FitRequest::BinomialLocationScale(_) => {
            ("Binomial location-scale", "binomial location-scale", true)
        }
        FitRequest::SurvivalLocationScale(_) => {
            ("Survival location-scale", "survival location-scale", true)
        }
        FitRequest::SurvivalTransformation(request) => {
            // cause_count is Result<usize, SurvivalError>; on error fall through
            // to the non-competing-risks branches (display-only path).
            let cause_count = gam::families::survival::cause_count_from_event_codes(
                request.spec.event_target.view(),
            )
            .unwrap_or(1);
            if cause_count > 1 {
                ("Cause-specific survival", "competing risks survival", true)
            } else {
                match request.spec.likelihood_mode {
                    gam::families::survival::construction::SurvivalLikelihoodMode::Weibull => {
                        ("Survival Weibull", "survival", true)
                    }
                    _ => ("Survival", "survival", true),
                }
            }
        }
        FitRequest::BernoulliMarginalSlope(_) => {
            ("Bernoulli marginal-slope", "bernoulli marginal-slope", true)
        }
        FitRequest::SurvivalMarginalSlope(_) => {
            ("Survival marginal-slope", "survival marginal-slope", true)
        }
        FitRequest::LatentSurvival(_) => ("Latent survival", "latent survival", true),
        FitRequest::LatentBinary(_) => ("Latent binary", "latent binary", true),
        FitRequest::TransformationNormal(_) => {
            ("Transformation-normal", "transformation-normal", true)
        }
        FitRequest::DispersionLocationScale(_) => (
            "Dispersion location-scale",
            "dispersion location-scale",
            true,
        ),
    }
}

fn parse_predict_options(options_json: Option<&str>) -> Result<PyPredictOptions, String> {
    let options = match options_json {
        Some(raw) if !raw.trim().is_empty() => serde_json::from_str::<PyPredictOptions>(raw)
            .map_err(|err| format!("invalid predict options json: {err}"))?,
        _ => PyPredictOptions::default(),
    };
    if let Some(interval) = options.interval {
        if !(0.0 < interval && interval < 1.0) {
            return Err(format!(
                "prediction interval must be in (0, 1); got {interval}"
            ));
        }
    }
    Ok(options)
}

fn dataset_with_model_schema(
    model: &FittedModel,
    headers: &[String],
    rows: &[Vec<String>],
) -> Result<EncodedDataset, String> {
    // Headers-only validation up-front (mirrors the missing-column path in
    // schema_check). We deliberately skip the schema_check call here because
    // schema_check would otherwise do a full encode internally, doubling the
    // O(N·p) FFI ingest cost. Any encoding errors below surface directly.
    let expected_names = required_prediction_columns(model)?;
    let present_names = headers.iter().cloned().collect::<BTreeSet<_>>();
    let missing = expected_names
        .difference(&present_names)
        .map(|name| format!("missing required column '{name}'"))
        .collect::<Vec<_>>();
    if !missing.is_empty() {
        return Err(missing.join(" "));
    }
    // Drop columns the model does not reference *before* encoding, so an
    // unrelated ID/label column with a held-out categorical level is ignored
    // rather than aborting predict against the training schema (#840).
    let (headers, rows) = project_frame_to_model_columns(model, headers, rows)?;
    let schema = model.require_data_schema()?;
    let records = string_records_from_rows(&headers, &rows)?;
    let policy =
        UnseenCategoryPolicy::encode_unknown_for_columns(model.random_effect_group_columns());
    encode_recordswith_schema(headers, records, schema, policy)
}

fn dataset_from_xy_arrays(
    x: ArrayView2<'_, f64>,
    y: ArrayView2<'_, f64>,
    formula: &str,
) -> Result<EncodedDataset, String> {
    if x.nrows() == 0 || y.nrows() == 0 {
        return Err("array data cannot be empty".to_string());
    }
    if x.nrows() != y.nrows() {
        return Err(format!(
            "X/Y row mismatch: X has {} rows but Y has {} rows",
            x.nrows(),
            y.nrows()
        ));
    }
    if x.ncols() == 0 {
        return Err("X must have at least one column".to_string());
    }
    if y.ncols() == 0 {
        return Err("Y must have at least one column".to_string());
    }
    let response_name = response_column_name(formula).unwrap_or_else(|| "y".to_string());
    let mut headers = Vec::<String>::with_capacity(y.ncols() + x.ncols());
    if y.ncols() == 1 {
        headers.push(response_name);
    } else {
        headers.extend((0..y.ncols()).map(|index| format!("y{index}")));
    }
    headers.extend((0..x.ncols()).map(|index| format!("x{index}")));
    ensure_unique_headers(&headers)?;

    let mut values = Array2::<f64>::zeros((x.nrows(), headers.len()));
    values.slice_mut(ndarray::s![.., 0..y.ncols()]).assign(&y);
    values.slice_mut(ndarray::s![.., y.ncols()..]).assign(&x);
    dataset_from_numeric_array(headers, values)
}

fn dataset_from_x_array_with_model_schema(
    model: &FittedModel,
    x: ArrayView2<'_, f64>,
) -> Result<EncodedDataset, String> {
    if x.nrows() == 0 {
        return Err("array data cannot be empty".to_string());
    }
    if x.ncols() == 0 {
        return Err("X must have at least one column".to_string());
    }
    // Issue #341 — principled scope for predict_array:
    //
    // Positional X has a well-defined meaning only when the model itself
    // was fit from a positional array (i.e. via `fit_array`), because
    // `fit_array` is the only path that synthesizes a known column ordering
    // (`x0, x1, ..., x{p-1}`) and bakes that ordering into both the saved
    // schema and the user's formula. For models fit from a named table
    // (`fit(df, formula)`) the schema speaks the user's own column names
    // and there is NO way at the `.values` boundary to tell `df[["a","b"]]`
    // from `df[["b","a"]]` apart — both arrive as an anonymous (n, 2) f64
    // block. Quietly choosing one ordering (e.g. "training_headers order")
    // produces silently wrong predictions on column swap, which is exactly
    // the kind of footgun a principled API must refuse. So: refuse.
    //
    // Detection is cheap and exact. `fit_array` writes training_headers as
    // `[<response>, "x0", "x1", ..., "x{p-1}"]` (or `[y0..yk-1, x0..xp-1]`
    // for multi-response), which means the predictor-column suffix of
    // training_headers is exactly the sorted positional sequence
    // `x0..x{p-1}`. Anything else is a table-fit and must use
    // `Model.predict(df_or_dict)` instead.
    let schema = model.require_data_schema()?;
    let payload = model.payload();
    let training_headers = payload.training_headers.as_ref().ok_or_else(|| {
        "predict_array requires a model fitted via fit_array (no training headers saved); \
             call Model.predict(df_or_dict) instead so predictor columns can be matched by name"
            .to_string()
    })?;
    let required = required_prediction_columns(model)?;
    let positional_feature_names: Vec<String> = training_headers
        .iter()
        .filter(|name| required.contains(name.as_str()))
        .cloned()
        .collect();
    let expected_positional: Vec<String> = (0..positional_feature_names.len())
        .map(|index| format!("x{index}"))
        .collect();
    if positional_feature_names != expected_positional {
        return Err(format!(
            "predict_array is only defined for models fitted via fit_array \
             (predictor columns must be the synthetic positional sequence \
             x0..x{{p-1}}); this model was fitted from a named table with \
             predictor columns {positional_feature_names:?}. Call \
             Model.predict(df_or_dict) instead so columns can be matched by name."
        ));
    }
    if positional_feature_names.len() != x.ncols() {
        return Err(format!(
            "predict_array expected {} positional feature column(s) to match \
             the fit_array training schema, but the input X has {} column(s)",
            positional_feature_names.len(),
            x.ncols()
        ));
    }
    dataset_from_numeric_array_with_schema(positional_feature_names, x.to_owned(), schema)
}

fn dataset_from_numeric_array(
    headers: Vec<String>,
    values: Array2<f64>,
) -> Result<EncodedDataset, String> {
    ensure_unique_headers(&headers)?;
    validate_numeric_array_values(&headers, values.view())?;
    let mut schema_cols = Vec::<SchemaColumn>::with_capacity(headers.len());
    let mut column_kinds = Vec::<ColumnKindTag>::with_capacity(headers.len());
    for (j, name) in headers.iter().enumerate() {
        let kind = infer_numeric_array_column_kind(values.column(j));
        column_kinds.push(kind);
        schema_cols.push(SchemaColumn {
            name: name.clone(),
            kind,
            levels: Vec::new(),
        });
    }
    Ok(EncodedDataset {
        headers,
        values,
        schema: DataSchema {
            columns: schema_cols,
        },
        column_kinds,
    })
}

fn dataset_from_numeric_array_with_schema(
    headers: Vec<String>,
    values: Array2<f64>,
    schema: &DataSchema,
) -> Result<EncodedDataset, String> {
    ensure_unique_headers(&headers)?;
    validate_numeric_array_values(&headers, values.view())?;
    let schema_byname = schema
        .columns
        .iter()
        .map(|column| (column.name.as_str(), column))
        .collect::<HashMap<_, _>>();
    let mut schema_cols = Vec::<SchemaColumn>::with_capacity(headers.len());
    let mut column_kinds = Vec::<ColumnKindTag>::with_capacity(headers.len());
    for (j, name) in headers.iter().enumerate() {
        let column = schema_byname.get(name.as_str()).ok_or_else(|| {
            format!("array column '{name}' was not present in the training schema")
        })?;
        match column.kind {
            ColumnKindTag::Categorical => {
                return Err(format!(
                    "array FFI only supports numeric continuous/binary columns; column '{name}' is categorical in the training schema"
                ));
            }
            ColumnKindTag::Binary => {
                for (row, value) in values.column(j).iter().enumerate() {
                    if (*value - 0.0).abs() >= 1e-12 && (*value - 1.0).abs() >= 1e-12 {
                        return Err(format!(
                            "column '{name}' is binary in schema but row {} has value {}; expected 0 or 1",
                            row + 1,
                            value
                        ));
                    }
                }
            }
            ColumnKindTag::Continuous => {}
        }
        column_kinds.push(column.kind);
        schema_cols.push((*column).clone());
    }
    Ok(EncodedDataset {
        headers,
        values,
        schema: DataSchema {
            columns: schema_cols,
        },
        column_kinds,
    })
}

fn ensure_unique_headers(headers: &[String]) -> Result<(), String> {
    if headers.is_empty() {
        return Err("table must have at least one column".to_string());
    }
    if headers.iter().any(|header| header.trim().is_empty()) {
        return Err("table headers must be non-empty strings".to_string());
    }
    let mut unique_headers = BTreeSet::<String>::new();
    for header in headers {
        if !unique_headers.insert(header.clone()) {
            return Err(format!("duplicate column name '{header}'"));
        }
    }
    Ok(())
}

fn validate_numeric_array_values(
    headers: &[String],
    values: ArrayView2<'_, f64>,
) -> Result<(), String> {
    if values.nrows() == 0 {
        return Err("array data cannot be empty".to_string());
    }
    if values.ncols() != headers.len() {
        return Err(format!(
            "array column count {} does not match header count {}",
            values.ncols(),
            headers.len()
        ));
    }
    for ((row, col), value) in values.indexed_iter() {
        if !value.is_finite() {
            return Err(format!(
                "non-finite value at row {}, column '{}'",
                row + 1,
                headers[col]
            ));
        }
    }
    Ok(())
}

fn infer_numeric_array_column_kind(column: ArrayView1<'_, f64>) -> ColumnKindTag {
    if column
        .iter()
        .all(|value| (*value - 0.0).abs() < 1e-12 || (*value - 1.0).abs() < 1e-12)
    {
        ColumnKindTag::Binary
    } else {
        ColumnKindTag::Continuous
    }
}

fn response_column_name(formula: &str) -> Option<String> {
    let candidate = formula.split('~').next()?.trim();
    if candidate.is_empty() || candidate.starts_with("Surv(") {
        None
    } else {
        Some(candidate.to_string())
    }
}

fn prediction_model_class_label(model: &FittedModel) -> String {
    let payload = model.payload();
    match &payload.family_state {
        FittedFamily::Survival {
            survival_likelihood,
            ..
        } => match survival_likelihood
            .as_deref()
            .or(payload.survival_likelihood.as_deref())
        {
            _ if payload
                .survival_cause_count
                .is_some_and(|cause_count| cause_count > 1) =>
            {
                "competing risks survival".to_string()
            }
            Some("marginal-slope") => "survival marginal-slope".to_string(),
            Some("location-scale") => "survival location-scale".to_string(),
            None
            | Some("latent")
            | Some("latent-binary")
            | Some("transformation")
            | Some("weibull")
            | Some("royston-parmar")
            | Some(_) => model.predict_model_class().name().to_string(),
        },
        FittedFamily::LatentSurvival { .. } => "latent survival".to_string(),
        FittedFamily::LatentBinary { .. } => "latent binary".to_string(),
        FittedFamily::Standard { .. }
        | FittedFamily::LocationScale { .. }
        | FittedFamily::MarginalSlope { .. }
        | FittedFamily::TransformationNormal { .. } => {
            model.predict_model_class().name().to_string()
        }
    }
}

/// The columns a prediction frame *must* carry for this model. Delegates to the
/// single shared authority on the formula→columns contract
/// (`FittedModel::prediction_required_columns`) so the CLI and PyFFI predict
/// paths agree exactly (including a smooth's `by=` grouping column).
fn required_prediction_columns(model: &FittedModel) -> Result<BTreeSet<String>, String> {
    model.prediction_required_columns()
}

/// Columns a fitted model can legitimately consume from a prediction frame.
///
/// This is the model's *input contract*: every variable the formula names
/// (features, interaction margins, random-effect grouping columns, a smooth's
/// `by=` column), plus the offset / noise-offset / latent-`z` / survival
/// entry-exit columns surfaced by [`required_prediction_columns`], plus the
/// response column (needed for the conformal-calibration fold and for survival
/// / transformation-normal label-bearing frames), plus the prior-weights column
/// when the model was fitted with one (needed by the generative-replicate path
/// to reconstruct heteroskedastic observation noise `Var(y_i)=sigma^2/w_i`,
/// #2025/#2033).
///
/// Any column *not* in this set is irrelevant to the model — a row ID, a
/// grouping/label column carried for bookkeeping, an auxiliary measurement —
/// and must be ignored at predict time rather than strictly re-encoded against
/// the training schema. See [`project_frame_to_model_columns`] for why.
fn prediction_consumable_columns(model: &FittedModel) -> Result<BTreeSet<String>, String> {
    let mut consumable = required_prediction_columns(model)?;
    if let Some(response) = response_column_name(model.payload().formula.as_str()) {
        consumable.insert(response);
    }
    // Retain the prior-weights column when the model carried one, so it survives
    // projection and the replicate path can resolve per-row weights rather than
    // erroring on a frame that *does* include them (#2033 regression of #2025).
    // Harmless for ordinary predict, which never resolves the weight column.
    if let Some(weight) = model.weight_column.as_deref() {
        consumable.insert(weight.to_string());
    }
    Ok(consumable)
}

/// Project a prediction frame onto the columns the model actually references,
/// preserving the input column order.
///
/// A fitted GAM is a function of exactly the variables in its formula (+
/// offset / weights / response). Columns the model never references must not
/// participate in prediction: in particular they must *not* be strictly
/// re-encoded against the training schema. Without this projection, an
/// unrelated categorical column — e.g. a `color`/`group`/`id` label kept in
/// the frame for downstream bookkeeping — is re-validated against the training
/// levels, so a held-out CV fold carrying a brand-new level aborts predict
/// with `unseen level '…' in categorical column '…'` (the classic
/// leave-one-group-out foot-gun, #840). Dropping such columns here mirrors
/// mgcv / glm / scikit-learn `Pipeline` semantics, where extra frame columns
/// are simply ignored.
///
/// All downstream prediction machinery resolves columns *by name* (the term
/// spec is remapped through `training_headers` → name → prediction `col_map`,
/// and offset/clip resolution look columns up by name with a graceful skip),
/// so narrowing the encoded dataset to a subset of named columns is safe.
fn project_frame_to_model_columns(
    model: &FittedModel,
    headers: &[String],
    rows: &[Vec<String>],
) -> Result<(Vec<String>, Vec<Vec<String>>), String> {
    let consumable = prediction_consumable_columns(model)?;
    let keep: Vec<usize> = headers
        .iter()
        .enumerate()
        .filter(|(_, name)| consumable.contains(name.as_str()))
        .map(|(idx, _)| idx)
        .collect();
    // Keep the frame verbatim when there is nothing to drop (the common case
    // where the caller passes exactly the model's columns), or when a row's
    // width disagrees with the header count — in the latter case we defer to
    // the canonical width validation in `string_records_from_rows` instead of
    // risking an out-of-bounds projection here.
    let width_consistent = rows.iter().all(|row| row.len() == headers.len());
    // Never collapse the frame to zero columns: a covariate-free model
    // (`y ~ 1`) consumes none of the held-out frame's columns, but the frame
    // still carries the one thing prediction needs — its row count. Dropping
    // every column hands `string_records_from_rows` an empty table and aborts
    // with "table must have at least one column" on a perfectly valid frame
    // (#1316). Keeping the frame verbatim is safe because every downstream
    // consumer resolves columns by name and an intercept-only model references
    // none of them.
    if keep.is_empty() || keep.len() == headers.len() || !width_consistent {
        return Ok((headers.to_vec(), rows.to_vec()));
    }
    let filtered_headers = keep.iter().map(|&i| headers[i].clone()).collect::<Vec<_>>();
    let filtered_rows = rows
        .iter()
        .map(|row| keep.iter().map(|&i| row[i].clone()).collect::<Vec<_>>())
        .collect::<Vec<_>>();
    Ok((filtered_headers, filtered_rows))
}

fn string_records_from_rows(
    headers: &[String],
    rows: &[Vec<String>],
) -> Result<Vec<StringRecord>, String> {
    if headers.is_empty() {
        return Err("table must have at least one column".to_string());
    }
    if headers.iter().any(|header| header.trim().is_empty()) {
        return Err("table headers must be non-empty strings".to_string());
    }
    let mut unique_headers = BTreeSet::<String>::new();
    for header in headers {
        if !unique_headers.insert(header.clone()) {
            return Err(format!("duplicate column name '{header}'"));
        }
    }
    rows.iter()
        .enumerate()
        .map(|(index, row)| {
            if row.len() != headers.len() {
                return Err(format!(
                    "row {} has width {} but expected {}",
                    index + 1,
                    row.len(),
                    headers.len()
                ));
            }
            // A typed Python frame marks categorical-dtype cells with a leading
            // sentinel (forces factor inference at fit time, #1317/#1318). The
            // saved schema stores clean level labels, so strip the marker here
            // before the schema-guided encode matches a cell against a level.
            let cleaned: Vec<&str> = row
                .iter()
                .map(|cell| gam::data::strip_categorical_sentinel(cell).0)
                .collect();
            Ok(StringRecord::from(cleaned))
        })
        .collect()
}

fn periodic_bspline_basis_dense_via_spec(
    t: ArrayView1<'_, f64>,
    domain: (f64, f64),
    degree: usize,
    num_basis: usize,
) -> Result<Array2<f64>, String> {
    let (left, right) = domain;
    let period = right - left;
    if !(period.is_finite() && period > 0.0) {
        return Err(format!(
            "periodic B-spline domain must be a finite ordered interval; got ({left}, {right})"
        ));
    }
    // The FFI returns only the dense value basis, but the shared periodic spec
    // validator requires a realizable derivative order. Use curvature when
    // the polynomial degree supports it and slope roughness otherwise.
    let penalty_order = degree.min(2);
    let spec = PeriodicBSplineBasisSpec::new(degree, num_basis, period, left, penalty_order);
    build_periodic_bspline_basis_1d(t, &spec)
        .map_err(|err| format!("failed to evaluate periodic B-spline basis: {err}"))
}

/// Dense `(N, K)` periodic cyclic B-spline derivative of the requested
/// `order`, on the closed parameter circle `domain = (left, right)` with
/// `num_basis` cyclic control points.
///
/// `order == 0` returns the periodic value basis (the partition of unity);
/// `order == 1` returns the exact closed-form first derivative by squeezing
/// the `(N, K, 1)` jet from `periodic_bspline_first_derivative_nd` — the same
/// jet `basis_with_jet` and `PeriodicSplineCurve::evaluate_derivative` rely
/// on, so the dense matrix and the modelling path agree to machine precision.
/// Because the value basis is a partition of unity, each derivative row sums
/// to ~0. Orders ≥ 2 have no exposed periodic jet and are rejected with a
/// precise message rather than the old blanket "no longer exposed" error.
fn periodic_bspline_derivative_dense(
    t: ArrayView1<'_, f64>,
    domain: (f64, f64),
    degree: usize,
    num_basis: usize,
    order: usize,
) -> Result<Array2<f64>, String> {
    match order {
        0 => periodic_bspline_basis_dense_via_spec(t, domain, degree, num_basis),
        1 => {
            let coords = column_array(t);
            let jet =
                periodic_bspline_first_derivative_nd(coords.view(), domain, degree, num_basis)
                    .map_err(|err| {
                        format!("failed to evaluate periodic B-spline derivative: {err}")
                    })?;
            Ok(jet.index_axis_move(Axis(2), 0))
        }
        _ => Err(format!(
            "periodic B-spline derivative is available in closed form for order 0 (value) \
             and order 1 (first derivative); order={order} (second and higher derivatives) \
             is not exposed for the periodic cyclic basis"
        )),
    }
}

fn bspline_basis_impl(
    t: ArrayView1<'_, f64>,
    knots: ArrayView1<'_, f64>,
    degree: usize,
    periodic: bool,
) -> Result<Array2<f64>, String> {
    validate_vector("t", t)?;
    validate_vector("knots", knots)?;
    if periodic {
        let (left, right, num_basis) = periodic_knot_domain(knots)?;
        periodic_bspline_basis_dense_via_spec(t, (left, right), degree, num_basis)
    } else {
        let (basis, _) = create_basis::<Dense>(
            t,
            gam::terms::basis::KnotSource::Provided(knots),
            degree,
            BasisOptions::value(),
        )
        .map_err(|err| format!("failed to evaluate B-spline basis: {err}"))?;
        Ok((*basis).clone())
    }
}

fn bspline_basis_derivative_impl(
    t: ArrayView1<'_, f64>,
    knots: ArrayView1<'_, f64>,
    degree: usize,
    order: usize,
    periodic: bool,
) -> Result<Array2<f64>, String> {
    validate_vector("t", t)?;
    validate_vector("knots", knots)?;
    if periodic {
        let (left, right, num_basis) = periodic_knot_domain(knots)?;
        return periodic_bspline_derivative_dense(t, (left, right), degree, num_basis, order);
    }
    let options = match order {
        0 => BasisOptions::value(),
        1 => BasisOptions::first_derivative(),
        2 => BasisOptions::second_derivative(),
        _ => {
            return Err(format!(
                "B-spline derivative supports orders 0, 1, and 2; got order={order}"
            ));
        }
    };
    let (basis, _) = create_basis::<Dense>(
        t,
        gam::terms::basis::KnotSource::Provided(knots),
        degree,
        options,
    )
    .map_err(|err| format!("failed to evaluate B-spline derivative: {err}"))?;
    Ok((*basis).clone())
}

/// Compute the data-metric Duchon radial chart once over the complete position
/// collection. Strict operator policy forces the raw-Gram pass through row
/// chunks, so this never constructs the concatenated `n × p` design that the
/// position-batched API is designed to avoid.
fn duchon_position_radial_reparam_streamed(
    t: ArrayView1<'_, f64>,
    centers: ArrayView1<'_, f64>,
    m: usize,
    periodic: bool,
    period: Option<f64>,
) -> Result<Option<Array2<f64>>, String> {
    validate_vector("t", t)?;
    validate_vector("centers", centers)?;
    if m == 0 {
        return Err("Duchon m must be at least 1".to_string());
    }
    if periodic {
        return Ok(None);
    }
    if period.is_some() {
        return Err("Duchon period is only valid when periodic=true".to_string());
    }
    let data = column_array(t);
    let center_matrix = column_array(centers);
    let spec = DuchonBasisSpec {
        radial_reparam: None,
        center_strategy: CenterStrategy::UserProvided(center_matrix),
        periodic: None,
        length_scale: None,
        power: 0.0,
        nullspace_order: duchon_nullspace_from_m(m),
        identifiability: SpatialIdentifiability::None,
        aniso_log_scales: None,
        operator_penalties: Default::default(),
        boundary: OneDimensionalBoundary::Open,
    };
    let mut workspace = gam::terms::basis::BasisWorkspace::with_policy(
        gam::ResourcePolicy::analytic_operator_required(),
    );
    let built =
        gam::terms::basis::build_duchon_basiswithworkspace(data.view(), &spec, &mut workspace)
            .map_err(|err| format!("failed to freeze global Duchon radial chart: {err}"))?;
    match built.metadata {
        gam::terms::basis::BasisMetadata::Duchon { radial_reparam, .. } => Ok(radial_reparam),
        other => Err(format!(
            "global Duchon radial-chart build returned unexpected metadata {:?}",
            std::mem::discriminant(&other)
        )),
    }
}

fn duchon_basis_1d_impl_with_radial_reparam(
    t: ArrayView1<'_, f64>,
    centers: ArrayView1<'_, f64>,
    m: usize,
    periodic: bool,
    period: Option<f64>,
    radial_reparam: Option<&Array2<f64>>,
) -> Result<Array2<f64>, String> {
    validate_vector("t", t)?;
    validate_vector("centers", centers)?;
    if m == 0 {
        return Err("Duchon m must be at least 1".to_string());
    }
    if periodic && radial_reparam.is_some() {
        return Err(
            "periodic Duchon positions do not admit an open-domain radial reparameterization"
                .to_string(),
        );
    }
    let data = column_array(t);
    let center_matrix = column_array(centers);
    let spec = DuchonBasisSpec {
        radial_reparam: radial_reparam.cloned(),
        center_strategy: CenterStrategy::UserProvided(center_matrix),
        periodic: None,
        length_scale: None,
        power: 0.0,
        nullspace_order: duchon_nullspace_from_m(m),
        identifiability: SpatialIdentifiability::None,
        aniso_log_scales: None,
        operator_penalties: Default::default(),
        boundary: OneDimensionalBoundary::Open,
    };
    if periodic {
        // Honor the explicit domain-wrap `period`; auto-derive (None) only when
        // the caller did not supply one. Matches the penalty path (gam#580).
        let periods_1d: Option<[f64; 1]> = period.map(|p| [p]);
        let built = build_duchon_basis_mixed_periodicity_auto(
            data.view(),
            &spec,
            &[true],
            periods_1d.as_ref().map(|p| p.as_slice()),
        )
        .map_err(|err| format!("failed to evaluate Duchon basis: {err}"))?;
        return built
            .design
            .try_to_dense_by_chunks("duchon_basis_1d_impl")
            .map_err(|err| format!("failed to evaluate Duchon basis: {err}"));
    }
    let built = build_duchon_basis(data.view(), &spec)
        .map_err(|err| format!("failed to evaluate Duchon basis: {err}"))?;
    built
        .design
        .try_to_dense_by_chunks("duchon_basis_1d_impl")
        .map_err(|err| format!("failed to evaluate Duchon basis: {err}"))
}

fn duchon_basis_1d_derivative_impl_with_radial_reparam(
    t: ArrayView1<'_, f64>,
    centers: ArrayView1<'_, f64>,
    m: usize,
    order: usize,
    periodic: bool,
    period: Option<f64>,
    radial_reparam: Option<&Array2<f64>>,
) -> Result<Array2<f64>, String> {
    validate_vector("t", t)?;
    validate_vector("centers", centers)?;
    if m == 0 {
        return Err("Duchon m must be at least 1".to_string());
    }
    gam::terms::basis::create_duchon_basis_1d_derivative_dense_with_radial_reparam(
        t,
        centers,
        0.0,
        duchon_nullspace_from_m(m),
        periodic,
        if periodic { period } else { None },
        radial_reparam.map(|v| v.view()),
        order,
    )
    .map_err(|err| format!("failed to evaluate Duchon basis derivative: {err}"))
}

fn smoothness_penalty_impl(
    knots: ArrayView1<'_, f64>,
    degree: usize,
    order: usize,
) -> Result<(Array2<f64>, Array2<f64>), String> {
    validate_vector("knots", knots)?;
    if knots.len() <= degree + 1 {
        return Err(format!(
            "knot vector is too short for degree={degree}: got {} knots",
            knots.len()
        ));
    }
    let penalty = bspline_derivative_penalty_matrix(knots, degree, order)
        .map_err(|err| format!("failed to build smoothness penalty: {err}"))?;
    let (null_basis, _) = gam::linalg::faer_ndarray::rrqr_nullspace_basis(
        &penalty,
        gam::linalg::faer_ndarray::default_rrqr_rank_alpha(),
    )
    .map_err(|err| format!("failed to build penalty null basis: {err}"))?;
    Ok((penalty, null_basis))
}

#[cfg(test)]
mod latent_stationarity_tests {
    //! Regression tests for the latent outer-solve convergence test (issue
    //! #954). `gaussian_reml_optimize_latent` decides `converged` from
    //! [`latent_relative_stationarity`], which must be shift-invariant (the
    //! objective value cannot enter), scale-invariant, preserve the #879 O(n)
    //! calibration via the initial-iterate gradient, and never certify a
    //! blown-up iterate.
    use super::latent_relative_stationarity;

    const GRAD_TOL: f64 = 1.0e-8;

    /// #954 from the FFI angle: the convergence decision must not depend on the
    /// objective *magnitude*. The retired formula was
    /// `rel_old = ‖∇f‖ · ‖t‖_typ / max(|f|, 1)`; a large objective value drove
    /// it below `grad_tol` and falsely certified a non-stationary latent. The
    /// new measure does not take `f` at all, so for a FIXED gradient pair it is
    /// constant across every objective value — and a genuinely non-stationary
    /// gradient ratio never passes, no matter how large `|f|` grows.
    #[test]
    fn convergence_is_invariant_to_objective_shift() {
        // A latent that is NOT stationary: its gradient is twice the initial
        // gradient scale (the optimizer made no progress).
        let grad_norm = 2.0;
        let grad0_norm = 1.0;
        let new_measure = latent_relative_stationarity(grad_norm, grad0_norm);

        // The shift-invariant measure flags it as non-stationary, period.
        assert!(
            new_measure > GRAD_TOL,
            "non-stationary latent (‖∇f‖={grad_norm}, ‖∇f₀‖={grad0_norm}) must never \
             certify converged; rel={new_measure}"
        );

        // And it is bit-identical regardless of the objective value `f`, which
        // the new formula does not reference. The OLD formula, recomputed here,
        // WOULD flip to "converged" once |f| is large enough — the #954 bug.
        let t_scale = 1.0_f64; // ‖t‖_typ floored at 1
        for &f in &[0.0_f64, 1.0, 1.0e3, 1.0e9, 1.0e15] {
            // New: objective never enters, so the decision is unchanged.
            assert_eq!(
                latent_relative_stationarity(grad_norm, grad0_norm),
                new_measure,
                "objective value {f} must not change the stationarity measure"
            );
            // Old (buggy) formula, for contrast.
            let rel_old = grad_norm * t_scale / f.abs().max(1.0);
            if f >= 1.0e9 {
                assert!(
                    rel_old <= GRAD_TOL,
                    "sanity: the retired formula DID falsely converge at |f|={f} \
                     (rel_old={rel_old}), which is exactly the #954 bug the new \
                     measure removes"
                );
            }
        }
    }

    /// Scale-invariance: `f → c·f` scales both gradient norms by `c`, leaving the
    /// ratio fixed, so `grad_tol` is a genuine *relative* tolerance. This holds
    /// in the floor-inactive regime `c·‖∇f₀‖ ≥ 1`; the `max(·, 1)` floor
    /// deliberately breaks exact scale-invariance near unit scale to recover the
    /// absolute test (see [`small_initial_gradient_floors_to_absolute_test`]).
    #[test]
    fn convergence_is_scale_invariant() {
        let grad_norm = 3.0e-9;
        let grad0_norm = 100.0;
        let base = latent_relative_stationarity(grad_norm, grad0_norm);
        // Every c keeps c·grad0_norm ≥ 50 ≫ 1, so the floor stays inactive.
        for &c in &[0.5_f64, 2.0, 1.0e3, 1.0e6] {
            assert!(c * grad0_norm >= 1.0);
            let scaled = latent_relative_stationarity(c * grad_norm, c * grad0_norm);
            assert!(
                (scaled - base).abs() <= 1.0e-18 + 1.0e-12 * base.abs(),
                "scaling the objective by {c} must not change the relative measure: \
                 base={base}, scaled={scaled}"
            );
        }
    }

    /// #879 O(n) calibration: a genuinely stationary latent near interpolation
    /// has an O(n) raw gradient AND an O(n) initial gradient. The bare absolute
    /// test `‖∇f‖ ≤ grad_tol` would reject it, but the relative measure divides
    /// the O(n) magnitude out and correctly reports converged.
    #[test]
    fn on_calibration_relative_measure_certifies_near_interpolation() {
        let n = 5_000.0;
        let grad0_norm = 4.0 * n; // O(n) gradient at the seed
        // A genuinely stationary latent near interpolation: its raw gradient is
        // still O(n) (the profiled `n·log σ̂²` term), so the bare absolute test
        // would reject it.
        let grad_norm = 1.0e-9 * n; // = 5e-6, far above grad_tol = 1e-8
        assert!(
            grad_norm > GRAD_TOL,
            "the raw O(n) gradient {grad_norm} would fail the absolute test"
        );
        // The relative measure divides the O(n) magnitude out and certifies it:
        // (1e-9·n) / (4·n) = 2.5e-10 ≤ grad_tol.
        let rel = latent_relative_stationarity(grad_norm, grad0_norm);
        assert!(
            rel <= GRAD_TOL,
            "a stationary latent (‖∇f‖/‖∇f₀‖ = {rel}) must report converged despite \
             its O(n) raw gradient {grad_norm}"
        );
    }

    /// The `max(·, 1)` floor reduces the test to the absolute `‖∇f‖ ≤ grad_tol`
    /// when the initial gradient is itself small (a near-stationary or
    /// degenerate seed), matching `relative_stationarity` in optimizer.rs.
    #[test]
    fn small_initial_gradient_floors_to_absolute_test() {
        let grad0_norm = 0.0; // degenerate seed → floor to 1
        assert_eq!(latent_relative_stationarity(1.0e-9, grad0_norm), 1.0e-9);
        assert_eq!(latent_relative_stationarity(5.0, grad0_norm), 5.0);
        // A tiny but nonzero seed gradient (< 1) is also floored to 1.
        assert_eq!(latent_relative_stationarity(2.0e-9, 0.3), 2.0e-9);
    }

    /// A blown-up iterate (non-finite gradient at either point) is never
    /// stationary.
    #[test]
    fn non_finite_gradient_is_never_stationary() {
        assert_eq!(
            latent_relative_stationarity(f64::INFINITY, 1.0),
            f64::INFINITY
        );
        assert_eq!(latent_relative_stationarity(f64::NAN, 1.0), f64::INFINITY);
        assert_eq!(
            latent_relative_stationarity(1.0, f64::INFINITY),
            f64::INFINITY
        );
        assert_eq!(latent_relative_stationarity(1.0, f64::NAN), f64::INFINITY);
    }
}

#[cfg(test)]
mod batch_tests {
    use super::*;
    use ndarray::{array, s};

    fn assert_close(lhs: ArrayView2<'_, f64>, rhs: ArrayView2<'_, f64>, tol: f64) {
        assert_eq!(lhs.dim(), rhs.dim());
        for ((i, j), value) in lhs.indexed_iter() {
            let diff = (*value - rhs[[i, j]]).abs();
            assert!(
                diff <= tol,
                "matrix mismatch at ({i}, {j}): lhs={}, rhs={}, diff={diff}",
                value,
                rhs[[i, j]]
            );
        }
    }

    #[test]
    fn ordered_prediction_columns_places_noise_scale_after_mean_columns() {
        // Issue #365 (secondary defect): the fitted location-scale `noise_scale`
        // must be a first-class prediction column reachable from Python and
        // ordered with the other model outputs, not appended arbitrarily after
        // user/id columns. A location-scale prediction emits the standard mean
        // columns plus `noise_scale`; the ordered schema must interleave it
        // right after `mean_upper` (the last preferred mean column) and keep any
        // non-preferred extras behind the preferred block.
        let columns_json = r#"{
            "mean_upper": [2.0],
            "noise_scale": [0.7],
            "linear_predictor": [1.0],
            "row_id": [42.0],
            "mean": [1.1],
            "std_error": [0.2],
            "mean_lower": [0.1]
        }"#;
        let ordered_json = ordered_prediction_columns(columns_json).expect("ordering must succeed");
        // `ordered_prediction_columns` serialises keys in emission order via the
        // manual `ordered_json_object_string` writer, so the textual byte order
        // of the `"key":` tokens is the authoritative column order (round-trip
        // through serde_json::Value would re-sort and lose it).
        let expected = [
            "linear_predictor",
            "mean",
            "std_error",
            "mean_lower",
            "mean_upper",
            "noise_scale",
            "row_id",
        ];
        let positions: Vec<usize> = expected
            .iter()
            .map(|key| {
                ordered_json
                    .find(&format!("\"{key}\":"))
                    .unwrap_or_else(|| {
                        panic!("emitted JSON must contain key {key}: {ordered_json}")
                    })
            })
            .collect();
        for w in positions.windows(2) {
            assert!(
                w[0] < w[1],
                "noise_scale must be ordered immediately after the mean columns and \
                 before non-preferred extras; got JSON {ordered_json}"
            );
        }
    }

    #[test]
    fn gaussian_reml_batched_matches_single_fit_on_ragged_offsets() {
        let x = array![
            [1.0, 0.0],
            [1.0, 1.0],
            [1.0, 2.0],
            [1.0, -1.0],
            [1.0, 0.5],
            [1.0, 1.5],
        ];
        let y = array![[0.0], [1.0], [1.8], [-1.0], [0.2], [1.1]];
        let weights = array![1.0, 0.7, 1.3, 1.0, 1.1, 0.9];
        let penalty = array![[0.0, 0.0], [0.0, 1.0]];
        let offsets = array![0_usize, 3_usize, 6_usize];

        let batched = gaussian_reml_fit_batched_impl(
            x.view(),
            y.view(),
            offsets.view(),
            penalty.view(),
            Some(weights.view()),
            Some(0.5),
        )
        .unwrap();

        for b in 0..2 {
            let start = offsets[b];
            let end = offsets[b + 1];
            let single = gaussian_reml_multi_closed_form_with_cache(
                x.slice(s![start..end, ..]),
                y.slice(s![start..end, ..]),
                penalty.view(),
                Some(weights.slice(s![start..end])),
                Some(0.5),
                None,
            )
            .unwrap();
            assert_eq!(batched.statuses[b], "ok");
            assert!((batched.lambdas[b] - single.lambda).abs() < 1.0e-10);
            assert!((batched.reml_grad_lambdas[b] - single.reml_grad_lambda).abs() < 1.0e-10);
            assert!((batched.reml_hess_lambdas[b] - single.reml_hess_lambda).abs() < 1.0e-10);
            assert!((batched.reml_grad_rhos[b] - single.reml_grad_rho).abs() < 1.0e-10);
            assert!((batched.reml_hess_rhos[b] - single.reml_hess_rho).abs() < 1.0e-10);
            assert_close(
                batched.coefficients.slice(s![b, .., ..]),
                single.coefficients.view(),
                1.0e-10,
            );
            assert_close(
                batched.fitted.slice(s![start..end, ..]),
                single.fitted.view(),
                1.0e-10,
            );
        }
    }

    #[test]
    fn gaussian_reml_batched_backward_matches_single_backward_on_ragged_offsets() {
        let x = array![
            [1.0, 0.0],
            [1.0, 1.0],
            [1.0, 2.0],
            [1.0, -1.0],
            [1.0, 0.5],
            [1.0, 1.5],
        ];
        let y = array![[0.0], [1.0], [1.8], [-1.0], [0.2], [1.1]];
        let weights = array![1.0, 0.7, 1.3, 1.0, 1.1, 0.9];
        let penalty = array![[0.0, 0.0], [0.0, 1.0]];
        let offsets = array![0_usize, 3_usize, 6_usize];
        let grad_lambda = array![0.17, -0.03];
        let grad_coefficients =
            Array3::from_shape_vec((2, 2, 1), vec![0.2, -0.1, 0.05, 0.07]).unwrap();
        let grad_fitted = array![[0.01], [0.03], [-0.02], [0.04], [-0.01], [0.02]];
        let grad_reml_score = array![-0.11, 0.09];

        let batched = gaussian_reml_fit_batched_backward_impl(
            x.view(),
            y.view(),
            offsets.view(),
            penalty.view(),
            Some(weights.view()),
            Some(0.5),
            Some(grad_lambda.view()),
            Some(grad_coefficients.view()),
            Some(grad_fitted.view()),
            Some(grad_reml_score.view()),
            None,
            None,
        )
        .unwrap();

        for b in 0..2 {
            let start = offsets[b];
            let end = offsets[b + 1];
            let single = gaussian_reml_multi_closed_form_backward(
                x.slice(s![start..end, ..]),
                y.slice(s![start..end, ..]),
                penalty.view(),
                Some(weights.slice(s![start..end])),
                Some(0.5),
                grad_lambda[b],
                Some(grad_coefficients.slice(s![b, .., ..])),
                Some(grad_fitted.slice(s![start..end, ..])),
                grad_reml_score[b],
                0.0,
            )
            .unwrap();
            assert_eq!(batched.statuses[b], "ok");
            assert_close(
                batched.grad_x.slice(s![start..end, ..]),
                single.grad_x.view(),
                1.0e-10,
            );
            assert_close(
                batched.grad_y.slice(s![start..end, ..]),
                single.grad_y.view(),
                1.0e-10,
            );
            for row in start..end {
                assert!(
                    (batched.grad_weights[row] - single.grad_weights[row - start]).abs() <= 1.0e-10
                );
            }
        }
    }

    #[test]
    fn position_batched_backward_grad_t_matches_direct_t_finite_difference() {
        let t = array![
            0.08, 0.16, 0.27, 0.39, 0.51, 0.64, 0.76, 0.89, 0.10, 0.19, 0.31, 0.43, 0.56, 0.68,
            0.80, 0.92
        ];
        let y = Array2::from_shape_fn((t.len(), 2), |(row, output)| {
            let u = t[row];
            let scale = output as f64 + 1.0;
            0.3 + 0.4 * scale * u + 0.15 * (2.0 * u + 0.2 * scale).sin()
        });
        let offsets = array![0_usize, 8_usize, 16_usize];
        let knots = array![0.0, 0.0, 0.0, 0.0, 0.25, 0.5, 0.75, 1.0, 1.0, 1.0, 1.0];
        let penalty = Array2::from_diag(&array![0.2, 0.35, 0.8, 1.4, 2.2, 3.1, 4.0]);
        let weights = array![
            1.0, 0.9, 1.1, 1.2, 0.85, 1.05, 0.95, 1.07, 1.0, 1.15, 0.88, 1.04, 0.93, 1.08, 0.97,
            1.12
        ];
        let grad_lambda = array![0.0, 0.0];
        let grad_reml_score = array![0.0, 0.0];
        let grad_coefficients = Array3::zeros((2, penalty.nrows(), y.ncols()));
        let grad_fitted = Array2::from_shape_fn(y.dim(), |(row, output)| {
            0.02 * ((row as f64 + 1.0) * (output as f64 + 1.5)).sin()
        });

        let analytic = gaussian_reml_fit_positions_batched_backward_impl(
            t.view(),
            y.view(),
            offsets.view(),
            knots.view(),
            "bspline",
            3,
            false,
            None,
            penalty.view(),
            Some(weights.view()),
            Some(0.7),
            Some(grad_lambda.view()),
            Some(grad_coefficients.view()),
            Some(grad_fitted.view()),
            Some(grad_reml_score.view()),
            None,
            None,
            0,
            None,
        )
        .expect("position batched backward");

        let loss = |candidate_t: ArrayView1<'_, f64>| -> f64 {
            let fit = gaussian_reml_fit_positions_batched_impl(
                candidate_t,
                y.view(),
                offsets.view(),
                knots.view(),
                "bspline",
                3,
                false,
                None,
                penalty.view(),
                Some(weights.view()),
                Some(0.7),
                None,
                0,
            )
            .expect("position batched forward");
            let mut value = 0.0;
            for row in 0..y.nrows() {
                for j in 0..y.ncols() {
                    value += grad_fitted[[row, j]] * fit.fitted[[row, j]];
                }
            }
            value
        };

        let eps = 1.0e-6;
        for row in 0..t.len() {
            let mut plus = t.clone();
            let mut minus = t.clone();
            plus[row] += eps;
            minus[row] -= eps;
            let fd = (loss(plus.view()) - loss(minus.view())) / (2.0 * eps);
            let diff = (analytic.grad_t[row] - fd).abs();
            let tol = 5.0e-5_f64.max(5.0e-5 * analytic.grad_t[row].abs().max(fd.abs()));
            assert!(
                diff <= tol,
                "grad_t[{row}] mismatch: analytic={:.12e}, finite_difference={:.12e}, diff={diff:.3e}, tol={tol:.3e}",
                analytic.grad_t[row],
                fd
            );
        }
    }

    /// Issue #876: the circle latent optimizer must recover a *circular* latent
    /// from clean on-circle data, not collapse every row to one coordinate.
    ///
    /// The data lie (up to tiny noise) exactly on a unit circle in the leading
    /// two response columns. Started from the all-zero default latent — the
    /// collapsed configuration — the periodic spectral seed spreads the rows
    /// around the circle and the Riemannian trust region polishes them. We assert
    /// (a) the recovered angles are *not collapsed* (their dispersion is large)
    /// and (b) they track the true generating angle, up to the circle's
    /// rotation/reflection gauge, via the circular correlation of the unit
    /// vectors (cos/sin) — which is gauge-equivariant under a constant rotation.
    #[test]
    fn circle_latent_recovers_circle_not_collapse() {
        use std::f64::consts::TAU;

        let n_obs = 40usize;
        let latent_dim = 1usize;
        // Deterministic angles spread around the circle (sorted, distinct).
        let true_theta: Vec<f64> = (0..n_obs)
            .map(|i| -std::f64::consts::PI + (i as f64 + 0.5) / n_obs as f64 * TAU)
            .collect();
        // 5-D response: first two columns trace the unit circle, the rest are
        // tiny deterministic perturbations (a near-noise pad).
        let y = Array2::from_shape_fn((n_obs, 5), |(row, col)| {
            let th = true_theta[row];
            match col {
                0 => th.cos(),
                1 => th.sin(),
                _ => 0.01 * ((row as f64 + 1.3) * (col as f64 + 0.7)).sin(),
            }
        });
        // Duchon decoder centers (the issue's geometry: a 1-D linspace).
        let n_centers = 12usize;
        let centers = Array2::from_shape_fn((n_centers, 1), |(i, _)| {
            -std::f64::consts::PI + i as f64 / (n_centers - 1) as f64 * TAU
        });
        let penalty = Array2::<f64>::eye(n_centers);

        // Reproduce the optimize-latent driver on the circle manifold from the
        // collapsed all-zero default start with the periodic spectral seed.
        let caller_t = Array1::<f64>::zeros(n_obs * latent_dim);
        let start = latent_spectral_seed_start(
            y.view(),
            centers.view(),
            "circle",
            n_obs,
            latent_dim,
            10,
            caller_t.view(),
        )
        .expect("periodic spectral seed");
        // The seed itself must already be spread (the whole point of #876).
        let seed_std = {
            let mean = start.iter().sum::<f64>() / start.len() as f64;
            (start.iter().map(|v| (v - mean).powi(2)).sum::<f64>() / start.len() as f64).sqrt()
        };
        assert!(
            seed_std > 0.3,
            "periodic seed must spread the latent, not start collapsed; std={seed_std}"
        );

        let problem = LatentOuterProblem {
            y: y.clone(),
            centers: centers.clone(),
            penalty: penalty.clone(),
            weights: None,
            aux_u: None,
            dim_selection: None,
            family: AuxPriorFamily::Ridge,
            aux_strength: None,
            init_lambda: None,
            sigma_eff_mode: SigmaEffMode::Profiled,
            n_obs,
            latent_dim,
            m: 2,
            basis_kind: "duchon".to_string(),
            tensor_knots: None,
            tensor_knot_offsets: None,
            tensor_degrees: None,
            periodic: latent_manifold_periodic_descriptor("circle", latent_dim),
        };
        let manifold_box = build_latent_outer_manifold("circle", n_obs, latent_dim)
            .expect("circle latent manifold");
        let manifold_ref: &dyn gam::geometry::RiemannianManifold = manifold_box.as_ref();
        let trust_region = gam::geometry::RiemannianTrustRegion {
            radius: 1.0,
            max_radius: 1.0e6,
            max_iter: 200,
            grad_tol: 1.0e-8,
        };
        let mut objective = LatentOuterObjective { problem: &problem };
        let recovered = trust_region
            .minimize(manifold_ref, &mut objective, start.view())
            .expect("circle latent optimize");

        // (a) Not collapsed: the recovered angles keep a large spread.
        let mean = recovered.iter().sum::<f64>() / recovered.len() as f64;
        let rec_std = (recovered.iter().map(|v| (v - mean).powi(2)).sum::<f64>()
            / recovered.len() as f64)
            .sqrt();
        assert!(
            rec_std > 0.3,
            "circle latent collapsed (std={rec_std}); expected a spread circular latent"
        );

        // (b) Circular correlation with the true angle, gauge-equivariant: the
        // mean resultant length of the angular differences θ̂ − θ. For a perfect
        // recovery up to a constant rotation this is ≈ 1; an unrelated latent
        // gives ≈ 0.
        let (mut cs, mut ss) = (0.0f64, 0.0f64);
        for n in 0..n_obs {
            let diff = recovered[n] - true_theta[n];
            cs += diff.cos();
            ss += diff.sin();
        }
        let resultant = (cs * cs + ss * ss).sqrt() / n_obs as f64;
        // Reflection gauge: the circle is also free to flip orientation, so test
        // both θ̂ and −θ̂ and keep the better resultant.
        let (mut cs_r, mut ss_r) = (0.0f64, 0.0f64);
        for n in 0..n_obs {
            let diff = -recovered[n] - true_theta[n];
            cs_r += diff.cos();
            ss_r += diff.sin();
        }
        let resultant_r = (cs_r * cs_r + ss_r * ss_r).sqrt() / n_obs as f64;
        let best = resultant.max(resultant_r);
        assert!(
            best > 0.85,
            "recovered circle latent does not track the true angle \
             (mean resultant length={best}); collapse/degenerate recovery"
        );
    }
}

fn validate_vector(name: &str, values: ArrayView1<'_, f64>) -> Result<(), String> {
    if values.is_empty() {
        return Err(format!("{name} cannot be empty"));
    }
    if values.iter().any(|value| !value.is_finite()) {
        return Err(format!("{name} must contain only finite values"));
    }
    Ok::<(), _>(())
}

fn periodic_knot_domain(knots: ArrayView1<'_, f64>) -> Result<(f64, f64, usize), String> {
    if knots.len() < 2 {
        return Err("periodic knots must contain at least start and end".to_string());
    }
    let left = knots[0];
    let right = knots[knots.len() - 1];
    if left >= right {
        return Err(format!(
            "periodic knot domain must be increasing; got [{left}, {right}]"
        ));
    }
    Ok((left, right, knots.len() - 1))
}

fn duchon_nullspace_from_m(m: usize) -> DuchonNullspaceOrder {
    match m {
        1 => DuchonNullspaceOrder::Zero,
        2 => DuchonNullspaceOrder::Linear,
        other => DuchonNullspaceOrder::Degree(other - 1),
    }
}

fn pyffi_duchon_previous_nullspace_order(order: DuchonNullspaceOrder) -> DuchonNullspaceOrder {
    match order {
        DuchonNullspaceOrder::Zero => DuchonNullspaceOrder::Zero,
        DuchonNullspaceOrder::Linear => DuchonNullspaceOrder::Zero,
        DuchonNullspaceOrder::Degree(2) => DuchonNullspaceOrder::Linear,
        DuchonNullspaceOrder::Degree(k) => DuchonNullspaceOrder::Degree(k - 1),
    }
}

fn pyffi_duchon_polynomial_block(
    points: ArrayView2<'_, f64>,
    order: DuchonNullspaceOrder,
) -> Array2<f64> {
    let n_rows = points.nrows();
    let dim = points.ncols();
    match order {
        DuchonNullspaceOrder::Zero => Array2::<f64>::ones((n_rows, 1)),
        DuchonNullspaceOrder::Linear => {
            let mut poly = Array2::<f64>::zeros((n_rows, dim + 1));
            poly.column_mut(0).fill(1.0);
            for axis in 0..dim {
                poly.column_mut(axis + 1).assign(&points.column(axis));
            }
            poly
        }
        DuchonNullspaceOrder::Degree(degree) => {
            let exponents = monomial_exponents(dim, degree);
            let mut poly = Array2::<f64>::zeros((n_rows, exponents.len()));
            for (col, alpha) in exponents.iter().enumerate() {
                for row in 0..n_rows {
                    let mut value = 1.0;
                    for axis in 0..dim {
                        let exp = alpha[axis];
                        if exp != 0 {
                            value *= points[[row, axis]].powi(exp as i32);
                        }
                    }
                    poly[[row, col]] = value;
                }
            }
            poly
        }
    }
}

fn pyffi_duchon_effective_nullspace_order(
    centers: ArrayView2<'_, f64>,
    order: DuchonNullspaceOrder,
) -> DuchonNullspaceOrder {
    let mut effective = order;
    while effective != DuchonNullspaceOrder::Zero
        && centers.nrows() <= pyffi_duchon_polynomial_block(centers, effective).ncols()
    {
        effective = pyffi_duchon_previous_nullspace_order(effective);
    }
    effective
}

fn pyffi_duchon_kernel_constraint_nullspace(
    centers: ArrayView2<'_, f64>,
    order: DuchonNullspaceOrder,
) -> Result<Array2<f64>, String> {
    let polynomial_block = pyffi_duchon_polynomial_block(centers, order);
    gam::linalg::faer_ndarray::rrqr_nullspace_basis(
        &polynomial_block,
        gam::linalg::faer_ndarray::default_rrqr_rank_alpha(),
    )
    .map(|(null_basis, _)| null_basis)
    .map_err(|err| format!("failed to build Duchon kernel constraint nullspace: {err}"))
}

/// Parse the ``nullspace_order`` keyword on the primitive Duchon bindings.
/// Accepted strings:
/// ``"zero"`` (constant nullspace), ``"linear"`` (constant + linear),
/// or ``"degree<k>"`` for k ≥ 2 (polynomials of total degree ≤ k).
fn parse_nullspace_order(raw: Option<&str>) -> PyResult<DuchonNullspaceOrder> {
    let Some(raw) = raw else {
        return Err(py_value_error(
            "nullspace_order is required; pass 'zero', 'linear', or 'degree<k>'".to_string(),
        ));
    };
    let trimmed = raw.trim();
    let lower = trimmed.to_ascii_lowercase();
    match lower.as_str() {
        "zero" => Ok(DuchonNullspaceOrder::Zero),
        "linear" => Ok(DuchonNullspaceOrder::Linear),
        other if other.starts_with("degree") => {
            let suffix = other.trim_start_matches("degree").trim_start_matches('=');
            let k: usize = suffix.parse().map_err(|_| {
                py_value_error(format!(
                    "invalid nullspace_order '{raw}'; expected 'zero', 'linear', or 'degree<k>' with integer k ≥ 2"
                ))
            })?;
            if k < 2 {
                return Err(py_value_error(format!(
                    "nullspace_order 'degree{k}' must have k ≥ 2; use 'zero' or 'linear' instead"
                )));
            }
            Ok(DuchonNullspaceOrder::Degree(k))
        }
        _ => Err(py_value_error(format!(
            "invalid nullspace_order '{raw}'; expected 'zero', 'linear', or 'degree<k>'"
        ))),
    }
}

/// Resolved Duchon hybrid-mode configuration as derived from the public
/// keyword surface (``length_scale``, ``nullspace_order``, ``power``).
/// All three primitives share this resolver so the auto-resolution policy
/// matches the formula path 1:1.
struct DuchonHybridConfig {
    length_scale: Option<f64>,
    nullspace_order: DuchonNullspaceOrder,
    power: f64,
}

/// Resolve the (nullspace_order, power) pair given the optional public
/// keywords. ``max_op`` is the highest radial-derivative order required by
/// the downstream consumer of this spec:
///
/// * ``max_op = 0`` — consumers that never assemble operator penalties: both
///   the Python ``duchon_basis`` PyFFI (returns only the design matrix) AND
///   the ``duchon_function_norm_penalty`` PyFFI (returns only the native
///   reproducing-norm Gram ``PenaltySource::Primary`` plus a null-space
///   shrinkage ridge). Only kernel-existence (``2(p+s) > d``) and the
///   pure-Duchon CPD guard (``2s < d``) apply; D1 / D2 collocation is not
///   required. These two primitives MUST resolve identically so the penalty
///   matches the basis it penalizes (gam#880).
/// * ``max_op = 2`` — collocation up to the curvature operator (mass +
///   tension + stiffness). No PyFFI primitive routes through this any longer;
///   it is retained for any future consumer that re-introduces the closed-form
///   collocation operator Grams. The formula path does NOT route through this
///   resolver either: the non-periodic Euclidean smooth term resolves
///   ``(nullspace_order, power)`` via the cubic structural rule (``Linear``
///   null space, spectral power ``s = (d−1)/2``) and ships the native
///   reproducing-norm Gram plus a null-space shrinkage ridge, not the operator
///   triplet.
///
/// When ``any_periodic`` is set, the spec is destined for the mixed-periodicity
/// (cylinder/torus) builder, whose additive per-axis reproducing kernel is
/// derived only for the pure-polyharmonic spectrum (Sobolev tail ``s = 0``):
/// the periodic Bernoulli Green's function and the non-periodic Sobolev kernel
/// have no validated fractional-power generalization. The auto-power branch
/// therefore pins ``power = 0`` on periodic requests, keeping the SAME cubic
/// ``Linear`` null space (so the ``{1, y}`` polynomial block still matches
/// ``duchon_p_from_nullspace_order``) rather than emitting the Euclidean
/// ``s = (d−1)/2`` default the periodic builder cannot represent.
fn resolve_duchon_hybrid_config(
    dim: usize,
    length_scale: Option<f64>,
    nullspace_order: Option<&str>,
    explicit_power: Option<f64>,
    max_op: usize,
    any_periodic: bool,
) -> PyResult<DuchonHybridConfig> {
    let requested_nullspace = parse_nullspace_order(nullspace_order)?;
    // Pure, auto-power requests (no `length_scale`, no explicit `power`) resolve
    // via the SAME cubic structural default the formula/CLI front-ends use
    // (`duchon_cubic_default`): an affine (`Linear`) null space plus the
    // fractional spectral power `s = (d-1)/2`, i.e. the r³ Duchon kernel in every
    // dimension. This is what keeps the kernel admissible — `2(p+s) = d+1 > d`
    // with only the `d+1` affine columns — and, crucially, robust to the
    // center-count-driven null-space degradation that the integer-power operator
    // resolver below cannot survive: that path escalates the null space to absorb
    // the pure-mode CPD order at an *integer* `s` (e.g. d=2 ⇒ `Degree(2)`, six
    // polynomial columns, `s=0`), but on sparse centers the escalated polynomial
    // block degrades back down while `s` stays put, leaving `2(p+s) ≤ d` and an
    // inadmissible kernel (gam#880). The integer-power resolver is reserved for
    // an explicit `power` or the hybrid Matérn-blended kernel (`length_scale`),
    // whose partial-fraction spectrum is only defined for integer `s`.
    if length_scale.is_none() && explicit_power.is_none() {
        let (nullspace_order, cubic_power) = duchon_cubic_default(dim);
        // The mixed-periodicity reproducing kernel supports only s = 0 (pure
        // polyharmonic); pin the auto power to 0 there while keeping the cubic
        // `Linear` null space so the periodic builder accepts the auto-resolved
        // spec instead of rejecting the Euclidean s = (d−1)/2 default.
        let power = if any_periodic { 0.0 } else { cubic_power };
        return Ok(DuchonHybridConfig {
            length_scale,
            nullspace_order,
            power,
        });
    }
    let (resolved_nullspace, auto_power) =
        resolve_duchon_orders(dim, requested_nullspace, max_op, length_scale);
    let power = explicit_power.unwrap_or(auto_power as f64);
    Ok(DuchonHybridConfig {
        length_scale,
        nullspace_order: resolved_nullspace,
        power,
    })
}

fn column_array(values: ArrayView1<'_, f64>) -> Array2<f64> {
    values.to_owned().insert_axis(Axis(1))
}

fn fit_with_null_space_logdet(
    design: &TermCollectionDesign,
    fit: &gam::solver::estimate::UnifiedFitResult,
) -> Result<gam::solver::estimate::UnifiedFitResult, String> {
    let mut fit = fit.clone();
    let (null_dim, logdet) = compute_null_space_metadata(design, &fit)?;
    fit.artifacts.null_space_dim = Some(null_dim);
    fit.artifacts.null_space_logdet = Some(logdet);
    Ok(fit)
}

fn compute_null_space_metadata(
    design: &TermCollectionDesign,
    fit: &gam::solver::estimate::UnifiedFitResult,
) -> Result<(usize, f64), String> {
    let hessian = fit
        .penalized_hessian()
        .ok_or_else(|| "null-space Hessian logdet requires fitted penalized Hessian".to_string())?;
    let hessian_dim = hessian.nrows();
    if hessian.ncols() != hessian_dim {
        return Err(format!(
            "null-space Hessian logdet requires a square Hessian, got {}x{}",
            hessian.nrows(),
            hessian.ncols()
        ));
    }
    let m = design.design.ncols();
    // The penalty null-space normalizer is defined over the MEAN design's
    // penalty topology (`design.penalties`, whose `col_range`s index the mean
    // coefficients). A flexible-link / link-wiggle fit (#1596) appends an extra
    // penalized warp block to the JOINT coefficient vector, so the fitted
    // penalized Hessian is larger than the mean design (e.g. mean=2, joint=11).
    // The warp block is a separate penalized component and the mean is the
    // leading block of the joint coefficient layout, so restrict to the leading
    // `m×m` sub-Hessian — the mean block's penalized curvature — to evaluate the
    // mean topology normalizer. For an ordinary (no-wiggle) fit `hessian_dim ==
    // m` and this is the full Hessian, unchanged.
    if hessian_dim < m {
        return Err(format!(
            "null-space Hessian logdet design/Hessian mismatch: design has {m} columns but \
             Hessian is only {hessian_dim}x{hessian_dim}"
        ));
    }
    let hessian = if hessian_dim > m {
        hessian.slice(s![0..m, 0..m]).to_owned()
    } else {
        hessian.clone()
    };
    let p = m;

    // #757: A smooth-free model (`y ~ x1 + x2`, any family) carries no penalty
    // blocks, so the assembled penalty is identically zero and its "null space"
    // is the entire coefficient space. This metadata is the Tierney-Kadane /
    // topology normalizer `log|Nᵀ H N|` over the *penalty* null space — a
    // quantity that only discriminates among penalized-smooth topologies and is
    // vacuous for a fully-parametric GLM (there is no penalized prior to
    // Laplace-integrate; a REML restricted-likelihood already carries the
    // fixed-effect `log|XᵀWX|` term). With an all-zero penalty the code below
    // would Cholesky-factor the full Hessian in a basis that does not round-trip
    // for the rank-zero penalty, which rejected every smooth-free fit from the
    // Python payload path even though the fit converged and the CLI (which never
    // computes this) accepts it. Treat "no penalty" as "no null-space
    // normalizer", consistent with the full-penalty-rank (`q == 0`) branch below.
    if design.penalties.is_empty() {
        return Ok((0, 0.0));
    }

    let mut penalty = Array2::<f64>::zeros((p, p));
    for (idx, block) in design.penalties.iter().enumerate() {
        let range = block.col_range.clone();
        if range.start > range.end
            || range.end > p
            || block.local.nrows() != range.len()
            || block.local.ncols() != range.len()
        {
            return Err(format!(
                "null-space Hessian logdet penalty {idx} shape mismatch: range {}..{}, local {}x{}, p={p}",
                range.start,
                range.end,
                block.local.nrows(),
                block.local.ncols()
            ));
        }
        penalty
            .slice_mut(s![range.clone(), range])
            .scaled_add(1.0, &block.local);
    }

    let (null_basis, _) = gam::linalg::faer_ndarray::rrqr_nullspace_basis(
        &penalty,
        gam::linalg::faer_ndarray::default_rrqr_rank_alpha(),
    )
    .map_err(|err| format!("failed to compute penalty null-space basis: {err}"))?;
    let q = null_basis.ncols();
    if q == 0 {
        return Ok((0, 0.0));
    }

    let projected = hessian.dot(&null_basis);
    let mut restricted = null_basis.t().dot(&projected);
    restricted = (&restricted + &restricted.t()) * 0.5;
    let chol = restricted
        .cholesky(Side::Lower)
        .map_err(|err| format!("null-space Hessian is not positive definite: {err}"))?;
    let logdet = 2.0 * chol.diag().iter().map(|value| value.ln()).sum::<f64>();
    if logdet.is_finite() {
        Ok((q, logdet))
    } else {
        Err(format!("null-space Hessian logdet is not finite: {logdet}"))
    }
}

/// Precompute the exact Gaussian-identity jackknife+ statistics (#942) at fit
/// time, *iff* the model is eligible: a standard Gaussian-identity GLM with unit
/// prior weights (no `weight_column`), an offset-free primary predictor (an
/// offset shifts η but the jackknife+ residuals are formed on the response, so
/// it would have to be threaded into the design; out of scope here), and a
/// fitted penalized Hessian `M = XᵀX + Sλ` available. Returns `None` (never an
/// error) for any ineligible model — predict then falls back to the model-based
/// band with honest provenance.
///
/// The penalized Hessian stored in `FitGeometry` *is* `M` for this family
/// (unit working weights, dispersion-unscaled), so the substrate replays the
/// exact normal matrix the fit used — no penalty re-derivation.
fn gaussian_jackknife_plus_stats_for_standard_fit(
    formula: &str,
    dataset: &EncodedDataset,
    fit_config: &FitConfig,
    family: &LikelihoodSpec,
    saved_fit: &gam::solver::estimate::UnifiedFitResult,
    design: &TermCollectionDesign,
) -> Option<gam::inference::full_conformal::GaussianJackknifePlusStats> {
    if !matches!(family.response, ResponseFamily::Gaussian) {
        return None;
    }
    if !matches!(family.link, InverseLink::Standard(StandardLink::Identity)) {
        return None;
    }
    if fit_config.weight_column.is_some() {
        return None;
    }
    if fit_config.offset_column.is_some() {
        return None;
    }
    if design.affine_offset.iter().any(|value| *value != 0.0) {
        return None;
    }
    // A wiggle-augmented link breaks the Gaussian-identity closed form.
    if fit_config.flexible_link {
        return None;
    }
    let response_name = response_column_name(formula)?;
    let col_map = dataset.column_map();
    let response_col = *col_map.get(&response_name)?;
    let y = dataset.values.column(response_col).to_owned();
    let x = design.design.try_to_dense_arc("jackknife+ design").ok()?;
    if x.nrows() != y.len() {
        return None;
    }
    let m = saved_fit.penalized_hessian()?;
    if m.nrows() != x.ncols() || m.ncols() != x.ncols() {
        return None;
    }
    let weights = Array1::<f64>::ones(y.len());
    // `from_design_unit_weight_normal_matrix` re-validates unit weights and the
    // shapes; any internal degeneracy (e.g. a leverage-one row) returns Err,
    // which we swallow to `None` — predict stays valid, just without the magic.
    gam::inference::full_conformal::GaussianJackknifePlusStats::from_design_unit_weight_normal_matrix(
        x.as_ref(),
        &y,
        &weights,
        m,
    )
    .ok()
}

fn build_standard_payload(
    formula: String,
    dataset: &EncodedDataset,
    fit_config: &FitConfig,
    family: LikelihoodSpec,
    saved_fit: &gam::solver::estimate::UnifiedFitResult,
    design: &TermCollectionDesign,
    resolved_termspec: TermCollectionSpec,
    adaptive_regularization_diagnostics: Option<
        gam::terms::smooth::AdaptiveRegularizationDiagnostics,
    >,
    wiggle_knots: Option<Vec<f64>>,
    wiggle_degree: Option<usize>,
    wiggle_penalty_metadata: Option<gam::families::wiggle::WigglePenaltyMetadata>,
    wiggle_saved_warp_beta: Option<Vec<f64>>,
    wiggle_saved_index_shift: Option<Vec<f64>>,
) -> Result<FittedModelPayload, String> {
    let saved_fit = fit_with_null_space_logdet(design, saved_fit)?;
    let latent_cloglog_state =
        if matches!(
            (&family.response, &family.link),
            (ResponseFamily::Binomial, InverseLink::LatentCLogLog(_))
        ) {
            Some(saved_latent_cloglog_state_from_fit(&saved_fit).expect(
                "latent-cloglog-binomial fit must produce an explicit latent-cloglog state",
            ))
        } else {
            saved_latent_cloglog_state_from_fit(&saved_fit)
        };
    let family_link = family.link_function();
    let family_inverse_link = family.link.clone();
    let estimator = gam::families::fit_orchestration::expectile_tau_for_config(fit_config)
        .map_err(|error| format!("failed to persist estimator metadata: {error}"))?
        .map_or(FittedEstimator::Likelihood, |tau| {
            FittedEstimator::Expectile { tau }
        });
    let family_name = match estimator {
        FittedEstimator::Likelihood => family.name().to_string(),
        FittedEstimator::Expectile { tau } => format!("expectile({tau})"),
    };
    // #942 MAGIC: precompute the exact Gaussian-identity jackknife+ substrate
    // (distribution-free, finite-sample ≥level coverage with no held-out fold)
    // while the training design + response are still in hand. `None` for any
    // ineligible model; predict falls back to the model-based band.
    let jackknife_plus_stats = match estimator {
        FittedEstimator::Likelihood => gaussian_jackknife_plus_stats_for_standard_fit(
            &formula, dataset, fit_config, &family, &saved_fit, design,
        ),
        FittedEstimator::Expectile { .. } => None,
    };
    // #1098 MAGIC: precompute the EXACT Gaussian-identity full-conformal
    // substrate (distribution-free finite-sample set, no held-out fold) under
    // the same eligibility as jackknife+. Computed here while `formula`/`family`/
    // `saved_fit` are still owned — the payload constructor moves them below.
    let full_conformal_substrate = match estimator {
        FittedEstimator::Likelihood => exact_full_conformal_substrate_for_standard_fit(
            &formula, dataset, fit_config, &family, &saved_fit, design,
        ),
        FittedEstimator::Expectile { .. } => None,
    };
    let mut payload = FittedModelPayload::new(
        MODEL_PAYLOAD_VERSION,
        formula,
        ModelKind::Standard,
        FittedFamily::Standard {
            likelihood: family,
            link: StandardLink::try_from(family_link).ok(),
            latent_cloglog_state,
            mixture_state: saved_mixture_state_from_fit(&saved_fit),
            sas_state: saved_sas_state_from_fit(&saved_fit),
        },
        family_name,
    );
    payload.estimator = estimator;
    payload.unified = Some(saved_fit.clone());
    payload.fit_result = Some(saved_fit);
    payload.data_schema = Some(dataset.schema.clone());
    payload.link = Some(family_inverse_link);
    payload.linkwiggle_knots = wiggle_knots;
    payload.linkwiggle_degree = wiggle_degree;
    payload.linkwiggle_penalty_metadata = wiggle_penalty_metadata;
    // The frozen standard link-wiggle fit residualizes its design in
    // observation space without changing the coefficient chart. Persist the
    // exact LinkWiggle block copy used by replay; saved-model validation
    // requires it to agree bit-for-bit with the joint fit so point prediction
    // and covariance cannot drift into different frames.
    payload.beta_link_wiggle = wiggle_saved_warp_beta;
    // #2141: persist the frozen-index shift so predict evaluates the warp basis
    // at the frozen index `η̂` the fit pinned `B(η̂)` at, not at the de-aliased
    // base predictor — reproducing the fitted `q` and deviance at predict time.
    payload.link_wiggle_index_shift = wiggle_saved_index_shift;
    payload.set_training_feature_metadata(dataset.headers.clone(), dataset.feature_ranges());
    payload.resolved_termspec = Some(resolved_termspec);
    payload.adaptive_regularization_diagnostics = adaptive_regularization_diagnostics;
    payload.offset_column = fit_config.offset_column.clone();
    payload.noise_offset_column = fit_config.noise_offset_column.clone();
    // Persist the analytic prior-weights column so `Model.sample_replicates`
    // can re-resolve the per-row weights and draw heteroskedastic Gaussian
    // observation noise `sigma_i = sigma_hat / sqrt(w_i)` (#2025). Without this
    // the replicate path only saw the pooled scalar sigma_hat.
    payload.weight_column = fit_config.weight_column.clone();
    payload.gaussian_jackknife_plus = jackknife_plus_stats;
    payload.full_conformal = full_conformal_substrate;
    Ok(payload)
}

/// Precompute the exact Gaussian-identity full-conformal substrate at fit time
/// under the SAME eligibility gate as the jackknife+ substrate
/// ([`gaussian_jackknife_plus_stats_for_standard_fit`]): Gaussian-identity,
/// unit-weight, offset-free, link-wiggle-free, with the converged penalized
/// normal matrix `M = XᵀX + Sλ` available. Returns `None` (never an error) for
/// any ineligible model; predict then errors clearly or falls back.
fn exact_full_conformal_substrate_for_standard_fit(
    formula: &str,
    dataset: &EncodedDataset,
    fit_config: &FitConfig,
    family: &LikelihoodSpec,
    saved_fit: &gam::solver::estimate::UnifiedFitResult,
    design: &TermCollectionDesign,
) -> Option<gam::inference::full_conformal::ExactFullConformalSubstrate> {
    if !matches!(family.response, ResponseFamily::Gaussian) {
        return None;
    }
    if !matches!(family.link, InverseLink::Standard(StandardLink::Identity)) {
        return None;
    }
    if fit_config.weight_column.is_some() {
        return None;
    }
    if fit_config.offset_column.is_some() {
        return None;
    }
    if design.affine_offset.iter().any(|value| *value != 0.0) {
        return None;
    }
    if fit_config.flexible_link {
        return None;
    }
    let response_name = response_column_name(formula)?;
    let col_map = dataset.column_map();
    let response_col = *col_map.get(&response_name)?;
    let y = dataset.values.column(response_col).to_owned();
    let x = design
        .design
        .try_to_dense_arc("full-conformal design")
        .ok()?;
    if x.nrows() != y.len() {
        return None;
    }
    let m = saved_fit.penalized_hessian()?;
    if m.nrows() != x.ncols() || m.ncols() != x.ncols() {
        return None;
    }
    let weights = Array1::<f64>::ones(y.len());
    gam::inference::full_conformal::ExactFullConformalSubstrate::from_design_unit_weight_normal_matrix(
        x.as_ref(),
        &y,
        &weights,
        m,
    )
    .ok()
}

fn build_transformation_normal_ffi_payload(
    formula: String,
    dataset: &EncodedDataset,
    fit_config: &FitConfig,
    tn_result: TransformationNormalFitResult,
) -> Result<FittedModelPayload, String> {
    let frozen_covariate = freeze_term_collection_from_design(
        &tn_result.covariate_spec_resolved,
        &tn_result.covariate_design,
    )
    .map_err(|err| format!("failed to freeze transformation-normal covariate spec: {err}"))?;

    // Thin adapter over the shared core assembler; the FFI freezes the
    // covariate spec from its design and reads the offset column from the
    // FitConfig. See `assemble_transformation_normal_payload`.
    Ok(assemble_transformation_normal_payload(
        TransformationNormalInputs {
            formula,
            data_schema: dataset.schema.clone(),
            resolved_covariate_spec: frozen_covariate,
            fit_result: tn_result.fit.clone(),
            family: &tn_result.family,
            score_calibration: tn_result.score_calibration.clone(),
        },
        SavedModelSourceMetadata {
            training_headers: dataset.headers.clone(),
            training_feature_ranges: Some(dataset.feature_ranges()),
            offset_column: fit_config.offset_column.clone(),
            noise_offset_column: None,
        },
    ))
}

fn build_bernoulli_marginal_slope_ffi_payload(
    formula: String,
    dataset: &EncodedDataset,
    fit_config: &FitConfig,
    base_link: InverseLink,
    frailty: gam::families::survival::lognormal_kernel::FrailtySpec,
    ms_result: BernoulliMarginalSlopeFitResult,
) -> Result<FittedModelPayload, String> {
    let frozen_marginal = freeze_term_collection_from_design(
        &ms_result.marginalspec_resolved,
        &ms_result.marginal_design,
    )
    .map_err(|err| format!("failed to freeze marginal spec: {err}"))?;
    let frozen_logslope = freeze_term_collection_from_design(
        &ms_result.logslopespec_resolved,
        &ms_result.logslope_design,
    )
    .map_err(|err| format!("failed to freeze logslope spec: {err}"))?;

    let logslope_formula = fit_config
        .logslope_formula
        .clone()
        .ok_or_else(|| "bernoulli marginal-slope requires logslope_formula".to_string())?;
    let z_column = fit_config
        .z_column
        .clone()
        .ok_or_else(|| "bernoulli marginal-slope requires z_column".to_string())?;

    // Thin adapter over the shared core assembler. The FFI's source-specific
    // work is freezing term collections from their designs, reading the
    // logslope formula / z column / offset columns from the FitConfig, and
    // persisting headers without per-feature ranges; the semantic payload is
    // assembled by the same core path the CLI uses, so the two save routes
    // produce identical contracts.
    assemble_bernoulli_marginal_slope_payload(
        BernoulliMarginalSlopeInputs {
            formula,
            data_schema: dataset.schema.clone(),
            logslope_formula,
            z_column,
            resolved_marginalspec: frozen_marginal,
            resolved_logslopespec: frozen_logslope,
            fit_result: ms_result.fit.clone(),
            p_marginal: ms_result.marginal_design.design.ncols(),
            baseline_marginal: ms_result.baseline_marginal,
            baseline_logslope: ms_result.baseline_logslope,
            latent_z_normalization: SavedLatentZNormalization {
                mean: ms_result.z_normalization.mean,
                sd: ms_result.z_normalization.sd,
            },
            latent_measure: ms_result.latent_measure.clone(),
            latent_z_rank_int_calibration: ms_result.latent_z_rank_int_calibration.clone(),
            latent_z_conditional_calibration: ms_result.latent_z_conditional_calibration.clone(),
            score_warp_runtime: ms_result.score_warp_runtime.as_ref(),
            link_dev_runtime: ms_result.link_dev_runtime.as_ref(),
            base_link,
            frailty,
        },
        SavedModelSourceMetadata {
            training_headers: dataset.headers.clone(),
            training_feature_ranges: None,
            offset_column: fit_config.offset_column.clone(),
            noise_offset_column: fit_config.noise_offset_column.clone(),
        },
    )
}

fn build_survival_marginal_slope_ffi_payload(
    formula: String,
    dataset: &EncodedDataset,
    fit_config: &FitConfig,
    frailty: gam::families::survival::lognormal_kernel::FrailtySpec,
    ms_result: SurvivalMarginalSlopeFitResult,
) -> Result<FittedModelPayload, String> {
    use gam::families::survival::construction::{
        build_survival_time_basis, parse_survival_baseline_config, parse_survival_likelihood_mode,
        parse_survival_time_basis_config, resolve_survival_marginal_slope_time_anchor_value,
        survival_marginal_slope_offset_baseline_config,
    };
    use ndarray::s;

    let frozen_marginal = freeze_term_collection_from_design(
        &ms_result.marginalspec_resolved,
        &ms_result.marginal_design,
    )
    .map_err(|err| format!("failed to freeze survival marginal spec: {err}"))?;
    let frozen_logslope = freeze_term_collection_from_design(
        &ms_result.logslopespec_resolved,
        &ms_result.logslope_design,
    )
    .map_err(|err| format!("failed to freeze survival logslope spec: {err}"))?;

    let logslope_formula = fit_config
        .logslope_formula
        .clone()
        .unwrap_or_else(|| "same-as-main".to_string());
    let z_column = fit_config
        .z_column
        .clone()
        .ok_or_else(|| "survival marginal-slope requires z_column".to_string())?;
    let parsed = parse_formula(&formula)
        .map_err(|err| format!("failed to re-parse survival marginal formula: {err}"))?;
    let (entryname, exitname, eventname) = parse_surv_response(&parsed.response)?
        .ok_or_else(|| "survival marginal-slope FFI requires Surv(...) response".to_string())?;
    let col_map: HashMap<String, usize> = dataset
        .headers
        .iter()
        .enumerate()
        .map(|(i, h)| (h.clone(), i))
        .collect();
    // `entryname == None` is the right-censored shorthand `Surv(time, event)`:
    // entry times are synthesized as zero, no column lookup required.
    let entry_idx: Option<usize> = entryname
        .as_deref()
        .map(|name| {
            col_map
                .get(name)
                .copied()
                .ok_or_else(|| format!("entry column '{name}' not found"))
        })
        .transpose()?;
    let exit_idx = *col_map
        .get(&exitname)
        .ok_or_else(|| format!("exit column '{exitname}' not found"))?;
    let n = dataset.values.nrows();
    let mut age_entry = Array1::<f64>::zeros(n);
    let mut age_exit = Array1::<f64>::zeros(n);
    for i in 0..n {
        let entry_val = entry_idx.map_or(0.0, |idx| dataset.values[[i, idx]]);
        let (t0, t1) = gam::families::survival::construction::normalize_survival_time_pair(
            entry_val,
            dataset.values[[i, exit_idx]],
            i,
        )?;
        age_entry[i] = t0;
        age_exit[i] = t1;
    }
    let baseline_cfg = parse_survival_baseline_config(
        &fit_config.baseline_target,
        fit_config.baseline_scale,
        fit_config.baseline_shape,
        fit_config.baseline_rate,
        fit_config.baseline_makeham,
    )?;
    let likelihood_mode = parse_survival_likelihood_mode(&fit_config.survival_likelihood)?;
    let time_cfg = if parsed.timewiggle.is_some() {
        gam::families::survival::construction::SurvivalTimeBasisConfig::None
    } else {
        parse_survival_time_basis_config(
            &fit_config.time_basis,
            fit_config.time_degree,
            fit_config.time_num_internal_knots,
            fit_config.time_smooth_lambda,
        )?
    };
    let time_anchor =
        resolve_survival_marginal_slope_time_anchor_value(&age_entry, &age_exit, None)?;
    let time_build = build_survival_time_basis(
        &age_entry,
        &age_exit,
        time_cfg,
        Some((
            fit_config.time_num_internal_knots,
            fit_config.time_smooth_lambda,
        )),
    )?;
    let timewiggle = match (
        ms_result.time_wiggle_knots.as_ref(),
        ms_result.time_wiggle_degree,
        ms_result.time_wiggle_ncols,
    ) {
        (None, None, 0) => None,
        (Some(knots), Some(degree), ncols) if ncols > 0 => {
            let beta_time = &ms_result
                .fit
                .blocks
                .first()
                .ok_or_else(|| {
                    "survival marginal-slope FFI fit is missing its time block".to_string()
                })?
                .beta;
            let p_base = time_build.x_exit_time.ncols();
            if beta_time.len() != p_base + ncols {
                return Err(format!(
                    "survival marginal-slope FFI timewiggle width mismatch: time beta={}, base={p_base}, wiggle={ncols}",
                    beta_time.len(),
                ));
            }
            Some(SurvivalTimewiggle {
                degree,
                knots: knots.to_vec(),
                penalty_orders: parsed
                    .timewiggle
                    .as_ref()
                    .map(|config| config.penalty_orders.clone()),
                double_penalty: parsed
                    .timewiggle
                    .as_ref()
                    .map(|config| config.double_penalty),
                beta: SurvivalTimewiggleBeta::Single(beta_time.slice(s![p_base..]).to_vec()),
            })
        }
        _ => {
            return Err(
                "survival marginal-slope FFI fit has incomplete timewiggle authority".to_string(),
            );
        }
    };
    let saved_offset_baseline =
        survival_marginal_slope_offset_baseline_config(&age_exit, &baseline_cfg);

    // Thin adapter over the shared core assembler. The FFI's source-specific
    // work is re-deriving the survival response columns, baseline config, and
    // time basis from the formula + FitConfig and freezing its term collections
    // from their designs; the semantic payload is assembled by the same core
    // path the CLI uses, so the two save routes produce identical contracts.
    Ok(assemble_survival_marginal_slope_payload(
        SurvivalMarginalSlopeInputs {
            formula,
            data_schema: dataset.schema.clone(),
            fit_result: ms_result.fit.clone(),
            frailty,
            survival_entry: entryname,
            survival_exit: exitname,
            survival_event: eventname,
            survivalspec: "net".to_string(),
            baseline_cfg: saved_offset_baseline,
            time_basis: SavedSurvivalTimeBasis::from_build(&time_build, time_anchor),
            ridge_lambda: fit_config.ridge_lambda,
            survival_likelihood_label: survival_likelihood_modename(likelihood_mode).to_string(),
            resolved_marginalspec: frozen_marginal,
            resolved_logslopespec: frozen_logslope,
            logslope_formula,
            z_column,
            latent_z_normalization: SavedLatentZNormalization {
                mean: ms_result.z_normalization.mean,
                sd: ms_result.z_normalization.sd,
            },
            baseline_logslope: ms_result.baseline_slope,
            timewiggle,
            score_warp_runtime: ms_result.score_warp_runtime.as_ref(),
            link_dev_runtime: ms_result.link_dev_runtime.as_ref(),
            influence_absorber_width: ms_result.influence_absorber_width,
            influence_absorber_design: ms_result.influence_absorber_design.as_ref(),
            score_covariance: &ms_result.score_covariance,
        },
        SavedModelSourceMetadata {
            training_headers: dataset.headers.clone(),
            training_feature_ranges: Some(dataset.feature_ranges()),
            offset_column: fit_config.offset_column.clone(),
            noise_offset_column: fit_config.noise_offset_column.clone(),
        },
    ))
}

fn build_survival_transformation_ffi_payload(
    formula: String,
    dataset: &EncodedDataset,
    fit_config: &FitConfig,
    rp_result: gam::families::fit_orchestration::SurvivalTransformationFitResult,
) -> Result<FittedModelPayload, String> {
    use gam::families::survival::construction::survival_likelihood_modename;
    use ndarray::s;

    let parsed = parse_formula(&formula)
        .map_err(|err| format!("failed to re-parse survival transformation formula: {err}"))?;
    let (entryname, exitname, eventname) = parse_surv_response(&parsed.response)?
        .ok_or_else(|| "survival transformation FFI requires Surv(...) response".to_string())?;
    let likelihood_label = survival_likelihood_modename(rp_result.likelihood_mode).to_string();

    let cause_count = rp_result.fit.blocks.len().max(1);
    let is_joint_cause_specific = cause_count > 1;

    // Source-specific work: extract the baseline-timewiggle coefficients from
    // the differently-shaped fit struct (one block for net, one per cause for
    // joint cause-specific). The canonical payload is then assembled by the same
    // shared core the CLI uses.
    let timewiggle = rp_result
        .baseline_timewiggle
        .as_ref()
        .map(|timewiggle| -> Result<SurvivalTimewiggle, String> {
            let start = rp_result.time_base_ncols;
            let end = start + timewiggle.ncols;
            let beta = if is_joint_cause_specific {
                let mut by_cause = Vec::with_capacity(cause_count);
                for (cause_idx, block) in rp_result.fit.blocks.iter().enumerate() {
                    if block.beta.len() < end {
                        return Err(format!(
                            "joint cause-specific survival timewiggle beta mismatch for cause {}: beta has {}, needs {end}",
                            cause_idx + 1,
                            block.beta.len()
                        ));
                    }
                    by_cause.push(block.beta.slice(s![start..end]).to_vec());
                }
                SurvivalTimewiggleBeta::ByCause(by_cause)
            } else {
                let beta = &rp_result.fit.beta;
                if beta.len() < end {
                    return Err(format!(
                        "survival transformation timewiggle beta mismatch: beta has {}, needs {end}",
                        beta.len()
                    ));
                }
                SurvivalTimewiggleBeta::Single(beta.slice(s![start..end]).to_vec())
            };
            Ok(SurvivalTimewiggle {
                degree: timewiggle.degree,
                knots: timewiggle.knots.to_vec(),
                penalty_orders: parsed.timewiggle.as_ref().map(|cfg| cfg.penalty_orders.clone()),
                double_penalty: parsed.timewiggle.as_ref().map(|cfg| cfg.double_penalty),
                beta,
            })
        })
        .transpose()?;

    let payload = assemble_survival_transformation_payload(
        SurvivalTransformationInputs {
            formula,
            data_schema: dataset.schema.clone(),
            fit_result: rp_result.fit.clone(),
            survival_entry: entryname,
            survival_exit: exitname,
            survival_event: eventname,
            survivalspec: if is_joint_cause_specific {
                "cause-specific".to_string()
            } else {
                "net".to_string()
            },
            cause_count: is_joint_cause_specific.then_some(cause_count),
            baseline_cfg: rp_result.baseline_cfg.clone(),
            time_basis: rp_result.time_basis.clone(),
            ridge_lambda: fit_config.ridge_lambda,
            survival_likelihood_label: likelihood_label,
            resolved_termspec: rp_result.resolvedspec,
            survival_beta_time: None,
            timewiggle,
        },
        SavedModelSourceMetadata {
            training_headers: dataset.headers.clone(),
            training_feature_ranges: Some(dataset.feature_ranges()),
            offset_column: fit_config.offset_column.clone(),
            noise_offset_column: None,
        },
    );
    Ok(payload)
}

fn build_gaussian_location_scale_ffi_payload(
    formula: String,
    dataset: &EncodedDataset,
    fit_config: &FitConfig,
    ls_result: GaussianLocationScaleFitResult,
    response_scale: f64,
) -> Result<FittedModelPayload, String> {
    let frozen_meanspec = freeze_term_collection_from_design(
        &ls_result.fit.meanspec_resolved,
        &ls_result.fit.mean_design,
    )
    .map_err(|err| format!("failed to freeze gaussian location-scale mean spec: {err}"))?;
    let frozen_noisespec = freeze_term_collection_from_design(
        &ls_result.fit.noisespec_resolved,
        &ls_result.fit.noise_design,
    )
    .map_err(|err| format!("failed to freeze gaussian location-scale noise spec: {err}"))?;

    let noise_formula = fit_config
        .noise_formula
        .clone()
        .ok_or_else(|| "gaussian location-scale requires noise_formula".to_string())?;

    let fit = ls_result.fit.fit;
    let scale_beta = fit
        .block_by_role(BlockRole::Scale)
        .map(|block| block.beta.to_vec());
    let wiggle = location_scale_wiggle_from_parts(
        ls_result.wiggle_knots,
        ls_result.wiggle_degree,
        ls_result.beta_link_wiggle,
    );

    // Thin adapter over the shared core assembler; the FFI freezes the mean and
    // noise specs from their designs and reads offset columns from the
    // FitConfig. See `assemble_location_scale_payload`.
    assemble_location_scale_payload(
        LocationScaleInputs {
            formula,
            data_schema: dataset.schema.clone(),
            noise_formula,
            resolved_termspec: frozen_meanspec,
            resolved_termspec_noise: frozen_noisespec,
            fit_result: fit,
            beta_noise: scale_beta,
            wiggle,
        },
        LocationScaleResponse::Gaussian {
            response_scale,
            base_link: None,
        },
        SavedModelSourceMetadata {
            training_headers: dataset.headers.clone(),
            training_feature_ranges: Some(dataset.feature_ranges()),
            offset_column: fit_config.offset_column.clone(),
            noise_offset_column: fit_config.noise_offset_column.clone(),
        },
    )
}

/// Map the optional `(knots, degree, beta)` link-wiggle parts a location-scale
/// fit may produce into the shared [`LocationScaleWiggle`] form. All three are
/// present together or not at all.
fn location_scale_wiggle_from_parts(
    knots: Option<Array1<f64>>,
    degree: Option<usize>,
    beta_link_wiggle: Option<Vec<f64>>,
) -> Option<LocationScaleWiggle> {
    match (knots, degree, beta_link_wiggle) {
        (Some(knots), Some(degree), Some(beta_link_wiggle)) => Some(LocationScaleWiggle {
            knots: knots.to_vec(),
            degree,
            beta_link_wiggle,
        }),
        _ => None,
    }
}

fn build_binomial_location_scale_ffi_payload(
    formula: String,
    dataset: &EncodedDataset,
    fit_config: &FitConfig,
    link_kind: InverseLink,
    weights: &Array1<f64>,
    ls_result: BinomialLocationScaleFitResult,
) -> Result<FittedModelPayload, String> {
    let frozen_meanspec = freeze_term_collection_from_design(
        &ls_result.fit.meanspec_resolved,
        &ls_result.fit.mean_design,
    )
    .map_err(|err| format!("failed to freeze binomial location-scale threshold spec: {err}"))?;
    let frozen_noisespec = freeze_term_collection_from_design(
        &ls_result.fit.noisespec_resolved,
        &ls_result.fit.noise_design,
    )
    .map_err(|err| format!("failed to freeze binomial location-scale noise spec: {err}"))?;

    let noise_formula = fit_config
        .noise_formula
        .clone()
        .ok_or_else(|| "binomial location-scale requires noise_formula".to_string())?;

    let dense_mean = ls_result
        .fit
        .mean_design
        .design
        .try_to_dense_by_chunks("binomial location-scale mean design")?;
    let dense_noise = ls_result
        .fit
        .noise_design
        .design
        .try_to_dense_by_chunks("binomial location-scale noise design")?;
    let non_intercept_start = ls_result
        .fit
        .noise_design
        .intercept_range
        .end
        .min(ls_result.fit.noise_design.design.ncols());
    let binomial_noise_transform =
        build_scale_deviation_transform(&dense_mean, &dense_noise, weights, non_intercept_start)
            .map_err(|err| format!("failed to encode binomial noise transform: {err}"))?;

    let fit = ls_result.fit.fit;
    let scale_beta = fit
        .block_by_role(BlockRole::Scale)
        .map(|block| block.beta.to_vec());
    let wiggle = location_scale_wiggle_from_parts(
        ls_result.wiggle_knots,
        ls_result.wiggle_degree,
        ls_result.beta_link_wiggle,
    );

    // Thin adapter over the shared core assembler; the FFI freezes the threshold
    // and noise specs from their designs, encodes the binomial noise
    // scale-deviation transform, and reads offset columns from the FitConfig.
    // See `assemble_location_scale_payload`.
    assemble_location_scale_payload(
        LocationScaleInputs {
            formula,
            data_schema: dataset.schema.clone(),
            noise_formula,
            resolved_termspec: frozen_meanspec,
            resolved_termspec_noise: frozen_noisespec,
            fit_result: fit,
            beta_noise: scale_beta,
            wiggle,
        },
        LocationScaleResponse::Binomial {
            link: link_kind,
            noise_transform: &binomial_noise_transform,
        },
        SavedModelSourceMetadata {
            training_headers: dataset.headers.clone(),
            training_feature_ranges: Some(dataset.feature_ranges()),
            offset_column: fit_config.offset_column.clone(),
            noise_offset_column: fit_config.noise_offset_column.clone(),
        },
    )
}

/// Assemble the saved-model payload for a genuine-dispersion location-scale fit
/// (#913): NegativeBinomial / Gamma / Beta / Tweedie with a `noise_formula` on
/// the overdispersion channel. Mirrors the CLI dispersion save path
/// (`assemble_location_scale_payload` + `LocationScaleResponse::Dispersion`),
/// deriving the persisted likelihood and mean base-link from the single
/// source of truth on [`DispersionFamilyKind`]. The log-precision block
/// coefficients ride in `beta_noise`; there is no link-wiggle and no response
/// standardization for these families.
fn build_dispersion_location_scale_ffi_payload(
    formula: String,
    dataset: &EncodedDataset,
    fit_config: &FitConfig,
    kind: DispersionFamilyKind,
    ls_result: DispersionLocationScaleFitResult,
) -> Result<FittedModelPayload, String> {
    let frozen_meanspec = freeze_term_collection_from_design(
        &ls_result.fit.meanspec_resolved,
        &ls_result.fit.mean_design,
    )
    .map_err(|err| format!("failed to freeze dispersion location-scale mean spec: {err}"))?;
    let frozen_noisespec = freeze_term_collection_from_design(
        &ls_result.fit.noisespec_resolved,
        &ls_result.fit.noise_design,
    )
    .map_err(|err| format!("failed to freeze dispersion location-scale noise spec: {err}"))?;

    let noise_formula = fit_config
        .noise_formula
        .clone()
        .ok_or_else(|| "dispersion location-scale requires noise_formula".to_string())?;

    let fit = ls_result.fit.fit;
    let scale_beta = fit
        .block_by_role(BlockRole::Scale)
        .map(|block| block.beta.to_vec());

    assemble_location_scale_payload(
        LocationScaleInputs {
            formula,
            data_schema: dataset.schema.clone(),
            noise_formula,
            resolved_termspec: frozen_meanspec,
            resolved_termspec_noise: frozen_noisespec,
            fit_result: fit,
            beta_noise: scale_beta,
            wiggle: None,
        },
        LocationScaleResponse::Dispersion {
            likelihood: kind.likelihood_spec(),
            base_link: kind.base_link(),
            family_tag: kind.family_tag(),
        },
        SavedModelSourceMetadata {
            training_headers: dataset.headers.clone(),
            training_feature_ranges: Some(dataset.feature_ranges()),
            offset_column: fit_config.offset_column.clone(),
            noise_offset_column: fit_config.noise_offset_column.clone(),
        },
    )
}

fn build_survival_location_scale_ffi_payload(
    formula: String,
    dataset: &EncodedDataset,
    fit_config: &FitConfig,
    weights: &Array1<f64>,
    ls_result: gam::families::fit_orchestration::SurvivalLocationScaleFitResult,
) -> Result<FittedModelPayload, String> {
    use gam::families::survival::construction::{
        build_survival_time_basis, parse_survival_baseline_config, parse_survival_likelihood_mode,
        parse_survival_time_basis_config, resolve_survival_time_anchor_value,
    };
    use ndarray::{Array2, s};

    // Re-derive survival metadata from the formula and FitConfig so we can
    // reproduce the saved model layout that the CLI persists.
    let parsed = gam::inference::formula_dsl::parse_formula(&formula)
        .map_err(|err| format!("failed to re-parse survival formula for FFI payload: {err}"))?;
    let (entryname, exitname, eventname) = parse_surv_response(&parsed.response)?
        .ok_or_else(|| "survival location-scale FFI requires Surv(...) response".to_string())?;
    let col_map: HashMap<String, usize> = dataset
        .headers
        .iter()
        .enumerate()
        .map(|(i, h)| (h.clone(), i))
        .collect();
    let entry_idx: Option<usize> = entryname
        .as_deref()
        .map(|name| {
            col_map
                .get(name)
                .copied()
                .ok_or_else(|| format!("entry column '{name}' not found"))
        })
        .transpose()?;
    let exit_idx = *col_map
        .get(&exitname)
        .ok_or_else(|| format!("exit column '{exitname}' not found"))?;
    let n = dataset.values.nrows();
    let mut age_entry = Array1::<f64>::zeros(n);
    let mut age_exit = Array1::<f64>::zeros(n);
    for i in 0..n {
        let entry_val = entry_idx.map_or(0.0, |idx| dataset.values[[i, idx]]);
        let (t0, t1) = gam::families::survival::construction::normalize_survival_time_pair(
            entry_val,
            dataset.values[[i, exit_idx]],
            i,
        )?;
        age_entry[i] = t0;
        age_exit[i] = t1;
    }
    let baseline_cfg = parse_survival_baseline_config(
        &fit_config.baseline_target,
        fit_config.baseline_scale,
        fit_config.baseline_shape,
        fit_config.baseline_rate,
        fit_config.baseline_makeham,
    )?;
    let likelihood_mode = parse_survival_likelihood_mode(&fit_config.survival_likelihood)?;
    let time_cfg = if parsed.timewiggle.is_some() {
        gam::families::survival::construction::SurvivalTimeBasisConfig::None
    } else {
        parse_survival_time_basis_config(
            &fit_config.time_basis,
            fit_config.time_degree,
            fit_config.time_num_internal_knots,
            fit_config.time_smooth_lambda,
        )?
    };
    let time_anchor = resolve_survival_time_anchor_value(&age_entry, None)?;
    let mut time_build = build_survival_time_basis(
        &age_entry,
        &age_exit,
        time_cfg,
        Some((
            fit_config.time_num_internal_knots,
            fit_config.time_smooth_lambda,
        )),
    )?;
    let resolved_time_cfg =
        gam::families::survival::construction::resolved_survival_time_basis_config_from_build(
            &time_build.basisname,
            time_build.degree,
            time_build.knots.as_ref(),
            time_build.keep_cols.as_ref(),
            time_build.smooth_lambda,
        )?;
    let time_anchor_row = gam::families::survival::construction::evaluate_survival_time_basis_row(
        time_anchor,
        &resolved_time_cfg,
    )?;
    gam::families::survival::construction::center_survival_time_designs_at_anchor(
        &mut time_build.x_entry_time,
        &mut time_build.x_exit_time,
        &time_anchor_row,
    )?;

    let fitted_inverse_link = ls_result.inverse_link.clone();
    // Compact the inner UnifiedFitResult and apply the fitted link state so
    // downstream prediction can recover the inverse-link parameters from the
    // saved fit_result. Mirrors the CLI's
    // compact_saved_survival_location_scale_fit_result helper.
    let mut fit_result = ls_result.fit.fit.clone();
    apply_inverse_link_state_to_fit_result(&mut fit_result, &fitted_inverse_link);
    fit_result.artifacts.survival_link_wiggle_knots = ls_result.wiggle_knots.clone();
    fit_result.artifacts.survival_link_wiggle_degree = ls_result.wiggle_degree;

    let resolved_thresholdspec = freeze_term_collection_from_design(
        &ls_result.fit.resolved_thresholdspec,
        &ls_result.fit.threshold_design,
    )
    .map_err(|err| err.to_string())?;
    let resolved_log_sigmaspec = freeze_term_collection_from_design(
        &ls_result.fit.resolved_log_sigmaspec,
        &ls_result.fit.log_sigma_design,
    )
    .map_err(|err| err.to_string())?;

    // Thin adapter over the shared core assembler. The FFI's source-specific
    // work above re-derives the survival metadata and compacts the fit result
    // with the fitted link state; the canonical payload is assembled by the
    // same path the CLI uses.
    Ok(assemble_survival_location_scale_payload(
        SurvivalLocationScaleInputs {
            formula,
            data_schema: dataset.schema.clone(),
            fit_result,
            fitted_inverse_link: fitted_inverse_link.clone(),
            linkwiggle_degree: ls_result.wiggle_degree,
            linkwiggle_knots: ls_result.wiggle_knots.as_ref().map(|k| k.to_vec()),
            beta_link_wiggle: ls_result
                .fit
                .fit
                .beta_link_wiggle()
                .as_ref()
                .map(|b| b.to_vec()),
            baseline_timewiggle: None,
            survival_entry: entryname,
            survival_exit: exitname,
            survival_event: eventname,
            survivalspec: "net".to_string(),
            baseline_cfg,
            time_basis: SavedSurvivalTimeBasis::from_build(&time_build, time_anchor),
            ridge_lambda: fit_config.ridge_lambda,
            survival_likelihood_label: survival_likelihood_modename(likelihood_mode).to_string(),
            time_parameterization: ls_result.fit.time_parameterization,
            threshold_time_basis: ls_result.fit.threshold_time_basis.clone(),
            log_sigma_time_basis: ls_result.fit.log_sigma_time_basis.clone(),
            formula_noise: None,
            survival_beta_time: ls_result.fit.fit.beta_time().to_vec(),
            survival_beta_threshold: ls_result.fit.fit.beta_threshold().to_vec(),
            survival_beta_log_sigma: ls_result.fit.fit.beta_log_sigma().to_vec(),
            resolved_thresholdspec,
            resolved_log_sigmaspec,
        },
        SavedModelSourceMetadata {
            training_headers: dataset.headers.clone(),
            training_feature_ranges: Some(dataset.feature_ranges()),
            offset_column: fit_config.offset_column.clone(),
            noise_offset_column: fit_config.noise_offset_column.clone(),
        },
    ))
}

fn build_latent_survival_ffi_payload(
    formula: String,
    dataset: &EncodedDataset,
    fit_config: &FitConfig,
    request_frailty: gam::families::survival::lognormal_kernel::FrailtySpec,
    lat_result: gam::families::survival::latent::LatentSurvivalTermFitResult,
) -> Result<FittedModelPayload, String> {
    build_latent_window_ffi_payload(
        formula,
        dataset,
        fit_config,
        request_frailty,
        lat_result.fit,
        lat_result.resolvedspec,
        lat_result.design,
        Some(lat_result.latent_sd),
        true,
    )
}

fn build_latent_binary_ffi_payload(
    formula: String,
    dataset: &EncodedDataset,
    fit_config: &FitConfig,
    request_frailty: gam::families::survival::lognormal_kernel::FrailtySpec,
    lat_result: gam::families::survival::latent::LatentBinaryTermFitResult,
) -> Result<FittedModelPayload, String> {
    build_latent_window_ffi_payload(
        formula,
        dataset,
        fit_config,
        request_frailty,
        lat_result.fit,
        lat_result.resolvedspec,
        lat_result.design,
        None,
        false,
    )
}

fn build_latent_window_ffi_payload(
    formula: String,
    dataset: &EncodedDataset,
    fit_config: &FitConfig,
    request_frailty: gam::families::survival::lognormal_kernel::FrailtySpec,
    fit: gam::solver::estimate::UnifiedFitResult,
    resolvedspec: gam::terms::smooth::TermCollectionSpec,
    cov_design: gam::terms::smooth::TermCollectionDesign,
    learned_latent_sd: Option<f64>,
    is_survival: bool,
) -> Result<FittedModelPayload, String> {
    use gam::families::survival::construction::{
        build_survival_time_basis, parse_survival_baseline_config,
        parse_survival_time_basis_config, resolve_survival_time_anchor_value,
    };

    let parsed = gam::inference::formula_dsl::parse_formula(&formula).map_err(|err| {
        format!("failed to re-parse latent survival formula for FFI payload: {err}")
    })?;
    let (entryname, exitname, eventname) = parse_surv_response(&parsed.response)?
        .ok_or_else(|| "latent survival/binary FFI requires Surv(...) response".to_string())?;
    let col_map: HashMap<String, usize> = dataset
        .headers
        .iter()
        .enumerate()
        .map(|(i, h)| (h.clone(), i))
        .collect();
    let entry_idx: Option<usize> = entryname
        .as_deref()
        .map(|name| {
            col_map
                .get(name)
                .copied()
                .ok_or_else(|| format!("entry column '{name}' not found"))
        })
        .transpose()?;
    let exit_idx = *col_map
        .get(&exitname)
        .ok_or_else(|| format!("exit column '{exitname}' not found"))?;
    let n = dataset.values.nrows();
    let mut age_entry = Array1::<f64>::zeros(n);
    let mut age_exit = Array1::<f64>::zeros(n);
    for i in 0..n {
        let entry_val = entry_idx.map_or(0.0, |idx| dataset.values[[i, idx]]);
        let (t0, t1) = gam::families::survival::construction::normalize_survival_time_pair(
            entry_val,
            dataset.values[[i, exit_idx]],
            i,
        )?;
        age_entry[i] = t0;
        age_exit[i] = t1;
    }
    let baseline_cfg = parse_survival_baseline_config(
        &fit_config.baseline_target,
        fit_config.baseline_scale,
        fit_config.baseline_shape,
        fit_config.baseline_rate,
        fit_config.baseline_makeham,
    )?;
    let time_cfg = parse_survival_time_basis_config(
        &fit_config.time_basis,
        fit_config.time_degree,
        fit_config.time_num_internal_knots,
        fit_config.time_smooth_lambda,
    )?;
    let time_anchor = resolve_survival_time_anchor_value(&age_entry, None)?;
    let time_build = build_survival_time_basis(
        &age_entry,
        &age_exit,
        time_cfg,
        Some((
            fit_config.time_num_internal_knots,
            fit_config.time_smooth_lambda,
        )),
    )?;

    // For latent survival, splice the fitted latent_sd into the persisted
    // HazardMultiplier frailty (mirrors CLI behaviour at main.rs:5541).
    let saved_family = if is_survival {
        let frailty = match (&request_frailty, learned_latent_sd) {
            (
                gam::families::survival::lognormal_kernel::FrailtySpec::HazardMultiplier {
                    sigma_fixed: None,
                    loading,
                },
                Some(sigma),
            ) => gam::families::survival::lognormal_kernel::FrailtySpec::HazardMultiplier {
                sigma_fixed: Some(sigma),
                loading: *loading,
            },
            _ => request_frailty.clone(),
        };
        FittedFamily::LatentSurvival { frailty }
    } else {
        FittedFamily::LatentBinary {
            frailty: request_frailty.clone(),
        }
    };
    let model_class_label = if is_survival {
        "latent-survival".to_string()
    } else {
        "latent-binary".to_string()
    };
    let likelihood_label = if is_survival {
        "latent".to_string()
    } else {
        "latent-binary".to_string()
    };

    let beta_time = fit.beta_time().to_vec();
    let resolved_termspec = freeze_term_collection_from_design(&resolvedspec, &cov_design)
        .map_err(|err| err.to_string())?;

    Ok(assemble_latent_window_payload(
        LatentWindowInputs {
            formula,
            data_schema: dataset.schema.clone(),
            fit_result: fit,
            family: saved_family,
            model_class_label,
            likelihood_label,
            survival_entry: entryname,
            survival_exit: exitname,
            survival_event: eventname,
            baseline_cfg,
            time_basis: SavedSurvivalTimeBasis::from_build(&time_build, time_anchor),
            ridge_lambda: fit_config.ridge_lambda,
            beta_time,
            resolved_termspec,
        },
        SavedModelSourceMetadata {
            training_headers: dataset.headers.clone(),
            training_feature_ranges: Some(dataset.feature_ranges()),
            offset_column: fit_config.offset_column.clone(),
            noise_offset_column: fit_config.noise_offset_column.clone(),
        },
    ))
}

fn predict_table_survival(
    model: &FittedModel,
    dataset: &EncodedDataset,
    options: &PyPredictOptions,
) -> Result<String, String> {
    if model
        .payload()
        .survival_cause_count
        .is_some_and(|cause_count| cause_count > 1)
    {
        let result = predict_competing_risks_survival_result(model, dataset, options)?;
        return serialize_competing_risks_prediction_payload(result, options.interval);
    }
    let result = predict_survival_result(model, dataset, options)?;
    serialize_survival_prediction_payload(model, result)
}

fn predict_competing_risks_survival_result(
    model: &FittedModel,
    dataset: &EncodedDataset,
    options: &PyPredictOptions,
) -> Result<gam::families::survival::predict::CompetingRisksPredictResult, String> {
    use gam::families::survival::predict::{
        SurvivalPredictEstimand, SurvivalPredictRequest, SurvivalPredictionCovarianceMode,
        predict_competing_risks_survival,
    };

    let col_map = dataset.column_map();
    let payload = model.payload();
    let training_headers = payload.training_headers.as_ref();
    let primary_offset =
        resolve_offset_column(dataset, &col_map, payload.offset_column.as_deref())?;
    let noise_offset = ndarray::Array1::<f64>::zeros(dataset.values.nrows());
    let time_grid_slice: Option<&[f64]> = options.time_grid.as_deref();
    let covariance_mode = if options.interval.is_some() {
        match parse_covariance_mode(options.covariance_mode.as_deref())?
            .unwrap_or(gam_predict::InferenceCovarianceMode::SmoothingCorrected)
        {
            gam_predict::InferenceCovarianceMode::Conditional => {
                SurvivalPredictionCovarianceMode::Conditional
            }
            gam_predict::InferenceCovarianceMode::SmoothingCorrected => {
                SurvivalPredictionCovarianceMode::SmoothingCorrected
            }
        }
    } else {
        // Posterior-mean points always integrate the conditional posterior;
        // covariance_mode controls uncertainty only.
        SurvivalPredictionCovarianceMode::Conditional
    };
    let request = SurvivalPredictRequest {
        model,
        data: dataset.values.view(),
        col_map: &col_map,
        training_headers,
        primary_offset: &primary_offset,
        noise_offset: &noise_offset,
        time_grid: time_grid_slice,
        // Issue #342: uncertainty is requested iff the user passed
        // `interval=`. The inner kernel keeps its own boolean knob — that
        // is an internal cost-control, not a public API flag.
        with_uncertainty: options.interval.is_some(),
        estimand: SurvivalPredictEstimand::PosteriorMean,
    };
    Ok(predict_competing_risks_survival(request, covariance_mode)?)
}

fn predict_survival_result(
    model: &FittedModel,
    dataset: &EncodedDataset,
    options: &PyPredictOptions,
) -> Result<gam::families::survival::predict::SurvivalPredictResult, String> {
    use gam::families::survival::predict::{
        SurvivalPredictEstimand, SurvivalPredictRequest, predict_survival,
    };

    let col_map = dataset.column_map();
    let payload = model.payload();
    let training_headers = payload.training_headers.as_ref();
    let primary_offset =
        resolve_offset_column(dataset, &col_map, payload.offset_column.as_deref())?;
    let supports_noise_offset = matches!(
        gam::families::survival::predict::require_saved_survival_likelihood_mode(model)?,
        gam::families::survival::construction::SurvivalLikelihoodMode::LocationScale
            | gam::families::survival::construction::SurvivalLikelihoodMode::MarginalSlope
    );
    let noise_offset = if supports_noise_offset {
        resolve_offset_column(dataset, &col_map, payload.noise_offset_column.as_deref())?
    } else {
        ndarray::Array1::<f64>::zeros(dataset.values.nrows())
    };
    let time_grid_slice: Option<&[f64]> = options.time_grid.as_deref();
    let request = SurvivalPredictRequest {
        model,
        data: dataset.values.view(),
        col_map: &col_map,
        training_headers,
        primary_offset: &primary_offset,
        noise_offset: &noise_offset,
        time_grid: time_grid_slice,
        // Issue #342: uncertainty is requested iff the user passed
        // `interval=`. The inner kernel keeps its own boolean knob — that
        // is an internal cost-control, not a public API flag.
        with_uncertainty: options.interval.is_some(),
        estimand: SurvivalPredictEstimand::PosteriorMean,
    };
    Ok(predict_survival(request)?)
}

fn serialize_survival_prediction_payload(
    model: &FittedModel,
    result: gam::families::survival::predict::SurvivalPredictResult,
) -> Result<String, String> {
    // Rowwise flatten for JSON transport.
    let n = result.hazard.nrows();
    let t = result.hazard.ncols();
    let mut hazard_rows = Vec::with_capacity(n);
    let mut survival_rows = Vec::with_capacity(n);
    let mut cumulative_rows = Vec::with_capacity(n);
    for i in 0..n {
        let mut hrow = Vec::with_capacity(t);
        let mut srow = Vec::with_capacity(t);
        let mut crow = Vec::with_capacity(t);
        for j in 0..t {
            hrow.push(result.hazard[[i, j]]);
            srow.push(result.survival[[i, j]]);
            crow.push(result.cumulative_hazard[[i, j]]);
        }
        hazard_rows.push(hrow);
        survival_rows.push(srow);
        cumulative_rows.push(crow);
    }
    let survival_se_rows = result.survival_se.as_ref().map(|se| {
        let mut rows: Vec<Vec<f64>> = Vec::with_capacity(n);
        for i in 0..n {
            let mut row = Vec::with_capacity(t);
            for j in 0..t {
                row.push(se[[i, j]]);
            }
            rows.push(row);
        }
        rows
    });
    let eta_se_vec = result.eta_se.as_ref().map(|v| v.to_vec());

    // Populate explicit probability columns. For survival models the event
    // probability is `1 - survival_prob`; an ambiguous `mean` column makes
    // fixed-time Brier/calibration scoring easy to invert.
    let mut columns = BTreeMap::<String, Vec<f64>>::new();
    columns.insert(
        "linear_predictor".to_string(),
        result.linear_predictor.to_vec(),
    );
    let survival_col: Vec<f64> = (0..n)
        .map(|i| result.survival[[i, t.saturating_sub(1)]])
        .collect();
    let failure_col: Vec<f64> = survival_col
        .iter()
        .map(|s| (1.0 - *s).clamp(0.0, 1.0))
        .collect();
    columns.insert("survival_prob".to_string(), survival_col);
    columns.insert("failure_prob".to_string(), failure_col);

    let likelihood_mode_str = match result.likelihood_mode {
        gam::families::survival::construction::SurvivalLikelihoodMode::MarginalSlope => {
            "marginal-slope"
        }
        gam::families::survival::construction::SurvivalLikelihoodMode::LocationScale => {
            "location-scale"
        }
        gam::families::survival::construction::SurvivalLikelihoodMode::Transformation => {
            "transformation"
        }
        gam::families::survival::construction::SurvivalLikelihoodMode::Weibull => "weibull",
        gam::families::survival::construction::SurvivalLikelihoodMode::Latent => "latent",
        gam::families::survival::construction::SurvivalLikelihoodMode::LatentBinary => {
            "latent-binary"
        }
    };
    let model_class_label = match result.likelihood_mode {
        gam::families::survival::construction::SurvivalLikelihoodMode::MarginalSlope => {
            "survival marginal-slope".to_string()
        }
        gam::families::survival::construction::SurvivalLikelihoodMode::LocationScale => {
            "survival location-scale".to_string()
        }
        gam::families::survival::construction::SurvivalLikelihoodMode::Latent => {
            "latent survival".to_string()
        }
        gam::families::survival::construction::SurvivalLikelihoodMode::Transformation
        | gam::families::survival::construction::SurvivalLikelihoodMode::Weibull
        | gam::families::survival::construction::SurvivalLikelihoodMode::LatentBinary => {
            model.predict_model_class().name().to_string()
        }
    };
    let survival_payload = SurvivalPredictionPayload {
        class: "survival_prediction",
        model_class: model_class_label,
        likelihood_mode: likelihood_mode_str.to_string(),
        times: result.times.clone(),
        hazard: hazard_rows,
        survival: survival_rows,
        cumulative_hazard: cumulative_rows,
        linear_predictor: result.linear_predictor.to_vec(),
        columns,
        survival_se: survival_se_rows,
        eta_se: eta_se_vec,
    };
    serde_json::to_string(&survival_payload)
        .map_err(|err| format!("failed to serialize survival prediction payload: {err}"))
}

fn competing_risks_surface_bounds(
    point: &[Array2<f64>],
    standard_error: &[Array2<f64>],
    z: f64,
    lower_limit: f64,
    upper_limit: f64,
    label: &str,
) -> Result<(Vec<Array2<f64>>, Vec<Array2<f64>>), String> {
    if point.len() != standard_error.len() {
        return Err(format!(
            "competing-risks {label} interval cause count mismatch: point={}, standard_error={}",
            point.len(),
            standard_error.len(),
        ));
    }
    let mut lower = Vec::with_capacity(point.len());
    let mut upper = Vec::with_capacity(point.len());
    for (cause, (point_surface, se_surface)) in point.iter().zip(standard_error).enumerate() {
        if point_surface.dim() != se_surface.dim() {
            return Err(format!(
                "competing-risks {label} interval shape mismatch for cause {}: point={:?}, standard_error={:?}",
                cause + 1,
                point_surface.dim(),
                se_surface.dim(),
            ));
        }
        let mut lower_surface = Array2::<f64>::zeros(point_surface.raw_dim());
        let mut upper_surface = Array2::<f64>::zeros(point_surface.raw_dim());
        for ((row, column), &point_value) in point_surface.indexed_iter() {
            let se = se_surface[[row, column]];
            if !(point_value.is_finite() && se.is_finite() && se >= 0.0) {
                return Err(format!(
                    "competing-risks {label} interval requires finite point and non-negative finite SE for cause {}, row {row}, time column {column}; got point={point_value}, se={se}",
                    cause + 1,
                ));
            }
            lower_surface[[row, column]] = (point_value - z * se).max(lower_limit);
            upper_surface[[row, column]] = (point_value + z * se).min(upper_limit);
        }
        lower.push(lower_surface);
        upper.push(upper_surface);
    }
    Ok((lower, upper))
}

fn competing_risks_matrix_bounds(
    point: &Array2<f64>,
    standard_error: &Array2<f64>,
    z: f64,
    lower_limit: f64,
    upper_limit: f64,
    label: &str,
) -> Result<(Array2<f64>, Array2<f64>), String> {
    let (mut lower, mut upper) = competing_risks_surface_bounds(
        std::slice::from_ref(point),
        std::slice::from_ref(standard_error),
        z,
        lower_limit,
        upper_limit,
        label,
    )?;
    Ok((lower.remove(0), upper.remove(0)))
}

fn competing_risks_vector_bounds(
    point: &[Array1<f64>],
    standard_error: &[Array1<f64>],
    z: f64,
    label: &str,
) -> Result<(Vec<Array1<f64>>, Vec<Array1<f64>>), String> {
    if point.len() != standard_error.len() {
        return Err(format!(
            "competing-risks {label} interval cause count mismatch: point={}, standard_error={}",
            point.len(),
            standard_error.len(),
        ));
    }
    let mut lower = Vec::with_capacity(point.len());
    let mut upper = Vec::with_capacity(point.len());
    for (cause, (point_vector, se_vector)) in point.iter().zip(standard_error).enumerate() {
        if point_vector.len() != se_vector.len() {
            return Err(format!(
                "competing-risks {label} interval length mismatch for cause {}: point={}, standard_error={}",
                cause + 1,
                point_vector.len(),
                se_vector.len(),
            ));
        }
        let mut lower_vector = Array1::<f64>::zeros(point_vector.len());
        let mut upper_vector = Array1::<f64>::zeros(point_vector.len());
        for row in 0..point_vector.len() {
            let point_value = point_vector[row];
            let se = se_vector[row];
            if !(point_value.is_finite() && se.is_finite() && se >= 0.0) {
                return Err(format!(
                    "competing-risks {label} interval requires finite point and non-negative finite SE for cause {}, row {row}; got point={point_value}, se={se}",
                    cause + 1,
                ));
            }
            lower_vector[row] = point_value - z * se;
            upper_vector[row] = point_value + z * se;
        }
        lower.push(lower_vector);
        upper.push(upper_vector);
    }
    Ok((lower, upper))
}

fn vectors_to_nested(vectors: &[Array1<f64>]) -> Vec<Vec<f64>> {
    vectors.iter().map(|vector| vector.to_vec()).collect()
}

fn serialize_competing_risks_prediction_payload(
    result: gam::families::survival::predict::CompetingRisksPredictResult,
    interval_level: Option<f64>,
) -> Result<String, String> {
    let covariance_source = match (interval_level, result.covariance_source) {
        (Some(_), Some(source)) => Some(source.as_str()),
        (Some(_), None) => {
            return Err(
                "competing-risks interval is missing resolved covariance provenance".to_string(),
            );
        }
        (None, Some(source)) => {
            return Err(format!(
                "competing-risks prediction resolved {} covariance without an interval request",
                source.as_str()
            ));
        }
        (None, None) => None,
    };
    let (
        hazard_lower,
        hazard_upper,
        survival_lower,
        survival_upper,
        cumulative_hazard_lower,
        cumulative_hazard_upper,
        cif_lower,
        cif_upper,
        overall_survival_lower,
        overall_survival_upper,
        eta_lower,
        eta_upper,
    ) = if let Some(level) = interval_level {
        let z = gam::inference::probability::standard_normal_quantile(0.5 + 0.5 * level)
            .map_err(|error| {
                format!(
                    "competing-risks prediction interval cannot construct the normal quantile for level {level}: {error}"
                )
            })?;
        if !z.is_finite() || z <= 0.0 {
            return Err(format!(
                "competing-risks prediction interval produced invalid normal quantile for level {level}: {z}"
            ));
        }
        let hazard_se = result.hazard_se.as_ref().ok_or_else(|| {
            "competing-risks interval requested but posterior hazard SE is missing".to_string()
        })?;
        let survival_se = result.survival_se.as_ref().ok_or_else(|| {
            "competing-risks interval requested but posterior survival SE is missing".to_string()
        })?;
        let cumulative_hazard_se = result.cumulative_hazard_se.as_ref().ok_or_else(|| {
            "competing-risks interval requested but posterior cumulative-hazard SE is missing"
                .to_string()
        })?;
        let cif_se = result.cif_se.as_ref().ok_or_else(|| {
            "competing-risks interval requested but posterior CIF SE is missing".to_string()
        })?;
        let overall_survival_se = result.overall_survival_se.as_ref().ok_or_else(|| {
            "competing-risks interval requested but posterior overall-survival SE is missing"
                .to_string()
        })?;
        let eta_se = result.eta_se.as_ref().ok_or_else(|| {
            "competing-risks interval requested but posterior eta SE is missing".to_string()
        })?;
        let (hazard_lower, hazard_upper) = competing_risks_surface_bounds(
            &result.hazard,
            hazard_se,
            z,
            0.0,
            f64::INFINITY,
            "hazard",
        )?;
        let (survival_lower, survival_upper) =
            competing_risks_surface_bounds(&result.survival, survival_se, z, 0.0, 1.0, "survival")?;
        let (cumulative_hazard_lower, cumulative_hazard_upper) = competing_risks_surface_bounds(
            &result.cumulative_hazard,
            cumulative_hazard_se,
            z,
            0.0,
            f64::INFINITY,
            "cumulative hazard",
        )?;
        let (cif_lower, cif_upper) = competing_risks_surface_bounds(
            &result.cif,
            cif_se,
            z,
            0.0,
            1.0,
            "cumulative incidence",
        )?;
        let (overall_survival_lower, overall_survival_upper) = competing_risks_matrix_bounds(
            &result.overall_survival,
            overall_survival_se,
            z,
            0.0,
            1.0,
            "overall survival",
        )?;
        let (eta_lower, eta_upper) =
            competing_risks_vector_bounds(&result.linear_predictor, eta_se, z, "eta")?;
        (
            Some(hazard_lower),
            Some(hazard_upper),
            Some(survival_lower),
            Some(survival_upper),
            Some(cumulative_hazard_lower),
            Some(cumulative_hazard_upper),
            Some(cif_lower),
            Some(cif_upper),
            Some(overall_survival_lower),
            Some(overall_survival_upper),
            Some(eta_lower),
            Some(eta_upper),
        )
    } else {
        if result.hazard_se.is_some()
            || result.survival_se.is_some()
            || result.cumulative_hazard_se.is_some()
            || result.cif_se.is_some()
            || result.overall_survival_se.is_some()
            || result.eta_se.is_some()
        {
            return Err(
                "competing-risks posterior SEs were produced without an interval level".to_string(),
            );
        }
        (
            None, None, None, None, None, None, None, None, None, None, None, None,
        )
    };
    let mut columns = BTreeMap::<String, Vec<f64>>::new();
    for (endpoint_idx, name) in result.endpoint_names.iter().enumerate() {
        let suffix = name.replace('-', "_");
        columns.insert(
            format!("eta_{suffix}"),
            result.linear_predictor[endpoint_idx].to_vec(),
        );
        let t_last = result.cif[endpoint_idx].ncols().saturating_sub(1);
        columns.insert(
            format!("failure_prob_{suffix}"),
            (0..result.cif[endpoint_idx].nrows())
                .map(|i| result.cif[endpoint_idx][[i, t_last]])
                .collect(),
        );
        if let (
            Some(eta_se),
            Some(eta_lower),
            Some(eta_upper),
            Some(cif_se),
            Some(cif_lower),
            Some(cif_upper),
        ) = (
            result.eta_se.as_ref(),
            eta_lower.as_ref(),
            eta_upper.as_ref(),
            result.cif_se.as_ref(),
            cif_lower.as_ref(),
            cif_upper.as_ref(),
        ) {
            columns.insert(
                format!("eta_{suffix}_std_error"),
                eta_se[endpoint_idx].to_vec(),
            );
            columns.insert(
                format!("eta_{suffix}_lower"),
                eta_lower[endpoint_idx].to_vec(),
            );
            columns.insert(
                format!("eta_{suffix}_upper"),
                eta_upper[endpoint_idx].to_vec(),
            );
            columns.insert(
                format!("failure_prob_{suffix}_std_error"),
                (0..cif_se[endpoint_idx].nrows())
                    .map(|i| cif_se[endpoint_idx][[i, t_last]])
                    .collect(),
            );
            columns.insert(
                format!("failure_prob_{suffix}_lower"),
                (0..cif_lower[endpoint_idx].nrows())
                    .map(|i| cif_lower[endpoint_idx][[i, t_last]])
                    .collect(),
            );
            columns.insert(
                format!("failure_prob_{suffix}_upper"),
                (0..cif_upper[endpoint_idx].nrows())
                    .map(|i| cif_upper[endpoint_idx][[i, t_last]])
                    .collect(),
            );
        }
    }
    let t_last = result.overall_survival.ncols().saturating_sub(1);
    columns.insert(
        "overall_survival".to_string(),
        (0..result.overall_survival.nrows())
            .map(|i| result.overall_survival[[i, t_last]])
            .collect(),
    );
    if let (Some(se), Some(lower), Some(upper)) = (
        result.overall_survival_se.as_ref(),
        overall_survival_lower.as_ref(),
        overall_survival_upper.as_ref(),
    ) {
        columns.insert(
            "overall_survival_std_error".to_string(),
            (0..se.nrows()).map(|i| se[[i, t_last]]).collect(),
        );
        columns.insert(
            "overall_survival_lower".to_string(),
            (0..lower.nrows()).map(|i| lower[[i, t_last]]).collect(),
        );
        columns.insert(
            "overall_survival_upper".to_string(),
            (0..upper.nrows()).map(|i| upper[[i, t_last]]).collect(),
        );
    }
    let likelihood_mode_str = match result.likelihood_mode {
        gam::families::survival::construction::SurvivalLikelihoodMode::MarginalSlope => {
            "marginal-slope"
        }
        gam::families::survival::construction::SurvivalLikelihoodMode::LocationScale => {
            "location-scale"
        }
        gam::families::survival::construction::SurvivalLikelihoodMode::Transformation => {
            "transformation"
        }
        gam::families::survival::construction::SurvivalLikelihoodMode::Weibull => "weibull",
        gam::families::survival::construction::SurvivalLikelihoodMode::Latent => "latent",
        gam::families::survival::construction::SurvivalLikelihoodMode::LatentBinary => {
            "latent-binary"
        }
    };
    let payload = serde_json::json!({
        "class": "competing_risks_prediction",
        "model_class": "competing risks survival",
        "likelihood_mode": likelihood_mode_str,
        "covariance_source": covariance_source,
        "endpoint_names": result.endpoint_names,
        "times": result.times,
        "interval_level": interval_level,
        "hazard": matrices_to_nested(&result.hazard),
        "hazard_se": result.hazard_se.as_ref().map(|value| matrices_to_nested(value)),
        "hazard_lower": hazard_lower.as_ref().map(|value| matrices_to_nested(value)),
        "hazard_upper": hazard_upper.as_ref().map(|value| matrices_to_nested(value)),
        "survival": matrices_to_nested(&result.survival),
        "survival_se": result.survival_se.as_ref().map(|value| matrices_to_nested(value)),
        "survival_lower": survival_lower.as_ref().map(|value| matrices_to_nested(value)),
        "survival_upper": survival_upper.as_ref().map(|value| matrices_to_nested(value)),
        "cumulative_hazard": matrices_to_nested(&result.cumulative_hazard),
        "cumulative_hazard_se": result
            .cumulative_hazard_se
            .as_ref()
            .map(|value| matrices_to_nested(value)),
        "cumulative_hazard_lower": cumulative_hazard_lower
            .as_ref()
            .map(|value| matrices_to_nested(value)),
        "cumulative_hazard_upper": cumulative_hazard_upper
            .as_ref()
            .map(|value| matrices_to_nested(value)),
        "cif": matrices_to_nested(&result.cif),
        "cif_se": result.cif_se.as_ref().map(|value| matrices_to_nested(value)),
        "cif_lower": cif_lower.as_ref().map(|value| matrices_to_nested(value)),
        "cif_upper": cif_upper.as_ref().map(|value| matrices_to_nested(value)),
        "overall_survival": matrix_to_nested(&result.overall_survival),
        "overall_survival_se": result
            .overall_survival_se
            .as_ref()
            .map(matrix_to_nested),
        "overall_survival_lower": overall_survival_lower.as_ref().map(matrix_to_nested),
        "overall_survival_upper": overall_survival_upper.as_ref().map(matrix_to_nested),
        "linear_predictor": result
            .linear_predictor
            .iter()
            .map(|eta| eta.to_vec())
            .collect::<Vec<_>>(),
        "eta_se": result.eta_se.as_ref().map(|value| vectors_to_nested(value)),
        "eta_lower": eta_lower.as_ref().map(|value| vectors_to_nested(value)),
        "eta_upper": eta_upper.as_ref().map(|value| vectors_to_nested(value)),
        "columns": columns,
    });
    serde_json::to_string(&payload)
        .map_err(|err| format!("failed to serialize competing-risks prediction payload: {err}"))
}

fn matrix_to_nested(matrix: &Array2<f64>) -> Vec<Vec<f64>> {
    (0..matrix.nrows())
        .map(|i| (0..matrix.ncols()).map(|j| matrix[[i, j]]).collect())
        .collect()
}

fn matrices_to_nested(matrices: &[Array2<f64>]) -> Vec<Vec<Vec<f64>>> {
    matrices.iter().map(matrix_to_nested).collect()
}

/// Round-trip a `ManifoldSAE.to_dict()` JSON payload through the Rust-owned
/// serde schema (`ManifoldSaePayload`, issue #2091) and return the re-serialized
/// payload. The v3 boundary is exact: all fields are required, runtime
/// diagnostics are persisted, and unknown/deprecated keys are rejected.
#[pyfunction(signature = (payload_json))]
fn sae_manifold_payload_roundtrip(payload_json: &str) -> PyResult<String> {
    crate::manifold::manifold_sae_payload::roundtrip_json(payload_json).map_err(py_value_error)
}

// --- Rust-owned #[pyclass] ManifoldSAE (issue #2091) ----------------------
//
// The Rust-owned model handle. It wraps the serde `ManifoldSaePayload` and
// exposes the flat attribute surface consumers read (dense arrays, config
// scalars, diagnostic/certificate report blocks, and per-atom objects) via
// `#[getter]`s. Serialization, OOS encoding/reconstruction, steering, Fisher
// attachment, and native fit construction all terminate on this class; there
// is no Python model adapter or alternate payload schema.

/// Build a numpy `(N,)` array from a flat `Vec<f64>`.
fn manifold_sae_vec1<'py>(py: Python<'py>, v: &[f64]) -> Bound<'py, PyArray1<f64>> {
    Array1::from(v.to_vec()).into_pyarray(py)
}

/// Build a numpy `(R, C)` array from nested `Vec`s, rejecting a ragged payload.
fn manifold_sae_vec2<'py>(py: Python<'py>, v: &[Vec<f64>]) -> PyResult<Bound<'py, PyArray2<f64>>> {
    let rows = v.len();
    let cols = v.first().map_or(0, Vec::len);
    let mut flat = Vec::with_capacity(rows * cols);
    for row in v {
        if row.len() != cols {
            return Err(py_value_error(
                "ManifoldSaeCore: ragged 2-D array in payload".to_string(),
            ));
        }
        flat.extend_from_slice(row);
    }
    Array2::from_shape_vec((rows, cols), flat)
        .map(|a| a.into_pyarray(py))
        .map_err(|e| py_value_error(e.to_string()))
}

/// Build a numpy `(D0, D1, D2)` array from triply-nested `Vec`s.
fn manifold_sae_vec3<'py>(
    py: Python<'py>,
    v: &[Vec<Vec<f64>>],
) -> PyResult<Bound<'py, PyArray3<f64>>> {
    let d0 = v.len();
    let d1 = v.first().map_or(0, Vec::len);
    let d2 = v.first().and_then(|m| m.first()).map_or(0, Vec::len);
    let mut flat = Vec::with_capacity(d0 * d1 * d2);
    for mat in v {
        if mat.len() != d1 {
            return Err(py_value_error(
                "ManifoldSaeCore: ragged 3-D array in payload".to_string(),
            ));
        }
        for row in mat {
            if row.len() != d2 {
                return Err(py_value_error(
                    "ManifoldSaeCore: ragged 3-D array in payload".to_string(),
                ));
            }
            flat.extend_from_slice(row);
        }
    }
    Array3::from_shape_vec((d0, d1, d2), flat)
        .map(|a| a.into_pyarray(py))
        .map_err(|e| py_value_error(e.to_string()))
}

/// A Python list of per-atom `(R_k, C)` numpy arrays (e.g. `coords`,
/// `decoder_blocks`).
fn manifold_sae_list2<'py>(
    py: Python<'py>,
    mats: &[Vec<Vec<f64>>],
) -> PyResult<Bound<'py, PyList>> {
    let list = PyList::empty(py);
    for m in mats {
        list.append(manifold_sae_vec2(py, m)?)?;
    }
    Ok(list)
}

/// A report/certificate block: the stored dict, or Python `None`.
fn manifold_sae_report(py: Python<'_>, value: &Option<serde_json::Value>) -> PyResult<PyObject> {
    match value {
        None => Ok(py.None()),
        Some(v) => json_value_to_py(py, v.clone()),
    }
}

/// Rebuild the dense `(M_k·p, M_k·p)` decoder covariance the atom surface exposes
/// from the compact per-channel factor `(p, M_k, M_k)` stored on disk (#2091):
/// block-diagonal across channels, `cov[b1·p + c, b2·p + c] = factor[c][b1][b2]`.
/// This is the same reassembly as `decoder_cov_from_channel_factors`, so
/// `atom.decoder_covariance` reproduces the shape band exactly (cross-channel
/// entries, which no band reads, are zero).
fn manifold_sae_dense_cov<'py>(
    py: Python<'py>,
    factors: &[Vec<Vec<f64>>],
) -> PyResult<Bound<'py, PyArray2<f64>>> {
    let p = factors.len();
    let m = factors.first().map_or(0, Vec::len);
    let side = m * p;
    let mut cov = Array2::<f64>::zeros((side, side));
    for (c, block) in factors.iter().enumerate() {
        if block.len() != m {
            return Err(py_value_error(
                "AtomCore.decoder_covariance: ragged per-channel factor".to_string(),
            ));
        }
        for (b1, row) in block.iter().enumerate() {
            if row.len() != m {
                return Err(py_value_error(
                    "AtomCore.decoder_covariance: non-square per-channel block".to_string(),
                ));
            }
            for (b2, &value) in row.iter().enumerate() {
                cov[[b1 * p + c, b2 * p + c]] = value;
            }
        }
    }
    Ok(cov.into_pyarray(py))
}

/// Build an owned `(R, C)` ndarray from nested `Vec`s (for feeding Rust-owned
/// model state into the shared steering/OOS rebuild without a numpy round-trip).
fn manifold_sae_owned2(v: &[Vec<f64>]) -> PyResult<Array2<f64>> {
    let rows = v.len();
    let cols = v.first().map_or(0, Vec::len);
    let mut flat = Vec::with_capacity(rows * cols);
    for row in v {
        if row.len() != cols {
            return Err(py_value_error(
                "ManifoldSaeCore: ragged 2-D state array".to_string(),
            ));
        }
        flat.extend_from_slice(row);
    }
    Array2::from_shape_vec((rows, cols), flat).map_err(|e| py_value_error(e.to_string()))
}

/// Sparsity summary stats for an `(n_rows, K)` assignment matrix returned by
/// `sae_manifold_fit*`. Returns `(avg_active_atoms, mean_assignment_mass)` where
/// "active" is `assignment >= threshold`.
fn manifold_assignment_summary_from_array(
    assignments: ArrayView2<'_, f64>,
    threshold: f64,
) -> Result<(f64, f64), String> {
    if !threshold.is_finite() {
        return Err("assignment summary threshold must be finite".to_string());
    }
    let (n_rows, k) = assignments.dim();
    if n_rows == 0 || k == 0 {
        return Err("assignment summary requires a non-empty matrix".to_string());
    }
    let n_entries = n_rows
        .checked_mul(k)
        .ok_or_else(|| "assignment summary shape is too large".to_string())?;
    let mut active_total = 0_usize;
    let mut mass_total = 0.0_f64;
    for &assignment in assignments.iter() {
        if !assignment.is_finite() {
            return Err("assignment summary contains a non-finite value".to_string());
        }
        mass_total += assignment;
        if !mass_total.is_finite() {
            return Err("assignment summary mass overflowed".to_string());
        }
        if assignment >= threshold {
            active_total += 1;
        }
    }
    Ok((
        active_total as f64 / n_rows as f64,
        mass_total / n_entries as f64,
    ))
}

/// Build an owned `(D0, D1, D2)` ndarray from triply-nested `Vec`s.
fn manifold_sae_owned3(v: &[Vec<Vec<f64>>]) -> PyResult<Array3<f64>> {
    let d0 = v.len();
    let d1 = v.first().map_or(0, Vec::len);
    let d2 = v.first().and_then(|m| m.first()).map_or(0, Vec::len);
    let mut flat = Vec::with_capacity(d0 * d1 * d2);
    for mat in v {
        if mat.len() != d1 {
            return Err(py_value_error(
                "ManifoldSaeCore: ragged 3-D state array".to_string(),
            ));
        }
        for row in mat {
            if row.len() != d2 {
                return Err(py_value_error(
                    "ManifoldSaeCore: ragged 3-D state array".to_string(),
                ));
            }
            flat.extend_from_slice(row);
        }
    }
    Array3::from_shape_vec((d0, d1, d2), flat).map_err(|e| py_value_error(e.to_string()))
}

/// Build the packed row metric owned by a fitted model and validate the
/// serialized Fisher fields as one coherent state.  This runs exactly once at
/// construction/load time (and once for each explicit `attach_fisher` call),
/// never from `steer`.
fn manifold_sae_resident_fisher_metric(
    payload: &crate::manifold::manifold_sae_payload::ManifoldSaePayload,
) -> PyResult<Option<gam::inference::row_metric::RowMetric>> {
    let n_rows = payload.fitted.len();
    let p_out = payload.fitted.first().map_or(0, Vec::len);
    match &payload.fisher_factors {
        None => {
            if payload.fisher_provenance.is_some()
                || payload.fisher_factor_kind.is_some()
                || payload.fisher_mass_residual.is_some()
            {
                return Err(py_value_error(
                    "ManifoldSAE: Fisher provenance, factor status, and residual mass require retained fisher_factors"
                        .to_string(),
                ));
            }
            match payload.metric_provenance.as_str() {
                // A structured-residual fit uses its estimated whitening metric
                // while fitting, but the artifact retains no output-behavior
                // shard. Steering is therefore geometry-only after load, just as
                // it is for Euclidean fits; the persisted provenance remains an
                // honest account of the metric used to fit the dictionary.
                "Euclidean" | "WhitenedStructured" => Ok(None),
                provenance => Err(py_value_error(format!(
                    "ManifoldSAE: metric provenance {provenance:?} requires retained fisher_factors"
                ))),
            }
        }
        Some(factors) => {
            let provenance = payload.fisher_provenance.as_deref().ok_or_else(|| {
                py_value_error(
                    "ManifoldSAE: fisher_provenance is required when fisher_factors are present"
                        .to_string(),
                )
            })?;
            let factor_kind = payload.fisher_factor_kind.as_deref().ok_or_else(|| {
                py_value_error(
                    "ManifoldSAE: fisher_factor_kind is required when fisher_factors are present"
                        .to_string(),
                )
            })?;
            let factors = manifold_sae_owned3(factors)?;
            let mass = payload
                .fisher_mass_residual
                .as_ref()
                .map(|values| Array1::from(values.clone()));
            let request = SaeFisherRowMetricRequest::from_tag(
                factors.view(),
                n_rows,
                p_out,
                Some(provenance),
                Some(factor_kind),
                mass.as_ref().map(|values| values.view()),
            )
            .map_err(py_value_error)?;
            let metric = build_sae_fisher_row_metric(request).map_err(py_value_error)?;
            let label = gam::terms::sae::manifold::metric_provenance_label(metric.provenance());
            if payload.metric_provenance != label {
                return Err(py_value_error(format!(
                    "ManifoldSAE: metric_provenance {:?} disagrees with Fisher metric {:?}",
                    payload.metric_provenance, label
                )));
            }
            Ok(Some(metric))
        }
    }
}

/// Parse a JSON array of numbers into an owned `Array1` (for the hybrid-split
/// linear-image `b0`/`b1`/`v` vectors read from the stored payload).
fn manifold_sae_json_vec1(value: Option<&serde_json::Value>) -> PyResult<Array1<f64>> {
    let arr = value.and_then(|v| v.as_array()).ok_or_else(|| {
        py_value_error("hybrid_split linear_image b0/b1/v must be a JSON array".to_string())
    })?;
    let mut out = Vec::with_capacity(arr.len());
    for x in arr {
        out.push(x.as_f64().ok_or_else(|| {
            py_value_error("hybrid_split linear_image array has a non-numeric entry".to_string())
        })?);
    }
    Ok(Array1::from(out))
}

/// Rust port of `ManifoldSAE._hybrid_linear_images_for_oos` (#1228/#2091): read
/// the trained dictionary's hybrid-collapsed straight sub-models out of the
/// stored `hybrid_split` block as `(atom_idx, t_bar, b0, b1, v)` tuples for the
/// OOS reconstruction. Mirrors the Python exactly — an entry without a
/// `linear_image` is skipped, `v` is `None` for an ordinary straight image and
/// `Some` for a collapse-rescued slot, and an empty result maps to `None` (all
/// -curved OOS reconstruction).
fn manifold_sae_hybrid_linear_images(
    hybrid_split: &Option<serde_json::Value>,
) -> PyResult<Option<Vec<(usize, f64, Array1<f64>, Array1<f64>, Option<Array1<f64>>)>>> {
    let Some(hs) = hybrid_split else {
        return Ok(None);
    };
    let Some(atoms) = hs.get("atoms").and_then(|a| a.as_array()) else {
        return Ok(None);
    };
    let mut images = Vec::new();
    for entry in atoms {
        let linear_image = entry.get("linear_image");
        let Some(li) = linear_image else {
            continue;
        };
        // Mirror the Python `if not li: continue` — skip a null / empty
        // `linear_image` (an all-curved slot carries no straight image).
        let is_empty_container = li.as_object().map(|o| o.is_empty()).unwrap_or(false)
            || li.as_array().map(|a| a.is_empty()).unwrap_or(false);
        if li.is_null() || is_empty_container {
            continue;
        }
        let atom_idx = li.get("atom_idx").and_then(|v| v.as_f64()).ok_or_else(|| {
            py_value_error("hybrid_split linear_image missing atom_idx".to_string())
        })? as usize;
        let t_bar = li
            .get("t_bar")
            .and_then(|v| v.as_f64())
            .ok_or_else(|| py_value_error("hybrid_split linear_image missing t_bar".to_string()))?;
        let b0 = manifold_sae_json_vec1(li.get("b0"))?;
        let b1 = manifold_sae_json_vec1(li.get("b1"))?;
        let v = match li.get("v") {
            None => None,
            Some(x) if x.is_null() => None,
            Some(x) => Some(manifold_sae_json_vec1(Some(x))?),
        };
        images.push((atom_idx, t_bar, b0, b1, v));
    }
    Ok(if images.is_empty() {
        None
    } else {
        Some(images)
    })
}

/// A single fitted atom's object surface (#2091). Mirrors the attributes
/// `SaeManifoldAtomFit` exposed (`atom.basis`, `atom.decoder_coefficients`,
/// `atom.decoder_covariance` reconstructed dense, the shape band, …) so
/// `ManifoldSaeCore.atoms` stays a list of objects consumers read by attribute,
/// not a list of dicts. Additive: the Python dataclass remains the live facade.
#[pyclass(module = "gamfit._rust", name = "AtomCore")]
pub(crate) struct AtomCore {
    inner: crate::manifold::manifold_sae_payload::AtomPayload,
}

#[pymethods]
impl AtomCore {
    #[getter]
    fn basis(&self) -> String {
        self.inner.basis.clone()
    }
    #[getter]
    fn decoder_coefficients<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyArray2<f64>>> {
        manifold_sae_vec2(py, &self.inner.decoder_coefficients)
    }
    #[getter]
    fn assignments<'py>(&self, py: Python<'py>) -> Bound<'py, PyArray1<f64>> {
        manifold_sae_vec1(py, &self.inner.assignments)
    }
    #[getter]
    fn coords<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyArray2<f64>>> {
        manifold_sae_vec2(py, &self.inner.coords)
    }
    #[getter]
    fn coords_u_arc<'py>(&self, py: Python<'py>) -> Option<Bound<'py, PyArray1<f64>>> {
        self.inner
            .coords_u_arc
            .as_ref()
            .map(|v| manifold_sae_vec1(py, v))
    }
    #[getter]
    fn evidence(&self) -> Option<f64> {
        self.inner.evidence
    }
    #[getter]
    fn active_dim(&self) -> i64 {
        self.inner.active_dim
    }
    /// The DENSE `(M_k·p, M_k·p)` posterior covariance (or `None`), rebuilt from
    /// the compact per-channel factor stored on disk.
    #[getter]
    fn decoder_covariance<'py>(
        &self,
        py: Python<'py>,
    ) -> PyResult<Option<Bound<'py, PyArray2<f64>>>> {
        match &self.inner.decoder_covariance_channel_factors {
            None => Ok(None),
            Some(factors) => Ok(Some(manifold_sae_dense_cov(py, factors)?)),
        }
    }
    #[getter]
    fn shape_band_coords<'py>(
        &self,
        py: Python<'py>,
    ) -> PyResult<Option<Bound<'py, PyArray2<f64>>>> {
        match &self.inner.shape_band_coords {
            None => Ok(None),
            Some(v) => Ok(Some(manifold_sae_vec2(py, v)?)),
        }
    }
    #[getter]
    fn shape_band_mean<'py>(&self, py: Python<'py>) -> PyResult<Option<Bound<'py, PyArray2<f64>>>> {
        match &self.inner.shape_band_mean {
            None => Ok(None),
            Some(v) => Ok(Some(manifold_sae_vec2(py, v)?)),
        }
    }
    #[getter]
    fn shape_band_sd<'py>(&self, py: Python<'py>) -> PyResult<Option<Bound<'py, PyArray2<f64>>>> {
        match &self.inner.shape_band_sd {
            None => Ok(None),
            Some(v) => Ok(Some(manifold_sae_vec2(py, v)?)),
        }
    }
    #[getter]
    fn functional_evidence(&self, py: Python<'_>) -> PyResult<PyObject> {
        manifold_sae_report(py, &self.inner.functional_evidence)
    }
}

/// Rust-owned fitted `ManifoldSAE` model handle (#2091).  The Python-visible
/// type is the public model itself; there is no wrapper/adapter class.
#[pyclass(module = "gamfit._rust", name = "ManifoldSAE")]
pub(crate) struct ManifoldSaeCore {
    inner: crate::manifold::manifold_sae_payload::ManifoldSaePayload,
    /// Packed, validated Fisher row metric. `RowMetric` keeps its factor matrix
    /// behind `Arc`, so cloning it into a steering request is O(1).
    fisher_metric: Option<gam::inference::row_metric::RowMetric>,
    /// Observable contract counter: construction/attach increments this once;
    /// repeated `steer` calls leave it unchanged.
    fisher_metric_build_count: usize,
}

impl ManifoldSaeCore {
    pub(crate) fn from_payload(
        inner: crate::manifold::manifold_sae_payload::ManifoldSaePayload,
    ) -> PyResult<Self> {
        let fisher_metric = manifold_sae_resident_fisher_metric(&inner)?;
        let fisher_metric_build_count = usize::from(fisher_metric.is_some());
        Ok(Self {
            inner,
            fisher_metric,
            fisher_metric_build_count,
        })
    }

    /// Translate the artifact's strict non-negative integer harmonic field into
    /// the optional wire representation consumed by the common array marshaller.
    /// Topology interpretation belongs exclusively to
    /// `gam_sae::manifold::persisted_oos_atom_specs`.
    fn persisted_harmonic_orders(&self) -> PyResult<Vec<Option<usize>>> {
        self.inner
            .n_harmonics
            .iter()
            .enumerate()
            .map(|(atom, &order)| {
                usize::try_from(order).map(Some).map_err(|_| {
                    py_value_error(format!(
                        "ManifoldSAE: n_harmonics[{atom}] must be non-negative; got {order}"
                    ))
                })
            })
            .collect()
    }

    fn description_length_report(
        &self,
        l_param_bits: Option<f64>,
    ) -> PyResult<Option<gam::terms::sae::description_length::ManifoldFitDl>> {
        if self.inner.atoms.is_empty() || self.inner.fitted.is_empty() {
            return Ok(None);
        }
        if l_param_bits.is_some_and(|value| !value.is_finite() || value < 0.0) {
            return Err(py_value_error(
                "ManifoldSAE.description_length: l_param_bits must be finite and non-negative"
                    .to_string(),
            ));
        }
        let assignments = manifold_sae_owned2(&self.inner.assignments)?;
        let coords = self
            .inner
            .coords
            .iter()
            .map(|block| manifold_sae_owned2(block))
            .collect::<PyResult<Vec<_>>>()?;
        let coord_views = coords.iter().map(|block| block.view()).collect::<Vec<_>>();
        let n_params = self
            .inner
            .decoder_blocks
            .iter()
            .try_fold(0_usize, |total, block| {
                let block_size = block
                    .iter()
                    .try_fold(0_usize, |rows, row| rows.checked_add(row.len()))?;
                total.checked_add(block_size)
            })
            .ok_or_else(|| {
                py_value_error("ManifoldSAE decoder parameter count overflowed".to_string())
            })?;
        let n_params = i64::try_from(n_params).map_err(|_| {
            py_value_error("ManifoldSAE decoder parameter count exceeds i64".to_string())
        })?;
        manifold_description_length_from_arrays(
            assignments.view(),
            &coord_views,
            self.inner.reconstruction_r2,
            n_params,
            l_param_bits,
            1.0e-8,
        )
        .map(Some)
        .map_err(py_value_error)
    }

    /// Build the OOS argument bundle from this handle's state and run the
    /// frozen-decoder Newton solve, returning the full payload dict
    /// (`assignments_z`, `on_atom_coords_t`, `logits`, `fitted`). The Rust-owned
    /// counterpart of `ManifoldSAE._oos_payload`: it threads the trained geometry,
    /// terminal ρ* (`selected_log_*`), learnable-α flag, and hybrid-collapsed
    /// straight sub-models exactly as the Python does, so the returned arrays are
    /// bitwise-identical to the dataclass OOS path. No warm start is supplied
    /// (`initial_logits`/`initial_coords` are `None`), matching a bare
    /// `reconstruct`/`encode` call. The coordinate ridge takes the same `1e-6`
    /// default the `sae_manifold_predict_oos` pyfunction applies when Python
    /// omits it; there is no decoder ridge in a frozen-decoder solve.
    fn oos_payload_dict<'py>(
        &self,
        py: Python<'py>,
        x_new: PyReadonlyArray2<'py, f64>,
    ) -> PyResult<Py<PyDict>> {
        let inner = &self.inner;
        let decoder_owned: Vec<Array2<f64>> = inner
            .decoder_blocks
            .iter()
            .map(|b| manifold_sae_owned2(b))
            .collect::<PyResult<_>>()?;
        let duchon_owned: Vec<Option<Array2<f64>>> = inner
            .duchon_centers
            .iter()
            .map(|c| c.as_ref().map(|m| manifold_sae_owned2(m)).transpose())
            .collect::<PyResult<_>>()?;
        let atom_dim: Vec<usize> = inner.atom_dims.iter().map(|&d| d.max(0) as usize).collect();
        let basis_sizes: Vec<usize> = inner
            .basis_sizes
            .iter()
            .map(|&s| s.max(0) as usize)
            .collect();
        let n_harm = self.persisted_harmonic_orders()?;
        let hybrid = manifold_sae_hybrid_linear_images(&inner.hybrid_split)?;
        let decoder_views: Vec<ndarray::ArrayView2<'_, f64>> =
            decoder_owned.iter().map(|a| a.view()).collect();
        let request = sae_oos_request_from_arrays(
            x_new.as_array(),
            inner.basis_kinds.clone(),
            atom_dim,
            &decoder_views,
            &duchon_owned,
            n_harm,
            basis_sizes,
            inner.alpha,
            inner.tau,
            inner.assignment.clone(),
            inner.max_iter.max(0) as usize,
            inner.learning_rate,
            // Coordinate ridge: Python `_oos_payload` omits it, so the
            // `sae_manifold_predict_oos` pyfunction supplies this `1e-6` default.
            1.0e-6,
            None,
            None,
            inner.threshold_gate_threshold,
            inner.top_k.map(|t| t.max(0) as usize),
            hybrid,
            inner.selected_log_lambda_sparse,
            inner.selected_log_lambda_smooth.clone(),
            inner.selected_log_ard.clone(),
            inner.learnable_alpha,
        )
        .map_err(py_value_error)?;
        let report =
            gam::terms::sae::manifold::run_sae_manifold_oos(request).map_err(py_value_error)?;
        sae_oos_report_to_pydict(py, report)
    }
}

#[pymethods]
impl ManifoldSaeCore {
    /// Construct from a `ManifoldSAE.to_dict()` payload dict. The dict is
    /// serialized (it is already JSON-able — lists, not numpy) and validated
    /// through the strict `ManifoldSaePayload::from_json` schema.
    #[new]
    fn new(py: Python<'_>, payload: &Bound<'_, PyDict>) -> PyResult<Self> {
        Self::from_dict(py, payload)
    }

    #[staticmethod]
    fn from_dict(py: Python<'_>, payload: &Bound<'_, PyDict>) -> PyResult<Self> {
        let json_mod = py.import("json")?;
        let dumped = json_mod.getattr("dumps")?.call1((payload,))?;
        let json_str: String = dumped.extract()?;
        let inner = crate::manifold::manifold_sae_payload::ManifoldSaePayload::from_json(&json_str)
            .map_err(py_value_error)?;
        Self::from_payload(inner)
    }

    #[staticmethod]
    fn from_json(py: Python<'_>, payload_json: &str) -> PyResult<Py<ManifoldSaeCore>> {
        let inner =
            crate::manifold::manifold_sae_payload::ManifoldSaePayload::from_json(payload_json)
                .map_err(py_value_error)?;
        Py::new(py, Self::from_payload(inner)?)
    }

    #[staticmethod]
    fn load(py: Python<'_>, path: std::path::PathBuf) -> PyResult<Py<ManifoldSaeCore>> {
        let payload_json = std::fs::read_to_string(path)
            .map_err(|error| py_value_error(format!("ManifoldSAE.load: {error}")))?;
        Self::from_json(py, &payload_json)
    }

    /// Re-serialize the complete v5 artifact as a Python dict.
    fn to_dict(&self, py: Python<'_>) -> PyResult<PyObject> {
        let json_str = self.inner.to_json().map_err(py_value_error)?;
        let value: serde_json::Value =
            serde_json::from_str(&json_str).map_err(|e| py_value_error(e.to_string()))?;
        json_value_to_py(py, value)
    }

    /// The canonical JSON payload string (what `save()` writes).
    fn to_json(&self) -> PyResult<String> {
        self.inner.to_json().map_err(py_value_error)
    }

    fn save(&self, path: std::path::PathBuf) -> PyResult<()> {
        let payload = self.inner.to_json().map_err(py_value_error)?;
        std::fs::write(path, payload)
            .map_err(|error| py_value_error(format!("ManifoldSAE.save: {error}")))
    }

    fn __repr__(&self) -> String {
        let n_rows = self.inner.fitted.len();
        let p_out = self.inner.fitted.first().map_or(0, Vec::len);
        format!(
            "ManifoldSAE(K={}, n={}, p={}, topology={:?}, assignment={:?}, r2={:.3})",
            self.inner.atoms.len(),
            n_rows,
            p_out,
            self.inner.atom_topology,
            self.inner.assignment,
            self.inner.reconstruction_r2,
        )
    }

    /// Fit-level code length from the native manifold-SAE description-length
    /// kernel. Every input is read from this immutable fitted artifact.
    #[pyo3(signature = (*, l_param_bits=None))]
    fn description_length(
        &self,
        py: Python<'_>,
        l_param_bits: Option<f64>,
    ) -> PyResult<Option<PyObject>> {
        self.description_length_report(l_param_bits)?
            .as_ref()
            .map(|report| manifold_description_length_to_pydict(py, report))
            .transpose()
    }

    /// Compact fitted-model report assembled from persisted native fields and
    /// the same Rust description-length/assignment-summary kernels used by the
    /// standalone FFI functions.
    fn summary(&self, py: Python<'_>) -> PyResult<Py<PyDict>> {
        let description = self.description_length_report(None)?;
        let description_py = description
            .as_ref()
            .map(|report| manifold_description_length_to_pydict(py, report))
            .transpose()?;
        let assignments = manifold_sae_owned2(&self.inner.assignments)?;
        let (avg_active_atoms, mean_assignment_mass) =
            manifold_assignment_summary_from_array(assignments.view(), 1.0e-8)
                .map_err(py_value_error)?;
        let common_dim = self.inner.atom_dims.first().copied().filter(|first| {
            self.inner
                .atom_dims
                .iter()
                .all(|dimension| dimension == first)
        });

        let active_dims = self
            .inner
            .atoms
            .iter()
            .map(|atom| atom.active_dim)
            .collect::<Vec<_>>();
        let atom_functionals = PyList::empty(py);
        for atom in &self.inner.atoms {
            atom_functionals.append(manifold_sae_report(py, &atom.functional_evidence)?)?;
        }

        let selected_smooth = self
            .inner
            .selected_log_lambda_smooth
            .as_ref()
            .map(|values| values.iter().map(|value| value.exp()).collect::<Vec<_>>());
        let out = PyDict::new(py);
        out.set_item(
            "bits_per_token",
            description.as_ref().map(|report| report.bits_per_token),
        )?;
        out.set_item("description_length", description_py)?;
        out.set_item("K", self.inner.atoms.len())?;
        out.set_item("d_atom", common_dim)?;
        out.set_item("atom_dims", self.inner.atom_dims.clone())?;
        out.set_item("atom_topology", self.inner.atom_topology.clone())?;
        out.set_item("atom_topologies", self.inner.atom_topologies.clone())?;
        out.set_item("assignment", self.inner.assignment.clone())?;
        out.set_item("alpha", self.inner.alpha)?;
        out.set_item("learnable_alpha", self.inner.learnable_alpha)?;
        out.set_item("penalized_loss_score", self.inner.penalized_loss_score)?;
        out.set_item(
            "penalized_quasi_laplace_criterion",
            self.inner.penalized_quasi_laplace_criterion,
        )?;
        out.set_item("reconstruction_r2", self.inner.reconstruction_r2)?;
        out.set_item("dispersion", self.inner.dispersion)?;
        out.set_item("avg_active_atoms", avg_active_atoms)?;
        out.set_item("mean_assignment_mass", mean_assignment_mass)?;
        out.set_item("active_dims", active_dims)?;
        out.set_item("atom_functionals", atom_functionals)?;
        out.set_item(
            "diagnostics",
            json_value_to_py(py, self.inner.diagnostics.clone())?,
        )?;
        out.set_item("cotrain", manifold_sae_report(py, &self.inner.cotrain)?)?;
        out.set_item("primitives", self.inner.primitive_names.clone())?;
        out.set_item("selected_smooth_lambdas", selected_smooth)?;
        Ok(out.unbind())
    }

    /// In-sample dense reconstruction `(N, p)` rebuilt from the stored per-atom
    /// coordinates, assignment codes, and decoder blocks — the Rust-owned
    /// counterpart of `ManifoldSAE.reconstruct_training`. Reads the codes from
    /// this handle's own state and calls the SAME pure-Rust core
    /// (`reconstruct_persisted_atom_set`) the `sae_manifold_reconstruct_ffi`
    /// pyfunction uses, so the output is bitwise-identical to the dataclass path.
    fn reconstruct_training<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyArray2<f64>>> {
        let inner = &self.inner;
        let basis_kinds: Vec<SaeAtomBasisKind> = inner
            .basis_kinds
            .iter()
            .map(|name| sae_atom_basis_kind_from_str(name))
            .collect();
        let atom_dims: Vec<usize> = inner.atom_dims.iter().map(|&d| d.max(0) as usize).collect();
        let decoder_owned: Vec<Array2<f64>> = inner
            .decoder_blocks
            .iter()
            .map(|b| manifold_sae_owned2(b))
            .collect::<PyResult<_>>()?;
        let coord_owned: Vec<Array2<f64>> = inner
            .coords
            .iter()
            .map(|c| manifold_sae_owned2(c))
            .collect::<PyResult<_>>()?;
        let assignments = manifold_sae_owned2(&inner.assignments)?;
        // Mirror the Python p_out: fitted columns when there are no atoms, else
        // the trained decoder block's output width.
        let p_out = if inner.decoder_blocks.is_empty() {
            inner.fitted.first().map_or(0, Vec::len)
        } else {
            inner.decoder_blocks[0].first().map_or(0, Vec::len)
        };
        let decoder_views: Vec<ndarray::ArrayView2<'_, f64>> =
            decoder_owned.iter().map(|a| a.view()).collect();
        let coord_views: Vec<ndarray::ArrayView2<'_, f64>> =
            coord_owned.iter().map(|a| a.view()).collect();
        let out = gam::terms::sae::manifold::reconstruct_persisted_atom_set(
            &basis_kinds,
            &atom_dims,
            &decoder_views,
            &coord_views,
            assignments.view(),
            p_out,
        )
        .map_err(py_value_error)?;
        Ok(out.into_pyarray(py))
    }

    /// Dense reconstruction `(N, P)` from EXTERNALLY-supplied assignment codes
    /// `(N, K)`, decoded against this handle's stored per-atom coordinates and
    /// frozen decoder blocks — the accessor SAEBench's SCR / unlearning arm reads
    /// (zero a latent's code column, then decode to measure the ablated output).
    /// Identical persisted-atom-set assembler as
    /// [`reconstruct_training`](Self::reconstruct_training); only the assignment
    /// matrix is the caller's `codes` instead of the stored assignments. Like
    /// `reconstruct_training` it decodes each atom at its trained coordinate and
    /// does NOT re-apply the joint fit's hybrid collapse. The core validates
    /// `codes` has `K` columns (== atom count) and `p_out`-consistent widths.
    fn reconstruct_from_assignments<'py>(
        &self,
        py: Python<'py>,
        codes: PyReadonlyArray2<'py, f64>,
    ) -> PyResult<Bound<'py, PyArray2<f64>>> {
        let inner = &self.inner;
        let basis_kinds: Vec<SaeAtomBasisKind> = inner
            .basis_kinds
            .iter()
            .map(|name| sae_atom_basis_kind_from_str(name))
            .collect();
        let atom_dims: Vec<usize> = inner.atom_dims.iter().map(|&d| d.max(0) as usize).collect();
        let decoder_owned: Vec<Array2<f64>> = inner
            .decoder_blocks
            .iter()
            .map(|b| manifold_sae_owned2(b))
            .collect::<PyResult<_>>()?;
        let coord_owned: Vec<Array2<f64>> = inner
            .coords
            .iter()
            .map(|c| manifold_sae_owned2(c))
            .collect::<PyResult<_>>()?;
        let p_out = if inner.decoder_blocks.is_empty() {
            inner.fitted.first().map_or(0, Vec::len)
        } else {
            inner.decoder_blocks[0].first().map_or(0, Vec::len)
        };
        let decoder_views: Vec<ndarray::ArrayView2<'_, f64>> =
            decoder_owned.iter().map(|a| a.view()).collect();
        let coord_views: Vec<ndarray::ArrayView2<'_, f64>> =
            coord_owned.iter().map(|a| a.view()).collect();
        let out = gam::terms::sae::manifold::reconstruct_persisted_atom_set(
            &basis_kinds,
            &atom_dims,
            &decoder_views,
            &coord_views,
            codes.as_array(),
            p_out,
        )
        .map_err(py_value_error)?;
        Ok(out.into_pyarray(py))
    }

    /// Frozen anchor dictionary `(K, P)` — one representative ambient direction
    /// per atom, the linear-`(K, P)`-dictionary analog SAEBench's curved
    /// `audit_sae` arm consumes when collapsing the curved per-atom decoders.
    ///
    /// MODELING CHOICE (documented, principled, basis-agnostic): row `k` is the
    /// MEAN decoded ambient direction of atom `k` over its OWN fitted coordinates,
    /// `mean_i Φ_k(coords_k[i])·B_k` — the "center" of the curved atom's shape.
    /// This is deliberately the atom's INTRINSIC mean shape (each row decoded at
    /// unit assignment weight, NOT its activation-weighted contribution), so an
    /// atom's dictionary direction does not shrink just because it was rarely
    /// active. For a d=1 Fourier atom it coincides with the DC / constant-harmonic
    /// decoder row; for a linear atom it is the mean of its linear image. Chosen
    /// over "decode at the single mean coordinate" because a coordinate-space mean
    /// is ill-defined for a periodic atom (angles need a circular mean), whereas
    /// the ambient-image mean is always well-defined. The per-atom decode reuses
    /// the same persisted-atom-set assembler as `reconstruct_training`; the row
    /// reduction is done in Rust. An atom with zero fitted rows yields a zero row.
    fn frozen_dictionary<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyArray2<f64>>> {
        let inner = &self.inner;
        let k_atoms = inner.decoder_blocks.len();
        let p_out = if inner.decoder_blocks.is_empty() {
            inner.fitted.first().map_or(0, Vec::len)
        } else {
            inner.decoder_blocks[0].first().map_or(0, Vec::len)
        };
        let mut dictionary = Array2::<f64>::zeros((k_atoms, p_out));
        for k in 0..k_atoms {
            let kind = sae_atom_basis_kind_from_str(&inner.basis_kinds[k]);
            let atom_dim = inner.atom_dims[k].max(0) as usize;
            let decoder = manifold_sae_owned2(&inner.decoder_blocks[k])?;
            let coords = manifold_sae_owned2(&inner.coords[k])?;
            let n_rows = coords.nrows();
            if n_rows == 0 {
                continue;
            }
            // Decode atom k ALONE at its stored coordinates (unit weight) through
            // the shared assembler, then average the ambient rows (Rust-side).
            let assignments = Array2::<f64>::ones((n_rows, 1));
            let decoder_view = decoder.view();
            let coords_view = coords.view();
            let decoded = gam::terms::sae::manifold::reconstruct_persisted_atom_set(
                std::slice::from_ref(&kind),
                std::slice::from_ref(&atom_dim),
                std::slice::from_ref(&decoder_view),
                std::slice::from_ref(&coords_view),
                assignments.view(),
                p_out,
            )
            .map_err(py_value_error)?;
            let mean = decoded.mean_axis(ndarray::Axis(0)).ok_or_else(|| {
                py_value_error("frozen_dictionary: atom decoded to zero rows".to_string())
            })?;
            dictionary.row_mut(k).assign(&mean);
        }
        Ok(dictionary.into_pyarray(py))
    }

    /// Steering plan with output dosimetry for one atom — the Rust-owned
    /// counterpart of `ManifoldSAE.steer` (#980/#2091). Reads the model geometry
    /// (decoder blocks, coords, logits, and the attached output-Fisher shard)
    /// from this handle's own state and routes through the SAME
    /// `steer_delta_from_arrays` rebuild the `sae_steer_delta` pyfunction uses, so
    /// the Fisher shard is NOT re-marshalled across the FFI boundary per call
    /// (acceptance bullet 2) and the returned plan is bitwise-identical to the
    /// dataclass path. Mirrors the Python steer's exact `n_harmonics` gate
    /// (`periodic`/`torus` only) so the rebuilt basis matches the trained design.
    #[pyo3(signature = (atom_k, metric_row, amplitude, t_from, t_to))]
    fn steer<'py>(
        &self,
        py: Python<'py>,
        atom_k: usize,
        metric_row: usize,
        amplitude: f64,
        t_from: PyReadonlyArray1<'py, f64>,
        t_to: PyReadonlyArray1<'py, f64>,
    ) -> PyResult<Py<PyDict>> {
        let inner = &self.inner;
        let decoder_owned: Vec<Array2<f64>> = inner
            .decoder_blocks
            .iter()
            .map(|b| manifold_sae_owned2(b))
            .collect::<PyResult<_>>()?;
        let coord_owned: Vec<Array2<f64>> = inner
            .coords
            .iter()
            .map(|c| manifold_sae_owned2(c))
            .collect::<PyResult<_>>()?;
        let duchon_owned: Vec<Option<Array2<f64>>> = inner
            .duchon_centers
            .iter()
            .map(|c| c.as_ref().map(|m| manifold_sae_owned2(m)).transpose())
            .collect::<PyResult<_>>()?;
        let logits_owned = manifold_sae_owned2(&inner.low_level_logits)?;
        let atom_dim: Vec<usize> = inner.atom_dims.iter().map(|&d| d.max(0) as usize).collect();
        let basis_sizes: Vec<usize> = inner
            .basis_sizes
            .iter()
            .map(|&s| s.max(0) as usize)
            .collect();
        let n_harm = self.persisted_harmonic_orders()?;
        let decoder_views: Vec<ndarray::ArrayView2<'_, f64>> =
            decoder_owned.iter().map(|a| a.view()).collect();
        let coord_views: Vec<ndarray::ArrayView2<'_, f64>> =
            coord_owned.iter().map(|a| a.view()).collect();
        let top_k = inner
            .top_k
            .map(|support| {
                usize::try_from(support).map_err(|_| {
                    py_value_error(format!(
                        "ManifoldSAE.steer: saved top_k must be non-negative; got {support}"
                    ))
                })
            })
            .transpose()?;
        let plan = steer_delta_with_metric_from_arrays(
            atom_k,
            metric_row,
            amplitude,
            t_from.as_array(),
            t_to.as_array(),
            &inner.basis_kinds,
            &atom_dim,
            &decoder_views,
            &duchon_owned,
            &n_harm,
            &basis_sizes,
            &coord_views,
            logits_owned.view(),
            inner.assignment.as_str(),
            top_k,
            inner.tau,
            inner.alpha,
            inner.threshold_gate_threshold,
            self.fisher_metric.clone(),
        )?;
        steer_plan_to_pydict(py, plan)
    }

    /// Target-dose steering (gh#2263): solve for the amplitude that lands a
    /// requested output-KL dose on one atom chord, instead of asking the caller
    /// for a meaningless raw amplitude. The required `request` mapping contains
    /// `atom_k`, `metric_row`, `target_nats`, `t_from`, `t_to`, `tol_rel`,
    /// `max_iter`, and `readout_tol_rel`. Returns the
    /// closed-form seed `a0 = sqrt(2 q*/(dgᵀ M dg))` plus, when a plan-aware
    /// `probe` is supplied, the closed-loop-corrected amplitude and one atomic
    /// observation mapping with `effective_delta`, `exact_directional_nats`,
    /// `measured_nats`, and required optional
    /// `certified_attainable_upper_nats`. The latter is `None` unless the callback
    /// can prove a global upper bound for every non-negative amplitude on this
    /// exact chord; pointwise plateaus are not certificates. The callback receives
    /// the complete public steer-plan mapping, not a scalar amplitude, so it cannot
    /// silently execute a different activation move. Reuses the
    /// SAME frozen-dictionary rebuild as `steer`.
    #[pyo3(signature = (request, probe = None))]
    fn steer_to_target<'py>(
        &self,
        py: Python<'py>,
        request: &Bound<'py, PyDict>,
        probe: Option<PyObject>,
    ) -> PyResult<Py<PyDict>> {
        let ManifoldSteerToTargetRequest {
            atom_k,
            metric_row,
            target_nats,
            t_from,
            t_to,
            config,
        } = ManifoldSteerToTargetRequest::from_pydict(request)?;
        let inner = &self.inner;
        // Same frozen-dictionary marshalling as `steer` (kept inline so the two
        // steering entries rebuild from identical handle state).
        let decoder_owned: Vec<Array2<f64>> = inner
            .decoder_blocks
            .iter()
            .map(|b| manifold_sae_owned2(b))
            .collect::<PyResult<_>>()?;
        let coord_owned: Vec<Array2<f64>> = inner
            .coords
            .iter()
            .map(|c| manifold_sae_owned2(c))
            .collect::<PyResult<_>>()?;
        let duchon_owned: Vec<Option<Array2<f64>>> = inner
            .duchon_centers
            .iter()
            .map(|c| c.as_ref().map(|m| manifold_sae_owned2(m)).transpose())
            .collect::<PyResult<_>>()?;
        let logits_owned = manifold_sae_owned2(&inner.low_level_logits)?;
        let atom_dim: Vec<usize> = inner.atom_dims.iter().map(|&d| d.max(0) as usize).collect();
        let basis_sizes: Vec<usize> = inner
            .basis_sizes
            .iter()
            .map(|&s| s.max(0) as usize)
            .collect();
        let n_harm = self.persisted_harmonic_orders()?;
        let decoder_views: Vec<ndarray::ArrayView2<'_, f64>> =
            decoder_owned.iter().map(|a| a.view()).collect();
        let coord_views: Vec<ndarray::ArrayView2<'_, f64>> =
            coord_owned.iter().map(|a| a.view()).collect();
        let top_k = inner
            .top_k
            .map(|support| {
                usize::try_from(support).map_err(|_| {
                    py_value_error(format!(
                        "ManifoldSAE.steer_to_target: saved top_k must be non-negative; got {support}"
                    ))
                })
            })
            .transpose()?;

        // The external model adapter receives the exact public plan and returns
        // the three inseparable quantities the Rust dose contract validates.
        let mut probe_boxed: Option<
            Box<gam::inference::steering::AppliedDoseProbe<'_>>,
        > = probe.map(|obj| {
            let boxed: Box<gam::inference::steering::AppliedDoseProbe<'_>> =
                Box::new(move |plan: &gam::inference::steering::SteerPlan| {
                    Python::attach(|py| {
                        let plan_dict = steer_plan_to_pydict(py, plan.clone()).map_err(|error| {
                            format!("steer_to_target could not serialize probe plan: {error}")
                        })?;
                        let result = obj
                            .call1(py, (plan_dict,))
                            .map_err(|error| format!("steer_to_target probe raised: {error}"))?;
                        let result = result.bind(py).cast::<PyDict>().map_err(|error| {
                            format!(
                                "steer_to_target probe must return a mapping with effective_delta, \
                                 exact_directional_nats, measured_nats, and \
                                 certified_attainable_upper_nats: {error}"
                            )
                        })?;
                        let required = |key: &str| {
                            result
                                .get_item(key)
                                .map_err(|error| error.to_string())?
                                .ok_or_else(|| {
                                    format!("steer_to_target probe result is missing {key:?}")
                                })
                        };
                        let effective_delta = required("effective_delta")?
                            .extract::<Vec<f64>>()
                            .map_err(|error| {
                                format!(
                                    "steer_to_target probe effective_delta must be a float list: {error}"
                                )
                            })?;
                        let exact_directional_nats = required("exact_directional_nats")?
                            .extract::<f64>()
                            .map_err(|error| {
                                format!(
                                    "steer_to_target probe exact_directional_nats must be a float: {error}"
                                )
                            })?;
                        let measured_nats = required("measured_nats")?
                            .extract::<f64>()
                            .map_err(|error| {
                                format!(
                                    "steer_to_target probe measured_nats must be a float: {error}"
                                )
                            })?;
                        let certified_attainable_upper_nats =
                            required("certified_attainable_upper_nats")?
                                .extract::<Option<f64>>()
                                .map_err(|error| {
                                    format!(
                                        "steer_to_target probe \
                                         certified_attainable_upper_nats must be a float or \
                                         None: {error}"
                                    )
                                })?;
                        Ok(gam::inference::steering::AppliedDoseObservation {
                            effective_delta: Array1::from(effective_delta),
                            exact_directional_nats,
                            measured_nats,
                            certified_attainable_upper_nats,
                        })
                    })
                });
            boxed
        });
        let probe_ref = probe_boxed.as_deref_mut();

        let plan = steer_to_target_from_arrays(
            SteerToTargetArraysRequest {
                atom_k,
                metric_row,
                target_nats,
                config,
                t_from: t_from.view(),
                t_to: t_to.view(),
                atom_basis: &inner.basis_kinds,
                atom_dim: &atom_dim,
                decoder_blocks: &decoder_views,
                duchon_centers: &duchon_owned,
                n_harmonics_list: &n_harm,
                basis_size_list: &basis_sizes,
                coords: &coord_views,
                logits: logits_owned.view(),
                assignment_kind: inner.assignment.as_str(),
                top_k,
                tau: inner.tau,
                alpha: inner.alpha,
                threshold_gate_threshold: inner.threshold_gate_threshold,
                fisher_metric: self.fisher_metric.clone(),
            },
            probe_ref,
        )?;
        target_dose_plan_to_pydict(py, plan)
    }

    /// Held-out dense reconstruction `(N, p)` of `x_new` — the Rust-owned
    /// counterpart of `ManifoldSAE.reconstruct` for out-of-sample rows. Runs the
    /// frozen-decoder OOS Newton solve through the SAME typed gam-sae entry the
    /// `sae_manifold_predict_oos` pyfunction uses, reading the trained
    /// geometry, terminal ρ*, and hybrid-collapsed straight sub-models from this
    /// handle's own state (no per-call re-marshalling), and returns the payload's
    /// `fitted` array. Unlike the dataclass path there is no training-data
    /// shortcut: every call runs the OOS solve.
    fn reconstruct<'py>(
        &self,
        py: Python<'py>,
        x_new: PyReadonlyArray2<'py, f64>,
    ) -> PyResult<PyObject> {
        let payload = self.oos_payload_dict(py, x_new)?;
        let bound = payload.bind(py);
        let fitted = bound
            .get_item("fitted")?
            .ok_or_else(|| py_value_error("OOS payload missing 'fitted'".to_string()))?;
        Ok(fitted.unbind())
    }

    fn predict<'py>(
        &self,
        py: Python<'py>,
        x_new: PyReadonlyArray2<'py, f64>,
    ) -> PyResult<PyObject> {
        let payload = self.oos_payload_dict(py, x_new)?;
        let bound = payload.bind(py);
        let fitted = bound
            .get_item("fitted")?
            .ok_or_else(|| py_value_error("OOS payload missing 'fitted'".to_string()))?;
        Ok(fitted.unbind())
    }

    /// One coherent converged latent state.
    ///
    /// With no input, this returns the state persisted by the training fit.
    /// With `x_new`, it runs one frozen-decoder OOS solve and returns its
    /// reconstruction, assignments, logits, and coordinates together; no field
    /// is recomputed by a separate inference call.
    #[pyo3(signature = (x_new=None))]
    fn converged_latents<'py>(
        &self,
        py: Python<'py>,
        x_new: Option<PyReadonlyArray2<'py, f64>>,
    ) -> PyResult<Py<PyDict>> {
        if let Some(values) = x_new {
            let payload = self.oos_payload_dict(py, values)?;
            let raw = payload.bind(py);
            let assignments = raw
                .get_item("assignments_z")?
                .ok_or_else(|| py_value_error("OOS payload missing 'assignments_z'".to_string()))?;
            let fitted = raw
                .get_item("fitted")?
                .ok_or_else(|| py_value_error("OOS payload missing 'fitted'".to_string()))?;
            let logits = raw
                .get_item("logits")?
                .ok_or_else(|| py_value_error("OOS payload missing 'logits'".to_string()))?;
            let atoms = raw
                .get_item("atoms")?
                .ok_or_else(|| py_value_error("OOS payload missing 'atoms'".to_string()))?;
            let atoms = atoms.cast::<PyList>()?;
            let coords = PyList::empty(py);
            let atom_images = PyList::empty(py);
            for atom in atoms.iter() {
                let atom = atom.cast::<PyDict>()?;
                let block = atom.get_item("on_atom_coords_t")?.ok_or_else(|| {
                    py_value_error("OOS atom payload missing 'on_atom_coords_t'".to_string())
                })?;
                coords.append(block)?;
                let reconstruction = atom.get_item("atom_reconstruction")?.ok_or_else(|| {
                    py_value_error("OOS atom payload missing 'atom_reconstruction'".to_string())
                })?;
                atom_images.append(reconstruction)?;
            }
            let out = PyDict::new(py);
            out.set_item("fitted", fitted)?;
            out.set_item("assignments", assignments)?;
            out.set_item("logits", logits)?;
            out.set_item("coords", coords)?;
            // Native, unweighted per-atom decoded images g_k(t_ik). Consumers
            // obtain the additive contribution as a_ik * g_k(t_ik).
            // Exposing the already-materialized OOS report prevents Python
            // callers from replaying an incomplete basis zoo (and getting
            // sphere / torus / Mobius decoding wrong).
            out.set_item("atom_images", atom_images)?;
            return Ok(out.unbind());
        }

        let out = PyDict::new(py);
        out.set_item("fitted", manifold_sae_vec2(py, &self.inner.fitted)?)?;
        out.set_item(
            "assignments",
            manifold_sae_vec2(py, &self.inner.assignments)?,
        )?;
        out.set_item(
            "logits",
            manifold_sae_vec2(py, &self.inner.low_level_logits)?,
        )?;
        out.set_item("coords", manifold_sae_list2(py, &self.inner.coords)?)?;
        Ok(out.unbind())
    }

    /// Held-out soft assignment codes `(N, K)` for `x_new` — the Rust-owned
    /// counterpart of `ManifoldSAE.encode`. Runs the same frozen-decoder OOS
    /// solve as [`reconstruct`](Self::reconstruct) and returns the payload's
    /// `assignments_z` array.
    fn encode<'py>(
        &self,
        py: Python<'py>,
        x_new: PyReadonlyArray2<'py, f64>,
    ) -> PyResult<PyObject> {
        let payload = self.oos_payload_dict(py, x_new)?;
        let bound = payload.bind(py);
        let codes = bound
            .get_item("assignments_z")?
            .ok_or_else(|| py_value_error("OOS payload missing 'assignments_z'".to_string()))?;
        Ok(codes.unbind())
    }

    // --- dense numeric getters -------------------------------------------
    #[getter]
    fn fitted<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyArray2<f64>>> {
        manifold_sae_vec2(py, &self.inner.fitted)
    }
    #[getter]
    fn assignments<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyArray2<f64>>> {
        manifold_sae_vec2(py, &self.inner.assignments)
    }
    #[getter]
    fn low_level_logits<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyArray2<f64>>> {
        manifold_sae_vec2(py, &self.inner.low_level_logits)
    }
    #[getter]
    fn training_mean<'py>(&self, py: Python<'py>) -> Bound<'py, PyArray1<f64>> {
        manifold_sae_vec1(py, &self.inner.training_mean)
    }
    #[getter]
    fn tier0_scale<'py>(&self, py: Python<'py>) -> Option<Bound<'py, PyArray1<f64>>> {
        self.inner
            .tier0_scale
            .as_ref()
            .map(|scale| manifold_sae_vec1(py, scale))
    }
    #[getter]
    fn coords<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyList>> {
        manifold_sae_list2(py, &self.inner.coords)
    }
    #[getter]
    fn decoder_blocks<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyList>> {
        manifold_sae_list2(py, &self.inner.decoder_blocks)
    }
    /// Typed manifold-crosscoder layout and reports, or `None` for a plain SAE.
    #[getter]
    fn crosscoder(&self, py: Python<'_>) -> PyResult<PyObject> {
        match &self.inner.crosscoder {
            Some(payload) => {
                let value = serde_json::to_value(payload)
                    .map_err(|error| py_value_error(error.to_string()))?;
                json_value_to_py(py, value)
            }
            None => Ok(py.None()),
        }
    }
    /// The per-atom object surface — a list of [`AtomCore`] handles, each read by
    /// attribute (`atom.basis`, `atom.decoder_coefficients`, …), NOT a list of
    /// dicts. This preserves the `SaeManifoldAtomFit` duck-type consumers use.
    #[getter]
    fn atoms<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyList>> {
        let list = PyList::empty(py);
        for atom in &self.inner.atoms {
            list.append(Py::new(
                py,
                AtomCore {
                    inner: atom.clone(),
                },
            )?)?;
        }
        Ok(list)
    }
    #[getter]
    fn duchon_centers<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyList>> {
        let list = PyList::empty(py);
        for center in &self.inner.duchon_centers {
            match center {
                None => list.append(py.None())?,
                Some(m) => list.append(manifold_sae_vec2(py, m)?)?,
            }
        }
        Ok(list)
    }
    #[getter]
    fn fisher_factors<'py>(&self, py: Python<'py>) -> PyResult<Option<Bound<'py, PyArray3<f64>>>> {
        match &self.inner.fisher_factors {
            None => Ok(None),
            Some(v) => Ok(Some(manifold_sae_vec3(py, v)?)),
        }
    }
    /// Atomically validate, pack, and install an output-Fisher shard.  No field
    /// becomes visible until the complete `RowMetric` has been constructed, so
    /// a failed attach leaves the prior model state untouched.
    #[pyo3(signature = (factors, provenance, factor_kind, mass_residual=None))]
    fn attach_fisher<'py>(
        &mut self,
        factors: PyReadonlyArray3<'py, f64>,
        provenance: String,
        factor_kind: String,
        mass_residual: Option<PyReadonlyArray1<'py, f64>>,
    ) -> PyResult<()> {
        let n_rows = self.inner.fitted.len();
        let p_out = self.inner.fitted.first().map_or(0, Vec::len);
        let request = SaeFisherRowMetricRequest::from_tag(
            factors.as_array(),
            n_rows,
            p_out,
            Some(&provenance),
            Some(&factor_kind),
            mass_residual.as_ref().map(|values| values.as_array()),
        )
        .map_err(py_value_error)?;
        let metric = build_sae_fisher_row_metric(request).map_err(py_value_error)?;
        let metric_label =
            gam::terms::sae::manifold::metric_provenance_label(metric.provenance()).to_string();
        let factors_nested = factors
            .as_array()
            .outer_iter()
            .map(|matrix| matrix.rows().into_iter().map(|row| row.to_vec()).collect())
            .collect();
        let mass_nested = mass_residual.map(|values| values.as_array().to_vec());

        self.inner.fisher_factors = Some(factors_nested);
        self.inner.fisher_mass_residual = mass_nested;
        self.inner.fisher_provenance = Some(provenance);
        self.inner.fisher_factor_kind = Some(factor_kind);
        self.inner.metric_provenance = metric_label;
        self.fisher_metric = Some(metric);
        self.fisher_metric_build_count += 1;
        Ok(())
    }

    /// Explicitly remove the Fisher state.  Detach is a separate operation;
    /// `attach_fisher(None)` is not a compatibility alias.
    fn detach_fisher(&mut self) {
        self.inner.fisher_factors = None;
        self.inner.fisher_mass_residual = None;
        self.inner.fisher_provenance = None;
        self.inner.fisher_factor_kind = None;
        self.inner.metric_provenance = "Euclidean".to_string();
        self.fisher_metric = None;
    }

    #[getter]
    fn fisher_metric_build_count(&self) -> usize {
        self.fisher_metric_build_count
    }
    #[getter]
    fn fisher_mass_residual<'py>(&self, py: Python<'py>) -> Option<Bound<'py, PyArray1<f64>>> {
        self.inner
            .fisher_mass_residual
            .as_ref()
            .map(|v| manifold_sae_vec1(py, v))
    }
    #[getter]
    fn selected_log_lambda_smooth<'py>(
        &self,
        py: Python<'py>,
    ) -> Option<Bound<'py, PyArray1<f64>>> {
        self.inner
            .selected_log_lambda_smooth
            .as_ref()
            .map(|v| manifold_sae_vec1(py, v))
    }
    /// Selected per-atom smoothing precisions on their natural scale.
    ///
    /// The fitted payload stores log-precisions because the native outer solve
    /// works in unconstrained coordinates.  Exponentiation belongs here in the
    /// Rust-owned model surface, not in a torch wrapper that would otherwise
    /// duplicate model math.
    #[getter]
    fn selected_lambda_smooth<'py>(&self, py: Python<'py>) -> Option<Bound<'py, PyArray1<f64>>> {
        self.inner
            .selected_log_lambda_smooth
            .as_ref()
            .map(|values| {
                let lambdas = values.iter().map(|value| value.exp()).collect::<Vec<_>>();
                manifold_sae_vec1(py, &lambdas)
            })
    }
    #[getter]
    fn selected_log_ard<'py>(&self, py: Python<'py>) -> PyResult<Option<Bound<'py, PyList>>> {
        match &self.inner.selected_log_ard {
            None => Ok(None),
            Some(v) => {
                let list = PyList::empty(py);
                for row in v {
                    list.append(manifold_sae_vec1(py, row))?;
                }
                Ok(Some(list))
            }
        }
    }

    // --- scalar config getters -------------------------------------------
    #[getter]
    fn schema(&self) -> String {
        self.inner.schema.clone()
    }
    #[getter]
    fn atom_topology(&self) -> String {
        self.inner.atom_topology.clone()
    }
    #[getter]
    fn assignment(&self) -> String {
        self.inner.assignment.clone()
    }
    #[getter]
    fn assignment_label(&self) -> String {
        self.inner.assignment_label.clone()
    }
    #[getter]
    fn metric_provenance(&self) -> String {
        self.inner.metric_provenance.clone()
    }
    #[getter]
    fn fisher_provenance(&self) -> Option<String> {
        self.inner.fisher_provenance.clone()
    }
    #[getter]
    fn fisher_factor_kind(&self) -> Option<String> {
        self.inner.fisher_factor_kind.clone()
    }
    #[getter]
    fn structure_certificate_json(&self) -> Option<String> {
        self.inner.structure_certificate.clone()
    }
    #[getter]
    fn alpha(&self) -> f64 {
        self.inner.alpha
    }
    #[getter]
    fn learnable_alpha(&self) -> bool {
        self.inner.learnable_alpha
    }
    #[getter]
    fn tau(&self) -> f64 {
        self.inner.tau
    }
    #[getter]
    fn sparsity_strength(&self) -> f64 {
        self.inner.sparsity_strength
    }
    #[getter]
    fn smoothness(&self) -> f64 {
        self.inner.smoothness
    }
    #[getter]
    fn learning_rate(&self) -> f64 {
        self.inner.learning_rate
    }
    #[getter]
    fn max_iter(&self) -> i64 {
        self.inner.max_iter
    }
    #[getter]
    fn random_state(&self) -> i64 {
        self.inner.random_state
    }
    #[getter]
    fn top_k(&self) -> Option<i64> {
        self.inner.top_k
    }
    #[getter]
    fn threshold_gate_threshold(&self) -> f64 {
        self.inner.threshold_gate_threshold
    }
    #[getter]
    fn dispersion(&self) -> f64 {
        self.inner.dispersion
    }
    /// Whether the immutable fit installed top-1 OOS projection routing.
    #[getter]
    fn oos_projection_top1(&self) -> bool {
        self.inner.oos_projection_top1
    }
    #[getter]
    fn reconstruction_r2(&self) -> f64 {
        self.inner.reconstruction_r2
    }
    /// The honest penalized-loss score (`None` for closed-form payloads).
    #[getter]
    fn penalized_loss_score(&self) -> Option<f64> {
        self.inner.penalized_loss_score
    }
    /// Terminal custom penalized quasi-Laplace scalar; not normalized LAML/REML/evidence.
    #[getter]
    fn penalized_quasi_laplace_criterion(&self) -> f64 {
        self.inner.penalized_quasi_laplace_criterion
    }
    #[getter]
    fn selected_log_lambda_sparse(&self) -> Option<f64> {
        self.inner.selected_log_lambda_sparse
    }
    /// The number of atoms the fit settled on (= `len(atoms)`).
    #[getter]
    fn chosen_k(&self) -> usize {
        self.inner.atoms.len()
    }

    // --- string / int list getters ---------------------------------------
    #[getter]
    fn atom_topologies(&self) -> Vec<String> {
        self.inner.atom_topologies.clone()
    }
    #[getter]
    fn primitive_names(&self) -> Vec<String> {
        self.inner.primitive_names.clone()
    }
    #[getter]
    fn basis_specs(&self) -> Vec<String> {
        self.inner.basis_specs.clone()
    }
    #[getter]
    fn basis_kinds(&self) -> Vec<String> {
        self.inner.basis_kinds.clone()
    }
    #[getter]
    fn atom_dims(&self) -> Vec<i64> {
        self.inner.atom_dims.clone()
    }
    #[getter]
    fn basis_sizes(&self) -> Vec<i64> {
        self.inner.basis_sizes.clone()
    }
    #[getter]
    fn n_harmonics(&self) -> Vec<i64> {
        self.inner.n_harmonics.clone()
    }

    // --- diagnostic / certificate report-block getters -------------------
    #[getter]
    fn diagnostics(&self, py: Python<'_>) -> PyResult<PyObject> {
        json_value_to_py(py, self.inner.diagnostics.clone())
    }
    #[getter]
    fn solver_plan(&self, py: Python<'_>) -> PyResult<PyObject> {
        manifold_sae_report(py, &self.inner.solver_plan)
    }
    #[getter]
    fn atom_two_lens(&self, py: Python<'_>) -> PyResult<PyObject> {
        manifold_sae_report(py, &self.inner.atom_two_lens)
    }
    #[getter]
    fn residual_gauge(&self, py: Python<'_>) -> PyResult<PyObject> {
        manifold_sae_report(py, &self.inner.residual_gauge)
    }
    #[getter]
    fn incoherence_report(&self, py: Python<'_>) -> PyResult<PyObject> {
        manifold_sae_report(py, &self.inner.incoherence_report)
    }
    #[getter]
    fn curvature_report(&self, py: Python<'_>) -> PyResult<PyObject> {
        manifold_sae_report(py, &self.inner.curvature_report)
    }
    #[getter]
    fn coordinate_fidelity(&self, py: Python<'_>) -> PyResult<PyObject> {
        manifold_sae_report(py, &self.inner.coordinate_fidelity)
    }
    #[getter]
    fn topology_persistence(&self, py: Python<'_>) -> PyResult<PyObject> {
        manifold_sae_report(py, &self.inner.topology_persistence)
    }
    #[getter]
    fn atom_inference_reports(&self, py: Python<'_>) -> PyResult<PyObject> {
        manifold_sae_report(py, &self.inner.atom_inference)
    }
    #[getter]
    fn certificates(&self, py: Python<'_>) -> PyResult<PyObject> {
        manifold_sae_report(py, &self.inner.certificates)
    }
    #[getter]
    fn cotrain(&self, py: Python<'_>) -> PyResult<PyObject> {
        manifold_sae_report(py, &self.inner.cotrain)
    }
    #[getter]
    fn hybrid_split(&self, py: Python<'_>) -> PyResult<PyObject> {
        manifold_sae_report(py, &self.inner.hybrid_split)
    }
    /// The persisted outer-ρ termination verdict/ledger (#2235).
    #[getter]
    fn termination(&self, py: Python<'_>) -> PyResult<PyObject> {
        manifold_sae_report(py, &self.inner.termination)
    }
    /// The persisted per-pass structured-residual alternation diagnostics (#2021).
    #[getter]
    fn structured_residual_diagnostics(&self, py: Python<'_>) -> PyResult<PyObject> {
        json_value_to_py(
            py,
            serde_json::Value::Array(self.inner.structured_residual_diagnostics.clone()),
        )
    }
}

#[cfg(test)]
#[path = "../../tests/src_modules/lib_tests.rs"]
mod tests;
