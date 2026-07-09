use super::*;

#[derive(Clone)]
enum LatentInitSpec {
    Pca,
    Random,
    Explicit(Array2<f64>),
}

#[derive(Clone)]
struct LatentAuxPriorSpec {
    u: Array2<f64>,
    family: AuxPriorFamily,
    strength: AuxPriorStrength,
}

#[derive(Clone)]
struct LatentDimSelectionSpec {
    init_log_precision: Option<Array1<f64>>,
}

#[derive(Clone)]
struct LatentAuxOutcomeSpec {
    family: gam_terms::decoders::behavioral_head::AuxOutcomeFamily,
    /// Behavioral labels, length `n`. Binomial: 0/1; Multinomial: class index.
    y: Array1<f64>,
    /// Optional per-row head weight (semi-supervised); `None` ⇒ all rows
    /// labeled with unit weight. `0.0` on a row excludes it from the head
    /// channel — the missing-label seam.
    row_weights: Option<Array1<f64>>,
    /// ARD log-precision seed composed with the head (length `d`).
    init_log_precision: Option<Array1<f64>>,
}

#[derive(Clone)]
struct LatentManifoldSpec {
    manifold: LatentManifold,
    auto: bool,
}

#[derive(Clone)]
struct LatentSpec {
    target: String,
    n: usize,
    d: usize,
    init: LatentInitSpec,
    manifold: LatentManifoldSpec,
    retraction_registry: LatentRetractionRegistry,
    aux_prior: Option<LatentAuxPriorSpec>,
    dim_selection: Option<LatentDimSelectionSpec>,
    aux_outcome: Option<LatentAuxOutcomeSpec>,
    explicit_none_mode: bool,
}

fn json_array2(value: &JsonValue, context: &str) -> Result<Array2<f64>, String> {
    let rows = value
        .as_array()
        .ok_or_else(|| format!("{context} must be a two-dimensional numeric array"))?;
    let n = rows.len();
    let first = rows
        .first()
        .and_then(|row| row.as_array())
        .ok_or_else(|| format!("{context} must contain array rows"))?;
    let d = first.len();
    let mut out = Array2::<f64>::zeros((n, d));
    for (i, row_value) in rows.iter().enumerate() {
        let row = row_value
            .as_array()
            .ok_or_else(|| format!("{context} row {i} must be an array"))?;
        if row.len() != d {
            return Err(format!(
                "{context} row {i} has length {}, expected {d}",
                row.len()
            ));
        }
        for (j, cell) in row.iter().enumerate() {
            let value = cell
                .as_f64()
                .ok_or_else(|| format!("{context}[{i}][{j}] must be a finite number"))?;
            if !value.is_finite() {
                return Err(format!("{context}[{i}][{j}] must be finite"));
            }
            out[[i, j]] = value;
        }
    }
    Ok(out)
}

fn json_array1(value: &JsonValue, context: &str) -> Result<Array1<f64>, String> {
    let values = value
        .as_array()
        .ok_or_else(|| format!("{context} must be a numeric array"))?;
    let mut out = Array1::<f64>::zeros(values.len());
    for (idx, cell) in values.iter().enumerate() {
        let value = cell
            .as_f64()
            .ok_or_else(|| format!("{context}[{idx}] must be a finite number"))?;
        if !value.is_finite() {
            return Err(format!("{context}[{idx}] must be finite"));
        }
        out[idx] = value;
    }
    Ok(out)
}

fn parse_latent_manifold(
    value: Option<&JsonValue>,
    d: usize,
    context: &str,
) -> Result<LatentManifoldSpec, String> {
    let Some(value) = value.filter(|value| !value.is_null()) else {
        return Ok(LatentManifoldSpec {
            manifold: LatentManifold::Euclidean,
            auto: true,
        });
    };
    if value
        .as_str()
        .is_some_and(|s| s.eq_ignore_ascii_case("auto"))
    {
        return Ok(LatentManifoldSpec {
            manifold: LatentManifold::Euclidean,
            auto: true,
        });
    }
    let parse_named = |name: &str| -> Result<LatentManifold, String> {
        match name.to_ascii_lowercase().as_str() {
            "euclidean" | "r" | "real" => Ok(LatentManifold::Euclidean),
            "circle" | "s1" | "periodic" => {
                let radians = LatentManifold::Circle {
                    period: std::f64::consts::TAU,
                };
                if d == 1 {
                    Ok(radians)
                } else {
                    Ok(LatentManifold::Product(
                        (0..d).map(|_| radians.clone()).collect(),
                    ))
                }
            }
            "sphere" | "sn" => Ok(LatentManifold::Sphere { dim: d }),
            "torus" => Ok(LatentManifold::Product(
                (0..d)
                    .map(|_| LatentManifold::Circle {
                        period: std::f64::consts::TAU,
                    })
                    .collect(),
            )),
            "cylinder" => {
                if d < 2 {
                    return Err(format!("{context}='cylinder' requires d >= 2"));
                }
                let mut parts = Vec::with_capacity(d);
                parts.push(LatentManifold::Circle {
                    period: std::f64::consts::TAU,
                });
                for _ in 1..d {
                    parts.push(LatentManifold::Euclidean);
                }
                Ok(LatentManifold::Product(parts))
            }
            other => Err(format!(
                "{context} must be 'auto', 'euclidean', 'circle', 'sphere', 'torus', or 'cylinder'; got '{other}'"
            )),
        }
    };
    let manifold = if let Some(name) = value.as_str() {
        parse_named(name)?
    } else if let Some(obj) = value.as_object() {
        let kind = obj
            .get("type")
            .or_else(|| obj.get("kind"))
            .and_then(JsonValue::as_str)
            .unwrap_or("euclidean");
        match kind.to_ascii_lowercase().as_str() {
            "auto" => {
                return Ok(LatentManifoldSpec {
                    manifold: LatentManifold::Euclidean,
                    auto: true,
                });
            }
            "interval" => {
                let lo = obj
                    .get("lo")
                    .or_else(|| obj.get("min"))
                    .and_then(JsonValue::as_f64)
                    .ok_or_else(|| format!("{context}.lo is required for interval"))?;
                let hi = obj
                    .get("hi")
                    .or_else(|| obj.get("max"))
                    .and_then(JsonValue::as_f64)
                    .ok_or_else(|| format!("{context}.hi is required for interval"))?;
                if !(lo.is_finite() && hi.is_finite() && lo < hi) {
                    return Err(format!("{context} interval requires finite lo < hi"));
                }
                LatentManifold::Interval { lo, hi }
            }
            other => parse_named(other)?,
        }
    } else if let Some(items) = value.as_array() {
        let mut parts = Vec::with_capacity(items.len());
        for (idx, item) in items.iter().enumerate() {
            parts
                .push(parse_latent_manifold(Some(item), 1, &format!("{context}[{idx}]"))?.manifold);
        }
        LatentManifold::Product(parts)
    } else {
        return Err(format!(
            "{context} must be a string, object, or product array"
        ));
    };
    if manifold.ambient_dim(d) != d {
        return Err(format!(
            "{context} ambient dimension {} does not match latent d={d}",
            manifold.ambient_dim(d)
        ));
    }
    Ok(LatentManifoldSpec {
        manifold,
        auto: false,
    })
}

fn parse_retraction_kind(
    value: &JsonValue,
    fallback_dim: usize,
    context: &str,
) -> Result<RetractionKind, String> {
    let parse_named = |name: &str| -> Result<RetractionKind, String> {
        match name.to_ascii_lowercase().as_str() {
            "euclidean" | "r" | "real" => Ok(RetractionKind::euclidean(fallback_dim)),
            "circle" | "s1" | "periodic" => {
                if fallback_dim == 1 {
                    Ok(RetractionKind::Circle)
                } else {
                    Ok(RetractionKind::Product(ProductRetraction {
                        parts: (0..fallback_dim).map(|_| RetractionKind::Circle).collect(),
                    }))
                }
            }
            "sphere" | "sn" => Ok(RetractionKind::Sphere { dim: fallback_dim }),
            other => Err(format!(
                "{context} must be 'euclidean', 'circle', 'sphere', or a product; got '{other}'"
            )),
        }
    };
    if let Some(name) = value.as_str() {
        return parse_named(name);
    }
    if let Some(items) = value.as_array() {
        let mut parts = Vec::with_capacity(items.len());
        for (idx, item) in items.iter().enumerate() {
            parts.push(parse_retraction_kind(
                item,
                1,
                &format!("{context}[{idx}]"),
            )?);
        }
        return Ok(RetractionKind::Product(ProductRetraction { parts }));
    }
    let obj = value
        .as_object()
        .ok_or_else(|| format!("{context} must be a string, object, or product array"))?;
    let kind = obj
        .get("type")
        .or_else(|| obj.get("kind"))
        .and_then(JsonValue::as_str)
        .unwrap_or("euclidean");
    match kind.to_ascii_lowercase().as_str() {
        "euclidean" | "r" | "real" => {
            let dim = obj
                .get("dim")
                .or_else(|| obj.get("d"))
                .and_then(JsonValue::as_u64)
                .map_or(fallback_dim, |value| value as usize);
            if dim == 0 {
                return Err(format!("{context}.dim must be positive"));
            }
            Ok(RetractionKind::euclidean(dim))
        }
        "circle" | "s1" | "periodic" => Ok(RetractionKind::Circle),
        "sphere" | "sn" => {
            let dim = obj
                .get("dim")
                .or_else(|| obj.get("d"))
                .and_then(JsonValue::as_u64)
                .map_or(fallback_dim, |value| value as usize);
            if dim == 0 {
                return Err(format!("{context}.dim must be positive"));
            }
            Ok(RetractionKind::Sphere { dim })
        }
        "product" => {
            let items = obj
                .get("parts")
                .or_else(|| obj.get("components"))
                .and_then(JsonValue::as_array)
                .ok_or_else(|| format!("{context}.parts is required for product retraction"))?;
            let mut parts = Vec::with_capacity(items.len());
            for (idx, item) in items.iter().enumerate() {
                parts.push(parse_retraction_kind(
                    item,
                    1,
                    &format!("{context}.parts[{idx}]"),
                )?);
            }
            Ok(RetractionKind::Product(ProductRetraction { parts }))
        }
        other => parse_named(other),
    }
}

fn parse_latent_retraction(
    value: Option<&JsonValue>,
    d: usize,
    context: &str,
) -> Result<LatentRetractionRegistry, String> {
    let Some(value) = value.filter(|value| !value.is_null()) else {
        return Ok(LatentRetractionRegistry::all_euclidean());
    };
    let kind = parse_retraction_kind(value, d, context)?;
    let registry = LatentRetractionRegistry::new(kind);
    registry.validate_dim(d, context)?;
    Ok(registry)
}

fn parse_latent_specs(payload: Option<&JsonValue>) -> Result<Vec<LatentSpec>, String> {
    let Some(payload) = payload.filter(|value| !value.is_null()) else {
        return Ok(Vec::new());
    };
    let map = payload
        .as_object()
        .ok_or_else(|| "latents must be a JSON object keyed by formula symbol".to_string())?;
    let mut specs = Vec::with_capacity(map.len());
    for (key, raw) in map {
        let obj = raw
            .as_object()
            .ok_or_else(|| format!("latents['{key}'] must be an object"))?;
        let target = obj
            .get("name")
            .and_then(JsonValue::as_str)
            .unwrap_or(key)
            .to_string();
        let n = obj
            .get("n")
            .and_then(JsonValue::as_u64)
            .ok_or_else(|| format!("latents['{key}'].n is required"))? as usize;
        let d = obj
            .get("d")
            .and_then(JsonValue::as_u64)
            .ok_or_else(|| format!("latents['{key}'].d is required"))? as usize;
        if n == 0 || d == 0 {
            return Err(format!("latents['{key}'] requires positive n and d"));
        }
        let manifold = parse_latent_manifold(
            obj.get("manifold"),
            d,
            &format!("latents['{key}'].manifold"),
        )?;
        let retraction_registry = parse_latent_retraction(
            obj.get("retraction"),
            d,
            &format!("latents['{key}'].retraction"),
        )?;
        let init = match obj.get("init") {
            None => LatentInitSpec::Pca,
            Some(value)
                if value
                    .as_str()
                    .is_some_and(|s| s.eq_ignore_ascii_case("pca")) =>
            {
                LatentInitSpec::Pca
            }
            Some(value)
                if value
                    .as_str()
                    .is_some_and(|s| s.eq_ignore_ascii_case("random")) =>
            {
                LatentInitSpec::Random
            }
            Some(value) => {
                LatentInitSpec::Explicit(json_array2(value, &format!("latents['{key}'].init"))?)
            }
        };
        let aux_prior = match obj.get("aux_prior").filter(|value| !value.is_null()) {
            None => None,
            Some(value) => {
                let aux = value
                    .as_object()
                    .ok_or_else(|| format!("latents['{key}'].aux_prior must be an object"))?;
                let u = json_array2(
                    aux.get("u")
                        .ok_or_else(|| format!("latents['{key}'].aux_prior.u is required"))?,
                    &format!("latents['{key}'].aux_prior.u"),
                )?;
                let family = match aux
                    .get("family")
                    .and_then(JsonValue::as_str)
                    .unwrap_or("ridge")
                    .to_ascii_lowercase()
                    .as_str()
                {
                    "ridge" => AuxPriorFamily::Ridge,
                    "linear" => AuxPriorFamily::Linear,
                    other => {
                        return Err(format!(
                            "latents['{key}'].aux_prior.family must be 'ridge' or 'linear', got '{other}'"
                        ));
                    }
                };
                let strength = match aux.get("strength") {
                    None => AuxPriorStrength::Fixed(1.0),
                    Some(value)
                        if value
                            .as_str()
                            .is_some_and(|s| s.eq_ignore_ascii_case("auto")) =>
                    {
                        AuxPriorStrength::Auto
                    }
                    Some(value) => {
                        let mu = value.as_f64().ok_or_else(|| {
                            format!(
                                "latents['{key}'].aux_prior.strength must be positive or 'auto'"
                            )
                        })?;
                        if !mu.is_finite() || mu <= 0.0 {
                            return Err(format!(
                                "latents['{key}'].aux_prior.strength must be positive"
                            ));
                        }
                        AuxPriorStrength::Fixed(mu)
                    }
                };
                Some(LatentAuxPriorSpec {
                    u,
                    family,
                    strength,
                })
            }
        };
        let dim_selection = match obj.get("dim_selection") {
            None | Some(JsonValue::Bool(false)) => None,
            Some(JsonValue::Bool(true)) => Some(LatentDimSelectionSpec {
                init_log_precision: None,
            }),
            Some(value) => {
                let dim = value.as_object().ok_or_else(|| {
                    format!("latents['{key}'].dim_selection must be a bool or object")
                })?;
                let init_log_precision = dim
                    .get("init_log_precision")
                    .map(|value| {
                        json_array1(
                            value,
                            &format!("latents['{key}'].dim_selection.init_log_precision"),
                        )
                    })
                    .transpose()?;
                Some(LatentDimSelectionSpec { init_log_precision })
            }
        };
        let aux_outcome = match obj.get("aux_outcome").filter(|value| !value.is_null()) {
            None => None,
            Some(value) => {
                use gam_terms::decoders::behavioral_head::AuxOutcomeFamily;
                let ao = value
                    .as_object()
                    .ok_or_else(|| format!("latents['{key}'].aux_outcome must be an object"))?;
                let family = match ao
                    .get("family")
                    .and_then(JsonValue::as_str)
                    .unwrap_or("binomial")
                    .to_ascii_lowercase()
                    .as_str()
                {
                    "binomial" => AuxOutcomeFamily::Binomial,
                    "multinomial" => {
                        let n_classes = ao
                            .get("n_classes")
                            .and_then(JsonValue::as_u64)
                            .ok_or_else(|| {
                                format!(
                                    "latents['{key}'].aux_outcome.n_classes is required for multinomial"
                                )
                            })? as usize;
                        AuxOutcomeFamily::Multinomial { n_classes }
                    }
                    other => {
                        return Err(format!(
                            "latents['{key}'].aux_outcome.family must be 'binomial' or 'multinomial', got '{other}'"
                        ));
                    }
                };
                let y = json_array1(
                    ao.get("y")
                        .ok_or_else(|| format!("latents['{key}'].aux_outcome.y is required"))?,
                    &format!("latents['{key}'].aux_outcome.y"),
                )?;
                if y.len() != n {
                    return Err(format!(
                        "latents['{key}'].aux_outcome.y has length {}, expected n = {n}",
                        y.len()
                    ));
                }
                let row_weights = ao
                    .get("row_weights")
                    .filter(|value| !value.is_null())
                    .map(|value| {
                        json_array1(value, &format!("latents['{key}'].aux_outcome.row_weights"))
                    })
                    .transpose()?;
                if let Some(w) = row_weights.as_ref()
                    && w.len() != n
                {
                    return Err(format!(
                        "latents['{key}'].aux_outcome.row_weights has length {}, expected n = {n}",
                        w.len()
                    ));
                }
                let init_log_precision = ao
                    .get("init_log_precision")
                    .map(|value| {
                        json_array1(
                            value,
                            &format!("latents['{key}'].aux_outcome.init_log_precision"),
                        )
                    })
                    .transpose()?;
                Some(LatentAuxOutcomeSpec {
                    family,
                    y,
                    row_weights,
                    init_log_precision,
                })
            }
        };
        if dim_selection.is_some() && aux_prior.is_none() && aux_outcome.is_none() {
            return Err(format!(
                "latents['{key}'] uses dim_selection without aux_prior or aux_outcome; ARD alone is not an identifiable latent-coordinate gauge"
            ));
        }
        if aux_outcome.is_some() && aux_prior.is_some() {
            return Err(format!(
                "latents['{key}'] specifies both aux_prior and aux_outcome; the auxiliary signal is either a prior (gauge-pin covariate) or a modeled outcome (behavioral head), not both"
            ));
        }
        let explicit_none_mode = obj
            .get("id_mode")
            .or_else(|| obj.get("mode"))
            .and_then(JsonValue::as_str)
            .is_some_and(|s| s.eq_ignore_ascii_case("none"));
        if aux_prior.is_none()
            && dim_selection.is_none()
            && aux_outcome.is_none()
            && !explicit_none_mode
        {
            return Err(format!(
                "latents['{key}'] requires aux_prior or aux_outcome for identifiable joint REML; pass id_mode='none' only when a separate gauge fix is supplied"
            ));
        }
        specs.push(LatentSpec {
            target,
            n,
            d,
            init,
            manifold,
            retraction_registry,
            aux_prior,
            dim_selection,
            aux_outcome,
            explicit_none_mode,
        });
    }
    Ok(specs)
}

fn deterministic_unit(seed: &mut u64) -> f64 {
    *seed = seed
        .wrapping_mul(6364136223846793005)
        .wrapping_add(1442695040888963407);
    ((*seed >> 11) as f64) * (1.0 / ((1_u64 << 53) as f64))
}

fn initial_latent_matrix(spec: &LatentSpec, y: ArrayView1<'_, f64>) -> Result<Array2<f64>, String> {
    match &spec.init {
        LatentInitSpec::Explicit(matrix) => {
            if matrix.nrows() != spec.n || matrix.ncols() != spec.d {
                return Err(format!(
                    "latent '{}' explicit init has shape {}x{}, expected {}x{}",
                    spec.target,
                    matrix.nrows(),
                    matrix.ncols(),
                    spec.n,
                    spec.d
                ));
            }
            Ok(matrix.clone())
        }
        LatentInitSpec::Random => {
            let mut seed = 0x9E3779B97F4A7C15_u64 ^ ((spec.n as u64) << 32) ^ spec.d as u64;
            let mut out = Array2::<f64>::zeros((spec.n, spec.d));
            for value in out.iter_mut() {
                *value = deterministic_unit(&mut seed);
            }
            Ok(out)
        }
        LatentInitSpec::Pca => {
            let mut out = Array2::<f64>::zeros((spec.n, spec.d));
            let mean = y.iter().sum::<f64>() / y.len().max(1) as f64;
            let var = y
                .iter()
                .map(|v| {
                    let centered = *v - mean;
                    centered * centered
                })
                .sum::<f64>()
                / y.len().max(1) as f64;
            let sd = var.sqrt().max(1e-12);
            for n in 0..spec.n {
                out[[n, 0]] = (y[n] - mean) / sd;
            }
            if spec.d > 1 {
                let mut seed = 0xD1B54A32D192ED03_u64 ^ ((spec.n as u64) << 16) ^ spec.d as u64;
                for n in 0..spec.n {
                    for axis in 1..spec.d {
                        out[[n, axis]] = deterministic_unit(&mut seed) - 0.5;
                    }
                }
            }
            Ok(out)
        }
    }
}

fn latent_id_mode(spec: &LatentSpec) -> Result<LatentIdMode, String> {
    if let Some(ao) = spec.aux_outcome.as_ref() {
        use gam_terms::decoders::behavioral_head::BehavioralHead;
        if let Some(init) = ao.init_log_precision.as_ref()
            && init.len() != spec.d
        {
            return Err(format!(
                "latent '{}' aux_outcome.init_log_precision has length {}, expected {}",
                spec.target,
                init.len(),
                spec.d
            ));
        }
        let head = match ao.row_weights.as_ref() {
            Some(w) => BehavioralHead::new(ao.family, ao.y.clone(), w.clone()),
            None => BehavioralHead::fully_supervised(ao.family, ao.y.clone()),
        }
        .map_err(|e| format!("latent '{}' aux_outcome head: {e}", spec.target))?;
        return Ok(LatentIdMode::AuxOutcome {
            head,
            init_log_precision: ao.init_log_precision.clone(),
        });
    }
    match (&spec.aux_prior, &spec.dim_selection) {
        (Some(aux), Some(dim)) => {
            if let Some(init) = dim.init_log_precision.as_ref()
                && init.len() != spec.d
            {
                return Err(format!(
                    "latent '{}' dim_selection.init_log_precision has length {}, expected {}",
                    spec.target,
                    init.len(),
                    spec.d
                ));
            }
            Ok(LatentIdMode::AuxPriorDimSelection {
                u: aux.u.clone(),
                family: aux.family,
                strength: aux.strength,
                init_log_precision: dim.init_log_precision.clone(),
            })
        }
        (Some(aux), None) => Ok(LatentIdMode::AuxPrior {
            u: aux.u.clone(),
            family: aux.family,
            strength: aux.strength,
        }),
        (None, None) if spec.explicit_none_mode => Ok(LatentIdMode::None),
        (None, None) => Err(format!(
            "latent '{}' requires aux_prior for identifiable joint REML; pass id_mode='none' only when a separate gauge fix is supplied",
            spec.target
        )),
        (None, Some(_)) => Err(format!(
            "latent '{}' dim_selection requires aux_prior for identifiability",
            spec.target
        )),
    }
}

pub(super) fn prepare_standard_latent_coord(
    parsed: &ParsedFormula,
    data: &Dataset,
    y: ArrayView1<'_, f64>,
    config: &FitConfig,
) -> Result<Option<(Dataset, ParsedFormula, StandardLatentCoordConfig)>, String> {
    let specs = parse_latent_specs(config.latents.as_ref())?;
    let analytic_penalties = descriptors::build_analytic_penalty_registry_from_descriptors(
        config.latents.as_ref(),
        config.analytic_penalties.as_ref(),
    )?;
    if config.topology_auto_selector.is_some() && specs.is_empty() {
        return Err(
            "TopologyAutoSelector requires a Smooth with latent coords; pass latents={...}"
                .to_string(),
        );
    }
    if specs.is_empty() {
        return Ok(None);
    }
    if specs.len() != 1 {
        return Err(
            "standard latent-coordinate REML currently accepts exactly one latent smooth term"
                .to_string(),
        );
    }
    let spec = specs.into_iter().next().unwrap();
    if let Some(selector) = config.topology_auto_selector.as_ref()
        && let Some(requested) = selector.latent.as_ref()
        && requested != &spec.target
    {
        return Err(format!(
            "TopologyAutoSelector requested latent {requested:?}, but the formula path materialized latent {:?}",
            spec.target
        ));
    }
    if spec.n != data.values.nrows() || spec.n != y.len() {
        return Err(format!(
            "latent '{}' row count {} does not match data rows {}",
            spec.target,
            spec.n,
            data.values.nrows()
        ));
    }
    if let Some(aux) = spec.aux_prior.as_ref()
        && aux.u.nrows() != spec.n
    {
        return Err(format!(
            "latent '{}' aux_prior.u has {} rows, expected {}",
            spec.target,
            aux.u.nrows(),
            spec.n
        ));
    }

    let matrix = initial_latent_matrix(&spec, y)?;
    let id_mode = latent_id_mode(&spec)?;
    let latent_values = Arc::new(LatentCoordValues::from_matrix_with_manifold_and_retraction(
        matrix.view(),
        id_mode,
        spec.manifold.manifold.clone(),
        spec.retraction_registry.clone(),
    ));

    let base_cols = data.values.ncols();
    let mut values = Array2::<f64>::zeros((data.values.nrows(), base_cols + spec.d));
    values.slice_mut(s![.., ..base_cols]).assign(&data.values);
    let mut headers = data.headers.clone();
    let mut columns = data.schema.columns.clone();
    let mut column_kinds = data.column_kinds.clone();
    let mut synthetic_vars = Vec::with_capacity(spec.d);
    let mut feature_cols = Vec::with_capacity(spec.d);
    for axis in 0..spec.d {
        let name = format!("{}__latent{}", spec.target, axis);
        let col = base_cols + axis;
        values.column_mut(col).assign(&matrix.column(axis));
        headers.push(name.clone());
        columns.push(SchemaColumn {
            name: name.clone(),
            kind: ColumnKindTag::Continuous,
            levels: Vec::new(),
        });
        column_kinds.push(ColumnKindTag::Continuous);
        synthetic_vars.push(name);
        feature_cols.push(col);
    }
    let augmented = Dataset {
        headers,
        values,
        schema: DataSchema { columns },
        column_kinds,
    };

    let mut rewritten = parsed.clone();
    let mut matched = false;
    for term in &mut rewritten.terms {
        if let ParsedTerm::Smooth { vars, .. } = term
            && vars.len() == 1
            && vars[0] == spec.target
        {
            *vars = synthetic_vars.clone();
            matched = true;
        }
    }
    if !matched {
        return Err(format!(
            "latents provided '{}' but no formula smooth term s({}, ...) was found",
            spec.target, spec.target
        ));
    }

    Ok(Some((
        augmented,
        rewritten,
        StandardLatentCoordConfig {
            values: latent_values,
            term_index: gam_problem::SmoothTermIdx::placeholder(),
            feature_cols,
            manifold: spec.manifold.manifold,
            manifold_auto: spec.manifold.auto,
            retraction_registry: spec.retraction_registry,
            analytic_penalties: (!analytic_penalties.penalties.is_empty())
                .then(|| Arc::new(analytic_penalties)),
        },
    )))
}

pub(super) fn smooth_basis_feature_cols_for_latent(
    basis: &gam_terms::smooth::SmoothBasisSpec,
) -> Option<Vec<usize>> {
    match basis {
        gam_terms::smooth::SmoothBasisSpec::BSpline1D { feature_col, .. } => {
            Some(vec![*feature_col])
        }
        gam_terms::smooth::SmoothBasisSpec::ThinPlate { feature_cols, .. }
        | gam_terms::smooth::SmoothBasisSpec::Sphere { feature_cols, .. }
        | gam_terms::smooth::SmoothBasisSpec::ConstantCurvature { feature_cols, .. }
        | gam_terms::smooth::SmoothBasisSpec::Matern { feature_cols, .. }
        | gam_terms::smooth::SmoothBasisSpec::MeasureJet { feature_cols, .. }
        | gam_terms::smooth::SmoothBasisSpec::Duchon { feature_cols, .. }
        | gam_terms::smooth::SmoothBasisSpec::Pca { feature_cols, .. }
        | gam_terms::smooth::SmoothBasisSpec::TensorBSpline { feature_cols, .. } => {
            Some(feature_cols.clone())
        }
        gam_terms::smooth::SmoothBasisSpec::BySmooth { smooth, .. } => {
            smooth_basis_feature_cols_for_latent(smooth)
        }
        gam_terms::smooth::SmoothBasisSpec::ByVariable { inner, .. }
        | gam_terms::smooth::SmoothBasisSpec::FactorSumToZero { inner, .. } => {
            smooth_basis_feature_cols_for_latent(inner)
        }
        gam_terms::smooth::SmoothBasisSpec::FactorSmooth { .. } => None,
    }
}

pub(super) fn natural_latent_manifold_for_basis(
    basis: &gam_terms::smooth::SmoothBasisSpec,
    d: usize,
) -> LatentManifold {
    match basis {
        gam_terms::smooth::SmoothBasisSpec::BSpline1D { spec, .. } => {
            if let gam_terms::basis::BSplineKnotSpec::PeriodicUniform { data_range, .. } =
                &spec.knotspec
            {
                LatentManifold::Circle {
                    period: data_range.1 - data_range.0,
                }
            } else {
                LatentManifold::Euclidean
            }
        }
        gam_terms::smooth::SmoothBasisSpec::Sphere { .. } => LatentManifold::Sphere { dim: d },
        gam_terms::smooth::SmoothBasisSpec::Duchon { spec, .. }
            if spec.periodic.is_some() && d == 1 =>
        {
            let period = spec
                .periodic
                .as_ref()
                .and_then(|v| v.first().copied().flatten())
                .unwrap_or(std::f64::consts::TAU);
            LatentManifold::Circle { period }
        }
        gam_terms::smooth::SmoothBasisSpec::TensorBSpline { spec, .. } => {
            let parts: Vec<LatentManifold> = spec
                .marginalspecs
                .iter()
                .map(|margin| {
                    if let gam_terms::basis::BSplineKnotSpec::PeriodicUniform { data_range, .. } =
                        &margin.knotspec
                    {
                        LatentManifold::Circle {
                            period: data_range.1 - data_range.0,
                        }
                    } else {
                        LatentManifold::Euclidean
                    }
                })
                .collect();
            if parts.iter().all(|part| part.is_euclidean()) {
                LatentManifold::Euclidean
            } else {
                LatentManifold::Product(parts)
            }
        }
        gam_terms::smooth::SmoothBasisSpec::BySmooth { smooth, .. } => {
            natural_latent_manifold_for_basis(smooth, d)
        }
        gam_terms::smooth::SmoothBasisSpec::ByVariable { inner, .. }
        | gam_terms::smooth::SmoothBasisSpec::FactorSumToZero { inner, .. } => {
            natural_latent_manifold_for_basis(inner, d)
        }
        gam_terms::smooth::SmoothBasisSpec::ThinPlate { .. }
        // ConstantCurvature: the chart coordinates are Euclidean-valued (any
        // finite point for κ ≥ 0; the latent optimizer's chart-validity is the
        // term's own concern), so the latent retraction stays Euclidean. A
        // κ-aware latent seed/retraction is part of the later ψ-channel stage.
        | gam_terms::smooth::SmoothBasisSpec::ConstantCurvature { .. }
        | gam_terms::smooth::SmoothBasisSpec::Matern { .. }
        | gam_terms::smooth::SmoothBasisSpec::MeasureJet { .. }
        | gam_terms::smooth::SmoothBasisSpec::Duchon { .. }
        | gam_terms::smooth::SmoothBasisSpec::Pca { .. }
        | gam_terms::smooth::SmoothBasisSpec::FactorSmooth { .. } => LatentManifold::Euclidean,
    }
}
