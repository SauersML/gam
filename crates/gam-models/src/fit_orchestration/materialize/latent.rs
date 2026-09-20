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
            _ => latent_manifold_from_descriptor(value, d, context)?,
        }
    } else if let Some(items) = value.as_array() {
        let mut parts = Vec::with_capacity(items.len());
        for (idx, item) in items.iter().enumerate() {
            let item_context = format!("{context}[{idx}]");
            if item.is_string() || latent_only_object(item) {
                parts.push(parse_latent_manifold(Some(item), 1, &item_context)?.manifold);
            } else {
                let spec = gam_geometry::ManifoldSpec::from_descriptor(item)
                    .map_err(|message| format!("{item_context}: {message}"))?;
                push_latent_descriptor_parts(&spec, &item_context, &mut parts)?;
            }
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
    let spec = gam_geometry::ManifoldSpec::from_descriptor(value)
        .map_err(|message| format!("{context}: {message}"))?;
    retraction_from_spec(&spec, context)
}

/// Whether an array item is an object the latent parser owns (`interval`, `auto`) rather than a manifold descriptor.
fn latent_only_object(item: &JsonValue) -> bool {
    item.get("type")
        .or_else(|| item.get("kind"))
        .and_then(JsonValue::as_str)
        .is_some_and(|kind| kind.eq_ignore_ascii_case("interval") || kind.eq_ignore_ascii_case("auto"))
}

/// The latent manifold a manifold descriptor names for a `d`-dimensional latent. The descriptor is read by
/// [`gam_geometry::ManifoldSpec::from_descriptor`], the one reader of the contract every manifold class emits through
/// `ManifoldSpec::descriptor`, so an emitter and this parser cannot disagree about a field.
fn latent_manifold_from_descriptor(
    value: &JsonValue,
    d: usize,
    context: &str,
) -> Result<LatentManifold, String> {
    let spec = gam_geometry::ManifoldSpec::from_descriptor(value)
        .map_err(|message| format!("{context}: {message}"))?;
    let mut parts = Vec::new();
    push_latent_descriptor_parts(&spec, context, &mut parts)?;
    if parts.iter().all(LatentManifold::is_euclidean) {
        if parts.len() != d {
            return Err(format!(
                "{context} ambient dimension {} does not match latent d={d}",
                parts.len()
            ));
        }
        return Ok(LatentManifold::Euclidean);
    }
    if parts.len() == 1 {
        return Ok(parts.swap_remove(0));
    }
    Ok(LatentManifold::Product(parts))
}

/// Append the latent axes a descriptor spans. Inside a product `Euclidean` is one scalar axis, so a `k`-dimensional
/// Euclidean part spans `k` axes and a `k`-torus spans `k` circles; an intrinsic `S^n` is the unit sphere in `R^(n+1)`.
/// Nested products flatten.
fn push_latent_descriptor_parts(
    spec: &gam_geometry::ManifoldSpec,
    context: &str,
    parts: &mut Vec<LatentManifold>,
) -> Result<(), String> {
    use gam_geometry::ManifoldSpec as Spec;
    let radians = LatentManifold::Circle {
        period: std::f64::consts::TAU,
    };
    match spec {
        Spec::Euclidean(dim) => parts.extend(std::iter::repeat_n(LatentManifold::Euclidean, *dim)),
        Spec::Circle => parts.push(radians),
        Spec::Sphere { intrinsic_dim } => parts.push(LatentManifold::Sphere {
            dim: intrinsic_dim + 1,
        }),
        Spec::Torus { dim } => parts.extend(std::iter::repeat_n(radians, *dim)),
        Spec::Product(inner) => {
            for part in inner {
                push_latent_descriptor_parts(part, context, parts)?;
            }
        }
        Spec::Grassmann { .. } => return Err(not_a_latent_manifold(context, "grassmann")),
        Spec::Stiefel { .. } => return Err(not_a_latent_manifold(context, "stiefel")),
        Spec::Spd { .. } => return Err(not_a_latent_manifold(context, "spd")),
    }
    Ok(())
}

/// The retraction a descriptor names: a `k`-dimensional Euclidean retraction, one circle retraction per torus axis, an
/// intrinsic `S^n` as the unit sphere in `R^(n+1)`, and a product part by part.
fn retraction_from_spec(
    spec: &gam_geometry::ManifoldSpec,
    context: &str,
) -> Result<RetractionKind, String> {
    use gam_geometry::ManifoldSpec as Spec;
    match spec {
        Spec::Euclidean(dim) => Ok(RetractionKind::euclidean(*dim)),
        Spec::Circle => Ok(RetractionKind::Circle),
        Spec::Sphere { intrinsic_dim } => Ok(RetractionKind::Sphere {
            dim: intrinsic_dim + 1,
        }),
        Spec::Torus { dim } => Ok(RetractionKind::Product(ProductRetraction {
            parts: vec![RetractionKind::Circle; *dim],
        })),
        Spec::Product(inner) => inner
            .iter()
            .map(|part| retraction_from_spec(part, context))
            .collect::<Result<Vec<_>, _>>()
            .map(|parts| RetractionKind::Product(ProductRetraction { parts })),
        Spec::Grassmann { .. } => Err(not_a_latent_manifold(context, "grassmann")),
        Spec::Stiefel { .. } => Err(not_a_latent_manifold(context, "stiefel")),
        Spec::Spd { .. } => Err(not_a_latent_manifold(context, "spd")),
    }
}

fn not_a_latent_manifold(context: &str, kind: &str) -> String {
    format!(
        "{context}: a {kind} descriptor is not a latent coordinate manifold; latent coordinates take euclidean, \
         circle, sphere, torus, interval, or a product of them"
    )
}

#[cfg(test)]
mod descriptor_tests {
    use super::*;
    use gam_geometry::ManifoldSpec;

    fn latent_cases() -> Vec<(ManifoldSpec, usize, LatentManifold, RetractionKind)> {
        let radians = LatentManifold::Circle {
            period: std::f64::consts::TAU,
        };
        let circles = RetractionKind::Product(ProductRetraction {
            parts: vec![RetractionKind::Circle; 3],
        });
        vec![
            (
                ManifoldSpec::Euclidean(3),
                3,
                LatentManifold::Euclidean,
                RetractionKind::euclidean(3),
            ),
            (ManifoldSpec::Circle, 1, radians.clone(), RetractionKind::Circle),
            (
                ManifoldSpec::Sphere { intrinsic_dim: 2 },
                3,
                LatentManifold::Sphere { dim: 3 },
                RetractionKind::Sphere { dim: 3 },
            ),
            (
                ManifoldSpec::Torus { dim: 3 },
                3,
                LatentManifold::Product(vec![radians.clone(); 3]),
                circles.clone(),
            ),
            (
                ManifoldSpec::Product(vec![ManifoldSpec::Circle; 3]),
                3,
                LatentManifold::Product(vec![radians.clone(); 3]),
                circles,
            ),
            (
                ManifoldSpec::Product(vec![ManifoldSpec::Circle, ManifoldSpec::Euclidean(2)]),
                3,
                LatentManifold::Product(vec![
                    radians,
                    LatentManifold::Euclidean,
                    LatentManifold::Euclidean,
                ]),
                RetractionKind::Product(ProductRetraction {
                    parts: vec![RetractionKind::Circle, RetractionKind::euclidean(2)],
                }),
            ),
        ]
    }

    /// Every latent-admissible manifold descriptor, as the manifold classes emit it through
    /// `ManifoldSpec::descriptor`, reads through both latent parsers as the manifold and retraction it names. This pins
    /// the emitter/parser drift that refused `ProductManifold(...).to_json()` as a latent manifold (#2627).
    #[test]
    fn every_latent_descriptor_reads_through_both_parsers() {
        for (spec, d, manifold, retraction) in latent_cases() {
            let descriptor = spec.descriptor();
            assert_eq!(
                parse_latent_manifold(Some(&descriptor), d, "latents['t'].manifold")
                    .map(|parsed| (parsed.manifold, parsed.auto)),
                Ok((manifold, false)),
                "manifold descriptor {descriptor} at d={d}"
            );
            assert_eq!(
                parse_retraction_kind(&descriptor, d, "latents['t'].retraction"),
                Ok(retraction),
                "retraction descriptor {descriptor} at d={d}"
            );
        }
    }

    /// A product's parts passed as a bare array read as the same product.
    #[test]
    fn descriptor_parts_array_reads_as_the_product() {
        let parts = serde_json::Value::Array(vec![ManifoldSpec::Circle.descriptor(); 3]);
        let radians = LatentManifold::Circle {
            period: std::f64::consts::TAU,
        };
        assert_eq!(
            parse_latent_manifold(Some(&parts), 3, "latents['t'].manifold").map(|parsed| parsed.manifold),
            Ok(LatentManifold::Product(vec![radians; 3]))
        );
    }

    /// Frame and SPD descriptors are refused by name, and a Euclidean descriptor must span the latent's dimension.
    #[test]
    fn non_latent_descriptors_and_mismatched_dimensions_are_refused() {
        for spec in [
            ManifoldSpec::Grassmann { k: 2, n: 3 },
            ManifoldSpec::Stiefel { k: 1, n: 3 },
            ManifoldSpec::Spd { n: 2 },
        ] {
            let descriptor = spec.descriptor();
            let manifold = parse_latent_manifold(Some(&descriptor), 3, "latents['t'].manifold")
                .map(|parsed| parsed.manifold);
            assert!(
                manifold
                    .as_ref()
                    .is_err_and(|message| message.contains("not a latent coordinate manifold")),
                "{descriptor}: {manifold:?}"
            );
            let retraction = parse_retraction_kind(&descriptor, 3, "latents['t'].retraction");
            assert!(
                retraction
                    .as_ref()
                    .is_err_and(|message| message.contains("not a latent coordinate manifold")),
                "{descriptor}: {retraction:?}"
            );
        }
        let flat = ManifoldSpec::Euclidean(2).descriptor();
        let mismatch = parse_latent_manifold(Some(&flat), 3, "latents['t'].manifold").map(|parsed| parsed.manifold);
        assert_eq!(
            mismatch,
            Err("latents['t'].manifold ambient dimension 2 does not match latent d=3".to_string())
        );
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
                // An unset strength is REML-selected, the same default as the
                // direct latent-fit entry points (`aux_strength=None` ⇒ auto)
                // and the `LatentCoord` contract: a fixed μ is a user choice.
                let strength = match aux.get("strength") {
                    None | Some(JsonValue::Null) => AuxPriorStrength::Auto,
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
        if aux_outcome.is_some()
            && dim_selection
                .as_ref()
                .is_some_and(|dim| dim.init_log_precision.is_some())
        {
            // The behavioral head always composes its own ARD block, seeded by
            // `aux_outcome.init_log_precision`; a `dim_selection` seed would be
            // dropped without a word.
            return Err(format!(
                "latents['{key}'] sets dim_selection.init_log_precision with aux_outcome; the aux_outcome head always carries ARD, seed it with aux_outcome.init_log_precision"
            ));
        }
        // `none` is the only identification mode named explicitly; every
        // other mode is implied by the gauge fields present. Any other value,
        // or `none` next to a gauge field, would otherwise be ignored.
        let explicit_none_mode = match obj
            .get("id_mode")
            .or_else(|| obj.get("mode"))
            .filter(|value| !value.is_null())
        {
            None => false,
            Some(value)
                if value
                    .as_str()
                    .is_some_and(|s| s.eq_ignore_ascii_case("none")) =>
            {
                if aux_prior.is_some() || dim_selection.is_some() || aux_outcome.is_some() {
                    return Err(format!(
                        "latents['{key}'] sets id_mode='none' together with aux_prior, aux_outcome or dim_selection; drop id_mode or the gauge fields"
                    ));
                }
                true
            }
            Some(other) => {
                return Err(format!(
                    "latents['{key}'].id_mode must be 'none' (other modes follow from aux_prior, aux_outcome and dim_selection); got {other}"
                ));
            }
        };
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

#[cfg(test)]
mod aux_prior_strength_default_tests {
    use super::*;

    fn strength_of(aux_prior: JsonValue) -> AuxPriorStrength {
        let payload = serde_json::json!({
            "t": {"n": 3, "d": 1, "aux_prior": aux_prior}
        });
        let specs = parse_latent_specs(Some(&payload)).expect("latent spec parses");
        specs[0]
            .aux_prior
            .as_ref()
            .expect("aux_prior is carried")
            .strength
    }

    /// An omitted (or null) aux-prior strength is REML-selected, matching the
    /// direct latent-fit entry points; an explicit value stays fixed.
    #[test]
    fn unset_aux_prior_strength_is_reml_selected() {
        let u = serde_json::json!([[0.0], [1.0], [2.0]]);
        assert!(matches!(
            strength_of(serde_json::json!({"u": u})),
            AuxPriorStrength::Auto
        ));
        assert!(matches!(
            strength_of(serde_json::json!({"u": u, "strength": null})),
            AuxPriorStrength::Auto
        ));
        assert!(matches!(
            strength_of(serde_json::json!({"u": u, "strength": "auto"})),
            AuxPriorStrength::Auto
        ));
        assert!(matches!(
            strength_of(serde_json::json!({"u": u, "strength": 2.5})),
            AuxPriorStrength::Fixed(mu) if mu == 2.5
        ));
    }
}

#[cfg(test)]
mod latent_gauge_field_tests {
    use super::*;

    fn parse(latent: JsonValue) -> Result<Vec<LatentSpec>, String> {
        parse_latent_specs(Some(&serde_json::json!({ "t": latent })))
    }

    /// Gauge fields that the id-mode resolution would drop are refused by name
    /// instead of being ignored.
    #[test]
    fn dropped_gauge_fields_are_refused() {
        let u = serde_json::json!([[0.0], [1.0], [2.0]]);
        let outcome = serde_json::json!({"family": "binomial", "y": [0.0, 1.0, 0.0]});

        let seeded_ard_with_head = parse(serde_json::json!({
            "n": 3, "d": 1, "aux_outcome": outcome,
            "dim_selection": {"init_log_precision": [0.5]}
        }));
        assert!(
            seeded_ard_with_head
                .as_ref()
                .is_err_and(|error| error.contains("aux_outcome.init_log_precision")),
            "{:?}",
            seeded_ard_with_head.err()
        );

        let none_with_prior = parse(serde_json::json!({
            "n": 3, "d": 1, "aux_prior": {"u": u}, "id_mode": "none"
        }));
        assert!(none_with_prior.is_err_and(|error| error.contains("id_mode='none'")));

        let unknown_mode = parse(serde_json::json!({
            "n": 3, "d": 1, "aux_prior": {"u": u}, "id_mode": "isometry"
        }));
        assert!(unknown_mode.is_err_and(|error| error.contains("id_mode must be 'none'")));
    }

    /// The accepted spellings still parse: `none` alone, ARD switched on next to
    /// the head (which carries ARD anyway), and the head's own seed.
    #[test]
    fn consistent_gauge_fields_still_parse() {
        let outcome = serde_json::json!({
            "family": "binomial", "y": [0.0, 1.0, 0.0], "init_log_precision": [0.5]
        });
        let none = parse(serde_json::json!({"n": 3, "d": 1, "id_mode": "none"}))
            .expect("id_mode='none' alone parses");
        assert!(none[0].explicit_none_mode);
        parse(serde_json::json!({
            "n": 3, "d": 1, "aux_outcome": outcome, "dim_selection": true
        }))
        .expect("dim_selection=true beside aux_outcome parses");
    }
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
            let sd = var.sqrt();
            // A response whose spread sits inside its mean's rounding band
            // `γ_{n+1}·max|y|` is constant to working precision: it has no axis to
            // standardize along, so the leading coordinate starts at zero.
            let magnitude = y.iter().fold(0.0_f64, |acc, v| acc.max(v.abs()));
            let resolvable =
                sd > gam_linalg::roundoff::accumulation_growth(y.len() + 1) * magnitude;
            for n in 0..spec.n {
                out[[n, 0]] = if resolvable { (y[n] - mean) / sd } else { 0.0 };
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
    if specs.is_empty() {
        return Ok(None);
    }
    if specs.len() != 1 {
        return Err(
            "standard latent-coordinate REML currently accepts exactly one latent smooth term"
                .to_string(),
        );
    }
    let Some(spec) = specs.into_iter().next() else {
        return Err(
            "standard latent-coordinate REML found no latent smooth term to materialize"
                .to_string(),
        );
    };
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
    // This route evaluates the analytic penalties on the latent coordinates alone
    // and installs no decoder jets, so an isometry penalty, a function of the
    // decoder Jacobian, has no value here and is refused by name.
    analytic_penalties
        .isometry_evaluation_precondition(
            gam_terms::IsometryEvaluationOrder::Hessian,
            spec.n * spec.d,
        )
        .map_err(|reason| {
            format!(
                "latent '{}': the latent-coordinate REML route supplies no decoder jets for an \
                 isometry penalty ({reason})",
                spec.target
            )
        })?;

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
