//! Fit structural proposals in the autonomous native LM on clean AND intervened logits.
//! EXPORT SETTINGS.json FRESH_OUT host|cuda [HELDOUT_EXPORT] [--profile]. Training proposal search, not acceptance.
//! --profile synchronizes the device around every named stage and writes FRESH_OUT/PROFILE.json.
use gam_gpu::{tensor::Device, GpuPolicy};
use gam_mpd::{
    acceptance::{structural_cost, CostCache},
    artifact::Artifact,
    canonical_artifact::CanonicalArtifactCache,
    engine::sha256,
    composed_rule_search::{self, Grammar, UseSpec},
    device_program::DeviceProgram,
    down_edit_family::{self, Direction, Family as DownFamily},
    import::import_language_model,
    intervention_program::{self, Control, ControlValue},
    native_parameter_edit,
    operator_program::{remap_node, FamilyInputs, Node, OperatorBody, OperatorProgram},
    parameter_response_program, program_learned_dag,
    program_structure_search::{
        self, EvaluatedArtifact, Evaluation as StructureEvaluation, Metric, Mutation,
    },
    resident_causal_fit::{self, Episode, Settings as FitSettings},
    run_check::{layer_nodes, split_sites, LayerNodes},
};
use ndarray::{Array1, Array2};
use serde::Deserialize;
use serde_json::{json, Value};
use std::{
    collections::{BTreeMap, BTreeSet},
    io::Write,
    path::Path,
    sync::Arc,
    time::Instant,
};

#[derive(Clone, Deserialize, serde::Serialize)]
#[serde(tag = "kind", rename_all = "snake_case", deny_unknown_fields)]
enum NativeControl {
    MlpOutput {
        layer: usize,
    },
    AttentionOutput {
        layer: usize,
    },
    /// A global gain of a native operator retained identically in every candidate.
    RetainedOperator {
        name: String,
    },
}
#[derive(Clone, Deserialize, serde::Serialize)]
#[serde(deny_unknown_fields)]
struct Case {
    label: String,
    group: String,
    gains: Vec<f64>,
    #[serde(default)]
    down_amplitudes: Vec<f64>,
    #[serde(default)]
    parameter_amplitudes: Vec<f64>,
}
#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct Evaluation {
    export_sha256: String,
    sequences: usize,
    #[serde(default)]
    cases: Option<Vec<Case>>,
}
#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct FrozenSharedBody {
    /// Standalone body-source pool. Initialized boundary maps are not a fitted discovery export.
    pool: String,
    pool_sha256: String,
    /// SHA of the sibling pool.with_extension("json") declaration, including discovery uses.
    declaration_sha256: String,
}
#[derive(Deserialize, serde::Serialize)]
#[serde(deny_unknown_fields)]
struct DownDirection {
    output: Vec<f64>,
    hidden: Vec<f64>,
}
#[derive(Deserialize, serde::Serialize)]
#[serde(deny_unknown_fields)]
struct DownSettings {
    layer: usize,
    directions: Vec<DownDirection>,
    #[serde(default = "default_response_weight")]
    response_weight: f64,
}
#[derive(Deserialize, serde::Serialize)]
#[serde(deny_unknown_fields)]
struct NativeEditSettings {
    target_operator: usize,
    directions: Vec<Vec<Vec<f64>>>,
}
fn build_parameter_edits(
    native: &OperatorProgram,
    config: &NativeEditSettings,
) -> Result<native_parameter_edit::Family, String> {
    let directions = config
        .directions
        .iter()
        .map(|rows| {
            let width = rows.first().ok_or("empty native edit matrix")?.len();
            if width == 0 || rows.iter().any(|r| r.len() != width) {
                return Err("ragged native edit matrix".into());
            }
            Array2::from_shape_vec(
                (rows.len(), width),
                rows.iter().flatten().copied().collect(),
            )
            .map_err(|e| e.to_string())
        })
        .collect::<Result<Vec<_>, String>>()?;
    let (flat, _) = gam_mpd::artifact_device::mapped_inlined(native)?;
    if flat.nodes[match &flat.nodes[flat.output] {
        Node::Readout { input, .. } => *input,
        _ => flat.output,
    }]
    .operators()
    .contains(&config.target_operator)
    {
        return Err("native parameter edits to final readout unsupported".into());
    }
    fn depends(nodes: &[Node], node: usize, operator: usize) -> bool {
        let mut pending = vec![node];
        let mut seen = BTreeSet::new();
        while let Some(n) = pending.pop() {
            if !seen.insert(n) {
                continue;
            }
            if nodes[n].operators().contains(&operator) {
                return true;
            }
            pending.extend(nodes[n].arguments());
        }
        false
    }
    if !flat.nodes.iter().enumerate().any(|(n, node)| {
        matches!(node, Node::Pointwise { .. } | Node::Hadamard { .. })
            && depends(&flat.nodes, n, config.target_operator)
    }) {
        return Err("native parameter edit must feed a downstream nonlinear computation".into());
    }
    native_parameter_edit::build(native, config.target_operator, &directions)
}
fn reject_native_edit_target_gain(target: usize, controls: &[Control]) -> Result<(), String> {
    if controls
        .iter()
        .any(|c| matches!(c,Control::GlobalOperatorScale {operator} if *operator==target))
    {
        return Err(
            "simultaneous gain on native edit target unsupported: no implicit edit/gain order"
                .into(),
        );
    }
    Ok(())
}
fn validate_parameter_edit_configuration(settings: &Settings) -> Result<(), String> {
    let count = settings
        .native_parameter_edits
        .as_ref()
        .map_or(0, |c| c.directions.len());
    if let Some(config) = &settings.native_parameter_edits {
        if settings.structural_search.is_none()
            || settings.down_edit_family.is_some()
            || settings.joint_response_search.is_some()
            || config.directions.is_empty()
        {
            return Err("native_parameter_edits requires structural_search and nonempty dense directions; down/fixed-bank joint modes are unsupported".into());
        }
    }
    for cases in std::iter::once(settings.cases.as_slice()).chain(
        settings
            .evaluation
            .as_ref()
            .and_then(|e| e.cases.as_deref()),
    ) {
        if cases.iter().any(|c| {
            c.parameter_amplitudes.len() != count
                || c.parameter_amplitudes.iter().any(|a| !a.is_finite())
        }) {
            return Err("one finite amplitude per native parameter edit required".into());
        }
    }
    if count > 0
        && (!settings.cases.iter().any(|c| {
            c.gains.iter().all(|g| *g == 1.) && c.parameter_amplitudes.iter().all(|a| *a == 0.)
        }) || !settings
            .cases
            .iter()
            .flat_map(|c| &c.parameter_amplitudes)
            .any(|a| *a > 0.)
            || !settings
                .cases
                .iter()
                .flat_map(|c| &c.parameter_amplitudes)
                .any(|a| *a < 0.))
    {
        return Err(
            "native parameter edits require clean plus prespecified signed training amplitudes"
                .into(),
        );
    }
    Ok(())
}
fn default_response_weight() -> f64 {
    1.
}
#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct Settings {
    export_sha256: String,
    layers: usize,
    #[serde(default)]
    uses: Vec<usize>,
    #[serde(default)]
    width: usize,
    sequences: usize,
    context: usize,
    /// Host label cache plus numeric native forward buffers; excludes library/allocator overhead.
    teacher_numeric_bytes: usize,
    /// Optional compact labels for structural search or finite down-weight response mode.
    #[serde(default)]
    fixed_head_targets: Option<FixedHeadSettings>,
    /// Immutable native codewords and decoded buffers only; zero disables cache.
    #[serde(default)]
    native_codec_bytes: usize,
    #[serde(default = "empty_grammar")]
    grammar: Grammar,
    /// Explicit finite subinventory for separate, reproducible compute allocations.
    #[serde(default)]
    expression_ids: Vec<usize>,
    #[serde(default)]
    require_interior_learned: bool,
    #[serde(default)]
    frozen_shared_body: Option<FrozenSharedBody>,
    #[serde(default)]
    native_initialization: bool,
    #[serde(default)]
    down_edit_family: Option<DownSettings>,
    #[serde(default)]
    native_parameter_edits: Option<NativeEditSettings>,
    #[serde(default)]
    structural_search: Option<StructuralSettings>,
    #[serde(default)]
    joint_response_search: Option<program_learned_dag::Settings>,
    controls: Vec<NativeControl>,
    cases: Vec<Case>,
    fit: FitSettings,
    seed: u64,
    #[serde(default)]
    evaluation: Option<Evaluation>,
}
#[derive(Clone, Deserialize, serde::Serialize)]
#[serde(deny_unknown_fields)]
struct FixedHeadSettings {
    tile_rows: usize,
}

/// Label-independent capture metadata; never masquerades as a full-logit episode.
#[derive(Clone)]
struct ResponseEpisode {
    label: String,
    inputs: FamilyInputs,
    scored: Option<Vec<bool>>,
    endpoint_bytes: usize,
}
fn response_metadata(episodes: &[Episode]) -> Vec<ResponseEpisode> {
    episodes
        .iter()
        .map(|e| ResponseEpisode {
            label: e.label.clone(),
            inputs: e.inputs.clone(),
            scored: e.scored.clone(),
            endpoint_bytes: e.target_logits.len().saturating_mul(8),
        })
        .collect()
}
struct CausalEpisodes {
    metadata: Vec<ResponseEpisode>,
    full: Option<Vec<Episode>>,
    compact: Option<Vec<resident_causal_fit::FixedHeadEpisode>>,
    tile_rows: usize,
    target_metadata: Value,
}
impl CausalEpisodes {
    fn from_family(
        d: &Device,
        lowered: &intervention_program::Compiled,
        source: &OperatorProgram,
        controls: &[Control],
        family: &FamilyInputs,
        cases: &[Case],
        labels: &[Array2<f64>],
        config: Option<&FixedHeadSettings>,
        teacher_bytes: usize,
    ) -> Result<Self, String> {
        let Some(config) = config else {
            let full = episodes(lowered, source, controls, family, cases, labels, None)?;
            return Ok(Self {
                metadata: response_metadata(&full),
                full: Some(full),
                compact: None,
                tile_rows: 0,
                target_metadata: json!({"backend":"full_logits"}),
            });
        };
        if config.tile_rows == 0 || !labels.is_empty() {
            return Err(
                "compact episodes require positive tile_rows and no full-logit intermediate".into(),
            );
        }
        let teacher = resident_causal_fit::fixed_head_target::Teacher::new(
            d,
            &lowered.program,
            config.tile_rows,
            teacher_bytes,
        )?;
        let mut metadata = Vec::new();
        let mut compact = Vec::new();
        let mut bytes = 0usize;
        let mut entropy_bytes = 0usize;
        for case in cases {
            let inputs = lowered.family(family, &values(source, controls, family.rows, case)?)?;
            let target = teacher.target(&inputs, None)?;
            let target_bytes = target
                .numeric_bytes()
                .checked_add(
                    target
                        .rows()
                        .checked_mul(8)
                        .ok_or("entropy bytes overflow")?,
                )
                .ok_or("target bytes overflow")?;
            bytes = bytes
                .checked_add(target.numeric_bytes())
                .ok_or("compact bytes overflow")?;
            entropy_bytes = entropy_bytes
                .checked_add(target.rows() * 8)
                .ok_or("entropy bytes overflow")?;
            metadata.push(ResponseEpisode {
                label: case.label.clone(),
                inputs: inputs.clone(),
                scored: None,
                endpoint_bytes: target_bytes,
            });
            compact.push(resident_causal_fit::FixedHeadEpisode {
                label: case.label.clone(),
                group: case.group.clone(),
                inputs,
                target,
            });
        }
        Ok(Self {
            metadata,
            full: None,
            compact: Some(compact),
            tile_rows: config.tile_rows,
            target_metadata: json!({"backend":"fixed_head","tile_rows":config.tile_rows,"resident_mu_numeric_bytes":bytes,"host_entropy_numeric_bytes":entropy_bytes,"full_logit_target_bytes":0,"teacher_numeric_bytes_limit":teacher_bytes,"scope":"Immutable bias-free dense head sufficient statistics; full candidate vocabulary normalization retained. Operational arithmetic, not certified acceptance."}),
        })
    }
    // Rebind only candidate-owned inputs; immutable native labels are shared.
    fn rebind(
        teacher: Option<&Self>,
        lowered: &intervention_program::Compiled,
        source: &OperatorProgram,
        controls: &[Control],
        family: &FamilyInputs,
        cases: &[Case],
        labels: &[Array2<f64>],
        down: Option<&DownFamily>,
    ) -> Result<Self, String> {
        if let Some((teacher, targets)) =
            teacher.and_then(|t| t.compact.as_ref().map(|targets| (t, targets)))
        {
            if targets.len() != cases.len() || !labels.is_empty() {
                return Err("compact case count differs or full logits supplied".into());
            }
            let mut metadata = Vec::new();
            let mut compact = Vec::new();
            for (case, teacher) in cases.iter().zip(targets) {
                if teacher.label != case.label || teacher.group != case.group {
                    return Err("compact target case identity differs".into());
                }
                let inputs = lowered.family(
                    &episode_family(family, case, down)?,
                    &values(source, controls, family.rows, case)?,
                )?;
                metadata.push(ResponseEpisode {
                    label: case.label.clone(),
                    inputs: inputs.clone(),
                    scored: None,
                    endpoint_bytes: teacher
                        .target
                        .numeric_bytes()
                        .checked_add(
                            teacher
                                .target
                                .rows()
                                .checked_mul(8)
                                .ok_or("entropy bytes overflow")?,
                        )
                        .ok_or("endpoint bytes overflow")?,
                });
                compact.push(resident_causal_fit::FixedHeadEpisode {
                    label: case.label.clone(),
                    group: case.group.clone(),
                    inputs,
                    target: teacher.target.clone(),
                });
            }
            Ok(Self {
                metadata,
                full: None,
                compact: Some(compact),
                tile_rows: teacher.tile_rows,
                target_metadata: teacher.target_metadata.clone(),
            })
        } else {
            let full = episodes(lowered, source, controls, family, cases, labels, down)?;
            Ok(Self {
                metadata: response_metadata(&full),
                full: Some(full),
                compact: None,
                tile_rows: 0,
                target_metadata: json!({"backend":"full_logits"}),
            })
        }
    }
    fn standalone(
        d: &Device,
        lowered: &intervention_program::Compiled,
        source: &OperatorProgram,
        controls: &[Control],
        family: &FamilyInputs,
        cases: &[Case],
        labels: &[Array2<f64>],
        config: Option<&FixedHeadSettings>,
        teacher_bytes: usize,
        down: Option<&DownFamily>,
        original_layers: &[LayerNodes],
        specs: &[NativeControl],
    ) -> Result<Self, String> {
        let Some(down) = down else {
            return Self::from_family(
                d,
                lowered,
                source,
                controls,
                family,
                cases,
                labels,
                config,
                teacher_bytes,
            );
        };
        let Some(config) = config else {
            let full = episodes(lowered, source, controls, family, cases, labels, Some(down))?;
            return Ok(Self {
                metadata: response_metadata(&full),
                full: Some(full),
                compact: None,
                tile_rows: 0,
                target_metadata: json!({"backend":"full_logits"}),
            });
        };
        if config.tile_rows == 0 || !labels.is_empty() {
            return Err("invalid compact down targets".into());
        }
        let mut compact: Vec<resident_causal_fit::FixedHeadEpisode> = Vec::new();
        let mut mu_bytes = 0usize;
        let mut entropy_bytes = 0usize;
        for case in cases {
            // Literal checkpoint matrix edits, never candidate response branches.
            let edited = down.literal_native(&case.down_amplitudes)?;
            let native = Artifact::native(&edited)?;
            let mapped = crate::controls(&native, &edited, original_layers, specs)?;
            let teacher_graph = intervention_program::compile(&edited, &mapped)?;
            let retained = mu_bytes
                .checked_add(entropy_bytes)
                .ok_or("compact cache overflow")?;
            let teacher = resident_causal_fit::fixed_head_target::Teacher::new(
                d,
                &teacher_graph.program,
                config.tile_rows,
                teacher_bytes
                    .checked_sub(retained)
                    .ok_or("compact cache exceeds budget")?,
            )?;
            let inputs =
                teacher_graph.family(family, &values(&edited, &mapped, family.rows, case)?)?;
            let target = teacher.target(&inputs, None)?;
            let target = if let Some(reference) = compact.first() {
                target.with_shared_head(&reference.target)?
            } else {
                target
            };
            mu_bytes = mu_bytes
                .checked_add(target.numeric_bytes())
                .ok_or("compact cache overflow")?;
            entropy_bytes = entropy_bytes
                .checked_add(target.rows().checked_mul(8).ok_or("entropy overflow")?)
                .ok_or("entropy cache overflow")?;
            compact.push(resident_causal_fit::FixedHeadEpisode {
                label: case.label.clone(),
                group: case.group.clone(),
                inputs,
                target,
            });
        }
        let teacher = Self {
            metadata: Vec::new(),
            full: None,
            compact: Some(compact),
            tile_rows: config.tile_rows,
            target_metadata: json!({"backend":"fixed_head","tile_rows":config.tile_rows,
                "resident_mu_numeric_bytes":mu_bytes,"host_entropy_numeric_bytes":entropy_bytes,
                "full_logit_target_bytes":0,"teacher_numeric_bytes_limit":teacher_bytes,
                "scope":"Literal edited native checkpoint teachers; immutable head sufficient statistics; candidate-owned augmented response inputs. Operational arithmetic, not certified acceptance."}),
        };
        Self::rebind(
            Some(&teacher),
            lowered,
            source,
            controls,
            family,
            cases,
            labels,
            Some(down),
        )
    }
    fn response_capture_budget(&self, limit: usize) -> Result<usize, String> {
        if self.compact.is_none() {
            return Ok(limit);
        }
        let bytes = self.metadata.iter().try_fold(0usize, |sum, e| {
            sum.checked_add(e.endpoint_bytes)
                .ok_or("compact endpoint cache overflow")
        })?;
        limit
            .checked_sub(bytes)
            .ok_or_else(|| "compact endpoint cache exceeds response capture budget".into())
    }
    fn fit(
        &self,
        d: &Device,
        source: &OperatorProgram,
        responses: Option<&resident_causal_fit::NativeResponses>,
        trainable: &[usize],
        settings: FitSettings,
    ) -> Result<resident_causal_fit::Fit, String> {
        gam_gpu::trace::within("fit", d, || match (&self.compact, responses) {
            (Some(e), Some(r)) => resident_causal_fit::fit_fixed_head_with_native(
                d,
                source,
                e,
                r,
                trainable,
                settings,
                self.tile_rows,
            ),
            (Some(e), None) => resident_causal_fit::fit_fixed_head(
                d,
                source,
                e,
                trainable,
                settings,
                self.tile_rows,
            ),
            (None, Some(r)) => resident_causal_fit::fit_with_native(
                d,
                source,
                self.full
                    .as_deref()
                    .ok_or("full-logit dispatch missing episodes")?,
                r,
                trainable,
                settings,
            ),
            (None, None) => resident_causal_fit::fit(
                d,
                source,
                self.full
                    .as_deref()
                    .ok_or("full-logit dispatch missing episodes")?,
                trainable,
                settings,
            ),
        })
    }
    fn measure(
        &self,
        d: &Device,
        source: &OperatorProgram,
        responses: Option<&resident_causal_fit::NativeResponses>,
        bytes: usize,
    ) -> Result<resident_causal_fit::Measurement, String> {
        gam_gpu::trace::within("measure", d, || match (&self.compact, responses) {
            (Some(e), Some(r)) => resident_causal_fit::measure_fixed_head_with_native(
                d,
                source,
                e,
                r,
                bytes,
                self.tile_rows,
            ),
            (Some(e), None) => {
                resident_causal_fit::measure_fixed_head(d, source, e, bytes, self.tile_rows)
            }
            (None, Some(r)) => resident_causal_fit::measure_with_native(
                d,
                source,
                self.full
                    .as_deref()
                    .ok_or("full-logit dispatch missing episodes")?,
                r,
                bytes,
            ),
            (None, None) => resident_causal_fit::measure(
                d,
                source,
                self.full
                    .as_deref()
                    .ok_or("full-logit dispatch missing episodes")?,
                bytes,
            ),
        })
    }
}

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct StructuralSettings {
    settings: program_structure_search::Settings,
    constraints: program_structure_search::Constraints,
    /// Combined frozen-evaluation limit for direct expressions, shared DAGs,
    /// and learned DAGs, in deterministic training admission order.
    #[serde(default)]
    max_expression_evaluations: usize,
    /// Optional mean native-trajectory response loss during proposal fitting.
    /// Separate from the unchanged teacher-input Local acceptance diagnostic.
    #[serde(default)]
    native_response_weight: Option<f64>,
}
fn empty_grammar() -> Grammar {
    Grammar {
        arguments: 1,
        max_operations: 0,
        max_expressions: 1,
        unary: vec![],
        binary: vec![],
        affine: false,
    }
}
fn save(path: &Path, value: &Value) -> Result<(), String> {
    std::fs::write(
        path,
        serde_json::to_vec_pretty(value).map_err(|e| e.to_string())?,
    )
    .map_err(|e| e.to_string())
}

// Preserve state fidelity as an independent selection axis for joint programs.
// This is a maximum of sampled episode/observable means, not a per-token bound.
fn maximum_response_error(m: &resident_causal_fit::Measurement) -> Result<f64, String> {
    let mut largest = 0.0_f64;
    if m.episodes.is_empty() || m.episodes.iter().any(|e| e.responses.is_empty()) {
        return Err("response selection requires measured native observables".into());
    }
    for response in m.episodes.iter().flat_map(|e| &e.responses) {
        let error = response.mean_normalized_squared_error;
        if !error.is_finite() || error < 0. || response.scored_rows == 0 {
            return Err("invalid measured native response error".into());
        }
        largest = largest.max(error);
    }
    Ok(largest)
}

fn training_dominates(b: &Value, a: &Value, joint: bool) -> bool {
    let (Some(ac), Some(bc), Some(ae), Some(be)) = (
        a["c32"].as_u64(),
        b["c32"].as_u64(),
        a["training_kl"].as_f64(),
        b["training_kl"].as_f64(),
    ) else {
        return false;
    };
    if joint {
        let (Some(ar), Some(br)) = (
            a["maximum_episode_response_error"].as_f64(),
            b["maximum_episode_response_error"].as_f64(),
        ) else {
            // Missing measurements are unknown, never an assumed zero.
            return false;
        };
        bc <= ac && be <= ae && br <= ar && (bc < ac || be < ae || br < ar)
    } else {
        bc <= ac && be <= ae && (bc < ac || be < ae)
    }
}
fn controls(
    artifact: &Artifact,
    native: &OperatorProgram,
    layers: &[LayerNodes],
    specs: &[NativeControl],
) -> Result<Vec<Control>, String> {
    specs
        .iter()
        .map(|spec| match spec {
            NativeControl::MlpOutput { layer } | NativeControl::AttentionOutput { layer } => {
                let l = layers.get(*layer).ok_or("controlled native layer absent")?;
                let n = if matches!(spec, NativeControl::MlpOutput { .. }) {
                    l.mlp
                } else {
                    l.attention
                };
                Ok(Control::NodeScale {
                    node: artifact
                        .place(n)
                        .ok_or("candidate does not retain declared native boundary")?,
                })
            }
            NativeControl::RetainedOperator { name } => {
                let source: Vec<_> = native
                    .operators
                    .iter()
                    .filter(|op| op.name == *name)
                    .collect();
                if source.len() != 1 {
                    return Err(format!("unique native operator required: {name}"));
                }
                let matches: Vec<_> = artifact
                    .program
                    .operators
                    .iter()
                    .enumerate()
                    .filter(|(_, op)| op.as_ref() == source[0].as_ref())
                    .collect();
                if matches.len() != 1 {
                    return Err(format!(
                        "candidate must retain identical global operator: {name}"
                    ));
                }
                Ok(Control::GlobalOperatorScale {
                    operator: matches[0].0,
                })
            }
        })
        .collect()
}
fn values(
    program: &OperatorProgram,
    controls: &[Control],
    rows: usize,
    case: &Case,
) -> Result<Vec<ControlValue>, String> {
    if case.gains.len() != controls.len() || case.gains.iter().any(|v| !v.is_finite()) {
        return Err("one finite gain per declared control required".into());
    }
    controls
        .iter()
        .zip(&case.gains)
        .map(|(c, &gain)| match c {
            Control::NodeScale { node } => Ok(ControlValue::NodeMask(Array2::from_elem(
                (
                    rows,
                    program
                        .node_interface(*node)
                        .map_err(|e| e.to_string())?
                        .width(),
                ),
                gain,
            ))),
            Control::GlobalOperatorScale { .. } => Ok(ControlValue::GlobalScale(gain)),
        })
        .collect()
}
fn episodes(
    lowered: &intervention_program::Compiled,
    source: &OperatorProgram,
    controls: &[Control],
    family: &FamilyInputs,
    cases: &[Case],
    targets: &[Array2<f64>],
    down: Option<&DownFamily>,
) -> Result<Vec<Episode>, String> {
    if cases.len() != targets.len() {
        return Err("episode target count mismatch".into());
    }
    cases
        .iter()
        .zip(targets)
        .map(|(case, target)| {
            Ok(Episode {
                label: case.label.clone(),
                group: case.group.clone(),
                inputs: lowered.family(
                    &episode_family(family, case, down)?,
                    &values(source, controls, family.rows, case)?,
                )?,
                target_logits: target.clone(),
                scored: None,
            })
        })
        .collect()
}
fn episode_family(
    base: &FamilyInputs,
    case: &Case,
    down: Option<&DownFamily>,
) -> Result<FamilyInputs, String> {
    match down {
        Some(down) => down.inputs(base, &case.down_amplitudes),
        None if case.down_amplitudes.is_empty() => Ok(base.clone()),
        None => Err("down amplitudes without declared family".into()),
    }
}
fn validate_cases(
    cases: &[Case],
    controls: usize,
    directions: usize,
    require_clean: bool,
) -> Result<(), String> {
    if cases.is_empty()
        || cases
            .iter()
            .map(|c| &c.label)
            .collect::<BTreeSet<_>>()
            .len()
            != cases.len()
        || cases.iter().any(|c| {
            c.gains.len() != controls
                || c.down_amplitudes.len() != directions
                || c.gains
                    .iter()
                    .chain(&c.down_amplitudes)
                    .any(|v| !v.is_finite())
        })
        || (require_clean
            && !cases.iter().any(|c| {
                c.gains.iter().all(|g| *g == 1.)
                    && c.down_amplitudes.iter().all(|a| *a == 0.)
                    && c.parameter_amplitudes.iter().all(|a| *a == 0.)
            }))
    {
        return Err("unique finite cases matching controls/directions and an explicit zero-edit clean case required".into());
    }
    Ok(())
}
fn remap_layers(layers: &[LayerNodes], map: &[usize]) -> Vec<LayerNodes> {
    layers
        .iter()
        .map(|l| LayerNodes {
            stream: map[l.stream],
            normed_stream: map[l.normed_stream],
            queries: l.queries.iter().map(|n| map[*n]).collect(),
            keys: l.keys.iter().map(|n| map[*n]).collect(),
            values: l.values.iter().map(|n| map[*n]).collect(),
            reads: l.reads.iter().map(|n| map[*n]).collect(),
            attention: map[l.attention],
            attended: map[l.attended],
            normed: map[l.normed],
            pre: map[l.pre],
            active: map[l.active],
            mlp: map[l.mlp],
            residual: map[l.residual],
        })
        .collect()
}
fn fixed_response_writers(
    candidate: &Artifact,
    down: &DownFamily,
) -> Result<BTreeSet<usize>, String> {
    let write = candidate
        .place(down.native_write)
        .ok_or("augmented response write absent")?;
    let Node::Call { rule, .. } = candidate.program.nodes[write] else {
        return Err("response wrapper Call required".into());
    };
    let rule = &candidate.program.rules[rule];
    let Node::Affine { terms, .. } = &rule.nodes[rule.output] else {
        return Err("response wrapper affine output required".into());
    };
    if terms.len() < down.direction_operators.len() {
        return Err("response direction terms absent".into());
    }
    let last = &terms[terms.len() - down.direction_operators.len()..];
    let mut fixed = BTreeSet::new();
    for ((_, op), &(u, _)) in last.iter().zip(&down.direction_operators) {
        if !same_dense(
            &candidate.program.operators[*op],
            &down.program.operators[u],
        ) {
            return Err("fixed response writer differs from declared native direction".into());
        }
        fixed.insert(*op);
    }
    Ok(fixed)
}
fn verify_fixed_directions(
    candidate: &Artifact,
    down: &DownFamily,
    ids: &[(usize, usize)],
) -> Result<(), String> {
    if ids.len() != down.direction_operators.len() {
        return Err("saved direction count mismatch".into());
    }
    for (&(u, v), &(source_u, source_v)) in ids.iter().zip(&down.direction_operators) {
        if !same_dense(
            candidate.program.operators.get(u).ok_or("saved U absent")?,
            &down.program.operators[source_u],
        ) || !same_dense(
            candidate.program.operators.get(v).ok_or("saved V absent")?,
            &down.program.operators[source_v],
        ) {
            return Err("fixed native edit directions changed".into());
        }
    }
    Ok(())
}
/// Native labels use actually edited checkpoint matrices, not the augmented branch.
fn family_teacher_targets(
    d: &Device,
    native: &OperatorProgram,
    layers: &[LayerNodes],
    specs: &[NativeControl],
    base: &FamilyInputs,
    cases: &[Case],
    down: Option<&DownFamily>,
    numeric_bytes: usize,
    teacher_bytes: usize,
) -> Result<(Vec<Array2<f64>>, usize), String> {
    if let Some(down) = down {
        let classes = native
            .node_interface(native.output)
            .map_err(|e| e.to_string())?
            .width();
        let label_bytes = base
            .rows
            .checked_mul(classes)
            .and_then(|n| n.checked_mul(8))
            .ok_or("teacher label bytes overflow")?;
        let cache_extra = label_bytes
            .checked_mul(cases.len().saturating_sub(1))
            .ok_or("teacher label cache overflow")?;
        let one_budget = teacher_bytes
            .checked_sub(cache_extra)
            .ok_or("teacher label cache exceeds budget")?;
        let mut targets = vec![];
        let mut peak = 0;
        for case in cases {
            let edited = down.literal_native(&case.down_amplitudes)?;
            let source = Artifact::native(&edited)?;
            let mapped = controls(&source, &edited, layers, specs)?;
            let (mut labels, plan) = teacher_targets(
                d,
                &edited,
                &mapped,
                base,
                std::slice::from_ref(case),
                numeric_bytes,
                one_budget,
            )?;
            peak = peak.max(
                plan.checked_add(cache_extra)
                    .ok_or("teacher plan overflow")?,
            );
            targets.append(&mut labels);
        }
        Ok((targets, peak))
    } else {
        let source = Artifact::native(native)?;
        let mapped = controls(&source, native, layers, specs)?;
        teacher_targets(
            d,
            native,
            &mapped,
            base,
            cases,
            numeric_bytes,
            teacher_bytes,
        )
    }
}
fn canonical(artifact: &Artifact) -> Result<(Artifact, Vec<u8>), String> {
    let bytes = artifact.f32_literals()?.to_bytes()?;
    let decoded = Artifact::from_bytes(&bytes, &artifact.program.declarations)?;
    if decoded.to_bytes()? != bytes {
        return Err("noncanonical candidate wire replay".into());
    }
    if decoded.program.nodes != artifact.program.nodes
        || decoded.program.output != artifact.program.output
        || decoded.program.operators.len() != artifact.program.operators.len()
        || decoded.program.rules.len() != artifact.program.rules.len()
        || decoded
            .program
            .rules
            .iter()
            .zip(&artifact.program.rules)
            .any(|(a, b)| a.nodes != b.nodes || a.output != b.output || a.inputs != b.inputs)
        || decoded
            .program
            .operators
            .iter()
            .zip(&artifact.program.operators)
            .any(|(a, b)| a.rows != b.rows || a.cols != b.cols)
        || decoded.places != artifact.places
    {
        return Err("wire replay changed executable reference indices".into());
    }
    Ok((decoded, bytes))
}

fn canonical_using(
    artifact: &Artifact,
    cache: Option<&CanonicalArtifactCache>,
) -> Result<(Artifact, Vec<u8>, Value), String> {
    gam_gpu::trace::within_host("canonical", || canonical_timed(artifact, cache))
}
fn canonical_timed(
    artifact: &Artifact,
    cache: Option<&CanonicalArtifactCache>,
) -> Result<(Artifact, Vec<u8>, Value), String> {
    if let Some(cache) = cache {
        let replay = cache.canonical(artifact)?;
        Ok((
            replay.decoded,
            replay.bytes,
            serde_json::to_value(replay.timings).map_err(|e| e.to_string())?,
        ))
    } else {
        let start = Instant::now();
        let (decoded, bytes) = canonical(artifact)?;
        Ok((
            decoded,
            bytes,
            json!({"uncached_total_seconds":start.elapsed().as_secs_f64()}),
        ))
    }
}
fn teacher_targets(
    d: &Device,
    source: &OperatorProgram,
    controls: &[Control],
    family: &FamilyInputs,
    cases: &[Case],
    numeric_bytes: usize,
    teacher_numeric_bytes: usize,
) -> Result<(Vec<Array2<f64>>, usize), String> {
    let graph = intervention_program::compile(source, controls)?;
    let resident = DeviceProgram::compile_values_bounded(d, &graph.program, numeric_bytes)?;
    let classes = graph
        .program
        .node_interface(graph.program.output)
        .map_err(|e| e.to_string())?
        .width();
    let labels = family
        .rows
        .checked_mul(classes)
        .and_then(|n| n.checked_mul(8))
        .and_then(|n| n.checked_mul(cases.len()))
        .ok_or("teacher label bytes overflow")?;
    let trace = resident
        .edited_bytes_per_row()
        .checked_mul(family.rows)
        .and_then(|n| n.checked_mul(4))
        .ok_or("teacher trace bytes overflow")?;
    let attention = family
        .rows
        .checked_mul(family.rows)
        .and_then(|n| n.checked_mul(8 * 12))
        .ok_or("teacher attention bytes overflow")?;
    let planned = resident
        .operator_numeric_bytes()?
        .checked_add(labels)
        .and_then(|n| n.checked_add(trace))
        .and_then(|n| n.checked_add(attention))
        .ok_or("teacher plan overflow")?;
    if planned > teacher_numeric_bytes {
        return Err(format!(
            "teacher numeric plan {planned} exceeds {teacher_numeric_bytes}"
        ));
    }
    let mut targets = Vec::new();
    for case in cases {
        let input = graph.family(family, &values(source, controls, family.rows, case)?)?;
        let trace = resident.forward(&input)?;
        targets.push(
            d.download(trace.value(graph.program.output)?)
                .map_err(|e| e.to_string())?,
        );
    }
    Ok((targets, planned))
}

/// Fixed native trajectory labels at every changed observable boundary. These
/// arrays only enter the loss; the candidate still executes from its own states.
fn native_response_targets_records(
    d: &Device,
    native: &OperatorProgram,
    candidate: &Artifact,
    records: &[ResponseEpisode],
    weight: f64,
    limit: usize,
) -> Result<(resident_causal_fit::NativeResponses, Value), String> {
    let mut boundaries = BTreeMap::new();
    for b in &candidate.blocks {
        if boundaries
            .insert(b.native_write, b.write)
            .is_some_and(|old| old != b.write)
        {
            return Err("conflicting native response boundaries".into());
        }
        if candidate
            .program
            .node_interface(b.write)
            .map_err(|e| e.to_string())?
            != native
                .node_interface(b.native_write)
                .map_err(|e| e.to_string())?
        {
            return Err("native response boundary changed interface".into());
        }
    }
    native_response_targets_mapped(d, native, &boundaries, records, weight, limit)
}
#[cfg(test)]
fn native_response_targets(
    d: &Device,
    native: &OperatorProgram,
    candidate: &Artifact,
    episodes: &[Episode],
    weight: f64,
    limit: usize,
) -> Result<(resident_causal_fit::NativeResponses, Value), String> {
    let mut boundaries = BTreeMap::new();
    for b in &candidate.blocks {
        if boundaries
            .insert(b.native_write, b.write)
            .is_some_and(|old| old != b.write)
        {
            return Err("one native response has conflicting candidate boundaries".into());
        }
        if candidate
            .program
            .node_interface(b.write)
            .map_err(|e| e.to_string())?
            != native
                .node_interface(b.native_write)
                .map_err(|e| e.to_string())?
        {
            return Err("native response boundary has changed interface".into());
        }
    }
    native_response_targets_mapped(
        d,
        native,
        &boundaries,
        &response_metadata(episodes),
        weight,
        limit,
    )
}

// Targets are independently executed native states; this map identifies only
// candidate-owned observations, never supplies candidate forward inputs.
fn native_response_targets_mapped(
    d: &Device,
    native: &OperatorProgram,
    boundaries: &BTreeMap<usize, usize>,
    episodes: &[ResponseEpisode],
    weight: f64,
    limit: usize,
) -> Result<(resident_causal_fit::NativeResponses, Value), String> {
    if !weight.is_finite() || weight <= 0. || episodes.is_empty() || boundaries.is_empty() {
        return Err(
            "positive native-response weight and nonempty explicit boundaries/episodes required"
                .into(),
        );
    }
    if episodes.iter().any(|e| {
        e.inputs.rows == 0
            || e.scored
                .as_ref()
                .is_some_and(|m| m.len() != e.inputs.rows || !m.iter().any(|v| *v))
    }) {
        return Err("nonempty exact response row domains required".into());
    }
    if boundaries.keys().any(|n| *n >= native.nodes.len()) {
        return Err("native response node out of range".into());
    }
    let (mut expanded, mapping) = gam_mpd::artifact_device::mapped_inlined(native)?;
    let last = boundaries
        .keys()
        .map(|n| mapping[*n])
        .max()
        .ok_or("no response nodes")?;
    // Keep all earlier nodes, including requested nodes outside the last one's ancestry.
    expanded.nodes.truncate(last + 1);
    expanded.output = last;
    let resident = DeviceProgram::compile_values_bounded(d, &expanded, limit)?;
    let rows = episodes.iter().try_fold(0usize, |sum, e| {
        sum.checked_add(e.inputs.rows)
            .ok_or("response row count overflow")
    })?;
    let widths = boundaries.keys().try_fold(0usize, |sum, n| {
        sum.checked_add(
            native
                .node_interface(*n)
                .map_err(|e| e.to_string())?
                .width(),
        )
        .ok_or_else(|| "response width overflow".to_string())
    })?;
    let label_bytes = rows
        .checked_mul(widths)
        .and_then(|n| n.checked_mul(8))
        .ok_or("response label bytes overflow")?;
    let maximum_rows = episodes
        .iter()
        .map(|e| e.inputs.rows)
        .max()
        .ok_or("no episodes")?;
    let trace_bytes = resident
        .edited_bytes_per_row()
        .checked_mul(maximum_rows)
        .and_then(|n| n.checked_mul(4))
        .ok_or("response trace bytes overflow")?;
    let attention_bytes = maximum_rows
        .checked_mul(maximum_rows)
        .and_then(|n| n.checked_mul(8 * 12))
        .ok_or("response attention bytes overflow")?;
    let existing_logits = episodes.iter().try_fold(0usize, |sum, e| {
        sum.checked_add(e.endpoint_bytes)
            .ok_or("response endpoint labels overflow")
    })?;
    let planned = resident
        .operator_numeric_bytes()?
        .checked_add(label_bytes)
        .and_then(|n| n.checked_add(trace_bytes))
        .and_then(|n| n.checked_add(attention_bytes))
        .and_then(|n| n.checked_add(existing_logits))
        .ok_or("response plan overflow")?;
    if planned > limit {
        return Err(format!(
            "native response capture numeric plan {planned} exceeds {limit}"
        ));
    }
    let mut labels = Vec::new();
    for episode in episodes {
        let trace = resident.forward(&episode.inputs)?;
        let captured = boundaries
            .keys()
            .map(|&node| {
                let values = d
                    .download(trace.value(mapping[node])?)
                    .map_err(|e| e.to_string())?;
                if values.iter().any(|v| !v.is_finite()) {
                    return Err("nonfinite native response target".to_string());
                }
                Ok((node, values))
            })
            .collect::<Result<BTreeMap<_, _>, String>>()?;
        labels.push(captured);
    }
    response_targets_from_captures(boundaries, episodes, labels, weight, planned)
}
fn response_targets_from_captures(
    boundaries: &BTreeMap<usize, usize>,
    episodes: &[ResponseEpisode],
    labels: Vec<BTreeMap<usize, Array2<f64>>>,
    weight: f64,
    planned: usize,
) -> Result<(resident_causal_fit::NativeResponses, Value), String> {
    let mut scales = BTreeMap::new();
    for &node in boundaries.keys() {
        // Scaled sum of squares avoids overflow while computing the pooled RMS
        // vector norm. The same fixed scale is used in every control episode.
        let (mut largest, mut squares) = (0.0_f64, 0.0_f64);
        let mut scored_rows = 0usize;
        for (episode, values) in episodes.iter().zip(&labels) {
            for (row, value) in values[&node].outer_iter().enumerate() {
                if episode.scored.as_ref().is_some_and(|m| !m[row]) {
                    continue;
                }
                scored_rows += 1;
                for &x in value {
                    let x = x.abs();
                    if x > largest {
                        squares = 1. + squares * (largest / x).powi(2);
                        largest = x;
                    } else if x > 0. {
                        squares += (x / largest).powi(2);
                    }
                }
            }
        }
        if scored_rows == 0 {
            return Err("no scored response rows".into());
        }
        let rms = largest * (squares / scored_rows as f64).sqrt();
        if !rms.is_finite() {
            return Err("nonfinite response RMS".into());
        }
        scales.insert(node, (rms, if rms == 0. { 1. } else { rms }));
    }
    let mut responses = BTreeMap::new();
    for (episode, values) in episodes.iter().zip(labels) {
        let targets = values
            .into_iter()
            .map(
                |(native_node, values)| resident_causal_fit::NativeResponseTarget {
                    label: format!("native_node_{native_node}"),
                    source_node: boundaries[&native_node],
                    values,
                    scored: episode.scored.clone(),
                    scale: scales[&native_node].1,
                    weight: weight / boundaries.len() as f64,
                },
            )
            .collect();
        if responses.insert(episode.label.clone(), targets).is_some() {
            return Err("duplicate response episode labels".into());
        }
    }
    Ok((
        responses,
        json!({"weight":weight,"native_to_candidate":boundaries,
        "native_rms_and_fixed_scale":scales,"planned_numeric_bytes":planned,
        "scope":"Fixed controlled-native trajectory targets at all distinct changed observable exits. Candidate states never replaced by labels. Weighted mean of normalized squared response errors plus output KL, distinct from teacher-input Local. Zero native RMS uses explicit unit absolute scale; training rows only."}),
    ))
}
fn down_response_targets(
    d: &Device,
    down: &DownFamily,
    native_controls: &[Control],
    family: &FamilyInputs,
    cases: &[Case],
    logits: &[Array2<f64>],
    observed_nodes: &[usize],
    weight: f64,
    limit: usize,
) -> Result<(resident_causal_fit::NativeResponses, Value), String> {
    if observed_nodes.len() != down.response_nodes.len()
        && observed_nodes.len() != down.response_nodes.len() + 1
    {
        return Err("response observation count differs".into());
    }
    let mut paths: Vec<_> = down.response_nodes.iter().map(|n| vec![*n]).collect();
    if observed_nodes.len() == down.response_nodes.len() + 1 {
        paths.insert(0, vec![down.clean_output]);
    }
    let teacher = intervention_program::compile_observed(&down.program, native_controls, &paths)?;
    if !logits.is_empty() && logits.len() != cases.len() {
        return Err("response label count differs".into());
    }
    let teacher_episodes = cases
        .iter()
        .enumerate()
        .map(|(index, case)| {
            Ok(ResponseEpisode {
                label: case.label.clone(),
                inputs: teacher.family(
                    &episode_family(family, case, Some(down))?,
                    &values(&down.program, native_controls, family.rows, case)?,
                )?,
                scored: None,
                endpoint_bytes: logits.get(index).map_or(0, |x| x.len() * 8),
            })
        })
        .collect::<Result<Vec<_>, String>>()?;
    let map = teacher
        .observed_nodes
        .iter()
        .copied()
        .zip(observed_nodes.iter().copied())
        .collect();
    let (targets, mut provenance) = native_response_targets_mapped(
        d,
        &teacher.program,
        &map,
        &teacher_episodes,
        weight,
        limit,
    )?;
    provenance["includes_clean_output"] =
        json!(observed_nodes.len() == down.response_nodes.len() + 1);
    provenance["scope"]=json!("Direct native clean-output (when requested) and unscaled scalar native down-edit coefficient targets under each case's upstream controls, before enclosing MLP-output masks. Candidate-owned coefficient functions; independent native teacher inputs. Not a rank-identification certificate or Local acceptance.");
    Ok((targets, provenance))
}

fn freeze_evaluation_ids(
    frontier: &[Value],
    expressions: &[usize],
    multiple_uses: bool,
) -> Vec<String> {
    let mut ids = vec!["native".to_string()];
    for expression in expressions {
        let shared = format!("expression{expression}-shared");
        let untied = format!("expression{expression}-untied");
        if frontier
            .iter()
            .any(|v| v.as_str() == Some(&shared) || v.as_str() == Some(&untied))
        {
            ids.push(shared);
            if multiple_uses {
                ids.push(untied);
            }
        }
    }
    ids
}
fn freeze_joint_evaluation_ids(frontier: &[Value], candidates: &[usize]) -> Vec<String> {
    let mut ids = vec!["native".into()];
    for candidate in candidates {
        let id = format!("joint{candidate}-learned");
        if frontier.iter().any(|v| v.as_str() == Some(&id)) {
            ids.push(id);
        }
    }
    ids
}
fn add_native_capacity_controls(
    ids: &mut Vec<String>,
    expressions: &[usize],
    multiple_uses: bool,
    native_initialization: bool,
) {
    if native_initialization {
        for expression in expressions {
            for arm in ["shared", "untied"] {
                if arm == "untied" && !multiple_uses {
                    continue;
                }
                let id = format!("expression{expression}-{arm}");
                if !ids.contains(&id) {
                    ids.push(id);
                }
            }
        }
    }
}
fn same_native(a: &OperatorProgram, b: &OperatorProgram) -> bool {
    a.declarations == b.declarations
        && a.bases == b.bases
        && a.nodes == b.nodes
        && a.output == b.output
        && a.rules.len() == b.rules.len()
        && a.rules
            .iter()
            .zip(&b.rules)
            .all(|(a, b)| a.inputs == b.inputs && a.nodes == b.nodes && a.output == b.output)
        && a.operators.len() == b.operators.len()
        && a.operators.iter().zip(&b.operators).all(|(a, b)| {
            if a.name != b.name || a.rows != b.rows || a.cols != b.cols || a.body != b.body {
                return false;
            }
            let x = a.matrix_cow();
            let y = b.matrix_cow();
            x.dim() == y.dim()
                && x.iter()
                    .zip(y.iter())
                    .all(|(x, y)| x.to_bits() == y.to_bits())
        })
}
fn disjoint_token_sequences(
    train: &FamilyInputs,
    heldout: &FamilyInputs,
    context: usize,
) -> Result<(), String> {
    let tokens = |family: &FamilyInputs| -> Result<Vec<u32>, String> {
        match family.slots.first() {
            Some(gam_mpd::operator_program::SlotValues::Tokens(v)) if v.len() == family.rows => {
                Ok(v.clone())
            }
            _ => Err("token panel required".into()),
        }
    };
    let train = tokens(train)?;
    let heldout = tokens(heldout)?;
    if context == 0 || train.len() % context != 0 || heldout.len() % context != 0 {
        return Err("complete fixed-context token sequences required".into());
    }
    let train_rows: BTreeSet<_> = train.chunks_exact(context).collect();
    if heldout
        .chunks_exact(context)
        .any(|row| train_rows.contains(row))
    {
        return Err("heldout contains a training token sequence".into());
    }
    Ok(())
}
#[derive(Deserialize)]
#[serde(tag = "kind", rename_all = "snake_case", deny_unknown_fields)]
enum StoredControl {
    NodeScale { candidate_node: usize },
    GlobalOperatorScale { candidate_operator: usize },
}
fn stored_controls(map: &Value, specs: &[NativeControl]) -> Result<Vec<Control>, String> {
    if map["native_controls"] != serde_json::to_value(specs).map_err(|e| e.to_string())? {
        return Err("saved native control declaration differs".into());
    }
    let stored: Vec<StoredControl> =
        serde_json::from_value(map["candidate_controls"].clone()).map_err(|e| e.to_string())?;
    if stored.len() != specs.len() {
        return Err("saved control count differs".into());
    }
    Ok(stored
        .into_iter()
        .map(|c| match c {
            StoredControl::NodeScale { candidate_node } => Control::NodeScale {
                node: candidate_node,
            },
            StoredControl::GlobalOperatorScale { candidate_operator } => {
                Control::GlobalOperatorScale {
                    operator: candidate_operator,
                }
            }
        })
        .collect())
}
fn control_map(controls: &[Control], specs: &[NativeControl]) -> Value {
    json!({"native_controls":specs,"candidate_controls":controls.iter().map(|c|match c{Control::NodeScale{node}=>json!({"kind":"node_scale","candidate_node":node}),Control::GlobalOperatorScale{operator}=>json!({"kind":"global_operator_scale","candidate_operator":operator})}).collect::<Vec<_>>()})
}
fn selection(
    inventory: &composed_rule_search::Inventory,
    ids: &[usize],
    require: bool,
) -> Result<composed_rule_search::SharingSelection, String> {
    if ids.is_empty()
        || ids.iter().any(|&id| id >= inventory.expressions.len())
        || ids.iter().copied().collect::<BTreeSet<_>>().len() != ids.len()
    {
        return Err("unique expression IDs inside declared inventory required".into());
    }
    let selected = composed_rule_search::interior_learned_selection(inventory);
    if require {
        for &id in ids {
            if !selected.analyses[id].has_interior_learned_sharing {
                return Err(format!(
                    "expression{id} lacks declared interior learned sharing; boundary/non-affine forms remain available in a separate baseline run"
                ));
            }
        }
    }
    Ok(selected)
}
fn body_dense_indices(program: &OperatorProgram, rule: usize) -> Result<Vec<usize>, String> {
    let body = program.rules.get(rule).ok_or("shared body rule absent")?;
    let mut ids = BTreeSet::new();
    for node in &body.nodes {
        match node {
            Node::Affine { terms, bias } => {
                ids.extend(terms.iter().map(|(_, op)| *op));
                ids.extend(*bias);
            }
            Node::Constant { operator } | Node::Transposed { operator, .. } => {
                ids.insert(*operator);
            }
            Node::Call { .. } => {
                return Err(
                    "transfer source requires direct generated body, not nested foreign calls"
                        .into(),
                );
            }
            _ => continue,
        }
    }
    Ok(ids
        .into_iter()
        .filter(|index| matches!(program.operators[*index].body, OperatorBody::Dense { .. }))
        .collect())
}
fn same_dense(
    a: &gam_mpd::operator_program::Operator,
    b: &gam_mpd::operator_program::Operator,
) -> bool {
    if a.rows != b.rows || a.cols != b.cols {
        return false;
    }
    match (&a.body, &b.body) {
        (
            OperatorBody::Dense {
                values: a,
                present: ap,
                precision: ad,
            },
            OperatorBody::Dense {
                values: b,
                present: bp,
                precision: bd,
            },
        ) => {
            a.dim() == b.dim()
                && ap == bp
                && ad == bd
                && a.iter().zip(b).all(|(x, y)| x.to_bits() == y.to_bits())
        }
        _ => false,
    }
}
/// Pointer identity identifies retained native operators; interface-specialized boundary
/// maps are newly allocated Dense parameters, so no opaque decoded/name lookup is needed.
#[cfg(test)]
fn graft_parameters(
    candidate: &Artifact,
    native: &OperatorProgram,
    proposal: &composed_rule_search::Proposal,
) -> Result<(Vec<usize>, BTreeMap<usize, usize>), String> {
    graft_parameters_except(candidate, native, proposal, &BTreeSet::new())
}
fn graft_parameters_except(
    candidate: &Artifact,
    native: &OperatorProgram,
    proposal: &composed_rule_search::Proposal,
    fixed: &BTreeSet<usize>,
) -> Result<(Vec<usize>, BTreeMap<usize, usize>), String> {
    let trainable: Vec<_> = candidate
        .program
        .operators
        .iter()
        .enumerate()
        .filter_map(|(id, op)| {
            (matches!(op.body, OperatorBody::Dense { .. })
                && !fixed.contains(&id)
                && !native.operators.iter().any(|held| Arc::ptr_eq(held, op)))
            .then_some(id)
        })
        .collect();
    if trainable.len() != proposal.trainable.len() {
        let actual:Vec<_>=trainable.iter().map(|&id|{
            let op=&candidate.program.operators[id];
            let origins:Vec<_>=proposal.trainable.iter().copied().filter(|&source|{
                let held=&proposal.program.operators[source];
                op.rows.width()==held.rows.width() && op.cols.width()==held.cols.width() && match (&op.body,&held.body) {
                    (OperatorBody::Dense{values:a,..},OperatorBody::Dense{values:b,..})=>a.dim()==b.dim() && a.iter().zip(b).all(|(x,y)|x.to_bits()==y.to_bits()),
                    _=>false,
                }
            }).collect();
            json!({"candidate_operator":id,"name":op.name,"rows":format!("{:?}",op.rows),"cols":format!("{:?}",op.cols),"matching_proposal_parameter_values":origins})
        }).collect();
        return Err(format!("graft parameter count differs: actual={} expected={}; unsupported parameter specialization or loss; operator_diagnostic={}",trainable.len(),proposal.trainable.len(),serde_json::to_string(&actual).map_err(|e|e.to_string())?));
    }
    let mut body_map = BTreeMap::new();
    for rule in 0..proposal.program.rules.len() {
        for source in body_dense_indices(&proposal.program, rule)? {
            let matches: Vec<_> = candidate
                .program
                .operators
                .iter()
                .enumerate()
                .filter(|(_, op)| Arc::ptr_eq(op, &proposal.program.operators[source]))
                .map(|(id, _)| id)
                .collect();
            if matches.len() != 1 {
                return Err("shared body operator lost or ambiguously specialized".into());
            }
            body_map.insert(source, matches[0]);
        }
    }
    Ok((trainable, body_map))
}
/// Follow affine occurrences in the unchanged compiled-node prefix. This maps
/// parameter owners across interface specialization without numerical/name matching.
fn joint_graft_parameters(
    candidate: &Artifact,
    write: usize,
    compiled: &program_learned_dag::Compiled,
) -> Result<Vec<usize>, String> {
    let place = candidate.place(write).ok_or("joint graft write absent")?;
    let Node::Call { rule, .. } = candidate.program.nodes[place] else {
        return Err("joint graft wrapper absent".into());
    };
    let nodes = &candidate.program.rules[rule].nodes;
    let owners: BTreeSet<_> = compiled.trainable_operator_ids.iter().copied().collect();
    let mut mapping = BTreeMap::new();
    for (index, source) in compiled.program.nodes.iter().enumerate() {
        if let Node::Affine { terms, bias } = source {
            let Some(Node::Affine {
                terms: actual,
                bias: actual_bias,
            }) = nodes.get(index)
            else {
                return Err("joint affine occurrence absent".into());
            };
            if terms.len() != actual.len() || bias.is_some() != actual_bias.is_some() {
                return Err("joint affine occurrence changed".into());
            }
            for (source, target) in terms
                .iter()
                .map(|(_, op)| *op)
                .chain(bias.iter().copied())
                .zip(
                    actual
                        .iter()
                        .map(|(_, op)| *op)
                        .chain(actual_bias.iter().copied()),
                )
            {
                if owners.contains(&source) {
                    if mapping
                        .insert(source, target)
                        .is_some_and(|held| held != target)
                    {
                        return Err("joint parameter owner split by boundary specialization".into());
                    }
                }
            }
        }
    }
    if mapping.len() != owners.len() {
        return Err("joint graft lost parameter owner".into());
    }
    let ids: Vec<_> = mapping.values().copied().collect();
    if ids.iter().copied().collect::<BTreeSet<_>>().len() != ids.len() {
        return Err("joint graft merged parameter owners".into());
    }
    Ok(ids)
}
fn copy_body(source: &OperatorProgram, target: &mut OperatorProgram) -> Result<(), String> {
    if source.rules.len() != 1 {
        return Err("transfer source must contain one shared rule".into());
    }
    let from = body_dense_indices(source, 0)?;
    if from.is_empty() {
        return Err("transfer source has no learned body coefficients".into());
    }
    for rule in 0..target.rules.len() {
        let to = body_dense_indices(target, rule)?;
        if from.len() != to.len() {
            return Err("transfer body parameter inventory differs".into());
        }
        let mut ops: Vec<_> = (0..source.operators.len()).collect();
        for (&a, &b) in from.iter().zip(&to) {
            ops[a] = b;
        }
        let mut expected = source.rules[0].clone();
        let nodes: Vec<_> = (0..expected.nodes.len()).collect();
        for node in &mut expected.nodes {
            remap_node(node, &nodes, &ops, &[], &[]);
        }
        let actual = &target.rules[rule];
        if expected.inputs != actual.inputs
            || expected.nodes != actual.nodes
            || expected.output != actual.output
        {
            return Err("transfer expression topology/interfaces differ".into());
        }
        for (&a, &b) in from.iter().zip(&to) {
            let from = &source.operators[a];
            let to = Arc::make_mut(&mut target.operators[b]);
            if from.rows != to.rows || from.cols != to.cols {
                return Err("transfer body dimensions differ".into());
            }
            to.body = from.body.clone();
        }
    }
    Ok(())
}
fn verify_body(
    proposal: &OperatorProgram,
    saved: &OperatorProgram,
    map: &BTreeMap<usize, usize>,
) -> Result<(), String> {
    for (&source, &graft) in map {
        if !same_dense(&proposal.operators[source], &saved.operators[graft]) {
            return Err("frozen body f32 coefficient bits changed".into());
        }
    }
    Ok(())
}
fn load_body(
    spec: &FrozenSharedBody,
    expression: &composed_rule_search::Expr,
    width: usize,
    uses: &[usize],
    checkpoint: &str,
) -> Result<(Artifact, Value), String> {
    let path = Path::new(&spec.pool);
    let declaration = path.with_extension("json");
    if sha256(path)? != spec.pool_sha256 || sha256(&declaration)? != spec.declaration_sha256 {
        return Err("frozen body source/declaration SHA mismatch".into());
    }
    let record: Value =
        serde_json::from_slice(&std::fs::read(declaration).map_err(|e| e.to_string())?)
            .map_err(|e| e.to_string())?;
    if record["expression"] != serde_json::to_value(expression).map_err(|e| e.to_string())?
        || record["width"].as_u64() != Some(width as u64)
        || record["native_checkpoint_sha256"].as_str() != Some(checkpoint)
        || record["pool_sha256"].as_str() != Some(&spec.pool_sha256)
    {
        return Err("frozen body expression/width/checkpoint identity differs".into());
    }
    let previous: Vec<usize> =
        serde_json::from_value(record["discovery_uses"].clone()).map_err(|e| e.to_string())?;
    if previous.is_empty() || previous.iter().any(|old| uses.contains(old)) {
        return Err("transfer requires disjoint declared native uses".into());
    }
    let widths: Vec<usize> =
        serde_json::from_value(record["declarations"]["raw_slot_widths"].clone())
            .map_err(|e| e.to_string())?;
    if record["declarations"]["domains"] != json!([])
        || record["declarations"]["parameters"].as_u64() != Some(0)
        || widths.len() != previous.len()
        || widths.iter().any(|w| *w == 0)
    {
        return Err("explicit standalone Raw pool declarations required".into());
    }
    let declarations = gam_mpd::operator_program::Declarations {
        domains: vec![],
        slots: widths
            .into_iter()
            .map(|width| gam_mpd::operator_program::Slot::Raw { width })
            .collect(),
        parameters: 0,
    };
    let bytes = std::fs::read(path).map_err(|e| e.to_string())?;
    let artifact = Artifact::from_bytes(&bytes, &declarations)?;
    if artifact.to_bytes()? != bytes
        || !artifact.blocks.is_empty()
        || !artifact.exceptions.is_empty()
        || !artifact.controls.is_empty()
        || !artifact.derived.is_empty()
    {
        return Err("standalone canonical frozen body pool required".into());
    }
    if !composed_rule_search::sharing_analysis(expression).has_interior_learned_sharing {
        return Err("frozen transfer requires an interior learned body".into());
    }
    if artifact.f32_literals()?.to_bytes()? != bytes {
        return Err("frozen source must already be ordinary f32 literals".into());
    }
    Ok((artifact, record))
}

fn native_uses(
    native: &OperatorProgram,
    layers: &[LayerNodes],
    uses: &[usize],
) -> Result<Vec<gam_mpd::native_mlp_initialization::NativeUse>, String> {
    uses.iter()
        .map(|&index| {
            let layer = layers
                .get(index)
                .ok_or("native initialization layer index")?;
            let read = match native.nodes.get(layer.active) {
                Some(Node::Pointwise { input, .. }) => *input,
                _ => {
                    return Err(
                        "native initialization requires primitive unary MLP activation".into(),
                    );
                }
            };
            Ok(gam_mpd::native_mlp_initialization::NativeUse {
                input: layer.normed,
                read,
                active: layer.active,
                write: layer.mlp,
            })
        })
        .collect()
}
// Explicit operational proposal convention, not an arithmetic enclosure.
const OPERATIONAL_NEGATIVE_KL_TOLERANCE: f64 = 1e-10;
fn operational_kl_axis(raw: f64) -> Result<f64, String> {
    if !raw.is_finite() || raw < -OPERATIONAL_NEGATIVE_KL_TOLERANCE {
        return Err(format!(
            "operational KL {raw} is nonfinite or below declared negative roundoff tolerance {}",
            OPERATIONAL_NEGATIVE_KL_TOLERANCE
        ));
    }
    Ok(raw.max(0.))
}
fn rounding_policy() -> Value {
    json!({"negative_kl_tolerance":OPERATIONAL_NEGATIVE_KL_TOLERANCE,"search_axis":"finite raw KL in [-tolerance,0) maps to zero for nonnegative proposal search axes only; more-negative/nonfinite scores reject","raw_reports":"INITIAL_TRAIN.json,TRAIN.json,and heldout measurements preserve unmodified fitter scores","scope":"Operational-only integration convention, not a proved rounding bound or accuracy certificate. Fitter, gradients, snapshots and certified acceptance are unchanged. Validates each reported episode mean/maximum, group mean and objective; no unreported per-row minimum certificate."})
}
fn structure_evaluation(
    artifact: &Artifact,
    measured: &resident_causal_fit::Measurement,
    local_error: f64,
    costs: &mut CostCache,
) -> Result<StructureEvaluation, String> {
    if !local_error.is_finite() {
        return Err("nonfinite structural Local measurement".into());
    }
    for episode in &measured.episodes {
        operational_kl_axis(episode.mean_kl)?;
        operational_kl_axis(episode.maximum_scored_row_kl)?;
    }
    Ok(StructureEvaluation {
        fidelity: vec![Metric {
            name: "maximum_group_mean_kl".into(),
            value: operational_kl_axis(measured.objective)?,
        }],
        local_errors: vec![Metric {
            name: "sampled_d_local".into(),
            value: local_error,
        }],
        intervention_errors: measured
            .groups
            .iter()
            .map(|(name, value)| {
                Ok(Metric {
                    name: name.clone(),
                    value: operational_kl_axis(*value)?,
                })
            })
            .collect::<Result<Vec<_>, String>>()?,
        description_bits: structural_cost(artifact, costs)?.total() as f64,
    })
}
fn frozen_structural_ids(
    frontier: &[usize],
    records: &[program_structure_search::CandidateRecord],
    expression_limit: usize,
) -> (Vec<usize>, Vec<usize>) {
    let expressions: Vec<_> = records
        .iter()
        .filter(|r| {
            matches!(
                r.mutation,
                Some(
                    Mutation::SynthesizeExpression { .. }
                        | Mutation::SynthesizeSharedDAG { .. }
                        | Mutation::SynthesizeLearnedDAG { .. }
                )
            )
        })
        .take(expression_limit)
        .map(|r| r.id)
        .collect();
    let mut ids = vec![0];
    for id in frontier.iter().chain(&expressions) {
        if !ids.contains(id) {
            ids.push(*id);
        }
    }
    (ids, expressions)
}
fn structural_artifact_path(
    out: &Path,
    result: &program_structure_search::SearchResult,
    id: usize,
) -> Result<std::path::PathBuf, String> {
    if id == 0 {
        return Ok(out.join("controlled-native.artifact"));
    }
    let attempt = result
        .report
        .attempts
        .iter()
        .find(|a| {
            a.candidate_id == Some(id)
                && matches!(
                    a.status,
                    program_structure_search::AttemptStatus::FittedAdmitted
                )
        })
        .ok_or("admitted candidate attempt missing")?;
    Ok(out.join(format!(
        "structural-attempt-{:06}/program.artifact",
        attempt.attempt_id
    )))
}
/// Literal teachers and augmented candidate inputs are constructed separately.
fn parameter_edit_episodes(
    d: &Device,
    edits: &native_parameter_edit::Family,
    controlled: &intervention_program::Compiled,
    native: &OperatorProgram,
    native_controls: &[Control],
    original: &OperatorProgram,
    settings: &Settings,
    family: &FamilyInputs,
    cases: &[Case],
) -> Result<CausalEpisodes, String> {
    let layers = if settings.controls.iter().any(|c| {
        matches!(
            c,
            NativeControl::MlpOutput { .. } | NativeControl::AttentionOutput { .. }
        )
    }) {
        layer_nodes(original, settings.layers)?
    } else {
        Vec::new()
    };
    let mut full = Vec::new();
    let mut compact: Vec<resident_causal_fit::FixedHeadEpisode> = Vec::new();
    let mut metadata = Vec::new();
    let mut retained = 0usize;
    for case in cases {
        let literal = edits.literal_native(&case.parameter_amplitudes)?;
        let teacher_controls = controls(
            &Artifact::native(&literal)?,
            &literal,
            &layers,
            &settings.controls,
        )?;
        let teacher_graph = intervention_program::compile(&literal, &teacher_controls)?;
        let budget = settings
            .teacher_numeric_bytes
            .checked_sub(retained)
            .ok_or("native edit labels exceed teacher budget")?;
        let labels = if settings.fixed_head_targets.is_some() {
            Vec::new()
        } else {
            teacher_targets(
                d,
                &literal,
                &teacher_controls,
                family,
                std::slice::from_ref(case),
                settings.fit.numeric_bytes,
                budget,
            )?
            .0
        };
        let mut teacher = CausalEpisodes::from_family(
            d,
            &teacher_graph,
            &literal,
            &teacher_controls,
            family,
            std::slice::from_ref(case),
            &labels,
            settings.fixed_head_targets.as_ref(),
            budget,
        )?;
        let inputs = controlled.family(
            &edits.inputs(family, &case.parameter_amplitudes)?,
            &values(native, native_controls, family.rows, case)?,
        )?;
        let mut record = teacher.metadata.remove(0);
        record.inputs = inputs.clone();
        if let Some(episodes) = &mut teacher.compact {
            if let Some(reference) = compact.first() {
                episodes[0].target = episodes[0].target.with_shared_head(&reference.target)?;
            } else {
                let head_bytes = original
                    .node_interface(original.output)
                    .map_err(|e| e.to_string())?
                    .width()
                    .checked_mul(episodes[0].target.width())
                    .and_then(|n| n.checked_mul(8))
                    .ok_or("shared immutable head bytes overflow")?;
                record.endpoint_bytes = record
                    .endpoint_bytes
                    .checked_add(head_bytes)
                    .ok_or("shared head endpoint bytes overflow")?;
            }
        }

        retained = retained
            .checked_add(record.endpoint_bytes)
            .ok_or("native edit target bytes overflow")?;
        metadata.push(record);
        if let Some(mut episodes) = teacher.full {
            episodes[0].inputs = inputs;
            full.extend(episodes);
        } else if let Some(mut episodes) = teacher.compact {
            episodes[0].inputs = inputs;
            compact.extend(episodes);
        }
    }
    let compact_mode = settings.fixed_head_targets.is_some();
    Ok(CausalEpisodes {
        metadata,
        full: (!compact_mode).then_some(full),
        compact: compact_mode.then_some(compact),
        tile_rows: settings
            .fixed_head_targets
            .as_ref()
            .map_or(0, |s| s.tile_rows),
        target_metadata: json!({"backend":if compact_mode{"fixed_head"}else{"full_logits"},"literal_native_parameter_edits":true,"resident_endpoint_numeric_bytes":retained,"full_logit_target_bytes":if compact_mode{0}else{retained},"teacher_numeric_bytes_limit":settings.teacher_numeric_bytes,"scope":"Each native teacher literally edits the original stored matrix before normal controls; candidate receives independent Raw edit controls. Compact mode never constructs full logits."}),
    })
}
fn parameter_edit_response_targets(
    d: &Device,
    edits: &native_parameter_edit::Family,
    controlled: &intervention_program::Compiled,
    original: &OperatorProgram,
    settings: &Settings,
    family: &FamilyInputs,
    cases: &[Case],
    candidate: &Artifact,
    records: &[ResponseEpisode],
    weight: f64,
    limit: usize,
) -> Result<(resident_causal_fit::NativeResponses, Value), String> {
    if cases.len() != records.len() {
        return Err("native edit response cases differ".into());
    }
    let original_to_controlled: Vec<_> = edits
        .node_mapping
        .iter()
        .map(|n| controlled.root_mapping[*n])
        .collect();
    let inverse: BTreeMap<_, _> = original_to_controlled
        .iter()
        .enumerate()
        .map(|(n, m)| (*m, n))
        .collect();
    let mut boundaries = BTreeMap::new();
    let mut originals = BTreeMap::new();
    for block in &candidate.blocks {
        let original_node = *inverse.get(&block.native_write).ok_or("native edit response requires an original native observable exit; synthetic lowering exits are unsupported")?;
        if candidate
            .program
            .node_interface(block.write)
            .map_err(|e| e.to_string())?
            != original
                .node_interface(original_node)
                .map_err(|e| e.to_string())?
        {
            return Err("native edit response interface changed".into());
        }
        if boundaries
            .insert(block.native_write, block.write)
            .is_some_and(|n| n != block.write)
        {
            return Err("conflicting native edit response boundaries".into());
        }
        originals.insert(block.native_write, original_node);
    }
    if boundaries.is_empty() {
        return Err("native edit response requires changed original exits".into());
    }
    let layers = if settings.controls.iter().any(|c| {
        matches!(
            c,
            NativeControl::MlpOutput { .. } | NativeControl::AttentionOutput { .. }
        )
    }) {
        layer_nodes(original, settings.layers)?
    } else {
        Vec::new()
    };
    let endpoints = records.iter().try_fold(0usize, |n, r| {
        n.checked_add(r.endpoint_bytes)
            .ok_or("response endpoint overflow")
    })?;
    let mut labels = Vec::new();
    let mut retained = 0usize;
    let mut planned = 0usize;
    for (case, record) in cases.iter().zip(records) {
        let literal = edits.literal_native(&case.parameter_amplitudes)?;
        let teacher_controls = controls(
            &Artifact::native(&literal)?,
            &literal,
            &layers,
            &settings.controls,
        )?;
        let teacher = intervention_program::compile(&literal, &teacher_controls)?;
        let map: BTreeMap<_, _> = originals
            .iter()
            .map(|(n, o)| (teacher.root_mapping[*o], boundaries[n]))
            .collect();
        let teacher_record = ResponseEpisode {
            label: case.label.clone(),
            inputs: teacher.family(
                family,
                &values(&literal, &teacher_controls, family.rows, case)?,
            )?,
            scored: record.scored.clone(),
            endpoint_bytes: record.endpoint_bytes,
        };
        let other = endpoints
            .checked_sub(record.endpoint_bytes)
            .and_then(|n| n.checked_add(retained))
            .ok_or("native edit capture budget overflow")?;
        let available = limit
            .checked_sub(other)
            .ok_or("native edit response labels exceed budget")?;
        let (mut captured, provenance) = native_response_targets_mapped(
            d,
            &teacher.program,
            &map,
            std::slice::from_ref(&teacher_record),
            weight,
            available,
        )?;
        planned = planned.max(
            (provenance["planned_numeric_bytes"]
                .as_u64()
                .ok_or("response plan absent")? as usize)
                .checked_add(other)
                .ok_or("native edit response plan overflow")?,
        );
        let by_candidate: BTreeMap<_, _> = boundaries.iter().map(|(n, c)| (*c, *n)).collect();
        let values = captured
            .remove(&case.label)
            .ok_or("literal response labels absent")?
            .into_iter()
            .map(|t| (by_candidate[&t.source_node], t.values))
            .collect::<BTreeMap<_, _>>();
        retained = retained
            .checked_add(values.values().map(|v| v.len() * 8).sum::<usize>())
            .ok_or("native response labels overflow")?;
        labels.push(values);
    }
    let (targets, mut provenance) =
        response_targets_from_captures(&boundaries, records, labels, weight, planned)?;
    provenance["original_to_controlled_root_mapping"] = json!(original_to_controlled);
    provenance["literal_original_observations"] = json!(originals);
    provenance["scope"]=json!("Literal stored-matrix edited original-native states under declared other controls; one pooled TRAIN scale per observable across all edit episodes. Synthetic lowering exits refused. Sampled response error, not mechanism recovery or original Local acceptance.");
    Ok((targets, provenance))
}
fn structural_run(
    d: &Device,
    settings: &Settings,
    structural: &StructuralSettings,
    native: &OperatorProgram,
    original_native: &OperatorProgram,
    layers: &[LayerNodes],
    family: &FamilyInputs,
    targets: &[Array2<f64>],
    out: &Path,
    heldout_export: Option<&Path>,
    cache: Option<&CanonicalArtifactCache>,
    costs: &mut CostCache,
    parameter_edits: Option<&native_parameter_edit::Family>,
) -> Result<(), String> {
    // These values belong to the fixed-bank baseline. Structural source has augmented
    // declarations/codewords, so its standalone replay uses the ordinary codec.
    if cache.is_some() {
        return Err("structural_search native_codec_bytes must be zero: cache source must witness the augmented declarations, not the original native artifact".into());
    }
    if structural
        .native_response_weight
        .is_some_and(|w| !w.is_finite() || w <= 0.)
    {
        return Err("native_response_weight must be finite and positive when supplied".into());
    }
    save(&out.join("ROUNDING_POLICY.json"), &rounding_policy())?;
    let native_controls = controls(
        &Artifact::native(native)?,
        native,
        layers,
        &settings.controls,
    )?;
    let controlled = intervention_program::compile(native, &native_controls)?;
    let training = if let Some(edits) = parameter_edits {
        parameter_edit_episodes(
            d,
            edits,
            &controlled,
            native,
            &native_controls,
            original_native,
            settings,
            family,
            &settings.cases,
        )?
    } else {
        CausalEpisodes::from_family(
            d,
            &controlled,
            native,
            &native_controls,
            family,
            &settings.cases,
            targets,
            settings.fixed_head_targets.as_ref(),
            settings.teacher_numeric_bytes,
        )?
    };
    save(&out.join("TEACHER_TARGETS.json"), &training.target_metadata)?;
    let mut all_inputs = training
        .metadata
        .first()
        .ok_or("no controlled episodes")?
        .inputs
        .clone();
    for e in training.metadata.iter().skip(1) {
        all_inputs = all_inputs.append(&e.inputs).map_err(|e| e.to_string())?;
    }
    let local =
        gam_mpd::acceptance::Local::new(&controlled.program, all_inputs, None, settings.context);
    let (initial_source, initial_bytes) = canonical(&Artifact::native(&controlled.program)?)?;
    std::fs::write(out.join("controlled-native.artifact"), initial_bytes)
        .map_err(|e| e.to_string())?;
    save(
        &out.join("CONTROLLED_SOURCE.json"),
        &json!({"native_parameter_edits":settings.native_parameter_edits,"original_to_edit_lowered_root_mapping":parameter_edits.map(|e|&e.node_mapping),"original_to_final_controlled_mapping":parameter_edits.map(|e|e.node_mapping.iter().map(|n|controlled.root_mapping[*n]).collect::<Vec<_>>()),"fixed_edit_direction_operator_ids":parameter_edits.map(|e|&e.direction_operators),"original_controls":settings.controls,"original_to_controlled_root_mapping":controlled.root_mapping,"control_slots":controlled.control_slots.iter().map(|s|json!({"control":s.control,"slot":s.slot,"width":s.width})).collect::<Vec<_>>(),"cases":settings.cases,"source_scope":"Original native graph augmented once with declared Raw control inputs and ordinary arithmetic. Structural candidate predicts from its own states and these controls; no per-candidate native intervention translator.","local_scope":"Measured controlled-native parent states over all declared training control episodes, RMS scale over that augmented family; not original unconditioned acceptance Dlocal.","global_scale_arithmetic":"Post-contribution multiplication, real-algebra equivalent to native shared-weight gain; not literal edited-checkpoint binary arithmetic equality.","codec_scope":"Ordinary standalone augmented artifact; original-native codec cache intentionally unsupported."}),
    )?;
    let initial_measure =
        training.measure(d, &initial_source.program, None, settings.fit.numeric_bytes)?;
    save(
        &out.join("INITIAL_TRAIN.json"),
        &serde_json::to_value(&initial_measure).map_err(|e| e.to_string())?,
    )?;
    let initial = EvaluatedArtifact {
        evaluation: structure_evaluation(
            &initial_source,
            &initial_measure,
            local
                .measure(&initial_source)?
                .worst()
                .map_or(0., |b| b.worst),
            costs,
        )?,
        artifact: initial_source,
    };
    let mut callback_index = 0usize;
    let mut search_settings = structural.settings.clone();
    if parameter_edits.is_some() || search_settings.joint_observation_places.is_some() {
        // Native observations keep their original meaning through both lowerings.
        // Synthetic edit multiplications are execution machinery, not native targets.
        let original_observations = if let Some(declared) = &search_settings.joint_observation_places {
            if declared.iter().any(|n| *n >= original_native.nodes.len()) {
                return Err("joint observation absent from original native graph".into());
            }
            declared.clone()
        } else {
            original_native.nodes.iter().enumerate().filter_map(|(n, node)| {
                matches!(node, Node::Pointwise { .. } | Node::Hadamard { .. }).then_some(n)
            }).collect()
        };
        search_settings.joint_observation_places = Some(original_observations.iter()
            .map(|n| controlled.root_mapping[parameter_edits.map_or(*n, |edits| edits.node_mapping[*n])]).collect());
    }
    save(&out.join("SEARCH_SETTINGS.json"), &serde_json::to_value(&search_settings).map_err(|e|e.to_string())?)?;
    let result = program_structure_search::search(
        &controlled.program,
        initial,
        &search_settings,
        &structural.constraints,
        |request| {
            let root = out.join(format!("structural-attempt-{:06}", request.attempt_id));
            callback_index += 1;
            std::fs::create_dir(&root).map_err(|e| e.to_string())?;
            if let Some(initialization) = request.initialization {
                save(&root.join("PARENT_INITIALIZATION.json"),
                    &serde_json::to_value(initialization).map_err(|e| e.to_string())?)?;
            }
            save(
                &root.join("DECLARATION.json"),
                &json!({"attempt_id":request.attempt_id,"parent_id":request.parent_id,"depth":request.depth,"mutation":request.mutation,"parent_evaluation":request.parent.evaluation,"trainable":request.trainable_operator_ids}),
            )?;
            std::fs::write(
                root.join("proposed.artifact"),
                request.candidate.to_bytes()?,
            )
            .map_err(|e| e.to_string())?;
            let attempt = (|| -> Result<EvaluatedArtifact, String> {
                let mut candidate = request.candidate.clone();
                if matches!(request.mutation, Mutation::ExactExtract { .. })
                    && !request.trainable_operator_ids.is_empty()
                {
                    return Err("exact extraction must remain fit-free".into());
                }
                if parameter_edits.is_some_and(|e| {
                    request
                        .trainable_operator_ids
                        .iter()
                        .any(|id| e.direction_operators.contains(id))
                }) {
                    return Err("fixed native edit directions cannot be fitted".into());
                }
                let mut fitted_responses = None;
                if !request.trainable_operator_ids.is_empty() {
                    let fitted = if let Some(weight) = structural.native_response_weight {
                        let (response_targets, provenance) = if let Some(edits) = parameter_edits {
                            parameter_edit_response_targets(
                                d,
                                edits,
                                &controlled,
                                original_native,
                                settings,
                                family,
                                &settings.cases,
                                &candidate,
                                &training.metadata,
                                weight,
                                settings.teacher_numeric_bytes,
                            )?
                        } else {
                            native_response_targets_records(
                                d,
                                &controlled.program,
                                &candidate,
                                &training.metadata,
                                weight,
                                settings.teacher_numeric_bytes,
                            )?
                        };
                        save(&root.join("NATIVE_RESPONSE_TARGETS.json"), &provenance)?;
                        let fitted = training.fit(
                            d,
                            &candidate.program,
                            Some(&response_targets),
                            request.trainable_operator_ids,
                            settings.fit.clone(),
                        )?;
                        fitted_responses = Some(response_targets);
                        fitted
                    } else {
                        training.fit(
                            d,
                            &candidate.program,
                            None,
                            request.trainable_operator_ids,
                            settings.fit.clone(),
                        )?
                    };
                    save(
                        &root.join("FIT.json"),
                        &serde_json::to_value(&fitted.report).map_err(|e| e.to_string())?,
                    )?;
                    candidate.program = fitted.program;
                }
                let (saved, bytes) = canonical(&candidate)?;
                if let Some(edits) = parameter_edits {
                    for id in &edits.direction_operators {
                        if !same_dense(
                            saved
                                .program
                                .operators
                                .get(*id)
                                .ok_or("saved native edit direction absent")?,
                            &request.parent.artifact.program.operators[*id],
                        ) {
                            return Err("saved native edit direction changed".into());
                        }
                    }
                }
                std::fs::write(root.join("program.artifact"), bytes).map_err(|e| e.to_string())?;
                let local_measure = local.measure(&saved)?;
                let local_error = local_measure.worst().map_or(0., |b| b.worst);
                if !local_error.is_finite() {
                    return Err("nonfinite structural Local measurement".into());
                }
                if request.trainable_operator_ids.is_empty() {
                    if let Some(limit) = structural
                        .constraints
                        .max_local_errors
                        .iter()
                        .find(|m| m.name == "sampled_d_local")
                    {
                        if local_error > limit.value {
                            save(
                                &root.join("LOCAL_SCREEN.json"),
                                &json!({
                                    "status":"local_screen_rejected", "local":local_measure,
                                    "sampled_d_local":local_error, "constraint":limit,
                                    "run":"not_measured", "scope":"decoded augmented control-family Local proposal constraint",
                                    "artifact_sha256":sha256(&root.join("program.artifact"))?
                                }),
                            )?;
                            return Err("local_screen_rejected: decoded augmented Local exceeds declared constraint; Run not measured".into());
                        }
                    }
                }
                let measured =
                    training.measure(d, &saved.program, None, settings.fit.numeric_bytes)?;
                save(
                    &root.join("TRAIN.json"),
                    &serde_json::to_value(&measured).map_err(|e| e.to_string())?,
                )?;
                if let Some(responses) = &fitted_responses {
                    let supervised = training.measure(
                        d,
                        &saved.program,
                        Some(responses),
                        settings.fit.numeric_bytes,
                    )?;
                    save(
                        &root.join("TRAIN_RESPONSES.json"),
                        &json!({"measurement":supervised,"pure_kl_report":"TRAIN.json","native_labels_reused":true,"target_backend":training.target_metadata}),
                    )?;
                }
                let evaluation = structure_evaluation(&saved, &measured, local_error, costs)?;
                save(
                    &root.join("STATUS.json"),
                    &json!({"status":"training_measured","trainable":request.trainable_operator_ids,"evaluation":evaluation,"artifact_sha256":sha256(&root.join("program.artifact"))?}),
                )?;
                Ok(EvaluatedArtifact {
                    artifact: saved,
                    evaluation,
                })
            })();
            if let Err(error) = &attempt {
                save(
                    &root.join("STATUS.json"),
                    &json!({"status":if root.join("LOCAL_SCREEN.json").exists(){"local_screen_rejected"}else{"unresolved"},"error":error,"run":if root.join("LOCAL_SCREEN.json").exists(){Some("not_measured")}else{None}}),
                )?;
            }
            attempt
        },
    )?;
    let metadata:Vec<_>=result.candidates.iter().map(|c|json!({"id":c.id,"parent_id":c.parent_id,"depth":c.depth,"mutation":c.mutation,"evaluation":c.evaluated.evaluation,"artifact_path":format!("structure-{}/program.artifact",c.id)})).collect();
    save(
        &out.join("STRUCTURAL_SEARCH.json"),
        &json!({"report":result.report,"frontier":result.frontier,"beam":result.beam,"candidates":metadata}),
    )?;
    let (frozen, expression_hypotheses) = frozen_structural_ids(
        &result.frontier,
        &result.report.admitted_candidates,
        structural.max_expression_evaluations,
    );
    let mut persisted: Vec<_> = result.candidates.iter().map(|c| c.id).collect();
    for id in &frozen {
        if !persisted.contains(id) {
            persisted.push(*id);
        }
    }
    for id in &persisted {
        let record = result
            .report
            .admitted_candidates
            .iter()
            .find(|r| r.id == *id)
            .ok_or("admitted candidate metadata missing")?;
        let root = out.join(format!("structure-{id}"));
        std::fs::create_dir(&root).map_err(|e| e.to_string())?;
        std::fs::hard_link(
            structural_artifact_path(out, &result, *id)?,
            root.join("program.artifact"),
        )
        .map_err(|e| e.to_string())?;
        save(
            &root.join("LINEAGE.json"),
            &json!({"record":record,"artifact_sha256":sha256(&root.join("program.artifact"))?}),
        )?;
    }
    save(
        &out.join("FROZEN_EVALUATION_IDS.json"),
        &json!({"ids":frozen,"compression_frontier":result.frontier,"expression_hypotheses":expression_hypotheses,"max_expression_evaluations":structural.max_expression_evaluations,"policy":"Native baseline plus training compression frontier plus first declared-N admitted direct-expression, shared-DAG, or learned-DAG hypotheses under one combined limit in deterministic admission order, including dominated/evicted states. Frozen before heldout access.","scope":"Hypothesis admission is not discovery or understanding; augmented-source sampled Local and operational causal KL."}),
    )?;
    let mut heldout_rows = Vec::new();
    if let (Some(eval), Some(export)) = (&settings.evaluation, heldout_export) {
        if sha256(&export.join("export.json"))? != eval.export_sha256 {
            return Err("heldout export SHA mismatch".into());
        }
        let imported = import_language_model(export, eval.sequences, settings.context)?;
        let heldout_native = split_sites(&imported.program)?;
        if !same_native(original_native, &heldout_native) {
            return Err("heldout original model differs".into());
        }
        disjoint_token_sequences(family, &imported.contract.family, settings.context)?;
        let cases = eval.cases.as_deref().unwrap_or(&settings.cases);
        let (labels, plan) = if settings.fixed_head_targets.is_some() || parameter_edits.is_some() {
            (Vec::new(), 0)
        } else {
            family_teacher_targets(
                d,
                original_native,
                layers,
                &settings.controls,
                &imported.contract.family,
                cases,
                None,
                settings.fit.numeric_bytes,
                settings.teacher_numeric_bytes,
            )?
        };
        let eval_episodes = if let Some(edits) = parameter_edits {
            parameter_edit_episodes(
                d,
                edits,
                &controlled,
                native,
                &native_controls,
                original_native,
                settings,
                &imported.contract.family,
                cases,
            )?
        } else {
            CausalEpisodes::from_family(
                d,
                &controlled,
                native,
                &native_controls,
                &imported.contract.family,
                cases,
                &labels,
                settings.fixed_head_targets.as_ref(),
                settings.teacher_numeric_bytes,
            )?
        };
        save(
            &out.join("HELDOUT_TEACHER_TARGETS.json"),
            &eval_episodes.target_metadata,
        )?;
        save(
            &out.join("HELDOUT_PROVENANCE.json"),
            &json!({"native":imported.record,"teacher_numeric_bytes":if settings.fixed_head_targets.is_some(){Value::Null}else{json!(plan)},"target_metadata":eval_episodes.target_metadata,"scope":"fit-disjoint frozen panel; augmented source control slots reused, no heldout fitting/reselection"}),
        )?;
        for id in &frozen {
            let candidate = result
                .report
                .admitted_candidates
                .iter()
                .find(|c| c.id == *id)
                .ok_or("frozen structural ID missing")?;
            let bytes = std::fs::read(out.join(format!("structure-{id}/program.artifact")))
                .map_err(|e| e.to_string())?;
            let lineage: Value = serde_json::from_slice(
                &std::fs::read(out.join(format!("structure-{id}/LINEAGE.json")))
                    .map_err(|e| e.to_string())?,
            )
            .map_err(|e| e.to_string())?;
            if lineage["artifact_sha256"].as_str()
                != Some(&sha256(
                    &out.join(format!("structure-{id}/program.artifact")),
                )?)
            {
                return Err("frozen structural saved-byte SHA differs".into());
            }
            let saved = Artifact::from_bytes(&bytes, &controlled.program.declarations)?;
            if saved.to_bytes()? != bytes {
                return Err("independent structural saved replay differs".into());
            }
            let c32 = structural_cost(&saved, costs)?.total();
            if c32 as f64 != candidate.evaluation.description_bits {
                return Err("saved structural C32 differs from training".into());
            }
            let measurement =
                eval_episodes.measure(d, &saved.program, None, settings.fit.numeric_bytes)?;
            heldout_rows.push(json!({"id":id,"measurement":measurement,"c32":c32}));
        }
    }
    save(
        &out.join("REPORT.json"),
        &json!({"structural_report":result.report,"rounding_policy":rounding_policy(),"retained_candidates":metadata,"frozen_ids":frozen,"compression_frontier":result.frontier,"expression_hypotheses":expression_hypotheses,"heldout":heldout_rows,"callback_calls":callback_index,"scope":"Bounded measured structural proposal search on explicitly intervention-conditioned native graph, not automatic understanding; no per-candidate native reexecution inside predictor; native computation outside affected rule uses remains fixed"}),
    )
}

struct JointLocalTargets {
    inputs: Vec<Array2<f64>>,
    targets: Array2<f64>,
    groups: Vec<gam_mpd::resident_rule_fit::OutputGroup>,
}

/// Training-only native values. The returned program still computes every value itself.
fn capture_joint_local_targets(
    d: &Device,
    parent: &OperatorProgram,
    input: usize,
    outputs: &[usize],
    family: &FamilyInputs,
    numeric_bytes: usize,
) -> Result<JointLocalTargets, String> {
    let last = outputs.iter().copied().chain(std::iter::once(input)).max()
        .ok_or("joint local observations absent")?;
    let mut prefix = parent.clone();
    prefix.nodes.truncate(last + 1);
    prefix.output = last;
    let resident = DeviceProgram::compile_values_bounded(d, &prefix, numeric_bytes)?;
    let planned = resident.bytes_per_row().checked_mul(family.rows)
        .and_then(|n| n.checked_mul(2))
        .and_then(|n| resident.operator_numeric_bytes().ok().and_then(|p| n.checked_add(p)))
        .ok_or("joint local capture size overflow")?;
    if planned > numeric_bytes {
        return Err(format!("joint local capture numeric plan {planned} exceeds {numeric_bytes}"));
    }
    let trace = resident.forward(family)?;
    let inputs = vec![d.download(trace.value(input)?).map_err(|e| e.to_string())?];
    let panels = outputs.iter().map(|n| d.download(trace.value(*n)?)
        .map_err(|e| e.to_string())).collect::<Result<Vec<_>, String>>()?;
    let targets = ndarray::concatenate(ndarray::Axis(1),
        &panels.iter().map(|p| p.view()).collect::<Vec<_>>()).map_err(|e| e.to_string())?;
    let mut start = 0;
    let groups = panels.iter().enumerate().map(|(i, p)| {
        let end = start + p.ncols();
        let group = gam_mpd::resident_rule_fit::OutputGroup {
            label: if i == 0 { "clean".into() } else { format!("response{}", i - 1) },
            start, end,
        };
        start = end;
        group
    }).collect();
    Ok(JointLocalTargets { inputs, targets, groups })
}

fn prefit_joint_linear(
    d: &Device,
    program: &OperatorProgram,
    trainable: &[usize],
    targets: &JointLocalTargets,
    numeric_bytes: usize,
) -> Result<(OperatorProgram, Value), String> {
    let measure = |p: &OperatorProgram| gam_mpd::resident_rule_fit::measure_grouped(
        d, p, &targets.inputs, &targets.targets, &targets.groups, numeric_bytes, 128);
    let initial = measure(program)?;
    let fit = gam_mpd::program_linear_fit::prefit(program, &targets.inputs,
        &targets.targets, trainable, Default::default(), numeric_bytes)?;
    let proposed = measure(&fit.program)?;
    let accepted = proposed.maximum < initial.maximum;
    let report = json!({"initial":initial,"proposed":proposed,"accepted":accepted,
        "linear_fit":fit.report,"selection":"Strict improvement of full TRAIN normalized maximum; no validation used. Fixed native input states only; no hybrid-state refresh yet. Local least squares is a proposal, not the acceptance objective."});
    Ok((if accepted { fit.program } else { program.clone() }, report))
}

fn validate_joint_configuration(settings: &Settings) -> Result<(), String> {
    if settings.joint_response_search.is_some()
        && (settings.width != 0
            || !settings.expression_ids.is_empty()
            || settings.require_interior_learned)
    {
        return Err("joint_response_search supplies its own bounded inventory and widths; legacy width, expression IDs and require_interior_learned must be absent".into());
    }
    if settings.joint_response_search.is_some() {
        let config = settings
            .down_edit_family
            .as_ref()
            .ok_or("joint response search requires down_edit_family")?;
        if settings.uses != vec![config.layer]
            || settings.structural_search.is_some()
            || settings.native_initialization
            || settings.frozen_shared_body.is_some()
        {
            return Err("joint response search supports exactly the declared down layer, without structural search, native initialization or body transfer".into());
        }
    }
    Ok(())
}
fn run() -> Result<(), String> {
    let mut args: Vec<_> = std::env::args().skip(1).collect();
    let profile = args.iter().any(|a| a == "--profile");
    args.retain(|a| a != "--profile");
    gam_gpu::trace::time_stages(profile);
    if args.len() != 4 && args.len() != 5 {
        return Err("EXPORT SETTINGS.json FRESH_OUT host|cuda [HELDOUT_EXPORT] [--profile]".into());
    }
    let export = Path::new(&args[0]);
    let config_path = Path::new(&args[1]);
    let out = Path::new(&args[2]);
    let config_bytes = std::fs::read(config_path).map_err(|e| e.to_string())?;
    let settings: Settings = serde_json::from_slice(&config_bytes).map_err(|e| e.to_string())?;
    validate_joint_configuration(&settings)?;
    validate_parameter_edit_configuration(&settings)?;
    if settings.evaluation.is_some() != (args.len() == 5) {
        return Err(
            "HELDOUT_EXPORT argument required exactly when evaluation config is present".into(),
        );
    }
    if settings.evaluation.as_ref().is_some_and(|v| {
        v.sequences == 0
            || v.export_sha256.len() != 64
            || !v.export_sha256.bytes().all(|b| b.is_ascii_hexdigit())
    }) {
        return Err("positive evaluation sequences and 64-digit export SHA required".into());
    }
    if sha256(&export.join("export.json"))? != settings.export_sha256 {
        return Err("native export identity mismatch".into());
    }
    if out.exists() {
        return Err("fresh output directory required".into());
    }
    if settings.layers == 0
        || (settings.structural_search.is_none()
            && (settings.uses.is_empty()
                || (settings.joint_response_search.is_none() && settings.width == 0)))
        || settings.context == 0
        || settings.sequences == 0
        || settings.uses.iter().copied().collect::<BTreeSet<_>>().len() != settings.uses.len()
        || settings.uses.iter().any(|&i| i >= settings.layers)
        || (settings.controls.is_empty()
            && settings.down_edit_family.is_none()
            && settings.native_parameter_edits.is_none())
    {
        return Err("positive dimensions, unique uses/cases and explicit clean plus intervention cases required".into());
    }
    if settings.structural_search.is_some()
        && (settings.down_edit_family.is_some()
            || settings.frozen_shared_body.is_some()
            || settings.native_initialization
            || settings.native_codec_bytes != 0
            || !settings.uses.is_empty()
            || settings.width != 0
            || !settings.expression_ids.is_empty())
    {
        return Err("structural_search supplies automatic regions/library moves; manual uses/width/expression IDs and down/transfer/native-initialization/native-codec modes cannot be combined".into());
    }
    let direction_count = settings
        .down_edit_family
        .as_ref()
        .map_or(0, |d| d.directions.len());
    validate_cases(
        &settings.cases,
        settings.controls.len(),
        direction_count,
        true,
    )?;
    if let Some(eval) = &settings.evaluation {
        if let Some(cases) = &eval.cases {
            validate_cases(cases, settings.controls.len(), direction_count, false)?;
        }
    }
    if let Some(down) = &settings.down_edit_family {
        if down
            .directions
            .iter()
            .flat_map(|d| d.output.iter().chain(&d.hidden))
            .any(|v| !v.is_finite() || f64::from(*v as f32) != *v)
        {
            return Err("declared direction entries must be exact finite f32 values, supplied as their f64 JSON values".into());
        }
        if down.directions.is_empty()
            || !settings.uses.contains(&down.layer)
            || settings.native_initialization
            || settings.frozen_shared_body.is_some()
        {
            return Err("down family requires a replaced declared layer, nonempty directions, no native initialization or frozen-body transfer".into());
        }
        if !settings
            .cases
            .iter()
            .flat_map(|c| &c.down_amplitudes)
            .any(|a| *a > 0.)
            || !settings
                .cases
                .iter()
                .flat_map(|c| &c.down_amplitudes)
                .any(|a| *a < 0.)
        {
            return Err(
                "down family requires prespecified positive AND negative nonzero training edits"
                    .into(),
            );
        }
    }
    if settings.native_initialization && settings.frozen_shared_body.is_some() {
        return Err("native initialization and frozen shared body are mutually exclusive".into());
    }
    let inventory = composed_rule_search::enumerate(&settings.grammar)?;
    let sharing =
        if settings.structural_search.is_some() || settings.joint_response_search.is_some() {
            composed_rule_search::interior_learned_selection(&inventory)
        } else {
            selection(
                &inventory,
                &settings.expression_ids,
                settings.require_interior_learned,
            )?
        };
    if settings.frozen_shared_body.is_some() && settings.expression_ids.len() != 1 {
        return Err("transfer config requires one frozen expression ID".into());
    }
    let export_record: Value = serde_json::from_slice(
        &std::fs::read(export.join("export.json")).map_err(|e| e.to_string())?,
    )
    .map_err(|e| e.to_string())?;
    let checkpoint = export_record["source"]["checkpoint_sha256"]
        .as_str()
        .or_else(|| export_record["source"]["weights_sha256"].as_str())
        .ok_or("native checkpoint/weights SHA absent")?;
    if let (Some(a), Some(b)) = (
        export_record["source"]["checkpoint_sha256"].as_str(),
        export_record["source"]["weights_sha256"].as_str(),
    ) {
        if a != b {
            return Err("conflicting native weight lineage aliases".into());
        }
    }
    let frozen_body = settings
        .frozen_shared_body
        .as_ref()
        .map(|spec| {
            load_body(
                spec,
                &inventory.expressions[settings.expression_ids[0]],
                settings.width,
                &settings.uses,
                checkpoint,
            )
        })
        .transpose()?;
    let d = match args[3].as_str() {
        "host" => Device::host(),
        "cuda" => Device::accelerator(GpuPolicy::Required)
            .map_err(|e| e.to_string())?
            .ok_or("CUDA required")?,
        _ => return Err("host|cuda backend required".into()),
    };
    if args[3] == "cuda" && (d.is_host() || !d.float64()) {
        return Err("real float64 accelerator required".into());
    }
    if settings
        .fixed_head_targets
        .as_ref()
        .is_some_and(|s| s.tile_rows == 0)
    {
        return Err("fixed_head_targets tile_rows must be positive".into());
    }
    if settings.fixed_head_targets.is_some()
        && settings.structural_search.is_none()
        && settings.down_edit_family.is_none()
    {
        return Err("fixed_head_targets requires structural_search or down_edit_family".into());
    }
    let started = Instant::now();
    let imported = import_language_model(export, settings.sequences, settings.context)?;
    let original_native = split_sites(&imported.program)?;
    let original_layers = layer_nodes(&original_native, settings.layers)?;
    let parameter_edits = settings
        .native_parameter_edits
        .as_ref()
        .map(|c| build_parameter_edits(&original_native, c))
        .transpose()?;
    if let Some(config) = &settings.native_parameter_edits {
        let original_controls = controls(
            &Artifact::native(&original_native)?,
            &original_native,
            &original_layers,
            &settings.controls,
        )?;
        reject_native_edit_target_gain(config.target_operator, &original_controls)?;
    }
    let down = settings
        .down_edit_family
        .as_ref()
        .map(|config| {
            let layer = original_layers
                .get(config.layer)
                .ok_or("down family layer absent")?;
            let directions: Vec<_> = config
                .directions
                .iter()
                .map(|d| Direction {
                    output: Array1::from(d.output.clone()),
                    hidden: Array1::from(d.hidden.clone()),
                })
                .collect();
            down_edit_family::build(&original_native, layer.normed, layer.mlp, &directions)
        })
        .transpose()?;
    let native = parameter_edits.as_ref().map_or_else(
        || {
            down.as_ref()
                .map_or_else(|| original_native.clone(), |f| f.program.clone())
        },
        |f| f.compiled.program.clone(),
    );
    let layers = parameter_edits.as_ref().map_or_else(
        || {
            down.as_ref().map_or_else(
                || original_layers.clone(),
                |f| remap_layers(&original_layers, &f.node_mapping),
            )
        },
        |f| remap_layers(&original_layers, &f.node_mapping),
    );
    let base = Artifact::native(&native)?;
    let cache_started = Instant::now();
    let native_codec = if settings.native_codec_bytes == 0 {
        None
    } else {
        Some(CanonicalArtifactCache::new(
            &base,
            settings.native_codec_bytes,
        )?)
    };
    let native_codec_initialization_seconds = cache_started.elapsed().as_secs_f64();
    let family = &imported.contract.family;
    let target_controls = controls(&base, &native, &layers, &settings.controls)?;
    let (targets, teacher_plan) =
        if settings.fixed_head_targets.is_some() || parameter_edits.is_some() {
            (Vec::new(), 0)
        } else {
            family_teacher_targets(
                &d,
                &original_native,
                &original_layers,
                &settings.controls,
                family,
                &settings.cases,
                down.as_ref(),
                settings.fit.numeric_bytes,
                settings.teacher_numeric_bytes,
            )?
        };
    let native_teacher_seconds = started.elapsed().as_secs_f64();
    let mut stage_seconds = BTreeMap::<String, f64>::new();
    stage_seconds.insert("native_import_and_teachers".into(), native_teacher_seconds);
    stage_seconds.insert(
        "native_codec_initialization".into(),
        native_codec_initialization_seconds,
    );
    let mut use_specs = settings
        .uses
        .iter()
        .map(|&i| {
            Ok(UseSpec {
                input_width: native
                    .node_interface(layers[i].normed)
                    .map_err(|e| e.to_string())?
                    .width(),
                output_width: native
                    .node_interface(layers[i].mlp)
                    .map_err(|e| e.to_string())?
                    .width(),
            })
        })
        .collect::<Result<Vec<_>, String>>()?;
    if let Some(config) = &settings.down_edit_family {
        let width = native
            .node_interface(layers[config.layer].normed)
            .map_err(|e| e.to_string())?
            .width();
        use_specs.extend((0..config.directions.len()).map(|_| UseSpec {
            input_width: width,
            output_width: 1,
        }));
    }
    std::fs::create_dir_all(out).map_err(|e| e.to_string())?;
    if let Some(config) = &settings.down_edit_family {
        if !config.response_weight.is_finite() || config.response_weight <= 0. {
            return Err("down response_weight must be finite and positive".into());
        }
        let amplitudes = Array2::from_shape_vec(
            (settings.cases.len(), config.directions.len()),
            settings
                .cases
                .iter()
                .flat_map(|c| c.down_amplitudes.iter().copied())
                .collect(),
        )
        .map_err(|e| e.to_string())?;
        let report = down_edit_family::control_design(&amplitudes, 1e-12)?;
        save(
            &out.join("DOWN_CONTROL_DESIGN.json"),
            &json!({"report":report,"relative_tolerance":1e-12,"scope":"Edit-coordinate diagnostic only; cases may have different upstream states/gains. Deficiency does not reject independently supervised coefficient functions."}),
        )?;
    }

    std::fs::write(out.join("SETTINGS.json"), &config_bytes).map_err(|e| e.to_string())?;
    save(
        &out.join("INVENTORY.json"),
        &serde_json::to_value(&inventory).map_err(|e| e.to_string())?,
    )?;
    save(
        &out.join("PROVENANCE.json"),
        &json!({"native":imported.record,"export_sha256":settings.export_sha256,
        "settings_sha256":sha256(config_path)?,"binary_sha256":sha256(&std::env::current_exe().map_err(|e|e.to_string())?)?,
        "rows_per_episode":family.rows,"controls":settings.controls,"cases":settings.cases,
        "scope":"training proposal search with optional frozen heldout measurement; no acceptance claim; declared boundary masks, retained shared-weight gains, and optional finite native down-weight edit family",
        "fitting_objective":if down.is_some() || settings.structural_search.as_ref().is_some_and(|s|s.native_response_weight.is_some()){"maximum named-group mean of episode KL plus weighted direct native response loss"}else{"maximum named-group mean of episode mean teacher-to-candidate KL"},"ranking_objective":if settings.joint_response_search.is_some(){"Pareto in saved-f32 pure KL, C32 and maximum sampled episode/observable mean normalized squared response error; all causal sequence rows scored"}else{"pure saved-f32 teacher-to-candidate KL; response loss reported separately; all causal sequence rows scored"},
        "native_supervision":if settings.structural_search.as_ref().is_some_and(|s|s.native_response_weight.is_some()) {"native logits plus controlled native observable responses"}else if down.is_some(){"native logits plus direct scalar down-edit response coefficients; candidate runs its own complete states"}else{"native logits only; candidate runs its complete autonomous states"},
        "teacher_planned_numeric_bytes":if settings.fixed_head_targets.is_some(){Value::Null}else{json!(teacher_plan)},"fixed_head_targets":settings.fixed_head_targets,"teacher_plan_scope":"compact target/cache metadata in TEACHER_TARGETS.json; no full-logit target intermediate in compact mode",
        "native_codec_bytes":settings.native_codec_bytes,"native_codec_initialization_seconds":native_codec_initialization_seconds,"native_codec_preflight":native_codec.as_ref().map(|c|c.preflight()),"native_codec_stats":native_codec.as_ref().map(|c|c.stats()),
        "native_codec_budget_scope":"packed native codewords plus cached decoded numeric buffers only; excludes caller-owned source arrays, per-operator construction/lattice temporaries, full messages, label/execution buffers, metadata and allocator/library overhead; construction excluded from warm timings",
        "literal_down_arithmetic":"binary64 stored D updates in declared order D += a*u*v; original graph executes updated matrix; no float32 checkpoint-store or augmented-branch bitwise equivalence claim",
        "joint_response_search":settings.joint_response_search,"joint_search_scope":"bounded enumerated learned multi-output DAG; exactly one declared down layer; shared computations are actual nodes; exact corresponding parent subtrees inherit coefficients with per-owner provenance; changed pieces require fitting; inventory truncation disclosed in JOINT_INVENTORY.json",
        "native_parameter_edits":settings.native_parameter_edits,"native_parameter_edit_scope":"Finite additive dense directions lowered as Raw control contributions before downstream nonlinearities; independent literal edited original-native teachers; structural_search only. No mechanism recovery claim, no bit-parity between augmented and literal arithmetic; saved source/directions use paid ordinary f32 codec.",
        "down_edit_family":settings.down_edit_family,"down_family_scope":"restricted declared native down-weight directions; teachers run literal edited matrices, predictor runs own response functions; fixed directions serialized and excluded from fitting; no arbitrary native-edit mapping; body export/transfer disabled in this mode",
        "control_scope":"activation boundary scaling is not a global weight edit; retained_operator gain affects all its invocations; no mapping claimed for removed internal coordinates",
        "weight_gain_arithmetic":"gain applied to every computed contribution, algebraically equivalent to scaling the shared operator; not a bit-exact claim about rounding edited checkpoint literals before GEMM",
        "cost_scope":"full saved base-program C32 with native bindings; declared controls are external test operations, not an encoded general weight-intervention translator",
        "numerical_scope":"operational float64 KL proposal scores, not certified enclosures",
        "training_data_scope":"previously available export; no new untouched confirmation panel"}),
    )?;
    save(
        &out.join("SHARING_ANALYSIS.json"),
        &json!({"require_interior_learned":settings.require_interior_learned,"selected_expression_ids":settings.expression_ids,"inventory_selection":sharing,"excluded_scope":"baseline IDs remain available in separate default runs; default enumeration is unchanged"}),
    )?;
    if let Some((_, record)) = &frozen_body {
        save(&out.join("TRANSFER_SOURCE.json"), record)?;
    }
    let mut costs = CostCache::default();
    if let Some(structural) = &settings.structural_search {
        return structural_run(
            &d,
            &settings,
            structural,
            &native,
            &original_native,
            &layers,
            family,
            &targets,
            out,
            args.get(4).map(|p| Path::new(p)),
            native_codec.as_ref(),
            &mut costs,
            parameter_edits.as_ref(),
        );
    }
    let (saved_native, native_bytes, native_canonical_phases) =
        canonical_using(&base, native_codec.as_ref())?;
    save(
        &out.join("NATIVE_CANONICAL_TIMINGS.json"),
        &native_canonical_phases,
    )?;
    std::fs::write(out.join("native.artifact"), &native_bytes).map_err(|e| e.to_string())?;
    save(&out.join("NATIVE_CONTROL_MAP.json"), &{
        let mut map = control_map(&target_controls, &settings.controls);
        map["down_edit_family"] =
            serde_json::to_value(&settings.down_edit_family).map_err(|e| e.to_string())?;
        map
    })?;
    let native_cost = structural_cost(&saved_native, &mut costs)?.total();
    let saved_native_graph =
        intervention_program::compile(&saved_native.program, &target_controls)?;
    let native_episodes = CausalEpisodes::standalone(
        &d,
        &saved_native_graph,
        &saved_native.program,
        &target_controls,
        family,
        &settings.cases,
        &targets,
        settings.fixed_head_targets.as_ref(),
        settings.teacher_numeric_bytes,
        down.as_ref(),
        &original_layers,
        &settings.controls,
    )?;
    save(
        &out.join("TEACHER_TARGETS.json"),
        &native_episodes.target_metadata,
    )?;
    let native_measure = native_episodes.measure(
        &d,
        &saved_native_graph.program,
        None,
        settings.fit.numeric_bytes,
    )?;
    save(
        &out.join("NATIVE_TRAIN.json"),
        &serde_json::to_value(&native_measure).map_err(|e| e.to_string())?,
    )?;

    let native_teacher_cache = if native_episodes.compact.is_some() {
        Some(native_episodes)
    } else {
        None
    };
    let mut journal =
        std::fs::File::create(out.join("journal.jsonl")).map_err(|e| e.to_string())?;
    let mut rows = vec![
        json!({"id":"native","c32":native_cost,"training_kl":native_measure.objective,"status":"training_measured","artifact_sha256":sha256(&out.join("native.artifact"))?,"control_map_sha256":sha256(&out.join("NATIVE_CONTROL_MAP.json"))?}),
    ];
    let joint_inventory = if let Some(search) = &settings.joint_response_search {
        let layer = settings.down_edit_family.as_ref().unwrap().layer;
        let inputs = vec![gam_mpd::operator_program::Interface::native(
            native
                .node_interface(layers[layer].normed)
                .map_err(|e| e.to_string())?
                .width(),
        )
        .map_err(|e| e.to_string())?];
        let mut outputs = vec![gam_mpd::operator_program::Interface::native(
            native
                .node_interface(layers[layer].mlp)
                .map_err(|e| e.to_string())?
                .width(),
        )
        .map_err(|e| e.to_string())?];
        outputs.extend(
            (0..direction_count).map(|_| gam_mpd::operator_program::Interface::native(1).unwrap()),
        );
        let inventory = program_learned_dag::enumerate_interfaces(&inputs, &outputs, search)?;
        save(
            &out.join("JOINT_INVENTORY.json"),
            &serde_json::to_value(&inventory).map_err(|e| e.to_string())?,
        )?;
        Some((inputs, outputs, inventory))
    } else {
        None
    };
    let mut joint_local_targets = None;
    let candidate_ids: Vec<_> = joint_inventory.as_ref().map_or_else(
        || settings.expression_ids.clone(),
        |(_, _, i)| (0..i.proposals.len()).collect(),
    );
    for &id in &candidate_ids {
        for shared in [true, false] {
            if !shared && (use_specs.len() == 1 || joint_inventory.is_some()) {
                continue;
            }
            let arm = if joint_inventory.is_some() {
                "learned"
            } else if shared {
                "shared"
            } else {
                "untied"
            };
            let name = if joint_inventory.is_some() {
                format!("joint{id}-{arm}")
            } else {
                format!("expression{id}-{arm}")
            };
            let root = out.join(&name);
            std::fs::create_dir(&root).map_err(|e| e.to_string())?;
            save(
                &root.join("DECLARATION.json"),
                &json!({"expression_id":id,"expression":if let Some((_,_,j))=&joint_inventory {serde_json::to_value(&j.proposals[id]).map_err(|e|e.to_string())?}else{serde_json::to_value(&inventory.expressions[id]).map_err(|e|e.to_string())?},"sharing_analysis":if joint_inventory.is_some(){Value::Null}else{serde_json::to_value(&sharing.analyses[id]).map_err(|e|e.to_string())?},"arm":arm,"uses":settings.uses,"response_uses":direction_count,"body_export_supported":down.is_none(),"transfer":frozen_body.is_some(),"freeze_policy":if frozen_body.is_some() && shared {"body frozen; new maps only"}else if frozen_body.is_some(){"same initial body; independent per-use body adaptation"}else{"all proposal coefficients trainable"}}),
            )?;
            let attempt = (|| -> Result<Value, String> {
                let compile = if shared {
                    composed_rule_search::compile
                } else {
                    composed_rule_search::compile_untied
                };
                let mut joint = joint_inventory
                    .as_ref()
                    .map(|(inputs, outputs, inventory)| {
                        let parent = down.as_ref().ok_or("joint response parent absent")?;
                        let layer = settings.down_edit_family.as_ref().ok_or("joint response layer absent")?.layer;
                        let parent_outputs = std::iter::once(parent.clean_output)
                            .chain(parent.response_nodes.iter().copied()).collect::<Vec<_>>();
                        program_learned_dag::compile_program_inheriting(
                            inputs,
                            outputs,
                            &inventory.proposals[id].expressions,
                            settings.joint_response_search.as_ref().unwrap(),
                            &parent.program,
                            &[layers[layer].normed],
                            &parent_outputs,
                        )
                    })
                    .transpose()?;
                if let Some(joint) = &mut joint {
                    save(&root.join("PARENT_INITIALIZATION.json"),
                        &serde_json::to_value(&joint.initialization).map_err(|e| e.to_string())?)?;
                    if joint.initialization.random_elements != 0 {
                        if joint_local_targets.is_none() {
                            let parent = down.as_ref().ok_or("joint parent absent")?;
                            let layer = settings.down_edit_family.as_ref().ok_or("joint layer absent")?.layer;
                            let outputs = std::iter::once(parent.clean_output)
                                .chain(parent.response_nodes.iter().copied()).collect::<Vec<_>>();
                            let clean_inputs = parent.inputs(family, &vec![0.; direction_count])?;
                            joint_local_targets = Some(capture_joint_local_targets(&d, &parent.program,
                                layers[layer].normed, &outputs, &clean_inputs, settings.fit.numeric_bytes)?);
                        }
                        let (program, report) = prefit_joint_linear(&d, &joint.program,
                            &joint.trainable_operator_ids, joint_local_targets.as_ref().ok_or("local target capture absent")?,
                            settings.fit.numeric_bytes)?;
                        joint.program = program;
                        save(&root.join("LOCAL_LINEAR_PREFIT.json"), &report)?;
                    }
                }
                let proposal = if joint.is_none() {
                    let mut proposal = compile(
                        &inventory.expressions[id],
                        settings.width,
                        &use_specs,
                        settings.seed,
                    )?;
                    if settings.native_initialization {
                        proposal = gam_mpd::native_mlp_initialization::initialize(
                            &proposal,
                            &inventory.expressions[id],
                            &native,
                            &native_uses(&native, &layers, &settings.uses)?,
                        )?;
                    }
                    if let Some((source, _)) = &frozen_body {
                        copy_body(&source.program, &mut proposal.program)?;
                    }
                    Some(proposal)
                } else {
                    None
                };
                let proposal_program = joint
                    .as_ref()
                    .map(|j| &j.program)
                    .unwrap_or_else(|| &proposal.as_ref().unwrap().program);
                let graft_started = Instant::now();
                let mut candidate = base.clone();
                let mut response_nodes = Vec::new();
                for (slot, &layer) in settings.uses.iter().enumerate() {
                    let clean = if joint.is_none() {
                        Some(composed_rule_search::function(
                            proposal.as_ref().unwrap(),
                            slot,
                        )?)
                    } else {
                        None
                    };
                    if settings
                        .down_edit_family
                        .as_ref()
                        .is_some_and(|config| config.layer == layer)
                    {
                        let down = down.as_ref().ok_or("declared down family absent")?;
                        let responses = if joint.is_some() {
                            Vec::new()
                        } else {
                            (0..direction_count)
                                .map(|j| {
                                    composed_rule_search::function(
                                        proposal.as_ref().unwrap(),
                                        settings.uses.len() + j,
                                    )
                                })
                                .collect::<Result<Vec<_>, _>>()?
                        };
                        let u = down
                            .direction_operators
                            .iter()
                            .map(|(u, _)| down.program.operators[*u].clone())
                            .collect::<Vec<_>>();
                        let target = native
                            .node_interface(layers[layer].mlp)
                            .map_err(|e| e.to_string())?;
                        let composed = if let Some(joint) = &joint {
                            parameter_response_program::compose_joint_with_outputs(
                                &joint.program,
                                joint.output_nodes[0],
                                &joint.output_nodes[1..],
                                &u,
                                &target,
                            )?
                        } else {
                            parameter_response_program::compose_with_outputs_on_interface(
                                clean.as_ref().unwrap(),
                                &responses,
                                &u,
                                &target,
                            )?
                        };
                        if joint.is_some() {
                            response_nodes.push(composed.clean_output.node);
                        }
                        response_nodes.extend(composed.response_nodes);
                        let function = composed.program;
                        let mut reads = vec![layers[layer].normed];
                        reads.extend(&down.control_nodes);
                        candidate = candidate.replace_function_inputs(
                            &format!("composed-mlp-response-{layer}"),
                            &function,
                            &reads,
                            layers[layer].mlp,
                        )?;
                    } else {
                        candidate = candidate.replace_function(
                            &format!("composed-mlp-{layer}"),
                            clean.as_ref().unwrap(),
                            layers[layer].normed,
                            layers[layer].mlp,
                        )?;
                    }
                }
                let fixed = down
                    .as_ref()
                    .map(|d| fixed_response_writers(&candidate, d))
                    .transpose()?
                    .unwrap_or_default();
                let (mut trainable, body_map) = if let Some(proposal) = &proposal {
                    graft_parameters_except(&candidate, &native, proposal, &fixed)?
                } else {
                    let joint = joint.as_ref().unwrap();
                    let down = down.as_ref().unwrap();
                    (
                        joint_graft_parameters(&candidate, down.native_write, joint)?,
                        BTreeMap::new(),
                    )
                };
                let direction_ids = down
                    .as_ref()
                    .map(|d| {
                        d.retain_directions_excluding(
                            &mut candidate,
                            &trainable.iter().copied().collect(),
                        )
                    })
                    .transpose()?
                    .unwrap_or_default();
                if frozen_body.is_some() && shared {
                    let frozen: BTreeSet<_> = body_map.values().copied().collect();
                    trainable.retain(|index| !frozen.contains(index));
                    verify_body(proposal_program, &candidate.program, &body_map)?;
                }
                if trainable.is_empty() {
                    return Err("graft lost all trainable boundary maps".into());
                }
                // Names are discovery metadata and intentionally absent from the codec.
                // Bind indices first; canonical() checks their structure survives replay.
                let mapped = controls(&candidate, &native, &layers, &settings.controls)?;
                let mapping: Vec<_> = mapped
                    .iter()
                    .map(|c| match c {
                        Control::NodeScale { node } => {
                            json!({"kind":"node_scale","candidate_node":node})
                        }
                        Control::GlobalOperatorScale { operator } => {
                            json!({"kind":"global_operator_scale","candidate_operator":operator})
                        }
                    })
                    .collect();
                save(
                    &root.join("CONTROL_MAP.json"),
                    &json!({"native_controls":settings.controls,"candidate_controls":mapping,"down_edit_family":settings.down_edit_family,"fixed_direction_operator_ids":direction_ids,"response_nodes":response_nodes,"observation_order":if joint.is_some(){"clean output, then scalar response coefficients"}else{"scalar response coefficients"}}),
                )?;
                let graft_seconds = graft_started.elapsed().as_secs_f64();
                let canonical_started = Instant::now();
                let (mut candidate, _, canonical_before_phases) =
                    canonical_using(&candidate, native_codec.as_ref())?;
                let canonical_before_seconds = canonical_started.elapsed().as_secs_f64();
                let response_paths: Vec<_> = if let Some(down) = &down {
                    response_nodes
                        .iter()
                        .map(|n| {
                            Ok(vec![
                                candidate
                                    .place(down.native_write)
                                    .ok_or("down write place absent")?,
                                *n,
                            ])
                        })
                        .collect::<Result<_, String>>()?
                } else {
                    Vec::new()
                };
                let lowered = intervention_program::compile_observed(
                    &candidate.program,
                    &mapped,
                    &response_paths,
                )?;
                let training = CausalEpisodes::rebind(
                    native_teacher_cache.as_ref(),
                    &lowered,
                    &candidate.program,
                    &mapped,
                    family,
                    &settings.cases,
                    &targets,
                    down.as_ref(),
                )?;
                let fit_started = Instant::now();
                let mut retained_response_targets = None;
                let fitted = if let (Some(down), Some(config)) = (&down, &settings.down_edit_family)
                {
                    let (response_targets, provenance) = down_response_targets(
                        &d,
                        down,
                        &target_controls,
                        family,
                        &settings.cases,
                        &targets,
                        &lowered.observed_nodes,
                        config.response_weight,
                        training.response_capture_budget(settings.fit.numeric_bytes)?,
                    )?;
                    save(&root.join("RESPONSE_TARGETS.json"), &provenance)?;
                    let fitted = training.fit(
                        &d,
                        &lowered.program,
                        Some(&response_targets),
                        &trainable,
                        settings.fit.clone(),
                    )?;
                    retained_response_targets = Some((response_targets, provenance));
                    fitted
                } else {
                    training.fit(&d, &lowered.program, None, &trainable, settings.fit.clone())?
                };
                save(
                    &root.join("FIT.json"),
                    &serde_json::to_value(&fitted.report).map_err(|e| e.to_string())?,
                )?;
                let fit_seconds = fit_started.elapsed().as_secs_f64();
                let canonical_started = Instant::now();
                candidate.program = lowered.restore(&fitted.program)?;
                let (saved, bytes, canonical_after_phases) =
                    canonical_using(&candidate, native_codec.as_ref())?;
                let canonical_after_seconds = canonical_started.elapsed().as_secs_f64();
                std::fs::write(root.join("program.artifact"), &bytes).map_err(|e| e.to_string())?;
                if frozen_body.is_some() && shared {
                    verify_body(proposal_program, &saved.program, &body_map)?;
                }
                if let Some(down) = &down {
                    verify_fixed_directions(&saved, down, &direction_ids)?;
                }
                if shared && down.is_none() {
                    let mut body_source = proposal_program.clone();
                    for (&pool, &graft) in &body_map {
                        Arc::make_mut(&mut body_source.operators[pool]).body =
                            saved.program.operators[graft].body.clone();
                    }
                    let body_artifact = Artifact::native(&body_source)?.f32_literals()?;
                    let (body_saved, body_bytes) = canonical(&body_artifact)?;
                    if body_saved.program.declarations.domains.len() != 0
                        || body_saved.program.declarations.parameters != 0
                    {
                        return Err("body source must have standalone Raw declarations".into());
                    }
                    let path = root.join("BODY_SOURCE.artifact");
                    std::fs::write(&path, body_bytes).map_err(|e| e.to_string())?;
                    save(
                        &path.with_extension("json"),
                        &json!({"pool_sha256":sha256(&path)?,"expression":inventory.expressions[id],"width":settings.width,"discovery_uses":settings.uses,"native_checkpoint_sha256":checkpoint,"declarations":{"domains":[],"parameters":0,"raw_slot_widths":use_specs.iter().map(|u|u.input_width).collect::<Vec<_>>()},"body_operator_map":body_map,"source_candidate_sha256":sha256(&root.join("program.artifact"))?,"scope":"fitted shared body source; pool boundary maps remain original initialization, NOT fitted discovery exports"}),
                    )?;
                }

                let saved_paths: Vec<_> = if let Some(down) = &down {
                    response_nodes
                        .iter()
                        .map(|n| {
                            Ok(vec![
                                saved
                                    .place(down.native_write)
                                    .ok_or("down write place absent")?,
                                *n,
                            ])
                        })
                        .collect::<Result<_, String>>()?
                } else {
                    Vec::new()
                };
                let saved_lowered =
                    intervention_program::compile_observed(&saved.program, &mapped, &saved_paths)?;
                let training = CausalEpisodes::rebind(
                    native_teacher_cache.as_ref(),
                    &saved_lowered,
                    &saved.program,
                    &mapped,
                    family,
                    &settings.cases,
                    &targets,
                    down.as_ref(),
                )?;
                let measured = training.measure(
                    &d,
                    &saved_lowered.program,
                    None,
                    settings.fit.numeric_bytes,
                )?;
                save(
                    &root.join("TRAIN.json"),
                    &serde_json::to_value(&measured).map_err(|e| e.to_string())?,
                )?;
                let mut response_error = None;
                if let Some((mut responses, provenance)) = retained_response_targets {
                    if lowered.observed_nodes.len() != saved_lowered.observed_nodes.len() {
                        return Err("saved response observation count changed".into());
                    }
                    let remap: BTreeMap<_, _> = lowered
                        .observed_nodes
                        .iter()
                        .copied()
                        .zip(saved_lowered.observed_nodes.iter().copied())
                        .collect();
                    for targets in responses.values_mut() {
                        for target in targets {
                            target.source_node = *remap
                                .get(&target.source_node)
                                .ok_or("saved response observation missing")?;
                        }
                    }
                    let supervised = training.measure(
                        &d,
                        &saved_lowered.program,
                        Some(&responses),
                        settings.fit.numeric_bytes,
                    )?;
                    response_error = Some(maximum_response_error(&supervised)?);
                    save(
                        &root.join("TRAIN_RESPONSES.json"),
                        &json!({"measurement":supervised,"targets":provenance,"pure_kl_report":"TRAIN.json","native_labels_reused":true}),
                    )?;
                }
                let cost_started = Instant::now();
                let c32 = structural_cost(&saved, &mut costs)?.total();
                let structural_cost_seconds = cost_started.elapsed().as_secs_f64();
                Ok(
                    json!({"id":name,"expression_id":id,"arm":arm,"c32":c32,"training_kl":measured.objective,"maximum_episode_response_error":response_error,
                    "artifact_sha256":sha256(&root.join("program.artifact"))?,"control_map_sha256":sha256(&root.join("CONTROL_MAP.json"))?,"trainable":trainable,"sharing_analysis":if joint_inventory.is_some(){Value::Null}else{serde_json::to_value(&sharing.analyses[id]).map_err(|e|e.to_string())?},"transfer":frozen_body.is_some(),"body_frozen":frozen_body.is_some() && shared,"body_graft_operator_map":body_map,"native_initialization":settings.native_initialization,"native_initialization_scope":"primitive native-width capacity control, not discovery","stage_seconds":{"graft":graft_seconds,"canonical_before_fit":canonical_before_seconds,"canonical_after_fit":canonical_after_seconds,"canonical_before_phases":canonical_before_phases,"canonical_after_phases":canonical_after_phases,"fit":fit_seconds,"structural_cost":structural_cost_seconds},"status":"training_measured"}),
                )
            })();
            let result = match attempt {
                Ok(v) => v,
                Err(error) => json!({"id":name,"status":"unresolved","error":error}),
            };
            save(&root.join("STATUS.json"), &result)?;
            serde_json::to_writer(&mut journal, &result).map_err(|e| e.to_string())?;
            journal.write_all(b"\n").map_err(|e| e.to_string())?;
            journal.flush().map_err(|e| e.to_string())?;
            eprintln!("{result}");
            rows.push(result);
        }
    }
    let frontier: Vec<_> = rows
        .iter()
        .filter(|a| a["training_kl"].is_number())
        .filter(|a| {
            !rows
                .iter()
                .any(|b| training_dominates(b, a, joint_inventory.is_some()))
        })
        .map(|v| v["id"].clone())
        .collect();
    save(
        &out.join("TRAINING_PARETO.json"),
        &json!({"ids":frontier,"native_response_axis":joint_inventory.is_some(),"scope":"Training C32/pure-KL Pareto; joint mode additionally requires no worse maximum sampled episode/observable mean normalized squared response error. Missing response measurements remain unknown. No holdout consulted, no universal fidelity guarantee."}),
    )?;
    let mut frozen = if joint_inventory.is_some() {
        freeze_joint_evaluation_ids(&frontier, &candidate_ids)
    } else {
        freeze_evaluation_ids(&frontier, &settings.expression_ids, use_specs.len() > 1)
    };
    add_native_capacity_controls(
        &mut frozen,
        &settings.expression_ids,
        use_specs.len() > 1,
        settings.native_initialization,
    );
    save(
        &out.join("FROZEN_EVALUATION_IDS.json"),
        &json!({"ids":frozen,"training_pareto_ids":frontier,"joint_response_inventory":joint_inventory.is_some(),"policy":if joint_inventory.is_some(){"native always plus learned joint candidates on training pure-KL/C32/native-response Pareto; no legacy expression counterparts"}else{"native always; both matched shared/untied controls of every training-Pareto expression, including failed counterparts; native-initialized capacity controls additionally mandatory, independent of cost dominance"},"native_capacity_controls_mandatory":settings.native_initialization,"frozen_before_heldout_export_access":true}),
    )?;
    let mut heldout_rows = Vec::new();
    let mut heldout_provenance = Value::Null;
    if let Some(evaluation) = &settings.evaluation {
        let heldout_started = Instant::now();
        let prepared = (|| -> Result<(FamilyInputs, Vec<Array2<f64>>, CausalEpisodes), String> {
            let heldout_export = Path::new(&args[4]);
            if sha256(&heldout_export.join("export.json"))? != evaluation.export_sha256 {
                return Err("heldout export SHA mismatch".into());
            }
            let imported_heldout =
                import_language_model(heldout_export, evaluation.sequences, settings.context)?;
            let heldout_native = split_sites(&imported_heldout.program)?;
            if !same_native(&original_native, &heldout_native) {
                return Err("heldout native graph or original numerical weights differ".into());
            }
            disjoint_token_sequences(family, &imported_heldout.contract.family, settings.context)?;
            let eval_cases = evaluation.cases.as_deref().unwrap_or(&settings.cases);
            let (labels, planned) = if settings.fixed_head_targets.is_some() {
                (Vec::new(), 0)
            } else {
                family_teacher_targets(
                    &d,
                    &original_native,
                    &original_layers,
                    &settings.controls,
                    &imported_heldout.contract.family,
                    eval_cases,
                    down.as_ref(),
                    settings.fit.numeric_bytes,
                    settings.teacher_numeric_bytes,
                )?
            };
            heldout_provenance = json!({"native":imported_heldout.record,"export_sha256":evaluation.export_sha256,"rows_per_episode":imported_heldout.contract.family.rows,"teacher_planned_numeric_bytes":planned,"model_identity":"same original graph/interfaces and numerical weight bits; wire-omitted provenance ignored","sequence_overlap":"all heldout fixed-context token sequences checked absent from training","scope":"previously project-seen, fit-disjoint panel; not untouched confirmation. No updates/reselection, operational F64 KL only"});
            save(&out.join("HELDOUT_PROVENANCE.json"), &heldout_provenance)?;
            let panel = CausalEpisodes::standalone(
                &d,
                &saved_native_graph,
                &saved_native.program,
                &target_controls,
                &imported_heldout.contract.family,
                eval_cases,
                &labels,
                settings.fixed_head_targets.as_ref(),
                settings.teacher_numeric_bytes,
                down.as_ref(),
                &original_layers,
                &settings.controls,
            )?;
            save(
                &out.join("HELDOUT_TEACHER_TARGETS.json"),
                &panel.target_metadata,
            )?;
            Ok((imported_heldout.contract.family, labels, panel))
        })();
        stage_seconds.insert(
            "heldout_load_and_teachers".into(),
            heldout_started.elapsed().as_secs_f64(),
        );
        let heldout_eval_started = Instant::now();
        for id in &frozen {
            let attempt = (|| -> Result<Value, String> {
                let (eval_family, labels, teacher_panel) =
                    prepared.as_ref().map_err(Clone::clone)?;
                let training = rows
                    .iter()
                    .find(|v| v["id"].as_str() == Some(id))
                    .ok_or("frozen candidate missing from training ledger")?;
                if training["status"] != "training_measured" {
                    return Err(
                        "frozen matched control failed during training; unresolved retained".into(),
                    );
                }
                let (artifact_path, map_path) = if id == "native" {
                    (
                        out.join("native.artifact"),
                        out.join("NATIVE_CONTROL_MAP.json"),
                    )
                } else {
                    (
                        out.join(id).join("program.artifact"),
                        out.join(id).join("CONTROL_MAP.json"),
                    )
                };
                if training["artifact_sha256"].as_str() != Some(&sha256(&artifact_path)?) {
                    return Err("saved candidate SHA mismatch".into());
                }
                if training["control_map_sha256"].as_str() != Some(&sha256(&map_path)?) {
                    return Err("saved control-map SHA mismatch".into());
                }
                let bytes = std::fs::read(&artifact_path).map_err(|e| e.to_string())?;
                let artifact = gam_gpu::trace::within_host("heldout.decode", || {
                    if let Some(cache) = &native_codec {
                        cache.decode_saved(&bytes, &native.declarations)
                    } else {
                        Artifact::from_bytes(&bytes, &native.declarations)
                    }
                })?;
                if gam_gpu::trace::within_host("heldout.reencode", || artifact.to_bytes())? != bytes {
                    return Err("heldout ordinary saved-byte canonical replay differs".into());
                }
                let c32 = structural_cost(&artifact, &mut costs)?.total();
                if training["c32"].as_u64() != Some(c32) {
                    return Err("heldout saved C32 differs from frozen training price".into());
                }
                let mapping: Value =
                    serde_json::from_slice(&std::fs::read(&map_path).map_err(|e| e.to_string())?)
                        .map_err(|e| e.to_string())?;
                if mapping["down_edit_family"]
                    != serde_json::to_value(&settings.down_edit_family)
                        .map_err(|e| e.to_string())?
                {
                    return Err("saved down-family declaration differs".into());
                }
                if id != "native" {
                    if let Some(down) = &down {
                        let ids: Vec<(usize, usize)> =
                            serde_json::from_value(mapping["fixed_direction_operator_ids"].clone())
                                .map_err(|e| e.to_string())?;
                        verify_fixed_directions(&artifact, down, &ids)?;
                    }
                }
                let mapped = stored_controls(&mapping, &settings.controls)?;
                let response_nodes: Vec<usize> = if id != "native" && down.is_some() {
                    serde_json::from_value(mapping["response_nodes"].clone())
                        .map_err(|e| e.to_string())?
                } else {
                    Vec::new()
                };
                let paths: Vec<_> = if let Some(down) = &down {
                    response_nodes
                        .iter()
                        .map(|n| {
                            Ok(vec![
                                artifact
                                    .place(down.native_write)
                                    .ok_or("down write place absent")?,
                                *n,
                            ])
                        })
                        .collect::<Result<_, String>>()?
                } else {
                    Vec::new()
                };
                let lowered =
                    intervention_program::compile_observed(&artifact.program, &mapped, &paths)?;
                let eval_episodes = CausalEpisodes::rebind(
                    Some(teacher_panel),
                    &lowered,
                    &artifact.program,
                    &mapped,
                    eval_family,
                    evaluation.cases.as_deref().unwrap_or(&settings.cases),
                    labels,
                    down.as_ref(),
                )?;
                let measured = gam_gpu::trace::within("heldout.measure", &d, || {
                    eval_episodes.measure(&d, &lowered.program, None, settings.fit.numeric_bytes)
                })?;
                let supervised = if id != "native" {
                    if let (Some(down), Some(config)) = (&down, &settings.down_edit_family) {
                        let (mut responses, mut provenance) = down_response_targets(
                            &d,
                            down,
                            &target_controls,
                            eval_family,
                            evaluation.cases.as_deref().unwrap_or(&settings.cases),
                            labels,
                            &lowered.observed_nodes,
                            config.response_weight,
                            eval_episodes.response_capture_budget(settings.fit.numeric_bytes)?,
                        )?;
                        let training_provenance: Value = serde_json::from_slice(
                            &std::fs::read(out.join(id).join("RESPONSE_TARGETS.json"))
                                .map_err(|e| e.to_string())?,
                        )
                        .map_err(|e| e.to_string())?;
                        for targets in responses.values_mut() {
                            for target in targets {
                                let key = target
                                    .label
                                    .strip_prefix("native_node_")
                                    .ok_or("response label schema differs")?;
                                let scale = training_provenance["native_rms_and_fixed_scale"][key]
                                    [1]
                                .as_f64()
                                .ok_or("frozen training scale missing")?;
                                if !scale.is_finite() || scale <= 0. {
                                    return Err("invalid frozen training scale".into());
                                }
                                target.scale = scale;
                            }
                        }
                        provenance["native_rms_and_fixed_scale"] =
                            training_provenance["native_rms_and_fixed_scale"].clone();
                        provenance["scale_scope"] = json!(
                            "Frozen training scales; heldout values do not redefine normalization."
                        );
                        Some(
                            json!({"measurement":eval_episodes.measure(&d,&lowered.program,Some(&responses),settings.fit.numeric_bytes)?,"targets":provenance}),
                        )
                    } else {
                        None
                    }
                } else {
                    None
                };
                Ok(
                    json!({"id":id,"status":"heldout_measured","c32":c32,"measurement":measured,"direct_response_supervision":supervised,"artifact_sha256":training["artifact_sha256"],"control_map_sha256":training["control_map_sha256"],"scope":"frozen candidate measurement only; no Local or acceptance certificate"}),
                )
            })();
            let record = match attempt {
                Ok(v) => v,
                Err(error) => json!({"id":id,"status":"heldout_unresolved","error":error}),
            };
            let path = if id == "native" {
                out.join("NATIVE_HELDOUT.json")
            } else {
                out.join(id).join("HELDOUT.json")
            };
            save(&path, &record)?;
            serde_json::to_writer(&mut journal, &record).map_err(|e| e.to_string())?;
            journal.write_all(b"\n").map_err(|e| e.to_string())?;
            journal.flush().map_err(|e| e.to_string())?;
            heldout_rows.push(record);
        }
        stage_seconds.insert(
            "heldout_evaluation".into(),
            heldout_eval_started.elapsed().as_secs_f64(),
        );
        save(
            &out.join("HELDOUT_REPORT.json"),
            &json!({"frozen_ids":frozen,"candidates":heldout_rows,"provenance":heldout_provenance,"setup_error":prepared.as_ref().err(),"seconds":heldout_started.elapsed().as_secs_f64(),"training_frontier_unchanged":true}),
        )?;
    }
    save(
        &out.join("REPORT.json"),
        &json!({"candidates":rows,"stage_seconds":stage_seconds,"native_codec_usage":native_codec.as_ref().map(|c|c.usage()),"native_codec_stats":native_codec.as_ref().map(|c|c.stats()),"native_initialization":settings.native_initialization,"sharing_selection":sharing,"require_interior_learned":settings.require_interior_learned,"frozen_shared_body_transfer":frozen_body.is_some(),"frozen_evaluation_ids":frozen,"heldout":heldout_rows,"heldout_provenance":heldout_provenance,"seconds":started.elapsed().as_secs_f64(),"scope":"finite training search only; unresolved failures remain unresolved; not native mechanism recovery or VPD comparison"}),
    )
}
fn main() -> Result<(), String> {
    let result = run();
    if gam_gpu::trace::timing_stages() {
        let totals = gam_gpu::trace::stage_totals();
        for t in &totals {
            eprintln!("profile {:<24} {:>8} x {:>12.6} s = {:>10.3} s", t.name, t.count, t.seconds / t.count.max(1) as f64, t.seconds);
        }
        if let Some(out) = std::env::args().filter(|a| a != "--profile").nth(3) {
            save(&Path::new(&out).join("PROFILE.json"), &json!({"inclusive_stage_seconds":totals,"scope":"--profile: each named stage synchronizes its device at both ends, so device work is charged to the stage that queued it; nested stages are included in their parents; host/device overlap is removed"}))?;
        }
    }
    result
}

#[cfg(test)]
mod tests {
    use super::*;
    use gam_mpd::{
        composed_rule_search::{Binary, Expr, Unary},
        operator_program::{Node, SlotValues},
    };
    #[test]
    fn joint_selection_preserves_native_state_fidelity_and_unknown_measurements() {
        let accurate = json!({"c32":100,"training_kl":0.02,"maximum_episode_response_error":0.001});
        let wrong_state = json!({"c32":90,"training_kl":0.01,"maximum_episode_response_error":1.0});
        assert!(training_dominates(&wrong_state, &accurate, false));
        assert!(!training_dominates(&wrong_state, &accurate, true));
        assert!(!training_dominates(&accurate, &wrong_state, true));
        let better = json!({"c32":90,"training_kl":0.01,"maximum_episode_response_error":0.0001});
        assert!(training_dominates(&better, &accurate, true));
        let unmeasured = json!({"c32":1,"training_kl":0.0});
        assert!(!training_dominates(&unmeasured, &accurate, true));
        assert!(!training_dominates(&accurate, &unmeasured, true));
    }
    #[test]
    fn operational_kl_axes_disclose_tiny_negative_and_refuse_invalid_scores() {
        assert_eq!(operational_kl_axis(-8.2e-14).expect("roundoff scale"), 0.);
        assert_eq!(
            operational_kl_axis(-OPERATIONAL_NEGATIVE_KL_TOLERANCE)
                .expect("declared inclusive bound"),
            0.
        );
        assert_eq!(operational_kl_axis(0.25).expect("positive score"), 0.25);
        for raw in [-1.01e-10, f64::NAN, f64::INFINITY, f64::NEG_INFINITY] {
            assert!(operational_kl_axis(raw).is_err());
        }
        assert_eq!(
            rounding_policy()["negative_kl_tolerance"],
            OPERATIONAL_NEGATIVE_KL_TOLERANCE
        );
    }
    #[test]
    fn compact_driver_dispatch_preserves_kl_native_response_fit_and_saved_replay() {
        use gam_mpd::operator_program::{
            exact_precision, Declarations, Interface, Law, Operator, Slot,
        };
        use ndarray::array;
        let dense = |a: Array2<f64>| {
            Arc::new(
                Operator::dense(
                    "fixture",
                    Interface::native(a.nrows()).expect("rows"),
                    Interface::native(a.ncols()).expect("cols"),
                    a.clone(),
                    exact_precision(a.iter().copied()).expect("precision"),
                    Default::default(),
                )
                .expect("dense"),
            )
        };
        let native = OperatorProgram {
            declarations: Declarations {
                domains: vec![],
                slots: vec![Slot::Raw { width: 2 }],
                parameters: 0,
            },
            bases: vec![],
            rules: vec![],
            operators: vec![
                dense(array![[1., 0.25], [-0.5, 1.]]),
                dense(array![[1., 0.], [-1., 0.], [0., 1.], [0., -1.], [0.5, 0.5]]),
            ],
            nodes: vec![
                Node::Raw { slot: 0 },
                Node::Affine {
                    terms: vec![(0, 0)],
                    bias: None,
                },
                Node::Pointwise {
                    input: 1,
                    laws: vec![Law::Relu],
                },
                Node::Affine {
                    terms: vec![(2, 1)],
                    bias: None,
                },
            ],
            output: 3,
        };
        let base = FamilyInputs {
            rows: 3,
            slots: vec![SlotValues::Raw(array![[0.5, 0.2], [-0.3, 0.7], [1., -0.2]])],
            layout: None,
        };
        let local = capture_joint_local_targets(&Device::host(), &native, 0, &[3], &base, 1 << 24)
            .expect("training-only local capture");
        let mut changed = native.clone();
        changed.operators[1] = dense(Array2::zeros((5, 2)));
        let (repaired, report) = prefit_joint_linear(&Device::host(), &changed, &[1], &local, 1 << 24)
            .expect("fixed-feature coefficient fit");
        assert_eq!(report["accepted"], true);
        assert!(report["proposed"]["maximum"].as_f64().unwrap() < 1e-10);
        assert_eq!(repaired.operators[0], native.operators[0]);
        let unseen = FamilyInputs { rows: 2,
            slots: vec![SlotValues::Raw(array![[2., -0.4], [-1., 3.]])], layout: None };
        let expected = native.execute(&unseen, false).expect("native unseen").values[3].clone();
        let actual = repaired.execute(&unseen, false).expect("autonomous unseen").values[3].clone();
        assert!(expected.iter().zip(actual.iter()).all(|(a,b)| (a-b).abs() < 1e-10));
        let controls = vec![Control::NodeScale { node: 0 }];
        let cases = vec![
            Case {
                label: "clean".into(),
                group: "clean".into(),
                gains: vec![1.],
                down_amplitudes: vec![],
                parameter_amplitudes: vec![],
            },
            Case {
                label: "half".into(),
                group: "edits".into(),
                gains: vec![0.5],
                down_amplitudes: vec![],
                parameter_amplitudes: vec![],
            },
        ];
        let d = Device::host();
        let lowered = intervention_program::compile(&native, &controls).expect("controls");
        let (labels, _) = teacher_targets(&d, &native, &controls, &base, &cases, 1 << 24, 1 << 24)
            .expect("full parity labels");
        let full = CausalEpisodes::from_family(
            &d,
            &lowered,
            &native,
            &controls,
            &base,
            &cases,
            &labels,
            None,
            1 << 24,
        )
        .expect("full episodes");
        let compact = CausalEpisodes::from_family(
            &d,
            &lowered,
            &native,
            &controls,
            &base,
            &cases,
            &[],
            Some(&FixedHeadSettings { tile_rows: 2 }),
            1 << 24,
        )
        .expect("compact direct generation");
        assert!(compact.full.is_none());
        assert_eq!(compact.target_metadata["full_logit_target_bytes"], 0);
        assert_eq!(
            compact.target_metadata["resident_mu_numeric_bytes"],
            3 * 2 * 2 * 8
        );
        let node = lowered.root_mapping[1];
        let map = BTreeMap::from([(node, node)]);
        let (responses, _) = native_response_targets_mapped(
            &d,
            &lowered.program,
            &map,
            &compact.metadata,
            1.,
            1 << 24,
        )
        .expect("native responses independent of label backend");
        let mut candidate = lowered.program.clone();
        let mut reader = (*candidate.operators[0]).clone();
        if let OperatorBody::Dense { values, .. } = &mut reader.body {
            *values *= 0.8;
        }
        candidate.operators[0] = Arc::new(reader);
        for supervision in [None, Some(&responses)] {
            let a = full
                .measure(&d, &candidate, supervision, 1 << 24)
                .expect("full score");
            let b = compact
                .measure(&d, &candidate, supervision, 1 << 24)
                .expect("compact score");
            assert!((a.objective - b.objective).abs() < 1e-10);
            let settings = FitSettings {
                iterations: 2,
                learning_rate: 0.01,
                beta1: 0.9,
                beta2: 0.999,
                epsilon: 1e-8,
                numeric_bytes: 1 << 24,
                schedule: None,
                arithmetic: resident_causal_fit::FitArithmetic::F64,
                exact_scan_every: 1,
            };
            let a = full
                .fit(&d, &candidate, supervision, &[0], settings.clone())
                .expect("full fit");
            let b = compact
                .fit(&d, &candidate, supervision, &[0], settings)
                .expect("compact fit");
            let aa = a.program.operators[0].matrix();
            let bb = b.program.operators[0].matrix();
            assert!(aa.iter().zip(&bb).all(|(x, y)| (x - y).abs() < 1e-10));
            let saved = canonical(&Artifact::native(&b.program).expect("artifact"))
                .expect("ordinary decoded replay")
                .0;
            let af = full
                .measure(&d, &saved.program, supervision, 1 << 24)
                .expect("saved full");
            let bf = compact
                .measure(&d, &saved.program, supervision, 1 << 24)
                .expect("saved compact");
            assert!((af.objective - bf.objective).abs() < 1e-10);
        }
        let settings = FitSettings {
            iterations: 1,
            learning_rate: 0.01,
            beta1: 0.9,
            beta2: 0.999,
            epsilon: 1e-8,
            numeric_bytes: 1 << 24,
            schedule: None,
            arithmetic: resident_causal_fit::FitArithmetic::F64,
            exact_scan_every: 1,
        };
        assert!(compact.fit(&d, &candidate, None, &[1], settings).is_err());
        assert!(CausalEpisodes::from_family(
            &d,
            &lowered,
            &native,
            &controls,
            &base,
            &cases,
            &labels,
            Some(&FixedHeadSettings { tile_rows: 2 }),
            1 << 24
        )
        .is_err());
        let head_controls = vec![Control::GlobalOperatorScale { operator: 1 }];
        let head_edited = intervention_program::compile(&native, &head_controls)
            .expect("global head control graph");
        assert!(CausalEpisodes::from_family(
            &d,
            &head_edited,
            &native,
            &head_controls,
            &base,
            &cases,
            &[],
            Some(&FixedHeadSettings { tile_rows: 2 }),
            1 << 24
        )
        .is_err());
    }
    #[test]
    fn tied_edits_hide_false_coefficients_but_direct_targets_identify_them() {
        use gam_mpd::operator_program::{exact_precision, Operator};
        use gam_mpd::operator_program::{Declarations, Interface, Slot};
        use ndarray::array;
        let dense = |a: Array2<f64>| {
            Arc::new(
                Operator::dense(
                    "fixture",
                    Interface::native(a.nrows()).expect("rows"),
                    Interface::native(a.ncols()).expect("cols"),
                    a.clone(),
                    exact_precision(a.iter().copied()).expect("precision"),
                    Default::default(),
                )
                .expect("dense"),
            )
        };
        let program = |false_attribution: bool| OperatorProgram {
            declarations: Declarations {
                domains: vec![],
                slots: vec![Slot::Raw { width: 2 }],
                parameters: 0,
            },
            bases: vec![],
            rules: vec![],
            operators: vec![
                dense(if false_attribution {
                    array![[1., 1.]]
                } else {
                    array![[1., 0.]]
                }),
                dense(if false_attribution {
                    array![[0., 0.]]
                } else {
                    array![[0., 1.]]
                }),
                dense(array![[1.], [-1.]]),
            ],
            nodes: vec![
                Node::Raw { slot: 0 },
                Node::Affine {
                    terms: vec![(0, 0)],
                    bias: None,
                },
                Node::Affine {
                    terms: vec![(0, 1)],
                    bias: None,
                },
                Node::Affine {
                    terms: vec![(1, 2), (2, 2)],
                    bias: None,
                },
            ],
            output: 3,
        };
        let teacher = program(false);
        let candidate = program(true);
        let inputs = FamilyInputs {
            rows: 3,
            slots: vec![SlotValues::Raw(array![[1., 2.], [-2., 1.], [0.5, -1.]])],
            layout: None,
        };
        let labels = teacher.execute(&inputs, false).expect("teacher").values[3].clone();
        let episodes = vec![Episode {
            label: "tied".into(),
            group: "tied".into(),
            inputs,
            target_logits: labels,
            scored: None,
        }];
        let map = BTreeMap::from([(1, 1), (2, 2)]);
        let (responses, _) = native_response_targets_mapped(
            &Device::host(),
            &teacher,
            &map,
            &response_metadata(&episodes),
            1.,
            1 << 24,
        )
        .expect("scalar targets");
        let kl = resident_causal_fit::measure(&Device::host(), &candidate, &episodes, 1 << 24)
            .expect("KL");
        let direct = resident_causal_fit::measure_with_native(
            &Device::host(),
            &candidate,
            &episodes,
            &responses,
            1 << 24,
        )
        .expect("direct loss");
        assert!(kl.objective.abs() < 1e-12);
        assert!(direct.objective > 0.1);
        let fit = resident_causal_fit::fit_with_native(
            &Device::host(),
            &candidate,
            &episodes,
            &responses,
            &[0, 1],
            FitSettings {
                iterations: 200,
                learning_rate: 0.05,
                beta1: 0.9,
                beta2: 0.999,
                epsilon: 1e-8,
                numeric_bytes: 1 << 24,
                schedule: None,
                arithmetic: resident_causal_fit::FitArithmetic::F64,
                exact_scan_every: 1,
            },
        )
        .expect("coefficient fit");
        let saved = canonical(&Artifact::native(&fit.program).expect("artifact"))
            .expect("ordinary saved replay")
            .0;
        let final_loss = resident_causal_fit::measure_with_native(
            &Device::host(),
            &saved.program,
            &episodes,
            &responses,
            1 << 24,
        )
        .expect("saved direct metric");
        assert!(final_loss.objective < direct.objective * 0.01);
        // Individually applied directions now expose the attribution improvement.
        let input = &episodes[0].inputs;
        let actual = saved
            .program
            .execute(input, false)
            .expect("candidate own states");
        let expected = teacher.execute(input, false).expect("native coefficients");
        for node in [1, 2] {
            assert!(actual.values[node]
                .iter()
                .zip(&expected.values[node])
                .all(|(a, b)| (a - b).abs() < 0.02));
        }
    }
    #[test]
    fn interior_requirement_refuses_boundaries_and_preserves_baseline_ids() {
        let grammar = Grammar {
            arguments: 1,
            max_operations: 3,
            max_expressions: 10000,
            unary: vec![Unary::GeluTanh],
            binary: vec![composed_rule_search::Binary::Multiply],
            affine: true,
        };
        let inventory = composed_rule_search::enumerate(&grammar).expect("inventory");
        assert!(selection(&inventory, &[6, 10], false).is_ok());
        assert!(selection(&inventory, &[6], true).is_err());
        assert!(selection(&inventory, &[10], true).is_err());
        let inner = Expr::Affine(Box::new(Expr::Unary(
            Unary::GeluTanh,
            Box::new(Expr::Argument(0)),
        )));
        assert_eq!(
            inventory.expressions[15],
            Expr::Unary(Unary::GeluTanh, Box::new(inner.clone()))
        );
        assert_eq!(
            inventory.expressions[32],
            Expr::Binary(
                composed_rule_search::Binary::Multiply,
                Box::new(Expr::Argument(0)),
                Box::new(inner)
            )
        );
        let chosen = selection(&inventory, &[15, 32], true).expect("interior IDs");
        assert!(chosen.interior_indices.contains(&15));
        assert!(chosen.interior_indices.contains(&32));
    }
    #[test]
    fn copied_body_is_exact_for_shared_and_initial_untied_transfer() {
        let expr = Expr::Unary(
            Unary::GeluTanh,
            Box::new(Expr::Affine(Box::new(Expr::Unary(
                Unary::GeluTanh,
                Box::new(Expr::Argument(0)),
            )))),
        );
        let uses = [UseSpec {
            input_width: 3,
            output_width: 3,
        }; 2];
        let source = composed_rule_search::compile(&expr, 4, &uses, 7).expect("source");
        let source = Artifact::native(&source.program)
            .expect("source artifact")
            .f32_literals()
            .expect("source f32");
        for shared in [true, false] {
            let mut destination = if shared {
                composed_rule_search::compile(&expr, 4, &uses, 91)
            } else {
                composed_rule_search::compile_untied(&expr, 4, &uses, 91)
            }
            .expect("destination");
            copy_body(&source.program, &mut destination.program).expect("copy exact body");
            let source_ids = body_dense_indices(&source.program, 0).expect("source IDs");
            for rule in 0..destination.program.rules.len() {
                let ids = body_dense_indices(&destination.program, rule).expect("destination IDs");
                for (a, b) in source_ids.iter().zip(ids.iter()) {
                    assert!(same_dense(
                        &source.program.operators[*a],
                        &destination.program.operators[*b]
                    ));
                }
            }
            assert_eq!(destination.program.rules.len(), if shared { 1 } else { 2 });
        }
    }
    #[test]
    fn frozen_evaluation_includes_native_and_matched_controls_without_eval_selection() {
        assert_eq!(
            freeze_evaluation_ids(&[json!("expression6-shared")], &[6, 10], true),
            vec!["native", "expression6-shared", "expression6-untied"]
        );
        assert_eq!(freeze_evaluation_ids(&[], &[6, 10], true), vec!["native"]);
        assert_eq!(
            freeze_evaluation_ids(&[json!("expression10-shared")], &[6, 10], false),
            vec!["native", "expression10-shared"]
        );
    }
    #[test]
    fn heldout_rejects_identical_and_partially_overlapping_token_sequences() {
        let family = |tokens: Vec<u32>| FamilyInputs {
            rows: tokens.len(),
            slots: vec![SlotValues::Tokens(tokens)],
            layout: None,
        };
        let train = family(vec![1, 2, 3, 4]);
        assert!(disjoint_token_sequences(&train, &family(vec![1, 2, 3, 4]), 2).is_err());
        assert!(disjoint_token_sequences(&train, &family(vec![5, 6, 1, 2]), 2).is_err());
        assert!(disjoint_token_sequences(&train, &family(vec![5, 6, 7, 8]), 2).is_ok());
    }
    #[test]
    fn wire_metadata_does_not_define_training_or_control_identity() {
        let proposal = composed_rule_search::compile(
            &Expr::Unary(
                Unary::GeluTanh,
                Box::new(Expr::Affine(Box::new(Expr::Argument(0)))),
            ),
            3,
            &[UseSpec {
                input_width: 3,
                output_width: 3,
            }; 2],
            19,
        )
        .expect("valid regression fixture");
        let artifact = Artifact::native(&proposal.program).expect("valid regression fixture");
        let operator = proposal
            .program
            .operators
            .iter()
            .position(|op| op.name == "shared internal affine matrix")
            .expect("valid regression fixture");
        let specs = vec![NativeControl::RetainedOperator {
            name: proposal.program.operators[operator].name.clone(),
        }];
        let mapping =
            controls(&artifact, &proposal.program, &[], &specs).expect("valid regression fixture");
        let (saved, _) = canonical(&artifact).expect("valid regression fixture");
        assert_ne!(
            saved.program.operators[operator].name,
            proposal.program.operators[operator].name
        );
        assert!(controls(&saved, &proposal.program, &[], &specs).is_err());
        let lowered = intervention_program::compile(&saved.program, &mapping)
            .expect("valid regression fixture");
        let base = FamilyInputs {
            rows: 2,
            slots: vec![SlotValues::Raw(Array2::ones((2, 3))); 2],
            layout: None,
        };
        let case = Case {
            label: "half".into(),
            group: "global".into(),
            down_amplitudes: vec![],
            parameter_amplitudes: vec![],
            gains: vec![0.5],
        };
        let input = lowered
            .family(
                &base,
                &values(&saved.program, &mapping, 2, &case).expect("valid regression fixture"),
            )
            .expect("valid regression fixture");
        let trace = lowered
            .program
            .execute(&input, false)
            .expect("valid regression fixture");
        assert!(trace.values[lowered.program.output]
            .iter()
            .all(|v| v.is_finite()));
        let restored = lowered
            .restore(&lowered.program)
            .expect("valid regression fixture");
        assert_eq!(restored, saved.program);
    }
    #[test]
    fn graft_exports_both_uses_and_preserves_shared_body() {
        let expr = Expr::Unary(
            Unary::GeluTanh,
            Box::new(Expr::Affine(Box::new(Expr::Argument(0)))),
        );
        let proposal = composed_rule_search::compile(
            &expr,
            3,
            &[UseSpec {
                input_width: 3,
                output_width: 3,
            }; 2],
            21,
        )
        .expect("valid regression fixture");
        // Start with two independent native functions, then graft the two shared uses.
        let native_proposal = composed_rule_search::compile_untied(
            &expr,
            3,
            &[UseSpec {
                input_width: 3,
                output_width: 3,
            }; 2],
            77,
        )
        .expect("valid regression fixture");
        let Node::Concat { parts } = &native_proposal.program.nodes[native_proposal.program.output]
        else {
            panic!("concat");
        };
        let reads: Vec<_> = native_proposal
            .program
            .nodes
            .iter()
            .enumerate()
            .filter_map(|(i, n)| matches!(n, Node::Raw { .. }).then_some(i))
            .collect();
        let mut artifact =
            Artifact::native(&native_proposal.program).expect("valid regression fixture");
        for slot in 0..2 {
            artifact = artifact
                .replace_function(
                    "searched use",
                    &composed_rule_search::function(&proposal, slot)
                        .expect("valid regression fixture"),
                    reads[slot],
                    parts[slot],
                )
                .expect("valid regression fixture");
        }
        let count = artifact
            .program
            .operators
            .iter()
            .filter(|op| op.name == "shared internal affine matrix")
            .count();
        assert_eq!(count, 1);
        let (parameters, body_map) =
            graft_parameters(&artifact, &native_proposal.program, &proposal)
                .expect("mapped parameters");
        assert_eq!(parameters.len(), proposal.trainable.len());
        assert_eq!(body_map.len(), 2);
        let (decoded, _) = canonical(&artifact).expect("ordinary replay");
        let source = Artifact::native(&proposal.program)
            .expect("source")
            .f32_literals()
            .expect("source f32");
        verify_body(&source.program, &decoded.program, &body_map).expect("unchanged body bits");
        let input = FamilyInputs {
            rows: 3,
            slots: vec![
                SlotValues::Raw(Array2::from_elem((3, 3), 0.4)),
                SlotValues::Raw(Array2::from_elem((3, 3), -0.7)),
            ],
            layout: None,
        };
        let expected = proposal
            .program
            .execute(&input, false)
            .expect("valid regression fixture");
        let actual = artifact
            .program
            .execute(&input, false)
            .expect("valid regression fixture");
        assert_eq!(
            expected.values[proposal.program.output],
            actual.values[artifact.program.output]
        );
        canonical(&artifact).expect("valid regression fixture");
    }
    #[test]
    fn native_capacity_controls_are_frozen_even_when_cost_dominated() {
        let mut ids = vec!["native".into()];
        add_native_capacity_controls(&mut ids, &[1], true, true);
        assert_eq!(
            ids,
            vec!["native", "expression1-shared", "expression1-untied"]
        );
        add_native_capacity_controls(&mut ids, &[1], true, true);
        assert_eq!(ids.len(), 3);
        let mut regular = vec!["native".into()];
        add_native_capacity_controls(&mut regular, &[1], true, false);
        assert_eq!(regular, vec!["native"]);
    }
    #[test]
    fn finite_down_cases_require_zero_clean_and_reject_missing_nonfinite_controls() {
        let clean = Case {
            label: "clean".into(),
            group: "clean".into(),
            gains: vec![],
            down_amplitudes: vec![0.],
            parameter_amplitudes: vec![],
        };
        let edited = Case {
            label: "negative".into(),
            group: "edits".into(),
            gains: vec![],
            down_amplitudes: vec![-0.5],
            parameter_amplitudes: vec![],
        };
        assert!(validate_cases(&[clean.clone(), edited.clone()], 0, 1, true).is_ok());
        assert!(validate_cases(&[edited], 0, 1, true).is_err());
        let mut bad = clean.clone();
        bad.down_amplitudes = vec![f64::NAN];
        assert!(validate_cases(&[bad], 0, 1, false).is_err());
        assert!(validate_cases(&[clean], 0, 2, true).is_err());
    }
    #[test]
    fn joint_configuration_rejects_legacy_fields_and_extra_sites_before_import() {
        let mut config = json!({"export_sha256":"", "layers":1,"uses":[0],"sequences":1,"context":1,"teacher_numeric_bytes":1000,"controls":[],"cases":[],"fit":{"iterations":1,"learning_rate":0.01,"beta1":0.9,"beta2":0.999,"epsilon":1e-8,"numeric_bytes":1000},"seed":1,
            "down_edit_family":{"layer":0,"directions":[{"output":[1.],"hidden":[1.]}]},
            "joint_response_search":{"latent_widths":[1],"unary":[],"binary":[],"affine_bias":false,"require_shared":false,"max_operations":2,"max_affine_parameters":2,"max_parameter_elements":10,"max_expression_states":20,"max_tuple_checks":20,"max_tuples":10,"max_body_nodes":10,"seed":1}});
        let check = |v: Value| {
            validate_joint_configuration(&serde_json::from_value::<Settings>(v).expect("settings"))
        };
        assert!(check(config.clone()).is_ok());
        config["width"] = json!(2);
        assert!(check(config.clone()).is_err());
        config["width"] = json!(0);
        config["expression_ids"] = json!([0]);
        assert!(check(config.clone()).is_err());
        config["expression_ids"] = json!([]);
        config["uses"] = json!([0, 1]);
        assert!(check(config.clone()).is_err());
        config["uses"] = json!([0]);
        config["down_edit_family"] = Value::Null;
        assert!(check(config).is_err());
    }
    #[test]
    fn joint_inventory_ids_freeze_without_legacy_expression_indices() {
        assert_eq!(
            freeze_joint_evaluation_ids(&[json!("joint14-learned"), json!("native")], &[0, 14, 99]),
            vec!["native", "joint14-learned"]
        );
        assert_eq!(freeze_joint_evaluation_ids(&[], &[14]), vec!["native"]);
    }
    #[test]
    fn removed_down_family_runs_literal_teachers_shared_response_fit_and_saved_replay() {
        use gam_mpd::operator_program::{
            exact_precision, Declarations, Interface, Law, Operator, Slot, SlotValues,
        };
        use ndarray::array;
        let dense = |values: Array2<f64>| {
            Arc::new(
                Operator::dense(
                    "fixture",
                    Interface::native(values.nrows()).expect("rows"),
                    Interface::native(values.ncols()).expect("cols"),
                    values.clone(),
                    exact_precision(values.iter().copied()).expect("precision"),
                    Default::default(),
                )
                .expect("dense"),
            )
        };
        let mut grouped_writer = (*dense(array![[1., -0.5], [0.25, 0.75]])).clone();
        grouped_writer.rows =
            Interface::uniform(2, 1, gam_mpd::operator_program::LabelKind::Unit, 0)
                .expect("native unit groups");
        if let OperatorBody::Dense { present, .. } = &mut grouped_writer.body {
            *present = Array2::from_elem((2, 1), true);
        }
        let original = OperatorProgram {
            declarations: Declarations {
                domains: vec![],
                slots: vec![Slot::Raw { width: 2 }],
                parameters: 0,
            },
            bases: vec![],
            rules: vec![],
            operators: vec![
                dense(array![[1., 0.5], [-0.25, 1.]]),
                Arc::new(grouped_writer),
                {
                    let mut head = (*dense(array![[1., -0.25], [0.25, 0.75], [-0.5, 0.5]])).clone();
                    head.cols =
                        Interface::uniform(2, 1, gam_mpd::operator_program::LabelKind::Unit, 0)
                            .expect("head native groups");
                    if let OperatorBody::Dense { present, .. } = &mut head.body {
                        *present = Array2::from_elem((1, 2), true);
                    }
                    Arc::new(head)
                },
            ],
            nodes: vec![
                Node::Raw { slot: 0 },
                Node::Affine {
                    terms: vec![(0, 0)],
                    bias: None,
                },
                Node::Pointwise {
                    input: 1,
                    laws: vec![Law::Relu],
                },
                Node::Affine {
                    terms: vec![(2, 1)],
                    bias: None,
                },
                Node::Affine {
                    terms: vec![(3, 2)],
                    bias: None,
                },
            ],
            output: 4,
        };
        let down = down_edit_family::build(
            &original,
            0,
            3,
            &[Direction {
                output: array![1., -1.],
                hidden: array![0.5, -0.25],
            }],
        )
        .expect("family");
        let base = FamilyInputs {
            rows: 2,
            slots: vec![SlotValues::Raw(array![[0.6, -0.2], [-0.4, 0.8]])],
            layout: None,
        };
        let cases = [
            Case {
                label: "clean".into(),
                group: "clean".into(),
                gains: vec![],
                down_amplitudes: vec![0.],
                parameter_amplitudes: vec![],
            },
            Case {
                label: "positive".into(),
                group: "edits".into(),
                gains: vec![],
                down_amplitudes: vec![0.5],
                parameter_amplitudes: vec![],
            },
            Case {
                label: "negative".into(),
                group: "edits".into(),
                gains: vec![],
                down_amplitudes: vec![-0.5],
                parameter_amplitudes: vec![],
            },
        ];
        let device = Device::host();
        let (targets, _) = family_teacher_targets(
            &device,
            &original,
            &[],
            &[],
            &base,
            &cases,
            Some(&down),
            1_000_000,
            1_000_000,
        )
        .expect("literal teachers");
        let native_lowered =
            intervention_program::compile(&down.program, &[]).expect("native lowering");
        let compact_teachers = CausalEpisodes::standalone(
            &device,
            &native_lowered,
            &down.program,
            &[],
            &base,
            &cases,
            &[],
            Some(&FixedHeadSettings { tile_rows: 1 }),
            1_000_000,
            Some(&down),
            &[],
            &[],
        )
        .expect("literal compact teachers");
        assert!(compact_teachers.full.is_none());
        assert_eq!(
            compact_teachers.target_metadata["full_logit_target_bytes"],
            0
        );
        assert_eq!(
            compact_teachers.target_metadata["resident_mu_numeric_bytes"],
            2 * 2 * 3 * 8
        );
        for (case, target) in cases.iter().zip(&targets) {
            let literal = down.literal_native(&case.down_amplitudes).expect("literal");
            let expected = literal
                .execute(&base, false)
                .expect("original native state")
                .values[literal.output]
                .clone();
            assert!(expected
                .iter()
                .zip(target)
                .all(|(a, b)| (a - b).abs() < 1e-12));
        }
        let expression = Expr::Unary(
            Unary::Relu,
            Box::new(Expr::Affine(Box::new(Expr::Argument(0)))),
        );
        for arm in 0..4 {
            let shared = arm != 1;
            let compiler = if shared {
                composed_rule_search::compile
            } else {
                composed_rule_search::compile_untied
            };
            let proposal = if arm < 2 {
                Some(
                    compiler(
                        &expression,
                        2,
                        &[
                            UseSpec {
                                input_width: 2,
                                output_width: 2,
                            },
                            UseSpec {
                                input_width: 2,
                                output_width: 1,
                            },
                        ],
                        7,
                    )
                    .expect("joint clean and response pool"),
                )
            } else {
                None
            };
            let joint = if arm >= 2 {
                use program_learned_dag::{Expr as J, TypeRef};
                let search = program_learned_dag::Settings {
                    latent_widths: vec![2],
                    unary: vec![Unary::Relu],
                    binary: vec![],
                    affine_bias: false,
                    require_shared: true,
                    max_operations: 3,
                    max_affine_parameters: 3,
                    max_parameter_elements: 40,
                    max_expression_states: 1000,
                    max_tuple_checks: 100000,
                    max_tuples: 1000,
                    max_body_nodes: 20,
                    seed: 7,
                };
                let shared = J::Unary(
                    Unary::Relu,
                    Box::new(J::Affine {
                        parameter: 0,
                        output: TypeRef::Input(0),
                        input: Box::new(J::Argument(0)),
                        bias: false,
                    }),
                );
                let clean = if arm == 3 {
                    shared.clone()
                } else {
                    J::Affine {
                        parameter: 1,
                        output: TypeRef::Exit(0),
                        input: Box::new(shared.clone()),
                        bias: false,
                    }
                };
                let response = J::Affine {
                    parameter: 2,
                    output: TypeRef::Exit(1),
                    input: Box::new(shared),
                    bias: false,
                };
                let expressions = if arm == 2 {
                    let inventory = program_learned_dag::enumerate_interfaces(
                        &[Interface::native(2).unwrap()],
                        &[Interface::native(2).unwrap(), Interface::native(1).unwrap()],
                        &search,
                    )
                    .expect("bounded reachable inventory");
                    assert!(inventory.checked_tuples > 0);
                    // Select a syntactic integration fixture, not a numerical winner.
                    inventory
                        .proposals
                        .iter()
                        .find(|p| {
                            fn nonlinear(e: &J, found: &mut BTreeSet<J>) {
                                match e {
                                    J::Unary(_, x) => {
                                        found.insert(e.clone());
                                        nonlinear(x, found);
                                    }
                                    J::Affine { input, .. } => nonlinear(input, found),
                                    J::Binary(_, a, b) => {
                                        nonlinear(a, found);
                                        nonlinear(b, found);
                                    }
                                    J::Argument(_) => {}
                                }
                            }
                            let mut a = BTreeSet::new();
                            let mut b = BTreeSet::new();
                            nonlinear(&p.expressions[0], &mut a);
                            nonlinear(&p.expressions[1], &mut b);
                            a.intersection(&b).next().is_some()
                        })
                        .expect("enumerated shared nonlinear fixture")
                        .expressions
                        .clone()
                } else {
                    vec![clean, response]
                };
                Some(
                    program_learned_dag::compile_program(
                        &[Interface::native(2).unwrap()],
                        &[Interface::native(2).unwrap(), Interface::native(1).unwrap()],
                        &expressions,
                        &search,
                    )
                    .expect("distinct joint expressions sharing real ReLU"),
                )
            } else {
                None
            };
            let mut composed = if let Some(joint) = &joint {
                parameter_response_program::compose_joint_with_outputs(
                    &joint.program,
                    joint.output_nodes[0],
                    &joint.output_nodes[1..],
                    &[down.program.operators[down.direction_operators[0].0].clone()],
                    &down.program.node_interface(down.native_write).unwrap(),
                )
                .expect("joint compose")
            } else {
                parameter_response_program::compose_with_outputs_on_interface(
                    &composed_rule_search::function(proposal.as_ref().unwrap(), 0).expect("clean"),
                    &[
                        composed_rule_search::function(proposal.as_ref().unwrap(), 1)
                            .expect("response"),
                    ],
                    &[down.program.operators[down.direction_operators[0].0].clone()],
                    &down
                        .program
                        .node_interface(down.native_write)
                        .expect("native output groups"),
                )
                .expect("compose")
            };
            if joint.is_some() {
                composed
                    .response_nodes
                    .insert(0, composed.clean_output.node);
            }
            let mut candidate = Artifact::native(&down.program)
                .expect("native")
                .replace_function_inputs(
                    "removed down",
                    &composed.program,
                    &[down.native_read, down.control_nodes[0]],
                    down.native_write,
                )
                .expect("whole block graft");
            assert!(candidate.place(down.node_mapping[1]).is_none());
            assert!(candidate.place(down.node_mapping[2]).is_none());
            let fixed = fixed_response_writers(&candidate, &down).expect("fixed writer roles");
            let trainable = if let Some(joint) = &joint {
                let ids = joint_graft_parameters(&candidate, down.native_write, joint)
                    .expect("mapped joint parameter owners");
                assert_eq!(ids.len(), joint.trainable_operator_ids.len());
                ids
            } else {
                graft_parameters_except(
                    &candidate,
                    &down.program,
                    proposal.as_ref().unwrap(),
                    &fixed,
                )
                .expect("real pool parameters")
                .0
            };
            assert!(trainable.iter().all(|id| !fixed.contains(id)));
            if arm == 3 {
                let joint = joint.as_ref().unwrap();
                let identity = &composed.program.operators[joint.program.operators.len()];
                let ids: Vec<_> = candidate
                    .program
                    .operators
                    .iter()
                    .enumerate()
                    .filter_map(|(id, op)| Arc::ptr_eq(op, identity).then_some(id))
                    .collect();
                assert_eq!(
                    ids.len(),
                    1,
                    "fixed grouped clean identity has an explicit owner"
                );
                assert!(
                    !trainable.contains(&ids[0]),
                    "fallback identity must never be fitted"
                );
            }

            let ids = down
                .retain_directions_excluding(&mut candidate, &trainable.iter().copied().collect())
                .expect("paid immutable directions");
            let (candidate, _) = canonical(&candidate).expect("F32 initial graph");
            verify_fixed_directions(&candidate, &down, &ids).expect("fixed literals");
            let paths: Vec<_> = composed
                .response_nodes
                .iter()
                .map(|n| vec![candidate.place(down.native_write).expect("write"), *n])
                .collect();
            let lowered = intervention_program::compile_observed(&candidate.program, &[], &paths)
                .expect("nested observations");
            let panels = episodes(
                &lowered,
                &candidate.program,
                &[],
                &base,
                &cases,
                &targets,
                Some(&down),
            )
            .expect("autonomous edit episodes");
            let (responses, provenance) = down_response_targets(
                &device,
                &down,
                &[],
                &base,
                &cases,
                &targets,
                &lowered.observed_nodes,
                1.,
                1_000_000,
            )
            .expect("native clean and scalar labels");
            assert_eq!(
                responses["clean"].len(),
                if joint.is_some() { 2 } else { 1 }
            );
            assert!(provenance["native_rms_and_fixed_scale"].is_object());
            let compact_panels = CausalEpisodes::rebind(
                Some(&compact_teachers),
                &lowered,
                &candidate.program,
                &[],
                &base,
                &cases,
                &[],
                Some(&down),
            )
            .expect("candidate compact signed inputs");
            let (compact_responses, _) = down_response_targets(
                &device,
                &down,
                &[],
                &base,
                &cases,
                &[],
                &lowered.observed_nodes,
                1.,
                compact_panels
                    .response_capture_budget(1_000_000)
                    .expect("label reserve"),
            )
            .expect("scalar labels without any full logits");
            for (label, targets) in &responses {
                for (a, b) in targets.iter().zip(&compact_responses[label]) {
                    assert_eq!(a.values, b.values);
                    assert_eq!(a.scale.to_bits(), b.scale.to_bits());
                }
            }
            let compact_fit = compact_panels
                .fit(
                    &device,
                    &lowered.program,
                    Some(&compact_responses),
                    &trainable,
                    FitSettings {
                        iterations: 2,
                        learning_rate: 0.01,
                        beta1: 0.9,
                        beta2: 0.999,
                        epsilon: 1e-8,
                        numeric_bytes: 1_000_000,
                        schedule: None,
                        arithmetic: resident_causal_fit::FitArithmetic::F64,
                        exact_scan_every: 1,
                    },
                )
                .expect("compact joint fit");
            let fit = resident_causal_fit::fit_with_native(
                &device,
                &lowered.program,
                &panels,
                &responses,
                &trainable,
                FitSettings {
                    iterations: 2,
                    learning_rate: 0.01,
                    beta1: 0.9,
                    beta2: 0.999,
                    epsilon: 1e-8,
                    numeric_bytes: 1_000_000,
                    schedule: None,
                    arithmetic: resident_causal_fit::FitArithmetic::F64,
                    exact_scan_every: 1,
                },
            )
            .expect("joint causal fit");
            for id in &trainable {
                let a = fit.program.operators[*id].matrix();
                let b = compact_fit.program.operators[*id].matrix();
                assert!(a.iter().zip(b.iter()).all(|(x, y)| (x - y).abs() < 1e-9));
            }
            let mut result = candidate.clone();
            result.program = lowered.restore(&fit.program).expect("source graph restore");
            let (saved, bytes) = canonical(&result).expect("saved F32");
            assert_eq!(
                Artifact::from_bytes(&bytes, &down.program.declarations)
                    .expect("ordinary decode")
                    .to_bytes()
                    .expect("reencode"),
                bytes
            );
            verify_fixed_directions(&saved, &down, &ids).expect("directions remain fixed");
            saved.validate_coverage(&down.program).expect("coverage");
            let cost = structural_cost(&saved, &mut CostCache::default()).expect("complete C32");
            assert!(cost.literals >= 4);
            let paths: Vec<_> = composed
                .response_nodes
                .iter()
                .map(|n| vec![saved.place(down.native_write).expect("saved write"), *n])
                .collect();
            let replay = intervention_program::compile_observed(&saved.program, &[], &paths)
                .expect("saved observations");
            assert_eq!(replay.observed_nodes, lowered.observed_nodes);
            let panels = episodes(
                &replay,
                &saved.program,
                &[],
                &base,
                &cases,
                &targets,
                Some(&down),
            )
            .expect("saved episodes");
            let measured =
                resident_causal_fit::measure(&device, &replay.program, &panels, 1_000_000)
                    .expect("full saved metrics");
            assert!(measured.objective.is_finite());
            let supervised = resident_causal_fit::measure_with_native(
                &device,
                &replay.program,
                &panels,
                &responses,
                1_000_000,
            )
            .expect("saved direct coefficient metrics");
            assert!(supervised.objective.is_finite());
            let compact_saved = CausalEpisodes::rebind(
                Some(&compact_teachers),
                &replay,
                &saved.program,
                &[],
                &base,
                &cases,
                &[],
                Some(&down),
            )
            .expect("decoded compact inputs");
            let compact_kl = compact_saved
                .measure(&device, &replay.program, None, 1_000_000)
                .expect("decoded compact KL");
            let compact_joint = compact_saved
                .measure(&device, &replay.program, Some(&responses), 1_000_000)
                .expect("decoded compact responses");
            assert!((measured.objective - compact_kl.objective).abs() < 1e-10);
            assert!((supervised.objective - compact_joint.objective).abs() < 1e-10);
        }
    }
    #[test]
    fn compact_down_teacher_refuses_edits_to_final_readout_head() {
        use gam_mpd::operator_program::exact_precision;
        use gam_mpd::operator_program::{Declarations, Interface, Law, Operator, Slot};
        use ndarray::array;
        let dense = |name: &str| {
            Arc::new(
                Operator::dense(
                    name,
                    Interface::native(2).expect("rows"),
                    Interface::native(2).expect("cols"),
                    Array2::eye(2),
                    exact_precision([0., 1.]).expect("exact"),
                    Default::default(),
                )
                .expect("dense"),
            )
        };
        let source = OperatorProgram {
            declarations: Declarations {
                domains: vec![],
                slots: vec![Slot::Raw { width: 2 }],
                parameters: 0,
            },
            bases: vec![],
            rules: vec![],
            operators: vec![dense("reader"), dense("head")],
            nodes: vec![
                Node::Raw { slot: 0 },
                Node::Affine {
                    terms: vec![(0, 0)],
                    bias: None,
                },
                Node::Pointwise {
                    input: 1,
                    laws: vec![Law::Relu],
                },
                Node::Affine {
                    terms: vec![(2, 1)],
                    bias: None,
                },
            ],
            output: 3,
        };
        let down = down_edit_family::build(
            &source,
            0,
            3,
            &[Direction {
                output: array![1., 0.],
                hidden: array![1., 0.],
            }],
        )
        .expect("finite head edit family");
        let lowered = intervention_program::compile(&down.program, &[]).expect("lowered");
        let family = FamilyInputs {
            rows: 1,
            slots: vec![SlotValues::Raw(array![[1., 0.5]])],
            layout: None,
        };
        let cases = [
            Case {
                label: "clean".into(),
                group: "clean".into(),
                gains: vec![],
                down_amplitudes: vec![0.],
                parameter_amplitudes: vec![],
            },
            Case {
                label: "edited".into(),
                group: "edits".into(),
                gains: vec![],
                down_amplitudes: vec![0.5],
                parameter_amplitudes: vec![],
            },
        ];
        let error = CausalEpisodes::standalone(
            &Device::host(),
            &lowered,
            &down.program,
            &[],
            &family,
            &cases,
            &[],
            Some(&FixedHeadSettings { tile_rows: 1 }),
            1_000_000,
            Some(&down),
            &[],
            &[],
        )
        .err()
        .expect("fixed-head backend cannot represent changed readout");
        assert!(error.contains("incompatible immutable heads"));
    }
    #[test]
    fn optional_native_codec_preserves_exact_saved_bytes_cost_and_full_measurements() {
        let expression = Expr::Unary(
            Unary::GeluTanh,
            Box::new(Expr::Affine(Box::new(Expr::Argument(0)))),
        );
        let proposal = composed_rule_search::compile(
            &expression,
            2,
            &[UseSpec {
                input_width: 2,
                output_width: 2,
            }],
            7,
        )
        .expect("proposal");
        let source = Artifact::native(&proposal.program)
            .expect("source")
            .f32_literals()
            .expect("F32 source");
        let cache = CanonicalArtifactCache::new(&source, 1_000_000).expect("bounded cache");
        let (plain, bytes, _) = canonical_using(&source, None).expect("ordinary");
        let (cached, cached_bytes, _) = canonical_using(&source, Some(&cache)).expect("cached");
        assert_eq!(bytes, cached_bytes);
        assert_eq!(plain.program, cached.program);
        assert_eq!(
            structural_cost(&plain, &mut CostCache::default()).expect("C32"),
            structural_cost(&cached, &mut CostCache::default()).expect("cached C32")
        );
        let family = FamilyInputs {
            rows: 2,
            slots: vec![gam_mpd::operator_program::SlotValues::Raw(Array2::ones((
                2, 2,
            )))],
            layout: None,
        };
        let target = plain.execute(&family).expect("target").values[plain.program.output].clone();
        let episodes = [Episode {
            label: "clean".into(),
            group: "clean".into(),
            inputs: family,
            target_logits: target,
            scored: None,
        }];
        let device = Device::host();
        let a = resident_causal_fit::measure(&device, &plain.program, &episodes, 1_000_000)
            .expect("plain scores");
        let b = resident_causal_fit::measure(&device, &cached.program, &episodes, 1_000_000)
            .expect("cached scores");
        assert_eq!(
            serde_json::to_value(a).expect("plain"),
            serde_json::to_value(b).expect("cached")
        );
        assert_eq!(
            cache
                .decode_saved(&bytes, &source.program.declarations)
                .expect("saved decode")
                .to_bytes()
                .expect("ordinary replay"),
            bytes
        );
    }
    #[test]
    fn structural_mode_runs_persistent_reuse_with_finite_control_and_saved_replay() {
        use gam_mpd::operator_program::{
            exact_precision, Declarations, Interface, Law, Operator, Slot,
        };
        use ndarray::array;
        let dense = |name: &str, values: Array2<f64>| {
            Arc::new(
                Operator::dense(
                    name,
                    Interface::native(values.nrows()).expect("rows"),
                    Interface::native(values.ncols()).expect("cols"),
                    values.clone(),
                    exact_precision(values.iter().copied()).expect("precision"),
                    Default::default(),
                )
                .expect("dense"),
            )
        };
        let mut native = OperatorProgram {
            declarations: Declarations {
                domains: vec![],
                slots: vec![Slot::Raw { width: 1 }],
                parameters: 0,
            },
            bases: vec![],
            rules: vec![],
            operators: vec![
                dense("reader-a", array![[1.]]),
                dense("reader-b", array![[1.]]),
                dense("head", array![[1., 0.25], [-0.5, 1.]]),
            ],
            nodes: vec![
                Node::Raw { slot: 0 },
                Node::Affine {
                    terms: vec![(0, 0)],
                    bias: None,
                },
                Node::Pointwise {
                    input: 1,
                    laws: vec![Law::GeluTanh],
                },
                Node::Affine {
                    terms: vec![(0, 1)],
                    bias: None,
                },
                Node::Pointwise {
                    input: 3,
                    laws: vec![Law::GeluTanh],
                },
                Node::Concat { parts: vec![2, 4] },
                Node::Affine {
                    terms: vec![(5, 2)],
                    bias: None,
                },
            ],
            output: 6,
        };
        Arc::make_mut(&mut native.operators[2]).cols = Interface::new(
            [
                Interface::native(1).expect("scalar"),
                Interface::native(1).expect("scalar"),
            ]
            .iter()
            .flat_map(|i| i.groups().iter().copied())
            .collect(),
        )
        .expect("concat interface");
        if let OperatorBody::Dense { present, .. } =
            &mut Arc::make_mut(&mut native.operators[2]).body
        {
            *present = Array2::from_elem((1, 2), true);
        }
        let family = FamilyInputs {
            rows: 4,
            slots: vec![SlotValues::Raw(array![[-1.], [-0.25], [0.5], [1.25]])],
            layout: None,
        };
        let specs = vec![
            NativeControl::RetainedOperator {
                name: "reader-a".into(),
            },
            NativeControl::RetainedOperator {
                name: "reader-b".into(),
            },
        ];
        let cases = vec![
            Case {
                label: "clean".into(),
                group: "clean".into(),
                gains: vec![1., 1.],
                down_amplitudes: vec![],
                parameter_amplitudes: vec![],
            },
            Case {
                label: "signed edit".into(),
                group: "edit_a".into(),
                gains: vec![-0.5, 1.],
                down_amplitudes: vec![],
                parameter_amplitudes: vec![],
            },
        ];
        let mut cases = cases;
        cases.push(Case {
            label: "independent B edit".into(),
            group: "edit_b".into(),
            gains: vec![1., 2.],
            down_amplitudes: vec![],
            parameter_amplitudes: vec![],
        });
        let d = Device::host();
        let base = Artifact::native(&native).expect("native");
        let mapped = controls(&base, &native, &[], &specs).expect("control");
        let (targets, _) = family_teacher_targets(
            &d,
            &native,
            &[],
            &specs,
            &family,
            &cases,
            None,
            1 << 26,
            1 << 26,
        )
        .expect("literal teacher");
        let metric = |name: &str| Metric {
            name: name.into(),
            value: 1.,
        };
        let structural = StructuralSettings {
            settings: program_structure_search::Settings {
                region_limits: gam_mpd::program_regions::Limits {
                    max_internal_nodes: 2,
                    max_inputs: 2,
                    max_regions: 24,
                    max_states: 100,
                },
                max_depth: 2,
                max_callback_calls: 80,
                max_move_attempts: 200,
                beam_width: 8,
                max_frontier: 8,
                max_argument_bindings: 2,
                max_compound_pairs: 24,
                learned_dag_search: None,
                shared_dag_search: None,
                expression_search: None,
                preserve_native_places: vec![],
                joint_observation_places: None,
            },
            constraints: program_structure_search::Constraints {
                max_fidelity: vec![metric("maximum_group_mean_kl")],
                max_local_errors: vec![metric("sampled_d_local")],
                max_intervention_errors: vec![metric("clean"), metric("edit_a"), metric("edit_b")],
            },
            max_expression_evaluations: 0,
            native_response_weight: Some(1.),
        };
        let settings = Settings {
            export_sha256: "0".repeat(64),
            layers: 1,
            uses: vec![],
            width: 0,
            sequences: 1,
            context: 4,
            fixed_head_targets: None,
            teacher_numeric_bytes: 1 << 26,
            native_codec_bytes: 0,
            grammar: empty_grammar(),
            expression_ids: vec![],
            require_interior_learned: false,
            frozen_shared_body: None,
            native_initialization: false,
            joint_response_search: None,
            native_parameter_edits: None,
            down_edit_family: None,
            structural_search: None,
            controls: specs,
            cases,
            fit: FitSettings {
                iterations: 1,
                learning_rate: 0.001,
                beta1: 0.9,
                beta2: 0.999,
                epsilon: 1e-8,
                numeric_bytes: 1 << 26,
                schedule: None,
                arithmetic: resident_causal_fit::FitArithmetic::F64,
                exact_scan_every: 1,
            },
            seed: 1,
            evaluation: None,
        };
        let out =
            std::env::temp_dir().join(format!("mpd-structural-driver-{}", std::process::id()));
        if out.exists() {
            std::fs::remove_dir_all(&out).expect("old fixture");
        }
        std::fs::create_dir(&out).expect("output");
        structural_run(
            &d,
            &settings,
            &structural,
            &native,
            &native,
            &[],
            &family,
            &targets,
            &out,
            None,
            None,
            &mut CostCache::default(),
            None,
        )
        .expect("real structural callback");
        let report: Value = serde_json::from_slice(
            &std::fs::read(out.join("STRUCTURAL_SEARCH.json")).expect("searchreport"),
        )
        .expect("json");
        let records = report["report"]["admitted_candidates"]
            .as_array()
            .expect("lineage");
        assert!(
            records
                .iter()
                .any(|r| r["depth"] == 2 && r["mutation"]["kind"] == "reuse_existing_rule"),
            "{report}"
        );
        let attempts = report["report"]["attempts"].as_array().expect("attempts");
        assert!(attempts
            .iter()
            .any(|a| a["depth"] == 2 && a["mutation"]["kind"] == "reuse_existing_rule"));
        assert!(
            attempts
                .iter()
                .any(|a| a["mutation"]["kind"] == "extract_and_reuse"
                    && a["status"] == "fitted_admitted"
                    && a["trainable_operator_ids"]
                        .as_array()
                        .is_some_and(|ids| !ids.is_empty())),
            "compound move must reach actual jointly fitted Dense reuse before single scaffolding"
        );
        let mut fit_replay = false;
        for entry in std::fs::read_dir(&out).expect("directories") {
            let p = entry.expect("entry").path();
            if p.join("FIT.json").exists() {
                let fit: Value =
                    serde_json::from_slice(&std::fs::read(p.join("FIT.json")).expect("fit report"))
                        .expect("fit JSON");
                assert!(p.join("NATIVE_RESPONSE_TARGETS.json").exists());
                let initial = fit["initial"]["episodes"]
                    .as_array()
                    .expect("episode measurements");
                assert!(initial
                    .iter()
                    .all(|e| e["responses"].as_array().is_some_and(|r| !r.is_empty())));
                for episode in initial {
                    let kl = episode["mean_kl"].as_f64().expect("separate KL");
                    let response = episode["responses"]
                        .as_array()
                        .expect("response terms")
                        .iter()
                        .map(|r| r["weighted_loss"].as_f64().expect("response loss"))
                        .sum::<f64>();
                    assert!(
                        (episode["total_loss"].as_f64().expect("total") - kl - response).abs()
                            < 1e-12
                    );
                }
                let bytes = std::fs::read(p.join("program.artifact")).expect("savedartifact");
                let restored = Artifact::from_bytes(
                    &bytes,
                    &intervention_program::compile(&native, &mapped)
                        .expect("controlled source")
                        .program
                        .declarations,
                )
                .expect("ordinary replay");
                assert_eq!(restored.to_bytes().expect("bytes"), bytes);
                assert!(
                    restored.program.declarations.slots.len() > native.declarations.slots.len()
                );
                fit_replay = true;
            }
        }
        assert!(
            fit_replay,
            "reuse must execute resident fitter, not only structural metadata"
        );
        // Reuse the real structural-driver fixture with two upstream coordinates,
        // signed edits and an independent retained gain in the same episodes.
        let mut edit_settings = settings;
        edit_settings.controls = vec![NativeControl::RetainedOperator {
            name: "reader-b".into(),
        }];
        edit_settings.native_parameter_edits = Some(NativeEditSettings {
            target_operator: 0,
            directions: vec![vec![vec![0.5]], vec![vec![-0.25]]],
        });
        edit_settings.structural_search = Some(structural);
        edit_settings.cases = vec![
            Case {
                label: "clean".into(),
                group: "clean".into(),
                gains: vec![1.],
                down_amplitudes: vec![],
                parameter_amplitudes: vec![0., 0.],
            },
            Case {
                label: "positive".into(),
                group: "edit_a".into(),
                gains: vec![1.],
                down_amplitudes: vec![],
                parameter_amplitudes: vec![0.75, 0.],
            },
            Case {
                label: "negative mixed".into(),
                group: "edit_b".into(),
                gains: vec![2.],
                down_amplitudes: vec![],
                parameter_amplitudes: vec![-0.5, 0.25],
            },
        ];
        validate_parameter_edit_configuration(&edit_settings).expect("explicit signed edit design");
        assert!(
            reject_native_edit_target_gain(0, &[Control::GlobalOperatorScale { operator: 0 }])
                .is_err()
        );
        assert!(
            reject_native_edit_target_gain(0, &[Control::GlobalOperatorScale { operator: 1 }])
                .is_ok()
        );
        let edits = build_parameter_edits(
            &native,
            edit_settings.native_parameter_edits.as_ref().unwrap(),
        )
        .expect("upstream coordinates");
        let edit_native = &edits.compiled.program;
        let gains = controls(
            &Artifact::native(edit_native).unwrap(),
            edit_native,
            &[],
            &edit_settings.controls,
        )
        .unwrap();
        let controlled = intervention_program::compile(edit_native, &gains).unwrap();
        let original_map: Vec<_> = edits
            .node_mapping
            .iter()
            .map(|n| controlled.root_mapping[*n])
            .collect();
        let full = parameter_edit_episodes(
            &d,
            &edits,
            &controlled,
            edit_native,
            &gains,
            &native,
            &edit_settings,
            &family,
            &edit_settings.cases,
        )
        .expect("literal full targets");
        edit_settings.fixed_head_targets = Some(FixedHeadSettings { tile_rows: 2 });
        let compact = parameter_edit_episodes(
            &d,
            &edits,
            &controlled,
            edit_native,
            &gains,
            &native,
            &edit_settings,
            &family,
            &edit_settings.cases,
        )
        .expect("literal compact targets");
        assert_eq!(compact.target_metadata["full_logit_target_bytes"], 0);
        for (case, episode) in edit_settings.cases.iter().zip(full.full.as_ref().unwrap()) {
            let literal = edits.literal_native(&case.parameter_amplitudes).unwrap();
            let cs = controls(
                &Artifact::native(&literal).unwrap(),
                &literal,
                &[],
                &edit_settings.controls,
            )
            .unwrap();
            let graph = intervention_program::compile(&literal, &cs).unwrap();
            let expected = graph
                .program
                .execute(
                    &graph
                        .family(&family, &values(&literal, &cs, family.rows, case).unwrap())
                        .unwrap(),
                    false,
                )
                .unwrap();
            assert_eq!(episode.target_logits, expected.values[graph.program.output]);
        }
        use program_learned_dag::{Expr as J, TypeRef};
        let affine = |parameter, input| J::Affine {
            parameter,
            output: TypeRef::Exit(0),
            input: Box::new(input),
            bias: false,
        };
        let edited_terms = J::Binary(
            Binary::Add,
            Box::new(affine(
                1,
                J::Binary(
                    Binary::Multiply,
                    Box::new(J::Argument(0)),
                    Box::new(J::Argument(1)),
                ),
            )),
            Box::new(affine(
                2,
                J::Binary(
                    Binary::Multiply,
                    Box::new(J::Argument(0)),
                    Box::new(J::Argument(2)),
                ),
            )),
        );
        let equation = J::Unary(
            Unary::Relu,
            Box::new(J::Binary(
                Binary::Add,
                Box::new(affine(0, J::Argument(0))),
                Box::new(edited_terms),
            )),
        );
        let search = program_learned_dag::Settings {
            latent_widths: vec![1],
            unary: vec![Unary::Relu],
            binary: vec![Binary::Add, Binary::Multiply],
            affine_bias: false,
            require_shared: false,
            max_operations: 8,
            max_affine_parameters: 3,
            max_parameter_elements: 10,
            max_expression_states: 100,
            max_tuple_checks: 100,
            max_tuples: 10,
            max_body_nodes: 20,
            seed: 3,
        };
        let mut compiled = program_learned_dag::compile_program(
            &vec![Interface::native(1).unwrap(); 3],
            &[Interface::native(1).unwrap(), Interface::native(1).unwrap()],
            &[equation, J::Argument(0)],
            &search,
        )
        .unwrap();
        compiled.program.output = compiled.output_nodes[0];
        let edit_inputs: Vec<_> = edits
            .compiled
            .control_slots
            .iter()
            .map(|slot| {
                let root = edit_native
                    .nodes
                    .iter()
                    .position(|n| matches!(n,Node::Raw {slot:s} if *s==slot.slot))
                    .unwrap();
                controlled.root_mapping[root]
            })
            .collect();
        let candidate = Artifact::native(&controlled.program)
            .unwrap()
            .replace_function_inputs(
                "learned nonlinear edit response",
                &compiled.program,
                &[original_map[0], edit_inputs[0], edit_inputs[1]],
                original_map[2],
            )
            .unwrap();
        let trainable = joint_graft_parameters(&candidate, original_map[2], &compiled).unwrap();
        let (responses, provenance) = parameter_edit_response_targets(
            &d,
            &edits,
            &controlled,
            &native,
            &edit_settings,
            &family,
            &edit_settings.cases,
            &candidate,
            &full.metadata,
            1.,
            1 << 26,
        )
        .expect("literal native state observations");
        assert_eq!(
            provenance["literal_original_observations"][original_map[2].to_string()],
            2
        );
        let scale = responses["clean"][0].scale;
        assert!(responses
            .values()
            .all(|r| r[0].scale.to_bits() == scale.to_bits()));
        let fit = full
            .fit(
                &d,
                &candidate.program,
                Some(&responses),
                &trainable,
                edit_settings.fit.clone(),
            )
            .expect("full edited fit");
        let compact_fit = compact
            .fit(
                &d,
                &candidate.program,
                Some(&responses),
                &trainable,
                edit_settings.fit.clone(),
            )
            .expect("compact edited fit");
        for id in &trainable {
            assert!(fit.program.operators[*id]
                .matrix()
                .iter()
                .zip(compact_fit.program.operators[*id].matrix().iter())
                .all(|(a, b)| (a - b).abs() < 1e-9));
        }
        let mut fitted = candidate;
        fitted.program = fit.program;
        let (saved, bytes) = canonical(&fitted).unwrap();
        assert_eq!(
            Artifact::from_bytes(&bytes, &controlled.program.declarations)
                .unwrap()
                .to_bytes()
                .unwrap(),
            bytes
        );
        let (_, replay_provenance) = parameter_edit_response_targets(
            &d,
            &edits,
            &controlled,
            &native,
            &edit_settings,
            &family,
            &edit_settings.cases,
            &saved,
            &full.metadata,
            1.,
            1 << 26,
        )
        .unwrap();
        assert_eq!(
            replay_provenance["literal_original_observations"],
            provenance["literal_original_observations"]
        );
        assert!(compact
            .measure(&d, &saved.program, Some(&responses), 1 << 26)
            .unwrap()
            .objective
            .is_finite());
        // The structural source is the edit-conditioned graph, so region discovery
        // sees edit controls inside native nonlinear computations.
        let edit_out = out.join("upstream-edits");
        std::fs::create_dir(&edit_out).unwrap();
        structural_run(
            &d,
            &edit_settings,
            edit_settings.structural_search.as_ref().unwrap(),
            edit_native,
            &native,
            &[],
            &family,
            &[],
            &edit_out,
            None,
            None,
            &mut CostCache::default(),
            Some(&edits),
        )
        .expect("actual edited structural runner");
        let report: Value = serde_json::from_slice(
            &std::fs::read(edit_out.join("CONTROLLED_SOURCE.json")).unwrap(),
        )
        .unwrap();
        assert_eq!(
            report["original_to_edit_lowered_root_mapping"],
            json!(edits.node_mapping)
        );
        let mut bad = edit_settings.native_parameter_edits.take().unwrap();
        bad.target_operator = 2;
        assert!(
            build_parameter_edits(&native, &bad).is_err(),
            "readout edits refused"
        );
        std::fs::remove_dir_all(out).expect("cleanup");
    }
    #[test]
    fn response_capture_uses_native_signed_episodes_and_inlined_multi_consumer_nodes() {
        use gam_mpd::operator_program::{Declarations, Interface, Law, Rule, Slot, SlotValues};
        let native = OperatorProgram {
            declarations: Declarations {
                domains: vec![],
                slots: vec![Slot::Raw { width: 2 }, Slot::Raw { width: 2 }],
                parameters: 0,
            },
            bases: vec![],
            operators: vec![],
            rules: vec![Rule {
                name: "signed activation control".into(),
                inputs: vec![Interface::native(2).unwrap(); 2],
                nodes: vec![
                    Node::Param { index: 0 },
                    Node::Param { index: 1 },
                    Node::Hadamard { left: 0, right: 1 },
                ],
                output: 2,
            }],
            nodes: vec![
                Node::Raw { slot: 0 },
                Node::Raw { slot: 1 },
                Node::Call {
                    rule: 0,
                    arguments: vec![0, 1],
                },
                Node::Pointwise {
                    input: 2,
                    laws: vec![Law::Relu],
                },
                Node::Pointwise {
                    input: 2,
                    laws: vec![Law::Silu],
                },
                Node::Concat { parts: vec![3, 4] },
            ],
            output: 5,
        };
        let mut candidate = Artifact::native(&native).unwrap();
        candidate.blocks = [3, 4]
            .into_iter()
            .map(|node| gam_mpd::artifact::Binding {
                name: format!("exit_{node}"),
                native_reads: vec![0, 1],
                native_write: node,
                reads: vec![0, 1],
                write: node,
            })
            .collect();
        candidate.program.nodes[3] = Node::Pointwise {
            input: 2,
            laws: vec![Law::Zero],
        };
        let episodes = [1., -0.5]
            .into_iter()
            .enumerate()
            .map(|(index, gain)| {
                let inputs = FamilyInputs {
                    rows: 2,
                    layout: None,
                    slots: vec![
                        SlotValues::Raw(ndarray::array![[1., -2.], [3., 0.]]),
                        SlotValues::Raw(Array2::from_elem((2, 2), gain)),
                    ],
                };
                let target_logits =
                    native.execute(&inputs, false).unwrap().values[native.output].clone();
                Episode {
                    label: format!("case{index}"),
                    group: "train".into(),
                    inputs,
                    target_logits,
                    scored: None,
                }
            })
            .collect::<Vec<_>>();
        let (labels, provenance) =
            native_response_targets(&Device::host(), &native, &candidate, &episodes, 2., 1 << 24)
                .unwrap();
        for episode in &episodes {
            let trace = native.execute(&episode.inputs, false).unwrap();
            for target in &labels[&episode.label] {
                assert_eq!(target.values, trace.values[target.source_node]);
                assert_eq!(target.weight, 1.);
            }
        }
        assert_ne!(labels["case0"][0].values, labels["case1"][0].values);
        assert_eq!(labels["case0"][0].scale, labels["case1"][0].scale);
        assert!(
            labels["case0"][0].values.iter().any(|v| *v != 0.),
            "labels must not come from candidate's zeroed consumer"
        );
        assert_eq!(
            labels["case1"][0].values,
            ndarray::array![[0., 1.], [0., 0.]]
        );
        let planned = provenance["planned_numeric_bytes"].as_u64().unwrap() as usize;
        assert!(native_response_targets(
            &Device::host(),
            &native,
            &candidate,
            &episodes,
            2.,
            planned - 1
        )
        .is_err());
    }
    #[test]
    fn dominated_equation_hypotheses_freeze_without_heldout_selection() {
        let region = gam_mpd::program_regions::Region {
            native_reads: vec![0],
            native_write: 1,
            current_reads: vec![0],
            current_write: 1,
            current_internal_nodes: vec![1],
            internal_native_places: vec![1],
            source_program_nodes: 2,
        };
        let evaluation = |cost| StructureEvaluation {
            fidelity: vec![Metric {
                name: "kl".into(),
                value: 0.,
            }],
            local_errors: vec![Metric {
                name: "local".into(),
                value: 0.,
            }],
            intervention_errors: vec![Metric {
                name: "edits".into(),
                value: 0.,
            }],
            description_bits: cost,
        };
        let joint_region = gam_mpd::program_joint_regions::Region {
            anchor_mode: gam_mpd::program_joint_regions::AnchorMode::Interior,
            native_reads: vec![0],
            current_reads: vec![0],
            native_writes: vec![1],
            current_writes: vec![1],
            current_internal_nodes: vec![1],
            producer_native: 1,
            erased_native_places: vec![],
            source_program_nodes: 2,
        };
        let records = vec![
            program_structure_search::CandidateRecord {
                id: 0,
                parent_id: None,
                depth: 0,
                mutation: None,
                evaluation: evaluation(1.),
                encoded_size_bytes: 1,
            },
            program_structure_search::CandidateRecord {
                id: 7,
                parent_id: Some(0),
                depth: 1,
                mutation: Some(Mutation::SynthesizeExpression {
                    region: region.clone(),
                    expression: Expr::Argument(0),
                    native_arguments: vec![0],
                }),
                evaluation: evaluation(100.),
                encoded_size_bytes: 10,
            },
            program_structure_search::CandidateRecord {
                id: 9,
                parent_id: Some(0),
                depth: 1,
                mutation: Some(Mutation::SynthesizeSharedDAG {
                    region: joint_region.clone(),
                    expressions: vec![Expr::Argument(0)],
                    native_arguments: vec![0],
                }),
                evaluation: evaluation(200.),
                encoded_size_bytes: 20,
            },
            program_structure_search::CandidateRecord {
                id: 11,
                parent_id: Some(0),
                depth: 1,
                mutation: Some(Mutation::SynthesizeLearnedDAG {
                    region: joint_region,
                    expressions: vec![gam_mpd::program_learned_dag::Expr::Argument(0)],
                    native_arguments: vec![0],
                }),
                evaluation: evaluation(300.),
                encoded_size_bytes: 30,
            },
        ];
        assert_eq!(frozen_structural_ids(&[0], &records, 0), (vec![0], vec![]));
        assert_eq!(
            frozen_structural_ids(&[0], &records, 1),
            (vec![0, 7], vec![7])
        );
        assert_eq!(
            frozen_structural_ids(&[0], &records, 2),
            (vec![0, 7, 9], vec![7, 9])
        );
        assert_eq!(
            frozen_structural_ids(&[0], &records, 3),
            (vec![0, 7, 9, 11], vec![7, 9, 11])
        );
        assert_eq!(
            frozen_structural_ids(&[0, 11], &records, 1),
            (vec![0, 11, 7], vec![7])
        );
    }
    #[test]
    fn expression_search_rejects_clean_cancellation_under_independent_native_controls() {
        controlled_cancellation_gate(false, false);
    }
    #[test]
    fn joint_dag_search_preserves_two_observable_consumers_under_controls_and_replay() {
        // Implementation calibration only: this supplied synthetic teacher is not evidence of native algorithm discovery.
        controlled_cancellation_gate(true, false);
    }
    fn controlled_cancellation_gate(joint: bool, square: bool) {
        let gate_started = Instant::now();
        use gam_mpd::operator_program::{exact_precision, Declarations, Interface, Operator, Slot};
        use ndarray::array;
        let dense = |name: &str, v: Array2<f64>| {
            Arc::new(
                Operator::dense(
                    name,
                    Interface::native(v.nrows()).expect("rows"),
                    Interface::native(v.ncols()).expect("cols"),
                    v.clone(),
                    exact_precision(v.iter().copied()).expect("precision"),
                    Default::default(),
                )
                .expect("dense"),
            )
        };
        let mut native = OperatorProgram {
            declarations: Declarations {
                domains: vec![],
                slots: vec![Slot::Raw { width: 1 }],
                parameters: 0,
            },
            operators: vec![
                dense("A", array![[1.]]),
                dense("B", array![[1.]]),
                dense("plus", array![[1.]]),
                dense("minus", array![[-1.]]),
                dense("head", array![[1.], [-1.]]),
            ],
            bases: vec![],
            rules: vec![],
            nodes: vec![
                Node::Raw { slot: 0 },
                Node::Affine {
                    terms: vec![(0, 0)],
                    bias: None,
                },
                Node::Affine {
                    terms: vec![(0, 1)],
                    bias: None,
                },
                Node::Affine {
                    terms: vec![(1, 2), (2, 3)],
                    bias: None,
                },
                Node::Affine {
                    terms: vec![(3, 4)],
                    bias: None,
                },
            ],
            output: 4,
        };
        if joint {
            native.operators.pop();
            native.nodes.truncate(4);
            native.nodes.extend([
                if square {
                    Node::Hadamard { left: 3, right: 3 }
                } else {
                    Node::Pointwise {
                        input: 3,
                        laws: vec![gam_mpd::operator_program::Law::Relu],
                    }
                },
                Node::Pointwise {
                    input: 3,
                    laws: vec![gam_mpd::operator_program::Law::Silu],
                },
                Node::Concat { parts: vec![4, 5] },
            ]);
            native.output = 6;
        }
        let family = FamilyInputs {
            rows: 3,
            slots: vec![SlotValues::Raw(array![[-1.], [0.5], [1.5]])],
            layout: None,
        };
        let specs = vec![
            NativeControl::RetainedOperator { name: "A".into() },
            NativeControl::RetainedOperator { name: "B".into() },
        ];
        let cases = vec![
            Case {
                label: "clean".into(),
                group: "clean".into(),
                gains: vec![1., 1.],
                down_amplitudes: vec![],
                parameter_amplitudes: vec![],
            },
            Case {
                label: "A".into(),
                group: "edited".into(),
                gains: vec![2., 1.],
                down_amplitudes: vec![],
                parameter_amplitudes: vec![],
            },
            Case {
                label: "B negative".into(),
                group: "edited".into(),
                gains: vec![1., -0.5],
                down_amplitudes: vec![],
                parameter_amplitudes: vec![],
            },
        ];
        let d = Device::host();
        let (targets, _) = family_teacher_targets(
            &d,
            &native,
            &[],
            &specs,
            &family,
            &cases,
            None,
            1 << 26,
            1 << 26,
        )
        .expect("targets");
        assert!(targets[0].iter().all(|v| *v == 0.));
        assert!(targets[1].iter().any(|v| v.abs() > 0.5));
        let metric = |name: &str| Metric {
            name: name.into(),
            value: 1e-5,
        };
        let mut structural = StructuralSettings {
            settings: program_structure_search::Settings {
                region_limits: gam_mpd::program_regions::Limits {
                    max_internal_nodes: 1,
                    max_inputs: 2,
                    max_regions: 20,
                    max_states: 40,
                },
                max_depth: 1,
                max_callback_calls: 80,
                max_move_attempts: 300,
                beam_width: 4,
                max_frontier: 4,
                max_argument_bindings: 2,
                max_compound_pairs: 8,
                preserve_native_places: vec![],
                joint_observation_places: None,
                learned_dag_search: None,
                shared_dag_search: None,
                expression_search: Some(program_structure_search::ExpressionSettings {
                    grammar: Grammar {
                        arguments: 2,
                        max_operations: 1,
                        max_expressions: 100,
                        unary: vec![],
                        binary: vec![composed_rule_search::Binary::Subtract],
                        affine: false,
                    },
                    max_enumerations_per_parent: 20,
                    max_move_attempts: 100,
                    max_callback_calls: 60,
                }),
            },
            constraints: program_structure_search::Constraints {
                max_fidelity: vec![metric("maximum_group_mean_kl")],
                max_local_errors: vec![Metric {
                    name: "sampled_d_local".into(),
                    value: 10.,
                }],
                max_intervention_errors: vec![metric("clean"), metric("edited")],
            },
            max_expression_evaluations: 2,
            native_response_weight: None,
        };
        if joint {
            // This fixture measures fanout discovery; observable-chain coverage has separate tests.
            structural.settings.joint_observation_places = Some(vec![]);
            structural.constraints.max_local_errors[0].value = 1e-6;
            structural.settings.expression_search = None;
            structural.settings.max_callback_calls = 2000;
            structural.settings.max_move_attempts = 10000;
            structural.settings.shared_dag_search =
                Some(program_structure_search::SharedDAGSettings {
                    region_limits: gam_mpd::program_joint_regions::Limits {
                        max_internal_nodes: 3,
                        max_inputs: 2,
                        max_exits: 2,
                        max_regions: 30,
                        max_states: 200,
                    },
                    grammar: Grammar {
                        arguments: 2,
                        max_operations: 3,
                        max_expressions: 400,
                        unary: vec![composed_rule_search::Unary::Silu],
                        binary: vec![
                            composed_rule_search::Binary::Subtract,
                            composed_rule_search::Binary::Multiply,
                        ],
                        affine: false,
                    },
                    max_enumerations_per_parent: 30,
                    max_tuple_checks: 160000,
                    max_tuples_per_enumeration: 10000,
                    max_body_nodes: 12,
                    max_move_attempts: 9000,
                    max_callback_calls: 1900,
                });
        }
        if joint && !square {
            let proposal = structural
                .settings
                .shared_dag_search
                .as_mut()
                .expect("joint settings");
            // Complete two-argument, two-operation grammar (58 trees), not a seeded formula.
            proposal.grammar.max_operations = 2;
            proposal.grammar.max_expressions = 100;
            proposal.grammar.unary = vec![
                composed_rule_search::Unary::Relu,
                composed_rule_search::Unary::Silu,
            ];
            proposal.grammar.binary = vec![composed_rule_search::Binary::Subtract];
            let inventory = gam_mpd::composed_rule_search::enumerate(&proposal.grammar)
                .expect("complete grammar");
            assert!(!inventory.truncated);
            assert_eq!(inventory.intermediate_expressions, 58);
            proposal.max_tuple_checks = 58 * 58;
            proposal.max_tuples_per_enumeration = 58 * 58;
        }
        let settings = Settings {
            export_sha256: "0".repeat(64),
            layers: 1,
            uses: vec![],
            width: 0,
            sequences: 1,
            context: 3,
            fixed_head_targets: None,
            teacher_numeric_bytes: 1 << 26,
            native_codec_bytes: 0,
            grammar: empty_grammar(),
            expression_ids: vec![],
            require_interior_learned: false,
            frozen_shared_body: None,
            native_initialization: false,
            joint_response_search: None,
            native_parameter_edits: None,
            down_edit_family: None,
            structural_search: None,
            controls: specs,
            cases,
            fit: FitSettings {
                iterations: 1,
                learning_rate: 0.001,
                beta1: 0.9,
                beta2: 0.999,
                epsilon: 1e-8,
                numeric_bytes: 1 << 26,
                schedule: None,
                arithmetic: resident_causal_fit::FitArithmetic::F64,
                exact_scan_every: 1,
            },
            seed: 1,
            evaluation: None,
        };
        let out = std::env::temp_dir().join(format!(
            "mpd-structural-expression-driver-{joint}-{square}-{}",
            std::process::id()
        ));
        if out.exists() {
            std::fs::remove_dir_all(&out).expect("old");
        }
        std::fs::create_dir(&out).expect("out");
        structural_run(
            &d,
            &settings,
            &structural,
            &native,
            &native,
            &[],
            &family,
            &targets,
            &out,
            None,
            None,
            &mut CostCache::default(),
            None,
        )
        .expect("actual structural driver");
        let report: Value = serde_json::from_slice(
            &std::fs::read(out.join("STRUCTURAL_SEARCH.json")).expect("report"),
        )
        .expect("json");
        let attempts = report["report"]["attempts"].as_array().expect("attempts");
        let kind = if joint {
            "synthesize_shared_dag"
        } else {
            "synthesize_expression"
        };
        assert!(
            attempts
                .iter()
                .any(|a| a["mutation"]["kind"] == kind && a["status"] == "fitted_admitted"),
            "no admitted {kind}; counts={}",
            report["report"]["counts"]
        );
        let mut canceled_wrong = false;
        for attempt in attempts.iter().filter(|a| a["mutation"]["kind"] == kind) {
            let id = attempt["attempt_id"].as_u64().expect("id");
            let p = out.join(format!("structural-attempt-{id:06}/TRAIN.json"));
            if !p.exists() {
                let root = out.join(format!("structural-attempt-{id:06}"));
                if joint && !canceled_wrong && root.join("LOCAL_SCREEN.json").exists() {
                    let defs = controls(
                        &Artifact::native(&native).expect("source"),
                        &native,
                        &[],
                        &settings.controls,
                    )
                    .expect("defs");
                    let lowered =
                        intervention_program::compile(&native, &defs).expect("lowered controls");
                    let bytes =
                        std::fs::read(root.join("program.artifact")).expect("screened artifact");
                    let candidate = Artifact::from_bytes(&bytes, &lowered.program.declarations)
                        .expect("screened decode");
                    let clean = episodes(
                        &lowered,
                        &native,
                        &defs,
                        &family,
                        &settings.cases[..1],
                        &targets[..1],
                        None,
                    )
                    .expect("clean case");
                    let output = candidate
                        .program
                        .execute(&clean[0].inputs, false)
                        .expect("own clean forward")
                        .values[candidate.program.output]
                        .clone();
                    let clean_kl = (0..family.rows)
                        .map(|r| gam_mpd::acceptance::kl_logits(targets[0].row(r), output.row(r)).0)
                        .sum::<f64>()
                        / family.rows as f64;
                    if clean_kl.abs() < 1e-8 {
                        assert_ne!(attempt["status"], "fitted_admitted");
                        canceled_wrong = true;
                    }
                }
                continue;
            }
            let m: Value =
                serde_json::from_slice(&std::fs::read(p).expect("measure")).expect("json");
            if m["groups"]["clean"].as_f64().expect("clean").abs() < 1e-8
                && m["groups"]["edited"].as_f64().expect("edited") > 0.01
            {
                assert_ne!(attempt["status"], "fitted_admitted");
                canceled_wrong = true;
            }
        }
        assert!(
            canceled_wrong,
            "clean-equivalent cancellation must be rejected by independently controlled Local or KL"
        );
        let frozen: Value = serde_json::from_slice(
            &std::fs::read(out.join("FROZEN_EVALUATION_IDS.json")).expect("freeze"),
        )
        .expect("json");
        assert!(!frozen["expression_hypotheses"]
            .as_array()
            .expect("hypotheses")
            .is_empty());
        if joint {
            let control_defs = controls(
                &Artifact::native(&native).expect("native"),
                &native,
                &[],
                &settings.controls,
            )
            .expect("controls");
            let compiled =
                intervention_program::compile(&native, &control_defs).expect("compile controls");
            let expected: Vec<_> = [4, 5].iter().map(|n| compiled.root_mapping[*n]).collect();
            // Freeze by training admission before constructing any heldout targets.
            let chosen = attempts
                .iter()
                .find(|a| {
                    a["mutation"]["kind"] == kind
                        && a["status"] == "fitted_admitted"
                        && a["mutation"]["region"]["native_writes"]
                            .as_array()
                            .is_some_and(|w| {
                                w.len() == 2 && expected.iter().all(|n| w.contains(&json!(n)))
                            })
                })
                .expect("joint equation admission");
            let id = chosen["attempt_id"].as_u64().expect("attempt");
            save(&out.join("GATE_FROZEN_JOINT_ATTEMPT.json"), &json!({"attempt_id":id,"selection":"first training-admitted shared DAG replacing both declared observable consumers; implementation calibration only"})).expect("freeze gate before heldout targets");
            let path = out.join(format!("structural-attempt-{id:06}/program.artifact"));
            let bytes = std::fs::read(path).expect("saved joint");
            let decoded =
                Artifact::from_bytes(&bytes, &compiled.program.declarations).expect("decode");
            assert_eq!(decoded.to_bytes().expect("reencode"), bytes);
            let exits: Vec<_> = decoded.blocks.iter().map(|b| b.native_write).collect();
            assert!(
                expected.iter().all(|n| exits.contains(n)),
                "observable consumers must have Local relations: {exits:?} vs {expected:?}"
            );
            let heldout = FamilyInputs {
                rows: 3,
                slots: vec![SlotValues::Raw(array![[0.3], [-0.8], [2.1]])],
                layout: None,
            };
            let heldout_cases = vec![Case {
                label: "unseen combined signed controls".into(),
                group: "heldout".into(),
                gains: vec![-1.2, 0.7],
                down_amplitudes: vec![],
                parameter_amplitudes: vec![],
            }];
            let (targets, _) = family_teacher_targets(
                &d,
                &native,
                &[],
                &settings.controls,
                &heldout,
                &heldout_cases,
                None,
                1 << 26,
                1 << 26,
            )
            .expect("heldout targets");
            let heldout_episodes = episodes(
                &compiled,
                &native,
                &control_defs,
                &heldout,
                &heldout_cases,
                &targets,
                None,
            )
            .expect("heldout episodes");
            let measure =
                resident_causal_fit::measure(&d, &decoded.program, &heldout_episodes, 1 << 26)
                    .expect("joint heldout KL");
            assert!(
                measure.objective < 1e-6,
                "heldout joint KL {}",
                measure.objective
            );
            let local = gam_mpd::acceptance::Local::new(
                &compiled.program,
                heldout_episodes[0].inputs.clone(),
                None,
                3,
            );
            let local_measure = local.measure(&decoded).expect("heldout Local");
            assert!(local_measure.worst().expect("exits").worst < 1e-6);
            save(&out.join("GATE_HELDOUT.json"),&json!({"scope":"synthetic implementation gate only; not native algorithm discovery","selected_attempt":id,"saved_artifact_sha256":sha256(&out.join(format!("structural-attempt-{id:06}/program.artifact"))).expect("SHA"),"ordinary_replay_bytes_equal":true,"heldout_kl":measure,"heldout_local":local_measure,"observable_native_writes":expected})).expect("save gate evidence");
        }
        if joint {
            eprintln!(
                "joint DAG implementation gate seconds={} counts={}",
                gate_started.elapsed().as_secs_f64(),
                report["report"]["counts"]
            );
        }
        if joint {
            eprintln!("joint gate evidence retained at {}", out.display());
        } else {
            std::fs::remove_dir_all(out).expect("cleanup");
        }
    }
}
