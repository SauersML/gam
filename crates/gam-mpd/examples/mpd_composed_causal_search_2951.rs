//! Fit structural proposals in the autonomous native LM on clean AND intervened logits.
//! EXPORT SETTINGS.json FRESH_OUT host|cuda [HELDOUT_EXPORT]. Training proposal search, not acceptance.
use gam_gpu::{GpuPolicy, tensor::Device};
use gam_mpd::{
    acceptance::{CostCache, structural_cost},
    artifact::Artifact,
    canonical_artifact::CanonicalArtifactCache,
    coder_capture::sha256,
    composed_rule_search::{self, Grammar, UseSpec},
    device_program::DeviceProgram,
    down_edit_family::{self, Direction, Family as DownFamily},
    import::import_language_model,
    intervention_program::{self, Control, ControlValue},
    operator_program::{FamilyInputs, Node, OperatorBody, OperatorProgram, remap_node},
    parameter_response_program,
    resident_causal_fit::{self, Episode, Settings as FitSettings},
    run_check::{LayerNodes, layer_nodes, split_sites},
};
use ndarray::{Array1, Array2};
use serde::Deserialize;
use serde_json::{Value, json};
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
}
#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct Settings {
    export_sha256: String,
    layers: usize,
    uses: Vec<usize>,
    width: usize,
    sequences: usize,
    context: usize,
    /// Host label cache plus numeric native forward buffers; excludes library/allocator overhead.
    teacher_numeric_bytes: usize,
    /// Immutable native codewords and decoded buffers only; zero disables cache.
    #[serde(default)]
    native_codec_bytes: usize,
    grammar: Grammar,
    /// Explicit finite subinventory for separate, reproducible compute allocations.
    expression_ids: Vec<usize>,
    #[serde(default)]
    require_interior_learned: bool,
    #[serde(default)]
    frozen_shared_body: Option<FrozenSharedBody>,
    #[serde(default)]
    native_initialization: bool,
    #[serde(default)]
    down_edit_family: Option<DownSettings>,
    controls: Vec<NativeControl>,
    cases: Vec<Case>,
    fit: FitSettings,
    seed: u64,
    #[serde(default)]
    evaluation: Option<Evaluation>,
}
fn save(path: &Path, value: &Value) -> Result<(), String> {
    std::fs::write(
        path,
        serde_json::to_vec_pretty(value).map_err(|e| e.to_string())?,
    )
    .map_err(|e| e.to_string())
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
                c.gains.iter().all(|g| *g == 1.) && c.down_amplitudes.iter().all(|a| *a == 0.)
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
        return Err(
            "graft parameter count differs; unsupported parameter specialization or loss".into(),
        );
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
fn run() -> Result<(), String> {
    let args: Vec<_> = std::env::args().skip(1).collect();
    if args.len() != 4 && args.len() != 5 {
        return Err("EXPORT SETTINGS.json FRESH_OUT host|cuda [HELDOUT_EXPORT]".into());
    }
    let export = Path::new(&args[0]);
    let config_path = Path::new(&args[1]);
    let out = Path::new(&args[2]);
    let config_bytes = std::fs::read(config_path).map_err(|e| e.to_string())?;
    let settings: Settings = serde_json::from_slice(&config_bytes).map_err(|e| e.to_string())?;
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
        || settings.uses.is_empty()
        || settings.width == 0
        || settings.context == 0
        || settings.sequences == 0
        || settings.uses.iter().copied().collect::<BTreeSet<_>>().len() != settings.uses.len()
        || settings.uses.iter().any(|&i| i >= settings.layers)
        || (settings.controls.is_empty() && settings.down_edit_family.is_none())
    {
        return Err("positive dimensions, unique uses/cases and explicit clean plus intervention cases required".into());
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
    let sharing = selection(
        &inventory,
        &settings.expression_ids,
        settings.require_interior_learned,
    )?;
    if settings.frozen_shared_body.is_some() && settings.expression_ids.len() != 1 {
        return Err("transfer config requires one frozen expression ID".into());
    }
    let export_record: Value = serde_json::from_slice(
        &std::fs::read(export.join("export.json")).map_err(|e| e.to_string())?,
    )
    .map_err(|e| e.to_string())?;
    let checkpoint = export_record["source"]["checkpoint_sha256"]
        .as_str()
        .ok_or("native checkpoint SHA absent")?;
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
    let started = Instant::now();
    let imported = import_language_model(export, settings.sequences, settings.context)?;
    let original_native = split_sites(&imported.program)?;
    let original_layers = layer_nodes(&original_native, settings.layers)?;
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
    let native = down
        .as_ref()
        .map_or_else(|| original_native.clone(), |f| f.program.clone());
    let layers = down.as_ref().map_or_else(
        || original_layers.clone(),
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
    let (targets, teacher_plan) = family_teacher_targets(
        &d,
        &original_native,
        &original_layers,
        &settings.controls,
        family,
        &settings.cases,
        down.as_ref(),
        settings.fit.numeric_bytes,
        settings.teacher_numeric_bytes,
    )?;
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
        "objective":"maximum named-group mean of episode mean teacher-to-candidate KL; all causal sequence rows scored",
        "native_supervision":"native logits only; candidate runs its complete autonomous states",
        "teacher_planned_numeric_bytes":teacher_plan,
        "native_codec_bytes":settings.native_codec_bytes,"native_codec_initialization_seconds":native_codec_initialization_seconds,"native_codec_preflight":native_codec.as_ref().map(|c|c.preflight()),"native_codec_stats":native_codec.as_ref().map(|c|c.stats()),
        "native_codec_budget_scope":"packed native codewords plus cached decoded numeric buffers only; excludes caller-owned source arrays, per-operator construction/lattice temporaries, full messages, label/execution buffers, metadata and allocator/library overhead; construction excluded from warm timings",
        "literal_down_arithmetic":"binary64 stored D updates in declared order D += a*u*v; original graph executes updated matrix; no float32 checkpoint-store or augmented-branch bitwise equivalence claim",
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
    let native_episodes = episodes(
        &saved_native_graph,
        &saved_native.program,
        &target_controls,
        family,
        &settings.cases,
        &targets,
        down.as_ref(),
    )?;
    let native_measure = resident_causal_fit::measure(
        &d,
        &saved_native_graph.program,
        &native_episodes,
        settings.fit.numeric_bytes,
    )?;
    save(
        &out.join("NATIVE_TRAIN.json"),
        &serde_json::to_value(&native_measure).map_err(|e| e.to_string())?,
    )?;
    drop(native_episodes);
    let mut journal =
        std::fs::File::create(out.join("journal.jsonl")).map_err(|e| e.to_string())?;
    let mut rows = vec![
        json!({"id":"native","c32":native_cost,"training_kl":native_measure.objective,"status":"training_measured","artifact_sha256":sha256(&out.join("native.artifact"))?,"control_map_sha256":sha256(&out.join("NATIVE_CONTROL_MAP.json"))?}),
    ];
    for &id in &settings.expression_ids {
        for shared in [true, false] {
            if !shared && use_specs.len() == 1 {
                continue;
            }
            let arm = if shared { "shared" } else { "untied" };
            let name = format!("expression{id}-{arm}");
            let root = out.join(&name);
            std::fs::create_dir(&root).map_err(|e| e.to_string())?;
            save(
                &root.join("DECLARATION.json"),
                &json!({"expression_id":id,"expression":inventory.expressions[id],"sharing_analysis":sharing.analyses[id],"arm":arm,"uses":settings.uses,"response_uses":direction_count,"body_export_supported":down.is_none(),"transfer":frozen_body.is_some(),"freeze_policy":if frozen_body.is_some() && shared {"body frozen; new maps only"}else if frozen_body.is_some(){"same initial body; independent per-use body adaptation"}else{"all proposal coefficients trainable"}}),
            )?;
            let attempt = (|| -> Result<Value, String> {
                let compile = if shared {
                    composed_rule_search::compile
                } else {
                    composed_rule_search::compile_untied
                };
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
                let graft_started = Instant::now();
                let mut candidate = base.clone();
                for (slot, &layer) in settings.uses.iter().enumerate() {
                    let clean = composed_rule_search::function(&proposal, slot)?;
                    if settings
                        .down_edit_family
                        .as_ref()
                        .is_some_and(|config| config.layer == layer)
                    {
                        let down = down.as_ref().ok_or("declared down family absent")?;
                        let responses = (0..direction_count)
                            .map(|j| {
                                composed_rule_search::function(&proposal, settings.uses.len() + j)
                            })
                            .collect::<Result<Vec<_>, _>>()?;
                        let u = down
                            .direction_operators
                            .iter()
                            .map(|(u, _)| down.program.operators[*u].clone())
                            .collect::<Vec<_>>();
                        let function = parameter_response_program::compose(&clean, &responses, &u)?;
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
                            &clean,
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
                let (mut trainable, body_map) =
                    graft_parameters_except(&candidate, &native, &proposal, &fixed)?;
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
                    verify_body(&proposal.program, &candidate.program, &body_map)?;
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
                    &json!({"native_controls":settings.controls,"candidate_controls":mapping,"down_edit_family":settings.down_edit_family,"fixed_direction_operator_ids":direction_ids}),
                )?;
                let graft_seconds = graft_started.elapsed().as_secs_f64();
                let canonical_started = Instant::now();
                let (mut candidate, _, canonical_before_phases) =
                    canonical_using(&candidate, native_codec.as_ref())?;
                let canonical_before_seconds = canonical_started.elapsed().as_secs_f64();
                let lowered = intervention_program::compile(&candidate.program, &mapped)?;
                let training = episodes(
                    &lowered,
                    &candidate.program,
                    &mapped,
                    family,
                    &settings.cases,
                    &targets,
                    down.as_ref(),
                )?;
                let fit_started = Instant::now();
                let fitted = resident_causal_fit::fit(
                    &d,
                    &lowered.program,
                    &training,
                    &trainable,
                    settings.fit.clone(),
                )?;
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
                    verify_body(&proposal.program, &saved.program, &body_map)?;
                }
                if let Some(down) = &down {
                    verify_fixed_directions(&saved, down, &direction_ids)?;
                }
                if shared && down.is_none() {
                    let mut body_source = proposal.program.clone();
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

                let saved_lowered = intervention_program::compile(&saved.program, &mapped)?;
                let training = episodes(
                    &saved_lowered,
                    &saved.program,
                    &mapped,
                    family,
                    &settings.cases,
                    &targets,
                    down.as_ref(),
                )?;
                let measured = resident_causal_fit::measure(
                    &d,
                    &saved_lowered.program,
                    &training,
                    settings.fit.numeric_bytes,
                )?;
                save(
                    &root.join("TRAIN.json"),
                    &serde_json::to_value(&measured).map_err(|e| e.to_string())?,
                )?;
                let cost_started = Instant::now();
                let c32 = structural_cost(&saved, &mut costs)?.total();
                let structural_cost_seconds = cost_started.elapsed().as_secs_f64();
                Ok(
                    json!({"id":name,"expression_id":id,"arm":arm,"c32":c32,"training_kl":measured.objective,
                    "artifact_sha256":sha256(&root.join("program.artifact"))?,"control_map_sha256":sha256(&root.join("CONTROL_MAP.json"))?,"trainable":trainable,"sharing_analysis":sharing.analyses[id],"transfer":frozen_body.is_some(),"body_frozen":frozen_body.is_some() && shared,"body_graft_operator_map":body_map,"native_initialization":settings.native_initialization,"native_initialization_scope":"primitive native-width capacity control, not discovery","stage_seconds":{"graft":graft_seconds,"canonical_before_fit":canonical_before_seconds,"canonical_after_fit":canonical_after_seconds,"canonical_before_phases":canonical_before_phases,"canonical_after_phases":canonical_after_phases,"fit":fit_seconds,"structural_cost":structural_cost_seconds},"status":"training_measured"}),
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
            !rows.iter().any(|b| {
                let (Some(ac), Some(bc), Some(ae), Some(be)) = (
                    a["c32"].as_u64(),
                    b["c32"].as_u64(),
                    a["training_kl"].as_f64(),
                    b["training_kl"].as_f64(),
                ) else {
                    return false;
                };
                bc <= ac && be <= ae && (bc < ac || be < ae)
            })
        })
        .map(|v| v["id"].clone())
        .collect();
    save(
        &out.join("TRAINING_PARETO.json"),
        &json!({"ids":frontier,"scope":"training cost/KL candidates for later independent Local/Run assessment; no holdout consulted"}),
    )?;
    let mut frozen =
        freeze_evaluation_ids(&frontier, &settings.expression_ids, use_specs.len() > 1);
    add_native_capacity_controls(
        &mut frozen,
        &settings.expression_ids,
        use_specs.len() > 1,
        settings.native_initialization,
    );
    save(
        &out.join("FROZEN_EVALUATION_IDS.json"),
        &json!({"ids":frozen,"training_pareto_ids":frontier,"policy":"native always; both matched shared/untied controls of every training-Pareto expression, including failed counterparts; native-initialized capacity controls additionally mandatory, independent of cost dominance","native_capacity_controls_mandatory":settings.native_initialization,"frozen_before_heldout_export_access":true}),
    )?;
    let mut heldout_rows = Vec::new();
    let mut heldout_provenance = Value::Null;
    if let Some(evaluation) = &settings.evaluation {
        let heldout_started = Instant::now();
        let prepared = (|| -> Result<(FamilyInputs, Vec<Array2<f64>>), String> {
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
            let (labels, planned) = family_teacher_targets(
                &d,
                &original_native,
                &original_layers,
                &settings.controls,
                &imported_heldout.contract.family,
                eval_cases,
                down.as_ref(),
                settings.fit.numeric_bytes,
                settings.teacher_numeric_bytes,
            )?;
            heldout_provenance = json!({"native":imported_heldout.record,"export_sha256":evaluation.export_sha256,"rows_per_episode":imported_heldout.contract.family.rows,"teacher_planned_numeric_bytes":planned,"model_identity":"same original graph/interfaces and numerical weight bits; wire-omitted provenance ignored","sequence_overlap":"all heldout fixed-context token sequences checked absent from training","scope":"previously project-seen, fit-disjoint panel; not untouched confirmation. No updates/reselection, operational F64 KL only"});
            save(&out.join("HELDOUT_PROVENANCE.json"), &heldout_provenance)?;
            Ok((imported_heldout.contract.family, labels))
        })();
        stage_seconds.insert(
            "heldout_load_and_teachers".into(),
            heldout_started.elapsed().as_secs_f64(),
        );
        let heldout_eval_started = Instant::now();
        for id in &frozen {
            let attempt = (|| -> Result<Value, String> {
                let (eval_family, labels) = prepared.as_ref().map_err(Clone::clone)?;
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
                let artifact = if let Some(cache) = &native_codec {
                    cache.decode_saved(&bytes, &native.declarations)?
                } else {
                    Artifact::from_bytes(&bytes, &native.declarations)?
                };
                if artifact.to_bytes()? != bytes {
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
                let lowered = intervention_program::compile(&artifact.program, &mapped)?;
                let eval_episodes = episodes(
                    &lowered,
                    &artifact.program,
                    &mapped,
                    eval_family,
                    evaluation.cases.as_deref().unwrap_or(&settings.cases),
                    labels,
                    down.as_ref(),
                )?;
                let measured = resident_causal_fit::measure(
                    &d,
                    &lowered.program,
                    &eval_episodes,
                    settings.fit.numeric_bytes,
                )?;
                Ok(
                    json!({"id":id,"status":"heldout_measured","c32":c32,"measurement":measured,"artifact_sha256":training["artifact_sha256"],"control_map_sha256":training["control_map_sha256"],"scope":"frozen candidate measurement only; no Local or acceptance certificate"}),
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
    run()
}

#[cfg(test)]
mod tests {
    use super::*;
    use gam_mpd::{
        composed_rule_search::{Expr, Unary},
        operator_program::{Node, SlotValues},
    };
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
        assert!(
            trace.values[lowered.program.output]
                .iter()
                .all(|v| v.is_finite())
        );
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
        };
        let edited = Case {
            label: "negative".into(),
            group: "edits".into(),
            gains: vec![],
            down_amplitudes: vec![-0.5],
        };
        assert!(validate_cases(&[clean.clone(), edited.clone()], 0, 1, true).is_ok());
        assert!(validate_cases(&[edited], 0, 1, true).is_err());
        let mut bad = clean.clone();
        bad.down_amplitudes = vec![f64::NAN];
        assert!(validate_cases(&[bad], 0, 1, false).is_err());
        assert!(validate_cases(&[clean], 0, 2, true).is_err());
    }
    #[test]
    fn removed_down_family_runs_literal_teachers_shared_response_fit_and_saved_replay() {
        use gam_mpd::operator_program::{
            Declarations, Interface, Law, Operator, Slot, SlotValues, exact_precision,
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
                dense(array![[1., -0.5], [0.25, 0.75]]),
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
            },
            Case {
                label: "positive".into(),
                group: "edits".into(),
                gains: vec![],
                down_amplitudes: vec![0.5],
            },
            Case {
                label: "negative".into(),
                group: "edits".into(),
                gains: vec![],
                down_amplitudes: vec![-0.5],
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
        for (case, target) in cases.iter().zip(&targets) {
            let literal = down.literal_native(&case.down_amplitudes).expect("literal");
            let expected = literal
                .execute(&base, false)
                .expect("original native state")
                .values[literal.output]
                .clone();
            assert!(
                expected
                    .iter()
                    .zip(target)
                    .all(|(a, b)| (a - b).abs() < 1e-12)
            );
        }
        let expression = Expr::Unary(
            Unary::Relu,
            Box::new(Expr::Affine(Box::new(Expr::Argument(0)))),
        );
        for shared in [true, false] {
            let compiler = if shared {
                composed_rule_search::compile
            } else {
                composed_rule_search::compile_untied
            };
            let proposal = compiler(
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
            .expect("joint clean and response pool");
            let function = parameter_response_program::compose(
                &composed_rule_search::function(&proposal, 0).expect("clean"),
                &[composed_rule_search::function(&proposal, 1).expect("response")],
                &[down.program.operators[down.direction_operators[0].0].clone()],
            )
            .expect("compose");
            let mut candidate = Artifact::native(&down.program)
                .expect("native")
                .replace_function_inputs(
                    "removed down",
                    &function,
                    &[down.native_read, down.control_nodes[0]],
                    down.native_write,
                )
                .expect("whole block graft");
            assert!(candidate.place(down.node_mapping[1]).is_none());
            assert!(candidate.place(down.node_mapping[2]).is_none());
            let fixed = fixed_response_writers(&candidate, &down).expect("fixed writer roles");
            let (trainable, _) =
                graft_parameters_except(&candidate, &down.program, &proposal, &fixed)
                    .expect("real joint parameters");
            assert_eq!(trainable.len(), proposal.trainable.len());
            assert!(trainable.iter().all(|id| !fixed.contains(id)));
            let ids = down
                .retain_directions_excluding(&mut candidate, &trainable.iter().copied().collect())
                .expect("paid immutable directions");
            let (candidate, _) = canonical(&candidate).expect("F32 initial graph");
            verify_fixed_directions(&candidate, &down, &ids).expect("fixed literals");
            let lowered =
                intervention_program::compile(&candidate.program, &[]).expect("ordinary controls");
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
            let fit = resident_causal_fit::fit(
                &device,
                &lowered.program,
                &panels,
                &trainable,
                FitSettings {
                    iterations: 2,
                    learning_rate: 0.01,
                    beta1: 0.9,
                    beta2: 0.999,
                    epsilon: 1e-8,
                    numeric_bytes: 1_000_000,
                },
            )
            .expect("joint causal fit");
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
            let replay = intervention_program::compile(&saved.program, &[]).expect("replay graph");
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
        }
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
}
