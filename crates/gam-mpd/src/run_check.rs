//! `D_run` of a language model's artifact under `counterfactual`'s declared episodes (#2951).
//!
//! # The native program
//!
//! [`split_sites`] rewrites `import::import_language_model`'s program so that every decomposed
//! site's output is a node: the `o` output (`Σ_h W_O,h read_h`) and the `down_proj` output each
//! get their own node, which the residual node then adds to the stream, `x ← x + out`, as
//! `counterfactual`'s decoder adds it. A replaced attention or MLP block then writes its own
//! output in the native interface, not the stream it is added to. [`layer_nodes`] names every
//! layer's site nodes.
//!
//! # The episodes
//!
//! The native side is `counterfactual`'s own: its binary64 [`Decoder`] runs each episode's actions
//! through its [`Program`] with the native donor states, exactly as `counterfactual::evaluate` runs
//! the native model. The explanation's side is the decoded [`Artifact`] executed as an operator
//! program, each action applied at the places it holds:
//!
//! * a head's or a neuron's input scale ([`InputChange::Scale`] at an `o` or `down_proj` site) scales
//!   the head's attention read or the neuron's activation;
//! * input mixes change the site's own incoming state before its replacement rule executes;
//!   output mixes change the result afterward. For the native model, every
//!   decomposed map of the decoder is linear (no bias; refused otherwise), so mixing its input
//!   mixes its output with the same weight;
//! * a weight edit at a site ([`OutputChange::Add`]) adds `L (Rᵀ x)` to the site's output from its
//!   input `x`.
//!
//! The donor state is the explanation's own: the artifact run clean on the donor passage. An
//! action whose places the artifact does not hold (a neuron of an MLP its rules replaced) is not
//! applied: the explanation runs clean there and predicts no effect, which its score charges.
//! Output mixes act on each held head independently; `EpisodeScore::unheld` counts missing
//! native edited places, including the missing heads of a partially held output site. Input
//! scales require only their affected head or activation. A composite input mix with partially
//! held incoming places is refused. An absent intermediate output can use an enclosing
//! declared block's write only when the native graph proves every path from the edited
//! block input to that write passes through the site's outputs. This preserves an input
//! intervention through composition without claiming omitted internal neurons are held.
//! Other absent output boundaries are refused. A weight edit with a held
//! output but no incoming place is likewise refused; wholly absent interventions run clean.
//!
//! The artifact's final norm and unembedding must be the native ones (places it holds, the same
//! operators), so its next-token distributions are the decoder's `log_probs` of its own final
//! residual. Each episode is scored as `counterfactual::score` scores it: mean `KL(native ‖ P)` per
//! token over the rows from the first changed row on, with the native effect and top-1 agreement
//! over the same rows; the rounding of each row's KL is [`kl_logits`]'s.

use super::acceptance::{EpisodeScore, RunCheck, kl_logits};
use super::artifact::Artifact;
use super::counterfactual::{Action, Decoder, Donor, Forward, InputChange, KINDS, Maps, OutputChange, Program, Rows, Spec, top1_rows};
use super::operator_program::{FamilyInputs, Node, Operator, OperatorBody, OperatorProgram, SequenceLayout, SlotValues, remap_node};
use gam_linalg::faer_ndarray::{fast_ab, fast_abt};
use ndarray::{Array1, Array2, Axis, s};
use std::collections::BTreeMap;
use std::ops::Range;
use std::sync::Arc;

/// One layer's site nodes in the split native program ([`split_sites`]).
#[derive(Clone, Debug, Default, PartialEq)]
pub struct LayerNodes {
    /// The residual stream entering the layer, and its normed copy (the `q`, `k`, `v` sites' input).
    pub stream: usize,
    pub normed_stream: usize,
    /// Per head its query node, per key-value head its key and value nodes.
    pub queries: Vec<usize>,
    pub keys: Vec<usize>,
    pub values: Vec<usize>,
    /// Per head its attention read (the `o` site's input, the head's columns).
    pub reads: Vec<usize>,
    /// The `o` site's output, and the residual after attention (the stream plus it).
    pub attention: usize,
    pub attended: usize,
    /// The `c_fc` site's input (the normed stream), output and the activations (`down_proj`'s input).
    pub normed: usize,
    pub pre: usize,
    pub active: usize,
    /// The `down_proj` site's output, and the residual after the MLP (`attended` plus it).
    pub mlp: usize,
    pub residual: usize,
}

/// Parse `blocks.{l}.{part}`.
fn site_part(name: &str) -> Option<(usize, &str)> {
    let rest = name.strip_prefix("blocks.")?;
    let (layer, part) = rest.split_once('.')?;
    Some((layer.parse().ok()?, part))
}

fn writes_site(program: &OperatorProgram, op: usize) -> bool {
    site_part(&program.operators[op].name).is_some_and(|(_, part)| part == "down_proj" || part.strip_prefix('o').is_some_and(|h| h.parse::<usize>().is_ok()))
}

/// The language model's program with every `o` and `down_proj` output its own node (module note):
/// each affine node adding a site's terms to the stream becomes the site's terms alone, followed by
/// the stream plus that node through the identity.
pub fn split_sites(program: &OperatorProgram) -> Result<OperatorProgram, String> {
    let mut out = program.clone();
    let mut nodes: Vec<Node> = Vec::with_capacity(program.nodes.len() + 16);
    let mut map: Vec<usize> = Vec::with_capacity(program.nodes.len());
    let identity: Vec<usize> = (0..program.operators.len()).collect();
    let (bases, rules): (Vec<usize>, Vec<usize>) = ((0..program.bases.len()).collect(), (0..program.rules.len()).collect());
    for (index, node) in program.nodes.iter().enumerate() {
        let mut node = node.clone();
        remap_node(&mut node, &map, &identity, &bases, &rules);
        if let Node::Affine { terms, bias } = &node
            && terms.iter().any(|(_, op)| writes_site(program, *op))
        {
            let (site, rest): (Vec<(usize, usize)>, Vec<(usize, usize)>) = terms.iter().partition(|(_, op)| writes_site(program, *op));
            let [(stream, skip)] = rest[..] else {
                return Err(format!("node {index}: a site write with {} other terms", rest.len()));
            };
            if !matches!(program.operators[skip].body, OperatorBody::Identity) {
                return Err(format!("node {index}: the stream enters through {}, not the identity", program.operators[skip].name));
            }
            nodes.push(Node::Affine { terms: site, bias: *bias });
            let written = nodes.len() - 1;
            nodes.push(Node::Affine { terms: vec![(stream, skip), (written, skip)], bias: None });
        } else {
            nodes.push(node);
        }
        map.push(nodes.len() - 1);
    }
    out.output = map[program.output];
    out.nodes = nodes;
    out.interfaces().map_err(|e| e.to_string())?;
    Ok(out)
}

/// The split native program's site nodes, found by its operators' names (`blocks.{l}.q{h}`, …).
pub fn layer_nodes(native: &OperatorProgram, layers: usize) -> Result<Vec<LayerNodes>, String> {
    let name = |op: usize| native.operators[op].name.as_str();
    let mut out = vec![LayerNodes::default(); layers];
    let mut found = vec![[false; 3]; layers];
    for (index, node) in native.nodes.iter().enumerate() {
        if let Node::Affine { terms, bias } = node {
            for (argument, op) in terms {
                let Some((layer, part)) = site_part(name(*op)) else {
                    continue;
                };
                if layer >= layers {
                    return Err(format!("operator {} beyond {layers} layers", name(*op)));
                }
                if bias.is_some() {
                    return Err(format!("{} has a bias: its input mix is not its output mix", name(*op)));
                }
                let nodes = &mut out[layer];
                let indexed = |prefix: &str| part.strip_prefix(prefix).and_then(|h| h.parse::<usize>().ok());
                if let Some(h) = indexed("q") {
                    put(&mut nodes.queries, h, index);
                    nodes.normed_stream = *argument;
                } else if let Some(g) = indexed("k") {
                    put(&mut nodes.keys, g, index);
                } else if let Some(g) = indexed("v") {
                    put(&mut nodes.values, g, index);
                } else if indexed("o").is_some() {
                    nodes.attention = index;
                    found[layer][0] = true;
                } else if part == "c_fc" {
                    nodes.normed = *argument;
                    nodes.pre = index;
                    found[layer][1] = true;
                } else if part == "down_proj" {
                    nodes.active = *argument;
                    nodes.mlp = index;
                    found[layer][2] = true;
                }
            }
            // The residual nodes: the stream plus a site's output through the identity.
            if let [(stream, _), (written, _)] = terms[..] {
                for nodes in out.iter_mut() {
                    if written == nodes.attention && nodes.attention != 0 {
                        nodes.stream = stream;
                        nodes.attended = index;
                    } else if written == nodes.mlp && nodes.mlp != 0 {
                        nodes.residual = index;
                    }
                }
            }
        } else if let Node::Attend { query, .. } = node {
            for nodes in out.iter_mut() {
                if let Some(h) = nodes.queries.iter().position(|q| q == query) {
                    put(&mut nodes.reads, h, index);
                }
            }
        }
    }
    for (layer, nodes) in out.iter().enumerate() {
        if !found[layer].iter().all(|f| *f)
            || nodes.reads.len() != nodes.queries.len()
            || nodes.queries.is_empty()
            || nodes.attended == 0
            || nodes.residual == 0
        {
            return Err(format!("layer {layer}: the native program is not a split sequential decoder of named sites (split_sites)"));
        }
    }
    Ok(out)
}

fn put(list: &mut Vec<usize>, at: usize, value: usize) {
    if list.len() <= at {
        list.resize(at + 1, usize::MAX);
    }
    list[at] = value;
}

/// One edit of a node of `P`.
#[derive(Clone, Debug)]
enum NodeEdit {
    Scale {
        rows: Rows,
        columns: Range<usize>,
        scale: f64,
    },
    /// `value[row] ← (1 − α) value[row] + α donor` on `columns`.
    Mix {
        row: usize,
        columns: Range<usize>,
        alpha: f64,
        donor: DonorKey,
    },
    /// `value += (input R) Lᵀ`.
    AddMap {
        input: usize,
        left: Arc<Array2<f64>>,
        right: Arc<Array2<f64>>,
    },
}

fn site_inputs(nodes: &LayerNodes, kind: usize) -> Vec<usize> {
    match kind {
        0..=2 => vec![nodes.normed_stream],
        3 => nodes.reads.clone(),
        4 => vec![nodes.normed],
        _ => vec![nodes.active],
    }
}
fn site_outputs(nodes: &LayerNodes, kind: usize) -> Vec<usize> {
    match kind {
        0 => nodes.queries.clone(),
        1 => nodes.keys.clone(),
        2 => nodes.values.clone(),
        3 => vec![nodes.attention],
        4 => vec![nodes.pre],
        _ => vec![nodes.mlp],
    }
}

/// A scale names only the incoming coordinates it changes. Mix names the whole
/// site input; a partial composite input cannot be reconstructed by this adapter.
fn add_input_spec(
    specs: &mut BTreeMap<usize, (Vec<usize>, Vec<usize>)>,
    artifact: &Artifact,
    nodes: &LayerNodes,
    action: &Action,
    head_dim: usize,
) -> Result<(), String> {
    let Action::Input { change, .. } = action else {
        return Ok(());
    };
    let site = action.site();
    let kind = site % KINDS.len();
    let inputs = match change {
        InputChange::Mix { .. } => site_inputs(nodes, kind),
        InputChange::Scale { cols: (a, b), .. } => {
            let native = match kind {
                3 => {
                    if head_dim == 0 || b <= a || *b > (a / head_dim + 1) * head_dim {
                        return Err(format!("a head scale over columns {a}..{b} spans heads or is empty"));
                    }
                    *nodes.reads.get(a / head_dim).ok_or("head scale beyond the site's heads")?
                }
                5 => nodes.active,
                _ => return Err(format!("an input scale at a {} site", KINDS[kind])),
            };
            // A missing affected head is a genuine no-op; unrelated absent heads
            // do not impose a composite mapping requirement on this scale.
            if artifact.place(native).is_none() {
                return Ok(());
            }
            vec![native]
        }
    };
    let spec = specs.entry(site).or_insert_with(|| (Vec::new(), site_outputs(nodes, kind)));
    for input in inputs {
        if !spec.0.contains(&input) {
            spec.0.push(input);
        }
    }
    Ok(())
}

struct MappedEdits {
    held: Vec<(usize, NodeEdit)>,
    /// Omitted upstream controls must act before ordinary edits at their boundary.
    pre_write: Vec<(usize, NodeEdit)>,
    /// Native edited places this action leaves clean in P.
    unheld: usize,
}

/// Preserve source-local order, then native write-local order. These phases
/// follow the native graph, not the order unrelated actions appear in a list.
fn prepend_controls(edits: &mut BTreeMap<usize, Vec<NodeEdit>>, controls: BTreeMap<usize, Vec<NodeEdit>>) {
    for (node, mut before) in controls {
        if let Some(mut after) = edits.remove(&node) { before.append(&mut after); }
        edits.insert(node, before);
    }
}

/// Execution-only edge forks: edits of one site's input must not spill to siblings.
/// `specs` maps site to (native inputs, native outputs).
fn lift_input_boundaries(
    native: &OperatorProgram,
    artifact: &Artifact,
    specs: &mut BTreeMap<usize, (Vec<usize>, Vec<usize>)>,
) -> Result<(), String> {
    // An input edge into a composed block can be moved to its exposed input
    // when there is no native path bypassing the edited site. This is
    // a graph proof, not a fitted sensitivity or a new no-effect convention.
    fn reaches(program: &OperatorProgram, write: usize, target: usize, stops: &[usize]) -> bool {
        let mut seen = vec![false; program.nodes.len()];
        let mut stack = vec![write];
        while let Some(node) = stack.pop() {
            if node == target { return true; }
            if seen[node] || stops.contains(&node) { continue; }
            seen[node] = true;
            stack.extend(program.nodes[node].arguments());
        }
        false
    }
    for (&site, (inputs, outputs)) in specs.iter_mut() {
        if inputs.iter().chain(outputs.iter()).any(|&n| n >= native.nodes.len()) {
            return Err(format!("site {site}: native input/output outside graph"));
        }
        // Existing complete boundaries and missing inputs keep their existing
        // handling. Partially held output families must not be silently merged.
        if !inputs.iter().all(|n| artifact.place(*n).is_some()) || outputs.iter().any(|n| artifact.place(*n).is_some()) {
            continue;
        }
        let eligible: std::collections::BTreeSet<_> = artifact.blocks.iter().filter(|block| {
            block.native_write < native.nodes.len()
                && artifact.place(block.native_write) == Some(block.write)
                && inputs.iter().all(|input| block.native_reads.iter().zip(&block.reads).any(|(n,p)| n == input && artifact.place(*n) == Some(*p)))
                && outputs.iter().all(|&output| reaches(native, block.native_write, output, &block.native_reads))
                && inputs.iter().all(|&input| !reaches(native, block.native_write, input, outputs))
                // The fork stops at held causal places. An exposed interior
                // boundary would need its own ordered intervention treatment;
                // do not lift across it as if it were a private expression.
                && !(0..block.native_write).any(|node| !block.native_reads.contains(&node)
                    && artifact.place(node).is_some()
                    && reaches(native, block.native_write, node, &block.native_reads))
        }).map(|block| block.native_write).collect();
        match eligible.len() {
            0 => {}, // fork_inputs supplies the explicit unsupported-boundary error.
            1 => *outputs = vec![*eligible.first().ok_or("missing unique block write")?],
            _ => return Err(format!("site {site}: ambiguous enclosing intervention boundary")),
        }
    }
    Ok(())
}

fn fork_inputs(
    artifact: &Artifact,
    specs: &BTreeMap<usize, (Vec<usize>, Vec<usize>)>,
) -> Result<(Artifact, BTreeMap<(usize, usize), usize>, Vec<usize>), String> {
    let p = &artifact.program;
    let mut out = artifact.clone();
    out.program.nodes.clear();
    let mut map = vec![usize::MAX; p.nodes.len()];
    let mut originals = vec![usize::MAX; p.nodes.len()];
    let operators: Vec<_> = (0..p.operators.len()).collect();
    let bases: Vec<_> = (0..p.bases.len()).collect();
    let rules: Vec<_> = (0..p.rules.len()).collect();
    let mut forks = BTreeMap::new();
    let mut targets: BTreeMap<usize, Vec<(usize, Vec<(usize, usize)>)>> = BTreeMap::new();
    for (&site, (inputs, outputs)) in specs {
        let held_inputs: Vec<_> = inputs.iter().filter_map(|n| artifact.place(*n).map(|own| (*n, own))).collect();
        if held_inputs.is_empty() {
            continue;
        }
        if held_inputs.len() != inputs.len() {
            return Err(format!("site {site}: partially held composite input cannot be forked faithfully"));
        }
        let held_outputs: Vec<_> = outputs.iter().filter_map(|n| artifact.place(*n)).collect();
        if held_outputs.len() != outputs.len() || held_outputs.is_empty() {
            return Err(format!("site {site}: held input has an absent output boundary; cannot fork the site faithfully"));
        }
        let inputs = held_inputs;
        for write in held_outputs {
            targets.entry(write).or_default().push((site, inputs.clone()));
        }
    }
    for (index, node) in p.nodes.iter().enumerate() {
        let mut node = node.clone();
        remap_node(&mut node, &map, &operators, &bases, &rules);
        out.program.nodes.push(node);
        map[index] = out.program.nodes.len() - 1;
        originals[index] = map[index];
        if let Some(sites) = targets.get(&index) {
            if sites.len() != 1 {
                return Err("multiple named sites share a held write; cannot fork their input edges independently".into());
            }
            let (site, inputs) = &sites[0];
            let mut bound = map.clone();
            for &(native, own) in inputs {
                let fork = if let Some(&fork) = forks.get(&(*site, native)) {
                    fork
                } else {
                    let interface = p.node_interface(own).map_err(|e| e.to_string())?;
                    let op = out.program.operators.len();
                    out.program.operators.push(Arc::new(Operator::identity("episode input", interface)));
                    out.program.nodes.push(Node::Affine { terms: vec![(map[own], op)], bias: None });
                    let fork = out.program.nodes.len() - 1;
                    forks.insert((*site, native), fork);
                    fork
                };
                bound[own] = fork;
            }
            // Other held places are intervenable causal boundaries, not private
            // subexpressions of this site. Recomputing them would erase their edits.
            let boundaries: std::collections::BTreeSet<_> =
                artifact.places.iter().map(|(_, own)| *own).filter(|own| *own != index && !inputs.iter().any(|(_, input)| input == own)).collect();
            // Only nodes in this write's cone can need cloning.
            let mut cone = vec![false; p.nodes.len()];
            cone[index] = true;
            for i in (0..=index).rev() {
                if !cone[i] || boundaries.contains(&i) || inputs.iter().any(|(_, own)| *own == i) {
                    continue;
                }
                for argument in p.nodes[i].arguments() {
                    cone[argument] = true;
                }
            }
            let mut changed = vec![false; p.nodes.len()];
            for &(_, own) in inputs {
                changed[own] = true;
            }
            for i in 0..=index {
                if !cone[i] || boundaries.contains(&i) || inputs.iter().any(|(_, own)| *own == i) {
                    continue;
                }
                if p.nodes[i].arguments().iter().any(|a| changed[*a]) {
                    let mut copy = p.nodes[i].clone();
                    remap_node(&mut copy, &bound, &operators, &bases, &rules);
                    out.program.nodes.push(copy);
                    bound[i] = out.program.nodes.len() - 1;
                    changed[i] = true;
                    // Exceptions are part of the executed replacement, also on the forked cone.
                    for exception in artifact.exceptions.iter().filter(|e| e.node == i) {
                        let mut exception = exception.clone();
                        exception.node = bound[i];
                        out.exceptions.push(exception);
                    }
                }
            }
            map[index] = bound[index];
        }
    }
    out.program.output = map[p.output];
    for (_, own) in &mut out.places {
        *own = map[*own];
    }
    for control in &mut out.controls {
        control.write = map[control.write];
    }
    for block in &mut out.blocks {
        for read in &mut block.reads {
            *read = map[*read];
        }
        block.write = map[block.write];
    }
    // Original exceptions refer to original remapped nodes; cloned copies already use new indices.
    for exception in out.exceptions.iter_mut().take(artifact.exceptions.len()) {
        exception.node = originals[exception.node];
    }
    out.program.interfaces().map_err(|e| e.to_string())?;
    Ok((out, forks, map))
}

fn site_edits(
    layers: &[LayerNodes],
    artifact: &Artifact,
    clean: &Artifact,
    forks: &BTreeMap<(usize, usize), usize>,
    action: &Action,
    head_dim: usize,
) -> Result<MappedEdits, String> {
    let site = action.site();
    let (layer, kind) = (site / KINDS.len(), site % KINDS.len());
    let nodes = layers.get(layer).ok_or_else(|| format!("site {site} beyond the decoder"))?;
    let place = |native: usize| artifact.place(native);
    let input_place = |native: usize| forks.get(&(site, native)).copied();
    let mut mapped = MappedEdits { held: Vec::new(), pre_write: Vec::new(), unheld: 0 };
    match action {
        Action::Input { change: InputChange::Scale { rows, cols: (a, b), scale }, .. } => {
            let (native, columns) = match kind {
                3 => {
                    if head_dim == 0 || b <= a || *b > (a / head_dim + 1) * head_dim {
                        return Err(format!("a head scale over columns {a}..{b} spans heads or is empty"));
                    }
                    let head = a / head_dim;
                    (*nodes.reads.get(head).ok_or("head scale beyond the site's heads")?, a - head * head_dim..b - head * head_dim)
                }
                5 => (nodes.active, *a..*b),
                _ => return Err(format!("an input scale at a {} site", KINDS[kind])),
            };
            match input_place(native) {
                Some(node) => {
                    let width = artifact.program.node_interface(node).map_err(|e| e.to_string())?.width();
                    if columns.is_empty() || columns.end > width {
                        return Err("input scale beyond the held interface".into());
                    }
                    mapped.held.push((node, NodeEdit::Scale { rows: *rows, columns, scale: *scale }));
                }
                None if place(native).is_none() => {
                    if let Some(control) = artifact.controls.iter().find(|c| c.native_source == native)
                        && columns == (0..control.width)
                    {
                        let width = artifact.program.node_interface(control.write).map_err(|e|e.to_string())?.width();
                        mapped.pre_write.push((control.write, NodeEdit::Scale { rows: *rows, columns: 0..width, scale: *scale }));
                    } else {
                        mapped.unheld = 1;
                    }
                },
                None => {
                    return Err(format!("site {site}: held scale input has no executable fork"));
                }
            }
        }
        Action::Input { change: InputChange::Mix { row, alpha }, .. } => {
            let inputs = site_inputs(nodes, kind);
            let held = inputs.iter().filter(|n| place(**n).is_some()).count();
            if held == 0 {
                mapped.unheld = inputs.len();
            } else {
                if held != inputs.len() {
                    return Err(format!("site {site}: partially held composite input mix is unsupported"));
                }
                for native in inputs {
                    let node = input_place(native).ok_or_else(|| format!("site {site}: held mix input has no executable fork"))?;
                    let width = artifact.program.node_interface(node).map_err(|e| e.to_string())?.width();
                    let donor = clean.place(native).ok_or("held mix input has no clean donor place")?;
                    mapped.held.push((node, NodeEdit::Mix { row: *row, columns: 0..width, alpha: *alpha, donor: DonorKey { node: donor, row: *row } }));
                }
            }
        }
        Action::Output { change: OutputChange::Mix { row, alpha }, .. } => {
            for native in site_outputs(nodes, kind) {
                if let Some(node) = place(native) {
                    let width = artifact.program.node_interface(node).map_err(|e| e.to_string())?.width();
                    let donor = clean.place(native).ok_or("held output mix has no clean donor place")?;
                    mapped.held.push((node, NodeEdit::Mix { row: *row, columns: 0..width, alpha: *alpha, donor: DonorKey { node: donor, row: *row } }));
                } else {
                    mapped.unheld += 1;
                }
            }
        }
        Action::Output { change: OutputChange::Add { left, right }, .. } => {
            let (write, input) = match kind {
                4 => (nodes.pre, nodes.normed),
                5 => (nodes.mlp, nodes.active),
                _ => return Err(format!("a weight edit at a {} site", KINDS[kind])),
            };
            if let Some(node) = place(write) {
                let input = input_place(input).or_else(|| place(input)).ok_or_else(|| format!("site {site}: held weight-edit output has no incoming place"))?;
                if input >= node {
                    return Err("weight-edit input is not before its output".into());
                }
                mapped.held.push((node, NodeEdit::AddMap { input, left: left.clone(), right: right.clone() }));
            } else {
                mapped.unheld = 1;
            }
        }
    }
    Ok(mapped)
}

/// A donor state of `P`: a node's row.
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord)]
struct DonorKey {
    node: usize,
    row: usize,
}

fn apply_node_edits(
    node: usize,
    value: &mut Array2<f64>,
    earlier: &[Array2<f64>],
    edits: &BTreeMap<usize, Vec<NodeEdit>>,
    donor: &BTreeMap<DonorKey, Array1<f64>>,
) -> Result<(), String> {
    for edit in edits.get(&node).map_or(&[][..], Vec::as_slice) {
        match edit {
            NodeEdit::Scale { rows, columns, scale } => {
                for (r, mut row) in value.outer_iter_mut().enumerate() {
                    if matches!(rows, Rows::All) || *rows == Rows::One(r) {
                        row.slice_mut(s![columns.clone()]).mapv_inplace(|v| v * scale);
                    }
                }
            }
            NodeEdit::Mix { row, columns, alpha, donor: key } => {
                let d = donor.get(key).ok_or("a donor state not recorded")?;
                let mut target = value.slice_mut(s![*row, columns.clone()]);
                let mixed = &target.to_owned() * (1.0 - alpha) + &(d * *alpha);
                target.assign(&mixed);
            }
            NodeEdit::AddMap { input, left, right } => {
                *value += &fast_abt(&fast_ab(&earlier[*input], right.as_ref()), left.as_ref());
            }
        }
    }
    Ok(())
}

/// Apply the same action order as apply_node_edits without downloading live values.
fn apply_device_node_edits<'a>(
    device: &gam_gpu::tensor::Device,
    node: usize,
    rows: usize,
    root_value: impl Fn(usize) -> Result<&'a gam_gpu::tensor::Tensor, String>,
    edits: &BTreeMap<usize, Vec<NodeEdit>>,
    donor: &BTreeMap<DonorKey, Array1<f64>>,
) -> Result<Option<gam_gpu::tensor::Tensor>, String> {
    use gam_gpu::tensor::{Arithmetic, Op};
    let Some(actions) = edits.get(&node) else {
        return Ok(None);
    };
    let original = root_value(node)?;
    let width = original.cols();
    let mut value = device.copy(original).map_err(|e| e.to_string())?;
    for action in actions {
        match action {
            NodeEdit::Scale { rows: affected, columns, scale } => {
                if columns.end > width {
                    return Err("device scale outside interface".into());
                }
                let mut mask = Array2::ones((rows, width));
                for row in 0..rows {
                    if matches!(affected, Rows::All) || *affected == Rows::One(row) {
                        mask.slice_mut(s![row, columns.clone()]).fill(*scale);
                    }
                }
                let mask = device.upload(mask.view()).map_err(|e| e.to_string())?;
                let mut scaled = device.zeros(rows, width).map_err(|e| e.to_string())?;
                device.hadamard(&mut scaled, &value, &mask, false).map_err(|e| e.to_string())?;
                value = scaled;
            }
            NodeEdit::Mix { row, columns, alpha, donor: key } => {
                if *row >= rows || columns.end > width {
                    return Err("device mix outside interface".into());
                }
                let d = donor.get(key).ok_or("a donor state not recorded")?;
                if d.len() != columns.len() {
                    return Err("device donor width differs from mix interface".into());
                }
                let mut mask = Array2::ones((rows, width));
                mask.slice_mut(s![*row, columns.clone()]).fill(1.0 - alpha);
                let mask = device.upload(mask.view()).map_err(|e| e.to_string())?;
                let mut mixed = device.zeros(rows, width).map_err(|e| e.to_string())?;
                device.hadamard(&mut mixed, &value, &mask, false).map_err(|e| e.to_string())?;
                let mut donor_row = Array2::zeros((1, width));
                donor_row.slice_mut(s![0, columns.clone()]).assign(d);
                let donor_row = device.upload(donor_row.view()).map_err(|e| e.to_string())?;
                let donor_rows = device.broadcast_rows(&donor_row, rows).map_err(|e| e.to_string())?;
                let mut mask = Array2::zeros((rows, width));
                mask.slice_mut(s![*row, columns.clone()]).fill(1.0);
                let mask = device.upload(mask.view()).map_err(|e| e.to_string())?;
                let mut addition = device.zeros(rows, width).map_err(|e| e.to_string())?;
                device.hadamard(&mut addition, &donor_rows, &mask, false).map_err(|e| e.to_string())?;
                device.axpy(&mut mixed, *alpha, &addition).map_err(|e| e.to_string())?;
                value = mixed;
            }
            NodeEdit::AddMap { input, left, right } => {
                let incoming = root_value(*input)?;
                if left.nrows() != width || right.nrows() != incoming.cols() || left.ncols() != right.ncols() {
                    return Err("device weight-edit factors differ from held interfaces".into());
                }
                let left = device.upload(left.view()).map_err(|e| e.to_string())?;
                let right = device.upload(right.view()).map_err(|e| e.to_string())?;
                let mut hidden = device.zeros(rows, right.cols()).map_err(|e| e.to_string())?;
                device.gemm(&mut hidden, 1.0, incoming, Op::N, &right, Op::N, 0.0, Arithmetic::F64).map_err(|e| e.to_string())?;
                let mut addition = device.zeros(rows, width).map_err(|e| e.to_string())?;
                device.gemm(&mut addition, 1.0, &hidden, Op::N, &left, Op::T, 0.0, Arithmetic::F64).map_err(|e| e.to_string())?;
                device.axpy(&mut value, 1.0, &addition).map_err(|e| e.to_string())?;
            }
        }
    }
    Ok(Some(value))
}

struct TeacherEpisodes {
    residuals: Vec<Array2<f64>>,
    native_effects: Vec<f64>,
}

#[derive(Default)]
struct RunTimers {
    teacher: std::sync::atomic::AtomicU64,
    teacher_donor_forward: std::sync::atomic::AtomicU64,
    teacher_clean_forward: std::sync::atomic::AtomicU64,
    teacher_episode_forward: std::sync::atomic::AtomicU64,
    teacher_effect_readout: std::sync::atomic::AtomicU64,
    cpu_metric: std::sync::atomic::AtomicU64,
    gpu_head_logits_download: std::sync::atomic::AtomicU64,
    cpu_head_normalize: std::sync::atomic::AtomicU64,
    cpu_head: std::sync::atomic::AtomicU64,
    compile: std::sync::atomic::AtomicU64,
    native_compile: std::sync::atomic::AtomicU64,
    planning: std::sync::atomic::AtomicU64,
    donor: std::sync::atomic::AtomicU64,
    forward: std::sync::atomic::AtomicU64,
    readout: std::sync::atomic::AtomicU64,
}

struct RunTimer<'a>(&'a std::sync::atomic::AtomicU64, std::time::Instant);

impl RunTimer<'_> {
    fn start(counter: &std::sync::atomic::AtomicU64) -> RunTimer<'_> {
        RunTimer(counter, std::time::Instant::now())
    }
}

impl Drop for RunTimer<'_> {
    fn drop(&mut self) {
        self.0.fetch_add(self.1.elapsed().as_nanos().min(u64::MAX as u128) as u64, std::sync::atomic::Ordering::Relaxed);
    }
}

/// Cumulative elapsed seconds inside each execution stage, including saved-byte replays.
/// Parallel episode/donor durations are summed, so these are not additive wall times.
/// CUDA calls are timed through returned host states; this is not kernel-only timing.
#[derive(serde::Serialize)]
pub struct RunTiming {
    pub teacher_initialization: f64,
    pub teacher_donor_forward: f64,
    pub teacher_clean_forward: f64,
    pub teacher_episode_forward: f64,
    pub teacher_effect_readout: f64,
    /// Existing CPU KL/argmax reductions in teacher effects and candidate scoring.
    pub cpu_metric_reductions: f64,
    pub gpu_head_upload_norm_gemm_download: f64,
    pub cpu_head_log_normalization: f64,
    pub cpu_head_complete_log_probs: f64,
    pub candidate_compile: f64,
    pub native_source_compile: f64,
    pub intervention_planning: f64,
    pub donor_execution: f64,
    pub explained_execution: f64,
    pub output_readout_and_kl: f64,
}

#[derive(Clone, Copy, PartialEq)]
enum MetricMode { CpuOracle, GpuProposal, GpuChecked }

struct ScoredEpisode {
    score: Option<EpisodeScore>,
    checked: Option<crate::fixed_metric_device::Episode>,
}

/// Uncertified GPU metric proposal. It is not acceptance evidence and does not
/// implement RunCheck; independent CPU metric replay determines accepted verdicts.
#[derive(serde::Serialize)]
pub struct GpuMetricProposalEpisode {
    pub id: String,
    pub group: String,
    pub kl_estimate: f64,
    pub conditional_reduction_error_estimate: f64,
    pub native_effect_cpu_metric: f64,
    pub top1_agree: f64,
    pub unheld: usize,
}

struct NativeDeviceSource {
    interner: crate::decoded_intern::DecodedOperatorInterner,
    resident: crate::artifact_device::Resident,
}

/// `D_run` of a language model's artifact (module note).
pub struct LanguageRun<'a> {
    decoder: &'a Decoder,
    native: &'a OperatorProgram,
    spec: &'a Spec,
    passages: &'a [Vec<u32>],
    /// The native program's site nodes and final residual node.
    layers: Vec<LayerNodes>,
    /// Episodes run at once.
    pub parallel: usize,
    /// Output rows read through the unembedding at once.
    pub tile: usize,
    device: Option<gam_gpu::tensor::Device>,
    trace_bytes_limit: usize,
    native_device_source: Option<NativeDeviceSource>,
    native_readout: Option<crate::native_readout::Resident>,
    readout_lock: std::sync::Mutex<()>,
    teachers: std::sync::OnceLock<Result<TeacherEpisodes, String>>,
    timers: RunTimers,
}

impl<'a> LanguageRun<'a> {
    pub fn new(decoder: &'a Decoder, native: &'a OperatorProgram, spec: &'a Spec, passages: &'a [Vec<u32>], parallel: usize) -> Result<Self, String> {
        let layers = layer_nodes(native, decoder.layers())?;
        Ok(Self { decoder, native, spec, passages, layers, parallel, tile: 64, device: None, trace_bytes_limit: 0, native_device_source: None, native_readout: None, readout_lock: std::sync::Mutex::new(()), teachers: std::sync::OnceLock::new(), timers: RunTimers::default() })
    }

    /// Diagnostic timing only; no measurement changes fidelity or acceptance.
    pub fn timing(&self) -> RunTiming {
        let seconds = |counter: &std::sync::atomic::AtomicU64| counter.load(std::sync::atomic::Ordering::Relaxed) as f64 * 1e-9;
        RunTiming {
            teacher_initialization: seconds(&self.timers.teacher),
            teacher_donor_forward: seconds(&self.timers.teacher_donor_forward),
            teacher_clean_forward: seconds(&self.timers.teacher_clean_forward),
            teacher_episode_forward: seconds(&self.timers.teacher_episode_forward),
            teacher_effect_readout: seconds(&self.timers.teacher_effect_readout),
            cpu_metric_reductions: seconds(&self.timers.cpu_metric),
            gpu_head_upload_norm_gemm_download: seconds(&self.timers.gpu_head_logits_download),
            cpu_head_log_normalization: seconds(&self.timers.cpu_head_normalize),
            cpu_head_complete_log_probs: seconds(&self.timers.cpu_head),
            candidate_compile: seconds(&self.timers.compile), native_source_compile: seconds(&self.timers.native_compile),
            intervention_planning: seconds(&self.timers.planning), donor_execution: seconds(&self.timers.donor),
            explained_execution: seconds(&self.timers.forward), output_readout_and_kl: seconds(&self.timers.readout),
        }
    }

    /// Explicit hybrid backend: P's forwards and interventions run CUDA f64 without
    /// fallback; immutable native teachers, local fidelity and readout/KL remain CPU.
    /// The limit covers retained intermediate values only, not complete GPU allocation.
    pub fn with_cuda(mut self, device: gam_gpu::tensor::Device, trace_bytes_limit: usize) -> Result<Self, String> {
        if self.native_readout.is_some() { return Err("cannot replace CUDA forward backend after head installation".into()); }
        if !cfg!(target_os = "linux") || device.is_host() || !device.float64() || trace_bytes_limit == 0 {
            return Err("CUDA LanguageRun needs a float64 Linux accelerator and positive trace byte limit".into());
        }
        // A newly selected device cannot retain tensors compiled on the old one.
        self.native_device_source = None;
        self.device = Some(device);
        self.trace_bytes_limit = trace_bytes_limit;
        Ok(self)
    }

    /// Opt-in immutable source for exact native parameter reuse across candidates.
    /// The caller supplies the same decoded native artifact used to load its bank.
    /// Each assessed P is already decoded: only exact interface/body interning is
    /// repeated, never encoding, rounding or decoding. Candidate traces, exceptions
    /// and donor rows remain independent; this source is never forwarded or edited.
    pub fn with_cuda_native_source(mut self, source: &Artifact) -> Result<Self, String> {
        if self.native_device_source.is_some() {
            return Err("native CUDA source is already installed".into());
        }
        let device = self.device.as_ref().ok_or("native CUDA sharing needs the explicit CUDA backend")?;
        if !source.blocks.is_empty() || !source.exceptions.is_empty() || !source.derived.is_empty() {
            return Err("native CUDA source must have no replacements, exceptions or derivations".into());
        }
        source.validate_coverage(self.native)?;
        let (source, _) = self.truncated(source)?;
        let interner = crate::decoded_intern::DecodedOperatorInterner::new(&source)?;
        let timer = RunTimer::start(&self.timers.native_compile);
        let resident = crate::artifact_device::Resident::from_decoded(device, &source)?;
        drop(timer);
        self.native_device_source = Some(NativeDeviceSource { interner, resident });
        Ok(self)
    }

    /// Opt-in head-only CUDA f64. Teacher residual forwards remain CPU; native
    /// effects and candidate scores use CUDA logits plus unchanged CPU normalization/KL.
    /// Numerical parity is measured, not a bit-exact or full-network certificate.
    /// Must be installed before initializing immutable teacher scores.
    pub fn with_cuda_readout(mut self, budget: crate::native_readout::Budget) -> Result<Self, String> {
        if self.teachers.get().is_some() || self.native_readout.is_some() {
            return Err("CUDA readout must be selected once before teacher initialization".into());
        }
        let device = self.device.as_ref().ok_or("CUDA readout requires explicit CUDA forward backend")?;
        self.native_readout = Some(crate::native_readout::Resident::new(device.clone(), self.decoder, budget)?);
        Ok(self)
    }

    fn readout_rows(&self) -> usize {
        self.tile.max(1).min(self.native_readout.as_ref().map_or(usize::MAX, |h| h.tile_rows()))
    }

    fn log_probs(&self, residual: &Array2<f64>) -> Result<Array2<f64>, String> {
        match &self.native_readout {
            Some(head) => {
                let (probabilities,times)=head.log_probs_profiled(residual)?;
                self.timers.gpu_head_logits_download.fetch_add(times[0],std::sync::atomic::Ordering::Relaxed);
                self.timers.cpu_head_normalize.fetch_add(times[1],std::sync::atomic::Ordering::Relaxed);
                Ok(probabilities)
            }
            None => {
                let timer=RunTimer::start(&self.timers.cpu_head);
                let probabilities=self.decoder.log_probs(residual);
                drop(timer);
                Ok(probabilities)
            },
        }
    }

    pub fn cuda_native_sharing(&self) -> bool { self.native_device_source.is_some() }

    pub fn backend_name(&self) -> &'static str {
        if self.native_readout.is_some() { "hybrid: explained and native head CUDA f64; cached teacher residual CPU; normalization/KL CPU" } else if self.device.is_some() { "hybrid: explained CUDA f64; cached teacher and readout/KL CPU" } else { "CPU f64; cached native teachers" }
    }

    /// Read the existing immutable native teacher through this runner's unchanged readout.
    /// `rows` is an explicit scoring domain, not the episode's state-check interface_rows.
    /// This initializes the same teacher cache as episodes() and performs no new native forward.
    pub fn native_episode_log_probs(&self, episode: usize, rows: Range<usize>) -> Result<Array2<f64>, String> {
        let teachers = self.teacher_episodes()?;
        let residual = teachers.residuals.get(episode).ok_or("native episode index absent")?;
        if rows.start >= rows.end || rows.end > residual.nrows() { return Err("native readout rows outside episode".into()); }
        let guard = self.readout_lock.lock().map_err(|_| "native readout lock poisoned")?;
        let result = self.log_probs(&residual.slice(s![rows, ..]).to_owned());
        drop(guard);
        result
    }

    /// Fixed references belong to this runner's immutable borrowed dataset, never
    /// to a global cache or guessed data key. No candidate can modify their inputs.
    fn teacher_episodes(&self) -> Result<&TeacherEpisodes, String> {
        self.teachers
            .get_or_init(|| {
                let timer = RunTimer::start(&self.timers.teacher);
                let pool = rayon::ThreadPoolBuilder::new().num_threads(self.parallel.max(1)).build().map_err(|e| e.to_string())?;
                let result = pool.install(|| {
                    use rayon::prelude::*;
                    let decoder = self.decoder;
                    let all_rows: Vec<usize> = (0..self.spec.rows).collect();
                    let native_donors: BTreeMap<usize, Donor> = self
                        .spec
                        .episodes
                        .iter()
                        .filter_map(|e| e.donor)
                        .collect::<std::collections::BTreeSet<_>>()
                        .into_par_iter()
                        .map(|d| {
                            let keys: Vec<(usize, usize, bool)> = self
                                .spec
                                .episodes
                                .iter()
                                .filter(|e| e.donor == Some(d))
                                .flat_map(|e| e.actions.iter().filter_map(Action::donor_state))
                                .collect::<std::collections::BTreeSet<_>>()
                                .into_iter()
                                .collect();
                            let mut native = Program::new(Maps::Native(decoder), &[], None);
                            native.record = keys.iter().map(|k| (*k, None)).collect();
                            let forward_timer = RunTimer::start(&self.timers.teacher_donor_forward);
                            decoder.forward(&self.passages[d], &mut native, &[]);
                            drop(forward_timer);
                            let states = native
                                .record
                                .into_iter()
                                .map(|(k, v)| v.map(|v| (k, v)).ok_or("a native donor state not reached"))
                                .collect::<Result<_, _>>()?;
                            Ok((d, Donor { states }))
                        })
                        .collect::<Result<_, String>>()?;
                    let clean: BTreeMap<usize, Forward> = self
                        .spec
                        .episodes
                        .iter()
                        .map(|e| e.passage)
                        .collect::<std::collections::BTreeSet<_>>()
                        .into_par_iter()
                        .map(|p| {
                            let mut native = Program::new(Maps::Native(decoder), &[], None);
                            let forward_timer = RunTimer::start(&self.timers.teacher_clean_forward);
                            let forward = decoder.forward(&self.passages[p], &mut native, &all_rows);
                            drop(forward_timer);
                            (p, forward)
                        })
                        .collect();
                    let none = Donor::default();
                    let measured = self
                        .spec
                        .episodes
                        .par_iter()
                        .map(|episode| {
                            let native_donor = episode.donor.and_then(|d| native_donors.get(&d)).unwrap_or(&none);
                            let mut native = Program::new(Maps::Native(decoder), &episode.actions, Some(native_donor));
                            let forward_timer = RunTimer::start(&self.timers.teacher_episode_forward);
                            let reference = decoder.forward(&self.passages[episode.passage], &mut native, &episode.interface_rows);
                            drop(forward_timer);
                            let from = episode.actions.iter().map(Action::first_row).min().unwrap_or(0);
                            let rows = reference.residual.nrows();
                            if from > rows {
                                return Err("teacher intervention starts beyond passage".to_string());
                            }
                            let head_guard = self.native_readout.as_ref().map(|_| self.readout_lock.lock()).transpose().map_err(|_| "native readout lock poisoned")?;
                            let effect_timer = RunTimer::start(&self.timers.teacher_effect_readout);
                            let mut effect = 0.0;
                            let mut start = from;
                            while start < rows {
                                let end = (start + self.readout_rows()).min(rows);
                                let p = self.log_probs(&reference.residual.slice(s![start..end, ..]).to_owned())?;
                                let c = self.log_probs(&clean[&episode.passage].residual.slice(s![start..end, ..]).to_owned())?;
                                let metric_timer = RunTimer::start(&self.timers.cpu_metric);
                                for row in 0..end - start {
                                    effect += kl_logits(p.row(row), c.row(row)).0;
                                }
                                drop(metric_timer);
                                start = end;
                            }
                            drop(effect_timer);
                            drop(head_guard);
                            Ok((reference.residual, effect / (rows - from).max(1) as f64))
                        })
                        .collect::<Result<Vec<_>, String>>()?;
                    let (residuals, native_effects) = measured.into_iter().unzip();
                    Ok(TeacherEpisodes { residuals, native_effects })
                });
                drop(timer);
                result
            })
            .as_ref()
            .map_err(Clone::clone)
    }

    fn family(&self, passage: usize) -> FamilyInputs {
        let tokens: Vec<u32> = self.passages[passage].iter().take(self.spec.rows).copied().collect();
        let rows = tokens.len();
        FamilyInputs {
            rows,
            slots: vec![SlotValues::Tokens(tokens)],
            layout: Some(SequenceLayout { sequence: vec![0; rows], position: (0..rows as u32).collect() }),
        }
    }

    fn edits(
        &self,
        artifact: &Artifact,
        clean: &Artifact,
        forks: &BTreeMap<(usize, usize), usize>,
        action: &Action,
        head_dim: usize,
    ) -> Result<MappedEdits, String> {
        site_edits(&self.layers, artifact, clean, forks, action, head_dim)
    }

    /// `P`'s run on `passage` under `edits` (node → its edits, in action order), its donor states at
    /// `record`, and its final and per-layer residuals.
    fn run(
        &self,
        program: &Artifact,
        passage: usize,
        edits: &BTreeMap<usize, Vec<NodeEdit>>,
        donor: &BTreeMap<DonorKey, Array1<f64>>,
        record: &[DonorKey],
        residuals: &[usize],
        base: Option<&crate::artifact_device::Resident>,
    ) -> Result<(Vec<Array2<f64>>, BTreeMap<DonorKey, Array1<f64>>), String> {
        let family = self.family(passage);
        if let Some(device) = &self.device {
            // Share this candidate's unchanged operator tensors across graph forks.
            // Each episode still owns only its own live intermediate trace.
            let base = base.ok_or("CUDA run has no candidate base resident")?;
            let resident = crate::artifact_device::Resident::from_decoded_sharing(base, program)?;
            let estimate = resident.estimated_intermediate_bytes(family.rows)?;
            if estimate > self.trace_bytes_limit {
                return Err(format!("CUDA intermediate values need at least {estimate} bytes, exceeding declared trace limit {}", self.trace_bytes_limit));
            }
            let trace = resident.forward_edited_intermediates(&family, |node, trace| {
                apply_device_node_edits(device, node, family.rows, |root| resident.root_value(trace, root), edits, donor)
            })?;
            let mut recorded = BTreeMap::new();
            for key in record {
                if key.row >= family.rows {
                    return Err("CUDA donor row beyond passage".into());
                }
                let row = device.rows_of(resident.root_value(&trace, key.node)?, key.row, 1).map_err(|e| e.to_string())?;
                recorded.insert(*key, device.download(&row).map_err(|e| e.to_string())?.row(0).to_owned());
            }
            let states =
                residuals.iter().map(|root| device.download(resident.root_value(&trace, *root)?).map_err(|e| e.to_string())).collect::<Result<_, _>>()?;
            return Ok((states, recorded));
        }
        let mut recorded = BTreeMap::new();
        let trace = program.execute_edited(&family, |node, value, earlier| {
            apply_node_edits(node, value, earlier, edits, donor)?;
            for key in record.iter().filter(|k| k.node == node) {
                recorded.insert(*key, value.row(key.row).to_owned());
            }
            Ok(())
        })?;
        Ok((residuals.iter().map(|n| trace.values[*n].clone()).collect(), recorded))
    }

    /// `artifact` truncated at its final residual, with the native tail checked: the final norm and
    /// the unembedding after it must be places of `P` computing the native nodes with the same
    /// operators.
    fn truncated(&self, artifact: &Artifact) -> Result<(Artifact, Vec<usize>), String> {
        crate::native_control::validate(artifact, self.native)?;
        let last = self.layers.last().ok_or("no layer")?.residual;
        check_native_tail(self.native, artifact, last)?;
        let program = if self.device.is_some() { artifact.clone() } else { artifact.truncated(last)? };
        let residuals: Vec<usize> = self
            .layers
            .iter()
            .map(|l| program.place(l.residual).ok_or_else(|| format!("P does not hold the residual node {}", l.residual)))
            .collect::<Result<_, _>>()?;
        Ok((program, residuals))
    }
}

/// The decoder may supply the final head only when it is exactly the artifact's head.
/// Comparing node kinds and matrix values alone misses rewired inputs, normalization constants,
/// and artifact exceptions: all of those change the prediction that would actually execute.
fn check_native_tail(native: &OperatorProgram, artifact: &Artifact, last: usize) -> Result<(), String> {
    let places: Vec<usize> = (0..native.nodes.len()).map(|n| artifact.place(n).unwrap_or(usize::MAX)).collect();
    if places[native.output] != artifact.program.output {
        return Err("P's output is not the native head's output".to_string());
    }
    let bases: Vec<usize> = (0..native.bases.len()).collect();
    let rules: Vec<usize> = (0..native.rules.len()).collect();
    for index in last + 1..native.nodes.len() {
        let node = places[index];
        let mine = artifact.program.nodes.get(node).ok_or_else(|| format!("P does not hold the native tail node {index}"))?;
        let theirs = &native.nodes[index];
        if theirs.arguments().iter().any(|a| places[*a] == usize::MAX) {
            return Err(format!("P does not hold an input of native tail node {index}"));
        }
        if artifact.exceptions.iter().any(|e| e.node == node) {
            return Err(format!("P has an exception at native tail node {index}"));
        }
        let (my_ops, their_ops) = (mine.operators(), theirs.operators());
        if my_ops.len() != their_ops.len() {
            return Err(format!("P's node {node} is not the native tail node {index}"));
        }
        let mut operators = vec![usize::MAX; native.operators.len()];
        for (&a, b) in my_ops.iter().zip(their_ops) {
            let (ours, original) = (&artifact.program.operators[a], &native.operators[b]);
            let equal_values = match (&ours.body, &original.body) {
                (OperatorBody::Diagonal { values: a, .. }, OperatorBody::Diagonal { values: b, .. }) => a == b,
                (OperatorBody::Dense { values: a, present: pa, .. }, OperatorBody::Dense { values: b, present: pb, .. }) => a == b && pa == pb,
                (OperatorBody::Identity, OperatorBody::Identity) => true,
                _ => ours.matrix() == original.matrix(),
            };
            if ours.rows != original.rows || ours.cols != original.cols || !equal_values {
                return Err(format!("P's operator {a} differs at native tail node {index}"));
            }
            operators[b] = a;
        }
        let mut expected = theirs.clone();
        remap_node(&mut expected, &places, &operators, &bases, &rules);
        if expected != *mine {
            return Err(format!("P's node {node} is not the native tail node {index}"));
        }
    }
    Ok(())
}

impl LanguageRun<'_> {
    fn score_episodes(&self, artifact: &Artifact, mode: MetricMode) -> Result<Vec<EpisodeScore>, String> {
        self.score_episodes_detailed(artifact,mode,None)?.into_iter().map(|episode|episode.score.ok_or_else(|| "legacy metric score absent".to_string())).collect()
    }

    fn score_episodes_detailed(&self, artifact: &Artifact, mode: MetricMode, checked: Option<&crate::fixed_metric_device::Resident>) -> Result<Vec<ScoredEpisode>, String> {
        if mode==MetricMode::GpuChecked && checked.is_none() {return Err("checked GPU metric backend absent".into());}

        if mode==MetricMode::GpuProposal && self.native_readout.is_none() { return Err("GPU metric proposals require explicit native CUDA readout".into()); }
        use rayon::prelude::*;
        let head_dim = self.native.node_interface(self.layers[0].reads[0]).map_err(|e| e.to_string())?.width();
        let compile_timer = RunTimer::start(&self.timers.compile);
        let (mut program, residuals) = self.truncated(artifact)?;
        let base = self.device.as_ref().map(|device| match &self.native_device_source {
            Some(source) => {
                // Acceptance decoded this assessment afresh, so bank-load Arcs do
                // not survive. Exact re-interning restores only identical source
                // parameters and leaves P's graph/edits/exception order unchanged.
                source.interner.intern(&mut program);
                crate::artifact_device::Resident::from_decoded_sharing(&source.resident, &program)
            }
            None => crate::artifact_device::Resident::from_decoded(device, &program),
        }).transpose()?;
        drop(compile_timer);
        let teachers = self.teacher_episodes()?;
        // Each episode's edits of P, and the donor states they read.
        let planning_timer = RunTimer::start(&self.timers.planning);
        let mut plans = Vec::with_capacity(self.spec.episodes.len());
        let mut wanted: BTreeMap<usize, Vec<DonorKey>> = BTreeMap::new();
        for episode in &self.spec.episodes {
            let mut specs = BTreeMap::new();
            for action in &episode.actions {
                if matches!(action, Action::Input { .. }) {
                    let site = action.site();
                    let nodes = self.layers.get(site / KINDS.len()).ok_or("site beyond decoder")?;
                    add_input_spec(&mut specs, &program, nodes, action, head_dim)?;
                }
            }
            lift_input_boundaries(self.native, &program, &mut specs)?;
            let (episode_program, forks, map) = fork_inputs(&program, &specs)?;
            let episode_residuals: Vec<_> = residuals.iter().map(|n| map[*n]).collect();
            let mut edits: BTreeMap<usize, Vec<NodeEdit>> = BTreeMap::new();
            let mut controls: BTreeMap<usize, Vec<NodeEdit>> = BTreeMap::new();
            let mut unheld = 0;
            for action in &episode.actions {
                let mapped = self.edits(&episode_program, &program, &forks, action, head_dim)?;
                unheld += mapped.unheld;
                for (node, edit) in mapped.pre_write { controls.entry(node).or_default().push(edit); }
                for (node, edit) in mapped.held {
                    if let (NodeEdit::Mix { donor, .. }, Some(d)) = (&edit, episode.donor) {
                        let keys = wanted.entry(d).or_default();
                        if !keys.contains(donor) {
                            keys.push(*donor);
                        }
                    }
                    edits.entry(node).or_default().push(edit);
                }
            }
            prepend_controls(&mut edits, controls);
            plans.push((episode_program, episode_residuals, edits, unheld));
        }
        drop(planning_timer);
        let make_donor = |(d, keys): (&usize, &Vec<DonorKey>)| {
            let timer = RunTimer::start(&self.timers.donor);
            let result = self.run(&program, *d, &BTreeMap::new(), &BTreeMap::new(), keys, &residuals, base.as_ref())?;
            drop(timer);
            Ok((*d, result.1))
        };
        let own_donors: BTreeMap<usize, BTreeMap<DonorKey, Array1<f64>>> = if self.device.is_some() {
            wanted.iter().map(make_donor).collect::<Result<_, String>>()?
        } else {
            wanted.par_iter().map(make_donor).collect::<Result<_, String>>()?
        };
        let empty = BTreeMap::new();
        let score_one = |(index, (episode, (episode_program, episode_residuals, edits, unheld))): (
            usize,
            (&super::counterfactual::Episode, &(Artifact, Vec<usize>, BTreeMap<usize, Vec<NodeEdit>>, usize)),
        )|
         -> Result<ScoredEpisode, String> {
            let reference = &teachers.residuals[index];
            let own = episode.donor.and_then(|d| own_donors.get(&d)).unwrap_or(&empty);
            let forward_timer = RunTimer::start(&self.timers.forward);
            let (states, _) = self.run(episode_program, episode.passage, edits, own, &[], episode_residuals, base.as_ref())?;
            drop(forward_timer);
            let head_guard = self.native_readout.as_ref().map(|_| self.readout_lock.lock()).transpose().map_err(|_| "native readout lock poisoned")?;
            let readout_timer = RunTimer::start(&self.timers.readout);
            let explained = Forward {
                residual: states.last().ok_or("no residual")?.clone(),
                layers: states.iter().map(|x| x.select(Axis(0), &episode.interface_rows)).collect(),
            };
            let from = episode.actions.iter().map(Action::first_row).min().unwrap_or(0);
            let rows = reference.nrows();
            let (mut kl, mut error, mut agree) = (0.0, 0.0, 0.0);
            let mut fixed=checked.map(|resident|resident.stream());
            let mut start = from;
            while start < rows {
                let end = (start + self.readout_rows()).min(rows);
                let native_rows=reference.slice(s![start..end, ..]).to_owned();
                let explained_rows=explained.residual.slice(s![start..end, ..]).to_owned();
                match mode {
                    MetricMode::CpuOracle => {
                        let p = self.log_probs(&native_rows)?;
                        let q = self.log_probs(&explained_rows)?;
                        let metric_timer = RunTimer::start(&self.timers.cpu_metric);
                        for r in 0..end - start {
                            let (value, rounding) = kl_logits(p.row(r), q.row(r));
                            kl += value;
                            error += rounding;
                        }
                        agree += top1_rows(&p, &q).iter().filter(|a| **a).count() as f64;
                        drop(metric_timer);
                    }
                    MetricMode::GpuChecked => {
                        // Identical head calls and CPU normalization as CpuOracle;
                        // only the KL metric consumes buffered copies of these arrays.
                        let p=self.log_probs(&native_rows)?;
                        let q=self.log_probs(&explained_rows)?;
                        fixed.as_mut().ok_or("checked stream absent")?.append(&p,&q,start)?;
                    }
                    MetricMode::GpuProposal => {
                        let head=self.native_readout.as_ref().ok_or("GPU metric proposal head absent")?;
                        let proposed=head.proposal_metrics(&native_rows,&explained_rows)?;
                        for row in proposed {
                            kl+=row.kl_estimate;
                            error+=row.conditional_reduction_error_estimate;
                            agree+=f64::from(row.top1_equal);
                        }
                    }
                }
                start = end;
            }
            if mode==MetricMode::GpuChecked {
                let episode=fixed.ok_or("checked stream absent")?.finish(episode.id.clone(),episode.group.clone(),from,rows,teachers.native_effects[index],*unheld)?;
                drop(readout_timer);
                drop(head_guard);
                return Ok(ScoredEpisode {score:None,checked:Some(episode)});
            }
            drop(readout_timer);
            drop(head_guard);
            let n = (rows - from).max(1) as f64;
            Ok(ScoredEpisode {score:Some(EpisodeScore {
                id: episode.id.clone(),
                group: episode.group.clone(),
                kl: kl / n,
                numerical_error: (error / n).next_up(),
                native_effect: teachers.native_effects[index],
                top1_agree: agree / n,
                unheld: *unheld,
            }),checked:None})
        };
        let pairs: Vec<_> = self.spec.episodes.iter().zip(plans.iter()).enumerate().collect();
        let mut out = Vec::with_capacity(pairs.len());
        for chunk in pairs.chunks(if self.device.is_some() { 1 } else { self.parallel.max(1) }) {
            let scored: Vec<Result<ScoredEpisode, String>> = if self.device.is_some() {
                chunk.iter().map(|pair| score_one(*pair)).collect()
            } else {
                chunk.par_iter().map(|pair| score_one(*pair)).collect()
            };
            for s in scored {
                out.push(s?);
            }
        }
        Ok(out)
    }

    /// Opt-in fixed-array checked metric endpoint. The head and CPU normalization
    /// calls are unchanged. Exact interval group means remain separate from
    /// RunCheck's default CPU operational scores and acceptance decisions.
    pub fn checked_metric_episodes(&self, artifact: &Artifact, budget: crate::fixed_metric_device::Budget, compare_cpu: bool) -> Result<crate::fixed_metric_device::Measure, String> {
        let device=self.device.as_ref().ok_or("checked metrics require explicit CUDA backend")?;
        let resident=crate::fixed_metric_device::Resident::new(device.clone(),self.decoder.embedding().nrows(),self.readout_rows(),budget,compare_cpu)?;
        let episodes=self.score_episodes_detailed(artifact,MetricMode::GpuChecked,Some(&resident))?.into_iter().map(|row|row.checked.ok_or_else(||"checked episode absent".to_string())).collect::<Result<Vec<_>,String>>()?;
        Ok(crate::fixed_metric_device::Measure::of(episodes,&resident))
    }

    /// Fast proposal/ranking diagnostics only. The fixed CPU teacher residuals
    /// and CPU-metric native effects are reused. No tolerance or rejection logic
    /// is applied; every accepted candidate needs independent episodes() replay.
    pub fn gpu_metric_proposals(&self, artifact: &Artifact) -> Result<Vec<GpuMetricProposalEpisode>, String> {
        self.score_episodes(artifact,MetricMode::GpuProposal).map(|rows| rows.into_iter().map(|r| GpuMetricProposalEpisode {
            id:r.id,group:r.group,kl_estimate:r.kl,conditional_reduction_error_estimate:r.numerical_error,
            native_effect_cpu_metric:r.native_effect,top1_agree:r.top1_agree,unheld:r.unheld,
        }).collect())
    }
}

impl RunCheck for LanguageRun<'_> {
    fn episodes(&self, artifact: &Artifact) -> Result<Vec<EpisodeScore>, String> {
        self.score_episodes(artifact,MetricMode::CpuOracle)
    }
}

#[cfg(test)]
mod input_mix_tests {
    use super::*;
    use crate::operator_program::{Declarations, Interface, Provenance, Rule, Slot, exact_precision};
    use ndarray::array;

    fn fixture(nonlinear: bool) -> (Artifact, Vec<LayerNodes>, usize) {
        let interface = Interface::native(1).unwrap();
        let column = |value: f64| {
            Arc::new(
                Operator::dense(
                    "constant",
                    interface.clone(),
                    Interface::constant(),
                    array![[value]],
                    exact_precision([value]).unwrap(),
                    Provenance::native("constant"),
                )
                .unwrap(),
            )
        };
        let mut nodes = vec![Node::Raw { slot: 0 }];
        let operators = vec![Arc::new(Operator::identity("I", interface.clone())), column(-2.0), column(-6.0)];
        let mut rules = Vec::new();
        if nonlinear {
            rules.push(Rule {
                name: "nonlinear replacement".into(),
                inputs: vec![interface],
                output: 4,
                nodes: vec![
                    Node::Param { index: 0 },
                    Node::Affine { terms: vec![(0, 0)], bias: Some(1) },
                    Node::Affine { terms: vec![(0, 0)], bias: Some(2) },
                    Node::Hadamard { left: 1, right: 2 },
                    Node::Affine { terms: vec![(0, 0), (3, 0)], bias: None },
                ],
            });
            nodes.push(Node::Call { rule: 0, arguments: vec![0] });
        } else {
            nodes.push(Node::Affine { terms: vec![(0, 0)], bias: None });
        }
        let write = nodes.len() - 1;
        // A sibling using the same input is deliberately outside the edited site's cone.
        nodes.push(Node::Affine { terms: vec![(0, 0)], bias: None });
        let sibling = nodes.len() - 1;
        nodes.push(Node::Affine { terms: vec![(write, 0), (sibling, 0)], bias: None });
        let program = OperatorProgram {
            declarations: Declarations { domains: vec![], slots: vec![Slot::Raw { width: 1 }], parameters: 0 },
            bases: vec![],
            operators,
            rules,
            output: nodes.len() - 1,
            nodes,
        };
        let artifact = Artifact::native(&program).unwrap();
        let layer = LayerNodes { active: 0, mlp: write, normed_stream: 0, queries: vec![write], keys: vec![sibling], ..LayerNodes::default() };
        (artifact, vec![layer], sibling)
    }
    fn family(value: f64) -> FamilyInputs {
        FamilyInputs { rows: 1, slots: vec![SlotValues::Raw(array![[value]])], layout: None }
    }
    fn execute(actions: &[Action], artifact: &Artifact, layers: &[LayerNodes]) -> (Array2<f64>, Array2<f64>) {
        let mut specs = BTreeMap::new();
        for action in actions {
            if matches!(action, Action::Input { .. }) {
                let site = action.site();
                add_input_spec(&mut specs, artifact, &layers[site / KINDS.len()], action, 1).unwrap();
            }
        }
        let (program, forks, _) = fork_inputs(artifact, &specs).unwrap();
        let clean_donor = artifact.execute(&family(6.0)).unwrap();
        let mut edits = BTreeMap::new();
        let mut donor = BTreeMap::new();
        let mut controls: BTreeMap<usize, Vec<NodeEdit>> = BTreeMap::new();
        for action in actions {
            let mapped = site_edits(layers, &program, artifact, &forks, action, 1).unwrap();
            for (node, edit) in mapped.pre_write { controls.entry(node).or_default().push(edit); }
            for (node, edit) in mapped.held {
                if let NodeEdit::Mix { donor: key, .. } = &edit {
                    donor.insert(*key, clean_donor.values[key.node].row(key.row).to_owned());
                }
                edits.entry(node).or_insert_with(Vec::new).push(edit);
            }
        }
        prepend_controls(&mut edits, controls);
        let trace = program.execute_edited(&family(2.0), |node, value, earlier| apply_node_edits(node, value, earlier, &edits, &donor)).unwrap();
        (trace.values[program.place(layers[0].mlp).unwrap()].clone(), trace.values[program.place(layers[0].keys[0]).unwrap()].clone())
    }
    fn input_mix() -> Action {
        Action::Input { site: 5, change: InputChange::Mix { row: 0, alpha: 0.5 } }
    }
    #[test]
    fn nonlinear_rule_executes_after_input_mix() {
        let (artifact, layers, _) = fixture(true);
        assert_eq!(artifact.execute(&family(2.0)).unwrap().values[layers[0].mlp][[0, 0]], 2.0);
        assert_eq!(artifact.execute(&family(6.0)).unwrap().values[layers[0].mlp][[0, 0]], 6.0);
        let (write, sibling) = execute(&[input_mix()], &artifact, &layers);
        assert_eq!(write[[0, 0]], 0.0); // current output-mix translation falsely returns native 4
        assert_eq!(sibling[[0, 0]], 2.0);
    }
    #[test]
    fn input_phase_order_and_output_add_use_intervened_input() {
        let (artifact, layers, _) = fixture(false);
        let scale = Action::Input { site: 5, change: InputChange::Scale { rows: Rows::All, cols: (0, 1), scale: 2.0 } };
        assert_eq!(execute(&[input_mix(), scale.clone()], &artifact, &layers).0[[0, 0]], 8.0);
        assert_eq!(execute(&[scale, input_mix()], &artifact, &layers).0[[0, 0]], 5.0);
        let add = Action::Output { site: 5, change: OutputChange::Add { left: Arc::new(array![[1.0]]), right: Arc::new(array![[1.0]]) } };
        assert_eq!(execute(&[add, input_mix()], &artifact, &layers).0[[0, 0]], 8.0);
    }
    #[test]
    fn q_input_mix_keeps_k_sibling_clean() {
        let (artifact, layers, _) = fixture(false);
        let action = Action::Input { site: 0, change: InputChange::Mix { row: 0, alpha: 0.5 } };
        let (q, k) = execute(&[action], &artifact, &layers);
        assert_eq!(q[[0, 0]], 4.0);
        assert_eq!(k[[0, 0]], 2.0);
    }
    #[test]
    fn missing_input_place_does_not_mix_output() {
        let (mut artifact, layers, _) = fixture(false);
        artifact.places.retain(|(native, _)| *native != layers[0].active);
        assert_eq!(execute(&[input_mix()], &artifact, &layers).0[[0, 0]], 2.0);
    }

    fn composed_mlp(bypass: bool) -> (OperatorProgram, Artifact, Vec<LayerNodes>) {
        use crate::artifact::{Argument, Callee};
        let interface = Interface::native(1).unwrap();
        let native = OperatorProgram {
            declarations: Declarations { domains: vec![], slots: vec![Slot::Raw { width: 1 }], parameters: 0 },
            bases: vec![], rules: vec![],
            operators: vec![Arc::new(Operator::identity("I", interface.clone()))],
            nodes: vec![
                Node::Raw { slot: 0 },
                Node::Affine { terms: vec![(0, 0)], bias: None },
                Node::Pointwise { input: 1, laws: vec![crate::operator_program::Law::Relu] },
                Node::Affine { terms: if bypass { vec![(2, 0), (0, 0)] } else { vec![(2, 0)] }, bias: None },
                Node::Affine { terms: vec![(0, 0)], bias: None },
                Node::Affine { terms: vec![(3, 0), (4, 0)], bias: None },
            ], output: 5,
        };
        let body = Rule { name: "composed ReLU".into(), inputs: vec![interface], nodes: vec![
            Node::Param { index: 0 },
            Node::Pointwise { input: 0, laws: vec![crate::operator_program::Law::Relu] },
            Node::Affine { terms: if bypass { vec![(1, 0), (0, 0)] } else { vec![(1, 0)] }, bias: None },
        ], output: 2 };
        let candidate = Artifact::native(&native).unwrap().replace_block("MLP", Callee::New(body), vec![Argument::Native(0)], 3, vec![]).unwrap();
        let layers = vec![LayerNodes { normed: 0, pre: 1, active: 2, mlp: 3, keys: vec![4], ..LayerNodes::default() }];
        (native, candidate, layers)
    }

    #[test]
    fn native_entry_dominance_lifts_input_mix_through_composed_rule() {
        let (native, artifact, layers) = composed_mlp(false);
        assert!(artifact.place(1).is_none() && artifact.place(2).is_none());
        let action = Action::Input { site: 4, change: InputChange::Mix { row: 0, alpha: 0.5 } };
        let mut specs = BTreeMap::new();
        add_input_spec(&mut specs, &artifact, &layers[0], &action, 1).unwrap();
        lift_input_boundaries(&native, &artifact, &mut specs).unwrap();
        assert_eq!(specs[&4].1, vec![3]);
        let (program, forks, _) = fork_inputs(&artifact, &specs).unwrap();
        let mapped = site_edits(&layers, &program, &artifact, &forks, &action, 1).unwrap();
        assert_eq!(mapped.unheld, 0);
        let donor_trace = artifact.execute(&family(6.0)).unwrap();
        let (mut edits, mut donors) = (BTreeMap::new(), BTreeMap::new());
        for (node, edit) in mapped.held {
            if let NodeEdit::Mix { donor, .. } = &edit {
                donors.insert(*donor, donor_trace.values[donor.node].row(donor.row).to_owned());
            }
            edits.entry(node).or_insert_with(Vec::new).push(edit);
        }
        let trace = program.execute_edited(&family(-2.0), |node, value, earlier| apply_node_edits(node, value, earlier, &edits, &donors)).unwrap();
        assert_eq!(trace.values[program.place(3).unwrap()][[0,0]], 2.0); // ReLU(mix(-2,6)), not mix(ReLU(-2),ReLU(6))=3.
        assert_eq!(trace.values[program.place(4).unwrap()][[0,0]], -2.0); // Shared sibling is untouched.
        let omitted = Action::Input { site: 5, change: InputChange::Scale { rows: Rows::All, cols: (0,1), scale: 0.0 } };
        let missing = site_edits(&layers, &program, &artifact, &forks, &omitted, 1).unwrap();
        assert_eq!(missing.unheld, 1);
        assert!(missing.held.is_empty());
    }

    #[test]
    fn paid_omitted_scale_precedes_write_mix_in_either_episode_order() {
        let (native, plain, layers) = composed_mlp(false);
        let artifact = plain.with_uniform_scale_control(&native, 2, 3).unwrap();
        for scale in [0.0, 2.0] {
            let source = Action::Input { site: 5, change: InputChange::Scale { rows: Rows::All, cols: (0, 1), scale } };
            let write = Action::Output { site: 5, change: OutputChange::Mix { row: 0, alpha: 0.5 } };
            for actions in [[source.clone(), write.clone()], [write.clone(), source.clone()]] {
                let (value, sibling) = execute(&actions, &artifact, &layers);
                assert_eq!(value[[0, 0]], scale + 3.0);
                assert_eq!(sibling[[0, 0]], 2.0);
                assert_eq!(execute(&actions, &plain, &layers).0[[0, 0]], 4.0);
            }
        }
    }

    #[test]
    fn paid_control_follows_write_when_upstream_input_is_forked() {
        let (native, plain, layers) = composed_mlp(false);
        let artifact = plain.with_uniform_scale_control(&native, 2, 3).unwrap();
        let input = Action::Input { site: 4, change: InputChange::Mix { row: 0, alpha: 0.5 } };
        let source = Action::Input { site: 5, change: InputChange::Scale { rows: Rows::All, cols: (0, 1), scale: 2.0 } };
        let mut specs = BTreeMap::new();
        add_input_spec(&mut specs, &artifact, &layers[0], &input, 1).unwrap();
        lift_input_boundaries(&native, &artifact, &mut specs).unwrap();
        let (program, forks, _) = fork_inputs(&artifact, &specs).unwrap();
        assert_eq!(program.controls[0].write, program.place(3).unwrap());
        assert_ne!(program.controls[0].write, artifact.controls[0].write);
        let donor_trace = artifact.execute(&family(6.0)).unwrap();
        let (mut edits, mut donors, mut controls) = (BTreeMap::new(), BTreeMap::new(), BTreeMap::new());
        for action in [source, input] {
            let mapped = site_edits(&layers, &program, &artifact, &forks, &action, 1).unwrap();
            assert_eq!(mapped.unheld, 0);
            for (node, edit) in mapped.pre_write { controls.entry(node).or_insert_with(Vec::new).push(edit); }
            for (node, edit) in mapped.held {
                if let NodeEdit::Mix { donor, .. } = &edit { donors.insert(*donor, donor_trace.values[donor.node].row(donor.row).to_owned()); }
                edits.entry(node).or_insert_with(Vec::new).push(edit);
            }
        }
        prepend_controls(&mut edits, controls);
        let trace = program.execute_edited(&family(2.0), |node, value, earlier| apply_node_edits(node, value, earlier, &edits, &donors)).unwrap();
        assert_eq!(trace.values[program.place(3).unwrap()][[0, 0]], 8.0);
        assert_eq!(trace.values[program.place(4).unwrap()][[0, 0]], 2.0);
    }

    #[test]
    fn native_bypass_prevents_lifting_a_site_intervention_to_a_whole_rule() {
        let (native, artifact, layers) = composed_mlp(true);
        let action = Action::Input { site: 4, change: InputChange::Mix { row: 0, alpha: 0.5 } };
        let mut specs = BTreeMap::new();
        add_input_spec(&mut specs, &artifact, &layers[0], &action, 1).unwrap();
        lift_input_boundaries(&native, &artifact, &mut specs).unwrap();
        assert_eq!(specs[&4].1, vec![1]);
        assert!(fork_inputs(&artifact, &specs).err().unwrap().contains("absent output boundary"));
    }

    #[test]
    fn lifting_does_not_skip_an_exposed_internal_causal_place() {
        let (native, _, layers) = composed_mlp(false);
        let mut artifact = Artifact::native(&native).unwrap().bind("MLP", &[0], 3).unwrap();
        artifact.places.retain(|(node,_)| *node != 1);
        let action = Action::Input { site: 4, change: InputChange::Mix { row: 0, alpha: 0.5 } };
        let mut specs = BTreeMap::new();
        add_input_spec(&mut specs, &artifact, &layers[0], &action, 1).unwrap();
        lift_input_boundaries(&native, &artifact, &mut specs).unwrap();
        assert_eq!(specs[&4].1, vec![1]); // Exposed activation at2 forbids treating wholebody as private.
        assert!(fork_inputs(&artifact, &specs).is_err());
    }
    #[test]
    fn cloning_k_input_must_preserve_q_output_intervention() {
        // Native q=x, k=2x. Candidate k=q+x shares the held q place.
        // Clean fidelity is exact, but an output q intervention is now a causal
        // dependency of k. Input editing k must not erase that dependency.
        let (mut artifact, mut layers, _) = fixture(false);
        let q = layers[0].queries[0];
        let k = layers[0].keys[0];
        artifact.program.nodes[k] = Node::Affine { terms: vec![(q, 0), (0, 0)], bias: None };
        layers[0].mlp = k; // execute() returns this held output for assertions.
        let actions = [
            Action::Output { site: 0, change: OutputChange::Mix { row: 0, alpha: 1.0 } },
            Action::Input { site: 1, change: InputChange::Mix { row: 0, alpha: 0.5 } },
        ];
        assert_eq!(artifact.execute(&family(2.0)).unwrap().values[k][[0, 0]], 4.0);
        assert_eq!(artifact.execute(&family(6.0)).unwrap().values[k][[0, 0]], 12.0);
        let (k_value, _) = execute(&actions, &artifact, &layers);
        assert_eq!(k_value[[0, 0]], 10.0);
        // Unfixed fork_inputs reexecutes q=x from k's mixed x=4, giving k=8,
        // which equals the native counterfactual and falsely hides dependence.
    }
    #[test]
    fn cloned_rule_exceptions_apply_once_and_input_exception_precedes_mix() {
        let (mut artifact, layers, _) = fixture(true);
        artifact.exceptions.extend([
            crate::artifact::Exception { context: vec![], node: 0, column: 0, value: 2.0 },
            crate::artifact::Exception { context: vec![], node: layers[0].mlp, column: 0, value: 1.0 },
        ]);
        // Target incoming place is 2+2=4; donor is 6+2=8. Fork mix gives 6,
        // rule returns f(6)=6, and the cloned output's exception adds exactly 1.
        let (write, sibling) = execute(&[input_mix()], &artifact, &layers);
        assert_eq!(write[[0, 0]], 7.0);
        assert_eq!(sibling[[0, 0]], 4.0);
    }
    #[test]
    fn partial_output_mix_edits_held_heads_and_counts_missing_places() {
        let (mut artifact, mut layers, sibling) = fixture(false);
        let held = layers[0].queries[0];
        layers[0].queries.push(sibling);
        artifact.places.retain(|(native, _)| *native != sibling);
        let action = Action::Output { site: 0, change: OutputChange::Mix { row: 0, alpha: 1.0 } };
        let mapped = site_edits(&layers, &artifact, &artifact, &BTreeMap::new(), &action, 1).unwrap();
        assert_eq!(mapped.held.len(), 1);
        assert_eq!(mapped.held[0].0, held);
        assert_eq!(mapped.unheld, 1);
        let donor_trace = artifact.execute(&family(6.0)).unwrap();
        let mut edits = BTreeMap::new();
        let mut donors = BTreeMap::new();
        for (node, edit) in mapped.held {
            if let NodeEdit::Mix { donor, .. } = &edit {
                donors.insert(*donor, donor_trace.values[donor.node].row(donor.row).to_owned());
            }
            edits.entry(node).or_insert_with(Vec::new).push(edit);
        }
        let trace = artifact.execute_edited(&family(2.0), |node, value, earlier| apply_node_edits(node, value, earlier, &edits, &donors)).unwrap();
        assert_eq!(trace.values[held][[0, 0]], 6.0);
        assert_eq!(trace.values[sibling][[0, 0]], 2.0);
        assert_eq!(trace.values[artifact.program.output][[0, 0]], 8.0);
    }

    #[test]
    fn composite_input_mix_refuses_partial_inputs_or_missing_outputs() {
        let (mut artifact, mut layers, sibling) = fixture(false);
        let held = layers[0].queries[0];
        layers[0].reads = vec![held, sibling];
        layers[0].attention = artifact.program.output;
        artifact.places.retain(|(native, _)| *native != sibling);
        let action = Action::Input { site: 3, change: InputChange::Mix { row: 0, alpha: 0.5 } };
        let mut specs = BTreeMap::new();
        add_input_spec(&mut specs, &artifact, &layers[0], &action, 1).unwrap();
        assert!(fork_inputs(&artifact, &specs).err().unwrap().contains("partially held"));
        // All incoming places absent is a genuine no-op, even when output is held.
        artifact.places.retain(|(native, _)| *native != held);
        let (program, forks, _) = fork_inputs(&artifact, &specs).unwrap();
        let mapped = site_edits(&layers, &program, &artifact, &forks, &action, 1).unwrap();
        assert!(mapped.held.is_empty());
        assert_eq!(mapped.unheld, 2);
        // One shared q input held but only some q output heads held is unsupported.
        let (mut artifact, mut layers, sibling) = fixture(false);
        layers[0].queries.push(sibling);
        artifact.places.retain(|(native, _)| *native != sibling);
        let action = Action::Input { site: 0, change: InputChange::Mix { row: 0, alpha: 0.5 } };
        let mut specs = BTreeMap::new();
        add_input_spec(&mut specs, &artifact, &layers[0], &action, 1).unwrap();
        assert!(fork_inputs(&artifact, &specs).err().unwrap().contains("absent output boundary"));
    }

    #[test]
    fn head_scale_requires_only_the_affected_head_and_combines_held_subsets() {
        let (mut artifact, mut layers, sibling) = fixture(false);
        let held = layers[0].queries[0];
        layers[0].reads = vec![held, sibling];
        layers[0].attention = artifact.program.output;
        artifact.places.retain(|(native, _)| *native != sibling);
        let actions = [
            Action::Input { site: 3, change: InputChange::Scale { rows: Rows::All, cols: (0, 1), scale: 3.0 } },
            Action::Input { site: 3, change: InputChange::Scale { rows: Rows::All, cols: (1, 2), scale: 0.0 } },
        ];
        let mut specs = BTreeMap::new();
        for action in &actions {
            add_input_spec(&mut specs, &artifact, &layers[0], action, 1).unwrap();
        }
        assert_eq!(specs[&3].0, vec![held]);
        let (program, forks, _) = fork_inputs(&artifact, &specs).unwrap();
        let mut edits = BTreeMap::new();
        let mut unheld = 0;
        for action in &actions {
            let mapped = site_edits(&layers, &program, &artifact, &forks, action, 1).unwrap();
            unheld += mapped.unheld;
            for (node, edit) in mapped.held {
                edits.entry(node).or_insert_with(Vec::new).push(edit);
            }
        }
        assert_eq!(unheld, 1);
        let trace = program.execute_edited(&family(2.0), |node, value, earlier| apply_node_edits(node, value, earlier, &edits, &BTreeMap::new())).unwrap();
        assert_eq!(trace.values[program.place(layers[0].attention).unwrap()][[0, 0]], 8.0);
        assert_eq!(trace.values[program.place(held).unwrap()][[0, 0]], 2.0);
    }

    #[test]
    fn weight_edit_with_held_output_needs_its_input_place() {
        let (mut artifact, layers, _) = fixture(false);
        artifact.places.retain(|(native, _)| *native != layers[0].active);
        let action = Action::Output { site: 5, change: OutputChange::Add { left: Arc::new(array![[1.0]]), right: Arc::new(array![[1.0]]) } };
        assert!(site_edits(&layers, &artifact, &artifact, &BTreeMap::new(), &action, 1).err().unwrap().contains("incoming place"));
        artifact.places.retain(|(native, _)| *native != layers[0].mlp);
        let mapped = site_edits(&layers, &artifact, &artifact, &BTreeMap::new(), &action, 1).unwrap();
        assert!(mapped.held.is_empty());
        assert_eq!(mapped.unheld, 1);
    }
}

#[cfg(test)]
mod tail_contract_tests {
    use super::*;
    use crate::artifact::Exception;
    use crate::operator_program::{Declarations, Interface, Operator, Slot};

    fn fixture() -> (OperatorProgram, Artifact) {
        let native = OperatorProgram {
            declarations: Declarations { domains: vec![], slots: vec![Slot::Raw { width: 2 }], parameters: 0 },
            operators: vec![Arc::new(Operator::identity("head", Interface::native(2).unwrap()))],
            bases: vec![],
            rules: vec![],
            nodes: vec![
                Node::Raw { slot: 0 },
                Node::Affine { terms: vec![(0, 0)], bias: None },
                Node::RmsNorm { input: 1, epsilon: 0.25 },
                Node::Affine { terms: vec![(2, 0)], bias: None },
            ],
            output: 3,
        };
        let artifact = Artifact::native(&native).unwrap();
        (native, artifact)
    }

    #[test]
    fn head_reuse_checks_wiring_and_norm_constants() {
        let (native, mut artifact) = fixture();
        assert!(check_native_tail(&native, &artifact, 1).is_ok());
        artifact.program.nodes[2] = Node::RmsNorm { input: 1, epsilon: 4.0 };
        assert!(check_native_tail(&native, &artifact, 1).is_err());
        artifact.program.nodes[2] = native.nodes[2].clone();
        artifact.program.nodes[3] = Node::Affine { terms: vec![(1, 0)], bias: None };
        assert!(check_native_tail(&native, &artifact, 1).is_err());
    }

    #[test]
    fn head_reuse_cannot_discard_exceptions_or_retargeted_output() {
        let (native, mut artifact) = fixture();
        artifact.exceptions.push(Exception { context: vec![], node: 2, column: 0, value: 1.0 });
        assert!(check_native_tail(&native, &artifact, 1).is_err());
        artifact.exceptions.clear();
        artifact.program.output = 1;
        assert!(check_native_tail(&native, &artifact, 1).is_err());
    }

    #[test]
    fn head_reuse_accepts_reindexed_native_head() {
        let (native, mut artifact) = fixture();
        artifact.program.nodes.insert(2, Node::Affine { terms: vec![(1, 0)], bias: None });
        artifact.program.nodes[4] = Node::Affine { terms: vec![(3, 0)], bias: None };
        artifact.program.output = 4;
        artifact.places = vec![(0, 0), (1, 1), (2, 3), (3, 4)];
        assert!(check_native_tail(&native, &artifact, 1).is_ok());
    }
}

#[cfg(test)]
mod cuda_edit_reference_tests {
    use super::*;
    #[test]
    fn device_edits_preserve_order_partial_rows_and_low_rank_input() {
        let device = gam_gpu::tensor::Device::host();
        let input = ndarray::array![[0.2, -0.7], [1.1, 0.4]];
        // This starting value includes the prior exception addition. The edit
        // callback must consume it, rather than reconstruct a clean value.
        let value = ndarray::array![[3.4, 0.6], [-0.3, 1.7]];
        let key = DonorKey { node: 7, row: 1 };
        let donor = BTreeMap::from([(key.clone(), ndarray::array![2.3])]);
        let edits = BTreeMap::from([(
            1,
            vec![
                NodeEdit::Scale { rows: Rows::One(0), columns: 0..1, scale: 0.3 },
                NodeEdit::Mix { row: 1, columns: 1..2, alpha: 0.4, donor: key },
                NodeEdit::AddMap { input: 0, left: Arc::new(ndarray::array![[0.6], [-0.2]]), right: Arc::new(ndarray::array![[0.3], [0.8]]) },
            ],
        )]);
        let mut expected = value.clone();
        apply_node_edits(1, &mut expected, &[input.clone()], &edits, &donor).unwrap();
        let roots = [device.upload(input.view()).unwrap(), device.upload(value.view()).unwrap()];
        let actual = apply_device_node_edits(&device, 1, 2, |n| Ok(&roots[n]), &edits, &donor).unwrap().unwrap();
        let actual = device.download(&actual).unwrap();
        for (a, b) in actual.iter().zip(expected.iter()) {
            assert!((a - b).abs() < 1e-12, "{a} differs from {b}");
        }
        assert!(apply_device_node_edits(&device, 99, 2, |_| Err("unmaterialized head".into()), &edits, &donor).unwrap().is_none());
    }
}
