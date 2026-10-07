//! VPD's slices with intrinsic gates as a library explanation (#2951, the main line on vpd4l):
//! `library_mdl` fits it like any library (snapshot scoring, removal through the whole program,
//! the per-token budget), on the shared verbatim experiments.
//!
//! The start is `vpd_start`'s: per arm, components of VPD's rank-one slices (site, index), each with
//! one gate that reads only its own model's input at its own row, so the explanation is causal and
//! autonomous: no gating network, no all-on pass. A component is on at a row iff its gate's
//! pre-activation is positive, `‖V_bᵀx‖ − τ_b` (its own read) or `g_bᵀx + c_b − τ_b` (a separate
//! direction), and then all its slices run (`Node::GroupNorm`, `Node::Gated`). Its rank, the slices
//! it runs, counts against the per-token budget. The remainder `W − Σ U_i V_iᵀ` is dropped: the
//! explanation does not use it, so it is not charged.
//!
//! Per layer, a component's gate reads at one of four stages: the attention's input (q, k, v
//! slices), the heads' concatenated outputs (o slices gated on their own read), the MLP's input
//! (c_fc slices) and its activations (down_proj slices gated on their own read). The attention
//! input stage's read `V_A` stacks every q, k and v slice of the stage's components, grouped by
//! component, so `‖V_bᵀx‖` is over all of a component's input-side slices; the gated activations
//! are then split by fixed selections into the q, k and v slices, and each head's q, k and v are
//! the heads' rows of those slices' writes. A write-side slice (o, down_proj) of a component gated
//! at the block's input takes that gate's column (a fixed selection of the gate's columns).
//!
//! Training uses the expected gate: each stage holds a fixed operator `{stage}.softness` (one
//! entry per component, `HARD` in the program as built, so evaluation is the hard gate) that
//! `library_mdl` sets to the threshold's posterior deviation `σ_b` for a pass with a gradient, with
//! the threshold at its mean there, so the gate is `Φ(z_b / σ_b)`, the step `H` integrated exactly
//! over the threshold's posterior, and the gate's threshold and direction take gradients.
//!
//! Prior groups: per slice its read row and its write column (over every head for q, k and v),
//! per direction gate its row, and per stage of a layer its thresholds (one group, as a transcoder
//! block's gate biases are). Each head recomputes the attention input stage's gated activations;
//! the heads share the stage's operators.

use crate::{
    artifact::{Argument, Artifact, Callee, Owner},
    explanation_battery::{KINDS, Kind, load_factors},
    library_mdl::{Cells, Explanation, Group, Layer, mean_squares},
    operator_program::{FamilyInputs, Interface, LabelKind, Node, Operator, OperatorProgram, Provenance, Rule, Trace, exact_precision},
    run_check::LayerNodes,
};
use ndarray::{Array2, Axis, s};
use serde::Deserialize;
use std::{collections::BTreeMap, path::Path};

/// The softness of the hard gate: `Φ(z / s)` at `s = 10⁻³⁰` (a normal float32) is `H(z)` for every
/// `|z| > 10⁻²⁹`, and ½ at `z = 0`, where an own gate's read `V_bᵀx` is zero and so is the
/// component's output. At `s = 0` the gate `z / s` is undefined at `z = 0`: a component with
/// `τ_b = 0` on a row whose read is zero (an MLP whose components are all off) made the forward NaN.
pub const HARD: f64 = 1e-30;

fn error(e: impl std::fmt::Display) -> String {
    format!("library vpd: {e}")
}

/// A component's gate read, as `vpd_start` writes it.
#[derive(Clone, Debug, Deserialize)]
#[serde(rename_all = "snake_case")]
enum Read {
    /// `|v_iᵀx|`: slice `index` of `site`, at that site's input (a component's input-side slices).
    Own([usize; 2]),
    /// `gᵀx + c` at `site`'s input (`coefficients`: `g` then `c`).
    Direction { site: usize, coefficients: Vec<f64> },
}

/// One component of a start file.
#[derive(Clone, Debug, Deserialize)]
struct Component {
    read: Read,
    tau: f64,
    slices: Vec<[usize; 2]>,
}

#[derive(Deserialize)]
struct ArmRecord {
    arm: String,
    components: Vec<Component>,
}

fn kind_of(site: usize) -> Kind {
    KINDS[site % KINDS.len()]
}

/// The stage of a layer at which a gate reading `site`'s input is computed.
fn stage(site: usize) -> usize {
    match kind_of(site) {
        Kind::Query | Kind::Key | Kind::Value => 0,
        Kind::Output => 1,
        Kind::Up => 2,
        Kind::Down => 3,
    }
}

fn dense(name: &str, rows: Interface, cols: Interface, values: Array2<f64>) -> Result<Operator, String> {
    let precision = exact_precision(values.iter().copied()).map_err(error)?;
    Operator::dense(name, rows, cols, values, precision, Provenance::derived(&[&Provenance::native("vpd")], "VPD slices with intrinsic gates".into())).map_err(error)
}

/// Rows grouped by component: one group per entry of `widths` (all positive).
fn grouped(widths: &[usize]) -> Result<Interface, String> {
    Interface::new(widths.iter().enumerate().map(|(b, &width)| crate::operator_program::Group { width, label: crate::operator_program::Label::new(LabelKind::Unit, b as u32) }).collect()).map_err(error)
}

fn units(width: usize) -> Result<Interface, String> {
    Interface::uniform(width, 1, LabelKind::Unit, 0).map_err(error)
}

/// A 0/1 selection of `picked` (rows) from `from` columns.
fn selection(picked: &[usize], from: usize) -> Array2<f64> {
    let mut out = Array2::zeros((picked.len(), from));
    for (r, &c) in picked.iter().enumerate() {
        out[[r, c]] = 1.0;
    }
    out
}

fn index_of(program: &OperatorProgram, name: &str) -> Result<usize, String> {
    let found: Vec<usize> = program.operators.iter().enumerate().filter(|(_, op)| op.name == name).map(|(i, _)| i).collect();
    match found[..] {
        [i] => Ok(i),
        _ => Err(error(format!("no unique operator {name}"))),
    }
}

/// One arm's explanation of the split native program `native` with its `layers`, from VPD's
/// `decomposition` and the start file `start` (`vpd_start`'s components, arm `arm`).
pub fn explanation(native: &OperatorProgram, layers: &[LayerNodes], decomposition: &Path, start: &Path, arm: &str) -> Result<Explanation, String> {
    let factors = load_factors(decomposition)?;
    let records: Vec<ArmRecord> = serde_json::from_slice(&std::fs::read(start).map_err(|e| error(format!("{}: {e}", start.display())))?).map_err(error)?;
    let components = records.into_iter().find(|r| r.arm == arm).ok_or_else(|| error(format!("{}: no arm {arm}", start.display())))?.components;
    if factors.len() != KINDS.len() * layers.len() {
        return Err(error(format!("{} sites for {} layers", factors.len(), layers.len())));
    }
    let read_site = |c: &Component| match &c.read {
        Read::Own([site, _]) => *site,
        Read::Direction { site, .. } => *site,
    };
    let mut artifact = Artifact::native(native)?;
    // Per layer and stage, the components gated there.
    let mut at: BTreeMap<(usize, usize), Vec<usize>> = BTreeMap::new();
    for (b, c) in components.iter().enumerate() {
        let site = read_site(c);
        if c.slices.iter().any(|[s, _]| s / KINDS.len() != site / KINDS.len()) {
            return Err(error(format!("component {b}: slices outside its gate's layer")));
        }
        at.entry((site / KINDS.len(), stage(site))).or_default().push(b);
    }
    let direction = components.iter().any(|c| matches!(c.read, Read::Direction { .. }));
    let interfaces = native.interfaces().map_err(error)?;
    let node_interface = |node: usize| -> Result<Interface, String> { interfaces.get(node).cloned().ok_or_else(|| error(format!("no node {node}"))) };
    for (l, layer) in layers.iter().enumerate() {
        let name = format!("library.l{l}");
        let site = |kind: Kind| KINDS.len() * l + KINDS.iter().position(|k| *k == kind).unwrap_or(0);
        let slices_on = |b: usize, s: usize| -> Vec<usize> { components[b].slices.iter().filter(|[t, _]| *t == s).map(|[_, i]| *i).collect() };
        // ---------------------------------------------------------------- the attention's input stage
        let a_comps = at.get(&(l, 0)).cloned().unwrap_or_default();
        let (q, k, v) = (site(Kind::Query), site(Kind::Key), site(Kind::Value));
        // The stacked read: per component its q, k, v slices in that order; each slice's row.
        let mut read_rows: Vec<(usize, usize)> = Vec::new();
        let mut widths = Vec::new();
        for &b in &a_comps {
            let mut w = 0;
            for s in [q, k, v] {
                for i in slices_on(b, s) {
                    read_rows.push((s, i));
                    w += 1;
                }
            }
            if w == 0 {
                return Err(error(format!("component {b}: gated at the attention's input with no q, k or v slice")));
            }
            widths.push(w);
        }
        if a_comps.is_empty() {
            return Err(error(format!("layer {l}: no component at the attention's input")));
        }
        let x = layer.normed_stream;
        if layer.reads.is_empty() {
            return Err(error(format!("layer {l}: no heads")));
        }
        // Each head's q, k and v are one map of the normed stream (no head norm).
        for &read in &layer.reads {
            let Node::Attend { query, key, value, .. } = &native.nodes[read] else { return Err(error(format!("layer {l}: a head read is not an attention node"))) };
            for node in [*query, *key, *value] {
                if !matches!(&native.nodes[node], Node::Affine { terms, bias: None } if terms.len() == 1 && terms[0].0 == x) {
                    return Err(error(format!("layer {l}: a head's projection is not one map of the normed stream")));
                }
            }
        }
        let x_interface = node_interface(x)?;
        let d = x_interface.width();
        let stacked = grouped(&widths)?;
        let r_a = read_rows.len();
        let v_a = Array2::from_shape_fn((r_a, d), |(r, j)| factors[read_rows[r].0].v[[j, read_rows[r].1]]);
        let gate_ops = |prefix: &str, comps: &[usize], cols: &Interface| -> Result<Vec<Operator>, String> {
            let count = comps.len();
            let mut ops = Vec::new();
            match direction {
                false => {
                    ops.push(Operator::identity(format!("{prefix}.gate_identity"), units(count)?));
                    ops.push(dense(&format!("{prefix}.threshold"), units(count)?, Interface::constant(), Array2::from_shape_fn((count, 1), |(b, _)| -components[comps[b]].tau))?);
                }
                true => {
                    let w = cols.width();
                    let g = Array2::from_shape_fn((count, w), |(b, j)| match &components[comps[b]].read {
                        Read::Direction { coefficients, .. } => coefficients[j],
                        Read::Own(_) => 0.0,
                    });
                    let c = Array2::from_shape_fn((count, 1), |(b, _)| match &components[comps[b]].read {
                        Read::Direction { coefficients, .. } => coefficients[w] - components[comps[b]].tau,
                        Read::Own(_) => 0.0,
                    });
                    ops.push(dense(&format!("{prefix}.direction"), units(count)?, cols.clone(), g)?);
                    ops.push(dense(&format!("{prefix}.threshold"), units(count)?, Interface::constant(), c)?);
                }
            }
            ops.push(dense(&format!("{prefix}.softness"), units(count)?, Interface::constant(), Array2::from_elem((count, 1), HARD))?);
            Ok(ops)
        };
        // The attention input stage's nodes from `input` (node 0 of a rule's nodes so far): the
        // stacked read, the gate, the gate's softness, the gated activations; returns (gated node,
        // gate node, softness node).
        let stage_nodes = |nodes: &mut Vec<Node>, input: usize, (read, gate_a, gate_b, soft): (usize, usize, usize, usize)| -> (usize, usize, usize) {
            // A direction gate and its softness come before the reads they gate, so each row reads
            // only its components on (`DeviceProgram`'s gated reads); an own gate reads them.
            let (a, z, s) = if direction {
                nodes.push(Node::Affine { terms: vec![(input, gate_a)], bias: Some(gate_b) });
                nodes.push(Node::Constant { operator: soft });
                nodes.push(Node::Affine { terms: vec![(input, read)], bias: None });
                (nodes.len() - 1, nodes.len() - 3, nodes.len() - 2)
            } else {
                nodes.push(Node::Affine { terms: vec![(input, read)], bias: None });
                let a = nodes.len() - 1;
                nodes.push(Node::GroupNorm { input: a });
                nodes.push(Node::Affine { terms: vec![(nodes.len() - 1, gate_a)], bias: Some(gate_b) });
                nodes.push(Node::Constant { operator: soft });
                (a, nodes.len() - 2, nodes.len() - 1)
            };
            nodes.push(Node::Gated { value: a, gate: z, scale: Some(s) });
            (nodes.len() - 1, z, s)
        };
        let selections: Vec<(usize, Vec<usize>)> = [q, k, v].iter().map(|&s| (s, (0..r_a).filter(|&r| read_rows[r].0 == s).collect())).collect();
        for (h, &read) in layer.reads.iter().enumerate() {
            let Node::Attend { query, key, value, scale, rotary, causal } = native.nodes[read].clone() else {
                return Err(error(format!("layer {l} head {h}: the read is not an attention node")));
            };
            let head_rows = |node: usize| -> Result<Interface, String> { node_interface(node) };
            let hd = head_rows(query)?.width();
            let base = artifact.program.operators.len();
            let mut operators = Vec::new();
            let shared = |artifact: &Artifact, part: &str| index_of(&artifact.program, &format!("{name}.attn.{part}"));
            let (read_op, gate_a, gate_b, soft) = if h == 0 {
                operators.push(dense(&format!("{name}.attn.read"), stacked.clone(), x_interface.clone(), v_a.clone())?);
                let mut gates = gate_ops(&format!("{name}.attn"), &a_comps, &x_interface)?;
                operators.append(&mut gates);
                for (s, picked) in &selections {
                    if !picked.is_empty() {
                        operators.push(dense(&format!("{name}.attn.select{}", s % KINDS.len()), units(picked.len())?, stacked.clone(), selection(picked, r_a))?);
                    }
                }
                (base, base + 1, base + 2, base + 3)
            } else {
                let gate_names = if direction { ("direction", "threshold") } else { ("gate_identity", "threshold") };
                (shared(&artifact, "read")?, shared(&artifact, gate_names.0)?, shared(&artifact, gate_names.1)?, shared(&artifact, "softness")?)
            };
            let mut nodes = vec![Node::Param { index: 0 }];
            let (gated, _, _) = stage_nodes(&mut nodes, 0, (read_op, gate_a, gate_b, soft));
            let mut projections = Vec::new();
            for (j, (s, picked)) in selections.iter().enumerate() {
                let rows = head_rows([query, key, value][j])?;
                let w_name = format!("{name}.h{h}.{}", ["q", "k", "v"][j]);
                if picked.is_empty() {
                    return Err(error(format!("layer {l}: no {} slice", ["q", "k", "v"][j])));
                }
                let select = if h == 0 {
                    base + operators.iter().position(|o| o.name == format!("{name}.attn.select{}", s % KINDS.len())).ok_or("selection")?
                } else {
                    shared(&artifact, &format!("select{}", s % KINDS.len()))?
                };
                let u = Array2::from_shape_fn((rows.width(), picked.len()), |(r, c)| factors[*s].u[[read_rows[picked[c]].1, h * hd + r]]);
                operators.push(dense(&w_name, rows, units(picked.len())?, u)?);
                let write = base + operators.len() - 1;
                nodes.push(Node::Affine { terms: vec![(gated, select)], bias: None });
                nodes.push(Node::Affine { terms: vec![(nodes.len() - 1, write)], bias: None });
                projections.push(nodes.len() - 1);
            }
            nodes.push(Node::Attend { query: projections[0], key: projections[1], value: projections[2], scale, rotary, causal });
            let rule = Rule { name: format!("{name}.h{h}"), inputs: vec![x_interface.clone()], output: nodes.len() - 1, nodes };
            artifact = artifact.replace_block(&format!("{name}.h{h}"), Callee::New(rule), vec![Argument::Native(x)], read, operators)?;
        }
        // ---------------------------------------------------------------- the attention's output
        let o = site(Kind::Output);
        let o_own = at.get(&(l, 1)).cloned().unwrap_or_default();
        // The o slices: those of components gated at the attention's input, then of components
        // gated on their own o read; per o-carrying component its slices.
        let o_carriers: Vec<usize> = a_comps.iter().copied().filter(|&b| !slices_on(b, o).is_empty()).chain(o_own.iter().copied()).collect();
        if !o_carriers.is_empty() {
            let o_rows: Vec<usize> = o_carriers.iter().flat_map(|&b| slices_on(b, o)).collect();
            let o_widths: Vec<usize> = o_carriers.iter().map(|&b| slices_on(b, o).len()).collect();
            let reads_cols: Vec<Interface> = layer.reads.iter().map(|&r| node_interface(r)).collect::<Result<_, _>>()?;
            let concat = concat_interface(&reads_cols)?;
            let width = concat.width();
            let o_stacked = grouped(&o_widths)?;
            let out_rows = node_interface(layer.attention)?;
            let base = artifact.program.operators.len();
            let mut operators = vec![
                dense(&format!("{name}.o.read"), o_stacked.clone(), concat.clone(), Array2::from_shape_fn((o_rows.len(), width), |(r, j)| factors[o].v[[j, o_rows[r]]]))?,
                dense(&format!("{name}.o.write"), out_rows.clone(), o_stacked.clone(), Array2::from_shape_fn((out_rows.width(), o_rows.len()), |(r, c)| factors[o].u[[o_rows[c], r]]))?,
            ];
            let heads = layer.reads.len();
            let mut nodes: Vec<Node> = (0..heads).map(|index| Node::Param { index }).collect();
            nodes.push(Node::Param { index: heads });
            nodes.push(Node::Concat { parts: (0..heads).collect() });
            let c = nodes.len() - 1;
            // The o reads after their gate when the gate does not read them (no own norm gates), so
            // each row reads only its components on (`DeviceProgram`'s gated reads).
            let late = (!o_own.is_empty() && !direction).then(|| {
                nodes.push(Node::Affine { terms: vec![(c, base)], bias: None });
                nodes.len() - 1
            });
            // The gate columns of the o carriers: from the attention input stage's gate (recomputed
            // here from x), then the own o gates.
            let (mut gate_parts, mut soft_parts) = (Vec::new(), Vec::new());
            let carried: Vec<usize> = o_carriers.iter().filter(|b| a_comps.contains(b)).map(|b| a_comps.iter().position(|a| a == b).unwrap_or(0)).collect();
            if !carried.is_empty() {
                let gate_names = if direction { ("direction", "threshold") } else { ("gate_identity", "threshold") };
                let stage_ops = (
                    index_of(&artifact.program, &format!("{name}.attn.read"))?,
                    index_of(&artifact.program, &format!("{name}.attn.{}", gate_names.0))?,
                    index_of(&artifact.program, &format!("{name}.attn.{}", gate_names.1))?,
                    index_of(&artifact.program, &format!("{name}.attn.softness"))?,
                );
                let (_, z, s) = stage_nodes(&mut nodes, heads, stage_ops);
                operators.push(dense(&format!("{name}.o.select_gate"), units(carried.len())?, units(a_comps.len())?, selection(&carried, a_comps.len()))?);
                nodes.push(Node::Affine { terms: vec![(z, base + operators.len() - 1)], bias: None });
                gate_parts.push(nodes.len() - 1);
                nodes.push(Node::Affine { terms: vec![(s, base + operators.len() - 1)], bias: None });
                soft_parts.push(nodes.len() - 1);
            }
            if !o_own.is_empty() {
                // Own o gates read the o carriers' own rows: the group norms of the read's own groups.
                let own_first = carried.len();
                operators.push(dense(&format!("{name}.o.select_own"), units(o_own.len())?, units(o_carriers.len())?, selection(&(own_first..o_carriers.len()).collect::<Vec<_>>(), o_carriers.len()))?);
                let select = base + operators.len() - 1;
                let mut gates = gate_ops(&format!("{name}.o"), &o_own, &concat)?;
                let first = base + operators.len();
                operators.append(&mut gates);
                if direction {
                    nodes.push(Node::Affine { terms: vec![(c, first)], bias: Some(first + 1) });
                } else {
                    nodes.push(Node::GroupNorm { input: late.ok_or_else(|| error(format!("layer {l}: own o gates without the o reads")))? });
                    nodes.push(Node::Affine { terms: vec![(nodes.len() - 1, select)], bias: None });
                    nodes.push(Node::Affine { terms: vec![(nodes.len() - 1, first)], bias: Some(first + 1) });
                }
                gate_parts.push(nodes.len() - 1);
                nodes.push(Node::Constant { operator: first + 2 });
                soft_parts.push(nodes.len() - 1);
            }
            let (z, s) = joined(&mut nodes, gate_parts, soft_parts);
            let a_o = late.unwrap_or_else(|| {
                nodes.push(Node::Affine { terms: vec![(c, base)], bias: None });
                nodes.len() - 1
            });
            nodes.push(Node::Gated { value: a_o, gate: z, scale: Some(s) });
            nodes.push(Node::Affine { terms: vec![(nodes.len() - 1, base + 1)], bias: None });
            let mut inputs = reads_cols.clone();
            inputs.push(x_interface.clone());
            let arguments = layer.reads.iter().map(|&r| Argument::Native(r)).chain([Argument::Native(x)]).collect();
            let rule = Rule { name: format!("{name}.o"), inputs, output: nodes.len() - 1, nodes };
            artifact = artifact.replace_block(&format!("{name}.o"), Callee::New(rule), arguments, layer.attention, operators)?;
        }
        // ---------------------------------------------------------------- the MLP
        let (fc, dn) = (site(Kind::Up), site(Kind::Down));
        let f_comps = at.get(&(l, 2)).cloned().unwrap_or_default();
        let d_own = at.get(&(l, 3)).cloned().unwrap_or_default();
        let Node::Pointwise { input: pre, laws } = native.nodes[layer.active].clone() else {
            return Err(error(format!("layer {l}: the MLP activation is not one pointwise law")));
        };
        let up_rows = node_interface(pre)?;
        let h2 = layer.normed;
        let h2_interface = node_interface(h2)?;
        if f_comps.is_empty() && d_own.is_empty() {
            // No component in the MLP (an attention-only model's zero MLP): the block adds zero.
            let out_rows = node_interface(layer.mlp)?;
            let zero = dense(&format!("{name}.mlp.zero"), out_rows.clone(), Interface::constant(), Array2::zeros((out_rows.width(), 1)))?;
            let base = artifact.program.operators.len();
            let rule = Rule { name: format!("{name}.mlp"), inputs: vec![h2_interface.clone()], output: 1, nodes: vec![Node::Param { index: 0 }, Node::Constant { operator: base }] };
            artifact = artifact.replace_block(&format!("{name}.mlp"), Callee::New(rule), vec![Argument::Native(h2)], layer.mlp, vec![zero])?;
            continue;
        }
        let fc_rows: Vec<usize> = f_comps.iter().flat_map(|&b| slices_on(b, fc)).collect();
        let fc_widths: Vec<usize> = f_comps.iter().map(|&b| slices_on(b, fc).len()).collect();
        if fc_widths.iter().any(|w| *w == 0) || f_comps.is_empty() {
            return Err(error(format!("layer {l}: an MLP component gated at the MLP's input without a c_fc slice")));
        }
        let dn_carriers: Vec<usize> = f_comps.iter().copied().filter(|&b| !slices_on(b, dn).is_empty()).chain(d_own.iter().copied()).collect();
        let dn_rows: Vec<usize> = dn_carriers.iter().flat_map(|&b| slices_on(b, dn)).collect();
        let dn_widths: Vec<usize> = dn_carriers.iter().map(|&b| slices_on(b, dn).len()).collect();
        let out_rows = node_interface(layer.mlp)?;
        let (fc_stacked, dn_stacked) = (grouped(&fc_widths)?, grouped(&dn_widths)?);
        let d2 = h2_interface.width();
        let hidden = up_rows.width();
        let base = artifact.program.operators.len();
        let mut operators = vec![
            dense(&format!("{name}.mlp.fc_read"), fc_stacked.clone(), h2_interface.clone(), Array2::from_shape_fn((fc_rows.len(), d2), |(r, j)| factors[fc].v[[j, fc_rows[r]]]))?,
            dense(&format!("{name}.mlp.fc_write"), up_rows.clone(), fc_stacked.clone(), Array2::from_shape_fn((hidden, fc_rows.len()), |(r, c)| factors[fc].u[[fc_rows[c], r]]))?,
            dense(&format!("{name}.mlp.dn_read"), dn_stacked.clone(), up_rows.clone(), Array2::from_shape_fn((dn_rows.len(), hidden), |(r, j)| factors[dn].v[[j, dn_rows[r]]]))?,
            dense(&format!("{name}.mlp.dn_write"), out_rows.clone(), dn_stacked.clone(), Array2::from_shape_fn((out_rows.width(), dn_rows.len()), |(r, c)| factors[dn].u[[dn_rows[c], r]]))?,
        ];
        let mut gates = gate_ops(&format!("{name}.mlp.fc"), &f_comps, &h2_interface)?;
        operators.append(&mut gates);
        let mut nodes = vec![Node::Param { index: 0 }];
        let (gated, z_f, s_f) = stage_nodes(&mut nodes, 0, (base, base + 4, base + 5, base + 6));
        nodes.push(Node::Affine { terms: vec![(gated, base + 1)], bias: None });
        nodes.push(Node::Pointwise { input: nodes.len() - 1, laws: laws.clone() });
        let act = nodes.len() - 1;
        // The down reads after their gate when the gate does not read them (as the o reads').
        let late = (!d_own.is_empty() && !direction).then(|| {
            nodes.push(Node::Affine { terms: vec![(act, base + 2)], bias: None });
            nodes.len() - 1
        });
        let (mut gate_parts, mut soft_parts) = (Vec::new(), Vec::new());
        let carried: Vec<usize> = dn_carriers.iter().filter(|b| f_comps.contains(b)).map(|b| f_comps.iter().position(|a| a == b).unwrap_or(0)).collect();
        if !carried.is_empty() {
            operators.push(dense(&format!("{name}.mlp.dn_select_gate"), units(carried.len())?, units(f_comps.len())?, selection(&carried, f_comps.len()))?);
            nodes.push(Node::Affine { terms: vec![(z_f, base + operators.len() - 1)], bias: None });
            gate_parts.push(nodes.len() - 1);
            nodes.push(Node::Affine { terms: vec![(s_f, base + operators.len() - 1)], bias: None });
            soft_parts.push(nodes.len() - 1);
        }
        if !d_own.is_empty() {
            let own_first = carried.len();
            operators.push(dense(&format!("{name}.mlp.dn_select_own"), units(d_own.len())?, units(dn_carriers.len())?, selection(&(own_first..dn_carriers.len()).collect::<Vec<_>>(), dn_carriers.len()))?);
            let select = base + operators.len() - 1;
            let mut gates = gate_ops(&format!("{name}.mlp.dn"), &d_own, &up_rows)?;
            let first = base + operators.len();
            operators.append(&mut gates);
            if direction {
                nodes.push(Node::Affine { terms: vec![(act, first)], bias: Some(first + 1) });
            } else {
                nodes.push(Node::GroupNorm { input: late.ok_or_else(|| error(format!("layer {l}: own down gates without the down reads")))? });
                nodes.push(Node::Affine { terms: vec![(nodes.len() - 1, select)], bias: None });
                nodes.push(Node::Affine { terms: vec![(nodes.len() - 1, first)], bias: Some(first + 1) });
            }
            gate_parts.push(nodes.len() - 1);
            nodes.push(Node::Constant { operator: first + 2 });
            soft_parts.push(nodes.len() - 1);
        }
        let (z, s) = joined(&mut nodes, gate_parts, soft_parts);
        let a_dn = late.unwrap_or_else(|| {
            nodes.push(Node::Affine { terms: vec![(act, base + 2)], bias: None });
            nodes.len() - 1
        });
        nodes.push(Node::Gated { value: a_dn, gate: z, scale: Some(s) });
        nodes.push(Node::Affine { terms: vec![(nodes.len() - 1, base + 3)], bias: None });
        let rule = Rule { name: format!("{name}.mlp"), inputs: vec![h2_interface.clone()], output: nodes.len() - 1, nodes };
        artifact = artifact.replace_block(&format!("{name}.mlp"), Callee::New(rule), vec![Argument::Native(h2)], layer.mlp, operators)?;
    }
    groups_of(artifact, layers, direction)
}

/// The prior groups, trainable operators and layers of the built artifact (module note).
fn groups_of(artifact: Artifact, layers: &[LayerNodes], direction: bool) -> Result<Explanation, String> {
    let program = &artifact.program;
    let named = |name: &str| index_of(program, name);
    let mut groups: Vec<Group> = Vec::new();
    let mut trainable: Vec<usize> = Vec::new();
    let mut out: Vec<Layer> = layers.iter().map(|sites| Layer { sites: sites.clone(), heads: Vec::new(), functions: Vec::new(), sink: None, thresholds: Vec::new(), components: Vec::new() }).collect();
    // A group per row (or column) of an operator, named.
    let rows_of = |op: usize| program.operators[op].rows.width();
    let cols_of = |op: usize| program.operators[op].cols.width();
    for (l, layer) in out.iter_mut().enumerate() {
        let name = format!("library.l{l}");
        let heads = layers[l].reads.len();
        // The attention input stage: each stacked read row with its write columns in every head.
        let read = named(&format!("{name}.attn.read"))?;
        trainable.push(read);
        let mut per_component: BTreeMap<usize, Vec<usize>> = BTreeMap::new();
        let stacked = program.operators[read].rows.clone();
        let mut row = 0;
        let selects: Vec<Option<usize>> = (0..3).map(|j| named(&format!("{name}.attn.select{j}")).ok()).collect();
        // Per stacked row: its map (q, k, v) and its column in that map's writes.
        let mut column_in: Vec<(usize, usize)> = Vec::new();
        let mut counters = [0usize; 3];
        let select_values: Vec<Option<Array2<f64>>> = selects.iter().map(|s| s.map(|op| program.operators[op].matrix())).collect();
        for r in 0..rows_of(read) {
            let j = select_values.iter().position(|m| m.as_ref().is_some_and(|m| m.column(r).iter().any(|v| *v != 0.0))).ok_or_else(|| error("a stacked row in no selection"))?;
            column_in.push((j, counters[j]));
            counters[j] += 1;
        }
        for (b, group) in stacked.groups().iter().enumerate() {
            for _ in 0..group.width {
                let (j, c) = column_in[row];
                let mut cells = vec![];
                groups.push(Group { name: format!("{name}.attn.c{b}.r{row}.read"), cells: vec![Cells { operator: read, rows: vec![row], cols: 0..cols_of(read) }] });
                per_component.entry(b).or_default().push(groups.len() - 1);
                for h in 0..heads {
                    let write = named(&format!("{name}.h{h}.{}", ["q", "k", "v"][j]))?;
                    cells.push(Cells { operator: write, rows: (0..rows_of(write)).collect(), cols: c..c + 1 });
                }
                groups.push(Group { name: format!("{name}.attn.c{b}.r{row}.write"), cells });
                per_component.entry(b).or_default().push(groups.len() - 1);
                row += 1;
            }
        }
        for h in 0..heads {
            for part in ["q", "k", "v"] {
                trainable.push(named(&format!("{name}.h{h}.{part}"))?);
            }
        }
        // The o slices: read rows and write columns; per carrier its component index in the stage.
        if let Ok(o_read) = named(&format!("{name}.o.read")) {
            let o_write = named(&format!("{name}.o.write"))?;
            trainable.extend([o_read, o_write]);
            let o_groups = program.operators[o_read].rows.clone();
            let mut r = 0;
            for (k, group) in o_groups.groups().iter().enumerate() {
                let mut mine = Vec::new();
                for _ in 0..group.width {
                    groups.push(Group { name: format!("{name}.o.k{k}.r{r}.read"), cells: vec![Cells { operator: o_read, rows: vec![r], cols: 0..cols_of(o_read) }] });
                    mine.push(groups.len() - 1);
                    groups.push(Group { name: format!("{name}.o.k{k}.r{r}.write"), cells: vec![Cells { operator: o_write, rows: (0..rows_of(o_write)).collect(), cols: r..r + 1 }] });
                    mine.push(groups.len() - 1);
                    r += 1;
                }
                layer.components.push(mine);
            }
        }
        layer.components.extend(per_component.into_values());
        // The attention's gates: direction rows, and the thresholds of each stage as one group.
        let mut thresholds = Vec::new();
        for prefix in ["attn", "o"] {
            if direction {
                if let Ok(g) = named(&format!("{name}.{prefix}.direction")) {
                    trainable.push(g);
                    for b in 0..rows_of(g) {
                        groups.push(Group { name: format!("{name}.{prefix}.g{b}"), cells: vec![Cells { operator: g, rows: vec![b], cols: 0..cols_of(g) }] });
                        thresholds.push(groups.len() - 1);
                    }
                }
            }
            if let Ok(t) = named(&format!("{name}.{prefix}.threshold")) {
                trainable.push(t);
                groups.push(Group { name: format!("{name}.{prefix}.thresholds"), cells: vec![Cells { operator: t, rows: (0..rows_of(t)).collect(), cols: 0..1 }] });
                thresholds.push(groups.len() - 1);
            }
        }
        layer.components.push(thresholds);
        // The MLP: per carrier of c_fc and down slices, its groups (none for an MLP with no
        // component, which adds zero).
        if named(&format!("{name}.mlp.zero")).is_ok() {
            continue;
        }
        let (fc_read, fc_write, dn_read, dn_write) = (named(&format!("{name}.mlp.fc_read"))?, named(&format!("{name}.mlp.fc_write"))?, named(&format!("{name}.mlp.dn_read"))?, named(&format!("{name}.mlp.dn_write"))?);
        trainable.extend([fc_read, fc_write, dn_read, dn_write]);
        let mut r = 0;
        for (b, group) in program.operators[fc_read].rows.clone().groups().iter().enumerate() {
            let mut mine = Vec::new();
            for _ in 0..group.width {
                groups.push(Group { name: format!("{name}.mlp.c{b}.fc{r}.read"), cells: vec![Cells { operator: fc_read, rows: vec![r], cols: 0..cols_of(fc_read) }] });
                mine.push(groups.len() - 1);
                groups.push(Group { name: format!("{name}.mlp.c{b}.fc{r}.write"), cells: vec![Cells { operator: fc_write, rows: (0..rows_of(fc_write)).collect(), cols: r..r + 1 }] });
                mine.push(groups.len() - 1);
                r += 1;
            }
            layer.functions.push(mine);
        }
        let mut r = 0;
        for (k, group) in program.operators[dn_read].rows.clone().groups().iter().enumerate() {
            let mut mine = Vec::new();
            for _ in 0..group.width {
                groups.push(Group { name: format!("{name}.mlp.k{k}.dn{r}.read"), cells: vec![Cells { operator: dn_read, rows: vec![r], cols: 0..cols_of(dn_read) }] });
                mine.push(groups.len() - 1);
                groups.push(Group { name: format!("{name}.mlp.k{k}.dn{r}.write"), cells: vec![Cells { operator: dn_write, rows: (0..rows_of(dn_write)).collect(), cols: r..r + 1 }] });
                mine.push(groups.len() - 1);
                r += 1;
            }
            layer.functions.push(mine);
        }
        for prefix in ["fc", "dn"] {
            if direction {
                if let Ok(g) = named(&format!("{name}.mlp.{prefix}.direction")) {
                    trainable.push(g);
                    for b in 0..rows_of(g) {
                        groups.push(Group { name: format!("{name}.mlp.{prefix}.g{b}"), cells: vec![Cells { operator: g, rows: vec![b], cols: 0..cols_of(g) }] });
                        layer.thresholds.push(groups.len() - 1);
                    }
                }
            }
            if let Ok(t) = named(&format!("{name}.mlp.{prefix}.threshold")) {
                trainable.push(t);
                groups.push(Group { name: format!("{name}.mlp.{prefix}.thresholds"), cells: vec![Cells { operator: t, rows: (0..rows_of(t)).collect(), cols: 0..1 }] });
                layer.thresholds.push(groups.len() - 1);
            }
        }
    }
    trainable.sort_unstable();
    trainable.dedup();
    let reference = mean_squares(&artifact.program, &groups);
    Ok(Explanation { artifact, trainable, groups, layers: out, removed: Vec::new(), fixed_nats: 0.0, reference, reads: Vec::new() })
}

/// The gate and softness nodes of a write-side stage from their parts (carried, then own): the part
/// itself when one, else their concatenation.
/// The uses of `M`'s maps in an explanation [`explanation`] built (`artifact::Owner::uses`), where
/// each map is a sum of gated slices no single operator holds: per layer, each head's q, k and v map
/// (the head rule's input into the node writing the head's projection), each head's block of the o
/// map (the heads' concatenated outputs into the o rule's output), the c_fc map (the MLP rule's input
/// into its pre-activations) and the down map (the activations into the MLP's output). A native
/// weight edit compiles through them (`weight_edit`). Empty for an artifact this module did not
/// build. `explanation` does not record them in the artifact, so the identities of its checkpoints
/// stay as they are; whoever compiles an edit adds them.
pub fn uses(native: &OperatorProgram, layers: &[LayerNodes], artifact: &Artifact) -> Result<Vec<Owner>, String> {
    let program = &artifact.program;
    if index_of(program, "library.l0.attn.read").is_err() {
        return Ok(Vec::new());
    }
    let rule = |name: &str| program.rules.iter().find(|r| r.name == name).ok_or_else(|| error(format!("no rule {name}")));
    let applying = |body: &Rule, op: &str| -> Result<usize, String> {
        let at = index_of(program, op)?;
        body.nodes.iter().position(|n| matches!(n, Node::Affine { terms, .. } if terms.iter().any(|t| t.1 == at))).ok_or_else(|| error(format!("{}: no node applies {op}", body.name)))
    };
    let map_of = |node: usize, input: usize| -> Result<&Operator, String> {
        match native.nodes.get(node) {
            Some(Node::Affine { terms, .. }) => terms.iter().find(|t| t.0 == input).map(|t| native.operators[t.1].as_ref()).ok_or_else(|| error(format!("native node {node} reads no map of node {input}"))),
            _ => Err(error(format!("native node {node} is not a map"))),
        }
    };
    let owner = |body: &Rule, w: &Operator, cols: std::ops::Range<usize>, role: &str, at: (usize, usize)| Owner {
        rows: 0..w.rows.width(),
        cols,
        body: body.name.clone(),
        site: body.name.clone(),
        native: w.name.clone(),
        native_rows: 0..w.rows.width(),
        native_cols: 0..w.cols.width(),
        role: role.to_string(),
        uses: Some(at),
        ..Owner::default()
    };
    let mut out = Vec::new();
    for (l, layer) in layers.iter().enumerate() {
        let name = format!("library.l{l}");
        for (h, &read) in layer.reads.iter().enumerate() {
            let Node::Attend { query, key, value, .. } = &native.nodes[read] else { return Err(error(format!("layer {l} head {h}: the read is not an attention node"))) };
            let body = rule(&format!("{name}.h{h}"))?;
            for (part, node) in ["q", "k", "v"].into_iter().zip([*query, *key, *value]) {
                let w = map_of(node, layer.normed_stream)?;
                out.push(owner(body, w, 0..w.cols.width(), part, (0, applying(body, &format!("{name}.h{h}.{part}"))?)));
            }
        }
        if let Ok(body) = rule(&format!("{name}.o")) {
            let concat = body.nodes.iter().position(|n| matches!(n, Node::Concat { .. })).ok_or_else(|| error(format!("{}: no concatenation of the heads", body.name)))?;
            let mut offset = 0;
            for &read in &layer.reads {
                let w = map_of(layer.attention, read)?;
                let width = w.cols.width();
                out.push(owner(body, w, offset..offset + width, "o", (concat, body.output)));
                offset += width;
            }
        }
        let body = rule(&format!("{name}.mlp"))?;
        // An MLP with no component adds zero: no slice of c_fc or down is P's.
        if !body.nodes.iter().any(|n| matches!(n, Node::Pointwise { .. })) {
            continue;
        }
        let Node::Pointwise { input: pre, .. } = native.nodes[layer.active] else { return Err(error(format!("layer {l}: the MLP activation is not one pointwise law"))) };
        let fc = map_of(pre, layer.normed)?;
        out.push(owner(body, fc, 0..fc.cols.width(), "gate", (0, applying(body, &format!("{name}.mlp.fc_write"))?)));
        let act = body.nodes.iter().position(|n| matches!(n, Node::Pointwise { .. })).ok_or_else(|| error(format!("{}: no activations", body.name)))?;
        let down = map_of(layer.mlp, layer.active)?;
        out.push(owner(body, down, 0..down.cols.width(), "out", (act, body.output)));
    }
    Ok(out)
}

fn joined(nodes: &mut Vec<Node>, gates: Vec<usize>, softs: Vec<usize>) -> (usize, usize) {
    if gates.len() == 1 {
        return (gates[0], softs[0]);
    }
    nodes.push(Node::Concat { parts: gates });
    nodes.push(Node::Concat { parts: softs });
    (nodes.len() - 2, nodes.len() - 1)
}

/// The interface of the concatenation of `parts`.
fn concat_interface(parts: &[Interface]) -> Result<Interface, String> {
    Interface::new(parts.iter().flat_map(|p| p.groups().to_vec()).collect()).map_err(error)
}

/// `M`'s operator of each site kind, in [`KINDS`] order, by its export name.
pub const EXPORT_NAMES: [&str; 6] = ["attn.q_proj", "attn.k_proj", "attn.v_proj", "attn.o_proj", "mlp.c_fc", "mlp.down_proj"];

/// One component of [`dump_parts`]: per operator of `M` its slices' writes and reads (as columns),
/// its gate, a direction gate's `g`, and its hard gate on every row.
struct Dumped {
    slices: BTreeMap<String, (Vec<Vec<f64>>, Vec<Vec<f64>>)>,
    gate: serde_json::Value,
    g: Option<Vec<f64>>,
    on: Vec<bool>,
}

/// A stage's gate of one component: on per row, `τ_b`, and a direction gate's `g`.
type StageGate = (Vec<bool>, f64, Option<Vec<f64>>);

/// Native node `node`'s value in `P`'s run `trace`.
fn node_value<'a>(artifact: &Artifact, trace: &'a Trace, node: usize) -> Result<&'a Array2<f64>, String> {
    let at = artifact.place(node).ok_or_else(|| error(format!("P holds no node {node}")))?;
    trace.values.get(at).ok_or_else(|| error(format!("no value of node {at}")))
}

fn push_slice(part: &mut Dumped, operator: String, write: Vec<f64>, read: Vec<f64>) {
    let entry = part.slices.entry(operator).or_default();
    entry.0.push(write);
    entry.1.push(read);
}

fn set_gate(part: &mut Dumped, read: String, (on, tau, g): StageGate) {
    let kind = if g.is_some() { "direction" } else { "own" };
    part.gate = serde_json::json!({"kind": kind, "read": read, "tau": tau});
    part.g = g;
    part.on = on;
}

/// One arm's components as parts, in the layout of the toy gate's harness
/// (`bench/toys_2951/score_toys.py`), at `artifact`'s values (a checkpoint's posterior mean, or
/// the start's): per component its slices' writes `U` and reads `V` on each of `M`'s operators it
/// spans (export names, `W = U Vᵀ`), its gate (own: `z = ‖V_bᵀx‖ − τ_b` over its reads at its
/// stage; direction: `z = g_bᵀx − τ_b`, the threshold's constant folded into `τ_b`), and its hard
/// gate `z > 0` on every row of `inputs` in `P`'s own run (`artifact` executed on the host; a down
/// gate on its own read reads the MLP's activations recomputed from the gated `c_fc` reads, as the
/// MLP's rule computes them). Written to `dir`: `parts.json`, each slice set's `U` (out × r) and
/// `V` (in × r) and each direction's `g` as float64, and `active.f64` (rows × components). Heads are
/// taken to own their k and v (no grouped-query sharing). Returns the number of components.
pub fn dump_parts(native: &OperatorProgram, layers: &[LayerNodes], artifact: &Artifact, start: &Path, arm: &str, inputs: &FamilyInputs, dir: &Path) -> Result<usize, String> {
    let records: Vec<ArmRecord> = serde_json::from_slice(&std::fs::read(start).map_err(|e| error(format!("{}: {e}", start.display())))?).map_err(error)?;
    let components = records.into_iter().find(|r| r.arm == arm).ok_or_else(|| error(format!("{}: no arm {arm}", start.display())))?.components;
    let read_site = |c: &Component| match &c.read {
        Read::Own([site, _]) => *site,
        Read::Direction { site, .. } => *site,
    };
    // Per layer and stage, the components gated there, as `explanation` orders them.
    let mut at: BTreeMap<(usize, usize), Vec<usize>> = BTreeMap::new();
    for (b, c) in components.iter().enumerate() {
        let site = read_site(c);
        at.entry((site / KINDS.len(), stage(site))).or_default().push(b);
    }
    let direction = components.iter().any(|c| matches!(c.read, Read::Direction { .. }));
    let trace = artifact.execute(inputs)?;
    let rows = inputs.rows;
    let value = |node: usize| node_value(artifact, &trace, node);
    let matrix = |name: &str| -> Result<Array2<f64>, String> { Ok(artifact.program.operators[index_of(&artifact.program, name)?].matrix()) };
    let interfaces = native.interfaces().map_err(error)?;
    // A stage's gates, per component of the stage (its rows of the stacked `read`, `widths` of
    // them in order), from the stage's input `x`.
    let gates = |x: &Array2<f64>, read: &Array2<f64>, widths: &[usize], prefix: &str| -> Result<Vec<StageGate>, String> {
        let threshold = matrix(&format!("{prefix}.threshold"))?;
        let mut out = Vec::with_capacity(widths.len());
        if direction {
            let g = matrix(&format!("{prefix}.direction"))?;
            for k in 0..widths.len() {
                let z = x.dot(&g.row(k)) + threshold[[k, 0]];
                out.push((z.iter().map(|v| *v > 0.0).collect(), -threshold[[k, 0]], Some(g.row(k).to_vec())));
            }
        } else {
            let a = x.dot(&read.t());
            let mut first = 0;
            for (k, &w) in widths.iter().enumerate() {
                let norms = a.slice(s![.., first..first + w]).map_axis(Axis(1), |r| r.dot(&r).sqrt());
                out.push((norms.iter().map(|n| n + threshold[[k, 0]] > 0.0).collect(), -threshold[[k, 0]], None));
                first += w;
            }
        }
        Ok(out)
    };
    let mut parts: Vec<Dumped> = components.iter().map(|_| Dumped { slices: BTreeMap::new(), gate: serde_json::Value::Null, g: None, on: vec![false; rows] }).collect();
    for (l, layer) in layers.iter().enumerate() {
        let name = format!("library.l{l}");
        let site = |kind: Kind| KINDS.len() * l + KINDS.iter().position(|k| *k == kind).unwrap_or(0);
        let slices_on = |b: usize, s: usize| components[b].slices.iter().filter(|[t, _]| *t == s).count();
        let export = |s: usize| format!("blocks.{l}.{}", EXPORT_NAMES[s % KINDS.len()]);
        // ---------------------------------------------------------------- the attention's input stage
        let a_comps = at.get(&(l, 0)).cloned().unwrap_or_default();
        let (q, k, v) = (site(Kind::Query), site(Kind::Key), site(Kind::Value));
        // Per stacked row its component and site, as `explanation` stacks them.
        let mut read_rows: Vec<(usize, usize)> = Vec::new();
        let mut widths = Vec::new();
        for &b in &a_comps {
            let mut w = 0;
            for s in [q, k, v] {
                for _ in 0..slices_on(b, s) {
                    read_rows.push((b, s));
                    w += 1;
                }
            }
            widths.push(w);
        }
        if !a_comps.is_empty() {
            let read = matrix(&format!("{name}.attn.read"))?;
            let heads = layer.reads.len();
            let writes: Vec<Vec<Array2<f64>>> = (0..heads).map(|h| ["q", "k", "v"].iter().map(|p| matrix(&format!("{name}.h{h}.{p}"))).collect()).collect::<Result<_, _>>()?;
            // A row's write is its column among its map's rows, in every head.
            let mut column = [0usize; 3];
            for (r, &(b, s)) in read_rows.iter().enumerate() {
                let j = [q, k, v].iter().position(|t| *t == s).unwrap_or(0);
                let c = column[j];
                column[j] += 1;
                let u: Vec<f64> = (0..heads).flat_map(|h| writes[h][j].column(c).to_vec()).collect();
                push_slice(&mut parts[b], export(s), u, read.row(r).to_vec());
            }
            for (i, g) in gates(value(layer.normed_stream)?, &read, &widths, &format!("{name}.attn"))?.into_iter().enumerate() {
                let b = a_comps[i];
                let first = read_rows.iter().find(|(c, _)| *c == b).map_or(q, |(_, s)| *s);
                set_gate(&mut parts[b], export(first), g);
            }
        }
        // ---------------------------------------------------------------- the attention's output
        let o = site(Kind::Output);
        let o_own = at.get(&(l, 1)).cloned().unwrap_or_default();
        let o_carriers: Vec<usize> = a_comps.iter().copied().filter(|&b| slices_on(b, o) > 0).chain(o_own.iter().copied()).collect();
        if !o_carriers.is_empty() {
            let (read, write) = (matrix(&format!("{name}.o.read"))?, matrix(&format!("{name}.o.write"))?);
            let (mut r, mut own_rows, mut own_widths) = (0, Vec::new(), Vec::new());
            for &b in &o_carriers {
                let n = slices_on(b, o);
                for _ in 0..n {
                    push_slice(&mut parts[b], export(o), write.column(r).to_vec(), read.row(r).to_vec());
                    if o_own.contains(&b) {
                        own_rows.push(r);
                    }
                    r += 1;
                }
                if o_own.contains(&b) {
                    own_widths.push(n);
                }
            }
            if !o_own.is_empty() {
                // The o gates on their own read read every head's output, concatenated.
                let heads: Vec<&Array2<f64>> = layer.reads.iter().map(|&n| value(n)).collect::<Result<_, _>>()?;
                let views: Vec<_> = heads.iter().map(|h| h.view()).collect();
                let concat = ndarray::concatenate(Axis(1), &views).map_err(error)?;
                for (i, g) in gates(&concat, &read.select(Axis(0), &own_rows), &own_widths, &format!("{name}.o"))?.into_iter().enumerate() {
                    set_gate(&mut parts[o_own[i]], export(o), g);
                }
            }
        }
        // ---------------------------------------------------------------- the MLP
        let (fc, dn) = (site(Kind::Up), site(Kind::Down));
        let f_comps = at.get(&(l, 2)).cloned().unwrap_or_default();
        let d_own = at.get(&(l, 3)).cloned().unwrap_or_default();
        if f_comps.is_empty() {
            continue;
        }
        let (fc_read, fc_write) = (matrix(&format!("{name}.mlp.fc_read"))?, matrix(&format!("{name}.mlp.fc_write"))?);
        let (dn_read, dn_write) = (matrix(&format!("{name}.mlp.dn_read"))?, matrix(&format!("{name}.mlp.dn_write"))?);
        let (mut r, mut f_widths) = (0, Vec::new());
        for &b in &f_comps {
            let n = slices_on(b, fc);
            for _ in 0..n {
                push_slice(&mut parts[b], export(fc), fc_write.column(r).to_vec(), fc_read.row(r).to_vec());
                r += 1;
            }
            f_widths.push(n);
        }
        let h2 = value(layer.normed)?;
        let f_gates = gates(h2, &fc_read, &f_widths, &format!("{name}.mlp.fc"))?;
        let dn_carriers: Vec<usize> = f_comps.iter().copied().filter(|&b| slices_on(b, dn) > 0).chain(d_own.iter().copied()).collect();
        let (mut r, mut own_rows, mut own_widths) = (0, Vec::new(), Vec::new());
        for &b in &dn_carriers {
            let n = slices_on(b, dn);
            for _ in 0..n {
                push_slice(&mut parts[b], export(dn), dn_write.column(r).to_vec(), dn_read.row(r).to_vec());
                if d_own.contains(&b) {
                    own_rows.push(r);
                }
                r += 1;
            }
            if d_own.contains(&b) {
                own_widths.push(n);
            }
        }
        if !d_own.is_empty() {
            // The activations in P's run: the c_fc reads gated per component, written, through the
            // MLP's law.
            let mut gated = h2.dot(&fc_read.t());
            let mut first = 0;
            for (k, &w) in f_widths.iter().enumerate() {
                for (row, on) in f_gates[k].0.iter().enumerate() {
                    if !on {
                        gated.slice_mut(s![row, first..first + w]).fill(0.0);
                    }
                }
                first += w;
            }
            let mut act = gated.dot(&fc_write.t());
            let Node::Pointwise { input: pre, laws } = &native.nodes[layer.active] else {
                return Err(error(format!("layer {l}: the MLP activation is not one pointwise law")));
            };
            for (group, law) in laws.iter().enumerate() {
                let law = *law;
                act.slice_mut(s![.., interfaces[*pre].range(group)]).mapv_inplace(|t| law.apply(t));
            }
            for (i, g) in gates(&act, &dn_read.select(Axis(0), &own_rows), &own_widths, &format!("{name}.mlp.dn"))?.into_iter().enumerate() {
                set_gate(&mut parts[d_own[i]], export(dn), g);
            }
        }
        for (i, g) in f_gates.into_iter().enumerate() {
            set_gate(&mut parts[f_comps[i]], export(fc), g);
        }
    }
    std::fs::create_dir_all(dir).map_err(error)?;
    let write = |file: &str, values: &[f64]| -> Result<(), String> {
        let bytes: Vec<u8> = values.iter().flat_map(|v| v.to_le_bytes()).collect();
        std::fs::write(dir.join(file), bytes).map_err(error)
    };
    let mut records = Vec::with_capacity(parts.len());
    for (b, part) in parts.iter().enumerate() {
        let mut slices = serde_json::Map::new();
        for (op, (us, vs)) in &part.slices {
            // U as out × r and V as in × r, row-major.
            let (out, input) = (us[0].len(), vs[0].len());
            let u: Vec<f64> = (0..out).flat_map(|i| us.iter().map(move |c| c[i])).collect();
            let v: Vec<f64> = (0..input).flat_map(|i| vs.iter().map(move |c| c[i])).collect();
            write(&format!("p{b}.{op}.U.f64"), &u)?;
            write(&format!("p{b}.{op}.V.f64"), &v)?;
            slices.insert(op.clone(), serde_json::json!({"U": format!("p{b}.{op}.U.f64"), "V": format!("p{b}.{op}.V.f64"), "rank": us.len()}));
        }
        let mut gate = part.gate.clone();
        if let Some(g) = &part.g {
            write(&format!("p{b}.g.f64"), g)?;
            gate["g"] = serde_json::json!(format!("p{b}.g.f64"));
        }
        records.push(serde_json::json!({"name": format!("component {b}"), "slices": slices, "gate": gate}));
    }
    let active: Vec<f64> = (0..rows).flat_map(|row| parts.iter().map(move |p| if p.on[row] { 1.0 } else { 0.0 })).collect();
    write("active.f64", &active)?;
    let record = serde_json::json!({"parts": records, "active": "active.f64", "kept": ["wte", "lm_head"], "fitter": format!("library_vpd, arm {arm}")});
    std::fs::write(dir.join("parts.json"), serde_json::to_vec_pretty(&record).map_err(error)?).map_err(error)?;
    Ok(parts.len())
}
