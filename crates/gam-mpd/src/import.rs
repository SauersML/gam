//! Import a model export (`export.json` plus raw little-endian float64 `<name>.f64` tensors, the
//! layout `bench/mpd_engine_export_2951.py` writes) into an operator program and its contract.
//!
//! Supported kinds, each read from the export's `config` and `forward` conventions:
//!
//! * `transformer`: `x_0[j] = W_E[tok_j] + W_pos[j]`; per layer, per head (row blocks of
//!   `blocks.l.W_Q/W_K/W_V`, column blocks of `blocks.l.W_O`), `q = W_Q_h x + b_Q_h`, likewise `k`,
//!   `v`, `A = softmax_j(q_i·k_j / √d_head)` over keys `j ≤ i` when causal, `x ← x + Σ_h W_O_h Σ_j
//!   A_ij v_j + b_O`; then, when the layer has `W_in`, `x ← x + W_out relu(W_in x + b_in) + b_out`;
//!   logits `W_U x + b_U` at each declared readout position. Positions are unrolled; each position's
//!   keys and values are computed once per layer and read by every query.
//! * `residual_mlp`: `x_0 = bits W_E + b_E`, residual ReLU blocks, logits `W_U x + b_U`; each output
//!   logit is an independent Bernoulli variable, read as the two-class distribution `(0, ℓ)`.
//! * `rnn`: `h_t = relu(W_ih x_t + W_hh h_{t−1})`, `h_0 = 0`, logits `W_U h_T`; one shared
//!   `W_ih` and `W_hh` at every step.
//!
//! A bias with no file is zero and gets no operator. Every tensor enters as its own operator at the
//! finest lattice that holds it exactly; tied uses reference one operator.
//!
//! The contract's family is the export's samples (`inputs.f64`), one unit per sample, each read at
//! every readout position. It is complete when the samples are every input of a finite token or bit
//! domain, each once; otherwise it is a sample, with its population bounds stated at the conventional
//! 0.95 confidence (a reporting level: nothing is selected by it).

use crate::contract::{Contract, FamilyKind};
use crate::operator_program::{
    Basis, Declarations, Domain, FamilyInputs, Interface, LabelKind, Law, Node, Operator, OperatorProgram, Provenance,
    Rotary, Scale, SequenceLayout, Slot, SlotValues, exact_precision,
};
use ndarray::{Array2, Axis, s};
use serde_json::Value;
use std::collections::BTreeSet;
use std::path::Path;
use std::sync::Arc;

/// An imported model: its native program and the contract of its declared samples.
pub struct Imported {
    pub name: String,
    pub kind: String,
    pub program: OperatorProgram,
    pub contract: Contract,
    pub record: Value,
}

fn read_f64(path: &Path, rows: usize, cols: usize) -> Result<Array2<f64>, String> {
    let bytes = std::fs::read(path).map_err(|error| format!("{}: {error}", path.display()))?;
    if bytes.len() != rows * cols * 8 {
        return Err(format!("{}: {} bytes for {rows}×{cols}", path.display(), bytes.len()));
    }
    let values: Vec<f64> = bytes
        .chunks_exact(8)
        .map(|c| f64::from_le_bytes([c[0], c[1], c[2], c[3], c[4], c[5], c[6], c[7]]))
        .collect();
    Array2::from_shape_vec((rows, cols), values).map_err(|error| error.to_string())
}

fn shape_of(value: &Value) -> Result<(usize, usize), String> {
    let dims: Vec<usize> = value
        .as_array()
        .ok_or("a shape is not a list")?
        .iter()
        .map(|v| v.as_u64().map(|v| v as usize).ok_or("a shape entry is not an integer"))
        .collect::<Result<_, _>>()?;
    match dims[..] {
        [n] => Ok((1, n)),
        [r, c] => Ok((r, c)),
        _ => Err(format!("shape {dims:?} is not one or two axes")),
    }
}

struct Tensors<'a> {
    dir: &'a Path,
    record: &'a Value,
}

impl Tensors<'_> {
    fn has(&self, name: &str) -> bool {
        self.record["files"].get(name).is_some()
    }

    fn get(&self, name: &str) -> Result<Array2<f64>, String> {
        let (rows, cols) = shape_of(&self.record["files"][name]["shape"]).map_err(|e| format!("{name}: {e}"))?;
        read_f64(&self.dir.join(format!("{name}.f64")), rows, cols)
    }

    /// A stored row vector as a column.
    fn column(&self, name: &str) -> Result<Array2<f64>, String> {
        Ok(self.get(name)?.t().to_owned())
    }
}

fn config(record: &Value, key: &str) -> Result<usize, String> {
    record["config"][key].as_u64().map(|v| v as usize).ok_or_else(|| format!("config.{key}"))
}

/// Builds a program node by node.
struct Builder {
    operators: Vec<Arc<Operator>>,
    nodes: Vec<Node>,
}

impl Builder {
    fn operator(&mut self, name: &str, rows: &Interface, cols: &Interface, values: Array2<f64>) -> Result<usize, String> {
        let precision = exact_precision(values.iter().copied()).map_err(|e| e.to_string())?;
        let op = Operator::dense(name, rows.clone(), cols.clone(), values, precision, Provenance::native(name))
            .map_err(|e| e.to_string())?;
        self.operators.push(Arc::new(op));
        Ok(self.operators.len() - 1)
    }

    fn node(&mut self, node: Node) -> usize {
        self.nodes.push(node);
        self.nodes.len() - 1
    }
}

fn interface(count: usize, width: usize, kind: LabelKind) -> Result<Interface, String> {
    Interface::uniform(count, width, kind, 0).map_err(|e| e.to_string())
}

/// Whether the export in `dir` is a rotary language model (its config names `rope_theta`).
pub fn is_language_model(dir: &Path) -> Result<bool, String> {
    let text = std::fs::read_to_string(dir.join("export.json")).map_err(|e| e.to_string())?;
    let record: Value = serde_json::from_str(&text).map_err(|e| e.to_string())?;
    Ok(record["config"].get("rope_theta").is_some())
}

/// The export in `dir` as a program and contract.
pub fn import(dir: &Path) -> Result<Imported, String> {
    let text = std::fs::read_to_string(dir.join("export.json")).map_err(|e| e.to_string())?;
    let record: Value = serde_json::from_str(&text).map_err(|e| e.to_string())?;
    if record["config"].get("rope_theta").is_some() {
        return Err("a rotary language model export: use import_language_model".to_string());
    }
    let kind = record["kind"].as_str().ok_or("export.json: kind")?.to_string();
    let name = record["model"].as_str().unwrap_or("model").to_string();
    let (sample_rows, sample_cols) = shape_of(&record["samples"]["shape"])?;
    let samples = read_f64(&dir.join(record["samples"]["file"].as_str().unwrap_or("inputs.f64")), sample_rows, sample_cols)?;
    let tensors = Tensors { dir, record: &record };
    let (program, slots, complete, readout_slots) = match kind.as_str() {
        "transformer" => transformer(&tensors, &record, &samples)?,
        "residual_mlp" => residual_mlp(&tensors, &record, &samples)?,
        "rnn" => rnn(&tensors, &record, &samples)?,
        other => return Err(format!("unsupported kind {other}")),
    };
    let family = FamilyInputs { rows: sample_rows, slots, layout: None };
    let kind_of = if complete {
        FamilyKind::Complete { description: format!("every input of {name}'s domain") }
    } else {
        FamilyKind::Sample {
            population: record["input"]["generator"].as_str().unwrap_or("the export's generator").to_string(),
            confidence: 0.95,
            units: (0..sample_rows).collect(),
        }
    };
    let readouts = program.1;
    let contract = Contract {
        declarations: program.0.declarations.clone(),
        family,
        kind: kind_of,
        observations: 1,
        readouts,
        readout_slots,
    };
    Ok(Imported { name, kind, program: program.0, contract, record })
}

/// Whether the rows of `samples` are every point of `domain^cols`, each once.
fn enumerates(samples: &Array2<f64>, domain: usize) -> bool {
    let cols = samples.ncols() as u32;
    let total = (domain as u128).checked_pow(cols);
    if total != Some(samples.nrows() as u128) {
        return false;
    }
    let distinct: BTreeSet<Vec<u64>> = samples.outer_iter().map(|row| row.iter().map(|v| *v as u64).collect()).collect();
    distinct.len() == samples.nrows()
}

type Built = ((OperatorProgram, usize), Vec<SlotValues>, bool, Option<Vec<Vec<usize>>>);

fn transformer(tensors: &Tensors<'_>, record: &Value, samples: &Array2<f64>) -> Result<Built, String> {
    let (layers, heads, d, dh) = (config(record, "n_layers")?, config(record, "n_heads")?, config(record, "d_model")?, config(record, "d_head")?);
    let vocab = record["input"]["vocab_size"].as_u64().ok_or("input.vocab_size")? as usize;
    let classes = record["output"]["n_classes"].as_u64().ok_or("output.n_classes")? as usize;
    let causal = record["config"]["causal"].as_bool().unwrap_or(true);
    let positions = samples.ncols();
    let readout_positions: Vec<usize> = record["output"]["readout_positions"]
        .as_array()
        .ok_or("output.readout_positions")?
        .iter()
        .map(|v| v.as_u64().map(|v| v as usize).ok_or("a readout position"))
        .collect::<Result<_, _>>()?;
    let tokens = interface(vocab, 1, LabelKind::Token)?;
    let model = Interface::native(d).map_err(|e| e.to_string())?;
    let head = Interface::native(dh).map_err(|e| e.to_string())?;
    let class_interface = interface(classes, 1, LabelKind::Token)?;
    let constant = Interface::constant();
    let mut b = Builder { operators: Vec::new(), nodes: Vec::new() };
    let w_e = b.operator("W_E", &model, &tokens, tensors.get("W_E")?.t().to_owned())?;
    let w_pos = tensors.get("W_pos")?;
    let identity = {
        b.operators.push(Arc::new(Operator::identity("I", model.clone())));
        b.operators.len() - 1
    };
    let mut x: Vec<usize> = Vec::with_capacity(positions);
    for j in 0..positions {
        let feature = b.node(Node::Feature { slot: j, basis: 0 });
        let pos = b.operator(&format!("W_pos{j}"), &model, &constant, w_pos.row(j).to_owned().insert_axis(Axis(1)))?;
        x.push(b.node(Node::Affine { terms: vec![(feature, w_e)], bias: Some(pos) }));
    }
    for l in 0..layers {
        let prefix = format!("blocks.{l}.");
        let bias = |b: &mut Builder, name: &str, rows: &Interface, range: Option<(usize, usize)>| -> Result<Option<usize>, String> {
            let full = format!("{prefix}{name}");
            if !tensors.has(&full) {
                return Ok(None);
            }
            let column = tensors.column(&full)?;
            let column = match range {
                Some((a, e)) => column.slice(s![a..e, ..]).to_owned(),
                None => column,
            };
            Ok(Some(b.operator(&format!("{full}[{}]", range.map_or(0, |r| r.0 / dh)), rows, &constant, column)?))
        };
        let (w_q, w_k, w_v, w_o) = (
            tensors.get(&format!("{prefix}W_Q"))?,
            tensors.get(&format!("{prefix}W_K"))?,
            tensors.get(&format!("{prefix}W_V"))?,
            tensors.get(&format!("{prefix}W_O"))?,
        );
        let mut mixes: Vec<Vec<(usize, usize)>> = vec![Vec::new(); positions];
        for h in 0..heads {
            let rows = (h * dh, (h + 1) * dh);
            let q_op = b.operator(&format!("{prefix}W_Q{h}"), &head, &model, w_q.slice(s![rows.0..rows.1, ..]).to_owned())?;
            let k_op = b.operator(&format!("{prefix}W_K{h}"), &head, &model, w_k.slice(s![rows.0..rows.1, ..]).to_owned())?;
            let v_op = b.operator(&format!("{prefix}W_V{h}"), &head, &model, w_v.slice(s![rows.0..rows.1, ..]).to_owned())?;
            let o_op = b.operator(&format!("{prefix}W_O{h}"), &model, &head, w_o.slice(s![.., rows.0..rows.1]).to_owned())?;
            let (bq, bk, bv) = (
                bias(&mut b, "b_Q", &head, Some(rows))?,
                bias(&mut b, "b_K", &head, Some(rows))?,
                bias(&mut b, "b_V", &head, Some(rows))?,
            );
            let keys: Vec<usize> = x.iter().map(|&xj| b.node(Node::Affine { terms: vec![(xj, k_op)], bias: bk })).collect();
            let values: Vec<usize> = x.iter().map(|&xj| b.node(Node::Affine { terms: vec![(xj, v_op)], bias: bv })).collect();
            for i in 0..positions {
                let q = b.node(Node::Affine { terms: vec![(x[i], q_op)], bias: bq });
                let visible: Vec<usize> = (0..positions).filter(|&j| !causal || j <= i).collect();
                let scores: Vec<usize> = visible
                    .iter()
                    .map(|&j| b.node(Node::Bilinear { left: q, right: keys[j], scale: Scale::InverseSqrt(dh as u32) }))
                    .collect();
                let weights = b.node(Node::Softmax { scores });
                let payloads: Vec<(usize, usize)> = visible.iter().enumerate().map(|(c, &j)| (c, values[j])).collect();
                let mix = b.node(Node::Mix { weights, payloads });
                mixes[i].push((mix, o_op));
            }
        }
        let b_o = bias(&mut b, "b_O", &model, None)?;
        for i in 0..positions {
            let mut terms = vec![(x[i], identity)];
            terms.extend(mixes[i].iter().copied());
            x[i] = b.node(Node::Affine { terms, bias: b_o });
        }
        if tensors.has(&format!("{prefix}W_in")) {
            let w_in = tensors.get(&format!("{prefix}W_in"))?;
            let hidden = w_in.nrows();
            let units = interface(hidden, 1, LabelKind::Unit)?;
            let w_in = b.operator(&format!("{prefix}W_in"), &units, &model, w_in)?;
            let w_out = b.operator(&format!("{prefix}W_out"), &model, &units, tensors.get(&format!("{prefix}W_out"))?)?;
            let (b_in, b_out) = (bias(&mut b, "b_in", &units, None)?, bias(&mut b, "b_out", &model, None)?);
            for xi in x.iter_mut() {
                let pre = b.node(Node::Affine { terms: vec![(*xi, w_in)], bias: b_in });
                let act = b.node(Node::Pointwise { input: pre, laws: vec![Law::Relu; hidden] });
                *xi = b.node(Node::Affine { terms: vec![(*xi, identity), (act, w_out)], bias: b_out });
            }
        }
    }
    let w_u = b.operator("W_U", &class_interface, &model, tensors.get("W_U")?)?;
    let b_u = if tensors.has("b_U") { Some(b.operator("b_U", &class_interface, &constant, tensors.column("b_U")?)?) } else { None };
    let mut readouts = Vec::new();
    for &position in &readout_positions {
        let logits = b.node(Node::Affine { terms: vec![(x[position], w_u)], bias: b_u });
        readouts.push(b.node(Node::Readout { input: logits, basis: 1 }));
    }
    let output = if readouts.len() == 1 { readouts[0] } else { b.node(Node::Concat { parts: readouts.clone() }) };
    let declarations = Declarations { parameters: 0,
        domains: vec![Domain { size: vocab, cycle: None }, Domain { size: classes, cycle: None }],
        slots: vec![Slot::Token { domain: 0 }; positions],
    };
    let mut program = OperatorProgram { rules: Vec::new(),
        declarations,
        bases: vec![Basis::Indicator { domain: 0 }, Basis::Indicator { domain: 1 }],
        operators: b.operators,
        nodes: b.nodes,
        output,
    };
    program.prune();
    let slots = (0..positions).map(|j| SlotValues::Tokens(samples.column(j).iter().map(|v| *v as u32).collect())).collect();
    // A causal model's readout at position i may read positions 0..=i only.
    let readout_slots = causal.then(|| readout_positions.iter().map(|&i| (0..=i).collect()).collect());
    Ok(((program, readout_positions.len()), slots, enumerates(samples, vocab), readout_slots))
}

fn residual_mlp(tensors: &Tensors<'_>, record: &Value, samples: &Array2<f64>) -> Result<Built, String> {
    let (layers, d, bits) = (config(record, "n_layers")?, config(record, "d_model")?, config(record, "n_inputs")?);
    let outputs = record["output"]["n_outputs"].as_u64().ok_or("output.n_outputs")? as usize;
    let model = Interface::native(d).map_err(|e| e.to_string())?;
    let input = Interface::native(bits).map_err(|e| e.to_string())?;
    let constant = Interface::constant();
    let mut b = Builder { operators: Vec::new(), nodes: Vec::new() };
    b.operators.push(Arc::new(Operator::identity("I", model.clone())));
    let identity = 0;
    let w_e = b.operator("W_E", &model, &input, tensors.get("W_E")?.t().to_owned())?;
    let b_e = if tensors.has("b_E") { Some(b.operator("b_E", &model, &constant, tensors.column("b_E")?)?) } else { None };
    let raw = b.node(Node::Raw { slot: 0 });
    let mut x = b.node(Node::Affine { terms: vec![(raw, w_e)], bias: b_e });
    for l in 0..layers {
        let prefix = format!("blocks.{l}.");
        let w_in = tensors.get(&format!("{prefix}W_in"))?;
        let hidden = w_in.nrows();
        let units = interface(hidden, 1, LabelKind::Unit)?;
        let w_in = b.operator(&format!("{prefix}W_in"), &units, &model, w_in)?;
        let w_out = b.operator(&format!("{prefix}W_out"), &model, &units, tensors.get(&format!("{prefix}W_out"))?)?;
        let b_in = if tensors.has(&format!("{prefix}b_in")) {
            Some(b.operator(&format!("{prefix}b_in"), &units, &constant, tensors.column(&format!("{prefix}b_in"))?)?)
        } else {
            None
        };
        let b_out = if tensors.has(&format!("{prefix}b_out")) {
            Some(b.operator(&format!("{prefix}b_out"), &model, &constant, tensors.column(&format!("{prefix}b_out"))?)?)
        } else {
            None
        };
        let pre = b.node(Node::Affine { terms: vec![(x, w_in)], bias: b_in });
        let act = b.node(Node::Pointwise { input: pre, laws: vec![Law::Relu; hidden] });
        x = b.node(Node::Affine { terms: vec![(x, identity), (act, w_out)], bias: b_out });
    }
    // Each output logit ℓ_c is a Bernoulli variable, read as the two-class logits (0, ℓ_c): the
    // readout rows (c, 0) are absent blocks, so they cost nothing and execute as zero.
    let pairs = interface(2 * outputs, 1, LabelKind::Token)?;
    let w_u = tensors.get("W_U")?;
    let mut values = Array2::<f64>::zeros((2 * outputs, d));
    let mut present = Array2::from_elem((2 * outputs, 1), false);
    for c in 0..outputs {
        values.row_mut(2 * c + 1).assign(&w_u.row(c));
        present[[2 * c + 1, 0]] = true;
    }
    let precision = exact_precision(values.iter().copied()).map_err(|e| e.to_string())?;
    b.operators.push(Arc::new(
        Operator::blocks("W_U", pairs.clone(), model.clone(), values, present, precision, Provenance::native("W_U"))
            .map_err(|e| e.to_string())?,
    ));
    let w_u_op = b.operators.len() - 1;
    let b_u = if tensors.has("b_U") {
        let bias = tensors.column("b_U")?;
        let mut values = Array2::<f64>::zeros((2 * outputs, 1));
        let mut present = Array2::from_elem((2 * outputs, 1), false);
        for c in 0..outputs {
            values[[2 * c + 1, 0]] = bias[[c, 0]];
            present[[2 * c + 1, 0]] = true;
        }
        let precision = exact_precision(values.iter().copied()).map_err(|e| e.to_string())?;
        b.operators.push(Arc::new(
            Operator::blocks("b_U", pairs.clone(), constant, values, present, precision, Provenance::native("b_U"))
                .map_err(|e| e.to_string())?,
        ));
        Some(b.operators.len() - 1)
    } else {
        None
    };
    let logits = b.node(Node::Affine { terms: vec![(x, w_u_op)], bias: b_u });
    let output = b.node(Node::Readout { input: logits, basis: 0 });
    let declarations = Declarations { parameters: 0,
        domains: vec![Domain { size: 2 * outputs, cycle: None }],
        slots: vec![Slot::Raw { width: bits }],
    };
    let mut program = OperatorProgram { rules: Vec::new(),
        declarations,
        bases: vec![Basis::Indicator { domain: 0 }],
        operators: b.operators,
        nodes: b.nodes,
        output,
    };
    program.prune();
    Ok(((program, outputs), vec![SlotValues::Raw(samples.clone())], enumerates(samples, 2), None))
}

fn rnn(tensors: &Tensors<'_>, record: &Value, samples: &Array2<f64>) -> Result<Built, String> {
    let (hidden, steps) = (config(record, "hidden")?, config(record, "seq_len")?);
    let classes = record["output"]["n_classes"].as_u64().ok_or("output.n_classes")? as usize;
    let state = interface(hidden, 1, LabelKind::Unit)?;
    let scalar = Interface::native(1).map_err(|e| e.to_string())?;
    let class_interface = interface(classes, 1, LabelKind::Token)?;
    let mut b = Builder { operators: Vec::new(), nodes: Vec::new() };
    let w_ih = b.operator("W_ih", &state, &scalar, tensors.get("W_ih")?)?;
    let w_hh = b.operator("W_hh", &state, &state, tensors.get("W_hh")?)?;
    let w_u = b.operator("W_U", &class_interface, &state, tensors.get("W_U")?)?;
    let mut h: Option<usize> = None;
    for t in 0..steps {
        let input = b.node(Node::Raw { slot: t });
        let mut terms = vec![(input, w_ih)];
        if let Some(previous) = h {
            terms.push((previous, w_hh));
        }
        let pre = b.node(Node::Affine { terms, bias: None });
        h = Some(b.node(Node::Pointwise { input: pre, laws: vec![Law::Relu; hidden] }));
    }
    let last = h.ok_or("an rnn with no steps")?;
    let logits = b.node(Node::Affine { terms: vec![(last, w_u)], bias: None });
    let output = b.node(Node::Readout { input: logits, basis: 0 });
    let declarations = Declarations { parameters: 0,
        domains: vec![Domain { size: classes, cycle: None }],
        slots: vec![Slot::Raw { width: 1 }; steps],
    };
    let program = OperatorProgram { rules: Vec::new(),
        declarations,
        bases: vec![Basis::Indicator { domain: 0 }],
        operators: b.operators,
        nodes: b.nodes,
        output,
    };
    let slots = (0..steps).map(|t| SlotValues::Raw(samples.slice(s![.., t..t + 1]).to_owned())).collect();
    Ok(((program, 1), slots, false, Some(vec![(0..steps).collect()])))
}

/// A pre-norm rotary language model export as a per-position program over the first `sequences`
/// token rows at positions `0..context`: every row runs one shared local program, and attention
/// reads the rows of its own sequence up to its position. The readout at each row is the
/// next-token distribution. Each sequence is one unit of a sampled family.
///
/// `config`: d_model, n_layers, n_heads, n_kv_heads, head_dim, vocab, rope_theta, rope_pairing,
/// norm_eps, mlp_act, tied_embeddings, and optionally `rotary_dims` (a partial rotary on the first
/// dims of each head; default head_dim), `norm` (`"rms"`, the default, or `"layer"`: the mean is
/// removed first, `x − 1 (1ᵀx)/d`, a rank-one operator shared by every norm), `parallel_residual`
/// (attention and MLP read the same stream, GPT-NeoX), `qk_norm` (an RMS norm with a gain on each
/// head's query and key before the rotary, Qwen3) and `mlp_gated` (`down(act(gate x) ⊙ up x)`).
/// Tensors: `wte` (vocab × d), `lm_head` when untied, `blocks.{l}.attn.{q,k,v,o}_proj`,
/// `blocks.{l}.attn.{q,k}_norm.gain`, `blocks.{l}.mlp.{c_fc,gate_proj,down_proj}`,
/// `blocks.{l}.rms{1,2}.gain`, `final_norm.gain`; any of these with a `.bias` suffix (a
/// projection's or a layer norm's bias) enters as a bias operator; and the token rows `tokens`.
pub fn import_language_model(dir: &Path, sequences: usize, context: usize) -> Result<Imported, String> {
    let text = std::fs::read_to_string(dir.join("export.json")).map_err(|e| e.to_string())?;
    let record: Value = serde_json::from_str(&text).map_err(|e| e.to_string())?;
    let tensors = Tensors { dir, record: &record };
    let (layers, heads, kv_heads, hd, d, vocab) = (
        config(&record, "n_layers")?,
        config(&record, "n_heads")?,
        config(&record, "n_kv_heads")?,
        config(&record, "head_dim")?,
        config(&record, "d_model")?,
        config(&record, "vocab")?,
    );
    let flag = |key: &str| record["config"][key].as_bool().unwrap_or(false);
    let (parallel, qk_norm, gated) = (flag("parallel_residual"), flag("qk_norm"), flag("mlp_gated"));
    let layer_norm = match record["config"]["norm"].as_str() {
        None | Some("rms") => false,
        Some("layer") => true,
        Some(other) => return Err(format!("unsupported norm {other}")),
    };
    let tied = record["config"]["tied_embeddings"].as_bool().unwrap_or(true);
    let rotary_dims = record["config"]["rotary_dims"].as_u64().map_or(hd, |v| v as usize);
    let theta = record["config"]["rope_theta"].as_f64().ok_or("config.rope_theta")?;
    let epsilon = record["config"]["norm_eps"].as_f64().ok_or("config.norm_eps")?;
    let half_split = record["config"]["rope_pairing"].as_str() == Some("rotate_half");
    let act = match record["config"]["mlp_act"].as_str() {
        Some("gelu_tanh") => Law::GeluTanh,
        Some("gelu") => Law::Gelu,
        Some("silu") => Law::Silu,
        Some("relu") => Law::Relu,
        other => return Err(format!("unsupported mlp activation {other:?}")),
    };
    if theta.fract() != 0.0 || theta <= 0.0 || theta > f64::from(u32::MAX) {
        return Err(format!("rope_theta {theta} is not a positive integer"));
    }
    let rotary = Rotary { base: theta as u32, dims: rotary_dims as u32, half_split };
    // The residual stream is in coordinate groups, so a norm gain is a diagonal of present blocks.
    let model = interface(d, 1, LabelKind::Unit)?;
    let head = Interface::native(hd).map_err(|e| e.to_string())?;
    let constant = Interface::constant();
    let tokens_interface = interface(vocab, 1, LabelKind::Token)?;
    let coordinates = model.clone();
    let mut b = Builder { operators: Vec::new(), nodes: Vec::new() };
    b.operators.push(Arc::new(Operator::identity("I", model.clone())));
    let identity = 0;
    let embedding = b.operator("wte", &model, &tokens_interface, tensors.get("wte")?.t().to_owned())?;
    // A norm gain is a diagonal operator: one present block per coordinate, d reals.
    let gain = |b: &mut Builder, name: &str| -> Result<usize, String> {
        let g = tensors.get(name)?;
        let mut values = Array2::<f64>::zeros((d, d));
        let mut present = Array2::from_elem((d, d), false);
        for i in 0..d {
            values[[i, i]] = g[[0, i]];
            present[[i, i]] = true;
        }
        let precision = exact_precision(g.iter().copied()).map_err(|e| e.to_string())?;
        let op = Operator::blocks(name, coordinates.clone(), coordinates.clone(), values, present, precision, Provenance::native(name))
            .map_err(|e| e.to_string())?;
        b.operators.push(Arc::new(op));
        Ok(b.operators.len() - 1)
    };
    // A stored bias (a row vector, sliced to `range`) as a column operator on `rows`.
    let bias = |b: &mut Builder, name: &str, rows: &Interface, range: Option<(usize, usize)>| -> Result<Option<usize>, String> {
        if !tensors.has(name) {
            return Ok(None);
        }
        let column = tensors.column(name)?;
        let column = match range {
            Some((first, last)) => column.slice(s![first..last, ..]).to_owned(),
            None => column,
        };
        let label = range.map_or(name.to_string(), |(first, _)| format!("{name}[{}]", first / hd));
        Ok(Some(b.operator(&label, rows, &constant, column)?))
    };
    // A layer norm removes the mean first: `x − 1 (1ᵀx)/d`, one rank-one operator for every norm.
    let centring = if layer_norm {
        let precision = exact_precision([-1.0 / d as f64, 1.0]).map_err(|e| e.to_string())?;
        let op = Operator::low_rank(
            "centre",
            model.clone(),
            model.clone(),
            Array2::from_elem((d, 1), -1.0 / d as f64),
            Array2::from_elem((1, d), 1.0),
            precision,
            Provenance::native("layer norm mean"),
        )
        .map_err(|e| e.to_string())?;
        b.operators.push(Arc::new(op));
        Some(b.operators.len() - 1)
    } else {
        None
    };
    let norm = |b: &mut Builder, x: usize, name: &str| -> Result<usize, String> {
        let input = match centring {
            Some(centre) => b.node(Node::Affine { terms: vec![(x, identity), (x, centre)], bias: None }),
            None => x,
        };
        let g = gain(b, &format!("{name}.gain"))?;
        let beta = bias(b, &format!("{name}.bias"), &model, None)?;
        let normed = b.node(Node::RmsNorm { input, epsilon });
        Ok(b.node(Node::Affine { terms: vec![(normed, g)], bias: beta }))
    };
    // A head norm (Qwen3's q_norm, k_norm): an RMS norm over the head, then its gain, a dense
    // diagonal operator on the head.
    let head_norm = |b: &mut Builder, x: usize, name: &str| -> Result<usize, String> {
        let g = tensors.get(name)?;
        let op = b.operator(name, &head, &head, Array2::from_diag(&g.row(0)))?;
        let normed = b.node(Node::RmsNorm { input: x, epsilon });
        Ok(b.node(Node::Affine { terms: vec![(normed, op)], bias: None }))
    };
    let feature = b.node(Node::Feature { slot: 0, basis: 0 });
    let embedded = b.node(Node::Affine { terms: vec![(feature, embedding)], bias: None });
    let mut x = embedded;
    for l in 0..layers {
        let prefix = format!("blocks.{l}.");
        let h = norm(&mut b, x, &format!("{prefix}rms1"))?;
        let (wq, wk, wv, wo) = (
            tensors.get(&format!("{prefix}attn.q_proj"))?,
            tensors.get(&format!("{prefix}attn.k_proj"))?,
            tensors.get(&format!("{prefix}attn.v_proj"))?,
            tensors.get(&format!("{prefix}attn.o_proj"))?,
        );
        let group = heads / kv_heads.max(1);
        let mut keys = Vec::new();
        for g in 0..kv_heads {
            let range = (g * hd, (g + 1) * hd);
            let rows = s![range.0..range.1, ..];
            let k_op = b.operator(&format!("{prefix}k{g}"), &head, &model, wk.slice(rows).to_owned())?;
            let v_op = b.operator(&format!("{prefix}v{g}"), &head, &model, wv.slice(rows).to_owned())?;
            let k_bias = bias(&mut b, &format!("{prefix}attn.k_proj.bias"), &head, Some(range))?;
            let v_bias = bias(&mut b, &format!("{prefix}attn.v_proj.bias"), &head, Some(range))?;
            let mut k = b.node(Node::Affine { terms: vec![(h, k_op)], bias: k_bias });
            if qk_norm {
                k = head_norm(&mut b, k, &format!("{prefix}attn.k_norm.gain"))?;
            }
            let v = b.node(Node::Affine { terms: vec![(h, v_op)], bias: v_bias });
            keys.push((k, v));
        }
        let mut terms = vec![(x, identity)];
        for hh in 0..heads {
            let range = (hh * hd, (hh + 1) * hd);
            let q_op = b.operator(&format!("{prefix}q{hh}"), &head, &model, wq.slice(s![range.0..range.1, ..]).to_owned())?;
            let o_op = b.operator(&format!("{prefix}o{hh}"), &model, &head, wo.slice(s![.., range.0..range.1]).to_owned())?;
            let q_bias = bias(&mut b, &format!("{prefix}attn.q_proj.bias"), &head, Some(range))?;
            let mut q = b.node(Node::Affine { terms: vec![(h, q_op)], bias: q_bias });
            if qk_norm {
                q = head_norm(&mut b, q, &format!("{prefix}attn.q_norm.gain"))?;
            }
            let (k, v) = keys[hh / group.max(1)];
            let read = b.node(Node::Attend {
                query: q,
                key: k,
                value: v,
                scale: Scale::InverseSqrt(hd as u32),
                rotary: Some(rotary),
                causal: true,
            });
            terms.push((read, o_op));
        }
        let o_bias = bias(&mut b, &format!("{prefix}attn.o_proj.bias"), &model, None)?;
        let attended = b.node(Node::Affine { terms, bias: o_bias });
        // A parallel block's MLP reads the stream the attention read; a sequential one reads the
        // stream after it.
        let h2 = norm(&mut b, if parallel { x } else { attended }, &format!("{prefix}rms2"))?;
        let c_fc = tensors.get(&format!("{prefix}mlp.c_fc"))?;
        let hidden = c_fc.nrows();
        let neurons = interface(hidden, 1, LabelKind::Unit)?;
        let up = b.operator(&format!("{prefix}c_fc"), &neurons, &model, c_fc)?;
        let up_bias = bias(&mut b, &format!("{prefix}mlp.c_fc.bias"), &neurons, None)?;
        let down = b.operator(&format!("{prefix}down_proj"), &model, &neurons, tensors.get(&format!("{prefix}mlp.down_proj"))?)?;
        let down_bias = bias(&mut b, &format!("{prefix}mlp.down_proj.bias"), &model, None)?;
        let pre = b.node(Node::Affine { terms: vec![(h2, up)], bias: up_bias });
        let active = if gated {
            let gate = b.operator(&format!("{prefix}gate_proj"), &neurons, &model, tensors.get(&format!("{prefix}mlp.gate_proj"))?)?;
            let gate_bias = bias(&mut b, &format!("{prefix}mlp.gate_proj.bias"), &neurons, None)?;
            let gate_pre = b.node(Node::Affine { terms: vec![(h2, gate)], bias: gate_bias });
            let gate_active = b.node(Node::Pointwise { input: gate_pre, laws: vec![act; hidden] });
            b.node(Node::Hadamard { left: gate_active, right: pre })
        } else {
            b.node(Node::Pointwise { input: pre, laws: vec![act; hidden] })
        };
        x = b.node(Node::Affine { terms: vec![(attended, identity), (active, down)], bias: down_bias });
    }
    let h = norm(&mut b, x, "final_norm")?;
    let logits = if tied {
        b.node(Node::Transposed { input: h, operator: embedding })
    } else {
        let head_op = b.operator("lm_head", &tokens_interface, &model, tensors.get("lm_head")?)?;
        b.node(Node::Affine { terms: vec![(h, head_op)], bias: None })
    };
    let output = b.node(Node::Readout { input: logits, basis: 0 });
    let declarations = Declarations {
        parameters: 0,
        domains: vec![Domain { size: vocab, cycle: None }],
        slots: vec![Slot::Token { domain: 0 }],
    };
    let program = OperatorProgram {
        rules: Vec::new(),
        declarations: declarations.clone(),
        bases: vec![Basis::Indicator { domain: 0 }],
        operators: b.operators,
        nodes: b.nodes,
        output,
    };
    let (token_rows, token_cols) = shape_of(&record["files"]["tokens"]["shape"])?;
    if sequences > token_rows || context > token_cols {
        return Err(format!("{sequences} sequences of {context} tokens from a {token_rows}×{token_cols} table"));
    }
    let tokens = read_f64(&dir.join("tokens.f64"), token_rows, token_cols)?;
    let mut ids = Vec::new();
    let (mut sequence, mut position) = (Vec::new(), Vec::new());
    for row in 0..sequences {
        for pos in 0..context {
            ids.push(tokens[[row, pos]] as u32);
            sequence.push(row as u32);
            position.push(pos as u32);
        }
    }
    let rows = ids.len();
    let family = FamilyInputs {
        rows,
        slots: vec![SlotValues::Tokens(ids)],
        layout: Some(SequenceLayout { sequence: sequence.clone(), position }),
    };
    let contract = Contract {
        declarations,
        family,
        kind: FamilyKind::Sample {
            population: "held-out token rows of the export".to_string(),
            confidence: 0.95,
            units: sequence.iter().map(|s| *s as usize).collect(),
        },
        observations: 1,
        readouts: 1,
        readout_slots: None,
    };
    Ok(Imported { name: "language_model".to_string(), kind: "language_model".to_string(), program, contract, record })
}
