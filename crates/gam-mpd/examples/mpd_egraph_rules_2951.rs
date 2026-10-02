//! Equality saturation and rule discovery on trained toys (#2951).
//!
//! `mpd_egraph_rules_2951 modadd EXPORT_DIR` or `mpd_egraph_rules_2951 resid_mlp EXPORT_DIR`
//!
//! `modadd`: `EXPORT_DIR` holds the run's tensors as raw little-endian float64 `<name>.f64` files
//! and `export.json` (their shapes and the run's config), e.g. `~/mpd-data/engine/p31_s0` from
//! `~/mpd-data/tiny_modadd/addition_p31_s0.pt`. The network is one layer read at the `=` position:
//! `x_j = W_E[t_j] + W_pos[j]` for the tokens `(a, b, =)`, softmax attention from `=` over the
//! three positions, the residual, a ReLU MLP and the unembedding. The family is all `p²` inputs.
//!
//! `resid_mlp`: `EXPORT_DIR` is what `bench/mpd_resid_mlp_toys_2951.py` writes (`manifest.json`,
//! float64 `.npy`), e.g. `~/mpd-data/resid_mlp/L2`: `r = x W_E`, per layer `r += W_out relu(W_in r)`,
//! `y = r W_Eᵀ`, the readout the embedding read transposed (the tie). The family is sparse
//! features, each row one or two active with dyadic values.
//!
//! Either model enters as an operator program with one operator per native tensor (tied where the
//! network ties them) and the native laws. The program is normalized by saturation
//! (`egraph::normalize`), rules are discovered on the normalized e-graph
//! (`antiunify::discover_rules`), and the report (JSON on stdout) gives the decoded bits and real
//! counts of the native, normalized and rule-compressed messages, the saturation's size, each
//! accepted rule, and two checks on the family: the normalized program computes the native output
//! within the two programs' execution bands, and the library decodes every call site's operator
//! bit-exactly.

use gam_runtime::resource::MemoryGovernor;
use gam_mpd::antiunify::discover_rules;
use gam_mpd::egraph::normalize;
use gam_mpd::operator_program::{
    Basis, Declarations, Domain, FamilyInputs, Interface, LabelKind, Law, Node, Operator, OperatorProgram, Provenance,
    Scale, Slot, SlotValues, exact_precision,
};
use ndarray::{Array2, Axis, s};
use serde_json::{Value, json};
use std::path::Path;
use std::process::ExitCode;

/// Tensor `name` of the export: raw little-endian float64 in C order, shaped by `export.json`.
fn read_tensor(dir: &Path, record: &Value, name: &str) -> Result<Array2<f64>, String> {
    let shape = record["files"][name]["shape"]
        .as_array()
        .ok_or_else(|| format!("export.json: no shape for {name}"))?
        .iter()
        .map(|v| v.as_u64().map(|v| v as usize).ok_or_else(|| format!("export.json: {name} shape")))
        .collect::<Result<Vec<_>, _>>()?;
    let (rows, cols) = match shape[..] {
        [rows, cols] => (rows, cols),
        [len] => (1, len),
        _ => return Err(format!("{name}: shape {shape:?}")),
    };
    let path = dir.join(format!("{name}.f64"));
    let bytes = std::fs::read(&path).map_err(|error| format!("{}: {error}", path.display()))?;
    if bytes.len() != rows * cols * 8 {
        return Err(format!("{}: {} bytes for {rows}×{cols}", path.display(), bytes.len()));
    }
    let values = bytes.chunks_exact(8).map(|c| f64::from_le_bytes(c.try_into().expect("eight bytes"))).collect();
    Array2::from_shape_vec((rows, cols), values).map_err(|error| error.to_string())
}

fn dense(name: &str, rows: &Interface, cols: &Interface, values: Array2<f64>) -> Result<Operator, String> {
    let precision = exact_precision(values.iter().copied()).map_err(|error| error.to_string())?;
    Operator::dense(name, rows.clone(), cols.clone(), values, precision, Provenance::native(name)).map_err(|error| error.to_string())
}

/// The native modular-addition network as an operator program, and every `(a, b)` at `=`.
fn modadd(dir: &Path) -> Result<(OperatorProgram, FamilyInputs), String> {
    let text = std::fs::read_to_string(dir.join("export.json")).map_err(|error| error.to_string())?;
    let record: Value = serde_json::from_str(&text).map_err(|error| error.to_string())?;
    let field = |name: &str| record["config"][name].as_u64().map(|v| v as usize).ok_or_else(|| format!("config.{name}"));
    let (p, heads, dh, d, n) = (field("p")?, field("n_heads")?, field("d_head")?, field("d_model")?, field("d_mlp")?);
    let tensor = |name: &str| read_tensor(dir, &record, name);
    let err = |e: gam_mpd::operator_program::ProgramError| e.to_string();
    let declarations = Declarations {
        domains: vec![Domain { size: p + 1, cycle: None }, Domain { size: p, cycle: None }],
        slots: vec![Slot::Token { domain: 0 }; 3],
        parameters: 0,
    };
    let tokens = Interface::uniform(p + 1, 1, LabelKind::Token, 0).map_err(err)?;
    let classes = Interface::uniform(p, 1, LabelKind::Token, 0).map_err(err)?;
    let model = Interface::native(d).map_err(err)?;
    let head = Interface::native(dh).map_err(err)?;
    let units = Interface::uniform(n, 1, LabelKind::Unit, 0).map_err(err)?;
    let constant = Interface::constant();
    let (w_e, w_pos, w_q, w_k, w_v, w_o) =
        (tensor("W_E")?, tensor("W_pos")?, tensor("W_Q")?, tensor("W_K")?, tensor("W_V")?, tensor("W_O")?);
    let (w_in, b_in, w_out, b_out, w_u) = (tensor("W_in")?, tensor("b_in")?, tensor("W_out")?, tensor("b_out")?, tensor("W_U")?);
    let mut operators = vec![dense("W_E", &model, &tokens, w_e.t().to_owned())?];
    for j in 0..3 {
        operators.push(dense(&format!("pos{j}"), &model, &constant, w_pos.row(j).to_owned().insert_axis(Axis(1)))?);
    }
    let mut nodes: Vec<Node> = (0..3).map(|slot| Node::Feature { slot, basis: 0 }).collect();
    for j in 0..3 {
        nodes.push(Node::Affine { terms: vec![(j, 0)], bias: Some(j + 1) });
    }
    let x = [3usize, 4, 5];
    let mut mixes = Vec::new();
    for h in 0..heads {
        let rows = s![h * dh..(h + 1) * dh, ..];
        operators.push(dense(&format!("W_Q{h}"), &head, &model, w_q.slice(rows).to_owned())?);
        operators.push(dense(&format!("W_K{h}"), &head, &model, w_k.slice(rows).to_owned())?);
        operators.push(dense(&format!("W_V{h}"), &head, &model, w_v.slice(rows).to_owned())?);
        operators.push(dense(&format!("W_O{h}"), &model, &head, w_o.slice(s![.., h * dh..(h + 1) * dh]).to_owned())?);
        let (q_op, k_op, v_op, o_op) = (operators.len() - 4, operators.len() - 3, operators.len() - 2, operators.len() - 1);
        nodes.push(Node::Affine { terms: vec![(x[2], q_op)], bias: None });
        let q = nodes.len() - 1;
        let mut scores = Vec::new();
        for &xj in &x {
            nodes.push(Node::Affine { terms: vec![(xj, k_op)], bias: None });
            nodes.push(Node::Bilinear { left: q, right: nodes.len() - 1, scale: Scale::InverseSqrt(dh as u32) });
            scores.push(nodes.len() - 1);
        }
        nodes.push(Node::Softmax { scores });
        let weights = nodes.len() - 1;
        let mut payloads = Vec::new();
        for (j, &xj) in x.iter().enumerate() {
            nodes.push(Node::Affine { terms: vec![(xj, v_op)], bias: None });
            payloads.push((j, nodes.len() - 1));
        }
        nodes.push(Node::Mix { weights, payloads });
        mixes.push((nodes.len() - 1, o_op));
    }
    operators.push(Operator::identity("I", model.clone()));
    let identity = operators.len() - 1;
    let mut mid_terms = mixes;
    mid_terms.push((x[2], identity));
    nodes.push(Node::Affine { terms: mid_terms, bias: None });
    let mid = nodes.len() - 1;
    operators.push(dense("W_in", &units, &model, w_in)?);
    operators.push(dense("b_in", &units, &constant, b_in.t().to_owned())?);
    nodes.push(Node::Affine { terms: vec![(mid, operators.len() - 2)], bias: Some(operators.len() - 1) });
    nodes.push(Node::Pointwise { input: nodes.len() - 1, laws: vec![Law::Relu; n] });
    let act = nodes.len() - 1;
    operators.push(dense("W_out", &model, &units, w_out)?);
    operators.push(dense("b_out", &model, &constant, b_out.t().to_owned())?);
    nodes.push(Node::Affine { terms: vec![(mid, identity), (act, operators.len() - 2)], bias: Some(operators.len() - 1) });
    let fin = nodes.len() - 1;
    operators.push(dense("W_U", &classes, &model, w_u)?);
    nodes.push(Node::Affine { terms: vec![(fin, operators.len() - 1)], bias: None });
    nodes.push(Node::Readout { input: nodes.len() - 1, basis: 1 });
    let program = OperatorProgram {
        declarations,
        bases: vec![Basis::Indicator { domain: 0 }, Basis::Indicator { domain: 1 }],
        operators: operators.into_iter().map(std::sync::Arc::new).collect(),
        rules: Vec::new(),
        output: nodes.len() - 1,
        nodes,
    };
    let pairs: Vec<(u32, u32)> = (0..p as u32).flat_map(|a| (0..p as u32).map(move |b| (a, b))).collect();
    let family = FamilyInputs {
        rows: pairs.len(),
        slots: vec![
            SlotValues::Tokens(pairs.iter().map(|q| q.0).collect()),
            SlotValues::Tokens(pairs.iter().map(|q| q.1).collect()),
            SlotValues::Tokens(vec![p as u32; pairs.len()]),
        ],
        layout: None,
    };
    Ok((program, family))
}

/// A little-endian float64 C-order `.npy` array.
fn read_npy(path: &Path) -> Result<Array2<f64>, String> {
    let bytes = std::fs::read(path).map_err(|error| format!("{}: {error}", path.display()))?;
    if bytes.len() < 10 || &bytes[..6] != b"\x93NUMPY" {
        return Err(format!("{}: not an .npy file", path.display()));
    }
    let (length, start) = match bytes[6] {
        1 => (usize::from(u16::from_le_bytes([bytes[8], bytes[9]])), 10),
        _ => (u32::from_le_bytes([bytes[8], bytes[9], bytes[10], bytes[11]]) as usize, 12),
    };
    let header = std::str::from_utf8(&bytes[start..start + length]).map_err(|error| error.to_string())?;
    if !header.contains("'descr': '<f8'") || !header.contains("'fortran_order': False") {
        return Err(format!("{}: {header}", path.display()));
    }
    let shape = header.split("'shape': (").nth(1).and_then(|rest| rest.split(')').next()).ok_or("an .npy shape")?;
    let dims: Vec<usize> = shape.split(',').filter(|d| !d.trim().is_empty()).map(|d| d.trim().parse().map_err(|e| format!("{e}"))).collect::<Result<_, _>>()?;
    let [rows, cols] = dims[..] else { return Err(format!("{}: shape {dims:?}", path.display())) };
    let data = &bytes[start + length..];
    if data.len() != rows * cols * 8 {
        return Err(format!("{}: {} bytes for {rows}×{cols}", path.display(), data.len()));
    }
    let values = data.chunks_exact(8).map(|c| f64::from_le_bytes(c.try_into().expect("eight bytes"))).collect();
    Array2::from_shape_vec((rows, cols), values).map_err(|error| error.to_string())
}

/// The native residual MLP as an operator program, and a family of sparse feature vectors.
fn resid_mlp(dir: &Path) -> Result<(OperatorProgram, FamilyInputs), String> {
    let text = std::fs::read_to_string(dir.join("manifest.json")).map_err(|error| error.to_string())?;
    let manifest: Value = serde_json::from_str(&text).map_err(|error| error.to_string())?;
    let field = |name: &str| manifest[name].as_u64().map(|v| v as usize).ok_or_else(|| format!("manifest.{name}"));
    let (layers, features, width, hidden) = (field("layers")?, field("features")?, field("width")?, field("hidden")?);
    let err = |e: gam_mpd::operator_program::ProgramError| e.to_string();
    let x = Interface::native(features).map_err(err)?;
    let r = Interface::native(width).map_err(err)?;
    let units = Interface::uniform(hidden, 1, LabelKind::Unit, 0).map_err(err)?;
    let embed = read_npy(&dir.join("W_E.npy"))?;
    let mut operators = vec![dense("W_E", &r, &x, embed.t().to_owned())?, Operator::identity("I", r.clone())];
    let mut nodes = vec![Node::Raw { slot: 0 }, Node::Affine { terms: vec![(0, 0)], bias: None }];
    let mut stream = 1;
    for layer in 0..layers {
        operators.push(dense(&format!("W_in{layer}"), &units, &r, read_npy(&dir.join(format!("W_in{layer}.npy")))?)?);
        operators.push(dense(&format!("W_out{layer}"), &r, &units, read_npy(&dir.join(format!("W_out{layer}.npy")))?)?);
        let (read, write) = (operators.len() - 2, operators.len() - 1);
        nodes.push(Node::Affine { terms: vec![(stream, read)], bias: None });
        nodes.push(Node::Pointwise { input: nodes.len() - 1, laws: vec![Law::Relu; hidden] });
        nodes.push(Node::Affine { terms: vec![(stream, 1), (nodes.len() - 1, write)], bias: None });
        stream = nodes.len() - 1;
    }
    nodes.push(Node::Transposed { input: stream, operator: 0 });
    let program = OperatorProgram {
        declarations: Declarations { domains: Vec::new(), slots: vec![Slot::Raw { width: features }], parameters: 0 },
        bases: Vec::new(),
        operators: operators.into_iter().map(std::sync::Arc::new).collect(),
        rules: Vec::new(),
        output: nodes.len() - 1,
        nodes,
    };
    let rows = 2 * features;
    let mut values = Array2::<f64>::zeros((rows, features));
    for row in 0..rows {
        values[[row, row % features]] = ((row * 37 % 64) as f64 - 32.0) / 32.0;
        if row >= features {
            values[[row, (row * 7 + 3) % features]] = ((row * 11 % 64) as f64 - 32.0) / 32.0;
        }
    }
    Ok((program, FamilyInputs { rows, slots: vec![SlotValues::Raw(values)], layout: None }))
}

/// The largest excess of `|left − right|` over the two programs' summed bands at the output.
fn band_excess(left: &OperatorProgram, right: &OperatorProgram, inputs: &FamilyInputs) -> Result<(f64, f64), String> {
    let a = left.execute(inputs, true).map_err(|e| e.to_string())?.banded(left.output);
    let b = right.execute(inputs, true).map_err(|e| e.to_string())?.banded(right.output);
    let mut excess = f64::NEG_INFINITY;
    let mut largest = 0.0_f64;
    for ((index, x), y) in a.values.indexed_iter().zip(b.values.iter()) {
        largest = largest.max((x - y).abs());
        excess = excess.max((x - y).abs() - (a.bands[index] + b.bands[index]));
    }
    Ok((largest, excess))
}

fn run() -> Result<(), String> {
    let usage = "usage: mpd_egraph_rules_2951 (modadd | resid_mlp) EXPORT_DIR";
    let (model, dir) = (std::env::args().nth(1).ok_or(usage)?, std::env::args().nth(2).ok_or(usage)?);
    let (program, family) = match model.as_str() {
        "modadd" => modadd(Path::new(&dir))?,
        "resid_mlp" => resid_mlp(Path::new(&dir))?,
        _ => return Err(usage.to_string()),
    };
    let start = std::time::Instant::now();
    let normalization = normalize(&program, MemoryGovernor::global()).map_err(|e| e.to_string())?;
    let saturated = start.elapsed().as_secs_f64();
    let library = discover_rules(&normalization).map_err(|e| e.to_string())?;
    let discovered = start.elapsed().as_secs_f64() - saturated;
    let (largest, excess) = band_excess(&program, &normalization.program, &family)?;
    let decoded = library.decode_instances().map_err(|e| e.to_string())?;
    let exact = decoded.iter().all(|(index, matrix)| *matrix == library.program.operators[*index].matrix());
    let report = &normalization.saturation.report;
    let out = json!({
        "model": model,
        "export": dir,
        "native_bits": normalization.native_bits,
        "native_reals": program.real_count(),
        "normalized_reals": normalization.program.real_count(),
        "normalized_bits": normalization.bits,
        "extraction_bits": normalization.extraction.bits,
        "library_bits_before": library.bits_before,
        "library_bits_after": library.bits_after,
        "saturation": {
            "stop": format!("{:?}", report.stop),
            "iterations": report.iterations,
            "nodes": report.nodes,
            "classes": report.classes,
            "leaves": report.leaves,
            "seconds": saturated,
        },
        "discovery_seconds": discovered,
        "operators": { "native": program.operators.len(), "normalized": normalization.program.operators.len() },
        "rules": library.rules.iter().map(|rule| json!({
            "skeleton": rule.skeleton,
            "holes": rule.holes.len(),
            "calls": rule.calls.iter().map(|call| json!({
                "family": format!("{:?}", call.binding.family),
                "bits": call.binding.bits,
                "residual_bits": call.binding.residual_bits,
            })).collect::<Vec<_>>(),
            "saving": rule.saving,
        })).collect::<Vec<_>>(),
        "normalized_output": { "largest_difference": largest, "largest_excess_over_bands": excess },
        "call_sites_decode_exactly": exact,
    });
    println!("{}", serde_json::to_string_pretty(&out).map_err(|e| e.to_string())?);
    if excess > 0.0 || !exact {
        return Err("a check failed".to_string());
    }
    Ok(())
}

fn main() -> ExitCode {
    match run() {
        Ok(()) => ExitCode::SUCCESS,
        Err(message) => {
            eprintln!("{message}");
            ExitCode::FAILURE
        }
    }
}
