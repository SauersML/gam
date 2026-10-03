//! The decomposition engine on exported models, with default settings (#2951 blind benchmark).
//!
//! `mpd_engine_blind_2951 MODEL_DIR OUT_DIR [SCREENINGS CERTIFICATIONS]`
//!
//! `MODEL_DIR` holds an `export.json` and raw float64 tensors (`gam_mpd::import`
//! reads transformers, residual MLPs and RNNs). The contract is the export's samples at its declared
//! readouts. No tolerance is declared: each program is chosen by its two-part code, over a ladder of
//! observations per sample `n = 10⁶, 10⁵, …, 1`, each search starting from the previous (larger-n)
//! rung's program, so the report is the frontier of program bits against observations.
//!
//! Written to `OUT_DIR`: `report.json` (per rung: program bits, data bits, maximal row KL, argmax
//! disagreements, population bounds for a sampled family, and the component view) and, per rung,
//! the program's decoded message `program_n{n}.bits` (raw bytes; its length in bits is in the
//! report) and the same program as JSON, `program_n{n}.json`: its bases, every operator (interface
//! groups by label kind, index and width; body kind, precision, present blocks and reals;
//! provenance sources and derivation), every node (kind, arguments, laws, scales, rotary) and its
//! rules, verbatim, for tools that read programs without decoding the message.

use gam_mpd::import::{import, import_language_model, is_language_model};
use gam_mpd::engine::{Budget, decompose_from, library};
use gam_mpd::operator_program::{Basis, Interface, Node, OperatorBody, OperatorProgram};
use gam_mpd::view::view;
use serde_json::{Value, json};
use std::path::PathBuf;

fn interface_json(interface: &Interface) -> Value {
    json!(interface.groups().iter().map(|g| json!([format!("{:?}", g.label.kind), g.label.index, g.width])).collect::<Vec<_>>())
}

fn node_json(node: &Node) -> Value {
    match node {
        Node::Feature { slot, basis } => json!({"kind": "Feature", "slot": slot, "basis": basis}),
        Node::Raw { slot } => json!({"kind": "Raw", "slot": slot}),
        Node::Constant { operator } => json!({"kind": "Constant", "operator": operator}),
        Node::Affine { terms, bias } => json!({"kind": "Affine", "terms": terms, "bias": bias}),
        Node::Bilinear { left, right, scale } => json!({"kind": "Bilinear", "left": left, "right": right, "scale": format!("{scale:?}")}),
        Node::Softmax { scores } => json!({"kind": "Softmax", "scores": scores}),
        Node::Mix { weights, payloads } => json!({"kind": "Mix", "weights": weights, "payloads": payloads}),
        Node::Pointwise { input, laws } => {
            json!({"kind": "Pointwise", "input": input, "laws": laws.iter().map(|l| format!("{l:?}")).collect::<Vec<_>>()})
        }
        Node::Hadamard { left, right } => json!({"kind": "Hadamard", "left": left, "right": right}),
        Node::Readout { input, basis } => json!({"kind": "Readout", "input": input, "basis": basis}),
        Node::Outer { left, right } => json!({"kind": "Outer", "left": left, "right": right}),
        Node::Concat { parts } => json!({"kind": "Concat", "parts": parts}),
        Node::Param { index } => json!({"kind": "Param", "index": index}),
        Node::Call { rule, arguments } => json!({"kind": "Call", "rule": rule, "arguments": arguments}),
        Node::Gain { input, coefficient } => json!({"kind": "Gain", "input": input, "coefficient": format!("{coefficient:?}")}),
        Node::Attend { query, key, value, scale, rotary, causal } => json!({
            "kind": "Attend", "query": query, "key": key, "value": value, "scale": format!("{scale:?}"),
            "rotary": rotary.map(|r| json!({"base": r.base, "dims": r.dims, "half_split": r.half_split})), "causal": causal,
        }),
        Node::RmsNorm { input, epsilon } => json!({"kind": "RmsNorm", "input": input, "epsilon": epsilon}),
        Node::Transposed { input, operator } => json!({"kind": "Transposed", "input": input, "operator": operator}),
    }
}

/// The program verbatim as JSON (module note).
fn program_json(program: &OperatorProgram) -> Value {
    let matrix = |m: &ndarray::Array2<f64>| json!(m.outer_iter().map(|row| row.to_vec()).collect::<Vec<_>>());
    json!({
        "bases": program.bases.iter().map(|b| match b {
            Basis::Indicator { domain } => json!({"kind": "Indicator", "domain": domain}),
            Basis::Characters { domain, positions, declared } => {
                json!({"kind": "Characters", "domain": domain, "positions": positions, "declared": declared})
            }
        }).collect::<Vec<_>>(),
        "operators": program.operators.iter().map(|op| {
            let body = match &op.body {
                OperatorBody::Identity => json!({"kind": "Identity"}),
                OperatorBody::Dense { values, present, precision } => json!({
                    "kind": "Dense", "fraction_bits": precision.fraction_bits(),
                    "present": present.indexed_iter().filter(|(_, k)| **k).map(|((r, c), _)| [r, c]).collect::<Vec<_>>(),
                    "values": matrix(values),
                }),
                OperatorBody::LowRank { left, right, precision } => json!({
                    "kind": "LowRank", "fraction_bits": precision.fraction_bits(), "left": matrix(left), "right": matrix(right),
                }),
                OperatorBody::Diagonal { values, precision } => json!({
                    "kind": "Diagonal", "fraction_bits": precision.fraction_bits(), "values": values.to_vec(),
                }),
            };
            json!({
                "name": op.name, "rows": interface_json(&op.rows), "cols": interface_json(&op.cols), "body": body,
                "sources": op.provenance.sources, "derivation": op.provenance.derivation,
            })
        }).collect::<Vec<_>>(),
        "nodes": program.nodes.iter().map(node_json).collect::<Vec<_>>(),
        "rules": program.rules.iter().map(|rule| json!({
            "name": rule.name, "inputs": rule.inputs.iter().map(interface_json).collect::<Vec<_>>(),
            "nodes": rule.nodes.iter().map(node_json).collect::<Vec<_>>(), "output": rule.output,
        })).collect::<Vec<_>>(),
        "output": program.output,
    })
}

fn main() -> Result<(), String> {
    gam_mpd::engine::log_to_stderr();
    let args: Vec<String> = std::env::args().collect();
    let usage = "mpd_engine_blind_2951 MODEL_DIR OUT_DIR [SCREENINGS CERTIFICATIONS]";
    let dir = PathBuf::from(args.get(1).ok_or(usage)?);
    let out = PathBuf::from(args.get(2).ok_or(usage)?);
    let screenings: u64 = args.get(3).map_or(Ok(1 << 20), |v| v.parse()).map_err(|e| format!("SCREENINGS: {e}"))?;
    let certifications: u64 = args.get(4).map_or(Ok(1 << 12), |v| v.parse()).map_err(|e| format!("CERTIFICATIONS: {e}"))?;
    std::fs::create_dir_all(&out).map_err(|e| e.to_string())?;
    // A language model's family is its first 4 token rows at 64 positions.
    let imported = if is_language_model(&dir)? { import_language_model(&dir, 4, 64)? } else { import(&dir)? };
    let library = library();
    let budget = Budget { screenings, certifications, ..Budget::default() };
    let model = &imported.program;
    let native_bits = model.code_bits().map_err(|e| e.to_string())?;
    let mut start = model.clone();
    let mut frontier = Vec::new();
    // Descending: the search only removes and coarsens, so each rung starts from the program of
    // the rung with more observations (an ascending ladder would start n = 10 from n = 1's
    // program, which has already dropped what larger n keeps, and cannot regrow it).
    for exponent in (0..=6u32).rev() {
        let n = 10u64.pow(exponent);
        let mut contract = imported.contract.clone();
        contract.observations = n;
        let started = std::time::Instant::now();
        let result = decompose_from(model, &start, &contract, &library, &budget).map_err(|e| e.to_string())?;
        let evaluation = &result.score.evaluation;
        let program_view = view(&result.program).map_err(|e| e.to_string())?;
        let message = result.program.encode().map_err(|e| e.to_string())?;
        let mut bytes = Vec::with_capacity((message.len_bits() as usize).div_ceil(8));
        let mut reader = message.reader();
        let mut byte = 0u8;
        for index in 0..message.len_bits() {
            byte = (byte << 1) | u8::from(reader.read_bit().map_err(|e| format!("{e:?}"))?);
            if index % 8 == 7 {
                bytes.push(byte);
                byte = 0;
            }
        }
        if message.len_bits() % 8 != 0 {
            bytes.push(byte << (8 - message.len_bits() % 8));
        }
        std::fs::write(out.join(format!("program_n{n}.bits")), &bytes).map_err(|e| e.to_string())?;
        std::fs::write(out.join(format!("program_n{n}.json")), program_json(&result.program).to_string()).map_err(|e| e.to_string())?;
        frontier.push(json!({
            "observations": n,
            "program_bits": result.score.program_bits,
            "structure_bits": result.score.structure_bits,
            "precision_bits": result.score.precision_bits,
            "explanation_bits": result.score.explanation.bits,
            "explanation_bits_per_input": result.score.explanation.bits_per_input(),
            "active_per_input": result.score.explanation.mean_active(),
            "data_bits": result.score.data_bits,
            "reals": result.program.real_count(),
            "max_kl": evaluation.max_kl.upper_bound(),
            "argmax_disagreements": evaluation.argmax_disagreements,
            "argmax_uncertified": evaluation.argmax_uncertified,
            "population": result.score.population.as_ref().map(|p| json!({
                "units": p.units, "disagreeing": p.disagreeing,
                "fixed_program_upper": p.fixed_program_upper, "selected_program_upper": p.selected_program_upper,
            })),
            "unchanged_native_bits": program_view.unchanged_native_bits,
            "components": program_view.components.iter().filter(|c| c.bits > 0).map(|c| json!({
                "name": c.name, "reads": c.reads, "writes": c.writes, "applied_by": c.applied_by, "uses": c.uses,
                "reals": c.reals, "bits": c.bits, "sources": c.sources, "native": c.native_unchanged,
            })).collect::<Vec<_>>(),
            "curve": result.curve.iter().map(|c| json!({
                "structure_bits": c.structure_bits, "rest_bits": c.rest_bits, "description": c.description,
            })).collect::<Vec<_>>(),
            "knee": result.knee().map(|c| json!({"structure_bits": c.structure_bits, "rest_bits": c.rest_bits, "description": c.description})),
            "stop": format!("{:?}", result.stop),
            "seconds": started.elapsed().as_secs_f64(),
        }));
        eprintln!(
            "{} n={n}: {} + {:.1} bits (native {native_bits}), max KL <= {:e}, {} argmax disagreements, {:.1}s",
            imported.name,
            result.score.program_bits,
            result.score.data_bits,
            evaluation.max_kl.upper_bound().unwrap_or(f64::INFINITY),
            evaluation.argmax_disagreements,
            started.elapsed().as_secs_f64()
        );
        start = result.program;
    }
    let report = json!({
        "model": imported.name,
        "kind": imported.kind,
        "config": imported.record["config"],
        "native_bits": native_bits,
        "native_reals": model.real_count(),
        "rows": imported.contract.family.rows,
        "readouts": imported.contract.readouts,
        "frontier": frontier,
    });
    std::fs::write(out.join("report.json"), serde_json::to_string_pretty(&report).map_err(|e| e.to_string())?)
        .map_err(|e| e.to_string())?;
    Ok(())
}
