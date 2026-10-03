#![cfg(test)]
//! A device-executed program against the CPU's own execution, on a small rotary language model
//! (two layers, grouped keys, tanh GELU, tied unembedding) over three sequences in one batch.
//!
//! The bands: the CPU's banded execution bounds `|cpu − exact|` per node value (its entry's box
//! plus its row's ball); the device
//! evaluates the same expressions in float64 (its sums in another order, its `exp`/`tanh` within
//! the same ulps), so its error obeys the same bound and the two agree within twice the band.
//! The KL moves by at most `Σ_c |q_c − p_c| · max_c |Δz_c| ≤ 2 max_c |Δz_c|` under a logit change
//! `Δz`, plus each evaluation's own rounding, `γ_{V+8} Σ_c p_c (|ln p_c| + |ln q_c|)` for `V`
//! classes. Reverse passes and tangents only propose, and proposals may run in f32: they agree
//! with the CPU's within the f32 proposal band, `(w + 2)·2⁻²⁴` of the largest entry, `w` the
//! widest contraction. Every check runs on the host backend, and on a CUDA device when one is
//! present (a runtime probe; absent, the device half has nothing to run).

use super::derivatives::{jvp, vjp};
use super::device_program::DeviceProgram;
use super::import::{hugging_face_language_model, import_language_model};
use super::operator_program::{FamilyInputs, Node, Operator, OperatorBody, OperatorProgram, SequenceLayout};
use gam_gpu::GpuPolicy;
use gam_gpu::tensor::{Arithmetic, Device};
use ndarray::{Array1, Array2};
use std::collections::BTreeMap;
use std::path::Path;

const D: usize = 8;
const HEADS: usize = 2;
const HEAD_DIM: usize = 4;
const HIDDEN: usize = 16;
const VOCAB: usize = 13;
const CONTEXT: usize = 6;
const SEQUENCES: usize = 3;
const U: f64 = f64::EPSILON / 2.0;

pub(super) fn noise(seed: usize) -> f64 {
    let mut x = (seed as u64).wrapping_mul(0x9E37_79B9_7F4A_7C15) ^ 0xD1B5_4A32_D192_ED03;
    x ^= x >> 33;
    x = x.wrapping_mul(0xFF51_AFD7_ED55_8CCD);
    x ^= x >> 33;
    (x >> 11) as f64 / (1u64 << 52) as f64 - 1.0
}

fn write(dir: &Path, files: &mut serde_json::Map<String, serde_json::Value>, name: &str, values: &Array2<f64>) {
    let bytes: Vec<u8> = values.iter().flat_map(|v| v.to_le_bytes()).collect();
    std::fs::write(dir.join(format!("{name}.f64")), bytes).expect("written");
    files.insert(name.to_string(), serde_json::json!({"shape": [values.nrows(), values.ncols()]}));
}

/// A small export (`import::import_language_model`'s format) and its program and family.
pub(super) fn fixture() -> (OperatorProgram, FamilyInputs) {
    fixture_sized(D, VOCAB, CONTEXT, SEQUENCES)
}

/// [`fixture`] at width `d`, `vocab` classes and `sequences` sequences of `context` tokens.
pub(super) fn fixture_sized(d: usize, vocab: usize, context: usize, sequences: usize) -> (OperatorProgram, FamilyInputs) {
    let dir = std::env::temp_dir().join(format!("gam-mpd-device-program-{}-{:?}-{d}-{vocab}", std::process::id(), std::thread::current().id()));
    std::fs::create_dir_all(&dir).expect("dir");
    let mut files = serde_json::Map::new();
    let mut seed = 0;
    let mut random = |rows: usize, cols: usize, scale: f64| {
        seed += 1000;
        let base = seed;
        Array2::from_shape_fn((rows, cols), |(i, j)| scale * noise(base + i * cols + j))
    };
    write(&dir, &mut files, "wte", &random(vocab, d, 1.0));
    for l in 0..2 {
        let p = format!("blocks.{l}.");
        write(&dir, &mut files, &format!("{p}attn.q_proj"), &random(HEADS * HEAD_DIM, d, 0.5));
        write(&dir, &mut files, &format!("{p}attn.k_proj"), &random(HEAD_DIM, d, 0.5));
        write(&dir, &mut files, &format!("{p}attn.v_proj"), &random(HEAD_DIM, d, 0.5));
        write(&dir, &mut files, &format!("{p}attn.o_proj"), &random(d, HEADS * HEAD_DIM, 0.5));
        write(&dir, &mut files, &format!("{p}mlp.c_fc"), &random(HIDDEN, d, 0.5));
        write(&dir, &mut files, &format!("{p}mlp.down_proj"), &random(d, HIDDEN, 0.5));
        write(&dir, &mut files, &format!("{p}rms1.gain"), &(random(1, d, 0.2) + 1.0));
        write(&dir, &mut files, &format!("{p}rms2.gain"), &(random(1, d, 0.2) + 1.0));
    }
    write(&dir, &mut files, "final_norm.gain", &(random(1, d, 0.2) + 1.0));
    let tokens = Array2::from_shape_fn((sequences, context), |(s, t)| ((noise(7 * s + t + 99) + 1.0) * 0.5 * vocab as f64) as usize as f64 % vocab as f64);
    write(&dir, &mut files, "tokens", &tokens);
    let record = serde_json::json!({
        "config": {
            "d_model": d, "n_layers": 2, "n_heads": HEADS, "n_kv_heads": 1, "head_dim": HEAD_DIM,
            "vocab": vocab, "rope_theta": 10000.0, "rope_pairing": "rotate_half", "norm_eps": 1e-6,
            "mlp_act": "gelu_tanh", "tied_embeddings": true,
        },
        "files": files,
    });
    std::fs::write(dir.join("export.json"), record.to_string()).expect("written");
    let imported = import_language_model(&dir, sequences, context).expect("imported");
    std::fs::remove_dir_all(&dir).expect("removed");
    (imported.program, imported.contract.family)
}

/// The host reference, and a CUDA device when one is present.
pub(super) fn devices() -> Vec<Device> {
    let mut out = vec![Device::host()];
    if let Some(device) = Device::accelerator(GpuPolicy::Auto).expect("a probe that does not fault") {
        out.push(device);
    }
    out
}

pub(super) fn gamma(n: usize) -> f64 {
    n as f64 * U / (1.0 - n as f64 * U)
}

pub(super) fn softmax(z: ndarray::ArrayView1<'_, f64>) -> Array1<f64> {
    let m = z.iter().copied().fold(f64::NEG_INFINITY, f64::max);
    let e = z.mapv(|v| (v - m).exp());
    let total = e.sum();
    e / total
}

fn target_of(logits: &Array2<f64>) -> Array2<f64> {
    Array2::from_shape_fn(logits.dim(), |(r, c)| logits[[r, c]] + 0.7 * noise(31 * r + c + 5))
}

/// Within the f32 proposal band of the reference's largest entry.
pub(super) fn assert_proposal(what: &str, device: &Array2<f64>, cpu: &Array2<f64>, widest: usize) {
    let scale = cpu.iter().fold(0.0_f64, |m, v| m.max(v.abs()));
    let band = (widest + 2) as f64 * 2f64.powi(-24) * scale;
    let worst = device.iter().zip(cpu).fold(0.0_f64, |m, (a, b)| m.max((a - b).abs()));
    assert!(worst <= band, "{what}: device and CPU differ by {worst:e}, band {band:e}");
}

#[test]
fn device_forward_and_kl_match_the_cpu_within_bands() {
    let (program, family) = fixture();
    let cpu = program.execute(&family, true).expect("cpu");
    // Each value's enclosure: its entrywise box plus its row's ℓ₂ ball.
    let bands = cpu.bands.as_ref().expect("bands");
    let balls = cpu.balls.as_ref().expect("balls");
    let logits = &cpu.values[program.output];
    let target = target_of(logits);
    for device in devices() {
        let lowered = DeviceProgram::compile(&device, &program).expect("lowered");
        let trace = lowered.forward(&family).expect("forward");
        for (n, value) in trace.values.iter().enumerate() {
            let Some(value) = value else { continue };
            let got = device.download(value).expect("download");
            for ((r, c), a) in got.indexed_iter() {
                let band = bands[n][[r, c]] + balls[n][r];
                let gap = (a - cpu.values[n][[r, c]]).abs();
                assert!(gap <= 2.0 * band, "{}: node {n} ({r},{c}) differs by {gap:e}, band {band:e}", device.name());
            }
        }
        let device_logits = lowered.logits(&trace, 0, family.rows).expect("logits");
        let target_tensor = device.upload(target.view()).expect("upload");
        let (kl, _) = lowered.kl(&trace, &target_tensor, None).expect("kl");
        for r in 0..family.rows {
            let (p, q) = (softmax(target.row(r)), softmax(logits.row(r)));
            let expected: f64 = (0..VOCAB).filter(|&c| p[c] > 0.0).map(|c| p[c] * (p[c].ln() - q[c].ln())).sum();
            let moved = (0..VOCAB).fold(0.0_f64, |m, c| m.max((device_logits[[r, c]] - logits[[r, c]]).abs()));
            let logit_band = bands[program.output].row(r).iter().fold(0.0_f64, |m, v| m.max(*v)) + balls[program.output][r];
            assert!(moved <= 2.0 * logit_band, "logits of row {r}");
            let magnitude: f64 = (0..VOCAB).map(|c| p[c] * (p[c].ln().abs() + q[c].ln().abs())).sum();
            let band = 2.0 * moved + 2.0 * gamma(VOCAB + 8) * magnitude;
            assert!((kl[r] - expected).abs() <= band, "{}: KL of row {r} differs by {:e}, band {band:e}", device.name(), (kl[r] - expected).abs());
        }
    }
}

#[test]
fn device_reverse_pass_and_tangent_match_the_cpu_within_the_proposal_band() {
    let (program, family) = fixture();
    let cpu = program.execute(&family, false).expect("cpu");
    let logits = &cpu.values[program.output];
    let target = target_of(logits);
    // The CPU's KL cotangent, pulled back through every node.
    let cotangent = Array2::from_shape_fn(logits.dim(), |(r, c)| softmax(logits.row(r))[c] - softmax(target.row(r))[c]);
    let back = vjp(&program, &family, &cpu, cotangent).expect("cpu vjp");
    // A tangent of one query head and one MLP map; the CPU's output Fisher quadratic on it.
    let pick = |name: &str| program.operators.iter().position(|op| op.name == name).expect(name);
    let mut tangents = BTreeMap::new();
    for name in ["blocks.0.q1", "blocks.1.c_fc"] {
        let op = pick(name);
        let (rows, cols) = program.operators[op].matrix().dim();
        tangents.insert(op, Array2::from_shape_fn((rows, cols), |(i, j)| noise(op * 977 + i * cols + j)));
    }
    let output_tangent = jvp(&program, &family, &cpu, &tangents).expect("cpu jvp");
    let mut quadratic = 0.0;
    for r in 0..family.rows {
        let q = softmax(logits.row(r));
        let t = output_tangent.row(r);
        let mean: f64 = q.iter().zip(t.iter()).map(|(a, b)| a * b).sum();
        quadratic += q.iter().zip(t.iter()).map(|(a, b)| a * (b - mean) * (b - mean)).sum::<f64>();
    }
    for device in devices() {
        let lowered = DeviceProgram::compile(&device, &program).expect("lowered");
        let trace = lowered.forward(&family).expect("forward");
        let target_tensor = device.upload(target.view()).expect("upload");
        let (_, seed) = lowered.kl(&trace, &target_tensor, None).expect("kl");
        let keep: Vec<usize> = (0..=lowered.hidden()).filter(|n| trace.values[*n].is_some()).collect();
        let kept = lowered.vjp(&trace, seed, &keep, Arithmetic::F64).expect("vjp");
        for (n, g) in &kept {
            let reference = back[*n].as_ref().expect("the CPU's cotangent reaches it too");
            assert_proposal(&format!("{} cotangent of node {n}", device.name()), &device.download(g).expect("download"), reference, VOCAB.max(HIDDEN));
        }
        assert_eq!(kept.len(), keep.len(), "a cotangent reaches every resident node");
        let tangent = lowered.jvp(&trace, &tangents, Arithmetic::F64).expect("jvp").expect("a tangent reaches the head");
        let device_quadratic = lowered.quadratic(&trace, &tangent, None, Arithmetic::F64).expect("quadratic");
        let band = (VOCAB + 2) as f64 * 2f64.powi(-24) * quadratic.abs();
        assert!((device_quadratic - quadratic).abs() <= band, "{}: quadratic {device_quadratic} against {quadratic}", device.name());
    }
}

#[test]
fn a_family_whose_sequences_are_not_equal_blocks_is_refused() {
    let (program, family) = fixture();
    let layout = family.layout.as_ref().expect("layout");
    // The last sequence's rows relabelled as two shorter sequences.
    let mut sequence = layout.sequence.clone();
    let last = sequence.len() - 2;
    sequence[last] = 77;
    sequence[last + 1] = 77;
    let ragged = FamilyInputs { layout: Some(SequenceLayout { sequence, position: layout.position.clone() }), ..family };
    let lowered = DeviceProgram::compile(&Device::host(), &program).expect("lowered");
    assert!(lowered.forward(&ragged).is_err());
}

#[test]
fn shared_sampled_head_seeds_match_independent_head_pullbacks() {
    let (program, family) = fixture();
    let uniforms: Vec<Vec<f64>> = (0..5).map(|s| (0..family.rows).map(|r| ((r * 13 + s * 7) % 101) as f64 / 101.0).collect()).collect();
    let scored: Vec<bool> = (0..family.rows).map(|r| r % 3 != 1).collect();
    for device in devices() {
        let lowered = DeviceProgram::compile(&device, &program).expect("lowered");
        let trace = lowered.forward(&family).expect("trace");
        for flags in [None, Some(scored.as_slice())] {
            let seeds = lowered.sampled_many(&trace, &uniforms, flags).expect("shared seeds");
            for (u, seed) in uniforms.iter().zip(seeds) {
                let reference = lowered.sampled(&trace, u, flags, Arithmetic::F64).expect("independent seed");
                let reference = device.download(&reference).expect("download");
                let actual = device.download(&seed).expect("download");
                for ((r, c), value) in actual.indexed_iter() {
                    assert!((value - reference[[r, c]]).abs() < 1e-12, "{}: seed at ({r}, {c})", device.name());
                    if flags.is_some_and(|s| !s[r]) { assert_eq!(*value, 0.0); }
                }
            }
        }
        assert!(lowered.sampled_many(&trace, &[], None).expect("no samples").is_empty());
        assert!(lowered.sampled_many(&trace, &[vec![0.5]], None).is_err());
    }
}

/// `program` as the importer used to build it: each norm gain a dense matrix with its diagonal
/// blocks present, and the token feature also read by an unread concatenation, so its one-hot rows
/// are formed and every affine term reads them as a matrix.
fn dense_path(program: &OperatorProgram, feature: usize) -> OperatorProgram {
    let mut dense = program.clone();
    for op in &mut dense.operators {
        if let OperatorBody::Diagonal { values, precision } = &op.body {
            let n = values.len();
            let groups = (op.rows.group_count(), op.cols.group_count());
            let present = Array2::from_shape_fn(groups, |(r, c)| r == c || groups == (1, 1));
            let blocks = Operator::blocks(op.name.clone(), op.rows.clone(), op.cols.clone(), Array2::from_diag(values), present, *precision, op.provenance.clone())
                .expect("blocks");
            assert_eq!(blocks.diagonal().expect("diagonal").len(), n);
            *op = std::sync::Arc::new(blocks);
        }
    }
    dense.nodes.push(Node::Concat { parts: vec![feature] });
    dense
}

#[test]
fn norm_gains_stay_diagonal_and_the_embedding_is_a_gather_equal_to_the_dense_path() {
    let (program, family) = fixture();
    let gains: Vec<usize> = (0..program.operators.len()).filter(|&o| program.operators[o].name.ends_with(".gain")).collect();
    assert_eq!(gains.len(), 5, "two norms per block and the final norm");
    for &g in &gains {
        assert!(matches!(program.operators[g].body, OperatorBody::Diagonal { .. }), "{} is held as a diagonal", program.operators[g].name);
    }
    let feature = program.nodes.iter().position(|n| matches!(n, Node::Feature { .. })).expect("a token feature");
    assert!(program.gathered_tokens(feature, &family).is_some());
    let trace = program.execute(&family, true).expect("banded");
    // The one-hot rows are never formed: the feature's value and band hold no columns.
    assert_eq!(trace.values[feature].dim(), (family.rows, 0));
    assert_eq!(trace.bands.as_ref().expect("bands")[feature].dim(), (family.rows, 0));

    let dense = dense_path(&program, feature);
    assert!(dense.gathered_tokens(feature, &family).is_none());
    let reference = dense.execute(&family, true).expect("dense banded");
    assert_eq!(reference.values[feature].dim(), (family.rows, VOCAB));
    for node in (0..program.nodes.len()).filter(|n| *n != feature) {
        assert_eq!(trace.values[node], reference.values[node], "node {node}");
    }
    let output = trace.band(program.output).expect("band");
    assert!(output.iter().all(|r| r.is_finite()));
    // Unbanded, both paths are the same arithmetic.
    let plain = program.execute(&family, false).expect("plain");
    assert_eq!(plain.values[program.output], reference.values[program.output]);

    // Coding: the diagonal is shorter than the dense gain, and the message decodes to it.
    for &g in &gains {
        let (structure, reals) = program.operators[g].code_bits().expect("bits");
        let (dense_structure, dense_reals) = dense.operators[g].code_bits().expect("dense bits");
        assert_eq!(reals, dense_reals, "the same reals on the same lattice");
        assert!(structure < dense_structure, "{structure} against {dense_structure}");
    }
    let message = program.encode().expect("encodes");
    assert_eq!(message.len_bits(), program.code_bits().expect("bits"));
    let decoded = OperatorProgram::decode(&message, &program.declarations).expect("decodes");
    for (a, b) in decoded.operators.iter().zip(&program.operators) {
        assert_eq!(a.body, b.body);
    }

    // Derivatives: a gain's tangent as its diagonal row, and the embedding's tangent read by the
    // gather, against the dense path's matrix tangents.
    let gain = gains[0];
    let width = program.operators[gain].rows.width();
    let row = Array2::from_shape_fn((1, width), |(_, c)| noise(4000 + c));
    let embedding = program.nodes.iter().find_map(|n| match n {
        Node::Affine { terms, .. } if terms.iter().any(|(a, _)| *a == feature) => Some(terms[0].1),
        _ => None,
    }).expect("the embedding term");
    let shape = program.operators[embedding].matrix().dim();
    let de = Array2::from_shape_fn(shape, |(i, j)| noise(5000 + i * shape.1 + j));
    let structural = jvp(&program, &family, &plain, &[(gain, row.clone()), (embedding, de.clone())].into_iter().collect()).expect("jvp");
    let full = jvp(&dense, &family, &reference, &[(gain, Array2::from_diag(&row.row(0))), (embedding, de)].into_iter().collect()).expect("dense jvp");
    let scale = full.iter().fold(0.0_f64, |m, v| m.max(v.abs()));
    let worst = structural.iter().zip(&full).fold(0.0_f64, |m, (a, b)| m.max((a - b).abs()));
    assert!(worst <= 1e-12 * scale, "tangents differ by {worst:e} at scale {scale:e}");

    // Execution from a base trace: a gain moved to a coarser lattice, propagated incrementally.
    let mut coarse = program.clone();
    let OperatorBody::Diagonal { values, .. } = &program.operators[gain].body else { unreachable!() };
    let precision = super::precision::DeclaredPrecision::new(4).expect("precision");
    coarse.operators[gain] = std::sync::Arc::new(
        Operator::diag("coarse", program.operators[gain].rows.clone(), values.clone(), precision, Default::default()).expect("diag"),
    );
    let incremental = coarse.execute_incremental(&family, &program, &plain).expect("incremental");
    let again = coarse.execute(&family, false).expect("full");
    let worst = incremental.iter().zip(&again.values[coarse.output]).fold(0.0_f64, |m, (a, b)| m.max((a - b).abs()));
    assert!(worst <= 1e-9, "incremental and full execution differ by {worst:e}");
}

/// A Hugging Face directory whose `config.json` is a small Llama's with `extra` merged in.
fn hugging_face_config(name: &str, extra: serde_json::Value) -> std::path::PathBuf {
    let dir = std::env::temp_dir().join(format!("gam-mpd-hf-{name}-{}", std::process::id()));
    std::fs::create_dir_all(&dir).expect("dir");
    let mut config = serde_json::json!({
        "model_type": "llama", "hidden_act": "silu", "hidden_size": 8, "num_attention_heads": 2, "num_key_value_heads": 1,
        "intermediate_size": 12, "num_hidden_layers": 1, "rope_theta": 10000.0, "rms_norm_eps": 1e-6,
    });
    if let (Some(base), serde_json::Value::Object(more)) = (config.as_object_mut(), extra) {
        base.extend(more);
    }
    std::fs::write(dir.join("config.json"), config.to_string()).expect("config");
    dir
}

#[test]
fn the_importer_refuses_options_it_does_not_compute() {
    let refused = |name: &str, extra: serde_json::Value, expected: &str| {
        let dir = hugging_face_config(name, extra);
        let error = match hugging_face_language_model(&dir, 0..1) {
            Ok(_) => panic!("{name}: imported"),
            Err(error) => error,
        };
        std::fs::remove_dir_all(&dir).expect("removed");
        assert!(error.contains(expected), "{name}: {error}");
    };
    refused("rope", serde_json::json!({"rope_scaling": {"rope_type": "llama3", "factor": 32.0}}), "rope_scaling");
    refused("linear", serde_json::json!({"rope_scaling": {"type": "linear", "factor": 2.0}}), "rope_scaling");
    refused("window", serde_json::json!({"use_sliding_window": true, "sliding_window": 4}), "sliding");
    refused("mistral", serde_json::json!({"sliding_window": 4}), "sliding");
    refused("layers", serde_json::json!({"layer_types": ["sliding_attention"]}), "sliding");
    refused("partial", serde_json::json!({"partial_rotary_factor": 0.5}), "partial_rotary_factor");
    // A sharded checkpoint: its index names shards the importer does not read.
    let dir = hugging_face_config("sharded", serde_json::json!({}));
    std::fs::write(dir.join("model.safetensors.index.json"), "{\"weight_map\": {}}").expect("index");
    let error = hugging_face_language_model(&dir, 0..1).err().expect("refused");
    std::fs::remove_dir_all(&dir).expect("removed");
    assert!(error.contains("sharded"), "{error}");
    // A window that is declared and switched off, as Qwen2 writes it, is the full attention.
    let dir = hugging_face_config("off", serde_json::json!({"use_sliding_window": false, "sliding_window": 4, "rope_scaling": null}));
    let error = hugging_face_language_model(&dir, 0..1).err().expect("no weights");
    std::fs::remove_dir_all(&dir).expect("removed");
    assert!(!error.contains("sliding") && !error.contains("rope"), "{error}");
}

#[test]
fn cached_batch_refreshes_tokens_and_positions_without_changing_old_traces() {
    use super::operator_program::SlotValues;
    let (program, mut family) = fixture();
    for device in devices() {
        let lowered = DeviceProgram::compile(&device, &program).expect("compile");
        let old = lowered.forward(&family).expect("old");
        let old_logits = lowered.logits(&old, 0, family.rows).expect("logits");
        if let SlotValues::Tokens(tokens) = &mut family.slots[0] { tokens[0] = (tokens[0] + 1) % VOCAB as u32; }
        for p in &mut family.layout.as_mut().expect("layout").position { *p += 3; }
        let fresh = DeviceProgram::compile(&device, &program).expect("fresh compile");
        let cached_trace = lowered.forward(&family).expect("updated");
        let fresh_trace = fresh.forward(&family).expect("fresh");
        assert_eq!(lowered.logits(&cached_trace, 0, family.rows).expect("cached logits"), fresh.logits(&fresh_trace, 0, family.rows).expect("fresh logits"));
        assert_eq!(lowered.logits(&old, 0, family.rows).expect("old still valid"), old_logits);
    }
}
