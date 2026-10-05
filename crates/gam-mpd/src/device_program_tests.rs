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

use super::device_program::DeviceProgram;
use super::import::{hugging_face_language_model, import_language_model};
use super::operator_program::{FamilyInputs, OperatorProgram, SequenceLayout};
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
    (imported.program, imported.family)
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
        for n in (0..trace.len()).filter(|n| trace.has(*n)) {
            let got = device.download(trace.value(n).expect("value")).expect("download");
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

#[test]
fn frozen_prefix_forwards_and_reverses_like_the_whole_program() {
    let (program, family) = fixture();
    let trainable = program.operators.iter().position(|op| op.name == "blocks.1.c_fc").expect("a layer-1 map");
    let embedding = program.operators.iter().position(|op| op.name == "blocks.0.c_fc").expect("a layer-0 map");
    let seed = Array2::from_shape_fn((family.rows, VOCAB), |(r, c)| noise(6100 + r * VOCAB + c));
    for device in devices() {
        let mut lowered = DeviceProgram::compile_values(&device, &program).expect("values");
        lowered.prepare_dense_parameters(&[trainable]).expect("trainable");
        let frozen = lowered.freeze(&family, BTreeMap::new(), &[trainable], &[]).expect("freeze");
        assert!(frozen.is_frozen(0) && !frozen.is_frozen(program.output));
        let whole = lowered.forward(&family).expect("whole");
        let part = lowered.forward_frozen(&frozen).expect("frozen");
        let output = |t: &super::device_program::DeviceTrace| device.download(t.value(program.output).expect("output")).expect("download");
        assert_eq!(output(&whole), output(&part), "{}", device.name());
        let seeds = || BTreeMap::from([(program.output, device.upload(seed.view()).expect("seed"))]);
        let (_, a) = lowered.vjp_values_dense(&whole, seeds(), &[], &[trainable], Arithmetic::F64).expect("whole reverse");
        let (_, b) = lowered.vjp_values_dense(&part, seeds(), &[], &[trainable], Arithmetic::F64).expect("frozen reverse");
        assert_eq!(device.download(&a[&trainable]).expect("a"), device.download(&b[&trainable]).expect("b"), "{}", device.name());
        // A training step moves only the trainable map: the frozen values still hold.
        let moved = device.download(lowered.dense_parameter(trainable).expect("parameter")).expect("download") * 0.5;
        lowered.replace_dense_parameter(trainable, device.upload(moved.view()).expect("upload")).expect("replace");
        let (whole, part) = (lowered.forward(&family).expect("whole"), lowered.forward_frozen(&frozen).expect("frozen"));
        assert_eq!(output(&whole), output(&part), "{}: after a step", device.name());
        // Any other map changing makes them stale.
        lowered.prepare_dense_parameters(&[embedding]).expect("another");
        assert!(lowered.forward_frozen(&frozen).is_err());
    }
}
