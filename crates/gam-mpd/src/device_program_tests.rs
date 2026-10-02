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
use super::import::import_language_model;
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
    let dir = std::env::temp_dir().join(format!("gam-mpd-device-program-{}-{:?}", std::process::id(), std::thread::current().id()));
    std::fs::create_dir_all(&dir).expect("dir");
    let mut files = serde_json::Map::new();
    let mut seed = 0;
    let mut random = |rows: usize, cols: usize, scale: f64| {
        seed += 1000;
        let base = seed;
        Array2::from_shape_fn((rows, cols), |(i, j)| scale * noise(base + i * cols + j))
    };
    write(&dir, &mut files, "wte", &random(VOCAB, D, 1.0));
    for l in 0..2 {
        let p = format!("blocks.{l}.");
        write(&dir, &mut files, &format!("{p}attn.q_proj"), &random(HEADS * HEAD_DIM, D, 0.5));
        write(&dir, &mut files, &format!("{p}attn.k_proj"), &random(HEAD_DIM, D, 0.5));
        write(&dir, &mut files, &format!("{p}attn.v_proj"), &random(HEAD_DIM, D, 0.5));
        write(&dir, &mut files, &format!("{p}attn.o_proj"), &random(D, HEADS * HEAD_DIM, 0.5));
        write(&dir, &mut files, &format!("{p}mlp.c_fc"), &random(HIDDEN, D, 0.5));
        write(&dir, &mut files, &format!("{p}mlp.down_proj"), &random(D, HIDDEN, 0.5));
        write(&dir, &mut files, &format!("{p}rms1.gain"), &(random(1, D, 0.2) + 1.0));
        write(&dir, &mut files, &format!("{p}rms2.gain"), &(random(1, D, 0.2) + 1.0));
    }
    write(&dir, &mut files, "final_norm.gain", &(random(1, D, 0.2) + 1.0));
    let tokens = Array2::from_shape_fn((SEQUENCES, CONTEXT), |(s, t)| ((noise(7 * s + t + 99) + 1.0) * 6.4) as usize as f64 % VOCAB as f64);
    write(&dir, &mut files, "tokens", &tokens);
    let record = serde_json::json!({
        "config": {
            "d_model": D, "n_layers": 2, "n_heads": HEADS, "n_kv_heads": 1, "head_dim": HEAD_DIM,
            "vocab": VOCAB, "rope_theta": 10000.0, "rope_pairing": "rotate_half", "norm_eps": 1e-6,
            "mlp_act": "gelu_tanh", "tied_embeddings": true,
        },
        "files": files,
    });
    std::fs::write(dir.join("export.json"), record.to_string()).expect("written");
    let imported = import_language_model(&dir, SEQUENCES, CONTEXT).expect("imported");
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
