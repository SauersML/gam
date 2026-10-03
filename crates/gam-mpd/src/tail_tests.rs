#![cfg(test)]
//! The decoder tail against the imported program of the same checkpoint: the same logits, and its
//! pullback the program's own reverse pass.

use super::derivatives::vjp;
use super::import::hugging_face_language_model;
use super::masked::Head;
use super::operator_program::{FamilyInputs, Node, SequenceLayout, SlotValues};
use super::tail::DecoderTail;
use ndarray::Array2;
use std::path::Path;

fn noise(seed: usize) -> f64 {
    let mut x = (seed as u64).wrapping_mul(0x9E37_79B9_7F4A_7C15) ^ 0xD1B5_4A32_D192_ED03;
    x ^= x >> 33;
    x = x.wrapping_mul(0xFF51_AFD7_ED55_8CCD);
    x ^= x >> 33;
    (x >> 11) as f64 / (1u64 << 52) as f64 - 1.0
}

const D: usize = 8;
const MLP: usize = 12;
const VOCAB: usize = 11;
const KV: usize = 4;

/// A two-block Qwen2-shaped checkpoint (two heads sharing one key/value head, q/k/v biases, tied
/// readout), written as `config.json` and an F32 `model.safetensors`.
fn checkpoint(dir: &Path) {
    let mut tensors: Vec<(String, Vec<usize>)> = vec![("model.embed_tokens.weight".into(), vec![VOCAB, D]), ("model.norm.weight".into(), vec![D])];
    for l in 0..2 {
        let p = |s: &str| format!("model.layers.{l}.{s}");
        tensors.extend([
            (p("input_layernorm.weight"), vec![D]),
            (p("self_attn.q_proj.weight"), vec![D, D]),
            (p("self_attn.q_proj.bias"), vec![D]),
            (p("self_attn.k_proj.weight"), vec![KV, D]),
            (p("self_attn.k_proj.bias"), vec![KV]),
            (p("self_attn.v_proj.weight"), vec![KV, D]),
            (p("self_attn.v_proj.bias"), vec![KV]),
            (p("self_attn.o_proj.weight"), vec![D, D]),
            (p("post_attention_layernorm.weight"), vec![D]),
            (p("mlp.gate_proj.weight"), vec![MLP, D]),
            (p("mlp.up_proj.weight"), vec![MLP, D]),
            (p("mlp.down_proj.weight"), vec![D, MLP]),
        ]);
    }
    let (mut header, mut data) = (serde_json::Map::new(), Vec::new());
    for (k, (name, shape)) in tensors.iter().enumerate() {
        let n: usize = shape.iter().product();
        let begin = data.len();
        for i in 0..n {
            let v = if shape.len() == 1 && name.ends_with("norm.weight") { 1.0 + 0.3 * noise(1000 * k + i) } else { 0.6 * noise(1000 * k + i) };
            data.extend_from_slice(&(v as f32).to_le_bytes());
        }
        header.insert(name.clone(), serde_json::json!({"dtype": "F32", "shape": shape, "data_offsets": [begin, data.len()]}));
    }
    let header = serde_json::to_vec(&serde_json::Value::Object(header)).expect("header");
    let mut file = (header.len() as u64).to_le_bytes().to_vec();
    file.extend_from_slice(&header);
    file.extend_from_slice(&data);
    std::fs::write(dir.join("model.safetensors"), file).expect("weights");
    let config = serde_json::json!({
        "model_type": "qwen2", "hidden_act": "silu", "hidden_size": D, "num_attention_heads": 2, "num_key_value_heads": 1,
        "intermediate_size": MLP, "num_hidden_layers": 2, "rope_theta": 10000.0, "rms_norm_eps": 1e-6, "tie_word_embeddings": true,
    });
    std::fs::write(dir.join("config.json"), config.to_string()).expect("config");
}

#[test]
fn the_tail_is_the_imported_program_forward_and_backward() {
    let dir = std::env::temp_dir().join(format!("gam_mpd_tail_{}", std::process::id()));
    std::fs::create_dir_all(&dir).expect("dir");
    checkpoint(&dir);
    let (program, _) = hugging_face_language_model(&dir, 1..2).expect("imports");
    let tail = DecoderTail::new(&dir, 1, 1 << 20).expect("tail");
    // Two sequences (3 and 4 rows), positions from 0.
    let rows = 7;
    let x = Array2::from_shape_fn((rows, D), |(r, c)| noise(50 + D * r + c));
    let inputs = FamilyInputs {
        rows,
        slots: vec![SlotValues::Raw(x.clone())],
        layout: Some(SequenceLayout { sequence: vec![0, 0, 0, 1, 1, 1, 1], position: vec![0, 1, 2, 0, 1, 2, 3] }),
    };
    let scored = vec![false, true, true, false, false, true, true];
    let trace = program.execute(&inputs, false).expect("executes");
    let reference = &trace.values[program.output];
    let logits = tail.logits(&inputs, &x, &scored).expect("logits");
    let largest = reference.iter().fold(0.0_f64, |m, v| m.max(v.abs()));
    for r in 0..rows {
        for c in 0..VOCAB {
            let expected = if scored[r] { reference[[r, c]] } else { 0.0 };
            assert!((logits[[r, c]] - expected).abs() <= 1e-12 * largest, "logit ({r}, {c}): {} against {expected}", logits[[r, c]]);
        }
    }
    let cotangent = Array2::from_shape_fn((rows, VOCAB), |(r, c)| if scored[r] { noise(900 + VOCAB * r + c) } else { 0.0 });
    let back = vjp(&program, &inputs, &trace, cotangent.clone()).expect("vjp");
    let raw = program.nodes.iter().position(|n| matches!(n, Node::Raw { .. })).expect("the raw input");
    let expected = back[raw].clone().expect("a cotangent at the input");
    let pulled = tail.pullback(&inputs, &x, &scored, &cotangent).expect("pullback");
    let largest = expected.iter().fold(0.0_f64, |m, v| m.max(v.abs()));
    for r in 0..rows {
        for c in 0..D {
            assert!((pulled[[r, c]] - expected[[r, c]]).abs() <= 1e-11 * largest, "pullback ({r}, {c}): {} against {}", pulled[[r, c]], expected[[r, c]]);
        }
    }
    std::fs::remove_dir_all(&dir).expect("cleanup");
}
