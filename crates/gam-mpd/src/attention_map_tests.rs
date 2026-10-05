use super::*;
use crate::import::import_language_model;
use serde_json::json;
use std::{
    path::PathBuf,
    sync::atomic::{AtomicU64, Ordering},
};

struct Export(PathBuf);
impl Drop for Export {
    fn drop(&mut self) {
        std::fs::remove_dir_all(&self.0).expect("remove temporary export fixture");
    }
}
fn export(layers: usize, heads: usize, kv: usize, qk: bool, gated: bool) -> Export {
    static SERIAL: AtomicU64 = AtomicU64::new(0);
    let dir = std::env::temp_dir().join(format!(
        "mpd-attention-map-{}-{}",
        std::process::id(),
        SERIAL.fetch_add(1, Ordering::Relaxed)
    ));
    std::fs::create_dir(&dir).expect("valid attention mapping fixture");
    let mut files = serde_json::Map::new();
    let mut write = |name: &str, rows: usize, cols: usize, gain: bool| {
        let values: Vec<f64> = (0..rows * cols)
            .map(|i| {
                if gain {
                    1. + 0.01 * (i as f64)
                } else {
                    ((i * 7 + 3) as f64).sin() * 0.2
                }
            })
            .collect();
        // HF exports contain exactly widened f32 checkpoint literals.
        let bytes: Vec<u8> = values
            .iter()
            .flat_map(|x| f64::from(*x as f32).to_le_bytes())
            .collect();
        std::fs::write(dir.join(format!("{name}.f64")), bytes)
            .expect("valid attention mapping fixture");
        files.insert(name.into(), json!({"shape":[rows,cols]}));
    };
    // Odd head width and model width unequal to query_heads*head_width exercise
    // generic GQA dimensions rather than the old Decoder's reshaping assumptions.
    let (d, dh, hidden, vocab) = (6, 3, 9, 7);
    write("wte", vocab, d, false);
    for layer in 0..layers {
        for part in ["rms1.gain", "rms2.gain"] {
            write(&format!("blocks.{layer}.{part}"), 1, d, true);
        }
        for (part, rows, cols) in [
            ("attn.q_proj", heads * dh, d),
            ("attn.k_proj", kv * dh, d),
            ("attn.v_proj", kv * dh, d),
            ("attn.o_proj", d, heads * dh),
            ("mlp.c_fc", hidden, d),
            ("mlp.down_proj", d, hidden),
        ] {
            write(&format!("blocks.{layer}.{part}"), rows, cols, false);
        }
        if qk {
            for part in ["attn.q_norm.gain", "attn.k_norm.gain"] {
                write(&format!("blocks.{layer}.{part}"), 1, dh, true);
            }
        }
        if gated {
            write(&format!("blocks.{layer}.mlp.gate_proj"), hidden, d, false);
        }
    }
    write("final_norm.gain", 1, d, true);
    drop(write);
    std::fs::write(
        dir.join("tokens.f64"),
        [1_f64, 2., 3.]
            .iter()
            .flat_map(|x| x.to_le_bytes())
            .collect::<Vec<_>>(),
    )
    .expect("valid attention mapping fixture");
    files.insert("tokens".into(), json!({"shape":[1,3]}));
    let record = json!({"config":{"n_layers":layers,"d_model":d,"n_heads":heads,"n_kv_heads":kv,"head_dim":dh,"vocab":vocab,"norm":"rms","norm_eps":0.00001,"rope_theta":10000.,"rotary_dims":2,"rope_pairing":"rotate_half","qk_norm":qk,"mlp_gated":gated,"mlp_act":if gated {"silu"}else {"gelu_tanh"},"tied_embeddings":true},"files":files});
    std::fs::write(
        dir.join("export.json"),
        serde_json::to_vec(&record).expect("valid attention mapping fixture"),
    )
    .expect("valid attention mapping fixture");
    Export(dir)
}
#[test]
fn imported_four_layer_maps_all_heads_without_rewriting_native_graph() {
    let dir = export(4, 2, 2, false, false);
    let imported = import_language_model(&dir.0, 1, 3).expect("valid attention mapping fixture");
    let p = &imported.program;
    let nodes = p.nodes.clone();
    let operators = p.operators.clone();
    let output = p.output;
    let native = Artifact::native(p).expect("valid attention mapping fixture");
    for layer in 0..4 {
        let m = AttentionLayerMap::of(p, layer).expect("valid attention mapping fixture");
        assert_eq!(m.native_layer, layer);
        assert_eq!(m.heads.len(), 2);
        for h in &m.heads {
            assert_eq!(h.head, h.kv_head);
            assert_eq!(h.raw_query, h.query);
            assert_eq!(h.raw_key, h.key);
            assert!(h.rotary.is_some());
        }
        let bound = m.bind(&native).expect("valid attention mapping fixture");
        assert_eq!(bound.blocks[0].native_reads, m.native_reads());
        assert_eq!(bound.blocks[0].native_write, m.output);
        assert_eq!(bound.places, native.places);
        assert_eq!(p.nodes, nodes);
        assert_eq!(p.operators, operators);
        assert_eq!(p.output, output);
    }
}
#[test]
fn qwen_gqa_normalized_lineage_explicit_layer_id_and_copy_roundtrip() {
    let dir = export(1, 4, 2, true, true);
    let mut imported =
        import_language_model(&dir.0, 1, 3).expect("valid attention mapping fixture");
    for op in &mut imported.program.operators {
        let renamed = op.name.replace("blocks.0.", "blocks.27.");
        std::sync::Arc::make_mut(op).name = renamed;
    }
    let p = &imported.program;
    let m = AttentionLayerMap::of(p, 27).expect("valid attention mapping fixture");
    assert_eq!(m.native_layer, 27);
    assert!(AttentionLayerMap::of(p, 0).is_err());
    assert_eq!(m.heads.len(), 4);
    assert_eq!(
        m.heads.iter().map(|h| h.kv_head).collect::<Vec<_>>(),
        vec![0, 0, 1, 1]
    );
    for h in &m.heads {
        assert_ne!(h.query, h.raw_query);
        assert_ne!(h.key, h.raw_key);
        assert!(h.query_norm.is_some());
        assert_eq!(
            h.rotary
                .as_ref()
                .expect("valid attention mapping fixture")
                .dims,
            2
        );
        assert!(h.causal);
    }
    assert_eq!(m.heads[0].key, m.heads[1].key);
    assert_eq!(m.heads[0].value, m.heads[1].value);
    let base = Artifact::native(p)
        .expect("valid attention mapping fixture")
        .f32_literals()
        .expect("valid attention mapping fixture");
    for head in 0..m.heads.len() {
        let changed = m
            .bind(
                &base
                    .derive(
                        m.heads[head].output_operator,
                        m.copy_law(head).expect("valid attention mapping fixture"),
                        0.25,
                        vec![],
                    )
                    .expect("valid attention mapping fixture"),
            )
            .expect("valid attention mapping fixture")
            .f32_literals()
            .expect("valid attention mapping fixture");
        changed
            .validate_coverage(p)
            .expect("valid attention mapping fixture");
        assert_eq!(changed.program.nodes, p.nodes);
        assert_eq!(changed.places, base.places);
        let decoded = Artifact::from_bytes(
            &changed.to_bytes().expect("valid attention mapping fixture"),
            &p.declarations,
        )
        .expect("valid attention mapping fixture");
        decoded
            .validate_coverage(p)
            .expect("valid attention mapping fixture");
        assert_eq!(decoded.places, base.places);
        assert_eq!(decoded.blocks[0].native_reads, vec![m.skip, m.normed_input]);
        let edits = [
            m.heads[head].raw_query,
            m.heads[head].query,
            m.heads[head].key,
            m.heads[head].value,
            m.heads[head].read,
            m.output,
        ];
        for edit in edits {
            assert_eq!(decoded.place(edit), Some(edit));
            let run = |program: &OperatorProgram| {
                program
                    .execute_edited(&imported.family, |node, value, _| {
                        if node == edit {
                            *value *= 0.5;
                        }
                        Ok(())
                    })
                    .expect("valid attention mapping fixture")
            };
            assert_eq!(
                run(&changed.program).values[p.output],
                run(&decoded.program).values[p.output]
            );
        }
    }
}
#[test]
fn rejects_query_bypass_wrong_gqa_or_ambiguous_output() {
    let dir = export(1, 4, 2, true, true);
    let imported = import_language_model(&dir.0, 1, 3).expect("valid attention mapping fixture");
    let p = &imported.program;
    let m = AttentionLayerMap::of(p, 0).expect("valid attention mapping fixture");
    let mut bypass = p.clone();
    let Node::Attend { query, .. } = &mut bypass.nodes[m.heads[0].read] else {
        panic!("fixture")
    };
    *query = m.heads[0].raw_query;
    assert!(
        AttentionLayerMap::of(&bypass, 0)
            .unwrap_err()
            .contains("bypasses")
    );
    let mut wrong = p.clone();
    let Node::Attend { value, .. } = &mut wrong.nodes[m.heads[0].read] else {
        panic!("fixture")
    };
    *value = m.heads[2].value;
    assert!(AttentionLayerMap::of(&wrong, 0).is_err());
    let mut duplicate = p.clone();
    duplicate
        .operators
        .push(p.operators[m.heads[0].output_operator].clone());
    assert!(AttentionLayerMap::of(&duplicate, 0).is_err());
    let mut branch = p.clone();
    let Node::Affine { terms, .. } = &mut branch.nodes[m.output] else {
        panic!("fixture")
    };
    terms.push((m.skip, m.skip_operator));
    assert!(AttentionLayerMap::of(&branch, 0).is_err());
}
#[test]
fn rejects_orphan_final_gain_and_extra_projection_reader() {
    let dir = export(1, 4, 2, true, true);
    let imported = import_language_model(&dir.0, 1, 3).expect("valid attention mapping fixture");
    let p = &imported.program;
    let m = AttentionLayerMap::of(p, 0).expect("valid attention mapping fixture");
    let mut orphan = p.clone();
    orphan.output = m.output;
    assert!(AttentionLayerMap::of(&orphan, 0).is_err());
    let mut extra = p.clone();
    extra.nodes.push(Node::RmsNorm {
        input: m.heads[0].raw_query,
        epsilon: 0.001,
    });
    assert!(AttentionLayerMap::of(&extra, 0).is_err());
}
