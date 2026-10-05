//! Shared test fixtures for `gam_mpd` (#2951).
//!
#![cfg(test)]

use std::path::{Path, PathBuf};

/// The ledger a test's kernels reserve on when the test does not assert on
/// reservations: a private governor, so no test draws on or reads the process-wide
/// ledger. Its budget, `2^34` bytes, is far above any fixture's footprint and far
/// below what a planted refusal requests; a test that asserts on its reservations
/// builds its own `MemoryGovernor::with_budget_bytes` instead.
pub fn test_governor() -> &'static gam_runtime::resource::MemoryGovernor {
    static GOVERNOR: std::sync::OnceLock<gam_runtime::resource::MemoryGovernor> = std::sync::OnceLock::new();
    GOVERNOR.get_or_init(|| gam_runtime::resource::MemoryGovernor::with_budget_bytes(1 << 34))
}

/// A small random decoder export (`layers` layers, 2 heads of 4, MLP 16, vocabulary 11) with six
/// token rows of 12.
pub fn tiny_export(tag: &str, layers: usize) -> PathBuf {
    let dir = std::env::temp_dir().join(format!("gam_mpd_{tag}_{}", std::process::id()));
    std::fs::create_dir_all(&dir).expect("temp dir");
    let mut state = 0x2545_F491_4F6C_DD1Du64;
    let mut draw = |n: usize, scale: f64| -> Vec<f64> {
        (0..n)
            .map(|_| {
                state ^= state << 13;
                state ^= state >> 7;
                state ^= state << 17;
                ((state >> 11) as f64 / (1u64 << 53) as f64 - 0.5) * scale
            })
            .collect()
    };
    let mut files = serde_json::Map::new();
    let mut write = |dir: &Path, name: &str, shape: [usize; 2], values: Vec<f64>| {
        let bytes: Vec<u8> = values.iter().flat_map(|v| v.to_le_bytes()).collect();
        std::fs::write(dir.join(format!("{name}.f64")), bytes).expect("write tensor");
        files.insert(name.to_string(), serde_json::json!({"shape": shape}));
    };
    let (d, mlp, vocab, rows, context) = (8, 16, 11, 6, 12);
    write(&dir, "wte", [vocab, d], draw(vocab * d, 2.0));
    write(&dir, "final_norm.gain", [1, d], draw(d, 0.4).iter().map(|g| 1.0 + g).collect());
    for l in 0..layers {
        for (name, shape) in [("attn.q_proj", [d, d]), ("attn.k_proj", [d, d]), ("attn.v_proj", [d, d]), ("attn.o_proj", [d, d]), ("mlp.c_fc", [mlp, d]), ("mlp.down_proj", [d, mlp])] {
            write(&dir, &format!("blocks.{l}.{name}"), shape, draw(shape[0] * shape[1], 1.0));
        }
        for g in ["rms1", "rms2"] {
            write(&dir, &format!("blocks.{l}.{g}.gain"), [1, d], draw(d, 0.4).iter().map(|v| 1.0 + v).collect());
        }
    }
    let tokens: Vec<f64> = draw(rows * context, 1.0).iter().map(|x| ((x + 0.5) * vocab as f64).floor().min(vocab as f64 - 1.0)).collect();
    write(&dir, "tokens", [rows, context], tokens);
    let record = serde_json::json!({
        "config": {"d_model": d, "n_layers": layers, "n_heads": 2, "n_kv_heads": 2, "head_dim": 4, "d_mlp": mlp, "vocab": vocab, "rope_theta": 10000.0,
                   "rope_pairing": "rotate_half", "norm_eps": 1e-6, "mlp_act": "gelu_tanh", "tied_embeddings": true},
        "files": files,
    });
    std::fs::write(dir.join("export.json"), record.to_string()).expect("export.json");
    dir
}
