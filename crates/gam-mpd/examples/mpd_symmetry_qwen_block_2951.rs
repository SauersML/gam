//! Symmetries of one Qwen3 block and of its vocabulary, with nothing declared (#2951).
//!
//! `mpd_symmetry_qwen_block_2951 EXPORT_DIR [LAYER]`
//!
//! `EXPORT_DIR` is what `bench/mpd_symmetry_export_2951.py qwen3 ... --config ...` writes: per layer the
//! attention tensors and gains, and the tied token embedding `embed`, as raw little-endian
//! float64 with shapes in `export.json`.
//!
//! 1. **The block's commutant.** The joint operators of layer `LAYER` (default 0) are the
//!    rank-two query/key plane operators `A_hj`, `B_hj` and the value/output maps `C_h`
//!    ([`query_key_operators`], [`value_output_operators`], with the input norm's gain folded).
//!    [`operator_commutant`] reads their commutant on the residual stream from factors, with no
//!    `d² × d²` system. A positive control runs the same members pinched to a hidden split of the
//!    stream (`P A P + P⊥ A P⊥`, `P` a seeded half-rank projector), where the commutant must be
//!    exactly two-dimensional.
//! 2. **The vocabulary.** [`candidate_subdomains`] refines the tied embedding's token domain by
//!    its band-certified invariants and reads the twin classes. The embedding is the only
//!    per-token table a tied Qwen3 reads (it is also the unembedding), so a permutation of twins
//!    leaves every stored tensor unchanged and is an exact symmetry of the network function.
//!
//! JSON on stdout.

use gam_linalg::faer_ndarray::{FaerQr, fast_ab, fast_abt, fast_atb};
use gam_mpd::attention::{
    AffineProjection, AttentionGeometry, NativeAttention, RotaryEmbedding, RotaryPairing,
};
use gam_mpd::joint_operators::{FactoredOperator, query_key_operators, value_output_operators};
use gam_mpd::symmetry::{OperatorCommutant, candidate_subdomains, operator_commutant};
use ndarray::{Array1, Array2, Axis};
use rand::rngs::StdRng;
use rand::{RngExt, SeedableRng};
use serde_json::json;
use std::io::Read;
use std::path::{Path, PathBuf};
use std::time::Instant;

fn read_tensor(dir: &Path, record: &serde_json::Value, name: &str) -> Result<Array2<f64>, String> {
    let shape = record["files"][name]["shape"]
        .as_array()
        .ok_or_else(|| format!("export.json: no shape for {name}"))?
        .iter()
        .map(|v| v.as_u64().map(|v| v as usize).ok_or_else(|| format!("export.json: {name} shape")))
        .collect::<Result<Vec<_>, _>>()?;
    let [rows, cols] = shape[..] else { return Err(format!("{name}: shape {shape:?} is not two axes")) };
    // Streamed in 8 MiB chunks, so the 1.2 GB embedding is never held twice.
    let path = dir.join(format!("{name}.f64"));
    let mut file = std::fs::File::open(&path).map_err(|error| format!("{}: {error}", path.display()))?;
    let length = file.metadata().map_err(|error| error.to_string())?.len() as usize;
    if length != rows * cols * 8 {
        return Err(format!("{}: {length} bytes for {rows}×{cols}", path.display()));
    }
    let mut values = Vec::with_capacity(rows * cols);
    let mut buffer = vec![0u8; 1 << 23];
    while values.len() < rows * cols {
        let take = buffer.len().min((rows * cols - values.len()) * 8);
        file.read_exact(&mut buffer[..take]).map_err(|error| format!("{}: {error}", path.display()))?;
        values.extend(buffer[..take].chunks_exact(8).map(|c| f64::from_le_bytes([c[0], c[1], c[2], c[3], c[4], c[5], c[6], c[7]])));
    }
    Array2::from_shape_vec((rows, cols), values).map_err(|error| error.to_string())
}

fn summary(commutant: &OperatorCommutant, seconds: f64) -> serde_json::Value {
    json!({
        "seconds": seconds,
        "members": commutant.members,
        "exact_dim": commutant.exact_dim,
        "blocks": commutant.blocks,
        "eigenvalue_band": commutant.eigenvalue_band,
        "separation_over_band": commutant.separation_over_band,
        "split_coupling": commutant.split_coupling,
        "split_bound": commutant.split_bound,
        "attempts": commutant.attempts,
    })
}

fn main() -> Result<(), String> {
    let mut args = std::env::args().skip(1);
    let dir = PathBuf::from(args.next().ok_or("usage: mpd_symmetry_qwen_block_2951 EXPORT_DIR [LAYER]")?);
    let layer: usize = args.next().map_or(Ok(0), |v| v.parse().map_err(|e| format!("LAYER: {e}")))?;
    let text = std::fs::read_to_string(dir.join("export.json")).map_err(|e| e.to_string())?;
    let record: serde_json::Value = serde_json::from_str(&text).map_err(|e| e.to_string())?;
    let field = |name: &str| record[name].as_u64().map(|v| v as usize).ok_or(format!("export.json: {name}"));
    let real = |name: &str| record[name].as_f64().ok_or(format!("export.json: {name}"));
    let (heads, kv_heads, head_dim) = (field("heads")?, field("kv_heads")?, field("d_head")?);
    let tensor = |short: &str| read_tensor(&dir, &record, &format!("L{layer}.{short}"));
    let (w_q, w_k, w_v, w_o) = (tensor("W_Q")?, tensor("W_K")?, tensor("W_V")?, tensor("W_O")?);
    let row = |m: Array2<f64>| -> Array1<f64> { m.row(0).to_owned() };
    let (norm, q_norm, k_norm) = (row(tensor("norm")?), row(tensor("q_norm")?), row(tensor("k_norm")?));
    let model_dim = w_q.ncols();
    let geometry = AttentionGeometry { model_dim, n_heads: heads, n_kv_heads: kv_heads, head_dim };
    let theta = real("rope_theta")?;
    let rotary = RotaryEmbedding {
        pairing: RotaryPairing::HalfSplit,
        inverse_frequencies: (0..head_dim / 2).map(|i| theta.powf(-(2.0 * i as f64) / head_dim as f64)).collect(),
        attention_scaling: 1.0,
    };
    let affine = |weight: Array2<f64>| {
        let rows = weight.nrows();
        AffineProjection { weight, bias: Array1::zeros(rows) }
    };
    let native = NativeAttention::new(
        geometry,
        rotary,
        1.0 / (head_dim as f64).sqrt(),
        affine(w_q),
        affine(w_k),
        affine(w_v),
        affine(w_o),
    )
    .map_err(|e| e.to_string())?
    .with_query_key_norm(real("rms_norm_eps")?, q_norm, k_norm)
    .map_err(|e| e.to_string())?;
    let qk = query_key_operators(&native, Some(norm.view())).map_err(|e| format!("{e:?}"))?;
    let ov = value_output_operators(&native, Some(norm.view())).map_err(|e| format!("{e:?}"))?;
    let mut family: Vec<&FactoredOperator> = Vec::new();
    for h in 0..heads {
        for j in 0..qk.planes() {
            family.push(qk.cosine(h, j));
            family.push(qk.sine(h, j));
        }
        if let Some(pass) = qk.pass_through(h) {
            family.push(pass);
        }
        family.push(ov.head(h));
    }
    let started = Instant::now();
    let commutant = operator_commutant(&family, 0).map_err(|e| e.to_string())?;
    let block = summary(&commutant, started.elapsed().as_secs_f64());
    // Positive control: the members pinched to a seeded split P ⊕ P⊥.
    let width = family[0].width();
    let mut rng = StdRng::seed_from_u64(2951);
    let draw = Array2::from_shape_fn((width, width / 2), |_| rng.random_range(-1.0..1.0));
    let (q, _) = draw.qr().map_err(|e| e.to_string())?;
    let projector = fast_abt(&q, &q);
    let complement = Array2::<f64>::eye(width) - &projector;
    let pinched: Vec<FactoredOperator> = family
        .iter()
        .map(|m| {
            // P L Rᵀ P + P⊥ L Rᵀ P⊥ = [P L, P⊥ L] [P R, P⊥ R]ᵀ.
            let (left, right) = (m.left().to_owned(), m.right().to_owned());
            let stack = |a: Array2<f64>, b: Array2<f64>| ndarray::concatenate(Axis(1), &[a.view(), b.view()]).expect("same rows");
            FactoredOperator::new(
                stack(fast_ab(&projector, &left), fast_ab(&complement, &left)),
                stack(fast_ab(&projector, &right), fast_ab(&complement, &right)),
            )
            .expect("finite factors")
        })
        .collect();
    let started = Instant::now();
    let control = operator_commutant(&pinched.iter().collect::<Vec<_>>(), 0).map_err(|e| e.to_string())?;
    let control_summary = summary(&control, started.elapsed().as_secs_f64());
    // Each recovered block's share inside the planted half (1 or 0 when the split is found).
    let captured: Vec<f64> = (0..control.isotypic.blocks.len())
        .map(|index| {
            let columns = control.isotypic.block_columns(index).to_owned();
            fast_atb(&q, &columns).iter().map(|v| v * v).sum::<f64>() / columns.ncols() as f64
        })
        .collect();
    drop(pinched);
    // The vocabulary.
    let started = Instant::now();
    let embed = read_tensor(&dir, &record, "embed")?;
    let subdomains = candidate_subdomains(&[embed.view()]).map_err(|e| e.to_string())?;
    let vocabulary_seconds = started.elapsed().as_secs_f64();
    let class_sizes: Vec<usize> = subdomains.twins.iter().map(|c| c.len()).collect();
    let open: Vec<usize> = subdomains.open_cells().iter().map(|c| c.len()).collect();
    let report = json!({
        "export": dir.display().to_string(),
        "layer": layer,
        "model_dim": model_dim,
        "block": block,
        "pinched_control": {
            "commutant": control_summary,
            "planted_share_per_block": captured,
        },
        "vocabulary": {
            "tokens": subdomains.n,
            "seconds": vocabulary_seconds,
            "rounds": subdomains.rounds,
            "cells": subdomains.cells.len(),
            "twin_classes": subdomains.twins.len(),
            "twin_class_sizes": class_sizes,
            "twinned_tokens": class_sizes.iter().sum::<usize>(),
            "quotient_size": subdomains.quotient_size(),
            "log2_twin_group_order": subdomains.log2_twin_order(),
            "max_twin_defect": subdomains.max_twin_defect,
            "exact_band": subdomains.exact_band,
            "open_cell_sizes": open,
            "first_class_head": subdomains.twins.first().map(|c| c.iter().take(6).collect::<Vec<_>>()),
        },
    });
    println!("{}", serde_json::to_string_pretty(&report).map_err(|e| e.to_string())?);
    Ok(())
}
