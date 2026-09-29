//! Implementation-gauge census of a Qwen3 checkpoint (#2951).
//!
//! Runs the `gam_mpd::gauge` detectors on every layer of the model
//! exported by `bench/mpd_gauge_census_2951.py` and reports, per layer and in total,
//! the parameter count, the certified orbit dimension (resolved / at most), the exact
//! null coordinates, and `real_coordinates_at_most`: the parameters minus the resolved
//! orbit dimension and the null coordinates. The difference is the part of the
//! checkpoint that is pure implementation convention.
//!
//! The families, their charges and the tied residual stream belong to
//! `gam_mpd::gauge_census` (`decoder_layer_census`,
//! `tied_residual_census`), which the MPD surface's `gauge_census` operation also runs.
//! This example only reads the exported checkpoint and writes the receipt.
//!
//! ```text
//! uv run --no-project --with numpy python bench/mpd_gauge_census_2951.py --out /tmp/census
//! cargo run --release -p gam-mpd --example mpd_gauge_census_2951 -- \
//!     --export /tmp/census --out experiments/issue-2951/receipts/gauge_census_Qwen3-0.6B-Base.json
//! ```

use gam_mpd::attention::{AttentionGeometry, RotaryEmbedding, RotaryPairing};
use gam_mpd::gauge::GaugeFamily;
use gam_mpd::gauge_census::{
    CensusCharge, DecoderLayerTensors, decoder_layer_census, tied_residual_census,
};
use gam_mpd::operators::DeclaredGauge;
use memmap2::Mmap;
use ndarray::{Array1, Array2, Axis};
use serde_json::{Map, Value, json};
use std::collections::BTreeMap;
use std::fs::File;
use std::path::{Path, PathBuf};
use std::process::ExitCode;
use std::time::Instant;


fn main() -> ExitCode {
    match run() {
        Ok(()) => ExitCode::SUCCESS,
        Err(error) => {
            println!("[mpd_gauge_census_2951] error: {error}");
            ExitCode::FAILURE
        }
    }
}

fn flag(args: &[String], name: &str) -> Result<PathBuf, String> {
    args.windows(2)
        .find(|pair| pair[0] == name)
        .map(|pair| PathBuf::from(&pair[1]))
        .ok_or_else(|| format!("missing {name}"))
}

fn float64(path: &Path) -> Result<(Vec<usize>, Vec<f64>), String> {
    let file = File::open(path).map_err(|err| format!("open {}: {err}", path.display()))?;
    // SAFETY: the exported array is opened read-only and never written through this
    // mapping for the lifetime of the read.
    let mmap = unsafe { Mmap::map(&file).map_err(|err| format!("mmap {}: {err}", path.display()))? };
    let (shape, data_off) = f8_header(&mmap, path)?;
    let count: usize = shape.iter().product();
    let end = data_off + 8 * count;
    if end != mmap.len() {
        return Err(format!("{} holds {} bytes; its header needs {end}", path.display(), mmap.len()));
    }
    let values = mmap[data_off..end]
        .chunks_exact(8)
        .map(|chunk| f64::from_le_bytes(chunk.try_into().expect("eight bytes")))
        .collect();
    Ok((shape, values))
}

/// The shape and data offset of a little-endian, C-order `<f8` `.npy` (format 1.0 or 2.0).
fn f8_header(bytes: &[u8], path: &Path) -> Result<(Vec<usize>, usize), String> {
    let bad = |what: &str| format!("{}: {what}", path.display());
    if bytes.len() < 10 || &bytes[..6] != b"\x93NUMPY" {
        return Err(bad("not an .npy file"));
    }
    let (len, start) = match bytes[6] {
        1 => (u16::from_le_bytes([bytes[8], bytes[9]]) as usize, 10),
        2 | 3 if bytes.len() >= 12 => (u32::from_le_bytes([bytes[8], bytes[9], bytes[10], bytes[11]]) as usize, 12),
        _ => return Err(bad("unsupported .npy version")),
    };
    let header = std::str::from_utf8(bytes.get(start..start + len).ok_or_else(|| bad("truncated header"))?)
        .map_err(|_| bad("header is not UTF-8"))?;
    if !header.contains("'descr': '<f8'") || !header.contains("'fortran_order': False") {
        return Err(bad("must hold C-order <f8 values"));
    }
    let open = header.find("'shape': (").ok_or_else(|| bad("no shape"))? + "'shape': (".len();
    let close = open + header[open..].find(')').ok_or_else(|| bad("no shape"))?;
    let shape = header[open..close]
        .split(',')
        .map(str::trim)
        .filter(|extent| !extent.is_empty())
        .map(|extent| extent.parse::<usize>().map_err(|_| bad("bad shape")))
        .collect::<Result<Vec<_>, _>>()?;
    Ok((shape, start + len))
}

fn matrix(path: &Path) -> Result<Array2<f64>, String> {
    let (shape, values) = float64(path)?;
    let [rows, cols] = shape[..] else {
        return Err(format!("{} must have two axes; it has {shape:?}", path.display()));
    };
    Array2::from_shape_vec((rows, cols), values).map_err(|err| format!("{}: {err}", path.display()))
}

fn vector(path: &Path) -> Result<Array1<f64>, String> {
    let (shape, values) = float64(path)?;
    if shape.len() != 1 {
        return Err(format!("{} must have one axis; it has {shape:?}", path.display()));
    }
    Ok(Array1::from(values))
}

fn usize_field(manifest: &Value, key: &str) -> Result<usize, String> {
    manifest[key]
        .as_u64()
        .map(|value| value as usize)
        .ok_or_else(|| format!("manifest.{key} missing"))
}

fn charge_json(charge: &CensusCharge) -> Value {
    json!({
        "parameters": charge.parameters,
        "orbit_resolved": charge.orbit_resolved,
        "orbit_at_most": charge.orbit_at_most,
        "null": charge.null,
        "real_coordinates_at_most": charge.real_coordinates_at_most(),
        "convention_fraction": charge.convention_fraction(),
    })
}

fn family_json(family: &GaugeFamily, charge: &CensusCharge) -> Value {
    let mut row = charge_json(charge);
    let fields = row.as_object_mut().expect("object");
    fields.insert("continuous".into(), json!(format!("{:?}", family.continuous)));
    fields.insert("discrete".into(), json!(format!("{:?}", family.discrete)));
    fields.insert(
        "declared_gauge".into(),
        json!(match family.declared_gauge() {
            Ok(Some(DeclaredGauge::Blocks(blocks))) => format!("Blocks(rank {}, {} blocks)", blocks.rank(), blocks.block_count()),
            Ok(Some(gauge)) => format!("{gauge:?}"),
            Ok(None) => "none".into(),
            Err(err) => format!("refused: {err:?}"),
        }),
    );
    row
}

fn run() -> Result<(), String> {
    let args: Vec<String> = std::env::args().collect();
    let export = flag(&args, "--export")?;
    let out = flag(&args, "--out")?;
    let manifest: Value = serde_json::from_str(
        &std::fs::read_to_string(export.join("manifest.json")).map_err(|err| format!("manifest: {err}"))?,
    )
    .map_err(|err| format!("manifest: {err}"))?;
    let d = usize_field(&manifest, "hidden_size")?;
    let layers = usize_field(&manifest, "num_hidden_layers")?;
    let n_heads = usize_field(&manifest, "num_attention_heads")?;
    let n_kv = usize_field(&manifest, "num_key_value_heads")?;
    let hd = usize_field(&manifest, "head_dim")?;
    let vocab = usize_field(&manifest, "vocab_size")?;
    let theta = manifest["rope_theta"].as_f64().ok_or("manifest.rope_theta missing")?;
    let eps = manifest["rms_norm_eps"].as_f64().ok_or("manifest.rms_norm_eps missing")?;
    if manifest["tie_word_embeddings"] != json!(true) {
        return Err("the census models a tied embedding/unembedding only".into());
    }
    let stored: BTreeMap<String, usize> = serde_json::from_value(manifest["stored_parameters"].clone())
        .map_err(|err| format!("manifest.stored_parameters: {err}"))?;
    let checkpoint_parameters: usize = stored.values().sum();

    let geometry = AttentionGeometry {
        model_dim: d,
        n_heads,
        n_kv_heads: n_kv,
        head_dim: hd,
    };
    // Qwen3's rotary: half-split planes, inv_freq_j = θ^(−2j/hd); only the frequencies'
    // distinctness and their range (0, π) enter the normed family's derivation.
    let rotary = RotaryEmbedding {
        pairing: RotaryPairing::HalfSplit,
        inverse_frequencies: (0..hd / 2).map(|j| theta.powf(-((2 * j) as f64) / hd as f64)).collect(),
        attention_scaling: 1.0,
    };
    let mut total = CensusCharge::default();
    let mut by_family: BTreeMap<&str, CensusCharge> = BTreeMap::new();
    let mut layer_rows = Vec::with_capacity(layers);
    let started = Instant::now();
    for layer in 0..layers {
        let dir = export.join(format!("layer_{layer:02}"));
        let load = |name: &str| matrix(&dir.join(format!("{name}.npy")));
        let loadv = |name: &str| vector(&dir.join(format!("{name}.npy")));
        let (q, k, v, o) = (load("q")?, load("k")?, load("v")?, load("o")?);
        let (gate, up, down) = (load("gate")?, load("up")?, load("down")?);
        let (input_norm, post_norm) = (loadv("input_norm")?, loadv("post_norm")?);
        let (q_norm, k_norm) = (loadv("q_norm")?, loadv("k_norm")?);
        let layer_parameters: usize = stored
            .iter()
            .filter(|(name, _)| name.starts_with(&format!("model.layers.{layer}.")))
            .map(|(_, count)| count)
            .sum();

        let census = decoder_layer_census(
            geometry,
            &rotary,
            1.0 / (hd as f64).sqrt(),
            eps,
            DecoderLayerTensors {
                query: q.view(),
                key: k.view(),
                value: v.view(),
                output: o.view(),
                query_key_norm: Some((q_norm.view(), k_norm.view())),
                gate: gate.view(),
                up: up.view(),
                down: down.view(),
                input_norm: input_norm.view(),
                post_norm: post_norm.view(),
            },
        )
        .map_err(|err| format!("layer {layer}: {err:?}"))?;
        let mut families = Map::new();
        for (name, charge) in census.charges() {
            by_family.entry(name).or_default().add(charge);
            let row = match name {
                "ov" => {
                    let mut row = charge_json(&charge);
                    row["groups"] = json!(n_kv);
                    row["per_group"] = json!(format!("GL({hd})"));
                    row["exact"] = json!(census.ov_exact());
                    row
                }
                "qk" => family_json(&census.qk, &charge),
                "input_norm" => family_json(&census.input_norm, &charge),
                "post_norm" => family_json(&census.post_norm, &charge),
                _ => family_json(&census.swiglu, &charge),
            };
            families.insert(name.into(), row);
        }
        let layer_tally = census.charge();
        if layer_tally.parameters != layer_parameters {
            return Err(format!(
                "layer {layer}: families cover {} coordinates, the checkpoint stores {layer_parameters}",
                layer_tally.parameters
            ));
        }
        total.add(layer_tally);
        let mut row = charge_json(&layer_tally);
        row["layer"] = json!(layer);
        row["families"] = Value::Object(families);
        println!(
            "layer {layer:2}: params {} orbit {} null {} real<= {} ({:.4}% convention) [{:.1}s]",
            layer_tally.parameters,
            layer_tally.orbit_resolved,
            layer_tally.null,
            layer_tally.real_coordinates_at_most(),
            100.0 * (layer_tally.orbit_resolved + layer_tally.null) as f64 / layer_tally.parameters as f64,
            started.elapsed().as_secs_f64()
        );
        layer_rows.push(row);
    }

    // Residual stream, restricted by the tie.
    let final_norm = vector(&export.join("final_norm.npy"))?;
    let embed_rows = matrix(&export.join("embed_rows.npy"))?;
    let tied = tied_residual_census(final_norm.view(), embed_rows.view(), vocab).map_err(|err| format!("residual: {err:?}"))?;
    let residual = tied.charge;
    let (embed_rank, untied_dimension, tied_dimension) = (tied.embedding_rank, tied.untied.dimension(), tied.tied_dimension);
    let largest_group = tied.equal_gain_groups.first().copied().unwrap_or(0);
    total.add(residual);
    by_family.entry("residual").or_default().add(residual);
    if total.parameters != checkpoint_parameters {
        return Err(format!("census covers {} coordinates, the checkpoint stores {checkpoint_parameters}", total.parameters));
    }
    let mut residual_row = charge_json(&residual);
    residual_row["stream_group_untied"] = json!(format!("{:?}", tied.untied));
    residual_row["stream_orbit_untied"] = json!(untied_dimension);
    residual_row["final_gain_distinct_values"] = json!(tied.equal_gain_groups.len());
    residual_row["final_gain_largest_equal_group"] = json!(largest_group);
    residual_row["embedding_rank_resolved_on_rows"] = json!([embed_rank, embed_rows.len_of(Axis(0))]);
    residual_row["final_norm_gain_gauge"] = json!("none: tied embedding (its coordinate scales would rescale the embedding writes)");

    let families_total: Map<String, Value> = by_family.iter().map(|(name, tally)| (name.to_string(), charge_json(tally))).collect();
    let mut receipt = Map::new();
    receipt.insert("schema".into(), json!("mpd_gauge_census_2951/v1"));
    receipt.insert("model".into(), manifest["model"].clone());
    receipt.insert("snapshot".into(), manifest["snapshot"].clone());
    receipt.insert(
        "config".into(),
        json!({"d": d, "layers": layers, "heads": n_heads, "kv_heads": n_kv, "head_dim": hd, "intermediate": manifest["intermediate_size"], "vocab": vocab, "rope_theta": theta, "rms_norm_eps": eps, "tied": true}),
    );
    receipt.insert(
        "evidence".into(),
        json!("orbit_resolved: ranks above the SVD backward-error band (Weyl lower bound) and exact bit tests; null: exact zero tests on the stored bits; real_coordinates_at_most = parameters - orbit_resolved - null is an upper bound on the coordinates a code must carry, never a claim that no further invariance exists. Discrete gauges (unit permutations, sign flips, quarter turns) are reported but carry no dimension."),
    );
    receipt.insert(
        "additivity".into(),
        json!("each family moves a tensor no other family moves (norm gains, q_norm/k_norm gains, embedding) or acts on disjoint coordinates (value rows / output columns, up rows / down columns, coincident query/key plane rows), so orbit dimensions add"),
    );
    receipt.insert("total".into(), charge_json(&total));
    receipt.insert("by_family".into(), Value::Object(families_total));
    receipt.insert("residual".into(), residual_row);
    receipt.insert("layers".into(), Value::Array(layer_rows));

    let body: Vec<String> = receipt
        .iter()
        .map(|(key, value)| format!("{}: {}", json!(key), serde_json::to_string(value).expect("json")))
        .collect();
    std::fs::write(&out, format!("{{\n{}\n}}\n", body.join(",\n"))).map_err(|err| format!("write {}: {err}", out.display()))?;
    println!(
        "TOTAL params {} orbit {} (at most {}) null {} real<= {} convention {:.4}%",
        total.parameters,
        total.orbit_resolved,
        total.orbit_at_most,
        total.null,
        total.real_coordinates_at_most(),
        100.0 * (total.orbit_resolved + total.null) as f64 / total.parameters as f64
    );
    for (name, tally) in &by_family {
        println!("  {name:10} orbit {:>9} null {:>7} of {:>10}", tally.orbit_resolved, tally.null, tally.parameters);
    }
    println!(
        "residual: untied O({d}) would be {untied_dimension}; tie leaves {tied_dimension} ({} distinct final gains, largest group {largest_group}); embed rank {embed_rank}",
        tied.equal_gain_groups.len()
    );
    println!("wrote {}", out.display());
    Ok(())
}
