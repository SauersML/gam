//! Implementation-gauge census of a Qwen3 checkpoint (#2951).
//!
//! Runs the `parameter_decomposition::gauge` detectors on every layer of the model
//! exported by `bench/mpd_gauge_census_2951.py` and reports, per layer and in total,
//! the parameter count, the certified orbit dimension (resolved / at most), the exact
//! null coordinates, and `real_coordinates_at_most`: the parameters minus the resolved
//! orbit dimension and the null coordinates. The difference is the part of the
//! checkpoint that is pure implementation convention.
//!
//! Families, per layer:
//! - `ov`: one `LinearPassthrough` per key/value group, reading the group's value head
//!   and writing through the output columns of every query head sharing it (`GL(hd)`).
//! - `qk`: `QueryKeyGauge` on the native attention with Qwen3's `q_norm`/`k_norm`
//!   (the normed rotary family: plane scales and coincident-plane rotations).
//! - `swiglu`: `SwigluUnits` (one nonzero up/down scale per unit).
//! - `input_norm`, `post_norm`: `NormGain` with their linear reads.
//!
//! Globally: the residual-stream basis. `ResidualStreamGauge` gives `O(d)` for RMSNorm
//! reads. A tied embedding/unembedding restricts it: the embedding write forces
//! `E ↦ E Qᵀ` and the final-norm read `E diag(w) ↦ E diag(w) Qᵀ`, so `Q diag(w) Qᵀ`
//! must stay diagonal. The identity component is the block-orthogonal group over the
//! groups of bitwise-equal final gains, of dimension `Σ m_v (m_v − 1)/2`, and its orbit
//! has that dimension when `E` has full column rank (resolved on a row subset, which
//! bounds the rank from below). The final norm's own coordinate scales are not a gauge
//! under the tie: they would rescale the embedding's writes.
//!
//! Additivity: every family moves a tensor no other family moves (each norm gain, the
//! `q_norm`/`k_norm` gains, the embedding), or acts on coordinates disjoint from the
//! other families' (value rows and output columns, up rows and down columns, the rows
//! of coincident query/key planes). So the joint orbit's tangent is the direct sum of
//! the families' tangents and the resolved orbit dimensions add.
//!
//! ```text
//! uv run --no-project --with numpy python bench/mpd_gauge_census_2951.py --out /tmp/census
//! cargo run --release -p gam-sae --example mpd_gauge_census_2951 -- \
//!     --export /tmp/census --out experiments/issue-2951/receipts/gauge_census_Qwen3-0.6B-Base.json
//! ```

use gam_linalg::faer_ndarray::FaerSvd;
use gam_linalg::roundoff::factor_singular_band;
use gam_sae::parameter_decomposition::attention::{
    AffineProjection, AttentionGeometry, NativeAttention, RotaryEmbedding, RotaryPairing,
};
use gam_sae::parameter_decomposition::operators::DeclaredGauge;
use gam_sae::parameter_decomposition::gauge::{
    GaugeFamily, LinearPassthrough, NormGain, QueryKeyGauge, ResidualRead, ResidualStreamGauge, SwigluUnits,
};
use memmap2::Mmap;
use ndarray::{Array1, Array2, Axis, s};
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

/// Running totals of one row of the census.
#[derive(Clone, Copy, Default)]
struct Tally {
    parameters: usize,
    orbit_resolved: usize,
    orbit_at_most: usize,
    null: usize,
}

impl Tally {
    fn add(&mut self, other: Tally) {
        self.parameters += other.parameters;
        self.orbit_resolved += other.orbit_resolved;
        self.orbit_at_most += other.orbit_at_most;
        self.null += other.null;
    }

    fn real_at_most(&self) -> usize {
        self.parameters - self.orbit_resolved - self.null
    }

    fn json(&self) -> Value {
        json!({
            "parameters": self.parameters,
            "orbit_resolved": self.orbit_resolved,
            "orbit_at_most": self.orbit_at_most,
            "null": self.null,
            "real_coordinates_at_most": self.real_at_most(),
            "convention_fraction": (self.orbit_resolved + self.null) as f64 / self.parameters as f64,
        })
    }
}

/// A family's orbit and null coordinates, charged against `parameters` stored
/// coordinates of this row (reads shared with another family are charged there).
fn charge(family: &GaugeFamily, parameters: usize) -> Tally {
    Tally {
        parameters,
        orbit_resolved: family.orbit_dimension.resolved,
        orbit_at_most: family.orbit_dimension.at_most,
        null: family.null_coordinates,
    }
}

fn family_json(family: &GaugeFamily, parameters: usize) -> Value {
    let mut row = charge(family, parameters).json();
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

/// Resolved rank above the SVD's backward-error band, as the gauge owner resolves it.
fn resolved_rank(matrix: &Array2<f64>) -> Result<usize, String> {
    let sigma = matrix.svd(false, false).map_err(|err| format!("svd: {err:?}"))?.1;
    let sigma_max = sigma.iter().fold(0.0_f64, |largest, &value| largest.max(value));
    let band = factor_singular_band(matrix.nrows(), matrix.ncols(), sigma_max);
    Ok(sigma.iter().filter(|&&value| value > band).count())
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
    let group = n_heads / n_kv;

    let mut total = Tally::default();
    let mut by_family: BTreeMap<&str, Tally> = BTreeMap::new();
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

        let mut families = Map::new();
        let mut layer_tally = Tally::default();
        let mut record = |name: &'static str, tally: Tally, row: Value, by_family: &mut BTreeMap<&str, Tally>| {
            layer_tally.add(tally);
            by_family.entry(name).or_default().add(tally);
            families.insert(name.into(), row);
        };

        // OV: one GL(hd) pass-through per key/value group.
        let mut ov = Tally::default();
        let mut ov_exact = true;
        for g in 0..n_kv {
            let read = v.slice(s![g * hd..(g + 1) * hd, ..]).to_owned();
            let mut write = Array2::<f64>::zeros((group * d, hd));
            for member in 0..group {
                let h = g * group + member;
                write
                    .slice_mut(s![member * d..(member + 1) * d, ..])
                    .assign(&o.slice(s![.., h * hd..(h + 1) * hd]));
            }
            let family = LinearPassthrough::new(read, write)
                .and_then(|block| block.family())
                .map_err(|err| format!("layer {layer} ov group {g}: {err:?}"))?;
            ov_exact &= family.orbit_dimension.is_exact();
            ov.add(charge(&family, family.parameter_coordinates));
        }
        let mut ov_row = ov.json();
        ov_row["groups"] = json!(n_kv);
        ov_row["per_group"] = json!(format!("GL({hd})"));
        ov_row["exact"] = json!(ov_exact);
        record("ov", ov, ov_row, &mut by_family);

        // QK behind q_norm / k_norm.
        let zeros = |n: usize| Array1::<f64>::zeros(n);
        let native = NativeAttention::new(
            geometry,
            rotary.clone(),
            1.0 / (hd as f64).sqrt(),
            AffineProjection { weight: q.clone(), bias: zeros(n_heads * hd) },
            AffineProjection { weight: k.clone(), bias: zeros(n_kv * hd) },
            AffineProjection { weight: v.clone(), bias: zeros(n_kv * hd) },
            AffineProjection { weight: o, bias: zeros(d) },
        )
        .and_then(|native| native.with_query_key_norm(eps, q_norm, k_norm))
        .map_err(|err| format!("layer {layer} attention: {err:?}"))?;
        let qk_family = QueryKeyGauge::new(&native)
            .and_then(|gauge| gauge.family())
            .map_err(|err| format!("layer {layer} qk: {err:?}"))?;
        let qk_parameters = q.len() + k.len() + 2 * hd;
        record("qk", charge(&qk_family, qk_parameters), family_json(&qk_family, qk_parameters), &mut by_family);
        drop(native);

        // Norm gains; their reads are charged to qk / ov / swiglu.
        let input_family = NormGain::new(input_norm, None, vec![q, k, v])
            .map(|norm| norm.family())
            .map_err(|err| format!("layer {layer} input norm: {err:?}"))?;
        record("input_norm", charge(&input_family, d), family_json(&input_family, d), &mut by_family);
        let post_family = NormGain::new(post_norm, None, vec![gate.clone(), up.clone()])
            .map(|norm| norm.family())
            .map_err(|err| format!("layer {layer} post norm: {err:?}"))?;
        record("post_norm", charge(&post_family, d), family_json(&post_family, d), &mut by_family);

        let swiglu_family = SwigluUnits::new(gate, up, down)
            .map(|units| units.family())
            .map_err(|err| format!("layer {layer} swiglu: {err:?}"))?;
        record(
            "swiglu",
            charge(&swiglu_family, swiglu_family.parameter_coordinates),
            family_json(&swiglu_family, swiglu_family.parameter_coordinates),
            &mut by_family,
        );

        if layer_tally.parameters != layer_parameters {
            return Err(format!(
                "layer {layer}: families cover {} coordinates, the checkpoint stores {layer_parameters}",
                layer_tally.parameters
            ));
        }
        total.add(layer_tally);
        let mut row = layer_tally.json();
        row["layer"] = json!(layer);
        row["families"] = Value::Object(families);
        println!(
            "layer {layer:2}: params {} orbit {} null {} real<= {} ({:.4}% convention) [{:.1}s]",
            layer_tally.parameters,
            layer_tally.orbit_resolved,
            layer_tally.null,
            layer_tally.real_at_most(),
            100.0 * (layer_tally.orbit_resolved + layer_tally.null) as f64 / layer_tally.parameters as f64,
            started.elapsed().as_secs_f64()
        );
        layer_rows.push(row);
    }

    // Residual stream, restricted by the tie.
    let final_norm = vector(&export.join("final_norm.npy"))?;
    let embed_rows = matrix(&export.join("embed_rows.npy"))?;
    let embed_rank = resolved_rank(&embed_rows)?;
    let stream = ResidualStreamGauge::new(d, &[ResidualRead::RmsNorm]).map_err(|err| format!("stream: {err:?}"))?;
    let untied_dimension = stream.continuous().dimension();
    let mut equal_gains: BTreeMap<u64, usize> = BTreeMap::new();
    for &gain in final_norm.iter() {
        *equal_gains.entry(gain.to_bits()).or_default() += 1;
    }
    let tied_dimension: usize = equal_gains.values().map(|&m| m * (m - 1) / 2).sum();
    let largest_group = equal_gains.values().copied().max().unwrap_or(0);
    let embed_parameters = vocab * d;
    let residual = Tally {
        parameters: embed_parameters + d,
        orbit_resolved: if embed_rank == d { tied_dimension } else { 0 },
        orbit_at_most: tied_dimension,
        null: 0,
    };
    total.add(residual);
    by_family.entry("residual").or_default().add(residual);
    if total.parameters != checkpoint_parameters {
        return Err(format!("census covers {} coordinates, the checkpoint stores {checkpoint_parameters}", total.parameters));
    }
    let mut residual_row = residual.json();
    residual_row["stream_group_untied"] = json!(format!("{:?}", stream.continuous()));
    residual_row["stream_orbit_untied"] = json!(untied_dimension);
    residual_row["final_gain_distinct_values"] = json!(equal_gains.len());
    residual_row["final_gain_largest_equal_group"] = json!(largest_group);
    residual_row["embedding_rank_resolved_on_rows"] = json!([embed_rank, embed_rows.len_of(Axis(0))]);
    residual_row["final_norm_gain_gauge"] = json!("none: tied embedding (its coordinate scales would rescale the embedding writes)");

    let families_total: Map<String, Value> = by_family.iter().map(|(name, tally)| (name.to_string(), tally.json())).collect();
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
    receipt.insert("total".into(), total.json());
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
        total.real_at_most(),
        100.0 * (total.orbit_resolved + total.null) as f64 / total.parameters as f64
    );
    for (name, tally) in &by_family {
        println!("  {name:10} orbit {:>9} null {:>7} of {:>10}", tally.orbit_resolved, tally.null, tally.parameters);
    }
    println!(
        "residual: untied O({d}) would be {untied_dimension}; tie leaves {tied_dimension} ({} distinct final gains, largest group {largest_group}); embed rank {embed_rank}",
        equal_gains.len()
    );
    println!("wrote {}", out.display());
    Ok(())
}
