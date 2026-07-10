//! Real Qwen3-8B weight-frame catalog against L18 residual activations.
//!
//! Usage:
//!
//! ```text
//! cargo run -p gam-sae --example weight_frames_qwen -- \
//!   <model-dir> <resid_L18.npy> <output-results.md> \
//!   [--layer N] [--sample-rows N] [--frame-rank N]
//! ```

use faer::Side;
use gam_linalg::faer_ndarray::{FaerEigh, FaerSvd, fast_ab, fast_abt, fast_atb};
use gam_sae::frames::GrassmannFrame;
use gam_sae::manifold::{
    ChartOccupancyStatus, InFrameCurvedConfig, WeightFrameCatalog, WeightFrameCatalogConfig,
    WeightFrameCatalogEntry, WeightFrameMatrix, WeightFrameOccupancy, WeightFrameSource,
    fit_inframe_curved_weight_frame_catalog, frame_catalog_from_weight_matrices,
};
use ndarray::{Array1, Array2, ArrayView2, s};
use serde_json::Value;
use std::collections::BTreeMap;
use std::fs::{self, File};
use std::io::{Read, Seek, SeekFrom, Write};
use std::path::{Path, PathBuf};
use std::process::ExitCode;
use std::time::Instant;

#[derive(Clone, Debug)]
struct Args {
    model_dir: PathBuf,
    residual_path: PathBuf,
    output_path: PathBuf,
    layer: usize,
    sample_rows: usize,
    frame_rank: usize,
    min_rows: usize,
    basis_size: usize,
    occupancy_null_multiple: f64,
}

#[derive(Clone, Debug)]
struct QwenConfig {
    hidden_size: usize,
    head_dim: usize,
    num_attention_heads: usize,
    num_key_value_heads: usize,
}

#[derive(Clone, Debug)]
struct TensorMeta {
    dtype: String,
    shape: Vec<usize>,
    start: usize,
    end: usize,
}

#[derive(Clone, Debug)]
struct FrameOccupancySummary {
    frame_index: usize,
    source: WeightFrameSource,
    rows: usize,
    row_fraction: f64,
    projected_energy_fraction: f64,
    mean_best_row_capture: f64,
}

#[derive(Clone, Debug)]
struct VarianceReport {
    raw_sink_ev: f64,
    post_peel_total_energy: f64,
    weight_union_rank: usize,
    weight_frame_ev: f64,
    data_svd_same_rank_ev: f64,
}

fn main() -> ExitCode {
    match parse_args().and_then(run) {
        Ok(()) => ExitCode::SUCCESS,
        Err(err) => {
            println!("[weight_frames_qwen] error: {err}");
            ExitCode::FAILURE
        }
    }
}

fn parse_args() -> Result<Args, String> {
    let raw: Vec<String> = std::env::args().collect();
    if raw.len() < 4 {
        return Err(
            "usage: weight_frames_qwen <model-dir> <resid_L18.npy> <output-results.md> [--layer N] [--sample-rows N] [--frame-rank N]".to_string(),
        );
    }
    let mut args = Args {
        model_dir: PathBuf::from(&raw[1]),
        residual_path: PathBuf::from(&raw[2]),
        output_path: PathBuf::from(&raw[3]),
        layer: 18,
        sample_rows: 4096,
        frame_rank: 16,
        min_rows: 32,
        basis_size: 5,
        occupancy_null_multiple: 4.0,
    };
    let mut i = 4usize;
    while i < raw.len() {
        let key = raw[i].as_str();
        let value = raw
            .get(i + 1)
            .ok_or_else(|| format!("missing value for {key}"))?;
        match key {
            "--layer" => args.layer = parse_positive_usize(value, key)?,
            "--sample-rows" => args.sample_rows = parse_positive_usize(value, key)?,
            "--frame-rank" => args.frame_rank = parse_positive_usize(value, key)?,
            "--min-rows" => args.min_rows = parse_positive_usize(value, key)?,
            "--basis-size" => args.basis_size = parse_positive_usize(value, key)?,
            "--occupancy-null-multiple" => {
                args.occupancy_null_multiple = value
                    .parse::<f64>()
                    .map_err(|err| format!("{key}: expected finite f64: {err}"))?;
                if !(args.occupancy_null_multiple.is_finite() && args.occupancy_null_multiple > 0.0)
                {
                    return Err(format!(
                        "{key}: expected positive finite multiplier, got {}",
                        args.occupancy_null_multiple
                    ));
                }
            }
            other => return Err(format!("unknown argument {other}")),
        }
        i += 2;
    }
    Ok(args)
}

fn parse_positive_usize(value: &str, key: &str) -> Result<usize, String> {
    match value.parse::<usize>() {
        Ok(v) if v > 0 => Ok(v),
        Ok(v) => Err(format!("{key}: expected positive usize, got {v}")),
        Err(err) => Err(format!("{key}: expected usize: {err}")),
    }
}

fn run(args: Args) -> Result<(), String> {
    let started = Instant::now();
    let qwen = read_qwen_config(&args.model_dir.join("config.json"))?;
    validate_args(&args, &qwen)?;
    println!(
        "loading Qwen layer {} tensors from {}",
        args.layer,
        args.model_dir.display()
    );
    let catalog = build_layer_weight_catalog(&args.model_dir, args.layer, &qwen, args.frame_rank)?;
    println!(
        "built catalog with {} frames at output dim {}",
        catalog.entries().len(),
        catalog.output_dim()
    );
    println!(
        "loading activation sample {} rows from {}",
        args.sample_rows,
        args.residual_path.display()
    );
    let (n_full, p_raw, mut activations) =
        read_npy_subsample_f64(&args.residual_path, args.sample_rows)?;
    if p_raw != qwen.hidden_size {
        return Err(format!(
            "activation dim {p_raw} does not match model hidden size {}",
            qwen.hidden_size
        ));
    }
    center_columns(&mut activations);
    let (raw_eigenvalues, raw_eigenvectors, raw_total) = covariance_eigh(activations.view())?;
    let raw_sink_ev = if raw_total > 0.0 {
        raw_eigenvalues[0] / raw_total
    } else {
        0.0
    };
    peel_component(
        &mut activations,
        raw_eigenvectors.column(0).to_owned().view(),
    );
    let post_total = matrix_energy(activations.view());
    let union_frame = orthonormal_union_frame(&catalog)?;
    let weight_frame_ev = projected_ev(activations.view(), union_frame.view(), post_total)?;
    let same_rank = union_frame
        .ncols()
        .min(raw_eigenvalues.len().saturating_sub(1));
    let data_svd_same_rank_ev = if post_total > 0.0 {
        raw_eigenvalues
            .iter()
            .skip(1)
            .take(same_rank)
            .copied()
            .sum::<f64>()
            / post_total
    } else {
        0.0
    };
    let variance = VarianceReport {
        raw_sink_ev,
        post_peel_total_energy: post_total,
        weight_union_rank: union_frame.ncols(),
        weight_frame_ev,
        data_svd_same_rank_ev,
    };
    let (occupancies, occupancy_summary) =
        assign_weight_frame_occupancy(activations.view(), &catalog, &args, post_total)?;
    let config = InFrameCurvedConfig {
        frame_rank_min: 1,
        frame_rank_max: args.frame_rank,
        min_rows: args.min_rows,
        ..Default::default()
    };
    let result = fit_inframe_curved_weight_frame_catalog(
        activations.view(),
        &catalog,
        &occupancies,
        n_full,
        &config,
    )?;
    let markdown = render_markdown(
        &args,
        &qwen,
        n_full,
        activations.nrows(),
        &variance,
        &occupancy_summary,
        &result,
        started.elapsed().as_secs_f64(),
    );
    if let Some(parent) = args.output_path.parent() {
        fs::create_dir_all(parent)
            .map_err(|err| format!("create output dir {}: {err}", parent.display()))?;
    }
    let mut file = File::create(&args.output_path)
        .map_err(|err| format!("create {}: {err}", args.output_path.display()))?;
    file.write_all(markdown.as_bytes())
        .map_err(|err| format!("write {}: {err}", args.output_path.display()))?;
    println!("wrote {}", args.output_path.display());
    Ok(())
}

fn validate_args(args: &Args, qwen: &QwenConfig) -> Result<(), String> {
    if qwen.hidden_size == 0
        || qwen.head_dim == 0
        || qwen.num_attention_heads == 0
        || qwen.num_key_value_heads == 0
    {
        return Err("Qwen config dimensions must be positive".to_string());
    }
    if qwen.num_attention_heads % qwen.num_key_value_heads != 0 {
        return Err(format!(
            "num_attention_heads {} must be divisible by num_key_value_heads {}",
            qwen.num_attention_heads, qwen.num_key_value_heads
        ));
    }
    if qwen.num_attention_heads * qwen.head_dim != qwen.hidden_size {
        return Err(format!(
            "heads * head_dim = {} but hidden_size = {}",
            qwen.num_attention_heads * qwen.head_dim,
            qwen.hidden_size
        ));
    }
    if args.frame_rank > qwen.head_dim {
        return Err(format!(
            "frame rank {} cannot exceed head_dim {} for per-head OV frames",
            args.frame_rank, qwen.head_dim
        ));
    }
    Ok(())
}

fn read_qwen_config(path: &Path) -> Result<QwenConfig, String> {
    let text = fs::read_to_string(path).map_err(|err| format!("read {}: {err}", path.display()))?;
    let value: Value =
        serde_json::from_str(&text).map_err(|err| format!("parse {}: {err}", path.display()))?;
    Ok(QwenConfig {
        hidden_size: json_usize(&value, "hidden_size")?,
        head_dim: json_usize(&value, "head_dim")?,
        num_attention_heads: json_usize(&value, "num_attention_heads")?,
        num_key_value_heads: json_usize(&value, "num_key_value_heads")?,
    })
}

fn json_usize(value: &Value, key: &str) -> Result<usize, String> {
    value
        .get(key)
        .and_then(Value::as_u64)
        .map(|v| v as usize)
        .ok_or_else(|| format!("config missing integer {key}"))
}

fn build_layer_weight_catalog(
    model_dir: &Path,
    layer: usize,
    qwen: &QwenConfig,
    frame_rank: usize,
) -> Result<WeightFrameCatalog, String> {
    let o_name = format!("model.layers.{layer}.self_attn.o_proj.weight");
    let v_name = format!("model.layers.{layer}.self_attn.v_proj.weight");
    let down_name = format!("model.layers.{layer}.mlp.down_proj.weight");
    let o_proj = read_model_tensor_2d(model_dir, &o_name)?;
    let v_proj = read_model_tensor_2d(model_dir, &v_name)?;
    if o_proj.dim() != (qwen.hidden_size, qwen.hidden_size) {
        return Err(format!(
            "{o_name} shape {:?} != ({}, {})",
            o_proj.dim(),
            qwen.hidden_size,
            qwen.hidden_size
        ));
    }
    let expected_v_rows = qwen.num_key_value_heads * qwen.head_dim;
    if v_proj.dim() != (expected_v_rows, qwen.hidden_size) {
        return Err(format!(
            "{v_name} shape {:?} != ({expected_v_rows}, {})",
            v_proj.dim(),
            qwen.hidden_size
        ));
    }
    let mut components = Vec::with_capacity(qwen.num_attention_heads);
    let heads_per_kv = qwen.num_attention_heads / qwen.num_key_value_heads;
    for head in 0..qwen.num_attention_heads {
        let o_start = head * qwen.head_dim;
        let kv_head = head / heads_per_kv;
        let v_start = kv_head * qwen.head_dim;
        let o_head = o_proj
            .slice(s![.., o_start..o_start + qwen.head_dim])
            .to_owned();
        let v_head = v_proj
            .slice(s![v_start..v_start + qwen.head_dim, ..])
            .to_owned();
        let ov_factor = ov_left_svd_factor(o_head.view(), v_head.view())?;
        components.push(WeightFrameMatrix {
            source: WeightFrameSource::AttentionHeadOv { layer, head },
            matrix: ov_factor,
        });
    }
    let mut entries = frame_catalog_from_weight_matrices(
        &components,
        &WeightFrameCatalogConfig {
            frame_rank_min: frame_rank,
            frame_rank_max: frame_rank,
            ..Default::default()
        },
    )?
    .entries()
    .to_vec();
    let down_proj = read_model_tensor_2d(model_dir, &down_name)?;
    if down_proj.nrows() != qwen.hidden_size {
        return Err(format!(
            "{down_name} rows {} != hidden_size {}",
            down_proj.nrows(),
            qwen.hidden_size
        ));
    }
    let mlp_entry = left_image_entry_from_wide_matrix(
        WeightFrameSource::MlpDownProjection { layer },
        down_proj.view(),
        frame_rank,
    )?;
    entries.push(mlp_entry);
    WeightFrameCatalog::new(entries)
}

fn ov_left_svd_factor(
    w_o_head: ArrayView2<'_, f64>,
    w_v_head: ArrayView2<'_, f64>,
) -> Result<Array2<f64>, String> {
    if w_o_head.ncols() != w_v_head.nrows() {
        return Err(format!(
            "OV factor shape mismatch: W_O head cols {} != W_V head rows {}",
            w_o_head.ncols(),
            w_v_head.nrows()
        ));
    }
    let vv_t = fast_abt(&w_v_head.to_owned(), &w_v_head.to_owned());
    let (evals, evecs) = vv_t
        .eigh(Side::Lower)
        .map_err(|err| format!("OV W_V W_V^T eigh failed: {err}"))?;
    let d = evals.len();
    let mut scaled = Array2::<f64>::zeros((d, d));
    for col in 0..d {
        let scale = evals[col].max(0.0).sqrt();
        for row in 0..d {
            scaled[[row, col]] = evecs[[row, col]] * scale;
        }
    }
    Ok(fast_ab(&w_o_head.to_owned(), &scaled))
}

fn left_image_entry_from_wide_matrix(
    source: WeightFrameSource,
    matrix: ArrayView2<'_, f64>,
    frame_rank: usize,
) -> Result<WeightFrameCatalogEntry, String> {
    let covariance = fast_abt(&matrix.to_owned(), &matrix.to_owned());
    let (evals, evecs) = covariance
        .eigh(Side::Lower)
        .map_err(|err| format!("left-image covariance eigh failed: {err}"))?;
    let mut order: Vec<usize> = (0..evals.len()).collect();
    order.sort_by(|&a, &b| evals[b].total_cmp(&evals[a]));
    let rank = frame_rank.min(order.len());
    let mut frame = Array2::<f64>::zeros((matrix.nrows(), rank));
    let mut gauge = Array1::<f64>::zeros(rank);
    let mut singular_values = Array1::<f64>::zeros(evals.len());
    for (out_idx, &eig_idx) in order.iter().enumerate() {
        let sv = evals[eig_idx].max(0.0).sqrt();
        singular_values[out_idx] = sv;
        if out_idx < rank {
            gauge[out_idx] = sv;
            for row in 0..matrix.nrows() {
                frame[[row, out_idx]] = evecs[[row, eig_idx]];
            }
        }
    }
    Ok(WeightFrameCatalogEntry {
        source,
        frame: GrassmannFrame::from_orthonormal(frame, gauge)?,
        singular_values,
        matrix_rows: matrix.nrows(),
        matrix_cols: matrix.ncols(),
    })
}

fn read_model_tensor_2d(model_dir: &Path, tensor_name: &str) -> Result<Array2<f64>, String> {
    let index_path = model_dir.join("model.safetensors.index.json");
    let index_text = fs::read_to_string(&index_path)
        .map_err(|err| format!("read {}: {err}", index_path.display()))?;
    let index_value: Value = serde_json::from_str(&index_text)
        .map_err(|err| format!("parse {}: {err}", index_path.display()))?;
    let file_name = index_value
        .get("weight_map")
        .and_then(|v| v.get(tensor_name))
        .and_then(Value::as_str)
        .ok_or_else(|| format!("{tensor_name} not found in {}", index_path.display()))?;
    read_safetensor_2d(&model_dir.join(file_name), tensor_name)
}

fn read_safetensor_2d(path: &Path, tensor_name: &str) -> Result<Array2<f64>, String> {
    let mut file = File::open(path).map_err(|err| format!("open {}: {err}", path.display()))?;
    let mut raw_len = [0u8; 8];
    file.read_exact(&mut raw_len)
        .map_err(|err| format!("read safetensors header len {}: {err}", path.display()))?;
    let header_len = u64::from_le_bytes(raw_len) as usize;
    let mut header_bytes = vec![0u8; header_len];
    file.read_exact(&mut header_bytes)
        .map_err(|err| format!("read safetensors header {}: {err}", path.display()))?;
    let header_text = std::str::from_utf8(&header_bytes)
        .map_err(|err| format!("{} header is not utf8: {err}", path.display()))?;
    let header: Value = serde_json::from_str(header_text)
        .map_err(|err| format!("parse safetensors header {}: {err}", path.display()))?;
    let meta = parse_tensor_meta(&header, tensor_name)?;
    if meta.shape.len() != 2 {
        return Err(format!(
            "{tensor_name}: expected 2-D tensor, got {:?}",
            meta.shape
        ));
    }
    let rows = meta.shape[0];
    let cols = meta.shape[1];
    let elem = dtype_size(&meta.dtype)?;
    let expected = rows
        .checked_mul(cols)
        .and_then(|v| v.checked_mul(elem))
        .ok_or_else(|| format!("{tensor_name}: byte length overflow"))?;
    if meta.end.saturating_sub(meta.start) != expected {
        return Err(format!(
            "{tensor_name}: byte span {} does not match expected {expected}",
            meta.end.saturating_sub(meta.start)
        ));
    }
    let data_start = 8u64 + header_len as u64 + meta.start as u64;
    file.seek(SeekFrom::Start(data_start))
        .map_err(|err| format!("seek {tensor_name} in {}: {err}", path.display()))?;
    let mut bytes = vec![0u8; expected];
    file.read_exact(&mut bytes)
        .map_err(|err| format!("read {tensor_name} in {}: {err}", path.display()))?;
    let mut out = Array2::<f64>::zeros((rows, cols));
    match meta.dtype.as_str() {
        "BF16" => {
            for i in 0..rows {
                for j in 0..cols {
                    let b = (i * cols + j) * 2;
                    let raw = u16::from_le_bytes([bytes[b], bytes[b + 1]]);
                    out[[i, j]] = f32::from_bits((raw as u32) << 16) as f64;
                }
            }
        }
        "F16" => {
            for i in 0..rows {
                for j in 0..cols {
                    let b = (i * cols + j) * 2;
                    out[[i, j]] = f16_to_f32(u16::from_le_bytes([bytes[b], bytes[b + 1]])) as f64;
                }
            }
        }
        "F32" => {
            for i in 0..rows {
                for j in 0..cols {
                    let b = (i * cols + j) * 4;
                    out[[i, j]] =
                        f32::from_le_bytes([bytes[b], bytes[b + 1], bytes[b + 2], bytes[b + 3]])
                            as f64;
                }
            }
        }
        other => return Err(format!("{tensor_name}: unsupported dtype {other}")),
    }
    Ok(out)
}

fn parse_tensor_meta(header: &Value, tensor_name: &str) -> Result<TensorMeta, String> {
    let tensor = header
        .get(tensor_name)
        .ok_or_else(|| format!("{tensor_name} not present in safetensors shard"))?;
    let dtype = tensor
        .get("dtype")
        .and_then(Value::as_str)
        .ok_or_else(|| format!("{tensor_name}: missing dtype"))?
        .to_string();
    let shape = tensor
        .get("shape")
        .and_then(Value::as_array)
        .ok_or_else(|| format!("{tensor_name}: missing shape"))?
        .iter()
        .map(|v| {
            v.as_u64()
                .map(|n| n as usize)
                .ok_or_else(|| format!("{tensor_name}: non-integer shape entry"))
        })
        .collect::<Result<Vec<_>, _>>()?;
    let offsets = tensor
        .get("data_offsets")
        .and_then(Value::as_array)
        .ok_or_else(|| format!("{tensor_name}: missing data_offsets"))?;
    if offsets.len() != 2 {
        return Err(format!("{tensor_name}: expected two data offsets"));
    }
    let start = offsets[0]
        .as_u64()
        .map(|v| v as usize)
        .ok_or_else(|| format!("{tensor_name}: invalid start offset"))?;
    let end = offsets[1]
        .as_u64()
        .map(|v| v as usize)
        .ok_or_else(|| format!("{tensor_name}: invalid end offset"))?;
    Ok(TensorMeta {
        dtype,
        shape,
        start,
        end,
    })
}

fn dtype_size(dtype: &str) -> Result<usize, String> {
    match dtype {
        "BF16" | "F16" => Ok(2),
        "F32" => Ok(4),
        other => Err(format!("unsupported safetensors dtype {other}")),
    }
}

fn center_columns(x: &mut Array2<f64>) {
    let n = x.nrows();
    let p = x.ncols();
    for col in 0..p {
        let mut mean = 0.0_f64;
        for row in 0..n {
            mean += x[[row, col]];
        }
        mean /= n.max(1) as f64;
        for row in 0..n {
            x[[row, col]] -= mean;
        }
    }
}

fn covariance_eigh(x: ArrayView2<'_, f64>) -> Result<(Array1<f64>, Array2<f64>, f64), String> {
    let covariance = fast_atb(&x.to_owned(), &x.to_owned());
    let (evals, evecs) = covariance
        .eigh(Side::Lower)
        .map_err(|err| format!("activation covariance eigh failed: {err}"))?;
    let mut order: Vec<usize> = (0..evals.len()).collect();
    order.sort_by(|&a, &b| evals[b].total_cmp(&evals[a]));
    let mut sorted_vals = Array1::<f64>::zeros(evals.len());
    let mut sorted_vecs = Array2::<f64>::zeros(evecs.dim());
    for (out_idx, &eig_idx) in order.iter().enumerate() {
        sorted_vals[out_idx] = evals[eig_idx].max(0.0);
        for row in 0..evecs.nrows() {
            sorted_vecs[[row, out_idx]] = evecs[[row, eig_idx]];
        }
    }
    let total = sorted_vals.iter().copied().sum::<f64>();
    Ok((sorted_vals, sorted_vecs, total))
}

fn peel_component(x: &mut Array2<f64>, component: ndarray::ArrayView1<'_, f64>) {
    for mut row in x.outer_iter_mut() {
        let mut score = 0.0_f64;
        for col in 0..row.len() {
            score += row[col] * component[col];
        }
        for col in 0..row.len() {
            row[col] -= score * component[col];
        }
    }
}

fn matrix_energy(x: ArrayView2<'_, f64>) -> f64 {
    x.iter().map(|v| v * v).sum::<f64>()
}

fn orthonormal_union_frame(catalog: &WeightFrameCatalog) -> Result<Array2<f64>, String> {
    let p = catalog.output_dim();
    let cols = catalog
        .entries()
        .iter()
        .map(|e| e.frame.rank())
        .sum::<usize>();
    let mut stacked = Array2::<f64>::zeros((p, cols));
    let mut offset = 0usize;
    for entry in catalog.entries() {
        let frame = entry.frame.frame();
        for col in 0..frame.ncols() {
            for row in 0..p {
                stacked[[row, offset + col]] = frame[[row, col]];
            }
        }
        offset += frame.ncols();
    }
    let (u_opt, sv, _vt) = stacked
        .svd(true, false)
        .map_err(|err| format!("weight union frame SVD failed: {err}"))?;
    let u = u_opt.ok_or_else(|| "weight union frame SVD returned no U".to_string())?;
    let max_sv = sv.iter().copied().fold(0.0_f64, f64::max);
    if !(max_sv > 0.0) {
        return Err("weight union frame has zero singular spectrum".to_string());
    }
    let rank = sv.iter().filter(|&&v| v > max_sv * 1.0e-10).count();
    let mut out = Array2::<f64>::zeros((p, rank));
    for col in 0..rank {
        for row in 0..p {
            out[[row, col]] = u[[row, col]];
        }
    }
    Ok(out)
}

fn projected_ev(
    x: ArrayView2<'_, f64>,
    frame: ArrayView2<'_, f64>,
    total_energy: f64,
) -> Result<f64, String> {
    if x.ncols() != frame.nrows() {
        return Err(format!(
            "projection shape mismatch: x cols {} != frame rows {}",
            x.ncols(),
            frame.nrows()
        ));
    }
    if !(total_energy.is_finite() && total_energy > 0.0) {
        return Ok(0.0);
    }
    let coords = fast_ab(&x.to_owned(), &frame.to_owned());
    Ok(matrix_energy(coords.view()) / total_energy)
}

fn assign_weight_frame_occupancy(
    x: ArrayView2<'_, f64>,
    catalog: &WeightFrameCatalog,
    args: &Args,
    total_energy: f64,
) -> Result<(Vec<WeightFrameOccupancy>, Vec<FrameOccupancySummary>), String> {
    let n = x.nrows();
    let frame_count = catalog.entries().len();
    let mut row_energy = vec![0.0_f64; n];
    for row in 0..n {
        for col in 0..x.ncols() {
            row_energy[row] += x[[row, col]] * x[[row, col]];
        }
    }
    let mut best_frame = vec![usize::MAX; n];
    let mut best_energy = vec![0.0_f64; n];
    let mut frame_energy = vec![0.0_f64; frame_count];
    for (frame_index, entry) in catalog.entries().iter().enumerate() {
        let coords = fast_ab(&x.to_owned(), &entry.frame.frame().to_owned());
        for row in 0..n {
            let mut energy = 0.0_f64;
            for col in 0..coords.ncols() {
                energy += coords[[row, col]] * coords[[row, col]];
            }
            frame_energy[frame_index] += energy;
            if energy > best_energy[row] {
                best_energy[row] = energy;
                best_frame[row] = frame_index;
            }
        }
    }
    let mut rows_by_frame: Vec<Vec<usize>> = (0..frame_count)
        .map(|index| Vec::with_capacity(index.min(1)))
        .collect();
    for row in 0..n {
        let frame_index = best_frame[row];
        if frame_index == usize::MAX || row_energy[row] <= 0.0 {
            continue;
        }
        let rank = catalog.entries()[frame_index].frame.rank();
        let null_capture = rank as f64 / x.ncols().max(1) as f64;
        let required = args.occupancy_null_multiple * null_capture;
        if best_energy[row] / row_energy[row] >= required {
            rows_by_frame[frame_index].push(row);
        }
    }
    let mut occupancies = Vec::with_capacity(frame_count);
    let mut summaries = Vec::with_capacity(frame_count);
    for (frame_index, rows) in rows_by_frame.into_iter().enumerate() {
        let entry = &catalog.entries()[frame_index];
        let mean_best = if rows.is_empty() {
            0.0
        } else {
            rows.iter()
                .map(|&row| best_energy[row] / row_energy[row].max(f64::MIN_POSITIVE))
                .sum::<f64>()
                / rows.len() as f64
        };
        summaries.push(FrameOccupancySummary {
            frame_index,
            source: entry.source.clone(),
            rows: rows.len(),
            row_fraction: rows.len() as f64 / n.max(1) as f64,
            projected_energy_fraction: if total_energy > 0.0 {
                frame_energy[frame_index] / total_energy
            } else {
                0.0
            },
            mean_best_row_capture: mean_best,
        });
        occupancies.push(WeightFrameOccupancy {
            frame_index,
            rows,
            basis_size: args.basis_size,
        });
    }
    Ok((occupancies, summaries))
}

fn render_markdown(
    args: &Args,
    qwen: &QwenConfig,
    n_full: usize,
    n_used: usize,
    variance: &VarianceReport,
    occupancy: &[FrameOccupancySummary],
    result: &gam_sae::manifold::InFrameCurvedResult,
    elapsed_seconds: f64,
) -> String {
    let mut lines = Vec::new();
    lines.push("# Qwen3-8B L18 Weight-Sourced Frames".to_string());
    lines.push(String::new());
    lines.push("## Setup".to_string());
    lines.push(format!(
        "- Layer: {}. Model hidden size: {}, heads: {}, KV heads: {}, head dim: {}.",
        args.layer,
        qwen.hidden_size,
        qwen.num_attention_heads,
        qwen.num_key_value_heads,
        qwen.head_dim
    ));
    lines.push(format!(
        "- Activation rows: {n_used} sampled from {n_full}; position/sink component peeled before all post-peel measurements."
    ));
    lines.push(format!(
        "- Catalog: {} frames, per-frame rank {}, occupancy gate = best-frame row capture >= {:.3}x rank/hidden null.",
        occupancy.len(),
        args.frame_rank,
        args.occupancy_null_multiple
    ));
    lines.push(format!("- Wall seconds on MSI: {elapsed_seconds:.3}."));
    lines.push(String::new());
    lines.push("## Variance Span".to_string());
    lines.push(format!(
        "- Raw top position/sink PC EV: {:.6}.",
        variance.raw_sink_ev
    ));
    lines.push(format!(
        "- Post-peel total centered energy: {:.6e}.",
        variance.post_peel_total_energy
    ));
    lines.push(format!(
        "- Weight-frame union rank: {}.",
        variance.weight_union_rank
    ));
    lines.push(format!(
        "- EV captured by weight-frame union: {:.6}.",
        variance.weight_frame_ev
    ));
    lines.push(format!(
        "- EV captured by data-SVD frame at same rank: {:.6}.",
        variance.data_svd_same_rank_ev
    ));
    lines.push(String::new());
    lines.push("## Mechanism Attribution".to_string());
    lines.push(
        "| rank | source | occupied rows | row fraction | projected EV | mean best-row capture | chart status | accepted | margin |"
            .to_string(),
    );
    lines.push("|---:|---|---:|---:|---:|---:|---|---:|---:|".to_string());
    let records_by_frame = records_by_frame_index(result);
    let mut order: Vec<usize> = (0..occupancy.len()).collect();
    order.sort_by(|&a, &b| occupancy[b].rows.cmp(&occupancy[a].rows));
    for (rank, &idx) in order.iter().take(12).enumerate() {
        let summary = &occupancy[idx];
        let record = records_by_frame.get(&summary.frame_index);
        let status = record
            .map(|r| format!("{:?}", r.occupancy_status))
            .unwrap_or_else(|| "Missing".to_string());
        let accepted = record.map(|r| r.evidence.selected_by_bic).unwrap_or(false);
        let margin = record.map(|r| r.evidence.margin).unwrap_or(0.0);
        lines.push(format!(
            "| {} | {} | {} | {:.6} | {:.6} | {:.6} | {} | {} | {:.6} |",
            rank + 1,
            source_label(&summary.source),
            summary.rows,
            summary.row_fraction,
            summary.projected_energy_fraction,
            summary.mean_best_row_capture,
            status,
            accepted,
            margin
        ));
    }
    lines.push(String::new());
    lines.push("## Chartable-Unoccupied Frames".to_string());
    let mut unoccupied = occupancy
        .iter()
        .filter(|summary| {
            records_by_frame
                .get(&summary.frame_index)
                .map(|record| record.occupancy_status == ChartOccupancyStatus::ChartableUnoccupied)
                .unwrap_or(false)
        })
        .collect::<Vec<_>>();
    unoccupied.sort_by(|a, b| a.frame_index.cmp(&b.frame_index));
    if unoccupied.is_empty() {
        lines.push("- None under the configured best-frame occupancy gate.".to_string());
    } else {
        for summary in unoccupied.iter().take(24) {
            lines.push(format!(
                "- {}: row fraction {:.6}, projected EV {:.6}.",
                source_label(&summary.source),
                summary.row_fraction,
                summary.projected_energy_fraction
            ));
        }
    }
    lines.push(String::new());
    lines.push("## Curved Fit Summary".to_string());
    lines.push(format!(
        "- Accepted regions: {} of {}.",
        result.selected_regions.len(),
        result.records.len()
    ));
    lines.push(format!(
        "- In-frame border coeffs: {}; dense border coeffs avoided for accepted charts: {}; border shrink: {:.3}x.",
        result.ledger.inframe_border_coeffs,
        result.ledger.dense_border_coeffs,
        result.ledger.border_shrink()
    ));
    lines.push(format!(
        "- In-frame covariance bytes: {}; dense covariance bytes avoided: {}; covariance shrink: {:.3}x.",
        result.ledger.inframe_cov_bytes,
        result.ledger.dense_cov_bytes,
        result.ledger.cov_shrink()
    ));
    lines.push(String::new());
    lines.join("\n")
}

fn records_by_frame_index(
    result: &gam_sae::manifold::InFrameCurvedResult,
) -> BTreeMap<usize, &gam_sae::manifold::RegionRecord> {
    let mut out = BTreeMap::new();
    for record in &result.records {
        if let Some(frame_index) = record.frame_catalog_index {
            let previous = out.insert(frame_index, record);
            if previous.is_some() {
                println!("duplicate frame record for catalog index {frame_index}");
            }
        }
    }
    out
}

fn source_label(source: &WeightFrameSource) -> String {
    match source {
        WeightFrameSource::AttentionHeadOv { layer, head } => format!("L{layer} head {head} OV"),
        WeightFrameSource::MlpDownProjection { layer } => format!("L{layer} MLP down"),
    }
}

fn parse_npy_header(
    file: &mut File,
    path: &Path,
) -> Result<(usize, usize, usize, bool, u64), String> {
    let mut magic = [0u8; 6];
    file.read_exact(&mut magic)
        .map_err(|err| format!("read npy magic {}: {err}", path.display()))?;
    if &magic != b"\x93NUMPY" {
        return Err(format!("{}: not a .npy file", path.display()));
    }
    let mut version = [0u8; 2];
    file.read_exact(&mut version)
        .map_err(|err| format!("read npy version {}: {err}", path.display()))?;
    let header_len = if version[0] == 1 {
        let mut raw = [0u8; 2];
        file.read_exact(&mut raw)
            .map_err(|err| format!("read v1 header len {}: {err}", path.display()))?;
        u16::from_le_bytes(raw) as usize
    } else if version[0] == 2 || version[0] == 3 {
        let mut raw = [0u8; 4];
        file.read_exact(&mut raw)
            .map_err(|err| format!("read v2 header len {}: {err}", path.display()))?;
        u32::from_le_bytes(raw) as usize
    } else {
        return Err(format!(
            "{}: unsupported npy version {}.{}",
            path.display(),
            version[0],
            version[1]
        ));
    };
    let mut header_bytes = vec![0u8; header_len];
    file.read_exact(&mut header_bytes)
        .map_err(|err| format!("read npy header {}: {err}", path.display()))?;
    let header = String::from_utf8(header_bytes)
        .map_err(|err| format!("{}: header utf8: {err}", path.display()))?;
    let is_f4 = header.contains("'<f4'") || header.contains("\"<f4\"");
    let is_f2 = header.contains("'<f2'") || header.contains("\"<f2\"");
    if !(is_f4 || is_f2) {
        return Err(format!(
            "{}: expected little-endian f32 or f16 npy, header={header:?}",
            path.display()
        ));
    }
    if header.contains("True") {
        return Err(format!(
            "{}: fortran_order=True is not supported",
            path.display()
        ));
    }
    let shape_start = header
        .find('(')
        .ok_or_else(|| format!("{}: missing shape", path.display()))?;
    let shape_end = header[shape_start..]
        .find(')')
        .map(|idx| idx + shape_start)
        .ok_or_else(|| format!("{}: unterminated shape", path.display()))?;
    let dims = header[shape_start + 1..shape_end]
        .split(',')
        .filter_map(|part| {
            let trimmed = part.trim();
            if trimmed.is_empty() {
                None
            } else {
                Some(trimmed.parse::<usize>())
            }
        })
        .collect::<Result<Vec<_>, _>>()
        .map_err(|err| format!("{}: parse npy shape: {err}", path.display()))?;
    if dims.len() != 2 {
        return Err(format!(
            "{}: expected 2-D npy, got {dims:?}",
            path.display()
        ));
    }
    let data_offset = file
        .stream_position()
        .map_err(|err| format!("{}: stream position: {err}", path.display()))?;
    let elem = if is_f4 { 4 } else { 2 };
    Ok((dims[0], dims[1], elem, is_f4, data_offset))
}

fn read_npy_subsample_f64(path: &Path, cap: usize) -> Result<(usize, usize, Array2<f64>), String> {
    let mut file = File::open(path).map_err(|err| format!("open {}: {err}", path.display()))?;
    let (n_full, p, elem, is_f4, data_offset) = parse_npy_header(&mut file, path)?;
    let take = cap.min(n_full);
    let row_bytes = p
        .checked_mul(elem)
        .ok_or_else(|| format!("{}: row byte size overflow", path.display()))?;
    let mut out = Array2::<f64>::zeros((take, p));
    let mut buf = vec![0u8; row_bytes];
    for i in 0..take {
        let src_row = i * n_full / take.max(1);
        let byte_offset = data_offset + (src_row * row_bytes) as u64;
        file.seek(SeekFrom::Start(byte_offset))
            .map_err(|err| format!("seek row {src_row} {}: {err}", path.display()))?;
        file.read_exact(&mut buf)
            .map_err(|err| format!("read row {src_row} {}: {err}", path.display()))?;
        if is_f4 {
            for c in 0..p {
                let b = c * 4;
                out[[i, c]] =
                    f32::from_le_bytes([buf[b], buf[b + 1], buf[b + 2], buf[b + 3]]) as f64;
            }
        } else {
            for c in 0..p {
                let b = c * 2;
                out[[i, c]] = f16_to_f32(u16::from_le_bytes([buf[b], buf[b + 1]])) as f64;
            }
        }
    }
    Ok((n_full, p, out))
}

fn f16_to_f32(h: u16) -> f32 {
    let sign = (h >> 15) & 0x1;
    let exp = (h >> 10) & 0x1f;
    let mant = h & 0x3ff;
    let bits = if exp == 0 {
        if mant == 0 {
            (sign as u32) << 31
        } else {
            let mut m = mant as u32;
            let mut e: i32 = -1;
            while (m & 0x400) == 0 {
                m <<= 1;
                e -= 1;
            }
            m &= 0x3ff;
            let exp32 = (127 - 15 + 1 + e) as u32;
            ((sign as u32) << 31) | (exp32 << 23) | (m << 13)
        }
    } else if exp == 0x1f {
        ((sign as u32) << 31) | (0xff << 23) | ((mant as u32) << 13)
    } else {
        let exp32 = (exp as i32 - 15 + 127) as u32;
        ((sign as u32) << 31) | (exp32 << 23) | ((mant as u32) << 13)
    };
    f32::from_bits(bits)
}
