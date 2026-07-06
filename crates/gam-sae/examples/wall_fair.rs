use gam_sae::sparse_dict::{
    BlockChartComposeConfig, BlockSparseConfig, compose_block_coordinate_charts,
    fit_block_sparse_dictionary,
};
use ndarray::Array2;
use serde_json::{Value, json};
use std::fs::{self, File};
use std::io::{Read, Seek, SeekFrom, Write};
use std::path::{Path, PathBuf};
use std::time::Instant;

#[derive(Clone, Debug)]
struct Args {
    data_root: PathBuf,
    output_dir: PathBuf,
    sample_rows: usize,
    min_stratum_rows: usize,
    flat_blocks: usize,
    curved_blocks: usize,
    block_size: usize,
    chart_basis: usize,
    block_topk: usize,
    max_epochs: usize,
    minibatch: usize,
    block_tile: usize,
    seed: u64,
}

#[derive(Clone, Debug)]
struct NpyHeader {
    rows: usize,
    cols: usize,
    data_offset: u64,
}

#[derive(Clone, Debug)]
struct Stratum {
    index: usize,
    exp_lo: usize,
    exp_hi: usize,
    rows: Vec<usize>,
    sum_energy: f64,
    sumsq_energy: f64,
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let args = parse_args()?;
    fs::create_dir_all(&args.output_dir)?;
    let started = Instant::now();
    let mut layers = Vec::new();
    for layer in [18usize, 30usize] {
        layers.push(run_layer(layer, &args)?);
    }
    let payload = json!({
        "experiment": "wall_fair",
        "engine": "gam-sae::sparse_dict::fit_block_sparse_dictionary + compose_block_coordinate_charts",
        "elapsed_seconds": started.elapsed().as_secs_f64(),
        "settings": {
            "sample_rows": args.sample_rows,
            "min_stratum_rows": args.min_stratum_rows,
            "flat_blocks": args.flat_blocks,
            "curved_blocks": args.curved_blocks,
            "block_size": args.block_size,
            "chart_basis": args.chart_basis,
            "block_topk": args.block_topk,
            "max_epochs": args.max_epochs,
            "minibatch": args.minibatch,
            "block_tile": args.block_tile,
            "seed": args.seed,
            "pair_screen": false
        },
        "layers": layers
    });
    let numbers_path = args.output_dir.join("numbers.json");
    let mut numbers = File::create(&numbers_path)?;
    numbers.write_all(serde_json::to_string_pretty(&payload)?.as_bytes())?;
    numbers.write_all(b"\n")?;
    let results_path = args.output_dir.join("results.md");
    let mut results = File::create(&results_path)?;
    results.write_all(render_markdown(&payload).as_bytes())?;
    println!("wrote {}", numbers_path.display());
    println!("wrote {}", results_path.display());
    Ok(())
}

fn parse_args() -> Result<Args, String> {
    let raw: Vec<String> = std::env::args().collect();
    if raw.len() < 3 {
        return Err(
            "usage: wall_fair <data-root> <output-dir> [--sample-rows N] [--flat-blocks G]"
                .to_string(),
        );
    }
    let mut args = Args {
        data_root: PathBuf::from(&raw[1]),
        output_dir: PathBuf::from(&raw[2]),
        sample_rows: 20_000,
        min_stratum_rows: 512,
        flat_blocks: 24,
        curved_blocks: 0,
        block_size: 4,
        chart_basis: 4,
        block_topk: 2,
        max_epochs: 8,
        minibatch: 512,
        block_tile: 512,
        seed: 1729,
    };
    let mut i = 3usize;
    while i < raw.len() {
        let key = raw[i].as_str();
        let value = raw
            .get(i + 1)
            .ok_or_else(|| format!("missing value for {key}"))?;
        match key {
            "--sample-rows" => args.sample_rows = parse_usize(value, key)?,
            "--min-stratum-rows" => args.min_stratum_rows = parse_usize(value, key)?,
            "--flat-blocks" => args.flat_blocks = parse_usize(value, key)?,
            "--curved-blocks" => args.curved_blocks = parse_usize(value, key)?,
            "--block-size" => args.block_size = parse_usize(value, key)?,
            "--chart-basis" => args.chart_basis = parse_usize(value, key)?,
            "--block-topk" => args.block_topk = parse_usize(value, key)?,
            "--max-epochs" => args.max_epochs = parse_usize(value, key)?,
            "--minibatch" => args.minibatch = parse_usize(value, key)?,
            "--block-tile" => args.block_tile = parse_usize(value, key)?,
            "--seed" => args.seed = value.parse::<u64>().map_err(|e| format!("{key}: {e}"))?,
            other => return Err(format!("unknown argument {other}")),
        }
        i += 2;
    }
    if args.curved_blocks == 0 {
        args.curved_blocks = (args.flat_blocks * args.block_size)
            .checked_div(args.block_size + args.chart_basis)
            .unwrap_or(1)
            .max(1);
    }
    Ok(args)
}

fn parse_usize(value: &str, key: &str) -> Result<usize, String> {
    value
        .parse::<usize>()
        .map_err(|e| format!("{key}: expected usize: {e}"))
}

fn run_layer(layer: usize, args: &Args) -> Result<Value, String> {
    let label = format!("qwen3_8b_L{layer}");
    let path = args
        .data_root
        .join("harvest_out")
        .join("qwen3_8b_wikitext")
        .join(format!("resid_L{layer}.npy"));
    println!("{label}: loading {}", path.display());
    let (mut x, source_shape) = load_even_rows(&path, args.sample_rows, args.seed + layer as u64)?;
    let sink_fraction = peel_sink(&mut x, args.seed + 10_000 + layer as u64, 3);
    let strata = build_strata(&x);
    let mut fitted = Vec::new();
    let mut skipped = Vec::new();
    for stratum in &strata {
        if stratum.rows.len() < args.min_stratum_rows {
            skipped.push(json!({
                "stratum": stratum.index,
                "n_rows": stratum.rows.len(),
                "exp_lo": stratum.exp_lo,
                "exp_hi": stratum.exp_hi,
                "reason": "below_min_stratum_rows"
            }));
            continue;
        }
        match fit_stratum(&label, stratum, &x, args) {
            Ok(value) => fitted.push(value),
            Err(err) => skipped.push(json!({
                "stratum": stratum.index,
                "n_rows": stratum.rows.len(),
                "exp_lo": stratum.exp_lo,
                "exp_hi": stratum.exp_hi,
                "reason": "fit_error",
                "message": err
            })),
        }
    }
    let total_energy = fitted
        .iter()
        .map(|v| number(v, "target_energy"))
        .sum::<Result<f64, String>>()?;
    let flat_rss = fitted
        .iter()
        .map(|v| Ok(number(v, "flat_floor")? * number(v, "target_energy")?))
        .sum::<Result<f64, String>>()?;
    let curved_rss = fitted
        .iter()
        .map(|v| Ok(number(v, "curved_floor")? * number(v, "target_energy")?))
        .sum::<Result<f64, String>>()?;
    let drops = fitted
        .iter()
        .map(|v| number(v, "drop"))
        .collect::<Result<Vec<_>, _>>()?;
    let curvatures = fitted
        .iter()
        .map(|v| number(v, "curvature_proxy"))
        .collect::<Result<Vec<_>, _>>()?;
    Ok(json!({
        "label": label,
        "path": path.display().to_string(),
        "source_shape": source_shape,
        "sample_rows": x.nrows(),
        "dimension": x.ncols(),
        "sink_top_pc_fraction": sink_fraction,
        "strata_total": strata.len(),
        "strata_fitted": fitted.len(),
        "strata_skipped": skipped,
        "covered_rows": fitted.iter().map(|v| v["n_rows"].as_u64().unwrap_or(0)).sum::<u64>(),
        "flat_blocks": args.flat_blocks,
        "curved_blocks": args.curved_blocks,
        "block_size": args.block_size,
        "chart_basis": args.chart_basis,
        "flat_total_params": args.flat_blocks * args.block_size * x.ncols(),
        "curved_total_params": args.curved_blocks * (args.block_size + args.chart_basis) * x.ncols(),
        "pooled_flat_floor": flat_rss / total_energy.max(1.0e-30),
        "pooled_curved_floor": curved_rss / total_energy.max(1.0e-30),
        "pooled_drop": (flat_rss - curved_rss) / total_energy.max(1.0e-30),
        "mean_drop": mean(&drops),
        "mean_curvature_proxy": mean(&curvatures),
        "curvature_drop_correlation": correlation(&curvatures, &drops),
        "strata": fitted
    }))
}

fn number(value: &Value, key: &str) -> Result<f64, String> {
    value[key]
        .as_f64()
        .ok_or_else(|| format!("missing numeric field {key}"))
}

fn load_even_rows(path: &Path, take: usize, phase: u64) -> Result<(Array2<f32>, Vec<usize>), String> {
    let mut file = File::open(path).map_err(|e| format!("open {}: {e}", path.display()))?;
    let header = read_npy_header(&mut file)?;
    let n_take = take.min(header.rows);
    let row_bytes = header
        .cols
        .checked_mul(std::mem::size_of::<f32>())
        .ok_or("row byte count overflow")?;
    let mut out = Array2::<f32>::zeros((n_take, header.cols));
    let mut buf = vec![0u8; row_bytes];
    let stride = (header.rows / n_take.max(1)).max(1);
    let offset = (phase as usize) % stride;
    for i in 0..n_take {
        let src_row = (offset + i * header.rows / n_take).min(header.rows - 1);
        let byte_offset = header.data_offset + (src_row * row_bytes) as u64;
        file.seek(SeekFrom::Start(byte_offset))
            .map_err(|e| format!("seek row {src_row}: {e}"))?;
        file.read_exact(&mut buf)
            .map_err(|e| format!("read row {src_row}: {e}"))?;
        for c in 0..header.cols {
            let j = c * 4;
            out[[i, c]] = f32::from_le_bytes([buf[j], buf[j + 1], buf[j + 2], buf[j + 3]]);
        }
    }
    Ok((out, vec![header.rows, header.cols]))
}

fn read_npy_header(file: &mut File) -> Result<NpyHeader, String> {
    let mut magic = [0u8; 6];
    file.read_exact(&mut magic)
        .map_err(|e| format!("read npy magic: {e}"))?;
    if &magic != b"\x93NUMPY" {
        return Err("not an npy file".to_string());
    }
    let mut version = [0u8; 2];
    file.read_exact(&mut version)
        .map_err(|e| format!("read npy version: {e}"))?;
    let header_len = if version[0] == 1 {
        let mut raw = [0u8; 2];
        file.read_exact(&mut raw)
            .map_err(|e| format!("read v1 header len: {e}"))?;
        u16::from_le_bytes(raw) as usize
    } else if version[0] == 2 || version[0] == 3 {
        let mut raw = [0u8; 4];
        file.read_exact(&mut raw)
            .map_err(|e| format!("read v2 header len: {e}"))?;
        u32::from_le_bytes(raw) as usize
    } else {
        return Err(format!("unsupported npy version {}.{}", version[0], version[1]));
    };
    let mut header_bytes = vec![0u8; header_len];
    file.read_exact(&mut header_bytes)
        .map_err(|e| format!("read npy header: {e}"))?;
    let header = String::from_utf8(header_bytes).map_err(|e| format!("header utf8: {e}"))?;
    if !(header.contains("'descr': '<f4'") || header.contains("\"descr\": \"<f4\"")) {
        return Err(format!("expected little-endian f32 npy, header={header:?}"));
    }
    if header.contains("True") {
        return Err("fortran_order=True is not supported".to_string());
    }
    let shape_start = header
        .find('(')
        .ok_or_else(|| format!("missing shape in header {header:?}"))?;
    let shape_end = header[shape_start..]
        .find(')')
        .map(|v| v + shape_start)
        .ok_or_else(|| format!("unterminated shape in header {header:?}"))?;
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
        .map_err(|e| format!("parse shape: {e}"))?;
    if dims.len() != 2 {
        return Err(format!("expected 2D shape, got {dims:?}"));
    }
    let data_offset = file
        .stream_position()
        .map_err(|e| format!("stream position: {e}"))?;
    Ok(NpyHeader {
        rows: dims[0],
        cols: dims[1],
        data_offset,
    })
}

fn peel_sink(x: &mut Array2<f32>, seed: u64, iterations: usize) -> f64 {
    let n = x.nrows();
    let p = x.ncols();
    let mut mean = vec![0.0f64; p];
    for row in x.outer_iter() {
        for c in 0..p {
            mean[c] += row[c] as f64;
        }
    }
    for v in &mut mean {
        *v /= n.max(1) as f64;
    }
    let mut total = 0.0f64;
    for mut row in x.outer_iter_mut() {
        for c in 0..p {
            row[c] -= mean[c] as f32;
            total += (row[c] as f64) * (row[c] as f64);
        }
    }
    let mut v = (0..p)
        .map(|c| (((c as u64 + 1) ^ seed) as f64 * 0.001).sin())
        .collect::<Vec<_>>();
    normalize(&mut v);
    let mut scores = vec![0.0f64; n];
    for _iteration in 0..iterations {
        for i in 0..n {
            let mut acc = 0.0f64;
            for c in 0..p {
                acc += x[[i, c]] as f64 * v[c];
            }
            scores[i] = acc;
        }
        let mut z = vec![0.0f64; p];
        for i in 0..n {
            let score = scores[i];
            for c in 0..p {
                z[c] += x[[i, c]] as f64 * score;
            }
        }
        normalize(&mut z);
        v = z;
    }
    let mut sink_energy = 0.0f64;
    for i in 0..n {
        let mut score = 0.0f64;
        for c in 0..p {
            score += x[[i, c]] as f64 * v[c];
        }
        scores[i] = score;
        sink_energy += score * score;
    }
    for i in 0..n {
        for c in 0..p {
            x[[i, c]] -= (scores[i] * v[c]) as f32;
        }
    }
    sink_energy / total.max(1.0e-30)
}

fn normalize(v: &mut [f64]) {
    let norm = v.iter().map(|x| x * x).sum::<f64>().sqrt().max(1.0e-30);
    for value in v {
        *value /= norm;
    }
}

fn build_strata(x: &Array2<f32>) -> Vec<Stratum> {
    let mut bins: Vec<Vec<usize>> = vec![Vec::new(); 2048];
    let mut energies = vec![0.0f64; x.nrows()];
    for i in 0..x.nrows() {
        let mut e = 0.0f64;
        for c in 0..x.ncols() {
            e += (x[[i, c]] as f64) * (x[[i, c]] as f64);
        }
        energies[i] = e;
        let bin = if e.is_finite() && e > 0.0 {
            ((e.to_bits() >> 52) & 0x7ff) as usize
        } else {
            0
        };
        bins[bin].push(i);
    }
    let mut strata = Vec::new();
    for (exp, rows) in bins.into_iter().enumerate() {
        if rows.is_empty() {
            continue;
        }
        let (sum, sumsq) = energy_moments(&rows, &energies);
        strata.push(Stratum {
            index: 0,
            exp_lo: exp,
            exp_hi: exp,
            rows,
            sum_energy: sum,
            sumsq_energy: sumsq,
        });
    }
    let cap = sturges_cap(x.nrows());
    while strata.len() > cap && strata.len() >= 2 {
        let first = strata.remove(0);
        let second = strata.remove(0);
        let mut rows = first.rows;
        rows.extend(second.rows);
        strata.insert(
            0,
            Stratum {
                index: 0,
                exp_lo: first.exp_lo,
                exp_hi: second.exp_hi,
                rows,
                sum_energy: first.sum_energy + second.sum_energy,
                sumsq_energy: first.sumsq_energy + second.sumsq_energy,
            },
        );
    }
    for (idx, stratum) in strata.iter_mut().enumerate() {
        stratum.index = idx;
    }
    strata
}

fn energy_moments(rows: &[usize], energies: &[f64]) -> (f64, f64) {
    let mut sum = 0.0f64;
    let mut sumsq = 0.0f64;
    for &row in rows {
        let e = energies[row];
        sum += e;
        sumsq += e * e;
    }
    (sum, sumsq)
}

fn sturges_cap(n: usize) -> usize {
    if n <= 1 {
        1
    } else {
        usize::BITS as usize - n.leading_zeros() as usize
    }
}

fn fit_stratum(
    label: &str,
    stratum: &Stratum,
    x: &Array2<f32>,
    args: &Args,
) -> Result<Value, String> {
    let xs = take_rows(x, &stratum.rows);
    println!(
        "{label} stratum {} rows={} flat_blocks={} curved_blocks={}",
        stratum.index,
        xs.nrows(),
        args.flat_blocks,
        args.curved_blocks
    );
    let flat_cfg = block_config(args.flat_blocks, args);
    let flat = fit_block_sparse_dictionary(xs.view(), &flat_cfg)?;
    let flat_recon = flat.reconstruct();
    let flat_floor = energy_floor(&xs, &flat_recon);

    let curved_cfg = block_config(args.curved_blocks, args);
    let curved = fit_block_sparse_dictionary(xs.view(), &curved_cfg)?;
    let curved_base = curved.reconstruct();
    let chart_cfg = BlockChartComposeConfig {
        block_size: args.block_size,
        block_topk: args.block_topk,
        gamma: curved.gamma,
        residual_target: true,
        min_firings: 32,
        max_blocks: args.curved_blocks,
        crossfit_folds: 2,
        alpha: 0.10,
        min_effect: 0.0,
        whitening_ridge: 1.0e-8,
        pair_screen: false,
        pair_top_blocks: 0,
        max_pairs: 0,
        pair_min_cofirings: 0,
        pair_min_score: 1.0,
    };
    let composed = compose_block_coordinate_charts(
        xs.view(),
        curved.decoder.view(),
        curved.blocks.view(),
        curved.codes.view(),
        &chart_cfg,
    )?;
    let curved_floor = energy_floor(&xs, &composed.reconstructed);
    let curvature_proxy = correction_proxy(&curved_base, &composed.reconstructed, energy(&xs));
    let records = composed
        .block_records
        .iter()
        .map(|record| {
            json!({
                "block0": record.block0,
                "screen_score": record.screen_score,
                "n_rows": record.evidence.n_rows,
                "linear_loss": record.evidence.linear_loss,
                "chart_loss": record.evidence.chart_loss,
                "deviance_gain": record.evidence.deviance_gain,
                "margin": record.evidence.margin,
                "accepted": record.evidence.accepted
            })
        })
        .collect::<Vec<_>>();
    let n = stratum.rows.len() as f64;
    let mean_energy = stratum.sum_energy / n.max(1.0);
    let var = (stratum.sumsq_energy / n.max(1.0) - mean_energy * mean_energy).max(0.0);
    Ok(json!({
        "stratum": stratum.index,
        "exp_lo": stratum.exp_lo,
        "exp_hi": stratum.exp_hi,
        "unbiased_exp_lo": if stratum.exp_lo == 0 { Value::Null } else { json!(stratum.exp_lo as i64 - 1023) },
        "unbiased_exp_hi": if stratum.exp_hi == 0 { Value::Null } else { json!(stratum.exp_hi as i64 - 1023) },
        "n_rows": stratum.rows.len(),
        "mean_energy": mean_energy,
        "std_energy": var.sqrt(),
        "target_energy": energy(&xs),
        "flat_blocks": args.flat_blocks,
        "curved_blocks": args.curved_blocks,
        "flat_params": args.flat_blocks * args.block_size * xs.ncols(),
        "curved_params": args.curved_blocks * (args.block_size + args.chart_basis) * xs.ncols(),
        "flat_floor": flat_floor,
        "curved_floor": curved_floor,
        "drop": flat_floor - curved_floor,
        "curvature_proxy": curvature_proxy,
        "flat_ev": flat.explained_variance,
        "curved_base_ev": curved.explained_variance,
        "flat_epochs": flat.epochs,
        "curved_epochs": curved.epochs,
        "flat_converged": flat.converged,
        "curved_converged": curved.converged,
        "selected_blocks": composed.selected_blocks,
        "accepted_blocks": composed.accepted_blocks,
        "n_chart_records": records.len(),
        "n_accepted_charts": composed.accepted_blocks.len(),
        "chart_records": records
    }))
}

fn block_config(n_blocks: usize, args: &Args) -> BlockSparseConfig {
    BlockSparseConfig {
        n_blocks,
        block_size: args.block_size,
        block_topk: args.block_topk,
        max_epochs: args.max_epochs,
        minibatch: args.minibatch,
        block_tile: args.block_tile,
        frame_ridge: 1.0e-9,
        aux_k: (n_blocks / 8).max(1),
        tolerance: 1.0e-5,
        ..BlockSparseConfig::default()
    }
}

fn take_rows(x: &Array2<f32>, rows: &[usize]) -> Array2<f32> {
    let mut out = Array2::<f32>::zeros((rows.len(), x.ncols()));
    for (i, &row) in rows.iter().enumerate() {
        for c in 0..x.ncols() {
            out[[i, c]] = x[[row, c]];
        }
    }
    out
}

fn energy(x: &Array2<f32>) -> f64 {
    x.iter()
        .map(|&v| {
            let y = v as f64;
            y * y
        })
        .sum()
}

fn energy_floor(target: &Array2<f32>, prediction: &Array2<f32>) -> f64 {
    let rss = target
        .iter()
        .zip(prediction.iter())
        .map(|(&x, &y)| {
            let d = (x - y) as f64;
            d * d
        })
        .sum::<f64>();
    rss / energy(target).max(1.0e-30)
}

fn correction_proxy(base: &Array2<f32>, curved: &Array2<f32>, target_energy: f64) -> f64 {
    let e = base
        .iter()
        .zip(curved.iter())
        .map(|(&x, &y)| {
            let d = (y - x) as f64;
            d * d
        })
        .sum::<f64>();
    (e / target_energy.max(1.0e-30)).sqrt()
}

fn mean(values: &[f64]) -> Value {
    if values.is_empty() {
        Value::Null
    } else {
        json!(values.iter().sum::<f64>() / values.len() as f64)
    }
}

fn correlation(xs: &[f64], ys: &[f64]) -> Value {
    if xs.len() < 2 || xs.len() != ys.len() {
        return Value::Null;
    }
    let mx = xs.iter().sum::<f64>() / xs.len() as f64;
    let my = ys.iter().sum::<f64>() / ys.len() as f64;
    let mut num = 0.0f64;
    let mut xx = 0.0f64;
    let mut yy = 0.0f64;
    for (&x, &y) in xs.iter().zip(ys) {
        let dx = x - mx;
        let dy = y - my;
        num += dx * dy;
        xx += dx * dx;
        yy += dy * dy;
    }
    let denom = (xx * yy).sqrt();
    if denom <= 0.0 {
        Value::Null
    } else {
        json!(num / denom)
    }
}

fn render_markdown(payload: &Value) -> String {
    let mut out = String::new();
    out.push_str("# Fair Wall-Closure\n\n");
    out.push_str("Actual Rust block-sparse and curved block-coordinate chart fits.\n\n");
    out.push_str("Flat and curved lanes are parameter matched by charging each curved block as `(block_size + chart_basis) * p` parameters and using fewer curved blocks.\n\n");
    out.push_str("## Pooled Layer Floors\n\n");
    out.push_str("| layer | sink frac | fitted strata | rows | flat params | curved params | flat floor | curved floor | drop | curvature/drop r |\n");
    out.push_str("| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |\n");
    if let Some(layers) = payload["layers"].as_array() {
        for layer in layers {
            out.push_str(&format!(
                "| {} | {:.6} | {}/{} | {} | {} | {} | {:.6} | {:.6} | {:.6} | {} |\n",
                layer["label"].as_str().unwrap_or("layer"),
                layer["sink_top_pc_fraction"].as_f64().unwrap_or(0.0),
                layer["strata_fitted"].as_u64().unwrap_or(0),
                layer["strata_total"].as_u64().unwrap_or(0),
                layer["covered_rows"].as_u64().unwrap_or(0),
                layer["flat_total_params"].as_u64().unwrap_or(0),
                layer["curved_total_params"].as_u64().unwrap_or(0),
                layer["pooled_flat_floor"].as_f64().unwrap_or(0.0),
                layer["pooled_curved_floor"].as_f64().unwrap_or(0.0),
                layer["pooled_drop"].as_f64().unwrap_or(0.0),
                fmt_optional(layer["curvature_drop_correlation"].as_f64())
            ));
        }
        for layer in layers {
            out.push_str(&format!(
                "\n## {} Strata\n\n",
                layer["label"].as_str().unwrap_or("layer")
            ));
            out.push_str("| stratum | rows | exp2 range | flat floor | curved floor | drop | curvature | accepted charts |\n");
            out.push_str("| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |\n");
            if let Some(strata) = layer["strata"].as_array() {
                for row in strata {
                    out.push_str(&format!(
                        "| {} | {} | {}..{} | {:.6} | {:.6} | {:.6} | {:.6} | {}/{} |\n",
                        row["stratum"].as_u64().unwrap_or(0),
                        row["n_rows"].as_u64().unwrap_or(0),
                        fmt_exp(&row["unbiased_exp_lo"]),
                        fmt_exp(&row["unbiased_exp_hi"]),
                        row["flat_floor"].as_f64().unwrap_or(0.0),
                        row["curved_floor"].as_f64().unwrap_or(0.0),
                        row["drop"].as_f64().unwrap_or(0.0),
                        row["curvature_proxy"].as_f64().unwrap_or(0.0),
                        row["n_accepted_charts"].as_u64().unwrap_or(0),
                        row["n_chart_records"].as_u64().unwrap_or(0)
                    ));
                }
            }
            if let Some(skipped) = layer["strata_skipped"].as_array() {
                if !skipped.is_empty() {
                    out.push_str("\nSkipped strata:\n");
                    for row in skipped {
                        out.push_str(&format!(
                            "- stratum {} rows={} reason={}\n",
                            row["stratum"].as_u64().unwrap_or(0),
                            row["n_rows"].as_u64().unwrap_or(0),
                            row["reason"].as_str().unwrap_or("unknown")
                        ));
                    }
                }
            }
        }
    }
    out.push_str("\nPositive drop means the curved lane has a lower reconstruction floor.\n");
    out.push_str("Curvature is the RMS chart correction energy relative to stratum target energy.\n");
    out
}

fn fmt_optional(value: Option<f64>) -> String {
    match value {
        Some(v) => format!("{v:.6}"),
        None => "NA".to_string(),
    }
}

fn fmt_exp(value: &Value) -> String {
    match value.as_i64() {
        Some(v) => v.to_string(),
        None => "NA".to_string(),
    }
}
