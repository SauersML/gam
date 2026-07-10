use gam_sae::front_door::admit_sae_fit;
use gam_sae::sparse_dict::{BlockSparseConfig, BlockSparseStreamState};
use memmap2::Mmap;
use ndarray::Array2;
use serde_json::json;
use std::fs::{self, File};
use std::io::Write;
use std::path::{Path, PathBuf};
use std::time::Instant;

#[derive(Clone, Debug)]
struct Args {
    input: PathBuf,
    out_dir: PathBuf,
    rows: usize,
    atoms: usize,
    block_size: usize,
    block_topk: usize,
    epochs: usize,
    minibatch: usize,
    block_tile: usize,
    pc_iters: usize,
    frame_ridge: f64,
    tolerance: f64,
    aux_k: usize,
    raw_ok: bool,
    post_peel: bool,
    n_peeled: usize,
    pca_dim: Option<usize>,
    gpu_policy: gam_gpu::GpuPolicy,
}

#[derive(Clone, Copy, Debug)]
struct NpyHeader {
    rows: usize,
    cols: usize,
    data_offset: usize,
}

struct NpyMatrix {
    mmap: Mmap,
    header: NpyHeader,
}

#[derive(Clone, Debug)]
struct Peel {
    mean: Vec<f64>,
    pc: Vec<f64>,
    centered_energy: f64,
    sink_energy: f64,
}

#[derive(Clone, Copy, Debug)]
struct Rss {
    current_bytes: u64,
    peak_bytes: u64,
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let args = parse_args()?;
    fs::create_dir_all(&args.out_dir)?;
    let started = Instant::now();
    let matrix = open_npy(&args.input)?;
    let n_used = args.rows.min(matrix.header.rows);
    let p = matrix.header.cols;
    if n_used == 0 || p == 0 {
        return Err("scale_k requires a non-empty N x P input".into());
    }
    if args.atoms % args.block_size != 0 {
        return Err("atoms must be divisible by block-size".into());
    }
    validate_run_contract(&args, n_used, p)?;

    // Device residency for the block-gate score-routing path (#2227/#1017). The
    // block router dispatches admitted minibatches to the CUDA device lane when
    // the mode is non-Off and a runtime is present. Under `--gpu required` a
    // missing runtime or a below-break-even minibatch is a hard, up-front error
    // (fail fast) rather than a silent CPU run or a GPU slot burned on a late
    // crash; the per-minibatch admission uses the same `minibatch × K` score
    // floor the router enforces.
    gam_gpu::configure_global_policy(args.gpu_policy);
    let route_plan = gam_gpu::DictionaryScoreRoutePlan::default_for_shape(
        args.minibatch.min(n_used),
        args.atoms,
        p,
    );
    println!(
        "[scale_k] gpu={:?} per-minibatch score elems = {} (device_admitted={}, break-even={})",
        args.gpu_policy,
        args.minibatch.min(n_used).saturating_mul(args.atoms),
        route_plan.device_admitted,
        route_plan.device_min_score_elems,
    );
    if args.gpu_policy == gam_gpu::GpuPolicy::Required {
        if gam_gpu::GpuRuntime::global().is_none() {
            return Err(
                "--gpu required but no CUDA runtime is available on this host (run on the A100 box)"
                    .into(),
            );
        }
        if !route_plan.device_admitted {
            return Err(format!(
                "--gpu required but minibatch {} x K {} = {} score elems is below the device \
                 launch break-even {}; raise --minibatch to at least {}",
                args.minibatch.min(n_used),
                args.atoms,
                args.minibatch.min(n_used).saturating_mul(args.atoms),
                route_plan.device_min_score_elems,
                route_plan
                    .device_min_score_elems
                    .div_ceil(args.atoms.max(1)),
            )
            .into());
        }
    }

    let admission = admit_sae_fit(n_used, p, args.atoms)?;
    if !admission.uses_sparse_codes() {
        return Err(format!(
            "front_door did not admit sparse/block lane for N={n_used}, P={p}, K={}",
            args.atoms
        )
        .into());
    }

    let rss_start = read_rss();
    let peel_start = Instant::now();
    let peel = compute_top_pc_peel(&matrix, n_used, args.pc_iters)?;
    let peel_seconds = peel_start.elapsed().as_secs_f64();
    let rss_after_peel = read_rss();

    let decoder_start = Instant::now();
    let decoder = coordinate_block_decoder(args.atoms, p, args.block_size);
    let decoder_seconds = decoder_start.elapsed().as_secs_f64();
    let rss_after_decoder = read_rss();

    let mut cfg = BlockSparseConfig::new(args.atoms / args.block_size, args.block_size);
    cfg.block_topk = args.block_topk;
    cfg.max_epochs = args.epochs;
    cfg.minibatch = args.minibatch;
    cfg.block_tile = args.block_tile;
    cfg.frame_ridge = args.frame_ridge;
    cfg.tolerance = args.tolerance;
    cfg.aux_k = args.aux_k;

    let state_start = Instant::now();
    let mut state = BlockSparseStreamState::new_with_decoder(decoder, &cfg)?;
    let state_seconds = state_start.elapsed().as_secs_f64();
    let rss_after_state = read_rss();

    let train_start = Instant::now();
    let mut shard = Array2::<f32>::zeros((args.minibatch.min(n_used), p));
    let mut epoch_reports = Vec::new();
    let mut last_ev = f64::NEG_INFINITY;
    for epoch_index in 0..args.epochs {
        let mut row0 = 0usize;
        let mut shard_reports = Vec::new();
        while row0 < n_used {
            let take = (n_used - row0).min(shard.nrows());
            fill_transformed_rows(
                &matrix,
                row0,
                take,
                &peel,
                shard.slice_mut(ndarray::s![0..take, ..]),
            )?;
            let stats = state.partial_fit(shard.slice(ndarray::s![0..take, ..]))?;
            shard_reports.push(json!({
                "row0": row0,
                "rows": stats.rows,
                "rss": stats.rss,
                "alive_blocks": stats.alive_blocks,
                "peak_rss_bytes": read_rss().peak_bytes,
            }));
            row0 += take;
        }
        let stats = state.end_epoch()?;
        let solve = stats.decoder_solve_stats;
        let cg_rate_bound_base = solve.cg_kappa_hat.map(|kappa| {
            let root = kappa.sqrt();
            (root - 1.0) / (root + 1.0)
        });
        last_ev = stats.explained_variance;
        // Coarse per-epoch liveness for the CUDA validation run (a handful of
        // lines, never per-shard): a stalled device route stops advancing this
        // line, and under `--gpu required` a route that cannot make progress is a
        // typed error rather than a silent hang.
        println!(
            "[scale_k] epoch {}/{} ev={:.6} revived={} dead={} elapsed={:.1}s",
            epoch_index + 1,
            args.epochs,
            stats.explained_variance,
            stats.revived,
            stats.dead,
            started.elapsed().as_secs_f64(),
        );
        epoch_reports.push(json!({
            "epoch": epoch_index + 1,
            "reported_epoch": stats.epoch,
            "explained_variance": stats.explained_variance,
            "revived": stats.revived,
            "dead": stats.dead,
            "gamma": stats.gamma,
            "converged": stats.converged,
            "refresh_solver": {
                "default": "matrix_free_cg_streamed_sparse_mod",
                "tiny_component_solver": "dense_cholesky",
                "mean_cofiring_degree": solve.mean_cofiring_degree,
                "giant_component_fraction": solve.giant_component_fraction,
                "component_count": solve.component_count,
                "max_component_size": solve.max_component_size,
                "cg_columns": solve.cg_columns,
                "cg_iterations": solve.cg_iterations,
                "cg_kappa_hat": solve.cg_kappa_hat,
                "cg_rate_bound_base": cg_rate_bound_base,
                "cg_relative_residual": solve.cg_relative_residual,
                "cg_residual_stop": solve.cg_residual_stop,
                "stopping_rule": "relative normal-equation residual <= frame_ridge rank-charge floor",
                "minibatch_admission": "refresh atom k only when n_k >= (z_alpha*sigma/(a_bar_k*margin_k))^2; otherwise accumulate",
            },
            "shards": shard_reports.len(),
        }));
        if stats.converged {
            break;
        }
    }
    let train_seconds = train_start.elapsed().as_secs_f64();
    let rss_after_train = read_rss();

    let charge_start = Instant::now();
    let charges = state.block_rank_charges(n_used)?;
    let charge_seconds = charge_start.elapsed().as_secs_f64();
    let mut curved_blocks = 0usize;
    let mut linear_blocks = 0usize;
    let mut active_blocks = 0usize;
    let mut kept_blocks = 0usize;
    for i in 0..charges.block.len() {
        if charges.n_eff[i] > 0.0 {
            active_blocks += 1;
        }
        if charges.kept[i] {
            kept_blocks += 1;
            if charges.d_eff[i] > 1.0 {
                curved_blocks += 1;
            } else {
                linear_blocks += 1;
            }
        }
    }
    let inactive_or_rejected_blocks = cfg
        .n_blocks
        .saturating_sub(curved_blocks)
        .saturating_sub(linear_blocks);
    let rss_final = read_rss();

    let response_bytes = bytes_nxp(n_used, p);
    let dense_nxk_bytes = bytes_nxp(n_used, args.atoms);
    let expected_payload_bytes = expected_payload_upper_bytes(&args, n_used, p);
    let measured_peak = [
        rss_start.peak_bytes,
        rss_after_peel.peak_bytes,
        rss_after_decoder.peak_bytes,
        rss_after_state.peak_bytes,
        rss_after_train.peak_bytes,
        rss_final.peak_bytes,
    ]
    .into_iter()
    .max()
    .unwrap_or(rss_final.peak_bytes);
    let no_dense_nxk_assertion = measured_peak < dense_nxk_bytes;
    let no_second_nxp_assertion =
        measured_peak < rss_start.current_bytes + expected_payload_bytes + response_bytes;
    if !no_dense_nxk_assertion {
        return Err(format!(
            "peak RSS {} was not below dense N x K payload {}",
            measured_peak, dense_nxk_bytes
        )
        .into());
    }
    if !no_second_nxp_assertion {
        return Err(format!(
            "peak RSS {} exceeded expected streaming payload plus one N x P margin {}",
            measured_peak,
            rss_start.current_bytes + expected_payload_bytes + response_bytes
        )
        .into());
    }

    let elapsed = started.elapsed().as_secs_f64();
    let last_refresh_solver = epoch_reports
        .last()
        .map(|epoch| epoch["refresh_solver"].clone())
        .unwrap_or_else(|| json!({}));
    let result = json!({
        "experiment": "scale_k_curved_block_front_door",
        "input": args.input.display().to_string(),
        "source_shape": [matrix.header.rows, matrix.header.cols],
        "run_contract": {
            "N": n_used,
            "p": p,
            "K": args.atoms,
            "post_peel": args.post_peel,
            "n_peeled": args.n_peeled,
            "pca_dim": args.pca_dim,
            "peak_rss": measured_peak,
            "no_dense_nxk_allocation_by_peak_rss": no_dense_nxk_assertion,
            "raw_ok": args.raw_ok,
        },
        "N": n_used,
        "K": args.atoms,
        "post_peel": args.post_peel,
        "n_peeled": args.n_peeled,
        "pca_dim": args.pca_dim,
        "peak_rss": measured_peak,
        "rows_used": n_used,
        "p": p,
        "largest_k_reached": args.atoms,
        "n_blocks": cfg.n_blocks,
        "block_size": args.block_size,
        "block_topk": args.block_topk,
        "epochs_requested": args.epochs,
        "epochs_run": state.epochs_run(),
        "minibatch": args.minibatch,
        "block_tile": args.block_tile,
        "front_door": {
            "lane": format!("{:?}", admission.lane),
            "dense_assignment_cells": admission.dense_assignment_cells,
            "response_cells": admission.response_cells,
            "dense_assignment_bytes_f32": dense_nxk_bytes,
            "response_bytes_f32": response_bytes,
        },
        "peel": {
            "method": "streaming centered top principal component",
            "pc_iters": args.pc_iters,
            "centered_energy": peel.centered_energy,
            "sink_energy": peel.sink_energy,
            // 1.0e-30 is a divide-by-zero floor on the denominator, well below any real energy.
            "absorbed_centered_fraction": peel.sink_energy / peel.centered_energy.max(1.0e-30),
            "seconds": peel_seconds,
        },
        "timing": {
            "decoder_init_seconds": decoder_seconds,
            "state_init_seconds": state_seconds,
            "train_seconds": train_seconds,
            "charge_seconds": charge_seconds,
            "wall_seconds": elapsed,
            // 1.0e-30 is a divide-by-zero floor on the elapsed-seconds denominator.
            "rows_per_train_second": n_used as f64 * state.epochs_run() as f64 / train_seconds.max(1.0e-30),
        },
        "rss": {
            "start_current_bytes": rss_start.current_bytes,
            "after_peel_peak_bytes": rss_after_peel.peak_bytes,
            "after_decoder_peak_bytes": rss_after_decoder.peak_bytes,
            "after_state_peak_bytes": rss_after_state.peak_bytes,
            "after_train_peak_bytes": rss_after_train.peak_bytes,
            "final_peak_bytes": rss_final.peak_bytes,
            "measured_peak_bytes": measured_peak,
            "expected_payload_upper_bytes": expected_payload_bytes,
        },
        "invariants": {
            "sparse_front_door": admission.uses_sparse_codes(),
            "no_dense_nxk_by_peak_rss": no_dense_nxk_assertion,
            "no_second_full_nxp_by_payload_bound": no_second_nxp_assertion,
        },
        "fit": {
            "final_ev": last_ev,
            "active_blocks": active_blocks,
            "kept_blocks": kept_blocks,
            "curved_blocks": curved_blocks,
            "linear_blocks": linear_blocks,
            "inactive_or_rejected_blocks": inactive_or_rejected_blocks,
            "curved_atoms": curved_blocks * args.block_size,
            "linear_atoms": linear_blocks * args.block_size,
            "inactive_or_rejected_atoms": inactive_or_rejected_blocks * args.block_size,
        },
        "last_refresh_solver": last_refresh_solver,
        "epochs": epoch_reports,
    });

    let numbers_path = args.out_dir.join("numbers.json");
    let mut numbers = File::create(&numbers_path)?;
    numbers.write_all(serde_json::to_string_pretty(&result)?.as_bytes())?;
    numbers.write_all(b"\n")?;

    let results_path = args.out_dir.join("results.md");
    let mut results = File::create(&results_path)?;
    results.write_all(render_markdown(&result).as_bytes())?;

    println!("wrote {}", numbers_path.display());
    println!("wrote {}", results_path.display());
    Ok(())
}

fn parse_args() -> Result<Args, String> {
    let raw: Vec<String> = std::env::args().collect();
    if raw.len() < 3 {
        return Err(
            "usage: scale_k <input.npy> <out-dir> [--rows N] [--atoms K] [--epochs N] \
             [--minibatch M] [--gpu required|auto|off] \
             --post-peel --n-peeled N --pca-dim D [--raw-ok]"
                .to_string(),
        );
    }
    let mut args = Args {
        input: PathBuf::from(&raw[1]),
        out_dir: PathBuf::from(&raw[2]),
        rows: usize::MAX,
        atoms: 100_000, // reporting-only: demo default dictionary size for the scaling example.
        block_size: 4,
        block_topk: 2,
        epochs: 1,
        minibatch: 128,
        block_tile: 1024,
        pc_iters: 3,
        frame_ridge: 1.0e-9, // reporting-only: demo default frame ridge for the scaling example.
        tolerance: 1.0e-5,   // reporting-only: demo default convergence tolerance.
        aux_k: 0,
        raw_ok: false,
        post_peel: false,
        n_peeled: 0,
        pca_dim: None,
        // Default Auto: the block router dispatches admitted minibatches to the
        // CUDA block-gate device path on a CUDA host, else the CPU router.
        gpu_policy: gam_gpu::GpuPolicy::Auto,
    };
    let mut i = 3usize;
    while i < raw.len() {
        let key = raw[i].as_str();
        match key {
            "--raw-ok" => {
                args.raw_ok = true;
                i += 1;
                continue;
            }
            "--post-peel" => {
                args.post_peel = true;
                i += 1;
                continue;
            }
            _ => {}
        }
        let value = raw
            .get(i + 1)
            .ok_or_else(|| format!("missing value for {key}"))?;
        match key {
            "--rows" => args.rows = parse_usize(value, key)?,
            "--atoms" => args.atoms = parse_usize(value, key)?,
            "--block-size" => args.block_size = parse_usize(value, key)?,
            "--block-topk" => args.block_topk = parse_usize(value, key)?,
            "--epochs" => args.epochs = parse_usize(value, key)?,
            "--minibatch" => args.minibatch = parse_usize(value, key)?,
            "--block-tile" => args.block_tile = parse_usize(value, key)?,
            "--pc-iters" => args.pc_iters = parse_usize(value, key)?,
            "--frame-ridge" => args.frame_ridge = parse_f64(value, key)?,
            "--tolerance" => args.tolerance = parse_f64(value, key)?,
            "--aux-k" => args.aux_k = parse_usize(value, key)?,
            "--n-peeled" => args.n_peeled = parse_usize(value, key)?,
            "--pca-dim" => args.pca_dim = Some(parse_usize(value, key)?),
            "--gpu" => {
                args.gpu_policy = gam_gpu::GpuPolicy::parse(value)
                    .ok_or_else(|| format!("--gpu must be required|auto|off, got {value}"))?;
            }
            other => return Err(format!("unknown argument {other}")),
        }
        i += 2;
    }
    if args.block_size == 0 || args.block_topk == 0 || args.epochs == 0 || args.minibatch == 0 {
        return Err("block-size, block-topk, epochs, and minibatch must be positive".to_string());
    }
    Ok(args)
}

fn validate_run_contract(args: &Args, n_used: usize, p: usize) -> Result<(), String> {
    if args.raw_ok {
        if let Some(pca_dim) = args.pca_dim {
            validate_pca_dim(pca_dim, n_used, p)?;
        }
        return Ok(());
    }
    if !args.post_peel {
        return Err(
            "scale_k refuses raw/full-width activation runs by default: pass a post-peel, \
             PCA-reduced input with --post-peel --n-peeled N --pca-dim D, or pass --raw-ok \
             to intentionally run the raw matrix"
                .to_string(),
        );
    }
    if args.n_peeled == 0 {
        return Err(
            "scale_k requires --n-peeled > 0 for the default post-peel/PCA-reduced run".to_string(),
        );
    }
    let pca_dim = args.pca_dim.ok_or_else(|| {
        "scale_k requires --pca-dim D for the default post-peel/PCA-reduced run".to_string()
    })?;
    validate_pca_dim(pca_dim, n_used, p)?;
    Ok(())
}

fn validate_pca_dim(pca_dim: usize, n_used: usize, p: usize) -> Result<(), String> {
    if pca_dim == 0 {
        return Err("--pca-dim must be positive".to_string());
    }
    if pca_dim != p {
        return Err(format!(
            "--pca-dim must match the input column count after PCA reduction: pca_dim={pca_dim}, input_p={p}"
        ));
    }
    if pca_dim >= n_used {
        return Err(format!(
            "--pca-dim must be smaller than the rows used so the run is genuinely PCA-reduced: pca_dim={pca_dim}, N={n_used}"
        ));
    }
    Ok(())
}

fn parse_usize(value: &str, key: &str) -> Result<usize, String> {
    value
        .parse::<usize>()
        .map_err(|e| format!("{key}: expected usize: {e}"))
}

fn parse_f64(value: &str, key: &str) -> Result<f64, String> {
    value
        .parse::<f64>()
        .map_err(|e| format!("{key}: expected f64: {e}"))
}

fn open_npy(path: &Path) -> Result<NpyMatrix, String> {
    let file = File::open(path).map_err(|e| format!("open {}: {e}", path.display()))?;
    // SAFETY: the experiment opens the input read-only and never writes through
    // this mapping; the banked activation file is immutable for the run.
    let mmap = unsafe { Mmap::map(&file).map_err(|e| format!("mmap {}: {e}", path.display()))? };
    let header = parse_npy_header(&mmap, path)?;
    let row_bytes = header
        .cols
        .checked_mul(std::mem::size_of::<f32>())
        .ok_or("row byte overflow")?;
    let data_bytes = header
        .rows
        .checked_mul(row_bytes)
        .ok_or("data byte overflow")?;
    let end = header
        .data_offset
        .checked_add(data_bytes)
        .ok_or("npy end offset overflow")?;
    if end > mmap.len() {
        return Err(format!(
            "{} is truncated: data end {} > file len {}",
            path.display(),
            end,
            mmap.len()
        ));
    }
    Ok(NpyMatrix { mmap, header })
}

fn parse_npy_header(bytes: &[u8], path: &Path) -> Result<NpyHeader, String> {
    // 12 = npy fixed preamble: 6-byte magic + 2 version bytes + 4-byte (v2/v3) header length.
    if bytes.len() < 12 || &bytes[0..6] != b"\x93NUMPY" {
        return Err(format!("{} is not a .npy file", path.display()));
    }
    let major = bytes[6];
    let (header_len, data_offset) = if major == 1 {
        let len = u16::from_le_bytes([bytes[8], bytes[9]]) as usize;
        (len, 10usize + len)
    } else if major == 2 || major == 3 {
        let len = u32::from_le_bytes([bytes[8], bytes[9], bytes[10], bytes[11]]) as usize;
        (len, 12usize + len) // 12 = npy v2/v3 preamble length before the header dict.
    } else {
        return Err(format!("unsupported npy version {}.{}", major, bytes[7]));
    };
    if data_offset > bytes.len() {
        return Err(format!("{} has a truncated npy header", path.display()));
    }
    let header_start = data_offset - header_len;
    let header = std::str::from_utf8(&bytes[header_start..data_offset])
        .map_err(|e| format!("{} header is not UTF-8: {e}", path.display()))?;
    if !(header.contains("'descr': '<f4'") || header.contains("\"descr\": \"<f4\"")) {
        return Err(format!(
            "{} must be little-endian float32, header={header:?}",
            path.display()
        ));
    }
    if header.contains("True") {
        return Err(format!(
            "{} must be C-order, header={header:?}",
            path.display()
        ));
    }
    let shape_open = header
        .find('(')
        .ok_or_else(|| format!("{} header missing shape", path.display()))?;
    let shape_close = header[shape_open..]
        .find(')')
        .map(|v| v + shape_open)
        .ok_or_else(|| format!("{} header has unterminated shape", path.display()))?;
    let dims = header[shape_open + 1..shape_close]
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
        .map_err(|e| format!("{} shape parse error: {e}", path.display()))?;
    if dims.len() != 2 {
        return Err(format!(
            "{} must be rank-2, got shape {dims:?}",
            path.display()
        ));
    }
    Ok(NpyHeader {
        rows: dims[0],
        cols: dims[1],
        data_offset,
    })
}

fn compute_top_pc_peel(matrix: &NpyMatrix, rows: usize, pc_iters: usize) -> Result<Peel, String> {
    let p = matrix.header.cols;
    let mut mean = vec![0.0f64; p];
    for row in 0..rows {
        for c in 0..p {
            mean[c] += matrix.value(row, c)? as f64;
        }
    }
    for value in &mut mean {
        *value /= rows as f64;
    }

    let mut centered_energy = 0.0f64;
    for row in 0..rows {
        for (c, &mc) in mean.iter().enumerate() {
            let v = matrix.value(row, c)? as f64 - mc;
            centered_energy += v * v;
        }
    }

    let mut pc = deterministic_unit_vector(p);
    let mut scores = vec![0.0f64; rows];
    for iteration_index in 0..pc_iters {
        for (row, score_ref) in scores.iter_mut().enumerate().take(rows) {
            let mut score = 0.0f64;
            for (c, (&mc, &vc)) in mean.iter().zip(pc.iter()).enumerate() {
                score += (matrix.value(row, c)? as f64 - mc) * vc;
            }
            *score_ref = score;
        }
        let mut next = vec![0.0f64; p];
        for (row, &score) in scores.iter().enumerate().take(rows) {
            for (c, (&mc, next_c)) in mean.iter().zip(next.iter_mut()).enumerate() {
                *next_c += (matrix.value(row, c)? as f64 - mc) * score;
            }
        }
        normalize(&mut next);
        pc = next;
        if iteration_index + 1 == pc_iters {
            continue;
        }
    }

    let mut sink_energy = 0.0f64;
    for row in 0..rows {
        let mut score = 0.0f64;
        for (c, (&mc, &vc)) in mean.iter().zip(pc.iter()).enumerate() {
            score += (matrix.value(row, c)? as f64 - mc) * vc;
        }
        sink_energy += score * score;
    }

    Ok(Peel {
        mean,
        pc,
        centered_energy,
        sink_energy,
    })
}

impl NpyMatrix {
    fn value(&self, row: usize, col: usize) -> Result<f32, String> {
        if row >= self.header.rows || col >= self.header.cols {
            return Err(format!(
                "npy index out of bounds ({row}, {col}) for shape ({}, {})",
                self.header.rows, self.header.cols
            ));
        }
        let index = row
            .checked_mul(self.header.cols)
            .and_then(|v| v.checked_add(col))
            .ok_or("npy element index overflow")?;
        let offset = self
            .header
            .data_offset
            .checked_add(index * std::mem::size_of::<f32>())
            .ok_or("npy byte offset overflow")?;
        Ok(f32::from_le_bytes([
            self.mmap[offset],
            self.mmap[offset + 1],
            self.mmap[offset + 2],
            self.mmap[offset + 3],
        ]))
    }
}

fn fill_transformed_rows(
    matrix: &NpyMatrix,
    row0: usize,
    rows: usize,
    peel: &Peel,
    mut out: ndarray::ArrayViewMut2<'_, f32>,
) -> Result<(), String> {
    let p = matrix.header.cols;
    for local in 0..rows {
        let source_row = row0 + local;
        let mut score = 0.0f64;
        for c in 0..p {
            score += (matrix.value(source_row, c)? as f64 - peel.mean[c]) * peel.pc[c];
        }
        for c in 0..p {
            let centered = matrix.value(source_row, c)? as f64 - peel.mean[c];
            out[[local, c]] = (centered - score * peel.pc[c]) as f32;
        }
    }
    Ok(())
}

fn deterministic_unit_vector(p: usize) -> Vec<f64> {
    let mut state = 0x9e37_79b9_7f4a_7c15u64;
    let mut out = vec![0.0f64; p];
    for value in &mut out {
        state = splitmix64(state);
        let unit = ((state >> 11) as f64) * (1.0 / ((1u64 << 53) as f64));
        *value = unit - 0.5;
    }
    normalize(&mut out);
    out
}

fn normalize(v: &mut [f64]) {
    // 1.0e-30 is a divide-by-zero floor on the vector norm before normalization.
    let norm = v.iter().map(|x| x * x).sum::<f64>().sqrt().max(1.0e-30);
    for value in v {
        *value /= norm;
    }
}

fn coordinate_block_decoder(atoms: usize, p: usize, b: usize) -> Array2<f32> {
    let mut decoder = Array2::<f32>::zeros((atoms, p));
    let blocks = atoms / b;
    let mut state = 0xd1b5_4a32_d192_ed03u64;
    for block in 0..blocks {
        let mut used = Vec::with_capacity(b);
        for r in 0..b {
            state = splitmix64(state ^ block as u64 ^ ((r as u64) << 32));
            let mut coord = (state as usize) % p;
            while used.contains(&coord) {
                coord = (coord + 1) % p;
            }
            used.push(coord);
            let sign = if (state >> 63) == 0 { 1.0 } else { -1.0 };
            decoder[[block * b + r, coord]] = sign;
        }
    }
    decoder
}

fn splitmix64(mut x: u64) -> u64 {
    x = x.wrapping_add(0x9e37_79b9_7f4a_7c15);
    let mut z = x;
    z = (z ^ (z >> 30)).wrapping_mul(0xbf58_476d_1ce4_e5b9);
    z = (z ^ (z >> 27)).wrapping_mul(0x94d0_49bb_1331_11eb);
    z ^ (z >> 31)
}

fn read_rss() -> Rss {
    let status = std::fs::read_to_string("/proc/self/status").unwrap_or_default();
    let mut current_kb = 0u64;
    let mut peak_kb = 0u64;
    for line in status.lines() {
        if let Some(rest) = line.strip_prefix("VmRSS:") {
            current_kb = parse_status_kb(rest);
        } else if let Some(rest) = line.strip_prefix("VmHWM:") {
            peak_kb = parse_status_kb(rest);
        }
    }
    Rss {
        current_bytes: current_kb * 1024,
        peak_bytes: peak_kb * 1024,
    }
}

fn parse_status_kb(rest: &str) -> u64 {
    rest.split_whitespace()
        .next()
        .and_then(|v| v.parse::<u64>().ok())
        .unwrap_or(0)
}

fn bytes_nxp(n: usize, p: usize) -> u64 {
    n.saturating_mul(p)
        .saturating_mul(std::mem::size_of::<f32>()) as u64
}

fn expected_payload_upper_bytes(args: &Args, rows: usize, p: usize) -> u64 {
    let blocks = args.atoms / args.block_size;
    let decoder = bytes_nxp(args.atoms, p);
    let normal_rhs = blocks
        .saturating_mul(p)
        .saturating_mul(args.block_size)
        .saturating_mul(std::mem::size_of::<f64>()) as u64;
    let cofiring_edges = blocks
        .saturating_mul(args.block_topk)
        .saturating_mul(args.block_topk.saturating_sub(1))
        .saturating_div(2)
        .saturating_mul(args.block_size)
        .saturating_mul(args.block_size)
        .saturating_mul(std::mem::size_of::<((u32, u32), f64)>()) as u64;
    let second = blocks
        .saturating_mul(args.block_size)
        .saturating_mul(args.block_size)
        .saturating_mul(std::mem::size_of::<f64>())
        .saturating_mul(2) as u64;
    let shard_rows = args.minibatch.min(rows);
    let shard = bytes_nxp(shard_rows, p);
    let scores = shard_rows
        .saturating_mul(args.block_tile)
        .saturating_mul(args.block_size)
        .saturating_mul(std::mem::size_of::<f32>()) as u64;
    let codes = shard_rows
        .saturating_mul(args.block_topk)
        .saturating_mul(args.block_size + 2)
        .saturating_mul(std::mem::size_of::<f32>()) as u64;
    decoder + normal_rhs + cofiring_edges + second + shard + scores + codes
}

fn render_markdown(value: &serde_json::Value) -> String {
    let fit = &value["fit"];
    let rss = &value["rss"];
    let timing = &value["timing"];
    let front = &value["front_door"];
    let peel = &value["peel"];
    let contract = &value["run_contract"];
    let solver = &value["last_refresh_solver"];
    let cg_kappa_hat = solver["cg_kappa_hat"]
        .as_f64()
        .map(|v| format!("{v:.6}"))
        .unwrap_or_else(|| "null".to_string());
    let cg_rate_bound_base = solver["cg_rate_bound_base"]
        .as_f64()
        .map(|v| format!("{v:.6}"))
        .unwrap_or_else(|| "null".to_string());
    format!(
        "# Curved Atoms at Scale\n\n\
         Input: `{}`\n\n\
         | rows | P | K | blocks | block size | block top-k | lane |\n\
         | ---: | ---: | ---: | ---: | ---: | ---: | --- |\n\
         | {} | {} | {} | {} | {} | {} | {} |\n\n\
         ## Run Contract\n\n\
         - N: `{}`; p: `{}`; K: `{}`.\n\
         - Post-peel: `{}` with `{}` peeled direction(s); PCA dim: `{}`.\n\
         - Peak RSS: `{}` bytes; no dense N x K allocation by peak RSS: `{}`.\n\n\
         | metric | value |\n\
         | --- | ---: |\n\
         | wall seconds | {:.3} |\n\
         | train seconds | {:.3} |\n\
         | rows / train second | {:.3} |\n\
         | peak RSS bytes | {} |\n\
         | dense N x K f32 bytes | {} |\n\
         | response N x P f32 bytes | {} |\n\
         | final EV | {:.8} |\n\
         | active blocks | {} |\n\
         | kept blocks | {} |\n\
         | curved atoms | {} |\n\
         | linear atoms | {} |\n\
         | inactive or rejected atoms | {} |\n\n\
         ## Front-Door Invariants\n\n\
         - Sparse front door: `{}`.\n\
         - Peak RSS below dense N x K payload: `{}`.\n\
         - Peak RSS below streaming payload plus one full N x P margin: `{}`.\n\
         - The streaming path stores block routes shard-local only; it does not persist an N x K assignment or a second full response matrix.\n\n\
         ## Refresh Solver\n\n\
         - Default: `{}`; tiny components use `{}`.\n\
         - Stopping rule: `{}`.\n\
         - Routability admission: `{}`.\n\
         - Mean co-firing degree: `{:.6}`; giant-component fraction: `{:.6}`; CG kappa-hat: `{}`; rate bound base: `{}`.\n\
         - CG columns: `{}`; CG iterations: `{}`; relative residual: `{:.6e}`; residual stop: `{:.6e}`.\n\n\
         ## Position-0 Peel\n\n\
         Method: {}. Absorbed centered fraction: {:.8}. Peel seconds: {:.3}.\n\n\
         ## Notes\n\n\
         Curved atoms are counted as atoms in evidence-kept blocks whose realised rank charge exceeds one; kept one-dimensional blocks are counted as linear. Blocks that did not clear the rank-charge margin are reported separately as inactive or rejected.\n",
        value["input"].as_str().unwrap_or(""),
        value["rows_used"].as_u64().unwrap_or(0),
        value["p"].as_u64().unwrap_or(0),
        value["largest_k_reached"].as_u64().unwrap_or(0),
        value["n_blocks"].as_u64().unwrap_or(0),
        value["block_size"].as_u64().unwrap_or(0),
        value["block_topk"].as_u64().unwrap_or(0),
        front["lane"].as_str().unwrap_or(""),
        contract["N"].as_u64().unwrap_or(0),
        contract["p"].as_u64().unwrap_or(0),
        contract["K"].as_u64().unwrap_or(0),
        contract["post_peel"].as_bool().unwrap_or(false),
        contract["n_peeled"].as_u64().unwrap_or(0),
        contract["pca_dim"]
            .as_u64()
            .map(|v| v.to_string())
            .unwrap_or_else(|| "null".to_string()),
        contract["peak_rss"].as_u64().unwrap_or(0),
        contract["no_dense_nxk_allocation_by_peak_rss"]
            .as_bool()
            .unwrap_or(false),
        timing["wall_seconds"].as_f64().unwrap_or(0.0),
        timing["train_seconds"].as_f64().unwrap_or(0.0),
        timing["rows_per_train_second"].as_f64().unwrap_or(0.0),
        rss["measured_peak_bytes"].as_u64().unwrap_or(0),
        front["dense_assignment_bytes_f32"].as_u64().unwrap_or(0),
        front["response_bytes_f32"].as_u64().unwrap_or(0),
        fit["final_ev"].as_f64().unwrap_or(0.0),
        fit["active_blocks"].as_u64().unwrap_or(0),
        fit["kept_blocks"].as_u64().unwrap_or(0),
        fit["curved_atoms"].as_u64().unwrap_or(0),
        fit["linear_atoms"].as_u64().unwrap_or(0),
        fit["inactive_or_rejected_atoms"].as_u64().unwrap_or(0),
        value["invariants"]["sparse_front_door"]
            .as_bool()
            .unwrap_or(false),
        value["invariants"]["no_dense_nxk_by_peak_rss"]
            .as_bool()
            .unwrap_or(false),
        value["invariants"]["no_second_full_nxp_by_payload_bound"]
            .as_bool()
            .unwrap_or(false),
        solver["default"].as_str().unwrap_or(""),
        solver["tiny_component_solver"].as_str().unwrap_or(""),
        solver["stopping_rule"].as_str().unwrap_or(""),
        solver["minibatch_admission"].as_str().unwrap_or(""),
        solver["mean_cofiring_degree"].as_f64().unwrap_or(0.0),
        solver["giant_component_fraction"].as_f64().unwrap_or(0.0),
        cg_kappa_hat,
        cg_rate_bound_base,
        solver["cg_columns"].as_u64().unwrap_or(0),
        solver["cg_iterations"].as_u64().unwrap_or(0),
        solver["cg_relative_residual"].as_f64().unwrap_or(0.0),
        solver["cg_residual_stop"].as_f64().unwrap_or(0.0),
        peel["method"].as_str().unwrap_or(""),
        peel["absorbed_centered_fraction"].as_f64().unwrap_or(0.0),
        peel["seconds"].as_f64().unwrap_or(0.0),
    )
}
