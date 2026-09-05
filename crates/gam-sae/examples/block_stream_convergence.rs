//! Measure streamed gamma/frame convergence on a bounded prefix of an NPY corpus.
//! Usage: block_stream_convergence corpus.npy rows atoms block_size topk epochs

use gam_sae::sparse_dict::{
    BlockSparseConfig, BlockSparseStreamState, block_sparse_dictionary_transform,
};
use ndarray::{Array2, ArrayView2};
use rayon::prelude::*;
use std::io::Write;
use std::path::Path;
use std::time::Instant;

#[path = "support/f16.rs"]
mod f16;
#[path = "support/npy_header.rs"]
mod npy_header;

// Independently price the last proposal with the old and freshly chosen
// supports. This separates frame descent from a discontinuity at a top-k tie.
fn audit_proposal(
    x: ArrayView2<'_, f32>,
    before: ArrayView2<'_, f32>,
    after: ArrayView2<'_, f32>,
    gamma: f32,
    config: &BlockSparseConfig,
) -> Result<serde_json::Value, String> {
    let b = config.block_size;
    let route = |decoder| {
        block_sparse_dictionary_transform(x, decoder, 1.0, b, config.block_topk, config.block_tile)
    };
    let (old_blocks, old_gates, _) = route(before)?;
    let (new_blocks, new_gates, _) = route(after)?;
    let row_stats: Vec<_> = (0..x.nrows())
        .into_par_iter()
        .map(|row| {
            let project =
                |decoder: ArrayView2<'_, f32>, blocks: &Array2<u32>, gates: &Array2<f32>| {
                    let mut total = vec![0.0_f64; x.ncols()];
                    for slot in 0..blocks.ncols() {
                        if gates[[row, slot]] == 0.0 {
                            continue;
                        }
                        let block = blocks[[row, slot]] as usize;
                        for axis in 0..b {
                            let direction = decoder.row(block * b + axis);
                            let weight: f64 = direction
                                .iter()
                                .zip(x.row(row).iter())
                                .map(|(&u, &v)| u as f64 * v as f64)
                                .sum();
                            for (out, &u) in total.iter_mut().zip(direction.iter()) {
                                *out += weight * u as f64;
                            }
                        }
                    }
                    total
                };
            let old = project(before, &old_blocks, &old_gates);
            let fixed = project(after, &old_blocks, &old_gates);
            let fresh = project(after, &new_blocks, &new_gates);
            let loss = |total: &[f64]| {
                x.row(row)
                    .iter()
                    .zip(total)
                    .map(|(&v, &p)| (v as f64 - gamma as f64 * p).powi(2))
                    .sum::<f64>()
            };
            let mut old_support: Vec<_> = old_blocks
                .row(row)
                .iter()
                .zip(old_gates.row(row).iter())
                .filter_map(|(&block, &gate)| (gate != 0.0).then_some(block))
                .collect();
            let mut new_support: Vec<_> = new_blocks
                .row(row)
                .iter()
                .zip(new_gates.row(row).iter())
                .filter_map(|(&block, &gate)| (gate != 0.0).then_some(block))
                .collect();
            old_support.sort_unstable();
            new_support.sort_unstable();
            (
                loss(&old),
                loss(&fixed),
                loss(&fresh),
                usize::from(old_support != new_support),
            )
        })
        .collect();
    let frame_stats: Vec<_> = (0..config.n_blocks)
        .into_par_iter()
        .map(|block| {
            let old = before
                .slice(ndarray::s![block * b..(block + 1) * b, ..])
                .mapv(f64::from);
            let new = after
                .slice(ndarray::s![block * b..(block + 1) * b, ..])
                .mapv(f64::from);
            let old_gram = old.dot(&old.t());
            let new_gram = new.dot(&new.t());
            let cross = old.dot(&new.t());
            let squared = |a: &Array2<f64>| a.iter().map(|v| v * v).sum::<f64>();
            let scale = squared(&old_gram) + squared(&new_gram);
            let displacement = if scale == 0.0 {
                0.0
            } else {
                ((scale - 2.0 * squared(&cross)).max(0.0) / scale).sqrt()
            };
            (block, displacement, old_gram, new_gram)
        })
        .collect();
    let worst = frame_stats
        .iter()
        .max_by(|a, b| a.1.total_cmp(&b.1))
        .unwrap();
    let usage = (0..old_blocks.nrows())
        .map(|row| {
            (0..old_blocks.ncols())
                .filter(|&slot| {
                    old_gates[[row, slot]] != 0.0 && old_blocks[[row, slot]] as usize == worst.0
                })
                .count()
        })
        .sum::<usize>();
    Ok(serde_json::json!({
        "old_support_old_frames_rss": row_stats.iter().map(|v| v.0).sum::<f64>(),
        "old_support_proposed_frames_rss": row_stats.iter().map(|v| v.1).sum::<f64>(),
        "fresh_support_proposed_frames_rss": row_stats.iter().map(|v| v.2).sum::<f64>(),
        "rows_changing_support": row_stats.iter().map(|v| v.3).sum::<usize>(),
        "worst_projector_block": worst.0,
        "worst_projector_residual": worst.1,
        "worst_projector_block_usage": usage,
        "worst_projector_old_gram": worst.2,
        "worst_projector_new_gram": worst.3,
        "gamma": gamma
    }))
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let args: Vec<String> = std::env::args().collect();
    if args.len() != 7 {
        return Err(
            "usage: block_stream_convergence corpus.npy rows atoms block_size topk epochs".into(),
        );
    }
    let path = Path::new(&args[1]);
    let rows: usize = args[2].parse()?;
    let atoms: usize = args[3].parse()?;
    let block_size: usize = args[4].parse()?;
    let topk: usize = args[5].parse()?;
    let epochs: usize = args[6].parse()?;
    if rows == 0 || block_size == 0 || atoms == 0 || atoms % block_size != 0 {
        return Err(
            "rows and block size must be positive; atoms must partition into complete blocks"
                .into(),
        );
    }
    let file = std::fs::File::open(path)?;
    // SAFETY: the input is an immutable activation artifact for the duration
    // of this measurement. The mapping is read-only and byte accesses below
    // are checked against its length before constructing the bounded prefix.
    let mapped = unsafe { memmap2::MmapOptions::new().map(&file)? };
    let (total_rows, p, element_bytes, is_f32, offset) =
        npy_header::parse_npy_header(&mapped, path)?;
    if rows > total_rows {
        return Err("requested prefix exceeds corpus rows".into());
    }
    let elements = rows.checked_mul(p).ok_or("prefix shape overflow")?;
    let end = elements
        .checked_mul(element_bytes)
        .and_then(|bytes| offset.checked_add(bytes))
        .ok_or("prefix byte length overflow")?;
    let payload = mapped.get(offset..end).ok_or("truncated NPY payload")?;
    let values = payload
        .chunks_exact(element_bytes)
        .map(|bytes| {
            if is_f32 {
                f32::from_le_bytes(bytes.try_into().expect("four-byte chunk"))
            } else {
                f16::f16_to_f32(u16::from_le_bytes(
                    bytes.try_into().expect("two-byte chunk"),
                ))
            }
        })
        .collect();
    let x = Array2::from_shape_vec((rows, p), values)?;
    let mut config = BlockSparseConfig::new(atoms / block_size, block_size);
    config.block_topk = topk;
    config.max_epochs = epochs;
    config.minibatch = rows;
    config.aux_k = 0;
    let started = Instant::now();
    let mut state = BlockSparseStreamState::new(x.view(), &config)?;
    let mut output = std::io::stdout().lock();
    writeln!(
        output,
        "{}",
        serde_json::json!({
            "rows": rows, "features": p, "atoms": atoms, "block_size": block_size,
            "topk": topk, "tolerance": config.tolerance, "seed_seconds": started.elapsed().as_secs_f64()
        })
    )?;
    output.flush()?;
    for epoch_index in 0..epochs {
        let previous = (epoch_index + 1 == epochs).then(|| state.decoder().to_owned());
        let pass = Instant::now();
        state.partial_fit(x.view())?;
        let stream_seconds = pass.elapsed().as_secs_f64();
        let refresh = Instant::now();
        let stats = state.end_epoch()?;
        let refresh_seconds = refresh.elapsed().as_secs_f64();
        writeln!(
            output,
            "{}",
            serde_json::json!({
                "epoch": stats.epoch, "ev": stats.explained_variance, "gamma": stats.gamma,
                "gamma_residual": stats.gamma_residual, "frame_residual": stats.frame_residual,
                "converged": stats.converged, "seconds": pass.elapsed().as_secs_f64(),
                "stream_seconds": stream_seconds, "refresh_seconds": refresh_seconds
            })
        )?;
        output.flush()?;
        if let Some(previous) = previous {
            let started = Instant::now();
            let audit = audit_proposal(
                x.view(),
                previous.view(),
                state.decoder(),
                stats.gamma,
                &config,
            )?;
            writeln!(
                output,
                "{}",
                serde_json::json!({
                    "proposal_audit": audit, "audit_seconds": started.elapsed().as_secs_f64()
                })
            )?;
            output.flush()?;
        }
        if stats.converged {
            let fit = state.finalize()?;
            writeln!(
                output,
                "{}",
                serde_json::json!({"final_ev": fit.explained_variance, "gamma": fit.gamma})
            )?;
            return Ok(());
        }
    }
    Err("stream remains unconverged; no fit artifact was produced".into())
}
