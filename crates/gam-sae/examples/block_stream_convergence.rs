//! Measure streamed gamma/frame convergence on a bounded prefix of an NPY corpus.
//! Usage: block_stream_convergence corpus.npy rows atoms block_size topk epochs

use gam_sae::sparse_dict::{BlockSparseConfig, BlockSparseStreamState};
use ndarray::Array2;
use std::io::Write;
use std::path::Path;
use std::time::Instant;

#[path = "support/f16.rs"]
mod f16;
#[path = "support/npy_header.rs"]
mod npy_header;

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
    for _ in 0..epochs {
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
