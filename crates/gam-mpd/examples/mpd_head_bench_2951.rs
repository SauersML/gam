//! Where a language model's forward spends its time on one sequence, and what the logits head
//! costs in each arithmetic (#2951).
//!
//! `mpd_head_bench_2951 EXPORT_DIR [CONTEXT] [REPEATS]`
//!
//! On the first export sequence (`CONTEXT` tokens, default 512) it times, median of `REPEATS`
//! (default 3): the whole float64 forward (`OperatorProgram::execute`); the head alone, the hidden
//! rows times the unembedding (`h A`, `gam_linalg::faer_ndarray::fast_ab`) in float64; the same
//! product in f32 on the CPU (faer) and on the Apple GPU (`gam_gpu::banded`, when present); and the
//! per-row KL with its cotangent against the model's own logits (`gam_mpd::masked::kl`).

use gam_gpu::banded::{BandedArithmetic, banded_matmul};
use gam_mpd::import::import_language_model;
use gam_mpd::masked::{Target, kl};
use gam_mpd::operator_program::Node;
use std::time::Instant;

fn median(mut v: Vec<f64>) -> f64 {
    v.sort_by(f64::total_cmp);
    v[v.len() / 2]
}

fn timed<T>(repeats: usize, mut body: impl FnMut() -> Result<T, String>) -> Result<(f64, T), String> {
    let mut seconds = Vec::new();
    let mut last = None;
    for _ in 0..repeats {
        let started = Instant::now();
        last = Some(body()?);
        seconds.push(started.elapsed().as_secs_f64());
    }
    Ok((median(seconds), last.ok_or("no repeats")?))
}

fn main() -> Result<(), String> {
    let args: Vec<String> = std::env::args().collect();
    let usage = "mpd_head_bench_2951 EXPORT_DIR [CONTEXT] [REPEATS]";
    let export = std::path::PathBuf::from(args.get(1).ok_or(usage)?);
    let context: usize = args.get(2).map_or(Ok(512), |v| v.parse()).map_err(|e| format!("CONTEXT: {e}"))?;
    let repeats: usize = args.get(3).map_or(Ok(3), |v| v.parse()).map_err(|e| format!("REPEATS: {e}"))?;
    gam_gpu::configure_global_policy(gam_gpu::GpuPolicy::Auto);
    let imported = import_language_model(&export, 1, context)?;
    let program = &imported.program;
    let family = &imported.contract.family;
    let (forward, trace) = timed(repeats, || program.execute(family, false).map_err(|e| e.to_string()))?;
    let logits_node = match &program.nodes[program.output] {
        Node::Readout { input, .. } => *input,
        _ => program.output,
    };
    let Node::Transposed { input, operator } = &program.nodes[logits_node] else {
        return Err("the head is not a transposed unembedding".to_string());
    };
    let hidden = &trace.values[*input];
    let a = program.operators[*operator].matrix();
    let (head, logits) = timed(repeats, || Ok(gam_linalg::faer_ndarray::fast_ab(hidden, &a)))?;
    let (h32, a32) = (hidden.mapv(|v| v as f32), a.mapv(|v| v as f32));
    let (head_f32, _) = timed(repeats, || Ok(h32.dot(&a32)))?;
    let (head_metal, _) = timed(repeats, || banded_matmul(gam_gpu::GpuPolicy::Auto, BandedArithmetic::F32, hidden.view(), a.view()).map_err(|e| e.to_string()))?;
    let target = Target::every_row(logits.clone());
    let (kl_seconds, _) = timed(repeats, || Ok(kl(&target, &logits)))?;
    let report = serde_json::json!({
        "rows": hidden.nrows(), "hidden": hidden.ncols(), "classes": a.ncols(),
        "forward_f64_seconds": forward,
        "head_f64_seconds": head, "head_share_of_forward": head / forward,
        "head_f32_cpu_seconds": head_f32, "head_f32_metal_seconds": head_metal,
        "kl_seconds": kl_seconds,
        "threads": rayon::current_num_threads(),
    });
    eprintln!("{report}");
    Ok(())
}
