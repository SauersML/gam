use gam_sae::sparse_dict::{ScoreRouteStats, TileScorer};
use ndarray::Array2;
use std::io::Write;
use std::time::Instant;

fn parse_arg(args: &[String], idx: usize, default: usize) -> usize {
    args.get(idx)
        .and_then(|s| s.parse::<usize>().ok())
        .unwrap_or(default)
}

fn parse_mode(args: &[String]) -> Result<gam_gpu::GpuMode, String> {
    match args.get(6).map(String::as_str).unwrap_or("required") {
        "auto" => Ok(gam_gpu::GpuMode::Auto),
        "required" => Ok(gam_gpu::GpuMode::Required),
        "off" => Ok(gam_gpu::GpuMode::Off),
        other => Err(format!("mode must be off|auto|required, got {other}")),
    }
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let args: Vec<String> = std::env::args().collect();
    let n_rows = parse_arg(&args, 1, 256);
    let n_atoms = parse_arg(&args, 2, 4096);
    let p = parse_arg(&args, 3, 48);
    let active = parse_arg(&args, 4, 4);
    let tile = parse_arg(&args, 5, 1024);
    let mode = parse_mode(&args)?;

    let rows = Array2::<f32>::from_shape_fn((n_rows, p), |(r, c)| {
        (((r * 31 + c * 17 + 3) as f32) * 0.013).sin() * 0.9
    });
    let mut decoder = Array2::<f32>::from_shape_fn((n_atoms, p), |(a, c)| {
        (((a * 7 + c * 5 + 1) as f32) * 0.011).cos()
    });
    for mut row in decoder.outer_iter_mut() {
        let norm = row.iter().map(|v| v * v).sum::<f32>().sqrt().max(1e-12);
        row.mapv_inplace(|v| v / norm);
    }

    let scorer = TileScorer::new(active, tile);
    let start = Instant::now();
    let routed = scorer
        .route_minibatch_with_mode(rows.view(), decoder.view(), mode)
        .expect("route benchmark");
    let elapsed = start.elapsed();

    let mut stats = ScoreRouteStats::default();
    stats.record_result(&routed);
    let checksum: u64 = routed
        .selections
        .iter()
        .flat_map(|row| {
            row.iter()
                .map(|(atom, score)| (*atom as u64) ^ score.to_bits() as u64)
        })
        .fold(0u64, |acc, v| acc.wrapping_mul(16_777_619) ^ v);
    writeln!(
        std::io::stdout(),
        "rows={n_rows} atoms={n_atoms} p={p} active={active} tile={tile} mode={mode} \
         path={:?} elapsed_ms={:.3} admitted={} tiles={} peak_score_mb={:.2} \
         score_elems={} dot_flops_lower_bound={} device_dtoh_kb={:.2} \
         unfused_score_dtoh_avoided_mb={:.2} checksum={checksum}",
        routed.path,
        elapsed.as_secs_f64() * 1000.0,
        stats.admitted_minibatches,
        stats.score_tiles,
        stats.peak_score_bytes as f64 / (1024.0 * 1024.0),
        stats.score_elements,
        stats.dot_flops_lower_bound,
        stats.device_dtoh_bytes as f64 / 1024.0,
        stats.unfused_score_dtoh_bytes_avoided as f64 / (1024.0 * 1024.0),
    )?;
    Ok(())
}
