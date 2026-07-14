#[cfg(target_os = "linux")]
use gam_sae::sparse_dict::{
    BlockRoutePath, DEVICE_BLOCK_GATE_MIN_ELEMS, route_blocks_cpu, route_blocks_required,
};
#[cfg(target_os = "linux")]
use ndarray::Array2;
use std::process::ExitCode;
#[cfg(target_os = "linux")]
use std::time::Instant;

#[cfg(target_os = "linux")]
fn fixture(n_rows: usize, n_blocks: usize, b: usize, p: usize) -> (Array2<f32>, Array2<f32>) {
    let rows = Array2::from_shape_fn((n_rows, p), |(i, c)| {
        (((i * 29 + c * 13) as f32) * 0.017).sin() * 0.8
    });
    let mut decoder = Array2::from_shape_fn((n_blocks * b, p), |(a, c)| {
        (((a * 11 + c * 3) as f32) * 0.009).cos()
    });
    for g in 0..n_blocks {
        let mut block = decoder.slice(ndarray::s![g * b..g * b + b, ..]).to_owned();
        gram_schmidt_rows(&mut block);
        for r in 0..b {
            for c in 0..p {
                decoder[[g * b + r, c]] = block[[r, c]];
            }
        }
    }
    (rows, decoder)
}

#[cfg(target_os = "linux")]
fn gram_schmidt_rows(block: &mut Array2<f32>) {
    let rows = block.nrows();
    let cols = block.ncols();
    for i in 0..rows {
        for j in 0..i {
            let mut dot = 0.0f32;
            for c in 0..cols {
                dot += block[[i, c]] * block[[j, c]];
            }
            for c in 0..cols {
                block[[i, c]] -= dot * block[[j, c]];
            }
        }
        let mut norm2 = 0.0f32;
        for c in 0..cols {
            norm2 += block[[i, c]] * block[[i, c]];
        }
        let norm = norm2.sqrt().max(1.0e-12);
        for c in 0..cols {
            block[[i, c]] /= norm;
        }
    }
}

#[cfg(target_os = "linux")]
fn run() -> Result<(), String> {
    let m = 1024usize;
    let g = 16_384usize;
    let b = 3usize;
    let k = 8usize;
    let p = 128usize;
    let krows = g * b;
    if m * krows < DEVICE_BLOCK_GATE_MIN_ELEMS {
        return Err(format!(
            "fixture {m}x{krows} is below DEVICE_BLOCK_GATE_MIN_ELEMS={DEVICE_BLOCK_GATE_MIN_ELEMS}"
        ));
    }
    let (rows, decoder) = fixture(m, g, b, p);

    match gam_gpu::GpuRuntime::resolve(gam_gpu::GpuPolicy::Auto) {
        Ok(Some(_)) => {}
        Ok(None) => {
            println!("[block-gate speedup] no CUDA device; hardware benchmark skipped");
            return Ok(());
        }
        Err(error) => return Err(format!("CUDA admission failed: {error}")),
    }
    gam_gpu::GpuRuntime::require()
        .map_err(|error| format!("CUDA was admitted but Required resolution failed: {error}"))?;

    match route_blocks_required(rows.view(), decoder.view(), b, k, gam_gpu::GpuPolicy::Auto) {
        Ok((warm_route, warm_path, warm_bytes)) => {
            println!(
                "[block-gate speedup] warm-up path={warm_path:?} rows={} dtoh={warm_bytes}B",
                warm_route.len()
            );
        }
        Err(err) => {
            return Err(format!("warm-up route under Auto failed: {err}"));
        }
    }

    let cpu_start = Instant::now();
    let cpu = route_blocks_cpu(rows.view(), decoder.view(), g, b, k);
    let cpu_secs = cpu_start.elapsed().as_secs_f64();

    // The PRODUCTION CPU lane: route_blocks_required under Auto (the blocked-
    // GEMM fallback when no device runs). Timed against the scalar oracle
    // above, with per-row support parity accounted, so the CPU-side speedup
    // of the fallback swap is a measured number, not an inference.
    let fallback_start = Instant::now();
    let (fallback, fallback_path, _fallback_bytes) =
        route_blocks_required(rows.view(), decoder.view(), b, k, gam_gpu::GpuPolicy::Auto)
            .map_err(|err| format!("Auto route failed: {err}"))?;
    let fallback_secs = fallback_start.elapsed().as_secs_f64();
    let support_mismatches = cpu
        .iter()
        .zip(fallback.iter())
        .filter(|(a, c)| {
            let sa: std::collections::BTreeSet<u32> = a.iter().map(|e| e.0).collect();
            let sc: std::collections::BTreeSet<u32> = c.iter().map(|e| e.0).collect();
            sa != sc
        })
        .count();
    println!(
        "[block-gate speedup] production CPU lane path={fallback_path:?}: {fallback_secs:.4}s \
         vs scalar oracle {cpu_secs:.4}s = {:.1}x, support mismatches {support_mismatches}/{m}",
        cpu_secs / fallback_secs.max(1e-12)
    );

    let device_start = Instant::now();
    let (routed, path, dtoh) = route_blocks_required(
        rows.view(),
        decoder.view(),
        b,
        k,
        gam_gpu::GpuPolicy::Required,
    )
    .map_err(|error| format!("GpuPolicy::Required route failed: {error}"))?;
    let device_secs = device_start.elapsed().as_secs_f64();
    if path != BlockRoutePath::Device {
        return Err(format!(
            "GpuPolicy::Required returned {path:?}, expected Device"
        ));
    }
    for (row, (dev_sel, cpu_sel)) in routed.iter().zip(&cpu).enumerate() {
        if dev_sel.len() != cpu_sel.len() {
            return Err(format!("row {row}: selection length differs"));
        }
        for (slot, ((dev_block, dev_gate), (cpu_block, cpu_gate))) in
            dev_sel.iter().zip(cpu_sel).enumerate()
        {
            if dev_block != cpu_block {
                return Err(format!(
                    "row {row} slot {slot}: block differs device={dev_block} cpu={cpu_block}"
                ));
            }
            let tol = 1.0e-5 * cpu_gate.abs().max(1.0);
            if (dev_gate - cpu_gate).abs() > tol {
                return Err(format!(
                    "row {row} slot {slot}: gate differs device={dev_gate} cpu={cpu_gate} tol={tol}"
                ));
            }
        }
    }
    println!(
        "[block-gate speedup] m={m} G={g} b={b} k={k} P={p} K={krows}: CPU {cpu_secs:.4}s device {device_secs:.4}s speedup {:.1}x dtoh={dtoh}B",
        cpu_secs / device_secs.max(1.0e-9)
    );
    Ok(())
}

#[cfg(target_os = "linux")]
fn main() -> ExitCode {
    if let Err(err) = run() {
        eprintln!("[block-gate speedup] error: {err}");
        return ExitCode::FAILURE;
    }
    ExitCode::SUCCESS
}

#[cfg(not(target_os = "linux"))]
fn main() -> ExitCode {
    println!("[block-gate speedup] CUDA block-gate measurement is Linux-only");
    ExitCode::SUCCESS
}
