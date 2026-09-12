#[cfg(target_os = "linux")]
use gam_sae::sparse_dict::{
    BlockRoutePath, DEVICE_BLOCK_GATE_MIN_ELEMS, code_block_shortlists_cpu,
    route_and_code_blocks_required, route_blocks_cpu, route_blocks_required,
};
#[cfg(target_os = "linux")]
use ndarray::{Array2, Array3};

/// Fixed-width block codes: admitted blocks, their gates and γ-free projections.
#[cfg(target_os = "linux")]
type BlockCodes = (Array2<u32>, Array2<f32>, Array3<f64>);

/// The first slot where two fixed-width codings differ in any bit, or `None`
/// when blocks, gates and projections are all bit-identical.
#[cfg(target_os = "linux")]
fn first_code_mismatch(device: &BlockCodes, host: &BlockCodes) -> Option<String> {
    let (device_blocks, device_gates, device_projections) = device;
    let (host_blocks, host_gates, host_projections) = host;
    if device_projections.dim() != host_projections.dim() {
        return Some(format!(
            "shape differs device={:?} host={:?}",
            device_projections.dim(),
            host_projections.dim()
        ));
    }
    let (rows, width, b) = host_projections.dim();
    for row in 0..rows {
        for slot in 0..width {
            if device_blocks[[row, slot]] != host_blocks[[row, slot]] {
                return Some(format!(
                    "row {row} slot {slot}: admitted block differs device={} host={}",
                    device_blocks[[row, slot]],
                    host_blocks[[row, slot]]
                ));
            }
            if device_gates[[row, slot]].to_bits() != host_gates[[row, slot]].to_bits() {
                return Some(format!(
                    "row {row} slot {slot}: gate differs device={:e} host={:e}",
                    device_gates[[row, slot]],
                    host_gates[[row, slot]]
                ));
            }
            for axis in 0..b {
                let (device_value, host_value) = (
                    device_projections[[row, slot, axis]],
                    host_projections[[row, slot, axis]],
                );
                if device_value.to_bits() != host_value.to_bits() {
                    return Some(format!(
                        "row {row} slot {slot} axis {axis}: projection differs \
                         device={device_value:e} host={host_value:e}"
                    ));
                }
            }
        }
    }
    None
}
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

    // #2826: a device route also codes each row on the device. On the device's
    // own shortlists those codes must equal the host coder's to the bit: the
    // admitted blocks and their order, the gates and the γ-free projections.
    // Row 0 is zeroed so a row with no candidate is exercised, and γ = 2.5 makes
    // every admission after the unconditional first one raise the tied loss.
    let mut coded_rows = rows.clone();
    coded_rows.row_mut(0).fill(0.0);
    let (shortlists, shortlist_path, _) = route_blocks_required(
        coded_rows.view(),
        decoder.view(),
        b,
        k,
        gam_gpu::GpuPolicy::Required,
    )
    .map_err(|error| format!("GpuPolicy::Required route of the coding fixture failed: {error}"))?;
    if shortlist_path != BlockRoutePath::Device {
        return Err(format!(
            "coding fixture route returned {shortlist_path:?}, expected Device"
        ));
    }
    for gamma in [0.75f32, 2.5] {
        let device_start = Instant::now();
        let (device_codes, device_path) = route_and_code_blocks_required(
            coded_rows.view(),
            decoder.view(),
            gamma,
            b,
            k,
            gam_gpu::GpuPolicy::Required,
        )
        .map_err(|error| format!("GpuPolicy::Required route and code failed (gamma={gamma}): {error}"))?;
        let device_secs = device_start.elapsed().as_secs_f64();
        if device_path != BlockRoutePath::Device {
            return Err(format!(
                "GpuPolicy::Required route and code returned {device_path:?}, expected Device"
            ));
        }
        let host_start = Instant::now();
        let host_codes = code_block_shortlists_cpu(
            coded_rows.view(),
            decoder.view(),
            gamma,
            b,
            k,
            shortlists.clone(),
        );
        let host_secs = host_start.elapsed().as_secs_f64();
        if let Some(mismatch) = first_code_mismatch(&device_codes, &host_codes) {
            return Err(format!("device coding (gamma={gamma}) differs from host coding: {mismatch}"));
        }
        // Positive control: the comparison must see a one-bit change in a single
        // admitted projection.
        let mut perturbed = host_codes.clone();
        let admitted_slot = (0..m)
            .flat_map(|row| (0..k).map(move |slot| (row, slot)))
            .find(|&(row, slot)| perturbed.1[[row, slot]] != 0.0)
            .ok_or_else(|| format!("gamma={gamma}: host coding admitted no block"))?;
        let value = &mut perturbed.2[[admitted_slot.0, admitted_slot.1, 0]];
        *value = f64::from_bits(value.to_bits() ^ 1);
        if first_code_mismatch(&device_codes, &perturbed).is_none() {
            return Err(format!(
                "positive control failed: a one-bit projection change at {admitted_slot:?} was not detected"
            ));
        }
        let admitted = host_codes.1.iter().filter(|&&gate| gate != 0.0).count();
        let empty_rows = (0..m)
            .filter(|&row| host_codes.1.row(row).iter().all(|&gate| gate == 0.0))
            .count();
        println!(
            "[block-gate speedup] device coding gamma={gamma}: {m} rows bit-identical to host \
             coding on the device shortlists ({admitted} admitted blocks, {empty_rows} rows with \
             none; one-bit positive control detected); device route+code {device_secs:.4}s, \
             host code {host_secs:.4}s"
        );
    }
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
