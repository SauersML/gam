//! Training throughput of a masked library's step on this Mac's Apple GPU against its CPU (#2951).
//!
//! `mpd_metal_throughput_2951 EXPORT_DIR LIBRARY_DIR MODE BATCHES [SECONDS] [MICRO] [SITES]`
//!
//! `EXPORT_DIR` is a language-model export (`gam_mpd::import::import_language_model`, 512-token
//! sequences), `LIBRARY_DIR` a library per site (`{site}.v.f64`, `{site}.u.f64`, as
//! `mpd_site_fit_2951` writes them; a site without files, or whose name does not start with
//! `SITES` when given, stays native). `BATCHES` is a comma list
//! of sequences per step. Each batch is timed for `SECONDS` (default 20) after one untimed step;
//! the line per batch gives sequence-steps per second.
//!
//! `MODE` `metal` and `cpu` time the masked step a training step proposes from: the masked forward
//! with its KL and the KL's cotangent, every site's gradients (masks, `V`, `U`, brought to the
//! host), a two-draw sampled-label Fisher of the masks and written nodes, and the reads'
//! covariances, at masks one in four on: `metal` on the Apple GPU in f32 (`gam_gpu::tensor`, the
//! clean model's logits formed there as the target), `cpu` on the CPU path of `gam_mpd::masked` in
//! float64. `MODE` `train` and `host` time `gam_mpd::device_train`'s step (every site's sets, the
//! masked forward's KL and box charge with their gradient, `MICRO` sequences at a time (default
//! 4), then one Adam update, all resident): `train` on the Apple GPU in f32, `host` on the
//! float64 host reference backend.

use gam_gpu::GpuPolicy;
use gam_gpu::tensor::{Arithmetic, Device};
use gam_linalg::faer_ndarray::fast_atb;
use gam_mpd::blocks::Generic;
use gam_mpd::device_program::DeviceProgram;
use gam_mpd::device_train::{Settings, Trainer, statistics};
use gam_mpd::import::import_language_model;
use gam_mpd::masked::{self, Library, Masked, Target, matrix, read_values, sites};
use gam_mpd::masked_device::{Accelerated, DeviceTarget};
use ndarray::{Array1, Array2};
use std::path::Path;
use std::time::Instant;

const CONTEXT: usize = 512;

fn read_f64(path: &Path, cols: usize) -> Result<Array2<f64>, String> {
    let bytes = std::fs::read(path).map_err(|e| format!("{}: {e}", path.display()))?;
    let values: Vec<f64> = bytes.chunks_exact(8).map(|c| f64::from_le_bytes([c[0], c[1], c[2], c[3], c[4], c[5], c[6], c[7]])).collect();
    Array2::from_shape_vec((values.len() / cols, cols), values).map_err(|e| format!("{}: {e}", path.display()))
}

/// Deterministic noise in `[-1, 1)`.
fn noise(seed: usize) -> f64 {
    let mut x = (seed as u64).wrapping_mul(0x9E37_79B9_7F4A_7C15) ^ 0xD1B5_4A32_D192_ED03;
    x ^= x >> 33;
    x = x.wrapping_mul(0xFF51_AFD7_ED55_8CCD);
    x ^= x >> 33;
    (x >> 11) as f64 / (1u64 << 52) as f64 - 1.0
}

fn masks_for(masked: &Masked, rows: usize) -> Vec<Array2<f64>> {
    masked
        .all_blocks()
        .into_iter()
        .enumerate()
        .map(|(k, blocks)| Array2::from_shape_fn((rows, blocks), |(r, c)| if noise(7919 * k + 104_729 * r + c) < -0.5 { 1.0 } else { 0.0 }))
        .collect()
}

/// The first `count` sequences of `all`.
fn family_of_count(all: &gam_mpd::operator_program::FamilyInputs, count: usize) -> gam_mpd::operator_program::FamilyInputs {
    all.select(&(0..count * CONTEXT).collect::<Vec<_>>())
}

/// The phases of a masked step, in order.
const PHASES: [&str; 4] = ["forward", "gradients", "fisher", "covariances"];

/// The phases of a trainer's step.
const TRAINER: [&str; 2] = ["passes", "update"];

/// Mean seconds per run of `step` over `budget` seconds, after one untimed run, and per phase
/// (`step` adds each phase's seconds to its slot).
fn timed(budget: f64, mut step: impl FnMut(&mut [f64; 4]) -> Result<(), String>) -> Result<(f64, usize, [f64; 4]), String> {
    step(&mut [0.0; 4])?;
    let started = Instant::now();
    let (mut runs, mut phases) = (0usize, [0.0; 4]);
    loop {
        step(&mut phases)?;
        runs += 1;
        if started.elapsed().as_secs_f64() >= budget {
            return Ok((started.elapsed().as_secs_f64() / runs as f64, runs, phases.map(|p| p / runs as f64)));
        }
    }
}

/// `body`'s result, its seconds added to `slot`.
fn phase<T>(slot: &mut f64, body: impl FnOnce() -> Result<T, String>) -> Result<T, String> {
    let started = Instant::now();
    let out = body();
    *slot += started.elapsed().as_secs_f64();
    out
}

fn report(mode: &str, batch: usize, names: &[&str], (per_step, runs, phases): (f64, usize, [f64; 4])) {
    let parts: Vec<String> = names.iter().zip(phases).map(|(name, s)| format!("{name} {s:.3}")).collect();
    println!("{mode} batch {batch}: {per_step:.3} s per step ({runs} timed; {}), {:.2} sequence-steps/s", parts.join(", "), batch as f64 / per_step);
}

fn main() -> Result<(), String> {
    gam_mpd::engine::log_to_stderr();
    let args: Vec<String> = std::env::args().collect();
    let usage = "mpd_metal_throughput_2951 EXPORT_DIR LIBRARY_DIR MODE BATCHES [SECONDS]";
    let export = Path::new(args.get(1).ok_or(usage)?);
    let given = Path::new(args.get(2).ok_or(usage)?);
    let mode = args.get(3).ok_or(usage)?.as_str();
    let batches: Vec<usize> = args.get(4).ok_or(usage)?.split(',').map(|b| b.parse().map_err(|e| format!("BATCHES: {e}"))).collect::<Result<_, _>>()?;
    let seconds: f64 = args.get(5).map_or(Ok(20.0), |v| v.parse()).map_err(|e| format!("SECONDS: {e}"))?;
    let micro: usize = args.get(6).map_or(Ok(4), |v| v.parse()).map_err(|e| format!("MICRO: {e}"))?;
    let prefix = args.get(7).map_or("", String::as_str);
    let largest = batches.iter().copied().max().ok_or("BATCHES: none")?;

    let imported = import_language_model(export, largest, CONTEXT)?;
    let model = &imported.program;
    let all = &imported.contract.family;
    let (mut chosen, mut libraries) = (Vec::new(), Vec::new());
    for site in sites(model) {
        let v_path = given.join(format!("{}.v.f64", site.name));
        if !v_path.exists() || !site.name.starts_with(prefix) {
            continue;
        }
        let (d_out, d_in) = matrix(model, &site)?.dim();
        libraries.push(Library { v: read_f64(&v_path, d_in)?, u: read_f64(&given.join(format!("{}.u.f64", site.name)), d_out)?, mean: Array1::zeros(d_in) });
        chosen.push(site);
    }
    let pieces: usize = libraries.iter().map(|l| l.v.nrows()).sum();
    if mode == "train" || mode == "host" {
        let device = if mode == "host" {
            Device::host()
        } else {
            Device::single_precision(GpuPolicy::Required).map_err(|e| e.to_string())?.filter(|d| !d.float64()).ok_or("no Apple GPU")?
        };
        let arithmetic = if device.float64() { Arithmetic::F64 } else { Arithmetic::F32 };
        let observations = (all.rows) as f64;
        let measured = statistics(&device, model, &chosen, &[family_of_count(all, micro.min(largest))], 2, 0xDE5C)?;
        let describe = Generic::new(&measured, observations);
        drop(measured);
        let masked = Masked::build(model, chosen.clone(), libraries)?;
        let settings = Settings { observations, rate: 3e-4, betas: (0.9, 0.999), epsilon: 1e-12, draws: 2, seed: 0x5E7 };
        let mut trainer = Trainer::new(&device, model, &chosen, masked, &describe, settings, arithmetic)?;
        eprintln!("{} sites, {pieces} subcomponents on {} ({arithmetic:?})", chosen.len(), device.name());
        for &batch in &batches {
            let parts: Vec<_> = (0..batch).collect::<Vec<_>>().chunks(micro).map(|c| all.select(&c.iter().flat_map(|s| s * CONTEXT..(s + 1) * CONTEXT).collect::<Vec<_>>())).collect();
            let measured = timed(seconds, |t| {
                for part in &parts {
                    phase(&mut t[0], || trainer.train(part))?;
                }
                phase(&mut t[1], || trainer.update())?;
                device.synchronize().map_err(|e| e.to_string())
            })?;
            report(mode, batch, &TRAINER, measured);
        }
        return Ok(());
    }
    let masked = Masked::build(model, chosen, libraries)?;
    eprintln!("{} sites, {pieces} subcomponents, {} sequences", masked.sites.len(), all.rows / CONTEXT);
    let family_of = |count: usize| family_of_count(all, count);

    match mode {
        "metal" => {
            let device = Device::single_precision(GpuPolicy::Required).map_err(|e| e.to_string())?.ok_or("no Apple GPU")?;
            if device.float64() {
                return Err(format!("{} is not the Apple GPU", device.name()));
            }
            let mut clean = DeviceProgram::compile(&device, model)?;
            clean.set_arithmetic(Arithmetic::F32);
            let accelerated = Accelerated::new(&device, &masked, Arithmetic::F32)?;
            eprintln!("{}: {} bytes per row resident", device.name(), accelerated.program().bytes_per_row());
            for &batch in &batches {
                let base = family_of(batch);
                let masks = masks_for(&masked, base.rows);
                let family = masked.family(&base, &masks);
                let target = DeviceTarget::resident(clean.logits_on_device(&clean.forward(&base)?)?, None);
                let measured = timed(seconds, |t| {
                    // The forward's KL is read on the host, so each phase ends with its work done.
                    let state = phase(&mut t[0], || accelerated.forward(&family, &target))?;
                    phase(&mut t[1], || accelerated.gradients(&masked, &state, &masks))?;
                    phase(&mut t[2], || accelerated.fisher(&masked, &state, &target, 2, 0x5EED, true))?;
                    phase(&mut t[3], || accelerated.covariances(&masked, &state))?;
                    Ok(())
                })?;
                report("metal", batch, &PHASES, measured);
            }
        }
        "cpu" => {
            for &batch in &batches {
                let base = family_of(batch);
                let masks = masks_for(&masked, base.rows);
                let family = masked.family(&base, &masks);
                let target = Target::every_row(model.execute(&base, false).map_err(|e| e.to_string())?.values[model.output].clone());
                let measured = timed(seconds, |t| {
                    let (_, trace, cotangent) = phase(&mut t[0], || masked::forward(&masked, &family, &target))?;
                    phase(&mut t[1], || masked::gradients(&masked, &family, &trace, &masks, cotangent))?;
                    phase(&mut t[2], || masked::fisher(&masked, &family, &trace, &target, 2, 0x5EED, true))?;
                    phase(&mut t[3], || {
                        for site in &masked.sites {
                            let reads = read_values(&trace, site)?;
                            std::hint::black_box(fast_atb(&reads, &reads) / family.rows as f64);
                        }
                        Ok(())
                    })?;
                    Ok(())
                })?;
                report("cpu", batch, &PHASES, measured);
            }
        }
        other => return Err(format!("MODE {other}: expected metal or cpu")),
    }
    Ok(())
}
