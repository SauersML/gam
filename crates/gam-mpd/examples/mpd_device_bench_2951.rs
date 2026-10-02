//! Throughput of the masked-pieces trainer's hot path on this host's device (#2951): run first on
//! every rented instance, to accept or reject it.
//!
//! `mpd_device_bench_2951 EXPORT_DIR [SECONDS] [GPU] [OUT.json]`
//!
//! `EXPORT_DIR` is a language-model export (`gam_mpd::import::import_language_model`); the
//! trainer's shapes are its sites with exact libraries of `min(d_in, d_out)` pieces (the Fisher-SVD
//! start's size), masks one in four on, 512-token sequences. `GPU` (`off`, `auto` (default) or
//! `required`) picks the device (`gam_gpu::tensor::Device::accelerator`); with none, the host's
//! float64 reference backend is measured instead. Within `SECONDS` (default 60) it measures, at
//! batches of 1, 2, 4, … sequences up to what the device memory holds:
//!
//! * the float64 masked forward with its KL (what every keep/refuse decision reads);
//! * the mask-gradient reverse pass in float64, f32 and TF32 (the proposals);
//! * one selection round (forward and KL, mask gradients, two sampled-label Fishers, the
//!   proposal's forward) and one pieces step (gradients, written Fishers, read covariances, the
//!   curvature tangent, two backtracking forwards), as sequences per second end to end;
//!
//! and, once, the CPU path's masked forward on one sequence. Before timing, the device's KL on
//! the first sequence is checked against the CPU's within the band of the CPU's banded execution
//! (twice the logits' band, plus each evaluation's rounding); a host failing it exits non-zero.
//! Results go to stdout and, as JSON, to `OUT.json` when given.

use gam_gpu::GpuPolicy;
use gam_gpu::tensor::{Arithmetic, Device};
use gam_mpd::device_program::DeviceProgram;
use gam_mpd::import::import_language_model;
use gam_mpd::masked::{self, Library, Masked, Target, matrix, sites};
use gam_mpd::masked_device::Accelerated;
use gam_mpd::operator_program::FamilyInputs;
use ndarray::{Array1, Array2};
use serde_json::json;
use std::collections::BTreeMap;
use std::io::Write;
use std::time::Instant;

const CONTEXT: usize = 512;
const U: f64 = f64::EPSILON / 2.0;

/// Deterministic noise in `[-1, 1)`.
fn noise(seed: usize) -> f64 {
    let mut x = (seed as u64).wrapping_mul(0x9E37_79B9_7F4A_7C15) ^ 0xD1B5_4A32_D192_ED03;
    x ^= x >> 33;
    x = x.wrapping_mul(0xFF51_AFD7_ED55_8CCD);
    x ^= x >> 33;
    (x >> 11) as f64 / (1u64 << 52) as f64 - 1.0
}

/// An exact library of `min(d_in, d_out)` pieces: `W = Σ u_c v_cᵀ` with the identity on the
/// narrower side.
fn exact_library(w: &Array2<f64>) -> Library {
    let (d_out, d_in) = w.dim();
    if d_out <= d_in {
        Library { v: w.clone(), u: Array2::eye(d_out), mean: Array1::zeros(d_in) }
    } else {
        Library { v: Array2::eye(d_in), u: w.t().to_owned(), mean: Array1::zeros(d_in) }
    }
}

fn masks_for(masked: &Masked, rows: usize, salt: usize) -> Vec<Array2<f64>> {
    masked
        .libraries
        .iter()
        .enumerate()
        .map(|(k, library)| Array2::from_shape_fn((rows, library.v.nrows()), |(r, c)| if noise(salt + 7919 * k + 104_729 * r + c) < -0.5 { 1.0 } else { 0.0 }))
        .collect()
}

/// Runs `body` until `budget` seconds pass (at least once); returns the mean seconds per run.
fn timed(budget: f64, mut body: impl FnMut() -> Result<(), String>) -> Result<f64, String> {
    let started = Instant::now();
    let mut runs = 0usize;
    loop {
        body()?;
        runs += 1;
        if started.elapsed().as_secs_f64() >= budget {
            return Ok(started.elapsed().as_secs_f64() / runs as f64);
        }
    }
}

fn family_of(all: &FamilyInputs, first: usize, count: usize) -> FamilyInputs {
    all.select(&(first * CONTEXT..(first + count) * CONTEXT).collect::<Vec<_>>())
}

fn main() -> Result<(), String> {
    gam_mpd::engine::log_to_stderr();
    let args: Vec<String> = std::env::args().collect();
    let usage = "mpd_device_bench_2951 EXPORT_DIR [SECONDS] [GPU] [OUT.json]";
    let export = std::path::PathBuf::from(args.get(1).ok_or(usage)?);
    let seconds: f64 = args.get(2).map_or(Ok(60.0), |v| v.parse()).map_err(|e| format!("SECONDS: {e}"))?;
    let policy = args.get(3).map_or("auto", String::as_str);
    let policy = GpuPolicy::parse(policy).ok_or_else(|| format!("GPU {policy}: expected off, auto or required"))?;
    gam_gpu::configure_global_policy(policy);
    let out = args.get(4).map(std::path::PathBuf::from);
    let started = Instant::now();

    let record: serde_json::Value =
        serde_json::from_str(&std::fs::read_to_string(export.join("export.json")).map_err(|e| e.to_string())?).map_err(|e| e.to_string())?;
    let available = record["files"]["tokens"]["shape"][0].as_u64().ok_or("export: tokens shape")? as usize;
    let device = Device::accelerator(policy).map_err(|e| e.to_string())?.unwrap_or_else(Device::host);
    let memory = device.memory().map_err(|e| e.to_string())?;
    eprintln!("device: {} ({:?} bytes free/total)", device.name(), memory);

    // Up to as many sequences as the largest batch can use.
    let imported = import_language_model(&export, available.min(64), CONTEXT)?;
    let model = &imported.program;
    let all = &imported.contract.family;
    let sequences = all.rows / CONTEXT;
    let chosen = sites(model);
    let libraries: Vec<Library> = chosen.iter().map(|s| matrix(model, s).map(|w| exact_library(&w))).collect::<Result<_, _>>()?;
    let pieces: usize = libraries.iter().map(|l| l.v.nrows()).sum();
    let masked = Masked::build(model, chosen, libraries)?;
    eprintln!("{} sites, {pieces} pieces, {sequences} sequences available", masked.sites.len());

    let clean = DeviceProgram::compile(&device, model)?;
    let accelerated = Accelerated::new(&device, &masked, Arithmetic::F64)?;
    let classes = accelerated.program().classes();
    // The bytes one row holds at once: the trace, its cotangents, the target's logits.
    let per_row = 2 * accelerated.program().bytes_per_row() + 8 * classes;
    let largest = match memory {
        Some((free, _)) => (free * 6 / 10 / (per_row * CONTEXT)).max(1),
        None => 2,
    }
    .min(sequences);
    eprintln!("{} bytes per row; batches up to {largest} sequences", per_row);

    // The device's targets: the clean model's logits, formed on the device.
    let target_of = |family: &FamilyInputs| -> Result<Target, String> {
        let trace = clean.forward(family)?;
        Ok(Target::every_row(clean.logits(&trace, 0, family.rows)?))
    };

    // The check: the device's KL on the first sequence against the CPU's, within the band.
    let check_started = Instant::now();
    let one = family_of(all, 0, 1);
    let target = target_of(&one)?;
    let masks = masks_for(&masked, CONTEXT, 1);
    let family = masked.family(&one, &masks);
    let banded = masked.program.execute(&family, true).map_err(|e| e.to_string())?;
    let logits = &banded.values[masked.program.output];
    let band = &banded.bands.as_ref().ok_or("no bands")?[masked.program.output];
    let ball = &banded.balls.as_ref().ok_or("no balls")?[masked.program.output];
    let (cpu_kl, _) = masked::kl(&target, logits);
    let state = accelerated.forward(&family, &accelerated.target(&target)?)?;
    let mut worst: f64 = 0.0;
    for r in 0..CONTEXT {
        let softmax = |z: ndarray::ArrayView1<f64>| {
            let m = z.iter().copied().fold(f64::NEG_INFINITY, f64::max);
            let e = z.mapv(|v| (v - m).exp());
            let t = e.sum();
            e / t
        };
        let (p, q) = (softmax(target.logits.row(r)), softmax(logits.row(r)));
        let magnitude: f64 = p.iter().zip(q.iter()).map(|(a, b)| a * (a.ln().abs() + b.max(f64::MIN_POSITIVE).ln().abs())).sum();
        let n = (classes + 8) as f64;
        let allowed = 4.0 * (band.row(r).iter().fold(0.0_f64, |m, v| m.max(*v)) + ball[r]) + 2.0 * n * U / (1.0 - n * U) * magnitude;
        let gap = (state.kl[r] - cpu_kl[r]).abs();
        worst = worst.max(gap / allowed);
        if gap > allowed {
            return Err(format!("check failed: row {r}'s device KL {} against the CPU's {} (band {allowed:e})", state.kl[r], cpu_kl[r]));
        }
    }
    let check_seconds = check_started.elapsed().as_secs_f64();
    eprintln!("check: device KL within band on {CONTEXT} rows (worst {worst:.3} of its band), {check_seconds:.1}s");

    // The CPU path's masked forward on one sequence.
    let cpu_started = Instant::now();
    masked::forward(&masked, &family, &target)?;
    let cpu_forward = cpu_started.elapsed().as_secs_f64();
    eprintln!("cpu masked forward: {cpu_forward:.2}s per sequence");

    let mut batches = Vec::new();
    let mut b = 1;
    while b <= largest {
        batches.push(b);
        b *= 2;
    }
    let remaining = (seconds - started.elapsed().as_secs_f64()).max(5.0);
    let share = remaining / (batches.len() as f64 * 6.0);
    let mut rows_out = Vec::new();
    for &batch in &batches {
        let base = family_of(all, 0, batch);
        let rows = base.rows;
        let target = target_of(&base)?;
        let on_device = accelerated.target(&target)?;
        let masks = masks_for(&masked, rows, 3);
        let family = masked.family(&base, &masks);
        let forward = timed(share, || accelerated.forward(&family, &on_device).map(|_| ()))?;
        let state = accelerated.forward(&family, &on_device)?;
        let mut reverse = BTreeMap::new();
        for (name, arithmetic) in [("f64", Arithmetic::F64), ("f32", Arithmetic::F32), ("tf32", Arithmetic::Tf32)] {
            let proposing = Accelerated::new(&device, &masked, arithmetic)?;
            reverse.insert(name, timed(share / 3.0, || proposing.mask_gradients(&masked, &state).map(|_| ()))?);
        }
        let proposing = Accelerated::new(&device, &masked, Arithmetic::Tf32)?;
        let round = timed(share, || {
            let state = proposing.forward(&family, &on_device)?;
            proposing.mask_gradients(&masked, &state)?;
            proposing.fisher(&masked, &state, &on_device, 2, 0x5EED, false)?;
            proposing.forward(&family, &on_device).map(|_| ())
        })?;
        // A direction on every library operator (`V`, centring, `U` of every site).
        let tangents: BTreeMap<usize, Array2<f64>> = masked
            .program
            .operators
            .iter()
            .enumerate()
            .filter(|(_, op)| op.name.contains('·'))
            .map(|(i, op)| (i, op.matrix() * 1e-3))
            .collect();
        let step = timed(share, || {
            let state = proposing.forward(&family, &on_device)?;
            proposing.gradients(&masked, &state, &masks)?;
            proposing.fisher(&masked, &state, &on_device, 2, 0xF00D, true)?;
            proposing.covariances(&masked, &state)?;
            proposing.quadratic(&state, &on_device, &tangents)?;
            proposing.forward(&family, &on_device)?;
            proposing.forward(&family, &on_device).map(|_| ())
        })?;
        let per_second = |s: f64| batch as f64 / s;
        let row = json!({
            "batch": batch,
            "forward_kl_f64_seconds": forward, "forward_kl_f64_sequences_per_second": per_second(forward),
            "mask_gradients_seconds": reverse,
            "mask_gradients_sequences_per_second": reverse.iter().map(|(k, v)| (k.to_string(), json!(per_second(*v)))).collect::<serde_json::Map<_, _>>(),
            "selection_round_sequences_per_second": per_second(round),
            "pieces_step_sequences_per_second": per_second(step),
        });
        eprintln!("{row}");
        rows_out.push(row);
        if started.elapsed().as_secs_f64() > seconds {
            break;
        }
    }
    let best = rows_out
        .iter()
        .map(|r| r["selection_round_sequences_per_second"].as_f64().unwrap_or(0.0))
        .fold(0.0_f64, f64::max);
    let report = json!({
        "device": device.name(),
        "memory": memory,
        "sites": masked.sites.len(),
        "pieces": pieces,
        "classes": classes,
        "check": {"rows": CONTEXT, "worst_fraction_of_band": worst, "seconds": check_seconds},
        "cpu_masked_forward_seconds_per_sequence": cpu_forward,
        "batches": rows_out,
        "best_selection_round_sequences_per_second": best,
        "seconds": started.elapsed().as_secs_f64(),
    });
    writeln!(std::io::stdout(), "{}", serde_json::to_string_pretty(&report).map_err(|e| e.to_string())?).map_err(|e| e.to_string())?;
    if let Some(path) = out {
        std::fs::write(&path, report.to_string()).map_err(|e| format!("{}: {e}", path.display()))?;
    }
    Ok(())
}
