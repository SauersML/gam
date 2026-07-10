//! End-to-end tiered fit at high `K` with the CUDA block-gate routing lane
//! engaged (#2023 / #1017), the tiered analogue of `examples/scale_k.rs`.
//!
//! `scale_k` drives the STREAMING block-sparse lane directly; this harness drives
//! the composed [`fit_tiered`] spine (Tier-0 mean → Tier-1 block-sparse bulk →
//! optional Tier-2 curved co-fit) so the Tier-1 router is exercised through the
//! same GPU dispatch a real tiered fit uses. The dispatch honours the
//! process-wide [`gam_gpu::GpuPolicy`] this harness sets:
//!
//! * `--gpu required` — the Tier-1 router MUST run each admitted minibatch on the
//!   device; a missing runtime or a device fault is a typed error, never a silent
//!   CPU fallback. This is the mode the MSI/A100 validation uses to prove the
//!   device path actually ran at `K≈1e4`.
//! * `--gpu auto` (default) — device when admitted and above break-even, else the
//!   CPU oracle (the fallback is logged once by the block-gate router).
//! * `--gpu off` — CPU router only (the portable baseline; runs anywhere).
//!
//! The device route carries the #2227 bounded-progress checkpoints, so a device
//! stall surfaces as a tile-attributed error rather than a silent multi-minute
//! hang. Under `RUST_LOG=info` the block-gate router logs its one-shot
//! engagement verdict; under `RUST_LOG=debug` the device route emits per-tile
//! progress heartbeats.
//!
//! The activations are deterministic planted structure (a handful of shared
//! linear directions plus per-row noise), so the harness is self-contained and
//! runnable without a data file; the point is to exercise the GPU routing lane at
//! scale, not to recover a specific dictionary.
//!
//! # Seeding wall (why N defaults modest)
//!
//! [`fit_tiered`]'s one-shot Tier-1 ([`gam_sae::sparse_dict::fit_block_sparse_dictionary`])
//! seeds its `K` frames with a SERIAL farthest-point pass over the corpus
//! (`sparse_dict::update::seed_decoder`), an `O(K·N·P)` cost that at `K≈1e4`
//! dominates the fit and is unrelated to routing. The streaming lane
//! (`examples/scale_k.rs`) sidesteps it with `new_with_decoder` + a cheap
//! coordinate seed; the one-shot lane has no such hook yet. So this harness
//! defaults to a modest `N` (device admission depends on `minibatch·K`, not `N`,
//! so routing still engages at `K≈1e4`); raise `--rows` to push corpus size once
//! the one-shot lane accepts a caller-supplied seed. See the run's wall time: at
//! high `K` the seed pass, not the GPU route, is the current scaling wall.

use gam_sae::tiered::{TieredFitConfig, fit_tiered};
use ndarray::Array2;
use std::process::ExitCode;
use std::time::Instant;

struct Args {
    rows: usize,
    p: usize,
    n_blocks: usize,
    block_size: usize,
    block_topk: usize,
    epochs: usize,
    minibatch: usize,
    tier2: bool,
    gpu_policy: gam_gpu::GpuPolicy,
}

impl Args {
    fn atoms(&self) -> usize {
        self.n_blocks * self.block_size
    }
}

fn parse_args() -> Result<Args, String> {
    // Defaults land on the #2023/#1017 target width: P=64, K=1e4 (2500 blocks of
    // size 4), a minibatch that clears the device launch break-even
    // (minibatch·K ≥ 2^20), and Tier-2 off (the linear-bulk lane is what the GPU
    // routing serves). N defaults modest because the one-shot Tier-1 seeder is
    // serial O(K·N·P) (see the module note); routing admission is N-independent,
    // so the device lane still engages at this width. Raise --rows to the #2023
    // N=1e5 target on the A100 box (expect the serial seed pass to dominate).
    let mut args = Args {
        rows: 32_768,
        p: 64,
        n_blocks: 2_500,
        block_size: 4,
        block_topk: 8,
        epochs: 3,
        minibatch: 512,
        tier2: false,
        gpu_policy: gam_gpu::GpuPolicy::Auto,
    };
    let raw: Vec<String> = std::env::args().skip(1).collect();
    let mut i = 0usize;
    while i < raw.len() {
        let key = raw[i].as_str();
        if key == "--tier2" {
            args.tier2 = true;
            i += 1;
            continue;
        }
        let value = raw
            .get(i + 1)
            .ok_or_else(|| format!("missing value for {key}"))?;
        match key {
            "--rows" => args.rows = parse_usize(value, key)?,
            "--p" => args.p = parse_usize(value, key)?,
            "--blocks" => args.n_blocks = parse_usize(value, key)?,
            "--block-size" => args.block_size = parse_usize(value, key)?,
            "--block-topk" => args.block_topk = parse_usize(value, key)?,
            "--epochs" => args.epochs = parse_usize(value, key)?,
            "--minibatch" => args.minibatch = parse_usize(value, key)?,
            "--gpu" => {
                args.gpu_policy = gam_gpu::GpuPolicy::parse(value)
                    .ok_or_else(|| format!("--gpu must be required|auto|off, got {value}"))?;
            }
            other => return Err(format!("unknown argument {other}")),
        }
        i += 2;
    }
    if args.p == 0 || args.rows == 0 || args.block_size == 0 || args.n_blocks == 0 {
        return Err("rows, p, blocks, and block-size must be positive".to_string());
    }
    if args.block_size > args.p {
        return Err(format!(
            "block-size {} cannot exceed P={} (a block's b orthonormal rows live in R^P)",
            args.block_size, args.p
        ));
    }
    Ok(args)
}

fn parse_usize(value: &str, key: &str) -> Result<usize, String> {
    value
        .parse::<usize>()
        .map_err(|e| format!("{key}: expected usize: {e}"))
}

/// Deterministic planted activations: `n_dirs` shared linear directions in `R^P`
/// with per-row amplitudes, plus a small deterministic per-element wobble so no
/// column is exactly constant. f64 (the tiered driver's input dtype).
fn planted_activations(rows: usize, p: usize) -> Array2<f64> {
    let n_dirs = 8usize.min(p);
    // Fixed orth.-ish direction bank from a hash; not orthonormalised (the fit
    // does not need it) but distinct enough to plant separable structure.
    let mut dirs = vec![0.0f64; n_dirs * p];
    let mut state = 0x2545_f491_4f6c_dd1du64;
    for slot in dirs.iter_mut() {
        state = splitmix64(state);
        *slot = ((state >> 11) as f64) * (1.0 / ((1u64 << 53) as f64)) - 0.5;
    }
    Array2::from_shape_fn((rows, p), |(i, c)| {
        let mut v = 0.0f64;
        for d in 0..n_dirs {
            // Row amplitude for direction d: smooth, decorrelated across d.
            let amp = ((i as f64) * (0.017 + 0.003 * d as f64) + d as f64).sin();
            v += amp * dirs[d * p + c];
        }
        // Deterministic small wobble.
        v + 0.01 * (((i * 31 + c * 17) as f64) * 0.013).cos()
    })
}

fn splitmix64(mut x: u64) -> u64 {
    x = x.wrapping_add(0x9e37_79b9_7f4a_7c15);
    let mut z = x;
    z = (z ^ (z >> 30)).wrapping_mul(0xbf58_476d_1ce4_e5b9);
    z = (z ^ (z >> 27)).wrapping_mul(0x94d0_49bb_1331_11eb);
    z ^ (z >> 31)
}

fn run() -> Result<(), String> {
    let args = parse_args()?;

    gam_gpu::configure_global_policy(args.gpu_policy);

    let k = args.atoms();
    let admitted = gam_gpu::DictionaryScoreRoutePlan::default_for_shape(args.minibatch, k, args.p);
    println!(
        "[tiered gpu scale] N={} P={} K={} (blocks={} b={}) topk={} epochs={} minibatch={} \
         tier2={} gpu={:?}",
        args.rows,
        args.p,
        k,
        args.n_blocks,
        args.block_size,
        args.block_topk,
        args.epochs,
        args.minibatch,
        args.tier2,
        args.gpu_policy,
    );
    println!(
        "[tiered gpu scale] per-minibatch device admission: minibatch*K = {} score elems, \
         device_admitted={} (break-even={})",
        args.minibatch.saturating_mul(k),
        admitted.device_admitted,
        admitted.device_min_score_elems,
    );
    if args.gpu_policy == gam_gpu::GpuPolicy::Required && !admitted.device_admitted {
        return Err(format!(
            "--gpu required but the minibatch {} x K {} is below the device launch break-even {}; \
             raise --minibatch (need minibatch >= {})",
            args.minibatch,
            k,
            admitted.device_min_score_elems,
            admitted.device_min_score_elems.div_ceil(k.max(1)),
        ));
    }
    if args.gpu_policy == gam_gpu::GpuPolicy::Required && gam_gpu::GpuRuntime::global().is_none() {
        return Err(
            "--gpu required but no CUDA runtime is available on this host; run on the A100 box"
                .to_string(),
        );
    }

    let z = planted_activations(args.rows, args.p);

    let mut config = if args.tier2 {
        TieredFitConfig::tiered(args.n_blocks, args.block_size)
    } else {
        TieredFitConfig::linear_bulk(args.n_blocks, args.block_size)
    };
    config.tier1.block_topk = args.block_topk;
    config.tier1.max_epochs = args.epochs;
    config.tier1.minibatch = args.minibatch;

    let started = Instant::now();
    let report = fit_tiered(z.view(), &config)?;
    let wall = started.elapsed().as_secs_f64();

    let n_dead = report
        .tier1
        .block_utilization
        .iter()
        .filter(|&&u| u == 0.0)
        .count();
    println!(
        "[tiered gpu scale] DONE wall={wall:.2}s tier1_ev={:.6} composed_ev={:.6} epochs_run={} \
         dead_blocks={}/{} pc_reseed_events={} births={} refusals={}",
        report.tier1.explained_variance,
        report.explained_variance,
        report.tier1.epochs,
        n_dead,
        args.n_blocks,
        report.ledger.pc_reseed_events,
        report.ledger.n_births,
        report.ledger.n_refusals,
    );
    println!(
        "[tiered gpu scale] rows/s (tier1 route+refresh) ~ {:.0}",
        (args.rows as f64) * (report.tier1.epochs.max(1) as f64) / wall.max(1.0e-9),
    );
    Ok(())
}

fn main() -> ExitCode {
    // Route logs to stderr if the caller initialised a backend; harmless if not.
    if let Err(err) = run() {
        eprintln!("[tiered gpu scale] error: {err}");
        return ExitCode::FAILURE;
    }
    ExitCode::SUCCESS
}
