//! #2023 acceptance measurement at K=2000, unblocked by the #2275 best-effort
//! completion path.
//!
//! Before #2275, `fit_tiered` hard-errored when Tier-1's frame fixed point did not
//! certify — which it never does at `K ≫ intrinsic-rank` — so this measurement was
//! impossible (Tier-2 never ran). With the best-effort path Tier-1 returns its best
//! fixed point with an OPEN certificate and Tier-2 runs on its residual, so we can
//! finally report the three #2023 acceptance numbers at scale:
//!
//! 1. **Open certificate contents** — `certified`, `frame_residual` vs `tolerance`,
//!    how many of the `K` blocks stayed live (settled) vs fell dead.
//! 2. **Held-out EV (Tier-1)** — fit the linear bulk on train, transform the held-out
//!    split through the frozen decoder, reconstruct, and score EV against the shared
//!    Tier-0 mean. This is the honest generalisation number (the Tier-1 lane has an
//!    out-of-sample transform; the Tier-2 co-fit is in-sample, so its delta below is
//!    reported in-sample and labelled as such).
//! 3. **Tier-2-adds-EV** — in-sample composed EV (Tier-1 + curved co-fit on the
//!    Tier-1 residual) minus the Tier-1-only EV on the same training corpus.
//!
//! The corpus is a deterministic planted mixture of a linear bulk plus many curved
//! (circle) factors — more circles than the block-TopK budget can host as full 2-D
//! linear blocks, so the linear tier leaves residual a 1-coordinate curved chart can
//! recover more cheaply. That budget pressure is what makes the Tier-2 delta a real
//! measurement rather than a foregone zero.

use gam_sae::sparse_dict::{block_sparse_dictionary_transform, reconstruct_block_sparse_rows};
use gam_sae::tiered::{TieredFitConfig, TieredSeedPolicy, fit_tiered};
use ndarray::{Array1, Array2, ArrayView2, Axis};
use std::process::ExitCode;
use std::time::Instant;

struct Args {
    train_rows: usize,
    test_rows: usize,
    p: usize,
    n_blocks: usize,
    block_size: usize,
    block_topk: usize,
    aux_k: usize,
    epochs: usize,
    n_circles: usize,
    n_linear: usize,
}

impl Args {
    fn k(&self) -> usize {
        self.n_blocks * self.block_size
    }
}

fn parse_args() -> Result<Args, String> {
    // Defaults: K = 1000 blocks x b=2 = 2000 atoms, block-TopK 8 (== 16 active scalar
    // coords, the #2023 K=2000 comparison budget). N_train modest so the co-fit's
    // chart compose over block pairs stays a moderate one-off runtime.
    let mut a = Args {
        train_rows: 8_192,
        test_rows: 2_048,
        p: 64,
        n_blocks: 1_000,
        block_size: 2,
        block_topk: 8,
        aux_k: 4,
        epochs: 12,
        n_circles: 24,
        n_linear: 8,
    };
    let mut it = std::env::args().skip(1);
    while let Some(flag) = it.next() {
        let mut val = || {
            it.next()
                .ok_or_else(|| format!("flag {flag} needs a value"))?
                .parse::<usize>()
                .map_err(|e| format!("flag {flag}: {e}"))
        };
        match flag.as_str() {
            "--train-rows" => a.train_rows = val()?,
            "--test-rows" => a.test_rows = val()?,
            "--p" => a.p = val()?,
            "--n-blocks" => a.n_blocks = val()?,
            "--block-size" => a.block_size = val()?,
            "--block-topk" => a.block_topk = val()?,
            "--aux-k" => a.aux_k = val()?,
            "--epochs" => a.epochs = val()?,
            "--n-circles" => a.n_circles = val()?,
            "--n-linear" => a.n_linear = val()?,
            other => return Err(format!("unknown flag {other}")),
        }
    }
    if a.p < 2 * a.n_circles + a.n_linear {
        return Err(format!(
            "need P >= 2*n_circles + n_linear = {}; got P={}",
            2 * a.n_circles + a.n_linear,
            a.p
        ));
    }
    Ok(a)
}

fn splitmix64(mut x: u64) -> u64 {
    x = x.wrapping_add(0x9e37_79b9_7f4a_7c15);
    let mut z = x;
    z = (z ^ (z >> 30)).wrapping_mul(0xbf58_476d_1ce4_e5b9);
    z = (z ^ (z >> 27)).wrapping_mul(0x94d0_49bb_1331_11eb);
    z ^ (z >> 31)
}

/// Deterministic planted mixture: `n_circles` curved factors (each a circle in its
/// own 2-D coordinate subspace, cols `[2c, 2c+1]`), plus `n_linear` shared linear
/// ramp directions in the trailing columns, plus light noise. Row `i` populates
/// every circle at a decorrelated phase, so the curved structure is genuinely
/// present (not one circle per row). Rows are generated from a global index so the
/// train and test splits are drawn i.i.d. from the same generator.
fn planted(n: usize, p: usize, n_circles: usize, n_linear: usize, index_base: u64) -> Array2<f32> {
    let mut x = Array2::<f32>::zeros((n, p));
    let lin_start = 2 * n_circles;
    for i in 0..n {
        let gi = index_base + i as u64;
        let ph = (gi as f64) * 0.201_357;
        for c in 0..n_circles {
            let theta = ph * (1.0 + c as f64 * 0.31) + c as f64;
            x[[i, 2 * c]] = theta.cos() as f32;
            x[[i, 2 * c + 1]] = theta.sin() as f32;
        }
        // Linear bulk: shared ramp directions with per-row random amplitudes.
        let mut s = splitmix64(gi ^ 0x51ed_2701_a13f_7c4d);
        for l in 0..n_linear {
            let col = lin_start + l;
            if col >= p {
                break;
            }
            s = splitmix64(s);
            let amp = ((s >> 11) as f64 / (1u64 << 53) as f64) * 2.0 - 1.0;
            x[[i, col]] += amp as f32;
        }
        // Light deterministic noise on any remaining columns.
        for col in (lin_start + n_linear)..p {
            s = splitmix64(s);
            let noise = (((s >> 11) as f64 / (1u64 << 53) as f64) - 0.5) * 0.02;
            x[[i, col]] = noise as f32;
        }
    }
    x
}

/// EV of `recon` against `target`, total sum of squares taken about `mean` (the
/// shared Tier-0 baseline): `1 − ‖target − recon‖² / ‖target − mean‖²`.
fn ev_vs_mean(target: ArrayView2<'_, f32>, recon: ArrayView2<'_, f32>, mean: &Array1<f64>) -> f64 {
    let mut rss = 0.0f64;
    let mut tss = 0.0f64;
    for (tr, rr) in target.rows().into_iter().zip(recon.rows()) {
        for c in 0..target.ncols() {
            let d = tr[c] as f64 - rr[c] as f64;
            rss += d * d;
            let dm = tr[c] as f64 - mean[c];
            tss += dm * dm;
        }
    }
    if tss <= 0.0 { f64::NAN } else { 1.0 - rss / tss }
}

fn run() -> Result<(), String> {
    let args = parse_args()?;
    let z_train = planted(args.train_rows, args.p, args.n_circles, args.n_linear, 0);
    let z_test = planted(
        args.test_rows,
        args.p,
        args.n_circles,
        args.n_linear,
        args.train_rows as u64,
    );
    let z_train_f64 = z_train.mapv(|v| v as f64);

    println!(
        "[k2000] N_train={} N_test={} P={} K={} (blocks={} b={}) block_topk={} aux_k={} epochs={} \
         circles={} linear={}",
        args.train_rows,
        args.test_rows,
        args.p,
        args.k(),
        args.n_blocks,
        args.block_size,
        args.block_topk,
        args.aux_k,
        args.epochs,
        args.n_circles,
        args.n_linear,
    );

    // --- Tier-1 only (linear-bulk baseline) ---
    let mut lin = TieredFitConfig::linear_bulk(args.n_blocks, args.block_size);
    lin.tier1_seed = TieredSeedPolicy::Auto; // Auto picks the cheap coordinate seed at K=2000.
    lin.tier1.block_topk = args.block_topk;
    lin.tier1.aux_k = args.aux_k;
    lin.tier1.max_epochs = args.epochs;
    let t0 = Instant::now();
    let lin_report = fit_tiered(z_train_f64.view(), &lin)?;
    let lin_wall = t0.elapsed().as_secs_f64();

    let cert = &lin_report.tier1.convergence;
    let n_live = lin_report
        .tier1
        .block_utilization
        .iter()
        .filter(|&&u| u > 0.0)
        .count();
    println!(
        "[k2000] CERTIFICATE certified={} frame_residual={:.3e} tolerance={:.3e} \
         ev_residual={:.3e} gamma_residual={:.3e} routing_residual={:.3e} \
         reconstruction_residual={:.3e} accepted_births={} polar_failures={} \
         live_blocks={}/{} epochs_run={} wall={:.1}s",
        cert.certified,
        cert.frame_residual,
        cert.tolerance,
        cert.ev_residual,
        cert.gamma_residual,
        cert.routing_residual,
        cert.reconstruction_residual,
        cert.accepted_births,
        cert.polar_failures,
        n_live,
        args.k(),
        lin_report.tier1.epochs,
        lin_wall,
    );

    let ev_t1_in = lin_report.tier1.explained_variance;

    // --- Held-out Tier-1 EV: transform the test split through the frozen decoder. ---
    let mean = &lin_report.tier0.mean;
    let r0_test = &z_test - &mean.mapv(|v| v as f32).view().insert_axis(Axis(0));
    let (blocks_te, _gates_te, codes_te) = block_sparse_dictionary_transform(
        r0_test.view(),
        lin_report.tier1.decoder.view(),
        lin_report.tier1.gamma,
        args.block_size,
        args.block_topk,
        lin.tier1.block_tile,
    )?;
    let recon_te = reconstruct_block_sparse_rows(
        lin_report.tier1.decoder.view(),
        blocks_te.view(),
        codes_te.view(),
        args.block_size,
    )?;
    // Held-out EV about the (train) Tier-0 mean: r0_test is already test − mean, so the
    // baseline TSS is ‖r0_test‖² and RSS is ‖r0_test − recon‖²; add the mean back on
    // both sides so ev_vs_mean scores the reconstruction in z-space consistently.
    let recon_te_z = &recon_te + &mean.mapv(|v| v as f32).view().insert_axis(Axis(0));
    let ev_t1_heldout = ev_vs_mean(z_test.view(), recon_te_z.view(), mean);
    println!(
        "[k2000] TIER1 in_sample_ev={:.6} held_out_ev={:.6}",
        ev_t1_in, ev_t1_heldout
    );

    // --- Tiered (Tier-1 + Tier-2 curved co-fit on the residual) ---
    let mut tiered = TieredFitConfig::tiered(args.n_blocks, args.block_size);
    tiered.tier1_seed = TieredSeedPolicy::Auto;
    tiered.tier1.block_topk = args.block_topk;
    tiered.tier1.aux_k = args.aux_k;
    tiered.tier1.max_epochs = args.epochs;
    let t1 = Instant::now();
    let tiered_report = fit_tiered(z_train_f64.view(), &tiered)?;
    let tiered_wall = t1.elapsed().as_secs_f64();
    let ev_composed_in = tiered_report.explained_variance;
    let tier2_rounds = tiered_report
        .tier2
        .as_ref()
        .map(|r| r.rounds.len())
        .unwrap_or(0);

    println!(
        "[k2000] COMPOSED in_sample_ev={:.6} tier2_rounds={} births={} refusals={} \
         pc_reseed_events={} wall={:.1}s",
        ev_composed_in,
        tier2_rounds,
        tiered_report.ledger.n_births,
        tiered_report.ledger.n_refusals,
        tiered_report.ledger.pc_reseed_events,
        tiered_wall,
    );
    println!(
        "[k2000] TIER2_ADDS_EV (in-sample) = composed {:.6} − tier1 {:.6} = {:+.6}",
        ev_composed_in,
        ev_t1_in,
        ev_composed_in - ev_t1_in,
    );
    println!("[k2000] DONE");
    Ok(())
}

fn main() -> ExitCode {
    match run() {
        Ok(()) => ExitCode::SUCCESS,
        Err(e) => {
            eprintln!("[k2000] error: {e}");
            ExitCode::FAILURE
        }
    }
}
