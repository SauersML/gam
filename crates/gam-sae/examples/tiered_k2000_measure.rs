//! #2023 acceptance measurement at K=2000.
//!
//! Fits the linear bulk alone and the full tiered cadence on one planted corpus. Tier-1
//! returns a certified frame fixed point, or its best fixed point with `certified =
//! false` once the captured-fraction plateau holds (#2275), and refuses otherwise. A
//! returned Tier-1 hands its residual to Tier-2, so the harness reports the three #2023
//! acceptance numbers at scale:
//!
//! 1. **Certificate contents** — `frame_residual` vs `tolerance`,
//!    how many of the `K` blocks stayed live (settled) vs fell dead.
//! 2. **Held-out EV (Tier-1)** — fit the linear bulk on train, transform the held-out
//!    split through the frozen decoder, reconstruct, and score EV against the shared
//!    Tier-0 mean. This is the honest generalisation number (the Tier-1 lane has an
//!    out-of-sample transform; the Tier-2 curved refinement is in-sample, so its delta
//!    below is reported in-sample and labelled as such).
//! 3. **Tier-2-adds-EV** — in-sample composed EV (Tier-1 + the curved support-sparse
//!    refinement on the Tier-1 residual) minus the Tier-1-only EV on the same training
//!    corpus.
//!
//! The corpus is a deterministic planted mixture of a linear bulk plus many curved
//! (circle) factors — more circles than the block-TopK budget can host as full 2-D
//! linear blocks, so the linear tier leaves residual a 1-coordinate curved chart can
//! recover more cheaply. That budget pressure is what makes the Tier-2 delta a real
//! measurement rather than a foregone zero.
//!
//! Before the K=2000 fit the harness runs a **planted-ring control** (#2023 lead
//! ruling): a centered ring in two of `P = 4` columns, charted by the same curved
//! dictionary on two targets — the Tier-0-centered data the public support-sparse
//! fit charts, and the residual a linear peel at the derived width (`G = P`,
//! `b = 1`, block TopK `s = 2`) leaves. Two linear atoms reconstruct a centered ring
//! exactly, so the control measures whether the curved atoms can still find the
//! ring on the peeled residual.

use gam_sae::manifold::{SaeSupportSparseFit, SaeSupportSparseFitRequest, fit_sae_support_sparse};
use gam_sae::sparse_dict::{block_sparse_dictionary_transform, reconstruct_block_sparse_rows};
use gam_sae::tiered::{TieredFitConfig, fit_tiered};
use ndarray::{Array1, Array2, ArrayView2, Axis};
use std::process::ExitCode;
use std::time::Instant;

mod common;
use common::splitmix64;

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
    // coords, the #2023 K=2000 comparison budget). N_train modest so the curved
    // Tier-2 refinement stays a moderate one-off runtime.
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
    if tss <= 0.0 {
        f64::NAN
    } else {
        1.0 - rss / tss
    }
}

/// The curved dictionary both ring-control arms fit: 8 periodic atoms, TopK
/// `s = 2`, at the public entry's inner tolerance.
fn ring_curved_fit(target: ArrayView2<'_, f64>) -> Result<SaeSupportSparseFit, String> {
    fit_sae_support_sparse(SaeSupportSparseFitRequest {
        target,
        atom_basis: vec!["periodic".to_string(); 8],
        atom_dim: vec![1; 8],
        support_k: 2,
        initial_smoothness: 1.0,
        max_outer_iter: 32,
        trust_radius: 1.0,
        random_state: 0xC0FF_EE00_D15E_A5E5,
    })
}

/// How much of the planted ring a curved fit found.
///
/// `ring_ev` is the share of the ring's energy that the curved reconstruction
/// (`fitted − training_mean`, in the fit's own target space) carries in the two
/// ring columns. `phase_lock` is the largest per-atom phase-locking value
/// `|mean exp(i(2πt ∓ θ))|` between an atom's chart coordinate `t` (period 1) and
/// the planted phase `θ`, over atoms routed on at least a sixteenth of the rows;
/// the returned count is that atom's routed rows.
fn ring_scores(
    fit: &SaeSupportSparseFit,
    ring: ArrayView2<'_, f64>,
    phase: &Array1<f64>,
) -> (f64, f64, usize) {
    let n = ring.nrows();
    let mut rss = 0.0_f64;
    let mut tss = 0.0_f64;
    for i in 0..n {
        for c in 0..2 {
            let curved = fit.fitted[[i, c]] - fit.training_mean[c];
            rss += (ring[[i, c]] - curved).powi(2);
            tss += ring[[i, c]].powi(2);
        }
    }
    let ring_ev = 1.0 - rss / tss;
    let term = &fit.outer.term;
    let k = term.k_atoms();
    let mut sums = vec![[0.0_f64; 4]; k];
    let mut counts = vec![0usize; k];
    for i in 0..n {
        for (slot, &atom) in term.assignment.support_indices(i).iter().enumerate() {
            let angle = std::f64::consts::TAU * term.assignment.coords_for_slot(i, slot)[0];
            let atom = atom as usize;
            sums[atom][0] += (angle - phase[i]).cos();
            sums[atom][1] += (angle - phase[i]).sin();
            sums[atom][2] += (angle + phase[i]).cos();
            sums[atom][3] += (angle + phase[i]).sin();
            counts[atom] += 1;
        }
    }
    let mut best = (0.0_f64, 0usize);
    for atom in 0..k {
        if counts[atom] * 16 < n {
            continue;
        }
        let rows = counts[atom] as f64;
        let lock = sums[atom][0]
            .hypot(sums[atom][1])
            .max(sums[atom][2].hypot(sums[atom][3]))
            / rows;
        if lock > best.0 {
            best = (lock, counts[atom]);
        }
    }
    (ring_ev, best.0, best.1)
}

/// The planted-ring control (#2023 lead ruling): does the curved dictionary find
/// a centered ring on the Tier-0 target, and on the residual of a linear peel at
/// the derived width? Refusals print their error; they are results too.
fn ring_control() -> Result<(), String> {
    let n = 512usize;
    let p = 4usize;
    let support_k = 2usize;
    let mut z = Array2::<f64>::zeros((n, p));
    let mut ring = Array2::<f64>::zeros((n, 2));
    let mut phase = Array1::<f64>::zeros(n);
    let mut s = 0x2023_0000_2576_0001_u64;
    for i in 0..n {
        let theta = std::f64::consts::TAU * i as f64 / n as f64;
        phase[i] = theta;
        ring[[i, 0]] = theta.cos();
        ring[[i, 1]] = theta.sin();
        for c in 0..p {
            s = splitmix64(s);
            let noise = (((s >> 11) as f64 / (1u64 << 53) as f64) - 0.5) * 0.04;
            let signal = if c < 2 { ring[[i, c]] } else { 0.0 };
            z[[i, c]] = signal + noise;
        }
    }
    println!(
        "[ring] N={n} P={p} ring in cols 0,1 + uniform noise ±0.02 on every column; curved: 8 \
         periodic atoms, s={support_k}; peel: G={p} b=1 block_topk={support_k}"
    );

    let t0 = Instant::now();
    match ring_curved_fit(z.view()) {
        Ok(fit) => {
            let (ring_ev, lock, rows) = ring_scores(&fit, ring.view(), &phase);
            println!(
                "[ring] tier0_target r2={:.6} ring_ev={ring_ev:.6} phase_lock={lock:.4} \
                 (atom rows {rows}/{n}) retained={} outer_iters={} wall={:.1}s",
                fit.reconstruction_r2,
                fit.retained_atom_indices.len(),
                fit.outer.outer_iterations,
                t0.elapsed().as_secs_f64(),
            );
        }
        Err(error) => println!(
            "[ring] tier0_target refused after {:.1}s: {error}",
            t0.elapsed().as_secs_f64()
        ),
    }

    let mut peel = TieredFitConfig::linear_bulk(p, 1);
    peel.tier1.block_topk = support_k;
    peel.tier1.aux_k = support_k;
    peel.tier1.max_epochs = 200;
    let t1 = Instant::now();
    let peel_report = match fit_tiered(z.view(), &peel) {
        Ok(report) => report,
        Err(error) => {
            println!(
                "[ring] peel refused after {:.1}s: {error}",
                t1.elapsed().as_secs_f64()
            );
            return Ok(());
        }
    };
    let mean = &peel_report.tier0.mean;
    let centered = &z - &mean.view().insert_axis(Axis(0));
    let centered_f32 = centered.mapv(|v| v as f32);
    let (blocks, _gates, codes) = block_sparse_dictionary_transform(
        centered_f32.view(),
        peel_report.tier1.decoder.view(),
        peel_report.tier1.gamma,
        1,
        peel_report.tier1.block_topk,
        peel.tier1.block_tile,
    )?;
    let linear = reconstruct_block_sparse_rows(
        peel_report.tier1.decoder.view(),
        blocks.view(),
        codes.view(),
        1,
    )?;
    let residual = &centered - &linear.mapv(|v| v as f64);
    let residual_share = residual.iter().map(|v| v * v).sum::<f64>()
        / centered.iter().map(|v| v * v).sum::<f64>();
    println!(
        "[ring] peel tier1_ev={:.6} certified={} residual_energy_share={residual_share:.6} \
         wall={:.1}s",
        peel_report.tier1.explained_variance,
        peel_report.tier1.convergence.certified,
        t1.elapsed().as_secs_f64(),
    );
    let t2 = Instant::now();
    match ring_curved_fit(residual.view()) {
        Ok(fit) => {
            let (ring_ev, lock, rows) = ring_scores(&fit, ring.view(), &phase);
            println!(
                "[ring] peeled_target r2={:.6} ring_ev={ring_ev:.6} phase_lock={lock:.4} \
                 (atom rows {rows}/{n}) retained={} outer_iters={} wall={:.1}s",
                fit.reconstruction_r2,
                fit.retained_atom_indices.len(),
                fit.outer.outer_iterations,
                t2.elapsed().as_secs_f64(),
            );
        }
        Err(error) => println!(
            "[ring] peeled_target refused after {:.1}s: {error}",
            t2.elapsed().as_secs_f64()
        ),
    }
    Ok(())
}

fn run() -> Result<(), String> {
    ring_control()?;
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
        "[k2000] CERTIFICATE frame_residual={:.3e} tolerance={:.3e} \
         ev_residual={:.3e} gamma_residual={:.3e} routing_residual={:.3e} \
         reconstruction_residual={:.3e} accepted_births={} polar_failures={} \
         live_blocks={}/{} epochs_run={} wall={:.1}s",
        // The residual fields are the evidence behind `cert.certified`: a fit
        // that neither certifies nor reaches its captured-fraction plateau errors
        // out before finalization instead of returning.
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

    // --- Tiered (Tier-1 + Tier-2 curved support-sparse refinement on the residual) ---
    let mut tiered = TieredFitConfig::tiered(args.n_blocks, args.block_size);
    tiered.tier1.block_topk = args.block_topk;
    tiered.tier1.aux_k = args.aux_k;
    tiered.tier1.max_epochs = args.epochs;
    let t1 = Instant::now();
    let tiered_report = fit_tiered(z_train_f64.view(), &tiered)?;
    let tiered_wall = t1.elapsed().as_secs_f64();
    let ev_composed_in = tiered_report.explained_variance;
    let tier2_outer_iters = tiered_report
        .tier2
        .as_ref()
        .map(|r| r.outer_iterations)
        .unwrap_or(0);

    println!(
        "[k2000] COMPOSED in_sample_ev={:.6} tier2_outer_iters={} births={} refusals={} \
         pc_reseed_events={} wall={:.1}s",
        ev_composed_in,
        tier2_outer_iters,
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
    env_logger::init();
    match run() {
        Ok(()) => ExitCode::SUCCESS,
        Err(e) => {
            eprintln!("[k2000] error: {e}");
            ExitCode::FAILURE
        }
    }
}
