//! Alternating minibatched trainer: route → sparse codes → decoder refresh →
//! unit-norm projection. No dense `N×K` object is ever formed.
//!
//! The decoder refresh is the **method of optimal directions** (MOD) restricted
//! to the sparse support. With codes fixed, the reconstruction loss
//! `Σ_i ‖x_i − Σ_j c_{ij} d_{a_{ij}}‖²` is quadratic in the decoder `D` and its
//! normal equations are `D (CᵀC + ρI) = CᵀX`, where `C` is the (sparse, never
//! materialised) `N×K` code matrix. We accumulate `A = CᵀC` (`K×K`, but only
//! the few entries touched by co-active atoms are non-zero) and `B = CᵀX`
//! (`K×P`) by streaming minibatches, then solve **to the rank-charge floor**.
//! For a clean `top_s = 1` lane `A` is diagonal and the refresh is a per-atom
//! rescaled mean — exactly MOD / k-SVD's dictionary step. For the general
//! `s > 1` case the coupling `A` is non-diagonal, and the co-firing graph
//! percolates at realistic scale, so connected components are diagnostics rather
//! than a useful dense-solve decomposition. The default coupled solve is
//! therefore matrix-free conjugate gradients: every Gram-vector product touches
//! only the streamed sparse normal equations (`O(K + nnz)`) and no dense `K×K`
//! block is formed. Dense Cholesky is retained only for genuinely tiny connected
//! components. CG stops when the relative normal-equation residual is below the
//! ridge/charge floor, and its Lanczos tridiagonal supplies the condition
//! estimate reported with the epoch diagnostics.

use super::codes::{SparseCode, solve_row_codes};
use super::scoring::{ScoreRoutePath, ScoreRouteStats, TileScorer};
use super::{SparseDictConfig, SparseDictFit};
use ndarray::{Array2, ArrayView2, Axis};
use rayon::prelude::*;
use std::collections::HashMap;
use std::time::Instant;

/// Route + sparse-code every row of `x`, processing the rows in minibatches of
/// `config.minibatch` so the peak score working set is `minibatch × score_tile`
/// (never `N × K`). Within a minibatch the rows are routed by the shared
/// [`TileScorer::route_minibatch_dispatch`] policy: GPU score-blocks when
/// admitted, otherwise the batched CPU GEMM router. The per-row active-set code
/// solves run in parallel. The returned `Vec<SparseCode>` is in global row order,
/// identical to a serial row-at-a-time pass up to f32 GEMM rounding.
pub(super) fn route_and_code_all(
    x: ArrayView2<'_, f32>,
    decoder: ArrayView2<'_, f32>,
    scorer: &TileScorer,
    s: usize,
    code_ridge: f32,
    minibatch: usize,
    score_mode: gam_gpu::GpuMode,
    mut score_route_stats: Option<&mut ScoreRouteStats>,
) -> Result<Vec<SparseCode>, String> {
    let n = x.nrows();
    let batch = minibatch.max(1);
    if n == 0 {
        return Ok(Vec::new());
    }

    // Probe the first minibatch to learn whether this fit routes on the device
    // or on the host. The score GEMM is by far the dominant cost of the whole
    // fit — O(N·K·P) per pass — and `ndarray`'s `.dot` is single-threaded
    // `matrixmultiply`, so a serial minibatch loop pins the entire pass to ONE
    // core (≈100 h/pass at K≈32k, N≈96k, P≈2048). When the route lands on the
    // host we must fan it across all cores; the device path stays serial so the
    // CUDA score-block calls are never issued concurrently.
    let first_end = batch.min(n);
    let first_block = x.slice(ndarray::s![0..first_end, ..]);
    let first_routed = scorer.route_minibatch_with_mode(first_block, decoder, score_mode)?;
    let path = first_routed.path;
    if let Some(stats) = score_route_stats.as_deref_mut() {
        stats.record_result(&first_routed);
    }
    let first_active = first_routed.selections;
    let mut codes: Vec<SparseCode> = first_block
        .axis_iter(Axis(0))
        .into_par_iter()
        .zip(first_active.into_par_iter())
        .map(|(row, active)| solve_row_codes(row, decoder, &active, s, code_ridge))
        .collect();

    if path == ScoreRoutePath::Cpu {
        // Host route: fan the remaining rows across cores at minibatch
        // granularity. Each chunk runs the batched CPU score GEMM (serial per
        // chunk, so the decoder tile is reused across the whole minibatch) plus
        // its own independent per-row active-set code solves; the chunk is the
        // parallel unit, so there is no nested rayon fork-join. Per-row routing
        // and code solves depend only on their own row, so the concatenation is
        // order-identical to the serial pass up to f32 GEMM rounding.
        let plan = gam_gpu::DictionaryScoreRoutePlan::default_for_shape(
            batch,
            decoder.nrows(),
            decoder.ncols(),
        );
        if first_end < n {
            let rest = x.slice(ndarray::s![first_end.., ..]);
            let chunk_codes: Vec<Vec<SparseCode>> = rest
                .axis_chunks_iter(Axis(0), batch)
                .into_par_iter()
                .map(|chunk| {
                    let routed = scorer.route_minibatch(chunk, decoder);
                    chunk
                        .axis_iter(Axis(0))
                        .zip(routed.into_iter())
                        .map(|(row, active)| solve_row_codes(row, decoder, &active, s, code_ridge))
                        .collect::<Vec<SparseCode>>()
                })
                .collect();
            for chunk in chunk_codes {
                // One route record per minibatch, mirroring the serial path's
                // per-minibatch accounting (counts are order-independent).
                if let Some(stats) = score_route_stats.as_deref_mut() {
                    stats.record(plan, ScoreRoutePath::Cpu);
                }
                codes.extend(chunk);
            }
        }
    } else {
        // Device route: keep the serial per-minibatch dispatch so CUDA
        // score-block launches are never concurrent.
        let mut start = first_end;
        while start < n {
            let end = (start + batch).min(n);
            let block = x.slice(ndarray::s![start..end, ..]);
            let routed = scorer.route_minibatch_with_mode(block, decoder, score_mode)?;
            if let Some(stats) = score_route_stats.as_deref_mut() {
                stats.record_result(&routed);
            }
            let active_lists = routed.selections;
            let mut block_codes: Vec<SparseCode> = block
                .axis_iter(Axis(0))
                .into_par_iter()
                .zip(active_lists.into_par_iter())
                .map(|(row, active)| solve_row_codes(row, decoder, &active, s, code_ridge))
                .collect();
            codes.append(&mut block_codes);
            start = end;
        }
    }
    Ok(codes)
}

pub(super) fn run(
    x: ArrayView2<'_, f32>,
    config: &SparseDictConfig,
) -> Result<SparseDictFit, String> {
    validate(x, config)?;
    let n = x.nrows();
    let p = x.ncols();
    let k = config.n_atoms;
    let s = config.active.min(k).max(1);

    let fit_start = Instant::now();
    let mut decoder = seed_decoder(x, k);
    unit_norm_rows(&mut decoder);
    // Coarse phase heartbeat on the same channel as the score-router DECLINE
    // (log::warn survives the RUST_LOG=warn harnesses that drop log::info), so a
    // multi-hour host fit is never silent. Emitted at seed / initial-route /
    // per-epoch cadence only — never per row or per minibatch.
    log::warn!(
        "[SAE sparse_dict] seeded decoder N={n} P={p} K={k} s={s} \
         seed_s={:.1} (route + refresh follow)",
        fit_start.elapsed().as_secs_f64(),
    );

    let scorer = TileScorer::new(s, config.score_tile);
    let mut score_route_stats = ScoreRouteStats::default();
    let mut pending_eq = DecoderNormalEq::zeros(k, p);
    let mut prev_ev = f64::NEG_INFINITY;
    let mut converged = false;
    let mut epochs_run = 0usize;
    let mut decoder_solve_stats = DecoderSolveStats::default();

    // (a)+(b) route + sparse codes for every row against the seeded, unit-normed
    // decoder, in minibatches: each minibatch is routed by one batched score block
    // per column tile (peak score working set `minibatch × score_tile`, never
    // `N × K`) — on the GPU when the process admits a device and the block clears
    // the break-even, else the parallel CPU GEMM — and its per-row active-set code
    // solves run in parallel. These codes feed the first decoder refresh.
    let initial_route_start = Instant::now();
    let mut codes = route_and_code_all(
        x,
        decoder.view(),
        &scorer,
        s,
        config.code_ridge,
        config.minibatch,
        config.score_mode,
        Some(&mut score_route_stats),
    )?;
    log::warn!(
        "[SAE sparse_dict] initial route done: minibatches={} device={} cpu={} \
         route_s={:.1} elapsed_s={:.1}",
        score_route_stats.minibatches,
        score_route_stats.device_minibatches,
        score_route_stats.cpu_minibatches,
        initial_route_start.elapsed().as_secs_f64(),
        fit_start.elapsed().as_secs_f64(),
    );

    for epoch in 0..config.max_epochs {
        epochs_run = epoch + 1;
        let epoch_start = Instant::now();

        // (c) decoder refresh: stream the current codes into each atom's pending
        // normal equations, then refresh only atoms whose evidence clears the
        // routability SE gate. Deferred atoms keep accumulating across epochs.
        pending_eq.accumulate(x, &codes);
        let sigma = residual_scale(x, &codes, decoder.view());
        let (stats, gate) = solve_decoder_with_routability_gate(
            &mut decoder,
            &pending_eq,
            config.decoder_ridge as f64,
            sigma,
        );
        decoder_solve_stats = stats;
        pending_eq.clear_refreshed_atoms(&gate);
        let refresh_secs = epoch_start.elapsed().as_secs_f64();

        // (d) unit-norm projection (identifies code scale) + stable sign.
        unit_norm_rows(&mut decoder);

        // (e) dead-atom revival. Atoms that fired for no row this epoch are re-
        // seeded onto the current worst-reconstructed rows' residual directions.
        // Without this, a large dictionary leaves most atoms at their seed (see
        // the dead counts in the fit report / #1026): effective `K` collapses to a
        // handful of live atoms, EV is non-monotone in `K`, and the lane never
        // climbs toward reconstruction parity. Reviving toward high-residual rows
        // is the standard dead-feature resampling that makes every atom load-
        // bearing, so adding atoms can only help. It runs only while dead atoms
        // remain, so a fully-alive small-`K` dictionary is untouched.
        let revived = revive_dead_atoms(x, &codes, &mut decoder);
        if revived > 0 {
            unit_norm_rows(&mut decoder);
        }

        // (a)+(b) FRESH codes against the just-refreshed, unit-normed decoder.
        // These are the codes that define the post-epoch model, so they (i) feed
        // the NEXT epoch's refresh and (ii) score the convergence EV below. This
        // re-route deliberately replaces the previous STALE-code EV (which scored
        // the new decoder against codes solved before the refresh + normalisation):
        // the convergence decision now uses exactly the codes that define the
        // returned model, so there is no stale-code surrogate gap.
        codes = route_and_code_all(
            x,
            decoder.view(),
            &scorer,
            s,
            config.code_ridge,
            config.minibatch,
            config.score_mode,
            Some(&mut score_route_stats),
        )?;

        let route_secs = epoch_start.elapsed().as_secs_f64() - refresh_secs;

        // Convergence-decision EV, computed from the FRESH post-normalisation codes.
        let ev = explained_variance(x, &codes, decoder.view());
        let improve = ev - prev_ev;

        // Per-epoch heartbeat on the log::warn channel (log::info is dropped by
        // the RUST_LOG=warn harnesses, which is why a multi-hour host fit went
        // silent). A hang in the refresh or route is visible at round cadence,
        // and the CG certificate (giant component size, the a-priori κ bound,
        // any typed non-convergence) is on the same line.
        log::warn!(
            "[SAE epoch {}/{}] ev={:.6} improve={:.3e} revived={} refresh_s={:.2} \
             route_s={:.2} elapsed_s={:.1} max_component={} cg_columns={} cg_nonconverged={} \
             cg_kappa_bound={:?} cg_relative_residual={:.3e}",
            epochs_run,
            config.max_epochs,
            ev,
            improve,
            revived,
            refresh_secs,
            route_secs,
            fit_start.elapsed().as_secs_f64(),
            decoder_solve_stats.max_component_size,
            decoder_solve_stats.cg_columns,
            decoder_solve_stats.cg_nonconverged_columns,
            decoder_solve_stats.cg_kappa_bound,
            decoder_solve_stats.cg_relative_residual,
        );
        // #1026 — do NOT declare convergence while dead atoms are still being
        // revived. Revival runs at most one atom per row per epoch, so a large
        // dictionary populates its tail over SEVERAL epochs; a one-epoch EV
        // plateau can occur mid-population (a revived atom needs the next route to
        // start firing). Stopping on that plateau froze the dictionary with a
        // still-dead tail — effective `K` below the requested `K`, the exact
        // under-population this issue tracks. Requiring `revived == 0` forces every
        // atom to become load-bearing before the plateau test can fire, so the
        // returned dictionary uses its full budget and its EV climbs with `K`
        // toward reconstruction parity. Once no atom is dead, revival returns 0 and
        // the ordinary EV-plateau test governs stopping exactly as before.
        if revived == 0 && improve.abs() <= config.tolerance && epoch > 0 {
            converged = true;
            break;
        }
        prev_ev = ev;
    }

    // `codes` already hold the re-route against the final, unit-normed decoder
    // (the last epoch's step-(a)+(b)), so the stored codes and the returned EV
    // match the returned dictionary exactly — no extra reroute needed, and the
    // returned EV equals the convergence-decision EV from the final epoch.
    let final_ev = explained_variance(x, &codes, decoder.view());

    let (indices, code_mat) = pack_codes(&codes, n, s);
    Ok(SparseDictFit {
        decoder,
        indices,
        codes: code_mat,
        explained_variance: final_ev,
        epochs: epochs_run,
        converged,
        active: s,
        score_route_stats,
        decoder_solve_stats,
    })
}

/// The unified **linear fast kernel** (design gam#2232, Increment 2, plug points
/// 1–3): the fixed-support linear-atom (`d = 1`) inner solve of the ONE engine.
///
/// This is the exact alternation of [`run`] — `route → s×s active-set code solve
/// → MOD sparse decoder refresh → unit-norm` — but parameterized by a SINGLE
/// shared ridge coordinate `shared_rho` that feeds BOTH
///
///   * the per-row active-set code/gate solve (plug point 1,
///     [`super::codes::solve_row_codes`]), and
///   * the per-atom decoder normal-equation refresh (plug point 2,
///     [`solve_decoder_with_routability_gate`]),
///
/// with routing kept on [`TileScorer::top_s_online`] (plug point 3), never
/// materializing `N×K`. Collapsing the historical TWO independent ridges
/// (`code_ridge`, `decoder_ridge`) into ONE shared `shared_rho` is the `d = 1`
/// specialization of the framed curved refresh's single shared variance
/// component, and it is the precondition for the shared-REML selection of that
/// component (plug point 4): a single ρ coordinate the outer evidence loop
/// selects instead of two magic constants.
///
/// At `shared_rho = config.code_ridge = config.decoder_ridge` (the shared default
/// `1e-6`) this kernel is BIT-IDENTICAL to the historical [`run`] — pinned by the
/// TEMPORARY parity gate `linear_fast_kernel_matches_legacy_run` (design
/// Increment 2 shim; removed in Increment 6). It is invoked from the unified
/// engine's inner-solve seam; [`super::fit_sparse_dictionary`] remains a thin
/// wrapper over [`run`] for now (its two-ridge FFI surface is preserved until
/// Increment 5 re-points callers to the REML schedule).
pub fn run_linear_fast_kernel(
    x: ArrayView2<'_, f32>,
    config: &SparseDictConfig,
    shared_rho: f64,
) -> Result<SparseDictFit, String> {
    let mut unified = *config;
    // ONE shared variance coordinate drives both the code and the decoder ridge:
    // the `d = 1` specialization carries a single ρ, not two.
    unified.code_ridge = shared_rho as f32;
    unified.decoder_ridge = shared_rho as f32;
    run(x, &unified)
}

/// Sufficient statistics for the linear block's ONE shared REML variance
/// component (design gam#2232, Increment 2, plug point 4).
///
/// The decoder refresh is `P` independent ridge regressions `(A + ρI) D_{:,c} =
/// B_{:,c}` (`A = CᵀC` the `K×K` code Gram, `B = CᵀX`) that SHARE the single ridge
/// `ρ`, with an identity roughness penalty `ρ‖D‖²_F`. Reading `ρ = σ²/τ²` (noise
/// variance over decoder prior variance) makes the refresh a Gaussian ridge whose
/// evidence-optimal ρ is a Fellner–Schall / MacKay fixed point over exactly these
/// aggregates.
#[derive(Clone, Copy, Debug)]
pub struct LinearBlockRemlStats {
    /// Per-column effective degrees of freedom `γ = tr(A (A + ρI)⁻¹)` of the code
    /// Gram at the current ρ. Identical across the `P` output columns because the
    /// ridge operator `(A + ρI)⁻¹` is column-independent, so the pooled effective
    /// dof is `P·γ`.
    pub gram_edof: f64,
    /// Output dimension `P` (number of decoder columns sharing ρ).
    pub p_cols: usize,
    /// Decoder penalty energy `‖D‖²_F = Σ_{k,c} D_{kc}²` (identity roughness), i.e.
    /// the roughness quadratic form of the just-refreshed decoder.
    pub penalty_energy: f64,
    /// Reconstruction residual sum of squares `Σ_i ‖x_i − Σ_j c_{ij} d_{a_{ij}}‖²`.
    pub rss: f64,
    /// Rows `N` (the ridge regressions have `N·P` total observations).
    pub n_obs: usize,
}

/// One Fellner–Schall / MacKay evidence fixed-point update of the linear block's
/// shared ridge ρ (design gam#2232, Increment 2, plug point 4).
///
/// For the shared-ρ pooled ridge (see [`LinearBlockRemlStats`]) the REML fixed
/// point is the standard evidence recursion:
///
/// ```text
///   γ_tot = P · tr(A (A + ρI)⁻¹)                (pooled effective dof)
///   σ̂²    = RSS / (N·P − γ_tot)                 (REML residual variance)
///   τ̂²    = ‖D‖²_F / γ_tot                       (decoder prior variance)
///   ρ_new = σ̂² / τ̂² = γ_tot · σ̂² / ‖D‖²_F
/// ```
///
/// This is the ONE shared REML variance component of the design — no per-atom
/// λ, no new optimizer, the same Fellner–Schall fixed point the outer engine
/// runs, specialized to the `d = 1` linear block. It is guarded to stay strictly
/// positive and finite so the next `(A + ρI)` solve stays SPD; a degenerate
/// aggregate (all dof consumed, zero penalty energy) leaves ρ unchanged.
pub fn linear_shared_rho_fs_step(stats: &LinearBlockRemlStats, rho: f64) -> f64 {
    let gamma_tot = (stats.p_cols as f64) * stats.gram_edof;
    let total_obs = (stats.n_obs.saturating_mul(stats.p_cols)) as f64;
    let resid_dof = (total_obs - gamma_tot).max(1.0);
    let sigma2 = stats.rss / resid_dof;
    let energy = stats.penalty_energy;
    if !(energy > 0.0) || !gamma_tot.is_finite() || gamma_tot <= 0.0 {
        return rho;
    }
    let rho_new = gamma_tot * sigma2 / energy;
    if rho_new.is_finite() && rho_new > 0.0 {
        rho_new
    } else {
        rho
    }
}

/// Variance ceiling for the matrix-free effective-dof estimator, expressed as a
/// fraction of the trace it estimates (design gam#2232, Increment 2, plug 4).
///
/// The edof is estimated by a Hutchinson (Rademacher) stochastic-trace probe of
/// the symmetric PSD operator whose trace is the complementary dof
/// `c = tr(ρ(A+ρI)⁻¹)` (its spectrum `ρ/(λ+ρ)` lies in `(0, 1]`). For such a
/// matrix `M ⪰ 0` with spectrum in `[0, 1]` the single-probe Rademacher variance
/// is `2(‖M‖²_F − Σ Mᵢᵢ²) ≤ 2‖M‖²_F = 2 Σμ² ≤ 2 Σμ = 2 tr(M)` (using `μ² ≤ μ`
/// on `[0, 1]`); averaging `m` independent probes gives
/// `Var(ĉ) ≤ 2 tr(M) / m`. Requiring that variance to be at most a fraction `v`
/// of the trace it estimates — `Var(ĉ) ≤ v · tr(M)` — fixes the probe count
/// `m = ⌈2/v⌉` with NO dependence on the (unknown) spectrum: it is the universal
/// PSD-trace bound. This is the single documented quality knob; the probe count
/// is derived from it, not tuned.
const EDOF_TRACE_VARIANCE_PER_UNIT_TRACE: f64 = 0.05;

/// Relative normal-equation residual at which each Hutchinson trace solve stops.
///
/// The solve `(A + ρI) w = z` supplies the per-probe `zᵀ(A+ρI)⁻¹z`; a residual
/// this small is orders of magnitude below the stochastic Hutchinson error
/// (variance `≈ v·tr` per [`EDOF_TRACE_VARIANCE_PER_UNIT_TRACE`]), so the
/// per-solve bias never dominates — the same tolerance convention the crate's
/// REML trace probes use elsewhere (`gam_linalg::pcg`, ≈1e-6 per-probe residual).
const EDOF_TRACE_CG_RELATIVE_RESIDUAL: f64 = 1.0e-6;

/// Matrix-free effective degrees of freedom `γ = tr(A(A+ρI)⁻¹)` of the shared-ρ
/// code Gram `A = CᵀC`, via Hutchinson stochastic-trace probes solved with the
/// existing sparse normal-equation conjugate-gradient matvec (design gam#2232,
/// Increment 2, plug 4). Never materialises the dense `K×K` Gram: the operator
/// touches only the streamed `diag`/`off` entries, so this scales to `K ≈ 32k`.
///
/// The trace is taken on the COMPLEMENTARY operator
/// `γ = K − tr(ρ(A+ρI)⁻¹)` because `ρ(A+ρI)⁻¹` has spectrum in `(0, 1]` and, at
/// the small linear-block ridge, most of the `K` dof are retained, so the
/// complementary trace is the low-variance quantity to sample (see
/// [`EDOF_TRACE_VARIANCE_PER_UNIT_TRACE`]). Each probe is one CG solve of
/// `(A + ρI) w = z` for a deterministic Rademacher `z`; the estimate is
/// `K − ρ·mean_probe(zᵀw)`, clamped to `[0, K]`.
fn hutchinson_gram_edof(
    diag: &[f64],
    off: &HashMap<(u32, u32), f64>,
    rho: f64,
    k: usize,
) -> f64 {
    if k == 0 {
        return 0.0;
    }

    // Symmetric coupling adjacency (sorted per atom for deterministic matvec),
    // exactly the structure `solve_decoder` builds for the refresh solve.
    let mut neigh: Vec<Vec<(u32, f64)>> = vec![Vec::new(); k];
    for (&(a, b), &val) in off.iter() {
        neigh[a as usize].push((b, val));
        neigh[b as usize].push((a, val));
    }
    for list in neigh.iter_mut() {
        list.sort_by_key(|&(nb, _)| nb);
    }

    // Matrix-free `(A + ρI)·v` over the whole dictionary: `O(K + nnz)`, no dense
    // block. `A + ρI ⪰ ρI ≻ 0`, so every probe solve is SPD.
    let matvec = |v: &[f64]| -> Vec<f64> {
        let mut y = vec![0.0f64; k];
        for a in 0..k {
            let mut acc = (diag[a] + rho) * v[a];
            for &(nb, val) in &neigh[a] {
                acc += val * v[nb as usize];
            }
            y[a] = acc;
        }
        y
    };

    // A-priori Gershgorin condition bound κ̂ ≥ κ(A+ρI): the smallest eigenvalue
    // is at least the ridge floor ρ (M ⪰ ρI), Gershgorin caps the largest. This
    // sets the SAME derived CG iteration cap `⌈½√κ·ln(2√κ/ε)⌉` the refresh solve
    // uses, so a probe solve cannot spin unbounded on a near-singular giant block.
    let mut lambda_max_bound = 0.0f64;
    for a in 0..k {
        let mut off_abs = 0.0f64;
        for &(_, val) in &neigh[a] {
            off_abs += val.abs();
        }
        lambda_max_bound = lambda_max_bound.max(diag[a] + rho + off_abs);
    }
    let lambda_min = rho.max(DEAD_DENOM);
    let kappa_bound = (lambda_max_bound / lambda_min).max(1.0);
    let root = kappa_bound.sqrt();
    let charge_floor = EDOF_TRACE_CG_RELATIVE_RESIDUAL;
    let chebyshev = 0.5 * root * (2.0 * root / charge_floor).ln();
    let finite = k.saturating_mul(2).saturating_add(16);
    let cap = (chebyshev.max(0.0).ceil() as usize)
        .saturating_add(16)
        .min(finite)
        .max(1);

    // Probe count derived from the documented variance target: m = ⌈2/v⌉.
    let m_probes = (2.0 / EDOF_TRACE_VARIANCE_PER_UNIT_TRACE).ceil() as usize;
    let m_probes = m_probes.max(1);

    // Deterministic per-probe Rademacher signs from the crate's canonical
    // `splitmix64` mixer (NO `rand` crate): the seed is a content hash of the
    // Gram diagonal, ρ, and K, so the estimate is reproducible across runs and
    // moves with the problem. Mixing `probe · atom` gives independent ±1 draws.
    let mut base_seed = gam_linalg::utils::splitmix64_hash((k as u64) ^ rho.to_bits());
    base_seed = gam_linalg::utils::splitmix64_hash(base_seed ^ (off.len() as u64).wrapping_add(1));
    for &d in diag.iter() {
        base_seed = gam_linalg::utils::splitmix64_hash(base_seed ^ d.to_bits());
    }

    let mut complementary_trace_acc = 0.0f64;
    for probe in 0..m_probes {
        let probe_salt = gam_linalg::utils::splitmix64_hash(base_seed ^ (probe as u64).wrapping_add(1));
        let mut z = vec![0.0f64; k];
        for (a, zi) in z.iter_mut().enumerate() {
            let h = gam_linalg::utils::splitmix64_hash(probe_salt ^ (a as u64).wrapping_add(1));
            // High bit of a well-mixed hash → an unbiased ±1 Rademacher draw.
            *zi = if h >> 63 == 0 { 1.0 } else { -1.0 };
        }
        let result = cg_solve(&matvec, &z, charge_floor, cap);
        // zᵀ(A+ρI)⁻¹z; ρ·zᵀ(A+ρI)⁻¹z is one sample of tr(ρ(A+ρI)⁻¹).
        let zt_minv_z: f64 = z.iter().zip(result.x.iter()).map(|(zi, wi)| zi * wi).sum();
        complementary_trace_acc += rho * zt_minv_z;
    }
    let complementary_trace = complementary_trace_acc / m_probes as f64;
    (k as f64 - complementary_trace).clamp(0.0, k as f64)
}

/// Assemble ONLY the shared-ρ code Gram `A = CᵀC` (diagonal + strictly-upper
/// couplings) from a fit's stored fixed-width routing — WITHOUT the `K×P`
/// right-hand side `B = CᵀX`. The edof trace `tr(A(A+ρI)⁻¹)` needs `A` alone, so
/// skipping `B` avoids a `K×P` allocation at `K ≈ 32k`. This is exactly the `A`
/// part of [`DecoderNormalEq::accumulate`] (same padding contract: a padded slot
/// carries a zero code and is skipped, and a repeated support index folds into
/// the diagonal), so it matches the `A` the decoder refresh actually solved.
fn code_gram_from_routing(
    indices: ArrayView2<'_, u32>,
    codes: ArrayView2<'_, f32>,
    k: usize,
) -> (Vec<f64>, HashMap<(u32, u32), f64>) {
    let mut diag = vec![0.0f64; k];
    let mut off: HashMap<(u32, u32), f64> = HashMap::new();
    let s = indices.ncols();
    for i in 0..indices.nrows() {
        for a in 0..s {
            let ca = codes[[i, a]] as f64;
            if ca == 0.0 {
                continue;
            }
            let ka = indices[[i, a]];
            diag[ka as usize] += ca * ca;
            for b in (a + 1)..s {
                let cb = codes[[i, b]] as f64;
                if cb == 0.0 {
                    continue;
                }
                let kb = indices[[i, b]];
                if ka == kb {
                    diag[ka as usize] += 2.0 * ca * cb;
                    continue;
                }
                let key = if ka < kb { (ka, kb) } else { (kb, ka) };
                *off.entry(key).or_insert(0.0) += ca * cb;
            }
        }
    }
    (diag, off)
}

/// Reconstruction residual sum of squares `Σ_i ‖x_i − Σ_j c_{ij} d_{a_{ij}}‖²` of
/// a fit's stored routing against its decoder — the `RSS` aggregate the shared-ρ
/// REML fixed point consumes.
fn reconstruction_rss(x: ArrayView2<'_, f32>, fit: &SparseDictFit) -> f64 {
    let p = x.ncols();
    let s = fit.indices.ncols();
    let mut rss = 0.0f64;
    let mut recon = vec![0.0f64; p];
    for i in 0..x.nrows() {
        for r in recon.iter_mut() {
            *r = 0.0;
        }
        for a in 0..s {
            let cj = fit.codes[[i, a]] as f64;
            if cj == 0.0 {
                continue;
            }
            let drow = fit.decoder.row(fit.indices[[i, a]] as usize);
            for (c, r) in recon.iter_mut().enumerate() {
                *r += cj * drow[c] as f64;
            }
        }
        let xi = x.row(i);
        for c in 0..p {
            let d = xi[c] as f64 - recon[c];
            rss += d * d;
        }
    }
    rss
}

/// Pooled aggregates for the linear block's ONE shared REML variance component
/// at ridge `rho` (design gam#2232, Increment 2, plug 4).
///
/// Reconstructs the shared-ρ code Gram `A = CᵀC` from the fit's
/// stored routing and computes the matrix-free effective dof
/// `γ = tr(A(A+ρI)⁻¹)` ([`hutchinson_gram_edof`]) together with the reconstruction
/// `RSS` and the decoder penalty energy `‖D‖²_F` — exactly the aggregates
/// [`linear_shared_rho_fs_step`] consumes. Matrix-free throughout (no dense
/// `K×K`, no `K×P` right-hand side), so it holds at `K ≈ 32k`.
pub fn linear_block_reml_stats(
    x: ArrayView2<'_, f32>,
    fit: &SparseDictFit,
    rho: f64,
) -> LinearBlockRemlStats {
    let k = fit.decoder.nrows();
    let (diag, off) = code_gram_from_routing(fit.indices.view(), fit.codes.view(), k);
    let gram_edof = hutchinson_gram_edof(&diag, &off, rho, k);
    let penalty_energy: f64 = fit.decoder.iter().map(|&d| (d as f64) * (d as f64)).sum();
    let rss = reconstruction_rss(x, fit);
    LinearBlockRemlStats {
        gram_edof,
        p_cols: x.ncols(),
        penalty_energy,
        rss,
        n_obs: x.nrows(),
    }
}

/// Iteration cap for the shared-ρ REML schedule (design gam#2232, Increment 2,
/// plug 4). The Fellner–Schall / MacKay evidence recursion for a single variance
/// component contracts geometrically near its fixed point, and each iteration is
/// a FULL linear-kernel fit, so a small cap bounds the schedule at a few times a
/// single fit while comfortably covering the handful of steps to the stochastic
/// edof floor. A run that has not settled by the cap returns its last fit rather
/// than spinning.
const MAX_REML_SCHEDULE_ITERS: usize = 8;

/// Relative-ρ stopping band for the shared-ρ REML schedule, DERIVED from the
/// edof probe variance target: the Hutchinson estimator resolves a PSD trace to a
/// coefficient of variation of at best `√v` (variance-per-unit-trace `v`, at unit
/// trace; see [`EDOF_TRACE_VARIANCE_PER_UNIT_TRACE`]), so a relative-ρ change
/// below `√v` is already within the stochastic resolution of the edof that drives
/// the update — iterating tighter would chase Monte-Carlo noise. Not a tuned knob:
/// it is `√v` of the one documented quality target.
fn reml_schedule_rho_rel_tol() -> f64 {
    EDOF_TRACE_VARIANCE_PER_UNIT_TRACE.sqrt()
}

/// The shared-ρ REML schedule (design gam#2232, Increment 2, plug 4): the outer
/// evidence loop that SELECTS the ONE shared linear-block ridge instead of taking
/// two magic constants. It alternates a full [`run_linear_fast_kernel`] at the
/// current ρ with one [`linear_shared_rho_fs_step`] Fellner–Schall update built
/// from the matrix-free aggregates [`linear_block_reml_stats`], to the fixed
/// point.
///
/// The initial ρ is the shared default ridge (`config.decoder_ridge`, equal to
/// `config.code_ridge` on the shared-default entry) — the historical magic
/// constant becomes only the WARM START of the evidence loop. Iteration stops
/// when the relative ρ change falls below the solver-derived stochastic floor
/// ([`reml_schedule_rho_rel_tol`]) — at which point the current fit already
/// reflects a ρ within that band, so no redundant refit is issued — or at the
/// documented cap [`MAX_REML_SCHEDULE_ITERS`].
pub fn run_linear_reml_schedule(
    x: ArrayView2<'_, f32>,
    config: &SparseDictConfig,
) -> Result<SparseDictFit, String> {
    // Warm start at the shared-default ridge; from here ρ is REML-selected.
    let mut rho = config.decoder_ridge as f64;
    let mut fit = run_linear_fast_kernel(x, config, rho)?;
    let tol = reml_schedule_rho_rel_tol();

    for iter in 0..MAX_REML_SCHEDULE_ITERS {
        let stats = linear_block_reml_stats(x, &fit, rho);
        let rho_new = linear_shared_rho_fs_step(&stats, rho);
        let rel_change = (rho_new - rho).abs() / rho.abs().max(f64::MIN_POSITIVE);
        // Per-iteration heartbeat on the warn channel (survives RUST_LOG=warn
        // harnesses), at outer-loop cadence only — never per row or minibatch.
        log::warn!(
            "[SAE reml-schedule iter {}/{}] rho={:.6e} rho_new={:.6e} rel_change={:.3e} \
             edof={:.2} rss={:.6e} penalty_energy={:.6e} tol={:.3e}",
            iter + 1,
            MAX_REML_SCHEDULE_ITERS,
            rho,
            rho_new,
            rel_change,
            stats.gram_edof,
            stats.rss,
            stats.penalty_energy,
            tol,
        );
        if rel_change <= tol {
            // The current `fit` was produced at `rho`, which is within `tol` of
            // `rho_new`: it already reflects the fixed point. Stop without a
            // redundant refit.
            break;
        }
        rho = rho_new;
        fit = run_linear_fast_kernel(x, config, rho)?;
    }
    Ok(fit)
}

fn validate(x: ArrayView2<'_, f32>, config: &SparseDictConfig) -> Result<(), String> {
    if x.nrows() == 0 || x.ncols() == 0 {
        return Err("fit_sparse_dictionary requires a non-empty N×P matrix".to_string());
    }
    if !x.iter().all(|v| v.is_finite()) {
        return Err("fit_sparse_dictionary input must be finite".to_string());
    }
    if config.n_atoms == 0 {
        return Err("fit_sparse_dictionary requires K >= 1".to_string());
    }
    if config.active == 0 {
        return Err("fit_sparse_dictionary requires active (top_s) >= 1".to_string());
    }
    if config.max_epochs == 0 {
        return Err("fit_sparse_dictionary requires max_epochs >= 1".to_string());
    }
    if !(config.code_ridge.is_finite() && config.code_ridge >= 0.0) {
        return Err("fit_sparse_dictionary code_ridge must be finite and non-negative".to_string());
    }
    if !(config.decoder_ridge.is_finite() && config.decoder_ridge >= 0.0) {
        return Err(
            "fit_sparse_dictionary decoder_ridge must be finite and non-negative".to_string(),
        );
    }
    if !config.tolerance.is_finite() {
        return Err("fit_sparse_dictionary tolerance must be finite".to_string());
    }
    Ok(())
}

/// Seed atoms with a deterministic k-means++-style farthest-point pass on the
/// rows, so the initial dictionary already spans the data's principal
/// directions (no RNG -> reproducible). For `K > N` the extra atoms wrap.
pub(super) fn seed_decoder(x: ArrayView2<'_, f32>, k: usize) -> Array2<f32> {
    let n = x.nrows();
    let p = x.ncols();
    let mut decoder = Array2::<f32>::zeros((k, p));

    // First atom: the largest-norm row.
    let mut first = 0usize;
    let mut best = f32::NEG_INFINITY;
    for i in 0..n {
        let r = x.row(i);
        let nrm: f32 = r.iter().map(|v| v * v).sum();
        if nrm > best {
            best = nrm;
            first = i;
        }
    }
    decoder.row_mut(0).assign(&x.row(first));

    // Row-parallel distance refresh + reduction. Each row's `min_dist2[i]`
    // update reads only row `i` and the single previous atom, so the parallel
    // pass is elementwise-independent and bit-identical to the serial sweep;
    // the argmax reduction breaks ties toward the LOWER row index (the serial
    // scan's `>` comparison), keeping the seed deterministic. This pass was the
    // measured single-thread wall at dictionary scale (`O(K·N·P)` serial ≈ 2 h
    // at K=32k, N=96k, P=2048 — creditscope #1026); the work is unchanged, only
    // spread across rows.
    let mut min_dist2 = vec![f32::INFINITY; n];
    for atom in 1..k {
        let prev = decoder.row(atom - 1);
        let chosen = if atom < n {
            let (bi, _bv) = min_dist2
                .par_iter_mut()
                .enumerate()
                .map(|(i, md)| {
                    let xi = x.row(i);
                    let mut d2 = 0.0f32;
                    for c in 0..p {
                        let d = xi[c] - prev[c];
                        d2 += d * d;
                    }
                    if d2 < *md {
                        *md = d2;
                    }
                    (i, *md)
                })
                .reduce(
                    || (usize::MAX, f32::NEG_INFINITY),
                    |a, b| {
                        // Strictly-greater wins; on ties keep the lower index —
                        // exactly the serial scan's first-max semantics.
                        if b.1 > a.1 || (b.1 == a.1 && b.0 < a.0) { b } else { a }
                    },
                );
            bi
        } else {
            // K > N wrap: no distance refresh needed, the atom repeats a row.
            atom % n
        };
        decoder.row_mut(atom).assign(&x.row(chosen));
    }
    decoder
}

/// The assembled sparse decoder normal equations `(A + ρI) D = B`, with
/// `A = CᵀC` (`K×K`, symmetric PSD) and `B = CᵀX` (`K×P`), where the code matrix
/// `C` is never materialised. Only atom pairs that co-fire in some row appear in
/// `A`, so the coupling is sparse: `diag` holds `A_kk`, `off` holds the strictly
/// upper-triangular couplings `A_{kl}` (`k < l`), and `b` holds `B`.
pub(super) struct DecoderNormalEq {
    /// `A_kk = Σ_i c_{ik}²`, length `K`.
    pub(super) diag: Vec<f64>,
    /// `B = CᵀX`, `K×P`.
    pub(super) b: Array2<f64>,
    /// Off-diagonal couplings `A_{kl}` keyed by `(k, l)` with `k < l`.
    pub(super) off: HashMap<(u32, u32), f64>,
    /// Non-zero code firings per atom over the accumulated refresh window.
    pub(super) firings: Vec<usize>,
    /// Sum of absolute code amplitudes per atom over the accumulated window.
    pub(super) amplitude_sum: Vec<f64>,
}

impl DecoderNormalEq {
    /// An empty (`A = 0`, `B = 0`) `K×P` system, ready to have shards streamed
    /// into it via [`Self::accumulate`]. Used by the streaming trainer to build
    /// the epoch's normal equations one shard at a time.
    pub(super) fn zeros(k: usize, p: usize) -> Self {
        Self {
            diag: vec![0.0f64; k],
            b: Array2::<f64>::zeros((k, p)),
            off: HashMap::new(),
            firings: vec![0; k],
            amplitude_sum: vec![0.0; k],
        }
    }

    /// Stream one shard's `(x, codes)` into the running normal equations,
    /// adding its `CᵀC` / `CᵀX` contributions. Summing a corpus's shards this
    /// way yields exactly the same `(A, B)` as `assemble_normal_eq` (the
    /// test-only full-batch reference implementation) over the
    /// concatenation (addition is associative; the per-row contributions are
    /// independent), so the streaming decoder refresh equals the full-batch one.
    pub(super) fn accumulate(&mut self, x: ArrayView2<'_, f32>, codes: &[SparseCode]) {
        let p = self.b.ncols();
        for (row_idx, code) in codes.iter().enumerate() {
            let xi = x.row(row_idx);
            let xi_slice = xi.as_slice();
            for a in 0..code.indices.len() {
                let ca = code.codes[a] as f64;
                if ca == 0.0 {
                    continue;
                }
                let ka = code.indices[a];
                self.firings[ka as usize] += 1;
                self.amplitude_sum[ka as usize] += ca.abs();
                self.diag[ka as usize] += ca * ca;
                let brow = ka as usize;
                let mut brow_view = self.b.row_mut(brow);
                match (brow_view.as_slice_mut(), xi_slice) {
                    (Some(bs), Some(xs)) => {
                        for (bref, &xv) in bs.iter_mut().zip(xs.iter()) {
                            *bref += ca * xv as f64;
                        }
                    }
                    _ => {
                        for c in 0..p {
                            brow_view[c] += ca * xi[c] as f64;
                        }
                    }
                }
                for bsel in (a + 1)..code.indices.len() {
                    let cb = code.codes[bsel] as f64;
                    if cb == 0.0 {
                        continue;
                    }
                    let kb = code.indices[bsel];
                    if ka == kb {
                        self.diag[ka as usize] += 2.0 * ca * cb;
                        continue;
                    }
                    let key = if ka < kb { (ka, kb) } else { (kb, ka) };
                    *self.off.entry(key).or_insert(0.0) += ca * cb;
                }
            }
        }
    }

    /// Drop accumulated rows for atoms that just refreshed. Deferred atoms keep
    /// their diagonal/right-hand-side statistics streaming; couplings touching a
    /// refreshed atom are discarded because one endpoint's decoder row changed.
    pub(super) fn clear_refreshed_atoms(&mut self, gate: &[RoutabilityGateDecision]) {
        for decision in gate.iter() {
            if !decision.refresh {
                continue;
            }
            let atom = decision.atom;
            self.diag[atom] = 0.0;
            self.firings[atom] = 0;
            self.amplitude_sum[atom] = 0.0;
            self.b.row_mut(atom).fill(0.0);
        }
        self.off
            .retain(|&(a, b), _| !gate[a as usize].refresh && !gate[b as usize].refresh);
    }
}

/// An atom is "dead" this epoch when its regularised self-energy `A_kk + ρ` is
/// at or below this floor: it never fired (and, since couplings require two
/// non-zero codes, it is then necessarily isolated). Such atoms keep their
/// seeded direction so a later epoch can still route rows to them.
pub(super) const DEAD_DENOM: f64 = 1.0e-12;

/// Percolation-derived size ceiling for the exact dense-Cholesky path.
///
/// The co-firing graph is, at realistic scale, an Erdős–Rényi graph `G(K, p)`:
/// each of the `N` rows lights `s` atoms, depositing `C(s,2)` co-firing edges,
/// so the mean degree is `D = 2|E|/K ≈ N·s²/K`. Erdős–Rényi's theorem places the
/// **giant-component birth exactly at mean degree `D = 1`**, and *at that
/// critical point the largest component has size `Θ(K^{2/3})`*: strictly below
/// criticality every component is smaller, and strictly above it anything of
/// size `≫ K^{2/3}` has been swallowed by the single giant. `K^{2/3}` is thus
/// the intrinsic size scale of the percolation transition — the frontier that
/// separates the genuinely-small sub/critical debris (whose exact dense
/// Cholesky costs at most `O((K^{2/3})³) = O(K²)`, i.e. never more than forming
/// the ambient `K×K` normal equations themselves) from giant-scale blocks, where
/// a per-component dense factorisation is fiction and matrix-free CG is the only
/// honest solve. We therefore route components of size `≤ ⌈K^{2/3}⌉` to dense
/// Cholesky and everything larger to CG. No tuned constant enters: the exponent
/// `2/3` is the Erdős–Rényi critical-component exponent (`θ = 2/3`, a theorem,
/// not a knob), and the threshold is that critical-window component scaling
/// evaluated at the live `K` — it moves with the problem, so there is no magic
/// block size to outgrow.
pub(super) fn direct_solve_size_threshold(k: usize) -> usize {
    if k == 0 {
        return 0;
    }
    // ⌈K^{2/3}⌉: the critical-window largest-component scale. `ceil` keeps the
    // smallest coupled blocks (a single co-firing edge, `K^{2/3} ≥ 1`) on the
    // exact path where dense factorisation is unconditionally cheapest.
    (k as f64).powf(2.0 / 3.0).ceil() as usize
}

/// Solver/percolation certificate for one decoder MOD refresh.
#[derive(Clone, Copy, Debug)]
pub struct DecoderSolveStats {
    /// Mean degree of the co-firing graph, `2|E|/K`.
    pub mean_cofiring_degree: f64,
    /// Largest connected component size divided by `K`.
    pub giant_component_fraction: f64,
    /// Number of connected components in the co-firing graph, including isolated
    /// singleton atoms.
    pub component_count: usize,
    /// Largest connected component size.
    pub max_component_size: usize,
    /// Decoder columns solved by CG.
    pub cg_columns: usize,
    /// Total CG iterations across solved columns.
    pub cg_iterations: usize,
    /// Largest condition estimate recovered from CG's Lanczos tridiagonal.
    pub cg_kappa_hat: Option<f64>,
    /// Largest final relative normal-equation residual among CG solves.
    pub cg_relative_residual: f64,
    /// Relative residual threshold used by CG, derived from the ridge/charge
    /// floor rather than an arbitrary numerical tolerance.
    pub cg_residual_stop: f64,
    /// Decoder columns whose CG did NOT reach the charge floor before the
    /// conditioning-derived iteration cap (or broke down on a non-SPD step).
    /// Non-zero means at least one giant-scale co-firing block was too
    /// ill-conditioned to solve to tolerance and fell back to the ridge-diagonal
    /// best effort — a TYPED non-convergence, surfaced instead of a silent spin.
    pub cg_nonconverged_columns: usize,
    /// Largest a-priori Gershgorin condition-number bound over the CG-solved
    /// components. This is the `κ̂` that sets the derived iteration cap
    /// `⌈½√κ̂·ln(2√κ̂/ε)⌉` before any CG step runs, so an ill-conditioned block is
    /// diagnosable up front (not only after the Lanczos estimate matures).
    pub cg_kappa_bound: Option<f64>,
}

impl Default for DecoderSolveStats {
    fn default() -> Self {
        Self {
            mean_cofiring_degree: 0.0,
            giant_component_fraction: 0.0,
            component_count: 0,
            max_component_size: 0,
            cg_columns: 0,
            cg_iterations: 0,
            cg_kappa_hat: None,
            cg_relative_residual: 0.0,
            cg_residual_stop: 0.0,
            cg_nonconverged_columns: 0,
            cg_kappa_bound: None,
        }
    }
}

impl DecoderSolveStats {
    fn record_cg(&mut self, result: &CgSolveResult) {
        self.cg_columns += 1;
        self.cg_iterations += result.iterations;
        self.cg_relative_residual = self.cg_relative_residual.max(result.relative_residual);
        if result.stop != CgStop::Converged {
            self.cg_nonconverged_columns += 1;
        }
        if let Some(kappa) = result.kappa_hat {
            self.cg_kappa_hat = Some(self.cg_kappa_hat.map_or(kappa, |old| old.max(kappa)));
        }
    }

    fn record_kappa_bound(&mut self, bound: f64) {
        self.cg_kappa_bound = Some(self.cg_kappa_bound.map_or(bound, |old| old.max(bound)));
    }
}

#[derive(Clone, Copy, Debug)]
pub(super) struct RoutabilityGateDecision {
    pub(super) atom: usize,
    pub(super) refresh: bool,
    pub(super) firings: usize,
    pub(super) mean_amplitude: f64,
    pub(super) z_alpha: f64,
    pub(super) margin: f64,
    pub(super) threshold: f64,
    pub(super) standard_error: f64,
}

fn routability_z_alpha(firings: usize) -> f64 {
    // BIC's one-parameter charge is `0.5 ln n`; equating it to a Gaussian
    // tail exponent `z^2/2` gives the confidence radius without a tuned knob.
    (firings.max(2) as f64).ln().sqrt()
}

pub(super) fn routability_gate_decisions(
    eq: &DecoderNormalEq,
    residual_scale: f64,
) -> Vec<RoutabilityGateDecision> {
    (0..eq.diag.len())
        .map(|atom| {
            let firings = eq.firings[atom];
            if firings == 0 || eq.diag[atom] <= DEAD_DENOM {
                return RoutabilityGateDecision {
                    atom,
                    refresh: false,
                    firings,
                    mean_amplitude: 0.0,
                    z_alpha: routability_z_alpha(firings),
                    margin: 0.0,
                    threshold: f64::INFINITY,
                    standard_error: f64::INFINITY,
                };
            }
            let n = firings as f64;
            let mean_amplitude = eq.amplitude_sum[atom] / n;
            let z_alpha = routability_z_alpha(firings);
            let charge_floor = if residual_scale > 0.0 {
                residual_scale * z_alpha / n.sqrt()
            } else {
                0.0
            };
            // The routability margin is the fraction of the mean amplitude that
            // survives the charge floor. A starved atom (mean_amplitude below the
            // floor) has NO surviving margin: clamp at zero so the quantity is
            // `>= 0` by construction and can never enter a downstream expression as
            // a negative shrink. Semantically identical to the previous negative /
            // NEG_INFINITY value — a non-positive margin already forces
            // `threshold = +INF` below, deferring the atom — but it removes the
            // sign hazard entirely: the gate can defer or refresh, never negate.
            let margin = if mean_amplitude > 0.0 {
                (1.0 - charge_floor / mean_amplitude).max(0.0)
            } else {
                0.0
            };
            let standard_error = if residual_scale > 0.0 && mean_amplitude > 0.0 {
                residual_scale / (mean_amplitude * n.sqrt())
            } else if mean_amplitude > 0.0 {
                0.0
            } else {
                f64::INFINITY
            };
            let threshold = if margin > 0.0 && mean_amplitude > 0.0 {
                let denom = mean_amplitude * margin;
                (z_alpha * residual_scale / denom).powi(2)
            } else {
                f64::INFINITY
            };
            RoutabilityGateDecision {
                atom,
                refresh: n >= threshold,
                firings,
                mean_amplitude,
                z_alpha,
                margin,
                threshold,
                standard_error,
            }
        })
        .collect()
}

pub(super) fn solve_decoder_with_routability_gate(
    decoder: &mut Array2<f32>,
    eq: &DecoderNormalEq,
    ridge: f64,
    residual_scale: f64,
) -> (DecoderSolveStats, Vec<RoutabilityGateDecision>) {
    let gate = routability_gate_decisions(eq, residual_scale);
    let mut candidate = decoder.clone();
    let stats = solve_decoder(&mut candidate, eq, ridge);
    for decision in gate.iter() {
        if !decision.refresh {
            // A deferred atom keeps its previous decoder row and accumulates
            // firing evidence across epochs. Surface the routability evidence
            // trail so a persistently-held-back atom is diagnosable without a
            // debugger: `n < threshold` because the mean amplitude cannot yet
            // clear the `z_alpha * residual_scale` charge floor by the required
            // `margin` (see `routability_gate_decisions`).
            log::debug!(
                "[SAE routability] atom {} deferred: firings={} mean_amplitude={:.4} \
                 z_alpha={:.4} margin={:.4} standard_error={:.4} threshold={:.4}",
                decision.atom,
                decision.firings,
                decision.mean_amplitude,
                decision.z_alpha,
                decision.margin,
                decision.standard_error,
                decision.threshold,
            );
            continue;
        }
        let src = candidate.row(decision.atom);
        let mut dst = decoder.row_mut(decision.atom);
        dst.assign(&src);
    }
    (stats, gate)
}

/// Re-seed atoms that fired for no row this epoch (dead atoms) onto the current
/// worst-reconstructed rows' residual directions — the "dead-feature resampling"
/// that lets a large dictionary actually use all `K` atoms (#1026).
///
/// Pointing a fresh atom at the largest reconstruction error is the greedy step
/// that reduces RSS the most; distinct dead atoms take distinct high-residual
/// rows so revived atoms do not duplicate each other. The residual is computed
/// under the current (just-refreshed, unit-normed) decoder and the `codes` that
/// produced this epoch's routing, so it reflects the live model's error. Only the
/// residual *direction* is installed (raw, un-normed); the caller re-runs the
/// unit-norm + sign projection. At most one atom is revived per distinct row per
/// epoch — with more dead atoms than rows the remainder revive on later epochs as
/// the residual field changes, which is the standard bounded-resample cadence.
///
/// Returns the number of atoms revived (0 leaves the decoder untouched, so a
/// fully-alive small-`K` dictionary pays only the usage scan).
fn revive_dead_atoms(
    x: ArrayView2<'_, f32>,
    codes: &[SparseCode],
    decoder: &mut Array2<f32>,
) -> usize {
    let n = x.nrows();
    let p = x.ncols();
    let k = decoder.nrows();

    // Which atoms fired (non-zero code) for at least one row this epoch.
    let mut alive = vec![false; k];
    for code in codes.iter() {
        for (j, &idx) in code.indices.iter().enumerate() {
            if code.codes[j] != 0.0 {
                alive[idx as usize] = true;
            }
        }
    }
    let dead: Vec<usize> = (0..k).filter(|&a| !alive[a]).collect();
    if dead.is_empty() {
        return 0;
    }

    // Per-row residual under the current model, and its squared norm for ranking.
    let mut resid = Array2::<f32>::zeros((n, p));
    let mut resid_norm2 = vec![0.0f64; n];
    for i in 0..n {
        let xi = x.row(i);
        let mut ri = resid.row_mut(i);
        for c in 0..p {
            ri[c] = xi[c];
        }
        let code = &codes[i];
        for j in 0..code.indices.len() {
            let cj = code.codes[j];
            if cj == 0.0 {
                continue;
            }
            let drow = decoder.row(code.indices[j] as usize);
            for c in 0..p {
                ri[c] -= cj * drow[c];
            }
        }
        let mut acc = 0.0f64;
        for c in 0..p {
            acc += ri[c] as f64 * ri[c] as f64;
        }
        resid_norm2[i] = acc;
    }

    // Rows ranked by descending residual energy (ties by ascending index →
    // deterministic). Only rows with real residual can seed a useful atom.
    let mut order: Vec<usize> = (0..n).collect();
    order.sort_by(|&a, &b| {
        resid_norm2[b]
            .partial_cmp(&resid_norm2[a])
            .unwrap_or(std::cmp::Ordering::Equal)
            .then_with(|| a.cmp(&b))
    });

    let mut revived = 0usize;
    for (t, &atom) in dead.iter().enumerate() {
        if t >= n {
            break; // one atom per distinct row this epoch
        }
        let row = order[t];
        if resid_norm2[row] <= (DEAD_DENOM as f64) {
            break; // remaining rows are already reconstructed — nothing to seed
        }
        let src = resid.row(row);
        let mut dst = decoder.row_mut(atom);
        for c in 0..p {
            dst[c] = src[c];
        }
        revived += 1;
    }
    revived
}

/// Solve `(A + ρI) D = B` exactly, writing the solved rows into `decoder`.
///
/// Atoms are walked in ascending index order and grouped into connected
/// components via BFS over the symmetric coupling adjacency; each component is
/// sorted (canonical order) before solving so the result is bit-reproducible
/// regardless of `HashMap` iteration order. Dead atoms ([`DEAD_DENOM`]) and
/// atoms with no co-firing partner keep / take the trivial solve.
pub(super) fn solve_decoder(
    decoder: &mut Array2<f32>,
    eq: &DecoderNormalEq,
    ridge: f64,
) -> DecoderSolveStats {
    let k = eq.diag.len();
    let p = eq.b.ncols();

    // Symmetric coupling adjacency, sorted per atom for deterministic assembly.
    let mut neigh: Vec<Vec<(u32, f64)>> = vec![Vec::new(); k];
    for (&(a, b), &val) in eq.off.iter() {
        neigh[a as usize].push((b, val));
        neigh[b as usize].push((a, val));
    }
    for list in neigh.iter_mut() {
        list.sort_by_key(|&(nb, _)| nb);
    }

    let mut stats = DecoderSolveStats {
        mean_cofiring_degree: if k == 0 {
            0.0
        } else {
            2.0 * eq.off.len() as f64 / k as f64
        },
        cg_residual_stop: ridge.max(DEAD_DENOM),
        ..DecoderSolveStats::default()
    };

    // Exact dense Cholesky is confined to components below the percolation
    // critical-component scale; everything larger is a giant-scale block solved
    // matrix-free by CG (see `direct_solve_size_threshold`).
    let direct_threshold = direct_solve_size_threshold(k);

    let mut visited = vec![false; k];
    for start in 0..k {
        if visited[start] {
            continue;
        }
        if neigh[start].is_empty() {
            // Isolated atom: diagonal (singleton) solve, exact in one shot.
            visited[start] = true;
            stats.component_count += 1;
            stats.max_component_size = stats.max_component_size.max(1);
            let denom = eq.diag[start] + ridge;
            if denom <= DEAD_DENOM {
                // Dead atom: keep its seeded direction (no permanent collapse).
                continue;
            }
            for c in 0..p {
                decoder[[start, c]] = (eq.b[[start, c]] / denom) as f32;
            }
            continue;
        }
        // Gather the whole connected component by BFS, then canonicalise order.
        let mut comp = vec![start];
        visited[start] = true;
        let mut head = 0usize;
        while head < comp.len() {
            let node = comp[head];
            head += 1;
            for &(nb, _) in &neigh[node] {
                let nb = nb as usize;
                if !visited[nb] {
                    visited[nb] = true;
                    comp.push(nb);
                }
            }
        }
        comp.sort_unstable();
        stats.component_count += 1;
        stats.max_component_size = stats.max_component_size.max(comp.len());
        solve_component(
            decoder,
            eq,
            ridge,
            &comp,
            &neigh,
            p,
            direct_threshold,
            &mut stats,
        );
    }
    if k > 0 {
        stats.giant_component_fraction = stats.max_component_size as f64 / k as f64;
    }

    // Percolation + conditioning certificate for this refresh. Surfacing the
    // giant-component fraction, mean degree, and the CG Lanczos κ̂ every epoch
    // makes the percolating-regime diagnosis (and any ill-conditioned block)
    // readable without a debugger — the co-firing graph is one giant component
    // at scale, so the exact-solve threshold `⌈K^{2/3}⌉` is expected to bind.
    log::debug!(
        "[SAE percolation] K={k} mean_degree={:.4} giant_fraction={:.4} \
         components={} max_component={} direct_threshold={direct_threshold} \
         cg_columns={} cg_iterations={} cg_kappa_hat={:?} cg_kappa_bound={:?} \
         cg_nonconverged_columns={} cg_relative_residual={:.3e} cg_residual_stop={:.3e}",
        stats.mean_cofiring_degree,
        stats.giant_component_fraction,
        stats.component_count,
        stats.max_component_size,
        stats.cg_columns,
        stats.cg_iterations,
        stats.cg_kappa_hat,
        stats.cg_kappa_bound,
        stats.cg_nonconverged_columns,
        stats.cg_relative_residual,
        stats.cg_residual_stop,
    );
    stats
}

/// Solve one connected component's block: dense SPD Cholesky when the block is
/// below the percolation critical-component scale (`direct_threshold`, see
/// [`direct_solve_size_threshold`]), else matrix-free CG. `comp` is the
/// component's atom indices in ascending order; `neigh` is the global sorted
/// adjacency.
fn solve_component(
    decoder: &mut Array2<f32>,
    eq: &DecoderNormalEq,
    ridge: f64,
    comp: &[usize],
    neigh: &[Vec<(u32, f64)>],
    p: usize,
    direct_threshold: usize,
    stats: &mut DecoderSolveStats,
) {
    let m = comp.len();
    // Local atom -> block-row index map (comp is sorted, so this is canonical).
    let mut local: HashMap<usize, usize> = HashMap::with_capacity(m);
    for (i, &a) in comp.iter().enumerate() {
        local.insert(a, i);
    }

    if m <= direct_threshold {
        // Assemble the dense block (A_sub + ρI) and the m×P right-hand side, then
        // solve all P columns from one Cholesky factor.
        let mut mat = Array2::<f64>::zeros((m, m));
        let mut rhs = Array2::<f64>::zeros((m, p));
        for (i, &a) in comp.iter().enumerate() {
            mat[[i, i]] = eq.diag[a] + ridge;
            for &(nb, val) in &neigh[a] {
                if let Some(&j) = local.get(&(nb as usize)) {
                    mat[[i, j]] = val;
                }
            }
            for c in 0..p {
                rhs[[i, c]] = eq.b[[a, c]];
            }
        }
        let sol = cholesky_solve_block(&mat, &rhs);
        for (i, &a) in comp.iter().enumerate() {
            for c in 0..p {
                decoder[[a, c]] = sol[[i, c]] as f32;
            }
        }
        return;
    }

    // Default coupled path: solve each column by matrix-free CG. The operator is
    // the component-restricted symmetric mat-vec and touches only stored sparse
    // co-firing entries.
    let matvec = |xloc: &[f64]| -> Vec<f64> {
        let mut y = vec![0.0f64; m];
        for (i, &a) in comp.iter().enumerate() {
            let mut acc = (eq.diag[a] + ridge) * xloc[i];
            for &(nb, val) in &neigh[a] {
                if let Some(&j) = local.get(&(nb as usize)) {
                    acc += val * xloc[j];
                }
            }
            y[i] = acc;
        }
        y
    };
    let charge_floor = ridge.max(DEAD_DENOM);

    // A-priori spectral bounds of the component operator M = A_sub + ρI via
    // Gershgorin discs over the stored (in-component) co-firing entries. M is SPD
    // with M ⪰ ρI, so the true smallest eigenvalue is at least the regularisation
    // floor; Gershgorin caps the largest. Their ratio is a rigorous condition
    // bound κ̂ ≥ κ(M) that sets a DERIVED iteration cap, so CG cannot spin
    // unbounded on a near-singular giant block (near-duplicate atoms in an
    // overcomplete dictionary over a low-dim post-peel space).
    let mut lambda_max_bound = 0.0f64;
    let mut lambda_min_bound = f64::INFINITY;
    for &a in comp {
        let mut off_abs = 0.0f64;
        for &(nb, val) in &neigh[a] {
            if local.contains_key(&(nb as usize)) {
                off_abs += val.abs();
            }
        }
        let center = eq.diag[a] + ridge;
        lambda_max_bound = lambda_max_bound.max(center + off_abs);
        lambda_min_bound = lambda_min_bound.min(center - off_abs);
    }
    let lambda_min = lambda_min_bound.max(ridge).max(DEAD_DENOM);
    let kappa_bound = (lambda_max_bound / lambda_min).max(1.0);
    stats.record_kappa_bound(kappa_bound);
    let root = kappa_bound.sqrt();
    // ⌈½√κ·ln(2√κ/ε)⌉: CG's Chebyshev bound on the steps to reach relative 2-norm
    // residual ε = charge_floor. The √κ inside the log is the A-norm→2-norm
    // residual correction, making this a genuine UPPER bound on the iterations
    // needed — a well-conditioned block still converges well inside it (no early
    // cut, since κ̂ ≥ κ), while a giant near-singular block is bounded instead of
    // spinning. Exact CG also terminates in ≤ m steps; keep the historical +16
    // round-off allowance and never exceed that finite-termination bound.
    let chebyshev = 0.5 * root * (2.0 * root / charge_floor).ln();
    let finite = m.saturating_mul(2).saturating_add(16);
    let round_off_grace = 16usize;
    let cap = (chebyshev.max(0.0).ceil() as usize)
        .saturating_add(round_off_grace)
        .min(finite)
        .max(1);

    let mut component_failed = false;
    for c in 0..p {
        let mut bvec = vec![0.0f64; m];
        let mut bnorm2 = 0.0f64;
        for (i, &a) in comp.iter().enumerate() {
            bvec[i] = eq.b[[a, c]];
            bnorm2 += bvec[i] * bvec[i];
        }
        if bnorm2.sqrt() <= DEAD_DENOM {
            for &a in comp {
                decoder[[a, c]] = 0.0;
            }
            continue;
        }
        if component_failed {
            // A prior column already proved this block un-solvable to tolerance;
            // do not grind CG on every remaining column — take the ridge-diagonal
            // best effort (exact if the block were decoupled) so the epoch still
            // makes progress rather than repeating the same non-converging spin.
            write_diagonal_column(decoder, eq, ridge, comp, c);
            continue;
        }
        let result = cg_solve(&matvec, &bvec, charge_floor, cap);
        stats.record_cg(&result);
        if result.stop == CgStop::Converged {
            for (i, &a) in comp.iter().enumerate() {
                decoder[[a, c]] = result.x[i] as f32;
            }
        } else {
            // TYPED non-convergence: the derived cap was hit or CG broke down on a
            // non-SPD step for a near-singular block. Surface it loudly (once per
            // component) and fall back to the ridge-diagonal solve for this and all
            // remaining columns instead of a silent unbounded spin.
            component_failed = true;
            log::warn!(
                "[SAE CG] component size={m} did not converge: stop={:?} iters={} \
                 rel_residual={:.3e} charge_floor={:.3e} kappa_bound={:.3e} cap={cap}; \
                 falling back to ridge-diagonal for remaining columns",
                result.stop,
                result.iterations,
                result.relative_residual,
                charge_floor,
                kappa_bound,
            );
            write_diagonal_column(decoder, eq, ridge, comp, c);
        }
    }
}

/// Ridge-diagonal best-effort solve of one decoder column over a component: the
/// exact solve if the block were decoupled (`d_a = b_a / (A_aa + ρ)`), used as
/// the honest fallback once matrix-free CG is known not to reach the charge floor
/// on an ill-conditioned block.
fn write_diagonal_column(
    decoder: &mut Array2<f32>,
    eq: &DecoderNormalEq,
    ridge: f64,
    comp: &[usize],
    c: usize,
) {
    for &a in comp {
        let denom = eq.diag[a] + ridge;
        decoder[[a, c]] = if denom > DEAD_DENOM {
            (eq.b[[a, c]] / denom) as f32
        } else {
            0.0
        };
    }
}

/// Why a CG solve returned. Only [`CgStop::Converged`] means the column reached
/// the charge floor; the others are TYPED non-convergence the caller surfaces and
/// falls back on rather than spinning.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum CgStop {
    /// Relative normal-equation residual fell to/below the charge floor.
    Converged,
    /// A non-SPD / non-finite curvature step (`pᵀAp ≤ 0`) or a non-finite `β`.
    Breakdown,
    /// The conditioning-derived iteration cap was hit before the charge floor.
    CapReached,
}

struct CgSolveResult {
    x: Vec<f64>,
    iterations: usize,
    relative_residual: f64,
    kappa_hat: Option<f64>,
    stop: CgStop,
}

fn cg_solve<F>(matvec: &F, b: &[f64], charge_floor: f64, cap: usize) -> CgSolveResult
where
    F: Fn(&[f64]) -> Vec<f64>,
{
    use gam_linalg::pcg::{DotReduction, PcgStop, pcg_core};

    let n = b.len();
    let bnorm = b.iter().map(|v| v * v).sum::<f64>().sqrt();
    // Preserve the historical near-zero-rhs short-circuit: a right-hand side at
    // or below the dead-denominator floor carries no informative solution, so
    // return the zero iterate as converged rather than iterating on noise.
    // (`pcg_core`'s own early-out fires only for an EXACT-zero rhs; retaining
    // this keeps the whole `‖b‖ ≤ DEAD_DENOM` band byte-identical to the prior
    // hand-rolled loop.)
    if bnorm <= DEAD_DENOM {
        return CgSolveResult {
            x: vec![0.0; n],
            iterations: 0,
            relative_residual: 0.0,
            kappa_hat: None,
            stop: CgStop::Converged,
        };
    }

    // Delegate the CG recurrence to the shared `gam_linalg::pcg` core — the
    // single source of truth that exists precisely to end hand-rolled CG drift.
    // This path is unpreconditioned (all-ones Jacobi diagonal), uses the
    // bit-reproducible serial reduction, and disables residual refresh, which
    // reproduces the prior loop's pure-recurrence residual exactly. The
    // per-iteration alpha/beta trace is requested so the Lanczos condition
    // estimate `kappa_hat` is reconstructed unchanged on the converged and
    // cap-reached paths that feed the derived iteration cap downstream.
    let rhs = ndarray::Array1::from_vec(b.to_vec());
    let precond = ndarray::Array1::<f64>::from_elem(n, 1.0);
    let mut solution = ndarray::Array1::<f64>::zeros(n);
    let apply = |v: &ndarray::Array1<f64>, out: &mut ndarray::Array1<f64>| {
        let av = matvec(v.as_slice().expect("pcg direction vector is contiguous"));
        out.assign(&ndarray::Array1::from_vec(av));
    };
    let result = pcg_core(
        apply,
        &rhs.view(),
        &precond.view(),
        charge_floor,
        cap,
        0,
        true,
        DotReduction::Serial,
        &mut solution.view_mut(),
    );

    let relative_residual = if result.rhs_norm > 0.0 {
        result.final_residual_norm / result.rhs_norm
    } else {
        0.0
    };
    let kappa_hat = result
        .diagnostics
        .as_ref()
        .and_then(|d| kappa_from_cg_tridiagonal(&d.alpha, &d.beta));
    let stop = match result.stop {
        PcgStop::Converged => CgStop::Converged,
        PcgStop::MaxIters => CgStop::CapReached,
        PcgStop::Breakdown | PcgStop::BadPreconditioner => CgStop::Breakdown,
    };

    CgSolveResult {
        x: solution.to_vec(),
        iterations: result.iterations,
        relative_residual,
        kappa_hat,
        stop,
    }
}

fn kappa_from_cg_tridiagonal(alphas: &[f64], betas: &[f64]) -> Option<f64> {
    use faer::Side;
    use gam_linalg::faer_ndarray::FaerEigh;

    let n = alphas.len();
    if n == 0 {
        return None;
    }
    let mut tri = Array2::<f64>::zeros((n, n));
    for i in 0..n {
        let mut diag = 1.0 / alphas[i];
        if i > 0 {
            diag += betas[i - 1] / alphas[i - 1];
            let off = betas[i - 1].sqrt() / alphas[i - 1];
            tri[[i - 1, i]] = off;
            tri[[i, i - 1]] = off;
        }
        tri[[i, i]] = diag;
    }
    let Ok((evals, _evecs)) = tri.eigh(Side::Lower) else {
        return None;
    };
    let mut min_eval = f64::INFINITY;
    let mut max_eval = 0.0f64;
    for &eval in evals.iter() {
        if eval.is_finite() && eval > 0.0 {
            min_eval = min_eval.min(eval);
            max_eval = max_eval.max(eval);
        }
    }
    if min_eval.is_finite() && max_eval >= min_eval {
        Some(max_eval / min_eval)
    } else {
        None
    }
}

/// Dense SPD solve `mat · X = rhs` (multiple RHS columns) via Cholesky, with the
/// same tiny-ridge bump fallback as the per-row code solve for near-singular /
/// exactly-collinear blocks, and a diagonal last resort.
fn cholesky_solve_block(mat: &Array2<f64>, rhs: &Array2<f64>) -> Array2<f64> {
    use faer::Side;
    use gam_linalg::faer_ndarray::FaerCholesky;

    let m = mat.nrows();
    let mut a = mat.clone();
    let mut bump = 0.0f64;
    for _attempt in 0..6 {
        if let Ok(factor) = a.cholesky(Side::Lower) {
            return factor.solve_mat(rhs);
        }
        bump = if bump == 0.0 { 1.0e-8 } else { bump * 16.0 };
        a = mat.clone();
        for i in 0..m {
            a[[i, i]] += bump;
        }
    }
    // Degenerate beyond recovery: per-row diagonal solve (independent atoms).
    let p = rhs.ncols();
    let mut out = Array2::<f64>::zeros((m, p));
    for i in 0..m {
        let d = mat[[i, i]].max(DEAD_DENOM);
        for c in 0..p {
            out[[i, c]] = rhs[[i, c]] / d;
        }
    }
    out
}

pub(super) fn unit_norm_rows(decoder: &mut Array2<f32>) {
    for mut row in decoder.outer_iter_mut() {
        let nrm: f32 = row.iter().map(|v| v * v).sum::<f32>().sqrt();
        if nrm > 1.0e-12 {
            row.mapv_inplace(|v| v / nrm);
            // Orient by first significant component for a stable sign.
            let mut sign = 1.0f32;
            for &v in row.iter() {
                if v.abs() > 1.0e-9 {
                    sign = v.signum();
                    break;
                }
            }
            if sign < 0.0 {
                row.mapv_inplace(|v| -v);
            }
        }
    }
}

fn explained_variance(
    x: ArrayView2<'_, f32>,
    codes: &[SparseCode],
    decoder: ArrayView2<'_, f32>,
) -> f64 {
    let n = x.nrows();
    let p = x.ncols();
    // Column means for TSS.
    let mut means = vec![0.0f64; p];
    for i in 0..n {
        let xi = x.row(i);
        for c in 0..p {
            means[c] += xi[c] as f64;
        }
    }
    for c in 0..p {
        means[c] /= n as f64;
    }

    let mut rss = 0.0f64;
    let mut tss = 0.0f64;
    let mut recon = vec![0.0f64; p];
    for i in 0..n {
        for c in 0..p {
            recon[c] = 0.0;
        }
        let code = &codes[i];
        for j in 0..code.indices.len() {
            let cj = code.codes[j] as f64;
            if cj == 0.0 {
                continue;
            }
            let drow = decoder.row(code.indices[j] as usize);
            for c in 0..p {
                recon[c] += cj * drow[c] as f64;
            }
        }
        let xi = x.row(i);
        for c in 0..p {
            let r = xi[c] as f64 - recon[c];
            rss += r * r;
            let t = xi[c] as f64 - means[c];
            tss += t * t;
        }
    }
    if tss <= 1.0e-24 {
        if rss <= 1.0e-24 { 1.0 } else { 0.0 }
    } else {
        1.0 - rss / tss
    }
}

fn residual_scale(
    x: ArrayView2<'_, f32>,
    codes: &[SparseCode],
    decoder: ArrayView2<'_, f32>,
) -> f64 {
    let n = x.nrows();
    let p = x.ncols();
    let mut rss = 0.0f64;
    let mut recon = vec![0.0f64; p];
    for i in 0..n {
        for c in 0..p {
            recon[c] = 0.0;
        }
        let code = &codes[i];
        for j in 0..code.indices.len() {
            let cj = code.codes[j] as f64;
            if cj == 0.0 {
                continue;
            }
            let drow = decoder.row(code.indices[j] as usize);
            for c in 0..p {
                recon[c] += cj * drow[c] as f64;
            }
        }
        let xi = x.row(i);
        for c in 0..p {
            let r = xi[c] as f64 - recon[c];
            rss += r * r;
        }
    }
    (rss / (n * p) as f64).sqrt()
}

fn pack_codes(codes: &[SparseCode], n: usize, s: usize) -> (Array2<u32>, Array2<f32>) {
    let mut indices = Array2::<u32>::zeros((n, s));
    let mut code_mat = Array2::<f32>::zeros((n, s));
    for (i, code) in codes.iter().enumerate() {
        for j in 0..s {
            indices[[i, j]] = code.indices[j];
            code_mat[[i, j]] = code.codes[j];
        }
    }
    (indices, code_mat)
}

#[cfg(test)]
mod exact_solve_tests {
    use super::{
        CgStop, DecoderNormalEq, cg_solve, explained_variance, route_and_code_all, solve_decoder,
        solve_decoder_with_routability_gate,
    };
    use crate::sparse_dict::codes::SparseCode;
    use crate::sparse_dict::scoring::TileScorer;
    use crate::sparse_dict::{SparseDictConfig, fit_sparse_dictionary};
    use ndarray::{Array2, ArrayView2};
    use std::collections::HashMap;

    /// Full-batch reference assembly of the sparse decoder normal equations
    /// `(A + ρI) D = B` from the fixed codes/supports (`ρ` is applied at solve
    /// time, so this returns the bare `A`/`B`). Kept only as an independent
    /// oracle for the streaming [`DecoderNormalEq::accumulate`] path that
    /// production uses — summing a corpus's shards through `accumulate` must
    /// yield exactly this batch `(A, B)`.
    fn assemble_normal_eq(
        x: ArrayView2<'_, f32>,
        codes: &[SparseCode],
        k: usize,
        p: usize,
    ) -> DecoderNormalEq {
        let mut diag = vec![0.0f64; k];
        let mut b = Array2::<f64>::zeros((k, p));
        let mut off: HashMap<(u32, u32), f64> = HashMap::new();
        let mut firings = vec![0usize; k];
        let mut amplitude_sum = vec![0.0f64; k];

        for (row_idx, code) in codes.iter().enumerate() {
            let xi = x.row(row_idx);
            let xi_slice = xi.as_slice();
            for a in 0..code.indices.len() {
                let ca = code.codes[a] as f64;
                if ca == 0.0 {
                    continue;
                }
                let ka = code.indices[a];
                firings[ka as usize] += 1;
                amplitude_sum[ka as usize] += ca.abs();
                diag[ka as usize] += ca * ca;
                let brow = ka as usize;
                let mut brow_view = b.row_mut(brow);
                match (brow_view.as_slice_mut(), xi_slice) {
                    (Some(bs), Some(xs)) => {
                        for (bref, &xv) in bs.iter_mut().zip(xs.iter()) {
                            *bref += ca * xv as f64;
                        }
                    }
                    _ => {
                        for c in 0..p {
                            brow_view[c] += ca * xi[c] as f64;
                        }
                    }
                }
                for bsel in (a + 1)..code.indices.len() {
                    let cb = code.codes[bsel] as f64;
                    if cb == 0.0 {
                        continue;
                    }
                    let kb = code.indices[bsel];
                    if ka == kb {
                        // Same atom appearing twice (padding) — fold into diagonal.
                        diag[ka as usize] += 2.0 * ca * cb;
                        continue;
                    }
                    let key = if ka < kb { (ka, kb) } else { (kb, ka) };
                    *off.entry(key).or_insert(0.0) += ca * cb;
                }
            }
        }

        DecoderNormalEq {
            diag,
            b,
            off,
            firings,
            amplitude_sum,
        }
    }

    impl DecoderNormalEq {
        /// Symmetric sparse mat-vec `y = (A + ρI) x` for one decoder column `x`
        /// (length `K`). Whole-system form used by the exactness tests to measure
        /// the normal-equation residual (the block solver uses a
        /// component-restricted variant inline). Touches only the non-zero
        /// couplings, so it is `O(K + nnz)` and never forms a dense `K×K` matrix.
        fn matvec_col(&self, ridge: f64, x: &[f64]) -> Vec<f64> {
            let k = self.diag.len();
            let mut y = vec![0.0f64; k];
            for i in 0..k {
                y[i] = (self.diag[i] + ridge) * x[i];
            }
            for (&(a, b), &val) in self.off.iter() {
                y[a as usize] += val * x[b as usize];
                y[b as usize] += val * x[a as usize];
            }
            y
        }
    }

    /// A small synthetic decoder-update problem with OVERLAPPING supports (`s = 3`):
    /// five codes whose atom sets slide around the 5-atom dictionary so every atom
    /// fires and many atom pairs co-fire — i.e. the coupled `s > 1` regime, not the
    /// decoupled diagonal one. Returns `(x, codes, k, p)`.
    fn overlapping_problem() -> (Array2<f32>, Vec<SparseCode>, usize, usize) {
        let k = 5usize;
        let p = 4usize;
        // Overlapping 3-atom supports (a sliding window) with generic codes.
        let supports: [[u32; 3]; 5] = [[0, 1, 2], [1, 2, 3], [2, 3, 4], [3, 4, 0], [4, 0, 1]];
        let codevals: [[f32; 3]; 5] = [
            [1.0, 0.5, -0.3],
            [0.7, -0.2, 0.4],
            [-0.6, 0.9, 0.1],
            [0.3, -0.5, 0.8],
            [0.2, 0.6, -0.4],
        ];
        let codes: Vec<SparseCode> = supports
            .iter()
            .zip(codevals.iter())
            .map(|(idx, cv)| SparseCode {
                indices: idx.to_vec(),
                codes: cv.to_vec(),
            })
            .collect();
        let n = codes.len();
        let mut x = Array2::<f32>::zeros((n, p));
        for i in 0..n {
            for c in 0..p {
                x[[i, c]] = (((i * 7 + c * 3 + 1) % 13) as f32 - 6.0) / 4.0;
            }
        }
        (x, codes, k, p)
    }

    fn accumulate_constant_rows(
        eq: &mut DecoderNormalEq,
        atom: u32,
        rows: usize,
        code: f32,
        row: [f32; 2],
    ) {
        let mut x = Array2::<f32>::zeros((rows, 2));
        for i in 0..rows {
            x[[i, 0]] = row[0];
            x[[i, 1]] = row[1];
        }
        let codes: Vec<SparseCode> = (0..rows)
            .map(|_| SparseCode {
                indices: vec![atom],
                codes: vec![code],
            })
            .collect();
        eq.accumulate(x.view(), &codes);
    }

    /// Relative normal-equation residual `‖(A+ρI)D − B‖_F / ‖B‖_F`, summed over all
    /// decoder columns, using the same sparse operator the solver uses.
    fn normal_eq_residual(eq: &DecoderNormalEq, decoder: &Array2<f32>, ridge: f64) -> f64 {
        let k = eq.diag.len();
        let p = eq.b.ncols();
        let mut rss = 0.0f64;
        let mut bss = 0.0f64;
        for c in 0..p {
            let dcol: Vec<f64> = (0..k).map(|i| decoder[[i, c]] as f64).collect();
            let y = eq.matvec_col(ridge, &dcol);
            for i in 0..k {
                let r = y[i] - eq.b[[i, c]];
                rss += r * r;
                bss += eq.b[[i, c]] * eq.b[[i, c]];
            }
        }
        if bss <= 0.0 { 0.0 } else { (rss / bss).sqrt() }
    }

    #[test]
    fn routability_gate_refreshes_well_fired_and_defers_starved_atom() {
        let mut eq = DecoderNormalEq::zeros(2, 2);
        accumulate_constant_rows(&mut eq, 0, 64, 1.0, [2.0, 0.0]);
        accumulate_constant_rows(&mut eq, 1, 1, 1.0, [0.0, 3.0]);

        let mut decoder = Array2::<f32>::zeros((2, 2));
        decoder[[0, 1]] = 1.0;
        decoder[[1, 0]] = 1.0;
        let (_stats, gate) = solve_decoder_with_routability_gate(&mut decoder, &eq, 0.0, 1.0);

        assert!(gate[0].refresh, "well-fired atom must refresh");
        assert!(
            gate[0].standard_error <= gate[0].margin,
            "well-fired atom should clear the SE-to-margin gate"
        );
        assert!(!gate[1].refresh, "starved atom must defer");
        assert!(
            gate[1].standard_error > gate[1].margin,
            "starved atom's refresh SE should exceed its charge-floor margin"
        );
        assert!(
            decoder[[0, 0]] > 1.9 && decoder[[0, 1]].abs() < 1.0e-6,
            "admitted atom should take its MOD row"
        );
        assert!(
            decoder[[1, 0]] > 0.9 && decoder[[1, 1]].abs() < 1.0e-6,
            "deferred atom should keep its previous row"
        );
    }

    #[test]
    fn deferred_atom_accumulates_until_routability_threshold_crosses() {
        let mut eq = DecoderNormalEq::zeros(1, 2);
        let mut decoder = Array2::<f32>::zeros((1, 2));
        decoder[[0, 1]] = 1.0;

        accumulate_constant_rows(&mut eq, 0, 1, 1.0, [3.0, 0.0]);
        let (_stats_first, first_gate) =
            solve_decoder_with_routability_gate(&mut decoder, &eq, 0.0, 1.0);
        eq.clear_refreshed_atoms(&first_gate);

        assert!(!first_gate[0].refresh, "single firing should defer");
        assert_eq!(
            eq.firings[0], 1,
            "deferred atom's firing evidence must remain accumulated"
        );
        assert!(
            decoder[[0, 1]] > 0.9,
            "deferred atom must keep its old decoder direction"
        );

        accumulate_constant_rows(&mut eq, 0, 63, 1.0, [3.0, 0.0]);
        let (_stats_second, second_gate) =
            solve_decoder_with_routability_gate(&mut decoder, &eq, 0.0, 1.0);
        eq.clear_refreshed_atoms(&second_gate);

        assert!(
            second_gate[0].refresh,
            "accumulated firings should cross the routability threshold"
        );
        assert_eq!(
            eq.firings[0], 0,
            "refreshed atom's consumed evidence should be cleared"
        );
        assert!(
            decoder[[0, 0]] > 2.9 && decoder[[0, 1]].abs() < 1.0e-6,
            "eventually admitted atom should install its MOD row"
        );
    }

    fn connected_tridiagonal_eq(k: usize, p: usize) -> DecoderNormalEq {
        let mut diag = vec![0.0f64; k];
        for (i, d) in diag.iter_mut().enumerate() {
            *d = 1.8 + 0.03 * i as f64;
        }
        let mut off = std::collections::HashMap::new();
        for i in 0..(k - 1) {
            off.insert((i as u32, (i + 1) as u32), -0.25);
        }
        let mut b = Array2::<f64>::zeros((k, p));
        for i in 0..k {
            for c in 0..p {
                b[[i, c]] = ((i * 5 + c * 7 + 3) % 17) as f64 / 11.0 - 0.6;
            }
        }
        DecoderNormalEq {
            diag,
            b,
            off,
            firings: vec![4; k],
            amplitude_sum: vec![4.0; k],
        }
    }

    #[test]
    fn exact_solver_drives_normal_eq_residual_below_tolerance() {
        // The decoder update must solve the coupled normal equations EXACTLY (to
        // tolerance) for s > 1 / overlapping supports — not approximate them with a
        // fixed number of sweeps.
        let (x, codes, k, p) = overlapping_problem();
        let ridge = 1.0e-6f64;
        let eq = assemble_normal_eq(x.view(), &codes, k, p);
        // Guard: the supports really do couple atoms (we are exercising the coupled
        // path, not a disguised diagonal solve).
        assert!(
            !eq.off.is_empty(),
            "test problem must have off-diagonal coupling (overlapping supports)"
        );

        let mut decoder = Array2::<f32>::zeros((k, p));
        solve_decoder(&mut decoder, &eq, ridge);

        // The internal solve is f64 (Cholesky residual ~1e-15), but the returned
        // decoder is f32, so the measurable relative residual bottoms out at the f32
        // floor (~1e-7). Asserting < 1e-6 proves the update CONVERGED to f32 precision
        // — it is not a fixed sweep-count approximation — without chasing a tolerance
        // f32 cannot represent.
        let rel = normal_eq_residual(&eq, &decoder, ridge);
        assert!(
            rel < 1.0e-6,
            "coupled decoder solve must drive ‖(A+ρI)D−B‖/‖B‖ to the f32 floor \
             (< 1e-6), got {rel}"
        );
    }

    #[test]
    fn block_solve_matches_independent_dense_solve() {
        // Exactness cross-check: the connected-component block solve must agree with
        // a single dense Cholesky of the WHOLE assembled (A+ρI) system. (Equivalently,
        // the result has converged — there is no sweep cap that, if raised, would
        // move it.)
        use faer::Side;
        use gam_linalg::faer_ndarray::FaerCholesky;

        let (x, codes, k, p) = overlapping_problem();
        let ridge = 1.0e-6f64;
        let eq = assemble_normal_eq(x.view(), &codes, k, p);

        let mut decoder = Array2::<f32>::zeros((k, p));
        solve_decoder(&mut decoder, &eq, ridge);

        // Dense full system (A + ρI) D = B, solved independently.
        let mut mat = Array2::<f64>::zeros((k, k));
        for i in 0..k {
            mat[[i, i]] = eq.diag[i] + ridge;
        }
        for (&(a, b), &val) in eq.off.iter() {
            mat[[a as usize, b as usize]] = val;
            mat[[b as usize, a as usize]] = val;
        }
        let factor = mat.cholesky(Side::Lower).expect("dense SPD system");
        let dense = factor.solve_mat(&eq.b);

        for i in 0..k {
            for c in 0..p {
                let got = decoder[[i, c]] as f64;
                let want = dense[[i, c]];
                assert!(
                    (got - want).abs() <= 1.0e-5 + 1.0e-5 * want.abs(),
                    "block solve [{i},{c}] = {got} disagrees with dense solve {want}"
                );
            }
        }
    }

    #[test]
    fn matrix_free_cg_matches_dense_solve_to_charge_floor() {
        use faer::Side;
        use gam_linalg::faer_ndarray::FaerCholesky;

        let k = 12usize;
        let p = 3usize;
        let ridge = 1.0e-5f64;
        let eq = connected_tridiagonal_eq(k, p);
        let mut decoder = Array2::<f32>::zeros((k, p));
        let stats = solve_decoder(&mut decoder, &eq, ridge);
        assert_eq!(stats.component_count, 1);
        assert_eq!(stats.max_component_size, k);
        assert_eq!(stats.cg_columns, p);
        assert!(
            stats.cg_relative_residual <= ridge,
            "CG residual {} must stop below charge floor {ridge}",
            stats.cg_relative_residual
        );

        let mut mat = Array2::<f64>::zeros((k, k));
        for i in 0..k {
            mat[[i, i]] = eq.diag[i] + ridge;
        }
        for (&(a, b), &val) in eq.off.iter() {
            mat[[a as usize, b as usize]] = val;
            mat[[b as usize, a as usize]] = val;
        }
        let dense = mat
            .cholesky(Side::Lower)
            .expect("dense SPD system")
            .solve_mat(&eq.b);
        let mut diff2 = 0.0f64;
        let mut dense2 = 0.0f64;
        for i in 0..k {
            for c in 0..p {
                let diff = decoder[[i, c]] as f64 - dense[[i, c]];
                diff2 += diff * diff;
                dense2 += dense[[i, c]] * dense[[i, c]];
            }
        }
        let rel = (diff2 / dense2).sqrt();
        assert!(
            rel <= 5.0 * ridge,
            "CG decoder must match dense solve to the charge floor, rel={rel}, floor={ridge}"
        );
        assert!(
            stats.cg_kappa_hat.is_some(),
            "CG path must report a Lanczos condition estimate"
        );
    }

    #[test]
    fn direct_solve_threshold_tracks_percolation_scale_not_a_constant() {
        use super::direct_solve_size_threshold;
        // The exact-solve ceiling is the Erdős–Rényi critical-component scale
        // ⌈K^{2/3}⌉ — it MUST move with K (no frozen magic block size), and it
        // must sit strictly below K for any coupled dictionary so a single giant
        // component is never dense-factorised.
        assert_eq!(direct_solve_size_threshold(0), 0);
        assert_eq!(direct_solve_size_threshold(1), 1);
        for &k in &[8usize, 12, 64, 1024, 100_000] {
            let tau = direct_solve_size_threshold(k);
            let want = (k as f64).powf(2.0 / 3.0).ceil() as usize;
            assert_eq!(tau, want, "threshold must equal ⌈K^{{2/3}}⌉ for K={k}");
            assert!(
                tau < k,
                "a giant (size-K) component must exceed the dense threshold at K={k} (got {tau})"
            );
        }
        // It is genuinely a function of K, not a constant: the value grows with K.
        assert!(direct_solve_size_threshold(100_000) > direct_solve_size_threshold(12));
    }

    #[test]
    fn cg_lanczos_kappa_matches_true_condition_number() {
        let eigenvalues = [1.0f64, 1.7, 2.9, 4.6, 8.0, 13.0];
        let b = vec![1.0f64; eigenvalues.len()];
        let matvec = |x: &[f64]| -> Vec<f64> {
            eigenvalues
                .iter()
                .zip(x.iter())
                .map(|(&lambda, &xi)| lambda * xi)
                .collect()
        };
        let result = cg_solve(&matvec, &b, 1.0e-14, eigenvalues.len() + 2);
        let got = result.kappa_hat.expect("Lanczos kappa");
        let want = eigenvalues[eigenvalues.len() - 1] / eigenvalues[0];
        assert!(
            (got - want).abs() <= 1.0e-8 * want,
            "Lanczos κ̂ {got} must match true condition {want}"
        );
    }

    #[test]
    fn cg_reports_cap_reached_when_iterations_exhausted() {
        // A spread SPD spectrum needs several CG steps; a cap of 1 must return the
        // TYPED `CapReached` (not a silent partial), with iterations == cap and a
        // finite iterate — the signal the refresh uses to fall back instead of
        // spinning.
        let eigenvalues = [1.0f64, 5.0, 25.0, 125.0, 625.0];
        let b = vec![1.0f64; eigenvalues.len()];
        let matvec = |x: &[f64]| -> Vec<f64> {
            eigenvalues.iter().zip(x.iter()).map(|(&l, &xi)| l * xi).collect()
        };
        let result = cg_solve(&matvec, &b, 1.0e-12, 1);
        assert_eq!(result.stop, CgStop::CapReached);
        assert_eq!(result.iterations, 1);
        assert!(result.x.iter().all(|v| v.is_finite()));
    }

    #[test]
    fn cg_reports_breakdown_on_indefinite_operator() {
        // A non-SPD operator (a negative eigenvalue) makes some pᵀAp ≤ 0; CG must
        // return a TYPED `Breakdown` rather than iterate on negative curvature.
        let eigenvalues = [1.0f64, -3.0, 2.0];
        let b = vec![1.0f64, 1.0, 1.0];
        let matvec = |x: &[f64]| -> Vec<f64> {
            eigenvalues.iter().zip(x.iter()).map(|(&l, &xi)| l * xi).collect()
        };
        let result = cg_solve(&matvec, &b, 1.0e-12, 64);
        assert_eq!(result.stop, CgStop::Breakdown);
        assert!(result.iterations <= 64);
        assert!(result.x.iter().all(|v| v.is_finite()));
    }

    #[test]
    fn near_singular_giant_component_terminates_with_typed_failure() {
        // A path-graph (chain) co-firing Gram with coupling 0.5 is the symmetric
        // tridiagonal Toeplitz `tridiag(0.5, 1, 0.5)`, whose eigenvalues fill
        // `(0, 2)` densely — the smallest is `≈ ½(π/(k+1))²`, so at k=200 the
        // condition number is `~10⁴`. A GENERIC right-hand side excites the whole
        // spread spectrum (the worst case for CG), so reaching the 1e-9 charge
        // floor needs `~½√κ·ln(2/ε) ≈ 1.3e3` iterations — far beyond the derived
        // cap `≤ 2m+16 = 416`. The solve must therefore (i) bound the iterations
        // (no unbounded spin), (ii) report the large a-priori κ bound, (iii)
        // register a TYPED non-convergence, and (iv) leave a FINITE decoder via the
        // ridge-diagonal fallback.
        let k = 200usize;
        let p = 2usize;
        let diag = vec![1.0f64; k];
        let mut off = HashMap::new();
        for a in 0..(k - 1) {
            off.insert((a as u32, (a + 1) as u32), 0.5);
        }
        let mut b = Array2::<f64>::zeros((k, p));
        for i in 0..k {
            // Generic RHS spanning the whole spectrum (not an eigenvector, so CG
            // cannot shortcut on a clustered spectrum).
            b[[i, 0]] = ((i * 7 + 3) % 11) as f64 - 5.0;
            b[[i, 1]] = ((i * 5 + 1) % 13) as f64 - 6.0;
        }
        let eq = DecoderNormalEq {
            diag,
            b,
            off,
            firings: vec![4; k],
            amplitude_sum: vec![4.0; k],
        };
        let mut decoder = Array2::<f32>::zeros((k, p));
        let ridge = 1.0e-9f64;
        let stats = solve_decoder(&mut decoder, &eq, ridge);

        assert_eq!(stats.max_component_size, k, "path graph is one giant component");
        let kappa_bound = stats.cg_kappa_bound.expect("a-priori kappa bound recorded");
        assert!(
            kappa_bound > 1.0e6,
            "near-singular block must report a large a-priori kappa bound, got {kappa_bound}"
        );
        assert!(
            stats.cg_nonconverged_columns >= 1,
            "an under-resolved column must be a TYPED non-convergence, not a silent spin"
        );
        assert!(
            stats.cg_iterations <= (2 * k + 16) * p,
            "iterations must be bounded by the derived cap, got {}",
            stats.cg_iterations
        );
        assert!(
            decoder.iter().all(|v| v.is_finite()),
            "the ridge-diagonal fallback must leave a finite decoder"
        );
    }

    #[test]
    fn linear_fast_kernel_matches_legacy_run() {
        // TEMPORARY bit-parity gate (design gam#2232, Increment 2 shim; removed in
        // Increment 6). The unified linear fast kernel collapses the two historical
        // ridges into ONE shared ρ; at the shared default (code_ridge ==
        // decoder_ridge) it MUST reproduce the legacy `fit_sparse_dictionary`
        // (= `run`) output bit-for-bit, so the unification introduces no behavioral
        // change at the default. This pins that contract: same decoder, indices,
        // and codes to the last bit.
        use super::run_linear_fast_kernel;
        let (n, p, k) = (48usize, 5usize, 7usize);
        let mut x = Array2::<f32>::zeros((n, p));
        for i in 0..n {
            for c in 0..p {
                x[[i, c]] = (((i * 5 + c * 3 + 2) % 11) as f32 - 5.0) / 5.0;
            }
        }
        let config = SparseDictConfig {
            n_atoms: k,
            active: 2, // s > 1: exercises the coupled decoder solve on both paths
            minibatch: 16,
            max_epochs: 20,
            score_tile: 8,
            code_ridge: 1.0e-6,
            decoder_ridge: 1.0e-6, // shared default: kernel must equal legacy run
            tolerance: 1.0e-9,
            score_mode: gam_gpu::GpuMode::Off,
        };
        assert_eq!(
            config.code_ridge, config.decoder_ridge,
            "parity gate requires the shared-default (equal-ridge) config"
        );

        // Reference is the legacy fixed-ridge alternation `run` directly — NOT the
        // public `fit_sparse_dictionary`, whose shared-default entry now layers the
        // outer shared-ρ REML schedule on top (design gam#2232 Increment 2 plug 4).
        // The parity contract this pins is `run_linear_fast_kernel == run` at the
        // shared default ridge (the inner-solve seam), independent of that outer
        // selection loop.
        let legacy = super::run(x.view(), &config).expect("legacy run");
        let unified = run_linear_fast_kernel(x.view(), &config, config.decoder_ridge as f64)
            .expect("unified linear fast kernel");

        assert_eq!(
            legacy.decoder, unified.decoder,
            "unified kernel decoder must be bit-identical to legacy run"
        );
        assert_eq!(
            legacy.indices, unified.indices,
            "unified kernel indices must be bit-identical to legacy run"
        );
        assert_eq!(
            legacy.codes, unified.codes,
            "unified kernel codes must be bit-identical to legacy run"
        );
        assert_eq!(
            legacy.explained_variance, unified.explained_variance,
            "unified kernel EV must be bit-identical to legacy run"
        );
    }

    #[test]
    fn shared_rho_fs_step_matches_closed_form_evidence_fixed_point() {
        // Plug point 4 math (design gam#2232, Increment 2): the shared-ρ
        // Fellner–Schall / MacKay evidence fixed point is pure arithmetic over the
        // pooled linear-block aggregates. Pin it against a hand-computed value.
        use super::{LinearBlockRemlStats, linear_shared_rho_fs_step};
        let stats = LinearBlockRemlStats {
            gram_edof: 2.5,
            p_cols: 3,
            penalty_energy: 4.0,
            rss: 10.0,
            n_obs: 8,
        };
        // γ_tot = 3·2.5 = 7.5; resid_dof = 24 − 7.5 = 16.5; σ̂² = 10/16.5;
        // ρ_new = 7.5·σ̂²/4 = 1.1363636363636365.
        let rho_new = linear_shared_rho_fs_step(&stats, 1.0e-3);
        assert!(
            (rho_new - 1.136_363_636_363_636_5).abs() < 1.0e-12,
            "FS step must match the closed-form evidence fixed point, got {rho_new}"
        );

        // Degenerate aggregates leave ρ unchanged (SPD-preserving guard):
        // zero penalty energy, and non-positive effective dof.
        let zero_energy = LinearBlockRemlStats { penalty_energy: 0.0, ..stats };
        assert_eq!(linear_shared_rho_fs_step(&zero_energy, 7.0e-4), 7.0e-4);
        let zero_edof = LinearBlockRemlStats { gram_edof: 0.0, ..stats };
        assert_eq!(linear_shared_rho_fs_step(&zero_edof, 7.0e-4), 7.0e-4);

        // All dof consumed (γ_tot ≥ N·P) floors the residual dof at 1 rather than
        // dividing by a non-positive denominator, keeping ρ_new finite/positive.
        let saturated = LinearBlockRemlStats {
            gram_edof: 100.0,
            p_cols: 3,
            penalty_energy: 4.0,
            rss: 10.0,
            n_obs: 8,
        };
        let rho_sat = linear_shared_rho_fs_step(&saturated, 1.0e-3);
        assert!(rho_sat.is_finite() && rho_sat > 0.0, "saturated dof keeps ρ finite/positive");
    }

    /// Deterministic splitmix-backed uniform draw in `[0, 1)` (NO `rand` crate),
    /// the crate's canonical test PRNG pattern.
    fn next_unit(state: &mut u64) -> f64 {
        let h = gam_linalg::utils::splitmix64(state);
        (h >> 11) as f64 / (1u64 << 53) as f64
    }

    /// Assemble the DENSE `K×K` Gram `A` from the sparse `(diag, off)` the
    /// matrix-free estimator consumes — test-only oracle for the exact edof.
    fn densify_gram(diag: &[f64], off: &HashMap<(u32, u32), f64>, k: usize) -> Array2<f64> {
        let mut a = Array2::<f64>::zeros((k, k));
        for i in 0..k {
            a[[i, i]] = diag[i];
        }
        for (&(r, c), &v) in off.iter() {
            a[[r as usize, c as usize]] = v;
            a[[c as usize, r as usize]] = v;
        }
        a
    }

    /// Exact `γ = tr(A(A+ρI)⁻¹) = tr((A+ρI)⁻¹ A)` via a dense Cholesky solve of
    /// `(A+ρI) Y = A`, then `tr(Y)`.
    fn exact_gram_edof(a: &Array2<f64>, rho: f64) -> f64 {
        use faer::Side;
        use gam_linalg::faer_ndarray::FaerCholesky;
        let k = a.nrows();
        let mut m = a.clone();
        for i in 0..k {
            m[[i, i]] += rho;
        }
        let y = m
            .cholesky(Side::Lower)
            .expect("A+ρI is SPD")
            .solve_mat(a);
        (0..k).map(|i| y[[i, i]]).sum()
    }

    #[test]
    fn hutchinson_gram_edof_matches_exact_dense_trace() {
        // Plug point 4 (design gam#2232, Increment 2): the matrix-free Hutchinson
        // edof `tr(A(A+ρI)⁻¹)` must agree with the exact dense trace within the
        // stochastic tolerance DERIVED from the estimator's own variance target.
        // Small `K=32` so the dense oracle is cheap; the estimator itself never
        // forms the dense Gram.
        use super::{
            EDOF_TRACE_VARIANCE_PER_UNIT_TRACE, code_gram_from_routing, hutchinson_gram_edof,
        };
        let (k, s, n) = (32usize, 3usize, 400usize);

        // Deterministic 3-sparse routing over the 32 atoms with non-trivial
        // co-firing (so `A` has a genuine off-diagonal coupling graph, not a
        // diagonal degenerate case).
        let mut indices = Array2::<u32>::zeros((n, s));
        let mut codes = Array2::<f32>::zeros((n, s));
        let mut rng = 0x51E2_D3C4_A5B6_9788u64;
        for i in 0..n {
            for j in 0..s {
                // Distinct-per-slot atoms spread across the dictionary.
                let atom = ((i * (j + 1) * 7 + j * 5 + 1) % k) as u32;
                indices[[i, j]] = atom;
                codes[[i, j]] = (next_unit(&mut rng) as f32 - 0.5) * 2.0;
            }
        }

        let (diag, off) = code_gram_from_routing(indices.view(), codes.view(), k);
        let a_dense = densify_gram(&diag, &off, k);

        for &rho in &[1.0e-3_f64, 1.0e-1, 1.0] {
            let exact = exact_gram_edof(&a_dense, rho);
            let approx = hutchinson_gram_edof(&diag, &off, rho, k);

            // Derived tolerance: with `m = ⌈2/v⌉` probes and complementary trace
            // `c = K − γ`, the Rademacher estimator's standard error is at most
            // `√(2c/m)` (universal PSD-trace bound); allow 6 σ (a very safe tail).
            let probes = (2.0 / EDOF_TRACE_VARIANCE_PER_UNIT_TRACE).ceil();
            let c = (k as f64 - exact).max(0.0);
            let sd_bound = (2.0 * c / probes).sqrt();
            let tol = 6.0 * sd_bound + 1.0e-6;
            assert!(
                (approx - exact).abs() <= tol,
                "Hutchinson edof {approx} vs exact {exact} at rho={rho} exceeds derived \
                 6σ tolerance {tol} (c={c}, probes={probes})"
            );
            // The estimate is a valid effective-dof: in `[0, K]`.
            assert!(
                approx >= 0.0 && approx <= k as f64 + 1.0e-9,
                "edof {approx} must lie in [0, K]"
            );
        }
    }

    #[test]
    fn shared_rho_fixed_point_converges_and_tracks_planted_noise() {
        // Plug point 4 (design gam#2232, Increment 2): the shared-ρ FS fixed point
        // must (1) CONVERGE on a planted problem and (2) TRACK the planted noise —
        // a noisier reconstruction target selects a LARGER shared ridge
        // (ρ* = γ·σ̂²/‖D‖²_F grows with the residual variance). We iterate exactly
        // the schedule's loop body (fit → stats → FS step) so the test exercises
        // production math, and read the fixed point off at two noise levels.
        use super::{
            linear_block_reml_stats, linear_shared_rho_fs_step, reml_schedule_rho_rel_tol,
            run_linear_fast_kernel,
        };

        // Planted 2-sparse mixture over K orthonormal-ish atoms + additive noise.
        fn planted_noisy(n: usize, p: usize, k: usize, noise: f32, seed: u64) -> Array2<f32> {
            let mut atoms = Array2::<f32>::zeros((k, p));
            for atom in 0..k {
                // Deterministic near-orthonormal-ish rows (unit-normed).
                let mut norm = 0.0f64;
                for c in 0..p {
                    let v = (((atom * 13 + c * 7 + 3) % 17) as f32 - 8.0) / 8.0;
                    atoms[[atom, c]] = v;
                    norm += (v as f64) * (v as f64);
                }
                let inv = 1.0 / norm.sqrt().max(1.0e-12) as f32;
                for c in 0..p {
                    atoms[[atom, c]] *= inv;
                }
            }
            let mut rng = seed;
            let mut x = Array2::<f32>::zeros((n, p));
            for i in 0..n {
                let a0 = (i % k) as usize;
                let a1 = ((i / k + 1) % k) as usize;
                let c0 = 0.6 + 0.4 * next_unit(&mut rng) as f32;
                let c1 = 0.2 + 0.3 * next_unit(&mut rng) as f32;
                for c in 0..p {
                    let clean = c0 * atoms[[a0, c]] + c1 * atoms[[a1, c]];
                    let eps = noise * (next_unit(&mut rng) as f32 - 0.5) * 2.0;
                    x[[i, c]] = clean + eps;
                }
            }
            x
        }

        let (n, p, k) = (300usize, 12usize, 24usize);
        let config = SparseDictConfig {
            n_atoms: k,
            active: 2,
            minibatch: 64,
            max_epochs: 40,
            score_tile: 12,
            code_ridge: 1.0e-6,
            decoder_ridge: 1.0e-6,
            tolerance: 1.0e-9,
            score_mode: gam_gpu::GpuMode::Off,
        };

        // Iterate the schedule's loop body to the fixed point and return (ρ*, last
        // relative change) so we can assert convergence.
        let fixed_point = |x: ArrayView2<'_, f32>| -> (f64, f64) {
            let mut rho = config.decoder_ridge as f64;
            let mut last_rel = f64::INFINITY;
            for _ in 0..16 {
                let fit = run_linear_fast_kernel(x, &config, rho).expect("kernel fit");
                let stats = linear_block_reml_stats(x, &fit, rho);
                let rho_new = linear_shared_rho_fs_step(&stats, rho);
                last_rel = (rho_new - rho).abs() / rho.abs().max(f64::MIN_POSITIVE);
                rho = rho_new;
            }
            (rho, last_rel)
        };

        let x_low = planted_noisy(n, p, k, 0.03, 0x1111_2222_3333_4444);
        let x_high = planted_noisy(n, p, k, 0.40, 0x1111_2222_3333_4444);
        let (rho_low, rel_low) = fixed_point(x_low.view());
        let (rho_high, rel_high) = fixed_point(x_high.view());

        // (1) Both fixed points are finite, strictly positive, and SETTLED: the
        // last relative move is small (the FS evidence recursion contracted).
        assert!(
            rho_low.is_finite() && rho_low > 0.0 && rho_high.is_finite() && rho_high > 0.0,
            "shared ρ* must be finite and positive (low={rho_low}, high={rho_high})"
        );
        // Settled = the last relative move is within the schedule's derived
        // stopping band (the stochastic edof floor √v); iterating tighter would
        // chase Monte-Carlo noise, so this IS convergence for this estimator.
        let band = reml_schedule_rho_rel_tol();
        assert!(
            rel_low <= band && rel_high <= band,
            "FS fixed point must settle within the derived stopping band {band}: \
             last relative moves low={rel_low} high={rel_high}"
        );

        // (2) NOISE TRACKING: the noisier target selects the larger shared ridge.
        assert!(
            rho_high > rho_low,
            "shared ρ* must grow with planted noise: high-noise ρ*={rho_high} \
             must exceed low-noise ρ*={rho_low}"
        );
    }

    #[test]
    fn reml_schedule_held_out_ev_matches_or_beats_magic_ridge() {
        // Plug point 4 (design gam#2232, Increment 2): the shared-ρ REML schedule
        // (the new default entry) must NOT regress held-out reconstruction EV
        // versus the legacy fixed magic ridge — the risk pin for #1026 through the
        // new entry. Objective metric: OUT-OF-SAMPLE explained variance (frozen
        // decoder, fresh test-row codes), so the REML selection is judged on real
        // predictive quality, not on reproducing the magic-ridge decoder.
        use super::{run_linear_fast_kernel, run_linear_reml_schedule};
        use crate::sparse_dict::codes::solve_row_codes;

        // Held-out EV of a frozen decoder on a fresh block (production path).
        fn held_out_ev(
            decoder: ArrayView2<'_, f32>,
            x_test: ArrayView2<'_, f32>,
            s: usize,
            tile: usize,
            code_ridge: f32,
        ) -> f64 {
            let n = x_test.nrows();
            let p = x_test.ncols();
            let scorer = TileScorer::new(s, tile);
            let mut means = vec![0.0f64; p];
            for i in 0..n {
                for c in 0..p {
                    means[c] += x_test[[i, c]] as f64;
                }
            }
            for m in means.iter_mut() {
                *m /= n as f64;
            }
            let mut rss = 0.0f64;
            let mut tss = 0.0f64;
            for i in 0..n {
                let row = x_test.row(i);
                let active = scorer.route_row(row, decoder);
                let code = solve_row_codes(row, decoder, &active, s, code_ridge);
                let mut recon = vec![0.0f64; p];
                for j in 0..code.indices.len() {
                    let cj = code.codes[j] as f64;
                    if cj == 0.0 {
                        continue;
                    }
                    let drow = decoder.row(code.indices[j] as usize);
                    for c in 0..p {
                        recon[c] += cj * drow[c] as f64;
                    }
                }
                for c in 0..p {
                    let r = x_test[[i, c]] as f64 - recon[c];
                    rss += r * r;
                    let t = x_test[[i, c]] as f64 - means[c];
                    tss += t * t;
                }
            }
            if tss <= 1.0e-24 {
                if rss <= 1.0e-24 { 1.0 } else { 0.0 }
            } else {
                1.0 - rss / tss
            }
        }

        // Planted 2-sparse mixture with modest noise (so REML has a real ridge to
        // select), deterministic 80/20 stride split.
        let (k, p, n) = (24usize, 12usize, 500usize);
        let mut atoms = Array2::<f32>::zeros((k, p));
        for atom in 0..k {
            let mut norm = 0.0f64;
            for c in 0..p {
                let v = (((atom * 11 + c * 5 + 2) % 13) as f32 - 6.0) / 6.0;
                atoms[[atom, c]] = v;
                norm += (v as f64) * (v as f64);
            }
            let inv = 1.0 / norm.sqrt().max(1.0e-12) as f32;
            for c in 0..p {
                atoms[[atom, c]] *= inv;
            }
        }
        let mut rng = 0x0BAD_C0FF_EE12_3456u64;
        let mut x = Array2::<f32>::zeros((n, p));
        for i in 0..n {
            let a0 = i % k;
            let a1 = (i / k + 1) % k;
            let c0 = 0.6 + 0.4 * next_unit(&mut rng) as f32;
            let c1 = 0.2 + 0.3 * next_unit(&mut rng) as f32;
            for c in 0..p {
                let clean = c0 * atoms[[a0, c]] + c1 * atoms[[a1, c]];
                let eps = 0.15 * (next_unit(&mut rng) as f32 - 0.5) * 2.0;
                x[[i, c]] = clean + eps;
            }
        }
        let mut train_rows = Vec::new();
        let mut test_rows = Vec::new();
        for i in 0..n {
            if i % 5 == 0 {
                test_rows.push(i);
            } else {
                train_rows.push(i);
            }
        }
        let mut x_train = Array2::<f32>::zeros((train_rows.len(), p));
        for (r, &i) in train_rows.iter().enumerate() {
            x_train.row_mut(r).assign(&x.row(i));
        }
        let mut x_test = Array2::<f32>::zeros((test_rows.len(), p));
        for (r, &i) in test_rows.iter().enumerate() {
            x_test.row_mut(r).assign(&x.row(i));
        }

        let s = 2usize;
        let tile = 12usize;
        let config = SparseDictConfig {
            n_atoms: k,
            active: s,
            minibatch: 128,
            max_epochs: 60,
            score_tile: tile,
            code_ridge: 1.0e-6,
            decoder_ridge: 1.0e-6,
            tolerance: 1.0e-9,
            score_mode: gam_gpu::GpuMode::Off,
        };

        // Magic-ridge baseline: legacy fixed-ridge fit at the default 1e-6.
        let magic = run_linear_fast_kernel(x_train.view(), &config, config.decoder_ridge as f64)
            .expect("magic-ridge fit");
        // REML-selected shared ρ: the new default schedule.
        let reml = run_linear_reml_schedule(x_train.view(), &config).expect("reml schedule fit");

        let magic_ev = held_out_ev(magic.decoder.view(), x_test.view(), s, tile, config.code_ridge);
        let reml_ev = held_out_ev(reml.decoder.view(), x_test.view(), s, tile, config.code_ridge);

        // REML must MATCH-OR-BEAT the magic ridge on held-out EV (small epsilon
        // absorbs f32 routing noise); it selects the ridge by evidence rather than
        // pinning a constant, so it cannot do materially worse out of sample.
        assert!(
            reml_ev + 1.0e-3 >= magic_ev,
            "REML-selected shared ρ held-out EV {reml_ev} must match-or-beat the \
             magic-ridge baseline {magic_ev}"
        );
    }

    #[test]
    fn returned_ev_is_fresh_code_ev_no_stale_gap() {
        // The convergence-decision EV (= the returned EV) must be the EV of the codes
        // FRESHLY routed against the final normalised decoder — not a stale-code
        // surrogate. We recompute that EV from the public fit's decoder and assert it
        // matches the reported one to f32 rounding.
        let (n, p, k) = (60usize, 6usize, 8usize);
        let mut x = Array2::<f32>::zeros((n, p));
        for i in 0..n {
            for c in 0..p {
                x[[i, c]] = (((i * 3 + c * 7 + 1) % 11) as f32 - 5.0) / 5.0;
            }
        }
        let config = SparseDictConfig {
            n_atoms: k,
            active: 2, // s > 1: exercises the coupled decoder solve
            minibatch: 16,
            max_epochs: 25,
            score_tile: 8,
            code_ridge: 1.0e-6,
            decoder_ridge: 1.0e-6,
            tolerance: 1.0e-9,
            score_mode: gam_gpu::GpuMode::Off,
        };
        let fit = fit_sparse_dictionary(x.view(), &config).expect("fit");
        let s = fit.active;
        assert!(s > 1, "test must run the coupled s>1 lane");

        let scorer = TileScorer::new(s, config.score_tile);
        let codes = route_and_code_all(
            x.view(),
            fit.decoder.view(),
            &scorer,
            s,
            config.code_ridge,
            config.minibatch,
            config.score_mode,
            None,
        )
        .expect("fresh route");
        let fresh_ev = explained_variance(x.view(), &codes, fit.decoder.view());
        assert!(
            (fresh_ev - fit.explained_variance).abs() < 1.0e-6,
            "returned EV {} must equal fresh-code EV {fresh_ev} (no stale-code gap)",
            fit.explained_variance
        );
    }
}
