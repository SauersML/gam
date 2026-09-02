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
use super::{SparseDictConfig, SparseDictConvergence, SparseDictFit};
use gam_linalg::pcg::{
    CpuPcgBlockBackend, PcgCoreResult, PcgStop, SymmetricLowRankPreconditioner, pcg_multi_core,
};
use ndarray::{Array2, ArrayView2, Axis};
use rayon::prelude::*;
use std::collections::HashMap;
use std::fmt;
use std::time::Instant;

/// Typed failure from the sparse-dictionary optimizer.
#[derive(Clone, Debug)]
pub enum SparseDictionaryError {
    InvalidInput {
        reason: String,
    },
    NumericalFailure {
        reason: String,
    },
    InnerNonConvergence {
        epochs: usize,
        explained_variance: f64,
        ev_residual: f64,
        tolerance: f64,
        accepted_births: usize,
        decoder_fixed_point_residual: f64,
        routing_residual: f64,
        solve_residual: f64,
        solve_tolerance: f64,
        decoder_nonconverged_columns: usize,
        decoder_dense_cholesky_declines: usize,
    },
    TraceNonConvergence {
        rho: f64,
        probe: usize,
        iterations: usize,
        residual: f64,
        tolerance: f64,
    },
    InvalidRemlEvidence {
        reason: String,
    },
}

impl SparseDictionaryError {
    fn invalid_input(reason: impl Into<String>) -> Self {
        Self::InvalidInput {
            reason: reason.into(),
        }
    }
}

impl From<String> for SparseDictionaryError {
    fn from(reason: String) -> Self {
        Self::NumericalFailure { reason }
    }
}

impl fmt::Display for SparseDictionaryError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::InvalidInput { reason } | Self::NumericalFailure { reason } => {
                f.write_str(reason)
            }
            Self::InnerNonConvergence {
                epochs,
                explained_variance,
                ev_residual,
                tolerance,
                accepted_births,
                decoder_fixed_point_residual,
                routing_residual,
                solve_residual,
                solve_tolerance,
                decoder_nonconverged_columns,
                decoder_dense_cholesky_declines,
            } => write!(
                f,
                "fit_sparse_dictionary did not converge after {epochs} epochs: EV \
                 {explained_variance:.6}, EV residual {ev_residual:.3e} (tolerance \
                 {tolerance:.3e}), decoder fixed-point residual \
                 {decoder_fixed_point_residual:.3e}, routing residual {routing_residual:.3e}, \
                 accepted births {accepted_births}, linear-solve residual \
                 {solve_residual:.3e} (tolerance {solve_tolerance:.3e}), nonconverged decoder \
                 columns {decoder_nonconverged_columns}, dense Cholesky declines routed to CG \
                 {decoder_dense_cholesky_declines}"
            ),
            Self::TraceNonConvergence {
                rho,
                probe,
                iterations,
                residual,
                tolerance,
            } => write!(
                f,
                "fit_sparse_dictionary REML trace solve did not converge at rho={rho:.6e}, \
                 probe {probe}, after {iterations} iterations: relative residual \
                 {residual:.3e} exceeds {tolerance:.3e}"
            ),
            Self::InvalidRemlEvidence { reason } => {
                write!(
                    f,
                    "fit_sparse_dictionary REML evidence is invalid: {reason}"
                )
            }
        }
    }
}

impl std::error::Error for SparseDictionaryError {}

impl From<SparseDictionaryError> for String {
    fn from(error: SparseDictionaryError) -> Self {
        error.to_string()
    }
}

/// Certified inner work state. This is deliberately not a [`SparseDictFit`]:
/// the outer REML fixed point must also settle before the public model exists.
#[derive(Clone, Debug)]
pub(crate) struct SparseDictIterate {
    pub(crate) decoder: Array2<f32>,
    pub(crate) indices: Array2<u32>,
    pub(crate) codes: Array2<f32>,
    pub(crate) epochs: usize,
    pub(crate) active: usize,
    pub(crate) score_route_stats: ScoreRouteStats,
    pub(crate) decoder_solve_stats: DecoderSolveStats,
    inner_ev_residual: f64,
    decoder_fixed_point_residual: f64,
    routing_residual: f64,
    inner_tolerance: f64,
    accepted_births: usize,
    live_atom_high_water: usize,
    support_saturated: bool,
    /// Whether the decoder AND routing fixed-point residuals ALSO closed to
    /// `inner_tolerance` (arm 1). `false` marks a **best-effort** iterate returned
    /// at `K` above the intrinsic rank, where the `>rank` spurious support
    /// directions rotate freely in the equivalent-optima manifold and the routing
    /// residual legitimately cannot close (#2275) — the objective (EV) has
    /// plateaued but the discrete routing keeps churning. Convergence itself is
    /// decided by the gauge-invariant EV plateau, so both certified and open
    /// iterates are returned; only a still-climbing objective (or a failed linear
    /// subsolve) is a genuine non-convergence error. Mirrors
    /// [`super::block::BlockSparseConvergence::certified`].
    certified: bool,
}

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
    score_mode: gam_gpu::GpuPolicy,
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

/// Gauge-invariant displacement of two unit-row dictionaries. Active atoms are
/// compared as rank-one projectors (`1 - cos² θ`), so a harmless sign flip is
/// zero; a transition between active and dormant (zero) capacity is one.
fn decoder_fixed_point_residual(previous: &Array2<f32>, next: &Array2<f32>) -> f64 {
    previous
        .axis_iter(Axis(0))
        .zip(next.axis_iter(Axis(0)))
        .map(|(left, right)| {
            let left_norm2 = left.iter().map(|&v| (v as f64) * (v as f64)).sum::<f64>();
            let right_norm2 = right.iter().map(|&v| (v as f64) * (v as f64)).sum::<f64>();
            if left_norm2 <= DEAD_DENOM && right_norm2 <= DEAD_DENOM {
                return 0.0;
            }
            if left_norm2 <= DEAD_DENOM || right_norm2 <= DEAD_DENOM {
                return 1.0;
            }
            let dot = left
                .iter()
                .zip(right.iter())
                .map(|(&a, &b)| (a as f64) * (b as f64))
                .sum::<f64>();
            (1.0 - dot * dot / (left_norm2 * right_norm2)).clamp(0.0, 1.0)
        })
        .fold(0.0, f64::max)
}

/// Fixed-point residual of the exposed sparse routing. It is the larger of the
/// relative coefficient displacement and the reconstruction displacement,
/// evaluated without materialising either `N×K` codes or a second `N×P` matrix.
fn routing_fixed_point_residual(
    x: ArrayView2<'_, f32>,
    previous_decoder: ArrayView2<'_, f32>,
    previous: &[SparseCode],
    next_decoder: ArrayView2<'_, f32>,
    next: &[SparseCode],
) -> f64 {
    let mut code_delta2 = 0.0f64;
    let mut code_scale2 = 0.0f64;
    let mut reconstruction_delta2 = 0.0f64;
    let mut data_scale2 = 0.0f64;

    for row in 0..x.nrows() {
        let old = &previous[row];
        let new = &next[row];
        for (slot, &atom) in old.indices.iter().enumerate() {
            let old_value = old.codes[slot] as f64;
            if old_value == 0.0 {
                continue;
            }
            let new_value = new
                .indices
                .iter()
                .zip(new.codes.iter())
                .filter(|(candidate, _)| **candidate == atom)
                .map(|(_, &value)| value as f64)
                .sum::<f64>();
            let delta = new_value - old_value;
            code_delta2 += delta * delta;
            code_scale2 += old_value * old_value + new_value * new_value;
        }
        for (slot, &atom) in new.indices.iter().enumerate() {
            let new_value = new.codes[slot] as f64;
            if new_value == 0.0
                || old
                    .indices
                    .iter()
                    .zip(old.codes.iter())
                    .any(|(&candidate, &value)| candidate == atom && value != 0.0)
            {
                continue;
            }
            code_delta2 += new_value * new_value;
            code_scale2 += new_value * new_value;
        }

        for column in 0..x.ncols() {
            let old_value = old
                .indices
                .iter()
                .zip(old.codes.iter())
                .map(|(&atom, &code)| {
                    (code as f64) * previous_decoder[[atom as usize, column]] as f64
                })
                .sum::<f64>();
            let new_value = new
                .indices
                .iter()
                .zip(new.codes.iter())
                .map(|(&atom, &code)| (code as f64) * next_decoder[[atom as usize, column]] as f64)
                .sum::<f64>();
            let delta = new_value - old_value;
            reconstruction_delta2 += delta * delta;
            let observed = x[[row, column]] as f64;
            data_scale2 += observed * observed;
        }
    }

    let code_residual = if code_scale2 > 0.0 {
        code_delta2 / code_scale2
    } else {
        0.0
    };
    let reconstruction_residual = if data_scale2 > 0.0 {
        reconstruction_delta2 / data_scale2
    } else if reconstruction_delta2 == 0.0 {
        0.0
    } else {
        f64::INFINITY
    };
    code_residual.max(reconstruction_residual)
}

/// Scale-free EV-plateau fraction for the linear trainer's best-effort arm
/// (#2275), mirroring [`super::block`]'s `BLOCK_EV_PLATEAU_FRACTION`: a round is
/// stationary when it captured less than this fraction of the total EV
/// improvement achieved since entry, so it fires at the achievable plateau
/// wherever it sits (~1e-6 well-posed, ~1e-4 over-complete).
const LINEAR_EV_PLATEAU_FRACTION: f64 = 1.0e-3;
/// Stationary rounds required (within the trailing [`LINEAR_EV_PLATEAU_WINDOW`])
/// before returning a best-effort open iterate — prevents a transient early flat
/// from exiting a still-climbing fit, and sets the confirmation horizon (a budget
/// too short to accumulate this many stationary rounds cannot confirm a plateau,
/// so it stays a typed non-convergence).
const LINEAR_EV_PLATEAU_MIN_ROUNDS: usize = 3;
/// Trailing window over which [`LINEAR_EV_PLATEAU_MIN_ROUNDS`] stationary rounds
/// confirm the plateau (#2396). At K >> rank the discrete top-s routing is a limit
/// cycle: the objective oscillates within a band while its mean creeps to the
/// achievable plateau, so a STRICT consecutive-round counter is reset by every
/// up-swing and can miss a genuinely-bounded objective (surfaced on real OLMO
/// activations, where the fit reaches two-in-a-row repeatedly but an up-swing
/// resets it before three). Requiring MIN_ROUNDS stationary rounds within a window
/// of `MIN_ROUNDS + 1` tolerates exactly one such up-swing — a minimal debounce
/// that is a strict superset of the consecutive rule (three in a row ⇒ three in the
/// last four), so it only ever confirms MORE, never exits a still-climbing fit
/// (whose rounds are non-stationary and never fill the window), and keeps the
/// confirmation horizon (a budget shorter than the window cannot confirm).
const LINEAR_EV_PLATEAU_WINDOW: usize = LINEAR_EV_PLATEAU_MIN_ROUNDS + 1;
/// Consecutive rounds without a new high-water mark in the number of live atoms
/// before fixed-cardinality birth swaps are treated as saturated support.
///
/// A residual-row proposal firing on the same row that seeded it is not evidence
/// of structural progress. Progress means expanding the live support; once its
/// cardinality has set no new high for this whole window, accepted proposals are
/// replacements on the current support manifold. They may still improve the
/// objective, so saturation alone never exits: the independent EV-plateau window
/// must also confirm. Matching that window gives both signals the same minimum
/// observation horizon and avoids any guessed `K/N` capacity threshold (#2400).
const LINEAR_SUPPORT_SATURATION_ROUNDS: usize = LINEAR_EV_PLATEAU_WINDOW;

/// Captured-fraction EV-plateau detector for the best-effort/open arm (#2396).
///
/// At `K >> rank` the alternation need not converge at all: the discrete routing
/// puts it in a LIMIT CYCLE whose objective oscillates at a fixed amplitude
/// forever, so `|ΔEV|` does not tend to zero and no round-to-round smallness test
/// can hold. What is nevertheless true, and is what the open arm certifies, is
/// that the ACHIEVABLE objective has stopped improving: the running best over the
/// returnable iterates sets no further high. That statement is about a monotone
/// non-decreasing sequence, so it cannot be confused by the sign of any
/// individual round — and it is scored against the climb achieved since entry,
/// which keeps it scale-free (it fires wherever the plateau sits, ~1e-6
/// well-posed, ~1e-4 over-complete) with no absolute threshold to tune.
///
/// Reading it off the round-to-round change instead is what made the earlier form
/// unsound. It scored the UPWARD share `max(ΔEV, 0)` against `next_ev − entry_ev`,
/// and both halves fail on a descent: the numerator is identically zero for any
/// round that moved downhill, and the denominator is `≤ 0` for any round sitting
/// below where the fit entered, which took the "no climb to divide by" branch.
/// A fit that was monotonically getting WORSE therefore reported a plateau on
/// every round and was returned on the first window it filled.
///
/// The other half of making this honest is [`BestOpenIterate`]: certifying that
/// the achievable objective stopped improving obliges the return to hand back the
/// iterate that ATTAINS it, not whichever point of the cycle the confirming round
/// happened to land on. With no climb at all to measure against — the fit never
/// once beat the state it entered with — only a genuine numerical standstill
/// counts, which is arm 1's own test; a fit still moving below its entry EV has
/// nothing to hand back and stays a typed non-convergence.
#[derive(Clone, Copy, Debug)]
struct EvPlateau {
    entry_ev: f64,
    best_ev: f64,
}

impl EvPlateau {
    fn new(entry_ev: f64) -> Self {
        Self {
            entry_ev,
            best_ev: entry_ev,
        }
    }

    /// Record the EV of this round's returnable iterate and report whether the
    /// achievable objective has stopped improving at it. `ev_residual` is
    /// `|EV(T(z)) − EV(z)|`, the magnitude of the move the round made.
    fn observe(&mut self, candidate_ev: f64, ev_residual: f64, fixed_point_tol: f64) -> bool {
        let improvement = (candidate_ev - self.best_ev).max(0.0);
        if candidate_ev > self.best_ev {
            self.best_ev = candidate_ev;
        }
        if ev_residual <= fixed_point_tol {
            return true;
        }
        let climb = self.best_ev - self.entry_ev;
        climb > f64::MIN_POSITIVE && improvement / climb < LINEAR_EV_PLATEAU_FRACTION
    }
}

/// The best returnable iterate seen so far, and the fixed-point evidence measured
/// AT it (#2396).
///
/// [`EvPlateau`] certifies that the achievable objective stopped improving. That
/// is a claim about the running maximum, so the object handed back has to be the
/// one attaining that maximum: returning the confirming round's own iterate would
/// certify a level the returned model does not have, and on a descending
/// trajectory it would hand back a strictly degraded state. Each field is the
/// evidence recorded for THIS state's own transition, so the certificate travels
/// with the model rather than describing some later round.
struct BestOpenIterate {
    decoder: Array2<f32>,
    codes: Vec<SparseCode>,
    explained_variance: f64,
    ev_residual: f64,
    decoder_residual: f64,
    routing_residual: f64,
    accepted_births: usize,
    support_saturated: bool,
    decoder_solve_stats: DecoderSolveStats,
}

/// Arm-2 verdict for one round: may the open arm count this round toward a
/// confirmed plateau? (#2396/#2400)
///
/// An ABSOLUTE certificate still requires zero accepted births — that is arm 1's
/// own test and this predicate is not consulted for it. The open arm additionally
/// admits fixed-cardinality birth SWAPS, but only once live support has stopped
/// setting new highs for a full confirmation window: a residual-row proposal that
/// fires on the same row that seeded it says nothing about structural progress,
/// whereas a support cardinality that is still growing says the fit has not
/// finished recruiting. Saturation alone never admits anything — the independent
/// objective test must plateau too — and `epoch > 0` skips the first post-entry
/// round, whose climb denominator is still forming.
fn open_round_is_stationary(
    epoch: usize,
    accepted_births: usize,
    support_saturated: bool,
    numerically_sound: bool,
    objective_plateaued: bool,
) -> bool {
    let structure_stationary = accepted_births == 0 || support_saturated;
    epoch > 0 && structure_stationary && numerically_sound && objective_plateaued
}

#[derive(Clone, Copy, Debug)]
struct LiveSupportGrowth {
    high_water: usize,
    rounds_without_growth: usize,
}

impl LiveSupportGrowth {
    fn new(initial_live_atoms: usize) -> Self {
        Self {
            high_water: initial_live_atoms,
            rounds_without_growth: 0,
        }
    }

    fn observe(&mut self, live_atoms: usize) -> bool {
        if live_atoms > self.high_water {
            self.high_water = live_atoms;
            self.rounds_without_growth = 0;
        } else {
            self.rounds_without_growth = self.rounds_without_growth.saturating_add(1);
        }
        self.rounds_without_growth >= LINEAR_SUPPORT_SATURATION_ROUNDS
    }
}

/// Per-term rounding scale for the fixed-point convergence floor (#2396).
///
/// The certified arm compares three O(1)-normalized residuals — the EV change
/// (`explained_variance ∈ [0, 1]`), the unit-normed decoder fixed-point residual,
/// and the normalized routing residual — against `config.tolerance`. No
/// floating-point fixed-point iteration can drive such a residual below the
/// rounding error accumulated in computing it (a reduction over up to
/// `max(n, k, p)` terms, each carrying ~`EPSILON` relative error), so the
/// effective convergence tolerance floors at `SPARSE_DICT_FIXED_POINT_ROUNDING *
/// max(n, k, p)`. This makes a `config.tolerance` of exactly `0.0` — "converge as
/// tightly as the arithmetic allows" — certify a machine-precision fixed point
/// (residuals at the ~1e-15 rounding floor) instead of rejecting it as
/// non-convergent. The floor stays many orders of magnitude below any genuine
/// non-convergence residual (~1e-4 and up), so a still-moving objective can never
/// be laundered as converged.
const SPARSE_DICT_FIXED_POINT_ROUNDING: f64 = 32.0 * f64::EPSILON;

/// Seed one inner alternation, then run it to its fixed point at the ridges
/// carried by `config`.
///
/// `decoder_recycle` is OWNED BY THE CALLER, not by this call (#2742). The
/// recycle space carries the break-even latch, whose documented scope is the
/// FIT; constructing it here made it reset on every outer REML iteration, so a
/// correction already measured as a loss was rebuilt and re-rejected up to
/// [`REML_SCHEDULE_MAX_OUTER_ITERS`] times. The per-fit subspace HISTORY is
/// still local to one inner run — [`DecoderRecycleSpace::begin_fit`] clears the
/// directions while preserving the latch — so only the decision survives, not
/// the stale coarse space it was measured on.
pub(super) fn run_seeded(
    x: ArrayView2<'_, f32>,
    config: &SparseDictConfig,
    decoder_recycle: &mut DecoderRecycleSpace,
) -> Result<SparseDictIterate, SparseDictionaryError> {
    validate(x, config)?;
    let n = x.nrows();
    let p = x.ncols();
    let k = config.n_atoms;
    let s = config.active.min(k).max(1);

    let fit_start = Instant::now();
    let mut decoder = seed_decoder(x, k);
    unit_norm_rows(&mut decoder)?;
    // Coarse phase heartbeat on the same channel as the score-router DECLINE
    // (log::warn survives the RUST_LOG=warn harnesses that drop log::info), so a
    // multi-hour host fit is never silent. Emitted at seed / initial-route /
    // per-epoch cadence only — never per row or per minibatch.
    log::warn!(
        "[SAE sparse_dict] seeded decoder N={n} P={p} K={k} s={s} \
         seed_s={:.1} (route + refresh follow)",
        fit_start.elapsed().as_secs_f64(),
    );

    run_from_decoder(x, config, decoder, decoder_recycle, fit_start)
}

/// Run one inner alternation from an already-normalized decoder.
///
/// Both legal callers establish the decoder's provenance explicitly: the seeded
/// entry normalizes [`seed_decoder`], while the REML continuation consumes the
/// [`SparseDictIterate`] returned at the preceding shared ridge. Codes are never
/// carried across this boundary: the first operation below routes and solves them
/// afresh at `config.code_ridge`, so every normal equation and fixed-point
/// certificate belongs to the current ridge (#2441).
fn run_from_decoder(
    x: ArrayView2<'_, f32>,
    config: &SparseDictConfig,
    mut decoder: Array2<f32>,
    decoder_recycle: &mut DecoderRecycleSpace,
    fit_start: Instant,
) -> Result<SparseDictIterate, SparseDictionaryError> {
    let n = x.nrows();
    let p = x.ncols();
    let k = config.n_atoms;
    let s = config.active.min(k).max(1);
    if decoder.dim() != (k, p) {
        return Err(SparseDictionaryError::NumericalFailure {
            reason: format!(
                "sparse-dictionary inner start has decoder shape {:?}, expected ({k}, {p})",
                decoder.dim()
            ),
        });
    }
    if !decoder.iter().all(|value| value.is_finite()) {
        return Err(SparseDictionaryError::NumericalFailure {
            reason: "sparse-dictionary inner start has a non-finite decoder".to_string(),
        });
    }

    let scorer = TileScorer::new(s, config.score_tile);
    let mut score_route_stats = ScoreRouteStats::default();
    let mut epochs_run = 0usize;
    let mut decoder_solve_stats = DecoderSolveStats::default();
    // Start this inner run's coarse-space history empty (the first refresh must
    // run at rank 0 to supply its own Jacobi baseline) while KEEPING the
    // caller-owned break-even latch, which is scoped to the whole fit (#2742).
    decoder_recycle.begin_fit(k);
    let mut ev_residual = f64::INFINITY;
    let mut decoder_residual = f64::INFINITY;
    let mut routing_residual = f64::INFINITY;
    let mut accepted_births = 0usize;

    // (a)+(b) route + sparse codes for every row against the current, unit-normed
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
    let mut current_ev = explained_variance(x, &codes, decoder.view());
    let mut live_support = LiveSupportGrowth::new(live_atom_count(&codes, k));
    // Effective fixed-point tolerance: never demand tighter closure than the
    // arithmetic can express (#2396). A `config.tolerance` of `0.0` asks for the
    // tightest achievable fixed point, which in floating point is the rounding
    // floor of the residual reductions, not literal zero.
    let fixed_point_tol = config
        .tolerance
        .max(SPARSE_DICT_FIXED_POINT_ROUNDING * (n.max(k).max(p) as f64));
    // Captured-fraction EV-plateau detector (arm 2 best-effort): "how big was
    // this round's move against the climb achieved since entry". Stationary
    // rounds within a trailing window mark the achievable plateau; the window
    // (not a strict consecutive run) is what makes the signal robust to the
    // K>>rank routing limit cycle (#2396).
    let mut ev_plateau = EvPlateau::new(current_ev);
    // Trailing window of per-round stationary flags; a majority-with-one-tolerance
    // of these confirms a best-effort plateau, robust to the K>>rank routing limit
    // cycle (#2396). See [`LINEAR_EV_PLATEAU_WINDOW`].
    let mut plateau_flags: std::collections::VecDeque<bool> =
        std::collections::VecDeque::with_capacity(LINEAR_EV_PLATEAU_WINDOW);
    // The iterate the open arm will hand back if it confirms a plateau: the best
    // returnable state seen, carrying the evidence measured at it (see
    // [`BestOpenIterate`]).
    let mut best_open: Option<BestOpenIterate> = None;

    for epoch in 0..config.max_epochs {
        epochs_run = epoch + 1;
        let epoch_start = Instant::now();

        // `decoder` + `codes` is the canonical state being tested. Advance the
        // full deterministic map once, then certify THIS input state from the
        // distance to its image. Returning the input (not the freshly mutated
        // image) makes the model arrays exactly the state whose fixed-point
        // residuals were measured.
        let certified_decoder = decoder.clone();
        let certified_codes = codes.clone();
        let certified_ev = current_ev;

        // (c) decoder refresh from this state alone. Re-accumulating the same
        // corpus across epochs made the update depend on hidden, non-model state
        // and prevented a replayable fixed-point certificate. Streaming still
        // assembles one epoch from shards; the one-shot map assembles one corpus
        // exactly once per step.
        let mut normal_eq = DecoderNormalEq::zeros(k, p);
        normal_eq.accumulate(x, &certified_codes);
        let accumulate_secs = epoch_start.elapsed().as_secs_f64();
        let sigma = residual_scale(x, &codes, decoder.view());
        let sigma_secs = epoch_start.elapsed().as_secs_f64() - accumulate_secs;
        let (stats, _gate) = solve_decoder_with_routability_gate_recycled(
            &mut decoder,
            &normal_eq,
            config.decoder_ridge as f64,
            sigma,
            config.score_mode,
            decoder_recycle,
        )?;
        decoder_solve_stats = stats;
        let refresh_secs = epoch_start.elapsed().as_secs_f64();

        // (d) unit-norm projection (identifies code scale) + stable sign.
        unit_norm_rows(&mut decoder)?;

        // (e) dead-atom revival. Atoms that fired for no row this epoch are re-
        // seeded onto the current worst-reconstructed rows' residual directions.
        // Without this, a large dictionary leaves most atoms at their seed (see
        // the dead counts in the fit report / #1026): effective `K` collapses to a
        // handful of live atoms, EV is non-monotone in `K`, and the lane never
        // climbs toward reconstruction parity. Reviving toward high-residual rows
        // is the standard dead-feature resampling that makes every atom load-
        // bearing, so adding atoms can only help. It runs only while dead atoms
        // remain, so a fully-alive small-`K` dictionary is untouched.
        let revived_atoms = revive_dead_atoms(x, &codes, &mut decoder);
        if !revived_atoms.is_empty() {
            unit_norm_rows(&mut decoder)?;
        }

        // (a)+(b) FRESH codes against the just-refreshed, unit-normed decoder.
        // These are the codes that define the post-epoch model, so they (i) feed
        // the NEXT epoch's refresh and (ii) score the convergence EV below. This
        // re-route deliberately replaces the previous STALE-code EV (which scored
        // the new decoder against codes solved before the refresh + normalisation):
        // the convergence decision now uses exactly the codes that define the
        // returned model, so there is no stale-code surrogate gap.
        let mut next_codes = route_and_code_all(
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
        let next_ev = explained_variance(x, &next_codes, decoder.view());
        let improve = next_ev - certified_ev;
        let mut revived_mask = vec![false; k];
        for &atom in &revived_atoms {
            revived_mask[atom] = true;
        }
        let mut accepted_mask = vec![false; k];
        let mut next_alive = vec![false; k];
        for code in &next_codes {
            for (slot, &atom) in code.indices.iter().enumerate() {
                let atom = atom as usize;
                if code.codes[slot] == 0.0 {
                    continue;
                }
                next_alive[atom] = true;
                if revived_mask[atom] {
                    accepted_mask[atom] = true;
                }
            }
        }
        accepted_births = accepted_mask.iter().filter(|accepted| **accepted).count();
        let next_live_atoms = next_alive.iter().filter(|&&alive| alive).count();
        let support_saturated = live_support.observe(next_live_atoms);

        // Rejected residual-row proposals are dormant capacity, not trained
        // model parameters. Null them before measuring/adopting the next state so
        // held-out transforms can never expose an arbitrary rejected direction.
        // `accepted_mask` already records, for every revived atom, whether any
        // fresh code fired it with a nonzero value — reuse it instead of
        // rescanning all `n` codes per revived atom (O(revived·n·s) → O(revived)).
        for &atom in &revived_atoms {
            if !accepted_mask[atom] {
                decoder.row_mut(atom).fill(0.0);
            }
        }

        ev_residual = improve.abs();
        decoder_residual = decoder_fixed_point_residual(&certified_decoder, &decoder);
        routing_residual = routing_fixed_point_residual(
            x,
            certified_decoder.view(),
            &certified_codes,
            decoder.view(),
            &next_codes,
        );

        // Per-epoch heartbeat on the log::warn channel (log::info is dropped by
        // the RUST_LOG=warn harnesses, which is why a multi-hour host fit went
        // silent). A hang in the refresh or route is visible at round cadence,
        // and the CG certificate (giant component size, the a-priori κ bound,
        // any typed non-convergence) is on the same line.
        log::warn!(
            "[SAE epoch {}/{}] ev={:.6} improve={:.3e} ev_resid={:.3e} decoder_resid={:.3e} \
             routing_resid={:.3e} births={} revived={} live={}/{} no_growth={} \
             support_saturated={} refresh_s={:.2} route_s={:.2} elapsed_s={:.1} \
             accumulate_s={:.3} sigma_s={:.3} graph_build_s={:.3} \
             precond_s={:.3} cg_solve_s={:.3} block_sweeps={} \
             precond_cost_ratio={:.3} recycling_admitted={} \
             mean_degree={:.1} giant_fraction={:.4} max_component={} \
             max_component_nnz={} operator_build_s={:.3} \
             cg_columns={} cg_iterations={} recycled_rank={} tile_columns={} \
             device_cols={} \
             cg_nonconverged={} cg_kappa_bound={:?} cg_relative_residual={:.3e}",
            epochs_run,
            config.max_epochs,
            next_ev,
            improve,
            ev_residual,
            decoder_residual,
            routing_residual,
            accepted_births,
            revived_atoms.len(),
            next_live_atoms,
            live_support.high_water,
            live_support.rounds_without_growth,
            support_saturated,
            refresh_secs,
            route_secs,
            fit_start.elapsed().as_secs_f64(),
            accumulate_secs,
            sigma_secs,
            decoder_solve_stats.graph_build_seconds,
            decoder_solve_stats.cg_preconditioner_seconds,
            decoder_solve_stats.cg_solve_seconds,
            decoder_solve_stats.cg_block_sweeps,
            decoder_solve_stats.cg_preconditioner_cost_ratio,
            decoder_solve_stats.cg_recycling_admitted,
            decoder_solve_stats.mean_cofiring_degree,
            decoder_solve_stats.giant_component_fraction,
            decoder_solve_stats.max_component_size,
            decoder_solve_stats.cg_max_component_nnz,
            decoder_solve_stats.cg_operator_build_seconds,
            decoder_solve_stats.cg_columns,
            decoder_solve_stats.cg_iterations,
            decoder_solve_stats.cg_recycled_rank,
            decoder_solve_stats.cg_min_tile_columns,
            decoder_solve_stats.device_refresh_columns,
            decoder_solve_stats.cg_nonconverged_columns,
            decoder_solve_stats.cg_kappa_bound,
            decoder_solve_stats.cg_relative_residual,
        );
        // #2275/#2023 trichotomy (mirrors `super::block`): a fit is returned when
        // the OBJECTIVE has settled — either at the absolute fixed point (arm 1,
        // certified) or at the achievable EV plateau with the discrete routing
        // still churning (arm 2, best-effort open at K >> rank). Only a
        // still-climbing objective — or a failed linear subsolve — is a genuine
        // non-convergence error (arm 3, below).
        //
        // An absolute certificate still requires zero accepted births. The open
        // arm additionally admits fixed-cardinality birth swaps only after live
        // support has stopped setting new highs for a full confirmation window;
        // the independent objective window below must plateau too. Raw
        // normal-equation convergence alone cannot certify a model because unit
        // projection and rerouting happen afterward.
        //
        // Soundness is a property of the ANSWER, not of which solver produced
        // it. A dense-factorization decline routes the same component through
        // block CG below; that solve certifies itself through its convergence
        // status and residual. The decline count is therefore path telemetry,
        // never an independent veto on a certified CG answer.
        let numerically_sound = decoder_solve_stats.cg_nonconverged_columns == 0
            && decoder_solve_stats.cg_relative_residual
                <= decoder_solve_stats.cg_residual_stop.max(f64::MIN_POSITIVE);
        let structure_settled = accepted_births == 0;

        // Arm 1 CERTIFIED: EV, decoder AND routing residuals all closed. Checked
        // first so an exactly-determined fit is certified, never demoted.
        let certified_fixed_point = structure_settled
            && numerically_sound
            && ev_residual <= fixed_point_tol
            && decoder_residual <= fixed_point_tol
            && routing_residual <= fixed_point_tol;

        // Arm 1 returns the state it just certified: every residual closed, so it
        // IS the fixed point and there is nothing better to look for.
        if certified_fixed_point {
            let (indices, code_mat) = pack_codes(&certified_codes, n, s);
            return Ok(SparseDictIterate {
                decoder: certified_decoder,
                indices,
                codes: code_mat,
                epochs: epochs_run,
                active: s,
                score_route_stats,
                decoder_solve_stats,
                inner_ev_residual: ev_residual,
                decoder_fixed_point_residual: decoder_residual,
                routing_residual,
                inner_tolerance: config.tolerance,
                accepted_births,
                live_atom_high_water: live_support.high_water,
                support_saturated,
                certified: true,
            });
        }

        // Arm 2 book-keeping. The state whose transition was just measured is the
        // candidate the open arm would hand back, so record it whenever it is the
        // best one seen and score the plateau on that running best (see
        // [`EvPlateau`] / [`BestOpenIterate`]). The gauge-invariant OBJECTIVE is
        // the convergence criterion; the decoder-gauge and routing residuals are
        // recorded, not gated on — at K >> rank the spurious support directions
        // rotate freely and the routing residual legitimately cannot close.
        if best_open
            .as_ref()
            .is_none_or(|best| certified_ev > best.explained_variance)
        {
            best_open = Some(BestOpenIterate {
                decoder: certified_decoder,
                codes: certified_codes,
                explained_variance: certified_ev,
                ev_residual,
                decoder_residual,
                routing_residual,
                accepted_births,
                support_saturated,
                decoder_solve_stats,
            });
        }
        let objective_plateaued = ev_plateau.observe(certified_ev, ev_residual, fixed_point_tol);
        // A round is STATIONARY when the structure is settled, the subsolve sound,
        // and the achievable objective stopped improving. Confirm a best-effort
        // plateau on MIN_ROUNDS stationary rounds within the trailing window
        // (tolerating one up-swing of the routing limit cycle; see
        // LINEAR_EV_PLATEAU_WINDOW). `epoch > 0` skips the first post-entry round,
        // whose climb denominator is still forming.
        let stationary = open_round_is_stationary(
            epoch,
            accepted_births,
            support_saturated,
            numerically_sound,
            objective_plateaued,
        );
        plateau_flags.push_back(stationary);
        while plateau_flags.len() > LINEAR_EV_PLATEAU_WINDOW {
            plateau_flags.pop_front();
        }
        let best_effort_open =
            plateau_flags.iter().filter(|&&s| s).count() >= LINEAR_EV_PLATEAU_MIN_ROUNDS;

        if best_effort_open {
            let best =
                best_open.expect("a confirmed plateau has observed at least one returnable round");
            let (indices, code_mat) = pack_codes(&best.codes, n, s);
            return Ok(SparseDictIterate {
                decoder: best.decoder,
                indices,
                codes: code_mat,
                epochs: epochs_run,
                active: s,
                score_route_stats,
                decoder_solve_stats: best.decoder_solve_stats,
                inner_ev_residual: best.ev_residual,
                decoder_fixed_point_residual: best.decoder_residual,
                routing_residual: best.routing_residual,
                inner_tolerance: config.tolerance,
                accepted_births: best.accepted_births,
                live_atom_high_water: live_support.high_water,
                support_saturated: best.support_saturated,
                certified: false,
            });
        }
        codes = std::mem::take(&mut next_codes);
        current_ev = next_ev;
    }

    Err(SparseDictionaryError::InnerNonConvergence {
        epochs: epochs_run,
        explained_variance: current_ev,
        ev_residual,
        tolerance: config.tolerance,
        accepted_births,
        decoder_fixed_point_residual: decoder_residual,
        routing_residual,
        solve_residual: decoder_solve_stats.cg_relative_residual,
        solve_tolerance: decoder_solve_stats.cg_residual_stop,
        decoder_nonconverged_columns: decoder_solve_stats.cg_nonconverged_columns,
        decoder_dense_cholesky_declines: decoder_solve_stats.dense_cholesky_declines,
    })
}

fn live_atom_count(codes: &[SparseCode], k: usize) -> usize {
    let mut alive = vec![false; k];
    for code in codes {
        for (slot, &atom) in code.indices.iter().enumerate() {
            if code.codes[slot] != 0.0 {
                alive[atom as usize] = true;
            }
        }
    }
    alive.iter().filter(|&&is_alive| is_alive).count()
}

/// The unified **linear fast kernel** (design gam#2232, Increment 2, plug points
/// 1–3): the fixed-support linear-atom (`d = 1`) inner solve of the ONE engine.
///
/// This is the seeded entry to the exact alternation of [`run_from_decoder`] —
/// `route → s×s active-set code solve → MOD sparse decoder refresh → unit-norm`
/// — parameterized by a SINGLE shared ridge coordinate `shared_rho` that feeds
/// BOTH
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
/// At `shared_rho = config.code_ridge = config.decoder_ridge` this kernel is
/// [`run_seeded`] itself (the unified config sets both ridges to the one shared ρ
/// and delegates); the TEMPORARY Increment-2 bit-parity gate that pinned this
/// identity during the migration was removed in Increment 6, the identity now
/// being structural. It is invoked from the unified
/// engine's inner-solve seam; [`super::fit_sparse_dictionary`] is the
/// shared-default entry to the REML schedule (Increment 5), and the single
/// public entry reaches it at ANY `K` through the explicit linear-dictionary
/// admission (`front_door::admit_linear_dictionary`, Increment 5b).
///
/// `decoder_recycle` is threaded from the caller so the decoder-recycle
/// break-even latch spans the whole fit rather than one inner solve (#2742);
/// the outer REML schedule reuses ONE space across all of its iterations.
pub(crate) fn run_linear_fast_kernel(
    x: ArrayView2<'_, f32>,
    config: &SparseDictConfig,
    shared_rho: f64,
    decoder_recycle: &mut DecoderRecycleSpace,
) -> Result<SparseDictIterate, SparseDictionaryError> {
    let mut unified = *config;
    // ONE shared variance coordinate drives both the code and the decoder ridge:
    // the `d = 1` specialization carries a single ρ, not two.
    unified.code_ridge = shared_rho as f32;
    unified.decoder_ridge = shared_rho as f32;
    run_seeded(x, &unified, decoder_recycle)
}

/// Continue the unified linear fast kernel from the preceding REML iterate.
///
/// The prior iterate is consumed, so its `K×P` decoder moves into the next inner
/// solve without a clone and its old-ridge packed codes are dropped. The shared
/// ridge is installed before [`run_from_decoder`] performs its mandatory fresh
/// route, making the continuation a warm start of model parameters only — never
/// a stale-code or stale-certificate shortcut (#2441).
fn continue_linear_fast_kernel(
    x: ArrayView2<'_, f32>,
    config: &SparseDictConfig,
    shared_rho: f64,
    prior: SparseDictIterate,
    decoder_recycle: &mut DecoderRecycleSpace,
) -> Result<SparseDictIterate, SparseDictionaryError> {
    let mut unified = *config;
    unified.code_ridge = shared_rho as f32;
    unified.decoder_ridge = shared_rho as f32;
    validate(x, &unified)?;

    let fit_start = Instant::now();
    let SparseDictIterate {
        decoder,
        indices,
        codes,
        ..
    } = prior;
    // The next inner run reroutes at its new ridge. Release the old packed state
    // before that corpus pass instead of retaining two N×s code sets at once.
    drop((indices, codes));
    let n = x.nrows();
    let p = x.ncols();
    let k = unified.n_atoms;
    let s = unified.active.min(k).max(1);
    log::warn!(
        "[SAE sparse_dict] continued prior decoder N={n} P={p} K={k} s={s} \
         (fresh route at rho={shared_rho:.6e} follows)"
    );
    run_from_decoder(x, &unified, decoder, decoder_recycle, fit_start)
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
/// runs, specialized to the `d = 1` linear block. Invalid or boundary evidence
/// is a typed error; returning the old `ρ` would manufacture a false zero outer
/// residual and is therefore forbidden.
pub fn linear_shared_rho_fs_step(
    stats: &LinearBlockRemlStats,
    rho: f64,
) -> Result<f64, SparseDictionaryError> {
    if !(rho.is_finite() && rho > 0.0) {
        return Err(SparseDictionaryError::InvalidRemlEvidence {
            reason: format!("rho must be finite and positive; got {rho}"),
        });
    }
    let gamma_tot = (stats.p_cols as f64) * stats.gram_edof;
    let total_obs = (stats.n_obs.saturating_mul(stats.p_cols)) as f64;
    if !(gamma_tot.is_finite() && gamma_tot > 0.0 && gamma_tot < total_obs) {
        return Err(SparseDictionaryError::InvalidRemlEvidence {
            reason: format!(
                "pooled effective dof must lie strictly inside (0, {total_obs}); got {gamma_tot}"
            ),
        });
    }
    if !(stats.rss.is_finite() && stats.rss >= 0.0) {
        return Err(SparseDictionaryError::InvalidRemlEvidence {
            reason: format!("RSS must be finite and non-negative; got {}", stats.rss),
        });
    }
    if !(stats.penalty_energy.is_finite() && stats.penalty_energy > 0.0) {
        return Err(SparseDictionaryError::InvalidRemlEvidence {
            reason: format!(
                "decoder penalty energy must be finite and positive; got {}",
                stats.penalty_energy
            ),
        });
    }
    let resid_dof = total_obs - gamma_tot;
    let sigma2 = stats.rss / resid_dof;
    let rho_new = gamma_tot * sigma2 / stats.penalty_energy;
    if !(rho_new.is_finite() && rho_new > 0.0) {
        return Err(SparseDictionaryError::InvalidRemlEvidence {
            reason: format!("Fellner-Schall update produced invalid rho {rho_new}"),
        });
    }
    Ok(rho_new)
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
) -> Result<f64, SparseDictionaryError> {
    if !(rho.is_finite() && rho > 0.0) {
        return Err(SparseDictionaryError::InvalidRemlEvidence {
            reason: format!("trace ridge must be finite and positive; got {rho}"),
        });
    }
    if k == 0 {
        return Ok(0.0);
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
    let residual_tolerance = decoder_solve_relative_tolerance();
    let chebyshev = 0.5 * root * (2.0 * root / residual_tolerance).ln();
    let cap = (chebyshev.max(0.0).ceil() as usize).min(k).max(1);

    // Probe count derived from the documented variance target: m = ⌈2/v⌉.
    let m_probes = (2.0 / EDOF_TRACE_VARIANCE_PER_UNIT_TRACE).ceil() as usize;
    let m_probes = m_probes.max(1);

    // Deterministic per-probe Rademacher signs from the crate's canonical
    // `splitmix64` mixer (NO `rand` crate): the probe bank is a content hash of
    // the Gram and K, deliberately independent of rho. Every outer iteration
    // therefore evaluates one deterministic fixed-point map rather than changing
    // its Monte-Carlo sample with the optimization coordinate.
    let mut base_seed = gam_linalg::utils::splitmix64_hash(k as u64);
    base_seed = gam_linalg::utils::splitmix64_hash(base_seed ^ (off.len() as u64).wrapping_add(1));
    for &d in diag.iter() {
        base_seed = gam_linalg::utils::splitmix64_hash(base_seed ^ d.to_bits());
    }

    let mut complementary_trace_acc = 0.0f64;
    for probe in 0..m_probes {
        let probe_salt =
            gam_linalg::utils::splitmix64_hash(base_seed ^ (probe as u64).wrapping_add(1));
        let mut z = vec![0.0f64; k];
        for (a, zi) in z.iter_mut().enumerate() {
            let h = gam_linalg::utils::splitmix64_hash(probe_salt ^ (a as u64).wrapping_add(1));
            // High bit of a well-mixed hash → an unbiased ±1 Rademacher draw.
            *zi = if h >> 63 == 0 { 1.0 } else { -1.0 };
        }
        let result = cg_solve(&matvec, &z, residual_tolerance, cap);
        if result.stop != CgStop::Converged {
            // Failure-path diagnostic only — never logged on the converged hot
            // path. (The Lanczos condition estimate now rides the production
            // block path's stats, not this scalar solve.)
            log::warn!(
                "sparse-dict Hutchinson trace probe {probe} CG non-convergence: \
                 iterations={} residual={:.3e}",
                result.iterations,
                result.relative_residual,
            );
            return Err(SparseDictionaryError::TraceNonConvergence {
                rho,
                probe,
                iterations: result.iterations,
                residual: result.relative_residual,
                tolerance: residual_tolerance,
            });
        }
        // zᵀ(A+ρI)⁻¹z; ρ·zᵀ(A+ρI)⁻¹z is one sample of tr(ρ(A+ρI)⁻¹).
        let zt_minv_z: f64 = z.iter().zip(result.x.iter()).map(|(zi, wi)| zi * wi).sum();
        complementary_trace_acc += rho * zt_minv_z;
    }
    let complementary_trace = complementary_trace_acc / m_probes as f64;
    let edof = k as f64 - complementary_trace;
    if !(edof.is_finite() && (0.0..=k as f64).contains(&edof)) {
        return Err(SparseDictionaryError::InvalidRemlEvidence {
            reason: format!(
                "Hutchinson effective dof must lie in [0, {k}]; got {edof} without clamping"
            ),
        });
    }
    Ok(edof)
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
fn reconstruction_rss_from_parts(
    x: ArrayView2<'_, f32>,
    decoder: ArrayView2<'_, f32>,
    indices: ArrayView2<'_, u32>,
    codes: ArrayView2<'_, f32>,
) -> f64 {
    let p = x.ncols();
    let s = indices.ncols();
    let mut rss = 0.0f64;
    let mut recon = vec![0.0f64; p];
    for i in 0..x.nrows() {
        for r in recon.iter_mut() {
            *r = 0.0;
        }
        for a in 0..s {
            let cj = codes[[i, a]] as f64;
            if cj == 0.0 {
                continue;
            }
            let drow = decoder.row(indices[[i, a]] as usize);
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

fn linear_block_reml_stats_from_parts(
    x: ArrayView2<'_, f32>,
    decoder: ArrayView2<'_, f32>,
    indices: ArrayView2<'_, u32>,
    codes: ArrayView2<'_, f32>,
    rho: f64,
) -> Result<LinearBlockRemlStats, SparseDictionaryError> {
    let k = decoder.nrows();
    let n = x.nrows();
    let p = x.ncols();
    let (diag, off) = code_gram_from_routing(indices, codes, k);
    let raw_gram_edof = hutchinson_gram_edof(&diag, &off, rho, k)?;
    // The true effective dof γ = tr(A(A+ρI)⁻¹) = Σ λ_i/(λ_i+ρ) is STRICTLY below
    // rank(A) ≤ min(K, N): each eigenterm is < 1 for ρ > 0, and the code Gram
    // A = CᵀC has rank ≤ rank(C) ≤ min(N, K). The internal estimate is already
    // clamped to [0, K]. The extra constraint that matters for the shared-ρ evidence
    // is that the pooled dof γ_tot = P·γ leave positive residual dof, i.e. γ < N.
    // When K ≥ N (the interpolating regime, γ → N) the Hutchinson estimate can
    // overshoot N by its sampling error, making γ_tot > N·P and tripping
    // `linear_shared_rho_fs_step`'s "pooled dof inside (0, N·P)" guard on pure
    // estimator noise (#2396: real OLMO fit at K=512 > N=508 read γ_tot = 32524.8 >
    // 32512). Cap γ at `N − 1/P`, so γ_tot stays strictly inside (0, N·P) and
    // σ̂² = RSS/(N·P − γ_tot) is well-posed — the interpolating fit then drives ρ
    // toward the identifiability boundary via the schedule's existing handling
    // instead of hard-erroring on a sampling overshoot. The cap is keyed to N, NOT
    // min(K, N): a non-interpolating fit (K < N) already has γ < K < N − 1/P via the
    // internal [0, K] clamp, so this NEVER binds there and its evidence is unchanged.
    let edof_ceiling = ((n as f64) - 1.0 / (p.max(1) as f64)).max(0.0);
    let gram_edof = raw_gram_edof.min(edof_ceiling);
    let penalty_energy: f64 = decoder.iter().map(|&d| (d as f64) * (d as f64)).sum();
    let rss = reconstruction_rss_from_parts(x, decoder, indices, codes);
    Ok(LinearBlockRemlStats {
        gram_edof,
        p_cols: p,
        penalty_energy,
        rss,
        n_obs: n,
    })
}

/// Log-rho stopping band for the shared-ρ REML schedule. A variance coordinate
/// perturbs a quadratic criterion to second order at its fixed point, so the
/// coordinate residual corresponding to an objective tolerance `ε` is `√ε`.
/// The arithmetic floor prevents requesting distinctions below f64 resolution.
fn reml_schedule_rho_log_tol(inner_tolerance: f64) -> f64 {
    inner_tolerance.sqrt().max(f64::EPSILON.sqrt())
}

/// Hard cap on the shared-ρ REML schedule's outer Fellner–Schall iterations
/// (#2396). The FS map ρ ↦ ρ_new is a contraction toward its fixed point for a
/// CERTIFIED inner solve, but a best-effort inner solve (certified = false, the
/// K >> rank routing limit cycle) makes the map NOISY: ρ oscillates within a band
/// about its interior fixed point and the per-step log-change is floored by the
/// inner EV-plateau noise. Without a cap the loop — which otherwise stops only at
/// `log_change ≤ tol` or the ρ→0 identifiability boundary — would not terminate on
/// non-interpolating over-complete data, whose ρ fixed point is INTERIOR (never
/// reaches the boundary) and whose step never falls below the machine-precision
/// band. The cap is a backstop: a well-behaved schedule settles within its
/// (best-effort-aware) band far sooner, so the cap is reached only when the FS map
/// is genuinely noise-floored, at which point the best-effort iterate is returned.
const REML_SCHEDULE_MAX_OUTER_ITERS: usize = 64;

/// The shared-ρ REML schedule (design gam#2232, Increment 2, plug 4): the outer
/// evidence loop that SELECTS the ONE shared linear-block ridge instead of taking
/// two magic constants. It seeds [`run_linear_fast_kernel`] once, then alternates
/// a consuming decoder continuation at the current ρ with one
/// [`linear_shared_rho_fs_step`] Fellner–Schall update built from the matrix-free
/// aggregates [`linear_block_reml_stats`], to the fixed point.
///
/// The initial ρ is the shared default ridge (`config.decoder_ridge`, equal to
/// `config.code_ridge` on the shared-default entry) — the historical magic
/// constant becomes only the WARM START of the evidence loop. Iteration stops
/// when the symmetric log-ρ change falls below the objective-derived floor
/// ([`reml_schedule_rho_log_tol`]) — at which point the current fit already
/// reflects a ρ within that band, so no redundant refit is issued. There is no
/// fixed pass count: an unsettled outer iterate is work, not a model.
///
/// The ONE [`DecoderRecycleSpace`] of the fit is constructed here and threaded
/// through every outer iteration (#2742): its break-even latch is documented as
/// one-way for the FIT, which is only true if the space outlives the inner
/// kernels this loop issues.
pub fn run_linear_reml_schedule(
    x: ArrayView2<'_, f32>,
    config: &SparseDictConfig,
) -> Result<SparseDictFit, SparseDictionaryError> {
    let mut decoder_recycle = DecoderRecycleSpace::new(config.n_atoms);
    run_linear_reml_schedule_with_recycle(x, config, &mut decoder_recycle)
}

/// [`run_linear_reml_schedule`] with the fit-scoped recycle space supplied by
/// the caller, so a test can observe that the outer loop never resets the
/// break-even latch it was handed.
/// How each inner solve obtained its decoder is not schedule-private control
/// flow: it is published on the fit as `seeded_inner_runs` /
/// `continued_inner_runs`, so the once-per-schedule farthest-point seed (#2441)
/// is a checkable property of every returned fit rather than of a test-only
/// observation seam with a production no-op sink (#2804).
fn run_linear_reml_schedule_with_recycle(
    x: ArrayView2<'_, f32>,
    config: &SparseDictConfig,
    decoder_recycle: &mut DecoderRecycleSpace,
) -> Result<SparseDictFit, SparseDictionaryError> {
    validate(x, config)?;
    if config.code_ridge != config.decoder_ridge {
        return Err(SparseDictionaryError::invalid_input(format!(
            "fit_sparse_dictionary has one shared REML ridge, so code_ridge ({}) and \
             decoder_ridge ({}) must be equal",
            config.code_ridge, config.decoder_ridge
        )));
    }
    let data_energy = x
        .iter()
        .map(|&value| (value as f64) * (value as f64))
        .sum::<f64>();
    if data_energy == 0.0 {
        // Analytic null boundary: no variance component is identifiable because
        // both signal and residual energy are exactly zero. The unique predictive
        // function is nevertheless known (zero), so return that certified null
        // model with rho at the null boundary instead of arbitrary seeded atoms.
        let active = config.active.min(config.n_atoms).max(1);
        let tolerance = reml_schedule_rho_log_tol(config.tolerance);
        return Ok(SparseDictFit {
            decoder: Array2::<f32>::zeros((config.n_atoms, x.ncols())),
            indices: Array2::<u32>::zeros((x.nrows(), active)),
            codes: Array2::<f32>::zeros((x.nrows(), active)),
            explained_variance: 1.0,
            epochs: 0,
            convergence: SparseDictConvergence {
                inner_ev_residual: 0.0,
                inner_tolerance: config.tolerance,
                decoder_residual: 0.0,
                decoder_tolerance: config.tolerance,
                routing_residual: 0.0,
                routing_tolerance: config.tolerance,
                outer_rho_residual: 0.0,
                outer_tolerance: tolerance,
                selected_rho: f64::INFINITY,
                outer_iterations: 0,
                seeded_inner_runs: 0,
                continued_inner_runs: 0,
                accepted_births: 0,
                live_atom_high_water: 0,
                support_saturated: false,
                certified: true,
            },
            active,
            score_route_stats: ScoreRouteStats::default(),
            decoder_solve_stats: DecoderSolveStats::default(),
        });
    }
    // Warm start at the caller's shared ridge; from here ρ is REML-selected.
    let mut rho = config.decoder_ridge as f64;
    // The caller's seed ridge is the interior anchor for the #2275 ρ-boundary
    // diagnosis (see the trace-failure arm below): a trace solve that fails only
    // AFTER Fellner–Schall has driven ρ STRICTLY below it is on the descent to the
    // identifiability edge, not an interior numerical failure.
    let initial_ridge = rho;
    let seeded_inner_runs = 1usize;
    let mut continued_inner_runs = 0usize;
    let mut fit = run_linear_fast_kernel(x, config, rho, decoder_recycle)?;
    let tol = reml_schedule_rho_log_tol(config.tolerance);
    let mut outer_iterations = 0usize;

    loop {
        outer_iterations += 1;
        let stats = match linear_block_reml_stats_from_parts(
            x,
            fit.decoder.view(),
            fit.indices.view(),
            fit.codes.view(),
            rho,
        ) {
            Ok(stats) => stats,
            Err(SparseDictionaryError::TraceNonConvergence { .. }) if rho < initial_ridge => {
                // #2275 OUTER ρ-BOUNDARY — a DERIVED identifiability-edge diagnosis,
                // not a rescue of a failed solve.
                //
                // The Fellner–Schall step is ρ_new = γ·σ² / ‖penalty‖ with
                // σ² = RSS / (N·P − γ). For an over-complete fit (K ≫ intrinsic
                // rank) the atoms interpolate, so RSS → 0 ⇒ σ² → 0 ⇒ ρ_new → 0:
                // the FS recursion on a rank-deficient design has NO interior fixed
                // point, it monotonically drives ρ to the zero boundary. That is a
                // property of the evidence on this data, not a transient.
                //
                // The edof γ = tr(A(A+ρI)⁻¹) of the code Gram A = CᵀC is estimated
                // by a Hutchinson trace CG. Over-complete ⇒ A is rank-deficient
                // (λ_min = 0), so cond(A+ρI) ≈ λ_max/ρ → ∞ as ρ → 0 and the CG,
                // which needs ≈ √cond iterations, cannot converge within its cap
                // once ρ ≲ λ_max / cap². So the trace becoming UNRESOLVABLE below
                // the seed ridge is the exact numerical signature of ρ having
                // reached the identifiability edge — the ρ-space image of the same
                // open frame the inner arm reports.
                //
                // best-effort OPEN is the honest classification here: the fit's
                // OBJECTIVE quantities are already converged (the inner run at this
                // ρ RETURNED — EV, decoder and routing residuals are all recorded);
                // the only thing that cannot be resolved is the edof trace, which is
                // a ρ-SELECTION diagnostic, not part of the reconstruction. There is
                // no interior ρ fixed point to certify, so we return the last
                // resolvable fit as open (certified = false, outer residual left at
                // ∞) instead of hard-erroring a converged reconstruction.
                //
                // `rho < initial_ridge` scopes this strictly to the descent to the
                // boundary; a trace failure AT OR ABOVE the seed ridge is off the
                // boundary (interior ρ) and remains a genuine numerical error that
                // still propagates.
                return schedule_fit_from_iterate(
                    x,
                    config,
                    fit,
                    false,
                    rho,
                    f64::INFINITY,
                    tol,
                    outer_iterations,
                    (seeded_inner_runs, continued_inner_runs),
                );
            }
            Err(err) => return Err(err),
        };
        let rho_new = linear_shared_rho_fs_step(&stats, rho)?;
        let log_change = (rho_new.ln() - rho.ln()).abs();
        // Per-iteration heartbeat on the warn channel (survives RUST_LOG=warn
        // harnesses), at outer-loop cadence only — never per row or minibatch.
        log::warn!(
            "[SAE reml-schedule iter {}] rho={:.6e} rho_new={:.6e} log_change={:.3e} \
             edof={:.2} rss={:.6e} penalty_energy={:.6e} tol={:.3e}",
            outer_iterations,
            rho,
            rho_new,
            log_change,
            stats.gram_edof,
            stats.rss,
            stats.penalty_energy,
            tol,
        );
        // Best-effort-aware stopping band (#2396). A CERTIFIED inner fixed point
        // pins ρ to the machine-precision band `tol = √(config.tolerance)`. But a
        // best-effort inner iterate (certified = false, K >> rank) carries an
        // EV-plateau residual `inner_ev_residual ≫ config.tolerance` that the FS map
        // amplifies into the ρ step, so ρ cannot be resolved tighter than
        // `√(inner_ev_residual)`. Widen the band to that HONEST floor for an open
        // fit — the same √ objective-to-ρ relationship, evaluated at the ACHIEVED
        // inner precision rather than the requested one — so the schedule settles at
        // the achievable ρ precision instead of grinding a noise-floored step.
        let effective_tol = if fit.certified {
            tol
        } else {
            tol.max(reml_schedule_rho_log_tol(fit.inner_ev_residual))
        };
        if log_change <= effective_tol {
            // The current `fit` was produced at `rho`, which is within the band of
            // `rho_new`: it already reflects the fixed point. Stop without a
            // redundant refit. The inner fit's certificate propagates: a
            // best-effort-open inner iterate (#2275, K >> rank) yields a
            // best-effort-open schedule fit.
            let certified = fit.certified;
            return schedule_fit_from_iterate(
                x,
                config,
                fit,
                certified,
                rho,
                log_change,
                effective_tol,
                outer_iterations,
                (seeded_inner_runs, continued_inner_runs),
            );
        }
        if outer_iterations >= REML_SCHEDULE_MAX_OUTER_ITERS {
            // Termination guarantee (#2396): a best-effort inner solve makes the FS
            // map noisy, so ρ oscillates within a band about its interior fixed
            // point and even the widened band may never be met on a single step.
            // Return the current best-effort iterate with its ρ residual recorded
            // honestly (open certificate), rather than looping unboundedly. A
            // CERTIFIED schedule cannot reach here — its map contracts and meets the
            // band first.
            return schedule_fit_from_iterate(
                x,
                config,
                fit,
                false,
                rho,
                log_change,
                effective_tol,
                outer_iterations,
                (seeded_inner_runs, continued_inner_runs),
            );
        }
        rho = rho_new;
        continued_inner_runs += 1;
        fit = continue_linear_fast_kernel(x, config, rho, fit, decoder_recycle)?;
    }
}

/// Assemble the public [`SparseDictFit`] from a settled inner iterate and the
/// outer REML schedule's certificate. Shared by the schedule's two exits — the
/// converged ρ fixed point (`certified` carries the inner arm's verdict) and the
/// #2275 ρ-boundary best-effort return (`certified = false`, `outer_rho_residual`
/// left open at `∞`).
///
/// API honesty (#2275): the returned EV must be the EV of the returned MODEL, and
/// the returned codes must be exactly what a caller reconstructs by routing the
/// returned decoder at the code ridge THEY passed — which is how held-out
/// prediction routes (see `held_out_ev`). But the inner REML runs override the
/// code ridge with the shared variance ρ (`run_linear_fast_kernel`: one ρ drives
/// both the code and decoder ridge, so the evidence sees a single variance), so
/// the inner iterate's packed codes and cached EV are at ρ, not at the caller's
/// `config.code_ridge`; the dead-atom nulling can stale them further. Re-route the
/// codes ONCE against the returned decoder at the caller's `config.code_ridge` and
/// report THAT model's EV, so the returned artifact is self-consistent with how it
/// will be used. One route + one EV, at the schedule's single exit — never per
/// epoch, so not on the hot path.
fn schedule_fit_from_iterate(
    x: ArrayView2<'_, f32>,
    config: &SparseDictConfig,
    fit: SparseDictIterate,
    certified: bool,
    selected_rho: f64,
    outer_rho_residual: f64,
    outer_tolerance: f64,
    outer_iterations: usize,
    inner_runs: (usize, usize),
) -> Result<SparseDictFit, SparseDictionaryError> {
    let (seeded_inner_runs, continued_inner_runs) = inner_runs;
    let n = x.nrows();
    let s = fit.active;
    let scorer = TileScorer::new(s, config.score_tile);
    let final_codes = route_and_code_all(
        x,
        fit.decoder.view(),
        &scorer,
        s,
        config.code_ridge,
        config.minibatch,
        config.score_mode,
        None,
    )?;
    let final_ev = explained_variance(x, &final_codes, fit.decoder.view());
    let final_live_atoms = live_atom_count(&final_codes, config.n_atoms);
    let (indices, codes) = pack_codes(&final_codes, n, s);
    Ok(SparseDictFit {
        convergence: SparseDictConvergence {
            inner_ev_residual: fit.inner_ev_residual,
            inner_tolerance: fit.inner_tolerance,
            decoder_residual: fit.decoder_fixed_point_residual,
            decoder_tolerance: fit.inner_tolerance,
            routing_residual: fit.routing_residual,
            routing_tolerance: fit.inner_tolerance,
            outer_rho_residual,
            outer_tolerance,
            selected_rho,
            outer_iterations,
            seeded_inner_runs,
            continued_inner_runs,
            accepted_births: fit.accepted_births,
            live_atom_high_water: fit.live_atom_high_water.max(final_live_atoms),
            support_saturated: fit.support_saturated,
            certified,
        },
        decoder: fit.decoder,
        indices,
        codes,
        explained_variance: final_ev,
        epochs: fit.epochs,
        active: fit.active,
        score_route_stats: fit.score_route_stats,
        decoder_solve_stats: fit.decoder_solve_stats,
    })
}

fn validate(
    x: ArrayView2<'_, f32>,
    config: &SparseDictConfig,
) -> Result<(), SparseDictionaryError> {
    if x.nrows() == 0 || x.ncols() == 0 {
        return Err(SparseDictionaryError::invalid_input(
            "fit_sparse_dictionary requires a non-empty N×P matrix",
        ));
    }
    if !x.iter().all(|v| v.is_finite()) {
        return Err(SparseDictionaryError::invalid_input(
            "fit_sparse_dictionary input must be finite",
        ));
    }
    if config.n_atoms == 0 {
        return Err(SparseDictionaryError::invalid_input(
            "fit_sparse_dictionary requires K >= 1",
        ));
    }
    if config.active == 0 {
        return Err(SparseDictionaryError::invalid_input(
            "fit_sparse_dictionary requires active (top_s) >= 1",
        ));
    }
    if config.max_epochs == 0 {
        return Err(SparseDictionaryError::invalid_input(
            "fit_sparse_dictionary requires max_epochs >= 1",
        ));
    }
    if !(config.code_ridge.is_finite() && config.code_ridge > 0.0) {
        return Err(SparseDictionaryError::invalid_input(
            "fit_sparse_dictionary code_ridge must be finite and positive",
        ));
    }
    if !(config.decoder_ridge.is_finite() && config.decoder_ridge > 0.0) {
        return Err(SparseDictionaryError::invalid_input(
            "fit_sparse_dictionary decoder_ridge must be finite and positive",
        ));
    }
    if !(config.tolerance.is_finite() && config.tolerance >= 0.0) {
        return Err(SparseDictionaryError::invalid_input(
            "fit_sparse_dictionary tolerance must be finite and non-negative",
        ));
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
                        if b.1 > a.1 || (b.1 == a.1 && b.0 < a.0) {
                            b
                        } else {
                            a
                        }
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
        // Scalar and coupling statistics: one serial walk (cheap relative to
        // the `O(N·s·P)` right-hand-side accumulation below).
        for code in codes.iter() {
            for a in 0..code.indices.len() {
                let ca = code.codes[a] as f64;
                if ca == 0.0 {
                    continue;
                }
                let ka = code.indices[a];
                self.firings[ka as usize] += 1;
                self.amplitude_sum[ka as usize] += ca.abs();
                self.diag[ka as usize] += ca * ca;
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
        // `B += CᵀX`, parallelized over DISJOINT atom-row blocks of `B`: each
        // task owns a contiguous block of atom rows and replays the full
        // (row, atom) walk in the same ascending order the serial loop used,
        // touching only the atoms in its block. Every `B[k][c]` entry
        // therefore accumulates its contributions in exactly the historical
        // order — bit-identical regardless of the partition — while the
        // epoch's dominant `O(N·s·P)` flops use every core (#1017: this pass
        // was part of the serial epoch wall next to the idle accelerator).
        // The re-walked code indices are `O(N·s)` per block, noise against
        // the `O(N·s·P)` right-hand-side arithmetic they gate.
        if p == 0 {
            return;
        }
        let k_atoms = self.diag.len();
        let atom_block = k_atoms.div_ceil(ACCUMULATE_ATOM_BLOCKS).max(1);
        let b_slice = self
            .b
            .as_slice_mut()
            .expect("normal-equation rhs is standard layout");
        b_slice
            .par_chunks_mut(atom_block * p)
            .enumerate()
            .for_each(|(block_idx, bchunk)| {
                let k0 = block_idx * atom_block;
                let k1 = k0 + bchunk.len() / p;
                for (row_idx, code) in codes.iter().enumerate() {
                    let xi = x.row(row_idx);
                    let xi_slice = xi.as_slice();
                    for a in 0..code.indices.len() {
                        let ca = code.codes[a] as f64;
                        if ca == 0.0 {
                            continue;
                        }
                        let ka = code.indices[a] as usize;
                        if ka < k0 || ka >= k1 {
                            continue;
                        }
                        let brow = &mut bchunk[(ka - k0) * p..(ka - k0 + 1) * p];
                        match xi_slice {
                            Some(xs) => {
                                for (bref, &xv) in brow.iter_mut().zip(xs.iter()) {
                                    *bref += ca * xv as f64;
                                }
                            }
                            None => {
                                for (c, bref) in brow.iter_mut().enumerate() {
                                    *bref += ca * xi[c] as f64;
                                }
                            }
                        }
                    }
                }
            });
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

/// Cross-refresh coarse space for the giant decoder solve.
///
/// A decoder correction is `δD = (A + ρI)⁻¹(B - (A + ρI)D₀)`: it amplifies
/// precisely the low-energy non-diagonal modes that make a warm Jacobi solve
/// expensive. We retain a memory-bounded reservoir of the previous refresh's
/// globally hardest correction columns, rank-reveal it against each current
/// component, and rebuild its exact Galerkin operator against the current normal
/// equations. This is Krylov recycling without retaining a second copy of the
/// normal equations or an epoch-specific factorization.
///
/// Directions use global atom coordinates so component splits/merges are safe:
/// every current component restricts and re-orthogonalizes them before use.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
struct DecoderRecyclePriority {
    /// Dominant scalar operator work spent on this column in the prior solve.
    operator_work: u128,
    iterations: usize,
    component_anchor: usize,
    column: usize,
}

impl Ord for DecoderRecyclePriority {
    fn cmp(&self, other: &Self) -> std::cmp::Ordering {
        self.operator_work
            .cmp(&other.operator_work)
            .then_with(|| self.iterations.cmp(&other.iterations))
            // Canonical smaller identifiers win exact work ties.
            .then_with(|| other.component_anchor.cmp(&self.component_anchor))
            .then_with(|| other.column.cmp(&self.column))
    }
}

impl PartialOrd for DecoderRecyclePriority {
    fn partial_cmp(&self, other: &Self) -> Option<std::cmp::Ordering> {
        Some(self.cmp(other))
    }
}

struct DecoderRecycleCandidate {
    direction: Vec<f64>,
    priority: DecoderRecyclePriority,
}

pub(super) struct DecoderRecycleSpace {
    rows: usize,
    directions: Vec<Vec<f64>>,
    next_candidates: Vec<DecoderRecycleCandidate>,
    next_capacity: usize,
    /// Operator applications per decoder column observed at a rank-ZERO
    /// (plain Jacobi) refresh of this fit — the baseline the recycled
    /// correction has to beat. The first refresh always supplies it: there is
    /// no history yet, so it necessarily runs at rank 0.
    jacobi_sweeps_per_column: Option<f64>,
    /// Whether the recycled correction is still admitted. See
    /// [`Self::score_refresh`].
    admitted: bool,
}

impl DecoderRecycleSpace {
    pub(super) fn new(rows: usize) -> Self {
        Self {
            rows,
            directions: Vec::new(),
            next_candidates: Vec::new(),
            next_capacity: 0,
            jacobi_sweeps_per_column: None,
            admitted: true,
        }
    }

    /// Whether a recycled correction may be built for the next component.
    ///
    /// Once [`Self::score_refresh`] has measured the correction failing to pay
    /// for itself on THIS fit's operator, this is `false` for the rest of the
    /// fit and every remaining refresh runs plain Jacobi block CG. "The fit"
    /// means the whole fit, across outer REML iterations, which holds only
    /// because the schedule owns the space above [`run_from_decoder`] (#2742)
    /// and [`Self::begin_fit`] preserves the latch.
    fn admitted(&self) -> bool {
        self.admitted
    }

    /// Start one inner alternation ([`run_from_decoder`]) on a caller-owned space.
    ///
    /// The coarse-space HISTORY is per inner run: the directions were measured
    /// on the operator at the previous ρ, and the break-even scoring requires
    /// the first refresh of a run to have no history so it supplies its own
    /// rank-zero Jacobi baseline. The break-even DECISION is per fit and is
    /// deliberately not cleared here — re-probing a correction already measured
    /// as a loss is exactly the cost the latch exists to avoid (#2742).
    fn begin_fit(&mut self, rows: usize) {
        assert_eq!(
            self.rows, rows,
            "decoder recycle space must retain the dictionary row dimension across the fit"
        );
        self.directions.clear();
        self.next_candidates.clear();
        self.next_capacity = 0;
    }

    /// Decide whether the recycled correction earned its cost on the refresh
    /// that just finished, and latch it off if it did not.
    ///
    /// # Why this is measured rather than bounded
    ///
    /// [`decoder_recycle_rank_bound`] permits a rank whose apply work is up to
    /// ONE extra operator application per sweep (`2mr ≤ m+nnz`). That is a
    /// bound on what the correction may COST; nothing anywhere bounded what it
    /// had to RETURN. So a correction that finds no useful direction is not
    /// merely unhelpful — it is a guaranteed loss of up to a factor of two on
    /// the dominant term of the refresh, taken silently.
    ///
    /// Measured on the sparse-dictionary decoder operator (`sae_decoder_refresh_scaling`,
    /// same node, same binary, recycling forced to rank 0 as the only change):
    /// the correction reduced operator applications by **0.3%-1.4%** across
    /// five shapes spanning `P ∈ [256, 2048]` and `K ∈ [1024, 2048]`, while the
    /// refresh itself ran **1.68×-2.89× slower**. Per-column iteration counts
    /// were 22-29 with the correction and 23-29 without it, with zero
    /// non-converged columns either way: the Jacobi-scaled operator is already
    /// well enough conditioned that there is nothing for a coarse space to
    /// remove. That is a property of this operator, not of the recycling
    /// machinery, which is why the decision is taken from THIS fit's own
    /// measurement instead of deleting the capability — a genuinely
    /// ill-conditioned decoder still gets it, and keeps it.
    ///
    /// The break-even is the correction's own cost ratio, not a tuned number.
    /// A rank-`r` correction adds `cost_ratio = 2mr/(m+nnz)` operator-equivalents
    /// to every sweep, so it is worth having exactly when
    /// `sweeps_recycled · (1 + cost_ratio) ≤ sweeps_jacobi`.
    ///
    /// The latch is one-way on purpose. Re-probing would reintroduce the cost
    /// it exists to avoid on every probe, and the measurement above is not
    /// marginal — nothing about a fit's later epochs makes a coarse space that
    /// removed 1% of the work start removing 50% of it.
    ///
    /// One-way means one-way for the FIT, and that is a statement about who
    /// OWNS this struct, not about this method (#2742). While the space was
    /// constructed inside the seeded runner, the latch reset at every outer REML
    /// iteration boundary — [`run_linear_reml_schedule`] can evaluate up to
    /// [`REML_SCHEDULE_MAX_OUTER_ITERS`] inner runs, and a correction already
    /// proved a loss was rebuilt and re-rejected on each one (measured:
    /// readmitted at refreshes 8, 13 and 20 of one fit; 4.504 s of 31.920 s of
    /// refresh time spent on corrections, 14 of 14 rejected). The space is now
    /// created once per fit by the schedule and threaded through every inner
    /// run, so the scope of this decision is the scope this comment claims.
    fn score_refresh(&mut self, sweeps: usize, columns: usize, rank: usize, cost_ratio: f64) {
        if columns == 0 || sweeps == 0 {
            return;
        }
        let per_column = sweeps as f64 / columns as f64;
        if rank == 0 {
            self.jacobi_sweeps_per_column = Some(match self.jacobi_sweeps_per_column {
                Some(previous) => previous.min(per_column),
                None => per_column,
            });
            return;
        }
        if let Some(baseline) = self.jacobi_sweeps_per_column
            && per_column * (1.0 + cost_ratio) > baseline
        {
            self.admitted = false;
        }
    }

    fn begin_refresh(&mut self, rows: usize) {
        assert_eq!(
            self.rows, rows,
            "decoder recycle space must retain the dictionary row dimension"
        );
        self.next_candidates.clear();
        self.next_capacity = 0;
    }

    fn finish_refresh(&mut self) {
        if !self.next_candidates.is_empty() {
            self.next_candidates
                .sort_by(|left, right| right.priority.cmp(&left.priority));
            self.directions = self
                .next_candidates
                .drain(..)
                .map(|candidate| candidate.direction)
                .collect();
        }
        self.next_capacity = 0;
    }

    /// Add one solved correction after weighted normalization.
    ///
    /// The `D` inner product is the Euclidean inner product after Jacobi
    /// scaling (`y = D¹/²x`), exactly the coordinate used by the coarse
    /// preconditioner. A bounded max-priority reservoir prevents an early
    /// component from exhausting the refresh-wide history budget: later
    /// candidates replace weaker entries by observed operator work, with stable
    /// component/column tie breaks. Rank revelation is deferred to the next
    /// current-operator setup, where one BLAS-3 pivoted QR handles all selected
    /// columns together (including dependencies introduced by graph
    /// splits/merges).
    fn retain_component_correction(
        &mut self,
        comp: &[usize],
        diagonal: &[f64],
        correction: &[f64],
        rank_bound: usize,
        operator_entries: usize,
        iterations: usize,
        column: usize,
    ) {
        self.next_capacity = self.next_capacity.max(rank_bound);
        if self.next_capacity == 0
            || comp.is_empty()
            || correction.len() != comp.len()
            || diagonal.len() != comp.len()
        {
            return;
        }
        let norm = correction
            .iter()
            .zip(diagonal.iter())
            .map(|(&x, &d)| d * x * x)
            .sum::<f64>()
            .sqrt();
        if !norm.is_finite() || norm == 0.0 {
            return;
        }
        let mut global = vec![0.0f64; self.rows];
        for (i, &atom) in comp.iter().enumerate() {
            global[atom] = correction[i] / norm;
        }
        let priority = DecoderRecyclePriority {
            operator_work: (operator_entries as u128).saturating_mul(iterations as u128),
            iterations,
            component_anchor: comp[0],
            column,
        };
        let candidate = DecoderRecycleCandidate {
            direction: global,
            priority,
        };
        if self.next_candidates.len() < self.next_capacity {
            self.next_candidates.push(candidate);
            return;
        }

        let weakest = self
            .next_candidates
            .iter()
            .enumerate()
            .min_by(|(_, left), (_, right)| left.priority.cmp(&right.priority))
            .map(|(index, _)| index)
            .expect("positive recycle capacity has a full non-empty reservoir");
        if priority > self.next_candidates[weakest].priority {
            self.next_candidates[weakest] = candidate;
        }
    }
}

/// An atom is "dead" this epoch when its regularised self-energy `A_kk + ρ` is
/// at or below this floor: it never fired (and, since couplings require two
/// non-zero codes, it is then necessarily isolated). Such atoms keep their
/// seeded direction so a later epoch can still route rows to them.
pub(super) const DEAD_DENOM: f64 = 1.0e-12;

/// Dimensionless residual target for the f64 normal-equation solve. This is the
/// square root of unit roundoff: below it, the residual norm is dominated by the
/// dot products used to evaluate that norm. Crucially it is independent of the
/// REML ridge `ρ`; regularisation changes conditioning, never what “solved” means.
fn decoder_solve_relative_tolerance() -> f64 {
    f64::EPSILON.sqrt()
}

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
    /// Stored off-diagonal entries in the largest CG component's restricted
    /// operator. `max_component_size` reports how many atoms the giant
    /// component holds; this reports how much operator there is to apply, and
    /// the two move independently — a birth that rewires the co-firing graph
    /// can leave the atom count flat while densifying the coupling. Every
    /// block-CG iteration walks this structure once, so it is the work per
    /// iteration (#2441).
    pub cg_max_component_nnz: usize,
    /// Seconds spent assembling the restricted CSR operators, summed over the
    /// CG-solved components of one refresh. Separated from the solve so a
    /// refresh that inflates can be attributed to building the operator versus
    /// iterating on it, rather than inferred from the correlation with births.
    pub cg_operator_build_seconds: f64,
    /// Largest recycled Galerkin coarse-space rank used by a CG component.
    /// Zero is the first-refresh / no-independent-history case.
    pub cg_recycled_rank: usize,
    /// Largest condition estimate recovered from CG's Lanczos tridiagonal.
    pub cg_kappa_hat: Option<f64>,
    /// Largest final relative normal-equation residual among CG solves.
    pub cg_relative_residual: f64,
    /// Dimensionless relative residual threshold used by CG, derived solely from
    /// f64 arithmetic precision and independent of the model ridge.
    pub cg_residual_stop: f64,
    /// Decoder columns whose CG did NOT reach the charge floor before the
    /// conditioning-derived iteration cap (or broke down on a non-SPD step).
    /// Non-zero means at least one giant-scale co-firing block was too
    /// ill-conditioned to solve to tolerance. Its previous decoder column is
    /// retained and the enclosing fit reports TYPED non-convergence instead of
    /// installing a substitute or spinning silently.
    pub cg_nonconverged_columns: usize,
    /// Dense-component Cholesky shortcuts that declined. The component is routed
    /// through block CG instead, so this is path telemetry; CG convergence and
    /// its residual determine whether the resulting answer is sound.
    pub dense_cholesky_declines: usize,
    /// Decoder columns whose block-CG solve ran on the device-resident CUDA
    /// backend. `0` on a CPU-only refresh — surfaced in the epoch heartbeat so
    /// a "device-resident" label can never silently cover a CPU reality
    /// (#1017's recurring misreporting pattern).
    pub device_refresh_columns: usize,
    /// Largest a-priori Gershgorin condition-number bound for the Jacobi-scaled
    /// operator over the CG-solved components. It sizes the first-refresh
    /// Chebyshev cap and remains a directly comparable spectrum-drift diagnostic
    /// after recycling engages; recycled solves use the algebraic `m`-step
    /// finite-termination bound because this Jacobi bound no longer describes
    /// their low-rank-corrected operator.
    pub cg_kappa_bound: Option<f64>,
    /// Narrowest block-CG column tile any component of this refresh solved on.
    ///
    /// This is the width that decides whether the refresh is running the BLOCK
    /// recurrence #1017 landed or the per-column one it replaced: the tile is
    /// what the operator traversal and the three inner products are amortized
    /// over, and it is not a constant — `solve_component` derives it from what
    /// the recycled history left of the `K×P` envelope. A refresh whose tile has
    /// collapsed toward 1 pays the per-column price with the block solve's name
    /// on it, which is invisible on every other field of this struct
    /// (#2283/#2441: 5.4×-26× refresh inflation at flat `nnz` and flat
    /// iteration count). `0` means no component reached the CG path.
    pub cg_min_tile_columns: usize,
    /// Seconds spent building the symmetric co-firing adjacency and walking its
    /// connected components — everything between the assembled normal equations
    /// and the first component solve.
    ///
    /// This is one of the four phases the refresh wall can hide in, and the
    /// only one that is unavoidably serial over `|E|`: the adjacency fill walks
    /// every stored coupling into per-atom vectors, and the BFS walks them
    /// again. Reported so that "the refresh is slow" can be attributed instead
    /// of inferred (#2283/#2441).
    pub graph_build_seconds: f64,
    /// Seconds spent constructing the recycled Galerkin preconditioner —
    /// restriction of the retained directions, the rank-revealing QR, the fresh
    /// operator action and the SPD proof — summed over CG-solved components.
    ///
    /// Separated from [`Self::cg_solve_seconds`] because the preconditioner is
    /// the OPTIONAL half of the solve: if this is a large share of the refresh,
    /// or if the solve it accelerates does not shrink by more than this costs,
    /// the accelerator is a net loss and the answer is to stop paying for it,
    /// not to make it cheaper.
    pub cg_preconditioner_seconds: f64,
    /// Seconds inside the block-CG recurrence itself (`solve_block_cg`), summed
    /// over components and column tiles. Excludes the operator CSR build
    /// ([`Self::cg_operator_build_seconds`]) and the preconditioner build.
    pub cg_solve_seconds: f64,
    /// Operator applications the block recurrence actually performed: summed
    /// over column tiles, the MAXIMUM per-column iteration count in each tile.
    ///
    /// [`Self::cg_iterations`] sums per-column iteration counts, which is the
    /// right convergence statistic and the WRONG work statistic: the block
    /// advances every column of a tile together, so a tile whose hardest column
    /// needs 200 iterations applies the operator 200 times no matter how fast
    /// its other columns converged, and the sum scales with the column count on
    /// top of that. This field is the exact number of `A·P` applications, which
    /// is what a traffic or flop model of the refresh has to be denominated in.
    pub cg_block_sweeps: usize,
    /// Extra operator applications per sweep that the recycled correction adds,
    /// `2mr/(m+nnz)`, maximised over the CG-solved components of this refresh.
    ///
    /// This is the price the correction charges. Paired with
    /// [`Self::cg_block_sweeps`] it is the whole break-even test: a correction
    /// costing `cost_ratio` must remove a `cost_ratio/(1+cost_ratio)` share of
    /// the sweeps to be worth building, and until this field existed nothing
    /// on the trace let anyone check whether it did.
    pub cg_preconditioner_cost_ratio: f64,
    /// Whether the recycled correction is still admitted for this fit. Flips to
    /// `false` for good on the first refresh where it fails its break-even.
    pub cg_recycling_admitted: bool,
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
            cg_max_component_nnz: 0,
            cg_operator_build_seconds: 0.0,
            cg_recycled_rank: 0,
            cg_kappa_hat: None,
            cg_relative_residual: 0.0,
            cg_residual_stop: 0.0,
            cg_nonconverged_columns: 0,
            dense_cholesky_declines: 0,
            device_refresh_columns: 0,
            cg_kappa_bound: None,
            cg_min_tile_columns: 0,
            graph_build_seconds: 0.0,
            cg_preconditioner_seconds: 0.0,
            cg_solve_seconds: 0.0,
            cg_block_sweeps: 0,
            cg_preconditioner_cost_ratio: 0.0,
            cg_recycling_admitted: true,
        }
    }
}

impl DecoderSolveStats {
    /// Fold one block-CG column certificate ([`PcgCoreResult`]) into the
    /// refresh statistics — the same accounting the historical per-column
    /// `cg_solve` recording performed (`MaxIters`/`Breakdown` both count as a
    /// non-converged column; `kappa_hat` is the Lanczos estimate from the
    /// column's own `alpha`/`beta` trace).
    fn record_block_column(&mut self, core: &PcgCoreResult, kappa_hat: Option<f64>) {
        self.cg_columns += 1;
        self.cg_iterations += core.iterations;
        let relative_residual = if core.rhs_norm > 0.0 {
            core.final_residual_norm / core.rhs_norm
        } else {
            0.0
        };
        self.cg_relative_residual = self.cg_relative_residual.max(relative_residual);
        if core.stop != PcgStop::Converged {
            self.cg_nonconverged_columns += 1;
        }
        if let Some(kappa) = kappa_hat {
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

pub(super) fn solve_decoder_with_routability_gate_recycled(
    decoder: &mut Array2<f32>,
    eq: &DecoderNormalEq,
    ridge: f64,
    residual_scale: f64,
    gpu: gam_gpu::GpuPolicy,
    recycle: &mut DecoderRecycleSpace,
) -> Result<(DecoderSolveStats, Vec<RoutabilityGateDecision>), String> {
    let gate = routability_gate_decisions(eq, residual_scale);
    let mut candidate = decoder.clone();
    let stats = solve_decoder_recycled(&mut candidate, eq, ridge, gpu, recycle)?;
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
    Ok((stats, gate))
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
/// Returns the atom indices whose residual-row birth proposals were installed.
/// The fresh route decides which proposals are accepted; convergence requires
/// zero accepted births, not zero proposals (the latter is impossible whenever
/// `K > N·s`).
fn revive_dead_atoms(
    x: ArrayView2<'_, f32>,
    codes: &[SparseCode],
    decoder: &mut Array2<f32>,
) -> Vec<usize> {
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
        return Vec::new();
    }

    // Per-row residual under the current model, and its squared norm for
    // ranking. Every value here is row-local (the norm fold never crosses
    // rows), so parallelizing over rows is bit-identical to the historical
    // serial pass.
    let mut resid = Array2::<f32>::zeros((n, p));
    let mut resid_norm2 = vec![0.0f64; n];
    let decoder_view = decoder.view();
    resid
        .as_slice_mut()
        .expect("freshly allocated residual block is standard layout")
        .par_chunks_mut(p)
        .zip(resid_norm2.par_iter_mut())
        .enumerate()
        .for_each(|(i, (ri, norm2))| {
            let xi = x.row(i);
            for c in 0..p {
                ri[c] = xi[c];
            }
            let code = &codes[i];
            for j in 0..code.indices.len() {
                let cj = code.codes[j];
                if cj == 0.0 {
                    continue;
                }
                let drow = decoder_view.row(code.indices[j] as usize);
                for c in 0..p {
                    ri[c] -= cj * drow[c];
                }
            }
            let mut acc = 0.0f64;
            for c in 0..p {
                acc += ri[c] as f64 * ri[c] as f64;
            }
            *norm2 = acc;
        });

    // Rows ranked by descending residual energy (ties by ascending index →
    // deterministic). Only rows with real residual can seed a useful atom.
    let mut order: Vec<usize> = (0..n).collect();
    order.sort_by(|&a, &b| {
        resid_norm2[b]
            .partial_cmp(&resid_norm2[a])
            .unwrap_or(std::cmp::Ordering::Equal)
            .then_with(|| a.cmp(&b))
    });

    let mut revived = Vec::new();
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
        revived.push(atom);
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
fn solve_decoder_recycled(
    decoder: &mut Array2<f32>,
    eq: &DecoderNormalEq,
    ridge: f64,
    gpu: gam_gpu::GpuPolicy,
    recycle: &mut DecoderRecycleSpace,
) -> Result<DecoderSolveStats, String> {
    let k = eq.diag.len();
    let p = eq.b.ncols();
    recycle.begin_refresh(k);

    let graph_build_start = Instant::now();
    // Symmetric coupling adjacency, sorted per atom for deterministic assembly.
    let mut neigh: Vec<Vec<(u32, f64)>> = vec![Vec::new(); k];
    for (&(a, b), &val) in eq.off.iter() {
        neigh[a as usize].push((b, val));
        neigh[b as usize].push((a, val));
    }
    // Per-atom adjacency sorts are independent; parallelizing them changes
    // nothing about the (deterministic) sorted result. At production scale
    // this is ~2|E| entries across K lists - a measurable serial slice of
    // the refresh now that the solve itself is fast (#1017).
    neigh.par_iter_mut().for_each(|list| {
        list.sort_by_key(|&(nb, _)| nb);
    });
    let adjacency_seconds = graph_build_start.elapsed().as_secs_f64();

    let mut stats = DecoderSolveStats {
        mean_cofiring_degree: if k == 0 {
            0.0
        } else {
            2.0 * eq.off.len() as f64 / k as f64
        },
        cg_residual_stop: decoder_solve_relative_tolerance(),
        ..DecoderSolveStats::default()
    };

    // Exact dense Cholesky is confined to components below the percolation
    // critical-component scale; everything larger is a giant-scale block solved
    // matrix-free by CG (see `direct_solve_size_threshold`).
    let direct_threshold = direct_solve_size_threshold(k);

    stats.graph_build_seconds = adjacency_seconds;
    // The BFS below is interleaved with the solves, so its cost is accumulated
    // per component rather than measured as one span.
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
        let component_walk_start = Instant::now();
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
        stats.graph_build_seconds += component_walk_start.elapsed().as_secs_f64();
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
            gpu,
            &mut stats,
            recycle,
        )?;
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
         components={} max_component={} max_component_nnz={} operator_build_s={:.3} \
         graph_build_s={:.3} precond_s={:.3} cg_solve_s={:.3} block_sweeps={} \
         precond_cost_ratio={:.3} recycling_admitted={} \
         direct_threshold={direct_threshold} \
         cg_columns={} cg_iterations={} recycled_rank={} tile_columns={} \
         cg_kappa_hat={:?} cg_kappa_bound={:?} \
         cg_nonconverged_columns={} cg_relative_residual={:.3e} cg_residual_stop={:.3e}",
        stats.mean_cofiring_degree,
        stats.giant_component_fraction,
        stats.component_count,
        stats.max_component_size,
        stats.cg_max_component_nnz,
        stats.cg_operator_build_seconds,
        stats.graph_build_seconds,
        stats.cg_preconditioner_seconds,
        stats.cg_solve_seconds,
        stats.cg_block_sweeps,
        stats.cg_preconditioner_cost_ratio,
        stats.cg_recycling_admitted,
        stats.cg_columns,
        stats.cg_iterations,
        stats.cg_recycled_rank,
        stats.cg_min_tile_columns,
        stats.cg_kappa_hat,
        stats.cg_kappa_bound,
        stats.cg_nonconverged_columns,
        stats.cg_relative_residual,
        stats.cg_residual_stop,
    );
    recycle.score_refresh(
        stats.cg_block_sweeps,
        stats.cg_columns,
        stats.cg_recycled_rank,
        stats.cg_preconditioner_cost_ratio,
    );
    stats.cg_recycling_admitted = recycle.admitted();
    recycle.finish_refresh();
    Ok(stats)
}

/// Largest recycled rank whose apply work and resident storage are both
/// dominated by objects this solve already owns.
///
/// A rank-`r` symmetric correction performs two `m×r×tile` contractions. The
/// sparse operator performs `(m + nnz)×tile` multiply-adds, so
/// `2mr ≤ m+nnz` prevents the preconditioner from costing more than the operator
/// matvec it accompanies. Resident prior history plus the two current low-rank
/// factors consumes `(K+2m)r` f64 values, and a tile also owns `r×tile` coarse
/// coefficients; bounding those together by the already-materialized `K×P`
/// right-hand side preserves the decoder lane's no-OOM scale contract. The next
/// history is accumulated while solving and is independently bounded by the
/// same `r≤P` contract; at refresh completion it replaces, rather than joins,
/// the prior history.
///
/// # The history is budgeted BESIDE the block tile, never out of it
///
/// The `K×P` envelope above used to be a single pot that the history drew from
/// first and the block-CG column tile got the remainder of ([`solve_component`]
/// subtracted the history from the same `K·P` before dividing by the per-column
/// tile state). That ordering is the defect, because the tile width is not a
/// performance dial — it is the thing that makes this a BLOCK solve.
/// [`pcg_multi_core`]'s three inner products per iteration parallelize by
/// splitting the tile-wide output into fixed-width column chunks, so a narrow
/// tile runs every reduction on one thread; and at `tile = 1` the elementwise
/// row updates degrade to one rayon task per scalar. That is exactly the
/// per-column solve #1017 removed, reinstated from inside it.
///
/// Measured by the `sae_decoder_refresh_scaling` bench, same node, same binary,
/// only this bound changed: the shared pot drove `tile_columns` from 12 at the
/// first (rank-zero) refresh to **1** at every refresh after it, and the refresh
/// inflated **5.8×-29.8×** across the epochs of a single fit while the CG
/// iteration count rose only 1.1×-2.1× and the co-firing graph's `nnz` stayed
/// flat to 5%. The inflation was never the graph and never the Krylov work; it
/// was the block collapsing to a column.
///
/// So the two allocations are separated: the tile keeps the whole `K·P`
/// envelope and the history gets its own, `r ≤ K·P / (K + 2m + 1)`. The sum is
/// still `Θ(K·P)` — the no-OOM scale contract is about the ORDER, and both
/// terms are far under their bounds in practice — but the tile is no longer a
/// remainder, so it cannot be squeezed to nothing by an accelerator that is
/// optional in the first place. `r` is unchanged or larger at every shape,
/// including the production decoder (`K = 32672, P = 2048`), where it stays at
/// its work bound of 190 while the tile widens from 295 back to 409 columns.
fn decoder_recycle_rank_bound(k_total: usize, p: usize, m: usize, nnz: usize) -> usize {
    if m == 0 {
        return 0;
    }
    let work_bound = ((m as u128 + nnz as u128) / (2u128 * m as u128)) as usize;
    let rhs_values = k_total as u128 * p as u128;
    let memory_bound = (rhs_values / (k_total as u128 + 2 * m as u128 + 1)) as usize;
    work_bound.min(memory_bound).min(m).min(p)
}

/// Build the exact Galerkin inverse on the previous refresh's correction
/// subspace, expressed as a symmetric low-rank correction to Jacobi.
fn recycled_component_preconditioner(
    recycle: &DecoderRecycleSpace,
    comp: &[usize],
    row_ptr: &[u32],
    csr_cols: &[u32],
    csr_vals: &[f64],
    diagonal: &[f64],
    rank_bound: usize,
) -> Result<SymmetricLowRankPreconditioner, String> {
    use gam_linalg::faer_ndarray::{default_rrqr_rank_alpha, rrqr_with_permutation};

    let m = comp.len();
    let inverse_diagonal: Vec<f64> = diagonal.iter().map(|&d| d.recip()).collect();
    if rank_bound == 0 || recycle.directions.is_empty() {
        return Ok(SymmetricLowRankPreconditioner::jacobi(inverse_diagonal));
    }

    // Current-coordinate basis U = orth(D¹/² Z). Restricting global recycled
    // directions here makes component splits/merges exact rather than a stale
    // graph-identity assumption. Column-pivoted Householder QR identifies any
    // dependencies introduced by a split. The preconditioner's validating
    // factory performs the second thin Householder QR itself, then proves the
    // resulting Galerkin operator SPD before the private factors can exist.
    // Do not truncate the global history before restriction. Directions from
    // several old components are stored in one deterministic sequence; after a
    // split, the first `rank_bound` entries can all belong to a different
    // current component even though useful directions occur later. Restrict
    // every direction that has support here, then let the *current* RRQR choose
    // at most `rank_bound` independent columns.
    let relevant: Vec<usize> = recycle
        .directions
        .iter()
        .enumerate()
        .filter_map(|(q, direction)| comp.iter().any(|&atom| direction[atom] != 0.0).then_some(q))
        .collect();
    if relevant.is_empty() {
        return Ok(SymmetricLowRankPreconditioner::jacobi(inverse_diagonal));
    }
    let mut raw = Array2::<f64>::zeros((m, relevant.len()));
    for (q, &source) in relevant.iter().enumerate() {
        for (i, &atom) in comp.iter().enumerate() {
            raw[[i, q]] = diagonal[i].sqrt() * recycle.directions[source][atom];
        }
    }
    let rrqr = rrqr_with_permutation(&raw, default_rrqr_rank_alpha())
        .map_err(|err| format!("decoder recycled coarse-space RRQR failed: {err}"))?;
    let rank = rrqr.rank.min(rank_bound);
    if rank == 0 {
        return Ok(SymmetricLowRankPreconditioner::jacobi(inverse_diagonal));
    }
    let mut independent = Array2::<f64>::zeros((m, rank));
    for q in 0..rank {
        for i in 0..m {
            independent[[i, q]] = raw[[i, rrqr.column_permutation[q]]];
        }
    }
    drop(raw);

    // S U for S = D⁻¹/²(A+ρI)D⁻¹/². The scaled diagonal is exactly one;
    // neighbors retain the canonical CSR order used by the solve itself.
    let inverse_sqrt: Vec<f64> = diagonal.iter().map(|&d| d.sqrt().recip()).collect();
    SymmetricLowRankPreconditioner::from_scaled_subspace(
        inverse_diagonal,
        independent,
        |basis, image| {
            let rank = basis.ncols();
            image
                .as_slice_mut()
                .expect("fresh Galerkin image is standard layout")
                .par_chunks_mut(rank)
                .enumerate()
                .for_each(|(i, image_row)| {
                    image_row.copy_from_slice(
                        basis
                            .row(i)
                            .as_slice()
                            .expect("Galerkin basis row is contiguous"),
                    );
                    // Walk the CSR structure once for the whole coarse block.
                    // For every fixed q the additions remain in canonical
                    // ascending-neighbor order, while structure is loaded once.
                    for edge in row_ptr[i] as usize..row_ptr[i + 1] as usize {
                        let j = csr_cols[edge] as usize;
                        let scaled_value = csr_vals[edge] * inverse_sqrt[i] * inverse_sqrt[j];
                        let neighbor_row = basis.row(j);
                        let neighbor = neighbor_row
                            .as_slice()
                            .expect("Galerkin basis row is contiguous");
                        for q in 0..rank {
                            image_row[q] += scaled_value * neighbor[q];
                        }
                    }
                });
        },
    )
    .map_err(|err| format!("decoder recycled coarse preconditioner failed: {err}"))
}

/// Solve one connected component's block: dense SPD Cholesky when the block is
/// below the percolation critical-component scale (`direct_threshold`, see
/// [`direct_solve_size_threshold`]), else matrix-free BLOCK CG over all `P`
/// decoder columns at once. `comp` is the component's atom indices in
/// ascending order; `neigh` is the global sorted adjacency.
///
/// # Why a block solve (#1017)
///
/// The giant co-firing component shares ONE operator across every decoder
/// column. Solving the columns one at a time re-walks that operator's sparse
/// structure per column per CG iteration — at the measured production shape
/// (K = 32 000, P = 2048, ≈200 iterations/column) that is petabytes of
/// redundant structure traffic and was the entire epoch wall (69 174 s of
/// serial refresh next to 13.9 s of routed device compute, #1017). The block
/// solve advances all columns together off one CSR traversal per iteration
/// ([`pcg_multi_core`]), each column keeping its own `alpha`/`beta`/stopping
/// state. The CPU and device backends are bit-identical to one another from the
/// same cached decoder seed; both use the same symmetric recycled-Galerkin
/// preconditioner and unchanged residual certificate.
fn solve_component(
    decoder: &mut Array2<f32>,
    eq: &DecoderNormalEq,
    ridge: f64,
    comp: &[usize],
    neigh: &[Vec<(u32, f64)>],
    p: usize,
    direct_threshold: usize,
    gpu: gam_gpu::GpuPolicy,
    stats: &mut DecoderSolveStats,
    recycle: &mut DecoderRecycleSpace,
) -> Result<(), String> {
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
        if let Some(sol) = cholesky_solve_block(&mat, &rhs) {
            for (i, &a) in comp.iter().enumerate() {
                for c in 0..p {
                    decoder[[a, c]] = sol[[i, c]] as f32;
                }
            }
            return Ok(());
        }
        // Dense Cholesky is the small-component fast path, not the component's
        // only solver. Returning `Ok(())` here used to leave the decoder block
        // untouched while claiming that the refresh succeeded. Record the
        // declined shortcut, then fall through to the matrix-free block-CG path
        // below, which solves the same normal equations and carries its own
        // convergence and residual certificate.
        stats.dense_cholesky_declines += 1;
    }

    // Default coupled path: one matrix-free BLOCK CG over all live columns.
    //
    // CSR restricted to the component, in local (block-row) indices. A
    // connected component is neighbor-closed, so every stored neighbor of a
    // member is itself a member. The per-row entry order is the per-atom
    // ascending-original-id order of `neigh` — and `local` is a monotone map
    // (both `comp` and each adjacency list are ascending) — so the block
    // operator's per-column summation order (diagonal first, then ascending
    // neighbors) is EXACTLY the legacy per-column matvec's order.
    let operator_build_start = Instant::now();
    let nnz: usize = comp.iter().map(|&a| neigh[a].len()).sum();
    let mut row_ptr: Vec<u32> = Vec::with_capacity(m + 1);
    let mut csr_cols: Vec<u32> = Vec::with_capacity(nnz);
    let mut csr_vals: Vec<f64> = Vec::with_capacity(nnz);
    row_ptr.push(0);
    for &a in comp {
        for &(nb, val) in &neigh[a] {
            let j = *local
                .get(&(nb as usize))
                .expect("connected component must be neighbor-closed");
            csr_cols.push(j as u32);
            csr_vals.push(val);
        }
        row_ptr.push(csr_cols.len() as u32);
    }
    let diag_ridge: Vec<f64> = comp.iter().map(|&a| eq.diag[a] + ridge).collect();
    stats.cg_max_component_nnz = stats.cg_max_component_nnz.max(nnz);
    stats.cg_operator_build_seconds += operator_build_start.elapsed().as_secs_f64();
    let residual_tolerance = decoder_solve_relative_tolerance();

    // A-priori spectral bounds of the symmetrically Jacobi-scaled operator
    // D⁻¹/² M D⁻¹/², where M = A_sub + ρI and D = diag(M). This is the SPD
    // operator whose spectrum controls the PCG recurrence below. Gershgorin
    // bounds its largest eigenvalue. The smallest eigenvalue is bounded both
    // by Gershgorin and by M ⪰ ρI ⇒ D⁻¹/²MD⁻¹/² ⪰ ρ/max(D) I. Their ratio is
    // therefore a genuine upper condition bound for the actual recurrence,
    // not the stale unpreconditioned bound.
    let mut lambda_max_bound = 0.0f64;
    let mut lambda_min_bound = f64::INFINITY;
    let max_diagonal = diag_ridge.iter().copied().fold(0.0f64, f64::max);
    for (i, &a) in comp.iter().enumerate() {
        let mut off_abs = 0.0f64;
        for &(nb, val) in &neigh[a] {
            if let Some(&j) = local.get(&(nb as usize)) {
                off_abs += val.abs() / (diag_ridge[i] * diag_ridge[j]).sqrt();
            }
        }
        lambda_max_bound = lambda_max_bound.max(1.0 + off_abs);
        lambda_min_bound = lambda_min_bound.min(1.0 - off_abs);
    }
    let ridge_floor = if max_diagonal > 0.0 {
        ridge / max_diagonal
    } else {
        0.0
    };
    let lambda_min = lambda_min_bound.max(ridge_floor).max(DEAD_DENOM);
    let kappa_bound = (lambda_max_bound / lambda_min).max(1.0);
    stats.record_kappa_bound(kappa_bound);
    let root = kappa_bound.sqrt();
    // ⌈½√κ·ln(2√κ/ε)⌉: CG's Chebyshev bound on the steps to reach relative 2-norm
    // residual ε. The √κ inside the log is the A-norm→2-norm
    // residual correction, making this a genuine UPPER bound on the iterations
    // needed — a well-conditioned block still converges well inside it (no early
    // cut, since κ̂ ≥ κ), while a giant near-singular block is bounded instead of
    // spinning. Exact CG terminates in at most `m` steps in exact arithmetic; a
    // cap hit in floating point is typed non-convergence.
    let chebyshev = 0.5 * root * (2.0 * root / residual_tolerance).ln();
    let jacobi_cap = (chebyshev.max(0.0).ceil() as usize).min(m).max(1);

    // Split live columns from dead ones (right-hand-side norm at/below the
    // dead-denominator floor). The dead-column norm below is the same strict
    // ascending fold the legacy per-column gather performed, so the live/dead
    // split is bit-for-bit the historical one; dead columns are zeroed and —
    // exactly as before — never enter CG or the solve statistics.
    let live_columns: Vec<usize> = {
        let mut live_flags = vec![false; p];
        live_flags.par_iter_mut().enumerate().for_each(|(c, live)| {
            let mut bnorm2 = 0.0f64;
            for &a in comp {
                let b = eq.b[[a, c]];
                bnorm2 += b * b;
            }
            *live = bnorm2.sqrt() > DEAD_DENOM;
        });
        for (c, &live) in live_flags.iter().enumerate() {
            if !live {
                for &a in comp {
                    decoder[[a, c]] = 0.0;
                }
            }
        }
        live_flags
            .iter()
            .enumerate()
            .filter_map(|(c, &live)| live.then_some(c))
            .collect()
    };
    if live_columns.is_empty() {
        return Ok(());
    }

    let k_total = eq.diag.len();
    // The correction is admitted only while it is still paying for itself on
    // this fit's own operator (see `DecoderRecycleSpace::score_refresh`).
    let rank_bound = if recycle.admitted() {
        decoder_recycle_rank_bound(k_total, p, m, nnz)
    } else {
        0
    };
    let preconditioner_start = Instant::now();
    let preconditioner = recycled_component_preconditioner(
        recycle,
        comp,
        &row_ptr,
        &csr_cols,
        &csr_vals,
        &diag_ridge,
        rank_bound,
    )?;
    stats.cg_preconditioner_seconds += preconditioner_start.elapsed().as_secs_f64();
    let recycled_rank = preconditioner.rank();
    stats.cg_recycled_rank = stats.cg_recycled_rank.max(recycled_rank);
    // Cost the correction charges every sweep, in operator applications: the
    // two `m×r×tile` contractions against the operator's own `(m+nnz)×tile`.
    let cost_ratio = if m + nnz == 0 {
        0.0
    } else {
        2.0 * m as f64 * recycled_rank as f64 / (m + nnz) as f64
    };
    stats.cg_preconditioner_cost_ratio = stats.cg_preconditioner_cost_ratio.max(cost_ratio);

    // The Jacobi Gershgorin/Chebyshev cap above does not describe the new
    // low-rank-preconditioned operator. A recycled solve therefore uses CG's
    // algebraic finite-termination bound `m`; stopping remains exclusively the
    // unchanged residual certificate, and a floating-point cap hit remains typed
    // non-convergence. The first refresh (rank zero) keeps the tighter proved
    // Jacobi cap.
    let cap = if recycled_rank == 0 {
        jacobi_cap
    } else {
        m.max(1)
    };

    // Column-tile width: PCG owns five `m × tile` blocks (`X`, `R`, `Z`, `P`,
    // `AP`) plus `rank × tile` coarse coefficients, and the tile is sized
    // against the already-materialized `K × P` normal-equation RHS — an
    // allocation the caller has already proved feasible.
    //
    // The recycled history (`K × rank` retained directions plus two `m × rank`
    // current factors) is NOT subtracted here. It is bounded separately, by
    // `decoder_recycle_rank_bound`, against its own `K × P` envelope. Charging
    // it to the tile's budget instead is what collapsed `tile_columns` to 1
    // from the second refresh of every fit onward and turned this block solve
    // back into the per-column solve #1017 removed — measured at 5.8×-29.8×
    // refresh inflation over the epochs of one fit (#2283/#2441). Both terms
    // remain `Θ(K·P)`, so the no-OOM scale contract is unchanged in order; what
    // changes is that the mandatory structure no longer funds the optional
    // accelerator out of its own width.
    let rhs_values = k_total.saturating_mul(p);
    let tile_state_per_column = 5usize.saturating_mul(m).saturating_add(recycled_rank);
    let tile_columns = (rhs_values / tile_state_per_column).max(1);

    stats.cg_min_tile_columns = if stats.cg_min_tile_columns == 0 {
        tile_columns.min(live_columns.len())
    } else {
        stats
            .cg_min_tile_columns
            .min(tile_columns.min(live_columns.len()))
    };
    for tile in live_columns.chunks(tile_columns) {
        let t = tile.len();
        let mut rhs_block = Array2::<f64>::zeros((m, t));
        let mut initial_block = Array2::<f64>::zeros((m, t));
        {
            let rhs_slice = rhs_block
                .as_slice_mut()
                .expect("freshly allocated block is standard layout");
            let initial_slice = initial_block
                .as_slice_mut()
                .expect("freshly allocated block is standard layout");
            rhs_slice
                .par_chunks_mut(t)
                .zip(initial_slice.par_chunks_mut(t))
                .enumerate()
                .for_each(|(i, (rhs_row, initial_row))| {
                    let a = comp[i];
                    for (j, &c) in tile.iter().enumerate() {
                        rhs_row[j] = eq.b[[a, c]];
                        initial_row[j] = decoder[[a, c]] as f64;
                    }
                });
        }

        let solve_start = Instant::now();
        let (results, solution, on_device) = solve_block_cg(
            gpu,
            &row_ptr,
            &csr_cols,
            &csr_vals,
            &diag_ridge,
            rhs_block,
            initial_block,
            preconditioner.clone(),
            residual_tolerance,
            cap,
        )?;
        stats.cg_solve_seconds += solve_start.elapsed().as_secs_f64();
        stats.cg_block_sweeps += results
            .iter()
            .map(|core| core.iterations)
            .max()
            .unwrap_or(0);
        if on_device {
            stats.device_refresh_columns += t;
        }

        // Retain a proportional share of each tile's hardest certified
        // corrections, so the next epoch samples the whole decoder rather than
        // whichever columns happened to occupy the first tile. Ties are broken
        // by global decoder-column index.
        let quota = if live_columns.is_empty() {
            0
        } else {
            rank_bound.saturating_mul(t).div_ceil(live_columns.len())
        };
        let mut hard_columns: Vec<usize> = results
            .iter()
            .enumerate()
            .filter_map(|(j, core)| (core.stop == PcgStop::Converged).then_some(j))
            .collect();
        hard_columns.sort_by(|&left, &right| {
            results[right]
                .iterations
                .cmp(&results[left].iterations)
                .then_with(|| tile[left].cmp(&tile[right]))
        });
        for &j in hard_columns.iter().take(quota) {
            let c = tile[j];
            let correction: Vec<f64> = comp
                .iter()
                .enumerate()
                .map(|(i, &atom)| solution[[i, j]] - decoder[[atom, c]] as f64)
                .collect();
            recycle.retain_component_correction(
                comp,
                &diag_ridge,
                &correction,
                rank_bound,
                m.saturating_add(nnz),
                results[j].iterations,
                c,
            );
        }

        for (j, (&c, core)) in tile.iter().zip(results.iter()).enumerate() {
            let kappa_hat = core
                .diagnostics
                .as_ref()
                .and_then(|d| kappa_from_cg_tridiagonal(&d.alpha, &d.beta));
            stats.record_block_column(core, kappa_hat);
            if core.stop == PcgStop::Converged {
                for (i, &a) in comp.iter().enumerate() {
                    decoder[[a, c]] = solution[[i, j]] as f32;
                }
            } else {
                // The derived cap was hit or CG broke down. Keep the previous
                // decoder column; the recorded failure forbids the enclosing
                // optimizer from minting a model, with no diagonal substitute.
                let relative_residual = if core.rhs_norm > 0.0 {
                    core.final_residual_norm / core.rhs_norm
                } else {
                    0.0
                };
                log::warn!(
                    "[SAE CG] component size={m} did not converge: stop={:?} iters={} \
                     rel_residual={:.3e} residual_tolerance={:.3e} \
                     kappa_bound={:.3e} cap={cap}",
                    core.stop,
                    core.iterations,
                    relative_residual,
                    residual_tolerance,
                    kappa_bound,
                );
            }
        }
    }
    Ok(())
}

/// Run the component-restricted block CG on the best admitted backend: the
/// device-resident CUDA backend when the platform, the fit's GPU policy, and
/// the workload admit it, else the rayon CPU backend. Both backends drive the
/// SAME shared recurrence ([`pcg_multi_core`]) and honor the same per-column
/// summation-order contract, so backend choice never changes a result bit
/// (pinned by the device parity test in `decoder_gpu`).
///
/// Under `GpuPolicy::Required` a missing CUDA platform/device is a typed
/// error, never a silent CPU continuation; under `Auto` an admission decline
/// falls back to the CPU backend, while a post-admission device fault
/// panics loudly inside the device backend (no misleading CPU retry).
fn solve_block_cg(
    gpu: gam_gpu::GpuPolicy,
    row_ptr: &[u32],
    csr_cols: &[u32],
    csr_vals: &[f64],
    diag_ridge: &[f64],
    rhs_block: Array2<f64>,
    initial_block: Array2<f64>,
    preconditioner: SymmetricLowRankPreconditioner,
    residual_tolerance: f64,
    cap: usize,
) -> Result<(Vec<PcgCoreResult>, Array2<f64>, bool), String> {
    #[cfg(target_os = "linux")]
    {
        if let Some(mut device) = super::decoder_gpu::DeviceBlockCgBackend::try_new(
            gpu,
            row_ptr,
            csr_cols,
            csr_vals,
            diag_ridge,
            &rhs_block,
            &initial_block,
            &preconditioner,
        )? {
            let results = pcg_multi_core(&mut device, residual_tolerance, cap, true);
            let solution = device.take_solution()?;
            return Ok((results, solution, true));
        }
    }
    #[cfg(not(target_os = "linux"))]
    if gpu == gam_gpu::GpuPolicy::Required {
        return Err(
            "sparse_dict decoder refresh: gpu=required but the CUDA backend is not compiled \
             on this platform"
                .to_string(),
        );
    }

    let apply = |pblk: &Array2<f64>, apblk: &mut Array2<f64>| {
        let t = pblk.ncols();
        let ps = pblk.as_slice().expect("block CG state is standard layout");
        let out = apblk
            .as_slice_mut()
            .expect("block CG state is standard layout");
        out.par_chunks_mut(t).enumerate().for_each(|(i, out_row)| {
            let d = diag_ridge[i];
            let base_i = i * t;
            for (c, slot) in out_row.iter_mut().enumerate() {
                *slot = d * ps[base_i + c];
            }
            for e in row_ptr[i] as usize..row_ptr[i + 1] as usize {
                let v = csr_vals[e];
                let base_j = csr_cols[e] as usize * t;
                for (c, slot) in out_row.iter_mut().enumerate() {
                    *slot += v * ps[base_j + c];
                }
            }
        });
    };
    let mut backend = CpuPcgBlockBackend::new_with_preconditioner(
        rhs_block,
        initial_block,
        preconditioner,
        apply,
    );
    let results = pcg_multi_core(&mut backend, residual_tolerance, cap, true);
    let solution = backend.into_solution();
    Ok((results, solution, false))
}

/// Why a CG solve returned. Only [`CgStop::Converged`] means the column reached
/// the precision-derived residual floor; every other status forbids a fit.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum CgStop {
    /// Relative normal-equation residual fell to/below the precision floor.
    Converged,
    /// A non-SPD / non-finite curvature step (`pᵀAp ≤ 0`) or a non-finite `β`.
    Breakdown,
    /// The conditioning-derived iteration cap was hit before the residual floor.
    CapReached,
}

struct CgSolveResult {
    x: Vec<f64>,
    iterations: usize,
    relative_residual: f64,
    stop: CgStop,
}

fn cg_solve<F>(matvec: &F, b: &[f64], residual_tolerance: f64, cap: usize) -> CgSolveResult
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
            stop: CgStop::Converged,
        };
    }

    // Delegate the CG recurrence to the shared `gam_linalg::pcg` core — the
    // single source of truth that exists precisely to end hand-rolled CG drift.
    // This path is unpreconditioned (all-ones Jacobi diagonal), uses the
    // bit-reproducible serial reduction, and disables residual refresh, which
    // reproduces the prior loop's pure-recurrence residual exactly. No
    // diagnostics are requested: the trace-probe caller consumes only the
    // iterate and the stop certificate (the decoder refresh, which does need
    // the Lanczos trace, runs through the block path's `pcg_multi_core`).
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
        residual_tolerance,
        cap,
        0,
        false,
        DotReduction::Serial,
        &mut solution.view_mut(),
    );

    let relative_residual = if result.rhs_norm > 0.0 {
        result.final_residual_norm / result.rhs_norm
    } else {
        0.0
    };
    let stop = match result.stop {
        PcgStop::Converged => CgStop::Converged,
        PcgStop::MaxIters => CgStop::CapReached,
        PcgStop::Breakdown | PcgStop::BadPreconditioner => CgStop::Breakdown,
    };

    CgSolveResult {
        x: solution.to_vec(),
        iterations: result.iterations,
        relative_residual,
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

/// Dense SPD solve `mat · X = rhs` (multiple RHS columns) via the stated matrix.
/// A failed factorization declines this fast path; the caller records it and
/// routes the same component through residual-certified block CG.
fn cholesky_solve_block(mat: &Array2<f64>, rhs: &Array2<f64>) -> Option<Array2<f64>> {
    use faer::Side;
    use gam_linalg::faer_ndarray::FaerCholesky;

    let factor = mat.cholesky(Side::Lower).ok()?;
    Some(factor.solve_mat(rhs))
}

pub(super) fn unit_norm_rows(decoder: &mut Array2<f32>) -> Result<(), String> {
    for (atom, mut row) in decoder.outer_iter_mut().enumerate() {
        let nrm: f32 = row.iter().map(|v| v * v).sum::<f32>().sqrt();
        if !nrm.is_finite() {
            return Err(format!(
                "decoder atom {atom} has a non-finite norm before gauge normalization"
            ));
        }
        // An exactly zero row is an explicitly dead atom and carries no scale;
        // the revival step handles it. Every nonzero row is normalized exactly,
        // with no epsilon-defined scale convention.
        if nrm > 0.0 {
            row.mapv_inplace(|v| v / nrm);
            // Orient by the first nonzero component for a stable sign.
            let mut sign = 1.0f32;
            for &v in row.iter() {
                if v != 0.0 {
                    sign = v.signum();
                    break;
                }
            }
            if sign < 0.0 {
                row.mapv_inplace(|v| -v);
            }
        }
    }
    Ok(())
}

/// Fixed row-chunk width for the deterministic parallel reconstruction
/// reductions ([`explained_variance`], [`residual_scale`]). Chunk boundaries
/// are derived from the row index alone — never from the thread count — and
/// per-chunk partials are combined in ascending chunk order, so the reductions
/// are deterministic and thread-count-independent (the same discipline as the
/// block-CG column tiles). The value itself only balances scheduling overhead
/// against parallel width; it cannot change which rows contribute what.
const RECONSTRUCTION_ROW_CHUNK: usize = 1024;

/// Number of disjoint atom-row blocks the `B += CᵀX` accumulation is split
/// into. Purely a work-partition width (every `B` entry's value is identical
/// under any partition — see [`DecoderNormalEq::accumulate`]); 256 keeps
/// every host well-fed while bounding the per-block code-walk overhead.
const ACCUMULATE_ATOM_BLOCKS: usize = 256;

/// Per-chunk `(rss, tss)` partials of the reconstruction pass. Within a chunk
/// the fold order is exactly the historical serial one; across chunks the
/// partials are combined in ascending order. This reassociates the global sum
/// relative to the old fully-serial fold (an ulp-scale shift, far below the
/// `config.tolerance`-scale decisions the EV feeds) in exchange for running
/// the epoch's `O(N·s·P)` reconstruction — previously a serial wall next to
/// the parallel refresh (#1017) — across all cores.
fn reconstruction_rss_tss_chunks(
    x: ArrayView2<'_, f32>,
    codes: &[SparseCode],
    decoder: ArrayView2<'_, f32>,
    means: Option<&[f64]>,
) -> (f64, f64) {
    let p = x.ncols();
    let partials: Vec<(f64, f64)> = codes
        .par_chunks(RECONSTRUCTION_ROW_CHUNK)
        .enumerate()
        .map(|(chunk_idx, chunk)| {
            let row0 = chunk_idx * RECONSTRUCTION_ROW_CHUNK;
            let mut rss = 0.0f64;
            let mut tss = 0.0f64;
            let mut recon = vec![0.0f64; p];
            for (offset, code) in chunk.iter().enumerate() {
                let i = row0 + offset;
                for slot in recon.iter_mut() {
                    *slot = 0.0;
                }
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
                    if let Some(means) = means {
                        let t = xi[c] as f64 - means[c];
                        tss += t * t;
                    }
                }
            }
            (rss, tss)
        })
        .collect();
    partials
        .into_iter()
        .fold((0.0, 0.0), |(rss, tss), (pr, pt)| (rss + pr, tss + pt))
}

fn explained_variance(
    x: ArrayView2<'_, f32>,
    codes: &[SparseCode],
    decoder: ArrayView2<'_, f32>,
) -> f64 {
    let n = x.nrows();
    let p = x.ncols();
    // Column means for TSS: per-chunk column partials, combined in ascending
    // chunk order (deterministic, thread-count-independent).
    let mean_partials: Vec<Vec<f64>> = (0..n)
        .collect::<Vec<_>>()
        .par_chunks(RECONSTRUCTION_ROW_CHUNK)
        .map(|rows| {
            let mut sums = vec![0.0f64; p];
            for &i in rows {
                let xi = x.row(i);
                for c in 0..p {
                    sums[c] += xi[c] as f64;
                }
            }
            sums
        })
        .collect();
    let mut means = vec![0.0f64; p];
    for partial in &mean_partials {
        for c in 0..p {
            means[c] += partial[c];
        }
    }
    for c in 0..p {
        means[c] /= n as f64;
    }

    let (rss, tss) = reconstruction_rss_tss_chunks(x, codes, decoder, Some(&means));
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
    let (rss, _) = reconstruction_rss_tss_chunks(x, codes, decoder, None);
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
        CgStop, DecoderNormalEq, DecoderRecycleSpace, EvPlateau, LINEAR_EV_PLATEAU_FRACTION,
        LINEAR_SUPPORT_SATURATION_ROUNDS, LiveSupportGrowth, SparseDictionaryError, cg_solve,
        explained_variance, kappa_from_cg_tridiagonal, open_round_is_stationary, pcg_multi_core,
        recycled_component_preconditioner, route_and_code_all, run_seeded,
        solve_decoder_recycled,
        solve_decoder_with_routability_gate_recycled,
    };
    use crate::sparse_dict::codes::SparseCode;
    use crate::sparse_dict::scoring::TileScorer;
    use crate::sparse_dict::{SparseDictConfig, fit_sparse_dictionary};
    use ndarray::{Array2, ArrayView2};
    use std::collections::HashMap;

    /// Test-local one-refresh adapter. Production deliberately owns one recycle
    /// space across epochs; tests that isolate a single solve make that lifetime
    /// explicit here instead of adding a test-only item to the production module.
    fn solve_decoder(
        decoder: &mut Array2<f32>,
        eq: &DecoderNormalEq,
        ridge: f64,
        gpu: gam_gpu::GpuPolicy,
    ) -> Result<super::DecoderSolveStats, String> {
        let mut recycle = DecoderRecycleSpace::new(eq.diag.len());
        solve_decoder_recycled(decoder, eq, ridge, gpu, &mut recycle)
    }

    fn solve_decoder_with_routability_gate(
        decoder: &mut Array2<f32>,
        eq: &DecoderNormalEq,
        ridge: f64,
        residual_scale: f64,
        gpu: gam_gpu::GpuPolicy,
    ) -> Result<
        (
            super::DecoderSolveStats,
            Vec<super::RoutabilityGateDecision>,
        ),
        String,
    > {
        let mut recycle = DecoderRecycleSpace::new(eq.diag.len());
        solve_decoder_with_routability_gate_recycled(
            decoder,
            eq,
            ridge,
            residual_scale,
            gpu,
            &mut recycle,
        )
    }

    /// The plateau detector decides whether a still-open fit may be returned at
    /// all, so its failure mode is a model minted from a non-converged iterate.
    /// Drive it directly over the trajectory shapes that separate "the achievable
    /// objective stopped improving" from "the objective is still moving".
    ///
    /// The detector is only half the contract: because it certifies the running
    /// MAXIMUM, `run_from_decoder` must hand back the iterate attaining that
    /// maximum. That coupling is what makes the limit-cycle case below sound
    /// rather than a licence to return an arbitrary cycle point.
    #[test]
    fn ev_plateau_certifies_the_achievable_objective_not_a_round_2396() {
        // No climb at all and a trajectory that falls hard every round. This is
        // exactly the pair of conditions a per-round upward-share ratio reads as
        // "settled" — the upward share is identically zero and the current EV sits
        // below entry — and it is the one case where there is nothing better than
        // the entry state to hand back, so it must stay a non-convergence.
        let mut falling = EvPlateau::new(0.90);
        for candidate_ev in [0.85_f64, 0.80, 0.75, 0.70] {
            assert!(
                !falling.observe(candidate_ev, 0.05, 1.0e-12),
                "a fit that never beat its entry EV and is still moving has nothing \
                 to return (ev={candidate_ev})"
            );
        }

        // The limit cycle: the fit climbs, then oscillates. The down-swing IS a
        // plateau of the achievable objective — no further high is being set — and
        // the iterate handed back is the 0.90 one that attained it, not the 0.80
        // the confirming round happens to sit on.
        let mut cycling = EvPlateau::new(0.50);
        assert!(!cycling.observe(0.90, 0.40, 1.0e-12), "the climb itself");
        assert!(
            cycling.observe(0.80, 0.10, 1.0e-12),
            "a cycle that sets no new high has exhausted the achievable objective"
        );
        assert_eq!(cycling.best_ev, 0.90, "the running max never regresses");

        // A new high that is negligible against the climb already achieved is a
        // plateau; the detector is scale-free, not thresholded on an absolute EV.
        let mut settled = EvPlateau::new(0.50);
        assert!(!settled.observe(0.90, 0.40, 1.0e-12), "the climb itself");
        let negligible = 0.40 * LINEAR_EV_PLATEAU_FRACTION / 10.0;
        assert!(
            settled.observe(0.90 + negligible, negligible, 1.0e-12),
            "a new high negligible against the climb is a plateau"
        );

        // A still-climbing fit is never a plateau, which is the property the
        // detector existed for in the first place.
        let mut climbing = EvPlateau::new(0.10);
        assert!(!climbing.observe(0.40, 0.30, 1.0e-12));
        assert!(!climbing.observe(0.60, 0.20, 1.0e-12));
        assert!(!climbing.observe(0.75, 0.15, 1.0e-12));

        // With no climb to divide by, only a genuine numerical standstill counts —
        // and that standstill is arm 1's own test, so the open arm adds nothing
        // unsound there.
        let mut flat = EvPlateau::new(0.30);
        assert!(flat.observe(0.30, 0.0, 1.0e-12), "an exact standstill");
        assert!(
            !flat.observe(0.29, 1.0e-2, 1.0e-12),
            "no climb to compare against means a moving round is not a plateau"
        );
    }

    /// Over-complete rows that no finite unit-atom dictionary reproduces exactly:
    /// each row mixes two of the `p` axes at a ratio that advances deterministically
    /// row to row, so the top-`s` routing keeps re-partitioning a continuum of
    /// directions and the alternation churns instead of landing on a fixed point.
    fn over_complete_rows(n: usize, p: usize) -> Array2<f32> {
        let mut x = Array2::<f32>::zeros((n, p));
        for row in 0..n {
            let first = row % p;
            let second = (row * 5 + 3) % p;
            let share = ((row * 37) % 101) as f32 / 101.0;
            x[[row, first]] += 1.0 - share;
            x[[row, second]] += share;
        }
        x
    }

    /// #2396 instrument: the production budget→EV trace of the over-complete inner
    /// alternation, which is the data behind the trajectory plot on the issue.
    ///
    /// Each epoch budget too short to confirm a plateau reports, in its typed
    /// non-convergence, the EV its trajectory had reached and the three
    /// fixed-point residuals at that point — so sweeping the budget enumerates the
    /// trajectory itself. The budget that confirms reports the EV of the model
    /// actually returned. The contract this data supports is asserted by
    /// `open_arm_returns_the_best_iterate_its_trajectory_reached_2396`; this test
    /// only prints, so the numbers in the write-up can be reproduced exactly.
    #[test]
    fn zz_measure_2396_open_arm_budget_ev_trace() {
        let (k, p, n, s) = (64usize, 16usize, 400usize, 2usize);
        let x = over_complete_rows(n, p);
        for max_epochs in 2..=14usize {
            let config = SparseDictConfig {
                n_atoms: k,
                active: s,
                minibatch: 128,
                max_epochs,
                score_tile: 16,
                code_ridge: 1.0e-6,
                decoder_ridge: 1.0e-6,
                tolerance: 1.0e-9,
                score_mode: gam_gpu::GpuPolicy::Off,
            };
            match run_seeded(
                x.view(),
                &config,
                &mut DecoderRecycleSpace::new(config.n_atoms),
            ) {
                Err(SparseDictionaryError::InnerNonConvergence {
                    explained_variance,
                    ev_residual,
                    decoder_fixed_point_residual,
                    routing_residual,
                    ..
                }) => eprintln!(
                    "[#2396 trace] budget={max_epochs} status=open_unconfirmed \
                     ev={explained_variance:.12} ev_resid={ev_residual:.6e} \
                     decoder_resid={decoder_fixed_point_residual:.6e} \
                     routing_resid={routing_residual:.6e}"
                ),
                Err(other) => panic!("unexpected typed failure at budget {max_epochs}: {other}"),
                Ok(iterate) => {
                    let scorer = TileScorer::new(iterate.active, config.score_tile);
                    let codes = route_and_code_all(
                        x.view(),
                        iterate.decoder.view(),
                        &scorer,
                        iterate.active,
                        config.code_ridge,
                        config.minibatch,
                        config.score_mode,
                        None,
                    )
                    .expect("re-route the returned decoder");
                    eprintln!(
                        "[#2396 trace] budget={max_epochs} status=returned certified={} \
                         ev={:.12} ev_resid={:.6e} decoder_resid={:.6e} routing_resid={:.6e} \
                         births={} saturated={}",
                        iterate.certified,
                        explained_variance(x.view(), &codes, iterate.decoder.view()),
                        iterate.inner_ev_residual,
                        iterate.decoder_fixed_point_residual,
                        iterate.routing_residual,
                        iterate.accepted_births,
                        iterate.support_saturated,
                    );
                    break;
                }
            }
        }
    }

    /// #2396 — the other half of the open-arm contract. [`EvPlateau`] certifies
    /// that the ACHIEVABLE objective stopped improving, which is a claim about the
    /// running maximum, so the model returned has to ATTAIN that maximum.
    ///
    /// Sweep the epoch budget on an over-complete fit. Every budget too short to
    /// confirm a plateau reports, in its typed non-convergence, the EV its
    /// trajectory had reached — so the sweep enumerates points the trajectory
    /// actually passed through. The budget that does confirm must then return a
    /// model no worse than any of them. Handing back the confirming round's own
    /// iterate fails this exactly when the limit cycle confirms on a down-swing,
    /// which is the case the plateau rule is there to admit.
    #[test]
    fn open_arm_returns_the_best_iterate_its_trajectory_reached_2396() {
        let (k, p, n, s) = (64usize, 16usize, 400usize, 2usize);
        let x = over_complete_rows(n, p);
        let mut trajectory: Vec<(usize, f64)> = Vec::new();
        let mut returned: Option<(usize, f64, bool)> = None;

        for max_epochs in 2..=14usize {
            let config = SparseDictConfig {
                n_atoms: k,
                active: s,
                minibatch: 128,
                max_epochs,
                score_tile: 16,
                code_ridge: 1.0e-6,
                decoder_ridge: 1.0e-6,
                tolerance: 1.0e-9,
                score_mode: gam_gpu::GpuPolicy::Off,
            };
            match run_seeded(
                x.view(),
                &config,
                &mut DecoderRecycleSpace::new(config.n_atoms),
            ) {
                Err(SparseDictionaryError::InnerNonConvergence {
                    explained_variance: reached,
                    ..
                }) => trajectory.push((max_epochs, reached)),
                Err(other) => panic!("unexpected typed failure at budget {max_epochs}: {other}"),
                Ok(iterate) => {
                    // Re-route against the returned decoder, exactly as the public
                    // entry scores a fit, so this is the EV of the returned MODEL
                    // rather than a number the optimizer carried along.
                    let scorer = TileScorer::new(iterate.active, config.score_tile);
                    let codes = route_and_code_all(
                        x.view(),
                        iterate.decoder.view(),
                        &scorer,
                        iterate.active,
                        config.code_ridge,
                        config.minibatch,
                        config.score_mode,
                        None,
                    )
                    .expect("re-route the returned decoder");
                    let ev = explained_variance(x.view(), &codes, iterate.decoder.view());
                    returned = Some((max_epochs, ev, iterate.certified));
                    break;
                }
            }
        }

        let (budget, returned_ev, certified) =
            returned.expect("the over-complete fit must confirm a plateau within the sweep");
        assert!(
            !trajectory.is_empty(),
            "no budget was too short to confirm, so the sweep never observed the \
             trajectory it is comparing against (returned at budget {budget})"
        );
        for &(short_budget, reached) in &trajectory {
            assert!(
                returned_ev >= reached,
                "the returned model (EV {returned_ev:.9}, budget {budget}, \
                 certified={certified}) is worse than a state its own trajectory \
                 passed through (EV {reached:.9} at budget {short_budget}); a \
                 plateau certified on the running maximum must return the iterate \
                 that attains it"
            );
        }
    }

    /// #2400 — the churning-structure arm, driven through the production decision
    /// rather than replayed. A dictionary at `K ≫ N·s` accepts residual-row births
    /// every single round forever, so `accepted_births == 0` never arrives and the
    /// open arm would otherwise be unreachable no matter how settled the objective
    /// is. `LiveSupportGrowth` is what separates that from real recruitment, and
    /// `open_round_is_stationary` is where the two meet.
    ///
    /// Feed it a fixed-cardinality swap sequence — positive births, constant live
    /// support, objective plateaued — and require that the round is refused for the
    /// entire confirmation window and admitted only after it, and that a single
    /// genuinely new live atom withdraws the admission again.
    #[test]
    fn churning_births_are_admitted_only_after_the_support_saturates_2400() {
        const LIVE: usize = 40;
        let mut support = LiveSupportGrowth::new(LIVE);

        // Fixed-cardinality swaps: three proposals fire every round, live support
        // never moves. Structure is NOT settled, so admission rests entirely on
        // saturation — and saturation must take the full window.
        for round in 1..LINEAR_SUPPORT_SATURATION_ROUNDS {
            let saturated = support.observe(LIVE);
            assert!(
                !open_round_is_stationary(round, 3, saturated, true, true),
                "births are still churning and support has not saturated at round \
                 {round}; admitting here would mint a model from structure the fit \
                 has not finished recruiting"
            );
        }
        let saturated = support.observe(LIVE);
        assert!(
            saturated,
            "the full window of fixed-cardinality swaps must saturate the support"
        );
        assert!(
            open_round_is_stationary(LINEAR_SUPPORT_SATURATION_ROUNDS, 3, saturated, true, true),
            "once support has set no new high for the full window the swaps are \
             replacements on a fixed support, and a plateaued objective is admissible"
        );

        // Real recruitment withdraws it immediately: one new live atom resets the
        // window, so the very next churning round is refused again.
        let after_growth = support.observe(LIVE + 1);
        assert!(
            !after_growth,
            "a genuinely new live atom resets saturation immediately"
        );
        assert!(
            !open_round_is_stationary(
                LINEAR_SUPPORT_SATURATION_ROUNDS + 1,
                3,
                after_growth,
                true,
                true
            ),
            "recruitment restarts the confirmation window; the open arm must refuse \
             until the support has been quiet for a full window again"
        );

        // Saturation alone is never enough: the objective test and the subsolve
        // are independent and each can veto on its own.
        assert!(
            !open_round_is_stationary(9, 3, true, true, false),
            "a still-improving objective is never stationary, saturated or not"
        );
        assert!(
            !open_round_is_stationary(9, 3, true, false, true),
            "an unsound linear subsolve is never stationary"
        );
        // ...and the first post-entry round is skipped regardless, since its climb
        // denominator has not formed yet.
        assert!(
            !open_round_is_stationary(0, 0, true, true, true),
            "the entry round cannot be evidence of a plateau"
        );
    }

    #[test]
    fn live_support_growth_distinguishes_recruitment_from_fixed_cardinality_swaps_2400() {
        let mut support = LiveSupportGrowth::new(12);

        for stalled_round in 1..LINEAR_SUPPORT_SATURATION_ROUNDS {
            assert!(
                !support.observe(12),
                "support must not saturate before the full confirmation window; \
                 stalled_round={stalled_round}"
            );
        }
        assert!(
            support.observe(12),
            "fixed-cardinality birth swaps must saturate after the full window"
        );

        assert!(
            !support.observe(13),
            "a genuinely new live atom must reset saturation immediately"
        );
        assert_eq!(support.high_water, 13);
        assert_eq!(support.rounds_without_growth, 0);
    }

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
        let (_stats, gate) = solve_decoder_with_routability_gate(
            &mut decoder,
            &eq,
            0.0,
            1.0,
            gam_gpu::GpuPolicy::Auto,
        )
        .expect("decoder refresh");

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
        let (_stats_first, first_gate) = solve_decoder_with_routability_gate(
            &mut decoder,
            &eq,
            0.0,
            1.0,
            gam_gpu::GpuPolicy::Auto,
        )
        .expect("decoder refresh");
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
        let (_stats_second, second_gate) = solve_decoder_with_routability_gate(
            &mut decoder,
            &eq,
            0.0,
            1.0,
            gam_gpu::GpuPolicy::Auto,
        )
        .expect("decoder refresh");
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
        solve_decoder(&mut decoder, &eq, ridge, gam_gpu::GpuPolicy::Auto).expect("decoder refresh");

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
        solve_decoder(&mut decoder, &eq, ridge, gam_gpu::GpuPolicy::Auto).expect("decoder refresh");

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
        let stats = solve_decoder(&mut decoder, &eq, ridge, gam_gpu::GpuPolicy::Auto)
            .expect("decoder refresh");
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
    fn retained_decoder_seed_removes_repeated_refresh_work() {
        let (k, p) = (64usize, 8usize);
        let ridge = 1.0e-5f64;
        let eq = connected_tridiagonal_eq(k, p);
        let mut decoder = Array2::<f32>::zeros((k, p));

        let cold = solve_decoder(&mut decoder, &eq, ridge, gam_gpu::GpuPolicy::Off)
            .expect("cold decoder refresh");
        let warm = solve_decoder(&mut decoder, &eq, ridge, gam_gpu::GpuPolicy::Off)
            .expect("warm decoder refresh");

        assert_eq!(cold.cg_nonconverged_columns, 0);
        assert_eq!(warm.cg_nonconverged_columns, 0);
        assert!(
            warm.cg_iterations < cold.cg_iterations,
            "the retained decoder must reduce exact repeated-system work: cold={} warm={}",
            cold.cg_iterations,
            warm.cg_iterations
        );
        assert!(
            warm.cg_relative_residual <= warm.cg_residual_stop,
            "the warm solve must satisfy the same residual certificate: residual={:.3e} stop={:.3e}",
            warm.cg_relative_residual,
            warm.cg_residual_stop
        );
    }

    /// A dense Gram spectrum with a constant diagonal isolates the remaining
    /// #2441 mechanism: Jacobi is only a scalar, while the non-diagonal
    /// eigenvalue spread worsens between epochs. The cached decoder is reset to
    /// zero before every solve so this gate cannot pass from the already-proved
    /// warm `X0`; only the recycled Galerkin subspace can remove the growth.
    #[test]
    fn recycled_coarse_space_flattens_non_diagonal_conditioning_drift() {
        use faer::Side;
        use gam_linalg::faer_ndarray::FaerCholesky;

        let (k, p, hard_rank) = (64usize, 32usize, 8usize);
        let inv_sqrt_k = (k as f64).sqrt().recip();
        let hadamard = |row: usize, col: usize| {
            if (row & col).count_ones() % 2 == 0 {
                inv_sqrt_k
            } else {
                -inv_sqrt_k
            }
        };
        let fixture = |condition: f64| {
            let mut eigenvalues = vec![1.0f64; k];
            for (q, value) in eigenvalues.iter_mut().take(hard_rank).enumerate() {
                *value = condition.powf(-(q as f64) / (hard_rank - 1) as f64);
            }
            let mut dense = Array2::<f64>::zeros((k, k));
            for i in 0..k {
                for j in 0..k {
                    let mut value = 0.0f64;
                    for (q, &lambda) in eigenvalues.iter().enumerate() {
                        value += hadamard(i, q) * lambda * hadamard(j, q);
                    }
                    dense[[i, j]] = value;
                }
            }
            let mut off = HashMap::new();
            for i in 0..k {
                for j in (i + 1)..k {
                    if dense[[i, j]] != 0.0 {
                        off.insert((i as u32, j as u32), dense[[i, j]]);
                    }
                }
            }
            let mut truth = Array2::<f64>::zeros((k, p));
            let mut b = Array2::<f64>::zeros((k, p));
            for c in 0..p {
                for q in 0..hard_rank {
                    let mix = if (q & c).count_ones() % 2 == 0 {
                        1.0
                    } else {
                        -1.0
                    };
                    for i in 0..k {
                        b[[i, c]] += hadamard(i, q) * mix;
                        truth[[i, c]] += hadamard(i, q) * mix / eigenvalues[q];
                    }
                }
            }
            (
                DecoderNormalEq {
                    diag: (0..k).map(|i| dense[[i, i]]).collect(),
                    b,
                    off,
                    firings: vec![k; k],
                    amplitude_sum: vec![k as f64; k],
                },
                dense,
                truth,
            )
        };

        let conditions = [4.0f64, 1.0e3, 1.0e6];
        let mut recycle = DecoderRecycleSpace::new(k);
        let mut recycled_iterations = Vec::new();
        let mut recycled_ranks = Vec::new();
        let mut cold_iterations = Vec::new();
        for &condition in &conditions {
            let (eq, dense, truth) = fixture(condition);

            let mut cold_decoder = Array2::<f32>::zeros((k, p));
            let cold = solve_decoder(&mut cold_decoder, &eq, 0.0, gam_gpu::GpuPolicy::Off)
                .expect("Jacobi control solve");
            cold_iterations.push(cold.cg_iterations);

            // Deliberately erase the ordinary warm seed. The recycle state is
            // the only information allowed to cross this boundary.
            let mut decoder = Array2::<f32>::zeros((k, p));
            let stats = solve_decoder_recycled(
                &mut decoder,
                &eq,
                0.0,
                gam_gpu::GpuPolicy::Off,
                &mut recycle,
            )
            .expect("recycled solve");
            assert_eq!(stats.cg_nonconverged_columns, 0);
            assert!(stats.cg_relative_residual <= stats.cg_residual_stop);
            recycled_iterations.push(stats.cg_iterations);
            recycled_ranks.push(stats.cg_recycled_rank);

            // Independent dense Cholesky oracle: recycling changes only the
            // exact-solve path, never the solved normal equations.
            let dense_solution = dense
                .cholesky(Side::Lower)
                .expect("fixture SPD")
                .solve_mat(&eq.b);
            let mut error2 = 0.0f64;
            let mut oracle2 = 0.0f64;
            let mut planted2 = 0.0f64;
            for i in 0..k {
                for c in 0..p {
                    let got = decoder[[i, c]] as f64;
                    error2 += (got - dense_solution[[i, c]]).powi(2);
                    oracle2 += dense_solution[[i, c]].powi(2);
                    planted2 += (dense_solution[[i, c]] - truth[[i, c]]).powi(2);
                }
            }
            assert!(
                (planted2 / oracle2).sqrt() <= 1.0e-8,
                "dense oracle must recover the planted solution at condition={condition}"
            );
            assert!(
                (error2 / oracle2).sqrt() <= 2.0e-5,
                "recycled solve must retain dense exactness at condition={condition}: rel={:.3e}",
                (error2 / oracle2).sqrt()
            );
        }

        assert_eq!(
            recycled_ranks[0], 0,
            "the first refresh has no historical subspace"
        );
        assert!(
            recycled_ranks[1..].iter().all(|&rank| rank >= hard_rank),
            "the prior certified corrections must recover the complete hard subspace: \
             {recycled_ranks:?}"
        );
        let post_recycle = &recycled_iterations[1..];
        assert!(
            post_recycle.iter().copied().max().unwrap()
                <= post_recycle.iter().copied().min().unwrap() + p,
            "conditioning drift must add at most one aggregate iteration per RHS after recycling: \
             recycled={recycled_iterations:?}, cold={cold_iterations:?}"
        );
        assert!(
            cold_iterations[2] > 2 * recycled_iterations[2],
            "the high-condition Jacobi control must expose the non-diagonal cost removed by \
             recycling: recycled={recycled_iterations:?}, cold={cold_iterations:?}"
        );
    }

    #[test]
    fn recycled_space_restricts_before_capping_after_a_graph_split() {
        let recycle = DecoderRecycleSpace {
            rows: 4,
            // The first stored direction belongs to the other side of the
            // split. Capping before restriction used to hide the useful second
            // direction entirely when rank_bound == 1.
            directions: vec![vec![1.0, 0.0, 0.0, 0.0], vec![0.0, 0.0, 1.0, 0.0]],
            next_candidates: Vec::new(),
            next_capacity: 0,
            ..DecoderRecycleSpace::new(4)
        };
        let preconditioner = recycled_component_preconditioner(
            &recycle,
            &[2, 3],
            &[0, 1, 2],
            &[1, 0],
            &[0.25, 0.25],
            &[1.0, 1.0],
            1,
        )
        .expect("current split component Galerkin preconditioner");
        assert_eq!(
            preconditioner.rank(),
            1,
            "a useful later global direction must survive component restriction"
        );
    }

    #[test]
    fn recycled_space_reservoir_keeps_global_hardness_not_visit_order() {
        fn filled(reverse: bool) -> DecoderRecycleSpace {
            let mut recycle = DecoderRecycleSpace::new(4);
            recycle.begin_refresh(4);
            let weak = || {
                (
                    vec![0usize, 1usize],
                    vec![1.0f64, 1.0f64],
                    vec![1.0f64, 0.0f64],
                    2usize,
                    3usize,
                )
            };
            let hard = || {
                (
                    vec![2usize, 3usize],
                    vec![1.0f64, 1.0f64],
                    vec![1.0f64, 0.0f64],
                    9usize,
                    7usize,
                )
            };
            let mut retain = |candidate: (Vec<usize>, Vec<f64>, Vec<f64>, usize, usize)| {
                let (comp, diagonal, correction, iterations, column) = candidate;
                recycle.retain_component_correction(
                    &comp,
                    &diagonal,
                    &correction,
                    1,
                    4,
                    iterations,
                    column,
                );
            };
            if reverse {
                retain(hard());
                retain(weak());
            } else {
                retain(weak());
                retain(hard());
            }
            drop(retain);
            recycle.finish_refresh();
            recycle
        }

        let forward = filled(false);
        let reverse = filled(true);
        assert_eq!(forward.directions, reverse.directions);
        assert_eq!(forward.directions.len(), 1);
        assert_eq!(
            forward.directions[0],
            vec![0.0, 0.0, 1.0, 0.0],
            "the later harder component must replace an earlier weaker candidate"
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
        let mut rhs = Array2::<f64>::zeros((eigenvalues.len(), 1));
        for (i, slot) in rhs.column_mut(0).iter_mut().enumerate() {
            *slot = b[i];
        }
        let mut backend = gam_linalg::pcg::CpuPcgBlockBackend::new(
            rhs,
            Array2::<f64>::zeros((eigenvalues.len(), 1)),
            vec![1.0; eigenvalues.len()],
            |pblk: &Array2<f64>, apblk: &mut Array2<f64>| {
                let out = matvec(pblk.column(0).to_owned().as_slice().expect("contiguous"));
                for (i, slot) in apblk.column_mut(0).iter_mut().enumerate() {
                    *slot = out[i];
                }
            },
        );
        let results = pcg_multi_core(&mut backend, 1.0e-14, eigenvalues.len() + 2, true);
        let diag = results[0].diagnostics.as_ref().expect("diagnostics trace");
        let got = kappa_from_cg_tridiagonal(&diag.alpha, &diag.beta).expect("Lanczos kappa");
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
        // finite iterate — evidence the refresh propagates without substituting
        // another solve.
        let eigenvalues = [1.0f64, 5.0, 25.0, 125.0, 625.0];
        let b = vec![1.0f64; eigenvalues.len()];
        let matvec = |x: &[f64]| -> Vec<f64> {
            eigenvalues
                .iter()
                .zip(x.iter())
                .map(|(&l, &xi)| l * xi)
                .collect()
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
            eigenvalues
                .iter()
                .zip(x.iter())
                .map(|(&l, &xi)| l * xi)
                .collect()
        };
        let result = cg_solve(&matvec, &b, 1.0e-12, 64);
        assert_eq!(result.stop, CgStop::Breakdown);
        assert!(result.iterations <= 64);
        assert!(result.x.iter().all(|v| v.is_finite()));
    }

    #[test]
    fn near_singular_giant_component_is_bounded_and_resolves_via_finite_termination() {
        // A path-graph (chain) co-firing Gram with coupling 0.5 is the symmetric
        // tridiagonal Toeplitz `tridiag(0.5, 1, 0.5)`, a GIANT single component whose
        // a-priori Gershgorin condition bound is enormous: the interior discs reach
        // `center - radius = 1 - 1 = 0`, so `λ_min` floors at the ridge and
        // `κ̂ = λ_max_bound / ridge ≈ 2 / 1e-9 = 2e9`. That a-priori bound is what
        // sizes the DERIVED iteration cap, guaranteeing no unbounded spin.
        //
        // #2396 correction: the earlier expectation that this block registers a
        // TYPED non-convergence was mathematically wrong. Two facts make the giant
        // near-singular block RESOLVABLE within the cap, and CG resolves it:
        //   * The a-priori Gershgorin κ (2e9) massively overestimates the TRUE
        //     conditioning. The chain's true `λ_min` is `≈ ½(π/(k+1))² ≈ 1.2e-4` at
        //     k=200 (the ridge is negligible against it), so the true `κ ≈ 1.6e4`.
        //   * CG on a k-dimensional SPD system finite-terminates within k steps (the
        //     Krylov space is exhausted), and the cap is `min(chebyshev, m) = m = k`.
        //     For this moderate true conditioning f64 CG reaches the precision floor
        //     at/near step k — an empirical sweep confirms rel-residual ~1e-15 (well
        //     below the √ε stop) for k up to several thousand.
        // So the honest, verified contract for a giant a-priori-near-singular block
        // is: iterations BOUNDED by the derived cap (no spin), the large a-priori κ
        // bound REPORTED, the block RESOLVED to the √ε precision floor (finite
        // termination), and a FINITE decoder with no garbage substitute.
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
        let stats = solve_decoder(&mut decoder, &eq, ridge, gam_gpu::GpuPolicy::Off)
            .expect("decoder refresh");

        assert_eq!(
            stats.max_component_size, k,
            "path graph is one giant component"
        );
        let kappa_bound = stats.cg_kappa_bound.expect("a-priori kappa bound recorded");
        assert!(
            kappa_bound > 1.0e6,
            "near-singular block must report a large a-priori kappa bound, got {kappa_bound}"
        );
        // BOUNDED: iterations never exceed the derived cap (`≤ m·p` across the tile),
        // so a near-singular block can never spin unbounded.
        assert!(
            stats.cg_iterations <= k * p,
            "iterations must be bounded by the derived cap, got {}",
            stats.cg_iterations
        );
        // RESOLVED: finite-termination CG drives the block to the precision floor —
        // the giant a-priori-near-singular block is resolvable, not a non-convergence.
        assert_eq!(
            stats.cg_nonconverged_columns, 0,
            "finite-termination CG must resolve the giant block within the cap; got {} \
             non-converged columns (rel_resid={:.3e}, stop={:.3e})",
            stats.cg_nonconverged_columns, stats.cg_relative_residual, stats.cg_residual_stop
        );
        assert!(
            stats.cg_relative_residual <= stats.cg_residual_stop,
            "the resolved block's relative residual {:.3e} must sit at/below the √ε stop {:.3e}",
            stats.cg_relative_residual,
            stats.cg_residual_stop
        );
        assert!(
            decoder.iter().all(|v| v.is_finite()),
            "the refreshed decoder must be finite (no garbage substitute)"
        );
    }

    #[test]
    fn cg_cap_reached_is_typed_nonconvergence_never_a_substitute() {
        // #2396: the TYPED non-convergence path (a genuinely under-resolved column)
        // is a safety net — for well-formed SPD blocks finite-termination CG resolves
        // within the cap (see the giant-component test above), so exercise the path
        // directly by capping the iteration budget BELOW what the system needs.
        //
        // A k-dim `tridiag(0.5, 1+ρ, 0.5)` SPD system with a generic RHS needs many
        // CG steps; a cap of 1 cannot resolve it. The contract: CG must report
        // `CapReached` (a TYPED non-convergence, never masqueraded as `Converged`),
        // record a relative residual strictly above the stop, and leave a FINITE
        // partial iterate — never a NaN/garbage substitute. Raising the cap to the
        // full dimension then RESOLVES the same system, proving the fixture is
        // genuinely solvable and the cap was the only thing withheld.
        let k = 24usize;
        let ridge = 1.0e-9f64;
        let matvec = |v: &[f64]| -> Vec<f64> {
            let mut out = vec![0.0f64; k];
            for i in 0..k {
                out[i] = (1.0 + ridge) * v[i];
                if i > 0 {
                    out[i] += 0.5 * v[i - 1];
                }
                if i + 1 < k {
                    out[i] += 0.5 * v[i + 1];
                }
            }
            out
        };
        let b: Vec<f64> = (0..k).map(|i| ((i * 7 + 3) % 11) as f64 - 5.0).collect();
        let stop_tol = f64::EPSILON.sqrt();

        let capped = cg_solve(&matvec, &b, stop_tol, 1);
        assert_eq!(
            capped.stop,
            CgStop::CapReached,
            "a cap below the system's need must be a TYPED CapReached, not Converged"
        );
        assert!(
            capped.relative_residual > stop_tol,
            "an under-resolved solve must record a residual above the stop; got {:.3e} <= {:.3e}",
            capped.relative_residual,
            stop_tol
        );
        assert!(
            capped.x.iter().all(|v| v.is_finite()),
            "the partial iterate must stay finite (no garbage substitute)"
        );

        // Same system, full budget: it is genuinely solvable — CG resolves it.
        let resolved = cg_solve(&matvec, &b, stop_tol, 4 * k);
        assert_eq!(
            resolved.stop,
            CgStop::Converged,
            "with the full budget the same SPD system must resolve to the precision floor"
        );
        assert!(
            resolved.relative_residual <= stop_tol,
            "resolved relative residual {:.3e} must sit at/below the stop {:.3e}",
            resolved.relative_residual,
            stop_tol
        );
    }

    #[test]
    fn shared_rho_fs_step_matches_closed_form_criterion_fixed_point() {
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
        let rho_new = linear_shared_rho_fs_step(&stats, 1.0e-3).expect("valid FS evidence");
        assert!(
            (rho_new - 1.136_363_636_363_636_5).abs() < 1.0e-12,
            "FS step must match the closed-form evidence fixed point, got {rho_new}"
        );

        // Degenerate aggregates are typed errors. Returning the old rho would
        // falsely report an exact outer fixed point.
        let zero_energy = LinearBlockRemlStats {
            penalty_energy: 0.0,
            ..stats
        };
        assert!(linear_shared_rho_fs_step(&zero_energy, 7.0e-4).is_err());
        let zero_edof = LinearBlockRemlStats {
            gram_edof: 0.0,
            ..stats
        };
        assert!(linear_shared_rho_fs_step(&zero_edof, 7.0e-4).is_err());

        // All dof consumed is invalid evidence, not a floored denominator.
        let saturated = LinearBlockRemlStats {
            gram_edof: 100.0,
            p_cols: 3,
            penalty_energy: 4.0,
            rss: 10.0,
            n_obs: 8,
        };
        assert!(linear_shared_rho_fs_step(&saturated, 1.0e-3).is_err());
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
        let y = m.cholesky(Side::Lower).expect("A+ρI is SPD").solve_mat(a);
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
            let approx = hutchinson_gram_edof(&diag, &off, rho, k)
                .expect("every trace probe must reach its residual certificate");

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
        // Plug point 4 (design gam#2232, Increment 2): the shared-ρ REML schedule
        // must (1) TERMINATE with a settled shared ridge on a planted problem and
        // (2) TRACK the planted noise — a noisier reconstruction target selects a
        // LARGER shared ridge (ρ* = γ·σ̂²/‖D‖²_F grows with the residual variance).
        //
        // #2396: this exercises the PRODUCTION entry `run_linear_reml_schedule`
        // directly, rather than a hand-rolled 16-step loop asserting a single FS
        // step below √tolerance. For this OVER-COMPLETE planted problem (K=24 atoms
        // in a p=12 space, so K >> intrinsic rank) the inner solve is legitimately
        // best-effort — the discrete top-s routing is a limit cycle whose EV-plateau
        // noise the FS map amplifies into the ρ step, so ρ oscillates within a band
        // about its INTERIOR fixed point and a single step never reaches the
        // machine-precision band. The correct contract is therefore that the
        // schedule TERMINATES (bounded outer iterations + best-effort-aware band),
        // selects a positive finite ρ settled within its honest band, and tracks the
        // planted noise — NOT that the FS pins ρ to √tolerance, which is
        // unachievable for a best-effort inner solve and which production never
        // requires.
        use super::run_linear_reml_schedule;

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
        // The over-complete high-noise inner solve reaches its best-effort plateau
        // only after ~40-50 epochs (the routing limit cycle's mean creeps in), so
        // budget generously — the CORRECT gate errors on a genuinely-unconverged
        // inner solve, and starving the budget is what previously surfaced this as a
        // spurious InnerNonConvergence.
        let config = SparseDictConfig {
            n_atoms: k,
            active: 2,
            minibatch: 64,
            max_epochs: 80,
            score_tile: 12,
            code_ridge: 1.0e-6,
            decoder_ridge: 1.0e-6,
            tolerance: 1.0e-9,
            score_mode: gam_gpu::GpuPolicy::Off,
        };

        let x_low = planted_noisy(n, p, k, 0.03, 0x1111_2222_3333_4444);
        let x_high = planted_noisy(n, p, k, 0.40, 0x1111_2222_3333_4444);

        // The production schedule TERMINATES (bounded outer iterations) with a
        // best-effort-open ρ selection for each noise level.
        let low = run_linear_reml_schedule(x_low.view(), &config).expect("low-noise reml schedule");
        let high =
            run_linear_reml_schedule(x_high.view(), &config).expect("high-noise reml schedule");
        let rho_low = low.convergence.selected_rho;
        let rho_high = high.convergence.selected_rho;

        // (1) Both selections are finite, strictly positive, and SETTLED within the
        // schedule's honest (best-effort-aware) band — the schedule stopped because
        // ρ reached the achievable precision, not because it ran out of steps.
        assert!(
            rho_low.is_finite() && rho_low > 0.0 && rho_high.is_finite() && rho_high > 0.0,
            "shared ρ* must be finite and positive (low={rho_low}, high={rho_high})"
        );
        for (label, fit) in [("low", &low), ("high", &high)] {
            assert!(
                fit.convergence.outer_rho_residual <= fit.convergence.outer_tolerance,
                "{label}-noise schedule must settle within its band: outer_rho_residual={} \
                 vs band={}",
                fit.convergence.outer_rho_residual,
                fit.convergence.outer_tolerance
            );
            assert!(
                fit.convergence.outer_iterations >= 1
                    && fit.convergence.outer_iterations <= super::REML_SCHEDULE_MAX_OUTER_ITERS,
                "{label}-noise schedule must terminate within the outer-iteration cap; got {}",
                fit.convergence.outer_iterations
            );
        }

        // (2) NOISE TRACKING: the noisier target selects the larger shared ridge.
        assert!(
            rho_high > rho_low,
            "shared ρ* must grow with planted noise: high-noise ρ*={rho_high} \
             must exceed low-noise ρ*={rho_low}"
        );
    }

    #[test]
    fn reml_schedule_terminates_on_noise_floored_interior_fixed_point() {
        // #2396 termination guarantee (different angle from the noise-tracking test):
        // the outer FS loop is an uncapped `loop {}` that stops only at
        // `log_change ≤ band` or the ρ→0 identifiability boundary. For a
        // non-interpolating OVER-COMPLETE fit the ρ fixed point is INTERIOR (ρ never
        // reaches the boundary) and the best-effort inner solve makes the FS map
        // noisy, so the machine-precision band `√tolerance` is never met on a single
        // step. Before the fix that combination did not terminate. This test pins
        // that the schedule now RETURNS — bounded outer iterations, best-effort-open
        // certificate, ρ residual settled within the honest (best-effort-aware)
        // band — on exactly that regime.
        use super::run_linear_reml_schedule;

        // Deterministic over-complete planted mixture (K=32 atoms in p=10, so K >>
        // rank), with enough additive noise that the atoms cannot interpolate — the
        // RSS stays bounded away from zero, so the FS ρ fixed point is INTERIOR.
        let (n, p, k) = (256usize, 10usize, 32usize);
        let mut atoms = Array2::<f32>::zeros((k, p));
        for a in 0..k {
            let mut norm = 0.0f64;
            for c in 0..p {
                let v = (((a * 11 + c * 5 + 2) % 13) as f32 - 6.0) / 6.0;
                atoms[[a, c]] = v;
                norm += (v as f64) * (v as f64);
            }
            let inv = (1.0 / norm.sqrt().max(1.0e-12)) as f32;
            for c in 0..p {
                atoms[[a, c]] *= inv;
            }
        }
        let mut rng = 0x0BAD_C0DE_1234_5678u64;
        let mut x = Array2::<f32>::zeros((n, p));
        for i in 0..n {
            let a0 = i % k;
            let a1 = (i / k + 3) % k;
            for c in 0..p {
                let clean = 0.7 * atoms[[a0, c]] + 0.3 * atoms[[a1, c]];
                let eps = 0.30 * (next_unit(&mut rng) as f32 - 0.5) * 2.0;
                x[[i, c]] = clean + eps;
            }
        }
        let config = SparseDictConfig {
            n_atoms: k,
            active: 2,
            minibatch: 64,
            max_epochs: 80,
            score_tile: 10,
            code_ridge: 1.0e-6,
            decoder_ridge: 1.0e-6,
            tolerance: 1.0e-9,
            score_mode: gam_gpu::GpuPolicy::Off,
        };

        let fit = run_linear_reml_schedule(x.view(), &config).expect(
            "the schedule must terminate (return), not loop, on a noise-floored interior ρ",
        );
        assert!(
            fit.convergence.outer_iterations >= 1
                && fit.convergence.outer_iterations <= super::REML_SCHEDULE_MAX_OUTER_ITERS,
            "outer iterations must be bounded by the cap; got {}",
            fit.convergence.outer_iterations
        );
        assert!(
            fit.convergence.selected_rho.is_finite() && fit.convergence.selected_rho > 0.0,
            "an interior ρ fixed point must be finite and positive; got {}",
            fit.convergence.selected_rho
        );
        assert!(
            fit.convergence.outer_rho_residual <= fit.convergence.outer_tolerance,
            "the returned ρ must sit within the honest best-effort band: residual={} vs band={}",
            fit.convergence.outer_rho_residual,
            fit.convergence.outer_tolerance
        );
        // The honest band is WIDER than the machine-precision √tolerance because the
        // inner solve is best-effort here — proving the fix widened the band rather
        // than tightening the fit.
        assert!(
            !fit.convergence.certified,
            "a K >> rank best-effort inner solve yields an OPEN schedule certificate"
        );
        assert!(
            fit.convergence.outer_tolerance >= super::reml_schedule_rho_log_tol(config.tolerance),
            "the best-effort band must be at least the machine-precision √tolerance band"
        );
    }

    #[test]
    fn edof_estimate_clamped_below_dof_budget_for_interpolating_fit() {
        // #2396: when K ≥ N the code Gram A = CᵀC has rank ≤ N, so the true edof
        // γ = tr(A(A+ρI)⁻¹) < N and the pooled dof γ_tot = P·γ < N·P. The Hutchinson
        // estimate is clamped only to [0, K] internally, so it can overshoot that
        // rank bound by its sampling error — which then trips
        // `linear_shared_rho_fs_step`'s "pooled dof inside (0, N·P)" guard on pure
        // estimator noise (the real OLMO K=512 > N=508 fit read γ_tot = 32524.8 >
        // 32512). `linear_block_reml_stats_from_parts` now clamps the estimate to the
        // rank bound less the minimal residual dof, so the pooled dof stays strictly
        // inside (0, N·P) and the FS step accepts it as valid (interpolating)
        // evidence. Build a genuinely interpolating fit (K = 24 atoms over n = 8
        // rows, p = 3) whose near-orthogonal codes drive the raw estimate to the
        // rank ceiling, and assert both the clamp and that the FS step no longer
        // errors.
        use super::{linear_block_reml_stats_from_parts, linear_shared_rho_fs_step};
        let (n, p, k, s) = (8usize, 3usize, 24usize, 2usize);
        let mut x = Array2::<f32>::zeros((n, p));
        for i in 0..n {
            for c in 0..p {
                x[[i, c]] = (((i * 5 + c * 3 + 1) % 7) as f32 - 3.0) / 3.0;
            }
        }
        // Distinct atom pairs per row (24 atoms, 8 rows × 2 slots = 16 slots) with
        // unit codes — the code Gram is a near-identity block, so its edof at a tiny
        // ridge is essentially its full rank min(K, N) = N = 8.
        let mut indices = Array2::<u32>::zeros((n, s));
        let mut codes = Array2::<f32>::zeros((n, s));
        for i in 0..n {
            for j in 0..s {
                indices[[i, j]] = (2 * i + j) as u32;
                codes[[i, j]] = 1.0;
            }
        }
        let decoder = Array2::<f32>::from_elem((k, p), 0.1);
        let rho = 1.0e-9f64;
        let stats = linear_block_reml_stats_from_parts(
            x.view(),
            decoder.view(),
            indices.view(),
            codes.view(),
            rho,
        )
        .expect("stats");
        let ceiling = (n as f64) - 1.0 / (p as f64);
        assert!(
            stats.gram_edof <= ceiling + 1.0e-12,
            "edof must be clamped to N less the minimal residual dof: \
             got {} vs ceiling {ceiling}",
            stats.gram_edof
        );
        // The pooled dof is now strictly inside (0, N·P), so the FS step accepts it
        // (it would have errored on a raw overshoot > N per column).
        let total_obs = (n * p) as f64;
        assert!(
            (stats.p_cols as f64) * stats.gram_edof < total_obs,
            "pooled dof {} must be strictly below N·P {total_obs}",
            (stats.p_cols as f64) * stats.gram_edof
        );
        assert!(
            linear_shared_rho_fs_step(&stats, rho).is_ok(),
            "the FS step must accept the clamped interpolating evidence, not reject it"
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
            score_mode: gam_gpu::GpuPolicy::Off,
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

    #[test]
    fn tolerance_zero_certifies_machine_precision_fixed_point() {
        // #2396: a `config.tolerance` of exactly 0.0 asks for the tightest
        // achievable fixed point. In floating point that is the residual rounding
        // floor (~1e-15 for O(1)-normalized residuals), never literal zero, so the
        // certified arm must floor the comparison at machine precision — otherwise a
        // GENUINE, machine-precision fixed point is rejected as non-convergent (the
        // `sparse_fit_records_score_route_stats` shape). This gate exercises the
        // certificate FLAG (not just that the fit returns): a well-posed, exactly
        // 1-sparse problem reaches an exact fixed point, so under tolerance 0.0 the
        // fit must both RETURN and carry a CERTIFIED certificate whose residuals sit
        // at the rounding floor.
        let (k, p, n) = (4usize, 8usize, 48usize);
        // Deterministic near-orthogonal unit atoms; each row is EXACTLY one atom
        // (1-sparse), so the alternation has a unique, exactly-attainable fixed
        // point: route → decode recovers the atoms and EV → 1 at machine precision.
        let mut atoms = Array2::<f32>::zeros((k, p));
        for a in 0..k {
            let mut norm = 0.0f64;
            for c in 0..p {
                let v = (((a * 5 + c * 3 + 1) % 7) as f32 - 3.0) + if c == a { 4.0 } else { 0.0 };
                atoms[[a, c]] = v;
                norm += (v as f64) * (v as f64);
            }
            let inv = (1.0 / norm.sqrt().max(1.0e-12)) as f32;
            for c in 0..p {
                atoms[[a, c]] *= inv;
            }
        }
        let mut x = Array2::<f32>::zeros((n, p));
        for i in 0..n {
            let a = i % k;
            let scale = 1.0 + 0.5 * ((i / k) as f32);
            for c in 0..p {
                x[[i, c]] = scale * atoms[[a, c]];
            }
        }
        let config = SparseDictConfig {
            n_atoms: k,
            active: 1,
            minibatch: 16,
            max_epochs: 200,
            score_tile: 8,
            code_ridge: 1.0e-6,
            decoder_ridge: 1.0e-6,
            tolerance: 0.0,
            score_mode: gam_gpu::GpuPolicy::Off,
        };
        let fit = fit_sparse_dictionary(x.view(), &config).expect(
            "#2396: a machine-precision fixed point must certify under tolerance 0.0, not error",
        );
        assert!(
            fit.convergence.certified,
            "a well-posed exact fixed point must CERTIFY under tolerance 0.0; got \
             certified=false (ev_resid={:.3e}, decoder_resid={:.3e}, routing_resid={:.3e})",
            fit.convergence.inner_ev_residual,
            fit.convergence.decoder_residual,
            fit.convergence.routing_residual
        );
        // The certified residuals sit at the rounding floor, not literal zero — the
        // exact property that an absolute `tolerance == 0.0` could never satisfy
        // before the machine-precision floor.
        assert!(
            fit.convergence.inner_ev_residual < 1.0e-9 && fit.convergence.inner_ev_residual >= 0.0,
            "certified EV residual must be finite and at the rounding floor; got {:.3e}",
            fit.convergence.inner_ev_residual
        );
        assert!(
            fit.explained_variance > 0.999_999,
            "an exact 1-sparse fit must reconstruct at EV≈1; got {}",
            fit.explained_variance
        );
    }

}

/// #2742: the decoder-recycle break-even latch is documented as one-way for the
/// FIT. That is a claim about the OWNER of [`DecoderRecycleSpace`], and it was
/// false while the space was constructed inside [`run_seeded`]: the outer REML
/// schedule starts one seeded inner run and continues it once per outer iteration
/// (up to
/// [`REML_SCHEDULE_MAX_OUTER_ITERS`]), so every boundary readmitted a correction
/// already measured as a loss.
#[cfg(test)]
mod decoder_recycle_latch_scope_2742_tests {
    use super::{
        DecoderRecycleSpace, REML_SCHEDULE_MAX_OUTER_ITERS, run_linear_reml_schedule,
        run_linear_reml_schedule_with_recycle, run_seeded,
    };
    use crate::sparse_dict::SparseDictConfig;
    use ndarray::Array2;

    /// Planted 2-sparse corpus with a deterministic bleed, so the alternation has
    /// real structure to refresh against.
    fn planted(n: usize, p: usize) -> Array2<f32> {
        let mut x = Array2::<f32>::zeros((n, p));
        for row in 0..n {
            let first = row % p;
            let second = (row * 5 + 3) % p;
            let share = ((row * 37) % 101) as f32 / 101.0;
            x[[row, first]] += 1.0 - share;
            x[[row, second]] += share;
        }
        x
    }

    /// Deterministic splitmix-backed uniform draw in `[0, 1)` (NO `rand` crate),
    /// the crate's canonical test PRNG pattern.
    fn next_unit(state: &mut u64) -> f64 {
        let h = gam_linalg::utils::splitmix64(state);
        (h >> 11) as f64 / (1u64 << 53) as f64
    }

    /// The noisy planted 2-sparse mixture the shared-rho schedule is exercised on
    /// elsewhere in this file (`reml_schedule_held_out_ev_matches_or_beats_magic_ridge`):
    /// a fit the outer loop actually settles, so a schedule assertion is about the
    /// latch rather than about a fixture that fails to converge.
    fn planted_mixture(n: usize, p: usize, k: usize) -> Array2<f32> {
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
        x
    }

    fn config(k: usize, max_epochs: usize) -> SparseDictConfig {
        SparseDictConfig {
            n_atoms: k,
            active: 2,
            minibatch: 128,
            max_epochs,
            score_tile: 16,
            code_ridge: 1.0e-6,
            decoder_ridge: 1.0e-6,
            tolerance: 1.0e-9,
            score_mode: gam_gpu::GpuPolicy::Off,
        }
    }

    /// The schedule fixture: the mixture above at the budget and tile the settled
    /// schedule test uses.
    fn schedule_fixture() -> (Array2<f32>, SparseDictConfig, usize) {
        let (k, p, n) = (24usize, 12usize, 500usize);
        let mut config = config(k, 60);
        config.score_tile = p;
        (planted_mixture(n, p, k), config, k)
    }

    /// Drive the latch off exactly the way a failed break-even does: a rank-zero
    /// refresh supplies the Jacobi baseline, then a rank-one refresh needs twice
    /// the sweeps per column, which cannot pay for itself at ANY cost ratio.
    fn latched_off(k: usize) -> DecoderRecycleSpace {
        let mut recycle = DecoderRecycleSpace::new(k);
        let columns = k;
        recycle.score_refresh(columns, columns, 0, 0.0);
        recycle.score_refresh(2 * columns, columns, 1, 0.0);
        assert!(
            !recycle.admitted(),
            "fixture precondition: score_refresh must latch a doubled sweep count off"
        );
        recycle
    }

    #[test]
    fn latch_survives_every_inner_run_of_a_fit_2742() {
        let (k, p, n) = (32usize, 16usize, 256usize);
        let x = planted(n, p);
        let config = config(k, 3);
        let mut recycle = latched_off(k);

        // Each seeded run is an independent inner solve. The latch must not be
        // readmitted by any of them; `run_seeded` may legitimately report typed
        // non-convergence at this budget, which is irrelevant to the latch.
        for iteration in 0..3usize {
            drop(run_seeded(x.view(), &config, &mut recycle));
            assert!(
                !recycle.admitted(),
                "inner run {iteration} readmitted a correction already measured as a loss"
            );
        }
    }

    /// A fresh space is admitted on entry, so the assertion above is about the
    /// latch and not about a field that is `false` no matter what.
    #[test]
    fn a_fresh_space_enters_a_run_admitted_2742() {
        let (k, p, n) = (32usize, 16usize, 256usize);
        let x = planted(n, p);
        let config = config(k, 1);
        let mut recycle = DecoderRecycleSpace::new(k);
        assert!(recycle.admitted(), "a fresh recycle space must be admitted");
        drop(run_seeded(x.view(), &config, &mut recycle));
    }

    #[test]
    fn the_outer_reml_schedule_never_resets_the_latch_2742() {
        let (x, config, k) = schedule_fixture();
        let mut recycle = latched_off(k);

        let fit = run_linear_reml_schedule_with_recycle(x.view(), &config, &mut recycle)
            .expect("schedule fit");

        // Non-vacuity: the loop must actually have crossed at least one outer
        // boundary, which is where the latch used to be reconstructed.
        assert!(
            fit.convergence.outer_iterations >= 2,
            "fixture is vacuous: the schedule took {} outer iteration(s), so no boundary was crossed",
            fit.convergence.outer_iterations
        );
        assert!(
            fit.convergence.outer_iterations <= REML_SCHEDULE_MAX_OUTER_ITERS,
            "outer iterations must stay within the schedule cap"
        );
        assert!(
            !recycle.admitted(),
            "the outer schedule reset the break-even latch across {} outer iterations",
            fit.convergence.outer_iterations
        );
    }

    #[test]
    fn outer_reml_schedule_seeds_once_then_continues_2441() {
        let (x, config, k) = schedule_fixture();
        let mut recycle = DecoderRecycleSpace::new(k);

        let fit = run_linear_reml_schedule_with_recycle(x.view(), &config, &mut recycle)
            .expect("schedule fit");

        assert!(
            fit.convergence.outer_iterations >= 2,
            "fixture is vacuous: continuation requires an outer boundary"
        );
        assert_eq!(
            fit.convergence.seeded_inner_runs,
            1,
            "the O(K*N*P) farthest-point seed is a once-per-schedule operation"
        );
        assert_eq!(
            fit.convergence.seeded_inner_runs + fit.convergence.continued_inner_runs,
            fit.convergence.outer_iterations,
            "every evaluated outer iterate must have exactly one inner-start decision"
        );
        assert!(
            fit.convergence.continued_inner_runs >= 1,
            "every later rho must consume the prior iterate instead of cold-seeding"
        );
    }

    /// The public entry constructs the ONE space of the fit and delegates, so the
    /// scoped variant the tests above drive is the same code path production runs.
    #[test]
    fn the_public_schedule_entry_agrees_with_the_scoped_variant_2742() {
        let (x, config, k) = schedule_fixture();

        let public = run_linear_reml_schedule(x.view(), &config).expect("public schedule fit");
        let mut recycle = DecoderRecycleSpace::new(k);
        let scoped = run_linear_reml_schedule_with_recycle(x.view(), &config, &mut recycle)
            .expect("scoped schedule fit");

        assert_eq!(
            public.convergence.outer_iterations, scoped.convergence.outer_iterations,
            "the public entry must run the same schedule as the scoped variant"
        );
        assert_eq!(
            public.decoder, scoped.decoder,
            "the public entry must return the same decoder as the scoped variant"
        );
    }
}
