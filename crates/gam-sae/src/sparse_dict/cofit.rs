//! Joint co-fitting of the linear block tier and the curved chart tier
//! (residual-orthogonality trap closure).
//!
//! # The trap this closes
//!
//! The block-chart compose lane ([`super::block_chart`]) fits curved charts to
//! the **least-squares residual** of a frozen linear dictionary. But an LS
//! residual is orthogonal to the fitted span: the linear tiling has already
//! absorbed the local tangent *and* the curvature into where it placed its
//! atoms, so what is left in the residual is high-frequency sawtooth
//! quantisation noise between atoms — exactly the thing a *smooth* chart cannot
//! represent. The one-shot fit-curved-on-linear-residual protocol therefore
//! hands the curved lane a target from which the very structure it is meant to
//! find has been removed by construction.
//!
//! # The fix: monotone two-block coordinate descent
//!
//! Model the reconstruction as two **additive** tiers,
//! `x̂ = L(codes) + C(charts)`, and alternate two block solves that both descend
//! the SAME penalised joint objective
//!
//! ```text
//!   J(codes, charts) = ‖target − L − C‖²_F  +  λ_lin · ‖codes‖²_F
//! ```
//!
//! (the linear tier's ridge is explicit in `J`; the curved tier's complexity
//! penalty is realised as the compose lane's cross-fit / e-BH acceptance charge,
//! which admits a chart only when its cross-validated gain exceeds its
//! information charge — surfaced per round as [`CofitRound::curved_charge`]).
//!
//! * **Block A — linear tier refit.** With the charts (hence `C`) held fixed and
//!   the block routing frozen, re-solve the per-row active-set ridge
//!   least-squares codes against the *chart-adjusted* target `target − C`. This
//!   is an exact block minimisation of `J` over the linear codes (the previous
//!   codes are always feasible), so it is **provably monotone**: `J` cannot
//!   increase. It is precisely the step the one-shot protocol never takes — the
//!   linear tier stops chasing the curvature that the chart already explains, so
//!   its atoms are freed to model the genuinely linear part.
//! * **Block B — curved joint fit.** With the linear codes held fixed, re-fit the
//!   charts against the *linear-adjusted* target `target − L` through the
//!   existing curved surface ([`compose_block_coordinate_charts`]). The compose
//!   lane's acceptance is cross-fit gated rather than a pure held-in minimiser,
//!   so this step is **guarded**: the candidate is committed only when it does
//!   not increase `J`. The previous chart set is always available as the
//!   fallback, so the round is monotone by construction either way.
//!
//! Each committed round therefore has `J[r] ≤ J[r-1]` up to a tiny numerical
//! slack; the loop runs a small fixed number of rounds or until the relative
//! decrease of `J` stalls. The curved solver's internals are **untouched** — it
//! is called through its existing public surface with an adjusted target.

use std::collections::HashSet;

use ndarray::{Array2, Array3, ArrayView2, ArrayView3};

use super::block_chart::{
    BlockChartComposeConfig, BlockChartComposeResult, compose_block_coordinate_charts,
};
use super::codes::solve_row_codes;
use super::coordinate::explained_variance_from_reconstruction;

/// Configuration for [`cofit_block_and_curved`].
#[derive(Clone, Debug)]
pub struct CofitConfig {
    /// Maximum number of A/B alternation rounds after the one-shot round 0.
    pub max_rounds: usize,
    /// Stop early once the relative decrease of the joint objective between two
    /// consecutive rounds falls below this (a stalled descent).
    pub rel_tol: f64,
    /// Linear-tier ridge `λ_lin` on the per-row active-set least-squares codes.
    pub code_ridge: f32,
    /// Relative slack for the monotone-non-increase invariant. A round whose
    /// objective exceeds the previous by more than
    /// `monotone_slack · (|J_prev| + 1)` is a bug and aborts the fit.
    pub monotone_slack: f64,
    /// Curved-tier compose configuration. Its `block_size`, `block_topk` and
    /// `gamma` are overwritten from the passed routing/frames each round so the
    /// tiers always agree on geometry; `residual_target` is forced on (the
    /// co-fit *is* the principled residual protocol).
    pub chart: BlockChartComposeConfig,
}

impl Default for CofitConfig {
    fn default() -> Self {
        Self {
            max_rounds: 6,
            rel_tol: 1.0e-4,
            code_ridge: 1.0e-6,
            monotone_slack: 1.0e-6,
            chart: BlockChartComposeConfig::default(),
        }
    }
}

/// Per-round telemetry of the co-fit alternation.
#[derive(Clone, Debug)]
pub struct CofitRound {
    /// Round index (`0` is the one-shot fit-curved-on-linear-residual baseline;
    /// `≥1` are A/B alternation rounds).
    pub round: usize,
    /// Joint objective `J = ‖target − (L+C)‖²_F + λ_lin‖codes‖²_F` at round end.
    pub objective: f64,
    /// Reconstruction term `‖target − (L+C)‖²_F` (Frobenius SSE).
    pub recon_sse: f64,
    /// Linear-tier ridge energy `λ_lin · ‖codes‖²_F`.
    pub linear_ridge: f64,
    /// Total e-BH acceptance charge of the accepted charts this round (the curved
    /// tier's complexity penalty, enforced as an acceptance gate).
    pub curved_charge: f64,
    /// Composed explained variance (`1 − RSS/TSS`, mean baseline).
    pub explained_variance: f64,
    /// Number of accepted curved charts (single blocks + pairs) this round.
    pub n_accepted_charts: usize,
    /// Whether the linear block A step strictly reduced the objective this round
    /// (always a non-increase; `false` when it was already at the block optimum).
    pub linear_improved: bool,
    /// Whether the curved block B candidate was committed (`false` = the guard
    /// kept the previous chart set because the candidate did not reduce `J`).
    pub curved_committed: bool,
}

/// Result of a co-fit run.
#[derive(Clone, Debug)]
pub struct CofitReport {
    /// Composed reconstruction `L + C`, `N×P`.
    pub reconstructed: Array2<f32>,
    /// Linear-tier reconstruction `L` over the chart-*unowned* blocks, `N×P`
    /// (the blocks a chart replaced are excluded — they live in `C`).
    pub linear_reconstruction: Array2<f32>,
    /// Additive curved correction `C = composed − L`, `N×P` (the lifted chart
    /// coordinates of the accepted, chart-owned blocks).
    pub curved_correction: Array2<f32>,
    /// Refit linear-tier codes, `N×k×b`, at the frozen routing.
    pub codes: Array3<f32>,
    /// Final composed explained variance.
    pub explained_variance: f64,
    /// Per-round telemetry (index 0 is the one-shot baseline).
    pub rounds: Vec<CofitRound>,
    /// Final curved compose result (chart records, acceptances, screens).
    pub compose: BlockChartComposeResult,
}

/// Co-fit the block-sparse linear tier and the curved chart tier by monotone
/// two-block coordinate descent.
///
/// `decoder` (`K×P`, `K = G·b`), `blocks` (`N×k`) and `codes` (`N×k×b`) are a
/// frozen block-sparse routing (typically a [`super::BlockSparseFit`]); `gamma`
/// is that fit's shared tied scalar (used only by the curved lane's residual
/// re-encode). The routing (which blocks fired per row) is held fixed throughout;
/// the linear *codes* and the curved *charts* are the two descent blocks.
///
/// Returns a [`CofitReport`] whose `rounds[0]` is the one-shot baseline and whose
/// composed reconstruction is the co-fit result. Fails closed if the recorded
/// objectives ever violate monotone non-increase beyond the configured slack.
pub fn cofit_block_and_curved(
    target: ArrayView2<'_, f32>,
    decoder: ArrayView2<'_, f32>,
    blocks: ArrayView2<'_, u32>,
    codes: ArrayView3<'_, f32>,
    gamma: f32,
    config: &CofitConfig,
) -> Result<CofitReport, String> {
    let (n, k) = blocks.dim();
    let b = codes.shape()[2];
    if b == 0 {
        return Err("cofit: block_size (codes.shape[2]) must be >= 1".to_string());
    }
    if decoder.nrows() == 0 || decoder.nrows() % b != 0 {
        return Err(format!(
            "cofit: decoder rows {} must be a positive multiple of block_size {b}",
            decoder.nrows()
        ));
    }
    if target.nrows() != n {
        return Err(format!(
            "cofit: target has {} rows but routing has {n}",
            target.nrows()
        ));
    }
    if target.ncols() != decoder.ncols() {
        return Err(format!(
            "cofit: target has P={} but decoder has P={}",
            target.ncols(),
            decoder.ncols()
        ));
    }
    if codes.shape() != [n, k, b] {
        return Err(format!(
            "cofit: codes shape {:?} does not match ({n}, {k}, {b})",
            codes.shape()
        ));
    }
    if config.max_rounds == 0 {
        return Err("cofit: max_rounds must be >= 1".to_string());
    }
    let g_total = decoder.nrows() / b;

    // Chart config with the tiers' shared geometry stamped in (defensive: the
    // curved lane must see the same block size / routing width / tied scalar).
    let mut chart_cfg = config.chart.clone();
    chart_cfg.block_size = b;
    chart_cfg.block_topk = k;
    chart_cfg.gamma = gamma;
    chart_cfg.residual_target = true;

    // Fixed active-slot mask: a slot with an all-zero code is a routing pad (the
    // block lane's orphan / short-row padding contract), never a live block.
    let active_slots: Vec<Vec<usize>> = (0..n)
        .map(|i| {
            (0..k)
                .filter(|&j| (0..b).any(|r| codes[[i, j, r]] != 0.0))
                .collect()
        })
        .collect();

    let target_owned = target.to_owned();
    let tss = frobenius_tss(target);

    let mut work = codes.to_owned();

    // ---- Round 0: the one-shot fit-curved-on-linear-residual baseline. ----
    let compose0 = compose_block_coordinate_charts(target, decoder, blocks, work.view(), &chart_cfg)?;
    let mut accepted = accepted_set(&compose0);
    zero_owned_codes(&mut work, blocks, &accepted);
    let mut composed = compose0.reconstructed.clone();
    let mut curved = curved_correction(decoder, blocks, work.view(), b, &accepted, &composed)?;
    let mut charge = accepted_charge(&compose0);
    let mut n_charts =
        compose0.selected_chart_blocks.len() + compose0.selected_chart_pairs.len();
    let mut compose_result = compose0;

    let mut rounds: Vec<CofitRound> = Vec::with_capacity(config.max_rounds + 1);
    let mut objective = record_round(
        &mut rounds,
        0,
        RoundTelemetry {
            target: target.view(),
            composed: &composed,
            codes: &work,
            ridge: config.code_ridge,
            charge,
            n_charts,
            linear_improved: true,
            curved_committed: !accepted.is_empty(),
        },
    )?;

    // ---- Rounds 1.. : alternate linear refit (A) then curved refit (B). ----
    for round in 1..=config.max_rounds {
        let prev_objective = objective;

        // (A) Linear tier refit against the chart-adjusted target `target − C`,
        //     over the chart-*unowned* active slots only. Exact per-row block
        //     minimisation ⇒ provably monotone in J.
        let adjusted = &target_owned - &curved;
        let linear_improved = refit_linear_tier(
            &mut work,
            LinearRefitInputs {
                adjusted: adjusted.view(),
                decoder,
                blocks,
                b,
                accepted: &accepted,
                active_slots: &active_slots,
                ridge: config.code_ridge,
            },
        );
        let l_unowned = reconstruct_masked(decoder, blocks, work.view(), b, &accepted, false)?;
        let composed_a = &l_unowned + &curved;
        let objective_a = joint_objective(target.view(), &composed_a, &work, config.code_ridge);
        assert_monotone(objective_a, prev_objective, config.monotone_slack, round, "A")?;

        // (B) Curved tier refit against the linear-adjusted target `target − L`,
        //     through the existing compose surface. Guarded commit.
        let candidate = compose_block_coordinate_charts(target, decoder, blocks, work.view(), &chart_cfg)?;
        let cand_accepted = accepted_set(&candidate);
        let cand_composed = candidate.reconstructed.clone();
        let objective_b = joint_objective(target.view(), &cand_composed, &work, config.code_ridge);

        let slack = config.monotone_slack * (objective_a.abs() + 1.0);
        let curved_committed = objective_b <= objective_a + slack;
        if curved_committed {
            accepted = cand_accepted;
            zero_owned_codes(&mut work, blocks, &accepted);
            composed = cand_composed;
            curved = curved_correction(decoder, blocks, work.view(), b, &accepted, &composed)?;
            charge = accepted_charge(&candidate);
            n_charts =
                candidate.selected_chart_blocks.len() + candidate.selected_chart_pairs.len();
            compose_result = candidate;
            objective = objective_b;
        } else {
            // Keep the previous chart set; the committed reconstruction is the
            // block-A composed (better linear on the frozen charts).
            composed = composed_a;
            objective = objective_a;
        }

        record_round(
            &mut rounds,
            round,
            RoundTelemetry {
                target: target.view(),
                composed: &composed,
                codes: &work,
                ridge: config.code_ridge,
                charge,
                n_charts,
                linear_improved,
                curved_committed,
            },
        )?;
        assert_monotone(objective, prev_objective, config.monotone_slack, round, "round")?;

        // Stall test on the relative objective decrease.
        let denom = prev_objective.abs().max(tss).max(1.0);
        if (prev_objective - objective) / denom < config.rel_tol {
            break;
        }
    }

    let linear_reconstruction = reconstruct_masked(decoder, blocks, work.view(), b, &accepted, false)?;
    let curved_final = &composed - &linear_reconstruction;
    let explained_variance = explained_variance_from_reconstruction(target, composed.view())?;
    // Guard against an inconsistent report (bounds the accepted block indices).
    // Fails closed in every build: an out-of-range block id is a co-fit bug.
    if let Some(&bad) = accepted.iter().find(|&&g| g >= g_total) {
        return Err(format!(
            "cofit: accepted chart block {bad} out of range (g_total={g_total})"
        ));
    }

    Ok(CofitReport {
        reconstructed: composed,
        linear_reconstruction,
        curved_correction: curved_final,
        codes: work,
        explained_variance,
        rounds,
        compose: compose_result,
    })
}

/// The global set of BIC-selected chart-owned block indices from a compose result.
fn accepted_set(result: &BlockChartComposeResult) -> HashSet<usize> {
    let mut s = HashSet::new();
    for &g in &result.selected_chart_blocks {
        s.insert(g);
    }
    for &(g0, g1) in &result.selected_chart_pairs {
        s.insert(g0);
        s.insert(g1);
    }
    s
}

/// Total complexity charge of the BIC-selected charts.
fn accepted_charge(result: &BlockChartComposeResult) -> f64 {
    let mut charge = 0.0;
    for rec in &result.block_records {
        if rec.evidence.selected_by_bic {
            charge += rec.evidence.charge;
        }
    }
    for rec in &result.pair_records {
        if rec.evidence.selected_by_bic {
            charge += rec.evidence.charge;
        }
    }
    charge
}

/// Reconstruct the linear tier over the chart-owned (`keep_accepted = true`) or
/// chart-unowned (`false`) blocks only: sum a slot's rank-`b` contribution iff
/// its block's membership in `accepted` matches `keep_accepted`.
fn reconstruct_masked(
    decoder: ArrayView2<'_, f32>,
    blocks: ArrayView2<'_, u32>,
    codes: ArrayView3<'_, f32>,
    b: usize,
    accepted: &HashSet<usize>,
    keep_accepted: bool,
) -> Result<Array2<f32>, String> {
    let (n, k) = blocks.dim();
    let g = decoder.nrows() / b;
    let p = decoder.ncols();
    let mut out = Array2::<f32>::zeros((n, p));
    for i in 0..n {
        for j in 0..k {
            let block = blocks[[i, j]] as usize;
            if block >= g {
                return Err(format!(
                    "reconstruct_masked: block index {block} out of range 0..{g}"
                ));
            }
            if accepted.contains(&block) != keep_accepted {
                continue;
            }
            for r in 0..b {
                let code = codes[[i, j, r]];
                if code == 0.0 {
                    continue;
                }
                let atom = decoder.row(block * b + r);
                for c in 0..p {
                    out[[i, c]] += code * atom[c];
                }
            }
        }
    }
    Ok(out)
}

/// The additive curved correction `C = composed − L_unowned`, i.e. the lifted
/// chart coordinates of the accepted (chart-owned) blocks. Because
/// `composed = L_unowned + (chart lifts of the owned blocks)` by the compose
/// lane's construction, subtracting the unowned linear reconstruction isolates

/// Zero the linear codes of every chart-OWNED active slot. A chart-owned
/// block's reconstruction lives entirely in the curved correction `C`; its
/// linear code no longer participates in `L_unowned`, so the exact minimizer
/// of `J = ‖x − L_unowned − C‖² + λ‖codes‖²` in those slots is `0`. Leaving
/// stale owned-code mass in place would let `λ‖codes‖²` price codes the
/// reconstruction does not use, distorting the recorded objective and the
/// stall test. If ownership later recedes, the block-A exact per-row solve
/// re-fits the slot from zero.
fn zero_owned_codes(codes: &mut Array3<f32>, blocks: ArrayView2<'_, u32>, accepted: &HashSet<usize>) {
    let (n, slots, width) = codes.dim();
    for i in 0..n {
        for j in 0..slots.min(blocks.ncols()) {
            if accepted.contains(&(blocks[[i, j]] as usize)) {
                for r in 0..width {
                    codes[[i, j, r]] = 0.0;
                }
            }
        }
    }
}

/// the pure curved contribution — a term that does NOT depend on the unowned
/// blocks' codes, so it stays frozen through the next block-A refit.
fn curved_correction(
    decoder: ArrayView2<'_, f32>,
    blocks: ArrayView2<'_, u32>,
    codes: ArrayView3<'_, f32>,
    b: usize,
    accepted: &HashSet<usize>,
    composed: &Array2<f32>,
) -> Result<Array2<f32>, String> {
    let l_unowned = reconstruct_masked(decoder, blocks, codes, b, accepted, false)?;
    Ok(composed - &l_unowned)
}

/// Frozen inputs to the block-A linear refit: the chart-adjusted target, the
/// dictionary/routing views, the chart-owned block set, and the per-row active
/// slots — everything the refit reads but never writes.
struct LinearRefitInputs<'a> {
    adjusted: ArrayView2<'a, f32>,
    decoder: ArrayView2<'a, f32>,
    blocks: ArrayView2<'a, u32>,
    b: usize,
    accepted: &'a HashSet<usize>,
    active_slots: &'a [Vec<usize>],
    ridge: f32,
}

/// Block A: re-solve the per-row active-set ridge least-squares codes for the
/// chart-unowned active slots against `adjusted = target − C`, writing the result
/// back in place. Chart-owned slots (curved-owned blocks) and routing pads are
/// left untouched. Returns whether any row's solve moved the codes.
fn refit_linear_tier(codes: &mut Array3<f32>, inputs: LinearRefitInputs<'_>) -> bool {
    let LinearRefitInputs {
        adjusted,
        decoder,
        blocks,
        b,
        accepted,
        active_slots,
        ridge,
    } = inputs;
    let n = blocks.nrows();
    let mut improved = false;
    for i in 0..n {
        // Chart-unowned active slots for this row (each a distinct fired block).
        let slots: Vec<usize> = active_slots[i]
            .iter()
            .copied()
            .filter(|&j| !accepted.contains(&(blocks[[i, j]] as usize)))
            .collect();
        if slots.is_empty() {
            continue;
        }
        // Flatten the slots' `b` atoms into a joint active-set design; solve the
        // combined least-squares (handles overlapping block spans exactly).
        let mut active: Vec<(u32, f32)> = Vec::with_capacity(slots.len() * b);
        for &j in &slots {
            let block = blocks[[i, j]] as usize;
            for r in 0..b {
                active.push(((block * b + r) as u32, 0.0));
            }
        }
        let s = active.len();
        let solved = solve_row_codes(adjusted.row(i), decoder, &active, s, ridge);
        for (pos, &j) in slots.iter().enumerate() {
            for r in 0..b {
                let new = solved.codes[pos * b + r];
                if (codes[[i, j, r]] - new).abs() > 0.0 {
                    improved = true;
                }
                codes[[i, j, r]] = new;
            }
        }
    }
    improved
}

/// Frobenius reconstruction SSE `‖target − recon‖²_F`.
fn frobenius_sse(target: ArrayView2<'_, f32>, recon: &Array2<f32>) -> f64 {
    let mut sse = 0.0;
    for i in 0..target.nrows() {
        for c in 0..target.ncols() {
            let d = target[[i, c]] as f64 - recon[[i, c]] as f64;
            sse += d * d;
        }
    }
    sse
}

/// Total sum of squares of `target` about its per-column mean.
fn frobenius_tss(target: ArrayView2<'_, f32>) -> f64 {
    let (n, p) = target.dim();
    let mut means = vec![0.0f64; p];
    for i in 0..n {
        for c in 0..p {
            means[c] += target[[i, c]] as f64;
        }
    }
    for m in &mut means {
        *m /= n.max(1) as f64;
    }
    let mut tss = 0.0;
    for i in 0..n {
        for c in 0..p {
            let d = target[[i, c]] as f64 - means[c];
            tss += d * d;
        }
    }
    tss
}

/// The linear-tier ridge energy `λ_lin · ‖codes‖²_F`.
fn linear_ridge_energy(codes: &Array3<f32>, ridge: f32) -> f64 {
    let mut acc = 0.0f64;
    for &c in codes.iter() {
        acc += (c as f64) * (c as f64);
    }
    ridge as f64 * acc
}

/// The joint objective `J = ‖target − recon‖²_F + λ_lin‖codes‖²_F`.
fn joint_objective(
    target: ArrayView2<'_, f32>,
    recon: &Array2<f32>,
    codes: &Array3<f32>,
    ridge: f32,
) -> f64 {
    frobenius_sse(target, recon) + linear_ridge_energy(codes, ridge)
}

/// One round's telemetry inputs: the reconstruction state to score plus the
/// round's chart/step outcomes, bundled so `record_round` reads one coherent
/// snapshot per round.
struct RoundTelemetry<'a> {
    target: ArrayView2<'a, f32>,
    composed: &'a Array2<f32>,
    codes: &'a Array3<f32>,
    ridge: f32,
    charge: f64,
    n_charts: usize,
    linear_improved: bool,
    curved_committed: bool,
}

/// Append a round's telemetry and return its joint objective.
fn record_round(
    rounds: &mut Vec<CofitRound>,
    round: usize,
    telemetry: RoundTelemetry<'_>,
) -> Result<f64, String> {
    let RoundTelemetry {
        target,
        composed,
        codes,
        ridge,
        charge,
        n_charts,
        linear_improved,
        curved_committed,
    } = telemetry;
    let recon_sse = frobenius_sse(target, composed);
    let linear_ridge = linear_ridge_energy(codes, ridge);
    let objective = recon_sse + linear_ridge;
    let explained_variance = explained_variance_from_reconstruction(target, composed.view())?;
    rounds.push(CofitRound {
        round,
        objective,
        recon_sse,
        linear_ridge,
        curved_charge: charge,
        explained_variance,
        n_accepted_charts: n_charts,
        linear_improved,
        curved_committed,
    });
    Ok(objective)
}

/// Internal invariant: a step's objective must not exceed the previous by more
/// than the relative slack. Fails closed (a violation is a co-fit bug).
fn assert_monotone(
    objective: f64,
    prev: f64,
    slack_rel: f64,
    round: usize,
    stage: &str,
) -> Result<(), String> {
    let slack = slack_rel * (prev.abs() + 1.0);
    if objective > prev + slack {
        return Err(format!(
            "cofit: monotone non-increase violated at round {round} stage {stage}: \
             J={objective} exceeds previous {prev} beyond slack {slack}"
        ));
    }
    Ok(())
}

#[cfg(test)]
mod cofit_tests {
    use super::*;
    use crate::sparse_dict::reconstruct_block_sparse_rows;

    /// Orthonormal-per-block decoder for the planted trap. Three blocks of size
    /// b=2 in P=5:
    ///   block 0 = {e0, e1}          (linear, disjoint span)
    ///   block 1 = {(e1+e2)/√2, (e1−e2)/√2}  (linear, OVERLAPS block 0 on e1)
    ///   block 2 = {e3, e4}          (the planted circle's plane)
    /// The block-0 / block-1 overlap on e1 makes the tied per-block projection
    /// codes suboptimal (they double-count e1), so the block-A least-squares
    /// refit strictly improves the linear tier.
    fn planted_decoder() -> Array2<f32> {
        let s = 1.0f32 / 2.0f32.sqrt();
        // Rows are atoms; 6 atoms × 5 dims.
        Array2::from_shape_vec(
            (6, 5),
            vec![
                1.0, 0.0, 0.0, 0.0, 0.0, // e0
                0.0, 1.0, 0.0, 0.0, 0.0, // e1
                0.0, s, s, 0.0, 0.0, // (e1+e2)/√2
                0.0, s, -s, 0.0, 0.0, // (e1−e2)/√2
                0.0, 0.0, 0.0, 1.0, 0.0, // e3
                0.0, 0.0, 0.0, 0.0, 1.0, // e4
            ],
        )
        .unwrap()
    }

    /// Planted data: a linear part in span{e0,e1,e2} plus a unit circle in the
    /// {e3,e4} plane, tiny noise. Returns (target, planted angles).
    fn planted_data(n: usize) -> (Array2<f32>, Vec<f64>) {
        let mut x = Array2::<f32>::zeros((n, 5));
        let mut theta = Vec::with_capacity(n);
        for i in 0..n {
            // Deterministic pseudo-random coefficients (no rng dependency).
            let a = ((i * 7 + 1) % 17) as f32 / 17.0 - 0.5;
            let bb = ((i * 13 + 5) % 19) as f32 / 19.0 - 0.5;
            let cc = ((i * 5 + 3) % 23) as f32 / 23.0 - 0.5;
            let t = 2.0 * std::f64::consts::PI * (i as f64) / (n as f64);
            let noise = 0.002 * (((i * 3) % 11) as f32 / 11.0 - 0.5);
            x[[i, 0]] = a;
            x[[i, 1]] = bb;
            x[[i, 2]] = cc;
            x[[i, 3]] = t.cos() as f32 + noise;
            x[[i, 4]] = t.sin() as f32 - noise;
            theta.push(t);
        }
        (x, theta)
    }

    /// Tied per-block routing: every row fires all three blocks; the within-block
    /// code is the tied projection `z_g = x D_gᵀ` (γ = 1). This is exactly what a
    /// converged block-sparse fit stores.
    fn tied_routing(x: &Array2<f32>, decoder: &Array2<f32>, b: usize) -> (Array2<u32>, Array3<f32>) {
        let n = x.nrows();
        let g = decoder.nrows() / b;
        let mut blocks = Array2::<u32>::zeros((n, g));
        let mut codes = Array3::<f32>::zeros((n, g, b));
        for i in 0..n {
            for gg in 0..g {
                blocks[[i, gg]] = gg as u32;
                for r in 0..b {
                    let atom = decoder.row(gg * b + r);
                    let mut dot = 0.0f32;
                    for c in 0..decoder.ncols() {
                        dot += x[[i, c]] * atom[c];
                    }
                    codes[[i, gg, r]] = dot;
                }
            }
        }
        (blocks, codes)
    }

    fn ev(x: &Array2<f32>, recon: &Array2<f32>) -> f64 {
        explained_variance_from_reconstruction(x.view(), recon.view()).unwrap()
    }

    fn chart_cfg_small() -> BlockChartComposeConfig {
        BlockChartComposeConfig {
            block_size: 2,
            block_topk: 3,
            min_firings: 8,
            crossfit_folds: 4,
            pair_screen: false,
            ..BlockChartComposeConfig::default()
        }
    }

    #[test]
    fn cofit_beats_one_shot_and_recovers_angle() {
        let n = 240;
        let decoder = planted_decoder();
        let (x, theta) = planted_data(n);
        let (blocks, codes) = tied_routing(&x, &decoder, 2);

        let config = CofitConfig {
            max_rounds: 5,
            code_ridge: 1.0e-6,
            chart: chart_cfg_small(),
            ..CofitConfig::default()
        };
        let report =
            cofit_block_and_curved(x.view(), decoder.view(), blocks.view(), codes.view(), 1.0, &config)
                .expect("cofit runs");

        // (a) The co-fit's composed reconstruction beats the one-shot baseline
        //     (round 0) on explained variance — the linear tier, freed of the
        //     curvature the chart explains, stops double-counting e1.
        let one_shot_ev = report.rounds[0].explained_variance;
        assert!(
            report.explained_variance > one_shot_ev + 1.0e-4,
            "co-fit EV {} should beat one-shot EV {}",
            report.explained_variance,
            one_shot_ev
        );

        // A curved chart must actually have been accepted for block 2 (the plane
        // holding the circle) — otherwise this is a pure-linear result.
        assert!(
            report.rounds.last().unwrap().n_accepted_charts >= 1,
            "expected at least one accepted curved chart"
        );

        // (a, cont.) The curved atom's recovered coordinates correlate with the
        //     planted angle. The chart lives in {e3, e4}; read the recovered
        //     angle off the curved correction there. Use the rotation-invariant
        //     complex correlation magnitude to allow a global chart-frame gauge.
        let c = &report.curved_correction;
        let mut re = 0.0f64;
        let mut im = 0.0f64;
        let mut mag = 0.0f64;
        for i in 0..n {
            let hat = (c[[i, 4]] as f64).atan2(c[[i, 3]] as f64);
            let d = theta[i] - hat;
            re += d.cos();
            im += d.sin();
            mag += 1.0;
        }
        let rho = (re * re + im * im).sqrt() / mag.max(1.0);
        assert!(
            rho > 0.9,
            "recovered circle angle should track the planted angle (ρ={rho})"
        );
    }

    #[test]
    fn objective_is_monotone_across_rounds() {
        let n = 240;
        let decoder = planted_decoder();
        let (x, _theta) = planted_data(n);
        let (blocks, codes) = tied_routing(&x, &decoder, 2);

        let config = CofitConfig {
            max_rounds: 6,
            chart: chart_cfg_small(),
            ..CofitConfig::default()
        };
        let report =
            cofit_block_and_curved(x.view(), decoder.view(), blocks.view(), codes.view(), 1.0, &config)
                .expect("cofit runs");

        assert!(report.rounds.len() >= 2, "expected multiple rounds");
        let slack = config.monotone_slack;
        for w in report.rounds.windows(2) {
            let prev = w[0].objective;
            let cur = w[1].objective;
            assert!(
                cur <= prev + slack * (prev.abs() + 1.0),
                "objective rose from {prev} to {cur}"
            );
        }
    }

    #[test]
    fn empty_curved_tier_reproduces_pure_linear_fit() {
        let n = 200;
        let decoder = planted_decoder();
        let (x, _theta) = planted_data(n);
        let (blocks, codes) = tied_routing(&x, &decoder, 2);

        // Force the curved tier empty: select no blocks ⇒ no charts accepted.
        let mut chart = chart_cfg_small();
        chart.max_blocks = 0;
        let config = CofitConfig {
            max_rounds: 3,
            code_ridge: 1.0e-6,
            chart,
            ..CofitConfig::default()
        };
        let report =
            cofit_block_and_curved(x.view(), decoder.view(), blocks.view(), codes.view(), 1.0, &config)
                .expect("cofit runs");

        // No charts anywhere.
        assert_eq!(report.rounds.last().unwrap().n_accepted_charts, 0);
        // The curved correction is identically zero.
        let max_c = report
            .curved_correction
            .iter()
            .fold(0.0f32, |m, &v| m.max(v.abs()));
        assert!(max_c < 1.0e-5, "curved correction should vanish (max {max_c})");

        // The composed reconstruction equals an independent per-row least-squares
        // linear solve over all fired blocks (the pure linear fit).
        let b = 2usize;
        let mut ref_codes = Array3::<f32>::zeros((n, decoder.nrows() / b, b));
        for i in 0..n {
            let mut active: Vec<(u32, f32)> = Vec::new();
            for gg in 0..(decoder.nrows() / b) {
                for r in 0..b {
                    active.push(((gg * b + r) as u32, 0.0));
                }
            }
            let s = active.len();
            let solved = solve_row_codes(x.row(i), decoder.view(), &active, s, 1.0e-6);
            for (t, code) in solved.codes.iter().enumerate() {
                ref_codes[[i, t / b, t % b]] = *code;
            }
        }
        let ref_recon =
            reconstruct_block_sparse_rows(decoder.view(), blocks.view(), ref_codes.view(), b).unwrap();
        let ev_ref = ev(&x, &ref_recon);
        assert!(
            (report.explained_variance - ev_ref).abs() < 1.0e-6,
            "empty-curved co-fit EV {} should match the pure linear LS fit EV {}",
            report.explained_variance,
            ev_ref
        );
        // And it must strictly beat the tied one-shot baseline (LS fixes the e1
        // double-count the tied projection introduced).
        assert!(report.explained_variance > report.rounds[0].explained_variance + 1.0e-4);
    }
}
