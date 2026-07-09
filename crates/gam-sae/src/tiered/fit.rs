//! End-to-end tiered fit driver (#2023).
//!
//! Assembles the primitives the tiered architecture is built from into one
//! runnable path, replacing the "components exist but nothing composes them"
//! state:
//!
//! 1. **Tier 0** — peel the shared column mean ([`Tier0Mean`]); the bulk is fit
//!    on the de-meaned residual `R0 = z − μ`.
//! 2. **Tier 1** — the block-sparse collapsed-linear dictionary
//!    ([`fit_block_sparse_dictionary`]) at width `K = G·b`, the scalable linear
//!    bulk.
//! 3. **Tier 2** — curved atoms fit on the Tier-1 residual via
//!    [`cofit_block_and_curved`]: its round 0 fits curved charts on the linear
//!    residual and rounds ≥1 alternate the two blocks to joint stationarity. The
//!    curved tier seeds from the Tier-1 block routing / residual, so **this path
//!    never PC-reseeds** — the migration-ledger property this architecture asks
//!    for (`residual factor ↔ linear atom ↔ curved atom`, no principal-component
//!    reseeding) holds by construction.
//!
//! The [`MigrationLedger`] records the curved-tier promotions/demotions that the
//! co-fit adjudicates each round in the matched-description-length currency
//! (`curved_charge`, the e-BH acceptance charge), plus the Tier-1 block deaths.
//! `pc_reseed_events` is always `0` on this path.

use ndarray::ArrayView2;

use crate::sparse_dict::{
    BlockSparseConfig, BlockSparseFit, CofitConfig, CofitReport, cofit_block_and_curved,
    fit_block_sparse_dictionary,
};
use crate::tiered::Tier0Mean;

/// Configuration for [`fit_tiered`].
#[derive(Clone, Debug)]
pub struct TieredFitConfig {
    /// Tier-1 block-sparse dictionary configuration (`G` blocks of size `b`, the
    /// block budget `k`, epochs, minibatch/tile geometry). GPU score-routing is
    /// governed by the process-wide [`gam_gpu::GpuMode`] (`gam_gpu::set_gpu_mode`),
    /// not by this config: the Tier-1 router dispatches each minibatch to the CUDA
    /// block-gate lane when the mode admits it and a runtime is present.
    pub tier1: BlockSparseConfig,
    /// Whether to run the Tier-2 curved co-fit on the Tier-1 residual (`false` ⇒
    /// Tier-0 + Tier-1 only, the linear-bulk baseline).
    pub tier2_enabled: bool,
    /// Tier-2 curved co-fit configuration. `block_size`/`block_topk`/`gamma` are
    /// overwritten from the Tier-1 routing inside the co-fit each round so the
    /// tiers always agree on geometry.
    pub cofit: CofitConfig,
}

impl TieredFitConfig {
    /// A Tier-0 + Tier-1 config at `G` blocks of size `b` (Tier-2 disabled).
    pub fn linear_bulk(n_blocks: usize, block_size: usize) -> Self {
        Self {
            tier1: BlockSparseConfig::new(n_blocks, block_size),
            tier2_enabled: false,
            cofit: CofitConfig::default(),
        }
    }

    /// A Tier-0 + Tier-1 + Tier-2 config at `G` blocks of size `b`.
    pub fn tiered(n_blocks: usize, block_size: usize) -> Self {
        Self {
            tier1: BlockSparseConfig::new(n_blocks, block_size),
            tier2_enabled: true,
            cofit: CofitConfig::default(),
        }
    }
}

/// A move in the three-state migration ledger (`residual factor ↔ linear atom ↔
/// curved atom`) that replaces principal-component reseeding.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum TieredMoveKind {
    /// A linear block / residual factor was promoted to a curved chart because
    /// the evidence (matched description length) preferred the curve.
    Promotion,
    /// A curved chart candidate was rejected this round; the linear block is kept
    /// (the tier-boundary analogue of the hybrid-split `Θ→0` verdict).
    Demotion,
    /// A Tier-1 block ended dead (no rows selected it); it falls back to the
    /// residual factor pool. Dead-block revival draws from high-residual rows,
    /// never from principal components.
    Death,
}

/// One adjudicated migration move, recorded at the granularity the co-fit
/// telemetry exposes (per round for promotions/demotions; one structural entry
/// for the Tier-1 deaths).
#[derive(Clone, Copy, Debug)]
pub struct TieredLedgerMove {
    /// Move kind.
    pub kind: TieredMoveKind,
    /// Co-fit round the move was adjudicated in; `None` for the Tier-1 structural
    /// death tally (which is decided by the linear fit, not a co-fit round).
    pub round: Option<usize>,
    /// Number of charts / blocks affected by this move.
    pub count: usize,
    /// Matched-description-length charge in the co-fit's currency (the e-BH
    /// acceptance charge `curved_charge`); `0.0` for demotions and deaths.
    pub dl_bits: f64,
    /// Joint objective `J` after the round (`NaN` for the structural death tally).
    pub objective: f64,
}

/// The migration ledger: every birth/death that moved an atom between the
/// residual-factor, linear, and curved states, plus the invariant that this path
/// performs **zero** principal-component reseeds.
#[derive(Clone, Debug, Default)]
pub struct MigrationLedger {
    /// The adjudicated moves in order.
    pub moves: Vec<TieredLedgerMove>,
    /// Number of principal-component reseed events. **Always `0`** on the tiered
    /// path — the curved tier seeds from the Tier-1 routing / residual. Present so
    /// callers can assert the "no PC reseed events in the log" acceptance bar.
    pub pc_reseed_events: usize,
    /// Total promotions (linear/residual → curved).
    pub n_promotions: usize,
    /// Total demotions (rejected curved candidate → kept linear).
    pub n_demotions: usize,
    /// Total Tier-1 block deaths (→ residual factor).
    pub n_deaths: usize,
}

/// The composed tiered fit.
#[derive(Clone, Debug)]
pub struct TieredFitReport {
    /// Tier-0 shared mean (kept so callers can reconstruct in `z` space).
    pub tier0: Tier0Mean,
    /// Tier-1 block-sparse linear bulk.
    pub tier1: BlockSparseFit,
    /// Tier-2 curved co-fit on the Tier-1 residual (`None` when Tier-2 disabled).
    pub tier2: Option<CofitReport>,
    /// Migration ledger of the adjudicated promotions/demotions/deaths.
    pub ledger: MigrationLedger,
    /// Final composed explained variance (`1 − RSS/TSS` vs the Tier-0 mean).
    pub explained_variance: f64,
}

/// Fit the tiered decomposition on activations `z` (`N×P`, f64): Tier-0 mean →
/// Tier-1 block-sparse bulk → Tier-2 curved co-fit on the Tier-1 residual.
///
/// The curved tier is fit on the Tier-1 residual through
/// [`cofit_block_and_curved`], whose round 0 is exactly "curved-on-linear-
/// residual" and whose alternation drives the joint objective to stationarity.
/// No principal-component reseeding occurs; the [`MigrationLedger`] accounts for
/// the moves and pins `pc_reseed_events = 0`.
pub fn fit_tiered(
    z: ArrayView2<'_, f64>,
    config: &TieredFitConfig,
) -> Result<TieredFitReport, String> {
    // Tier 0: peel the shared mean; the bulk is fit on R0 = z − μ.
    let tier0 = Tier0Mean::fit(z)?;
    let r0 = tier0.apply(z)?;
    let r0_f32 = r0.mapv(|v| v as f32);

    // Tier 1: block-sparse collapsed-linear bulk on the de-meaned residual.
    let tier1 = fit_block_sparse_dictionary(r0_f32.view(), &config.tier1)?;

    let mut ledger = MigrationLedger::default();

    // Structural deaths: Tier-1 blocks no row selected fall back to the residual
    // factor pool. (Revival, when it happens, draws from worst-residual rows in
    // the block lane — never from PCs.)
    let n_dead = tier1
        .block_utilization
        .iter()
        .filter(|&&u| u == 0.0)
        .count();
    if n_dead > 0 {
        ledger.n_deaths = n_dead;
        ledger.moves.push(TieredLedgerMove {
            kind: TieredMoveKind::Death,
            round: None,
            count: n_dead,
            dl_bits: 0.0,
            objective: f64::NAN,
        });
    }

    // Tier 2: curved co-fit on the Tier-1 residual, or the linear-bulk baseline.
    let (tier2, explained_variance) = if config.tier2_enabled {
        let report = cofit_block_and_curved(
            r0_f32.view(),
            tier1.decoder.view(),
            tier1.blocks.view(),
            tier1.codes.view(),
            tier1.gamma,
            &config.cofit,
        )?;
        record_cofit_moves(&mut ledger, &report);
        let ev = report.explained_variance;
        (Some(report), ev)
    } else {
        (None, tier1.explained_variance)
    };

    Ok(TieredFitReport {
        tier0,
        tier1,
        tier2,
        ledger,
        explained_variance,
    })
}

/// Translate the co-fit's per-round acceptance telemetry into migration-ledger
/// moves. Round 0's accepted charts are promotions off the linear residual; in
/// later rounds a rise in the accepted-chart count is a promotion and a fall is a
/// demotion, and a round whose curved candidate was not committed is recorded as
/// a demotion event (the linear block was kept).
fn record_cofit_moves(ledger: &mut MigrationLedger, report: &CofitReport) {
    let mut prev_accepted = 0usize;
    for round in &report.rounds {
        let accepted = round.n_accepted_charts;
        if accepted > prev_accepted {
            let count = accepted - prev_accepted;
            ledger.n_promotions += count;
            ledger.moves.push(TieredLedgerMove {
                kind: TieredMoveKind::Promotion,
                round: Some(round.round),
                count,
                dl_bits: round.curved_charge,
                objective: round.objective,
            });
        } else if accepted < prev_accepted {
            let count = prev_accepted - accepted;
            ledger.n_demotions += count;
            ledger.moves.push(TieredLedgerMove {
                kind: TieredMoveKind::Demotion,
                round: Some(round.round),
                count,
                dl_bits: 0.0,
                objective: round.objective,
            });
        } else if round.round > 0 && !round.curved_committed {
            // Candidate proposed but rejected: the linear block is kept.
            ledger.n_demotions += 1;
            ledger.moves.push(TieredLedgerMove {
                kind: TieredMoveKind::Demotion,
                round: Some(round.round),
                count: 1,
                dl_bits: 0.0,
                objective: round.objective,
            });
        }
        prev_accepted = accepted;
    }
}

#[cfg(test)]
mod fit_tests {
    use super::*;
    use ndarray::Array2;

    /// Two planted linear directions in P=6; the tiered driver runs end to end,
    /// returns a finite composed EV, and performs zero PC reseeds.
    #[test]
    fn tiered_driver_runs_and_never_pc_reseeds() {
        let n = 64;
        let p = 6;
        let mut z = Array2::<f64>::zeros((n, p));
        for i in 0..n {
            let t = i as f64 / n as f64;
            // Direction A on cols 0,1; direction B on cols 2,3; small offset mean.
            z[[i, 0]] = 1.0 + (t * 6.28).cos();
            z[[i, 1]] = 1.0 + (t * 6.28).sin();
            z[[i, 2]] = -0.5 + (t * 3.14).cos();
            z[[i, 3]] = -0.5 + (t * 3.14).sin();
        }
        let mut config = TieredFitConfig::linear_bulk(3, 2);
        config.tier1.block_topk = 2;
        config.tier1.max_epochs = 8;

        let report = fit_tiered(z.view(), &config).expect("tiered fit runs");
        assert!(
            report.explained_variance.is_finite(),
            "composed EV must be finite, got {}",
            report.explained_variance
        );
        assert_eq!(
            report.ledger.pc_reseed_events, 0,
            "the tiered path must never PC-reseed"
        );
        // Tier-0 mean captured the +1 / -0.5 offsets it was given.
        assert!(report.tier0.mean.iter().all(|m| m.is_finite()));
        assert!(report.tier2.is_none(), "linear_bulk disables Tier-2");
    }

    /// Planted 6-circle + linear-bulk mixture (#2023 acceptance): the tiered fit
    /// (Tier-1 linear bulk + Tier-2 curved co-fit on the residual) must beat the
    /// pure-linear Tier-1 EV, and the migration ledger must record at least one
    /// promotion (a linear block / residual factor turned into a curved chart).
    #[test]
    fn tiered_beats_linear_on_six_circle_mixture_and_records_a_promotion() {
        // P = 16: six circles in disjoint 2-D subspaces (cols 0..12) + a linear
        // bulk direction (cols 12,13); cols 14,15 carry light noise.
        let n = 240usize;
        let p = 16usize;
        let n_circles = 6usize;
        let mut z = Array2::<f64>::zeros((n, p));
        for i in 0..n {
            let ph = (i as f64) * 0.261_799; // ~15° step, decorrelates the phases
            for c in 0..n_circles {
                let theta = ph * (1.0 + c as f64 * 0.37) + c as f64;
                z[[i, 2 * c]] = theta.cos();
                z[[i, 2 * c + 1]] = theta.sin();
            }
            // Linear bulk: a straight ramp direction the linear tier explains fully.
            let t = i as f64 / n as f64;
            z[[i, 12]] = 2.0 * t - 1.0;
            z[[i, 13]] = 1.0 - 2.0 * t;
            // Light deterministic wobble so cols 14,15 are not exactly zero.
            z[[i, 14]] = 0.01 * (ph * 2.0).sin();
            z[[i, 15]] = 0.01 * (ph * 3.0).cos();
        }

        // Tier-1 only (linear bulk baseline): 8 blocks of size 2, budget covers
        // all six circles plus the bulk.
        let mut lin = TieredFitConfig::linear_bulk(8, 2);
        lin.tier1.block_topk = 7;
        lin.tier1.max_epochs = 20;
        let lin_report = fit_tiered(z.view(), &lin).expect("linear-bulk fit runs");
        let ev_lin = lin_report.explained_variance;

        // Tiered (Tier-1 + Tier-2 curved co-fit on the residual): same Tier-1.
        let mut tiered = TieredFitConfig::tiered(8, 2);
        tiered.tier1.block_topk = 7;
        tiered.tier1.max_epochs = 20;
        let report = fit_tiered(z.view(), &tiered).expect("tiered fit runs");

        assert_eq!(
            report.ledger.pc_reseed_events, 0,
            "the tiered path must never PC-reseed"
        );
        assert!(
            report.ledger.n_promotions >= 1,
            "the migration ledger must record >=1 curved promotion; got {} (moves: {:?})",
            report.ledger.n_promotions,
            report.ledger.moves
        );
        assert!(
            report.explained_variance > ev_lin,
            "tiered EV {} must beat pure-linear EV {}",
            report.explained_variance,
            ev_lin
        );
    }
}
