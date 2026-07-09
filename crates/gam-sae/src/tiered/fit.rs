//! Tiered fit as a **seed policy + alternation cadence** of the unified engine
//! (#2023; unification #2232 Increment 4).
//!
//! "Tiered" is not a separate model or a separate public API — it is a *schedule*
//! that the one SAE engine runs: the residual tier is a **round-0 warm start**,
//! and the alternation to joint stationarity is the fit. Increment 4 deleted the
//! public tiered surface (the `sae_manifold_fit_tiered` FFI entry and its
//! `gamfit._sae_spectral` Python wrapper); the tiered flow is now reached only
//! through the unified engine + the seed/cadence composition (`examples/
//! compose_tiers.py`: linear warm start via `sparse_dictionary_fit` → curved
//! alternation via `sae_manifold_fit`).
//!
//! This module keeps the in-crate orchestrator [`fit_tiered`] that expresses that
//! schedule directly in Rust, used by the risk-pin tests + the `tiered_gpu_scale`
//! example. It is **internal, not public API**, and is slated to fold into
//! `sae_manifold_fit`'s inner arrow-Schur driver in Increment 5 (the fold needs
//! the `sparse_dict` inner-solve seam owned elsewhere + the central build loop);
//! it survives here only as the delegating expression of the two schedule phases:
//!
//! **(a) Seed policy** — Tier-0 peels the shared column mean ([`Tier0Mean`]; the
//!    bulk is fit on `R0 = z − μ`), then Tier-1 warm-starts the linear bulk: the
//!    block-sparse collapsed-linear dictionary ([`fit_block_sparse_dictionary`])
//!    at width `K = G·b`, the linear-atom special case of the one dictionary.
//!    Births only ever draw from this residual-factor pool — never a principal
//!    component — so `pc_reseed_events == 0` holds by construction.
//!
//! **(b) Alternation cadence** — Tier-2 curved atoms are fit on the Tier-1
//!    residual via [`cofit_block_and_curved`]: round 0 is the curved-on-linear-
//!    residual warm start and rounds ≥1 alternate the two blocks to joint
//!    stationarity. This is the same inner/outer descent the unified engine runs,
//!    warm-started from the seed above.
//!
//! The unified [`SaeMigrationLedger`] records the curved-tier promotions
//! (births) / demotions (refusals) that the cadence adjudicates each round in the
//! matched-description-length currency (`curved_charge`, the e-BH acceptance
//! charge, banked as `dl_bits`), plus the Tier-1 block deaths.
//! `pc_reseed_events` is always `0` on this path.

use ndarray::ArrayView2;

use crate::migration_ledger::{BirthSeed, MoveEvidence, MoveReason, MoveStage, SaeMigrationLedger};
use crate::sparse_dict::{
    BlockSparseConfig, BlockSparseFit, CofitConfig, CofitReport, cofit_block_and_curved,
    fit_block_sparse_dictionary,
};
use crate::tiered::Tier0Mean;

/// Configuration for [`fit_tiered`]. Internal (in-crate) only — the public
/// tiered surface was removed in unification Increment 4; this is the seed/cadence
/// schedule config, slated to fold into `sae_manifold_fit`'s driver in Inc 5.
#[derive(Clone, Debug)]
pub struct TieredFitConfig {
    /// Tier-1 block-sparse dictionary configuration (`G` blocks of size `b`, the
    /// block budget `k`, epochs, minibatch/tile geometry). GPU score-routing is
    /// governed by the process-wide [`gam_gpu::GpuPolicy`] (`gam_gpu::configure_global_policy`),
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

/// The composed tiered fit.
#[derive(Clone, Debug)]
pub struct TieredFitReport {
    /// Tier-0 shared mean (kept so callers can reconstruct in `z` space).
    pub tier0: Tier0Mean,
    /// Tier-1 block-sparse linear bulk.
    pub tier1: BlockSparseFit,
    /// Tier-2 curved co-fit on the Tier-1 residual (`None` when Tier-2 disabled).
    pub tier2: Option<CofitReport>,
    /// Unified migration ledger of the adjudicated births / deaths / refusals.
    pub ledger: SaeMigrationLedger,
    /// Final composed explained variance (`1 − RSS/TSS` vs the Tier-0 mean).
    pub explained_variance: f64,
}

/// Run the seed policy + alternation cadence on activations `z` (`N×P`, f64):
/// Tier-0 mean + Tier-1 block-sparse linear warm start (the seed) → Tier-2 curved
/// co-fit on the Tier-1 residual (the cadence).
///
/// **Internal (in-crate) only.** The public tiered FFI/Python surface was deleted
/// in unification Increment 4; this orchestrator is the in-Rust expression of the
/// schedule for the risk-pin tests + `tiered_gpu_scale` example, to be folded into
/// `sae_manifold_fit`'s inner arrow-Schur driver in Increment 5.
///
/// The curved tier is fit on the Tier-1 residual through
/// [`cofit_block_and_curved`], whose round 0 is exactly "curved-on-linear-
/// residual" and whose alternation drives the joint objective to stationarity.
/// No principal-component reseeding occurs; the [`SaeMigrationLedger`] accounts
/// for the moves and pins `pc_reseed_events = 0`.
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

    let mut ledger = SaeMigrationLedger::new();

    // Structural deaths: Tier-1 blocks no row selected fall back to the residual
    // factor pool. (Revival, when it happens, draws from worst-residual rows in
    // the block lane — never from PCs.)
    let n_dead = tier1
        .block_utilization
        .iter()
        .filter(|&&u| u == 0.0)
        .count();
    if n_dead > 0 {
        ledger.death(
            MoveStage::Linear,
            MoveReason::DeadRouting,
            n_dead,
            None,
            MoveEvidence::none(),
            f64::NAN,
        );
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

/// Translate the co-fit's per-round acceptance telemetry into unified
/// migration-ledger moves. Round 0's accepted charts are births off the linear
/// residual (a linear atom promoted to a curved chart); in later rounds a rise in
/// the accepted-chart count is a birth and a fall is a refusal (the curved
/// candidate was demoted back to the linear block), and a round whose curved
/// candidate was not committed is a refusal (the linear block was kept). Curved
/// births seed from the Tier-1 linear routing — never a principal component — so
/// the ledger's `pc_reseed_events` invariant holds by construction.
fn record_cofit_moves(ledger: &mut SaeMigrationLedger, report: &CofitReport) {
    let mut prev_accepted = 0usize;
    for round in &report.rounds {
        let accepted = round.n_accepted_charts;
        if accepted > prev_accepted {
            let count = accepted - prev_accepted;
            ledger.birth(
                MoveStage::Curved,
                BirthSeed::LinearAtom,
                count,
                Some(round.round),
                MoveEvidence::from_dl_bits(round.curved_charge),
                round.objective,
            );
        } else if accepted < prev_accepted {
            let count = prev_accepted - accepted;
            ledger.refuse(
                MoveStage::Curved,
                MoveReason::EvidenceInsufficient,
                count,
                Some(round.round),
                MoveEvidence::none(),
                round.objective,
            );
        } else if round.round > 0 && !round.curved_committed {
            // Candidate proposed but rejected: the linear block is kept.
            ledger.refuse(
                MoveStage::Curved,
                MoveReason::EvidenceInsufficient,
                1,
                Some(round.round),
                MoveEvidence::none(),
                round.objective,
            );
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
            report.ledger.n_births >= 1,
            "the migration ledger must record >=1 curved birth; got {} (moves: {:?})",
            report.ledger.n_births,
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
