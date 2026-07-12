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
//! matched-description-length currency (`curved_charge`, the per-chart BIC
//! complexity charge — a descriptive selection gate, not an FDR-controlled e-BH
//! discovery — banked as `dl_bits`), plus the Tier-1 block deaths.
//! `pc_reseed_events` is always `0` on this path.

use ndarray::ArrayView2;

use crate::migration_ledger::{BirthSeed, MoveEvidence, MoveReason, MoveStage, SaeMigrationLedger};
use crate::sparse_dict::{
    BlockSeedPolicy, BlockSparseConfig, BlockSparseFit, CofitConfig, CofitReport,
    cofit_block_and_curved, fit_block_sparse_dictionary_with_seed,
};
use crate::tiered::Tier0Mean;

/// Serial farthest-point block seed budget in element-ops (`N·P·G·b`). Above this
/// the `O(N·P·K)` corpus pass dominates the whole Tier-1 fit (measured to be the
/// scaling wall at `K ≈ 1e4`, unrelated to routing), so [`TieredSeedPolicy::Auto`]
/// switches to the `O(K·b)` coordinate-partition seed. Below it the data-aware
/// farthest-point seed is affordable and gives the more coherent starting blocks.
const FARTHEST_POINT_SEED_MAX_OPS: u128 = 1_000_000_000;

/// How Tier-1 seeds its `K = G·b` block frames. The default [`Auto`] keeps the
/// data-aware farthest-point seed at small/moderate `K` and switches to the cheap
/// coordinate-partition seed once the serial farthest-point pass would dominate —
/// the "Tier-1 K>small" entry that makes a `K ≈ 1e4` tiered fit tractable end to end
/// without a caller flag (#2023).
///
/// [`Auto`]: TieredSeedPolicy::Auto
#[derive(Clone, Copy, Debug, PartialEq, Eq, Default)]
pub enum TieredSeedPolicy {
    /// Pick by the farthest-point seed cost `N·P·G·b` against
    /// [`FARTHEST_POINT_SEED_MAX_OPS`].
    #[default]
    Auto,
    /// Force the data-aware farthest-point seed regardless of `K`.
    FarthestPoint,
    /// Force the cheap coordinate-partition seed regardless of `K`.
    CoordinatePartition,
}

impl TieredSeedPolicy {
    /// Resolve to a concrete [`BlockSeedPolicy`] for a corpus of `n` rows and the
    /// Tier-1 block geometry (`G` blocks of size `b` in `ℝ^P`).
    fn resolve(self, n: usize, p: usize, config: &BlockSparseConfig) -> BlockSeedPolicy {
        match self {
            TieredSeedPolicy::FarthestPoint => BlockSeedPolicy::FarthestPoint,
            TieredSeedPolicy::CoordinatePartition => BlockSeedPolicy::CoordinatePartition,
            TieredSeedPolicy::Auto => {
                let ops = (n as u128)
                    * (p as u128)
                    * (config.n_blocks as u128)
                    * (config.block_size as u128);
                if ops > FARTHEST_POINT_SEED_MAX_OPS {
                    BlockSeedPolicy::CoordinatePartition
                } else {
                    BlockSeedPolicy::FarthestPoint
                }
            }
        }
    }
}

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
    /// How Tier-1 seeds its `K` block frames. [`TieredSeedPolicy::Auto`] (default)
    /// switches to the cheap coordinate-partition seed once the serial
    /// farthest-point pass would dominate, so a `K ≈ 1e4` tiered fit runs end to end.
    pub tier1_seed: TieredSeedPolicy,
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
            tier1_seed: TieredSeedPolicy::Auto,
            tier2_enabled: false,
            cofit: CofitConfig::default(),
        }
    }

    /// A Tier-0 + Tier-1 + Tier-2 config at `G` blocks of size `b`.
    pub fn tiered(n_blocks: usize, block_size: usize) -> Self {
        Self {
            tier1: BlockSparseConfig::new(n_blocks, block_size),
            tier1_seed: TieredSeedPolicy::Auto,
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

    // Tier 1: block-sparse collapsed-linear bulk on the de-meaned residual. The
    // seed policy resolves against the corpus size + block geometry so a K≈1e4 bulk
    // skips the serial O(N·P·K) farthest-point pass (the large-K entry, #2023).
    let seed_policy = config
        .tier1_seed
        .resolve(r0_f32.nrows(), r0_f32.ncols(), &config.tier1);
    let tier1 =
        fit_block_sparse_dictionary_with_seed(r0_f32.view(), &config.tier1, seed_policy)?;

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
        // K=6 over ~2 planted planes leaves under-utilised blocks; give the frame
        // fixed point AuxK revival + enough epochs to certify (the same budget the
        // block-lane `coordinate_partition_seed_fits_end_to_end` test certifies at).
        config.tier1.aux_k = 3;
        config.tier1.max_epochs = 200;

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
        assert!(
            report.tier1.convergence.frame_residual <= report.tier1.convergence.tolerance,
            "every returned tiered fit must carry a closed frame certificate"
        );
    }

    /// #2275: at `K ≫ intrinsic-rank` the frame-projector fixed point legitimately
    /// does not certify — ~`K − rank` blocks are structurally spurious and AuxK
    /// revival churns their frames every epoch, pinning `frame_residual` above
    /// tolerance. Such an iterate must be refused before Tier-2 consumes it.
    #[test]
    fn tiered_refuses_open_certificate_at_k_gg_rank_2275() {
        // Rank-1 planted structure (a single direction in cols 0,1) in P=8, fit with
        // K = G·b = 16 blocks of size b=1: ~15 blocks are structurally spurious.
        let n = 96usize;
        let p = 8usize;
        let mut z = Array2::<f64>::zeros((n, p));
        for i in 0..n {
            let t = (i as f64) * 0.2;
            z[[i, 0]] = t.cos();
            z[[i, 1]] = t.sin();
        }
        let mut config = TieredFitConfig::tiered(16, 1); // K=16 ≫ intrinsic rank
        config.tier1.block_topk = 4;
        config.tier1.aux_k = 4; // revival ON: spurious frames churn -> cannot certify
        config.tier1.max_epochs = 40;

        fit_tiered(z.view(), &config)
            .expect_err("Tier-2 must never consume an unconverged Tier-1 iterate");
    }

    /// #2275: the block entry returns typed nonconvergence with the configured
    /// tolerance and open residual; it never mints an uncertified fit.
    #[test]
    fn block_sparse_open_fixed_point_is_a_typed_error_2275() {
        use crate::sparse_dict::{
            BlockSeedPolicy, BlockSparseConfig, BlockSparseFitError,
            fit_block_sparse_dictionary_with_seed,
        };
        let n = 96usize;
        let p = 8usize;
        let mut x = Array2::<f32>::zeros((n, p));
        for i in 0..n {
            let t = (i as f32) * 0.2;
            x[[i, 0]] = t.cos();
            x[[i, 1]] = t.sin();
        }
        let mut config = BlockSparseConfig::new(16, 1);
        config.block_topk = 4;
        config.aux_k = 4;
        config.max_epochs = 40;

        let err = fit_block_sparse_dictionary_with_seed(
            x.view(),
            &config,
            BlockSeedPolicy::FarthestPoint,
        )
        .expect_err("Err-contract entry must reject the non-certified K ≫ rank fit");
        match err {
            BlockSparseFitError::NonConvergence {
                frame_residual,
                tolerance,
                ..
            } => {
                assert_eq!(tolerance, config.tolerance);
                assert!(
                    frame_residual > tolerance,
                    "the typed error must report the open frame residual"
                );
            }
            other => panic!("expected NonConvergence, got {other:?}"),
        }
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
        // 8 blocks over 7 planes (6 circles + bulk) leaves a spare block; AuxK
        // revival + enough epochs let the frame fixed point certify so fit_tiered
        // returns rather than erroring on non-convergence.
        lin.tier1.aux_k = 3;
        lin.tier1.max_epochs = 200;
        let lin_report = fit_tiered(z.view(), &lin).expect("linear-bulk fit runs");
        let ev_lin = lin_report.explained_variance;

        // Tiered (Tier-1 + Tier-2 curved co-fit on the residual): same Tier-1.
        let mut tiered = TieredFitConfig::tiered(8, 2);
        tiered.tier1.block_topk = 7;
        tiered.tier1.aux_k = 3;
        tiered.tier1.max_epochs = 200;
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

    /// `TieredSeedPolicy::Auto` keeps the data-aware farthest-point seed at small
    /// `K` and switches to the cheap coordinate-partition seed once the serial
    /// `N·P·G·b` pass would blow the budget — the "Tier-1 K>small" entry decision.
    #[test]
    fn auto_seed_switches_at_the_farthest_point_budget() {
        // Small geometry (well under the 1e9-op budget) → farthest-point.
        let small = TieredFitConfig::linear_bulk(8, 2);
        assert_eq!(
            small.tier1_seed.resolve(240, 16, &small.tier1),
            BlockSeedPolicy::FarthestPoint,
            "small-K tiered fit must keep the data-aware seed"
        );
        // K≈1e4 at the #2023 target width (N=1e5, P=64) → N·P·G·b ≫ 1e9 → cheap seed.
        let large = TieredFitConfig::linear_bulk(2_500, 4);
        assert_eq!(
            large.tier1_seed.resolve(100_000, 64, &large.tier1),
            BlockSeedPolicy::CoordinatePartition,
            "large-K tiered fit must switch to the coordinate-partition seed"
        );
        // Explicit overrides ignore the budget.
        let mut forced = TieredFitConfig::linear_bulk(2_500, 4);
        forced.tier1_seed = TieredSeedPolicy::FarthestPoint;
        assert_eq!(
            forced.tier1_seed.resolve(100_000, 64, &forced.tier1),
            BlockSeedPolicy::FarthestPoint
        );
        forced.tier1_seed = TieredSeedPolicy::CoordinatePartition;
        assert_eq!(
            forced.tier1_seed.resolve(240, 16, &forced.tier1),
            BlockSeedPolicy::CoordinatePartition
        );
    }

    /// The coordinate-partition seed carries a full tiered fit end to end (Tier-0
    /// mean → Tier-1 bulk on the cheap seed → Tier-2 curved co-fit on the residual),
    /// producing a finite composed EV and never PC-reseeding. This is the large-`K`
    /// entry's fit path exercised at a small `K` (the seed is what changes, not the
    /// engine), so the test stays fast while still driving every stage.
    #[test]
    fn coordinate_seed_carries_a_full_tiered_fit() {
        let n = 240usize;
        let p = 16usize;
        let n_circles = 6usize;
        let mut z = Array2::<f64>::zeros((n, p));
        for i in 0..n {
            let ph = (i as f64) * 0.261_799;
            for c in 0..n_circles {
                let theta = ph * (1.0 + c as f64 * 0.37) + c as f64;
                z[[i, 2 * c]] = theta.cos();
                z[[i, 2 * c + 1]] = theta.sin();
            }
            let t = i as f64 / n as f64;
            z[[i, 12]] = 2.0 * t - 1.0;
            z[[i, 13]] = 1.0 - 2.0 * t;
            z[[i, 14]] = 0.01 * (ph * 2.0).sin();
            z[[i, 15]] = 0.01 * (ph * 3.0).cos();
        }

        let mut config = TieredFitConfig::tiered(8, 2);
        config.tier1_seed = TieredSeedPolicy::CoordinatePartition;
        config.tier1.block_topk = 7;
        // The coordinate seed is a colder start than farthest-point, so give the
        // frame fixed point AuxK revival + enough epochs to certify (Tier-1 must
        // return Ok for the Tier-2 co-fit to run).
        config.tier1.aux_k = 3;
        config.tier1.max_epochs = 200;
        let report =
            fit_tiered(z.view(), &config).expect("coordinate-seeded tiered fit runs end to end");
        assert!(
            report.explained_variance.is_finite() && report.explained_variance > 0.0,
            "coordinate-seeded composed EV must be finite and positive, got {}",
            report.explained_variance
        );
        assert_eq!(
            report.ledger.pc_reseed_events, 0,
            "the coordinate-seeded tiered path must never PC-reseed"
        );
        assert!(report.tier2.is_some(), "tiered config must run Tier-2");
    }
}
