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
//! **(b) Curved refinement** — Tier-2 charts the Tier-1 residual `R1 = R0 − L`
//!    through the canonical overcomplete hard-TopK support-sparse engine
//!    ([`run_sae_support_outer`], driven exactly as the public support-sparse fit
//!    entry drives it: [`build_sae_support_seed`] → [`build_sae_support_term_seed`]
//!    → grouped-LAML outer solve). The residual's local mean is peeled before the
//!    fit and added back on reconstruction, so the curved correction `C` lives in
//!    residual space and the composed model is `μ + L + C`. This is the `K > P`
//!    representation — the front door refuses any resident `N×K` alternative — so
//!    the Tier-2 dictionary width must exceed the residual dimension.
//!
//! The unified [`SaeMigrationLedger`] records every retained curved atom as a
//! chart promoted from the Tier-1 linear residual support (a curved birth seeded
//! [`crate::migration_ledger::BirthSeed::LinearAtom`]), the atoms pruned for zero
//! support mass as structural curved deaths, and the Tier-1 block deaths. The
//! support-sparse lane prices complexity through its grouped-LAML smoothing, not a
//! per-move description-length charge, so those curved moves carry no `dl_bits`.
//! `pc_reseed_events` is always `0` on this path.

use ndarray::{Array1, ArrayView2, Axis};

use gam_solve::rho_optimizer::OuterCriterionCertificate;

use crate::front_door::{SaeFitLane, admit_topk_manifold};
use crate::manifold::{
    SaeSupportFixedPointReport, SaeSupportOuterRequest, SaeSupportSeedRequest,
    SaeSupportSparseTerm, SaeSupportTermSeedRequest, build_sae_support_seed,
    build_sae_support_term_seed, run_sae_support_outer, sae_support_effective_atom_dims,
};
use crate::migration_ledger::{BirthSeed, MoveEvidence, MoveReason, MoveStage, SaeMigrationLedger};
use crate::sparse_dict::{
    BlockSeedPolicy, BlockSparseConfig, BlockSparseFit, fit_block_sparse_dictionary_with_seed,
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

/// Tier-2 curved refinement configuration: the overcomplete hard-TopK
/// support-sparse dictionary fit on the Tier-1 residual (#2023). The residual
/// is charted by `n_atoms` curved atoms of the declared `atom_basis`/`atom_dim`
/// family under a per-row `support_k` TopK support, and its smoothing strengths
/// are selected by the grouped-LAML outer engine ([`run_sae_support_outer`]).
///
/// The support-sparse lane is the ONLY representation of a `K > P` TopK curved
/// dictionary (the front door refuses any resident `N×K` alternative), so
/// `n_atoms` must exceed the residual output dimension `P`; a `K ≤ P` request is
/// rejected loudly rather than silently demoted.
#[derive(Clone, Debug)]
pub struct Tier2SupportConfig {
    /// Curved chart family every atom draws from (e.g. `"periodic"` for circular
    /// charts), passed verbatim to the support-sparse atom planner.
    pub atom_basis: String,
    /// Public per-atom dimension (periodic entries are harmonic resolution; the
    /// live chart width is resolved by [`sae_support_effective_atom_dims`]).
    pub atom_dim: usize,
    /// Overcomplete curved dictionary width `K`. Must exceed the residual `P`.
    pub n_atoms: usize,
    /// Per-row TopK curved support width `s` (`1 <= s <= n_atoms`).
    pub support_k: usize,
    /// Initial isotropic smoothing strength seeding the outer LAML search.
    pub initial_smoothness: f64,
    /// Outer (smoothing-selection) iteration budget.
    pub max_outer_iter: usize,
    /// Inner (fixed-point) iteration budget.
    pub max_inner_iter: usize,
    /// Inner fixed-point stationarity tolerance.
    pub inner_tolerance: f64,
    /// Inner coordinate trust radius.
    pub trust_radius: f64,
    /// Deterministic seed for the support routing and Hutchinson trace probes.
    pub random_state: u64,
}

impl Default for Tier2SupportConfig {
    fn default() -> Self {
        Self {
            atom_basis: "periodic".to_string(),
            atom_dim: 1,
            n_atoms: 256,
            support_k: 4,
            initial_smoothness: 1.0,
            max_outer_iter: 64,
            max_inner_iter: 256,
            inner_tolerance: 1.0e-8,
            trust_radius: 1.0,
            random_state: 0xC0FF_EE00_D15E_A5E5,
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
    /// Whether to run the Tier-2 curved refinement on the Tier-1 residual
    /// (`false` ⇒ Tier-0 + Tier-1 only, the linear-bulk baseline).
    pub tier2_enabled: bool,
    /// Tier-2 curved support-sparse refinement configuration (the overcomplete
    /// hard-TopK dictionary fit on the Tier-1 residual).
    pub tier2: Tier2SupportConfig,
}

impl TieredFitConfig {
    /// A Tier-0 + Tier-1 config at `G` blocks of size `b` (Tier-2 disabled).
    pub fn linear_bulk(n_blocks: usize, block_size: usize) -> Self {
        Self {
            tier1: BlockSparseConfig::new(n_blocks, block_size),
            tier1_seed: TieredSeedPolicy::Auto,
            tier2_enabled: false,
            tier2: Tier2SupportConfig::default(),
        }
    }

    /// A Tier-0 + Tier-1 + Tier-2 config at `G` blocks of size `b`.
    pub fn tiered(n_blocks: usize, block_size: usize) -> Self {
        Self {
            tier1: BlockSparseConfig::new(n_blocks, block_size),
            tier1_seed: TieredSeedPolicy::Auto,
            tier2_enabled: true,
            tier2: Tier2SupportConfig::default(),
        }
    }
}

/// Tier-2 curved refinement outcome: the converged overcomplete support-sparse
/// dictionary fit on the Tier-1 residual, carrying the same information the
/// former dense co-fit report exposed — the composed explained variance, the
/// fitted atom states, and the convergence certificate — expressed in the
/// support-sparse engine's terms.
#[derive(Clone, Debug)]
pub struct Tier2SupportFit {
    /// Local mean peeled from the Tier-1 residual before the curved fit; added
    /// back on reconstruction so the curved correction lives in residual space.
    pub mean: Array1<f64>,
    /// Converged support-sparse curved term: one heterogeneous chart per
    /// retained atom, holding its decoder block and coordinates (the atom states).
    pub term: SaeSupportSparseTerm,
    /// Per-atom smoothing strengths selected by the grouped-LAML outer engine.
    pub lambda_smooth: Vec<f64>,
    /// Terminal LAML criterion at the certified smoothing optimum.
    pub criterion: f64,
    /// Inner fixed-point certificate (raw, undamped recurrence at stationarity).
    pub fixed_point: SaeSupportFixedPointReport,
    /// Outer stationarity certificate; [`OuterCriterionCertificate::certifies`]
    /// holds for every returned fit.
    pub outer_certificate: OuterCriterionCertificate,
    /// Outer (smoothing-selection) iterations to the certified optimum.
    pub outer_iterations: usize,
    /// Requested curved dictionary width `K`.
    pub requested_atoms: usize,
    /// Retained curved atoms (occupied support after zero-mass dead-atom pruning).
    pub retained_atoms: usize,
    /// Composed explained variance (`1 − RSS/TSS` of μ + L + C vs the Tier-0 mean).
    pub explained_variance: f64,
}

/// The composed tiered fit.
#[derive(Clone, Debug)]
pub struct TieredFitReport {
    /// Tier-0 shared mean (kept so callers can reconstruct in `z` space).
    pub tier0: Tier0Mean,
    /// Tier-1 block-sparse linear bulk.
    pub tier1: BlockSparseFit,
    /// Tier-2 curved support-sparse refinement on the Tier-1 residual (`None`
    /// when Tier-2 disabled).
    pub tier2: Option<Tier2SupportFit>,
    /// Unified migration ledger of the adjudicated births / deaths / refusals.
    pub ledger: SaeMigrationLedger,
    /// Final composed explained variance (`1 − RSS/TSS` vs the Tier-0 mean).
    pub explained_variance: f64,
}

/// Run the seed policy + curved refinement on activations `z` (`N×P`, f64):
/// Tier-0 mean + Tier-1 block-sparse linear warm start (the seed) → Tier-2 curved
/// support-sparse refinement on the Tier-1 residual.
///
/// **Internal (in-crate) only.** The public tiered FFI/Python surface was deleted
/// in unification Increment 4; this orchestrator is the in-Rust expression of the
/// schedule for the risk-pin tests + `tiered_gpu_scale` example, to be folded into
/// `sae_manifold_fit`'s inner arrow-Schur driver in Increment 5.
///
/// The curved tier is fit on the Tier-1 residual through the canonical
/// support-sparse engine ([`fit_tier2_support`] → [`run_sae_support_outer`]),
/// whose returned fit carries a certified inner fixed point and outer stationarity
/// certificate. No principal-component reseeding occurs; the [`SaeMigrationLedger`]
/// accounts for the curved births / deaths and pins `pc_reseed_events = 0`.
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
    let tier1 = fit_block_sparse_dictionary_with_seed(r0_f32.view(), &config.tier1, seed_policy)?;

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

    // Tier 2: curved support-sparse refinement on the Tier-1 residual, or the
    // linear-bulk baseline.
    let (tier2, explained_variance) = if config.tier2_enabled {
        let fit = fit_tier2_support(r0.view(), &tier1, &config.tier2)?;
        record_support_moves(&mut ledger, &fit);
        let ev = fit.explained_variance;
        (Some(fit), ev)
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

/// Fit the Tier-2 curved refinement: the overcomplete hard-TopK support-sparse
/// dictionary on the Tier-1 residual `R1 = R0 − L`, driven end to end through the
/// canonical support-sparse engine (seed → term → grouped-LAML outer solve),
/// exactly as the public support-sparse fit entry drives it.
///
/// The residual's local mean is peeled before the fit and added back on
/// reconstruction, so the curved correction `C` lives in residual space and the
/// composed model is `μ + L + C`. The returned explained variance measures that
/// composed reconstruction against the Tier-0 mean baseline (`TSS = ‖R0‖²`, since
/// `R0` is exactly de-meaned).
fn fit_tier2_support(
    r0: ArrayView2<'_, f64>,
    tier1: &BlockSparseFit,
    config: &Tier2SupportConfig,
) -> Result<Tier2SupportFit, String> {
    let (n_obs, output_dim) = r0.dim();

    // Tier-1 residual R1 = R0 − L: the curved tier refines exactly the structure
    // the linear bulk left behind.
    let linear = tier1.reconstruct();
    if linear.dim() != (n_obs, output_dim) {
        return Err(format!(
            "fit_tier2_support: Tier-1 reconstruction {:?} does not match residual ({n_obs}, {output_dim})",
            linear.dim()
        ));
    }
    let mut residual = r0.to_owned();
    for row in 0..n_obs {
        for column in 0..output_dim {
            residual[[row, column]] -= linear[[row, column]] as f64;
        }
    }

    // Peel the residual's local mean before charting (added back on reconstruct).
    let mean = residual
        .mean_axis(Axis(0))
        .ok_or_else(|| "fit_tier2_support: residual mean_axis returned None".to_string())?;
    let centered = &residual - &mean.view().insert_axis(Axis(0));

    let requested_atoms = config.n_atoms;
    let atom_basis = vec![config.atom_basis.clone(); requested_atoms];
    let atom_dim = vec![config.atom_dim; requested_atoms];
    let effective_dims = sae_support_effective_atom_dims(&atom_basis, &atom_dim)?;
    let d_max = effective_dims.iter().copied().max().unwrap_or(1);
    let admission =
        admit_topk_manifold(n_obs, output_dim, requested_atoms, d_max, config.support_k)?;
    if admission.lane != SaeFitLane::CurvedStreaming {
        return Err(format!(
            "fit_tier2_support: the curved refinement is the overcomplete support-sparse lane, \
             which requires K > P (CurvedStreaming admission); got lane {:?} at N={n_obs}, \
             P={output_dim}, K={requested_atoms}. Widen the Tier-2 dictionary past the residual \
             dimension",
            admission.lane
        ));
    }
    let seed = build_sae_support_seed(SaeSupportSeedRequest {
        target: centered.view(),
        atom_basis: &atom_basis,
        atom_dim: &atom_dim,
        support_k: config.support_k,
        random_state: config.random_state,
        admission,
    })?;
    let retained_atom_indices = seed.retained_atom_indices;
    let retained_atoms = retained_atom_indices.len();
    let retained_basis = retained_atom_indices
        .iter()
        .map(|&atom| atom_basis[atom].clone())
        .collect::<Vec<_>>();
    let retained_dim = retained_atom_indices
        .iter()
        .map(|&atom| atom_dim[atom])
        .collect::<Vec<_>>();
    let term_seed = build_sae_support_term_seed(SaeSupportTermSeedRequest {
        assignment: seed.assignment,
        atom_basis: retained_basis,
        atom_dim: retained_dim,
        output_dim,
        random_state: config.random_state,
    })?;
    let ard_precisions = (0..term_seed.term.k_atoms())
        .map(|atom| vec![1.0; term_seed.term.assignment.atom_coord_dim(atom)])
        .collect::<Vec<_>>();
    let outer = run_sae_support_outer(SaeSupportOuterRequest {
        term: term_seed.term,
        target: centered.clone(),
        initial_smoothness: config.initial_smoothness,
        ard_precisions,
        max_outer_iter: config.max_outer_iter,
        max_inner_iter: config.max_inner_iter,
        inner_tolerance: config.inner_tolerance,
        trust_radius: config.trust_radius,
        random_state: config.random_state,
    })
    .map_err(|error| error.to_string())?;

    // Composed residual against R0 is exactly the curved fit's residual on the
    // centered target: (R0 − L) − (mean + Ĉ) = centered − Ĉ.
    let curved_centered = outer.term.reconstruct()?;
    let mut rss = 0.0f64;
    for row in 0..n_obs {
        for column in 0..output_dim {
            let delta = centered[[row, column]] - curved_centered[[row, column]];
            rss += delta * delta;
        }
    }
    let tss = r0.iter().map(|value| value * value).sum::<f64>();
    let explained_variance = if tss > 0.0 { 1.0 - rss / tss } else { f64::NAN };

    Ok(Tier2SupportFit {
        mean,
        term: outer.term,
        lambda_smooth: outer.lambda_smooth,
        criterion: outer.criterion,
        fixed_point: outer.fixed_point,
        outer_certificate: outer.outer_certificate,
        outer_iterations: outer.outer_iterations,
        requested_atoms,
        retained_atoms,
        explained_variance,
    })
}

/// Translate the Tier-2 support-sparse outcome into unified migration-ledger
/// moves. Every retained curved atom is a chart promoted from the Tier-1 linear
/// residual support ([`BirthSeed::LinearAtom`] — never a principal component, so
/// the `pc_reseed_events` invariant holds by construction); the atoms pruned for
/// zero support mass at the seed boundary are structural curved deaths
/// ([`MoveReason::DeadRouting`]). The support-sparse lane prices complexity
/// through its grouped-LAML smoothing rather than a per-move description-length
/// charge, so the moves carry no `dl_bits` evidence (an unscored structural tally,
/// not a fabricated charge).
fn record_support_moves(ledger: &mut SaeMigrationLedger, fit: &Tier2SupportFit) {
    if fit.retained_atoms > 0 {
        ledger.birth(
            MoveStage::Curved,
            BirthSeed::LinearAtom,
            fit.retained_atoms,
            Some(0),
            MoveEvidence::none(),
            fit.criterion,
        );
    }
    let pruned = fit.requested_atoms - fit.retained_atoms;
    if pruned > 0 {
        ledger.death(
            MoveStage::Curved,
            MoveReason::DeadRouting,
            pruned,
            None,
            MoveEvidence::none(),
            fit.criterion,
        );
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
        // #2275 TRICHOTOMY, best-effort arm: K=6 over ~2 planted planes is over-complete,
        // so the fit reaches its EV plateau (returns) while the spurious blocks keep the
        // frame residual open. The honest verdict is `certified = false` with the open
        // frame residual RECORDED (finite) — a returnable best-effort fit, not an error
        // and not a false "certified". (The certified arm is exercised by the
        // exactly-determined block-lane fits in `block_tests.rs`.)
        assert!(
            !report.tier1.convergence.certified,
            "an over-complete linear-bulk fit is BEST-EFFORT (certified=false); got certified=true, frame_residual={} tol={}",
            report.tier1.convergence.frame_residual,
            report.tier1.convergence.tolerance
        );
        assert!(
            report.tier1.convergence.frame_residual.is_finite(),
            "the open frame residual must be recorded (finite) on a best-effort certificate"
        );
    }

    /// #2275: at `K ≫ intrinsic-rank` the frame-projector fixed point legitimately
    /// does not certify — ~`K − rank` blocks are structurally spurious and AuxK
    /// revival churns their frames every epoch, pinning `frame_residual` above
    /// tolerance. The fit's OBJECTIVE (reconstruction EV / routing scale) still
    /// reaches its achievable plateau, so the tiered driver RETURNS a Tier-1 fit
    /// carrying a typed OPEN certificate (`certified = false`) and runs Tier-2 on its
    /// residual — it does NOT collapse to `Err` and skip Tier-2 (the wrong contract
    /// the #2023 checkpoint sweep installed and laundered green by inverting this
    /// test; see the #2275 history on fba60f1f2/c21cc2c77).
    #[test]
    fn tiered_returns_best_effort_open_certificate_at_k_gg_rank_2275() {
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

        // The objective plateaus, so the tiered fit RETURNS instead of erroring — that
        // IS the #2275 acceptance criterion.
        let report = fit_tiered(z.view(), &config)
            .expect("#2275: best-effort tiered fit must RETURN at K ≫ rank, not error");

        // Typed OPEN certificate: not certified, and the frame residual quantifies how
        // open — while the objective residuals sit at their achievable plateau.
        assert!(
            !report.tier1.convergence.certified,
            "K ≫ rank fit must carry an OPEN certificate; got certified=true              (frame_residual={}, tol={})",
            report.tier1.convergence.frame_residual,
            report.tier1.convergence.tolerance
        );
        assert!(
            report.tier1.convergence.frame_residual > report.tier1.convergence.tolerance,
            "an open certificate must report frame_residual above tolerance; got {} <= {}",
            report.tier1.convergence.frame_residual,
            report.tier1.convergence.tolerance
        );
        // Best-effort (arm 2) means the EV reached its achievable PLATEAU
        // (captured-fraction stationarity), NOT that ev_residual closed to the absolute
        // tolerance — at K >> rank it plateaus above tol just as the frame does. The
        // residual is RECORDED (finite); "no tolerance softening" is pinned by the
        // exact-tolerance assertion below.
        assert!(
            report.tier1.convergence.ev_residual.is_finite(),
            "the plateaued objective residual must be recorded (finite); got {}",
            report.tier1.convergence.ev_residual
        );
        assert!(
            report.tier1.explained_variance.is_finite(),
            "best-effort Tier-1 EV must be finite"
        );
        // Tier-2 RAN on the best-effort Tier-1 residual — the clobbered contract never
        // reached it.
        assert!(
            report.tier2.is_some(),
            "#2275: Tier-2 must run on the best-effort Tier-1 residual"
        );
        assert!(
            report.explained_variance.is_finite(),
            "composed EV must be finite on the best-effort path"
        );
        // No tolerance softening: the open certificate is measured against the SAME
        // configured tolerance, unchanged.
        assert_eq!(
            report.tier1.convergence.tolerance, config.tier1.tolerance,
            "#2275 must NOT soften tolerance; the open certificate uses the configured tol"
        );
    }

    /// #2275: at `K ≫ intrinsic-rank` the block entry RETURNS the best-effort fit with
    /// a typed OPEN certificate (`certified = false`, frame residual above tolerance,
    /// objective residuals at their achievable plateau) — it does NOT collapse the
    /// objective-converged iterate to `Err`. The convergence decision is the
    /// gauge-invariant objective plateau, not an absolute floor on the frame
    /// fixed-point residual that an over-complete frame cannot reach (#2023/#2275).
    #[test]
    fn block_sparse_open_fixed_point_returns_open_certificate_2275() {
        use crate::sparse_dict::{
            BlockSeedPolicy, BlockSparseConfig, fit_block_sparse_dictionary_with_seed,
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

        let fit = fit_block_sparse_dictionary_with_seed(
            x.view(),
            &config,
            BlockSeedPolicy::FarthestPoint,
        )
        .expect("#2275: the block entry must RETURN the objective-converged open fit");
        let c = &fit.convergence;
        assert!(
            !c.certified,
            "a K ≫ rank fit must carry an OPEN certificate (certified=false); got              certified=true (frame_residual={}, tol={})",
            c.frame_residual, c.tolerance
        );
        assert!(
            c.frame_residual > c.tolerance,
            "an open certificate must report frame_residual above tolerance; got {} <= {}",
            c.frame_residual, c.tolerance
        );
        assert!(
            c.ev_residual.is_finite(),
            "the plateaued objective residual must be recorded (finite); got {}",
            c.ev_residual
        );
        // No tolerance softening: the certificate is measured against the configured tol.
        assert_eq!(c.tolerance, config.tolerance, "#2275 must NOT soften tolerance");
    }

    /// Planted 6-circle + linear-bulk mixture (#2023 acceptance): the tiered fit
    /// (Tier-1 linear bulk + Tier-2 curved support-sparse refinement on the
    /// residual) must not regress the pure-linear Tier-1 EV, its Tier-2 must return
    /// a certified support-sparse fit, and the migration ledger must record the
    /// retained curved atoms as promotions off the linear residual.
    #[test]
    fn tiered_curved_refinement_is_certified_and_records_promotions() {
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

        // Tiered (Tier-1 + Tier-2 curved support-sparse refinement): same Tier-1.
        // The curved dictionary is overcomplete (K = 24 > P = 16), the K>P
        // support-sparse lane the front door admits.
        let mut tiered = TieredFitConfig::tiered(8, 2);
        tiered.tier1.block_topk = 7;
        tiered.tier1.aux_k = 3;
        tiered.tier1.max_epochs = 200;
        tiered.tier2.n_atoms = 24;
        tiered.tier2.support_k = 2;
        tiered.tier2.max_outer_iter = 24;
        tiered.tier2.max_inner_iter = 128;
        let report = fit_tiered(z.view(), &tiered).expect("tiered fit runs");

        let tier2 = report.tier2.as_ref().expect("Tier-2 curved refinement ran");
        // The support-sparse engine only returns a certified fixed point + outer
        // stationarity certificate; a returned Tier-2 IS the certified path.
        assert!(
            tier2.outer_certificate.certifies() && tier2.outer_certificate.is_stationary(),
            "Tier-2 must carry a certifying outer stationarity certificate"
        );
        assert!(
            tier2.fixed_point.recurred,
            "Tier-2 inner fixed point must have recurred"
        );
        assert!(
            tier2.retained_atoms >= 1 && tier2.term.k_atoms() == tier2.retained_atoms,
            "Tier-2 must retain >=1 occupied curved atom (got {})",
            tier2.retained_atoms
        );

        assert_eq!(
            report.ledger.pc_reseed_events, 0,
            "the tiered path must never PC-reseed"
        );
        assert_eq!(
            report.ledger.n_births, tier2.retained_atoms,
            "every retained curved atom is a promotion off the linear residual"
        );
        // A curved refinement (which also peels the residual's own mean) can never
        // do worse than the pure-linear tier it refines.
        assert!(
            report.explained_variance >= ev_lin - 1.0e-9,
            "tiered EV {} must not regress pure-linear EV {}",
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
        // return Ok for the Tier-2 refinement to run).
        config.tier1.aux_k = 3;
        config.tier1.max_epochs = 200;
        // Overcomplete curved dictionary (K = 24 > P = 16) for the support-sparse lane.
        config.tier2.n_atoms = 24;
        config.tier2.support_k = 2;
        config.tier2.max_outer_iter = 24;
        config.tier2.max_inner_iter = 128;
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
        let tier2 = report
            .tier2
            .as_ref()
            .expect("tiered config must run Tier-2");
        assert!(
            tier2.outer_certificate.certifies(),
            "the Tier-2 support-sparse refinement must return a certified fit"
        );
    }

    /// Focused #2023 gate: on a tiny two-circle fixture the Tier-2 branch drives
    /// the overcomplete support-sparse engine (never the dense co-fit) end to end,
    /// and its report carries the support-sparse provenance — a converged term with
    /// occupied curved atoms, a certifying outer stationarity certificate, a
    /// recurred inner fixed point, and per-atom smoothing — mapped into the
    /// migration ledger as linear-residual curved promotions.
    #[test]
    fn tier2_branch_constructs_the_support_sparse_path() {
        // P = 4: two disjoint planted circles (cols 0,1 and 2,3) at different
        // frequencies. Tier-1 charts each plane linearly; the residual it leaves
        // (the circles' curvature) is exactly what the curved Tier-2 refines.
        let n = 96usize;
        let p = 4usize;
        let mut z = Array2::<f64>::zeros((n, p));
        for i in 0..n {
            let ph = (i as f64) * 0.19;
            z[[i, 0]] = ph.cos();
            z[[i, 1]] = ph.sin();
            z[[i, 2]] = (1.7 * ph).cos();
            z[[i, 3]] = (1.7 * ph).sin();
        }

        let mut config = TieredFitConfig::tiered(2, 2);
        config.tier1.block_topk = 2;
        config.tier1.aux_k = 2;
        config.tier1.max_epochs = 200;
        // Overcomplete curved dictionary (K = 8 > P = 4): the K>P support-sparse
        // lane, the only representation the front door admits for it.
        config.tier2.atom_basis = "periodic".to_string();
        config.tier2.atom_dim = 1;
        config.tier2.n_atoms = 8;
        config.tier2.support_k = 2;
        config.tier2.max_outer_iter = 32;
        config.tier2.max_inner_iter = 256;

        let report = fit_tiered(z.view(), &config).expect("tiny two-circle tiered fit runs");
        let tier2 = report
            .tier2
            .as_ref()
            .expect("the Tier-2 curved refinement branch must have run");

        // Support-sparse provenance: a certified fit with occupied atom states.
        assert!(
            tier2.outer_certificate.certifies() && tier2.outer_certificate.is_stationary(),
            "Tier-2 must return a certifying outer stationarity certificate"
        );
        assert!(
            tier2.fixed_point.recurred,
            "Tier-2 inner fixed point must have recurred"
        );
        assert!(
            tier2.retained_atoms >= 1
                && tier2.retained_atoms <= tier2.requested_atoms
                && tier2.term.k_atoms() == tier2.retained_atoms,
            "Tier-2 must retain 1..={} occupied curved atoms (got {})",
            tier2.requested_atoms,
            tier2.retained_atoms
        );
        assert_eq!(
            tier2.lambda_smooth.len(),
            tier2.term.k_atoms(),
            "each retained curved atom carries its selected smoothing strength"
        );
        assert_eq!(tier2.mean.len(), p, "the peeled residual mean spans P");
        assert!(
            tier2.term.atoms.iter().all(|atom| atom
                .decoder_coefficients
                .iter()
                .all(|value| value.is_finite())),
            "every retained curved atom must carry finite decoder coefficients"
        );

        // Ledger provenance: retained atoms are curved promotions off the linear
        // residual, never PC reseeds.
        assert_eq!(
            report.ledger.n_births, tier2.retained_atoms,
            "every retained curved atom is one curved birth"
        );
        assert_eq!(
            report.ledger.pc_reseed_events, 0,
            "the support-sparse Tier-2 path must never PC-reseed"
        );
        assert!(
            report.explained_variance.is_finite(),
            "composed EV must be finite, got {}",
            report.explained_variance
        );
    }
}
