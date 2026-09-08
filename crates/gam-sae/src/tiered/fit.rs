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
//! example. Increment 5 folded its **seed half** into the public entry: the seed
//! policy is now the standalone [`fit_linear_peel`] stage, which the public
//! support-sparse fit runs before seeding the curved engine (so the two lanes
//! share one declaration of the peel instead of two spellings of it), while
//! [`fit_tiered`] remains the in-crate expression of the full cadence:
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

use ndarray::{Array1, Array2, ArrayView2, Axis};

use gam_solve::rho_optimizer::OuterCriterionCertificate;

use crate::front_door::{SaeFitLane, admit_topk_manifold};
use crate::manifold::{
    SAE_SUPPORT_INNER_FIXED_POINT_MAX_ITER, SaeSupportFixedPointReport, SaeSupportOuterRequest,
    SaeSupportSeedRequest, SaeSupportSparseTerm, SaeSupportTermSeedRequest, build_sae_support_seed,
    build_sae_support_term_seed, run_sae_support_outer, sae_support_effective_atom_dims,
};
use crate::migration_ledger::{BirthSeed, MoveEvidence, MoveReason, MoveStage, SaeMigrationLedger};
use crate::sparse_dict::{
    BlockSeedPolicy, BlockSparseConfig, BlockSparseFit, block_sparse_dictionary_transform,
    fit_block_sparse_dictionary_with_seed, reconstruct_block_sparse_rows,
};
use crate::tiered::Tier0Mean;
use crate::tiered::code_space::{
    CodeSpacePromotionReport, harvest_code_space_promotions, linear_distortion_floor,
};

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
    /// `FARTHEST_POINT_SEED_MAX_OPS`.
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
            max_inner_iter: SAE_SUPPORT_INNER_FIXED_POINT_MAX_ITER,
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

/// The seed half of the schedule (Tier-0 + Tier-1) as a standalone stage: the
/// **linear peel**. Increment 5 gives the public support-sparse entry the same
/// warm start the in-crate tiered driver has, so both reach the curved engine
/// through this one declaration rather than two spellings of it.
///
/// It carries no width knob of its own — [`LinearPeelConfig::derive`] reads the
/// geometry off the data and the curved request.
#[derive(Clone, Debug)]
pub struct LinearPeelConfig {
    /// Tier-1 block geometry (`G` blocks of size `b`, block budget `k`).
    pub tier1: BlockSparseConfig,
    /// How Tier-1 seeds its `K = G·b` block frames.
    pub tier1_seed: TieredSeedPolicy,
}

impl LinearPeelConfig {
    /// Derive the Tier-1 block geometry from the corpus width `P`, the curved
    /// dictionary's effective per-atom dimension `d_max`, and the caller's per-row
    /// support budget `support_k`. Every field is read off a quantity the caller
    /// already declared:
    ///
    /// * block size `b = d_max` — a linear atom is the curvature-free special case
    ///   of a curved atom of the same intrinsic dimension (see [`crate::tiered`]),
    ///   so a `d`-dimensional chart's linear counterpart is a `d`-dimensional block;
    /// * blocks `G = P / b` — the linear bulk of an `N×P` corpus is spanned by at
    ///   most `P` directions, so `K_lin = G·b ≤ P` is the widest linear dictionary
    ///   the data identifies. Past it the frame fixed point enters the #2275
    ///   over-complete regime where spurious frames rotate freely; charting what
    ///   the linear span cannot reach is the curved tier's job, not the peel's;
    /// * block budget `k = min(support_k, G)` — the declared per-row sparsity,
    ///   spent on blocks instead of curved atoms.
    ///
    /// Every remaining knob stays at [`BlockSparseConfig::default`]; the peel
    /// introduces no constant of its own.
    pub fn derive(output_dim: usize, d_max: usize, support_k: usize) -> Result<Self, String> {
        if output_dim == 0 || d_max == 0 || support_k == 0 {
            return Err(format!(
                "LinearPeelConfig::derive requires P >= 1, d_max >= 1 and support_k >= 1; got \
                 P={output_dim}, d_max={d_max}, support_k={support_k}"
            ));
        }
        let n_blocks = output_dim / d_max;
        if n_blocks == 0 {
            return Err(format!(
                "LinearPeelConfig::derive: a curved atom of dimension d_max={d_max} has no linear \
                 counterpart in P={output_dim} — the linear bulk cannot carry a block wider than \
                 the corpus. Reduce d_atom or disable the linear peel"
            ));
        }
        let mut tier1 = BlockSparseConfig::new(n_blocks, d_max);
        tier1.block_topk = support_k.min(n_blocks);
        // The peel inherits the tiered lane's own operating practice: revival
        // draws from worst-residual rows at the routing width (its tests run
        // aux_k = block_topk), and the epoch cap must not bind before the
        // captured-fraction plateau rule can terminate — the plateau is the
        // stopping criterion; the cap is a correctness bound only. Measured
        // on the real 250k-row Qwen chart: without these, Tier-1 refuses at
        // 30 epochs with frame residual 1.0; with them it converges and the
        // peel hands the curved engine a residual 30% lighter.
        tier1.aux_k = tier1.block_topk;
        tier1.max_epochs = 10 * tier1.max_epochs.max(1);
        Ok(Self {
            tier1,
            tier1_seed: TieredSeedPolicy::Auto,
        })
    }
}

/// The frozen linear peel: everything needed to reproduce `μ + L` on rows the fit
/// never saw, and nothing that depends on the training corpus. This is what a
/// fitted public model stores and what its serialization round-trips.
#[derive(Clone, Debug)]
pub struct LinearPeelState {
    /// Tier-0 shared mean `μ`, length `P`.
    pub mean: Array1<f64>,
    /// Frozen Tier-1 block frames, `K×P` (`K = G·b`).
    pub decoder: Array2<f64>,
    /// Tier-1's tied encoder scalar `γ`.
    pub gamma: f64,
    /// Block size `b`.
    pub block_size: usize,
    /// Block routing budget `k`.
    pub block_topk: usize,
    /// Block-tile width for the out-of-sample route.
    pub block_tile: usize,
    /// Local mean of the Tier-1 residual, peeled before charting.
    pub residual_mean: Array1<f64>,
}

/// The linear bulk `L` on rows `z` (`N×P`): de-mean by `μ`, route the rows against
/// the frozen block frames, and decode.
///
/// The peel defines `L` through THIS map both at fit time and at prediction time,
/// rather than through the trainer's stored codes: those were last encoded before
/// the final `γ` refresh, so a curved tier fit on a residual built from them would
/// be charting a residual the deployed model can never reproduce.
fn route_linear_bulk(
    mean: &Array1<f64>,
    decoder: ArrayView2<'_, f64>,
    gamma: f64,
    block_size: usize,
    block_topk: usize,
    block_tile: usize,
    z: ArrayView2<'_, f64>,
) -> Result<Array2<f64>, String> {
    if z.ncols() != mean.len() {
        return Err(format!(
            "route_linear_bulk: z has P={} but the peel spans P={}",
            z.ncols(),
            mean.len()
        ));
    }
    let centered = &z - &mean.view().insert_axis(Axis(0));
    let (blocks, _gates, codes) = block_sparse_dictionary_transform(
        centered.view(),
        decoder,
        gamma,
        block_size,
        block_topk,
        block_tile,
    )?;
    let linear = reconstruct_block_sparse_rows(decoder, blocks.view(), codes.view(), block_size)?;
    Ok(linear)
}

impl LinearPeelState {
    /// The constant the composed model adds back: `μ + mean(R1)`. Both are
    /// row-independent, so the curved tier only ever sees their sum.
    pub fn composed_mean(&self) -> Array1<f64> {
        &self.mean + &self.residual_mean
    }

    /// The linear bulk `L` on rows `z` — the same map the fit itself used, so a
    /// held-out row and a training row are charted identically.
    pub fn linear_reconstruct(&self, z: ArrayView2<'_, f64>) -> Result<Array2<f64>, String> {
        route_linear_bulk(
            &self.mean,
            self.decoder.view(),
            self.gamma,
            self.block_size,
            self.block_topk,
            self.block_tile,
            z,
        )
    }

    /// The whole additive offset the curved tier works around on rows `z`:
    /// `μ + L(z) + mean(R1)`. The composed model is `offset(z) + C` and the target
    /// the curved tier charts is `z − offset(z)`, so a caller needs exactly this one
    /// quantity for both directions — and computes the route only once.
    pub fn offset(&self, z: ArrayView2<'_, f64>) -> Result<Array2<f64>, String> {
        let linear = self.linear_reconstruct(z)?;
        Ok(&linear + &self.composed_mean().view().insert_axis(Axis(0)))
    }
}

/// A fitted linear peel: the Tier-0 mean, the Tier-1 linear bulk, and the
/// centered residual the curved tier charts.
#[derive(Clone, Debug)]
pub struct LinearPeel {
    /// Tier-0 shared mean `μ`.
    pub tier0: Tier0Mean,
    /// Tier-1 block-sparse linear bulk.
    pub tier1: BlockSparseFit,
    /// Tier-1 reconstruction `L` on the training rows (`N×P`, f64).
    pub linear: Array2<f64>,
    /// Local mean of `R1 = R0 − L`, peeled before charting and added back on
    /// reconstruction so the curved correction lives in residual space.
    pub residual_mean: Array1<f64>,
    /// `R1 − mean(R1)` — the target the curved tier charts.
    pub residual: Array2<f64>,
    /// `‖R0‖²`: the Tier-0 baseline energy a composed EV is measured against.
    pub baseline_energy: f64,
    /// Block-tile width the fit routed with; carried so a prediction routes the
    /// same way rather than picking its own tiling.
    pub block_tile: usize,
}

impl LinearPeel {
    /// The frozen, corpus-independent half of this peel.
    pub fn state(&self) -> LinearPeelState {
        LinearPeelState {
            mean: self.tier0.mean.clone(),
            decoder: self.tier1.decoder.clone(),
            gamma: self.tier1.gamma,
            block_size: self.tier1.block_size,
            block_topk: self.tier1.block_topk,
            block_tile: self.block_tile,
            residual_mean: self.residual_mean.clone(),
        }
    }
}

/// Run the seed policy — Tier-0 mean peel, then the Tier-1 block-sparse linear
/// warm start — and hand back the centered residual `R1 = R0 − L − mean(R1)` the
/// curved engine charts.
///
/// This is the single declaration of the peel: [`fit_tiered`] and the public
/// support-sparse entry both reach the curved engine through it.
pub fn fit_linear_peel(
    z: ArrayView2<'_, f64>,
    config: &LinearPeelConfig,
) -> Result<LinearPeel, String> {
    // Tier 0: peel the shared mean; the bulk is fit on R0 = z − μ.
    let tier0 = Tier0Mean::fit(z)?;
    let r0 = tier0.apply(z)?;

    // Tier 1: block-sparse collapsed-linear bulk on the de-meaned residual. The
    // seed policy resolves against the corpus size + block geometry so a K≈1e4 bulk
    // skips the serial O(N·P·K) farthest-point pass (the large-K entry, #2023).
    let seed_policy = config
        .tier1_seed
        .resolve(r0.nrows(), r0.ncols(), &config.tier1);
    let tier1 = fit_block_sparse_dictionary_with_seed(r0.view(), &config.tier1, seed_policy)?;

    let (n_obs, output_dim) = r0.dim();
    let linear = route_linear_bulk(
        &tier0.mean,
        tier1.decoder.view(),
        tier1.gamma,
        tier1.block_size,
        tier1.block_topk,
        config.tier1.block_tile,
        z,
    )?;
    if linear.dim() != (n_obs, output_dim) {
        return Err(format!(
            "fit_linear_peel: Tier-1 reconstruction {:?} does not match residual ({n_obs}, {output_dim})",
            linear.dim()
        ));
    }
    let tier1_residual = &r0 - &linear;

    // Peel the residual's local mean before charting (added back on reconstruct).
    let residual_mean = tier1_residual
        .mean_axis(Axis(0))
        .ok_or_else(|| "fit_linear_peel: residual mean_axis returned None".to_string())?;
    let residual = &tier1_residual - &residual_mean.view().insert_axis(Axis(0));
    let baseline_energy = r0.iter().map(|value| value * value).sum::<f64>();

    Ok(LinearPeel {
        tier0,
        tier1,
        linear,
        residual_mean,
        residual,
        baseline_energy,
        block_tile: config.tier1.block_tile,
    })
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
    /// The code-space curvature census: every Tier-1 block adjudicated as a
    /// linear community for curved replacement, in bits, on its CODE cloud —
    /// the substrate where in-span curvature is visible after the linear tier
    /// reconstructs it exactly (the residual substrate is blind to that move by
    /// construction; see `crate::tiered::code_space`).
    pub code_space: CodeSpacePromotionReport,
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
/// schedule for the risk-pin tests + `tiered_gpu_scale` example. Increment 5 moved
/// its seed half into `fit_linear_peel`, which the public support-sparse entry
/// now runs too, so this function is a cadence over the same two phases and not a
/// second implementation of them.
///
/// The curved tier is fit on the Tier-1 residual through the canonical
/// support-sparse engine (`fit_tier2_support` → [`run_sae_support_outer`]),
/// whose returned fit carries a certified inner fixed point and outer stationarity
/// certificate. No principal-component reseeding occurs; the [`SaeMigrationLedger`]
/// accounts for the curved births / deaths and pins `pc_reseed_events = 0`.
pub fn fit_tiered(
    z: ArrayView2<'_, f64>,
    config: &TieredFitConfig,
) -> Result<TieredFitReport, String> {
    // Tier 0 + Tier 1: the shared linear peel.
    let peel = fit_linear_peel(
        z,
        &LinearPeelConfig {
            tier1: config.tier1,
            tier1_seed: config.tier1_seed,
        },
    )?;

    let mut ledger = SaeMigrationLedger::new();

    // Structural deaths: Tier-1 blocks no row selected fall back to the residual
    // factor pool. (Revival, when it happens, draws from worst-residual rows in
    // the block lane — never from PCs.)
    let n_dead = peel
        .tier1
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

    // Code-space curvature census: adjudicate every Tier-1 block's CODE cloud
    // for curved replacement, in bits, BEFORE the residual tier runs. In-span
    // curvature (a ring two linear atoms reconstruct exactly) leaves the
    // residual identically zero, so this pass is the only tier substrate with
    // power against it; the distortion floor and L0 are measured off the fit.
    let tolerance = linear_distortion_floor(peel.residual.view(), peel.baseline_energy)?;
    let code_space = harvest_code_space_promotions(&peel.tier1, z.nrows(), tolerance)?;
    let census_proposals = code_space.proposals.iter().chain(
        code_space
            .pair_proposals
            .iter()
            .map(|verdict| &verdict.proposal),
    );
    for proposal in census_proposals {
        let evidence = MoveEvidence::from_dl_bits(proposal.dl_old - proposal.dl_new);
        if proposal.accept {
            ledger.birth(
                MoveStage::Curved,
                BirthSeed::LinearAtom,
                1,
                None,
                evidence,
                proposal.dl_new,
            );
        } else {
            ledger.refuse(
                MoveStage::Curved,
                MoveReason::EvidenceInsufficient,
                1,
                None,
                evidence,
                proposal.dl_new,
            );
        }
    }

    // Tier 2: curved support-sparse refinement on the Tier-1 residual, or the
    // linear-bulk baseline.
    let (tier2, explained_variance) = if config.tier2_enabled {
        let fit = fit_tier2_support(&peel, &config.tier2)?;
        record_support_moves(&mut ledger, &fit);
        let ev = fit.explained_variance;
        (Some(fit), ev)
    } else {
        (None, peel.tier1.explained_variance)
    };

    Ok(TieredFitReport {
        tier0: peel.tier0,
        tier1: peel.tier1,
        tier2,
        code_space,
        ledger,
        explained_variance,
    })
}

/// Fit the Tier-2 curved refinement: the overcomplete hard-TopK support-sparse
/// dictionary on the [`LinearPeel`]'s centered residual, driven end to end through
/// the canonical support-sparse engine (seed → term → grouped-LAML outer solve),
/// exactly as the public support-sparse fit entry drives it.
///
/// The peel already removed the residual's local mean, which is added back on
/// reconstruction, so the curved correction `C` lives in residual space and the
/// composed model is `μ + L + C`. The returned explained variance measures that
/// composed reconstruction against the Tier-0 mean baseline (`TSS = ‖R0‖²`, since
/// `R0` is exactly de-meaned).
fn fit_tier2_support(
    peel: &LinearPeel,
    config: &Tier2SupportConfig,
) -> Result<Tier2SupportFit, String> {
    let (n_obs, output_dim) = peel.residual.dim();
    let mean = peel.residual_mean.clone();
    let centered = peel.residual.clone();

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
    let tss = peel.baseline_energy;
    let explained_variance = crate::tiered::explained_variance_from_sums(rss, tss);

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
mod peel_tests {
    use super::*;
    use ndarray::Array2;

    /// The planted fixture the Increment-5 peel is about: a straight linear bulk
    /// (cols 0,1) the linear tier explains exactly, a circle (cols 2,3) whose
    /// curvature is exactly what survives a linear fit, and a non-zero column mean
    /// so Tier-0 has something to peel. Geometry mirrors the converging tiered test
    /// `tier2_branch_constructs_the_support_sparse_path` (n=96, P=4, K=8).
    fn planted_bulk_plus_curvature(n: usize) -> Array2<f64> {
        let mut z = Array2::<f64>::zeros((n, 4));
        for i in 0..n {
            let t = i as f64 / n as f64;
            let phase = (i as f64) * 0.19;
            z[[i, 0]] = 1.5 + (2.0 * t - 1.0);
            z[[i, 1]] = -0.5 + (1.0 - 2.0 * t);
            z[[i, 2]] = 0.25 + phase.cos();
            z[[i, 3]] = -0.75 + phase.sin();
        }
        z
    }

    /// The curved half of the public support-sparse entry, verbatim in its stage
    /// order (seed → term seed → grouped-LAML outer solve). `gam-sae` cannot call
    /// the pyffi entry, so this stands in for it: what these tests vary is only the
    /// TARGET the entry hands the engine — peeled residual vs mean-centered data.
    fn chart_curved(
        centered: ArrayView2<'_, f64>,
        n_atoms: usize,
        support_k: usize,
    ) -> Result<
        (
            SaeSupportSparseTerm,
            SaeSupportFixedPointReport,
            OuterCriterionCertificate,
        ),
        String,
    > {
        let (n_obs, output_dim) = centered.dim();
        let atom_basis = vec!["periodic".to_string(); n_atoms];
        let atom_dim = vec![1usize; n_atoms];
        let effective_dims = sae_support_effective_atom_dims(&atom_basis, &atom_dim)?;
        let d_max = effective_dims.iter().copied().max().unwrap_or(1);
        let admission = admit_topk_manifold(n_obs, output_dim, n_atoms, d_max, support_k)?;
        let seed = build_sae_support_seed(SaeSupportSeedRequest {
            target: centered,
            atom_basis: &atom_basis,
            atom_dim: &atom_dim,
            support_k,
            random_state: 0xC0FF_EE00_D15E_A5E5,
            admission,
        })?;
        let retained = seed.retained_atom_indices.clone();
        let term_seed = build_sae_support_term_seed(SaeSupportTermSeedRequest {
            assignment: seed.assignment,
            atom_basis: vec!["periodic".to_string(); retained.len()],
            atom_dim: vec![1usize; retained.len()],
            output_dim,
            random_state: 0xC0FF_EE00_D15E_A5E5,
        })?;
        let ard_precisions = (0..term_seed.term.k_atoms())
            .map(|atom| vec![1.0; term_seed.term.assignment.atom_coord_dim(atom)])
            .collect::<Vec<_>>();
        let outer = run_sae_support_outer(SaeSupportOuterRequest {
            term: term_seed.term,
            target: centered.to_owned(),
            initial_smoothness: 1.0,
            ard_precisions,
            max_outer_iter: 32,
            max_inner_iter: 256,
            // The public entry's relative inner tolerance (#2517).
            inner_tolerance: 1.0e-4,
            trust_radius: 1.0,
            random_state: 0xC0FF_EE00_D15E_A5E5,
        })
        .map_err(|error| error.to_string())?;
        Ok((outer.term, outer.fixed_point, outer.outer_certificate))
    }

    /// The width derivation reads the block geometry off the corpus and the curved
    /// request, and refuses a block wider than the corpus rather than inventing one.
    #[test]
    fn derived_peel_width_mirrors_the_curved_request() {
        let config = LinearPeelConfig::derive(16, 2, 3).expect("derives");
        assert_eq!(config.tier1.block_size, 2, "b = d_max");
        assert_eq!(config.tier1.n_blocks, 8, "G = P / b");
        assert_eq!(
            config.tier1.n_atoms(),
            16,
            "K_lin = P, the identifiable width"
        );
        assert_eq!(config.tier1.block_topk, 3, "k = support_k");
        // support_k above the block count cannot fire more blocks than exist.
        let narrow = LinearPeelConfig::derive(4, 1, 9).expect("derives");
        assert_eq!(narrow.tier1.block_topk, 4);
        // A block wider than the corpus has no linear counterpart.
        let refused = LinearPeelConfig::derive(3, 4, 1);
        assert!(
            refused.is_err(),
            "d_max > P must be refused, not rounded down to a block the caller never asked for"
        );
    }

    /// #2232 Increment 5, arm (a): on a planted linear-bulk + curved-residual
    /// fixture the peeled path drives the support engine to a certified outer
    /// stationarity point with a RECURRED inner fixed point — the convergence the
    /// public entry needs the peel to reach.
    #[test]
    fn peeled_support_fit_converges_on_planted_bulk_plus_curvature() {
        let z = planted_bulk_plus_curvature(96);
        let config = LinearPeelConfig::derive(z.ncols(), 1, 2).expect("peel geometry derives");
        let peel = fit_linear_peel(z.view(), &config).expect("the linear peel converges");
        let (term, fixed_point, certificate) =
            chart_curved(peel.residual.view(), 8, 2).expect("the peeled curved fit runs");

        assert!(
            certificate.certifies() && certificate.is_stationary(),
            "the peeled fit must carry a certifying outer stationarity certificate"
        );
        assert!(
            fixed_point.recurred,
            "the peeled inner fixed point must have RECURRED; got {fixed_point:?}"
        );
        assert!(
            term.k_atoms() >= 1,
            "the peeled fit must retain a curved atom"
        );
    }

    /// Arm (b): the composition is exact. The peel's additive offset `μ + L +
    /// mean(R1)` plus the curved correction `C` reproduces `μ + L + mean(R1) + C`
    /// term for term, and the offset a prediction recomputes on the training rows is
    /// the offset the fit itself used — so `reconstruct` is the same model the fit
    /// reported, not a second one.
    #[test]
    fn composed_reconstruction_is_mu_plus_linear_plus_curved() {
        let z = planted_bulk_plus_curvature(96);
        let config = LinearPeelConfig::derive(z.ncols(), 1, 2).expect("peel geometry derives");
        let peel = fit_linear_peel(z.view(), &config).expect("the linear peel converges");

        // The peel is an exact additive decomposition of the data.
        for row in 0..z.nrows() {
            for column in 0..z.ncols() {
                let parts = peel.tier0.mean[column]
                    + peel.linear[[row, column]]
                    + peel.residual_mean[column]
                    + peel.residual[[row, column]];
                assert!(
                    (parts - z[[row, column]]).abs() <= 1.0e-12 * z[[row, column]].abs().max(1.0),
                    "μ + L + mean(R1) + R1c must reproduce z at ({row}, {column}): {parts} vs {}",
                    z[[row, column]]
                );
            }
        }

        let state = peel.state();
        let offset = state.offset(z.view()).expect("offset recomputes");
        // Re-routing the training rows against the frozen frames returns the SAME
        // linear bulk the fit used — the peel defines L through the frozen map at
        // fit time too, so this is an identity, not an approximation.
        for row in 0..z.nrows() {
            for column in 0..z.ncols() {
                let fitted_offset = peel.tier0.mean[column]
                    + peel.linear[[row, column]]
                    + peel.residual_mean[column];
                assert!(
                    (offset[[row, column]] - fitted_offset).abs() <= 1.0e-12,
                    "recomputed offset {} != fitted offset {fitted_offset} at ({row}, {column})",
                    offset[[row, column]]
                );
            }
        }

        let (term, _fixed_point, _certificate) =
            chart_curved(peel.residual.view(), 8, 2).expect("the peeled curved fit runs");
        let curved = term.reconstruct().expect("curved reconstruction");
        let composed = &offset + &curved;
        for row in 0..z.nrows() {
            for column in 0..z.ncols() {
                let expected = peel.tier0.mean[column]
                    + peel.linear[[row, column]]
                    + peel.residual_mean[column]
                    + curved[[row, column]];
                assert!(
                    (composed[[row, column]] - expected).abs() <= 1.0e-12,
                    "composed reconstruction {} != μ + L + mean(R1) + C = {expected} at ({row}, {column})",
                    composed[[row, column]]
                );
            }
        }
    }

    /// Arm (c): with the peel disabled the engine still sees exactly the
    /// mean-centered target the public entry has always handed it — bit-identical,
    /// so `linear_peel=False` is the pre-Increment-5 fit and not an approximation
    /// of it. The peeled target is a different one; the peel is the discriminator.
    #[test]
    fn peel_disabled_target_is_the_mean_centered_target() {
        let z = planted_bulk_plus_curvature(96);
        let training_mean = z.mean_axis(Axis(0)).expect("column mean");
        let unpeeled = &z - &training_mean.view().insert_axis(Axis(0));

        let config = LinearPeelConfig::derive(z.ncols(), 1, 2).expect("peel geometry derives");
        let peel = fit_linear_peel(z.view(), &config).expect("the linear peel converges");

        // Tier-0 IS the column mean, so the peel's first stage is bit-identical to
        // the unpeeled centering; everything after it is the linear bulk.
        for column in 0..z.ncols() {
            assert_eq!(
                peel.tier0.mean[column], training_mean[column],
                "Tier-0 must be exactly the column mean the unpeeled entry removes"
            );
        }
        let linear_energy: f64 = peel.linear.iter().map(|value| value * value).sum();
        assert!(
            linear_energy > 0.0,
            "the planted straight bulk must give the peel something to remove"
        );
        let mut max_delta = 0.0f64;
        for row in 0..z.nrows() {
            for column in 0..z.ncols() {
                max_delta =
                    max_delta.max((peel.residual[[row, column]] - unpeeled[[row, column]]).abs());
            }
        }
        assert!(
            max_delta > 1.0e-6,
            "the peeled target must DIFFER from the mean-centered one; got max delta {max_delta}"
        );

        // Disabling the peel is a SUPPORTED configuration, not a broken one —
        // but the unpeeled engine stalling on bulk-carrying data is the very
        // defect this increment addresses (#2517), so the arm may legally
        // refuse. What it must never do is fail any other way.
        match chart_curved(unpeeled.view(), 8, 2) {
            Ok((_term, _fixed_point, certificate)) => assert!(
                certificate.certifies(),
                "an unpeeled fit that returns must return certified"
            ),
            Err(error) => assert!(
                error.to_string().contains("did not recur"),
                "an unpeeled refusal must be the engine's own stall, got: {error}"
            ),
        }
    }
}

#[cfg(test)]
mod fit_tests {
    use super::*;
    use ndarray::Array2;

    fn assert_block_certificate(c: &crate::sparse_dict::BlockSparseConvergence, tolerance: f64) {
        assert_eq!(c.tolerance, tolerance);
        assert_eq!(c.accepted_births, 0);
        assert_eq!(c.polar_failures, 0);
        for residual in [
            c.ev_residual,
            c.gamma_residual,
            c.frame_residual,
            c.routing_residual,
            c.reconstruction_residual,
        ] {
            assert!(
                residual.is_finite() && residual <= tolerance,
                "open certificate: {c:?}"
            );
        }
    }

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
        assert_block_certificate(&report.tier1.convergence, config.tier1.tolerance);
    }

    /// Overcomplete Tier-1 must converge before Tier-2 consumes its residual.
    #[test]
    fn tiered_returns_closed_certificate_at_k_gg_rank_2275() {
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
        config.tier1.aux_k = 4; // Births must settle before certification.
        config.tier1.max_epochs = 40;
        // Tier-2 is a witness that converged Tier-1 reaches the
        // curved lane. Use the smallest overcomplete dictionary instead of the
        // production default; K=P+4 preserves K>P without importing unrelated
        // support conditioning into this convergence test.
        config.tier2.n_atoms = p + 4;
        config.tier2.support_k = 1;

        let report = fit_tiered(z.view(), &config)
            .expect("overcomplete Tier-1 must converge and reach Tier-2");
        assert_block_certificate(&report.tier1.convergence, config.tier1.tolerance);
        assert!(report.tier1.explained_variance.is_finite());
        assert!(
            report.tier2.is_some(),
            "Tier-2 must run on a converged Tier-1 residual"
        );
        assert!(report.explained_variance.is_finite());
    }

    /// The public overcomplete block entry must return a closed certificate.
    #[test]
    fn block_sparse_overcomplete_fit_returns_closed_certificate_2275() {
        use crate::sparse_dict::{
            BlockSeedPolicy, BlockSparseConfig, fit_block_sparse_dictionary_with_seed,
        };
        let n = 96usize;
        let p = 8usize;
        let mut x = Array2::<f64>::zeros((n, p));
        for i in 0..n {
            let t = (i as f64) * 0.2;
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
        .expect("the overcomplete block entry must converge");
        assert_block_certificate(&fit.convergence, config.tolerance);
    }

    /// Small deterministic two-circle corpus shared by the Tier-2 path tests.
    /// P=4 is the smallest output that contains two independent curved planes;
    /// keeping N=96 supplies several rotations at both frequencies without making
    /// path/provenance tests pay for a large support-operator benchmark.
    fn two_circle_fixture_2634() -> Array2<f64> {
        let n = 96usize;
        let mut z = Array2::<f64>::zeros((n, 4));
        for i in 0..n {
            let phase = i as f64 * 0.19;
            z[[i, 0]] = phase.cos();
            z[[i, 1]] = phase.sin();
            z[[i, 2]] = (1.7 * phase).cos();
            z[[i, 3]] = (1.7 * phase).sin();
        }
        z
    }

    /// Planted two-circle acceptance witness (#2023): the tiered fit (Tier-1
    /// linear bulk + Tier-2 curved support-sparse refinement on the residual)
    /// must not regress the pure-linear Tier-1 EV, its Tier-2 must return a
    /// certified support-sparse fit, and the migration ledger must record the
    /// retained curved atoms as promotions off the linear residual.
    #[test]
    fn tiered_curved_refinement_is_certified_and_records_promotions() {
        let z = two_circle_fixture_2634();
        let p = z.ncols();

        // Two rank-1/top-1 blocks are the smallest identified linear bulk that
        // leaves curvature for Tier-2 instead of selecting all P directions.
        let mut lin = TieredFitConfig::linear_bulk(2, 1);
        lin.tier1.block_topk = 1;
        lin.tier1.aux_k = 2;
        lin.tier1.max_epochs = 200;
        let lin_report = fit_tiered(z.view(), &lin).expect("linear-bulk fit runs");
        let ev_lin = lin_report.explained_variance;

        // Tiered (Tier-1 + Tier-2 curved support-sparse refinement): same Tier-1.
        // K=P+1 is the smallest overcomplete support-sparse dictionary.
        let mut tiered = TieredFitConfig::tiered(2, 1);
        tiered.tier1.block_topk = 1;
        tiered.tier1.aux_k = 2;
        tiered.tier1.max_epochs = 200;
        tiered.tier2.n_atoms = p + 1;
        tiered.tier2.support_k = 1;
        tiered.tier2.max_outer_iter = 32;
        tiered.tier2.max_inner_iter = 256;
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

    /// The census the residual substrate cannot run: a planted circle that ONE
    /// Tier-1 block (`b=2`) reconstructs EXACTLY leaves a zero block residual,
    /// so no residual-mining tier can ever see it — yet the code-space census
    /// discovers the ring in the block's code cloud. At this fixture's width the
    /// dictionary is NOT overcomplete (`G = L0`), so the support dividend that
    /// funds a circle's promotion is zero and the prescreen must DEFER: the
    /// honest end-to-end behavior is recognition + a recorded refusal, never a
    /// birth bought with no compression to pay for it (the acceptance arm lives
    /// in the overcomplete hand-minted census tests). This is the #2502 in-span
    /// curvature move wired end to end from `fit_tiered`.
    #[test]
    fn code_space_census_recognizes_and_defers_a_zero_residual_planted_ring() {
        use std::f64::consts::TAU;
        // A pure circle in cols 0,1 of P=4; evenly spaced phases for full
        // ring coverage. Cols 2,3 carry nothing, so one b=2 block spans the
        // corpus exactly and the post-Tier-1 residual is numerically zero.
        let n = 96usize;
        let mut z = Array2::<f64>::zeros((n, 4));
        for i in 0..n {
            let theta = TAU * (i as f64) / (n as f64);
            z[[i, 0]] = theta.cos();
            z[[i, 1]] = theta.sin();
        }
        let mut config = TieredFitConfig::linear_bulk(1, 2);
        config.tier1.block_topk = 1;
        config.tier1.max_epochs = 200;
        let report = fit_tiered(z.view(), &config).expect("single-block tiered fit runs");

        let census = &report.code_space;
        assert_eq!(census.n_blocks_scanned, 1);
        assert_eq!(
            census.n_communities, 1,
            "the fired 2-atom block must reach the adjudicator"
        );
        let proposal = &census.proposals[0];
        // Recognition: the ring geometry is seen (span ≈ 2, ring screens pass)
        // and the ATOMIC ledger genuinely prefers the curved chart …
        assert!(
            proposal.verdict.recommend_curl,
            "the census must recognize the planted ring geometrically: {proposal:?}"
        );
        assert!(
            proposal.dl_new < proposal.dl_old,
            "the atomic ledger must prefer the circle (dl_new={}, dl_old={})",
            proposal.dl_new,
            proposal.dl_old
        );
        // … but at G = L0 the support dividend is zero, so the conservative
        // prescreen defers rather than buys the wider harmonic decoder.
        assert!(
            proposal.crossover_prescreen_bits <= 0.0,
            "with no overcompleteness the prescreen cannot pay: {}",
            proposal.crossover_prescreen_bits
        );
        assert_eq!(census.n_accepted, 0, "deferred, not bought");
        // Ledger provenance: the deferral is a recorded Curved REFUSAL; no birth.
        assert_eq!(report.ledger.n_births, 0);
        assert!(
            report.ledger.n_refusals >= 1,
            "the ledger must record the deferred promotion"
        );
        assert_eq!(report.ledger.pc_reseed_events, 0);
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
        let z = two_circle_fixture_2634();
        let p = z.ncols();

        let mut config = TieredFitConfig::tiered(2, 1);
        config.tier1_seed = TieredSeedPolicy::CoordinatePartition;
        config.tier1.block_topk = 1;
        config.tier1.aux_k = 2;
        config.tier1.max_epochs = 200;
        // Smallest overcomplete curved dictionary for this seed-path witness.
        config.tier2.n_atoms = p + 1;
        config.tier2.support_k = 1;
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
        let z = two_circle_fixture_2634();
        let p = z.ncols();

        // Two b=2 blocks with topk=2 select all K=P directions on every row:
        // the reconstruction is the identity and the partition is
        // nonidentified, so there is neither compression nor a curved residual
        // for Tier-2 to refine. Two rank-1 blocks with topk=1 are the minimal
        // identified two-circle witness.
        let mut config = TieredFitConfig::tiered(2, 1);
        config.tier1.block_topk = 1;
        config.tier1.aux_k = 2;
        config.tier1.max_epochs = 200;
        // The smallest overcomplete curved dictionary: K=P+1 is enough to pin
        // the support-sparse lane, which is the only representation the front
        // door admits for K>P, without making this routing/provenance witness a
        // quadrature-scale benchmark.
        config.tier2.atom_basis = "periodic".to_string();
        config.tier2.atom_dim = 1;
        config.tier2.n_atoms = p + 1;
        config.tier2.support_k = 1;
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
                .decoder_coefficients()
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
