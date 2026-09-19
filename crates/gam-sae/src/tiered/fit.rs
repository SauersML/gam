//! Tiered fit as a **seed policy + alternation cadence** of the unified engine
//! (#2023; unification #2232 Increment 4).
//!
//! "Tiered" is not a separate model or a separate public API — it is a *schedule*
//! that the one SAE engine runs: the residual tier is a **round-0 warm start**,
//! and the alternation to joint stationarity is the fit. Increment 4 deleted the
//! public tiered surface (the `sae_manifold_fit_tiered` FFI entry and its
//! `gamfit._sae_spectral` Python wrapper); the tiered flow is now reached only
//! through the unified engine + the seed/cadence composition (`experiments/
//! compose_tiers.py`: linear warm start via `sparse_dictionary_fit` → curved
//! alternation via `sae_manifold_fit`).
//!
//! This module keeps the in-crate orchestrator [`fit_tiered`] that expresses that
//! schedule directly in Rust. Production reaches its linear-bulk half through
//! [`linear_bulk_census`], which the public support-sparse fit runs as its
//! code-space census (#2023 lead ruling). The full cadence is reached only by its
//! tests and the `tiered_dominance`, `tiered_gpu_scale` and `tiered_k2000_measure`
//! examples. The seed half is the [`fit_linear_peel`] stage and the full cadence is:
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
//!    ([`fit_sae_support_sparse`], the one path the public support-sparse fit also
//!    runs: seed → term seed → grouped-LAML outer solve). The residual's local mean
//!    is peeled before the
//!    fit and added back on reconstruction, so the curved correction `C` lives in
//!    residual space and the composed model is `μ + L + C`. This is the `K > P`
//!    representation — the front door refuses any resident `N×K` alternative — so
//!    the Tier-2 dictionary width must exceed the residual dimension.
//!
//! The unified [`SaeMigrationLedger`] records three accounts.
//! - Tier-1: the `G` blocks born at the data-row seed, each committed revival as
//!   one death and one birth of its block, and the blocks still dead at the end,
//!   so linear births minus linear deaths is the number of live blocks.
//! - The code-space census's verdicts: accepted promotions as admitted moves (the
//!   census installs nothing, so they add no atom) and refusals.
//! - Tier-2's own account folded from its support fit: every curved atom born at
//!   the support seed from projections of the Tier-1 residual rows (the
//!   residual-factor pool), and the atoms that seed pruned for zero support mass.
//!   The support-sparse lane prices complexity through its grouped-LAML smoothing,
//!   not a per-move description-length charge, so those curved moves carry no
//!   `dl_bits`.
//!
//! `pc_reseed_events` is always `0` on this path.

use ndarray::{Array1, Array2, ArrayView2, Axis};

use gam_solve::rho_optimizer::OuterCriterionCertificate;

use crate::manifold::{
    SaeSupportFixedPointReport, SaeSupportSparseFitRequest, SaeSupportSparseTerm,
    fit_sae_support_sparse,
};
use crate::migration_ledger::{BirthSeed, MoveEvidence, MoveReason, MoveStage, SaeMigrationLedger};
use crate::sparse_dict::{
    BlockSparseConfig, BlockSparseFit, block_sparse_dictionary_transform,
    fit_block_sparse_dictionary, reconstruct_block_sparse_rows,
};
use crate::tiered::Tier0Mean;
use crate::tiered::code_space::{
    CodeSpacePromotionReport, harvest_code_space_promotions, linear_distortion_floor,
};

/// Tier-2 curved refinement configuration: the overcomplete hard-TopK
/// support-sparse dictionary fit on the Tier-1 residual (#2023). The residual
/// is charted by `n_atoms` curved atoms of the declared `atom_basis`/`atom_dim`
/// family under a per-row `support_k` TopK support, and its smoothing strengths
/// are selected by the grouped-LAML outer engine
/// ([`crate::manifold::run_sae_support_outer`]).
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
    /// live chart width is resolved by
    /// [`crate::manifold::sae_support_effective_atom_dims`]).
    pub atom_dim: usize,
    /// Overcomplete curved dictionary width `K`. Must exceed the residual `P`.
    pub n_atoms: usize,
    /// Per-row TopK curved support width `s` (`1 <= s <= n_atoms`).
    pub support_k: usize,
    /// Initial isotropic smoothing strength seeding the outer LAML search.
    pub initial_smoothness: f64,
    /// Outer (smoothing-selection) iteration budget.
    pub max_outer_iter: usize,
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
            trust_radius: 1.0,
            random_state: 0xC0FF_EE00_D15E_A5E5,
        }
    }
}

/// Configuration for [`fit_tiered`]: the seed/cadence schedule. The public tiered
/// surface was removed in unification Increment 4.
#[derive(Clone, Debug)]
pub struct TieredFitConfig {
    /// Tier-1 block-sparse dictionary configuration (`G` blocks of size `b`, the
    /// block budget `k`, epochs, minibatch/tile geometry). GPU score-routing is
    /// governed by the process-wide [`gam_gpu::GpuPolicy`] (`gam_gpu::configure_global_policy`),
    /// not by this config: the Tier-1 router dispatches each minibatch to the CUDA
    /// block-gate lane when the mode admits it and a runtime is present.
    pub tier1: BlockSparseConfig,
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
            tier2_enabled: false,
            tier2: Tier2SupportConfig::default(),
        }
    }

    /// A Tier-0 + Tier-1 + Tier-2 config at `G` blocks of size `b`.
    pub fn tiered(n_blocks: usize, block_size: usize) -> Self {
        Self {
            tier1: BlockSparseConfig::new(n_blocks, block_size),
            tier2_enabled: true,
            tier2: Tier2SupportConfig::default(),
        }
    }
}

/// The seed half of the schedule (Tier-0 + Tier-1) as a standalone stage: the
/// **linear peel** [`fit_tiered`] runs before the curved refinement.
#[derive(Clone, Debug)]
pub struct LinearPeelConfig {
    /// Tier-1 block geometry (`G` blocks of size `b`, block budget `k`).
    pub tier1: BlockSparseConfig,
}

/// The linear bulk `L` on rows `z` (`N×P`): de-mean by `μ`, route the rows against
/// the frozen block frames, and decode.
///
/// The peel defines `L` through THIS map rather than through the trainer's stored
/// codes: those were last encoded before the final `γ` refresh, so a curved tier fit
/// on a residual built from them would be charting a residual the frozen frames
/// never reproduce.
fn route_linear_bulk(
    mean: &Array1<f64>,
    decoder: ArrayView2<'_, f32>,
    gamma: f32,
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
    let centered_f32 = centered.mapv(|value| value as f32);
    let (blocks, _gates, codes) = block_sparse_dictionary_transform(
        centered_f32.view(),
        decoder,
        gamma,
        block_size,
        block_topk,
        block_tile,
    )?;
    let linear = reconstruct_block_sparse_rows(decoder, blocks.view(), codes.view(), block_size)?;
    Ok(linear.mapv(|value| value as f64))
}

/// A fitted linear peel: the Tier-0 mean, the Tier-1 linear bulk, and the
/// centered residual the curved tier charts.
#[derive(Clone, Debug)]
pub struct LinearPeel {
    /// Tier-0 shared mean `μ`.
    pub tier0: Tier0Mean,
    /// Tier-1 block-sparse linear bulk.
    pub tier1: BlockSparseFit,
    /// Local mean of `R1 = R0 − L`, peeled before charting and added back on
    /// reconstruction so the curved correction lives in residual space.
    pub residual_mean: Array1<f64>,
    /// `R1 − mean(R1)` — the target the curved tier charts.
    pub residual: Array2<f64>,
    /// `‖R0‖²`: the Tier-0 baseline energy a composed EV is measured against.
    pub baseline_energy: f64,
}

/// Run the seed policy — Tier-0 mean peel, then the Tier-1 block-sparse linear
/// warm start — and hand back the centered residual `R1 = R0 − L − mean(R1)` the
/// curved engine charts.
pub fn fit_linear_peel(
    z: ArrayView2<'_, f64>,
    config: &LinearPeelConfig,
) -> Result<LinearPeel, String> {
    // Tier 0: peel the shared mean; the bulk is fit on R0 = z − μ.
    let tier0 = Tier0Mean::fit(z)?;
    let r0 = tier0.apply(z)?;
    let r0_f32 = r0.mapv(|value| value as f32);

    // Tier 1: block-sparse collapsed-linear bulk on the de-meaned residual, seeded
    // from its rows at every width (#2023), so every block is a residual-factor birth.
    let tier1 = fit_block_sparse_dictionary(r0_f32.view(), &config.tier1)?;

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
        residual_mean,
        residual,
        baseline_energy,
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
    /// Terminal support quasi-Laplace criterion at the certified smoothing optimum
    /// (#2933 F27: not comparable with a dense quasi-Laplace score).
    pub criterion: crate::front_door::SaeCriterionScore,
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
    /// The support fit's own migration account: its seed births and prunings.
    pub migration: SaeMigrationLedger,
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
/// Production reaches it with Tier-2 off, through [`linear_bulk_census`]. The public
/// tiered FFI/Python surface was deleted in unification Increment 4, so the full
/// cadence is called only by its tests and the `tiered_*` examples.
///
/// The curved tier is fit on the Tier-1 residual through the canonical
/// support-sparse engine (`fit_tier2_support` → [`fit_sae_support_sparse`]),
/// whose returned fit carries a certified inner fixed point and outer stationarity
/// certificate. No principal-component reseeding occurs; the [`SaeMigrationLedger`]
/// accounts for every Tier-1 block and every curved birth and death, and pins
/// `pc_reseed_events = 0`.
pub fn fit_tiered(
    z: ArrayView2<'_, f64>,
    config: &TieredFitConfig,
) -> Result<TieredFitReport, String> {
    // Tier 0 + Tier 1: the shared linear peel.
    let peel = fit_linear_peel(
        z,
        &LinearPeelConfig {
            tier1: config.tier1,
        },
    )?;

    let mut ledger = SaeMigrationLedger::new();

    // Tier-1 births: all `G` block frames are read off rows of `R0`, the residual
    // before any atom exists, so each block is born from the residual-factor pool.
    ledger.birth(
        MoveStage::Linear,
        BirthSeed::ResidualFactor,
        peel.tier1.block_utilization.len(),
        None,
        MoveEvidence::none(),
        f64::NAN,
    );
    // Revivals: each committed residual-row birth installed a new frame in a block
    // no row selected, so it is one death and one birth of that block. The lane
    // admitted each only when rows select the new block and its deviance gain beats
    // its rank charge; that margin is not carried out of the fit, so these moves
    // are unscored here.
    let revivals = peel.tier1.committed_births;
    if revivals > 0 {
        ledger.death(
            MoveStage::Linear,
            MoveReason::DeadRouting,
            revivals,
            None,
            MoveEvidence::none(),
            f64::NAN,
        );
        ledger.birth(
            MoveStage::Linear,
            BirthSeed::ResidualFactor,
            revivals,
            None,
            MoveEvidence::none(),
            f64::NAN,
        );
    }

    // Structural deaths: Tier-1 blocks no row selects at the end fall back to the
    // residual factor pool. With the births above, linear births minus linear
    // deaths is the number of live blocks.
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
    // An accepted proposal is admitted, not born: the census adjudicates the
    // replacement and mutates nothing, so no curved atom joins the model here.
    for proposal in census_proposals {
        let evidence = MoveEvidence::from_dl_bits(proposal.dl_old - proposal.dl_new);
        if proposal.accept {
            ledger.admit(
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

/// The linear bulk the public support-sparse fit audits with (#2023 lead ruling):
/// `G = ⌊P/b⌋` blocks of size `b`, the widest chart the curved dictionary uses, with
/// block TopK `min(s, G)`. Every other knob is the block lane's own default.
fn derived_linear_bulk_config(
    output_dim: usize,
    block_size: usize,
    support_k: usize,
) -> Result<TieredFitConfig, String> {
    if block_size == 0 || support_k == 0 {
        return Err(format!(
            "linear_bulk_census requires block_size >= 1 and support_k >= 1; got \
             b={block_size}, s={support_k}"
        ));
    }
    let n_blocks = output_dim / block_size;
    if n_blocks == 0 {
        return Err(format!(
            "linear_bulk_census: a block of size {block_size} does not fit in P={output_dim}"
        ));
    }
    let mut config = TieredFitConfig::linear_bulk(n_blocks, block_size);
    config.tier1.block_topk = support_k.min(n_blocks);
    Ok(config)
}

/// Tier-1 linear bulk at the derived width and its code-space census, fit on the
/// Tier-0-centered target (#2023 lead ruling).
///
/// The public curved fit charts the target itself, never this bulk's residual: two
/// linear atoms reconstruct a centered ring exactly, so a chart of the residual is
/// blind to in-span curvature (`code_space`). This bulk is the adjudicated account
/// beside it instead: its dead blocks, and which linear communities the census
/// prices, in bits, as curving.
pub fn linear_bulk_census(
    z: ArrayView2<'_, f64>,
    block_size: usize,
    support_k: usize,
) -> Result<TieredFitReport, String> {
    let config = derived_linear_bulk_config(z.ncols(), block_size, support_k)?;
    fit_tiered(z, &config)
}

impl TieredFitReport {
    /// The linear bulk and its code-space census as a payload record: the bulk's
    /// geometry, fit and certificate, the census tallies in bits, and the migration
    /// ledger of its dead blocks and adjudicated promotions.
    #[must_use]
    pub fn census_json(&self) -> serde_json::Value {
        let tier1 = &self.tier1;
        let census = &self.code_space;
        let live_blocks = tier1
            .block_utilization
            .iter()
            .filter(|&&utilization| utilization > 0.0)
            .count();
        serde_json::json!({
            "linear_bulk": {
                "n_blocks": tier1.block_utilization.len(),
                "block_size": tier1.block_size,
                "block_topk": tier1.block_topk,
                "live_blocks": live_blocks,
                "explained_variance": tier1.explained_variance,
                "epochs": tier1.epochs,
                "certified": tier1.convergence.certified,
                "frame_residual": tier1.convergence.frame_residual,
                "tolerance": tier1.convergence.tolerance,
            },
            "census": {
                "n_blocks_scanned": census.n_blocks_scanned,
                "n_communities": census.n_communities,
                "n_accepted": census.n_accepted,
                "pair_proposals": census.pair_proposals.len(),
                "dl_saved_bits": census.dl_saved_bits,
                "fraction_curved": census.fraction_curved,
                "distortion_floor": census.tolerance,
                "l0": census.l0,
            },
            "ledger": self.ledger.to_json(),
        })
    }
}

/// Fit the Tier-2 curved refinement: the overcomplete hard-TopK support-sparse
/// dictionary on the [`LinearPeel`]'s centered residual, through
/// [`fit_sae_support_sparse`], the path the public support-sparse fit runs.
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
    let requested_atoms = config.n_atoms;
    let fit = fit_sae_support_sparse(SaeSupportSparseFitRequest {
        target: peel.residual.view(),
        atom_basis: vec![config.atom_basis.clone(); requested_atoms],
        atom_dim: vec![config.atom_dim; requested_atoms],
        support_k: config.support_k,
        initial_smoothness: config.initial_smoothness,
        max_outer_iter: config.max_outer_iter,
        trust_radius: config.trust_radius,
        random_state: config.random_state,
    })?;
    // Composed residual against R0 is the curved fit's own residual on the peel's
    // centered target: (R0 − L) − (residual mean + fitted) = residual − fitted.
    let rss = peel
        .residual
        .iter()
        .zip(fit.fitted.iter())
        .map(|(truth, fitted)| (truth - fitted).powi(2))
        .sum::<f64>();
    let explained_variance =
        crate::tiered::explained_variance_from_sums(rss, peel.baseline_energy);
    Ok(Tier2SupportFit {
        mean: &peel.residual_mean + &Array1::from(fit.training_mean),
        retained_atoms: fit.retained_atom_indices.len(),
        term: fit.outer.term,
        lambda_smooth: fit.outer.lambda_smooth,
        criterion: fit.outer.criterion,
        fixed_point: fit.outer.fixed_point,
        outer_certificate: fit.outer.outer_certificate,
        outer_iterations: fit.outer.outer_iterations,
        requested_atoms,
        explained_variance,
        migration: fit.migration,
    })
}

/// Fold the Tier-2 support fit's own migration account into the tiered ledger.
///
/// The support lane seeds each curved atom from projections of the Tier-1
/// residual rows — the residual-factor pool — and prunes the atoms no row selected
/// at that boundary, so its ledger is the account of Tier-2's births and deaths.
/// The former tally re-recorded those births as promotions from linear atoms,
/// which no seed in this lane is.
fn record_support_moves(ledger: &mut SaeMigrationLedger, fit: &Tier2SupportFit) {
    for mv in &fit.migration.moves {
        ledger.record(mv.clone());
    }
}

#[cfg(test)]
mod fit_tests {
    use super::*;
    use crate::migration_ledger::SaeMove;
    use ndarray::Array2;

    /// The births and deaths a ledger records on one rung.
    fn stage_tally(ledger: &SaeMigrationLedger, stage: MoveStage) -> (usize, usize) {
        ledger
            .moves
            .iter()
            .fold((0, 0), |(births, deaths), mv| match &mv.kind {
                SaeMove::Birth { stage: rung, .. } if *rung == stage => (births + mv.count, deaths),
                SaeMove::Death { stage: rung, .. } if *rung == stage => (births, deaths + mv.count),
                _ => (births, deaths),
            })
    }

    /// #2023 criterion 3 on Tier-1: the linear rung records the `G` seed blocks and
    /// every committed revival as births, and every revival and the blocks dead at
    /// the end as deaths, so births minus deaths is the number of live blocks.
    fn assert_linear_blocks_accounted(report: &TieredFitReport) {
        let (births, deaths) = stage_tally(&report.ledger, MoveStage::Linear);
        let tier1 = &report.tier1;
        let live = tier1.block_utilization.iter().filter(|&&u| u > 0.0).count();
        assert_eq!(
            births,
            tier1.block_utilization.len() + tier1.committed_births,
            "every seed block and every revival is one linear birth"
        );
        assert_eq!(
            births - deaths,
            live,
            "linear births minus linear deaths must be the live block count"
        );
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
        // #2275/#2825: K=6 over ~2 planted planes is over-complete, and this fit used
        // to reach only an EV plateau with the frame residual pinned open. That was a
        // property of a SUPPORT STEP THAT WAS NOT A DESCENT STEP — the top-k gate rule
        // is the exact minimiser of the tied loss only for mutually orthogonal
        // projectors, so on an over-complete dictionary it could raise the objective
        // the frame and γ steps lower, and a fixed point it never descended toward
        // could not be certified. With the support step admitting a block only when
        // that block lowers the row's loss, the frame residual closes to 2.6e-8
        // against the same untouched 1e-6 tolerance.
        assert!(
            report.tier1.convergence.certified,
            "an over-complete linear-bulk fit must now CERTIFY; got certified=false, frame_residual={} tol={}",
            report.tier1.convergence.frame_residual, report.tier1.convergence.tolerance
        );
        assert!(
            report.tier1.convergence.frame_residual <= report.tier1.convergence.tolerance,
            "a certified fit must report frame_residual at or below tolerance; got {} > {}",
            report.tier1.convergence.frame_residual,
            report.tier1.convergence.tolerance
        );
    }

    /// #2275/#2825: at `K ≫ intrinsic-rank` this fit was read as legitimately
    /// un-certifiable — ~`K − rank` blocks are structurally spurious, AuxK revival
    /// churns their frames, and `frame_residual` sat above tolerance however many
    /// epochs it was given. That reading was wrong about its own cause. The support
    /// step ranked blocks by `‖P_g x‖` and took the top `k`, which minimises the tied
    /// loss ONLY for mutually orthogonal projectors; over-complete blocks overlap, so
    /// the step could raise the objective the frame and γ steps lower and the
    /// alternation had no fixed point to converge to. With the support step descending
    /// that objective the fit CERTIFIES, and Tier-2 — which used to fail its own
    /// support-sparse fixed point on this residual — converges too.
    #[test]
    fn tiered_certifies_at_k_gg_rank_once_the_support_step_descends_2275_2825() {
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
        // Tier-2 is only a witness that Tier-1's fit still reaches the
        // curved lane. Use the smallest overcomplete dictionary instead of the
        // production default; K=P+4 preserves K>P without importing unrelated
        // support conditioning into this trichotomy test.
        config.tier2.n_atoms = p + 4;
        config.tier2.support_k = 1;

        // The objective plateaus, so the tiered fit RETURNS instead of erroring — that
        // IS the #2275 acceptance criterion.
        let report = fit_tiered(z.view(), &config)
            .expect("#2275: the tiered fit must RETURN at K ≫ rank, not error");

        // #2825: this fit now CERTIFIES. The open certificate was not a property of
        // `K ≫ rank` — it was a property of a support step that could raise the very
        // objective the frame and γ steps lower, so the alternation had no fixed point
        // to reach. With the support step descending that objective the frame residual
        // closes to 1.1e-7 against the same untouched 1e-6 tolerance.
        assert!(
            report.tier1.convergence.certified,
            "K ≫ rank fit must now CERTIFY; got certified=false (frame_residual={}, tol={})",
            report.tier1.convergence.frame_residual, report.tier1.convergence.tolerance
        );
        assert!(
            report.tier1.convergence.frame_residual <= report.tier1.convergence.tolerance,
            "a certified fit must report frame_residual at or below tolerance; got {} > {}",
            report.tier1.convergence.frame_residual,
            report.tier1.convergence.tolerance
        );
        // The EV residual is RECORDED (finite) whether or not it reached the absolute
        // tolerance; the frame certificate above is what decides convergence, and
        // "no tolerance softening" is pinned by the exact-tolerance assertion below.
        assert!(
            report.tier1.convergence.ev_residual.is_finite(),
            "the plateaued objective residual must be recorded (finite); got {}",
            report.tier1.convergence.ev_residual
        );
        assert!(
            report.tier1.explained_variance.is_finite(),
            "Tier-1 EV must be finite"
        );
        // Tier-2 RAN on the Tier-1 residual — the clobbered contract never
        // reached it.
        assert!(
            report.tier2.is_some(),
            "#2275: Tier-2 must run on the Tier-1 residual"
        );
        assert!(
            report.explained_variance.is_finite(),
            "composed EV must be finite"
        );
        // No tolerance softening: the open certificate is measured against the SAME
        // configured tolerance, unchanged.
        assert_eq!(
            report.tier1.convergence.tolerance, config.tier1.tolerance,
            "#2275 must NOT soften tolerance; the open certificate uses the configured tol"
        );
    }

    /// #2275/#2825: at `K ≫ intrinsic-rank` the block entry now returns a CERTIFIED
    /// fit. The frame fixed-point residual an over-complete frame was said to be unable
    /// to reach was reachable all along; what could not reach it was an alternation
    /// whose support step was not a descent step on the shared objective.
    #[test]
    fn block_sparse_fixed_point_certifies_once_the_support_step_descends_2275_2825() {
        use crate::sparse_dict::{BlockSparseConfig, fit_block_sparse_dictionary};
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

        let fit = fit_block_sparse_dictionary(x.view(), &config)
            .expect("#2275: the block entry must RETURN the converged fit");
        let c = &fit.convergence;
        // #2825: the block entry now certifies this fit. See the two tiered cases
        // above — the open certificate measured a support step that was not a descent
        // step, not an over-complete frame that cannot reach a fixed point.
        assert!(
            c.certified,
            "a K ≫ rank fit must now CERTIFY; got certified=false (frame_residual={}, tol={})",
            c.frame_residual, c.tolerance
        );
        assert!(
            c.frame_residual <= c.tolerance,
            "a certified fit must report frame_residual at or below tolerance; got {} > {}",
            c.frame_residual,
            c.tolerance
        );
        assert!(
            c.ev_residual.is_finite(),
            "the plateaued objective residual must be recorded (finite); got {}",
            c.ev_residual
        );
        // No tolerance softening: the certificate is measured against the configured tol.
        assert_eq!(
            c.tolerance, config.tolerance,
            "#2275 must NOT soften tolerance"
        );
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
            stage_tally(&report.ledger, MoveStage::Curved).0,
            tier2.retained_atoms,
            "every retained curved atom is a promotion off the linear residual"
        );
        assert_linear_blocks_accounted(&report);
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
    /// discovers the ring in the block's code cloud and adjudicates it in bits. At
    /// this fixture's width the dictionary is NOT overcomplete (`G = L0`), so the
    /// spectra-only birth priority is negative — but it is a heuristic and may not
    /// veto (#2933 F22). The decision is the atomic ledger's, and the migration
    /// ledger records exactly that decision. This is the #2502 in-span curvature
    /// move wired end to end from `fit_tiered`.
    #[test]
    fn code_space_census_adjudicates_a_zero_residual_planted_ring_by_its_ledger() {
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
        println!(
            "PROBE_TIERED dl_old={} dl_new={} accept={} prescreen={:?} phase_code={:?} verdict={:?}",
            proposal.dl_old,
            proposal.dl_new,
            proposal.accept,
            proposal.crossover_prescreen,
            proposal.curved_phase_code,
            proposal.verdict
        );
        // Recognition: the ring geometry is seen (span ≈ 2, κ ring, full coverage).
        assert!(
            proposal.verdict.recognized,
            "the census must recognize the planted ring geometrically: {proposal:?}"
        );
        // At G = L0 the support dividend is zero and the priority is negative …
        assert!(
            proposal
                .crossover_prescreen
                .bits()
                .is_some_and(|bits| bits <= 0.0),
            "with no overcompleteness the priority cannot pay: {:?}",
            proposal.crossover_prescreen
        );
        // … and it does not decide: acceptance is the atomic ledger's verdict alone.
        assert_eq!(
            proposal.accept,
            proposal.dl_new < proposal.dl_old,
            "a recognized ring is accepted exactly when its ledger pays (dl_new={}, dl_old={})",
            proposal.dl_new,
            proposal.dl_old
        );
        assert_eq!(census.n_accepted, usize::from(proposal.accept));
        // Ledger provenance: the census mutates nothing, so an accepted verdict is a
        // Curved ADMISSION and never a curved birth, a refused one is a Curved
        // refusal, and the linear rung accounts for the one block.
        let curved_admissions: usize = report
            .ledger
            .moves
            .iter()
            .filter(|mv| {
                matches!(
                    mv.kind,
                    SaeMove::Admit {
                        stage: MoveStage::Curved,
                        ..
                    }
                )
            })
            .map(|mv| mv.count)
            .sum();
        assert_eq!(curved_admissions, census.n_accepted);
        assert_eq!(
            stage_tally(&report.ledger, MoveStage::Curved).0,
            0,
            "an admitted census proposal installs no curved atom"
        );
        assert_eq!(report.ledger.n_admitted, census.n_accepted);
        assert_linear_blocks_accounted(&report);
        if !proposal.accept {
            assert!(
                report.ledger.n_refusals >= 1,
                "the ledger must record the refused promotion"
            );
        }
        assert_eq!(report.ledger.pc_reseed_events, 0);
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
        // residual, never PC reseeds, and every Tier-1 block is accounted for.
        assert_eq!(
            stage_tally(&report.ledger, MoveStage::Curved).0,
            tier2.retained_atoms,
            "every retained curved atom is one curved birth"
        );
        assert_linear_blocks_accounted(&report);
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

    /// #2023 ruling: the public fit's linear bulk is `G = ⌊P/b⌋` blocks of size
    /// `b` with block TopK `min(s, G)`, and a block wider than the corpus refuses.
    #[test]
    fn derived_linear_bulk_width_follows_the_chart_width_and_support_2023() {
        let narrow = derived_linear_bulk_config(4, 1, 2).expect("P=4, b=1, s=2");
        assert_eq!(
            (narrow.tier1.n_blocks, narrow.tier1.block_size, narrow.tier1.block_topk),
            (4, 1, 2)
        );
        assert!(!narrow.tier2_enabled, "the census runs no curved tier");
        let capped = derived_linear_bulk_config(5, 2, 3).expect("P=5, b=2, s=3");
        assert_eq!(
            (capped.tier1.n_blocks, capped.tier1.block_size, capped.tier1.block_topk),
            (2, 2, 2)
        );
        assert!(derived_linear_bulk_config(2, 3, 1).is_err());
        assert!(derived_linear_bulk_config(4, 1, 0).is_err());
    }

    /// The census record carries the bulk's geometry and certificate, the census
    /// tallies and the ledger, on the certified two-circle linear bulk.
    #[test]
    fn census_json_records_the_bulk_census_and_ledger_2023() {
        let z = two_circle_fixture_2634();
        let mut config = TieredFitConfig::linear_bulk(2, 1);
        config.tier1.block_topk = 1;
        config.tier1.aux_k = 2;
        config.tier1.max_epochs = 200;
        let report = fit_tiered(z.view(), &config).expect("linear-bulk fit runs");
        assert_linear_blocks_accounted(&report);
        let record = report.census_json();
        assert_eq!(record["linear_bulk"]["n_blocks"].as_u64(), Some(2));
        assert_eq!(record["linear_bulk"]["block_size"].as_u64(), Some(1));
        assert_eq!(
            record["census"]["n_blocks_scanned"].as_u64(),
            Some(report.code_space.n_blocks_scanned as u64)
        );
        assert_eq!(record["ledger"]["pc_reseed_events"].as_u64(), Some(0));
        assert!(record["ledger"]["moves"].is_array());
    }

    /// #2023 criterion 3 on the public support-sparse entry: the fit's migration
    /// ledger accounts for every requested atom — born at the support seed, pruned
    /// there, or left with no row by a support move — records no principal-component
    /// reseed, and the payload census record is exactly the typed census (its bulk,
    /// or its refusal).
    #[test]
    fn public_support_fit_accounts_for_every_birth_and_death_2023() {
        use crate::manifold::{SaeSupportSparseFitRequest, fit_sae_support_sparse_with_census};
        let z = two_circle_fixture_2634();
        let p = z.ncols();
        let censused = fit_sae_support_sparse_with_census(SaeSupportSparseFitRequest {
            target: z.view(),
            atom_basis: vec!["periodic".to_string(); p + 1],
            atom_dim: vec![1; p + 1],
            support_k: 1,
            initial_smoothness: 1.0,
            max_outer_iter: 32,
            trust_radius: 1.0,
            random_state: 0xC0FF_EE00_D15E_A5E5,
        })
        .expect("public support-sparse fit on the two-circle fixture");
        let fit = &censused.fit;
        assert!(
            fit.outer.outer_certificate.certifies() && fit.outer.fixed_point.recurred,
            "the public entry returns only a certified, recurred fit"
        );
        let migration = &fit.migration;
        assert_eq!(migration.pc_reseed_events, 0, "the support lane never PC-reseeds");
        assert_eq!(
            migration.n_births,
            fit.retained_atom_indices.len(),
            "every retained atom is one seed birth"
        );
        let term = &fit.outer.term;
        let live = (0..term.k_atoms())
            .filter(|&atom| {
                (0..term.n_obs())
                    .any(|row| term.assignment.support_indices(row).contains(&(atom as u32)))
            })
            .count();
        assert_eq!(
            migration.n_deaths,
            fit.requested_atoms - live,
            "every requested atom not live at the end is one death: pruned at the seed or \
             left with no row by a support move"
        );
        assert_eq!(migration.n_refusals, 0);
        let record = censused.census_json();
        match &censused.linear_bulk_census {
            Ok(report) => {
                assert_eq!(
                    record["linear_bulk"]["n_blocks"].as_u64(),
                    Some(report.tier1.block_utilization.len() as u64)
                );
                assert_eq!(report.tier1.block_utilization.len(), p);
                assert_eq!(report.tier1.block_topk, 1);
                assert!(report.tier2.is_none());
            }
            Err(reason) => {
                assert_eq!(record["refused"].as_str(), Some(reason.as_str()));
            }
        }
    }
}
