//! Sequential Atom Composition (SAC) — the forward-stagewise + backfitting
//! driver that builds a `K`-atom curved dictionary ONE atom at a time instead of
//! cold-starting the simultaneous joint fit of all `K` atoms.
//!
//! # Why this exists
//!
//! On real activations the cold-start joint fit of `K > 1` manifold atoms
//! co-collapses: every atom's PC-pair seed lands in the same worthless basin, the
//! symmetric configuration is a stationary point the vanishing repulsion cannot
//! escape, and the #976 collapse-guard stack (reseed-onto-distinct-PCs, arrival
//! floors, separation barrier) then oscillates against the optimizer with
//! *declining* EV. Meanwhile every K=1 curved fit succeeds. SAC is the
//! architectural response: retire the simultaneous cold-start joint fit from the
//! training path and build `K` from proven K=1 fits (forward-stagewise fitting +
//! backfitting — Hastie–Tibshirani — meets k-SVD, where the per-atom update is a
//! certified 1-manifold penalized LAML fit rather than a rank-1 SVD). The joint solver
//! survives, demoted to a warm, guards-off polish and to evaluating the joint
//! evidence once at a converged point.
//!
//! # The three phases (SAC master plan, Part 2)
//!
//! **Phase 1 — forward births.** Maintain a running residual `R = target −
//! fitted` and a running whitened covariance `Σ` (a [`StructuredResidualModel`]
//! refit on `R` each birth, so the whitened likelihood applies from atom one).
//! Each birth SEEDS from the residual (the top gate-weighted residual factor
//! direction), fits the proven K=1 driver under `Σ`, and races two candidates:
//! (A) a genuinely-new atom whose topology is chosen by evidence at birth
//! ([`crate::structure_harvest::apply_structure_move`] → the #977 topology race),
//! versus (B) *extending the previous atom's chart* — refitting the last atom so
//! it can absorb the residual arc it left behind (stagewise arc-tiling, caught at
//! birth rather than by post-hoc fusion). Acceptance is an EVIDENCE gate (the
//! frozen joint penalized LAML criterion strictly improves) PLUS an explicit MINIMUM-EFFECT
//! floor ([`StagewiseConfig::min_effect_ev`]): with frontier-scale `n`, evidence
//! alone keeps true-but-trivial wiggles forever, so salience is a separate,
//! explicit dial (a config knob whose null-recovering default is `0.0`, never a
//! magic constant). Births stop after two consecutive rejections, or when the
//! residual carries no structured factor above the idiosyncratic-noise floor
//! (`Σ.factor_rank() == 0`), or at the caller's `max_births` safety cap.
//! Co-collapse is impossible by construction: atoms never compete inside one
//! Hessian, and the RESEED guard stack (the #976 active-mass / decoder-norm
//! reseed + co-collapse reseed-all) is DISARMED
//! ([`SaeManifoldTerm::set_guards_enabled`] `false`) — the K=1 path never trips it
//! anyway. Note this disarms only the *reseed* machinery: the separation barrier
//! (`add_sae_separation_barrier`) is assembled unconditionally and is NOT gated by
//! this toggle, so it remains as a dormant, load-bearing collinearity safety net
//! for the K≥2 backfitting polish below.
//!
//! **Phase 2 — backfitting sweeps.** Greedy ordering misassigns credit where
//! atoms overlap. Each sweep first re-solves the per-row routing jointly given all
//! current atoms at FROZEN decoders ([`SaeManifoldTerm::run_fixed_decoder_arrow_schur`]
//! — the sparse-coding step), then runs a warm, guards-off joint polish at fixed
//! ρ. Both are line-searched descent on the penalized objective, so at fixed ρ the
//! sweep is monotone by construction; the loop stops when a sweep no longer
//! strictly improves EV. (ρ moves via EFS BETWEEN sweeps in the caller's outer
//! cascade; this driver operates at the ρ it is handed, which is exactly where the
//! block-coordinate monotonicity holds.)
//!
//! **Phase 3 — terminal joint assembly.** [`terminal_joint_assembly`] merges an
//! optional Tier-1 bulk term with the SAC-composed atoms via
//! [`SaeManifoldTerm::merge_tiers`] and runs a SINGLE frozen (`inner_max_iter ==
//! 0`, the #850 freeze) arrow-Schur pass — evaluate-don't-optimize — to read the
//! joint Laplace evidence at the converged point without moving β. That freeze is
//! already exposed by [`SaeManifoldTerm::penalized_laml_criterion`] at `inner_max_iter ==
//! 0`, so no new `frozen_evaluate` primitive is needed; [`frozen_joint_penalized_laml`]
//! is the thin, named wrapper this module and its callers use.
//!
//! # Determinism & SPEC
//!
//! No RNG, no clock, no wall-clock budget, no grid search. Every threshold is a
//! typed config knob (defaulting to the null-recovering value) or a data-derived
//! quantity (the residual model's evidence-selected factor rank is the salience
//! oracle). penalized LAML throughout (the inner fits and the frozen evidence pass are the
//! same penalized LAML criterion every term is scored by).

use ndarray::{Array1, Array2, ArrayView2};

use gam_solve::inference::residual_factor::{ResidualFactorInput, StructuredResidualModel};
use gam_solve::structure_search::StructureMove;

use crate::structure_harvest::apply_structure_move;

use super::*;

/// Inner-fit knobs + the two explicit SAC dials, all typed (no env levers, no
/// magic constants). The inner-solve numbers mirror the ones the outer SAE fit
/// drives its Arrow-Schur joint fit with; the two SAC-specific dials
/// ([`Self::min_effect_ev`], the birth/sweep caps) default to null-recovering
/// values so an unconfigured driver grows K purely on evidence.
#[derive(Clone, Copy, Debug)]
pub struct StagewiseConfig {
    /// Inner Newton iterations for a full per-birth / per-sweep fit.
    pub inner_max_iter: usize,
    /// Inner Newton step size.
    pub learning_rate: f64,
    /// Ext-coordinate ridge.
    pub ridge_ext_coord: f64,
    /// β ridge.
    pub ridge_beta: f64,
    /// Hard safety cap on how many atoms the forward-birth phase may add on top of
    /// the seed atom. A BOUND, not a stop criterion (the two-consecutive-rejection
    /// rule and the residual-structure test do the stopping); it only guarantees
    /// termination on pathological inputs.
    pub max_births: usize,
    /// Maximum backfitting sweeps. Each sweep is monotone at fixed ρ; the loop
    /// also stops early when a sweep no longer strictly improves EV.
    pub max_backfit_sweeps: usize,
    /// Explicit MINIMUM-EFFECT (salience) floor on ΔEV a birth must clear on top
    /// of the evidence gate. `0.0` (the default) recovers evidence-only
    /// acceptance; a positive value suppresses true-but-trivial wiggles at
    /// frontier `n`. An explicit dial, never a magic constant.
    pub min_effect_ev: f64,
    /// Residual-factor ladder cap per birth (the number of candidate factor
    /// directions the evidence ladder scores when mining the residual for a seed).
    pub max_factor_rank: usize,
    /// Install the `Σ`-whitened per-row metric on each birth so the K=1 fits run
    /// under the structured residual covariance (the whitened likelihood from atom
    /// one). `false` keeps the isotropic path (e.g. when the caller has already
    /// installed an output-Fisher metric it must not be clobbered).
    pub structured_whitening: bool,
}

impl Default for StagewiseConfig {
    fn default() -> Self {
        Self {
            inner_max_iter: 64,
            learning_rate: 1.0,
            ridge_ext_coord: 1e-6,
            ridge_beta: 1e-6,
            max_births: 32,
            max_backfit_sweeps: 4,
            min_effect_ev: 0.0,
            max_factor_rank: 4,
            structured_whitening: true,
        }
    }
}

/// Which candidate won a birth race.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum BirthKind {
    /// A genuinely-new atom, topology chosen by evidence at birth.
    NewAtom,
    /// The previous atom's chart was extended to absorb the residual (arc-tiling
    /// caught at birth); `K` did NOT grow.
    ChartExtension,
}

/// Why the forward-birth phase stopped.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum StagewiseStop {
    /// Two consecutive birth rounds were rejected (the planned stop).
    TwoConsecutiveRejections,
    /// The `max_births` safety cap was reached.
    MaxBirths,
    /// The residual carried no structured factor above the idiosyncratic-noise
    /// floor (`Σ.factor_rank() == 0`) — nothing left to mine.
    NoResidualStructure,
    /// #2138 — the host requested cancellation (Python interrupt): the forward
    /// phase stopped early and the current best-so-far dictionary is returned.
    Cancelled,
}

/// One birth round's outcome, recorded for the honesty surface (never silent).
#[derive(Clone, Copy, Debug)]
pub struct BirthRecord {
    /// The winning candidate kind (only meaningful when `accepted`).
    pub kind: BirthKind,
    /// ΔEV the winning candidate achieved over the pre-round state.
    pub delta_ev: f64,
    /// Explained residual energy `‖Λ_:,0‖²` of the top factor the seed came from
    /// — the birth's dose, reported so a trivial-but-real wiggle is visible.
    pub factor_energy: f64,
    /// Frozen joint penalized LAML criterion before the round (lower is better evidence).
    pub joint_penalized_laml_before: f64,
    /// Frozen joint penalized LAML criterion of the winning candidate (or the unchanged
    /// pre-round value when the round was rejected).
    pub joint_penalized_laml_after: f64,
    /// Whether a candidate cleared BOTH the evidence gate and the minimum-effect
    /// floor and was adopted.
    pub accepted: bool,
}

/// The full SAC report: the birth ledger, the by-construction-monotone EV traces,
/// and the terminal joint evidence.
#[derive(Clone, Debug)]
pub struct StagewiseReport {
    /// Number of births that grew `K` (accepted `NewAtom` rounds).
    pub births_accepted: usize,
    /// Number of rejected birth rounds.
    pub births_rejected: usize,
    /// Per-round birth records (accepted and rejected), in order.
    pub birth_records: Vec<BirthRecord>,
    /// EV after the seed fit and after each ACCEPTED birth. Non-decreasing
    /// because every adopted candidate is gated on measured `ΔEV ≥ min_effect_ev
    /// ≥ 0` — the recorded trace is monotone as long as the underlying candidate
    /// fits stay healthy, which for K ≥ 2 relies on the separation barrier
    /// (dormant, not gated by `guards_enabled`) keeping atoms from collapsing
    /// collinear (its gate is inactive while pairwise `c² < 0.5`).
    pub ev_trace: Vec<f64>,
    /// EV after each backfitting sweep. Each sweep is line-searched descent on the
    /// PENALIZED objective (monotone there); the recorded EV trace is non-decreasing
    /// under the keep-best acceptance (a sweep that does not strictly improve EV is
    /// reverted), again while atoms stay separated (barrier gate inactive, `c² < 0.5`).
    pub backfit_ev_trace: Vec<f64>,
    /// Why the forward-birth phase stopped.
    pub stopped_reason: StagewiseStop,
    /// The frozen (evaluate-don't-optimize) joint penalized LAML criterion of the final
    /// composed dictionary — the terminal Phase-3 evidence.
    pub terminal_joint_penalized_laml: f64,
    /// The loss breakdown at the frozen terminal state.
    pub terminal_joint_loss: SaeManifoldLoss,
}

/// The composed dictionary + its ρ + the SAC report.
#[derive(Clone, Debug)]
pub struct StagewiseResult {
    pub term: SaeManifoldTerm,
    pub rho: SaeManifoldRho,
    pub report: StagewiseReport,
}

/// Stagewise progress event emitted at durable SAC phase boundaries.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum StagewiseEventKind {
    SeedReady,
    BirthRoundStarted,
    ResidualModelStarted,
    ResidualModelFitted,
    CurrentEvidenceStarted,
    CurrentEvidenceFinished,
    CandidateStarted,
    CandidateFinished,
    BirthAccepted,
    BirthRejected,
    BackfitSweepStarted,
    BackfitSweepAccepted,
    BackfitSweepRejected,
    TerminalEvidenceCompleted,
}

/// A real-time, checkpoint-capable view of the current SAC state. When
/// `checkpoint` is true, `term`/`rho` name a durable parent state that can be
/// serialized and resumed by the caller; candidate events are progress-only.
pub struct StagewiseProgress<'a> {
    pub event: StagewiseEventKind,
    pub birth_round: usize,
    pub backfit_sweep: usize,
    pub candidate: Option<BirthKind>,
    pub accepted: Option<bool>,
    pub checkpoint: bool,
    pub k_atoms: usize,
    pub births_accepted: usize,
    pub births_rejected: usize,
    pub ev: Option<f64>,
    pub factor_energy: Option<f64>,
    pub joint_penalized_laml_before: Option<f64>,
    pub joint_penalized_laml_after: Option<f64>,
    pub terminal_joint_penalized_laml: Option<f64>,
    pub term: &'a SaeManifoldTerm,
    pub rho: &'a SaeManifoldRho,
}

/// Callback hook for progress and per-birth checkpointing. The callback may
/// return an error to abort the fit cleanly.
pub type StagewiseProgressCallback<'cb> =
    dyn for<'event> FnMut(StagewiseProgress<'event>) -> Result<(), String> + 'cb;

fn emit_stagewise_progress(
    progress: &mut Option<&mut StagewiseProgressCallback<'_>>,
    event: StagewiseProgress<'_>,
) -> Result<(), String> {
    // Birth-round OUTCOMES are additionally logged at the default-visible level
    // (the same grade the #1026 incumbent-restore and shape-uncertainty
    // fallback lines use; the crate's default filter is Warn): a stagewise fit
    // on real data is a multi-hour loop of exactly `births + rejections` such
    // events, and a driver that wires no progress callback (the pyffi
    // `sae_manifold_fit_stagewise` path) was LOG-SILENT for the whole fit — a
    // 7 h T2 circle run produced zero lines, so a live fit was
    // indistinguishable from a hang. Cadence is bounded by `max_births + 2`
    // per fit, so this is not a per-row/per-iteration hot-path log.
    match event.event {
        StagewiseEventKind::BirthAccepted | StagewiseEventKind::BirthRejected => {
            let fmt = |v: Option<f64>| v.map_or_else(|| "-".to_string(), |x| format!("{x:.4}"));
            log::warn!(
                "[stagewise] birth round {} {:?}: K={} accepted={} rejected={} ev={} penalized_laml {} -> {}",
                event.birth_round,
                event.event,
                event.k_atoms,
                event.births_accepted,
                event.births_rejected,
                fmt(event.ev),
                fmt(event.joint_penalized_laml_before),
                fmt(event.joint_penalized_laml_after),
            );
        }
        _ => {}
    }
    if let Some(callback) = progress.as_deref_mut() {
        callback(event)?;
    }
    Ok(())
}

fn current_residual(
    term: &SaeManifoldTerm,
    target: ArrayView2<'_, f64>,
) -> Result<Array2<f64>, String> {
    let fitted = term.try_fitted()?;
    Ok(&target.to_owned() - &fitted)
}

/// Frozen (`inner_max_iter == 0`, the #850 freeze) joint penalized LAML criterion of a term
/// at its current `(t, β)` — evaluate-don't-optimize. This is the joint-Laplace
/// evidence at a fixed converged state (`loss.total() + extra penalties + ½

/// Refresh the structured per-row metric on the FINAL residual before a
/// terminal frozen joint evidence: the birth loop installs Σ⁻¹ fitted BEFORE
/// each birth, so after the last accepted birth the installed metric still
/// describes the PREVIOUS dictionary's residual. Scoring the terminal composed
/// tier under that stale Σ prices the data term as ½·R_{t+1}ᵀ·M_t·R_{t+1};
/// the stated convention is the running covariance of the residual actually
/// being scored. No-op when structured whitening is off or the final residual
/// carries no factor structure.
fn refresh_terminal_row_metric(
    term: &mut SaeManifoldTerm,
    target: ArrayView2<'_, f64>,
    config: &StagewiseConfig,
) -> Result<(), String> {
    if !config.structured_whitening {
        return Ok(());
    }
    let residual = current_residual(term, target)?;
    match fit_residual_covariance_on(term, residual, config) {
        Ok(Some((_, model))) => term.set_row_metric(model.row_metric(target.nrows())?)?,
        // No factor structure left in the final residual — nothing to refresh.
        Ok(None) => {}
        // A DEGENERATE final residual (e.g. a fully-explained target leaves
        // R ≈ 0, whose factor solve hits a non-PD pivot) is the same
        // nothing-to-refresh case, not a fit failure: keep the running metric
        // the last birth installed rather than aborting a converged fit at the
        // terminal bookkeeping step.
        Err(err) => {
            log::debug!("stagewise terminal Σ refresh skipped (degenerate final residual): {err}");
        }
    }
    Ok(())
}

/// log|H| − Occam`), the quantity the birth evidence gate and the terminal
/// assembly compare on. Lower is better evidence.
pub fn frozen_joint_penalized_laml(
    term: &mut SaeManifoldTerm,
    target: ArrayView2<'_, f64>,
    rho: &SaeManifoldRho,
    registry: Option<&AnalyticPenaltyRegistry>,
    config: &StagewiseConfig,
) -> Result<(f64, SaeManifoldLoss), String> {
    term.penalized_laml_criterion(
        target,
        rho,
        registry,
        0,
        config.learning_rate,
        config.ridge_ext_coord,
        config.ridge_beta,
    )
}

/// Reconstruction explained variance of a term against `target` (the centered EV
/// every SAE fit is scored by). `NaN` when EV is undefined (degenerate variance),
/// which the callers treat as "no improvement".
fn ev_of(term: &SaeManifoldTerm, target: ArrayView2<'_, f64>) -> f64 {
    match term.try_fitted() {
        Ok(fitted) => reconstruction_explained_variance(target, fitted.view()).unwrap_or(f64::NAN),
        Err(_) => f64::NAN,
    }
}

/// Per-row activity coordinate the residual-factor scale law `c(z)` is smooth in:
/// the total assignment mass on each row (an activation-strength summary — rows
/// the dictionary routes strongly should carry less unexplained factor energy).
fn activity_of(term: &SaeManifoldTerm) -> Array1<f64> {
    let assignments = term.assignment.assignments();
    let n = assignments.nrows();
    (0..n).map(|r| assignments.row(r).sum()).collect()
}

/// Fit the running structured residual-covariance `Σ` on an ALREADY-COMPUTED
/// residual — the pooled `R = target − fitted` or a stratum-local masked `R`.
/// Returns `None` when the residual is empty/single-channel (no factor
/// subspace). The stagewise loop computes the pooled residual once, applies the
/// [`stratum_local_birth_residual`] screen, and mines whichever residual clears
/// the router floor — but the factor fit and per-row activity are otherwise
/// identical, so this is the shared body. (The unscreened pooled path lives on
/// only as the `fit_residual_covariance` oracle in the `tests` module, which the
/// stratum-local tests compare against.)
fn fit_residual_covariance_on(
    term: &SaeManifoldTerm,
    residual: Array2<f64>,
    config: &StagewiseConfig,
) -> Result<Option<(Array2<f64>, StructuredResidualModel)>, String> {
    let (n, p) = residual.dim();
    if n == 0 || p < 2 {
        return Ok(None);
    }
    let activity = activity_of(term);
    let max_rank = config.max_factor_rank.min(p.saturating_sub(1)).max(1);
    StructuredResidualModel::fit(ResidualFactorInput {
        residuals: residual.view(),
        activity: activity.view(),
        max_factor_rank: max_rank,
    })
    .map(|model| Some((residual, model)))
    .map_err(|err| format!("fit_residual_covariance: structured residual fit failed: {err}"))
}

/// The residual the next birth should be mined on, chosen by the routing floor.
///
/// #P3 — stratum-local births. Compute the pooled residual `R = target − fitted`,
/// then screen it with [`stratum_local_birth_residual`] against the router floor
/// `routability_floor(p, K, 1, 1)` at `K = k_atoms + max_births` (the widest this run
/// can grow — derived, no new knob). If a rare high-residual STRATUM carries a
/// dominant direction whose LOCAL own-subspace energy fraction clears the floor while
/// the POOLED fraction does NOT, mine the birth on that stratum's rows (the masked
/// residual); the diffuse structure that sits below the floor on the pooled residual
/// by construction is then reachable. When the pooled residual already routes (small
/// `K`, or a globally dominant direction) the screen changes nothing — the pooled
/// residual is mined, bit-for-bit as before.
fn birth_mining_residual(
    term: &SaeManifoldTerm,
    target: ArrayView2<'_, f64>,
    config: &StagewiseConfig,
) -> Result<Array2<f64>, String> {
    let pooled = current_residual(term, target)?;
    let (n, p) = pooled.dim();
    if n == 0 || p < 2 {
        return Ok(pooled);
    }
    // Router width the eventual dictionary competes at: the current atoms plus the
    // remaining birth budget (the widest this run reaches). `max(2)` keeps the
    // union-bound log well defined for the very first birth off a single seed.
    let k_router = (term.k_atoms() + config.max_births).max(2);
    let floor = crate::routability::routability_floor(p, k_router, 1, 1.0);
    let min_routable = crate::routability::minimum_routable_energy(&floor);
    // Only re-target when the POOLED birth would actually fail the floor: if the
    // pooled dominant direction already clears it, the historical pooled path is
    // kept unchanged (no behavior change where births already routed).
    let all_rows: Vec<usize> = (0..n).collect();
    let pooled_fraction = dominant_energy_fraction(pooled.view(), &all_rows);
    if pooled_fraction >= min_routable {
        return Ok(pooled);
    }
    match stratum_local_birth_residual(pooled.view(), &floor) {
        Some(pick) => Ok(pick.masked_residual),
        None => Ok(pooled),
    }
}

fn fit_single_atom_response_in_place(
    term: &mut SaeManifoldTerm,
    rho: &mut SaeManifoldRho,
    atom_idx: usize,
    response: ArrayView2<'_, f64>,
    registry: Option<&AnalyticPenaltyRegistry>,
    config: &StagewiseConfig,
) -> Result<(), String> {
    let n = term.n_obs();
    let k = term.k_atoms();
    if atom_idx >= k {
        return Err(format!(
            "fit_single_atom_response_in_place: atom {atom_idx} out of range (K={k})"
        ));
    }
    let sub_atom = term.atoms[atom_idx].clone();
    let coord_block = term.assignment.coords[atom_idx].clone();
    let mut sub_logits = Array2::<f64>::zeros((n, 1));
    for row in 0..n {
        sub_logits[[row, 0]] = term.assignment.logits[[row, atom_idx]];
    }
    let sub_assignment =
        SaeAssignment::with_mode(sub_logits, vec![coord_block], term.assignment.mode)?;
    let mut sub_term = SaeManifoldTerm::new(vec![sub_atom], sub_assignment)?;
    sub_term.set_guards_enabled(false);
    // The residual decoder is already the physical fitted function. Its
    // magnitude stays in the coefficients so smoothness, Newton curvature, and
    // the line-search value all describe the same model.
    if let Some(w) = term.row_loss_weights().map(|w| w.to_vec()) {
        sub_term.set_row_loss_weights(w)?;
    }
    if let Some(metric) = term.row_metric().cloned() {
        sub_term.set_row_metric(metric)?;
    }
    // Memory wall: seed this curved atom's low-rank decoder frame from the
    // residual it will explain, BEFORE the dense fit, so the arrow-Schur assembly
    // takes its `frames_engaged` factored path (border `M·r`, posterior
    // covariance `(M·r)²`) instead of materializing the dense `O((M·p)²)` joint
    // Hessian that OOMs at LLM width. Each K=1 atom that keeps its frame carries
    // it into the merged dictionary, so `terminal_joint_assembly`'s frozen pass
    // over the K-atom term also assembles the `Σ M_k·r` factored border.
    //
    // Only when frames will actually engage: the factored β-tier identity
    // `Φᵀ(G⊗I)Φ = G⊗(UᵀU)` is exact only for an isotropic likelihood, so a
    // whitened term (`whitens_likelihood`) stays on the certified dense path
    // (#974). Seeding under whitening would project the decoder to rank `r` for
    // no assembly benefit, so skip it there.
    if sub_term
        .row_metric()
        .map(|m| !m.whitens_likelihood())
        .unwrap_or(true)
    {
        let frame_rows: Vec<usize> = (0..n).collect();
        crate::manifold::activate_residual_frame(
            &mut sub_term.atoms[0],
            response,
            &frame_rows,
            &crate::manifold::InFrameCurvedConfig::default(),
        )?;
    }
    let mut sub_rho = SaeManifoldRho::with_per_atom_smooth(
        rho.log_lambda_sparse,
        vec![*rho.log_lambda_smooth.get(atom_idx).unwrap_or(&0.0)],
        vec![
            rho.log_ard
                .get(atom_idx)
                .cloned()
                .unwrap_or_else(|| Array1::zeros(0)),
        ],
    );
    sub_term.run_joint_fit_arrow_schur(
        response,
        &mut sub_rho,
        registry,
        config.inner_max_iter,
        config.learning_rate,
        config.ridge_ext_coord,
        config.ridge_beta,
    )?;

    term.atoms[atom_idx] = sub_term.atoms[0].clone();
    term.assignment.coords[atom_idx] = sub_term.assignment.coords[0].clone();
    for row in 0..n {
        term.assignment.logits[[row, atom_idx]] = sub_term.assignment.logits[[row, 0]];
    }
    if atom_idx < rho.log_lambda_smooth.len() {
        rho.log_lambda_smooth[atom_idx] = sub_rho.log_lambda_smooth[0];
    }
    if atom_idx < rho.log_ard.len() {
        rho.log_ard[atom_idx] = sub_rho.log_ard[0].clone();
    }
    term.assignment.frozen_logits = None;
    term.last_row_layout = None;
    term.last_frames_active = false;
    term.border_hbb_workspace = Array2::<f64>::zeros((0, 0));
    Ok(())
}

/// Per-row ANCHOR WEIGHT for birth-seed selection: how UNCONTESTED each row is by
/// the existing dictionary. A birth is "singly-attributable" (#2080) when the new
/// factor is present on rows where the existing atoms are NOT active — those rows
/// give the born atom its own territory, so the joint gate cannot re-route it onto
/// an incumbent atom's rows (the co-collapse magnet). We measure contestedness by
/// the per-row total assignment mass (`activity_of`) and reward the rows the
/// dictionary leaves cold: `w_i = max_activity − activity_i` (≥ 0). When the routing
/// is UNIFORM (every row equally contested — e.g. the very first birth off a single
/// seed active everywhere) all weights collapse to 0 and the caller falls back to the
/// historical dominant-energy pick, so small-K behavior is unchanged.
fn birth_anchor_weights(term: &SaeManifoldTerm) -> Array1<f64> {
    let activity = activity_of(term);
    let m_max = activity.iter().copied().fold(0.0_f64, f64::max);
    if m_max > 0.0 {
        activity.mapv(|m| (m_max - m).max(0.0))
    } else {
        // No existing activity at all (nothing to contest): every row is an anchor.
        Array1::ones(activity.len())
    }
}

/// A birth-candidate seed. `decoder` is the born atom's decoder in atom-0's basis;
/// `energy` is the chosen direction's explained variance (the reported dose). When
/// the residual carries a genuine rank-2 (circular) structure, `circle_coords` is
/// `Some(t)` — a PHASE-ALIGNED per-row coordinate `(n, 1)` — so the born atom is
/// seeded directly ON a circle (harmonic decoder rows + aligned chart) rather than
/// at the DC-row stationary point that leaves cos/sin dead (#2101). `None` is the
/// rank-1 / shared-factor fallback: the historical row-0 (DC) seed that
/// [`crate::structure_harvest::apply_structure_move`]'s `Birth` races the topology on.
struct BirthSeed {
    decoder: Array2<f64>,
    energy: f64,
    circle_coords: Option<Array2<f64>>,
    /// Per-row OWN-PRESENCE gate seed for the born circle (#2109): on a row where its
    /// 2-plane energy `ρ_i²` clears the derived noise floor `2·λ₊`, the entry is the
    /// log signal-to-noise ratio `ln(ρ_i² / 2·λ₊)` — a routing logit derived from the
    /// born circle's OWN presence strength, not incumbent activity. Absent rows carry
    /// `f64::NEG_INFINITY` (the conservative birth default). `born_circle_atom` routes
    /// each present row at the STRONGER of this own-presence gate and the incumbent
    /// per-row logit scale, so a circle genuinely present on incumbent-SPARSE rows
    /// (low/negative `inc_max`) still gets a gate strong enough to ESTABLISH under ordered Beta--Bernoulli
    /// (the flat `BIRTH_SEED_LOGIT` starves it, and the incumbent scale is weak where
    /// the circle actually lives). Derived from `ρ_i` + the existing `λ₊` floor, no new
    /// constant. `None` for the rank-1 / shared-factor DC fallback.
    circle_gate: Option<Vec<f64>>,
}

fn template_accepts_circle_births(term: &SaeManifoldTerm) -> bool {
    term.atoms
        .first()
        .map(|atom| atom.basis_kind == SaeAtomBasisKind::Periodic && atom.basis_size() >= 3)
        .unwrap_or(false)
}

/// Lift a residual-factor direction to an `(m, p)` birth decoder in atom 0's basis:
/// the `p`-vector direction placed on the constant (row-0) basis row, exactly the
/// contract [`crate::structure_harvest::apply_structure_move`]'s `Birth` expects
/// (`born_atom` then races the topology and reshapes the seed). Returns the decoder
/// and the chosen factor's explained energy (the reported dose).
///
/// #2080 — ANCHOR-SCORED birth-seed selection. The evidence ladder already selected
/// `r` factor directions that each earn their complexity; the OLD selection then
/// always birthed column 0 (the dominant residual VARIANCE). On entangled / decaying-
/// amplitude data that repeatedly grabs the same dominant residual mixture, so
/// successive births pile onto one direction and are born as degenerate rank-1 lines
/// (red-tree's "2/6 factors recovered"). Instead, among the evidence-worthy columns
/// pick the one whose residual support is most concentrated on ANCHOR rows — rows the
/// existing dictionary leaves uncontested (`birth_anchor_weights`) — i.e. the most
/// SINGLY-ATTRIBUTABLE factor, which lands on its own territory and separates rather
/// than co-collapsing. Ties (and a uniform, contrast-free routing) fall back to the
/// energy order, so the first birth off a single seed is byte-for-byte the old pick.
fn top_factor_birth_decoder(
    term: &SaeManifoldTerm,
    model: &StructuredResidualModel,
    residual: ArrayView2<'_, f64>,
) -> Option<BirthSeed> {
    let r = model.factor_rank();
    if r == 0 {
        return None;
    }
    let factor = model.factor(); // (p, r), columns in descending explained energy
    let p = factor.nrows();
    let (n, p_res) = residual.dim();
    if p_res != p || n == 0 {
        return None;
    }
    // #2109 — MIRROR the #2101 rank-2 circle seed + presence-derived gate into the
    // shared-factor (entangled-residual) birth path. When the residual actually
    // carries a genuine DEGENERATE 2-plane (a real circle, not a rank-1 shared
    // factor), seed the born atom directly ON that circle with the own-presence gate
    // — exactly the disjoint principal path — rather than the flat row-0 DC seed that
    // dies under ordered Beta--Bernoulli on incumbent-sparse rows. Only a real circle (`circle_coords`
    // Some) is adopted; a genuine rank-1 shared factor returns a DC seed here, which
    // we IGNORE and fall through to the anchor-scored factor pick below, so the #2080
    // factor-selection behavior on non-circle residuals is unchanged. The circle
    // detection + its noise floor are derived from the SAME residual, no new constant.
    if let Some(circle) = residual_principal_birth_candidate(term, residual) {
        if circle.circle_coords.is_some() {
            return Some(circle);
        }
    }
    let anchor_w = birth_anchor_weights(term);
    let anchor_total: f64 = anchor_w.iter().sum();
    // No anchor CONTRAST (uniform routing) ⇒ the historical dominant-energy pick.
    let use_anchor = anchor_total > 0.0;

    // Score each evidence-worthy factor direction by the FRACTION of its residual
    // support energy `(R·û_j)²` that lands on anchor (uncontested) rows. Columns are
    // energy-ordered, and a strict `>` keeps the lower (higher-energy) index on an
    // exact tie — so a uniform anchor field reproduces the column-0 pick exactly.
    let mut best_j = 0usize;
    let mut best_score = f64::NEG_INFINITY;
    if use_anchor {
        for j in 0..r {
            let col = factor.column(j);
            let energy: f64 = col.iter().map(|v| v * v).sum();
            if !(energy > 0.0) {
                continue;
            }
            let inv_norm = 1.0 / energy.sqrt();
            let mut num = 0.0_f64; // Σ_i w_i · s_i²
            let mut den = 0.0_f64; // Σ_i s_i²
            for i in 0..n {
                let mut proj = 0.0_f64;
                for out in 0..p {
                    proj += residual[[i, out]] * col[out];
                }
                let s = (proj * inv_norm) * (proj * inv_norm);
                num += anchor_w[i] * s;
                den += s;
            }
            if den <= 0.0 {
                continue;
            }
            let score = num / den;
            if score > best_score {
                best_score = score;
                best_j = j;
            }
        }
    }

    let chosen = if use_anchor { best_j } else { 0 };
    let energy: f64 = factor.column(chosen).iter().map(|v| v * v).sum();
    if !(energy > 0.0) {
        return None;
    }
    let m = term.atoms[0].basis_size();
    let mut decoder = Array2::<f64>::zeros((m, p));
    for out in 0..p {
        decoder[[0, out]] = factor[[out, chosen]];
    }
    // Genuine rank-1 shared factor: keep the historical row-0 (DC) seed + topology
    // race. (A degenerate 2-plane circle in this residual was already caught and
    // returned as a rank-2 circle seed by the #2109 mirror at the top of this fn.)
    Some(BirthSeed {
        decoder,
        energy,
        circle_coords: None,
        circle_gate: None,
    })
}

/// #2080 DISJOINT-extraction fallback birth candidate. [`StructuredResidualModel`]
/// detects only SHARED low-rank (off-diagonal, correlated) residual structure. A
/// dictionary of DISJOINT factors — orthogonal circles each in its own 2-plane with
/// independent phases — leaves a nearly BLOCK-DIAGONAL residual covariance, which the
/// evidence ladder correctly attributes to the idiosyncratic diagonal `D`, so
/// `factor_rank() == 0` and the forward-birth phase would STOP at `k = 1` despite
/// abundant remaining structure (the observed disjoint 6-circle `1/6` recovery).
///
/// When the factor model is rank-0 but the residual still carries ABOVE-NOISE
/// variance, seed the birth from the residual's dominant PRINCIPAL direction(s)
/// instead. The stop is now a DERIVED noise floor, not `factor_rank == 0`: the
/// residual-covariance eigenspectrum is thresholded at the Marchenko–Pastur top edge
/// `λ₊ = σ̂²·(1 + √(p/n))²` — the analytic largest eigenvalue a sample covariance of
/// white noise at aspect `p/n` produces — with `σ̂²` the median of the lower-half
/// eigenvalues (a robust noise scale, unbiased while ≤ p/2 directions are signal).
/// A direction above `λ₊` is real structure; none above ⇒ the residual is noise and
/// the phase stops. No magic constant — the edge is the null distribution.
///
/// This is a FALLBACK to unblock GROWTH only. The birth EVIDENCE gate + anchor
/// scoring downstream remain the quality control that decides birth-or-stop, so a
/// variance-seeded candidate is safe: a weak one is rejected by the gate. It does
/// NOT replace the factor/anchor seed path — on real activations global principal
/// components are semantic mush, so the factor+anchor path stays PRIMARY and this
/// engages ONLY when the factor model finds nothing but the noise-floor test says
/// structure remains. (Do not "simplify" this to always-PCA.) The chosen direction
/// is anchor-scored exactly like the factor path (fraction of residual support on
/// the dictionary's uncontested rows).
fn residual_principal_birth_candidate(
    term: &SaeManifoldTerm,
    residual: ArrayView2<'_, f64>,
) -> Option<BirthSeed> {
    let (n, p) = residual.dim();
    if n < 2 || p == 0 || term.atoms.is_empty() {
        return None;
    }
    // Residual eigenstructure + the derived Marchenko-Pastur floor context,
    // computed by the shared ISA producer module (`isa_seed::isa_eigen_parts`):
    // the median-eigenvalue noise scale, the analytic MP top edge
    // `lambda_plus = sigma^2 (1 + sqrt(p/n))^2`, the above-floor index set, and the
    // bottom-quartile certificate noise scale. `None` means no direction clears
    // the floor: the residual is noise and the forward-birth phase stops (the
    // derived-floor stop, not `factor_rank == 0`).
    let parts = isa_eigen_parts(residual).ok()??;
    // Anchor-score the above-floor directions exactly like the factor path.
    let anchor_w = birth_anchor_weights(term);
    let mut best = parts.above[0];
    if anchor_w.iter().sum::<f64>() > 0.0 {
        let mut best_score = f64::NEG_INFINITY;
        for &k in &parts.above {
            let col = parts.evecs.column(k); // unit-norm eigenvector
            let mut num = 0.0_f64;
            let mut den = 0.0_f64;
            for i in 0..n {
                let mut proj = 0.0_f64;
                for j in 0..p {
                    proj += residual[[i, j]] * col[j];
                }
                let si = proj * proj;
                num += anchor_w[i] * si;
                den += si;
            }
            if den > 0.0 {
                let score = num / den;
                if score > best_score {
                    best_score = score;
                    best = k;
                }
            }
        }
    }
    let energy = parts.evals[best].max(0.0);
    if !(energy > 0.0) {
        return None;
    }
    let m = term.atoms[0].basis_size();

    // #2101 / #2111 CIRCLE SEED via the ISA joint-rotation producer. A disjoint
    // circle occupies a rank-2 PLANE (its cos/sin axes carry ~equal variance),
    // so the residual's dominant structure is a 2-plane, not one direction; and
    // on a DENSE product-of-circles residual whitening exhausts second order,
    // so eigenvector pairing returns Davis-Kahan BLENDS across circles (the
    // K >= 2 co-collapse). The identifying signal blends cannot mimic is FOURTH
    // order: `isa_extract_certified_plane` whitens the above-floor subspace,
    // runs multistart 2-plane Jacobi rotations maximizing the independence
    // contrast `(kappa - 2)^2` (dense clean circle kappa ~ 1, gated circle 1/q,
    // Gaussian blend exactly 2), and accepts only on the analytic-anchor
    // certificate — see `isa_seed` for the math and derivations. The single-birth
    // path still captures the full above-floor span and emits only the first
    // certified plane from that resolved joint queue; the batched path below
    // consumes every certified plane from one joint split. A residual carrying only
    // blends/saddles certifies nothing and falls through to the rank-1 seed
    // (no hallucinated circle birth).
    if template_accepts_circle_births(term) && parts.above.len() >= 2 {
        let joint_span_parts = capture_signal_span(residual, parts.above.len())
            .ok()
            .flatten();
        if let Some(cand) = joint_span_parts
            .as_ref()
            .and_then(|span| isa_extract_certified_plane(residual, span, &IsaSeedConfig::default()))
        {
            // Decoder on the cos/sin harmonic rows at the LS harmonic
            // amplitudes; phase chart + own-presence gate carried through
            // unchanged (the #2109 contract).
            let mut decoder = Array2::<f64>::zeros((m, p));
            for j in 0..p {
                decoder[[1, j]] = cand.amplitudes[0] * cand.basis[[j, 0]];
                decoder[[2, j]] = cand.amplitudes[1] * cand.basis[[j, 1]];
            }
            return Some(BirthSeed {
                decoder,
                energy,
                circle_coords: Some(cand.phases_turns),
                circle_gate: Some(cand.gate_logits),
            });
        }
    }

    // TRAILHEAD (identifiability converse to the MP floor): the floor above
    // certifies "there IS above-noise signal" but NOT that the leading direction
    // is IDENTIFIABLE. When the top above-floor eigenvalues are ~equal AND share
    // support (an isotropic/rotationally-degenerate plateau — e.g. a Gaussian
    // null's leading bulk after linear T1), the sample top eigenvector is a
    // Davis–Kahan blend fixed by noise, so a rank-1 seed from it is a doomed atom
    // the inner joint fit can only reject at ΔEV≈0 — the null-calibration birth
    // thrash. A birth gate that STOPS here would end that thrash, but it must key
    // on eigenvector STABILITY, not the raw eigengap alone: disjoint equal-energy
    // blocks (block-diagonal support) have pinned, identifiable eigenvectors and
    // must still seed (stagewise deflation takes the rest). Calibrating that
    // stability test (and proving it helps on a real null residual without
    // suppressing genuine weak births) is deferred — do not gate on a bare
    // relative-eigengap threshold, which conflates the two regimes.

    // Rank-1 fallback (a genuine line, a partially-extracted circle, or a
    // non-periodic template): keep the historical row-0 (DC) seed + topology race.
    let amp = energy.sqrt();
    let mut decoder = Array2::<f64>::zeros((m, p));
    for j in 0..p {
        decoder[[0, j]] = amp * parts.evecs[[j, best]];
    }
    Some(BirthSeed {
        decoder,
        energy,
        circle_coords: None,
        circle_gate: None,
    })
}

fn isa_birth_seed_batch(
    term: &SaeManifoldTerm,
    residual: ArrayView2<'_, f64>,
    max_planes: usize,
) -> Result<Vec<BirthSeed>, String> {
    if max_planes == 0 || !template_accepts_circle_births(term) {
        return Ok(Vec::new());
    }
    let harvest = isa_deflationary_producer(residual, max_planes, &IsaSeedConfig::default())?;
    Ok(harvest
        .planes
        .iter()
        .map(|cand| plane_to_birth_seed(term, cand))
        .collect())
}

/// Refit a SINGLE atom `k` in place on its leave-one-atom-out partial residual —
/// the k-SVD-style certified per-atom update. The LOO residual is
/// `e_k = target − Σ_{j≠k} a_j g_j = target − fitted + a_k g_k` (computed with the
/// exact per-row gate weights, mirroring
/// [`SaeManifoldTerm::per_atom_loao_explained_variance`]), so atom `k` is refit to
/// the structure the rest of the dictionary leaves behind. Atom `k` is extracted
/// as a K=1 sub-term (guards off — a single atom never trips them), fit with the
/// proven K=1 driver, and its decoder / coordinates / routing column written back.
///
/// Used both as the Phase-2 per-atom refit and as the chart-EXTENSION birth
/// candidate (refit the last atom on its LOO residual so it can absorb the arc it
/// left behind). Independent-gate modes (JumpReLU / ordered Beta--Bernoulli) refit exactly-additively;
/// under Softmax the K=1 sub-gate is the constant `1`, so the sub-fit sees the
/// full partial residual (classical additive backfitting) and the subsequent warm
/// polish reconciles the re-normalized joint gate.
fn refit_single_atom_in_place(
    term: &mut SaeManifoldTerm,
    rho: &SaeManifoldRho,
    atom_idx: usize,
    target: ArrayView2<'_, f64>,
    registry: Option<&AnalyticPenaltyRegistry>,
    config: &StagewiseConfig,
) -> Result<(), String> {
    let n = term.n_obs();
    let p = term.output_dim();
    let k = term.k_atoms();
    if atom_idx >= k {
        return Err(format!(
            "refit_single_atom_in_place: atom {atom_idx} out of range (K={k})"
        ));
    }
    // Leave-one-atom-out partial residual e_k (n × p).
    let full = term.try_fitted_for_rho(rho)?;
    let mut e_k = &target.to_owned() - &full;
    let mut g_buf = vec![0.0_f64; p];
    for row in 0..n {
        let weights = term.assignment.try_assignments_row(row)?;
        let a_k = weights[atom_idx];
        if a_k == 0.0 {
            continue;
        }
        term.atoms[atom_idx].fill_decoded_row(row, &mut g_buf);
        let mut e_row = e_k.row_mut(row);
        for out in 0..p {
            e_row[out] += a_k * g_buf[out];
        }
    }

    let mut rho_scratch = rho.clone();
    fit_single_atom_response_in_place(
        term,
        &mut rho_scratch,
        atom_idx,
        e_k.view(),
        registry,
        config,
    )
}

/// One backfitting sweep at FIXED ρ: (1) re-solve the per-row routing jointly at
/// frozen decoders (the sparse-coding step), then (2) a warm joint polish with the
/// RESEED guards disarmed. Both are line-searched descent on the penalized
/// objective, so the sweep is monotone in that objective by construction. The
/// K ≥ 2 joint polish still leans on the separation barrier
/// (`add_sae_separation_barrier`, assembled unconditionally — NOT gated by
/// `guards_enabled`) as its collinearity defense: disarming the guards removes
/// only the reseed machinery, not that barrier. A step that fails to assemble is a
/// no-op (never a hard error — the state is left as the last good iterate).
fn backfit_sweep(
    term: &mut SaeManifoldTerm,
    rho: &mut SaeManifoldRho,
    target: ArrayView2<'_, f64>,
    registry: Option<&AnalyticPenaltyRegistry>,
    config: &StagewiseConfig,
) -> Result<(), String> {
    term.set_guards_enabled(false);
    // Routing step: re-solve gates + coordinates jointly at frozen decoders. A
    // failed assemble is a no-op (the state stays at the last good iterate); the
    // `.ok()` discards the must-use result without an underscore-let.
    term.run_fixed_decoder_arrow_schur(
        target,
        rho,
        registry,
        1,
        config.learning_rate,
        config.ridge_ext_coord,
    )
    .ok();
    // Warm joint polish (guards off): reassigns credit across atoms via the
    // damped Newton line search. Warm-started from the composed dictionary, so the
    // co-collapse that bites the cold-start joint fit cannot arise here.
    term.run_joint_fit_arrow_schur(
        target,
        rho,
        registry,
        config.inner_max_iter,
        config.learning_rate,
        config.ridge_ext_coord,
        config.ridge_beta,
    )?;
    Ok(())
}

/// Run the forward-birth phase, run the backfitting sweeps, and report the
/// terminal frozen joint evidence. `seed` MUST be a fitted single-atom (K=1) term
/// carrying the initial basis/topology and converged decoder/coordinates; `rho`
/// its matching ρ. `sample_weights` (optional, length `n`) are the subsampler's
/// per-row stratified importance weights, installed on every fit via the
/// reconstruction-weight seam.
///
/// The returned `term` is the SAC-composed curved tier (`K ≥ 1` atoms); pass it to
/// [`terminal_joint_assembly`] to merge it with a Tier-1 bulk term and read the
/// joint evidence over the full composed dictionary.
pub fn fit_stagewise(
    seed: SaeManifoldTerm,
    mut rho: SaeManifoldRho,
    target: ArrayView2<'_, f64>,
    registry: Option<&AnalyticPenaltyRegistry>,
    sample_weights: Option<&[f64]>,
    config: &StagewiseConfig,
    mut progress: Option<&mut StagewiseProgressCallback<'_>>,
    // #2138 — cooperative cancellation. When the host (the pyffi fit driver) sets
    // this after a Python interrupt, the forward-birth loop and backfit sweeps
    // bail early (returning the best-so-far dictionary with `StagewiseStop::
    // Cancelled`) so a detached compose worker stops instead of running a hung fit
    // to completion. `None` ⇒ the historical, uninterruptible path, bit-for-bit.
    cancel: Option<&std::sync::atomic::AtomicBool>,
) -> Result<StagewiseResult, String> {
    let n = target.nrows();
    if seed.k_atoms() != 1 {
        return Err(format!(
            "fit_stagewise: seed must be a single-atom (K=1) term; got K={}",
            seed.k_atoms()
        ));
    }
    if seed.n_obs() != n {
        return Err(format!(
            "fit_stagewise: seed n_obs {} != target rows {n}",
            seed.n_obs()
        ));
    }
    let mut term = seed;
    // The K=1 lane bypasses the entire #976 guard stack (it never trips it) so the
    // per-atom / backfitting refits are provably reseed-free.
    term.set_guards_enabled(false);
    if let Some(w) = sample_weights {
        if w.len() != n {
            return Err(format!(
                "fit_stagewise: sample_weights length {} != target rows {n}",
                w.len()
            ));
        }
        term.set_row_loss_weights(w.to_vec())?;
    }

    // ── Phase 1a — fitted K=1 seed checkpoint ────────────────────────────────
    // The caller owns the proven K=1 fit. The Python stagewise adapter constructs
    // this seed via `sae_manifold_fit`; re-solving it here duplicated the most
    // expensive p-wide work and, on real p=2048 residuals, could consume the whole
    // smoke timeout before the first progress callback. SAC starts from that
    // fitted atom and emits a durable checkpoint immediately.
    let mut ev_trace = vec![ev_of(&term, target)];
    let mut birth_records: Vec<BirthRecord> = Vec::new();
    let mut births_accepted = 0usize;
    let mut births_rejected = 0usize;
    let mut consecutive_rejections = 0usize;
    emit_stagewise_progress(
        &mut progress,
        StagewiseProgress {
            event: StagewiseEventKind::SeedReady,
            birth_round: 0,
            backfit_sweep: 0,
            candidate: None,
            accepted: Some(true),
            checkpoint: true,
            k_atoms: term.k_atoms(),
            births_accepted,
            births_rejected,
            ev: ev_trace.last().copied(),
            factor_energy: None,
            joint_penalized_laml_before: None,
            joint_penalized_laml_after: None,
            terminal_joint_penalized_laml: None,
            term: &term,
            rho: &rho,
        },
    )?;

    // ── Phase 1b — forward births ──────────────────────────────────────────────
    let mut birth_round = 0usize;
    let stopped_reason = loop {
        // #2138 — bail before starting another birth round if the host cancelled.
        if cancel.is_some_and(|c| c.load(std::sync::atomic::Ordering::Relaxed)) {
            break StagewiseStop::Cancelled;
        }
        if births_accepted >= config.max_births {
            break StagewiseStop::MaxBirths;
        }
        if consecutive_rejections >= 2 {
            break StagewiseStop::TwoConsecutiveRejections;
        }
        let round = birth_round;
        birth_round += 1;
        let entry_ev = ev_of(&term, target);
        emit_stagewise_progress(
            &mut progress,
            StagewiseProgress {
                event: StagewiseEventKind::BirthRoundStarted,
                birth_round: round,
                backfit_sweep: 0,
                candidate: None,
                accepted: None,
                checkpoint: true,
                k_atoms: term.k_atoms(),
                births_accepted,
                births_rejected,
                ev: Some(entry_ev),
                factor_energy: None,
                joint_penalized_laml_before: None,
                joint_penalized_laml_after: None,
                terminal_joint_penalized_laml: None,
                term: &term,
                rho: &rho,
            },
        )?;
        // Prepare the residual metric for this birth from the CURRENT term.
        // Earlier revisions queued every plane from one ISA split and then fitted
        // later queued seeds against that old residual snapshot.  That violates
        // the serial greedy contract unless the later K=1 fit is constrained to
        // the seed's output block: a dense torus shares all rows, and an
        // unconstrained single-atom fit can be attracted by strong circles that
        // have already been accepted by previous queued births.  Recomputing the
        // residual after every accepted birth is the stagewise deflation
        // mechanism; the ISA split may still be joint inside one producer call,
        // but only its best currently-certified plane is consumed by this serial
        // path.
        emit_stagewise_progress(
            &mut progress,
            StagewiseProgress {
                event: StagewiseEventKind::ResidualModelStarted,
                birth_round: round,
                backfit_sweep: 0,
                candidate: None,
                accepted: None,
                checkpoint: false,
                k_atoms: term.k_atoms(),
                births_accepted,
                births_rejected,
                ev: Some(entry_ev),
                factor_energy: None,
                joint_penalized_laml_before: None,
                joint_penalized_laml_after: None,
                terminal_joint_penalized_laml: None,
                term: &term,
                rho: &rho,
            },
        )?;
        // Certified circle residuals are tried first on this freshly-deflated
        // residual. If no circle certifies, use the anchor-scored shared-factor
        // seed, then the #2080 residual-principal rank-1 fallback.
        //
        // #P3 — the mined residual is STRATUM-LOCAL: if the pooled residual's
        // dominant direction sits below the router floor (diffuse structure invisible
        // to any width-p gate on the pooled average) but a rare high-residual stratum
        // carries it above the floor locally, mine the birth on that stratum's rows.
        // `birth_mining_residual` returns the pooled residual unchanged whenever the
        // pooled birth already routes, so small-K behavior is untouched.
        let mining_residual = birth_mining_residual(&term, target, config)?;
        let Some((residual, model)) = fit_residual_covariance_on(&term, mining_residual, config)?
        else {
            break StagewiseStop::NoResidualStructure;
        };
        let seed = if let Some(seed) = isa_birth_seed_batch(&term, residual.view(), 1)?
            .into_iter()
            .next()
        {
            seed
        } else {
            let Some(seed) = top_factor_birth_decoder(&term, &model, residual.view())
                .or_else(|| residual_principal_birth_candidate(&term, residual.view()))
            else {
                break StagewiseStop::NoResidualStructure;
            };
            seed
        };
        let factor_energy = seed.energy;
        if config.structured_whitening {
            // Install Σ^{-1} as the per-row whitened metric (carried into clones).
            term.set_row_metric(model.row_metric(n)?)?;
        }
        emit_stagewise_progress(
            &mut progress,
            StagewiseProgress {
                event: StagewiseEventKind::ResidualModelFitted,
                birth_round: round,
                backfit_sweep: 0,
                candidate: None,
                accepted: None,
                checkpoint: false,
                k_atoms: term.k_atoms(),
                births_accepted,
                births_rejected,
                ev: Some(entry_ev),
                factor_energy: Some(factor_energy),
                joint_penalized_laml_before: None,
                joint_penalized_laml_after: None,
                terminal_joint_penalized_laml: None,
                term: &term,
                rho: &rho,
            },
        )?;
        emit_stagewise_progress(
            &mut progress,
            StagewiseProgress {
                event: StagewiseEventKind::CurrentEvidenceStarted,
                birth_round: round,
                backfit_sweep: 0,
                candidate: None,
                accepted: None,
                checkpoint: false,
                k_atoms: term.k_atoms(),
                births_accepted,
                births_rejected,
                ev: Some(entry_ev),
                factor_energy: Some(factor_energy),
                joint_penalized_laml_before: None,
                joint_penalized_laml_after: None,
                terminal_joint_penalized_laml: None,
                term: &term,
                rho: &rho,
            },
        )?;
        let (current_penalized_laml, _) = frozen_joint_penalized_laml(&mut term, target, &rho, registry, config)?;
        let cur_ev = ev_of(&term, target);
        emit_stagewise_progress(
            &mut progress,
            StagewiseProgress {
                event: StagewiseEventKind::CurrentEvidenceFinished,
                birth_round: round,
                backfit_sweep: 0,
                candidate: None,
                accepted: None,
                checkpoint: false,
                k_atoms: term.k_atoms(),
                births_accepted,
                births_rejected,
                ev: Some(cur_ev),
                factor_energy: Some(factor_energy),
                joint_penalized_laml_before: Some(current_penalized_laml),
                joint_penalized_laml_after: None,
                terminal_joint_penalized_laml: None,
                term: &term,
                rho: &rho,
            },
        )?;

        // Candidate A — a genuinely-new atom (topology raced at birth).
        emit_stagewise_progress(
            &mut progress,
            StagewiseProgress {
                event: StagewiseEventKind::CandidateStarted,
                birth_round: round,
                backfit_sweep: 0,
                candidate: Some(BirthKind::NewAtom),
                accepted: None,
                checkpoint: false,
                k_atoms: term.k_atoms(),
                births_accepted,
                births_rejected,
                ev: Some(cur_ev),
                factor_energy: Some(factor_energy),
                joint_penalized_laml_before: Some(current_penalized_laml),
                joint_penalized_laml_after: None,
                terminal_joint_penalized_laml: None,
                term: &term,
                rho: &rho,
            },
        )?;
        // #2101: a circle seed (rank-2 2-plane + phase-aligned coordinate) is built
        // DIRECTLY as a Periodic atom — bypassing the topology race, which
        // parameterizes the born circle with the TEMPLATE's coordinate (the wrong
        // phase for a fresh disjoint circle). The rank-1 / shared-factor fallback
        // keeps the historical DC-row seed + race.
        let born_move = match &seed.circle_coords {
            Some(coords) => crate::structure_harvest::born_circle_atom(
                &term,
                &rho,
                seed.decoder.clone(),
                coords.clone(),
                seed.circle_gate.clone().unwrap_or_else(|| vec![0.0; n]),
            ),
            None => apply_structure_move(
                &term,
                &rho,
                &StructureMove::Birth { candidate: 0 },
                std::slice::from_ref(&seed.decoder),
            ),
        };
        let mut cand_a = born_move
            .and_then(|(mut cand_term, mut cand_rho)| {
                cand_term.set_guards_enabled(false);
                let born = cand_term.k_atoms() - 1;
                fit_single_atom_response_in_place(
                    &mut cand_term,
                    &mut cand_rho,
                    born,
                    residual.view(),
                    registry,
                    config,
                )?;
                let (penalized_laml, _) =
                    frozen_joint_penalized_laml(&mut cand_term, target, &cand_rho, registry, config)?;
                let ev = ev_of(&cand_term, target);
                Ok((cand_term, cand_rho, penalized_laml, ev))
            })
            .ok();
        if let Some((cand_term, cand_rho, penalized_laml, ev)) = cand_a.as_ref() {
            emit_stagewise_progress(
                &mut progress,
                StagewiseProgress {
                    event: StagewiseEventKind::CandidateFinished,
                    birth_round: round,
                    backfit_sweep: 0,
                    candidate: Some(BirthKind::NewAtom),
                    accepted: None,
                    checkpoint: false,
                    k_atoms: cand_term.k_atoms(),
                    births_accepted,
                    births_rejected,
                    ev: Some(*ev),
                    factor_energy: Some(factor_energy),
                    joint_penalized_laml_before: Some(current_penalized_laml),
                    joint_penalized_laml_after: Some(*penalized_laml),
                    terminal_joint_penalized_laml: None,
                    term: cand_term,
                    rho: cand_rho,
                },
            )?;
        } else {
            emit_stagewise_progress(
                &mut progress,
                StagewiseProgress {
                    event: StagewiseEventKind::CandidateFinished,
                    birth_round: round,
                    backfit_sweep: 0,
                    candidate: Some(BirthKind::NewAtom),
                    accepted: Some(false),
                    checkpoint: false,
                    k_atoms: term.k_atoms(),
                    births_accepted,
                    births_rejected,
                    ev: Some(cur_ev),
                    factor_energy: Some(factor_energy),
                    joint_penalized_laml_before: Some(current_penalized_laml),
                    joint_penalized_laml_after: None,
                    terminal_joint_penalized_laml: None,
                    term: &term,
                    rho: &rho,
                },
            )?;
        }

        // Candidate B — extend the previous atom's chart (arc-tiling). Refit the
        // last atom on its LOO residual so it can absorb the residual it left
        // behind; K does NOT grow.
        let mut cand_b = if term.k_atoms() > 1 {
            emit_stagewise_progress(
                &mut progress,
                StagewiseProgress {
                    event: StagewiseEventKind::CandidateStarted,
                    birth_round: round,
                    backfit_sweep: 0,
                    candidate: Some(BirthKind::ChartExtension),
                    accepted: None,
                    checkpoint: false,
                    k_atoms: term.k_atoms(),
                    births_accepted,
                    births_rejected,
                    ev: Some(cur_ev),
                    factor_energy: Some(factor_energy),
                    joint_penalized_laml_before: Some(current_penalized_laml),
                    joint_penalized_laml_after: None,
                    terminal_joint_penalized_laml: None,
                    term: &term,
                    rho: &rho,
                },
            )?;
            let last = term.k_atoms() - 1;
            let mut cand_term = term.clone();
            let mut cand_rho = rho.clone();
            let built = (|| -> Result<(SaeManifoldTerm, SaeManifoldRho, f64, f64), String> {
                refit_single_atom_in_place(
                    &mut cand_term,
                    &cand_rho,
                    last,
                    target,
                    registry,
                    config,
                )?;
                cand_term.set_guards_enabled(false);
                cand_term.run_joint_fit_arrow_schur(
                    target,
                    &mut cand_rho,
                    registry,
                    config.inner_max_iter,
                    config.learning_rate,
                    config.ridge_ext_coord,
                    config.ridge_beta,
                )?;
                let (penalized_laml, _) =
                    frozen_joint_penalized_laml(&mut cand_term, target, &cand_rho, registry, config)?;
                let ev = ev_of(&cand_term, target);
                Ok((cand_term, cand_rho, penalized_laml, ev))
            })();
            let out = built.ok();
            if let Some((cand_term, cand_rho, penalized_laml, ev)) = out.as_ref() {
                emit_stagewise_progress(
                    &mut progress,
                    StagewiseProgress {
                        event: StagewiseEventKind::CandidateFinished,
                        birth_round: round,
                        backfit_sweep: 0,
                        candidate: Some(BirthKind::ChartExtension),
                        accepted: None,
                        checkpoint: false,
                        k_atoms: cand_term.k_atoms(),
                        births_accepted,
                        births_rejected,
                        ev: Some(*ev),
                        factor_energy: Some(factor_energy),
                        joint_penalized_laml_before: Some(current_penalized_laml),
                        joint_penalized_laml_after: Some(*penalized_laml),
                        terminal_joint_penalized_laml: None,
                        term: cand_term,
                        rho: cand_rho,
                    },
                )?;
            } else {
                emit_stagewise_progress(
                    &mut progress,
                    StagewiseProgress {
                        event: StagewiseEventKind::CandidateFinished,
                        birth_round: round,
                        backfit_sweep: 0,
                        candidate: Some(BirthKind::ChartExtension),
                        accepted: Some(false),
                        checkpoint: false,
                        k_atoms: term.k_atoms(),
                        births_accepted,
                        births_rejected,
                        ev: Some(cur_ev),
                        factor_energy: Some(factor_energy),
                        joint_penalized_laml_before: Some(current_penalized_laml),
                        joint_penalized_laml_after: None,
                        terminal_joint_penalized_laml: None,
                        term: &term,
                        rho: &rho,
                    },
                )?;
            }
            out
        } else {
            // With K=1 the "chart extension" arm is exactly the seed fit repeated
            // against the same target under the same ρ. Skipping it removes one
            // full K=1 solve from the first birth without changing the candidate
            // set in any meaningful way; real arc-tiling only exists once a later
            // atom has left a leave-one-out residual.
            None
        };

        // Gate: strictly-improved joint evidence AND ΔEV ≥ the minimum-effect
        // floor. Among the candidates that clear both gates, the lower penalized LAML wins.
        let passes = |penalized_laml: f64, ev: f64| -> bool {
            penalized_laml.is_finite()
                && penalized_laml < current_penalized_laml
                && ev.is_finite()
                && (ev - cur_ev) >= config.min_effect_ev
        };
        let a_ok = cand_a
            .as_ref()
            .map(|&(_, _, r, e)| passes(r, e))
            .unwrap_or(false);
        let b_ok = cand_b
            .as_ref()
            .map(|&(_, _, r, e)| passes(r, e))
            .unwrap_or(false);

        let choose_a = match (a_ok, b_ok) {
            (true, true) => {
                let ar = cand_a.as_ref().unwrap().2;
                let br = cand_b.as_ref().unwrap().2;
                ar <= br
            }
            (true, false) => true,
            (false, true) => false,
            (false, false) => {
                births_rejected += 1;
                consecutive_rejections += 1;
                birth_records.push(BirthRecord {
                    kind: BirthKind::NewAtom,
                    delta_ev: 0.0,
                    factor_energy,
                    joint_penalized_laml_before: current_penalized_laml,
                    joint_penalized_laml_after: current_penalized_laml,
                    accepted: false,
                });
                emit_stagewise_progress(
                    &mut progress,
                    StagewiseProgress {
                        event: StagewiseEventKind::BirthRejected,
                        birth_round: round,
                        backfit_sweep: 0,
                        candidate: None,
                        accepted: Some(false),
                        checkpoint: true,
                        k_atoms: term.k_atoms(),
                        births_accepted,
                        births_rejected,
                        ev: Some(cur_ev),
                        factor_energy: Some(factor_energy),
                        joint_penalized_laml_before: Some(current_penalized_laml),
                        joint_penalized_laml_after: Some(current_penalized_laml),
                        terminal_joint_penalized_laml: None,
                        term: &term,
                        rho: &rho,
                    },
                )?;
                continue;
            }
        };

        let (kind, (cand_term, cand_rho, penalized_laml_after, ev_after)) = if choose_a {
            (BirthKind::NewAtom, cand_a.take().unwrap())
        } else {
            (BirthKind::ChartExtension, cand_b.take().unwrap())
        };
        term = cand_term;
        rho = cand_rho;
        births_accepted += 1;
        consecutive_rejections = 0;
        birth_records.push(BirthRecord {
            kind,
            delta_ev: ev_after - cur_ev,
            factor_energy,
            joint_penalized_laml_before: current_penalized_laml,
            joint_penalized_laml_after: penalized_laml_after,
            accepted: true,
        });
        ev_trace.push(ev_after);
        emit_stagewise_progress(
            &mut progress,
            StagewiseProgress {
                event: StagewiseEventKind::BirthAccepted,
                birth_round: round,
                backfit_sweep: 0,
                candidate: Some(kind),
                accepted: Some(true),
                checkpoint: true,
                k_atoms: term.k_atoms(),
                births_accepted,
                births_rejected,
                ev: Some(ev_after),
                factor_energy: Some(factor_energy),
                joint_penalized_laml_before: Some(current_penalized_laml),
                joint_penalized_laml_after: Some(penalized_laml_after),
                terminal_joint_penalized_laml: None,
                term: &term,
                rho: &rho,
            },
        )?;
    };

    // ── Phase 2 — backfitting sweeps (keep-best, monotone by construction) ─────
    // Each sweep minimizes the PENALIZED objective (the routing step and the warm
    // joint polish are both line-searched descent on it), whose optimum can trade
    // a hair of raw reconstruction EV for smoothness. So raw EV is not monotone
    // under the penalized descent alone. A keep-best acceptance makes the reported
    // EV trace non-decreasing BY CONSTRUCTION: a sweep is adopted only if it
    // strictly improves EV; the first non-improving sweep is reverted and the loop
    // stops (converged). Pure convergence test — no magic tolerance.
    let mut backfit_ev_trace: Vec<f64> = Vec::new();
    let mut prev_ev = *ev_trace.last().unwrap_or(&f64::NEG_INFINITY);
    for sweep in 0..config.max_backfit_sweeps {
        // #2138 — on host cancel, stop the terminal backfitting early and fall
        // through to Phase 3 so a cancelled compose returns its best-so-far
        // dictionary promptly instead of grinding every remaining sweep.
        if cancel.is_some_and(|c| c.load(std::sync::atomic::Ordering::Relaxed)) {
            break;
        }
        emit_stagewise_progress(
            &mut progress,
            StagewiseProgress {
                event: StagewiseEventKind::BackfitSweepStarted,
                birth_round,
                backfit_sweep: sweep,
                candidate: None,
                accepted: None,
                checkpoint: false,
                k_atoms: term.k_atoms(),
                births_accepted,
                births_rejected,
                ev: Some(prev_ev),
                factor_energy: None,
                joint_penalized_laml_before: None,
                joint_penalized_laml_after: None,
                terminal_joint_penalized_laml: None,
                term: &term,
                rho: &rho,
            },
        )?;
        let term_snapshot = term.clone();
        let rho_snapshot = rho.clone();
        backfit_sweep(&mut term, &mut rho, target, registry, config)?;
        let ev = ev_of(&term, target);
        if ev > prev_ev {
            backfit_ev_trace.push(ev);
            prev_ev = ev;
            emit_stagewise_progress(
                &mut progress,
                StagewiseProgress {
                    event: StagewiseEventKind::BackfitSweepAccepted,
                    birth_round,
                    backfit_sweep: sweep,
                    candidate: None,
                    accepted: Some(true),
                    checkpoint: true,
                    k_atoms: term.k_atoms(),
                    births_accepted,
                    births_rejected,
                    ev: Some(ev),
                    factor_energy: None,
                    joint_penalized_laml_before: None,
                    joint_penalized_laml_after: None,
                    terminal_joint_penalized_laml: None,
                    term: &term,
                    rho: &rho,
                },
            )?;
        } else {
            term = term_snapshot;
            rho = rho_snapshot;
            emit_stagewise_progress(
                &mut progress,
                StagewiseProgress {
                    event: StagewiseEventKind::BackfitSweepRejected,
                    birth_round,
                    backfit_sweep: sweep,
                    candidate: None,
                    accepted: Some(false),
                    checkpoint: true,
                    k_atoms: term.k_atoms(),
                    births_accepted,
                    births_rejected,
                    ev: Some(prev_ev),
                    factor_energy: None,
                    joint_penalized_laml_before: None,
                    joint_penalized_laml_after: None,
                    terminal_joint_penalized_laml: None,
                    term: &term,
                    rho: &rho,
                },
            )?;
            break;
        }
    }

    // ── Phase 3 — terminal frozen joint evidence of the composed tier ──────────
    refresh_terminal_row_metric(&mut term, target, config)?;
    let (terminal_joint_penalized_laml, terminal_joint_loss) =
        frozen_joint_penalized_laml(&mut term, target, &rho, registry, config)?;

    // Re-arm the collapse-guard stack before the composed dictionary escapes: the
    // guards-off lane is an INTERNAL economy for the K=1 / backfitting refits, but
    // the returned artifact must ship armed so any downstream non-frozen refit gets
    // the normal supervision (mirrors `terminal_joint_assembly`).
    term.set_guards_enabled(true);
    emit_stagewise_progress(
        &mut progress,
        StagewiseProgress {
            event: StagewiseEventKind::TerminalEvidenceCompleted,
            birth_round,
            backfit_sweep: backfit_ev_trace.len(),
            candidate: None,
            accepted: Some(true),
            checkpoint: true,
            k_atoms: term.k_atoms(),
            births_accepted,
            births_rejected,
            ev: Some(prev_ev),
            factor_energy: None,
            joint_penalized_laml_before: None,
            joint_penalized_laml_after: Some(terminal_joint_penalized_laml),
            terminal_joint_penalized_laml: Some(terminal_joint_penalized_laml),
            term: &term,
            rho: &rho,
        },
    )?;

    Ok(StagewiseResult {
        term,
        rho,
        report: StagewiseReport {
            births_accepted,
            births_rejected,
            birth_records,
            ev_trace,
            backfit_ev_trace,
            stopped_reason,
            terminal_joint_penalized_laml,
            terminal_joint_loss,
        },
    })
}

// ═══════════════════════════════════════════════════════════════════════════
//  Batched birth-racing (THRPT) — PARALLEL candidate generation + acceptance
// ═══════════════════════════════════════════════════════════════════════════
//
// # Why this exists
//
// [`fit_stagewise`] discovers curved atoms SERIALLY: fit → harvest residual →
// race ONE seed → accept/reject → refit residual → repeat. Each birth is minutes
// on ~2 cores while ~30 cores idle, so the composed dictionary tops out at tens
// of curved atoms. The in-frame path made each curved fit a tiny INDEPENDENT
// problem (r ≈ 8–32 in-frame coords, per-region rows), so births in DIFFERENT
// residual regions are statistically independent given the current dictionary —
// nothing forces them through one serial loop.
//
// [`fit_stagewise_batched`] exploits that: from ONE residual snapshot it
// harvests MANY candidate seeds (the [`isa_deflationary_producer`] extracts every
// certifiable circle 2-plane from one joint split — exactly the seed set
// the serial loop would mine one-per-round), RACES their K=1 fits concurrently
// across cores (rayon), and accepts a DISJOINT batch in one round.
//
// # The statistical care — batch-greedy / matching-pursuit orthogonality
//
// Accepting many births from ONE residual snapshot risks double-booking the same
// variance twice (two candidates explaining overlapping structure). The clean,
// provable criterion is the batch-OMP block-orthogonality condition:
//
//   In greedy matching pursuit, selecting a BATCH of atoms in one iteration
//   equals selecting them SEQUENTIALLY iff the selected atoms' design blocks are
//   mutually orthogonal — the off-diagonal Gram block is zero — so the residual
//   update from accepting one leaves the others' fit and selection score
//   unchanged (Tropp 2004; Pati–Rezaiifar–Krishnaprasad OMP; the block/group
//   generalization is standard batch-greedy theory).
//
// The off-diagonal Gram block between two born atoms is a sum over ROWS and OUTPUT
// DIMS of the product of their (gated) reconstructions, so it vanishes when the two
// atoms are disjoint on EITHER axis — two independent routes to the sufficient
// orthogonality:
//
//   (1) DISJOINT GATE ROW SUPPORT under an INDEPENDENT gate (ordered Beta--Bernoulli / ThresholdGate):
//       a born atom's gate is exactly `0` on rows below its threshold, so its
//       weighted K=1 objective + gradient read ONLY its support rows. Two atoms on
//       disjoint rows never see each other's rows.
//
//   (2) DISJOINT OUTPUT-DIM SUPPORT: a born decoder writes only its own ambient
//       columns (a circle in an orthogonal 2-plane occupies just its cos/sin dims),
//       so its K=1 fit reads ONLY those residual columns. Two atoms in orthogonal
//       ambient planes are block-orthogonal even when they share EVERY row — the
//       dense-torus case, where the row test alone would wrongly force a serial
//       loop. The accepted set is pairwise block-orthogonal iff every pair is
//       disjoint on rows OR on output dims.
//
// Under either route,
//
//   fit_of(cand_j against the ORIGINAL residual R₀)
//     == fit_of(cand_j against R₁ = R₀ − Σ_{i<j} contribution_i)
//
// EXACTLY (not up to tolerance — the off-block entries drop out of the sum
// entirely), so the FIT and the per-birth reconstruction ΔEV charge are invariant
// to acceptance order. (The RAW joint-penalized LAML VALUE does not decompose additively —
// its Laplace term carries a globally-pooled dispersion that couples all rows — but
// the FIT, the accepted SET, and the ΔEV do, which is what parity needs.) This is
// the parity license (see `tests_batched_parity_*`): batched must equal serial
// statistically, or the speed is fake.
//
// Candidates that intersect an accepted atom on BOTH axes VIOLATE the orthogonality
// condition, so we accept only the BEST of such an overlapping group (lowest joint
// penalized LAML — the same keep-lowest tiebreak as the serial A-vs-B race) and REQUEUE the
// rest: they are dropped this round and re-mined next round against the UPDATED
// residual, recovering pure greedy on the conflicting part. No magic constant — the
// criterion is per-pair support disjointness on either axis, and the per-round
// fan-out is a safety BOUND (`max_candidates_per_round`), not a tuning knob.

/// Cost BOUND on the parallel candidate fan-out per batched birth round: the
/// ISA producer harvests at most this many certified planes from one jointly
/// rotated residual snapshot, and they are raced concurrently. A safety bound (like
/// [`StagewiseConfig::max_births`]), NOT a tuning knob — a larger value only
/// exposes more of the residual's already-present structure to one parallel pass.
#[derive(Clone, Copy, Debug)]
pub struct BatchedStagewiseConfig {
    /// The shared inner-fit + SAC dials (identical semantics to the serial path).
    pub base: StagewiseConfig,
    /// Upper bound on candidates harvested + raced per round.
    pub max_candidates_per_round: usize,
}

impl Default for BatchedStagewiseConfig {
    fn default() -> Self {
        Self {
            base: StagewiseConfig::default(),
            // Mirrors the serial `max_births` safety cap: a round never fans out
            // wider than the whole dictionary is allowed to grow.
            max_candidates_per_round: StagewiseConfig::default().max_births,
        }
    }
}

/// Per-round bookkeeping for the batched driver's honesty surface: how many
/// candidates one residual snapshot produced, how many cleared the birth gate,
/// how many were co-accepted (a disjoint batch), and how many passing candidates
/// were requeued because they conflicted with a better one this round.
#[derive(Clone, Copy, Debug)]
pub struct BatchRoundRecord {
    pub candidates_generated: usize,
    pub candidates_passing_gate: usize,
    pub co_accepted: usize,
    pub requeued_overlap: usize,
}

/// The batched SAC result: the composed dictionary, its ρ, the standard
/// [`StagewiseReport`], and the per-round batch ledger (for the births/round
/// multiplier).
#[derive(Clone, Debug)]
pub struct BatchedStagewiseResult {
    pub term: SaeManifoldTerm,
    pub rho: SaeManifoldRho,
    pub report: StagewiseReport,
    pub batch_records: Vec<BatchRoundRecord>,
}

/// A candidate that has been RACED: its born atom fully fit as a K=1 sub-problem,
/// its frozen joint evidence + EV measured as a `K+1` term, and its gate row
/// support recorded (the rows the batch-greedy disjointness test reads).
#[derive(Clone)]
struct RacedCandidate {
    born_atom: SaeManifoldAtom,
    born_coord: LatentCoordValues,
    born_logit_col: Vec<f64>,
    born_ard: Array1<f64>,
    born_log_lambda_smooth: f64,
    /// Seed-gate support rows (finite gate ⇒ the atom can be active there). For a
    /// global shared-factor DC seed this is ALL rows, so it never co-accepts and
    /// the round reduces to exactly one serial birth.
    support: Vec<usize>,
    /// Output (ambient) dimensions the born decoder occupies: the columns `j` whose
    /// decoder energy `Σ_row β[row,j]²` clears a relative floor. A born atom writes
    /// ONLY these columns, so its K=1 fit reads ONLY these residual columns — the
    /// SECOND route to the batch-OMP block-orthogonality license (a candidate on
    /// disjoint output dims is block-orthogonal to the accepted set even when it
    /// SHARES rows, e.g. a dense torus whose circles live in orthogonal ambient
    /// planes). Co-acceptance holds when EITHER the rows OR the output dims are
    /// disjoint; the row-only test alone cannot co-accept a dense multi-circle image.
    out_support: Vec<usize>,
    penalized_laml: f64,
    ev: f64,
    energy: f64,
}

/// Lift one ISA-certified circle plane to a [`BirthSeed`] (harmonic decoder on the
/// cos/sin rows + phase-aligned chart + own-presence gate) — the same seed the
/// serial `residual_principal_birth_candidate` circle path builds, so a raced
/// plane is byte-identical to the serial per-round pick.
fn plane_to_birth_seed(term: &SaeManifoldTerm, cand: &IsaPlaneCandidate) -> BirthSeed {
    let m = term.atoms[0].basis_size();
    let p = term.output_dim();
    let mut decoder = Array2::<f64>::zeros((m, p));
    for j in 0..p {
        decoder[[1, j]] = cand.amplitudes[0] * cand.basis[[j, 0]];
        decoder[[2, j]] = cand.amplitudes[1] * cand.basis[[j, 1]];
    }
    let energy = cand.amplitudes[0].powi(2) + cand.amplitudes[1].powi(2);
    BirthSeed {
        decoder,
        energy,
        circle_coords: Some(cand.phases_turns.clone()),
        circle_gate: Some(cand.gate_logits.clone()),
    }
}

/// Race ONE birth seed: build the born atom (circle-direct for a rank-2 plane
/// seed, else the topology-race `Birth`), fit it as a K=1 sub-problem against the
/// residual, and measure its frozen joint evidence + EV as a `K+1` term — exactly
/// the serial candidate-A construction, refactored so a batch of seeds can be
/// raced in parallel. Pure over `(term, rho, residual, target)`: it clones the
/// term internally and mutates nothing shared, so it is safe under `par_iter`.
fn race_birth_seed(
    term: &SaeManifoldTerm,
    rho: &SaeManifoldRho,
    seed: &BirthSeed,
    residual: ArrayView2<'_, f64>,
    target: ArrayView2<'_, f64>,
    registry: Option<&AnalyticPenaltyRegistry>,
    config: &StagewiseConfig,
) -> Result<RacedCandidate, String> {
    let k = term.k_atoms();
    let n = term.assignment.logits.nrows();
    let born_move = match &seed.circle_coords {
        Some(coords) => crate::structure_harvest::born_circle_atom(
            term,
            rho,
            seed.decoder.clone(),
            coords.clone(),
            seed.circle_gate.clone().unwrap_or_else(|| vec![0.0; n]),
        ),
        None => apply_structure_move(
            term,
            rho,
            &StructureMove::Birth { candidate: 0 },
            std::slice::from_ref(&seed.decoder),
        ),
    };
    let (mut cand_term, mut cand_rho) = born_move?;
    cand_term.set_guards_enabled(false);
    fit_single_atom_response_in_place(
        &mut cand_term,
        &mut cand_rho,
        k,
        residual,
        registry,
        config,
    )?;
    let (penalized_laml, _) =
        frozen_joint_penalized_laml(&mut cand_term, target, &cand_rho, registry, config)?;
    let ev = ev_of(&cand_term, target);
    // Support: a circle seed's own-presence gate marks the rows it can fire on
    // (finite gate); a shared-factor DC seed is global (all rows), which forces a
    // singleton accept (batched round == serial round on non-circle residuals).
    let support: Vec<usize> = match &seed.circle_gate {
        Some(g) => (0..n).filter(|&i| g[i].is_finite()).collect(),
        None => (0..n).collect(),
    };
    // Output-dim support: the columns the born decoder actually writes. A circle in
    // an orthogonal ambient plane occupies only its 2 (cos/sin) output dims, so a
    // dense torus's per-circle candidates are output-disjoint even though they share
    // every row — the second block-orthogonality route the acceptance test reads.
    let decoder = &cand_term.atoms[k].decoder_coefficients;
    let p_out = decoder.ncols();
    let mut col_energy = vec![0.0_f64; p_out];
    for row in 0..decoder.nrows() {
        for j in 0..p_out {
            col_energy[j] += decoder[[row, j]] * decoder[[row, j]];
        }
    }
    let total_energy: f64 = col_energy.iter().sum();
    // Relative floor: a column carrying < 1% of the decoder's energy is treated as
    // UNoccupied. Such a column contributes < 1% to any off-block Gram entry (which
    // scales with the product of the two atoms' column energies), so ignoring it
    // keeps the batch block-orthogonal to O(1%) — while excluding the finite-sample
    // cross-talk a least-squares decoder fit leaks into an orthogonal atom's dims
    // (~1/√n per row, which a 1e-6 floor would wrongly count as genuine occupancy
    // and so never co-accept two truly output-disjoint circles).
    let out_thresh = 1e-2 * total_energy;
    let out_support: Vec<usize> = (0..p_out).filter(|&j| col_energy[j] > out_thresh).collect();
    let born_logit_col: Vec<f64> = (0..n)
        .map(|r| cand_term.assignment.logits[[r, k]])
        .collect();
    Ok(RacedCandidate {
        born_atom: cand_term.atoms[k].clone(),
        born_coord: cand_term.assignment.coords[k].clone(),
        born_logit_col,
        born_ard: cand_rho.log_ard[k].clone(),
        born_log_lambda_smooth: cand_rho.log_lambda_smooth[k],
        support,
        out_support,
        penalized_laml,
        ev,
        energy: seed.energy,
    })
}

/// Append an ALREADY-FITTED born atom (its decoder/coords/logit column produced
/// by [`race_birth_seed`]) to `term` WITHOUT refitting — the batch-assembly
/// primitive. Mirrors the logit / coord / ρ growth `born_atom` performs, so an
/// atom raced against the `K`-atom term joins the running `K+1`-atom term
/// identically; under an independent gate (the batch-greedy precondition) the
/// pre-fit gate column stays exact because the gate does not renormalize across
/// atoms.
fn append_fitted_atom(
    term: &SaeManifoldTerm,
    rho: &SaeManifoldRho,
    atom: SaeManifoldAtom,
    coord: LatentCoordValues,
    logit_col: &[f64],
    ard: Array1<f64>,
    log_lambda_smooth: f64,
) -> Result<(SaeManifoldTerm, SaeManifoldRho), String> {
    let k = term.k_atoms();
    let n = term.assignment.logits.nrows();
    if logit_col.len() != n {
        return Err(format!(
            "append_fitted_atom: logit column length {} != n_obs {n}",
            logit_col.len()
        ));
    }
    let mut atoms = term.atoms.clone();
    atoms.push(atom);
    let mut logits = Array2::<f64>::zeros((n, k + 1));
    for row in 0..n {
        for col in 0..k {
            logits[[row, col]] = term.assignment.logits[[row, col]];
        }
        logits[[row, k]] = logit_col[row];
    }
    let mut coords = term.assignment.coords.clone();
    coords.push(coord);
    let assignment = SaeAssignment::with_mode(logits, coords, term.assignment.mode)?;
    let child = SaeManifoldTerm::new(atoms, assignment)?;
    let mut child_rho = rho.clone();
    child_rho.log_ard.push(ard);
    child_rho.log_lambda_smooth.push(log_lambda_smooth);
    Ok((child, child_rho))
}

/// Batch-greedy disjoint selection: from the gate-passing candidate indices in
/// best-first `order`, return the maximal prefix-greedy subset that is pairwise
/// BLOCK-ORTHOGONAL — every accepted pair disjoint on ROWS or on OUTPUT DIMS (the
/// two routes documented in the module note; a born atom reads only its own rows
/// AND writes only its own output columns, so disjointness on either axis zeroes
/// the cross Gram block) — capped at `max_accept`, plus the count requeued for a
/// conflict or the cap. Extracted so the co-acceptance criterion is unit-testable
/// apart from the residual-mining + append machinery.
fn select_disjoint_batch(
    raced: &[RacedCandidate],
    order: &[usize],
    max_accept: usize,
) -> (Vec<usize>, usize) {
    let intersects = |a: &[usize], b: &[usize]| -> bool {
        let (small, large) = if a.len() <= b.len() { (a, b) } else { (b, a) };
        let set: std::collections::HashSet<usize> = large.iter().copied().collect();
        small.iter().any(|i| set.contains(i))
    };
    let mut accepted: Vec<usize> = Vec::new();
    let mut accepted_supports: Vec<(&[usize], &[usize])> = Vec::new();
    let mut requeued = 0usize;
    for &idx in order {
        if accepted.len() >= max_accept {
            requeued += 1;
            continue;
        }
        let c = &raced[idx];
        let conflict = accepted_supports
            .iter()
            .any(|(ar, ad)| intersects(&c.support, ar) && intersects(&c.out_support, ad));
        if conflict {
            requeued += 1;
            continue;
        }
        accepted_supports.push((&c.support, &c.out_support));
        accepted.push(idx);
    }
    (accepted, requeued)
}

/// Batched forward-birth SAC: the parallel-racing counterpart of [`fit_stagewise`]
/// Phase 1. Each round harvests MANY candidate seeds from one residual snapshot
/// (the [`isa_deflationary_producer`] joint split, i.e. the seeds the serial
/// loop would mine one-per-round), RACES their K=1 fits concurrently, and
/// accepts a maximal DISJOINT-support batch (the batch-OMP orthogonality
/// criterion documented above). Overlapping passing candidates are requeued to
/// the next round's residual. Phases 2 (backfitting) and 3 (terminal evidence)
/// are then run identically to the serial driver, so the returned artifact is the
/// same shape.
///
/// PARITY: on planted multi-structure data whose atoms occupy disjoint gate
/// supports under an independent gate, the accepted atom set, per-birth charges,
/// and terminal joint evidence equal [`fit_stagewise`]'s up to order. When no
/// plane certifies (a non-circle residual), the round harvests a single
/// shared-factor / principal seed with GLOBAL support, so it accepts at most one
/// atom — the batched driver then reduces to the serial loop exactly.
pub fn fit_stagewise_batched(
    seed: SaeManifoldTerm,
    mut rho: SaeManifoldRho,
    target: ArrayView2<'_, f64>,
    registry: Option<&AnalyticPenaltyRegistry>,
    sample_weights: Option<&[f64]>,
    config: &BatchedStagewiseConfig,
) -> Result<BatchedStagewiseResult, String> {
    use rayon::prelude::*;

    let n = target.nrows();
    if seed.k_atoms() != 1 {
        return Err(format!(
            "fit_stagewise_batched: seed must be a single-atom (K=1) term; got K={}",
            seed.k_atoms()
        ));
    }
    if seed.n_obs() != n {
        return Err(format!(
            "fit_stagewise_batched: seed n_obs {} != target rows {n}",
            seed.n_obs()
        ));
    }
    let base = &config.base;
    let mut term = seed;
    term.set_guards_enabled(false);
    if let Some(w) = sample_weights {
        if w.len() != n {
            return Err(format!(
                "fit_stagewise_batched: sample_weights length {} != target rows {n}",
                w.len()
            ));
        }
        term.set_row_loss_weights(w.to_vec())?;
    }

    let mut ev_trace = vec![ev_of(&term, target)];
    let mut birth_records: Vec<BirthRecord> = Vec::new();
    let mut batch_records: Vec<BatchRoundRecord> = Vec::new();
    let mut births_accepted = 0usize;
    let mut births_rejected = 0usize;
    let mut consecutive_reject_rounds = 0usize;

    // ── Phase 1 — batched forward births ───────────────────────────────────────
    let stopped_reason = loop {
        if births_accepted >= base.max_births {
            break StagewiseStop::MaxBirths;
        }
        if consecutive_reject_rounds >= 2 {
            break StagewiseStop::TwoConsecutiveRejections;
        }
        // #P3 — stratum-local mining (see `birth_mining_residual`): the batched round
        // harvests its candidate planes from the SAME floor-cleared residual the
        // serial loop mines, so a diffuse tail structure below the pooled floor is
        // reachable here too. Pooled residual unchanged when it already routes.
        let mining_residual = birth_mining_residual(&term, target, base)?;
        let Some((residual, model)) = fit_residual_covariance_on(&term, mining_residual, base)?
        else {
            break StagewiseStop::NoResidualStructure;
        };
        if base.structured_whitening {
            term.set_row_metric(model.row_metric(n)?)?;
        }

        // Batched candidate GENERATION from one residual snapshot: harvest every
        // certifiable circle 2-plane (the serial per-round seed sequence, all at
        // once). When nothing certifies, fall back to a single shared-factor /
        // principal seed with global support (a serial-equivalent singleton round).
        let harvest = isa_deflationary_producer(
            residual.view(),
            config.max_candidates_per_round,
            &IsaSeedConfig::default(),
        )?;
        let mut seeds: Vec<BirthSeed> = harvest
            .planes
            .iter()
            .map(|c| plane_to_birth_seed(&term, c))
            .collect();
        if seeds.is_empty() {
            let Some(fallback) = top_factor_birth_decoder(&term, &model, residual.view())
                .or_else(|| residual_principal_birth_candidate(&term, residual.view()))
            else {
                break StagewiseStop::NoResidualStructure;
            };
            seeds.push(fallback);
        }
        let candidates_generated = seeds.len();

        // Current joint evidence + EV (the birth gate's reference), computed once
        // before the parallel race so the closures borrow `term` immutably.
        let (current_penalized_laml, _) = frozen_joint_penalized_laml(&mut term, target, &rho, registry, base)?;
        let cur_ev = ev_of(&term, target);

        // ── PARALLEL racing: fit every candidate's K=1 sub-problem concurrently ──
        // Each `race_birth_seed` clones `term` internally and mutates nothing
        // shared; a candidate whose fit errors is dropped (mirrors the serial
        // `.ok()`), never aborting the round.
        let raced: Vec<RacedCandidate> = seeds
            .par_iter()
            .filter_map(|s| {
                race_birth_seed(&term, &rho, s, residual.view(), target, registry, base).ok()
            })
            .collect();

        // Birth gate: strictly-improved joint evidence AND ΔEV ≥ the minimum-effect
        // floor — identical to the serial `passes` predicate.
        let passes = |c: &RacedCandidate| -> bool {
            c.penalized_laml.is_finite()
                && c.penalized_laml < current_penalized_laml
                && c.ev.is_finite()
                && (c.ev - cur_ev) >= base.min_effect_ev
        };
        let mut order: Vec<usize> = (0..raced.len()).filter(|&i| passes(&raced[i])).collect();
        let candidates_passing_gate = order.len();
        // Best (lowest joint penalized LAML) first — the same keep-lowest tiebreak the serial
        // A-vs-B race uses; ties fall to the earlier (higher-energy) harvest index.
        order.sort_by(|&a, &b| {
            raced[a]
                .penalized_laml
                .partial_cmp(&raced[b].penalized_laml)
                .unwrap_or(std::cmp::Ordering::Equal)
                .then(a.cmp(&b))
        });

        // ── DISJOINT batch-greedy acceptance ───────────────────────────────────
        // Accept, in best-first order, a maximal pairwise BLOCK-ORTHOGONAL subset —
        // every accepted pair disjoint on ROWS or on OUTPUT DIMS (see
        // `select_disjoint_batch` + the module note) — capped at the remaining birth
        // budget. Each accepted atom is appended by its already-raced fit.
        let remaining = base.max_births.saturating_sub(births_accepted);
        let (accepted_idx, requeued_overlap) = select_disjoint_batch(&raced, &order, remaining);
        let mut co_accepted = 0usize;
        for &idx in &accepted_idx {
            let c = &raced[idx];
            let (next_term, next_rho) = append_fitted_atom(
                &term,
                &rho,
                c.born_atom.clone(),
                c.born_coord.clone(),
                &c.born_logit_col,
                c.born_ard.clone(),
                c.born_log_lambda_smooth,
            )?;
            term = next_term;
            rho = next_rho;
            births_accepted += 1;
            co_accepted += 1;
            let running_ev = ev_of(&term, target);
            birth_records.push(BirthRecord {
                kind: BirthKind::NewAtom,
                // Charge this birth exactly what the serial loop would: its own
                // K+1 joint-evidence delta over the round's reference (additive
                // across the disjoint batch).
                delta_ev: c.ev - cur_ev,
                factor_energy: c.energy,
                joint_penalized_laml_before: current_penalized_laml,
                joint_penalized_laml_after: c.penalized_laml,
                accepted: true,
            });
            ev_trace.push(running_ev);
        }

        batch_records.push(BatchRoundRecord {
            candidates_generated,
            candidates_passing_gate,
            co_accepted,
            requeued_overlap,
        });

        if co_accepted == 0 {
            births_rejected += 1;
            consecutive_reject_rounds += 1;
            birth_records.push(BirthRecord {
                kind: BirthKind::NewAtom,
                delta_ev: 0.0,
                factor_energy: raced.first().map(|c| c.energy).unwrap_or(0.0),
                joint_penalized_laml_before: current_penalized_laml,
                joint_penalized_laml_after: current_penalized_laml,
                accepted: false,
            });
        } else {
            consecutive_reject_rounds = 0;
        }
        if births_accepted >= base.max_births {
            break StagewiseStop::MaxBirths;
        }
    };

    // ── Phase 2 — backfitting sweeps (keep-best, monotone by construction) ──────
    let mut prev_ev = *ev_trace.last().unwrap_or(&f64::NEG_INFINITY);
    for _sweep in 0..base.max_backfit_sweeps {
        let term_snapshot = term.clone();
        let rho_snapshot = rho.clone();
        backfit_sweep(&mut term, &mut rho, target, registry, base)?;
        let ev = ev_of(&term, target);
        if ev > prev_ev {
            prev_ev = ev;
        } else {
            term = term_snapshot;
            rho = rho_snapshot;
            break;
        }
    }

    // ── Phase 3 — terminal frozen joint evidence of the composed tier ───────────
    refresh_terminal_row_metric(&mut term, target, base)?;
    let (terminal_joint_penalized_laml, terminal_joint_loss) =
        frozen_joint_penalized_laml(&mut term, target, &rho, registry, base)?;
    term.set_guards_enabled(true);

    Ok(BatchedStagewiseResult {
        term,
        rho,
        report: StagewiseReport {
            births_accepted,
            births_rejected,
            birth_records,
            ev_trace,
            backfit_ev_trace: Vec::new(),
            stopped_reason,
            terminal_joint_penalized_laml,
            terminal_joint_loss,
        },
        batch_records,
    })
}

/// Phase 3 — terminal joint assembly. Merge a Tier-1 bulk term (`primary`) with
/// the SAC-composed curved tier (`secondary`) via [`SaeManifoldTerm::merge_tiers`]
/// and run a SINGLE frozen (evaluate-don't-optimize, `inner_max_iter == 0`)
/// arrow-Schur pass over the merged dictionary to read its joint Laplace evidence
/// WITHOUT moving β. Returns the merged term + ρ and the frozen joint evidence.
///
/// Nothing the simultaneous joint fit uniquely provided is lost: simultaneous
/// credit assignment was recovered by the backfitting sweeps, joint evidence by
/// this terminal pass. If the joint Hessian is non-PD here, that is information
/// (residual gauge), surfaced as the criterion — resolved by the gauge quotient
/// (Workstream B), never by a mid-flight barrier.
pub fn terminal_joint_assembly(
    primary: SaeManifoldTerm,
    primary_rho: &SaeManifoldRho,
    secondary: SaeManifoldTerm,
    secondary_rho: &SaeManifoldRho,
    target: ArrayView2<'_, f64>,
    registry: Option<&AnalyticPenaltyRegistry>,
    config: &StagewiseConfig,
) -> Result<(SaeManifoldTerm, SaeManifoldRho, f64, SaeManifoldLoss), String> {
    let (mut merged, merged_rho) =
        SaeManifoldTerm::merge_tiers(primary, primary_rho, secondary, secondary_rho)?;
    // Disarm the reseed guards ONLY for the frozen (evaluate-don't-optimize) pass,
    // then RE-ARM before returning: guards-off is the K=1 / backfitting rationale,
    // but the composed artifact must ship with the collapse-guard stack armed so
    // any downstream non-frozen refit / FFI consumer of the returned dictionary
    // gets the normal supervision (a disarmed term would silently skip it).
    merged.set_guards_enabled(false);
    let (penalized_laml, loss) =
        frozen_joint_penalized_laml(&mut merged, target, &merged_rho, registry, config)?;
    merged.set_guards_enabled(true);
    Ok((merged, merged_rho, penalized_laml, loss))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::manifold::{
        AssignmentMode, PeriodicHarmonicEvaluator, SaeAssignment, SaeAtomBasisKind,
        SaeBasisEvaluator, SaeManifoldAtom,
    };

    /// Unscreened pooled-residual covariance oracle: `Σ` fit on the full
    /// `R = target − fitted` with no stratum-local floor screen. Production
    /// routes through [`birth_mining_residual`]; the stratum-local tests use
    /// this as the comparison oracle.
    fn fit_residual_covariance(
        term: &SaeManifoldTerm,
        target: ArrayView2<'_, f64>,
        config: &StagewiseConfig,
    ) -> Result<Option<(Array2<f64>, StructuredResidualModel)>, String> {
        let residual = current_residual(term, target)?;
        fit_residual_covariance_on(term, residual, config)
    }
    use gam_terms::latent::LatentManifold;
    use ndarray::Array2;
    use std::sync::Arc;

    const ON: f64 = 6.0;
    const OFF: f64 = -6.0;

    /// A K=1 test config: tiny inner budgets so the real Arrow-Schur fits stay
    /// fast on the laptop, evidence-only acceptance (`min_effect_ev = 0`), no
    /// whitening (keeps the tiny synthetic isotropic and the fits cheap).
    fn test_config() -> StagewiseConfig {
        StagewiseConfig {
            inner_max_iter: 24,
            learning_rate: 1.0,
            ridge_ext_coord: 1e-6,
            ridge_beta: 1e-6,
            max_births: 3,
            max_backfit_sweeps: 2,
            min_effect_ev: 0.0,
            max_factor_rank: 3,
            structured_whitening: false,
        }
    }

    /// One circle atom over the shared row-fraction coordinate, decoder seeded so
    /// its reconstruction is a distinct direction in output space.
    fn circle_atom(
        name: &str,
        evaluator: &Arc<PeriodicHarmonicEvaluator>,
        coords: &Array2<f64>,
        dir_a: usize,
        dir_b: usize,
        p: usize,
    ) -> (SaeManifoldAtom, Array2<f64>) {
        let (phi, jet) = evaluator.evaluate(coords.view()).unwrap();
        let mut decoder = Array2::<f64>::zeros((3, p));
        decoder[[1, dir_a % p]] = 1.0;
        decoder[[2, dir_b % p]] = 1.0;
        let atom = SaeManifoldAtom::new_with_provided_function_gram(
            name.to_string(),
            SaeAtomBasisKind::Periodic,
            1,
            phi,
            jet,
            decoder,
            Array2::<f64>::eye(3),
        )
        .unwrap()
        .with_basis_second_jet(evaluator.clone());
        (atom, coords.clone())
    }

    /// Build a K-atom periodic softmax term from an ON/OFF routing table.
    fn build_term(
        atoms: Vec<SaeManifoldAtom>,
        coord_blocks: Vec<Array2<f64>>,
        active: &[Vec<bool>],
    ) -> (SaeManifoldTerm, SaeManifoldRho) {
        let n = active.len();
        let k = atoms.len();
        let mut logits = Array2::<f64>::zeros((n, k));
        for (row, atom_active) in active.iter().enumerate() {
            for (atom, &on) in atom_active.iter().enumerate() {
                logits[[row, atom]] = if on { ON } else { OFF };
            }
        }
        let assignment = SaeAssignment::from_blocks_with_mode_and_manifolds(
            logits,
            coord_blocks,
            vec![LatentManifold::Circle { period: 1.0 }; k],
            AssignmentMode::softmax(1.0),
        )
        .unwrap();
        let term = SaeManifoldTerm::new(atoms, assignment).unwrap();
        let rho = SaeManifoldRho::new(0.0, 0.0, vec![Array1::<f64>::zeros(1); k]);
        (term, rho)
    }

    fn fitted_seed(
        mut seed: SaeManifoldTerm,
        mut rho: SaeManifoldRho,
        target: ArrayView2<'_, f64>,
        config: &StagewiseConfig,
    ) -> (SaeManifoldTerm, SaeManifoldRho) {
        seed.set_guards_enabled(false);
        seed.run_joint_fit_arrow_schur(
            target,
            &mut rho,
            None,
            config.inner_max_iter,
            config.learning_rate,
            config.ridge_ext_coord,
            config.ridge_beta,
        )
        .expect("test seed K=1 fit must complete before stagewise entry");
        (seed, rho)
    }

    fn is_non_decreasing(xs: &[f64]) -> bool {
        // Non-decreasing within a small relative slack that absorbs the
        // line-search's terminal rounding — the guarantee is monotone descent, and
        // a strict `>=` on floating EV can trip on a sub-ulp wobble at the optimum.
        xs.windows(2).all(|w| {
            let tol = 1e-9 * (1.0 + w[0].abs());
            w[1] >= w[0] - tol
        })
    }

    /// A structured-residual fit error is a data/solver contract violation, not
    /// "no residual structure".  The stagewise driver may stop cleanly on an
    /// empty/single-channel residual, but it must not silently swallow a
    /// non-finite residual and continue as if the residual producer merely found
    /// nothing useful.
    #[test]
    fn residual_covariance_propagates_invalid_residual_errors() {
        let n = 8usize;
        let p = 2usize;
        let evaluator = Arc::new(PeriodicHarmonicEvaluator::new(3).unwrap());
        let coords = Array2::<f64>::from_shape_fn((n, 1), |(row, _)| row as f64 / n as f64);
        let (atom0, cb0) = circle_atom("seed", &evaluator, &coords, 0, 1, p);
        let (term, _rho) = build_term(vec![atom0], vec![cb0], &vec![vec![true]; n]);
        let mut target = Array2::<f64>::zeros((n, p));
        target[[3, 1]] = f64::NAN;

        let err = fit_residual_covariance(&term, target.view(), &test_config())
            .expect_err("non-finite residuals must be reported, not downgraded to None");
        assert!(
            err.contains("structured residual fit failed")
                && err.contains("residuals must be finite"),
            "unexpected error: {err}"
        );
    }

    /// Planted two-circles: the target is a genuine two-atom dictionary image, but
    /// the seed is a single circle atom. SAC's forward births must NOT lose EV —
    /// `ev_trace` is non-decreasing by construction — and the driver must complete
    /// with a finite terminal joint evidence, growing K when a birth clears the
    /// evidence gate.
    #[test]
    fn stagewise_recovers_planted_two_circles_ev_monotone() {
        let n = 48usize;
        let p = 4usize;
        let evaluator = Arc::new(PeriodicHarmonicEvaluator::new(3).unwrap());
        let coords = Array2::<f64>::from_shape_fn((n, 1), |(row, _)| row as f64 / n as f64);
        let (atom0, cb0) = circle_atom("t0", &evaluator, &coords, 0, 1, p);
        let (atom1, cb1) = circle_atom("t1", &evaluator, &coords, 2, 3, p);
        // Truth: two atoms, first active on the top half, second on the bottom.
        let active_truth: Vec<Vec<bool>> = (0..n).map(|r| vec![r < n / 2, r >= n / 2]).collect();
        let (truth, _truth_rho) = build_term(
            vec![atom0.clone(), atom1.clone()],
            vec![cb0.clone(), cb1.clone()],
            &active_truth,
        );
        let target = truth.fitted();

        // Seed: a single circle atom, active on every row.
        let config = test_config();
        let (seed, rho) = build_term(vec![atom0], vec![cb0], &vec![vec![true]; n]);
        let (seed, rho) = fitted_seed(seed, rho, target.view(), &config);
        let result = fit_stagewise(seed, rho, target.view(), None, None, &config, None, None)
            .expect("fit_stagewise must complete on planted two-circles");

        assert!(
            is_non_decreasing(&result.report.ev_trace),
            "EV must be monotone non-decreasing in births by construction; got {:?}",
            result.report.ev_trace
        );
        assert!(
            result.report.terminal_joint_penalized_laml.is_finite(),
            "terminal frozen joint penalized LAML must be finite"
        );
        // The final EV must be at least the seed EV (a birth can only be adopted if
        // it clears the ΔEV ≥ 0 floor).
        let seed_ev = result.report.ev_trace[0];
        let final_ev = *result.report.ev_trace.last().unwrap();
        assert!(
            final_ev >= seed_ev - 1e-9,
            "final EV {final_ev} must not fall below the seed EV {seed_ev}"
        );
        assert_eq!(
            result.term.k_atoms(),
            1 + result.report.births_accepted,
            "K must equal the seed atom plus the accepted new-atom births"
        );
    }

    /// Duplicate-atom rejection: when the seed already reconstructs the target, the
    /// residual carries no structured factor, so no birth is accepted — K stays 1
    /// and the phase stops on rejections or the empty-residual signal.
    #[test]
    fn duplicate_atom_birth_is_rejected() {
        let n = 40usize;
        let p = 4usize;
        let evaluator = Arc::new(PeriodicHarmonicEvaluator::new(3).unwrap());
        let coords = Array2::<f64>::from_shape_fn((n, 1), |(row, _)| row as f64 / n as f64);
        let (atom0, cb0) = circle_atom("t0", &evaluator, &coords, 0, 1, p);
        let (truth, _rho) =
            build_term(vec![atom0.clone()], vec![cb0.clone()], &vec![vec![true]; n]);
        let target = truth.fitted();

        // Exercise the explicit salience dial: a birth must add ≥ 1% EV. A target
        // already reconstructed by the seed leaves no residual clearing that floor,
        // so every birth is rejected (evidence gate ∪ minimum-effect floor).
        let config = StagewiseConfig {
            min_effect_ev: 0.01,
            ..test_config()
        };
        let (seed, rho) = build_term(vec![atom0], vec![cb0], &vec![vec![true]; n]);
        let (seed, rho) = fitted_seed(seed, rho, target.view(), &config);
        let result = fit_stagewise(seed, rho, target.view(), None, None, &config, None, None)
            .expect("fit_stagewise must complete on a fully-explained target");

        assert_eq!(
            result.report.births_accepted, 0,
            "a duplicate/empty residual must yield no accepted births"
        );
        assert_eq!(
            result.term.k_atoms(),
            1,
            "K must stay at the single seed atom"
        );
        assert!(
            is_non_decreasing(&result.report.ev_trace),
            "EV trace must remain monotone"
        );
    }

    /// #2080 — anchor-scored birth selection must prefer the SINGLY-ATTRIBUTABLE
    /// residual factor (supported on rows the existing dictionary leaves UNCONTESTED)
    /// over the dominant-VARIANCE factor, under an ordered Beta--Bernoulli routing whose per-row mass
    /// varies. Also pins the fallback: a UNIFORM routing (no anchor contrast) keeps
    /// the historical dominant-energy (column-0) pick.
    #[test]
    fn anchor_scored_birth_prefers_uncontested_factor_2080() {
        use gam_solve::inference::residual_factor::{ResidualFactorInput, StructuredResidualModel};
        let n = 120usize;
        let p = 6usize;
        let h = n / 2; // rows [0,h) contested (existing atom active), [h,n) anchor.
        // Residual: two rank-1 FACTOR directions (shared, correlated across two
        // channels each — the structured model captures off-diagonal correlation, so
        // a single independent channel would read as pure diagonal noise). A STRONG
        // factor dA (channels 0,1) lives on the CONTESTED rows; a WEAKER factor dB
        // (channels 2,3) lives on the ANCHOR rows.
        let inv_sqrt2 = 1.0 / 2.0_f64.sqrt();
        let d_a = [inv_sqrt2, inv_sqrt2, 0.0, 0.0, 0.0, 0.0];
        let d_b = [0.0, 0.0, inv_sqrt2, inv_sqrt2, 0.0, 0.0];
        let mut residual = Array2::<f64>::zeros((n, p));
        for i in 0..n {
            let s = (std::f64::consts::TAU * i as f64 / 11.0).cos(); // zero-mean wiggle
            let (dir, amp) = if i < h { (&d_a, 3.0) } else { (&d_b, 2.0) };
            for j in 0..p {
                residual[[i, j]] = amp * s * dir[j];
                // Small idiosyncratic noise on every channel for a well-posed D.
                residual[[i, j]] += 0.04 * ((i * 7 + j * 13) as f64).sin();
            }
        }
        let uniform_act = Array1::<f64>::ones(n);
        let model = StructuredResidualModel::fit(ResidualFactorInput {
            residuals: residual.view(),
            activity: uniform_act.view(),
            max_factor_rank: 2,
        })
        .unwrap();
        assert!(model.factor_rank() >= 2, "need both planted factors");

        // Seed atom over the row-fraction coordinate.
        let evaluator = Arc::new(PeriodicHarmonicEvaluator::new(3).unwrap());
        let coords = Array2::<f64>::from_shape_fn((n, 1), |(row, _)| row as f64 / n as f64);
        let (atom0, cb0) = circle_atom("t0", &evaluator, &coords, 0, 1, p);

        // Helper: build a 1-atom ordered Beta--Bernoulli term from a per-row logit column.
        let build_ibp = |logit: &dyn Fn(usize) -> f64| -> SaeManifoldTerm {
            let mut logits = Array2::<f64>::zeros((n, 1));
            for row in 0..n {
                logits[[row, 0]] = logit(row);
            }
            let assignment = SaeAssignment::from_blocks_with_mode_and_manifolds(
                logits,
                vec![cb0.clone()],
                vec![LatentManifold::Circle { period: 1.0 }],
                AssignmentMode::ordered_beta_bernoulli(1.0, 1.0, false),
            )
            .unwrap();
            SaeManifoldTerm::new(vec![atom0.clone()], assignment).unwrap()
        };

        // CONTESTED-vs-ANCHOR routing: existing atom ACTIVE on [0,h) (high logit),
        // INACTIVE on [h,n) (low logit) — so [h,n) are the uncontested anchor rows.
        let contrast_term = build_ibp(&|row| if row < h { 3.0 } else { -3.0 });
        let act = activity_of(&contrast_term);
        assert!(
            act[0] > act[n - 1] + 1e-6,
            "ordered Beta--Bernoulli activity must be higher on contested rows (got {} vs {})",
            act[0],
            act[n - 1]
        );
        let decoder = top_factor_birth_decoder(&contrast_term, &model, residual.view())
            .unwrap()
            .decoder;
        // Chosen p-direction sits on the constant (row-0) basis row.
        let pick_strong = decoder[[0, 0]].hypot(decoder[[0, 1]]); // contested dA (0,1)
        let pick_anchor = decoder[[0, 2]].hypot(decoder[[0, 3]]); // uncontested dB (2,3)
        assert!(
            pick_anchor > pick_strong,
            "anchor-scored birth must pick the UNCONTESTED (dB, channels 2,3) factor, not the \
             dominant-variance (dA, channels 0,1) one: |dB|={pick_anchor:.4} |dA|={pick_strong:.4}"
        );

        // FALLBACK: uniform routing ⇒ no anchor contrast ⇒ dominant-energy column 0
        // (channel 0, the higher-variance planted factor).
        let uniform_term = build_ibp(&|_| 0.5);
        let decoder_u = top_factor_birth_decoder(&uniform_term, &model, residual.view())
            .unwrap()
            .decoder;
        let u_strong = decoder_u[[0, 0]].hypot(decoder_u[[0, 1]]);
        let u_anchor = decoder_u[[0, 2]].hypot(decoder_u[[0, 3]]);
        assert!(
            u_strong > u_anchor,
            "uniform routing must fall back to the dominant-energy factor (dA, channels 0,1): \
             |dA|={u_strong:.4} |dB|={u_anchor:.4}"
        );
    }

    /// #2080 — the disjoint-extraction fallback must FIRE on a block-diagonal
    /// (disjoint) residual the structured factor model reports as rank-0, and must
    /// REJECT pure noise (the derived Marchenko–Pastur floor is the stop criterion).
    #[test]
    fn residual_principal_fallback_fires_on_disjoint_not_noise_2080() {
        let n = 400usize;
        let p = 8usize;
        // 1-atom term (only basis_size + activity are read by the fallback).
        let evaluator = Arc::new(PeriodicHarmonicEvaluator::new(3).unwrap());
        let coords = Array2::<f64>::from_shape_fn((n, 1), |(row, _)| row as f64 / n as f64);
        let (atom0, cb0) = circle_atom("t0", &evaluator, &coords, 0, 1, p);
        let (term, _rho) = build_term(vec![atom0], vec![cb0], &vec![vec![true]; n]);

        let mut state = 0xC0FFEE_1234_5678_u64;
        let mut rng = || {
            state = state
                .wrapping_mul(6364136223846793005)
                .wrapping_add(1442695040888963407);
            ((state >> 33) as f64) / ((1u64 << 31) as f64) - 1.0
        };

        // BLOCK-DIAGONAL disjoint residual: two independent correlated signals on
        // channels {0,1} and {2,3} (zero cross-block correlation → the factor model
        // attributes it to D → rank-0), tiny isotropic noise everywhere.
        let mut residual = Array2::<f64>::zeros((n, p));
        for i in 0..n {
            let a = rng();
            let b = rng();
            residual[[i, 0]] = 2.0 * a;
            residual[[i, 1]] = 2.0 * a; // signal A: correlated 0,1
            residual[[i, 2]] = 1.5 * b;
            residual[[i, 3]] = 1.5 * b; // signal B: correlated 2,3
            for j in 0..p {
                residual[[i, j]] += 0.03 * rng();
            }
        }
        let seed = residual_principal_birth_candidate(&term, residual.view()).expect(
            "disjoint block-diagonal residual must yield a fallback candidate \
             (structure above the derived MP noise floor)",
        );
        let (decoder, energy) = (seed.decoder, seed.energy);
        assert!(energy > 0.0 && energy.is_finite());
        // Two UNEQUAL independent signals (var 4 vs 2.25) are NOT a circle — the
        // eigenvalue-degeneracy gate rejects the 2-plane, so this exercises the
        // rank-1 row-0 fallback (circle_coords None, direction on the constant row).
        assert!(
            seed.circle_coords.is_none(),
            "unequal independent signals must NOT be seeded as a circle"
        );
        // The chosen direction must be a real signal direction (mass on channels 0-3),
        // not a noise channel.
        let sig_mass: f64 = (0..4).map(|j| decoder[[0, j]].powi(2)).sum();
        let noise_mass: f64 = (4..p).map(|j| decoder[[0, j]].powi(2)).sum();
        assert!(
            sig_mass > noise_mass,
            "fallback birth direction must land on the signal block (0-3), not noise: \
             sig={sig_mass:.3e} noise={noise_mass:.3e}"
        );

        // PURE NOISE: no direction above the MP floor ⇒ None ⇒ stop growing.
        let mut noise = Array2::<f64>::zeros((n, p));
        for i in 0..n {
            for j in 0..p {
                noise[[i, j]] = rng();
            }
        }
        assert!(
            residual_principal_birth_candidate(&term, noise.view()).is_none(),
            "pure-noise residual must be below the derived MP floor ⇒ no candidate (stop)"
        );
    }

    /// #2101 RECOVERY GUARD — a genuine disjoint CIRCLE residual must be seeded as a
    /// rank-2 circle: the 2-plane on the cos/sin HARMONIC rows (NOT the DC row-0), a
    /// phase-aligned coordinate that SPANS and recovers the planted angle up to gauge.
    /// This is the birth-seed fix that breaks the DC stationary point (#2101); the
    /// old row-0 seed produced a constant (cos/sin dead) and this asserts against it.
    #[test]
    fn residual_principal_seeds_circle_as_rank2_not_dc_2101() {
        let n = 240usize;
        let p = 8usize;
        let evaluator = Arc::new(PeriodicHarmonicEvaluator::new(3).unwrap());
        let coords = Array2::<f64>::from_shape_fn((n, 1), |(row, _)| row as f64 / n as f64);
        let (atom0, cb0) = circle_atom("t0", &evaluator, &coords, 0, 1, p);
        let (term, _rho) = build_term(vec![atom0], vec![cb0], &vec![vec![true]; n]);

        // A real circle on channels (2,3): equal-variance cos/sin axes + tiny noise.
        let mut state = 0x5EED_2101_u64;
        let mut rng = || {
            state = state
                .wrapping_mul(6364136223846793005)
                .wrapping_add(1442695040888963407);
            ((state >> 33) as f64) / ((1u64 << 31) as f64)
        };
        let mut residual = Array2::<f64>::zeros((n, p));
        let mut planted = vec![0.0_f64; n];
        for i in 0..n {
            let theta = std::f64::consts::TAU * rng();
            planted[i] = theta;
            residual[[i, 2]] = theta.cos();
            residual[[i, 3]] = theta.sin();
            for j in 0..p {
                residual[[i, j]] += 0.02 * (rng() - 0.5);
            }
        }
        let seed = residual_principal_birth_candidate(&term, residual.view())
            .expect("a real circle residual must yield a birth candidate");
        let born_coords = seed.circle_coords.clone().expect(
            "a circle residual must be seeded as a rank-2 CIRCLE (circle_coords Some), \
             not a DC direction",
        );

        // Harmonic (cos/sin) rows carry the mass; the DC row-0 is ~0.
        let dc: f64 = (0..p)
            .map(|j| seed.decoder[[0, j]].powi(2))
            .sum::<f64>()
            .sqrt();
        let harm: f64 = (0..p)
            .map(|j| seed.decoder[[1, j]].powi(2) + seed.decoder[[2, j]].powi(2))
            .sum::<f64>()
            .sqrt();
        assert!(
            harm > 10.0 * dc.max(1e-9),
            "circle seed must put mass on the cos/sin rows, not the DC row: harm={harm:.3} dc={dc:.3}"
        );
        // The 2-plane must land on the planted channels (2,3), not elsewhere.
        let on_plane: f64 = [2usize, 3]
            .iter()
            .map(|&j| seed.decoder[[1, j]].powi(2) + seed.decoder[[2, j]].powi(2))
            .sum();
        let off_plane: f64 = (0..p)
            .filter(|&j| j != 2 && j != 3)
            .map(|j| seed.decoder[[1, j]].powi(2) + seed.decoder[[2, j]].powi(2))
            .sum();
        assert!(
            on_plane > off_plane,
            "circle seed 2-plane must land on the planted channels (2,3): on={on_plane:.3} off={off_plane:.3}"
        );

        // The phase-aligned coordinate SPANS (breaks the DC stationary point).
        let cmin = born_coords.iter().copied().fold(f64::INFINITY, f64::min);
        let cmax = born_coords
            .iter()
            .copied()
            .fold(f64::NEG_INFINITY, f64::max);
        assert!(
            cmax - cmin > 0.5,
            "seeded coordinate must span the circle (breaks the stationary point); range={:.3}",
            cmax - cmin
        );
        // ...and recovers the planted angle up to gauge (reflection ± + phase). Score
        // the best-aligned circular RMSE over both reflections.
        let mut best_rmse = f64::INFINITY;
        for &sign in &[1.0_f64, -1.0] {
            let (mut cs, mut sn) = (0.0_f64, 0.0_f64);
            for i in 0..n {
                let r = std::f64::consts::TAU * born_coords[[i, 0]] - sign * planted[i];
                cs += r.cos();
                sn += r.sin();
            }
            let phase = sn.atan2(cs);
            let mut sse = 0.0_f64;
            for i in 0..n {
                let mut e =
                    (std::f64::consts::TAU * born_coords[[i, 0]] - sign * planted[i] - phase)
                        .rem_euclid(std::f64::consts::TAU);
                if e > std::f64::consts::PI {
                    e -= std::f64::consts::TAU;
                }
                sse += e * e;
            }
            best_rmse = best_rmse.min((sse / n as f64).sqrt());
        }
        assert!(
            best_rmse < 0.15,
            "seeded coordinate must recover the planted circle phase up to gauge; \
             gauge-aligned circular RMSE = {best_rmse:.3} rad"
        );
    }

    /// #2111 κ-NULL CERTIFICATE — the born-circle producer must REJECT a blended 2-plane.
    /// Positive control: a clean single circle is seeded as a rank-2 circle (`circle_coords`
    /// Some). Null: TWO independent circles superimposed on the SAME output 2-plane form a
    /// genuine BLEND (`radius² = 2 + 2cos Δ` for independent angles ⇒ `κ = 3/2`, not a
    /// constant-radius circle) — the κ-null certificate must refuse to seed it as a clean
    /// circle and fall through to the rank-1 seed (`circle_coords` None). This is exactly the
    /// case a flat κ cutoff would miss (a two-circle blend sits at `κ = 5/4`, far below the
    /// CLT value 2) but the analytic-anchor midpoint gate catches.
    #[test]
    fn certificate_rejects_two_circle_blend_2111() {
        let n = 400usize;
        let p = 8usize;
        let evaluator = Arc::new(PeriodicHarmonicEvaluator::new(3).unwrap());
        let coords = Array2::<f64>::from_shape_fn((n, 1), |(row, _)| row as f64 / n as f64);
        let (atom0, cb0) = circle_atom("t0", &evaluator, &coords, 0, 1, p);
        let (term, _rho) = build_term(vec![atom0], vec![cb0], &vec![vec![true]; n]);

        let mut state = 0x2111_B1E4_u64;
        let mut rng = || {
            state = state
                .wrapping_mul(6364136223846793005)
                .wrapping_add(1442695040888963407);
            ((state >> 11) as f64) / ((1u64 << 53) as f64)
        };

        // Positive control: a clean single circle on channels (2, 3) — must seed a circle.
        let mut clean = Array2::<f64>::zeros((n, p));
        for i in 0..n {
            let th = std::f64::consts::TAU * rng();
            clean[[i, 2]] = th.cos();
            clean[[i, 3]] = th.sin();
            for j in 0..p {
                clean[[i, j]] += 0.02 * (rng() - 0.5);
            }
        }
        let clean_seed = residual_principal_birth_candidate(&term, clean.view())
            .expect("clean circle must yield a birth candidate");
        assert!(
            clean_seed.circle_coords.is_some(),
            "positive control: a clean single circle must be seeded as a rank-2 circle"
        );

        // Null: two INDEPENDENT circles on the SAME 2-plane (channels 0, 1). The plane's
        // radius is not constant (κ ≈ 3/2), so no clean circle lives in it.
        let mut blend = Array2::<f64>::zeros((n, p));
        for i in 0..n {
            let a = std::f64::consts::TAU * rng();
            let b = std::f64::consts::TAU * rng();
            blend[[i, 0]] = a.cos() + b.cos();
            blend[[i, 1]] = a.sin() + b.sin();
            for j in 0..p {
                blend[[i, j]] += 0.02 * (rng() - 0.5);
            }
        }
        let blend_seed = residual_principal_birth_candidate(&term, blend.view())
            .expect("blend residual still yields a (rank-1) birth candidate");
        assert!(
            blend_seed.circle_coords.is_none(),
            "κ-null certificate must REJECT the two-circle blend (κ≈1.5 > analytic-anchor \
             gate) and fall through to the rank-1 seed, not born it as a clean circle"
        );
    }

    /// #2111 κ-DEFLATION extraction on the DENSE torus — the load-bearing case (d > 2). A
    /// residual carrying ALL SIX circles (dense product-of-circles, the regime where
    /// eigenvector pairing returns a Davis–Kahan blend) must yield ONE CLEAN circle: the
    /// born 2-plane concentrates on a single true circle's channels (2c, 2c+1) — max
    /// energy-fraction ≫ the second circle's. This is the Rust mirror of the prototype's
    /// 6/6 @ overlap 0.999 at n ≥ 300 (#2111); n = 320 puts the 4th-order stats out of the
    /// small-sample floor.
    #[test]
    fn kappa_deflation_extracts_clean_circle_from_dense_torus_2111() {
        let n = 320usize;
        let p = 16usize;
        let ncirc = 6usize;
        let amps: Vec<f64> = (0..ncirc)
            .map(|c| 1.0 - 0.45 * (c as f64) / ((ncirc - 1) as f64))
            .collect();
        let evaluator = Arc::new(PeriodicHarmonicEvaluator::new(3).unwrap());
        let coords = Array2::<f64>::from_shape_fn((n, 1), |(row, _)| row as f64 / n as f64);
        let (atom0, cb0) = circle_atom("t0", &evaluator, &coords, 0, 1, p);
        let (term, _rho) = build_term(vec![atom0], vec![cb0], &vec![vec![true]; n]);

        let mut s = 0x2111_D0BE_u64;
        let mut rng = || {
            s = s
                .wrapping_mul(6364136223846793005)
                .wrapping_add(1442695040888963407);
            ((s >> 11) as f64) / ((1u64 << 53) as f64)
        };
        let mut residual = Array2::<f64>::zeros((n, p));
        for i in 0..n {
            for c in 0..ncirc {
                let th = std::f64::consts::TAU * rng();
                residual[[i, 2 * c]] += amps[c] * th.cos();
                residual[[i, 2 * c + 1]] += amps[c] * th.sin();
            }
            for j in 0..p {
                residual[[i, j]] += 0.05 * (rng() - 0.5);
            }
        }

        let seed = residual_principal_birth_candidate(&term, residual.view())
            .expect("dense torus must yield a birth candidate");
        let dec =
            seed.circle_coords.as_ref().map(|_| &seed.decoder).expect(
                "dense torus must be seeded as a CLEAN rank-2 circle (κ-deflation), not DC",
            );

        // Per-circle energy fraction of the born 2-plane (cos/sin rows on channels 2c,2c+1).
        let total: f64 = (0..p)
            .map(|j| dec[[1, j]].powi(2) + dec[[2, j]].powi(2))
            .sum();
        assert!(total > 0.0, "born plane must carry mass");
        let mut fracs: Vec<f64> = (0..ncirc)
            .map(|c| {
                let e = dec[[1, 2 * c]].powi(2)
                    + dec[[2, 2 * c]].powi(2)
                    + dec[[1, 2 * c + 1]].powi(2)
                    + dec[[2, 2 * c + 1]].powi(2);
                e / total
            })
            .collect();
        fracs.sort_by(|a, b| b.total_cmp(a));
        assert!(
            fracs[0] > 0.80 && fracs[1] < 0.20,
            "κ-deflation must isolate ONE clean circle from the dense torus: top channel-pair \
             energy fraction {:.3} (want > 0.80), second {:.3} (want < 0.20) — a blended plane \
             would spread across circles",
            fracs[0],
            fracs[1]
        );
    }

    /// #2109 — a born circle PRESENT on incumbent-SPARSE rows must SURVIVE. The
    /// #2101 fix routed the born gate at the incumbent per-row logit scale
    /// (`inc_max`), which is low/negative exactly where an incumbent-sparse circle
    /// lives, so the born circle re-collapses under ordered Beta--Bernoulli (the #3 starvation
    /// resurfacing at scale). The #2109 fix routes each present row at the STRONGER
    /// of `inc_max` and the born circle's OWN presence gate `ln(ρ_i²/2·λ₊)`, so the
    /// circle keeps a strong gate where the incumbents do not cover it. This test
    /// FAILS with the `inc_max`-only gate (the born logit on the circle's rows is the
    /// incumbent's very-negative `inc_max`, and the K=1 ordered Beta--Bernoulli sub-fit collapses ‖B‖ to
    /// ~1e-4) and PASSES with the own-presence gate.
    #[test]
    fn born_circle_survives_on_incumbent_sparse_rows_2109() {
        let n = 160usize;
        let p = 8usize;
        let h = n / 2; // rows [0,h): incumbent-active, NO circle. [h,n): the circle.
        let mut state = 0x2109_5A17_0000_0001u64;
        let mut rng = || {
            state = state
                .wrapping_mul(6364136223846793005)
                .wrapping_add(1442695040888963407);
            ((state >> 33) as f64) / ((1u64 << 31) as f64)
        };
        // Residual: a real circle on channels (0,1) with UNIFORM (deterministic) phase
        // so its cos/sin axes carry EXACTLY equal population variance (a robustly
        // DEGENERATE 2-plane the seed detector accepts), present ONLY on the incumbent-
        // SPARSE rows [h,n); rows [0,h) carry no circle, only tiny isotropic noise.
        let m = n - h;
        let mut residual = Array2::<f64>::zeros((n, p));
        for i in 0..n {
            if i >= h {
                let theta = std::f64::consts::TAU * ((i - h) as f64) / m as f64;
                residual[[i, 0]] = theta.cos();
                residual[[i, 1]] = theta.sin();
            }
            for j in 0..p {
                residual[[i, j]] += 0.02 * (rng() - 0.5);
            }
        }

        // Incumbent K=1 ordered Beta--Bernoulli term: one circle atom on channels (4,5), co-present on
        // [0,h) (high logit) and INACTIVE on [h,n) (very negative logit) — so `inc_max`
        // is deeply negative exactly where the born circle lives.
        let evaluator = Arc::new(PeriodicHarmonicEvaluator::new(3).unwrap());
        let coords = Array2::<f64>::from_shape_fn((n, 1), |(row, _)| row as f64 / n as f64);
        let (inc_atom, inc_cb) = circle_atom("inc", &evaluator, &coords, 4, 5, p);
        let mut inc_logits = Array2::<f64>::zeros((n, 1));
        for row in 0..n {
            inc_logits[[row, 0]] = if row < h { 4.0 } else { -6.0 };
        }
        let inc_assignment = SaeAssignment::from_blocks_with_mode_and_manifolds(
            inc_logits,
            vec![inc_cb],
            vec![LatentManifold::Circle { period: 1.0 }],
            AssignmentMode::ordered_beta_bernoulli(0.7, 1.0, false),
        )
        .unwrap();
        let mut term = SaeManifoldTerm::new(vec![inc_atom], inc_assignment).unwrap();
        term.set_guards_enabled(false);
        let rho = SaeManifoldRho::new(0.0, 0.0, vec![Array1::<f64>::zeros(1)]);

        // The disjoint principal path seeds the circle with the OWN-presence gate.
        let seed = residual_principal_birth_candidate(&term, residual.view())
            .expect("an incumbent-sparse circle must still yield a birth candidate");
        let born_coords = seed
            .circle_coords
            .clone()
            .expect("residual must be seeded as a rank-2 circle");
        let gate = seed
            .circle_gate
            .clone()
            .expect("a circle seed must carry the own-presence gate");
        // Presence is detected on the circle's rows [h,n) (finite gate), not [0,h).
        let present_on_circle = (h..n).filter(|&i| gate[i].is_finite()).count();
        let present_off_circle = (0..h).filter(|&i| gate[i].is_finite()).count();
        assert!(
            present_on_circle > (n - h) / 2 && present_off_circle < h / 4,
            "own-presence must fire on the circle's rows, not the empty ones: \
             on={present_on_circle}/{} off={present_off_circle}/{h}",
            n - h
        );

        // Build the born atom and read its seeded gate column BEFORE the sub-fit.
        let (child, mut child_rho) = crate::structure_harvest::born_circle_atom(
            &term,
            &rho,
            seed.decoder.clone(),
            born_coords,
            gate,
        )
        .expect("born_circle_atom");
        let born = child.k_atoms() - 1;
        // DETERMINISTIC gate guard: on the circle's rows the born logit must be routed
        // at the STRONG own-presence gate (≫0), NOT the incumbent's deeply-negative
        // inc_max (−6). This is the exact discriminator: `inc_max`-only would seed −6.
        let mut min_born_logit_on_circle = f64::INFINITY;
        for row in h..n {
            min_born_logit_on_circle =
                min_born_logit_on_circle.min(child.assignment.logits[[row, born]]);
        }
        assert!(
            min_born_logit_on_circle > 1.0,
            "born circle on incumbent-sparse rows must seed a STRONG own-presence gate \
             (>1), not the incumbent's negative inc_max (−6); got min={min_born_logit_on_circle:.3}"
        );

        // BEHAVIORAL guard: the K=1 ordered Beta--Bernoulli birth sub-fit must keep the born circle
        // ESTABLISHED — ‖B‖ stays O(1) rather than collapsing to ~1e-4.
        let config = StagewiseConfig {
            inner_max_iter: 40,
            learning_rate: 1.0,
            ridge_ext_coord: 1e-6,
            ridge_beta: 1e-6,
            max_births: 1,
            max_backfit_sweeps: 1,
            min_effect_ev: 0.0,
            max_factor_rank: 3,
            structured_whitening: false,
        };
        let mut child = child;
        fit_single_atom_response_in_place(
            &mut child,
            &mut child_rho,
            born,
            residual.view(),
            None,
            &config,
        )
        .expect("K=1 born-circle sub-fit must complete");
        let born_norm = child.atoms[born]
            .decoder_coefficients
            .iter()
            .map(|v| v * v)
            .sum::<f64>()
            .sqrt();
        assert!(
            born_norm.is_finite() && born_norm > 0.3,
            "born circle must SURVIVE the ordered Beta--Bernoulli sub-fit on incumbent-sparse rows \
             (‖B‖ O(1)); got ‖B‖={born_norm:.3e} (a collapse to ~1e-4 is the #2109 bug)"
        );
    }

    /// #2109 — the shared-factor / ENTANGLED birth path must MIRROR the #2101 rank-2
    /// circle seed. When the entangled residual (a genuine shared factor makes the
    /// model rank ≥ 1, so `top_factor_birth_decoder` is the primary path) ALSO carries
    /// a degenerate 2-plane circle, the born atom must be seeded ON that circle (cos/sin
    /// harmonic rows + phase coordinate + own-presence gate), not the flat row-0 DC seed
    /// that dies under ordered Beta--Bernoulli. A genuine rank-1 shared factor still gets the DC seed.
    #[test]
    fn top_factor_birth_mirrors_circle_seed_2109() {
        use gam_solve::inference::residual_factor::{ResidualFactorInput, StructuredResidualModel};
        let n = 200usize;
        let p = 8usize;
        // Residual = an ENTANGLED (shared-factor) circle: its cos axis loads on the
        // CORRELATED channel pair (0,1) and its sin axis on the CORRELATED pair (2,3),
        // so the structured factor model reads a genuine rank-2 SHARED factor (off-
        // diagonal correlation ⇒ `top_factor_birth_decoder` is the active path, the
        // entangled regime) — while the two equal-variance axes (uniform phase) form a
        // degenerate 2-plane the #2109 mirror must detect and seed as a circle, not the
        // flat row-0 DC factor seed that dies under ordered Beta--Bernoulli on incumbent-sparse rows.
        let mut residual = Array2::<f64>::zeros((n, p));
        for i in 0..n {
            let theta = std::f64::consts::TAU * (i as f64) / n as f64;
            let (c, s) = (theta.cos(), theta.sin());
            // The circle's cos axis loads on channels (0,1) and its sin axis on (2,3),
            // so each axis is a CORRELATED 2-channel direction (cov(0,1)=cov(2,3)=½ ≠ 0)
            // — the structured factor model reads the residual as a genuine rank-2
            // SHARED factor (`top_factor_birth_decoder` is the active path, the entangled
            // regime). Yet the two axes carry EQUAL variance (uniform phase ⇒ exactly
            // degenerate), so it is also a 2-plane circle the #2109 mirror must detect.
            residual[[i, 0]] = c;
            residual[[i, 1]] = c;
            residual[[i, 2]] = s;
            residual[[i, 3]] = s;
            for j in 0..p {
                residual[[i, j]] += 0.02 * ((i * 7 + j * 5) as f64).sin();
            }
        }

        let evaluator = Arc::new(PeriodicHarmonicEvaluator::new(3).unwrap());
        let coords = Array2::<f64>::from_shape_fn((n, 1), |(row, _)| row as f64 / n as f64);
        let (atom0, cb0) = circle_atom("t0", &evaluator, &coords, 0, 1, p);
        // Uniform routing (matches the proven factor-detection regime): the mirror fires
        // on the degenerate 2-plane before the anchor scoring, independent of contrast.
        let logits = Array2::<f64>::from_elem((n, 1), 0.5);
        let assignment = SaeAssignment::from_blocks_with_mode_and_manifolds(
            logits,
            vec![cb0],
            vec![LatentManifold::Circle { period: 1.0 }],
            AssignmentMode::ordered_beta_bernoulli(0.7, 1.0, false),
        )
        .unwrap();
        let term = SaeManifoldTerm::new(vec![atom0], assignment).unwrap();

        let activity = activity_of(&term);
        let model = StructuredResidualModel::fit(ResidualFactorInput {
            residuals: residual.view(),
            activity: activity.view(),
            max_factor_rank: 2,
        })
        .unwrap();
        assert!(
            model.factor_rank() >= 1,
            "the correlated cos/sin axes must make the factor model rank ≥ 1 so \
             top_factor_birth_decoder is the active path; got rank {}",
            model.factor_rank()
        );

        let seed = top_factor_birth_decoder(&term, &model, residual.view())
            .expect("the entangled path must yield a birth seed");
        // MIRROR: the entangled path now seeds a rank-2 CIRCLE, not a DC row.
        assert!(
            seed.circle_coords.is_some(),
            "top_factor_birth_decoder must MIRROR the #2101 circle seed on a degenerate \
             2-plane residual (circle_coords Some), not the flat DC seed"
        );
        let gate = seed
            .circle_gate
            .clone()
            .expect("the mirrored circle seed must carry the own-presence gate");
        assert!(
            gate.iter().filter(|g| g.is_finite()).count() > n / 2,
            "the mirrored circle must mark its present rows with a finite own-presence gate"
        );
        // The 2-plane must land on the planted circle channels (0,1,2,3), on the cos/sin
        // harmonic rows — the hallmark of the rank-2 seed vs the DC row-0 seed.
        let dc: f64 = (0..p)
            .map(|j| seed.decoder[[0, j]].powi(2))
            .sum::<f64>()
            .sqrt();
        let harm_on: f64 = [0usize, 1, 2, 3]
            .iter()
            .map(|&j| seed.decoder[[1, j]].powi(2) + seed.decoder[[2, j]].powi(2))
            .sum();
        let harm_off: f64 = (4..p)
            .map(|j| seed.decoder[[1, j]].powi(2) + seed.decoder[[2, j]].powi(2))
            .sum();
        assert!(
            harm_on > harm_off && harm_on.sqrt() > 10.0 * dc.max(1e-9),
            "mirrored circle seed must put its 2-plane on the cos/sin rows of channels \
             (0,1,2,3): harm_on={harm_on:.3} harm_off={harm_off:.3} dc={dc:.3}"
        );
    }

    #[test]
    fn progress_callback_emits_pre_birth_checkpoints() {
        let n = 32usize;
        let p = 4usize;
        let evaluator = Arc::new(PeriodicHarmonicEvaluator::new(3).unwrap());
        let coords = Array2::<f64>::from_shape_fn((n, 1), |(row, _)| row as f64 / n as f64);
        let (atom0, cb0) = circle_atom("t0", &evaluator, &coords, 0, 1, p);
        let (atom1, cb1) = circle_atom("t1", &evaluator, &coords, 2, 3, p);
        let active_truth: Vec<Vec<bool>> = (0..n).map(|r| vec![r < n / 2, r >= n / 2]).collect();
        let (truth, _rho) = build_term(
            vec![atom0.clone(), atom1],
            vec![cb0.clone(), cb1],
            &active_truth,
        );
        let target = truth.fitted();
        let config = StagewiseConfig {
            max_births: 1,
            max_backfit_sweeps: 0,
            ..test_config()
        };
        let (seed, rho) = build_term(vec![atom0], vec![cb0], &vec![vec![true]; n]);
        let (seed, rho) = fitted_seed(seed, rho, target.view(), &config);
        let mut events: Vec<(StagewiseEventKind, bool, usize, Option<BirthKind>)> = Vec::new();
        let mut progress = |event: StagewiseProgress<'_>| -> Result<(), String> {
            events.push((
                event.event,
                event.checkpoint,
                event.k_atoms,
                event.candidate,
            ));
            Ok(())
        };

        fit_stagewise(
            seed,
            rho,
            target.view(),
            None,
            None,
            &config,
            Some(&mut progress),
            None,
        )
        .expect("fit_stagewise must complete while emitting progress");

        assert_eq!(
            events.first().map(|event| event.0),
            Some(StagewiseEventKind::SeedReady),
            "the first callback must expose the fitted K=1 seed"
        );
        assert_eq!(
            events.get(1).map(|event| event.0),
            Some(StagewiseEventKind::BirthRoundStarted),
            "the second callback must expose a durable birth-round checkpoint"
        );
        assert_eq!(events[0].1, true, "seed_ready must be checkpointable");
        assert_eq!(
            events[1].1, true,
            "birth_round_started must be checkpointable before residual work"
        );
        assert_eq!(events[0].2, 1, "seed checkpoint must be K=1");

        let pos = |kind: StagewiseEventKind| -> usize {
            events
                .iter()
                .position(|event| event.0 == kind)
                .expect("expected progress event")
        };
        assert!(
            pos(StagewiseEventKind::ResidualModelStarted)
                < pos(StagewiseEventKind::CurrentEvidenceStarted),
            "residual-fit progress must precede current-evidence progress"
        );
        assert!(
            pos(StagewiseEventKind::CurrentEvidenceStarted)
                < pos(StagewiseEventKind::CandidateStarted),
            "current-evidence progress must precede candidate fitting"
        );
        assert!(
            events
                .iter()
                .any(|event| event.0 == StagewiseEventKind::CandidateStarted
                    && event.3 == Some(BirthKind::NewAtom)),
            "first birth must report the new-atom candidate"
        );
    }

    /// Backfitting monotonicity: each sweep is block-coordinate descent at fixed ρ,
    /// so the per-sweep EV trace is non-decreasing.
    #[test]
    fn backfitting_ev_is_monotone() {
        let n = 48usize;
        let p = 4usize;
        let evaluator = Arc::new(PeriodicHarmonicEvaluator::new(3).unwrap());
        let coords = Array2::<f64>::from_shape_fn((n, 1), |(row, _)| row as f64 / n as f64);
        let (atom0, cb0) = circle_atom("t0", &evaluator, &coords, 0, 1, p);
        let (atom1, cb1) = circle_atom("t1", &evaluator, &coords, 2, 3, p);
        let active_truth: Vec<Vec<bool>> = (0..n).map(|r| vec![r < n / 2, r >= n / 2]).collect();
        let (truth, _rho) = build_term(
            vec![atom0.clone(), atom1.clone()],
            vec![cb0.clone(), cb1.clone()],
            &active_truth,
        );
        let target = truth.fitted();
        let config = test_config();
        let (seed, rho) = build_term(vec![atom0], vec![cb0], &vec![vec![true]; n]);
        let (seed, rho) = fitted_seed(seed, rho, target.view(), &config);
        let result = fit_stagewise(seed, rho, target.view(), None, None, &config, None, None)
            .expect("fit_stagewise must complete");
        assert!(
            is_non_decreasing(&result.report.backfit_ev_trace),
            "backfitting EV must be monotone non-decreasing; got {:?}",
            result.report.backfit_ev_trace
        );
    }

    // ─────────────────────────────────────────────────────────────────────────
    //  Batched birth-racing (THRPT) — parity + batching-happened
    // ─────────────────────────────────────────────────────────────────────────

    /// Build a K-atom periodic term under an arbitrary gate mode from an ON/OFF
    /// routing table (mirrors `build_term`, which is softmax-only).
    fn build_term_gate(
        atoms: Vec<SaeManifoldAtom>,
        coord_blocks: Vec<Array2<f64>>,
        active: &[Vec<bool>],
        mode: AssignmentMode,
    ) -> (SaeManifoldTerm, SaeManifoldRho) {
        let n = active.len();
        let k = atoms.len();
        let mut logits = Array2::<f64>::zeros((n, k));
        for (row, atom_active) in active.iter().enumerate() {
            for (atom, &on) in atom_active.iter().enumerate() {
                logits[[row, atom]] = if on { ON } else { OFF };
            }
        }
        let assignment = SaeAssignment::from_blocks_with_mode_and_manifolds(
            logits,
            coord_blocks,
            vec![LatentManifold::Circle { period: 1.0 }; k],
            mode,
        )
        .unwrap();
        let term = SaeManifoldTerm::new(atoms, assignment).unwrap();
        let rho = SaeManifoldRho::new(0.0, 0.0, vec![Array1::<f64>::zeros(1); k]);
        (term, rho)
    }

    /// A hand-built disjoint circle birth seed: harmonic decoder on dirs `(a, b)`,
    /// an own-presence gate finite ONLY on `rows` (−∞ elsewhere), and a phase
    /// coordinate sweeping `[0, 1)` over `rows`. Under a ThresholdGate the −∞ →
    /// BIRTH_SEED_LOGIT off-support rows are gated EXACTLY off.
    fn disjoint_circle_seed(n: usize, p: usize, a: usize, b: usize, rows: &[usize]) -> BirthSeed {
        let mut decoder = Array2::<f64>::zeros((3, p));
        decoder[[1, a]] = 1.0;
        decoder[[2, b]] = 1.0;
        let mut gate = vec![f64::NEG_INFINITY; n];
        let mut phases = Array2::<f64>::zeros((n, 1));
        for (pos, &r) in rows.iter().enumerate() {
            gate[r] = 2.0;
            phases[[r, 0]] = pos as f64 / rows.len() as f64;
        }
        BirthSeed {
            decoder,
            energy: 1.0,
            circle_coords: Some(phases),
            circle_gate: Some(gate),
        }
    }

    /// Deterministic LCG uniform in [0,1) — a local RNG for planting clean circle
    /// geometry the ISA κ-certificate can certify (the softmax `build_term` image
    /// carries structured linear phase the fourth-order contrast rejects). Mirrors
    /// the `isa_seed` test generator so the driver is exercised on the same clean
    /// planted-circle distribution the producer's own gates prove it recovers.
    fn lcg_uniform(state: &mut u64) -> f64 {
        *state = state
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        ((*state >> 11) as f64) / ((1u64 << 53) as f64)
    }

    fn lcg_normal(state: &mut u64) -> f64 {
        let u1 = lcg_uniform(state).max(1e-12);
        let u2 = lcg_uniform(state);
        (-2.0 * u1.ln()).sqrt() * (std::f64::consts::TAU * u2).cos()
    }

    /// Axis-aligned dense torus: circle `c` lives in ambient plane `{2c, 2c+1}`,
    /// dense (every row on every circle), independent uniform phase per circle. This
    /// is the co-acceptance regime the ISA producer certifies cleanly (a rotation of
    /// its proven random-frame torus) AND whose circles occupy DISJOINT OUTPUT DIMS —
    /// so the born atoms are block-orthogonal via output columns even though they
    /// share every row, the case the output-dim disjointness route co-accepts.
    fn planted_axis_dense_circles(
        n: usize,
        p: usize,
        k: usize,
        amp: f64,
        sigma: f64,
        seed: u64,
    ) -> Array2<f64> {
        let mut state = seed;
        let mut data = Array2::<f64>::zeros((n, p));
        for i in 0..n {
            for c in 0..k {
                let th = std::f64::consts::TAU * lcg_uniform(&mut state);
                data[[i, 2 * c]] += amp * th.cos();
                data[[i, 2 * c + 1]] += amp * th.sin();
            }
            for j in 0..p {
                data[[i, j]] += sigma * lcg_normal(&mut state);
            }
        }
        data
    }

    /// THE PARITY LICENSE (batch-OMP orthogonality). Two DISJOINT planted circles:
    /// A on rows `[0,h)` in ambient plane (0,1), B on rows `[h,n)` in plane (2,3).
    /// Under a ThresholdGate the born atom's gate is EXACTLY 0 off its support, so
    /// B's K=1 fit reads only rows `[h,n)`. The A-deflated residual R1 equals the
    /// original R0 on those rows (A gates off there), so B's fit against R0 (the
    /// batched pass) is IDENTICAL to its fit against R1 (the serial pass). Prove it
    /// bit-for-bit: accepting the disjoint batch in one snapshot is not an
    /// approximation of the greedy sequence — it IS the greedy sequence.
    #[test]
    fn batched_disjoint_birth_fit_matches_serial_bit_for_bit() {
        let n = 40usize;
        let p = 6usize;
        let h = n / 2;
        let evaluator = Arc::new(PeriodicHarmonicEvaluator::new(3).unwrap());
        let coords = Array2::<f64>::from_shape_fn((n, 1), |(r, _)| r as f64 / n as f64);

        // Target: circle A on [0,h) in dirs (0,1); circle B on [h,n) in dirs (2,3).
        let mut target = Array2::<f64>::zeros((n, p));
        for r in 0..h {
            let th = std::f64::consts::TAU * (r as f64) / (h as f64);
            target[[r, 0]] = th.cos();
            target[[r, 1]] = th.sin();
        }
        for r in h..n {
            let th = std::f64::consts::TAU * ((r - h) as f64) / ((n - h) as f64);
            target[[r, 2]] = th.cos();
            target[[r, 3]] = th.sin();
        }

        // K=1 ThresholdGate seed: a circle atom on dirs (0,1), active on [0,h).
        let mode = AssignmentMode::threshold_gate(1.0, -3.0);
        let (atom0, cb0) = circle_atom("seed", &evaluator, &coords, 0, 1, p);
        let active: Vec<Vec<bool>> = (0..n).map(|r| vec![r < h]).collect();
        let (mut seed, mut rho) = build_term_gate(vec![atom0], vec![cb0], &active, mode);
        seed.set_guards_enabled(false);
        let config = test_config();
        seed.run_joint_fit_arrow_schur(
            target.view(),
            &mut rho,
            None,
            config.inner_max_iter,
            config.learning_rate,
            config.ridge_ext_coord,
            config.ridge_beta,
        )
        .expect("seed K=1 fit");

        let rows_a: Vec<usize> = (0..h).collect();
        let rows_b: Vec<usize> = (h..n).collect();
        let seed_a = disjoint_circle_seed(n, p, 0, 1, &rows_a);
        let seed_b = disjoint_circle_seed(n, p, 2, 3, &rows_b);

        let r0 = current_residual(&seed, target.view()).unwrap();
        // Batched: B raced against the ORIGINAL residual R0.
        let b_batched = race_birth_seed(
            &seed,
            &rho,
            &seed_b,
            r0.view(),
            target.view(),
            None,
            &config,
        )
        .expect("race B against R0");
        // Serial: accept A first, deflate to R1, then race B against R1.
        let a = race_birth_seed(
            &seed,
            &rho,
            &seed_a,
            r0.view(),
            target.view(),
            None,
            &config,
        )
        .expect("race A against R0");
        let (term_a, rho_a) = append_fitted_atom(
            &seed,
            &rho,
            a.born_atom.clone(),
            a.born_coord.clone(),
            &a.born_logit_col,
            a.born_ard.clone(),
            a.born_log_lambda_smooth,
        )
        .expect("append A");
        let r1 = current_residual(&term_a, target.view()).unwrap();
        // R1 must equal R0 on B's rows (A gates off there) — the orthogonality
        // precondition, checked directly.
        let mut rows_b_diff = 0.0_f64;
        for &r in &rows_b {
            for j in 0..p {
                rows_b_diff += (r0[[r, j]] - r1[[r, j]]).abs();
            }
        }
        assert!(
            rows_b_diff < 1e-9,
            "R0 and R1 must be identical on B's disjoint rows; L1 diff {rows_b_diff}"
        );
        let b_serial = race_birth_seed(
            &term_a,
            &rho_a,
            &seed_b,
            r1.view(),
            target.view(),
            None,
            &config,
        )
        .expect("race B against R1");

        let d_batched = &b_batched.born_atom.decoder_coefficients;
        let d_serial = &b_serial.born_atom.decoder_coefficients;
        let decoder_diff = (d_batched - d_serial).mapv(f64::abs).sum();
        assert!(
            decoder_diff < 1e-6,
            "disjoint birth B must fit IDENTICALLY against R0 (batched) and R1 (serial); \
             decoder L1 diff {decoder_diff}"
        );
        // The per-birth CHARGE that MUST be invariant to acceptance order is the
        // reconstruction ΔEV, not the raw joint-penalized LAML delta. B touches only rows
        // `[h,n)` (its gate is 0 elsewhere), and the EV denominator `SS_tot` is fixed
        // by the target, so B's ΔEV = (SS_res reduction on B's rows)/SS_tot is
        // identical whether or not A — disjoint from B — is already present:
        //
        //   batched:  EV(seed+B)   − EV(seed)     (A absent)
        //   serial:   EV(seed+A+B) − EV(seed+A)   (A present)
        //
        // The RAW joint-penalized LAML delta does NOT decompose this way: the frozen Laplace
        // criterion carries a globally-pooled dispersion term that couples all rows,
        // so adding A (which shrinks the residual on `[0,h)`) rescales B's penalized LAML
        // charge even though B's FIT is bit-identical (asserted above). Charging on
        // the additive quantity (ΔEV) is the honest disjoint-support parity claim.
        let seed_ev = ev_of(&seed, target.view());
        let term_a_ev = ev_of(&term_a, target.view());
        let charge_batched = b_batched.ev - seed_ev;
        let charge_serial = b_serial.ev - term_a_ev;
        assert!(
            (charge_batched - charge_serial).abs() < 1e-6,
            "disjoint birth B marginal ΔEV charge must match batched vs serial; \
             {charge_batched} vs {charge_serial}"
        );
    }

    /// DRIVER PARITY. An AXIS-ALIGNED dense torus (`q` circles, circle `c` in
    /// ambient plane `{2c, 2c+1}`, every row on every circle) at the ISA producer's
    /// proven regime, driven from an UNFIT K=1 softmax seed. (The seed is passed
    /// unfit on purpose: a warm K=1 joint fit — with free per-row coords and a free
    /// full-`p` decoder — chases a rank-2 blend across ALL circles and contaminates
    /// the residual so the κ certificate rejects it; an unfit seed leaves round 1 a
    /// clean multi-circle residual the producer certifies. Softmax, not a
    /// ThresholdGate: the born circle's own-presence gate is starved below a −3
    /// threshold on this synthetic and would never clear the birth gate, whereas
    /// under softmax every born atom contributes and the drivers grow K.)
    ///
    /// The serial and batched drivers must reach the SAME atom count (±1) and a
    /// comparable final EV — real parity where BOTH grow K, not a vacuous 0-vs-0.
    /// (The batched co-acceptance MACHINERY — that it is not a serial loop in
    /// disguise — is proven directly on the selection primitive by
    /// [`batched_round_co_accepts_via_both_routes`]; through the full driver the
    /// softmax gate couples the atoms so co-acceptance reduces to one per round,
    /// while the primitive test exhibits the disjoint batch the independent-gate
    /// production path co-accepts.)
    #[test]
    fn batched_driver_matches_serial_and_batches() {
        let n = 900usize;
        let p = 16usize;
        let q = 3usize; // circles in ambient planes {0,1},{2,3},{4,5}; dims 6-15 noise
        let evaluator = Arc::new(PeriodicHarmonicEvaluator::new(3).unwrap());
        let coords = Array2::<f64>::from_shape_fn((n, 1), |(r, _)| r as f64 / n as f64);
        let target = planted_axis_dense_circles(n, p, q, 1.0, 0.03, 0x2111_A11E_u64);

        let mode = AssignmentMode::softmax(1.0);
        let mut config = test_config();
        config.max_births = 8;
        config.max_backfit_sweeps = 1;

        // An UNFIT K=1 softmax seed (see the doc-comment): its near-zero
        // reconstruction leaves round 1 a clean q-circle residual to mine.
        let build_seed = || {
            let (atom0, cb0) = circle_atom("seed", &evaluator, &coords, 0, 1, p);
            let (mut seed, rho) =
                build_term_gate(vec![atom0], vec![cb0], &vec![vec![true]; n], mode);
            seed.set_guards_enabled(false);
            (seed, rho)
        };

        let (seed_s, rho_s) = build_seed();
        let serial = fit_stagewise(
            seed_s,
            rho_s,
            target.view(),
            None,
            None,
            &config,
            None,
            None,
        )
        .expect("serial driver");
        assert!(
            serial.report.births_accepted >= 2,
            "serial driver must grow K on the planted {q}-circle image for the parity \
             comparison to be meaningful; births_accepted={}",
            serial.report.births_accepted
        );

        let (seed_b, rho_b) = build_seed();
        let batch_config = BatchedStagewiseConfig {
            base: config,
            max_candidates_per_round: 8,
        };
        let batched =
            fit_stagewise_batched(seed_b, rho_b, target.view(), None, None, &batch_config)
                .expect("batched driver");

        assert!(
            batched.report.births_accepted >= 2,
            "batched driver must also grow K on the planted {q}-circle image; \
             births_accepted={}",
            batched.report.births_accepted
        );
        assert!(
            is_non_decreasing(&batched.report.ev_trace),
            "batched EV must be monotone non-decreasing; got {:?}",
            batched.report.ev_trace
        );
        assert!(
            batched.report.terminal_joint_penalized_laml.is_finite(),
            "batched terminal joint penalized LAML must be finite"
        );

        let serial_k = serial.term.k_atoms() as i64;
        let batched_k = batched.term.k_atoms() as i64;
        assert!(
            (serial_k - batched_k).abs() <= 1,
            "batched atom count {batched_k} must match serial {serial_k} within 1"
        );

        let serial_ev = *serial.report.ev_trace.last().unwrap();
        let batched_ev = *batched.report.ev_trace.last().unwrap();
        assert!(
            (serial_ev - batched_ev).abs() < 0.1,
            "batched final EV {batched_ev} must match serial {serial_ev} within 0.1"
        );
    }

    /// CO-ACCEPTANCE MACHINERY (not a serial loop in disguise). The disjoint-batch
    /// selection [`select_disjoint_batch`] — the criterion that decides how many
    /// births one residual snapshot co-accepts — must co-accept ≥2 candidates via
    /// BOTH block-orthogonality routes and co-accept only ONE when a pair overlaps
    /// on both axes. The candidates carry a REAL raced born atom (from
    /// [`race_birth_seed`], so the struct is exactly what production builds); only
    /// the `support` / `out_support` sets — the two fields the selection reads — are
    /// set to the geometry under test. (That the raced fields THEMSELVES separate on
    /// disjoint planted structure is exercised end-to-end by the driver + bit-for-bit
    /// tests; here we pin down the SELECTION logic independent of the fit's
    /// dominant-variance migration.)
    #[test]
    fn batched_round_co_accepts_via_both_routes() {
        let n = 40usize;
        let p = 8usize;
        let h = n / 2;
        let evaluator = Arc::new(PeriodicHarmonicEvaluator::new(3).unwrap());
        let coords = Array2::<f64>::from_shape_fn((n, 1), |(r, _)| r as f64 / n as f64);
        let config = test_config();

        // A real raced candidate to use as a valid template (its born atom / coords /
        // ρ are genuine); we then clone it and set the support geometry per case.
        let (atom0, cb0) = circle_atom("seed", &evaluator, &coords, 0, 1, p);
        let (mut seed_t, rho_t) = build_term_gate(
            vec![atom0],
            vec![cb0],
            &vec![vec![true]; n],
            AssignmentMode::threshold_gate(1.0, -3.0),
        );
        seed_t.set_guards_enabled(false);
        let target = Array2::<f64>::from_shape_fn((n, p), |(r, j)| {
            if j < 2 {
                let th = std::f64::consts::TAU * (r as f64) / (n as f64);
                if j == 0 { th.cos() } else { th.sin() }
            } else {
                0.0
            }
        });
        let r0 = current_residual(&seed_t, target.view()).unwrap();
        let sa = disjoint_circle_seed(n, p, 0, 1, &(0..n).collect::<Vec<_>>());
        let template = race_birth_seed(
            &seed_t,
            &rho_t,
            &sa,
            r0.view(),
            target.view(),
            None,
            &config,
        )
        .expect("race template");
        let with_support = |rows: Vec<usize>, dims: Vec<usize>| -> RacedCandidate {
            let mut c = template.clone();
            c.support = rows;
            c.out_support = dims;
            c
        };

        // Route 1 — ROW-disjoint (shared output dims): must co-accept both.
        let rows_a: Vec<usize> = (0..h).collect();
        let rows_b: Vec<usize> = (h..n).collect();
        let raced_rows = vec![
            with_support(rows_a.clone(), vec![0, 1]),
            with_support(rows_b.clone(), vec![0, 1]),
        ];
        let (accepted, requeued) = select_disjoint_batch(&raced_rows, &[0, 1], 8);
        assert_eq!(
            accepted.len(),
            2,
            "two ROW-disjoint candidates must co-accept (accepted={accepted:?}, requeued={requeued})"
        );

        // Route 2 — OUTPUT-DIM-disjoint (shared rows): the route the row-only test
        // could never co-accept — must co-accept both.
        let all_rows: Vec<usize> = (0..n).collect();
        let raced_dims = vec![
            with_support(all_rows.clone(), vec![0, 1]),
            with_support(all_rows.clone(), vec![2, 3]),
        ];
        let (accepted2, requeued2) = select_disjoint_batch(&raced_dims, &[0, 1], 8);
        assert_eq!(
            accepted2.len(),
            2,
            "two OUTPUT-DIM-disjoint candidates must co-accept \
             (accepted={accepted2:?}, requeued={requeued2})"
        );

        // Overlap on BOTH axes — the only conflict: exactly one co-accepts, one requeues.
        let raced_conflict = vec![
            with_support(all_rows.clone(), vec![0, 1]),
            with_support(all_rows.clone(), vec![0, 1]),
        ];
        let (accepted3, requeued3) = select_disjoint_batch(&raced_conflict, &[0, 1], 8);
        assert_eq!(
            (accepted3.len(), requeued3),
            (1, 1),
            "candidates overlapping on BOTH rows and output dims must NOT co-accept; \
             accepted={accepted3:?} requeued={requeued3}"
        );
    }

    // The births/round MULTIPLIER *timing* benchmark that lived here was removed:
    // it was ignore-marked (a pure stderr wall-clock report, no assertion, "not
    // a CI gate per SPEC"), and ignore-marked timing benches are banned workspace-
    // wide by `build.rs`. Its behavioural guarantee — that the batched selection
    // co-accepts ≥2 block-orthogonal births in one round (not a serial loop in
    // disguise) — is enforced non-ignored by `batched_round_co_accepts_via_both_routes`;
    // the standalone births/sec timing lives in the `stagewise_batched_births` bench.
}
