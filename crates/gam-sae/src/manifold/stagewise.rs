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
//! certified 1-manifold REML fit rather than a rank-1 SVD). The joint solver
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
//! frozen joint REML criterion strictly improves) PLUS an explicit MINIMUM-EFFECT
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
//! already exposed by [`SaeManifoldTerm::reml_criterion`] at `inner_max_iter ==
//! 0`, so no new `frozen_evaluate` primitive is needed; [`frozen_joint_evidence`]
//! is the thin, named wrapper this module and its callers use.
//!
//! # Determinism & SPEC
//!
//! No RNG, no clock, no wall-clock budget, no grid search. Every threshold is a
//! typed config knob (defaulting to the null-recovering value) or a data-derived
//! quantity (the residual model's evidence-selected factor rank is the salience
//! oracle). REML throughout (the inner fits and the frozen evidence pass are the
//! same REML criterion every term is scored by).

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
    /// Frozen joint REML criterion before the round (lower is better evidence).
    pub joint_reml_before: f64,
    /// Frozen joint REML criterion of the winning candidate (or the unchanged
    /// pre-round value when the round was rejected).
    pub joint_reml_after: f64,
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
    /// The frozen (evaluate-don't-optimize) joint REML criterion of the final
    /// composed dictionary — the terminal Phase-3 evidence.
    pub terminal_joint_reml: f64,
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
    pub joint_reml_before: Option<f64>,
    pub joint_reml_after: Option<f64>,
    pub terminal_joint_reml: Option<f64>,
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

/// Frozen (`inner_max_iter == 0`, the #850 freeze) joint REML criterion of a term
/// at its current `(t, β)` — evaluate-don't-optimize. This is the joint-Laplace
/// evidence at a fixed converged state (`loss.total() + extra penalties + ½
/// log|H| − Occam`), the quantity the birth evidence gate and the terminal
/// assembly compare on. Lower is better evidence.
pub fn frozen_joint_evidence(
    term: &mut SaeManifoldTerm,
    target: ArrayView2<'_, f64>,
    rho: &SaeManifoldRho,
    registry: Option<&AnalyticPenaltyRegistry>,
    config: &StagewiseConfig,
) -> Result<(f64, SaeManifoldLoss), String> {
    term.reml_criterion(
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

/// Fit the running structured residual-covariance `Σ` on `R = target − fitted`.
/// Returns `None` when the residual is empty/single-channel (no factor subspace).
fn fit_residual_covariance(
    term: &SaeManifoldTerm,
    target: ArrayView2<'_, f64>,
    config: &StagewiseConfig,
) -> Result<Option<(Array2<f64>, StructuredResidualModel)>, String> {
    let residual = current_residual(term, target)?;
    let (n, p) = residual.dim();
    if n == 0 || p < 2 {
        return Ok(None);
    }
    let activity = activity_of(term);
    let max_rank = config.max_factor_rank.min(p.saturating_sub(1)).max(1);
    match StructuredResidualModel::fit(ResidualFactorInput {
        residuals: residual.view(),
        activity: activity.view(),
        max_factor_rank: max_rank,
    }) {
        Ok(model) => Ok(Some((residual, model))),
        // A degenerate residual fit is a stop signal, not an error.
        Err(_) => Ok(None),
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
    if let Some(w) = term.row_loss_weights().map(|w| w.to_vec()) {
        sub_term.set_row_loss_weights(w)?;
    }
    if let Some(metric) = term.row_metric().cloned() {
        sub_term.set_row_metric(metric)?;
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

/// Lift the top residual-factor direction to an `(m, p)` birth decoder in atom
/// 0's basis: the `p`-vector direction placed on the constant (row-0) basis row,
/// exactly the contract [`crate::structure_harvest::apply_structure_move`]'s
/// `Birth` expects (`born_atom` then races the topology and reshapes the seed).
/// Returns the decoder and the top factor's explained energy (the reported dose).
fn top_factor_birth_decoder(
    term: &SaeManifoldTerm,
    model: &StructuredResidualModel,
) -> Option<(Array2<f64>, f64)> {
    if model.factor_rank() == 0 {
        return None;
    }
    let factor = model.factor(); // (p, r), columns in descending explained energy
    let p = factor.nrows();
    let energy: f64 = factor.column(0).iter().map(|v| v * v).sum();
    if !(energy > 0.0) {
        return None;
    }
    let m = term.atoms[0].basis_size();
    let mut decoder = Array2::<f64>::zeros((m, p));
    for out in 0..p {
        decoder[[0, out]] = factor[[out, 0]];
    }
    Some((decoder, energy))
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
/// left behind). Independent-gate modes (JumpReLU / IBP) refit exactly-additively;
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
        let weights = term.assignment.try_assignments_row_for_rho(row, rho)?;
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
            joint_reml_before: None,
            joint_reml_after: None,
            terminal_joint_reml: None,
            term: &term,
            rho: &rho,
        },
    )?;

    // ── Phase 1b — forward births ──────────────────────────────────────────────
    let mut birth_round = 0usize;
    let stopped_reason = loop {
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
                checkpoint: false,
                k_atoms: term.k_atoms(),
                births_accepted,
                births_rejected,
                ev: Some(entry_ev),
                factor_energy: None,
                joint_reml_before: None,
                joint_reml_after: None,
                terminal_joint_reml: None,
                term: &term,
                rho: &rho,
            },
        )?;
        // Refit Σ on the current residual and install the whitened metric so the
        // candidate fits run under the structured covariance from atom one.
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
                joint_reml_before: None,
                joint_reml_after: None,
                terminal_joint_reml: None,
                term: &term,
                rho: &rho,
            },
        )?;
        let Some((residual, model)) = fit_residual_covariance(&term, target, config)? else {
            break StagewiseStop::NoResidualStructure;
        };
        let Some((birth_decoder, factor_energy)) = top_factor_birth_decoder(&term, &model) else {
            break StagewiseStop::NoResidualStructure;
        };
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
                joint_reml_before: None,
                joint_reml_after: None,
                terminal_joint_reml: None,
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
                joint_reml_before: None,
                joint_reml_after: None,
                terminal_joint_reml: None,
                term: &term,
                rho: &rho,
            },
        )?;
        let (cur_reml, _) = frozen_joint_evidence(&mut term, target, &rho, registry, config)?;
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
                joint_reml_before: Some(cur_reml),
                joint_reml_after: None,
                terminal_joint_reml: None,
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
                joint_reml_before: Some(cur_reml),
                joint_reml_after: None,
                terminal_joint_reml: None,
                term: &term,
                rho: &rho,
            },
        )?;
        let mut cand_a = apply_structure_move(
            &term,
            &rho,
            &StructureMove::Birth { candidate: 0 },
            std::slice::from_ref(&birth_decoder),
        )
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
            let (reml, _) =
                frozen_joint_evidence(&mut cand_term, target, &cand_rho, registry, config)?;
            let ev = ev_of(&cand_term, target);
            Ok((cand_term, cand_rho, reml, ev))
        })
        .ok();
        if let Some((cand_term, cand_rho, reml, ev)) = cand_a.as_ref() {
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
                    joint_reml_before: Some(cur_reml),
                    joint_reml_after: Some(*reml),
                    terminal_joint_reml: None,
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
                    joint_reml_before: Some(cur_reml),
                    joint_reml_after: None,
                    terminal_joint_reml: None,
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
                    joint_reml_before: Some(cur_reml),
                    joint_reml_after: None,
                    terminal_joint_reml: None,
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
                let (reml, _) =
                    frozen_joint_evidence(&mut cand_term, target, &cand_rho, registry, config)?;
                let ev = ev_of(&cand_term, target);
                Ok((cand_term, cand_rho, reml, ev))
            })();
            let out = built.ok();
            if let Some((cand_term, cand_rho, reml, ev)) = out.as_ref() {
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
                        joint_reml_before: Some(cur_reml),
                        joint_reml_after: Some(*reml),
                        terminal_joint_reml: None,
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
                        joint_reml_before: Some(cur_reml),
                        joint_reml_after: None,
                        terminal_joint_reml: None,
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
        // floor. Among the candidates that clear both gates, the lower REML wins.
        let passes = |reml: f64, ev: f64| -> bool {
            reml.is_finite()
                && reml < cur_reml
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
                    joint_reml_before: cur_reml,
                    joint_reml_after: cur_reml,
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
                        joint_reml_before: Some(cur_reml),
                        joint_reml_after: Some(cur_reml),
                        terminal_joint_reml: None,
                        term: &term,
                        rho: &rho,
                    },
                )?;
                continue;
            }
        };

        let (kind, (cand_term, cand_rho, reml_after, ev_after)) = if choose_a {
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
            joint_reml_before: cur_reml,
            joint_reml_after: reml_after,
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
                joint_reml_before: Some(cur_reml),
                joint_reml_after: Some(reml_after),
                terminal_joint_reml: None,
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
                joint_reml_before: None,
                joint_reml_after: None,
                terminal_joint_reml: None,
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
                    joint_reml_before: None,
                    joint_reml_after: None,
                    terminal_joint_reml: None,
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
                    joint_reml_before: None,
                    joint_reml_after: None,
                    terminal_joint_reml: None,
                    term: &term,
                    rho: &rho,
                },
            )?;
            break;
        }
    }

    // ── Phase 3 — terminal frozen joint evidence of the composed tier ──────────
    let (terminal_joint_reml, terminal_joint_loss) =
        frozen_joint_evidence(&mut term, target, &rho, registry, config)?;

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
            joint_reml_before: None,
            joint_reml_after: Some(terminal_joint_reml),
            terminal_joint_reml: Some(terminal_joint_reml),
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
            terminal_joint_reml,
            terminal_joint_loss,
        },
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
    let (reml, loss) = frozen_joint_evidence(&mut merged, target, &merged_rho, registry, config)?;
    merged.set_guards_enabled(true);
    Ok((merged, merged_rho, reml, loss))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::manifold::{
        AssignmentMode, PeriodicHarmonicEvaluator, SaeAssignment, SaeAtomBasisKind,
        SaeBasisEvaluator, SaeManifoldAtom,
    };
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
        let atom = SaeManifoldAtom::new(
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
        let result = fit_stagewise(seed, rho, target.view(), None, None, &config, None)
            .expect("fit_stagewise must complete on planted two-circles");

        assert!(
            is_non_decreasing(&result.report.ev_trace),
            "EV must be monotone non-decreasing in births by construction; got {:?}",
            result.report.ev_trace
        );
        assert!(
            result.report.terminal_joint_reml.is_finite(),
            "terminal frozen joint REML must be finite"
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
        let result = fit_stagewise(seed, rho, target.view(), None, None, &config, None)
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
        let result = fit_stagewise(seed, rho, target.view(), None, None, &config, None)
            .expect("fit_stagewise must complete");
        assert!(
            is_non_decreasing(&result.report.backfit_ev_trace),
            "backfitting EV must be monotone non-decreasing; got {:?}",
            result.report.backfit_ev_trace
        );
    }
}
