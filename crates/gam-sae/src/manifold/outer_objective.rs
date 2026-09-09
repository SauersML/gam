use super::*;
use gam_math::special::bessel_i0_log_and_ratio;
use gam_solve::rho_optimizer::{
    FixedPointCertificateEval, FixedPointCoordinateCertificate, OuterResult,
};

pub(crate) fn reconstruction_explained_variance(
    target: ArrayView2<'_, f64>,
    fitted: ArrayView2<'_, f64>,
) -> Option<f64> {
    if target.dim() != fitted.dim() {
        return None;
    }
    let (n, p) = target.dim();
    if n == 0 || p == 0 {
        return None;
    }
    let mut means = vec![0.0_f64; p];
    for col in 0..p {
        let mut acc = 0.0;
        for row in 0..n {
            acc += target[[row, col]];
        }
        means[col] = acc / n as f64;
    }
    let mut ssr = 0.0_f64;
    let mut sst = 0.0_f64;
    for row in 0..n {
        for col in 0..p {
            let residual = target[[row, col]] - fitted[[row, col]];
            ssr += residual * residual;
            let centered = target[[row, col]] - means[col];
            sst += centered * centered;
        }
    }
    if ssr.is_finite() && sst.is_finite() && sst > f64::MIN_POSITIVE {
        Some(1.0 - ssr / sst)
    } else {
        None
    }
}

/// Observable telemetry for the amortized basin-entry accelerator: attempted
/// evaluations, positive and zero-row certificates, and failures. Failures are
/// propagated by the caller and never converted into a different solve path.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct AmortizedWarmStartTelemetry {
    /// Outer evals that invoked the warm-start (gradient + value-probe lanes).
    pub attempts: usize,
    /// Evals where the amortized encoder certified ≥1 row → a real warm-start.
    pub warm_started_evals: usize,
    /// Evals where the encoder validly certified zero rows.
    pub zero_certified_evals: usize,
    /// Evals where the warm-start path returned an error.
    pub failed_attempts: usize,
    /// Total certified (row, atom) coords warm-started across all evals.
    pub total_rows_warm_started: usize,
}

/// #2235 — outer termination ledger: one per fit, ticked by every criterion
/// evaluation lane. This is pure accounting:
///
/// * A fit object exists ONLY when the outer bridge concludes through its own
///   convergence/stopping logic. There is no freeze, no deadline-return, no
///   "best-effort fit" lane — an incomplete optimization must never mint a
///   consumable fit (that would remove all pressure to fix the solver; the
///   user's moral-hazard rule).
/// * Convergence and non-convergence belong to the shared outer optimizer. This
///   application ledger never substitutes an evaluation-count or wall-clock
///   deadline for the optimizer's analytic certificate. Wall survival is the
///   checkpoint/resume lane's job (`persistent_warm_start`).
#[derive(Debug, Clone)]
pub(crate) struct OuterTerminationLedger {
    /// Total criterion evaluations across all lanes.
    evals: u64,
    /// Eval count at the last MATERIAL improvement of the best cost.
    last_improvement_eval: u64,
    /// Best (lowest) finite criterion value seen.
    best_cost: Option<f64>,
    /// Fit wall-clock start.
    wall_start: std::time::Instant,
}

impl OuterTerminationLedger {
    pub(crate) fn new() -> Self {
        Self {
            evals: 0,
            last_improvement_eval: 0,
            best_cost: None,
            wall_start: std::time::Instant::now(),
        }
    }

    /// Record one finite criterion value; returns `true` on a MATERIAL
    /// improvement of the best cost (the caller's checkpoint-bank signal).
    ///
    /// This is the single point every outer criterion evaluation passes
    /// through, so it is also where the fit says where it has got to (#2472).
    /// A long SAE fit used to emit nothing at all between its start and its
    /// return, which meant a run killed at any point — a CI timeout, a lost
    /// box — yielded no trace, no iteration count, and no criterion history.
    /// One line per evaluation is bounded by the evaluation count rather than
    /// by the inner iteration count, so it stays readable on a fit that runs
    /// for hours.
    ///
    /// `gradient_norm` is optional because most lanes reaching here are
    /// value-only (the EFS, streaming and criterion-only routes carry no
    /// gradient); those print `grad=na` rather than reporting a zero gradient
    /// that was never measured.
    pub(crate) fn record(&mut self, cost: f64, gradient_norm: Option<f64>) -> bool {
        self.evals += 1;
        let gradient_field = match gradient_norm {
            Some(norm) => format!("{norm:.6e}"),
            None => "na".to_string(),
        };
        if !cost.is_finite() {
            log::info!(
                "[SAE/outer] eval={} criterion={:.9e} grad={} best={:.9e} improved=false",
                self.evals,
                cost,
                gradient_field,
                self.best_cost.unwrap_or(f64::NAN),
            );
            return false;
        }
        let improved = match self.best_cost {
            None => true,
            // Material improvement at the same scale the inner stall gate
            // uses: a relative decrease beyond the EV-degradation tolerance.
            Some(best) => cost < best - SAE_FINAL_EV_DEGRADATION_TOL * (1.0 + best.abs()),
        };
        if improved {
            self.best_cost = Some(match self.best_cost {
                Some(best) => best.min(cost),
                None => cost,
            });
            self.last_improvement_eval = self.evals;
        }
        log::info!(
            "[SAE/outer] eval={} criterion={:.9e} grad={} best={:.9e} improved={improved}",
            self.evals,
            cost,
            gradient_field,
            self.best_cost.unwrap_or(cost),
        );
        improved
    }

    /// Resume accounting from a checkpoint. The wall clock restarts because it
    /// is telemetry, never a solver deadline.
    pub(crate) fn seed_from_checkpoint(
        &mut self,
        evals: u64,
        last_improvement_eval: u64,
        best_cost: Option<f64>,
    ) {
        self.evals = evals;
        self.last_improvement_eval = last_improvement_eval.min(evals);
        self.best_cost = best_cost.filter(|c| c.is_finite());
    }

    /// Snapshot the ledger counters for a checkpoint write.
    pub(crate) fn checkpoint_counters(&self) -> (u64, u64, Option<f64>) {
        (self.evals, self.last_improvement_eval, self.best_cost)
    }

    /// New multi-start seed: start its improvement telemetry at the current
    /// count; total evaluations and wall measurement remain fit-global.
    pub(crate) fn reset_improvement_baseline(&mut self) {
        self.last_improvement_eval = self.evals;
    }

    pub(crate) fn report(&self, verdict: SaeOuterVerdict) -> SaeOuterTermination {
        SaeOuterTermination {
            verdict,
            evals: self.evals,
            evals_since_improvement: self.evals.saturating_sub(self.last_improvement_eval),
            wall: self.wall_start.elapsed(),
        }
    }
}

/// #2235 — how the outer search of a minted fit concluded. Every variant is a
/// CONVERGED ending (non-convergence raises a typed error before a fit
/// exists), so this is certificate provenance, not a success/failure flag —
/// there is deliberately no budget/freeze variant.
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum SaeOuterVerdict {
    /// The generic outer ρ-search ran and concluded with this certificate
    /// (gradient-stationary / criterion-flat #2241 / recurrent-incumbent).
    Search(OuterConvergedVia),
    /// No outer ρ-search ran: the caller pinned ρ, so only the inner solve's
    /// KKT certificate applies.
    FixedRho,
    /// No optimizer ran: the caller-installed inner state and ρ independently
    /// passed the same analytic inner-KKT and outer-criterion stationarity
    /// authorities a native fit must pass (#2263).
    Audited(OuterConvergedVia),
}

impl SaeOuterVerdict {
    /// Stable wire name; the enums own the vocabulary so bindings marshal
    /// instead of mapping (precedent: ba57254af).
    pub fn as_str(&self) -> &'static str {
        match self {
            Self::Search(via) => via.as_str(),
            Self::FixedRho => "fixed_rho",
            Self::Audited(_) => "audited_stationary",
        }
    }
}

/// #2235 — outer-search accounting carried out of a CONVERGED fit (the only
/// kind that exists: a defect raises before a fit is minted).
#[derive(Debug, Clone, Copy)]
pub struct SaeOuterTermination {
    /// Which certificate concluded the search (#2235/#2241).
    pub verdict: SaeOuterVerdict,
    pub evals: u64,
    pub evals_since_improvement: u64,
    pub wall: std::time::Duration,
}

#[derive(Debug)]
pub struct SaeIntoFittedResult {
    pub term: SaeManifoldTerm,
    pub rho: SaeManifoldRho,
    pub loss: SaeManifoldLoss,
    /// Terminal value of the custom penalized quasi-Laplace criterion at the
    /// outer stationary state, before the optional image-frozen post-fit chart
    /// canonicalization. This is distinct from `loss.total()`; consult
    /// `charts_canonicalized` to know whether the returned term is a transported
    /// chart representative of that state. The scalar uses the declared
    /// PSD/Gauss--Newton factor and rank charges; it is not normalized evidence.
    pub penalized_quasi_laplace_criterion: f64,
    /// True when post-fit chart canonicalization changed any atom's chart.
    pub charts_canonicalized: bool,
    /// #2235 — how the outer search ended (verdict + eval/wall ledger).
    pub termination: SaeOuterTermination,
}

/// A converged fixed-`rho` inner state that lies on the boundary of its
/// fixed-`K` stratum.  It is intentionally not a fit: the stage orchestrator
/// must either remove a proper subset and re-run the reduced outer problem to
/// certification, or materialize the exact Tier-0 null when every atom
/// vanished.
pub(crate) struct SaeVanishedStageState {
    pub term: SaeManifoldTerm,
    pub rho: SaeManifoldRho,
    pub atoms: VanishedAtoms,
}

impl SaeIntoFittedResult {
    pub fn invalidates_pre_final_shape_uncertainty(&self) -> bool {
        self.charts_canonicalized
    }
}

impl AmortizedWarmStartTelemetry {
    /// Fold one warm-start outcome into the running tally. `Ok(rows)` with
    /// `rows > 0` is a genuine warm-start; `Ok(0)` is a valid zero-row
    /// certificate; `Err` is a propagated failure.
    pub(crate) fn record(&mut self, outcome: &Result<usize, String>) {
        self.attempts += 1;
        match outcome {
            Ok(0) => self.zero_certified_evals += 1,
            Ok(rows) => {
                self.warm_started_evals += 1;
                self.total_rows_warm_started += rows;
            }
            Err(_) => self.failed_attempts += 1,
        }
    }
}

/// #2080 — probe telemetry for the outer penalized quasi-Laplace ρ-search. Counts how the outer
/// objective spends its criterion evaluations so the wide-`p` acceptance test can
/// assert a BOUNDED probe budget (not a wall-clock limit — SPEC bans time
/// budgets). Every counter is a plain evaluation tally; the fields are read after
/// a fit via [`SaeManifoldOuterObjective::probe_telemetry`].
///
/// The load-bearing metric is `infeasible_*`: at a wide-`p` planted-circle fit the
/// outer line search overshoots into the adjacent indefinite (non-PD Laplace)
/// basin on nearly every probe. Historically each such probe ground the inner
/// refinement budget (up to `64×inner_max_iter`) before refusing; the #2080 fix
/// makes an infeasible PROBE return the typed refusal after a single diagnostic
/// pass, so `infeasible_*` can be large while the fit still terminates in a
/// bounded number of criterion evals.
/// Why a ρ-probe has no defined penalized quasi-Laplace value.
///
/// # One table, because there were two and they had drifted
///
/// Recoverability (`is_recoverable_value_probe_refusal`) and telemetry kind
/// (`OuterProbeTelemetry::record_refusal_kind`) are two READS of one
/// classification. Until #2593 each parsed the rendered message with its own
/// independent substring ladder, and the two had already fallen out of step:
/// the gated-design and co-collapse refusals were recoverable but counted by
/// nobody, and the telemetry ladder accepted a bare "Schur complement Cholesky
/// failed" where the recoverability ladder required a non-PD pivot as well.
///
/// The classification is a property of WHAT REFUSED, so it wants to be decided
/// once. `classify` is now the only place in this crate that reads the prose,
/// and both consumers derive from it; adding a kind without a counter fails
/// `every_refusal_kind_is_counted_exactly_once`.
///
/// # Why this is still a parse, and what it is waiting on
///
/// The right home for this bit is the producer: each refusal site knows its own
/// class and should hand out a typed value, exactly as
/// `EstimationError::TrialPointRefused` now does for the custom-family boundary.
/// gam-sae cannot do that yet — its inner spine is `Result<_, String>` end to
/// end (149 such signatures across the three modules on this path alone, ~400
/// across the manifold module), so there is no type between producer and
/// consumer to carry it. Collapsing four readers to one does not need that
/// refactor and does not block it: when the spine is typed, `classify` is the
/// single call site to delete.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum ProbeRefusalKind {
    /// The fixed-ρ inner solve exhausted its budget without reaching KKT.
    InnerNotConverged,
    /// A per-row H_tt block was not positive definite before KKT stationarity.
    NonPdPerRow,
    /// The reduced joint-Hessian Schur complement was indefinite.
    NonPdSchur,
    /// A gate turned an atom off at every row (#2087).
    AllZeroGatedDesign,
    /// Certified same-state decoder disappearance after the reseed budget
    /// (#2089 / #2362).
    TotalCoCollapse,
}

impl ProbeRefusalKind {
    /// Every kind. `infeasible_total` and the coverage test iterate this, so a
    /// new variant cannot be added and then silently left uncounted.
    pub(crate) const ALL: [Self; 5] = [
        Self::InnerNotConverged,
        Self::NonPdPerRow,
        Self::NonPdSchur,
        Self::AllZeroGatedDesign,
        Self::TotalCoCollapse,
    ];

    /// The phrase this crate writes to mark an `InnerNotConverged` refusal.
    ///
    /// #2598 — the four markers this crate owns each have exactly one home now.
    /// The producer INTERPOLATES the phrase and [`Self::classify`] matches the
    /// same function, so the two copies that used to be maintained separately
    /// cannot drift apart. (`NonPdSchur` has no marker here: gam-solve owns both
    /// that wording and its reader — see the note in `classify`.)
    pub(crate) fn inner_not_converged_marker() -> &'static str {
        "inner solve did not converge at fixed ρ"
    }

    /// The phrase this crate writes to mark a `NonPdPerRow` refusal.
    ///
    /// #2598 — this is where the drift had already happened, fatally. `classify`
    /// looked for
    ///
    /// ```text
    /// undamped criterion factorization hit a non-PD per-row H_tt block before KKT
    /// ```
    ///
    /// and not one of the three sites that raise the refusal emits it: the two
    /// probe arms say "undamped EVIDENCE factorization hit a non-PD per-row H_tt
    /// block before KKT stationarity", and the stationary arm says "stationary
    /// undamped criterion factorization HAS a non-PD per-row H_tt block that
    /// spectral unit-stiffness deflation could not condition". So `classify`
    /// answered `None` for every per-row refusal the crate is able to produce,
    /// `is_recoverable_value_probe_refusal` was false, and the outer optimizer
    /// aborted the fit at exactly the ρ all three call-site comments say it must
    /// read as +∞ and steer away from. `infeasible_non_pd_per_row` counted zero.
    ///
    /// The phrase below is the one common to all three producers, which is also
    /// what the pre-#2593 TELEMETRY ladder carried — #2593 collapsed two ladders
    /// into one and kept the wrong survivor. Every rendered message is unchanged
    /// byte for byte, because this string IS the substring it replaced.
    pub(crate) fn non_pd_per_row_marker() -> &'static str {
        "non-PD per-row H_tt block"
    }

    /// The phrase this crate writes to mark an `AllZeroGatedDesign` refusal.
    pub(crate) fn all_zero_gated_design_marker() -> &'static str {
        "gated off at every row (all-zero gated design)"
    }

    /// The phrase this crate writes to mark a `TotalCoCollapse` refusal.
    pub(crate) fn total_co_collapse_marker() -> &'static str {
        "did not escape total co-collapse"
    }

    /// Classify a rendered refusal, or `None` when it is a genuine defect.
    ///
    /// `None` is the fail-loud default: a message this does not recognise stays
    /// fatal and propagates, so a producer that rewords itself loses ρ-locality
    /// (one wasted seed) rather than having a real defect masked as +∞.
    pub(crate) fn classify(err: &str) -> Option<Self> {
        if err.contains(Self::inner_not_converged_marker()) {
            return Some(Self::InnerNotConverged);
        }
        if err.contains(Self::non_pd_per_row_marker()) {
            return Some(Self::NonPdPerRow);
        }
        // #1782 — at a seed ρ, a K>1 threshold-gate/softmax (or a rank-deficient
        // euclidean/linear) fit's OFF-OPTIMUM inner state can leave the
        // reduced joint-Hessian Schur complement indefinite, so the undamped
        // Schur-complement Cholesky in `run_joint_fit_arrow_schur` /
        // `converge_inner_for_undamped_logdet` refuses with
        // `ArrowSchurError::SchurFactorFailed` (rendered
        // "arrow-Schur: Schur complement Cholesky failed: … not positive
        // definite"). That is the SAME infeasible-ρ-probe class as the
        // per-row non-PD refusal above: the indefinite basin is
        // adjacent to the PD optimum, so the outer optimizer must read it as
        // +∞ and steer back into the PD region rather than reject the seed and
        // abort the whole fit ("no candidate seeds passed outer startup
        // validation"). `ordered_beta_bernoulli`+`circle`'s seed lands in the PD region and
        // never trips this, which is exactly why it converged on identical
        // data while the other assignments/topologies did not.
        //
        // Requires BOTH markers so a genuine shape / dimension / non-finite
        // Schur defect (a `SchurFactorFailed` whose reason is NOT a non-PD
        // pivot, e.g. "non-finite entry" or "non-square") still hard-errors
        // and is not silently masked as a recoverable probe.
        //
        // #2598 — that conjunct used to be two string literals HERE, matching
        // the `Display` impl of a type in ANOTHER crate. Rewording either
        // message in gam-solve reclassified every recoverable Schur refusal as
        // a fatal defect, silently, with nothing failing. The wording now lives
        // beside the wording it reads: `ArrowSchurError` owns both the
        // rendering and this reader, and its own
        // `rendered_verdict_matches_the_value_verdict_for_every_variant_2598`
        // pins the rendered reader to the value predicate
        // (`is_non_pd_schur_complement`) for every variant. Nothing about the
        // classification changes; what changes is that a reword can no longer
        // land without failing a test.
        //
        // This stays a parse rather than a match on the value because the spine
        // between the refusal and here is `Result<_, String>` — the rest of
        // #2598.
        if ArrowSchurError::rendered_is_non_pd_schur_complement(err) {
            return Some(Self::NonPdSchur);
        }
        // #2087 — at a seed ρ a K>1 threshold-gate assignment can give an
        // atom OFF at every row, so the sequential-deflation refit's gated design
        // `diag(a_·k)·Φ_k` is all-zero and the reduced joint problem is
        // rank-deficient with an undefined quasi-Laplace score — the SAME infeasible-ρ
        // class as the non-PD Schur / Hessian refusals above. `run_joint_fit_arrow_schur`
        // → `enforce_decoder_norm_guard` → `refit_decoder_sequential_deflation`
        // surfaces the DISTINCT "gated off at every row (all-zero gated design)"
        // marker (NOT the generic `solve_design_least_squares` "zero numerical rank",
        // which stays fatal for genuinely defective designs), so the outer solver
        // reads it as an infeasible trial and steers ρ back to where the gate
        // turns atoms on rather than treating it as a finite objective value
        // with "no candidate seeds passed outer startup validation".
        if err.contains(Self::all_zero_gated_design_marker()) {
            return Some(Self::AllZeroGatedDesign);
        }
        // #2089 — a ρ whose smoothing / sparsity penalty makes every gated
        // decoder numerically disappear, or produces #2362 structural
        // co-collapse, after the bounded reseed multi-start is a genuine
        // infeasibility of that ρ — the same class as the non-PD Hessian /
        // all-zero gated-design probes above. A neighbouring, weaker-penalty ρ admits a non-degenerate
        // fit, so the outer optimizer must read this as an infeasible trial
        // (+∞) and steer ρ back toward the feasible region
        // NOT abort the entire alpha="auto" search the first time a line search
        // overshoots into a co-collapsing ρ. Aborting there fails fits that have a
        // perfectly good feasible ρ the search had not yet reached; and letting
        // the reseed multi-start GRIND at every such probe (the pre-guard
        // behaviour) is exactly what thrashed the host to an OOM / watchdog
        // SIGKILL (exit 137). `run_joint_fit_arrow_schur` emits this DISTINCT
        // "did not escape total co-collapse" marker only after the reseed budget
        // is spent and same-state disappearance is still certified, so a
        // healthy or merely-uncompetitive fit never trips it.
        if err.contains(Self::total_co_collapse_marker()) {
            return Some(Self::TotalCoCollapse);
        }
        None
    }
}

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct OuterProbeTelemetry {
    /// Full penalized quasi-Laplace criterion evaluations requested through the generic outer
    /// lanes. Accepted gradient/EFS lanes commit their solved basin; value-only
    /// comparison probes restore the incumbent state before returning.
    pub criterion_calls: usize,
    /// Infeasible probes by refusal kind (non-PD Laplace log-det at that ρ).
    pub infeasible_non_pd_per_row: usize,
    pub infeasible_schur: usize,
    /// Probes refused because the inner solve did not converge at fixed ρ.
    pub infeasible_inner_not_converged: usize,
    /// Probes refused because a gate turned an atom off at every row, leaving
    /// the sequential-deflation refit an all-zero gated design (#2087).
    pub infeasible_all_zero_gated_design: usize,
    /// Probes refused because the reseed budget was spent and same-state
    /// decoder disappearance was still certified (#2089 / #2362).
    pub infeasible_total_co_collapse: usize,
    /// Outer criterion evaluations that returned the optimizer's conventional
    /// infeasible value (`+inf`) because the quasi-Laplace score was undefined or
    /// the fixed-ρ inner solve refused. A finite, data-collapsed fit is not an
    /// infeasible objective value: collapse remains a structural ledger verdict
    /// while penalized quasi-Laplace remains the sole optimized criterion.
    pub infeasible_criterion_evals: usize,
    /// Basin-bundle lower-envelope telemetry (see [`BasinBundle`]). The outer
    /// value lanes evaluate `V*(ρ) = min_b V_b(ρ)` over a memory-admitted bundle of saved
    /// inner basins instead of the single hysteretic warm-start trajectory
    /// (#2230/#2087). These counters make the envelope's work observable.
    ///
    /// `basin_envelope_evals` — value-lane evaluations that ran the envelope
    /// (dense-admitted, `inner_max_iter > 0`; the streaming / freeze bypass does
    /// not increment it). `basin_admissions` — distinct new basins admitted to
    /// the bundle across the fit (a growth event, not a duplicate-replace).
    /// `basin_envelope_rescues` — envelope evals where a SAVED basin beat the
    /// fresh discovery trajectory by more than the inner objective stall
    /// tolerance, i.e. where the single-trajectory criterion would have jumped
    /// UP across a basin boundary and the envelope held it down. `basin_max_members`
    /// — the largest bundle size reached. `basin_member_capacity` is the
    /// cgroup-aware host-memory admission bound; exhausting it refuses the fit
    /// rather than evicting a branch and returning an inexact envelope.
    pub basin_envelope_evals: usize,
    pub basin_admissions: usize,
    pub basin_envelope_rescues: usize,
    pub basin_max_members: usize,
    pub basin_member_capacity: usize,
    /// Scalar continuation waypoints installed before their rho-spine solve.
    /// Already-finite literal seeds must leave this at zero.
    pub reactive_scalar_installs: usize,
    /// Installed waypoints that were bit-identical to the objective's literal
    /// target scalar state.
    pub reactive_target_restores: usize,
}

impl OuterProbeTelemetry {
    /// Count one refused probe under its own kind.
    ///
    /// Reads [`ProbeRefusalKind`] rather than re-parsing the message. Before
    /// #2593 this was a SECOND independent substring ladder over the same
    /// string, and the two had already drifted apart in both directions:
    ///
    /// * `AllZeroGatedDesign` and `TotalCoCollapse` were recoverable but had no
    ///   counter here at all, so every such probe was invisible to
    ///   `infeasible_total()` — the load-bearing metric of #2080's bounded
    ///   probe-budget test, which therefore under-reported on exactly the
    ///   co-collapsing ρ that motivated the guard;
    /// * this ladder matched a bare "Schur complement Cholesky failed" while
    ///   the recoverability predicate required it AND "not positive definite",
    ///   so the two disagreed about what a Schur refusal even is.
    ///
    /// Deriving both from one classification is what makes that drift
    /// unrepresentable, and `every_refusal_kind_is_counted_exactly_once` fails
    /// if a future kind is added without a counter.
    fn record_refusal_kind(&mut self, err: &str) {
        let Some(kind) = ProbeRefusalKind::classify(err) else {
            return;
        };
        *self.counter_mut(kind) += 1;
    }

    fn counter_mut(&mut self, kind: ProbeRefusalKind) -> &mut usize {
        match kind {
            ProbeRefusalKind::InnerNotConverged => &mut self.infeasible_inner_not_converged,
            ProbeRefusalKind::NonPdPerRow => &mut self.infeasible_non_pd_per_row,
            ProbeRefusalKind::NonPdSchur => &mut self.infeasible_schur,
            ProbeRefusalKind::AllZeroGatedDesign => &mut self.infeasible_all_zero_gated_design,
            ProbeRefusalKind::TotalCoCollapse => &mut self.infeasible_total_co_collapse,
        }
    }

    /// Total infeasible probes across all refusal kinds.
    ///
    /// Sums over `ProbeRefusalKind::ALL` so a new kind is included here the
    /// moment it exists, instead of being silently omitted the way
    /// `AllZeroGatedDesign` and `TotalCoCollapse` were.
    pub fn infeasible_total(&self) -> usize {
        ProbeRefusalKind::ALL
            .iter()
            .map(|kind| self.counter_of(*kind))
            .sum()
    }

    fn counter_of(&self, kind: ProbeRefusalKind) -> usize {
        match kind {
            ProbeRefusalKind::InnerNotConverged => self.infeasible_inner_not_converged,
            ProbeRefusalKind::NonPdPerRow => self.infeasible_non_pd_per_row,
            ProbeRefusalKind::NonPdSchur => self.infeasible_schur,
            ProbeRefusalKind::AllZeroGatedDesign => self.infeasible_all_zero_gated_design,
            ProbeRefusalKind::TotalCoCollapse => self.infeasible_total_co_collapse,
        }
    }
}

/// #2080 (a) — probe→accepted warm-start handoff.
///
/// The generic outer line search evaluates its cost probes through the
/// value-only lane (`eval_with_order(Value)` / `eval_cost` →
/// `authoritative_envelope_value_probe` →
/// `evaluate_authoritative_value_probe`), each of which drives the FULL inner
/// `(t, β)` Newton solve to KKT convergence at the probed ρ — starting from the
/// accepted basin — and then RESTORES the accepted term, discarding that
/// converged state. The accepted point of a successful line search is always
/// the ρ of its last successful value probe, and the engine then re-evaluates
/// it through the gradient lane (`eval`), which historically re-ran the
/// identical deterministic inner convergence from the identical accepted
/// basin — a full redundant inner solve per outer iteration.
///
/// This handoff retains the probe's converged term (a move, not a clone: it is
/// swapped out against the restored saved term) keyed by the BITWISE probed ρ.
/// The next criterion-driving call TAKES it unconditionally — so it can never
/// survive past any other evaluation that might move the accepted basin — and
/// installs it as the inner warm start only when its ρ matches bitwise.
///
/// WHY THE CRITERION VALUE IS UNCHANGED: the Laplace criterion is defined at
/// the inner KKT optimum at the evaluated ρ (`converge_inner_for_undamped_logdet`
/// refuses to rank an off-optimum state). The probe reached that optimum by the
/// exact deterministic iteration sequence the accepted evaluation would have
/// re-run (same entry state — the accepted basin — same ρ, same solver
/// configuration), so installing the probe's converged state warm-starts the
/// accepted evaluation AT the same converged optimum; the criterion's own
/// convergence loop still runs (its KKT gate passes immediately) and the single
/// stationary factorization prices the same log|H|. Same converged optimum,
/// fewer iterations to reach it.
struct ProbeConvergedHandoff {
    /// Flattened ρ of the probe, compared BITWISE (`f64::to_bits`) so only an
    /// exact re-evaluation of the same probed point consumes the state.
    rho_flat: Array1<f64>,
    /// The probe's fully converged term state at `rho_flat`. The receiving
    /// evaluation still treats it as a warm start and independently checks the
    /// same KKT stationarity condition before pricing value or gradient.
    term: SaeManifoldTerm,
}

/// #2231 Inc-B (stage 1) — crosscoder block-relevance PRICING state.
///
/// When present, the outer objective prices the per-block relevance coordinates
/// `log λ_ℓ` (`SaeManifoldRho::log_lambda_block`): every eval lane rescales the
/// stacked target's output-block columns by `√λ_ℓ` at ρ-materialization, and the
/// criterion carries the change-of-variables Jacobian `−Σ_ℓ (n·p_ℓ/2)·log λ_ℓ`.
///
/// INVARIANT: the stacked target handed to [`SaeManifoldOuterObjective::new`] is
/// the UNSCALED augmented target (all `λ_ℓ = 1`); this state owns every `√λ_ℓ`
/// scaling thereafter, always rewriting a moved block FROM `pristine_blocks`
/// (never multiplicatively), so thousands of evals cannot drift.
#[derive(Clone)]
struct CrosscoderBlockPricing {
    /// Anchor width `p_x` — the leading `[0, p_x)` target columns, never scaled.
    p_x: usize,
    /// Per-output-block widths `p_ℓ`, length `L-1`, in stacked-column order and
    /// matching the ρ template's `log_lambda_block`.
    block_dims: Vec<usize>,
    /// PRISTINE (unscaled, `λ_ℓ = 1`) copy of the non-anchor target columns
    /// `[p_x, p̃)` — the drift-free source every `apply_block_scaling` rewrite
    /// reads from. Block `ℓ` occupies `[Σ_{m<ℓ} p_m, Σ_{m<ℓ} p_m + p_ℓ)` here.
    pristine_blocks: Array2<f64>,
    /// Last-applied per-block `log λ_ℓ` (length `L-1`). Seeded to `0` (`λ = 1`,
    /// the as-handed target), so `apply_block_scaling` rewrites a block only when
    /// its ρ `log λ` moves off the currently materialized value.
    last_log_lambda: Vec<f64>,
}

/// Full transactional checkpoint for one reactive coupled waypoint. The value
/// lane is a trial probe and may mutate routing/decoder state before refusing;
/// retaining only β would not restore the accepted basin.
struct ReactiveWaypointCheckpoint {
    term: SaeManifoldTerm,
    target: Array2<f64>,
    registry_isometry_weights: Vec<f64>,
    current_rho: SaeManifoldRho,
    last_loss: Option<SaeManifoldLoss>,
    terminal_penalized_quasi_laplace_criterion: Option<f64>,
    seeded_beta: Option<Array1<f64>>,
    probe_converged_handoff: Option<ProbeConvergedHandoff>,
    basin_bundle: BasinBundle<SaeManifoldTerm>,
    termination: OuterTerminationLedger,
    fit_verdict: Option<SaeOuterVerdict>,
    crosscoder_blocks: Option<CrosscoderBlockPricing>,
}

struct MatrixFreeOuterArtifacts {
    /// The `B` majorizer operator the exact-stationarity solve reassembles
    /// `A = B + ΔC` on top of.
    system: ArrowSchurSystem,
    /// #2515 — the factor cache of the exact observed information whose reduced
    /// Schur produced `logdet_derivative_bundle`. It travels WITH the bundle so
    /// the outer gradient cannot contract `A`'s `S⁻¹` against `B`'s row blocks.
    exact_a_cache: ArrowFactorCache,
    logdet_derivative_bundle: RationalLogdetDerivativeBundle,
    efs_inverse_probe_bundle: Option<(Vec<Array1<f64>>, Vec<Array1<f64>>)>,
}

pub(crate) struct OuterCriterionEvaluation {
    pub(crate) cost: f64,
    loss: SaeManifoldLoss,
    cache: ArrowFactorCache,
    matrix_free: Option<MatrixFreeOuterArtifacts>,
}

pub struct SaeManifoldOuterObjective {
    pub(crate) term: SaeManifoldTerm,
    /// Pristine term to restore from on `reset` (multi-start baseline).
    pub(crate) baseline_term: SaeManifoldTerm,
    pub(crate) target: Array2<f64>,
    pub(crate) registry: Option<AnalyticPenaltyRegistry>,
    /// Literal isometry weights owned by the real objective. Reactive scalar
    /// continuation may temporarily loosen them, and every reset restores this
    /// exact vector.
    baseline_isometry_weights: Vec<f64>,
    /// ρ template carrying the per-atom ARD dims; `from_flat` reads its
    /// layout. Updated to each evaluated ρ so `into_fitted` can report the
    /// last ρ the engine settled on.
    pub(crate) current_rho: SaeManifoldRho,
    /// Pristine ρ to restore from on `reset`.
    pub(crate) baseline_rho: SaeManifoldRho,
    pub(crate) inner_max_iter: usize,
    pub(crate) learning_rate: f64,
    pub(crate) ridge_ext_coord: f64,
    pub(crate) ridge_beta: f64,
    /// Last inner loss breakdown observed (for `into_fitted`).
    pub(crate) last_loss: Option<SaeManifoldLoss>,
    /// Full criterion value is stamped only by a fixed-rho certificate or a
    /// certified outer result; ordinary diagnostic evaluations do not mint it.
    pub(crate) terminal_penalized_quasi_laplace_criterion: Option<f64>,
    /// Optional warm-start β slot. When the cache / continuation walk seeds a
    /// β, the next inner solve opens from it instead of cold.
    pub(crate) seeded_beta: Option<Array1<f64>>,
    /// #1207 — running tally of amortized warm-start outcomes, so a silent cold
    /// fallback is observable instead of hidden behind `.ok()`.
    pub(crate) warm_start_telemetry: AmortizedWarmStartTelemetry,
    /// #1033 — when set, the term's assignment ROUTING is frozen (amortized): the
    /// gates are pinned to a ρ-invariant predicted routing once before the ρ-search
    /// and the inner solve never re-optimizes the logits, so every outer ρ
    /// evaluation reuses ONE routing instead of re-solving the per-row gates. OFF
    /// by default — the historical free-logit ρ-search is unchanged. This is the
    /// opt-in lever for the n-independent outer loop; the n-scaling timing is
    /// verified on the cluster.
    /// #2080 — outer probe telemetry (criterion/infeasible counts). Read via
    /// [`Self::probe_telemetry`] after the fit for the wide-`p` acceptance test.
    pub(crate) probe_telemetry: OuterProbeTelemetry,
    /// #2138 — cooperative cancellation. When the pyffi fit driver sets this after
    /// a Python interrupt, the next `eval`/`eval_cost` returns a recoverable
    /// `RemlOptimizationFailed` so an abandoned worker thread unwinds and stops
    /// rather than running a hung fit to completion. `None` ⇒ historical path.
    pub(crate) cancel_flag: Option<std::sync::Arc<std::sync::atomic::AtomicBool>>,
    /// #2080 (a) — the last successful value probe's converged inner state (see
    /// [`ProbeConvergedHandoff`]). Single-shot: taken by the next
    /// criterion-driving call and cleared by every state-swapping seam
    /// (`reset`, seed installation, subsample engage/restore, homotopy entry).
    probe_converged_handoff: Option<ProbeConvergedHandoff>,
    /// #2080 — the frozen per-outer-solve rational log-det surrogate lane. When
    /// the streaming criterion takes the matrix-free massive-K evidence branch
    /// (dense `k×k` reduced Schur over budget), the `log|S|` term is estimated by
    /// this desync-safe rational surrogate instead of SLQ; below that branch it
    /// stays dormant (`plan == None`), so small/dense fits are byte-unchanged. The
    /// lane self-heals across basin mutations (its plan rebuilds when the reduced-
    /// Schur dimension changes), so it is NOT cleared with `probe_converged_handoff`.
    surrogate_lane: Option<SurrogateLaneState>,
    /// #2230/#2087 — the basin lower-envelope bundle. The historical outer
    /// criterion is the hysteretic single-trajectory value `V_{b(warm,ρ)}(ρ)`:
    /// whichever inner basin the warm-started solve at ρ happens to land in. That
    /// value JUMPS at basin-boundary crossings, which is the measured pathology
    /// (hours of `[#1026] restoring inner-fit reconstruction incumbent` churn = the
    /// outer line search oscillating across a boundary it cannot represent). This
    /// bundle holds a small set of saved converged inner basins; every value-lane
    /// evaluation re-converges each member from its own state (warm ⇒ cheap), plus
    /// runs the one historical discovery trajectory, and returns the MINIMUM —
    /// the continuous, piecewise-smooth lower envelope `V*(ρ) = min_b V_b(ρ)`. The
    /// argmin basin's converged state is handed to the gradient lane (via
    /// `probe_converged_handoff`), so the accepted point's analytic λ-gradient
    /// prices the argmin basin (envelope theorem, exact a.e.). Admitting a basin
    /// can only LOWER the envelope, so discovery strictly improves the criterion
    /// surface. Cleared with `probe_converged_handoff` at every accepted-basin /
    /// row-support seam; bypassed in the streaming and `inner_max_iter == 0`
    /// freeze regimes (see `authoritative_envelope_value_probe`).
    basin_bundle: BasinBundle<SaeManifoldTerm>,
    /// #2235 — outer termination ledger (verdicts: engine-stopped / incumbent-
    /// stationary / budget-exhausted). Freezes the criterion once a verdict
    /// fires so the bridge converges onto the banked incumbent.
    pub(crate) termination: OuterTerminationLedger,
    /// Explicit proof of which converged optimization owns the currently
    /// installed `(term, rho, loss)` state. `None` means UNCERTIFIED, not
    /// fixed-rho: ordinary objective evaluations may populate `last_loss`, but
    /// only [`Self::fit_at_fixed_rho`] or [`Self::certify_outer_result`] may
    /// stamp a fit-producing verdict.
    fit_verdict: Option<SaeOuterVerdict>,
    /// True only while auditing a caller-installed external state (#2263). The
    /// ordinary analytic outer evaluation warm-starts coordinates from the
    /// amortized encoder before refining; that is an optimization action and is
    /// forbidden when the subject is the exact installed state.
    audit_installed_state: bool,
    /// SPEC wall-survival: the full-`N` data fingerprint + content-addressed
    /// store path for the fit checkpoint (see [`super::checkpoint`]). Computed
    /// once at construction on the full-data target. Checkpoints are written
    /// best-effort at every MATERIAL improvement of the outer best cost, and the
    /// file is removed when a converged fit is minted (its purpose is wall
    /// survival, not cross-fit caching — `persistent_warm_start` covers that).
    pub(crate) checkpoint_fingerprint: super::checkpoint::SaeCheckpointFingerprint,
    pub(crate) checkpoint_path: std::path::PathBuf,
    /// #2231 Inc-B (stage 1) — optional crosscoder block-relevance pricing. `None`
    /// for a plain SAE, in which case `apply_block_scaling`/`block_jacobian` both
    /// early-return and every lane is byte-identical to the historical path.
    /// Installed by [`Self::with_crosscoder_blocks`].
    crosscoder_blocks: Option<CrosscoderBlockPricing>,
    /// Present only while one reactive coupled waypoint is being evaluated.
    /// Success commits the probe handoff; failure restores this full snapshot.
    reactive_waypoint_checkpoint: Option<ReactiveWaypointCheckpoint>,
}

/// #2230/#2087 exact basin-bundle memory admission.
///
/// A present-value or work-count rule cannot prove global dominance of one
/// basin over another, so an admitted branch is never evicted. Retained states
/// live on the host. Reserve one conservative direct-solve peak for the active
/// criterion evaluation, then charge every saved state another full direct-solve
/// peak even though a cloned term contains only a subset of that workspace. The
/// resulting capacity is deliberately conservative and comes from the same
/// cgroup-aware host budget as the SAE streaming plan. Reaching it is an
/// explicit feasibility error from `BasinBundle::admit`, not an inexact envelope.
fn basin_bundle_member_capacity(term: &SaeManifoldTerm) -> usize {
    // #2560 — derive from the reading the term captured once at construction,
    // not from a fresh probe. Available memory moves with every other process
    // on the box, and this function previously probed twice: the plan below was
    // sized against one reading and the capacity against another, so a
    // neighbouring allocation between them changed the answer.
    let host_available = term.host_available_bytes;
    let host_budget = super::sae_host_in_core_budget_from_available(host_available);
    let total_basis: usize = term.atoms.iter().map(SaeManifoldAtom::basis_size).sum();
    let d_max = term
        .atoms
        .iter()
        .map(SaeManifoldAtom::latent_dim)
        .max()
        .unwrap_or(0);
    let border_dim = if term.any_frame_active() {
        term.factored_border_dim()
    } else {
        term.beta_dim()
    };
    let plan = super::sae_streaming_plan_from_budget(
        term.n_obs(),
        total_basis,
        term.k_atoms(),
        d_max,
        border_dim,
        host_budget,
        super::SAE_CPU_L2_CACHE_BYTES * super::SAE_CHUNK_CACHE_MULTIPLE,
        host_available,
    );
    if !plan.direct_logdet_admitted() {
        return 0;
    }
    let bytes_per_saved_state = plan
        .estimated_direct_peak_bytes
        .max(plan.estimated_full_batch_bytes)
        .max(std::mem::size_of::<SaeManifoldTerm>());
    host_budget.saturating_sub(plan.estimated_direct_peak_bytes) / bytes_per_saved_state
}

/// Both admitted evidence routes expose the complete analytic joint-Hessian IFT
/// gradient. The streaming route uses the frozen rational-logdet probe bundle
/// and one matrix-free adjoint solve; it never manufactures a zero derivative or
/// retries through the dense factorization.
pub(crate) fn sae_outer_gradient_capability() -> Derivative {
    Derivative::Analytic
}

/// The assignment-strength coordinate handled by Hybrid-EFS's full analytic
/// criterion gradient. Every assignment family with a structurally present
/// sparse coordinate uses this lane, including learnable ordered Beta--Bernoulli
/// concentration: its prior value, inner-mode response, and log-determinant
/// derivative must be differentiated as one penalized quasi-Laplace scalar.
pub(crate) fn assignment_strength_gradient_coordinate(rho: &SaeManifoldRho) -> Option<usize> {
    rho.sparse_flat_index()
}

/// #2080 surrogate-lane policy (SAE side) for the derived-rank rational `log|S|`
/// surrogate that supersedes SLQ on the matrix-free massive-K criterion path.
/// Probe count and seed mirror the SLQ lane it replaces; the deflation target is
/// one order under the inner-objective stall tolerance — `log|S|` is the
/// criterion's dominant term at wide `k`, so the Hutchinson error bar must sit
/// well inside the tolerance that certifies the ρ-search stationary.
///
/// This is the SAE evidence surrogate's ONE policy. The support-sparse grouped
/// LAML lane reads it too (`support_outer::evidence_log_det`): both criteria
/// take a matrix-free `log|S|` and differentiate it, so a second policy beside
/// this one would be two answers to the same question (#2576, #2470).
const SAE_SURROGATE_LANE_QUADRATURE_REL_TOL: f64 = 1.0e-8;
const SAE_SURROGATE_LANE_POWER_ITERS: usize = 40;
const SAE_SURROGATE_LANE_CG_REL_TOL: f64 = 1.0e-8;
const SAE_SURROGATE_LANE_CG_MAX_ITERS: usize = 20_000;
const SAE_SURROGATE_LANE_DEFLATION_MAX_RANK: usize = 128;
const SAE_SURROGATE_LANE_DEFLATION_SUBSPACE_ITERS: usize = 4;

pub(crate) fn sae_surrogate_lane_config() -> SurrogateLaneConfig {
    SurrogateLaneConfig {
        num_probes: SCHUR_SLQ_LOGDET_PROBES,
        seed: SCHUR_SLQ_LOGDET_SEED,
        rel_tol: SAE_SURROGATE_LANE_QUADRATURE_REL_TOL,
        power_iters: SAE_SURROGATE_LANE_POWER_ITERS,
        cg_rel_tol: SAE_SURROGATE_LANE_CG_REL_TOL,
        cg_max_iters: SAE_SURROGATE_LANE_CG_MAX_ITERS,
        deflation_max_rank: SAE_SURROGATE_LANE_DEFLATION_MAX_RANK,
        deflation_subspace_iters: SAE_SURROGATE_LANE_DEFLATION_SUBSPACE_ITERS,
        deflation_target_std_err_rel: 0.1 * SAE_MANIFOLD_INNER_OBJECTIVE_STALL_REL_TOL,
    }
}

impl SaeManifoldOuterObjective {
    fn curvature_seed(term: &SaeManifoldTerm) -> Vec<(usize, f64)> {
        term.atoms
            .iter()
            .enumerate()
            .filter_map(|(atom, value)| {
                value
                    .geometry_plan()
                    .and_then(SaeAtomGeometryPlan::constant_curvature)
                    .map(|kappa| (atom, kappa))
            })
            .collect()
    }

    /// Install every curvature-dependent reference Gram for this outer point.
    /// All fallible geometry work completes before the first atom is changed,
    /// so an invalid trial curvature is a clean domain refusal rather than a
    /// partially-mutated dictionary.
    fn apply_curvature_state(&mut self, rho: &SaeManifoldRho) -> Result<(), String> {
        let mut prepared = Vec::with_capacity(rho.kappa.len());
        for (&atom_index, &kappa) in rho.kappa_atoms.iter().zip(rho.kappa.iter()) {
            let atom = self.term.atoms.get(atom_index).ok_or_else(|| {
                format!(
                    "curvature coordinate names atom {atom_index}, outside K={}",
                    self.term.atoms.len()
                )
            })?;
            let already_installed = atom
                .geometry_plan()
                .and_then(SaeAtomGeometryPlan::constant_curvature)
                .is_some_and(|current| current.to_bits() == kappa.to_bits())
                && atom.smooth_penalty_kappa_derivative().is_some();
            if !already_installed {
                prepared.push((atom_index, atom.prepare_constant_curvature(kappa)?));
            }
        }
        for (atom_index, state) in prepared {
            self.term.atoms[atom_index].commit_prepared_constant_curvature(state);
        }
        Ok(())
    }

    /// Flat indices and scale-equivariant raw-curvature rails, derived from the
    /// same typed geometry plans that build `S(kappa)` and `dS/dkappa`.
    fn curvature_domain_bounds(&self) -> Result<Vec<(usize, f64, f64)>, EstimationError> {
        let mut out = Vec::with_capacity(self.baseline_rho.kappa.len());
        for &atom_index in &self.baseline_rho.kappa_atoms {
            let flat = self
                .baseline_rho
                .kappa_flat_index(atom_index)
                .ok_or_else(|| {
                    EstimationError::InvalidInput(format!(
                        "curvature atom {atom_index} has no flat outer coordinate"
                    ))
                })?;
            let atom = self.baseline_term.atoms.get(atom_index).ok_or_else(|| {
                EstimationError::InvalidInput(format!(
                    "curvature coordinate names atom {atom_index}, outside K={}",
                    self.baseline_term.atoms.len()
                ))
            })?;
            let (lower, upper) = atom
                .geometry_plan()
                .ok_or_else(|| {
                    EstimationError::InvalidInput(format!(
                        "curvature atom {atom_index} has no typed geometry plan"
                    ))
                })?
                .constant_curvature_domain()
                .map_err(EstimationError::InvalidInput)?
                .ok_or_else(|| {
                    EstimationError::InvalidInput(format!(
                        "atom {atom_index} owns a curvature coordinate but its metric is not constant-curvature"
                    ))
                })?;
            out.push((flat, lower, upper));
        }
        Ok(out)
    }

    pub(crate) fn current_rho_flat(&self) -> Array1<f64> {
        self.current_rho.to_flat()
    }

    /// Re-evaluate one committed terminal coordinate with the idempotent dense
    /// or streaming criterion and return its typed vanished-atom boundary, if
    /// any. Speculative line-search probes continue to see disappearance as an
    /// infeasible trial; only this stage-owner call may turn the terminal
    /// coordinate into a change of model dimension.
    pub(crate) fn vanished_stage_state_at(
        &self,
        rho_flat: ArrayView1<'_, f64>,
    ) -> Result<Option<SaeVanishedStageState>, String> {
        let rho = self.baseline_rho.from_flat(rho_flat)?;
        let mut term = self.term.clone();
        let evaluated = if term.streaming_plan()?.direct_logdet_admitted() {
            term.penalized_quasi_laplace_criterion_with_cache(
                self.target.view(),
                &rho,
                self.registry.as_ref(),
                self.inner_max_iter,
                self.learning_rate,
                self.ridge_ext_coord,
                self.ridge_beta,
            )
        } else {
            term.penalized_quasi_laplace_criterion_streaming_exact_with_cache(
                self.target.view(),
                &rho,
                self.registry.as_ref(),
                self.inner_max_iter,
                self.learning_rate,
                self.ridge_ext_coord,
                self.ridge_beta,
            )
        };
        let atoms = match evaluated {
            Ok(_) => return Ok(None),
            Err(err @ SaeCriterionError::IndefiniteObservedInformation { .. }) => {
                return Err(err.to_string());
            }
            Err(SaeCriterionError::Numerical(message)) => return Err(message),
            Err(SaeCriterionError::VanishedAtoms(atoms)) => atoms,
        };

        let vanished = atoms.as_btree_set();
        for atlas in term.chart_atlases() {
            let removed = atlas
                .charts()
                .iter()
                .filter(|chart| vanished.contains(chart))
                .count();
            if removed > 0 && removed < atlas.charts().len() {
                return Err(format!(
                    "vanished-atom boundary would partially delete live atlas {:?}; \
                     chart-atlas disappearance must be adjudicated at semantic-atlas granularity",
                    atlas.charts()
                ));
            }
        }
        Ok(Some(SaeVanishedStageState { term, rho, atoms }))
    }

    pub fn new(
        mut term: SaeManifoldTerm,
        target: Array2<f64>,
        registry: Option<AnalyticPenaltyRegistry>,
        init_rho: SaeManifoldRho,
        inner_max_iter: usize,
        learning_rate: f64,
        ridge_ext_coord: f64,
        ridge_beta: f64,
    ) -> Self {
        // The objective owns the typed flat layout. Bind assignment-strength
        // presence to the actual term so K=1 Softmax and hard TopK cannot enter
        // as held/frozen rho coordinates through a manually constructed seed.
        let init_rho = init_rho
            .for_assignment(term.assignment.mode)
            .with_curvature(Self::curvature_seed(&term));
        term.expected_criterion_gauge_deflated_directions = None;
        term.criterion_gauge_deflation_reanchors = 0;
        term.criterion_gauge_deflation_last_delta_sign = 0;
        term.dictionary_cocollapse_reseeds = 0;
        term.best_cocollapse_incumbent = None;
        term.structural_cocollapse_reseeds = 0;
        let baseline_term = term.clone();
        let baseline_rho = init_rho.clone();
        let baseline_isometry_weights = registry
            .as_ref()
            .map(AnalyticPenaltyRegistry::isometry_scalar_weights)
            .unwrap_or_default();
        let term_k_atoms = term.k_atoms();
        let basin_member_capacity = basin_bundle_member_capacity(&term);
        // SPEC wall-survival fingerprint on the full-data target.
        let checkpoint_fingerprint =
            super::checkpoint::SaeCheckpointFingerprint::of_target(target.view(), term_k_atoms);
        let checkpoint_path =
            super::checkpoint::SaeFitCheckpoint::default_store_path(&checkpoint_fingerprint);
        Self {
            term,
            baseline_term,
            target,
            registry,
            baseline_isometry_weights,
            current_rho: init_rho,
            baseline_rho,
            inner_max_iter,
            learning_rate,
            ridge_ext_coord,
            ridge_beta,
            last_loss: None,
            terminal_penalized_quasi_laplace_criterion: None,
            seeded_beta: None,
            warm_start_telemetry: AmortizedWarmStartTelemetry::default(),
            probe_telemetry: OuterProbeTelemetry::default(),
            cancel_flag: None,
            probe_converged_handoff: None,
            surrogate_lane: Some(SurrogateLaneState::new(sae_surrogate_lane_config())),
            basin_bundle: BasinBundle::new(basin_member_capacity),
            // #2235 — outer-search accounting + the non-convergence forcing
            // function (stationarity defect raises a typed error; a fit object
            // only ever exists from a converged optimization).
            termination: OuterTerminationLedger::new(),
            fit_verdict: None,
            audit_installed_state: false,
            checkpoint_fingerprint,
            checkpoint_path,
            crosscoder_blocks: None,
            reactive_waypoint_checkpoint: None,
        }
    }

    /// Evaluate one converged outer sample through the selected storage route.
    /// The streaming variant returns the exact matrix-free operator and frozen
    /// selected-inverse bundle that produced the value; callers must consume
    /// them together or reject the sample.
    pub(crate) fn evaluate_outer_criterion_route(
        &mut self,
        rho: &SaeManifoldRho,
        direct_logdet_admitted: bool,
        need_efs_inverse_probes: bool,
    ) -> Result<OuterCriterionEvaluation, SaeCriterionError> {
        if direct_logdet_admitted {
            let (cost, loss, cache) = self.term.penalized_quasi_laplace_criterion_with_cache(
                self.target.view(),
                rho,
                self.registry.as_ref(),
                self.inner_max_iter,
                self.learning_rate,
                self.ridge_ext_coord,
                self.ridge_beta,
            )?;
            return Ok(OuterCriterionEvaluation {
                cost,
                loss,
                cache,
                matrix_free: None,
            });
        }

        let lane = self.surrogate_lane.as_mut().ok_or_else(|| {
            SaeCriterionError::Numerical(
                "streaming outer evaluation requires the frozen rational-logdet surrogate lane"
                    .to_string(),
            )
        })?;
        let evaluated = self
            .term
            .penalized_quasi_laplace_streaming_outer_evaluation(
                self.target.view(),
                rho,
                self.registry.as_ref(),
                self.inner_max_iter,
                self.learning_rate,
                self.ridge_ext_coord,
                self.ridge_beta,
                lane,
                need_efs_inverse_probes,
            )?;
        Ok(OuterCriterionEvaluation {
            cost: evaluated.cost,
            loss: evaluated.loss,
            cache: evaluated.cache,
            matrix_free: Some(MatrixFreeOuterArtifacts {
                system: evaluated.system,
                exact_a_cache: evaluated.exact_a_cache,
                logdet_derivative_bundle: evaluated.logdet_derivative_bundle,
                efs_inverse_probe_bundle: evaluated.efs_inverse_probe_bundle,
            }),
        })
    }

    /// Complete analytic derivative of the exact value represented by
    /// `evaluation`. Dense and streaming storage differ only in their inverse
    /// action; all explicit, trace, Occam, rank-response, and single-adjoint IFT
    /// channels are assembled by the same authority.
    pub(crate) fn analytic_gradient_for_outer_evaluation(
        &self,
        rho: &SaeManifoldRho,
        evaluation: &OuterCriterionEvaluation,
    ) -> Result<Array1<f64>, OuterGradientError> {
        let components = if let Some(matrix_free) = evaluation.matrix_free.as_ref() {
            let derivative_vectors = &matrix_free.logdet_derivative_bundle.vectors;
            let solver = DeflatedArrowSolver::plain(&evaluation.cache);
            self.term
                .analytic_outer_rho_gradient_components_with_bundle(
                    self.target.view(),
                    rho,
                    &evaluation.loss,
                    &evaluation.cache,
                    &solver,
                    // #2515 — the ranked criterion on this lane is
                    // `½log|S_A| + rank_charge` (`rank_adjusted_quasi_laplace_-
                    // complexity` takes `½(log_det − log_det_tt)` and BOTH come
                    // off `exact_a_evidence_system`, so the per-row t-block
                    // log-dets cancel and the reduced Schur of `A` is the whole
                    // operator exposure). Its derivative is therefore
                    // `½tr(S_A⁻¹ ∂S_A/∂ρ)`, which the from-probes channels
                    // reconstruct only if the row geometry and the `S⁻¹` come from
                    // the same operator. `cache` stays `B`: it is the Newton/IFT
                    // scale that `solve_exact_stationarity_matrix_free` rebuilds
                    // `A = B + ΔC` on top of, and promoting it would double-count
                    // `ΔC`.
                    Some(BundleEvidenceGeometry {
                        operator: EvidenceOperator::ExactObservedInformation,
                        cache: &matrix_free.exact_a_cache,
                        probes: derivative_vectors,
                        sinv: derivative_vectors,
                    }),
                    Some(&matrix_free.system),
                )?
        } else {
            let lambda_smooth = rho
                .lambda_smooth_vec()
                .map_err(OuterGradientError::internal)?;
            let solver = self
                .term
                .outer_gradient_arrow_solver(&evaluation.cache, &lambda_smooth)?;
            self.term
                .analytic_outer_rho_gradient_components_with_bundle(
                    self.target.view(),
                    rho,
                    &evaluation.loss,
                    &evaluation.cache,
                    &solver,
                    None,
                    None,
                )?
        };
        let mut gradient = components.gradient();
        if let Some(block_grad) = self
            .block_log_lambda_gradient(rho)
            .map_err(OuterGradientError::internal)?
        {
            // The block weights are NOT the last sub-vector any more: #2604
            // appends per-atom curvature after them. Locating the block tail by
            // `len - block_len` was correct only while it was last, and would
            // silently write the block gradient into the curvature slots for any
            // dictionary carrying both. Subtract every tail that follows it.
            let trailing = rho.kappa.len();
            let tail = gradient.len() - trailing - block_grad.len();
            for (block, value) in block_grad.into_iter().enumerate() {
                gradient[tail + block] += value;
            }
        }
        Ok(gradient)
    }

    /// #2231 Inc-B (stage 1) — enable crosscoder block-relevance PRICING.
    ///
    /// `p_x` is the anchor width (leading `[0, p_x)` target columns, never
    /// scaled); `block_dims` are the `L-1` output-block widths in stacked-column
    /// order. Snapshots a PRISTINE (unscaled) copy of the non-anchor columns
    /// `[p_x, p̃)` — the drift-free source every `apply_block_scaling` reads —
    /// and seeds the last-applied `log λ_ℓ` to `0` (`λ = 1`, matching the target
    /// as handed in per the `CrosscoderBlockPricing` invariant).
    ///
    /// Validation (typed `String` error):
    /// - `p_x + Σ block_dims == target.ncols()` (the stacked augmented width);
    /// - `block_dims.len() == baseline_rho.log_lambda_block.len()` (the ρ
    ///   template's block coordinate count);
    /// - the outer row-subsample (`row_loss_weights`, #991 designed subsample)
    ///   must NOT be engaged: the pristine block copy would have to be restricted
    ///   to the sampled rows and the Jacobian's `n` reduced to the effective
    ///   sample size — deferred to a later stage, so refuse loudly here rather
    ///   than price on a full-`N` pristine copy that desyncs from a subsampled
    ///   fit target.
    ///
    /// A plain SAE never calls this (leaving `crosscoder_blocks == None`), so an
    /// empty `block_dims` is rejected — it would carry no coordinates to price.
    pub fn with_crosscoder_blocks(
        mut self,
        p_x: usize,
        block_dims: Vec<usize>,
    ) -> Result<Self, String> {
        if p_x == 0 {
            return Err("with_crosscoder_blocks: anchor width p_x must be non-zero".to_string());
        }
        if block_dims.is_empty() {
            return Err(
                "with_crosscoder_blocks: block_dims is empty — a plain SAE must not install \
                 crosscoder pricing (leave crosscoder_blocks = None)"
                    .to_string(),
            );
        }
        let block_total: usize = block_dims.iter().sum();
        let p_tot = self.target.ncols();
        if p_x + block_total != p_tot {
            return Err(format!(
                "with_crosscoder_blocks: p_x ({p_x}) + Σ block_dims ({block_total}) = {} \
                 must equal the stacked target width p̃ = {p_tot}",
                p_x + block_total
            ));
        }
        let template_blocks = self.baseline_rho.log_lambda_block.len();
        if block_dims.len() != template_blocks {
            return Err(format!(
                "with_crosscoder_blocks: block_dims length ({}) must match the ρ template's \
                 log_lambda_block count ({template_blocks})",
                block_dims.len()
            ));
        }
        if self.term.row_loss_weights.is_some() {
            return Err(
                "with_crosscoder_blocks: the outer row-subsample (row_loss_weights, #991) is \
                 engaged; block pricing needs the pristine copy restricted to the sampled rows \
                 and the Jacobian n set to the effective sample size — deferred (stage 1)"
                    .to_string(),
            );
        }
        // #2231 Inc C — border-growth admission at the stacked width p̃. The
        // row-count admissions are already correct at output_dim = p̃, but the
        // arrow-Schur border is the one quantity QUADRATIC in the layer count
        // (beta_dim = Σ M_k·p̃ through the beta_dim² Hessian workspace); the
        // framed border (factored_border_dim) is p̃-independent. Admit the
        // border this fit will actually carry; the refusal names the frame
        // default as the remedy instead of silently narrowing the target.
        let (budget_bytes, _) = super::sae_host_in_core_budget_bytes();
        crate::front_door::admit_crosscoder_border(
            self.term.factored_border_dim(),
            self.term.beta_dim(),
            budget_bytes,
        )?;
        let pristine_blocks = self.target.slice(s![.., p_x..]).to_owned();
        // Mirror the spans onto the term so the outer-ρ gradient assembler can
        // build the block coordinates' IFT RHS (the −½·Γᵀθ̂_ρ adjoint channel
        // completing the analytic block gradient).
        self.term.crosscoder_pricing_spans = Some((p_x, block_dims.clone()));
        self.crosscoder_blocks = Some(CrosscoderBlockPricing {
            p_x,
            last_log_lambda: vec![0.0; block_dims.len()],
            block_dims,
            pristine_blocks,
        });
        Ok(self)
    }

    /// #2231 Inc-B (stage 1) — rewrite the stacked target's output-block columns
    /// to `√λ_ℓ · Y_ℓ` for the ρ under evaluation, reading each moved block FROM
    /// `pristine_blocks` (idempotent, drift-free). A block is rewritten only when
    /// its `log λ_ℓ` differs from the last materialized value, so a re-evaluation
    /// at the same ρ is a no-op. NO-OP entirely when crosscoder pricing is off
    /// (plain SAE byte-identity). Called at the ρ-materialization point of every
    /// `&mut self` eval lane so no inner solve ever reads a stale-scaled target.
    fn apply_block_scaling(&mut self, rho: &SaeManifoldRho) -> Result<(), String> {
        self.term.assignment.validate_rho_domain(rho)?;
        self.apply_curvature_state(rho)?;
        // Disjoint field borrows: the pricing state and the target are rewritten
        // together, so destructure `self` rather than route through a `self`
        // method that would alias both.
        let Self {
            target,
            crosscoder_blocks: Some(blocks),
            ..
        } = self
        else {
            return Ok(());
        };
        // The builder pinned `block_dims.len() == log_lambda_block.len()`; guard
        // defensively so a mismatched ρ can never scale a wrong column range.
        if rho.log_lambda_block.len() != blocks.block_dims.len() {
            return Err(format!(
                "crosscoder block log-strength count {} != pricing block count {}",
                rho.log_lambda_block.len(),
                blocks.block_dims.len()
            ));
        }
        // Collect the moved blocks' pristine column spans + scales first, then
        // rewrite in ONE parallel row pass over contiguous row slices. The
        // former column-by-column walk touched a stride-p̃ element every access
        // (a cache/TLB miss per element on a row-major target) and made two
        // passes (assign, then scale); large-width crosscoders paid that on
        // every outer ρ evaluation. The row-major fused copy is the
        // memcpy-speed version of the same idempotent pristine→target rewrite.
        let mut moved: Vec<(usize, usize, f64)> = Vec::new(); // (pristine_off, p_l, √λ)
        let mut pristine_off = 0usize;
        for l in 0..blocks.block_dims.len() {
            let p_l = blocks.block_dims[l];
            let new_ll = rho.log_lambda_block[l];
            if new_ll != blocks.last_log_lambda[l] {
                moved.push((pristine_off, p_l, (0.5 * new_ll).exp()));
                blocks.last_log_lambda[l] = new_ll;
            }
            pristine_off += p_l;
        }
        if moved.is_empty() {
            return Ok(());
        }
        let p_x = blocks.p_x;
        let pristine = &blocks.pristine_blocks;
        use rayon::prelude::*;
        target
            .axis_iter_mut(ndarray::Axis(0))
            .into_par_iter()
            .zip(pristine.axis_iter(ndarray::Axis(0)).into_par_iter())
            .for_each(|(mut dst_row, src_row)| {
                let src = src_row
                    .to_slice()
                    .expect("pristine block rows are contiguous");
                let dst = dst_row
                    .as_slice_mut()
                    .expect("stacked target rows are contiguous");
                for &(off, p_l, sqrt_lambda) in &moved {
                    let dst_span = &mut dst[p_x + off..p_x + off + p_l];
                    let src_span = &src[off..off + p_l];
                    for (d, &s) in dst_span.iter_mut().zip(src_span) {
                        *d = s * sqrt_lambda;
                    }
                }
            });
        Ok(())
    }

    /// #2231 Inc-B (stage 1) — the block-relevance change-of-variables Jacobian
    /// added to every eval lane's final cost BEFORE `termination.record`.
    ///
    /// Derivation: the outer criterion is the UNIT-dispersion penalized Laplace
    /// form (`#F1` — no `φ̂` factor; `loss.data_fit` is the raw half-SSE of the
    /// fit to the stacked target). Scaling output block `ℓ`'s target columns by
    /// `√λ_ℓ` is a change of variables `Y_ℓ ↦ √λ_ℓ·Y_ℓ` over `n·p_ℓ` entries;
    /// its log-Jacobian contributes `−(n·p_ℓ/2)·log λ_ℓ` to the criterion (the
    /// `√λ_ℓ = exp(½ log λ_ℓ)` per entry, `n·p_ℓ` entries). Summed over the
    /// `L-1` output blocks,
    ///
    /// ```text
    /// block_jacobian(ρ) = −Σ_ℓ (n·p_ℓ/2)·log λ_ℓ.
    /// ```
    ///
    /// With the scaled-block residual `R_ℓ` flowing through the half-SSE data
    /// term, `∂C/∂log λ_ℓ = ½·λ_ℓ·R_ℓ − n·p_ℓ/2`, stationary at
    /// `λ_ℓ = n·p_ℓ/R_ℓ` and coercive at both ends (`λ→0` the Jacobian wall
    /// `+∞`, `λ→∞` the scaled residual `+∞`) — the interior minimum the Inc-B
    /// contract pins assert. Returns `0` when crosscoder pricing is off (plain
    /// SAE byte-identity).
    fn block_jacobian(&self, rho: &SaeManifoldRho) -> f64 {
        let Some(blocks) = self.crosscoder_blocks.as_ref() else {
            return 0.0;
        };
        let n = self.target.nrows() as f64;
        blocks
            .block_dims
            .iter()
            .zip(rho.log_lambda_block.iter())
            .map(|(&p_l, &log_lambda)| -(n * p_l as f64 / 2.0) * log_lambda)
            .sum()
    }

    /// #2231 Inc-B (stage 2) — the per-output-block SCALED residual sum of
    /// squares `R̃_ℓ = ‖r̃_ℓ‖²` at the current fitted state, over each block's
    /// stacked-column span `[p_x + Σ_{m<ℓ} p_m, …)`.
    ///
    /// `r̃ = fitted − self.target` is the residual against the ALREADY block-scaled
    /// target (every eval lane calls `apply_block_scaling` before the inner solve),
    /// so `R̃_ℓ` is the scaled-block residual the `#F1` unit-dispersion data term
    /// `½‖r̃‖² = ½(R_x + Σ_ℓ R̃_ℓ)` already carries. In UNSCALED form
    /// `R̃_ℓ = λ_ℓ·R_ℓ` where `R_ℓ = ‖r̃_ℓ‖²/λ_ℓ` is the block's honest-units
    /// residual (the quantity `run_multiblock_reml_fit`'s `augmented_block_rss`
    /// reports); the two coincide at `λ_ℓ = 1`. Returns `None` when crosscoder
    /// pricing is off (plain SAE). The reconstruction is read from the CONVERGED
    /// fitted state, so callers must invoke this only after the lane's inner solve.
    fn block_scaled_rss(&self, rho: &SaeManifoldRho) -> Result<Option<Vec<f64>>, String> {
        let Some(blocks) = self.crosscoder_blocks.as_ref() else {
            return Ok(None);
        };
        let residual = self.term.reconstruction_residual(self.target.view(), rho)?;
        let mut out = Vec::with_capacity(blocks.block_dims.len());
        let mut off = blocks.p_x;
        for &p_l in &blocks.block_dims {
            let mut rss = 0.0_f64;
            for row in residual.rows() {
                for j in off..off + p_l {
                    let r = row[j];
                    rss += r * r;
                }
            }
            out.push(rss);
            off += p_l;
        }
        Ok(Some(out))
    }

    /// #2231 Inc-B (stage 2) — the EXPLICIT block-coordinate gradient channels
    /// `½·R̃_ℓ − n·p_ℓ/2`, one entry per output block, or `None` for a plain
    /// SAE. NOT the complete `∂C/∂log λ_ℓ` on its own — see below.
    ///
    /// Derivation (UNIT-dispersion `#F1`). Scaling block `ℓ`'s target columns by
    /// `√λ_ℓ` enters the criterion in three places: the raw half-SSE data term
    /// (through `R̃_ℓ`), the change-of-variables Jacobian `−(n·p_ℓ/2)·log λ_ℓ`
    /// ([`Self::block_jacobian`]), and the Laplace `½log|H|` term through the
    /// fitted state's response `θ̂(λ_ℓ)`. At the inner optimum the envelope
    /// theorem cancels the penalized-loss response, and the Gauss–Newton `H` at
    /// FIXED θ is target-independent, but the `½log|H(θ̂(λ_ℓ))|` chain-rule
    /// channel survives: it is the same `−½·Γᵀθ̂_ρ` adjoint every other ρ
    /// coordinate carries, supplied by the components assembler via
    /// [`SaeManifoldTerm::crosscoder_block_ift_rhs`] (RHS `−½·Jᵀ_M Z̃^{(ℓ)}`
    /// through the exact-stationarity solve). This function returns only the
    /// EXPLICIT channels — the data derivative `∂(½‖r̃‖²)/∂log λ_ℓ = ½·R̃_ℓ`
    /// (with `R̃_ℓ = ‖r̃_ℓ‖² = λ_ℓ·R_ℓ`) plus the Jacobian `−n·p_ℓ/2` — which
    /// the gradient lane ADDS to the assembler's tail (never overwrites; #2087).
    /// The explicit channels alone are stationary at `R̃_ℓ = n·p_ℓ`
    /// (`λ_ℓ = n·p_ℓ/R_ℓ`), the Fellner–Schall proposal root, and coercive at
    /// both ends; the adjoint shifts the true root by an `O(dim H/(n·p_ℓ))`
    /// relative correction.
    fn block_log_lambda_gradient(&self, rho: &SaeManifoldRho) -> Result<Option<Vec<f64>>, String> {
        let Some(scaled_rss) = self.block_scaled_rss(rho)? else {
            return Ok(None);
        };
        let blocks = self
            .crosscoder_blocks
            .as_ref()
            .expect("block_scaled_rss returned Some ⇒ crosscoder pricing is installed");
        let n = self.target.nrows() as f64;
        Ok(Some(
            blocks
                .block_dims
                .iter()
                .zip(scaled_rss.iter())
                .map(|(&p_l, &r_tilde)| 0.5 * r_tilde - 0.5 * n * p_l as f64)
                .collect(),
        ))
    }

    /// SPEC wall-survival: bank a resumable checkpoint at a MATERIAL improvement
    /// of the outer best cost. Best-effort — a checkpoint write must never abort
    /// a fit (the error is logged, not raised).
    pub(crate) fn bank_checkpoint(&self, rho_flat: &Array1<f64>) {
        let (evals, last_improvement_eval, best_cost) = self.termination.checkpoint_counters();
        let rho_owned = rho_flat.to_vec();
        // serde_json refuses non-finite floats, and the ledger's best cost is
        // finite by construction (`record` skips non-finite values); sanitize
        // the EV the same way so a degenerate probe can never wedge the write.
        let incumbent_ev = self
            .term
            .dictionary_reconstruction_ev(self.target.view(), &self.current_rho)
            .ok()
            .filter(|ev| ev.is_finite())
            .unwrap_or(-1.0);
        let ckpt = super::checkpoint::SaeFitCheckpoint::capture(
            &self.term,
            &self.checkpoint_fingerprint,
            &rho_owned,
            super::checkpoint::SaeCheckpointLedger {
                evals,
                last_improvement_eval,
                best_cost,
            },
            incumbent_ev,
        );
        if let Some(dir) = self.checkpoint_path.parent()
            && let Err(e) = std::fs::create_dir_all(dir)
        {
            log::warn!("SAE fit checkpoint: create dir {}: {e}", dir.display());
            return;
        }
        if let Err(e) = ckpt.save_atomic(&self.checkpoint_path) {
            log::warn!("SAE fit checkpoint: {e}");
        }
    }

    /// SPEC wall-survival: attempt to resume from a banked checkpoint for this
    /// exact data fingerprint. On a verified hit, installs the banked incumbent
    /// into the term (and the baseline term, so a multi-start `reset` re-opens
    /// from the banked state rather than the cold seed), seeds the termination
    /// ledger counters, and returns the banked outer ρ to open the search at.
    /// Structural incompatibility or mutable-state install failure is logged and
    /// the fit proceeds cold. A shape-compatible checkpoint whose rho violates
    /// the objective's mathematical domain is different: it is a typed refusal,
    /// because silently replacing that optimization state would conceal corrupt
    /// outer coordinates.
    pub fn try_resume_from_checkpoint(
        &mut self,
        expected_rho_len: usize,
    ) -> Result<Option<Vec<f64>>, String> {
        self.fit_verdict = None;
        if !self.checkpoint_path.exists() {
            return Ok(None);
        }
        let ckpt = match super::checkpoint::SaeFitCheckpoint::load(&self.checkpoint_path) {
            Ok(c) => c,
            Err(e) => {
                log::warn!("SAE fit checkpoint resume: {e}; fitting cold");
                return Ok(None);
            }
        };
        if let Err(e) = ckpt.verify_compatible(&self.checkpoint_fingerprint, expected_rho_len) {
            log::warn!("SAE fit checkpoint resume: {e}; fitting cold");
            return Ok(None);
        }
        if let Err(e) = self
            .baseline_rho
            .from_flat(ArrayView1::from(ckpt.rho_flat.as_slice()))
        {
            return Err(format!(
                "SAE fit checkpoint resume refused invalid rho payload: {e}"
            ));
        }
        let install_result = ckpt.install_into(&mut self.term);
        if install_result.is_ok()
            && let Err(e) = ckpt.install_into(&mut self.baseline_term)
        {
            log::warn!("SAE fit checkpoint resume (baseline): {e}");
        }
        if let Err(e) = install_result {
            log::warn!("SAE fit checkpoint resume: {e}; fitting cold");
            return Ok(None);
        }
        self.termination.seed_from_checkpoint(
            ckpt.ledger.evals,
            ckpt.ledger.last_improvement_eval,
            ckpt.ledger.best_cost,
        );
        log::warn!(
            "SAE fit checkpoint resume: installed banked incumbent from {} \
             (evals {}, best cost {:?}); the resumed search must still converge on its own",
            self.checkpoint_path.display(),
            ckpt.ledger.evals,
            ckpt.ledger.best_cost,
        );
        Ok(Some(ckpt.rho_flat))
    }

    /// Remove the banked checkpoint after a CONVERGED fit is minted: its
    /// purpose is wall survival of an in-flight optimization, not cross-fit
    /// caching (`persistent_warm_start` covers that). Best-effort.
    pub fn remove_checkpoint(&self) {
        if self.checkpoint_path.exists()
            && let Err(e) = std::fs::remove_file(&self.checkpoint_path)
        {
            log::warn!(
                "SAE fit checkpoint: remove {}: {e}",
                self.checkpoint_path.display()
            );
        }
    }

    /// #2138 — install a cooperative cancellation flag shared with the pyffi fit
    /// driver's calling thread. On a Python interrupt the caller sets it, and the
    /// next outer `eval`/`eval_cost` bails with a recoverable error so the
    /// detached worker thread terminates instead of finishing a hung fit.
    pub fn set_cancel_flag(&mut self, flag: std::sync::Arc<std::sync::atomic::AtomicBool>) {
        self.cancel_flag = Some(flag);
    }

    /// `Err` if a host cancellation was requested (see [`Self::set_cancel_flag`]);
    /// a cheap relaxed load, no-op when no flag is installed.
    fn check_cancelled(&self) -> Result<(), EstimationError> {
        if let Some(flag) = &self.cancel_flag {
            if flag.load(std::sync::atomic::Ordering::Relaxed) {
                return Err(EstimationError::RemlOptimizationFailed(
                    "SAE fit cancelled by host (Python interrupt)".to_string(),
                ));
            }
        }
        Ok(())
    }

    /// #2080 — the accumulated outer probe telemetry (criterion/infeasible
    /// evaluation counts). The wide-`p` acceptance test asserts these counts stay
    /// bounded (a PROBE-COUNT budget, per SPEC's ban on wall-clock budgets).
    pub fn probe_telemetry(&self) -> OuterProbeTelemetry {
        self.probe_telemetry
    }

    /// Record one amortized warm-start attempt. Once selected, this accelerator
    /// is part of the declared optimization path: an encoder/atlas failure is
    /// propagated instead of silently changing the basin-entry algorithm.
    fn record_warm_start(&mut self, outcome: Result<usize, String>) -> Result<(), String> {
        // The row count is the telemetry's business (folded in above); the
        // caller only needs the failure to propagate.
        self.warm_start_telemetry.record(&outcome);
        outcome?;
        Ok(())
    }

    /// Stamp the currently installed state with a successful outer search's
    /// analytic convergence evidence.
    ///
    /// Merely receiving `an OuterResult whose solver claimed convergence` is insufficient:
    /// the result must carry both the shared engine's explicit `converged_via`
    /// verdict and a valid analytic criterion certificate, and its rho must be
    /// bit-identical to the state currently installed on this objective. This
    /// closes the #2230 hole where any successful evaluation populated
    /// `last_loss` and `into_fitted` silently interpreted an absent search
    /// verdict as `FixedRho`.
    pub fn certify_outer_result(&mut self, result: &OuterResult) -> Result<(), String> {
        self.fit_verdict = None;
        self.terminal_penalized_quasi_laplace_criterion = None;
        if !result.converged() {
            return Err("outer result is not converged".to_string());
        }
        let via = result
            .converged_via()
            .ok_or_else(|| "converged outer result is missing converged_via".to_string())?;
        let certificate = result.criterion_certificate.as_ref().ok_or_else(|| {
            "converged outer result is missing its analytic criterion certificate".to_string()
        })?;
        if !certificate.certifies() {
            return Err(format!(
                "outer criterion certificate does not certify the installed state: {}",
                certificate.summary()
            ));
        }
        if self.last_loss.is_none() {
            return Err("outer result has no installed converged inner loss".to_string());
        }
        let installed_rho = self.current_rho.to_flat();
        let rho_matches = installed_rho.len() == result.rho.len()
            && installed_rho
                .iter()
                .zip(result.rho.iter())
                .all(|(installed, certified)| installed.to_bits() == certified.to_bits());
        if !rho_matches {
            return Err(format!(
                "outer result rho does not match the installed state (certified={:?}, installed={:?})",
                result.rho, installed_rho
            ));
        }
        if !result.final_value.is_finite() {
            return Err("converged outer result has a non-finite final criterion value".into());
        }
        self.terminal_penalized_quasi_laplace_criterion = Some(result.final_value);
        self.fit_verdict = Some(SaeOuterVerdict::Search(via));
        Ok(())
    }

    /// Freeze every basin-entry accelerator so the next analytic evaluation
    /// audits the exact installed `(term, rho)` rather than moving coordinates
    /// before measuring stationarity.
    pub(crate) fn for_installed_state_audit(mut self) -> Self {
        self.audit_installed_state = true;
        self.inner_max_iter = 0;
        self
    }

    /// Record criterion work performed by an outer search, never by the
    /// zero-step installed-state audit. Every objective order funnels through
    /// this gate so value-only agreement checks cannot silently increment the
    /// optimization ledger while derivative orders do not (#2653).
    fn record_search_criterion(&mut self, cost: f64, gradient_norm: Option<f64>) -> bool {
        !self.audit_installed_state && self.termination.record(cost, gradient_norm)
    }

    /// Stamp a caller-installed state only after the shared exact-point outer
    /// certificate has passed. No search ran, so its provenance is distinct from
    /// [`Self::certify_outer_result`].
    pub(crate) fn certify_installed_state_audit(
        &mut self,
        result: &OuterResult,
    ) -> Result<(), String> {
        self.fit_verdict = None;
        self.terminal_penalized_quasi_laplace_criterion = None;
        if !self.audit_installed_state {
            return Err("installed-state audit was not enabled on this objective".to_string());
        }
        if result.iterations != 0 || !result.converged() {
            return Err("installed-state audit result is not a zero-step convergence".to_string());
        }
        let via = result
            .converged_via()
            .ok_or_else(|| "installed-state audit is missing converged_via".to_string())?;
        let certificate = result.criterion_certificate.as_ref().ok_or_else(|| {
            "installed-state audit is missing its analytic criterion certificate".to_string()
        })?;
        if !certificate.certifies() {
            return Err(format!(
                "installed-state outer certificate does not certify: {}",
                certificate.summary()
            ));
        }
        if self.last_loss.is_none() {
            return Err("installed-state audit has no evaluated inner loss".to_string());
        }
        let installed_rho = self.current_rho.to_flat();
        if installed_rho.len() != result.rho.len()
            || installed_rho
                .iter()
                .zip(result.rho.iter())
                .any(|(installed, certified)| installed.to_bits() != certified.to_bits())
        {
            return Err("installed-state audit rho does not match the evaluated state".to_string());
        }
        if !result.final_value.is_finite() {
            return Err("installed-state audit produced a non-finite criterion".to_string());
        }
        self.terminal_penalized_quasi_laplace_criterion = Some(result.final_value);
        self.fit_verdict = Some(SaeOuterVerdict::Audited(via));
        Ok(())
    }

    /// Consume a converged objective, returning the exact certified `(term, ρ)`
    /// pair and its inner loss. A merely evaluated objective is an error: only a
    /// completed fixed-ρ solve or an explicitly certified outer search may mint
    /// a fit.
    pub fn into_fitted(self) -> Result<SaeIntoFittedResult, String> {
        let verdict = self.fit_verdict.ok_or_else(|| {
            "SaeManifoldOuterObjective::into_fitted: installed state is not explicitly certified; \
             run fit_at_fixed_rho or certify a converged OuterResult before minting a fit"
                .to_string()
        })?;
        let termination_report = self.termination.report(verdict);
        let Self {
            term,
            target,
            registry,
            current_rho,
            last_loss,
            terminal_penalized_quasi_laplace_criterion,
            ..
        } = self;
        let mut fitted_rho = current_rho;
        let mut fitted = term;
        if last_loss.is_none() {
            return Err(
                "SaeManifoldOuterObjective::into_fitted: certified state has no converged inner loss"
                    .to_string(),
            );
        }
        let penalized_quasi_laplace_criterion = terminal_penalized_quasi_laplace_criterion.ok_or_else(|| {
            "SaeManifoldOuterObjective::into_fitted: terminal state has no penalized quasi-Laplace criterion value"
                .to_string()
        })?;

        // Do not arbitrate the certified terminal state against historical
        // reconstruction-EV incumbents or construction seeds here. Those states
        // were optimized at different ρ values (or never optimized) and pairing
        // one with `current_rho` after the outer certificate creates a fit object
        // that is not a stationary point of its reported objective. Basin
        // selection belongs inside the objective's lower-envelope evaluation,
        // before the analytic outer certificate is issued.
        // #1019 — the post-fit assembly seam: canonicalize every eligible
        // atom's chart to its canonical Diff(M) representative (arc length
        // for d = 1, minimum-isometry-defect flow for d = 2 torus atoms)
        // BEFORE the fitted term is handed to the payload / residual-gauge
        // certificate. Internally objective-gated and image-frozen (the
        // fitted state is restored verbatim on any failure or tolerance
        // breach), so the fit this returns is never degraded — an error here
        // is a refused canonicalization, not a broken fit.
        let pre_canonical_flags = fitted
            .atoms
            .iter()
            .map(|atom| atom.chart_canonicalized)
            .collect::<Vec<_>>();
        if let Err(err) =
            fitted.canonicalize_charts_post_fit(target.view(), &fitted_rho, registry.as_ref())
        {
            log::debug!("into_fitted: chart canonicalization refused: {err}");
        }
        let charts_canonicalized = fitted
            .atoms
            .iter()
            .zip(pre_canonical_flags.iter())
            .any(|(atom, before)| atom.chart_canonicalized != *before);
        if fitted
            .assignment
            .persist_resolved_ordered_beta_bernoulli_alpha(&fitted_rho)
        {
            fitted_rho.log_lambda_sparse = 0.0;
        }
        let fitted_loss = fitted.loss(target.view(), &fitted_rho)?;
        let termination = termination_report;
        log::warn!(
            "[#2235] outer search concluded: {} evals ({} since last improvement, wall {:.1?})",
            termination.evals,
            termination.evals_since_improvement,
            termination.wall
        );
        Ok(SaeIntoFittedResult {
            term: fitted,
            rho: fitted_rho,
            loss: fitted_loss,
            penalized_quasi_laplace_criterion,
            charts_canonicalized,
            termination,
        })
    }

    /// Posterior shape uncertainty of the fitted atoms — per-atom decoder
    /// covariance and ambient bands (see
    /// [`SaeManifoldTerm::assemble_shape_uncertainty`]).
    ///
    /// Recomputes the converged joint-Hessian Laplace factor at the settled ρ
    /// — the same undamped Direct factor the penalized quasi-Laplace criterion forms at the inner
    /// optimum — and reads the per-atom covariance and bands off its cached
    /// Schur factor, scaling by the Gaussian reconstruction dispersion `φ̂`.
    /// The term is already at the optimum after the outer fit, so the inner
    /// re-solve converges immediately. Call before [`Self::into_fitted`].
    /// The most recent curvature-homotopy entry walk outcome on the live term
    /// (#1007), or `None` when no walk has run. Surfaced on the objective so the
    /// arrival / bifurcation / collapse outcome is observable without consuming
    /// the objective via [`Self::into_fitted`].
    pub fn curvature_walk_report(&self) -> Option<&CurvatureWalkReport> {
        self.term.curvature_walk_report()
    }

    pub fn decoder_shape_uncertainty(&mut self) -> Result<SaeShapeUncertainty, String> {
        // #2080 (a) — this diagnostic runs its own inner solves against the
        // accepted basin; drop any pending probe handoff.
        self.probe_converged_handoff = None;
        // #2230/#2087 — the ρ search is over; drop the saved basins too.
        self.basin_bundle.clear();
        let rho = self.current_rho.clone();
        let plan = self.term.streaming_plan()?.admitted_or_error(
            self.term.n_obs(),
            self.term.output_dim(),
            self.term.k_atoms(),
        )?;
        if !plan.direct_logdet_admitted() {
            let loss = self.term.loss(self.target.view(), &rho)?;
            let n_scalar = (self.term.n_obs().saturating_mul(self.term.output_dim())).max(1) as f64;
            let dispersion = (2.0 * loss.data_fit / n_scalar).max(f64::MIN_POSITIVE);
            return Ok(self.term.unavailable_shape_uncertainty(dispersion));
        }
        // Re-form the strict undamped joint factor at the settled ρ. A failure is
        // an inference failure; it is never replaced by a different covariance.
        let saved_term = self.term.clone();
        let evaluated = self.term.penalized_quasi_laplace_criterion_with_cache(
            self.target.view(),
            &rho,
            self.registry.as_ref(),
            self.inner_max_iter,
            self.learning_rate,
            self.ridge_ext_coord,
            self.ridge_beta,
        );
        let (_cost, loss, cache) = match evaluated {
            Ok(evaluated) => evaluated,
            Err(err) => {
                self.term = saved_term;
                return Err(err.to_string());
            }
        };
        let residual = self
            .term
            .reconstruction_residual(self.target.view(), &rho)?;
        let dispersion =
            self.term
                .reconstruction_dispersion(&loss, &cache, &rho, Some(residual.view()))?;
        self.term.assemble_shape_uncertainty(&cache, dispersion)
    }

    /// Record the discrete fitted-data collapse verdict without changing the
    /// penalized quasi-Laplace objective. The verdict feeds structure search and the final fit
    /// ledger; it is not a smooth term and therefore cannot be added to a cost
    /// that is paired with the analytic derivative of the penalized quasi-Laplace scalar (#2253).
    fn record_fit_data_collapse_verdict(&mut self, rho: &SaeManifoldRho) -> Result<(), String> {
        self.term.record_fit_data_collapse_if_needed(
            self.target.view(),
            rho,
            self.inner_max_iter,
        )?;
        Ok(())
    }

    /// Whether a value probe has no defined penalized quasi-Laplace score. Such a state is
    /// not admitted to the handoff or basin bundle. Finite collapsed fits remain
    /// ordinary penalized quasi-Laplace values; their separate structural verdict is recorded above.
    fn probe_value_is_infeasible(value: f64) -> bool {
        !value.is_finite()
    }

    /// Whether a refused probe is ρ-local — the criterion is undefined HERE and
    /// a neighbouring ρ admits a fit — rather than a genuine defect.
    ///
    /// One line, because the classification lives in [`ProbeRefusalKind`] and
    /// the telemetry counters read the same table (#2593).
    pub(crate) fn is_recoverable_value_probe_refusal(err: &str) -> bool {
        ProbeRefusalKind::classify(err).is_some()
    }

    /// #2080 (a) — take the single-shot probe handoff, returning its converged
    /// term ONLY when the stored ρ matches `rho_flat` BITWISE. The take is
    /// unconditional (match or not), so a handoff can never survive past any
    /// criterion-driving call and go stale against a moved accepted basin: the
    /// only state it can ever warm-start is the very next evaluation, and only
    /// at the exact ρ whose converged optimum it holds.
    fn take_probe_converged_handoff(
        &mut self,
        rho_flat: ArrayView1<'_, f64>,
    ) -> Option<SaeManifoldTerm> {
        let handoff = self.probe_converged_handoff.take()?;
        let matches = handoff.rho_flat.len() == rho_flat.len()
            && handoff
                .rho_flat
                .iter()
                .zip(rho_flat.iter())
                .all(|(a, b)| a.to_bits() == b.to_bits());
        if matches { Some(handoff.term) } else { None }
    }

    /// Evaluate the authoritative penalized quasi-Laplace criterion at
    /// `rho_flat`, updating the cached ρ / loss and optionally priming the inner
    /// solve from a seeded β. Every finite production value is driven to the
    /// same idempotent fixed point differentiated by the analytic gradient; the
    /// raw reduced-budget term policy is intentionally not exposed through the
    /// outer objective. Returns `(cost, β̂)`.
    pub(crate) fn evaluate_authoritative_criterion(
        &mut self,
        rho_flat: ArrayView1<'_, f64>,
    ) -> Result<(f64, Array1<f64>), String> {
        self.evaluate_authoritative_inner(rho_flat, false)
    }

    /// Shared authoritative inner drive. Everything around the full-refine
    /// criterion — probe handoff installation, seeded-β and amortized latent
    /// warm starts, and the collapse ledger — is shared across outer lanes.
    fn evaluate_authoritative_inner(
        &mut self,
        rho_flat: ArrayView1<'_, f64>,
        basin_installed: bool,
    ) -> Result<(f64, Array1<f64>), String> {
        // Any new criterion drive may change the installed inner state. A fit
        // certificate is single-use evidence for the exact state/rho pair that
        // produced it, never a sticky success flag.
        self.fit_verdict = None;
        let rho = self.baseline_rho.from_flat(rho_flat)?;
        // #2231 Inc-B — materialize the block-relevance target scaling for THIS ρ
        // before any inner solve reads `self.target`. Every value/refine/member/
        // discovery lane funnels through this one drive, so a single idempotent
        // rewrite keeps them all coherent (no-op for a plain SAE).
        self.apply_block_scaling(&rho)?;
        // #2080 (a) — install the last value probe's converged inner state when
        // this evaluation re-visits the exact same ρ (the line-search accept
        // pattern). The state IS the inner KKT optimum this solve would converge
        // to from the accepted basin (see `ProbeConvergedHandoff`), so the
        // criterion value is unchanged — the solve below merely reaches its
        // stationarity gate immediately instead of re-tracing the probe's
        // deterministic Newton trajectory. The pending seeded-β hint (if any)
        // was already applied by the probe before it converged, so it must not
        // be re-applied on top of the converged state; likewise the amortized
        // latent warm-start is skipped — it is a basin-ENTRY heuristic, and the
        // installed state is already AT the converged optimum for this ρ.
        // #2230/#2087 — `basin_installed` (the basin-bundle member lane): the
        // caller has already installed a saved converged basin state into
        // `self.term` and wants it re-converged AT that state, so this evaluation
        // must NOT consult the probe handoff (it would clobber the installed
        // basin) and must skip the seeded-β re-apply and the amortized latent
        // basin-ENTRY warm start — exactly the handoff path's semantics, since an
        // installed member is already at (or near) its basin's converged optimum.
        let probe_handoff_installed = if basin_installed {
            true
        } else if let Some(converged) = self.take_probe_converged_handoff(rho_flat) {
            self.term = converged;
            self.seeded_beta = None;
            true
        } else {
            false
        };
        if let Some(beta) = self.seeded_beta.take() {
            // Warm-start the inner decoder coefficients before the solve.
            if beta.len() != self.term.beta_dim() {
                return Err(format!(
                    "seeded decoder has length {}; expected {}",
                    beta.len(),
                    self.term.beta_dim()
                ));
            }
            self.term.set_flat_beta(beta.view())?;
        }
        // #1154 item 2 (Design A) — warm-start the inner latent coords from the
        // amortized encoder built on the CURRENT dictionary. At outer step m this
        // seeds the inner solve from the per-chart IFT predictor of the dictionary
        // settled at step m−1, refined to the SAME stationary point (so the penalized quasi-Laplace
        // λ-gradient is untouched). A first-build / degenerate atlas may
        // certify zero rows, but an actual encoder/atlas error aborts the
        // evaluation rather than silently selecting a different basin-entry path.
        if !probe_handoff_installed && !self.audit_installed_state {
            let warm_start_outcome = self
                .term
                .warm_start_latents_from_amortized_encoder(self.target.view(), &rho);
            self.record_warm_start(warm_start_outcome)?;
        }
        let criterion = self
            .term
            .penalized_quasi_laplace_criterion_with_refine_policy_and_lane(
                self.target.view(),
                &rho,
                self.registry.as_ref(),
                self.inner_max_iter,
                self.learning_rate,
                self.ridge_ext_coord,
                self.ridge_beta,
                true,
                self.surrogate_lane.as_mut(),
            );
        let (penalized_quasi_laplace_cost, loss) = match criterion {
            Ok(evaluated) => evaluated,
            Err(SaeCriterionError::VanishedAtoms(atoms)) => {
                log::debug!(
                    "SAE criterion reached fixed-K structural boundary at rho={:?}: {atoms}",
                    rho.to_flat()
                );
                let loss = self.term.loss(self.target.view(), &rho)?;
                let beta_hat = self.term.flatten_beta();
                self.current_rho = rho;
                self.last_loss = Some(loss);
                self.probe_telemetry.infeasible_criterion_evals += 1;
                return Ok((f64::INFINITY, beta_hat));
            }
            // #2336 — an indefinite exact `A` leaves the Laplace normaliser
            // `½log|A|` UNDEFINED at this ρ, so this evaluation is INFEASIBLE, not
            // defective. That is the same class `is_recoverable_value_probe_refusal`
            // already maps to `+inf`, for the reason its #1782 note gives: the
            // indefinite basin is adjacent to the PD optimum, so the outer solver
            // must read `+∞` and steer back into the PD region rather than abort the
            // whole fit. #2330 Phase-2a made `½log|A|` the ranked value, which is
            // what made this reachable — the majorizer `B` was PD by construction and
            // could never trip it. Escaping the saddle is the ACCEPTED lane's job
            // (the #2336 terminal escape, upstream in the criterion); by the time a
            // refusal surfaces here the escape is already exhausted, and a probe must
            // stay probe-infeasible rather than grind.
            Err(err @ SaeCriterionError::IndefiniteObservedInformation { .. }) => {
                self.probe_telemetry.record_refusal_kind(&err.to_string());
                log::debug!("SAE criterion mapped indefinite-A refusal to +inf: {err}");
                let loss = self.term.loss(self.target.view(), &rho)?;
                let beta_hat = self.term.flatten_beta();
                self.current_rho = rho;
                self.last_loss = Some(loss);
                self.probe_telemetry.infeasible_criterion_evals += 1;
                return Ok((f64::INFINITY, beta_hat));
            }
            Err(SaeCriterionError::Numerical(message)) => return Err(message),
        };
        let beta_hat = self.term.flatten_beta();
        // ONE criterion everywhere. Every outer lane — BFGS/ARC descent, the
        // line-search value probes, cross-seed ranking, EFS backtracking, and
        // final selection — prices the SAME penalized quasi-Laplace criterion `f(ρ)` whose
        // exact implicit gradient `∇f` the gradient lane returns. The former
        // #1154 amortized-encoder consistency fold `c(ρ)` ranked seeds/EFS
        // states by `f+c` while optimization descended `f` alone (c had no
        // gradient), so the selected fit was not stationary for the criterion
        // that selected it — the objective↔gradient desync class (#931/#1206)
        // moved from the line search into selection. The fold is removed from
        // every fitting/ranking lane; encoder consistency remains available as
        // a pure diagnostic (`penalized_quasi_laplace_criterion_cotrained`). The fitted-data
        // collapse detector is a structural ledger verdict, not an objective
        // fold: changing a finite penalized quasi-Laplace value by a constant sentinel would pair
        // that post-hoc value with the analytic penalized quasi-Laplace derivative (#2253).
        self.record_fit_data_collapse_verdict(&rho)?;
        let cost = if penalized_quasi_laplace_cost.is_finite() {
            penalized_quasi_laplace_cost
        } else {
            // The criterion function returned Ok with a NON-FINITE value —
            // this is the assembled-value class (a non-finite Laplace
            // normalizer / rank charge / dispersion term at the converged
            // cache), distinct from the typed-refusal class the mapping
            // sites name. A silent ∞ here made every downstream
            // 'infeasible at the requested rho' failure untraceable.
            log::debug!(
                "SAE criterion assembled a NON-FINITE value {penalized_quasi_laplace_cost:.6e} \
                 (loss total {:.6e}) at the converged inner state — mapping to +inf",
                loss.total()
            );
            self.probe_telemetry.infeasible_criterion_evals += 1;
            f64::INFINITY
        };
        self.current_rho = rho;
        self.last_loss = Some(loss);
        Ok((cost, beta_hat))
    }

    /// Fit the SAE inner problem once at a caller-selected rho, committing the
    /// resulting basin without running the outer-rho search or its derivative
    /// lanes.
    pub fn fit_at_fixed_rho(&mut self, rho_flat: ArrayView1<'_, f64>) -> Result<(), String> {
        self.fit_verdict = None;
        self.terminal_penalized_quasi_laplace_criterion = None;
        let rho_state = self.baseline_rho.from_flat(rho_flat.clone())?;
        let (criterion, _) = self.evaluate_authoritative_criterion(rho_flat)?;
        let jacobian = self.block_jacobian(&rho_state);
        let cost = criterion + jacobian;
        if !cost.is_finite() {
            // Decompose the non-finite total so the failure names its source:
            // an ∞ criterion is the infeasibility sentinel (which component
            // refused is logged at the mapping sites), while a non-finite
            // block Jacobian is its own defect class.
            return Err(format!(
                "SaeManifoldOuterObjective::fit_at_fixed_rho: penalized quasi-Laplace criterion \
                 is infeasible at the requested rho (criterion={criterion:.6e}, \
                 block_jacobian={jacobian:.6e})"
            ));
        }
        self.terminal_penalized_quasi_laplace_criterion = Some(cost);
        self.fit_verdict = Some(SaeOuterVerdict::FixedRho);
        Ok(())
    }

    /// Evaluate a value-only rho probe without committing the inner basin it
    /// reaches. The generic line search may reject this point, so its solved
    /// coordinates/decoder must not become the warm-start state for later
    /// probes or for the accepted iterate. This path exposes no work-policy
    /// switch: a finite value is always the authoritative fixed-point
    /// criterion, never a reduced-budget iterate.
    fn evaluate_authoritative_value_probe(
        &mut self,
        rho_flat: ArrayView1<'_, f64>,
    ) -> Result<(f64, Array1<f64>), String> {
        let saved_term = self.term.clone();
        let saved_rho = self.current_rho.clone();
        let saved_loss = self.last_loss.clone();
        let saved_seeded_beta = self.seeded_beta.clone();
        let result = self.evaluate_authoritative_inner(rho_flat, false);
        // #2080 (a) — instead of discarding the probe's converged inner state,
        // hand it off (a move: swapped against the restored `saved_term`, no
        // extra clone) to the next evaluation at this exact ρ — the line-search
        // accept pattern re-evaluates the accepted point at the ρ of its last
        // successful value probe. Only a genuinely converged finite value is
        // worth handing off; a refused or non-finite probe never defines usable
        // penalized quasi-Laplace score.
        match &result {
            Ok((cost, _beta)) if !Self::probe_value_is_infeasible(*cost) => {
                let converged = std::mem::replace(&mut self.term, saved_term);
                self.probe_converged_handoff = Some(ProbeConvergedHandoff {
                    rho_flat: rho_flat.to_owned(),
                    term: converged,
                });
            }
            _ => {
                self.term = saved_term;
            }
        }
        self.current_rho = saved_rho;
        self.last_loss = saved_loss;
        self.seeded_beta = saved_seeded_beta;
        result
    }

    /// #2230/#2087 — evaluate the basin lower envelope `V*(ρ) = min_b V_b(ρ)` for
    /// the value lanes (`eval_cost`, `eval_with_order(Value)`), replacing the
    /// single hysteretic warm-start trajectory. The steps:
    ///
    /// 1. **Bypass.** In the streaming / matrix-free regime (no dense per-round
    ///    assembly, so the value path is the cost-only streaming cascade) and
    ///    under the `inner_max_iter == 0` FREEZE contract (verbatim reuse, no
    ///    exploration), the bundle is bypassed and the historical single probe is
    ///    returned byte-for-byte. The streaming state snapshot is a subsampled /
    ///    matrix-free term whose per-basin re-convergence has no dense factor, and
    ///    the freeze lane must not re-converge anything.
    /// 2. **Seed.** On the first envelope evaluation the bundle admits the current
    ///    accepted basin (`self.term`).
    /// 3. **Discovery.** Run the ONE historical warm-start probe from the accepted
    ///    basin (`evaluate_authoritative_value_probe`). It consumes the seeded-β /
    ///    amortized-encoder warm start, parks its converged term in the probe
    ///    handoff, and — crucially — is the mechanism by which a basin JUMP is
    ///    discovered (its warm start can cross a boundary at a far ρ).
    /// 4. **Members.** Re-converge every saved basin from its OWN state through
    ///    the authoritative value-probe drive (`basin_installed = true`: no
    ///    warm-start, no seed) — members near their optimum re-converge in a
    ///    round or two.
    /// 5. **Admit + envelope.** Admit the discovery basin (new basin ⇒ grow;
    ///    duplicate ⇒ keep the better value). The envelope value is the bundle
    ///    argmin over {members ∪ discovery}; the argmin basin's converged state is
    ///    installed as the probe handoff so the subsequent gradient eval prices
    ///    THAT basin (envelope theorem). Admission can only LOWER the envelope.
    ///
    /// Every discovery/member evaluation uses the same authoritative full-refine
    /// policy as the analytic gradient lane. Warm member states remain cheap in
    /// realised iterations, but no finite reduced-budget iterate is an admissible
    /// envelope value. Retention is bounded only by memory admission; a
    /// work-count cap would make the envelope inexact.
    fn authoritative_envelope_value_probe(
        &mut self,
        rho_flat: ArrayView1<'_, f64>,
    ) -> Result<(f64, Array1<f64>), String> {
        // (1) Bypass: streaming/matrix-free (no dense per-basin factor to
        // re-converge) or the freeze contract (verbatim reuse). Byte-for-byte
        // historical single trajectory.
        if self.inner_max_iter == 0 || !self.term.streaming_plan()?.direct_logdet_admitted() {
            return self.evaluate_authoritative_value_probe(rho_flat);
        }

        // (2) Seed the bundle with the accepted entry basin on first use. The
        // placeholder +∞ value is overwritten the first time this member is
        // re-converged below.
        if self.basin_bundle.is_empty() {
            self.basin_bundle
                .admit_distinct(self.term.clone(), f64::INFINITY)
                .map_err(|error| format!("SAE basin-envelope seed admission refused: {error}"))?;
        }

        // (3) Discovery trajectory — the historical single warm-start probe. Sets
        // the probe handoff (when finite) to its converged term at this exact ρ.
        let discovery = self.evaluate_authoritative_value_probe(rho_flat);
        let discovery_cost = match &discovery {
            Ok((cost, _)) if !Self::probe_value_is_infeasible(*cost) => Some(*cost),
            _ => None,
        };
        // Reclaim the discovery basin's converged term from the handoff it just
        // parked (bitwise ρ match, so this retrieves exactly that term). The
        // envelope argmin's handoff is re-installed at the end.
        let discovery_term = self.take_probe_converged_handoff(rho_flat);

        // (4) Re-converge every saved member from its own state (cheap, pure). The
        // bundle is moved out of `self` so the closure can borrow `&mut self` for
        // the per-member inner drive; restored immediately after.
        let mut bundle = std::mem::replace(&mut self.basin_bundle, BasinBundle::new(0));
        let member_eval = bundle.evaluate(|state: &SaeManifoldTerm| {
            let (res, converged) = self.converge_member_criterion(rho_flat, state);
            res.map(|value| (converged, value))
        });

        // (5) Admit the discovery basin and read the envelope. `same_basin_at_rho`
        // needs the centered target variance normalizer; compute it once.
        let rho_state = self.baseline_rho.from_flat(rho_flat)?;
        let ss_tot =
            super::fit_drivers::TargetCenteredColStats::compute(self.target.view()).ss_tot();
        let len_before = bundle.len();
        if let (Some(term), Some(cost)) = (discovery_term, discovery_cost) {
            let admission = bundle.admit(term, cost, |a, b| {
                Self::same_basin_at_rho(a, b, &rho_state, ss_tot)
            });
            if let Err(error) = admission {
                self.basin_bundle = bundle;
                return Err(format!(
                    "SAE exact basin-envelope discovery admission refused: {error}"
                ));
            }
        }
        let grew = bundle.len() > len_before;
        let bundle_len = bundle.len();
        // Envelope argmin over {members ∪ discovery}. Prefer the argmin member if
        // any member is finite; otherwise fall back to the discovery result.
        let envelope = bundle
            .argmin()
            .filter(|m| m.last_value.is_finite())
            .map(|m| (m.last_value, m.state.flatten_beta(), m.state.clone()));
        self.basin_bundle = bundle;

        // Telemetry.
        self.probe_telemetry.basin_envelope_evals += 1;
        if grew {
            self.probe_telemetry.basin_admissions += 1;
        }
        self.probe_telemetry.basin_max_members =
            self.probe_telemetry.basin_max_members.max(bundle_len);
        self.probe_telemetry.basin_member_capacity = self.basin_bundle.member_capacity();

        match envelope {
            Some((env_value, env_beta, env_term)) => {
                // A rescue: a SAVED basin beat the fresh discovery trajectory by
                // more than the inner objective stall tolerance — the single
                // trajectory would have jumped UP across a boundary here.
                if let Some(dcost) = discovery_cost {
                    let stall = SAE_MANIFOLD_INNER_OBJECTIVE_STALL_REL_TOL * dcost.abs().max(1.0);
                    if dcost - env_value > stall {
                        self.probe_telemetry.basin_envelope_rescues += 1;
                    }
                }
                // Install the argmin basin's converged state as the handoff so the
                // gradient lane prices THIS basin (envelope theorem). Only a
                // finite penalized quasi-Laplace envelope is worth handing off.
                if !Self::probe_value_is_infeasible(env_value) {
                    self.probe_converged_handoff = Some(ProbeConvergedHandoff {
                        rho_flat: rho_flat.to_owned(),
                        term: env_term,
                    });
                }
                Ok((env_value, env_beta))
            }
            // Every member AND the discovery trajectory were infeasible at this ρ.
            // Return the discovery verdict verbatim; the caller maps a recoverable
            // refusal to the optimizer's conventional infeasible value.
            None => {
                drop(member_eval);
                discovery
            }
        }
    }

    /// Install the authoritative lower-envelope basin at `rho_flat` into the
    /// accepted objective state. An exact-rho value-probe handoff is already the
    /// selected envelope argmin and is consumed directly; otherwise the shared
    /// envelope selector is run and its finite argmin handoff is required.
    ///
    /// Returns `false` only when every admissible basin has an undefined
    /// quasi-Laplace value at this rho. A finite selector result without its
    /// exact-rho converged-state handoff is a protocol violation, never a reason
    /// to continue from a different warm-start trajectory.
    fn install_authoritative_envelope_basin(
        &mut self,
        rho_flat: ArrayView1<'_, f64>,
    ) -> Result<bool, String> {
        if let Some(converged) = self.take_probe_converged_handoff(rho_flat) {
            self.term = converged;
            self.seeded_beta = None;
            return Ok(true);
        }

        let (cost, _beta) = self.authoritative_envelope_value_probe(rho_flat)?;
        if Self::probe_value_is_infeasible(cost) {
            return Ok(false);
        }
        let converged = self
            .take_probe_converged_handoff(rho_flat)
            .ok_or_else(|| {
                "SAE basin-envelope protocol violated: a finite probe at the requested rho did not install its exact-rho converged-state handoff"
                    .to_string()
            })?;
        self.term = converged;
        self.seeded_beta = None;
        Ok(true)
    }

    /// Re-converge one saved basin `member` at `rho_flat` through the
    /// authoritative full-refine drive, returning `(criterion,
    /// converged_term)`. PURE w.r.t. `self`: `term`, `current_rho`,
    /// `last_loss`, and `seeded_beta` are all saved and restored.
    /// `basin_installed = true` so the installed converged state is NOT
    /// re-warm-started (no amortized encoder entry heuristic) and does NOT
    /// consume the pending β seed (the seed belongs to the discovery trajectory).
    fn converge_member_criterion(
        &mut self,
        rho_flat: ArrayView1<'_, f64>,
        member: &SaeManifoldTerm,
    ) -> (Result<f64, String>, SaeManifoldTerm) {
        let saved_term = std::mem::replace(&mut self.term, member.clone());
        let saved_rho = self.current_rho.clone();
        let saved_loss = self.last_loss.clone();
        // Members must not touch the pending seed — take it out for the duration.
        let saved_seeded_beta = self.seeded_beta.take();
        let res = self
            .evaluate_authoritative_inner(rho_flat, true)
            .map(|(cost, _beta)| cost);
        let converged = std::mem::replace(&mut self.term, saved_term);
        self.current_rho = saved_rho;
        self.last_loss = saved_loss;
        self.seeded_beta = saved_seeded_beta;
        (res, converged)
    }

    /// Basin-identity test for two converged SAE terms evaluated at the SAME ρ:
    /// the two dictionaries lie in the same inner basin iff their fitted
    /// reconstructions `Ŷ = Φ·B` coincide to within the fit's own explained-
    /// variance equality band. The reconstruction and the target-variance
    /// normalizer are both GAUGE-INVARIANT (chart rotation/reflection and
    /// cross-atom relabeling leave `Ŷ` unchanged), so this discriminates genuine
    /// distinct local minima — which fit the data differently — without splitting
    /// one basin reached through two different gauges. The threshold is
    /// `SAE_FINAL_EV_DEGRADATION_TOL`, the SAME normalized band the inner keep-best
    /// (`prefer_candidate_basin`) treats as "equal EV": two fits whose
    /// reconstructions differ by less than that in explained-variance units are
    /// the fit's own definition of the same basin, so no new constant is minted.
    /// A state that cannot be decoded at this ρ is treated as a distinct basin
    /// (an over-admit consumes one memory-admitted saved state and one extra cheap
    /// solve; a false MERGE would silently lose a basin).
    fn same_basin_at_rho(
        a: &SaeManifoldTerm,
        b: &SaeManifoldTerm,
        rho: &SaeManifoldRho,
        ss_tot: f64,
    ) -> bool {
        if !(ss_tot > 0.0) {
            return false;
        }
        let (Ok(fa), Ok(fb)) = (a.try_fitted_for_rho(rho), b.try_fitted_for_rho(rho)) else {
            return false;
        };
        if fa.dim() != fb.dim() {
            return false;
        }
        let mut diff_sq = 0.0_f64;
        for (x, y) in fa.iter().zip(fb.iter()) {
            let d = x - y;
            diff_sq += d * d;
        }
        (diff_sq / ss_tot) < SAE_FINAL_EV_DEGRADATION_TOL
    }

    /// Fellner-Schall / Mackay multiplicative fixed-point step on ρ at
    /// `rho_flat`. Runs the inner `(t, β)` solve to convergence at fixed ρ
    /// (sharing the single Direct factor with the penalized quasi-Laplace criterion), then
    /// returns `(cost, additive-log-steps, β̂)`.
    ///
    /// All ρ coords are log-quantities, so the engine's additive step
    /// `rho_new = rho + step` IS the multiplicative FS update. Per coord:
    /// - ARD axis (k,j): `α_new = n / (‖t_kj‖² + tr_kj(H⁻¹))` (unit-dispersion
    ///   MacKay fixed point, #F1 — no `φ̂`),
    ///   `step = ln α_new − log_ard[k][j]`. The `tr_kj(H⁻¹)` posterior
    ///   variance (from the selected-inverse latent diagonal) is exactly the
    ///   term the deleted `α=n/‖t‖²` rule dropped, so α cannot collapse on a
    ///   degenerate axis: as `‖t‖²→0`, `tr_kj(H⁻¹)→1/α` bounds the
    ///   denominator and the fixed point has a finite root.
    /// - λ_smooth[k] (per-atom, #1556): `λ_k_new = [p·rank S_k − tr_k(S_β⁻¹ M_k)]
    ///   / B_kᵀ(S_k⊗I_p)B_k` (Wood-Fasiolo EFS, already per-coordinate, #F1 — no
    ///   `φ̂`),
    ///   `step = ln λ_k_new − log_lambda_smooth[k]`, written into each atom's own
    ///   step slot `1+k`.
    /// - λ_sparse: 0.0 — the assignment-sparsity priors (softmax entropy,
    ///   gated L1, ordered Beta--Bernoulli) are non-quadratic, so no Gaussian-logdet FS fixed
    ///   point exists; it stays cost-driven (the cascade still moves it via
    ///   the cost path when EFS is not the active lane for that coord).
    /// #2330 — the `_` below discards the ONLY copy of the refusal diagnosis.
    ///
    /// When an evaluation is refused, `infeasible_evaluation` records why in every
    /// coordinate certificate (`fixed-point evidence unavailable: {reason}`) and
    /// [`Self::efs_step_with_certificate`] returns those certificates next to the
    /// eval. They are dropped here, so callers see only `cost = INFINITY` with
    /// `psi_gradient: None` and cannot tell a refusal from a structurally absent
    /// coordinate. That ambiguity has already produced one misattributed test
    /// failure (`tests_streaming_outer_gradient_2026.rs`).
    ///
    /// This is not a partially-used channel: `efs_step_with_certificate` has no
    /// other caller, and `eval_fixed_point_certificate` builds its own certificates
    /// from `eval` instead. So every certificate this path constructs — one
    /// formatted string per coordinate, per EFS step — is built and thrown away.
    ///
    /// The repair is a caller that keeps them, NOT a new field on `EfsEval`, which
    /// has 26 construction sites and would duplicate information this function
    /// already computes.
    pub(crate) fn efs_step(&mut self, rho_flat: ArrayView1<'_, f64>) -> Result<EfsEval, String> {
        self.efs_step_with_certificate(rho_flat)
            .map(|(evaluation, _)| evaluation)
    }

    /// Compute the iteration step and the separate final-proof residuals in one
    /// factorization. The step surface may hold a coordinate at zero; the proof
    /// surface marks that coordinate uncovered unless zero is the residual of a
    /// numerically defined, root-equivalent analytic equation.
    fn efs_step_with_certificate(
        &mut self,
        rho_flat: ArrayView1<'_, f64>,
    ) -> Result<(EfsEval, Vec<FixedPointCoordinateCertificate>), String> {
        self.fit_verdict = None;
        self.probe_telemetry.criterion_calls += 1;
        let rho = self.baseline_rho.from_flat(rho_flat)?;
        let n_params = rho.to_flat().len();
        // #2231 Inc-B — scale the block columns for this ρ before the EFS inner
        // solve reads `self.target` (idempotent; no-op for a plain SAE).
        self.apply_block_scaling(&rho)?;
        let direct_logdet_admitted = self.term.streaming_plan()?.direct_logdet_admitted();
        let infeasible_evaluation = |reason: &str| {
            (
                EfsEval {
                    cost: f64::INFINITY,
                    steps: vec![0.0_f64; n_params],
                    beta: None,
                    psi_gradient: None,
                    psi_indices: None,
                    inner_hessian_scale: None,
                    logdet_enclosure_gap: None,
                    consecutive_restored_incumbents: None,
                },
                (0..n_params)
                    .map(|index| {
                        FixedPointCoordinateCertificate::uncovered(format!(
                            "coordinate {index}: fixed-point evidence unavailable: {reason}"
                        ))
                    })
                    .collect(),
            )
        };

        // #2609/#2087 — select the SAME lower-envelope basin as the value and
        // analytic lanes BEFORE crossing the EFS accepted-basin mutation seam.
        // Clearing first made a cold EFS call drive the criterion from the raw
        // LSQ term while `eval` selected and installed the finite envelope
        // argmin: one rho therefore returned +inf or a finite value solely by
        // call order. An exact-rho handoff is still valid before this mutation;
        // otherwise the shared selector installs its converged argmin. The
        // streaming and `inner_max_iter == 0` freeze contracts retain their
        // historical single-pass paths: the envelope itself deliberately
        // bypasses multi-basin reconvergence in those regimes, and pre-running
        // it here would only duplicate the dominant evaluation.
        if direct_logdet_admitted && self.inner_max_iter != 0 {
            match self.install_authoritative_envelope_basin(rho_flat) {
                Ok(true) => {}
                Ok(false) => {
                    self.probe_converged_handoff = None;
                    self.basin_bundle.clear();
                    self.current_rho = rho;
                    return Ok(infeasible_evaluation(
                        "the authoritative basin envelope is infeasible at this rho",
                    ));
                }
                Err(err) if Self::is_recoverable_value_probe_refusal(&err) => {
                    self.probe_telemetry.record_refusal_kind(&err);
                    self.probe_telemetry.infeasible_criterion_evals += 1;
                    self.probe_converged_handoff = None;
                    self.basin_bundle.clear();
                    self.current_rho = rho;
                    return Ok(infeasible_evaluation(
                        "infeasible penalized quasi-Laplace basin envelope",
                    ));
                }
                Err(err) => return Err(err),
            }
        }

        // #2080/#2230 — selection above consumed any valid exact-rho handoff.
        // The EFS criterion drive below now commits that selected basin, so no
        // handoff or saved member keyed to the pre-commit accepted state may
        // survive across the mutation.
        self.probe_converged_handoff = None;
        self.basin_bundle.clear();
        if let Some(beta) = self.seeded_beta.take()
            && beta.len() == self.term.beta_dim()
        {
            self.term.set_flat_beta(beta.view())?;
        }
        // #1026 massive-K: in the streaming regime the dense evidence cache is
        // infeasible (O((K·M·p)²)), so `penalized_quasi_laplace_criterion_with_cache` hard-errors
        // ("cost-only streaming route is required"). But the EFS lane IS the
        // intended streaming-regime descent, and its ARD/smoothness traces below
        // are already matrix-free-gated — they only need the per-row factored
        // arrow cache, which the streaming criterion produces (and now returns).
        // Route through it so the Fellner–Schall step runs matrix-free at large K;
        // dense-admitted fits keep the byte-for-byte dense path.
        // #2080: the streaming criterion emits one indivisible value/gradient
        // artifact: factor cache, exact matrix-free operator, and the frozen
        // `(probes, S^-1 probes)` bundle. Reassembling the operator after the
        // value would both duplicate the dominant pass and risk differentiating
        // a different functional.
        let criterion = self.evaluate_outer_criterion_route(&rho, direct_logdet_admitted, true);
        let evaluation = match criterion {
            Ok(evaluated) => evaluated,
            Err(SaeCriterionError::VanishedAtoms(atoms)) => {
                log::debug!("SAE EFS probe reached fixed-K structural boundary: {atoms}");
                self.probe_telemetry.infeasible_criterion_evals += 1;
                self.current_rho = rho;
                return Ok(infeasible_evaluation("vanished-atom structural boundary"));
            }
            // #1782 — the EFS lane IS the SAE seed-startup-VALIDATION lane
            // (`run_fixed_point_outer_solver` → `eval_step(seed)` → `eval_efs` →
            // `efs_step`). At a seed ρ a K>1 threshold-gate/softmax (or rank-deficient
            // euclidean/linear) fit's off-optimum inner state can leave the
            // reduced joint-Hessian Schur complement indefinite, so the undamped
            // Laplace factorization refuses ("Schur complement Cholesky failed:
            // … not positive definite"), and any other infeasible-ρ-probe class
            // (non-PD per-row / cross-row joint Hessian, inner non-convergence).
            // A recoverable refusal means the quasi-Laplace score is undefined at
            // this ρ. It is an infeasible fixed-point evaluation, not a finite
            // pseudo-objective with zero updates. Returning `+inf` and uncovered
            // coordinates lets the fixed-point runner reject/backtrack without
            // ever certifying the point. Genuine defects still propagate.
            Err(SaeCriterionError::Numerical(err))
                if Self::is_recoverable_value_probe_refusal(&err) =>
            {
                self.probe_telemetry.record_refusal_kind(&err);
                log::debug!("SAE criterion eval mapped refusal to +inf: {err}");
                self.probe_telemetry.infeasible_criterion_evals += 1;
                self.current_rho = rho;
                return Ok(infeasible_evaluation(
                    "infeasible penalized quasi-Laplace score",
                ));
            }
            // #2336 — an indefinite exact `A` leaves the Laplace normaliser
            // `½log|A|` UNDEFINED at this ρ, so this evaluation is INFEASIBLE, not
            // defective. That is the same class `is_recoverable_value_probe_refusal`
            // already maps to `+inf`, for the reason its #1782 note gives: the
            // indefinite basin is adjacent to the PD optimum, so the outer solver
            // must read `+∞` and steer back into the PD region rather than abort the
            // whole fit. #2330 Phase-2a made `½log|A|` the ranked value, which is
            // what made this reachable — the majorizer `B` was PD by construction and
            // could never trip it. Escaping the saddle is the ACCEPTED lane's job
            // (the #2336 terminal escape, upstream in the criterion); by the time a
            // refusal surfaces here the escape is already exhausted, and a probe must
            // stay probe-infeasible rather than grind.
            Err(err @ SaeCriterionError::IndefiniteObservedInformation { .. }) => {
                self.probe_telemetry.record_refusal_kind(&err.to_string());
                log::debug!("SAE criterion mapped indefinite-A refusal to +inf: {err}");
                self.probe_telemetry.infeasible_criterion_evals += 1;
                self.current_rho = rho;
                return Ok(infeasible_evaluation(
                    "infeasible penalized quasi-Laplace score (indefinite exact A)",
                ));
            }
            Err(SaeCriterionError::Numerical(err)) => return Err(err),
        };
        let cost = evaluation.cost;
        self.record_fit_data_collapse_verdict(&rho)?;
        self.current_rho = rho.clone();
        if !cost.is_finite() {
            self.probe_telemetry.infeasible_criterion_evals += 1;
            return Ok(infeasible_evaluation(
                "the penalized quasi-Laplace criterion is non-finite",
            ));
        }

        // The MacKay/Fellner–Schall fixed point uses the observed row count.
        // Design-honesty weights are mean-one and only redistribute the weighted
        // coordinate sum of squares in the denominator.
        let n_eff = self.term.n_obs() as f64;
        let sumsq = self.term.ard_coord_sumsq();
        // The assignment-strength ψ coordinate has no EFS equation. Its one
        // exact gradient component comes from the complete all-coordinate
        // assembler; single-adjoint form makes this the same one solve the old
        // coordinate-specialized forward response paid.
        let complete_gradient = if rho.sparse_flat_index().is_some() || !rho.kappa.is_empty() {
            Some(
                self.analytic_gradient_for_outer_evaluation(&rho, &evaluation)
                    .map_err(|error| error.to_string())?,
            )
        } else {
            None
        };
        let cache = &evaluation.cache;
        let inverse_probe_bundle = evaluation
            .matrix_free
            .as_ref()
            .and_then(|artifacts| artifacts.efs_inverse_probe_bundle.as_ref());
        let traces = if let Some((probes, sinv)) = inverse_probe_bundle.as_ref() {
            self.term
                .ard_inverse_traces_from_probes(cache, probes, sinv)
                .map_err(|e| {
                    format!("SaeManifoldOuterObjective::efs_step: ARD traces (matrix-free): {e}")
                })?
        } else {
            self.term
                .ard_inverse_traces(cache)
                .map_err(|e| format!("SaeManifoldOuterObjective::efs_step: ARD traces: {e}"))?
        };

        // Build the flat step vector in `to_flat` layout (#1556): optional
        // assignment strength, then per-atom log_lambda_smooth, then ARD.
        let mut steps = vec![0.0_f64; n_params];
        let mut fixed_point_coordinates = (0..n_params)
            .map(|index| {
                FixedPointCoordinateCertificate::uncovered(format!(
                    "coordinate {index}: no root-equivalent fixed-point equation was evaluated"
                ))
            })
            .collect::<Vec<_>>();
        let mut psi_gradient = Vec::new();
        let mut psi_indices = Vec::new();

        // Assignment strength (when present): use the COMPLETE analytic
        // derivative of the same penalized quasi-Laplace scalar returned as `cost`.
        // This includes the explicit assignment-prior derivative, the inner-mode
        // response, and the log-determinant adjoint. In particular, learnable
        // ordered Beta--Bernoulli concentration must not take a separate occupancy-only
        // marginal fixed point: that root omits criterion terms and therefore
        // does not share this objective's stationarity equation.
        if let Some(sparse_index) = rho.sparse_flat_index() {
            assert_eq!(
                assignment_strength_gradient_coordinate(&rho),
                Some(sparse_index)
            );
            let gradient = complete_gradient
                .as_ref()
                .expect("sparse rho coordinate requested its complete analytic gradient")
                [sparse_index];
            // A normalized negative gradient is a bounded feasible-descent
            // update whose zero is exactly the full criterion root.
            let gradient_scale = gradient.abs().max(1.0);
            let step = -gradient / gradient_scale;
            steps[sparse_index] = step;
            fixed_point_coordinates[sparse_index] =
                FixedPointCoordinateCertificate::covered(step, 1.0);
            psi_gradient.push(gradient);
            psi_indices.push(sparse_index);
        }

        // Raw sectional curvature has no multiplicative EFS equation. Move it
        // by the complete analytic derivative of the same criterion, exactly as
        // the assignment-strength psi coordinate. These are the only atoms in
        // `kappa_atoms`; flat atoms emit no dummy zero-gradient coordinates.
        for &atom in &rho.kappa_atoms {
            let coordinate = rho.kappa_flat_index(atom).ok_or_else(|| {
                format!(
                    "SaeManifoldOuterObjective::efs_step: atom {atom} has curvature state but no flat coordinate"
                )
            })?;
            let gradient = complete_gradient
                .as_ref()
                .expect("curvature coordinate requested its complete analytic gradient")
                [coordinate];
            let step = -gradient / gradient.abs().max(1.0);
            steps[coordinate] = step;
            fixed_point_coordinates[coordinate] =
                FixedPointCoordinateCertificate::covered(step, 1.0);
            psi_gradient.push(gradient);
            psi_indices.push(coordinate);
        }

        // λ_smooth (layout-derived K-coordinate block): per-atom Wood-Fasiolo EFS multiplicative
        // update (#1556). The EFS fixed point is already per-coordinate, so each
        // atom `k` gets `λ_k_new = (rank_k − edof_k)/energy_k` written into its
        // own step slot. `rank_k = r_k·rank(S_k)`, `edof_k = tr_k(H⁻¹ M_k)`, and
        // `energy_k = <B_k, S_k B_k>` are the per-atom splits of the historical
        // global totals. The penalized-dimension `rank_k` uses the atom's
        // `border_frame_rank()` r_k — the number of decoder channels the `S_k`
        // roughness penalty actually acts on (`r_k == p` on the full-`B` path, the
        // smaller frame rank when a Grassmann frame is active), NOT the full output
        // dim `p`. This matches the criterion's EDF trace / penalty energy / Occam
        // derivative (all `border_frame_rank`-based); using `p` when `r_k < p`
        // overcounted the FS numerator by `(p−r_k)·rank(S_k)` and drove
        // `λ_smooth` too high on frame-active fits.
        let k_smooth = rho.log_lambda_smooth.len();
        let lambda_smooth_vec = rho.lambda_smooth_vec()?;
        let quad_per_atom = self.term.decoder_smoothness_quadratic_form_per_atom()?;
        // #2080: reuse the SAME shared (probes, S⁻¹·probes) bundle taken once above
        // for the ARD trace. When present, the smoothness EDF is the matrix-free
        // tr(S⁻¹·M_k) off that bundle (no dense `beta_inv`); otherwise (dense-
        // admitted, or no lane) fall back to the dense selected-inverse trace.
        let eff_dof_per_atom = if let Some((probes, sinv)) = inverse_probe_bundle.as_ref() {
            self.term
                .decoder_smoothness_effective_dof_per_atom_from_probes(
                    probes,
                    sinv,
                    &lambda_smooth_vec,
                )
                .map_err(|e| {
                    format!("SaeManifoldOuterObjective::efs_step: smooth dof (matrix-free): {e}")
                })?
        } else {
            self.term
                .decoder_smoothness_effective_dof_per_atom(&cache, &lambda_smooth_vec)
                .map_err(|e| format!("SaeManifoldOuterObjective::efs_step: smooth dof: {e}"))?
        };
        for atom_idx in 0..k_smooth {
            let coordinate = rho.smooth_flat_index(atom_idx);
            let lambda_k = lambda_smooth_vec[atom_idx];
            let rank_k = (self.term.atoms[atom_idx].border_frame_rank() as f64)
                * (SaeManifoldTerm::symmetric_rank(self.term.atoms[atom_idx].smooth_penalty())?
                    as f64);
            let quad_k = quad_per_atom[atom_idx];
            let eff_dof_k = eff_dof_per_atom[atom_idx];
            // Guard the FS ratio against a vanishing penalty energy or a
            // non-positive numerator (transient far from the optimum) by holding
            // that atom's λ fixed (step 0) — the cost path still moves it then.
            if !(quad_k > 0.0) {
                fixed_point_coordinates[coordinate] = FixedPointCoordinateCertificate::uncovered(
                    format!("atom {atom_idx} smoothness energy is not positive"),
                );
            } else if !(rank_k - eff_dof_k > 0.0) {
                fixed_point_coordinates[coordinate] = FixedPointCoordinateCertificate::uncovered(
                    format!("atom {atom_idx} smoothness rank-minus-edf numerator is not positive"),
                );
            } else if !(lambda_k > 0.0 && lambda_k.is_finite()) {
                fixed_point_coordinates[coordinate] = FixedPointCoordinateCertificate::uncovered(
                    format!("atom {atom_idx} smoothness precision is not finite and positive"),
                );
            } else {
                // #F1 — NO dispersion factor. The outer objective the value/gradient
                // lanes minimize is the UNIT-dispersion penalized Laplace criterion
                // `v = ½‖r‖² + ½Σ_k λ_k·B_kᵀS_kB_k + ½log|H| − ½Σ_k rank_k·log λ_k`
                // (`penalized_quasi_laplace_criterion_*`: `loss.data_fit` is the raw half-SSE, with no
                // `1/φ̂` on the data term and no `(np/2)·ln φ̂` scale term). Its
                // stationarity in `ρ_k = log λ_k` — using `edof_k = tr(H⁻¹·λ_k S_k)`
                // so `tr(H⁻¹S_k) = edof_k/λ_k` — is
                //   ½B_kᵀS_kB_k + ½·edof_k/λ_k − ½·rank_k/λ_k = 0
                //   ⇒ λ_k = (rank_k − edof_k)/B_kᵀS_kB_k,
                // with NO `φ̂`. The former `φ̂·(…)` fixed point was the textbook
                // ESTIMATED-scale GAM update; against this unit-scale criterion it
                // walked to `φ̂·λ*`, so the EFS lane and the value lane optimized two
                // different objectives inside one solve. Matches the φ̂-free value
                // gradient (`reml_occam_log_lambda_smooth_derivative` +
                // `decoder_smoothness_value_per_atom`).
                let lambda_new = (rank_k - eff_dof_k) / quad_k;
                if lambda_new.is_finite() && lambda_new > 0.0 {
                    let step = lambda_new.ln() - rho.log_lambda_smooth[atom_idx];
                    steps[coordinate] = step;
                    fixed_point_coordinates[coordinate] =
                        FixedPointCoordinateCertificate::covered(step, 1.0);
                } else {
                    fixed_point_coordinates[coordinate] =
                        FixedPointCoordinateCertificate::uncovered(format!(
                            "atom {atom_idx} smoothness equation proposed a non-finite precision"
                        ));
                }
            }
        }

        // ARD axes (after the layout-derived smooth block): Mackay fixed point
        // with posterior variance
        // (Gaussian closed form on Euclidean axes; the exact von-Mises root on
        // periodic axes, see `von_mises_ard_precision`).
        // #1026 shared-ARD: in `Shared` mode several atoms alias ONE outer
        // coordinate `sparse_dim+K+axis`, so the fixed point pools the evidence across the
        // atoms owning the axis — `α_axis_new = (count·n) / Σ_k(‖t_kj‖²+tr_kj)`
        // (#F1 — no `φ̂`) — and writes a single step. Walking a raw per-atom cursor there indexes
        // past the flat length `sparse_dim+K+max_d` (OOB) and splits one shared strength
        // across phantom slots. In `PerAtom` mode each `(k, axis)` is its own
        // coordinate and this reduces to the historical per-atom Mackay update.
        // Per-(atom, axis) periodicity: a PERIODIC (Circle) axis's empirical-Bayes
        // precision is the von-Mises root (`von_mises_ard_precision`), NOT the
        // Gaussian closed form `denom` alone encodes; a non-periodic (Euclidean)
        // axis keeps the exact Gaussian Mackay/FS update unchanged.
        let ard_periods: Vec<Vec<Option<f64>>> = self
            .term
            .assignment
            .coords
            .iter()
            .map(|c| c.effective_axis_periods())
            .collect();
        match rho.ard_sharing() {
            ArdSharing::PerAtom => {
                for (k, axis_logard) in rho.log_ard.iter().enumerate() {
                    for (j, &logard_kj) in axis_logard.iter().enumerate() {
                        let coordinate = rho.ard_flat_index(k, j);
                        let denom = sumsq[k][j] + traces[k][j];
                        if denom > 0.0 {
                            // #F1 — NO dispersion factor (same unit-dispersion
                            // criterion as λ_smooth). The Gaussian coordinate prior
                            // contributes `+½α‖t‖² − ½·n_eff·log α` to the unit-scale
                            // `v`, and `½log|H|` contributes `½α·tr(H⁻¹)`; stationarity
                            // in `log α` gives `α(‖t‖² + tr) = n_eff`, i.e. `α_new =
                            // n_eff/denom` with NO `φ̂` — matching the φ̂-free value
                            // gradient `ard_log_precision_explicit_derivatives`
                            // (`normalizer_deriv = −½·n_eff`). The former `φ̂·n_eff/…`
                            // walked the ARD precision to `φ̂·α*`.
                            let alpha_gauss = n_eff / denom;
                            let alpha_new = match ard_periods[k].get(j).copied().flatten() {
                                Some(period) => von_mises_ard_precision(
                                    alpha_gauss,
                                    std::f64::consts::TAU / period,
                                ),
                                None => alpha_gauss,
                            };
                            if alpha_new.is_finite() && alpha_new > 0.0 {
                                let step = alpha_new.ln() - logard_kj;
                                steps[coordinate] = step;
                                fixed_point_coordinates[coordinate] =
                                    FixedPointCoordinateCertificate::covered(step, 1.0);
                            } else {
                                fixed_point_coordinates[coordinate] =
                                    FixedPointCoordinateCertificate::uncovered(format!(
                                        "atom {k} ARD axis {j} equation proposed a non-finite precision"
                                    ));
                            }
                        } else {
                            fixed_point_coordinates[coordinate] =
                                FixedPointCoordinateCertificate::uncovered(format!(
                                    "atom {k} ARD axis {j} posterior second moment is not positive"
                                ));
                        }
                    }
                }
            }
            ArdSharing::Shared => {
                let max_d = rho.max_ard_axes();
                for axis in 0..max_d {
                    let mut denom = 0.0_f64;
                    let mut count = 0usize;
                    let mut shared_logard = 0.0_f64;
                    let mut shared_period: Option<f64> = None;
                    for (k, axis_logard) in rho.log_ard.iter().enumerate() {
                        if axis < axis_logard.len() {
                            denom += sumsq[k][axis] + traces[k][axis];
                            // Broadcast table: every owner carries the same value.
                            shared_logard = axis_logard[axis];
                            // Owners aliasing one shared axis share its geometry, so
                            // the period is common; take the first owner's.
                            if shared_period.is_none() {
                                shared_period = ard_periods[k].get(axis).copied().flatten();
                            }
                            count += 1;
                        }
                    }
                    let coordinate = rho.ard_flat_index(0, axis);
                    if count == 0 {
                        fixed_point_coordinates[coordinate] =
                            FixedPointCoordinateCertificate::uncovered(format!(
                                "shared ARD axis {axis} has no owning atom"
                            ));
                    } else if !(denom > 0.0) {
                        fixed_point_coordinates[coordinate] =
                            FixedPointCoordinateCertificate::uncovered(format!(
                                "shared ARD axis {axis} posterior second moment is not positive"
                            ));
                    } else {
                        // #F1 — NO dispersion factor (see the PerAtom branch). The
                        // shared axis pools `count` owners' evidence, so `n_eff` is
                        // lifted by `count`; the φ̂-free form is `α_new =
                        // count·n_eff/denom`.
                        let alpha_gauss = n_eff * (count as f64) / denom;
                        let alpha_new = match shared_period {
                            Some(period) => {
                                von_mises_ard_precision(alpha_gauss, std::f64::consts::TAU / period)
                            }
                            None => alpha_gauss,
                        };
                        if alpha_new.is_finite() && alpha_new > 0.0 {
                            let step = alpha_new.ln() - shared_logard;
                            steps[coordinate] = step;
                            fixed_point_coordinates[coordinate] =
                                FixedPointCoordinateCertificate::covered(step, 1.0);
                        } else {
                            fixed_point_coordinates[coordinate] =
                                FixedPointCoordinateCertificate::uncovered(format!(
                                    "shared ARD axis {axis} equation proposed a non-finite precision"
                                ));
                        }
                    }
                }
            }
        }

        // Block weights (trailing L-1 coordinates): the crosscoder block-relevance
        // Fellner–Schall step (#2231 Inc-B stage 2). The `#F1` criterion's
        // explicit data + Jacobian channels are stationary in `log λ_ℓ` at
        // `R̃_ℓ = n·p_ℓ` (block ½·R̃_ℓ − n·p_ℓ/2 = 0; see
        // `block_log_lambda_gradient`), and `R̃_ℓ = λ_ℓ·R_ℓ`, so the
        // multiplicative fixed point `λ_ℓ_new = n·p_ℓ/R_ℓ = λ_ℓ·n·p_ℓ/R̃_ℓ`
        // becomes the ADDITIVE log-space step `Δlog λ_ℓ = ln(n·p_ℓ/R̃_ℓ)`. This
        // is a PROPOSAL heuristic (like the λ_smooth/ARD EFS steps above): the
        // full analytic gradient additionally carries the `−½·Γᵀθ̂_ρ` Laplace
        // adjoint (`crosscoder_block_ift_rhs`), an `O(dim H / (n·p_ℓ))` relative
        // correction the quasi-Newton lane prices exactly; EFS proposals are
        // still accepted only on criterion improvement, so the heuristic root
        // cannot bias the fitted λ. Held (step 0) for a block with no residual
        // variance (`R̃_ℓ ≤ 0`: perfectly reconstructed / unidentifiable) or a
        // non-finite proposal, matching the λ_smooth/ARD guards above. No-op for a
        // plain SAE.
        if let Some(scaled_rss) = self.block_scaled_rss(&rho)? {
            let n = self.term.n_obs() as f64;
            let blocks = self
                .crosscoder_blocks
                .as_ref()
                .expect("block_scaled_rss returned Some ⇒ crosscoder pricing is installed");
            let tail = n_params - rho.kappa.len() - blocks.block_dims.len();
            for (l, (&p_l, &r_tilde)) in blocks.block_dims.iter().zip(scaled_rss.iter()).enumerate()
            {
                let coordinate = tail + l;
                if r_tilde > 0.0 {
                    let step = (n * p_l as f64 / r_tilde).ln();
                    if step.is_finite() {
                        steps[coordinate] = step;
                        fixed_point_coordinates[coordinate] =
                            FixedPointCoordinateCertificate::uncovered(format!(
                                "crosscoder block {l} EFS proposal omits the logdet IFT adjoint and is not a complete stationarity equation"
                            ));
                    } else {
                        fixed_point_coordinates[coordinate] =
                            FixedPointCoordinateCertificate::uncovered(format!(
                                "crosscoder block {l} equation proposed a non-finite update"
                            ));
                    }
                } else {
                    fixed_point_coordinates[coordinate] =
                        FixedPointCoordinateCertificate::uncovered(format!(
                            "crosscoder block {l} scaled residual energy is not positive"
                        ));
                }
            }
        }

        let beta_hat = self.term.flatten_beta();
        self.last_loss = Some(evaluation.loss);
        let consecutive_restored_incumbents = self
            .term
            .best_fit_incumbent
            .as_ref()
            .map(|incumbent| incumbent.consecutive_inner_restores);
        Ok((
            EfsEval {
                cost,
                steps,
                beta: Some(beta_hat),
                psi_gradient: (!psi_gradient.is_empty()).then(|| Array1::from_vec(psi_gradient)),
                psi_indices: (!psi_indices.is_empty()).then_some(psi_indices),
                inner_hessian_scale: None,
                logdet_enclosure_gap: None,
                consecutive_restored_incumbents,
            },
            fixed_point_coordinates,
        ))
    }
}

/// Correct the Gaussian Mackay/Fellner–Schall ARD precision proposal to the
/// EXACT von-Mises empirical-Bayes fixed point on a PERIODIC axis.
///
/// The closed-form update `α_gauss = n_eff/(Σ q + tr H⁻¹)` (#F1 — no `φ̂`) is the
/// stationary precision only for a Gaussian coordinate prior, whose normalized
/// log-partition contributes `−½ n_eff log α` (ρ-derivative `−½ n_eff`). On a
/// periodic (von-Mises) axis the normalized prior's log-partition is
/// `log P − η + log I0(η)`, `η = α/κ²`, whose ρ-derivative is
/// `n_eff·η·(A(η)−1)` with `A(η) = I1(η)/I0(η)` — the Gaussian `−½ n_eff` is only
/// its `η→∞` limit (`A(η) ≈ 1 − 1/(2η)`). Setting the criterion's ρ-derivative to
/// zero over the SAME `denom = Σ q + tr H⁻¹` the Gaussian update uses collapses to
///   `A(η) = 1 − 1/(2·η_gauss)`,  `η_gauss = α_gauss/κ²`,
/// so the correction → `α_gauss` in the `η→∞` limit (`A(η) ≈ 1 − 1/(2η)`) and only
/// re-scales the diffuse regime the Gaussian surrogate mis-ranks. It differs from
/// `α_gauss` at every finite η by design, so the bit-for-bit-unchanged guarantee
/// holds only for Euclidean (`period = None`) axes, which bypass this function
/// entirely. When the target ratio leaves `(0,1)` the root is ill-posed
/// (`η_gauss ≤ ½`: maximally diffuse) and the Gaussian proposal is returned
/// unchanged (no regression). `A` is strictly increasing on `(0,∞)`, so the root
/// is found by monotone safeguarded bisection using the crate's stable `I1/I0`
/// evaluator. The posterior-variance term keeps the plain Fellner–Schall trace
/// surrogate `T = Σ w·(H⁻¹)ᵢᵢ` (not the exact `cos`-weighted `Σ w·(α cos κt)ᵢ(H⁻¹)ᵢᵢ`);
/// this refines the analytically-dominant normalizer channel to the von-Mises form
/// while the outer ρ-gradient (`ard_log_precision_explicit_derivatives`) stays the
/// exact, value-consistent objective the step is safeguarded against.
fn von_mises_ard_precision(alpha_gauss: f64, kappa: f64) -> f64 {
    if !(alpha_gauss.is_finite() && alpha_gauss > 0.0 && kappa.is_finite() && kappa > 0.0) {
        return alpha_gauss;
    }
    let kappa2 = kappa * kappa;
    let eta_gauss = alpha_gauss / kappa2;
    // Exact stationarity over the shared denominator: A(η) = 1 − 1/(2·η_gauss).
    let a_target = 1.0 - 0.5 / eta_gauss;
    if !(a_target > 0.0 && a_target < 1.0) {
        return alpha_gauss;
    }
    let a_of = |eta: f64| bessel_i0_log_and_ratio(eta).1;
    // Bracket the monotone root around η_gauss (A increasing in η).
    let mut lo = eta_gauss;
    let mut hi = eta_gauss;
    let mut guard = 0;
    while lo > f64::MIN_POSITIVE && a_of(lo) > a_target && guard < 256 {
        lo *= 0.5;
        guard += 1;
    }
    guard = 0;
    while hi.is_finite() && a_of(hi) < a_target && guard < 256 {
        hi *= 2.0;
        guard += 1;
    }
    if !(lo.is_finite() && hi.is_finite() && lo > 0.0 && hi > lo) {
        return alpha_gauss;
    }
    for _ in 0..80 {
        let mid = 0.5 * (lo + hi);
        if a_of(mid) < a_target {
            lo = mid;
        } else {
            hi = mid;
        }
    }
    let alpha = kappa2 * 0.5 * (lo + hi);
    if alpha.is_finite() && alpha > 0.0 {
        alpha
    } else {
        alpha_gauss
    }
}

/// Exact scale of the decoder data curvature relative to one unit of an atom's
/// native smoothing penalty. This is the largest generalized eigenvalue of
/// `(G, P)` on `range(P)`, computed as
/// `lambda_max(P⁺¹ᐟ² G P⁺¹ᐟ²)`. Choosing `lambda` at this value is the
/// *minimal* native penalty strength that is at least as strong as the
/// likelihood on every penalized decoder direction. The former trace bound
/// `tr(P⁺G)` had the same dominance property but summed the curvature of every
/// mode, inflating the entry by up to the penalty rank and over-smoothing the
/// fixed-rho corrector. Null directions remain identified by the likelihood and
/// are deliberately absent from this scale.
fn reactive_smooth_curvature_scale(
    term: &SaeManifoldTerm,
    assignments: &Array2<f64>,
    atom_idx: usize,
) -> Result<Option<f64>, String> {
    let atom = &term.atoms[atom_idx];
    let m = atom.basis_values.ncols();
    if atom.smooth_penalty().dim() != (m, m) {
        return Err(format!(
            "reactive rho domain: atom {atom_idx} smooth penalty shape {:?} != ({m}, {m})",
            atom.smooth_penalty().dim()
        ));
    }
    let penalty_geometry =
        gam_linalg::utils::rank_certified_psd_pseudoinverse(atom.smooth_penalty(), 1.0e-10)
            .map_err(|error| format!("reactive rho domain penalty spectrum failed: {error}"))?;
    let rank = penalty_geometry.rank();
    let penalty_pinv = penalty_geometry.into_pseudoinverse();
    if rank == 0 {
        return Ok(None);
    }

    let whitens = term
        .row_metric
        .as_ref()
        .is_some_and(gam_problem::RowMetric::whitens_likelihood);
    let mut data_gram = Array2::<f64>::zeros((m, m));
    for row in 0..term.n_obs() {
        let honesty_weight = term
            .row_loss_weights
            .as_ref()
            .map_or(1.0, |weights| weights[row]);
        let metric_norm_bound = match term.row_metric.as_ref() {
            Some(metric) if whitens => metric.row_traces()[row],
            _ => 1.0,
        };
        let gate = assignments[[row, atom_idx]];
        let weight = honesty_weight * metric_norm_bound * gate * gate;
        if !(weight.is_finite() && weight >= 0.0) {
            return Err(format!(
                "reactive rho domain: atom {atom_idx} row {row} has invalid data-curvature weight {weight}"
            ));
        }
        for left in 0..m {
            let weighted_left = weight * atom.basis_values[[row, left]];
            for right in 0..m {
                data_gram[[left, right]] += weighted_left * atom.basis_values[[row, right]];
            }
        }
    }

    // Form the PSD square root of P⁺ on exactly the retained penalty range.
    // The declared penalty pseudoinverse cutoff owns the rank decision;
    // retaining the largest `rank` eigenpairs here reuses that decision without
    // reviving tiny numerical eigenvalues in the null space. This second EVD is
    // strict too: no jitter may change the retained range.
    let (pinv_eigenvalues, pinv_eigenvectors) =
        gam_linalg::faer_ndarray::strict_symmetric_eigh(&penalty_pinv, Side::Lower)
            .map_err(|error| format!("reactive rho domain P⁺ spectrum failed: {error}"))?;
    if !pinv_eigenvalues.iter().all(|value| value.is_finite()) {
        return Err(format!(
            "reactive rho domain: atom {atom_idx} P⁺ spectrum is non-finite"
        ));
    }
    let mut order: Vec<usize> = (0..m).collect();
    order.sort_by(|&left, &right| {
        pinv_eigenvalues[right]
            .partial_cmp(&pinv_eigenvalues[left])
            .unwrap_or(std::cmp::Ordering::Equal)
    });
    let mut scaled_vectors = Array2::<f64>::zeros((m, m));
    for &col in order.iter().take(rank) {
        let eigenvalue = pinv_eigenvalues[col];
        if !(eigenvalue.is_finite() && eigenvalue > 0.0) {
            return Err(format!(
                "reactive rho domain: atom {atom_idx} retained P⁺ eigenvalue is invalid ({eigenvalue})"
            ));
        }
        let scale = eigenvalue.sqrt();
        for row in 0..m {
            scaled_vectors[[row, col]] = pinv_eigenvectors[[row, col]] * scale;
        }
    }
    let pinv_sqrt = scaled_vectors.dot(&pinv_eigenvectors.t());
    let mut standardized_curvature = pinv_sqrt.dot(&data_gram).dot(&pinv_sqrt);
    for row in 0..m {
        for col in 0..row {
            let symmetric =
                0.5 * (standardized_curvature[[row, col]] + standardized_curvature[[col, row]]);
            standardized_curvature[[row, col]] = symmetric;
            standardized_curvature[[col, row]] = symmetric;
        }
    }
    let (generalized_eigenvalues, _) = standardized_curvature
        .eigh(Side::Lower)
        .map_err(|error| format!("reactive rho domain generalized spectrum failed: {error}"))?;
    if !generalized_eigenvalues
        .iter()
        .all(|value| value.is_finite())
    {
        return Err(format!(
            "reactive rho domain: atom {atom_idx} generalized decoder spectrum is non-finite"
        ));
    }
    let largest = generalized_eigenvalues
        .iter()
        .copied()
        .fold(0.0_f64, f64::max);
    Ok((largest > 0.0).then_some(largest))
}

/// Exact observed Gauss--Newton scale of one Euclidean latent coordinate at the
/// diffuse scalar entry. Native Gaussian ARD curvature has unit coefficient
/// before multiplying by `alpha`, so this is the data-curvature-matching
/// precision for that axis.
///
/// A periodic axis deliberately returns `None`: its exact von-Mises Hessian is
/// `alpha * cos(kappa * t)` and therefore changes sign around the chart. Raising
/// `alpha` cannot convexify a full circle; it instead makes the half-chart around
/// the antipode *more* indefinite and contracts a genuine loop toward the
/// arbitrary phase origin. Periodic ARD consequently has no legal heavy-entry
/// value above its literal target. Keeping that coordinate fixed is still a
/// lockstep scalar path (its entry and target endpoints are identical).
fn reactive_ard_curvature_scale(
    term: &SaeManifoldTerm,
    assignments: &Array2<f64>,
    atom_idx: usize,
    axis: usize,
) -> Result<Option<f64>, String> {
    let periods = term.assignment.coords[atom_idx].effective_axis_periods();
    if periods.get(axis).copied().flatten().is_some() {
        return Ok(None);
    }
    observed_ard_curvature_scale(term, assignments, atom_idx, axis).map(Some)
}

/// The observed latent Gauss--Newton curvature of one ARD axis, with NO geometry
/// policy attached: `max_row  w_row * ||d(gated decode)/dt||^2` in the row metric.
///
/// This is the whole body [`reactive_ard_curvature_scale`] used to have. It is
/// split out because that function answers a CONTINUATION question and this
/// quantity answers a DOMAIN one, and #2691 is the case where conflating them
/// cost the coordinate its upper face. Nothing in this loop reads the axis
/// period — the tangent norms come from the basis Jacobian and the decoder — so
/// the periodic decline was never a statement that the number does not exist for
/// a circle. It exists, and `periodic_ard_domain_upper` consumes it.
///
/// Native Gaussian ARD curvature has unit coefficient before multiplying by
/// `alpha`, so this IS the data-curvature-matching precision for the axis: at
/// `alpha` equal to this value the prior's curvature equals the strongest
/// curvature the data puts on the coordinate, and above it the prior owns the
/// coordinate outright.
fn observed_ard_curvature_scale(
    term: &SaeManifoldTerm,
    assignments: &Array2<f64>,
    atom_idx: usize,
    axis: usize,
) -> Result<f64, String> {
    let atom = &term.atoms[atom_idx];
    let p = atom.decoder_coefficients().ncols();
    let m = atom.decoder_coefficients().nrows();
    if atom.basis_jacobian.dim().1 != m || axis >= atom.basis_jacobian.dim().2 {
        return Err(format!(
            "reactive rho domain: atom {atom_idx} axis {axis} is incompatible with basis Jacobian {:?} and decoder {:?}",
            atom.basis_jacobian.dim(),
            atom.decoder_coefficients().dim()
        ));
    }
    let whitens = term
        .row_metric
        .as_ref()
        .is_some_and(gam_problem::RowMetric::whitens_likelihood);
    let mut tangent = vec![0.0_f64; p];
    let mut maximum = 0.0_f64;
    for row in 0..term.n_obs() {
        tangent.fill(0.0);
        let gate = assignments[[row, atom_idx]];
        for basis in 0..m {
            let coefficient = gate * atom.basis_jacobian[[row, basis, axis]];
            for out in 0..p {
                tangent[out] += coefficient * atom.decoder_coefficients()[[basis, out]];
            }
        }
        let tangent_norm_sq = match term.row_metric.as_ref() {
            Some(metric) if whitens => metric
                .whiten_residual_row(row, ArrayView1::from(tangent.as_slice()))
                .into_iter()
                .map(|value| value * value)
                .sum::<f64>(),
            _ => tangent.iter().map(|value| value * value).sum(),
        };
        let honesty_weight = term
            .row_loss_weights
            .as_ref()
            .map_or(1.0, |weights| weights[row]);
        let curvature = honesty_weight * tangent_norm_sq;
        if !(curvature.is_finite() && curvature >= 0.0) {
            return Err(format!(
                "reactive rho domain: atom {atom_idx} axis {axis} row {row} has invalid latent data curvature {curvature}"
            ));
        }
        maximum = maximum.max(curvature);
    }
    Ok(maximum)
}

/// Objective-owned legal rho upper face for dense reactive SAE fits.
///
/// The generic `+30` box corresponds to a penalty strength around `1e13` and is
/// outside the SAE objective's structural domain: at that strength every atom's
/// gated reconstruction signal is shrunk under the backward-error vanishing
/// boundary, so `DictionaryCollapseVerdict::all_decoders_vanished`
/// (`manifold::fit_drivers`) holds before continuation can start — the
/// dictionary is numerically gone, not merely uncompetitive.
///
/// This sentence used to name a `q/n` "signal-free null floor", which #2498
/// deleted. The hard collapse decision no longer rests on a training-`R²`
/// statistic at all: it is proved from the fitted state's floating-point
/// residual backward-error envelope, so the bound above is about decoder signal
/// disappearing into numerical noise rather than about explained variance
/// falling under a sample-size ratio.
///
/// This contract instead uses the objective's literal entry assignments and
/// native penalty geometry. Decoder smoothness is bounded by the exact largest
/// generalized eigenvalue of `(G, P)`; Euclidean ARD is bounded by its observed
/// latent Gauss--Newton curvature. Periodic ARD stays at its literal target
/// because the von-Mises Hessian changes sign and therefore cannot define a
/// convexifying heavy-entry direction. Every bound is at least the literal
/// target strength. No criterion probe or fitted-state trial participates in
/// constructing the box.
fn reactive_rho_domain_upper(
    term: &SaeManifoldTerm,
    rho: &SaeManifoldRho,
    entry_temperature: f64,
) -> Result<Array1<f64>, String> {
    term.assignment.validate_rho_domain(rho)?;
    let mut entry_term = term.clone();
    entry_term
        .assignment
        .mode
        .set_temperature(entry_temperature)?;
    entry_term.temperature_schedule = None;
    let assignments = entry_term.assignment.try_assignments()?;
    let target = rho.to_flat();
    let mut upper = target.clone();
    let mut largest_native_scale = 0.0_f64;

    for atom_idx in 0..rho.k_atoms() {
        if let Some(scale) = reactive_smooth_curvature_scale(&entry_term, &assignments, atom_idx)? {
            largest_native_scale = largest_native_scale.max(scale);
            let index = rho.smooth_flat_index(atom_idx);
            let target_strength = target[index].exp();
            upper[index] = target_strength.max(scale).ln();
        }
        for axis in 0..rho.log_ard[atom_idx].len() {
            if let Some(scale) =
                reactive_ard_curvature_scale(&entry_term, &assignments, atom_idx, axis)?
            {
                largest_native_scale = largest_native_scale.max(scale);
                let index = rho.ard_flat_index(atom_idx, axis);
                let target_strength = target[index].exp();
                upper[index] = upper[index].max(target_strength.max(scale).ln());
            }
        }
    }

    // Fixed-alpha ordered Beta--Bernoulli carries no assignment-strength dependence. Every other
    // present assignment coordinate is capped on the same largest observed
    // native-curvature scale, rather than inheriting the unrelated generic
    // `exp(30)` strength.
    if let Some(index) = rho.sparse_flat_index()
        && !matches!(
            entry_term.assignment.mode,
            AssignmentMode::OrderedBetaBernoulli {
                learnable_alpha: false,
                ..
            }
        )
        && largest_native_scale > 0.0
    {
        let target_strength = target[index].exp();
        upper[index] = target_strength.max(largest_native_scale).ln();
    }

    if upper.iter().all(|value| value.is_finite()) {
        Ok(upper)
    } else {
        Err(format!(
            "reactive rho domain produced a non-finite upper face: {upper:?}"
        ))
    }
}

/// #2691 — the domain face for a PERIODIC ARD axis, in log precision, one entry
/// per `(atom, axis)` that carries a period. It is the MINIMUM of two derived
/// faces: the chart-resolution face below, and the data-curvature-matching
/// precision `observed_ard_curvature_scale`. Neither is a chosen number.
///
/// [`reactive_ard_curvature_scale`] declines a periodic axis, and its reason is
/// a CONTINUATION argument: raising `alpha` cannot convexify a full circle, so
/// there is no legal heavy-entry value above the literal target. That answers
/// whether `alpha` may be RAISED as a reactive entry point. It does not answer
/// what the coordinate's admissible DOMAIN is, and declining to answer is how
/// the coordinate inherited `gam_problem::log_strength::LOG_STRENGTH_MAX` — a
/// binary64 normal-range / overflow-margin policy, by its own module doc. A
/// number about floating-point representability was serving as the admissible
/// set of a quantity about chart geometry.
///
/// The geometric face, derived from quantities this system already owns:
///
/// * the per-axis prior is `V(t) = (alpha / kappa^2) * (1 - cos(kappa t))`
///   with `kappa = TAU / P` (`atom::ArdAxisPrior`), whose
///   curvature at its unique minimum `t = 0` is exactly `alpha`. A precision
///   `alpha` is a concentration of scale `sigma_t = alpha^(-1/2)` — the
///   precision-to-scale relation of the prior itself, not a calibration;
/// * the chart coordinate the occupancy adjudicator reads is the UNIT-PERIOD
///   fold `u = t / P` (`coordinate_fidelity::fold_for_occupancy_weighted`), so
///   that same prior scale is `sigma_u = 1 / (P * sqrt(alpha))`;
/// * `coordinate_fidelity::classify_occupancy` returns
///   `coordinate_fidelity::OccupancyLaw::Collapsed` once the
///   occupied extent falls below its own data-derived resolution floor
///   `sigma_floor = 1 / (2n)` — half the mean spacing of `n` points on the unit
///   circle. That is this system's own statement of when a chart stops being a
///   chart.
///
/// Requiring the prior's own scale to remain representable by the chart:
///
/// ```text
///   sigma_u(alpha) >= sigma_floor
///     <=>  1 / (P * sqrt(alpha)) >= 1 / (2n)
///     <=>  alpha <= (2n / P)^2
///     <=>  log alpha <= 2 * (ln(2n) - ln P)
/// ```
///
/// Past that face the prior asks for structure finer than anything the
/// coordinate can resolve, so `Collapsed` is forced no matter what the data
/// says. Every input is a dimension or a geometry the system already knows:
/// `n` is the row count, `P` is the axis period
/// (`SaeCoordinates::effective_axis_periods`), the `2n` denominator is
/// `sigma_floor`'s own, and the `-1/2` exponent is the prior's precision-scale
/// relation. No literal is introduced, and nothing here is calibrated.
///
/// `n` is deliberately the full row count rather than the per-atom support
/// effective sample size the weighted adjudicator uses: `ess <= n`, so this is
/// the MOST PERMISSIVE form of the rule — the face below is the largest
/// precision at which any support weighting could still resolve the chart.
///
/// The resolution face ALONE is necessary and not sufficient, and the gap was
/// measured rather than reasoned: on the #2691 circle fixture (n=70, p=8,
/// sigma=0.352) circular recovery R^2 falls 0.969 -> 0.037 between `alpha = 1e1`
/// and `alpha = 1e2`, TWO ORDERS below the resolution face at `alpha = 1.96e4`,
/// because `Collapsed` is a strictly weaker condition than "the chart still
/// carries the structure". That gap is what the second face closes, and the
/// second face is not a tightened version of the first -- it is a different
/// quantity, the largest curvature the DATA puts on the coordinate, which the
/// objective already computes for every Euclidean axis.
///
/// A Euclidean axis is absent on purpose. Its `u` fold is normalized by the
/// OBSERVED span, so `sigma_floor` carries no absolute length there and this
/// derivation has no content; that axis already receives the curvature face
/// through [`reactive_ard_curvature_scale`] on the reactive path.
pub(crate) fn periodic_ard_domain_upper(
    term: &SaeManifoldTerm,
    rho: &SaeManifoldRho,
    assignments: &Array2<f64>,
) -> Result<Vec<(usize, f64)>, String> {
    let n_rows = term.n_obs();
    let mut faces = Vec::new();
    if n_rows < 2 {
        return Ok(faces);
    }
    for atom_idx in 0..rho.k_atoms() {
        if atom_idx >= term.assignment.coords.len() {
            return Err(format!(
                "periodic ARD chart-resolution domain: atom {atom_idx} has no assignment \
                 coordinates ({} present)",
                term.assignment.coords.len()
            ));
        }
        let periods = term.assignment.coords[atom_idx].effective_axis_periods();
        for axis in 0..rho.log_ard[atom_idx].len() {
            let Some(period) = periods.get(axis).copied().flatten() else {
                continue;
            };
            if !(period.is_finite() && period > 0.0) {
                return Err(format!(
                    "periodic ARD chart-resolution domain: atom {atom_idx} axis {axis} has \
                     non-positive period {period}"
                ));
            }
            let resolution_face = 2.0 * ((2.0 * n_rows as f64) / period).ln();
            if !resolution_face.is_finite() {
                return Err(format!(
                    "periodic ARD chart-resolution domain: atom {atom_idx} axis {axis} produced a \
                     non-finite face from n={n_rows} period={period}"
                ));
            }
            // #2691 second face — the DATA-CURVATURE-MATCHING precision. The
            // resolution face alone was measured too loose by two orders on this
            // issue's own fixture (circular recovery R^2 fell 0.969 -> 0.037
            // between alpha = 1e1 and 1e2, while the resolution face sits at
            // alpha = 1.96e4), because `Collapsed` is a strictly weaker condition
            // than "the chart still carries the structure". The binding quantity
            // is the one the objective already computes for EUCLIDEAN axes and
            // then declines to compute here: `observed_ard_curvature_scale`, the
            // largest curvature the data puts on this coordinate. Native Gaussian
            // ARD curvature has unit coefficient before `alpha`, so `alpha` equal
            // to it is exactly where the prior's curvature matches the data's;
            // above it the prior owns the coordinate. Same derivation discipline
            // as the resolution face: the number is read off the fitted operator,
            // not chosen.
            let curvature = observed_ard_curvature_scale(term, assignments, atom_idx, axis)?;
            let face = if curvature > 0.0 {
                resolution_face.min(curvature.ln())
            } else {
                // A coordinate the data puts NO curvature on has no evidence face;
                // the resolution face is then the only derived statement available.
                resolution_face
            };
            if !face.is_finite() {
                return Err(format!(
                    "periodic ARD domain: atom {atom_idx} axis {axis} produced a non-finite face \
                     from resolution={resolution_face} curvature={curvature}"
                ));
            }
            faces.push((rho.ard_flat_index(atom_idx, axis), face));
        }
    }
    Ok(faces)
}

impl OuterObjective for SaeManifoldOuterObjective {
    fn capability(&self) -> OuterCapability {
        let gradient = sae_outer_gradient_capability();
        // SAE's Fellner--Schall/MacKay updates are useful simultaneous proposal
        // directions, but their zeros omit the profiled criterion's state-
        // response/third-order channels. They may drive a fixed-point solver
        // only when the complete analytic gradient is available to certify the
        // terminal KKT root. Matrix-free SAE currently lacks that proof surface
        // and must refuse rather than mint a surrogate fixed point (#2253).
        let exact_gradient_certificate = matches!(gradient, Derivative::Analytic);
        let psi_gradient_dim =
            usize::from(assignment_strength_gradient_coordinate(&self.baseline_rho).is_some())
                + self.baseline_rho.kappa.len();
        OuterCapability {
            // The planner always has an analytic outer update. Two regimes:
            //  * Dense-admitted: the exact analytic outer gradient is assembled
            //    from the joint-Hessian IFT (`outer_gradient_arrow_solver`), for
            //    every assignment mode, including ordered Beta--Bernoulli (#1006).
            //  * Matrix-free (dense criterion factor exceeds the in-core budget,
            //    e.g. large-K / wide-border duchon): the rational value emits one
            //    frozen inverse-probe bundle, and the complete gradient consumes
            //    it plus one matrix-free adjoint solve. No dense cache or synthetic
            //    zero derivative enters this route.
            gradient,
            // The profiled SAE criterion currently exposes an exact analytic
            // gradient but no exact second derivative. Never advertise curvature
            // manufactured by perturbing ρ and re-solving the inner problem: that
            // is finite differencing, is basin-history dependent, and violates the
            // production derivative contract. An exact fixed-stratum HVP can
            // replace this declaration when its adjoint derivative is implemented.
            hessian: DeclaredHessianForm::Unavailable,
            n_params: self.baseline_rho.to_flat().len(),
            // Softmax/threshold fits have one non-FS coordinate: assignment
            // strength. Mark it as the Hybrid-EFS analytic-gradient block so
            // scalable EFS updates still own smoothness/ARD while this coordinate
            // moves by its exact penalized quasi-Laplace gradient. Small dense fits still select the
            // ordinary full-gradient BFGS plan at the existing crossover.
            psi_dim: if exact_gradient_certificate {
                psi_gradient_dim
            } else {
                0
            },
            // The SAE path minimizes its explicitly named custom quasi-Laplace
            // criterion. The extended Fellner--Schall fixed point needs only the traces
            // tr(H⁻¹ S_c) (decoder_smoothness_effective_dof + ard_inverse_traces),
            // never a finite-difference or autodiff gradient — which is required
            // here because the per-atom-ARD outer problem is O(K)-dimensional and a
            // gradient/BFGS descent over it costs O(K) inner fits per step,
            // intractable at large K. EFS updates all coords SIMULTANEOUSLY from a
            // single trace pass, so it scales. The #1023 boundary-collapse (EFS
            // railing λ_smooth and collapsing the decoder to the mean) is guarded
            // two ways now: efs_step's update targets the finite penalized quasi-Laplace stationary
            // point λ_new = (rank−edof)/energy (#F1 — the unit-dispersion fixed
            // point the value criterion's ∂/∂ρ = 0 defines; `rank−edof ≤ rank`
            // bounded and `energy > 0`, so λ cannot rail to a mean-collapse).
            // Fitted-data collapse is recorded separately as a structure-search
            // verdict and never changes this fixed-point objective.
            fixed_point_available: exact_gradient_certificate,
            barrier_config: None,
            prefer_gradient_only: false,
            disable_fixed_point: !exact_gradient_certificate,
        }
    }

    fn eval_cost(&mut self, rho: &Array1<f64>) -> Result<f64, EstimationError> {
        self.check_cancelled()?;
        // Value-only cross-seed ranking / EFS path (seed screening, final
        // selection, and backtracking). Although no derivative is consumed at
        // this iterate, its finite value selects a seed/state and therefore
        // prices the same fully converged `f(ρ)` as the analytic gradient lane.
        // A reduced-budget finite iterate would define a second objective; only
        // an explicit refusal may represent unfinished computation.
        self.probe_telemetry.criterion_calls += 1;
        // #2230/#2087 — descend the basin lower envelope V*(ρ)=min_b V_b(ρ) here
        // instead of the single hysteretic warm-start trajectory. The shared
        // authoritative drive is also used by line-search and gradient handoff
        // repair; the envelope owns streaming / freeze bypass semantics.
        match self.authoritative_envelope_value_probe(rho.view()) {
            Ok((cost, _beta)) => {
                // #2231 Inc-B — price the block-relevance Jacobian into the SAME
                // cost that flows to `termination.record` (0 for a plain SAE).
                let rho_state = self
                    .baseline_rho
                    .from_flat(rho.view())
                    .map_err(EstimationError::InvalidInput)?;
                let cost = cost + self.block_jacobian(&rho_state);
                if !cost.is_finite() {
                    return Ok(f64::INFINITY);
                }
                if self.reactive_waypoint_checkpoint.is_none()
                    && self.record_search_criterion(cost, None)
                {
                    self.bank_checkpoint(rho);
                }
                Ok(cost)
            }
            // A recoverable fixed-ρ refusal means the quasi-Laplace score is
            // undefined. Cost-only objectives use `+inf` as their conventional
            // infeasible result; no finite pseudo-objective is introduced.
            Err(err) if Self::is_recoverable_value_probe_refusal(&err) => {
                self.probe_telemetry.record_refusal_kind(&err);
                log::debug!("SAE criterion eval mapped refusal to +inf: {err}");
                self.probe_telemetry.infeasible_criterion_evals += 1;
                Ok(f64::INFINITY)
            }
            Err(err) => Err(EstimationError::RemlOptimizationFailed(err)),
        }
    }

    fn eval(&mut self, rho: &Array1<f64>) -> Result<OuterEval, EstimationError> {
        self.check_cancelled()?;
        self.probe_telemetry.criterion_calls += 1;
        let rho_state = self
            .baseline_rho
            .from_flat(rho.view())
            .map_err(EstimationError::InvalidInput)?;
        // #2231 Inc-B — scale the block columns for this ρ before either the
        // streaming value path or the dense `penalized_quasi_laplace_criterion_with_cache` below
        // reads `self.target` (idempotent; no-op for a plain SAE).
        self.apply_block_scaling(&rho_state)
            .map_err(EstimationError::InvalidInput)?;
        // #1026 — matrix-free (streaming) regime: the dense joint-Hessian evidence
        // cache does not exist, so the analytic gradient lane below
        // (`penalized_quasi_laplace_criterion_with_cache` → `outer_gradient_arrow_solver`) cannot run
        // and hard-errors ("cost-only streaming route is required"). The outer plan
        // descends ρ via the value + Fellner–Schall (EFS) route
        // (`fixed_point_available`), which never consumes this gradient — but the
        // generic seed startup-VALIDATION still probes this gradient lane, and its
        // hard error rejects EVERY seed ("no candidate seeds passed outer startup
        // validation") for any large-K / wide-border (duchon) fit whose dense
        // criterion factor exceeds the in-core budget. Route it to the SAME streaming
        // value path the `Value` order uses: validation then gets a finite streaming
        // penalized quasi-Laplace cost (paired with a zero gradient it never consumes) and the fit
        // proceeds on the EFS lane. Dense-admitted fits never enter this branch and
        // are byte-for-byte unchanged.
        if !self.audit_installed_state
            && !self
                .term
                .streaming_plan()
                .map_err(EstimationError::RemlOptimizationFailed)?
                .direct_logdet_admitted()
        {
            // Seed validation still selects whether this fit exists, so its
            // streaming value must be the same fully converged fixed point used
            // by every dense ranking/value lane.
            let (cost, _beta_hat) = match self.evaluate_authoritative_criterion(rho.view()) {
                Ok(evaluated) => evaluated,
                // A recoverable refusal means the streaming quasi-Laplace score is
                // undefined at this ρ. Return the objective contract's typed
                // infeasible evaluation, never a finite surrogate value.
                Err(err) if Self::is_recoverable_value_probe_refusal(&err) => {
                    self.probe_telemetry.record_refusal_kind(&err);
                    log::debug!("SAE criterion eval mapped refusal to +inf: {err}");
                    self.probe_telemetry.infeasible_criterion_evals += 1;
                    return Ok(OuterEval::infeasible(rho.len()));
                }
                Err(err) => return Err(EstimationError::RemlOptimizationFailed(err)),
            };
            // #2231 Inc-B — price the block Jacobian into the streaming-lane cost
            // (0 for a plain SAE), so the recorded and returned value agree.
            let cost = cost + self.block_jacobian(&rho_state);
            if !cost.is_finite() {
                return Ok(OuterEval::infeasible(rho.len()));
            }
            if self.record_search_criterion(cost, None) {
                self.bank_checkpoint(rho);
            }
            return Ok(OuterEval {
                cost,
                gradient: Array1::zeros(rho.len()),
                hessian: HessianValue::Unavailable,
                inner_beta_hint: None,
            });
        }
        // #2080/#2087/#2253/#2510 — every dense analytic sample begins in the
        // authoritative envelope argmin. The shared installer consumes a valid
        // exact-rho probe handoff or runs the selector on a miss; either path
        // installs a converged state and consumes the pending seeded-β hint.
        // The selector owns the amortized basin-entry warm start, so no second
        // warm-start drive is permitted after installation.
        match self.install_authoritative_envelope_basin(rho.view()) {
            Ok(true) => {}
            Ok(false) => return Ok(OuterEval::infeasible(rho.len())),
            Err(err) if Self::is_recoverable_value_probe_refusal(&err) => {
                self.probe_telemetry.record_refusal_kind(&err);
                log::debug!("SAE criterion eval mapped refusal to +inf: {err}");
                self.probe_telemetry.infeasible_criterion_evals += 1;
                return Ok(OuterEval::infeasible(rho.len()));
            }
            Err(err) => return Err(EstimationError::RemlOptimizationFailed(err)),
        }
        // Dense and streaming analytic samples use one route-selected authority.
        // The streaming artifact retains the exact matrix-free system and frozen
        // inverse-probe bundle that produced its rational-logdet value; no dense
        // retry or independently reassembled operator is allowed.
        // #1782 — a RECOVERABLE inner-solve refusal (a probed ρ whose undamped
        // joint Hessian is non-PD / whose inner solve cannot converge at that ρ)
        // is an INFEASIBLE-ρ signal, NOT a fatal defect: the value-only lanes
        // (`Value` order above, streaming branch) already map it to a `+∞`
        // infeasible eval so the outer optimizer steers back into the PD region.
        // This gradient lane previously `?`-propagated the SAME refusal as a fatal
        // `RemlOptimizationFailed`, which — because the SAE fit runs a single
        // seed (`max_seeds = 1`, no fallback) — aborted the WHOLE fit at "no
        // candidate seeds passed outer startup validation" for the assignment /
        // topology combinations whose seed or a walk probe lands on such a ρ,
        // while ordered_beta_bernoulli (whose seed happens to stay PD) survived. Treat it the
        // same infeasible way here so the three lanes agree; a genuinely
        // non-recoverable error still propagates.
        let direct_logdet_admitted = self
            .term
            .streaming_plan()
            .map_err(EstimationError::RemlOptimizationFailed)?
            .direct_logdet_admitted();
        let evaluation =
            match self.evaluate_outer_criterion_route(&rho_state, direct_logdet_admitted, false) {
                Ok(evaluated) => evaluated,
                Err(SaeCriterionError::VanishedAtoms(atoms)) => {
                    log::debug!(
                        "SAE analytic evaluation reached fixed-K structural boundary: {atoms}"
                    );
                    self.probe_telemetry.infeasible_criterion_evals += 1;
                    return Ok(OuterEval::infeasible(rho.len()));
                }
                // A non-PD per-row/cross-row/Schur factor has no defined Laplace
                // evidence at this ρ. Return the objective contract's typed
                // infeasible evaluation so the optimizer rejects/backtracks. A
                // finite sentinel here would be a different objective. Genuine
                // evaluation defects still hard-error below.
                Err(SaeCriterionError::Numerical(err))
                    if Self::is_recoverable_value_probe_refusal(&err) =>
                {
                    self.probe_telemetry.record_refusal_kind(&err);
                    log::debug!("SAE criterion eval mapped refusal to +inf: {err}");
                    self.probe_telemetry.infeasible_criterion_evals += 1;
                    return Ok(OuterEval::infeasible(rho.len()));
                }
                // #2336 — an indefinite exact `A` leaves the Laplace normaliser
                // `½log|A|` UNDEFINED at this ρ, so this evaluation is INFEASIBLE, not
                // defective. That is the same class `is_recoverable_value_probe_refusal`
                // already maps to `+inf`, for the reason its #1782 note gives: the
                // indefinite basin is adjacent to the PD optimum, so the outer solver
                // must read `+∞` and steer back into the PD region rather than abort the
                // whole fit. #2330 Phase-2a made `½log|A|` the ranked value, which is
                // what made this reachable — the majorizer `B` was PD by construction and
                // could never trip it. Escaping the saddle is the ACCEPTED lane's job
                // (the #2336 terminal escape, upstream in the criterion); by the time a
                // refusal surfaces here the escape is already exhausted, and a probe must
                // stay probe-infeasible rather than grind.
                Err(err @ SaeCriterionError::IndefiniteObservedInformation { .. }) => {
                    self.probe_telemetry.record_refusal_kind(&err.to_string());
                    log::debug!("SAE criterion mapped indefinite-A refusal to +inf: {err}");
                    self.probe_telemetry.infeasible_criterion_evals += 1;
                    return Ok(OuterEval::infeasible(rho.len()));
                }
                Err(SaeCriterionError::Numerical(err)) => {
                    return Err(EstimationError::RemlOptimizationFailed(err));
                }
            };
        let cost = evaluation.cost;
        self.record_fit_data_collapse_verdict(&rho_state)
            .map_err(EstimationError::RemlOptimizationFailed)?;
        if !cost.is_finite() {
            self.probe_telemetry.infeasible_criterion_evals += 1;
            return Ok(OuterEval::infeasible(rho.len()));
        }
        let gradient = self
            .analytic_gradient_for_outer_evaluation(&rho_state, &evaluation)
            .map_err(EstimationError::from)?;
        let beta_hat = self.term.flatten_beta();
        // PATH C (#2253) — assemble the exact fixed-stratum outer Hessian from the
        // landed analytic channels. The assembler currently REFUSES (only the
        // solver-free explicit channel is implemented), so this yields
        // `Unavailable` and the planner stays on the analytic-gradient BFGS route
        // that `capability()` declares. When every channel lands the assembler
        // succeeds, this becomes a `Dense` curvature, and `capability()` flips to
        // `Dense` so the small-dense planner routes ARC through it.
        let hessian = match self.term.exact_fixed_stratum_outer_hessian(
            self.target.view(),
            &rho_state,
            &evaluation.loss,
            &evaluation.cache,
        ) {
            Ok(dense) => HessianValue::Dense(dense),
            Err(_incomplete) => HessianValue::Unavailable,
        };
        // #1206 — the gradient lane (`OuterEvalOrder::ValueAndGradient`, consumed
        // by the outer BFGS Armijo line search) MUST return a cost whose gradient
        // is the gradient we return: the consistent pair `(f, ∇f)` for the pure
        // penalized quasi-Laplace criterion — the SAME criterion every value/ranking/EFS lane prices
        // (one coherent objective; see `evaluate_authoritative_inner`). Collapse was
        // recorded above as a structural verdict and leaves this value unchanged.
        // #2231 Inc-B — price the block Jacobian into the gradient lane's cost so
        // the value it records matches the value/ranking/EFS lanes (0 for a plain
        // SAE). This Jacobian and the block-tail gradient populated above
        // (`½·R̃_ℓ − n·p_ℓ/2`) are the desync-safe (#2087) `(value, gradient)` pair:
        // the cost carries `−(n·p_ℓ/2)·log λ_ℓ`, whose derivative is the `−n·p_ℓ/2`
        // half of that gradient entry, and the scaled-block residual carries the
        // `½·R̃_ℓ` half through the data term.
        let cost = cost + self.block_jacobian(&rho_state);
        // The gradient is the EXACT implicit derivative: `outer_gradient_arrow_
        // solver` solves the implicit-function system through the rank-revealing
        // gauge/decoder-null deflation (Rayleigh-band + Faddeev–Popov stiffness),
        // and a genuinely singular system surfaced above as a typed
        // `OuterGradientError` instead of a degraded direction. No secondary
        // finite-difference safeguard is layered on top (SPEC: FD never leaves
        // tests) — a near-flat inner direction that corrupts the `Γ·θ̂_ρ`
        // envelope term is a deflation-candidate gap to fix in
        // `outer_gradient_arrow_solver`, not something to paper over with a
        // differenced value path.
        self.current_rho = rho_state;
        self.last_loss = Some(evaluation.loss);
        if self.record_search_criterion(cost, Some(gradient.dot(&gradient).sqrt())) {
            self.bank_checkpoint(rho);
        }
        Ok(OuterEval {
            cost,
            gradient,
            hessian,
            inner_beta_hint: Some(beta_hat),
        })
    }

    fn eval_with_order(
        &mut self,
        rho: &Array1<f64>,
        order: OuterEvalOrder,
    ) -> Result<OuterEval, EstimationError> {
        // #2138 — cover the line-search cost-probe lane too: the `Value` order is
        // called directly by the outer bridge (bypassing `eval`/`eval_cost`), so
        // without this a cancelled worker parked in a long probe sequence would
        // keep grinding. Idempotent for the gradient orders (they also delegate to
        // `eval`, which checks again); no-op when no cancel flag is installed.
        self.check_cancelled()?;
        match order {
            OuterEvalOrder::Value => {
                // The `Value` order is the BFGS / ARC LINE-SEARCH cost probe
                // (see `solver/rho_optimizer/bridges.rs`). Its cost is compared
                // against steps whose DIRECTION came from `eval`'s penalized
                // quasi-Laplace `∇f`, which is the exact implicit derivative
                // through the FULLY converged inner fixed point
                // (`penalized_quasi_laplace_criterion_with_cache`, the idempotent
                // `gradient_stationary && criterion_fixed_point` root). The line
                // search can only accept a step when the value it ranks prices
                // the SAME inner state that gradient differentiates. The former
                // line-search-probe drive priced a FREEZE / coarse-KKT iterate
                // that at real scale (n≈44k, ill-conditioned inner solve) sits
                // ~1% off that fixed point, so NO step reduced the ranked value
                // while pointing down the gradient — BFGS backtracked to
                // `StepSizeTooSmall` at iteration 1 and shipped the coarse value,
                // which the outer certification then rejected against the
                // idempotent analytic sample ("cost-only value disagrees with
                // analytic-sample value"). Price through the SAME `Criterion`
                // drive the GRADIENT lane (`eval` →
                // `penalized_quasi_laplace_criterion_with_cache`) and the outer
                // certification use — the authoritative FULL-refine budget.
                // Every value/ranking lane shares this authority: a coarse
                // iterate ~1% off it (real scale, #2228) makes every step fail
                // Armijo (StepSizeTooSmall), and using that iterate to rank seeds
                // selects against a different function. At small n both budgets
                // reach the fixed point so this is a no-op (tier0 unchanged); the
                // regression test
                // `value_lane_prices_at_shared_fixed_point` pins the invariant so a
                // future rewrite reintroducing `false` here goes red.
                let (cost, beta_hat) = match self.authoritative_envelope_value_probe(rho.view()) {
                    Ok(evaluated) => evaluated,
                    // A recoverable non-PD/non-converged probe has undefined
                    // quasi-Laplace score. `OuterEval::infeasible` is the
                    // line-search contract for rejection/backtracking and carries
                    // no derivative.
                    Err(err) if Self::is_recoverable_value_probe_refusal(&err) => {
                        self.probe_telemetry.record_refusal_kind(&err);
                        log::debug!("SAE criterion eval mapped refusal to +inf: {err}");
                        self.probe_telemetry.infeasible_criterion_evals += 1;
                        // A reactive waypoint is a typed domain transaction,
                        // not an opaque line-search comparison. Preserve the
                        // objective's exact refusal reason so continuation can
                        // report why the legal entry or a refined waypoint was
                        // undefined. The surrounding transaction still rolls
                        // the complete objective state back before refinement.
                        if self.reactive_waypoint_checkpoint.is_some() {
                            return Err(EstimationError::RemlOptimizationFailed(format!(
                                "reactive coupled waypoint has undefined penalized quasi-Laplace score: {err}"
                            )));
                        }
                        return Ok(OuterEval::infeasible(rho.len()));
                    }
                    Err(err) => return Err(EstimationError::RemlOptimizationFailed(err)),
                };
                // #2231 Inc-B — price the block Jacobian into the line-search
                // probe cost (0 for a plain SAE) so the value the outer search
                // ranks matches the gradient/EFS lanes.
                let rho_state = self
                    .baseline_rho
                    .from_flat(rho.view())
                    .map_err(EstimationError::InvalidInput)?;
                let cost = cost + self.block_jacobian(&rho_state);
                if !cost.is_finite() {
                    return Ok(OuterEval::infeasible(rho.len()));
                }
                if self.reactive_waypoint_checkpoint.is_none()
                    && self.record_search_criterion(cost, None)
                {
                    self.bank_checkpoint(rho);
                }
                Ok(OuterEval {
                    cost,
                    gradient: Array1::zeros(rho.len()),
                    hessian: HessianValue::Unavailable,
                    inner_beta_hint: Some(beta_hat),
                })
            }
            OuterEvalOrder::ValueAndGradient => self.eval(rho),
            OuterEvalOrder::ValueGradientHessian => self.eval(rho),
        }
    }

    fn eval_efs(&mut self, rho: &Array1<f64>) -> Result<EfsEval, EstimationError> {
        // #2138 — the Fellner–Schall route is a primary outer descent path with its
        // own inner solve (bypassing `eval`/`eval_cost`), so cover it too.
        self.check_cancelled()?;
        let mut eval = self
            .efs_step(rho.view())
            .map_err(EstimationError::RemlOptimizationFailed)?;
        // #2231 Inc-B — price the block Jacobian into the EFS cost (0 for a plain
        // SAE) so the value recorded and returned matches the value/gradient lanes.
        // `efs_step` already populated the block-tail Fellner–Schall step
        // `Δlog λ_ℓ = ln(n·p_ℓ/R̃_ℓ)`, so the EFS descent moves the block λ toward
        // the same root the gradient lane vanishes at.
        let rho_state = self
            .baseline_rho
            .from_flat(rho.view())
            .map_err(EstimationError::InvalidInput)?;
        eval.cost += self.block_jacobian(&rho_state);
        if self.record_search_criterion(eval.cost, None) {
            self.bank_checkpoint(rho);
        }
        Ok(eval)
    }

    fn eval_fixed_point_certificate(
        &mut self,
        rho: &Array1<f64>,
    ) -> Result<FixedPointCertificateEval, EstimationError> {
        self.check_cancelled()?;
        // EFS/MacKay step zeros are not the stationarity equations of the full
        // profiled quasi-Laplace objective: the exact gradient also contains
        // logdet state response, the third-order correction, and rank-response
        // channels. Re-evaluate the authoritative analytic sample and expose its
        // negative gradient as the signed feasible-descent residual. The generic
        // fixed-point certificate projects this vector at the rho box and applies
        // the same tolerance as the first-order optimizer, so an EFS proposal can
        // accelerate iteration but can never certify a different root (#2253).
        let evaluation = self.eval(rho)?;
        let coordinates = evaluation
            .gradient
            .iter()
            .map(|&gradient| FixedPointCoordinateCertificate::covered(-gradient, 1.0))
            .collect();
        Ok(FixedPointCertificateEval {
            cost: evaluation.cost,
            coordinates,
        })
    }

    fn reset(&mut self) {
        self.reactive_waypoint_checkpoint = None;
        self.fit_verdict = None;
        self.term = self.baseline_term.clone();
        if let Some(registry) = self.registry.as_mut() {
            registry.set_isometry_scalar_weights(&self.baseline_isometry_weights);
        }
        self.current_rho = self.baseline_rho.clone();
        self.last_loss = None;
        self.terminal_penalized_quasi_laplace_criterion = None;
        self.seeded_beta = None;
        // #2080 (a) — a reset replaces the accepted basin; a probe handoff from
        // the previous seed's basin must not warm-start the new one.
        self.probe_converged_handoff = None;
        // #2230/#2087 — a multi-start reset starts a NEW outer walk; the previous
        // seed's saved basins are meaningless for it.
        self.basin_bundle.clear();
        self.termination.reset_improvement_baseline();
    }

    fn seed_inner_state(&mut self, beta: &Array1<f64>) -> Result<SeedOutcome, EstimationError> {
        self.fit_verdict = None;
        // An empty-β seed means "no warm-start available; preserve the
        // objective-owned state" and must be a no-op. Typed reactive-domain
        // entry begins without a coefficient hint, while cache replay only
        // calls this hook for a populated coefficient vector owned by the exact
        // outer seed. Only a populated β must match the decoder dimension.
        if beta.is_empty() {
            // NoSlot says that this call installed nothing. A subsequent exact
            // evaluation may publish a populated `inner_beta_hint` for the next
            // typed reactive waypoint.
            return Ok(SeedOutcome::NoSlot);
        }
        if beta.len() != self.term.beta_dim() {
            return Err(EstimationError::RemlOptimizationFailed(format!(
                "SaeManifoldOuterObjective::seed_inner_state: β length {} != decoder dim {}",
                beta.len(),
                self.term.beta_dim()
            )));
        }
        self.seeded_beta = Some(beta.clone());
        // #2080 (a) — a freshly installed β seed is a NEW instruction the pending
        // probe trajectory never saw; drop the handoff so the next evaluation
        // applies the seed instead of a converged state that predates it.
        self.probe_converged_handoff = None;
        // #2230/#2087 — a fresh β seed is a NEW instruction the saved basins never
        // saw; drop them so the envelope re-seeds from the seeded accepted basin.
        self.basin_bundle.clear();
        Ok(SeedOutcome::Installed)
    }

    fn outer_domain_upper_bound(&self) -> Result<Option<Array1<f64>>, EstimationError> {
        self.baseline_term
            .assignment
            .validate_rho_domain(&self.baseline_rho)
            .map_err(EstimationError::InvalidInput)?;
        let mut log_strength_upper = self.baseline_rho.flat_domain_upper_bound();
        if let Some((_, alpha_upper)) = self
            .baseline_term
            .assignment
            .learnable_alpha_rho_domain()
            .map_err(EstimationError::InvalidInput)?
            && let (Some(bounds), Some(index)) = (
                log_strength_upper.as_mut(),
                self.baseline_rho.sparse_flat_index(),
            )
        {
            bounds[index] = bounds[index].min(alpha_upper);
        }
        let curvature_bounds = self.curvature_domain_bounds()?;
        // #2691 — the periodic ARD face, applied on BOTH exits of this function.
        // The reactive construction below never runs at all for K < 2, and
        // declines periodic axes even when it does, so without this the
        // chart-coordinate precision keeps the generic binary64 representability
        // face and the outer search is free to drive the prior past the point
        // where the chart stops carrying structure. The gate reads the baseline
        // term's own assignments because the data-curvature half of the face is
        // a gated quantity; a domain query that cannot read them is a real
        // failure and is reported rather than degraded to the looser face.
        let baseline_assignments = self
            .baseline_term
            .assignment
            .try_assignments()
            .map_err(EstimationError::RemlOptimizationFailed)?;
        let chart_faces = periodic_ard_domain_upper(
            &self.baseline_term,
            &self.baseline_rho,
            &baseline_assignments,
        )
        .map_err(EstimationError::RemlOptimizationFailed)?;
        // Every other face in this function keeps the literal target strength
        // inside the box (`target_strength.max(scale)`); this one does the same,
        // so a caller-installed ARD entry can never be made infeasible by a
        // domain query.
        let target = self.baseline_rho.to_flat();
        let Some(contract) = self.reactive_domain_scalar_contract()? else {
            if let Some(bounds) = log_strength_upper.as_mut() {
                for &(index, face) in &chart_faces {
                    bounds[index] = bounds[index].min(face.max(target[index]));
                }
                for &(index, _, upper) in &curvature_bounds {
                    bounds[index] = upper;
                }
            }
            return Ok(log_strength_upper);
        };
        // The reactive entry replaces the invalid common cold dictionary with a
        // deterministic disjoint-chart placement. Derive the legal rho face
        // from that SAME placed geometry, not from the common-chart literal
        // seed: its gated basis Grams (and Euclidean ARD curvature for flat
        // atoms) are the operators the entry corrector will actually see.
        // Work on a clone because querying an optimizer domain is read-only.
        let mut entry_term = self.baseline_term.clone();
        entry_term
            .assignment
            .mode
            .set_temperature(contract.entry().assignment_temperature)
            .map_err(EstimationError::RemlOptimizationFailed)?;
        entry_term.temperature_schedule = None;
        entry_term
            .place_reactive_entry_disjoint_charts(self.target.view())
            .map_err(|error| {
                EstimationError::RemlOptimizationFailed(format!(
                    "reactive rho domain could not construct its separated entry geometry: {error}"
                ))
            })?;
        let mut reactive_upper = reactive_rho_domain_upper(
            &entry_term,
            &self.baseline_rho,
            contract.entry().assignment_temperature,
        )
        .map_err(EstimationError::RemlOptimizationFailed)?;
        if let Some(log_strength_upper) = log_strength_upper {
            for index in 0..reactive_upper.len() {
                reactive_upper[index] = reactive_upper[index].min(log_strength_upper[index]);
            }
        }
        // #2691 — the same chart-resolution face on the reactive exit. The
        // reactive construction leaves a periodic ARD coordinate at its literal
        // target, which is not a claim about the domain.
        for &(index, face) in &chart_faces {
            reactive_upper[index] = reactive_upper[index].min(face.max(target[index]));
        }
        // Reactive-domain construction knows only log-strength coordinates.
        // Curvature is a raw, scale-dependent coordinate, so its typed geometry
        // rail replaces (rather than intersects) that generic placeholder.
        for &(index, _, upper) in &curvature_bounds {
            reactive_upper[index] = upper;
        }
        Ok(Some(reactive_upper))
    }

    fn outer_domain_lower_bound(&self) -> Result<Option<Array1<f64>>, EstimationError> {
        self.baseline_term
            .assignment
            .validate_rho_domain(&self.baseline_rho)
            .map_err(EstimationError::InvalidInput)?;
        let mut lower = self.baseline_rho.flat_domain_lower_bound();
        if let Some((alpha_lower, _)) = self
            .baseline_term
            .assignment
            .learnable_alpha_rho_domain()
            .map_err(EstimationError::InvalidInput)?
            && let (Some(bounds), Some(index)) =
                (lower.as_mut(), self.baseline_rho.sparse_flat_index())
        {
            bounds[index] = bounds[index].max(alpha_lower);
        }
        if let Some(bounds) = lower.as_mut() {
            for (index, curvature_lower, _) in self.curvature_domain_bounds()? {
                bounds[index] = curvature_lower;
            }
        }
        Ok(lower)
    }

    /// Dense K≥2 joint fits may have undefined quasi-Laplace score at the literal
    /// cold seed even though a finite basin is connected from a diffuse routing
    /// state. The entry temperature is derived from the objective's own routing
    /// logits: it is the smallest temperature that puts every active logit on
    /// unit scale. Isometry entry weights are zero, while the target retains the
    /// literal per-penalty vector (including heterogeneous weights).
    fn reactive_domain_scalar_contract(
        &self,
    ) -> Result<Option<gam_solve::continuation_path::ContinuationScalarContract>, EstimationError>
    {
        if self.baseline_term.k_atoms() < 2
            || !self
                .baseline_term
                .streaming_plan()
                .map_err(EstimationError::RemlOptimizationFailed)?
                .direct_logdet_admitted()
        {
            return Ok(None);
        }

        let target_temperature = self.baseline_term.assignment.mode.temperature();
        let routing_logits = self
            .baseline_term
            .assignment
            .frozen_logits
            .as_ref()
            .unwrap_or(&self.baseline_term.assignment.logits);
        let threshold = match self.baseline_term.assignment.mode {
            AssignmentMode::ThresholdGate { threshold, .. } => threshold,
            _ => 0.0,
        };
        let mut routing_scale = 0.0_f64;
        for &logit in routing_logits {
            let centered = logit - threshold;
            if !centered.is_finite() {
                return Err(EstimationError::RemlOptimizationFailed(
                    "reactive scalar continuation found a non-finite literal routing logit"
                        .to_string(),
                ));
            }
            routing_scale = routing_scale.max(centered.abs());
        }
        let entry = gam_solve::continuation_path::ContinuationScalarState::new(
            target_temperature.max(routing_scale),
            vec![0.0; self.baseline_isometry_weights.len()],
        )
        .map_err(EstimationError::RemlOptimizationFailed)?;
        let target = gam_solve::continuation_path::ContinuationScalarState::new(
            target_temperature,
            self.baseline_isometry_weights.clone(),
        )
        .map_err(EstimationError::RemlOptimizationFailed)?;
        gam_solve::continuation_path::ContinuationScalarContract::new(entry, target)
            .map(Some)
            .map_err(EstimationError::RemlOptimizationFailed)
    }

    fn install_reactive_domain_scalar_state(
        &mut self,
        state: &gam_solve::continuation_path::ContinuationScalarState,
    ) -> Result<(), EstimationError> {
        let contract = self
            .reactive_domain_scalar_contract()?
            .ok_or_else(|| {
                EstimationError::RemlOptimizationFailed(
                    "reactive scalar waypoint requested from an objective without a dense K>=2 contract"
                        .to_string(),
                )
            })?;
        if state.isometry_weights.len() != self.baseline_isometry_weights.len() {
            return Err(EstimationError::RemlOptimizationFailed(format!(
                "reactive scalar waypoint isometry dimension {} != literal target dimension {}",
                state.isometry_weights.len(),
                self.baseline_isometry_weights.len(),
            )));
        }

        self.fit_verdict = None;
        self.term
            .assignment
            .mode
            .set_temperature(state.assignment_temperature)
            .map_err(EstimationError::RemlOptimizationFailed)?;
        let installing_entry = state.bitwise_eq(contract.entry());
        let restoring_target = state.bitwise_eq(contract.target());
        // A private inner schedule must not advance or overwrite any coupled
        // waypoint, including the exact s=0 solve. The literal baseline schedule
        // is restored atomically only after that target solve commits.
        self.term.temperature_schedule = None;
        if let Some(registry) = self.registry.as_mut() {
            registry.set_isometry_scalar_weights(&state.isometry_weights);
        }
        if installing_entry {
            if self.reactive_waypoint_checkpoint.is_none() {
                return Err(EstimationError::RemlOptimizationFailed(
                    "reactive scalar entry placement requires an active full-state waypoint transaction"
                        .to_string(),
                ));
            }
            // The literal seed was already evaluated before the runner opened
            // this repair path and its penalized quasi-Laplace score was undefined. Do not carry
            // that invalid basin into the supposedly legal entry merely because
            // its decoder coefficients are nonzero: a common cold seed fits every
            // atom independently to the full target, so summing those decoders is
            // badly off-model and the ordinary zero-decoder cold-start detector
            // cannot recognize it. At the exact diffuse/heavy-smoothing endpoint,
            // install the objective's data-derived disjoint chart + sequential
            // decoder placement. Periodic atoms use the certified joint ISA split
            // when the measure supports it and the residual-PCA peel otherwise.
            // This mutation is inside the waypoint's full-state transaction: a
            // failed entry rolls back every coordinate, logit, decoder, frame, and
            // scalar; an accepted entry commits the separated basin that warms the
            // next coupled waypoint. Finite literal seeds never open this path,
            // and later accepted warm waypoints are not reinitialized.
            self.term
                .place_reactive_entry_disjoint_charts(self.target.view())
                .map_err(|err| {
                    EstimationError::RemlOptimizationFailed(format!(
                        "reactive scalar entry could not install its separated legal basin: {err}"
                    ))
                })?;
            // Rebuild the exact face from the just-installed geometry. This is
            // bitwise the same deterministic placement used by
            // `outer_domain_upper_bound`; parsing it through the baseline rho
            // layout gives the per-atom lambda values the first corrector will
            // evaluate. Refit the separated decoders at those strengths before
            // the joint solve so the heavy entry does not begin with the large
            // score of an unpenalized full-signal decoder.
            let entry_rho_flat = reactive_rho_domain_upper(
                &self.term,
                &self.baseline_rho,
                state.assignment_temperature,
            )
            .map_err(EstimationError::RemlOptimizationFailed)?;
            let entry_rho = self
                .baseline_rho
                .from_flat(entry_rho_flat.view())
                .map_err(EstimationError::InvalidInput)?;
            self.term
                .refit_reactive_entry_decoders_at_smooth_face(
                    self.target.view(),
                    &entry_rho,
                )
                .map_err(|err| {
                    EstimationError::RemlOptimizationFailed(format!(
                        "reactive scalar entry could not fit its separated decoders at the legal smooth face: {err}"
                    ))
                })?;
        }
        self.probe_converged_handoff = None;
        self.basin_bundle.clear();
        self.probe_telemetry.reactive_scalar_installs += 1;
        if restoring_target {
            self.probe_telemetry.reactive_target_restores += 1;
        }
        Ok(())
    }

    fn begin_reactive_domain_waypoint(&mut self) -> Result<(), EstimationError> {
        if self.reactive_waypoint_checkpoint.is_some() {
            return Err(EstimationError::RemlOptimizationFailed(
                "reactive coupled waypoint began while another waypoint transaction was active"
                    .to_string(),
            ));
        }
        let bundle_capacity = self.basin_bundle.member_capacity();
        let basin_bundle =
            std::mem::replace(&mut self.basin_bundle, BasinBundle::new(bundle_capacity));
        let registry_isometry_weights = self
            .registry
            .as_ref()
            .map(AnalyticPenaltyRegistry::isometry_scalar_weights)
            .unwrap_or_default();
        self.reactive_waypoint_checkpoint = Some(ReactiveWaypointCheckpoint {
            term: self.term.clone(),
            target: self.target.clone(),
            registry_isometry_weights,
            current_rho: self.current_rho.clone(),
            last_loss: self.last_loss.clone(),
            terminal_penalized_quasi_laplace_criterion: self
                .terminal_penalized_quasi_laplace_criterion,
            seeded_beta: self.seeded_beta.clone(),
            probe_converged_handoff: self.probe_converged_handoff.take(),
            basin_bundle,
            termination: self.termination.clone(),
            fit_verdict: self.fit_verdict,
            crosscoder_blocks: self.crosscoder_blocks.clone(),
        });
        Ok(())
    }

    fn commit_reactive_domain_waypoint(
        &mut self,
        rho: &Array1<f64>,
    ) -> Result<(), EstimationError> {
        if self.reactive_waypoint_checkpoint.is_none() {
            return Err(EstimationError::RemlOptimizationFailed(
                "reactive coupled waypoint commit had no active transaction".to_string(),
            ));
        }
        let converged_term = self
            .take_probe_converged_handoff(rho.view())
            .ok_or_else(|| {
                EstimationError::RemlOptimizationFailed(
                    "reactive coupled waypoint produced no exact-rho converged full-state handoff"
                        .to_string(),
                )
            })?;
        let rho_state = self
            .baseline_rho
            .from_flat(rho.view())
            .map_err(EstimationError::InvalidInput)?;
        let target_contract = self.reactive_domain_scalar_contract()?.ok_or_else(|| {
            EstimationError::RemlOptimizationFailed(
                "active reactive waypoint lost its scalar contract before commit".to_string(),
            )
        })?;
        let committed_isometry_weights = self
            .registry
            .as_ref()
            .map(AnalyticPenaltyRegistry::isometry_scalar_weights)
            .unwrap_or_default();
        let committed_scalar = gam_solve::continuation_path::ContinuationScalarState::new(
            converged_term.assignment.mode.temperature(),
            committed_isometry_weights,
        )
        .map_err(EstimationError::RemlOptimizationFailed)?;
        let committed_literal_target = committed_scalar.bitwise_eq(target_contract.target());
        let loss = converged_term
            .loss(self.target.view(), &rho_state)
            .map_err(EstimationError::RemlOptimizationFailed)?;
        self.term = converged_term;
        if committed_literal_target {
            self.term.temperature_schedule = self.baseline_term.temperature_schedule.clone();
            self.term
                .assignment
                .mode
                .set_temperature(target_contract.target().assignment_temperature)
                .map_err(EstimationError::RemlOptimizationFailed)?;
        }
        self.current_rho = rho_state;
        self.last_loss = Some(loss);
        self.seeded_beta = None;
        self.fit_verdict = None;
        self.terminal_penalized_quasi_laplace_criterion = None;
        self.reactive_waypoint_checkpoint = None;
        Ok(())
    }

    fn rollback_reactive_domain_waypoint(&mut self) -> Result<(), EstimationError> {
        let checkpoint = self.reactive_waypoint_checkpoint.take().ok_or_else(|| {
            EstimationError::RemlOptimizationFailed(
                "reactive coupled waypoint rollback had no active transaction".to_string(),
            )
        })?;
        self.term = checkpoint.term;
        self.target = checkpoint.target;
        if let Some(registry) = self.registry.as_mut() {
            registry.set_isometry_scalar_weights(&checkpoint.registry_isometry_weights);
        }
        self.current_rho = checkpoint.current_rho;
        self.last_loss = checkpoint.last_loss;
        self.terminal_penalized_quasi_laplace_criterion =
            checkpoint.terminal_penalized_quasi_laplace_criterion;
        self.seeded_beta = checkpoint.seeded_beta;
        self.probe_converged_handoff = checkpoint.probe_converged_handoff;
        self.basin_bundle = checkpoint.basin_bundle;
        self.termination = checkpoint.termination;
        self.fit_verdict = checkpoint.fit_verdict;
        self.crosscoder_blocks = checkpoint.crosscoder_blocks;
        Ok(())
    }
}

pub(crate) fn sae_manifold_newton_directional_decrease(
    sys: &ArrowSchurSystem,
    delta_ext_coord: ArrayView1<'_, f64>,
    delta_beta: ArrayView1<'_, f64>,
) -> f64 {
    // delta_ext_coord has variable-stride layout for heterogeneous systems.
    assert_eq!(delta_ext_coord.len(), sys.row_offsets[sys.rows.len()]);
    assert_eq!(delta_beta.len(), sys.k);
    let mut gradient_dot_step = 0.0;
    for (row_idx, row) in sys.rows.iter().enumerate() {
        let row_base = sys.row_offsets[row_idx];
        let di = sys.row_dims[row_idx];
        for axis in 0..di {
            gradient_dot_step += row.gt[axis] * delta_ext_coord[row_base + axis];
        }
    }
    for idx in 0..sys.k {
        gradient_dot_step += sys.gb[idx] * delta_beta[idx];
    }
    -gradient_dot_step
}

/// Per-atom decoder-smoothness GEMM `S_k · B_k`, batched across ALL GPUs.
///
/// Every atom contributes one dense product of its `(m_k × m_k)` smoothness
/// penalty `S_k` with its `(m_k × p)` decoder coefficients `B_k`. These products
/// are independent across atoms, so the per-atom axis is the natural batch /
/// device-fan-out dimension. This helper:
///
///   * groups atoms by identical `(m_k, p)` shape (the strided-batched cuBLAS
///     GEMM requires a uniform tile),
///   * for each group with ≥ 2 atoms whose aggregate flop count clears the
///     dispatch threshold, partitions the group's atoms across every available
///     device with [`crate::gpu::pool::scatter_batched`] and runs one
///     `try_fast_abt_strided_batched` per device tile (computing
///     `S_k · B_k = S_k · (B_kᵀ)ᵀ`),
///   * uses the exact ndarray `S_k.dot(B_k)` when no GPU is admitted; once a
///     runtime is admitted, pool or batched-GEMM failures are propagated rather
///     than disguised as CPU eligibility.
///
/// Returns one `S_k · B_k` matrix per atom, in atom order. `symmetrize`
/// pre-symmetrises each `S_k` (the assembly path needs `½(S+Sᵀ)`); the value /
/// quadratic-form callers pass `false` since the quadratic form only sees the
/// symmetric part regardless.
pub(crate) fn batched_smooth_sb(
    sb_inputs: &[(ArrayView2<'_, f64>, ArrayView2<'_, f64>)],
    symmetrize: bool,
    gpu_policy: gam_gpu::GpuPolicy,
) -> Result<Vec<Array2<f64>>, String> {
    let n_atoms = sb_inputs.len();
    // Materialise the (optionally symmetrised) S factors once; the GPU tile and
    // the CPU fallback both read these, so a single pass keeps the two routes
    // numerically identical.
    let s_mats: Vec<Array2<f64>> = sb_inputs
        .iter()
        .map(|(s, _)| {
            if symmetrize {
                let m = s.nrows();
                let mut sym = Array2::<f64>::zeros((m, m));
                for i in 0..m {
                    for j in 0..m {
                        sym[[i, j]] = 0.5 * (s[[i, j]] + s[[j, i]]);
                    }
                }
                sym
            } else {
                s.to_owned()
            }
        })
        .collect();

    // Exact CPU product for a single atom, reused by the no-GPU route and groups
    // that are structurally ineligible for batching.
    let cpu_one = |idx: usize| -> Array2<f64> { s_mats[idx].dot(&sb_inputs[idx].1) };

    // Group atom indices by uniform (m, p) shape; only same-shape groups can ride
    // a strided-batched GEMM tile.
    let mut groups: std::collections::BTreeMap<(usize, usize), Vec<usize>> =
        std::collections::BTreeMap::new();
    for (idx, (_, b)) in sb_inputs.iter().enumerate() {
        let m = s_mats[idx].nrows();
        let p = b.ncols();
        groups.entry((m, p)).or_default().push(idx);
    }

    // The op a group's device tile actually issues: one strided-batched
    // `A·Bᵀ` with `A = S_k` (m×m) and `B = B_kᵀ` (p×m), i.e. `batch = |group|`,
    // `m = m_k`, `n = p`, `k = m_k`.
    let group_op =
        |atoms: usize, m: usize, p: usize| crate::gpu::linalg_dispatch::DispatchOp::BatchedGemm {
            batch: atoms,
            m,
            n: p,
            k: m,
        };

    // Size gate BEFORE the device probe (startup-tax ordering fix): when NO
    // group's tile op could be admitted by any reachable policy, every atom
    // would take `cpu_one` anyway, so return without resolving `GpuRuntime`
    // (whose first resolution creates a CUDA primary context on every GPU).
    if !groups
        .iter()
        .any(|(&(m, p), members)| group_op(members.len(), m, p).admissible_under_any_policy())
    {
        return Ok((0..n_atoms).map(cpu_one).collect());
    }

    let rt = match crate::gpu::device_runtime::GpuRuntime::resolve(gpu_policy)
        .map_err(|error| format!("decoder-smoothness CUDA admission failed: {error}"))?
    {
        Some(rt) => rt,
        None => return Ok((0..n_atoms).map(cpu_one).collect()),
    };

    let mut out: Vec<Option<Array2<f64>>> = (0..n_atoms).map(|_| None).collect();
    for ((m, p), members) in groups {
        // Singletons and tiny groups gain nothing from batched device launch;
        // the single-product `fast_*` shim (size-gated) already handles a large
        // lone GEMM, so route those straight through the CPU-or-shim helper.
        //
        // The second condition is the SAME admission the tile's
        // `try_fast_abt_strided_batched_with_policy` will run, on the SAME op,
        // asked once here. Previously the caller pre-screened on a group's
        // aggregate flops against `MIN_CALIBRATABLE_GEMM_FLOPS` — the most
        // permissive bound ANY policy can carry — and then treated the
        // calibrated policy's stricter (and correct) decline inside the scatter
        // as a fatal error. At the LLM decoder shape (m=6, p=2048, 8 atoms) the
        // whole group is 1.2 MFLOP: far below a real device's calibrated
        // crossover, so a GPU host FAILED fits that a CPU host completes. The
        // decline is a routing verdict, not a fault — the exact `cpu_one`
        // product is the right continuation, and a post-admission scatter
        // failure below is still fatal.
        if members.len() < 2
            || m == 0
            || p == 0
            || crate::gpu::linalg_dispatch::route_through_gpu_with_policy(
                group_op(members.len(), m, p),
                gpu_policy,
            )
            .is_none()
        {
            for &idx in &members {
                out[idx] = Some(cpu_one(idx));
            }
            continue;
        }
        // Build the per-tile batched inputs lazily inside the device closure so
        // each device only packs the atoms it owns. `items` carries the member
        // atom indices; `scatter_batched` slices it per device ordinal.
        let mut items: Vec<usize> = members.clone();
        let s_ref = &s_mats;
        // Collect per-tile results into a side channel keyed by atom index, then
        // splice them in after scatter completes (scatter's closure borrows
        // `items` immutably-per-tile and must stay `Sync`).
        let tile_results: std::sync::Mutex<Vec<(usize, Array2<f64>)>> =
            std::sync::Mutex::new(Vec::with_capacity(members.len()));
        let ok = crate::gpu::pool::scatter_batched(rt, &mut items, |_, slice| {
            if slice.is_empty() {
                return Some(());
            }
            let batch = slice.len();
            // A = stacked S_k  (batch, m, m); B = stacked B_kᵀ (batch, p, m) so
            // that `A · Bᵀ` per tile yields `S_k · B_k` (batch, m, p).
            let mut a = Array3::<f64>::zeros((batch, m, m));
            let mut bt = Array3::<f64>::zeros((batch, p, m));
            for (t, &idx) in slice.iter().enumerate() {
                let s = &s_ref[idx];
                let b = &sb_inputs[idx].1;
                for i in 0..m {
                    for j in 0..m {
                        a[[t, i, j]] = s[[i, j]];
                    }
                }
                for i in 0..p {
                    for j in 0..m {
                        bt[[t, i, j]] = b[[j, i]];
                    }
                }
            }
            let prod = crate::gpu::try_fast_abt_strided_batched_with_policy(
                a.view(),
                bt.view(),
                gpu_policy,
            )?;
            let mut sink = tile_results.lock().expect("tile_results mutex poisoned");
            for (t, &idx) in slice.iter().enumerate() {
                sink.push((idx, prod.slice(s![t, .., ..]).to_owned()));
            }
            Some(())
        });
        // The scatter closure has returned, so all borrows of `items`/`s_mats`/
        // `tile_results` are released; write the results back into `out`.
        match ok {
            Some(()) => {
                let sink = tile_results
                    .into_inner()
                    .expect("tile_results mutex poisoned");
                for (idx, mat) in sink {
                    out[idx] = Some(mat);
                }
                // A successful scatter must produce every member exactly once.
                for &idx in &members {
                    if out[idx].is_none() {
                        return Err(format!(
                            "decoder-smoothness device scatter omitted atom {idx}"
                        ));
                    }
                }
            }
            None => {
                return Err(format!(
                    "decoder-smoothness device scatter declined admitted group m={m}, p={p}, atoms={}",
                    members.len()
                ));
            }
        }
    }
    out.into_iter()
        .enumerate()
        .map(|(idx, slot)| {
            slot.ok_or_else(|| format!("decoder-smoothness result missing atom {idx}"))
        })
        .collect()
}

/// A detected bifurcation on the curvature-homotopy branch (#1007): the arrow
/// factor's smallest Cholesky pivot collapsed below the safe-SPD tolerance at a
/// homotopy parameter `η`, so the optimal branch the tracker was following lost
/// strict positive-definiteness. Recorded on [`CurvatureWalkReport`] and never
/// silent — the walk returns control to the documented multi-seed cascade.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct CurvatureBifurcation {
    /// Homotopy parameter at which the pivot collapsed.
    pub eta: f64,
    /// The smallest arrow-factor pivot observed at `eta` (Hessian-scale, i.e.
    /// squared lower-Cholesky diagonal); below the safe-SPD floor.
    pub min_pivot: f64,
}

/// Outcome of one certified curvature-homotopy entry walk (#1007).
///
/// The tracker walks the basis curvature dial `η` from the Eckart-Young anchor
/// (`η = 0`, global by construction) to the full curved basis (`η = 1`),
/// predictor-corrector style, holding the per-pivot positivity invariant. This
/// report makes the outcome observable on the fit payload: `arrived` says the
/// walk reached `η = 1` on the certified branch; `bifurcation` records the first
/// detected pivot collapse (if any); `collapse_events` mirrors the inner active
/// -mass guard's verdict at the arrival state; `eta_steps` / `step_halvings`
/// are the walk's cost. A walk that did not arrive (degenerate anchor or a
/// recorded bifurcation) hands control back to the multi-seed cascade.
#[derive(Debug, Clone)]
pub struct CurvatureWalkReport {
    /// Whether the walk reached `η = 1` on the certified optimal branch.
    pub arrived: bool,
    /// Eckart-Young (SVD low-rank) residual-ceiling energy at `η = 0`: the
    /// certified rank bound the base-topology relaxation is solved against (a
    /// lower bound on the residual at every η, not a linearity claim).
    pub anchor_residual_norm_sq: f64,
    /// First detected branch bifurcation (pivot collapse), or `None` when the
    /// pivot stayed strictly positive across the whole walk.
    pub bifurcation: Option<CurvatureBifurcation>,
    /// Number of accepted `η` waypoints (anchor → 1).
    pub eta_steps: usize,
    /// Number of `η`-step halvings forced by a shrinking min-pivot.
    pub step_halvings: usize,
    /// Number of inner active-mass collapse events recorded at the arrival
    /// state (the same `#976` guard ledger the cascade reads); a clean walk
    /// arrives with this empty.
    pub collapse_events: usize,
    /// Number of scaffold re-seeds the walk itself triggered. A certified walk
    /// from the global anchor reaches `η = 1` with zero reseeds.
    pub reseeds: usize,
}

#[derive(Debug, Clone)]
pub struct LinearSpanAtomAnchor {
    pub gate_weight: f64,
    pub frame: GrassmannFrame,
    pub decoder_coordinates: Array2<f64>,
    pub singular_values: Array1<f64>,
}

#[derive(Debug, Clone)]
pub struct LinearSpanAnchor {
    pub atoms: Vec<LinearSpanAtomAnchor>,
    pub reconstruction: Array2<f64>,
    pub residual_norm_sq: f64,
}

pub(crate) fn sae_cholesky_solve_neg_gradient(
    h: ArrayView2<'_, f64>,
    g: ArrayView1<'_, f64>,
) -> Result<Array1<f64>, String> {
    let n = h.nrows();
    if h.ncols() != n || g.len() != n {
        return Err(format!(
            "sae_cholesky_solve_neg_gradient: shape mismatch H={:?}, g={}",
            h.dim(),
            g.len()
        ));
    }
    let mut l = Array2::<f64>::zeros((n, n));
    for i in 0..n {
        for j in 0..=i {
            let mut sum = h[[i, j]];
            for k in 0..j {
                sum -= l[[i, k]] * l[[j, k]];
            }
            if i == j {
                if !(sum.is_finite() && sum > 0.0) {
                    return Err(format!("non-positive Cholesky pivot at {i}: {sum}"));
                }
                l[[i, j]] = sum.sqrt();
            } else {
                l[[i, j]] = sum / l[[j, j]];
            }
        }
    }
    let mut y = Array1::<f64>::zeros(n);
    for i in 0..n {
        let mut sum = -g[i];
        for k in 0..i {
            sum -= l[[i, k]] * y[k];
        }
        y[i] = sum / l[[i, i]];
    }
    let mut x = Array1::<f64>::zeros(n);
    for ii in 0..n {
        let i = n - 1 - ii;
        let mut sum = y[i];
        for k in i + 1..n {
            sum -= l[[k, i]] * x[k];
        }
        x[i] = sum / l[[i, i]];
    }
    if !x.iter().all(|v| v.is_finite()) {
        return Err("sae_cholesky_solve_neg_gradient: non-finite solution".into());
    }
    Ok(x)
}

pub(crate) fn solve_basis_transport(
    new_phi: ArrayView2<'_, f64>,
    old_phi: ArrayView2<'_, f64>,
) -> Result<Array2<f64>, String> {
    solve_design_least_squares(new_phi, old_phi)
}

pub(crate) fn transport_smooth_penalty_for_decoder(
    decoder_transport: ArrayView2<'_, f64>,
    old_smooth_penalty: ArrayView2<'_, f64>,
) -> Result<Array2<f64>, String> {
    let m = decoder_transport.nrows();
    if decoder_transport.ncols() != m {
        return Err(format!(
            "transport_smooth_penalty_for_decoder: decoder transport must be square; got {:?}",
            decoder_transport.dim()
        ));
    }
    if old_smooth_penalty.dim() != (m, m) {
        return Err(format!(
            "transport_smooth_penalty_for_decoder: smooth penalty shape {:?} != ({m}, {m})",
            old_smooth_penalty.dim()
        ));
    }
    let transport_inverse =
        solve_design_least_squares(decoder_transport, Array2::<f64>::eye(m).view())?;
    Ok(fast_atb(
        &transport_inverse,
        &fast_ab(&old_smooth_penalty.to_owned(), &transport_inverse),
    ))
}

pub(crate) fn solve_design_least_squares(
    design: ArrayView2<'_, f64>,
    rhs: ArrayView2<'_, f64>,
) -> Result<Array2<f64>, String> {
    if design.nrows() != rhs.nrows() {
        return Err(format!(
            "solve_design_least_squares: row mismatch design={} rhs={}",
            design.nrows(),
            rhs.nrows()
        ));
    }
    let (u_opt, sigma, vt_opt) = design
        .to_owned()
        .svd(true, true)
        .map_err(|err| format!("solve_design_least_squares: SVD failed: {err}"))?;
    let u = u_opt.ok_or_else(|| "solve_design_least_squares: SVD omitted U".to_string())?;
    let vt = vt_opt.ok_or_else(|| "solve_design_least_squares: SVD omitted Vt".to_string())?;
    let smax = sigma.iter().fold(0.0_f64, |acc, &v| acc.max(v));
    if !(smax.is_finite() && smax > 0.0) {
        return Err("solve_design_least_squares: design has zero numerical rank".to_string());
    }
    let cutoff = smax * f64::EPSILON * (design.nrows().max(design.ncols()) as f64);
    let coeffs = u.t().dot(&rhs);
    let mut scaled = Array2::<f64>::zeros(coeffs.dim());
    for row in 0..sigma.len() {
        if sigma[row] > cutoff {
            let inv = 1.0 / sigma[row];
            for col in 0..rhs.ncols() {
                scaled[[row, col]] = inv * coeffs[[row, col]];
            }
        }
    }
    Ok(vt.t().dot(&scaled))
}

#[cfg(test)]
mod linear_parity_anchor_1026_tests {
    //! #1026 — reconstruction-parity instrument + gate for the LINEAR-SAE
    //! Eckart-Young anchor.
    //!
    //! For a purely-LINEAR dictionary the reconstruction ceiling is the
    //! rank-(Σ_k basis_size_k) PCA / Eckart-Young projection of the target (the
    //! best linear subspace of that total rank). [`linear_span_anchor`] is the
    //! η=0 primitive that seeds the curvature walk with exactly that projection
    //! via sequential per-atom residual SVDs, so — independent of the downstream
    //! routing / inner Newton — its OWN reconstruction must attain the PCA
    //! ceiling at the dictionary's total rank. If it does, any end-to-end
    //! linear-SAE parity shortfall is a DOWNSTREAM (routing / canonicalization)
    //! effect, not an anchor defect; if it does not, the anchor itself loses
    //! reconstructible variance the linear dictionary is entitled to. This test
    //! pins the anchor at the ceiling so a regression that weakens the
    //! sequential-deflation parity (wrong per-atom rank, gate mishandling, a
    //! non-orthogonal deflation) is caught.
    //!
    //! ## #1026 routing-bound finding (why a GATED linear SAE under-reconstructs)
    //!
    //! The anchor reaches the rank-(K·d) PCA ceiling because its NEUTRAL gates
    //! ([`neutral_gate_weights`]: softmax `1/K`, ordered Beta--Bernoulli prior) keep every atom ON for
    //! every row, so all `K·d` decoder directions are available to reconstruct
    //! each row — exactly the unrestricted linear subspace PCA uses. A FITTED
    //! softmax/ordered Beta--Bernoulli SAE instead routes each row through learned gates, so its
    //! per-row reconstruction is `Σ_k a_k(row)·γ_k(t_k(row))` — a gate-WEIGHTED
    //! (softmax: simplex `Σ_k a_k ≈ 1`) combination whose per-row effective rank is
    //! bounded by that row's active-atom count. End-to-end linear-SAE parity with
    //! PCA is therefore REACHABLE iff each row's active rank ≥ the data's local
    //! rank — i.e. with dense-enough routing (high `top_k` / low sparsity `λ`); the
    //! residual gap under SPARSE routing is the price of sparsity, not a defect.
    //! The engine already retains the anchor-quality basin where reachable: the
    //! [`SaeManifoldOuterObjective::into_fitted`] seed-basin + pristine-seed
    //! fallbacks restore the anchor-seeded state whenever the inner solve degrades
    //! EV. The parity-vs-sparsity tradeoff is the genuine #1026 frontier; the
    //! UNGATED linear/background tier (a linear atom routed with `a_k ≡ 1`, added
    //! to the gated curved residual) is the architectural lever that lets the
    //! linear component carry full-rank variance while curved atoms stay sparse.

    use super::*;

    /// Explained variance of the least-squares projection of `target` (n×p) onto
    /// the column span of a design matrix `phi` (n×m). The design's first column is
    /// an intercept in every caller below, so the projection is mean-aware and the
    /// EV denominator (column-centered SST) is consistent. Solved via the normal
    /// equations with a tiny ridge for numerical PD safety.
    fn ls_projection_ev(phi: ArrayView2<'_, f64>, target: ArrayView2<'_, f64>) -> f64 {
        let m = phi.ncols();
        let gram = phi.t().dot(&phi) + Array2::<f64>::eye(m) * 1.0e-10;
        let rhs = phi.t().dot(&target);
        let coeffs = gam_linalg::faer_ndarray::FaerCholesky::cholesky(&gram, faer::Side::Lower)
            .map(|c| c.solve_mat(&rhs))
            .expect("design Gram must be SPD");
        let fitted = phi.dot(&coeffs);
        reconstruction_explained_variance(target, fitted.view()).expect("projection EV finite")
    }

    /// #1026 HYBRID curved+linear dictionary (ladder item 2) — the CPU-provable
    /// per-active-expressivity invariant: on data that is a LINEAR component in one
    /// latent coordinate PLUS a CURVED (periodic) component in another, a hybrid
    /// dictionary that pairs a LINEAR atom with a CURVED (periodic-harmonic) atom
    /// reconstructs STRICTLY MORE variance than EITHER a pure-linear dictionary OR
    /// a pure-curved dictionary of the same composition alone. This is the issue's
    /// "high-confidence hybrid-dominance" argument made concrete on synthetic data:
    /// a degree-1 line cannot bend to the periodic wave (so curved-alone misses the
    /// linear ramp's intercept/slope only partially via its own basis, and
    /// linear-alone misses the wave entirely), while the union of the two bases
    /// spans both. We fit each candidate by the EXACT least-squares projection onto
    /// its basis design (built from the SAME production evaluators the SAE uses:
    /// the linear `{1, z}` design and `PeriodicHarmonicEvaluator`'s `{1, sinθ,
    /// cosθ}`), so the comparison is pure CPU linear algebra with no corpus, no
    /// inner Newton, and no GPU. The real large-K hybrid EV-vs-K curve on the Qwen
    /// corpus stays corpus/GPU-gated; this pins the SIGN of the hybrid advantage
    /// (hybrid > max(linear, curved), strictly) which needs no corpus.
    #[test]
    fn hybrid_curved_plus_linear_beats_either_alone_1026() {
        let n = 80usize;
        let p = 5usize;
        // Two independent latent coordinates: a linear factor z and a periodic
        // angle θ ∈ [0, 1) (period 1). The signal is a linear ramp in z PLUS a
        // genuine circular wave in θ that no degree-1 line in θ can represent.
        let zf: Vec<f64> = (0..n).map(|i| ((i as f64 + 1.0) * 0.21).sin()).collect();
        let theta: Vec<f64> = (0..n).map(|i| ((i as f64) * 0.6180339887) % 1.0).collect();
        // Per-channel coefficients for the linear ramp and the sin/cos wave.
        let a0 = Array1::from_shape_fn(p, |c| 0.5 + 0.3 * (c as f64));
        let a1 = Array1::from_shape_fn(p, |c| (((c + 1) % 4) as f64 - 1.5) * 0.8);
        let bs = Array1::from_shape_fn(p, |c| (((c * 2 + 1) % 5) as f64 - 2.0) * 0.9);
        let bc = Array1::from_shape_fn(p, |c| (((c * 3 + 2) % 5) as f64 - 2.0) * 0.7);
        let two_pi = std::f64::consts::TAU;
        let target = Array2::from_shape_fn((n, p), |(i, c)| {
            a0[c]
                + a1[c] * zf[i]
                + bs[c] * (two_pi * theta[i]).sin()
                + bc[c] * (two_pi * theta[i]).cos()
        });

        // LINEAR-only design: {1, z} (the pure-linear dictionary's reach).
        let mut phi_lin = Array2::<f64>::ones((n, 2));
        for i in 0..n {
            phi_lin[[i, 1]] = zf[i];
        }
        // CURVED-only design: the production periodic-harmonic basis {1, sinθ, cosθ}.
        let eval = PeriodicHarmonicEvaluator::new(3).unwrap();
        let theta_coords = Array2::from_shape_fn((n, 1), |(i, _)| theta[i]);
        let (phi_curved, _jet) = eval.evaluate(theta_coords.view()).unwrap();
        // HYBRID design: linear {z} tier concatenated with the curved {sinθ, cosθ}
        // atom (single shared intercept) — the union basis the hybrid SAE realizes
        // (a linear background atom + a curved atom in one fit).
        let mut phi_hybrid = Array2::<f64>::ones((n, 4));
        for i in 0..n {
            phi_hybrid[[i, 1]] = zf[i];
            phi_hybrid[[i, 2]] = phi_curved[[i, 1]]; // sinθ
            phi_hybrid[[i, 3]] = phi_curved[[i, 2]]; // cosθ
        }

        let ev_lin = ls_projection_ev(phi_lin.view(), target.view());
        let ev_curved = ls_projection_ev(phi_curved.view(), target.view());
        let ev_hybrid = ls_projection_ev(phi_hybrid.view(), target.view());
        println!(
            "[#1026] hybrid dominance: linear-only EV={ev_lin:.6}  curved-only EV={ev_curved:.6}  \
             hybrid EV={ev_hybrid:.6}  hybrid−max(either)={:.6}",
            ev_hybrid - ev_lin.max(ev_curved)
        );

        assert!(
            ev_lin.is_finite() && ev_curved.is_finite() && ev_hybrid.is_finite(),
            "all three projection EVs must be finite: lin={ev_lin}, curved={ev_curved}, \
             hybrid={ev_hybrid}"
        );
        // The hybrid spans BOTH components, so it captures (essentially) all the
        // variance — strictly more than either single-geometry dictionary, each of
        // which is blind to the other component. A regression that broke the
        // periodic basis (curved collapses to the linear reach) or the linear tier
        // would shrink this gap.
        assert!(
            ev_hybrid > ev_lin + 0.05,
            "#1026 hybrid: union basis EV {ev_hybrid:.6} must STRICTLY beat linear-only \
             {ev_lin:.6} (the curved atom captures the periodic wave a line cannot)"
        );
        assert!(
            ev_hybrid > ev_curved + 0.05,
            "#1026 hybrid: union basis EV {ev_hybrid:.6} must STRICTLY beat curved-only \
             {ev_curved:.6} (the linear tier captures the z-ramp the periodic atom cannot)"
        );
        // And the hybrid essentially saturates: the union basis is the exact
        // generating model, so its projection EV is ~1 (within LS/ridge rounding).
        assert!(
            ev_hybrid > 0.999,
            "#1026 hybrid: the union basis is the exact generating model, so its \
             projection EV must be ~1; got {ev_hybrid:.6}"
        );
    }
}

#[cfg(test)]
mod decoder_smoothness_dispatch_2393_tests {
    //! #2393 — the decoder-smoothness batched GEMM must ROUTE, never refuse.
    //!
    //! `batched_smooth_sb` pre-screened a group on its aggregate flops against
    //! `MIN_CALIBRATABLE_GEMM_FLOPS` (the most permissive floor ANY policy can
    //! carry) and then treated the CALIBRATED policy's stricter decline inside
    //! the device scatter as a fatal error. At the SAE LLM decoder shape
    //! (`m=6`, `p=2048`, 8 atoms) the whole group is 1.2 MFLOP — far below a
    //! real device's measured crossover — so on a CUDA host every SAE fit that
    //! reached this call died with "device scatter declined admitted group",
    //! while the identical fit completed on a CPU-only host. The decline is a
    //! correct routing verdict; the exact host product is the continuation.

    use super::batched_smooth_sb;
    use ndarray::Array2;

    /// The LLM decoder group must produce the exact per-atom products under
    /// every policy the process can be in, on a device host and a CPU host
    /// alike. `Auto` is the production policy; `Off` pins the host arm.
    #[test]
    fn llm_decoder_group_routes_instead_of_refusing() {
        const M: usize = 6;
        const P: usize = 2048;
        const ATOMS: usize = 8;

        let s_mats: Vec<Array2<f64>> = (0..ATOMS)
            .map(|atom| {
                Array2::from_shape_fn((M, M), |(i, j)| {
                    if i == j {
                        1.0 + 0.1 * (atom as f64)
                    } else {
                        0.01 * ((i + j + atom) as f64).sin()
                    }
                })
            })
            .collect();
        let b_mats: Vec<Array2<f64>> = (0..ATOMS)
            .map(|atom| {
                Array2::from_shape_fn((M, P), |(i, j)| {
                    0.05 * (((i + 1) * (j + 3) + atom) as f64 * 0.0037).cos()
                })
            })
            .collect();
        let expected: Vec<Array2<f64>> = (0..ATOMS)
            .map(|atom| s_mats[atom].dot(&b_mats[atom]))
            .collect();

        for policy in [crate::gpu::GpuPolicy::Auto, crate::gpu::GpuPolicy::Off] {
            let inputs: Vec<_> = (0..ATOMS)
                .map(|atom| (s_mats[atom].view(), b_mats[atom].view()))
                .collect();
            let got = batched_smooth_sb(&inputs, false, policy).unwrap_or_else(|error| {
                panic!(
                    "#2393: the LLM decoder group must route to a product under \
                     {policy}, not refuse; got error: {error}"
                )
            });
            assert_eq!(got.len(), ATOMS);
            for (atom, product) in got.iter().enumerate() {
                assert_eq!(
                    product, &expected[atom],
                    "#2393: atom {atom} product differs from the exact S·B under {policy}"
                );
            }
        }
    }
}

/// #2593 — the coverage the two independent substring ladders did not have.
#[cfg(test)]
mod probe_refusal_classification_2593_tests {
    use super::{ArrowSchurError, OuterProbeTelemetry, ProbeRefusalKind, SaeManifoldOuterObjective};

    /// One representative rendered message per kind, taken from the producer
    /// that emits it.
    ///
    /// #2598 — the Schur arm is RENDERED FROM THE PRODUCER rather than
    /// transcribed from it. A hard-coded copy of another crate's `Display`
    /// output makes this gate self-referential: gam-solve rewords, production
    /// stops classifying, and the test keeps passing on a string production no
    /// longer emits. Building the message from a real `ArrowSchurError` makes
    /// the input the production text by construction.
    fn representative(kind: ProbeRefusalKind) -> String {
        match kind {
            ProbeRefusalKind::InnerNotConverged => {
                "SaeManifoldTerm::penalized_quasi_laplace_criterion: inner solve did not \
                 converge at fixed ρ; refusing to rank an off-optimum state"
                    .to_string()
            }
            // #2598 — this arm used to be a FICTION. It spelled out "undamped
            // criterion factorization hit a non-PD per-row H_tt block before KKT
            // stationarity" to match the classifier's needle, under the doc
            // comment above claiming it came from the producer. No producer
            // emits that sentence, so this gate certified a classifier that
            // answered `None` for every per-row refusal the crate can raise.
            // Below is one of the three real renderings, all of which are pinned
            // by `every_per_row_producer_rendering_classifies_as_non_pd_per_row`.
            ProbeRefusalKind::NonPdPerRow => {
                "SaeManifoldTerm::penalized_quasi_laplace_criterion: undamped evidence \
                 factorization hit a non-PD per-row H_tt block before KKT stationarity \
                 at an infeasible-ρ probe"
                    .to_string()
            }
            ProbeRefusalKind::NonPdSchur => ArrowSchurError::SchurFactorFailed {
                reason: "leading minor is not positive definite".to_string(),
            }
            .to_string(),
            ProbeRefusalKind::AllZeroGatedDesign => {
                "run_joint_fit_arrow_schur: atom 2 is gated off at every row (all-zero \
                 gated design)"
                    .to_string()
            }
            ProbeRefusalKind::TotalCoCollapse => {
                "run_joint_fit_arrow_schur: reseed budget spent and the fit did not \
                 escape total co-collapse"
                    .to_string()
            }
        }
    }

    /// Every kind classifies to itself, is recoverable, and lands in exactly one
    /// counter that `infeasible_total` sums.
    ///
    /// This is the gate the crate lacked. `AllZeroGatedDesign` and
    /// `TotalCoCollapse` were recoverable with NO counter at all, so
    /// `infeasible_total()` under-reported precisely on the co-collapsing ρ
    /// that #2080's bounded probe-budget test exists to observe. Iterating
    /// `ALL` means a kind added without a counter cannot pass.
    #[test]
    fn every_refusal_kind_is_counted_exactly_once() {
        for kind in ProbeRefusalKind::ALL {
            let message = representative(kind);
            assert_eq!(
                ProbeRefusalKind::classify(&message),
                Some(kind),
                "representative message must classify as its own kind: {message}"
            );
            assert!(
                SaeManifoldOuterObjective::is_recoverable_value_probe_refusal(&message),
                "a classified refusal is ρ-local by construction: {message}"
            );
            let mut telemetry = OuterProbeTelemetry::default();
            telemetry.record_refusal_kind(&message);
            assert_eq!(
                telemetry.infeasible_total(),
                1,
                "{kind:?} must increment exactly one counter that infeasible_total sums"
            );
        }
    }

    /// A genuine defect stays fatal and uncounted. `None` is the fail-loud
    /// default: an unrecognised message must never be masked as +∞.
    #[test]
    fn an_unclassified_defect_is_fatal_and_uncounted() {
        let defect = "SaeManifoldTerm::penalized_quasi_laplace_criterion: \
                      arrow_log_det_from_cache returned None (undamped joint Hessian \
                      log-det unavailable for the Laplace normaliser)";
        assert_eq!(ProbeRefusalKind::classify(defect), None);
        assert!(!SaeManifoldOuterObjective::is_recoverable_value_probe_refusal(
            defect
        ));
        let mut telemetry = OuterProbeTelemetry::default();
        telemetry.record_refusal_kind(defect);
        assert_eq!(telemetry.infeasible_total(), 0);
    }

    /// #2598 — all three per-row producers, which the old needle missed.
    ///
    /// These are the messages `construction_quasi_laplace.rs` renders at its
    /// three per-row refusal sites, with the run-time numbers elided. Against the
    /// pre-#2598 needle ("undamped criterion factorization hit a non-PD per-row
    /// H_tt block before KKT") every one classified as `None`: fatal, uncounted,
    /// and aborting the fit at a ρ all three call-site comments say must be read
    /// as +∞. Two say "evidence" where the needle said "criterion"; the third
    /// says "criterion factorization HAS" where the needle said "hit".
    ///
    /// The producers interpolate
    /// [`ProbeRefusalKind::non_pd_per_row_marker`], so the phrase has one home
    /// and cannot drift again. This test is the three-way coverage that
    /// drift-freedom does not by itself give: that each surrounding sentence
    /// really does carry the marker.
    #[test]
    fn every_per_row_producer_rendering_classifies_as_non_pd_per_row() {
        let renderings = [
            "SaeManifoldTerm::penalized_quasi_laplace_criterion: stationary undamped \
             criterion factorization has a non-PD per-row H_tt block that spectral \
             unit-stiffness deflation could not condition",
            "SaeManifoldTerm::penalized_quasi_laplace_criterion: undamped evidence \
             factorization hit a non-PD per-row H_tt block before KKT stationarity \
             at an infeasible-ρ probe; returning the typed infeasible refusal \
             without grinding the probe refinement budget",
            "SaeManifoldTerm::penalized_quasi_laplace_criterion: undamped evidence \
             factorization hit a non-PD per-row H_tt block before KKT stationarity \
             and the refinement budget was exhausted",
        ];
        for rendering in renderings {
            assert_eq!(
                ProbeRefusalKind::classify(rendering),
                Some(ProbeRefusalKind::NonPdPerRow),
                "a per-row refusal the crate actually renders must classify as \
                 NonPdPerRow: {rendering}"
            );
            assert!(
                SaeManifoldOuterObjective::is_recoverable_value_probe_refusal(rendering),
                "the outer optimizer must read this ρ as +∞ and steer, not abort: \
                 {rendering}"
            );
            let mut telemetry = OuterProbeTelemetry::default();
            telemetry.record_refusal_kind(rendering);
            assert_eq!(
                telemetry.infeasible_non_pd_per_row, 1,
                "infeasible_non_pd_per_row counted zero of these before #2598: {rendering}"
            );
        }
    }

    /// The Schur conjunct is load-bearing and the two ladders disagreed on it:
    /// the telemetry one matched a bare "Schur complement Cholesky failed".
    /// A `SchurFactorFailed` whose reason is NOT a non-PD pivot is a real
    /// defect and must stay fatal.
    #[test]
    fn a_schur_failure_that_is_not_a_non_pd_pivot_stays_fatal() {
        let defect = "arrow-Schur: Schur complement Cholesky failed: non-finite entry";
        assert_eq!(ProbeRefusalKind::classify(defect), None);
        assert!(!SaeManifoldOuterObjective::is_recoverable_value_probe_refusal(
            defect
        ));
    }
}
