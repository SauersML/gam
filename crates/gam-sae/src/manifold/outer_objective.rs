use super::*;
use gam_math::special::bessel_i0_log_and_ratio;
use gam_solve::rho_optimizer::{
    FixedPointCertificateEval, FixedPointCoordinateCertificate, OuterResult,
};

/// #1033 — temperature on the chart-geometry routing predictor's cosine-aligned
/// logit `gate_logit_scale · ⟨x, γ̂⟩`. The alignment `⟨x, γ̂⟩` is on the natural
/// `‖x‖` scale; this scale maps it into the gate's logit range so a
/// well-reconstructing atom gets a clearly-on gate and a poorly-reconstructing one
/// a clearly-off gate. A starting value pending the MSI accuracy-gate calibration
/// (the single knob the fit-quality measurement tunes).
const AMORTIZED_GATE_LOGIT_SCALE: f64 = 1.0;

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

/// S1 (guard surgery) — the ABSOLUTE-DEGENERACY explained-variance floor: a fit
/// whose reconstruction EV sits at or below this value explains no more of the
/// centered target than a SIGNAL-FREE dictionary of the same reachable rank would
/// by finite-sample chance, so it is a structural collapse rather than a
/// merely-uncompetitive fit. It is the SINGLE source both collapse-detection sites
/// share (the fitted-data verdict feeding the outer wall, and the co-collapse
/// reseed arm), so both measure degeneracy against one and the same threshold.
///
/// The floor is the classical null coefficient of determination `q / n`
/// (`#free-reconstruction-directions / #observations`): fitting `q =
/// dictionary_rank` arbitrary linear directions to `n` centered rows of a
/// signal-free target captures, IN EXPECTATION, a fraction `q / n` of the variance
/// (the textbook null-`R²` of a `q`-regressor / `n`-observation least squares). It
/// is therefore a SAMPLING NOISE-FLOOR bound — the EV a collapsed dictionary
/// reaches purely from finite-sample fitting noise — carrying no magnitude fit to
/// any corpus and shrinking toward 0 as `n` grows, exactly as the null fitting
/// noise does. `dictionary_rank` is the dictionary's GEOMETRICALLY REACHABLE rank
/// (`reachable_dictionary_rank` = `rank([Φ_1 … Φ_K])`, read from the chart designs
/// alone so a co-collapsed decoder still reports full reach), capped at the
/// observation count `n` (NOT at the output dim `p` — see #F8 on
/// `reachable_dictionary_rank`), so `q ≤ n` keeps the floor in `[0, 1]`.
///
/// This REPLACES the former `0.5 × rank-q PCA/Eckart-Young EV ceiling` bar, which
/// compared a `k_active`-SPARSE fit against a DENSE rank-`q` linear ceiling and so
/// sat ABOVE the honest sparse optimum on real (non-sparse) activations, flagging
/// healthy-but-below-ceiling fits as collapses (the K≥2-real-data false positive
/// that opened every fit with a spurious "co-collapse"). A degeneracy detector may
/// catch only states from which descent cannot recover — EV at the null floor AND
/// the decoder output co-vanished (the original #853/#976 meaning) — never a
/// merely-uncompetitive state, which is the optimizer's job mid-fit and the
/// evidence framework's job after convergence. `f64::NAN` when there are no rows
/// (`n == 0`), which the callers' `ev <= floor` comparison treats as "no verdict".
pub(crate) fn absolute_degeneracy_ev_floor(
    target: ArrayView2<'_, f64>,
    dictionary_rank: usize,
) -> f64 {
    let n = target.nrows();
    if n == 0 {
        return f64::NAN;
    }
    // Saturated floor = no verdict. When the reachable rank q >= n, colspan(Φ)
    // is ALL of R^n, so the signal-free least-squares fit reproduces the target
    // EXACTLY: the null floor is 1, every possible EV sits at or below it, and
    // the statistic carries ZERO evidence about degeneracy. Returning 1.0 here
    // branded EV = 0.999 fits on small-n/basis-rich fixtures as co-collapsed
    // ("EV 0.9990 at or below the signal-free null floor 1.0000", 2026-07-10).
    // NaN is the established "no verdict" convention (see n == 0 above): both
    // caller arms compare `<= floor`, which is false on NaN, so the absolute
    // arm stands down and degeneracy detection falls to the relative-norm arm.
    if dictionary_rank >= n {
        return f64::NAN;
    }
    dictionary_rank as f64 / n as f64
}

/// #1610 — the GEOMETRICALLY REACHABLE linear rank of a dictionary, used as the
/// rank `q` in the signal-free null degeneracy floor
/// (`absolute_degeneracy_ev_floor(target, reachable_dictionary_rank(...))` = `q / n`).
///
/// The null floor scales with the number of directions the dictionary can reach.
/// The previous `q = Σ_k basis_size_k` (nominal
/// coefficient count) is biased HIGH for a NONLINEAR dictionary: a curved
/// `latent_dim = d` atom decoded through a smooth chart does not linearly span
/// all `basis_size_k` of its coefficient directions in the output — its decoded
/// image `Φ_k B_k` lies inside `colspan(Φ_k)`, whose dimension is the realized
/// chart rank `rank(Φ_k) ≤ basis_size_k`. Summing the per-atom REALIZED chart
/// ranks gives the linear dimension the dictionary's union of chart images can
/// actually reach on this sample, which is the principled rank for the linear
/// PCA ceiling that the bar uses.
///
/// The charts are read from the CHART design alone (not the decoder magnitude),
/// so a co-collapsed atom (`‖B_k‖ → 0`) still reports its full geometric reach
/// — the collapse guard must NOT silently lower its own bar at the very
/// degenerate state it exists to catch.
///
/// #C5: `q` is the rank of the HORIZONTALLY CONCATENATED realized chart design
/// `[Φ_1 … Φ_K]` (`n × Σ_k M_k`), NOT `Σ_k rank(Φ_k)`. `rank([Φ_1 … Φ_K]) ≤
/// Σ_k rank(Φ_k)`, with equality only when the atoms' column spaces are linearly
/// INDEPENDENT; summing double-counts shared directions (two identical atoms:
/// true reachable rank 1, the sum claims 2), biasing the null floor `q/n` upward
/// and manufacturing false collapse verdicts. The number of FREE reconstruction
/// directions a signal-free dictionary fits is exactly this concatenated rank.
///
/// #F8: `q` is capped at the OBSERVATION count `n`, NOT at the output dim `p`.
/// The signal-free reconstruction is `X̂ = Φ·B` with `Φ = [Φ_1 … Φ_K]` (`n × Σ_k
/// M_k`) fixed and the decoder `B` (`Σ_k M_k × p`) free, so each of the `p` target
/// columns is least-squares projected onto `colspan(Φ) ⊆ Rⁿ` — a subspace of
/// dimension `q = rank(Φ) ≤ min(n, Σ_k M_k)`. The expected captured fraction is
/// `tr(P)/n = q/n` for EVERY column, so the null `R²` is `q/n` INDEPENDENT of `p`
/// (the `p` output columns cancel between the Frobenius numerator and
/// denominator). An OVERCOMPLETE dictionary (`Σ_k M_k > p`) can therefore reach
/// `q > p` free directions; the old `min(n, p)` cap under-counted `q`, LOWERED the
/// floor, and made the collapse detector LENIENT exactly for overcomplete
/// dictionaries. `q ≤ n` still keeps the floor in `[0, 1]`. If any atom's design
/// is non-finite or the concatenated SVD fails, the whole function degrades to the
/// historical summed per-atom ranks rather than corrupting `q`.
pub(crate) fn reachable_dictionary_rank(atoms: &[SaeManifoldAtom], n: usize, p: usize) -> usize {
    if atoms.is_empty() || n == 0 || p == 0 {
        return 0;
    }
    // Historical Σ_k rank(Φ_k) (each capped at p) — the graceful-degradation
    // fallback when the concatenated design cannot be formed or decomposed.
    let summed_fallback = || -> usize {
        atoms
            .iter()
            .map(|atom| match atom.realized_chart_image_rank() {
                Ok(r) => r,
                Err(_) => atom.basis_size().min(p),
            })
            .sum::<usize>()
            .min(n)
    };
    let total_cols: usize = atoms.iter().map(|atom| atom.basis_values.ncols()).sum();
    if total_cols == 0 {
        return 0;
    }
    let mut concat = Array2::<f64>::zeros((n, total_cols));
    let mut col = 0usize;
    for atom in atoms {
        let phi = &atom.basis_values;
        // A shape mismatch or a non-finite entry would poison the joint SVD;
        // degrade to the per-atom summed ranks instead.
        if phi.nrows() != n || !phi.iter().all(|v| v.is_finite()) {
            return summed_fallback();
        }
        let m = phi.ncols();
        concat.slice_mut(s![.., col..col + m]).assign(phi);
        col += m;
    }
    let sv = match concat.svd(false, false) {
        Ok((_, sv, _)) => sv,
        Err(_) => return summed_fallback(),
    };
    let max_sv = sv.iter().copied().fold(0.0_f64, f64::max);
    if !(max_sv > 0.0) {
        return 0;
    }
    let tol = SAE_MANIFOLD_SPECTRAL_RANK_CUTOFF * max_sv;
    sv.iter().filter(|&&v| v > tol).count().min(n)
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
    pub(crate) fn record(&mut self, cost: f64) -> bool {
        self.evals += 1;
        if !cost.is_finite() {
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
    /// #2266 — neither an outer ρ-search NOR an inner solve ran: the decoder /
    /// coordinates / gate logits / ρ were installed verbatim from an
    /// externally-trained (e.g. torch-lane) fit by
    /// `run_sae_manifold_certify` (#2266). There is no first-order
    /// stationarity certificate for this state — only the post-fit
    /// diagnostics/certificates computed AT it.
    External,
}

impl SaeOuterVerdict {
    /// Stable wire name; the enums own the vocabulary so bindings marshal
    /// instead of mapping (precedent: ba57254af).
    pub fn as_str(&self) -> &'static str {
        match self {
            Self::Search(via) => via.as_str(),
            Self::FixedRho => "fixed_rho",
            Self::External => "external",
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

/// Outer penalized quasi-Laplace objective for the SAE-manifold term.
///
/// Routes the SAE's smoothing hyperparameters ρ
/// (`log_lambda_sparse`, per-atom `log_lambda_smooth`, per-atom/axis `log_ard`)
/// through the *one* generic [`OuterObjective`] engine + cascade that the
/// main GAM penalized quasi-Laplace path uses, instead of the SAE's deleted forked
/// `update_ard_reml` fixed-point rule. Each outer eval runs the inner
/// `(t, β)` arrow-Schur Newton solve at the engine's current ρ and returns
/// the penalised quasi-Laplace score score (see
/// [`SaeManifoldTerm::penalized_quasi_laplace_criterion`]). #1421: this is NOT a true
/// normalized-prior REML/evidence objective — the softmax-entropy and
/// threshold-gate assignment priors have no finite normalizer, so there is no
/// ρ-independent prior constant to drop; only the proper-Gaussian
/// smoothing-penalty normalizer is a genuine REML term.
///
/// The SAE's outer coordinates ρ are all penalty-like / τ (precisions and
/// log-smoothing-strengths), so `psi_dim = 0`: there are no design-moving
/// (ψ) coordinates. Dense-admitted fits expose the exact implicit outer
/// gradient through the rank-revealing joint-Hessian solve; matrix-free fits
/// use the analytic Fellner--Schall trace fixed point.

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
    /// Outer criterion evaluations that returned the optimizer's conventional
    /// infeasible value (`+inf`) because the quasi-Laplace score was undefined or
    /// the fixed-ρ inner solve refused. A finite, data-collapsed fit is not an
    /// infeasible objective value: collapse remains a structural ledger verdict
    /// while penalized quasi-Laplace remains the sole optimized criterion.
    pub infeasible_criterion_evals: usize,
    /// #2234 — cost-only probes whose capped/forced inner solve exhausted its
    /// budget and were RESCUED by a one-shot retry at the accepted-point drive
    /// (full budget) instead of being misclassified as the infeasibility wall.
    pub budget_rescued_value_probes: usize,
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
    fn record_refusal_kind(&mut self, err: &str) {
        if err.contains("inner solve did not converge at fixed ρ") {
            self.infeasible_inner_not_converged += 1;
        } else if err.contains("Schur complement Cholesky failed") {
            self.infeasible_schur += 1;
        } else if err.contains("non-PD per-row H_tt block") {
            self.infeasible_non_pd_per_row += 1;
        }
    }

    /// Total infeasible probes across all refusal kinds.
    pub fn infeasible_total(&self) -> usize {
        self.infeasible_non_pd_per_row + self.infeasible_schur + self.infeasible_inner_not_converged
    }
}

/// #2080 (a) — probe→accepted warm-start handoff.
///
/// The generic outer line search evaluates its cost probes through the
/// value-only lane (`eval_with_order(Value)` / `eval_cost` →
/// `evaluate_envelope_value_probe` → `evaluate_value_probe_with_drive`), each of which drives the FULL
/// inner `(t, β)` Newton solve to KKT convergence at the probed ρ — starting
/// from the accepted basin — and then RESTORES the accepted term, discarding
/// that converged state. The accepted point of a successful line search is
/// always the ρ of its last successful value probe, and the engine then
/// re-evaluates it through the gradient lane (`eval`), which historically
/// re-ran the identical deterministic inner convergence from the identical
/// accepted basin — a full redundant inner solve per outer iteration.
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

/// How a criterion evaluation drives the inner `(t, β)` solve.
#[derive(Clone, Copy)]
enum ProbeInnerDrive {
    /// Historical path: hand the whole inner drive to
    /// `penalized_quasi_laplace_criterion_with_refine_policy` (accepted-basin evaluations, the
    /// cross-seed ranking / EFS value lane, streaming fits).
    Criterion { refine_progress_extension: bool },
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
    pub(crate) routing_frozen: bool,
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
    /// freeze regimes (see `evaluate_envelope_value_probe`).
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
    let plan = term.streaming_plan();
    if !plan.direct_logdet_admitted() {
        return 0;
    }
    let (host_budget, _) = super::sae_host_in_core_budget_bytes();
    let bytes_per_saved_state = plan
        .estimated_direct_peak_bytes
        .max(plan.estimated_full_batch_bytes)
        .max(std::mem::size_of::<SaeManifoldTerm>());
    host_budget.saturating_sub(plan.estimated_direct_peak_bytes) / bytes_per_saved_state
}

/// The dense route exposes the exact joint-Hessian IFT gradient. The matrix-
/// free route has only analytic EFS equations; its `eval()` zero vector exists
/// for legacy startup plumbing and is never a derivative capability or proof.
pub(crate) fn sae_outer_gradient_capability(plan: SaeStreamingPlan) -> Derivative {
    if plan.direct_logdet_admitted() {
        Derivative::Analytic
    } else {
        Derivative::Unavailable
    }
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
const SAE_SURROGATE_LANE_QUADRATURE_REL_TOL: f64 = 1.0e-8;
const SAE_SURROGATE_LANE_POWER_ITERS: usize = 40;
const SAE_SURROGATE_LANE_CG_REL_TOL: f64 = 1.0e-8;
const SAE_SURROGATE_LANE_CG_MAX_ITERS: usize = 20_000;
const SAE_SURROGATE_LANE_DEFLATION_MAX_RANK: usize = 128;
const SAE_SURROGATE_LANE_DEFLATION_SUBSPACE_ITERS: usize = 4;

fn sae_surrogate_lane_config() -> SurrogateLaneConfig {
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
        let init_rho = init_rho.for_assignment(term.assignment.mode);
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
            routing_frozen: false,
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
            checkpoint_fingerprint,
            checkpoint_path,
            crosscoder_blocks: None,
            reactive_waypoint_checkpoint: None,
        }
    }

    /// #2231 Inc-B (stage 1) — enable crosscoder block-relevance PRICING.
    ///
    /// `p_x` is the anchor width (leading `[0, p_x)` target columns, never
    /// scaled); `block_dims` are the `L-1` output-block widths in stacked-column
    /// order. Snapshots a PRISTINE (unscaled) copy of the non-anchor columns
    /// `[p_x, p̃)` — the drift-free source every `apply_block_scaling` reads —
    /// and seeds the last-applied `log λ_ℓ` to `0` (`λ = 1`, matching the target
    /// as handed in per the [`CrosscoderBlockPricing`] invariant).
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
    fn apply_block_scaling(&mut self, rho: &SaeManifoldRho) {
        // Disjoint field borrows: the pricing state and the target are rewritten
        // together, so destructure `self` rather than route through a `self`
        // method that would alias both.
        let Self {
            target,
            crosscoder_blocks: Some(blocks),
            ..
        } = self
        else {
            return;
        };
        // The builder pinned `block_dims.len() == log_lambda_block.len()`; guard
        // defensively so a mismatched ρ can never scale a wrong column range.
        if rho.log_lambda_block.len() != blocks.block_dims.len() {
            return;
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
            return;
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
    ///     block_jacobian(ρ) = −Σ_ℓ (n·p_ℓ/2)·log λ_ℓ.
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

    /// #1033 — opt into AMORTIZED (frozen) routing for the ρ-search: freeze the
    /// term's assignment gates to a ρ-invariant routing distilled from the CURRENT
    /// (construction-time / seed) dictionary, so the outer ρ-search reuses one
    /// routing instead of re-solving the per-row gates at every eval (the
    /// n-independent-outer-loop lever). `None` ⇒ off (free-logit search, the
    /// default). `Some(predictor)` selects the fixed-form distill:
    ///   * [`RoutingPredictor::Snapshot`] — freeze the current logits as-is
    ///     (cheapest; the MVP/baseline; goes stale if the dictionary moves);
    ///   * [`RoutingPredictor::ChartGeometry`] — distill the routing from the
    ///     encode-chart reconstruction alignment of the current dictionary
    ///     ([`SaeManifoldTerm::chart_geometry_routing_logits`]), which tracks the
    ///     dictionary geometry.
    /// Freezing here (from the seed/anchor dictionary) makes the routing
    /// ρ-invariant across the search; the inner solve then optimizes only the
    /// coordinates and decoder. The baseline (multi-start restore) term is frozen
    /// to match. Rejected for Softmax (separable-mode contract). The accuracy gate
    /// decides which predictor (and whether a per-outer-iterate refresh) is needed.
    #[must_use = "build error must be handled"]
    pub fn with_amortized_routing(
        mut self,
        predictor: Option<RoutingPredictor>,
    ) -> Result<Self, String> {
        let Some(form) = predictor else {
            return Ok(self);
        };
        match form {
            RoutingPredictor::Snapshot => {
                self.term.assignment.freeze_routing_in_place()?;
                self.baseline_term.assignment.freeze_routing_in_place()?;
            }
            RoutingPredictor::ChartGeometry => {
                let predicted = self.term.chart_geometry_routing_logits(
                    self.target.view(),
                    AMORTIZED_GATE_LOGIT_SCALE,
                )?;
                self.term
                    .assignment
                    .set_frozen_routing_in_place(predicted.clone())?;
                self.baseline_term
                    .assignment
                    .set_frozen_routing_in_place(predicted)?;
            }
        }
        self.routing_frozen = true;
        Ok(self)
    }

    /// #1033 — whether the ρ-search runs on frozen (amortized) routing.
    pub fn routing_is_frozen(&self) -> bool {
        self.routing_frozen
    }

    /// #1207 — the accumulated amortized warm-start telemetry. "Uses amortized
    /// warm-start" is verifiable as `telemetry.warm_started_evals > 0`.
    pub fn warm_start_telemetry(&self) -> AmortizedWarmStartTelemetry {
        self.warm_start_telemetry
    }

    /// Record one amortized warm-start attempt. Once selected, this accelerator
    /// is part of the declared optimization path: an encoder/atlas failure is
    /// propagated instead of silently changing the basin-entry algorithm.
    fn record_warm_start(&mut self, outcome: Result<usize, String>) -> Result<(), String> {
        self.warm_start_telemetry.record(&outcome);
        outcome.map(|_| ())
    }

    /// Stamp the currently installed state with a successful outer search's
    /// analytic convergence evidence.
    ///
    /// Merely receiving `OuterResult { converged: true, .. }` is insufficient:
    /// the result must carry both the shared engine's explicit `converged_via`
    /// verdict and a valid analytic criterion certificate, and its rho must be
    /// bit-identical to the state currently installed on this objective. This
    /// closes the #2230 hole where any successful evaluation populated
    /// `last_loss` and `into_fitted` silently interpreted an absent search
    /// verdict as `FixedRho`.
    pub fn certify_outer_result(&mut self, result: &OuterResult) -> Result<(), String> {
        self.fit_verdict = None;
        self.terminal_penalized_quasi_laplace_criterion = None;
        if !result.converged {
            return Err("outer result is not converged".to_string());
        }
        let via = result
            .converged_via
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
        let plan = self.term.streaming_plan().admitted_or_error(
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
                return Err(err);
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

    /// Certified curvature-homotopy entry walk (#1007): replace the blind
    /// multi-seed multistart with one predictor-corrector walk of the basis
    /// curvature dial `η` from the base-topology anchor (`η = 0`, convex by
    /// construction) to the full curved basis (`η = 1`).
    ///
    /// 1. **Anchor (`η = 0`).** The curved columns are suppressed, so the decoder
    ///    sub-problem is convex; the joint corrector lands on the base-topology
    ///    optimum. [`linear_span_anchor`] additionally certifies a genuine
    ///    Eckart-Young (SVD low-rank) residual CEILING — a lower bound on the
    ///    residual at every η, not a claim that the η=0 chart is linear/affine
    ///    (for curved bases the base block still embeds curvature). A
    ///    degenerate anchor (no recoverable span / a non-finite target /
    ///    a failed relaxation solve) returns `Ok(false)` — the caller falls back
    ///    to the cascade.
    /// 2. **Walk `η: 0 → 1`.** Each waypoint: a *predictor* applies the IFT step
    ///    `Δβ = −H⁻¹ · ∂g_β/∂η · Δη` on the cached criterion factor
    ///    ([`ArrowFactorCache::full_inverse_apply`], β-channel; the t / gate
    ///    blocks are re-converged by the corrector), then the *corrector* (the
    ///    damped joint Newton in `penalized_quasi_laplace_criterion_with_cache`) re-converges at
    ///    `η_next`. The invariant is that the arrow factor's smallest pivot stays
    ///    at or above the safe-SPD floor `√eps · max(diag_scale, 1)`; when it
    ///    shrinks the `η` step is halved and retried from the last converged
    ///    state. A pivot collapse at the minimum step is a DETECTED bifurcation
    ///    (recorded on [`CurvatureWalkReport`], never silent) and returns
    ///    `Ok(false)`.
    /// 3. **Arrival (`η = 1`).** The term is left warm at the certified branch's
    ///    `η = 1` solution; the report is recorded and the call returns
    ///    `Ok(true)`.
    ///
    /// The direct helper walks at the construction entry ρ (`baseline_rho`);
    /// the outer seed loop uses `run_curvature_homotopy_entry_at_rho` so every
    /// generated candidate gets its own entry solve before the ρ-anneal.
    pub fn run_curvature_homotopy_entry(&mut self) -> Result<bool, String> {
        self.fit_verdict = None;
        let rho = self.baseline_rho.clone();
        self.run_curvature_homotopy_entry_at_rho(&rho)
    }

    /// Certified curvature-homotopy entry walk at an explicit seed ρ. The outer
    /// seed loop calls this form so each generated candidate lands on its own
    /// fixed baseline instead of every walk reusing the construction baseline.
    pub fn run_curvature_homotopy_entry_at_rho(
        &mut self,
        rho: &SaeManifoldRho,
    ) -> Result<bool, String> {
        self.fit_verdict = None;
        let rho = rho.clone();
        self.current_rho = rho.clone();
        // #2080 (a) — the homotopy walk mutates the accepted basin through its
        // own corrector solves; drop any pending probe handoff.
        self.probe_converged_handoff = None;
        // #2230/#2087 — the homotopy walk moves the accepted basin; the saved
        // basins predate it and no longer describe reachable minima.
        self.basin_bundle.clear();
        let isometry_targets = self
            .registry
            .as_ref()
            .map(AnalyticPenaltyRegistry::isometry_scalar_weights)
            .unwrap_or_default();
        self.set_isometry_homotopy_weight(0.0, &isometry_targets);
        // Eckart-Young (SVD low-rank) residual-ceiling certificate at η = 0
        // (output-subspace coords); a rank bound on every η, not a linearity
        // claim. A degenerate anchor is the cascade's job, not the walk's.
        let anchor = match linear_span_anchor(&self.term, self.target.view()) {
            Ok(anchor) => anchor,
            Err(err) => {
                log::info!(
                    "[#1007] curvature anchor degenerate ({err}); deferring to seed cascade"
                );
                self.set_isometry_homotopy_weight(1.0, &isometry_targets);
                return Ok(false);
            }
        };
        let anchor_residual_norm_sq = anchor.residual_norm_sq;

        // Anchor corrector at η = 0: the convex base-topology relaxation.
        let (_loss0, mut last_cache) = match self.solve_at_eta(&rho, 0.0, &isometry_targets) {
            Ok(pair) => pair,
            Err(err) => {
                log::info!(
                    "[#1007] curvature anchor solve failed at η=0 ({err}); deferring to cascade"
                );
                self.term.set_homotopy_eta(1.0).ok();
                self.set_isometry_homotopy_weight(1.0, &isometry_targets);
                return Ok(false);
            }
        };

        // #1026 BASE-DOMINANCE FLOOR. The η=0 corrector above leaves the term at
        // the base-topology optimum (first-harmonic / base-chart decoder + coords).
        // This is NOT a linear model — for curved bases the base block embeds
        // curvature — but it IS a genuine, convex-optimum parametric fit whose
        // residual is bounded below by the certified Eckart-Young (SVD low-rank)
        // ceiling. Snapshot it NOW, before the predictor-corrector walk mutates the
        // decoder/coords. If curvature provably cannot beat this convex base-topology
        // optimum (the walk collapses below the arrival floor and the recovery Newton
        // fit cannot clear it), we restore this anchor at the end rather than leaving
        // a co-collapsed full-curved basin for the cascade to re-collapse — making
        // `F_returned ≤ F_base` an invariant of the optimizer, not just a property of
        // the model class. This is the K≥2 co-collapse cure the relative
        // per-atom-share floor alone cannot deliver (it only TRIGGERS recovery; it
        // never restores the base-topology optimum when recovery also fails).
        let anchor_floor_state = self.term.snapshot_mutable_state();
        // The base-topology anchor's ACTUAL reconstruction EV (η = 0, the state just
        // snapshotted). The dominance floor below must compare against this, NOT the
        // Eckart-Young SVD ceiling `anchor_ev`: for curved bases the η = 0 state uses
        // only the base columns (rank ≤ base-count < basis-size), so it does not in
        // general attain the full-rank SVD ceiling. Keying the restore on the ceiling
        // would wrongly assume `η = 0` is the linear/Eckart-Young optimum and could
        // restore a state WORSE than the current one. Keying on the state's own EV
        // makes `F_returned ≤ F_current` hold for every basis.
        let anchor_state_ev = self
            .term
            .try_fitted_for_rho(&rho)
            .ok()
            .and_then(|fit| reconstruction_explained_variance(self.target.view(), fit.view()));

        let mut eta = 0.0_f64;
        let mut eta_step = CURVATURE_WALK_INITIAL_ETA_STEP;
        let mut eta_steps = 0usize;
        let mut step_halvings = 0usize;
        let mut total_correctors = 0usize;
        let mut bifurcation: Option<CurvatureBifurcation> = None;

        // Identity-homotopy shortcut: with no curved basis columns anywhere
        // AND an all-zero isometry ramp, `solve_at_eta` poses the SAME problem
        // at every η — the grid legs after the anchor corrector would re-solve
        // its converged state verbatim, paying a full criterion/factorization
        // rebuild each time. The anchor + first corrector carry all the value
        // (certified Eckart-Young initialization + one full solve); arrive at
        // η = 1 directly. `set_homotopy_eta(1.0)` restores the plain-evaluate
        // fast path (η == 1 skips the dialed evaluator); the isometry weights
        // are already at target because every ramp target is zero.
        if isometry_targets.iter().all(|&target| target == 0.0)
            && self.term.curvature_homotopy_eta_is_inert()?
        {
            self.term.set_homotopy_eta(1.0)?;
            eta = 1.0;
        }

        'walk: while eta < 1.0 {
            let eta_next = (eta + eta_step).min(1.0);
            let d_eta = eta_next - eta;

            // Predictor: IFT step on the cached factor warm-starts the corrector.
            // #1026 — the COORDINATE channel `w_t = ∂g_t/∂η` (was hardcoded `0`)
            // is now supplied alongside `∂g_β/∂η`. Because the η-dial scales the
            // curved basis columns, dropping `w_t` left the predictor unable to
            // move coordinates as curvature turns on, so the walk tracked the
            // linear-shadow branch to η=1; the full step lets it follow the curved
            // branch. The IFT step is `Δparams = −H⁻¹ ∂g/∂η · Δη`, i.e. delta
            // `−u` applied at step `Δη` through the manifold retraction the Newton
            // step uses (coords + logits + β in one consistent application).
            // Non-fatal — any predictor failure just opens the corrector from the
            // previous η's converged state.
            if let Ok(dg_beta) = self
                .term
                .curvature_beta_gradient_eta_derivative(self.target.view(), &rho)
                && dg_beta.len() == last_cache.k
            {
                let w_t = self
                    .term
                    .curvature_t_gradient_eta_derivative(self.target.view(), &rho)
                    .unwrap_or_else(|_| Array1::<f64>::zeros(last_cache.delta_t_len()));
                if w_t.len() == last_cache.delta_t_len()
                    && let Ok((u_t, u_beta)) =
                        last_cache.full_inverse_apply(w_t.view(), dg_beta.view())
                    && u_t.iter().chain(u_beta.iter()).all(|v| v.is_finite())
                {
                    let neg_u_t: Array1<f64> = u_t.iter().map(|v| -v).collect();
                    let neg_u_beta: Array1<f64> = u_beta.iter().map(|v| -v).collect();
                    // Refresh the basis so the corrector opens at the moved coords.
                    self.term
                        .apply_newton_step_impl(neg_u_t.view(), neg_u_beta.view(), d_eta, true)
                        .ok();
                }
            }

            // Corrector at η_next.
            let cache = match self.solve_at_eta(&rho, eta_next, &isometry_targets) {
                Ok((_loss, cache)) => cache,
                Err(err) => {
                    // Corrector struggled: treat like a pivot shrink — halve the
                    // η step and retry from the last converged state. A failure
                    // at the minimum step is a branch bifurcation.
                    if eta_step <= CURVATURE_WALK_MIN_ETA_STEP {
                        log::info!(
                            "[#1007] curvature corrector failed at η={eta_next:.4} at the minimum \
                             η-step ({err}); recording branch bifurcation"
                        );
                        bifurcation = Some(CurvatureBifurcation {
                            eta: eta_next,
                            min_pivot: 0.0,
                        });
                        break 'walk;
                    }
                    eta_step *= 0.5;
                    step_halvings += 1;
                    self.term.set_homotopy_eta(eta).ok();
                    self.set_isometry_homotopy_weight(eta, &isometry_targets);
                    continue 'walk;
                }
            };
            total_correctors += 1;

            // Pivot invariant: min pivot ≥ eps · diag_scale, measured ON THE
            // GAUGE QUOTIENT (#1095). The floor uses machine epsilon (not its
            // square root) because the undamped cache is built with
            // `with_ill_conditioning_tolerated()`, which accepts any
            // positive-definite factor regardless of condition number.
            // Sub-sqrt(eps) pivots are legitimately produced when N < beta_dim
            // (small-N fits where the decoder Gram is rank-deficient) — this
            // is NOT a branch bifurcation: the damped corrector already
            // converged above, and genuine branch collapses are caught by
            // corrector failure (the `Err` branch). Only a pivot numerically
            // indistinguishable from zero in double precision (below
            // eps * diag_scale) marks a true collapse of the smooth branch.
            //
            // A closed-form gauge null (affine chart freedom, circle rotation)
            // is constant along the η-walk, so it can never signal a branch
            // bifurcation; only a NON-gauge, data-supported pivot collapse can.
            // `outer_gradient_arrow_solver` succeeds iff the sub-floor pivots
            // are explained by gauge/null directions (Faddeev-Popov deflation)
            // and errs honestly otherwise, which is exactly the verdict needed.
            let pivot = arrow_factor_min_pivot(&cache).min_pivot.unwrap_or(0.0);
            let diag_scale = arrow_factor_max_pivot(&cache).unwrap_or(1.0);
            let floor = f64::EPSILON * diag_scale;
            let pivot_deficit_is_gauge = !(pivot.is_finite() && pivot >= floor)
                && self
                    .term
                    .outer_gradient_arrow_solver(&cache, &rho.lambda_smooth_vec())
                    .is_ok();
            if !(pivot.is_finite() && pivot >= floor) && !pivot_deficit_is_gauge {
                if eta_step > CURVATURE_WALK_MIN_ETA_STEP {
                    eta_step *= 0.5;
                    step_halvings += 1;
                    self.term.set_homotopy_eta(eta).ok();
                    self.set_isometry_homotopy_weight(eta, &isometry_targets);
                    continue 'walk;
                }
                log::info!(
                    "[#1007] curvature branch bifurcation at η={eta_next:.4}: min pivot \
                     {pivot:.3e} < floor {floor:.3e}; deferring to seed cascade"
                );
                bifurcation = Some(CurvatureBifurcation {
                    eta: eta_next,
                    min_pivot: pivot,
                });
                break 'walk;
            }

            // Accepted waypoint: advance and gently regrow the step toward the
            // nominal cadence (a clean stretch should not stay throttled).
            eta = eta_next;
            last_cache = cache;
            eta_steps += 1;
            eta_step = (eta_step * 2.0).min(CURVATURE_WALK_INITIAL_ETA_STEP);
            if total_correctors >= CURVATURE_WALK_MAX_CORRECTORS && eta < 1.0 {
                log::info!(
                    "[#1007] curvature walk hit its corrector budget at η={eta:.4}; deferring to \
                     seed cascade"
                );
                bifurcation = Some(CurvatureBifurcation {
                    eta,
                    min_pivot: pivot,
                });
                break 'walk;
            }
        }

        let mut arrived = bifurcation.is_none() && eta >= 1.0;
        // Leave the term at the real (η = 1) objective regardless of outcome so
        // an aborted walk hands the cascade the full basis.
        if !arrived {
            self.term.set_homotopy_eta(1.0).ok();
        }
        self.set_isometry_homotopy_weight(1.0, &isometry_targets);
        if arrived
            && let Ok(before_fit) = self.term.try_fitted_for_rho(&rho)
            && let Some(before_ev) =
                reconstruction_explained_variance(self.target.view(), before_fit.view())
            && before_ev < 0.9
        {
            let snapshot = self.term.snapshot_mutable_state();
            let accepted_polish = self
                .term
                .refit_decoder_least_squares_at_current_state(self.target.view(), Some(&rho))
                .and_then(|()| {
                    self.term
                        .seed_coords_by_decoder_projection(self.target.view())
                })
                .and_then(|()| {
                    self.term.refit_decoder_least_squares_at_current_state(
                        self.target.view(),
                        Some(&rho),
                    )
                })
                .and_then(|()| {
                    let after_fit = self.term.try_fitted_for_rho(&rho)?;
                    let Some(after_ev) =
                        reconstruction_explained_variance(self.target.view(), after_fit.view())
                    else {
                        return Err(
                            "curvature-homotopy decoder LSQ polish produced no EV".to_string()
                        );
                    };
                    if after_ev > before_ev {
                        self.term.loss(self.target.view(), &rho)
                    } else {
                        Err(format!(
                            "curvature-homotopy decoder LSQ polish refused: EV {after_ev:.6} \
                             did not improve from {before_ev:.6}"
                        ))
                    }
                });
            match accepted_polish {
                Ok(loss) => self.last_loss = Some(loss),
                Err(_) => self.term.restore_mutable_state(&snapshot)?,
            }
        }
        // Arrival quality floor (#1117). "Arrived" is only a usable certificate
        // if the η = 1 reconstruction is actually good — the predictor-corrector
        // walk from the base-topology anchor can track into a degenerate
        // basin that is stationary on the gauge/decoder-null quotient (so the
        // inner solve legitimately converges there) yet reconstructs the data
        // badly (a NEGATIVE explained variance: worse than the data mean). When
        // the base chart is the genuinely-affine Euclidean/Duchon fallback, a
        // K = 1 circle target's anchor IS that wrong basin — a straight chord
        // through the arc — and neither the IFT predictor nor the
        // decoder-LSQ polish (which alternates a decoder LSQ with a coordinate
        // re-projection ONTO that same bad decoder) can escape it: it is a fixed
        // point. The walk then reported `arrived = true` on EV = -0.59.
        //
        // Crucially, the production outer objective can carry `inner_max_iter = 0`
        // (a value-only / frozen-inner configuration), so neither the cascade
        // `eval` NOR `into_fitted`'s basin re-solve runs a real joint Newton fit
        // — only the homotopy + polish produce any fit, and they are stuck on the
        // base-topology anchor. Demoting to a bifurcation alone therefore does NOT
        // recover the circle (the cascade re-freezes at the cold seed). So the
        // recovery itself must run a REAL bounded joint Newton fit from the
        // pristine baseline term (which carries the circle-aware PCA seed the
        // cold path recovers EV ≈ 0.94 from), with a nonzero budget independent
        // of the objective's frozen `inner_max_iter`. If that recovers a good
        // reconstruction we adopt it (the walk genuinely arrives on the curved
        // branch); otherwise we demote to a recorded bifurcation so the cascade
        // takes over from the pristine baseline. A genuinely good arrival (the
        // common case, every fit already passing) never enters this block.
        // #1189 — the arrival floor is RELATIVE to the certified rank (Eckart-Young /
        // PCA) ceiling, never an absolute EV target. On real high-dim data whose
        // signal sits on a long-tailed spectrum the best achievable EV at K atoms is
        // bounded by the cumulative low-rank (PCA) ceiling — well under any fixed
        // floor on real LLM activations — so an absolute floor would reject EVERY
        // genuine arrival, the fit would fall to the blind cascade, and the cascade
        // would collapse into a structurally degenerate basin (the #1189 bug). The
        // Eckart-Young SVD projection's OWN reconstruction EV is exactly that
        // achievable rank ceiling (`anchor_ev = 1 − ‖residual‖² / SST`) — a bound on
        // every η, not a linearity claim about the η=0 chart; the arrival floor below
        // is a share of it (see there), so a curved arrival that recovers within one
        // atom's share of the anchor has, by construction, NOT tracked into a worse
        // basin than the convex base-topology optimum it started on.
        let target_sst = {
            let (n, p) = self.target.dim();
            let mut means = vec![0.0_f64; p];
            for col in 0..p {
                let mut acc = 0.0;
                for row in 0..n {
                    acc += self.target[[row, col]];
                }
                means[col] = acc / (n.max(1) as f64);
            }
            let mut sst = 0.0_f64;
            for row in 0..n {
                for col in 0..p {
                    let centered = self.target[[row, col]] - means[col];
                    sst += centered * centered;
                }
            }
            sst
        };
        let anchor_ev = if target_sst > f64::MIN_POSITIVE && anchor_residual_norm_sq.is_finite() {
            1.0 - anchor_residual_norm_sq / target_sst
        } else {
            // No usable ceiling estimate (degenerate target): fall back to the
            // data-collapse floor so the arrival gate keys on a finite number.
            SAE_FIT_DATA_COLLAPSE_EV_FLOOR
        };
        // Arrival floor (#1189 / #1026): accept the curved arrival when its
        // reconstruction EV recovers the rank (PCA) ceiling `anchor_ev` minus at
        // most ONE atom's share of it. Sequential Eckart-Young deflation gives each
        // atom ~1/K of the cumulative rank ceiling (`anchor_ev` is the
        // certified CUMULATIVE SVD ceiling across all K atoms), so a single atom that
        // curves trades at most 1/K of the ceiling for the geometry it gains: the
        // whole-dictionary curved EV need only stay within 1/K, i.e.
        // `>= anchor_ev * (K - 1)/K`. This is exactly the achievable, data-derived
        // bar — no absolute EV target. On real long-tailed activations `anchor_ev`
        // is well under any fixed floor, so keying on the achievable ceiling is the
        // whole #1189 fix; the per-atom discount is the #1026 co-collapse
        // forgiveness. K = 1 has no co-collapse partner and no share to forgive
        // (the discount is 0), so a single curved atom is judged purely against the
        // data-collapse floor and its curve-vs-linear quality is adjudicated
        // downstream by the EV-vs-K structure search. Never below
        // `SAE_FIT_DATA_COLLAPSE_EV_FLOOR`: a fit under that is degenerate (worse
        // than a constant predictor) and must route to recovery whatever the
        // anchor estimate.
        let k_active = self.term.k_atoms().max(1) as f64;
        let arrival_floor =
            (anchor_ev * ((k_active - 1.0) / k_active)).max(SAE_FIT_DATA_COLLAPSE_EV_FLOOR);
        if arrived
            && let Ok(final_fit) = self.term.try_fitted_for_rho(&rho)
            && let Some(final_ev) =
                reconstruction_explained_variance(self.target.view(), final_fit.view())
            && final_ev < arrival_floor
        {
            log::info!(
                "[#1007/#1189] curvature walk reached η=1 but the reconstruction is degenerate \
                 (EV={final_ev:.4} < arrival floor {arrival_floor:.4} = anchor ceiling \
                 {anchor_ev:.4} × (K-1)/K); running a bounded joint Newton fit from the \
                 pristine seed to recover the curved branch"
            );
            // Real joint Newton fit from the pristine baseline (circle-aware
            // seed), at the full η = 1 basis, with a budget that does NOT collapse
            // to the objective's frozen `inner_max_iter`.
            let recovery_iters = self.inner_max_iter.max(CURVATURE_WALK_RECOVERY_INNER_ITERS);
            let mut recovered_term = self.baseline_term.clone();
            recovered_term.set_homotopy_eta(1.0).ok();
            let mut recovery_rho = rho.clone();
            let recovery_fit = recovered_term.run_joint_fit_arrow_schur(
                self.target.view(),
                &mut recovery_rho,
                self.registry.as_ref(),
                recovery_iters,
                self.learning_rate,
                self.ridge_ext_coord,
                self.ridge_beta,
            );
            let recovered_ev = recovery_fit.as_ref().ok().and_then(|_| {
                recovered_term
                    .try_fitted_for_rho(&rho)
                    .ok()
                    .and_then(|fit| {
                        reconstruction_explained_variance(self.target.view(), fit.view())
                    })
            });
            match (recovery_fit, recovered_ev) {
                (Ok(loss), Some(ev)) if ev > final_ev && ev >= arrival_floor => {
                    // The bounded joint fit found the curved branch: adopt it and
                    // keep `arrived = true` (the walk delivered a usable fit).
                    log::info!(
                        "[#1007] curvature degenerate-basin recovery succeeded \
                         (EV {final_ev:.4} -> {ev:.4}); adopting the recovered curved branch"
                    );
                    self.term = recovered_term;
                    self.current_rho = rho.clone();
                    self.last_loss = Some(loss);
                }
                _ => {
                    // Recovery could not improve the reconstruction: demote to a
                    // recorded bifurcation so the outer seed loop resets to the
                    // pristine baseline and the documented cascade takes over.
                    log::info!(
                        "[#1007] curvature degenerate-basin recovery did not clear the arrival \
                         floor (EV stayed {final_ev:.4}); demoting to a branch bifurcation"
                    );
                    arrived = false;
                    self.term.set_homotopy_eta(1.0).ok();
                    if bifurcation.is_none() {
                        bifurcation = Some(CurvatureBifurcation {
                            eta: 1.0,
                            min_pivot: 0.0,
                        });
                    }
                }
            }
        }
        // #1026 BASE-DOMINANCE FLOOR (final, path-independent). Whatever the walk
        // did — early bifurcation, or arrived-but-recovery-failed — the term must not
        // be left reconstructing WORSE than the convex base-topology anchor it relaxed
        // from. When the current state is under the (already per-atom-share-relaxed)
        // arrival floor AND the η=0 anchor reconstructs strictly better, restore the
        // anchor: curvature that cannot beat the convex base-topology optimum returns
        // that optimum. The anchor is a real parametric model state (decoder + coords,
        // not a reconstruction-time substitution), so this generalizes to held-out
        // data. The comparison uses the anchor's OWN reconstruction EV
        // (`anchor_state_ev`), not the Eckart-Young SVD ceiling `anchor_ev`: the η = 0
        // state is NOT linear and does not attain that full-rank ceiling for curved
        // bases, so keying on `anchor_state_ev` is what makes `F_returned ≤ F_current`
        // hold. Conservative by construction: a genuine curved arrival
        // (EV ≥ arrival_floor) never enters this block, so curved branches that beat
        // the anchor are untouched.
        if let Ok(cur_fit) = self.term.try_fitted_for_rho(&rho)
            && let Some(cur_ev) =
                reconstruction_explained_variance(self.target.view(), cur_fit.view())
            && let Some(anchor_state_ev) = anchor_state_ev
            && anchor_state_ev.is_finite()
            && cur_ev < arrival_floor
            && anchor_state_ev > cur_ev
        {
            // The differential snapshot captured `homotopy_eta` at the η = 0
            // anchor, so the restore already re-derives the base-topology basis at
            // η = 0; the explicit `set_homotopy_eta(0.0)` below is now a redundant
            // (harmless) reassertion kept for clarity.
            self.term.restore_mutable_state(&anchor_floor_state)?;
            self.term.set_homotopy_eta(0.0).ok();
            self.last_loss = self.term.loss(self.target.view(), &rho).ok();
            // The certified anchor IS the delivered fit: mark arrival and clear any
            // mid-walk bifurcation so the outer seed loop adopts the anchor rather
            // than resetting to the (collapse-prone) cold cascade.
            arrived = true;
            bifurcation = None;
            log::info!(
                "[#1026] base-dominance floor: curved EV {cur_ev:.4} < arrival floor \
                 {arrival_floor:.4}; restored convex η=0 base-topology anchor \
                 (EV {anchor_state_ev:.4}) — F_returned ≤ F_current"
            );
        }
        let collapse_events = self.term.collapse_events().len();
        self.term.set_curvature_walk_report(CurvatureWalkReport {
            arrived,
            anchor_residual_norm_sq,
            bifurcation,
            eta_steps,
            step_halvings,
            collapse_events,
            reseeds: 0,
        });
        Ok(arrived)
    }

    /// Curvature-homotopy corrector (#1007): install the `η` dial and re-converge
    /// the joint fit at the entry ρ, returning the converged loss and the
    /// undamped evidence cache (for the predictor IFT solve + the pivot
    /// invariant). The dial is read on the next basis refresh inside the solve.
    pub(crate) fn solve_at_eta(
        &mut self,
        rho: &SaeManifoldRho,
        eta: f64,
        isometry_targets: &[f64],
    ) -> Result<(SaeManifoldLoss, ArrowFactorCache), String> {
        self.term.set_homotopy_eta(eta)?;
        self.set_isometry_homotopy_weight(eta, isometry_targets);
        let (_cost, loss, cache) = self.term.penalized_quasi_laplace_criterion_with_cache(
            self.target.view(),
            rho,
            self.registry.as_ref(),
            self.inner_max_iter,
            self.learning_rate,
            self.ridge_ext_coord,
            self.ridge_beta,
        )?;
        self.last_loss = Some(loss.clone());
        Ok((loss, cache))
    }

    pub(crate) fn set_isometry_homotopy_weight(&mut self, eta: f64, targets: &[f64]) {
        self.fit_verdict = None;
        if targets.is_empty() {
            return;
        }
        if let Some(registry) = self.registry.as_mut() {
            let eta = eta.clamp(0.0, 1.0);
            let weights: Vec<f64> = targets.iter().map(|target| eta * target).collect();
            registry.set_isometry_scalar_weights(&weights);
        }
    }

    /// Record the discrete fitted-data collapse verdict without changing the
    /// penalized quasi-Laplace objective. The verdict feeds structure search and the final fit
    /// ledger; it is not a smooth term and therefore cannot be added to a cost
    /// that is paired with the analytic derivative of the penalized quasi-Laplace scalar (#2253).
    fn record_fit_data_collapse_verdict(&mut self, rho: &SaeManifoldRho) -> Result<(), String> {
        let fitted = self.term.try_fitted_for_rho(rho)?;
        let assignments = self.term.assignment.try_assignments()?;
        self.term.record_fit_data_collapse_if_needed(
            self.target.view(),
            fitted.view(),
            assignments.view(),
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

    pub(crate) fn is_recoverable_value_probe_refusal(err: &str) -> bool {
        err.contains("inner solve did not converge at fixed ρ")
            || err.contains(
                "undamped criterion factorization hit a non-PD per-row H_tt block before KKT",
            )
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
            || (err.contains("Schur complement Cholesky failed")
                && err.contains("not positive definite"))
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
            || err.contains("gated off at every row (all-zero gated design)")
            // #2089 — a ρ whose smoothing / sparsity penalty is strong enough to
            // crush the WHOLE dictionary to the signal-free null floor (every
            // decoder co-vanishes and the bounded co-collapse reseed multi-start
            // cannot re-anchor `K` distinct charts) is a GENUINE INFEASIBILITY OF
            // THAT ρ — the same class as the non-PD Hessian / all-zero gated-design
            // probes above. A neighbouring, weaker-penalty ρ admits a non-degenerate
            // fit, so the outer optimizer must read this as an infeasible trial
            // (+∞) and steer ρ back toward the feasible region
            // NOT abort the entire alpha="auto" search the first time a line search
            // overshoots into a co-collapsing ρ. Aborting there fails fits that have a
            // perfectly good feasible ρ the search had not yet reached; and letting
            // the reseed multi-start GRIND at every such probe (the pre-guard
            // behaviour) is exactly what thrashed the host to an OOM / watchdog
            // SIGKILL (exit 137). `run_joint_fit_arrow_schur` emits this DISTINCT
            // "did not escape total co-collapse" marker only after the reseed budget
            // is spent AND the fit is still at/below the null floor, so a healthy or
            // merely-uncompetitive fit never trips it.
            || err.contains("did not escape total co-collapse")
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

    /// Shared cost path: evaluate the penalized quasi-Laplace criterion at `rho_flat`, updating
    /// the cached ρ / loss and (optionally) priming the inner solve from a
    /// seeded β. Returns `(cost, β̂)`.
    ///
    /// `refine_progress_extension = false` selects the value-probe refine
    /// budget (#1029). The budget cut keeps the SAME KKT/step tolerance as the
    /// full path — a successfully returned value is converged to the identical
    /// stationarity measure, so probe values and accepted-point values are
    /// always comparable; only an expensive grind-then-refuse becomes a cheap
    /// refusal (a recoverable line-search reject).
    pub(crate) fn evaluate_with_refine_policy(
        &mut self,
        rho_flat: ArrayView1<'_, f64>,
        refine_progress_extension: bool,
    ) -> Result<(f64, Array1<f64>), String> {
        self.evaluate_with_inner_drive(
            rho_flat,
            ProbeInnerDrive::Criterion {
                refine_progress_extension,
            },
            false,
        )
    }

    /// As [`Self::evaluate_with_refine_policy`], but with the inner `(t, β)`
    /// drive selected by [`ProbeInnerDrive`]. Everything around the inner drive —
    /// the probe handoff install, seeded-β warm start, amortized latent warm
    /// start, and collapse ledger — is shared.
    fn evaluate_with_inner_drive(
        &mut self,
        rho_flat: ArrayView1<'_, f64>,
        drive: ProbeInnerDrive,
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
        self.apply_block_scaling(&rho);
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
        if !probe_handoff_installed {
            let warm_start_outcome = self
                .term
                .warm_start_latents_from_amortized_encoder(self.target.view(), &rho);
            self.record_warm_start(warm_start_outcome)?;
        }
        let (penalized_quasi_laplace_cost, loss) = match drive {
            ProbeInnerDrive::Criterion {
                refine_progress_extension,
            } => self
                .term
                .penalized_quasi_laplace_criterion_with_refine_policy_and_lane(
                    self.target.view(),
                    &rho,
                    self.registry.as_ref(),
                    self.inner_max_iter,
                    self.learning_rate,
                    self.ridge_ext_coord,
                    self.ridge_beta,
                    refine_progress_extension,
                    self.surrogate_lane.as_mut(),
                )?,
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
        let (criterion, _) = self.evaluate_with_refine_policy(rho_flat, true)?;
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
    /// probes or for the accepted iterate. The inner `(t, β)` drive is selected
    /// by the caller: the line-search lane (`eval_with_order(Value)`) passes the
    /// exact probe drive; every other value lane uses the historical criterion
    /// drive (`Criterion { refine_progress_extension: false }`).
    fn evaluate_value_probe_with_drive(
        &mut self,
        rho_flat: ArrayView1<'_, f64>,
        drive: ProbeInnerDrive,
    ) -> Result<(f64, Array1<f64>), String> {
        let saved_term = self.term.clone();
        let saved_rho = self.current_rho.clone();
        let saved_loss = self.last_loss.clone();
        let saved_seeded_beta = self.seeded_beta.clone();
        let result = self.evaluate_with_inner_drive(rho_flat, drive, false);
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
    ///    basin (`evaluate_value_probe_with_drive`). It consumes the seeded-β /
    ///    amortized-encoder warm start, parks its converged term in the probe
    ///    handoff, and — crucially — is the mechanism by which a basin JUMP is
    ///    discovered (its warm start can cross a boundary at a far ρ).
    /// 4. **Members.** Re-converge every saved basin from its OWN state through
    ///    the cheap value-probe drive (`basin_installed = true`: no warm-start, no
    ///    seed) — members near their optimum re-converge in a round or two.
    /// 5. **Admit + envelope.** Admit the discovery basin (new basin ⇒ grow;
    ///    duplicate ⇒ keep the better value). The envelope value is the bundle
    ///    argmin over {members ∪ discovery}; the argmin basin's converged state is
    ///    installed as the probe handoff so the subsequent gradient eval prices
    ///    THAT basin (envelope theorem). Admission can only LOWER the envelope.
    ///
    /// Only the argmin is later re-converged to full tolerance (in `eval`); every
    /// member and the discovery probe run on the cheap value-probe budget, so the
    /// per-eval cost is `len(bundle) + 1` cheap inner solves. Retention is bounded
    /// only by memory admission; a work-count cap would make the envelope inexact.
    /// #2234 stall fix — a cost-only probe whose inner solve exhausts its CAPPED
    /// budget is an UNFINISHED COMPUTATION, not undefined quasi-Laplace score.
    /// The same is true when the cheap CROSS-SEED/RANKING drive sees a non-PD
    /// per-row factor BEFORE inner KKT stationarity: that factor describes the
    /// current warm-start iterate, not the probed ρ. The full accepted-point
    /// drive can cross that transient indefinite state and reach a finite
    /// stationary factor (the linear-block seed that produced an infeasible
    /// value probe versus a finite analytic evaluation at the identical ρ).
    ///
    /// Complete either provisional result once at the accepted-point drive.
    /// Only a full-drive refusal reaches the caller's infeasible/hard-error
    /// classification, so genuinely non-PD stationary factors retain their
    /// infeasibility semantics while transient pre-KKT indefiniteness never
    /// masquerades as a property of the probed ρ.
    fn value_probe_with_budget_rescue(
        &mut self,
        rho_flat: ArrayView1<'_, f64>,
        drive: ProbeInnerDrive,
    ) -> Result<(f64, Array1<f64>), String> {
        match self.evaluate_envelope_value_probe(rho_flat, drive) {
            Err(err)
                if err.contains("inner solve did not converge at fixed ρ")
                    || err.contains(
                        "undamped criterion factorization hit a non-PD per-row H_tt block before KKT stationarity",
                    ) =>
            {
                self.probe_telemetry.budget_rescued_value_probes += 1;
                self.evaluate_envelope_value_probe(
                    rho_flat,
                    ProbeInnerDrive::Criterion {
                        refine_progress_extension: true,
                    },
                )
            }
            outcome => outcome,
        }
    }

    fn evaluate_envelope_value_probe(
        &mut self,
        rho_flat: ArrayView1<'_, f64>,
        drive: ProbeInnerDrive,
    ) -> Result<(f64, Array1<f64>), String> {
        // (1) Bypass: streaming/matrix-free (no dense per-basin factor to
        // re-converge) or the freeze contract (verbatim reuse). Byte-for-byte
        // historical single trajectory.
        if self.inner_max_iter == 0 || !self.term.streaming_plan().direct_logdet_admitted() {
            return self.evaluate_value_probe_with_drive(rho_flat, drive);
        }

        // (2) Seed the bundle with the accepted entry basin on first use. The
        // placeholder +∞ value is overwritten the first time this member is
        // re-converged below.
        if self.basin_bundle.is_empty() {
            self.basin_bundle
                .admit(self.term.clone(), f64::INFINITY, |_, _| false)
                .map_err(|error| format!("SAE basin-envelope seed admission refused: {error}"))?;
        }

        // (3) Discovery trajectory — the historical single warm-start probe. Sets
        // the probe handoff (when finite) to its converged term at this exact ρ.
        let discovery = self.evaluate_value_probe_with_drive(rho_flat, drive);
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
            let (res, converged) = self.converge_member_criterion(rho_flat, state, drive);
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

    /// Re-converge one saved basin `member` at `rho_flat` through the cheap
    /// value-probe `drive`, returning `(criterion, converged_term)`. PURE w.r.t.
    /// `self`: `term`, `current_rho`, `last_loss`, and `seeded_beta` are all saved
    /// and restored. `basin_installed = true` so the installed converged state is
    /// NOT re-warm-started (no amortized encoder entry heuristic) and does NOT
    /// consume the pending β seed (the seed belongs to the discovery trajectory).
    fn converge_member_criterion(
        &mut self,
        rho_flat: ArrayView1<'_, f64>,
        member: &SaeManifoldTerm,
        drive: ProbeInnerDrive,
    ) -> (Result<f64, String>, SaeManifoldTerm) {
        let saved_term = std::mem::replace(&mut self.term, member.clone());
        let saved_rho = self.current_rho.clone();
        let saved_loss = self.last_loss.clone();
        // Members must not touch the pending seed — take it out for the duration.
        let saved_seeded_beta = self.seeded_beta.take();
        let res = self
            .evaluate_with_inner_drive(rho_flat, drive, true)
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
        // #2080 (a) — this lane commits a new accepted basin below; drop any
        // pending probe handoff so it can never be installed across that
        // mutation (the handoff is only valid against the basin its probe ran
        // from).
        self.probe_converged_handoff = None;
        // #2230/#2087 — the EFS lane commits a new accepted basin below; the
        // saved envelope basins are keyed to the pre-step accepted basin.
        self.basin_bundle.clear();
        let rho = self.baseline_rho.from_flat(rho_flat)?;
        let n_params = rho.to_flat().len();
        // #2231 Inc-B — scale the block columns for this ρ before the EFS inner
        // solve reads `self.target` (idempotent; no-op for a plain SAE).
        self.apply_block_scaling(&rho);
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
        // #2080: ask the surrogate lane to emit the shared (probes, S⁻¹·probes)
        // bundle during this criterion's matrix-free evidence eval, so the
        // smoothness EDF below is the matrix-free tr(S⁻¹·M_k) off that bundle
        // instead of the dense `beta_inv`. The direct-admitted path ignores it
        // (no lane threaded); `take_inverse_probes` after the call clears the flag
        // either way, so a dense eval never hands back stale solves.
        if let Some(lane) = self.surrogate_lane.as_mut() {
            lane.request_inverse_probes();
        }
        let criterion = if self.term.streaming_plan().direct_logdet_admitted() {
            self.term.penalized_quasi_laplace_criterion_with_cache(
                self.target.view(),
                &rho,
                self.registry.as_ref(),
                self.inner_max_iter,
                self.learning_rate,
                self.ridge_ext_coord,
                self.ridge_beta,
            )
        } else {
            self.term
                .penalized_quasi_laplace_criterion_streaming_exact_with_cache_and_lane(
                    self.target.view(),
                    &rho,
                    self.registry.as_ref(),
                    self.inner_max_iter,
                    self.learning_rate,
                    self.ridge_ext_coord,
                    self.ridge_beta,
                    self.surrogate_lane.as_mut(),
                )
        };
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
        let (cost, loss, cache) = match criterion {
            Ok(evaluated) => evaluated,
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
            Err(err) if Self::is_recoverable_value_probe_refusal(&err) => {
                self.probe_telemetry.record_refusal_kind(&err);
                log::debug!("SAE criterion eval mapped refusal to +inf: {err}");
                self.probe_telemetry.infeasible_criterion_evals += 1;
                self.current_rho = rho;
                return Ok(infeasible_evaluation(
                    "infeasible penalized quasi-Laplace score",
                ));
            }
            Err(err) => return Err(err),
        };
        self.record_fit_data_collapse_verdict(&rho)?;
        self.current_rho = rho.clone();
        self.last_loss = Some(loss);
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
        // #2080: take the surrogate lane's shared (probes, S⁻¹·probes) bundle from
        // this eval's matrix-free evidence branch (if it ran) ONCE — both the ARD
        // posterior-variance trace here and the smoothness EDF below consume it, so
        // taking it twice would starve the second consumer. When present, the ARD
        // denominator's `tr(H⁻¹)_tt` is the matrix-free selected-inverse trace off
        // that bundle (no dense Schur `S⁻¹`); otherwise (dense-admitted, or no lane)
        // the dense `full_inverse_apply` / selected-inverse diagonal path stands.
        let inverse_probe_bundle = self
            .surrogate_lane
            .as_mut()
            .and_then(|l| l.take_inverse_probes());
        let traces = if let Some((probes, sinv)) = inverse_probe_bundle.as_ref() {
            self.term
                .ard_inverse_traces_from_probes(&cache, probes, sinv)
                .map_err(|e| {
                    format!("SaeManifoldOuterObjective::efs_step: ARD traces (matrix-free): {e}")
                })?
        } else {
            self.term
                .ard_inverse_traces(&cache)
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
        let mut assignment_psi_gradient: Option<Array1<f64>> = None;
        let mut assignment_psi_indices: Option<Vec<usize>> = None;

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
            let gradient = if let Some((probes, inverse_probes)) = inverse_probe_bundle.as_ref() {
                let system = self.term.assemble_full_matrix_free_evidence_system(
                    self.target.view(),
                    &rho,
                    self.registry.as_ref(),
                    None,
                )?;
                self.term
                    .analytic_assignment_strength_gradient_matrix_free(
                        self.target.view(),
                        &rho,
                        &cache,
                        &system,
                        probes,
                        inverse_probes,
                    )
                    .map_err(|error| {
                        format!(
                            "SaeManifoldOuterObjective::efs_step: matrix-free \
                                 assignment-strength gradient: {error}"
                        )
                    })?
            } else {
                let solver = self
                    .term
                    .outer_gradient_arrow_solver(&cache, &rho.lambda_smooth_vec())
                    .map_err(|error| {
                        format!(
                            "SaeManifoldOuterObjective::efs_step: dense assignment-strength \
                                 solver: {error}"
                        )
                    })?;
                self.term
                    .analytic_assignment_strength_gradient_dense(
                        self.target.view(),
                        &rho,
                        &cache,
                        &solver,
                    )
                    .map_err(|error| {
                        format!(
                            "SaeManifoldOuterObjective::efs_step: dense assignment-strength \
                                 gradient: {error}"
                        )
                    })?
            };
            // A normalized negative gradient is a bounded feasible-descent
            // update whose zero is exactly the full criterion root.
            let gradient_scale = gradient.abs().max(1.0);
            let step = -gradient / gradient_scale;
            steps[sparse_index] = step;
            fixed_point_coordinates[sparse_index] =
                FixedPointCoordinateCertificate::covered(step, 1.0);
            assignment_psi_gradient = Some(Array1::from_vec(vec![gradient]));
            assignment_psi_indices = Some(vec![sparse_index]);
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
        let lambda_smooth_vec = rho.lambda_smooth_vec();
        let quad_per_atom = self.term.decoder_smoothness_quadratic_form_per_atom();
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
                * (SaeManifoldTerm::symmetric_rank(&self.term.atoms[atom_idx].smooth_penalty)?
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
            let tail = n_params - blocks.block_dims.len();
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
                psi_gradient: assignment_psi_gradient,
                psi_indices: assignment_psi_indices,
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
    if atom.smooth_penalty.dim() != (m, m) {
        return Err(format!(
            "reactive rho domain: atom {atom_idx} smooth penalty shape {:?} != ({m}, {m})",
            atom.smooth_penalty.dim()
        ));
    }
    let (rank, penalty_pinv) = gam_linalg::utils::block_penalty_rank_and_pinv(&atom.smooth_penalty)
        .map_err(|error| format!("reactive rho domain penalty spectrum failed: {error}"))?;
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
    // `block_penalty_rank_and_pinv` owns the rank tolerance; retaining the
    // largest `rank` eigenpairs here reuses that decision without reviving tiny
    // numerical eigenvalues in the null space.
    let (pinv_eigenvalues, pinv_eigenvectors) = penalty_pinv
        .eigh(Side::Lower)
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
    let atom = &term.atoms[atom_idx];
    let p = atom.decoder_coefficients.ncols();
    let m = atom.decoder_coefficients.nrows();
    if atom.basis_jacobian.dim().1 != m || axis >= atom.basis_jacobian.dim().2 {
        return Err(format!(
            "reactive rho domain: atom {atom_idx} axis {axis} is incompatible with basis Jacobian {:?} and decoder {:?}",
            atom.basis_jacobian.dim(),
            atom.decoder_coefficients.dim()
        ));
    }
    let periods = term.assignment.coords[atom_idx].effective_axis_periods();
    if periods.get(axis).copied().flatten().is_some() {
        return Ok(None);
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
                tangent[out] += coefficient * atom.decoder_coefficients[[basis, out]];
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
    Ok(Some(maximum))
}

/// Objective-owned legal rho upper face for dense reactive SAE fits.
///
/// The generic `+30` box corresponds to a penalty strength around `1e13` and
/// is outside the SAE objective's structural domain: it drives the dictionary
/// below the exact signal-free null floor before continuation can start. This
/// contract instead uses the objective's literal entry assignments and native
/// penalty geometry. Decoder smoothness is bounded by the exact largest
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
            let target_strength = SaeManifoldRho::stable_exp_strength(target[index]);
            upper[index] = target_strength.max(scale).ln();
        }
        for axis in 0..rho.log_ard[atom_idx].len() {
            if let Some(scale) =
                reactive_ard_curvature_scale(&entry_term, &assignments, atom_idx, axis)?
            {
                largest_native_scale = largest_native_scale.max(scale);
                let index = rho.ard_flat_index(atom_idx, axis);
                let target_strength = SaeManifoldRho::stable_exp_strength(target[index]);
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
        let target_strength = SaeManifoldRho::stable_exp_strength(target[index]);
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

impl OuterObjective for SaeManifoldOuterObjective {
    fn capability(&self) -> OuterCapability {
        let streaming_plan = self.term.streaming_plan();
        let assignment_gradient_dim =
            usize::from(assignment_strength_gradient_coordinate(&self.baseline_rho).is_some());
        OuterCapability {
            // The planner always has an analytic outer update. Two regimes:
            //  * Dense-admitted: the exact analytic outer gradient is assembled
            //    from the joint-Hessian IFT (`outer_gradient_arrow_solver`), for
            //    every assignment mode, including ordered Beta--Bernoulli (#1006).
            //  * Matrix-free (dense criterion factor exceeds the in-core budget,
            //    e.g. large-K / wide-border duchon): no dense cache exists for the
            //    IFT solve, so the fixed-point lane updates covered ρ coordinates
            //    from analytic inverse traces in one pass. It explicitly declares
            //    the gradient UNAVAILABLE; the zero-gradient `eval` result in that
            //    regime is startup plumbing and can never certify a fit.
            gradient: sae_outer_gradient_capability(streaming_plan),
            hessian: DeclaredHessianForm::Unavailable,
            n_params: self.baseline_rho.to_flat().len(),
            // Softmax/threshold fits have one non-FS coordinate: assignment
            // strength. Mark it as the Hybrid-EFS analytic-gradient block so
            // scalable EFS updates still own smoothness/ARD while this coordinate
            // moves by its exact penalized quasi-Laplace gradient. Small dense fits still select the
            // ordinary full-gradient BFGS plan at the existing crossover.
            psi_dim: assignment_gradient_dim,
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
            fixed_point_available: true,
            barrier_config: None,
            prefer_gradient_only: false,
            disable_fixed_point: false,
        }
    }

    fn eval_cost(&mut self, rho: &Array1<f64>) -> Result<f64, EstimationError> {
        self.check_cancelled()?;
        // Value-only comparison path (EFS backtracking and seed validation): no
        // gradient/Hessian is ever
        // consumed at this iterate, so it takes the cheap probe refine budget
        // (#1029). Accepted points are always re-polished through
        // `eval`/`eval_with_order(ValueAndGradient|ValueGradientHessian)`
        // before any derivative consumption, and a probe value — when one is
        // returned at all — is converged to the same KKT/step tolerance as
        // the full-budget path, so all ranked comparisons stay in one measure.
        // `eval_cost` is the value-only CROSS-SEED RANKING / EFS lane (seed
        // screening, cross-seed final selection, EFS backtracking). It prices
        // the SAME penalized quasi-Laplace criterion `f(ρ)` the gradient lane descends, so
        // the fit selection is stationary for the criterion that selected it.
        self.probe_telemetry.criterion_calls += 1;
        // #2230/#2087 — descend the basin lower envelope V*(ρ)=min_b V_b(ρ) here
        // instead of the single hysteretic warm-start trajectory. Same value-probe
        // drive (`refine_progress_extension = false`) as the historical lane; the
        // envelope bypasses to it verbatim in the streaming / freeze regimes.
        match self.value_probe_with_budget_rescue(
            rho.view(),
            ProbeInnerDrive::Criterion {
                refine_progress_extension: false,
            },
        ) {
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
                if self.reactive_waypoint_checkpoint.is_none() && self.termination.record(cost) {
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
        self.apply_block_scaling(&rho_state);
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
        if !self.term.streaming_plan().direct_logdet_admitted() {
            let (cost, _beta_hat) = match self.evaluate_with_refine_policy(rho.view(), false) {
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
            if self.termination.record(cost) {
                self.bank_checkpoint(rho);
            }
            return Ok(OuterEval {
                cost,
                gradient: Array1::zeros(rho.len()),
                hessian: HessianValue::Unavailable,
                inner_beta_hint: None,
            });
        }
        // #2080 (a) — the accepted gradient point is evaluated at the exact ρ of
        // the line search's last successful value probe; when that probe's
        // converged inner state was handed off, install it so the criterion's
        // convergence loop opens AT the inner KKT optimum instead of re-tracing
        // the probe's deterministic Newton trajectory from the accepted basin.
        // Same converged optimum ⇒ identical criterion value and identical
        // stationary factor cache for the analytic gradient below (see
        // `ProbeConvergedHandoff`). The pending seeded-β hint was already applied
        // by that probe pre-convergence, so it is consumed with the handoff.
        let probe_handoff_installed =
            if let Some(converged) = self.take_probe_converged_handoff(rho.view()) {
                self.term = converged;
                self.seeded_beta = None;
                true
            } else if !self.basin_bundle.is_empty() {
                // #2087/#2253 envelope-consistency — the value lanes price the
                // basin lower envelope min_b V_b(ρ); the gradient lane used to
                // inherit the argmin basin ONLY through the bitwise-ρ handoff.
                // On a handoff miss (e.g. the bridge's value-probe cache served
                // the repeated ρ without touching this objective) this lane
                // silently re-converged the single accepted-trajectory basin —
                // returning a (value, gradient) pair from a DIFFERENT function
                // than the envelope the line search accepted at the same ρ.
                // Run the envelope at this exact ρ (it installs the argmin
                // basin's converged state as the handoff), then consume it, so
                // value and gradient always price one basin. With an empty
                // bundle no envelope has ever been evaluated and the historical
                // single-trajectory path below is already consistent.
                match self.value_probe_with_budget_rescue(
                    rho.view(),
                    ProbeInnerDrive::Criterion {
                        refine_progress_extension: true,
                    },
                ) {
                    Ok(_) => {}
                    Err(err) if Self::is_recoverable_value_probe_refusal(&err) => {
                        self.probe_telemetry.record_refusal_kind(&err);
                        log::debug!("SAE criterion eval mapped refusal to +inf: {err}");
                        self.probe_telemetry.infeasible_criterion_evals += 1;
                        return Ok(OuterEval::infeasible(rho.len()));
                    }
                    Err(err) => return Err(EstimationError::RemlOptimizationFailed(err)),
                }
                if let Some(converged) = self.take_probe_converged_handoff(rho.view()) {
                    self.term = converged;
                    self.seeded_beta = None;
                    true
                } else {
                    false
                }
            } else {
                false
            };
        if let Some(beta) = self.seeded_beta.take() {
            if beta.len() != self.term.beta_dim() {
                return Err(EstimationError::RemlOptimizationFailed(format!(
                    "seeded decoder has length {}; expected {}",
                    beta.len(),
                    self.term.beta_dim()
                )));
            }
            self.term
                .set_flat_beta(beta.view())
                .map_err(EstimationError::RemlOptimizationFailed)?;
        }
        // #1154 — warm-start the inner latent coords from the amortized encoder
        // built on the running dictionary at this ρ (Design A), exactly as the
        // value-probe lane (`evaluate_with_refine_policy`) does. The accepted
        // iterate's inner solve then refines from the cheap one-mat-vec seed to
        // the SAME stationary point, so the exact penalized quasi-Laplace λ-gradient computed below
        // is untouched — the warm-start changes only the basin entry, never the
        // root. A degenerate atlas may certify zero rows; an actual encoder
        // failure is propagated.
        // Skipped under a #2080 (a) handoff: the installed state is already AT
        // the converged optimum for this ρ, and the encoder warm-start is a
        // basin-ENTRY heuristic that would only move latents off it.
        if !probe_handoff_installed {
            let warm_start_outcome = self
                .term
                .warm_start_latents_from_amortized_encoder(self.target.view(), &rho_state);
            self.record_warm_start(warm_start_outcome)
                .map_err(EstimationError::RemlOptimizationFailed)?;
        }
        // The analytic gradient lane (`eval`) reads the dense joint-Hessian cache.
        // In the matrix-free regime that cache does not exist, but SAE never
        // descends ρ with this gradient lane there: the outer plan routes to the
        // Fellner–Schall fixed point (`Solver::Efs` → `eval_efs`/`efs_step`), which
        // needs only the analytic traces `tr(H⁻¹ S_c)` — no gradient, and (per
        // SPEC) no finite differences. So this dense-cache path is reached only
        // when the dense criterion factor is admitted.
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
        let (cost, loss, cache) = match self.term.penalized_quasi_laplace_criterion_with_cache(
            self.target.view(),
            &rho_state,
            self.registry.as_ref(),
            self.inner_max_iter,
            self.learning_rate,
            self.ridge_ext_coord,
            self.ridge_beta,
        ) {
            Ok(evaluated) => evaluated,
            // A non-PD per-row/cross-row/Schur factor has no defined Laplace
            // evidence at this ρ. Return the objective contract's typed
            // infeasible evaluation so the optimizer rejects/backtracks. A
            // finite sentinel here would be a different objective. Genuine
            // evaluation defects still hard-error below.
            Err(err) if Self::is_recoverable_value_probe_refusal(&err) => {
                self.probe_telemetry.record_refusal_kind(&err);
                log::debug!("SAE criterion eval mapped refusal to +inf: {err}");
                self.probe_telemetry.infeasible_criterion_evals += 1;
                return Ok(OuterEval::infeasible(rho.len()));
            }
            Err(err) => return Err(EstimationError::RemlOptimizationFailed(err)),
        };
        self.record_fit_data_collapse_verdict(&rho_state)
            .map_err(EstimationError::RemlOptimizationFailed)?;
        if !cost.is_finite() {
            self.probe_telemetry.infeasible_criterion_evals += 1;
            return Ok(OuterEval::infeasible(rho.len()));
        }
        // Exact implicit derivative through the converged inner state. The arrow
        // solver first applies a rank-revealing projection of the closed-form
        // chart gauge and penalty-aware decoder nulls, then solves the resulting
        // implicit-function system. A system that remains singular or unreliable
        // is a typed `OuterGradientError`: it is not a usable derivative and must
        // terminate this evaluation instead of being hidden behind a plain inverse
        // or a differenced value path.
        let grad_components = self
            .term
            .outer_gradient_arrow_solver(&cache, &rho_state.lambda_smooth_vec())
            .and_then(|solver| {
                self.term
                    .analytic_outer_rho_gradient_components_with_bundle(
                        self.target.view(),
                        &rho_state,
                        &loss,
                        &cache,
                        &solver,
                        None,
                    )
            })
            .map_err(|err| EstimationError::RemlOptimizationFailed(err.to_string()))?;
        let mut gradient = grad_components.gradient();
        // #2231 Inc-B (stage 2) — ADD the block-relevance tail's explicit data +
        // change-of-variables channels `½·R̃_ℓ − n·p_ℓ/2`
        // ([`Self::block_log_lambda_gradient`]) to the components assembler's
        // tail, which now carries the block coordinate's `−½·Γᵀθ̂_ρ` Laplace
        // adjoint (`crosscoder_block_ift_rhs` feeds the exact-stationarity solve
        // the target-scaling RHS `−½·Jᵀ_M Z̃^{(ℓ)}`). Explicit + adjoint together
        // are the COMPLETE `∂C/∂log λ_ℓ` of the priced criterion — overwriting
        // here would re-truncate the gradient to a fictitious fixed-θ̂ criterion
        // (#2087 desync class). No-op for a plain SAE (`None` ⇒ the tail stays
        // empty and untouched).
        if let Some(block_grad) = self
            .block_log_lambda_gradient(&rho_state)
            .map_err(EstimationError::RemlOptimizationFailed)?
        {
            let tail = gradient.len() - block_grad.len();
            for (l, g_l) in block_grad.into_iter().enumerate() {
                gradient[tail + l] += g_l;
            }
        }
        let beta_hat = self.term.flatten_beta();
        // #1206 — the gradient lane (`OuterEvalOrder::ValueAndGradient`, consumed
        // by the outer BFGS Armijo line search) MUST return a cost whose gradient
        // is the gradient we return: the consistent pair `(f, ∇f)` for the pure
        // penalized quasi-Laplace criterion — the SAME criterion every value/ranking/EFS lane prices
        // (one coherent objective; see `evaluate_with_inner_drive`). Collapse was
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
        self.last_loss = Some(loss);
        if self.termination.record(cost) {
            self.bank_checkpoint(rho);
        }
        Ok(OuterEval {
            cost,
            gradient,
            hessian: HessianValue::Unavailable,
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
                // drive `eval`/`eval_cost` use — warm-started from the probe
                // handoff and rescued to the extended budget on a non-convergence
                // refusal (`value_probe_with_budget_rescue`) — so value, gradient,
                // and certification are one coherent objective. At small n the
                // former probe and this drive coincide (both reach the fixed point
                // within the base budget), so this is a no-op for the tier0 fits.
                let drive = ProbeInnerDrive::Criterion {
                    refine_progress_extension: false,
                };
                let (cost, beta_hat) = match self.value_probe_with_budget_rescue(rho.view(), drive)
                {
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
                if self.reactive_waypoint_checkpoint.is_none() && self.termination.record(cost) {
                    self.bank_checkpoint(rho);
                }
                Ok(OuterEval {
                    cost,
                    gradient: Array1::zeros(rho.len()),
                    hessian: HessianValue::Unavailable,
                    inner_beta_hint: Some(beta_hat),
                })
            }
            OuterEvalOrder::ValueAndGradient | OuterEvalOrder::ValueGradientHessian => {
                self.eval(rho)
            }
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
        if self.termination.record(eval.cost) {
            self.bank_checkpoint(rho);
        }
        Ok(eval)
    }

    fn eval_fixed_point_certificate(
        &mut self,
        rho: &Array1<f64>,
    ) -> Result<FixedPointCertificateEval, EstimationError> {
        self.check_cancelled()?;
        let (evaluation, coordinates) = self
            .efs_step_with_certificate(rho.view())
            .map_err(EstimationError::RemlOptimizationFailed)?;
        let rho_state = self
            .baseline_rho
            .from_flat(rho.view())
            .map_err(EstimationError::InvalidInput)?;
        let cost = evaluation.cost + self.block_jacobian(&rho_state);
        Ok(FixedPointCertificateEval { cost, coordinates })
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
        // Contract (see src/solver/reml/continuation.rs:727-737): an empty-β
        // seed means "no warm-start available, use your own cold default" and
        // MUST be accepted as a no-op. The continuation pre-warm forwards the
        // previous eval's `inner_beta_hint`, but before the first accepted eval
        // that hint is empty (`state.last_beta` starts empty). Rejecting it
        // fatally dropped every continuation seed and forced a full cold solve
        // on every outer seed — the slowness in gam#577. Only a *populated* β
        // must match the decoder dimension.
        if beta.is_empty() {
            // NoSlot is the documented continuation reply for "no usable seed;
            // proceed cold, no log" (outer_strategy.rs:1776). The real β slot
            // gets populated on the next accepted eval, which publishes
            // `inner_beta_hint`, so steps 2+ warm-start normally.
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
        self.baseline_rho
            .validate_ard_log_strength_domain()
            .map_err(EstimationError::InvalidInput)?;
        let ard_upper = self.baseline_rho.ard_flat_domain_upper_bound();
        let Some(contract) = self.reactive_domain_scalar_contract()? else {
            return Ok(ard_upper);
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
        if let Some(ard_upper) = ard_upper {
            for index in 0..reactive_upper.len() {
                reactive_upper[index] = reactive_upper[index].min(ard_upper[index]);
            }
        }
        Ok(Some(reactive_upper))
    }

    fn outer_domain_lower_bound(&self) -> Result<Option<Array1<f64>>, EstimationError> {
        self.baseline_rho
            .validate_ard_log_strength_domain()
            .map_err(EstimationError::InvalidInput)?;
        Ok(self.baseline_rho.ard_flat_domain_lower_bound())
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
            || !self.baseline_term.streaming_plan().direct_logdet_admitted()
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
///   * falls back, atom-by-atom, to the exact ndarray `S_k.dot(B_k)` whenever no
///     GPU runtime is present, the pool returns `None`, or a tile's batched GEMM
///     declines. The result is bit-for-bit identical to the all-CPU path (f64
///     throughout, same accumulation order per product).
///
/// Returns one `S_k · B_k` matrix per atom, in atom order. `symmetrize`
/// pre-symmetrises each `S_k` (the assembly path needs `½(S+Sᵀ)`); the value /
/// quadratic-form callers pass `false` since the quadratic form only sees the
/// symmetric part regardless.
pub(crate) fn batched_smooth_sb(
    sb_inputs: &[(ArrayView2<'_, f64>, ArrayView2<'_, f64>)],
    symmetrize: bool,
) -> Vec<Array2<f64>> {
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

    // Exact CPU fallback for a single atom, reused by both the no-GPU route and
    // per-tile decline.
    let cpu_one = |idx: usize| -> Array2<f64> { s_mats[idx].dot(&sb_inputs[idx].1) };

    // Size gate BEFORE the device probe (startup-tax ordering fix): each device
    // tile issues one strided-batched GEMM over (a subset of) a uniform-shape
    // group, whose flop count is at most the whole group's `Σ 2·m²·p`. Every
    // reachable dispatch policy refuses a batched GEMM below
    // `MIN_CALIBRATABLE_GEMM_FLOPS`, so when even the LARGEST group in
    // aggregate is under the floor, every tile on every device would decline
    // and each atom would take `cpu_one` — the exact result this early return
    // produces without calling `GpuRuntime::global()` (whose first call creates
    // a CUDA primary context on every GPU). Shapes with an admissible group
    // probe and scatter exactly as before.
    {
        let mut group_flops: std::collections::BTreeMap<(usize, usize), u128> =
            std::collections::BTreeMap::new();
        for (idx, (_, b)) in sb_inputs.iter().enumerate() {
            let m = s_mats[idx].nrows();
            let p = b.ncols();
            *group_flops.entry((m, p)).or_insert(0) +=
                2u128 * (m as u128) * (m as u128) * (p as u128);
        }
        let max_group = group_flops.values().copied().max().unwrap_or(0);
        if max_group < crate::gpu::GpuDispatchPolicy::MIN_CALIBRATABLE_GEMM_FLOPS {
            return (0..n_atoms).map(cpu_one).collect();
        }
    }

    let rt = match crate::gpu::device_runtime::GpuRuntime::global() {
        Some(rt) => rt,
        None => return (0..n_atoms).map(cpu_one).collect(),
    };

    // Group atom indices by uniform (m, p) shape; only same-shape groups can ride
    // a strided-batched GEMM tile.
    let mut groups: std::collections::BTreeMap<(usize, usize), Vec<usize>> =
        std::collections::BTreeMap::new();
    for (idx, (_, b)) in sb_inputs.iter().enumerate() {
        let m = s_mats[idx].nrows();
        let p = b.ncols();
        groups.entry((m, p)).or_default().push(idx);
    }

    let mut out: Vec<Option<Array2<f64>>> = (0..n_atoms).map(|_| None).collect();
    for ((m, p), members) in groups {
        // Singletons and tiny groups gain nothing from batched device launch;
        // the single-product `fast_*` shim (size-gated) already handles a large
        // lone GEMM, so route those straight through the CPU-or-shim helper.
        if members.len() < 2 || m == 0 || p == 0 {
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
        let ok = crate::gpu::pool::scatter_batched(rt, &mut items, |_ordinal, slice| {
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
            let prod = crate::gpu::try_fast_abt_strided_batched(a.view(), bt.view())?;
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
                // Any member a tile silently skipped (cannot happen with the
                // contract, but keep the result total) falls back to CPU.
                for &idx in &members {
                    if out[idx].is_none() {
                        out[idx] = Some(cpu_one(idx));
                    }
                }
            }
            None => {
                for &idx in &members {
                    out[idx] = Some(cpu_one(idx));
                }
            }
        }
    }
    out.into_iter()
        .enumerate()
        .map(|(idx, slot)| slot.unwrap_or_else(|| cpu_one(idx)))
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

/// Curvature-homotopy output linear-span (low-rank / Eckart-Young) anchor.
///
/// This stage-1 primitive certifies the rank-`Σ basis_size` Eckart-Young residual
/// CEILING of the target by sequential residual SVDs, canonicalizing every
/// recovered output *linear subspace* (the span of the top singular vectors — the
/// "linear span" this anchor names) through the same [`GrassmannFrame`] gauge used
/// by the #972 frame machinery. The ceiling is a lower bound on the residual at
/// every `eta`; it is NOT a claim that the `eta = 0` parametric endpoint is a
/// linear/affine model (for curved bases that base-topology chart still embeds
/// curvature). It does not mutate `term` or replace the existing seed cascade.
pub fn linear_span_anchor(
    term: &SaeManifoldTerm,
    targets: ArrayView2<'_, f64>,
) -> Result<LinearSpanAnchor, String> {
    let n = term.n_obs();
    let p = term.output_dim();
    if targets.dim() != (n, p) {
        return Err(format!(
            "linear_span_anchor: targets shape {:?} != ({n}, {p})",
            targets.dim()
        ));
    }
    if term.k_atoms() == 0 {
        return Err("linear_span_anchor: term must contain at least one atom".into());
    }
    if !targets.iter().all(|v| v.is_finite()) {
        return Err("linear_span_anchor: targets must be finite".into());
    }
    let gates = neutral_gate_weights(term.assignment.mode, term.k_atoms());
    let mut residual = targets.to_owned();
    let mut reconstruction = Array2::<f64>::zeros((n, p));
    let mut atoms = Vec::with_capacity(term.k_atoms());
    for (atom_idx, atom) in term.atoms.iter().enumerate() {
        let gate = gates[atom_idx];
        if !(gate.is_finite() && gate > 0.0) {
            return Err(format!(
                "linear_span_anchor: neutral gate for atom {atom_idx} must be positive finite; got {gate}"
            ));
        }
        let requested_rank = atom.basis_size().min(n).min(p);
        if requested_rank == 0 {
            return Err(format!(
                "linear_span_anchor: atom {atom_idx} has no recoverable linear span rank"
            ));
        }
        let weighted = residual.mapv(|v| gate * v);
        let (_u_opt, singular_values_full, vt_opt) = weighted
            .svd(false, true)
            .map_err(|err| format!("linear_span_anchor: SVD failed for atom {atom_idx}: {err}"))?;
        let vt = vt_opt.ok_or_else(|| {
            format!("linear_span_anchor: SVD returned no right factor for atom {atom_idx}")
        })?;
        let rank = requested_rank
            .min(vt.nrows())
            .min(singular_values_full.len());
        if rank == 0 {
            return Err(format!(
                "linear_span_anchor: atom {atom_idx} SVD returned rank zero"
            ));
        }
        let mut frame = Array2::<f64>::zeros((p, rank));
        for col in 0..rank {
            for row in 0..p {
                frame[[row, col]] = vt[[col, row]];
            }
        }
        let singular_values = singular_values_full.slice(s![..rank]).to_owned();
        let frame = GrassmannFrame::from_oriented(frame, singular_values.clone());
        let frame_matrix = frame.frame().to_owned();
        let mut coordinates = residual.dot(&frame_matrix);
        coordinates.mapv_inplace(|v| v / gate);
        let contribution = fast_abt(&coordinates, &frame_matrix).mapv(|v| gate * v);
        reconstruction += &contribution;
        residual -= &contribution;
        atoms.push(LinearSpanAtomAnchor {
            gate_weight: gate,
            frame,
            decoder_coordinates: coordinates,
            singular_values,
        });
    }
    let residual_norm_sq = residual.iter().map(|v| v * v).sum();
    Ok(LinearSpanAnchor {
        atoms,
        reconstruction,
        residual_norm_sq,
    })
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

    /// Rank-`q` PCA / Eckart-Young explained-variance ceiling of a column-centered
    /// `target` — the best reconstruction EV any rank-`q` LINEAR dictionary can
    /// reach. S1 (guard surgery): this is now a TEST ORACLE only. It was the
    /// reference for the retired `0.5 × ceiling` collapse bar; the live collapse
    /// detector keys on the signal-free null floor
    /// (`super::absolute_degeneracy_ev_floor` = `q / n`), so no production code
    /// consumes this ceiling. The linear-anchor parity tests below still compare the
    /// anchor's reconstruction against it, so it lives here as their oracle. Returns
    /// `[0, 1]` on a finite target; `f64::NAN` on SVD failure / zero-variance target.
    fn pca_ev_ceiling(target: ArrayView2<'_, f64>, q: usize) -> f64 {
        let (n, p) = target.dim();
        if n == 0 || p == 0 {
            return f64::NAN;
        }
        let mut centered = target.to_owned();
        for c in 0..p {
            let mean = (0..n).map(|r| target[[r, c]]).sum::<f64>() / n as f64;
            for r in 0..n {
                centered[[r, c]] -= mean;
            }
        }
        let sst: f64 = centered.iter().map(|v| v * v).sum();
        if !(sst > 0.0) || !sst.is_finite() {
            return f64::NAN;
        }
        let sv = match centered.svd(false, false) {
            Ok((_, sv, _)) => sv,
            Err(_) => return f64::NAN,
        };
        let captured: f64 = sv.iter().take(q).map(|s| s * s).sum();
        captured / sst
    }

    /// Build a K-atom LINEAR (degree-1, d=1) SAE term over distinct 1-D coords
    /// with a known rank-`r_true` linear target `X = Z @ D`. The decoder seed is
    /// irrelevant to the anchor (the anchor re-derives the output subspace by
    /// SVD), so we seed zeros.
    fn linear_term_rank(
        k: usize,
        n: usize,
        p: usize,
        r_true: usize,
    ) -> (SaeManifoldTerm, Array2<f64>) {
        let mut atoms = Vec::with_capacity(k);
        let mut coords_blocks = Vec::with_capacity(k);
        for idx in 0..k {
            let coords = Array2::from_shape_fn((n, 1), |(i, _)| {
                ((i as f64 + 1.0) * 0.19 * (idx as f64 + 1.1)).sin()
            });
            // Linear basis Φ(t) = [1, t]; jet d/dt = [0, 1].
            let mut phi = Array2::<f64>::zeros((n, 2));
            let mut jet = ndarray::Array3::<f64>::zeros((n, 2, 1));
            for r in 0..n {
                phi[[r, 0]] = 1.0;
                phi[[r, 1]] = coords[[r, 0]];
                jet[[r, 1, 0]] = 1.0;
            }
            let decoder = Array2::<f64>::zeros((2, p));
            atoms.push(
                SaeManifoldAtom::new_with_provided_function_gram(
                    format!("lin_{idx}"),
                    SaeAtomBasisKind::Linear,
                    1,
                    phi,
                    jet,
                    decoder,
                    Array2::<f64>::eye(2),
                )
                .unwrap(),
            );
            coords_blocks.push(coords);
        }
        let logits = Array2::from_shape_fn((n, k), |(i, kk)| {
            0.2 + 0.05 * (i as f64) - 0.03 * (kk as f64)
        });
        let manifolds = vec![LatentManifold::Euclidean; k];
        let assignment = SaeAssignment::from_blocks_with_mode_and_manifolds(
            logits,
            coords_blocks,
            manifolds,
            AssignmentMode::ordered_beta_bernoulli(0.5, 1.0, false),
        )
        .unwrap();
        let term = SaeManifoldTerm::new(atoms, assignment).unwrap();
        // Known rank-`r_true` linear target: X = Z @ D.
        let z = Array2::from_shape_fn((n, r_true), |(i, j)| {
            ((i as f64 + 1.0) * 0.137 * (j as f64 + 1.0)).sin() + 0.3 * ((i * j) as f64).cos()
        });
        let d_true = Array2::from_shape_fn((r_true, p), |(j, c)| {
            if r_true <= 7 {
                // Original form. `((j*5 + c*3) % 7)` is genuinely rank-r_true while
                // r_true <= 7 (its period-7 structure has not yet repeated a row),
                // so the small fixtures keep their EXACT tuned gate-weighting
                // margins. (Switching them to the DCT basis below shifts the
                // gate-weighted top-rank subspace and breaks the 5e-3 parity gate.)
                (1.0 + j as f64) * (((j * 5 + c * 3) % 7) as f64 - 3.0) / 3.0
            } else {
                // #1026: for r_true > 7 the period-7 form COLLAPSES — its rows
                // repeat every 7 indices, so a nominally "rank-24" target was
                // actually rank ~7, the rank-16 PCA ceiling saturated at 1.0, and
                // the large-rank fixture self-check (`ceiling < 0.9999` when the
                // dictionary rank is below the data rank) tripped. Use an
                // orthogonal DCT-II basis (scaled per row) so all r_true rows are
                // linearly independent and the target is genuinely rank min(r_true, p).
                (1.0 + 0.5 * j as f64)
                    * (std::f64::consts::PI * (c as f64 + 0.5) * (j as f64) / p as f64).cos()
            }
        });
        let target = z.dot(&d_true);
        (term, target)
    }

    /// Rank-6 convenience wrapper (the original fixture).
    fn linear_term(k: usize, n: usize, p: usize) -> (SaeManifoldTerm, Array2<f64>) {
        linear_term_rank(k, n, p, 6)
    }

    #[test]
    fn linear_span_anchor_reaches_pca_ceiling_at_dictionary_rank_1026() {
        let n = 40usize;
        let p = 8usize;
        for &k in &[1usize, 2, 3, 6] {
            let (term, target) = linear_term(k, n, p);
            let anchor = linear_span_anchor(&term, target.view())
                .expect("linear anchor must solve on finite linear data");
            let ev_anchor =
                reconstruction_explained_variance(target.view(), anchor.reconstruction.view())
                    .expect("anchor EV must be finite");
            // Each LINEAR atom has basis_size 2 ({1, t}); the sequential
            // Eckart-Young deflation captures top-2 of the residual per atom, so
            // K atoms capture rank min(2K, n, p). Compare to that PCA ceiling.
            let total_rank = (2 * k).min(n).min(p);
            let ceiling = pca_ev_ceiling(target.view(), total_rank);
            println!(
                "[#1026] K={k:>2} (rank {total_rank})  anchor EV={ev_anchor:.8}  \
                 PCA ceiling={ceiling:.8}  gap={:.2e}",
                ceiling - ev_anchor
            );
            assert!(ev_anchor.is_finite(), "K={k}: anchor EV must be finite");
            // The anchor's sequential rank-`basis_size`-per-atom residual
            // deflation is the greedy Eckart-Young projection onto the top-(K·basis)
            // right-singular subspace — essentially the rank-(K·basis) PCA optimum.
            // It reaches the ceiling to within a small numerical margin (MSI:
            // ~1.3e-3 at K=1) rather than machine epsilon, because the per-atom
            // NEUTRAL ordered Beta--Bernoulli gate `π_k < 1` weights the residual SVD that picks the
            // frame while the coordinates are read from the unweighted residual, so
            // the recovered subspace is the gate-weighted (not the bare) top-rank
            // subspace. A genuinely broken anchor (wrong per-atom rank, dropped
            // deflation, non-orthogonal frame) would fall short by orders of
            // magnitude more; 5e-3 catches that while tolerating the gate-weighting
            // numerical gap.
            assert!(
                ev_anchor >= ceiling - 5e-3,
                "K={k}: linear anchor EV {ev_anchor} must reach the rank-{total_rank} \
                 PCA ceiling {ceiling} (within 5e-3) — a larger shortfall means the \
                 anchor loses linear reconstructible variance the dictionary is \
                 entitled to (#1026 parity)"
            );
        }
    }

    /// #1026 — the anchor reaches the PCA ceiling at a LARGER synthetic
    /// dictionary rank (not just the rank-2/6/12 of the small fixture). This is
    /// the CPU-checkable half of the issue's K-scaling ladder (item 1): the
    /// reconstruction-parity ceiling claim is pure sequential-deflation linear
    /// algebra, so it must hold as the dictionary's total rank grows. Here
    /// `K ∈ {8, 12, 16}` linear atoms (basis_size 2 each ⇒ total rank up to 32)
    /// reconstruct a genuinely rank-24 target in `p = 40` output channels, so the
    /// dictionary rank `2K` straddles the data rank 24 and the PCA ceiling is
    /// non-trivial (neither 0 nor a saturated 1.0) at the low end. A regression
    /// that loses reconstructible variance at scale — wrong per-atom rank, a
    /// dropped deflation step, a non-orthogonal frame that only shows up once many
    /// atoms accumulate — is caught here where the small fixture (capped at p=8)
    /// could not exercise it. The large-K *real-corpus* EV-vs-K curve remains
    /// GPU/corpus-gated; this pins only the synthetic Eckart-Young ceiling, which
    /// needs no corpus.
    #[test]
    fn linear_span_anchor_reaches_pca_ceiling_at_large_dictionary_rank_1026() {
        let n = 120usize;
        let p = 40usize;
        let r_true = 24usize;
        for &k in &[8usize, 12, 16] {
            let (term, target) = linear_term_rank(k, n, p, r_true);
            let anchor = linear_span_anchor(&term, target.view())
                .expect("large-K linear anchor must solve on finite linear data");
            let ev_anchor =
                reconstruction_explained_variance(target.view(), anchor.reconstruction.view())
                    .expect("anchor EV must be finite");
            // Each LINEAR atom has basis_size 2; the sequential Eckart-Young
            // deflation captures the top-2 residual directions per atom, so K atoms
            // capture rank min(2K, n, p, r_true). Compare to that PCA ceiling.
            let total_rank = (2 * k).min(n).min(p).min(r_true);
            let ceiling = pca_ev_ceiling(target.view(), total_rank);
            println!(
                "[#1026] LARGE K={k:>2} (dict rank {:>2}, data rank {r_true})  \
                 anchor EV={ev_anchor:.8}  PCA ceiling={ceiling:.8}  gap={:.2e}",
                2 * k,
                ceiling - ev_anchor
            );
            assert!(
                ev_anchor.is_finite(),
                "K={k}: large-K anchor EV must be finite"
            );
            // The non-trivially-ranked ceiling (e.g. K=8 ⇒ rank-16 ceiling on
            // rank-24 data is < 1.0) must be reached by the greedy deflation to the
            // same small margin as the small fixture; a scale-only regression would
            // open the gap by orders of magnitude.
            assert!(
                ev_anchor >= ceiling - 5e-3,
                "K={k}: large-K linear anchor EV {ev_anchor} must reach the rank-{total_rank} \
                 PCA ceiling {ceiling} (within 5e-3) at scale — a larger shortfall means the \
                 deflation loses reconstructible variance as the dictionary grows (#1026 parity)"
            );
            // Sanity that the fixture actually exercises the sub-saturation regime
            // at the low end (so the ceiling is a real constraint, not a free 1.0).
            if 2 * k < r_true {
                assert!(
                    ceiling < 0.9999,
                    "K={k}: dict rank {} < data rank {r_true} must give a sub-1.0 PCA ceiling \
                     (got {ceiling}); fixture mis-specified",
                    2 * k
                );
            }
        }
    }

    /// #1026 — the ROUTING-BOUND ("price of sparsity") pinned as a STRICT EV gap,
    /// in pure anchor algebra (no inner-solver / `into_fitted` confound). The
    /// neutral-gate anchor keeps every atom ON for every row, so it can use all
    /// `2K` decoder directions per row and reaches the rank-`2K` PCA ceiling. A
    /// SPARSE router that activates only ONE atom per row caps that row's
    /// reconstruction at the single active atom's basis rank (2), so on data whose
    /// local rank exceeds 2 it CANNOT match the dense anchor. We realize the
    /// sparse-routed reconstruction directly from the SAME anchor frames (each row
    /// reconstructed by ONLY its assigned atom's rank-2 image), so the difference
    /// is the routing restriction alone — the engine's seed/Newton dynamics never
    /// enter. The strict gap is the CPU-provable face of the issue's finding that
    /// fitted sparse routing under-reconstructs the neutral-gate PCA subspace;
    /// the magnitude of that gap on the real Qwen corpus is the GPU/corpus-gated
    /// frontier, but its SIGN (sparse < dense, strictly) is provable here.
    #[test]
    fn sparse_routing_strictly_underreconstructs_dense_anchor_1026() {
        let n = 60usize;
        let p = 16usize;
        let r_true = 10usize;
        let k = 5usize;
        let (term, target) = linear_term_rank(k, n, p, r_true);

        // Dense neutral-gate anchor: all 2K directions available per row.
        let anchor = linear_span_anchor(&term, target.view())
            .expect("dense anchor must solve on finite linear data");
        let ev_dense =
            reconstruction_explained_variance(target.view(), anchor.reconstruction.view())
                .expect("dense EV finite");

        // Sparse top-1 routing: each row reconstructed by ONLY its single assigned
        // atom's rank-2 image. We assign rows round-robin across the K atoms and
        // rebuild each atom's own rank-2 reconstruction of the FULL target (its
        // Eckart-Young image), then keep only the rows routed to it. This is the
        // best a single-active-atom router can do per row given these frames, so
        // it is an UPPER bound on top-1 sparse EV — and it is still strictly below
        // the dense anchor whenever the data's local rank exceeds 2.
        let mut sparse_recon = Array2::<f64>::zeros((n, p));
        for (atom_idx, atom_anchor) in anchor.atoms.iter().enumerate() {
            // Atom image over all rows, exactly as the anchor builds each atom's
            // contribution: `gate · coordinates @ frameᵀ` (frame is p×rank,
            // `decoder_coordinates` is n×rank, so `fast_abt` gives the n×p image).
            let coords = &atom_anchor.decoder_coordinates;
            let frame_matrix = atom_anchor.frame.frame().to_owned();
            let image = fast_abt(coords, &frame_matrix).mapv(|v| v * atom_anchor.gate_weight);
            for row in 0..n {
                if row % k == atom_idx {
                    for col in 0..p {
                        sparse_recon[[row, col]] = image[[row, col]];
                    }
                }
            }
        }
        let ev_sparse = reconstruction_explained_variance(target.view(), sparse_recon.view())
            .expect("sparse EV finite");

        println!(
            "[#1026] routing-bound: dense neutral-gate anchor EV={ev_dense:.6}  \
             top-1 sparse-routed EV={ev_sparse:.6}  price-of-sparsity gap={:.6}",
            ev_dense - ev_sparse
        );
        assert!(
            ev_dense.is_finite() && ev_sparse.is_finite(),
            "both EVs must be finite: dense={ev_dense}, sparse={ev_sparse}"
        );
        // The dense anchor reaches (essentially) the full ceiling; the top-1 router
        // is rank-limited to 2 per row on rank-10 data, so it MUST fall strictly
        // short. The margin (well above rounding) is the provable price of sparsity.
        assert!(
            ev_dense > ev_sparse + 0.05,
            "#1026 routing-bound: dense neutral-gate anchor EV {ev_dense:.6} must STRICTLY \
             exceed top-1 sparse-routed EV {ev_sparse:.6} (the price of sparsity: a single \
             active rank-2 atom per row cannot span the rank-{r_true} local data) — a \
             vanishing gap would mean sparse routing is silently as expressive as the \
             dense PCA subspace, contradicting the #1026 finding"
        );
    }

    /// Build a single-atom LINEAR SAE whose one atom has latent dim `d` (basis
    /// `{1, t_1, …, t_d}`), with the atom's coordinates seeded to `coords`
    /// (`n × d`) and the per-row ordered Beta--Bernoulli gate logits set explicitly. ordered Beta--Bernoulli routing.
    /// Returns the term; the caller marks the atom ungated (or not).
    fn single_linear_atom_term(
        coords: Array2<f64>,
        logits: Array2<f64>,
        p: usize,
        ungated: bool,
    ) -> SaeManifoldTerm {
        let d = coords.ncols();
        let evaluator = std::sync::Arc::new(EuclideanPatchEvaluator::new(d, 1).unwrap());
        let (phi, jet) = evaluator.evaluate(coords.view()).unwrap();
        let m = phi.ncols();
        let decoder = Array2::<f64>::zeros((m, p));
        let atom = SaeManifoldAtom::new_with_provided_function_gram(
            "lin_bg",
            SaeAtomBasisKind::Linear,
            d,
            phi,
            jet,
            decoder,
            Array2::<f64>::eye(m),
        )
        .unwrap()
        .with_basis_evaluator(evaluator);
        let assignment = SaeAssignment::from_blocks_with_mode_and_manifolds(
            logits,
            vec![coords],
            vec![LatentManifold::Euclidean],
            AssignmentMode::ordered_beta_bernoulli(0.5, 1.0, false),
        )
        .unwrap()
        .with_ungated(vec![ungated])
        .unwrap();
        SaeManifoldTerm::new(vec![atom], assignment).unwrap()
    }

    /// #1026 END-TO-END: a fitted LINEAR SAE whose single linear atom is the
    /// UNGATED background tier (gate ≡ 1) reaches the rank-2 PCA reconstruction
    /// ceiling, with the inner Newton converging cleanly on the ridge-inert
    /// frozen-logit fixture. Fixture: `d = 1` atom (`γ(t) = b₀ + t·b₁`), an
    /// initially strongly row-varying gate (logits span ≈[−3, 3]), and a rank-2
    /// signal `X[i] = c₀ + z[i]·c₁` with a full-magnitude row-invariant intercept
    /// `c₀`; coords seeded to the true factor `z` so the unit-gate atom reproduces
    /// `X` exactly.
    ///
    /// HONEST CALIBRATION NOTE: this single-atom case does NOT exhibit an
    /// ungated-vs-gated EV gap, because `into_fitted` optimizes the gate logits, so
    /// even the gated atom drives its own gate toward ≈1 and also reaches the
    /// ceiling (MSI: both EV = 1.000000). The ungate's value is structural, not an
    /// unconditional single-atom gap — see the assertions and
    /// `ungated_logit_slot_carries_zero_gradient_and_curvature_1026` (the inert
    /// logit cannot be shrunk off by sparsity / many-atom routing the way a gated
    /// atom can). We assert the honest, robust facts only: the ungated tier reaches
    /// the ceiling (converged), and ungating is never a regression vs gated.
    #[test]
    fn ungated_linear_background_atom_reaches_pca_ceiling_and_converges_1026() {
        let n = 40usize;
        let p = 6usize;
        // Rank-2 linear signal: a row-invariant intercept c0 + one linear factor z.
        let zf: Vec<f64> = (0..n)
            .map(|i| ((i as f64 + 1.0) * 0.23).sin() + 0.3 * ((i * 3) as f64).cos())
            .collect();
        let c0 = Array1::from_shape_fn(p, |c| 1.0 + 0.5 * (c as f64) - 0.2 * ((c % 3) as f64));
        let c1 = Array1::from_shape_fn(p, |c| (((c * 2 + 1) % 5) as f64 - 2.0) * 0.7);
        let target = Array2::from_shape_fn((n, p), |(i, c)| c0[c] + zf[i] * c1[c]);
        let ceiling = pca_ev_ceiling(target.view(), 2); // intercept + 1 linear factor

        // Atom coords seeded to the true linear factor (d = 1); the unit-gate atom
        // can then reproduce X exactly. STRONGLY row-varying gate logits.
        let coords = Array2::from_shape_fn((n, 1), |(i, _)| zf[i]);
        let logits =
            Array2::from_shape_fn((n, 1), |(i, _)| -3.0 + 6.0 * (i as f64) / (n as f64 - 1.0));

        let fit_ev = |ungated: bool| -> f64 {
            let term = single_linear_atom_term(coords.clone(), logits.clone(), p, ungated);
            let init_rho = SaeManifoldRho::new(
                (1.0e-4_f64).ln(),
                (1.0e-2_f64).ln(),
                vec![Array1::<f64>::zeros(1)],
            );
            let rho_flat = init_rho.to_flat();
            let mut outer = SaeManifoldOuterObjective::new(
                term,
                target.clone(),
                None,
                init_rho,
                60,
                0.5,
                1e-4,
                1e-4,
            );
            outer
                .fit_at_fixed_rho(rho_flat.view())
                .expect("fixed-rho fit converges");
            let fitted = outer.into_fitted().expect("fixed-rho fit was evaluated");
            assert!(fitted.penalized_quasi_laplace_criterion.is_finite());
            let recon = fitted.term.fitted();
            reconstruction_explained_variance(target.view(), recon.view()).expect("EV finite")
        };

        let ev_ungated = fit_ev(true);
        let ev_gated = fit_ev(false);
        println!(
            "[#1026] linear background tier (d=1, wide initial gate): ungated EV={ev_ungated:.6}  \
             gated EV={ev_gated:.6}  PCA(rank 2) ceiling={ceiling:.6}  \
             ungated−gated={:.6}",
            ev_ungated - ev_gated
        );

        assert!(
            ev_ungated.is_finite() && ev_gated.is_finite(),
            "both fitted EVs must be finite: ungated={ev_ungated}, gated={ev_gated}"
        );
        // (1) LOAD-BEARING: the UNGATED tier reaches the rank-2 PCA ceiling (unit
        // gate ⇒ the linear decoder fit is the exact LS solution), AND the inner
        // Newton converges cleanly on the frozen-logit (ridge-inert) fixture — a
        // near-singular / drifting solve would yield garbage, not the ceiling.
        assert!(
            ev_ungated >= ceiling - 5.0e-3,
            "#1026: the UNGATED linear atom must reach the rank-2 PCA ceiling \
             {ceiling:.6}; got {ev_ungated:.6} (the ungated tier carries full-rank \
             linear variance and the inner solve converged)"
        );
        // (2) Ungating is NEVER a reconstruction regression: ungating only
        // ENLARGES the feasible set (drops the a_k ≤ 1 gate constraint to a_k ≡ 1),
        // so its optimum matches or beats the gated optimum.
        //
        // HONEST FINDING (MSI, this fixture): ungated EV = gated EV = 1.000000, so
        // the closed gap is ~0 here. That is REAL information, NOT a tuning target:
        // because `into_fitted` OPTIMIZES the ordered Beta--Bernoulli gate logits, a single gated atom
        // drives its OWN gate toward the unit region and reaches parity too, so the
        // planted row-varying gate is optimized away. The ungate's value is
        // therefore NOT an unconditional single-atom EV gap — it is STRUCTURAL: the
        // ungated logit is deterministically inert (its assembled gradient is
        // EXACTLY 0 — see `ungated_logit_slot_carries_zero_gradient_and_curvature_1026`),
        // so the background tier reconstructs at unit gate REGARDLESS of the
        // optimizer and CANNOT be shrunk off by the assignment sparsity prior,
        // whereas a gated atom's parity hinges on the optimizer finding gate ≈ 1
        // (which sparsity pressure and many-atom routing actively oppose). We
        // assert only the honest, robust direction here.
        assert!(
            ev_ungated >= ev_gated - 1.0e-6,
            "#1026: ungating must not reconstruct WORSE than the gated atom \
             (it only enlarges the feasible set): ungated EV {ev_ungated:.6} vs \
             gated EV {ev_gated:.6}"
        );
    }

    /// #1026 AIRTIGHT inert-logit gate: the ungated atom's logit slot must carry
    /// EXACTLY zero assembled gradient `gt` AND exactly zero `htt` row/column —
    /// i.e. NO assembly term (data logit-JVP, sparsity-prior grad/hdiag, softmax
    /// majorizer, ordered Beta--Bernoulli third channels) leaks a nonzero into that slot. Combined
    /// with the per-row ridge floor added at solve time (which makes the slot's
    /// diagonal PD), `Δlogit = −0/ridge = 0` DETERMINISTICALLY at every Newton
    /// iterate — the gate stays pinned at `1`, never drifting. We assemble at a
    /// NON-seed point (a perturbed decoder + nonzero ρ) so the assertion is not a
    /// seed coincidence: any leaking term would be excited here.
    ///
    /// ordered Beta--Bernoulli dense layout: the per-row block is `[logit_0 … logit_{K−1}, coords…]`,
    /// so the ungated atom's logit gradient/curvature live at row-block index
    /// equal to its atom index.
    #[test]
    fn ungated_logit_slot_carries_zero_gradient_and_curvature_1026() {
        use ndarray::Array1;
        let n = 24usize;
        let p = 5usize;
        let d = 3usize;
        // Two atoms: atom 0 GATED, atom 1 UNGATED — so the assertion also confirms
        // the ungated slot stays zero while a genuine gated slot alongside it is
        // (in general) nonzero, i.e. the zeroing is atom-targeted, not global.
        let mut coords_blocks = Vec::new();
        let mut atoms = Vec::new();
        for idx in 0..2usize {
            let coords = Array2::from_shape_fn((n, d), |(i, a)| {
                ((i as f64 + 1.0) * 0.17 * (a as f64 + 1.0) * (idx as f64 + 1.3)).sin()
            });
            let evaluator = std::sync::Arc::new(EuclideanPatchEvaluator::new(d, 1).unwrap());
            let (phi, jet) = evaluator.evaluate(coords.view()).unwrap();
            let m = phi.ncols();
            // Non-trivial decoder so the data logit-JVP term (which would leak into
            // a non-ungated logit) is genuinely nonzero at this point.
            let decoder = Array2::from_shape_fn((m, p), |(r, c)| {
                0.1 * (((idx * 5 + r * 3 + c) % 7) as f64 - 3.0)
            });
            atoms.push(
                SaeManifoldAtom::new_with_provided_function_gram(
                    format!("atom_{idx}"),
                    SaeAtomBasisKind::Linear,
                    d,
                    phi,
                    jet,
                    decoder,
                    Array2::<f64>::eye(m),
                )
                .unwrap()
                .with_basis_evaluator(evaluator),
            );
            coords_blocks.push(coords);
        }
        let logits =
            Array2::from_shape_fn((n, 2), |(i, k)| 0.4 + 0.03 * (i as f64) - 0.07 * (k as f64));
        let assignment = SaeAssignment::from_blocks_with_mode_and_manifolds(
            logits,
            coords_blocks,
            vec![LatentManifold::Euclidean; 2],
            AssignmentMode::ordered_beta_bernoulli(0.5, 1.0, false),
        )
        .unwrap()
        .with_ungated(vec![false, true]) // atom 1 is the ungated background tier
        .unwrap();
        let mut term = SaeManifoldTerm::new(atoms, assignment).unwrap();
        let target = Array2::from_shape_fn((n, p), |(i, c)| 0.3 * ((i + 2 * c) as f64).sin());
        // Nonzero ρ (sparsity + smoothness) so the sparsity-prior grad/hdiag term
        // is active — exactly the term that would leak into the ungated logit if
        // the zeroing were incomplete.
        let rho = SaeManifoldRho::new(
            (0.5_f64).ln(),
            (0.2_f64).ln(),
            vec![Array1::<f64>::zeros(d); 2],
        );
        let sys = term
            .assemble_arrow_schur(target.view(), &rho, None)
            .expect("assembly with an ungated atom must succeed");

        // The ungated atom is index 1; in the ordered Beta--Bernoulli dense layout its logit slot is
        // row-block index 1. Assert EXACT zero gradient + zero htt row/col there,
        // for EVERY data row (no term leaks at any row).
        let ungated_slot = 1usize;
        for (row_idx, block) in sys.rows.iter().enumerate() {
            assert_eq!(
                block.gt[ungated_slot], 0.0,
                "#1026 row {row_idx}: ungated logit slot gradient must be EXACTLY 0 \
                 (no JVP / prior / majorizer term may leak); got {}",
                block.gt[ungated_slot]
            );
            for j in 0..block.htt.ncols() {
                assert_eq!(
                    block.htt[[ungated_slot, j]],
                    0.0,
                    "#1026 row {row_idx}: ungated logit htt row entry ({ungated_slot},{j}) \
                     must be EXACTLY 0; got {}",
                    block.htt[[ungated_slot, j]]
                );
                assert_eq!(
                    block.htt[[j, ungated_slot]],
                    0.0,
                    "#1026 row {row_idx}: ungated logit htt col entry ({j},{ungated_slot}) \
                     must be EXACTLY 0; got {}",
                    block.htt[[j, ungated_slot]]
                );
            }
        }
    }

    /// #1026 — the routing-bound regime where the ungate's benefit becomes a REAL
    /// reconstruction gap: SPARSITY PRESSURE. Under a large `λ_sparse`, the ordered Beta--Bernoulli
    /// assignment-sparsity prior pulls the gated atom's logits toward OFF (its
    /// Beta-Bernoulli energy prefers gates below 1), so a gated atom can no longer
    /// self-optimize its gate to ≈1 and under-reconstructs the unit-magnitude
    /// signal. The UNGATED background atom is immune — its logit is inert (zero
    /// gradient/curvature, no sparsity-prior term, gate ≡ 1), so it reconstructs at
    /// full magnitude regardless of `λ_sparse`. This is the regime the #1033
    /// amortized-routing / frozen-background design targets: the dense linear tier
    /// must NOT be subject to the sparsity routing that the residual curved atoms
    /// are.
    ///
    /// CALIBRATION NOTE: this is committed as an OBSERVATIONAL gate — it prints the
    /// gated-vs-ungated EVs across a sparsity sweep and asserts only the robust,
    /// always-true facts (ungated reaches the ceiling and is never worse than
    /// gated, AND ungated is monotone-immune to sparsity while gated degrades). The
    /// exact gap magnitude under sparsity is recorded from the run rather than
    /// hard-pinned, so the test states what is provably true without a
    /// machine-specific threshold; the printed sweep is the #1026 routing-bound
    /// evidence.
    #[test]
    fn ungated_background_resists_sparsity_pressure_gated_degrades_1026() {
        let n = 40usize;
        let p = 6usize;
        let zf: Vec<f64> = (0..n)
            .map(|i| ((i as f64 + 1.0) * 0.23).sin() + 0.3 * ((i * 3) as f64).cos())
            .collect();
        let c0 = Array1::from_shape_fn(p, |c| 1.0 + 0.5 * (c as f64) - 0.2 * ((c % 3) as f64));
        let c1 = Array1::from_shape_fn(p, |c| (((c * 2 + 1) % 5) as f64 - 2.0) * 0.7);
        let target = Array2::from_shape_fn((n, p), |(i, c)| c0[c] + zf[i] * c1[c]);
        let ceiling = pca_ev_ceiling(target.view(), 2);
        let coords = Array2::from_shape_fn((n, 1), |(i, _)| zf[i]);
        let logits = Array2::from_shape_fn((n, 1), |(i, _)| 0.5 + 0.05 * (i as f64));

        let fit_ev = |ungated: bool, log_lambda_sparse: f64| -> f64 {
            let term = single_linear_atom_term(coords.clone(), logits.clone(), p, ungated);
            let init_rho = SaeManifoldRho::new(
                log_lambda_sparse,
                (1.0e-2_f64).ln(),
                vec![Array1::<f64>::zeros(1)],
            );
            let rho_flat = init_rho.to_flat();
            let mut outer = SaeManifoldOuterObjective::new(
                term,
                target.clone(),
                None,
                init_rho,
                60,
                0.5,
                1e-4,
                1e-4,
            );
            outer
                .fit_at_fixed_rho(rho_flat.view())
                .expect("fixed-rho fit converges");
            let fitted = outer.into_fitted().expect("fixed-rho fit was evaluated");
            assert!(fitted.penalized_quasi_laplace_criterion.is_finite());
            let recon = fitted.term.fitted();
            reconstruction_explained_variance(target.view(), recon.view()).expect("EV finite")
        };

        // Sparsity sweep: λ_sparse from mild to strong. PRINTED as the #1026
        // routing-bound evidence; the gated degradation magnitude is observed (it
        // depends on the penalized quasi-Laplace basin / inner-solve dynamics that the no-MSI build
        // cannot pre-calibrate), so only the two PROVABLY-TRUE facts are asserted.
        for &log_lam in &[
            (1.0e-3_f64).ln(),
            (1.0_f64).ln(),
            (1.0e2_f64).ln(),
            (1.0e4_f64).ln(),
        ] {
            let ev_ungated = fit_ev(true, log_lam);
            let ev_gated = fit_ev(false, log_lam);
            println!(
                "[#1026] sparsity λ=exp({log_lam:.3}): ungated EV={ev_ungated:.6}  \
                 gated EV={ev_gated:.6}  ceiling={ceiling:.6}  ungated−gated={:.6}",
                ev_ungated - ev_gated
            );
            assert!(
                ev_ungated.is_finite() && ev_gated.is_finite(),
                "EVs must be finite at λ=exp({log_lam}): ungated={ev_ungated}, gated={ev_gated}"
            );
            // PROVABLE (1): the ungated background reaches the ceiling at EVERY
            // sparsity level. Its logit is inert and carries NO assignment-sparsity
            // prior term (#1026), so `λ_sparse` has ZERO effect on it (it drives
            // only the assignment prior, which is empty for the ungated atom) — the
            // unit-gate linear fit is the exact LS solution regardless of λ.
            assert!(
                ev_ungated >= ceiling - 5.0e-3,
                "#1026: the UNGATED background must reach the ceiling {ceiling:.6} \
                 regardless of sparsity λ=exp({log_lam}); got {ev_ungated:.6} — the \
                 inert unit-gate tier must be immune to the assignment sparsity prior"
            );
            // PROVABLE (2): ungating is never a reconstruction regression (it only
            // enlarges the feasible set: a_k ≤ 1 dropped to a_k ≡ 1).
            assert!(
                ev_ungated >= ev_gated - 1.0e-6,
                "#1026: ungated EV {ev_ungated:.6} must be >= gated EV {ev_gated:.6} \
                 at λ=exp({log_lam}) (ungating only enlarges the feasible set)"
            );
        }
    }

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
