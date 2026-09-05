use super::*;
use opt::{BacktrackConfig, backtracking_line_search};

pub(crate) struct OuterFirstOrderBridge<'a> {
    pub(crate) obj: &'a mut dyn OuterObjective,
    pub(crate) layout: OuterThetaLayout,
    /// Outer-aware inner-PIRLS cap atomic. When `Some`, the bridge stores
    /// a coarsen-then-tighten cap into it on every accepted gradient eval
    /// (see `first_order_inner_cap_schedule`).
    ///
    /// The cap is a perf optimization for the GRADIENT inner solve only: at
    /// the accepted ρ the warm-start is excellent, so a small cap converges
    /// the inner Newton and a still-non-converged result is honestly rejected
    /// as infeasible. But the line-search COST probe (`eval_cost`) evaluates a
    /// DIFFERENT trial ρ whose warm-start is worse; the same small cap can stop
    /// the inner solve short of its fixed point, returning a non-converged
    /// `f64::INFINITY` cost for a point that is actually feasible. With every
    /// trial step then reporting `∞`, no Wolfe/ARC step satisfies descent, the
    /// optimizer never leaves the accepted ρ, and the gradient re-evaluated
    /// there is identical iter after iter — the frozen-|g| outer stall in
    /// gam#787 (bernoulli matern marginal-slope) and gam#808 (survival
    /// marginal-slope). The line-search cost MUST be the same converged-inner
    /// objective the analytic envelope gradient differentiates; a capped
    /// surrogate is a different objective. So `eval_cost` UNCAPS the inner solve
    /// (stores `0` = full `pirls_config.max_iterations`) before delegating, and
    /// `eval_grad`/`eval_hessian` restore the scheduled cap on the next call.
    pub(crate) outer_inner_cap: Option<InnerProgressFeedback>,
    /// Counts gradient EVALUATIONS, which is not the same thing as outer
    /// iterations and is why this is no longer called `iter_count` (#2613): the
    /// Strong-Wolfe search evaluates the gradient at every trial that clears
    /// Armijo, so on #2392's recovery this reached 18 while `opt::Bfgs` had
    /// completed exactly ZERO iterations — and the refusal message published
    /// "after 18 outer iteration(s)". Accepted outer steps are counted by
    /// `CostStallGuard::accepted_iters`, fed from `on_step_accepted`; the
    /// inner-PIRLS schedule reads `InnerProgressFeedback.accepted_iter` from
    /// the same signal. This counter is for logging and for the
    /// "no gradient evaluation has ever succeeded on this seed" probe-refusal
    /// gate, both of which want evaluations.
    pub(crate) first_order_evals: usize,
    /// First observed `‖g‖` from `eval_grad`. Used by the schedule to
    /// compute the gradient-ratio (`last / initial`) — when the ratio
    /// drops, the optimizer is approaching convergence and the inner
    /// cap should lift to full so the cached β is at full tolerance.
    pub(crate) g_norm_initial: Option<f64>,
    /// `‖g‖` from the most recent eval. Stale by one outer iter relative
    /// to the cap that consumes it (the cap is set BEFORE the new eval),
    /// but for monotone-decreasing g_norm this is safe — it makes the
    /// cap conservatively LARGER than the truly-needed value, never
    /// smaller.
    pub(crate) last_g_norm: Option<f64>,
    /// Most recent derivative-evaluation point. Value-only line-search probes
    /// log their distance from this reference so hidden backtracking work is
    /// visible in STAGE traces.
    pub(crate) last_value_grad_rho: Option<Array1<f64>>,
    /// Exact memo for recent line-search value probes. BFGS can re-query the
    /// same rejected trial when switching Wolfe strategies; the SAE inner solve
    /// behind a Value probe is deterministic, so serving an identical rho from
    /// this memo preserves the objective while avoiding duplicate refinement
    /// work.
    pub(crate) value_probe_cache: Vec<ValueProbeCacheEntry>,
    /// Gradient-independent cost-stall convergence guard. `opt::Bfgs` only
    /// terminates on a small *projected gradient norm* (its stall exit ANDs
    /// gradient-smallness with cost-smallness), so on a fully-penalized
    /// (double-penalty) REML surface with a shallow, weakly-identified valley —
    /// where the REML score flatlines while `‖∇_ρ V‖` plateaus *above*
    /// tolerance — no opt-side exit ever fires and BFGS burns its entire
    /// `max_iterations` budget (each iteration spending many line-search +
    /// coordinate-rescue + jiggle probes) on every seed. That is the #1089
    /// pathology: a trivial n≈30..120 Gaussian fit emitting ~850k cost-only
    /// evaluations until a wall-clock budget kills it. This guard adds the
    /// missing mgcv-style score-change stop: it watches the accepted-iterate
    /// REML objective and, once it stops improving by more than a relative
    /// tolerance over a window of consecutive accepted outer steps, publishes
    /// the best-so-far iterate and signals BFGS to stop. The runner then
    /// classifies the run as *converged at the flat-valley floor* rather than
    /// non-converged — the remaining gradient lies along weakly-identified ρ
    /// directions that do not reduce the objective.
    pub(crate) cost_stall: Option<CostStallGuard>,
    /// Box bounds `(lower, upper)` on the outer parameter vector, used ONLY to
    /// form the bound-PROJECTED gradient norm the [`CostStallGuard`] stationarity
    /// test consumes. The cost-stall guard's documented contract is to certify a
    /// stall as converged iff the *projected* gradient at the best iterate clears
    /// the outer tolerance — the same KKT criterion `opt`'s primary exit uses
    /// (`GradientTolerance{ projected: true }`). Without the bounds here the guard
    /// fell back to the RAW gradient norm, which on a separation fit never clears
    /// the tolerance (the bound-active log-λ directions carry a persistent
    /// ∂V/∂ρ pushing further out of bounds), so a genuinely stationary
    /// bound-pinned optimum was reported NON-converged and the outer search
    /// re-seeded forever (#1082 separable-multinomial / penalized-tensor
    /// cycling). `None` ⇒ unconstrained, no projection (raw norm). Cheap to hold
    /// (the outer dimension is the smoothing-param count).
    pub(crate) cost_stall_bounds: Option<(Array1<f64>, Array1<f64>)>,
    /// Count of consecutive `eval_cost` calls that returned `Recoverable`
    /// without a single success in between. When every trial step in every
    /// search direction is infeasible (the inner solve refuses to converge at
    /// any neighboring ρ), BFGS would otherwise spend its full
    /// `max_iterations × line_search_budget` budget doing inner solves that
    /// all fail — the non-termination reported in issue #NaN-outer-loop.
    ///
    /// Once this counter exceeds [`PROBE_REFUSAL_FATAL_THRESHOLD`] and no
    /// gradient evaluation has ever been accepted on this seed (`first_order_evals ==
    /// 0`), the bridge escalates to `Fatal` so BFGS exits immediately via
    /// `ObjectiveFailed`. The seed loop treats that outcome as a rejected seed
    /// and moves on, keeping the cascade bounded.
    ///
    /// Reset to 0 on any successful cost evaluation so normal line-search
    /// noise (a few recoverable probes followed by an accepted step) never
    /// trips this guard.
    pub(crate) consecutive_probe_refusals: usize,
    /// Accepted-outer-step signal published by [`OuterAcceptObserver`] (#2613).
    ///
    /// The cost-stall guard above counts *accepted outer steps*. Before #2613
    /// the bridge folded every `eval_grad` call into it, on the premise —
    /// written into the fold site's own comment — that `opt::Bfgs` calls
    /// `eval_grad` only at accepted iterates and routes line-search probes
    /// through `eval_cost`. That premise is false: the Strong-Wolfe search
    /// evaluates the GRADIENT at every trial that clears Armijo, because the
    /// curvature condition `|g(α)ᵀd| ≤ c₂|g(0)ᵀd|` needs it. A zoom that
    /// bisects toward a point therefore feeds the guard a run of iterates whose
    /// costs differ negligibly — of course they do, they are converging — and
    /// the guard reads its own window of "6 consecutive accepted steps with no
    /// improvement" as a flat valley and halts the solver *inside* one
    /// iteration, before `opt`'s own rescue ladder (global-best salvage, then
    /// trust-region dogleg) gets a turn.
    ///
    /// `opt::OptimizerObserver`'s doc names this exact hazard — "which
    /// conflates trial-eval probes with real outer steps" — and gam already
    /// used `on_step_accepted` to drive the inner-PIRLS cap. This wires the
    /// same signal to the guard, which is the place it is load-bearing.
    ///
    /// `None` leaves the pre-#2613 fold-every-eval behaviour, which is what the
    /// routes without a cost-stall guard want anyway (they never fold).
    pub(crate) accepted_steps: Option<Arc<AcceptedStepLedger>>,
    /// First-order evaluations made since the last accepted step, oldest first.
    /// Drained by [`Self::drain_accepted_steps`]. Empty whenever
    /// `accepted_steps` is `None`.
    pub(crate) pending_first_order: Vec<PendingOuterEval>,
    /// `(ρ, cost)` of the last iterate known to be accepted — the seed, then
    /// each accepted step. The reference point for reconciling
    /// [`AcceptedOuterStep`] against [`Self::pending_first_order`].
    pub(crate) incumbent: Option<(Array1<f64>, f64)>,
}

pub(crate) const VALUE_PROBE_CACHE_CAPACITY: usize = 256;

pub(crate) const VALUE_PROBE_REJECT_COST_FLOOR: f64 = 1.0e11;

/// Number of consecutive recoverable `eval_cost` failures (every line-search
/// probe infeasible) before the bridge escalates to `Fatal` and forces an
/// immediate BFGS exit. This guard fires only before the first accepted
/// gradient step (`first_order_evals == 0`): once BFGS has accepted at least one
/// outer iteration the current ρ is feasible and isolated probe refusals are
/// normal line-search noise, not a stuck loop.
///
/// The threshold covers one full StrongWolfe attempt (up to 20 probes)
/// plus one backtracking fallback (up to 50 probes) with a small margin,
/// so a SINGLE failed direction does not fire the guard. Two consecutive
/// direction failures (120 probes) always does — once both Wolfe and
/// backtracking exhausted two complete directions with no success, the
/// neighborhood is globally infeasible and further BFGS iterations are
/// pure waste.
pub(crate) const PROBE_REFUSAL_FATAL_THRESHOLD: usize = 150;

/// Tighter probe-refusal threshold used when the bridge has never seen a
/// `eval_grad` call of its own — i.e. the seed (cost, gradient) was supplied
/// via `with_initial_sample` so `last_value_grad_rho` is `None` and every
/// `trial_rho_distance` prints as NaN.  In this case the seed gradient is
/// already confirmed feasible externally; if even the first line-search
/// direction exhausts its Wolfe probes without success (≈ 20 probes), the
/// neighborhood IS globally infeasible and further iterations just repeat
/// the same expensive inner solve 150 more times.  One generous Wolfe
/// budget (25 probes) is enough to confirm the failure; 13 seeds ×
/// 150 probes × ~3 s each would otherwise cause an observed ~97 min hang.
pub(crate) const PROBE_REFUSAL_FATAL_THRESHOLD_NAN_SEED: usize = 25;

/// Sentinel prefix embedded in the fatal [`ObjectiveEvalError`] message the
/// bridge returns when [`PROBE_REFUSAL_FATAL_THRESHOLD`] fires. The seed-loop
/// runner matches this prefix and routes the failed seed to
/// typed [`SeedRejection`] accounting rather than propagating a fatal error.
pub(crate) const PROBE_REFUSAL_FATAL_SENTINEL: &str = "OUTER_PROBE_REFUSAL_FATAL";

/// Sentinel embedded in the fatal [`ObjectiveEvalError`] message the bridge
/// returns when [`CostStallGuard`] halts BFGS on a cost stall. `opt::Bfgs`
/// preserves the message verbatim in [`BfgsError::ObjectiveFailed`]; the
/// seed-loop runner recognizes this sentinel and rebuilds an outer result from
/// the published best iterate. Whether that result is reported `converged` is
/// NOT decided here — it is carried on the published [`CostStallExit`], gated on
/// the projected gradient norm at the best iterate clearing the same outer
/// gradient tolerance the genuine convergence path uses. A cost stall whose
/// residual gradient still exceeds that tolerance is a flat-valley stall, not a
/// stationary optimum, and is reported `converged = false`.
pub(crate) const COST_STALL_CONVERGED_SENTINEL: &str = "OUTER_COST_STALL_CONVERGED";

/// Sentinel used only when ARC has no finite current sample to hand to its
/// second-order convergence gate. A consecutive run of infeasible probes may
/// stop the trajectory at its best feasible checkpoint, but it can never
/// certify that checkpoint as converged: the bridge does not have a synchronized
/// Hessian there. The runner therefore always maps this sentinel to a
/// non-converged result.
pub(crate) const ARC_INFEASIBLE_STALL_SENTINEL: &str = "OUTER_ARC_INFEASIBLE_STALL";

/// Sentinel returned when the dense-ARC route reaches a point its own terminal
/// certificate would accept, so the search stops there instead of grinding on
/// against a threshold nothing downstream applies (#2817).
///
/// A REML search and the certificate that judges it must be ONE standard.
/// `opt::Arc` stops on an absolute projected-gradient band; the certificate
/// accepts on the Newton decrement `½·gᵀH⁻¹g` against the criterion's own
/// resolution. Those are different tests in different units, and on a flat REML
/// valley the band is the far stricter one: measured on a gaussian n=50 000,
/// p=93, K=11 fit the band was `7.451e-4` while the certificate accepted at
/// `2.173e-2` — 29× wider — so no seed could ever stop itself and all six runs
/// burned their 200-iteration budget for a last-100 improvement of `4e-4` in a
/// criterion of `5.3e4`. The matrix-free route was given the decrement stop in
/// `a85b88535` (`MatrixFreeTrustRegion::with_model_decrement_tolerance`); the
/// dense route, which is the one a low-dimensional ρ⊕η spatial or multinomial
/// fit takes, had no equivalent and kept the defect.
///
/// The runner maps this sentinel to a CONVERGED result. Unlike
/// [`ARC_INFEASIBLE_STALL_SENTINEL`] the bridge holds a synchronized analytic
/// Hessian at this exact point and has evaluated the certificate's own rung on
/// it; and the mandatory final analytic certificate re-derives its verdict from
/// a fresh evaluation regardless, so this claims a STOP, never an exemption.
pub(crate) const ARC_CURVATURE_STATIONARY_SENTINEL: &str = "OUTER_ARC_CURVATURE_STATIONARY";

/// Verdict produced by folding one accepted outer iterate into
/// [`CostStallGuard::observe`].
pub(crate) enum CostStallVerdict {
    /// The objective is still improving (or the no-improvement window has not
    /// yet filled). Keep descending.
    Continue,
    /// The objective has stopped improving over the window AND the projected
    /// gradient norm at the best iterate clears the outer gradient tolerance:
    /// a genuine stationary optimum on a (legitimately) flat REML surface.
    Converged,
    /// The objective has stopped improving over the window but the projected
    /// gradient norm at the best iterate is still above the outer gradient
    /// tolerance: a weakly-identified flat-valley FLOOR with residual
    /// non-stationarity. Halting here is correct (no further cost progress is
    /// available), but the iterate is NOT a stationary optimum and must be
    /// reported `converged = false`.
    FlatValleyStall { residual_grad_norm: f64 },
    /// The objective has stopped improving over the window but the projected
    /// gradient at the best iterate is FAR above the certified-stationary band
    /// (`> escape_threshold` = the score-relative stationarity bound times the
    /// escape margin, capped at `FLAT_VALLEY_STALL_GRAD_CEILING`; the carried
    /// `escape_threshold` is the value actually compared). A genuine flat-valley floor is, by
    /// definition, flat: its residual gradient is at most modestly above the
    /// convergence tolerance. A residual orders of magnitude above tolerance is
    /// NOT a flat valley — it is a *stuck* stall, the signature of an
    /// inconsistent objective/gradient pair (the inner PIRLS did not converge at
    /// this ρ, so the cached cost and the analytic gradient disagree and no
    /// line-search direction can make cost progress even though the surface is
    /// steep). Halting here would ship a near-unpenalized overfit (#1426). The
    /// guard instead resets the no-improvement window so the optimizer keeps
    /// descending (the inner solve runs to tighter tolerance at the next
    /// iterate, restoring a trustworthy gradient), for a bounded number of
    /// escapes before falling back to a `FlatValleyStall` halt so a genuinely
    /// pathological surface still terminates.
    StuckKeepDescending {
        residual_grad_norm: f64,
        /// The score-relative keep-descending trigger the residual actually
        /// exceeded (NOT the legacy fixed ceiling) — logged so the message can
        /// never contradict its own numbers.
        escape_threshold: f64,
    },
}

/// Number of consecutive accepted outer iterates with negligible relative
/// objective improvement required before the cost-stall guard declares
/// convergence. Matches the spirit of `opt`'s own `StallPolicy { window: 3 }`
/// but, crucially, is gated on the cost alone (not on gradient smallness),
/// which is the condition `opt` never checks in isolation.
pub(crate) const COST_STALL_WINDOW: usize = 6;
pub(crate) const ARC_COST_STALL_WINDOW: usize = 3;
pub(crate) const COST_STALL_REL_TOL_FLOOR: f64 = 1.0e-7;
pub(crate) const COST_STALL_PROJECTED_GRAD_FLOOR: f64 = 1.0e-3;

/// Absolute ceiling on the best-iterate projected gradient norm for a cost
/// stall to be classified as a genuine flat-valley FLOOR (#1426). A flat valley
/// is, by definition, flat: the REML surface has plateaued AND its residual
/// gradient is at most modestly above the outer convergence tolerance
/// (`COST_STALL_PROJECTED_GRAD_FLOOR = 1e-3`); the bound-pinned near-separable
/// cases (#1082/#1237) certify with projected gradients well under O(1). A cost
/// stall whose projected gradient is far above this ceiling is NOT a flat valley
/// — it is a *stuck* stall produced by an inconsistent objective/gradient pair
/// (the inner PIRLS hit its iteration cap at this ρ, so the cached cost and the
/// analytic gradient disagree and no line-search direction makes cost progress
/// even though the surface is steep). Halting on such a stall shipped a silent
/// near-unpenalized full-basis overfit on ~7% of gamma/log datasets at default
/// k (#1426). At/above this ceiling the guard refuses to halt and keeps the
/// optimizer descending (see [`CostStallVerdict::StuckKeepDescending`]).
///
/// Set well above the legitimate flat-valley residual band (≲ O(1)) so the
/// near-separable multinomial / RKHS-collapse halts are unaffected, but far
/// below the #1426 stuck residual (|g| ≈ 11) so the gamma/log overfit is caught.
pub(crate) const FLAT_VALLEY_STALL_GRAD_CEILING: f64 = 5.0;

/// Score-relative stationarity tolerance for certifying a cost-stalled flat
/// valley (#1426/#1477). A cost stall whose best-iterate projected gradient
/// clears `FLAT_VALLEY_CONVERGED_REL_GRAD · (1 + |score|)` is reported
/// `converged = true` even when it exceeds the tight absolute
/// `COST_STALL_PROJECTED_GRAD_FLOOR = 1e-3`. The REML/LAML score for a
/// non-trivial fit is `O(1e2)–O(1e3)`, so this is a `O(0.1)` absolute gradient —
/// the residual a weakly-identified (near-zero-curvature) ρ coordinate floors at,
/// which mgcv's score-relative convergence certifies. Set so the wide
/// degeneracy-prior null-space valleys (correct fits, EDF well below the basis)
/// certify, while the #1426 stuck overfit (`|g| ≈ 11`, also above
/// `FLAT_VALLEY_STALL_GRAD_CEILING`) does not.
pub const FLAT_VALLEY_CONVERGED_REL_GRAD: f64 = 1.0e-3;

/// Multiplicative margin above the certified-stationary band
/// (`score_relative_grad_bound`) below which a cost stall is treated as a genuine
/// flat-valley FLOOR and halted directly, and above which it is treated as a
/// NON-stationary stall that is granted a [`CostStallVerdict::StuckKeepDescending`]
/// escape (#509). A stall whose residual gradient is within this factor of the
/// band is "essentially at the band" — descending further is not worth burning an
/// escape — so it halts; a stall meaningfully above the band still has real
/// feasible descent and is allowed to climb out. Set to `1.5×`: a stall within
/// 50% of the certified band is "essentially flat" and halts directly (preserving
/// the #1477 weakly-identified flat-valley floors, which floor AT their band), while
/// a stall well clear of the band — the #509 monotone seed-park floors at |g| ≈ 2
/// on a score ≈ 599 (band ≈ 0.6 ⇒ trigger ≈ 0.9, so 2 > 0.9 keeps descending) and
/// the #1426 stuck overfit at |g| ≈ 11 — is granted escapes. Capped at
/// `FLAT_VALLEY_STALL_GRAD_CEILING` so a very large score never raises the trigger
/// above the legacy ceiling.
pub(crate) const FLAT_VALLEY_STALL_ESCAPE_MARGIN: f64 = 1.5;

/// Absolute cap on [`FLAT_VALLEY_CONVERGED_REL_GRAD`]'s score-relative bound.
/// Without it a fit with a very large `|score|` would license certifying a large
/// projected gradient; capping at `1.0` keeps the certified band a genuinely
/// small absolute gradient regardless of score, and stays well below the
/// `FLAT_VALLEY_STALL_GRAD_CEILING = 5.0` stuck-stall band so a stuck overfit on a
/// large-score fit is never certified.
pub const FLAT_VALLEY_CONVERGED_ABS_GRAD_CAP: f64 = 1.0;

/// Score-relative flat-valley stationarity bound used by the in-loop ARC
/// cost-stall guard.
///
/// It had a second consumer — the post-fit certificate applied it through a rung
/// gated on `CostStallFlatValley` — until `9dd9b0842` deleted that rung (#2458:
/// an exit reason must not select the standard a point is judged by, and the
/// constant overruled the probe-noise measurement exactly where that measurement
/// declined). **The certificate does not call this function, and must not.**
///
/// Said plainly because the previous wording here — that centralizing the
/// formula "prevents the shipped-fit certificate from drifting away from the
/// guard" — outlived the consumer it named, and a reader who found the
/// certificate applying a different band reasonably concluded drift rather than
/// deletion (#2736).
///
/// PUBLIC so that tests asserting outer stationarity consume THIS value rather
/// than restating it. Three integration tests previously hand-copied
/// `1e-3 * (1 + |score|)` as a literal, which meant no gate could ever report
/// the constant itself wrong (#2519).
#[inline]
pub fn flat_valley_converged_grad_bound(score: f64) -> f64 {
    (FLAT_VALLEY_CONVERGED_REL_GRAD * (1.0 + score.abs())).min(FLAT_VALLEY_CONVERGED_ABS_GRAD_CAP)
}

/// What the stall window's own value scatter licenses about the gradient
/// (#2241, corrected by #2456).
///
/// The measurement is `σ̂/Δ`: the criterion's evaluation noise over the radius
/// the search actually probed. Below [`FLAT_VALLEY_CONVERGED_ABS_GRAD_CAP`]
/// that ratio is a genuine resolution certificate — no probe of the size the
/// line search takes can move the criterion by more than its own noise, so a
/// gradient under it is indistinguishable from zero *at the resolution the
/// criterion can be evaluated*.
///
/// ABOVE the ceiling it is not a weaker certificate; it is a different
/// statement. `σ̂/Δ = 5e1` says the criterion cannot resolve a gradient
/// anywhere below `5e1` — the instrument is blind in this regime, which is
/// evidence about the measurement, not about the point. The previous code
/// clamped the ratio with `.min(CAP)` and certified against the clamp, so a
/// failed measurement was replaced by a constant and the constant was then
/// accepted as a bound: `bound=1.000e0` on two fits whose objectives differ by
/// 4000x, because saturation is scale-invariant by construction. Worse, the
/// in-loop `converged` test read `|g| <= 1.0` as convergence, so the most
/// permissive threshold the code can produce was applied exactly in the regime
/// (`Δ → 0`) where the search is least converged.
///
/// A blind instrument certifies nothing, so [`Self::Unresolvable`] licenses no
/// bound at all and the stall falls back to the score-relative band — which is
/// never larger than the clamp, so this only ever tightens acceptance.
#[derive(Debug, Clone, Copy)]
pub(crate) enum ProbeNoiseVerdict {
    /// Fewer than three consecutive differences, or a degenerate probe radius:
    /// the window carried too little evidence to form the ratio.
    Unmeasured,
    /// `σ̂/Δ` measured within the admissible ceiling.
    Resolved {
        bound: f64,
        noise_floor: f64,
        probe_radius: f64,
    },
    /// `σ̂/Δ` measured above the ceiling: the criterion has no resolving power
    /// at this step scale, so no gradient bound follows from it.
    Unresolvable {
        ratio: f64,
        noise_floor: f64,
        probe_radius: f64,
    },
}

impl ProbeNoiseVerdict {
    /// The two quantities the ratio was formed from, whether or not the ratio
    /// licensed a bound: `(noise_floor σ̂, probe_radius Δ)`.
    ///
    /// `certified_bound` deliberately answers only "may this rung certify",
    /// which is `None` for both `Unmeasured` and `Unresolvable` — so a stall
    /// that measured σ̂ and Δ and found them unresolving is indistinguishable,
    /// downstream, from a stall that measured nothing. Those are different
    /// facts about a halt, and on the #1575 fixture they are the deciding ones:
    /// σ̂ is the per-step objective change the no-improvement window judged, and
    /// Δ is the radius the accepted steps actually moved. A window that filled
    /// because the search took microscopic steps and one that filled because
    /// the surface is genuinely flat differ in Δ, not in the verdict label.
    pub(crate) fn measured_scale(self) -> Option<(f64, f64)> {
        match self {
            Self::Unmeasured => None,
            Self::Resolved {
                noise_floor,
                probe_radius,
                ..
            }
            | Self::Unresolvable {
                noise_floor,
                probe_radius,
                ..
            } => Some((noise_floor, probe_radius)),
        }
    }

    /// The gradient bound this measurement licenses, if any.
    pub(crate) fn certified_bound(self) -> Option<f64> {
        match self {
            Self::Resolved { bound, .. } => Some(bound),
            Self::Unmeasured | Self::Unresolvable { .. } => None,
        }
    }
}

impl std::fmt::Display for ProbeNoiseVerdict {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Unmeasured => write!(f, "unmeasured (stall window carried too little evidence)"),
            Self::Resolved {
                bound,
                noise_floor,
                probe_radius,
            } => write!(
                f,
                "{bound:.3e} (sigma={noise_floor:.3e} / delta={probe_radius:.3e})"
            ),
            Self::Unresolvable {
                ratio,
                noise_floor,
                probe_radius,
            } => write!(
                f,
                "declined: sigma/delta={ratio:.3e} exceeds the {FLAT_VALLEY_CONVERGED_ABS_GRAD_CAP:.3e} \
                 resolution ceiling (sigma={noise_floor:.3e}, delta={probe_radius:.3e}), so the \
                 criterion resolves no gradient at this step scale"
            ),
        }
    }
}

/// Which term of the flat-valley acceptance `max` set the bound, alongside
/// every candidate term's value (#2456/#2465).
///
/// The refusal and the acceptance both used to report only the winning number,
/// and the terms saturate at the same `1.0` ceiling, so identical bounds came
/// out of unrelated derivations and the message invited the reader to reconcile
/// a bound with an objective it has no relation to. Carrying the whole `max`
/// makes the acceptance re-derivable from the run record.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum FlatValleyBoundTerm {
    /// The absolute outer gradient tolerance the ordinary BFGS path checks.
    SolverBand,
    /// `FLAT_VALLEY_CONVERGED_REL_GRAD·(1 + |score|)`, capped.
    ScoreRelative,
    /// The measured probe-noise resolution floor.
    ProbeNoise,
}

impl FlatValleyBoundTerm {
    pub(crate) fn label(self) -> &'static str {
        match self {
            Self::SolverBand => "solver-band",
            Self::ScoreRelative => "score-relative",
            Self::ProbeNoise => "probe-noise-floor",
        }
    }
}

/// The stationarity bound a cost-stall exit is judged against, carrying the
/// term that set it and the evidence behind each candidate (#2456).
#[derive(Debug, Clone, Copy)]
pub(crate) struct FlatValleyGradBound {
    pub(crate) value: f64,
    pub(crate) term: FlatValleyBoundTerm,
    pub(crate) solver_band: f64,
    /// The score-relative band BEFORE the absolute ceiling is applied. When it
    /// exceeds [`FLAT_VALLEY_CONVERGED_ABS_GRAD_CAP`] the reported `value` is
    /// the ceiling, which is a deliberate acceptance policy (it only tightens)
    /// — but it is then not the score-relative band, and saying so is the
    /// difference between a readable message and an unreconcilable one.
    pub(crate) score_relative_raw: f64,
    pub(crate) probe_noise: ProbeNoiseVerdict,
}

impl FlatValleyGradBound {
    fn new(solver_band: f64, score: f64, probe_noise: ProbeNoiseVerdict) -> Self {
        let score_relative_raw = FLAT_VALLEY_CONVERGED_REL_GRAD * (1.0 + score.abs());
        let score_relative = flat_valley_converged_grad_bound(score);
        let mut value = solver_band;
        let mut term = FlatValleyBoundTerm::SolverBand;
        if score_relative > value {
            value = score_relative;
            term = FlatValleyBoundTerm::ScoreRelative;
        }
        if let Some(noise) = probe_noise.certified_bound()
            && noise > value
        {
            value = noise;
            term = FlatValleyBoundTerm::ProbeNoise;
        }
        Self {
            value,
            term,
            solver_band,
            score_relative_raw,
            probe_noise,
        }
    }

    /// Whether the score-relative term was reported at the absolute ceiling
    /// rather than at the band its label names.
    pub(crate) fn score_relative_saturated(&self) -> bool {
        self.score_relative_raw > FLAT_VALLEY_CONVERGED_ABS_GRAD_CAP
    }
}

impl std::fmt::Display for FlatValleyGradBound {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "{:.3e} set by {}{}; terms: solver-band {:.3e}, score-relative {:.3e}{}, probe-noise {}",
            self.value,
            self.term.label(),
            if matches!(self.term, FlatValleyBoundTerm::ScoreRelative)
                && self.score_relative_saturated()
            {
                " (at the absolute ceiling, not the score-relative band)"
            } else {
                ""
            },
            self.solver_band,
            self.score_relative_raw,
            if self.score_relative_saturated() {
                format!(" capped to {FLAT_VALLEY_CONVERGED_ABS_GRAD_CAP:.3e}")
            } else {
                String::new()
            },
            self.probe_noise,
        )
    }
}

/// The incumbent state a stall escape was granted from, compared by raw bits.
///
/// Bit identity is the only comparison that supports the claim the escape cut
/// makes. The cut asserts that reopening the no-improvement window CANNOT
/// produce a different outcome, and that follows only from the search being in
/// the state it was in last time: a deterministic procedure replayed from an
/// identical state returns an identical result. Any tolerance weaker than
/// identity turns that proof into a guess about how much movement counts as
/// progress, and #2392 measured what such a guess costs — a run halted at
/// escape 1 of 8 with a residual gradient 165x its own keep-descending
/// threshold, because one window improved the best by less than roundoff while
/// the search was still moving.
///
/// Raw bits also keep the comparison total where a float comparison is not:
/// two non-finite incumbents match only when their payloads match, and `-0.0`
/// is not `0.0` — both errors, when made, are made on the safe side (grant the
/// escape, do not cut the budget).
#[derive(Clone, PartialEq, Eq)]
pub(crate) struct EscapeIncumbent {
    rho: Vec<u64>,
    value: u64,
    grad_norm: u64,
}

impl EscapeIncumbent {
    fn new(rho: &Array1<f64>, value: f64, grad_norm: f64) -> Self {
        Self {
            rho: rho.iter().map(|value| value.to_bits()).collect(),
            value: value.to_bits(),
            grad_norm: grad_norm.to_bits(),
        }
    }
}

/// Best iterate captured by a cost-stall convergence, handed from the bridge
/// (which is moved into `opt::Bfgs`) back to the seed-loop runner via the
/// guard's shared cell.
#[derive(Debug, Clone)]
pub(crate) struct CostStallExit {
    pub(crate) rho: Array1<f64>,
    pub(crate) value: f64,
    pub(crate) grad_norm: f64,
    /// Accepted outer iterates observed when the stall fired (for the runner's
    /// `OuterResult.iterations` field and logging).
    pub(crate) iterations: usize,
    /// Whether the best iterate is a genuine stationary optimum: `true` only
    /// when its projected gradient norm cleared the outer gradient tolerance
    /// (legitimately-flat REML surface). `false` for a flat-valley stall whose
    /// residual gradient remains above tolerance — the runner reports the
    /// rebuilt outer result as non-converged in that case.
    pub(crate) converged: bool,
    /// #2241 — the probe-noise-floor gradient bound σ̂/Δ measured at the stall
    /// (see [`CostStallGuard::probe_noise_verdict`]); `None` when the stall
    /// window carried too little evidence to measure it OR when the measured
    /// ratio exceeded the resolution ceiling, in which case the criterion
    /// resolves no gradient at this step scale and the rung licenses nothing
    /// (#2456). Carried onto `OuterResult.flat_noise_grad_bound` so the final
    /// analytic certificate judges the re-measured gradient against the same
    /// flat band the guard certified in the loop — and, just as importantly, so
    /// the certificate never adopts a `ProbeNoiseFloor` rung the guard did not
    /// actually measure.
    pub(crate) noise_grad_bound: Option<f64>,
    /// `(noise_floor σ̂, probe_radius Δ)` as measured at the stall, independent
    /// of whether their ratio licensed a bound. See
    /// [`ProbeNoiseVerdict::measured_scale`].
    pub(crate) probe_scale: Option<(f64, f64)>,
}

/// Tracks the monotone best accepted-iterate REML objective and a
/// no-improvement streak, firing a gradient-independent convergence once the
/// objective has effectively stopped decreasing. See the `cost_stall` field
/// doc on [`OuterFirstOrderBridge`] for the full rationale (#1089).
pub(crate) struct CostStallGuard {
    /// Relative improvement floor: an accepted step counts as "no improvement"
    /// when `(best - cost) <= rel_tol * (1 + |best|)`. Derived from the outer
    /// convergence tolerance so it tracks the configured precision rather than
    /// a free-standing magic constant.
    rel_tol: f64,
    /// Consecutive accepted-step window with no improvement before declaring
    /// convergence.
    window: usize,
    /// Projected outer gradient-norm threshold that the best iterate must clear
    /// for a cost stall to count as a genuine stationary optimum. This is the
    /// SAME threshold the normal BFGS convergence path uses
    /// (`outer_gradient_tolerance(config).abs`, which since #2613 is a function
    /// of the DECLARED problem and not of the seed the search started from),
    /// evaluated once at seed. A cost stall above this threshold is a
    /// flat-valley stall, reported `converged = false`.
    grad_threshold: f64,
    best_value: f64,
    best_rho: Option<Array1<f64>>,
    best_grad_norm: f64,
    /// Reduced-Hessian verdict synchronized with `best_rho`. `Some(false)`
    /// means the incumbent is a certified strict saddle and therefore cannot
    /// justify an infeasible-neighbourhood stall.
    best_hessian_psd: Option<bool>,
    no_improve_streak: usize,
    /// Consecutive infeasible (non-finite cost) outer trials since the last
    /// finite observation. On a near-separable multinomial fit ARC repeatedly
    /// probes the unbounded λ→0 separating region where the inner softmax solve
    /// does not converge and returns `OuterEval::infeasible` (cost=∞). Those
    /// trials never produce a finite descent, so the normal `no_improve_streak`
    /// (which only counts finite trials) can never fill its window — the loop
    /// would otherwise grind to `max_iter`. A run of `window` consecutive
    /// infeasible trials, once a finite best is in hand, is the same "no further
    /// real progress" signal and trips the same halt at the best feasible
    /// iterate (#1082/#1237).
    infeasible_streak: usize,
    accepted_iters: usize,
    /// Number of consecutive fruitless [`CostStallVerdict::StuckKeepDescending`]
    /// escapes granted on this seed (#1426), reset by any genuine super-floor
    /// improvement.
    ///
    /// This is a DIAGNOSTIC, reported in the escape log lines, and it gates
    /// nothing. It used to be compared against a fitted ceiling of 8 escapes,
    /// which was raised to whichever fixture was widest at the time. What
    /// actually bounds the escapes is [`Self::incumbent_at_last_escape`] (an
    /// escape from a bit-identical incumbent provably replays the previous one)
    /// together with the CONFIGURED outer iteration budget: each escape costs a
    /// full `window` of accepted or infeasible trials, so a run admits at most
    /// `max_iter / window` of them however this counter moves.
    ///
    /// `pub(crate)` so the escape tests can assert HOW MANY escapes a
    /// trajectory was granted. Both of them previously asserted only that a
    /// halt eventually arrived, which is satisfied by any ceiling >= 2.
    pub(crate) stuck_escapes: usize,
    /// The WHOLE incumbent at the moment the previous escape was granted --
    /// `(ρ, value, ‖g‖)` in raw bits -- so the NEXT stall can ask whether that
    /// escape produced any new state at all.
    ///
    /// An escape is a bet: the residual gradient says feasible descent remains,
    /// so reopen the window and let the search take it. The budget should be
    /// spent on escapes rather than on repetitions of one that provably cannot
    /// differ — and "provably" is the operative word. Reopening the window from
    /// a BIT-IDENTICAL incumbent replays a deterministic procedure from an
    /// identical state, so it must return an identical result; that is the
    /// geo_latlon pathology this exists for (eight escapes fired back to back
    /// at value=1.590092e2, |g|=3.055e-1 on every one).
    ///
    /// Keying it on the objective VALUE alone was too strong, and cut a run
    /// that was neither flat nor finished: #2392's exponentially stiff
    /// `wrong_rail` recovery halted at escape 1 of 8 with a residual
    /// `|g| = 2.479e2` — 165x its own keep-descending threshold of 1.5 and
    /// 5000x the stationarity bound it was about to be judged against — because
    /// one window had failed to improve the best by more than roundoff while
    /// the search itself was still moving. A window that visited new points
    /// gathered new information whether or not the incumbent improved, so the
    /// next escape is not a replay of it. `None` before the first escape of a
    /// streak, and cleared wherever [`Self::stuck_escapes`] is replenished.
    incumbent_at_last_escape: Option<EscapeIncumbent>,
    /// #2241 — the most recent trusted accepted iterates `(ρ_i, f_i)` (finite
    /// cost, inner solve converged), newest last, capped at `window + 1`
    /// entries. This is the raw evidence for the probe-noise-floor flat
    /// certificate: during a stall the consecutive value differences measure
    /// the criterion's own evaluation-noise scale, and the consecutive ρ
    /// distances measure the radius the search actually probed.
    recent: std::collections::VecDeque<(Array1<f64>, f64)>,
    /// Shared publication slot read by the seed-loop runner after
    /// `optimizer.run()` returns the sentinel error.
    exit: Arc<Mutex<Option<CostStallExit>>>,
}

impl CostStallGuard {
    pub(crate) fn new(
        rel_tol: f64,
        window: usize,
        grad_threshold: f64,
        exit: Arc<Mutex<Option<CostStallExit>>>,
    ) -> Self {
        Self {
            rel_tol,
            window,
            grad_threshold,
            best_value: f64::INFINITY,
            best_rho: None,
            best_grad_norm: f64::INFINITY,
            best_hessian_psd: None,
            no_improve_streak: 0,
            infeasible_streak: 0,
            accepted_iters: 0,
            stuck_escapes: 0,
            incumbent_at_last_escape: None,
            recent: std::collections::VecDeque::new(),
            exit,
        }
    }

    /// The iterate a stall is judged at: the recorded best, falling back to the
    /// current point for each field that is (pathologically) unset.
    fn best_iterate_or(
        &self,
        rho: &Array1<f64>,
        value: f64,
        grad_norm: f64,
    ) -> (Array1<f64>, f64, f64) {
        (
            self.best_rho.clone().unwrap_or_else(|| rho.clone()),
            if self.best_value.is_finite() {
                self.best_value
            } else {
                value
            },
            if self.best_grad_norm.is_finite() {
                self.best_grad_norm
            } else {
                grad_norm
            },
        )
    }

    /// Grant one stall escape unless it would replay the previous one.
    ///
    /// This is the whole termination argument for the escape mechanism, and it
    /// is a derivation rather than a budget. An escape reopens the
    /// no-improvement window and lets the search take another `window` steps
    /// from the incumbent. Reopening it from a BIT-IDENTICAL incumbent replays a
    /// deterministic procedure from an identical state, so it must return an
    /// identical result — that escape provably cannot differ and is refused.
    /// An escape whose window visited new points gathered new information even
    /// when the incumbent did not improve, so it is granted; the total number of
    /// such escapes is bounded by the CONFIGURED outer budget, because each one
    /// costs a full `window` of accepted or infeasible trials out of `max_iter`.
    ///
    /// Returns `true` when the escape was granted (and then reopens the
    /// no-improvement window as part of granting it).
    fn grant_escape_unless_replay(&mut self, incumbent: EscapeIncumbent) -> bool {
        if self.incumbent_at_last_escape.as_ref() == Some(&incumbent) {
            return false;
        }
        self.stuck_escapes = self.stuck_escapes.saturating_add(1);
        self.incumbent_at_last_escape = Some(incumbent);
        self.no_improve_streak = 0;
        true
    }

    /// Record one trusted accepted iterate into the #2241 noise-evidence
    /// buffer, keeping only the latest `window + 1` (⇒ `window` consecutive
    /// differences).
    fn record_recent(&mut self, rho: &Array1<f64>, value: f64) {
        self.recent.push_back((rho.clone(), value));
        while self.recent.len() > self.window + 1 {
            self.recent.pop_front();
        }
    }

    /// #2241 — probe-noise-floor gradient bound at a halted stall.
    ///
    /// Derivation. Over the stalled window the accepted iterates
    /// `(ρ_i, f_i)` satisfy: (i) the first-order model predicts
    /// `|f(ρ+d) − f(ρ)| ≤ ‖g‖·‖d‖ + o(‖d‖)` for moves of the size the search
    /// actually takes, and (ii) because the window is a stall (trend ≈ 0 by
    /// the `rel_tol` test), the consecutive differences `|f_i − f_{i−1}|` are
    /// dominated by the criterion's own evaluation noise — inner-solve
    /// truncation, reassembly order, quadrature — not by descent. So
    ///   σ̂ = median_i |f_i − f_{i−1}|   (robust noise-floor estimate),
    ///   Δ  = max_i ‖ρ_i − ρ_{i−1}‖₂    (radius the steps actually probed),
    /// and if `‖g_best‖ · Δ ≤ σ̂` then NO probe within the radius the line
    /// search is exploring can change the criterion by more than its own
    /// measurement noise: the surface is flat relative to its own noise scale
    /// and the best iterate is stationary at the resolution the criterion can
    /// be evaluated. Equivalently the gradient is certified below `σ̂/Δ`.
    ///
    /// Guards: σ̂ is floored at the roundoff scale `ε·(1+|f_best|)` (a
    /// byte-identical window cannot license an infinite bound of 0/Δ — it
    /// licenses exactly the roundoff resolution), and a ratio above
    /// [`FLAT_VALLEY_CONVERGED_ABS_GRAD_CAP`] is DECLINED rather than clamped
    /// (#2456): collapsed step sizes (Δ → 0 inflating σ̂/Δ) mean the criterion
    /// resolves nothing at this scale, and a blind instrument certifies
    /// nothing. Clamping instead returned the ceiling and then certified
    /// against it, which is how the most permissive threshold the code can
    /// produce came to be applied exactly where the search is least converged.
    /// The #1426 stuck stall (|g| ≈ 11) and the #509 seed-park (|g| ≈ 2) remain
    /// uncertifiable by this route under either treatment; declining is never
    /// the more permissive of the two, since it removes a term from a `max`.
    /// Returns [`ProbeNoiseVerdict::Unmeasured`] when fewer than three
    /// consecutive differences exist or the probe radius is degenerate.
    fn probe_noise_verdict(&self) -> ProbeNoiseVerdict {
        if self.recent.len() < 4 {
            return ProbeNoiseVerdict::Unmeasured;
        }
        let mut value_diffs: Vec<f64> = Vec::with_capacity(self.recent.len() - 1);
        let mut probe_radius = 0.0_f64;
        for (prev, next) in self.recent.iter().zip(self.recent.iter().skip(1)) {
            value_diffs.push((next.1 - prev.1).abs());
            let step = prev
                .0
                .iter()
                .zip(next.0.iter())
                .map(|(a, b)| (a - b) * (a - b))
                .sum::<f64>()
                .sqrt();
            probe_radius = probe_radius.max(step);
        }
        if !probe_radius.is_finite() || probe_radius <= 0.0 {
            return ProbeNoiseVerdict::Unmeasured;
        }
        value_diffs.sort_by(|a, b| a.total_cmp(b));
        let mid = value_diffs.len() / 2;
        let median = if value_diffs.len() % 2 == 1 {
            value_diffs[mid]
        } else {
            0.5 * (value_diffs[mid - 1] + value_diffs[mid])
        };
        if !median.is_finite() {
            return ProbeNoiseVerdict::Unmeasured;
        }
        let best_scale = if self.best_value.is_finite() {
            self.best_value.abs()
        } else {
            0.0
        };
        let noise_floor = median.max(f64::EPSILON * (1.0 + best_scale));
        let ratio = noise_floor / probe_radius;
        if !ratio.is_finite() {
            return ProbeNoiseVerdict::Unmeasured;
        }
        if ratio > FLAT_VALLEY_CONVERGED_ABS_GRAD_CAP {
            return ProbeNoiseVerdict::Unresolvable {
                ratio,
                noise_floor,
                probe_radius,
            };
        }
        ProbeNoiseVerdict::Resolved {
            bound: ratio,
            noise_floor,
            probe_radius,
        }
    }

    fn certified_grad_bound(&self) -> FlatValleyGradBound {
        FlatValleyGradBound::new(
            self.grad_threshold,
            self.best_value,
            self.probe_noise_verdict(),
        )
    }

    /// Register a precomputed feasible seed that the optimizer consumes from
    /// its internal cache instead of routing through the bridge. ARC's
    /// `with_initial_sample` path does exactly that: the first finite
    /// `(cost, gradient, Hessian)` is already known to the runner, so
    /// `eval_hessian` is not called at the seed. Without this hook the
    /// infeasible-trial stall path has no finite best iterate to halt back to
    /// when the next few ARC probes run into the λ→0 separating region.
    pub(crate) fn observe_seed(&mut self, rho: &Array1<f64>, value: f64, grad_norm: f64) {
        self.observe_seed_with_curvature(rho, value, grad_norm, None);
    }

    pub(crate) fn observe_second_order_seed(
        &mut self,
        rho: &Array1<f64>,
        value: f64,
        grad_norm: f64,
        hessian_psd: Option<bool>,
    ) {
        self.observe_seed_with_curvature(rho, value, grad_norm, hessian_psd);
    }

    fn observe_seed_with_curvature(
        &mut self,
        rho: &Array1<f64>,
        value: f64,
        grad_norm: f64,
        hessian_psd: Option<bool>,
    ) {
        if !value.is_finite() {
            return;
        }
        self.best_value = value;
        self.best_rho = Some(rho.clone());
        self.best_grad_norm = grad_norm;
        self.best_hessian_psd = hessian_psd;
        self.no_improve_streak = 0;
        self.infeasible_streak = 0;
        self.accepted_iters = self.accepted_iters.saturating_add(1);
        self.record_recent(rho, value);
        // Seed the shared exit cell so the budget-exhaustion path always has a
        // feasible best to fall back to, even if no later step improves (#1371).
        self.publish_best_so_far();
    }

    /// Fold one accepted-iterate `(ρ, cost, ‖g‖)` into the guard. Returns a
    /// [`CostStallVerdict`]: `Continue` while the score is still improving,
    /// `Converged` when the score has stalled AND the projected gradient norm
    /// at the best iterate clears the outer gradient tolerance (a genuine
    /// stationary optimum on a flat REML surface), or `FlatValleyStall` when
    /// the score has stalled but the residual gradient remains above tolerance
    /// (a weakly-identified flat valley that is NOT stationary). Either stalled
    /// verdict publishes the best iterate to the shared cell, tagged with its
    /// `converged` status.
    ///
    /// `inner_converged` is the inner-PIRLS convergence flag for THIS iterate's
    /// solve (read from the inner-progress feedback AFTER the outer eval ran). It
    /// is the load-bearing #1426 input: at the under-penalized (λ→0) ridge the
    /// inner PIRLS hits its iteration cap, so the cost it reports is a half-fit
    /// artifact that is SMALLER than the honest REML criterion at the
    /// well-penalized optimum, and the analytic gradient it pairs with is
    /// inconsistent. If such an iterate is allowed to become the guard's
    /// best-so-far it permanently anchors the best to the overfit basin — an
    /// uncapped, honest re-evaluation at a well-penalized ρ then carries a
    /// *higher* (correct) cost and can never displace it, so the loop ships the
    /// near-full-basis overfit. A non-converged inner solve is therefore NEVER
    /// recorded as best and NEVER counted toward a stall: its cost/gradient are
    /// untrustworthy, so the guard treats it as forced descent (streak reset) and
    /// keeps the optimizer moving until an honest, converged iterate lands.
    pub(crate) fn observe(
        &mut self,
        rho: &Array1<f64>,
        value: f64,
        grad_norm: f64,
        inner_converged: bool,
    ) -> CostStallVerdict {
        self.observe_with_curvature(rho, value, grad_norm, inner_converged, None)
    }

    pub(crate) fn observe_second_order(
        &mut self,
        rho: &Array1<f64>,
        value: f64,
        grad_norm: f64,
        inner_converged: bool,
        hessian_psd: Option<bool>,
    ) -> CostStallVerdict {
        self.observe_with_curvature(rho, value, grad_norm, inner_converged, hessian_psd)
    }

    fn observe_with_curvature(
        &mut self,
        rho: &Array1<f64>,
        value: f64,
        grad_norm: f64,
        inner_converged: bool,
        hessian_psd: Option<bool>,
    ) -> CostStallVerdict {
        if !value.is_finite() {
            // A non-finite accepted objective is the inner-solver's problem,
            // not a stall; reset so a later real descent is not falsely
            // credited as a no-improvement step. The dedicated infeasible-trial
            // bookkeeping lives in `observe_infeasible`; this finite-path entry
            // is left untouched for non-finite values.
            self.no_improve_streak = 0;
            return CostStallVerdict::Continue;
        }
        if !inner_converged {
            // #1426: the inner PIRLS hit its iteration cap at this ρ, so `value`
            // is a half-converged artifact and `grad_norm` is inconsistent with
            // it. Do NOT update best-so-far (a corrupted low cost at the λ→0
            // ridge would anchor the best to the overfit forever) and do NOT
            // count this toward a stall — reset the no-improvement streak so the
            // optimizer keeps stepping until a fully-converged inner solve lands
            // a trustworthy (cost, gradient) pair. The bridge separately uncaps
            // the inner solve when it detects a stuck stall, so subsequent solves
            // do converge and the best-so-far tracks an honest iterate.
            self.infeasible_streak = 0;
            self.no_improve_streak = 0;
            return CostStallVerdict::Continue;
        }
        // A finite trial means the inner solve produced a real cost: the
        // separating-region infeasible run is broken, so clear its streak.
        self.infeasible_streak = 0;
        self.accepted_iters = self.accepted_iters.saturating_add(1);
        self.record_recent(rho, value);
        let improvement = self.best_value - value;
        let floor = self.rel_tol * (1.0 + self.best_value.abs());
        if value < self.best_value {
            self.best_value = value;
            self.best_rho = Some(rho.clone());
            self.best_grad_norm = grad_norm;
            self.best_hessian_psd = hessian_psd;
            // Keep the shared exit cell tracking the best feasible iterate so the
            // ARC budget-exhaustion path can recover it instead of the optimizer's
            // last (possibly degenerate-corner) iterate (#1371).
            self.publish_best_so_far();
        }
        // KKT-stationary-at-bound (#1082/#1237). On a near-separable multinomial
        // the outer REML criterion keeps decreasing as λ→0, so several log-λ
        // directions slam to the lower box bound: the BOUND-PROJECTED gradient
        // is already below the outer tolerance (the iterate is KKT-stationary),
        // yet the raw cost keeps dropping by more than `floor` along those
        // bound-pinned directions — so a pure cost-improvement test resets the
        // no-improvement streak forever and the loop never certifies. Once the
        // projected gradient clears the stationarity threshold there is no
        // FEASIBLE descent left; further raw progress is bound-pinned drift, not
        // optimization. Treat such a trial as no-improvement so the window can
        // fill and the guard halts at the (stationary) best feasible iterate.
        // `opt::Arc`'s own gradient-tolerance check never trips here because it
        // tests the RAW gradient, which points out of the box forever.
        let kkt_stationary_at_bound = grad_norm.is_finite() && grad_norm <= self.grad_threshold;
        if improvement <= floor || kkt_stationary_at_bound {
            self.no_improve_streak = self.no_improve_streak.saturating_add(1);
        } else {
            self.no_improve_streak = 0;
            // A genuine super-floor improvement means the last stuck-stall
            // escape (if any) restored real descent, so the escape streak is
            // over: clear both the diagnostic count and the recorded incumbent
            // so the next stall is judged as a fresh streak rather than as a
            // replay of a window that predates real descent. Clearing the
            // incumbent is redundant with the bit-identity test (an improved
            // best cannot compare equal) and is kept because it makes the
            // state machine say what the streak means.
            self.stuck_escapes = 0;
            self.incumbent_at_last_escape = None;
        }
        if self.no_improve_streak < self.window {
            return CostStallVerdict::Continue;
        }
        // #2357: the finite window can fill while the recorded BEST is a
        // certified strict saddle — the guard tracks `best_hessian_psd` but the
        // finite-stall path never consulted it, so the stall halted at a
        // low-cost pass-through iterate whose curvature had not settled
        // (cold periodic-te() repro: window fills on three oscillating evals
        // near ρ₂≈10.7, halts to an earlier ρ₂≈5.4 point with hessian_psd=NO,
        // and the analytic certificate then refuses with INDEFINITE CURVATURE
        // AT INTERIOR OPTIMUM — while re-running warm from that very
        // checkpoint converges in a few more iterations). A strict saddle has
        // a certified local escape direction, so refuse the stall and return
        // control to cubic regularization exactly as the infeasible-run path
        // already does; the escape budget bounds consecutive fruitless
        // refusals so a surface that genuinely floors at a saddle still halts
        // (converged=false) after the budget instead of looping.
        if self.best_hessian_psd == Some(false) {
            let (best_rho, best_value, best_grad_norm) =
                self.best_iterate_or(rho, value, grad_norm);
            let incumbent = EscapeIncumbent::new(&best_rho, best_value, best_grad_norm);
            if self.grant_escape_unless_replay(incumbent) {
                log::warn!(
                    "[OUTER] ARC cost-stall window filled at a strict-saddle incumbent \
                     (hessian_psd=NO at best-so-far, value={:.6e}): refusing to certify the \
                     saddle and returning control to cubic regularization to exploit the \
                     negative curvature (escape {}).",
                    self.best_value,
                    self.stuck_escapes,
                );
                return CostStallVerdict::Continue;
            }
            log::info!(
                "[OUTER] ARC strict-saddle stall refusal cut at escape {}: the previous \
                 refusal reopened a full {}-step window and left the incumbent \
                 bit-identical (best={:.9e}, |g|={:.3e}), so refusing again replays the \
                 same window from the same state; halting.",
                self.stuck_escapes,
                self.window,
                best_value,
                best_grad_norm,
            );
        }
        self.publish_stall(rho, value, grad_norm)
    }

    /// Fold one INFEASIBLE outer trial (non-finite cost — typically a near-λ=0
    /// separating point whose inner softmax solve did not converge) into the
    /// guard. ARC keeps proposing these on a near-separable multinomial fit and
    /// they never reach `observe` with a finite value, so without this the
    /// no-improvement window can never fill and the outer loop grinds to
    /// `max_iter` (#1082/#1237). A run of `window` consecutive infeasible trials
    /// — provided a finite best is already in hand to halt back to — is the same
    /// "no further real progress" signal and trips the same halt.
    ///
    /// Returns `Continue` until the infeasible streak fills the window; never
    /// fires before any finite iterate has been recorded (there would be nothing
    /// to halt back to).
    pub(crate) fn observe_infeasible(&mut self, rho: &Array1<f64>) -> CostStallVerdict {
        if self.best_rho.is_none() || !self.best_value.is_finite() {
            // No feasible iterate recorded yet: an infeasible run this early is
            // the inner solver's startup problem, not a converged stall. Keep
            // descending so the optimizer can find its first feasible point.
            return CostStallVerdict::Continue;
        }
        self.infeasible_streak = self.infeasible_streak.saturating_add(1);
        if self.infeasible_streak < self.window {
            return CostStallVerdict::Continue;
        }
        if self.best_hessian_psd == Some(false) {
            // A strict saddle has a certified local escape direction. An
            // infeasible run only says ARC's current cubic step crossed the
            // profiled objective's domain wall; it cannot turn that saddle
            // into a terminal checkpoint. Clear the local run so ARC can raise
            // sigma, shrink the step, and exploit the known negative curvature.
            self.infeasible_streak = 0;
            self.no_improve_streak = 0;
            log::warn!(
                "[OUTER] ARC infeasible-probe run reached a strict-saddle incumbent; \
                 refusing the stall and returning control to cubic regularization"
            );
            return CostStallVerdict::Continue;
        }
        // Halt back to the best feasible iterate. Its projected gradient decides
        // converged-vs-flat-valley exactly as the finite stall path does.
        self.publish_stall(rho, self.best_value, self.best_grad_norm)
    }

    /// Publish the current finite probe as a constrained-stationary point.
    ///
    /// This is deliberately narrower than a generic cost stall: ARC can evaluate a
    /// bound-pinned separation probe whose raw gradient still points out of the
    /// feasible box. That point already satisfies the constrained KKT condition,
    /// but it may not be the guard's lowest raw-cost observation, so the generic
    /// best-so-far publication can resurrect an older non-stationary iterate and
    /// report `converged=false`. In that separation case the current feasible
    /// probe is the certificate-bearing point and should be returned directly.
    ///
    /// `grad_norm` MUST be the bound-PROJECTED gradient norm at the probe, not a
    /// presumed zero: the constrained-KKT certificate holds only when the pull on
    /// the railed axes is the *whole* residual. The projected norm retains any
    /// INTERIOR (feasible-descent) component, so `publish_stall` certifies
    /// `converged` iff the probe is genuinely stationary in the feasible subspace
    /// (#1426 — a flat-valley overfit that merely rails a couple of axes is
    /// reported NON-converged rather than certified behind a fake zero).
    pub(crate) fn observe_constrained_stationary(
        &mut self,
        rho: &Array1<f64>,
        value: f64,
        grad_norm: f64,
        inner_converged: bool,
        hessian_psd: Option<bool>,
    ) -> CostStallVerdict {
        if !value.is_finite() {
            return CostStallVerdict::Continue;
        }
        if !inner_converged {
            // #1426: a separation-stationary probe whose inner PIRLS did not
            // converge carries an untrustworthy (cost, gradient) pair. Adopting
            // it as the certificate-bearing optimum would ship a half-fit. Route
            // it through the ordinary finite-path observer, which (with
            // `inner_converged = false`) refuses to record it as best and keeps
            // the optimizer descending toward an honest, converged iterate.
            return self.observe_with_curvature(
                rho,
                value,
                grad_norm,
                inner_converged,
                hessian_psd,
            );
        }
        // A lower-bound separation probe is the certificate-bearing optimum ONLY
        // when it does not REGRESS the best feasible iterate already seen. In the
        // genuine near-separable case (#1082/#1237) the criterion decreases
        // monotonically toward the λ→0 bound, so the probe carries the lowest
        // cost and this guard is a no-op. But the SAME local test
        // (`lower_bound_outward_active_count`) also fires at an over-smoothing
        // collapse corner of a multi-penalty RKHS smooth (duchon/matern, #1355):
        // there a couple of operator penalties rail at the λ→0 LOWER bound (so
        // they look "separation-stationary") while OTHER penalties rail at the
        // λ→∞ UPPER bound and shrink the fit to a bare constant. Such a corner is
        // a spurious local KKT point whose REML cost is far WORSE than the
        // interior optimum the optimizer already passed through (commonly the
        // grid-prepass seed). Unconditionally adopting it discards that better
        // iterate and publishes a degenerate EDF≈1 constant fit. Only adopt the
        // probe when it does not regress the incumbent best.
        let regresses = self.best_value.is_finite()
            && value > self.best_value + self.rel_tol * (1.0 + self.best_value.abs());
        if regresses {
            // Spurious corner: keep the better incumbent. Fold the probe in as an
            // ordinary (non-improving) observation so `best_rho`/`best_grad_norm`
            // are preserved and the stall logic halts back to that incumbent
            // rather than to this corner.
            return self.observe_with_curvature(
                rho,
                value,
                grad_norm,
                inner_converged,
                hessian_psd,
            );
        }
        self.infeasible_streak = 0;
        self.accepted_iters = self.accepted_iters.saturating_add(1);
        self.best_value = value;
        self.best_rho = Some(rho.clone());
        self.best_grad_norm = grad_norm;
        self.best_hessian_psd = hessian_psd;
        self.no_improve_streak = self.window;
        self.publish_stall(rho, value, grad_norm)
    }

    /// Publish the best iterate to the shared exit cell and decide the stall
    /// verdict. Shared by the finite-stall and infeasible-stall paths.
    fn publish_stall(&mut self, rho: &Array1<f64>, value: f64, grad_norm: f64) -> CostStallVerdict {
        // Publish the best iterate. Prefer the recorded best; fall back to the
        // current point if (pathologically) none was stored.
        let (best_rho, best_value, best_grad_norm) = self.best_iterate_or(rho, value, grad_norm);
        // Convergence is STATIONARITY, measured RELATIVE TO THE SCORE SCALE.
        // A cost stall counts as a converged optimum when the projected gradient
        // at the best iterate clears EITHER (a) the absolute outer gradient
        // tolerance the genuine BFGS path checks, OR (b) a score-relative
        // stationarity bound `FLAT_VALLEY_CONVERGED_REL_GRAD · (1 + |score|)`.
        //
        // Bound (b) is the mgcv-aligned half (#1426/#1477). On a WEAKLY-IDENTIFIED
        // ρ coordinate — e.g. the double-penalty null-space log-λ of a smooth whose
        // null space is only weakly supported, under the wide degeneracy prior — the
        // REML surface is near-flat: the Hessian in that direction is ≈ 0, so the
        // projected gradient cannot be driven down to the tight ABSOLUTE
        // `COST_STALL_PROJECTED_GRAD_FLOOR = 1e-3` no matter how many outer steps
        // run; it floors at the valley's own residual (`O(0.1)` on a score of
        // `O(1e3)`). That is a genuine stationary optimum — mgcv certifies it (its
        // convergence is the gradient relative to the score, not an absolute 1e-3) —
        // and the cost-stall window has already proven the surface flattened. The
        // OLD absolute-only test reported these correct fits `converged = false`,
        // flooding the verdict with false alarms on exactly the fits the principled
        // wide-prior (no null-space over-shrink) produces. The score-relative bound
        // certifies them while a GENUINELY non-stationary residual — the #1426 stuck
        // overfit (|g| ≈ 11 on a score `O(1e3)`, i.e. above BOTH this bound and the
        // separate `FLAT_VALLEY_STALL_GRAD_CEILING`) — is still rejected and routed
        // to `StuckKeepDescending`, so no near-full-basis overfit is ever certified.
        let score_relative_grad_bound = flat_valley_converged_grad_bound(best_value);
        // #2241 — probe-noise-floor certificate: the stall window's own value
        // scatter and probed radius bound the gradient at which further probes
        // become indistinguishable from evaluation noise (derivation on
        // `probe_noise_verdict`). It composes with the score-relative band as a
        // second sufficient condition — but only when it RESOLVED (#2456);
        // above the resolution ceiling it is a statement that the criterion
        // measures nothing here, and it contributes no term.
        let certified = FlatValleyGradBound::new(
            self.grad_threshold,
            best_value,
            self.probe_noise_verdict(),
        );
        let noise_grad_bound = certified.probe_noise.certified_bound();
        let probe_scale = certified.probe_noise.measured_scale();
        let converged = best_grad_norm.is_finite() && best_grad_norm <= certified.value;
        if !converged
            && let ProbeNoiseVerdict::Unresolvable { .. } = certified.probe_noise
        {
            log::info!(
                "[OUTER] cost-stall probe-noise rung declined (#2456): |g|={best_grad_norm:.3e}, \
                 bound {certified}"
            );
        }
        if converged {
            if let Ok(mut slot) = self.exit.lock() {
                *slot = Some(CostStallExit {
                    rho: best_rho,
                    value: best_value,
                    grad_norm: best_grad_norm,
                    iterations: self.accepted_iters,
                    converged,
                    noise_grad_bound,
                    probe_scale,
                });
            }
            return CostStallVerdict::Converged;
        }
        // Distinguish a genuine flat-valley FLOOR (residual at its irreducible
        // band — the surface really HAS flattened, so no further descent is
        // available) from a NON-stationary stall whose residual gradient still
        // points to real feasible descent. The window filled, but a gradient
        // meaningfully above the certified-stationary band proves there IS a
        // descent direction left — the stall is an artifact of small per-step
        // cost progress, not of a flat surface. Two regimes produce such a stall:
        //
        //   * #1426 — the inner PIRLS hit its iteration cap at the under-penalized
        //     (λ→0) ridge, so the cached cost is a half-fit artifact and the
        //     analytic gradient (|g| ≈ 11) is inconsistent with it.
        //   * #509 — a shape-constrained (box-reparam β=Tγ) smooth whose inequality
        //     constraint is NON-binding: the constrained active-set inner solve is
        //     near-smooth but the cumulative-sum coordinate change makes the cost
        //     improve by less than the relative floor over the window near the
        //     integer seed, even though the projected gradient (|g| ≈ 2) still
        //     descends strongly toward the well-penalized REML optimum (verified:
        //     λ_wiggle 20 → 83, EDF 8.4 → 6.9, score −599 → −618, |g| → 1e-11).
        //
        // Both are cured the same way: refuse to halt and grant an escape — reset
        // the no-improvement window so the optimizer keeps descending (the inner
        // solve runs to tighter tolerance at the next iterate, restoring a
        // trustworthy gradient), for a bounded number of escapes before falling
        // back to a halt. The OLD gate keyed the escape on a fixed absolute
        // `FLAT_VALLEY_STALL_GRAD_CEILING = 5.0`, which let any stall in the
        // (certified-band, 5.0] residual band halt as a "flat valley" even though
        // its gradient still descended — silently parking the #509 monotone fit at
        // its seed (|g| ≈ 2 < 5.0). Keying the escape on the SCORE-RELATIVE
        // certified-stationary band instead (the same band that certifies
        // `converged` above, with a modest multiplicative margin so a stall sitting
        // essentially AT the band still halts directly rather than burning escapes)
        // makes the distinction scale-correct: a genuinely flat valley floors at
        // the band and halts after the escape budget, while a non-stationary stall
        // descends to true stationarity. The `FLAT_VALLEY_STALL_GRAD_CEILING` caps
        // the trigger so a very large score can never raise it above the legacy
        // ceiling — the #1426 stuck regime (|g| ≈ 11) is always granted escapes.
        let keep_descending_threshold = (FLAT_VALLEY_STALL_ESCAPE_MARGIN
            * score_relative_grad_bound)
            .min(FLAT_VALLEY_STALL_GRAD_CEILING);
        let non_stationary_stall =
            best_grad_norm.is_finite() && best_grad_norm > keep_descending_threshold;
        // Spend the budget on ESCAPES, not on repetitions of one that provably
        // did nothing. Each escape reopens a full `window` of accepted outer
        // steps, so a fruitless one is not free — measured on the geo_latlon
        // binomial fuzz scenarios, all eight fired back to back at a
        // BIT-IDENTICAL incumbent (value=1.590092e2, |g|=3.055e-1 on every one)
        // and the seven repeats cost about half the seed's wall clock before
        // the guard halted with the verdict escape 1 would have produced.
        //
        // The test is BIT-IDENTITY of the whole incumbent, not a tolerance on
        // the objective: the budget of eight was sized for a multi-shelf
        // descent whose 7th escape bought a 36-point objective drop (#2253),
        // and only an escape that left the search in the state it started from
        // is provably unrepeatable — reopening the window then replays a
        // deterministic procedure from an identical state. An escape whose
        // window visited new points gathered new information even when the
        // incumbent did not improve, and #2392 is what keying this on the
        // value alone cost: a still-descending run halted at escape 1 of 8
        // carrying |g| = 2.479e2 against a keep-descending threshold of 1.5.
        let escape_incumbent = EscapeIncumbent::new(&best_rho, best_value, best_grad_norm);
        if non_stationary_stall && self.grant_escape_unless_replay(escape_incumbent.clone()) {
            // The grant already reopened the no-improvement window. Reset the
            // infeasible streak too: the optimizer should be allowed a fresh
            // window of accepted AND infeasible steps to climb out of the stuck
            // state. Do NOT publish a halt — leave the shared exit cell tracking
            // the running best (`publish_best_so_far` already keeps it current)
            // so an eventual replay cut still recovers a sane iterate.
            self.infeasible_streak = 0;
            return CostStallVerdict::StuckKeepDescending {
                residual_grad_norm: best_grad_norm,
                escape_threshold: keep_descending_threshold,
            };
        }
        let previous_escape_replayed =
            self.incumbent_at_last_escape.as_ref() == Some(&escape_incumbent);
        if non_stationary_stall && previous_escape_replayed {
            log::info!(
                "[OUTER] cost-stall escape streak cut at {}: escape {} reopened a full                  {}-step window and left the incumbent bit-identical (best={:.9e}, |g|={:.3e}),                  so reopening it again replays the same window from the same state; halting.",
                self.stuck_escapes,
                self.stuck_escapes,
                self.window,
                best_value,
                best_grad_norm,
            );
        }
        if let Ok(mut slot) = self.exit.lock() {
            *slot = Some(CostStallExit {
                rho: best_rho,
                value: best_value,
                grad_norm: best_grad_norm,
                iterations: self.accepted_iters,
                converged,
                noise_grad_bound,
                probe_scale,
            });
        }
        CostStallVerdict::FlatValleyStall {
            residual_grad_norm: best_grad_norm,
        }
    }

    /// Publish the running best feasible iterate to the shared exit cell WITHOUT
    /// halting the optimizer (the verdict is discarded). Called on every accepted
    /// step that improves the best so the cell continuously tracks "best feasible
    /// iterate seen so far".
    ///
    /// This is the load-bearing hook for the ARC budget-exhaustion path
    /// (#1371). When ARC hits `max_iter` the seed-loop runner gets back only the
    /// optimizer's LAST iterate, which on a flat REML valley can be a degenerate
    /// box corner the trajectory wandered to (e.g. `ρ_nullspace → +∞` on a
    /// double-penalty smooth, which annihilates a genuine null-space linear
    /// trend and collapses the fit to a flat constant). Whenever this cell is
    /// populated, the runner returns the better of {last iterate, published best}
    /// — never returning an iterate whose REML objective is worse than one the
    /// optimizer already evaluated. Same spirit as the `observe_constrained_
    /// stationary` "do not adopt a corner that regresses the incumbent best"
    /// guard (#1355); here it covers the budget-exhaustion exit rather than the
    /// separation-probe exit.
    fn publish_best_so_far(&mut self) {
        let Some(best_rho) = self.best_rho.clone() else {
            return;
        };
        if !self.best_value.is_finite() {
            return;
        }
        // A running best-so-far snapshot is NEVER a convergence certificate: only
        // a genuine cost stall (`publish_stall`, whose window has proven the
        // surface flattened) or the second-order-deferred ARC gate may stamp
        // `converged = true`. Stamping it here on a merely-small projected
        // gradient certified a STILL-DESCENDING iterate — e.g. a bound-pinned
        // KKT-stationary point whose projected residual is transiently ≤ the
        // threshold mid-descent — and, because this publish returns
        // `CostStallVerdict::Continue`, the outer `observe_cost_stall` match never
        // runs `defer_finite_second_order_stall` to revoke it, so the false
        // convergence label survived to the exit cell (#2299 arc_bridge). The
        // snapshot exists ONLY so the budget-exhaustion path always has a feasible
        // best to halt back to (#1371/#2241); its convergence verdict is deferred
        // to ARC / the genuine-stall paths, so it is published `converged = false`.
        if let Ok(mut slot) = self.exit.lock() {
            *slot = Some(CostStallExit {
                rho: best_rho,
                value: self.best_value,
                grad_norm: self.best_grad_norm,
                iterations: self.accepted_iters,
                converged: false,
                // Not a halted stall: no noise-floor measurement is claimed
                // for a running best-so-far snapshot (#2241).
                noise_grad_bound: None,
                probe_scale: None,
            });
        }
    }

    /// A finite analytic second-order sample must reach ARC's synchronized
    /// projected-gradient + reduced-Hessian convergence gate. The generic
    /// cost-stall guard may still track the best feasible iterate, but it cannot
    /// halt the finite path from a first-order verdict. Reset only the finite
    /// stall window and revoke the provisional published convergence label;
    /// retain the best/recent evidence for budget-exhaustion recovery.
    fn defer_finite_second_order_stall(&mut self) {
        self.no_improve_streak = 0;
        if let Ok(mut slot) = self.exit.lock()
            && let Some(exit) = slot.as_mut()
        {
            exit.converged = false;
            exit.noise_grad_bound = None;
        }
    }

    /// An infeasible ARC probe carries no Hessian. Even when the generic guard's
    /// projected-gradient band classifies the stored best as stationary, the
    /// early-stop result is only a checkpoint until the mandatory outer
    /// certificate re-evaluates it with exact curvature.
    fn revoke_published_convergence(&mut self) {
        if let Ok(mut slot) = self.exit.lock()
            && let Some(exit) = slot.as_mut()
        {
            exit.converged = false;
        }
    }
}

#[derive(Clone)]
pub(crate) struct ValueProbeCacheEntry {
    rho: Array1<f64>,
    outcome: CachedValueProbeOutcome,
}

#[derive(Clone)]
pub(crate) enum CachedValueProbeOutcome {
    Cost(f64),
    Recoverable(String),
    Fatal(String),
}

/// Compact rendering of an outer θ point for the `[STAGE] outer eval` trace.
///
/// The trace used to report the VERDICT of an outer evaluation (its cost, its
/// gradient norm) without the POINT the verdict was measured at, so a log of a
/// finished search says how the criterion moved but not where — and "λ̂ railed
/// at the box" versus "λ̂ interior" is exactly the distinction that log has to
/// settle (#2596). Rendering θ costs one short string per evaluation on a path
/// whose per-evaluation work is an entire penalized fit.
///
/// Long vectors are elided after [`OUTER_TRACE_MAX_COORDS`] coordinates so a
/// wide model does not turn one log line into a page.
pub(crate) fn format_outer_theta(theta: &Array1<f64>) -> String {
    const OUTER_TRACE_MAX_COORDS: usize = 12;
    let shown = theta.len().min(OUTER_TRACE_MAX_COORDS);
    let mut out = String::with_capacity(4 + 9 * shown);
    out.push('[');
    for (i, value) in theta.iter().take(shown).enumerate() {
        if i > 0 {
            out.push(',');
        }
        out.push_str(&format!("{value:+.4}"));
    }
    if theta.len() > shown {
        out.push_str(&format!(",…+{}", theta.len() - shown));
    }
    out.push(']');
    out
}

pub(crate) fn trial_rho_distance(reference: Option<&Array1<f64>>, trial: &Array1<f64>) -> f64 {
    let Some(reference) = reference else {
        return f64::NAN;
    };
    if reference.len() != trial.len() {
        return f64::NAN;
    }
    reference
        .iter()
        .zip(trial.iter())
        .map(|(a, b)| {
            let d = b - a;
            d * d
        })
        .sum::<f64>()
        .sqrt()
}

pub(crate) fn same_outer_point(a: &Array1<f64>, b: &Array1<f64>) -> bool {
    a.len() == b.len()
        && a.iter()
            .zip(b.iter())
            .all(|(left, right)| left.to_bits() == right.to_bits())
}

pub(crate) fn cached_value_probe_result(
    outcome: &CachedValueProbeOutcome,
) -> Result<f64, ObjectiveEvalError> {
    match outcome {
        CachedValueProbeOutcome::Cost(cost) => Ok(*cost),
        CachedValueProbeOutcome::Recoverable(message) => {
            Err(ObjectiveEvalError::recoverable(message.clone()))
        }
        CachedValueProbeOutcome::Fatal(message) => Err(ObjectiveEvalError::fatal(message.clone())),
    }
}

pub(crate) fn cache_value_probe_result(
    result: &Result<f64, ObjectiveEvalError>,
) -> CachedValueProbeOutcome {
    match result {
        Ok(cost) => CachedValueProbeOutcome::Cost(*cost),
        Err(err) if err.is_recoverable() => {
            CachedValueProbeOutcome::Recoverable(err.message().to_string())
        }
        Err(err) => CachedValueProbeOutcome::Fatal(err.message().to_string()),
    }
}

pub(crate) fn value_probe_outcome_label(outcome: &CachedValueProbeOutcome) -> &'static str {
    match outcome {
        CachedValueProbeOutcome::Cost(_) => "cost",
        CachedValueProbeOutcome::Recoverable(_) => "recoverable",
        CachedValueProbeOutcome::Fatal(_) => "fatal",
    }
}

pub(crate) fn value_probe_reject_outcome(outcome: &CachedValueProbeOutcome) -> bool {
    match outcome {
        CachedValueProbeOutcome::Cost(cost) => *cost >= VALUE_PROBE_REJECT_COST_FLOOR,
        CachedValueProbeOutcome::Recoverable(_) | CachedValueProbeOutcome::Fatal(_) => true,
    }
}

pub(crate) fn remember_value_probe(
    cache: &mut Vec<ValueProbeCacheEntry>,
    rho: &Array1<f64>,
    outcome: CachedValueProbeOutcome,
) {
    if let Some(entry) = cache
        .iter_mut()
        .find(|entry| same_outer_point(&entry.rho, rho))
    {
        entry.outcome = outcome;
        return;
    }
    if cache.len() == VALUE_PROBE_CACHE_CAPACITY {
        cache.remove(0);
    }
    cache.push(ValueProbeCacheEntry {
        rho: rho.clone(),
        outcome,
    });
}

/// Classify a failure produced while evaluating one BFGS line-search value
/// probe.
///
/// [`EstimationError::HessianNotPositiveDefinite`] is deliberately fatal at
/// the general objective boundary: the same type can describe a structural
/// Hessian defect at a seed, accepted iterate, or final certificate. A value
/// probe is narrower. The incumbent has already supplied a feasible
/// cost/gradient sample, and this call asks only whether one different `rho`
/// is in the criterion's domain. If the inner penalized Hessian cannot be
/// factored there, the mathematically correct answer to the line search is
/// "reject this trial and shorten the step", not "abort the whole fit".
///
/// Keep that context-dependent verdict here instead of globally adding the
/// Hessian variant to `is_trial_point_infeasible`: gradient and terminal
/// evaluations continue to expose the failure as fatal.
fn into_line_search_value_probe_error(
    context: &str,
    err: EstimationError,
) -> ObjectiveEvalError {
    match err {
        // #2685: a saturated row, or an eta outside an inverse link's domain, is
        // a statement about the trial point's own linear predictor — the
        // incumbent evaluated fine and this probe asks only whether one
        // different theta is in the criterion's domain. Shorten the step.
        err @ EstimationError::HessianNotPositiveDefinite { .. }
        | err @ EstimationError::PirlsRowGeometryUnrepresentable { .. }
        | err @ EstimationError::InverseLinkDomainViolation { .. } => {
            ObjectiveEvalError::recoverable_from(err).with_context(context)
        }
        err => into_objective_error(context, err),
    }
}

#[cfg(test)]
mod line_search_value_probe_error_tests {
    use super::*;

    /// #2685: a saturated row at one value-only line-search trial rejects that
    /// trial. The row geometry's own payload is the trial point's `eta`, so the
    /// refusal is a statement about this theta, not about the problem.
    #[test]
    fn trial_row_geometry_refusal_is_local_to_the_value_probe_boundary_2685() {
        let trial_error = EstimationError::PirlsRowGeometryUnrepresentable {
            row: 0,
            quantity: "saturated Bernoulli row inconsistent with response",
            eta: 745.133_219_101_941_2,
            value: 0.0,
        };
        assert!(
            !trial_error.is_trial_point_infeasible(),
            "the row-geometry error remains fatal outside a proven line-search probe"
        );
        assert!(
            into_line_search_value_probe_error("outer eval_cost failed", trial_error)
                .is_recoverable(),
            "one saturated trial row must shorten the step, not abort the fit"
        );
        assert!(
            into_objective_error(
                "terminal outer evaluation failed",
                EstimationError::PirlsRowGeometryUnrepresentable {
                    row: 0,
                    quantity: "saturated Bernoulli row inconsistent with response",
                    eta: 745.133_219_101_941_2,
                    value: 0.0,
                },
            )
            .is_fatal(),
            "seed, gradient, and terminal row-geometry failures must stay fatal"
        );
    }

    /// #2273: an indefinite inner Hessian at one value-only line-search trial
    /// rejects that trial without weakening the type's general classification.
    #[test]
    fn trial_hessian_refusal_is_local_to_the_value_probe_boundary_2273() {
        let trial_error = EstimationError::HessianNotPositiveDefinite {
            min_eigenvalue: -3.055_337_809_473_694_1e-4,
        };
        assert!(
            !trial_error.is_trial_point_infeasible(),
            "the Hessian error remains fatal outside a proven line-search probe"
        );

        let objective_error =
            into_line_search_value_probe_error("outer eval_cost failed", trial_error);
        assert!(
            objective_error.is_recoverable(),
            "one infeasible line-search rho must shorten the step, not abort the fit"
        );
        assert!(matches!(
            objective_error.downcast_ref::<EstimationError>(),
            Some(EstimationError::HessianNotPositiveDefinite {
                min_eigenvalue
            }) if min_eigenvalue.to_bits()
                == (-3.055_337_809_473_694_1e-4_f64).to_bits()
        ));

        let terminal_error = into_objective_error(
            "terminal outer evaluation failed",
            EstimationError::HessianNotPositiveDefinite {
                min_eigenvalue: -3.055_337_809_473_694_1e-4,
            },
        );
        assert!(
            terminal_error.is_fatal(),
            "seed, gradient, and terminal Hessian failures must stay fatal"
        );
    }
}

impl ZerothOrderObjective for OuterFirstOrderBridge<'_> {
    fn eval_cost(&mut self, x: &Array1<f64>) -> Result<f64, ObjectiveEvalError> {
        // Consume any accepted-step signal `opt` published since the previous
        // evaluation before doing anything else: a stalled verdict must halt
        // this call rather than pay another inner solve first (#2613).
        self.drain_accepted_steps()?;
        // Per-axis line-search step caps now live natively in opt::Bfgs
        // (`with_axis_step_caps`), which shortens the BFGS direction before
        // line search instead of poisoning the Wolfe bracket with a
        // sentinel cost. This entry point can therefore stay honest: any
        // call that lands here is a real line-search probe, not a too-far
        // attempt the bridge needs to swat away.
        //
        // Uncap the inner solve for the line-search cost probe (see the field
        // doc on `outer_inner_cap`): the deciding cost MUST be the true
        // converged-inner objective the analytic gradient differentiates, not
        // the scheduled gradient-path cap which can stop a trial-ρ inner solve
        // short of its fixed point and report a spurious `∞`. `eval_grad`
        // restores the scheduled cap on the next call.
        if let Some(feedback) = self.outer_inner_cap.as_ref() {
            feedback
                .cap
                .store(SEED_SCREENING_UNCAPPED, Ordering::Relaxed);
        }
        self.layout
            .validate_point_len(x, "outer eval_cost failed")?;
        let trial_rho_distance = trial_rho_distance(self.last_value_grad_rho.as_ref(), x);
        let stage_start = std::time::Instant::now();
        if let Some(entry) = self
            .value_probe_cache
            .iter()
            .find(|entry| same_outer_point(&entry.rho, x))
        {
            let outcome_label = value_probe_outcome_label(&entry.outcome);
            log::info!(
                "[STAGE] outer eval start order=Value dim={} trial_rho_distance={:.3e} (first-order bridge, eval={}, cached=true)",
                x.len(),
                trial_rho_distance,
                self.first_order_evals
            );
            match &entry.outcome {
                CachedValueProbeOutcome::Cost(cost) => log::info!(
                    "[STAGE] outer eval end order=Value elapsed={:.3}s cost={:.6e} trial_rho_distance={:.3e} (first-order bridge, eval={}, cached=true)",
                    stage_start.elapsed().as_secs_f64(),
                    cost,
                    trial_rho_distance,
                    self.first_order_evals
                ),
                CachedValueProbeOutcome::Recoverable(_) | CachedValueProbeOutcome::Fatal(_) => {
                    log::info!(
                        "[STAGE] outer eval end order=Value elapsed={:.3}s outcome={} trial_rho_distance={:.3e} (first-order bridge, eval={}, cached=true)",
                        stage_start.elapsed().as_secs_f64(),
                        outcome_label,
                        trial_rho_distance,
                        self.first_order_evals
                    );
                }
            }
            return cached_value_probe_result(&entry.outcome);
        }
        log::info!(
            "[STAGE] outer eval start order=Value dim={} trial_rho_distance={:.3e} (first-order bridge, eval={})",
            x.len(),
            trial_rho_distance,
            self.first_order_evals
        );
        let result = self
            .obj
            .eval_with_order(x, OuterEvalOrder::Value)
            .map_err(|err| {
                into_line_search_value_probe_error("outer eval_cost failed", err)
            })
            .and_then(|eval| finite_cost_or_error("outer eval_cost failed", eval.cost));
        let cached_outcome = cache_value_probe_result(&result);
        remember_value_probe(&mut self.value_probe_cache, x, cached_outcome);
        match &result {
            Ok(cost) => {
                // A successful probe resets the consecutive-refusal counter: the
                // current ρ neighbourhood has at least one feasible point, so
                // isolated refusals on other directions are normal line-search
                // noise, not a globally-infeasible neighbourhood.
                self.consecutive_probe_refusals = 0;
                log::info!(
                    "[STAGE] outer eval end order=Value elapsed={:.3}s cost={:.6e} trial_rho_distance={:.3e} (first-order bridge, eval={}) theta={}",
                    stage_start.elapsed().as_secs_f64(),
                    cost,
                    trial_rho_distance,
                    self.first_order_evals,
                    format_outer_theta(x),
                );
            }
            Err(err) if err.is_recoverable() => {
                // The REASON is the whole content of an infeasible probe. A
                // line search that halves its step forever is diagnosable only
                // if the log says WHICH domain the trial point left; without it
                // the trail reads "outcome=recoverable" a hundred times and
                // names nothing. `ObjectiveEvalError: Display` already carries
                // the originating error's own message, so this costs one field.
                log::info!(
                    "[STAGE] outer eval end order=Value elapsed={:.3}s outcome=recoverable trial_rho_distance={:.3e} (first-order bridge, eval={}) theta={} reason={}",
                    stage_start.elapsed().as_secs_f64(),
                    trial_rho_distance,
                    self.first_order_evals,
                    format_outer_theta(x),
                    err,
                );
                if let Some(guard) = self.cost_stall.as_mut() {
                    match guard.observe_infeasible(x) {
                        CostStallVerdict::Continue => {}
                        CostStallVerdict::StuckKeepDescending {
                            residual_grad_norm,
                            escape_threshold,
                        } => {
                            // #1426: best feasible iterate carries a residual far
                            // above tolerance — not a flat valley. Keep going.
                            log::warn!(
                                "[OUTER] cost-stall STUCK (infeasible BFGS probes, NOT a flat \
                                 valley): residual |g|={:.3e} far above the certified-stationary \
                                 band (escape threshold {:.3e}); refusing \
                                 to halt-and-ship and continuing (escape {}, value={:.6e}).",
                                residual_grad_norm,
                                escape_threshold,
                                guard.stuck_escapes,
                                guard.best_value,
                            );
                        }
                        CostStallVerdict::Converged => {
                            log::info!(
                                "[OUTER] cost-stall convergence (infeasible BFGS probes): {} \
                                 consecutive infeasible probes after a finite seed/iterate; \
                                 accepting best-so-far as a stationary optimum (value={:.6e}).",
                                guard.infeasible_streak,
                                guard.best_value,
                            );
                            return Err(ObjectiveEvalError::fatal(COST_STALL_CONVERGED_SENTINEL.to_string()));
                        }
                        CostStallVerdict::FlatValleyStall { residual_grad_norm } => {
                            log::warn!(
                                "[OUTER] cost-stall halt (infeasible BFGS probes): {} \
                                 consecutive infeasible probes after a finite seed/iterate; \
                                 halting at best-so-far with residual |g|={:.3e} \
                                 (value={:.6e}).",
                                guard.infeasible_streak,
                                residual_grad_norm,
                                guard.best_value,
                            );
                            return Err(ObjectiveEvalError::fatal(COST_STALL_CONVERGED_SENTINEL.to_string()));
                        }
                    }
                }
                // Non-termination guard (#NaN-outer-loop): when every
                // line-search probe is infeasible and BFGS has never
                // accepted a gradient step (`first_order_evals == 0`), the
                // neighbourhood around the seed is globally degenerate.
                // BFGS would otherwise spend its entire max_iterations ×
                // line_search_budget doing inner solves that all fail.
                // Escalate to Fatal so BFGS exits immediately; the seed
                // loop routes it as a rejected seed.
                self.consecutive_probe_refusals = self.consecutive_probe_refusals.saturating_add(1);
                // When the bridge seed (cost, gradient) was supplied via
                // `with_initial_sample` the bridge's own `eval_grad` is
                // never called, so `last_value_grad_rho` stays `None` and
                // every `trial_rho_distance` prints as NaN.  The seed IS
                // feasible (it was evaluated externally), but if every
                // line-search probe is Recoverable from the very first
                // direction, the neighbourhood is globally infeasible.
                // Use the tighter NaN-seed threshold so the guard fires
                // after one generous Wolfe budget instead of 150 probes
                // (which, at ~3 s each × 13 seeds, would produce an
                // observed ~97 min hang on real D=5120 LLM activations).
                let threshold = if self.last_value_grad_rho.is_none() {
                    PROBE_REFUSAL_FATAL_THRESHOLD_NAN_SEED
                } else {
                    PROBE_REFUSAL_FATAL_THRESHOLD
                };
                if self.first_order_evals == 0 && self.consecutive_probe_refusals >= threshold {
                    log::warn!(
                        "[OUTER] probe-refusal non-termination guard fired after {} consecutive \
                         infeasible cost probes with no accepted gradient step \
                         (nan_seed={}); escalating to Fatal to abort this seed \
                         (first-order bridge, eval={})",
                        self.consecutive_probe_refusals,
                        self.last_value_grad_rho.is_none(),
                        self.first_order_evals,
                    );
                    return Err(ObjectiveEvalError::fatal(format!(
                            "{PROBE_REFUSAL_FATAL_SENTINEL}: {consecutive} consecutive \
                             infeasible probes with no accepted outer step",
                            consecutive = self.consecutive_probe_refusals,
                        )));
                }
            }
            Err(_err) => {
                log::info!(
                    "[STAGE] outer eval end order=Value elapsed={:.3}s outcome=fatal trial_rho_distance={:.3e} (first-order bridge, eval={})",
                    stage_start.elapsed().as_secs_f64(),
                    trial_rho_distance,
                    self.first_order_evals
                );
            }
        }
        result
    }
}

impl FirstOrderObjective for OuterFirstOrderBridge<'_> {
    fn eval_grad(&mut self, x: &Array1<f64>) -> Result<FirstOrderSample, ObjectiveEvalError> {
        self.layout.validate_point_len(x, "outer eval failed")?;
        // Consume any accepted-step signal `opt` published since the previous
        // evaluation before doing anything else: a stalled verdict must halt
        // this call rather than pay another inner solve first (#2613).
        self.drain_accepted_steps()?;
        // Drive the outer-aware inner-PIRLS cap from accepted outer
        // iterations, BEFORE invoking the inner solve. Cap stays fixed
        // within line-search cost probes (`eval_cost` never touches the
        // atomic). A cap of 0 means "no cap from this source"; the inner
        // solver still honors `pirls_max_iterations` and the screening cap.
        if let Some(feedback) = self.outer_inner_cap.as_ref() {
            let g_ratio = match (self.last_g_norm, self.g_norm_initial) {
                (Some(g), Some(g0)) if g0 > 0.0 => Some(g / g0),
                _ => None,
            };
            let snapshot = feedback.snapshot();
            let accepted_iter = feedback.accepted_iter.load(Ordering::Relaxed);
            let cap = first_order_inner_cap_schedule(accepted_iter, g_ratio, snapshot);
            let prev = feedback.cap.swap(cap, Ordering::Relaxed);
            if prev != cap {
                let ratio_str = match g_ratio {
                    Some(r) => format!("{:.3e}", r),
                    None => "n/a".to_string(),
                };
                let snap_str = match snapshot {
                    Some(s) => format!(
                        "last_iters={} converged={} ift_residual={} accept_rho={}",
                        s.last_iters,
                        s.last_converged,
                        match s.last_ift_residual {
                            Some(r) => format!("{:.3e}", r),
                            None => "n/a".to_string(),
                        },
                        match s.last_accept_rho {
                            Some(r) => format!("{:.3}", r),
                            None => "n/a".to_string(),
                        },
                    ),
                    None => "no-history".to_string(),
                };
                log::info!(
                    "[OUTER schedule] inner-PIRLS cap transition accepted_iter={} eval_count={} g_ratio={} {} prev={} new={} ({})",
                    accepted_iter,
                    self.first_order_evals,
                    ratio_str,
                    snap_str,
                    prev,
                    cap,
                    if cap == 0 { "uncapped" } else { "capped" }
                );
            }
        }
        let stage_start = std::time::Instant::now();
        log::info!(
            "[STAGE] outer eval start order=ValueAndGradient dim={} (first-order bridge, eval={})",
            x.len(),
            self.first_order_evals
        );
        let eval = self
            .obj
            .eval_with_order(x, OuterEvalOrder::ValueAndGradient)
            .map_err(|err| into_objective_error("outer eval failed", err))?;
        let eval = finite_outer_first_order_eval_or_error("outer eval failed", self.layout, eval)?;
        let g_norm = eval.gradient.iter().map(|v| v * v).sum::<f64>().sqrt();
        let gradient = eval.gradient;
        if self.g_norm_initial.is_none() && g_norm.is_finite() && g_norm > 0.0 {
            self.g_norm_initial = Some(g_norm);
        }
        if g_norm.is_finite() {
            self.last_g_norm = Some(g_norm);
        }
        self.last_value_grad_rho = Some(x.clone());
        // A successful gradient evaluation means the current ρ is feasible;
        // reset the consecutive-probe-refusal counter so the guard only fires
        // when ALL probes in EVERY subsequent direction fail.
        self.consecutive_probe_refusals = 0;
        self.value_probe_cache
            .retain(|entry| value_probe_reject_outcome(&entry.outcome));
        log::info!(
            "[STAGE] outer eval end order=ValueAndGradient elapsed={:.3}s cost={:.6e} |g|={:.3e} (first-order bridge, eval={}) theta={} g={}",
            stage_start.elapsed().as_secs_f64(),
            eval.cost,
            g_norm,
            self.first_order_evals,
            format_outer_theta(x),
            format_outer_theta(&gradient),
        );
        self.first_order_evals = self.first_order_evals.saturating_add(1);
        // Cost-stall bookkeeping (#1089, corrected by #2613). This evaluation
        // is NOT necessarily an accepted outer iterate. The premise that used
        // to sit here — "`eval_grad` is invoked by `opt::Bfgs` at each accepted
        // iterate (line-search COST probes go through `eval_cost`, not here)" —
        // is false: the Strong-Wolfe search evaluates the GRADIENT at every
        // trial that clears Armijo, because the curvature condition needs it.
        // So the sample is recorded here and folded by
        // `drain_accepted_steps` once `opt` reports, through
        // `OuterAcceptObserver`, which one it accepted. See the
        // `accepted_steps` field doc for what the conflation cost.
        //
        // #1426: read the inner-PIRLS convergence flag for the solve THIS eval
        // just ran (the feedback atomics are updated by `execute_pirls_if_needed`
        // after each non-screening solve, so the snapshot now reflects this ρ).
        // A non-finite / non-converged inner solve makes the reported
        // cost/gradient untrustworthy; the guard must not record it as
        // best-so-far nor count it toward a stall. `None` (no feedback wired)
        // defaults to `true` so routes without inner-cap feedback are unchanged.
        // It is captured HERE rather than at fold time because the snapshot is
        // only valid immediately after this ρ's solve.
        let inner_converged = inner_solve_converged(self.outer_inner_cap.as_ref());
        if self.cost_stall.is_some() {
            // The stall guard's stationarity test must use the bound-PROJECTED
            // gradient norm (KKT residual), not the raw `g_norm` above — a
            // separation fit pins log-λ directions at the bound with a
            // persistent out-of-bounds ∂V/∂ρ that inflates the raw norm forever
            // and otherwise blocks the converged verdict (#1082). The raw
            // `g_norm` is kept for the inner-cap schedule / logging.
            // Measured against the rail-relaxed box, the same one the terminal
            // certificate uses (#2412): a coordinate creeping onto the ceiling
            // must not keep an outward pull here that certification discards,
            // or the guard never reaches the verdict certification would give.
            let projected_grad_norm =
                rail_projected_gradient_norm(x, &gradient, self.cost_stall_bounds.as_ref());
            let sample = PendingOuterEval {
                rho: x.clone(),
                cost: eval.cost,
                projected_grad_norm,
                inner_converged,
            };
            match self.accepted_steps.is_some() {
                true => {
                    if self.pending_first_order.len() >= PENDING_FIRST_ORDER_CAPACITY {
                        self.pending_first_order.remove(0);
                    }
                    self.pending_first_order.push(sample);
                }
                // No accept signal wired (a caller that built the bridge
                // directly, e.g. a unit test): every gradient eval is folded,
                // which is the pre-#2613 behaviour and is safe on any driver
                // that really does call `eval_grad` once per accepted step.
                false => self.fold_accepted_iterate(&sample)?,
            }
        }
        Ok(FirstOrderSample {
            value: eval.cost,
            gradient,
        })
    }
}

impl OuterFirstOrderBridge<'_> {
    /// Consume every accepted outer step `opt` has reported since the last
    /// evaluation and fold the corresponding iterate into the cost-stall guard
    /// (#2613).
    ///
    /// Called at the top of `eval_cost` and `eval_grad`, which is the earliest
    /// the bridge can act: `on_step_accepted` fires *after* the line search
    /// that produced the step, so the signal is always one evaluation old. A
    /// stalled verdict returns the sentinel `Fatal` from whichever call
    /// observes it — an observer cannot stop `opt::Bfgs`, an error is the only
    /// in-band way.
    fn drain_accepted_steps(&mut self) -> Result<(), ObjectiveEvalError> {
        let Some(ledger) = self.accepted_steps.clone() else {
            return Ok(());
        };
        if self.cost_stall.is_none() {
            return Ok(());
        }
        let steps = ledger.drain();
        if steps.is_empty() {
            return Ok(());
        }
        let mut outcome = Ok(());
        for step in &steps {
            match self.resolve_accepted_iterate(step) {
                Some(sample) => {
                    self.incumbent = Some((sample.rho.clone(), sample.cost));
                    outcome = self.fold_accepted_iterate(&sample);
                    if outcome.is_err() {
                        break;
                    }
                }
                None => log::debug!(
                    "[OUTER] accepted outer step {} (step_norm={:.3e}, actual_decrease={:.3e}) \
                     matched none of the {} first-order evaluations since the last accept; \
                     the cost-stall guard skips it rather than fold a point opt did not accept",
                    step.iter,
                    step.step_norm,
                    step.actual_decrease,
                    self.pending_first_order.len(),
                ),
            }
        }
        self.pending_first_order.clear();
        outcome
    }

    /// Decide WHICH pending first-order evaluation is the iterate `opt`
    /// accepted.
    ///
    /// `opt::StepInfo` carries no point: only `actual_decrease = f_k − f_next`
    /// and `step_norm = ‖x_next − x_k‖`. Both are reconcilable against the
    /// bridge's own incumbent, and the cost is the sharper of the two (an f64
    /// criterion value pins an iterate; a step LENGTH does not distinguish two
    /// trials equidistant from `x_k`). So match on the reconstructed cost
    /// first, newest candidate wins, and fall back to the step length only when
    /// no cost matches — which happens when `opt` moved the incumbent through
    /// one of its rescue paths (global-best salvage, trust-region dogleg), none
    /// of which fire the observer, leaving this bridge's incumbent stale.
    fn resolve_accepted_iterate(&self, step: &AcceptedOuterStep) -> Option<PendingOuterEval> {
        if self.pending_first_order.is_empty() {
            return None;
        }
        let target = self
            .incumbent
            .as_ref()
            .map(|(_, cost)| cost - step.actual_decrease)
            .filter(|target| target.is_finite());
        if let Some(target) = target {
            let slack = ACCEPTED_STEP_COST_MATCH_ULPS * f64::EPSILON * (1.0 + target.abs());
            if let Some(found) = self
                .pending_first_order
                .iter()
                .rev()
                .find(|entry| (entry.cost - target).abs() <= slack)
            {
                return Some(found.clone());
            }
        }
        // Step-length fallback. Without a usable incumbent ρ there is nothing
        // to measure a length against, so take the most recent evaluation —
        // the line search returns the trial it evaluated last on every path
        // that does not go through a rescue.
        let Some((incumbent_rho, _)) = self.incumbent.as_ref() else {
            return self.pending_first_order.last().cloned();
        };
        if !step.step_norm.is_finite() {
            return self.pending_first_order.last().cloned();
        }
        self.pending_first_order
            .iter()
            .rev()
            .map(|entry| {
                let moved = entry
                    .rho
                    .iter()
                    .zip(incumbent_rho.iter())
                    .map(|(a, b)| (a - b) * (a - b))
                    .sum::<f64>()
                    .sqrt();
                (entry, (moved - step.step_norm).abs())
            })
            .min_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal))
            .map(|(entry, _)| entry.clone())
    }

    /// Fold one ACCEPTED outer iterate into the cost-stall guard and act on the
    /// verdict.
    ///
    /// When the REML score has stopped improving over `COST_STALL_WINDOW`
    /// consecutive accepted steps, halt BFGS by returning a sentinel `Fatal`.
    /// The runner rebuilds the outer result from the published best iterate —
    /// but whether that result is reported CONVERGED is decided by the guard's
    /// STATIONARITY test, not by cost-flatness alone: a stall whose projected
    /// gradient still exceeds the outer gradient tolerance is a flat-valley
    /// floor (`converged = false`), a stationary one is a real optimum
    /// (`converged = true`). Both share the sentinel; the verdict rides on the
    /// published `CostStallExit.converged`.
    fn fold_accepted_iterate(
        &mut self,
        sample: &PendingOuterEval,
    ) -> Result<(), ObjectiveEvalError> {
        let Some(guard) = self.cost_stall.as_mut() else {
            return Ok(());
        };
            match guard.observe(
                &sample.rho,
                sample.cost,
                sample.projected_grad_norm,
                sample.inner_converged,
            ) {
                CostStallVerdict::Continue => {}
                CostStallVerdict::StuckKeepDescending {
                    residual_grad_norm,
                    escape_threshold,
                } => {
                    // #1426: cost flatlined but the projected gradient is far
                    // above tolerance — a stuck stall from an inconsistent
                    // (non-converged inner solve) objective/gradient, NOT a flat
                    // valley. Do NOT halt: keep descending so the optimizer can
                    // climb out toward the well-penalized optimum. UNCAP the
                    // inner PIRLS so the next solves run to full tolerance — the
                    // stuck state is caused by the inner cap stopping PIRLS short
                    // of its fixed point, leaving the cost and the analytic
                    // gradient inconsistent; an uncapped solve restores a
                    // trustworthy gradient the outer step can actually use.
                    // #2349: raise the cold-reeval pulse alongside the uncap so
                    // a warm-start-hysteresis stall on a near-separating fit
                    // re-solves the next outer evaluations COLD (see the ARC arm
                    // and the `force_cold` field doc for the full rationale).
                    if let Some(feedback) = self.outer_inner_cap.as_ref() {
                        feedback.cap.store(0, Ordering::Relaxed);
                        feedback.force_cold.store(true, Ordering::Relaxed);
                    }
                    log::warn!(
                        "[OUTER] cost-stall STUCK (NOT a flat valley): REML objective improved \
                         < {:.3e} (relative) over {} accepted outer steps but the projected \
                         gradient is FAR above the certified-stationary band \
                         (|g|={:.3e} > escape threshold {:.3e}); refusing \
                         to halt-and-ship and continuing the descent (escape {}, value={:.6e}).",
                        guard.rel_tol,
                        guard.window,
                        residual_grad_norm,
                        escape_threshold,
                        guard.stuck_escapes,
                        guard.best_value,
                    );
                }
                CostStallVerdict::Converged => {
                    // Report the band that ACTUALLY certified: absolute
                    // tolerance, score-relative flat-valley bound, or the
                    // #2241 probe-noise floor — printing only the raw
                    // tolerance made a flat-band acceptance read as an
                    // arithmetic impossibility in the log.
                    log::info!(
                        "[OUTER] cost-stall convergence: REML objective improved < {:.3e} \
                         (relative) over {} consecutive accepted outer steps AND the projected \
                         gradient cleared a stationarity band (|g|={:.3e}; bound {}); accepting \
                         best-so-far as a stationary optimum (value={:.6e}).",
                        guard.rel_tol,
                        guard.window,
                        guard.best_grad_norm,
                        guard.certified_grad_bound(),
                        guard.best_value,
                    );
                    return Err(ObjectiveEvalError::fatal(COST_STALL_CONVERGED_SENTINEL.to_string()));
                }
                CostStallVerdict::FlatValleyStall { residual_grad_norm } => {
                    log::warn!(
                        "[OUTER] cost-stall FLAT-VALLEY STALL: REML objective improved < {:.3e} \
                         (relative) over {} consecutive accepted outer steps but the projected \
                         gradient is still ABOVE the outer tolerance (|g|={:.3e} > {:.3e}); \
                         halting on a weakly-identified ρ valley floor and reporting NON-CONVERGED \
                         (residual outer non-stationarity, value={:.6e}).",
                        guard.rel_tol,
                        guard.window,
                        residual_grad_norm,
                        guard.grad_threshold,
                        guard.best_value,
                    );
                    return Err(ObjectiveEvalError::fatal(COST_STALL_CONVERGED_SENTINEL.to_string()));
                }
            }
        Ok(())
    }
}

/// Outer gradient-decay ratio `‖g_now‖/‖g_initial‖` below which the outer is
/// treated as essentially converged: the inner cap is lifted entirely so the
/// cached β reaches full inner tolerance before the convergence guard runs.
pub(crate) const INNER_CAP_CONVERGENCE_OVERRIDE_RATIO: f64 = 0.01;

/// Floor on the adaptive inner-PIRLS cap. Any cap below this is below the
/// inner-Newton noise level and would reject usable warm-started steps.
pub(crate) const INNER_CAP_FLOOR: usize = 3;

/// Ceiling on the adaptive inner-PIRLS cap, set at the inner-Newton noise
/// floor at large scale; further iterations are pure waste once the warm
/// start is close.
pub(crate) const INNER_CAP_CEILING: usize = 64;

/// Adaptive inner-PIRLS cap schedule. Replaces the older hardcoded
/// iter-tier (3/5/10/20) and ratio-tier (0.50/0.20/0.05/0.01) schedule
/// with a cap driven by the inner solver's actual convergence behavior
/// — Eisenstat-Walker style for the inner Newton.
///
/// Inputs:
/// - `accepted_iters`: outer iter index, used only as a fallback when no
///   inner-progress feedback has arrived yet (first 1-2 outer iters).
/// - `g_ratio`: outer gradient-norm decay `‖g_now‖ / ‖g_initial‖`. When
///   this drops below 1% the outer is essentially converged; we lift
///   the cap fully so the cached β is at full inner tolerance and the
///   convergence guard does not have to re-pay a full inner solve.
/// - `last`: snapshot from `InnerProgressFeedback`. When present and
///   the previous solve converged, we set the cap to `last_iters + 2`
///   (a small margin in case ρ moved enough to need a couple more
///   iters); when the previous solve hit the cap, we double — a
///   geometric backoff that recovers from too-tight a cap without
///   thrashing.
///
/// A cap of 0 means "no cap from this source"; the inner solver still
/// honors `pirls_max_iterations` and the screening cap. The cap is
/// floored at 3 (anything less is below noise) and ceilinged at 64
/// (the inner noise floor at large scale; further iters would be
/// pure waste).
/// Did the inner PIRLS for the most recent non-screening outer eval converge?
///
/// Reads the inner-progress feedback snapshot the cap schedule already consumes.
/// Returns the snapshot's `last_converged` flag, defaulting to `true` when there
/// is no feedback wired (`feedback == None`) or no inner solve has reported yet
/// (`snapshot() == None`, e.g. the very first outer eval). Defaulting to `true`
/// keeps routes without inner-cap feedback — and the cold-start iterate — on
/// their existing behavior; the #1426 guard only *withholds trust* from an
/// iterate KNOWN to have a non-converged inner solve, it never invents distrust.
///
/// IMPORTANT: this must be read AFTER the outer eval has run (so the feedback
/// atomics, set by `execute_pirls_if_needed` post-solve, describe the solve at
/// the current ρ), and BEFORE the next eval's cap is computed.
pub(crate) fn inner_solve_converged(feedback: Option<&InnerProgressFeedback>) -> bool {
    match feedback.and_then(InnerProgressFeedback::snapshot) {
        Some(snap) => snap.last_converged,
        None => true,
    }
}

pub(crate) fn first_order_inner_cap_schedule(
    accepted_iters: usize,
    g_ratio: Option<f64>,
    last: Option<InnerProgressSnapshot>,
) -> usize {
    // Convergence override: when the outer is essentially converged the
    // cached β must be at full inner tolerance. This belt-and-suspenders
    // path is independent of inner-progress history because the outer
    // re-evaluation guard pays a full inner solve anyway — uncapping
    // here just avoids one wasted iter at low cap before the guard.
    if matches!(g_ratio, Some(r) if r < INNER_CAP_CONVERGENCE_OVERRIDE_RATIO) {
        return 0;
    }

    // Adaptive path: drive the cap from the inner solver's prior iter
    // count rather than a hardcoded tier.
    if let Some(snap) = last {
        let next = if snap.last_converged {
            // Converged in `last_iters` last time; pick a small margin
            // for ρ-step variability. The IFT predictor's residual
            // tells us how close the warm-start was to the KKT point:
            //   residual < 0.01  → next solve starts essentially AT the
            //                      KKT β, so +1 iter of margin suffices.
            //   residual < 0.10  → +2 (default, current behavior).
            //   residual ≥ 0.10  → predictor was poor (or fell back to
            //                      flat); the inner Newton has more
            //                      recovery work, so +4 to be safe.
            //   None             → no signal yet → +2 (default).
            // This wires the [IFT-QUALITY] feedback directly into the
            // adaptive cap, replacing the previous fixed +2.
            let mut margin = match snap.last_ift_residual {
                Some(r) if r < 0.01 => 1usize,
                Some(r) if r >= 0.10 => 4usize,
                _ => 2usize,
            };
            // LM model fidelity (commit 6445c079): if the previous
            // solve's accepted gain ratio was poor (model overstating
            // predicted reduction), the inner Newton's quadratic model
            // is unreliable. Bump margin by +2 — even a fast-converged
            // previous iter (small `last_iters`) provides weaker
            // evidence about the next solve's required effort when the
            // model is mis-calibrated. Threshold 0.5 is the textbook
            // "good agreement" cutoff for trust-region gain ratios.
            if matches!(snap.last_accept_rho, Some(r) if r < 0.5) {
                margin = margin.saturating_add(2);
            }
            snap.last_iters.saturating_add(margin)
        } else {
            // Hit the cap. Geometric backoff so we don't thrash on a
            // marginally-too-tight cap, but enforce floor of
            // last_iters+4 to actually grow.
            //
            // LM-fidelity escalation: if the previous solve's accepted
            // gain ratio was VERY poor (`accept_rho < 0.3`), the LM
            // model is severely mis-calibrated — doubling the cap may
            // not give the inner Newton enough headroom to find a
            // usable trust radius. Triple instead of doubling so we
            // don't waste another cycle hitting the cap. The 0.3
            // threshold is tighter than the +2-margin trigger (0.5)
            // because here we ALREADY know the iter budget was
            // insufficient AND the model was poor — both signals
            // pointing the same way.
            let multiplier = if matches!(snap.last_accept_rho, Some(r) if r < 0.3) {
                3
            } else {
                2
            };
            snap.last_iters
                .saturating_mul(multiplier)
                .max(snap.last_iters.saturating_add(4))
        };
        return next.clamp(INNER_CAP_FLOOR, INNER_CAP_CEILING);
    }

    // No feedback yet (first outer iter, or right after a screening
    // bundle reset). Coarse iter-count fallback for the first 1-2
    // outer iters so the cold-start cap is shallow even before the
    // adaptive signal kicks in.
    match accepted_iters {
        0 => 3,
        1 => 5,
        _ => 10,
    }
}

#[cfg(test)]
#[path = "inner_cap_schedule_tests.rs"]
mod inner_cap_schedule_tests;


/// Apply the accepted-iter inner-PIRLS cap schedule shared by the two ARC
/// bridges. `OuterFirstOrderBridge::eval_grad` and
/// `OuterSecondOrderBridge::eval_hessian` drive it from the same three fields,
/// so one body keeps the logged transition identical across both. The BFGS and
/// operator bridges use different schedule inputs and keep their own.
fn apply_arc_inner_cap_schedule(
    outer_inner_cap: Option<&InnerProgressFeedback>,
    last_g_norm: Option<f64>,
    g_norm_initial: Option<f64>,
) {
if let Some(feedback) = outer_inner_cap {
    // Use the observer-fed accepted-iter counter (opt 0.5.0
    // OptimizerObserver) instead of `eval_count / 2`; the
    // observer increments only on rho-accepted steps, so the
    // schedule no longer relaxes the cap on rejected trials.
    let arc_iter = feedback.accepted_iter.load(Ordering::Relaxed);
    let g_ratio = match (last_g_norm, g_norm_initial) {
        (Some(g), Some(g0)) if g0 > 0.0 => Some(g / g0),
        _ => None,
    };
    let snapshot = feedback.snapshot();
    let cap = first_order_inner_cap_schedule(arc_iter, g_ratio, snapshot);
    let prev = feedback.cap.swap(cap, Ordering::Relaxed);
    if prev != cap {
        let ratio_str = match g_ratio {
            Some(r) => format!("{:.3e}", r),
            None => "n/a".to_string(),
        };
        let snap_str = match snapshot {
            Some(s) => format!(
                "last_iters={} converged={} ift_residual={} accept_rho={}",
                s.last_iters,
                s.last_converged,
                match s.last_ift_residual {
                    Some(r) => format!("{:.3e}", r),
                    None => "n/a".to_string(),
                },
                match s.last_accept_rho {
                    Some(r) => format!("{:.3}", r),
                    None => "n/a".to_string(),
                },
            ),
            None => "no-history".to_string(),
        };
        log::info!(
            "[OUTER schedule] inner-PIRLS cap transition (ARC bridge) arc_iter={} g_ratio={} {} prev={} new={} ({})",
            arc_iter,
            ratio_str,
            snap_str,
            prev,
            cap,
            if cap == 0 { "uncapped" } else { "capped" }
        );
    }
}
}
pub(crate) struct OuterSecondOrderBridge<'a> {
    pub(crate) obj: &'a mut dyn OuterObjective,
    pub(crate) layout: OuterThetaLayout,
    pub(crate) hessian_source: HessianSource,
    /// When the evaluator returns `HessianValue::Operator(op)` and the
    /// operator advertises an exact dense route, the bridge may materialize the
    /// operator into a dense K×K matrix so the dense ARC path can run an exact
    /// factorization instead of operator-CG.
    pub(crate) materialize_operator_max_dim: usize,
    /// Counts gradient/Hessian evaluations so that progress is visible even
    /// when the upstream `opt` solver does not emit per-iteration logs of its
    /// own. Emitted at INFO from `eval_grad` and `eval_hessian` (the calls
    /// that gate one optimizer step); skipped on `eval_cost` so linesearch
    /// trial points do not flood the log. Also drives the outer-aware
    /// inner-PIRLS cap schedule (see `first_order_inner_cap_schedule`).
    pub(crate) eval_count: usize,
    /// Outer-aware inner-PIRLS cap atomic. When `Some`, the bridge stores
    /// a coarsen-then-tighten cap into it on every accepted eval_grad /
    /// eval_hessian call. Mirrors the BFGS-side wiring in
    /// `OuterFirstOrderBridge`. Cap is NEVER touched in `eval_cost` so
    /// line-search probes within an outer iter see a stable inner
    /// tolerance (Wolfe / trust-region acceptance both assume constant
    /// cost noise within a bracket).
    pub(crate) outer_inner_cap: Option<InnerProgressFeedback>,
    /// First observed `‖g‖` from `eval_grad`/`eval_hessian`. Used by the
    /// schedule's gradient-ratio gate so the cap lifts when the optimizer
    /// is approaching convergence, not just when iter count says so.
    pub(crate) g_norm_initial: Option<f64>,
    /// `‖g‖` from the most recent eval. See `OuterFirstOrderBridge` for
    /// the staleness rationale: monotone-decreasing g_norm means the cap
    /// is conservatively LARGER than truly needed, never smaller.
    pub(crate) last_g_norm: Option<f64>,
    /// Most recent derivative-evaluation point, used to log value-probe
    /// displacement in line-search / trial-acceptance STAGE traces.
    pub(crate) last_value_grad_rho: Option<Array1<f64>>,
    /// Cost-stall convergence guard (#1089/#1237). Identical role to the field
    /// on [`OuterFirstOrderBridge`]: `opt::Arc` calls `eval_grad` once per
    /// accepted (and rejected-then-recomputed) iterate, so folding the
    /// `(cost, projected ‖g‖)` sample in here counts outer descent steps. On a
    /// near-separable multinomial fit the unpenalized softmax MLE is unbounded,
    /// so the outer REML criterion keeps decreasing as λ→0 and several log-λ
    /// directions slam to the lower box bound and bounce — the ARC loop cycles
    /// to its `max_iter` cap without ever certifying a stationary point (#1237,
    /// the #1082 multinomial timeout). The guard halts ARC the moment the REML
    /// score stops improving, and the bound-PROJECTED KKT residual decides the
    /// converged verdict (a bound-pinned separating direction with a persistent
    /// out-of-bounds ∂V/∂ρ is stationary in the KKT sense even though its raw
    /// gradient never vanishes).
    pub(crate) cost_stall: Option<CostStallGuard>,
    /// `(lower, upper)` ρ box bounds for the bound-projected gradient norm the
    /// [`CostStallGuard`] stationarity test consumes. See the matching field on
    /// [`OuterFirstOrderBridge`].
    pub(crate) cost_stall_bounds: Option<(Array1<f64>, Array1<f64>)>,
    /// The criterion's relative resolution `outer_rel_cost_floor(config)` for
    /// the online decrement stop, or `None` on a route that does not apply it
    /// (which is every route with no synchronized analytic Hessian at the
    /// evaluated point). See [`ARC_CURVATURE_STATIONARY_SENTINEL`].
    pub(crate) curvature_stationary_floor: Option<f64>,
}

impl ZerothOrderObjective for OuterSecondOrderBridge<'_> {
    fn eval_cost(&mut self, x: &Array1<f64>) -> Result<f64, ObjectiveEvalError> {
        // Uncap the inner solve for the ARC line-search / trial-acceptance cost
        // probe. Identical rationale to `OuterFirstOrderBridge::eval_cost`: the
        // deciding cost must be the true converged-inner objective the analytic
        // gradient/Hessian differentiate, never the scheduled gradient-path cap
        // (which at a trial ρ can stop the inner solve short and report a
        // spurious `∞`, freezing the ARC at constant cost / |g| — gam#808
        // survival marginal-slope, gam#787 bernoulli matern marginal-slope).
        // `eval_grad`/`eval_hessian` restore the scheduled cap on the next call.
        if let Some(feedback) = self.outer_inner_cap.as_ref() {
            feedback
                .cap
                .store(SEED_SCREENING_UNCAPPED, Ordering::Relaxed);
        }
        self.layout
            .validate_point_len(x, "outer eval_cost failed")?;
        let trial_rho_distance = trial_rho_distance(self.last_value_grad_rho.as_ref(), x);
        let stage_start = std::time::Instant::now();
        log::info!(
            "[STAGE] outer eval start order=Value dim={} trial_rho_distance={:.3e}",
            x.len(),
            trial_rho_distance
        );
        let eval = self
            .obj
            .eval_with_order(x, OuterEvalOrder::Value)
            .map_err(|err| into_objective_error("outer eval_cost failed", err))?;
        let cost = finite_cost_or_error("outer eval_cost failed", eval.cost)?;
        log::info!(
            "[STAGE] outer eval end order=Value elapsed={:.3}s cost={:.6e} trial_rho_distance={:.3e}",
            stage_start.elapsed().as_secs_f64(),
            cost,
            trial_rho_distance
        );
        Ok(cost)
    }
}

impl FirstOrderObjective for OuterSecondOrderBridge<'_> {
    fn eval_grad(&mut self, x: &Array1<f64>) -> Result<FirstOrderSample, ObjectiveEvalError> {
        self.layout.validate_point_len(x, "outer eval failed")?;
        apply_arc_inner_cap_schedule(
            self.outer_inner_cap.as_ref(),
            self.last_g_norm,
            self.g_norm_initial,
        );
        let stage_start = std::time::Instant::now();
        log::info!(
            "[STAGE] outer eval start order=ValueAndGradient dim={}",
            x.len()
        );
        let eval = self
            .obj
            .eval_with_order(x, OuterEvalOrder::ValueAndGradient)
            .map_err(|err| into_objective_error("outer eval failed", err))?;
        let eval = finite_outer_first_order_eval_or_error("outer eval failed", self.layout, eval)?;
        self.eval_count += 1;
        let g_norm = eval.gradient.iter().map(|v| v * v).sum::<f64>().sqrt();
        if self.g_norm_initial.is_none() && g_norm.is_finite() && g_norm > 0.0 {
            self.g_norm_initial = Some(g_norm);
        }
        if g_norm.is_finite() {
            self.last_g_norm = Some(g_norm);
        }
        self.last_value_grad_rho = Some(x.clone());
        log::info!(
            "[STAGE] outer eval end order=ValueAndGradient elapsed={:.3}s cost={:.6e} |g|={:.3e}",
            stage_start.elapsed().as_secs_f64(),
            eval.cost,
            g_norm,
        );
        log::info!(
            "[OUTER] eval#{n} (grad) cost={cost:.6e} |g|={gnorm:.3e} rho=[{rho}]",
            n = self.eval_count,
            cost = eval.cost,
            gnorm = g_norm,
            rho = x
                .iter()
                .map(|v| format!("{v:.3}"))
                .collect::<Vec<_>>()
                .join(","),
        );
        // NOTE: the cost-stall guard lives in `eval_hessian` below, NOT here.
        // `opt::Arc`'s per-iterate oracle (`eval_cost_grad_hessian`) calls
        // `eval_hessian` for the (value, grad, Hessian) triple and never calls
        // this `eval_grad` on the ARC route, so folding the guard in here would
        // leave it dead and let the near-separable multinomial loop keep cycling
        // to `max_iter` (#1237). See `observe_cost_stall`.
        Ok(FirstOrderSample {
            value: eval.cost,
            gradient: eval.gradient,
        })
    }
}

impl OuterSecondOrderBridge<'_> {
    /// Fold one finite ARC oracle eval `(ρ, cost, grad)` into the cost-stall
    /// guard without allowing that first-order guard to halt the second-order
    /// solver.
    ///
    /// Cost-stall halt (#1089/#1237). `opt::Arc` evaluates the (value, gradient,
    /// Hessian) triple at every trial point — accepted or rejected — through
    /// `eval_hessian`, so observing here counts ARC's outer descent. On a
    /// near-separable multinomial fit the unpenalized softmax MLE is unbounded:
    /// the outer REML criterion keeps decreasing as λ→0, several log-λ
    /// directions slam to the lower box bound and bounce, and ARC otherwise
    /// cycles to its `max_iter` cap without certifying a stationary point. The
    /// guard halts the moment the REML score plateaus over `COST_STALL_WINDOW`
    /// evals; the converged verdict rides on the bound-PROJECTED gradient norm
    /// (KKT residual), NOT the raw `g_norm` — a bound-pinned separating
    /// direction with a persistent out-of-bounds ∂V/∂ρ is KKT-stationary even
    /// though its raw gradient never vanishes. A trial that beats the best cost
    /// resets the streak, so genuine descent never trips the halt.
    fn observe_cost_stall(
        &mut self,
        x: &Array1<f64>,
        cost: f64,
        gradient: &Array1<f64>,
        hessian: Option<&Array2<f64>>,
        hessian_psd: Option<bool>,
    ) -> Option<ObjectiveEvalError> {
        let bounds = self.cost_stall_bounds.clone();
        let separation_bound_stationary = {
            let Some(guard) = self.cost_stall.as_ref() else {
                return None;
            };
            lower_bound_outward_active_count(x, gradient, bounds.as_ref(), guard.grad_threshold)
                >= LOWER_BOUND_SEPARATION_ACTIVE_MIN
        };
        // #1426: inner-PIRLS convergence flag for the solve behind this eval (see
        // the matching read in the first-order bridge). A non-converged inner
        // solve at the λ→0 ridge reports a half-fit cost the guard must not adopt
        // as best-so-far. `None` (no feedback) defaults to `true`.
        let inner_converged = inner_solve_converged(self.outer_inner_cap.as_ref());
        let Some(guard) = self.cost_stall.as_mut() else {
            return None;
        };
        // Rail-relaxed box (#2412) — see the first-order bridge's matching
        // read. `separation_bound_stationary` above deliberately keeps the raw
        // box: it counts coordinates pinned at the LOWER bound for the
        // λ→0 separation fast-path (#1082/#1237), which is a different question
        // from whether a residual is stationary.
        let projected_g_norm = rail_projected_gradient_norm(x, gradient, bounds.as_ref());
        let verdict = if separation_bound_stationary {
            // #1426: certify the constrained-stationary fast-path with the REAL
            // bound-projected gradient norm, not a hardcoded 0.0. The projection
            // (`projected_gradient_norm`) already zeros the out-of-bounds pull on
            // the railed separation axes, so a genuine λ→0 separation
            // (#1082/#1237) still presents ‖g_proj‖ ≈ 0 and is certified
            // converged exactly as before. But a flat-valley OVERFIT that merely
            // *looks* separated on a couple of railed axes while retaining a large
            // INTERIOR (feasible-descent) gradient component — the Gamma/log λ→0
            // ridge of #1426 — keeps that interior mass in ‖g_proj‖, so
            // `publish_stall` honestly returns FlatValleyStall / converged=false
            // instead of shipping the overfit silently behind a fake zero.
            guard.observe_constrained_stationary(
                x,
                cost,
                projected_g_norm,
                inner_converged,
                hessian_psd,
            )
        } else {
            guard.observe_second_order(x, cost, projected_g_norm, inner_converged, hessian_psd)
        };
        let mut adjudicate_second_order = false;
        match verdict {
            CostStallVerdict::Continue => {}
            CostStallVerdict::StuckKeepDescending {
                residual_grad_norm,
                escape_threshold,
            } => {
                // #1426: cost flatlined but the projected gradient is far above
                // tolerance — a stuck stall, not a flat valley. Do not halt ARC.
                // Uncap the inner PIRLS so the next solves run to full tolerance
                // and the outer gradient becomes trustworthy (see the BFGS-side
                // arm for the full rationale). #2349: also raise the cold-reeval
                // pulse — on a near-separating profiled fit the stall is
                // warm-start value hysteresis, which an uncapped WARM solve does
                // not cure (it re-converges to the same warm-biased ridge point);
                // the next outer evaluations must re-solve COLD to hand the
                // optimizer a trajectory-independent surface it can descend.
                if let Some(feedback) = self.outer_inner_cap.as_ref() {
                    feedback.cap.store(0, Ordering::Relaxed);
                    feedback.force_cold.store(true, Ordering::Relaxed);
                }
                log::warn!(
                    "[OUTER] ARC cost-stall STUCK (NOT a flat valley): REML objective improved \
                     < {:.3e} (relative) over {} outer steps but the projected gradient is FAR \
                     above the certified-stationary band (|g|={:.3e} > escape threshold \
                     {:.3e}); refusing to halt-and-ship \
                     and continuing the descent (escape {}, value={:.6e}).",
                    guard.rel_tol,
                    guard.window,
                    residual_grad_norm,
                    escape_threshold,
                    guard.stuck_escapes,
                    guard.best_value,
                );
            }
            CostStallVerdict::Converged => {
                log::info!(
                    "[OUTER] ARC finite cost stall deferred to exact second-order convergence: \
                     REML objective improved < {:.3e} (relative) over {} consecutive outer \
                     steps and the stored best projected gradient is small (|g|={:.3e}; bound \
                     {}), but only ARC owns the synchronized reduced-Hessian certificate \
                     (value={:.6e}).",
                    guard.rel_tol,
                    guard.window,
                    guard.best_grad_norm,
                    guard.certified_grad_bound(),
                    guard.best_value,
                );
                guard.defer_finite_second_order_stall();
                adjudicate_second_order = true;
            }
            CostStallVerdict::FlatValleyStall { residual_grad_norm } => {
                log::warn!(
                    "[OUTER] ARC finite cost stall deferred: REML objective improved < {:.3e} \
                     (relative) over {} consecutive outer steps but the stored best projected \
                     gradient remains above tolerance (|g|={:.3e} > {:.3e}). Returning the \
                     current finite gradient + Hessian to ARC instead of halting at a stale \
                     best point (value={:.6e}).",
                    guard.rel_tol,
                    guard.window,
                    residual_grad_norm,
                    guard.grad_threshold,
                    guard.best_value,
                );
                guard.defer_finite_second_order_stall();
                adjudicate_second_order = true;
            }
        }
        // The guard's own verdict is FIRST-ORDER and, on this route, deferred:
        // only ARC holds a synchronized reduced Hessian at the point, so the
        // guard may not halt a second-order search on a gradient reading. What
        // it CAN do is say the criterion has stopped moving — and that is
        // exactly the condition under which the certificate's own
        // curvature-resolvability rung becomes the deciding test (#2817). The
        // deferral above is unchanged; what follows is the adjudication it was
        // always waiting for and never had.
        if adjudicate_second_order {
            let verdict = self.curvature_stationary_exit(x, cost, gradient, hessian, hessian_psd);
            if verdict.is_some() {
                return verdict;
            }
        }
        None
    }

    /// The certificate's own acceptance test, applied online to the point ARC
    /// has just evaluated (#2817).
    ///
    /// Returns the halt sentinel exactly when this point is one the terminal
    /// certificate would accept on its curvature-resolvability rung. Every
    /// input is the certificate's:
    ///
    /// * the reduced Hessian on the rail-relaxed free set must be PSD. That is
    ///   the certificate's own `certificate_hessian_is_psd` gate, reached here
    ///   through [`reduced_hessian_psd_at_point`] on a free set that is a
    ///   SUPERSET of the certificate's, so this can never certify curvature the
    ///   certificate would reject — and a strict saddle, where the criterion
    ///   still has a descent direction, fails it outright;
    /// * the Newton decrement `½·gᵀH⁻¹g` of the rail-projected gradient must be
    ///   finite and no larger than `floor·(1 + |V|)`. This calls
    ///   [`newton_predicted_decrease`], the same function the certificate's rung
    ///   calls, against the same tolerance, anchored at this point's own cost
    ///   exactly as the certificate anchors it at the certified point's;
    /// * the point must be the best feasible iterate the trajectory has
    ///   produced, so a trial step ARC was about to reject cannot end the run.
    ///
    /// It cannot stop a search that still has descent available. The decrement
    /// is curvature-scaled rather than a gradient threshold: a residual aligned
    /// with a near-flat Hessian direction — a linear ramp that DOES carry real
    /// descent — inflates `gᵀH⁻¹g` and is rejected, and only a residual that is
    /// small along the well-curved directions and nearly orthogonal to the flat
    /// ones passes.
    fn curvature_stationary_exit(
        &mut self,
        x: &Array1<f64>,
        cost: f64,
        gradient: &Array1<f64>,
        hessian: Option<&Array2<f64>>,
        hessian_psd: Option<bool>,
    ) -> Option<ObjectiveEvalError> {
        let floor = self.curvature_stationary_floor?;
        if hessian_psd != Some(true) || !cost.is_finite() || !floor.is_finite() || floor <= 0.0 {
            return None;
        }
        let hessian = hessian?;
        let rail_bounds = self.cost_stall_bounds.as_ref().map(rail_relaxed_bounds);
        let projected = project_gradient_vector(x, gradient, rail_bounds.as_ref());
        let decrement = super::run::newton_predicted_decrease(hessian, &projected)?;
        let tolerance = floor * (1.0 + cost.abs());
        if !decrement.is_finite() || decrement > tolerance {
            return None;
        }
        let guard = self.cost_stall.as_mut()?;
        // The incumbent test. `observe_cost_stall` has already folded this point
        // in, so the guard's best is the minimum over the trajectory INCLUDING
        // this point; `cost <= best` therefore says "this point IS the
        // incumbent" and nothing weaker. A trial the guard declined to adopt
        // (a non-converged inner solve, say) leaves the best where it was and
        // cannot end the run.
        if !(cost <= guard.best_value) {
            return None;
        }
        let projected_norm = projected.iter().map(|v| v * v).sum::<f64>().sqrt();
        log::info!(
            "[OUTER] ARC stopping at the point its own certificate accepts: \
             Newton ½gᵀH⁻¹g={decrement:.3e} ≤ criterion resolution \
             {tolerance:.3e} (= {floor:.3e}·(1+|V|) at |V|={cost:.6e}); reduced Hessian \
             PSD; |Pg|={projected_norm:.3e} after {iters} accepted outer iteration(s). The \
             absolute gradient band the solver was driven to is a different and unrelated \
             standard (#2817).",
            iters = guard.accepted_iters,
        );
        if let Ok(mut slot) = guard.exit.lock() {
            *slot = Some(CostStallExit {
                rho: x.clone(),
                value: cost,
                grad_norm: projected_norm,
                iterations: guard.accepted_iters,
                converged: true,
                // No stall window fired, so no probe-noise measurement is
                // claimed: the rung that stopped this run is the decrement.
                noise_grad_bound: None,
                probe_scale: None,
            });
        }
        Some(ObjectiveEvalError::fatal(
            ARC_CURVATURE_STATIONARY_SENTINEL.to_string(),
        ))
    }

    /// Fold one INFEASIBLE ARC trial (non-finite cost) into the cost-stall
    /// guard. Mirrors `observe_cost_stall` but routes through the guard's
    /// infeasible-streak path: a run of `COST_STALL_WINDOW` consecutive
    /// infeasible trials after a feasible best halts the outer loop at that
    /// best feasible iterate rather than letting ARC grind to `max_iter`
    /// probing the unbounded λ→0 separating region (#1082/#1237).
    fn observe_cost_stall_infeasible(&mut self, x: &Array1<f64>) -> Option<ObjectiveEvalError> {
        let guard = self.cost_stall.as_mut()?;
        match guard.observe_infeasible(x) {
            CostStallVerdict::Continue => None,
            CostStallVerdict::StuckKeepDescending {
                residual_grad_norm,
                escape_threshold,
            } => {
                // #1426: best feasible iterate carries a residual far above
                // tolerance — not a flat valley. Keep ARC descending.
                log::warn!(
                    "[OUTER] ARC cost-stall STUCK (infeasible run, NOT a flat valley): best \
                     feasible residual |g|={:.3e} far above the certified-stationary band \
                     (escape threshold {:.3e}); refusing to \
                     halt-and-ship and continuing (escape {}, value={:.6e}).",
                    residual_grad_norm,
                    escape_threshold,
                    guard.stuck_escapes,
                    guard.best_value,
                );
                None
            }
            CostStallVerdict::Converged => {
                log::warn!(
                    "[OUTER] ARC infeasible-probe stall: {} consecutive infeasible λ→0 trials \
                     after the best feasible iterate. Its stored projected gradient is small \
                     (|g|={:.3e}; bound {}) and its synchronized reduced Hessian does not \
                     certify negative curvature; halting only as a NON-CONVERGED checkpoint \
                     (value={:.6e}).",
                    guard.window,
                    guard.best_grad_norm,
                    guard.certified_grad_bound(),
                    guard.best_value,
                );
                guard.revoke_published_convergence();
                Some(ObjectiveEvalError::fatal(ARC_INFEASIBLE_STALL_SENTINEL.to_string()))
            }
            CostStallVerdict::FlatValleyStall { residual_grad_norm } => {
                log::warn!(
                    "[OUTER] ARC cost-stall halt (infeasible run): {} consecutive infeasible \
                     λ→0 trials after the best feasible iterate, whose projected gradient is \
                     still ABOVE the outer tolerance (|g|={:.3e} > {:.3e}); halting at the best \
                     feasible iterate and reporting NON-CONVERGED (value={:.6e}).",
                    guard.window,
                    residual_grad_norm,
                    guard.grad_threshold,
                    guard.best_value,
                );
                guard.revoke_published_convergence();
                Some(ObjectiveEvalError::fatal(ARC_INFEASIBLE_STALL_SENTINEL.to_string()))
            }
        }
    }
}

impl SecondOrderObjective for OuterSecondOrderBridge<'_> {
    fn eval_hessian(&mut self, x: &Array1<f64>) -> Result<SecondOrderSample, ObjectiveEvalError> {
        self.layout.validate_point_len(x, "outer eval failed")?;
        apply_arc_inner_cap_schedule(
            self.outer_inner_cap.as_ref(),
            self.last_g_norm,
            self.g_norm_initial,
        );
        let stage_start = std::time::Instant::now();
        log::info!(
            "[STAGE] outer eval start order=ValueGradientHessian dim={}",
            x.len()
        );
        let eval = self
            .obj
            .eval_with_order(x, OuterEvalOrder::ValueGradientHessian)
            .map_err(|err| into_objective_error("outer eval failed", err))?;
        // Infeasible (non-finite cost) trials are the near-separable
        // multinomial failure mode: ARC probes the unbounded λ→0 separating
        // region where the inner softmax solve does not converge. These never
        // reach `observe_cost_stall` below (validation rejects them), so feed
        // them to the guard's dedicated infeasible-streak path FIRST — a run of
        // consecutive infeasible trials after a feasible best halts the outer
        // loop at that best iterate instead of grinding to `max_iter`
        // (#1082/#1237).
        if !eval.cost.is_finite() {
            if let Some(err) = self.observe_cost_stall_infeasible(x) {
                return Err(err);
            }
        }
        let eval = finite_outer_eval_or_error("outer eval failed", self.layout, eval)?;
        self.eval_count += 1;
        let g_norm = eval.gradient.iter().map(|v| v * v).sum::<f64>().sqrt();
        if self.g_norm_initial.is_none() && g_norm.is_finite() && g_norm > 0.0 {
            self.g_norm_initial = Some(g_norm);
        }
        if g_norm.is_finite() {
            self.last_g_norm = Some(g_norm);
        }
        self.last_value_grad_rho = Some(x.clone());
        log::info!(
            "[STAGE] outer eval end order=ValueGradientHessian elapsed={:.3}s cost={:.6e} |g|={:.3e}",
            stage_start.elapsed().as_secs_f64(),
            eval.cost,
            g_norm,
        );
        log::info!(
            "[OUTER] eval#{n} (hess) cost={cost:.6e} |g|={gnorm:.3e} rho=[{rho}]",
            n = self.eval_count,
            cost = eval.cost,
            gnorm = g_norm,
            rho = x
                .iter()
                .map(|v| format!("{v:.3}"))
                .collect::<Vec<_>>()
                .join(","),
        );
        let hessian = build_bridge_hessian_for_source(
            self.hessian_source,
            eval.hessian,
            self.materialize_operator_max_dim,
        )?;
        // Rail-relaxed box here too (#2412). The strict-activity test inside
        // `reduced_hessian_psd_at_point` uses a 1e-10 proximity, which a bound
        // that is only ever approached in the limit can never satisfy — so a
        // coordinate running its smoothing parameter to the ceiling stays in
        // the free set and its still-descending direction can read indefinite,
        // blocking the guard's converged verdict.
        //
        // The gradient-sign requirement is kept, so this excludes only
        // coordinates that are BOTH within the rail margin AND pushing out of
        // the box. That leaves the guard's free set a superset of the
        // certificate's (which drops every margin-railed coordinate outright,
        // `certificate_hessian_is_psd_off_railed`), so the guard stays no more
        // permissive than the authority it answers to and can never certify
        // curvature the certificate would reject.
        let rail_bounds = self.cost_stall_bounds.as_ref().map(rail_relaxed_bounds);
        let hessian_psd = hessian.as_ref().and_then(|dense| {
            reduced_hessian_psd_at_point(
                x,
                &eval.gradient,
                dense,
                rail_bounds.as_ref().map(|(lower, upper)| (lower, upper)),
            )
        });
        // Observe finite cost progress, but never let this first-order guard
        // halt a second-order route. ARC must receive this exact sample so its
        // projected-gradient + reduced-Hessian gate can either certify a mode
        // or exploit negative curvature (#979).
        if let Some(stop) = self.observe_cost_stall(
            x,
            eval.cost,
            &eval.gradient,
            hessian.as_ref(),
            hessian_psd,
        ) {
            return Err(stop);
        }
        Ok(SecondOrderSample {
            value: eval.cost,
            gradient: eval.gradient,
            hessian,
        })
    }
}

// `OuterOperatorBridge` is the bridge that implements
// `opt::OperatorObjective` for `gam`'s outer objective — parallel to
// `OuterSecondOrderBridge` but produces `OperatorSample` whose
// Hessian is `HessianValue::Operator(_)` (or `Dense(_)` when the
// operator declares an exact materialization route).

/// `opt::OptimizerObserver` that increments
/// `InnerProgressFeedback.accepted_iter` on every accepted outer
/// step. Replaces the bridge-side `eval_count / 2` heuristic on
/// routes that see trial-and-rejection probing (ARC dense,
/// matrix-free TR). The bridge's inner-cap schedule reads
/// `accepted_iter` from the feedback channel instead of inferring
/// it from raw eval counts.
pub(crate) struct OuterAcceptObserver {
    /// Inner-PIRLS cap channel. `None` on routes that do not schedule the
    /// inner solve from the outer trajectory; the observer is still installed
    /// for [`Self::accepted_steps`] and [`Self::census`].
    pub(crate) feedback: Option<InnerProgressFeedback>,
    /// Trajectory census (#2735), read by the runner after the solver returns.
    /// `None` on routes whose summary does not report one.
    pub(crate) census: Option<Arc<OuterStepCensus>>,
    /// Accepted-outer-step ledger shared with [`OuterFirstOrderBridge`], which
    /// drains it to decide which of its own evaluations were accepted iterates
    /// (#2613). `None` on routes with no cost-stall guard.
    pub(crate) accepted_steps: Option<Arc<AcceptedStepLedger>>,
}

/// What a trust-region trajectory actually did, counted as `opt` reported it.
///
/// A walk that ends on its iteration budget hands back `final_value` and `‖g‖`
/// and nothing else, and those two cannot tell the two failures apart:
///
/// * a CRAWL — every step accepted, the radius never grown, each step buying a
///   little — which is a rate problem;
/// * a THRASH — steps rejected, the radius collapsing — which is a model
///   problem.
///
/// They need opposite repairs, and #2735 spent three issue comments inferring
/// which one it was. `opt` reports both facts per iteration through
/// [`OptimizerObserver`], at `debug`; the test harness's diagnostic backend is
/// fixed at `Info` in code and deliberately not configurable from the
/// environment, so on any run made through a test they are dark. The observer
/// is MOVED into the solver, so the counts have to leave through a shared cell
/// exactly as the accepted-step ledger's do.
#[derive(Debug, Clone, Copy)]
pub(crate) struct OuterStepCensusData {
    /// Steps `opt` accepted.
    pub(crate) accepted: usize,
    /// Steps `opt` rejected. A run with rejections shrank its radius.
    pub(crate) rejected: usize,
    /// Accepted steps that sat on the trust boundary (`‖s‖ ≥ 0.99·Δ`), which is
    /// `opt`'s own test for whether the radius — not the curvature — chose the
    /// step length. A walk whose accepted steps are all boundary-limited is
    /// being paced by its region.
    pub(crate) accepted_on_boundary: usize,
    /// Smallest and largest radius any step was taken under.
    pub(crate) radius_min: f64,
    pub(crate) radius_max: f64,
    /// Summed `f_k − f_trial` over accepted steps: what the whole walk bought.
    pub(crate) total_decrease: f64,
}

impl Default for OuterStepCensusData {
    fn default() -> Self {
        Self {
            accepted: 0,
            rejected: 0,
            accepted_on_boundary: 0,
            radius_min: f64::INFINITY,
            radius_max: 0.0,
            total_decrease: 0.0,
        }
    }
}

/// Shared cell the observer writes and the seed-loop runner reads once the
/// solver has returned. See [`OuterStepCensusData`].
#[derive(Debug, Default)]
pub(crate) struct OuterStepCensus {
    data: Mutex<OuterStepCensusData>,
}

impl OuterStepCensus {
    pub(crate) fn observe(&self, info: &StepInfo, accepted: bool) {
        let Ok(mut data) = self.data.lock() else {
            return;
        };
        if accepted {
            data.accepted += 1;
            if info.actual_decrease.is_finite() {
                data.total_decrease += info.actual_decrease;
            }
        } else {
            data.rejected += 1;
        }
        if let Some(radius) = info.trust_radius.filter(|r| r.is_finite()) {
            data.radius_min = data.radius_min.min(radius);
            data.radius_max = data.radius_max.max(radius);
            // `opt`'s own boundary test, reproduced rather than approximated:
            // the same `0.99` it uses to decide whether to grow the region.
            if accepted && info.step_norm >= 0.99 * radius {
                data.accepted_on_boundary += 1;
            }
        }
    }

    /// One line for the run summary, or `None` when nothing was observed (no
    /// step was ever taken, or no observer was installed).
    pub(crate) fn describe(&self) -> Option<String> {
        let data = self.data.lock().ok()?;
        if data.accepted == 0 && data.rejected == 0 {
            return None;
        }
        Some(format!(
            "steps accepted={} rejected={} boundary_limited={}/{} radius=[{:.3e}, {:.3e}] \
             total_decrease={:.6e}",
            data.accepted,
            data.rejected,
            data.accepted_on_boundary,
            data.accepted,
            data.radius_min,
            data.radius_max,
            data.total_decrease,
        ))
    }
}

impl OptimizerObserver for OuterAcceptObserver {
    fn on_step_accepted(&mut self, info: &StepInfo) {
        log::trace!(
            "outer step accepted iter={} step_norm={:.3e} predicted_decrease={:.3e} actual_decrease={:.3e}",
            info.iter,
            info.step_norm,
            info.predicted_decrease,
            info.actual_decrease,
        );
        if let Some(feedback) = self.feedback.as_ref() {
            feedback.accepted_iter.fetch_add(1, Ordering::Relaxed);
        }
        if let Some(ledger) = self.accepted_steps.as_ref() {
            ledger.push(AcceptedOuterStep {
                iter: info.iter,
                step_norm: info.step_norm,
                actual_decrease: info.actual_decrease,
            });
        }
        if let Some(census) = self.census.as_ref() {
            census.observe(info, true);
        }
    }

    fn on_step_rejected(&mut self, info: &StepInfo) {
        log::trace!(
            "outer step rejected iter={} step_norm={:.3e} predicted_decrease={:.3e} actual_decrease={:.3e}",
            info.iter,
            info.step_norm,
            info.predicted_decrease,
            info.actual_decrease,
        );
        if let Some(census) = self.census.as_ref() {
            census.observe(info, false);
        }
    }
}

/// One accepted outer step as `opt` reports it through
/// [`opt::OptimizerObserver::on_step_accepted`].
///
/// `StepInfo` carries no point, cost or gradient — only the two scalars below
/// plus the iteration index — so the bridge identifies WHICH of its own
/// evaluations was accepted by reconciling them against its incumbent. See
/// [`OuterFirstOrderBridge::drain_accepted_steps`].
#[derive(Debug, Clone, Copy)]
pub(crate) struct AcceptedOuterStep {
    pub(crate) iter: usize,
    /// `‖x_next − x_k‖`.
    pub(crate) step_norm: f64,
    /// `f_k − f_next`.
    pub(crate) actual_decrease: f64,
}

/// Accepted-outer-step channel from [`OuterAcceptObserver`] to
/// [`OuterFirstOrderBridge`] (#2613).
///
/// The observer and the objective are two separate values, both moved into
/// `opt::Bfgs`, so the only way for the accept signal to reach the objective's
/// cost-stall guard is a shared cell. Entries are pushed by the observer as
/// `opt` accepts steps and drained by the bridge on its next evaluation — the
/// accept fires *after* the line search that produced it, so the bridge cannot
/// consume it any earlier.
#[derive(Debug, Default)]
pub(crate) struct AcceptedStepLedger {
    steps: Mutex<Vec<AcceptedOuterStep>>,
}

impl AcceptedStepLedger {
    pub(crate) fn push(&self, step: AcceptedOuterStep) {
        if let Ok(mut steps) = self.steps.lock() {
            steps.push(step);
        }
    }

    /// Take everything queued so far, oldest first.
    pub(crate) fn drain(&self) -> Vec<AcceptedOuterStep> {
        match self.steps.lock() {
            Ok(mut steps) => std::mem::take(&mut *steps),
            Err(_) => Vec::new(),
        }
    }
}

/// One first-order outer evaluation the bridge has made and that `opt` has not
/// yet classified as accepted or rejected (#2613).
///
/// Everything the cost-stall guard needs is captured HERE, at evaluation time,
/// rather than re-derived when the accept arrives: `projected_grad_norm` needs
/// the rail-relaxed box, and `inner_converged` is a snapshot of the
/// inner-progress feedback that is only valid immediately after this ρ's solve.
#[derive(Debug, Clone)]
pub(crate) struct PendingOuterEval {
    pub(crate) rho: Array1<f64>,
    pub(crate) cost: f64,
    pub(crate) projected_grad_norm: f64,
    pub(crate) inner_converged: bool,
}

/// Cap on [`OuterFirstOrderBridge::pending_first_order`]. One BFGS iteration
/// spends at most a bracketing phase plus a 15-attempt zoom plus a
/// backtracking fallback plus a coordinate rescue; 256 is far above that and
/// bounds the memory a pathological direction can pin.
pub(crate) const PENDING_FIRST_ORDER_CAPACITY: usize = 256;

/// Relative slack when matching a pending evaluation's cost against the cost
/// reconstructed from `StepInfo.actual_decrease`.
///
/// `actual_decrease = f_k − f_next` is one subtraction of two f64s, and the
/// bridge reconstructs `f_next = incumbent − actual_decrease` with one more.
/// Each rounds at most a half-ulp of the larger operand, so eight ulps is a
/// generous two-sided envelope that still cannot collide two genuinely
/// different iterates: consecutive line-search trials on a criterion flat
/// enough to sit inside eight ulps of each other are the same point for every
/// purpose the guard has.
pub(crate) const ACCEPTED_STEP_COST_MATCH_ULPS: f64 = 8.0;

/// Bridge that exposes gam's outer objective as an
/// `opt::OperatorObjective`. Used on the matrix-free trust-region
/// route; the dense-Hessian / first-order routes still use
/// `OuterSecondOrderBridge` / `OuterFirstOrderBridge`.
pub(crate) struct OuterOperatorBridge<'a> {
    pub(crate) obj: &'a mut dyn OuterObjective,
    pub(crate) layout: OuterThetaLayout,
    /// Inner-PIRLS cap atomic, mirroring the BFGS / ARC bridges.
    pub(crate) outer_inner_cap: Option<InnerProgressFeedback>,
    /// Counts gradient/Hessian evaluations for the inner-cap schedule
    /// and progress logs.
    pub(crate) eval_count: usize,
    /// First observed `‖g‖`. Used by the inner-cap schedule's
    /// gradient-ratio gate.
    pub(crate) g_norm_initial: Option<f64>,
    /// `‖g‖` from the most recent eval.
    pub(crate) last_g_norm: Option<f64>,
    /// Most recent derivative-evaluation point, used to log value-probe
    /// displacement in line-search STAGE traces.
    pub(crate) last_value_grad_rho: Option<Array1<f64>>,
}

impl ZerothOrderObjective for OuterOperatorBridge<'_> {
    fn eval_cost(&mut self, x: &Array1<f64>) -> Result<f64, ObjectiveEvalError> {
        // Uncap the inner solve for the matrix-free TR line-search cost probe.
        // Identical rationale to the BFGS / ARC bridges: the deciding cost must
        // be the true converged-inner objective the analytic gradient/operator
        // Hessian differentiate, never the scheduled gradient-path cap (which at
        // a trial ρ can stop the inner solve short and report a spurious `∞`,
        // freezing the TR at constant cost / |g|). This is the route the
        // ψ-bearing matern bernoulli marginal-slope fit takes (gam#787);
        // `eval_value_grad_op` restores the scheduled cap on the next call.
        if let Some(feedback) = self.outer_inner_cap.as_ref() {
            feedback
                .cap
                .store(SEED_SCREENING_UNCAPPED, Ordering::Relaxed);
        }
        self.layout
            .validate_point_len(x, "outer eval_cost failed")?;
        let trial_rho_distance = trial_rho_distance(self.last_value_grad_rho.as_ref(), x);
        let stage_start = std::time::Instant::now();
        log::info!(
            "[STAGE] outer eval start order=Value dim={} trial_rho_distance={:.3e} (operator bridge)",
            x.len(),
            trial_rho_distance
        );
        let eval = self
            .obj
            .eval_with_order(x, OuterEvalOrder::Value)
            .map_err(|err| into_objective_error("outer eval_cost failed", err))?;
        let cost = finite_cost_or_error("outer eval_cost failed", eval.cost)?;
        log::info!(
            "[STAGE] outer eval end order=Value elapsed={:.3}s cost={:.6e} trial_rho_distance={:.3e} (operator bridge)",
            stage_start.elapsed().as_secs_f64(),
            cost,
            trial_rho_distance
        );
        Ok(cost)
    }
}

impl FirstOrderObjective for OuterOperatorBridge<'_> {
    fn eval_grad(&mut self, x: &Array1<f64>) -> Result<FirstOrderSample, ObjectiveEvalError> {
        self.layout.validate_point_len(x, "outer eval failed")?;
        let eval = self
            .obj
            .eval_with_order(x, OuterEvalOrder::ValueAndGradient)
            .map_err(|err| into_objective_error("outer eval failed", err))?;
        let eval = finite_outer_first_order_eval_or_error("outer eval failed", self.layout, eval)?;
        let g_norm = eval.gradient.iter().map(|v| v * v).sum::<f64>().sqrt();
        if self.g_norm_initial.is_none() && g_norm.is_finite() && g_norm > 0.0 {
            self.g_norm_initial = Some(g_norm);
        }
        if g_norm.is_finite() {
            self.last_g_norm = Some(g_norm);
        }
        self.last_value_grad_rho = Some(x.clone());
        Ok(FirstOrderSample {
            value: eval.cost,
            gradient: eval.gradient,
        })
    }
}

impl OperatorObjective for OuterOperatorBridge<'_> {
    fn eval_value_grad_op(
        &mut self,
        x: &Array1<f64>,
    ) -> Result<OperatorSample, ObjectiveEvalError> {
        self.layout.validate_point_len(x, "outer eval failed")?;
        // Drive the outer-aware inner-PIRLS cap, mirroring
        // OuterSecondOrderBridge::eval_grad / eval_hessian. Each
        // accepted outer iter calls eval_value_grad_op exactly once
        // (the matrix-free TR's inner CG uses HVPs, not full
        // evaluations), so we increment per call without the /2 the
        // ARC bridge needs.
        if let Some(feedback) = self.outer_inner_cap.as_ref() {
            let g_ratio = match (self.last_g_norm, self.g_norm_initial) {
                (Some(g), Some(g0)) if g0 > 0.0 => Some(g / g0),
                _ => None,
            };
            let snapshot = feedback.snapshot();
            let cap = first_order_inner_cap_schedule(self.eval_count, g_ratio, snapshot);
            let previous_cap = feedback.cap.swap(cap, Ordering::Relaxed);
            if previous_cap != cap {
                log::trace!("outer operator bridge updated inner cap from {previous_cap} to {cap}");
            }
        }
        let stage_start = std::time::Instant::now();
        log::info!(
            "[STAGE] outer eval start order=ValueGradientHessian dim={} (operator bridge)",
            x.len(),
        );
        let eval = self
            .obj
            .eval_with_order(x, OuterEvalOrder::ValueGradientHessian)
            .map_err(|err| into_objective_error("outer eval failed", err))?;
        let eval = finite_outer_eval_or_error("outer eval failed", self.layout, eval)?;
        self.eval_count += 1;
        let g_norm = eval.gradient.iter().map(|v| v * v).sum::<f64>().sqrt();
        if self.g_norm_initial.is_none() && g_norm.is_finite() && g_norm > 0.0 {
            self.g_norm_initial = Some(g_norm);
        }
        if g_norm.is_finite() {
            self.last_g_norm = Some(g_norm);
        }
        self.last_value_grad_rho = Some(x.clone());
        log::info!(
            "[STAGE] outer eval end elapsed={:.3}s cost={:.6e} |g|={:.3e} (operator bridge)",
            stage_start.elapsed().as_secs_f64(),
            eval.cost,
            g_norm,
        );
        Ok(OperatorSample {
            value: eval.cost,
            gradient: eval.gradient,
            hessian: eval.hessian,
        })
    }
}

// Helpers preserved across the Phase 6 rewrite. Both were previously
// shared with `run_operator_trust_region` (now deleted in favor of
// `opt::MatrixFreeTrustRegion`), but they remain in use by the dense
// ARC and BFGS arms of the seed loop.

/// Euclidean norm of the bound-PROJECTED gradient at `x` under box bounds
/// `(lower, upper)` — the KKT residual of a box-constrained minimization.
///
/// The descent direction for minimization is `-g_i`. For a component sitting on
/// its lower bound (`x_i ≤ lo_i`) the only feasible moves are upward (`+`): a
/// POSITIVE `g_i` asks for a downward step (`-g_i < 0`) that would drive `x_i`
/// below the bound — infeasible — so it is a KKT multiplier and is zeroed, while
/// a NEGATIVE `g_i` is a genuine feasible (upward) descent that must be kept; the
/// *retained* contribution is therefore `min(g_i, 0)`. Symmetrically at the upper
/// bound the feasible direction is downward, the infeasible pull is `g_i < 0`,
/// and the retained part is `max(g_i, 0)`. Interior components keep `g_i`. A
/// point is a constrained stationary optimum iff this projected norm is ~0,
/// matching `opt`'s `GradientTolerance{ projected: true }` exit. With no bounds
/// this is just `‖g‖₂`.
///
/// This matches the KKT convention used by the P-IRLS
/// [`crate::pirls::newton_solve::projected_gradient_norm`] (which drops
/// the `g_i > 0` infeasible-multiplier at an active lower bound and keeps the
/// `g_i < 0` feasible descent). An earlier version had both branches inverted —
/// it zeroed the *feasible-descent* component and kept the infeasible pull — so a
/// coordinate with a real interior descent off an active bound was reported as
/// constrained-stationary, which let the outer cost-stall guard certify a railed
/// optimum as converged (the #1074 quakes-trend / #1082 / #1426 railing).
#[inline]
/// KKT-projected gradient VECTOR at a box-constrained point.
///
/// Zeros the infeasible (bound-multiplier) component on every axis pinned at a
/// box bound, keeping only the feasible-descent part — the exact split
/// [`projected_gradient_norm`] takes the norm of. Callers that need the
/// direction (e.g. the curvature-scaled flat-valley Newton decrement in
/// `certify_outer_optimality`) consume this; `projected_gradient_norm` is its
/// Euclidean norm.
pub(crate) fn project_gradient_vector(
    x: &Array1<f64>,
    gradient: &Array1<f64>,
    bounds: Option<&(Array1<f64>, Array1<f64>)>,
) -> Array1<f64> {
    match bounds {
        Some((lower, upper)) => Array1::from_iter((0..gradient.len()).map(|i| {
            let gi = gradient[i];
            // Active lower bound: feasible moves are upward, so a positive g_i
            // (its downward step `-g_i` exits the box) is the infeasible
            // KKT-multiplier pull → drop it, keeping the feasible-descent
            // negative part.
            let gi = if x[i] <= lower[i] { gi.min(0.0) } else { gi };
            // Active upper bound: feasible moves are downward, so a negative g_i
            // (its upward step `-g_i` exits the box) is the infeasible pull →
            // drop it, keeping the feasible-descent positive part.
            if x[i] >= upper[i] { gi.max(0.0) } else { gi }
        })),
        None => gradient.clone(),
    }
}

pub(crate) fn projected_gradient_norm(
    x: &Array1<f64>,
    gradient: &Array1<f64>,
    bounds: Option<&(Array1<f64>, Array1<f64>)>,
) -> f64 {
    project_gradient_vector(x, gradient, bounds)
        .iter()
        .map(|v| v * v)
        .sum::<f64>()
        .sqrt()
}

/// The search box relaxed inward by [`CERTIFICATE_RAIL_MARGIN`] — the box a
/// projected *stationarity residual* must be measured against.
///
/// A penalty creeping toward the ±rho_bound infinite-smoothing ceiling never
/// reaches it exactly: each outer step only shrinks the gap, so it lands
/// strictly inside the box (the #2299 checkpoint sat at ρ=29.9938, not 30).
/// The rail detector flags such a coordinate railed by margin, but the exact
/// `x >= upper` / `x <= lower` test in [`project_gradient_vector`] reads it as
/// interior and keeps its outward pull in the residual.
///
/// That disagreement is only safe if both layers make it. They did not: the
/// terminal certificate relaxed the box (#2299) while every consumer inside
/// the search projected against the raw one, so the cost-stall guard measured
/// a residual the certificate would have discarded, never reached a stationary
/// verdict, and let the solver run to its iteration cap on points the
/// certificate goes on to accept (#2412). Route every projected-residual
/// consumer through this one box so "railed" means one thing.
///
/// Relaxing cannot certify a non-optimum: [`project_gradient_vector`] zeros
/// only the OUTWARD half (`.max(0.0)` / `.min(0.0)`), so a near-bound
/// coordinate that still has feasible-descent gradient keeps it and still
/// registers as a stationarity residual.
///
/// The per-coordinate margin is [`coordinate_rail_margin`], which is also what
/// the certificate's rail flag tests against, so the relaxed endpoints and the
/// flag are the same statement about the same coordinate.
pub(crate) fn rail_relaxed_bounds(
    bounds: &(Array1<f64>, Array1<f64>),
) -> (Array1<f64>, Array1<f64>) {
    let (lower, upper) = bounds;
    let margins: Vec<f64> = lower
        .iter()
        .zip(upper.iter())
        .map(|(lo, hi)| coordinate_rail_margin(*lo, *hi))
        .collect();
    (
        Array1::from_iter(lower.iter().zip(&margins).map(|(v, m)| v + m)),
        Array1::from_iter(upper.iter().zip(&margins).map(|(v, m)| v - m)),
    )
}

/// The inward rail margin for ONE coordinate — the single definition of "close
/// enough to a bound to count as railed", shared by [`rail_relaxed_bounds`] and
/// by the certificate's rail flag (`outer_coordinate_is_railed`). "Railed" is
/// therefore exactly `x <= lower + margin || x >= upper - margin`, and the two
/// layers cannot disagree about a coordinate.
///
/// The margin is capped at a quarter of the coordinate's own width. Without the
/// cap a box narrower than twice [`CERTIFICATE_RAIL_MARGIN`] has its two margin
/// bands cover the WHOLE interval, so every feasible point reads railed at both
/// ends — including the exact centre. That is not a conservative reading in
/// either consumer. The residual projector would zero the coordinate's pull
/// outright, and the certificate deletes railed rows and columns from its
/// reduced Hessian, so a fully-covered box leaves an empty interior sub-block
/// whose second-order condition passes vacuously: a silent false certification
/// on exactly the tightly-boxed coordinates that most need a real one.
///
/// Narrow boxes are the normal case for the non-ρ blocks a joint search carries
/// in the same θ vector. A constant-curvature term's raw-κ window is
/// `±CONSTANT_CURVATURE_KAPPA_CHART_FRACTION / R²` (width `1/R²`), so any data
/// whose farthest point sits at squared chart radius `R² ≥ 1` — every
/// standardised feature set — has a κ window at most `2 ×
/// CERTIFICATE_RAIL_MARGIN` wide, and an absolute margin would flag flat κ = 0
/// railed. κ = 0 is the *centre* of that window and the interior point the raw-κ
/// coordinate exists to keep reachable (#2462).
///
/// The cap keeps at least the middle half of every box interior, so the centre
/// is never railed and a relaxed interval can never invert. A fixed or
/// degenerate coordinate (`upper <= lower`, or a NaN width) gets no relaxation
/// and stays on the exact bound test.
pub(crate) fn coordinate_rail_margin(lower: f64, upper: f64) -> f64 {
    let quarter_width = (upper - lower) * 0.25;
    if quarter_width > 0.0 {
        CERTIFICATE_RAIL_MARGIN.min(quarter_width)
    } else {
        0.0
    }
}

/// Projected stationarity-residual norm measured against [`rail_relaxed_bounds`].
///
/// Every comparison of a projected residual against a stationarity bound must
/// go through here rather than projecting against the raw box, so the search
/// loop and the terminal certificate cannot disagree about whether a railed
/// coordinate contributes. Bound-free problems are unaffected.
pub(crate) fn rail_projected_gradient_norm(
    x: &Array1<f64>,
    gradient: &Array1<f64>,
    bounds: Option<&(Array1<f64>, Array1<f64>)>,
) -> f64 {
    match bounds {
        Some(bounds) => projected_gradient_norm(x, gradient, Some(&rail_relaxed_bounds(bounds))),
        None => projected_gradient_norm(x, gradient, None),
    }
}

/// Apply the same strict-complementarity critical-cone reduction used by the
/// optimizer, then adjudicate second-order stationarity with the final outer
/// certificate's PSD rule. This keeps the infeasible-stall guard from treating
/// an interior (or weak-bound) strict saddle as a terminal neighbourhood.
pub(crate) fn reduced_hessian_psd_at_point(
    x: &Array1<f64>,
    gradient: &Array1<f64>,
    hessian: &Array2<f64>,
    bounds: Option<(&Array1<f64>, &Array1<f64>)>,
) -> Option<bool> {
    let n = x.len();
    if gradient.len() != n || hessian.nrows() != n || hessian.ncols() != n {
        return None;
    }
    if bounds.is_some_and(|(lower, upper)| lower.len() != n || upper.len() != n) {
        return None;
    }
    let free: Vec<usize> = (0..n)
        .filter(|&index| {
            let Some((lower, upper)) = bounds else {
                return true;
            };
            let fixed = lower[index] == upper[index];
            let strict_lower = x[index] <= lower[index] + 1.0e-10 && gradient[index] > 0.0;
            let strict_upper = x[index] >= upper[index] - 1.0e-10 && gradient[index] < 0.0;
            !(fixed || strict_lower || strict_upper)
        })
        .collect();
    if free.is_empty() {
        return Some(true);
    }
    let reduced = Array2::from_shape_fn((free.len(), free.len()), |(row, column)| {
        hessian[[free[row], free[column]]]
    });
    certificate_hessian_is_psd(&reduced)
}

#[cfg(test)]
#[path = "projected_gradient_tests.rs"]
mod projected_gradient_tests;

pub(crate) const LOWER_BOUND_SEPARATION_ACTIVE_MIN: usize = 2;

/// Count log-precision axes pinned at their lower box bound while the raw
/// gradient still points farther out of bounds. On near-separable softmax fits,
/// those λ→0 axes can keep lowering the raw REML score even though the move is
/// infeasible; once several such axes are active, repeated ARC trials there are
/// constrained-stationary separation probes, not useful descent.
///
/// Sign convention (mirrors the KKT split in [`projected_gradient_norm`], fixed
/// in a14b712 for the #1074/#1082 railing class): the optimizer MINIMIZES the
/// cost and `gradient` is ∂cost/∂ρ, so at an active lower bound the descent step
/// `-g_i` exits the box exactly when `g_i > 0` — a POSITIVE gradient is the
/// infeasible outward/separation pull this counts. A NEGATIVE `g_i` is feasible
/// interior descent (kept by `projected_gradient_norm`'s `g_i.min(0.0)`) and is
/// NOT outward, so it must NOT be counted; doing so (the prior `< -outward_floor`
/// inversion) certified railed/under-fit axes that still had real descent as
/// separation-stationary.
#[inline]
pub(crate) fn lower_bound_outward_active_count(
    x: &Array1<f64>,
    gradient: &Array1<f64>,
    bounds: Option<&(Array1<f64>, Array1<f64>)>,
    grad_threshold: f64,
) -> usize {
    let Some((lower, _upper)) = bounds else {
        return 0;
    };
    let tol = 1.0e-10;
    let outward_floor = grad_threshold.max(COST_STALL_PROJECTED_GRAD_FLOOR);
    (0..x.len().min(gradient.len()).min(lower.len()))
        .filter(|&i| x[i] <= lower[i] + tol && gradient[i] > outward_floor)
        .count()
}

#[cfg(test)]
#[path = "lower_bound_outward_tests.rs"]
mod lower_bound_outward_tests;

#[inline]
pub(crate) fn project_to_bounds(
    x: &Array1<f64>,
    bounds: Option<&(Array1<f64>, Array1<f64>)>,
) -> Array1<f64> {
    match bounds {
        Some((lower, upper)) => {
            let mut out = x.clone();
            for idx in 0..out.len() {
                out[idx] = out[idx].clamp(lower[idx], upper[idx]);
            }
            out
        }
        None => x.clone(),
    }
}

/// Translate an `OuterEval`'s Hessian into the `Option<Array2<f64>>`
/// shape expected by `opt::SecondOrderSample`, enforcing the contract
/// implied by the planner's `HessianSource`.
///
/// For `HessianSource::Analytic` (the exact second-order route) a missing
/// or non-materializable Hessian is FATAL: returning `None` here would
/// invite `opt::SecondOrderCache::finite_difference_hessian` to silently
/// estimate the Hessian by finite-differencing the gradient, which (a)
/// throws away the analytic structure the route was selected for, and
/// (b) costs O(K) full outer evaluations per ARC iteration — at large-scale
/// scale, hours of work per silently-mis-routed step. The right
/// behavior on a planner/runtime mismatch is to surface it loudly so
/// the seed loop can either retry, demote the plan, or fail the seed.
///
/// Operator Hessians that *are* cheaply materializable (the operator's
/// `materialization` reports `Explicit` / `BatchedHvp` and the
/// dimension is below `materialize_operator_max_dim`) are converted to
/// dense in-place so dense ARC can run an exact factorization. Operator
/// Hessians that are NOT cheaply materializable should never arrive
/// here: the seed loop routes those to `run_operator_trust_region`
/// before constructing the bridge. Reaching this branch on the analytic
/// route means the runtime contradicted the seed-time decision, which
/// is the same kind of mismatch we treat as fatal.
///
/// For `HessianSource::BfgsApprox`, `EfsFixedPoint`, and
/// `HybridEfsFixedPoint` we deliberately return `None`: those routes do
/// not consume an analytic Hessian and feed the Hessian into a
/// quasi-Newton/fixed-point update instead. (Today these `HessianSource`
/// variants don't actually drive `opt`'s second-order solvers, but the
/// match preserves the original behavior in case a future routing
/// reuses this bridge.)
pub(crate) fn build_bridge_hessian_for_source(
    source: HessianSource,
    hessian: HessianValue,
    materialize_operator_max_dim: usize,
) -> Result<Option<Array2<f64>>, ObjectiveEvalError> {
    match source {
        HessianSource::Analytic => match hessian {
            HessianValue::Dense(h) => Ok(Some(h)),
            HessianValue::Operator(op)
                if op.materialization().is_available()
                    && op.dim() <= materialize_operator_max_dim =>
            {
                op.materialize_dense()
                    .map(Some)
                    .map_err(|error| ObjectiveEvalError::fatal(format!("outer Hessian operator materialization failed: {error}")))
            }
            HessianValue::Operator(op) => Err(ObjectiveEvalError::fatal(format!(
                    "outer plan declared HessianSource::Analytic but the runtime returned a \
                     non-materializable Hessian operator (dim={}, materialization={:?}); \
                     finite-difference Hessian estimation is not permitted on the analytic route",
                    op.dim(),
                    op.materialization(),
                ))),
            HessianValue::Unavailable => Err(ObjectiveEvalError::fatal("outer plan declared HessianSource::Analytic but the runtime returned \
                          HessianValue::Unavailable; finite-difference Hessian estimation is \
                          not permitted on the analytic route"
                    .to_string())),
        },
        HessianSource::BfgsApprox
        | HessianSource::EfsFixedPoint
        | HessianSource::HybridEfsFixedPoint => Ok(None),
    }
}

pub(crate) struct OuterFixedPointBridge<'a> {
    pub(crate) obj: &'a mut dyn OuterObjective,
    pub(crate) layout: OuterThetaLayout,
    pub(crate) barrier_config: Option<BarrierConfig>,
    pub(crate) fixed_point_tolerance: f64,
    /// Exact coefficient state produced by the most recent finite EFS
    /// evaluation, bound to the outer coordinate that produced it.
    ///
    /// The fixed-point driver owns this bridge while it runs.  Publishing the
    /// pair through a shared slot lets the runner preserve the inner basin when
    /// a later trial rho is refused and the analytic-gradient fallback resumes
    /// from the last finite fixed-point incumbent.
    pub(crate) evaluated_inner_seed: Arc<Mutex<Option<BoundInnerSeed>>>,
    /// Consecutive HybridEFS iterations whose ψ block was zeroed after
    /// exhausting backtracking. When this reaches
    /// [`MAX_CONSECUTIVE_PSI_STAGNATION`], the bridge surfaces the
    /// typed [`FirstOrderFallbackRequest`] so the runner aborts the
    /// HybridEFS attempt and the fallback ladder routes to a joint
    /// gradient-based solver where ψ stationarity ∇_ψ V = 0 can be enforced.
    pub(crate) consecutive_psi_zero_iters: usize,
    /// Restore streak reported by the previous fixed-point evaluation.  Keeping
    /// this in the bridge requires the certificate to recur across two distinct
    /// outer evaluations; multiple inner-refinement chunks at one rho cannot
    /// terminate the outer walk by themselves.
    pub(crate) last_restored_incumbent_streak: Option<usize>,
    /// Publication slot for the recurrent-restored-incumbent stop (#2235
    /// verdict 2). The bridge is moved into the `opt::FixedPoint` driver, so
    /// when the model-state fixed point fires the streak count is handed back
    /// to `run_fixed_point_outer_solver` through this shared cell and stamped
    /// onto the returned [`OuterResult`] as
    /// [`OuterConvergedVia::RecurrentIncumbent`]. `None` slot after the run
    /// means the walk stopped through the ordinary step-norm test instead.
    pub(crate) recurrent_incumbent_exit: Arc<Mutex<Option<usize>>>,
}

impl OuterFixedPointBridge<'_> {
    fn reject_nonstationary_tiny_psi_step(
        &self,
        step: &Array1<f64>,
        psi_indices: Option<&[usize]>,
        psi_gradient: Option<&Array1<f64>>,
        cost: f64,
    ) -> Result<(), ObjectiveEvalError> {
        let Some(psi_indices) = psi_indices else {
            return Ok(());
        };
        let Some(psi_gradient) = psi_gradient else {
            return Ok(());
        };
        let psi_step_inf = psi_indices
            .iter()
            .map(|&idx| step[idx].abs())
            .fold(0.0_f64, f64::max);
        let psi_grad_inf = psi_gradient.iter().map(|v| v.abs()).fold(0.0_f64, f64::max);
        if psi_step_inf <= self.fixed_point_tolerance && psi_grad_inf > self.fixed_point_tolerance {
            return Err(first_order_fallback_error(format!(
                "HybridEFS ψ nonstationary: ||Δψ||∞={:.3e} <= tol={:.3e} \
                 but raw ||gψ||∞={:.3e} (rho_dim={}, psi_dim={}, n_params={}, cost={:.6e})",
                psi_step_inf,
                self.fixed_point_tolerance,
                psi_grad_inf,
                self.layout.rho_dim(),
                self.layout.psi_dim,
                self.layout.n_params,
                cost,
            )));
        }
        Ok(())
    }
}

/// Maximum number of α halvings for the cost line search wrapping the EFS
/// step.
///
/// The Wood–Fasiolo paper proves that the EFS update direction is an *ascent
/// direction* for REML/LAML on penalty-like coordinates, but full-step
/// monotonicity is not guaranteed — both the original Fellner–Schall paper
/// and the extension recommend step-length control. We backtrack the entire
/// θ vector by halving α ∈ {1, 1/2, …, 1/2⁸ ≈ 0.004}, accepting the first
/// trial point with a strictly lower cost. With 8 halvings the smallest
/// trial step is ≈ 0.4% of the raw EFS step in every coordinate, which is
/// enough to clear pathologies near the identifiability boundary while
/// staying inside one cache-warm Hessian factorization budget.
pub(crate) const MAX_EFS_BACKTRACK: usize = 8;

/// Step components below this threshold (in θ-space) are treated as zero
/// for backtracking purposes — there is no point line-searching a step of
/// magnitude `1e-12`, and skipping the trial keeps the convergence path
/// numerically clean (no spurious cost decreases from ULP noise).
pub(crate) const EFS_NEGLIGIBLE_STEP: f64 = 1e-12;

/// Maximum infinity-norm of the EFS step (in θ-space) at which we skip the
/// cost line search and trust the multiplicative formula's quadratic
/// convergence. Above this, we always backtrack.
///
/// At small step magnitudes the canonical formula `Δρ = log((d−t)/q_eff)`
/// is itself a Newton step on the REML stationarity equation, with
/// quadratic local convergence. Under Wood–Fasiolo's Loewner-order
/// assumptions on the penalty derivative, sufficiently small steps are
/// always descent on `V`, so the line search would add an inner P-IRLS
/// solve per outer iteration with essentially zero chance of finding a
/// halving that beats the full step. The threshold is set to ~exp(0.5)
/// ≈ 1.65× change in any single λ_i (well inside the local-convergence
/// regime) and gates only the line-search call — the step itself is
/// applied unchanged, so correctness is preserved.
pub(crate) const EFS_LINESEARCH_THRESHOLD: f64 = 0.5;

/// Relative tolerance for the descent condition `c < current_cost` during
/// EFS backtracking. Without this, ULP-level cost noise near a fixed point
/// can cause spurious backtracking even when the step is mathematically
/// correct. We accept any trial whose cost is within
/// `EFS_COST_DESCENT_TOL · |current_cost|` of the current value.
pub(crate) const EFS_COST_DESCENT_TOL: f64 = 1e-12;

/// Maximum number of consecutive HybridEFS iterations whose ψ block was
/// zeroed before the bridge bails out and triggers a solver switch.
///
/// On hard problems (Matérn additive at large scale, Duchon60, anisotropic
/// joint penalties) a single zeroed-ψ iteration after exhausted backtracking
/// is already strong evidence the EFS ψ direction is not descent-correlated
/// at the current iterate; continuing on ρ alone with Δψ = 0 cannot enforce
/// ∇_ψ V = 0 and burns outer iterations on a non-stationary direction.
/// Bail out immediately so the fallback ladder routes to a joint
/// gradient-based solver (BFGS / L-BFGS) where ψ stationarity is part of
/// the optimality condition.
pub(crate) const MAX_CONSECUTIVE_PSI_STAGNATION: usize = 1;

impl FixedPointObjective for OuterFixedPointBridge<'_> {
    fn eval_step(&mut self, x: &Array1<f64>) -> Result<FixedPointSample, ObjectiveEvalError> {
        self.layout.validate_point_len(x, "outer EFS eval failed")?;
        let eval = match self.obj.eval_efs(x) {
            Ok(eval) => eval,
            Err(err @ EstimationError::GradientUnavailable { .. })
                if self.obj.capability().gradient == Derivative::Analytic =>
            {
                log::warn!(
                    "[STAGE] EFS -> gradient fallback: gradient unavailable at \
                     fixed-point dispatch; retrying with fixed-point disabled \
                     (rho_dim={}, psi_dim={}, n_params={})",
                    self.layout.rho_dim(),
                    self.layout.psi_dim,
                    self.layout.n_params,
                );
                return Err(first_order_fallback_error(format!(
                    "outer EFS eval failed: {err}"
                )));
            }
            // A REML/inner-solve failure is POINT-LOCAL evidence about this ρ —
            // another seed can solve fine — so it must surface as recoverable:
            // at seed evaluation the driver maps it to SeedRejected and the
            // seed cascade tries the next candidate without spending the
            // rejected slot's budget (#2367). Blanket-classifying it fatal
            // turned one invalid startup seed into a whole-fit
            // "Fatal outer-objective evaluation failure". Structural errors
            // (dimension mismatches, invalid layouts) remain fatal below.
            Err(err @ EstimationError::RemlOptimizationFailed(_)) => {
                return Err(
                    ObjectiveEvalError::recoverable_from(err).with_context("outer EFS eval failed")
                );
            }
            Err(err) => return Err(into_objective_error("outer EFS eval failed", err)),
        };
        self.layout
            .validate_efs_eval(&eval, "outer EFS eval failed")?;
        if !eval.cost.is_finite() {
            return Err(ObjectiveEvalError::recoverable(
                "outer EFS eval failed: objective returned a non-finite cost".to_string(),
            ));
        }
        // Reject non-finite EFS step components at the bridge boundary with
        // full diagnostic context (which coord, its value, and whether it is
        // a ρ or ψ coord). Without this, a NaN/Inf step flows into the
        // hybrid-EFS backtrack loop, which halves it via `NaN * 0.5^k = NaN`
        // until backtracking exhausts, then silently zeros the ψ block and
        // applies only the ρ step — masking the analytic-gradient bug that
        // produced the NaN. The opt crate's FixedPoint::run also detects
        // this downstream (opt 0.2.2 lib.rs:4949) but surfaces only the bare
        // `NonFiniteStep` variant with no context, which is not actionable.
        if let Some((idx, value)) = eval.steps.iter().enumerate().find(|(_, v)| !v.is_finite()) {
            let psi_indices = eval.psi_indices.as_deref();
            let coord_kind = match psi_indices {
                Some(indices) if indices.contains(&idx) => "ψ",
                Some(_) => "ρ/τ",
                None => "ρ",
            };
            return Err(ObjectiveEvalError::recoverable(format!(
                "outer EFS eval failed: non-finite {coord_kind} step at coord {idx} \
                 (step[{idx}]={value}, rho_dim={}, psi_dim={}, n_params={}, cost={:.6e})",
                self.layout.rho_dim(),
                self.layout.psi_dim,
                self.layout.n_params,
                eval.cost,
            )));
        }
        *self
            .evaluated_inner_seed
            .lock()
            .expect("fixed-point inner-state publication lock poisoned") = eval
            .beta
            .as_ref()
            .filter(|beta| beta.iter().all(|value| value.is_finite()))
            .map(|beta| BoundInnerSeed {
                theta: x.clone(),
                beta: beta.clone(),
            });
        if let Some(ref barrier_cfg) = self.barrier_config
            && let Some(ref beta) = eval.beta
        {
            // Scale-free precondition check for EFS. Wood–Fasiolo's
            // multiplicative log-λ update is derived under the
            // assumption that the inner Hessian is ≈ X'WX + S. A log
            // barrier adds τ/(β_j−l_j)² to the Hessian diagonal at the
            // constrained coords; when the tightest slack is much
            // smaller than the typical slack, that diagonal becomes
            // locally dominant and the EFS direction is no longer
            // guaranteed-ascent. Comparing slack *ratios* is
            // dimensionless — independent of τ, β scale, and the
            // inner-Hessian magnitude — which is exactly the regime
            // change EFS cannot represent. The earlier criterion
            // `barrier_curvature_is_significant(β, ref_diag=1.0, 0.01)`
            // was dimensionful and depended on three quantities the
            // bridge has no way to set correctly.
            //
            // Two principled triggers, each catching a distinct
            // failure mode of the EFS precondition:
            //  • `ratio = 0.1`        — asymmetric concentration:
            //    the worst slack is ≥10× tighter than the median.
            //    Catches the common "one coefficient hits its bound
            //    while others stay healthy" case.
            //  • `saturation = 1.0`   — absolute saturation:
            //    `max_j τ/Δ_j² ≥ 1`, i.e. at least one barrier-
            //    diagonal entry has reached the natural unit penalty
            //    scale. Catches the symmetric near-boundary regime
            //    that ratio-only checks would let through (median Δ
            //    also small, so min/median ratio stays near 1, but
            //    EFS's "ignore the barrier diagonal" assumption is
            //    still violated everywhere on the active set).
            const LOCAL_CONCENTRATION_RATIO: f64 = 0.1;
            const BARRIER_CURVATURE_SATURATION: f64 = 1.0;
            const BARRIER_CURVATURE_RELATIVE_THRESHOLD: f64 = 0.05;
            if let Some(hessian_scale) = eval.inner_hessian_scale
                && hessian_scale.is_finite()
                && hessian_scale > 0.0
                && barrier_cfg.barrier_curvature_is_significant(
                    beta,
                    hessian_scale,
                    BARRIER_CURVATURE_RELATIVE_THRESHOLD,
                )
            {
                return Err(first_order_fallback_error(format!(
                    "EFS barrier curvature significant relative to inner Hessian \
                         (rho_dim={}, psi_dim={}, n_params={}, cost={:.6e}, ref_diag={:.3e})",
                    self.layout.rho_dim(),
                    self.layout.psi_dim,
                    self.layout.n_params,
                    eval.cost,
                    hessian_scale,
                )));
            }
            if barrier_cfg.barrier_curvature_locally_concentrated(
                beta,
                LOCAL_CONCENTRATION_RATIO,
                BARRIER_CURVATURE_SATURATION,
            ) {
                return Err(first_order_fallback_error(format!(
                    "EFS barrier curvature locally concentrated \
                         (rho_dim={}, psi_dim={}, n_params={}, cost={:.6e})",
                    self.layout.rho_dim(),
                    self.layout.psi_dim,
                    self.layout.n_params,
                    eval.cost,
                )));
            }
        }
        let status = FixedPointStatus::Continue;

        let raw_step = Array1::from_vec(eval.steps);
        let psi_indices = eval.psi_indices.clone();
        self.reject_nonstationary_tiny_psi_step(
            &raw_step,
            psi_indices.as_deref(),
            eval.psi_gradient.as_ref(),
            eval.cost,
        )?;
        let max_step_abs = raw_step.iter().map(|s| s.abs()).fold(0.0_f64, f64::max);
        let current_cost = eval.cost;
        // #2241 — an objective may certify that consecutive inner solves
        // returned to the same banked incumbent after non-monotone boundary
        // mutations. One restore is merely a repair; a second consecutive
        // restore is the first possible recurrence and proves that another outer
        // update did not change the fitted state. Terminate on that model-state
        // certificate instead of guessing flatness from a relative-cost scalar
        // or exhausting an iteration budget. The current rho/cost is retained;
        // `FixedPointStatus::Stop` prevents the non-stationary proposal from
        // being applied.
        let restored_incumbent_recurred = match (
            self.last_restored_incumbent_streak,
            eval.consecutive_restored_incumbents,
        ) {
            (Some(previous), Some(current)) => current > previous && current > 1,
            _ => false,
        };
        self.last_restored_incumbent_streak = eval.consecutive_restored_incumbents;
        if restored_incumbent_recurred {
            let restores = eval.consecutive_restored_incumbents.unwrap_or_default();
            if psi_indices.is_some() {
                self.consecutive_psi_zero_iters = 0;
            }
            if let Ok(mut slot) = self.recurrent_incumbent_exit.lock() {
                *slot = Some(restores);
            }
            log::info!(
                "[OUTER] fixed-point convergence by recurrent restored incumbent: \
                 consecutive_restores={} cost={current_cost:.6e} rho_dim={} psi_dim={}",
                restores,
                self.layout.rho_dim(),
                self.layout.psi_dim,
            );
            return Ok(FixedPointSample {
                value: current_cost,
                step: raw_step,
                status: FixedPointStatus::Stop,
            });
        }
        if self.fixed_point_step_converged(x, &raw_step, psi_indices.as_deref()) {
            if psi_indices.is_some() {
                self.consecutive_psi_zero_iters = 0;
            }
            return Ok(FixedPointSample {
                value: current_cost,
                step: raw_step,
                status: FixedPointStatus::Stop,
            });
        }

        // Negligible raw step — the iteration is at (or numerically
        // indistinguishable from) a fixed point. Pass it through so the
        // outer step-norm convergence check fires; no point evaluating the
        // cost at x + 1e-30·s to chase ULP-level "improvements".
        if max_step_abs < EFS_NEGLIGIBLE_STEP {
            if psi_indices.is_some() {
                self.consecutive_psi_zero_iters = 0;
            }
            return Ok(FixedPointSample {
                value: current_cost,
                step: raw_step,
                status,
            });
        }

        // Small-step fast path. The canonical Wood–Fasiolo formula is
        // locally quadratically convergent, so once we are inside the
        // multiplicative-Newton basin (`||Δθ||∞ < EFS_LINESEARCH_THRESHOLD`)
        // a halving is essentially never accepted over the full step. Skip
        // the inner P-IRLS solve we'd otherwise burn on backtracking. When a
        // barrier is configured, every accepted rho-step must still pass
        // through the barrier-aware cost because feasibility can change even
        // under a small smoothing-parameter move. For hybrid runs we still
        // need to reset the ψ-stagnation counter.
        if self.barrier_config.is_none() && max_step_abs < EFS_LINESEARCH_THRESHOLD {
            if psi_indices.is_some() {
                self.consecutive_psi_zero_iters = 0;
            }
            return Ok(FixedPointSample {
                value: current_cost,
                step: raw_step,
                status,
            });
        }

        // ── Stage 1: full-vector cost backtracking ──
        //
        // Wood–Fasiolo gives ascent in the EFS direction but not full-step
        // monotonicity, so backtrack α ∈ {1, 1/2, …} on the *whole* step
        // vector (not just ψ). This is a uniform requirement: even on the
        // pure-ρ path, the additive log-λ formula is exact only at the
        // fixed point and is otherwise just a Newton-flavoured Wood–Fasiolo
        // surrogate that benefits from line search at large iterations.
        if let Some(scaled) = self.efs_backtrack(x, &raw_step, current_cost, MAX_EFS_BACKTRACK)? {
            if psi_indices.is_some() {
                self.consecutive_psi_zero_iters = 0;
            }
            return Ok(FixedPointSample {
                value: current_cost,
                step: scaled,
                status,
            });
        }

        // ── Stage 2 (hybrid only): ψ-zeroed retry ──
        //
        // Full-vector backtracking exhausted means *every* α we tried gave
        // a worse cost. On the hybrid path, the most common cause is a
        // bad ψ direction polluting an otherwise-good ρ step (preconditioned
        // gradient step on a near-singular ψ-ψ Gram matrix overshoots).
        // Try the ρ/τ block alone with the same backtracking schedule. If
        // that succeeds, we make progress on ρ this iteration; the ψ
        // stagnation counter advances and triggers the joint-solver
        // fallback once it crosses MAX_CONSECUTIVE_PSI_STAGNATION.
        if let Some(psi_idx) = psi_indices.as_ref() {
            let mut rho_only = raw_step.clone();
            for &i in psi_idx {
                rho_only[i] = 0.0;
            }
            let max_rho_abs = rho_only.iter().map(|s| s.abs()).fold(0.0_f64, f64::max);
            if max_rho_abs >= EFS_NEGLIGIBLE_STEP
                && let Some(scaled) =
                    self.efs_backtrack(x, &rho_only, current_cost, MAX_EFS_BACKTRACK)?
            {
                self.consecutive_psi_zero_iters = self.consecutive_psi_zero_iters.saturating_add(1);
                log::info!(
                    "[HYBRID-EFS] full-vector backtrack exhausted; ρ/τ-only step \
                         accepted. Consecutive ψ-zero iters = {}",
                    self.consecutive_psi_zero_iters,
                );
                if self.consecutive_psi_zero_iters >= MAX_CONSECUTIVE_PSI_STAGNATION {
                    log::info!(
                        "[STAGE] HybridEFS -> joint gradient (BFGS/L-BFGS) fallback: \
                             {} consecutive ψ-zero iterations after exhausted backtracking \
                             (rho_dim={}, psi_dim={}, n_params={}, cost={:.6e})",
                        self.consecutive_psi_zero_iters,
                        self.layout.rho_dim(),
                        self.layout.psi_dim,
                        self.layout.n_params,
                        current_cost,
                    );
                    return Err(first_order_fallback_error(format!(
                        "HybridEFS ψ stagnation: {} consecutive iterations \
                             exhausted backtracking and zeroed ψ step \
                             (rho_dim={}, psi_dim={}, n_params={}, cost={:.6e})",
                        self.consecutive_psi_zero_iters,
                        self.layout.rho_dim(),
                        self.layout.psi_dim,
                        self.layout.n_params,
                        current_cost,
                    )));
                }
                return Ok(FixedPointSample {
                    value: current_cost,
                    step: scaled,
                    status,
                });
            }
            // ρ/τ-only backtracking also failed — surface the typed
            // joint-solver request so the runner abandons EFS for this attempt.
            log::info!(
                "[STAGE] HybridEFS -> joint gradient fallback: ρ/τ-only step also \
                 failed all {} halvings (rho_dim={}, psi_dim={}, n_params={}, \
                 cost={:.6e})",
                MAX_EFS_BACKTRACK,
                self.layout.rho_dim(),
                self.layout.psi_dim,
                self.layout.n_params,
                current_cost,
            );
            return Err(first_order_fallback_error(format!(
                "HybridEFS step rejected after {} halvings on full vector \
                 and {} halvings on ρ/τ-only fallback \
                 (rho_dim={}, psi_dim={}, n_params={}, cost={:.6e})",
                MAX_EFS_BACKTRACK,
                MAX_EFS_BACKTRACK,
                self.layout.rho_dim(),
                self.layout.psi_dim,
                self.layout.n_params,
                current_cost,
            )));
        }

        // Pure-EFS path with full backtracking exhausted: there is no ψ block
        // to escape to. Surface the same typed request so the runner switches
        // to a gradient-based solver instead of looping.
        log::info!(
            "[STAGE] EFS -> gradient fallback: no α ∈ {{1, …, 2^-{}}} decreased the \
             cost (rho_dim={}, n_params={}, cost={:.6e})",
            MAX_EFS_BACKTRACK,
            self.layout.rho_dim(),
            self.layout.n_params,
            current_cost,
        );
        Err(first_order_fallback_error(format!(
            "EFS step rejected after {} halvings on pure-ρ vector \
             (rho_dim={}, n_params={}, cost={:.6e})",
            MAX_EFS_BACKTRACK,
            self.layout.rho_dim(),
            self.layout.n_params,
            current_cost,
        )))
    }
}

impl OuterFixedPointBridge<'_> {
    /// Backtrack the cost along `raw_step` by halving α ∈ {1, 1/2, …, 2^-k}
    /// up to `max_halvings` times. Returns `Some(α·raw_step)` for the first
    /// α that yields a strictly lower finite cost, or `None` when every
    /// evaluable trial is rejected. A typed objective error means the evaluation
    /// artifact could not be constructed and is propagated without further probes.
    fn efs_backtrack(
        &mut self,
        x: &Array1<f64>,
        raw_step: &Array1<f64>,
        current_cost: f64,
        max_halvings: usize,
    ) -> Result<Option<Array1<f64>>, ObjectiveEvalError> {
        // Relaxed Armijo: accept any trial within ULP noise of the current
        // cost. Pure `<` rejects ULP-noise dithering on flat regions of V
        // and forces unnecessary halvings.
        let cost_floor = current_cost + EFS_COST_DESCENT_TOL * current_cost.abs().max(1.0);
        // `bt` counts trials so the accepted step can report its halving count
        // (trial `bt` runs at α = 2^-bt). Recoverable domain refusals arrive as
        // `Ok(+∞)` and keep halving; `Err` is reserved for a broken evaluation
        // artifact and leaves the search immediately.
        let mut bt = 0usize;
        let accepted = backtracking_line_search::<_, ObjectiveEvalError>(
            BacktrackConfig {
                max_steps: max_halvings + 1,
                ..BacktrackConfig::default()
            },
            |alpha| {
                bt += 1;
                let trial_step = raw_step * alpha;
                let trial = x + &trial_step;
                let cost = self
                    .obj
                    .eval_cost(&trial)
                    .map_err(|error| {
                        into_objective_error("EFS backtracking cost evaluation failed", error)
                    })?;
                if !(cost.is_finite() && cost <= cost_floor) {
                    log::trace!(
                        "[EFS] backtrack α=2^-{bt}={alpha:.4e}: trial cost {cost:.6e} not below current {current_cost:.6e}, halving",
                        bt = bt - 1,
                    );
                }
                Ok(Some((cost, trial_step)))
            },
            |_, c| c.is_finite() && c <= cost_floor,
        )?;
        Ok(accepted.map(|step| {
            let halvings = bt - 1;
            if halvings > 0 {
                log::debug!(
                    "[EFS] backtrack accepted at α=2^-{halvings}={alpha:.4e} \
                     after {halvings} halvings (cost: {current_cost:.6e} → {c:.6e})",
                    alpha = step.step,
                    c = step.value,
                );
            }
            step.payload
        }))
    }

    fn fixed_point_step_converged(
        &self,
        x: &Array1<f64>,
        step: &Array1<f64>,
        psi_indices: Option<&[usize]>,
    ) -> bool {
        if x.len() != step.len() {
            return false;
        }
        for idx in 0..step.len() {
            let scale = match psi_indices {
                Some(indices) if indices.contains(&idx) => x[idx].abs().max(1.0),
                _ => 1.0,
            };
            let normalized = step[idx].abs() / scale;
            if !normalized.is_finite() || normalized > self.fixed_point_tolerance {
                return false;
            }
        }
        true
    }
}

pub(crate) fn solution_into_outer_result(
    solution: Solution,
    converged: bool,
    plan_used: OuterPlan,
) -> OuterResult {
    let mut result = OuterResult::new(
        solution.final_point,
        solution.final_value,
        solution.iterations,
        converged,
        plan_used,
    );
    result.final_grad_norm = solution.final_gradient_norm;
    result.final_gradient = solution.final_gradient;
    result.final_hessian = solution.final_hessian;
    // #2547: carry the solver's own verdict instead of reconstructing one.
    // Every route that produces an `opt::Solution` funnels through here,
    // so populating the stop reason at this single site is what makes it
    // present on the BFGS / ARC / Newton / fixed-point routes at all — it
    // used to be hand-set on the matrix-free branch only, leaving `None`
    // everywhere else and making "why did it stop" unanswerable.
    result.operator_stop_reason = Some(stop_reason_from(solution.termination));
    result.solver_termination = Some(solution.termination);
    result
}

/// Project `opt`'s [`TerminationReason`] onto gam's coarser
/// [`OperatorTrustRegionStopReason`].
///
/// A lossy projection kept for the retry orchestrator, which dispatches
/// on the coarse category. Consumers that need the test that actually
/// fired — and the quantity it was judged against — read
/// `OuterResult::termination`; that is the point of carrying it.
pub(crate) fn stop_reason_from(reason: TerminationReason) -> OperatorTrustRegionStopReason {
    match reason {
        // A cost stall whose best-iterate projected gradient cleared the
        // outer tolerance is a KKT-stationary success, and the retry
        // orchestrator has always treated it as `Converged`. Preserved
        // exactly rather than unified with the floor case, because
        // changing it would change which fits certify.
        //
        // Worth flagging while touching this: `CostStallFlatValley`'s own
        // doc says the final analytic certificate needs that provenance in
        // BOTH cases — "the in-loop guard may already have certified the
        // score-relative residual OR may have returned a non-stationary
        // floor; either way the final analytic certificate needs this
        // provenance". Mapping the certified half to `Converged` discards
        // it. Real question, separate change; `OuterResult::termination`
        // now carries the undiscarded fact either way.
        TerminationReason::CostStallStationary { .. } => OperatorTrustRegionStopReason::Converged,
        TerminationReason::CostStallFloor { .. } => {
            OperatorTrustRegionStopReason::CostStallFlatValley
        }
        TerminationReason::TrustRegionRejectFloor { .. } => {
            OperatorTrustRegionStopReason::RejectFloor
        }
        TerminationReason::IterationBudget { .. } => OperatorTrustRegionStopReason::IterationBudget,
        // A stop the solver stands behind: it applied a test and the test
        // passed. Not a trust-region event, so it reports as converged.
        TerminationReason::GradientTolerance { .. }
        | TerminationReason::SmallStepFlatObjective { .. }
        | TerminationReason::RelativeStationarityWindow { .. }
        | TerminationReason::ModelNoiseFloor { .. }
        // The caller's own resolution rung: the solver stopped because the
        // model's interior Newton decrement fell below the tolerance THIS
        // crate handed it — the certificate's own test, applied online.
        | TerminationReason::ModelDecrementTolerance { .. }
        | TerminationReason::StepNormTolerance { .. }
        | TerminationReason::FixedPointRequestedStop { .. } => {
            OperatorTrustRegionStopReason::Converged
        }
        // A failure. These used to join the arm above, on the premise that
        // they are "a hard failure the caller sees through the `Err` arm" --
        // which is false for `LineSearchFailed` on the path it actually
        // takes: `run_plan` returns `Ok(non-converged)` whenever the last
        // iterate is finite, because that iterate is a usable checkpoint. So
        // the caller sees no `Err`, and the coarse reason it does see said
        // `Converged`. A binomial/logit REML fit that never accepted a single
        // step reported `stop_reason=Converged after 1 outer iteration(s)`
        // (#2614). The comment immediately below already argues that
        // defaulting an unknown stop to `Converged` would be wrong; three
        // KNOWN failures were doing exactly that.
        TerminationReason::LineSearchFailed { .. }
        | TerminationReason::ObjectiveFailed
        | TerminationReason::NumericalFailure => OperatorTrustRegionStopReason::SolverFailure,
        // `TerminationReason` is `#[non_exhaustive]`: a variant added
        // upstream lands here rather than breaking the build. Treating an
        // unknown stop as `Converged` would be the wrong default, so it
        // maps to the budget category, which the orchestrator already
        // handles as "did not certify".
        _ => OperatorTrustRegionStopReason::IterationBudget,
    }
}

pub(crate) fn outer_result_with_gradient_norm(
    rho: Array1<f64>,
    final_value: f64,
    iterations: usize,
    final_grad_norm: Option<f64>,
    converged: bool,
    plan_used: OuterPlan,
) -> OuterResult {
    let mut result = OuterResult::new(rho, final_value, iterations, converged, plan_used);
    result.final_grad_norm = final_grad_norm;
    result
}

pub(crate) fn outer_result_with_gradient(
    rho: Array1<f64>,
    final_value: f64,
    iterations: usize,
    final_grad_norm: Option<f64>,
    final_gradient: Option<Array1<f64>>,
    converged: bool,
    plan_used: OuterPlan,
) -> OuterResult {
    let mut result = outer_result_with_gradient_norm(
        rho,
        final_value,
        iterations,
        final_grad_norm,
        converged,
        plan_used,
    );
    result.final_gradient = final_gradient;
    result
}

use gam_problem::diagnostics::format_top_abs as format_top_abs_components;

pub(crate) fn bfgs_line_search_failure_message(
    context: &str,
    solution: &Solution,
    max_attempts: usize,
    failure_reason: impl std::fmt::Debug,
) -> String {
    let grad_norm = solution
        .final_gradient_norm
        .or_else(|| {
            solution
                .final_gradient
                .as_ref()
                .map(|gradient| gradient.iter().map(|v| v * v).sum::<f64>().sqrt())
        })
        .unwrap_or(f64::NAN);
    let gradient_detail = solution
        .final_gradient
        .as_ref()
        .map(|gradient| format_top_abs_components(gradient, "top_abs_gradient", 6))
        .unwrap_or_else(|| "top_abs_gradient=<unavailable>".to_string());
    format!(
        "{context}: BFGS line search failed; reason={failure_reason:?} \
         max_attempts={max_attempts} iterations={} final_value={:.6e} \
         |g|={:.3e} func_evals={} grad_evals={} {} {}",
        solution.iterations,
        solution.final_value,
        grad_norm,
        solution.func_evals,
        solution.grad_evals,
        format_top_abs_components(&solution.final_point, "top_abs_rho", 6),
        gradient_detail,
    )
}

#[cfg(test)]
mod termination_provenance_tests {
    use super::*;

    /// #2547: the stop reason must be present on EVERY route, not only
    /// the matrix-free branch that used to hand-set it. Every route funnels
    /// through `solution_into_outer_result`, so exercising that funnel is
    /// what proves the coverage.
    #[test]
    fn every_solver_derived_result_carries_its_stop_reason() {
        let solution = Solution {
            final_point: Array1::from_vec(vec![0.5, -1.25]),
            final_value: -3.0,
            final_gradient: Some(Array1::from_vec(vec![1.0e-9, -2.0e-9])),
            final_hessian: None,
            final_gradient_norm: Some(2.236e-9),
            final_step_norm: None,
            stationarity_kind: opt::StationarityKind::ProjectedGradient,
            iterations: 12,
            func_evals: 30,
            grad_evals: 30,
            hess_evals: 0,
            termination: TerminationReason::GradientTolerance {
                grad_norm: 2.236e-9,
                threshold: 1.0e-8,
            },
        };
        let plan = OuterPlan {
            solver: crate::rho_optimizer::capability::Solver::Bfgs,
            hessian_source: crate::rho_optimizer::capability::HessianSource::BfgsApprox,
        };
        let result = solution_into_outer_result(solution, true, plan);
        let carried = result
            .solver_termination
            .expect("a solver-derived result must carry the solver's verdict");
        let evidence = carried
            .stationarity_evidence()
            .expect("a gradient-tolerance stop was decided against a threshold");
        assert!(evidence.measured <= evidence.threshold);
        assert_eq!(
            result.operator_stop_reason,
            Some(OperatorTrustRegionStopReason::Converged)
        );
    }

    /// A stop that made no stationarity claim must not present one. An
    /// iteration budget knows a gradient norm but never compared it to
    /// anything, and reporting it as evidence is the defect #2465 names.
    #[test]
    fn a_budget_exhaustion_reports_no_stationarity_evidence() {
        let reason = TerminationReason::IterationBudget {
            iterations: 200,
            grad_norm: 1.58e2,
            threshold: 1.0e-3,
        };
        assert!(reason.stationarity_evidence().is_none());
        assert!(reason.grad_norm().is_some());
        assert!(!reason.is_stationary_claim());
        assert_eq!(
            stop_reason_from(reason),
            OperatorTrustRegionStopReason::IterationBudget
        );
    }

    /// The weak exit and the strong one both succeed, and the projection
    /// to gam's coarse enum cannot tell them apart — which is exactly why
    /// `termination` is carried alongside it rather than replaced by it.
    #[test]
    fn the_relative_window_exit_is_distinguishable_only_through_termination() {
        let strong = TerminationReason::GradientTolerance {
            grad_norm: 1.0e-9,
            threshold: 1.0e-8,
        };
        let weak = TerminationReason::RelativeStationarityWindow {
            grad_inf: 6.984e-2,
            threshold: 1.0e-3 * (1.0 + 30.0),
            window: 3,
        };
        assert_eq!(stop_reason_from(strong), stop_reason_from(weak));
        let s = strong.stationarity_evidence().unwrap();
        let w = weak.stationarity_evidence().unwrap();
        assert_eq!(s.scaling, opt::StationarityScaling::Absolute);
        assert_eq!(w.scaling, opt::StationarityScaling::RelativeToIterate);
        assert!(w.threshold > s.threshold);
    }

    /// A stop that FAILED must not project onto the same coarse reason as a
    /// stop that passed a test.
    ///
    /// `LineSearchFailed` reached this projection reporting `Converged`, and
    /// the path that produces it is not hypothetical: `run_plan` returns
    /// `Ok(non-converged)` whenever the failed search's last iterate is
    /// finite, so the caller sees no `Err` and reads only this label. A
    /// binomial/logit REML fit that accepted no step at all reported
    /// `stop_reason=Converged after 1 outer iteration(s)` (#2614). The three
    /// failure variants are asserted together because they entered the
    /// defect together, in one `|` chain.
    #[test]
    fn a_failed_stop_does_not_project_onto_converged() {
        let passed = TerminationReason::GradientTolerance {
            grad_norm: 1.0e-9,
            threshold: 1.0e-8,
        };
        assert_eq!(
            stop_reason_from(passed),
            OperatorTrustRegionStopReason::Converged,
            "a satisfied gradient test is still a convergence"
        );
        for failed in [
            TerminationReason::LineSearchFailed {
                grad_norm: 1.484_825,
            },
            TerminationReason::ObjectiveFailed,
            TerminationReason::NumericalFailure,
        ] {
            let projected = stop_reason_from(failed);
            assert_eq!(
                projected,
                OperatorTrustRegionStopReason::SolverFailure,
                "{failed:?} is a failure, not a convergence; it projected to {projected:?}"
            );
            assert!(
                failed.stationarity_evidence().is_none(),
                "{failed:?} compared nothing to anything, so it has no stationarity evidence to \
                 report -- if it grows one, this projection needs revisiting"
            );
        }
    }
}
