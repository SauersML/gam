use super::decrement_bands::{decrement_stationarity_bound, outer_decrement_verdict};
use super::newton_polish::resolvable_decrease_evidence;
use super::*;
use gam_problem::{DominanceRefusalKind, StationarityRung, StationarityStandard};

use super::asymptote_certificate::{
    AsymptoteSample, AsymptoteSide, AsymptoteTolerances, AsymptoteVerdict, AsymptoteWindow,
    MIN_TAIL_SAMPLES, assess_coordinate,
};
use super::rail_face::{
    RailFaceLimitOutcome, RailFaceProof, RailFaceVerdict, certify_rail_face,
};

pub(crate) const OPERATOR_TRUST_RESTART_RADIUS_FLOOR: f64 = 1.0e-6;

/// Inner coefficient state bound to one exact outer seed.
///
/// A cached coefficient vector is only a valid initialization for the outer
/// coordinate that produced it.  Keeping the coordinate beside the vector
/// prevents a multi-start run from silently reusing one basin's coefficients
/// at a different seed.
#[derive(Clone, Debug)]
pub(crate) struct BoundInnerSeed {
    pub(crate) theta: Array1<f64>,
    pub(crate) beta: Array1<f64>,
}

/// Exact outer curvature bound to the one outer coordinate it was measured at.
///
/// It seeds the BFGS iter-0 metric only at a start bitwise equal to `theta`,
/// so a reseed or a resumed checkpoint elsewhere never inherits it.
#[derive(Clone, Debug)]
pub(crate) struct BoundOuterCurvature {
    pub(crate) theta: Array1<f64>,
    pub(crate) hessian: Array2<f64>,
}

pub(crate) fn outer_theta_bitwise_eq(left: &Array1<f64>, right: &Array1<f64>) -> bool {
    left.len() == right.len()
        && left
            .iter()
            .zip(right.iter())
            .all(|(left, right)| left.to_bits() == right.to_bits())
}

/// Why a search on the parent's own inputs did not take the parent's point.
pub(crate) const RESUME_DECLINED: &str = "the parent's certified point is not certified for this search's criterion (a pilot, an \
     evidence fit or an alternation round, or a criterion this version computes differently), \
     so this search ran as it runs cold";

/// `config` as the resume attempt for a warm start on the parent's own inputs
/// (gam#3002): the parent's point as the one seed, its inner mode installed
/// there, and its recorded criterion value, so the search accepts the point
/// where it stands or declines (`resume_prior_certificate`).
pub(crate) fn resume_config(
    config: &OuterConfig,
    warm_start: &gam_model_api::WarmStart,
) -> OuterConfig {
    let mut resume = config.clone();
    resume.initial_rho = Some(warm_start.theta.clone());
    resume.initial_inner_seed = Some(BoundInnerSeed {
        theta: warm_start.theta.clone(),
        beta: warm_start.beta.clone(),
    });
    resume.initial_rho_is_prior_terminal_certificate = true;
    resume.resume_value = Some(warm_start.value);
    resume
}

/// Install cached inner state only at the exact outer coordinate that owns it.
///
/// This function is called after a seed-attempt reset and immediately before
/// the literal seed can be evaluated. Generated multistart candidates never
/// inherit coefficients from another outer point.
pub(crate) fn install_matching_initial_inner_seed(
    obj: &mut dyn OuterObjective,
    config: &OuterConfig,
    seed: &Array1<f64>,
    context: &str,
) -> Result<(), EstimationError> {
    let Some(bound) = config.initial_inner_seed.as_ref() else {
        return Ok(());
    };
    if !outer_theta_bitwise_eq(&bound.theta, seed) {
        return Ok(());
    }
    match obj.seed_inner_state(&bound.beta)? {
        SeedOutcome::Installed => log::debug!(
            "[CACHE] beta-warm context={} theta_dim={} beta_dim={} action=installed",
            context,
            bound.theta.len(),
            bound.beta.len(),
        ),
        SeedOutcome::NoSlot => log::debug!(
            "[CACHE] beta-warm context={} theta_dim={} beta_dim={} action=skip \
             reason=objective_has_no_inner_beta_slot",
            context,
            bound.theta.len(),
            bound.beta.len(),
        ),
        SeedOutcome::Incompatible => log::debug!(
            "[CACHE] beta-warm context={} theta_dim={} beta_dim={} action=rho-only \
             reason=seed_beta_incompatible_with_inner_state",
            context,
            bound.theta.len(),
            bound.beta.len(),
        ),
    }
    Ok(())
}

/// Temporarily require a complete inner solve.
///
/// Search-time REML evaluations may deliberately cap P-IRLS, but a sample
/// used as mathematical evidence about the true profiled objective cannot.
/// Seed samples, terminal certificates, and final state installation therefore
/// lift the cap for the duration of their evaluation and restore the scheduler's
/// value afterward. Callers that may hold capped cached state reset the
/// objective before making the full-fidelity request.
pub(crate) struct FullFidelityInnerCapGuard<'a> {
    cap: &'a AtomicUsize,
    previous: usize,
}

impl<'a> FullFidelityInnerCapGuard<'a> {
    pub(crate) fn lift(feedback: &'a InnerProgressFeedback) -> Self {
        let cap = feedback.cap.as_ref();
        let previous = cap.swap(0, Ordering::Relaxed);
        Self { cap, previous }
    }
}

impl Drop for FullFidelityInnerCapGuard<'_> {
    fn drop(&mut self) {
        self.cap.store(self.previous, Ordering::Relaxed);
    }
}

/// Declared size of the problem behind an outer objective: the number of
/// observations and of inner coefficients. The decrement-band certificate
/// charges its floating-point formation error against these counts
/// (`outer_decrement_bands`); a route that declares no size takes no band
/// verdict.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub(crate) struct OuterProblemSize {
    pub(crate) n_obs: Option<usize>,
    pub(crate) p_coefficients: Option<usize>,
}

impl OuterProblemSize {
    /// The criterion's statistical resolution `τ_stat = η²/2 = 1/(2n)`, in the
    /// criterion's own absolute units, over the declared `n` observations.
    ///
    /// A point whose remaining decrease `G` to the exact optimum satisfies
    /// `½(θ − θ̂)ᵀH(θ − θ̂) ≤ G` has every linear functional `Lᵀθ` within
    /// `√(2G)·se(Lᵀθ̂)` of its value at the optimum (Cauchy–Schwarz in the
    /// `H` inner product), and by the delta method so does every smooth
    /// functional of the hyperparameters: edf, fitted values, AIC. The
    /// criterion is itself a first-order object — LAML is a Laplace
    /// approximation with `O(1/n)` absolute error at fixed dimension (Tierney &
    /// Kadane 1986), and Wald/edf inference built on `λ̂` carries `O(n^{-1/2})`
    /// error — so resolving the optimum to `η = n^{-1/2}` sampling SDs puts the
    /// optimization error below the intrinsic error of the inference built on
    /// it. A decrease below `τ_stat` changes no reported quantity by more than
    /// that. It is invariant to the units of `y` and to any additive constant
    /// in `V` (a Poisson `Σ log y!`), which `rel·(1 + |V|)` was not.
    ///
    /// `η² = 1/n` is the first-order choice. For Gaussian REML, whose criterion
    /// carries no Laplace error, `1/edf` or `1/(n − p)` could be argued instead;
    /// the choice moves `τ_stat` by a constant factor only (#3192).
    ///
    /// `None` when the route declares no observation count.
    pub(crate) fn statistical_resolution(&self) -> Option<f64> {
        self.n_obs
            .filter(|&n| n > 0)
            .map(|n| 0.5 / n as f64)
    }
}

/// Configuration for the outer optimization runner.
#[derive(Clone, Debug)]
pub(crate) struct OuterConfig {
    pub(crate) tolerance: f64,
    pub(crate) max_iter: usize,
    /// The model's canonical feasible outer domain. Every stationarity
    /// certificate and rail report reasons against this box.
    pub(crate) model_domain_bounds: Option<(Array1<f64>, Array1<f64>)>,
    /// Which faces of `model_domain_bounds` are the route's derived limit model,
    /// per coordinate, lower then upper (#2954, #2627). A face is one when the
    /// route derived it from the term's own limit (a #2812 resolvability edge,
    /// past which the term is at its null-space or unpenalized fit to the
    /// gradient's resolution); every other face is a representability literal,
    /// and box-KKT certifies nothing about the data there. `None`: the route
    /// declared none.
    pub(crate) model_domain_limit_faces: Option<(Vec<bool>, Vec<bool>)>,
    /// A temporary algorithmic subspace used only by an active-set polish.
    /// It may narrow the model domain but never changes the feasible cone that
    /// screening or mint is allowed to certify.
    pub(crate) search_bounds_override: Option<(Array1<f64>, Array1<f64>)>,
    /// Model-derived log-smoothing-parameter start (one entry per outer
    /// coordinate). Used as the single outer start when `initial_rho` is
    /// absent; see [`crate::rho_optimizer::run_plan::outer_start_point`].
    pub(crate) heuristic_log_lambdas: Option<Vec<f64>>,
    pub(crate) initial_rho: Option<Array1<f64>>,
    /// The lowest state an earlier search of this criterion evaluated: its resume checkpoint
    /// (#2953).
    ///
    /// The plan loop hands its best checkpoint to the next attempt with its iteration count
    /// zeroed (the loop has already counted that work), and starts its own best checkpoint from
    /// the one its caller carries. The plan runner starts from it as its own lowest evaluated
    /// state, so a candidate the search certifies is judged against it before it publishes
    /// (#2596, #2627), exactly as a state of its own search would be. With a single start per
    /// search, a carried state is the only other state a certified candidate can be judged
    /// against. It is in native order on entry to [`run_outer`], which permutes it with the rest
    /// of the configuration into the canonical frame the plan loop runs in.
    pub(crate) carried_checkpoint: Option<OuterResult>,
    pub(crate) initial_inner_seed: Option<BoundInnerSeed>,
    /// The exact analytic Hessian of the criterion a continuation search
    /// resumes, at the start it resumes from. The #784 corrected continuation
    /// starts at the certified Laplace optimum, where the Laplace criterion's
    /// analytic Hessian is exact and the correction's own curvature is a small
    /// perturbation of it; BFGS takes its inverse as the iter-0 metric, so the
    /// first step is a Newton step instead of a unit-length gradient step that
    /// the line search halves back down one full corrected evaluation at a time.
    /// The metric shapes the path only: BFGS reaches the same stationary point
    /// under any SPD initial metric.
    pub(crate) initial_curvature: Option<BoundOuterCurvature>,
    pub(crate) fallback_policy: FallbackPolicy,
    /// `initial_rho` came from a PRIOR FIT'S TERMINAL CERTIFICATE, not from a
    /// heuristic, a mid-run checkpoint, or a caller's guess.
    ///
    /// The distinction is not where the number was stored, it is what is known
    /// about it: a terminal certificate is a rho a previous outer run already
    /// certified as stationary, so re-deriving it can only move it.
    pub(crate) initial_rho_is_prior_terminal_certificate: bool,
    /// A resume attempt (gam#3002): the criterion value a prior fit certified at
    /// `initial_rho`. The search then accepts `initial_rho` where it stands when
    /// it is certified for THIS search's criterion (its value agrees within the
    /// rounding envelope and its projected gradient is inside the band), and
    /// otherwise declines with an error. It never searches
    /// (`resume_prior_certificate`).
    pub(crate) resume_value: Option<f64>,
    /// Outer-aware inner-PIRLS iteration cap.
    /// When set, the BFGS bridge drives this atomic on every accepted
    /// gradient eval to coarsen the inner Newton solve at early outer iters
    /// (when ρ is far from converged) and lift it back to full as
    /// convergence approaches. It does NOT suppress cache writes /
    /// warm-start updates / KKT enforcement; it is purely a budget. See
    /// `RemlObjectiveState::outer_inner_cap` for dual-cap semantics.
    pub(crate) outer_inner_cap: Option<InnerProgressFeedback>,
    pub(crate) operator_initial_trust_radius: Option<f64>,
    pub(crate) arc_initial_regularization: Option<f64>,
    /// BFGS line-search infinity-norm cap applied to the leading `rho_dim`
    /// outer parameters (log-λ axes). Documented natural step for
    /// `log(lambda)` is ≈ 5 (`e^5 ≈ 148`-fold smoothing-parameter change
    /// per accepted outer iter — matches typical quasi-Newton direction
    /// magnitude on flat REML surfaces). Setting this `None` disables the
    /// rho-axis cap entirely.
    pub(crate) bfgs_step_cap: Option<f64>,
    /// BFGS line-search infinity-norm cap applied to the trailing `psi_dim`
    /// outer parameters (kappa / aniso-log-scale axes). Required because
    /// the kernel scale axes need much tighter control (`e^1 ≈ 2.7`-fold
    /// per iter is plenty) — using the rho-axis cap here lets the optimizer
    /// jump kappa by orders of magnitude per step and oscillate. Setting
    /// this `None` disables the psi-axis cap.
    pub(crate) bfgs_step_cap_psi: Option<f64>,
    /// Optional persistent-cache session. When `Some`, every finite objective
    /// evaluation is written through to disk (rate-limited, atomic-rename)
    /// and the best on-disk rho is prepended as a seed at the start of each
    /// plan attempt. Defaulted off so test-only paths skip filesystem I/O.
    pub(crate) cache_session: Option<Arc<CacheSession>>,
    /// Optional mirror cache sessions. Checkpoints and successful finalize
    /// writes are also written to each of these sessions (different keys,
    /// shared store). Used for hierarchical broadcast: the current best ρ is
    /// written to the exact-key (primary) AND the data-independent
    /// seed-prefix key so the next fit with related structure can warm-start
    /// from this one, even after an interrupted run.
    pub(crate) cache_mirror_sessions: Vec<Arc<CacheSession>>,
    pub(crate) problem_size: OuterProblemSize,
    /// Converged exact outer Hessian `H(θ̂)` transferred from a prior
    /// structurally-matching fit via the persistent cache (a warm-start *hit*),
    /// in the full θ layout. When present and SPD, the BFGS host path seeds its
    /// iter-0 metric with `InitialMetric::DenseInverseHessian(H⁻¹)` so the first
    /// outer step is quasi-Newton instead of unscaled steepest descent — the
    /// dominant LOSO line-search-bracketing cost (each bracketing probe is a
    /// full inner joint-Newton re-solve). Strictly stronger than the scalar
    /// `1/‖g₀‖` metric: it carries the full anisotropic curvature, which across
    /// folds (one held-out point) is nearly identical to this fold's. Never
    /// changes the converged optimum — BFGS reaches `∇V=0` under any SPD initial
    /// metric. `None` on every cold-start / no-cache / pre-Hessian-schema path,
    /// which falls back to the scalar warm metric byte-for-byte.
    pub(crate) warm_start_outer_hessian: Option<Array2<f64>>,
    /// Per-ρ-coordinate structural keys, in the objective's NATIVE (formula)
    /// coordinate order, used to make the outer smoothing-parameter search
    /// invariant to the order the user wrote the smooth terms / tensor margins
    /// (#1538/#1539).
    ///
    /// When `Some` and the keys induce a non-identity canonical permutation,
    /// [`run_outer`] reorders the coordinate layout the optimizer sees into a
    /// stable canonical order (derived purely from the keys, never from the
    /// native position) before seeding/optimizing, and inverts the permutation
    /// on the returned ρ / gradient / Hessian so the caller still receives the
    /// native layout. Seeding, multistart and tie-breaking then all operate on
    /// the identical canonical layout for every term order, so both orders
    /// reach the same λ̂ and the same fitted surface. `None` (or an identity
    /// permutation) leaves the legacy native-order path byte-for-byte unchanged.
    pub(crate) rho_canonical_keys: Option<Vec<u64>>,
    /// The canonical run's coordinate map, `order[c]` = the native coordinate at
    /// canonical slot `c` (#2817). Set only by `canonicalize_outer_config`, on the
    /// config of the recursive canonical run, and read only through
    /// [`native_coordinate`](super::objective::native_coordinate): every
    /// coordinate a refusal or log line names inside that run is rendered in the
    /// caller's native order when the text is written, rather than patched after
    /// the run returns.
    pub(crate) native_coordinate_order: Option<Vec<usize>>,
    /// A CALLER'S absolute requirement on the projected outer gradient norm,
    /// honoured by the search and not merely checked at the end (#2568).
    ///
    /// The engine's own stationarity bound is a function of the FIT, not a
    /// constant of the engine: it is a widening ladder anchored on the
    /// criterion's magnitude and the optimizer's terminal state. That is right
    /// for minting a model, and it means two fits on the same data in the same
    /// call can be certified against bounds four orders apart -- measured on
    /// #2568 at `2.0708e-4` for one and **exactly `1.000e0`** for its
    /// companion, the latter being the saturated score-relative flat-valley
    /// value -- a rung since deleted (#2458), because it was a constant
    /// selected by an exit reason and it won precisely where the probe-noise
    /// MEASUREMENT had declined to license any bound. A consumer
    /// whose accuracy requirement is stricter than whatever the ladder happens
    /// to compute had no supported way to impose it.
    ///
    /// Reading `|Pg|` off the summary and thresholding it afterwards is not the
    /// same thing, and is the shape SPEC-20 exists to prevent: it mints a
    /// replayable, diagnostics-clean fit the consumer must then reject. Nothing
    /// made the optimizer *work harder* to reach the stricter standard.
    ///
    /// So this number enters in BOTH places that decide the outcome:
    ///
    /// * [`outer_gradient_tolerance`] floors the solver's convergence band at
    ///   it, so the outer loop keeps going instead of stopping at a looser
    ///   sealed bound -- the load-bearing half;
    /// * the certificate's bound ladder is CAPPED by it, so a point the engine
    ///   would have certified against a wider rung is refused, reporting
    ///   [`StationarityBoundSource::CallerRequirement`] and naming the engine's
    ///   own bound alongside it.
    ///
    /// The cap is applied to the ladder's TOP, after every widening rung, which
    /// is why it cannot be defeated by a rung that fires later. It never
    /// loosens: a requirement weaker than the engine's own bound leaves the
    /// engine's in force, because a caller asking for less accuracy than the
    /// engine already guarantees is not asking for anything.
    ///
    /// `None` reproduces today's behaviour byte-for-byte on every path.
    pub(crate) required_projected_gradient_norm: Option<f64>,
    /// Require the terminal mint to carry a measured, raw PSD Hessian.
    ///
    /// The general certificate may admit a tiny assembled negative direction
    /// when the gradient-residue floor proves it unresolved. Profiled
    /// nonconvex coefficient families cannot use that weaker answer: their
    /// selected branch is a local minimum only when the analytic outer Hessian
    /// itself is measured PSD. Declaring that requirement here lets the outer
    /// recovery loop escape/re-optimize instead of contradicting the certificate
    /// later during fit assembly.
    pub(crate) require_measured_psd: bool,
    /// A strict saddle was CERTIFIED earlier in this solve, so every later run
    /// searches on the declared analytic Hessian (#2939).
    ///
    /// A gradient-only preference (`OuterCapability::prefer_gradient_only`)
    /// keeps the order-four Hessian out of the search and prices it once at the
    /// mint, which is the right trade while the search is descending. Once the
    /// mint has measured an interior strict saddle and published its escape it
    /// is not. Along the escape ray the gradient grows only as `α·|λ_min|`, so
    /// the escape point still sits inside the solver's gradient band, and a
    /// positive-definite secant restart stops at iteration 0. The next mint
    /// finds the saddle again, and every round pays a whole certification. On
    /// #2939's survival marginal-slope fit the retry stopped at `|g| = 1.881`
    /// against a band of `1.958` after an escape at `λ_min = −4.185e−2`, and two
    /// rounds cost 5 m 10 s and 4 m 39 s. ARC stops only on a PSD free Hessian,
    /// so a curvature search from the escape point keeps following the negative
    /// curvature.
    ///
    /// Set only by the certify-last reseed loop and never cleared inside the
    /// solve, so the search route does not flip back to BFGS once a saddle is on
    /// record.
    pub(crate) curvature_search_latched: bool,
}

/// The outer search's iteration count when a caller declares none: no count.
///
/// Every outer route now ends a stalled search on a progress certificate rather
/// than a count (#2817):
/// - The dense ARC and matrix-free trust-region bridges, and the host BFGS
///   stuck-stall escapes, license another filled cost-stall window only after
///   resolved descent or a smaller incumbent residual
///   (`CostStallGuard::license_continuation`).
/// - The fixed-point and per-atom walks carry `FixedPointProgress`.
/// - The device BFGS walk carries opt's native cost stall.
///
/// A stationary point stops on the certificate's own rungs. The 200-iteration
/// default this replaces decided when a search that had not converged was
/// refused, which is the grind #2817 is about. A caller with an explicit
/// option-driven budget still passes it through `OuterProblem::with_max_iter`.
pub(crate) const UNBOUNDED_OUTER_ITERATIONS: usize = usize::MAX;

impl Default for OuterConfig {
    fn default() -> Self {
        Self {
            tolerance: 1e-5,
            required_projected_gradient_norm: None,
            require_measured_psd: false,
            max_iter: UNBOUNDED_OUTER_ITERATIONS,
            model_domain_bounds: None,
            model_domain_limit_faces: None,
            search_bounds_override: None,
            heuristic_log_lambdas: None,
            initial_rho: None,
            carried_checkpoint: None,
            initial_inner_seed: None,
            fallback_policy: FallbackPolicy::Automatic,
            initial_rho_is_prior_terminal_certificate: false,
            resume_value: None,
            outer_inner_cap: None,
            operator_initial_trust_radius: None,
            arc_initial_regularization: None,
            bfgs_step_cap: None,
            bfgs_step_cap_psi: None,
            cache_session: None,
            cache_mirror_sessions: Vec::new(),
            problem_size: OuterProblemSize::default(),
            warm_start_outer_hessian: None,
            initial_curvature: None,
            rho_canonical_keys: None,
            native_coordinate_order: None,
            curvature_search_latched: false,
        }
    }
}

// ─── OuterProblem builder ─────────────────────────────────────────────
//
// Declarative builder for outer optimization problems.  Derives
// OuterCapability flags from high-level inputs (gradient/hessian
// availability, psi dimension, EFS eligibility) so call sites never
// hand-copy capability flags.

/// Declarative outer-problem builder.  Produces both the
/// [`OuterCapability`] (what the objective can provide) and the
/// `OuterConfig` (how the runner should behave) from a small set
/// of high-level declarations.
#[derive(Clone)]
pub struct OuterProblem {
    n_params: usize,
    gradient: Derivative,
    hessian: DeclaredHessianForm,
    prefer_gradient_only: bool,
    disable_fixed_point: bool,
    psi_dim: usize,
    barrier_config: Option<BarrierConfig>,
    tolerance: f64,
    /// See [`OuterConfig::required_projected_gradient_norm`] (#2568).
    required_projected_gradient_norm: Option<f64>,
    require_measured_psd: bool,
    max_iter: usize,
    bounds: Option<(Array1<f64>, Array1<f64>)>,
    limit_faces: Option<(Vec<bool>, Vec<bool>)>,
    heuristic_log_lambdas: Option<Vec<f64>>,
    initial_rho: Option<Array1<f64>>,
    initial_curvature: Option<BoundOuterCurvature>,
    fallback_policy: FallbackPolicy,
    outer_inner_cap: Option<InnerProgressFeedback>,
    operator_initial_trust_radius: Option<f64>,
    arc_initial_regularization: Option<f64>,
    bfgs_step_cap: Option<f64>,
    bfgs_step_cap_psi: Option<f64>,
    cache_session: Option<Arc<CacheSession>>,
    cache_mirror_sessions: Vec<Arc<CacheSession>>,
    problem_size: OuterProblemSize,
    rho_canonical_keys: Option<Vec<u64>>,
    /// The inner mode at one joined multistart seed: installed only at that
    /// exact point (`install_matching_initial_inner_seed`).
    warm_start: Option<BoundInnerSeed>,
    /// A prior fit's certified point (`with_warm_start`, gam#3002).
    warm_start_source: Option<gam_model_api::WarmStart>,
    /// This problem is a multistart's resume attempt: a declined prior
    /// certificate is returned as the error, and the multistart runs the cold
    /// seeds itself.
    resume_only: bool,
}

impl OuterProblem {
    pub fn new(n_params: usize) -> Self {
        Self {
            n_params,
            gradient: Derivative::Unavailable,
            hessian: DeclaredHessianForm::Unavailable,
            // Closed-form/test objectives preserve explicit ARC availability;
            // production family-ladder entry points opt into #2359's
            // optimize-3/certify-4 protocol with
            // `with_prefer_gradient_only(true)`.
            prefer_gradient_only: false,
            disable_fixed_point: false,
            psi_dim: 0,
            barrier_config: None,
            tolerance: 1e-5,
            required_projected_gradient_norm: None,
            require_measured_psd: false,
            max_iter: UNBOUNDED_OUTER_ITERATIONS,
            bounds: None,
            limit_faces: None,
            heuristic_log_lambdas: None,
            initial_rho: None,
            initial_curvature: None,
            fallback_policy: FallbackPolicy::Automatic,
            outer_inner_cap: None,
            operator_initial_trust_radius: None,
            arc_initial_regularization: None,
            bfgs_step_cap: None,
            bfgs_step_cap_psi: None,
            cache_session: None,
            cache_mirror_sessions: Vec::new(),
            problem_size: OuterProblemSize::default(),
            rho_canonical_keys: None,
            warm_start: None,
            warm_start_source: None,
            resume_only: false,
        }
    }

    /// Supply per-ρ-coordinate structural keys (native/formula order) so the
    /// outer search is canonicalized to be invariant to the order the smooth
    /// terms / tensor margins were written (#1538/#1539). See
    /// `OuterConfig::rho_canonical_keys`.
    pub(crate) fn with_rho_canonical_keys(mut self, keys: Option<Vec<u64>>) -> Self {
        self.rho_canonical_keys = keys;
        self
    }

    pub fn with_gradient(mut self, d: Derivative) -> Self {
        self.gradient = d;
        self
    }
    pub fn with_hessian(mut self, form: DeclaredHessianForm) -> Self {
        self.hessian = form;
        self
    }
    /// Choose whether analytic Hessian work is reserved for terminal
    /// certification (`true`, the generic derivative-ladder protocol) or may
    /// also be consumed by the search (`false`, for closed-form objectives).
    pub fn with_prefer_gradient_only(mut self, prefer_gradient_only: bool) -> Self {
        self.prefer_gradient_only = prefer_gradient_only;
        self
    }
    /// Forbid the planner from selecting EFS/HybridEfs, even when the
    /// objective implements `eval_efs()` and the coordinate structure would
    /// otherwise make pure/hybrid EFS eligible.
    ///
    /// Callers use this for families where the Wood-Fasiolo structural
    /// property is known not to hold (e.g. GAMLSS/location-scale with
    /// β-dependent joint Hessian), so EFS would stagnate and burn budget
    /// before the automatic cascade falls back to gradient-based BFGS.
    pub fn with_disable_fixed_point(mut self, disable: bool) -> Self {
        self.disable_fixed_point = disable;
        self
    }
    // MEASURE-JET ψ REGISTRATION: the engine below is already complete for a
    // 3-coordinate measure-jet ψ group (s, α, ln τ) — `psi_dim` is generic,
    // `with_bounds` carries the s ∈ (0, 2) box (the same convention matern κ
    // uses for its log-κ window; no logistic reparameterization exists or is
    // needed in-house), `with_bfgs_step_cap_psi` caps per-iteration ψ moves,
    // and `DirectionalHyperParam::new_compact` (solver/reml/mod.rs) carries
    // penalty-only first/second/cross jets with `is_penalty_like`
    // auto-derived from the identically-zero design drift (∂X/∂ψ ≡ 0).
    // Every remaining registration arm is formula-layer dispatch in
    // src/terms/smooth.rs (eligibility in
    // `spatial_term_supports_hyper_optimization`, dims in
    // `spatial_dims_per_term`, seed/bounds/write-back on
    // `SpatialLogKappaCoords`, the per-trial rebuild in
    // `apply_log_kappa_to_term`, and the derivative bundle in
    // `try_build_spatial_term_log_kappa_derivative`, which currently returns
    // `Ok(None)` for `SmoothBasisSpec::MeasureJet`) plus the
    // `build_measure_jet_basis_psi_derivatives` producer in
    // src/terms/basis/measure_jet_smooth.rs; both are owned by the
    // measure-jet terms actor. Registration stays gated on those arms — do
    // NOT add measure-jet-specific branches to this engine.
    pub fn with_psi_dim(mut self, dim: usize) -> Self {
        self.psi_dim = dim;
        self
    }
    pub(crate) fn with_barrier(mut self, cfg: Option<BarrierConfig>) -> Self {
        self.barrier_config = cfg;
        self
    }
    pub fn with_tolerance(mut self, tol: f64) -> Self {
        self.tolerance = tol;
        self
    }
    /// The absolute stationarity tolerance this problem declares.
    pub fn tolerance(&self) -> f64 {
        self.tolerance
    }
    pub fn with_max_iter(mut self, n: usize) -> Self {
        self.max_iter = n;
        self
    }
    pub fn with_bounds(mut self, lo: Array1<f64>, hi: Array1<f64>) -> Self {
        self.bounds = Some((lo, hi));
        self
    }
    /// Declare which faces of the bounds are derived limit models (#2954), lower
    /// then upper, one flag per coordinate.
    pub fn with_limit_faces(mut self, lower: Vec<bool>, upper: Vec<bool>) -> Self {
        self.limit_faces = Some((lower, upper));
        self
    }
    pub fn with_heuristic_log_lambdas(mut self, h: Vec<f64>) -> Self {
        self.heuristic_log_lambdas = Some(h);
        self
    }
    pub fn with_initial_rho(mut self, rho: Array1<f64>) -> Self {
        self.initial_rho = Some(rho);
        self
    }
    /// Bind the exact outer Hessian measured at `theta` to this search; see
    /// [`OuterConfig::initial_curvature`].
    pub(crate) fn with_initial_curvature(mut self, theta: Array1<f64>, hessian: Array2<f64>) -> Self {
        self.initial_curvature = Some(BoundOuterCurvature { theta, hessian });
        self
    }
    /// Offer this search a prior fit's certified outer point (gam#3002). Every
    /// warm-start source reaches every outer search through this one rule, and
    /// [`Self::run`] and [`Self::run_certified_multistart`] derive from it what
    /// the point may do without changing the point the search reports for its
    /// own criterion (see [`gam_model_api::WarmStart`]):
    /// - on the parent's own inputs the point is offered as a prior certificate
    ///   first. The search accepts it with no outer iteration where it is
    ///   certified for this search's criterion (`resume_prior_certificate`), and
    ///   otherwise runs exactly as it runs cold, from a reset objective;
    /// - on other inputs the independent multistart adds it as one more seed, so
    ///   its argmin is taken over a superset of the cold seeds; a single search,
    ///   which certifies from its one derived start, runs cold and records why.
    /// A point of another outer dimension belongs to another search: this one
    /// runs cold.
    pub fn with_warm_start(mut self, warm_start: &gam_model_api::WarmStart) -> Self {
        self.warm_start_source = Some(warm_start.clone());
        self
    }
    /// This problem with no warm start: the search a cold fit runs.
    fn without_warm_start(&self) -> Self {
        let mut cold = self.clone();
        cold.warm_start = None;
        cold.warm_start_source = None;
        cold.resume_only = false;
        cold
    }
    /// The warm start this search can take, if its point has this search's outer
    /// dimension. A point of another dimension is recorded as not used here.
    fn warm_start_for_this_search(&self) -> Option<&gam_model_api::WarmStart> {
        let warm_start = self.warm_start_source.as_ref()?;
        if warm_start.theta.len() != self.n_params {
            warm_start.record(gam_model_api::WarmStartOutcome::NotUsed(
                "the parent's certified point has another outer dimension than this search",
            ));
            return None;
        }
        Some(warm_start)
    }
    /// Wire the bidirectional inner-PIRLS feedback channel.
    ///
    /// The outer bridge writes a coarsened iteration cap into
    /// `feedback.cap` on every accepted gradient/Hessian eval; the inner
    /// solver writes back into `feedback.last_iters` /
    /// `feedback.last_converged` after each inner solve so the
    /// next outer iter's schedule can adapt to the inner solver's
    /// actual convergence behavior. Typical caller passes
    /// `InnerProgressFeedback {
    ///     cap: Arc::clone(&reml_state.outer_inner_cap),
    ///     last_iters: Arc::clone(&reml_state.last_inner_iters),
    ///     last_converged: Arc::clone(&reml_state.last_inner_converged),
    /// }` so the inner and outer observe the same atomics.
    pub(crate) fn with_outer_inner_cap(mut self, feedback: InnerProgressFeedback) -> Self {
        self.outer_inner_cap = Some(feedback);
        self
    }

    /// Wire a one-shot "re-evaluate the inner solve COLD" signal that the outer
    /// cost-stall guard raises when it grants a STUCK-stall escape (#2349).
    ///
    /// A profiled objective whose inner solve is warm-started along the outer
    /// trajectory can carry value HYSTERESIS on a near-flat inner ridge — the
    /// multinomial simplex-boundary regime where the softmax Fisher weight
    /// `diag(p) − ppᵀ` collapses is the motivating case: two warm starts
    /// converge to different ridge points whose Laplace `½log|H(β)|`, hence the
    /// profiled objective, differ by more than the outer descent resolution, so
    /// the optimizer's step-acceptance cannot separate real descent from that
    /// hysteresis and grinds to `max_iter` at a non-stationary point. Uncapping
    /// the inner cycle budget does not cure it (a fully converged warm solve
    /// still lands on the warm-biased ridge point); the objective must re-solve
    /// COLD to see a consistent surface.
    ///
    /// The caller shares this `Arc<AtomicBool>` with its objective closure and
    /// consults it there, re-solving the inner problem from a canonical seed
    /// (dropping the warm cache) whenever the flag is raised. The signal rides
    /// the internal inner-cap feedback channel, but its `cap` slot is a private
    /// throwaway so wiring the signal never perturbs the caller's own inner-cap
    /// scheduling (custom families hold their real inner cap separately).
    /// Objectives that do not warm-start, or never near-separate, simply never
    /// observe the flag raised.
    ///
    /// `accepted_steps` becomes the channel's accepted-step counter, which the
    /// optimizer's accept observer advances once per accepted outer step. The
    /// caller reads it in the same closure so that only an accepted iterate
    /// seeds its warm start. A rejected trial that seeded the next trial with its
    /// own inner mode made the objective depend on search history (#2668).
    pub fn with_stuck_stall_cold_reeval_signal(
        self,
        signal: Arc<AtomicBool>,
        accepted_steps: Arc<AtomicUsize>,
    ) -> Self {
        self.with_outer_inner_cap(InnerProgressFeedback {
            cap: Arc::new(AtomicUsize::new(0)),
            accepted_iter: accepted_steps,
            // `last_iters == 0` ⇒ `snapshot()` returns `None` ⇒ no cap-schedule
            // adaptation is derived from this dummy; `last_converged == true`
            // matches the `None` default of `inner_solve_converged`, so
            // terminal-fidelity gating is byte-for-byte unchanged.
            last_iters: Arc::new(AtomicUsize::new(0)),
            last_converged: Arc::new(AtomicBool::new(true)),
            ift_residual: Arc::new(AtomicU64::new(f64::NAN.to_bits())),
            accept_rho: Arc::new(AtomicU64::new(f64::NAN.to_bits())),
            force_cold: signal,
        })
    }

    /// Require the returned fit's projected outer gradient norm to satisfy
    /// `|Pg| <= requirement`, and make the SEARCH pursue it (#2568).
    ///
    /// This is the caller-side floor the engine's own stationarity bound could
    /// not previously express. Without it the engine's data-scaled ladder is the
    /// only standard applied, and because that ladder is a function of the fit,
    /// a fit can be certified at `|Pg| = 5.564e-1` against a bound that
    /// saturated to exactly `1.000e0` while its companion on the same data in
    /// the same call was held to `2.0708e-4`.
    ///
    /// Two consequences, and the first is the point:
    ///
    /// 1. the outer loop's convergence band is floored at `requirement`, so it
    ///    keeps optimizing rather than stopping at a looser sealed bound;
    /// 2. the certificate cannot mint above `requirement` -- the bound ladder is
    ///    capped at it, so an unmet requirement is a typed refusal naming both
    ///    `requirement` and the engine's own bound.
    ///
    /// A refusal is the CORRECT outcome when the requirement is unreachable on
    /// the design: `|Pg| = 5.564e-1` may be a genuine floor of the criterion
    /// there, and saying so beats certifying against `1.0`. Callers who want the
    /// engine's judgement unmodified pass `None`, which is the default and is
    /// byte-for-byte today's behaviour.
    ///
    /// Non-finite and non-positive values are rejected rather than silently
    /// clamped: a requirement of `0.0` or `NaN` is not a stricter standard, it
    /// is an unsatisfiable one, and honouring it would grind the outer loop to
    /// `max_iter` and then refuse every fit.
    pub fn with_required_projected_gradient_norm(mut self, requirement: Option<f64>) -> Self {
        self.required_projected_gradient_norm = requirement.filter(|v| v.is_finite() && *v > 0.0);
        self
    }

    /// Cap the infinity-norm displacement of BFGS cost-only line-search probes
    /// on the **rho axes** (the first `n_params - psi_dim` outer parameters,
    /// = log-λ). Also scales the initial inverse metric so the first trial
    /// direction respects the same local budget coordinate-wise. Documented
    /// natural step on log-λ is ≈ 5; tighter values throttle BFGS and starve
    /// convergence on flat REML valleys.
    pub fn with_bfgs_step_cap(mut self, cap: Option<f64>) -> Self {
        self.bfgs_step_cap = cap.filter(|v| v.is_finite() && *v > 0.0);
        self
    }

    /// Cap the infinity-norm displacement of BFGS cost-only line-search probes
    /// on the **psi axes** (the trailing `psi_dim` outer parameters, = kappa
    /// or anisotropic log-scales). Mirrors [`Self::with_bfgs_step_cap`] but
    /// scoped to kernel-scale parameters whose natural step is much smaller
    /// than log-λ (≈ ln 2 per iter keeps kappa from oscillating). Without
    /// this split, a uniform rho-scale cap lets psi explode while a uniform
    /// psi-scale cap throttles rho — both fail the survival-marginal-slope
    /// path at large scale, where rho needs |d|≈5 while psi wants |d|≤1.
    pub fn with_bfgs_step_cap_psi(mut self, cap: Option<f64>) -> Self {
        self.bfgs_step_cap_psi = cap.filter(|v| v.is_finite() && *v > 0.0);
        self
    }

    pub fn with_cache_session(mut self, session: Arc<CacheSession>) -> Self {
        self.cache_session = Some(session);
        self
    }

    /// Attach mirror cache sessions that receive a broadcast copy of
    /// the final-result finalize write. See
    /// `OuterConfig::cache_mirror_sessions`.
    pub fn with_cache_mirror_sessions(mut self, sessions: Vec<Arc<CacheSession>>) -> Self {
        self.cache_mirror_sessions = sessions;
        self
    }

    pub fn with_problem_size(mut self, n_obs: usize, p_coefficients: usize) -> Self {
        self.problem_size = OuterProblemSize {
            n_obs: Some(n_obs),
            p_coefficients: Some(p_coefficients),
        };
        self
    }

    /// Override the fallback policy. Default is [`FallbackPolicy::Automatic`].
    ///
    /// Set [`FallbackPolicy::Disabled`] when the caller requires the primary
    /// plan to stand on its own. Exact-Hessian objectives use this to ensure
    /// failures surface on the analytic geometry instead of being reinterpreted
    /// by a different optimizer class.
    pub fn with_fallback_policy(mut self, policy: FallbackPolicy) -> Self {
        self.fallback_policy = policy;
        self
    }

    /// Demand a measured PSD analytic Hessian at the terminal mint.
    ///
    /// Use this when a downstream coefficient-mode selection is only defined
    /// for a certified local minimum, rather than for a merely stationary point
    /// whose tiny negative curvature was cleared by the gradient-residue floor.
    pub fn with_require_measured_psd(mut self, required: bool) -> Self {
        self.require_measured_psd = required;
        self
    }

    /// Derive the capability flags from the builder state.
    /// `fixed_point_available` is set to `false` here; `build_objective`
    /// overrides it based on whether an EFS closure is actually provided.
    fn capability(&self) -> OuterCapability {
        OuterCapability {
            gradient: self.gradient,
            hessian: self.hessian,
            prefer_gradient_only: self.prefer_gradient_only,
            disable_fixed_point: self.disable_fixed_point,
            n_params: self.n_params,
            psi_dim: self.psi_dim,
            fixed_point_available: false,
            barrier_config: self.barrier_config.clone(),
        }
    }

    /// Derive the runner configuration from the builder state.
    pub(crate) fn config(&self) -> OuterConfig {
        OuterConfig {
            tolerance: self.tolerance,
            required_projected_gradient_norm: self.required_projected_gradient_norm,
            require_measured_psd: self.require_measured_psd,
            max_iter: self.max_iter,
            model_domain_bounds: self.bounds.clone(),
            model_domain_limit_faces: self.limit_faces.clone(),
            search_bounds_override: None,
            heuristic_log_lambdas: self.heuristic_log_lambdas.clone(),
            initial_rho: self.initial_rho.clone(),
            carried_checkpoint: None,
            initial_inner_seed: self.warm_start.clone(),
            fallback_policy: self.fallback_policy,
            // Only a prior fit's certified point can establish these: the resume
            // attempt `Self::run` derives from a warm start, and the cache's final
            // hit, which says so where it sets `initial_rho`.
            initial_rho_is_prior_terminal_certificate: false,
            resume_value: None,
            outer_inner_cap: self.outer_inner_cap.clone(),
            operator_initial_trust_radius: self.operator_initial_trust_radius,
            arc_initial_regularization: self.arc_initial_regularization,
            bfgs_step_cap: self.bfgs_step_cap,
            bfgs_step_cap_psi: self.bfgs_step_cap_psi,
            cache_session: self.cache_session.clone(),
            cache_mirror_sessions: self.cache_mirror_sessions.clone(),
            problem_size: self.problem_size,
            // Populated only by the persistent-cache resume path in `run` after
            // a warm-start hit decodes a converged outer Hessian.
            warm_start_outer_hessian: None,
            initial_curvature: self.initial_curvature.clone(),
            rho_canonical_keys: self.rho_canonical_keys.clone(),
            // Set only on the recursive canonical run's config (#2817).
            native_coordinate_order: None,
            // Latched only by the certify-last reseed loop (#2939).
            curvature_search_latched: false,
        }
    }

    /// Construct a [`ClosureObjective`] with capability flags derived from the
    /// builder state **and** the closures actually provided.
    ///
    /// `fixed_point_available` is set to `true` when `efs_fn` is `Some`,
    /// regardless of whether `.with_efs()` was called.  This is the canonical
    /// way to create production objectives — it eliminates the drift risk of
    /// manually entering capability flags.
    pub fn build_objective<S, Fc, Fe, Fr, Fefs>(
        &self,
        state: S,
        cost_fn: Fc,
        eval_fn: Fe,
        reset_fn: Option<Fr>,
        efs_fn: Option<Fefs>,
    ) -> ClosureObjective<S, Fc, Fe, Fr, Fefs>
    where
        Fc: FnMut(&mut S, &Array1<f64>) -> Result<f64, EstimationError>,
        Fe: FnMut(&mut S, &Array1<f64>) -> Result<OuterEval, EstimationError>,
        Fr: FnMut(&mut S),
        Fefs: FnMut(&mut S, &Array1<f64>) -> Result<EfsEval, EstimationError>,
    {
        let mut cap = self.capability();
        // Derive fixed_point_available from whether the caller actually
        // provided an EFS hook, rather than relying on manual flags.
        cap.fixed_point_available = efs_fn.is_some();
        ClosureObjective {
            state,
            cap,
            cost_fn,
            eval_fn,
            eval_order_fn: None,
            reset_fn,
            efs_fn,
            fixed_point_certificate_fn: None,
            exact_polish_fn: None,
            rail_face_limit_fn: None,
            criterion_invariance_fn: None,
            criterion_rank_fn: None,
            seed_fn: None::<fn(&mut S, &Array1<f64>) -> Result<SeedOutcome, EstimationError>>,
            terminal_eval_order: None,
        }
    }

    /// Construct a [`ClosureObjective`] with an order-aware evaluation hook.
    ///
    /// This lets the runner request first-order vs second-order work based on
    /// the active outer plan while preserving the legacy eager `eval_fn`.
    pub fn build_objective_with_eval_order<S, Fc, Fe, Feo, Fr, Fefs>(
        &self,
        state: S,
        cost_fn: Fc,
        eval_fn: Fe,
        eval_order_fn: Feo,
        reset_fn: Option<Fr>,
        efs_fn: Option<Fefs>,
    ) -> ClosureObjective<S, Fc, Fe, Fr, Fefs, Feo>
    where
        Fc: FnMut(&mut S, &Array1<f64>) -> Result<f64, EstimationError>,
        Fe: FnMut(&mut S, &Array1<f64>) -> Result<OuterEval, EstimationError>,
        Feo: FnMut(&mut S, &Array1<f64>, OuterEvalOrder) -> Result<OuterEval, EstimationError>,
        Fr: FnMut(&mut S),
        Fefs: FnMut(&mut S, &Array1<f64>) -> Result<EfsEval, EstimationError>,
    {
        let mut cap = self.capability();
        cap.fixed_point_available = efs_fn.is_some();
        ClosureObjective {
            state,
            cap,
            cost_fn,
            eval_fn,
            eval_order_fn: Some(eval_order_fn),
            reset_fn,
            efs_fn,
            fixed_point_certificate_fn: None,
            exact_polish_fn: None,
            rail_face_limit_fn: None,
            criterion_invariance_fn: None,
            criterion_rank_fn: None,
            seed_fn: None::<fn(&mut S, &Array1<f64>) -> Result<SeedOutcome, EstimationError>>,
            terminal_eval_order: None,
        }
    }

    /// Run the outer optimization with a given objective.
    pub fn run(
        &self,
        obj: &mut dyn OuterObjective,
        context: &str,
    ) -> Result<OuterResult, EstimationError> {
        let mut config = self.config();
        let objective_lower = obj.outer_domain_lower_bound()?;
        let objective_upper = obj.outer_domain_upper_bound()?;
        if objective_lower.is_some() || objective_upper.is_some() {
            install_objective_domain(&mut config, self.n_params, objective_lower, objective_upper)?;
        }
        // A warm start (`with_warm_start`): on the parent's inputs the point is
        // offered as a prior certificate, and a decline leaves this search to run
        // exactly as it runs cold; on other inputs this search, which certifies
        // from its one derived start, runs cold. One warm-start source per
        // search, so the cold run takes no cache session either.
        if self.warm_start_source.is_some() {
            if let Some(warm_start) = self.warm_start_for_this_search() {
                if warm_start.same_inputs {
                    match run_outer(obj, &resume_config(&config, warm_start), context) {
                        Ok(result) => {
                            warm_start.record(gam_model_api::WarmStartOutcome::Resumed);
                            return Ok(result);
                        }
                        Err(decline) if self.resume_only => return Err(decline),
                        Err(decline) => {
                            log::debug!(
                                "[OUTER] {context}: the parent's certified point is not certified \
                                 for this search's criterion ({decline}); running the cold search"
                            );
                            warm_start
                                .record(gam_model_api::WarmStartOutcome::NotUsed(RESUME_DECLINED));
                            obj.reset();
                        }
                    }
                } else {
                    warm_start.record(gam_model_api::WarmStartOutcome::NotUsed(
                        "the fit's inputs differ from the parent's, and this search certifies from \
                         its one derived start, so a new start would change the point it reports",
                    ));
                }
            }
            return run_outer(obj, &config, context);
        }
        let Some(session) = config.cache_session.clone() else {
            return run_outer(obj, &config, context);
        };
        let key_hex = session.key().to_hex();
        let short_key = &key_hex[..8.min(key_hex.len())];
        let mut had_hit = false;
        let mut cached_inner_seed: Option<BoundInnerSeed> = None;
        if let Some(loaded) = session.try_load_with_source() {
            match classify_cache_entry_for_outer(&loaded, self.n_params) {
                CacheSeedDecision::ExactFinal {
                    rho,
                    beta,
                    iterations,
                    prior_obj_display,
                } => {
                    log::debug!(
                        "[CACHE] final-hit key={}.. context={} rho_dim={} prior_obj={:.6e} iter={} action=resume-and-recertify",
                        short_key,
                        context,
                        rho.len(),
                        prior_obj_display,
                        iterations,
                    );
                    config.initial_rho = Some(rho.clone());
                    config.initial_rho_is_prior_terminal_certificate = true;
                    if !beta.is_empty() {
                        cached_inner_seed = Some(BoundInnerSeed {
                            theta: rho,
                            beta: Array1::from_vec(beta),
                        });
                    }
                    had_hit = true;
                }
                CacheSeedDecision::Seed {
                    rho,
                    beta,
                    hessian,
                    prior_obj_display,
                    iteration,
                } => {
                    let beta_len = beta.len();
                    let beta_arr = if beta.is_empty() {
                        None
                    } else {
                        Some(Array1::from_vec(beta))
                    };
                    // Adopt the transferred converged outer Hessian only when it
                    // matches this fit's full-θ dimension; a dimension drift
                    // (structural change the cache key did not capture) falls
                    // back to the scalar warm metric in run_plan.
                    config.warm_start_outer_hessian = if self.hessian.is_analytic() {
                        hessian.and_then(|(dim, flat)| {
                            if dim == self.n_params && flat.len() == dim * dim {
                                Array2::from_shape_vec((dim, dim), flat).ok()
                            } else {
                                None
                            }
                        })
                    } else {
                        None
                    };
                    if config
                        .initial_rho
                        .as_ref()
                        .is_none_or(|initial| initial != rho)
                    {
                        log::debug!(
                            "[CACHE] hit  key={}.. context={} rho_dim={} beta_dim={} prior_obj={:.6e} iter={}",
                            short_key,
                            context,
                            rho.len(),
                            beta_len,
                            prior_obj_display,
                            iteration,
                        );
                        config.initial_rho = Some(rho.clone());
                        had_hit = true;
                    } else {
                        log::debug!(
                            "[CACHE] hit  key={}.. context={} rho_dim={} beta_dim={} already-aligned prior_obj={:.6e}",
                            short_key,
                            context,
                            rho.len(),
                            beta_len,
                            prior_obj_display,
                        );
                        had_hit = true;
                    }
                    if let Some(beta) = beta_arr {
                        cached_inner_seed = Some(BoundInnerSeed { theta: rho, beta });
                    }
                }
                CacheSeedDecision::Discard {
                    reason: "payload-shape-mismatch",
                    ..
                } => {
                    log::debug!(
                        "[CACHE] skip key={}.. context={} reason=payload-shape-mismatch n_params={}",
                        short_key,
                        context,
                        self.n_params,
                    );
                }
                CacheSeedDecision::Discard {
                    reason,
                    prior_obj_display,
                    all_rho_finite,
                } => {
                    log::debug!(
                        "[CACHE] skip key={}.. context={} reason={} prior_obj={:.6e} all_rho_finite={}",
                        short_key,
                        context,
                        reason,
                        prior_obj_display,
                        all_rho_finite.unwrap_or(false),
                    );
                }
            }
        } else {
            log::debug!(
                "[CACHE] miss key={}.. context={} reason=fresh-fingerprint n_params={}",
                short_key,
                context,
                self.n_params,
            );
        }
        // Preserve the ownership relation between a cached coefficient vector
        // and the exact outer coordinate that produced it. The runner installs
        // this seed only after resetting for that bitwise-matching candidate;
        // it is never replayed at another generated seed.
        config.initial_inner_seed = cached_inner_seed;
        let mut checkpointing = CheckpointingObjective::new(
            obj,
            Arc::clone(&session),
            config.cache_mirror_sessions.clone(),
        );
        let result = run_outer(&mut checkpointing, &config, context);
        // Attach β only when a beta-bearing evaluation surfaced it at this
        // exact final ρ. Scalar terminal audits carry no β and may follow an
        // evaluation elsewhere; a bare "last β" would manufacture a false
        // (ρ, β) pair and let machine history select the next fit's basin
        // (#2486). A rho-only payload is slower to resume but remains honest.
        let final_beta = result
            .as_ref()
            .ok()
            .and_then(|result| checkpointing.inner_beta_for(&result.rho));
        if let Ok(result) = result.as_ref()
            && result.final_value.is_finite()
            && result.converged()
            && result
                .criterion_certificate
                .as_ref()
                .is_some_and(OuterCriterionCertificate::certifies)
            && let Some(bytes) = encode_iterate(
                &result.rho,
                final_beta.as_ref(),
                result.final_hessian.as_ref(),
                result.final_value,
                result.iterations as u64,
            )
        {
            let saved = session.finalize(
                &bytes,
                Some(result.final_value),
                Some(result.iterations as u64),
            );
            if saved {
                log::debug!(
                    "[CACHE] save key={}.. context={} final_obj={:.6e} iter={} resumed={}",
                    short_key,
                    context,
                    result.final_value,
                    result.iterations,
                    had_hit,
                );
            }
            // Broadcast finalize to mirror keys. The seed-prefix mirror
            // exists so future fits with related-but-not-identical
            // structure can warm-start from this run via the dispatcher's
            // prefix lookup.
            for mirror in &config.cache_mirror_sessions {
                let mirror_saved = mirror.finalize(
                    &bytes,
                    Some(result.final_value),
                    Some(result.iterations as u64),
                );
                if mirror_saved {
                    let mirror_hex = mirror.key().to_hex();
                    log::debug!(
                        "[CACHE] save key={}.. context={} mirror final_obj={:.6e} iter={}",
                        &mirror_hex[..8.min(mirror_hex.len())],
                        context,
                        result.final_value,
                        result.iterations,
                    );
                }
            }
        }
        result
    }

    /// Run the outer optimization and return an unforgeable certified-result
    /// carrier.  Callers that only need checkpoints or diagnostics should use
    /// [`Self::run`]; fit assembly after an optimized outer coordinate must use
    /// this boundary so a caller-constructed [`OuterResult`] cannot mint
    /// convergence provenance.
    pub fn run_certified(
        &self,
        obj: &mut dyn OuterObjective,
        context: &str,
    ) -> Result<CertifiedOuterResult, EstimationError> {
        let result = self.run(obj, context)?;
        CertifiedOuterResult::from_optimizer_result(result).map_err(|reason| {
            EstimationError::RemlOptimizationFailed(format!(
                "{context}: outer result failed certified-fit validation: {reason}"
            ))
        })
    }
}

/// Internal outcome of one planned solver/multistart attempt.
///
/// Exhausted checkpoints carry resumable work only. They never pass through
/// finalization, cache promotion, or fitted-model
/// construction.
pub(crate) enum PlanRunOutcome {
    Converged(OuterResult),
    Exhausted(OuterResult),
    FirstOrderFallbackRequested(FirstOrderFallbackRequest),
    FixedPointContinuationRequested(FixedPointContinuationRequest),
    /// A certified candidate an evaluated state beat (#2596, #2627); see
    /// [`DominatedPlateau`].
    DominatedPlateau(DominatedPlateau),
}

/// A certified candidate that an evaluated but uncertified state of the same
/// attempt beats by more than the criterion's rounding envelope (#2596, #2627).
///
/// A certificate says the candidate is stationary, not that it is the best point
/// the search measured. On a face of the declared domain the criterion is flat
/// and `|Pg|` is negligible whatever it scores, so a face seed certifies in zero
/// iterations while the interior searches that reached far lower values fail to
/// certify. Such a candidate is not published. The incumbent is the resume
/// checkpoint, so the search continues from it.
pub(crate) struct DominatedPlateau {
    /// The certified candidate the attempt declined to publish.
    pub(crate) plateau: OuterResult,
    /// The lowest evaluated state of the attempt, re-evaluated at its own ρ, or at
    /// its stored value when the objective refused that re-evaluation (#2953). A
    /// continuation that ends lower moves it there.
    pub(crate) incumbent: OuterResult,
    /// The plateau's value minus the incumbent's value, at the decline.
    pub(crate) gap: f64,
    /// `outer_value_agreement_bound(plateau, incumbent)`, the resolution the gap
    /// was judged against.
    pub(crate) band: f64,
    /// How the search from the incumbent ended (#2953).
    pub(crate) continuation: DominanceContinuationStop,
}

/// How the search from the state that beat a declined certified optimum ended (#2953).
#[derive(Clone, Debug, PartialEq)]
pub enum DominanceContinuationStop {
    /// No continuation ran: the attempt was itself a continuation, which starts no other.
    NotRun,
    /// The continuation exhausted without certifying.
    Exhausted { final_value: f64 },
    /// The continuation declined another certified optimum.
    DominatedAgain { plateau_value: f64 },
    /// The continuation certified under the screening certificate, which the terminal
    /// certificate then refused.
    Certified { final_value: f64 },
    /// The continuation could not run.
    Failed { error: String },
}

impl std::fmt::Display for DominanceContinuationStop {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::NotRun => f.write_str("was not run, because the attempt was itself a continuation"),
            Self::Exhausted { final_value } => {
                write!(f, "exhausted at objective {final_value:.6e} without certifying")
            }
            Self::DominatedAgain { plateau_value } => write!(
                f,
                "declined another certified optimum at objective {plateau_value:.6e}"
            ),
            Self::Certified { final_value } => write!(
                f,
                "certified at objective {final_value:.6e} under the screening certificate, which \
                 the terminal certificate did not confirm"
            ),
            Self::Failed { error } => write!(f, "could not run ({error})"),
        }
    }
}

/// A certified optimum a plan attempt declined because an evaluated state beat it,
/// carried on the result the plan loop returns so a terminal refusal can report it
/// (#2953).
#[derive(Clone, Debug, PartialEq)]
pub struct DominatedPlateauRecord {
    /// Where the declined optimum sits.
    pub plateau_rho: Array1<f64>,
    /// The declined optimum's value.
    pub plateau_value: f64,
    /// `plateau_value` minus the value of the state that beat it: re-evaluated, or
    /// stored when the objective refused the re-evaluation.
    pub gap: f64,
    /// The criterion's rounding envelope the gap was judged against.
    pub band: f64,
    /// How the search from that state ended.
    pub continuation: DominanceContinuationStop,
}

/// A checkpoint as a later search carries it: the same state, with its iteration count
/// zeroed because the search that produced it has already counted that work (#2953).
pub(crate) fn carried_checkpoint_of(checkpoint: &OuterResult) -> OuterResult {
    let mut carried = checkpoint.clone();
    carried.iterations = 0;
    carried
}

/// Of two declined optima, the one a refusal reports: the lower (#2953).
pub(crate) fn lowest_dominated_plateau(
    kept: Option<DominatedPlateauRecord>,
    candidate: Option<DominatedPlateauRecord>,
) -> Option<DominatedPlateauRecord> {
    match (kept, candidate) {
        (Some(kept), Some(candidate)) if candidate.plateau_value < kept.plateau_value => {
            Some(candidate)
        }
        (Some(kept), _) => Some(kept),
        (None, candidate) => candidate,
    }
}

/// Which certificate concluded a CONVERGED outer run (#2235/#2241).
///
/// `OuterResult.converged == true` bundles genuinely different endings, each
/// with its own certificate. Distinguishing them is pure evidence for the
/// caller's termination report — every variant is a converged fit. There is
/// deliberately no "budget/freeze" variant: exhaustion is a typed error
/// carrying the resume checkpoint, never a minted fit (SPEC 20; the #2235
/// forcing-function redesign deleted the freeze lanes).
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum OuterConvergedVia {
    /// The bound-projected analytic gradient at the returned point cleared the
    /// solver's absolute/score-scaled stationarity tolerance.
    GradientStationary,
    /// Criterion-flat certificate (#2241/#2253): the criterion stalled over the
    /// cost-stall window and the residual projected gradient sits inside the
    /// flat certificate band — the point-anchored band and/or the
    /// curvature-scaled Newton-decrement bound (`newton_predicted_decrease`),
    /// under which a residual above the gradient-magnitude bands is still
    /// stationary when the second-order-predicted improvement `½·gᵀH⁻¹g` is below
    /// the outer objective tolerance. `certificate_bound` is the operative
    /// (widened) bound the residual actually cleared.
    CriterionFlat {
        residual_grad_norm: f64,
        certificate_bound: f64,
    },
    /// Every optimized coordinate carried an explicit analytic fixed-point
    /// equation and the KKT-projected residual cleared the solver tolerance.
    FixedPointStationary {
        projected_residual_inf_norm: f64,
        certificate_bound: f64,
    },
    /// Fellner–Schall model-state fixed point (#2235 verdict 2): two
    /// consecutive outer evaluations restored the same banked incumbent, so a
    /// further outer update provably does not change the fitted state. The
    /// analytic first-order certificate is still taken at the incumbent.
    RecurrentIncumbent { consecutive_restores: usize },
    /// Stationary-at-asymptote (#2348 Inc 1 / #2299 layer 3): the interior
    /// (non-railed) coordinates are gradient-stationary, and every coordinate
    /// railed at the infinite-/zero-smoothing box bound is certified on a
    /// confirmed exponential tail (Thm 2.1) whose fitted model has reached the
    /// rail limit to within the estimand tolerance. The typed rail supersedes
    /// the generic gradient/criterion-flat verdict for a railed optimum.
    AsymptoteStationary { rails: usize },
}

impl OuterConvergedVia {
    /// Stable wire name for termination reports; the enum owns the vocabulary
    /// so bindings marshal instead of mapping.
    pub fn as_str(&self) -> &'static str {
        match self {
            Self::GradientStationary => "converged_stationary",
            Self::CriterionFlat { .. } => "converged_criterion_flat",
            Self::FixedPointStationary { .. } => "converged_fixed_point",
            Self::RecurrentIncumbent { .. } => "incumbent_stationary",
            Self::AsymptoteStationary { .. } => "converged_asymptote_rail",
        }
    }
}

/// Typed lifecycle of an outer optimization result.
///
/// A solver claim and an analytic certificate are different stages, but they
/// belong to one state machine. Encoding them in one enum makes it impossible
/// to retain a certified success verdict after a later certificate refusal.
#[derive(Clone, Copy, Debug, PartialEq)]
enum OuterTermination {
    /// The solver exhausted or refused without claiming convergence.
    Exhausted,
    /// The solver claimed convergence, but no terminal analytic certificate
    /// currently authorizes the point. `proposed_via` is reserved for the
    /// recurrent-incumbent fixed-point signal that certification must corroborate.
    SolverClaimed {
        proposed_via: Option<OuterConvergedVia>,
    },
    /// A terminal analytic certificate authorizes the point and records why.
    Certified(OuterConvergedVia),
}

impl OuterTermination {
    fn from_solver_claim(claimed: bool) -> Self {
        if claimed {
            Self::SolverClaimed { proposed_via: None }
        } else {
            Self::Exhausted
        }
    }

    fn solver_claimed_convergence(self) -> bool {
        !matches!(self, Self::Exhausted)
    }

    fn is_certified(self) -> bool {
        matches!(self, Self::Certified(_))
    }

    fn certified_via(self) -> Option<OuterConvergedVia> {
        match self {
            Self::Certified(via) => Some(via),
            Self::Exhausted | Self::SolverClaimed { .. } => None,
        }
    }

    fn proposed_via(self) -> Option<OuterConvergedVia> {
        match self {
            Self::SolverClaimed { proposed_via } => proposed_via,
            Self::Exhausted | Self::Certified(_) => None,
        }
    }

    /// Revoke any earlier screening certificate before measuring a new
    /// screening/mint verdict. Only the model-state recurrent-incumbent signal
    /// survives as a proposal; ordinary gradient/rail verdicts must be re-earned.
    fn begin_certification(&mut self) {
        let proposed_via = match *self {
            Self::SolverClaimed {
                proposed_via: Some(via @ OuterConvergedVia::RecurrentIncumbent { .. }),
            }
            | Self::Certified(via @ OuterConvergedVia::RecurrentIncumbent { .. }) => Some(via),
            Self::Exhausted
            | Self::SolverClaimed { .. }
            | Self::Certified(_) => None,
        };
        if !matches!(*self, Self::Exhausted) {
            *self = Self::SolverClaimed { proposed_via };
        }
    }

    fn certify(&mut self, via: OuterConvergedVia) {
        *self = Self::Certified(via);
    }

    /// A refused analytic pass may retain the factual solver claim for resume
    /// policy, but never a proposed or certified success verdict.
    fn refuse_certificate(&mut self) {
        if !matches!(*self, Self::Exhausted) {
            *self = Self::SolverClaimed { proposed_via: None };
        }
    }
}

/// Which lane actually produced an [`OuterResult`].
///
/// `OuterResult::solver_termination` is `None` whenever no `opt` solver
/// produced the result, and its own doc names three ways that happens — a
/// cache short-circuit, a synthesized checkpoint, the per-atom Fellner–Schall
/// lane — without recording WHICH. A refusal then reads
/// `termination=<no opt solver produced this result>` and the reader is left to
/// infer the lane from iteration counts, which is the #2465 shape: the decision
/// is made, and the basis for it is dropped by the emitter that holds it. On
/// the #1575 binomial fixture that absence is the whole question — a result at
/// `|Pg| = 5.991e-1` after 13 of 300 permitted outer iterations, PSD Hessian,
/// nothing railed, is a search that stopped with descent still available, and
/// "which lane stopped it" is the first thing anyone needs.
///
/// [`OuterResult::new`] defaults to [`Self::Solver`]; the gam-side lanes that
/// synthesize a result overwrite it at their construction site.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum OuterResultOrigin {
    /// An `opt` solver ran and its solution was translated into this result.
    Solver,
    /// A seed was accepted as its own optimum with zero outer iterations.
    SeedAcceptedWithoutIteration,
    /// ARC exhausted its budget on a last iterate WORSE than the best feasible
    /// iterate it had seen, so the best iterate was substituted (#1371/#1476).
    ArcBestIterateSubstitution,
    /// A fixed-point walk (EFS / HybridEFS) exhausted its budget on a last iterate
    /// WORSE than the best iterate it had evaluated, so the best iterate was
    /// substituted, as ARC does (#1371, #2817).
    FixedPointBestIterateSubstitution,
    /// ARC hit a run of infeasible probes with no synchronized Hessian, so a
    /// checkpoint was rebuilt from the stored best iterate.
    ArcInfeasibleStallCheckpoint,
    /// A filled cost-stall window on the ARC route carried no progress since the
    /// previous one: no resolved descent and no contraction of the incumbent's
    /// projected gradient. The search was stopped at its best feasible iterate
    /// (#2817).
    ArcUnprogressingStallCheckpoint,
    /// The same unprogressing stop on the matrix-free trust-region route, which
    /// has no stall stop of its own (#2817).
    OperatorUnprogressingStallCheckpoint,
    /// ARC was stopped at a point its own terminal certificate accepts: the
    /// Newton decrement ½gᵀH⁻¹g of the rail-projected gradient sat at or below
    /// the criterion's resolution under a PSD reduced Hessian (#2817), or a
    /// strict-saddle incumbent inside the solver band carried negative curvature
    /// the criterion contradicted (#1082).
    ArcCurvatureStationaryStop,
    /// The BFGS cost-stall guard halted the search and published its best
    /// iterate, which was rebuilt into this result.
    BfgsCostStallExit,
    /// The per-atom Fellner–Schall frontier lane.
    PerAtomFellnerSchall,
    /// The parameter space is empty; there was nothing to optimize.
    EmptyParameterSpace,
    /// A caller-supplied point audited without running any optimizer.
    StationaryPointAudit,
}

/// Result of a completed outer optimization.
#[derive(Clone, Debug)]
pub struct OuterResult {
    /// Optimized log-smoothing parameters.
    pub rho: Array1<f64>,
    /// Final objective value.
    pub final_value: f64,
    /// Total outer iterations across all solver restarts.
    pub iterations: usize,
    /// Final gradient norm, when the solver computed an actual gradient.
    pub final_grad_norm: Option<f64>,
    /// Final value and gradient when the solver is gradient-based, with the ρ
    /// they were measured at.
    pub final_measurement: Option<OuterFirstOrderMeasurement>,
    /// The measurement a certificate pass displaced from `final_measurement`
    /// when it re-measured this ρ. A later pass at the same ρ re-measures from
    /// the same reset state and replays the earlier pass bit for bit, so without
    /// this record it would hold two copies of one measurement and never the
    /// independent one the solver took.
    pub displaced_measurement: Option<OuterFirstOrderMeasurement>,
    /// Final Hessian when the solver tracks one.
    pub final_hessian: Option<Array2<f64>>,
    /// Single authoritative termination lifecycle. Private so downstream
    /// callers cannot manufacture convergence without the optimizer transition.
    termination: OuterTermination,
    /// Which plan was actually used (may differ from initial if fallback fired).
    pub plan_used: OuterPlan,
    /// Final trust radius for the internal operator trust-region solver.
    ///
    /// A non-converged operator-ARC attempt may be restarted by the budget
    /// ladder. Restarting only from the last θ but resetting the trust radius
    /// is not a warm start: it replays the same rejected large trial steps.
    /// Carry this globalization state so retries resume from the scale the
    /// previous attempt already learned.
    pub operator_trust_radius: Option<f64>,
    /// Why the internal operator trust-region solver stopped.
    ///
    /// Derived from `Self::termination` in
    /// `bridges::solution_into_outer_result`; do not set it independently
    /// or the two can disagree.
    pub operator_stop_reason: Option<OperatorTrustRegionStopReason>,
    /// Which test the underlying `opt` solver stopped on, and the
    /// quantity it was decided against.
    ///
    /// Distinct from `OuterTermination`, which is gam's *certification*
    /// state machine (did a terminal analytic certificate authorize this
    /// point). This is the solver's own account of why it stopped
    /// searching, and the two answer different questions: a run can stop
    /// on a satisfied gradient test and still fail certification, or
    /// exhaust its budget and be certified by a later re-measurement.
    ///
    /// Carried from the solver rather than reconstructed here. Before
    /// `opt` reported this, `operator_stop_reason` was hand-populated on
    /// the matrix-free branch alone by matching the coarse
    /// `OptimizationStatus`, so every other route left it `None` — and
    /// `None` rendered identically to "there was nothing to say", which
    /// made #2547's "WHY it stopped is unrecorded" unanswerable without a
    /// new probe.
    ///
    /// `None` here means no `opt` solver produced this result at all (a
    /// cache short-circuit, a synthesized checkpoint, the per-atom
    /// Fellner–Schall lane) — an honest absence, not a dropped verdict.
    pub solver_termination: Option<TerminationReason>,
    /// First-order optimality self-audit at the returned point (#934).
    ///
    /// `None` when no analytic gradient was measured at termination
    /// (gradient-free solvers, cache-hit short-circuits, per-atom EFS) or
    /// when an audit probe failed to evaluate. Populated once by
    /// `run_outer` after the solver ladder returns, outside all hot loops.
    pub criterion_certificate: Option<OuterCriterionCertificate>,
    /// Probe-noise-floor gradient bound measured by the cost-stall guard at a
    /// halted stall (#2241): σ̂/Δ, the criterion's evaluation-noise floor over
    /// the stall window divided by the radius the accepted steps actually
    /// probed. Present only on results rebuilt from a cost-stall exit;
    /// `certify_outer_optimality` folds it into the stationarity bound so the
    /// final re-measured gradient is judged against the same flat certificate
    /// the guard granted.
    /// Why the solver's line search gave up, when it did (#2465).
    ///
    /// `opt` reports `LineSearchFailureReason` plus the attempt count on
    /// `BfgsError::LineSearchFailed`, and the two variants have opposite
    /// causes: `StepSizeTooSmall` means the direction WAS a descent direction
    /// and no usable step decreased the objective — an objective/gradient
    /// inconsistency or evaluation noise — while `MaxAttempts` means the
    /// bracketing never closed, a pathological landscape. The bridge used to
    /// drop both on the floor: a line-search failure whose last iterate is
    /// finite is returned as `Ok(non-converged)`, so the caller never sees the
    /// `Err` that carries them, and the certificate could say only
    /// `termination=line_search_failed(|g|=…)` — the verdict without the
    /// quantity it was decided against.
    ///
    /// `None` means no line search failed (or no `opt` solver produced this
    /// result at all).
    pub line_search_failure: Option<(LineSearchFailureReason, usize)>,
    /// Saddle-escape reseed point minted by a refused certification whose
    /// interior reduced Hessian is a certified strict saddle — small projected
    /// gradient, `hessian_psd = Some(false)`, no railed coordinate (#2357). A
    /// gradient-only convergence gate (ARC's, or the cost-stall guard's) can
    /// ARRIVE at such a saddle with its gradient already below tolerance and
    /// stop, even though the certified negative-curvature eigendirection is a
    /// strict descent direction the optimizer never took. This point is
    /// `ρ + α·v` for the most-negative-curvature eigenvector `v`, stepped off
    /// the saddle ridge to a strictly-lower objective; the plan runner reseeds
    /// the outer search ONCE from it (reseed gate closed so it cannot recurse),
    /// which lets the optimizer descend to the true PSD minimum exactly as an
    /// identical warm-started resume does by hand.
    pub saddle_escape_reseed: Option<Array1<f64>>,
    /// Wrong-rail pull-back reseed point minted by a refused certification whose
    /// coordinate sits AT the ρ box bound but whose clean-band probes prove the
    /// objective DECREASES as the coordinate moves INWARD (#2392). The outer
    /// search drove the coordinate to the wrong bound — its terminal gradient is
    /// deep-λ instrument noise, so the trust region never proposed the large
    /// inward move — while a drift-band-clean, above-noise-floor run of probes a
    /// few e-folds inside carries a pencil constant of the sign OPPOSITE the rail
    /// (descent points away from the bound, `∂V/∂ρ > 0` at an upper rail). This
    /// point moves that coordinate to its clean-band interior scale, where the
    /// gradient is informative again; the plan runner reseeds ONCE (gate closed)
    /// and the optimizer descends to the true interior optimum. Gated strictly on
    /// the opposite-sign clean-tail proof, so a GENUINE rail (descent toward the
    /// bound) never mints it and no real λ→∞ optimum is pulled off its rail.
    pub wrong_rail_reseed: Option<Array1<f64>>,
    /// Active-set reduction reseed minted by a refused certification whose
    /// INTERIOR is not stationary while a coordinate is railed at the ρ box with
    /// a deep-λ noise-floor gradient (#2392). The railed coordinate's
    /// ill-conditioned Hessian row poisons the joint Newton/ARC steps, so the
    /// interior cannot polish; freezing that coordinate at its bound and
    /// re-running lets the optimizer converge the interior in the well-conditioned
    /// REDUCED space. The reseed carries the frozen box (`bounds`, with
    /// `lower[k]==upper[k]==rail` for each frozen coordinate); the plan runner's
    /// re-certification under the ORIGINAL bounds then judges every pinned
    /// coordinate's KKT sign at the reduced optimum (an inward-feasible-descent
    /// gradient unfreezes it — no silent clamping of a coordinate that stops
    /// wanting the rail).
    pub active_set_reseed: Option<ActiveSetReseed>,
    /// `(noise_floor σ̂, probe_radius Δ)` the cost-stall guard measured over its
    /// stall window, when a stall produced this result. Reported as evidence and
    /// licensing no bound (#2817): σ̂ is the per-step objective change the
    /// no-improvement window judged and Δ is the radius the accepted steps moved,
    /// and a window that filled because the search took microscopic steps is told
    /// from one that filled because the surface is flat by Δ.
    pub cost_stall_probe_scale: Option<(f64, f64)>,
    /// Set when the search halted where its kept rank ends (#2939). See
    /// [`RankBoundaryStall`]. Reported, and never a converged claim.
    pub rank_boundary_stall: Option<RankBoundaryStall>,
    /// Which lane produced this result. See [`OuterResultOrigin`].
    pub origin: OuterResultOrigin,
    /// The lowest certified optimum a plan attempt of this search declined because an
    /// evaluated state beat it (#2596, #2627), or `None`. When the terminal certificate
    /// refuses this result and nothing continues it to a certified point, the refusal is
    /// [`EstimationError::DominatedCertifiedPlateau`], which reports it (#2953).
    pub dominated_plateau: Option<DominatedPlateauRecord>,
}

/// What a first-order search publishes when it halts where its kept rank ends (#2939): a
/// filled cost-stall window in which every trial was refused for keeping a different rank
/// than the one the search started on (#2765). The incumbent it halted at is published
/// non-converged, and the terminal certificate judges it.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct RankBoundaryStall {
    /// Kept rank of the face log-determinant this search searched.
    pub kept_rank: usize,
    /// Consecutive trials the filled window refused for leaving that rank.
    pub refused_trials: usize,
    /// The certificate's stationarity band at the incumbent's value, which the incumbent's
    /// projected gradient exceeded.
    pub band: f64,
}

/// An active-set reduction reseed (#2392): re-run the outer search with a set of
/// railed coordinates FROZEN at their box bounds so the optimizer polishes the
/// interior in the reduced space.
#[derive(Clone, Debug)]
pub struct ActiveSetReseed {
    /// The reseed point: the refused checkpoint with the frozen coordinates
    /// pinned at their bounds (`rho[k] == bounds.0[k] == bounds.1[k]`).
    pub rho: Array1<f64>,
    /// The reduced-space box: `lower[k] == upper[k] == rail` for every frozen
    /// coordinate `k`, the original bounds elsewhere.
    pub bounds: (Array1<f64>, Array1<f64>),
}

impl OuterResult {
    pub fn new(
        rho: Array1<f64>,
        final_value: f64,
        iterations: usize,
        solver_claimed_convergence: bool,
        plan_used: OuterPlan,
    ) -> Self {
        Self {
            rho,
            final_value,
            iterations,
            final_grad_norm: None,
            final_measurement: None,
            displaced_measurement: None,
            final_hessian: None,
            termination: OuterTermination::from_solver_claim(solver_claimed_convergence),
            plan_used,
            operator_trust_radius: None,
            operator_stop_reason: None,
            solver_termination: None,
            criterion_certificate: None,
            line_search_failure: None,
            saddle_escape_reseed: None,
            wrong_rail_reseed: None,
            active_set_reseed: None,
            cost_stall_probe_scale: None,
            rank_boundary_stall: None,
            origin: OuterResultOrigin::Solver,
            dominated_plateau: None,
        }
    }

    /// Whether this result owns a terminal analytic convergence certificate.
    pub fn converged(&self) -> bool {
        self.termination.is_certified()
    }

    /// Which analytic certificate concluded this run.
    pub fn converged_via(&self) -> Option<OuterConvergedVia> {
        self.termination.certified_via()
    }

    /// Whether the underlying solver claimed convergence before analytic
    /// certification. Certified results necessarily originated from a claim or
    /// an explicit stationary-point audit.
    pub(crate) fn solver_claimed_convergence(&self) -> bool {
        self.termination.solver_claimed_convergence()
    }

    /// Human-readable rendering of `final_grad_norm` for diagnostics. Returns
    /// `"n/a"` when no gradient was measured (gradient-free / cache-hit paths).
    pub fn final_grad_norm_report(&self) -> String {
        match self.final_grad_norm {
            Some(g) => format!("{g:.3e}"),
            None => "n/a".to_string(),
        }
    }
}

/// Validated evidence that an outer optimization terminated at a finite,
/// analytically certified optimum.
///
/// The inner [`OuterResult`] is private so downstream fit assembly cannot turn
/// a status boolean into convergence provenance. Construction consumes the
/// optimizer result and revalidates the certificate at the ownership boundary.
#[derive(Clone, Debug)]
pub struct CertifiedOuterResult {
    result: OuterResult,
}

impl CertifiedOuterResult {
    /// The sole constructor is reached from [`OuterProblem::run_certified`]
    /// after the optimizer has produced the result.  Keeping this private is
    /// load-bearing: `OuterResult` is also a public diagnostic/checkpoint
    /// payload, so a public conversion would let downstream code fabricate a
    /// certificate-shaped result without ever running an objective.
    fn from_optimizer_result(result: OuterResult) -> Result<Self, String> {
        if !result.converged() {
            return Err(format!(
                "outer optimization did not converge after {} iterations",
                result.iterations
            ));
        }
        if !result.final_value.is_finite() {
            return Err(format!(
                "outer optimization returned a non-finite objective: {}",
                result.final_value
            ));
        }
        if result.rho.iter().any(|value| !value.is_finite()) {
            return Err("outer optimization returned non-finite hyperparameters".to_string());
        }
        if result
            .final_grad_norm
            .is_some_and(|value| !value.is_finite() || value < 0.0)
        {
            return Err(format!(
                "outer optimization returned an invalid gradient norm: {:?}",
                result.final_grad_norm
            ));
        }
        let certificate = result
            .criterion_certificate
            .as_ref()
            .ok_or_else(|| "outer optimization returned no analytic certificate".to_string())?;
        if !certificate.certifies() {
            return Err(format!(
                "outer optimization certificate does not certify: {}",
                certificate.summary()
            ));
        }
        Ok(Self { result })
    }

    /// Exact optimizer-owned hyperparameter vector covered by the certificate.
    pub fn rho(&self) -> &Array1<f64> {
        &self.result.rho
    }

    pub fn iterations(&self) -> usize {
        self.result.iterations
    }

    pub fn final_value(&self) -> f64 {
        self.result.final_value
    }

    pub fn final_grad_norm(&self) -> Option<f64> {
        self.result.final_grad_norm
    }

    /// Exact analytic gradient re-measured by the optimizer-owned terminal
    /// certificate. Downstream selected-profile finalizers use this to prove
    /// that a retained objective payload is the one certified at `rho()`.
    pub fn final_gradient(&self) -> Option<&Array1<f64>> {
        self.result.final_gradient()
    }

    pub fn criterion_certificate(&self) -> &OuterCriterionCertificate {
        self.result
            .criterion_certificate
            .as_ref()
            .expect("CertifiedOuterResult always owns a validated certificate")
    }

    /// The analytic outer ρ-Hessian measured at the certified point, when the
    /// certification retained one. This is the curvature evidence behind the
    /// certificate's PSD verdict — and the `V_ρ = H_ρ⁻¹` input to first-order
    /// smoothing-correction inflation (#2346).
    pub fn final_hessian(&self) -> Option<&Array2<f64>> {
        self.result.final_hessian.as_ref()
    }
}

#[cfg(test)]
#[path = "certified_outer_result_tests.rs"]
mod certified_outer_result_tests;

#[path = "multistart.rs"]
mod multistart;
pub use multistart::MultistartOutcome;

/// Typed refusal from [`audit_stationary_point`]. The rejected point and every
/// analytic certificate field measured before refusal remain available to the
/// caller; `source` records why those measurements did not certify.
#[derive(Debug)]
pub struct OuterStationaryPointRejection {
    pub result: OuterResult,
    pub source: EstimationError,
}

impl std::fmt::Display for OuterStationaryPointRejection {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        std::fmt::Display::fmt(&self.source, f)
    }
}

impl std::error::Error for OuterStationaryPointRejection {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        Some(&self.source)
    }
}

/// Apply the shared analytic outer-optimality authority to one caller-supplied
/// point without running an optimizer or taking a step.
///
/// The objective controls whether evaluating that point mutates its profiled
/// state. Callers auditing an already-installed inner state must put their
/// objective in a frozen evaluation mode before calling this function.
/// `iterations == 0` in the returned result is structural: no optimization loop
/// exists on this path.
///
/// `n_obs` and `p_coefficients` are the problem size the criterion is summed
/// over, exactly as [`OuterProblem::with_problem_size`] declares them for a
/// search: the point is judged against the same statistical resolution
/// `τ_stat = 1/(2n)` a search of that criterion would stop at.
pub fn audit_stationary_point(
    obj: &mut dyn OuterObjective,
    rho: Array1<f64>,
    n_obs: usize,
    p_coefficients: usize,
    context: &str,
) -> Result<OuterResult, OuterStationaryPointRejection> {
    let config = OuterConfig {
        problem_size: OuterProblemSize {
            n_obs: Some(n_obs),
            p_coefficients: Some(p_coefficients),
        },
        ..OuterConfig::default()
    };
    audit_stationary_point_in(obj, config, rho, context)
}

/// [`audit_stationary_point`] under a caller's configuration. The point is judged
/// against the domain [`OuterProblem::run`] would search from that configuration:
/// the objective's declared faces within the configured model domain.
pub(crate) fn audit_stationary_point_in(
    obj: &mut dyn OuterObjective,
    mut config: OuterConfig,
    rho: Array1<f64>,
    context: &str,
) -> Result<OuterResult, OuterStationaryPointRejection> {
    let capability = obj.capability();
    let selected_plan = plan(&capability);
    // There is intentionally no independent value-only probe. The analytic
    // sample is the authority being audited, and infinity records that no
    // optimizer-produced terminal value exists to compare against it.
    let mut result = OuterResult::new(rho, f64::INFINITY, 0, false, selected_plan);
    result.origin = OuterResultOrigin::StationaryPointAudit;
    let installed = obj
        .outer_domain_lower_bound()
        .and_then(|lower| obj.outer_domain_upper_bound().map(|upper| (lower, upper)))
        .and_then(|(lower, upper)| {
            if lower.is_some() || upper.is_some() {
                install_objective_domain(&mut config, capability.n_params, lower, upper)
            } else {
                Ok(())
            }
        });
    if let Err(source) = installed {
        return Err(OuterStationaryPointRejection { result, source });
    }
    match certify_outer_optimality(obj, &config, context, &mut result) {
        Ok(certificate) => {
            result.criterion_certificate = Some(certificate);
            Ok(result)
        }
        Err(source) => Err(OuterStationaryPointRejection { result, source }),
    }
}

// ─── First-order optimality certificate (#934) ────────────────────────
//
// The objective↔gradient desync bug genus (#748, #752, #808, #901, …) has a
// universal signature: at the returned "optimum" the optimizer claims
// convergence while the criterion is not actually stationary there (or the
// optimizer stalls and rails λ). The certificate makes the engine check
// itself, once, at θ̂, on every generic outer fit — purely from the ANALYTIC
// objective, per SPEC rule 2 (finite differences never run outside tests): the
// KKT-projected analytic gradient norm against the same score-relative
// stationarity bound the outer loop already uses to accept flat-valley
// stalls (#1690), a scaled PSD probe of the tracked outer Hessian, and the
// λ-rail facts every desync postmortem asks for. It is the runtime
// enforcement layer for the criterion-atom architecture (#931).
//
// A failed certificate REJECTS the fit as typed non-convergence — never a
// warn-and-continue diagnostic — so a nonstationary point can never be
// minted into a fit (SPEC rule 20).

// The certificate's definiteness verdict, the one owner of the shift it is taken at, and the
// Newton decrement that travels with it are opt's bound-constrained second-order helpers (SPEC
// rule 24, #2900 row 24.2). The shift is the larger of a MEASURED curvature resolution (#2748:
// an outer Hessian is a derivative through an inner solve, a log-determinant and a trace
// contraction, so its error is not bounded by `√ε·‖H‖`) and the arithmetic
// `√ε·max(max|H_ii|, 1)`; `0.0` is the arithmetic shift. A verdict and the shift it was taken at
// travel together, so the smoothing correction re-judges a direction at the certificate's shift
// rather than at its own eigensolver's backward error.
pub(crate) use opt::{
    NegativeCurvatureClaim, certificate_curvature_shift,
    hessian_is_psd_at_resolution as certificate_hessian_is_psd_at_resolution,
    negative_curvature_claim, newton_predicted_decrease, newton_predicted_decrease_at_resolution,
};

/// PSD verdict of the outer Hessian restricted to its UN-RAILED coordinates
/// (#2299 box-KKT reduced-Hessian / critical-cone gate).
///
/// A coordinate railed on its domain face with an outward gradient is at the box-KKT
/// constrained optimum: its curvature direction is the flat/indefinite
/// infinite-smoothing plateau of a fully-saturated penalty (λ ~ 1e13), carrying
/// no feasible descent. Including it makes the FULL Hessian indefinite and used to
/// disable the very flatness certificate that exists to handle rails, so an
/// honest railed optimum ground to `max_iter` and refused. Judging PSD on the
/// INTERIOR (un-railed) sub-block is the standard reduced-Hessian condition: a
/// genuinely indefinite *interior* direction still keeps the sub-block non-PSD,
/// so this can never over-certify a real saddle. When every coordinate is railed
/// the interior is empty — there is no feasible curvature to certify and the rail
/// KKT signs are the whole certificate — so the empty sub-block is trivially PSD.
/// With no railed coordinate and no declared invariance it is exactly
/// [`certificate_hessian_is_psd_at_resolution`] at zero measured resolution.
///
/// # The invariance argument (#2676)
///
/// `invariance` carries the directions along which the criterion is EXACTLY
/// constant by construction of its penalty map (see
/// [`crate::penalty_invariance`]). They are removed from the judged subspace
/// alongside the railed coordinates, for the same reason: a coordinate railed
/// at its bound and a direction the criterion does not vary along are both
/// places where "is the curvature positive?" has no answer that is about the
/// fit. On such a direction `t' H_rho t = sum_k g_k t_k^2` identically, so its
/// sign is the sign of the disagreement between the gradient code and the
/// Hessian code — measured at `|sigma|/floor = 0.99925` on `geo_disease_matern`.
///
/// `None` reproduces the pre-#2676 sub-block extraction bit for bit; that is
/// what every objective declaring no invariance gets, which is nearly all of
/// them.
pub(crate) fn certificate_hessian_is_psd_off_railed(
    hessian: &Array2<f64>,
    railed: &[usize],
    invariance: Option<&Array2<f64>>,
) -> Option<bool> {
    certificate_hessian_is_psd_off_railed_at_resolution(hessian, railed, invariance, 0.0)
}

/// [`certificate_hessian_is_psd_off_railed`] carrying a **measured** curvature
/// resolution down to the definiteness test (#2748).
pub(crate) fn certificate_hessian_is_psd_off_railed_at_resolution(
    hessian: &Array2<f64>,
    railed: &[usize],
    invariance: Option<&Array2<f64>>,
    measured_resolution: f64,
) -> Option<bool> {
    let n = hessian.nrows();
    let deflate = invariance.filter(|basis| basis.nrows() == n && basis.ncols() > 0);
    if railed.is_empty() && deflate.is_none() {
        return certificate_hessian_is_psd_at_resolution(hessian, measured_resolution);
    }
    let judged = crate::penalty_invariance::judged_subspace_basis(n, railed, deflate);
    let Some(judged) = judged else {
        // Nothing left to judge: every coordinate is railed, or the criterion is
        // flat in every direction. There is no feasible curvature to certify and
        // the rail KKT signs are the whole certificate.
        return Some(true);
    };
    if deflate.is_none() {
        // Exactly the historical sub-block extraction: `judged` is the interior
        // indicator basis, so build the sub-block directly rather than through a
        // matrix product, keeping this path bit-identical.
        let railed_set: std::collections::BTreeSet<usize> = railed.iter().copied().collect();
        let interior: Vec<usize> = (0..n).filter(|k| !railed_set.contains(k)).collect();
        let mut sub = Array2::<f64>::zeros((interior.len(), interior.len()));
        for (i, &ri) in interior.iter().enumerate() {
            for (j, &rj) in interior.iter().enumerate() {
                sub[[i, j]] = hessian[[ri, rj]];
            }
        }
        return certificate_hessian_is_psd_at_resolution(&sub, measured_resolution);
    }
    let compressed = crate::penalty_invariance::compress_to_judged_subspace(hessian, &judged);
    certificate_hessian_is_psd_at_resolution(&compressed, measured_resolution)
}

/// The curvature resolution this site is entitled to: `‖δH‖₂` MEASURED from the
/// identities that are exactly zero in exact arithmetic (#2748), never assumed.
///
/// Two are available here and both are free:
///
/// * the **symmetrization defect** `‖(H − Hᵀ)/2‖₂`. A Hessian is symmetric for
///   any twice-continuously-differentiable criterion, and `H[i,j]` and `H[j,i]`
///   are separate accumulations of the same mixed partial, so whatever survives
///   is the assembly's error;
/// * the **penalty-map invariance residual** `‖T'(H − diag(g))T‖₂` on the
///   directions this certificate deflates. Along them the criterion is exactly
///   constant in λ, so `t'H t = Σ_k g_k t_k²` identically and the residual is
///   error and only error — in exactly the currency a `H + diag(|g|)` gate
///   spends.
///
/// Both are certified LOWER bounds on `‖δH‖₂`, so their maximum is the strongest
/// available fact. Returns `0.0` when neither can be taken, which leaves the
/// historical `√ε` shift in force unchanged.
pub(crate) fn measured_outer_curvature_resolution(
    hessian: &Array2<f64>,
    railed: &[usize],
    gradient: &Array1<f64>,
    invariance: Option<&Array2<f64>>,
) -> f64 {
    let n = hessian.nrows();
    if n == 0 || hessian.ncols() != n {
        return 0.0;
    }
    let symmetrization = gam_linalg::matrix::symmetrization_defect_2norm(hessian);
    let deflate = invariance.filter(|basis| basis.nrows() == n && basis.ncols() > 0);
    let invariance_residual = deflate
        .and_then(|basis| crate::penalty_invariance::judged_subspace_basis(n, railed, Some(basis)))
        .and_then(|judged| crate::penalty_invariance::deflated_directions(n, &judged))
        .and_then(|removed| {
            crate::penalty_invariance::invariance_residual_2norm(hessian, gradient, &removed)
        })
        .unwrap_or(0.0);
    symmetrization.max(invariance_residual).max(0.0)
}

/// Interior-PSD verdict judged ABOVE the per-coordinate gradient-residue noise
/// floor (#2349): PSD of `H + diag(|g|)` restricted to the un-excluded
/// coordinates.
///
/// The assembled ρ-Hessian's tail entries carry the #2298 trace-pair
/// cancellation residue: when the `λ²V_λλ` pair cancels to roundoff, the
/// surviving diagonal entry is `λV_λ = g_k` — gradient magnitude, corrupted
/// sign.
/// Measured on the #2349 multinomial checkpoint: the sole interior coordinate
/// had `g₁ = −1.0228e-3`, `H₁₁ = −1.0216e-3` (ratio 0.999), and that single
/// sub-resolution entry was the entire `interior Hessian sub-block not PSD`
/// refusal — the full 6×6 spectrum was `[−1.02e-3, 0.135, …, 0.904]`.
///
/// # Why the floor is safe — instrument resolution, not step economics
///
/// The residue is `O(|g_k|)`, and the sub-block is extracted AFTER flooring, so
/// only the *judged* (un-excluded) coordinates' gradients ever enter: a railed
/// coordinate's large `|g|` (measured 1.40 on the #2349 checkpoint) cannot
/// inflate the floor. Weyl bounds what remains exactly:
///
/// ```text
///     λ_min(H) + min|g| ≤ λ_min(H + diag|g|) ≤ λ_min(H) + max|g|
/// ```
///
/// so the floor absorbs **at most `max_k |g_k|` over the judged coordinates**.
/// Where this verdict can mint, those coordinates have passed gradient
/// stationarity, so that is at most the stationarity bound. A negative
/// eigenvalue smaller than that is not distinguishable from zero *by the
/// instrument that produced it* — the assembled tail entry IS `λV_λ = g`
/// (ratio 0.999 as measured) — while a genuine saddle survives untouched: the
/// #2357 trace's `λ_min ≈ −0.5` against `|g| ≈ 1e-3` floors to `−0.4973`, and a
/// `−1.50e-2` direction still refuses at a `1e-2` bound.
///
/// ## Do not widen this floor beyond `diag(|g|)`
///
/// Earlier revisions of this argument (and #2349 rounds 6/7) justified the
/// floor by an "exploitable improvement `≲ g²/2|H| ≈ 5e-7`". Both halves are
/// wrong and the pair is misleading in the permissive direction:
///
/// * the quoted number is `g²/2 = 5.23e-7`; the quoted formula evaluates to
///   `g²/(2|H|) = 5.12e-4` on the same measured `g = 1.0228e-3`,
///   `|H| = 1.0216e-3` — a 979× discrepancy. Since `|H| ≈ g` at exactly the
///   point this floor serves, that formula collapses to `≈ g/2`;
/// * and `g²/2|H|` is the Newton decrement, which bounds the improvement along
///   a direction of POSITIVE curvature. Along curvature `−ε` the model
///   `−g·t − ½εt²` is unbounded below, so the quantity it purports to bound
///   does not exist in the regime this floor exists for; the step is limited by
///   the trust radius, not the curvature.
///
/// The Weyl bound above is the correct and checkable statement. Anything that
/// widens the floor must re-derive against it, not against `g²/2|H|`.
pub(crate) fn certificate_meets_curvature_requirement(
    certificate: &OuterCriterionCertificate,
    require_measured_psd: bool,
    fidelity: CertificationFidelity,
) -> bool {
    matches!(fidelity, CertificationFidelity::Screening)
        || !require_measured_psd
        || certificate.hessian_psd() == Some(true)
        // #2612: a caller asking for a certified local minimum is asking for the
        // strongest statement the curvature evidence can support. When that
        // evidence has been CONTRADICTED by the criterion — every feasible step
        // along its reported negative eigenvector, over the whole range in which
        // the claim predicts a decrease the criterion can represent, failed to
        // lower the objective — the strongest supportable statement is that no
        // descent along it exists. Refusing here instead would refuse for the
        // ABSENCE of a measurement the route has just shown it cannot make,
        // which is the failure mode this flag's own doc block at
        // `with_require_measured_psd` warns about one case earlier. A claim the
        // criterion cannot resolve at any allowed step (#3036) supports the
        // same statement: no descent along it is representable.
        || certificate.curvature.withdrawn_by_criterion()
}

pub(crate) fn certificate_hessian_is_psd_off_railed_above_gradient_floor(
    hessian: &Array2<f64>,
    excluded: &[usize],
    gradient: &Array1<f64>,
    invariance: Option<&Array2<f64>>,
) -> Option<bool> {
    let n = hessian.nrows();
    if gradient.len() != n {
        return certificate_hessian_is_psd_off_railed(hessian, excluded, invariance);
    }
    let mut floored = hessian.clone();
    for k in 0..n {
        floored[[k, k]] += gradient[k].abs();
    }
    // The resolution is measured on the UNFLOORED `H` against the UNFLOORED
    // gradient: `diag(|g|)` is a deliberate softening of the test, not part of
    // the assembly, and the identities that measure `‖δH‖₂` are identities
    // about `(H, g)` as evaluated (#2748).
    let measured_resolution =
        measured_outer_curvature_resolution(hessian, excluded, gradient, invariance);
    certificate_hessian_is_psd_off_railed_at_resolution(
        &floored,
        excluded,
        invariance,
        measured_resolution,
    )
}

/// Measure the gradient-residue floor's clearance on the interior sub-block:
/// the sub-block's smallest eigenvalue as assembled, the floor it is judged
/// against (`max_k |g_k|` over exactly those coordinates), and whether
/// `H + diag(|g|)` is PSD there.
///
/// This RECORDS the verdict; it does not replace the raw measurement. See
/// [`certificate_hessian_is_psd_off_railed_above_gradient_floor`] for why the
/// Weyl bound makes the floor safe where it can mint, and
/// [`crate::model_types::CurvatureFloorClearance`] for why the two facts are
/// kept apart.
pub(crate) fn interior_curvature_floor_clearance(
    hessian: &Array2<f64>,
    excluded: &[usize],
    gradient: &Array1<f64>,
    invariance: Option<&Array2<f64>>,
) -> Option<CurvatureFloorClearance> {
    use faer::Side;
    use gam_linalg::faer_ndarray::FaerEigh;

    let n = hessian.nrows();
    if n == 0 || hessian.ncols() != n || gradient.len() != n {
        return None;
    }
    let excluded_set: std::collections::BTreeSet<usize> = excluded.iter().copied().collect();
    let interior: Vec<usize> = (0..n).filter(|k| !excluded_set.contains(k)).collect();
    if interior.is_empty() {
        return None;
    }
    let m = interior.len();
    let mut sub = Array2::<f64>::zeros((m, m));
    for (i, &ri) in interior.iter().enumerate() {
        for (j, &rj) in interior.iter().enumerate() {
            sub[[i, j]] = 0.5 * (hessian[[ri, rj]] + hessian[[rj, ri]]);
        }
    }
    if sub.iter().any(|v| !v.is_finite()) {
        return None;
    }
    // #2676: report the minimum of the block the verdict was actually reached
    // on. Reporting the raw interior minimum beside a verdict taken on the
    // deflated complement is what made every historical `[INDEF-HESS]` line
    // ambiguous: the number named a direction the decision no longer involved.
    let deflate = invariance.filter(|basis| basis.nrows() == n && basis.ncols() > 0);
    let sub = match deflate
        .and_then(|basis| crate::penalty_invariance::judged_subspace_basis(n, excluded, Some(basis)))
    {
        Some(judged) => crate::penalty_invariance::compress_to_judged_subspace(hessian, &judged),
        None => sub,
    };
    if sub.nrows() == 0 {
        return None;
    }
    let interior_min_eigenvalue = sub
        .eigh(Side::Lower)
        .ok()?
        .0
        .iter()
        .fold(f64::INFINITY, |acc, v| acc.min(*v));
    if !interior_min_eigenvalue.is_finite() {
        return None;
    }
    // The floor is the largest gradient among EXACTLY the judged coordinates —
    // the excluded ones never enter, so a railed coordinate's large |g| cannot
    // inflate it.
    let gradient_floor = interior
        .iter()
        .fold(0.0_f64, |acc, &k| acc.max(gradient[k].abs()));
    let cleared = certificate_hessian_is_psd_off_railed_above_gradient_floor(
        hessian, excluded, gradient, invariance,
    ) == Some(true);
    // The eigenvalue the verdict was ACTUALLY taken on: the same
    // `H + diag(|g|)`, on the same judged subspace, that
    // `certificate_hessian_is_psd_off_railed_above_gradient_floor` tests
    // (#2748). The two fields above are the ends of the Weyl sandwich
    // `λ_min(H) + min|g| ≤ λ_min(H + diag|g|) ≤ λ_min(H) + max|g|`, and a
    // reader given only the ends cannot tell why a curvature well inside
    // `max_k|g_k|` refused.
    let mut floored = hessian.clone();
    for k in 0..n {
        floored[[k, k]] += gradient[k].abs();
    }
    let floored_block = match invariance
        .filter(|basis| basis.nrows() == n && basis.ncols() > 0)
        .and_then(|basis| {
            crate::penalty_invariance::judged_subspace_basis(n, excluded, Some(basis))
        }) {
        Some(judged) => crate::penalty_invariance::compress_to_judged_subspace(&floored, &judged),
        None => {
            let mut block = Array2::<f64>::zeros((m, m));
            for (i, &ri) in interior.iter().enumerate() {
                for (j, &rj) in interior.iter().enumerate() {
                    block[[i, j]] = 0.5 * (floored[[ri, rj]] + floored[[rj, ri]]);
                }
            }
            block
        }
    };
    let floored_min_eigenvalue = if floored_block.nrows() == 0 {
        0.0
    } else {
        floored_block
            .eigh(Side::Lower)
            .ok()?
            .0
            .iter()
            .fold(f64::INFINITY, |acc, v| acc.min(*v))
    };
    let measured_resolution =
        measured_outer_curvature_resolution(hessian, excluded, gradient, invariance);
    Some(CurvatureFloorClearance {
        interior_min_eigenvalue,
        gradient_floor,
        floored_min_eigenvalue,
        measured_resolution,
        // The shift the verdict was decided at, from the SAME owner the PSD
        // test calls and on the SAME block it tested, so the number recorded
        // and the number applied cannot drift apart (#2748).
        decided_at_resolution: certificate_curvature_shift(&floored_block, measured_resolution),
        cleared,
    })
}

/// What the CRITERION said about a Hessian's reported negative direction
/// (#2357/#2155/#2612).
///
/// The escape has always been able to distinguish "the saddle is real, here is
/// the descending reseed" from "no descending trial was found", but only the
/// first was reported to the caller: the second reached the refusal as `None`,
/// where it was indistinguishable from "the escape was never runnable", and the
/// curvature refusal proceeded on the matrix's word alone.
///
/// Those are three different states and one of them is a MEASUREMENT of the
/// criterion, so they are three variants.
#[derive(Debug)]
pub(crate) enum SaddleAdjudication {
    /// A strictly-descending feasible point exists along the reported direction:
    /// the point is not a minimum, and this is the one-shot reseed.
    Descended(Array1<f64>),
    /// Every feasible step along the reported direction — both signs, from one
    /// e-fold down to the step at which the quadratic model's own predicted
    /// decrease reaches the criterion's resolution — failed to lower the
    /// objective. The claim has been falsified over its whole falsifiable
    /// range.
    Contradicted {
        /// Trials actually evaluated (finite cost, not clamped back onto ρ).
        probed: usize,
        /// Smallest step the ladder reached.
        smallest_step: f64,
        /// `½|λ_min|·α_min²` — what the claim predicted at that step, against
        /// which the criterion's resolution was the stopping standard.
        predicted_at_smallest: f64,
        /// The criterion's own resolution, i.e. the standard that bounded the
        /// ladder.
        objective_resolution: f64,
        /// Best objective seen, against the baseline it had to beat.
        best_seen_cost: f64,
    },
    /// The claim's falsifiable range is empty (#3036): even the largest step
    /// predicts a decrease `½|λ_min|·α_max²` the criterion cannot resolve, so no
    /// trial could confirm or falsify it and none was evaluated. A curvature the
    /// criterion cannot resolve cannot refuse the point.
    Unresolvable {
        /// The eigenvalue in dispute on the judged sub-block.
        lambda_min: f64,
        /// `½|λ_min|·α_max²`, the claim's prediction at its largest step.
        predicted_at_largest: f64,
        /// The resolution that prediction does not exceed.
        objective_resolution: f64,
    },
    /// The adjudication could not be run: no eigen-resolvable negative
    /// direction, nothing left to search after rails and invariance, an
    /// eigensolver failure, or trials that could not be evaluated at all.
    /// Nothing has been established about the point either way.
    Declined(String),
}

/// Escape point off a certified strict saddle in the free (un-railed) subspace
/// (#2357, generalised to the box-constrained case in #2155), and — when no
/// escape exists — the verdict that the criterion has CONTRADICTED the matrix
/// (#2612).
///
/// A gradient-only outer convergence gate — ARC's own, or the cost-stall guard's
/// — can ARRIVE at a point that is first-order stationary (`‖Pg‖ ≤ bound`) yet
/// sits on genuinely indefinite curvature in its INTERIOR (un-railed) directions,
/// and stop there because its gradient already cleared tolerance. The mandatory
/// analytic certificate then refuses the point as `INDEFINITE CURVATURE AT
/// INTERIOR OPTIMUM` — a verdict `certificate_hessian_is_psd_off_railed` reaches
/// on the reduced Hessian restricted to the un-railed coordinates, so it fires
/// whether or not some other coordinate happens to be railed. Such a point is a
/// saddle, not a minimum: the most-negative-curvature eigenvector `v` of that
/// reduced Hessian is a strict, box-feasible descent direction the optimizer
/// never took. An
/// identical warm-started resume escapes it trivially (its fresh cubic step moves
/// off the ridge, which is why the resume converges where the cold run refuses);
/// this reproduces that escape deterministically by stepping `ρ ± α·v` to a
/// strictly-lower objective and handing the point back as a one-shot reseed.
///
/// Termination is guaranteed: along a direction of negative curvature
/// `vᵀHv = λ_min < 0` at a near-stationary gradient,
/// `f(ρ ± αv) = f(ρ) ± α(g·v) + ½α²λ_min + o(α²)` strictly decreases for small
/// enough `α` once the sign is chosen so the first-order term is non-positive, so
/// the finite backtracking below always finds a descending feasible point when
/// one exists inside the box.
///
/// Returns `None` (no reseed; the ordinary refusal proceeds) when the Hessian
/// carries no eigen-resolvable negative direction, or no bounded step along it
/// clears the box projection with a strict objective decrease. Restores the
/// objective's profiled inner state to `rho` before returning either way, so the
/// refusal path that follows measures the checkpoint rather than the last probe.
pub(crate) fn adjudicate_negative_curvature(
    obj: &mut dyn OuterObjective,
    rho: &Array1<f64>,
    gradient: &Array1<f64>,
    hessian: &Array2<f64>,
    railed: &[usize],
    invariance: Option<&Array2<f64>>,
    baseline_cost: f64,
    objective_resolution: f64,
    bounds: &(Array1<f64>, Array1<f64>),
    context: &str,
) -> SaddleAdjudication {
    use faer::Side;
    use gam_linalg::faer_ndarray::FaerEigh;

    let n = hessian.nrows();
    if n == 0 || hessian.ncols() != n || hessian.iter().any(|v| !v.is_finite()) {
        // #2665: every `None` in this function is a DIFFERENT reason the escape
        // did not fire, and the caller records none of them -- the refusal that
        // follows says only that the curvature floor did not clear. On the
        // SAS/mixture cluster the escape is silently absent (measured: no mint
        // line at all, and four resumes at a bitwise-identical rho), and the
        // exits cannot be told apart from the run record. The sibling
        // NOT ATTEMPTED warning above covers the case where this function is
        // never called; these cover the case where it is called and declines.
        return SaddleAdjudication::Declined(format!(
            "the analytic Hessian is not a usable square finite matrix (rows={}, cols={}, \
             all_finite={})",
            n,
            hessian.ncols(),
            hessian.iter().all(|v| v.is_finite()),
        ));
    }
    // The escape direction lives in the INTERIOR (un-railed) subspace — the exact
    // reduced Hessian / critical cone that `certificate_hessian_is_psd_off_railed`
    // judges for the PSD verdict. A coordinate railed at a box bound with an
    // outward KKT gradient is already at its constrained optimum; its curvature is
    // the flat/indefinite infinite-smoothing plateau (λ ~ 1e13) and carries no
    // feasible descent. Including it would let the step chase that spurious
    // direction and simply re-rail. Restricting to the un-railed block yields a
    // feasible descent that holds every rail fixed, so the escape generalises from
    // the fully-interior saddle to a box-constrained one whose free-direction
    // reduced Hessian is indefinite (#2357 → #2155). With no rail this is exactly
    // the full-Hessian eigenproblem as before.
    let railed_set: std::collections::BTreeSet<usize> = railed.iter().copied().collect();
    let interior: Vec<usize> = (0..n).filter(|k| !railed_set.contains(k)).collect();
    if interior.is_empty() {
        // Every coordinate is railed: there is no feasible interior direction and
        // the rail KKT signs are the whole certificate.
        return SaddleAdjudication::Declined(format!(
            "every one of the {n} outer coordinates is railed, so there is no feasible interior \
             direction to descend"
        ));
    }
    // #2676: the escape must search the SAME subspace the certificate judged.
    // `judged_subspace_basis` returns the interior indicator basis when there is
    // no invariance, so `sub` and the lift below are bit-identical on that path;
    // with one, the escape stops being able to pick the criterion-invariant
    // direction — where the only "negative curvature" available is the
    // chain-rule term `sum_k g_k t_k^2`, i.e. the residual gradient wearing a
    // curvature's clothes — instead of the genuine saddle direction that
    // refused.
    let deflate = invariance.filter(|basis| basis.nrows() == n && basis.ncols() > 0);
    let Some(judged) = crate::penalty_invariance::judged_subspace_basis(n, railed, deflate) else {
        return SaddleAdjudication::Declined(
            "after removing the railed coordinates and the criterion's own invariance there is \
             no direction left to search"
                .to_string(),
        );
    };
    let m = judged.ncols();
    let sub = match deflate {
        Some(_) => crate::penalty_invariance::compress_to_judged_subspace(hessian, &judged),
        None => {
            let mut sub = Array2::<f64>::zeros((m, m));
            for (i, &ri) in interior.iter().enumerate() {
                for (j, &rj) in interior.iter().enumerate() {
                    sub[[i, j]] = hessian[[ri, rj]];
                }
            }
            sub
        }
    };
    let (eigenvalues, eigenvectors) = match sub.eigh(Side::Lower) {
        Ok(pair) => pair,
        Err(err) => {
            return SaddleAdjudication::Declined(format!(
                "the interior sub-block's eigendecomposition failed ({err})"
            ));
        }
    };
    // The SAME √ε·‖H‖ margin `certificate_hessian_is_psd_at_resolution` uses to separate a
    // genuine negative eigenvalue from O(ε·‖H‖) assembly roundoff: only a truly
    // negative direction — not a flat / near-semidefinite one — carries a descent
    // the reseed can exploit. Measured on the interior sub-block's diagonal so the
    // threshold matches the reduced PSD verdict exactly.
    let max_diag = interior
        .iter()
        .fold(0.0_f64, |acc, &j| acc.max(hessian[[j, j]].abs()));
    let neg_margin = f64::EPSILON.sqrt() * max_diag.max(1.0);
    let mut min_idx = 0usize;
    for k in 1..eigenvalues.len() {
        if eigenvalues[k] < eigenvalues[min_idx] {
            min_idx = k;
        }
    }
    if !(eigenvalues[min_idx] < -neg_margin) {
        // The certificate refuses on the FLOOR (`H + diag(|g|)` not PSD); this
        // gate admits on a sqrt(EPSILON)*||H|| ROUNDOFF margin. They are
        // different numbers, so a point can be refused for curvature AND
        // declined for escape, with no record of either bound. Print both so
        // the gap is measurable rather than inferred (#2665).
        return SaddleAdjudication::Declined(format!(
            "the interior sub-block's most negative eigenvalue does not clear the roundoff \
             margin: lambda_min={:.6e}, neg_margin={:.6e} (= sqrt(EPSILON) * max(1, max_k \
             |H_kk|) with max_diag={:.6e}), interior_dim={}",
            eigenvalues[min_idx], neg_margin, max_diag, m,
        ));
    }
    let v_sub = eigenvectors.column(min_idx);
    let dir_norm = v_sub.dot(&v_sub).sqrt();
    if !(dir_norm > 0.0) || !dir_norm.is_finite() {
        return SaddleAdjudication::Declined(format!(
            "the lambda_min={:.6e} eigenvector has an unusable norm {dir_norm:.6e}",
            eigenvalues[min_idx],
        ));
    }
    // Lift the judged eigenvector into the full ρ space through the same basis
    // the sub-block was taken in. Its rows are exactly zero on every railed
    // coordinate, so the backtracking step below still holds all rails fixed;
    // with no invariance the basis is the interior indicator matrix and this is
    // the historical scatter, multiplication by exact zeros and ones.
    let direction = judged.dot(&v_sub.mapv(|value| value / dir_norm));
    // First-order-consistent sign: move against the (tiny) gradient's projection
    // onto `v` so the linear term never opposes the curvature descent. With a
    // stationary gradient the tie is arbitrary; the opposite sign is tried below
    // regardless, which also covers a `v` that projects straight out of the box.
    let primary_sign = if gradient.dot(&direction) > 0.0 {
        -1.0
    } else {
        1.0
    };
    // The step ladder is DERIVED from what the claim predicts, not chosen
    // (#2612).
    //
    // One e-fold in log-λ is a macroscopic step across the saddle ridge, and
    // ARC refines from wherever this lands, so the largest step stays `1`. What
    // the old fixed ladder could not say is where to STOP: it halted at
    // `0.0625` because five entries had been written down, so a claim whose
    // descent only appears below that step was reported the same way as a claim
    // with no descent at all — and the refusal then proceeded on the matrix's
    // word either way.
    //
    // At a stationary point the quadratic model of the claim itself is
    //
    // ```text
    //     V(ρ ± αv) − V(ρ) ≈ ½ λ_min α²,     λ_min < 0
    // ```
    //
    // so the claim predicts a decrease of `½|λ_min|α²`. Once that falls to the
    // criterion's own resolution the claim predicts nothing the criterion can
    // represent, and no smaller step can falsify it. That step,
    //
    // ```text
    //     α_min = sqrt(2 · objective_resolution / |λ_min|),
    // ```
    //
    // is therefore the exact end of the claim's FALSIFIABLE RANGE — derived
    // from the eigenvalue in dispute and the same criterion resolution
    // (`outer_criterion_resolution`) the rail and cost-stall machinery already use, with no
    // constant chosen here. Probing from `1` down to it and finding no descent
    // in either sign is a measurement of the criterion that contradicts the
    // matrix; stopping earlier would only have been a statement about the
    // ladder.
    //
    // When `α_min ≥ α_max` that range is EMPTY (#3036): the largest step's
    // predicted decrease `½|λ_min|·α_max²` is already under the resolution, so
    // no trial can confirm or falsify the claim. The adjudication decides that
    // before any trial. Probing `α_max` anyway made the verdict a function of
    // whether two noise-level evaluations happened to succeed: they "contradict"
    // when they evaluate and "decline" when they fail, and the declined exit
    // refused the point on a curvature its criterion cannot resolve.
    let lambda_min = eigenvalues[min_idx];
    let alpha_max = NEGATIVE_CURVATURE_LADDER_LARGEST_STEP;
    let alpha_min = match negative_curvature_claim(lambda_min, alpha_max, objective_resolution) {
        Some(NegativeCurvatureClaim::Resolvable { alpha_min }) => alpha_min,
        Some(NegativeCurvatureClaim::Unresolvable {
            predicted_at_largest,
        }) => {
            log::debug!(
                "[CERTIFICATE] {context}: the reported negative curvature is UNRESOLVABLE by the \
                 criterion: lambda_min={lambda_min:.6e} on the judged sub-block predicts at most \
                 ½|λ_min|α_max² = {predicted_at_largest:.3e} at the largest step \
                 α_max={alpha_max}, which does not exceed the criterion's resolution \
                 {objective_resolution:.3e}. No trial was evaluated (#3036)."
            );
            return SaddleAdjudication::Unresolvable {
                lambda_min,
                predicted_at_largest,
                objective_resolution,
            };
        }
        // No usable resolution: keep the historical five-rung ladder's reach.
        None => 0.0625,
    };
    let mut escape_step_scales: Vec<f64> = Vec::new();
    let mut alpha = alpha_max;
    loop {
        escape_step_scales.push(alpha);
        // `f64::EPSILON` is where halving stops changing `ρ + αv` at all — a
        // property of the arithmetic, not a budget.
        if alpha <= alpha_min || alpha <= f64::EPSILON {
            break;
        }
        alpha *= 0.5;
    }
    // The strict-decrease floor is the CRITERION's resolution, not the
    // arithmetic's (#2612).
    //
    // The ladder above stops at `α_min = sqrt(2·objective_resolution/|λ_min|)`
    // on the stated ground that below it "the claim predicts nothing the
    // criterion can represent". A trial's MEASURED decrease is the same kind of
    // quantity as the claim's predicted one, so it has to be judged against the
    // same resolution: a step that lowers the objective by less than the
    // criterion can resolve has not descended, it has reproduced the noise the
    // ladder's own stopping rule was derived from. Accepting it as an escape
    // spends the one-shot reseed on a number the criterion cannot distinguish
    // from zero, and the retry — which cannot adjudicate again — then refuses on
    // the matrix's word, which is the state this whole block exists to prevent.
    //
    // Measured before this changed, with the floor at `16ε|V|` (roundoff) while
    // the ladder's limit used `objective_resolution`, i.e. the same function
    // holding two notions of "a decrease the criterion can represent" ten orders
    // apart:
    //
    // ```text
    //   penguins stride-3, unbiased probe: λ_min = −6.35e−7 … −1.99e−6,
    //     four reseeds minted on decreases 2e−6 … 4e−6 of an objective ≈ 2.158,
    //     against objective_resolution = 1.228e−3 — three orders BELOW it;
    //   banded quasi-separated, armed refit: λ_min = −9.19e−3 … −1.12e−2,
    //     three reseeds on decreases 3.4e−4, 1.4e−4, 5.0e−5 of ≈ 53.66,
    //     against a measured cost-stall noise floor of 1.91e−4.
    // ```
    //
    // Both fits then refused for lack of a certified optimum, and the fit that
    // shipped was the Firth/Jeffreys-armed one.
    //
    // Roundoff remains the hard lower limit — where `objective_resolution` is
    // absent or non-positive there is nothing derived to use, and a decrease
    // under `16ε|V|` is not a decrease under any reading.
    let roundoff_floor = baseline_cost.abs().max(1.0) * (16.0 * f64::EPSILON);
    let strict_floor = if objective_resolution.is_finite() && objective_resolution > 0.0 {
        objective_resolution.max(roundoff_floor)
    } else {
        roundoff_floor
    };
    // `(cost, point, sign, alpha)`. The step that produced the point is carried
    // because the ladder ANSWERS a different question from the one the reseed
    // asks (#2612): see [`expand_confirmed_descent`].
    let mut best: Option<(f64, Array1<f64>, f64, f64)> = None;
    // #2665 bookkeeping: "no descending trial", "every trial clamped back onto
    // rho" and "every trial evaluated non-finite" are three different failures
    // that all leave `best == None`. Count them so the declined exit below says
    // which one happened, and carry the best cost actually SEEN so the
    // shortfall against `baseline_cost - strict_floor` is a number. Only a trial
    // the criterion evaluated to a finite cost is `probed`: an evaluation that
    // errored or came back non-finite says nothing about the criterion there, so
    // it cannot falsify the claim.
    let mut probed = 0usize;
    let mut clamped_onto_rho = 0usize;
    let mut eval_failed = 0usize;
    let mut nonfinite = 0usize;
    let mut best_seen_cost = f64::INFINITY;
    for sign in [primary_sign, -primary_sign] {
        for &alpha in escape_step_scales.iter() {
            let mut exact = rho.clone();
            for i in 0..n {
                exact[i] += sign * alpha * direction[i];
            }
            let trial = project_to_bounds(&exact, Some(bounds));
            // A fully box-clamped trial that lands back on ρ probes nothing.
            if outer_theta_bitwise_eq(&trial, rho) {
                clamped_onto_rho += 1;
                continue;
            }
            match obj.eval_cost(&trial) {
                Ok(cost) if cost.is_finite() => {
                    probed += 1;
                    best_seen_cost = best_seen_cost.min(cost);
                    if cost < baseline_cost - strict_floor {
                        best = Some((cost, trial, sign, alpha));
                        // The ladder descends in α and the reseed only has to
                        // LEAVE the ridge — ARC refines from wherever it lands
                        // — so the first (largest) descending step is the
                        // escape. Continuing would spend the rest of the
                        // falsifiability ladder confirming a claim already
                        // confirmed, and that ladder is now derived rather than
                        // five rungs long.
                        break;
                    }
                }
                Ok(_) => nonfinite += 1,
                Err(_) => eval_failed += 1,
            }
        }
        if best.is_some() {
            break;
        }
    }
    // Restore the profiled inner state to the checkpoint ρ so the refusal path
    // that follows measures the checkpoint, not the last probe.
    if let Err(err) = obj.eval_cost(rho) {
        log::debug!(
            "[CERTIFICATE] {context}: failed to restore the objective to the checkpoint \
             after saddle-escape probing: {err}"
        );
    }
    if let Some((cost, _, sign, alpha)) = best {
        let descent = expand_confirmed_descent(
            obj,
            rho,
            &direction,
            LadderConfirmedStep {
                sign,
                alpha,
                cost,
                strict_floor,
            },
            bounds,
            context,
        );
        log::debug!(
            "[CERTIFICATE] {context}: the criterion CONFIRMS the interior strict saddle \
             (λ_min={lambda_min:.3e} < 0, |Pg| within band): a feasible step along its \
             eigenvector lowers the objective {:.6e} → {:.6e}; the caller decides whether that \
             point seeds a retry (#2357)",
            baseline_cost,
            descent.cost,
        );
        return SaddleAdjudication::Descended(descent.point);
    }
    let smallest_step = escape_step_scales
        .last()
        .copied()
        .unwrap_or(f64::INFINITY);
    let predicted_at_smallest = 0.5 * lambda_min.abs() * smallest_step * smallest_step;
    // `probed == 0` is not a contradiction: nothing was evaluated, so nothing
    // was falsified. The three ways that happens are counted separately for
    // exactly this reason (#2665).
    if probed == 0 {
        return SaddleAdjudication::Declined(format!(
            "a certified strict saddle (lambda_min={lambda_min:.6e}, neg_margin={neg_margin:.6e}) \
             produced no EVALUABLE trial at all: clamped_back_onto_rho={clamped_onto_rho}, \
             eval_failed={eval_failed}, non_finite={nonfinite} over {} step(s)",
            escape_step_scales.len(),
        ));
    }
    log::debug!(
        "[CERTIFICATE] {context}: the criterion CONTRADICTS the reported negative curvature. \
         lambda_min={lambda_min:.6e} on the judged sub-block, and {probed} evaluated trial(s) \
         along its eigenvector — both signs, steps {:.3e} down to {smallest_step:.3e} — lowered \
         the objective nowhere. The ladder ends where the claim's own predicted decrease \
         (½|λ_min|α² = {predicted_at_smallest:.3e}) reaches the criterion's resolution \
         ({objective_resolution:.3e}), so that is the WHOLE range in which the claim could have \
         been falsified. best cost seen={best_seen_cost:.9e} against baseline={:.9e} (needed \
         < {:.9e}); clamped_back_onto_rho={clamped_onto_rho}, eval_failed={eval_failed}, \
         non_finite={nonfinite}. The negative direction is a property of this matrix, not of \
         this point (#2612).",
        escape_step_scales.first().copied().unwrap_or(1.0),
        baseline_cost,
        baseline_cost - strict_floor,
    );
    SaddleAdjudication::Contradicted {
        probed,
        smallest_step,
        predicted_at_smallest,
        objective_resolution,
        best_seen_cost,
    }
}

/// The falsifiability ladder's own confirmed step, as handed to the expansion.
///
/// These four travel together — they are one measurement (a signed step along
/// the negative-curvature direction, the objective there, and the floor that
/// decision was strict against) — so they are one argument. Splitting them into
/// four positional `f64`s is what pushed `expand_confirmed_descent` over the
/// argument count and produced an `#[allow(clippy::too_many_arguments)]`, which
/// this repo bans outright: the lint is naming a real thing, and four adjacent
/// same-typed scalars at a call site are a transposition waiting to happen.
#[derive(Clone, Copy, Debug)]
struct LadderConfirmedStep {
    /// Which way along `direction` the ladder confirmed the descent.
    sign: f64,
    /// The step the ladder confirmed it at.
    alpha: f64,
    /// The objective there, in the ladder's instrument state.
    cost: f64,
    /// The decrease the acceptance was strict against: the criterion's
    /// resolution, floored at the objective's own round-off.
    strict_floor: f64,
}

/// The escape point a CONFIRMED negative-curvature descent actually supports
/// (#2612), after the step has been extended past the falsifiability ladder.
#[derive(Clone, Debug)]
struct ConfirmedDescent {
    /// Reseed point, already projected into the box.
    point: Array1<f64>,
    /// Step along `sign · direction` the point sits at.
    alpha: f64,
    /// Objective there, as measured in the expansion's own instrument state.
    cost: f64,
    /// Doublings evaluated. `0` means the ladder's own step stood — either
    /// nothing beyond it improved, or it was already the box intersection.
    expansions: usize,
    /// Whether the accepted step IS the box intersection along the ray, i.e.
    /// the descent ran to the constraint face rather than stopping inside it.
    on_box_face: bool,
}

/// Extend a confirmed negative-curvature descent to the step the criterion
/// actually supports, instead of the step the falsifiability ladder happened to
/// stop at (#2612).
///
/// # The two questions one ladder was answering
///
/// [`adjudicate_negative_curvature`] builds a single step ladder `α = 1, ½, ¼,
/// …` down to `α_min = sqrt(2·objective_resolution/|λ_min|)` and uses it twice.
/// As a falsifier it is exactly right: the smallest step at which the claim
/// `½|λ_min|α²` still predicts something the criterion can represent is the end
/// of the range in which the claim could be refuted, so probing DOWN from one
/// e-fold in log-λ is the whole falsifiable range and finding no descent in it
/// contradicts the matrix.
///
/// As a step rule it is wrong, and wrong in a direction the mathematics names.
/// Along a direction of negative curvature the quadratic model
///
/// ```text
///     V(ρ + αv) − V(ρ) ≈ α(g·v) + ½λ_min α²,    λ_min < 0
/// ```
///
/// decreases WITHOUT BOUND in `α` once the sign is chosen so the linear term is
/// non-positive. A model with no interior minimiser cannot supply a step length;
/// the step has to come from the objective itself and from the feasible box —
/// which is the standard treatment of a negative-curvature direction and is
/// exactly what a trust region does when its solution lands on the boundary.
/// Capping the reseed at the falsifier's largest rung silently asserts the
/// opposite: that one e-fold is as far as any such descent ever runs.
///
/// # What it cost, measured
///
/// On the `#2612` banded quasi-separated fixture the escape direction is `−e₁`
/// to six digits and the criterion falls monotonically along it all the way to
/// the box wall:
///
/// ```text
///   baseline        1.786314898942e1
///   ladder  α=1     1.786314894043e1
///   ladder  α=½     1.786314883184e1   <- the ladder's pick, decrease 1.6e-7
///   α=1             1.786314862766e1
///   α=2             1.786314814710e1
///   α=4             1.786314708132e1
///   α=8             1.786314488769e1   <- box intersection, decrease 4.1e-6
/// ```
///
/// so the wall step is worth **26×** the ladder's, and the BFGS resume seeded at
/// the ladder's point makes no progress at all (reseed and next refused point
/// bit-identical), leaving the escape as the only thing moving ρ — one e-fold
/// per escape, against an interior-escape count of 3 (since deleted, #2817), on a
/// ridge six e-folds long. The fit refused.
///
/// # The rule, and why it needs no constant
///
/// Double the confirmed step while the criterion strictly improves, clamped to
/// the exact box intersection `max_feasible_step_along`, and keep the best point
/// seen. Termination is structural: the box intersection is finite whenever the
/// ray moves any bounded coordinate, doubling reaches it in `⌈log₂(α_box/α)⌉`
/// steps, and any non-improving trial stops the sweep immediately. The accepted
/// point is always the lowest measured, so it is never worse than the ladder's.
///
/// # One evaluation is spent making the comparison honest
///
/// The incumbent's cost came from the falsifiability ladder, which ran before
/// the checkpoint restore, so it was measured in a different profiled-inner
/// state. Measured on the same fixture, the SAME point (`sign = −1, α = 1`)
/// evaluated in the ladder and again afterwards differs by `3.1e-7` — larger
/// than the descent being adjudicated — because the profiled criterion carries
/// warm-start hysteresis. Re-evaluating the incumbent here puts the whole
/// comparison chain in one instrument state.
fn expand_confirmed_descent(
    obj: &mut dyn OuterObjective,
    rho: &Array1<f64>,
    direction: &Array1<f64>,
    seed: LadderConfirmedStep,
    bounds: &(Array1<f64>, Array1<f64>),
    context: &str,
) -> ConfirmedDescent {
    let LadderConfirmedStep {
        sign,
        alpha,
        cost,
        strict_floor,
    } = seed;
    /// Runaway bound on the doubling sweep. Not a modelling choice — the sweep's
    /// END is the box intersection — but a bound on what a pathologically small
    /// confirmed step could ask for. Binding it is logged rather than silently
    /// truncating the range the escape claims to have searched.
    const MAX_EXPANSIONS: usize = 64;

    let n = rho.len();
    let ray = direction.mapv(|value| sign * value);
    let point_at = |alpha: f64| -> Array1<f64> {
        let mut point = rho.clone();
        for i in 0..n {
            point[i] += alpha * ray[i];
        }
        project_to_bounds(&point, Some(bounds))
    };
    let alpha_box = opt::max_feasible_step_along(rho, &ray, &bounds.0, &bounds.1);
    let mut best = ConfirmedDescent {
        point: point_at(alpha),
        alpha,
        cost,
        expansions: 0,
        on_box_face: alpha_box.is_finite() && alpha >= alpha_box,
    };
    if !(alpha.is_finite() && alpha > 0.0) || !strict_floor.is_finite() || strict_floor < 0.0 {
        return best;
    }
    // Nothing to extend into: the confirmed step already reaches (or was clamped
    // at) the box intersection, so the ray has no room left. Returning before the
    // re-measure below keeps this case exactly as cheap as it was.
    if !(alpha < alpha_box) {
        return best;
    }
    // The incumbent, re-measured in THIS instrument state so every comparison
    // below is between values the same profiled inner solve produced.
    if let Ok(reference) = obj.eval_cost(&best.point)
        && reference.is_finite()
    {
        best.cost = reference;
    }
    let mut expansions = 0usize;
    let mut truncated = false;
    let mut current = alpha;
    while current < alpha_box {
        if expansions >= MAX_EXPANSIONS {
            truncated = true;
            break;
        }
        let next = (2.0 * current).min(alpha_box);
        if !(next > current) || !next.is_finite() {
            break;
        }
        expansions += 1;
        let trial = point_at(next);
        // A doubling that lands back on ρ (the whole ray clamped away) probes
        // nothing and cannot be a reseed.
        if outer_theta_bitwise_eq(&trial, rho) {
            break;
        }
        match obj.eval_cost(&trial) {
            Ok(trial_cost) if trial_cost.is_finite() && trial_cost < best.cost - strict_floor => {
                best = ConfirmedDescent {
                    point: trial,
                    alpha: next,
                    cost: trial_cost,
                    expansions,
                    on_box_face: next >= alpha_box,
                };
                current = next;
            }
            _ => break,
        }
    }
    if best.expansions > 0 || truncated {
        log::debug!(
            "[CERTIFICATE] {context}: the confirmed negative-curvature descent was extended past \
             the falsifiability ladder's step alpha={alpha:.6e} to alpha={:.6e} over {} \
             doubling(s) ({} evaluated), objective {:.9e} -> {:.9e}; box intersection along the \
             ray is alpha_box={alpha_box:.6e} and the accepted step {} it (#2612).{}",
            best.alpha,
            best.expansions,
            expansions,
            cost,
            best.cost,
            if best.on_box_face { "IS" } else { "is inside" },
            if truncated {
                format!(" -- TRUNCATED at the {MAX_EXPANSIONS}-doubling budget, so the ray was not searched to the box")
            } else {
                String::new()
            },
        );
    }
    // Same contract as the adjudication's own checkpoint restore: leave the
    // profiled inner state at ρ, not at the last probe.
    if let Err(err) = obj.eval_cost(rho) {
        log::debug!(
            "[CERTIFICATE] {context}: failed to restore the objective to the checkpoint after \
             extending the negative-curvature descent: {err}"
        );
    }
    best
}

/// Which term of the stationarity bound's `max` chain actually set it.
///
/// `certify_outer_optimality` does not compute *a* bound; it takes the maximum of
/// up to five independently-derived quantities, and until #2458 nothing recorded
/// which one won. That is why a `bound=1.000e0` and a `bound=5.68e-6` came out of
/// the same message shape and telling them apart required reading this file --
/// the emitted number is a DERIVATIVE of a construction-site value, computed
/// later, so grepping the construction sites cannot match an observed bound.
///
/// The rungs are not interchangeable calibrations of one quantity. Only
/// [`Self::CurvatureResolvability`] is derived from what a gradient of that size
/// does to the criterion: `|Pg|·√(τ/Δpred)` simplifies to `√(2·h·τ)` -- the `|Pg|`
/// cancels -- i.e. the gradient magnitude below which the criterion cannot resolve
/// descent at all. The others are gradient-magnitude tests, blind to how curvature
/// maps a gradient to an objective change, which is exactly what this file's own
/// comment at the curvature block already says. Recording the rung is the
/// prerequisite for holding every route to the derived one: it makes "which
/// machinery did this route happen to have" an observable rather than an
/// inference.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum StationarityBoundSource {
    /// The caller's declared absolute band `config.tolerance`, used where the
    /// judged route published no per-coordinate gradient parts to derive the
    /// Theorem 9 band from (#2954). A function of the declared problem only: no
    /// row count, no objective offset.
    SolverBand,
    /// The per-coordinate Theorem 9 band (#2954): coordinate `j` certifies iff
    /// `|Pĝ_j| ≤ τ_j − ε_j`, with `τ_j = tolerance·(1 + s_j)` the resolution of
    /// the coordinate's own gradient scale `s_j = ½·rank_j + |½λ_jβ̂ᵀS_jβ̂|` and
    /// `ε_j` the rounding band of the evaluated gradient parts. Both are
    /// O(rank), never O(n), so the band is invariant to row count and to any
    /// additive offset in the criterion. A pass proves `|Pg_j| ≤ τ_j` for the
    /// exact gradient.
    CoordinateBand,
    /// The same test on a coordinate whose rounding band exceeds half its
    /// resolution, `2ε_j > τ_j` (#2954, fp-error-analysis §6.3). No computed
    /// component can prove `|Pg_j| ≤ τ_j` there, so the band is `ε_j`, the
    /// narrowest the arithmetic resolves: a pass proves `|Pg_j| ≤ 2ε_j`, the
    /// attainable resolution, and this label says the requested `τ_j` was out
    /// of the arithmetic's reach. Not the derived standard.
    ArithmeticLimited,
    /// `|Pg|·√(τ/Δpred)` = `√(2·h·τ)` (#2253/#2249/#2015/#2091) -- the only rung
    /// with a derivation from the criterion's own resolution.
    CurvatureResolvability,
    /// Twice the same-ρ spread between the run-recorded and certificate-time
    /// gradients (#2299): the measuring instrument's demonstrated noise.
    GradientReproducibility,
    /// `config.tolerance` judged against the EFS/fixed-point route's
    /// normalized residual `‖(θ⁺−θ)/scale‖_∞` -- not against a gradient norm
    /// at all. The route has no ladder: it exposes no analytic gradient, so
    /// none of the gradient-magnitude rungs above is even computable on it,
    /// and the certificate it mints (`OuterStationarityCertificate::FixedPoint`)
    /// already declares `bound: config.tolerance`. Recording it names the one
    /// thing the shared `bound=` field could not previously say -- that the
    /// number is a residual tolerance and the quantity beside it is a residual.
    ///
    /// Only the ONE refusal on that route which has actually formed the
    /// residual carries this. Its eight early exits formed none, and now say so.
    FixedPointResidual,
    /// The CALLER's `|Pg|` requirement (#2568), capping every rung above.
    ///
    /// The only member of this enum that is not the engine's own judgement, and
    /// the only one that ever TIGHTENS the bound. Reported when the caller's
    /// requirement is stricter than the ladder the engine would have applied, so
    /// a reader can tell "the engine refused this" from "the engine would have
    /// certified this and the caller would not" -- a distinction that matters
    /// because the second is not a defect in the fit.
    CallerRequirement,
    /// `|Pg|·√((tol − band_λ²)/λ̂²)` (#2954): the Newton-decrement verdict,
    /// rendered as a gradient bound along the measured direction. It certifies
    /// iff `λ̂² + band_λ² ≤ tol = max(τ_stat − band_f, band_f)` with `τ_stat =
    /// 1/(2n)` (C3), so no caller tolerance and no scale anchor enters, and it
    /// may TIGHTEN every rung above.
    NewtonDecrement,
    /// The decrement verdict was taken and could not certify anything: its own
    /// rounding reached the objective band, a flat direction carried gradient,
    /// or the reduced Hessian did not decompose (#2954). The bound is `0`.
    NewtonDecrementUndecided,
    /// A mint whose polish stopped short of a box face its step heads to, where
    /// that face is a representability literal rather than the term's derived
    /// limit model (#2954, #2627). Box-KKT certifies nothing about the data
    /// there, so the point is refused by this type instead of railed. The bound
    /// is the decrement verdict's.
    RepresentabilityFace,
    /// A mint whose polish backtracked along its Newton step, down to where the
    /// quadratic model's own decrease reaches `band_f`, and found no step that
    /// lowers the criterion by more than `band_f`, with no limit face lowering it
    /// either (#3012). The decrement promises a decrease the criterion does not
    /// deliver along the Newton direction, so the point is refused by this type.
    /// The bound is the decrement verdict's.
    NewtonBacktrackUnresolved,
}

impl StationarityBoundSource {
    pub(crate) fn label(self) -> &'static str {
        match self {
            Self::SolverBand => "solver-band",
            Self::CoordinateBand => "coordinate-band",
            Self::ArithmeticLimited => "arithmetic-limited",
            Self::CurvatureResolvability => "curvature-resolvability",
            Self::GradientReproducibility => "gradient-reproducibility",
            Self::FixedPointResidual => "fixed-point-residual",
            Self::CallerRequirement => "caller-requirement",
            Self::NewtonDecrement => "newton-decrement",
            Self::NewtonDecrementUndecided => "newton-decrement-undecided",
            Self::RepresentabilityFace => "representability-face",
            Self::NewtonBacktrackUnresolved => "newton-backtrack-unresolved",
        }
    }

    /// Whether this rung is the derived resolvability standard rather than a
    /// gradient-magnitude substitute adopted where that standard was unavailable.
    pub(crate) fn is_derived_standard(self) -> bool {
        // The standard is the decrement test, and #2458's whole point is that
        // WHICH derivative machinery produced the curvature must stop deciding
        // which standard a fit is held to. The answer is that every route
        // supplies exact curvature or none: there is no second, approximate
        // curvature rung, because a bound estimated by the code doing the
        // judging is not the same evidence and pretending otherwise is what
        // made the tiering invisible in the first place.
        matches!(
            self,
            Self::CurvatureResolvability
                | Self::CoordinateBand
                | Self::NewtonDecrement
                | Self::NewtonDecrementUndecided
        )
    }

    /// The neutral projection carried on the refusal, so a red states its own
    /// rung instead of leaving it to be inferred from the numbers.
    pub(crate) fn provenance(self) -> StationarityRung {
        StationarityRung {
            label: self.label(),
            derived_standard: self.is_derived_standard(),
        }
    }
}

/// A stationarity bound bundled with the rung that produced it (#2458).
///
/// The bound and its provenance travelled separately before this: the bound was
/// a bare `f64` argument and the rung stayed a local in the certificate block,
/// so every refusal outside that block reported a number with no way to say
/// which standard it came from. Bundling them makes it impossible to pass one
/// without deciding the other.
///
/// The source is NOT optional, and there is no constructor for "a number with
/// no standard". An earlier revision carried an `unrecorded` escape that 22 of
/// the 30 refusal paths took; naming those 22 was the first repair, and it was
/// half right. Twenty of them do not have an unclassified bound — they have NO
/// bound. They refuse before any stationarity residual exists (a failed
/// terminal evaluation, a malformed gradient, a non-converged inner state), and
/// the configured constant they used to report was never weighed against the
/// point. Those now carry [`StationarityStandard::NoComparison`] and print no
/// bound at all; only a route that formed a residual constructs one of these.
#[derive(Debug, Clone, Copy)]
pub(crate) struct StationarityBound {
    value: f64,
    source: StationarityBoundSource,
}

impl StationarityBound {
    /// A bound the rung ladder produced, carrying the rung it came from.
    pub(crate) fn from_ladder(value: f64, source: StationarityBoundSource) -> Self {
        Self { value, source }
    }

    /// The EFS/fixed-point route's standard: `config.tolerance` against a
    /// normalized fixed-point residual
    /// ([`StationarityBoundSource::FixedPointResidual`]).
    ///
    /// This is the bound `certify_fixed_point_optimality` mints its
    /// `OuterStationarityCertificate::FixedPoint` against, so the refusal that
    /// weighs a formed residual reports the route's own declared standard
    /// rather than a gradient tolerance it never used.
    pub(crate) fn fixed_point_residual(config: &OuterConfig) -> Self {
        Self::from_ladder(
            config.tolerance,
            StationarityBoundSource::FixedPointResidual,
        )
    }

    pub(crate) fn value(self) -> f64 {
        self.value
    }

    /// Whether this value was derived by mapping the caller's residual through
    /// the caller's Hessian. Such a value cannot cross into a reduced face:
    /// the face must mint its own bound from its own Hessian and residual.
    pub(crate) fn requires_face_local_derivation(self) -> bool {
        self.source.is_derived_standard()
    }

    pub(crate) fn rung(self) -> StationarityRung {
        self.source.provenance()
    }
}

impl From<StationarityBound> for StationarityStandard {
    fn from(bound: StationarityBound) -> Self {
        Self::Measured {
            bound: bound.value(),
            rung: bound.rung(),
        }
    }
}

/// The certificate's one-line summary with every coordinate named in the caller's
/// native order (#2817). Inside a canonical run the certificate's railed facts and
/// rails hold canonical slots; this renders a copy renamed by the same owner
/// `outer_result_to_native` uses, so a refusal never prints a canonical slot.
fn native_certificate_summary(certificate: &OuterCriterionCertificate, config: &OuterConfig) -> String {
    match config.native_coordinate_order.as_deref() {
        Some(order) => {
            let mut native = certificate.clone();
            criterion_certificate_to_native(&mut native, order);
            native.summary()
        }
        None => certificate.summary(),
    }
}

pub(super) fn outer_nonconvergence_error(
    context: &str,
    reason: &str,
    result: &OuterResult,
    projected_grad_norm: Option<f64>,
    stationarity_standard: impl Into<StationarityStandard>,
) -> EstimationError {
    // Solver provenance, appended to every outer non-convergence.
    //
    // The certificate answers "is this point stationary" — and when it is not,
    // the next question is always "then why did the search STOP here", which
    // the message did not answer. A binomial/logit P-spline (#1575/#1561)
    // refuses with `|Pg|=7.5e-1` against a `1.0e-2` bound, PSD Hessian, nothing
    // railed, all coordinates 18 e-folds inside the box, "after 7 outer
    // iteration(s)" — and the 7 is the whole mystery. `converged` distinguishes
    // a solver that CLAIMED convergence (a tolerance desync, and the only case
    // the #2273/#2374 resume will retry) from one that ran out of budget;
    // `operator_stop_reason` separates a trust-region reject floor from a
    // flat-valley cost stall from an iteration budget. All three are already on
    // the result and cost nothing to print.
    let reason = format!(
        "{reason}; solver provenance: origin={:?}, plan={}, claimed_converged={}{}{}{}",
        // #2465 at a fifth site. `termination=<no opt solver produced this
        // result>` says an `opt` solver did not decide this, and stops there;
        // `origin` says WHICH lane did. Both are already on the result.
        result.origin,
        result.plan_used,
        result.solver_claimed_convergence(),
        // #2547: `stop_reason` is the coarse projection. `termination` is
        // the test the solver actually applied plus the quantity it was
        // judged against, so a reader gets "the L-infinity window fired at
        // a threshold of 8.9e-1" rather than "it stopped". A stop that
        // made no stationarity claim at all (an iteration budget, a
        // collapsed trust region) says so, instead of leaving that to be
        // inferred from a bare gradient norm nothing compared it to.
        result
            .solver_termination
            .map(|t| {
                let evidence = match t.stationarity_evidence() {
                    Some(e) => format!(
                        " [measured={:.6e} vs threshold={:.6e}, {:?}, {:?}]",
                        e.measured, e.threshold, e.norm, e.scaling
                    ),
                    None => " [made no stationarity claim]".to_string(),
                };
                format!(", termination={t}{evidence}")
            })
            .unwrap_or_else(|| ", termination=<no opt solver produced this result>".to_string()),
        result
            .operator_stop_reason
            .map(|stop| format!(", stop_reason={stop:?}"))
            .unwrap_or_default(),
        result
            .converged_via()
            .map(|via| format!(", converged_via={via:?}"))
            .unwrap_or_default(),
    );
    // #2465: the line search's own verdict, when one failed. `StepSizeTooSmall`
    // and `MaxAttempts` are different defects with different repairs, and
    // "line_search_failed" alone distinguishes neither.
    // The stall window's evidence, when a stall produced this result: σ̂ and Δ,
    // reported and licensing no bound (#2817).
    let reason = match result.cost_stall_probe_scale {
        Some((noise_floor, probe_radius)) => format!(
            "{reason}, cost_stall_window=[noise_floor={noise_floor:.6e}, \
             probe_radius={probe_radius:.6e}]"
        ),
        None => reason,
    };
    // A halt where the search's kept rank ends names the rank, the refused trials that
    // filled the window, and the band the incumbent missed (#2939).
    let reason = match result.rank_boundary_stall {
        Some(stall) => format!(
            "{reason}, rank_boundary=[kept_rank={}, refused_trials={}, band={:.6e}: every \
             trial in the filled window kept a different rank, so the search stopped where its \
             rank ends]",
            stall.kept_rank, stall.refused_trials, stall.band,
        ),
        None => reason,
    };
    let reason = match result.line_search_failure {
        Some((failure_reason, max_attempts)) => format!(
            "{reason}, line_search={failure_reason:?} after {max_attempts} attempt(s) \
             [StepSizeTooSmall = the direction descended but no step improved the \
             objective; MaxAttempts = the bracket never closed]"
        ),
        None => reason,
    };
    EstimationError::RemlDidNotConverge {
        context: context.to_string(),
        reason,
        iterations: result.iterations,
        final_value: result.final_value,
        projected_grad_norm,
        stationarity_standard: stationarity_standard.into(),
        rho_checkpoint: result.rho.to_vec(),
    }
}

/// Roundoff envelope for two independently assembled values of the same outer
/// criterion at the same point.
///
/// Value-only and derivative-bearing evaluation lanes may use different
/// kernels and reduction trees, so bitwise identity is not a valid contract.
/// Their relative disagreement must nevertheless stay below the square root of
/// machine epsilon: beyond that scale the derivative sample is not evidence
/// about the scalar objective the optimizer's value lane ranks.
pub fn outer_value_agreement_bound(value_only: f64, derivative_sample: f64) -> f64 {
    f64::EPSILON.sqrt() * value_only.abs().max(derivative_sample.abs()).max(1.0)
}

/// `lane_inner_convergence` is `(value_lane, derivative_lane)`, each captured
/// IMMEDIATELY after its own evaluation — `inner_solve_converged` reads one
/// shared snapshot, so a flag sampled after both lanes describes only the
/// second (#2228). `None` means the caller did not sample that lane.
fn audit_outer_value_agreement(
    context: &str,
    value_only: f64,
    derivative_sample: f64,
    result: &mut OuterResult,
    projected_grad_norm: Option<f64>,
    stationarity_standard: impl Into<StationarityStandard>,
    lane_inner_convergence: (Option<bool>, Option<bool>),
) -> Result<(), EstimationError> {
    let bound = outer_value_agreement_bound(value_only, derivative_sample);
    let disagreement = (value_only - derivative_sample).abs();
    if disagreement <= bound {
        return Ok(());
    }
    // Which lane's inner solve converged decides whether this is roundoff
    // between two reduction trees (the bound's premise) or a warm-start basin
    // gap the bound was never derived for.
    let lane_note = |flag: Option<bool>| match flag {
        Some(true) => "converged",
        Some(false) => "NOT-CONVERGED",
        None => "unsampled",
    };
    let inner_evidence = format!(
        ", inner solve: value-lane={}, derivative-lane={}",
        lane_note(lane_inner_convergence.0),
        lane_note(lane_inner_convergence.1),
    );

    // The value-only lane is the scalar criterion authority. Preserve it on
    // the resumable checkpoint rather than the derivative lane's inconsistent
    // scalar; no certificate may be attached to this mixed evidence.
    result.final_value = value_only;
    Err(outer_nonconvergence_error(
        context,
        &format!(
            "cost-only value disagrees with analytic-sample value at the same outer point: \
             value-only={value_only:.16e}, analytic-sample={derivative_sample:.16e}, \
             disagreement={disagreement:.3e}, roundoff bound={bound:.3e}{inner_evidence}"
        ),
        result,
        projected_grad_norm,
        stationarity_standard,
    ))
}

fn certify_fixed_point_optimality(
    obj: &mut dyn OuterObjective,
    config: &OuterConfig,
    context: &str,
    result: &mut OuterResult,
    fidelity: CertificationFidelity,
) -> Result<OuterCriterionCertificate, EstimationError> {
    let layout = obj.capability().theta_layout();
    // A fixed-point residual is only a certificate for the scalar objective
    // when both lanes price that same objective at the same rho. Sample the
    // authoritative value lane first; the analytic fixed-point evaluator then
    // remains the installed terminal-state owner.
    //
    // MINT ONLY (#2359), for the same reason as on the analytic path: whether
    // the two lanes agree is terminal proof work, not a ranking signal, and on
    // this route it is the ONLY thing that distinguishes screening from mint —
    // there is no order-four ladder to reserve here.
    let value_only = match fidelity {
        CertificationFidelity::Screening => None,
        CertificationFidelity::Mint => {
            let sample = obj
                .eval_with_order(&result.rho, OuterEvalOrder::Value)
                .map_err(|err| {
                    outer_nonconvergence_error(
                        context,
                        &format!("terminal value-only certificate evaluation failed: {err}"),
                        result,
                        None,
                        StationarityStandard::NoComparison,
                    )
                })?
                .cost;
            if !sample.is_finite() {
                return Err(outer_nonconvergence_error(
                    context,
                    "terminal value-only certificate evaluation returned a non-finite objective value",
                    result,
                    None,
                    StationarityStandard::NoComparison,
                ));
            }
            Some(sample)
        }
    };
    // Sampled HERE, not after the certificate lane below: `inner_solve_converged`
    // returns one shared snapshot describing the most recent inner solve, so a
    // read taken after both lanes cannot speak for this one (#2228). Mirrors the
    // capture-immediately discipline already used for `terminal_inner_converged`.
    let value_lane_inner_converged =
        value_only.map(|_| inner_solve_converged(config.outer_inner_cap.as_ref()));
    // #2228: these two lanes run back-to-back on ONE `&mut obj` with no reset
    // between them. Criteria that fit `(t, β)` in place therefore warm-start
    // this certificate lane from wherever the value-only lane above stopped —
    // the lanes are not two evaluations of one function, the second is a
    // continuation of the first.
    //
    // That is the hazard `OuterObjective::owns_terminal_coefficient_mode`
    // already documents and measures ("disagree by a whole basin (measured:
    // 9.1931e2 vs 9.1671e2 on the cause-specific survival gate)"), but its
    // `obj.reset()` remedy fires once BEFORE `finalize_outer_result`, not
    // between the pair here.
    //
    // It matters because `outer_value_agreement_bound` — applied to exactly
    // these two scalars below — is a sqrt(EPSILON) ROUNDOFF envelope, derived
    // for "different kernels and reduction trees". A warm-start basin gap is
    // not roundoff.
    //
    // MEASURED CORRECTION (2026-07-31). I originally wrote here that the
    // audit's 1.5x..1.1e7x spread "has the shape of a divergence that usually
    // lands in the same basin and occasionally does not". A direct test of the
    // mild end refutes that for at least that end. On
    // `pure_duchon_aniso_fit_optimizes_without_introducing_hybrid_scale`:
    //
    //   value-only=-3.6708914555093685e1  analytic-sample=-3.6708920839323071e1
    //   disagreement=6.284e-6  bound=5.470e-7
    //   inner solve: value-lane=converged, derivative-lane=converged
    //
    // BOTH lanes converged. If the certificate lane warm-starts from the value
    // lane's converged point and itself converges, it should stay there and the
    // scalars should agree — so chaining does not explain this one, and the
    // spread is probably NOT a single mechanism. At |value| ~ 36.7 the bound is
    // sqrt(EPSILON)*36.7 = 5.47e-7 and the gap is ~11.5x it, i.e. the two cost
    // routes differ by ~1.7e-7 RELATIVE: an order above roundoff, many orders
    // below a basin gap.
    //
    // The live question for this end is therefore not state chaining but
    // whether sqrt(EPSILON) can cover two lanes that evaluate at different
    // ORDERS (`Value` vs `ValueAndGradient`/`VGH`) and hence through different
    // kernels — the same "machine constant standing in for a measured quantity"
    // shape as #2614's requested resolution. To test the OTHER end, capture the
    // same flags on a high-end (1.1e7x) firing; the instrumentation is on both
    // audit sites now, so it costs one run.
    //
    // Adding a reset here is NOT a safe drive-by: the same trait doc states
    // that a `false` (default) objective "retains the very state its evaluation
    // at result.rho depends on and must not be reset". Any fix has to reset
    // before BOTH lanes, under the `owns_terminal_coefficient_mode` guard, so
    // the pair shares one baseline instead of chaining.
    let evaluation = obj
        .eval_fixed_point_certificate(&result.rho)
        .map_err(|err| {
            outer_nonconvergence_error(
                context,
                &format!("analytic fixed-point certificate evaluation failed: {err}"),
                result,
                None,
                StationarityStandard::NoComparison,
            )
        })?;
    let certificate_lane_inner_converged = inner_solve_converged(config.outer_inner_cap.as_ref());
    if !certificate_lane_inner_converged {
        return Err(outer_nonconvergence_error(
            context,
            "terminal fixed-point evidence was evaluated at a non-converged inner state",
            result,
            None,
            StationarityStandard::NoComparison,
        ));
    }
    if evaluation.coordinates.len() != layout.n_params {
        return Err(outer_nonconvergence_error(
            context,
            &format!(
                "fixed-point certificate returned {} coordinates for an outer problem of dimension {}",
                evaluation.coordinates.len(),
                layout.n_params
            ),
            result,
            None,
            StationarityStandard::NoComparison,
        ));
    }
    if !evaluation.cost.is_finite() {
        return Err(outer_nonconvergence_error(
            context,
            "fixed-point certificate returned a non-finite objective value",
            result,
            None,
            StationarityStandard::NoComparison,
        ));
    }
    if let Some(value_only) = value_only {
        audit_outer_value_agreement(
            context,
            value_only,
            evaluation.cost,
            result,
            None,
            StationarityStandard::NoComparison,
            (
                value_lane_inner_converged,
                Some(certificate_lane_inner_converged),
            ),
        )?;
    }

    let mut normalized_updates = Vec::with_capacity(layout.n_params);
    let mut uncovered = Vec::new();
    for (index, coordinate) in evaluation.coordinates.iter().enumerate() {
        match coordinate {
            FixedPointCoordinateCertificate::Covered { update, scale }
                if update.is_finite() && scale.is_finite() && *scale > 0.0 =>
            {
                normalized_updates.push(*update / *scale);
            }
            FixedPointCoordinateCertificate::Covered { update, scale } => {
                uncovered.push(format!(
                    "coordinate {} has invalid covered residual update={update} scale={scale}",
                    native_coordinate(config.native_coordinate_order.as_deref(), index)
                ));
                normalized_updates.push(f64::NAN);
            }
            FixedPointCoordinateCertificate::Uncovered { reason } => {
                uncovered.push(format!(
                    "coordinate {}: {reason}",
                    native_coordinate(config.native_coordinate_order.as_deref(), index)
                ));
                normalized_updates.push(f64::NAN);
            }
        }
    }
    if !uncovered.is_empty() {
        return Err(outer_nonconvergence_error(
            context,
            &format!(
                "fixed-point certificate lacks root-equivalent analytic coverage: {}",
                uncovered.join("; ")
            ),
            result,
            None,
            StationarityStandard::NoComparison,
        ));
    }

    let (lower, upper) = outer_model_domain_bounds_template(config, layout.n_params);
    let mut raw_inf = 0.0_f64;
    let mut projected_inf = 0.0_f64;
    for index in 0..layout.n_params {
        let update = normalized_updates[index];
        raw_inf = raw_inf.max(update.abs());
        // `update` is a signed descent/update direction, the negative of the
        // gradient convention used by `projected_gradient_norm`: at a lower
        // bound a negative update points out of the box, and at an upper bound
        // a positive update does. Only those infeasible multiplier components
        // are removed.
        let projected = if result.rho[index] <= lower[index] {
            update.max(0.0)
        } else {
            update
        };
        let projected = if result.rho[index] >= upper[index] {
            projected.min(0.0)
        } else {
            projected
        };
        projected_inf = projected_inf.max(projected.abs());
    }

    result.final_value = evaluation.cost;
    result.final_grad_norm = None;
    result.final_measurement = None;
    result.displaced_measurement = None;
    result.final_hessian = None;

    let certificate = OuterCriterionCertificate {
        stationarity: OuterStationarityCertificate::FixedPoint {
            residual_inf_norm: raw_inf,
            projected_residual_inf_norm: projected_inf,
            bound: config.tolerance,
            rung: StationarityBoundSource::FixedPointResidual.provenance().into(),
            covered_coordinates: layout.n_params,
        },
        // The EFS/fixed-point route exposes no analytic Hessian, so there was
        // a curvature question and nothing could answer it (#2561).
        curvature: CurvatureEvidence::NotAvailable,
        lambdas_railed: certificate_railed_lambdas(&result.rho, layout.rho_dim(), config),
        railed_facts: railed_coordinate_facts(
            &result.rho,
            // #2624: the EVIDENCE field, unlike `lambdas_railed` above, is not
            // lambda-scoped -- it states the interval and margin each
            // coordinate was judged against, and the judgement is taken on the
            // theta-wide face. A psi/log-kappa coordinate pinned on its own
            // window was structurally absent from it.
            &certificate_railed_coordinates(&result.rho, config),
            config,
        ),
        newton_polish: None,
        curvature_floor: None,
    };
    result.criterion_certificate = Some(certificate.clone());
    if !certificate.certifies() {
        return Err(outer_nonconvergence_error(
            context,
            &native_certificate_summary(&certificate, config),
            result,
            Some(projected_inf),
            StationarityBound::fixed_point_residual(config),
        ));
    }

    let via = match result.termination.proposed_via() {
        Some(via @ OuterConvergedVia::RecurrentIncumbent { .. }) => via,
        _ => OuterConvergedVia::FixedPointStationary {
            projected_residual_inf_norm: projected_inf,
            certificate_bound: config.tolerance,
        },
    };
    result.termination.certify(via);
    log::debug!("[CERTIFICATE] {context}: {}", native_certificate_summary(&certificate, config));
    Ok(certificate)
}

/// Build the mandatory analytic optimality certificate at the returned point.
///
/// The objective is evaluated at the selected point through both its
/// authoritative value-only lane and its analytic derivative lane. The two
/// values must agree within their roundoff envelope before the derivative
/// sample can certify the scalar objective. Missing, malformed, non-finite, or
/// split-objective evidence is non-convergence: an optimizer status bit cannot
/// substitute for a stationarity certificate. Exact analytic curvature is
/// checked when the objective declares it and can materialize it; BFGS/EFS
/// solver geometry is never mistaken for objective curvature.
/// Which derivative fidelity a certification pass is allowed to spend.
///
/// #2359: the generic REML/LAML Hessian consumes the row-family derivative
/// ladder through order FOUR while its analytic gradient stops at order three,
/// so order four is a mint-time cost, not a per-candidate one. The outer search
/// screens every candidate stop it reaches — that screening is a first-order gate
/// (stationarity, KKT projection, rail facts), and `curvature_admissible()`
/// reads `hessian_psd != Some(false)`, so a `None` curvature verdict certifies
/// on stationarity alone. The one order-four evaluation belongs to the winner,
/// once, and its verdict is the one that mints.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum CertificationFidelity {
    /// Per-candidate stop gate. Never spends order four.
    Screening,
    /// The single terminal mint audit. Spends order four when the objective
    /// declares an analytic Hessian.
    Mint,
}

pub(crate) fn certify_outer_optimality(
    obj: &mut dyn OuterObjective,
    config: &OuterConfig,
    context: &str,
    result: &mut OuterResult,
) -> Result<OuterCriterionCertificate, EstimationError> {
    certify_outer_optimality_with_fidelity(obj, config, context, result, CertificationFidelity::Mint)
}

pub(crate) fn certify_outer_optimality_with_fidelity(
    obj: &mut dyn OuterObjective,
    config: &OuterConfig,
    context: &str,
    result: &mut OuterResult,
    fidelity: CertificationFidelity,
) -> Result<OuterCriterionCertificate, EstimationError> {
    result.termination.begin_certification();
    let terminal_cap_guard = config
        .outer_inner_cap
        .as_ref()
        .map(FullFidelityInnerCapGuard::lift);
    if terminal_cap_guard.is_some() || obj.owns_terminal_coefficient_mode() {
        // `reset` is deliberately conditional on the presence of the cap
        // contract.  Those are the REML/mixture objectives whose search cache
        // can contain a coarse inner state; uncapped objectives retain their
        // ordinary stateful certification semantics.
        //
        // The `owns_terminal_coefficient_mode()` disjunct (#2334) closes the
        // gap for cap-less objectives that install an owned coefficient mode:
        // the certifying re-eval below must start from the same clean baseline
        // that `finalize_outer_result` used, so the mode's objective bitwise
        // matches the certified `final_value` even when the inner solve is
        // bimodal at `rho_star`.
        obj.reset();
    }
    let outcome = certify_outer_optimality_at_terminal_fidelity(
        obj, config, context, result, true, fidelity, None,
    );
    drop(terminal_cap_guard);
    if outcome.is_err() {
        result.termination.refuse_certificate();
    }
    outcome
}

pub(super) fn certify_outer_optimality_at_terminal_fidelity(
    obj: &mut dyn OuterObjective,
    config: &OuterConfig,
    context: &str,
    result: &mut OuterResult,
    allow_certify_reseed: bool,
    fidelity: CertificationFidelity,
    // The Newton polish the mint has taken so far, `None` on the first call
    // (#2954).
    polish: Option<super::newton_polish::PolishWalk>,
) -> Result<OuterCriterionCertificate, EstimationError> {
    let capability = obj.capability();
    let layout = capability.theta_layout();
    layout
        .validate_point_len(&result.rho, "outer certificate point")
        .map_err(|err| {
            EstimationError::RemlOptimizationFailed(format!(
                "{context}: invalid outer certificate point: {err}"
            ))
        })?;
    // Certification is an ownership boundary: reinstall the model-domain
    // derivative face before either screening or mint evaluates the selected
    // point. This makes the terminal audit independent of whichever
    // canonicalized plan or prior fit last touched this thread's IFT state, and
    // prevents an active-set search override from surviving as derivative
    // geometry. Screening and mint can differ in derivative order, never in the
    // model whose feasible directions they differentiate (#2514).
    let model_domain_bounds_for_derivatives =
        outer_model_domain_bounds_template(config, layout.n_params);
    crate::estimate::reml::outer_eval::record_current_outer_rho_model_upper_bounds_for_ift(
        &model_domain_bounds_for_derivatives.1,
    );
    if result.rho.iter().any(|value| !value.is_finite()) {
        return Err(outer_nonconvergence_error(
            context,
            "the selected checkpoint contains non-finite coordinates",
            result,
            None,
            StationarityStandard::NoComparison,
        ));
    }
    if layout.n_params == 0 {
        let value = obj.eval_cost(&result.rho).map_err(|err| {
            outer_nonconvergence_error(
                context,
                &format!("zero-dimensional final objective evaluation failed: {err}"),
                result,
                Some(0.0),
                StationarityStandard::NoComparison,
            )
        })?;
        if !value.is_finite() {
            return Err(outer_nonconvergence_error(
                context,
                "the zero-dimensional final objective is non-finite",
                result,
                Some(0.0),
                StationarityStandard::NoComparison,
            ));
        }
        let certificate = OuterCriterionCertificate {
            stationarity: OuterStationarityCertificate::AnalyticGradient {
                grad_norm: 0.0,
                projected_grad_norm: 0.0,
                bound: outer_gradient_tolerance(config).abs,
                // No smoothing estimand: the empty score is stationary by
                // construction, not by clearing this band (#2530).
                rung: StationarityRung::EMPTY_ESTIMAND.into(),
            },
            // No estimand, so no curvature exists to be admissible — the
            // second-order twin of the EMPTY_ESTIMAND rung above (#2561).
            curvature: CurvatureEvidence::NoEstimand,
            lambdas_railed: Vec::new(),
            railed_facts: Vec::new(),
            newton_polish: None,
            curvature_floor: None,
        };
        result.final_value = value;
        result.final_grad_norm = Some(0.0);
        result.record_measurement_at_rho(value, Array1::zeros(0));
        result.final_hessian = None;
        result
            .termination
            .certify(OuterConvergedVia::GradientStationary);
        result.criterion_certificate = Some(certificate.clone());
        return Ok(certificate);
    }
    if matches!(result.plan_used.solver, Solver::Efs | Solver::HybridEfs)
        && capability.gradient != Derivative::Analytic
    {
        return certify_fixed_point_optimality(obj, config, context, result, fidelity);
    }
    if capability.gradient != Derivative::Analytic {
        return Err(outer_nonconvergence_error(
            context,
            "the objective exposes no analytic gradient for final certification",
            result,
            None,
            StationarityStandard::NoComparison,
        ));
    }

    // Sample the scalar authority FIRST and leave the derivative-bearing
    // evaluator as the terminal state owner. This is one same-rho audit, not a
    // finite-difference derivative: it catches a split value/gradient
    // implementation without violating the production no-FD contract.
    //
    // The order is load-bearing in two independent ways, and #2583 briefly
    // inverted it:
    //
    //   * Every real outer objective's derivative lane is a warm-started inner
    //     SOLVE, so evaluating it ADVANCES the inner state. A value sample
    //     taken afterwards reads the state that solve left behind, not the
    //     state it priced. The value lane, by contrast, reads the installed
    //     state without advancing it, so value-then-derivative is the only
    //     order in which both samples price ONE inner state.
    //   * The mint's derivative-bearing evaluation must be the LAST evaluation
    //     of the certification, because it is the terminal coefficient-mode
    //     owner every downstream consumer reads (#2359). A trailing value-only
    //     request re-installs a value-only pass as that owner.
    //
    // Inverting it also made the audit VACUOUS on the one route it was aimed
    // at: the REML objective's outer-eval cache serves any `Value` request from
    // whatever entry the derivative call just wrote, so the audit compared a
    // number with itself and could not fire. Agreement by cache lookup is not
    // agreement between two assemblies of the criterion.
    //
    // The genuine #2583 hazard — two INNER SOLVES at one ρ, disagreeing at
    // inner-tolerance scale rather than at the √ε scale this bound is
    // calibrated for — is addressed where it lives: the value lane's bundle is
    // stored under the shared ρ key, so the derivative lane that follows reuses
    // that inner solution instead of re-solving from it.
    //
    // BOTH fidelities pay this, deliberately. It is tempting to reserve it for
    // the mint the way order four is reserved (#2359), but the two are not
    // alike: order four is evidence the filter does not consume, while the
    // value-agreement audit is a REFUSAL GATE on the very number the
    // multistart ranks by. `retain_best_outer_checkpoint` orders candidates on
    // `final_value`; admitting a candidate whose value lane disagrees with its
    // derivative lane lets a desynced candidate win the ranking on a number
    // nothing has validated, and mint then refuses the whole fit instead of a
    // runner-up carrying it. Pinned by
    // `analytic_hessian_candidate_screening_requires_only_first_order_evidence_2414`,
    // which asserts screening's orders are exactly [Value, ValueAndGradient].
    //
    // The EFS/fixed-point route is different and IS gated on fidelity — see
    // `certify_fixed_point_optimality`. It returns above this block, and there
    // screening and mint are otherwise identical work (no order-four ladder to
    // reserve), so running the audit twice buys nothing at all.
    let value_only = obj
        .eval_with_order(&result.rho, OuterEvalOrder::Value)
        .map_err(|err| {
            outer_nonconvergence_error(
                context,
                &format!("terminal value-only certificate evaluation failed: {err}"),
                result,
                result.final_grad_norm,
                StationarityStandard::NoComparison,
            )
        })?
        .cost;
    // Sampled HERE, while the shared inner-progress snapshot still describes
    // THIS lane. The guard further down reads it after the analytic lane has
    // also run, so it can only speak for that one (#2228). Measured: this is the
    // audit site that actually fires in practice, and it was reporting
    // "value-lane=unsampled" because only the fixed-point route was wired.
    let value_lane_inner_converged = inner_solve_converged(config.outer_inner_cap.as_ref());
    if !value_only.is_finite() {
        return Err(outer_nonconvergence_error(
            context,
            "terminal value-only certificate evaluation returned a non-finite objective value",
            result,
            result.final_grad_norm,
            StationarityStandard::NoComparison,
        ));
    }

    // Order four is reserved for the mint audit (#2359). A screening pass takes
    // the same first-order evidence the no-analytic-Hessian path already
    // certifies on, so the multi-start keeps its filter without building the
    // order-four family tower once per seed.
    // One boolean drives BOTH the request below and the requirement at the
    // analytic-Hessian block: a pass that does not ask for curvature must not
    // then refuse the candidate for not supplying it. Splitting them made every
    // screened candidate of an analytic-Hessian objective fail certification
    // with "declared analytic curvature but returned none at the final point" —
    // a statement about this pass's own eval order, not about the candidate.
    //
    // #2596 addendum, and the reason this line is no longer the whole story:
    // the reservation is sound for the CURVATURE VERDICT (a missing verdict is
    // permissive) but NOT for the stationarity BOUND, one of whose rungs the
    // same Hessian owns and where a missing rung is strictly tightening. The
    // screening pass therefore re-acquires the Hessian, for the bound alone,
    // when the un-widened band would refuse — see `screening_bound_curvature`
    // below. That escalation is the reason this boolean can stay
    // fidelity-gated: it still governs the ORDINARY path, and order four still
    // costs nothing on any candidate that clears its first-order band.
    let wants_analytic_hessian =
        capability.hessian.is_analytic() && matches!(fidelity, CertificationFidelity::Mint);
    let order = if wants_analytic_hessian {
        OuterEvalOrder::ValueGradientHessian
    } else {
        OuterEvalOrder::ValueAndGradient
    };
    // #2954: the decrement verdict charges each gradient component on the
    // magnitudes of the channels it was summed from, which only this
    // evaluation's own parts carry. The capture is taken before the error is
    // surfaced, so a failed evaluation leaves it disarmed.
    crate::estimate::outer_eval_capture::begin_certificate_parts_capture();
    let evaluation = obj.eval_with_order(&result.rho, order);
    let terminal_evidence = super::newton_polish::terminal_certificate_evidence(
        &result.rho,
        polish.as_ref(),
        crate::estimate::outer_eval_capture::take_certificate_evidence(),
    );
    let evaluation = evaluation.map_err(|err| {
        outer_nonconvergence_error(
            context,
            &format!("analytic final-point evaluation failed: {err}"),
            result,
            result.final_grad_norm,
            StationarityStandard::NoComparison,
        )
    })?;

    let analytic_lane_inner_converged = inner_solve_converged(config.outer_inner_cap.as_ref());
    if !analytic_lane_inner_converged {
        return Err(outer_nonconvergence_error(
            context,
            "terminal analytic evidence was evaluated at a non-converged inner state",
            result,
            None,
            StationarityStandard::NoComparison,
        ));
    }
    layout
        .validate_gradient_len(&evaluation.gradient, "outer certificate gradient")
        .map_err(|err| {
            outer_nonconvergence_error(
                context,
                &format!("malformed analytic final gradient: {err}"),
                result,
                None,
                StationarityStandard::NoComparison,
            )
        })?;
    if !evaluation.cost.is_finite() || evaluation.gradient.iter().any(|value| !value.is_finite()) {
        return Err(outer_nonconvergence_error(
            context,
            "the analytic final-point value or gradient is non-finite",
            result,
            None,
            StationarityStandard::NoComparison,
        ));
    }

    let bounds = outer_model_domain_bounds_template(config, layout.n_params);
    // A penalty creeping toward its domain's infinite-smoothing face never reaches
    // it EXACTLY — each outer step only shrinks the gap, so it lands strictly inside the
    // box (the #2299 checkpoint sits at ρ=29.9938, not 30). `certificate_railed_lambdas`
    // then flags it railed via `CERTIFICATE_RAIL_MARGIN`, but the exact `x >= upper` /
    // `x <= lower` box-KKT projection treats it as INTERIOR and its outward pull inflates
    // |Pg| above the (tiny) stationarity bound — the fit refuses a genuine railed optimum.
    // Project the stationarity residual with the box endpoints relaxed inward by that SAME
    // rail margin, so "railed" means ONE thing to the detector AND the projector: a
    // within-tolerance coordinate whose gradient points OUT of the box has its KKT-multiplier
    // component removed rather than counted as a stationarity residual (#2299). The
    // projection only zeros the OUTWARD half (`.max(0.0)`/`.min(0.0)`), so a coordinate near
    // the bound that still has feasible-descent gradient keeps it and is never falsely
    // certified.
    let rail_projection_bounds = rail_relaxed_bounds(&bounds);
    let grad_norm = evaluation.gradient.dot(&evaluation.gradient).sqrt();
    // The terminal inner coefficients β(ρ̂), published by the REML bridge on
    // every eval (`inner_beta_hint`). Used to scale the estimand tolerance for
    // the asymptote-rail certificate (#2348 Inc 1).
    let terminal_beta = evaluation.inner_beta_hint.clone();
    // KKT-projected gradient VECTOR (not just its norm): the norm feeds the
    // stationarity certificate below, and the vector feeds the curvature-scaled
    // flat-valley Newton decrement (#2253/#2249/#2015) once the analytic Hessian
    // is in hand.
    let projected_gradient = project_gradient_vector(
        &result.rho,
        &evaluation.gradient,
        Some(&rail_projection_bounds),
    );
    let projected_grad_norm = projected_gradient.iter().map(|v| v * v).sum::<f64>().sqrt();
    // #2514: preserve the literal first-order geometry used by this pass without
    // dumping an O(p) vector for large outer problems. A temporary active-set
    // search box can differ from the model's feasible box; when that happens,
    // screening and mint may otherwise print identical rho/raw gradients while
    // silently projecting against different faces. Record every detector-active
    // or actually projected coordinate, capped only in the log payload.
    if log::log_enabled!(log::Level::Debug) {
        const PROJECTION_RECORD_LIMIT: usize = 32;
        let mut active_or_projected = 0usize;
        let mut projection_records = Vec::new();
        for k in 0..result.rho.len() {
            let detector_railed = outer_coordinate_is_railed(&result.rho, k, config);
            let raw = evaluation.gradient[k];
            let projected = projected_gradient[k];
            if detector_railed || raw.to_bits() != projected.to_bits() {
                active_or_projected += 1;
                if projection_records.len() < PROJECTION_RECORD_LIMIT {
                    projection_records.push((
                        k,
                        result.rho[k],
                        bounds.0[k],
                        bounds.1[k],
                        rail_projection_bounds.0[k],
                        rail_projection_bounds.1[k],
                        raw,
                        projected,
                        detector_railed,
                    ));
                }
            }
        }
        if active_or_projected > 0 {
            log::debug!(
                "[CERTIFICATE-PROJECTION] {context}: fidelity={fidelity:?},                  active_or_projected={active_or_projected},                  records(index,rho,model_lo,model_hi,projection_lo,projection_hi,raw_g,projected_g,detector_railed)={projection_records:?}{}",
                if active_or_projected > projection_records.len() {
                    " (truncated)"
                } else {
                    ""
                },
            );
        }
    }
    // #2954: the band is the per-coordinate Theorem 9 band of THIS evaluation's
    // gradient parts, `|Pĝ_j| ≤ max(ε_j, τ_j − ε_j)`, charged at the coordinate's own
    // O(rank) scale. The criterion value at the point used to anchor a widening
    // `τ·(1 + |V|)` here (#2613), which grew with the row count and with any
    // additive offset in V, neither of which the gradient's resolution depends
    // on.
    //
    // #2688: the band and its rung arrive TOGETHER. This used to be followed by
    // `let mut bound_source = SolverBand;`, so the engine's declared band, the
    // point-anchored widening and the caller's cap -- three quantities, one of
    // which is not a defect in the fit -- all reported one label.
    let band_at_point = outer_certificate_band_at(config, &projected_gradient, &terminal_evidence);
    let solver_bound = band_at_point.bound;
    let mut bound_source = band_at_point.source;
    if bound_source == StationarityBoundSource::CallerRequirement {
        // #2568's audit line, which until #2688 could not print on this path:
        // the cap it announces had already been applied silently in the band
        // helper, so the guard below that emits it was false by construction.
        log::debug!(
            "[2568-REQUIREMENT] {context}: caller requires |Pg| <= {:.6e}; \
             engine bound was {:.6e} (rung {}); measured |Pg| = \
             {projected_grad_norm:.6e}",
            solver_bound,
            band_at_point.engine_bound,
            band_at_point.engine_source.label(),
        );
    }
    // #2458: this used to open with a rung gated on
    // `operator_stop_reason == CostStallFlatValley` that installed the guard's
    // score-relative term — `1e-3·(1 + |score|)` capped at `1.0`, deleted with the
    // guard's own copy by #2817. Two things were wrong with it, and they compound.
    //
    // First, the gate is an EXIT REASON and the bound is a pure function of the
    // criterion value. The same point, with the same criterion, the same
    // gradient and the same curvature, was judged against two different
    // standards depending on which stop reason the operator happened to record.
    // That is this issue's thesis in its purest form: a predicate answering
    // "how did the loop stop?" consumed where a property of the objective is
    // needed.
    //
    // Second, and worse, the constant overruled the measurement taken in the same
    // regime exactly where that measurement had declined: the certificate
    // substituted a number that was never measured.
    //
    // #2817 then deleted the measurement's own rung. A cost-stall exit used to
    // carry the guard's `σ̂/Δ` here as a widening (#2241), but that ratio bounds
    // the gradient only along the directions the window's steps spanned, and
    // the directional version needs a Hessian no route publishing a guard claim
    // holds. Census on #2817 and #2241: every fit the rung certified certifies
    // through curvature resolvability instead.
    //
    // What remains in the ladder below is, in every rung: the configured band,
    // a measurement (gradient reproducibility), a derivation (`|Pg|·√(τ/Δpred)`),
    // or the caller's own requirement. No rung is a magic relative constant, and
    // none is selected by how the search exited.
    let mut stationarity_bound = solver_bound;
    // #2568/#2688 -- the caller's requirement is applied once, inside
    // `outer_certificate_band_at`, which labels the band it capped
    // (audited just above). A second cap used to sit here for a bound that a
    // widening between the two pushed back past the requirement; the only such
    // widening was the probe-noise rung, and with it deleted (#2817) nothing
    // above this point can widen the already-capped band, so that cap could
    // never fire and was removed.
    audit_outer_value_agreement(
        context,
        value_only,
        evaluation.cost,
        result,
        Some(projected_grad_norm),
        StationarityBound::from_ladder(stationarity_bound, bound_source),
        (
            Some(value_lane_inner_converged),
            Some(analytic_lane_inner_converged),
        ),
    )?;

    // The optimizer's own recorded best-iterate evidence, captured before the
    // fresh certificate-time measurement overwrites it below. When it was taken
    // at this ρ, it and `evaluation` are TWO independent measurements of the
    // objective at one point — the raw material for the gradient-reproducibility
    // floor further down, at zero additional objective evaluations.
    let run_recorded = result.final_measurement.take();
    // A previous certificate pass at this ρ (screening, before this mint)
    // replaced the solver's measurement with its own reset re-measurement, which
    // this pass's evaluation replays bit for bit. The measurement it displaced is
    // the independent one.
    let displaced = result
        .displaced_measurement
        .take()
        .filter(|measurement| measurement.is_at(&result.rho));

    // Install measured first-order evidence before any fallible curvature
    // processing. If curvature is malformed, the retained resume checkpoint
    // still carries the exact value/gradient that caused certification to stop.
    result.final_value = evaluation.cost;
    result.final_grad_norm = Some(projected_grad_norm);
    result.record_measurement_at_rho(evaluation.cost, evaluation.gradient);
    result.displaced_measurement = displaced.clone().or_else(|| {
        run_recorded
            .clone()
            .filter(|measurement| measurement.is_at(&result.rho))
    });

    // #2596 — a pass that spends LESS evidence must not produce a STRONGER
    // refusal than the pass that mints.
    //
    // The stationarity bound is a LADDER, and one of its rungs — the
    // curvature-resolvability widening below — is owned by the analytic outer
    // Hessian. `Screening` deliberately does not spend the order-four ladder
    // (#2359), and that reservation was argued on the CURVATURE CONJUNCT:
    // `curvature_admissible()` reads `hessian_psd != Some(false)`, so a `None`
    // curvature verdict certifies on stationarity alone. True — but the Hessian
    // ALSO owns a rung of the stationarity BOUND, and there `None` is not
    // permissive: it silently reverts screening to the un-widened solver band.
    // Screening therefore applied a strictly tighter standard than the mint, and
    // a candidate it refused was discarded rather than deferred.
    //
    // Measured (#2596, lognormal location-scale AFT with a double-penalty
    // `s(z, bs="tps", k=10)`): the BFGS converged to the correct interior optimum
    // ρ = (0.378, −4.975) at cost 4.1926 with |Pg| = 7.29e-5 against a solver
    // band of 5.19e-5 — refused by a factor of 1.4. Both interior seeds were
    // refused, the multi-start fell through to the seed lattice's
    // over-smoothing boundary candidate, and THAT certified vacuously (at a
    // railed corner the box-KKT projection makes |Pg| identically zero) and was
    // minted at cost 110.94. The published smoothing parameter's own LAML was
    // 26× worse than the one the optimizer had already measured, and the fitted
    // smooth carried none of its signal. Every sibling arm of the same suite
    // reached the mint and got the rung that would have saved this one
    // (`curvature-scaled flat-valley bound 1.537e-3 … widened from
    // gradient-band 2.359e-4`); which side of the band a fit lands on is which
    // side the last BFGS step stopped on, not a statistical distinction.
    //
    // So spend the ladder at screening too — but ONLY when the un-widened bound
    // would refuse, and ONLY for the bound. The escalated curvature never
    // reaches the curvature verdict or the rail certificate, so
    // this can only ever turn a screening refusal into a screening
    // certification and never the reverse. A fit that clears its first-order
    // band is byte-identical and pays nothing, so #2359's "order four exactly
    // once, at the mint" continues to hold for every healthy fit.
    let screening_bound_curvature = if !wants_analytic_hessian
        && capability.hessian.is_analytic()
        && projected_grad_norm > stationarity_bound
    {
        log::debug!(
            "[CERTIFICATE] {context}: screening's first-order band would refuse \
             (|Pg|={projected_grad_norm:.3e} > bound={stationarity_bound:.3e}, rung={}); \
             spending the order-four ladder so this refusal is judged by the SAME \
             stationarity bound the mint applies (#2596)",
            bound_source.label(),
        );
        // A failed or malformed escalation is NOT a refusal of the candidate: it
        // only means this rung is unavailable, which is exactly the state the
        // pass was already in. Fall through to the un-widened comparison.
        match obj.eval_with_order(&result.rho, OuterEvalOrder::ValueGradientHessian) {
            Ok(escalated) => match escalated.hessian.materialize_dense() {
                Ok(Some(hessian))
                    if layout
                        .validate_hessian_shape(&hessian, "outer certificate Hessian")
                        .is_ok()
                        && hessian.iter().all(|value| value.is_finite()) =>
                {
                    Some(hessian)
                }
                _ => {
                    log::debug!(
                        "[CERTIFICATE] {context}: the escalated order-four evaluation \
                         supplied no usable curvature; keeping the first-order bound"
                    );
                    None
                }
            },
            Err(error) => {
                log::debug!(
                    "[CERTIFICATE] {context}: the escalated order-four evaluation failed \
                     ({error}); keeping the first-order bound"
                );
                None
            }
        }
    } else {
        None
    };

    let analytic_hessian = if wants_analytic_hessian {
        match evaluation.hessian.materialize_dense() {
            Ok(Some(hessian)) => {
                layout
                    .validate_hessian_shape(&hessian, "outer certificate Hessian")
                    .map_err(|err| {
                        outer_nonconvergence_error(
                            context,
                            &format!("malformed analytic final Hessian: {err}"),
                            result,
                            Some(projected_grad_norm),
                            StationarityBound::from_ladder(stationarity_bound, bound_source),
                        )
                    })?;
                if hessian.iter().any(|value| !value.is_finite()) {
                    return Err(outer_nonconvergence_error(
                        context,
                        "the analytic final Hessian contains non-finite entries",
                        result,
                        Some(projected_grad_norm),
                        StationarityBound::from_ladder(stationarity_bound, bound_source),
                    ));
                }
                Some(hessian)
            }
            Ok(None) => {
                return Err(outer_nonconvergence_error(
                    context,
                    "the objective declared analytic curvature but returned none at the final point",
                    result,
                    Some(projected_grad_norm),
                    StationarityBound::from_ladder(stationarity_bound, bound_source),
                ));
            }
            Err(err) => {
                return Err(outer_nonconvergence_error(
                    context,
                    &format!("analytic final Hessian could not be certified: {err}"),
                    result,
                    Some(projected_grad_norm),
                    StationarityBound::from_ladder(stationarity_bound, bound_source),
                ));
            }
        }
    } else {
        None
    };

    // #2458 — the same escalation, for the routes that cannot take the one
    // above.
    //
    // The block above closes a gap between two FIDELITIES of one route. The
    // remaining gap is between two ROUTES: a route declaring
    // `DeclaredHessianForm::Unavailable` has no order-four ladder to spend at
    // either fidelity, so it can never reach the curvature-resolvability rung
    // and is held to the raw reproducibility band -- the strictest tier in the
    // subsystem, awarded to the route that knows the least. Measured across six
    // tests of one subsystem in one binary: bounds from 5.675e-6 to 4.771e-2, a
    // factor of 8,406, with one |Pg| = 4.637052e-7 graded against four of them.
    //
    // That gap is #2458 and it is real. It is NOT closed here, and the attempt
    // to close it here (`f2cae93ee`, reverted by `finite_difference_outer_hessian`'s
    // removal) is the reason this comment exists rather than a code block.
    // Forward-differencing the route's analytic gradient produces a number that
    // decides which fits are certified, and SPEC line 2 permits finite
    // differences only in tests. Production differences nothing: the psi audit
    // `run_plan.rs` once ran at an armed seed is now an analytic seed probe, and
    // the test that arms it forms its own difference (#2901).
    //
    // The correct fix is upstream and is being applied there: a route with no
    // analytic Hessian should acquire one, not have one estimated on its behalf
    // by the code judging it. The named producer -- the constant-curvature outer
    // problem, a ONE-parameter profiled Gaussian REML -- now derives its single
    // second derivative in closed form and declares `DeclaredHessianForm::Dense`
    // (`gam-models/src/fit_orchestration/drivers/spatial_optimization.rs`), so it
    // reaches `CurvatureResolvability` by the ordinary path with the family's own
    // exact curvature. A route that still declares `Unavailable` is held to the
    // raw band and the run record says so (`derived_standard=false`), which is a
    // typed inability to certify rather than a silently different standard.

    // Curvature-scaled stationarity (#2253/#2249/#2015/#2091). The re-measured
    // projected gradient can sit modestly ABOVE the gradient-magnitude bands even
    // though NO step reduces the objective by more than the outer
    // tolerance — a weakly-identified small-n fit reaches this by a flat-valley
    // cost-stall, and an *already-stationary* fit reaches it at iteration 0 when
    // the plan search exhausts without stepping (a 2-parameter Gaussian-linear
    // REML lands λ→0 at a genuine interior optimum whose |Pg|≈1e-7 sits just above
    // an absolute score·1e-9 gradient floor tighter than the REML gradient's
    // matrix-factorization round-off). Whether that residual is genuine descent is
    // a second-order question the flat bands above cannot answer: they are
    // gradient-magnitude tests, blind to how the local curvature maps a gradient
    // to an objective change. The Newton decrement `½·gᵀH⁻¹g` (see
    // `newton_predicted_decrease`) IS that map — the exact predicted improvement of
    // a safeguarded second-order step. When it is below the outer objective
    // tolerance, the point is stationary at the resolution the criterion can be
    // optimized ("no further descent possible"), independent of HOW the solver
    // stopped.
    //
    // Applied whenever a PSD-along-gradient analytic Hessian is in hand (NOT gated
    // to a specific exit reason: the decrement test is the certificate, the exit
    // reason is not). It can NEVER wrongly certify a fit with real available
    // descent: it only widens when `curvature_grad_bound > stationarity_bound`, so
    // a well-identified fit that already clears `solver_bound` is untouched; a
    // gradient aligned with a near-flat Hessian direction inflates the decrement
    // and is rejected; a globally indefinite Hessian is rejected independently by
    // the `hessian_psd` gate inside `certifies()`. The derived widening is a
    // genuine, direction-aware curvature-scaled GRADIENT bound — the largest ‖Pg‖
    // that, in this gradient's direction under this curvature, still predicts a
    // decrease of exactly `objective_tol` — not a constant bump: because the
    // decrement scales quadratically with ‖g‖ at fixed direction, that bound is
    // `‖Pg‖·√(objective_tol/Δpred)`, which clears the actual ‖Pg‖ iff
    // `Δpred ≤ objective_tol`.
    // `screening_bound_curvature` is the #2596 escalation: at `Mint` it is
    // always `None` and this reads `analytic_hessian` exactly as before; at
    // `Screening` it is `Some` only on the would-refuse path, and it feeds THIS
    // rung and nothing else.
    //
    // #2954: where the route declares its size the decrement on rounding bands IS
    // the standard, and it may tighten every first-order band above as well as
    // widen them. A genuine saddle keeps the first-order ladder, so the
    // negative-curvature adjudication below still runs on it.
    //
    // The coordinates railed on their domain faces (the infinite-smoothing ceiling). Their
    // saturated curvature direction makes the FULL Hessian indefinite, so the
    // flatness certificate below — and the final curvature gate — judge PSD on the
    // interior (un-railed) sub-block instead, or a rail-caused indefiniteness would
    // disable the very certificate that exists to certify a railed optimum (#2299).
    // The FACE the certificate reasons on: every θ coordinate against its own
    // bound, ρ and non-ρ alike. See `certificate_railed_coordinates` for why
    // the λ-block report cannot be used here.
    let certificate_railed = certificate_railed_coordinates(&result.rho, config);
    let decrement_decided = analytic_hessian
        .as_ref()
        .or(screening_bound_curvature.as_ref())
        .and_then(|hessian| {
            match outer_decrement_verdict(
                config,
                hessian,
                &projected_gradient,
                &certificate_railed,
                evaluation.cost,
                &terminal_evidence,
            ) {
                Ok(decision) => Some(decision),
                Err(reason) => {
                    log::debug!(
                        "[CERTIFICATE] {context}: Newton-decrement verdict not taken: {reason}; \
                         the first-order ladder decides (#2954)"
                    );
                    None
                }
            }
        })
        .and_then(|decision| {
            decrement_stationarity_bound(projected_grad_norm, &decision.verdict)
                .map(|decided| (decision, decided))
        });
    if let Some((decision, (bound, source))) = decrement_decided.as_ref() {
        let verdict = &decision.verdict;
        log::debug!(
            "[CERTIFICATE] {context}: Newton-decrement verdict {verdict:?} (tolerance \
             {:.3e} = max(τ_stat {:.3e} − band_f, band_f), arithmetic-limited {}; band_f = \
             channels {:.3e} + factor {:.3e} + inner residual {:.3e}); face {:?}, released \
             {:?}; bound {bound:.3e} (rung {}) replaces {stationarity_bound:.3e} (rung {}) at \
             |Pg|={projected_grad_norm:.3e} (#2954)",
            decision.tolerance.value(),
            decision.tolerance.tau_stat,
            decision.tolerance.arithmetic_limited(),
            decision.objective_band.channels,
            decision.objective_band.factor,
            decision.objective_band.inner_residual,
            decision.face,
            decision.released,
            source.label(),
            bound_source.label(),
        );
        stationarity_bound = *bound;
        bound_source = *source;
    } else if let Some(hessian) = analytic_hessian
        .as_ref()
        .or(screening_bound_curvature.as_ref())
        // At the criterion's curvature resolution, the same standard the ARC
        // bridge's in-loop stop takes its decrement at (#2817). A negative
        // eigenvalue below that resolution is withdrawn by the adjudication, not
        // refused; before the decrement travelled with it, this rung found no
        // factor there and certified strictly less than the loop had stopped on.
        && let Some(predicted_decrease) = newton_predicted_decrease_at_resolution(
            hessian,
            &projected_gradient,
            criterion_curvature_resolution(outer_criterion_resolution(config)),
        )
        && predicted_decrease.is_finite()
        && predicted_decrease > 0.0
    {
        // The criterion's resolution, the SAME one the cost-stall guard declares
        // the criterion stalled at (run_plan.rs), so certification asserts
        // nothing tighter than the loop already proved about this surface.
        let objective_tol = outer_criterion_resolution(config);
        let curvature_grad_bound =
            projected_grad_norm * (objective_tol / predicted_decrease).sqrt();
        if curvature_grad_bound.is_finite() && curvature_grad_bound > stationarity_bound {
            // The bound widens whenever it exceeds the gradient band, but it clears
            // |Pg| only when the decrement is inside the tolerance. Render the
            // comparison this point actually satisfies: the historical line always
            // printed `≤`, including on refusals where the decrement was 360× over.
            let relation = if predicted_decrease <= objective_tol { "≤" } else { ">" };
            log::debug!(
                "[CERTIFICATE] {context}: curvature-scaled flat-valley bound {curvature_grad_bound:.3e} \
                 (|Pg|={projected_grad_norm:.3e}, Newton ½gᵀH⁻¹g={predicted_decrease:.3e} {relation} tol {objective_tol:.3e}) \
                 widened from gradient-band {stationarity_bound:.3e}"
            );
            stationarity_bound = curvature_grad_bound;
            bound_source = StationarityBoundSource::CurvatureResolvability;
        }
    }

    // #2954: a resolvable decrement at the mint is polished, railed at a limit
    // model, or refused by name (`newton_polish::polish_the_mint`); it never
    // reaches the first-order ladder below.
    if matches!(fidelity, CertificationFidelity::Mint)
        && let Some((decision, _)) = decrement_decided.as_ref()
        && let Some(evidence) = resolvable_decrease_evidence(&decision.verdict)
    {
        let inputs = super::newton_polish::MintPolish {
            allow_certify_reseed,
            fidelity,
            polish,
            decision,
            evidence,
            cost: evaluation.cost,
            analytic_hessian: &analytic_hessian,
            projected_gradient: &projected_gradient,
            projected_grad_norm,
            bounds: &bounds,
            layout,
            stationarity_bound,
            bound_source,
        };
        return super::newton_polish::polish_the_mint(obj, config, context, result, inputs);
    }

    // Gradient-reproducibility floor (#2299 fully-saturated smooth). A
    // stationarity certificate cannot resolve below the reproducibility of its
    // own measuring instrument: at a rail-adjacent optimum (λ ~ 1e12, the term
    // collapsed onto its penalty null space, edf saturated) the analytic
    // gradient is a difference of enormous canceling log-det terms whose
    // evaluation drifts run to run, so |Pg| measures round-off, not slope —
    // observed as the SAME ρ returning |g| ∈ {2.5e-3 … 4.5e-2} across
    // consecutive evaluations while the objective stays flat to 1e-7.
    //
    // The certifier may already hold TWO independent measurements at this ρ:
    // the optimizer's recorded best-iterate measurement (`run_recorded`) and
    // the fresh certificate-time `evaluation` — so the instrument's
    // demonstrated noise costs ZERO additional objective evaluations (scripted
    // test objectives keep their exact call counts). A REAL residual gradient
    // reproduces (spread ≈ 0, no widening — genuine descent can never be
    // masked, and a deterministic objective yields bit-identical pairs), while
    // cancellation noise decorrelates (spread ~ |Pg|). The widening is gated
    // on the recorded measurement having been taken at exactly this ρ, and on
    // the two measurements' objective VALUES agreeing to the same relative
    // floor the cost-stall guard uses; the PSD gate below is unchanged.
    //
    // #2953: the point gate is what makes the spread a measure of noise. The
    // gradients of two DIFFERENT points differ by the slope between them, and
    // on a deterministic objective that is the only way the spread can be
    // nonzero, so without the gate the floor widened exactly where the
    // criterion was not flat.
    //
    // A decrement verdict is not widened here (#2954): measured gradient noise can
    // only make its decrement unresolvable, never make a resolvable decrease
    // stationary.
    if decrement_decided.is_none()
        && projected_grad_norm > stationarity_bound
        && let Some(prior) = run_recorded.as_ref()
        && !prior.is_at(&result.rho)
    {
        log::debug!(
            "[CERTIFICATE] {context}: gradient-reproducibility floor not applied: the \
             run-recorded measurement was taken at rho={:?}, not at the certified rho={:?} \
             (#2953)",
            prior.rho().to_vec(),
            result.rho.to_vec(),
        );
    }
    //
    // A mint that follows a screening pass at this ρ holds the screening's reset
    // re-measurement as `run_recorded`, a bit-for-bit replay of its own
    // evaluation, so the solver's measurement that screening displaced is
    // weighed too. Otherwise the mint refuses on a spread of exactly zero a
    // point the screening certified on the solver's evidence at the same ρ.
    for prior in run_recorded.iter().chain(displaced.iter()) {
        if decrement_decided.is_some()
            || projected_grad_norm <= stationarity_bound
            || !prior.is_at(&result.rho)
            || layout
                .validate_gradient_len(prior.gradient(), "outer run-recorded gradient")
                .is_err()
            || !prior.gradient().iter().all(|value| value.is_finite())
            || !prior.value().is_finite()
        {
            continue;
        }
        const GRADIENT_REPRODUCIBILITY_WIDENING: f64 = 2.0;
        let objective_tol = outer_criterion_resolution(config);
        let cost_drift = (prior.value() - evaluation.cost).abs();
        let prior_projected = project_gradient_vector(
            &result.rho,
            prior.gradient(),
            Some(&rail_projection_bounds),
        );
        let spread = (&prior_projected - &projected_gradient)
            .iter()
            .map(|v| v * v)
            .sum::<f64>()
            .sqrt();
        let repro_bound = GRADIENT_REPRODUCIBILITY_WIDENING * spread;
        if cost_drift <= objective_tol
            && repro_bound.is_finite()
            && repro_bound > stationarity_bound
            && projected_grad_norm <= repro_bound
        {
            log::debug!(
                "[CERTIFICATE] {context}: gradient-reproducibility floor widened the \
                 stationarity bound to {repro_bound:.3e} (|Pg|={projected_grad_norm:.3e}, \
                 same-ρ spread between the run-recorded and certificate-time gradients \
                 {spread:.3e}, cost drift {cost_drift:.3e} ≤ tol {objective_tol:.3e})"
            );
            stationarity_bound = repro_bound;
            bound_source = StationarityBoundSource::GradientReproducibility;
        }
    }

    // #2458/#2479 -- the bound's own provenance, emitted UNCONDITIONALLY rather
    // than only when a rung happens to widen. A certificate that does not carry
    // which of its five terms decided it can only be re-derived from source,
    // never audited; the same complaint this file's refusal messages make about
    // the fits they refuse. Fidelity and the exact rho identify whether a
    // screening verdict and the terminal mint measured the same candidate.
    // `derived_standard=false` is the actionable bit: it says this verdict rests
    // on a gradient-magnitude substitute because the resolvability form was
    // unavailable on this route, not because the problem called for it.
    log::debug!(
        "[CERTIFICATE-BOUND] {context}: fidelity={fidelity:?}, rho={:?}, \
         bound {stationarity_bound:.6e} set by {} (derived_standard={}, \
         |Pg|={projected_grad_norm:.6e}, |g|={grad_norm:.6e}, \
         solver_band={solver_bound:.6e}, cost={:.6e})",
        result.rho.to_vec(),
        bound_source.label(),
        bound_source.is_derived_standard(),
        evaluation.cost,
    );

    // Large-step flatness certificate (#2299 fully-saturated smooth). After the
    // reproducibility floor a coordinate that has collapsed EXACTLY onto its
    // penalty null space (λ ~ 1e12, edf saturated) can still carry a projected
    // gradient component that is DETERMINISTIC cancellation bias from the
    // 1e12-conditioned logdet derivative (≈ ε·κ·scale). Being deterministic it
    // reproduces run to run, so the spread-keyed reproducibility floor above
    // cannot see it; and the Newton decrement anti-rescues, because the near-null
    // Hessian direction inflates gᵀH⁻¹g by design. The decisive question is
    // second-order-independent: does the criterion actually MOVE along that
    // coordinate at MACROSCOPIC scale? This block answers it directly — it probes
    // the objective a full e-fold in λ to either side of a near-null-curvature
    // coordinate and, for coordinates whose value is provably flat there, removes
    // their measured gradient component (numerical bias, not slope) from the
    // projected gradient before the bound test. A coordinate whose large-step
    // value MOVES is left untouched, so a genuine pseudologdet ramp still refuses.
    //
    // Gated as narrowly as possible: it runs only when the certificate would
    // OTHERWISE refuse on |Pg|, only with an analytic Hessian that is PSD-within-
    // noise in hand, and only probes coordinates whose curvature row is below the
    // roundoff floor — so a well-conditioned objective (every scripted mock at its
    // certification point) probes nothing and pays zero extra evaluations.
    // #2676: the directions along which THIS criterion is exactly constant by
    // construction of its penalty map. Read once, at the certified point, and
    // threaded to every curvature test below, so the certificate and the
    // smoothing correction judge the same subspace instead of being able to
    // reach opposite verdicts on one matrix at one point. `None` for every
    // objective that declares no invariance, which restores the pre-#2676
    // behaviour bit for bit.
    let criterion_invariance = obj.criterion_invariant_directions(&result.rho);
    if let Some(basis) = criterion_invariance.as_ref() {
        log::debug!(
            "[CERTIFICATE] {context}: deflating {} criterion-invariant direction(s) of {} \
             before the curvature verdict -- their rho-curvature is the chain-rule term \
             `sum_k g_k t_k^2` identically, so its sign measures the gradient code against \
             the Hessian code, not the fit (#2676)",
            basis.ncols(),
            basis.nrows(),
        );
    }
    // The λ-block REPORT that ships on the certificate. Same predicate,
    // narrower scan, because `lambdas_railed` indexes smoothing parameters.
    let railed_lambda_block = certificate_railed_lambdas(&result.rho, layout.rho_dim(), config);

    // Typed stationary-at-asymptote rail certificate (#2348 Inc 1, #2299 layer 3,
    // #2337 Thm 2.1). Before falling through to the generic gradient/criterion-flat
    // verdict, POSITIVELY certify a railed optimum: the interior (non-railed)
    // coordinates are gradient-stationary, and each coordinate railed at the
    // infinite-/zero-smoothing box bound sits on a confirmed exponential tail whose
    // fitted model has already reached the rail limit to within the estimand
    // tolerance. This supersedes the untyped `lambdas_railed` flag with a proof that
    // the criterion improvement and coefficient travel still available by running to
    // the rail are both below tolerance.
    //
    // Computed in its own statement so the borrow of `analytic_hessian` ends before
    // the mint branch moves it onto the result. Gated to a genuinely railed optimum
    // with outward pull (`grad_norm` above the stationarity bound) and an analytic
    // Hessian: a well-conditioned interior fit, or a coordinate merely resting near a
    // bound with a vanishing gradient, probes nothing and keeps its ordinary verdict.
    let asymptote_objective_tol = outer_criterion_resolution(config);
    let rail_outcome = match analytic_hessian.as_ref() {
        Some(hessian) if !certificate_railed.is_empty() && grad_norm > stationarity_bound => {
            Some(try_certify_asymptote_rail(
                obj,
                &AsymptoteRailInputs {
                    rho: &result.rho,
                    projected_gradient: &projected_gradient,
                    railed: &certificate_railed,
                    layout,
                    hessian,
                    bounds: &bounds,
                    terminal_beta: terminal_beta.as_ref(),
                    stationarity_bound: StationarityBound::from_ladder(stationarity_bound, bound_source),
                    objective_tol: asymptote_objective_tol,
                    context,
                    native_coordinate_order: config.native_coordinate_order.as_deref(),
                },
            )?)
        }
        _ => None,
    };
    // A refused railed mint carries its typed decline reason into the final
    // refusal summary, so a railed non-mint names the gate that refused
    // instead of failing silently.
    let mut asymptote_rail_note: Option<String> = None;
    let mut probes_ran = rail_outcome.is_some();
    if let Some(outcome) = rail_outcome {
        match outcome {
            Err(reason) => asymptote_rail_note = Some(reason),
            Ok(minted) => {
                let (interior_projected_grad_norm, effective_interior_bound, rails) = minted;
                // The tail probes were derivative-bearing evaluations at probe
                // ρ's, so the EVALUATOR-side terminal-mode carrier now owns the
                // last probe, not the checkpoint (#2155 regression: every
                // custom-family at-point mint then failed the bitwise terminal
                // theta identity at fit assembly). Re-evaluate at the minted
                // point and ship ITS numbers as the terminal facts: the same
                // evaluation sets the evaluator carrier, so the optimizer
                // certificate and the owned mode are bitwise-identical by
                // construction. The certified stationarity facts (interior
                // norms, rails) remain the judged ones.
                let restored = obj
                    .eval_with_order(&result.rho, OuterEvalOrder::ValueAndGradient)
                    .map_err(|err| {
                        EstimationError::RemlOptimizationFailed(format!(
                            "{context}: failed to re-own the certified point after \
                             asymptote-rail probing: {err}"
                        ))
                    })?;
                result.final_value = restored.cost;
                let restored_projected = project_gradient_vector(
                    &result.rho,
                    &restored.gradient,
                    Some(&rail_projection_bounds),
                );
                result.final_grad_norm = Some(
                    restored_projected
                        .iter()
                        .map(|v| v * v)
                        .sum::<f64>()
                        .sqrt(),
                );
                result.record_measurement_at_rho(restored.cost, restored.gradient);
                let certificate = OuterCriterionCertificate {
                    stationarity: OuterStationarityCertificate::AsymptoteRail {
                        interior_projected_grad_norm,
                        // The bound that admitted the interior: the raw stationarity
                        // bound, or the curvature-scaled flat-valley widening when the
                        // interior sub-block's Newton decrement is below the loop's
                        // cost resolution (shared judgment with the Inc 2c mint).
                        bound: effective_interior_bound.value(),
                        rung: effective_interior_bound.rung().into(),
                        rails,
                    },
                    curvature: CurvatureEvidence::Measured { psd: true },
                    lambdas_railed: railed_lambda_block.clone(),
                    railed_facts: railed_coordinate_facts(
                        &result.rho,
                        // #2624: theta-wide, see `certificate_railed_coordinates`.
                        &certificate_railed,
                        config,
                    ),
                    newton_polish: None,
                    curvature_floor: None,
                };
                // Move the certified curvature onto the result; the mint path returns
                // immediately, so the fall-through below never observes the move.
                result.final_hessian = analytic_hessian;
                result.criterion_certificate = Some(certificate.clone());
                if !certificate.certifies() {
                    return Err(outer_nonconvergence_error(
                        context,
                        &native_certificate_summary(&certificate, config),
                        result,
                        Some(interior_projected_grad_norm),
                        StationarityBound::from_ladder(stationarity_bound, bound_source),
                    ));
                }
                result
                    .termination
                    .certify(OuterConvergedVia::AsymptoteStationary {
                        rails: certificate.stationarity.rails().len(),
                    });
                log::debug!("[CERTIFICATE] {context}: {}", native_certificate_summary(&certificate, config));
                return Ok(certificate);
            }
        }
    }

    let mut certified_projected_grad_norm = projected_grad_norm;
    if projected_grad_norm > stationarity_bound
        && let Some(hessian) = analytic_hessian.as_ref()
        && certificate_hessian_is_psd_off_railed(
            hessian,
            &certificate_railed,
            criterion_invariance.as_ref(),
        ) == Some(true)
    {
        let n = layout.n_params;
        // Curvature scale of the analytic outer Hessian: its dominant diagonal,
        // the same ‖H‖ scale `certificate_hessian_is_psd_at_resolution` and
        // `newton_predicted_decrease` regularize against. A coordinate's curvature
        // ROW is indistinguishable from the assembly's roundoff — it has no
        // curvature the arithmetic can resolve and has collapsed onto the penalty
        // null space — when its largest entry falls below the SAME √ε·‖H‖ margin
        // those two probes use to separate a real curvature direction from
        // O(ε·‖H‖) accumulation noise. This is the derivation of the threshold:
        // NULL_CURVATURE_REL = √ε (machine epsilon's square root, the assembled
        // Hessian's relative resolution), scaled by the Hessian's own max-diagonal
        // magnitude, floored at 1 exactly as the PSD/Newton shift is.
        let max_diag = (0..n).fold(0.0_f64, |acc, j| acc.max(hessian[[j, j]].abs()));
        let null_curvature_threshold = f64::EPSILON.sqrt() * max_diag.max(1.0);
        // The SAME criterion resolution the cost-stall guard and both widenings
        // above use: certification asserts nothing tighter about this surface's
        // macroscopic flatness than the loop already proved.
        let objective_tol = outer_criterion_resolution(config);
        // One e-fold in log-λ per coordinate (ρ IS log-λ): the +δ/−δ pair spans e²
        // in λ, a macroscopic move across which no genuine descent slope can hide.
        const LARGE_STEP_DELTA: f64 = 1.0;
        let mut saturated_flat: Vec<usize> = Vec::new();
        let mut probe_reports: Vec<String> = Vec::new();
        let mut probed_any = false;
        let mut probe_failed = false;
        for k in 0..n {
            let row_inf = (0..n).fold(0.0_f64, |acc, j| acc.max(hessian[[k, j]].abs()));
            // Only near-null-curvature coordinates the measured gradient actually
            // loads on can be responsible for |Pg| exceeding the band; skip every
            // other coordinate, so no probe fires on a well-conditioned surface.
            if row_inf > null_curvature_threshold || projected_gradient[k] == 0.0 {
                continue;
            }
            let mut plus = result.rho.clone();
            plus[k] += LARGE_STEP_DELTA;
            let mut minus = result.rho.clone();
            minus[k] -= LARGE_STEP_DELTA;
            probed_any = true;
            let (Ok(cost_plus), Ok(cost_minus)) = (obj.eval_cost(&plus), obj.eval_cost(&minus))
            else {
                // A failed probe is not evidence of flatness — refuse to classify
                // (conservative) and leave |Pg| intact for the bound test.
                probe_failed = true;
                break;
            };
            if !cost_plus.is_finite() || !cost_minus.is_finite() {
                probe_failed = true;
                break;
            }
            let up = (cost_plus - evaluation.cost).abs();
            let down = (cost_minus - evaluation.cost).abs();
            if up <= objective_tol && down <= objective_tol {
                saturated_flat.push(k);
                probe_reports.push(format!(
                    "k={} |ΔV|+={up:.3e} |ΔV|-={down:.3e}",
                    native_coordinate(config.native_coordinate_order.as_deref(), k)
                ));
            }
        }
        if !probe_failed && !saturated_flat.is_empty() {
            // Recompute |Pg| with the provably macroscopically-flat coordinates
            // removed: their measured gradient is deterministic cancellation bias,
            // not slope. Coordinates that moved keep their component and still count
            // against the bound.
            let reduced_sq = (0..n)
                .filter(|k| !saturated_flat.contains(k))
                .map(|k| projected_gradient[k] * projected_gradient[k])
                .sum::<f64>();
            certified_projected_grad_norm = reduced_sq.sqrt();
            let flat_list = saturated_flat
                .iter()
                .map(|&k| native_coordinate(config.native_coordinate_order.as_deref(), k).to_string())
                .collect::<Vec<_>>()
                .join(", ");
            let probe_summary = probe_reports.join("; ");
            log::debug!(
                "[CERTIFICATE] {context}: large-step flatness certificate classified \
                 coordinate(s) [{flat_list}] saturated-flat (curvature row ≤ \
                 {null_curvature_threshold:.3e}, probed Δ=±{LARGE_STEP_DELTA} with \
                 {probe_summary}, cost-flat to tol {objective_tol:.3e}); projected \
                 gradient reduced from {projected_grad_norm:.3e} to \
                 {certified_projected_grad_norm:.3e}"
            );
            // #2954: the saturated-flat coordinates join the face, and the decrement
            // is taken again on what remains.
            if decrement_decided.is_some() {
                let mut face = certificate_railed.clone();
                let mut reduced = projected_gradient.clone();
                for &k in &saturated_flat {
                    reduced[k] = 0.0;
                    face.push(k);
                }
                if let Ok(decision) = outer_decrement_verdict(
                    config,
                    hessian,
                    &reduced,
                    &face,
                    evaluation.cost,
                    &terminal_evidence,
                ) && let Some((bound, source)) =
                    decrement_stationarity_bound(certified_projected_grad_norm, &decision.verdict)
                {
                    let verdict = &decision.verdict;
                    log::debug!(
                        "[CERTIFICATE] {context}: Newton-decrement verdict {verdict:?} with the \
                         saturated-flat coordinate(s) on the face; bound {bound:.3e} (rung {}) at \
                         |Pg|={certified_projected_grad_norm:.3e} (#2954)",
                        source.label(),
                    );
                    stationarity_bound = bound;
                    bound_source = source;
                }
            }
        }
        // `eval_cost` warm-starts the inner solve, so the probes moved the objective
        // off the certified point. Restore it to ρ̂ once iff we actually probed, so
        // the downstream state sees the fitted point. A failure to re-evaluate the
        // same ρ that certified moments ago is a genuinely broken objective and
        // refuses conservatively.
        if probed_any {
            obj.eval_cost(&result.rho).map_err(|err| {
                outer_nonconvergence_error(
                    context,
                    &format!(
                        "failed to restore the objective to the certified point after \
                         flatness probing: {err}"
                    ),
                    result,
                    Some(certified_projected_grad_norm),
                    StationarityBound::from_ladder(stationarity_bound, bound_source),
                )
            })?;
        }
    }

    let mut certificate = OuterCriterionCertificate {
        stationarity: OuterStationarityCertificate::AnalyticGradient {
            grad_norm,
            projected_grad_norm: certified_projected_grad_norm,
            bound: stationarity_bound,
            rung: bound_source.provenance().into(),
        },
        // The RAW measurement — unchanged, and what every consumer that asks
        // for a genuine PSD certificate keeps receiving.
        curvature: match analytic_hessian.as_ref() {
            Some(hessian) => CurvatureEvidence::from_measurement(
                certificate_hessian_is_psd_off_railed(
                    hessian,
                    &certificate_railed,
                    criterion_invariance.as_ref(),
                ),
            ),
            // A screening pass deliberately declines the order-four ladder
            // (the documented design at `CertificationFidelity`); a Mint pass
            // reaching here simply has no analytic Hessian to test. Those were
            // the same `None` before #2561, which is why the design's own
            // promise — the winner's verdict is the one that mints — could not
            // be checked by anyone.
            None if matches!(fidelity, CertificationFidelity::Screening) => {
                CurvatureEvidence::NotSpent
            }
            None => CurvatureEvidence::NotAvailable,
        },
        lambdas_railed: railed_lambda_block.clone(),
        // #2624: `lambdas_railed` above is the lambda-scoped REPORT and stays
        // that way; `railed_facts` is the EVIDENCE for a decision taken on the
        // theta-wide face (`certificate_railed`, used by the off-railed PSD
        // test and the curvature floor immediately below), so it must carry the
        // same coordinates that decision was taken on. On an exact-joint
        // spatial route the psi coordinate is the only one carrying gradient,
        // and it was the one coordinate the refusal could not print.
        railed_facts: railed_coordinate_facts(&result.rho, &certificate_railed, config),
        // #2954: the Newton steps the mint took before judging, with `λ̂²` at the
        // judged point.
        newton_polish: polish.map(|walk| walk.record).map(|mut record| {
            if let Some((decision, _)) = decrement_decided.as_ref()
                && let opt::DecrementVerdict::Certified(evidence) = &decision.verdict
            {
                record.lambda_sq_after = evidence.lambda_sq;
            }
            record
        }),
        // The floor's verdict on that same curvature, recorded beside it.
        curvature_floor: analytic_hessian.as_ref().and_then(|hessian| {
            interior_curvature_floor_clearance(
                hessian,
                &certificate_railed,
                &projected_gradient,
                criterion_invariance.as_ref(),
            )
        }),
    };
    // Install the measured evidence before deciding its verdict.  A rejected
    // candidate is retained only as a resumable checkpoint, and that
    // checkpoint must carry the actual analytic residual/curvature evidence
    // that caused the rejection rather than the optimizer's stale terminal
    // status.
    result.final_hessian = analytic_hessian;
    result.criterion_certificate = Some(certificate.clone());
    // Screening deliberately spends no curvature. The caller's strict
    // second-order requirement belongs exclusively to the terminal mint; applying
    // it here would reject every first-order candidate as `NotSpent` and turn a
    // two-basin comparison into seed-budget exhaustion.
    let mut curvature_requirement_met =
        certificate_meets_curvature_requirement(&certificate, config.require_measured_psd, fidelity);
    // #2612 — ADJUDICATE a curvature refusal against the criterion, BEFORE
    // deciding it.
    //
    // This block used to live inside the refusal below, which meant its verdict
    // could only ever mint a reseed: a run that found no descending trial
    // returned `None`, indistinguishable from "the escape was never runnable",
    // and the refusal then proceeded on the matrix's word. But "no feasible step
    // along the reported negative eigenvector lowers the objective, anywhere in
    // the range where the claim predicts a decrease the criterion can represent"
    // is a MEASUREMENT of the criterion, and it contradicts the matrix. Spending
    // a refusal on evidence the criterion has just falsified is the failure mode
    // #2665 documented from the other side (an analytic `λ_min = −1721.5` whose
    // objective curvature along the same eigenvector is `+121.6`), and no
    // resolution bound can catch it — the matrix is not imprecise there, it is
    // wrong.
    //
    // Moving it here changes nothing about a real saddle: a descending trial
    // still mints the same one-shot reseed, and the refusal that follows is
    // still the refusal a genuinely indefinite point earns.
    // The ADJUDICATION is a measurement and is NOT gated by `allow_certify_reseed`
    // (#2612). That flag is the one-shot budget for the RESEED — it exists so
    // the retry pass cannot mint a second escape and recurse. Adjudicating
    // cannot recurse: it evaluates a bounded, derived ladder of trial points and
    // its only two outcomes here are "a descent exists" (which still needs the
    // budget to be spent, and is refused below when there is none) and "the
    // criterion contradicts the matrix" (which publishes no reseed at all).
    // Gating the measurement on the reseed budget made the retry pass refuse on
    // the matrix's word for want of a measurement it could have made for free —
    // the same "nobody looked" / "it was measured and it is a saddle" collapse
    // `CurvatureEvidence` was introduced to prevent.
    let strict_curvature_refused =
        config.require_measured_psd && certificate.hessian_psd() == Some(false);
    result.saddle_escape_reseed = None;
    if certificate.is_stationary()
        && (!certificate.curvature_not_refused() || strict_curvature_refused)
        && let Some(hessian) = result.final_hessian.clone()
        && let Some(gradient) = result.final_gradient().cloned()
    {
        probes_ran = true;
        let saddle_rho = result.rho.clone();
        let baseline_cost = result.final_value;
        // `curvature_not_refused()` is `false` exactly when the REDUCED
        // (off-railed) Hessian is indefinite, so a railed coordinate does not
        // waive the adjudication: rails are passed through and held fixed while
        // the step searches the free-direction saddle (#2155). They must be held
        // fixed on the SAME face the reduction was taken on — the certificate's
        // λ-block report would leave a railed ψ free to be stepped out of its
        // box.
        match adjudicate_negative_curvature(
            obj,
            &saddle_rho,
            &gradient,
            &hessian,
            &certificate_railed,
            criterion_invariance.as_ref(),
            baseline_cost,
            asymptote_objective_tol,
            &bounds,
            context,
        ) {
            SaddleAdjudication::Descended(point) => {
                // The reseed — and only the reseed — is one-shot. On the retry
                // pass the descent is still a real finding: the criterion agrees
                // with the matrix, so the refusal that follows is the refusal a
                // genuine saddle earns, and it is recorded as such rather than
                // as an escape that was never run.
                if allow_certify_reseed {
                    result.saddle_escape_reseed = Some(point);
                } else {
                    log::debug!(
                        "[CERTIFICATE] {context}: the criterion CONFIRMS the reported negative \
                         curvature (a feasible trial along its eigenvector lowers the objective \
                         by more than the criterion's resolution), and the one-shot escape \
                         reseed has already been spent on this fit. The refusal that follows is \
                         a measured saddle, not an unmeasured one (#2612)."
                    );
                }
            }
            SaddleAdjudication::Contradicted {
                probed,
                smallest_step,
                predicted_at_smallest,
                objective_resolution,
                best_seen_cost,
            } => {
                log::debug!(
                    "[CERTIFICATE] {context}: WITHDRAWING the curvature verdict — {probed} \
                     evaluated trial(s) down to step {smallest_step:.3e}, where the claim's own \
                     predicted decrease {predicted_at_smallest:.3e} reaches the criterion's \
                     resolution {objective_resolution:.3e}; best cost seen {best_seen_cost:.9e} \
                     never fell below the checkpoint. The certificate records \
                     `criterion-contradicted`, NOT a PSD claim (#2612)."
                );
                // The verdict is withdrawn, not inverted: nothing here showed
                // the point IS a minimum. `curvature_floor` goes with it —
                // every field in it (`interior_min_eigenvalue`, the floor, the
                // floored eigenvalue) is a statement about the matrix whose
                // negative direction has just been falsified, and reporting
                // them beside a withdrawn verdict is exactly the #2550
                // misdirection.
                certificate.curvature = CurvatureEvidence::CriterionContradicted;
                certificate.curvature_floor = None;
                result.criterion_certificate = Some(certificate.clone());
                curvature_requirement_met = certificate_meets_curvature_requirement(
                    &certificate,
                    config.require_measured_psd,
                    fidelity,
                );
            }
            SaddleAdjudication::Unresolvable {
                lambda_min,
                predicted_at_largest,
                objective_resolution,
            } => {
                log::debug!(
                    "[CERTIFICATE] {context}: WITHDRAWING the curvature verdict — \
                     lambda_min={lambda_min:.6e} predicts at most {predicted_at_largest:.3e} at \
                     the adjudication's largest step, which the criterion's resolution \
                     {objective_resolution:.3e} does not resolve. The certificate records \
                     `criterion-unresolvable`, NOT a PSD claim (#3036)."
                );
                // Withdrawn exactly as a contradicted verdict is, and for the
                // same reason: the floor's fields describe a negative direction
                // the criterion cannot represent at any step it may take.
                certificate.curvature = CurvatureEvidence::CriterionUnresolvable;
                certificate.curvature_floor = None;
                result.criterion_certificate = Some(certificate.clone());
                curvature_requirement_met = certificate_meets_curvature_requirement(
                    &certificate,
                    config.require_measured_psd,
                    fidelity,
                );
            }
            SaddleAdjudication::Declined(reason) => {
                log::debug!("[CERTIFICATE] {context}: saddle escape declined -- {reason}");
            }
        }
    }
    if !certificate.certifies() || !curvature_requirement_met {
        // Mint the #2392 reseeds fresh for THIS refused point: clear any value a
        // prior (multistart / pre-polish) certification of a different ρ left on
        // the result so the resume loop never consumes a stale pull-back/freeze.
        result.wrong_rail_reseed = None;
        result.active_set_reseed = None;
        // Reaching here means the #2612 adjudication above did NOT withdraw the
        // curvature verdict: either it minted a descending reseed (a real
        // saddle, and this refusal carries the retry), or it declined, or the
        // refusal is not about curvature at all.
        // #2665: the escape below is the remedy for a point that is first-order
        // stationary yet refused for curvature — exactly the SAS/mixture-link
        // failures, where `|Pg|` lands 1-10x BELOW its bound and only the
        // second-order conjunct refuses, on a MEASURED analytic Hessian with
        // `lambda_min ~ -1.6e3` against a `~1e-5` gradient floor.
        //
        // But the guard is a FIVE-way conjunction, and three of its conjuncts
        // can be false because something was never populated rather than
        // because a decision was taken. When that happens the refusal message
        // says only "indefinite curvature", and the fact that the remedy was
        // never even attempted leaves no trace at all. Name the blocking
        // conjunct at the moment of refusal so a saddle refusal can be
        // attributed instead of guessed.
        if certificate.is_stationary()
            && (!certificate.curvature_not_refused() || strict_curvature_refused)
        {
            // `allow_certify_reseed` is no longer a blocker here: the adjudication
            // runs on every refusal (#2612) and only the RESEED is one-shot, so
            // the two remaining conjuncts are the only ways the measurement can
            // fail to happen at all.
            let escape_blocker = if result.final_hessian.is_none() {
                Some("final_hessian=None (no terminal Hessian retained on this route)")
            } else if result.final_measurement.is_none() {
                Some("final_measurement=None (no terminal gradient retained on this route)")
            } else {
                None
            };
            if let Some(reason) = escape_blocker {
                log::debug!(
                    "[CERTIFICATE] {context}: negative-curvature saddle escape NOT ATTEMPTED at a \
                     first-order-stationary point that is being refused for curvature \
                     (hessian_psd={:?}, require_measured_psd={}); blocked by {reason}. The \
                     refusal that follows is therefore not evidence that no escape exists — \
                     the remedy was never run (#2665).",
                    certificate.hessian_psd(),
                    config.require_measured_psd,
                );
            }
        }
        // #2392 — wrong-rail pull-back and active-set reduction. A coordinate at
        // the ρ box whose deep-λ terminal gradient is instrument noise leaves the
        // outer search unable to move it: the trust region's local model is flat
        // there. Two evidence-gated one-shot reseeds recover the fit (both gated
        // by `allow_certify_reseed` so the retry pass cannot recurse):
        //   (1) WRONG-RAIL PULL-BACK: the coordinate's clean-band probes (a few
        //       e-folds inside, above the noise floor) prove the objective
        //       DECREASES inward — it was driven to the wrong bound. Reseed it at
        //       its clean-band interior scale so the optimizer descends to the
        //       true interior optimum. Fires ONLY on the opposite-sign clean-tail
        //       proof, so a genuine λ→∞/λ→0 rail is never pulled off its bound.
        //   (2) ACTIVE-SET REDUCTION: no wrong rail, but the INTERIOR is not
        //       stationary while a rail is present — the railed coordinate's
        //       ill-conditioned Hessian row poisons the joint step. Freeze the
        //       KKT-ACTIVE rail(s) at their bound and re-run so the interior
        //       converges in the reduced space; the plan runner re-certifies the
        //       polished point under the ORIGINAL box, so the reduction can
        //       never redefine the feasible set.
        //
        //       "Active" is decided by the PROJECTOR, at freeze time, not by
        //       distance to a bound (#2454): a coordinate on its bound whose
        //       projected gradient survives still has feasible descent and is
        //       left free. This is where the un-freeze has to happen. Doing it
        //       after the reduced solve is not available — the retry is one-shot
        //       (`allow_certify_reseed` is cleared on it, so no second reseed can be
        //       published) and the only path back off a frozen bound is the
        //       wrong-rail pull-back, which demands a clean opposite-sign
        //       exponential tail and declines on any coordinate that has none.
        // (1) takes precedence: a wrong rail must be pulled back, never frozen.
        if allow_certify_reseed && !certificate_railed.is_empty() {
            let beta_norm = terminal_beta
                .as_ref()
                .map(|b| b.dot(b).sqrt())
                .filter(|v| v.is_finite())
                .unwrap_or(0.0);
            let mut rail_tol =
                AsymptoteTolerances::exp4_rail_bands(ASYMPTOTE_ESTIMAND_REL_TOL * (1.0 + beta_norm));
            rail_tol.tail_drift_rel = RAIL_TAIL_DRIFT_REL;
            let (lower, upper) = &bounds;
            let mut wrong_rail_point: Option<Array1<f64>> = None;
            for &k in certificate_railed.iter() {
                if k >= result.rho.len() || k >= lower.len() || k >= upper.len() {
                    continue;
                }
                let side = if (upper[k] - result.rho[k]).abs() <= (result.rho[k] - lower[k]).abs() {
                    AsymptoteSide::Upper
                } else {
                    AsymptoteSide::Lower
                };
                if let Some(target) = detect_wrong_rail_pullback(
                    obj,
                    &result.rho,
                    k,
                    side,
                    &rail_tol,
                    (lower[k], upper[k]),
                )? {
                    let mut reseed = result.rho.clone();
                    reseed[k] = target;
                    wrong_rail_point = Some(reseed);
                    break;
                }
            }
            if let Some(reseed) = wrong_rail_point {
                result.wrong_rail_reseed = Some(reseed);
            } else if let Some(hessian) = result.final_hessian.as_ref() {
                // Active-set reduction needs curvature to prove that the free
                // subspace is genuinely unpolished. Wrong-rail pull-back above
                // is a first-order tail-sign proof and deliberately does NOT:
                // gradient-only BFGS objectives can rail incorrectly too, and
                // withholding a valid pull-back merely because they do not
                // materialize H would make recovery depend on solver class.
                let interior_indices =
                    interior_face_indices(&projected_gradient, &certificate_railed);
                let interior_not_stationary = !interior_indices.is_empty()
                    && certify_interior_stationarity(
                        &projected_gradient,
                        &hessian,
                        &interior_indices,
                        StationarityBound::from_ladder(stationarity_bound, bound_source),
                        asymptote_objective_tol,
                    )
                    .is_err();
                if interior_not_stationary {
                    let mut froz_lower = lower.clone();
                    let mut froz_upper = upper.clone();
                    let mut reseed = result.rho.clone();
                    let mut froze_any = false;
                    for &k in certificate_railed.iter() {
                        if k >= reseed.len() {
                            continue;
                        }
                        // Freeze the KKT-ACTIVE set, not the proximity set
                        // (#2454). `certificate_railed` is a DISTANCE test —
                        // "within `CERTIFICATE_RAIL_MARGIN` of a bound" — and
                        // being near a bound says nothing about whether the
                        // constraint is active. The active set is the one the
                        // projector already decided: at a bound it keeps only
                        // the feasible-descent half, so a coordinate whose
                        // projected component survives is one the search can
                        // still move INWARD, and `interior_face_indices` (three
                        // statements up) has already classified it as interior
                        // for exactly that reason.
                        //
                        // Freezing it anyway pins the coordinate carrying the
                        // descent that made `interior_not_stationary` true in
                        // the first place, and the reduced solve then converges
                        // everything ELSE around it. Measured on #2454's Matérn
                        // monotone fixture, whose ψ seed sits on its box edge:
                        // ψ is frozen at −2.4849 with `∂V/∂ψ = −4.3164`
                        // (FD-confirmed), the three ρ converge to
                        // `‖g_ρ‖ = 1.03e-4`, the solver reports
                        // `claimed_converged=true` off that reduced norm, and
                        // the re-certification under the original box then
                        // refuses at `|Pg| = 4.316e0` against a `7.06e-4`
                        // bound — with ψ bit-identical to its seed after 26
                        // outer iterations, because it was never free to move.
                        //
                        // The un-freeze the reduction's contract promises ("a
                        // frozen coordinate whose gradient turns inward
                        // re-certifies under the ORIGINAL box") cannot rescue
                        // this: the gradient had ALREADY turned inward when the
                        // freeze was taken, so the reduced solve is asked to
                        // polish a face that was never active.
                        if projected_gradient
                            .get(k)
                            .is_some_and(|component| *component != 0.0)
                        {
                            log::debug!(
                                "[ACTIVE-SET] {context}: coordinate {} is within the rail \
                                 margin of its bound but its projected gradient is \
                                 {:.6e} (feasible descent remains), so it is INTERIOR and \
                                 stays free rather than being frozen (#2454)",
                                native_coordinate(config.native_coordinate_order.as_deref(), k),
                                projected_gradient[k],
                            );
                            continue;
                        }
                        let rail = if (upper[k] - reseed[k]).abs() <= (reseed[k] - lower[k]).abs() {
                            upper[k]
                        } else {
                            lower[k]
                        };
                        reseed[k] = rail;
                        froz_lower[k] = rail;
                        froz_upper[k] = rail;
                        froze_any = true;
                    }
                    if froze_any {
                        result.active_set_reseed = Some(ActiveSetReseed {
                            rho: reseed,
                            bounds: (froz_lower, froz_upper),
                        });
                    }
                }
            }
        }
        // Carry the railed-mint decline evidence into the
        // refusal so a railed or budget-exhausted crawl explains which
        // certificate gate refused instead of failing silently.
        let mut summary = native_certificate_summary(&certificate, config);
        if !curvature_requirement_met {
            // This gate refuses on two OPPOSITE grounds and used to report both
            // with one sentence, which reads as an optimizer failure in either
            // case (#2641):
            //
            //   * `hessian_psd=no`  — curvature WAS measured and is indefinite.
            //     A verdict about the point. Refusing is correct; #2665 is such
            //     a case (λ_min = -1.6e3 against a 1e-5 floor).
            //   * `hessian_psd=n/a` — curvature was never measured at all, so
            //     there is no eigenvalue and no verdict. Nothing about the point
            //     has been established either way.
            //
            // The second splits again, and one branch is a CONFIGURATION
            // CONTRADICTION rather than anything the optimizer did: a caller can
            // declare `DeclaredHessianForm::Unavailable` (e.g. a lane that
            // deliberately avoids realizing an O(n) second-order slab) while the
            // same outer problem sets `require_measured_psd`. That combination
            // can never be satisfied by any amount of optimizer work, so say so
            // at the refusal instead of sending the reader to the solver.
            let detail = match certificate.hessian_psd() {
                Some(false) => "this objective requires a positive-semidefinite analytic Hessian \
                     at the selected local minimum, and the Hessian measured HERE is indefinite \
                     (`hessian_psd=no`) — a curvature verdict about this point, not a missing \
                     measurement"
                    .to_string(),
                _ if !capability.hessian.is_analytic() => format!(
                    "this objective requires a MEASURED positive-semidefinite analytic Hessian at \
                     the selected local minimum, but the problem declared \
                     `{:?}`, so no Hessian was ever evaluated and `hessian_psd` is \
                     unavailable. CONFIGURATION CONTRADICTION: the same outer problem both \
                     suppressed the analytic Hessian and required a measured one. No optimizer \
                     result can satisfy this — fix the construction (declare the Hessian, at \
                     least for the terminal certification evaluation) rather than the search",
                    capability.hessian
                ),
                _ => "this objective requires a MEASURED positive-semidefinite analytic Hessian \
                     at the selected local minimum. The Hessian is declared available, yet none \
                     was materialized at certification, so `hessian_psd` is unavailable and NO \
                     curvature verdict has been reached about this point"
                    .to_string(),
            };
            summary = format!("{summary}; {detail}");
        }
        // `summary()` prints `lambdas_railed`, which is the λ-block report. When
        // the face the certificate actually reasoned on is wider — a joint
        // [ρ, ψ] search with a κ coordinate on its data-derived window — say so,
        // or the refusal reads as `railed=[]` while the projector has already
        // discarded that coordinate's outward pull and nothing explains where
        // `|g|` went (#979).
        if certificate_railed.len() != railed_lambda_block.len() {
            summary = format!(
                "{summary}; outer coordinates railed (theta-wide, incl. non-rho blocks): \
                 {certificate_railed:?}"
            );
        }
        // #2465: `railed=[…]` without its box is unfalsifiable from the run
        // record. Every value here is live at the emission site.
        if !certificate_railed.is_empty() {
            summary = format!(
                "{summary}; rail tests: [{}]",
                rail_test_summary(&result.rho, &certificate_railed, config)
            );
        }
        // #2465 again, one level up: the `solver provenance` this refusal is
        // about to append reports the terminating `|g|` of the run that produced
        // `result`. When that run executed under an active-set reduction, its
        // box PINNED some coordinates (`lower == upper`) and its `|g|` therefore
        // ranges over the FREE ones only, while `|Pg|` above ranges over all of
        // θ under the model domain. The two are then not the same quantity, and
        // nothing in the string said so: on #2454's Matérn fixture the refusal
        // read `claimed_converged=true, gradient_tolerance(|g|=1.029235e-4 <
        // 5.487011e-4)` beside `|Pg|=4.316e0`, a four-order gap that is entirely
        // the pinned ψ coordinate and looks like a contradiction until the two
        // coordinate sets are named. Name them.
        if let Some((search_lower, search_upper)) = config.search_bounds_override.as_ref() {
            let pinned: Vec<usize> = (0..search_lower.len().min(search_upper.len()))
                .filter(|&k| search_lower[k] == search_upper[k])
                .collect();
            if !pinned.is_empty() {
                let pinned = native_coordinates(config.native_coordinate_order.as_deref(), &pinned);
                summary = format!(
                    "{summary}; NOTE the run that produced this point searched a REDUCED box \
                     with coordinate(s) {pinned:?} pinned (active-set reduction), so the \
                     solver-provenance |g| below ranges over the FREE coordinates only and is \
                     NOT comparable with the |Pg| above, which ranges over all of theta under \
                     the model domain"
                );
            }
        }
        if let Some(note) = asymptote_rail_note {
            summary = format!("{summary}; asymptote-rail declined: {note}");
        }
        return Err(outer_nonconvergence_error(
            context,
            &summary,
            result,
            Some(certified_projected_grad_norm),
            StationarityBound::from_ladder(stationarity_bound, bound_source),
        ));
    }

    // #2155 regression, the LAST carrier-stealing path: the rail-mint
    // attempts probe with derivative-bearing evaluations, and the
    // ORDINARY certificate can still certify after a declined attempt (e.g. a
    // KKT-railed projection whose raw gradient norm sits above the bound), so
    // this success would ship pre-probe terminal numbers while the evaluator's
    // terminal-mode carrier owns the last probe — refusing the bitwise theta
    // identity at custom-family fit assembly. Re-own the certified point with
    // one fresh evaluation and ship ITS numbers; the mint branches re-own for
    // themselves before their early returns, and the judged stationarity facts
    // above remain the measured pre-probe ones.
    if probes_ran {
        let restored = obj
            .eval_with_order(&result.rho, OuterEvalOrder::ValueAndGradient)
            .map_err(|err| {
                EstimationError::RemlOptimizationFailed(format!(
                    "{context}: failed to re-own the certified point after                      rail/tail probing: {err}"
                ))
            })?;
        result.final_value = restored.cost;
        let restored_projected = project_gradient_vector(
            &result.rho,
            &restored.gradient,
            Some(&rail_projection_bounds),
        );
        result.final_grad_norm = Some(
            restored_projected
                .iter()
                .map(|v| v * v)
                .sum::<f64>()
                .sqrt(),
        );
        result.record_measurement_at_rho(restored.cost, restored.gradient);
    }
    // #2235/#2241 — record WHICH certificate concluded this run. A
    // Fellner–Schall model-state fixed point was pre-stamped by the runner and
    // is preserved (this analytic certificate is its corroborating evidence);
    // otherwise the verdict is decided by which stationarity band the measured
    // projected gradient actually cleared: the solver's own tolerance
    // (gradient-stationary) or only the widened flat certificate band
    // (criterion-flat, #2241).
    let via = match result.termination.proposed_via() {
        Some(via @ OuterConvergedVia::RecurrentIncumbent { .. }) => via,
        _ if certified_projected_grad_norm <= solver_bound => {
            OuterConvergedVia::GradientStationary
        }
        _ => OuterConvergedVia::CriterionFlat {
            residual_grad_norm: certified_projected_grad_norm,
            certificate_bound: stationarity_bound,
        },
    };
    result.termination.certify(via);
    log::debug!("[CERTIFICATE] {context}: {}", native_certificate_summary(&certificate, config));
    Ok(certificate)
}

/// Estimand tolerance relative to the fitted coefficient scale for the
/// asymptote-rail certificate (#2348 Inc 1): the remaining coefficient travel
/// to the rail limit must fall below `ASYMPTOTE_ESTIMAND_REL_TOL·(1 + ‖β‖)` for
/// the fitted model to be certified equal to the rail-limit fit.
const ASYMPTOTE_ESTIMAND_REL_TOL: f64 = 1.0e-4;

/// Number of one-e-fold-in-`ρ` probes stepped back from a railed coordinate
/// toward the interior when reconstructing its exponential tail (#2348 Inc 1).
/// Enough to span both the finite-difference floor next to the rail (rejected)
/// and a confirmable-tail run further in.
// 18 e-folds: the window must REACH the finite-difference-clean constant-ĉ
// band from a coordinate railed AT the box ceiling. The fused-Hessian
// trajectory (#2348) rails fits at ρ=30 that previously stalled mid-box, and
// the #2299 fixture's clean band sits 13–16 e-folds inside — the old 12-probe
// window (sized for mid-box crawls) stopped one row short of it, so a fully
// confirmed tail declined with "no finite-difference-clean tail window". Six
// extra value+gradient evals, paid only at certification of railed fits.
const ASYMPTOTE_PROBE_COUNT: usize = 18;

/// Local confirmation resolution used only when the one-e-fold ladder cannot
/// find a clean run. A finite smoothing box can intersect a perfectly regular
/// asymptote before three whole e-folds of the leading-order tail are visible;
/// half-e-fold probes resolve that band without relaxing any certificate gate.
///
/// The probes remain equally spaced, which is load-bearing: the estimand
/// certificate interprets consecutive coefficient moves as one geometric
/// sequence. Six samples cover three e-folds, so the fallback still observes
/// curvature over a material interval instead of manufacturing constancy from
/// an arbitrarily small neighborhood.
const ASYMPTOTE_LOCAL_PROBE_DELTA: f64 = 0.5;
const ASYMPTOTE_LOCAL_PROBE_COUNT: usize = 6;

/// Read-only inputs to [`try_certify_asymptote_rail`], bundled so the certify
/// path passes one borrow rather than a long positional argument list.
struct AsymptoteRailInputs<'a> {
    rho: &'a Array1<f64>,
    projected_gradient: &'a Array1<f64>,
    railed: &'a [usize],
    /// Which slots of `rho` carry `log λ`. The active-set half of the
    /// certificate reads `railed` as-is; the tail law reads it through
    /// [`tail_law_coordinates`], which is the whole difference.
    layout: OuterThetaLayout,
    hessian: &'a Array2<f64>,
    bounds: &'a (Array1<f64>, Array1<f64>),
    terminal_beta: Option<&'a Array1<f64>>,
    /// The ladder bound the full problem was judged against, carrying the rung
    /// that set it. A gradient-magnitude rung is already in the exact projected
    /// residual currency used on the face. A curvature-derived rung is not:
    /// the interior judgment must REPLACE it with its own sub-block curvature
    /// bound, and the replacement's rung is what the resulting certificate —
    /// or refusal — must report (#2458/#2559).
    stationarity_bound: StationarityBound,
    /// The run's relative objective tolerance resolved at the certified cost —
    /// the same flat-valley floor the cost-stall guard and the curvature-scaled
    /// widening use. The railed interior judgment applies the identical
    /// Newton-decrement criterion on the interior SUB-BLOCK (the full-Hessian
    /// widening is disabled exactly when a noise-corrupted tail entry makes the
    /// full matrix non-PD).
    objective_tol: f64,
    context: &'a str,
    /// The canonical run's coordinate map, so every decline this certificate
    /// writes names native coordinates ([`native_coordinate`], #2817).
    native_coordinate_order: Option<&'a [usize]>,
}

impl AsymptoteRailInputs<'_> {
    /// Split the railed set into the coordinates the exponential tail law
    /// speaks for and the ones it does not (#2453).
    ///
    /// Both halves stay in the active set: they are equally deleted from the
    /// interior gradient and the interior Hessian sub-block, because that
    /// reasoning is about a bound and not about a quantity. Only the first
    /// half may be *certified* by — or *required* to produce — a tail, since
    /// `OuterThetaLayout::coordinate_is_log_smoothing` is what makes
    /// `ĉ = ∓e^{±ρ}·∂V/∂ρ` a theorem rather than an arithmetic accident.
    ///
    /// The second half needs no tail: its bound is attainable, so the
    /// outward-gradient complementarity the projector already enforces is a
    /// complete first-order certificate there.
    fn tail_law_coordinates(&self) -> (Vec<usize>, Vec<usize>) {
        self.railed
            .iter()
            .copied()
            .partition(|&k| self.layout.coordinate_is_log_smoothing(k))
    }
}

/// Attempt the typed stationary-at-asymptote rail certificate (#2348 Inc 1).
///
/// Returns `Some((interior_projected_grad_norm, rails))` when the interior
/// (non-railed) coordinates are gradient-stationary, the interior Hessian
/// sub-block is PSD, and EVERY railed coordinate is certified on a confirmed
/// exponential tail whose fitted model has reached the rail limit to within the
/// estimand tolerance. Returns `None` (fall through to the generic verdict) on
/// any failure — a non-stationary interior, indefinite interior curvature, or
/// any railed coordinate whose tail is not confirmable. Never errors on a
/// refusal; the only `Err` is a genuinely broken objective that cannot restore
/// its inner state to the certified point after probing.
fn try_certify_asymptote_rail(
    obj: &mut dyn OuterObjective,
    inputs: &AsymptoteRailInputs<'_>,
) -> Result<Result<(f64, StationarityBound, Vec<RailCoordinate>), String>, EstimationError> {
    let rho = inputs.rho;
    let projected_gradient = inputs.projected_gradient;
    let railed = inputs.railed;
    // The interior (non-railed) coordinates must be stationary in their own
    // right: the asymptote certificate speaks only to the railed directions,
    // never rescues a still-descending interior. Judged by the SAME two-stage
    // criterion as the Inc 2c at-point mint: a gradient-magnitude bound may
    // judge the exact KKT-projected residual directly; a curvature-derived
    // bound is reminted from the interior sub-block before it may admit
    // anything. A fit whose remaining interior Newton step would improve the
    // cost by less than the loop's own cost resolution is at its interior
    // optimum, and the residual gradient is the deep-λ instrument noise floor
    // (evaluations beside a saturated rail share the rail's logdet noise).
    let interior_indices = interior_face_indices(projected_gradient, railed);
    let (interior_projected_grad_norm, effective_interior_bound) =
        match certify_interior_stationarity(
            projected_gradient,
            inputs.hessian,
            &interior_indices,
            inputs.stationarity_bound,
            inputs.objective_tol,
        ) {
            Ok(certified) => certified,
            Err(reason) => return Ok(Err(reason)),
        };
    // The interior sub-block (railed coordinates removed) must be admissible
    // curvature for a minimum. A rail-caused indefiniteness in the saturated
    // direction is expected and excluded; genuine interior negative curvature is
    // not, and refuses the certificate.
    // #2676: read from the objective at THIS point, exactly as the generic
    // verdict does. A rail certificate that judged the invariance would decline
    // on the same rounding residual the generic path used to refuse on, and the
    // two paths would disagree about one matrix.
    let criterion_invariance = obj.criterion_invariant_directions(rho);
    if certificate_hessian_is_psd_off_railed_above_gradient_floor(
        inputs.hessian,
        railed,
        projected_gradient,
        criterion_invariance.as_ref(),
    ) != Some(true)
    {
        return Ok(Err("interior Hessian sub-block is not PSD".to_string()));
    }
    let beta_norm = inputs
        .terminal_beta
        .map(|b| b.dot(b).sqrt())
        .filter(|v| v.is_finite())
        .unwrap_or(0.0);
    let estimand_tol = ASYMPTOTE_ESTIMAND_REL_TOL * (1.0 + beta_norm);
    let mut tol = AsymptoteTolerances::exp4_rail_bands(estimand_tol);
    // Real REML tails hold ĉ to ~5e-3 relative, not the exp4 synthetic
    // characterization's 1e-3 (measured on the #2299 fixture during Inc 2c).
    tol.tail_drift_rel = RAIL_TAIL_DRIFT_REL;
    let (lower, upper) = inputs.bounds;

    // #2453: only the log-λ coordinates go to the tail law. A ψ rail stays in
    // the active set above (it is excluded from the interior gradient and the
    // interior sub-block just like any other bound-active coordinate) but is
    // certified by complementarity, not by an asymptote it does not have.
    let (tail_railed, box_railed) = inputs.tail_law_coordinates();
    if tail_railed.is_empty() {
        return Ok(Err(format!(
            "no railed coordinate parameterizes log λ; {} bound-active ψ coordinate(s) {:?} carry \
             no exponential tail to certify",
            box_railed.len(),
            native_coordinates(inputs.native_coordinate_order, &box_railed),
        )));
    }
    if !box_railed.is_empty() {
        log::debug!(
            "[CERTIFICATE] {}: {} bound-active ψ coordinate(s) {:?} held out of the tail law \
             (their box endpoints are attainable, so complementarity certifies them); tail law \
             runs on log-λ coordinate(s) {:?}",
            inputs.context,
            box_railed.len(),
            native_coordinates(inputs.native_coordinate_order, &box_railed),
            native_coordinates(inputs.native_coordinate_order, &tail_railed),
        );
    }

    // #2348 Inc 5 — the ANALYTIC face proof, first resort.
    //
    // Measuring a tail beside the ρ box asks the criterion for derivative
    // information exactly where its logdet pair cancels; and even when it
    // succeeds it speaks for ONE coordinate along ONE ray. The analytic route
    // forms the λ=∞ limit itself — the null-space-restricted fit, and the
    // exact first-order form of the logdet and trace terms there — and proves
    // the whole face against every way of coming off it: by that form's
    // positive definiteness, or, when the released penalty ranges are
    // independent (Σ rank A_j = q, every single-penalty face), by the exact
    // KKT test c_j > τ_j on the linear first-order law, or, when they overlap,
    // by a sound lower bound on that law over the whole release simplex. When
    // the objective cannot form that limit, or the proof does not hold, the
    // measured-tail path below is unchanged.
    match try_certify_face_analytically(obj, inputs, &tail_railed, estimand_tol)? {
        Ok((rails, proof)) => {
            log::debug!(
                "[CERTIFICATE] {}: analytic λ=∞ face proof on {} coordinate(s) via {:?}: \
                 statistic {:.6e} > band {:.3e}, joint pencil ĉ={:.6e}, remaining value gap \
                 {:.3e}, estimand travel {:.3e}",
                inputs.context,
                rails.len(),
                proof.route,
                proof.statistic,
                proof.band,
                proof.joint_tail_constant,
                proof.value_gap,
                proof.estimand_travel,
            );
            return Ok(Ok((
                interior_projected_grad_norm,
                effective_interior_bound,
                rails,
            )));
        }
        Err(reason) => log::debug!(
            "[CERTIFICATE] {}: analytic λ=∞ face proof declined ({reason}); measuring the tail \
             instead",
            inputs.context,
        ),
    }

    let mut rails: Vec<RailCoordinate> = Vec::new();
    let mut decline: Option<String> = None;
    let mut probed_any = false;
    for &k in tail_railed.iter() {
        if k >= rho.len() || k >= lower.len() || k >= upper.len() {
            decline = Some(format!(
                "railed coordinate {} outside the box layout",
                native_coordinate(inputs.native_coordinate_order, k)
            ));
            break;
        }
        // Which rail: the box endpoint the coordinate sits nearest. `Upper`
        // (λ → ∞) probes step ρ downward into the tail; `Lower` (λ → 0) step up.
        let side = if (upper[k] - rho[k]).abs() <= (rho[k] - lower[k]).abs() {
            AsymptoteSide::Upper
        } else {
            AsymptoteSide::Lower
        };
        probed_any = true;
        match build_and_assess_rail_coordinate(
            obj,
            rho,
            k,
            side,
            &tol,
            (lower[k], upper[k]),
            native_coordinate(inputs.native_coordinate_order, k),
        )? {
            Ok(rail) => rails.push(rail),
            Err(reason) => {
                decline = Some(reason);
                break;
            }
        }
    }

    // The probes warm-started the inner solve away from ρ̂; restore it so the
    // shipped fitted state sees the certified point. A failure here is a
    // genuinely broken objective, not a refusal.
    if probed_any {
        obj.eval_cost(rho).map_err(|err| {
            EstimationError::RemlOptimizationFailed(format!(
                "{}: failed to restore the objective to the certified point after \
                 asymptote-rail probing: {err}",
                inputs.context
            ))
        })?;
    }

    if let Some(reason) = decline {
        return Ok(Err(reason));
    }
    if rails.is_empty() {
        return Ok(Err(
            "no railed coordinate produced a certifiable tail".to_string()
        ));
    }
    Ok(Ok((
        interior_projected_grad_norm,
        effective_interior_bound,
        rails,
    )))
}

/// Prove the rail face analytically (#2348 Inc 5): ask the objective for the
/// exact λ→∞ limit, test the first-order form, and mint the rails from it.
///
/// The proof spends no criterion evaluation. Whether each objective's closed
/// form expands the criterion it actually minimizes is a property of that
/// code, pinned by the value- and gradient-domain tests beside
/// `RemlState::rail_face_limit`. It is not re-measured here by differencing
/// two criterion values against a slack.
///
/// Returns the minted rail coordinates plus the proof, or a human-readable
/// decline that the caller logs before falling back to the measured tail.
fn try_certify_face_analytically(
    obj: &mut dyn OuterObjective,
    inputs: &AsymptoteRailInputs<'_>,
    tail_railed: &[usize],
    estimand_tol: f64,
) -> Result<Result<(Vec<RailCoordinate>, RailFaceProof), String>, EstimationError> {
    let rho = inputs.rho;
    let (lower, upper) = inputs.bounds;
    // The analytic limit is the INFINITE-smoothing face. A coordinate railed at
    // the zero-smoothing bound is the opposite limit (the penalty leaves the
    // model rather than pinning it) and belongs to the measured-tail path.
    for &k in tail_railed.iter() {
        if k >= rho.len() || k >= lower.len() || k >= upper.len() {
            return Ok(Err("railed coordinate outside the box layout".to_string()));
        }
        if (upper[k] - rho[k]).abs() > (rho[k] - lower[k]).abs() {
            return Ok(Err(format!(
                "coordinate {} rails at the zero-smoothing bound; the analytic face limit \
                 covers λ→∞ only",
                native_coordinate(inputs.native_coordinate_order, k)
            )));
        }
    }
    let limit = match obj.rail_face_limit(rho, tail_railed)? {
        RailFaceLimitOutcome::Available(limit) => *limit,
        // The decline is typed, and the distinction is worth carrying into the
        // refusal: "outside this closed form" invites a different one, while
        // "the face is unavailable" is a statement about the face.
        RailFaceLimitOutcome::OutsideClosedForm { reason } => {
            return Ok(Err(format!("outside the analytic closed form: {reason}")));
        }
        RailFaceLimitOutcome::FaceUnavailable { reason } => {
            return Ok(Err(format!("the λ=∞ face is unavailable: {reason}")));
        }
    };
    let proof = match certify_rail_face(&limit) {
        RailFaceVerdict::Certified(proof) => proof,
        RailFaceVerdict::Refused { reason } => return Ok(Err(reason)),
    };
    // The face being optimal does not by itself mean the SHIPPED fit is the
    // limit fit; the estimand gate is the same one the measured path applies,
    // now answered by the exact first-order coefficient offset rather than a
    // geometric extrapolation of observed steps.
    if !(proof.estimand_travel <= estimand_tol) {
        return Ok(Err(format!(
            "λ=∞ face proven, but the shipped fit has not reached it: coefficient travel \
             {:.3e} > estimand tolerance {estimand_tol:.3e}",
            proof.estimand_travel
        )));
    }
    let rails: Vec<RailCoordinate> = limit
        .face
        .iter()
        .zip(limit.face_rho.iter())
        .zip(proof.tail_constants.iter())
        .map(|((&index, &rho_k), &tail_constant)| RailCoordinate {
            index,
            side: AsymptoteSide::Upper,
            tail_constant,
            // On the tail law the remaining value gap of one coordinate is
            // exactly `c_k·e^{−ρ_k}`; a coordinate the rest of the face has
            // already pinned reports `c_k = 0` because releasing it does not
            // move the criterion at all.
            value_gap: tail_constant * (-rho_k).exp(),
            estimand_travel_bound: proof.estimand_travel,
            // The face was PROVEN, so the standard it cleared is the route's
            // own rounding band on the assembled form — not a finite-difference
            // floor, and not a quantity comparable with one.
            evidence: RailTailEvidence::AnalyticFaceProof {
                route: proof.route,
                statistic: proof.statistic,
                band: proof.band,
            },
        })
        .collect();
    Ok(Ok((rails, proof)))
}

/// The coordinates a stationarity residual must still account for.
///
/// A coordinate leaves the interior only when the box has genuinely pinned it:
/// railed AND the projection zeroed its gradient, i.e. its entire pull was the
/// infeasible KKT multiplier. [`project_gradient_vector`] keeps a near-bound
/// coordinate's INWARD component on purpose — that component is feasible
/// descent — so dropping the row for *every* railed coordinate discards exactly
/// what the projector was written to preserve.
///
/// Deleting agrees with this only when every railed gradient points strictly
/// outward. `matern_nu_sweep_uniform_quality_on_sin1` is the counterexample
/// (#2471): coordinate 3 was reported railed while `1.2018e1` of its projected
/// gradient survived — 66.7x the stationarity bound, and 99.99% of the reported
/// `|Pg|`. With it deleted the interior norm read `1.986e-1`, i.e. 1.10x the
/// bound, which reads as "essentially converged" at a point still carrying that
/// much feasible descent. The certificate refused anyway, so the number misled
/// the reader rather than the verdict — but the ledger built on it classified a
/// genuine non-convergence as a railed-coordinate accounting artifact.
///
/// Since the projection either keeps a component unchanged or zeroes it, the
/// norm over these indices is exactly `‖Pg‖`. Where this differs from deleting
/// it includes MORE residual coordinates. Curvature evidence is deliberately
/// recomputed on that exact set: unlike the residual norm, a Newton decrement
/// is not transferable across Hessian subspaces or their regularization scales.
pub(crate) fn interior_face_indices(
    projected_gradient: &Array1<f64>,
    railed: &[usize],
) -> Vec<usize> {
    (0..projected_gradient.len())
        .filter(|k| !railed.contains(k) || projected_gradient[*k] != 0.0)
        .collect()
}

/// Interior stationarity judgment shared by the Inc 1 railed mint and the
/// Inc 2c at-point mint (#2348/#2559).
///
/// The supplied indices make `interior_grad_norm` the exact KKT-projected
/// residual of the face being judged: normally [`interior_face_indices`]
/// itself, and with already-proven tail coordinates removed on the Inc 2c
/// route. A gradient-magnitude ladder rung can therefore judge it directly.
/// [`StationarityBoundSource::CurvatureResolvability`] is different: its value
/// contains the caller's Hessian and Newton decrement, so reusing it here would
/// make the ordinary rail path's early comparison reduce to the caller's own
/// `Δpred <= objective_tol` test. It must instead be derived from `sub_h` and
/// `sub_g`. Exact zero is stationary without a curvature scale.
///
/// The face-local curvature path certifies only when the active-face Newton
/// step would improve the objective by at most `objective_tol` — the loop's
/// own cost resolution. Returns the bound that actually admitted the norm;
/// `Err` carries both caller and face-local evidence when descent remains.
pub(crate) fn certify_interior_stationarity(
    gradient: &Array1<f64>,
    hessian: &Array2<f64>,
    interior_indices: &[usize],
    stationarity_bound: StationarityBound,
    objective_tol: f64,
) -> Result<(f64, StationarityBound), String> {
    let interior_grad_norm = interior_indices
        .iter()
        .map(|&k| gradient[k] * gradient[k])
        .sum::<f64>()
        .sqrt();
    if interior_grad_norm <= stationarity_bound.value()
        && (interior_grad_norm == 0.0
            || !stationarity_bound.requires_face_local_derivation())
    {
        return Ok((interior_grad_norm, stationarity_bound));
    }
    let m = interior_indices.len();
    let mut sub_h = Array2::<f64>::zeros((m, m));
    let mut sub_g = Array1::<f64>::zeros(m);
    for (i, &ri) in interior_indices.iter().enumerate() {
        sub_g[i] = gradient[ri];
        for (j, &rj) in interior_indices.iter().enumerate() {
            sub_h[[i, j]] = hessian[[ri, rj]];
        }
    }
    match newton_predicted_decrease(&sub_h, &sub_g) {
        Some(predicted_decrease) if predicted_decrease.is_finite() && predicted_decrease > 0.0 => {
            if predicted_decrease <= objective_tol {
                let curvature_grad_bound =
                    interior_grad_norm * (objective_tol / predicted_decrease).sqrt();
                if curvature_grad_bound.is_finite() && curvature_grad_bound >= interior_grad_norm {
                    // The returned bound is no longer the caller's: it is the
                    // interior sub-block's own `|Pg_int|·√(τ/Δpred)`. Carrying
                    // the caller's rung with it would report a widened bound
                    // under the standard that did NOT set it (#2458).
                    return Ok((
                        interior_grad_norm,
                        StationarityBound::from_ladder(
                            curvature_grad_bound,
                            StationarityBoundSource::CurvatureResolvability,
                        ),
                    ));
                }
            }
            Err(format!(
                "interior not stationary: active-face |Pg|={interior_grad_norm:.3e}, \
                 caller bound {:.3e} from {}; active-face Newton decrement \
                 {predicted_decrease:.3e} > cost resolution {objective_tol:.3e}",
                stationarity_bound.value(),
                stationarity_bound.rung().label,
            ))
        }
        _ => Err(format!(
            "interior not stationary: active-face |Pg|={interior_grad_norm:.3e}, \
             caller bound {:.3e} from {}; the active-face Hessian and residual \
             yield no positive finite PD Newton decrement",
            stationarity_bound.value(),
            stationarity_bound.rung().label,
        )),
    }
}

/// Relative drift band for a measured rail-tail window, wider than the exp4
/// characterization band (1e-3). The window's evidentiary strength comes from
/// the EXTRAPOLATED-GAP margin, not the band tightness: a 1–2% spread in `ĉ`
/// across the clean run moves the extrapolated remaining gradient `ĉ·e^{∓ρ}`
/// by the same 1–2%, immaterial against the orders-of-magnitude margin the
/// rail decision demands — while the true tail on a REAL fixture still carries
/// visible sub-percent curvature contamination at probe depth (measured on
/// #2299: ĉ ∈ {6544, 6565, 6574} over three e-folds, drift 4.6e-3, against a
/// wildly swinging noise region above).
const RAIL_TAIL_DRIFT_REL: f64 = 1.0e-2;

/// Reconstruct one railed coordinate's exponential tail by probing the analytic
/// gradient back from the rail at coarse and, when needed, local resolution;
/// locate the longest finite-difference-clean run (rejecting the noise floor
/// adjacent to the rail); and assess it against the tail law (#2348 Inc 1 /
/// #2337 Thm 2.1). Returns the certified [`RailCoordinate`] or `None` if no
/// confirmable tail is found.
fn build_and_assess_rail_coordinate(
    obj: &mut dyn OuterObjective,
    rho: &Array1<f64>,
    coord: usize,
    side: AsymptoteSide,
    tol: &AsymptoteTolerances,
    domain: (f64, f64),
    native: usize,
) -> Result<Result<RailCoordinate, String>, EstimationError> {
    let window = match probe_tail_window(obj, rho, coord, side, tol, domain)? {
        (Some(window), _) => window,
        (None, rows) => {
            return Ok(Err(format!(
                "k={native}: no finite-difference-clean tail window; probes {rows}"
            )));
        }
    };
    match assess_coordinate(&window, tol) {
        AsymptoteVerdict::CertifiedAtAsymptote {
            side,
            tail_constant,
            value_gap,
            estimand_travel_bound,
        } => Ok(Ok(RailCoordinate {
            index: coord,
            side,
            tail_constant,
            value_gap,
            estimand_travel_bound,
            evidence: RailTailEvidence::ProbedTail {
                noise_floor: tol.tail_noise_floor,
                drift_band: tol.tail_drift_rel,
            },
        })),
        other => Ok(Err(format!("k={native}: tail verdict {other:?}"))),
    }
}

/// Detect a WRONG-RAIL coordinate (#2392): one sitting AT its ρ box bound whose
/// clean-band probes prove the objective strictly DECREASES as the coordinate
/// moves INWARD — the outer search drove it to the wrong bound. Returns the
/// interior ρ to reseed the coordinate at (the deepest drift-clean probe, where
/// `|g|` is largest and the descent is most informative) when the proof holds,
/// else `None`.
///
/// # Proof condition (evidence-gated; cannot launder a genuine λ→∞ / λ→0 optimum)
///
/// Probe up to [`ASYMPTOTE_PROBE_COUNT`] e-folds inward and let the FIRST
/// contiguous clean, drift-stable run of at least `MIN_TAIL_SAMPLES` decide
/// the local rail:
/// 1. above the gradient interior floor, `|g| > interior_grad_tol` (so a probe
///    whose gradient has decayed into finite-difference cancellation next to the
///    rail is excluded rather than read as a settled tail);
/// 2. above the pencil-constant noise floor, `|ĉ| > tail_noise_floor`, where
///    `ĉ = side.tail_constant(ρ, g)` uses the coordinate's ACTUAL rail side;
/// 3. drift-band-clean in `ĉ` within `tail_drift_rel` (the same constant-pencil
///    band the genuine tail uses — `run_drift_within_band` keys on `|mean|`, so a
///    uniformly-negative run is judged on its magnitude).
///
/// A first clean run with `ĉ < 0` proves descent AWAY from the bound and returns
/// its deepest point. A first clean run with `ĉ > 0` proves descent TOWARD the
/// bound and refuses the pull-back immediately. Deciding on the first clean run
/// is load-bearing: the question is the LOCAL orientation of the objective at
/// this rail. Continuing another fifteen expensive objective evaluations after
/// that proof could discover a remote sign reversal in the interior, but must
/// not use it to relabel a locally genuine bound as a wrong rail.
fn detect_wrong_rail_pullback(
    obj: &mut dyn OuterObjective,
    rho: &Array1<f64>,
    coord: usize,
    side: AsymptoteSide,
    tol: &AsymptoteTolerances,
    domain: (f64, f64),
) -> Result<Option<f64>, EstimationError> {
    const PROBE_DELTA: f64 = 1.0;
    const PROBE_DOMAIN_MARGIN: f64 = 1.0e-6;
    // Upper rail (ρ → +∞): step ρ DOWN into the interior. Lower rail: step UP.
    let sign = match side {
        AsymptoteSide::Upper => -1.0,
        AsymptoteSide::Lower => 1.0,
    };
    // The closest finite-difference-clean constant-pencil run is the local rail
    // evidence. Noise rows reset the run; a sign change starts a new candidate.
    let mut run_sign = 0_i8;
    let mut run_constants: Vec<f64> = Vec::with_capacity(MIN_TAIL_SAMPLES);
    for j in 1..=ASYMPTOTE_PROBE_COUNT {
        let stepped = rho[coord] + sign * (j as f64) * PROBE_DELTA;
        if stepped <= domain.0 + PROBE_DOMAIN_MARGIN || stepped >= domain.1 - PROBE_DOMAIN_MARGIN {
            break;
        }
        let mut probe = rho.clone();
        probe[coord] = stepped;
        let eval = match obj.eval_with_order(&probe, OuterEvalOrder::ValueAndGradient) {
            Ok(eval) => eval,
            Err(_) => break,
        };
        if !eval.cost.is_finite()
            || coord >= eval.gradient.len()
            || !eval.gradient[coord].is_finite()
        {
            break;
        }
        let gradient = eval.gradient[coord];
        let constant = side.tail_constant(stepped, gradient);
        let clean = constant.is_finite()
            && constant.abs() > tol.tail_noise_floor
            && gradient.abs() > tol.interior_grad_tol;
        if !clean {
            run_sign = 0;
            run_constants.clear();
            continue;
        }
        let constant_sign = if constant < 0.0 { -1 } else { 1 };
        if constant_sign != run_sign {
            run_sign = constant_sign;
            run_constants.clear();
        }
        run_constants.push(constant);
        if run_constants.len() > MIN_TAIL_SAMPLES {
            run_constants.remove(0);
        }
        if run_constants.len() >= MIN_TAIL_SAMPLES
            && run_drift_within_band(&run_constants, tol.tail_drift_rel)
        {
            return if run_sign < 0 {
                Ok(Some(stepped))
            } else {
                Ok(None)
            };
        }
    }
    Ok(None)
}

/// Probe one coordinate's tail toward the interior (the shared probing engine
/// of [`build_and_assess_rail_coordinate`]).
/// The one-e-fold ladder runs first; if it finds no clean run, a short
/// half-e-fold ladder resolves a narrower local band without mixing step sizes
/// in one estimand window. Returns the longest finite-difference-clean
/// constant-`ĉ` run (newest sample nearest `rho[coord]`), or `None` when neither
/// resolution contains at least `MIN_TAIL_SAMPLES` clean rows. The second
/// element dumps `(ρ, ∂V/∂ρ, ĉ)` evidence for every attempted resolution.
fn probe_tail_window(
    obj: &mut dyn OuterObjective,
    rho: &Array1<f64>,
    coord: usize,
    side: AsymptoteSide,
    tol: &AsymptoteTolerances,
    domain: (f64, f64),
) -> Result<(Option<AsymptoteWindow>, String), EstimationError> {
    let (coarse_window, coarse_rows) = probe_tail_window_at_resolution(
        obj,
        rho,
        coord,
        side,
        tol,
        domain,
        (1.0, ASYMPTOTE_PROBE_COUNT),
    )?;
    if coarse_window.is_some() {
        return Ok((coarse_window, coarse_rows));
    }

    // The coarse ladder is deliberately retained as the first pass: railed
    // dense REML fits can have several e-folds of cancellation noise beside
    // the box followed by a clean band far inside it. The local pass addresses
    // the complementary geometry exposed by #2358, where a modest finite box
    // contains a narrow but regular tail and unit steps skip over it.
    let (local_window, local_rows) = probe_tail_window_at_resolution(
        obj,
        rho,
        coord,
        side,
        tol,
        domain,
        (
            ASYMPTOTE_LOCAL_PROBE_DELTA,
            ASYMPTOTE_LOCAL_PROBE_COUNT,
        ),
    )?;
    Ok((
        local_window,
        format!("coarse[{coarse_rows}] local[{local_rows}]"),
    ))
}

/// Probe a single equally-spaced resolution of one coordinate's tail.
///
/// Keeping each returned window at one resolution is essential for
/// `assess_coordinate`: its coefficient-travel bound estimates a geometric
/// ratio from consecutive steps, which is only meaningful when their `Δρ`
/// values are identical.
fn probe_tail_window_at_resolution(
    obj: &mut dyn OuterObjective,
    rho: &Array1<f64>,
    coord: usize,
    side: AsymptoteSide,
    tol: &AsymptoteTolerances,
    domain: (f64, f64),
    resolution: (f64, usize),
) -> Result<(Option<AsymptoteWindow>, String), EstimationError> {
    let (probe_delta, probe_count) = resolution;
    // Strictly-inside guard for probes against the probed coordinate's own box
    // interval (#2388). The ρ-gradient assembly freezes any coordinate at (or
    // within 1e-8 of) its recorded upper bound to the #197 KKT projection — a
    // literal 0.0 — so a probe at or past a box bound samples the frozen-axis
    // convention, not the criterion's tail: a fabricated hard-zero tail that
    // the drift band can never confirm. Out-of-box points are outside the
    // λ-selection domain altogether; they are not evidence for or against a
    // tail, so the ladder stops at the last strictly-in-domain probe.
    const PROBE_DOMAIN_MARGIN: f64 = 1.0e-6;
    // Upper rail (ρ → +∞): step ρ DOWN into the tail. Lower rail: step UP.
    let sign = match side {
        AsymptoteSide::Upper => -1.0,
        AsymptoteSide::Lower => 1.0,
    };
    // rows[r] corresponds to probe j=r+1: r=0 is the point CLOSEST to the rail,
    // increasing r steps further into the interior (larger |grad|).
    let mut rows: Vec<(f64, f64, Option<Array1<f64>>)> = Vec::new();
    for j in 1..=probe_count {
        let stepped = rho[coord] + sign * (j as f64) * probe_delta;
        if stepped <= domain.0 + PROBE_DOMAIN_MARGIN || stepped >= domain.1 - PROBE_DOMAIN_MARGIN {
            break;
        }
        let mut probe = rho.clone();
        probe[coord] = stepped;
        let eval = match obj.eval_with_order(&probe, OuterEvalOrder::ValueAndGradient) {
            Ok(eval) => eval,
            // A failed probe is not evidence against a tail; stop probing and
            // assess whatever clean run the earlier probes established.
            Err(_) => break,
        };
        if !eval.cost.is_finite()
            || coord >= eval.gradient.len()
            || !eval.gradient[coord].is_finite()
        {
            break;
        }
        rows.push((probe[coord], eval.gradient[coord], eval.inner_beta_hint));
    }
    let rows_summary = rows
        .iter()
        .map(|(r, g, _)| {
            format!(
                "(ρ={r:.2}, g={g:.3e}, ĉ={:.3e})",
                side.tail_constant(*r, *g)
            )
        })
        .collect::<Vec<_>>()
        .join(" ");
    if rows.len() < MIN_TAIL_SAMPLES {
        return Ok((None, rows_summary));
    }

    // Per-row pencil constant ĉ and the element-clean predicate: ĉ above the
    // noise floor AND the gradient above the interior floor (so a row adjacent to
    // the rail, whose gradient has decayed into finite-difference cancellation, is
    // excluded rather than mistaken for a settled tail).
    let constants: Vec<f64> = rows
        .iter()
        .map(|(r, g, _)| side.tail_constant(*r, *g))
        .collect();
    let element_clean: Vec<bool> = rows
        .iter()
        .zip(&constants)
        .map(|((_, g, _), c)| {
            c.is_finite() && *c > tol.tail_noise_floor && g.abs() > tol.interior_grad_tol
        })
        .collect();

    // Longest contiguous run that is element-clean AND holds ĉ within the drift
    // band; ties broken toward the rail (smallest start) for the most settled
    // estimand.
    let mut best: Option<(usize, usize)> = None;
    for a in 0..rows.len() {
        if !element_clean[a] {
            continue;
        }
        for b in a..rows.len() {
            if !element_clean[b] {
                break;
            }
            if b - a + 1 < MIN_TAIL_SAMPLES {
                continue;
            }
            if !run_drift_within_band(&constants[a..=b], tol.tail_drift_rel) {
                continue;
            }
            let len = b - a + 1;
            match best {
                Some((ba, bb)) if bb - ba + 1 >= len => {}
                _ => best = Some((a, b)),
            }
        }
    }
    let (a, b) = match best {
        Some(run) => run,
        None => return Ok((None, rows_summary)),
    };

    // Build the window oldest → newest: newest (window `latest`) is the row
    // CLOSEST to the rail (r=a). A sample's coefficient move is ‖β(r) − β(r+1)‖,
    // the step from the next-farther retained row toward the rail.
    let mut window = AsymptoteWindow::with_capacity(b - a + 1);
    for r in (a..=b).rev() {
        let (rho_r, grad_r, beta_r) = &rows[r];
        let coef_step_norm = match (beta_r, rows.get(r + 1).map(|row| &row.2)) {
            (Some(cur), Some(Some(farther))) if cur.len() == farther.len() => {
                (cur - farther).iter().map(|v| v * v).sum::<f64>().sqrt()
            }
            _ => 0.0,
        };
        window.push(AsymptoteSample {
            rho: *rho_r,
            grad: *grad_r,
            coef_step_norm,
        });
    }

    Ok((Some(window), rows_summary))
}

/// Whether a run of pencil constants holds constant within the relative drift
/// band `(max − min)/|mean| ≤ band` (deterministic, ordered).
fn run_drift_within_band(constants: &[f64], band: f64) -> bool {
    if constants.len() < MIN_TAIL_SAMPLES {
        return false;
    }
    let mut sum = 0.0_f64;
    let mut lo = f64::INFINITY;
    let mut hi = f64::NEG_INFINITY;
    for &c in constants {
        if !c.is_finite() {
            return false;
        }
        sum += c;
        lo = lo.min(c);
        hi = hi.max(c);
    }
    let mean = sum / constants.len() as f64;
    if !(mean.abs() > 0.0) {
        return false;
    }
    (hi - lo) / mean.abs() <= band
}

/// Why the operator trust-region outer loop stopped.
///
/// The inhabitants of this enum are exactly the image of
/// `bridges::stop_reason_from`, which is its ONE producer: a value only ever
/// reaches a `RhoOptimizerResult` by mapping an `opt::TerminationReason`.
/// Adding a variant that map cannot emit adds a label, not a state — that is
/// how `RoutingMismatch` ("family returned a non-operator Hessian mid-flight")
/// came to sit here unreachable, and it was deleted for it (#2670).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum OperatorTrustRegionStopReason {
    Converged,
    RejectFloor,
    IterationBudget,
    /// The objective stopped changing on a criterion-flat surface. The
    /// in-loop guard may already have certified the score-relative residual or
    /// may have returned a non-stationary floor; either way the final analytic
    /// certificate needs this provenance to reproduce the guard's derived
    /// stationarity band exactly.
    CostStallFlatValley,
    /// The solver stopped because something failed, not because a test it
    /// stands behind was satisfied: the line search gave up, an objective
    /// evaluation failed, or the arithmetic went non-finite.
    ///
    /// These used to map to [`Self::Converged`], on the stated premise that
    /// they are "a hard failure the caller sees through the `Err` arm". That
    /// premise is false for the line-search case on the path it actually
    /// takes: `run_plan` turns a `BfgsError::LineSearchFailed` whose last
    /// iterate is finite into `Ok(non-converged)`, because that iterate is a
    /// usable checkpoint. The caller therefore sees no `Err`, and the coarse
    /// reason it does see said `Converged` — which is how a binomial/logit
    /// REML fit that never accepted a single step came to report
    /// `stop_reason=Converged after 1 outer iteration(s)` (#2614).
    ///
    /// Reporting-only today: no consumer branches on `Converged`, so this
    /// splits a label without moving a decision. Kept separate from
    /// [`Self::IterationBudget`] because "ran out of budget" and "could not
    /// take a step" call for different repairs.
    SolverFailure,
    /// The fixed-point map proposed a step below its step-norm threshold.
    ///
    /// A small step is not stationarity: on an EFS penalty coordinate the update
    /// is a ratio of traces, not the gradient the certificate bounds. This used
    /// to map to [`Self::Converged`]. `run_fixed_point_outer_solver` judges the
    /// stop with the screening certificate instead, so this label records which
    /// test fired and never stands for convergence (#2817).
    StepNormStall,
}

/// Roundoff-relative scale below which a drop in the certified objective between
/// two refusals of the certify-last reseed loop is numerical noise rather than
/// exploited descent (#2374, #2817). Both values are the certificate's own
/// evaluations, so a re-run that found no descent certifies a value no lower,
/// to roundoff, than the previous refusal's, while exploited descent lowers it
/// by orders of magnitude more than this scale. The gate MUST anchor on
/// roundoff, not the much larger cost-stall relative floor: a flat valley
/// crawls out in per-reseed steps far smaller than `rel_cost·(1 + |cost|)` (the
/// transformation-survival LAML moves ~4e-5 relative per reseed), and gating on
/// that coarser floor stops the crawl after a single hop and refuses a
/// well-posed fit.
const CERTIFY_RESUME_PROGRESS_REL: f64 = 32.0 * f64::EPSILON;

/// The kind of strategy change a refused mint certificate published.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum CertifyReseedKind {
    /// A negative-curvature escape stepped strictly below a strict saddle
    /// (#2357/#2155, #2612).
    SaddleEscape,
    /// A wrong-rail pull-back to the coordinate's clean-band interior (#2392).
    WrongRail,
    /// An active-set reduction: railed coordinates frozen so the interior
    /// polishes in the reduced box (#2392).
    ActiveSet,
}

/// A strategy change a refused mint certificate published: where the re-run
/// starts, and for an active-set reduction the reduced search box.
struct CertifyReseed {
    rho: Array1<f64>,
    search_bounds_override: Option<(Array1<f64>, Array1<f64>)>,
    kind: CertifyReseedKind,
}

/// Take the reseed a refused certificate published, in precedence order: a
/// saddle escape, then a wrong-rail pull-back, then an active-set reduction.
/// Every lower-precedence reseed is taken and dropped, so no stale reseed leaks
/// into a later iteration.
///
/// A published reseed is first-order evidence of where the search should go,
/// so it is taken whether or not the solver claimed convergence: for
/// exact-Hessian link-wiggle families the terminal certificate, not the in-loop
/// gate, is where stationarity is first reached. `None` means the certificate
/// published no strategy change, and the refusal stands: re-running the same
/// search from the refused checkpoint only continued it with more iterations
/// (SPEC rules 21 and 23, #2817).
fn take_certify_reseed(result: &mut OuterResult) -> Option<CertifyReseed> {
    let saddle_escape = result.saddle_escape_reseed.take();
    let wrong_rail = result.wrong_rail_reseed.take();
    let active_set = result.active_set_reseed.take();
    if let Some(reseed) = saddle_escape {
        return Some(CertifyReseed {
            rho: reseed,
            search_bounds_override: None,
            kind: CertifyReseedKind::SaddleEscape,
        });
    }
    if let Some(reseed) = wrong_rail {
        return Some(CertifyReseed {
            rho: reseed,
            search_bounds_override: None,
            kind: CertifyReseedKind::WrongRail,
        });
    }
    active_set.map(|reseed| CertifyReseed {
        rho: reseed.rho,
        search_bounds_override: Some(reseed.bounds),
        kind: CertifyReseedKind::ActiveSet,
    })
}

/// The kind of a dominated-plateau refusal at the refused checkpoint `result`, whose
/// certificate published `untaken_reseed` and could not take it (#2953).
///
/// The checkpoint is a strict saddle whose escape could not be taken when that reseed is a
/// saddle escape, or when its certificate measured first-order stationarity on inadmissible
/// curvature, so that the escape declined to publish one.
fn dominance_refusal_kind(
    result: &OuterResult,
    untaken_reseed: Option<CertifyReseedKind>,
) -> DominanceRefusalKind {
    use crate::model_types::CurvatureAdmissibility;
    // Every reseed kind and every curvature verdict is named, so an outcome added
    // later cannot fall into a kind by default.
    let untaken_escape = match untaken_reseed {
        Some(CertifyReseedKind::SaddleEscape) => true,
        Some(CertifyReseedKind::WrongRail | CertifyReseedKind::ActiveSet) | None => false,
    };
    let measured_saddle = result.criterion_certificate.as_ref().is_some_and(|certificate| {
        certificate.is_stationary()
            && match certificate.curvature_verdict() {
                CurvatureAdmissibility::Inadmissible { .. } => true,
                CurvatureAdmissibility::Admissible
                | CurvatureAdmissibility::Unevaluated { .. }
                | CurvatureAdmissibility::CriterionContradicted => false,
            }
    });
    if untaken_escape || measured_saddle {
        DominanceRefusalKind::IncumbentUnescapableSaddle
    } else {
        DominanceRefusalKind::DominanceUnresolved
    }
}

/// Attach a certify reseed's failure to run to the refusal it leaves standing (#2953).
fn with_reseed_failure(
    context: &str,
    refusal: EstimationError,
    kind: CertifyReseedKind,
    reseed_error: &EstimationError,
) -> EstimationError {
    match refusal {
        EstimationError::RemlDidNotConverge {
            context: refusal_context,
            reason,
            iterations,
            final_value,
            projected_grad_norm,
            stationarity_standard,
            rho_checkpoint,
        } => EstimationError::RemlDidNotConverge {
            context: refusal_context,
            reason: format!(
                "{reason}; the certificate's {kind:?} reseed could not run ({reseed_error})"
            ),
            iterations,
            final_value,
            projected_grad_norm,
            stationarity_standard,
            rho_checkpoint,
        },
        other => {
            log::debug!(
                "[OUTER] {context}: the certificate's {kind:?} reseed could not run \
                 ({reseed_error}); the refusal from the point it started at stands: {other}"
            );
            other
        }
    }
}

/// What `run_outer` returns when its terminal certificate refuses `result` and nothing
/// continues it (#2953). A result carrying a declined certified optimum refuses with
/// [`EstimationError::DominatedCertifiedPlateau`], which reports that optimum beside
/// the checkpoint that beat it. Any other result returns `refusal` as it is.
fn dominated_plateau_refusal(
    context: &str,
    result: &OuterResult,
    kind: DominanceRefusalKind,
    refusal: EstimationError,
) -> EstimationError {
    let Some(plateau) = result.dominated_plateau.as_ref() else {
        return refusal;
    };
    let incumbent_projected_grad_norm = match &refusal {
        EstimationError::RemlDidNotConverge {
            projected_grad_norm,
            ..
        } => *projected_grad_norm,
        _ => None,
    };
    EstimationError::DominatedCertifiedPlateau {
        context: context.to_string(),
        kind,
        plateau_rho: plateau.plateau_rho.to_vec(),
        plateau_value: plateau.plateau_value,
        incumbent_rho: result.rho.to_vec(),
        incumbent_value: result.final_value,
        incumbent_projected_grad_norm,
        gap: plateau.gap,
        band: plateau.band,
        continuation: plateau.continuation.to_string(),
        terminal_refusal: Box::new(refusal),
    }
}

/// Run the outer smoothing-parameter optimization.
///
/// This is the single entry point that replaces the scattered optimizer wiring
/// across estimate.rs, joint.rs, and custom_family.rs. It:
///
/// 1. Queries and canonicalizes the objective's capability declaration.
/// 2. Calls `plan()` to select solver + hessian source.
/// 3. Logs the plan and the analytic derivative capabilities it will consume.
/// 4. Generates seed candidates.
/// 5. Runs the chosen solver on candidates in heuristic order up to budget.
/// 6. If the configured fallback policy allows it, re-plans with degraded
///    capabilities chosen centrally inside outer_strategy and retries.
/// 7. Returns the best result (including which plan was actually used).
///
/// Do not wrap `run_outer` calls in try/catch with ad-hoc solver recovery.
/// Callers should declare only the primary capability and, at most, whether
/// automatic fallback is enabled at all.
pub(crate) fn run_outer(
    obj: &mut dyn OuterObjective,
    config: &OuterConfig,
    context: &str,
) -> Result<OuterResult, EstimationError> {
    // Permutation-invariant outer search (#1538/#1539). When the caller has
    // supplied per-coordinate structural keys that induce a non-identity
    // canonical order, run the ENTIRE outer pipeline (seeding, multistart,
    // optimization, and the #934 certificate) in that
    // canonical layout against a permuting wrapper, then map the result back to
    // the native layout. Seeding/tie-breaking then see byte-identical
    // coordinates for every term order, so both orders select the same λ̂.
    if let Some(keys) = config.rho_canonical_keys.as_ref()
        && let Some(perm) = canonical_permutation(keys)
    {
        let canonical_config = canonicalize_outer_config(config, &perm);
        let mut canonical_obj = CanonicalizedObjective::new(obj, perm.clone());
        // The runner's trail below prints ρ in canonical order; this line is what
        // lets a reader map it back. Only the returned result and refusal are
        // mapped to native order, both here.
        log::debug!(
            "[OUTER] {context}: searching in canonical coordinate order; canonical slot c \
             holds native coordinate perm[c], perm={perm:?}"
        );
        let result = run_outer(&mut canonical_obj, &canonical_config, context)
            .map_err(|error| outer_error_to_native(error, &perm))?;
        return Ok(outer_result_to_native(result, &perm));
    }
    let mut result = run_outer_uncertified(obj, config, context)?;
    if obj.begin_exact_polish() {
        // A sampled outer-derivative pilot is an optimization stage, never a
        // certifiable objective. Continue from its best checkpoint on the
        // family's exact full-data measure before the mandatory analytic
        // certificate. This transition is unconditional whenever the family
        // reports that a sample actually ran, so convergence before a nominal
        // phase budget cannot strand the optimizer on the stochastic surface
        // (#979: matrix-free TR stopped after 6 evaluations while the family
        // waited for a 12-evaluation counter).
        let pilot_iterations = result.iterations;
        let mut exact_config = config.clone();
        exact_config.initial_rho = Some(result.rho.clone());
        exact_config.heuristic_log_lambdas = None;
        // The pilot's checkpoint is the lowest state evaluated so far, and the polish starts
        // at it.
        exact_config.carried_checkpoint = None;
        exact_config.operator_initial_trust_radius = result.operator_trust_radius;
        exact_config.warm_start_outer_hessian = result.final_hessian.clone();
        log::debug!(
            "[OUTER] {context}: sampled derivative pilot completed after {} iteration(s); \
             continuing from its checkpoint on the exact full-data measure",
            pilot_iterations,
        );
        let mut polished = run_outer_uncertified(obj, &exact_config, context)?;
        polished.iterations = polished.iterations.saturating_add(pilot_iterations);
        polished.dominated_plateau = lowest_dominated_plateau(
            result.dominated_plateau.take(),
            polished.dominated_plateau.take(),
        );
        result = polished;
    }
    // Mandatory analytic optimality certificate (#934): once at the selected
    // point, outside every hot loop, for every solver path and every iteration
    // budget. Missing or failed evidence is typed non-convergence; there is no
    // max-iteration or logging-level bypass.
    //
    // #2273 STALE-TOLERANCE DESYNC RETRY.
    //
    // HISTORY, because the mechanism this was written for no longer exists.
    // The solver's in-loop threshold used to be resolved ONCE from the SEED's
    // cost scale (`rel_cost·(1+|seed_cost|)`) while this certificate re-derived
    // the same formula at the terminal point's own, often far smaller, cost. On
    // a perfectly-separated binomial the score plunges between the oversmoothed
    // heuristic seed and the first accepted step, so the solver declared
    // victory against a bound orders of magnitude looser than the one that then
    // refused it here (measured: |g|=8.1e-1 accepted in-loop vs bound 8.3e-3 at
    // certification, "NOT STATIONARY after 1 outer iteration"; the pass/fail
    // pattern was non-monotone in n because it tracked the seed-to-terminus
    // cost ratio, not identifiability). The retry removed that by construction:
    // re-seeded AT the refused checkpoint, the retry's seed cost IS the
    // certificate's cost.
    //
    // #2613 removed the anchor desync itself. The solver's band
    // (`outer_gradient_tolerance`) is now a function of the declared problem
    // and of nothing else, and this certificate's band
    // (`outer_certificate_band_at`, #2954) gives coordinate `j` the band
    // `τ_j − ε_j ≥ tolerance` whenever its rounding `ε_j` is below the
    // requested resolution `tolerance·s_j` of its own scale, so a point the
    // solver legitimately converged at is not refused here for being above
    // its band. Where rounding exceeds that resolution the band is the
    // arithmetic's own `ε_j`, labelled `arithmetic-limited`, rather than the
    // solver's band borrowed.
    //
    // What the retry still covers is a different desync with the same shape: a
    // FIDELITY one. Search-time evaluations may run under the inner-PIRLS cap,
    // so the gradient the solver stopped on and the gradient re-measured here
    // at full inner fidelity are not the same number. Re-seeding at the refused
    // checkpoint still collapses that difference, for the same reason. Bounded
    // to a single retry; only fires when the solver CLAIMED convergence (a
    // budget-exhausted result is not a desync — its refusal is genuine).
    // CERTIFICATION-LAST FIT OWNERSHIP. The terminal reinstallation
    // re-evaluates at `result.rho`, so any certificate measured BEFORE it
    // describes a state the caller never receives: on a nonconvex profile the
    // certificate-time inner mode and the finally-installed inner mode can sit
    // in different coefficient basins (measured on the cause-specific survival
    // gate as a stable bitwise mismatch, terminal 9.1931e2 vs certified
    // 9.1671e2, because the two paths prime the inner solve under different
    // eval orders). Running the terminal installation FIRST and certifying
    // LAST makes the certificate's own evaluation the final objective-state
    // installer, so the sealed terminal identity fit assembly binds against IS
    // the certified evidence — bitwise, by construction, independent of basin
    // multiplicity.
    let certify_and_install = |obj: &mut dyn OuterObjective,
                               result: &mut OuterResult|
     -> Result<OuterCriterionCertificate, EstimationError> {
        // Reinstall the selected point under cap=0 so the certificate below
        // measures the full-fidelity state belonging to `result.rho`, not
        // whatever capped search evaluation last touched the objective
        // (seeding beta alone does not restore weights, factors, or link
        // state). Reset forces a real installation instead of an LRU value hit.
        let terminal_cap_guard = config
            .outer_inner_cap
            .as_ref()
            .map(FullFidelityInnerCapGuard::lift);
        // Reset is conditional on the cap contract, mirroring
        // `certify_outer_optimality`'s own doctrine: REML/mixture
        // objectives with a cap can hold a coarse search cache that must
        // not be installed as terminal state, while uncapped stateful
        // objectives (reactive-domain entries among them) retain the very
        // state their evaluation at `result.rho` depends on — an
        // unconditional reset here wiped it and made the certification
        // evaluation non-finite on the reactive fixture.
        //
        // OR-in the terminal-coefficient-mode ownership signal (#2334):
        // objectives that install an owned coefficient mode here but hold
        // their inner cap in a different field (custom families) leave
        // `outer_inner_cap` `None`, so the cap gate alone never fires and
        // `finalize` here could land in a different inner basin than the
        // certifying re-eval below — a spurious bitwise bind failure on a
        // bimodal inner solve. Forcing the reset for mode-owning objectives
        // makes both installations start from the same clean baseline.
        if terminal_cap_guard.is_some() || obj.owns_terminal_coefficient_mode() {
            obj.reset();
        }
        let terminal_installation = obj.finalize_outer_result(&result.rho, &result.plan_used);
        let terminal_inner_converged = inner_solve_converged(config.outer_inner_cap.as_ref());
        drop(terminal_cap_guard);
        terminal_installation?;
        if !terminal_inner_converged {
            return Err(outer_nonconvergence_error(
                context,
                "final outer state installation did not converge at full inner fidelity",
                result,
                result.final_grad_norm,
                StationarityStandard::NoComparison,
            ));
        }
        certify_outer_optimality(obj, config, context, result)
    };
    // Certify-last reseed loop (#2357, #2348, #2392). When the mandatory analytic
    // certificate refuses and publishes a strategy change (`take_certify_reseed`),
    // the search re-runs from that reseed with a fresh metric. A refusal that
    // publishes no reseed is returned as is (#2817). The loop is bounded by
    // certified descent, not by a count (`certify_reseed_admitted`): after the
    // first reseed, one is taken only while the value the certificate evaluated at
    // the refused point is strictly below the value it evaluated at the previous
    // refusal. A re-run whose warm solver value improved but whose certified value
    // did not (#2155/#2363) returns the refusal instead of taking another reseed.
    let mut last_refused_certified_value: Option<f64> = None;
    // #2939 — once a mint has certified a strict saddle, every later run in this
    // solve searches on the declared curvature (`OuterConfig::curvature_search_latched`).
    let mut curvature_search_latched = config.curvature_search_latched;
    let mut saddle_escapes = 0usize;
    let certificate = loop {
        match certify_and_install(obj, &mut result) {
            Ok(certificate) => break certificate,
            Err(refusal) => {
                let Some(reseed) = take_certify_reseed(&mut result) else {
                    return Err(dominated_plateau_refusal(
                        context,
                        &result,
                        dominance_refusal_kind(&result, None),
                        refusal,
                    ));
                };
                // `result.final_value` is the certifying evaluation's value at the
                // refused point.
                let certified_value = result.final_value;
                if !certify_reseed_admitted(last_refused_certified_value, certified_value) {
                    return Err(dominated_plateau_refusal(
                        context,
                        &result,
                        dominance_refusal_kind(&result, Some(reseed.kind)),
                        refusal,
                    ));
                }
                last_refused_certified_value = Some(certified_value);
                let prior_iterations = result.iterations;
                log::debug!(
                    "[OUTER] {context}: analytic certification refused after \
                     {prior_iterations} iteration(s) (final_value={certified_value:.6e}); re-running \
                     from the certificate's reseed {} (#2357/#2348/#2392)",
                    match reseed.kind {
                        CertifyReseedKind::SaddleEscape => {
                            "off the negative-curvature saddle ridge"
                        }
                        CertifyReseedKind::WrongRail => {
                            "at the wrong-rail coordinate's clean-band interior scale"
                        }
                        CertifyReseedKind::ActiveSet => {
                            "with the poisoned rail frozen so the interior polishes in the reduced box"
                        }
                    }
                );
                let mut retry_cfg = config.clone();
                retry_cfg.initial_rho = Some(reseed.rho);
                // Active-set reduction (#2392): the polish runs in the REDUCED
                // (frozen) box so the interior converges without the railed
                // coordinate's ill-conditioned Hessian row poisoning the step. The
                // loop re-certifies the polished point under the ORIGINAL box at
                // the top of the next iteration (`certify_outer_optimality` reads
                // `model_domain_bounds`, which `search_bounds_override` cannot
                // redefine), so the reduction can narrow the SEARCH but never the
                // feasible set a certificate is judged against. `config.clone()`
                // reset each iteration, so the frozen box never persists past
                // this run.
                //
                // A coordinate is only frozen here if the projector had already
                // zeroed it (#2454), i.e. it was on an active constraint when the
                // freeze was taken. That test is at freeze time deliberately: the
                // retry is one-shot, so there is no second reduction to undo a
                // wrong freeze, and the only path back off a frozen bound is the
                // wrong-rail pull-back, which needs a clean opposite-sign
                // exponential tail and declines on any coordinate without one.
                if let Some(frozen_bounds) = reseed.search_bounds_override {
                    retry_cfg.search_bounds_override = Some(frozen_bounds);
                }
                retry_cfg.heuristic_log_lambdas = None;
                // The reseed starts strictly below the refused checkpoint, which is already
                // below every state an earlier search carried in, so nothing the re-run
                // certifies can be dominated by a carried state. Carrying one would only let a
                // re-run whose start is refused hand that older state back as its own stop
                // (#2953).
                retry_cfg.carried_checkpoint = None;
                // Every reseed lands at a genuinely different point, so the refused
                // checkpoint's metric (trust radius, outer Hessian) must not be
                // transferred into the restart.
                retry_cfg.operator_initial_trust_radius = None;
                retry_cfg.warm_start_outer_hessian = None;
                // A certified strict saddle latches the declared analytic Hessian into
                // the search where its escape point is inside the gradient band, so a
                // gradient-only restart would stop at iteration 0 (#2939), and at every
                // later escape (`saddle_escape_latch`). The latch survives every later
                // reseed kind in this solve.
                if reseed.kind == CertifyReseedKind::SaddleEscape {
                    curvature_search_latched = curvature_search_latched
                        || super::saddle_escape_latch::saddle_escape_needs_curvature_search(
                            obj,
                            config,
                            context,
                            retry_cfg.initial_rho.as_ref(),
                            saddle_escapes,
                        );
                    saddle_escapes += 1;
                }
                retry_cfg.curvature_search_latched = curvature_search_latched;
                obj.reset();
                match run_outer_uncertified(obj, &retry_cfg, context) {
                    Ok(mut retried) => {
                        retried.iterations = retried.iterations.saturating_add(prior_iterations);
                        retried.dominated_plateau = lowest_dominated_plateau(
                            result.dominated_plateau.take(),
                            retried.dominated_plateau.take(),
                        );
                        result = retried;
                    }
                    // The reseed could not even run. A fatal evaluation failure is why
                    // the fit stopped, so it propagates as it is: swallowing it published
                    // an older refusal as the reason the fit stopped (#2953). Any other
                    // failure (e.g. the checkpoint is a hard refusal wall for the
                    // objective) leaves the certification refusal from the point this
                    // iteration started at, which carries the checkpoint evidence, with
                    // the reseed's failure attached.
                    Err(reseed_error) if reseed_error.is_fatal_outer_evaluation() => {
                        return Err(reseed_error);
                    }
                    Err(reseed_error) => {
                        return Err(dominated_plateau_refusal(
                            context,
                            &result,
                            dominance_refusal_kind(&result, Some(reseed.kind)),
                            with_reseed_failure(context, refusal, reseed.kind, &reseed_error),
                        ));
                    }
                }
            }
        }
    };
    result.criterion_certificate = Some(certificate);
    Ok(result)
}

/// Build a CANONICAL-order copy of an [`OuterConfig`] for the
/// permutation-invariant outer search (#1538/#1539).
///
/// `perm[c]` is the native coordinate at canonical slot `c`. Every
/// per-coordinate config field (initial ρ seed, heuristic-λ seed, per-axis
/// bounds, transferred warm Hessian) is reordered native→canonical so the
/// optimizer's seeding and multistart operate entirely in canonical space;
/// scalar fields are copied verbatim. `rho_canonical_keys` is cleared so the
/// recursive [`run_outer`] frame runs the normal (identity-order) pipeline on
/// the already-canonical objective.
fn canonicalize_outer_config(config: &OuterConfig, perm: &[usize]) -> OuterConfig {
    // Permute a per-coordinate slice native→canonical; pass through any length
    // that does not match the permutation (defensive — should not occur).
    let permute_vec = |v: &[f64]| -> Vec<f64> {
        if v.len() == perm.len() {
            perm.iter().map(|&i| v[i]).collect()
        } else {
            v.to_vec()
        }
    };
    let permute_arr = |a: &Array1<f64>| -> Array1<f64> {
        if a.len() == perm.len() {
            Array1::from_iter(perm.iter().map(|&i| a[i]))
        } else {
            a.clone()
        }
    };
    let mut canonical = config.clone();
    canonical.rho_canonical_keys = None;
    canonical.native_coordinate_order = Some(perm.to_vec());
    if let Some(initial) = config.initial_rho.as_ref() {
        canonical.initial_rho = Some(permute_arr(initial));
    }
    if let Some(bound) = config.initial_inner_seed.as_ref() {
        canonical.initial_inner_seed = Some(BoundInnerSeed {
            theta: permute_arr(&bound.theta),
            beta: bound.beta.clone(),
        });
    }
    if let Some(h) = config.heuristic_log_lambdas.as_ref() {
        canonical.heuristic_log_lambdas = Some(permute_vec(h));
    }
    if let Some((lower, upper)) = config.model_domain_bounds.as_ref() {
        canonical.model_domain_bounds = Some((permute_arr(lower), permute_arr(upper)));
    }
    if let Some((lower, upper)) = config.model_domain_limit_faces.as_ref() {
        let permute_faces = |faces: &[bool]| -> Vec<bool> {
            if faces.len() == perm.len() {
                perm.iter().map(|&i| faces[i]).collect()
            } else {
                faces.to_vec()
            }
        };
        canonical.model_domain_limit_faces = Some((permute_faces(lower), permute_faces(upper)));
    }
    if let Some((lower, upper)) = config.search_bounds_override.as_ref() {
        canonical.search_bounds_override = Some((permute_arr(lower), permute_arr(upper)));
    }
    // A transferred dense outer Hessian is in native coordinate order; permute
    // it into canonical order so the BFGS warm metric stays aligned. (None on
    // the cold-start canonicalized path, so this is usually a no-op.)
    if let Some(h) = config.warm_start_outer_hessian.as_ref()
        && h.nrows() == perm.len()
        && h.ncols() == perm.len()
    {
        let mut hc = Array2::<f64>::zeros((perm.len(), perm.len()));
        for (a, &ia) in perm.iter().enumerate() {
            for (b, &ib) in perm.iter().enumerate() {
                hc[[a, b]] = h[[ia, ib]];
            }
        }
        canonical.warm_start_outer_hessian = Some(hc);
    }
    // A carried checkpoint is in native order. Mapping a result to native order under the
    // inverse permutation maps it to canonical order.
    if let Some(checkpoint) = config.carried_checkpoint.as_ref() {
        let mut inverse = vec![0usize; perm.len()];
        for (c, &i) in perm.iter().enumerate() {
            inverse[i] = c;
        }
        canonical.carried_checkpoint = Some(outer_result_to_native(checkpoint.clone(), &inverse));
    }
    if let Some(bound) = config.initial_curvature.as_ref()
        && bound.hessian.nrows() == perm.len()
        && bound.hessian.ncols() == perm.len()
    {
        canonical.initial_curvature = Some(BoundOuterCurvature {
            theta: permute_arr(&bound.theta),
            hessian: Array2::from_shape_fn((perm.len(), perm.len()), |(a, b)| {
                bound.hessian[[perm[a], perm[b]]]
            }),
        });
    }
    canonical
}

/// The solver ladder behind [`run_outer`], without the #934 self-audit.
pub(crate) fn run_outer_uncertified(
    obj: &mut dyn OuterObjective,
    config: &OuterConfig,
    context: &str,
) -> Result<OuterResult, EstimationError> {
    let cap = primary_capability_for_config(obj.capability(), config, context);
    cap.validate_layout(context)?;
    // #2370: reject a degenerate / inverted ρ-box up front, as a typed error.
    // Every downstream stage — the per-atom EFS path below and
    // `run_outer_with_plan` — projects seeds against these bounds with
    // `f64::clamp`, whose `min > max` (or NaN) precondition panics *inside the
    // Rust boundary* and surfaces as an opaque `GamfitError: ... panicked` across
    // the FFI, violating the fail-loudly contract. The configured box can invert
    // whenever an independently-derived upper bound drifts below the lower wall
    // (e.g. the custom-family effective-df ceiling vs. `rho_lower_bound`).
    // Validating the *effective* template here — the same one every consumer
    // reads — turns any such inversion into `EstimationError::InvalidInput`
    // regardless of how the bounds were constructed.
    {
        let (model_lo, model_hi) =
            outer_model_domain_bounds_template(config, cap.n_params);
        let (bound_lo, bound_hi) = outer_search_bounds_template(config, cap.n_params);
        if model_lo.len() != cap.n_params
            || model_hi.len() != cap.n_params
            || bound_lo.len() != cap.n_params
            || bound_hi.len() != cap.n_params
        {
            return Err(EstimationError::InvalidInput(format!(
                "{context}: outer bound dimension mismatch: parameters={},                  model_lower={}, model_upper={}, search_lower={}, search_upper={}",
                cap.n_params,
                model_lo.len(),
                model_hi.len(),
                bound_lo.len(),
                bound_hi.len(),
            )));
        }
        for i in 0..cap.n_params {
            let native = native_coordinate(config.native_coordinate_order.as_deref(), i);
            if !(model_lo[i].is_finite() && model_hi[i].is_finite())
                || model_lo[i] > model_hi[i]
            {
                return Err(EstimationError::InvalidInput(format!(
                    "{context}: outer model-domain bounds are invalid at coordinate {native}:                      lower={}, upper={}",
                    model_lo[i], model_hi[i]
                )));
            }
            if bound_lo[i] < model_lo[i] || bound_hi[i] > model_hi[i] {
                return Err(EstimationError::InvalidInput(format!(
                    "{context}: outer search bounds escape the model domain at coordinate {native}:                      model=[{}, {}], search=[{}, {}]",
                    model_lo[i], model_hi[i], bound_lo[i], bound_hi[i]
                )));
            }
            if !(bound_lo[i].is_finite() && bound_hi[i].is_finite()) {
                return Err(EstimationError::InvalidInput(format!(
                    "{context}: outer rho bounds are non-finite at coordinate {native}: \
                     lower={}, upper={}",
                    bound_lo[i], bound_hi[i]
                )));
            }

            // Report a collapsed interval with BOTH walls. `outer_bounds` below
            // is the backstop and rejects the same condition, but its message
            // names only the coordinate. The panic this guard replaced printed
            // `min = -10.0, max = -11.855421656441532`, and those two numbers
            // are what made #2370 diagnosable from a bug report alone: they
            // identify WHICH pair of independently-derived bounds drifted, and
            // by how much. A typed error must not be a weaker diagnostic than
            // the panic it replaced.
            //
            // Two tests constrain this string: `inverted_rho_box_is_a_typed_
            // error_not_a_clamp_panic_2370` greps for the word "bound", and
            // `the_inverted_box_refusal_carries_both_bound_values_2370` pins
            // both numeric walls. Keep both when rewording.
            if bound_lo[i] > bound_hi[i] {
                return Err(EstimationError::InvalidInput(format!(
                    "{context}: outer rho bounds are inverted at coordinate {native}: \
                     lower bound {} exceeds upper bound {}",
                    bound_lo[i], bound_hi[i]
                )));
            }
        }
        outer_bounds(&bound_lo, &bound_hi)
            .map_err(|err| EstimationError::InvalidInput(format!("{context}: {err}")))?;
    }
    if let Some(initial_rho) = config.initial_rho.as_ref() {
        cap.theta_layout()
            .validate_point_len(initial_rho, "initial outer seed")
            .map_err(|err| {
                EstimationError::fatal_objective_evaluation(
                    format!("{context}: initial outer seed validation"),
                    err,
                )
            })?;
    }
    // A resume attempt (gam#3002) accepts the prior certificate where it stands
    // or declines; it runs no plan, so no fallback can search from the point.
    if config.resume_value.is_some() {
        return resume_prior_certificate(obj, config, &cap, context);
    }
    // Frontier ρ-scaling auto-switch (#986): at per-atom-EFS-eligible frontier
    // rho dimension the decoupled per-atom fixed point is the primary outer
    // iteration; everything else falls through to the dense / standard path
    // below. Routed here so every entry point inherits it (magic by default).
    if let Some(result) = run_per_atom_efs_if_frontier(obj, config, context)? {
        if result.solver_claimed_convergence() {
            return Ok(result);
        }
        return Err(outer_nonconvergence_error(
            context,
            "per-atom EFS exhausted its iteration budget before the fixed-point step converged",
            &result,
            None,
            StationarityStandard::NoComparison,
        ));
    }

    if cap.n_params == 0 {
        let cost = obj.eval_cost(&Array1::zeros(0))?;
        let the_plan = plan(&cap);
        let mut result =
            outer_result_with_gradient_norm(Array1::zeros(0), cost, 0, Some(0.0), true, the_plan);
        result.origin = OuterResultOrigin::EmptyParameterSpace;
        return Ok(result);
    }

    // Build the ordered list of capabilities to attempt: primary first, then
    // any centrally-derived degraded capabilities. Aux direct-search has no
    // degraded ladder — a single attempt either succeeds or the failure is
    // surfaced to the caller.
    let fallback_attempts = match config.fallback_policy {
        FallbackPolicy::Automatic => automatic_fallback_attempts(&cap),
        FallbackPolicy::Disabled => Vec::new(),
    };
    let mut attempts: Vec<OuterCapability> = Vec::with_capacity(1 + fallback_attempts.len());
    attempts.push(cap.clone());
    for degraded in fallback_attempts {
        attempts.push(degraded);
    }

    let mut last_error: Option<EstimationError> = None;
    // A state an earlier search evaluated is this ladder's first checkpoint, so its first
    // attempt judges what it certifies against it as every later attempt does (#2953).
    let mut best_checkpoint: Option<OuterResult> = config
        .carried_checkpoint
        .as_ref()
        .map(carried_checkpoint_of);
    // #2953 — the lowest certified optimum an attempt declined because an evaluated
    // state beat it. It rides on the result this loop returns, so a terminal refusal
    // reports it instead of losing it.
    let mut dominated_plateau: Option<DominatedPlateauRecord> = None;
    // A recoverable refusal at the point proposed by EFS says nothing against
    // the finite incumbent that proposed it.  Carry that incumbent across the
    // plan boundary exactly once; the analytic-gradient fallback must resume
    // it before any unrelated seed is considered.
    let mut fixed_point_continuation: Option<FixedPointContinuationCheckpoint> = None;
    // A fixed-point walk can stop normally while its mandatory analytic
    // screening certificate disproves the proposed root.  `run_outer_with_plan`
    // returns that state as `Exhausted`, not `Converged`; retain its best finite
    // checkpoint so the already-declared analytic-gradient fallback starts
    // there instead of replaying the same refuted fixed-point walk or throwing
    // away useful work.
    let mut refuted_fixed_point_continuation: Option<OuterResult> = None;
    // The lowest finite state an earlier attempt of THIS ladder ended at without a
    // claim (#3306). A degraded plan changes how the search moves, not the
    // objective, so it resumes that state instead of re-searching from the seed:
    // restarting discarded every accepted step of the refused attempt. On a binary
    // Bernoulli marginal-slope fit the gradient-only BFGS attempt ended at the
    // certified value to seven digits, and exact-curvature ARC re-searched from
    // the seed. A state carried in from an earlier search is only this ladder's
    // comparator (`carried_checkpoint`), never its start: the caller chose this
    // ladder's seed, and a multistart member keeps its own basin.
    let mut ladder_incumbent: Option<OuterResult> = None;
    // Iterations spent by attempts whose results this function discards: a plan
    // the degraded ladder replaces, a fixed-point walk handed to BFGS.
    // `OuterResult.iterations` is the total across solver restarts and these are
    // the restarts, so the returned result carries them (#2817).
    let mut spent_iterations: usize = 0;

    'plan_attempts: for (attempt_idx, attempt_cap) in attempts.iter().enumerate() {
        let the_plan = plan(attempt_cap);
        if attempt_idx > 0 {
            log::trace!("[OUTER] {context}: primary plan failed; falling back to {the_plan}");
        }
        log_plan(context, attempt_cap, &the_plan);

        obj.reset();

        let mut attempt_config = config.clone();
        // The lowest state an earlier attempt evaluated is this attempt's first checkpoint, so a
        // candidate this attempt certifies above it is declined as one its own search beat
        // would be. Otherwise this attempt could publish the optimum an earlier attempt declined
        // (#2953).
        attempt_config.carried_checkpoint = best_checkpoint.as_ref().map(carried_checkpoint_of);
        if let Some(incumbent) = ladder_incumbent.as_ref() {
            attempt_config.initial_rho = Some(incumbent.rho.clone());
            // The configured inner seed belongs to the configured start, not to
            // this incumbent; the inner solve warm-starts from its own cache.
            attempt_config.initial_inner_seed = None;
            // A mid-run incumbent is not a terminal certificate imported from a
            // prior fit, and a transferred Hessian is bound to that prior fit's
            // rho, not to this state.
            attempt_config.initial_rho_is_prior_terminal_certificate = false;
            attempt_config.warm_start_outer_hessian = None;
            log::debug!(
                "[OUTER] {context}: resuming {the_plan} from the lowest finite state an \
                 earlier attempt of this ladder ended at ({:?}, {} iteration(s)): \
                 cost={:.6e}, |g|={:?}",
                incumbent.plan_used.solver,
                incumbent.iterations,
                incumbent.final_value,
                incumbent.final_grad_norm,
            );
        }
        if let Some(checkpoint) = fixed_point_continuation.take() {
            if !matches!(the_plan.solver, Solver::Bfgs) {
                return Err(EstimationError::RemlOptimizationFailed(format!(
                    "{context}: fixed-point continuation requires analytic-gradient BFGS, \
                     but the next declared plan is {the_plan}"
                )));
            }
            attempt_config.initial_rho = Some(checkpoint.point.clone());
            attempt_config.initial_inner_seed = checkpoint.inner_seed.clone();
            // This is a mid-run finite incumbent, not a terminal certificate
            // imported from a prior fit.  Leaving the original config's cache
            // provenance set could let the zero-iteration resume path accept
            // the checkpoint without ever running the promised BFGS polish.
            attempt_config.initial_rho_is_prior_terminal_certificate = false;
            // A transferred Hessian is bound to the prior fit's terminal rho,
            // not to this mid-run EFS incumbent.  BFGS must rebuild curvature
            // from gradients at the continued point instead of combining two
            // different checkpoints.
            attempt_config.warm_start_outer_hessian = None;
            // The finite incumbent is the continuation's single start.
            log::debug!(
                "[OUTER] {context}: resuming {the_plan} first from the last finite {:?} \
                 incumbent after {} iteration(s) of its plan attempt: cost={:.6e}, \
                 |step|={:.3e}, inner_beta={}",
                checkpoint.plan_used.solver,
                checkpoint.iterations,
                checkpoint.sample.value,
                checkpoint.sample.step.dot(&checkpoint.sample.step).sqrt(),
                checkpoint
                    .inner_seed
                    .as_ref()
                    .map_or(0, |seed| seed.beta.len()),
            );
        }
        if let Some(checkpoint) = refuted_fixed_point_continuation.take() {
            if !matches!(the_plan.solver, Solver::Bfgs) {
                return Err(EstimationError::RemlOptimizationFailed(format!(
                    "{context}: an analytically refuted fixed point requires \
                     analytic-gradient BFGS, but the next declared plan is {the_plan}"
                )));
            }
            attempt_config.initial_rho = Some(checkpoint.rho.clone());
            attempt_config.initial_inner_seed = None;
            attempt_config.initial_rho_is_prior_terminal_certificate = false;
            attempt_config.warm_start_outer_hessian = None;
            // The checkpoint is the continuation's single start.
            log::debug!(
                "[OUTER] {context}: analytic screening refuted the {:?} fixed point; \
                 resuming {the_plan} first from its best finite checkpoint after {} \
                 iteration(s): cost={:.6e}",
                checkpoint.plan_used.solver,
                checkpoint.iterations,
                checkpoint.final_value,
            );
        }

        // An exhausted iteration budget is a refusal, not a reason to rerun the
        // search for more iterations: the checkpoint and its iteration ledger go to
        // the certificate, which mints only a stationary point (SPEC rules 21 and
        // 23, #2817). Only a strategy change the plan itself requests continues the
        // attempt with another solver.
        let outcome = match run_outer_with_plan(
            obj,
            &attempt_config,
            context,
            attempt_cap,
            &the_plan,
            true,
        ) {
            Ok(PlanRunOutcome::Converged(result)) => Ok(result),
            Ok(PlanRunOutcome::FirstOrderFallbackRequested(request)) => {
                log::trace!(
                    "[OUTER] {context}: attempt {} (plan={the_plan}) requested a joint \
                     first-order fallback: {}",
                    attempt_idx + 1,
                    request.reason(),
                );
                last_error = Some(EstimationError::RemlOptimizationFailed(
                    request.reason().to_string(),
                ));
                spent_iterations = spent_iterations.saturating_add(request.spent_iterations());
                continue 'plan_attempts;
            }
            Ok(PlanRunOutcome::FixedPointContinuationRequested(request)) => {
                let has_bfgs_fallback = attempts
                    .get(attempt_idx + 1)
                    .is_some_and(|next| matches!(plan(next).solver, Solver::Bfgs));
                if !has_bfgs_fallback {
                    return Err(EstimationError::RemlOptimizationFailed(format!(
                        "{context}: {:?} refused a trial after {} iteration(s) of its \
                         plan attempt at rho={} (cost={:.6e}), but no analytic-gradient BFGS \
                         continuation is declared: {}",
                        request.checkpoint.plan_used.solver,
                        request.checkpoint.iterations,
                        request.checkpoint.point,
                        request.checkpoint.sample.value,
                        request.refusal,
                    )));
                }
                last_error = Some(EstimationError::RemlOptimizationFailed(format!(
                    "{:?} continuation requested after rho-local trial refusal: {}",
                    request.checkpoint.plan_used.solver, request.refusal,
                )));
                spent_iterations =
                    spent_iterations.saturating_add(request.checkpoint.iterations);
                fixed_point_continuation = Some(request.checkpoint);
                continue 'plan_attempts;
            }
            Ok(PlanRunOutcome::DominatedPlateau(dominated)) => {
                log::debug!(
                    "[OUTER] {context}: attempt {} (plan={the_plan}) declined a certified \
                     winner at cost {:.6e}, dominated by an evaluated state at cost {:.6e} \
                     (gap {:.3e} > rounding envelope {:.3e}); that state is the resume \
                     checkpoint (#2596, #2627)",
                    attempt_idx + 1,
                    dominated.plateau.final_value,
                    dominated.incumbent.final_value,
                    dominated.gap,
                    dominated.band,
                );
                dominated_plateau = lowest_dominated_plateau(
                    dominated_plateau,
                    Some(DominatedPlateauRecord {
                        plateau_rho: dominated.plateau.rho,
                        plateau_value: dominated.plateau.final_value,
                        gap: dominated.gap,
                        band: dominated.band,
                        continuation: dominated.continuation,
                    }),
                );
                Ok(dominated.incumbent)
            }
            Ok(PlanRunOutcome::Exhausted(result)) => {
                // `Exhausted` is a proof-bearing outcome: every solver
                // claim in this plan failed the mandatory analytic
                // screening certificate.  A fixed-point solver may still
                // leave `solver_claimed_convergence == true` on the retained
                // checkpoint because its heuristic update was zero.  Do
                // not collapse that checkpoint back into success below;
                // continue it with the analytic-gradient fallback that the
                // capability ladder already declared.
                //
                // #2822 — the same holds for a walk that stopped without a claim: the
                // unprogressing-walk guard and the iteration cap both hand back the best
                // iterate the walk evaluated. The declared BFGS attempt runs either way,
                // and it resumes from that checkpoint rather than restarting at the seed.
                // Restarting discarded the walk: planted-circle SAE fits stopped their EFS
                // walk at |Pg| ≈ 1e-3 on the guard (each step's decrease ~g²/h sits far
                // under the 1/(2n) resolution), BFGS re-ran from the seed, and the
                // terminal certificate refused the EFS checkpoint it could not beat.
                let has_bfgs_fallback = attempts
                    .get(attempt_idx + 1)
                    .is_some_and(|next| matches!(plan(next).solver, Solver::Bfgs));
                if matches!(the_plan.solver, Solver::Efs | Solver::HybridEfs)
                    && has_bfgs_fallback
                {
                    log::debug!(
                        "[OUTER] {context}: {:?} stopped (solver claim: {}), but no \
                         candidate passed analytic screening; continuing the best finite \
                         checkpoint with analytic-gradient BFGS",
                        the_plan.solver,
                        result.solver_claimed_convergence(),
                    );
                    last_error = Some(EstimationError::RemlOptimizationFailed(format!(
                        "{:?} checkpoint was refuted by analytic screening",
                        the_plan.solver,
                    )));
                    spent_iterations = spent_iterations.saturating_add(result.iterations);
                    refuted_fixed_point_continuation = Some(result);
                    continue 'plan_attempts;
                }
                Ok(result)
            }
            Err(e) => Err(e),
        };

        match outcome {
            Ok(mut result) => {
                if result.solver_claimed_convergence() {
                    result.iterations = result.iterations.saturating_add(spent_iterations);
                    result.dominated_plateau =
                        lowest_dominated_plateau(result.dominated_plateau.take(), dominated_plateau);
                    return Ok(result);
                }

                spent_iterations = spent_iterations.saturating_add(result.iterations);
                let improves_checkpoint = result.final_value.is_finite()
                    && best_checkpoint.as_ref().is_none_or(|checkpoint| {
                        !checkpoint.final_value.is_finite()
                            || result.final_value < checkpoint.final_value
                    });
                let improves_incumbent = result.final_value.is_finite()
                    && ladder_incumbent
                        .as_ref()
                        .is_none_or(|incumbent| result.final_value < incumbent.final_value);
                if improves_incumbent {
                    ladder_incumbent = Some(result.clone());
                }
                if improves_checkpoint {
                    best_checkpoint = Some(result);
                }

                let message = format!(
                    "{context}: attempt {} (plan={the_plan}) exhausted without convergence",
                    attempt_idx + 1
                );
                log::trace!("[OUTER] {message}; trying degraded fallback plan");
                last_error = Some(EstimationError::RemlOptimizationFailed(message));
            }
            Err(e) => {
                if e.is_fatal_outer_evaluation() {
                    return Err(e);
                }
                log::trace!(
                    "[OUTER] {context}: attempt {} (plan={the_plan}) failed: {e}",
                    attempt_idx + 1
                );
                last_error = Some(e);
            }
        }
    }

    if let Some(mut checkpoint) = best_checkpoint {
        // Every attempt that ended here, this checkpoint's own included, has
        // already added its iterations to `spent_iterations`.
        checkpoint.iterations = spent_iterations;
        // The solver ladder produced no result that its OWN internal
        // (raw-gradient) convergence test accepted — but that test cannot see a
        // railed or already-stationary optimum. At a smoothing parameter railed to
        // the ρ box floor (λ→0, e.g. an exact linear fit or a separated smooth),
        // the RAW gradient stays large along the railed axis — it "wants" to push
        // past the boundary — so the solver reports non-convergence and can take
        // zero steps, even though the KKT-PROJECTED gradient (which zeroes
        // outward-railed axes) is stationary and no feasible step reduces the
        // objective. Only the mandatory analytic certificate in `run_outer`
        // computes that projected gradient AND the curvature-scaled flat-valley
        // bound (½·gᵀH⁻¹g ≤ objective_tol), so IT, not this raw-gradient ladder, is
        // the sole authority on stationarity. Hand it the best finite checkpoint:
        // `certify_outer_optimality` mints iff the point is genuinely stationary
        // (interior, railed, or flat-valley) and returns typed non-convergence
        // otherwise, so a truly divergent fit is still rejected there.
        checkpoint.dominated_plateau =
            lowest_dominated_plateau(checkpoint.dominated_plateau.take(), dominated_plateau);
        return Ok(checkpoint);
    }

    Err(last_error.unwrap_or_else(|| {
        EstimationError::RemlOptimizationFailed(format!("all plan attempts exhausted ({context})"))
    }))
}

// ─── Frontier ρ-scaling auto-switch (issue #986) ─────────────────────────
//
// ARD-per-atom assigns one smoothing coordinate per dictionary atom, so the
// ρ-vector reaches 10^4–10^5 coordinates. A dense outer quasi-Newton over that
// materializes an O(K²) Hessian and is impossible at scale. When the ρ-dimension
// is frontier-scale AND every coordinate is penalty-like with a working
// fixed-point hook, route the PRIMARY outer iteration to the per-atom decoupled
// EFS path (`crate::estimate::reml::per_atom_efs`) instead of the dense
// ARC/BFGS lane. The decision is auto-derived from the coordinate count alone —
// there is no flag — and it is additive: the dense path is unchanged for small K
// and for any objective that is not per-atom-EFS-eligible.

/// Whether this capability is in the frontier ρ-scaling regime where the
/// per-atom decoupled EFS primary should take over from the dense outer.
///
/// Delegates the eligibility decision to
/// `crate::estimate::reml::per_atom_efs::per_atom_efs_eligible`, which
/// requires all-penalty-like coordinates, a working `eval_efs` hook,
/// fixed-point not disabled, and a frontier-scale ρ-dimension. This is the
/// single auto-switch predicate; `plan` keeps selecting the
/// dense or standard-EFS solver for everything below the frontier threshold.
pub(crate) fn is_per_atom_efs_frontier(cap: &OuterCapability) -> bool {
    crate::estimate::reml::per_atom_efs::per_atom_efs_eligible(cap)
}

/// Auto-switch entry point: when `cap` is frontier-scale per-atom-EFS-eligible,
/// run the per-atom decoupled EFS primary and return its [`OuterResult`];
/// otherwise return `Ok(None)` so the caller falls through to the existing dense
/// / standard-EFS path via [`OuterProblem::run`] / [`run_outer`].
///
/// Builds the same bounded seed and tolerance/budget the standard plan path
/// uses, takes the same single derived start (initial-ρ if supplied, else the
/// commensurate-curvature start — the per-atom fixed point is a contraction
/// near the optimum), then drives the per-atom EFS loop. The shared-border
/// topology defaults to disjoint (every atom owns a private penalty block — the
/// common ARD-per-atom case); callers with a known arrow-border overlap can run
/// the module's `run_per_atom_efs` directly with a populated
/// `SharedBorderTopology`.
///
/// Additive: this function neither mutates nor bypasses the dense path; it is
/// the pre-dispatch shortcut [`run_outer`] calls before the dense ladder.
pub(crate) fn run_per_atom_efs_if_frontier(
    obj: &mut dyn OuterObjective,
    config: &OuterConfig,
    context: &str,
) -> Result<Option<OuterResult>, EstimationError> {
    let cap = primary_capability_for_config(obj.capability(), config, context);
    cap.validate_layout(context)?;
    if !is_per_atom_efs_frontier(&cap) {
        return Ok(None);
    }

    let the_plan = plan(&cap);
    let rho_dim = cap.theta_layout().rho_dim();

    let model_domain_bounds = outer_model_domain_bounds_template(config, cap.n_params);
    crate::estimate::reml::outer_eval::record_current_outer_rho_model_upper_bounds_for_ift(
        &model_domain_bounds.1,
    );
    let (lower, upper) = outer_search_bounds_template(config, cap.n_params);

    // The per-atom multiplicative fixed point is locally contractive, so it
    // enters from the same single derived start as every other plan.
    let seed = crate::rho_optimizer::run_plan::outer_start_point(
        config,
        cap.n_params,
        &model_domain_bounds,
    )?;

    log::debug!(
        "[OUTER] {context}: frontier ρ-scaling (rho_dim={rho_dim}) → per-atom decoupled EFS primary"
    );

    let pa_cfg = crate::estimate::reml::per_atom_efs::PerAtomEfsConfig::new(
        config.tolerance,
        config.max_iter,
        lower,
        upper,
        outer_criterion_resolution(config),
    );
    let topology = crate::estimate::reml::per_atom_efs::SharedBorderTopology::disjoint(rho_dim);

    obj.reset();
    install_matching_initial_inner_seed(obj, config, &seed, context)?;
    let result =
        crate::estimate::reml::per_atom_efs::run_per_atom_efs(obj, &seed, &pa_cfg, &topology)?;
    Ok(Some(result.into_outer_result(the_plan)))
}

#[cfg(test)]
#[path = "inverted_rho_box_tests.rs"]
mod inverted_rho_box_tests;

/// The outer ρ box. Opt's box tolerance decides when a coordinate sits AT a
/// bound: its projected-gradient component is masked and outward direction
/// components are dropped. That asks whether an iterate is resolvably distinct
/// from the bound, and a smooth objective's minimizer is resolvable in ρ only to
/// `√ε`, the band the affine-face REML box (`constrained_gaussian_reml`) already
/// passes. It replaces a picked `1e-6` (#2469).
pub(crate) fn outer_bounds(lo: &Array1<f64>, hi: &Array1<f64>) -> Result<Bounds, EstimationError> {
    Bounds::new(lo.clone(), hi.clone(), f64::EPSILON.sqrt()).map_err(|err| {
        EstimationError::InvalidInput(format!("outer rho bounds are invalid: {err}"))
    })
}

/// The model domain the runner reasons against: the caller's declared domain,
/// else the supported log-strength domain `[LOG_STRENGTH_MIN, LOG_STRENGTH_MAX]`
/// on which `exp(ρ)` is evaluated exactly (`gam_problem::log_strength`). Every
/// production route declares a derived domain (#2812), so this default binds
/// only a caller that states none (SPEC rule 20, #2902 row 8).
pub(crate) fn outer_model_domain_bounds_template(
    config: &OuterConfig,
    n: usize,
) -> (Array1<f64>, Array1<f64>) {
    config.model_domain_bounds.clone().unwrap_or_else(|| {
        (
            Array1::<f64>::from_elem(n, gam_problem::LOG_STRENGTH_MIN),
            Array1::<f64>::from_elem(n, gam_problem::LOG_STRENGTH_MAX),
        )
    })
}

pub(crate) fn outer_search_bounds_template(
    config: &OuterConfig,
    n: usize,
) -> (Array1<f64>, Array1<f64>) {
    config
        .search_bounds_override
        .clone()
        .unwrap_or_else(|| outer_model_domain_bounds_template(config, n))
}

/// Intersect typed objective-domain faces with the caller's declared model
/// domain. The resulting box is the immutable feasible set used by every
/// stationarity certificate. Algorithmic active-set reduction is represented
/// separately by `search_bounds_override` and cannot redefine this domain.
pub(super) fn install_objective_domain(
    config: &mut OuterConfig,
    n_params: usize,
    objective_lower: Option<Array1<f64>>,
    objective_upper: Option<Array1<f64>>,
) -> Result<(), EstimationError> {
    let (mut lower, mut upper) = outer_model_domain_bounds_template(config, n_params);
    if lower.len() != n_params || upper.len() != n_params {
        return Err(EstimationError::InvalidInput(format!(
            "outer configured bounds dimension mismatch: parameters={n_params}, lower={}, upper={}",
            lower.len(),
            upper.len(),
        )));
    }
    if let Some(domain) = objective_lower.as_ref()
        && domain.len() != n_params
    {
        return Err(EstimationError::InvalidInput(format!(
            "outer objective-domain lower-bound dimension mismatch: parameters={n_params}, lower={}",
            domain.len()
        )));
    }
    if let Some(domain) = objective_upper.as_ref()
        && domain.len() != n_params
    {
        return Err(EstimationError::InvalidInput(format!(
            "outer objective-domain upper-bound dimension mismatch: parameters={n_params}, upper={}",
            domain.len()
        )));
    }
    for index in 0..n_params {
        let native = native_coordinate(config.native_coordinate_order.as_deref(), index);
        if let Some(domain) = objective_lower.as_ref() {
            let value = domain[index];
            if !value.is_finite() {
                return Err(EstimationError::InvalidInput(format!(
                    "outer objective-domain lower bound[{native}] must be finite; got {value}"
                )));
            }
            if value > lower[index]
                && let Some((faces, _)) = config.model_domain_limit_faces.as_mut()
                && let Some(face) = faces.get_mut(index)
            {
                *face = false;
            }
            lower[index] = lower[index].max(value);
        }
        if let Some(domain) = objective_upper.as_ref() {
            let value = domain[index];
            if !value.is_finite() {
                return Err(EstimationError::InvalidInput(format!(
                    "outer objective-domain upper bound[{native}] must be finite; got {value}"
                )));
            }
            if value < upper[index]
                && let Some((_, faces)) = config.model_domain_limit_faces.as_mut()
                && let Some(face) = faces.get_mut(index)
            {
                // The objective's own face cut the limit model's: the bound in
                // force is no longer the limit model (#2954).
                *face = false;
            }
            upper[index] = upper[index].min(value);
        }
        if !(lower[index].is_finite() && upper[index].is_finite() && lower[index] < upper[index]) {
            return Err(EstimationError::InvalidInput(format!(
                "outer objective-domain intersection is empty or non-finite at coordinate {native}: lower={}, upper={}",
                lower[index], upper[index]
            )));
        }
    }
    config.model_domain_bounds = Some((lower, upper));
    config.search_bounds_override = None;
    Ok(())
}

pub(crate) fn outer_tolerance(value: f64) -> Result<Tolerance, EstimationError> {
    Tolerance::new(value)
        .map_err(|err| EstimationError::InvalidInput(format!("outer tolerance is invalid: {err}")))
}

/// The step norm below which a fixed-point map has nothing left to resolve.
///
/// A map's proposed step is computed from traces and logs, so its own roundoff
/// is `√ε` relative to the coordinate it moves: a component below
/// `√ε·(1 + |θ_i|)` cannot be told from the map's arithmetic. Every coordinate
/// of the search box satisfies `|θ_i| ≤ max|bound|`, so a step whose every
/// component sits at that resolution has Euclidean norm at most
/// `√n·√ε·(1 + max|bound|)`. `opt::FixedPoint` stops once the projected step
/// reaches this norm, and `run_fixed_point_outer_solver` then judges the point
/// with the screening certificate. The threshold decides when that judgement is
/// spent, never what it concludes. It replaces the absolute `config.tolerance`,
/// which denominated a stationarity claim in step currency (#2817).
pub(crate) fn fixed_point_step_resolution(config: &OuterConfig, n_params: usize) -> f64 {
    let (lower, upper) = outer_search_bounds_template(config, n_params);
    let box_scale = lower
        .iter()
        .chain(upper.iter())
        .fold(0.0_f64, |scale, bound| scale.max(bound.abs()));
    f64::EPSILON.sqrt() * (n_params.max(1) as f64).sqrt() * (1.0 + box_scale)
}

/// The criterion's resolution in its own absolute units: the statistical
/// resolution `τ_stat = 1/(2n)` over the declared observations
/// ([`OuterProblemSize::statistical_resolution`], C3).
///
/// Every judgement of "the criterion cannot tell these apart" reads this one
/// number: the cost-stall guard's no-improvement test where the evaluations
/// carry no objective band, the ARC online stop and the matrix-free model
/// decrement, the curvature-resolvability and gradient-reproducibility rungs,
/// the asymptote-rail and large-step flatness certificates, and the
/// negative-curvature adjudication's falsifiable range. A decrease below
/// `τ_stat` moves no reported quantity by more than the `n^{-1/2}` sampling
/// error the inference built on the optimum already carries; it does not move
/// with the units of `y` or with an additive constant in `V`, which the
/// `rel·(1 + |V|)` floor it replaces did, and it shrinks as `n` grows (#2954).
///
/// `0.0` when the route declares no observation count: such a criterion has no
/// statistical resolution, so nothing is waived as unresolvable — a tolerance
/// test `x ≤ 0` passes only on exact equality and the rung it gates does not
/// fire.
pub(crate) fn outer_criterion_resolution(config: &OuterConfig) -> f64 {
    config
        .problem_size
        .statistical_resolution()
        .filter(|tau| tau.is_finite() && *tau > 0.0)
        .unwrap_or(0.0)
}

/// The largest step the negative-curvature adjudication takes along its eigenvector: one
/// e-fold of `log λ` ([`adjudicate_negative_curvature`]).
pub(crate) const NEGATIVE_CURVATURE_LADDER_LARGEST_STEP: f64 = 1.0;

/// The criterion's curvature resolution `2·τ` over its objective resolution
/// `τ` ([`outer_criterion_resolution`]; #1082, #2817).
///
/// Along an eigenvector of `λ < 0` at a stationary point the quadratic model
/// predicts the decrease `½|λ|α²`. The largest step the negative-curvature
/// adjudication takes is [`NEGATIVE_CURVATURE_LADDER_LARGEST_STEP`], so a direction
/// with `½|λ|·α_max² ≤ τ` predicts nothing the criterion resolves anywhere in the
/// range that could falsify it: this is `opt::unresolvable_curvature_magnitude` at
/// that step and resolution, the number the adjudication's own resolvability verdict
/// (`opt::negative_curvature_claim`, #3036) reads. The bridge's definiteness verdict,
/// the seed's verdict, and the decrement that both the in-loop stop and the
/// certificate's curvature rung take all read it too. `0.0` (the arithmetic shift
/// alone) when the resolution is zero or unusable.
pub(crate) fn criterion_curvature_resolution(objective_resolution: f64) -> f64 {
    opt::unresolvable_curvature_magnitude(
        NEGATIVE_CURVATURE_LADDER_LARGEST_STEP,
        objective_resolution,
    )
    .unwrap_or(0.0)
}

/// Whether the certified objective strictly dropped between two refusals of the
/// certify-last reseed loop (#2374, #2817).
///
/// `prior_value` is the value certified at the previous refusal and
/// `retried_value` the value certified at this one. A drop no larger than
/// `rel_cost_floor·(1 + min(|prior|, |retried|))` means the re-run found no
/// descent. Anchoring the floor on the SMALLER of the two costs keeps a tiny
/// uphill wobble from reading as progress, and a non-finite retried value is
/// never progress.
pub(crate) fn certify_resume_made_progress(
    prior_value: f64,
    retried_value: f64,
    rel_cost_floor: f64,
) -> bool {
    let floor = rel_cost_floor * (1.0 + prior_value.abs().min(retried_value.abs()));
    retried_value.is_finite() && retried_value < prior_value - floor
}

/// Whether the certify-last loop may take the reseed published at a refusal whose
/// certifying evaluation read `certified_value`, given the value certified at the
/// previous refusal (#2817). The first reseed is taken; every later one only after
/// strict certified descent (`certify_resume_made_progress` at the roundoff
/// scale), so the certified values strictly decrease across the loop and it ends
/// without a count.
fn certify_reseed_admitted(previous_certified_value: Option<f64>, certified_value: f64) -> bool {
    previous_certified_value.is_none_or(|previous| {
        certify_resume_made_progress(previous, certified_value, CERTIFY_RESUME_PROGRESS_REL)
    })
}

/// The stationarity band handed to the SOLVER, and to the cost-stall guard's
/// stationarity gate: the caller's absolute `tolerance`, capped by a caller
/// requirement.
///
/// It is a function of the DECLARED problem and of nothing else. `opt` resolves
/// a `GradientTolerance` exactly once, at run start, against the seed cost —
/// so handing it a `rel_cost` component makes the band a function of where a
/// seed happened to land. On #2392's exponentially stiff recovery that produced
/// an eighteen-order spread across the seeds of ONE fit: a generated lattice
/// seed at `ρ = 1.0`, where the criterion is `1.79e13`, gave
///
/// ```text
/// termination=gradient_tolerance(|g|=1.522998e-4 < 1.792397e8)
/// ```
///
/// i.e. the solver claimed convergence on the wrong rail against a threshold no
/// gradient can fail. A stationarity test that depends on the starting point is
/// not a stationarity test: two seeds converging to the same optimum must reach
/// the same verdict.
///
/// #2954: the band no longer grows with the row count. The REML/LAML
/// ρ-gradient `½[λ_k tr(H⁻¹S_k) − rank(S_k) + λ_k β̂ᵀS_kβ̂]` is a difference of
/// terms bounded by `rank(S_k)` and the penalty energy, not a sum over rows, so
/// the `n·√ε` floor and the declared-scale rung `τ·(1 + n)` it replaced charged
/// the gradient a resolution it does not have: at `n = 490` the floor was
/// `7.3e-6` and at `n = 300,000` the declared band was `6.0`, so a seed
/// certified in zero iterations. The certificate's per-coordinate band
/// ([`outer_certificate_band_at`]) is at least `τ_j − ε_j = tolerance +
/// (tolerance·s_j − ε_j)`, so at least this one wherever the rounding `ε_j`
/// stays below `tolerance·s_j`, the case at every rank-bearing coordinate short
/// of a criterion the arithmetic cannot resolve.
pub(crate) fn outer_gradient_tolerance(config: &OuterConfig) -> GradientTolerance {
    GradientTolerance {
        abs: outer_stationarity_band_and_rung(config).bound,
        rel_initial_grad: None,
        // Never delegated: `opt`'s only anchor is the seed. See above.
        rel_cost: None,
        projected: true,
    }
}

/// A certificate band together with the rung that produced it (#2688).
///
/// The band is decided several ways and every one of them used to reach the
/// call site as a bare `f64` that was then labelled `SolverBand`
/// unconditionally. Returning the pair is the invariant
/// [`StationarityBound::from_ladder`] already enforces one level up, pushed
/// down to where the number is decided.
#[derive(Debug, Clone, Copy)]
pub(crate) struct CertificateBandAt {
    /// The band the certificate applies.
    pub(crate) bound: f64,
    /// Which input produced [`Self::bound`].
    pub(crate) source: StationarityBoundSource,
    /// What the ENGINE would have applied with no caller requirement, reported
    /// beside `bound` so a reader can judge whether the requirement was
    /// reasonable. Equal to [`Self::bound`] unless the cap bound.
    pub(crate) engine_bound: f64,
    /// The rung that produced [`Self::engine_bound`]. Never `CallerRequirement`.
    pub(crate) engine_source: StationarityBoundSource,
}

/// Apply the caller's `|Pg|` requirement (#2568) to an engine band.
///
/// `min`, never `max`: a requirement looser than the engine's own band is not
/// a request for anything, and honouring it would let a caller *weaken* a
/// standard the engine derived. Strict `<`, and the rung says so (#2688).
fn cap_at_caller_requirement(
    config: &OuterConfig,
    engine_bound: f64,
    engine_source: StationarityBoundSource,
) -> CertificateBandAt {
    match config.required_projected_gradient_norm {
        Some(required) if required < engine_bound => CertificateBandAt {
            bound: required,
            source: StationarityBoundSource::CallerRequirement,
            engine_bound,
            engine_source,
        },
        _ => CertificateBandAt {
            bound: engine_bound,
            source: engine_source,
            engine_bound,
            engine_source,
        },
    }
}

/// The band a stationarity test applies where no evaluation's gradient parts
/// are in hand: the caller's absolute `tolerance`, capped by a caller
/// requirement. No criterion value enters, so the band is the same at every
/// point and at every row count (#2954).
pub(crate) fn outer_stationarity_band_and_rung(config: &OuterConfig) -> CertificateBandAt {
    cap_at_caller_requirement(
        config,
        config.tolerance,
        StationarityBoundSource::SolverBand,
    )
}

/// One gradient coordinate's Theorem 9 band (#2954): the statistical
/// tolerance `τ_j`, the rigorous rounding bound `ε_j` of the computed `ĝ_j`, and
/// the band `τ_j − ε_j` a computed component must clear.
#[derive(Debug, Clone, Copy, PartialEq)]
pub(crate) struct CoordinateBand {
    pub(crate) tau: f64,
    pub(crate) epsilon: f64,
}

impl CoordinateBand {
    /// `max(ε_j, τ_j − ε_j)`. Where `2ε_j ≤ τ_j` it is `τ_j − ε_j`, and
    /// `|ĝ_j| ≤ τ_j − ε_j` implies `|g_j| ≤ τ_j` for the exact gradient. Where
    /// `2ε_j > τ_j` an exact stationary point's computed component may sit
    /// anywhere in `[−ε_j, ε_j]`, beyond `τ_j − ε_j`, so that band would refuse
    /// stationary points; `ε_j` is the narrowest band every one of them clears,
    /// and a pass proves `|g_j| ≤ 2ε_j`.
    pub(crate) fn band(self) -> f64 {
        self.epsilon.max(self.tau - self.epsilon)
    }

    /// `2ε_j > τ_j`: the arithmetic cannot certify `τ_j` itself.
    pub(crate) fn is_arithmetic_limited(self) -> bool {
        2.0 * self.epsilon > self.tau
    }
}

/// The per-coordinate Theorem 9 bands of the gradient an evaluation published
/// its parts for (#2954), `None` for a coordinate it published none for.
///
/// * `τ_j = tolerance·(1 + s_j)` with `s_j = ½·rank(S_j) + |fixed_beta_j|`, the
///   magnitudes `∂V/∂ρ_j = ½[λ_j tr(H⁻¹S_j) − rank(S_j) + λ_j β̂ᵀS_jβ̂]` is a
///   difference of: `λ_j tr(H⁻¹S_j) ≤ rank(S_j)`, and `fixed_beta_j` is the
///   penalty-energy channel `½λ_jβ̂ᵀS_jβ̂`. The caller's `tolerance` is relative
///   to the scale of the terms the component is formed from, which is set by
///   the penalty's rank and the fitted function's roughness, never by `n`.
/// * `ε_j = γ_m·(|fixed_beta_j| + |logdet_h_j| + |logdet_s_j| + |kkt_j|)`, the
///   rounding of the four additive channels the component is summed from, at
///   the u-based growth `γ_m` over the sequential count `m = n + p²` the
///   Newton-decrement bands charge (`outer_decrement_bands`).
///
/// `None` overall when the route declares no problem size: there is no
/// formation count to charge `ε_j` at.
pub(crate) fn outer_coordinate_bands(
    config: &OuterConfig,
    coordinates: usize,
    evidence: &crate::estimate::outer_eval_capture::CertificateEvidence,
) -> Option<Vec<Option<CoordinateBand>>> {
    let size = &config.problem_size;
    let (Some(n_obs), Some(p_coefficients)) = (size.n_obs, size.p_coefficients) else {
        return None;
    };
    let growth = gam_linalg::roundoff::accumulation_growth(n_obs + p_coefficients * p_coefficients);
    Some(
        (0..coordinates)
            .map(|k| {
                let part = evidence.parts.iter().find(|part| part.index == k)?;
                let kkt = part.total - (part.fixed_beta + part.logdet_h + part.logdet_s);
                let scale = 0.5 * part.rank as f64 + part.fixed_beta.abs();
                let band = CoordinateBand {
                    tau: config.tolerance * (1.0 + scale),
                    epsilon: growth
                        * (part.fixed_beta.abs()
                            + part.logdet_h.abs()
                            + part.logdet_s.abs()
                            + kkt.abs()),
                };
                (band.tau.is_finite() && band.epsilon.is_finite()).then_some(band)
            })
            .collect(),
    )
}

/// The stationarity band a CERTIFICATE applies at the point it is judging,
/// from the gradient parts of the evaluation that measured `projected_gradient`
/// (#2954, Theorem 9 of the convergence theory's floating-point analysis).
///
/// Coordinate `j` certifies iff `|(Pĝ)_j| ≤ max(ε_j, τ_j − ε_j)`
/// ([`CoordinateBand`]); a coordinate the evaluation published no parts for is
/// held to the caller's absolute `tolerance`. A pass implies
/// `|(Pg)_j| ≤ max(τ_j, 2ε_j)` for the exact gradient. The per-coordinate test
/// is rendered as the scalar gauge the certificate compares `‖Pĝ‖₂` against,
///
/// ```text
/// bound = ‖Pĝ‖₂ / max_j (|(Pĝ)_j| / b_j),
/// ```
///
/// which `‖Pĝ‖₂` clears exactly when every coordinate clears its own `b_j`. A
/// coordinate whose projected component is exactly zero (held on its face, or
/// genuinely stationary) constrains nothing. At `Pĝ = 0` the bound is the
/// smallest band.
///
/// [`StationarityBoundSource::ArithmeticLimited`] labels a bound set by a
/// coordinate with `2ε_j > τ_j`: the verdict there is at the arithmetic's
/// resolution `2ε_j`, not at the requested `τ_j`, and the label says so.
///
/// Without a declared problem size there is no rounding to charge and the
/// declared band ([`outer_stationarity_band_and_rung`]) stands.
pub(crate) fn outer_certificate_band_at(
    config: &OuterConfig,
    projected_gradient: &Array1<f64>,
    evidence: &crate::estimate::outer_eval_capture::CertificateEvidence,
) -> CertificateBandAt {
    let Some(bands) = outer_coordinate_bands(config, projected_gradient.len(), evidence) else {
        return outer_stationarity_band_and_rung(config);
    };
    if bands.iter().all(Option::is_none) {
        return outer_stationarity_band_and_rung(config);
    }
    let band_of = |band: Option<CoordinateBand>| band.map_or(config.tolerance, CoordinateBand::band);
    let norm = projected_gradient.iter().map(|v| v * v).sum::<f64>().sqrt();
    let source_of = |band: Option<CoordinateBand>| match band {
        Some(band) if band.is_arithmetic_limited() => StationarityBoundSource::ArithmeticLimited,
        _ => StationarityBoundSource::CoordinateBand,
    };
    // The binding coordinate: the largest `|(Pĝ)_j| / b_j`. Every band is
    // positive (`max(ε_j, τ_j − ε_j) ≥ τ_j/2`), so the ratio is finite.
    let mut binding: Option<(f64, Option<CoordinateBand>)> = None;
    for (&component, &band) in projected_gradient.iter().zip(bands.iter()) {
        if component == 0.0 {
            continue;
        }
        let ratio = component.abs() / band_of(band);
        if binding.is_none_or(|(worst, _)| ratio > worst) {
            binding = Some((ratio, band));
        }
    }
    let Some((ratio, band)) = binding else {
        let (smallest, band) = bands.iter().fold(
            (f64::INFINITY, None),
            |(smallest, chosen), &band| {
                let b = band_of(band);
                if b < smallest { (b, band) } else { (smallest, chosen) }
            },
        );
        return cap_at_caller_requirement(config, smallest, source_of(band));
    };
    cap_at_caller_requirement(config, norm / ratio, source_of(band))
}

pub(crate) fn outer_max_iterations(value: usize) -> Result<MaxIterations, EstimationError> {
    MaxIterations::new(value)
        .map_err(|err| EstimationError::InvalidInput(format!("outer max_iter is invalid: {err}")))
}

pub(crate) fn sanitized_operator_trust_restart_radius(radius: Option<f64>) -> Option<f64> {
    radius
        .filter(|value| value.is_finite() && *value > 0.0)
        .map(|value| value.max(OPERATOR_TRUST_RESTART_RADIUS_FLOOR))
}

pub(crate) fn bfgs_axis_step_caps(
    config: &OuterConfig,
    layout: OuterThetaLayout,
) -> Option<Array1<f64>> {
    if config.bfgs_step_cap.is_none() && config.bfgs_step_cap_psi.is_none() {
        return None;
    }
    let mut caps = Array1::from_elem(layout.n_params, f64::INFINITY);
    if let Some(cap) = config.bfgs_step_cap {
        for i in 0..layout.rho_dim() {
            caps[i] = cap;
        }
    }
    if let Some(cap) = config.bfgs_step_cap_psi {
        for i in layout.rho_dim()..layout.n_params {
            caps[i] = cap;
        }
    }
    Some(caps)
}

pub(crate) enum FixedPointOuterRunError {
    SeedRejected(ObjectiveEvalError),
    IterationRejected(FixedPointContinuationRequest),
    ImmediateFallback(FirstOrderFallbackRequest),
    Failed(EstimationError),
}

/// Last complete fixed-point incumbent preceding a refused trial point.
///
/// `opt::FixedPoint` currently returns only the refused evaluation's message;
/// it does not return its still-finite incumbent on `ObjectiveFailed`.  This
/// carrier preserves the exact optimizer state needed to continue with a
/// different algorithm: outer point, criterion, proposed fixed-point step,
/// fixed-point status, the iterations its plan attempt completed (every seed the
/// attempt started, this walk included), plan, and the matching inner
/// coefficient state when the EFS producer supplied one.
#[derive(Clone, Debug)]
pub(crate) struct FixedPointContinuationCheckpoint {
    pub(crate) point: Array1<f64>,
    pub(crate) sample: FixedPointSample,
    pub(crate) iterations: usize,
    pub(crate) plan_used: OuterPlan,
    pub(crate) inner_seed: Option<BoundInnerSeed>,
}

/// Typed request to continue a fixed-point incumbent after one rho-local
/// refusal.
///
/// The refusal remains an [`ObjectiveEvalError`] all the way through the plan
/// boundary, so the fallback decision is based on the producer's recoverable
/// verdict rather than on message text.
#[derive(Clone, Debug)]
pub(crate) struct FixedPointContinuationRequest {
    pub(crate) checkpoint: FixedPointContinuationCheckpoint,
    pub(crate) refusal: ObjectiveEvalError,
}

/// Carries a fixed-point objective's complete typed refusal across the lossy
/// `opt` fixed-point return boundary.
///
/// `OuterFixedPointBridge::eval_step` returns a typed `ObjectiveEvalError`
/// whose kind is the producer's verdict: `Recoverable` for a refusal that is
/// a property of THIS rho (a non-finite cost, a non-finite EFS step, a
/// bubbled `RemlOptimizationFailed`, or any `EstimationError` for which
/// `is_trial_point_infeasible()` answers true), `Fatal` only for a structural
/// failure. `opt::FixedPoint::run` then does `err.into_message()` and hands
/// back `FixedPointError::ObjectiveFailed { message }`. This adapter retains
/// the original error in a publication slot, so its producer verdict, typed
/// source, and any first-order routing request all survive.
///
/// `run_fixed_point_outer_solver` used to answer that String with an
/// unconditional `fatal_outer_evaluation`, and `is_fatal_outer_evaluation()`
/// is a hard `return Err(e)` in BOTH the seed loop (`run_plan.rs`) and the
/// strategy ladder (`run.rs`). So one recoverable per-rho refusal at outer
/// iteration k killed the remaining seeds and the entire fallback ladder --
/// including the `disable_fixed_point` BFGS plan `automatic_fallback_attempts`
/// builds for precisely this situation. The SEED evaluation in the same
/// function never had this bug: it still holds the typed error there and asks
/// `is_recoverable()`. Only the iterations lost the verdict, in transit.
///
/// This adapter is the publication slot that gets it back -- the same device
/// `recurrent_incumbent_exit` uses to hand a value out of a moved bridge. The
/// slot is written on EVERY evaluation (cleared to `None` on success), so an
/// error can never be read stale.
pub(crate) struct RetainingObjective<ObjFn> {
    inner: ObjFn,
    last_error: Arc<Mutex<Option<ObjectiveEvalError>>>,
}

impl<ObjFn> RetainingObjective<ObjFn> {
    pub(crate) fn new(
        inner: ObjFn,
        last_error: Arc<Mutex<Option<ObjectiveEvalError>>>,
    ) -> Self {
        Self { inner, last_error }
    }

    fn publish<T>(&self, outcome: &Result<T, ObjectiveEvalError>) {
        *self
            .last_error
            .lock()
            .expect("objective error publication lock poisoned") =
            outcome.as_ref().err().cloned();
    }
}

impl<ObjFn> ZerothOrderObjective for RetainingObjective<ObjFn>
where
    ObjFn: ZerothOrderObjective,
{
    fn eval_cost(&mut self, x: &Array1<f64>) -> Result<f64, ObjectiveEvalError> {
        let outcome = self.inner.eval_cost(x);
        self.publish(&outcome);
        outcome
    }
}

impl<ObjFn> FirstOrderObjective for RetainingObjective<ObjFn>
where
    ObjFn: FirstOrderObjective,
{
    fn eval_grad(&mut self, x: &Array1<f64>) -> Result<FirstOrderSample, ObjectiveEvalError> {
        let outcome = self.inner.eval_grad(x);
        self.publish(&outcome);
        outcome
    }
}

impl<ObjFn> SecondOrderObjective for RetainingObjective<ObjFn>
where
    ObjFn: SecondOrderObjective,
{
    fn eval_hessian(&mut self, x: &Array1<f64>) -> Result<SecondOrderSample, ObjectiveEvalError> {
        let outcome = self.inner.eval_hessian(x);
        self.publish(&outcome);
        outcome
    }
}

impl<ObjFn> FixedPointObjective for RetainingObjective<ObjFn>
where
    ObjFn: FixedPointObjective,
{
    fn eval_step(&mut self, x: &Array1<f64>) -> Result<FixedPointSample, ObjectiveEvalError> {
        let outcome = self.inner.eval_step(x);
        self.publish(&outcome);
        outcome
    }
}

/// Retains every successful fixed-point sample as well as the typed error
/// retained by [`RetainingObjective`].
///
/// A failed evaluation belongs to the proposed *next* point.  Therefore it
/// must not overwrite `incumbent`: that slot remains the exact last finite
/// point from which another solver can continue. `best_iterate` keeps the
/// lowest finite sample, which a budget-exhausted walk publishes when its last
/// iterate is worse (#2817).
pub(crate) struct RetainingFixedPointObjective<ObjFn> {
    inner: RetainingObjective<ObjFn>,
    incumbent: Arc<Mutex<FixedPointContinuationCheckpoint>>,
    best_iterate: Arc<Mutex<FixedPointContinuationCheckpoint>>,
    evaluated_inner_seed: Arc<Mutex<Option<BoundInnerSeed>>>,
    successful_iterations: usize,
    plan_used: OuterPlan,
}

impl<ObjFn> RetainingFixedPointObjective<ObjFn> {
    pub(crate) fn new(
        inner: ObjFn,
        last_error: Arc<Mutex<Option<ObjectiveEvalError>>>,
        incumbent: Arc<Mutex<FixedPointContinuationCheckpoint>>,
        best_iterate: Arc<Mutex<FixedPointContinuationCheckpoint>>,
        evaluated_inner_seed: Arc<Mutex<Option<BoundInnerSeed>>>,
        plan_used: OuterPlan,
    ) -> Self {
        Self {
            inner: RetainingObjective::new(inner, last_error),
            incumbent,
            best_iterate,
            evaluated_inner_seed,
            successful_iterations: 0,
            plan_used,
        }
    }
}

impl<ObjFn> FixedPointObjective for RetainingFixedPointObjective<ObjFn>
where
    ObjFn: FixedPointObjective,
{
    fn eval_step(&mut self, x: &Array1<f64>) -> Result<FixedPointSample, ObjectiveEvalError> {
        let outcome = self.inner.eval_step(x);
        if let Ok(sample) = &outcome {
            self.successful_iterations = self.successful_iterations.saturating_add(1);
            let inner_seed = self
                .evaluated_inner_seed
                .lock()
                .expect("fixed-point inner-state publication lock poisoned")
                .clone()
                .filter(|seed| outer_theta_bitwise_eq(&seed.theta, x));
            let checkpoint = FixedPointContinuationCheckpoint {
                point: x.clone(),
                sample: sample.clone(),
                iterations: self.successful_iterations,
                plan_used: self.plan_used,
                inner_seed,
            };
            {
                let mut best = self
                    .best_iterate
                    .lock()
                    .expect("fixed-point best-iterate publication lock poisoned");
                if sample.value.is_finite()
                    && (!best.sample.value.is_finite() || sample.value < best.sample.value)
                {
                    *best = checkpoint.clone();
                }
            }
            *self
                .incumbent
                .lock()
                .expect("fixed-point incumbent publication lock poisoned") = checkpoint;
        }
        outcome
    }
}

/// Judge a fixed-point stop that is not itself a stationarity claim (a
/// step-norm stop, or a walk that stopped buying criterion improvement) by the
/// screening certificate the plan applies to every claim. A certified point is
/// returned as the walk's answer. A refused one is handed, as `continuation`,
/// to the analytic-gradient plan that `automatic_fallback_attempts` declares
/// for this capability; without that plan it is a resumable checkpoint, not a
/// candidate.
fn judge_fixed_point_stop(
    obj: &mut dyn OuterObjective,
    config: &OuterConfig,
    context: &str,
    label: &str,
    stop: &str,
    mut result: OuterResult,
    continuation: FixedPointContinuationCheckpoint,
) -> Result<OuterResult, FixedPointOuterRunError> {
    match certify_outer_optimality_with_fidelity(
        obj,
        config,
        context,
        &mut result,
        CertificationFidelity::Screening,
    ) {
        Ok(certificate) => {
            result.criterion_certificate = Some(certificate);
            Ok(result)
        }
        Err(refusal)
            if config.fallback_policy == FallbackPolicy::Automatic
                && obj.capability().gradient == Derivative::Analytic =>
        {
            log::debug!(
                "[OUTER] {context}: {label} {stop} stop after {} iteration(s) at \
                 cost={:.6e} is not stationary; continuing it with the \
                 analytic-gradient plan: {refusal}",
                result.iterations,
                result.final_value,
            );
            Err(FixedPointOuterRunError::IterationRejected(
                FixedPointContinuationRequest {
                    checkpoint: continuation,
                    refusal: ObjectiveEvalError::recoverable_from(refusal),
                },
            ))
        }
        Err(_) => {
            result.termination = OuterTermination::Exhausted;
            Ok(result)
        }
    }
}

pub(crate) fn run_fixed_point_outer_solver(
    obj: &mut dyn OuterObjective,
    layout: OuterThetaLayout,
    barrier_config: Option<BarrierConfig>,
    config: &OuterConfig,
    context: &str,
    seed: &Array1<f64>,
    the_plan: OuterPlan,
    label: &str,
    failure_prefix: &str,
) -> Result<OuterResult, FixedPointOuterRunError> {
    // Judge the seed before walking from it. The walk's own stop is a step-norm
    // test, never stationarity (see the certificate after the walk), so a seed
    // that is already stationary is walked anyway: a smoothing parameter on its
    // rail keeps proposing an outward EFS step, and nothing short of the
    // unprogressing-walk window ends it. On the ISLR `Default` logistic fit the
    // #784 corrected continuation starts from the certified Laplace optimum, which
    // is stationary under the correction too (the BFGS continuation later
    // certified it at zero iterations, |g| = 5.9e-6), yet the walk spent ~60
    // corrected evaluations there. The screening certificate is the one the walk's
    // stop is judged by, so passing it here is the same claim with zero steps.
    // A refusal is no verdict on the walk; it proceeds from the same seed.
    if obj.capability().gradient == Derivative::Analytic {
        let mut seed_result = OuterResult::new(seed.clone(), f64::NAN, 0, true, the_plan);
        if let Ok(certificate) = certify_outer_optimality_with_fidelity(
            obj,
            config,
            context,
            &mut seed_result,
            CertificationFidelity::Screening,
        ) {
            log::info!(
                "[OUTER] {context}: {label} seed is already stationary at cost={:.6e}; \
                 no fixed-point step taken",
                seed_result.final_value,
            );
            seed_result.criterion_certificate = Some(certificate);
            return Ok(seed_result);
        }
    }
    // Shared publication slot for the recurrent-restored-incumbent stop
    // (#2235 verdict 2): the bridge is moved into the driver, so the streak
    // count comes back through this cell and is stamped onto the returned
    // `OuterResult` below.
    let recurrent_incumbent_exit = Arc::new(Mutex::new(None));
    // Publication slot for the unprogressing-walk stop (#2817), read below the
    // same way.
    let unprogressing_exit = Arc::new(Mutex::new(None));
    let evaluated_inner_seed = Arc::new(Mutex::new(None));
    let mut objective = OuterFixedPointBridge {
        obj: &mut *obj,
        layout,
        barrier_config,
        config,
        evaluated_inner_seed: Arc::clone(&evaluated_inner_seed),
        consecutive_psi_zero_iters: 0,
        last_restored_incumbent_streak: None,
        recurrent_incumbent_exit: Arc::clone(&recurrent_incumbent_exit),
        // The same criterion resolution the gradient routes' cost-stall guard
        // uses, and its first-order window.
        progress: FixedPointProgress::new(outer_criterion_resolution(config), COST_STALL_WINDOW),
        unprogressing_exit: Arc::clone(&unprogressing_exit),
    };
    let seed_sample = match objective.eval_step(seed) {
        Ok(sample) => sample,
        Err(err) if first_order_fallback_request(&err).is_some() => {
            let request = first_order_fallback_request(&err)
                .expect("guard established a typed first-order fallback request")
                .clone();
            return Err(FixedPointOuterRunError::ImmediateFallback(request));
        }
        Err(err) if err.is_recoverable() => {
            return Err(FixedPointOuterRunError::SeedRejected(err));
        }
        Err(err) => {
            return Err(FixedPointOuterRunError::Failed(
                EstimationError::fatal_objective_evaluation(
                    "outer fixed-point seed evaluation",
                    err,
                ),
            ));
        }
    };
    let (lo, hi) = outer_search_bounds_template(config, layout.n_params);
    let bounds = outer_bounds(&lo, &hi).map_err(FixedPointOuterRunError::Failed)?;
    // The map's step at its own arithmetic resolution, not an absolute
    // tolerance: this stop only decides when screening is spent (below).
    let tol = outer_tolerance(fixed_point_step_resolution(config, layout.n_params))
        .map_err(FixedPointOuterRunError::Failed)?;
    let max_iter =
        outer_max_iterations(config.max_iter).map_err(FixedPointOuterRunError::Failed)?;
    // Publication slot for the complete producer error from the last failed
    // `eval_step`. `opt::FixedPoint` returns only its message, so this is the
    // ownership channel for the typed source and routing request.
    let last_step_error: Arc<Mutex<Option<ObjectiveEvalError>>> = Arc::new(Mutex::new(None));
    let seed_inner_state = evaluated_inner_seed
        .lock()
        .expect("fixed-point inner-state publication lock poisoned")
        .clone()
        .filter(|inner_seed| outer_theta_bitwise_eq(&inner_seed.theta, seed));
    let seed_checkpoint = FixedPointContinuationCheckpoint {
        point: seed.clone(),
        sample: seed_sample.clone(),
        iterations: 0,
        plan_used: the_plan,
        inner_seed: seed_inner_state,
    };
    // The best finite iterate the walk evaluated, kept beside the last one: a
    // budget-exhausted walk must not publish a worse point than it passed
    // through (#2817).
    let best_iterate = Arc::new(Mutex::new(seed_checkpoint.clone()));
    let incumbent = Arc::new(Mutex::new(seed_checkpoint));
    let objective = RetainingFixedPointObjective::new(
        objective,
        Arc::clone(&last_step_error),
        Arc::clone(&incumbent),
        Arc::clone(&best_iterate),
        Arc::clone(&evaluated_inner_seed),
        the_plan,
    );
    // The driver owns the bridge, and the bridge borrows `obj`; scoping the
    // driver releases that borrow so a step stop can be judged on `obj` below.
    let outcome = {
        let mut optimizer = FixedPoint::new(seed.clone(), objective)
            // Seed validation already paid the complete EFS inner solve. Reuse that
            // exact sample so iteration zero neither repeats the expensive solve nor
            // mistakes two evaluations at the identical rho for recurrent incumbent
            // evidence (#2241).
            .with_initial_sample(seed.clone(), seed_sample)
            .with_bounds(bounds)
            .with_tolerance(tol)
            .with_max_iterations(max_iter);
        optimizer.run()
    };
    match outcome {
        Ok(sol) => {
            let mut result = solution_into_outer_result(sol, true, the_plan);
            // Stamp the model-state fixed-point stop when the bridge published
            // one. Analytic screening corroborates that claim downstream.
            if let Some(consecutive_restores) =
                recurrent_incumbent_exit.lock().ok().and_then(|slot| *slot)
            {
                result.termination = OuterTermination::SolverClaimed {
                    proposed_via: Some(OuterConvergedVia::RecurrentIncumbent {
                        consecutive_restores,
                    }),
                };
                return Ok(result);
            }
            // The bridge stopped a walk that bought nothing since its previous
            // window (#2817). That is no convergence claim: the best iterate the
            // walk evaluated is the point it leaves behind. It is judged below
            // exactly as a step-norm stop is, because an unprogressing EFS walk
            // is the same failure one window later: the ratio-of-traces map has
            // stopped moving the criterion, which says nothing about the
            // gradient. On the K=1 generated-seed circle (#2153) the walk
            // stalled 32 iterations in at |g| = 7.3e-3, and publishing that
            // stall as the plan's terminal checkpoint skipped the analytic-
            // gradient continuation the capability declares.
            if let Some(evaluations) = unprogressing_exit.lock().ok().and_then(|slot| *slot) {
                let best = best_iterate
                    .lock()
                    .expect("fixed-point best-iterate publication lock poisoned")
                    .clone();
                let (mut checkpoint, continuation) = if best.sample.value.is_finite()
                    && (!result.final_value.is_finite() || best.sample.value < result.final_value)
                {
                    let mut substituted = OuterResult::new(
                        best.point.clone(),
                        best.sample.value,
                        result.iterations.max(evaluations),
                        false,
                        the_plan,
                    );
                    substituted.origin = OuterResultOrigin::FixedPointBestIterateSubstitution;
                    (substituted, best)
                } else {
                    let last = incumbent
                        .lock()
                        .expect("fixed-point incumbent publication lock poisoned")
                        .clone();
                    (result, last)
                };
                checkpoint.termination = OuterTermination::Exhausted;
                return judge_fixed_point_stop(
                    obj,
                    config,
                    context,
                    label,
                    "unprogressing-walk",
                    checkpoint,
                    continuation,
                );
            }
            // Every other stop is a step-norm test: the map proposed a step below
            // `config.tolerance`, through the bridge's per-coordinate test or opt's
            // L2 test. A small step is not stationarity. The EFS update is a ratio
            // of traces whose ZERO is the stationarity equation, but its magnitude
            // is not denominated in the gradient the certificate bounds. On an SAE
            // manifold fit (sw7-logs/m4_A_info.log) the walk stopped after 7m32s and
            // screening then refused the point at |Pg| = 2.010 against a band of
            // 9.699e-5. The seed was abandoned there, and the refuted checkpoint
            // reached the analytic-gradient plan only after every remaining seed
            // had stopped the same way.
            //
            // So the stop is judged here, by the screening certificate the plan
            // applies to every claim, while this incumbent can still be continued.
            let continuation = incumbent
                .lock()
                .expect("fixed-point incumbent publication lock poisoned")
                .clone();
            judge_fixed_point_stop(obj, config, context, label, "step-norm", result, continuation)
        }
        Err(FixedPointError::MaxIterationsReached { last_solution }) => {
            let step_norm = last_solution.final_step_norm.expect(
                "a fixed-point max-iteration solution must carry its final accepted step norm",
            );
            log::debug!(
                "[OUTER warning] {context}: {label} hit max_iter={} at final_value={:.6e} step_norm={:.3e}",
                config.max_iter,
                last_solution.final_value,
                step_norm,
            );
            let best = best_iterate
                .lock()
                .expect("fixed-point best-iterate publication lock poisoned")
                .clone();
            if best.sample.value.is_finite()
                && (!last_solution.final_value.is_finite()
                    || best.sample.value < last_solution.final_value)
            {
                log::debug!(
                    "[OUTER] {context}: {label} budget-exhaustion last iterate (value={:.6e}) \
                     is worse than the best iterate it evaluated (value={:.6e}, iteration {}); \
                     substituting the best iterate (#2817)",
                    last_solution.final_value,
                    best.sample.value,
                    best.iterations,
                );
                let mut result = OuterResult::new(
                    best.point,
                    best.sample.value,
                    last_solution.iterations,
                    false,
                    the_plan,
                );
                result.origin = OuterResultOrigin::FixedPointBestIterateSubstitution;
                return Ok(result);
            }
            Ok(solution_into_outer_result(*last_solution, false, the_plan))
        }
        Err(FixedPointError::ObjectiveFailed { .. }) => {
            let error = last_step_error
                .lock()
                .expect("fixed-point objective error publication lock poisoned")
                .take()
                .expect("FixedPoint::ObjectiveFailed must follow a failed classified eval_step");
            if error.is_recoverable() {
                let checkpoint = incumbent
                    .lock()
                    .expect("fixed-point incumbent publication lock poisoned")
                    .clone();
                return Err(FixedPointOuterRunError::IterationRejected(
                    FixedPointContinuationRequest {
                        checkpoint,
                        refusal: error,
                    },
                ));
            }
            Err(FixedPointOuterRunError::Failed(
                EstimationError::fatal_objective_evaluation(
                    "outer fixed-point evaluation",
                    error,
                ),
            ))
        }
        Err(e) => Err(FixedPointOuterRunError::Failed(
            EstimationError::RemlOptimizationFailed(format!("{failure_prefix}: {e:?}")),
        )),
    }
}

#[cfg(test)]
#[path = "asymptote_rail_certify_tests.rs"]
mod asymptote_rail_certify_tests;

#[cfg(test)]
#[path = "certify_resume_progress_tests.rs"]
mod certify_resume_progress_tests;

#[cfg(test)]
#[path = "outer_stationarity_band_tests.rs"]
mod outer_stationarity_band_tests;

#[cfg(test)]
#[path = "criterion_curvature_ladder_2748_tests.rs"]
mod criterion_curvature_ladder_2748_tests;

#[cfg(test)]
#[path = "saddle_adjudication_evaluable_trials_2665_tests.rs"]
mod saddle_adjudication_evaluable_trials_2665_tests;

#[cfg(test)]
#[path = "saddle_adjudication_unresolvable_3036_tests.rs"]
mod saddle_adjudication_unresolvable_3036_tests;

#[cfg(test)]
#[path = "canonical_checkpoint_order_tests.rs"]
mod canonical_checkpoint_order_tests;
#[cfg(test)]
#[path = "native_coordinate_order_tests.rs"]
mod native_coordinate_order_tests;
