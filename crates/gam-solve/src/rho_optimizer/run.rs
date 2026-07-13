use super::*;

pub(crate) const OPERATOR_TRUST_RESTART_RADIUS_FLOOR: f64 = 1.0e-6;

/// Hold terminal evidence at full inner-solve fidelity.
///
/// Search-time REML evaluations may deliberately cap P-IRLS.  A terminal
/// certificate or final state installation is a different operation: it must
/// run with the cap lifted, then restore the scheduler's value after the
/// operation completes.  The objective reset performed by the callers clears
/// any evaluation/P-IRLS entries created under the search state before the
/// full-fidelity request is made.
pub(crate) struct TerminalInnerCapGuard<'a> {
    cap: &'a AtomicUsize,
    previous: usize,
}

impl<'a> TerminalInnerCapGuard<'a> {
    pub(crate) fn lift(feedback: &'a InnerProgressFeedback) -> Self {
        let cap = feedback.cap.as_ref();
        let previous = cap.swap(0, Ordering::Relaxed);
        Self { cap, previous }
    }
}

impl Drop for TerminalInnerCapGuard<'_> {
    fn drop(&mut self) {
        self.cap.store(self.previous, Ordering::Relaxed);
    }
}

/// Configuration for the outer optimization runner.
#[derive(Clone, Debug)]
pub(crate) struct OuterConfig {
    pub(crate) tolerance: f64,
    /// Optional override for the *relative-cost-decrease* convergence stop,
    /// decoupled from `tolerance`. `outer_gradient_tolerance` normally derives
    /// BOTH the absolute projected-gradient floor
    /// (`max(tolerance, scale·√ε_machine)`)
    /// AND the relative-cost stop (`rel_cost = tolerance`) from the single
    /// `tolerance`. That conflation forces a caller who needs a *tight absolute
    /// floor* (to resolve λ to the genuine REML optimum at large `n`, where the
    /// floor is `scale·√ε_machine`) to also accept a *tight rel-cost stop*,
    /// which on a flat REML ridge never trips and grinds the optimizer to `max_iter` —
    /// dozens of surplus O(D·p³) Laplace-derivative outer iterations (the #1082
    /// multinomial smooth-by-factor wall-clock blow-up). When `Some(r)`, the
    /// rel-cost stop uses `r` while the absolute floor keeps using `tolerance`
    /// via `objective_scale`, so accuracy (absolute floor) and perf (loose
    /// rel-cost) are selected independently. `None` preserves the legacy coupling
    /// (`rel_cost = tolerance`) for every existing path byte-for-byte.
    pub(crate) rel_cost_tolerance: Option<f64>,
    pub(crate) max_iter: usize,
    pub(crate) bounds: Option<(Array1<f64>, Array1<f64>)>,
    pub(crate) seed_config: gam_problem::SeedConfig,
    pub(crate) rho_bound: f64,
    pub(crate) heuristic_lambdas: Option<Vec<f64>>,
    pub(crate) initial_rho: Option<Array1<f64>>,
    pub(crate) fallback_policy: FallbackPolicy,
    pub(crate) screening_cap: Option<Arc<AtomicUsize>>,
    pub(crate) screen_initial_rho: bool,
    /// Outer-aware inner-PIRLS iteration cap (sibling of `screening_cap`).
    /// When set, the BFGS bridge drives this atomic on every accepted
    /// gradient eval to coarsen the inner Newton solve at early outer iters
    /// (when ρ is far from converged) and lift it back to full as
    /// convergence approaches. Distinct from `screening_cap` in that it
    /// does NOT suppress cache writes / warm-start updates / KKT
    /// enforcement; it is purely a budget. See
    /// `RemlObjectiveState::outer_inner_cap` for dual-cap semantics.
    pub(crate) outer_inner_cap: Option<InnerProgressFeedback>,
    pub(crate) operator_initial_trust_radius: Option<f64>,
    pub(crate) arc_initial_regularization: Option<f64>,
    /// Optional scale factor for the objective's natural magnitude.
    /// Used to widen the absolute gradient-norm floor on objectives whose
    /// gradient lives on a non-unit scale (e.g. Gaussian-identity REML at
    /// large `n`, whose ∂/∂logλ inherits the O(n) likelihood constant).
    /// `None` falls back to the bare `tolerance` floor.
    pub(crate) objective_scale: Option<f64>,
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
    pub(crate) rho_uncertainty_problem_size: crate::rho_uncertainty::RhoUncertaintyProblemSize,
    /// Set by the persistent-cache resume path (`run`) when the outer seed
    /// originates from a warm-start cache *hit* — i.e. `config.initial_rho`
    /// (and, since 0.1.204, the inner β) was populated from a prior fit's
    /// persisted near-optimal iterate (`CacheSeedDecision::Seed`). On a hit
    /// the continuation pre-warm — which exists to anneal a COLD seed toward
    /// the optimum — is redundant work (the seed is already near-optimal), so
    /// the pre-warm step budget is dropped to zero and the run proceeds
    /// straight to the BFGS/Newton certificate from the cached iterate. The
    /// converged optimum is unchanged: warm start only sets the STARTING
    /// point. Defaulted `false`, so every cold-start / no-cache path keeps its
    /// existing continuation pre-warm budget byte-for-byte.
    pub(crate) warm_start_cache_hit: bool,
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
}

impl Default for OuterConfig {
    fn default() -> Self {
        Self {
            tolerance: 1e-5,
            rel_cost_tolerance: None,
            max_iter: 200,
            bounds: None,
            seed_config: gam_problem::SeedConfig::default(),
            rho_bound: 30.0,
            heuristic_lambdas: None,
            initial_rho: None,
            fallback_policy: FallbackPolicy::Automatic,
            screening_cap: None,
            screen_initial_rho: false,
            outer_inner_cap: None,
            operator_initial_trust_radius: None,
            arc_initial_regularization: None,
            objective_scale: None,
            bfgs_step_cap: None,
            bfgs_step_cap_psi: None,
            cache_session: None,
            cache_mirror_sessions: Vec::new(),
            rho_uncertainty_problem_size:
                crate::rho_uncertainty::RhoUncertaintyProblemSize::default(),
            warm_start_cache_hit: false,
            warm_start_outer_hessian: None,
            rho_canonical_keys: None,
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
/// [`OuterConfig`] (how the runner should behave) from a small set
/// of high-level declarations.
pub struct OuterProblem {
    n_params: usize,
    gradient: Derivative,
    hessian: DeclaredHessianForm,
    prefer_gradient_only: bool,
    disable_fixed_point: bool,
    psi_dim: usize,
    barrier_config: Option<BarrierConfig>,
    tolerance: f64,
    rel_cost_tolerance: Option<f64>,
    max_iter: usize,
    bounds: Option<(Array1<f64>, Array1<f64>)>,
    rho_bound: f64,
    seed_config: gam_problem::SeedConfig,
    heuristic_lambdas: Option<Vec<f64>>,
    initial_rho: Option<Array1<f64>>,
    fallback_policy: FallbackPolicy,
    screening_cap: Option<Arc<AtomicUsize>>,
    screen_initial_rho: bool,
    outer_inner_cap: Option<InnerProgressFeedback>,
    operator_initial_trust_radius: Option<f64>,
    arc_initial_regularization: Option<f64>,
    objective_scale: Option<f64>,
    bfgs_step_cap: Option<f64>,
    bfgs_step_cap_psi: Option<f64>,
    cache_session: Option<Arc<CacheSession>>,
    cache_mirror_sessions: Vec<Arc<CacheSession>>,
    rho_uncertainty_problem_size: crate::rho_uncertainty::RhoUncertaintyProblemSize,
    continuation_prewarm: bool,
    rho_canonical_keys: Option<Vec<u64>>,
}

impl OuterProblem {
    pub fn new(n_params: usize) -> Self {
        Self {
            n_params,
            gradient: Derivative::Unavailable,
            hessian: DeclaredHessianForm::Unavailable,
            prefer_gradient_only: false,
            disable_fixed_point: false,
            psi_dim: 0,
            barrier_config: None,
            tolerance: 1e-5,
            rel_cost_tolerance: None,
            max_iter: 200,
            bounds: None,
            rho_bound: 30.0,
            seed_config: gam_problem::SeedConfig::default(),
            heuristic_lambdas: None,
            initial_rho: None,
            fallback_policy: FallbackPolicy::Automatic,
            screening_cap: None,
            screen_initial_rho: false,
            outer_inner_cap: None,
            operator_initial_trust_radius: None,
            arc_initial_regularization: None,
            objective_scale: None,
            bfgs_step_cap: None,
            bfgs_step_cap_psi: None,
            cache_session: None,
            cache_mirror_sessions: Vec::new(),
            rho_uncertainty_problem_size:
                crate::rho_uncertainty::RhoUncertaintyProblemSize::default(),
            continuation_prewarm: true,
            rho_canonical_keys: None,
        }
    }

    /// Supply per-ρ-coordinate structural keys (native/formula order) so the
    /// outer search is canonicalized to be invariant to the order the smooth
    /// terms / tensor margins were written (#1538/#1539). See
    /// [`OuterConfig::rho_canonical_keys`].
    pub fn with_rho_canonical_keys(mut self, keys: Option<Vec<u64>>) -> Self {
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
    pub fn with_barrier(mut self, cfg: Option<BarrierConfig>) -> Self {
        self.barrier_config = cfg;
        self
    }
    pub fn with_tolerance(mut self, tol: f64) -> Self {
        self.tolerance = tol;
        self
    }
    pub fn with_max_iter(mut self, n: usize) -> Self {
        self.max_iter = n;
        self
    }
    pub fn with_bounds(mut self, lo: Array1<f64>, hi: Array1<f64>) -> Self {
        self.bounds = Some((lo, hi));
        self
    }
    pub fn with_rho_bound(mut self, b: f64) -> Self {
        self.rho_bound = b;
        self
    }
    pub fn with_seed_config(mut self, sc: gam_problem::SeedConfig) -> Self {
        self.seed_config = sc;
        self
    }
    pub fn with_heuristic_lambdas(mut self, h: Vec<f64>) -> Self {
        self.heuristic_lambdas = Some(h);
        self
    }
    pub fn with_initial_rho(mut self, rho: Array1<f64>) -> Self {
        self.initial_rho = Some(rho);
        self
    }
    /// Toggle the generic rho-continuation seed pre-warm. This does not affect
    /// objectives that require an explicit continuation path; it only controls
    /// the cheap-by-default pre-pass gated by `allow_continuation_prewarm()`.
    pub fn with_continuation_prewarm(mut self, enabled: bool) -> Self {
        self.continuation_prewarm = enabled;
        self
    }
    pub fn with_screening_cap(mut self, screening_cap: Arc<AtomicUsize>) -> Self {
        self.screening_cap = Some(screening_cap);
        self
    }
    /// Allow seed screening to rank the explicit initial rho against generated
    /// candidates even when the effective seed budget is one. The default keeps
    /// a user-provided initial point authoritative and avoids a separate
    /// screening pass.
    pub fn with_screen_initial_rho(mut self, screen_initial_rho: bool) -> Self {
        self.screen_initial_rho = screen_initial_rho;
        self
    }
    /// Wire the bidirectional inner-PIRLS feedback channel.
    ///
    /// The outer bridge writes a coarsened iteration cap into
    /// `feedback.cap` on every accepted gradient/Hessian eval; the inner
    /// solver writes back into `feedback.last_iters` /
    /// `feedback.last_converged` after each non-screening solve so the
    /// next outer iter's schedule can adapt to the inner solver's
    /// actual convergence behavior. Typical caller passes
    /// `InnerProgressFeedback {
    ///     cap: Arc::clone(&reml_state.outer_inner_cap),
    ///     last_iters: Arc::clone(&reml_state.last_inner_iters),
    ///     last_converged: Arc::clone(&reml_state.last_inner_converged),
    /// }` so the inner and outer observe the same atomics.
    pub fn with_outer_inner_cap(mut self, feedback: InnerProgressFeedback) -> Self {
        self.outer_inner_cap = Some(feedback);
        self
    }
    pub fn with_operator_initial_trust_radius(mut self, radius: Option<f64>) -> Self {
        self.operator_initial_trust_radius = sanitized_operator_trust_restart_radius(radius);
        self
    }

    /// Override the ARC initial cubic-regularization parameter sigma
    /// (default in `opt`: 1.0). Smaller sigma → less cubic penalty on the
    /// first step → larger first move on benign objectives. The matrix-
    /// free Newton-TR analog is `with_operator_initial_trust_radius`.
    ///
    /// Used by Gaussian-identity REML at large-scale n: the objective is
    /// quadratic-like in log-λ near the optimum (sigma is the right
    /// scale), and log-λ moves of 2–4 units in the early iters
    /// otherwise burn 4–8 iters of trust-region expansion before the
    /// model trusts the analytic Hessian.
    pub fn with_arc_initial_regularization(mut self, sigma: Option<f64>) -> Self {
        self.arc_initial_regularization = sigma.filter(|v| v.is_finite() && *v > 0.0);
        self
    }

    /// Set the objective's natural magnitude scale, used to derive an
    /// `n`-aware absolute gradient-norm floor. When set to `Some(s)`,
    /// the runner uses `abs_floor = max(tol, s * √ε_machine)` for the
    /// projected-gradient convergence check.
    ///
    /// Rationale: a fixed `abs = tol` (e.g. 1e-6) is appropriate when the
    /// objective and its gradient live on a unit scale, but Gaussian-
    /// identity REML carries an O(n) likelihood constant that flows into
    /// ∂/∂logλ. At large-scale n the floor becomes binding even when the
    /// relative-from-seed component (`rel_initial_grad * ‖g0‖`) declared
    /// convergence iters earlier — chasing sub-ULP changes in log-λ at
    /// the cost of repeated k²·n·p² analytic-Hessian assemblies.
    pub fn with_objective_scale(mut self, scale: Option<f64>) -> Self {
        self.objective_scale = scale.filter(|v| v.is_finite() && *v > 0.0);
        self
    }

    /// Decouple the *relative-cost-decrease* convergence stop from the
    /// absolute projected-gradient floor. By default both are derived from the
    /// single `with_tolerance` value (`abs = max(tol, scale·√ε_machine)`,
    /// `rel_cost = tol`). Supplying `Some(r)` here makes the rel-cost stop use
    /// `r` while the absolute floor keeps using `tolerance` (so a caller can
    /// keep a tight absolute floor for accuracy at large `n` AND a loose
    /// rel-cost stop for perf on a flat REML ridge — see #1082). `None` keeps
    /// the legacy coupling.
    pub fn with_rel_cost_tolerance(mut self, rel_cost: Option<f64>) -> Self {
        self.rel_cost_tolerance = rel_cost.filter(|v| v.is_finite() && *v > 0.0);
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
    /// [`OuterConfig::cache_mirror_sessions`].
    pub fn with_cache_mirror_sessions(mut self, sessions: Vec<Arc<CacheSession>>) -> Self {
        self.cache_mirror_sessions = sessions;
        self
    }

    pub fn with_problem_size(mut self, n_obs: usize, p_coefficients: usize) -> Self {
        self.rho_uncertainty_problem_size = crate::rho_uncertainty::RhoUncertaintyProblemSize {
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
            rel_cost_tolerance: self.rel_cost_tolerance,
            max_iter: self.max_iter,
            bounds: self.bounds.clone(),
            seed_config: self.seed_config,
            rho_bound: self.rho_bound,
            heuristic_lambdas: self.heuristic_lambdas.clone(),
            initial_rho: self.initial_rho.clone(),
            fallback_policy: self.fallback_policy,
            screening_cap: self.screening_cap.clone(),
            screen_initial_rho: self.screen_initial_rho,
            outer_inner_cap: self.outer_inner_cap.clone(),
            operator_initial_trust_radius: self.operator_initial_trust_radius,
            arc_initial_regularization: self.arc_initial_regularization,
            objective_scale: self.objective_scale,
            bfgs_step_cap: self.bfgs_step_cap,
            bfgs_step_cap_psi: self.bfgs_step_cap_psi,
            cache_session: self.cache_session.clone(),
            cache_mirror_sessions: self.cache_mirror_sessions.clone(),
            rho_uncertainty_problem_size: self.rho_uncertainty_problem_size,
            // Cold by construction. The persistent-cache resume path in `run`
            // flips this to `true` only after a warm-start cache *hit* installs
            // a near-optimal seed; every other entry point keeps the cold-start
            // continuation pre-warm budget byte-for-byte.
            warm_start_cache_hit: false,
            // Populated only by the persistent-cache resume path in `run` after
            // a warm-start hit decodes a converged outer Hessian; cold by
            // construction here, like `warm_start_cache_hit`.
            warm_start_outer_hessian: None,
            rho_canonical_keys: self.rho_canonical_keys.clone(),
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
            screening_proxy_fn: None::<fn(&mut S, &Array1<f64>) -> Result<f64, EstimationError>>,
            seed_fn: None::<fn(&mut S, &Array1<f64>) -> Result<SeedOutcome, EstimationError>>,
            continuation_prewarm: self.continuation_prewarm,
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
            screening_proxy_fn: None::<fn(&mut S, &Array1<f64>) -> Result<f64, EstimationError>>,
            seed_fn: None::<fn(&mut S, &Array1<f64>) -> Result<SeedOutcome, EstimationError>>,
            continuation_prewarm: self.continuation_prewarm,
        }
    }

    /// Construct a [`ClosureObjective`] with both an order-aware evaluation
    /// hook and a custom seed-screening ranking proxy. The proxy fires only
    /// when the cascade in `rank_seeds_with_screening` calls it; outside
    /// screening the regular cost path is unaffected.
    pub fn build_objective_with_screening_proxy<S, Fc, Fe, Feo, Fr, Fefs, Fsp>(
        &self,
        state: S,
        cost_fn: Fc,
        eval_fn: Fe,
        eval_order_fn: Feo,
        reset_fn: Option<Fr>,
        efs_fn: Option<Fefs>,
        screening_proxy_fn: Fsp,
    ) -> ClosureObjective<S, Fc, Fe, Fr, Fefs, Feo, Fsp>
    where
        Fc: FnMut(&mut S, &Array1<f64>) -> Result<f64, EstimationError>,
        Fe: FnMut(&mut S, &Array1<f64>) -> Result<OuterEval, EstimationError>,
        Feo: FnMut(&mut S, &Array1<f64>, OuterEvalOrder) -> Result<OuterEval, EstimationError>,
        Fr: FnMut(&mut S),
        Fefs: FnMut(&mut S, &Array1<f64>) -> Result<EfsEval, EstimationError>,
        Fsp: FnMut(&mut S, &Array1<f64>) -> Result<f64, EstimationError>,
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
            screening_proxy_fn: Some(screening_proxy_fn),
            seed_fn: None::<fn(&mut S, &Array1<f64>) -> Result<SeedOutcome, EstimationError>>,
            continuation_prewarm: self.continuation_prewarm,
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
            install_objective_domain(
                &mut config,
                self.n_params,
                objective_lower,
                objective_upper,
            )?;
        }
        let Some(session) = config.cache_session.clone() else {
            return run_outer(obj, &config, context);
        };
        let key_hex = session.key().to_hex();
        let short_key = &key_hex[..8.min(key_hex.len())];
        let mut had_hit = false;
        let mut cached_inner_beta: Option<Array1<f64>> = None;
        if let Some(loaded) = session.try_load_with_source() {
            match classify_cache_entry_for_outer(&loaded, self.n_params) {
                CacheSeedDecision::ExactFinal {
                    rho,
                    beta,
                    iterations,
                    prior_obj_display,
                } => {
                    log::info!(
                        "[CACHE] final-hit key={}.. context={} rho_dim={} prior_obj={:.6e} iter={} action=resume-and-recertify",
                        short_key,
                        context,
                        rho.len(),
                        prior_obj_display,
                        iterations,
                    );
                    config.initial_rho = Some(rho);
                    config.screen_initial_rho = false;
                    cached_inner_beta = (!beta.is_empty()).then(|| Array1::from_vec(beta));
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
                        log::info!(
                            "[CACHE] hit  key={}.. context={} rho_dim={} beta_dim={} prior_obj={:.6e} iter={}",
                            short_key,
                            context,
                            rho.len(),
                            beta_len,
                            prior_obj_display,
                            iteration,
                        );
                        config.initial_rho = Some(rho);
                        config.screen_initial_rho = false;
                        had_hit = true;
                    } else {
                        log::info!(
                            "[CACHE] hit  key={}.. context={} rho_dim={} beta_dim={} already-aligned prior_obj={:.6e}",
                            short_key,
                            context,
                            rho.len(),
                            beta_len,
                            prior_obj_display,
                        );
                        had_hit = true;
                    }
                    cached_inner_beta = beta_arr;
                }
                CacheSeedDecision::Discard {
                    reason: "payload-shape-mismatch",
                    ..
                } => {
                    log::info!(
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
                    log::info!(
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
            log::info!(
                "[CACHE] miss key={}.. context={} reason=fresh-fingerprint n_params={}",
                short_key,
                context,
                self.n_params,
            );
        }
        // Propagate the warm-start cache-hit signal into the config the runner
        // sees. On a hit the installed seed (ρ, and since 0.1.204 the inner β)
        // is already near-optimal, so the continuation pre-warm — which exists
        // purely to anneal a COLD seed — is redundant and is skipped downstream
        // (`run_plan::continuation_prewarm_step_budget`). The outer BFGS/Newton
        // still runs to its REML/KKT certificate, so the optimum is identical.
        config.warm_start_cache_hit = had_hit;
        let mut checkpointing = CheckpointingObjective::new(
            obj,
            Arc::clone(&session),
            config.cache_mirror_sessions.clone(),
        );
        // Inject the cached inner β (when present) so the family's PIRLS
        // opens at the prior converged iterate. Families that don't expose
        // a β slot inherit the trait's no-op default and silently ignore
        // the hint — that's a ρ-only resume, identical to the pre-β-cache
        // behavior, but never a regression. Families that DO expose β
        // (PIRLS-based GAMs, custom-family marginal slope, …) override
        // `seed_inner_state` to install β before the first eval.
        if let Some(beta) = cached_inner_beta.as_ref() {
            match checkpointing.seed_inner_state(beta) {
                Ok(SeedOutcome::Installed) => log::info!(
                    "[CACHE] beta-warm key={}.. context={} beta_dim={} action=installed",
                    short_key,
                    context,
                    beta.len(),
                ),
                Ok(SeedOutcome::NoSlot) => log::warn!(
                    "[CACHE] beta-warm key={}.. context={} beta_dim={} action=skip \
                     reason=objective_has_no_inner_beta_slot",
                    short_key,
                    context,
                    beta.len(),
                ),
                Ok(SeedOutcome::Incompatible) => log::info!(
                    // Not a warning: a row-relaxed cross-fit prefix seed
                    // (`cache_seed_key`) legitimately carries a β whose length
                    // reflects the PARENT fold's realized basis rank, which
                    // differs from this fold's per-block widths. The ρ seed is
                    // kept (already installed above); cross-length β transfer is
                    // the gauge-projected FitArtifact channel's job. This is a
                    // clean ρ-only resume, NOT a regression to full cold start.
                    "[CACHE] beta-warm key={}.. context={} beta_dim={} action=rho-only \
                     reason=seed_beta_length_incompatible_with_inner_blocks",
                    short_key,
                    context,
                    beta.len(),
                ),
                Err(err) => log::warn!(
                    "[CACHE] beta-warm key={}.. context={} beta_dim={} action=skip err={}",
                    short_key,
                    context,
                    beta.len(),
                    err,
                ),
            }
        }
        let result = run_outer(&mut checkpointing, &config, context);
        // Pull the most-recent inner β surfaced by the inner solver so the
        // finalize write encodes the (ρ, β) pair the BFGS optimum was
        // actually fitted at, not a ρ-only seed that resumes at cold β.
        let final_beta = checkpointing.last_inner_beta();
        if let Ok(result) = result.as_ref()
            && result.final_value.is_finite()
            && result.converged
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
                log::info!(
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
                    log::info!(
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
}

/// Internal outcome of one planned solver/multistart attempt.
///
/// Exhausted checkpoints carry resumable work only. They never pass through
/// finalization, cache promotion, uncertainty diagnostics, or fitted-model
/// construction.
pub(crate) enum PlanRunOutcome {
    Converged(OuterResult),
    Exhausted(OuterResult),
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
    /// flat certificate band — the score-relative stationarity bound
    /// (`flat_valley_converged_grad_bound`), the probe-noise-floor bound
    /// measured from the stall window's own value scatter, and/or the
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
        }
    }
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
    /// Final gradient when the solver is gradient-based.
    pub final_gradient: Option<Array1<f64>>,
    /// Final Hessian when the solver tracks one.
    pub final_hessian: Option<Array2<f64>>,
    /// Whether the optimizer converged to a stationary point.
    pub converged: bool,
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
    pub operator_stop_reason: Option<OperatorTrustRegionStopReason>,
    /// First-order optimality self-audit at the returned point (#934).
    ///
    /// `None` when no analytic gradient was measured at termination
    /// (gradient-free solvers, cache-hit short-circuits, per-atom EFS) or
    /// when an audit probe failed to evaluate. Populated once by
    /// [`run_outer`] after the solver ladder returns, outside all hot loops.
    pub criterion_certificate: Option<OuterCriterionCertificate>,
    /// Which certificate concluded a converged run (#2235/#2241). Stamped by
    /// [`certify_outer_optimality`] on every certified result (the
    /// Fellner–Schall lane pre-stamps `RecurrentIncumbent`, which certification
    /// preserves); `None` exactly on non-converged resume checkpoints.
    pub converged_via: Option<OuterConvergedVia>,
    /// Probe-noise-floor gradient bound measured by the cost-stall guard at a
    /// halted stall (#2241): σ̂/Δ, the criterion's evaluation-noise floor over
    /// the stall window divided by the radius the accepted steps actually
    /// probed. Present only on results rebuilt from a cost-stall exit;
    /// [`certify_outer_optimality`] folds it into the stationarity bound so the
    /// final re-measured gradient is judged against the same flat certificate
    /// the guard granted.
    pub flat_noise_grad_bound: Option<f64>,
    /// Post-fit PSIS diagnostic for whether sampled smoothing-parameter weights
    /// show evidence that plug-in REML/LAML intervals are unreliable. Populated
    /// once by [`run_outer`] when the exact rho Hessian is cheap enough to use.
    pub rho_uncertainty_diagnostic: Option<crate::rho_uncertainty::RhoUncertaintyDiagnostic>,
}

impl OuterResult {
    pub fn new(
        rho: Array1<f64>,
        final_value: f64,
        iterations: usize,
        converged: bool,
        plan_used: OuterPlan,
    ) -> Self {
        Self {
            rho,
            final_value,
            iterations,
            final_grad_norm: None,
            final_gradient: None,
            final_hessian: None,
            converged,
            plan_used,
            operator_trust_radius: None,
            operator_stop_reason: None,
            criterion_certificate: None,
            converged_via: None,
            flat_noise_grad_bound: None,
            rho_uncertainty_diagnostic: None,
        }
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
pub fn audit_stationary_point(
    obj: &mut dyn OuterObjective,
    rho: Array1<f64>,
    context: &str,
) -> Result<OuterResult, OuterStationaryPointRejection> {
    let config = OuterConfig::default();
    let selected_plan = plan(&obj.capability());
    // There is intentionally no independent value-only probe. The analytic
    // sample is the authority being audited, and infinity records that no
    // optimizer-produced terminal value exists to compare against it.
    let mut result = OuterResult::new(rho, f64::INFINITY, 0, false, selected_plan);
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
// objective, per SPEC rule 2 (finite differences never run outside tests;
// the FD gradient oracle now lives in the test-only `fd_audit` module): the
// KKT-projected analytic gradient norm against the same score-relative
// stationarity bound the outer loop already uses to accept flat-valley
// stalls (#1690), a scaled PSD probe of the tracked outer Hessian, and the
// λ-rail facts every desync postmortem asks for. It is the runtime
// enforcement layer for the criterion-atom architecture (#931).
//
// A failed certificate REJECTS the fit as typed non-convergence — never a
// warn-and-continue diagnostic — so a nonstationary point can never be
// minted into a fit (SPEC rule 20).

/// Cholesky positive-SEMIdefiniteness probe for the (small, outer-dim) final
/// Hessian, with a roundoff-scale diagonal shift. Returns `None` when the
/// matrix is empty, non-square, or non-finite; `Some(false)` when the shifted
/// matrix has a non-positive pivot — i.e. the curvature is genuinely
/// indefinite, not merely semidefinite-within-noise.
///
/// The shift is `√ε · max(1, max|H_ii|)`: eigenvalues assembled through
/// O(‖H‖)-scaled arithmetic carry O(ε·‖H‖) roundoff, so a `√ε`-relative
/// margin cleanly separates a true negative direction from accumulated
/// floating-point noise on a flat (near-semidefinite) valley.
pub(crate) fn certificate_hessian_is_psd(hessian: &Array2<f64>) -> Option<bool> {
    let n = hessian.nrows();
    if n == 0 || hessian.ncols() != n || hessian.iter().any(|v| !v.is_finite()) {
        return None;
    }
    let max_diag = (0..n).fold(0.0_f64, |acc, j| acc.max(hessian[[j, j]].abs()));
    let shift = f64::EPSILON.sqrt() * max_diag.max(1.0);
    let mut chol = hessian.clone();
    for j in 0..n {
        chol[[j, j]] += shift;
    }
    for j in 0..n {
        for k in 0..j {
            let l_jk = chol[[j, k]];
            for i in j..n {
                chol[[i, j]] -= chol[[i, k]] * l_jk;
            }
        }
        let pivot = chol[[j, j]];
        if !(pivot > 0.0) || !pivot.is_finite() {
            return Some(false);
        }
        let inv_sqrt = 1.0 / pivot.sqrt();
        for i in j..n {
            chol[[i, j]] *= inv_sqrt;
        }
    }
    Some(true)
}

/// Second-order predicted objective decrease of a safeguarded Newton step at a
/// flat-valley cost-stall exit (#2253/#2249/#2015).
///
/// On a flat-valley cost-stall exit the outer criterion has provably stopped
/// improving (the cost-stall window fired), yet the re-measured projected
/// gradient can sit modestly above the score-relative flat band on a
/// weakly-identified small-n fit (measured: |Pg| ≈ 0.072 vs a score-relative
/// band ≈ 0.053 on an n=84/p=64 K=1 circle). Whether that residual is genuine
/// available descent is a SECOND-ORDER question. The improvement a safeguarded
/// Newton step buys is the Newton decrement over two:
///
///     Δpred = ½ · gᵀ H⁻¹ g,
///
/// the textbook Newton stopping quantity (Boyd–Vandenberghe §9.5). When `Δpred`
/// is below the outer objective tolerance, no step can reduce the criterion by
/// more than that tolerance and the point is stationary at the resolution the
/// criterion can be optimized — the mathematically correct "no further descent
/// possible" criterion.
///
/// This is curvature-scaled, not a constant: because `H⁻¹` weights each gradient
/// component by the inverse eigenvalue, a residual aligned with a NEAR-FLAT
/// Hessian eigenvector (a linear ramp that DOES carry real descent) inflates
/// `gᵀ H⁻¹ g` toward the roundoff-regularized `|g_flat|² / shift` and is
/// REJECTED; only a residual that is small along the well-curved directions and
/// nearly orthogonal to the flat ones certifies. An indefinite Hessian never
/// reaches here — the certificate's curvature gate (`certificate_hessian_is_psd`)
/// rejects a genuinely indefinite point independently, and this factorization
/// returns `None` on a non-PSD shifted factor so the caller falls back to the
/// gradient-only bound.
///
/// `hessian` and `grad` are the analytic outer Hessian and the KKT-PROJECTED
/// gradient at the certified point. The shift `√ε · max|H_jj|` matches
/// [`certificate_hessian_is_psd`] so the definiteness verdict and this decrement
/// agree on the same regularized operator. Returns `None` when the shapes are
/// malformed, an entry is non-finite, the shifted factor is not PD, or the
/// resulting quadratic form is negative (which a PD factor rules out; retained
/// as a roundoff guard).
pub(crate) fn newton_predicted_decrease(hessian: &Array2<f64>, grad: &Array1<f64>) -> Option<f64> {
    let n = hessian.nrows();
    if n == 0 || hessian.ncols() != n || grad.len() != n {
        return None;
    }
    if hessian.iter().any(|v| !v.is_finite()) || grad.iter().any(|v| !v.is_finite()) {
        return None;
    }
    let max_diag = (0..n).fold(0.0_f64, |acc, j| acc.max(hessian[[j, j]].abs()));
    let shift = f64::EPSILON.sqrt() * max_diag.max(1.0);
    // Lower Cholesky factor L of H + shift·I (same regularization the PSD probe
    // uses), computed in place.
    let mut l = hessian.clone();
    for j in 0..n {
        l[[j, j]] += shift;
    }
    for j in 0..n {
        for k in 0..j {
            let l_jk = l[[j, k]];
            for i in j..n {
                l[[i, j]] -= l[[i, k]] * l_jk;
            }
        }
        let pivot = l[[j, j]];
        if !(pivot > 0.0) || !pivot.is_finite() {
            return None;
        }
        let inv_sqrt = 1.0 / pivot.sqrt();
        for i in j..n {
            l[[i, j]] *= inv_sqrt;
        }
    }
    // Solve (L Lᵀ) d = g for d = H_s⁻¹ g: forward-substitute L y = g, then
    // back-substitute Lᵀ d = y.
    let mut y = grad.clone();
    for j in 0..n {
        let mut s = y[j];
        for k in 0..j {
            s -= l[[j, k]] * y[k];
        }
        y[j] = s / l[[j, j]];
    }
    let mut d = y;
    for j in (0..n).rev() {
        let mut s = d[j];
        for k in (j + 1)..n {
            s -= l[[k, j]] * d[k];
        }
        d[j] = s / l[[j, j]];
    }
    let quad = grad.dot(&d); // gᵀ H_s⁻¹ g ≥ 0 for a PD factor.
    if !quad.is_finite() || quad < 0.0 {
        return None;
    }
    Some(0.5 * quad)
}

/// Smoothing coordinates (leading ρ block) railed against the outer box.
pub(crate) fn certificate_railed_lambdas(
    rho: &Array1<f64>,
    rho_dim: usize,
    config: &OuterConfig,
) -> Vec<usize> {
    (0..rho_dim.min(rho.len()))
        .filter(|&k| {
            let (lo, hi) = match config.bounds.as_ref() {
                Some((lo, hi)) if k < lo.len() && k < hi.len() => (lo[k], hi[k]),
                Some(_) => return false,
                None => (-config.rho_bound, config.rho_bound),
            };
            (rho[k] - lo).abs() <= CERTIFICATE_RAIL_MARGIN
                || (hi - rho[k]).abs() <= CERTIFICATE_RAIL_MARGIN
        })
        .collect()
}

fn outer_nonconvergence_error(
    context: &str,
    reason: &str,
    result: &OuterResult,
    projected_grad_norm: Option<f64>,
    stationarity_bound: f64,
) -> EstimationError {
    EstimationError::RemlDidNotConverge {
        context: context.to_string(),
        reason: reason.to_string(),
        iterations: result.iterations,
        final_value: result.final_value,
        projected_grad_norm,
        stationarity_bound,
        rho_checkpoint: result.rho.to_vec(),
    }
}

fn certify_fixed_point_optimality(
    obj: &mut dyn OuterObjective,
    config: &OuterConfig,
    context: &str,
    result: &mut OuterResult,
) -> Result<OuterCriterionCertificate, EstimationError> {
    let layout = obj.capability().theta_layout();
    let evaluation = obj
        .eval_fixed_point_certificate(&result.rho)
        .map_err(|err| {
            outer_nonconvergence_error(
                context,
                &format!("analytic fixed-point certificate evaluation failed: {err}"),
                result,
                None,
                config.tolerance,
            )
        })?;
    if !inner_solve_converged(config.outer_inner_cap.as_ref()) {
        return Err(outer_nonconvergence_error(
            context,
            "terminal fixed-point evidence was evaluated at a non-converged inner state",
            result,
            None,
            config.tolerance,
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
            config.tolerance,
        ));
    }
    if !evaluation.cost.is_finite() {
        return Err(outer_nonconvergence_error(
            context,
            "fixed-point certificate returned a non-finite objective value",
            result,
            None,
            config.tolerance,
        ));
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
                    "coordinate {index} has invalid covered residual update={update} scale={scale}"
                ));
                normalized_updates.push(f64::NAN);
            }
            FixedPointCoordinateCertificate::Uncovered { reason } => {
                uncovered.push(format!("coordinate {index}: {reason}"));
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
            config.tolerance,
        ));
    }

    let (lower, upper) = outer_bounds_template(config, layout.n_params);
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
    result.final_gradient = None;
    result.final_hessian = None;
    result.converged = false;

    let certificate = OuterCriterionCertificate {
        stationarity: OuterStationarityCertificate::FixedPoint {
            residual_inf_norm: raw_inf,
            projected_residual_inf_norm: projected_inf,
            bound: config.tolerance,
            covered_coordinates: layout.n_params,
        },
        hessian_psd: None,
        lambdas_railed: certificate_railed_lambdas(&result.rho, layout.rho_dim(), config),
    };
    result.criterion_certificate = Some(certificate.clone());
    if !certificate.certifies() {
        return Err(outer_nonconvergence_error(
            context,
            &certificate.summary(),
            result,
            Some(projected_inf),
            config.tolerance,
        ));
    }

    result.converged = true;
    result.converged_via = match result.converged_via {
        Some(via @ OuterConvergedVia::RecurrentIncumbent { .. }) => Some(via),
        _ => Some(OuterConvergedVia::FixedPointStationary {
            projected_residual_inf_norm: projected_inf,
            certificate_bound: config.tolerance,
        }),
    };
    log::info!("[CERTIFICATE] {context}: {}", certificate.summary());
    Ok(certificate)
}

/// Build the mandatory analytic optimality certificate at the returned point.
///
/// The objective is evaluated once at the selected point through its analytic
/// derivative path. Missing, malformed, or non-finite derivative evidence is
/// non-convergence: an optimizer status bit cannot substitute for a
/// stationarity certificate. Exact analytic curvature is checked when the
/// objective declares it and can materialize it; BFGS/EFS solver geometry is
/// never mistaken for objective curvature.
pub(crate) fn certify_outer_optimality(
    obj: &mut dyn OuterObjective,
    config: &OuterConfig,
    context: &str,
    result: &mut OuterResult,
) -> Result<OuterCriterionCertificate, EstimationError> {
    let terminal_cap_guard = config
        .outer_inner_cap
        .as_ref()
        .map(TerminalInnerCapGuard::lift);
    if terminal_cap_guard.is_some() {
        // `reset` is deliberately conditional on the presence of the cap
        // contract.  Those are the REML/mixture objectives whose search cache
        // can contain a coarse inner state; uncapped objectives retain their
        // ordinary stateful certification semantics.
        obj.reset();
    }
    let outcome = certify_outer_optimality_at_terminal_fidelity(obj, config, context, result);
    drop(terminal_cap_guard);
    outcome
}

fn certify_outer_optimality_at_terminal_fidelity(
    obj: &mut dyn OuterObjective,
    config: &OuterConfig,
    context: &str,
    result: &mut OuterResult,
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
    if result.rho.iter().any(|value| !value.is_finite()) {
        return Err(outer_nonconvergence_error(
            context,
            "the selected checkpoint contains non-finite coordinates",
            result,
            None,
            outer_gradient_tolerance(config).abs,
        ));
    }
    if layout.n_params == 0 {
        let value = obj.eval_cost(&result.rho).map_err(|err| {
            outer_nonconvergence_error(
                context,
                &format!("zero-dimensional final objective evaluation failed: {err}"),
                result,
                Some(0.0),
                outer_gradient_tolerance(config).abs,
            )
        })?;
        if !value.is_finite() {
            return Err(outer_nonconvergence_error(
                context,
                "the zero-dimensional final objective is non-finite",
                result,
                Some(0.0),
                outer_gradient_tolerance(config).abs,
            ));
        }
        let certificate = OuterCriterionCertificate {
            stationarity: OuterStationarityCertificate::AnalyticGradient {
                grad_norm: 0.0,
                projected_grad_norm: 0.0,
                bound: outer_gradient_tolerance(config).abs,
            },
            hessian_psd: None,
            lambdas_railed: Vec::new(),
        };
        result.final_value = value;
        result.final_grad_norm = Some(0.0);
        result.final_gradient = Some(Array1::zeros(0));
        result.final_hessian = None;
        result.converged = true;
        result.converged_via = Some(OuterConvergedVia::GradientStationary);
        result.criterion_certificate = Some(certificate.clone());
        return Ok(certificate);
    }
    if matches!(result.plan_used.solver, Solver::Efs | Solver::HybridEfs)
        && capability.gradient != Derivative::Analytic
    {
        return certify_fixed_point_optimality(obj, config, context, result);
    }
    if capability.gradient != Derivative::Analytic {
        return Err(outer_nonconvergence_error(
            context,
            "the objective exposes no analytic gradient for final certification",
            result,
            None,
            outer_gradient_tolerance(config).abs,
        ));
    }

    let order = if capability.hessian.is_analytic() {
        OuterEvalOrder::ValueGradientHessian
    } else {
        OuterEvalOrder::ValueAndGradient
    };
    let evaluation = obj.eval_with_order(&result.rho, order).map_err(|err| {
        outer_nonconvergence_error(
            context,
            &format!("analytic final-point evaluation failed: {err}"),
            result,
            result.final_grad_norm,
            outer_gradient_tolerance(config).abs,
        )
    })?;
    if !inner_solve_converged(config.outer_inner_cap.as_ref()) {
        return Err(outer_nonconvergence_error(
            context,
            "terminal analytic evidence was evaluated at a non-converged inner state",
            result,
            None,
            outer_gradient_tolerance(config).abs,
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
                outer_gradient_tolerance(config).abs,
            )
        })?;
    if !evaluation.cost.is_finite() || evaluation.gradient.iter().any(|value| !value.is_finite()) {
        return Err(outer_nonconvergence_error(
            context,
            "the analytic final-point value or gradient is non-finite",
            result,
            None,
            outer_gradient_tolerance(config).abs,
        ));
    }

    let bounds = outer_bounds_template(config, layout.n_params);
    let grad_norm = evaluation.gradient.dot(&evaluation.gradient).sqrt();
    // KKT-projected gradient VECTOR (not just its norm): the norm feeds the
    // stationarity certificate below, and the vector feeds the curvature-scaled
    // flat-valley Newton decrement (#2253/#2249/#2015) once the analytic Hessian
    // is in hand.
    let projected_gradient =
        project_gradient_vector(&result.rho, &evaluation.gradient, Some(&bounds));
    let projected_grad_norm = projected_gradient.iter().map(|v| v * v).sum::<f64>().sqrt();
    let solver_bound = outer_gradient_tolerance(config).threshold(evaluation.cost, grad_norm);
    let mut stationarity_bound = if matches!(
        result.operator_stop_reason,
        Some(OperatorTrustRegionStopReason::CostStallFlatValley)
    ) {
        solver_bound.max(flat_valley_converged_grad_bound(evaluation.cost))
    } else {
        solver_bound
    };
    // #2241 — a cost-stall exit carries the guard's measured probe-noise-floor
    // gradient bound σ̂/Δ. The certificate must judge the re-measured final
    // gradient against the same flat band the guard certified, or the guard's
    // noise-scale convergence would be granted in the loop and revoked here.
    if let Some(noise_bound) = result.flat_noise_grad_bound
        && noise_bound.is_finite()
    {
        stationarity_bound = stationarity_bound.max(noise_bound);
    }

    // Install measured first-order evidence before any fallible curvature
    // processing. If curvature is malformed, the retained resume checkpoint
    // still carries the exact value/gradient that caused certification to stop.
    result.final_value = evaluation.cost;
    result.final_grad_norm = Some(projected_grad_norm);
    result.final_gradient = Some(evaluation.gradient);
    result.converged = false;

    let analytic_hessian = if capability.hessian.is_analytic() {
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
                            stationarity_bound,
                        )
                    })?;
                if hessian.iter().any(|value| !value.is_finite()) {
                    return Err(outer_nonconvergence_error(
                        context,
                        "the analytic final Hessian contains non-finite entries",
                        result,
                        Some(projected_grad_norm),
                        stationarity_bound,
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
                    stationarity_bound,
                ));
            }
            Err(err) => {
                return Err(outer_nonconvergence_error(
                    context,
                    &format!("analytic final Hessian could not be certified: {err}"),
                    result,
                    Some(projected_grad_norm),
                    stationarity_bound,
                ));
            }
        }
    } else {
        None
    };

    // Curvature-scaled stationarity (#2253/#2249/#2015/#2091). The re-measured
    // projected gradient can sit modestly ABOVE the score-relative / probe-noise
    // bands even though NO step reduces the objective by more than the outer
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
    if let Some(hessian) = analytic_hessian.as_ref()
        && let Some(predicted_decrease) = newton_predicted_decrease(hessian, &projected_gradient)
        && predicted_decrease.is_finite()
        && predicted_decrease > 0.0
    {
        // The SAME relative cost floor the cost-stall guard used to declare the
        // criterion stalled (run_plan.rs), so certification asserts nothing
        // tighter than the loop already proved about this surface.
        let objective_tol = config
            .rel_cost_tolerance
            .unwrap_or(config.tolerance * 1.0e-2)
            .max(COST_STALL_REL_TOL_FLOOR)
            * (1.0 + evaluation.cost.abs());
        let curvature_grad_bound =
            projected_grad_norm * (objective_tol / predicted_decrease).sqrt();
        if curvature_grad_bound.is_finite() && curvature_grad_bound > stationarity_bound {
            log::info!(
                "[CERTIFICATE] {context}: curvature-scaled flat-valley bound {curvature_grad_bound:.3e} \
                 (|Pg|={projected_grad_norm:.3e}, Newton ½gᵀH⁻¹g={predicted_decrease:.3e} ≤ tol {objective_tol:.3e}) \
                 widened from gradient-band {stationarity_bound:.3e}"
            );
            stationarity_bound = curvature_grad_bound;
        }
    }

    let certificate = OuterCriterionCertificate {
        stationarity: OuterStationarityCertificate::AnalyticGradient {
            grad_norm,
            projected_grad_norm,
            bound: stationarity_bound,
        },
        hessian_psd: analytic_hessian.as_ref().and_then(certificate_hessian_is_psd),
        lambdas_railed: certificate_railed_lambdas(&result.rho, layout.rho_dim(), config),
    };
    // Install the measured evidence before deciding its verdict.  A rejected
    // candidate is retained only as a resumable checkpoint, and that
    // checkpoint must carry the actual analytic residual/curvature evidence
    // that caused the rejection rather than the optimizer's stale terminal
    // status.
    result.final_hessian = analytic_hessian;
    result.criterion_certificate = Some(certificate.clone());
    if !certificate.certifies() {
        result.converged = false;
        return Err(outer_nonconvergence_error(
            context,
            &certificate.summary(),
            result,
            Some(projected_grad_norm),
            stationarity_bound,
        ));
    }

    result.converged = true;
    // #2235/#2241 — record WHICH certificate concluded this run. A
    // Fellner–Schall model-state fixed point was pre-stamped by the runner and
    // is preserved (this analytic certificate is its corroborating evidence);
    // otherwise the verdict is decided by which stationarity band the measured
    // projected gradient actually cleared: the solver's own tolerance
    // (gradient-stationary) or only the widened flat certificate band
    // (criterion-flat, #2241).
    result.converged_via = match result.converged_via {
        Some(via @ OuterConvergedVia::RecurrentIncumbent { .. }) => Some(via),
        _ if projected_grad_norm <= solver_bound => Some(OuterConvergedVia::GradientStationary),
        _ => Some(OuterConvergedVia::CriterionFlat {
            residual_grad_norm: projected_grad_norm,
            certificate_bound: stationarity_bound,
        }),
    };
    log::info!("[CERTIFICATE] {context}: {}", certificate.summary());
    Ok(certificate)
}

pub(crate) fn compute_rho_uncertainty_diagnostic(
    obj: &mut dyn OuterObjective,
    config: &OuterConfig,
    context: &str,
    result: &mut OuterResult,
) -> crate::rho_uncertainty::RhoUncertaintyDiagnostic {
    let cap = obj.capability();
    let layout = cap.theta_layout();
    let rho_dim = layout.rho_dim();
    let gate = crate::rho_uncertainty::RhoUncertaintyCostGate {
        sample_count: 32,
        problem_size: config.rho_uncertainty_problem_size,
    };
    if let Err(reason) = crate::rho_uncertainty::cost_gate_allows(rho_dim, gate) {
        return crate::rho_uncertainty::RhoUncertaintyDiagnostic::skipped(reason, 0);
    }
    if result.rho.len() != layout.n_params {
        return crate::rho_uncertainty::RhoUncertaintyDiagnostic::skipped(
            format!(
                "final outer point length {} does not match objective dimension {}",
                result.rho.len(),
                layout.n_params
            ),
            0,
        );
    }

    let final_eval = match obj.eval_with_order(&result.rho, OuterEvalOrder::ValueGradientHessian) {
        Ok(eval) => eval,
        Err(err) => {
            return crate::rho_uncertainty::RhoUncertaintyDiagnostic::skipped(
                format!("final exact Hessian evaluation failed: {err}"),
                1,
            );
        }
    };
    let hessian = match final_eval.hessian.materialize_dense() {
        Ok(Some(hessian)) => hessian,
        Ok(None) => {
            return crate::rho_uncertainty::RhoUncertaintyDiagnostic::skipped(
                "exact outer Hessian unavailable at fitted rho",
                1,
            );
        }
        Err(message) => {
            return crate::rho_uncertainty::RhoUncertaintyDiagnostic::skipped(
                format!("exact outer Hessian materialization failed: {message}"),
                1,
            );
        }
    };
    if hessian.nrows() != layout.n_params || hessian.ncols() != layout.n_params {
        return crate::rho_uncertainty::RhoUncertaintyDiagnostic::skipped(
            format!(
                "exact outer Hessian shape {}x{} does not match objective dimension {}",
                hessian.nrows(),
                hessian.ncols(),
                layout.n_params
            ),
            1,
        );
    }
    // Persist the exact outer curvature at θ̂ when the solver did not already
    // track one. A gradient-based BFGS solve keeps its inverse-Hessian
    // internally and `opt` does not surface it, so `result.final_hessian` is
    // `None` on the BFGS path — yet the exact analytic `H(θ̂)` was just
    // materialized here for the rho-uncertainty diagnostic and is otherwise
    // discarded. Stashing it lets the persistent-cache finalize write carry the
    // converged curvature, so the NEXT structurally-matching fit (e.g. the next
    // LOSO fold, whose θ̂ and curvature are nearly identical) can seed BFGS with
    // `InitialMetric::DenseInverseHessian` and take a quasi-Newton first step
    // instead of rediscovering curvature through line-search bracketing. This
    // never changes a converged optimum (BFGS converges to ∇V=0 under any SPD
    // initial metric); it only reshapes the starting line-search path. Guarded
    // on finiteness and on the solver not already owning a Hessian, so the
    // exact-Newton / ARC paths (which DO populate `final_hessian`) are untouched.
    if result.final_hessian.is_none() && hessian.iter().all(|v| v.is_finite()) {
        result.final_hessian = Some(hessian.clone());
    }
    let mut hessian_rho = Array2::<f64>::zeros((rho_dim, rho_dim));
    for row in 0..rho_dim {
        for col in 0..rho_dim {
            hessian_rho[[row, col]] = hessian[[row, col]];
        }
    }
    let rho_hat = result.rho.slice(ndarray::s![..rho_dim]).to_owned();
    let theta_hat = result.rho.clone();
    let cost_hat = final_eval.cost;
    let final_beta_hint = final_eval.inner_beta_hint.clone();
    let diagnostic = {
        let mut served_hat_cost = false;
        let mut criterion = |rho: &Array1<f64>| -> Option<f64> {
            let is_hat = rho.len() == rho_hat.len()
                && rho
                    .iter()
                    .zip(rho_hat.iter())
                    .all(|(&left, &right)| left.to_bits() == right.to_bits());
            if is_hat && !served_hat_cost {
                served_hat_cost = true;
                return Some(cost_hat);
            }
            let mut theta = theta_hat.clone();
            for idx in 0..rho_dim {
                theta[idx] = rho[idx];
            }
            if let Some(beta) = final_beta_hint.as_ref()
                && obj.seed_inner_state(beta).is_err()
            {
                return None;
            }
            obj.eval_cost(&theta).ok()
        };
        crate::rho_uncertainty::rho_uncertainty_diagnostic(
            &rho_hat,
            &hessian_rho,
            gate,
            &mut criterion,
        )
    };
    if let Some(beta) = final_beta_hint.as_ref()
        && let Err(err) = obj.seed_inner_state(beta)
    {
        log::debug!(
            "[RHO uncertainty] {context}: final inner-state restore skipped after diagnostic ({err})"
        );
    }
    match &diagnostic.status {
        crate::rho_uncertainty::RhoUncertaintyStatus::NoEvidenceOfHeavyTails => {
            log::info!(
                "[RHO uncertainty] {context}: no heavy-tail evidence at sampled rho proposals k_hat={:.3} evals={}",
                diagnostic.k_hat.unwrap_or(f64::NAN),
                diagnostic.n_evaluations,
            );
        }
        crate::rho_uncertainty::RhoUncertaintyStatus::HeavyTailsDetected { k_hat } => {
            log::warn!(
                "[RHO uncertainty] {context}: heavy rho-importance tail detected k_hat={:.3} evals={}",
                k_hat,
                diagnostic.n_evaluations,
            );
        }
        crate::rho_uncertainty::RhoUncertaintyStatus::Skipped { reason } => {
            log::info!("[RHO uncertainty] {context}: skipped ({reason})");
        }
    }
    diagnostic
}

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
    /// Family returned a non-operator Hessian mid-flight after routing into
    /// the operator path. Best-effort `x_k` returned with this reason; the
    /// caller should consider re-fitting under a different solver class
    /// (e.g. BFGS gradient-only) instead of trusting the partial result.
    RoutingMismatch,
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
    // optimization, and the #934 certificate / uncertainty audits) in that
    // canonical layout against a permuting wrapper, then map the result back to
    // the native layout. Seeding/tie-breaking then see byte-identical
    // coordinates for every term order, so both orders select the same λ̂.
    if let Some(keys) = config.rho_canonical_keys.as_ref()
        && let Some(perm) = canonical_permutation(keys)
    {
        let canonical_config = canonicalize_outer_config(config, &perm);
        let mut canonical_obj = CanonicalizedObjective::new(obj, perm.clone());
        let result = run_outer(&mut canonical_obj, &canonical_config, context)?;
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
        exact_config.heuristic_lambdas = None;
        exact_config.seed_config.max_seeds = 1;
        exact_config.seed_config.seed_budget = 1;
        exact_config.screen_initial_rho = false;
        exact_config.warm_start_cache_hit = true;
        exact_config.operator_initial_trust_radius = result.operator_trust_radius;
        exact_config.warm_start_outer_hessian = result.final_hessian.clone();
        log::info!(
            "[OUTER] {context}: sampled derivative pilot completed after {} iteration(s); \
             continuing from its checkpoint on the exact full-data measure",
            pilot_iterations,
        );
        let mut polished = run_outer_uncertified(obj, &exact_config, context)?;
        polished.iterations = polished.iterations.saturating_add(pilot_iterations);
        result = polished;
    }
    // Mandatory analytic optimality certificate (#934): once at the selected
    // point, outside every hot loop, for every solver path and every iteration
    // budget. Missing or failed evidence is typed non-convergence; there is no
    // max-iteration or logging-level bypass.
    //
    // #2273 STALE-TOLERANCE DESYNC RETRY. The solver's in-loop convergence
    // threshold is resolved ONCE from the SEED's cost scale
    // (`rel_cost·(1+|seed_cost|)`), while this certificate re-derives the
    // same formula at the terminal point's own (often far smaller) cost — on
    // a perfectly-separated binomial the score plunges between the
    // oversmoothed heuristic seed and the first accepted step, so the solver
    // can declare victory against a bound orders of magnitude looser than
    // the one that then refuses it here (measured: |g|=8.1e-1 accepted
    // in-loop vs bound 8.3e-3 at certification, 'NOT STATIONARY after 1
    // outer iteration'; the pass/fail pattern was non-monotone in n because
    // it tracked the seed-to-terminus cost ratio, not identifiability). The
    // desync exists precisely because the tolerance anchor differs from the
    // terminus, so ONE re-run seeded AT the refused checkpoint removes it by
    // construction: the retry's seed cost IS the certificate's cost, its
    // in-loop bound equals the certificate bound, and the solver either
    // genuinely closes the remaining gradient gap or exhausts its budget and
    // takes the same typed refusal as before. Bounded to a single retry;
    // only fires when the solver CLAIMED convergence (a budget-exhausted
    // result is not a desync — its refusal is genuine).
    let claimed_converged = result.converged;
    result.criterion_certificate = match certify_outer_optimality(obj, config, context, &mut result)
    {
        Ok(certificate) => Some(certificate),
        Err(refusal) if claimed_converged => {
            let prior_iterations = result.iterations;
            log::info!(
                "[OUTER] {context}: solver convergence claim failed analytic \
                     certification after {prior_iterations} iteration(s); re-running once \
                     seeded at the refused checkpoint so the in-loop tolerance anchors to \
                     the terminal cost scale (#2273 stale-tolerance desync)"
            );
            let mut retry_cfg = config.clone();
            retry_cfg.initial_rho = Some(result.rho.clone());
            retry_cfg.heuristic_lambdas = None;
            retry_cfg.seed_config.max_seeds = 1;
            retry_cfg.seed_config.seed_budget = 1;
            retry_cfg.screen_initial_rho = false;
            retry_cfg.warm_start_cache_hit = true;
            retry_cfg.operator_initial_trust_radius = result.operator_trust_radius;
            retry_cfg.warm_start_outer_hessian = result.final_hessian.clone();
            obj.reset();
            match run_outer_uncertified(obj, &retry_cfg, context) {
                Ok(mut retried) => {
                    retried.iterations = retried.iterations.saturating_add(prior_iterations);
                    result = retried;
                    Some(certify_outer_optimality(obj, config, context, &mut result)?)
                }
                // The retry could not even run (e.g. the checkpoint is a
                // hard refusal wall for the objective): surface the
                // ORIGINAL certification refusal, which carries the
                // checkpoint evidence.
                Err(_) => return Err(refusal),
            }
        }
        Err(refusal) => return Err(refusal),
    };
    result.rho_uncertainty_diagnostic = Some(compute_rho_uncertainty_diagnostic(
        obj,
        config,
        context,
        &mut result,
    ));
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
    if let Some(initial) = config.initial_rho.as_ref() {
        canonical.initial_rho = Some(permute_arr(initial));
    }
    if let Some(h) = config.heuristic_lambdas.as_ref() {
        canonical.heuristic_lambdas = Some(permute_vec(h));
    }
    if let Some((lower, upper)) = config.bounds.as_ref() {
        canonical.bounds = Some((permute_arr(lower), permute_arr(upper)));
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
    if let Some(initial_rho) = config.initial_rho.as_ref() {
        cap.theta_layout()
            .validate_point_len(initial_rho, "initial outer seed")
            .map_err(|err| match err {
                ObjectiveEvalError::Recoverable { message }
                | ObjectiveEvalError::Fatal { message } => {
                    EstimationError::RemlOptimizationFailed(format!("{context}: {message}"))
                }
            })?;
    }
    crate::estimate::reml::outer_eval::clear_outer_ift_residual_energy_for_fit();

    // Frontier ρ-scaling auto-switch (#986): at per-atom-EFS-eligible frontier
    // rho dimension the decoupled per-atom fixed point is the primary outer
    // iteration; everything else falls through to the dense / standard path
    // below. Routed here so every entry point inherits it (magic by default).
    if let Some(result) = run_per_atom_efs_if_frontier(obj, config, context)? {
        if result.converged {
            return Ok(result);
        }
        return Err(outer_nonconvergence_error(
            context,
            "per-atom EFS exhausted its iteration budget before the fixed-point step converged",
            &result,
            None,
            outer_gradient_tolerance(config).abs,
        ));
    }

    if cap.n_params == 0 {
        let cost = obj.eval_cost(&Array1::zeros(0))?;
        let the_plan = plan(&cap);
        return Ok(outer_result_with_gradient_norm(
            Array1::zeros(0),
            cost,
            0,
            Some(0.0),
            true,
            the_plan,
        ));
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
    let mut best_checkpoint: Option<OuterResult> = None;

    for (attempt_idx, attempt_cap) in attempts.iter().enumerate() {
        let the_plan = plan(attempt_cap);
        if attempt_idx > 0 {
            log::debug!("[OUTER] {context}: primary plan failed; falling back to {the_plan}");
        }
        log_plan(context, attempt_cap, &the_plan);

        obj.reset();

        // ARC budget-exhaustion retry: when an Arc attempt runs out of
        // outer iterations, reseed a fresh Arc run from the previous
        // attempt's last ρ and trust radius. Inner caches (PIRLS LRU,
        // eval bundle, warm-start predictor, adaptive signals) are wiped
        // by `obj.reset()`; the operator-TR's Cauchy/Newton/CG state has
        // no resume API and is not preserved. The lever that changes for
        // the resumed run is the inner-PIRLS cap (uncapped via the
        // feedback handle), not `max_iter` — empirically the prior stall
        // was an inner-tolerance / model-fidelity issue, not an outer
        // budget shortfall, and doubling `max_iter` only replays the
        // same trajectory byte-for-byte. The retry is gated on observed
        // `‖g‖` progress so trajectories that made no headway fall
        // through to the degraded plan instead of replaying.
        let mut arc_retries_left: u32 = if matches!(the_plan.solver, Solver::Arc) {
            2
        } else {
            0
        };
        let mut retry_config: Option<OuterConfig> = None;
        // Tracks the previous ARC attempt's terminal `‖g‖`. The retry
        // gate compares attempt-over-attempt: if a retry didn't move
        // the gradient norm, the trajectory replayed (same seed, same
        // trust radius, cold caches, deterministic optimizer) and
        // further retries cannot help. First retry is unconditional
        // (no prior attempt to compare against).
        let mut prev_attempt_grad_norm: Option<f64> = None;

        let outcome = loop {
            // Bind the active config by cloning into a local owned value so
            // subsequent retry-config assignment does not collide with the
            // borrow used inside this iteration body.
            let active_config_owned: OuterConfig =
                retry_config.clone().unwrap_or_else(|| config.clone());
            let active_config: &OuterConfig = &active_config_owned;
            match run_outer_with_plan(obj, active_config, context, attempt_cap, &the_plan) {
                Ok(PlanRunOutcome::Converged(result)) => break Ok(result),
                Ok(PlanRunOutcome::Exhausted(result)) => {
                    if arc_retries_left == 0
                        || matches!(
                            result.operator_stop_reason,
                            Some(
                                OperatorTrustRegionStopReason::RejectFloor
                                    // #1690: a flat-valley cost-stall is a CONVERGED
                                    // cost plateau over the whole stall window, not a
                                    // budget shortfall. The ARC retry only reseeds
                                    // from the same last ρ with a reset trust radius
                                    // and the same deterministic operator state, so it
                                    // replays the identical trajectory and re-halts at
                                    // the same valley floor with the same |g| (verified
                                    // on the #1690 Gamma repro: two retries, each
                                    // returning |g|=0.3646 byte-for-byte). Treat it
                                    // like `RejectFloor` and stop — the genuine
                                    // stationarity verdict is reconciled downstream
                                    // against the authoritative shipped-β gradient
                                    // (`optimizer.rs`), and a non-stationary floor is
                                    // still reported non-converged. This skips the
                                    // wasted full-trajectory replay that dominated the
                                    // count-family slowdown.
                                    | OperatorTrustRegionStopReason::CostStallFlatValley
                            )
                        )
                    {
                        break Ok(result);
                    }
                    // Gate the retry on attempt-over-attempt `‖g‖`
                    // progress. The first retry is unconditional (no
                    // prior attempt). Subsequent retries fall through
                    // to the degraded plan when the gradient norm did
                    // not materially shrink — the deterministic
                    // optimizer with the same seed and trust radius
                    // would replay the same trajectory.
                    let Some(cur_grad_norm) = result.final_grad_norm else {
                        log::info!(
                            "[OUTER] {context}: ARC attempt exhausted budget at \
                             iter={} cost={:.6e} without a final gradient norm; \
                             falling through to degraded plan",
                            result.iterations,
                            result.final_value,
                        );
                        break Ok(result);
                    };
                    if let Some(prev_g) = prev_attempt_grad_norm {
                        let progressed = cur_grad_norm.is_finite()
                            && prev_g.is_finite()
                            && cur_grad_norm < 0.5 * prev_g;
                        if !progressed {
                            log::info!(
                                "[OUTER] {context}: ARC retry stalled at \
                                 iter={} cost={:.6e} |g|={:.6e} (prev |g|={:.6e}); \
                                 deterministic replay suspected, falling through \
                                 to degraded plan",
                                result.iterations,
                                result.final_value,
                                cur_grad_norm,
                                prev_g,
                            );
                            break Ok(result);
                        }
                    }
                    let next_trust_radius =
                        sanitized_operator_trust_restart_radius(result.operator_trust_radius);
                    log::info!(
                        "[OUTER] {context}: ARC attempt exhausted budget at \
                         iter={} cost={:.6e} |g|={:.6e}; resuming from last \
                         rho + trust_radius={:?}, inner-PIRLS uncapped \
                         (objective caches wiped; operator-TR Cauchy/Newton \
                         state is not resumable)",
                        result.iterations,
                        result.final_value,
                        cur_grad_norm,
                        next_trust_radius,
                    );
                    // Snapshot the cap-feedback handle before we
                    // reassign `retry_config` (which currently backs
                    // `active_config`'s borrow). `InnerProgressFeedback`
                    // is an Arc-wrapper bundle, so the clone is cheap.
                    let cap_feedback = active_config.outer_inner_cap.clone();
                    let mut next = active_config.clone();
                    prev_attempt_grad_norm = Some(cur_grad_norm);
                    next.initial_rho = Some(result.rho.clone());
                    next.operator_initial_trust_radius = next_trust_radius;
                    retry_config = Some(next);
                    arc_retries_left -= 1;
                    obj.reset();
                    // Lift any inner-PIRLS cap for the resumed run. The
                    // schedule's cold-start ladder (3/5/10) would
                    // re-coarsen exactly the inner solves whose tolerance
                    // is suspected to have starved the prior trajectory.
                    // The next outer iter consumes ρ near a near-stationary
                    // point where exact β / gradient / Hessian is the
                    // load-bearing input to the operator-TR geometry.
                    if let Some(feedback) = cap_feedback.as_ref() {
                        feedback.cap.store(0, Ordering::Relaxed);
                    }
                }
                Err(e) => break Err(e),
            }
        };

        match outcome {
            Ok(result) => {
                if result.converged {
                    return Ok(result);
                }

                let improves_checkpoint = result.final_value.is_finite()
                    && best_checkpoint.as_ref().is_none_or(|checkpoint| {
                        !checkpoint.final_value.is_finite()
                            || result.final_value < checkpoint.final_value
                    });
                if improves_checkpoint {
                    best_checkpoint = Some(result);
                }

                let message = format!(
                    "{context}: attempt {} (plan={the_plan}) exhausted without convergence",
                    attempt_idx + 1
                );
                log::debug!("[OUTER] {message}; trying degraded fallback plan");
                last_error = Some(EstimationError::RemlOptimizationFailed(message));
            }
            Err(e) => {
                log::debug!(
                    "[OUTER] {context}: attempt {} (plan={the_plan}) failed: {e}",
                    attempt_idx + 1
                );
                last_error = Some(e);
            }
        }
    }

    if let Some(checkpoint) = best_checkpoint {
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
/// [`crate::estimate::reml::per_atom_efs::per_atom_efs_eligible`], which
/// requires all-penalty-like coordinates, a working `eval_efs` hook,
/// fixed-point not disabled, and a frontier-scale ρ-dimension. This is the
/// single auto-switch predicate; `plan` keeps selecting the
/// dense or standard-EFS solver for everything below the frontier threshold.
pub fn is_per_atom_efs_frontier(cap: &OuterCapability) -> bool {
    crate::estimate::reml::per_atom_efs::per_atom_efs_eligible(cap)
}

/// Auto-switch entry point: when `cap` is frontier-scale per-atom-EFS-eligible,
/// run the per-atom decoupled EFS primary and return its [`OuterResult`];
/// otherwise return `Ok(None)` so the caller falls through to the existing dense
/// / standard-EFS path via [`OuterProblem::run`] / [`run_outer`].
///
/// Builds the same bounded seed and tolerance/budget the standard plan path
/// uses, picks the seed (initial-ρ if supplied, else the first generated
/// candidate — the per-atom fixed point is a contraction near the optimum and
/// does not need the multi-seed cascade the dense path runs for its non-convex
/// quasi-Newton surface), then drives the per-atom EFS loop. The shared-border
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

    let (lower, upper) = outer_bounds_template(config, cap.n_params);

    // Seed: cache/explicit initial ρ if present, otherwise the first generated
    // candidate. The per-atom multiplicative fixed point is locally
    // contractive, so a single seed suffices; the heavy multi-seed cascade
    // exists for the dense quasi-Newton's non-convex surface, not for EFS.
    let seed = match config.initial_rho.as_ref() {
        Some(initial) if initial.len() == cap.n_params => initial.clone(),
        _ => {
            let generated = crate::seeding::generate_rho_candidates(
                cap.n_params,
                config.heuristic_lambdas.as_deref(),
                &config.seed_config,
            );
            match generated.into_iter().next() {
                Some(first) => first,
                None => Array1::<f64>::zeros(cap.n_params),
            }
        }
    };

    log::info!(
        "[OUTER] {context}: frontier ρ-scaling (rho_dim={rho_dim}) → per-atom decoupled EFS primary"
    );

    let pa_cfg = crate::estimate::reml::per_atom_efs::PerAtomEfsConfig::new(
        config.tolerance,
        config.max_iter,
        lower,
        upper,
    );
    let topology = crate::estimate::reml::per_atom_efs::SharedBorderTopology::disjoint(rho_dim);

    obj.reset();
    let result =
        crate::estimate::reml::per_atom_efs::run_per_atom_efs(obj, &seed, &pa_cfg, &topology)?;
    Ok(Some(result.into_outer_result(the_plan)))
}

pub(crate) fn outer_bounds(lo: &Array1<f64>, hi: &Array1<f64>) -> Result<Bounds, EstimationError> {
    Bounds::new(lo.clone(), hi.clone(), 1e-6).map_err(|err| {
        EstimationError::InvalidInput(format!("outer rho bounds are invalid: {err}"))
    })
}

pub(crate) fn outer_bounds_template(config: &OuterConfig, n: usize) -> (Array1<f64>, Array1<f64>) {
    config.bounds.clone().unwrap_or_else(|| {
        (
            Array1::<f64>::from_elem(n, -config.rho_bound),
            Array1::<f64>::from_elem(n, config.rho_bound),
        )
    })
}

/// Intersect typed objective-domain faces with the caller's configured search
/// box. The resulting box is stored back on `config`, making it the one source
/// consumed by seed projection, continuation entry, every solver, and terminal
/// projected-stationarity certification.
pub(super) fn install_objective_domain(
    config: &mut OuterConfig,
    n_params: usize,
    objective_lower: Option<Array1<f64>>,
    objective_upper: Option<Array1<f64>>,
) -> Result<(), EstimationError> {
    let (mut lower, mut upper) = outer_bounds_template(config, n_params);
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
        if let Some(domain) = objective_lower.as_ref() {
            let value = domain[index];
            if !value.is_finite() {
                return Err(EstimationError::InvalidInput(format!(
                    "outer objective-domain lower bound[{index}] must be finite; got {value}"
                )));
            }
            lower[index] = lower[index].max(value);
        }
        if let Some(domain) = objective_upper.as_ref() {
            let value = domain[index];
            if !value.is_finite() {
                return Err(EstimationError::InvalidInput(format!(
                    "outer objective-domain upper bound[{index}] must be finite; got {value}"
                )));
            }
            upper[index] = upper[index].min(value);
        }
        if !(lower[index].is_finite() && upper[index].is_finite() && lower[index] < upper[index]) {
            return Err(EstimationError::InvalidInput(format!(
                "outer objective-domain intersection is empty or non-finite at coordinate {index}: lower={}, upper={}",
                lower[index], upper[index]
            )));
        }
    }
    config.bounds = Some((lower, upper));
    Ok(())
}

pub(crate) fn outer_tolerance(value: f64) -> Result<Tolerance, EstimationError> {
    Tolerance::new(value)
        .map_err(|err| EstimationError::InvalidInput(format!("outer tolerance is invalid: {err}")))
}

pub(crate) fn outer_gradient_tolerance(config: &OuterConfig) -> GradientTolerance {
    let abs = config
        .objective_scale
        // A matrix-factorization REML/LAML score cannot resolve relative
        // perturbations below the forward-error scale √ε. Requiring a smaller
        // absolute residual made gradient-only / operator-curvature objectives
        // impossible to certify unless an unrelated Hessian or probe-noise
        // rescue happened to be available (#2269). This is the arithmetic
        // resolution of the declared objective scale, not a fitted tolerance.
        .map(|scale| config.tolerance.max(scale * f64::EPSILON.sqrt()))
        .unwrap_or(config.tolerance);
    GradientTolerance {
        abs,
        rel_initial_grad: None,
        rel_cost: Some(config.rel_cost_tolerance.unwrap_or(config.tolerance)),
        projected: true,
    }
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
    SeedRejected(EstimationError),
    ImmediateFallback(EstimationError),
    Failed(EstimationError),
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
    // Shared publication slot for the recurrent-restored-incumbent stop
    // (#2235 verdict 2): the bridge is moved into the driver, so the streak
    // count comes back through this cell and is stamped onto the returned
    // `OuterResult` below.
    let recurrent_incumbent_exit = Arc::new(Mutex::new(None));
    let mut objective = OuterFixedPointBridge {
        obj,
        layout,
        barrier_config,
        fixed_point_tolerance: config.tolerance,
        consecutive_psi_zero_iters: 0,
        last_restored_incumbent_streak: None,
        recurrent_incumbent_exit: Arc::clone(&recurrent_incumbent_exit),
    };
    let seed_sample = match objective.eval_step(seed) {
        Ok(sample) => sample,
        Err(err) => {
            let err = match err {
                ObjectiveEvalError::Recoverable { message }
                | ObjectiveEvalError::Fatal { message } => {
                    EstimationError::RemlOptimizationFailed(message)
                }
            };
            if requests_immediate_first_order_fallback(&err.to_string()) {
                return Err(FixedPointOuterRunError::ImmediateFallback(err));
            }
            return Err(FixedPointOuterRunError::SeedRejected(err));
        }
    };
    let (lo, hi) = outer_bounds_template(config, layout.n_params);
    let bounds = outer_bounds(&lo, &hi).map_err(FixedPointOuterRunError::Failed)?;
    let tol = outer_tolerance(config.tolerance).map_err(FixedPointOuterRunError::Failed)?;
    let max_iter =
        outer_max_iterations(config.max_iter).map_err(FixedPointOuterRunError::Failed)?;
    let mut optimizer = FixedPoint::new(seed.clone(), objective)
        // Seed validation already paid the complete EFS inner solve. Reuse that
        // exact sample so iteration zero neither repeats the expensive solve nor
        // mistakes two evaluations at the identical rho for recurrent incumbent
        // evidence (#2241).
        .with_initial_sample(seed.clone(), seed_sample)
        .with_bounds(bounds)
        .with_tolerance(tol)
        .with_max_iterations(max_iter);
    match optimizer.run() {
        Ok(sol) => {
            let mut result = solution_into_outer_result(sol, true, the_plan);
            // Stamp the model-state fixed-point stop when the bridge published
            // one; `None` means the walk stopped through the ordinary
            // step-norm test instead.
            if let Some(consecutive_restores) =
                recurrent_incumbent_exit.lock().ok().and_then(|slot| *slot)
            {
                result.converged_via = Some(OuterConvergedVia::RecurrentIncumbent {
                    consecutive_restores,
                });
            }
            Ok(result)
        }
        Err(FixedPointError::MaxIterationsReached { last_solution }) => {
            log::warn!(
                "[OUTER warning] {context}: {label} hit max_iter={} at final_value={:.6e} step_norm={:.3e}",
                config.max_iter,
                last_solution.final_value,
                last_solution.final_gradient_norm.unwrap_or(f64::NAN),
            );
            Ok(solution_into_outer_result(*last_solution, false, the_plan))
        }
        Err(e) => Err(FixedPointOuterRunError::Failed(
            EstimationError::RemlOptimizationFailed(format!("{failure_prefix}: {e:?}")),
        )),
    }
}
