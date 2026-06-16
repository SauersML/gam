use super::*;

pub(crate) fn outer_strategy_contract_panic(message: impl Into<String>) -> ! {
    std::panic::panic_any(message.into())
}

/// Result of one outer objective evaluation.
///
/// The Hessian field uses [`HessianResult`] instead of `Option<Array2<f64>>`
/// to make the presence/absence of an analytic Hessian explicit and
/// pattern-matchable.
pub struct OuterEval {
    pub cost: f64,
    pub gradient: Array1<f64>,
    pub hessian: HessianResult,
    /// Optional inner-solver iterate at this ρ. Families whose inner
    /// solve produces a PIRLS β (GAM, custom-family marginal-slope, …)
    /// populate this so the persistent-cache layer can store `(ρ, β)`
    /// together. Empty for callers that have no inner state or don't
    /// expose it; the cache gracefully writes a ρ-only record then.
    pub inner_beta_hint: Option<Array1<f64>>,
}

impl OuterEval {
    /// Conventional representation of an infeasible trial point.
    ///
    /// `opt` translates the non-finite objective into a recoverable trial
    /// failure so trust-region/line-search solvers retreat without the caller
    /// needing to special-case infeasible regions locally.
    pub fn infeasible(n_params: usize) -> Self {
        Self {
            cost: f64::INFINITY,
            gradient: Array1::zeros(n_params),
            hessian: HessianResult::Unavailable,
            inner_beta_hint: None,
        }
    }
}

/// Explicit Hessian result replacing `Option<Array2<f64>>`.
pub enum HessianResult {
    /// Analytic Hessian was computed and returned.
    Analytic(Array2<f64>),
    /// Analytic Hessian is available as an exact Hessian-vector product.
    Operator(Arc<dyn OuterHessianOperator>),
    /// No analytic Hessian available for this model path.
    /// The runner must use the [`HessianSource`] from the [`OuterPlan`]
    /// to choose a declared first-order or derivative-free strategy.
    Unavailable,
}

impl Clone for OuterEval {
    fn clone(&self) -> Self {
        Self {
            cost: self.cost,
            gradient: self.gradient.clone(),
            hessian: self.hessian.clone(),
            inner_beta_hint: self.inner_beta_hint.clone(),
        }
    }
}

impl std::fmt::Debug for OuterEval {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("OuterEval")
            .field("cost", &self.cost)
            .field("gradient", &self.gradient)
            .field("hessian", &self.hessian)
            .finish()
    }
}

impl Clone for HessianResult {
    fn clone(&self) -> Self {
        match self {
            Self::Analytic(h) => Self::Analytic(h.clone()),
            Self::Operator(op) => Self::Operator(Arc::clone(op)),
            Self::Unavailable => Self::Unavailable,
        }
    }
}

impl std::fmt::Debug for HessianResult {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Analytic(h) => f
                .debug_tuple("Analytic")
                .field(&format!("{}x{}", h.nrows(), h.ncols()))
                .finish(),
            Self::Operator(op) => f
                .debug_tuple("Operator")
                .field(&format!("dim={}", op.dim()))
                .finish(),
            Self::Unavailable => f.write_str("Unavailable"),
        }
    }
}

impl HessianResult {
    /// Extract the Hessian matrix, panicking if unavailable.
    ///
    /// Only call this when the [`OuterPlan`] guarantees `HessianSource::Analytic`.
    pub fn unwrap_analytic(self) -> Array2<f64> {
        match self {
            HessianResult::Analytic(h) => h,
            // `unwrap_analytic`'s doc-comment pins its contract: caller must
            // hold an `OuterPlan` whose `HessianSource::Analytic` is dense.
            HessianResult::Operator(_) => {
                // SAFETY: reaching this arm means the plan promised dense
                // analytic but the family returned an operator — contract
                // violation; fail-fast surfaces the upstream plan bug.
                outer_strategy_contract_panic(
                    "expected dense analytic Hessian but got HessianResult::Operator",
                )
            }
            HessianResult::Unavailable => {
                // SAFETY: same contract as Operator above — `Unavailable`
                // means the family failed to produce the analytic Hessian
                // its `OuterPlan` declared.
                outer_strategy_contract_panic(
                    "expected analytic Hessian but got HessianResult::Unavailable",
                )
            }
        }
    }

    /// Returns `true` if an analytic Hessian is present in any exact form.
    pub fn is_analytic(&self) -> bool {
        matches!(
            self,
            HessianResult::Analytic(_) | HessianResult::Operator(_)
        )
    }

    /// Convert to the optional Hessian shape used by the opt bridge.
    pub fn into_option(self) -> Option<Array2<f64>> {
        match self {
            HessianResult::Analytic(h) => Some(h),
            HessianResult::Operator(_) => None,
            HessianResult::Unavailable => None,
        }
    }

    pub fn dim(&self) -> Option<usize> {
        match self {
            HessianResult::Analytic(h) => Some(h.nrows()),
            HessianResult::Operator(op) => Some(op.dim()),
            HessianResult::Unavailable => None,
        }
    }

    pub fn materialize_dense(&self) -> Result<Option<Array2<f64>>, String> {
        match self {
            HessianResult::Analytic(h) => Ok(Some(h.clone())),
            HessianResult::Operator(op) => op.materialize_dense().map(Some),
            HessianResult::Unavailable => Ok(None),
        }
    }

    pub fn add_rho_block_dense(&mut self, rho_block: &Array2<f64>) -> Result<(), String> {
        if rho_block.nrows() != rho_block.ncols() {
            return Err(OuterStrategyError::RhoBlockShape {
                reason: format!(
                    "rho-block Hessian update must be square, got {}x{}",
                    rho_block.nrows(),
                    rho_block.ncols()
                ),
            }
            .into());
        }
        match self {
            HessianResult::Analytic(h) => {
                if rho_block.nrows() > h.nrows() || rho_block.ncols() > h.ncols() {
                    return Err(OuterStrategyError::RhoBlockShape {
                        reason: format!(
                            "rho-block Hessian update shape mismatch: got {}x{}, outer Hessian is {}x{}",
                            rho_block.nrows(),
                            rho_block.ncols(),
                            h.nrows(),
                            h.ncols()
                        ),
                    }
                    .into());
                }
                let k = rho_block.nrows();
                let mut sl = h.slice_mut(ndarray::s![..k, ..k]);
                sl += rho_block;
                Ok(())
            }
            HessianResult::Operator(op) => {
                let base = Arc::clone(op);
                let dim = base.dim();
                if rho_block.nrows() > dim {
                    return Err(OuterStrategyError::RhoBlockShape {
                        reason: format!(
                            "rho-block Hessian update dimension mismatch: got {}x{}, operator dim is {}",
                            rho_block.nrows(),
                            rho_block.ncols(),
                            dim
                        ),
                    }
                    .into());
                }
                *self = HessianResult::Operator(Arc::new(RhoBlockAdditiveOuterHessian {
                    base,
                    rho_block: rho_block.clone(),
                    dim,
                }));
                Ok(())
            }
            HessianResult::Unavailable => Ok(()),
        }
    }
}

/// Result of an EFS (Extended Fellner-Schall) evaluation at a given rho.
///
/// Contains the REML/LAML cost at the current rho and the additive step
/// vector produced by `compute_efs_update`. The caller applies the step as
/// `rho_new[i] = rho[i] + steps[i]`.
///
/// For the hybrid EFS+preconditioned-gradient strategy, the steps vector
/// contains both EFS steps (for ρ coords) and preconditioned gradient steps
/// (for ψ coords). The `psi_gradient` field carries the raw ψ-block gradient
/// for optional backtracking.
#[derive(Clone, Debug)]
pub struct EfsEval {
    /// REML/LAML cost at the current rho (for convergence monitoring and
    /// comparing candidates).
    pub cost: f64,
    /// Additive steps. Length = n_rho + n_ext_coords.
    ///
    /// For pure EFS: steps for non-penalty-like coordinates are 0.0.
    /// For hybrid EFS: ρ-coords get standard EFS multiplicative steps,
    /// ψ-coords get preconditioned gradient steps `Δψ = -α G⁺ g_ψ`.
    pub steps: Vec<f64>,
    /// Current coefficient vector β̂ from the inner P-IRLS solve.
    /// Used by the EFS loop for the runtime barrier-curvature significance
    /// check when monotonicity constraints are present.
    pub beta: Option<Array1<f64>>,
    /// Raw REML/LAML gradient restricted to the ψ block (design-moving coords).
    ///
    /// Present only when the hybrid EFS strategy is active. Used by the
    /// outer iteration for backtracking on the ψ step: if the combined
    /// (ρ-EFS, ψ-gradient) step does not decrease V(θ), the ψ step size
    /// α is halved while keeping the ρ-EFS step fixed.
    ///
    /// This avoids re-evaluating the gradient during backtracking since
    /// the gradient was already computed as part of the hybrid EFS eval.
    pub psi_gradient: Option<Array1<f64>>,
    /// Indices into the full θ vector that correspond to ψ (design-moving)
    /// coordinates. Used by the backtracking logic to selectively scale
    /// only the ψ portion of the step.
    pub psi_indices: Option<Vec<usize>>,
    /// Representative curvature scale of the inner Hessian
    /// `H = X'W_HX + S_λ` (+ any barrier perturbation), at β̂.
    ///
    /// Production REML/LAML paths provide the geometric mean of the active
    /// Hessian spectrum, `exp(log|H|_+ / rank(H))`, which is basis-invariant,
    /// has curvature units, and is already available from the Hessian operator
    /// used to evaluate the objective. The EFS barrier check uses it for the
    /// dimensionally correct comparison `max_j τ/Δ_j² > threshold · scale`.
    pub inner_hessian_scale: Option<f64>,
    /// `Some(gap)` when the `½log|H|` term in `cost` was produced as a CERTIFIED
    /// TWO-SIDED ENCLOSURE (the #1011 block-preconditioned border-Schur bound)
    /// rather than an exact logdet — `gap` is the enclosure width that `cost`
    /// inherits. The EFS engine consults the decision-margin contract against
    /// its step tolerance: a step is only allowed to declare convergence when
    /// the cost it converged on is resolved more tightly than that tolerance.
    /// `None` (the default for every exact-logdet objective) preserves today's
    /// behavior bit-for-bit.
    pub logdet_enclosure_gap: Option<f64>,
}

impl EfsEval {
    /// Attach a certified logdet-enclosure gap to this eval (the #1011 contract).
    /// The EFS engine then refuses to declare convergence on a cost the
    /// enclosure does not pin down below the step tolerance, escalating instead.
    #[must_use]
    pub fn with_logdet_enclosure_gap(mut self, gap: Option<f64>) -> Self {
        self.logdet_enclosure_gap = gap;
        self
    }
}

/// Outcome of [`OuterObjective::seed_inner_state`].
///
/// Distinguishes two non-error outcomes that callers handle differently:
///
/// - [`SeedOutcome::Installed`] — the objective owns an inner-β slot and the
///   provided β has been stored there. The next `eval*` will warm-start from
///   this β.
/// - [`SeedOutcome::NoSlot`] — the objective has no inner-β slot at all. The
///   provided β is silently discarded. This is the contract reply for
///   objectives whose inner iterate is conceptually empty (e.g. line-search
///   bridges, screening proxies, fixed-spec objectives).
///
/// Genuine seeding failures (wrong dimension when a slot exists, internal
/// allocation faults, …) are reported via `Err(EstimationError)`.
///
/// The two non-error variants exist because the two real callers want
/// opposite behavior on the no-slot path:
///
/// - The outer cache warm-start path (`OuterProblem::run`) reads a `(ρ, β)`
///   pair from disk; if the objective has no β slot it must log loudly
///   ("β-bearing checkpoint silently degraded to ρ-only resume") so cache
///   provenance is auditable.
/// - The continuation walk (`prime_outer_seed`) forwards `inner_beta_hint`
///   from the previous step; if the objective has no β slot the walk
///   simply proceeds cold — no log, no error.
///
/// Encoding the distinction in the return type lets each caller branch on
/// the variant without inspecting error message strings (the previous
/// brittle approach, see git history for `is_no_hook` in continuation.rs).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SeedOutcome {
    /// The objective installed the provided β into its inner-β slot.
    Installed,
    /// The objective has no inner-β slot; the β was discarded.
    NoSlot,
    /// The objective owns an inner-β slot, but the provided β is
    /// structurally incompatible with this fit's inner block layout
    /// (its length does not match the per-block coefficient widths). The
    /// β was discarded and the fit resumes ρ-only.
    ///
    /// This is the load-time reply for a *row-relaxed* cross-fit seed
    /// (the `cache_seed_key` prefix channel): two folds of the same model
    /// share an ρ-dim, so the cached ρ transfers, but the realized basis
    /// rank — hence the inner β length — is row-population dependent and
    /// legitimately differs across folds (the LOSO p=37-vs-p=85 case).
    /// A length-mismatched seed β is therefore NOT an error: cross-length
    /// β transfer is delegated to the gauge-projected `FitArtifact`
    /// channel, which least-squares re-expresses the parent's raw β into
    /// this fold's reduced subspace. Reporting `Incompatible` here keeps
    /// the (correct) ρ seed and avoids a spurious full cold-start.
    Incompatible,
}

/// Common interface for outer smoothing-parameter objectives.
///
/// Every model path that optimizes smoothing parameters implements this trait.
/// The runner function consumes it and handles solver selection,
/// multi-start, and logging while delegating derivative fallback policy to
/// `opt`.
///
/// # Contract
///
/// - `capability()` must be stable (same result across calls).
/// - `eval()` may return `HessianResult::Unavailable` at individual trial
///   points even when `capability().hessian == Analytic`; `opt` degrades that
///   step to first-order behavior instead of requiring the objective to fake a
///   stale or non-finite Hessian.
/// - Use `eval_cost()` / `OuterEval::infeasible()` for infeasible trial points.
///   Return `Err(...)` for genuine evaluation breakdowns so the runner can mark
///   the step as a recoverable solver failure and escalate to the next declared
///   fallback plan if the full attempt still fails.
/// - `eval_cost()` is used only for cost-based optimization paths.
/// - `eval()` is the main evaluation path (cost + gradient + optional Hessian).
/// - `eval_efs()` is used only by the EFS solver. It runs the inner solve,
///   builds the `InnerSolution`, and computes the EFS step vector. The default
///   implementation returns an error; only objectives that support EFS need
///   to override it.
/// - `reset()` restores state to a clean baseline (for multi-start).
pub trait OuterObjective {
    /// Declare what this objective can compute analytically.
    fn capability(&self) -> OuterCapability;

    /// Evaluate cost only for cost-based optimization paths.
    fn eval_cost(&mut self, rho: &Array1<f64>) -> Result<f64, EstimationError>;

    /// Evaluate the seed-screening ranking proxy at this `rho`.
    ///
    /// Used exclusively by the `rank_seeds_with_screening` cascade. The
    /// default delegates to [`OuterObjective::eval_cost`], which preserves
    /// behavior for non-REML objectives.
    ///
    /// Concrete REML-state objectives override this to return the per-seed
    /// minimum penalized deviance observed during the inner P-IRLS solve
    /// (a monotonically descending quantity that remains a meaningful
    /// quality signal even at a 3-iteration screening cap), instead of the
    /// V_LAML criterion (which is dominated by a poorly-conditioned
    /// `0.5·log|H|` term at partial-fit β̂ and ranks seeds little better
    /// than random). The proxy fires *only* in screening mode; outside
    /// screening it must return the regular V_LAML cost so the optimization
    /// objective is unchanged.
    ///
    /// # Why the `eval_cost` default is correct for everyone else (#969)
    ///
    /// The partial-fit pathology is CAUSED by the screening cap: it is the
    /// `0.5·log|H|` term evaluated at a β̂ whose inner solve was truncated
    /// by `screening_max_inner_iterations`. An objective only suffers it if
    /// it (a) consumes that cap atomic AND (b) ranks on a curvature-bearing
    /// criterion at the truncated iterate — which is exactly the REML/LAML
    /// state-objective family, all of which override this method (or are
    /// built via `build_objective_with_screening_proxy`). Objectives that
    /// never wire the cap pay the full inner solve during screening, so
    /// their screened cost IS the true criterion — slower, but a correct
    /// ranking by definition, and a proxy could only degrade it. Any future
    /// objective that starts honoring the screening cap on a
    /// curvature-bearing criterion must override this with its own
    /// monotonically-descending inner quantity (the penalized-deviance
    /// pattern above generalizes: rank on the best inner merit seen, never
    /// on a curvature term at a truncated iterate).
    fn eval_screening_proxy(&mut self, rho: &Array1<f64>) -> Result<f64, EstimationError> {
        self.eval_cost(rho)
    }

    /// Evaluate cost + gradient + (if capable) Hessian.
    fn eval(&mut self, rho: &Array1<f64>) -> Result<OuterEval, EstimationError>;

    /// Evaluate the outer objective at the order requested by the active plan.
    ///
    /// The default preserves legacy behavior by delegating value-only requests
    /// to [`OuterObjective::eval_cost`] and derivative requests to
    /// [`OuterObjective::eval`].
    fn eval_with_order(
        &mut self,
        rho: &Array1<f64>,
        order: OuterEvalOrder,
    ) -> Result<OuterEval, EstimationError> {
        match order {
            OuterEvalOrder::Value => {
                let cost = self.eval_cost(rho)?;
                Ok(OuterEval {
                    cost,
                    gradient: Array1::zeros(rho.len()),
                    hessian: HessianResult::Unavailable,
                    inner_beta_hint: None,
                })
            }
            OuterEvalOrder::ValueAndGradient | OuterEvalOrder::ValueGradientHessian => {
                self.eval(rho)
            }
        }
    }

    /// Evaluate cost + EFS step vector. Only needed when the plan selects
    /// `Solver::Efs`. The default returns an error indicating EFS is not
    /// supported by this objective.
    fn eval_efs(&mut self, rho: &Array1<f64>) -> Result<EfsEval, EstimationError> {
        Err(EstimationError::RemlOptimizationFailed(format!(
            "EFS evaluation not implemented for this objective at rho_dim={}",
            rho.len()
        )))
    }

    /// Restore to a clean baseline for the next multi-start candidate.
    fn reset(&mut self);

    /// Seed the inner-solver iterate before the first eval, e.g. when the
    /// outer-iterate cache restored a `(ρ, β)` pair from a prior run, or
    /// when the continuation walk forwards `OuterEval::inner_beta_hint`
    /// from the previous step.
    ///
    /// Objectives make an explicit choice via the [`SeedOutcome`] return:
    /// implementations with an inner β slot return [`SeedOutcome::Installed`]
    /// after storing β; implementations without one return
    /// [`SeedOutcome::NoSlot`]. Genuine seeding failures (wrong dimension
    /// when a slot exists, etc.) are reported via `Err(EstimationError)`.
    ///
    /// Callers that need to distinguish "no slot" from "installed" (the
    /// outer cache warm-start path, which logs cache provenance) branch on
    /// the variant. Callers that don't care (the continuation walk, which
    /// only proceeds-cold when the hint is unusable) ignore it and only
    /// propagate `Err`.
    fn seed_inner_state(&mut self, beta: &Array1<f64>) -> Result<SeedOutcome, EstimationError>;

    /// Whether the objective can benefit from continuation pre-warm before
    /// the first solver eval at a candidate seed.
    ///
    /// Pre-warm is only correct for objectives with a real writable inner
    /// state slot: it evaluates an oversmoothed rho path before the seed and
    /// forwards non-empty `inner_beta_hint`s between steps. Generic synthetic
    /// objectives and rho-only cache probes must start at the chosen seed
    /// directly, otherwise the pre-warm becomes an observable extra eval and
    /// can clobber seed-dispatch bookkeeping with empty beta seeds.
    fn allow_continuation_prewarm(&self) -> bool {
        false
    }

    /// Optional opt-in to the device-resident outer REML BFGS-over-ρ driver
    /// (`crate::solver::gpu::reml_outer::run_reml_outer_on_device`). Returns
    /// `Some(adm)` when the objective is a REML evaluator whose
    /// `(spec, n, p, num_rho)` admission predicate accepts the device path,
    /// and `None` otherwise.
    ///
    /// The default returns `None` so non-REML objectives (line-search-only
    /// inner bridges, screening proxies, the EFS / hybrid-EFS sub-objectives)
    /// keep the host BFGS branch unconditionally — only the concrete
    /// REML-state objectives override this to consult
    /// [`crate::solver::estimate::reml::outer_eval::outer_reml_device_admission`].
    fn outer_device_admission(&self) -> Option<crate::gpu::policy::RemlOuterAdmission> {
        None
    }

    /// Whether every joint fit of this objective must ENTER through the
    /// [`crate::solver::continuation_path::ContinuationPath`] (heavy-smoothing
    /// entry) rather than being solved cold at the seed ρ*.
    ///
    /// The SAE-manifold joint objective overrides this to `true`: its joint
    /// `(logits, t, β)` block has a combinatorial active-set component that a
    /// cold solve can collapse, so it is entered at a heavy-smoothing regime
    /// and annealed down. Crucially, this flips the seed cascade's structural
    /// failure handling from REJECT to **DEMOTE-WITH-REASON**: a "cold"
    /// structural defect (rank/alias/active-set diagnosis from the seed
    /// pre-warm or the uniform-structural early-exit) is not a disqualification
    /// but a signal to RE-ENTER the same seed at a *heavier* ContinuationPath
    /// regime. The candidate set therefore never empties on a structural
    /// diagnosis — every demotion is recorded with its reason and routed to a
    /// heavier regime.
    ///
    /// The default `false` preserves the existing contract for every other
    /// objective: pre-warm stays an optimization (never a feasibility gate),
    /// and a uniform structural rejection still short-circuits the cascade.
    fn requires_continuation_path_entry(&self) -> bool {
        false
    }

    /// Run the objective's certified curvature-homotopy entry leg, if it has
    /// one, leaving the inner state warm at the real (`η = 1`) objective.
    ///
    /// An objective with a *certified anchor* — a point known by construction to
    /// be the global optimum of a relaxed problem — can replace the blind
    /// multi-seed multistart with a single predictor-corrector walk from that
    /// anchor to the true objective (#1007). The SAE-manifold objective
    /// overrides this: its `η = 0` Eckart-Young linear relaxation is convex and
    /// its optimum is certified by `linear_span_anchor`, so the walk in `η`
    /// tracks the unique optimal branch to `η = 1`. The walk monitors the
    /// arrow-factor min-pivot and halves the `η` step when it shrinks; a pivot
    /// collapse below tolerance is a DETECTED bifurcation (recorded on the fit
    /// payload, never silent), at which point the objective falls back to the
    /// documented multi-seed cascade.
    ///
    /// Returns:
    ///   * `None` — no certified anchor; use the standard seed cascade
    ///     (the default for every other objective).
    ///   * `Some(Ok(true))` — the walk arrived; the inner state is warm at the
    ///     certified `η = 1` solution and the seed cascade is bypassed.
    ///   * `Some(Ok(false))` — the anchor degenerated or the walk detected a
    ///     bifurcation; fall back to the multi-seed cascade (the report is
    ///     recorded on the objective for the fit payload).
    ///   * `Some(Err(_))` — a hard failure constructing the anchor.
    fn curvature_homotopy_entry(
        &mut self,
        rho: &Array1<f64>,
    ) -> Option<Result<bool, EstimationError>> {
        // Default: no certified anchor — but a non-finite seed is reported
        // here rather than silently handed to the seed cascade, mirroring the
        // hard-failure contract of the overriding implementations.
        if let Some(idx) = rho.iter().position(|v| !v.is_finite()) {
            return Some(Err(EstimationError::RemlOptimizationFailed(format!(
                "curvature-homotopy entry received non-finite rho[{idx}]"
            ))));
        }
        None
    }

    /// Let an objective declare that a seed is already a terminal outer result.
    /// Used for objectives with a certified high-quality construction seed where
    /// the generic rho optimizer can only degrade the fitted state.
    fn accept_seed_without_outer_iterations(
        &mut self,
        rho: &Array1<f64>,
    ) -> Result<Option<f64>, EstimationError> {
        if rho.is_empty() {
            return Ok(None);
        }
        Ok(None)
    }

    /// Re-install the selected outer result into the mutable objective before
    /// callers consume objective-owned fitted state. Optimizers may evaluate
    /// rejected trial points after the best point was found; without this final
    /// synchronization, stateful objectives can report the last trial fit rather
    /// than the returned `OuterResult::rho`.
    fn finalize_outer_result(
        &mut self,
        rho: &Array1<f64>,
        plan: &OuterPlan,
    ) -> Result<(), EstimationError> {
        log::debug!(
            "[OUTER] finalize: re-installing best rho into the objective (solver {:?})",
            plan.solver
        );
        self.eval_cost(rho).map(|_| ())
    }
}

// ─── Persistent warm-start checkpoint plumbing ────────────────────────
//
// `CheckpointingObjective` wraps any `OuterObjective` to write a copy of
// `(rho, cost, eval_id)` to disk on each finite evaluation. The on-disk
// [`crate::warm_start::Session`] rate-limits writes (≥2 s gap unless this iterate
// strictly improves on the best-so-far) so a tight inner loop never thrashes
// the filesystem. The same checkpoint is also broadcast to optional mirror
// sessions, which lets interrupted exact-key runs seed later related fits via
// their prefix key instead of waiting for a final converged write.

#[derive(serde::Serialize, serde::Deserialize)]
pub(crate) struct IteratePayload {
    /// Bump on incompatible payload changes; decode rejects mismatches.
    schema: u32,
    pub(crate) rho: Vec<f64>,
    /// Inner-solver iterate (PIRLS β) captured alongside ρ. The (ρ, β)
    /// pair lives on the implicit-function manifold β = β*(ρ); restoring
    /// ρ alone forces the next inner solve to reconstruct β from scratch.
    /// For saturated ρ (|ρ_i| near `rho_bound`) the inner Hessian
    /// `X'WX + Σ λ_i S_i` has condition number `≈ e^{2·rho_bound}` — Newton
    /// degrades to O(1/k) descent and the cycle budget exhausts before
    /// KKT. Caching β lets the resume start in Newton's quadratic basin
    /// regardless of where ρ lives. Empty when the family did not surface
    /// an inner-β hint at write time (still useful as a ρ-only seed).
    #[serde(default)]
    pub(crate) beta: Vec<f64>,
    /// Converged exact outer curvature `H(θ̂)` (full θ×θ, row-major flatten),
    /// captured alongside the (ρ, β) iterate. A gradient-based BFGS solve does
    /// not surface its accumulated inverse-Hessian, so the next
    /// structurally-matching fit (e.g. the next LOSO fold) otherwise restarts
    /// BFGS from an unscaled identity metric and rediscovers curvature through
    /// line-search bracketing — multiple full inner-solve value probes per
    /// accepted outer step. Persisting the converged curvature lets the resume
    /// seed `InitialMetric::DenseInverseHessian(H⁻¹)` for a quasi-Newton first
    /// step. Empty when no exact outer Hessian was available at write time
    /// (still a valid ρ/β seed). `hessian_dim²` must equal `hessian.len()`.
    #[serde(default)]
    pub(crate) hessian: Vec<f64>,
    /// Side length of the square `hessian` matrix (`hessian.len() == dim²`).
    /// Zero when no Hessian was persisted.
    #[serde(default)]
    pub(crate) hessian_dim: usize,
    pub(crate) cost: f64,
    eval_id: u64,
}

/// Entries with a different schema id are rejected by `decode_iterate`
/// so incompatible on-disk payloads fall through to cold start instead
/// of seeding the inner solve with a malformed iterate.
pub(crate) const ITERATE_PAYLOAD_SCHEMA: u32 = 2;

pub(crate) fn encode_iterate(
    rho: &Array1<f64>,
    beta: Option<&Array1<f64>>,
    hessian: Option<&Array2<f64>>,
    cost: f64,
    eval_id: u64,
) -> Option<Vec<u8>> {
    // Persist the converged outer curvature only when it is square and finite;
    // a non-finite or non-square Hessian is dropped (the resume falls back to a
    // ρ/β-only seed) so a malformed curvature can never corrupt a warm start.
    let (hessian_flat, hessian_dim) = match hessian {
        Some(h) if h.nrows() == h.ncols() && h.iter().all(|v| v.is_finite()) => {
            (h.iter().copied().collect::<Vec<f64>>(), h.nrows())
        }
        _ => (Vec::new(), 0),
    };
    let p = IteratePayload {
        schema: ITERATE_PAYLOAD_SCHEMA,
        rho: rho.to_vec(),
        beta: beta.map(|b| b.to_vec()).unwrap_or_default(),
        hessian: hessian_flat,
        hessian_dim,
        cost,
        eval_id,
    };
    serde_json::to_vec(&p).ok()
}

pub(crate) fn decode_iterate(bytes: &[u8], expected_rho_dim: usize) -> Option<IteratePayload> {
    let mut p: IteratePayload = serde_json::from_slice(bytes).ok()?;
    if p.schema != ITERATE_PAYLOAD_SCHEMA {
        return None;
    }
    if p.rho.len() != expected_rho_dim {
        return None;
    }
    if !p.rho.iter().all(|x| x.is_finite()) || !p.cost.is_finite() {
        return None;
    }
    if !p.beta.iter().all(|x| x.is_finite()) {
        return None;
    }
    // A persisted Hessian must be square (`dim²` entries) and finite to be
    // usable as a warm-start metric; an inconsistent or non-finite curvature is
    // scrubbed to "no Hessian" rather than rejecting the whole iterate, so the
    // ρ/β seed still warms the resume.
    if p.hessian_dim.saturating_mul(p.hessian_dim) != p.hessian.len()
        || !p.hessian.iter().all(|x| x.is_finite())
    {
        p.hessian = Vec::new();
        p.hessian_dim = 0;
    }
    Some(p)
}

/// Outcome of inspecting a cache entry as a seed for the outer optimizer.
///
/// The classifier rejects only entries that fail structural validity
/// (wrong dimension, non-finite payload). It does NOT reshape ρ based on
/// saturation: every finite, well-shaped entry is honored as the next
/// run's seed.
///
/// Previously this enum carried `saturated_coords` / `clamped_to` /
/// "all-coords-saturated-poisoned-entry" branches that pulled boundary
/// ρ inward or discarded fully-saturated entries. Those were read-side
/// band-aids over the real bug: the warm-start contract stored ρ but
/// not β, so resuming at boundary ρ forced PIRLS to recompute β from
/// cold-start against a Hessian with condition number `≈ e^{2·rho_bound}`,
/// and Newton degraded to O(1/k) descent that exhausted the cycle budget.
///
/// The contract is now `(ρ, β)`: the schema-2 iterate payload carries
/// both, and [`CheckpointingObjective`] refuses to persist a divergent
/// inner state (non-finite cost or β). Boundary ρ — when written under
/// the new invariant — is a *legitimate* finding (the smoothness wants
/// to be near-null), and the cached β puts the next inner solve at the
/// previously converged iterate where the gradient is already at zero.
/// No clamp or shape-based discard is needed.
#[derive(Debug)]
pub(crate) enum CacheSeedDecision {
    ExactFinal {
        rho: Array1<f64>,
        /// Optional inner β captured at the converged ρ. Empty when the
        /// payload didn't carry one (legacy ρ-only writes or families
        /// that don't surface β).
        beta: Vec<f64>,
        final_value: f64,
        iterations: usize,
        prior_obj_display: f64,
    },
    Seed {
        rho: Array1<f64>,
        /// Optional inner β to prime the next run's inner solver via
        /// [`OuterObjective::seed_inner_state`]. When non-empty, the
        /// dispatcher injects β before the first eval so the inner
        /// PIRLS opens at zero-gradient regardless of where ρ sits in
        /// the box.
        beta: Vec<f64>,
        /// Optional converged outer Hessian `H(θ̂)` from the prior fit, as a
        /// `(dim, row-major flatten)` pair. `None` when the payload carried no
        /// curvature (legacy ρ/β-only writes). Seeds the BFGS iter-0 metric on
        /// the resume so the first outer step is quasi-Newton.
        hessian: Option<(usize, Vec<f64>)>,
        prior_obj_display: f64,
        iteration: u64,
    },
    Discard {
        reason: &'static str,
        prior_obj_display: f64,
        all_rho_finite: Option<bool>,
    },
}

pub(crate) fn classify_cache_entry_for_outer(
    loaded: &crate::warm_start::LoadedEntry,
    expected_rho_dim: usize,
) -> CacheSeedDecision {
    let entry = &loaded.entry;
    let Some(payload) = decode_iterate(&entry.payload, expected_rho_dim) else {
        return CacheSeedDecision::Discard {
            reason: "payload-shape-mismatch",
            prior_obj_display: entry.objective.unwrap_or(f64::NAN),
            all_rho_finite: None,
        };
    };
    let cached_rho = Array1::from_vec(payload.rho);
    let prior_obj_display = entry.objective.unwrap_or(f64::NAN);
    if matches!(entry.objective, Some(v) if !v.is_finite()) {
        return CacheSeedDecision::Discard {
            reason: "non-finite-payload",
            prior_obj_display,
            all_rho_finite: Some(cached_rho.iter().all(|v| v.is_finite())),
        };
    }
    if !cached_rho.iter().all(|v| v.is_finite()) {
        return CacheSeedDecision::Discard {
            reason: "non-finite-payload",
            prior_obj_display,
            all_rho_finite: Some(false),
        };
    }
    if loaded.source == LoadSource::Exact && entry.kind == crate::warm_start::EntryKind::Final {
        return CacheSeedDecision::ExactFinal {
            rho: cached_rho,
            beta: payload.beta,
            final_value: entry.objective.unwrap_or(payload.cost),
            iterations: entry
                .iteration
                .unwrap_or(payload.eval_id)
                .min(usize::MAX as u64) as usize,
            prior_obj_display,
        };
    }
    let hessian = if payload.hessian_dim > 0
        && payload.hessian.len() == payload.hessian_dim * payload.hessian_dim
    {
        Some((payload.hessian_dim, payload.hessian))
    } else {
        None
    };
    CacheSeedDecision::Seed {
        rho: cached_rho,
        beta: payload.beta,
        hessian,
        prior_obj_display,
        iteration: entry.iteration.unwrap_or(payload.eval_id),
    }
}

pub(crate) fn cache_entry_would_help_outer(
    loaded: &crate::warm_start::LoadedEntry,
    expected_rho_dim: usize,
) -> bool {
    matches!(
        classify_cache_entry_for_outer(loaded, expected_rho_dim),
        CacheSeedDecision::ExactFinal { .. } | CacheSeedDecision::Seed { .. }
    )
}

pub(crate) struct CheckpointingObjective<'a> {
    inner: &'a mut dyn OuterObjective,
    session: Arc<CacheSession>,
    mirror_sessions: Vec<Arc<CacheSession>>,
    eval_counter: AtomicU64,
    /// Most-recent inner β surfaced via [`OuterEval::inner_beta_hint`]. The
    /// finalize path reads this so the `kind: Final` write encodes the
    /// (ρ, β) pair that the BFGS optimum was actually fitted at — without
    /// this the finalize would clobber per-eval checkpoint β state with a
    /// ρ-only payload, reintroducing the cold-β resume failure.
    last_inner_beta: std::sync::Mutex<Option<Array1<f64>>>,
}

impl<'a> CheckpointingObjective<'a> {
    pub(crate) fn new(
        inner: &'a mut dyn OuterObjective,
        session: Arc<CacheSession>,
        mirror_sessions: Vec<Arc<CacheSession>>,
    ) -> Self {
        Self {
            inner,
            session,
            mirror_sessions,
            eval_counter: AtomicU64::new(0),
            last_inner_beta: std::sync::Mutex::new(None),
        }
    }

    pub(crate) fn last_inner_beta(&self) -> Option<Array1<f64>> {
        self.last_inner_beta.lock().ok().and_then(|g| g.clone())
    }

    fn note(&self, rho: &Array1<f64>, beta: Option<&Array1<f64>>, cost: f64) {
        if !cost.is_finite() {
            return;
        }
        // If β is provided, require it to be finite; non-finite β is a
        // divergent inner state — persisting it would re-poison the cache.
        if let Some(b) = beta {
            if !b.iter().all(|v| v.is_finite()) {
                return;
            }
            if let Ok(mut guard) = self.last_inner_beta.lock() {
                *guard = Some(b.clone());
            }
        }
        let i = self.eval_counter.fetch_add(1, Ordering::Relaxed);
        // Per-eval checkpoints carry no converged outer Hessian (curvature is
        // only meaningful at the final optimum); the finalize write is where the
        // converged `H(θ̂)` is persisted for cross-fit warm starts.
        if let Some(bytes) = encode_iterate(rho, beta, None, cost, i) {
            self.session.checkpoint(&bytes, Some(cost), Some(i));
            for mirror in &self.mirror_sessions {
                mirror.checkpoint(&bytes, Some(cost), Some(i));
            }
        }
    }
}

impl<'a> OuterObjective for CheckpointingObjective<'a> {
    fn capability(&self) -> OuterCapability {
        self.inner.capability()
    }

    fn eval_cost(&mut self, rho: &Array1<f64>) -> Result<f64, EstimationError> {
        let v = self.inner.eval_cost(rho)?;
        // `eval_cost` carries no inner-β handle — persist ρ-only.
        self.note(rho, None, v);
        Ok(v)
    }

    fn eval_screening_proxy(&mut self, rho: &Array1<f64>) -> Result<f64, EstimationError> {
        // Screening proxies run at sub-converged β̂ and aren't a meaningful
        // best-so-far signal; forward without persisting.
        self.inner.eval_screening_proxy(rho)
    }

    fn eval(&mut self, rho: &Array1<f64>) -> Result<OuterEval, EstimationError> {
        let r = self.inner.eval(rho)?;
        self.note(rho, r.inner_beta_hint.as_ref(), r.cost);
        Ok(r)
    }

    fn eval_with_order(
        &mut self,
        rho: &Array1<f64>,
        order: OuterEvalOrder,
    ) -> Result<OuterEval, EstimationError> {
        let r = self.inner.eval_with_order(rho, order)?;
        self.note(rho, r.inner_beta_hint.as_ref(), r.cost);
        Ok(r)
    }

    fn eval_efs(&mut self, rho: &Array1<f64>) -> Result<EfsEval, EstimationError> {
        let r = self.inner.eval_efs(rho)?;
        // EfsEval has no inner-β hint surface yet — persist ρ-only.
        self.note(rho, None, r.cost);
        Ok(r)
    }

    fn seed_inner_state(&mut self, beta: &Array1<f64>) -> Result<SeedOutcome, EstimationError> {
        // Forward to the wrapped objective, then prime our last-inner-beta
        // cache so a subsequent finalize-write encodes the seeded β if no
        // eval surfaces a fresher β first. Only prime on actual install —
        // `NoSlot` means the inner solver will not see β, so the cache
        // entry would be a lie.
        let result = self.inner.seed_inner_state(beta);
        if matches!(result, Ok(SeedOutcome::Installed))
            && beta.iter().all(|v| v.is_finite())
            && let Ok(mut guard) = self.last_inner_beta.lock()
        {
            *guard = Some(beta.clone());
        }
        result
    }

    fn allow_continuation_prewarm(&self) -> bool {
        self.inner.allow_continuation_prewarm()
    }

    fn requires_continuation_path_entry(&self) -> bool {
        self.inner.requires_continuation_path_entry()
    }

    fn reset(&mut self) {
        self.inner.reset();
    }
}

/// Closure-based adapter for [`OuterObjective`].
///
/// This allows any call site to construct an `OuterObjective` from closures
/// without needing to define a wrapper struct or modify the state type.
/// Each call site wraps its existing methods into closures and passes them here.
pub struct ClosureObjective<
    S,
    Fc,
    Fe,
    Fr = fn(&mut S),
    Fefs = fn(&mut S, &Array1<f64>) -> Result<EfsEval, EstimationError>,
    Feo = fn(&mut S, &Array1<f64>, OuterEvalOrder) -> Result<OuterEval, EstimationError>,
    Fsp = fn(&mut S, &Array1<f64>) -> Result<f64, EstimationError>,
    Fseed = fn(&mut S, &Array1<f64>) -> Result<SeedOutcome, EstimationError>,
> {
    pub(crate) state: S,
    pub(crate) cap: OuterCapability,
    pub(crate) cost_fn: Fc,
    pub(crate) eval_fn: Fe,
    /// Optional order-aware eval closure. When `None`, `eval_with_order()`
    /// falls back to `eval()`.
    pub(crate) eval_order_fn: Option<Feo>,
    /// Optional reset closure. When `None`, `reset()` is a no-op.
    pub(crate) reset_fn: Option<Fr>,
    /// Optional EFS evaluation closure. When `None`, the default
    /// `OuterObjective::eval_efs` returns an error.
    pub(crate) efs_fn: Option<Fefs>,
    /// Optional seed-screening ranking proxy closure. When `None`,
    /// `eval_screening_proxy()` falls back to `eval_cost()` (the trait
    /// default), preserving legacy behavior for non-REML objectives.
    pub(crate) screening_proxy_fn: Option<Fsp>,
    /// Optional inner-state seeding closure. Objectives with PIRLS / Newton
    /// inner state install cached β here before the first outer eval.
    pub(crate) seed_fn: Option<Fseed>,
    /// Whether a seed hook should also opt the objective into generic
    /// continuation pre-warm. High-dimensional REML keeps the seed hook for
    /// cache/warm-start replay but declines the expensive rho-anneal pre-pass.
    pub(crate) continuation_prewarm: bool,
}

impl<S, Fc, Fe, Fr, Fefs, Feo, Fsp, Fseed> OuterObjective
    for ClosureObjective<S, Fc, Fe, Fr, Fefs, Feo, Fsp, Fseed>
where
    Fc: FnMut(&mut S, &Array1<f64>) -> Result<f64, EstimationError>,
    Fe: FnMut(&mut S, &Array1<f64>) -> Result<OuterEval, EstimationError>,
    Fr: FnMut(&mut S),
    Fefs: FnMut(&mut S, &Array1<f64>) -> Result<EfsEval, EstimationError>,
    Feo: FnMut(&mut S, &Array1<f64>, OuterEvalOrder) -> Result<OuterEval, EstimationError>,
    Fsp: FnMut(&mut S, &Array1<f64>) -> Result<f64, EstimationError>,
    Fseed: FnMut(&mut S, &Array1<f64>) -> Result<SeedOutcome, EstimationError>,
{
    fn capability(&self) -> OuterCapability {
        self.cap.clone()
    }

    fn eval_cost(&mut self, rho: &Array1<f64>) -> Result<f64, EstimationError> {
        crate::solver::estimate::reml::outer_eval::record_current_outer_theta_for_ift(rho);
        (self.cost_fn)(&mut self.state, rho)
    }

    fn eval_screening_proxy(&mut self, rho: &Array1<f64>) -> Result<f64, EstimationError> {
        crate::solver::estimate::reml::outer_eval::record_current_outer_theta_for_ift(rho);
        match self.screening_proxy_fn.as_mut() {
            Some(f) => f(&mut self.state, rho),
            None => (self.cost_fn)(&mut self.state, rho),
        }
    }

    fn eval(&mut self, rho: &Array1<f64>) -> Result<OuterEval, EstimationError> {
        crate::solver::estimate::reml::outer_eval::record_current_outer_theta_for_ift(rho);
        (self.eval_fn)(&mut self.state, rho)
    }

    fn eval_with_order(
        &mut self,
        rho: &Array1<f64>,
        order: OuterEvalOrder,
    ) -> Result<OuterEval, EstimationError> {
        crate::solver::estimate::reml::outer_eval::record_current_outer_theta_for_ift(rho);
        match self.eval_order_fn.as_mut() {
            Some(f) => f(&mut self.state, rho, order),
            None => (self.eval_fn)(&mut self.state, rho),
        }
    }

    fn eval_efs(&mut self, rho: &Array1<f64>) -> Result<EfsEval, EstimationError> {
        crate::solver::estimate::reml::outer_eval::record_current_outer_theta_for_ift(rho);
        match self.efs_fn.as_mut() {
            Some(f) => f(&mut self.state, rho),
            None => Err(EstimationError::RemlOptimizationFailed(
                "EFS evaluation not implemented for this objective".to_string(),
            )),
        }
    }

    fn seed_inner_state(&mut self, beta: &Array1<f64>) -> Result<SeedOutcome, EstimationError> {
        // Empty β: by convention, "no warm-start available" — treat as a
        // no-op install. Distinct from `NoSlot` because the objective may
        // very well have a slot; the caller just didn't supply a β to fill
        // it. Reporting `Installed` is correct: the slot's pre-existing
        // state (cold default) is the post-seed state.
        if beta.is_empty() {
            return Ok(SeedOutcome::Installed);
        }
        match self.seed_fn.as_mut() {
            Some(f) => f(&mut self.state, beta),
            // No hook installed — the objective owns no inner-β slot.
            // The caller decides whether this is a loud cache-provenance
            // event or a silent continuation-walk degradation.
            None => Ok(SeedOutcome::NoSlot),
        }
    }

    fn allow_continuation_prewarm(&self) -> bool {
        self.continuation_prewarm && self.seed_fn.is_some()
    }

    fn reset(&mut self) {
        if let Some(f) = self.reset_fn.as_mut() {
            f(&mut self.state);
        }
    }
}

impl<S, Fc, Fe, Fr, Fefs, Feo, Fsp> ClosureObjective<S, Fc, Fe, Fr, Fefs, Feo, Fsp>
where
    Fc: FnMut(&mut S, &Array1<f64>) -> Result<f64, EstimationError>,
    Fe: FnMut(&mut S, &Array1<f64>) -> Result<OuterEval, EstimationError>,
    Fr: FnMut(&mut S),
    Fefs: FnMut(&mut S, &Array1<f64>) -> Result<EfsEval, EstimationError>,
    Feo: FnMut(&mut S, &Array1<f64>, OuterEvalOrder) -> Result<OuterEval, EstimationError>,
    Fsp: FnMut(&mut S, &Array1<f64>) -> Result<f64, EstimationError>,
{
    pub fn with_seed_inner_state<Fseed>(
        self,
        seed_fn: Fseed,
    ) -> ClosureObjective<S, Fc, Fe, Fr, Fefs, Feo, Fsp, Fseed>
    where
        Fseed: FnMut(&mut S, &Array1<f64>) -> Result<SeedOutcome, EstimationError>,
    {
        ClosureObjective {
            state: self.state,
            cap: self.cap,
            cost_fn: self.cost_fn,
            eval_fn: self.eval_fn,
            eval_order_fn: self.eval_order_fn,
            reset_fn: self.reset_fn,
            efs_fn: self.efs_fn,
            screening_proxy_fn: self.screening_proxy_fn,
            seed_fn: Some(seed_fn),
            continuation_prewarm: self.continuation_prewarm,
        }
    }
}

pub(crate) fn into_objective_error(context: &str, err: EstimationError) -> ObjectiveEvalError {
    ObjectiveEvalError::recoverable(format!("{context}: {err}"))
}

pub(crate) fn finite_cost_or_error(context: &str, cost: f64) -> Result<f64, ObjectiveEvalError> {
    if cost.is_finite() {
        Ok(cost)
    } else {
        Err(ObjectiveEvalError::recoverable(format!(
            "{context}: objective returned a non-finite cost"
        )))
    }
}

pub(crate) fn finite_outer_eval_or_error(
    context: &str,
    layout: OuterThetaLayout,
    eval: OuterEval,
) -> Result<OuterEval, ObjectiveEvalError> {
    layout.validate_gradient_len(&eval.gradient, context)?;
    if !eval.cost.is_finite() {
        return Err(ObjectiveEvalError::recoverable(format!(
            "{context}: objective returned a non-finite cost"
        )));
    }
    if !eval.gradient.iter().all(|v| v.is_finite()) {
        return Err(ObjectiveEvalError::recoverable(format!(
            "{context}: objective returned a non-finite gradient"
        )));
    }
    match &eval.hessian {
        HessianResult::Analytic(hessian) => {
            layout.validate_hessian_shape(hessian, context)?;
            if !hessian.iter().all(|v| v.is_finite()) {
                return Err(ObjectiveEvalError::recoverable(format!(
                    "{context}: objective returned a non-finite Hessian"
                )));
            }
        }
        HessianResult::Operator(op) => {
            if op.dim() != layout.n_params {
                return Err(ObjectiveEvalError::recoverable(format!(
                    "{context}: outer Hessian operator dimension mismatch: got {}, expected {} (rho_dim={}, psi_dim={})",
                    op.dim(),
                    layout.n_params,
                    layout.rho_dim(),
                    layout.psi_dim
                )));
            }
        }
        HessianResult::Unavailable => {}
    }
    Ok(eval)
}

pub(crate) fn finite_outer_first_order_eval_or_error(
    context: &str,
    layout: OuterThetaLayout,
    eval: OuterEval,
) -> Result<OuterEval, ObjectiveEvalError> {
    layout.validate_gradient_len(&eval.gradient, context)?;
    if !eval.cost.is_finite() {
        return Err(ObjectiveEvalError::recoverable(format!(
            "{context}: objective returned a non-finite cost"
        )));
    }
    if !eval.gradient.iter().all(|v| v.is_finite()) {
        return Err(ObjectiveEvalError::recoverable(format!(
            "{context}: objective returned a non-finite gradient"
        )));
    }
    Ok(eval)
}

pub(crate) fn validate_second_order_seed_hessian(
    context: &str,
    layout: OuterThetaLayout,
    eval: &OuterEval,
) -> Result<(), ObjectiveEvalError> {
    if layout.n_params > SECOND_ORDER_GEOMETRY_PROBE_MAX_PARAMS || !eval.hessian.is_analytic() {
        return Ok(());
    }
    if matches!(
        &eval.hessian,
        HessianResult::Operator(op) if !op.materialization_capability().is_available()
    ) {
        return Ok(());
    }

    let Some(hessian) = eval.hessian.materialize_dense().map_err(|message| {
        ObjectiveEvalError::recoverable(format!(
            "{context}: analytic outer Hessian materialization failed during second-order seed validation: {message}"
        ))
    })?
    else {
        return Ok(());
    };

    layout.validate_hessian_shape(&hessian, context)?;
    if !hessian.iter().all(|value| value.is_finite()) {
        return Err(ObjectiveEvalError::recoverable(format!(
            "{context}: analytic outer Hessian probe encountered non-finite entries"
        )));
    }

    Ok(())
}
