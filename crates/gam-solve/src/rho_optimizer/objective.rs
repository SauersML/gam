use super::*;

// Re-exported here while the shared EFS contract lives in `gam-problem`.
pub use gam_problem::EfsEval;

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
                Ok(OuterEval::value_only(cost, rho.len(), None))
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
    /// (`crate::gpu::reml_outer::run_reml_outer_on_device`). Returns
    /// `Some(adm)` when the objective is a REML evaluator whose
    /// `(spec, n, p, num_rho)` admission predicate accepts the device path,
    /// and `None` otherwise.
    ///
    /// The default returns `None` so non-REML objectives (line-search-only
    /// inner bridges, screening proxies, the EFS / hybrid-EFS sub-objectives)
    /// keep the host BFGS branch unconditionally — only the concrete
    /// REML-state objectives override this to consult
    /// [`crate::estimate::reml::outer_eval::outer_reml_device_admission`].
    fn outer_device_admission(&self) -> Option<gam_gpu::policy::RemlOuterAdmission> {
        None
    }

    /// Whether every joint fit of this objective must ENTER through the
    /// [`crate::continuation_path::ContinuationPath`] (heavy-smoothing
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
        match plan.solver {
            Solver::Efs | Solver::HybridEfs => self.eval_efs(rho).map(|_| ()),
            Solver::Bfgs => self
                .eval_with_order(rho, OuterEvalOrder::ValueAndGradient)
                .map(|_| ()),
            Solver::Arc => self
                .eval_with_order(rho, OuterEvalOrder::ValueGradientHessian)
                .map(|_| ()),
        }
    }
}

// ─── Persistent warm-start checkpoint plumbing ────────────────────────
//
// `CheckpointingObjective` wraps any `OuterObjective` to write a copy of
// `(rho, cost, eval_id)` to disk on each finite evaluation. The on-disk
// [`gam_runtime::warm_start::Session`] rate-limits writes (≥2 s gap unless this iterate
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
    loaded: &gam_runtime::warm_start::LoadedEntry,
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
    if loaded.source == LoadSource::Exact && entry.kind == gam_runtime::warm_start::EntryKind::Final
    {
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

pub fn cache_entry_would_help_outer(
    loaded: &gam_runtime::warm_start::LoadedEntry,
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
    pub state: S,
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
        crate::estimate::reml::outer_eval::record_current_outer_theta_for_ift(rho);
        (self.cost_fn)(&mut self.state, rho)
    }

    fn eval_screening_proxy(&mut self, rho: &Array1<f64>) -> Result<f64, EstimationError> {
        crate::estimate::reml::outer_eval::record_current_outer_theta_for_ift(rho);
        match self.screening_proxy_fn.as_mut() {
            Some(f) => f(&mut self.state, rho),
            None => (self.cost_fn)(&mut self.state, rho),
        }
    }

    fn eval(&mut self, rho: &Array1<f64>) -> Result<OuterEval, EstimationError> {
        crate::estimate::reml::outer_eval::record_current_outer_theta_for_ift(rho);
        (self.eval_fn)(&mut self.state, rho)
    }

    fn eval_with_order(
        &mut self,
        rho: &Array1<f64>,
        order: OuterEvalOrder,
    ) -> Result<OuterEval, EstimationError> {
        crate::estimate::reml::outer_eval::record_current_outer_theta_for_ift(rho);
        match self.eval_order_fn.as_mut() {
            Some(f) => f(&mut self.state, rho, order),
            None => (self.eval_fn)(&mut self.state, rho),
        }
    }

    fn eval_efs(&mut self, rho: &Array1<f64>) -> Result<EfsEval, EstimationError> {
        crate::estimate::reml::outer_eval::record_current_outer_theta_for_ift(rho);
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

/// Shared first-order validation: gradient length, finite cost, finite gradient.
///
/// Extracted so the cost+gradient checks live in exactly one place — both the
/// full (`finite_outer_eval_or_error`) and first-order
/// (`finite_outer_first_order_eval_or_error`) validators delegate here, keeping
/// their error messages and check order bit-for-bit identical.
fn validate_outer_first_order(
    context: &str,
    layout: OuterThetaLayout,
    eval: &OuterEval,
) -> Result<(), ObjectiveEvalError> {
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
    Ok(())
}

pub(crate) fn finite_outer_eval_or_error(
    context: &str,
    layout: OuterThetaLayout,
    eval: OuterEval,
) -> Result<OuterEval, ObjectiveEvalError> {
    validate_outer_first_order(context, layout, &eval)?;
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
    validate_outer_first_order(context, layout, &eval)?;
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

// ─── Permutation-invariant outer coordinate canonicalization ──────────
//
// The additive-term-order (#1539) and tensor-margin-order (#1538) invariance
// bugs share one root cause: the outer smoothing-parameter optimizer resolves
// a flat double-penalty REML valley differently depending on the ORDER the
// penalty blocks are presented (seed placement, multistart, and tie-breaking
// all operate in native penalty-index order). The design and penalty are
// symmetric up to a block permutation, so the cure is permutation-invariance
// by construction: present the optimizer an identical CANONICAL coordinate
// layout regardless of native order, then map the optimized ρ back.
//
// The canonical order is a stable sort of the native coordinates by their
// structural key (see `PenaltyCoordinate::canonical_structural_key`), which is
// derived purely from each penalty's rotation-/placement-invariant content —
// never from its native position. Two formula orders therefore yield the SAME
// canonical layout, so the optimizer's seeding/multistart/tie-break all run on
// byte-identical coordinates and select identical λ̂.

/// Canonical→native index map: `perm[c]` is the native coordinate placed at
/// canonical position `c`.
///
/// Returns `None` when the keys are already in canonical order (the permutation
/// is the identity), so the legacy native-order path runs untouched.
pub(crate) fn canonical_permutation(keys: &[u64]) -> Option<Vec<usize>> {
    let n = keys.len();
    if n <= 1 {
        return None;
    }
    let mut perm: Vec<usize> = (0..n).collect();
    // Stable sort by structural key. Ties (structurally interchangeable
    // coordinates) keep their native relative order — harmless precisely
    // because tied coordinates produce identical fits under any assignment.
    perm.sort_by_key(|&i| keys[i]);
    if perm.iter().enumerate().all(|(c, &i)| c == i) {
        None
    } else {
        Some(perm)
    }
}

/// Reorder a native-layout ρ vector into canonical order: `out[c] = native[perm[c]]`.
fn permute_to_canonical(native: &Array1<f64>, perm: &[usize]) -> Array1<f64> {
    Array1::from_iter(perm.iter().map(|&i| native[i]))
}

/// Reorder a canonical-layout ρ vector back into native order:
/// `out[perm[c]] = canonical[c]`.
fn permute_to_native(canonical: &Array1<f64>, perm: &[usize]) -> Array1<f64> {
    let mut out = Array1::zeros(canonical.len());
    for (c, &i) in perm.iter().enumerate() {
        out[i] = canonical[c];
    }
    out
}

/// Map an `OuterResult` produced in CANONICAL coordinate order back to the
/// objective's native layout, in place. Permutes every per-coordinate array
/// (ρ, gradient, Hessian) consistently; scalar and diagnostic fields are
/// untouched.
pub(crate) fn outer_result_to_native(mut result: OuterResult, perm: &[usize]) -> OuterResult {
    if result.rho.len() == perm.len() {
        result.rho = permute_to_native(&result.rho, perm);
    }
    if let Some(g) = result.final_gradient.as_ref()
        && g.len() == perm.len()
    {
        result.final_gradient = Some(permute_to_native(g, perm));
    }
    if let Some(h) = result.final_hessian.as_ref()
        && h.nrows() == perm.len()
        && h.ncols() == perm.len()
    {
        // H_native[perm[a], perm[b]] = H_canon[a, b].
        let mut hn = Array2::<f64>::zeros((perm.len(), perm.len()));
        for (a, &ia) in perm.iter().enumerate() {
            for (b, &ib) in perm.iter().enumerate() {
                hn[[ia, ib]] = h[[a, b]];
            }
        }
        result.final_hessian = Some(hn);
    }
    result
}

/// Wraps any [`OuterObjective`] so the optimizer can work in a CANONICAL
/// coordinate order while the wrapped objective continues to receive ρ in its
/// NATIVE order. The optimizer hands canonical ρ to this wrapper; the wrapper
/// permutes canonical→native before forwarding to the inner objective, so the
/// inner objective (and any checkpointing/cache layer beneath it) sees native
/// ρ exactly as before. Capability shape (`n_params`, `psi_dim`, …) is
/// unchanged — only coordinate order differs.
pub(crate) struct CanonicalizedObjective<'a> {
    inner: &'a mut dyn OuterObjective,
    /// Canonical→native map: `perm[c]` is the native index at canonical slot `c`.
    perm: Vec<usize>,
}

impl<'a> CanonicalizedObjective<'a> {
    pub(crate) fn new(inner: &'a mut dyn OuterObjective, perm: Vec<usize>) -> Self {
        Self { inner, perm }
    }

    #[inline]
    fn to_native(&self, canonical: &Array1<f64>) -> Array1<f64> {
        if canonical.len() == self.perm.len() {
            permute_to_native(canonical, &self.perm)
        } else {
            // Defensive: a length the permutation does not cover is forwarded
            // verbatim rather than corrupted (should not occur for ρ-coords).
            canonical.clone()
        }
    }

    /// Map a native-order eval (gradient/Hessian) back into canonical order so
    /// the optimizer sees a self-consistent canonical objective.
    fn eval_to_canonical(&self, mut eval: OuterEval) -> OuterEval {
        if eval.gradient.len() == self.perm.len() {
            eval.gradient = permute_to_canonical(&eval.gradient, &self.perm);
        }
        eval.hessian = match eval.hessian {
            HessianResult::Analytic(h)
                if h.nrows() == self.perm.len() && h.ncols() == self.perm.len() =>
            {
                let mut hc = Array2::<f64>::zeros((self.perm.len(), self.perm.len()));
                for (a, &ia) in self.perm.iter().enumerate() {
                    for (b, &ib) in self.perm.iter().enumerate() {
                        hc[[a, b]] = h[[ia, ib]];
                    }
                }
                HessianResult::Analytic(hc)
            }
            other => other,
        };
        // `inner_beta_hint` is in the coefficient basis (not ρ-coordinate
        // order), so it is forwarded unchanged.
        eval
    }
}

impl<'a> OuterObjective for CanonicalizedObjective<'a> {
    fn capability(&self) -> OuterCapability {
        self.inner.capability()
    }

    fn eval_cost(&mut self, rho: &Array1<f64>) -> Result<f64, EstimationError> {
        let native = self.to_native(rho);
        self.inner.eval_cost(&native)
    }

    fn eval_screening_proxy(&mut self, rho: &Array1<f64>) -> Result<f64, EstimationError> {
        let native = self.to_native(rho);
        self.inner.eval_screening_proxy(&native)
    }

    fn eval(&mut self, rho: &Array1<f64>) -> Result<OuterEval, EstimationError> {
        let native = self.to_native(rho);
        let eval = self.inner.eval(&native)?;
        Ok(self.eval_to_canonical(eval))
    }

    fn eval_with_order(
        &mut self,
        rho: &Array1<f64>,
        order: OuterEvalOrder,
    ) -> Result<OuterEval, EstimationError> {
        let native = self.to_native(rho);
        let eval = self.inner.eval_with_order(&native, order)?;
        Ok(self.eval_to_canonical(eval))
    }

    fn eval_efs(&mut self, rho: &Array1<f64>) -> Result<EfsEval, EstimationError> {
        let native = self.to_native(rho);
        let mut efs = self.inner.eval_efs(&native)?;
        // `steps` has one entry per θ-coordinate (length = n_rho + n_ext). The
        // canonical permutation covers only the leading ρ-coordinate block, so
        // map exactly those native→canonical; any trailing ψ/ext steps keep
        // their position (the canonicalized path is ρ-only, psi_dim == 0).
        let m = self.perm.len();
        if efs.steps.len() >= m {
            let leading = Array1::from_iter(efs.steps.iter().take(m).copied());
            let canon_leading = permute_to_canonical(&leading, &self.perm);
            for (c, v) in canon_leading.iter().enumerate() {
                efs.steps[c] = *v;
            }
        }
        Ok(efs)
    }

    fn reset(&mut self) {
        self.inner.reset();
    }

    fn seed_inner_state(&mut self, beta: &Array1<f64>) -> Result<SeedOutcome, EstimationError> {
        // β is in the coefficient basis, not ρ-coordinate order — forward as-is.
        self.inner.seed_inner_state(beta)
    }

    fn allow_continuation_prewarm(&self) -> bool {
        self.inner.allow_continuation_prewarm()
    }

    fn requires_continuation_path_entry(&self) -> bool {
        self.inner.requires_continuation_path_entry()
    }

    fn accept_seed_without_outer_iterations(
        &mut self,
        rho: &Array1<f64>,
    ) -> Result<Option<f64>, EstimationError> {
        let native = self.to_native(rho);
        self.inner.accept_seed_without_outer_iterations(&native)
    }

    fn curvature_homotopy_entry(
        &mut self,
        rho: &Array1<f64>,
    ) -> Option<Result<bool, EstimationError>> {
        let native = self.to_native(rho);
        self.inner.curvature_homotopy_entry(&native)
    }

    fn finalize_outer_result(
        &mut self,
        rho: &Array1<f64>,
        plan: &OuterPlan,
    ) -> Result<(), EstimationError> {
        let native = self.to_native(rho);
        self.inner.finalize_outer_result(&native, plan)
    }

    fn outer_device_admission(&self) -> Option<gam_gpu::policy::RemlOuterAdmission> {
        // The device path optimizes in its own coordinate layout; canonicalized
        // problems route through the host BFGS/ARC path (where the permutation
        // is honored) rather than the device driver.
        None
    }
}
