//! Continuation / homotopy seed strategy for the outer REML loop.
//!
//! # Status
//!
//! Stage 1 of a five-stage rollout: this file currently contains only the
//! design note. No driver-file edits and no public API have been wired
//! yet. The module is not mounted in `src/solver/reml/mod.rs` until
//! Stage 2 lands the `ContinuationState` + `fit_with_continuation`
//! implementation. Keeping the design as a doc-comment-only file (no
//! types, no items) means the file compiles to nothing — there is no
//! dead code, no `pub` surface, no `allow(unused)` shim.
//!
//! # Motivation
//!
//! Biobank-scale `survival_marginal_slope` fits (n ≈ 195 780) currently
//! refuse every cold REML startup seed: PIRLS's joint Newton path cannot
//! converge from a generic starting β when ρ — the smoothing parameter
//! — already sits at its target value with very weak time-smoothing.
//! The inner-solver correctness work is happening in parallel
//! (`qd1-constraint-kkt`, `monotone-time-block`). This module addresses
//! a *separate* robustness gap: many real fits have a "hard" ρ region
//! that is reachable from a "soft" one by a smooth path but is not
//! reachable from cold start at any single ρ in that hard region. The
//! classical remedy — used by interior-point methods and simulated
//! annealing — is **continuation**: start at an oversmoothing ρ₀ where
//! the inner problem is strongly convex, and anneal ρ along a finite
//! path ρ₀ → ρ₁ → … → ρ* while warm-starting each step from the
//! previous accepted point.
//!
//! Concretely, the penalised Hessian at smoothing parameter ρ is
//!
//!   H_pen(β; ρ) = H_loglik(β) + Σᵢ exp(ρᵢ) · Sᵢ
//!
//! For large ρ, H_pen is dominated by the (positive semidefinite)
//! penalty block Σᵢ exp(ρᵢ) Sᵢ; if the sum of penalty matrices has
//! full numerical rank over the parameter space (which the canonical
//! penalty roots constructed in `assembly.rs` guarantee per-block), the
//! Newton step becomes globally well-conditioned. The continuation seed
//! drives ρ from this strongly-convex regime down to the user's target
//! ρ* while β tracks the path of inner optima.
//!
//! # Where this plugs into the existing pipeline
//!
//! The outer REML driver lives in `src/solver/outer_strategy.rs::run_outer_with_plan`.
//! That function:
//!   1. Calls `crate::seeding::generate_rho_candidates` to build a
//!      diverse set of ρ seeds (manifold seeds, neutral zero seed,
//!      heuristic-anchored seeds, first-order-fallback seeds).
//!   2. Optionally screens the seed list via `rank_seeds_with_screening`.
//!   3. For each seed, calls `obj.eval_with_order(...)` which runs the
//!      inner solve at that ρ — this is the call that PIRLS / blockwise
//!      Newton refuse from cold β when the fit is hard.
//!   4. Aggregates rejections into a `Vec<SeedRejection>` (built from
//!      the structured `InnerFailure` enum in
//!      `src/families/inner_status.rs`) and produces a `StartupStats`.
//!
//! Continuation slots **between steps 2 and 3**: instead of evaluating
//! each candidate seed `ρ_seed` directly at the target ρ*, the driver
//! evaluates a *path* `[ρ₀(ρ_seed), …, ρ_seed]` and reports the result
//! of the final point. The structured `InnerFailure` plumbing already
//! tells us *why* a step refused, which is exactly what the continuation
//! schedule needs to drive its shrink-on-failure / expand-on-success
//! adaptation. No new failure-classification machinery is required.
//!
//! # Stage 2 surface (planned)
//!
//! ```ignore
//! pub(crate) struct ContinuationState {
//!     /// Last accepted coefficient vector at `last_rho`.
//!     pub beta: Array1<f64>,
//!     /// Active-set indicator for constraint-bearing blocks; carried
//!     /// forward because warm-starting the active set across small ρ
//!     /// steps usually preserves feasibility and saves QP iterations.
//!     pub active_set: ActiveSetSnapshot,
//!     /// Last accepted trust-region radius from the inner Newton solve.
//!     /// Inherited by the next ρ step's inner solve as its initial
//!     /// trust radius; very effective when ρ steps are small.
//!     pub trust_radius: Option<f64>,
//!     /// Allocations that depend only on the design matrix (e.g.,
//!     /// per-row kernel scratch in `latent_inner`), reused across ρ
//!     /// steps to amortise setup cost — same data, different ρ.
//!     pub row_workspace_cache: RowKernelWorkspace,
//!     /// The ρ at which `beta` was accepted; the next step starts here.
//!     pub last_rho: Array1<f64>,
//! }
//!
//! pub(crate) fn fit_with_continuation(
//!     obj: &mut dyn OuterObjective,
//!     target_rho: &Array1<f64>,
//!     schedule: &ContinuationSchedule,
//! ) -> Result<ContinuationState, ContinuationError>;
//! ```
//!
//! `ContinuationError` will carry an `InnerFailure` so that the outer
//! cascade's existing structural-early-exit and `SeedRejection`
//! accounting work unchanged.
//!
//! # Choosing ρ₀ (oversmoothing start)
//!
//! Goal: pick ρ₀ such that the *penalty* term dominates the *likelihood*
//! Hessian at cold β=0, making H_pen well-conditioned regardless of the
//! local data geometry. The rule is:
//!
//!   exp(ρ₀ᵢ) · ‖Sᵢ‖_F ≥ κ · ‖H_loglik(β=0)‖_F  for every penalty i,
//!
//! with κ ≈ 32 (a conservative oversmoothing factor — large enough that
//! the penalty's spectrum dominates by an order of magnitude, small
//! enough that ρ₀ stays inside the bounded ρ box used by `seeding.rs`).
//! Concretely:
//!
//!   ρ₀ᵢ = max( ρ*ᵢ, log(κ · h0 / ‖Sᵢ‖_F) ),
//!
//! where `h0 = ‖H_loglik(β=0)‖_F` is cheap: it is the trace-norm of the
//! likelihood Hessian at cold start, available from the same per-row
//! workspaces PIRLS already builds. `ρ*ᵢ` is the user's target; the
//! max() guarantees that on "easy" fits where the target is already
//! more oversmoothed than κ·h0 requires, ρ₀ collapses to ρ* (i.e.,
//! continuation degenerates to cold start — see §"Magic-by-default").
//!
//! The estimate is computed once per fit attempt; it does *not* require
//! a full PIRLS step. For the biobank survival model the dominant cost
//! is the per-row kernel evaluation, which already lives in
//! `crate::solver::latent_inner` and is cached across continuation
//! steps anyway.
//!
//! # Annealing schedule
//!
//! Geometric, with adaptive shrink-on-failure and expand-on-success.
//! Parameterised by `(α_shrink, α_expand, ρ_step_max)`:
//!
//!   - Predict step k+1: `ρ_{k+1} = ρ_k + α · (ρ* − ρ_k)` with α
//!     starting at `ρ_step_max = 0.5` (halve the remaining distance).
//!   - On accept (inner returned `InnerStatus::Converged` or
//!     `ConstrainedStationary`): expand α ← min(1.0, α · α_expand),
//!     α_expand = 1.5. This lets the schedule grow toward "ρ* in one
//!     step" on well-behaved paths so easy fits cost ≤ 2 inner solves.
//!   - On refuse (inner returned `InnerStatus::Failed(InnerFailure::*)`):
//!     shrink α ← α · α_shrink, α_shrink = 0.5, and retry from the
//!     same `ρ_k`. The previously-accepted ρ_k stays valid; only the
//!     next *target* ρ_{k+1} changes.
//!   - Hard floor on α: 2⁻¹⁰ ≈ 0.001. If the schedule reaches this
//!     floor without making progress, the next strategy is "expand ρ₀
//!     outward" (more oversmoothing).
//!   - Hard ceiling on path length: 64 steps. Past that, return
//!     `ContinuationError::PathBudgetExhausted` carrying the last
//!     `InnerFailure` so the outer cascade reports the structural
//!     reason, not the schedule's own give-up.
//!
//! Adaptive ρ₀ expansion (outer retry): if the full schedule refuses
//! before reaching ρ*, double the oversmoothing factor κ → 2κ → 4κ
//! and restart the path. Capped at three doublings (κ ∈ {32, 64, 128,
//! 256}). Beyond that we declare the fit genuinely infeasible.
//!
//! # Warm-start payload across ρ steps
//!
//! Each step k → k+1 reuses:
//!   - `β_k` (the inner-accepted coefficients) as the initial β for
//!     the inner solve at ρ_{k+1}. The IFT-tangent extrapolation
//!     `β_{k+1}^0 = β_k + (∂β/∂ρ)|_{ρ_k} · (ρ_{k+1} − ρ_k)` is
//!     *not* used in Stage 2 — plain β_k is robust; the IFT tangent
//!     can be a Stage-5 perf experiment.
//!   - Active-set snapshot. For tensor-product / cylinder smooths and
//!     for the survival monotone-time block, the active set is the
//!     dominant cost driver of the inner QP. Empirically the active
//!     set at ρ_k is feasible for ρ_{k+1} after a small step;
//!     verifying feasibility is O(k), much cheaper than rebuilding.
//!   - Trust-region radius. Inheriting the radius preserves the local
//!     geometry the inner solve learned; this is the same trick the
//!     outer matrix-free TR solver uses for budget-bumped retries
//!     (see `OperatorTrustRegionStopReason::IterationBudget` handling
//!     in `outer_strategy.rs`).
//!   - Row-kernel workspaces. These depend only on the design matrix
//!     (predictors, knots, isometry references) and are ρ-independent.
//!     Allocating them once per `fit_with_continuation` call rather
//!     than once per ρ step is the main Stage-5 perf win.
//!
//! # Magic-by-default integration
//!
//! Per the project's "no new CLI flags" rule, continuation is enabled
//! unconditionally. The pre-screen in Stage 3 makes it free when the
//! fit is easy:
//!
//!   1. Estimate h0 = ‖H_loglik(β=0)‖_F at the first seed candidate.
//!      This is one Hessian evaluation we would do anyway for the cold
//!      eval.
//!   2. Compute ρ₀ per the rule above. If ρ₀ ≈ ρ* component-wise
//!      (within 0.5 in log-smoothing units), set `n_steps = 1` and the
//!      continuation reduces to the existing cold-start codepath with
//!      no overhead beyond the h0 estimate.
//!   3. Otherwise schedule a multi-step path. The number of steps is
//!      `ceil(max_i (ρ₀ᵢ − ρ*ᵢ) / log(2))` — i.e., one step per
//!      e-fold of oversmoothing decrease, before adaptive shrinking
//!      kicks in.
//!
//! This means continuation never *adds* inner solves on fits that
//! already cold-start cleanly: a 1-step path is exactly cold start,
//! and the h0 estimate is shared with the cold eval that would have
//! run anyway.
//!
//! # Failure semantics
//!
//! Continuation is *exact at each ρ*. No approximation; the inner
//! solver runs to its usual KKT certificate at every step. This
//! satisfies the "no approximations without opt-in" rule because the
//! only thing continuation changes is *which* ρ the inner solver is
//! asked about, not *how* the inner solver answers.
//!
//! A failure of `fit_with_continuation` (path budget exhausted, ρ₀
//! expansion exhausted, or `InnerFailure::LikelihoodFailure` at ρ₀
//! itself indicating the data is outside the family domain) is
//! reported with the underlying `InnerFailure` carried through, so
//! the outer cascade's `SeedRejection` / `StartupStats` / structural
//! early-exit logic in `outer_strategy.rs` continues to classify
//! these correctly without modification.
//!
//! # Coordination with peer teammates
//!
//! - `seed-accounting`: `InnerFailure` and `InnerStatus` from
//!   `src/families/inner_status.rs` are already stable
//!   (`CertRefused { diagnosis, carrying_block, message }`,
//!   `BudgetExhausted`, `TrustRegionFloor`, `LikelihoodFailure`,
//!   `Other`). Stage 2 can consume them directly.
//! - `qd1-constraint-kkt` / `monotone-time-block`: structural
//!   correctness fixes. Continuation is orthogonal: it lets the
//!   outer loop *reach* the regime where those fixes apply, by
//!   walking β through ρ-space rather than guessing it at ρ*.
//! - `nullspace-lead`, `diagnostician`, `cross-block-audit`:
//!   downstream consumers of the inner solve's KKT certificate; no
//!   contract change from continuation.
