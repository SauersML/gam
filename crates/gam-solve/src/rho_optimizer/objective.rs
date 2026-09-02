use super::*;
use super::rail_face::RailFaceLimitOutcome;

// Re-exported here while the shared EFS contract lives in `gam-problem`.
pub use gam_problem::{EfsEval, FixedPointCertificateEval, FixedPointCoordinateCertificate};

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
/// - The typed reactive continuation path forwards `inner_beta_hint` from the
///   previous solved waypoint; if the objective has no β slot the path
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
/// - `eval()` may return `HessianValue::Unavailable` at individual trial
///   points even when `capability().hessian == Analytic`; `opt` degrades that
///   step to first-order behavior instead of requiring the objective to fake a
///   stale or non-finite Hessian.
/// - Use `eval_cost()` / `OuterEval::infeasible()` for infeasible trial points.
///   Return `Err(...)` only when the evaluation artifact itself cannot be
///   constructed. Such errors are fatal across screening, multistart, and
///   solver plans; they are never reinterpreted as another numerical trial.
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

    /// Re-evaluate the terminal fixed point and provide an explicit analytic
    /// residual for every optimized coordinate.
    ///
    /// This is a proof surface, not an alias for [`Self::eval_efs`]: iteration
    /// steps may contain guarded or structurally unsupported zeros. The default
    /// refuses certification so an EFS-capable objective must deliberately
    /// describe complete, root-equivalent coordinate coverage before a fixed-
    /// point result can mint a fit.
    fn eval_fixed_point_certificate(
        &mut self,
        rho: &Array1<f64>,
    ) -> Result<FixedPointCertificateEval, EstimationError> {
        Err(EstimationError::RemlOptimizationFailed(format!(
            "fixed-point certification not implemented for this objective at rho_dim={}",
            rho.len()
        )))
    }

    /// Analytic λ→∞ limit data for a rail face (#2348 Inc 5).
    ///
    /// `face` lists the ρ-coordinates sitting at their infinite-smoothing
    /// bound. An objective that can form the limit EXACTLY — the fit
    /// restricted to the railed penalties' common null space, together with
    /// the analytic first-order form of the criterion's logdet and trace terms
    /// there — returns it here, and the outer certificate proves the face from
    /// it instead of probing a tail at finite λ.
    ///
    /// A decline is not a failure, and it is typed: `OutsideClosedForm` leaves
    /// room for a different closed form to apply, while `FaceUnavailable` is a
    /// statement about the face itself. Either way the caller keeps whatever
    /// evidence it already had. The default declines for every objective that
    /// has no analytic limit at all.
    fn rail_face_limit(
        &mut self,
        rho: &Array1<f64>,
        face: &[usize],
    ) -> Result<RailFaceLimitOutcome, EstimationError> {
        if face.iter().any(|&k| k >= rho.len()) {
            return Err(EstimationError::RemlOptimizationFailed(format!(
                "rail face {face:?} is outside the rho layout of dimension {}",
                rho.len()
            )));
        }
        Ok(RailFaceLimitOutcome::OutsideClosedForm {
            reason: "this objective has no analytic face limit".to_string(),
        })
    }

    /// The ρ-gradient of the soft numerical-guard BARRIER this objective adds
    /// to its criterion, if it adds one (#2545).
    ///
    /// `None` — the default — means "this objective carries no such barrier",
    /// and every consumer then behaves exactly as it did before this seam
    /// existed. `Some(g)` must be the barrier's own gradient at `rho`, in the
    /// same coordinate order as [`Self::eval`]'s gradient and of the same
    /// length, computed by the SAME code path the criterion used to ADD it. It
    /// must not be re-derived from the policy constants: the REML barrier is
    /// evaluated at the weight-anchored coordinate `ρ̃ = ρ − log g(w)`, so a
    /// closed form written against raw ρ agrees with the criterion only on
    /// unweighted fits and disagrees on every weighted one.
    ///
    /// "Same length as [`Self::eval`]'s gradient" means the full θ, including
    /// any trailing ψ/link block. The barrier acts on ρ only, so those entries
    /// are EXACTLY zero — never omitted, and never filled with the ρ block
    /// shifted along. [`ClosureObjective`] does that embedding from the
    /// declared [`OuterThetaLayout`] so no implementor writes the arithmetic
    /// (#2629).
    ///
    /// # What this is for, and the line it must not cross
    ///
    /// A `log cosh` barrier's gradient SATURATES at `w·a` instead of decaying,
    /// so at an upper rail the KKT projection (`gi.max(0.0)`) retains exactly
    /// that positive part and `|Pg| ≥ w·a` however clean the fit — a λ=∞ face
    /// can never register as stationary. The certificate therefore subtracts
    /// this term where the barrier provably is not part of the optimality
    /// condition: at coordinates pinned to a box bound (the box enforces the
    /// bound exactly, which is the barrier's entire job) and along the tail
    /// probes (where the `ĉ = −e^ρ·∂V/∂ρ` law is a statement about the
    /// criterion's data term, and the barrier is a known additive analytic
    /// term on top of it).
    ///
    /// It deliberately does NOT subtract at an INTERIOR stationarity test.
    /// There the optimizer descends the criterion WITH the barrier and stops
    /// where their sum vanishes; a certificate that judged the sum minus the
    /// barrier would judge a different function than was optimized and
    /// manufacture "solver converged, certificate refused" out of the
    /// disagreement.
    fn soft_rho_guard_gradient(&mut self, rho: &Array1<f64>) -> Option<Array1<f64>> {
        log::trace!(
            "[#2545] this objective declares no soft rho-guard barrier (rho_dim={})",
            rho.len()
        );
        None
    }

    /// Directions of the outer coordinate along which this criterion is EXACTLY
    /// constant by construction, at `theta` (#2676).
    ///
    /// Orthonormal columns, `theta.len() x d`, or `None` when the objective
    /// declares no such invariance — which is the default and which reproduces
    /// every pre-#2676 verdict bit for bit.
    ///
    /// A penalized criterion sees `lambda` only through
    /// `sum_i lambda_i (beta - mu_i)' S_i (beta - mu_i)`, so any `w` with
    /// `sum_i w_i S_i = 0` (plus the two conditions a nonzero `mu_i` imposes)
    /// leaves it unchanged along `lambda + s w`. Lifted to `rho = log lambda`
    /// by `t = diag(lambda)^{-1} w`, the exact chain rule
    /// `H_rho = diag(lambda) H_lambda diag(lambda) + diag(g_rho)` gives
    /// `t' H_rho t = sum_k g_k t_k^2` — the curvature there is a function of the
    /// GRADIENT, which the certificate has separately judged against its
    /// stationarity bound. Judging it again as curvature, against a floor that
    /// is the same quantity's absolute value, decides the certificate on a
    /// rounding residual.
    ///
    /// The certificate therefore deflates these directions before any PSD test
    /// and judges their orthogonal complement by the unchanged rule. See
    /// [`crate::penalty_invariance`] for the derivation, what deflating cannot
    /// hide, and why a wider floor is the wrong answer.
    ///
    /// # Who has opted in, and who has not
    ///
    /// Installed by the two REML arms and the spatial joint arm, whose criterion
    /// is built on a [`gam_terms::construction::CanonicalPenalty`] bundle that
    /// `PenaltyMapInvariance` reads directly.
    ///
    /// NOT installed on two routes that also set `require_measured_psd`,
    /// because on both of them the map from the penalty layout onto the outer
    /// rho vector is not derivable from where the objective is built — and a
    /// WRONG map deflates a direction the criterion is not flat along, which is
    /// strictly worse than deflating nothing:
    ///
    /// * the **custom-family** route, whose penalties live as
    ///   `ParameterBlockSpec::penalties` plus (for the one family that has them)
    ///   a `JointPenaltyBundle`;
    /// * the **n-block exact-joint spatial** route, whose criterion is a
    ///   caller-supplied evaluator (`exact_fn`) rather than a `RemlState`, so
    ///   the block-major concatenation of per-block penalties into rho is the
    ///   caller's contract and not visible here.
    ///
    /// Both keep the default, which is exactly the pre-#2676 behaviour, so
    /// nothing on them regresses. Neither carries a #2676 fixture: the
    /// `geo_disease_*_matern` cells route through `standard REML` and
    /// `iso-kappa joint REML`, both of which do install it.
    fn criterion_invariant_directions(&mut self, theta: &Array1<f64>) -> Option<Array2<f64>> {
        log::trace!(
            "[#2676] this objective declares no criterion invariance (theta_dim={})",
            theta.len()
        );
        None
    }

    /// Restore to a clean baseline for the next multi-start candidate.
    fn reset(&mut self);

    /// Whether this objective owns a terminal *coefficient* mode whose bitwise
    /// identity fit assembly will later bind against the certified outer value.
    ///
    /// The certification sequence (`run.rs`) installs the terminal state twice
    /// at `result.rho`: once via [`Self::finalize_outer_result`] (which the
    /// mode-owning evaluator uses to install its coefficient mode) and once via
    /// the analytic re-evaluation inside `certify_outer_optimality` (which sets
    /// `result.final_value`). On a nonconvex profiled objective those two
    /// evaluations can settle in *different* coefficient basins unless each is
    /// forced to re-install from the same clean baseline through [`Self::reset`]
    /// — otherwise they prime the inner solve off whatever warm state the
    /// preceding diagnostic/finalize left behind, and the mode's objective and
    /// the certified value disagree by a whole basin (measured: `9.1931e2` vs
    /// `9.1671e2` on the cause-specific survival gate).
    ///
    /// That terminal reset is otherwise gated on `config.outer_inner_cap`,
    /// which the REML/mixture objectives wire but the custom-family (and any
    /// other terminal-mode-owning closure) objective does not — it holds its
    /// inner cap in a different field and leaves `outer_inner_cap` `None`, so
    /// the reset never fires and the bitwise bind can spuriously fail on a
    /// bimodal inner solve. Returning `true` here forces the terminal reset
    /// *independently of the cap*, so `finalize` and `certify` provably come
    /// from one fresh evaluation at `rho_star`. It deliberately does NOT touch
    /// the `inner_solve_converged(config.outer_inner_cap)` gate: an objective
    /// that owns a terminal mode but does not populate the cap's convergence
    /// atomic keeps its own stateful convergence semantics.
    ///
    /// The default is `false`: an objective that owns no terminal coefficient
    /// mode (the reactive-domain fixture among them) retains the very state its
    /// evaluation at `result.rho` depends on and must not be reset.
    fn owns_terminal_coefficient_mode(&self) -> bool {
        false
    }

    /// Transition an objective that actually used an approximate derivative
    /// pilot to its exact full-data measure.
    ///
    /// The runner calls this once after the pilot solver returns a checkpoint.
    /// `true` means the objective changed measure and must be optimized again
    /// from that checkpoint before analytic certification. Exact objectives and
    /// pilots that never installed a sample return `false`.
    fn begin_exact_polish(&mut self) -> bool {
        false
    }

    /// Seed the inner-solver iterate before the first eval, e.g. when the
    /// outer-iterate cache restored a `(ρ, β)` pair from a prior run, or
    /// when a typed reactive continuation path forwards
    /// `OuterEval::inner_beta_hint`
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
    /// the variant. Callers that don't care (the reactive continuation path,
    /// which only proceeds cold when the hint is unusable) ignore it and only
    /// propagate `Err`.
    fn seed_inner_state(&mut self, beta: &Array1<f64>) -> Result<SeedOutcome, EstimationError>;

    /// Optional objective-owned hard upper domain for the outer coordinates.
    ///
    /// The generic optimizer intersects this vector with its configured box
    /// before projecting seeds, constructing a solver, or opening reactive
    /// continuation. Consequently the exact same upper endpoint is both the
    /// solver's legal box face and the continuation path's literal rho entry.
    /// `None` means the objective has no domain narrower than the configured
    /// generic box. An advertised vector must have `capability().n_params`
    /// finite entries; malformed contracts are typed runner errors.
    fn outer_domain_upper_bound(&self) -> Result<Option<Array1<f64>>, EstimationError> {
        Ok(None)
    }

    /// Optional objective-owned hard lower domain for the outer coordinates.
    ///
    /// This is intersected with the caller's configured box at the same single
    /// runner seam as [`Self::outer_domain_upper_bound`], before any seed,
    /// continuation waypoint, solver evaluation, or stationarity certificate can
    /// observe an out-of-domain coordinate.
    fn outer_domain_lower_bound(&self) -> Result<Option<Array1<f64>>, EstimationError> {
        Ok(None)
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
    /// `crate::estimate::reml::outer_eval::outer_reml_device_admission`.
    fn outer_device_admission(&self) -> Option<gam_gpu::policy::RemlOuterAdmission> {
        None
    }

    /// Typed scalar continuation contract for repairing a non-finite literal
    /// outer seed through [`crate::continuation_path::ContinuationPath`].
    ///
    /// This is a typed domain-entry capability, not a fallback objective. The
    /// objective supplies both the smoother entry state and its literal target
    /// state. `None` means this objective has no such domain homotopy. The
    /// runner always probes the real seed first, so merely supplying a contract
    /// performs no waypoint installation or heavy work on a finite seed.
    fn reactive_domain_scalar_contract(
        &self,
    ) -> Result<Option<crate::continuation_path::ContinuationScalarContract>, EstimationError> {
        Ok(None)
    }

    /// Install one scalar waypoint before the continuation rho spine evaluates
    /// the objective. Objectives that return `Some` from
    /// [`Self::reactive_domain_scalar_contract`] must override this method; the
    /// default is a typed contract refusal, never a silent no-op.
    fn install_reactive_domain_scalar_state(
        &mut self,
        state: &crate::continuation_path::ContinuationScalarState,
    ) -> Result<(), EstimationError> {
        Err(EstimationError::RemlOptimizationFailed(format!(
            "objective supplied a reactive-domain scalar contract but cannot install its \
             waypoint (temperature={}, isometry_dim={})",
            state.assignment_temperature,
            state.isometry_weights.len(),
        )))
    }

    /// Snapshot the objective's complete accepted inner state before a reactive
    /// coupled waypoint is installed. Contract-advertising objectives must make
    /// this transactional: a failed trial is restored by
    /// [`Self::rollback_reactive_domain_waypoint`].
    fn begin_reactive_domain_waypoint(&mut self) -> Result<(), EstimationError> {
        Err(EstimationError::RemlOptimizationFailed(
            "objective supplied a reactive-domain scalar contract but cannot checkpoint a waypoint"
                .to_string(),
        ))
    }

    /// Commit the converged full inner state produced by the value evaluation
    /// at `rho`. A coefficient-only handoff is insufficient: latent coordinates,
    /// routing logits, decoder frames, loss, and scalar state must advance as one
    /// accepted waypoint.
    fn commit_reactive_domain_waypoint(
        &mut self,
        rho: &Array1<f64>,
    ) -> Result<(), EstimationError> {
        Err(EstimationError::RemlOptimizationFailed(format!(
            "objective supplied a reactive-domain scalar contract but cannot commit a waypoint \
             (rho_dim={})",
            rho.len(),
        )))
    }

    /// Restore the full accepted state saved by
    /// [`Self::begin_reactive_domain_waypoint`] after an errored or non-finite
    /// trial.
    fn rollback_reactive_domain_waypoint(&mut self) -> Result<(), EstimationError> {
        Err(EstimationError::RemlOptimizationFailed(
            "objective supplied a reactive-domain scalar contract but cannot roll back a waypoint"
                .to_string(),
        ))
    }

    /// Run the objective's certified curvature-homotopy entry leg, if it has
    /// one, leaving the inner state warm at the real (`η = 1`) objective.
    ///
    /// An objective with a *certified anchor* — a point known by construction to
    /// be the global optimum of a relaxed problem — can replace the blind
    /// multi-seed multistart with a single predictor-corrector walk from that
    /// anchor to the true objective (#1007). The SAE-manifold objective
    /// overrides this: its `η = 0` base-topology relaxation is convex, and a
    /// genuine low-rank (Eckart-Young / SVD) residual ceiling is certified by
    /// `linear_span_anchor` — the `η = 0` endpoint is NOT a linear/affine model
    /// (for curved bases its base columns still embed curvature); "Eckart-Young"
    /// names the rank ceiling, not the chart. The walk in `η` tracks the unique
    /// optimal branch to `η = 1`. The walk monitors the
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

    /// Optional analytic evaluation order that must own the final installed
    /// objective state, independently of the solver plan that found `rho`.
    ///
    /// The default follows the solver (`EFS` finalizes through `eval_efs`,
    /// BFGS through first order, ARC through second order). Stateful profiled
    /// objectives may override this when only one evaluator produces the
    /// ownership payload consumed by fit assembly.
    fn terminal_eval_order(&self) -> Option<OuterEvalOrder> {
        None
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
        let order = self.terminal_eval_order().or(match plan.solver {
            Solver::Efs | Solver::HybridEfs => None,
            Solver::Bfgs => Some(OuterEvalOrder::ValueAndGradient),
            Solver::Arc => Some(OuterEvalOrder::ValueGradientHessian),
        });
        match order {
            Some(order) => {
                self.eval_with_order(rho, order)?;
            }
            None => {
                self.eval_efs(rho)?;
            }
        }
        Ok(())
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
/// Schema 3 invalidates every payload written before outer-Hessian provenance
/// was tied to the objective's declared analytic capability. In particular,
/// schema-2 SAE checkpoints may contain the now-deleted finite-difference
/// curvature and must never influence a resumed quasi-Newton metric (#2253).
pub(crate) const ITERATE_PAYLOAD_SCHEMA: u32 = 3;

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
/// The contract is now `(ρ, β)`: the current iterate payload carries
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
    /// Most-recent exact outer/inner state surfaced by one beta-bearing
    /// evaluation. Keeping ρ beside β is load-bearing: scalar certification
    /// probes carry no β, so a bare "last β" can otherwise be paired with a
    /// later certified ρ that never produced it (#2486).
    last_inner_state: std::sync::Mutex<Option<(Array1<f64>, Array1<f64>)>>,
    /// True only while the typed reactive-domain path evaluates an
    /// initialization waypoint. Those waypoints are transactional means of
    /// reaching the literal requested model, not candidate outer iterates, so
    /// they must never become persistent restart seeds.
    reactive_waypoint_active: AtomicBool,
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
            last_inner_state: std::sync::Mutex::new(None),
            reactive_waypoint_active: AtomicBool::new(false),
        }
    }

    pub(crate) fn inner_beta_for(&self, rho: &Array1<f64>) -> Option<Array1<f64>> {
        let guard = self.last_inner_state.lock().ok()?;
        beta_for_exact_rho(guard.as_ref(), rho)
    }

    /// The per-evaluation `(ρ, V, ∇V)` trail, at `debug`.
    ///
    /// Every outer objective the runner drives is wrapped here, so this is the
    /// one place that sees EVERY evaluation of EVERY route — the line search's
    /// trial points included. Without it, a `line_search=StepSizeTooSmall`
    /// refusal ("the direction descended but no step improved the objective")
    /// can only be investigated by rebuilding the design outside the fit and
    /// hoping the reconstruction is the same criterion; #2748's `haberman_5yr`
    /// arm spent a full measurement cycle discovering that a faithful-looking
    /// rebuild disagreed with the fit's own `|g|` by 35x. The trail settles
    /// that question from inside the fit, in the coordinates the solver uses.
    ///
    /// `ρ` is printed in full: the whole point is to be able to difference two
    /// consecutive trial points by hand, and a norm cannot be differenced.
    fn trace_eval(&self, rho: &Array1<f64>, cost: f64, gradient: Option<&Array1<f64>>, what: &str) {
        if !log::log_enabled!(log::Level::Debug) {
            return;
        }
        let gradient_norm = gradient.map_or(f64::NAN, |g| g.dot(g).sqrt());
        log::debug!(
            "[OUTER eval] #{} {what} cost={cost:.15e} |g|={gradient_norm:.6e} rho={:?}",
            self.eval_counter.load(Ordering::Relaxed),
            rho.to_vec(),
        );
    }

    fn note(&self, rho: &Array1<f64>, beta: Option<&Array1<f64>>, cost: f64) {
        if self.reactive_waypoint_active.load(Ordering::Relaxed) {
            return;
        }
        if !cost.is_finite() {
            return;
        }
        // If β is provided, require it to be finite; non-finite β is a
        // divergent inner state — persisting it would re-poison the cache.
        if let Some(b) = beta {
            if !b.iter().all(|v| v.is_finite()) {
                return;
            }
            if let Ok(mut guard) = self.last_inner_state.lock() {
                *guard = Some((rho.clone(), b.clone()));
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

fn beta_for_exact_rho(
    state: Option<&(Array1<f64>, Array1<f64>)>,
    rho: &Array1<f64>,
) -> Option<Array1<f64>> {
    let (producing_rho, beta) = state?;
    (producing_rho.len() == rho.len()
        && producing_rho
            .iter()
            .zip(rho.iter())
            .all(|(left, right)| left.to_bits() == right.to_bits()))
    .then(|| beta.clone())
}

#[cfg(test)]
mod checkpoint_state_pair_tests {
    use super::*;

    #[test]
    fn finalized_beta_requires_its_exact_producing_rho_2486() {
        let state = (
            Array1::from_vec(vec![1.0, -0.0]),
            Array1::from_vec(vec![3.0, 4.0]),
        );

        assert_eq!(
            beta_for_exact_rho(Some(&state), &Array1::from_vec(vec![1.0, -0.0])),
            Some(Array1::from_vec(vec![3.0, 4.0])),
        );
        assert!(
            beta_for_exact_rho(Some(&state), &Array1::from_vec(vec![1.0, 0.0])).is_none(),
            "even numerically equal but bit-distinct rho cannot borrow another evaluation's beta",
        );
        assert!(
            beta_for_exact_rho(Some(&state), &Array1::from_vec(vec![1.0])).is_none(),
            "a shape-mismatched rho cannot borrow beta",
        );
    }
}

impl<'a> OuterObjective for CheckpointingObjective<'a> {
    fn capability(&self) -> OuterCapability {
        self.inner.capability()
    }

    fn eval_cost(&mut self, rho: &Array1<f64>) -> Result<f64, EstimationError> {
        let v = self.inner.eval_cost(rho)?;
        self.trace_eval(rho, v, None, "value");
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
        self.trace_eval(rho, r.cost, Some(&r.gradient), "value+gradient");
        self.note(rho, r.inner_beta_hint.as_ref(), r.cost);
        Ok(r)
    }

    fn eval_with_order(
        &mut self,
        rho: &Array1<f64>,
        order: OuterEvalOrder,
    ) -> Result<OuterEval, EstimationError> {
        let r = self.inner.eval_with_order(rho, order)?;
        self.trace_eval(rho, r.cost, Some(&r.gradient), &format!("{order:?}"));
        self.note(rho, r.inner_beta_hint.as_ref(), r.cost);
        Ok(r)
    }

    fn eval_efs(&mut self, rho: &Array1<f64>) -> Result<EfsEval, EstimationError> {
        let r = self.inner.eval_efs(rho)?;
        // EfsEval has no inner-β hint surface yet — persist ρ-only.
        self.note(rho, None, r.cost);
        Ok(r)
    }

    fn eval_fixed_point_certificate(
        &mut self,
        rho: &Array1<f64>,
    ) -> Result<FixedPointCertificateEval, EstimationError> {
        let r = self.inner.eval_fixed_point_certificate(rho)?;
        self.note(rho, None, r.cost);
        Ok(r)
    }

    fn rail_face_limit(
        &mut self,
        rho: &Array1<f64>,
        face: &[usize],
    ) -> Result<RailFaceLimitOutcome, EstimationError> {
        self.inner.rail_face_limit(rho, face)
    }

    fn soft_rho_guard_gradient(&mut self, rho: &Array1<f64>) -> Option<Array1<f64>> {
        // A barrier gradient is a property of the wrapped criterion; the
        // checkpoint layer neither adds nor persists one.
        self.inner.soft_rho_guard_gradient(rho)
    }

    fn criterion_invariant_directions(&mut self, theta: &Array1<f64>) -> Option<Array2<f64>> {
        // The invariance is a property of the wrapped criterion's penalty map;
        // the checkpoint layer neither adds nor persists one.
        self.inner.criterion_invariant_directions(theta)
    }

    fn seed_inner_state(&mut self, beta: &Array1<f64>) -> Result<SeedOutcome, EstimationError> {
        // Forward to the wrapped objective, then prime our last-inner-beta
        // cache so a subsequent finalize-write encodes the seeded β if no
        // eval surfaces a fresher β first. Only prime on actual install —
        // `NoSlot` means the inner solver will not see β, so the cache
        // entry would be a lie.
        // A donated β has no producing ρ at this API boundary. It may seed the
        // next evaluation, but it cannot become final-state evidence until an
        // evaluation returns it together with the coordinate it was solved at.
        self.inner.seed_inner_state(beta)
    }

    fn terminal_eval_order(&self) -> Option<OuterEvalOrder> {
        self.inner.terminal_eval_order()
    }

    fn owns_terminal_coefficient_mode(&self) -> bool {
        // Forward the wrapped objective's ownership: the terminal reset must
        // still fire for a cap-less mode owner (e.g. a custom family) when its
        // fit routes through a cache session and is wrapped here (#2334).
        self.inner.owns_terminal_coefficient_mode()
    }

    fn reactive_domain_scalar_contract(
        &self,
    ) -> Result<Option<crate::continuation_path::ContinuationScalarContract>, EstimationError> {
        self.inner.reactive_domain_scalar_contract()
    }

    fn install_reactive_domain_scalar_state(
        &mut self,
        state: &crate::continuation_path::ContinuationScalarState,
    ) -> Result<(), EstimationError> {
        self.inner.install_reactive_domain_scalar_state(state)
    }

    fn begin_reactive_domain_waypoint(&mut self) -> Result<(), EstimationError> {
        self.inner.begin_reactive_domain_waypoint()?;
        self.reactive_waypoint_active
            .store(true, Ordering::Relaxed);
        Ok(())
    }

    fn commit_reactive_domain_waypoint(
        &mut self,
        rho: &Array1<f64>,
    ) -> Result<(), EstimationError> {
        let result = self.inner.commit_reactive_domain_waypoint(rho);
        self.reactive_waypoint_active
            .store(false, Ordering::Relaxed);
        result
    }

    fn rollback_reactive_domain_waypoint(&mut self) -> Result<(), EstimationError> {
        let result = self.inner.rollback_reactive_domain_waypoint();
        self.reactive_waypoint_active
            .store(false, Ordering::Relaxed);
        result
    }

    fn reset(&mut self) {
        self.reactive_waypoint_active
            .store(false, Ordering::Relaxed);
        self.inner.reset();
    }

    fn begin_exact_polish(&mut self) -> bool {
        self.inner.begin_exact_polish()
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
    /// dispatches value-only work to `cost_fn` and derivative-bearing work to
    /// `eval_fn`, matching the [`OuterObjective`] default contract.
    pub(crate) eval_order_fn: Option<Feo>,
    /// Optional reset closure. When `None`, `reset()` is a no-op.
    pub(crate) reset_fn: Option<Fr>,
    /// Optional EFS evaluation closure. When `None`, the default
    /// `OuterObjective::eval_efs` returns an error.
    pub(crate) efs_fn: Option<Fefs>,
    pub(crate) fixed_point_certificate_fn: Option<
        Box<dyn FnMut(&mut S, &Array1<f64>) -> Result<FixedPointCertificateEval, EstimationError>>,
    >,
    /// Optional single-shot transition from an approximate derivative pilot to
    /// the exact objective measure.
    pub(crate) exact_polish_fn: Option<Box<dyn FnMut(&mut S) -> bool>>,
    /// Optional analytic λ→∞ rail-face limit hook (#2348 Inc 5). Installed by
    /// objectives whose criterion has an exact closed-form limit at an
    /// infinite-smoothing face; `None` means the outer certificate falls back
    /// to measuring the tail.
    pub(crate) rail_face_limit_fn: Option<
        Box<
            dyn FnMut(
                &mut S,
                &Array1<f64>,
                &[usize],
            ) -> Result<RailFaceLimitOutcome, EstimationError>,
        >,
    >,
    /// Optional soft rho-guard barrier gradient hook (#2545). Installed by
    /// objectives whose criterion carries the unconditional `log cosh` barrier;
    /// `None` means "no barrier", and the certificate subtracts nothing.
    pub(crate) soft_rho_guard_gradient_fn:
        Option<Box<dyn FnMut(&mut S, &Array1<f64>) -> Array1<f64>>>,
    /// Optional criterion-invariance hook (#2676). Installed by objectives whose
    /// penalty map carries an exact linear redundancy; `None` means "no
    /// invariance", and the certificate deflates nothing — the pre-#2676
    /// behaviour, bit for bit.
    pub(crate) criterion_invariance_fn:
        Option<Box<dyn FnMut(&mut S, &Array1<f64>) -> Option<Array2<f64>>>>,
    /// Optional seed-screening ranking proxy closure. When `None`,
    /// `eval_screening_proxy()` falls back to `eval_cost()` (the trait
    /// default), preserving legacy behavior for non-REML objectives.
    pub(crate) screening_proxy_fn: Option<Fsp>,
    /// Optional inner-state seeding closure. Objectives with PIRLS / Newton
    /// inner state install cached β here before the first outer eval.
    pub(crate) seed_fn: Option<Fseed>,
    /// Analytic evaluator that must install the terminal owned state even when
    /// the selected optimization plan itself used EFS.
    pub(crate) terminal_eval_order: Option<OuterEvalOrder>,
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
            None => match order {
                OuterEvalOrder::Value => {
                    let cost = (self.cost_fn)(&mut self.state, rho)?;
                    Ok(OuterEval::value_only(cost, rho.len(), None))
                }
                OuterEvalOrder::ValueAndGradient | OuterEvalOrder::ValueGradientHessian => {
                    (self.eval_fn)(&mut self.state, rho)
                }
            },
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

    fn eval_fixed_point_certificate(
        &mut self,
        rho: &Array1<f64>,
    ) -> Result<FixedPointCertificateEval, EstimationError> {
        crate::estimate::reml::outer_eval::record_current_outer_theta_for_ift(rho);
        match self.fixed_point_certificate_fn.as_mut() {
            Some(f) => f(&mut self.state, rho),
            None => Err(EstimationError::RemlOptimizationFailed(
                "fixed-point certification not implemented for this closure objective".to_string(),
            )),
        }
    }

    fn rail_face_limit(
        &mut self,
        rho: &Array1<f64>,
        face: &[usize],
    ) -> Result<RailFaceLimitOutcome, EstimationError> {
        if face.iter().any(|&k| k >= rho.len()) {
            return Err(EstimationError::RemlOptimizationFailed(format!(
                "rail face {face:?} is outside the rho layout of dimension {}",
                rho.len()
            )));
        }
        match self.rail_face_limit_fn.as_mut() {
            Some(f) => f(&mut self.state, rho, face),
            None => Ok(RailFaceLimitOutcome::OutsideClosedForm {
                reason: "this objective does not implement an analytic face limit".to_string(),
            }),
        }
    }

    fn soft_rho_guard_gradient(&mut self, theta: &Array1<f64>) -> Option<Array1<f64>> {
        // The hook speaks ρ; the seam speaks θ. The barrier acts on ρ only —
        // `RemlState::build_prior` adds it to `grad[..k]` and to nothing else —
        // so on an objective whose outer coordinate is
        // `θ = [ρ (rho_dim), ψ/link (psi_dim)]` the publication must be
        // θ-length with EXACT zeros in the trailing block. Doing that
        // arithmetic here, from the DECLARED layout, rather than at each
        // construction site is what makes the two REML arms install a
        // byte-identical hook: standard REML (`psi_dim = 0`) sees the embedding
        // collapse to the identity, and the mixture/SAS arm gets the zeros it
        // needs without writing a single index (#2629).
        //
        // Why the layout and not `theta.len()`: a misalignment here is silent.
        // Every coordinate's barrier is the same order of magnitude, so
        // subtracting one coordinate's from another's is invisible in the norm
        // and surfaces only as a coordinate that never certifies.
        let layout = self.cap.theta_layout();
        if theta.len() != layout.n_params {
            log::trace!(
                "[#2545/#2629] barrier publication declined: theta length {} is not the \
                 declared n_params {} (rho_dim={}, psi_dim={})",
                theta.len(),
                layout.n_params,
                layout.rho_dim(),
                layout.psi_dim
            );
            return None;
        }
        let rho_dim = layout.rho_dim();
        let rho = theta.slice(ndarray::s![..rho_dim]).to_owned();
        let guard = self.soft_rho_guard_gradient_fn.as_mut()?(&mut self.state, &rho);
        // A hook that answers in the wrong shape is reported as an ABSENCE, not
        // spliced in at whatever length it returned: the consumers index this
        // array by outer coordinate, and a length mismatch would subtract one
        // coordinate's barrier from another's gradient. Reporting `None` costs
        // only the pre-#2545 behavior (the barrier stays in the residual).
        if guard.len() != rho_dim || !guard.iter().all(|v| v.is_finite()) {
            log::trace!(
                "[#2545/#2629] barrier publication declined: the hook returned {} entries \
                 for a rho block of {rho_dim}, or a non-finite one",
                guard.len()
            );
            return None;
        }
        if layout.psi_dim == 0 {
            return Some(guard);
        }
        let mut published = Array1::<f64>::zeros(layout.n_params);
        published.slice_mut(ndarray::s![..rho_dim]).assign(&guard);
        Some(published)
    }

    fn criterion_invariant_directions(&mut self, theta: &Array1<f64>) -> Option<Array2<f64>> {
        // Same seam discipline as the barrier hook above (#2629): the closure
        // speaks rho, the certificate speaks theta, and the psi/link block is
        // EXACTLY zero because the invariance lives entirely in the penalty
        // map. Doing the embedding here, from the declared layout, is what lets
        // the standard-REML and the exact-joint spatial arms install a
        // byte-identical hook.
        let layout = self.cap.theta_layout();
        if theta.len() != layout.n_params {
            log::trace!(
                "[#2676] invariance publication declined: theta length {} is not the declared \
                 n_params {} (rho_dim={}, psi_dim={})",
                theta.len(),
                layout.n_params,
                layout.rho_dim(),
                layout.psi_dim
            );
            return None;
        }
        let rho_dim = layout.rho_dim();
        let rho = theta.slice(ndarray::s![..rho_dim]).to_owned();
        let directions = self.criterion_invariance_fn.as_mut()?(&mut self.state, &rho)?;
        // A hook answering in the wrong shape is reported as an ABSENCE rather
        // than spliced in: deflating a direction that is not the criterion's
        // invariance would remove real curvature from the certificate's view,
        // which is the one failure this whole mechanism must not have.
        if directions.nrows() != rho_dim
            || directions.ncols() == 0
            || !directions.iter().all(|value| value.is_finite())
        {
            log::trace!(
                "[#2676] invariance publication declined: the hook returned a {}x{} block for a \
                 rho block of {rho_dim}, or a non-finite one",
                directions.nrows(),
                directions.ncols(),
            );
            return None;
        }
        if layout.psi_dim == 0 {
            return Some(directions);
        }
        let mut published = Array2::<f64>::zeros((layout.n_params, directions.ncols()));
        published
            .slice_mut(ndarray::s![..rho_dim, ..])
            .assign(&directions);
        Some(published)
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

    fn terminal_eval_order(&self) -> Option<OuterEvalOrder> {
        self.terminal_eval_order
    }

    fn reset(&mut self) {
        if let Some(f) = self.reset_fn.as_mut() {
            f(&mut self.state);
        }
    }

    fn owns_terminal_coefficient_mode(&self) -> bool {
        // A forced terminal eval order is set *precisely* to install this
        // objective's owned coefficient mode through one analytic evaluator at
        // `rho_star` (see `terminal_eval_order`'s field doc and
        // `with_terminal_eval_order`). So `terminal_eval_order.is_some()` is the
        // existing, single-source-of-truth marker that this closure objective
        // owns a terminal coefficient mode — no separate flag to keep in sync.
        // Only the custom-family builder sets it; every other closure objective
        // (REML search proxies, reactive fixtures) leaves it `None` and keeps
        // the default `false`.
        self.terminal_eval_order.is_some()
    }

    fn begin_exact_polish(&mut self) -> bool {
        self.exact_polish_fn
            .as_mut()
            .is_some_and(|transition| transition(&mut self.state))
    }
}

impl<S, Fc, Fe, Fr, Fefs, Feo, Fsp, Fseed> ClosureObjective<S, Fc, Fe, Fr, Fefs, Feo, Fsp, Fseed> {
    pub fn with_exact_polish<Fpolish>(mut self, transition: Fpolish) -> Self
    where
        Fpolish: FnMut(&mut S) -> bool + 'static,
    {
        self.exact_polish_fn = Some(Box::new(transition));
        self
    }

    /// Force final state installation through one analytic evaluator order.
    /// Search-time solver selection remains unchanged.
    pub fn with_terminal_eval_order(mut self, order: OuterEvalOrder) -> Self {
        self.terminal_eval_order = Some(order);
        self
    }

    /// Install the analytic λ→∞ rail-face limit hook (#2348 Inc 5).
    pub fn with_rail_face_limit<Fface>(mut self, limit: Fface) -> Self
    where
        Fface: FnMut(
                &mut S,
                &Array1<f64>,
                &[usize],
            ) -> Result<RailFaceLimitOutcome, EstimationError>
            + 'static,
    {
        self.rail_face_limit_fn = Some(Box::new(limit));
        self
    }

    /// Install the soft rho-guard barrier gradient hook (#2545).
    ///
    /// The closure must PROJECT the barrier gradient the criterion already
    /// added (for REML: `RemlState::soft_rho_guard_gradient`, which reads the
    /// same `SoftRhoGuardPriorAtom` `build_prior` reads), never recompute it
    /// from the policy constants — the barrier is evaluated at the
    /// weight-anchored coordinate, so a raw-ρ closed form is a different
    /// function on any weighted fit.
    ///
    /// The closure speaks **ρ**, not θ: it receives the leading `rho_dim`
    /// entries of the outer point and returns one entry per ρ-coordinate.
    /// [`OuterObjective::soft_rho_guard_gradient`] embeds that into the full θ
    /// with exact zeros in the ψ/link block, so an objective with auxiliary
    /// outer coordinates installs the SAME closure as one without — the layout
    /// arithmetic that the mixture/SAS arm would otherwise have had to
    /// hand-write (and that #2629 records as invisible when wrong) lives in one
    /// place, driven by the declared [`OuterThetaLayout`].
    ///
    /// A closure whose criterion is NOT built on `RemlState` must not install
    /// this hook at all: `None` is the correct answer for an objective that
    /// carries no barrier, and publishing a zero array would be indistinguishable
    /// from publishing a real one at the consumers.
    pub fn with_soft_rho_guard_gradient<Fguard>(mut self, guard: Fguard) -> Self
    where
        Fguard: FnMut(&mut S, &Array1<f64>) -> Array1<f64> + 'static,
    {
        self.soft_rho_guard_gradient_fn = Some(Box::new(guard));
        self
    }

    /// Publish the criterion's exact invariance directions (#2676).
    ///
    /// The closure receives the FULL outer point and returns orthonormal
    /// columns in the same coordinates, so an objective with auxiliary `psi` or
    /// link coordinates supplies the embedding itself (the rho block is what
    /// carries the invariance; every other coordinate is exactly zero).
    ///
    /// An objective whose criterion is not built on a penalty map must not
    /// install this hook: `None` is the correct answer, and publishing an empty
    /// matrix would be indistinguishable from publishing a real one.
    pub fn with_criterion_invariance<Finv>(mut self, invariance: Finv) -> Self
    where
        Finv: FnMut(&mut S, &Array1<f64>) -> Option<Array2<f64>> + 'static,
    {
        self.criterion_invariance_fn = Some(Box::new(invariance));
        self
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
            fixed_point_certificate_fn: self.fixed_point_certificate_fn,
            exact_polish_fn: self.exact_polish_fn,
            rail_face_limit_fn: self.rail_face_limit_fn,
            soft_rho_guard_gradient_fn: self.soft_rho_guard_gradient_fn,
            criterion_invariance_fn: self.criterion_invariance_fn,
            screening_proxy_fn: self.screening_proxy_fn,
            seed_fn: Some(seed_fn),
            terminal_eval_order: self.terminal_eval_order,
        }
    }
}
/// Classify an [`EstimationError`] for the outer objective boundary and
/// carry it across as a typed source.
///
/// # Why this is not a substring match
///
/// Recoverability is a property of what failed, and only the producer
/// knows it. This function used to decide it by testing whether the
/// *rendered message* contained a marker string, because the variant it
/// received — `CustomFamilyError::UnsupportedConfiguration` — meant "the
/// configuration is structurally unsupported" while the condition it
/// actually carried was "the inner solve missed its KKT condition at this
/// one theta". The marker existed to undo that mismatch after the fact.
///
/// The consequence was #2553: the same variant was classified RECOVERABLE
/// at a call site that could still see its type and FATAL here, where
/// only the text was left, so a trial the optimizer was equipped to
/// survive aborted the whole fit. A verdict carried in prose is one
/// `format!` away from silently changing meaning.
///
/// Both halves are fixed at their roots. The producer emits
/// [`CustomFamilyError::InnerSolveNotConverged`], a variant that *means*
/// the trial point is infeasible, and `is_trial_point_infeasible` is an
/// exhaustive match over the variants rather than a guess. `opt`'s
/// `ObjectiveEvalError` then carries the originating error as a typed
/// source, so any later layer that needs the classification downcasts to
/// it instead of re-deriving one.
pub(crate) fn into_objective_error(context: &str, err: EstimationError) -> ObjectiveEvalError {
    let kind = if err.is_trial_point_infeasible() {
        ObjectiveEvalKind::Recoverable
    } else {
        ObjectiveEvalKind::Fatal
    };
    ObjectiveEvalError::from_source(kind, err).with_context(context)
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
        HessianValue::Dense(hessian) => {
            layout.validate_hessian_shape(hessian, context)?;
            if !hessian.iter().all(|v| v.is_finite()) {
                return Err(ObjectiveEvalError::recoverable(format!(
                    "{context}: objective returned a non-finite Hessian"
                )));
            }
        }
        HessianValue::Operator(op) => {
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
        HessianValue::Unavailable => {}
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
        HessianValue::Operator(op) if !op.materialization().is_available()
    ) {
        return Ok(());
    }

    let Some(hessian) = eval.hessian.materialize_dense().map_err(|error| {
        ObjectiveEvalError::recoverable(format!(
            "{context}: analytic outer Hessian materialization failed during second-order seed validation: {error}"
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
            HessianValue::Dense(h)
                if h.nrows() == self.perm.len() && h.ncols() == self.perm.len() =>
            {
                let mut hc = Array2::<f64>::zeros((self.perm.len(), self.perm.len()));
                for (a, &ia) in self.perm.iter().enumerate() {
                    for (b, &ib) in self.perm.iter().enumerate() {
                        hc[[a, b]] = h[[ia, ib]];
                    }
                }
                HessianValue::Dense(hc)
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

    fn terminal_eval_order(&self) -> Option<OuterEvalOrder> {
        self.inner.terminal_eval_order()
    }

    fn owns_terminal_coefficient_mode(&self) -> bool {
        // Forward through the canonicalizing permutation wrapper so a cap-less
        // mode owner (e.g. a custom family) still gets the terminal reset when
        // its outer search runs in a non-identity canonical coordinate layout
        // (#2334). Ownership is coordinate-order-invariant.
        self.inner.owns_terminal_coefficient_mode()
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

    fn eval_fixed_point_certificate(
        &mut self,
        rho: &Array1<f64>,
    ) -> Result<FixedPointCertificateEval, EstimationError> {
        let native = self.to_native(rho);
        let mut evaluation = self.inner.eval_fixed_point_certificate(&native)?;
        if evaluation.coordinates.len() == self.perm.len() {
            evaluation.coordinates = self
                .perm
                .iter()
                .map(|&native_index| evaluation.coordinates[native_index].clone())
                .collect();
        }
        Ok(evaluation)
    }

    fn rail_face_limit(
        &mut self,
        rho: &Array1<f64>,
        face: &[usize],
    ) -> Result<RailFaceLimitOutcome, EstimationError> {
        // The face is a set of ρ-coordinates, so it permutes exactly like ρ.
        let native_rho = self.to_native(rho);
        let mut native_face = Vec::with_capacity(face.len());
        for &canonical in face.iter() {
            match self.perm.get(canonical).copied() {
                Some(native) => native_face.push(native),
                None => {
                    return Ok(RailFaceLimitOutcome::FaceUnavailable {
                        reason: format!(
                            "face coordinate {canonical} is outside the canonical permutation"
                        ),
                    });
                }
            }
        }
        let mut limit = match self.inner.rail_face_limit(&native_rho, &native_face)? {
            RailFaceLimitOutcome::Available(limit) => limit,
            declined => return Ok(declined),
        };
        // The inner objective reports its face in NATIVE indices (and may have
        // reordered it); map back so the certificate names canonical
        // coordinates, keeping every per-coordinate array aligned with it.
        let mut canonical_of_native = vec![usize::MAX; self.perm.len()];
        for (canonical, &native) in self.perm.iter().enumerate() {
            canonical_of_native[native] = canonical;
        }
        let mut canonical_face = Vec::with_capacity(limit.face.len());
        for &native in limit.face.iter() {
            match canonical_of_native.get(native).copied() {
                Some(canonical) if canonical != usize::MAX => canonical_face.push(canonical),
                _ => {
                    return Ok(RailFaceLimitOutcome::FaceUnavailable {
                        reason: format!(
                            "the reported face names native coordinate {native}, which the \
                             permutation does not cover"
                        ),
                    });
                }
            }
        }
        limit.face = canonical_face;
        Ok(RailFaceLimitOutcome::Available(limit))
    }

    fn soft_rho_guard_gradient(&mut self, rho: &Array1<f64>) -> Option<Array1<f64>> {
        // The barrier gradient is one entry per ρ-coordinate, so it permutes
        // exactly like `eval`'s gradient does in `eval_to_canonical`. Forgetting
        // this permutation would subtract a DIFFERENT coordinate's barrier
        // whenever the canonical layout is not the identity — and because every
        // coordinate's barrier is the same order of magnitude, the error would
        // be invisible in the norm and visible only as a coordinate that never
        // certifies.
        let native = self.to_native(rho);
        let guard = self.inner.soft_rho_guard_gradient(&native)?;
        (guard.len() == self.perm.len()).then(|| permute_to_canonical(&guard, &self.perm))
    }

    fn criterion_invariant_directions(&mut self, rho: &Array1<f64>) -> Option<Array2<f64>> {
        // The invariance columns are indexed by rho-coordinate, so their ROWS
        // permute exactly like `eval_to_canonical` permutes the gradient. The
        // columns index the invariance's own basis and do not permute. Getting
        // this wrong would deflate the wrong coordinate pattern — a direction
        // the criterion is NOT flat along — and silently hide real curvature.
        let native = self.to_native(rho);
        let directions = self.inner.criterion_invariant_directions(&native)?;
        if directions.nrows() != self.perm.len() {
            return None;
        }
        let mut canonical = Array2::<f64>::zeros(directions.dim());
        for (canonical_row, &native_row) in self.perm.iter().enumerate() {
            for column in 0..directions.ncols() {
                canonical[[canonical_row, column]] = directions[[native_row, column]];
            }
        }
        Some(canonical)
    }

    fn reset(&mut self) {
        self.inner.reset();
    }

    fn begin_exact_polish(&mut self) -> bool {
        self.inner.begin_exact_polish()
    }

    fn seed_inner_state(&mut self, beta: &Array1<f64>) -> Result<SeedOutcome, EstimationError> {
        // β is in the coefficient basis, not ρ-coordinate order — forward as-is.
        self.inner.seed_inner_state(beta)
    }

    fn reactive_domain_scalar_contract(
        &self,
    ) -> Result<Option<crate::continuation_path::ContinuationScalarContract>, EstimationError> {
        self.inner.reactive_domain_scalar_contract()
    }

    fn install_reactive_domain_scalar_state(
        &mut self,
        state: &crate::continuation_path::ContinuationScalarState,
    ) -> Result<(), EstimationError> {
        self.inner.install_reactive_domain_scalar_state(state)
    }

    fn begin_reactive_domain_waypoint(&mut self) -> Result<(), EstimationError> {
        self.inner.begin_reactive_domain_waypoint()
    }

    fn commit_reactive_domain_waypoint(
        &mut self,
        rho: &Array1<f64>,
    ) -> Result<(), EstimationError> {
        let native = self.to_native(rho);
        self.inner.commit_reactive_domain_waypoint(&native)
    }

    fn rollback_reactive_domain_waypoint(&mut self) -> Result<(), EstimationError> {
        self.inner.rollback_reactive_domain_waypoint()
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

#[cfg(test)]
mod trial_infeasibility_classification_tests {
    use super::*;
    use gam_problem::CustomFamilyError;

    /// #2553: one variant used to get both verdicts depending on which
    /// boundary it crossed, because the boundary read the rendered text.
    /// The classification is now a property of the type, so both call
    /// sites necessarily agree.
    #[test]
    fn inner_solve_nonconvergence_is_recoverable_and_carries_its_type() {
        let err = EstimationError::CustomFamily(CustomFamilyError::InnerSolveNotConverged {
            cycles: 12,
            terminal: None,
            kkt_residual: Some(4.2e-3),
            kkt_tol: Some(1e-8),
            theta_dim: 5,
            rho_dim: 3,
            psi_dim: 2,
        });
        assert!(err.is_trial_point_infeasible());

        let objective_err = into_objective_error("outer fixed-point evaluation", err);
        assert!(
            objective_err.is_recoverable(),
            "an infeasible trial must let the outer search back off, not abort the fit"
        );
        assert!(
            objective_err
                .message()
                .starts_with("outer fixed-point evaluation: "),
            "context must prefix the message: {}",
            objective_err.message()
        );
        // The producer's error is still reachable, so nothing downstream
        // has to re-derive the classification from prose.
        let source = objective_err
            .downcast_ref::<EstimationError>()
            .expect("the typed source must survive the boundary");
        assert!(matches!(
            source,
            EstimationError::CustomFamily(CustomFamilyError::InnerSolveNotConverged {
                cycles: 12,
                ..
            })
        ));
    }

    /// The conservative direction: a genuinely structural failure stays
    /// fatal. Widening recoverability would let the search grind through a
    /// problem that can never work.
    #[test]
    fn a_structural_configuration_failure_stays_fatal() {
        let err = EstimationError::CustomFamily(CustomFamilyError::UnsupportedConfiguration {
            reason: "this family does not support the requested link".to_string(),
        });
        assert!(!err.is_trial_point_infeasible());
        assert!(into_objective_error("outer EFS eval", err).is_fatal());
    }

    /// The two failures render with overlapping text but classify
    /// oppositely — precisely what a substring test could not do, and why
    /// one existed to be deleted.
    #[test]
    fn classification_does_not_depend_on_the_rendered_message() {
        let infeasible = EstimationError::CustomFamily(CustomFamilyError::InnerSolveNotConverged {
            cycles: 1,
            terminal: None,
            kkt_residual: Some(1.0),
            kkt_tol: Some(1e-8),
            theta_dim: 1,
            rho_dim: 1,
            psi_dim: 0,
        });
        let structural =
            EstimationError::CustomFamily(CustomFamilyError::UnsupportedConfiguration {
                reason: infeasible.to_string(),
            });
        // Byte-identical tails, opposite verdicts.
        assert!(structural.to_string().contains(&infeasible.to_string()));
        assert!(into_objective_error("ctx", infeasible).is_recoverable());
        assert!(into_objective_error("ctx", structural).is_fatal());
    }
}
