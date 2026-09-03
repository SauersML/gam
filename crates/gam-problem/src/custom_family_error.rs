//! Custom-family error type and its String conversions.

use thiserror::Error;

use crate::{IdentifiabilityAudit, MapUniquenessError};


#[derive(Debug, Clone, Copy, PartialEq)]
pub enum JointNewtonTerminalReason {
    CycleBudget,
    FullyRejectedExactFixedPoint {
        consecutive_cycles: usize,
        joint_trust_radius: f64,
        rejection_counts: [usize; 4],
    },
    FullyRejectedAtTrustRegionFloor {
        consecutive_cycles: usize,
        joint_trust_radius: f64,
        rejection_counts: [usize; 4],
    },
    /// The residual was still contracting — every step accepted, the model
    /// trusted — but at a geometric rate too slow to reach tolerance within
    /// the projection cap. The solve was descending, not stuck: on the #2695
    /// 1569 pair this is the scale coefficient walking the σ→0 ray, an
    /// objective with no finite minimizer along that direction at that ρ, and
    /// the outer needs to read it as under-penalization rather than as a
    /// failed seed.
    SlowGeometricRate {
        rate_per_cycle: f64,
        window_cycles: usize,
        projected_cycles_to_tolerance: usize,
        residual: f64,
        residual_tol: f64,
    },
}

impl std::fmt::Display for JointNewtonTerminalReason {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::CycleBudget => write!(f, "cycle budget"),
            Self::FullyRejectedExactFixedPoint {
                consecutive_cycles,
                joint_trust_radius,
                rejection_counts,
            } => write!(
                f,
                "complete rejected-cycle state repeated {consecutive_cycles} times at \
                 trust radius {joint_trust_radius:.6e}; rejects \
                 [model,likelihood,objective,feasibility]={rejection_counts:?}"
            ),
            Self::FullyRejectedAtTrustRegionFloor {
                consecutive_cycles,
                joint_trust_radius,
                rejection_counts,
            } => write!(
                f,
                "all attempts rejected for {consecutive_cycles} cycles at the absolute \
                 trust-region floor {joint_trust_radius:.6e}; rejects \
                 [model,likelihood,objective,feasibility]={rejection_counts:?}"
            ),
            Self::SlowGeometricRate {
                rate_per_cycle,
                window_cycles,
                projected_cycles_to_tolerance,
                residual,
                residual_tol,
            } => {
                if *rate_per_cycle < 1.0 {
                    write!(
                        f,
                        "residual {residual:.6e} still contracting at {rate_per_cycle:.4}x per \
                         cycle over the last {window_cycles} cycles, projected more than \
                         {projected_cycles_to_tolerance} further cycles to reach \
                         {residual_tol:.6e}: the solve was descending along a direction with \
                         no finite minimizer in reach, not stuck"
                    )
                } else {
                    write!(
                        f,
                        "residual {residual:.6e} is not contracting ({rate_per_cycle:.4}x per \
                         cycle over the last {window_cycles} cycles, every step accepted) and \
                         cannot reach {residual_tol:.6e}: the solve was descending along a \
                         direction with no finite minimizer in reach, not stuck"
                    )
                }
            }
        }
    }
}

/// The blockwise inner loop's terminal decision variables — the quantities its
/// convergence verdict is actually taken on.
///
/// The loop certifies with
/// `max_accepted_step <= step_tol && objective_change <= objective_tol`, and then
/// `joint_stationarity_ok || max_proposed_step <= step_tol`. Reporting only the
/// cycle count cannot say which of those four conjuncts failed, and they have
/// different causes: steps still large means the solve needs more cycles, steps
/// tiny with `joint_stationarity_ok == false` means the exact joint gate is the
/// blocker rather than the budget, and an `objective_change` above tolerance
/// means the iterate is still moving. This is deliberately NOT a KKT residual:
/// `BlockwiseInnerResult::kkt_residual` is `None` off a converged iterate on
/// purpose, because no caller may trust an IFT correction there, so the honest
/// diagnostic is the decision variables themselves rather than a residual
/// recomputed at a non-KKT point.
/// The stationarity residual denominated the way its own gate denominates it.
///
/// The inner joint-Newton gate is `R ≤ inner_tol · (1 + scale)` with
/// `scale = max(‖∇L‖∞, ‖Sβ‖∞, ‖∇Φ‖∞)`, so dividing through by `(1 + scale)`
/// gives the single scalar the gate actually tests against one fixed number:
///
/// ```text
/// relative_stationarity(R, scale) = R / (1 + scale) ≤ inner_tol   ⟺   gate accepts
/// ```
///
/// Two properties make this — and not `R`, and not `R/residual_tol` — the
/// column to rank a population of refusals on (gam#2713):
///
/// * It is comparable across solves. `R` alone is not: a single suite spans a
///   `1.18e10` range of `scale`, so an absolute `R = 5.3` at `scale = 5.3e6` is
///   stationary to one part in a million while `R = 5.3` at `scale = 1` is not
///   stationary at all. `R/residual_tol` is not either: it divides by
///   `inner_tol` as well, and `inner_tol` takes two different values in this
///   code (the `1e-6` default and the `1e-11` derivative-lane floor), so rows
///   from the two lanes are on axes that differ by five orders of magnitude.
/// * It handles the scale-free end explicitly rather than by accident. The
///   `1 +` is not cosmetic: at `scale → 0` there is no relative scale to speak
///   of and the criterion must degrade to the ABSOLUTE `R ≤ inner_tol`, which
///   is exactly what this expression does. A bare `R/scale` would instead
///   divide by zero and rank a perfectly-converged small-scale solve at
///   infinity. For `scale ≫ 1` the two agree to within `1/scale`.
///
/// Deliberately NOT applied to `best_stationarity_residual`: that value was
/// computed at a different iterate, whose `scale` this state does not carry.
/// Rescaling it by the terminal `scale` would produce a number that is neither
/// the best relative stationarity nor anything else.
#[must_use]
pub fn relative_stationarity(stationarity_residual: f64, stationarity_scale: f64) -> f64 {
    stationarity_residual / (1.0 + stationarity_scale)
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub enum InnerConvergenceTerminalState {
    /// The blockwise Gauss-Seidel route's terminal cycle.
    Blockwise {
        cycle: usize,
        max_accepted_step: f64,
        max_proposed_step: f64,
        step_tol: f64,
        objective_change: f64,
        objective_tol: f64,
        joint_stationarity_ok: bool,
    },
    /// The exact joint-Newton route's terminal cycle. This route DOES have a
    /// genuine stationarity residual (the blockwise one does not, off a
    /// converged iterate), and it has a third outcome the other lacks:
    /// `resolvable_negative_curvature` marks a first-order stationary STRICT
    /// SADDLE, where the score and the Newton proposal both vanish but the exact
    /// penalized Hessian has resolvable negative curvature. That refuses
    /// convergence deliberately, and it is nothing like exhausting a budget.
    JointNewton {
        cycle: usize,
        stationarity_residual: f64,
        residual_tol: f64,
        /// The magnitude the stationarity residual is denominated against:
        /// `max(‖∇L‖∞, ‖Sβ‖∞, ‖∇Φ‖∞)` at the terminal iterate, i.e. the `scale`
        /// in `residual_tol = inner_tol · (1 + scale)`.
        ///
        /// Carried because WITHOUT it the message cannot be ranked (gam#2713).
        /// The natural thing to do with a printed `residual (tol=…)` pair is to
        /// form `R/T` and read it as "N× over tolerance"; that ratio is
        /// `≈ (R/scale)/inner_tol`, so it mixes two different tolerances (the
        /// `1e-6` default and the derivative lane's `1e-11`
        /// `JOINT_LAML_DERIV_INNER_TOL_FLOOR`) and it is ANTI-correlated with
        /// convergence across part of the range. Measured over 41 refusal pairs
        /// from one survival sweep: a row printing `R/T = 238×` was stationary
        /// to `R/scale = 2.4e-9` — converged to nine digits — while a row
        /// printing `R/T = 1.4e3×` sat at `R/scale = 1.4e-3`, a million times
        /// less converged. Ranking on `R/T` sends triage to the first row.
        ///
        /// The comparable column is [`relative_stationarity`], printed below,
        /// which is the gate's own quantity: the gate accepts exactly when it
        /// is `≤ inner_tol`, so it is `0` at the optimum, `~1` where the
        /// residual has collapsed onto one of its own terms, and directly
        /// comparable across both `inner_tol` regimes.
        stationarity_scale: f64,
        step_inf: f64,
        step_tol: f64,
        resolvable_negative_curvature: bool,
        /// The smallest stationarity residual this solve actually computed, and
        /// how many cycles have passed since it last improved.
        ///
        /// The terminal residual alone cannot separate a solve that never got
        /// close from one that reached a near-tolerance point and then walked
        /// away from it, and those are different defects with different fixes.
        /// Measured on the transformation-normal wine arm (#2600): the terminal
        /// residual is `1.906e0` while the smallest this same solve computed is
        /// `1.578e-3` — 1200x better, within 1.9x of `residual_tol`, and reached
        /// 27 cycles earlier, after which every accepted step raised the
        /// residual again. Read from the terminal value alone that solve looks
        /// like it never approached stationarity; read with the best value it
        /// is a solve that drifted off a point it had essentially reached.
        best_stationarity_residual: f64,
        cycles_since_best_residual: usize,
        termination_reason: JointNewtonTerminalReason,
    },
}

impl std::fmt::Display for InnerConvergenceTerminalState {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Blockwise {
                cycle,
                max_accepted_step,
                max_proposed_step,
                step_tol,
                objective_change,
                objective_tol,
                joint_stationarity_ok,
            } => write!(
                f,
                "blockwise terminal cycle {cycle}: max_accepted_step={max_accepted_step:.6e} \
                 (tol={step_tol:.6e}), max_proposed_step={max_proposed_step:.6e}, \
                 objective_change={objective_change:.6e} (tol={objective_tol:.6e}), \
                 joint_stationarity_ok={joint_stationarity_ok}"
            ),
            Self::JointNewton {
                cycle,
                stationarity_residual,
                residual_tol,
                stationarity_scale,
                step_inf,
                step_tol,
                resolvable_negative_curvature,
                best_stationarity_residual,
                cycles_since_best_residual,
                termination_reason,
            } => write!(
                f,
                "joint-Newton terminal cycle {cycle}: \
                 stationarity_residual={stationarity_residual:.6e} (tol={residual_tol:.6e}), \
                 relative_stationarity={:.6e} \
                 (= residual/(1+scale), scale={stationarity_scale:.6e}; \
                 THIS is the comparable column, not residual/tol), \
                 step_inf={step_inf:.6e} (tol={step_tol:.6e}), \
                 resolvable_negative_curvature={resolvable_negative_curvature}, \
                 best_stationarity_residual={best_stationarity_residual:.6e} \
                 (last improved {cycles_since_best_residual} cycle(s) before this one), \
                 termination={termination_reason}",
                relative_stationarity(*stationarity_residual, *stationarity_scale),
            ),
        }
    }
}

/// Render the projected-KKT comparison in the inner-refusal message.
///
/// The pair used to be printed as `|r|_inf={:?} against tol={:?}`, which on the
/// common path renders `|r|_inf=None against tol=None` — **two absences laid out
/// as a comparison**. That reads as a measurement that was taken and came out
/// unfavourable, and it is the opposite: nothing was measured. It cost real time
/// on gam#2600, where the phrase sat in every refusal while the actual decision
/// variables (which the `[{terminal}]` block does carry) said something quite
/// different. A missing value has to say it is missing, and which side is
/// missing, because "the solver emitted no KKT diagnostic on this path" and "the
/// residual is 4e5x its tolerance" call for different next steps.
fn render_projected_kkt_comparison(residual: Option<f64>, tol: Option<f64>) -> String {
    match (residual, tol) {
        (Some(residual), Some(tol)) => format!(
            "projected KKT residual |r|_inf={residual:.6e} against tol={tol:.6e}"
        ),
        (Some(residual), None) => format!(
            "projected KKT residual |r|_inf={residual:.6e}; \
             no stationarity tolerance was recorded to compare it against"
        ),
        (None, Some(tol)) => format!(
            "no projected KKT residual was recorded; the stationarity tolerance \
             on this path was {tol:.6e}"
        ),
        (None, None) => "this solver path emits no typed projected-KKT diagnostic, so \
                         neither a residual nor a tolerance was recorded — read the \
                         terminal decision variables above instead"
            .to_string(),
    }
}

#[derive(Debug, Clone, Error)]
pub enum CustomFamilyError {
    #[error("custom-family invalid input in {context}: {reason}")]
    InvalidInput {
        context: &'static str,
        reason: String,
    },
    #[error("custom-family optimization error in {context}: {reason}")]
    Optimization {
        context: &'static str,
        reason: String,
    },
    #[error("{reason}")]
    DimensionMismatch { reason: String },
    #[error("{reason}")]
    NumericalFailure { reason: String },
    #[error("{reason}")]
    ConstraintViolation { reason: String },
    #[error("{reason}")]
    UnsupportedConfiguration { reason: String },
    /// The inner solve did not reach its KKT condition at THIS trial
    /// point, so the analytic outer gradient/Hessian cannot be exposed
    /// (they require `F_beta(beta, theta) = 0`).
    ///
    /// This is a statement about one `theta`, not about the problem: the
    /// outer search should treat the trial as infeasible, back off, and
    /// continue. It previously travelled as
    /// [`UnsupportedConfiguration`](Self::UnsupportedConfiguration) — a
    /// variant that *means* the configuration is structurally
    /// unsupported, i.e. fatal — with the real distinction encoded only
    /// in the message text. Downstream then had to recover it by
    /// substring-matching that text, and two call sites reached opposite
    /// verdicts on the same error (#2553). Choosing the variant that says
    /// what happened removes the need to guess.
    #[error(
        "custom-family inner solve did not converge after {cycles} cycle(s) [{}] \
         ({}); \
         refusing to expose profile objective derivatives for theta_dim={theta_dim} \
         (rho_dim={rho_dim}, psi_dim={psi_dim}). The analytic outer gradient/Hessian \
         require the inner KKT equation F_beta(beta, theta)=0; returning a value with \
         zero or shape-only derivatives is mathematically inconsistent. This trial \
         point is infeasible; the outer search may step away from it.",
        match terminal {
            Some(state) => state.to_string(),
            None => "no terminal convergence state was recorded".to_string(),
        },
        render_projected_kkt_comparison(*kkt_residual, *kkt_tol)
    )]
    InnerSolveNotConverged {
        cycles: usize,
        /// The decision variables the inner loop's verdict was taken on. See
        /// [`InnerConvergenceTerminalState`] — a cycle count alone cannot say
        /// which conjunct of the convergence test failed.
        terminal: Option<InnerConvergenceTerminalState>,
        /// Sup-norm of the projected KKT residual at the terminal inner iterate,
        /// i.e. the quantity this refusal was decided against. A cycle count
        /// alone cannot distinguish a solve that ran out of budget one order
        /// from its tolerance — where the budget is the thing to look at — from
        /// one sitting many orders away, which is a stalled or diverging solve
        /// and a different defect entirely. `None` when the producing solver
        /// path emits no typed KKT diagnostic (blockwise NR fallback,
        /// eager-stop), which is itself worth seeing in the refusal.
        kkt_residual: Option<f64>,
        /// The stationarity tolerance `kkt_residual` was compared against.
        kkt_tol: Option<f64>,
        theta_dim: usize,
        rho_dim: usize,
        psi_dim: usize,
    },
    #[error("{reason}")]
    BasisDecompositionFailed { reason: String },
    /// Pre-fit cross-block identifiability audit refused the fit. The
    /// joint design across `ParameterBlockSpec`s carries a rank
    /// deficiency that the post-`joint_null_rotation` absorption did
    /// not resolve: two or more blocks contribute the same direction,
    /// or a structural >2-way alias was detected without per-pair
    /// attribution. The full `IdentifiabilityAudit` is held so
    /// consumers (logs, structured-error sinks, the seed driver's
    /// classifier) can extract the alias pairs and the summary string
    /// without reparsing.
    #[error("identifiability audit refused the fit: {}", audit.summary)]
    IdentifiabilityFailure { audit: IdentifiabilityAudit },
    /// MAP estimate uniqueness condition `ker(J^T W J) ∩ ker(S) = {0}` is
    /// violated.  A null direction of `J^T W J` carries zero penalty
    /// curvature, so the posterior is flat along that direction and the
    /// MAP is non-unique.  The structured [`MapUniquenessError`] names the
    /// dominant block so the caller can add the missing penalty or remove
    /// the unpenalised direction.
    #[error("MAP estimate non-unique: {}", error)]
    MapUniquenessFailure { error: MapUniquenessError },
    /// A numerical verdict the inner solve reached AT ONE TRIAL POINT: no
    /// Laplace mode here, this active face's curvature refuses certification
    /// here, this quadratic subproblem is degenerate here.
    ///
    /// Like [`Self::InnerSolveNotConverged`] this is a statement about one
    /// `theta`, not about the problem — an indefinite coefficient point at one
    /// rho is an ordinary Laplace mode at another — so the outer search should
    /// reject the trial and step away, which is what the inner solver's own
    /// logs say should happen. It is a separate variant because
    /// `InnerSolveNotConverged` carries a fixed cycles/theta_dim/rho_dim/psi_dim
    /// shape and a message specifically about refusing to expose profile
    /// derivatives; reusing it for a curvature refusal would state something
    /// untrue.
    #[error("inner solve refused this trial point: {reason}")]
    TrialPointRefused { reason: String },
}

impl CustomFamilyError {
    /// A numerical refusal raised while evaluating at one trial point.
    ///
    /// The named constructor exists so a boundary that *knows* it is reporting
    /// a rho-local failure can say so, rather than leaning on the blanket
    /// `From<String>` below and hoping its default is right.
    pub fn trial_point(reason: impl Into<String>) -> Self {
        Self::TrialPointRefused {
            reason: reason.into(),
        }
    }

    /// Grade an already-typed error rho-local WITHOUT re-wrapping one that
    /// already says so.
    ///
    /// A boundary whose whole contract is "evaluate at this rho" answers the
    /// trial-point question for everything that crosses it (see the
    /// [`From<String>`] rationale below and gam#2590). Doing that with
    /// [`Self::trial_point`] on a value that is *already* a
    /// [`Self::TrialPointRefused`] renders the inner error to text and prefixes
    /// it a second time, which is how
    ///
    /// ```text
    /// inner solve refused this trial point: inner solve refused this trial
    ///   point: synthetic outer objective failure: block[0] evaluate()
    /// ```
    ///
    /// reached a user (gam#2667). The doubled prefix was cosmetic; the loss it
    /// made visible is not, because rendering to `String` discards the variant
    /// and only [`From<String>`]'s default put a classification back.
    ///
    /// So: keep the error untouched when it already answers the question
    /// (`is_trial_point_infeasible()`), and only render one that does not --
    /// which is the single case where the classification is genuinely being
    /// *changed* rather than restated.
    #[must_use]
    pub fn into_trial_point(self) -> Self {
        if self.is_trial_point_infeasible() {
            self
        } else {
            Self::TrialPointRefused {
                reason: self.to_string(),
            }
        }
    }
}

impl From<String> for CustomFamilyError {
    /// # Why this lands on `TrialPointRefused` and not `InvalidInput`
    ///
    /// A `String` cannot carry the one bit the outer smoothing search needs —
    /// is this failure a property of the trial point, or of the problem? — so
    /// any conversion from it must answer by default. This one used to answer
    /// `InvalidInput`, the variant [`Self::is_trial_point_infeasible`] returns
    /// `false` for, and gam-custom-family's inner solver reports *every*
    /// refusal as `Err(String)`. So "there is no Laplace mode at this rho", a
    /// verdict about one rho, was graded fatal and killed the whole fit at the
    /// first probe, at an optimizer whose seed loop has the correct branch one
    /// line above the one it took (gam#2590).
    ///
    /// The default is not a coin flip, because the two mistakes are not
    /// comparable:
    ///
    /// * A structural failure graded rho-local recurs at every probed rho. The
    ///   seed loop exhausts, the run still fails, and it fails quoting this
    ///   same reason — after a bounded number of cheap, identical inner
    ///   failures.
    /// * A rho-local refusal graded structural aborts a fit that was
    ///   perfectly fittable one rho away. Measured twice: #2553, #2590.
    ///
    /// So where the type system forces a guess, the guess must be
    /// "trial point". Where a caller knows better in either direction, it
    /// should construct the variant it means — [`Self::trial_point`] or the
    /// structural variant — instead of routing through here.
    fn from(value: String) -> Self {
        Self::TrialPointRefused { reason: value }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn regrading_a_trial_point_refusal_does_not_prefix_it_twice_2667() {
        let inner = CustomFamilyError::trial_point(
            "synthetic outer objective failure: block[0] evaluate()",
        );
        // The historical route: render to `String` at an internal boundary,
        // then let the boundary answer the trial-point question again.
        let round_tripped = CustomFamilyError::trial_point(inner.to_string());
        assert_eq!(
            round_tripped
                .to_string()
                .matches("inner solve refused this trial point:")
                .count(),
            2,
            "fixture must reproduce the doubling this test is about"
        );

        // The typed route says the same thing once.
        let regraded = inner.clone().into_trial_point();
        assert_eq!(
            regraded
                .to_string()
                .matches("inner solve refused this trial point:")
                .count(),
            1,
            "an error that already answers the question must not be re-wrapped: {regraded}"
        );
        assert_eq!(regraded.to_string(), inner.to_string());
        assert!(regraded.is_trial_point_infeasible());

        // An error that does NOT answer the question is genuinely reclassified,
        // and keeps its own text as the reason.
        let structural = CustomFamilyError::DimensionMismatch {
            reason: "log-lambda length mismatch: got 3, expected 4".to_string(),
        };
        let structural_text = structural.to_string();
        let regraded = structural.into_trial_point();
        assert!(regraded.is_trial_point_infeasible());
        assert!(
            regraded.to_string().contains(&structural_text),
            "reclassification must not drop the original text: {regraded}"
        );
    }

    #[test]
    fn two_absences_are_not_reported_as_a_comparison_2600() {
        // `|r|_inf=None against tol=None` reads as a measurement that came out
        // badly. Nothing was measured, and the message has to say which side is
        // missing: "no diagnostic on this path" and "the residual is 4e5x tol"
        // call for different next steps.
        let absent = CustomFamilyError::InnerSolveNotConverged {
            cycles: 53,
            terminal: None,
            kkt_residual: None,
            kkt_tol: None,
            theta_dim: 3,
            rho_dim: 3,
            psi_dim: 0,
        };
        let msg = absent.to_string();
        assert!(
            !msg.contains("None against"),
            "two absences must not be laid out as a comparison: {msg}"
        );
        assert!(
            msg.contains("emits no typed projected-KKT diagnostic"),
            "the message must name the absence as an absence: {msg}"
        );

        // With both present it still reads as the comparison it is.
        let measured = CustomFamilyError::InnerSolveNotConverged {
            cycles: 53,
            terminal: None,
            kkt_residual: Some(1.906428e0),
            kkt_tol: Some(8.307952e-4),
            theta_dim: 3,
            rho_dim: 3,
            psi_dim: 0,
        };
        let msg = measured.to_string();
        assert!(
            msg.contains("|r|_inf=1.906428e0 against tol=8.307952e-4"),
            "a real comparison must still render as one: {msg}"
        );

        // A half-present pair names WHICH half is missing rather than printing
        // `Some(..)`/`None` and leaving the reader to work it out.
        let half = CustomFamilyError::InnerSolveNotConverged {
            cycles: 7,
            terminal: None,
            kkt_residual: Some(4.069e3),
            kkt_tol: None,
            theta_dim: 1,
            rho_dim: 1,
            psi_dim: 0,
        };
        let msg = half.to_string();
        assert!(
            msg.contains("no stationarity tolerance was recorded"),
            "a half-present pair must name the missing half: {msg}"
        );
    }

    #[test]
    fn joint_newton_terminal_state_reports_the_best_residual_not_only_the_last_2600() {
        // The #2600 shape: a solve that reached 1.578e-3 (within 1.9x of tol)
        // and then drifted for 27 cycles to a terminal 1.906e0. A reader given
        // only the terminal value concludes the solve never approached
        // stationarity; the correct reading is that it did and left. Both
        // numbers and the distance back to the best one must be in the message.
        let state = InnerConvergenceTerminalState::JointNewton {
            cycle: 52,
            stationarity_residual: 1.906428e0,
            residual_tol: 8.307952e-4,
            // Consistent with the pair above: `tol = 1e-6 · (1 + scale)`.
            stationarity_scale: 829.7952,
            step_inf: 4.958893e0,
            step_tol: 8.493315e-5,
            resolvable_negative_curvature: true,
            best_stationarity_residual: 1.578e-3,
            cycles_since_best_residual: 27,
            termination_reason: JointNewtonTerminalReason::CycleBudget,
        };
        let msg = state.to_string();
        assert!(
            msg.contains("stationarity_residual=1.906428e0"),
            "message: {msg}"
        );
        assert!(
            msg.contains("best_stationarity_residual=1.578000e-3"),
            "message: {msg}"
        );
        assert!(
            msg.contains("27 cycle(s) before this one"),
            "message: {msg}"
        );
    }

    #[test]
    fn joint_newton_terminal_state_carries_the_column_that_ranks_correctly_2713() {
        // gam#2713: two refusals from one survival sweep, one per `inner_tol`
        // lane. Read as "N x over tolerance" — the only ratio the message used
        // to permit — they are ordered BACKWARDS relative to how converged they
        // are, so triage goes to the wrong row. The message must therefore
        // carry the denominator that fixes the ordering.
        //
        // A: derivative lane, `inner_tol = 1e-11`, scale = 43.807. Stationary
        //    to nine digits — converged — and it prints `R/T = 238x`.
        let converged = InnerConvergenceTerminalState::JointNewton {
            cycle: 12,
            stationarity_residual: 1.065281e-7,
            residual_tol: 1e-11 * (1.0 + 43.807),
            stationarity_scale: 43.807,
            step_inf: 1.0e-9,
            step_tol: 1.0e-10,
            resolvable_negative_curvature: false,
            best_stationarity_residual: 1.065281e-7,
            cycles_since_best_residual: 0,
            termination_reason: JointNewtonTerminalReason::CycleBudget,
        };
        // B: default lane, `inner_tol = 1e-6`, scale = 3.3392. A MILLION times
        //    less converged than A, and it prints the larger `R/T`.
        let far = InnerConvergenceTerminalState::JointNewton {
            cycle: 12,
            stationarity_residual: 1.4e-3 * (1.0 + 3.3392),
            residual_tol: 1e-6 * (1.0 + 3.3392),
            stationarity_scale: 3.3392,
            step_inf: 1.0e-3,
            step_tol: 1.0e-6,
            resolvable_negative_curvature: false,
            best_stationarity_residual: 1.4e-3 * (1.0 + 3.3392),
            cycles_since_best_residual: 0,
            termination_reason: JointNewtonTerminalReason::CycleBudget,
        };

        let (r_a, t_a, s_a) = (1.065281e-7, 1e-11 * (1.0 + 43.807), 43.807);
        let (r_b, t_b, s_b) = (1.4e-3 * (1.0 + 3.3392), 1e-6 * (1.0 + 3.3392), 3.3392);

        // The ratio a reader forms from the printed pair ranks A ABOVE B.
        assert!(
            r_a / t_a > 200.0 && r_b / t_b > 1000.0,
            "the two rows must reproduce the measured N x over tolerance values"
        );
        assert!(
            r_a / t_a < r_b / t_b,
            "sanity: both rows are 'over tolerance', and by that ratio they are \
             only ~6x apart"
        );

        // The comparable column ranks them the other way round, by six orders
        // of magnitude: A is converged, B is not.
        let rel_a = relative_stationarity(r_a, s_a);
        let rel_b = relative_stationarity(r_b, s_b);
        assert!(
            rel_a < 1e-8 && rel_b > 1e-4,
            "relative stationarity: A={rel_a:.3e} must be the converged row, \
             B={rel_b:.3e} the unconverged one"
        );
        assert!(
            rel_b / rel_a > 1e5,
            "the two rows differ by five-plus orders in relative stationarity \
             ({rel_a:.3e} vs {rel_b:.3e}) while their printed R/T differ by ~6x"
        );

        // ...and it is IN the message, for both, so no reader has to recover a
        // scale by inverting the tolerance formula.
        for (state, expected) in [(converged, rel_a), (far, rel_b)] {
            let msg = state.to_string();
            assert!(
                msg.contains(&format!("relative_stationarity={expected:.6e}")),
                "message must print the comparable column: {msg}"
            );
            assert!(
                msg.contains("scale="),
                "message must print the denominator it used: {msg}"
            );
        }
    }

    /// The scale-free end of [`relative_stationarity`]: with no scale to be
    /// relative to, the criterion is the absolute residual against `inner_tol`,
    /// NOT a division by zero.
    #[test]
    fn relative_stationarity_degrades_to_the_absolute_residual_at_zero_scale_2713() {
        let absolute = relative_stationarity(3.7e-9, 0.0);
        assert!(
            absolute.to_bits() == 3.7e-9_f64.to_bits(),
            "with no scale the criterion is the absolute residual, got {absolute:.6e}"
        );
        // And it agrees with the bare `R/scale` to within `1/scale` once there
        // IS a scale, so nothing is lost at the end where the relative reading
        // is the meaningful one.
        let (residual, scale) = (5.275447e0, 5.2754e6);
        let mixed = relative_stationarity(residual, scale);
        let bare = residual / scale;
        assert!(
            ((mixed - bare) / bare).abs() < 1e-5,
            "mixed={mixed:.6e} bare={bare:.6e}"
        );
    }

    #[test]
    fn invalid_input_display_contains_context_and_reason() {
        let err = CustomFamilyError::InvalidInput {
            context: "my_context",
            reason: "something broke".to_string(),
        };
        let msg = err.to_string();
        assert!(msg.contains("my_context"), "message: {msg}");
        assert!(msg.contains("something broke"), "message: {msg}");
    }

    #[test]
    fn optimization_display_contains_context_and_reason() {
        let err = CustomFamilyError::Optimization {
            context: "outer_loop",
            reason: "diverged".to_string(),
        };
        let msg = err.to_string();
        assert!(
            msg.contains("outer_loop") && msg.contains("diverged"),
            "message: {msg}"
        );
    }

    #[test]
    fn dimension_mismatch_displays_reason() {
        let err = CustomFamilyError::DimensionMismatch {
            reason: "3 vs 4".to_string(),
        };
        assert_eq!(err.to_string(), "3 vs 4");
    }

    #[test]
    fn numerical_failure_displays_reason() {
        let err = CustomFamilyError::NumericalFailure {
            reason: "NaN detected".to_string(),
        };
        assert_eq!(err.to_string(), "NaN detected");
    }

    #[test]
    fn a_string_boundary_refusal_is_recoverable_not_invalid_input() {
        // The regression this exists for (gam#2590): the refusal used to
        // arrive as `InvalidInput`, which classifies fatal, so an outer
        // optimizer explicitly built to step away from an infeasible trial
        // point aborted the whole fit at the first one it met.
        let err = CustomFamilyError::from("no Laplace mode at this rho".to_string());
        assert!(matches!(err, CustomFamilyError::TrialPointRefused { .. }));
        assert!(err.is_trial_point_infeasible());
        assert!(err.to_string().contains("no Laplace mode at this rho"));
        assert_eq!(
            CustomFamilyError::trial_point("x").to_string(),
            CustomFamilyError::from("x".to_string()).to_string(),
            "the named constructor and the blanket conversion must agree"
        );
        assert!(
            !CustomFamilyError::InvalidInput {
                context: "c",
                reason: "r".to_string(),
            }
            .is_trial_point_infeasible(),
            "`InvalidInput` must keep meaning what it says"
        );
    }

    /// #2689 deleted `impl From<CustomFamilyError> for String` so that a
    /// flattening is a compile error rather than a silent default. This test
    /// used to assert that impl and so could not compile once it was gone.
    ///
    /// The behaviour worth keeping is what the impl *delegated to*: rendering
    /// goes through `Display`, and an explicit `.to_string()` at a boundary
    /// that genuinely owns a `String` contract must still produce the reason
    /// verbatim. Asserting `Display` keeps that guarantee while leaving the
    /// flattening un-resurrectable.
    #[test]
    fn rendering_a_custom_family_error_uses_display() {
        let err = CustomFamilyError::NumericalFailure {
            reason: "singular".to_string(),
        };
        assert_eq!(err.to_string(), "singular");
    }
}

impl CustomFamilyError {
    /// Whether a failure of this kind invalidates the whole outer run or
    /// only the trial point it was produced at.
    ///
    /// The producer's judgement, made once against the variant. It
    /// replaces a downstream substring match on the rendered message that
    /// classified one variant two different ways depending on which call
    /// site it crossed (#2553).
    ///
    /// The match is deliberately exhaustive with no wildcard arm: a new
    /// variant must be classified when it is added, rather than
    /// defaulting to whichever answer happens to be listed last.
    #[must_use]
    pub fn is_trial_point_infeasible(&self) -> bool {
        match self {
            // The inner solve missed its KKT condition at THIS theta. The
            // outer search can step away; the problem is fine.
            Self::InnerSolveNotConverged { .. } => true,
            // Likewise rho-local: a numerical refusal evaluated at one trial
            // point, which becomes true or false by moving theta (gam#2590).
            Self::TrialPointRefused { .. } => true,
            // Everything else is a property of the configuration, the
            // data, or the numerics, and does not become true or false by
            // moving theta.
            Self::InvalidInput { .. }
            | Self::Optimization { .. }
            | Self::DimensionMismatch { .. }
            | Self::NumericalFailure { .. }
            | Self::ConstraintViolation { .. }
            | Self::UnsupportedConfiguration { .. }
            | Self::BasisDecompositionFailed { .. }
            | Self::IdentifiabilityFailure { .. }
            | Self::MapUniquenessFailure { .. } => false,
        }
    }
}
