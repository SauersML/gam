//! Custom-family error type and its String conversions.

use thiserror::Error;

use crate::{IdentifiabilityAudit, MapUniquenessError};


#[derive(Debug, Clone, PartialEq)]
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
    /// The residual was still contracting (`rate_per_cycle < 1` over the
    /// window), but at a geometric rate too slow to reach tolerance within the
    /// projection cap. A window whose residual did not contract is not this
    /// reason: it is [`Self::StalledOnDescendingRay`] when the accepted step
    /// descends a ray and [`Self::ResidualNotContracting`] otherwise (#2902).
    SlowGeometricRate {
        rate_per_cycle: f64,
        window_cycles: usize,
        projected_cycles_to_tolerance: usize,
        residual: f64,
        residual_tol: f64,
        /// The block whose penalty is too weak to close the ray, and the
        /// strength at which it would: `None` when no block's penalty opposes
        /// the accepted step (an unpenalized ray, which no ρ can close).
        ray: Option<RayRestoration>,
    },
    /// Over the projection window the residual did not contract
    /// (`rate_per_cycle >= 1`, or no positive starting residual), and the
    /// accepted step descended no ray a penalty strength could close. The
    /// solve is not converging slowly. It is not converging (#2902).
    ResidualNotContracting {
        rate_per_cycle: f64,
        window_cycles: usize,
        residual: f64,
        residual_tol: f64,
    },
    /// The solve left on a residual-stall or divergence guard while its last
    /// accepted step was still descending a direction no block's penalty
    /// closes (gam#2695).
    ///
    /// Read exactly like [`Self::SlowGeometricRate`]'s `ray`: the seed is
    /// under-penalized, not failed, and the outer restores it rather than
    /// discarding it. It is a SEPARATE reason because these exits certify
    /// nothing about a rate — the residual was flat or growing, not
    /// contracting slowly — and reporting them as a slow rate would claim a
    /// contraction that was not measured. On the #2695 1569 pair seeds 0-3
    /// leave here (`early-exit non-converged (divergence/stall guard)`) with
    /// every step accepted at a good model ratio, the objective still falling
    /// and `‖β‖∞` still growing, which is the ray this names.
    StalledOnDescendingRay {
        residual: f64,
        residual_tol: f64,
        cycles: usize,
        ray: RayRestoration,
    },
    /// The constrained fixed-point certificate declined the iterate the loop
    /// left on, and `condition` is the acceptance condition that failed, with its
    /// value and bound. Before this, the exit read `cycle budget` at any cycle
    /// count, so which condition declined had to be inferred (gam#979).
    ConstrainedFixedPointDeclined {
        condition: ConstrainedFixedPointCondition,
    },
    /// The joint Hessian source carried a non-finite entry at `cycle`, after
    /// the solve had moved β (gam#1088), so the penalized Hessian and its
    /// spectrum are degenerate and no certificate exists at this iterate.
    NonFiniteCurvature { cycle: usize },
    /// The inner state went non-finite (the gam#554 divergence guard). The
    /// three values are carried as they stood, so the message names which of
    /// them diverged.
    NonFiniteInnerState {
        residual: f64,
        objective: f64,
        log_likelihood: f64,
    },
    /// The certificate-candidate conditions routed the iterate to the KKT
    /// refusal and no constrained fixed-point condition declined it, so the
    /// refusal report's own classification is the reason (the #2695 1569 seed
    /// that read `cycle budget` at cycle 48 of 200 left here as
    /// `rank_deficient_H_pen`).
    KktCertificateRefused {
        diagnosis: crate::diagnostics::KktRefusalDiagnosis,
    },
    /// The residual stopped improving while the accepted steps were clipped by
    /// the trust region, and the accepted step descended no ray a penalty
    /// strength could close.
    ResidualStall {
        residual: f64,
        residual_tol: f64,
        best_residual: f64,
        cycles_without_improvement: usize,
        accepted_step_inf: f64,
        trust_radius: f64,
    },
    /// The residual stayed flat with every accepted step strictly inside the
    /// trust region and no acceptance certificate satisfied, and the accepted
    /// step descended no ray a penalty strength could close.
    FlatResidualStall {
        residual: f64,
        residual_tol: f64,
        best_residual: f64,
        cycles_without_improvement: usize,
    },
}

/// One acceptance condition of the custom-family joint Newton's constrained
/// fixed-point certificate, as it failed.
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum ConstrainedFixedPointCondition {
    /// The objective is still changing above its machine-eps floor.
    ObjectiveAboveFloor {
        objective_change: f64,
        objective_floor: f64,
    },
    /// The scalar Newton model's relative error exceeds its bound.
    ModelInexact { scalar_model_relerr: f64, bound: f64 },
    /// The accepted step is not finite or exceeds the stationarity tolerance.
    StepAboveTolerance {
        accepted_step_inf: f64,
        step_tol: f64,
    },
    /// `H_pen` has a numerical null space at the eigensolver resolution.
    HpenNullity { nullity: usize },
    /// `H_pen` could not be materialized or decomposed.
    HpenNullityUnavailable,
}

/// The block-level reading of a ray the joint Newton was descending when it
/// stopped: along the last accepted step `δ`, the likelihood term slopes
/// down by `likelihood_slope = ∇(−ℓ)·δ < 0` while block `block`'s penalty
/// slopes up by only `penalty_slope = (λ_b S_b β)·δ > 0`. The penalized
/// objective is stationary along `δ` at the strength ratio
/// `r = −likelihood_slope / penalty_slope > 1`, so raising every log
/// strength of that block by `log_strength_ratio = ln r` closes the ray at
/// the iterate the solve stopped on. The ratio is read off the two slopes the
/// solve already had; nothing here is a step size.
#[derive(Clone, Debug, PartialEq)]
pub struct RayRestoration {
    /// Index of the parameter block carrying the ray.
    pub block: usize,
    /// The outer ρ coordinates of that block's penalties: `rho_count` of
    /// them, contiguous from `rho_first` (ρ is laid out block by block).
    pub rho_first: usize,
    pub rho_count: usize,
    /// `ln r`, the amount every one of `rho_indices` has to rise.
    pub log_strength_ratio: f64,
    /// `∇(−ℓ)·δ` along the accepted step (negative: the likelihood descends).
    pub likelihood_slope: f64,
    /// `(λ_b S_b β)·δ` along the accepted step (positive: the penalty resists).
    pub penalty_slope: f64,
    /// `‖δ_b‖∞`, the block's share of the accepted step.
    pub block_step_inf: f64,
    /// The accepted step `δ = β_new − β_old` over the joint coefficients: the
    /// direction the solve was still descending. It is in the inner solve's
    /// reduced coordinates where the refusal is raised, and in raw joint order
    /// once the refusal leaves the fit, which lifts it through the
    /// identifiability gauge (#979).
    pub direction: std::sync::Arc<[f64]>,
}

impl RayRestoration {
    /// The outer ρ coordinates this restoration raises.
    pub fn rho_indices(&self) -> std::ops::Range<usize> {
        self.rho_first..self.rho_first + self.rho_count
    }
}

impl std::fmt::Display for RayRestoration {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "block {} is under-penalized along the accepted step (likelihood slope \
             {:.3e}, penalty slope {:.3e}, block step {:.3e}): the ray closes at \
             {:.4}x its penalty strength, i.e. rho[{}..{}] += {:.4}",
            self.block,
            self.likelihood_slope,
            self.penalty_slope,
            self.block_step_inf,
            self.log_strength_ratio.exp(),
            self.rho_first,
            self.rho_first + self.rho_count,
            self.log_strength_ratio,
        )
    }
}

/// How an inner joint Newton that stopped while descending a direction with no
/// finite minimizer in reach reports that direction, read off its typed terminal
/// reason (see [`CustomFamilyError::descending_ray_exit`]).
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum DescendingRayExit<'a> {
    /// A block's penalty opposes the ray, and raising that block's log
    /// strengths by [`RayRestoration::log_strength_ratio`] closes it.
    Closable(&'a RayRestoration),
    /// No block's penalty opposes the accepted step, so no `rho` closes the ray.
    Unpenalized,
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
                ray,
            } => {
                write!(
                    f,
                    "residual {residual:.6e} still contracting at {rate_per_cycle:.4}x per \
                     cycle over the last {window_cycles} cycles, projected more than \
                     {projected_cycles_to_tolerance} further cycles to reach \
                     {residual_tol:.6e}: the solve was descending along a direction with \
                     no finite minimizer in reach, not stuck"
                )?;
                match ray {
                    Some(ray) => write!(f, "; {ray}"),
                    None => write!(
                        f,
                        "; no block's penalty opposes the accepted step, so no penalty \
                         strength closes this ray"
                    ),
                }
            }
            Self::ResidualNotContracting {
                rate_per_cycle,
                window_cycles,
                residual,
                residual_tol,
            } => write!(
                f,
                "residual {residual:.6e} did not contract over the last {window_cycles} cycles \
                 ({rate_per_cycle:.4}x per cycle) and cannot reach {residual_tol:.6e}, and the \
                 accepted step descends no ray a penalty strength could close: the solve is not \
                 converging"
            ),
            Self::StalledOnDescendingRay {
                residual,
                residual_tol,
                cycles,
                ray,
            } => write!(
                f,
                "residual {residual:.6e} stalled or grew against {residual_tol:.6e} over \
                 {cycles} cycles while the accepted steps kept descending a direction with no \
                 finite minimizer in reach, so the seed is under-penalized rather than \
                 failed; {ray}"
            ),
            Self::ConstrainedFixedPointDeclined { condition } => write!(
                f,
                "the constrained fixed-point certificate declined: {condition}"
            ),
            Self::NonFiniteCurvature { cycle } => write!(
                f,
                "non-finite curvature at cycle {cycle}: the joint Hessian source carries a \
                 non-finite entry, so no certificate exists at this iterate"
            ),
            Self::NonFiniteInnerState {
                residual,
                objective,
                log_likelihood,
            } => {
                let diverged: Vec<String> = [
                    ("stationarity residual", *residual),
                    ("objective", *objective),
                    ("log-likelihood", *log_likelihood),
                ]
                .into_iter()
                .filter(|(_, value)| !value.is_finite())
                .map(|(name, value)| format!("{name}={value}"))
                .collect();
                write!(f, "non-finite inner state: {}", diverged.join(", "))
            }
            Self::KktCertificateRefused { diagnosis } => write!(
                f,
                "the KKT certificate refused the iterate: {}",
                diagnosis.as_str()
            ),
            Self::ResidualStall {
                residual,
                residual_tol,
                best_residual,
                cycles_without_improvement,
                accepted_step_inf,
                trust_radius,
            } => write!(
                f,
                "residual {residual:.6e} (tol {residual_tol:.6e}) stopped improving for \
                 {cycles_without_improvement} cycles with the accepted steps clipped by the \
                 trust region (accepted_step_inf={accepted_step_inf:.3e}, \
                 trust_radius={trust_radius:.3e}); best residual {best_residual:.6e}"
            ),
            Self::FlatResidualStall {
                residual,
                residual_tol,
                best_residual,
                cycles_without_improvement,
            } => write!(
                f,
                "residual {residual:.6e} (tol {residual_tol:.6e}) stayed flat for \
                 {cycles_without_improvement} cycles with every step inside the trust region \
                 and no acceptance certificate satisfied; best residual {best_residual:.6e}"
            ),
        }
    }
}

impl std::fmt::Display for ConstrainedFixedPointCondition {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::ObjectiveAboveFloor {
                objective_change,
                objective_floor,
            } => write!(
                f,
                "|Δobjective|={objective_change:.3e} is not ≤ objective_floor={objective_floor:.3e}"
            ),
            Self::ModelInexact {
                scalar_model_relerr,
                bound,
            } => write!(f, "scalar_relerr={scalar_model_relerr:.3e} is not ≤ {bound:.0e}"),
            Self::StepAboveTolerance {
                accepted_step_inf,
                step_tol,
            } => write!(
                f,
                "accepted_step_inf={accepted_step_inf:.3e} is not ≤ step_tol={step_tol:.3e}"
            ),
            Self::HpenNullity { nullity } => write!(
                f,
                "H_pen nullity={nullity} at the eigensolver resolution λ_max·√p·ε is not 0"
            ),
            Self::HpenNullityUnavailable => write!(
                f,
                "H_pen nullity unavailable (materialization or eigendecomposition failed)"
            ),
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

#[derive(Debug, Clone, PartialEq)]
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

impl JointNewtonTerminalReason {
    /// A stable snake_case label for this verdict, for language boundaries that
    /// need to branch on the reason without parsing its rendered text. The
    /// match is exhaustive so a new reason must be named when it is added.
    #[must_use]
    pub fn label(&self) -> &'static str {
        match self {
            Self::CycleBudget => "cycle_budget",
            Self::FullyRejectedExactFixedPoint { .. } => "fully_rejected_exact_fixed_point",
            Self::FullyRejectedAtTrustRegionFloor { .. } => "fully_rejected_at_trust_region_floor",
            Self::SlowGeometricRate { .. } => "slow_geometric_rate",
            Self::ResidualNotContracting { .. } => "residual_not_contracting",
            Self::StalledOnDescendingRay { .. } => "stalled_on_descending_ray",
            Self::ConstrainedFixedPointDeclined { .. } => "constrained_fixed_point_declined",
            Self::NonFiniteCurvature { .. } => "non_finite_curvature",
            Self::NonFiniteInnerState { .. } => "non_finite_inner_state",
            Self::KktCertificateRefused { .. } => "kkt_certificate_refused",
            Self::ResidualStall { .. } => "residual_stall",
            Self::FlatResidualStall { .. } => "flat_residual_stall",
        }
    }
}

impl InnerConvergenceTerminalState {
    /// The stable label of the verdict this terminal state records: the
    /// joint-Newton termination reason, or the blockwise route's own exit.
    #[must_use]
    pub fn reason_label(&self) -> &'static str {
        match self {
            Self::Blockwise { .. } => "blockwise_not_converged",
            Self::JointNewton {
                termination_reason, ..
            } => termination_reason.label(),
        }
    }
}

/// The facts a fit-ending non-convergence carries, read from its terminal
/// refusal for language boundaries (gam#2943). Every field is typed data; the
/// rendered message is built from the same facts.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct TerminalInnerModeEvidence<'a> {
    /// Spec name of the block holding the largest unresolved KKT residual.
    pub carrying_block: Option<&'a str>,
    /// Cycles the terminal inner solve ran.
    pub cycles: usize,
    /// The inner cycle budget it ran against, where the producer recorded one.
    pub cycle_budget: Option<usize>,
    /// Sup-norm of the projected KKT residual at the terminal iterate.
    pub kkt_residual: Option<f64>,
    /// The stationarity tolerance that residual was compared against.
    pub kkt_tol: Option<f64>,
    /// [`InnerConvergenceTerminalState::reason_label`] of the terminal verdict.
    pub terminal_reason: Option<&'static str>,
}

/// The message of [`CustomFamilyError::FitEndedWithoutCertifiedInnerMode`],
/// built from the terminal refusal's facts. It never reuses that refusal's own
/// text, which says the outer search may step away: untrue once the fit has
/// ended (gam#2943).
fn render_fit_ended_without_certified_inner_mode(refusal: &CustomFamilyError) -> String {
    let CustomFamilyError::InnerSolveNotConverged {
        cycles,
        terminal,
        kkt_residual,
        kkt_tol,
        cycle_budget,
        carrying_block,
        ..
    } = refusal
    else {
        return "custom-family fit ended without a certified inner mode; no fitted model was \
                assembled"
            .to_string();
    };
    let cycles_run = match cycle_budget {
        Some(budget) => format!("{cycles} of {budget} cycle(s)"),
        None => format!("{cycles} cycle(s) (no budget was recorded)"),
    };
    format!(
        "custom-family fit ended without a certified inner mode, so no fitted model was \
         assembled: the terminal inner solve stopped after {cycles_run} without certifying; \
         carrying block: {}; {}; terminal verdict: {}",
        carrying_block.as_deref().unwrap_or("not identified"),
        render_projected_kkt_comparison(*kkt_residual, *kkt_tol),
        match terminal {
            Some(state) => state.to_string(),
            None => "no terminal convergence state was recorded".to_string(),
        },
    )
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
        /// The inner cycle budget the solve ran against (`inner_max_cycles`),
        /// recorded where the refusal is built, or `None` where the producer
        /// had no budget in hand. A fit-ending boundary reads it here because
        /// it has no uniform handle on the options (gam#2943).
        cycle_budget: Option<usize>,
        /// The spec name of the block holding the largest unresolved projected
        /// KKT residual, where that residual is laid out in joint coefficient
        /// order; `None` where no such layout exists (gam#2943).
        carrying_block: Option<String>,
    },
    /// The fit ENDED without a certified inner mode, so no model was assembled.
    ///
    /// [`Self::InnerSolveNotConverged`] rejects one trial point that the outer
    /// search may step away from; this variant says that nothing stepped away
    /// and the fit is over. `refusal` is that terminal `InnerSolveNotConverged`
    /// record, held whole, so there is one field list for cycles, budget,
    /// carrying block and verdict. It is built only by
    /// [`CustomFamilyError::fit_ended_without_certified_inner_mode`], at the
    /// boundary that hands a fit to its caller, never inside a trial (gam#2943).
    ///
    /// It carries no Jeffreys arming evidence. It is minted above every arming
    /// consumer, and it may wrap the search's whole-search inner-refusal record,
    /// which arming never reads (#979, #2943).
    #[error("{}", render_fit_ended_without_certified_inner_mode(refusal))]
    FitEndedWithoutCertifiedInnerMode { refusal: Box<CustomFamilyError> },
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
    /// The outer smoothing search ended without a certified optimum after
    /// every strategy fallback, so no fit was assembled.
    ///
    /// `last_refusal` is the typed refusal of the search's last objective
    /// evaluation, or `None` when that evaluation did not refuse. It used to
    /// travel only inside [`Self::Optimization`]'s text, so a caller that acts
    /// on WHY the fit refused could only substring-match the message. The
    /// Jeffreys arming lifecycle is one: it arms on a descending ray, a null
    /// penalized Hessian or a divergent inner state (#979). The rendered message
    /// is the one that `Optimization` printed for this context.
    ///
    /// `search_inner_refusal` is the most recent uncertified inner solve
    /// (`InnerSolveNotConverged`) that any evaluation of the search raised. A
    /// later finite trial clears `last_refusal` but not this record, so the fit
    /// boundary (`FitFailure::ending_the_fit`) can name the inner solve that
    /// decided the fit (gam#2943). Nothing else reads it. Jeffreys arming reads
    /// `last_refusal`, because the search stepped away from every earlier refusal.
    #[error("custom-family optimization error in fit_custom_family outer smoothing: {reason}")]
    OuterSmoothingFailed {
        reason: String,
        last_refusal: Option<Box<CustomFamilyError>>,
        search_inner_refusal: Option<Box<CustomFamilyError>>,
        /// The outer search's own typed verdict, which `reason` renders. It is
        /// what decides the failure's category: a search whose every seed was
        /// refused and one that started and did not converge render into the
        /// same sentence here (#2937).
        outer_error: std::sync::Arc<crate::EstimationError>,
    },
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

    /// The ray the inner joint Newton was still descending when it stopped, or
    /// `None` for every other refusal.
    ///
    /// A ray is a property of the accepted step, so every terminal reason that
    /// can carry one is read here (gam#2695): the slow-rate exit, and the
    /// residual-stall and divergence exits that used to drop it. The outer seed
    /// loop restores a [`DescendingRayExit::Closable`] ray by raising the named
    /// strengths. Either kind says the objective the solve minimized has no
    /// finite minimizer in reach along that direction at this `rho` (#979).
    ///
    /// The match is exhaustive, so a new terminal reason must be classified when
    /// it is added.
    #[must_use]
    pub fn descending_ray_exit(&self) -> Option<DescendingRayExit<'_>> {
        let Self::InnerSolveNotConverged {
            terminal:
                Some(InnerConvergenceTerminalState::JointNewton {
                    termination_reason,
                    ..
                }),
            ..
        } = self
        else {
            return None;
        };
        match termination_reason {
            JointNewtonTerminalReason::SlowGeometricRate { ray: Some(ray), .. }
            | JointNewtonTerminalReason::StalledOnDescendingRay { ray, .. } => {
                Some(DescendingRayExit::Closable(ray))
            }
            JointNewtonTerminalReason::SlowGeometricRate { ray: None, .. } => {
                Some(DescendingRayExit::Unpenalized)
            }
            JointNewtonTerminalReason::CycleBudget
            | JointNewtonTerminalReason::FullyRejectedExactFixedPoint { .. }
            | JointNewtonTerminalReason::FullyRejectedAtTrustRegionFloor { .. }
            | JointNewtonTerminalReason::ResidualNotContracting { .. }
            | JointNewtonTerminalReason::ConstrainedFixedPointDeclined { .. }
            | JointNewtonTerminalReason::NonFiniteCurvature { .. }
            | JointNewtonTerminalReason::NonFiniteInnerState { .. }
            | JointNewtonTerminalReason::KktCertificateRefused { .. }
            | JointNewtonTerminalReason::ResidualStall { .. }
            | JointNewtonTerminalReason::FlatResidualStall { .. } => None,
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

    fn budget_exhausted_refusal() -> CustomFamilyError {
        CustomFamilyError::InnerSolveNotConverged {
            cycles: 8,
            terminal: Some(InnerConvergenceTerminalState::JointNewton {
                cycle: 8,
                stationarity_residual: 1.081e3,
                residual_tol: 1.338e-2,
                stationarity_scale: 1.0,
                step_inf: 7.314e-2,
                step_tol: 1.0e-8,
                resolvable_negative_curvature: false,
                best_stationarity_residual: 1.081e3,
                cycles_since_best_residual: 0,
                termination_reason: JointNewtonTerminalReason::CycleBudget,
            }),
            kkt_residual: Some(1.081e3),
            kkt_tol: Some(5.352e-2),
            theta_dim: 2,
            rho_dim: 2,
            psi_dim: 0,
            cycle_budget: Some(8),
            carrying_block: Some("slope_surface".to_string()),
        }
    }

    #[test]
    fn a_fit_ending_refusal_wraps_only_an_uncertified_inner_solve_2943() {
        let refusal = budget_exhausted_refusal();
        assert!(refusal.is_trial_point_infeasible());
        assert!(
            refusal.terminal_inner_mode_evidence().is_none(),
            "a trial refusal is not the end of a fit"
        );

        let ended = CustomFamilyError::fit_ended_without_certified_inner_mode(refusal);
        assert!(
            !ended.is_trial_point_infeasible(),
            "no outer search remains once a fit has ended"
        );
        assert_eq!(
            ended.terminal_inner_mode_evidence(),
            Some(TerminalInnerModeEvidence {
                carrying_block: Some("slope_surface"),
                cycles: 8,
                cycle_budget: Some(8),
                kkt_residual: Some(1.081e3),
                kkt_tol: Some(5.352e-2),
                terminal_reason: Some("cycle_budget"),
            })
        );
        let message = ended.to_string();
        for words in [
            "ended without a certified inner mode",
            "8 of 8 cycle(s)",
            "carrying block: slope_surface",
        ] {
            assert!(message.contains(words), "missing {words:?} in: {message}");
        }
        assert!(
            !message.contains("step away"),
            "a fit that has ended must not say the outer search may step away: {message}"
        );

        let other = CustomFamilyError::TrialPointRefused {
            reason: "indefinite active face".to_string(),
        };
        let passed = CustomFamilyError::fit_ended_without_certified_inner_mode(other);
        assert!(
            matches!(passed, CustomFamilyError::TrialPointRefused { .. }),
            "only an uncertified inner solve is marked as a fit ending"
        );
        assert!(passed.terminal_inner_mode_evidence().is_none());

        let blockwise = InnerConvergenceTerminalState::Blockwise {
            cycle: 3,
            max_accepted_step: 1.0e-2,
            max_proposed_step: 1.0e-2,
            step_tol: 1.0e-8,
            objective_change: 1.0e-3,
            objective_tol: 1.0e-9,
            joint_stationarity_ok: false,
        };
        assert_eq!(blockwise.reason_label(), "blockwise_not_converged");
    }

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
            cycle_budget: None,
            carrying_block: None,
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
            cycle_budget: None,
            carrying_block: None,
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
            cycle_budget: None,
            carrying_block: None,
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

    #[test]
    fn every_non_converged_joint_newton_exit_names_its_reason_2695() {
        // The #2695 1569 seed-4 shape: the KKT refusal at cycle 47 of 200
        // (`diagnosis: rank_deficient_H_pen`) reached the outer as
        // `termination=cycle budget`, a budget it never exhausted.
        let refused = CustomFamilyError::InnerSolveNotConverged {
            cycles: 48,
            terminal: Some(InnerConvergenceTerminalState::JointNewton {
                cycle: 47,
                stationarity_residual: 3.807134e2,
                residual_tol: 2.790772e-3,
                stationarity_scale: 2.790772e8,
                step_inf: 2.090760e-1,
                step_tol: 1.203860e-10,
                resolvable_negative_curvature: false,
                best_stationarity_residual: 2.079474e0,
                cycles_since_best_residual: 29,
                termination_reason: JointNewtonTerminalReason::KktCertificateRefused {
                    diagnosis: crate::diagnostics::KktRefusalDiagnosis::RankDeficientHPen,
                },
            }),
            kkt_residual: None,
            kkt_tol: None,
            theta_dim: 5,
            rho_dim: 5,
            psi_dim: 0,
            cycle_budget: None,
            carrying_block: None,
        };
        let msg = refused.to_string();
        assert!(
            msg.contains("the KKT certificate refused the iterate: rank_deficient_H_pen"),
            "the refusal must name its diagnosis: {msg}"
        );
        assert!(
            !msg.contains("cycle budget"),
            "a refusal at cycle 47 of 200 must not claim the budget: {msg}"
        );

        let labelled = [
            (
                JointNewtonTerminalReason::NonFiniteCurvature { cycle: 12 },
                "non-finite curvature at cycle 12",
            ),
            (
                JointNewtonTerminalReason::NonFiniteInnerState {
                    residual: f64::NAN,
                    objective: 4.0e1,
                    log_likelihood: f64::NEG_INFINITY,
                },
                "non-finite inner state: stationarity residual=NaN, log-likelihood=-inf",
            ),
            (
                JointNewtonTerminalReason::ResidualStall {
                    residual: 1.034e1,
                    residual_tol: 5.427e-4,
                    best_residual: 1.499e1,
                    cycles_without_improvement: 78,
                    accepted_step_inf: 2.601e-3,
                    trust_radius: 1.314e-2,
                },
                "stopped improving for 78 cycles with the accepted steps clipped by the trust region",
            ),
            (
                JointNewtonTerminalReason::FlatResidualStall {
                    residual: 2.623726e1,
                    residual_tol: 2.790772e-3,
                    best_residual: 1.845416e-1,
                    cycles_without_improvement: 22,
                },
                "stayed flat for 22 cycles with every step inside the trust region",
            ),
            (
                JointNewtonTerminalReason::ResidualNotContracting {
                    rate_per_cycle: 1.1096,
                    window_cycles: 16,
                    residual: 3.252e1,
                    residual_tol: 6.525e-7,
                },
                "did not contract over the last 16 cycles (1.1096x per cycle)",
            ),
        ];
        for (reason, label) in labelled {
            let text = reason.to_string();
            assert!(text.contains(label), "expected `{label}` in: {text}");
            assert!(!text.contains("cycle budget"), "{text}");
        }
        // The finite value is not reported as diverged.
        let partial = JointNewtonTerminalReason::NonFiniteInnerState {
            residual: f64::INFINITY,
            objective: 4.0e1,
            log_likelihood: -2.5e1,
        }
        .to_string();
        assert!(
            partial.contains("stationarity residual=inf") && !partial.contains("objective="),
            "{partial}"
        );
        // Negative control: the exit that really exhausted the budget still says so.
        assert_eq!(JointNewtonTerminalReason::CycleBudget.to_string(), "cycle budget");
    }

    #[test]
    fn descending_ray_exit_reads_every_ray_carrying_terminal_reason_979() {
        let ray = RayRestoration {
            block: 1,
            rho_first: 2,
            rho_count: 1,
            log_strength_ratio: 0.75,
            likelihood_slope: -3.0,
            penalty_slope: 1.4,
            block_step_inf: 0.2,
            direction: std::sync::Arc::from(vec![0.1, -0.2, 0.3]),
        };
        let refusal = |termination_reason| CustomFamilyError::InnerSolveNotConverged {
            cycles: 9,
            terminal: Some(InnerConvergenceTerminalState::JointNewton {
                cycle: 8,
                stationarity_residual: 1.0e-1,
                residual_tol: 1.0e-6,
                stationarity_scale: 1.0,
                step_inf: 1.0e-2,
                step_tol: 1.0e-8,
                resolvable_negative_curvature: false,
                best_stationarity_residual: 1.0e-1,
                cycles_since_best_residual: 0,
                termination_reason,
            }),
            kkt_residual: None,
            kkt_tol: None,
            theta_dim: 3,
            rho_dim: 3,
            psi_dim: 0,
            cycle_budget: None,
            carrying_block: None,
        };

        let stalled = refusal(JointNewtonTerminalReason::StalledOnDescendingRay {
            residual: 1.0e-1,
            residual_tol: 1.0e-6,
            cycles: 9,
            ray: ray.clone(),
        });
        assert_eq!(
            stalled.descending_ray_exit(),
            Some(DescendingRayExit::Closable(&ray))
        );

        let slow = |carried| {
            refusal(JointNewtonTerminalReason::SlowGeometricRate {
                rate_per_cycle: 0.9,
                window_cycles: 4,
                projected_cycles_to_tolerance: 120,
                residual: 1.0e-1,
                residual_tol: 1.0e-6,
                ray: carried,
            })
        };
        assert_eq!(
            slow(Some(ray.clone())).descending_ray_exit(),
            Some(DescendingRayExit::Closable(&ray))
        );
        assert_eq!(
            slow(None).descending_ray_exit(),
            Some(DescendingRayExit::Unpenalized)
        );

        // Negative controls: a residual that did not contract along no ray, a
        // blockwise terminal state, and a refusal that is not an inner exit carry
        // no ray.
        let not_contracting = refusal(JointNewtonTerminalReason::ResidualNotContracting {
            rate_per_cycle: 1.2,
            window_cycles: 4,
            residual: 1.0e-1,
            residual_tol: 1.0e-6,
        });
        assert_eq!(not_contracting.descending_ray_exit(), None);
        let blockwise = CustomFamilyError::InnerSolveNotConverged {
            cycles: 9,
            terminal: Some(InnerConvergenceTerminalState::Blockwise {
                cycle: 8,
                max_accepted_step: 1.0e-2,
                max_proposed_step: 1.0e-2,
                step_tol: 1.0e-8,
                objective_change: 1.0e-3,
                objective_tol: 1.0e-9,
                joint_stationarity_ok: false,
            }),
            kkt_residual: None,
            kkt_tol: None,
            theta_dim: 3,
            rho_dim: 3,
            psi_dim: 0,
            cycle_budget: None,
            carrying_block: None,
        };
        assert_eq!(blockwise.descending_ray_exit(), None);
        assert_eq!(
            CustomFamilyError::trial_point("no Laplace mode at this rho").descending_ray_exit(),
            None
        );
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
            | Self::MapUniquenessFailure { .. }
            // The whole search refused; its last refusal is carried for the
            // caller to read, not re-graded here.
            | Self::OuterSmoothingFailed { .. }
            // Minted only where a fit ends, so no outer search remains that
            // could step away (gam#2943).
            | Self::FitEndedWithoutCertifiedInnerMode { .. } => false,
        }
    }

    /// The fixed category of this failure (#2937). Exhaustive with no wildcard
    /// arm, for the reason [`Self::is_trial_point_infeasible`] is.
    #[must_use]
    pub fn failure_category(&self) -> crate::FailureCategory {
        use crate::FailureCategory;
        match self {
            // "custom-family optimization error in {context}": the outer or
            // inner optimizer of a context ended without its certificate.
            Self::Optimization { .. }
            | Self::InnerSolveNotConverged { .. }
            // The fit ended holding an uncertified inner solve (gam#2943).
            | Self::FitEndedWithoutCertifiedInnerMode { .. }
            // Reaching the boundary, a trial-point refusal means the search
            // never found a point it could evaluate.
            | Self::TrialPointRefused { .. } => FailureCategory::Convergence,
            Self::OuterSmoothingFailed { outer_error, .. } => outer_error.failure_category(),
            Self::InvalidInput { .. }
            | Self::UnsupportedConfiguration { .. }
            // Its producers refuse block specifications (duplicate names, bound
            // layouts) before any solve.
            | Self::ConstraintViolation { .. }
            | Self::IdentifiabilityFailure { .. }
            | Self::MapUniquenessFailure { .. } => FailureCategory::Input,
            Self::DimensionMismatch { .. } => FailureCategory::Invariant,
            Self::NumericalFailure { .. } | Self::BasisDecompositionFailed { .. } => {
                FailureCategory::Numerical
            }
        }
    }

    /// The `Enum::Variant` name of this error, the one a front end prints
    /// beside the message (#2937). A whole-search refusal is named by the
    /// outer search's verdict it carries.
    #[must_use]
    pub fn variant_name(&self) -> &'static str {
        match self {
            Self::InvalidInput { .. } => "CustomFamilyError::InvalidInput",
            Self::Optimization { .. } => "CustomFamilyError::Optimization",
            Self::DimensionMismatch { .. } => "CustomFamilyError::DimensionMismatch",
            Self::NumericalFailure { .. } => "CustomFamilyError::NumericalFailure",
            Self::ConstraintViolation { .. } => "CustomFamilyError::ConstraintViolation",
            Self::UnsupportedConfiguration { .. } => "CustomFamilyError::UnsupportedConfiguration",
            Self::InnerSolveNotConverged { .. } => "CustomFamilyError::InnerSolveNotConverged",
            Self::BasisDecompositionFailed { .. } => "CustomFamilyError::BasisDecompositionFailed",
            Self::IdentifiabilityFailure { .. } => "CustomFamilyError::IdentifiabilityFailure",
            Self::MapUniquenessFailure { .. } => "CustomFamilyError::MapUniquenessFailure",
            Self::TrialPointRefused { .. } => "CustomFamilyError::TrialPointRefused",
            Self::OuterSmoothingFailed { outer_error, .. } => outer_error.variant_name(),
            Self::FitEndedWithoutCertifiedInnerMode { .. } => {
                "CustomFamilyError::FitEndedWithoutCertifiedInnerMode"
            }
        }
    }

    /// Mark the terminal refusal of a fit that ended without a certified inner
    /// mode.
    ///
    /// This is the only constructor of
    /// [`Self::FitEndedWithoutCertifiedInnerMode`]. It wraps an
    /// [`Self::InnerSolveNotConverged`] whole and returns any other refusal
    /// unchanged, so the wrapped refusal always carries the cycles, budget,
    /// carrying block and verdict. Call it only where a fit is handed to its
    /// caller: inside a trial the same refusal must stay
    /// `InnerSolveNotConverged`, which the outer search steps away from
    /// (gam#2943).
    #[must_use]
    pub fn fit_ended_without_certified_inner_mode(refusal: CustomFamilyError) -> CustomFamilyError {
        match refusal {
            refusal @ Self::InnerSolveNotConverged { .. } => Self::FitEndedWithoutCertifiedInnerMode {
                refusal: Box::new(refusal),
            },
            other => other,
        }
    }

    /// The typed facts of a fit that ended without a certified inner mode, read
    /// from its terminal refusal, or `None` for every other error.
    #[must_use]
    pub fn terminal_inner_mode_evidence(&self) -> Option<TerminalInnerModeEvidence<'_>> {
        let Self::FitEndedWithoutCertifiedInnerMode { refusal } = self else {
            return None;
        };
        let Self::InnerSolveNotConverged {
            cycles,
            terminal,
            kkt_residual,
            kkt_tol,
            cycle_budget,
            carrying_block,
            ..
        } = refusal.as_ref()
        else {
            return None;
        };
        Some(TerminalInnerModeEvidence {
            carrying_block: carrying_block.as_deref(),
            cycles: *cycles,
            cycle_budget: *cycle_budget,
            kkt_residual: *kkt_residual,
            kkt_tol: *kkt_tol,
            terminal_reason: terminal
                .as_ref()
                .map(InnerConvergenceTerminalState::reason_label),
        })
    }
}
