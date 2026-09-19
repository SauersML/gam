//! Typed evidence that a custom-family fit armed its Jeffreys/Firth prior (#979).

use serde::{Deserialize, Serialize};

use crate::custom_family_error::{
    ConstrainedFixedPointCondition, CustomFamilyError, InnerConvergenceTerminalState,
    JointNewtonTerminalReason,
};
use crate::diagnostics::KktRefusalDiagnosis;

/// Why a custom-family fit armed its Jeffreys/Firth prior (#979, Jeffreys ruling (b)).
///
/// A family with a separation or under-identification regime fits its unarmed
/// objective first, and arms once, only when that fit's own refusal proves the
/// unarmed objective has no finite stationary point with positive-definite
/// information on the identified span. Each variant is one such proof, carrying
/// the numbers it was decided on. No variant is a band picked here: each verdict
/// is the producing solver's own certificate.
///
/// Every field is finite, so the evidence survives a saved model's JSON round
/// trip.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub enum JeffreysArmingEvidence {
    /// The inner joint Newton stopped while still descending a ray that block
    /// `block`'s penalty opposes but does not close at the selected strength.
    DescendingRay {
        block: usize,
        /// `ln r`, the rise in the block's log strengths that would close it.
        log_strength_ratio: f64,
        /// `∇(−ℓ)·δ` along the accepted step.
        likelihood_slope: f64,
        /// `(λ_b S_b β)·δ` along the accepted step.
        penalty_slope: f64,
        /// The accepted step `δ` the solve was still descending, in raw joint
        /// coefficient order: the direction a family's measured Jeffreys span can
        /// state (#979).
        direction: Vec<f64>,
    },
    /// The inner joint Newton was descending a direction no block's penalty
    /// opposes, so no `rho` bounds it.
    UnpenalizedDescendingRay { rate_per_cycle: f64 },
    /// The penalized Hessian has a numerical null space at the terminal iterate.
    /// `nullity` is the constrained fixed-point certificate's count, or `None`
    /// when the KKT refusal report diagnosed the rank deficiency without a count.
    NullPenalizedHessian { nullity: Option<usize> },
    /// The inner state went non-finite. Each flag says whether that decision
    /// value was still finite when the divergence guard fired.
    DivergentInnerState {
        residual_finite: bool,
        objective_finite: bool,
        log_likelihood_finite: bool,
    },
    /// A first-order stationary point whose exact penalized Hessian has
    /// resolvable negative curvature.
    StrictSaddle { stationarity_residual: f64 },
    /// The fit certified a constrained mode whose cone-truncated posterior is
    /// proved improper: the precision curves down along a feasible direction
    /// that leaves the active face. The counts are the negative inertia of the
    /// ambient precision `H`, of the reduced precision `M` on the constraint
    /// normals, and of `ZᵀHZ` on the lineality space. `copositive_minimum` is
    /// `min wᵀMw` over the simplex when the certificate computed it.
    ImproperConePosterior {
        ambient_negative: usize,
        reduced_negative: usize,
        lineality_negative: usize,
        copositive_minimum: Option<f64>,
    },
}

impl CustomFamilyError {
    /// The Jeffreys arming evidence this refusal carries, or `None` when it
    /// proves nothing about the unarmed objective's stationary points.
    ///
    /// A whole-search refusal is read through the typed refusal of its last
    /// objective evaluation (see [`CustomFamilyError::OuterSmoothingFailed`]).
    /// Its `search_inner_refusal` is never read here: the search stepped away
    /// from that refusal, so it proves nothing about the objective where the
    /// search ended (#2943). The match over joint-Newton terminal reasons is
    /// exhaustive, so a new reason must be graded when it is added.
    #[must_use]
    pub fn jeffreys_arming_evidence(&self) -> Option<JeffreysArmingEvidence> {
        let Self::InnerSolveNotConverged {
            terminal:
                Some(InnerConvergenceTerminalState::JointNewton {
                    stationarity_residual,
                    resolvable_negative_curvature,
                    termination_reason,
                    ..
                }),
            ..
        } = self
        else {
            return match self {
                Self::OuterSmoothingFailed { last_refusal, .. } => last_refusal
                    .as_deref()
                    .and_then(Self::jeffreys_arming_evidence),
                _ => None,
            };
        };
        let evidence = match termination_reason {
            JointNewtonTerminalReason::SlowGeometricRate { ray: Some(ray), .. }
            | JointNewtonTerminalReason::StalledOnDescendingRay { ray, .. } => {
                Some(JeffreysArmingEvidence::DescendingRay {
                    block: ray.block,
                    log_strength_ratio: ray.log_strength_ratio,
                    likelihood_slope: ray.likelihood_slope,
                    penalty_slope: ray.penalty_slope,
                    direction: ray.direction.to_vec(),
                })
            }
            JointNewtonTerminalReason::SlowGeometricRate {
                ray: None,
                rate_per_cycle,
                ..
            } => Some(JeffreysArmingEvidence::UnpenalizedDescendingRay {
                rate_per_cycle: *rate_per_cycle,
            }),
            JointNewtonTerminalReason::ConstrainedFixedPointDeclined {
                condition: ConstrainedFixedPointCondition::HpenNullity { nullity },
            } => Some(JeffreysArmingEvidence::NullPenalizedHessian {
                nullity: Some(*nullity),
            }),
            JointNewtonTerminalReason::KktCertificateRefused {
                diagnosis: KktRefusalDiagnosis::RankDeficientHPen,
            } => Some(JeffreysArmingEvidence::NullPenalizedHessian { nullity: None }),
            JointNewtonTerminalReason::NonFiniteInnerState {
                residual,
                objective,
                log_likelihood,
            } => Some(JeffreysArmingEvidence::DivergentInnerState {
                residual_finite: residual.is_finite(),
                objective_finite: objective.is_finite(),
                log_likelihood_finite: log_likelihood.is_finite(),
            }),
            // Budget, trust-region, contraction and active-set verdicts, and a
            // non-finite curvature entry, say the solve did not finish. They do
            // not say the objective has no finite stationary point.
            JointNewtonTerminalReason::CycleBudget
            | JointNewtonTerminalReason::FullyRejectedExactFixedPoint { .. }
            | JointNewtonTerminalReason::FullyRejectedAtTrustRegionFloor { .. }
            | JointNewtonTerminalReason::ResidualNotContracting { .. }
            | JointNewtonTerminalReason::ConstrainedFixedPointDeclined {
                condition:
                    ConstrainedFixedPointCondition::ObjectiveAboveFloor { .. }
                    | ConstrainedFixedPointCondition::ModelInexact { .. }
                    | ConstrainedFixedPointCondition::StepAboveTolerance { .. }
                    | ConstrainedFixedPointCondition::HpenNullityUnavailable,
            }
            | JointNewtonTerminalReason::NonFiniteCurvature { .. }
            | JointNewtonTerminalReason::KktCertificateRefused {
                diagnosis:
                    KktRefusalDiagnosis::PhantomMultiplierWithWellConditionedH
                    | KktRefusalDiagnosis::ActiveSetIncomplete
                    | KktRefusalDiagnosis::AliasingDetectedAtFit,
            }
            | JointNewtonTerminalReason::ResidualStall { .. }
            | JointNewtonTerminalReason::FlatResidualStall { .. } => None,
        };
        evidence.or_else(|| {
            resolvable_negative_curvature.then_some(JeffreysArmingEvidence::StrictSaddle {
                stationarity_residual: *stationarity_residual,
            })
        })
    }

    /// Replace the direction of the descending ray this refusal carries with
    /// `lift(direction)`, reading a whole-search refusal through its last
    /// objective refusal. A refusal that carries no ray is unchanged.
    ///
    /// The inner solve raises its ray in reduced coordinates. The fit calls this
    /// where the refusal leaves it, with the identifiability gauge's linear
    /// lift, so arming evidence reads the direction in raw joint order (#979).
    pub fn map_descending_ray_direction(&mut self, lift: &dyn Fn(&[f64]) -> std::sync::Arc<[f64]>) {
        // Both refusals a whole-search refusal carries are lifted, so they read in
        // one coordinate order (#2943).
        if let Self::OuterSmoothingFailed {
            last_refusal,
            search_inner_refusal,
            ..
        } = self
        {
            for refusal in [last_refusal, search_inner_refusal].into_iter().flatten() {
                refusal.map_descending_ray_direction(lift);
            }
        }
        if let Self::InnerSolveNotConverged {
            terminal:
                Some(InnerConvergenceTerminalState::JointNewton {
                    termination_reason:
                        JointNewtonTerminalReason::SlowGeometricRate { ray: Some(ray), .. }
                        | JointNewtonTerminalReason::StalledOnDescendingRay { ray, .. },
                    ..
                }),
            ..
        } = self
        {
            ray.direction = lift(&ray.direction);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::custom_family_error::RayRestoration;

    fn joint_newton_refusal(
        termination_reason: JointNewtonTerminalReason,
        resolvable_negative_curvature: bool,
    ) -> CustomFamilyError {
        CustomFamilyError::InnerSolveNotConverged {
            cycles: 9,
            terminal: Some(InnerConvergenceTerminalState::JointNewton {
                cycle: 8,
                stationarity_residual: 2.5e-2,
                residual_tol: 1.0e-6,
                stationarity_scale: 1.0,
                step_inf: 1.0e-2,
                step_tol: 1.0e-8,
                resolvable_negative_curvature,
                best_stationarity_residual: 2.5e-2,
                cycles_since_best_residual: 0,
                termination_reason,
            }),
            kkt_residual: None,
            kkt_tol: None,
            theta_dim: 3,
            rho_dim: 3,
            psi_dim: 0,
            cycle_budget: Some(9),
            carrying_block: None,
        }
    }

    #[test]
    fn only_the_last_evaluation_refusal_arms_a_whole_search_refusal_2943() {
        // A whole-search refusal carries two refusals (gam#2943). `last_refusal`
        // is the typed refusal of the search's last objective evaluation, and
        // arming reads it. `search_inner_refusal` is the search's most recent
        // uncertified inner solve, kept across the finite trials after it so the
        // fit boundary can name it. The search stepped away from that refusal, so
        // it proves nothing about the objective where the search ended.
        let stalled = joint_newton_refusal(
            JointNewtonTerminalReason::StalledOnDescendingRay {
                residual: 1.0e-1,
                residual_tol: 1.0e-6,
                cycles: 9,
                ray: RayRestoration {
                    block: 1,
                    rho_first: 2,
                    rho_count: 1,
                    log_strength_ratio: 0.75,
                    likelihood_slope: -3.0,
                    penalty_slope: 1.4,
                    block_step_inf: 0.2,
                    direction: std::sync::Arc::from(vec![0.1, -0.2, 0.3]),
                },
            },
            false,
        );
        let evidence = stalled.jeffreys_arming_evidence();
        assert!(evidence.is_some(), "the fixture must carry arming evidence");
        let nullity = joint_newton_refusal(
            JointNewtonTerminalReason::ConstrainedFixedPointDeclined {
                condition: ConstrainedFixedPointCondition::HpenNullity { nullity: 2 },
            },
            false,
        );
        assert!(
            nullity.jeffreys_arming_evidence().is_some()
                && nullity.jeffreys_arming_evidence() != evidence,
            "the whole-search record must carry evidence distinct from the last refusal's"
        );
        let budget_only = joint_newton_refusal(JointNewtonTerminalReason::CycleBudget, false);
        assert_eq!(
            budget_only.jeffreys_arming_evidence(),
            None,
            "the no-evidence ending must carry no evidence"
        );
        let search = |last_refusal: Option<CustomFamilyError>,
                      search_inner_refusal: Option<CustomFamilyError>| {
            CustomFamilyError::OuterSmoothingFailed {
                reason: "outer smoothing optimization failed".to_string(),
                last_refusal: last_refusal.map(Box::new),
                search_inner_refusal: search_inner_refusal.map(Box::new),
                outer_error: std::sync::Arc::new(crate::EstimationError::RemlOptimizationFailed(
                    "outer smoothing optimization failed".to_string(),
                )),
            }
        };
        assert_eq!(
            search(Some(stalled.clone()), Some(nullity)).jeffreys_arming_evidence(),
            evidence,
            "the last evaluation's refusal arms the search, never the whole-search record"
        );
        assert_eq!(
            search(None, Some(stalled.clone())).jeffreys_arming_evidence(),
            None,
            "a ray refusal the search stepped away from must not arm it"
        );
        assert_eq!(
            search(Some(budget_only), Some(stalled.clone())).jeffreys_arming_evidence(),
            None,
            "a search that ends on a refusal without evidence must not arm on an earlier ray"
        );
        assert_eq!(
            CustomFamilyError::fit_ended_without_certified_inner_mode(stalled)
                .jeffreys_arming_evidence(),
            None,
            "a fit-ending refusal is minted above every arming consumer and never arms"
        );
    }

    #[test]
    fn jeffreys_arming_evidence_reads_each_certified_refusal_979() {
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
        let stalled = joint_newton_refusal(
            JointNewtonTerminalReason::StalledOnDescendingRay {
                residual: 1.0e-1,
                residual_tol: 1.0e-6,
                cycles: 9,
                ray,
            },
            false,
        );
        let ray_evidence = JeffreysArmingEvidence::DescendingRay {
            block: 1,
            log_strength_ratio: 0.75,
            likelihood_slope: -3.0,
            penalty_slope: 1.4,
            direction: vec![0.1, -0.2, 0.3],
        };
        assert_eq!(stalled.jeffreys_arming_evidence(), Some(ray_evidence.clone()));

        let unpenalized = joint_newton_refusal(
            JointNewtonTerminalReason::SlowGeometricRate {
                rate_per_cycle: 0.9,
                window_cycles: 4,
                projected_cycles_to_tolerance: 120,
                residual: 1.0e-1,
                residual_tol: 1.0e-6,
                ray: None,
            },
            false,
        );
        assert_eq!(
            unpenalized.jeffreys_arming_evidence(),
            Some(JeffreysArmingEvidence::UnpenalizedDescendingRay { rate_per_cycle: 0.9 })
        );

        let nullity = joint_newton_refusal(
            JointNewtonTerminalReason::ConstrainedFixedPointDeclined {
                condition: ConstrainedFixedPointCondition::HpenNullity { nullity: 2 },
            },
            false,
        );
        assert_eq!(
            nullity.jeffreys_arming_evidence(),
            Some(JeffreysArmingEvidence::NullPenalizedHessian { nullity: Some(2) })
        );
        let rank_deficient = joint_newton_refusal(
            JointNewtonTerminalReason::KktCertificateRefused {
                diagnosis: KktRefusalDiagnosis::RankDeficientHPen,
            },
            false,
        );
        assert_eq!(
            rank_deficient.jeffreys_arming_evidence(),
            Some(JeffreysArmingEvidence::NullPenalizedHessian { nullity: None })
        );

        let divergent = joint_newton_refusal(
            JointNewtonTerminalReason::NonFiniteInnerState {
                residual: f64::NAN,
                objective: 1.0,
                log_likelihood: f64::NEG_INFINITY,
            },
            false,
        );
        let divergent_evidence = JeffreysArmingEvidence::DivergentInnerState {
            residual_finite: false,
            objective_finite: true,
            log_likelihood_finite: false,
        };
        assert_eq!(divergent.jeffreys_arming_evidence(), Some(divergent_evidence.clone()));

        let saddle = joint_newton_refusal(JointNewtonTerminalReason::CycleBudget, true);
        assert_eq!(
            saddle.jeffreys_arming_evidence(),
            Some(JeffreysArmingEvidence::StrictSaddle {
                stationarity_residual: 2.5e-2
            })
        );

        // A whole-search refusal is read through its last objective refusal.
        let search = |last_refusal: Option<CustomFamilyError>| {
            CustomFamilyError::OuterSmoothingFailed {
                reason: "outer smoothing optimization failed".to_string(),
                last_refusal: last_refusal.map(Box::new),
                search_inner_refusal: None,
                outer_error: std::sync::Arc::new(crate::EstimationError::RemlOptimizationFailed(
                    "outer smoothing optimization failed".to_string(),
                )),
            }
        };
        assert_eq!(
            search(Some(stalled)).jeffreys_arming_evidence(),
            Some(ray_evidence.clone())
        );
        assert_eq!(search(None).jeffreys_arming_evidence(), None);

        // The fit lifts the ray's direction where the refusal leaves it, through the
        // last refusal of a whole search, and a refusal without a ray is unchanged.
        let double = |direction: &[f64]| -> std::sync::Arc<[f64]> {
            direction.iter().map(|value| 2.0 * value).collect()
        };
        let mut lifted = search(Some(joint_newton_refusal(
            JointNewtonTerminalReason::StalledOnDescendingRay {
                residual: 1.0e-1,
                residual_tol: 1.0e-6,
                cycles: 9,
                ray: RayRestoration {
                    block: 1,
                    rho_first: 2,
                    rho_count: 1,
                    log_strength_ratio: 0.75,
                    likelihood_slope: -3.0,
                    penalty_slope: 1.4,
                    block_step_inf: 0.2,
                    direction: std::sync::Arc::from(vec![0.1, -0.2, 0.3]),
                },
            },
            false,
        )));
        lifted.map_descending_ray_direction(&double);
        assert_eq!(
            lifted.jeffreys_arming_evidence(),
            Some(JeffreysArmingEvidence::DescendingRay {
                block: 1,
                log_strength_ratio: 0.75,
                likelihood_slope: -3.0,
                penalty_slope: 1.4,
                direction: vec![0.2, -0.4, 0.6],
            })
        );
        let mut budget_only = joint_newton_refusal(JointNewtonTerminalReason::CycleBudget, false);
        budget_only.map_descending_ray_direction(&double);
        assert_eq!(budget_only.jeffreys_arming_evidence(), None);

        // Negative controls: an unfinished solve is not evidence, whichever exit
        // reported it.
        let not_contracting = joint_newton_refusal(
            JointNewtonTerminalReason::ResidualNotContracting {
                rate_per_cycle: 1.2,
                window_cycles: 4,
                residual: 1.0e-1,
                residual_tol: 1.0e-6,
            },
            false,
        );
        assert_eq!(not_contracting.jeffreys_arming_evidence(), None);
        let active_set = joint_newton_refusal(
            JointNewtonTerminalReason::KktCertificateRefused {
                diagnosis: KktRefusalDiagnosis::ActiveSetIncomplete,
            },
            false,
        );
        assert_eq!(active_set.jeffreys_arming_evidence(), None);
        let budget = joint_newton_refusal(JointNewtonTerminalReason::CycleBudget, false);
        assert_eq!(budget.jeffreys_arming_evidence(), None);
        assert_eq!(
            search(Some(not_contracting)).jeffreys_arming_evidence(),
            None
        );
        assert_eq!(
            CustomFamilyError::trial_point("no Laplace mode at this rho").jeffreys_arming_evidence(),
            None
        );

        // The evidence survives a saved model's JSON round trip.
        for evidence in [ray_evidence, divergent_evidence] {
            let wire = serde_json::to_string(&evidence).unwrap();
            let restored: JeffreysArmingEvidence = serde_json::from_str(&wire).unwrap();
            assert_eq!(restored, evidence);
        }
    }
}
