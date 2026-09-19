//! #2665: a saddle-escape trial the criterion could not evaluate falsifies nothing.
//!
//! `adjudicate_negative_curvature` withdraws a measured negative-curvature verdict only when
//! the criterion, evaluated along the disputed eigenvector, fails to descend anywhere in the
//! claim's falsifiable range. A trial whose evaluation errored or came back non-finite said
//! nothing about the criterion there, so it is not a probe. A ladder on which no trial
//! evaluated declines, and the verdict it was asked about stands.
//!
//! The criterion is `V(θ) = ½|θ|²`, convex, so no feasible step descends. The analytic
//! Hessian the adjudication is handed claims curvature `CLAIMED_CURVATURE` along the first
//! axis, which every evaluable trial along that axis contradicts.

use super::*;
use ndarray::{Array1, Array2, array};

const CLAIMED_CURVATURE: f64 = -1.0;
const RESOLUTION: f64 = 1.0e-6;
const REFUSAL_MARKER: &str = "the #2665 fixture refuses this trial";

/// `V(θ) = ½|θ|²`, refusing every trial off the checkpoint `θ = 0` whose step along the
/// disputed axis is at least `refused_from_step` long. Zero refuses every such trial.
struct RefusingCriterion {
    refused_from_step: f64,
    refused_trials: usize,
    evaluated_trials: usize,
}

impl RefusingCriterion {
    fn new(refused_from_step: f64) -> Self {
        Self {
            refused_from_step,
            refused_trials: 0,
            evaluated_trials: 0,
        }
    }

    fn value(&mut self, theta: &Array1<f64>) -> Result<f64, EstimationError> {
        let at_checkpoint = theta.iter().all(|coordinate| *coordinate == 0.0);
        if !at_checkpoint {
            if theta[0].abs() >= self.refused_from_step {
                self.refused_trials += 1;
                return Err(EstimationError::TrialPointRefused {
                    reason: REFUSAL_MARKER.to_string(),
                });
            }
            self.evaluated_trials += 1;
        }
        Ok(0.5 * theta.dot(theta))
    }
}

impl OuterObjective for RefusingCriterion {
    fn capability(&self) -> OuterCapability {
        OuterCapability {
            gradient: Derivative::Analytic,
            hessian: DeclaredHessianForm::Dense,
            n_params: 2,
            psi_dim: 0,
            fixed_point_available: false,
            barrier_config: None,
            prefer_gradient_only: false,
            disable_fixed_point: true,
        }
    }
    fn eval_cost(&mut self, theta: &Array1<f64>) -> Result<f64, EstimationError> {
        self.value(theta)
    }
    fn eval(&mut self, theta: &Array1<f64>) -> Result<OuterEval, EstimationError> {
        Ok(OuterEval {
            cost: self.value(theta)?,
            gradient: theta.clone(),
            hessian: HessianValue::Unavailable,
            inner_beta_hint: None,
        })
    }
    fn eval_with_order(
        &mut self,
        theta: &Array1<f64>,
        order: OuterEvalOrder,
    ) -> Result<OuterEval, EstimationError> {
        let full = self.eval(theta)?;
        Ok(match order {
            OuterEvalOrder::Value => OuterEval {
                cost: full.cost,
                gradient: Array1::<f64>::zeros(theta.len()),
                hessian: HessianValue::Unavailable,
                inner_beta_hint: None,
            },
            OuterEvalOrder::ValueAndGradient | OuterEvalOrder::ValueGradientHessian => full,
        })
    }
    fn reset(&mut self) {
        // A pure function of theta: no warm state to re-baseline.
    }
    fn seed_inner_state(&mut self, beta: &Array1<f64>) -> Result<SeedOutcome, EstimationError> {
        if beta.iter().any(|value| !value.is_finite()) {
            return Err(EstimationError::RemlOptimizationFailed(
                "the #2665 fixture was offered a non-finite inner seed".to_string(),
            ));
        }
        Ok(SeedOutcome::NoSlot)
    }
}

/// Adjudicate the claimed curvature at the checkpoint `θ = 0`, where the gradient is zero.
fn adjudicate(objective: &mut RefusingCriterion, context: &str) -> SaddleAdjudication {
    let theta = Array1::<f64>::zeros(2);
    let gradient = Array1::<f64>::zeros(2);
    let claimed: Array2<f64> = array![[CLAIMED_CURVATURE, 0.0], [0.0, 1.0]];
    let bounds = (array![-30.0, -30.0], array![30.0, 30.0]);
    adjudicate_negative_curvature(
        objective, &theta, &gradient, &claimed, &[], None, 0.0, RESOLUTION, &bounds, context,
    )
}

/// Every trial off the checkpoint is refused, so nothing was evaluated along the claim and
/// nothing contradicted it. Counting a refused trial as probed made this ladder report
/// `Contradicted`, and the certificate then withdrew a measured strict-saddle verdict on zero
/// evaluations.
#[test]
fn a_ladder_whose_every_trial_is_refused_declines_and_keeps_the_verdict_2665() {
    let mut objective = RefusingCriterion::new(0.0);
    let verdict = adjudicate(&mut objective, "#2665 every trial refused");
    let refused = objective.refused_trials;
    assert!(
        refused > 0 && objective.evaluated_trials == 0,
        "every trial off the checkpoint must have been refused: refused {refused}, evaluated {}",
        objective.evaluated_trials
    );
    let SaddleAdjudication::Declined(reason) = verdict else {
        panic!(
            "a ladder on which no trial evaluated falsifies nothing and must decline: \
             {verdict:?}"
        );
    };
    assert!(
        reason.contains(&format!("eval_failed={refused}, non_finite=0")),
        "the declined exit must count every refused trial as a failed evaluation: {reason}"
    );
}

/// Trials at the large steps are refused and the small ones evaluate. The criterion is convex,
/// so every evaluable trial contradicts the claim, and the contradiction counts exactly the
/// trials that evaluated.
#[test]
fn a_contradiction_counts_only_the_trials_that_evaluated_2665() {
    let mut objective = RefusingCriterion::new(0.1);
    let verdict = adjudicate(&mut objective, "#2665 large steps refused");
    assert!(
        objective.refused_trials > 0 && objective.evaluated_trials > 0,
        "the fixture must refuse the large steps and evaluate the small ones: refused {}, \
         evaluated {}",
        objective.refused_trials,
        objective.evaluated_trials
    );
    let SaddleAdjudication::Contradicted {
        probed,
        best_seen_cost,
        ..
    } = verdict
    else {
        panic!("a convex criterion contradicts the claimed negative curvature: {verdict:?}");
    };
    assert_eq!(
        probed, objective.evaluated_trials,
        "probed must count the trials that evaluated, not the {} that were refused",
        objective.refused_trials
    );
    assert!(
        best_seen_cost > 0.0,
        "no evaluated trial may reach the checkpoint's value on a convex criterion: \
         {best_seen_cost:e}"
    );
}
