//! #3036: a negative curvature the criterion cannot resolve at any adjudication step cannot
//! refuse the point, whatever its probes would have done.
//!
//! At a stationary point a claim `vᵀHv = λ_min < 0` predicts `V(ρ ± αv) − V(ρ) ≈ ½λ_min α²`
//! for every step `α ≤ α_max = 1` the adjudication may take. When `½|λ_min|·α_max²` does not
//! exceed the criterion's resolution, no allowed step can produce a decrease the criterion
//! represents: the claim's falsifiable range is empty. The adjudication used to probe `α_max`
//! anyway, so its verdict depended on whether two noise-level evaluations happened to succeed
//! — it "contradicted" the claim when they evaluated and declined when they failed, and the
//! declined exit refused the point on the matrix's word.
//!
//! The criterion here is `V(θ) = ½|θ|²` and refuses to evaluate every trial off the
//! checkpoint `θ = 0`, the failure the issue measured at the probes' inner solve.

use super::*;
use ndarray::{Array1, Array2, array};

/// The issue's instance (gnomon calibrate's Gaussian location-scale unit test at 40b5044e4f):
/// `λ_min(H) = −1.294787e-6` against `objective_resolution = 1.228631e-5`, so the largest
/// step predicts `6.47e-7`, 19x under the resolution.
const UNRESOLVABLE_CURVATURE: f64 = -1.294787e-6;
const RESOLUTION: f64 = 1.228631e-5;
/// A claim the same resolution does resolve: `½·1e-4 = 5e-5 > 1.228631e-5`.
const RESOLVABLE_CURVATURE: f64 = -1.0e-4;
/// The adjudication's largest step, one e-fold in log-λ.
const ALPHA_MAX: f64 = 1.0;

/// `V(θ) = ½|θ|²`, refusing every evaluation off the checkpoint and counting every
/// evaluation it was asked for.
#[derive(Default)]
struct RefusingCriterion {
    evaluations: usize,
    refused_trials: usize,
}

impl RefusingCriterion {
    fn value(&mut self, theta: &Array1<f64>) -> Result<f64, EstimationError> {
        self.evaluations += 1;
        if theta.iter().any(|coordinate| *coordinate != 0.0) {
            self.refused_trials += 1;
            return Err(EstimationError::TrialPointRefused {
                reason: "the #3036 fixture's inner solve stalls at every trial".to_string(),
            });
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
                "the #3036 fixture was offered a non-finite inner seed".to_string(),
            ));
        }
        Ok(SeedOutcome::NoSlot)
    }
}

/// Adjudicate a claimed curvature `lambda_min` along the first axis at the stationary
/// checkpoint `θ = 0`.
fn adjudicate(objective: &mut RefusingCriterion, lambda_min: f64) -> SaddleAdjudication {
    let theta = Array1::<f64>::zeros(2);
    let gradient = Array1::<f64>::zeros(2);
    let claimed: Array2<f64> = array![[lambda_min, 0.0], [0.0, 1.0]];
    let bounds = (array![-30.0, -30.0], array![30.0, 30.0]);
    adjudicate_negative_curvature(
        objective,
        &theta,
        &gradient,
        &claimed,
        &[],
        None,
        0.0,
        RESOLUTION,
        &bounds,
        "#3036 refusing criterion",
    )
}

/// The claim's resolvability is decided from `λ_min`, `α_max` and the resolution alone.
#[test]
fn a_negative_curvature_claim_is_resolvable_iff_its_largest_step_clears_the_resolution_3036() {
    let Some(NegativeCurvatureClaim::Unresolvable {
        predicted_at_largest,
    }) = negative_curvature_claim(UNRESOLVABLE_CURVATURE, ALPHA_MAX, RESOLUTION)
    else {
        panic!("the issue's claim predicts 6.47e-7 against a resolution of 1.23e-5");
    };
    assert_eq!(
        predicted_at_largest,
        0.5 * UNRESOLVABLE_CURVATURE.abs() * ALPHA_MAX * ALPHA_MAX
    );
    assert!(predicted_at_largest * 18.0 < RESOLUTION);

    assert_eq!(
        negative_curvature_claim(RESOLVABLE_CURVATURE, ALPHA_MAX, RESOLUTION),
        Some(NegativeCurvatureClaim::Resolvable {
            alpha_min: (2.0 * RESOLUTION / RESOLVABLE_CURVATURE.abs()).sqrt(),
        }),
        "a claim predicting 5e-5 at the largest step is falsifiable down to α_min"
    );

    // A prediction exactly at the resolution is not a decrease the criterion represents.
    let at_resolution = -2.0 * RESOLUTION / (ALPHA_MAX * ALPHA_MAX);
    assert!(matches!(
        negative_curvature_claim(at_resolution, ALPHA_MAX, RESOLUTION),
        Some(NegativeCurvatureClaim::Unresolvable { .. })
    ));

    // Inputs that carry no claim to judge.
    for (lambda_min, alpha_max, resolution) in [
        (0.0, ALPHA_MAX, RESOLUTION),
        (1.0e-3, ALPHA_MAX, RESOLUTION),
        (f64::NAN, ALPHA_MAX, RESOLUTION),
        (RESOLVABLE_CURVATURE, 0.0, RESOLUTION),
        (RESOLVABLE_CURVATURE, ALPHA_MAX, 0.0),
        (RESOLVABLE_CURVATURE, ALPHA_MAX, f64::INFINITY),
    ] {
        assert_eq!(
            negative_curvature_claim(lambda_min, alpha_max, resolution),
            None,
            "({lambda_min}, {alpha_max}, {resolution}) carries no claim"
        );
    }
}

/// The issue's instance: the claim is unresolvable, so the adjudication says so before any
/// trial. Before the repair it probed `α = 1` in both signs, both evaluations failed, and it
/// declined, which left the refusal standing on the matrix's word.
#[test]
fn an_unresolvable_claim_is_decided_with_no_trial_evaluated_3036() {
    let mut objective = RefusingCriterion::default();
    let verdict = adjudicate(&mut objective, UNRESOLVABLE_CURVATURE);
    let SaddleAdjudication::Unresolvable {
        lambda_min,
        predicted_at_largest,
        objective_resolution,
    } = verdict
    else {
        panic!(
            "a claim whose largest step predicts 6.47e-7 against a resolution of 1.23e-5 is \
             unresolvable whatever its probes do: {verdict:?} after {} refused trial(s)",
            objective.refused_trials
        );
    };
    // λ_min comes back from the symmetric eigensolve, within its normwise backward error
    // `p·ε·‖H‖₂` (Weyl) of the claimed entry; here `p = 2` and `‖H‖₂ = 1`.
    assert!(
        (lambda_min - UNRESOLVABLE_CURVATURE).abs() <= 2.0 * f64::EPSILON,
        "λ_min {lambda_min:e} must be the claimed {UNRESOLVABLE_CURVATURE:e}"
    );
    assert_eq!(objective_resolution, RESOLUTION);
    assert!(predicted_at_largest < objective_resolution);
    assert_eq!(
        objective.evaluations, 0,
        "resolvability is decided before any trial, so nothing may be evaluated"
    );
}

/// The rule decides only resolvability. A resolvable claim whose trials cannot be evaluated
/// has not been adjudicated, and it still declines.
#[test]
fn a_resolvable_claim_whose_trials_all_fail_still_declines_3036() {
    let mut objective = RefusingCriterion::default();
    let verdict = adjudicate(&mut objective, RESOLVABLE_CURVATURE);
    let refused = objective.refused_trials;
    assert!(refused > 0, "the resolvable claim must have been probed");
    let SaddleAdjudication::Declined(reason) = verdict else {
        panic!("a resolvable claim with no evaluable trial must decline: {verdict:?}");
    };
    assert!(
        reason.contains(&format!("eval_failed={refused}, non_finite=0")),
        "the declined exit must count every refused trial: {reason}"
    );
}

/// The certificate records an unresolvable claim as admissible evidence that publishes as
/// `null`, meets a caller's measured-PSD requirement as a contradicted claim does, and never
/// presents as a measured verdict.
#[test]
fn an_unresolvable_certificate_is_admissible_and_publishes_null_3036() {
    let certificate = crate::model_types::OuterCriterionCertificate {
        stationarity: crate::model_types::OuterStationarityCertificate::AnalyticGradient {
            grad_norm: 6.902e-5,
            projected_grad_norm: 6.902e-5,
            bound: 1.229e-3,
            rung: crate::model_types::CertifiedRung {
                label: "solver-band".to_string(),
                derived_standard: false,
            },
        },
        curvature: CurvatureEvidence::CriterionUnresolvable,
        lambdas_railed: Vec::new(),
        railed_facts: Vec::new(),
        newton_polish: None,
        curvature_floor: None,
    };
    assert_eq!(certificate.hessian_psd(), None);
    assert_eq!(
        certificate.curvature_verdict(),
        crate::model_types::CurvatureAdmissibility::Admissible
    );
    assert!(certificate.curvature_not_refused());
    assert!(certificate.certifies(), "{:?}", certificate.refusal());
    assert!(certificate_meets_curvature_requirement(
        &certificate,
        true,
        CertificationFidelity::Mint,
    ));
    assert!(CurvatureEvidence::CriterionUnresolvable.withdrawn_by_criterion());
    assert!(CurvatureEvidence::CriterionContradicted.withdrawn_by_criterion());
    assert!(!CurvatureEvidence::Measured { psd: false }.withdrawn_by_criterion());
    let published = serde_json::to_value(&certificate).expect("certificate serializes");
    assert_eq!(
        published.get("hessian_psd"),
        Some(&serde_json::Value::Null),
        "the payload's `hessian_psd` domain is unchanged: {published}"
    );
}
