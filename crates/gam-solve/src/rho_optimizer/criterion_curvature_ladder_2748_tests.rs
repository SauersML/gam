//! #2748/#2612 — the saddle adjudication decides from criterion VALUES along
//! the disputed eigenvector, and never forms a curvature from them.
//!
//! `adjudicate_negative_curvature` steps the criterion along the most negative
//! eigenvector of the analytic outer Hessian and mints an escape reseed only
//! when some feasible trial lowers the objective by more than the criterion's
//! resolution. It used to go further and fit a symmetric second-difference
//! ladder to those evaluations, publishing the fitted curvature as a measured
//! Hessian error and minting reseeds from it. A second difference of criterion
//! values is a finite difference on the fit-math path, which production code
//! may not contain, and that measurement is gone. These fixtures pin what
//! remains: a genuine saddle still descends through the value search alone,
//! and a descent the criterion cannot resolve is withdrawn rather than minted.
//!
//! Every fixture plants the criterion FIRST and derives the Hessian the
//! adjudication is handed from it, so each verdict is about a known function
//! rather than about a recorded run.

use super::*;
use ndarray::{Array1, Array2, array};

/// A criterion that is exactly `V(θ) = V₀ + g·θ + ½θᵀCθ + (M₄/24)(v·θ)⁴`, and
/// an analytic Hessian the adjudication is handed that may DISAGREE with `C`.
///
/// The quartic is along the probed direction only, which bounds the depth of
/// the well a negative curvature along `v` offers: the criterion turns back up
/// once `(M₄/24)α⁴` overtakes `½|c|α²`.
struct PlantedCriterion {
    baseline: f64,
    gradient: Array1<f64>,
    criterion_curvature: Array2<f64>,
    quartic_direction: Array1<f64>,
    fourth_derivative: f64,
    /// Amplitude of a deterministic, high-frequency, NOT-odd term standing in
    /// for the criterion's own evaluation error. Zero for the noiseless
    /// fixtures.
    evaluation_error: f64,
    evaluations: std::sync::Arc<std::sync::Mutex<usize>>,
}

impl PlantedCriterion {
    fn value(&self, theta: &Array1<f64>) -> f64 {
        let projection = self.quartic_direction.dot(theta);
        // The error term is a deterministic function of theta -- same theta,
        // same value, every lane, every host, no RNG -- with a phase offset so
        // it is neither even nor odd in the probed direction.
        let error = self.evaluation_error * (1_048_576.0 * projection + 1.0).sin();
        self.baseline
            + self.gradient.dot(theta)
            + 0.5 * theta.dot(&self.criterion_curvature.dot(theta))
            + self.fourth_derivative * projection.powi(4) / 24.0
            + error
    }
}

impl OuterObjective for PlantedCriterion {
    fn capability(&self) -> OuterCapability {
        OuterCapability {
            gradient: Derivative::Analytic,
            hessian: DeclaredHessianForm::Dense,
            n_params: self.gradient.len(),
            psi_dim: 0,
            fixed_point_available: false,
            barrier_config: None,
            prefer_gradient_only: false,
            disable_fixed_point: true,
        }
    }
    fn eval_cost(&mut self, theta: &Array1<f64>) -> Result<f64, EstimationError> {
        *self
            .evaluations
            .lock()
            .expect("the evaluation counter is not poisoned") += 1;
        Ok(self.value(theta))
    }
    fn eval(&mut self, theta: &Array1<f64>) -> Result<OuterEval, EstimationError> {
        Ok(OuterEval {
            cost: self.value(theta),
            gradient: &self.gradient + &self.criterion_curvature.dot(theta),
            hessian: HessianValue::Unavailable,
            inner_beta_hint: None,
        })
    }
    fn eval_with_order(
        &mut self,
        theta: &Array1<f64>,
        order: OuterEvalOrder,
    ) -> Result<OuterEval, EstimationError> {
        // A planted criterion has closed-form derivatives at every order, so
        // the value-only lane must agree with the derivative-bearing one at the
        // same theta. Honouring the order rather than ignoring it is what makes
        // that agreement a property of the fixture instead of an accident.
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
        // The planted criterion is a pure function of theta: there is no warm
        // state to re-baseline, and saying so is part of the fixture's claim
        // that every evaluation at one theta returns one value.
    }
    fn seed_inner_state(&mut self, beta: &Array1<f64>) -> Result<SeedOutcome, EstimationError> {
        if beta.iter().any(|value| !value.is_finite()) {
            return Err(EstimationError::RemlOptimizationFailed(
                "the planted criterion was offered a non-finite inner seed".to_string(),
            ));
        }
        Ok(SeedOutcome::NoSlot)
    }
}

/// The configuration every test below varies: a 3-coordinate criterion whose
/// curvature along `v = (1,0,-1)/√2` is `criterion_vv`, handed an analytic
/// Hessian claiming `analytic_vv` there and agreeing everywhere else.
struct Planted {
    criterion_vv: f64,
    analytic_vv: f64,
    fourth_derivative: f64,
    gradient_scale: f64,
}

/// Build the (objective, θ̂, g, H_analytic, v) the adjudication is handed.
fn planted(spec: Planted) -> (PlantedCriterion, Array1<f64>, Array1<f64>, Array2<f64>, Array1<f64>) {
    let root_half = 0.5_f64.sqrt();
    let v = array![root_half, 0.0, -root_half];
    let u1 = array![root_half, 0.0, root_half];
    let u2 = array![0.0_f64, 1.0, 0.0];
    // Well-curved off the probed direction, so `v` is unambiguously the
    // minimum-curvature eigenvector of both matrices.
    let (a, b) = (1.0_f64, 0.5_f64);
    let build = |vv: f64| {
        let mut matrix = Array2::<f64>::zeros((3, 3));
        for r in 0..3 {
            for c in 0..3 {
                matrix[[r, c]] = a * u1[r] * u1[c] + b * u2[r] * u2[c] + vv * v[r] * v[c];
            }
        }
        matrix
    };
    // A residual gradient of one sign on every coordinate, so the chain-rule
    // floor `Σ|g_k| v_k²` is unambiguous and equals `|g|` on this `v`.
    let gradient = Array1::from(vec![-spec.gradient_scale; 3]);
    let objective = PlantedCriterion {
        baseline: 2.224446222e3,
        gradient: gradient.clone(),
        criterion_curvature: build(spec.criterion_vv),
        quartic_direction: v.clone(),
        fourth_derivative: spec.fourth_derivative,
        evaluation_error: 0.0,
        evaluations: std::sync::Arc::new(std::sync::Mutex::new(0)),
    };
    (
        objective,
        Array1::<f64>::zeros(3),
        gradient,
        build(spec.analytic_vv),
        v,
    )
}

fn wide_bounds() -> (Array1<f64>, Array1<f64>) {
    (Array1::from(vec![-30.0; 3]), Array1::from(vec![30.0; 3]))
}

/// NEGATIVE CONTROL, and the one that keeps the escape from being a licence:
/// when the descent on offer is below the criterion's resolution, the escape
/// declines and the verdict is withdrawn.
///
/// The claim is an honest `-1e-4` along `v`, but the quartic turns the
/// criterion back up within `α ≈ 0.04`, so the well is only `~3e-8` deep. A
/// planted evaluation error of `3e-8` that is NOT odd in the probed direction
/// lets individual trials dip below the baseline; none dips by more than the
/// declared resolution `1e-7`, so nothing is minted.
#[test]
fn a_curvature_the_criterion_cannot_resolve_is_not_an_escape_2748() {
    let curvature = -1.0e-4_f64;
    let (mut objective, theta, gradient, hessian, _direction) = planted(Planted {
        criterion_vv: curvature,
        analytic_vv: curvature,
        fourth_derivative: 0.48,
        gradient_scale: 1.0e-12,
    });
    objective.evaluation_error = 3.0e-8;
    let baseline = objective.value(&theta);
    let bounds = wide_bounds();
    let declared_resolution = 1.0e-7_f64;

    let verdict = adjudicate_negative_curvature(
        &mut objective,
        &theta,
        &gradient,
        &hessian,
        &[],
        None,
        baseline,
        declared_resolution,
        &bounds,
        "planted #2748 unresolvable-curvature control",
    );
    let SaddleAdjudication::Contradicted {
        probed,
        objective_resolution,
        best_seen_cost,
        ..
    } = verdict
    else {
        panic!(
            "a descent below the criterion's resolution must NOT mint an escape: {verdict:?}"
        );
    };
    assert!(
        probed > 0,
        "the value search must actually have evaluated trials along the claim"
    );
    assert_eq!(
        objective_resolution, declared_resolution,
        "the withdrawal must be judged at the resolution it was handed"
    );
    assert!(
        best_seen_cost >= baseline - declared_resolution,
        "no trial on this fixture may beat the baseline by more than the resolution: best \
         {best_seen_cost:.12e} against baseline {baseline:.12e}"
    );
}

/// A REAL saddle is confirmed, not excused. The criterion genuinely descends
/// along `v`, so the adjudication mints the escape reseed from the value search
/// alone.
#[test]
fn a_genuine_saddle_still_descends_and_never_reaches_the_ladder_2748() {
    let (mut objective, theta, gradient, hessian, _direction) = planted(Planted {
        criterion_vv: -1.6e3,
        analytic_vv: -1.6e3,
        fourth_derivative: 1.0,
        gradient_scale: 1.0e-9,
    });
    let counter = std::sync::Arc::clone(&objective.evaluations);
    let baseline = objective.value(&theta);
    let bounds = wide_bounds();

    let verdict = adjudicate_negative_curvature(
        &mut objective,
        &theta,
        &gradient,
        &hessian,
        &[],
        None,
        baseline,
        1.0e-6,
        &bounds,
        "planted #2748 real-saddle control",
    );
    let SaddleAdjudication::Descended(point) = verdict else {
        panic!("a -1.6e3 curvature at a stationary point descends within one e-fold: {verdict:?}");
    };
    assert!(
        objective.value(&point) < baseline,
        "the minted reseed must be a strictly lower point"
    );
    // The escape found its descent on the first rung of the first sign.
    //
    // The evaluations it pays for are all step search, and every one of them
    // is derived (#2612): the falsification rung, the checkpoint restore, the
    // incumbent re-measured in the expansion's own instrument state, one per
    // doubling out to the box intersection along the ray, and the final restore.
    // On this planted criterion the descent really is unbounded inside the box —
    // `f(α) = baseline + g·α − 800α² + α⁴/24` does not turn back until
    // `α = √19200 ≈ 138.6`, far outside `α_box = 30/√½ ≈ 42.4` — so the
    // expansion runs to the face, which is the correct answer and not a cost to
    // be avoided. The bound below is that arithmetic, not a recorded count.
    let alpha_box = 30.0 / 0.5_f64.sqrt();
    let doublings = alpha_box.log2().ceil() as usize;
    let budget = 1 + 1 + 1 + doublings + 1;
    assert!(
        *counter.lock().expect("counter") <= budget,
        "a confirmed saddle must not pay for evaluations it does not need: \
         {} evaluations against a derived budget of {budget} (1 falsification rung + 1 checkpoint \
         restore + 1 incumbent re-measure + {doublings} doubling(s) to alpha_box={alpha_box:.4} + \
         1 restore)",
        *counter.lock().expect("counter")
    );
    // And the positive statement those evaluations bought: on a descent with no
    // interior minimiser the reseed is the box face, not the falsifier's
    // largest rung (#2612).
    let travelled = point
        .iter()
        .zip(theta.iter())
        .map(|(after, before)| (after - before).abs())
        .fold(0.0_f64, f64::max);
    assert!(
        (travelled - 30.0).abs() < 1e-9,
        "the descent runs to the box, so the reseed must sit ON it: travelled {travelled:.6e} \
         against a bound of 30. A reseed one e-fold out is the falsifiability ladder's rung being \
         reused as a step length: {point:?}"
    );
}
