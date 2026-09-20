//! #3318: P-IRLS certified a symmetric saddle of the inner objective.
//!
//! On the symmetric separated Firth fixtures the iterate reaches a point where
//! the gradient vanishes by symmetry but the objective curvature has a negative
//! eigenvalue. A first-order certificate accepts it and the Laplace layer then
//! refuses the indefinite Hessian. The second-order certificate must leave the
//! saddle along its negative-curvature direction and converge to the minimum.

use super::*;
use ndarray::array;

/// `F(β) = 1 + ½β₁² − ½β₂² + ¼β₂⁴`: a saddle at the origin, minima at
/// `(0, ±1)` with `F = ¾`. `β₂ = 0` is invariant under Newton, since `∂F/∂β₂`
/// vanishes on it, so a first-order method started there stays there.
struct SymmetricSaddleModel;

impl SymmetricSaddleModel {
    fn state(beta: &Coefficients, curvature: HessianCurvatureKind) -> WorkingState {
        let (b1, b2) = (beta.as_ref()[0], beta.as_ref()[1]);
        let objective = 1.0 + 0.5 * b1 * b1 - 0.5 * b2 * b2 + 0.25 * b2.powi(4);
        WorkingState {
            eta: LinearPredictor::new(array![b1, b2]),
            gradient: array![b1, -b2 + b2.powi(3)],
            hessian: gam_linalg::matrix::SymmetricMatrix::Dense(array![
                [1.0, 0.0],
                [0.0, -1.0 + 3.0 * b2 * b2]
            ]),
            log_likelihood: -objective,
            deviance: 2.0 * objective,
            deviance_magnitude: 2.0 * objective,
            penalty_term: 0.0,
            firth: FirthDiagnostics::Inactive,
            hessian_curvature: curvature,
            gradient_natural_scale: 1.0,
        }
    }
}

impl WorkingModel for SymmetricSaddleModel {
    fn update(&mut self, beta: &Coefficients) -> Result<WorkingState, EstimationError> {
        self.update_with_curvature(beta, HessianCurvatureKind::Observed)
    }

    fn update_with_curvature(
        &mut self,
        beta: &Coefficients,
        curvature: HessianCurvatureKind,
    ) -> Result<WorkingState, EstimationError> {
        Ok(Self::state(beta, curvature))
    }

    fn supports_observed_information_curvature(&self) -> bool {
        true
    }
}

#[test]
fn pirls_leaves_an_exact_symmetric_saddle_3318() {
    let options = WorkingModelPirlsOptions {
        max_iterations: 200,
        convergence_tolerance: 1e-8,
        adaptive_kkt_tolerance: None,
        max_step_halving: 30,
        firth_bias_reduction: false,
        coefficient_lower_bounds: None,
        linear_constraints: None,
        initial_lm_lambda: None,
    };
    let result = runworking_model_pirls(
        &mut SymmetricSaddleModel,
        Coefficients::new(array![1.0, 0.0]),
        &options,
        None,
    )
    .expect("the saddle model is smooth and finite everywhere");

    assert!(
        result.status.is_converged(),
        "#3318: status {:?} at beta {:?}",
        result.status,
        result.beta.as_ref()
    );
    let (b1, b2) = (result.beta.as_ref()[0], result.beta.as_ref()[1]);
    // The escape direction is ±e₂; the sign tie is broken toward the positive
    // largest component, so the minimum reached is (0, +1).
    assert!(
        b1.abs() <= 1e-6 && (b2 - 1.0).abs() <= 1e-6,
        "#3318: converged to ({b1:e}, {b2:e}), not the minimum (0, 1)"
    );
    let objective = 1.0 + 0.5 * b1 * b1 - 0.5 * b2 * b2 + 0.25 * b2.powi(4);
    assert!(
        (objective - 0.75).abs() <= 1e-10,
        "#3318: objective {objective} at the certified point, minimum 0.75"
    );
}

#[test]
fn pirls_certifies_a_positive_definite_minimum_unchanged_3318() {
    // Started on the far side of the minimum the curvature is positive
    // throughout, so the second-order certificate must not move the iterate.
    let options = WorkingModelPirlsOptions {
        max_iterations: 200,
        convergence_tolerance: 1e-8,
        adaptive_kkt_tolerance: None,
        max_step_halving: 30,
        firth_bias_reduction: false,
        coefficient_lower_bounds: None,
        linear_constraints: None,
        initial_lm_lambda: None,
    };
    let result = runworking_model_pirls(
        &mut SymmetricSaddleModel,
        Coefficients::new(array![0.5, -1.5]),
        &options,
        None,
    )
    .expect("the saddle model is smooth and finite everywhere");
    assert!(result.status.is_converged(), "status {:?}", result.status);
    let (b1, b2) = (result.beta.as_ref()[0], result.beta.as_ref()[1]);
    assert!(
        b1.abs() <= 1e-6 && (b2 + 1.0).abs() <= 1e-6,
        "converged to ({b1:e}, {b2:e}), not the nearby minimum (0, -1)"
    );
}
