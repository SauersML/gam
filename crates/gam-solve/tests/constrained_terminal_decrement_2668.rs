use gam_linalg::matrix::SymmetricMatrix;
use gam_problem::{Coefficients, EstimationError, LinearInequalityConstraints, LinearPredictor};
use gam_solve::pirls::{
    FirthDiagnostics, HessianCurvatureKind, PirlsStatus, WorkingModel,
    WorkingModelPirlsOptions, WorkingState, runworking_model_pirls,
};
use ndarray::array;

struct RoundedQuadratic;

impl WorkingModel for RoundedQuadratic {
    fn update(&mut self, beta: &Coefficients) -> Result<WorkingState, EstimationError> {
        // Two equally weighted observations have their mean halfway between
        // adjacent floats. The exact minimizer cannot be represented in f64.
        // x >= 0 is binding, with multiplier 1, and y remains a free direction.
        let a = 100_000.0_f64;
        let b = a.next_up();
        let x = beta.as_ref()[0];
        let y = beta.as_ref()[1];
        let ra = y - a;
        let rb = y - b;
        // This scaling makes the exact squared decrement EPSILON/2:
        // below the existing objective-resolution threshold, while the
        // tangent gradient remains much larger than the KKT tolerance.
        let weight = f64::EPSILON / (b - a).powi(2);
        let objective = 1.0 + x + 0.5 * x * x + 0.5 * weight * (ra * ra + rb * rb);
        Ok(WorkingState {
            eta: LinearPredictor::new(beta.as_ref().clone()),
            gradient: array![1.0 + x, weight * (ra + rb)],
            hessian: SymmetricMatrix::Dense(array![[1.0, 0.0], [0.0, 2.0 * weight]]),
            log_likelihood: -objective,
            deviance: 2.0 * objective,
            penalty_term: 0.0,
            firth: FirthDiagnostics::Inactive,
            ridge_used: 0.0,
            hessian_curvature: HessianCurvatureKind::Fisher,
            gradient_natural_scale: 1.0,
        })
    }
}

#[test]
fn terminal_face_decrement_certifies_a_representable_quadratic_minimum() {
    gam_runtime::test_support::install_diagnostic_logger();
    for (name, lower_bounds, linear_constraints, expected_x) in [
        ("lower bounds", Some(array![0.0, f64::NEG_INFINITY]), None, 0.0),
        (
            "linear rows",
            None,
            Some(LinearInequalityConstraints {
                a: array![[1.0, 0.0]],
                b: array![0.0],
            }),
            0.0,
        ),
        (
            "unbounded box",
            Some(array![f64::NEG_INFINITY, f64::NEG_INFINITY]),
            None,
            -1.0,
        ),
    ] {
        let options = WorkingModelPirlsOptions {
            // The first damped step makes resolvable progress. Undamped terminal
            // refinement then reaches the nearest float; the in-loop plateau
            // branch has not evaluated that final point.
            max_iterations: 1,
            convergence_tolerance: 1e-11,
            adaptive_kkt_tolerance: None,
            max_step_halving: 4,
            min_step_size: 0.0,
            firth_bias_reduction: false,
            coefficient_lower_bounds: lower_bounds,
            linear_constraints,
            initial_lm_lambda: None,
            arrow_schur: None,
        };
        let result = runworking_model_pirls(
            &mut RoundedQuadratic,
            Coefficients::new(array![0.0, 100_001.0]),
            &options,
            None,
        )
        .expect("a finite convex quadratic has a constrained minimum");
        assert_eq!(
            result.status,
            PirlsStatus::Converged,
            "{name}: the tangent Newton decrement certifies the nearest representable minimum"
        );
        assert_eq!(result.beta.as_ref()[0], expected_x);
        let y = result.beta.as_ref()[1];
        assert!(
            y == 100_000.0 || y == 100_000.0_f64.next_up(),
            "the reported free coefficient must minimize the quadratic among representable values"
        );
    }
}
