use gam_custom_family::{
    BlockwiseFitOptions, CustomFamily, CustomFamilyHyperLayout, FamilyEvaluation,
    evaluate_custom_family_joint_hyper,
};
use gam_linalg::matrix::{DesignMatrix, SymmetricMatrix};
use gam_problem::{BlockWorkingSet, EvalMode, ParameterBlockSpec, ParameterBlockState, PenaltyMatrix};
use ndarray::{Array1, Array2, array};

#[derive(Clone)]
struct JointQuartic;

impl CustomFamily for JointQuartic {
    fn evaluate(&self, states: &[ParameterBlockState]) -> Result<FamilyEvaluation, String> {
        let beta = states[0].beta[0];
        Ok(FamilyEvaluation {
            log_likelihood: 13.0 * beta / 6.0 - beta.powi(2) / 2.0 - beta.powi(4) / 24.0,
            blockworking_sets: vec![BlockWorkingSet::ExactNewton {
                gradient: array![13.0 / 6.0 - beta - beta.powi(3) / 6.0],
                hessian: SymmetricMatrix::Dense(array![[1.0 + beta.powi(2) / 2.0]]),
            }],
        })
    }

    fn has_explicit_joint_hessian(&self) -> bool {
        true
    }

    fn exact_newton_joint_hessian_beta_dependent(&self) -> bool {
        true
    }

    fn exact_newton_joint_hessian(
        &self,
        states: &[ParameterBlockState],
    ) -> Result<Option<Array2<f64>>, String> {
        Ok(Some(array![[1.0 + states[0].beta[0].powi(2) / 2.0]]))
    }

    fn exact_newton_joint_hessian_directional_derivative(
        &self,
        states: &[ParameterBlockState],
        direction: &Array1<f64>,
    ) -> Result<Option<Array2<f64>>, String> {
        Ok(Some(array![[states[0].beta[0] * direction[0]]]))
    }
}

#[test]
fn laml_value_and_gradient_use_the_same_nonlinear_mode() {
    // J(beta, rho) = beta²/2 + beta⁴/24 - 13 beta/6 + exp(rho) beta²/2.
    // At rho=0 its unique mode is beta=1, H=5/2 and beta_rho=-2/5.
    // Thus V=J+log(H)/2-rho/2=-9/8+log(5/2)/2 and V_rho=3/25.
    // A coarse coefficient solve introduces first-order error through log(H).
    let family = JointQuartic;
    let spec = ParameterBlockSpec {
        name: "quartic".to_owned(),
        design: DesignMatrix::Dense(gam_linalg::matrix::DenseDesignMatrix::from(array![[1.0]])),
        offset: array![0.0],
        penalties: vec![PenaltyMatrix::Dense(array![[1.0]])],
        nullspace_dims: vec![],
        initial_log_lambdas: array![0.0],
        initial_beta: Some(array![1.4]),
        gauge_priority: 100,
        jacobian_callback: None,
        stacked_design: None,
        stacked_offset: None,
    };
    let options = BlockwiseFitOptions {
        inner_tol: 1e-2,
        use_remlobjective: true,
        compute_covariance: false,
        ..BlockwiseFitOptions::default()
    };
    let layout = CustomFamilyHyperLayout::new(vec![vec![]], vec![], Array1::zeros(0))
        .expect("empty family hyperparameter layout");
    let evaluate = |mode| {
        evaluate_custom_family_joint_hyper(
            &family,
            std::slice::from_ref(&spec),
            &options,
            &array![0.0],
            &layout,
            None,
            mode,
        ).expect("strictly convex quartic has a finite Laplace mode")
    };
    let value = evaluate(EvalMode::ValueOnly);
    let derivative = evaluate(EvalMode::ValueAndGradient);
    let expected = -9.0 / 8.0 + 0.5 * 2.5_f64.ln();
    for result in [&value, &derivative] {
        assert!(result.inner_converged);
        assert!((result.objective - expected).abs() < 1e-9,
            "Laplace value {} differs from analytic {expected}", result.objective);
    }
    assert!((value.objective - derivative.objective).abs() < 1e-12);
    assert!((derivative.gradient[0] - 3.0 / 25.0).abs() < 1e-9,
        "Laplace gradient {} differs from analytic 3/25", derivative.gradient[0]);
}
