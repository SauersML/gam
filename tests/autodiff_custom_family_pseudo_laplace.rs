use gam::families::custom_family::{
    CustomFamilyBlockPsiDerivative, ExactNewtonOuterObjective, evaluate_custom_family_joint_hyper,
};
use gam::{
    BlockWorkingSet, BlockwiseFitOptions, CustomFamily, FamilyEvaluation, ParameterBlockSpec,
    ParameterBlockState,
};
use ndarray::{Array1, Array2, array};
use num_dual::{DualNum, first_derivative};

mod common;

#[derive(Clone)]
struct ScalarPseudoLaplaceRhoFamily {
    target: f64,
}

impl CustomFamily for ScalarPseudoLaplaceRhoFamily {
    fn evaluate(&self, block_states: &[ParameterBlockState]) -> Result<FamilyEvaluation, String> {
        let beta = block_states
            .first()
            .ok_or_else(|| "missing block 0".to_string())?
            .beta[0];
        let resid = beta - self.target;
        Ok(FamilyEvaluation {
            log_likelihood: -resid * resid,
            block_working_sets: vec![BlockWorkingSet::ExactNewton {
                gradient: array![-2.0 * resid],
                hessian: array![[2.0]],
            }],
        })
    }

    fn exact_newton_outer_objective(&self) -> ExactNewtonOuterObjective {
        ExactNewtonOuterObjective::PseudoLaplace
    }

    fn exact_newton_joint_hessian(
        &self,
        _block_states: &[ParameterBlockState],
    ) -> Result<Option<Array2<f64>>, String> {
        Ok(Some(array![[2.0]]))
    }

    fn exact_newton_hessian_directional_derivative(
        &self,
        _block_states: &[ParameterBlockState],
        _block_idx: usize,
        _d_beta: &Array1<f64>,
    ) -> Result<Option<Array2<f64>>, String> {
        Ok(Some(array![[0.0]]))
    }

    fn exact_newton_joint_hessian_directional_derivative(
        &self,
        _block_states: &[ParameterBlockState],
        _d_beta_flat: &Array1<f64>,
    ) -> Result<Option<Array2<f64>>, String> {
        Ok(Some(array![[0.0]]))
    }
}

#[derive(Clone)]
struct ScalarPseudoLaplacePsiFamily {
    psi: f64,
}

impl CustomFamily for ScalarPseudoLaplacePsiFamily {
    fn evaluate(&self, block_states: &[ParameterBlockState]) -> Result<FamilyEvaluation, String> {
        let beta = block_states
            .first()
            .ok_or_else(|| "missing block 0".to_string())?
            .beta[0];
        let resid = beta - self.psi;
        Ok(FamilyEvaluation {
            log_likelihood: -(resid * resid + 0.25 * self.psi * self.psi),
            block_working_sets: vec![BlockWorkingSet::ExactNewton {
                gradient: array![-2.0 * resid],
                hessian: array![[2.0]],
            }],
        })
    }

    fn exact_newton_outer_objective(&self) -> ExactNewtonOuterObjective {
        ExactNewtonOuterObjective::PseudoLaplace
    }

    fn exact_newton_hessian_directional_derivative(
        &self,
        _block_states: &[ParameterBlockState],
        _block_idx: usize,
        _d_beta: &Array1<f64>,
    ) -> Result<Option<Array2<f64>>, String> {
        Ok(Some(array![[0.0]]))
    }

    fn exact_newton_block_psi_gradient(
        &self,
        _block_states: &[ParameterBlockState],
        _block_idx: usize,
        _ctx: gam::families::custom_family::ExactNewtonPsiGradientContext<'_>,
    ) -> Result<Option<f64>, String> {
        Ok(Some(0.5 * self.psi))
    }
}

fn scalar_pseudo_laplace_rho_objective_numdual<D: DualNum<f64> + Copy>(rho: D, target: f64) -> D {
    let lambda = rho.exp();
    let beta_hat = D::from(2.0 * target) / (D::from(2.0) + lambda);
    let resid = beta_hat - D::from(target);
    resid * resid
        + D::from(0.5) * lambda * beta_hat * beta_hat
        + D::from(0.5) * (D::from(2.0) + lambda).ln()
}

fn scalar_pseudo_laplace_psi_objective_numdual<D: DualNum<f64> + Copy>(psi: D) -> D {
    D::from(0.25) * psi * psi + D::from(0.5) * D::from(2.0).ln()
}

fn scalar_pseudo_laplace_rho_objective_f64(rho: f64, target: f64) -> f64 {
    let lambda = rho.exp();
    let beta_hat = 2.0 * target / (2.0 + lambda);
    let resid = beta_hat - target;
    resid * resid + 0.5 * lambda * beta_hat * beta_hat + 0.5 * (2.0 + lambda).ln()
}

fn scalar_pseudo_laplace_psi_objective_f64(psi: f64) -> f64 {
    0.25 * psi * psi + 0.5 * 2.0_f64.ln()
}

#[test]
fn exact_newton_pseudo_laplace_rho_gradient_matches_num_dual_band() {
    let spec = ParameterBlockSpec {
        name: "rho_block".to_string(),
        design: gam::matrix::DesignMatrix::Dense(array![[1.0]]),
        offset: array![0.0],
        penalties: vec![Array2::eye(1)],
        initial_log_lambdas: array![0.0],
        initial_beta: Some(array![0.0]),
    };
    let derivative_blocks = vec![Vec::<CustomFamilyBlockPsiDerivative>::new()];
    let options = BlockwiseFitOptions {
        use_reml_objective: true,
        compute_covariance: false,
        ridge_floor: 1e-12,
        ..BlockwiseFitOptions::default()
    };
    let family = ScalarPseudoLaplaceRhoFamily { target: 1.4 };
    let rho_points = [-1.7, -0.8, 0.0, 0.9, 1.8];

    for rho in rho_points {
        let result = evaluate_custom_family_joint_hyper(
            &family,
            std::slice::from_ref(&spec),
            &options,
            &array![rho],
            &derivative_blocks,
            None,
            false,
        )
        .expect("pseudo-laplace rho hyper eval");
        let (value_nd, grad_nd) = first_derivative(
            |x| scalar_pseudo_laplace_rho_objective_numdual(x, family.target),
            rho,
        );
        let value_f64 = scalar_pseudo_laplace_rho_objective_f64(rho, family.target);
        let h = 1e-6;
        let grad_fd = (scalar_pseudo_laplace_rho_objective_f64(rho + h, family.target)
            - scalar_pseudo_laplace_rho_objective_f64(rho - h, family.target))
            / (2.0 * h);
        assert!(
            (result.objective - value_nd).abs() < 5e-12,
            "pseudo_laplace_rho x={rho:.6} objective mismatch: analytic={} num_dual={} closed_form={}",
            result.objective,
            value_nd,
            value_f64
        );
        assert_manual_ad_band!(
            "pseudo_laplace_rho",
            rho,
            "gradient",
            result.gradient[0],
            "num_dual" => grad_nd,
            "fd" => grad_fd
        );
    }
}

#[test]
fn exact_newton_pseudo_laplace_psi_gradient_matches_num_dual_band() {
    let spec = ParameterBlockSpec {
        name: "psi_block".to_string(),
        design: gam::matrix::DesignMatrix::Dense(array![[1.0]]),
        offset: array![0.0],
        penalties: vec![],
        initial_log_lambdas: Array1::zeros(0),
        initial_beta: Some(array![0.0]),
    };
    let deriv = CustomFamilyBlockPsiDerivative {
        penalty_index: 0,
        x_psi: Array2::zeros((1, 1)),
        s_psi: Array2::zeros((1, 1)),
        s_psi_components: Some(Vec::new()),
    };
    let derivative_blocks = vec![vec![deriv]];
    let options = BlockwiseFitOptions {
        use_reml_objective: true,
        compute_covariance: false,
        ridge_floor: 1e-12,
        ..BlockwiseFitOptions::default()
    };
    let psi_points = [-2.0, -0.7, 0.0, 0.5, 1.6];

    for psi in psi_points {
        let family = ScalarPseudoLaplacePsiFamily { psi };
        let result = evaluate_custom_family_joint_hyper(
            &family,
            std::slice::from_ref(&spec),
            &options,
            &Array1::zeros(0),
            &derivative_blocks,
            None,
            false,
        )
        .expect("pseudo-laplace psi hyper eval");
        let (value_nd, grad_nd) =
            first_derivative(scalar_pseudo_laplace_psi_objective_numdual, psi);
        let value_f64 = scalar_pseudo_laplace_psi_objective_f64(psi);
        let h = 1e-6;
        let grad_fd = (scalar_pseudo_laplace_psi_objective_f64(psi + h)
            - scalar_pseudo_laplace_psi_objective_f64(psi - h))
            / (2.0 * h);
        assert!(
            (result.objective - value_nd).abs() < 5e-12,
            "pseudo_laplace_psi x={psi:.6} objective mismatch: analytic={} num_dual={} closed_form={}",
            result.objective,
            value_nd,
            value_f64
        );
        assert_manual_ad_band!(
            "pseudo_laplace_psi",
            psi,
            "gradient",
            result.gradient[0],
            "num_dual" => grad_nd,
            "fd" => grad_fd
        );
    }
}
