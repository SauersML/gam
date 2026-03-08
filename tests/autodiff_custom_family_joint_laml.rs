use ad_trait::AD;
use ad_trait::differentiable_function::{DifferentiableFunctionTrait, ForwardAD};
use ad_trait::forward_ad::adfn::adfn;
use ad_trait::function_engine::FunctionEngine;
use autodiff::{F1, Float, diff};
use gam::families::custom_family::{
    CustomFamilyBlockPsiDerivative, evaluate_custom_family_joint_hyper,
};
use gam::matrix::{DesignMatrix, SymmetricMatrix};
use gam::{
    BlockWorkingSet, BlockwiseFitOptions, CustomFamily, FamilyEvaluation, ParameterBlockSpec,
    ParameterBlockState,
};
use ndarray::{Array1, Array2, array};
use num_dual::{DualNum, first_derivative};
use std::marker::PhantomData;

mod common;

#[derive(Clone)]
struct CoupledQuarticExactFamily {
    center: f64,
    quartic: f64,
    beta2_ridge: f64,
}

impl CoupledQuarticExactFamily {
    fn eta(&self, block_states: &[ParameterBlockState]) -> f64 {
        block_states[0].beta[0] + block_states[1].beta[0]
    }

    fn common(&self, eta: f64) -> f64 {
        (eta - self.center) + self.quartic * eta.powi(3)
    }

    fn curvature(&self, eta: f64) -> f64 {
        1.0 + 3.0 * self.quartic * eta * eta
    }
}

impl CustomFamily for CoupledQuarticExactFamily {
    fn evaluate(&self, block_states: &[ParameterBlockState]) -> Result<FamilyEvaluation, String> {
        let beta1 = block_states
            .first()
            .ok_or_else(|| "missing block 0".to_string())?
            .beta[0];
        let beta2 = block_states
            .get(1)
            .ok_or_else(|| "missing block 1".to_string())?
            .beta[0];
        let eta = beta1 + beta2;
        let common = self.common(eta);
        let curvature = self.curvature(eta);
        let nll = 0.5 * (eta - self.center).powi(2)
            + 0.25 * self.quartic * eta.powi(4)
            + 0.5 * self.beta2_ridge * beta2.powi(2);

        Ok(FamilyEvaluation {
            log_likelihood: -nll,
            block_working_sets: vec![
                BlockWorkingSet::ExactNewton {
                    gradient: array![-common],
                    hessian: SymmetricMatrix::Dense(array![[curvature]]),
                },
                BlockWorkingSet::ExactNewton {
                    gradient: array![-(common + self.beta2_ridge * beta2)],
                    hessian: SymmetricMatrix::Dense(array![[curvature + self.beta2_ridge]]),
                },
            ],
        })
    }

    fn exact_newton_joint_hessian(
        &self,
        block_states: &[ParameterBlockState],
    ) -> Result<Option<Array2<f64>>, String> {
        let eta = self.eta(block_states);
        let curvature = self.curvature(eta);
        Ok(Some(array![
            [curvature, curvature],
            [curvature, curvature + self.beta2_ridge]
        ]))
    }

    fn exact_newton_hessian_directional_derivative(
        &self,
        block_states: &[ParameterBlockState],
        block_idx: usize,
        d_beta: &Array1<f64>,
    ) -> Result<Option<Array2<f64>>, String> {
        let eta = self.eta(block_states);
        let factor = 6.0 * self.quartic * eta * d_beta[0];
        let value = match block_idx {
            0 | 1 => factor,
            _ => return Ok(None),
        };
        Ok(Some(array![[value]]))
    }

    fn exact_newton_joint_hessian_directional_derivative(
        &self,
        block_states: &[ParameterBlockState],
        d_beta_flat: &Array1<f64>,
    ) -> Result<Option<Array2<f64>>, String> {
        let eta = self.eta(block_states);
        let factor = 6.0 * self.quartic * eta * (d_beta_flat[0] + d_beta_flat[1]);
        Ok(Some(array![[factor, factor], [factor, factor]]))
    }
}

fn solve_eta_numdual<D: DualNum<f64> + Copy>(
    rho: D,
    center: f64,
    quartic: f64,
    beta2_ridge: f64,
) -> D {
    let lambda = rho.exp();
    let d = D::from(beta2_ridge);
    let c = D::from(center);
    let a = D::from(quartic);
    let coupling = lambda * d / (d + lambda);
    let mut eta = c / (D::one() + coupling);
    for _ in 0..20 {
        let f = eta - c + a * eta * eta * eta + coupling * eta;
        let fp = D::one() + D::from(3.0) * a * eta * eta + coupling;
        eta = eta - f / fp;
    }
    eta
}

fn coupled_quartic_objective_numdual<D: DualNum<f64> + Copy>(
    rho: D,
    center: f64,
    quartic: f64,
    beta2_ridge: f64,
) -> D {
    let lambda = rho.exp();
    let d = D::from(beta2_ridge);
    let eta = solve_eta_numdual(rho, center, quartic, beta2_ridge);
    let beta1 = eta * d / (d + lambda);
    let beta2 = eta * lambda / (d + lambda);
    let nll = D::from(0.5) * (eta - D::from(center)) * (eta - D::from(center))
        + D::from(0.25) * D::from(quartic) * eta * eta * eta * eta
        + D::from(0.5) * d * beta2 * beta2;
    let curvature = D::one() + D::from(3.0) * D::from(quartic) * eta * eta;
    let det = (curvature + lambda) * (curvature + d) - curvature * curvature;
    nll + D::from(0.5) * lambda * beta1 * beta1 + D::from(0.5) * det.ln() - D::from(0.5) * rho
}

fn coupled_quartic_objective_f64(rho: f64, center: f64, quartic: f64, beta2_ridge: f64) -> f64 {
    let lambda = rho.exp();
    let d = beta2_ridge;
    let eta = solve_eta_numdual(rho, center, quartic, beta2_ridge);
    let beta1 = eta * d / (d + lambda);
    let beta2 = eta * lambda / (d + lambda);
    let nll = 0.5 * (eta - center).powi(2) + 0.25 * quartic * eta.powi(4) + 0.5 * d * beta2.powi(2);
    let curvature = 1.0 + 3.0 * quartic * eta * eta;
    let det = (curvature + lambda) * (curvature + d) - curvature * curvature;
    nll + 0.5 * lambda * beta1 * beta1 + 0.5 * det.ln() - 0.5 * rho
}

fn coupled_quartic_objective_f1(rho: F1, center: f64, quartic: f64, beta2_ridge: f64) -> F1 {
    let lambda = rho.exp();
    let d = F1::cst(beta2_ridge);
    let c = F1::cst(center);
    let a = F1::cst(quartic);
    let coupling = lambda * d / (d + lambda);
    let mut eta = c / (F1::cst(1.0) + coupling);
    for _ in 0..20 {
        let f = eta - c + a * eta * eta * eta + coupling * eta;
        let fp = F1::cst(1.0) + F1::cst(3.0) * a * eta * eta + coupling;
        eta = eta - f / fp;
    }
    let beta1 = eta * d / (d + lambda);
    let beta2 = eta * lambda / (d + lambda);
    let nll = F1::cst(0.5) * (eta - c) * (eta - c)
        + F1::cst(0.25) * a * eta * eta * eta * eta
        + F1::cst(0.5) * d * beta2 * beta2;
    let curvature = F1::cst(1.0) + F1::cst(3.0) * a * eta * eta;
    let det = (curvature + lambda) * (curvature + d) - curvature * curvature;
    nll + F1::cst(0.5) * lambda * beta1 * beta1 + F1::cst(0.5) * det.ln() - F1::cst(0.5) * rho
}

#[derive(Clone)]
struct CoupledQuarticObjectiveFn<T: AD> {
    center: f64,
    quartic: f64,
    beta2_ridge: f64,
    _marker: PhantomData<T>,
}

impl<T: AD> CoupledQuarticObjectiveFn<T> {
    fn new(center: f64, quartic: f64, beta2_ridge: f64) -> Self {
        Self {
            center,
            quartic,
            beta2_ridge,
            _marker: PhantomData,
        }
    }

    fn to_other_ad_type<T2: AD>(&self) -> CoupledQuarticObjectiveFn<T2> {
        CoupledQuarticObjectiveFn::new(self.center, self.quartic, self.beta2_ridge)
    }
}

impl<T: AD> DifferentiableFunctionTrait<T> for CoupledQuarticObjectiveFn<T> {
    const NAME: &'static str = "CoupledQuarticObjectiveFn";

    fn call(&self, inputs: &[T], _freeze: bool) -> Vec<T> {
        let rho = inputs[0];
        let lambda = rho.exp();
        let d = T::constant(self.beta2_ridge);
        let c = T::constant(self.center);
        let a = T::constant(self.quartic);
        let coupling = lambda * d / (d + lambda);
        let mut eta = c / (T::one() + coupling);
        for _ in 0..20 {
            let f = eta - c + a * eta * eta * eta + coupling * eta;
            let fp = T::one() + T::constant(3.0) * a * eta * eta + coupling;
            eta = eta - f / fp;
        }
        let beta1 = eta * d / (d + lambda);
        let beta2 = eta * lambda / (d + lambda);
        let nll = T::constant(0.5) * (eta - c) * (eta - c)
            + T::constant(0.25) * a * eta * eta * eta * eta
            + T::constant(0.5) * d * beta2 * beta2;
        let curvature = T::one() + T::constant(3.0) * a * eta * eta;
        let det = (curvature + lambda) * (curvature + d) - curvature * curvature;
        vec![
            nll + T::constant(0.5) * lambda * beta1 * beta1 + T::constant(0.5) * det.ln()
                - T::constant(0.5) * rho,
        ]
    }

    fn num_inputs(&self) -> usize {
        1
    }

    fn num_outputs(&self) -> usize {
        1
    }
}

#[test]
fn exact_joint_quadratic_laml_gradient_matches_three_autodiff_engines() {
    let center = 0.7;
    let quartic = 0.18;
    let beta2_ridge = 1.4;
    let family = CoupledQuarticExactFamily {
        center,
        quartic,
        beta2_ridge,
    };
    let specs = vec![
        ParameterBlockSpec {
            name: "shape".to_string(),
            design: DesignMatrix::Dense(array![[1.0]]),
            offset: array![0.0],
            penalties: vec![Array2::eye(1)],
            initial_log_lambdas: array![0.0],
            initial_beta: Some(array![0.0]),
        },
        ParameterBlockSpec {
            name: "aux".to_string(),
            design: DesignMatrix::Dense(array![[1.0]]),
            offset: array![0.0],
            penalties: vec![],
            initial_log_lambdas: Array1::zeros(0),
            initial_beta: Some(array![0.0]),
        },
    ];
    let derivative_blocks = vec![
        Vec::<CustomFamilyBlockPsiDerivative>::new(),
        Vec::<CustomFamilyBlockPsiDerivative>::new(),
    ];
    let options = BlockwiseFitOptions {
        use_reml_objective: true,
        compute_covariance: false,
        ridge_floor: 1e-12,
        ..BlockwiseFitOptions::default()
    };
    let f_std = CoupledQuarticObjectiveFn::<f64>::new(center, quartic, beta2_ridge);
    let f_ad = f_std.to_other_ad_type::<adfn<1>>();
    let engine = FunctionEngine::new(f_std, f_ad, ForwardAD::new());
    let rho_points = [-1.2, -0.4, 0.0, 0.6, 1.1];

    for rho in rho_points {
        let result = evaluate_custom_family_joint_hyper(
            &family,
            &specs,
            &options,
            &array![rho],
            &derivative_blocks,
            None,
            false,
        )
        .expect("exact joint hyper eval");
        let (value_nd, grad_nd) = first_derivative(
            |x| coupled_quartic_objective_numdual(x, center, quartic, beta2_ridge),
            rho,
        );
        let grad_autodiff = diff(
            |x: F1| coupled_quartic_objective_f1(x, center, quartic, beta2_ridge),
            rho,
        );
        let (_value_ad, jac) = engine.derivative(&[rho]);
        let grad_fd = (coupled_quartic_objective_f64(rho + 1e-6, center, quartic, beta2_ridge)
            - coupled_quartic_objective_f64(rho - 1e-6, center, quartic, beta2_ridge))
            / (2.0 * 1e-6);

        assert!(
            (result.objective - value_nd).abs() < 5e-10,
            "exact_joint_laml objective mismatch at rho={rho}: analytic={} num_dual={} closed_form={}",
            result.objective,
            value_nd,
            coupled_quartic_objective_f64(rho, center, quartic, beta2_ridge)
        );
        assert_manual_ad_band!(
            "exact_joint_laml",
            rho,
            "gradient",
            result.gradient[0],
            "num_dual" => grad_nd,
            "ad_trait" => jac[(0, 0)],
            "fd" => grad_fd,
            "autodiff" => grad_autodiff
        );
    }
}
