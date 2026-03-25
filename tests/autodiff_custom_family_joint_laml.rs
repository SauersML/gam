use ad_trait::AD;
use ad_trait::differentiable_function::{DifferentiableFunctionTrait, ForwardAD};
use ad_trait::forward_ad::adfn::adfn;
use ad_trait::function_engine::FunctionEngine;
use autodiff::{F1, Float, diff};
use gam::custom_family::{
    BlockWorkingSet, BlockwiseFitOptions, CustomFamily, FamilyEvaluation, ParameterBlockSpec,
    ParameterBlockState, PenaltyMatrix,
};
use gam::families::custom_family::{
    CustomFamilyBlockPsiDerivative, evaluate_custom_family_joint_hyper,
};
use gam::matrix::{DesignMatrix, SymmetricMatrix};
use gam::pirls::LinearInequalityConstraints;
use ndarray::{Array1, Array2, array};
use num_dual::{DualNum, first_derivative};
use std::marker::PhantomData;

mod common;

#[derive(Clone)]
struct CoupledQuarticExactFamily {
    center: f64,
    quartic: f64,
    beta2ridge: f64,
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
            + 0.5 * self.beta2ridge * beta2.powi(2);

        Ok(FamilyEvaluation {
            log_likelihood: -nll,
            blockworking_sets: vec![
                BlockWorkingSet::ExactNewton {
                    gradient: array![-common],
                    hessian: SymmetricMatrix::Dense(array![[curvature]]),
                },
                BlockWorkingSet::ExactNewton {
                    gradient: array![-(common + self.beta2ridge * beta2)],
                    hessian: SymmetricMatrix::Dense(array![[curvature + self.beta2ridge]]),
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
            [curvature, curvature + self.beta2ridge]
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
    beta2ridge: f64,
) -> D {
    let lambda = rho.exp();
    let d = D::from(beta2ridge);
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

fn coupled_quarticobjective_numdual<D: DualNum<f64> + Copy>(
    rho: D,
    center: f64,
    quartic: f64,
    beta2ridge: f64,
) -> D {
    let lambda = rho.exp();
    let d = D::from(beta2ridge);
    let eta = solve_eta_numdual(rho, center, quartic, beta2ridge);
    let beta1 = eta * d / (d + lambda);
    let beta2 = eta * lambda / (d + lambda);
    let nll = D::from(0.5) * (eta - D::from(center)) * (eta - D::from(center))
        + D::from(0.25) * D::from(quartic) * eta * eta * eta * eta
        + D::from(0.5) * d * beta2 * beta2;
    let curvature = D::one() + D::from(3.0) * D::from(quartic) * eta * eta;
    let det = (curvature + lambda) * (curvature + d) - curvature * curvature;
    nll + D::from(0.5) * lambda * beta1 * beta1 + D::from(0.5) * det.ln() - D::from(0.5) * rho
}

fn coupled_quarticobjective_f64(rho: f64, center: f64, quartic: f64, beta2ridge: f64) -> f64 {
    let lambda = rho.exp();
    let d = beta2ridge;
    let eta = solve_eta_numdual(rho, center, quartic, beta2ridge);
    let beta1 = eta * d / (d + lambda);
    let beta2 = eta * lambda / (d + lambda);
    let nll = 0.5 * (eta - center).powi(2) + 0.25 * quartic * eta.powi(4) + 0.5 * d * beta2.powi(2);
    let curvature = 1.0 + 3.0 * quartic * eta * eta;
    let det = (curvature + lambda) * (curvature + d) - curvature * curvature;
    nll + 0.5 * lambda * beta1 * beta1 + 0.5 * det.ln() - 0.5 * rho
}

fn coupled_quarticobjective_f1(rho: F1, center: f64, quartic: f64, beta2ridge: f64) -> F1 {
    let lambda = rho.exp();
    let d = F1::cst(beta2ridge);
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
    beta2ridge: f64,
    marker: PhantomData<T>,
}

impl<T: AD> CoupledQuarticObjectiveFn<T> {
    fn new(center: f64, quartic: f64, beta2ridge: f64) -> Self {
        Self {
            center,
            quartic,
            beta2ridge,
            marker: PhantomData,
        }
    }

    fn to_other_ad_type<T2: AD>(&self) -> CoupledQuarticObjectiveFn<T2> {
        CoupledQuarticObjectiveFn::new(self.center, self.quartic, self.beta2ridge)
    }
}

impl<T: AD> DifferentiableFunctionTrait<T> for CoupledQuarticObjectiveFn<T> {
    const NAME: &'static str = "CoupledQuarticObjectiveFn";

    fn call(&self, inputs: &[T], _: bool) -> Vec<T> {
        let rho = inputs[0];
        let lambda = rho.exp();
        let d = T::constant(self.beta2ridge);
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

#[derive(Clone)]
struct LowerBoundConstrainedExactFamily {
    target: f64,
    lower: f64,
}

impl CustomFamily for LowerBoundConstrainedExactFamily {
    fn evaluate(&self, block_states: &[ParameterBlockState]) -> Result<FamilyEvaluation, String> {
        let beta = block_states
            .first()
            .ok_or_else(|| "missing block 0".to_string())?
            .beta[0];
        let resid = beta - self.target;
        Ok(FamilyEvaluation {
            log_likelihood: -0.5 * resid * resid,
            blockworking_sets: vec![BlockWorkingSet::ExactNewton {
                gradient: array![-resid],
                hessian: SymmetricMatrix::Dense(array![[1.0]]),
            }],
        })
    }

    fn exact_newton_joint_hessian(
        &self,
        _: &[ParameterBlockState],
    ) -> Result<Option<Array2<f64>>, String> {
        Ok(Some(array![[1.0]]))
    }

    fn exact_newton_hessian_directional_derivative(
        &self,
        _: &[ParameterBlockState],
        _: usize,
        _: &Array1<f64>,
    ) -> Result<Option<Array2<f64>>, String> {
        Ok(Some(array![[0.0]]))
    }

    fn exact_newton_joint_hessian_directional_derivative(
        &self,
        _: &[ParameterBlockState],
        _: &Array1<f64>,
    ) -> Result<Option<Array2<f64>>, String> {
        Ok(Some(array![[0.0]]))
    }

    fn block_linear_constraints(
        &self,
        _: &[ParameterBlockState],
        block_idx: usize,
        _: &ParameterBlockSpec,
    ) -> Result<Option<LinearInequalityConstraints>, String> {
        if block_idx != 0 {
            return Ok(None);
        }
        Ok(Some(LinearInequalityConstraints {
            a: array![[1.0]],
            b: array![self.lower],
        }))
    }
}

fn constrained_exactobjective_numdual<D: DualNum<f64> + Copy>(
    rho: D,
    target: f64,
    lower: f64,
) -> D {
    let lambda = rho.exp();
    let beta_hat = D::from(lower);
    let resid = beta_hat - D::from(target);
    D::from(0.5) * resid * resid
        + D::from(0.5) * lambda * beta_hat * beta_hat
        + D::from(0.5) * (D::one() + lambda).ln()
        - D::from(0.5) * rho
}

fn constrained_exactobjective_f64(rho: f64, target: f64, lower: f64) -> f64 {
    let lambda = rho.exp();
    let beta_hat = lower;
    let resid = beta_hat - target;
    0.5 * resid * resid + 0.5 * lambda * beta_hat * beta_hat + 0.5 * (1.0 + lambda).ln() - 0.5 * rho
}

fn constrained_exactobjective_f1(rho: F1, target: f64, lower: f64) -> F1 {
    let lambda = rho.exp();
    let beta_hat = F1::cst(lower);
    let resid = beta_hat - F1::cst(target);
    F1::cst(0.5) * resid * resid
        + F1::cst(0.5) * lambda * beta_hat * beta_hat
        + F1::cst(0.5) * (F1::cst(1.0) + lambda).ln()
        - F1::cst(0.5) * rho
}

#[derive(Clone)]
struct ConstrainedExactObjectiveFn<T: AD> {
    target: f64,
    lower: f64,
    marker: PhantomData<T>,
}

impl<T: AD> ConstrainedExactObjectiveFn<T> {
    fn new(target: f64, lower: f64) -> Self {
        Self {
            target,
            lower,
            marker: PhantomData,
        }
    }

    fn to_other_ad_type<T2: AD>(&self) -> ConstrainedExactObjectiveFn<T2> {
        ConstrainedExactObjectiveFn::new(self.target, self.lower)
    }
}

impl<T: AD> DifferentiableFunctionTrait<T> for ConstrainedExactObjectiveFn<T> {
    const NAME: &'static str = "ConstrainedExactObjectiveFn";

    fn call(&self, inputs: &[T], _: bool) -> Vec<T> {
        let rho = inputs[0];
        let lambda = rho.exp();
        let beta_hat = T::constant(self.lower);
        let resid = beta_hat - T::constant(self.target);
        vec![
            T::constant(0.5) * resid * resid
                + T::constant(0.5) * lambda * beta_hat * beta_hat
                + T::constant(0.5) * (T::one() + lambda).ln()
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
fn exact_joint_quadratic_lamlgradient_matches_three_autodiff_engines() {
    let center = 0.7;
    let quartic = 0.18;
    let beta2ridge = 1.4;
    let family = CoupledQuarticExactFamily {
        center,
        quartic,
        beta2ridge,
    };
    let specs = vec![
        ParameterBlockSpec {
            name: "shape".to_string(),
            design: DesignMatrix::Dense(gam::matrix::DenseDesignMatrix::from(array![[1.0]])),
            offset: array![0.0],
            penalties: vec![PenaltyMatrix::Dense(Array2::eye(1))],
            nullspace_dims: vec![0],
            initial_log_lambdas: array![0.0],
            initial_beta: Some(array![0.0]),
        },
        ParameterBlockSpec {
            name: "aux".to_string(),
            design: DesignMatrix::Dense(gam::matrix::DenseDesignMatrix::from(array![[1.0]])),
            offset: array![0.0],
            penalties: vec![],
            nullspace_dims: vec![],
            initial_log_lambdas: Array1::zeros(0),
            initial_beta: Some(array![0.0]),
        },
    ];
    let derivative_blocks = vec![
        Vec::<CustomFamilyBlockPsiDerivative>::new(),
        Vec::<CustomFamilyBlockPsiDerivative>::new(),
    ];
    let options = BlockwiseFitOptions {
        use_remlobjective: true,
        compute_covariance: false,
        ridge_floor: 1e-12,
        ..BlockwiseFitOptions::default()
    };
    let f_std = CoupledQuarticObjectiveFn::<f64>::new(center, quartic, beta2ridge);
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
            gam::families::custom_family::EvalMode::ValueAndGradient,
        )
        .expect("exact joint hyper eval");
        let (value_nd, grad_nd) = first_derivative(
            |x| coupled_quarticobjective_numdual(x, center, quartic, beta2ridge),
            rho,
        );
        let grad_autodiff = diff(
            |x: F1| coupled_quarticobjective_f1(x, center, quartic, beta2ridge),
            rho,
        );
        let (_, jac) = engine.derivative(&[rho]);
        let gradfd = (coupled_quarticobjective_f64(rho + 1e-6, center, quartic, beta2ridge)
            - coupled_quarticobjective_f64(rho - 1e-6, center, quartic, beta2ridge))
            / (2.0 * 1e-6);

        assert!(
            (result.objective - value_nd).abs() < 5e-10,
            "exact_joint_laml objective mismatch at rho={rho}: analytic={} num_dual={} closed_form={}",
            result.objective,
            value_nd,
            coupled_quarticobjective_f64(rho, center, quartic, beta2ridge)
        );
        assert_manual_ad_band!(
            "exact_joint_laml",
            rho,
            "gradient",
            result.gradient[0],
            "num_dual" => grad_nd,
            "ad_trait" => jac[(0, 0)],
            "fd" => gradfd,
            "autodiff" => grad_autodiff
        );
    }
}

#[test]
fn exact_joint_quadratic_lamlgradient_respects_active_constraint_tangent_space() {
    let target = -0.4;
    let lower = 0.6;
    let family = LowerBoundConstrainedExactFamily { target, lower };
    let specs = vec![ParameterBlockSpec {
        name: "constrained".to_string(),
        design: DesignMatrix::Dense(gam::matrix::DenseDesignMatrix::from(array![[1.0]])),
        offset: array![0.0],
        penalties: vec![PenaltyMatrix::Dense(Array2::eye(1))],
        nullspace_dims: vec![0],
        initial_log_lambdas: array![0.0],
        initial_beta: Some(array![lower]),
    }];
    let derivative_blocks = vec![Vec::<CustomFamilyBlockPsiDerivative>::new()];
    let options = BlockwiseFitOptions {
        use_remlobjective: true,
        compute_covariance: false,
        ridge_floor: 1e-12,
        ..BlockwiseFitOptions::default()
    };
    let f_std = ConstrainedExactObjectiveFn::<f64>::new(target, lower);
    let f_ad = f_std.to_other_ad_type::<adfn<1>>();
    let engine = FunctionEngine::new(f_std, f_ad, ForwardAD::new());
    let rho_points = [-1.4, -0.5, 0.0, 0.8, 1.6];

    for rho in rho_points {
        let result = evaluate_custom_family_joint_hyper(
            &family,
            &specs,
            &options,
            &array![rho],
            &derivative_blocks,
            None,
            gam::families::custom_family::EvalMode::ValueAndGradient,
        )
        .expect("constrained exact joint hyper eval");
        let (value_nd, grad_nd) = first_derivative(
            |x| constrained_exactobjective_numdual(x, target, lower),
            rho,
        );
        let grad_autodiff = diff(|x: F1| constrained_exactobjective_f1(x, target, lower), rho);
        let (_, jac) = engine.derivative(&[rho]);
        let gradfd = (constrained_exactobjective_f64(rho + 1e-6, target, lower)
            - constrained_exactobjective_f64(rho - 1e-6, target, lower))
            / (2.0 * 1e-6);

        assert!(
            (result.objective - value_nd).abs() < 5e-10,
            "constrained_exact_joint objective mismatch at rho={rho}: analytic={} num_dual={} closed_form={}",
            result.objective,
            value_nd,
            constrained_exactobjective_f64(rho, target, lower)
        );
        assert_manual_ad_band!(
            "constrained_exact_joint_laml",
            rho,
            "gradient",
            result.gradient[0],
            "num_dual" => grad_nd,
            "ad_trait" => jac[(0, 0)],
            "fd" => gradfd,
            "autodiff" => grad_autodiff
        );
    }
}

#[test]
fn exact_joint_quadratic_lamlgradient_requires_joint_stationarity() {
    let center = 0.7;
    let quartic = 0.18;
    let beta2ridge = 1.4;
    let family = CoupledQuarticExactFamily {
        center,
        quartic,
        beta2ridge,
    };
    let specs = vec![
        ParameterBlockSpec {
            name: "shape".to_string(),
            design: DesignMatrix::Dense(gam::matrix::DenseDesignMatrix::from(array![[1.0]])),
            offset: array![0.0],
            penalties: vec![PenaltyMatrix::Dense(Array2::eye(1))],
            nullspace_dims: vec![0],
            initial_log_lambdas: array![0.0],
            initial_beta: Some(array![0.0]),
        },
        ParameterBlockSpec {
            name: "aux".to_string(),
            design: DesignMatrix::Dense(gam::matrix::DenseDesignMatrix::from(array![[1.0]])),
            offset: array![0.0],
            penalties: vec![],
            nullspace_dims: vec![],
            initial_log_lambdas: Array1::zeros(0),
            initial_beta: Some(array![0.0]),
        },
    ];
    let derivative_blocks = vec![
        Vec::<CustomFamilyBlockPsiDerivative>::new(),
        Vec::<CustomFamilyBlockPsiDerivative>::new(),
    ];
    let options = BlockwiseFitOptions {
        use_remlobjective: true,
        compute_covariance: false,
        ridge_floor: 1e-12,
        inner_tol: 1e-3,
        ..BlockwiseFitOptions::default()
    };
    let f_std = CoupledQuarticObjectiveFn::<f64>::new(center, quartic, beta2ridge);
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
            gam::families::custom_family::EvalMode::ValueAndGradient,
        )
        .expect("exact joint hyper eval");
        let (value_nd, grad_nd) = first_derivative(
            |x| coupled_quarticobjective_numdual(x, center, quartic, beta2ridge),
            rho,
        );
        let grad_autodiff = diff(
            |x: F1| coupled_quarticobjective_f1(x, center, quartic, beta2ridge),
            rho,
        );
        let (_, jac) = engine.derivative(&[rho]);
        let gradfd = (coupled_quarticobjective_f64(rho + 1e-6, center, quartic, beta2ridge)
            - coupled_quarticobjective_f64(rho - 1e-6, center, quartic, beta2ridge))
            / (2.0 * 1e-6);

        assert!(
            (result.objective - value_nd).abs() < 5e-7,
            "exact_joint_stationarity objective mismatch at rho={rho}: analytic={} num_dual={} closed_form={}",
            result.objective,
            value_nd,
            coupled_quarticobjective_f64(rho, center, quartic, beta2ridge)
        );
        assert_manual_ad_band!(
            "exact_joint_stationarity",
            rho,
            "gradient",
            result.gradient[0],
            "num_dual" => grad_nd,
            "ad_trait" => jac[(0, 0)],
            "fd" => gradfd,
            "autodiff" => grad_autodiff
        );
    }
}
