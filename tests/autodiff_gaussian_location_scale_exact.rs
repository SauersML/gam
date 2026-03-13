use ad_trait::AD;
use ad_trait::differentiable_function::{DifferentiableFunctionTrait, ForwardAD};
use ad_trait::forward_ad::adfn::adfn;
use ad_trait::function_engine::FunctionEngine;
use autodiff::{F1, Float, diff};
use num_dual::{DualNum, first_derivative};
use std::marker::PhantomData;

mod common;

#[derive(Clone, Copy)]
struct GaussianPsiParams {
    y: f64,
    betamu: f64,
    beta_ls: f64,
    z_0: f64,
    z_1: f64,
    z_2: f64,
    x_0: f64,
    x_1: f64,
    x_2: f64,
    umu: f64,
    u_ls: f64,
}

#[derive(Clone, Copy)]
enum GaussianPsiQuantity {
    Objective,
    Score(usize),
    Hessian(usize, usize),
}

#[derive(Clone, Copy)]
struct GaussianManualRowGeometry {
    objective_psi: f64,
    score_psi: [f64; 2],
    hessian_psi: [[f64; 2]; 2],
    hessian_psi_drift: [[f64; 2]; 2],
}

fn z_numdual<D: DualNum<f64> + Copy>(psi: D, p: &GaussianPsiParams) -> D {
    D::from(p.z_0) + D::from(p.z_1) * psi + D::from(0.5 * p.z_2) * psi * psi
}

fn x_numdual<D: DualNum<f64> + Copy>(psi: D, p: &GaussianPsiParams) -> D {
    D::from(p.x_0) + D::from(p.x_1) * psi + D::from(0.5 * p.x_2) * psi * psi
}

fn z_f1(psi: F1, p: &GaussianPsiParams) -> F1 {
    F1::cst(p.z_0) + F1::cst(p.z_1) * psi + F1::cst(0.5 * p.z_2) * psi * psi
}

fn x_f1(psi: F1, p: &GaussianPsiParams) -> F1 {
    F1::cst(p.x_0) + F1::cst(p.x_1) * psi + F1::cst(0.5 * p.x_2) * psi * psi
}

fn z_ad<T: AD>(psi: T, p: &GaussianPsiParams) -> T {
    T::constant(p.z_0) + T::constant(p.z_1) * psi + T::constant(0.5 * p.z_2) * psi * psi
}

fn x_ad<T: AD>(psi: T, p: &GaussianPsiParams) -> T {
    T::constant(p.x_0) + T::constant(p.x_1) * psi + T::constant(0.5 * p.x_2) * psi * psi
}

fn gaussian_psi_quantity_numdual<D: DualNum<f64> + Copy>(
    psi: D,
    p: &GaussianPsiParams,
    quantity: GaussianPsiQuantity,
) -> D {
    let z = z_numdual(psi, p);
    let x = x_numdual(psi, p);
    let mu = D::from(p.betamu) * z;
    let eta = D::from(p.beta_ls) * x;
    let sigma = eta.exp();
    let r = D::from(p.y) - mu;
    let w = sigma.powi(-2);
    let alpha = r * w;
    let b = r * r * w;
    match quantity {
        GaussianPsiQuantity::Objective => D::from(0.5) * b + eta,
        GaussianPsiQuantity::Score(0) => -alpha * z,
        GaussianPsiQuantity::Score(1) => (D::one() - b) * x,
        GaussianPsiQuantity::Hessian(0, 0) => w * z * z,
        GaussianPsiQuantity::Hessian(0, 1) | GaussianPsiQuantity::Hessian(1, 0) => {
            D::from(2.0) * alpha * z * x
        }
        GaussianPsiQuantity::Hessian(1, 1) => D::from(2.0) * b * x * x,
        GaussianPsiQuantity::Score(_) | GaussianPsiQuantity::Hessian(_, _) => D::zero(),
    }
}

fn gaussian_psi_quantity_f1(psi: F1, p: &GaussianPsiParams, quantity: GaussianPsiQuantity) -> F1 {
    let z = z_f1(psi, p);
    let x = x_f1(psi, p);
    let mu = F1::cst(p.betamu) * z;
    let eta = F1::cst(p.beta_ls) * x;
    let sigma = eta.exp();
    let r = F1::cst(p.y) - mu;
    let w = sigma.powi(-2);
    let alpha = r * w;
    let b = r * r * w;
    match quantity {
        GaussianPsiQuantity::Objective => F1::cst(0.5) * b + eta,
        GaussianPsiQuantity::Score(0) => -alpha * z,
        GaussianPsiQuantity::Score(1) => (F1::cst(1.0) - b) * x,
        GaussianPsiQuantity::Hessian(0, 0) => w * z * z,
        GaussianPsiQuantity::Hessian(0, 1) | GaussianPsiQuantity::Hessian(1, 0) => {
            F1::cst(2.0) * alpha * z * x
        }
        GaussianPsiQuantity::Hessian(1, 1) => F1::cst(2.0) * b * x * x,
        GaussianPsiQuantity::Score(_) | GaussianPsiQuantity::Hessian(_, _) => F1::cst(0.0),
    }
}

fn gaussian_psi_quantity_ad<T: AD>(
    psi: T,
    p: &GaussianPsiParams,
    quantity: GaussianPsiQuantity,
) -> T {
    let z = z_ad(psi, p);
    let x = x_ad(psi, p);
    let mu = T::constant(p.betamu) * z;
    let eta = T::constant(p.beta_ls) * x;
    let sigma = eta.exp();
    let r = T::constant(p.y) - mu;
    let w = sigma.powi(-2);
    let alpha = r * w;
    let b = r * r * w;
    match quantity {
        GaussianPsiQuantity::Objective => T::constant(0.5) * b + eta,
        GaussianPsiQuantity::Score(0) => -alpha * z,
        GaussianPsiQuantity::Score(1) => (T::one() - b) * x,
        GaussianPsiQuantity::Hessian(0, 0) => w * z * z,
        GaussianPsiQuantity::Hessian(0, 1) | GaussianPsiQuantity::Hessian(1, 0) => {
            T::constant(2.0) * alpha * z * x
        }
        GaussianPsiQuantity::Hessian(1, 1) => T::constant(2.0) * b * x * x,
        GaussianPsiQuantity::Score(_) | GaussianPsiQuantity::Hessian(_, _) => T::zero(),
    }
}

#[derive(Clone)]
struct GaussianPsiFn<T: AD> {
    params: GaussianPsiParams,
    quantity: GaussianPsiQuantity,
    marker: PhantomData<T>,
}

impl<T: AD> GaussianPsiFn<T> {
    fn new(params: GaussianPsiParams, quantity: GaussianPsiQuantity) -> Self {
        Self {
            params,
            quantity,
            marker: PhantomData,
        }
    }

    fn to_other_ad_type<T2: AD>(&self) -> GaussianPsiFn<T2> {
        GaussianPsiFn::new(self.params, self.quantity)
    }
}

impl<T: AD> DifferentiableFunctionTrait<T> for GaussianPsiFn<T> {
    const NAME: &'static str = "GaussianPsiFn";

    fn call(&self, inputs: &[T], _: bool) -> Vec<T> {
        vec![gaussian_psi_quantity_ad(
            inputs[0],
            &self.params,
            self.quantity,
        )]
    }

    fn num_inputs(&self) -> usize {
        1
    }

    fn num_outputs(&self) -> usize {
        1
    }
}

fn gaussian_manualrow_geometry(psi: f64, p: &GaussianPsiParams) -> GaussianManualRowGeometry {
    let z = p.z_0 + p.z_1 * psi + 0.5 * p.z_2 * psi * psi;
    let x = p.x_0 + p.x_1 * psi + 0.5 * p.x_2 * psi * psi;
    let z_a = p.z_1 + p.z_2 * psi;
    let x_a = p.x_1 + p.x_2 * psi;
    let mu = p.betamu * z;
    let eta = p.beta_ls * x;
    let mu_a = p.betamu * z_a;
    let eta_a = p.beta_ls * x_a;
    let sigma = eta.exp();
    let r = p.y - mu;
    let w = sigma.powi(-2);
    let alpha = r * w;
    let b = r * r * w;
    let w_a = -2.0 * w * eta_a;
    let alpha_a = -w * mu_a - 2.0 * alpha * eta_a;
    let b_a = -2.0 * alpha * mu_a - 2.0 * b * eta_a;

    let objective_psi = -alpha * mu_a + (1.0 - b) * eta_a;
    let score_psi = [-alpha_a * z - alpha * z_a, -b_a * x + (1.0 - b) * x_a];
    let hessian_psi = [
        [
            w_a * z * z + w * (z_a * z + z * z_a),
            2.0 * (alpha_a * z * x + alpha * z_a * x + alpha * z * x_a),
        ],
        [
            2.0 * (alpha_a * z * x + alpha * z_a * x + alpha * z * x_a),
            2.0 * (b_a * x * x + b * (x_a * x + x * x_a)),
        ],
    ];

    let dotmu = z * p.umu;
    let dot_eta = x * p.u_ls;
    let dotmu_a = z_a * p.umu;
    let dot_eta_a = x_a * p.u_ls;
    let w_u = -2.0 * w * dot_eta;
    let c_u = -2.0 * w * dotmu - 4.0 * alpha * dot_eta;
    let d_u = -4.0 * alpha * dotmu - 4.0 * b * dot_eta;
    let w_a_u = 4.0 * w * dot_eta * eta_a - 2.0 * w * dot_eta_a;
    let c_a_u = -2.0 * w * dotmu_a
        + 4.0 * w * (dot_eta * mu_a + dotmu * eta_a)
        + 8.0 * alpha * dot_eta * eta_a
        - 4.0 * alpha * dot_eta_a;
    let d_a_u = 4.0 * w * dotmu * mu_a
        + 8.0 * alpha * (dot_eta * mu_a + dotmu * eta_a)
        + 8.0 * b * dot_eta * eta_a
        - 4.0 * alpha * dotmu_a
        - 4.0 * b * dot_eta_a;
    let hessian_psi_drift = [
        [
            w_a_u * z * z + w_u * (z_a * z + z * z_a),
            c_a_u * z * x + c_u * (z_a * x + z * x_a),
        ],
        [
            c_a_u * z * x + c_u * (z_a * x + z * x_a),
            d_a_u * x * x + d_u * (x_a * x + x * x_a),
        ],
    ];

    GaussianManualRowGeometry {
        objective_psi,
        score_psi,
        hessian_psi,
        hessian_psi_drift,
    }
}

fn expected_gaussian_fixed_beta_quantity(
    manual: &GaussianManualRowGeometry,
    quantity: GaussianPsiQuantity,
) -> f64 {
    match quantity {
        GaussianPsiQuantity::Objective => manual.objective_psi,
        GaussianPsiQuantity::Score(idx) => manual.score_psi[idx],
        GaussianPsiQuantity::Hessian(i, j) => manual.hessian_psi[i][j],
    }
}

fn eps_gaussianhessian_psi_numdual<D: DualNum<f64> + Copy>(
    eps: D,
    psi0: f64,
    p: &GaussianPsiParams,
    i: usize,
    j: usize,
) -> D {
    let betamu = D::from(p.betamu) + D::from(p.umu) * eps;
    let beta_ls = D::from(p.beta_ls) + D::from(p.u_ls) * eps;
    let psi = D::from(psi0);
    let z = z_numdual(psi, p);
    let x = x_numdual(psi, p);
    let z_a = D::from(p.z_1 + p.z_2 * psi0);
    let x_a = D::from(p.x_1 + p.x_2 * psi0);
    let mu = betamu * z;
    let eta = beta_ls * x;
    let mu_a = betamu * z_a;
    let eta_a = beta_ls * x_a;
    let sigma = eta.exp();
    let r = D::from(p.y) - mu;
    let w = sigma.powi(-2);
    let alpha = r * w;
    let b = r * r * w;
    let w_a = -D::from(2.0) * w * eta_a;
    let alpha_a = -w * mu_a - D::from(2.0) * alpha * eta_a;
    let b_a = -D::from(2.0) * alpha * mu_a - D::from(2.0) * b * eta_a;
    match (i, j) {
        (0, 0) => w_a * z * z + w * (z_a * z + z * z_a),
        (0, 1) | (1, 0) => D::from(2.0) * (alpha_a * z * x + alpha * z_a * x + alpha * z * x_a),
        (1, 1) => D::from(2.0) * (b_a * x * x + b * (x_a * x + x * x_a)),
        _ => D::zero(),
    }
}

fn eps_gaussianhessian_psi_f1(eps: F1, psi0: f64, p: &GaussianPsiParams, i: usize, j: usize) -> F1 {
    let betamu = F1::cst(p.betamu) + F1::cst(p.umu) * eps;
    let beta_ls = F1::cst(p.beta_ls) + F1::cst(p.u_ls) * eps;
    let psi = F1::cst(psi0);
    let z = z_f1(psi, p);
    let x = x_f1(psi, p);
    let z_a = F1::cst(p.z_1 + p.z_2 * psi0);
    let x_a = F1::cst(p.x_1 + p.x_2 * psi0);
    let mu = betamu * z;
    let eta = beta_ls * x;
    let mu_a = betamu * z_a;
    let eta_a = beta_ls * x_a;
    let sigma = eta.exp();
    let r = F1::cst(p.y) - mu;
    let w = sigma.powi(-2);
    let alpha = r * w;
    let b = r * r * w;
    let w_a = -F1::cst(2.0) * w * eta_a;
    let alpha_a = -w * mu_a - F1::cst(2.0) * alpha * eta_a;
    let b_a = -F1::cst(2.0) * alpha * mu_a - F1::cst(2.0) * b * eta_a;
    match (i, j) {
        (0, 0) => w_a * z * z + w * (z_a * z + z * z_a),
        (0, 1) | (1, 0) => F1::cst(2.0) * (alpha_a * z * x + alpha * z_a * x + alpha * z * x_a),
        (1, 1) => F1::cst(2.0) * (b_a * x * x + b * (x_a * x + x * x_a)),
        _ => F1::cst(0.0),
    }
}

fn eps_gaussianhessian_psi_ad<T: AD>(
    eps: T,
    psi0: f64,
    p: &GaussianPsiParams,
    i: usize,
    j: usize,
) -> T {
    let betamu = T::constant(p.betamu) + T::constant(p.umu) * eps;
    let beta_ls = T::constant(p.beta_ls) + T::constant(p.u_ls) * eps;
    let psi = T::constant(psi0);
    let z = z_ad(psi, p);
    let x = x_ad(psi, p);
    let z_a = T::constant(p.z_1 + p.z_2 * psi0);
    let x_a = T::constant(p.x_1 + p.x_2 * psi0);
    let mu = betamu * z;
    let eta = beta_ls * x;
    let mu_a = betamu * z_a;
    let eta_a = beta_ls * x_a;
    let sigma = eta.exp();
    let r = T::constant(p.y) - mu;
    let w = sigma.powi(-2);
    let alpha = r * w;
    let b = r * r * w;
    let w_a = -T::constant(2.0) * w * eta_a;
    let alpha_a = -w * mu_a - T::constant(2.0) * alpha * eta_a;
    let b_a = -T::constant(2.0) * alpha * mu_a - T::constant(2.0) * b * eta_a;
    match (i, j) {
        (0, 0) => w_a * z * z + w * (z_a * z + z * z_a),
        (0, 1) | (1, 0) => T::constant(2.0) * (alpha_a * z * x + alpha * z_a * x + alpha * z * x_a),
        (1, 1) => T::constant(2.0) * (b_a * x * x + b * (x_a * x + x * x_a)),
        _ => T::zero(),
    }
}

#[derive(Clone)]
struct EpsGaussianHessianPsiFn<T: AD> {
    params: GaussianPsiParams,
    psi0: f64,
    i: usize,
    j: usize,
    marker: PhantomData<T>,
}

impl<T: AD> EpsGaussianHessianPsiFn<T> {
    fn new(params: GaussianPsiParams, psi0: f64, i: usize, j: usize) -> Self {
        Self {
            params,
            psi0,
            i,
            j,
            marker: PhantomData,
        }
    }

    fn to_other_ad_type<T2: AD>(&self) -> EpsGaussianHessianPsiFn<T2> {
        EpsGaussianHessianPsiFn::new(self.params, self.psi0, self.i, self.j)
    }
}

impl<T: AD> DifferentiableFunctionTrait<T> for EpsGaussianHessianPsiFn<T> {
    const NAME: &'static str = "EpsGaussianHessianPsiFn";

    fn call(&self, inputs: &[T], _: bool) -> Vec<T> {
        vec![eps_gaussianhessian_psi_ad(
            inputs[0],
            self.psi0,
            &self.params,
            self.i,
            self.j,
        )]
    }

    fn num_inputs(&self) -> usize {
        1
    }

    fn num_outputs(&self) -> usize {
        1
    }
}

#[test]
fn gaussian_location_scale_joint_psi_fixed_beta_terms_match_three_autodiff_engines() {
    let params = GaussianPsiParams {
        y: 1.4,
        betamu: -0.6,
        beta_ls: 0.35,
        z_0: 1.2,
        z_1: -0.3,
        z_2: 0.15,
        x_0: -0.4,
        x_1: 0.8,
        x_2: -0.2,
        umu: 0.25,
        u_ls: -0.4,
    };
    let points = [-1.0, -0.2, 0.4, 1.1];
    let quantities = [
        ("objective_psi", GaussianPsiQuantity::Objective),
        ("score_psimu", GaussianPsiQuantity::Score(0)),
        ("score_psi_ls", GaussianPsiQuantity::Score(1)),
        ("hessian_psimumu", GaussianPsiQuantity::Hessian(0, 0)),
        ("hessian_psimuls", GaussianPsiQuantity::Hessian(0, 1)),
        ("hessian_psi_lsls", GaussianPsiQuantity::Hessian(1, 1)),
    ];

    for psi in points {
        let manual = gaussian_manualrow_geometry(psi, &params);
        for (name, quantity) in quantities {
            let expected = expected_gaussian_fixed_beta_quantity(&manual, quantity);
            let (_, d1_nd) =
                first_derivative(|x| gaussian_psi_quantity_numdual(x, &params, quantity), psi);
            let d1_autodiff = diff(|x| gaussian_psi_quantity_f1(x, &params, quantity), psi);
            let f_std = GaussianPsiFn::<f64>::new(params, quantity);
            let f_ad = f_std.to_other_ad_type::<adfn<1>>();
            let engine = FunctionEngine::new(f_std, f_ad, ForwardAD::new());
            let (_, jac) = engine.derivative(&[psi]);
            assert_manual_ad_band!(
                "gaussian_location_scale_joint_psi",
                psi,
                name,
                expected,
                "num_dual" => d1_nd,
                "autodiff" => d1_autodiff,
                "ad_trait" => jac[(0, 0)]
            );
        }
    }
}

#[test]
fn gaussian_location_scale_joint_psihessian_drift_matches_three_autodiff_engines() {
    let params = GaussianPsiParams {
        y: 1.4,
        betamu: -0.6,
        beta_ls: 0.35,
        z_0: 1.2,
        z_1: -0.3,
        z_2: 0.15,
        x_0: -0.4,
        x_1: 0.8,
        x_2: -0.2,
        umu: 0.25,
        u_ls: -0.4,
    };
    let psi0 = 0.37;
    let manual = gaussian_manualrow_geometry(psi0, &params);
    let pairs = [
        ("T_psimumu", 0usize, 0usize),
        ("T_psimuls", 0, 1),
        ("T_psi_lsls", 1, 1),
    ];
    for (name, i, j) in pairs {
        let expected = manual.hessian_psi_drift[i][j];
        let (_, d1_nd) = first_derivative(
            |eps| eps_gaussianhessian_psi_numdual(eps, psi0, &params, i, j),
            0.0,
        );
        let d1_autodiff = diff(
            |eps| eps_gaussianhessian_psi_f1(eps, psi0, &params, i, j),
            0.0,
        );
        let f_std = EpsGaussianHessianPsiFn::<f64>::new(params, psi0, i, j);
        let f_ad = f_std.to_other_ad_type::<adfn<1>>();
        let engine = FunctionEngine::new(f_std, f_ad, ForwardAD::new());
        let (_, jac) = engine.derivative(&[0.0]);
        assert_manual_ad_band!(
            "gaussian_location_scale_joint_psi_drift",
            psi0,
            name,
            expected,
            "num_dual" => d1_nd,
            "autodiff" => d1_autodiff,
            "ad_trait" => jac[(0, 0)]
        );
    }
}
