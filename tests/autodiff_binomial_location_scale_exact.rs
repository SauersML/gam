use ad_trait::AD;
use ad_trait::differentiable_function::{DifferentiableFunctionTrait, ForwardAD};
use ad_trait::forward_ad::adfn::adfn;
use ad_trait::function_engine::FunctionEngine;
use autodiff::{F1, Float, diff};
use num_dual::{DualNum, first_derivative, second_derivative, third_derivative};
use std::marker::PhantomData;

mod common;

const POLY_LOSS_KAPPA: f64 = 0.2;

#[derive(Clone, Copy)]
struct CombinedLocationScaleParams {
    beta_1: f64,
    beta_2: f64,
    beta_ls: f64,
    z1_0: f64,
    z1_1: f64,
    z1_2: f64,
    z2_0: f64,
    z2_1: f64,
    z2_2: f64,
    x_0: f64,
    x_1: f64,
    x_2: f64,
    u_1: f64,
    u_2: f64,
    u_ls: f64,
}

#[derive(Clone, Copy)]
enum PsiQuantity {
    Objective,
    Score(usize),
    Hessian(usize, usize),
}

#[derive(Clone, Copy)]
struct ManualRowGeometry {
    objective_psi: f64,
    score_psi: [f64; 3],
    hessian_psi: [[f64; 3]; 3],
    hessian_psi_drift: [[f64; 3]; 3],
}

fn poly_loss_numdual<D: DualNum<f64> + Copy>(q: D) -> D {
    D::from(0.5) * q * q + D::from(0.25 * POLY_LOSS_KAPPA) * q.powi(4)
}

fn poly_loss_f1(q: F1) -> F1 {
    F1::cst(0.5) * q * q + F1::cst(0.25 * POLY_LOSS_KAPPA) * q.powi(4)
}

fn poly_loss_ad<T: AD>(q: T) -> T {
    T::constant(0.5) * q * q + T::constant(0.25 * POLY_LOSS_KAPPA) * q.powi(4)
}

fn poly_loss_derivatives(q: f64) -> (f64, f64, f64, f64) {
    let r = q + POLY_LOSS_KAPPA * q.powi(3);
    let w = 1.0 + 3.0 * POLY_LOSS_KAPPA * q * q;
    let nu = 6.0 * POLY_LOSS_KAPPA * q;
    let tau = 6.0 * POLY_LOSS_KAPPA;
    (r, w, nu, tau)
}

fn z1_numdual<D: DualNum<f64> + Copy>(psi: D, p: &CombinedLocationScaleParams) -> D {
    D::from(p.z1_0) + D::from(p.z1_1) * psi + D::from(0.5 * p.z1_2) * psi * psi
}

fn z2_numdual<D: DualNum<f64> + Copy>(psi: D, p: &CombinedLocationScaleParams) -> D {
    D::from(p.z2_0) + D::from(p.z2_1) * psi + D::from(0.5 * p.z2_2) * psi * psi
}

fn x_numdual<D: DualNum<f64> + Copy>(psi: D, p: &CombinedLocationScaleParams) -> D {
    D::from(p.x_0) + D::from(p.x_1) * psi + D::from(0.5 * p.x_2) * psi * psi
}

fn z1_f1(psi: F1, p: &CombinedLocationScaleParams) -> F1 {
    F1::cst(p.z1_0) + F1::cst(p.z1_1) * psi + F1::cst(0.5 * p.z1_2) * psi * psi
}

fn z2_f1(psi: F1, p: &CombinedLocationScaleParams) -> F1 {
    F1::cst(p.z2_0) + F1::cst(p.z2_1) * psi + F1::cst(0.5 * p.z2_2) * psi * psi
}

fn x_f1(psi: F1, p: &CombinedLocationScaleParams) -> F1 {
    F1::cst(p.x_0) + F1::cst(p.x_1) * psi + F1::cst(0.5 * p.x_2) * psi * psi
}

fn z1_ad<T: AD>(psi: T, p: &CombinedLocationScaleParams) -> T {
    T::constant(p.z1_0) + T::constant(p.z1_1) * psi + T::constant(0.5 * p.z1_2) * psi * psi
}

fn z2_ad<T: AD>(psi: T, p: &CombinedLocationScaleParams) -> T {
    T::constant(p.z2_0) + T::constant(p.z2_1) * psi + T::constant(0.5 * p.z2_2) * psi * psi
}

fn x_ad<T: AD>(psi: T, p: &CombinedLocationScaleParams) -> T {
    T::constant(p.x_0) + T::constant(p.x_1) * psi + T::constant(0.5 * p.x_2) * psi * psi
}

fn psi_quantity_numdual<D: DualNum<f64> + Copy>(
    psi: D,
    p: &CombinedLocationScaleParams,
    quantity: PsiQuantity,
) -> D {
    let z1 = z1_numdual(psi, p);
    let z2 = z2_numdual(psi, p);
    let x = x_numdual(psi, p);
    let a = D::from(p.beta_1) * z1 + D::from(p.beta_2) * z2;
    let ell = D::from(p.beta_ls) * x;
    let s = (-ell).exp();
    let q = -a * s;
    let r = q + D::from(POLY_LOSS_KAPPA) * q.powi(3);
    let w = D::one() + D::from(3.0 * POLY_LOSS_KAPPA) * q * q;
    let b = [-s * z1, -s * z2, -q * x];
    let q02 = s * z1 * x;
    let q12 = s * z2 * x;
    let q22 = q * x * x;
    match quantity {
        PsiQuantity::Objective => poly_loss_numdual(q),
        PsiQuantity::Score(idx) => r * b[idx],
        PsiQuantity::Hessian(i, j) => {
            let qij = match (i, j) {
                (0, 2) | (2, 0) => q02,
                (1, 2) | (2, 1) => q12,
                (2, 2) => q22,
                _ => D::zero(),
            };
            w * b[i] * b[j] + r * qij
        }
    }
}

fn psi_quantity_f1(psi: F1, p: &CombinedLocationScaleParams, quantity: PsiQuantity) -> F1 {
    let z1 = z1_f1(psi, p);
    let z2 = z2_f1(psi, p);
    let x = x_f1(psi, p);
    let a = F1::cst(p.beta_1) * z1 + F1::cst(p.beta_2) * z2;
    let ell = F1::cst(p.beta_ls) * x;
    let s = (-ell).exp();
    let q = -a * s;
    let r = q + F1::cst(POLY_LOSS_KAPPA) * q.powi(3);
    let w = F1::cst(1.0) + F1::cst(3.0 * POLY_LOSS_KAPPA) * q * q;
    let b = [-s * z1, -s * z2, -q * x];
    let q02 = s * z1 * x;
    let q12 = s * z2 * x;
    let q22 = q * x * x;
    match quantity {
        PsiQuantity::Objective => poly_loss_f1(q),
        PsiQuantity::Score(idx) => r * b[idx],
        PsiQuantity::Hessian(i, j) => {
            let qij = match (i, j) {
                (0, 2) | (2, 0) => q02,
                (1, 2) | (2, 1) => q12,
                (2, 2) => q22,
                _ => F1::cst(0.0),
            };
            w * b[i] * b[j] + r * qij
        }
    }
}

fn psi_quantity_ad<T: AD>(psi: T, p: &CombinedLocationScaleParams, quantity: PsiQuantity) -> T {
    let z1 = z1_ad(psi, p);
    let z2 = z2_ad(psi, p);
    let x = x_ad(psi, p);
    let a = T::constant(p.beta_1) * z1 + T::constant(p.beta_2) * z2;
    let ell = T::constant(p.beta_ls) * x;
    let s = (-ell).exp();
    let q = -a * s;
    let r = q + T::constant(POLY_LOSS_KAPPA) * q.powi(3);
    let w = T::one() + T::constant(3.0 * POLY_LOSS_KAPPA) * q * q;
    let b = [-s * z1, -s * z2, -q * x];
    let q02 = s * z1 * x;
    let q12 = s * z2 * x;
    let q22 = q * x * x;
    match quantity {
        PsiQuantity::Objective => poly_loss_ad(q),
        PsiQuantity::Score(idx) => r * b[idx],
        PsiQuantity::Hessian(i, j) => {
            let qij = match (i, j) {
                (0, 2) | (2, 0) => q02,
                (1, 2) | (2, 1) => q12,
                (2, 2) => q22,
                _ => T::zero(),
            };
            w * b[i] * b[j] + r * qij
        }
    }
}

#[derive(Clone)]
struct PsiQuantityFn<T: AD> {
    params: CombinedLocationScaleParams,
    quantity: PsiQuantity,
    marker: PhantomData<T>,
}

impl<T: AD> PsiQuantityFn<T> {
    fn new(params: CombinedLocationScaleParams, quantity: PsiQuantity) -> Self {
        Self {
            params,
            quantity,
            marker: PhantomData,
        }
    }

    fn to_other_ad_type<T2: AD>(&self) -> PsiQuantityFn<T2> {
        PsiQuantityFn::new(self.params, self.quantity)
    }
}

impl<T: AD> DifferentiableFunctionTrait<T> for PsiQuantityFn<T> {
    const NAME: &'static str = "PsiQuantityFn";

    fn call(&self, inputs: &[T], _: bool) -> Vec<T> {
        vec![psi_quantity_ad(inputs[0], &self.params, self.quantity)]
    }

    fn num_inputs(&self) -> usize {
        1
    }

    fn num_outputs(&self) -> usize {
        1
    }
}

fn manualrow_geometry(psi: f64, p: &CombinedLocationScaleParams) -> ManualRowGeometry {
    let z1 = p.z1_0 + p.z1_1 * psi + 0.5 * p.z1_2 * psi * psi;
    let z2 = p.z2_0 + p.z2_1 * psi + 0.5 * p.z2_2 * psi * psi;
    let x = p.x_0 + p.x_1 * psi + 0.5 * p.x_2 * psi * psi;
    let z1_a = p.z1_1 + p.z1_2 * psi;
    let z2_a = p.z2_1 + p.z2_2 * psi;
    let x_a = p.x_1 + p.x_2 * psi;
    let z1_ab = p.z1_2;
    let z2_ab = p.z2_2;
    let x_ab = p.x_2;
    let a = p.beta_1 * z1 + p.beta_2 * z2;
    let a_a = p.beta_1 * z1_a + p.beta_2 * z2_a;
    let a_ab = p.beta_1 * z1_ab + p.beta_2 * z2_ab;
    let ell = p.beta_ls * x;
    let ell_a = p.beta_ls * x_a;
    let ell_ab = p.beta_ls * x_ab;
    let s = (-ell).exp();
    let q = -a * s;
    let h = -s * a_a - q * ell_a;
    let (r, w, nu, tau) = poly_loss_derivatives(q);
    let objective_psi = r * h;

    let b = [-s * z1, -s * z2, -q * x];
    let c = [
        s * (ell_a * z1 - z1_a),
        s * (ell_a * z2 - z2_a),
        -(h * x + q * x_a),
    ];
    let mut q_mat = [[0.0; 3]; 3];
    q_mat[0][2] = s * z1 * x;
    q_mat[2][0] = q_mat[0][2];
    q_mat[1][2] = s * z2 * x;
    q_mat[2][1] = q_mat[1][2];
    q_mat[2][2] = q * x * x;

    let mut r_a = [[0.0; 3]; 3];
    r_a[0][2] = s * (z1_a * x + z1 * x_a - ell_a * z1 * x);
    r_a[2][0] = r_a[0][2];
    r_a[1][2] = s * (z2_a * x + z2 * x_a - ell_a * z2 * x);
    r_a[2][1] = r_a[1][2];
    r_a[2][2] = h * x * x + q * (2.0 * x_a * x);

    let score_psi = [
        w * h * b[0] + r * c[0],
        w * h * b[1] + r * c[1],
        w * h * b[2] + r * c[2],
    ];

    let mut hessian_psi = [[0.0; 3]; 3];
    for i in 0..3 {
        for j in 0..3 {
            hessian_psi[i][j] = nu * h * b[i] * b[j]
                + w * (c[i] * b[j] + b[i] * c[j] + h * q_mat[i][j])
                + r * r_a[i][j];
        }
    }

    let u = [p.u_1, p.u_2, p.u_ls];
    let alpha = b[0] * u[0] + b[1] * u[1] + b[2] * u[2];
    let gamma = [
        q_mat[0][2] * u[2],
        q_mat[1][2] * u[2],
        q_mat[2][0] * u[0] + q_mat[2][1] * u[1] + q_mat[2][2] * u[2],
    ];
    let alpha_a = c[0] * u[0] + c[1] * u[1] + c[2] * u[2];
    let gamma_a = [
        r_a[0][2] * u[2],
        r_a[1][2] * u[2],
        r_a[2][0] * u[0] + r_a[2][1] * u[1] + r_a[2][2] * u[2],
    ];
    let c_u = [
        [0.0, 0.0, -s * (x * u[2]) * z1 * x],
        [0.0, 0.0, -s * (x * u[2]) * z2 * x],
        [
            -s * (x * u[2]) * z1 * x,
            -s * (x * u[2]) * z2 * x,
            alpha * x * x,
        ],
    ];
    let zeta = x * u[2];
    let zeta_a = x_a * u[2];
    let mut delta_a = [[0.0; 3]; 3];
    delta_a[0][2] =
        s * (-zeta * z1_a * x - zeta * z1 * x_a + ell_a * zeta * z1 * x - zeta_a * z1 * x);
    delta_a[2][0] = delta_a[0][2];
    delta_a[1][2] =
        s * (-zeta * z2_a * x - zeta * z2 * x_a + ell_a * zeta * z2 * x - zeta_a * z2 * x);
    delta_a[2][1] = delta_a[1][2];
    delta_a[2][2] = alpha_a * x * x + alpha * (2.0 * x_a * x);

    let mut hessian_psi_drift = [[0.0; 3]; 3];
    for i in 0..3 {
        for j in 0..3 {
            hessian_psi_drift[i][j] = r * delta_a[i][j]
                + w * alpha * r_a[i][j]
                + w * h * c_u[i][j]
                + w * (gamma_a[i] * b[j]
                    + b[i] * gamma_a[j]
                    + gamma[i] * c[j]
                    + c[i] * gamma[j]
                    + alpha_a * q_mat[i][j])
                + nu * alpha * (c[i] * b[j] + b[i] * c[j] + h * q_mat[i][j])
                + nu * h * (gamma[i] * b[j] + b[i] * gamma[j])
                + (tau * alpha * h + nu * alpha_a) * b[i] * b[j];
        }
    }

    let _ = (a_ab, ell_ab, x_ab); // kept in scope to emphasize the full q_ij setup.

    ManualRowGeometry {
        objective_psi,
        score_psi,
        hessian_psi,
        hessian_psi_drift,
    }
}

fn expected_fixed_beta_quantity(manual: &ManualRowGeometry, quantity: PsiQuantity) -> f64 {
    match quantity {
        PsiQuantity::Objective => manual.objective_psi,
        PsiQuantity::Score(idx) => manual.score_psi[idx],
        PsiQuantity::Hessian(i, j) => manual.hessian_psi[i][j],
    }
}

fn epshessian_psi_numdual<D: DualNum<f64> + Copy>(
    eps: D,
    psi0: f64,
    p: &CombinedLocationScaleParams,
    i: usize,
    j: usize,
) -> D {
    let beta_1 = D::from(p.beta_1) + D::from(p.u_1) * eps;
    let beta_2 = D::from(p.beta_2) + D::from(p.u_2) * eps;
    let beta_ls = D::from(p.beta_ls) + D::from(p.u_ls) * eps;
    let psi = D::from(psi0);
    let z1 = z1_numdual(psi, p);
    let z2 = z2_numdual(psi, p);
    let x = x_numdual(psi, p);
    let z1_a = D::from(p.z1_1 + p.z1_2 * psi0);
    let z2_a = D::from(p.z2_1 + p.z2_2 * psi0);
    let x_a = D::from(p.x_1 + p.x_2 * psi0);
    let a = beta_1 * z1 + beta_2 * z2;
    let a_a = beta_1 * z1_a + beta_2 * z2_a;
    let ell = beta_ls * x;
    let ell_a = beta_ls * x_a;
    let s = (-ell).exp();
    let q = -a * s;
    let h = -s * a_a - q * ell_a;
    let r = q + D::from(POLY_LOSS_KAPPA) * q.powi(3);
    let w = D::one() + D::from(3.0 * POLY_LOSS_KAPPA) * q * q;
    let nu = D::from(6.0 * POLY_LOSS_KAPPA) * q;
    let b = [-s * z1, -s * z2, -q * x];
    let c = [
        s * (ell_a * z1 - z1_a),
        s * (ell_a * z2 - z2_a),
        -(h * x + q * x_a),
    ];
    let mut q_mat = [[D::zero(); 3]; 3];
    q_mat[0][2] = s * z1 * x;
    q_mat[2][0] = q_mat[0][2];
    q_mat[1][2] = s * z2 * x;
    q_mat[2][1] = q_mat[1][2];
    q_mat[2][2] = q * x * x;
    let mut r_a = [[D::zero(); 3]; 3];
    r_a[0][2] = s * (z1_a * x + z1 * x_a - ell_a * z1 * x);
    r_a[2][0] = r_a[0][2];
    r_a[1][2] = s * (z2_a * x + z2 * x_a - ell_a * z2 * x);
    r_a[2][1] = r_a[1][2];
    r_a[2][2] = h * x * x + q * (D::from(2.0) * x_a * x);
    nu * h * b[i] * b[j] + w * (c[i] * b[j] + b[i] * c[j] + h * q_mat[i][j]) + r * r_a[i][j]
}

fn epshessian_psi_f1(
    eps: F1,
    psi0: f64,
    p: &CombinedLocationScaleParams,
    i: usize,
    j: usize,
) -> F1 {
    let beta_1 = F1::cst(p.beta_1) + F1::cst(p.u_1) * eps;
    let beta_2 = F1::cst(p.beta_2) + F1::cst(p.u_2) * eps;
    let beta_ls = F1::cst(p.beta_ls) + F1::cst(p.u_ls) * eps;
    let psi = F1::cst(psi0);
    let z1 = z1_f1(psi, p);
    let z2 = z2_f1(psi, p);
    let x = x_f1(psi, p);
    let z1_a = F1::cst(p.z1_1 + p.z1_2 * psi0);
    let z2_a = F1::cst(p.z2_1 + p.z2_2 * psi0);
    let x_a = F1::cst(p.x_1 + p.x_2 * psi0);
    let a = beta_1 * z1 + beta_2 * z2;
    let a_a = beta_1 * z1_a + beta_2 * z2_a;
    let ell = beta_ls * x;
    let ell_a = beta_ls * x_a;
    let s = (-ell).exp();
    let q = -a * s;
    let h = -s * a_a - q * ell_a;
    let r = q + F1::cst(POLY_LOSS_KAPPA) * q.powi(3);
    let w = F1::cst(1.0) + F1::cst(3.0 * POLY_LOSS_KAPPA) * q * q;
    let nu = F1::cst(6.0 * POLY_LOSS_KAPPA) * q;
    let b = [-s * z1, -s * z2, -q * x];
    let c = [
        s * (ell_a * z1 - z1_a),
        s * (ell_a * z2 - z2_a),
        -(h * x + q * x_a),
    ];
    let mut q_mat = [[F1::cst(0.0); 3]; 3];
    q_mat[0][2] = s * z1 * x;
    q_mat[2][0] = q_mat[0][2];
    q_mat[1][2] = s * z2 * x;
    q_mat[2][1] = q_mat[1][2];
    q_mat[2][2] = q * x * x;
    let mut r_a = [[F1::cst(0.0); 3]; 3];
    r_a[0][2] = s * (z1_a * x + z1 * x_a - ell_a * z1 * x);
    r_a[2][0] = r_a[0][2];
    r_a[1][2] = s * (z2_a * x + z2 * x_a - ell_a * z2 * x);
    r_a[2][1] = r_a[1][2];
    r_a[2][2] = h * x * x + q * (F1::cst(2.0) * x_a * x);
    nu * h * b[i] * b[j] + w * (c[i] * b[j] + b[i] * c[j] + h * q_mat[i][j]) + r * r_a[i][j]
}

fn epshessian_psi_ad<T: AD>(
    eps: T,
    psi0: f64,
    p: &CombinedLocationScaleParams,
    i: usize,
    j: usize,
) -> T {
    let beta_1 = T::constant(p.beta_1) + T::constant(p.u_1) * eps;
    let beta_2 = T::constant(p.beta_2) + T::constant(p.u_2) * eps;
    let beta_ls = T::constant(p.beta_ls) + T::constant(p.u_ls) * eps;
    let psi = T::constant(psi0);
    let z1 = z1_ad(psi, p);
    let z2 = z2_ad(psi, p);
    let x = x_ad(psi, p);
    let z1_a = T::constant(p.z1_1 + p.z1_2 * psi0);
    let z2_a = T::constant(p.z2_1 + p.z2_2 * psi0);
    let x_a = T::constant(p.x_1 + p.x_2 * psi0);
    let a = beta_1 * z1 + beta_2 * z2;
    let a_a = beta_1 * z1_a + beta_2 * z2_a;
    let ell = beta_ls * x;
    let ell_a = beta_ls * x_a;
    let s = (-ell).exp();
    let q = -a * s;
    let h = -s * a_a - q * ell_a;
    let r = q + T::constant(POLY_LOSS_KAPPA) * q.powi(3);
    let w = T::one() + T::constant(3.0 * POLY_LOSS_KAPPA) * q * q;
    let nu = T::constant(6.0 * POLY_LOSS_KAPPA) * q;
    let b = [-s * z1, -s * z2, q * x];
    let c = [
        s * (ell_a * z1 - z1_a),
        s * (ell_a * z2 - z2_a),
        -(h * x + q * x_a),
    ];
    let mut q_mat = [[T::zero(); 3]; 3];
    q_mat[0][2] = s * z1 * x;
    q_mat[2][0] = q_mat[0][2];
    q_mat[1][2] = s * z2 * x;
    q_mat[2][1] = q_mat[1][2];
    q_mat[2][2] = q * x * x;
    let mut r_a = [[T::zero(); 3]; 3];
    r_a[0][2] = s * (z1_a * x + z1 * x_a - ell_a * z1 * x);
    r_a[2][0] = r_a[0][2];
    r_a[1][2] = s * (z2_a * x + z2 * x_a - ell_a * z2 * x);
    r_a[2][1] = r_a[1][2];
    r_a[2][2] = h * x * x + q * (T::constant(2.0) * x_a * x);
    nu * h * b[i] * b[j] + w * (c[i] * b[j] + b[i] * c[j] + h * q_mat[i][j]) + r * r_a[i][j]
}

#[derive(Clone)]
struct EpsHessianPsiFn<T: AD> {
    params: CombinedLocationScaleParams,
    psi0: f64,
    i: usize,
    j: usize,
    marker: PhantomData<T>,
}

impl<T: AD> EpsHessianPsiFn<T> {
    fn new(params: CombinedLocationScaleParams, psi0: f64, i: usize, j: usize) -> Self {
        Self {
            params,
            psi0,
            i,
            j,
            marker: PhantomData,
        }
    }

    fn to_other_ad_type<T2: AD>(&self) -> EpsHessianPsiFn<T2> {
        EpsHessianPsiFn::new(self.params, self.psi0, self.i, self.j)
    }
}

impl<T: AD> DifferentiableFunctionTrait<T> for EpsHessianPsiFn<T> {
    const NAME: &'static str = "EpsHessianPsiFn";

    fn call(&self, inputs: &[T], _: bool) -> Vec<T> {
        vec![epshessian_psi_ad(
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

fn q_numdual<D: DualNum<f64> + Copy>(eta_ls: D, eta_t: f64) -> D {
    -D::from(eta_t) / eta_ls.exp()
}

fn q_f1(eta_ls: F1, eta_t: f64) -> F1 {
    -F1::cst(eta_t) / eta_ls.exp()
}

fn q_ad<T: AD>(eta_ls: T, eta_t: f64) -> T {
    -T::constant(eta_t) / eta_ls.exp()
}

#[derive(Clone)]
struct QFn<T: AD> {
    eta_t: f64,
    marker: PhantomData<T>,
}

impl<T: AD> QFn<T> {
    fn new(eta_t: f64) -> Self {
        Self {
            eta_t,
            marker: PhantomData,
        }
    }

    fn to_other_ad_type<T2: AD>(&self) -> QFn<T2> {
        QFn {
            eta_t: self.eta_t,
            marker: PhantomData,
        }
    }
}

impl<T: AD> DifferentiableFunctionTrait<T> for QFn<T> {
    const NAME: &'static str = "QFn";

    fn call(&self, inputs: &[T], _: bool) -> Vec<T> {
        vec![q_ad(inputs[0], self.eta_t)]
    }

    fn num_inputs(&self) -> usize {
        1
    }

    fn num_outputs(&self) -> usize {
        1
    }
}

fn scalingphi_numdual<D: DualNum<f64> + Copy>(psi: D, r: f64, eta: f64) -> D {
    let kappa = psi.exp();
    let t = kappa * D::from(r);
    (psi * D::from(eta)).exp() * (D::one() + t * t + t.powi(4))
}

fn scalingphi_f1(psi: F1, r: f64, eta: f64) -> F1 {
    let kappa = psi.exp();
    let t = kappa * F1::cst(r);
    (psi * F1::cst(eta)).exp() * (F1::cst(1.0) + t * t + t.powi(4))
}

fn scalingphi_ad<T: AD>(psi: T, r: f64, eta: f64) -> T {
    let kappa = psi.exp();
    let t = kappa * T::constant(r);
    (psi * T::constant(eta)).exp() * (T::one() + t * t + t.powi(4))
}

#[derive(Clone)]
struct ScalingPhiFn<T: AD> {
    r: f64,
    eta: f64,
    marker: PhantomData<T>,
}

impl<T: AD> ScalingPhiFn<T> {
    fn new(r: f64, eta: f64) -> Self {
        Self {
            r,
            eta,
            marker: PhantomData,
        }
    }

    fn to_other_ad_type<T2: AD>(&self) -> ScalingPhiFn<T2> {
        ScalingPhiFn {
            r: self.r,
            eta: self.eta,
            marker: PhantomData,
        }
    }
}

impl<T: AD> DifferentiableFunctionTrait<T> for ScalingPhiFn<T> {
    const NAME: &'static str = "ScalingPhiFn";

    fn call(&self, inputs: &[T], _: bool) -> Vec<T> {
        vec![scalingphi_ad(inputs[0], self.r, self.eta)]
    }

    fn num_inputs(&self) -> usize {
        1
    }

    fn num_outputs(&self) -> usize {
        1
    }
}

#[test]
fn nonwiggle_q_log_sigma_derivatives_match_three_autodiff_engines() {
    let eta_t = 1.7;
    let points = [-10.0, -4.5, -1.2, 0.0, 0.8, 2.3, 6.0];
    let f_std = QFn::<f64>::new(eta_t);
    let f_ad = f_std.to_other_ad_type::<adfn<1>>();
    let engine = FunctionEngine::new(f_std, f_ad, ForwardAD::new());

    for eta_ls in points {
        let q = -eta_t * (-eta_ls).exp();
        let dq = -q;
        let d2q = q;
        let d3q = -q;

        let (q_nd1, dq_nd1) = first_derivative(|x| q_numdual(x, eta_t), eta_ls);
        let (q_nd2, dq_nd2, d2q_nd) = second_derivative(|x| q_numdual(x, eta_t), eta_ls);
        let (q_nd3, dq_nd3, d2q_nd3, d3q_nd) = third_derivative(|x| q_numdual(x, eta_t), eta_ls);
        let dq_autodiff = diff(|x| q_f1(x, eta_t), eta_ls);
        let (_, jac) = engine.derivative(&[eta_ls]);

        assert_manual_ad_band!("nonwiggle_q", eta_ls, "q", q,
            "num_dual_1" => q_nd1, "num_dual_2" => q_nd2, "num_dual_3" => q_nd3);
        assert_manual_ad_band!("nonwiggle_q", eta_ls, "dq", dq,
            "num_dual_1" => dq_nd1, "num_dual_2" => dq_nd2, "num_dual_3" => dq_nd3,
            "autodiff" => dq_autodiff, "ad_trait" => jac[(0, 0)]);
        assert_manual_ad_band!("nonwiggle_q", eta_ls, "d2q", d2q,
            "num_dual_2" => d2q_nd, "num_dual_3" => d2q_nd3);
        assert_manual_ad_band!("nonwiggle_q", eta_ls, "d3q", d3q, "num_dual_3" => d3q_nd);
    }
}

#[test]
fn spatial_log_kappa_scaling_derivatives_match_three_autodiff_engines() {
    let psi0 = -0.23;
    let r = 0.71;
    let eta = -3.5;
    let kappa = psi0.exp();
    let t = kappa * r;
    let phi = kappa.powf(eta) * (1.0 + t * t + t.powi(4));
    let phi_r = kappa.powf(eta + 1.0) * (2.0 * t + 4.0 * t.powi(3));
    let phi_rr = kappa.powf(eta + 2.0) * (2.0 + 12.0 * t * t);
    let phi_psi = eta * phi + r * phi_r;
    let phi_psi_psi = eta * eta * phi + (2.0 * eta + 1.0) * r * phi_r + r * r * phi_rr;

    let (phi_nd1, phi_psi_nd1) = first_derivative(|x| scalingphi_numdual(x, r, eta), psi0);
    let (phi_nd2, phi_psi_nd2, phi_psi_psi_nd) =
        second_derivative(|x| scalingphi_numdual(x, r, eta), psi0);
    let phi_psi_autodiff = diff(|x| scalingphi_f1(x, r, eta), psi0);

    let f_std = ScalingPhiFn::<f64>::new(r, eta);
    let f_ad = f_std.to_other_ad_type::<adfn<1>>();
    let engine = FunctionEngine::new(f_std, f_ad, ForwardAD::new());
    let (_, jac) = engine.derivative(&[psi0]);

    assert_manual_ad_band!("spatial_log_kappaphi", psi0, "phi", phi,
        "num_dual_1" => phi_nd1, "num_dual_2" => phi_nd2);
    assert_manual_ad_band!("spatial_log_kappaphi", psi0, "phi_psi", phi_psi,
        "num_dual_1" => phi_psi_nd1, "num_dual_2" => phi_psi_nd2,
        "autodiff" => phi_psi_autodiff, "ad_trait" => jac[(0, 0)]);
    assert_manual_ad_band!("spatial_log_kappaphi", psi0, "phi_psi_psi", phi_psi_psi,
        "num_dual" => phi_psi_psi_nd);
}

#[test]
fn combined_location_scale_joint_psi_fixed_beta_terms_match_three_autodiff_engines() {
    let params = CombinedLocationScaleParams {
        beta_1: 0.8,
        beta_2: -0.35,
        beta_ls: -0.45,
        z1_0: 1.2,
        z1_1: -0.4,
        z1_2: 0.15,
        z2_0: -0.6,
        z2_1: 0.7,
        z2_2: -0.2,
        x_0: 0.9,
        x_1: -0.3,
        x_2: 0.1,
        u_1: 0.5,
        u_2: -0.2,
        u_ls: 0.35,
    };
    let points = [-1.1, -0.4, 0.0, 0.6, 1.3];
    let quantities = [
        ("objective_psi", PsiQuantity::Objective),
        ("score_psi_0", PsiQuantity::Score(0)),
        ("score_psi_1", PsiQuantity::Score(1)),
        ("score_psi_2", PsiQuantity::Score(2)),
        ("hessian_psi_00", PsiQuantity::Hessian(0, 0)),
        ("hessian_psi_01", PsiQuantity::Hessian(0, 1)),
        ("hessian_psi_02", PsiQuantity::Hessian(0, 2)),
        ("hessian_psi_11", PsiQuantity::Hessian(1, 1)),
        ("hessian_psi_12", PsiQuantity::Hessian(1, 2)),
        ("hessian_psi_22", PsiQuantity::Hessian(2, 2)),
    ];

    for psi in points {
        let manual = manualrow_geometry(psi, &params);
        for (name, quantity) in quantities {
            let expected = expected_fixed_beta_quantity(&manual, quantity);
            let (_, d1_nd) = first_derivative(|x| psi_quantity_numdual(x, &params, quantity), psi);
            let d1_autodiff = diff(|x| psi_quantity_f1(x, &params, quantity), psi);
            let f_std = PsiQuantityFn::<f64>::new(params, quantity);
            let f_ad = f_std.to_other_ad_type::<adfn<1>>();
            let engine = FunctionEngine::new(f_std, f_ad, ForwardAD::new());
            let (_, jac) = engine.derivative(&[psi]);
            assert_manual_ad_band!(
                "combined_location_scale_joint_psi",
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
fn combined_location_scale_joint_psihessian_drift_matches_three_autodiff_engines() {
    let params = CombinedLocationScaleParams {
        beta_1: 0.8,
        beta_2: -0.35,
        beta_ls: -0.45,
        z1_0: 1.2,
        z1_1: -0.4,
        z1_2: 0.15,
        z2_0: -0.6,
        z2_1: 0.7,
        z2_2: -0.2,
        x_0: 0.9,
        x_1: -0.3,
        x_2: 0.1,
        u_1: 0.5,
        u_2: -0.2,
        u_ls: 0.35,
    };
    let psi0 = 0.37;
    let manual = manualrow_geometry(psi0, &params);
    let pairs = [
        ("T_psi_00", 0usize, 0usize),
        ("T_psi_01", 0, 1),
        ("T_psi_02", 0, 2),
        ("T_psi_11", 1, 1),
        ("T_psi_12", 1, 2),
        ("T_psi_22", 2, 2),
    ];
    for (name, i, j) in pairs {
        let expected = manual.hessian_psi_drift[i][j];
        let (_, d1_nd) =
            first_derivative(|eps| epshessian_psi_numdual(eps, psi0, &params, i, j), 0.0);
        let d1_autodiff = diff(|eps| epshessian_psi_f1(eps, psi0, &params, i, j), 0.0);
        let f_std = EpsHessianPsiFn::<f64>::new(params, psi0, i, j);
        let f_ad = f_std.to_other_ad_type::<adfn<1>>();
        let engine = FunctionEngine::new(f_std, f_ad, ForwardAD::new());
        let (_, jac) = engine.derivative(&[0.0]);
        assert_manual_ad_band!(
            "combined_location_scale_joint_psi_drift",
            psi0,
            name,
            expected,
            "num_dual" => d1_nd,
            "autodiff" => d1_autodiff,
            "ad_trait" => jac[(0, 0)]
        );
    }
}
