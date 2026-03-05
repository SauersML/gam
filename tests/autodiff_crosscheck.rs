use ad_trait::AD;
use ad_trait::differentiable_function::{DifferentiableFunctionTrait, ForwardAD};
use ad_trait::forward_ad::adfn::adfn;
use ad_trait::function_engine::FunctionEngine;
use approx::assert_abs_diff_eq;
use autodiff::{F1, Float, diff};
use gam::families::sigma_link::{
    bounded_sigma_derivs_up_to_fourth_scalar, bounded_sigma_derivs_up_to_third_scalar,
};
use num_dual::{DualNum, first_derivative, second_derivative, third_derivative};
use std::marker::PhantomData;

#[derive(Clone, Debug)]
struct E2EData {
    a: Vec<f64>,
    b: Vec<f64>,
    y: Vec<f64>,
    w: Vec<f64>,
    sigma_min: f64,
    sigma_max: f64,
}

fn sigma_unclamped_numdual<D: DualNum<f64> + Copy>(eta: D, sigma_min: f64, sigma_max: f64) -> D {
    let span = (sigma_max - sigma_min).max(1e-12);
    let one = D::one();
    let p = one / (one + (-eta).exp());
    D::from(sigma_min) + D::from(span) * p
}

fn e2e_objective_numdual<D: DualNum<f64> + Copy>(theta: D, data: &E2EData) -> D {
    let mut acc = D::zero();
    for i in 0..data.a.len() {
        let eta = theta * D::from(data.a[i]) + D::from(data.b[i]);
        let s = sigma_unclamped_numdual(eta, data.sigma_min, data.sigma_max);
        let y = D::from(data.y[i]);
        let half = D::from(0.5);
        let phi = s.ln() + half * (y / s) * (y / s);
        acc += D::from(data.w[i]) * phi;
    }
    acc
}

fn e2e_objective_f64(theta: f64, data: &E2EData) -> f64 {
    let mut acc = 0.0;
    let span = (data.sigma_max - data.sigma_min).max(1e-12);
    for i in 0..data.a.len() {
        let eta = theta * data.a[i] + data.b[i];
        let p = 1.0 / (1.0 + (-eta).exp());
        let s = data.sigma_min + span * p;
        let y = data.y[i];
        let phi = s.ln() + 0.5 * (y / s) * (y / s);
        acc += data.w[i] * phi;
    }
    acc
}

fn e2e_manual_derivatives(theta: f64, data: &E2EData) -> (f64, f64, f64, f64) {
    let mut v = 0.0;
    let mut d1 = 0.0;
    let mut d2 = 0.0;
    let mut d3 = 0.0;

    for i in 0..data.a.len() {
        let eta = theta * data.a[i] + data.b[i];
        let (s, s1, s2, s3, _) =
            bounded_sigma_derivs_up_to_fourth_scalar(eta, data.sigma_min, data.sigma_max);

        let y = data.y[i];
        let y2 = y * y;

        let f1 = 1.0 / s - y2 / (s * s * s);
        let f2 = -1.0 / (s * s) + 3.0 * y2 / (s * s * s * s);
        let f3 = 2.0 / (s * s * s) - 12.0 * y2 / (s * s * s * s * s);

        let ai = data.a[i];
        let wi = data.w[i];

        let s_theta_1 = s1 * ai;
        let s_theta_2 = s2 * ai * ai;
        let s_theta_3 = s3 * ai * ai * ai;

        let phi = s.ln() + 0.5 * (y / s) * (y / s);
        v += wi * phi;
        d1 += wi * f1 * s_theta_1;
        d2 += wi * (f2 * s_theta_1 * s_theta_1 + f1 * s_theta_2);
        d3 += wi
            * (f3 * s_theta_1 * s_theta_1 * s_theta_1
                + 3.0 * f2 * s_theta_1 * s_theta_2
                + f1 * s_theta_3);
    }

    (v, d1, d2, d3)
}

#[derive(Clone)]
struct SigmaFn<T: AD> {
    sigma_min: T,
    span: T,
}

impl<T: AD> SigmaFn<T> {
    fn new(sigma_min: f64, sigma_max: f64) -> Self {
        Self {
            sigma_min: T::constant(sigma_min),
            span: T::constant((sigma_max - sigma_min).max(1e-12)),
        }
    }

    fn to_other_ad_type<T2: AD>(&self) -> SigmaFn<T2> {
        SigmaFn {
            sigma_min: self.sigma_min.to_other_ad_type::<T2>(),
            span: self.span.to_other_ad_type::<T2>(),
        }
    }
}

impl<T: AD> DifferentiableFunctionTrait<T> for SigmaFn<T> {
    const NAME: &'static str = "SigmaFn";

    fn call(&self, inputs: &[T], _freeze: bool) -> Vec<T> {
        let eta = inputs[0];
        let p = T::one() / (T::one() + (-eta).exp());
        vec![self.sigma_min + self.span * p]
    }

    fn num_inputs(&self) -> usize {
        1
    }

    fn num_outputs(&self) -> usize {
        1
    }
}

#[derive(Clone)]
struct E2EFn<T: AD> {
    data: E2EData,
    _marker: PhantomData<T>,
}

impl<T: AD> E2EFn<T> {
    fn new(data: E2EData) -> Self {
        Self {
            data,
            _marker: PhantomData,
        }
    }

    fn to_other_ad_type<T2: AD>(&self) -> E2EFn<T2> {
        E2EFn {
            data: self.data.clone(),
            _marker: PhantomData,
        }
    }
}

impl<T: AD> DifferentiableFunctionTrait<T> for E2EFn<T> {
    const NAME: &'static str = "E2EFn";

    fn call(&self, inputs: &[T], _freeze: bool) -> Vec<T> {
        let theta = inputs[0];
        let sigma_min = T::constant(self.data.sigma_min);
        let span = T::constant((self.data.sigma_max - self.data.sigma_min).max(1e-12));
        let mut acc = T::zero();

        for i in 0..self.data.a.len() {
            let eta = theta * T::constant(self.data.a[i]) + T::constant(self.data.b[i]);
            let p = T::one() / (T::one() + (-eta).exp());
            let s = sigma_min + span * p;
            let y = T::constant(self.data.y[i]);
            let w = T::constant(self.data.w[i]);
            let phi = s.ln() + T::constant(0.5) * (y / s) * (y / s);
            acc += w * phi;
        }

        vec![acc]
    }

    fn num_inputs(&self) -> usize {
        1
    }

    fn num_outputs(&self) -> usize {
        1
    }
}

#[test]
fn sigma_manual_matches_num_dual_first_through_third() {
    let sigma_min = 0.05;
    let sigma_max = 12.0;
    let points = [-7.0, -3.0, -1.25, -0.2, 0.0, 0.4, 1.7, 3.2, 6.5];

    for eta in points {
        let (s, d1, d2, d3) = bounded_sigma_derivs_up_to_third_scalar(eta, sigma_min, sigma_max);

        let (s_ad, d1_ad) =
            first_derivative(|x| sigma_unclamped_numdual(x, sigma_min, sigma_max), eta);
        let (s2_ad, d1_2_ad, d2_ad) =
            second_derivative(|x| sigma_unclamped_numdual(x, sigma_min, sigma_max), eta);
        let (s3_ad, d1_3_ad, d2_3_ad, d3_ad) =
            third_derivative(|x| sigma_unclamped_numdual(x, sigma_min, sigma_max), eta);

        assert_abs_diff_eq!(s, s_ad, epsilon = 1e-12);
        assert_abs_diff_eq!(s, s2_ad, epsilon = 1e-12);
        assert_abs_diff_eq!(s, s3_ad, epsilon = 1e-12);

        assert_abs_diff_eq!(d1, d1_ad, epsilon = 1e-12);
        assert_abs_diff_eq!(d1, d1_2_ad, epsilon = 1e-12);
        assert_abs_diff_eq!(d1, d1_3_ad, epsilon = 1e-12);

        assert_abs_diff_eq!(d2, d2_ad, epsilon = 1e-11);
        assert_abs_diff_eq!(d2, d2_3_ad, epsilon = 1e-11);

        assert_abs_diff_eq!(d3, d3_ad, epsilon = 1e-10);
    }
}

#[test]
fn sigma_manual_matches_autodiff_forward_mode_first_derivative() {
    let sigma_min: f64 = 0.1;
    let sigma_max: f64 = 6.2;
    let span = (sigma_max - sigma_min).max(1e-12);
    let points = [-6.0, -2.0, -0.3, 0.3, 1.4, 4.0, 6.0];

    for eta in points {
        let (_, d1, _, _) = bounded_sigma_derivs_up_to_third_scalar(eta, sigma_min, sigma_max);

        let d1_ad = diff(
            |x: F1| {
                let p = F1::cst(1.0) / (F1::cst(1.0) + (-x).exp());
                F1::cst(sigma_min) + F1::cst(span) * p
            },
            eta,
        );

        assert_abs_diff_eq!(d1, d1_ad, epsilon = 1e-12);
    }
}

#[test]
fn sigma_manual_matches_ad_trait_forward_mode_first_derivative() {
    let sigma_min = 0.2;
    let sigma_max = 4.4;
    let points = [-5.0, -1.1, -0.1, 0.7, 1.8, 3.0, 5.0];

    let f_std = SigmaFn::<f64>::new(sigma_min, sigma_max);
    let f_ad = f_std.to_other_ad_type::<adfn<1>>();
    let engine = FunctionEngine::new(f_std, f_ad, ForwardAD::new());

    for eta in points {
        let (_, d1, _, _) = bounded_sigma_derivs_up_to_third_scalar(eta, sigma_min, sigma_max);
        let (_value, jac) = engine.derivative(&[eta]);
        assert_abs_diff_eq!(d1, jac[(0, 0)], epsilon = 1e-12);
    }
}

#[test]
fn e2e_manual_derivatives_match_num_dual_and_first_order_ad_engines() {
    let data = E2EData {
        a: vec![0.4, -1.3, 0.8, 1.7, -0.9, 0.2],
        b: vec![-0.2, 1.1, -1.4, 0.9, 0.3, -0.7],
        y: vec![0.5, -1.2, 0.9, 1.7, -0.4, 0.2],
        w: vec![1.0, 0.8, 1.3, 0.7, 1.1, 0.9],
        sigma_min: 0.15,
        sigma_max: 5.5,
    };

    let f_std = E2EFn::<f64>::new(data.clone());
    let f_ad = f_std.to_other_ad_type::<adfn<1>>();
    let engine = FunctionEngine::new(f_std, f_ad, ForwardAD::new());

    let points = [-1.5, -0.9, -0.2, 0.0, 0.35, 0.9, 1.6];

    for theta in points {
        let (v_manual, d1_manual, d2_manual, d3_manual) = e2e_manual_derivatives(theta, &data);

        let (v_nd, d1_nd) = first_derivative(|t| e2e_objective_numdual(t, &data), theta);
        let (_v_nd2, _d1_nd2, d2_nd) =
            second_derivative(|t| e2e_objective_numdual(t, &data), theta);
        let (_v_nd3, _d1_nd3, _d2_nd3, d3_nd) =
            third_derivative(|t| e2e_objective_numdual(t, &data), theta);

        let d1_autodiff = diff(
            |x: F1| {
                let span = data.sigma_max - data.sigma_min;
                let mut acc = F1::cst(0.0);
                for i in 0..data.a.len() {
                    let eta = x * data.a[i] + data.b[i];
                    let p = F1::cst(1.0) / (F1::cst(1.0) + (-eta).exp());
                    let s = F1::cst(data.sigma_min) + F1::cst(span) * p;
                    let y = data.y[i];
                    let phi = s.ln() + F1::cst(0.5) * (y / s) * (y / s);
                    acc += data.w[i] * phi;
                }
                acc
            },
            theta,
        );

        let (values_ad_trait, jac_ad_trait) = engine.derivative(&[theta]);
        let d1_ad_trait = jac_ad_trait[(0, 0)];

        assert_abs_diff_eq!(v_manual, e2e_objective_f64(theta, &data), epsilon = 1e-12);
        assert_abs_diff_eq!(v_manual, v_nd, epsilon = 1e-12);
        assert_abs_diff_eq!(v_manual, values_ad_trait[0], epsilon = 1e-12);

        assert_abs_diff_eq!(d1_manual, d1_nd, epsilon = 1e-11);
        assert_abs_diff_eq!(d1_manual, d1_autodiff, epsilon = 1e-10);
        assert_abs_diff_eq!(d1_manual, d1_ad_trait, epsilon = 1e-10);

        assert_abs_diff_eq!(d2_manual, d2_nd, epsilon = 1e-9);
        assert_abs_diff_eq!(d3_manual, d3_nd, epsilon = 1e-8);
    }
}
