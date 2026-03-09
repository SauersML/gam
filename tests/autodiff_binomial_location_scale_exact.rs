use ad_trait::AD;
use ad_trait::differentiable_function::{DifferentiableFunctionTrait, ForwardAD};
use ad_trait::forward_ad::adfn::adfn;
use ad_trait::function_engine::FunctionEngine;
use autodiff::{F1, Float, diff};
use num_dual::{DualNum, first_derivative, second_derivative, third_derivative};
use std::marker::PhantomData;

mod common;

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
    _marker: PhantomData<T>,
}

impl<T: AD> QFn<T> {
    fn new(eta_t: f64) -> Self {
        Self {
            eta_t,
            _marker: PhantomData,
        }
    }

    fn to_other_ad_type<T2: AD>(&self) -> QFn<T2> {
        QFn {
            eta_t: self.eta_t,
            _marker: PhantomData,
        }
    }
}

impl<T: AD> DifferentiableFunctionTrait<T> for QFn<T> {
    const NAME: &'static str = "QFn";

    fn call(&self, inputs: &[T], _freeze: bool) -> Vec<T> {
        vec![q_ad(inputs[0], self.eta_t)]
    }

    fn num_inputs(&self) -> usize {
        1
    }

    fn num_outputs(&self) -> usize {
        1
    }
}

fn scaling_phi_numdual<D: DualNum<f64> + Copy>(psi: D, r: f64, eta: f64) -> D {
    let kappa = psi.exp();
    let t = kappa * D::from(r);
    (psi * D::from(eta)).exp() * (D::one() + t * t + t.powi(4))
}

fn scaling_phi_f1(psi: F1, r: f64, eta: f64) -> F1 {
    let kappa = psi.exp();
    let t = kappa * F1::cst(r);
    (psi * F1::cst(eta)).exp() * (F1::cst(1.0) + t * t + t.powi(4))
}

fn scaling_phi_ad<T: AD>(psi: T, r: f64, eta: f64) -> T {
    let kappa = psi.exp();
    let t = kappa * T::constant(r);
    (psi * T::constant(eta)).exp() * (T::one() + t * t + t.powi(4))
}

#[derive(Clone)]
struct ScalingPhiFn<T: AD> {
    r: f64,
    eta: f64,
    _marker: PhantomData<T>,
}

impl<T: AD> ScalingPhiFn<T> {
    fn new(r: f64, eta: f64) -> Self {
        Self {
            r,
            eta,
            _marker: PhantomData,
        }
    }

    fn to_other_ad_type<T2: AD>(&self) -> ScalingPhiFn<T2> {
        ScalingPhiFn {
            r: self.r,
            eta: self.eta,
            _marker: PhantomData,
        }
    }
}

impl<T: AD> DifferentiableFunctionTrait<T> for ScalingPhiFn<T> {
    const NAME: &'static str = "ScalingPhiFn";

    fn call(&self, inputs: &[T], _freeze: bool) -> Vec<T> {
        vec![scaling_phi_ad(inputs[0], self.r, self.eta)]
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
        let (_value, jac) = engine.derivative(&[eta_ls]);

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

    let (phi_nd1, phi_psi_nd1) = first_derivative(|x| scaling_phi_numdual(x, r, eta), psi0);
    let (phi_nd2, phi_psi_nd2, phi_psi_psi_nd) =
        second_derivative(|x| scaling_phi_numdual(x, r, eta), psi0);
    let phi_psi_autodiff = diff(|x| scaling_phi_f1(x, r, eta), psi0);

    let f_std = ScalingPhiFn::<f64>::new(r, eta);
    let f_ad = f_std.to_other_ad_type::<adfn<1>>();
    let engine = FunctionEngine::new(f_std, f_ad, ForwardAD::new());
    let (_value, jac) = engine.derivative(&[psi0]);

    assert_manual_ad_band!("spatial_log_kappa_phi", psi0, "phi", phi,
        "num_dual_1" => phi_nd1, "num_dual_2" => phi_nd2);
    assert_manual_ad_band!("spatial_log_kappa_phi", psi0, "phi_psi", phi_psi,
        "num_dual_1" => phi_psi_nd1, "num_dual_2" => phi_psi_nd2,
        "autodiff" => phi_psi_autodiff, "ad_trait" => jac[(0, 0)]);
    assert_manual_ad_band!("spatial_log_kappa_phi", psi0, "phi_psi_psi", phi_psi_psi,
        "num_dual" => phi_psi_psi_nd);
}
