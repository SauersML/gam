use ad_trait::AD;
use ad_trait::differentiable_function::{DifferentiableFunctionTrait, ForwardAD};
use ad_trait::forward_ad::adfn::adfn;
use ad_trait::function_engine::FunctionEngine;
use autodiff::{F1, Float, diff};
use gam::estimate::{ExternalOptimOptions, evaluate_external_theta_cost_gradient};
use gam::mixture_link::sas_inverse_link_jet_with_param_partials;
use gam::types::{LikelihoodFamily, SasLinkSpec, SasLinkState};
use ndarray::{Array1, Array2, array};
use num_dual::{DualNum, first_derivative};
use rand::{RngExt, SeedableRng};
use std::marker::PhantomData;

mod common;

const INV_SQRT_2PI: f64 = 0.398_942_280_401_432_7;
const SAS_U_CLAMP: f64 = 50.0;
fn build_tiny_design(n: usize) -> Array2<f64> {
    let mut x = Array2::<f64>::zeros((n, 3));
    for i in 0..n {
        let t = (i as f64 + 0.5) / n as f64;
        let x1 = -1.5 + 3.0 * t;
        x[[i, 0]] = 1.0;
        x[[i, 1]] = x1;
        x[[i, 2]] = (2.1 * x1).sin();
    }
    x
}

fn one_penalty_non_intercept(p: usize) -> Vec<Array2<f64>> {
    let mut s = Array2::<f64>::zeros((p, p));
    for j in 1..p {
        s[[j, j]] = 1.0;
    }
    vec![s]
}

fn central_fd_gradient<F>(theta: &Array1<f64>, mut f: F) -> Array1<f64>
where
    F: FnMut(&Array1<f64>) -> f64,
{
    let mut g = Array1::<f64>::zeros(theta.len());
    for j in 0..theta.len() {
        let h = 1e-4 * (1.0 + theta[j].abs());
        let mut tp = theta.clone();
        let mut tm = theta.clone();
        tp[j] += h;
        tm[j] -= h;
        let fp = f(&tp);
        let fm = f(&tm);
        g[j] = (fp - fm) / (2.0 * h);
    }
    g
}

fn sas_delta_from_raw_log_delta(raw_log_delta: f64) -> f64 {
    SasLinkState::new(0.0, raw_log_delta)
        .expect("sas state")
        .delta
}

fn tanh_bound_numdual<D: DualNum<f64> + Copy>(value: D, bound: f64) -> D {
    D::from(bound) * (value / D::from(bound)).tanh()
}

fn tanh_bound_f1(value: F1, bound: f64) -> F1 {
    F1::cst(bound) * (value / F1::cst(bound)).tanh()
}

fn tanh_bound_ad<T: AD>(value: T, bound: f64) -> T {
    T::constant(bound) * (value / T::constant(bound)).tanh()
}

fn tanh_bound_d1_numdual<D: DualNum<f64> + Copy>(value: D, bound: f64) -> D {
    let t = (value / D::from(bound)).tanh();
    D::one() - t * t
}

fn tanh_bound_d2_numdual<D: DualNum<f64> + Copy>(value: D, bound: f64) -> D {
    let b = D::from(bound);
    let t = (value / b).tanh();
    let s = D::one() - t * t;
    D::from(-2.0) * t * s / b
}

fn tanh_bound_d3_numdual<D: DualNum<f64> + Copy>(value: D, bound: f64) -> D {
    let b = D::from(bound);
    let t = (value / b).tanh();
    let s = D::one() - t * t;
    D::from(-2.0) * s * (D::one() - D::from(3.0) * t * t) / (b * b)
}

fn tanh_bound_d1_f1(value: F1, bound: f64) -> F1 {
    let t = (value / F1::cst(bound)).tanh();
    F1::cst(1.0) - t * t
}

fn tanh_bound_d2_f1(value: F1, bound: f64) -> F1 {
    let b = F1::cst(bound);
    let t = (value / b).tanh();
    let s = F1::cst(1.0) - t * t;
    F1::cst(-2.0) * t * s / b
}

fn tanh_bound_d1_ad<T: AD>(value: T, bound: f64) -> T {
    let t = (value / T::constant(bound)).tanh();
    T::one() - t * t
}

fn tanh_bound_d2_ad<T: AD>(value: T, bound: f64) -> T {
    let b = T::constant(bound);
    let t = (value / b).tanh();
    let s = T::one() - t * t;
    T::constant(-2.0) * t * s / b
}

fn normal_pdf_numdual<D: DualNum<f64> + Copy>(x: D) -> D {
    D::from(INV_SQRT_2PI) * (D::from(-0.5) * x * x).exp()
}

fn normal_pdf_f1(x: F1) -> F1 {
    F1::cst(INV_SQRT_2PI) * (F1::cst(-0.5) * x * x).exp()
}

fn normal_pdf_ad<T: AD>(x: T) -> T {
    T::constant(INV_SQRT_2PI) * (T::constant(-0.5) * x * x).exp()
}

fn sas_eta_d1_numdual<D: DualNum<f64> + Copy>(eta: f64, epsilon: D, log_delta: f64) -> D {
    let a = D::from(eta.asinh());
    let delta = D::from(sas_delta_from_raw_log_delta(log_delta));
    let u_raw = delta * a - epsilon;
    let u = tanh_bound_numdual(u_raw, SAS_U_CLAMP);
    let g1 = tanh_bound_d1_numdual(u_raw, SAS_U_CLAMP);
    let s = u.sinh();
    let c = u.cosh();
    let z = s;
    let q = (eta * eta + 1.0).sqrt();
    let z1 = c * g1 * delta / D::from(q);
    normal_pdf_numdual(z) * z1
}

fn sas_eta_d2_numdual<D: DualNum<f64> + Copy>(eta: f64, epsilon: D, log_delta: f64) -> D {
    let a = D::from(eta.asinh());
    let delta = D::from(sas_delta_from_raw_log_delta(log_delta));
    let u_raw = delta * a - epsilon;
    let u = tanh_bound_numdual(u_raw, SAS_U_CLAMP);
    let g1 = tanh_bound_d1_numdual(u_raw, SAS_U_CLAMP);
    let g2 = tanh_bound_d2_numdual(u_raw, SAS_U_CLAMP);
    let s = u.sinh();
    let c = u.cosh();
    let z = s;
    let q = (eta * eta + 1.0).sqrt();
    let inv_q = 1.0 / q;
    let inv_q3 = inv_q * inv_q * inv_q;
    let r1 = delta * D::from(inv_q);
    let r2 = delta * D::from(-eta * inv_q3);
    let u1 = g1 * r1;
    let u2 = g2 * r1 * r1 + g1 * r2;
    let z1 = c * u1;
    let z2 = s * u1 * u1 + c * u2;
    normal_pdf_numdual(z) * (z2 - z * z1 * z1)
}

fn sas_eta_d3_numdual<D: DualNum<f64> + Copy>(eta: f64, epsilon: D, log_delta: f64) -> D {
    let a = D::from(eta.asinh());
    let delta = D::from(sas_delta_from_raw_log_delta(log_delta));
    let u_raw = delta * a - epsilon;
    let u = tanh_bound_numdual(u_raw, SAS_U_CLAMP);
    let g1 = tanh_bound_d1_numdual(u_raw, SAS_U_CLAMP);
    let g2 = tanh_bound_d2_numdual(u_raw, SAS_U_CLAMP);
    let g3 = tanh_bound_d3_numdual(u_raw, SAS_U_CLAMP);
    let s = u.sinh();
    let c = u.cosh();
    let z = s;
    let q = (eta * eta + 1.0).sqrt();
    let inv_q = 1.0 / q;
    let inv_q3 = inv_q * inv_q * inv_q;
    let inv_q5 = inv_q3 * inv_q * inv_q;
    let r1 = delta * D::from(inv_q);
    let r2 = delta * D::from(-eta * inv_q3);
    let r3 = delta * D::from((2.0 * eta * eta - 1.0) * inv_q5);
    let u1 = g1 * r1;
    let u2 = g2 * r1 * r1 + g1 * r2;
    let u3 = g3 * r1 * r1 * r1 + D::from(3.0) * g2 * r1 * r2 + g1 * r3;
    let z1 = c * u1;
    let z2 = s * u1 * u1 + c * u2;
    let z3 = c * u1 * u1 * u1 + D::from(3.0) * s * u1 * u2 + c * u3;
    let phi = normal_pdf_numdual(z);
    phi * (z3 - D::from(3.0) * z * z1 * z2 + (z * z - D::one()) * z1 * z1 * z1)
}

fn sas_eta_d1_f1(eta: f64, epsilon: F1, log_delta: f64) -> F1 {
    let a = F1::cst(eta.asinh());
    let delta = F1::cst(sas_delta_from_raw_log_delta(log_delta));
    let u_raw = delta * a - epsilon;
    let u = tanh_bound_f1(u_raw, SAS_U_CLAMP);
    let g1 = tanh_bound_d1_f1(u_raw, SAS_U_CLAMP);
    let c = u.cosh();
    let z = u.sinh();
    let q = (eta * eta + 1.0).sqrt();
    let z1 = c * g1 * delta / F1::cst(q);
    normal_pdf_f1(z) * z1
}

fn sas_eta_d2_f1(eta: f64, epsilon: F1, log_delta: f64) -> F1 {
    let a = F1::cst(eta.asinh());
    let delta = F1::cst(sas_delta_from_raw_log_delta(log_delta));
    let u_raw = delta * a - epsilon;
    let u = tanh_bound_f1(u_raw, SAS_U_CLAMP);
    let g1 = tanh_bound_d1_f1(u_raw, SAS_U_CLAMP);
    let g2 = tanh_bound_d2_f1(u_raw, SAS_U_CLAMP);
    let s = u.sinh();
    let c = u.cosh();
    let z = s;
    let q = (eta * eta + 1.0).sqrt();
    let inv_q = 1.0 / q;
    let inv_q3 = inv_q * inv_q * inv_q;
    let r1 = delta * F1::cst(inv_q);
    let r2 = delta * F1::cst(-eta * inv_q3);
    let u1 = g1 * r1;
    let u2 = g2 * r1 * r1 + g1 * r2;
    let z1 = c * u1;
    let z2 = s * u1 * u1 + c * u2;
    normal_pdf_f1(z) * (z2 - z * z1 * z1)
}

fn sas_eta_d1_ad<T: AD>(eta: f64, epsilon: T, log_delta: f64) -> T {
    let a = T::constant(eta.asinh());
    let delta = T::constant(sas_delta_from_raw_log_delta(log_delta));
    let u_raw = delta * a - epsilon;
    let u = tanh_bound_ad(u_raw, SAS_U_CLAMP);
    let g1 = tanh_bound_d1_ad(u_raw, SAS_U_CLAMP);
    let c = u.cosh();
    let z = u.sinh();
    let q = (eta * eta + 1.0).sqrt();
    let z1 = c * g1 * delta / T::constant(q);
    normal_pdf_ad(z) * z1
}

fn sas_eta_d2_ad<T: AD>(eta: f64, epsilon: T, log_delta: f64) -> T {
    let a = T::constant(eta.asinh());
    let delta = T::constant(sas_delta_from_raw_log_delta(log_delta));
    let u_raw = delta * a - epsilon;
    let u = tanh_bound_ad(u_raw, SAS_U_CLAMP);
    let g1 = tanh_bound_d1_ad(u_raw, SAS_U_CLAMP);
    let g2 = tanh_bound_d2_ad(u_raw, SAS_U_CLAMP);
    let s = u.sinh();
    let c = u.cosh();
    let z = s;
    let q = (eta * eta + 1.0).sqrt();
    let inv_q = 1.0 / q;
    let inv_q3 = inv_q * inv_q * inv_q;
    let r1 = delta * T::constant(inv_q);
    let r2 = delta * T::constant(-eta * inv_q3);
    let u1 = g1 * r1;
    let u2 = g2 * r1 * r1 + g1 * r2;
    let z1 = c * u1;
    let z2 = s * u1 * u1 + c * u2;
    normal_pdf_ad(z) * (z2 - z * z1 * z1)
}

#[derive(Clone)]
struct SasD1Fn<T: AD> {
    eta: f64,
    log_delta: f64,
    _marker: PhantomData<T>,
}

impl<T: AD> SasD1Fn<T> {
    fn new(eta: f64, log_delta: f64) -> Self {
        Self {
            eta,
            log_delta,
            _marker: PhantomData,
        }
    }

    fn to_other_ad_type<T2: AD>(&self) -> SasD1Fn<T2> {
        SasD1Fn::new(self.eta, self.log_delta)
    }
}

impl<T: AD> DifferentiableFunctionTrait<T> for SasD1Fn<T> {
    const NAME: &'static str = "SasD1Fn";

    fn call(&self, inputs: &[T], _freeze: bool) -> Vec<T> {
        vec![sas_eta_d1_ad(self.eta, inputs[0], self.log_delta)]
    }

    fn num_inputs(&self) -> usize {
        1
    }

    fn num_outputs(&self) -> usize {
        1
    }
}

#[derive(Clone)]
struct SasD2Fn<T: AD> {
    eta: f64,
    log_delta: f64,
    _marker: PhantomData<T>,
}

impl<T: AD> SasD2Fn<T> {
    fn new(eta: f64, log_delta: f64) -> Self {
        Self {
            eta,
            log_delta,
            _marker: PhantomData,
        }
    }

    fn to_other_ad_type<T2: AD>(&self) -> SasD2Fn<T2> {
        SasD2Fn::new(self.eta, self.log_delta)
    }
}

impl<T: AD> DifferentiableFunctionTrait<T> for SasD2Fn<T> {
    const NAME: &'static str = "SasD2Fn";

    fn call(&self, inputs: &[T], _freeze: bool) -> Vec<T> {
        vec![sas_eta_d2_ad(self.eta, inputs[0], self.log_delta)]
    }

    fn num_inputs(&self) -> usize {
        1
    }

    fn num_outputs(&self) -> usize {
        1
    }
}

#[test]
fn sas_epsilon_eta_derivative_partials_match_three_autodiff_engines() {
    let cases = [
        (-1.2, -0.4, -0.3),
        (-0.5, 0.2, 0.1),
        (0.0, -0.2, 0.0),
        (0.8, 0.3, -0.2),
        (1.4, -0.1, 0.25),
    ];

    for (eta, epsilon, log_delta) in cases {
        let out = sas_inverse_link_jet_with_param_partials(eta, epsilon, log_delta);
        let h = 1e-6;
        let ep_p = gam::mixture_link::sas_inverse_link_jet(eta, epsilon + h, log_delta);
        let ep_m = gam::mixture_link::sas_inverse_link_jet(eta, epsilon - h, log_delta);
        let fd_d1 = (ep_p.d1 - ep_m.d1) / (2.0 * h);

        let (_d1_val, d1_nd) =
            first_derivative(|eps| sas_eta_d1_numdual(eta, eps, log_delta), epsilon);
        let (_d2_val, d2_nd) =
            first_derivative(|eps| sas_eta_d2_numdual(eta, eps, log_delta), epsilon);
        let (_d3_val, d3_nd) =
            first_derivative(|eps| sas_eta_d3_numdual(eta, eps, log_delta), epsilon);

        let d1_autodiff = diff(|eps| sas_eta_d1_f1(eta, eps, log_delta), epsilon);
        let d2_autodiff = diff(|eps| sas_eta_d2_f1(eta, eps, log_delta), epsilon);

        let d1_std = SasD1Fn::<f64>::new(eta, log_delta);
        let d1_ad = d1_std.to_other_ad_type::<adfn<1>>();
        let d1_engine = FunctionEngine::new(d1_std, d1_ad, ForwardAD::new());
        let (_v1, jac1) = d1_engine.derivative(&[epsilon]);

        let d2_std = SasD2Fn::<f64>::new(eta, log_delta);
        let d2_ad = d2_std.to_other_ad_type::<adfn<1>>();
        let d2_engine = FunctionEngine::new(d2_std, d2_ad, ForwardAD::new());
        let (_v2, jac2) = d2_engine.derivative(&[epsilon]);

        if (eta, epsilon, log_delta) == (-1.2, -0.4, -0.3) {
            println!(
                "debug_sas_eps_d1 eta={eta} eps={epsilon} ld={log_delta} manual={:.16e} fd={:.16e} num_dual={:.16e} autodiff={:.16e} ad_trait={:.16e}",
                out.djet_depsilon.d1,
                fd_d1,
                d1_nd,
                d1_autodiff,
                jac1[(0, 0)]
            );
        }

        assert_manual_ad_band!(
            "sas_djet_depsilon",
            eta,
            "d1",
            out.djet_depsilon.d1,
            "num_dual" => d1_nd,
            "autodiff" => d1_autodiff,
            "ad_trait" => jac1[(0, 0)]
        );
        assert_manual_ad_band!(
            "sas_djet_depsilon",
            eta,
            "d2",
            out.djet_depsilon.d2,
            "num_dual" => d2_nd,
            "autodiff" => d2_autodiff,
            "ad_trait" => jac2[(0, 0)]
        );
        assert_manual_ad_band!(
            "sas_djet_depsilon",
            eta,
            "d3",
            out.djet_depsilon.d3,
            "num_dual" => d3_nd
        );
    }
}

#[test]
fn sas_epsilon_partial_fd_at_problem_point() {
    let eta = -1.2;
    let epsilon = -0.4;
    let log_delta = -0.3;
    let out = sas_inverse_link_jet_with_param_partials(eta, epsilon, log_delta);
    let h = 1e-6;

    let ep_p = gam::mixture_link::sas_inverse_link_jet(eta, epsilon + h, log_delta);
    let ep_m = gam::mixture_link::sas_inverse_link_jet(eta, epsilon - h, log_delta);
    let fd_ep = (
        (ep_p.mu - ep_m.mu) / (2.0 * h),
        (ep_p.d1 - ep_m.d1) / (2.0 * h),
        (ep_p.d2 - ep_m.d2) / (2.0 * h),
        (ep_p.d3 - ep_m.d3) / (2.0 * h),
    );

    assert!(
        (out.djet_depsilon.mu - fd_ep.0).abs() < 5e-5,
        "mu partial mismatch at problem point: analytic={:.16e} fd={:.16e}",
        out.djet_depsilon.mu,
        fd_ep.0
    );
    assert!(
        (out.djet_depsilon.d1 - fd_ep.1).abs() < 5e-5,
        "d1 partial mismatch at problem point: analytic={:.16e} fd={:.16e}",
        out.djet_depsilon.d1,
        fd_ep.1
    );
    assert!(
        (out.djet_depsilon.d2 - fd_ep.2).abs() < 5e-5,
        "d2 partial mismatch at problem point: analytic={:.16e} fd={:.16e}",
        out.djet_depsilon.d2,
        fd_ep.2
    );
    assert!(
        (out.djet_depsilon.d3 - fd_ep.3).abs() < 5e-4,
        "d3 partial mismatch at problem point: analytic={:.16e} fd={:.16e}",
        out.djet_depsilon.d3,
        fd_ep.3
    );
}

#[test]
fn sas_exact_outer_gradient_seed19_epsilon_component_matches_profiled_fd() {
    let seed = 19_u64;
    let n = 20usize;
    let x = build_tiny_design(n);
    let w = Array1::<f64>::ones(n);
    let offset = Array1::<f64>::zeros(n);
    let s_list = one_penalty_non_intercept(x.ncols());

    let true_beta = array![-0.2, 0.9, -0.4];
    let eta = x.dot(&true_beta);
    let eps_true = 0.25;
    let ld_true = -0.20;
    let p = eta.mapv(|e| gam::mixture_link::sas_inverse_link_jet(e, eps_true, ld_true).mu);
    let mut rng = rand::rngs::StdRng::seed_from_u64(seed);
    let y = p.mapv(|pi| if rng.random::<f64>() < pi { 1.0 } else { 0.0 });

    let opts = ExternalOptimOptions {
        family: LikelihoodFamily::BinomialSas,
        mixture_link: None,
        optimize_mixture: false,
        sas_link: Some(SasLinkSpec {
            initial_epsilon: 0.0,
            initial_log_delta: 0.0,
        }),
        optimize_sas: true,
        max_iter: 80,
        tol: 1e-7,
        nullspace_dims: vec![1],
        linear_constraints: None,
        firth_bias_reduction: None,
    };

    let theta = array![0.10, 0.12, -0.18];
    let (_cost, analytic) = evaluate_external_theta_cost_gradient(
        y.view(),
        w.view(),
        x.view(),
        offset.view(),
        s_list.clone(),
        &theta,
        &opts,
    )
    .expect("analytic theta-gradient");

    let fd = central_fd_gradient(&theta, |t| {
        evaluate_external_theta_cost_gradient(
            y.view(),
            w.view(),
            x.view(),
            offset.view(),
            s_list.clone(),
            t,
            &opts,
        )
        .expect("fd theta cost")
        .0
    });

    let j = 1usize;
    let abs_err = (analytic[j] - fd[j]).abs();
    let scale = analytic[j].abs().max(fd[j].abs()).max(1e-5);
    let rel_err = abs_err / scale;
    assert!(
        abs_err < 3e-2 || rel_err < 1.5e-1,
        "SAS exact epsilon hypergradient disagrees with the profiled objective FD at seed={seed}: analytic={:.6e} fd={:.6e} abs={:.3e} rel={:.3e}. \
This indicates the assembled outer epsilon hypergradient is no longer exact.",
        analytic[j],
        fd[j],
        abs_err,
        rel_err
    );
}
