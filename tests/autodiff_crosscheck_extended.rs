use ad_trait::AD;
use ad_trait::differentiable_function::{DifferentiableFunctionTrait, ForwardAD};
use ad_trait::forward_ad::adfn::adfn;
use ad_trait::function_engine::FunctionEngine;
use approx::assert_abs_diff_eq;
use autodiff::{F1, Float, diff};
use gam::mixture_link::component_inverse_link_jet;
use gam::types::LinkComponent;
use num_dual::{DualNum, first_derivative, second_derivative, third_derivative};
use std::marker::PhantomData;

const INV_SQRT_2PI: f64 = 0.398_942_280_401_432_7;

fn normal_pdf_numdual<D: DualNum<f64> + Copy>(x: D) -> D {
    D::from(INV_SQRT_2PI) * (D::from(-0.5) * x * x).exp()
}

fn normal_pdf_f1(x: F1) -> F1 {
    F1::cst(INV_SQRT_2PI) * (F1::cst(-0.5) * x * x).exp()
}

fn normal_pdf_ad<T: AD>(x: T) -> T {
    T::constant(INV_SQRT_2PI) * (T::constant(-0.5) * x * x).exp()
}

fn logistic_numdual<D: DualNum<f64> + Copy>(x: D) -> D {
    D::one() / (D::one() + (-x).exp())
}

fn logistic_f1(x: F1) -> F1 {
    F1::cst(1.0) / (F1::cst(1.0) + (-x).exp())
}

fn logistic_ad<T: AD>(x: T) -> T {
    T::one() / (T::one() + (-x).exp())
}

fn weight_numdual<D: DualNum<f64> + Copy>(x: D) -> D {
    let m = logistic_numdual(x);
    m * (D::one() - m)
}

fn weight_f1(x: F1) -> F1 {
    let m = logistic_f1(x);
    m * (F1::cst(1.0) - m)
}

fn weight_ad<T: AD>(x: T) -> T {
    let m = logistic_ad(x);
    m * (T::one() - m)
}

fn assert_close_ctx(
    case: &str,
    eta: f64,
    quantity: &str,
    lhs_name: &str,
    lhs: f64,
    rhs_name: &str,
    rhs: f64,
    abs_tol: f64,
) {
    let abs = (lhs - rhs).abs();
    if abs > abs_tol {
        let rel = abs / rhs.abs().max(1e-16);
        panic!(
            "{case} eta={eta:.6} {quantity}: {lhs_name}={lhs:.16e} {rhs_name}={rhs:.16e} abs_err={abs:.3e} rel_err={rel:.3e} tol={abs_tol:.3e}"
        );
    }
}

#[derive(Clone)]
struct PdfFn<T: AD> {
    _marker: PhantomData<T>,
}

impl<T: AD> PdfFn<T> {
    fn new() -> Self {
        Self {
            _marker: PhantomData,
        }
    }

    fn to_other_ad_type<T2: AD>(&self) -> PdfFn<T2> {
        PdfFn::new()
    }
}

impl<T: AD> DifferentiableFunctionTrait<T> for PdfFn<T> {
    const NAME: &'static str = "PdfFn";

    fn call(&self, inputs: &[T], _freeze: bool) -> Vec<T> {
        vec![normal_pdf_ad(inputs[0])]
    }

    fn num_inputs(&self) -> usize {
        1
    }

    fn num_outputs(&self) -> usize {
        1
    }
}

#[derive(Clone)]
struct LogitFn<T: AD> {
    _marker: PhantomData<T>,
}

impl<T: AD> LogitFn<T> {
    fn new() -> Self {
        Self {
            _marker: PhantomData,
        }
    }

    fn to_other_ad_type<T2: AD>(&self) -> LogitFn<T2> {
        LogitFn::new()
    }
}

impl<T: AD> DifferentiableFunctionTrait<T> for LogitFn<T> {
    const NAME: &'static str = "LogitFn";

    fn call(&self, inputs: &[T], _freeze: bool) -> Vec<T> {
        vec![logistic_ad(inputs[0])]
    }

    fn num_inputs(&self) -> usize {
        1
    }

    fn num_outputs(&self) -> usize {
        1
    }
}

#[derive(Clone)]
struct WeightFn<T: AD> {
    _marker: PhantomData<T>,
}

impl<T: AD> WeightFn<T> {
    fn new() -> Self {
        Self {
            _marker: PhantomData,
        }
    }

    fn to_other_ad_type<T2: AD>(&self) -> WeightFn<T2> {
        WeightFn::new()
    }
}

impl<T: AD> DifferentiableFunctionTrait<T> for WeightFn<T> {
    const NAME: &'static str = "WeightFn";

    fn call(&self, inputs: &[T], _freeze: bool) -> Vec<T> {
        vec![weight_ad(inputs[0])]
    }

    fn num_inputs(&self) -> usize {
        1
    }

    fn num_outputs(&self) -> usize {
        1
    }
}

#[test]
fn mixture_probit_jet_matches_three_autodiff_engines() {
    let points = [-3.0, -1.4, -0.4, 0.0, 0.4, 1.1, 2.7];

    let f_std = PdfFn::<f64>::new();
    let f_ad = f_std.to_other_ad_type::<adfn<1>>();
    let engine = FunctionEngine::new(f_std, f_ad, ForwardAD::new());

    for eta in points {
        let j = component_inverse_link_jet(LinkComponent::Probit, eta);

        let (d1_nd, d2_nd) = first_derivative(normal_pdf_numdual, eta);
        let (_v2, _d1_2, d3_nd) = second_derivative(normal_pdf_numdual, eta);

        let d2_autodiff = diff(normal_pdf_f1, eta);
        let (_v, jac) = engine.derivative(&[eta]);
        let d2_ad_trait = jac[(0, 0)];

        assert_close_ctx(
            "probit_jet",
            eta,
            "d1",
            "manual",
            j.d1,
            "num_dual",
            d1_nd,
            1e-12,
        );
        assert_close_ctx(
            "probit_jet",
            eta,
            "d2",
            "manual",
            j.d2,
            "num_dual",
            d2_nd,
            1e-10,
        );
        assert_close_ctx(
            "probit_jet",
            eta,
            "d2",
            "manual",
            j.d2,
            "autodiff",
            d2_autodiff,
            1e-10,
        );
        assert_close_ctx(
            "probit_jet",
            eta,
            "d2",
            "manual",
            j.d2,
            "ad_trait",
            d2_ad_trait,
            1e-10,
        );
        assert_close_ctx(
            "probit_jet",
            eta,
            "d3",
            "manual",
            j.d3,
            "num_dual",
            d3_nd,
            1e-9,
        );

        if eta.abs() > 1e-8 {
            assert_eq!(
                j.d2.signum(),
                d2_nd.signum(),
                "probit_jet eta={eta:.6} d2 sign mismatch: manual={} num_dual={}",
                j.d2,
                d2_nd
            );
        }
    }
}

#[test]
fn mixture_logit_jet_matches_three_autodiff_engines() {
    let points = [-4.0, -1.6, -0.3, 0.0, 0.6, 1.9, 4.0];

    let f_std = LogitFn::<f64>::new();
    let f_ad = f_std.to_other_ad_type::<adfn<1>>();
    let engine = FunctionEngine::new(f_std, f_ad, ForwardAD::new());

    for eta in points {
        let j = component_inverse_link_jet(LinkComponent::Logit, eta);

        let (mu_nd, d1_nd) = first_derivative(logistic_numdual, eta);
        let (_mu_nd2, _d1_nd2, d2_nd) = second_derivative(logistic_numdual, eta);
        let (_mu_nd3, _d1_nd3, _d2_nd3, d3_nd) = third_derivative(logistic_numdual, eta);

        let d1_autodiff = diff(logistic_f1, eta);
        let (_v, jac) = engine.derivative(&[eta]);
        let d1_ad_trait = jac[(0, 0)];

        assert_close_ctx(
            "logit_jet",
            eta,
            "mu",
            "manual",
            j.mu,
            "num_dual",
            mu_nd,
            1e-12,
        );
        assert_close_ctx(
            "logit_jet",
            eta,
            "d1",
            "manual",
            j.d1,
            "num_dual",
            d1_nd,
            1e-12,
        );
        assert_close_ctx(
            "logit_jet",
            eta,
            "d1",
            "manual",
            j.d1,
            "autodiff",
            d1_autodiff,
            1e-12,
        );
        assert_close_ctx(
            "logit_jet",
            eta,
            "d1",
            "manual",
            j.d1,
            "ad_trait",
            d1_ad_trait,
            1e-12,
        );
        assert_close_ctx(
            "logit_jet",
            eta,
            "d2",
            "manual",
            j.d2,
            "num_dual",
            d2_nd,
            1e-11,
        );
        assert_close_ctx(
            "logit_jet",
            eta,
            "d3",
            "manual",
            j.d3,
            "num_dual",
            d3_nd,
            1e-10,
        );
    }
}

#[test]
fn firth_logistic_weight_derivatives_match_three_autodiff_engines() {
    let points = [-3.5, -1.8, -0.7, 0.0, 0.9, 1.7, 3.2];

    let f_std = WeightFn::<f64>::new();
    let f_ad = f_std.to_other_ad_type::<adfn<1>>();
    let engine = FunctionEngine::new(f_std, f_ad, ForwardAD::new());

    let h4 = 1e-3;
    for eta in points {
        let (w, w1_nd) = first_derivative(weight_numdual, eta);
        let (_w2_val, _w1_nd2, w2_nd) = second_derivative(weight_numdual, eta);
        let (_w3_val, _w1_nd3, _w2_nd3, w3_nd) = third_derivative(weight_numdual, eta);

        let m = logistic_numdual(eta);
        let t = 1.0 - 2.0 * m;
        let w1_manual = w * t;
        let w2_manual = w * t * t - 2.0 * w * w;
        let w3_manual = w * t * t * t - 8.0 * w * w * t;
        let w4_manual = w * t * t * t * t - 22.0 * w * w * t * t + 16.0 * w * w * w;

        let d1_autodiff = diff(weight_f1, eta);
        let (_v, jac) = engine.derivative(&[eta]);
        let d1_ad_trait = jac[(0, 0)];

        let w3_at = |x: f64| {
            let (_ww, _, _, w3x) = third_derivative(weight_numdual, x);
            w3x
        };
        let w4_fd = (w3_at(eta + h4) - w3_at(eta - h4)) / (2.0 * h4);

        assert_close_ctx(
            "firth_logistic_weight",
            eta,
            "w1",
            "manual",
            w1_manual,
            "num_dual",
            w1_nd,
            1e-12,
        );
        assert_close_ctx(
            "firth_logistic_weight",
            eta,
            "w1",
            "manual",
            w1_manual,
            "autodiff",
            d1_autodiff,
            1e-12,
        );
        assert_close_ctx(
            "firth_logistic_weight",
            eta,
            "w1",
            "manual",
            w1_manual,
            "ad_trait",
            d1_ad_trait,
            1e-12,
        );
        assert_close_ctx(
            "firth_logistic_weight",
            eta,
            "w2",
            "manual",
            w2_manual,
            "num_dual",
            w2_nd,
            1e-11,
        );
        assert_close_ctx(
            "firth_logistic_weight",
            eta,
            "w3",
            "manual",
            w3_manual,
            "num_dual",
            w3_nd,
            1e-10,
        );
        assert_close_ctx(
            "firth_logistic_weight",
            eta,
            "w4",
            "manual",
            w4_manual,
            "fd(num_dual w3)",
            w4_fd,
            2e-3,
        );
    }
}
