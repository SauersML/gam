use ad_trait::AD;
use ad_trait::differentiable_function::{DifferentiableFunctionTrait, ForwardAD};
use ad_trait::forward_ad::adfn::adfn;
use ad_trait::function_engine::FunctionEngine;
use autodiff::{F1, Float, diff};
use gam::mixture_link::component_inverse_link_jet;
use gam::types::LinkComponent;
use ndarray::{Array1, Array2, array};
use num_dual::{DualNum, first_derivative, second_derivative, third_derivative};
use std::marker::PhantomData;

mod common;

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

fn det3<D: DualNum<f64> + Copy>(m: &[[D; 3]; 3]) -> D {
    m[0][0] * (m[1][1] * m[2][2] - m[1][2] * m[2][1])
        - m[0][1] * (m[1][0] * m[2][2] - m[1][2] * m[2][0])
        + m[0][2] * (m[1][0] * m[2][1] - m[1][1] * m[2][0])
}

fn inv3<D: DualNum<f64> + Copy>(m: &[[D; 3]; 3]) -> [[D; 3]; 3] {
    let det = det3(m);
    let inv_det = D::one() / det;
    [
        [
            (m[1][1] * m[2][2] - m[1][2] * m[2][1]) * inv_det,
            (m[0][2] * m[2][1] - m[0][1] * m[2][2]) * inv_det,
            (m[0][1] * m[1][2] - m[0][2] * m[1][1]) * inv_det,
        ],
        [
            (m[1][2] * m[2][0] - m[1][0] * m[2][2]) * inv_det,
            (m[0][0] * m[2][2] - m[0][2] * m[2][0]) * inv_det,
            (m[0][2] * m[1][0] - m[0][0] * m[1][2]) * inv_det,
        ],
        [
            (m[1][0] * m[2][1] - m[1][1] * m[2][0]) * inv_det,
            (m[0][1] * m[2][0] - m[0][0] * m[2][1]) * inv_det,
            (m[0][0] * m[1][1] - m[0][1] * m[1][0]) * inv_det,
        ],
    ]
}

fn matmul3vec3<D: DualNum<f64> + Copy>(m: &[[D; 3]; 3], v: &[D; 3]) -> [D; 3] {
    let mut out = [D::zero(); 3];
    for i in 0..3 {
        out[i] = m[i][0] * v[0] + m[i][1] * v[1] + m[i][2] * v[2];
    }
    out
}

fn dot3<D: DualNum<f64> + Copy>(a: &[D; 3], b: &[D; 3]) -> D {
    a[0] * b[0] + a[1] * b[1] + a[2] * b[2]
}

fn firthphi_numdual<D: DualNum<f64> + Copy>(
    tau: D,
    x: &Array2<f64>,
    x_tau: &Array2<f64>,
    beta: &Array1<f64>,
) -> D {
    let mut info = [[D::zero(); 3]; 3];
    for i in 0..x.nrows() {
        let mut row = [D::zero(); 3];
        for j in 0..3 {
            row[j] = D::from(x[[i, j]]) + tau * D::from(x_tau[[i, j]]);
        }
        let eta = row[0] * D::from(beta[0]) + row[1] * D::from(beta[1]) + row[2] * D::from(beta[2]);
        let w = weight_numdual(eta);
        for a in 0..3 {
            for b in 0..3 {
                info[a][b] += w * row[a] * row[b];
            }
        }
    }
    D::from(0.5) * det3(&info).ln()
}

fn firthgradphi_numdual<D: DualNum<f64> + Copy>(
    tau: D,
    x: &Array2<f64>,
    x_tau: &Array2<f64>,
    beta: &Array1<f64>,
) -> [D; 3] {
    let mut rows = Vec::with_capacity(x.nrows());
    let mut w1 = Vec::with_capacity(x.nrows());
    let mut info = [[D::zero(); 3]; 3];

    for i in 0..x.nrows() {
        let mut row = [D::zero(); 3];
        for j in 0..3 {
            row[j] = D::from(x[[i, j]]) + tau * D::from(x_tau[[i, j]]);
        }
        let eta = row[0] * D::from(beta[0]) + row[1] * D::from(beta[1]) + row[2] * D::from(beta[2]);
        let mu = logistic_numdual(eta);
        let w = mu * (D::one() - mu);
        let t = D::one() - D::from(2.0) * mu;
        w1.push(w * t);
        rows.push(row);
        for a in 0..3 {
            for b in 0..3 {
                info[a][b] += w * row[a] * row[b];
            }
        }
    }

    let k = inv3(&info);
    let mut grad = [D::zero(); 3];
    for (i, row) in rows.iter().enumerate() {
        let kr = matmul3vec3(&k, row);
        let h = dot3(row, &kr);
        for j in 0..3 {
            grad[j] += D::from(0.5) * row[j] * w1[i] * h;
        }
    }
    grad
}

fn firth_tau_manual(
    x: &Array2<f64>,
    x_tau: &Array2<f64>,
    beta: &Array1<f64>,
) -> (f64, Array1<f64>) {
    let n = x.nrows();
    let eta = x.dot(beta);
    let mu = eta.mapv(|z| 1.0 / (1.0 + (-z).exp()));
    let w = &mu * &(1.0 - &mu);
    let t = 1.0 - 2.0 * &mu;
    let w1 = &w * &t;
    let w2 = &w * &(&t * &t) - 2.0 * &w * &w;
    let deta = x_tau.dot(beta);

    let mut info = [[0.0; 3]; 3];
    for i in 0..n {
        for a in 0..3 {
            for b in 0..3 {
                info[a][b] += w[i] * x[[i, a]] * x[[i, b]];
            }
        }
    }
    let k = inv3(&info);

    let mut h = Array1::<f64>::zeros(n);
    for i in 0..n {
        let row = [x[[i, 0]], x[[i, 1]], x[[i, 2]]];
        let kr = matmul3vec3(&k, &row);
        h[i] = dot3(&row, &kr);
    }

    let mut dot_i = [[0.0; 3]; 3];
    for i in 0..n {
        for a in 0..3 {
            for b in 0..3 {
                dot_i[a][b] += w[i] * x_tau[[i, a]] * x[[i, b]]
                    + w[i] * x[[i, a]] * x_tau[[i, b]]
                    + w1[i] * deta[i] * x[[i, a]] * x[[i, b]];
            }
        }
    }

    let mut phi_tau = 0.0;
    for i in 0..3 {
        for j in 0..3 {
            phi_tau += k[i][j] * dot_i[j][i];
        }
    }
    phi_tau *= 0.5;

    let mut dot_k = [[0.0; 3]; 3];
    for i in 0..3 {
        for j in 0..3 {
            for a in 0..3 {
                for b in 0..3 {
                    dot_k[i][j] -= k[i][a] * dot_i[a][b] * k[b][j];
                }
            }
        }
    }

    let mut dot_h = Array1::<f64>::zeros(n);
    for i in 0..n {
        let row = [x[[i, 0]], x[[i, 1]], x[[i, 2]]];
        let row_tau = [x_tau[[i, 0]], x_tau[[i, 1]], x_tau[[i, 2]]];
        let kk = matmul3vec3(&k, &row);
        let dk = matmul3vec3(&dot_k, &row);
        dot_h[i] = 2.0 * dot3(&row_tau, &kk) + dot3(&row, &dk);
    }

    let mut gphi_tau = 0.5 * x_tau.t().dot(&(&w1 * &h));
    let secondvec = &(&(&w2 * &deta) * &h) + &(&w1 * &dot_h);
    gphi_tau += &(0.5 * x.t().dot(&secondvec));

    (phi_tau, gphi_tau)
}

#[derive(Clone)]
struct PdfFn<T: AD> {
    marker: PhantomData<T>,
}

impl<T: AD> PdfFn<T> {
    fn new() -> Self {
        Self {
            marker: PhantomData,
        }
    }

    fn to_other_ad_type<T2: AD>(&self) -> PdfFn<T2> {
        PdfFn::new()
    }
}

impl<T: AD> DifferentiableFunctionTrait<T> for PdfFn<T> {
    const NAME: &'static str = "PdfFn";

    fn call(&self, inputs: &[T], _: bool) -> Vec<T> {
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
    marker: PhantomData<T>,
}

impl<T: AD> LogitFn<T> {
    fn new() -> Self {
        Self {
            marker: PhantomData,
        }
    }

    fn to_other_ad_type<T2: AD>(&self) -> LogitFn<T2> {
        LogitFn::new()
    }
}

impl<T: AD> DifferentiableFunctionTrait<T> for LogitFn<T> {
    const NAME: &'static str = "LogitFn";

    fn call(&self, inputs: &[T], _: bool) -> Vec<T> {
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
    marker: PhantomData<T>,
}

impl<T: AD> WeightFn<T> {
    fn new() -> Self {
        Self {
            marker: PhantomData,
        }
    }

    fn to_other_ad_type<T2: AD>(&self) -> WeightFn<T2> {
        WeightFn::new()
    }
}

impl<T: AD> DifferentiableFunctionTrait<T> for WeightFn<T> {
    const NAME: &'static str = "WeightFn";

    fn call(&self, inputs: &[T], _: bool) -> Vec<T> {
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

        let (_, d2_nd) = first_derivative(normal_pdf_numdual, eta);
        let (_, _, d3_nd) = second_derivative(normal_pdf_numdual, eta);

        let d2_autodiff = diff(normal_pdf_f1, eta);
        let (_, jac) = engine.derivative(&[eta]);
        let d2_ad_trait = jac[(0, 0)];

        assert!(j.d1.is_finite() && j.d1 > 0.0);
        assert_manual_ad_band!("probit_jet", eta, "d2", j.d2,
            "num_dual" => d2_nd, "autodiff" => d2_autodiff, "ad_trait" => d2_ad_trait);
        assert_manual_ad_band!("probit_jet", eta, "d3", j.d3, "num_dual" => d3_nd);

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
        let (_, _, d2_nd) = second_derivative(logistic_numdual, eta);
        let (_, _, _, d3_nd) = third_derivative(logistic_numdual, eta);

        let d1_autodiff = diff(logistic_f1, eta);
        let (_, jac) = engine.derivative(&[eta]);
        let d1_ad_trait = jac[(0, 0)];

        assert_manual_ad_band!("logit_jet", eta, "mu", j.mu, "num_dual" => mu_nd);
        assert_manual_ad_band!("logit_jet", eta, "d1", j.d1,
            "num_dual" => d1_nd, "autodiff" => d1_autodiff, "ad_trait" => d1_ad_trait);
        assert_manual_ad_band!("logit_jet", eta, "d2", j.d2, "num_dual" => d2_nd);
        assert_manual_ad_band!("logit_jet", eta, "d3", j.d3, "num_dual" => d3_nd);
    }
}

#[test]
fn firth_logisticweight_derivatives_match_three_autodiff_engines() {
    let points = [-3.5, -1.8, -0.7, 0.0, 0.9, 1.7, 3.2];

    let f_std = WeightFn::<f64>::new();
    let f_ad = f_std.to_other_ad_type::<adfn<1>>();
    let engine = FunctionEngine::new(f_std, f_ad, ForwardAD::new());

    let h4 = 1e-3;
    for eta in points {
        let (w, w1_nd) = first_derivative(weight_numdual, eta);
        let (_, _, w2_nd) = second_derivative(weight_numdual, eta);
        let (_, _, _, w3_nd) = third_derivative(weight_numdual, eta);

        let m = logistic_numdual(eta);
        let t = 1.0 - 2.0 * m;
        let w1_manual = w * t;
        let w2_manual = w * t * t - 2.0 * w * w;
        let w3_manual = w * t * t * t - 8.0 * w * w * t;
        let w4_manual = w * t * t * t * t - 22.0 * w * w * t * t + 16.0 * w * w * w;

        let d1_autodiff = diff(weight_f1, eta);
        let (_, jac) = engine.derivative(&[eta]);
        let d1_ad_trait = jac[(0, 0)];

        let w3_at = |x: f64| {
            let (_, _, _, w3x) = third_derivative(weight_numdual, x);
            w3x
        };
        let w4fd_h = (w3_at(eta + h4) - w3_at(eta - h4)) / (2.0 * h4);
        let h4_half = 0.5 * h4;
        let w4fd_h2 = (w3_at(eta + h4_half) - w3_at(eta - h4_half)) / (2.0 * h4_half);
        let h4_double = 2.0 * h4;
        let w4fd_2h = (w3_at(eta + h4_double) - w3_at(eta - h4_double)) / (2.0 * h4_double);

        assert_manual_ad_band!("firth_logisticweight", eta, "w1", w1_manual,
            "num_dual" => w1_nd, "autodiff" => d1_autodiff, "ad_trait" => d1_ad_trait);
        assert_manual_ad_band!("firth_logisticweight", eta, "w2", w2_manual, "num_dual" => w2_nd);
        assert_manual_ad_band!("firth_logisticweight", eta, "w3", w3_manual, "num_dual" => w3_nd);
        assert_manual_ad_band!("firth_logisticweight", eta, "w4", w4_manual,
            "fd(num_dual w3;h/2)" => w4fd_h2,
            "fd(num_dual w3;h)" => w4fd_h,
            "fd(num_dual w3;2h)" => w4fd_2h);
    }
}

#[test]
fn firth_tau_manual_matches_autodiff_band() {
    let x = array![
        [1.0, -1.0, 0.2],
        [1.0, -0.6, -0.3],
        [1.0, -0.1, 0.5],
        [1.0, 0.3, -0.7],
        [1.0, 0.8, 0.1],
        [1.0, 1.2, -0.4],
    ];
    let x_tau = array![
        [0.0, 0.15, -0.05],
        [0.0, -0.10, 0.02],
        [0.0, 0.08, 0.04],
        [0.0, -0.06, -0.03],
        [0.0, 0.05, 0.01],
        [0.0, -0.12, 0.06],
    ];
    let beta = array![0.1, -0.25, 0.2];

    let (phi_tau_manual, gphi_tau_manual) = firth_tau_manual(&x, &x_tau, &beta);
    let (_, phi_tau_nd) = first_derivative(|tau| firthphi_numdual(tau, &x, &x_tau, &beta), 0.0);

    let h = 1e-6;
    let phifd = (firthphi_numdual(h, &x, &x_tau, &beta) - firthphi_numdual(-h, &x, &x_tau, &beta))
        / (2.0 * h);

    assert_manual_ad_band!("firth_tau", 0.0, "phi_tau", phi_tau_manual,
        "num_dual" => phi_tau_nd,
        "fd" => phifd);

    for j in 0..3 {
        let (_, gj_nd) =
            first_derivative(|tau| firthgradphi_numdual(tau, &x, &x_tau, &beta)[j], 0.0);
        let gjfd = (firthgradphi_numdual(h, &x, &x_tau, &beta)[j]
            - firthgradphi_numdual(-h, &x, &x_tau, &beta)[j])
            / (2.0 * h);
        assert_manual_ad_band!("firth_tau", j as f64, "gphi_tau", gphi_tau_manual[j],
            "num_dual" => gj_nd,
            "fd" => gjfd);
    }
}
