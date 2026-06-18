//! Independent #931 oracle for a large-penalty LAML derivative.
//!
//! This deliberately does not finite-difference the production value path.
//! The oracle below owns the tiny inner solve, evaluates a probit Bernoulli
//! likelihood with `libm::erfc`, and differentiates the textbook LAML scalar
//! at rho = 6. The production analytic gradient must agree with that external
//! scalar derivative.

use gam::estimate::{ExternalOptimOptions, evaluate_externalgradient};
use gam::smooth::BlockwisePenalty;
use gam::types::{InverseLink, LikelihoodSpec, ResponseFamily, StandardLink};
use ndarray::{Array1, Array2, array};

const RHO: f64 = 6.0;
const FD_STEP: f64 = 2.5e-4;
const SQRT_2: f64 = std::f64::consts::SQRT_2;
const INV_SQRT_2PI: f64 = 0.398_942_280_401_432_7;

#[inline]
fn normal_cdf_erfc(x: f64) -> f64 {
    0.5 * libm::erfc(-x / SQRT_2)
}

#[inline]
fn normal_survival_erfc(x: f64) -> f64 {
    0.5 * libm::erfc(x / SQRT_2)
}

#[inline]
fn normal_pdf(x: f64) -> f64 {
    INV_SQRT_2PI * (-0.5 * x * x).exp()
}

#[derive(Clone)]
struct ProbitOracle {
    x: Array2<f64>,
    y: Array1<f64>,
}

impl ProbitOracle {
    fn new() -> Self {
        let x = array![
            [1.0, -0.90],
            [1.0, -0.55],
            [1.0, -0.25],
            [1.0, 0.05],
            [1.0, 0.30],
            [1.0, 0.60],
            [1.0, 0.85],
            [1.0, 1.10],
        ];
        let y = array![0.0, 0.0, 1.0, 0.0, 1.0, 1.0, 0.0, 1.0];
        Self { x, y }
    }

    fn nll_grad_hess(&self, beta: [f64; 2], rho: f64) -> (f64, [f64; 2], [[f64; 2]; 2]) {
        let lambda = rho.exp();
        let mut nll = 0.0;
        let mut grad = [0.0_f64; 2];
        let mut h = [[0.0_f64; 2]; 2];

        for i in 0..self.x.nrows() {
            let row = [self.x[[i, 0]], self.x[[i, 1]]];
            let eta = row[0] * beta[0] + row[1] * beta[1];
            let pdf = normal_pdf(eta);
            let (loss, d1, d2) = if self.y[i] == 1.0 {
                let p = normal_cdf_erfc(eta).max(1.0e-300);
                let r = pdf / p;
                (-p.ln(), -r, r * (eta + r))
            } else {
                let q = normal_survival_erfc(eta).max(1.0e-300);
                let r = pdf / q;
                (-q.ln(), r, r * (r - eta))
            };
            nll += loss;
            for a in 0..2 {
                grad[a] += d1 * row[a];
                for b in 0..2 {
                    h[a][b] += d2 * row[a] * row[b];
                }
            }
        }

        nll += 0.5 * lambda * beta[1] * beta[1];
        grad[1] += lambda * beta[1];
        h[1][1] += lambda;
        (nll, grad, h)
    }

    fn solve_mode(&self, rho: f64) -> ([f64; 2], [[f64; 2]; 2], f64) {
        let mut beta = [0.0_f64, 0.0_f64];
        for _ in 0..80 {
            let (f0, grad, h) = self.nll_grad_hess(beta, rho);
            let gnorm = grad[0].hypot(grad[1]);
            if gnorm < 2.0e-13 {
                break;
            }
            let det = h[0][0] * h[1][1] - h[0][1] * h[1][0];
            assert!(
                det.is_finite() && det > 1.0e-10,
                "oracle Hessian must stay well-conditioned, det={det:.6e}, h={h:?}"
            );
            let step = [
                (h[1][1] * grad[0] - h[0][1] * grad[1]) / det,
                (-h[1][0] * grad[0] + h[0][0] * grad[1]) / det,
            ];
            let directional = -(grad[0] * step[0] + grad[1] * step[1]);
            let mut alpha = 1.0_f64;
            let mut accepted = false;
            for _ in 0..40 {
                let trial = [beta[0] - alpha * step[0], beta[1] - alpha * step[1]];
                let (ft, _, _) = self.nll_grad_hess(trial, rho);
                if ft.is_finite() && ft <= f0 + 1.0e-4 * alpha * directional {
                    beta = trial;
                    accepted = true;
                    break;
                }
                alpha *= 0.5;
            }
            assert!(
                accepted,
                "oracle Newton line search failed at beta={beta:?}"
            );
        }
        let (nll, grad, h) = self.nll_grad_hess(beta, rho);
        assert!(
            grad[0].hypot(grad[1]) < 1.0e-10,
            "oracle inner solve did not reach stationarity: beta={beta:?}, grad={grad:?}"
        );
        (beta, h, nll)
    }

    fn laml_value(&self, rho: f64) -> f64 {
        let (_beta, h, penalized_nll) = self.solve_mode(rho);
        let det_h = h[0][0] * h[1][1] - h[0][1] * h[1][0];
        assert!(det_h > 0.0 && det_h.is_finite());
        // Fixed-dispersion Laplace/LAML: -ell + 0.5 beta'S beta
        // + 0.5 log|H| - 0.5 log|S_lambda|_+. The one penalized direction has
        // S eigenvalue 1, hence log|S_lambda|_+ = rho.
        penalized_nll + 0.5 * det_h.ln() - 0.5 * rho
    }

    fn fd_laml_derivative_at_rho_six(&self) -> f64 {
        (self.laml_value(RHO + FD_STEP) - self.laml_value(RHO - FD_STEP)) / (2.0 * FD_STEP)
    }
}

fn production_probit_gradient_at_rho_six(oracle: &ProbitOracle) -> f64 {
    let mut s = Array2::<f64>::zeros((2, 2));
    s[[1, 1]] = 1.0;
    let penalties = vec![BlockwisePenalty::new(0..2, s)];
    let opts = ExternalOptimOptions {
        latent_cloglog: None,
        mixture_link: None,
        optimize_mixture: false,
        sas_link: None,
        optimize_sas: false,
        family: LikelihoodSpec::new(
            ResponseFamily::Binomial,
            InverseLink::Standard(StandardLink::Probit),
        ),
        compute_inference: true,
        skip_rho_posterior_inference: false,
        max_iter: 400,
        tol: 1.0e-12,
        nullspace_dims: vec![1],
        linear_constraints: None,
        firth_bias_reduction: None,
        penalty_shrinkage_floor: None,
        rho_prior: Default::default(),
        kronecker_penalty_system: None,
        kronecker_factored: None,
        persist_warm_start_disk: false,
    };
    let weights = Array1::<f64>::ones(oracle.y.len());
    let offset = Array1::<f64>::zeros(oracle.y.len());
    let rho = array![RHO];
    evaluate_externalgradient(
        oracle.y.view(),
        weights.view(),
        oracle.x.clone(),
        offset.view(),
        &penalties,
        &opts,
        &rho,
    )
    .expect("production probit LAML gradient")[0]
}

#[test]
fn issue_931_rho_six_laml_derivative_matches_independent_erfc_oracle() {
    let oracle = ProbitOracle::new();
    let fd = oracle.fd_laml_derivative_at_rho_six();
    let analytic = production_probit_gradient_at_rho_six(&oracle);
    let scale = analytic.abs().max(fd.abs()).max(1.0);
    let rel = (analytic - fd).abs() / scale;
    assert!(
        rel < 2.0e-5,
        "#931 independent erfc LAML oracle disagrees at rho=6: analytic={analytic:.12e}, fd={fd:.12e}, rel={rel:.3e}"
    );
}
