//! Finite-difference check of the exact analytic LAML outer Hessian for the
//! non-Gaussian standard GLMs the outer search now consumes it for.
//!
//! The analytic `∂²V/∂ρ∂ρᵀ` carries the family's third and fourth
//! η-derivatives (through `dW/dη` and `d²W/dη²`, including the
//! observed-information terms of the non-canonical links). This module
//! differences the analytic *gradient* in ρ and requires the analytic
//! Hessian to match it for binomial-logit, binomial-probit,
//! binomial-cloglog, Poisson-log and Gamma-log.

use super::{RemlConfig, RemlState};
use crate::rho_optimizer::OuterEvalOrder;
use gam_problem::{
    GlmLikelihoodSpec, HessianValue, InverseLink, LikelihoodSpec, ResponseFamily, StandardLink,
};
use ndarray::{Array1, Array2};

const N: usize = 240;
const BUMPS: usize = 6;

/// Deterministic uniform stream (64-bit LCG, top 53 bits).
struct Lcg(u64);

impl Lcg {
    fn uniform(&mut self) -> f64 {
        self.0 = self
            .0
            .wrapping_mul(6_364_136_223_846_793_005)
            .wrapping_add(1_442_695_040_888_963_407);
        ((self.0 >> 11) as f64 + 0.5) / (1u64 << 53) as f64
    }
}

/// Intercept plus two Gaussian-bump blocks, one per covariate; each block
/// carries its own second-difference penalty, so the LAML surface has two
/// smoothing parameters.
fn design_and_penalties(x1: &[f64], x2: &[f64]) -> (Array2<f64>, Vec<Array2<f64>>) {
    let p = 1 + 2 * BUMPS;
    let mut x = Array2::<f64>::zeros((x1.len(), p));
    let width = 0.18;
    for i in 0..x1.len() {
        x[[i, 0]] = 1.0;
        for k in 0..BUMPS {
            let c = k as f64 / (BUMPS - 1) as f64;
            x[[i, 1 + k]] = (-(x1[i] - c).powi(2) / (2.0 * width * width)).exp();
            x[[i, 1 + BUMPS + k]] = (-(x2[i] - c).powi(2) / (2.0 * width * width)).exp();
        }
    }
    let mut d = Array2::<f64>::zeros((BUMPS - 2, BUMPS));
    for r in 0..BUMPS - 2 {
        d[[r, r]] = 1.0;
        d[[r, r + 1]] = -2.0;
        d[[r, r + 2]] = 1.0;
    }
    let local = d.t().dot(&d);
    let penalties = (0..2)
        .map(|block| {
            let mut s = Array2::<f64>::zeros((p, p));
            let off = 1 + block * BUMPS;
            s.slice_mut(ndarray::s![off..off + BUMPS, off..off + BUMPS])
                .assign(&local);
            s
        })
        .collect();
    (x, penalties)
}

fn simulate(family: &ResponseFamily, link: StandardLink) -> (Array1<f64>, Array2<f64>, Vec<Array2<f64>>) {
    let mut rng = Lcg(0x5eed_0f_9a11);
    let x1: Vec<f64> = (0..N).map(|_| rng.uniform()).collect();
    let x2: Vec<f64> = (0..N).map(|_| rng.uniform()).collect();
    let (x, penalties) = design_and_penalties(&x1, &x2);
    let y = (0..N)
        .map(|i| {
            let signal =
                (2.0 * std::f64::consts::PI * x1[i]).sin() * 0.9 + (x2[i] - 0.5) * 1.2;
            match family {
                ResponseFamily::Binomial => {
                    let mu = match link {
                        StandardLink::Logit => 1.0 / (1.0 + (-signal).exp()),
                        StandardLink::Probit => {
                            0.5 * (1.0 + libm_erf(signal / std::f64::consts::SQRT_2))
                        }
                        StandardLink::CLogLog => 1.0 - (-(signal - 0.4).exp()).exp(),
                        other => panic!("binomial link {other:?} not in this fixture"),
                    };
                    f64::from(u8::from(rng.uniform() < mu))
                }
                ResponseFamily::Poisson => {
                    // Inversion sampling of Poisson(mu).
                    let mu = (0.6 + 0.7 * signal).exp();
                    let u = rng.uniform();
                    let mut k = 0.0;
                    let mut pk = (-mu).exp();
                    let mut cdf = pk;
                    while u > cdf {
                        k += 1.0;
                        pk *= mu / k;
                        cdf += pk;
                    }
                    k
                }
                ResponseFamily::Gamma => {
                    // Shape-2 Gamma with mean mu: mu/2 times a sum of two
                    // unit exponentials.
                    let mu = (0.3 + 0.6 * signal).exp();
                    let e = -rng.uniform().ln() - rng.uniform().ln();
                    0.5 * mu * e
                }
                other => panic!("family {other:?} not in this fixture"),
            }
        })
        .collect();
    (y, x, penalties)
}

/// `erf` via the Abramowitz–Stegun 7.1.26 rational form; only used to draw the
/// probit fixture's responses, so its 1e-7 accuracy is irrelevant to the check.
fn libm_erf(z: f64) -> f64 {
    let t = 1.0 / (1.0 + 0.327_591_1 * z.abs());
    let poly = t
        * (0.254_829_592
            + t * (-0.284_496_736 + t * (1.421_413_741 + t * (-1.453_152_027 + t * 1.061_405_429))));
    let r = 1.0 - poly * (-z * z).exp();
    if z >= 0.0 { r } else { -r }
}

fn state<'a>(
    y: &'a Array1<f64>,
    w: &'a Array1<f64>,
    offset: &'a Array1<f64>,
    x: &Array2<f64>,
    penalties: &[Array2<f64>],
    cfg: &'a RemlConfig,
) -> RemlState<'a> {
    use crate::estimate::PenaltySpec;
    let p = x.ncols();
    let specs: Vec<PenaltySpec> = penalties.iter().cloned().map(PenaltySpec::Dense).collect();
    let nullspace = vec![2; specs.len()];
    let canonical = gam_terms::construction::canonicalize_penalty_specs(&specs, &nullspace, p, "test")
        .map(|(canonical, _)| canonical)
        .expect("canonicalize");
    RemlState::newwith_offset(
        y.view(),
        x.clone(),
        w.view(),
        offset.view(),
        canonical,
        p,
        cfg,
        Some(nullspace),
        None,
        None,
    )
    .expect("state")
}

fn assert_hessian_matches_fd(name: &str, likelihood: GlmLikelihoodSpec) {
    let family = likelihood.spec.response.clone();
    let link = match &likelihood.spec.link {
        InverseLink::Standard(link) => *link,
        other => panic!("{name}: non-standard link {other:?}"),
    };
    let (y, x, penalties) = simulate(&family, link);
    let w = Array1::<f64>::ones(N);
    let offset = Array1::<f64>::zeros(N);
    let cfg = RemlConfig::external(likelihood, 1e-10, false).with_max_iterations(500);
    let reml = state(&y, &w, &offset, &x, &penalties, &cfg);
    assert!(
        reml.analytic_outer_hessian_enabled(),
        "{name}: the analytic outer Hessian must be available"
    );

    for rho in [Array1::from(vec![0.5, -0.5]), Array1::from(vec![3.0, 1.5])] {
        let eval = reml
            .compute_outer_eval_with_order(&rho, OuterEvalOrder::ValueGradientHessian)
            .expect("analytic Hessian eval");
        let h = match eval.hessian {
            HessianValue::Dense(h) => h,
            HessianValue::Operator(_) | HessianValue::Unavailable => {
                panic!("{name}: expected a dense analytic Hessian")
            }
        };
        let scale = h.iter().fold(0.0_f64, |m, v| m.max(v.abs()));
        let delta = 1e-4;
        for col in 0..rho.len() {
            let mut rp = rho.clone();
            let mut rm = rho.clone();
            rp[col] += delta;
            rm[col] -= delta;
            let gp = reml
                .compute_outer_eval_with_order(&rp, OuterEvalOrder::ValueAndGradient)
                .expect("plus gradient")
                .gradient;
            let gm = reml
                .compute_outer_eval_with_order(&rm, OuterEvalOrder::ValueAndGradient)
                .expect("minus gradient")
                .gradient;
            for row in 0..rho.len() {
                let fd = (gp[row] - gm[row]) / (2.0 * delta);
                let an = h[[row, col]];
                let err = (fd - an).abs() / scale.max(1e-8);
                assert!(
                    err < 1e-4,
                    "{name} rho={rho:?} H[{row},{col}]: analytic={an:.10e} fd={fd:.10e} \
                     err/‖H‖∞={err:.3e}"
                );
            }
        }
    }
}

fn spec(family: ResponseFamily, link: StandardLink) -> GlmLikelihoodSpec {
    GlmLikelihoodSpec::canonical(LikelihoodSpec::new(family, InverseLink::Standard(link)))
}

#[test]
fn binomial_logit_outer_hessian_matches_gradient_fd() {
    assert_hessian_matches_fd(
        "binomial-logit",
        spec(ResponseFamily::Binomial, StandardLink::Logit),
    );
}

#[test]
fn binomial_probit_outer_hessian_matches_gradient_fd() {
    assert_hessian_matches_fd(
        "binomial-probit",
        spec(ResponseFamily::Binomial, StandardLink::Probit),
    );
}

#[test]
fn binomial_cloglog_outer_hessian_matches_gradient_fd() {
    assert_hessian_matches_fd(
        "binomial-cloglog",
        spec(ResponseFamily::Binomial, StandardLink::CLogLog),
    );
}

#[test]
fn poisson_log_outer_hessian_matches_gradient_fd() {
    assert_hessian_matches_fd(
        "poisson-log",
        spec(ResponseFamily::Poisson, StandardLink::Log),
    );
}

#[test]
fn gamma_log_outer_hessian_matches_gradient_fd() {
    // The λ-search holds the Gamma shape fixed (#1074), so the searched
    // surface is `V(ρ; k)` at a pinned `k`.
    assert_hessian_matches_fd(
        "gamma-log",
        spec(ResponseFamily::Gamma, StandardLink::Log).with_gamma_shape_frozen_for_search(2.0),
    );
}
