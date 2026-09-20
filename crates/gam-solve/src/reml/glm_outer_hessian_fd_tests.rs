//! Finite-difference check of the exact analytic LAML outer Hessian for the
//! non-Gaussian standard GLMs the outer search now consumes it for.
//!
//! The analytic `∂²V/∂ρ∂ρᵀ` carries the family's third and fourth
//! η-derivatives (through `dW/dη` and `d²W/dη²`, including the
//! observed-information terms of the non-canonical links). This module
//! differences the analytic *gradient* in ρ and requires the analytic
//! Hessian to match it for binomial-logit, binomial-probit,
//! binomial-cloglog, Poisson-log and Gamma-log.
#![cfg(test)]

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

fn simulate(
    family: &ResponseFamily,
    link: StandardLink,
) -> (Array1<f64>, Array2<f64>, Vec<Array2<f64>>) {
    let mut rng = Lcg(0x5eed_0f_9a11);
    let x1: Vec<f64> = (0..N).map(|_| rng.uniform()).collect();
    let x2: Vec<f64> = (0..N).map(|_| rng.uniform()).collect();
    let (x, penalties) = design_and_penalties(&x1, &x2);
    let y = (0..N)
        .map(|i| {
            let signal = (2.0 * std::f64::consts::PI * x1[i]).sin() * 0.9 + (x2[i] - 0.5) * 1.2;
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
            + t * (-0.284_496_736
                + t * (1.421_413_741 + t * (-1.453_152_027 + t * 1.061_405_429))));
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
    let canonical =
        gam_terms::construction::canonicalize_penalty_specs(&specs, &nullspace, p, "test")
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
    assert_hessian_matches_fd_on(name, likelihood, &y, &x, &penalties);
}

fn assert_hessian_matches_fd_on(
    name: &str,
    likelihood: GlmLikelihoodSpec,
    y: &Array1<f64>,
    x: &Array2<f64>,
    penalties: &[Array2<f64>],
) {
    let w = Array1::<f64>::ones(N);
    let offset = Array1::<f64>::zeros(N);
    let cfg = RemlConfig::external(likelihood, 1e-10, false).with_max_iterations(500);
    let reml = state(y, &w, &offset, x, penalties, &cfg);
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
        let gradient_scale = eval.gradient.iter().fold(1.0_f64, |m, v| m.max(v.abs()));
        for col in 0..rho.len() {
            // The analytic gradient against the central difference of the
            // cost it is the derivative of.
            let mut rp = rho.clone();
            let mut rm = rho.clone();
            rp[col] += delta;
            rm[col] -= delta;
            let vp = reml
                .compute_outer_eval_with_order(&rp, OuterEvalOrder::Value)
                .expect("plus value")
                .cost;
            let vm = reml
                .compute_outer_eval_with_order(&rm, OuterEvalOrder::Value)
                .expect("minus value")
                .cost;
            let fd = (vp - vm) / (2.0 * delta);
            let an = eval.gradient[col];
            let err = (fd - an).abs() / gradient_scale;
            assert!(
                err < 1e-5,
                "{name} rho={rho:?} g[{col}]: analytic={an:.10e} fd={fd:.10e} \
                 err/max(1,‖g‖∞)={err:.3e}"
            );
        }
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

/// A latched #784 block correction whose `Δ_b` has no closed-form ρ-Hessian
/// (`BlockQuadratureLatch::hessian_refusal`) defines a criterion with none: the
/// search runs on BFGS curvature, the unified VGH evaluation reports none, and
/// the Hessian the smoothing-corrected covariance inverts is a typed error
/// carrying the mathematical reason. The Laplace Hessian without `∂²Δ_b` is
/// never declared in its place (DOC-17), and no plug-in covariance is shipped
/// for it (pyGAM audit F17). A latch that carries its Hessian declares it;
/// that splice is checked against finite differences of the spliced gradient
/// in `regression_block_correction_outer_hessian_fd`, where the corrector is
/// linked.
#[test]
fn a_latched_block_correction_without_a_hessian_refuses_with_its_reason_784() {
    let (y, x, penalties) = simulate(&ResponseFamily::Binomial, StandardLink::Logit);
    let w = Array1::<f64>::ones(N);
    let offset = Array1::<f64>::zeros(N);
    let cfg = RemlConfig::external(
        spec(ResponseFamily::Binomial, StandardLink::Logit),
        1e-10,
        false,
    )
    .with_max_iterations(500);
    let reml = state(&y, &w, &offset, &x, &penalties, &cfg);
    let rho = Array1::from(vec![0.5, -0.5]);
    assert!(reml.analytic_outer_hessian_enabled());
    assert!(reml.compute_lamlhessian_consistent(&rho).is_ok());

    // Latch a one-direction block, as an admission does, whose Hessian the
    // curvature support refused.
    let reason = "the block's row curvature has no closed-form fourth η-derivative";
    reml.block_correction_admission
        .store(2, std::sync::atomic::Ordering::Relaxed);
    *reml
        .block_correction_axis_orders
        .lock()
        .unwrap_or_else(std::sync::PoisonError::into_inner) = Some(super::BlockQuadratureLatch {
        block_positions: vec![0],
        axis_orders: vec![8],
        axis_quadrature_errors: vec![0.0],
        axis_split: false,
        hessian_refusal: Some(reason.to_string()),
    });
    reml.reset_outer_seed_state();

    assert!(
        !reml.analytic_outer_hessian_enabled(),
        "a latched block correction without a ρ-Hessian declares no analytic outer Hessian"
    );
    let eval = reml
        .compute_outer_eval_with_order(&rho, OuterEvalOrder::ValueGradientHessian)
        .expect("value and gradient stay exact");
    assert!(matches!(eval.hessian, HessianValue::Unavailable));
    let refusal = reml
        .compute_lamlhessian_consistent(&rho)
        .expect_err("the smoothing correction's Hessian is a typed refusal");
    assert!(refusal.to_string().contains(reason), "{refusal}");

    // The same latch with its Hessian declares it again.
    reml.block_correction_axis_orders
        .lock()
        .unwrap_or_else(std::sync::PoisonError::into_inner)
        .as_mut()
        .expect("latched")
        .hessian_refusal = None;
    reml.reset_outer_seed_state();
    assert!(reml.analytic_outer_hessian_enabled());
    assert!(reml.compute_lamlhessian_consistent(&rho).is_ok());
}

/// Linear predictor of a generic variance × link cell as a function of the
/// fixture's signal `s ∈ [-1.5, 1.5]`, kept strictly inside the cell's
/// feasibility set so the truth is an interior point.
fn generic_cell_eta(family: &ResponseFamily, link: StandardLink, signal: f64) -> f64 {
    match (family, link) {
        (ResponseFamily::Gaussian, StandardLink::Log) => 0.5 + 0.3 * signal,
        (ResponseFamily::Gaussian | ResponseFamily::Gamma, StandardLink::Sqrt) => {
            1.5 + 0.3 * signal
        }
        (
            ResponseFamily::Gaussian | ResponseFamily::Gamma | ResponseFamily::InverseGaussian,
            StandardLink::InverseSquared | StandardLink::Inverse,
        ) => 1.0 + 0.3 * signal,
        (ResponseFamily::Poisson, StandardLink::Identity) => 4.0 + 1.5 * signal,
        (ResponseFamily::Poisson, StandardLink::Sqrt) => 2.0 + 0.5 * signal,
        (ResponseFamily::Poisson, StandardLink::Inverse) => 0.3 + 0.1 * signal,
        (ResponseFamily::Poisson, StandardLink::InverseSquared) => 0.1 + 0.03 * signal,
        (ResponseFamily::Gamma, StandardLink::Identity) => 2.0 + 0.8 * signal,
        (ResponseFamily::InverseGaussian, StandardLink::Identity) => 1.5 + 0.5 * signal,
        (ResponseFamily::InverseGaussian, StandardLink::Sqrt) => 1.2 + 0.25 * signal,
        (ResponseFamily::Binomial, StandardLink::Log) => -1.2 + 0.4 * signal,
        other => panic!("{other:?} is not a generic cell of this fixture"),
    }
}

fn generic_cell_mean(link: StandardLink, eta: f64) -> f64 {
    match link {
        StandardLink::Identity => eta,
        StandardLink::Log => eta.exp(),
        StandardLink::Sqrt => eta * eta,
        StandardLink::Inverse => 1.0 / eta,
        StandardLink::InverseSquared => eta.powf(-0.5),
        other => panic!("link {other:?} is not a generic-cell link of this fixture"),
    }
}

/// Inverse-Gaussian dispersion of the generic-cell fixture (`V = φμ³`).
const GENERIC_IG_PHI: f64 = 0.3;
/// Gaussian standard deviation of the generic-cell fixture, relative to the mean.
const GENERIC_GAUSSIAN_SD: f64 = 0.05;

fn simulate_generic(
    family: &ResponseFamily,
    link: StandardLink,
) -> (Array1<f64>, Array2<f64>, Vec<Array2<f64>>) {
    let mut rng = Lcg(0x6e_e71c_ce11);
    let x1: Vec<f64> = (0..N).map(|_| rng.uniform()).collect();
    let x2: Vec<f64> = (0..N).map(|_| rng.uniform()).collect();
    let (x, penalties) = design_and_penalties(&x1, &x2);
    let normal = |rng: &mut Lcg| {
        let (u1, u2) = (rng.uniform(), rng.uniform());
        (-2.0 * u1.ln()).sqrt() * (2.0 * std::f64::consts::PI * u2).cos()
    };
    let y = (0..N)
        .map(|i| {
            let signal = (2.0 * std::f64::consts::PI * x1[i]).sin() * 0.9 + (x2[i] - 0.5) * 1.2;
            let mu = generic_cell_mean(link, generic_cell_eta(family, link, signal));
            match family {
                ResponseFamily::Gaussian => mu * (1.0 + GENERIC_GAUSSIAN_SD * normal(&mut rng)),
                ResponseFamily::Poisson => {
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
                    let e = -rng.uniform().ln() - rng.uniform().ln();
                    0.5 * mu * e
                }
                ResponseFamily::InverseGaussian => {
                    // Michael–Schucany–Haas transformation with λ = 1/φ.
                    let lambda = 1.0 / GENERIC_IG_PHI;
                    let v = normal(&mut rng).powi(2);
                    let root = mu + mu * mu * v / (2.0 * lambda)
                        - mu / (2.0 * lambda)
                            * (4.0 * mu * lambda * v + mu * mu * v * v).sqrt();
                    if rng.uniform() <= mu / (mu + root) {
                        root
                    } else {
                        mu * mu / root
                    }
                }
                ResponseFamily::Binomial => f64::from(u8::from(rng.uniform() < mu)),
                other => panic!("family {other:?} not in this fixture"),
            }
        })
        .collect();
    (y, x, penalties)
}

/// The generic variance × link kernel's outer LAML gradient is the derivative
/// of its cost and its analytic outer Hessian the derivative of its gradient,
/// both through the exact third and fourth η-derivatives of the composed
/// `V ∘ μ(η)` row program. Dispersions are pinned for the λ search as the
/// production search pins them.
fn assert_generic_cell_matches_fd(family: ResponseFamily, link: StandardLink) {
    let name = format!("{}-{}", family.name(), link.name());
    let base = spec(family.clone(), link);
    assert!(
        base.spec.generic_edm_cell().is_some(),
        "{name} must route through the generic variance × link kernel"
    );
    let likelihood = match family {
        ResponseFamily::Gamma => base.with_gamma_shape_frozen_for_search(2.0),
        ResponseFamily::InverseGaussian => {
            base.with_dispersion_phi_frozen_for_search(GENERIC_IG_PHI)
        }
        ResponseFamily::Gaussian => {
            base.with_dispersion_phi_frozen_for_search(GENERIC_GAUSSIAN_SD * GENERIC_GAUSSIAN_SD)
        }
        _ => base,
    };
    let (y, x, penalties) = simulate_generic(&family, link);
    assert_hessian_matches_fd_on(&name, likelihood, &y, &x, &penalties);
}

macro_rules! generic_cell_fd_tests {
    ($($test:ident => ($family:expr, $link:expr);)*) => {$(
        #[test]
        fn $test() {
            assert_generic_cell_matches_fd($family, $link);
        }
    )*};
}

generic_cell_fd_tests! {
    gaussian_log_outer_derivatives_match_fd => (ResponseFamily::Gaussian, StandardLink::Log);
    gaussian_sqrt_outer_derivatives_match_fd => (ResponseFamily::Gaussian, StandardLink::Sqrt);
    gaussian_inverse_squared_outer_derivatives_match_fd =>
        (ResponseFamily::Gaussian, StandardLink::InverseSquared);
    poisson_identity_outer_derivatives_match_fd =>
        (ResponseFamily::Poisson, StandardLink::Identity);
    poisson_sqrt_outer_derivatives_match_fd => (ResponseFamily::Poisson, StandardLink::Sqrt);
    poisson_inverse_outer_derivatives_match_fd => (ResponseFamily::Poisson, StandardLink::Inverse);
    poisson_inverse_squared_outer_derivatives_match_fd =>
        (ResponseFamily::Poisson, StandardLink::InverseSquared);
    gamma_identity_outer_derivatives_match_fd => (ResponseFamily::Gamma, StandardLink::Identity);
    gamma_sqrt_outer_derivatives_match_fd => (ResponseFamily::Gamma, StandardLink::Sqrt);
    gamma_inverse_squared_outer_derivatives_match_fd =>
        (ResponseFamily::Gamma, StandardLink::InverseSquared);
    inverse_gaussian_identity_outer_derivatives_match_fd =>
        (ResponseFamily::InverseGaussian, StandardLink::Identity);
    inverse_gaussian_sqrt_outer_derivatives_match_fd =>
        (ResponseFamily::InverseGaussian, StandardLink::Sqrt);
    inverse_gaussian_inverse_outer_derivatives_match_fd =>
        (ResponseFamily::InverseGaussian, StandardLink::Inverse);
    binomial_log_outer_derivatives_match_fd => (ResponseFamily::Binomial, StandardLink::Log);
}
