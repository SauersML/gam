//! Student-t response family: the scale σ and degrees of freedom ν are LAML
//! hyperparameters estimated jointly with the smoothing parameters.
//!
//! Three gates:
//!
//! * the analytic outer gradient and Hessian in `θ = (ρ, ln σ, ln ν)` reproduce
//!   central differences of the same criterion (the value for the gradient,
//!   the analytic gradient for the Hessian), with ρ as the control arm;
//! * σ and ν are recovered from simulated Student-t data;
//! * on Gaussian data the fit walks ν to the Gaussian limit and reproduces the
//!   Gaussian fit's mean and scale.
#![cfg(test)]

use super::*;
use gam_problem::{InverseLink, LikelihoodSpec, ResponseFamily, StandardLink};
use gam_terms::smooth::BlockwisePenalty;
use ndarray::{Array1, Array2};
use rand::SeedableRng;
use rand::rngs::StdRng;
use rand_distr::Distribution;

/// Unpenalized columns: intercept and linear trend.
const NULL_COLUMNS: usize = 2;
/// Penalized sine columns `sin(kπx)`, `k = 1..=SINE_COLUMNS`.
const SINE_COLUMNS: usize = 10;

fn mean_function(x: f64) -> f64 {
    (3.0 * std::f64::consts::PI * x).sin() + 0.5 * x
}

/// Intercept, `x`, and `sin(kπx)` columns on `[0, 1]`. The penalty on the sine
/// block is `∫ f''² = ½ Σ (kπ)⁴ β_k²`, exactly diagonal in this basis.
fn sine_design(xs: &Array1<f64>) -> (Array2<f64>, BlockwisePenalty) {
    let p = NULL_COLUMNS + SINE_COLUMNS;
    let mut design = Array2::<f64>::zeros((xs.len(), p));
    for (i, &x) in xs.iter().enumerate() {
        design[[i, 0]] = 1.0;
        design[[i, 1]] = x;
        for k in 1..=SINE_COLUMNS {
            design[[i, NULL_COLUMNS + k - 1]] = (k as f64 * std::f64::consts::PI * x).sin();
        }
    }
    let mut s = Array2::<f64>::zeros((SINE_COLUMNS, SINE_COLUMNS));
    for k in 1..=SINE_COLUMNS {
        s[[k - 1, k - 1]] = 0.5 * (k as f64 * std::f64::consts::PI).powi(4);
    }
    (design, BlockwisePenalty::new(NULL_COLUMNS..p, s))
}

/// `n` equispaced covariates with `y = f(x) + noise`.
fn simulate(n: usize, seed: u64, mut noise: impl FnMut(&mut StdRng) -> f64) -> (Array1<f64>, Array1<f64>) {
    let mut rng = StdRng::seed_from_u64(seed);
    let xs = Array1::from_iter((0..n).map(|i| (i as f64 + 0.5) / n as f64));
    let y = xs.mapv(|x| mean_function(x) + noise(&mut rng));
    (xs, y)
}

fn student_t_family(sigma: f64, nu: f64) -> LikelihoodSpec {
    LikelihoodSpec::new(
        ResponseFamily::StudentT { sigma, nu },
        InverseLink::Standard(StandardLink::Identity),
    )
}

fn gaussian_family() -> LikelihoodSpec {
    LikelihoodSpec::new(
        ResponseFamily::Gaussian,
        InverseLink::Standard(StandardLink::Identity),
    )
}

fn fit(x: &Array2<f64>, y: &Array1<f64>, penalty: &BlockwisePenalty, family: LikelihoodSpec) -> UnifiedFitResult {
    let n = y.len();
    let weights = Array1::<f64>::ones(n);
    let offset = Array1::<f64>::zeros(n);
    let options = FitOptions {
        compute_inference: true,
        nullspace_dims: vec![0],
        ..FitOptions::default()
    };
    fit_gamwith_heuristic_log_lambdas(
        x.clone(),
        y.view(),
        weights.view(),
        offset.view(),
        std::slice::from_ref(penalty),
        None,
        family,
        &options,
    )
    .expect("the fit must converge")
}

fn fitted_student_t(fit: &UnifiedFitResult) -> (f64, f64) {
    match fit.likelihood_family.as_ref().map(|family| &family.response) {
        Some(ResponseFamily::StudentT { sigma, nu }) => (*sigma, *nu),
        other => panic!("a Student-t fit must report its fitted (sigma, nu); got {other:?}"),
    }
}

fn truth_mse(x: &Array2<f64>, xs: &Array1<f64>, beta: &Array1<f64>) -> f64 {
    let fitted = x.dot(beta);
    fitted
        .iter()
        .zip(xs.iter())
        .map(|(&f, &xi)| (f - mean_function(xi)).powi(2))
        .sum::<f64>()
        / xs.len() as f64
}

// ---------------------------------------------------------------------------
// Outer derivatives against finite differences.
// ---------------------------------------------------------------------------

const PROBE_N: usize = 150;
const PROBE_RHO: f64 = -6.0;
const PROBE_LOG_SIGMA: f64 = -1.0;
const PROBE_LOG_NU: f64 = 1.0;

fn probe_data() -> (Array2<f64>, Array1<f64>, BlockwisePenalty) {
    let t = rand_distr::StudentT::new(3.0).expect("valid degrees of freedom");
    let (xs, y) = simulate(PROBE_N, 20_260_919, |rng| 0.4 * t.sample(rng));
    let (x, penalty) = sine_design(&xs);
    (x, y, penalty)
}

/// Evaluate the joint criterion at `θ = (ρ, ln σ, ln ν)` on a fresh state. The
/// Student-t parameters live on the config the state reads its likelihood
/// from, so each point gets its own config and state; reusing one would risk
/// answering from a bundle cached at another point.
fn evaluate(
    theta: [f64; 3],
    mode: crate::estimate::reml::reml_outer_engine::EvalMode,
) -> crate::estimate::reml::reml_outer_engine::RemlLamlResult {
    let (x, y, penalty) = probe_data();
    let n = y.len();
    let p = x.ncols();
    let weights = Array1::<f64>::ones(n);
    let offset = Array1::<f64>::zeros(n);
    let config = RemlConfig::external(
        gam_spec::GlmLikelihoodSpec::canonical(student_t_family(theta[1].exp(), theta[2].exp())),
        1e-12,
        false,
    );
    let (canonical_penalties, active_nullspace_dims) =
        gam_terms::construction::canonicalize_penalty_specs(
            &[PenaltySpec::from_blockwise_ref(&penalty)],
            &[0],
            p,
            "Student-t LAML derivative fixture",
        )
        .expect("canonicalize the one-penalty fixture");
    let state = crate::estimate::reml::RemlState::newwith_offset(
        y.view(),
        x,
        weights.view(),
        offset.view(),
        canonical_penalties,
        p,
        &config,
        Some(active_nullspace_dims),
        None,
        None,
    )
    .expect("build the Student-t REML state");
    state
        .evaluate_unified_with_link_ext(&Array1::from_elem(1, theta[0]), mode)
        .expect("the Student-t criterion must evaluate at the probe point")
}

fn probe_theta() -> [f64; 3] {
    [PROBE_RHO, PROBE_LOG_SIGMA, PROBE_LOG_NU]
}

fn analytic_gradient(theta: [f64; 3]) -> Array1<f64> {
    let gradient = evaluate(
        theta,
        crate::estimate::reml::reml_outer_engine::EvalMode::ValueAndGradient,
    )
    .gradient
    .expect("ValueAndGradient must return a gradient");
    assert_eq!(
        gradient.len(),
        3,
        "the Student-t gradient must span rho, ln sigma and ln nu; a shorter one means \
         the fixture fell back to the rho-only criterion"
    );
    gradient
}

fn relative_error(analytic: f64, reference: f64) -> f64 {
    (analytic - reference).abs() / reference.abs().max(1.0e-8)
}

#[test]
fn student_t_outer_gradient_matches_finite_difference_of_the_criterion() {
    const BAR: f64 = 1.0e-4;
    const STEP: f64 = 1.0e-4;
    let base = probe_theta();
    let analytic = analytic_gradient(base);
    for (axis, label) in ["rho", "ln sigma", "ln nu"].into_iter().enumerate() {
        let mut plus = base;
        let mut minus = base;
        plus[axis] += STEP;
        minus[axis] -= STEP;
        let mode = crate::estimate::reml::reml_outer_engine::EvalMode::ValueOnly;
        let reference = (evaluate(plus, mode).cost - evaluate(minus, mode).cost) / (2.0 * STEP);
        let error = relative_error(analytic[axis], reference);
        assert!(
            error < BAR,
            "d LAML / d {label}: analytic {:.10e} vs central difference {reference:.10e} \
             (relative error {error:.3e} exceeds {BAR:.1e})",
            analytic[axis]
        );
    }
}

#[test]
fn student_t_outer_hessian_matches_finite_difference_of_the_gradient() {
    const BAR: f64 = 1.0e-3;
    const STEP: f64 = 1.0e-4;
    let base = probe_theta();
    let analytic = match evaluate(
        base,
        crate::estimate::reml::reml_outer_engine::EvalMode::ValueGradientHessian,
    )
    .hessian
    {
        gam_problem::HessianValue::Dense(hessian) => hessian,
        gam_problem::HessianValue::Operator(_) => {
            panic!("the Student-t gate needs the dense analytic outer Hessian; got an operator")
        }
        gam_problem::HessianValue::Unavailable => {
            panic!("the Student-t criterion returned no Hessian in ValueGradientHessian mode")
        }
    };
    assert_eq!(analytic.dim(), (3, 3));
    let mut reference = Array2::<f64>::zeros((3, 3));
    for axis in 0..3 {
        let mut plus = base;
        let mut minus = base;
        plus[axis] += STEP;
        minus[axis] -= STEP;
        let column = (analytic_gradient(plus) - analytic_gradient(minus)) / (2.0 * STEP);
        reference.column_mut(axis).assign(&column);
    }
    let reference = 0.5 * (&reference + &reference.t());
    let scale = reference.iter().map(|v| v * v).sum::<f64>().sqrt();
    for i in 0..3 {
        for j in 0..3 {
            let error = (analytic[[i, j]] - reference[[i, j]]).abs() / scale;
            assert!(
                error < BAR,
                "outer Hessian entry ({i},{j}): analytic {:.10e} vs finite difference \
                 {:.10e} (error {error:.3e} of the Hessian norm exceeds {BAR:.1e})\n\
                 analytic:\n{analytic:?}\nfinite difference:\n{reference:?}",
                analytic[[i, j]],
                reference[[i, j]]
            );
        }
    }
}

// ---------------------------------------------------------------------------
// End-to-end fits.
// ---------------------------------------------------------------------------

#[test]
fn student_t_fit_recovers_scale_and_degrees_of_freedom() {
    const SIGMA: f64 = 0.5;
    const NU: f64 = 4.0;
    let t = rand_distr::StudentT::new(NU).expect("valid degrees of freedom");
    let (xs, y) = simulate(3000, 4_004, |rng| SIGMA * t.sample(rng));
    let (x, penalty) = sine_design(&xs);
    let fitted = fit(&x, &y, &penalty, student_t_family(1.0, 1.0));
    let (sigma_hat, nu_hat) = fitted_student_t(&fitted);
    assert!(
        (sigma_hat / SIGMA).ln().abs() < 0.1,
        "sigma_hat = {sigma_hat} should recover sigma = {SIGMA}"
    );
    assert!(
        (nu_hat / NU).ln().abs() < 0.4,
        "nu_hat = {nu_hat} should recover nu = {NU}"
    );
    let mse = truth_mse(&x, &xs, &fitted.beta);
    assert!(mse < 0.01, "Student-t fit truth MSE {mse} is too large");
}

#[test]
fn student_t_fit_reaches_the_gaussian_limit_on_gaussian_data() {
    const SIGMA: f64 = 0.3;
    let normal = rand_distr::Normal::new(0.0, SIGMA).expect("valid normal");
    let (xs, y) = simulate(3000, 1_001, |rng| normal.sample(rng));
    let (x, penalty) = sine_design(&xs);
    let gaussian = fit(&x, &y, &penalty, gaussian_family());
    let student = fit(&x, &y, &penalty, student_t_family(1.0, 1.0));
    let (sigma_hat, nu_hat) = fitted_student_t(&student);
    assert!(
        nu_hat > 30.0,
        "Gaussian data must drive nu to the Gaussian limit; got nu_hat = {nu_hat}"
    );
    assert!(
        (sigma_hat / gaussian.standard_deviation).ln().abs() < 0.02,
        "Student-t sigma_hat = {sigma_hat} must match the Gaussian scale {}",
        gaussian.standard_deviation
    );
    let difference = (x.dot(&student.beta) - x.dot(&gaussian.beta))
        .iter()
        .fold(0.0_f64, |acc, v| acc.max(v.abs()));
    assert!(
        difference < 0.1 * SIGMA,
        "the Student-t mean must match the Gaussian mean in the Gaussian limit; max \
         difference {difference}"
    );
}

/// Heavy-tailed contamination: 5% of rows carry a scaled `t(1.5)` spike. The
/// Gaussian fit spends its scale on the spikes; the Student-t fit discounts
/// them and must track the truth at least as well.
#[test]
fn student_t_fit_is_robust_to_heavy_tailed_outliers() {
    let n = 300usize;
    let normal = rand_distr::Normal::new(0.0, 0.3).expect("valid normal");
    let spike = rand_distr::StudentT::new(1.5).expect("valid degrees of freedom");
    let (xs, mut y) = simulate(n, 303, |rng| normal.sample(rng));
    let mut rng = StdRng::seed_from_u64(304);
    for i in (0..n).step_by(20) {
        y[i] += 5.0 * spike.sample(&mut rng);
    }
    let (x, penalty) = sine_design(&xs);
    let gaussian = fit(&x, &y, &penalty, gaussian_family());
    let student = fit(&x, &y, &penalty, student_t_family(1.0, 1.0));
    let gaussian_mse = truth_mse(&x, &xs, &gaussian.beta);
    let student_mse = truth_mse(&x, &xs, &student.beta);
    let (_, nu_hat) = fitted_student_t(&student);
    assert!(
        student_mse < gaussian_mse,
        "Student-t truth MSE {student_mse} must beat Gaussian {gaussian_mse} under \
         heavy-tailed contamination (nu_hat = {nu_hat})"
    );
    assert!(student_mse < 0.02, "Student-t truth MSE {student_mse} is too large");
    assert!(
        nu_hat < 10.0,
        "contaminated data must pull nu into the heavy-tailed regime; got {nu_hat}"
    );
}
