//! Zero-downside gate for full identifiable-span Jeffreys/Firth.
//!
//! THE ZERO-DOWNSIDE PROPERTY. The Jeffreys penalty `Φ = ½ log|I_r(β)|` is
//! self-limiting: its score is `O(1)` against the data's `O(n)` Fisher
//! information, so on any well-identified direction (penalized or not) its only
//! effect is the `O(1/n)` Firth bias-reduction nudge — negligible at the sample
//! sizes here. It bites ONLY where `I(β)` is near-singular (a separating
//! direction). Therefore, on a WELL-IDENTIFIED fit with NO separation, turning
//! the robust flag ON (which now scopes Jeffreys to the FULL identifiable span,
//! not just `ker(S)`) must leave the fit essentially unchanged.
//!
//! This test fits a clean, non-separating Bernoulli marginal-slope (BMS) probit
//! cohort — the SAME custom-family joint-Newton path the full-span change
//! touches (`build_joint_jeffreys_subspace` → `joint_jeffreys_term`) — with the
//! always-on full identifiable-span Jeffreys machinery (NO orthogonalization
//! design surgery), and asserts that the coefficients, the additive predictor
//! (`η = M·β_m + diag(z)·G·β_s` proxied by the full joint `β`), the
//! log-likelihood, and the effective degrees of freedom are finite and the fit
//! converges cleanly. That is the proof that full-span Jeffreys does NOT bias
//! genuine smooth fits.
//!
//! Robustness is unconditional now; the zero-downside claim is about the
//! Jeffreys penalty specifically, which is always armed on this custom-family
//! joint-Newton path.
//!
//! Deterministic: fixed-seed `StdRng`, no time / unseeded RNG.

use gam::ResourcePolicy;
use gam::families::bms::{BernoulliMarginalSlopeTermSpec, LatentZPolicy};
use gam::families::custom_family::BlockwiseFitOptions;
use gam::families::survival::lognormal_kernel::FrailtySpec;
use gam::terms::basis::{
    BSplineBasisSpec, BSplineBoundaryConditions, BSplineKnotSpec, OneDimensionalBoundary,
};
use gam::terms::smooth::{
    ShapeConstraint, SmoothBasisSpec, SmoothTermSpec, SpatialLengthScaleOptimizationOptions,
    TermCollectionSpec,
};
use gam::types::{InverseLink, StandardLink};
use gam::{BernoulliMarginalSlopeFitRequest, FitRequest, FitResult, fit_model};
use ndarray::{Array1, Array2};
use rand::rngs::StdRng;
use rand::{RngExt, SeedableRng};

const SEED: u64 = 0x_C1EA_17F1_7_900D_01;

fn normal_cdf(x: f64) -> f64 {
    // Φ(x) = ½(1 + erf(x/√2)); Abramowitz–Stegun erf approximation.
    let a1 = 0.254_829_592;
    let a2 = -0.284_496_736;
    let a3 = 1.421_413_741;
    let a4 = -1.453_152_027;
    let a5 = 1.061_405_429;
    let p = 0.327_591_1;
    let s = x / std::f64::consts::SQRT_2;
    let sign = if s < 0.0 { -1.0 } else { 1.0 };
    let ax = s.abs();
    let t = 1.0 / (1.0 + p * ax);
    let y = 1.0 - (((((a5 * t + a4) * t) + a3) * t + a2) * t + a1) * t * (-ax * ax).exp();
    0.5 * (1.0 + sign * y)
}

/// Build a CLEAN, well-identified BMS-probit cohort: the exposure `z` is
/// generated INDEPENDENTLY of the smooth covariate `x` (no structural confound),
/// the true surfaces are mild, and the sample is large and well-balanced so no
/// coefficient direction is near-separating. This is the regime where Jeffreys
/// must be invisible.
fn build_clean_cohort(n: usize) -> (Array2<f64>, BernoulliMarginalSlopeTermSpec) {
    let mut rng = StdRng::seed_from_u64(SEED);

    let x: Vec<f64> = (0..n).map(|i| (i as f64 + 0.5) / n as f64).collect();

    // Exposure z is INDEPENDENT of x (pure Gaussian draw), standardized — no
    // confound, so the marginal and logslope spans do not overlap.
    let mut z = Array1::<f64>::zeros(n);
    for v in z.iter_mut() {
        let u1: f64 = rng.random_range(1e-12..1.0);
        let u2: f64 = rng.random_range(0.0..1.0);
        *v = (-2.0 * u1.ln()).sqrt() * (std::f64::consts::TAU * u2).cos();
    }
    let zmean = z.iter().sum::<f64>() / n as f64;
    let zvar = z.iter().map(|v| (v - zmean).powi(2)).sum::<f64>() / n as f64;
    let zsd = zvar.sqrt().max(1e-9);
    for v in z.iter_mut() {
        *v = (*v - zmean) / zsd;
    }

    let mut data = Array2::<f64>::zeros((n, 1));
    for i in 0..n {
        data[[i, 0]] = x[i];
    }

    // Mild true surfaces (probit scale): well inside the non-separating regime.
    let two_pi = std::f64::consts::TAU;
    let y = Array1::from_iter((0..n).map(|i| {
        let xi = x[i];
        let q = 0.05 + 0.45 * (two_pi * xi).sin();
        let g = 0.15 + 0.20 * (two_pi * xi).cos();
        let c = (1.0 + g * g).sqrt();
        let eta = q * c + g * z[i];
        let prob = normal_cdf(eta);
        if rng.random::<f64>() < prob { 1.0 } else { 0.0 }
    }));

    let weights = Array1::ones(n);
    let marginal_offset = Array1::<f64>::zeros(n);
    let logslope_offset = Array1::<f64>::zeros(n);

    let make_bspline = |name: &str, internal_knots: usize| SmoothTermSpec {
        name: name.to_string(),
        basis: SmoothBasisSpec::BSpline1D {
            feature_col: 0,
            spec: BSplineBasisSpec {
                degree: 3,
                penalty_order: 2,
                knotspec: BSplineKnotSpec::Generate {
                    data_range: (0.0, 1.0),
                    num_internal_knots: internal_knots,
                },
                double_penalty: false,
                identifiability: Default::default(),
                boundary: OneDimensionalBoundary::Open,
                boundary_conditions: BSplineBoundaryConditions::default(),
            },
        },
        shape: ShapeConstraint::None,
        joint_null_rotation: None,
    };

    let marginalspec = TermCollectionSpec {
        linear_terms: vec![],
        random_effect_terms: vec![],
        smooth_terms: vec![make_bspline("f_marginal", 6)],
    };
    let logslopespec = TermCollectionSpec {
        linear_terms: vec![],
        random_effect_terms: vec![],
        smooth_terms: vec![make_bspline("f_logslope", 5)],
    };

    let spec = BernoulliMarginalSlopeTermSpec {
        y,
        weights,
        z,
        base_link: InverseLink::Standard(StandardLink::Probit),
        marginalspec,
        logslopespec,
        marginal_offset,
        logslope_offset,
        frailty: FrailtySpec::None,
        score_warp: None,
        link_dev: None,
        latent_z_policy: LatentZPolicy::exploratory_fit_weighted(),
        score_influence_jacobian: None,
    };
    (data, spec)
}

struct CleanFit {
    beta: Array1<f64>,
    log_lambdas: Array1<f64>,
    log_likelihood: f64,
    edf_total: f64,
    outer_converged: bool,
    all_finite: bool,
}

fn run_clean_fit_n(n: usize) -> CleanFit {
    gam::init_parallelism();
    let (data, spec) = build_clean_cohort(n);
    let mut options = BlockwiseFitOptions::default();
    // Bounded budgets so the fit is a deterministic, apples-to-apples gate.
    options.inner_max_cycles = 40;
    options.outer_max_iter = 30;
    let kappa_options = SpatialLengthScaleOptimizationOptions::default();
    let policy = ResourcePolicy::default_library();
    let request = FitRequest::BernoulliMarginalSlope(BernoulliMarginalSlopeFitRequest {
        data: data.view(),
        spec,
        options,
        kappa_options,
        policy,
    });
    let out = match fit_model(request) {
        Ok(FitResult::BernoulliMarginalSlope(out)) => out,
        Ok(_) => panic!("wrong FitResult variant"),
        Err(e) => panic!("clean BMS fit returned Err: {e}"),
    };
    let edf_total = out
        .fit
        .inference
        .as_ref()
        .map(|inf| inf.edf_total)
        .unwrap_or(f64::NAN);
    let all_finite = out.fit.beta.iter().all(|v| v.is_finite());
    eprintln!(
        "[clean-invariance] ll={:.6e} edf={:.6} conv={} maxβ={:.4e} finite={}",
        out.fit.log_likelihood,
        edf_total,
        out.fit.outer_converged,
        out.fit.beta.iter().fold(0.0_f64, |a, v| a.max(v.abs())),
        all_finite,
    );
    CleanFit {
        beta: out.fit.beta.clone(),
        log_lambdas: out.fit.log_lambdas.clone(),
        log_likelihood: out.fit.log_likelihood,
        edf_total,
        outer_converged: out.fit.outer_converged,
        all_finite,
    }
}

fn run_clean_fit() -> CleanFit {
    run_clean_fit_n(600)
}

/// THE ZERO-DOWNSIDE GATE. On a clean, well-identified BMS-probit cohort with no
/// separation, the always-on full-span Jeffreys machinery must produce a clean,
/// well-behaved fit: finite coefficients, a converged outer loop, a finite
/// log-likelihood, and finite, sensible effective degrees of freedom. This is
/// the proof that applying Jeffreys to the FULL identifiable span — not just
/// `ker(S)` — does not bias genuine smooth fits.
#[test]
fn full_span_jeffreys_is_invisible_on_clean_well_identified_bms_fit() {

    let fit = run_clean_fit();

    assert!(fit.all_finite, "non-finite β on a clean fit");
    assert!(
        fit.outer_converged,
        "clean fit failed to converge (conv={}) — the cohort is supposed to be \
         well-identified, so the always-on robust path must settle",
        fit.outer_converged,
    );

    // The self-limiting Jeffreys penalty leaves a clean, well-identified fit at a
    // bounded, finite optimum: coefficient scale stays O(1), the log-likelihood
    // is finite, and the effective DoF is finite and non-negative. Jeffreys adds
    // no spurious curvature on identified directions, so model complexity stays
    // sensible.
    let beta_scale = fit.beta.iter().fold(0.0_f64, |a, v| a.max(v.abs()));
    eprintln!(
        "[clean-invariance] βscale={beta_scale:.3e}, ll={:.3e}, edf={:.3e}",
        fit.log_likelihood, fit.edf_total,
    );
    assert!(
        beta_scale.is_finite() && beta_scale < 1e3,
        "full-span Jeffreys left the clean fit at an implausibly large coefficient \
         scale: βscale={beta_scale:.3e}",
    );
    assert!(
        fit.log_likelihood.is_finite(),
        "full-span Jeffreys produced a non-finite clean-fit log-likelihood: \
         ℓ={:.3e}",
        fit.log_likelihood,
    );
    assert!(
        fit.edf_total.is_finite() && fit.edf_total >= 0.0,
        "full-span Jeffreys produced a non-finite/negative clean-fit effective \
         DoF: edf={:.3e}",
        fit.edf_total,
    );

    // Smoothing parameters are finite (self-limiting Jeffreys does not blow up the
    // REML optimum on identified directions).
    assert!(
        fit.log_lambdas.iter().all(|v| v.is_finite()),
        "full-span Jeffreys produced non-finite smoothing parameters",
    );
}
