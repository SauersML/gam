//! Repro for #979: bernoulli-marginal-slope with a *matern logslope smooth*
//! (not intercept-only) became pathologically slow on 0.1.189 vs 0.1.156.
//!
//! The issue's repro:
//!   event ~ matern(PC1, PC2, centers=4)
//!   logslope: matern(PC1, PC2, centers=4)
//!   family = bernoulli-marginal-slope, link = probit, rigid (no warp/dev).
//!
//! At centers=4, n~2500 the fit's continuation pre-warm alone took ~34s and
//! the inner joint-Newton ~28s (cycles=17) — absurd for so small a basis.
//! This test rebuilds the same shape at a tractable size and asserts the
//! whole fit completes well under a wall-clock budget.

use gam::ResourcePolicy;
use gam::families::bms::{BernoulliMarginalSlopeTermSpec, LatentZPolicy};
use gam::families::custom_family::BlockwiseFitOptions;
use gam::families::survival::lognormal_kernel::FrailtySpec;
use gam::terms::basis::{CenterStrategy, MaternBasisSpec, MaternIdentifiability, MaternNu};
use gam::terms::smooth::{
    ShapeConstraint, SmoothBasisSpec, SmoothTermSpec, SpatialLengthScaleOptimizationOptions,
    TermCollectionSpec,
};
use gam::types::{InverseLink, StandardLink};
use gam::{BernoulliMarginalSlopeFitRequest, FitRequest, FitResult, fit_model};
use ndarray::{Array1, Array2};
use rand::rngs::StdRng;
use rand::{RngExt, SeedableRng};
use std::time::Instant;

fn matern_smooth(name: &str, centers: usize) -> SmoothTermSpec {
    SmoothTermSpec {
        name: name.to_string(),
        basis: SmoothBasisSpec::Matern {
            feature_cols: vec![0, 1],
            spec: MaternBasisSpec {
                center_strategy: CenterStrategy::EqualMass {
                    num_centers: centers,
                },
                periodic: None,
                length_scale: 1.0.into(),
                nu: MaternNu::ThreeHalves,
                include_intercept: false,
                double_penalty: false,
                identifiability: MaternIdentifiability::default(),
                aniso_log_scales: None,
            },
            input_scales: None,
        },
        shape: ShapeConstraint::None,
        joint_null_rotation: None,
    }
}

fn erf_approx(x: f64) -> f64 {
    let a1 = 0.254829592;
    let a2 = -0.284496736;
    let a3 = 1.421413741;
    let a4 = -1.453152027;
    let a5 = 1.061405429;
    let p = 0.3275911;
    let sign = if x < 0.0 { -1.0 } else { 1.0 };
    let ax = x.abs();
    let t = 1.0 / (1.0 + p * ax);
    let y = 1.0 - (((((a5 * t + a4) * t) + a3) * t + a2) * t + a1) * t * (-ax * ax).exp();
    sign * y
}

fn build(n: usize, centers: usize) -> (Array2<f64>, BernoulliMarginalSlopeTermSpec) {
    let mut rng = StdRng::seed_from_u64(0x9797_0001);
    // Two PCs as the spatial coordinates (columns 0,1).
    let mut data = Array2::<f64>::zeros((n, 2));
    for i in 0..n {
        data[[i, 0]] = rng.random_range(-2.0..2.0);
        data[[i, 1]] = rng.random_range(-2.0..2.0);
    }
    // Latent score z ~ N(0,1) via Box-Muller.
    let mut z = Array1::<f64>::zeros(n);
    let mut i = 0usize;
    while i < n {
        let u1: f64 = rng.random_range(1e-12..1.0);
        let u2: f64 = rng.random_range(0.0..1.0);
        let r = (-2.0 * u1.ln()).sqrt();
        let theta = 2.0 * std::f64::consts::PI * u2;
        z[i] = r * theta.cos();
        if i + 1 < n {
            z[i + 1] = r * theta.sin();
        }
        i += 2;
    }
    // Truth: smooth spatial field on PCs plus a marginal-slope contribution.
    let true_eta: Array1<f64> = Array1::from_iter((0..n).map(|i| {
        let p1 = data[[i, 0]];
        let p2 = data[[i, 1]];
        let f = (0.8 * p1).sin() + 0.5 * (0.6 * p2).cos();
        let slope = 0.3 + 0.2 * p1; // log-slope varies over space
        f + slope * z[i]
    }));
    let y = Array1::from_iter(true_eta.iter().map(|&eta| {
        let p = 0.5 * (1.0 + erf_approx(eta / std::f64::consts::SQRT_2));
        if rng.random::<f64>() < p { 1.0 } else { 0.0 }
    }));

    let marginalspec = TermCollectionSpec {
        linear_terms: vec![],
        random_effect_terms: vec![],
        smooth_terms: vec![matern_smooth("f_pc", centers)],
    };
    let logslopespec = TermCollectionSpec {
        linear_terms: vec![],
        random_effect_terms: vec![],
        smooth_terms: vec![matern_smooth("ls_pc", centers)],
    };
    let spec = BernoulliMarginalSlopeTermSpec {
        y,
        weights: Array1::ones(n),
        z,
        base_link: InverseLink::Standard(StandardLink::Probit),
        marginalspec,
        logslopespec,
        marginal_offset: Array1::<f64>::zeros(n),
        logslope_offset: Array1::<f64>::zeros(n),
        frailty: FrailtySpec::None,
        score_warp: None,
        link_dev: None,
        latent_z_policy: LatentZPolicy::exploratory_fit_weighted(),
        score_influence_jacobian: None,
    };
    (data, spec)
}

#[test]
fn margslope_matern_logslope_timing() {
    gam::init_parallelism();
    let n = 1500;
    let centers = 4;
    let (data, spec) = build(n, centers);
    let request = FitRequest::BernoulliMarginalSlope(BernoulliMarginalSlopeFitRequest {
        data: data.view(),
        spec,
        options: BlockwiseFitOptions::default(),
        kappa_options: SpatialLengthScaleOptimizationOptions::default(),
        policy: ResourcePolicy::default_library(),
    });
    let start = Instant::now();
    let result = fit_model(request);
    let elapsed = start.elapsed().as_secs_f64();
    // Fit existence is the sealed convergence proof (SPEC 20).
    match result {
        Ok(FitResult::BernoulliMarginalSlope(out)) => {
            eprintln!(
                "[979-REPRO] n={n} centers={centers} total_s={elapsed:.2} outer_iters={} inner_cycles={} converged=certified",
                out.fit.outer_iterations, out.fit.inner_cycles
            );
        }
        Ok(_) => panic!("wrong FitResult variant"),
        Err(e) => panic!("fit failed: {e}"),
    };
    assert!(
        elapsed < 60.0,
        "margslope matern-logslope fit took {elapsed:.1}s at n={n} centers={centers} (budget 60s)"
    );
}

/// Above-the-cliff profiling repro (#979): centers=12, n=2000 — the regime
/// where the multi-seed continuation pre-warm *fires* and the binary
/// marginal-slope fit became intractable (>360s). Asserts the fit completes
/// well under a generous wall budget — the regression guard for the binary
/// arm of #979. Invoke directly:
///   cargo test --release --test bug_hunt_979_margslope_matern_logslope_slowdown \
///       margslope_matern_logslope_above_cliff -- --nocapture
#[test]
fn margslope_matern_logslope_above_cliff() {
    gam::init_parallelism();
    let n = 2000;
    let centers = 12;
    let (data, spec) = build(n, centers);
    let request = FitRequest::BernoulliMarginalSlope(BernoulliMarginalSlopeFitRequest {
        data: data.view(),
        spec,
        options: BlockwiseFitOptions::default(),
        kappa_options: SpatialLengthScaleOptimizationOptions::default(),
        policy: ResourcePolicy::default_library(),
    });
    let start = Instant::now();
    let result = fit_model(request);
    let elapsed = start.elapsed().as_secs_f64();
    // Fit existence is the sealed convergence proof (SPEC 20).
    match result {
        Ok(FitResult::BernoulliMarginalSlope(out)) => {
            eprintln!(
                "[979-ABOVE-CLIFF] n={n} centers={centers} total_s={elapsed:.2} outer_iters={} inner_cycles={} converged=certified",
                out.fit.outer_iterations, out.fit.inner_cycles
            );
        }
        Ok(_) => panic!("wrong FitResult variant"),
        Err(e) => panic!(
            "above-cliff marginal-slope fit failed after {elapsed:.2}s at n={n} centers={centers}: {e}"
        ),
    };
    assert!(
        elapsed < 120.0,
        "above-cliff margslope fit took {elapsed:.1}s at n={n} centers={centers} (budget 120s) — #979 binary slowdown"
    );
}
