//! Smoke test: bernoulli marginal-slope must complete quickly on a small,
//! well-conditioned problem.
//!
//! This is a guard against regressions in the inner-Newton / outer-κ
//! interplay that turn small problems into slow problems. The biobank-scale
//! reproducers (`tests/biobank_margslope_repro.rs`,
//! `tests/margslope_inner_pirls_scaling.rs`) are `#[ignore]`d because they
//! sweep n up to 100k+ — this file runs a single n=2000 fit and asserts
//! both convergence and a wall-clock budget that is generous for a healthy
//! solver but tight enough to catch a slow-loop regression like the CTN
//! exact-fn rejection cycle that recently cost ≥14h of CI.

use gam::families::bernoulli_marginal_slope::{
    BernoulliMarginalSlopeTermSpec, DeviationBlockConfig, LatentZPolicy,
};
use gam::families::custom_family::BlockwiseFitOptions;
use gam::families::lognormal_kernel::FrailtySpec;
use gam::resource::ResourcePolicy;
use gam::terms::basis::{BSplineBasisSpec, BSplineKnotSpec};
use gam::terms::smooth::{
    ShapeConstraint, SmoothBasisSpec, SmoothTermSpec, SpatialLengthScaleOptimizationOptions,
    TermCollectionSpec,
};
use gam::types::{InverseLink, LinkFunction};
use gam::{BernoulliMarginalSlopeFitRequest, FitRequest, FitResult, fit_model};
use ndarray::{Array1, Array2};
use rand::rngs::StdRng;
use rand::{RngExt, SeedableRng};
use std::time::Instant;

const SEED: u64 = 0x5CA1_AB1E_5C0F_E5A1;

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

fn build_problem(n: usize, flex: bool) -> (Array2<f64>, BernoulliMarginalSlopeTermSpec) {
    let mut rng = StdRng::seed_from_u64(SEED.wrapping_add(n as u64));
    let x_raw: Vec<f64> = (0..n).map(|_| rng.random_range(0.0..1.0)).collect();
    let mut data = Array2::<f64>::zeros((n, 1));
    for (i, &xi) in x_raw.iter().enumerate() {
        data[[i, 0]] = xi;
    }

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

    let two_pi = std::f64::consts::TAU;
    let true_eta: Array1<f64> = Array1::from_iter(
        x_raw
            .iter()
            .zip(z.iter())
            .map(|(&xi, &zi)| (two_pi * xi).sin() + 0.5 * (two_pi * 2.0 * xi).cos() + 0.3 * zi),
    );
    let y = Array1::from_iter(true_eta.iter().map(|&eta| {
        let p = 0.5 * (1.0 + erf_approx(eta / std::f64::consts::SQRT_2));
        if rng.random::<f64>() < p { 1.0 } else { 0.0 }
    }));
    let weights = Array1::ones(n);
    let marginal_offset = Array1::<f64>::zeros(n);
    let logslope_offset = Array1::<f64>::zeros(n);

    let smooth = SmoothTermSpec {
        name: "f_x".to_string(),
        basis: SmoothBasisSpec::BSpline1D {
            feature_col: 0,
            spec: BSplineBasisSpec {
                degree: 3,
                penalty_order: 2,
                knotspec: BSplineKnotSpec::Generate {
                    data_range: (0.0, 1.0),
                    num_internal_knots: 8,
                },
                double_penalty: false,
                identifiability: Default::default(),
            },
        },
        shape: ShapeConstraint::None,
    };
    let marginalspec = TermCollectionSpec {
        linear_terms: vec![],
        random_effect_terms: vec![],
        smooth_terms: vec![smooth],
    };
    let logslopespec = TermCollectionSpec {
        linear_terms: vec![],
        random_effect_terms: vec![],
        smooth_terms: vec![],
    };
    let (score_warp, link_dev) = if flex {
        let dev_cfg = DeviationBlockConfig::triple_penalty_default();
        (Some(dev_cfg.clone()), Some(dev_cfg))
    } else {
        (None, None)
    };
    let spec = BernoulliMarginalSlopeTermSpec {
        y,
        weights,
        z,
        base_link: InverseLink::Standard(LinkFunction::Probit),
        marginalspec,
        logslopespec,
        marginal_offset,
        logslope_offset,
        frailty: FrailtySpec::None,
        score_warp,
        link_dev,
        latent_z_policy: LatentZPolicy::exploratory_fit_weighted(),
    };
    (data, spec)
}

fn run_one(flex: bool, label: &str, budget_s: f64) {
    gam::init_parallelism();
    let (data, spec) = build_problem(2000, flex);
    let options = BlockwiseFitOptions::default();
    let kappa_options = SpatialLengthScaleOptimizationOptions::default();
    let policy = ResourcePolicy::default_library();
    let request = FitRequest::BernoulliMarginalSlope(BernoulliMarginalSlopeFitRequest {
        data: data.view(),
        spec,
        options,
        kappa_options,
        policy,
    });

    let start = Instant::now();
    let result = fit_model(request);
    let elapsed = start.elapsed().as_secs_f64();

    let out = match result {
        Ok(FitResult::BernoulliMarginalSlope(out)) => out,
        Ok(_) => panic!("{label}: wrong FitResult variant"),
        Err(e) => panic!("{label}: fit failed at n=2000 flex={flex}: {e}"),
    };

    eprintln!(
        "[{label}] n=2000 flex={flex} elapsed_s={elapsed:.3} outer_iters={} inner_cycles={} converged={}",
        out.fit.outer_iterations, out.fit.inner_cycles, out.fit.outer_converged
    );

    assert!(
        out.fit.outer_converged,
        "{label}: outer optimizer did not converge on small good-condition problem"
    );
    assert!(
        elapsed < budget_s,
        "{label}: small good-condition fit took {elapsed:.2}s, expected <{budget_s:.0}s — slow-loop regression?"
    );
}

#[test]
fn margslope_rigid_small_good_condition_completes_quickly() {
    // Rigid probit: closed-form vectorized inner solve. n=2000 should
    // complete in well under a second of compute on any reasonable
    // hardware; allow 30s to absorb CI-runner variance.
    run_one(false, "MS-RIGID-SMOKE", 30.0);
}

#[test]
fn margslope_flex_small_good_condition_completes_quickly() {
    // Flex probit: cubic score_warp + link_dev deviation blocks exercise
    // the per-row sextic-kernel cell evaluator at every inner-PIRLS
    // iteration. At n=2000 this is the biobank production code path on
    // small data — must still finish well within 60s on a healthy solver.
    run_one(true, "MS-FLEX-SMOKE", 60.0);
}
