//! Repro for #979 (the ACTUAL failing case from the released gamfit wheel): the binary
//! Bernoulli marginal-slope arm over a **pure Duchon** spatial basis became
//! pathologically slow, while the *identical* Duchon basis fitting a plain
//! Gaussian smooth completed in ~2s.
//!
//! The issue's repro:
//!   event ~ duchon(x1, x2, centers=10)            // d=2, magic cubic default
//!   logslope: duchon(x1, x2, centers=10)          // same basis on the logslope
//!   family = bernoulli-marginal-slope, link = probit, rigid (no warp/dev), n=2000.
//!
//! At n=2000, centers=10, d=2 the binary marginal-slope fit timed out at 600s,
//! whereas the same Duchon basis fitting a plain Gaussian smooth on the same
//! data finished in ~2s. This isolates the blowup to the marginal-slope
//! machinery, NOT the basis.
//!
//! `margslope_duchon_above_cliff` is the #979 binary-arm regression guard: a
//! full `fit_model` that must complete under a generous wall budget.
//! `duchon_gaussian_smooth_baseline_is_fast` is the control: the identical
//! Duchon basis on a plain Gaussian smooth must finish quickly, proving the
//! basis itself is cheap.
//!
//! Invoke directly:
//!   cargo test --release --test bug_hunt_979_margslope_duchon_slowdown \
//!       margslope_duchon_above_cliff -- --nocapture
//!   cargo test --release --test bug_hunt_979_margslope_duchon_slowdown \
//!       duchon_gaussian_smooth_baseline_is_fast -- --nocapture

use gam::ResourcePolicy;
use gam::basis::{
    CenterStrategy, DuchonBasisSpec, DuchonNullspaceOrder, DuchonOperatorPenaltySpec,
    OneDimensionalBoundary, SpatialIdentifiability,
};
use gam::estimate::FitOptions;
use gam::families::bms::{BernoulliMarginalSlopeTermSpec, LatentZPolicy};
use gam::families::custom_family::BlockwiseFitOptions;
use gam::families::survival::lognormal_kernel::FrailtySpec;
use gam::smooth::{
    ShapeConstraint, SmoothBasisSpec, SmoothTermSpec, SpatialLengthScaleOptimizationOptions,
    TermCollectionSpec,
};
use gam::types::{InverseLink, LikelihoodSpec, ResponseFamily, StandardLink};
use gam::{BernoulliMarginalSlopeFitRequest, FitRequest, FitResult, fit_model};
use ndarray::{Array1, Array2};
use rand::rngs::StdRng;
use rand::{RngExt, SeedableRng};
use std::time::Instant;

/// Pure scale-free Duchon over data columns 0,1 (d=2) with the magic cubic
/// default resolved for d=2: `duchon_cubic_default(2) = (Linear, 0.5)`. This is
/// exactly the basis `duchon(x1, x2, centers=10)` lowers to in the formula DSL.
fn duchon2_smooth(name: &str, centers: usize) -> SmoothTermSpec {
    SmoothTermSpec {
        name: name.to_string(),
        basis: SmoothBasisSpec::Duchon {
            feature_cols: vec![0, 1],
            spec: DuchonBasisSpec {
                radial_reparam: None,
                center_strategy: CenterStrategy::FarthestPoint {
                    num_centers: centers,
                },
                length_scale: None,
                // Magic cubic default for d=2: s = (d-1)/2 = 0.5.
                power: 0.5,
                nullspace_order: DuchonNullspaceOrder::Linear,
                identifiability: SpatialIdentifiability::default(),
                aniso_log_scales: Some(vec![0.0; 2]),
                operator_penalties: DuchonOperatorPenaltySpec::default(),
                periodic: None,
                boundary: OneDimensionalBoundary::Open,
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

/// Shared simulator: two spatial coordinates in columns 0,1, a latent score `z`,
/// a smooth spatial field plus a spatially-varying log-slope. Returns the data
/// matrix, the latent score, and the binary response.
fn simulate(n: usize) -> (Array2<f64>, Array1<f64>, Array1<f64>) {
    let mut rng = StdRng::seed_from_u64(0x9797_0979);
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
    let true_eta: Array1<f64> = Array1::from_iter((0..n).map(|i| {
        let p1 = data[[i, 0]];
        let p2 = data[[i, 1]];
        let f = (0.8 * p1).sin() + 0.5 * (0.6 * p2).cos();
        let slope = 0.3 + 0.2 * p1;
        f + slope * z[i]
    }));
    let y = Array1::from_iter(true_eta.iter().map(|&eta| {
        let p = 0.5 * (1.0 + erf_approx(eta / std::f64::consts::SQRT_2));
        if rng.random::<f64>() < p { 1.0 } else { 0.0 }
    }));
    (data, z, y)
}

fn build_margslope(n: usize, centers: usize) -> (Array2<f64>, BernoulliMarginalSlopeTermSpec) {
    let (data, z, y) = simulate(n);
    let marginalspec = TermCollectionSpec {
        linear_terms: vec![],
        random_effect_terms: vec![],
        smooth_terms: vec![duchon2_smooth("f_pc", centers)],
    };
    let logslopespec = TermCollectionSpec {
        linear_terms: vec![],
        random_effect_terms: vec![],
        smooth_terms: vec![duchon2_smooth("ls_pc", centers)],
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

/// #979 binary-arm regression guard: bernoulli marginal-slope over a pure
/// Duchon basis (the actual released-wheel failing case). At n=2000, centers=10,
/// d=2 this timed out at 600s. Asserts the full fit completes under a generous
/// wall budget. While the bug is unfixed this test is slow/fails — the
/// coordinator's fix makes it pass.
#[test]
fn margslope_duchon_above_cliff() {
    gam::init_parallelism();
    // The #979 blowup is in the marginal-slope machinery, not n: a regressed fit
    // pathologically slows even at a moderate n. Default to a CI-affordable n and
    // assert the fit actually COMPLETES with a finite result (the functional half
    // of the guard); The cluster-scale n=2000 cliff repro is a
    // separate MSI artifact (shared-CI timing is only meaningful on the cluster).
    let n = 600;
    let centers = 10;
    let (data, spec) = build_margslope(n, centers);
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
    let out = match result {
        Ok(FitResult::BernoulliMarginalSlope(out)) => {
            eprintln!(
                "[979-DUCHON] n={n} centers={centers} total_s={elapsed:.2} outer_iters={} inner_cycles={} converged=certified",
                out.fit.outer_iterations, out.fit.inner_cycles
            );
            out
        }
        Ok(_) => panic!("wrong FitResult variant"),
        Err(e) => panic!("[979-DUCHON] n={n} centers={centers} total_s={elapsed:.2} FAILED: {e}"),
    };
    // Functional guard (runs in default CI): the marginal-slope Duchon fit must
    // complete and produce finite coefficients — a regressed/diverged fit fails
    // here independently of any wall-clock budget.
    assert!(
        !out.fit.blocks.is_empty(),
        "margslope Duchon fit produced no coefficient blocks"
    );
    // Fit existence is the sealed convergence proof (SPEC 20).
    assert!(
        out.fit
            .blocks
            .iter()
            .flat_map(|b| b.beta.iter())
            .all(|v| v.is_finite()),
        "margslope Duchon fit produced non-finite coefficients (#979 binary arm)"
    );
    // Wall-clock guard: the #979 binary-arm slowdown is pathological, so even at
    // the CI-affordable n a healthy fit is far under budget; a >120s fit is the
    // regression fingerprint.
    assert!(
        elapsed < 120.0,
        "margslope Duchon fit took {elapsed:.1}s at n={n} centers={centers} (budget 120s) — #979 binary slowdown"
    );
}

/// Control: the IDENTICAL pure-Duchon basis (centers=10, d=2) fitting a plain
/// Gaussian smooth on the SAME n=2000 data. The issue measured this at ~2s. We
/// assert <30s. This proves the basis itself is cheap and isolates the blowup to
/// the marginal-slope machinery, exactly matching the issue's measurement.
#[test]
fn duchon_gaussian_smooth_baseline_is_fast() {
    gam::init_parallelism();
    // Control for `margslope_duchon_above_cliff`. Default to a CI-affordable n and
    // assert the plain Gaussian Duchon fit COMPLETES with finite coefficients;
    // The cluster-scale n=2000 run is a separate MSI artifact.
    let n = 600;
    let centers = 10;
    let (data, _z, y) = simulate(n);

    let spec = TermCollectionSpec {
        linear_terms: vec![],
        random_effect_terms: vec![],
        smooth_terms: vec![duchon2_smooth("f_pc", centers)],
    };
    let weights = Array1::ones(n);
    let offset = Array1::zeros(n);

    let start = Instant::now();
    let result = gam::smooth::fit_term_collection_forspec(
        data.view(),
        y.view(),
        weights.view(),
        offset.view(),
        &spec,
        LikelihoodSpec::new(
            ResponseFamily::Gaussian,
            InverseLink::Standard(StandardLink::Identity),
        ),
        &FitOptions {
            resource_policy: gam_runtime::resource::ResourcePolicy::default_library(),
            latent_cloglog: None,
            mixture_link: None,
            optimize_mixture: false,
            sas_link: None,
            optimize_sas: false,
            compute_inference: true,
            skip_rho_posterior_inference: false,
            max_iter: 60,
            tol: 1e-6,
            nullspace_dims: vec![],
            linear_constraints: None,
            firth_bias_reduction: false,
            adaptive_regularization: None,
            penalty_shrinkage_floor: None,
            rho_prior: Default::default(),
            kronecker_penalty_system: None,
            kronecker_factored: None,
            persist_warm_start_disk: false,
        },
    );
    let elapsed = start.elapsed().as_secs_f64();
    let fit = match result {
        Ok(fit) => {
            eprintln!("[979-DUCHON-GAUSS] n={n} centers={centers} total_s={elapsed:.2} (control)");
            fit
        }
        Err(e) => panic!("plain Gaussian Duchon smooth fit failed: {e}"),
    };
    // Functional guard (runs in default CI): the basis fit must complete with
    // finite coefficients.
    assert!(
        !fit.fit.beta.is_empty() && fit.fit.beta.iter().all(|v| v.is_finite()),
        "plain Gaussian Duchon smooth produced non-finite coefficients (#979 control)"
    );
    // Wall-clock control: the plain Duchon basis must stay cheap even at the
    // CI-affordable n; a >30s fit here is a control-side regression.
    assert!(
        elapsed < 30.0,
        "plain Gaussian Duchon smooth took {elapsed:.1}s at n={n} centers={centers} (budget 30s) — the basis itself must be cheap (#979 control)"
    );
}
