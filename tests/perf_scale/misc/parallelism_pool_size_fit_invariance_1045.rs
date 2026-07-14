//! #1045 correctness gate for the parallel thread-pool-sizing lever.
//!
//! The candidate perf change for #1045 is to shrink the rayon worker pool so
//! the per-fit `crossbeam_epoch` bookkeeping (which scales with the number of
//! pinned participants) stops dominating the small-`n` spatial-kappa inner
//! loop. That is only admissible if it does NOT move the fit: a smaller pool
//! changes how the non-deterministic `into_par_iter(...).try_reduce(...)` row
//! folds in the BMS objective/gradient reassociate their floating-point sums,
//! and a perturbed sum fed into the iterative REML optimizer near a flat
//! optimum can in principle steer it to a different `(ρ, λ)` selection.
//!
//! This test fits ONE fixed Bernoulli-marginal-slope matern-logslope model
//! twice — once on a wide worker pool and once on a narrow one — using scoped
//! `rayon::ThreadPool::install`, which sets `rayon::current_num_threads()` for
//! both the codebase's `into_par_iter` folds and faer's `Par::rayon(0)`
//! backend inside the closure. It asserts the REML-selected smoothing
//! parameters, the coefficients, the REML score, and the outer-iteration path
//! all agree to a tight tolerance. If they diverge, pool-sizing is a
//! correctness regression (not a perf win) and must be rejected in favour of
//! per-site deterministic gating.

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
                length_scale: gam::terms::basis::MaternLengthScale::fixed(1.0),
                nu: MaternNu::ThreeHalves,
                include_intercept: false,
                double_penalty: false,
                identifiability: MaternIdentifiability::default(),
                aniso_log_scales: None,
            },
            input_scale: None,
        },
        shape: ShapeConstraint::None,
        joint_null_rotation: None,
    }
}

fn erf_approx(x: f64) -> f64 {
    // Abramowitz-Stegun 7.1.26 — only used to synthesize labels, identical
    // across both fits so it cannot itself introduce a difference.
    let t = 1.0 / (1.0 + 0.327_591_1 * x.abs());
    let y = 1.0
        - (((((1.061_405_429 * t - 1.453_152_027) * t) + 1.421_413_741) * t - 0.284_496_736) * t
            + 0.254_829_592)
            * t
            * (-x * x).exp();
    let sign = if x < 0.0 { -1.0 } else { 1.0 };
    sign * y
}

fn build(n: usize, centers: usize) -> (Array2<f64>, BernoulliMarginalSlopeTermSpec) {
    let mut rng = StdRng::seed_from_u64(0x9797_0001);
    let mut data = Array2::<f64>::zeros((n, 2));
    for i in 0..n {
        data[[i, 0]] = rng.random_range(-2.0..2.0);
        data[[i, 1]] = rng.random_range(-2.0..2.0);
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

/// The four scalars + two vectors that pin down "which optimum REML selected".
struct FitDigest {
    log_lambdas: Array1<f64>,
    beta: Array1<f64>,
    reml_score: f64,
    outer_iterations: usize,
}

fn fit_once(data: &Array2<f64>, spec: &BernoulliMarginalSlopeTermSpec) -> FitDigest {
    let request = FitRequest::BernoulliMarginalSlope(BernoulliMarginalSlopeFitRequest {
        data: data.view(),
        spec: spec.clone(),
        options: BlockwiseFitOptions::default(),
        kappa_options: SpatialLengthScaleOptimizationOptions::default(),
        policy: ResourcePolicy::default_library(),
    });
    match fit_model(request) {
        Ok(FitResult::BernoulliMarginalSlope(out)) => FitDigest {
            log_lambdas: out.fit.log_lambdas.clone(),
            beta: out.fit.beta.clone(),
            reml_score: out.fit.reml_score,
            outer_iterations: out.fit.outer_iterations,
        },
        Ok(_) => panic!("wrong FitResult variant"),
        Err(e) => panic!("fit failed: {e}"),
    }
}

fn fit_on_pool(
    threads: usize,
    data: &Array2<f64>,
    spec: &BernoulliMarginalSlopeTermSpec,
) -> FitDigest {
    // Match faer's backend to the scoped pool too. `faer::Par::rayon(0)`
    // resolves to `current_num_threads()` *at the call site*, so it must be set
    // here (inside the install scope below would still read the global default
    // captured at `set_global_parallelism` time). The two fits run
    // sequentially, so mutating the global faer parallelism per fit is safe.
    let pool = rayon::ThreadPoolBuilder::new()
        .num_threads(threads)
        .build()
        .unwrap_or_else(|e| panic!("failed to build {threads}-thread pool: {e}"));
    pool.install(|| {
        faer::set_global_parallelism(faer::Par::rayon(threads));
        fit_once(data, spec)
    })
}

#[test]
fn bms_matern_fit_is_invariant_to_worker_pool_size() {
    // NB: deliberately do NOT call `gam::init_parallelism()` here — it would
    // freeze faer at `Par::rayon(current_num_threads())` (the global pool size)
    // before the per-pool `set_global_parallelism` calls below, defeating the
    // narrow-pool arm. Each `fit_on_pool` sets faer to match its scoped pool.
    let n = 1200;
    let centers = 4;
    let (data, spec) = build(n, centers);

    // Wide pool: as many workers as the host gives the global pool (the status
    // quo), floored at 4 so the gate always has room for a strictly-narrower
    // pool even on the small (4-vCPU) CI runners. Narrow pool: the #1045
    // candidate — a shrunk worker pool. Both pools are ≥2 workers so both fits
    // take the *parallel* row-reduction path (a `narrow == 1` pool would instead
    // fork onto the serial fold, testing serial-vs-parallel rather than the
    // pool-size invariance #1045 actually needs). `wide` is a rayon worker count,
    // not a core count, so building it does not require that many physical cores;
    // `narrow < wide` therefore holds on every host with ≥1 core and the
    // invariance assertions below always run.
    let wide = rayon::current_num_threads().max(4);
    let narrow = 2usize;
    assert!(
        narrow < wide,
        "wide worker pool ({wide}) must exceed the narrow pool ({narrow}) to exercise the invariance gate"
    );

    let a = fit_on_pool(wide, &data, &spec);
    let b = fit_on_pool(narrow, &data, &spec);

    // The convergence path itself must not fork: a different iteration count
    // means the smaller pool steered the optimizer elsewhere. (Both fits are
    // certified by construction — fit existence is the sealed convergence
    // proof, SPEC 20.)
    assert_eq!(
        a.outer_iterations, b.outer_iterations,
        "outer_iterations diverged between pools: {} ({wide}) vs {} ({narrow})",
        a.outer_iterations, b.outer_iterations
    );

    // REML-selected smoothing parameters: the object the optimizer chose.
    assert_eq!(
        a.log_lambdas.len(),
        b.log_lambdas.len(),
        "log_lambda dimension changed between pools"
    );
    for (k, (la, lb)) in a.log_lambdas.iter().zip(b.log_lambdas.iter()).enumerate() {
        assert!(
            (la - lb).abs() <= 1e-8 * la.abs().max(1.0) + 1e-10,
            "REML-selected log_lambda[{k}] diverged between pools: {la} ({wide}) vs {lb} ({narrow})"
        );
    }

    // Coefficients at the selected optimum.
    assert_eq!(a.beta.len(), b.beta.len(), "beta dimension changed");
    let mut max_beta_gap = 0.0_f64;
    for (ba, bb) in a.beta.iter().zip(b.beta.iter()) {
        max_beta_gap = max_beta_gap.max((ba - bb).abs());
    }
    let beta_scale = a.beta.iter().fold(0.0_f64, |m, &v| m.max(v.abs())).max(1.0);
    assert!(
        max_beta_gap <= 1e-7 * beta_scale,
        "coefficients diverged between pools: max |Δβ| = {max_beta_gap} (β scale {beta_scale})"
    );

    // REML score at the optimum.
    assert!(
        (a.reml_score - b.reml_score).abs() <= 1e-7 * a.reml_score.abs().max(1.0),
        "REML score diverged between pools: {} ({wide}) vs {} ({narrow})",
        a.reml_score,
        b.reml_score
    );
}
