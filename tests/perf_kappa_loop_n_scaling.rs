//! Measurement: is the spatial length-scale (κ) outer loop n-independent?
//!
//! The #1033b Chebyshev-in-ψ Gram tensor (`solver/psi_gram_tensor.rs`) makes
//! every κ-trial inside the spatial length-scale optimizer cost O(D²k²) — free
//! of the sample size n — by pre-expanding the conditioned design's Gram into a
//! certified polynomial in ψ = log κ. Only the *one-time* tensor build and the
//! *final* PIRLS assembly remain O(n). So as n grows, the wall-clock spent
//! inside the κ outer loop (beyond the single final fit) should stay roughly
//! flat rather than scaling with n.
//!
//! This harness isolates that κ-phase without any internal A/B switch (the
//! tensor auto-installs; there is no off-flag, by design). For each n it times:
//!   * `t_kappa`  — a fit with the κ outer loop ENABLED (several outer iters), and
//!   * `t_single` — the same fit with the loop DISABLED (one length-scale, one fit).
//! The difference `t_kappa - t_single` is the marginal cost of the κ search on
//! top of one ordinary fit, i.e. the κ-phase. If that marginal cost is
//! n-independent, the ratio across a 16× sweep in n is ~1, not ~16.
//!
//! Wall-clock on a shared cluster node is noisy, so this is a *measurement* I
//! read from the printed table — the only hard assertion is a catastrophe guard
//! (the κ-phase must not blow up super-linearly by an order of magnitude across
//! the sweep), which is a real tripwire, not a calibrated timing bound.

use gam::{
    FitRequest, FitResult, StandardFitRequest,
    basis::{CenterStrategy, MaternBasisSpec, MaternIdentifiability, MaternNu},
    estimate::FitOptions,
    smooth::{
        ShapeConstraint, SmoothBasisSpec, SmoothTermSpec, SpatialLengthScaleOptimizationOptions,
        TermCollectionSpec,
    },
    types::{InverseLink, LikelihoodSpec, ResponseFamily, StandardLink},
};
use ndarray::{Array1, Array2};
use std::time::Instant;

/// 1-D isotropic Gaussian-identity spatial fixture — exactly the tensor-eligible
/// path (`coord_dim == 1`, Gaussian + identity link). Deterministic truth keeps
/// this a geometry/timing check, not a stochastic power test.
fn simulate_1d_gaussian(n: usize) -> (Array2<f64>, Array1<f64>) {
    let mut x = Array2::<f64>::zeros((n, 1));
    let mut y = Array1::<f64>::zeros(n);
    for i in 0..n {
        let t = (i as f64) / (n as f64 - 1.0) * 6.0 - 3.0;
        x[[i, 0]] = t;
        // smooth signal + a tiny deterministic ripple so the optimizer has work
        y[i] = (1.3 * t).sin() + 0.25 * (3.0 * t).cos();
    }
    (x, y)
}

fn spec_1d() -> TermCollectionSpec {
    TermCollectionSpec {
        linear_terms: vec![],
        random_effect_terms: vec![],
        smooth_terms: vec![SmoothTermSpec {
            name: "matern_1d".to_string(),
            basis: SmoothBasisSpec::Matern {
                feature_cols: vec![0],
                spec: MaternBasisSpec {
                    center_strategy: CenterStrategy::FarthestPoint { num_centers: 24 },
                    periodic: None,
                    length_scale: 1.0,
                    nu: MaternNu::FiveHalves,
                    include_intercept: false,
                    double_penalty: true,
                    identifiability: MaternIdentifiability::CenterSumToZero,
                    aniso_log_scales: None,
                    nullspace_shrinkage_survived: None,
                },
                input_scales: None,
            },
            shape: ShapeConstraint::None,
            joint_null_rotation: None,
        }],
    }
}

fn fit_options() -> FitOptions {
    FitOptions {
        latent_cloglog: None,
        mixture_link: None,
        optimize_mixture: false,
        sas_link: None,
        optimize_sas: false,
        compute_inference: false,
        skip_rho_posterior_inference: false,
        max_iter: 30,
        tol: 1e-6,
        nullspace_dims: vec![],
        linear_constraints: None,
        firth_bias_reduction: false,
        adaptive_regularization: None,
        penalty_shrinkage_floor: None,
        rho_prior: Default::default(),
        kronecker_penalty_system: None,
        kronecker_factored: None,
    }
}

fn run_fit(n: usize, kappa_enabled: bool) -> f64 {
    let (x, y) = simulate_1d_gaussian(n);
    let weights = Array1::ones(n);
    let offset = Array1::zeros(n);
    let kappa_options = SpatialLengthScaleOptimizationOptions {
        enabled: kappa_enabled,
        // Several outer iterations so the κ-phase is a measurable share of the
        // total when enabled; one fit's worth of work when disabled.
        max_outer_iter: if kappa_enabled { 6 } else { 1 },
        rel_tol: 1e-5,
        log_step: std::f64::consts::LN_2,
        min_length_scale: 1e-2,
        max_length_scale: 1e2,
        pilot_subsample_threshold: 0,
    };

    let t0 = Instant::now();
    let result = gam::fit_model(FitRequest::Standard(StandardFitRequest {
        data: x,
        y,
        weights,
        offset,
        spec: spec_1d(),
        family: LikelihoodSpec::new(
            ResponseFamily::Gaussian,
            InverseLink::Standard(StandardLink::Identity),
        ),
        options: fit_options(),
        kappa_options,
        wiggle: None,
        coefficient_groups: Vec::new(),
        penalty_block_gamma_priors: Vec::new(),
        latent_coord: None,
        _marker: std::marker::PhantomData,
    }))
    .expect("1-D Matérn Gaussian fit should converge");
    let dt = t0.elapsed().as_secs_f64();

    let fitted = match result {
        FitResult::Standard(s) => s,
        _ => panic!("expected Standard fit result"),
    };
    assert!(
        fitted.fit.beta.iter().all(|v: &f64| v.is_finite()),
        "fit at n={n} (kappa={kappa_enabled}) produced non-finite coefficients"
    );
    dt
}

#[test]
fn kappa_outer_loop_is_n_independent() {
    // 16× sweep in n. Warm one fit first so allocator / one-time init costs do
    // not land on the first measured point.
    let _warm = run_fit(1000, true);

    let ns = [2_000usize, 8_000, 32_000];
    let mut kappa_phase = Vec::with_capacity(ns.len());

    eprintln!(
        "[kappa-n-scaling] {:>8}  {:>10}  {:>10}  {:>12}",
        "n", "t_kappa_s", "t_single_s", "kappa_phase_s"
    );
    for &n in &ns {
        // Best-of-2 to damp shared-node scheduling jitter.
        let t_kappa = run_fit(n, true).min(run_fit(n, true));
        let t_single = run_fit(n, false).min(run_fit(n, false));
        let phase = (t_kappa - t_single).max(0.0);
        kappa_phase.push(phase);
        eprintln!(
            "[kappa-n-scaling] {:>8}  {:>10.4}  {:>10.4}  {:>12.4}",
            n, t_kappa, t_single, phase
        );
    }

    // The headline ratio: κ-phase at the largest n vs the smallest. If the
    // κ-trials are n-free, this is ~O(1); if they still scale with n it tracks
    // the 16× growth in sample size.
    let first = kappa_phase.first().copied().unwrap_or(0.0).max(1e-4);
    let last = kappa_phase.last().copied().unwrap_or(0.0).max(1e-4);
    let n_ratio = (ns.last().unwrap() / ns.first().unwrap()) as f64; // 16
    let phase_ratio = last / first;
    eprintln!(
        "[kappa-n-scaling] n grew {n_ratio:.0}× ; kappa-phase grew {phase_ratio:.2}× \
         (n-independent ⇒ ~1×, n-linear ⇒ ~{n_ratio:.0}×)"
    );

    // Catastrophe guard only (not a calibrated timing bound): the κ-phase must
    // not scale super-linearly. Even with timing noise, an n-free κ loop cannot
    // grow faster than n itself — half the sample-size growth is a generous,
    // noise-tolerant tripwire that still fails loudly if the tensor lane
    // silently regresses to a per-trial n-pass.
    assert!(
        phase_ratio <= 0.5 * n_ratio,
        "kappa outer-loop phase grew {phase_ratio:.2}× across a {n_ratio:.0}× \
         increase in n — that is super-linear-ish and suggests the #1033b \
         Gram-tensor lane is no longer serving n-free κ-trials (per-trial work \
         fell back to an O(n) pass)"
    );
}
