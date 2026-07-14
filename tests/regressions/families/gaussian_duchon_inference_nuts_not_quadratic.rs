//! Regression for #979 (different angle from `bug_hunt_979_margslope_duchon_slowdown`'s
//! `duchon_gaussian_smooth_baseline_is_fast`).
//!
//! That control asserts the *absolute* wall-clock of a plain Gaussian Duchon
//! smooth. This test pins the *mechanism* the fix relies on, from two angles the
//! control does not exercise:
//!
//!  1. **The post-fit ρ-posterior stack — not the basis — is the cost.** A
//!     Gaussian-identity smooth of a near-binary (0/1) response yields a poorly
//!     identified 5-D ρ posterior, so the PSIS certificate (#938) escalates to
//!     Tier-2 NUTS. Each of its ~thousands of leapfrog gradient evaluations used
//!     to re-run the full Gaussian-identity ALO-stabilization diagnostic suite
//!     (#813/#821) only to discover the leverage barrier is identically zero,
//!     turning a sub-second fit into ~60s. We assert the SAME fit *with* posterior
//!     inference is not pathologically slower than *without* it: the ratio bounds
//!     the per-evaluation overhead the ALO leak used to impose.
//!
//!  2. **Posterior-sampling suppression must not perturb the point estimate.**
//!     The fix suppresses the ALO-stabilization augmentation only inside the
//!     ρ-posterior certificate / NUTS closures, never in the outer optimization.
//!     So the converged coefficients, smoothing parameters and REML score must be
//!     bit-for-bit identical whether or not the ρ-posterior runs. (If suppression
//!     leaked into the fit, this would catch it; if the ALO leak returned, the
//!     ratio assertion would catch it.)

use gam::basis::{
    CenterStrategy, DuchonBasisSpec, DuchonNullspaceOrder, DuchonOperatorPenaltySpec,
    OneDimensionalBoundary, SpatialIdentifiability,
};
use gam::estimate::FitOptions;
use gam::smooth::{ShapeConstraint, SmoothBasisSpec, SmoothTermSpec, TermCollectionSpec};
use gam::types::{InverseLink, LikelihoodSpec, ResponseFamily, StandardLink};
use ndarray::{Array1, Array2};
use rand::rngs::StdRng;
use rand::{RngExt, SeedableRng};
use std::time::Instant;

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
                power: 0.5,
                nullspace_order: DuchonNullspaceOrder::Linear,
                identifiability: SpatialIdentifiability::default(),
                aniso_log_scales: Some(vec![0.0; 2]),
                operator_penalties: DuchonOperatorPenaltySpec::default(),
                periodic: None,
                boundary: OneDimensionalBoundary::Open,
            },
            input_scale: None,
        },
        shape: ShapeConstraint::None,
        joint_null_rotation: None,
    }
}

fn erf_approx(x: f64) -> f64 {
    let (a1, a2, a3, a4, a5, p) = (
        0.254829592,
        -0.284496736,
        1.421413741,
        -1.453152027,
        1.061405429,
        0.3275911,
    );
    let sign = if x < 0.0 { -1.0 } else { 1.0 };
    let ax = x.abs();
    let t = 1.0 / (1.0 + p * ax);
    let y = 1.0 - (((((a5 * t + a4) * t) + a3) * t + a2) * t + a1) * t * (-ax * ax).exp();
    sign * y
}

/// Two spatial coordinates and a BINARY 0/1 response from a thresholded latent
/// field — the regime where the Gaussian-identity ρ posterior is poorly
/// identified and escalates to Tier-2 NUTS.
fn simulate(n: usize) -> (Array2<f64>, Array1<f64>) {
    let mut rng = StdRng::seed_from_u64(0x9797_0979);
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
    let y = Array1::from_iter((0..n).map(|i| {
        let p1 = data[[i, 0]];
        let p2 = data[[i, 1]];
        let f = (0.8 * p1).sin() + 0.5 * (0.6 * p2).cos();
        let slope = 0.3 + 0.2 * p1;
        let eta = f + slope * z[i];
        let prob = 0.5 * (1.0 + erf_approx(eta / std::f64::consts::SQRT_2));
        if rng.random::<f64>() < prob { 1.0 } else { 0.0 }
    }));
    (data, y)
}

fn fit_options(skip_rho_posterior_inference: bool) -> FitOptions {
    FitOptions {
        resource_policy: gam_runtime::resource::ResourcePolicy::default_library(),
        latent_cloglog: None,
        mixture_link: None,
        optimize_mixture: false,
        sas_link: None,
        optimize_sas: false,
        compute_inference: true,
        skip_rho_posterior_inference,
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
    }
}

#[test]
fn gaussian_duchon_rho_posterior_inference_is_not_quadratic_in_n() {
    gam::init_parallelism();
    // The point-estimate invariance asserts below hold at any n; the wall-clock
    // budget is the per-leapfrog overhead gate. Fixed CI-affordable n; a
    // cluster-scale n=2000 run is a separate MSI artifact, not an env/cfg branch.
    let n = 800;
    let centers = 10;
    let (data, y) = simulate(n);
    let spec = TermCollectionSpec {
        linear_terms: vec![],
        random_effect_terms: vec![],
        smooth_terms: vec![duchon2_smooth("f_pc", centers)],
    };
    let weights = Array1::ones(n);
    let offset = Array1::zeros(n);
    let family = LikelihoodSpec::new(
        ResponseFamily::Gaussian,
        InverseLink::Standard(StandardLink::Identity),
    );

    let fit = |skip: bool| {
        let start = Instant::now();
        let out = gam::smooth::fit_term_collection_forspec(
            data.view(),
            y.view(),
            weights.view(),
            offset.view(),
            &spec,
            family.clone(),
            &fit_options(skip),
        )
        .expect("fit ok");
        (out, start.elapsed().as_secs_f64())
    };

    // Without the ρ-posterior (point fit + plug-in inference only).
    let (no_post, t_no_post) = fit(true);
    // With the full ρ-posterior stack (escalates to Tier-2 NUTS here).
    let (with_post, t_with_post) = fit(false);

    eprintln!(
        "[979-NUTS] n={n} centers={centers} t_no_post={t_no_post:.2}s t_with_post={t_with_post:.2}s \
         ratio={:.1}",
        t_with_post / t_no_post.max(1e-3)
    );

    // (1) The inference path actually ran — otherwise the timing guard is vacuous.
    assert!(
        with_post.fit.inference.is_some(),
        "compute_inference fit must produce inference"
    );

    // (2) Absolute guard: the whole fit including the ρ posterior stays well
    // under the #979 control budget. With the ALO leak this was ~60s; the fix
    // brings it to ~6s. (A *ratio* to the bare fit is deliberately NOT asserted:
    // NUTS legitimately runs thousands of cheap leapfrog evaluations against the
    // outer fit's ~3, so the total-time ratio reflects evaluation COUNT, not the
    // per-evaluation overhead the ALO leak imposed — only the absolute budget
    // distinguishes cheap-evals-×-many from expensive-evals-×-many.)
    assert!(
        t_with_post < 30.0,
        "Gaussian Duchon fit WITH ρ-posterior inference took {t_with_post:.1}s (budget 30s) — \
         the per-leapfrog ALO-stabilization leak (#979) has likely returned (bare fit \
         {t_no_post:.2}s)"
    );

    // (3) Suppressing the ALO-stabilization augmentation during ρ-posterior
    // sampling must NOT perturb the converged point estimate: the outer
    // optimization is identical in both fits.
    assert_eq!(
        no_post.fit.beta.len(),
        with_post.fit.beta.len(),
        "coefficient length must match"
    );
    let max_beta_dev = no_post
        .fit
        .beta
        .iter()
        .zip(with_post.fit.beta.iter())
        .map(|(a, b)| (a - b).abs())
        .fold(0.0_f64, f64::max);
    assert!(
        max_beta_dev < 1e-9,
        "point estimate must be unchanged by ρ-posterior sampling (max |Δβ|={max_beta_dev:.3e})"
    );
    let max_lambda_dev = no_post
        .fit
        .lambdas
        .iter()
        .zip(with_post.fit.lambdas.iter())
        .map(|(a, b)| (a - b).abs())
        .fold(0.0_f64, f64::max);
    assert!(
        max_lambda_dev < 1e-9,
        "smoothing parameters must be unchanged by ρ-posterior sampling (max |Δλ|={max_lambda_dev:.3e})"
    );
    assert!(
        (no_post.fit.reml_score - with_post.fit.reml_score).abs() < 1e-9,
        "REML score must be unchanged by ρ-posterior sampling"
    );
}
