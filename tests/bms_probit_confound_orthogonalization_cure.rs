//! Proving test for the STRUCTURAL cure of BMS-probit under-identification.
//!
//! Constructs a deterministic, confounded Bernoulli marginal-slope (BMS) probit
//! cohort: a shared smooth covariate `x` drives BOTH the marginal surface and a
//! log-slope surface, and the exposure `z` is built to correlate strongly with
//! that same smooth covariate (the structural confound). Under the released
//! solver (robust flag OFF) the marginal index `M·β_m` and the score-weighted
//! log-slope `diag(s·z)·G·β_s` overlap in the same column span, leaving the
//! joint penalised Hessian rank-soft and the outer REML poorly conditioned —
//! the marginal coefficients drift large.
//!
//! With `RobustIdentification::Force` the `orthogonalize_confounds` mechanism
//! reparameterizes the log-slope design `G̃ = G − M·B` so its columns are
//! exactly W-orthogonal (in the rigid-pilot IRLS row metric) to the marginal
//! span; the cross-block Gram vanishes, the pinned overlap ridge is retired, and
//! the original-basis coefficients are recovered exactly. We assert:
//!
//!   1. Flag ON yields bounded marginal β (max|β_m| below a sane O(1) bound) AND
//!      outer REML convergence, materially better than OFF.
//!   2. The orthogonalization is exact: MᵀW·G̃ < 1e-10 in the pilot metric.
//!   3. The coefficient round-trip (β_m = β̃_m − B·β_s) is exact (< 1e-12) and
//!      the additive predictor M·β̃_m + G̃·β_s ≡ M·β_m + G·β_s is invariant.
//!
//! Deterministic: fixed-seed `StdRng`, no time/unseeded RNG.

use gam::families::bernoulli_marginal_slope::{BernoulliMarginalSlopeTermSpec, LatentZPolicy};
use gam::families::custom_family::BlockwiseFitOptions;
use gam::families::lognormal_kernel::FrailtySpec;
use gam::resource::ResourcePolicy;
use gam::solver::orthogonal_reparam::OrthogonalReparam;
use gam::solver::robust_identification::RobustConfig;
use gam::terms::basis::{
    BSplineBasisSpec, BSplineBoundaryConditions, BSplineKnotSpec, OneDimensionalBoundary,
};
use gam::terms::smooth::{
    ShapeConstraint, SmoothBasisSpec, SmoothTermSpec, SpatialLengthScaleOptimizationOptions,
    TermCollectionSpec,
};
use gam::types::{InverseLink, StandardLink};
use gam::{
    BernoulliMarginalSlopeFitRequest, FitRequest, FitResult, RobustIdentification, fit_model,
};
use ndarray::{Array1, Array2};
use rand::rngs::StdRng;
use rand::{RngExt, SeedableRng};

const SEED: u64 = 0x_B115_C0FF_EE_15_900D;

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

/// Build the confounded BMS-probit cohort.
///
/// Returns `(data, spec)` where the single covariate column is `x`. Both the
/// marginal and log-slope surfaces are B-splines over `x`; `z` correlates with
/// `x` (the confound). No spatial terms ⇒ the κ-locked fixed-design regime, so
/// the construction-time orthogonalization swap is exact.
fn build_confounded_cohort(n: usize) -> (Array2<f64>, BernoulliMarginalSlopeTermSpec) {
    let mut rng = StdRng::seed_from_u64(SEED);

    // Shared smooth covariate x ∈ [0, 1].
    let x: Vec<f64> = (0..n).map(|i| (i as f64 + 0.5) / n as f64).collect();

    // Exposure z correlates strongly with x (the structural confound), plus a
    // small idiosyncratic Gaussian jitter so z is not a perfect alias.
    let mut z = Array1::<f64>::zeros(n);
    for i in 0..n {
        let u1: f64 = rng.random_range(1e-12..1.0);
        let u2: f64 = rng.random_range(0.0..1.0);
        let jitter = (-2.0 * u1.ln()).sqrt() * (std::f64::consts::TAU * u2).cos();
        // Center x and scale: strong linear dependence on x + light noise.
        z[i] = 2.4 * (x[i] - 0.5) + 0.15 * jitter;
    }
    // Standardize z to unit-ish scale (the family standardizes internally too,
    // but keep the raw confound explicit here).
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

    // True surfaces (probit scale):
    //   marginal index q(x) = a smooth in x
    //   log-slope     g(x) = a smaller smooth in x
    // observed η = q·sqrt(1+(s g)²) + s·g·z  (s = 1 with no frailty)
    let two_pi = std::f64::consts::TAU;
    let y = Array1::from_iter((0..n).map(|i| {
        let xi = x[i];
        let q = -0.10 + 0.9 * (two_pi * xi).sin() + 0.4 * (two_pi * 2.0 * xi).cos();
        let g = 0.25 + 0.5 * (two_pi * xi).cos();
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
        smooth_terms: vec![make_bspline("f_marginal", 10)],
    };
    // Log-slope surface over the SAME covariate x — this is what overlaps the
    // marginal span once weighted by the x-correlated exposure z.
    let logslopespec = TermCollectionSpec {
        linear_terms: vec![],
        random_effect_terms: vec![],
        smooth_terms: vec![make_bspline("f_logslope", 8)],
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

struct FitSummary {
    max_abs_marginal_beta: f64,
    outer_converged: bool,
    outer_gradient_norm: Option<f64>,
    all_finite: bool,
}

fn run_fit(robust: RobustIdentification) -> FitSummary {
    gam::init_parallelism();
    let (data, spec) = build_confounded_cohort(400);
    let mut options = BlockwiseFitOptions::default();
    options.robust_identification = robust;
    // Lock out spatial κ optimization (no spatial terms anyway) so the
    // construction-time orthogonalization swap is exact.
    let kappa_options = SpatialLengthScaleOptimizationOptions::default();
    let policy = ResourcePolicy::default_library();
    let request = FitRequest::BernoulliMarginalSlope(BernoulliMarginalSlopeFitRequest {
        data: data.view(),
        spec,
        options,
        kappa_options,
        policy,
    });
    let result = fit_model(request);
    let out = match result {
        Ok(FitResult::BernoulliMarginalSlope(out)) => out,
        Ok(_) => panic!("wrong FitResult variant"),
        Err(e) => {
            // A hard runaway/refusal under OFF is itself the ill-conditioning
            // signal; surface it as an "unbounded / non-converged" summary so
            // the ON-vs-OFF contrast still holds.
            eprintln!("[confound-cure] robust={robust:?} fit returned Err: {e}");
            return FitSummary {
                max_abs_marginal_beta: f64::INFINITY,
                outer_converged: false,
                outer_gradient_norm: None,
                all_finite: false,
            };
        }
    };
    let marginal_beta = out
        .fit
        .blocks
        .first()
        .map(|b| b.beta.clone())
        .unwrap_or_else(|| Array1::zeros(0));
    let max_abs_marginal_beta = marginal_beta
        .iter()
        .fold(0.0_f64, |acc, v| acc.max(v.abs()));
    let all_finite = out.fit.beta.iter().all(|v| v.is_finite());
    eprintln!(
        "[confound-cure] robust={robust:?} max|β_m|={max_abs_marginal_beta:.4e} \
         outer_converged={} |g|={:?} all_finite={all_finite}",
        out.fit.outer_converged, out.fit.outer_gradient_norm,
    );
    FitSummary {
        max_abs_marginal_beta,
        outer_converged: out.fit.outer_converged,
        outer_gradient_norm: out.fit.outer_gradient_norm,
        all_finite,
    }
}

/// Headline cure: flag ON bounds the marginal β to O(1) and the outer REML
/// converges with a small final gradient norm, materially better than OFF.
#[test]
fn robust_on_bounds_marginal_beta_and_converges_outer_reml() {
    assert!(file!().ends_with(".rs"));

    let off = run_fit(RobustIdentification::Off);
    let on = run_fit(RobustIdentification::Force);

    // The cure must produce bounded, finite marginal coefficients.
    assert!(
        on.all_finite,
        "flag ON produced non-finite coefficients (cure failed)"
    );
    assert!(
        on.max_abs_marginal_beta < 12.0,
        "flag ON did not bound the marginal β to O(1): max|β_m|={:.4e} (expected < 12)",
        on.max_abs_marginal_beta
    );
    // Outer REML must converge under the cure.
    assert!(
        on.outer_converged,
        "flag ON outer REML did not converge (cure failed): |g|={:?}",
        on.outer_gradient_norm
    );
    if let Some(g_on) = on.outer_gradient_norm {
        assert!(
            g_on < 1e-2,
            "flag ON final outer gradient norm not small: |g|={g_on:.4e}"
        );
    }

    // The cure must be MATERIALLY better than OFF: either OFF leaves a larger
    // marginal coefficient, or OFF fails to converge / produces non-finite β.
    let on_strictly_better = (!off.outer_converged && on.outer_converged)
        || (!off.all_finite && on.all_finite)
        || (off.max_abs_marginal_beta > on.max_abs_marginal_beta * 1.5);
    assert!(
        on_strictly_better,
        "flag ON was not materially better than OFF: OFF(max|β_m|={:.4e}, conv={}, finite={}) \
         vs ON(max|β_m|={:.4e}, conv={}, finite={})",
        off.max_abs_marginal_beta,
        off.outer_converged,
        off.all_finite,
        on.max_abs_marginal_beta,
        on.outer_converged,
        on.all_finite,
    );
}

/// Exactness of the structural reparameterization on a design pair that mirrors
/// the cohort's overlap: build a marginal block `M` and a log-slope block `G`
/// that overlaps it, orthogonalize under a positive pilot metric `W`, and
/// assert MᵀW·G̃ ≈ 0 (< 1e-10) and the coefficient round-trip is exact (< 1e-12).
#[test]
fn orthogonalization_is_exact_and_round_trip_is_lossless() {
    assert!(file!().ends_with(".rs"));
    let n = 120;
    let mut rng = StdRng::seed_from_u64(SEED ^ 0xA5A5_A5A5);

    // Marginal block: constant + two smooth-ish columns in x.
    let mut m = Array2::<f64>::zeros((n, 3));
    // Log-slope block: overlaps M (its first column is M's smooth plus noise),
    // second column is a fresh direction.
    let mut g = Array2::<f64>::zeros((n, 2));
    let mut w = Array1::<f64>::zeros(n);
    for i in 0..n {
        let t = (i as f64 + 0.5) / n as f64;
        m[[i, 0]] = 1.0;
        m[[i, 1]] = (std::f64::consts::TAU * t).sin();
        m[[i, 2]] = (std::f64::consts::TAU * 2.0 * t).cos();
        // G col 0 overlaps M col 1 (the confound), col 1 is independent.
        g[[i, 0]] = m[[i, 1]] + 0.05 * rng.random_range(-1.0..1.0);
        g[[i, 1]] = (std::f64::consts::TAU * 3.0 * t).sin();
        // Positive IRLS-style row metric.
        w[i] = 0.25 + 0.75 * rng.random_range(0.0..1.0);
    }

    let reparam = OrthogonalReparam::build(
        RobustConfig::from_policy(RobustIdentification::Force),
        m.view(),
        g.view(),
        &w,
    )
    .expect("orthogonal reparam build should succeed")
    .expect("Force flag ⇒ Some");

    // 1. MᵀW·G̃ ≈ 0.
    let g_tilde = reparam.reparameterized_confound().to_owned();
    let mut max_cross = 0.0_f64;
    let p_m = m.ncols();
    let p_c = g_tilde.ncols();
    for a in 0..p_m {
        for b in 0..p_c {
            let mut acc = 0.0;
            for i in 0..n {
                acc += m[[i, a]] * w[i] * g_tilde[[i, b]];
            }
            max_cross = max_cross.max(acc.abs());
        }
    }
    assert!(
        max_cross < 1e-10,
        "MᵀW·G̃ not orthogonal in the pilot metric: max|entry|={max_cross:e}"
    );

    // 2. Coefficient round-trip is exact and the additive predictor is invariant.
    let beta_m_reparam = Array1::from_vec(vec![0.4, -1.1, 0.7]);
    let beta_s = Array1::from_vec(vec![1.8, -0.6]);
    let eta_reparam = m.dot(&beta_m_reparam) + g_tilde.dot(&beta_s);
    let (beta_m, beta_s_out) = reparam
        .recover_original(&beta_m_reparam, &beta_s)
        .expect("recover_original should succeed");
    let eta_original = m.dot(&beta_m) + g.dot(&beta_s_out);
    let max_dpred = (&eta_reparam - &eta_original)
        .iter()
        .fold(0.0_f64, |acc, v| acc.max(v.abs()));
    assert!(
        max_dpred < 1e-12,
        "additive predictor changed under round-trip: max|Δη|={max_dpred:e}"
    );
    // Log-slope coefficients are untouched by the reparameterization.
    let max_dbetas = (&beta_s_out - &beta_s)
        .iter()
        .fold(0.0_f64, |acc, v| acc.max(v.abs()));
    assert!(
        max_dbetas == 0.0,
        "log-slope coefficients changed under round-trip: {max_dbetas:e}"
    );
}
