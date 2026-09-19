//! #1404 / #1464 regression guard: the constant-curvature (`curv`) smooth's
//! fitted curvature estimand must IDENTIFY THE SIGN of the true curvature — a
//! negative κ for hyperbolic-shaped data and a positive κ for spherical-shaped
//! data, instead of railing to the positive chart bound for every dataset.
//!
//! This drives the production free-curvature fit, whose continuous
//! Gaussian REML likelihood profile selects κ before the baseline fit. It
//! therefore guards the user-visible estimand without substituting either the
//! deleted plain-RSS profiler or the raw fixed-fit diagnostic.
//!
//! Reference-as-truth: the response is a member of gam's own constant-curvature
//! span at the true κ (see `curved_response`), and every assertion is on gam's
//! fitted κ.

use gam::estimate::FitOptions;
use gam::smooth::{
    ShapeConstraint, SmoothBasisSpec, SmoothTermSpec, SpatialLengthScaleOptimizationOptions,
    TermCollectionSpec, fit_term_collectionwith_spatial_length_scale_optimization,
    get_constant_curvature_kappa,
};
use gam::terms::basis::{
    CenterStrategy, ConstantCurvatureBasisSpec, ConstantCurvatureIdentifiability,
    build_constant_curvature_basis, realized_constant_curvature_length_scale,
};
use gam::types::LikelihoodSpec;
use ndarray::{Array1, Array2};

/// Forwards the κ route's own trace records (`[#1464-trace]`, `[spatial-kappa]`)
/// to stderr, so a run names the route that produced κ̂ whichever way the
/// assertions go.
struct KappaTraceLogger;

impl log::Log for KappaTraceLogger {
    fn enabled(&self, metadata: &log::Metadata<'_>) -> bool {
        metadata.level() <= log::Level::Info
    }
    fn log(&self, record: &log::Record<'_>) {
        let message = record.args().to_string();
        if message.starts_with("[#1464-trace]") || message.starts_with("[spatial-kappa]") {
            eprintln!("{message}");
        }
    }
    fn flush(&self) {}
}

static KAPPA_TRACE_LOGGER: KappaTraceLogger = KappaTraceLogger;

fn install_kappa_trace_logger() {
    // Losing the race to an already-installed logger leaves that logger in place.
    if log::set_logger(&KAPPA_TRACE_LOGGER).is_ok() {
        log::set_max_level(log::LevelFilter::Info);
    }
}

use gam::utils::splitmix64;
fn next_unit(state: &mut u64) -> f64 {
    (splitmix64(state) >> 11) as f64 / (1u64 << 53) as f64
}
fn next_gauss(state: &mut u64) -> f64 {
    let u1 = next_unit(state).max(1.0e-12);
    let u2 = next_unit(state);
    (-2.0 * u1.ln()).sqrt() * (std::f64::consts::TAU * u2).cos()
}

/// Reproducible data on a disk of radius 0.45.
fn disk_points(n: usize, seed: u64) -> Array2<f64> {
    let mut st = seed;
    let mut pts = Array2::<f64>::zeros((n, 2));
    let mut filled = 0usize;
    while filled < n {
        let a = 2.0 * next_unit(&mut st) - 1.0;
        let b = 2.0 * next_unit(&mut st) - 1.0;
        if a * a + b * b > 1.0 {
            continue;
        }
        pts[[filled, 0]] = a * 0.45;
        pts[[filled, 1]] = b * 0.45;
        filled += 1;
    }
    pts
}

/// The one curvature term the plant and the fit share: a modest farthest-point
/// center set keeps each analytic likelihood-profile evaluation cheap, and the
/// range is pinned to the κ = 0 reference length.
fn curvature_spec(ell_ref: f64) -> ConstantCurvatureBasisSpec {
    ConstantCurvatureBasisSpec {
        center_strategy: CenterStrategy::FarthestPoint { num_centers: 12 },
        kappa: 0.0,
        kappa_fixed: false,
        length_scale: ell_ref,
        length_scale_fixed: true,
        double_penalty: false,
        identifiability: ConstantCurvatureIdentifiability::CenterSumToZero,
    }
}

/// A member of the κ⋆ span of the fitted term plus noise: the term's own
/// columns at the TRUE κ, weighted `w_j = 1/(1+j)` and standardized to unit SD,
/// so the truth is in the model being estimated and in no other member of the
/// family.
///
/// This fixture used to plant a single kernel section about the chart ORIGIN,
/// `2·k_κ⋆(d_κ⋆(x, 0)) − 1`. That plant is curvature-BLIND as a function class:
/// `d_κ(x, 0)` is a strictly monotone reparametrization of the chart radius at
/// every κ (the argument `constant_curvature_kappa_inference_e2e` gives for its
/// own generator), so κ is identified only by which radial profiles the 12
/// centers happen to make. Measured on that plant (lane probe 1246907 at
/// 95115c8a1f): the fixed-κ REML score and the deviance both fall monotonically
/// from κ = −4.8 to +4.8 for truths −2, 0 and +2 alike, at 12 and at 48
/// centers, so every fit rails at the positive chart bound. On this in-span
/// plant the production route recovers κ⋆ = −2, −0.75, +0.75 and +2 as −2.026,
/// −0.783, +0.711 and +1.965 (lane probe 1252018).
fn curved_response(
    data: &Array2<f64>,
    spec: &ConstantCurvatureBasisSpec,
    kappa_true: f64,
    seed: u64,
) -> Array1<f64> {
    let mut truth = spec.clone();
    truth.kappa = kappa_true;
    truth.kappa_fixed = true;
    let basis = build_constant_curvature_basis(data.view(), &truth)
        .expect("the planted κ⋆ geometry must be inside its own chart");
    let design = basis.design.to_dense();
    let n = data.nrows();
    let mut y = Array1::<f64>::zeros(n);
    for j in 0..design.ncols() {
        let w = 1.0 / (1.0 + j as f64);
        for i in 0..n {
            y[i] += w * design[(i, j)];
        }
    }
    let mean = y.iter().sum::<f64>() / n as f64;
    let sd = (y.iter().map(|v| (v - mean) * (v - mean)).sum::<f64>() / n as f64).sqrt();
    assert!(
        sd > 0.0,
        "the planted κ⋆ = {kappa_true} signal collapsed to a constant"
    );
    let mut st = seed ^ 0xD1B5_4A32;
    for i in 0..n {
        y[i] = (y[i] - mean) / sd + 0.02 * next_gauss(&mut st);
    }
    y
}

/// Fit the free-curvature production model for a given true curvature.
fn fitted_kappa(data: &Array2<f64>, ell_ref: f64, kappa_true: f64) -> f64 {
    let spec = curvature_spec(ell_ref);
    let y = curved_response(data, &spec, kappa_true, 11);
    let resolved_spec = TermCollectionSpec {
        linear_terms: Vec::new(),
        random_effect_terms: Vec::new(),
        smooth_terms: vec![SmoothTermSpec {
            frozen_parametric_residualization: None,
            name: "curvature".to_string(),
            basis: SmoothBasisSpec::ConstantCurvature {
                feature_cols: vec![0, 1],
                spec,
            },
            shape: ShapeConstraint::None,
            joint_null_rotation: None,
        }],
    };
    let weights = Array1::<f64>::ones(data.nrows());
    let offset = Array1::<f64>::zeros(data.nrows());
    let options = FitOptions::default();
    let fitted = fit_term_collectionwith_spatial_length_scale_optimization(
        data.view(),
        y,
        weights,
        offset,
        &resolved_spec,
        LikelihoodSpec::gaussian_identity(),
        &options,
        &SpatialLengthScaleOptimizationOptions::default(),
    )
    .expect("free-curvature production fit");
    get_constant_curvature_kappa(&fitted.resolvedspec, 0)
        .expect("fitted constant-curvature term must retain kappa")
}

#[test]
fn curv_production_estimand_identifies_curvature_sign_both_ways() {
    install_kappa_trace_logger();
    let data = disk_points(220, 0xC0FF_EE12);
    // κ=0 reference length (auto chart spacing) — the L(κ) target is pinned to it.
    let ell_ref = realized_constant_curvature_length_scale(data.view(), 0.0).unwrap();

    let k_hyp = fitted_kappa(&data, ell_ref, -2.0);
    let k_sph = fitted_kappa(&data, ell_ref, 2.0);
    eprintln!("[#1404] curvature-sign recovery: hyperbolic κ̂={k_hyp:.2}  spherical κ̂={k_sph:.2}");

    assert!(
        k_hyp < 0.0,
        "hyperbolic truth (κ⋆=−2) must recover NEGATIVE curvature; got κ̂={k_hyp} \
         (the #1464 bug rails this to the +chart bound)"
    );
    assert!(
        k_sph > 0.0,
        "spherical truth (κ⋆=+2) must recover POSITIVE curvature; got κ̂={k_sph}"
    );
    // The two signs must be genuinely DISTINGUISHED, not a coincidence of one bound.
    assert!(
        k_hyp < k_sph,
        "curvature estimand must separate hyperbolic from spherical truth: \
         hyperbolic κ̂={k_hyp} should be below spherical κ̂={k_sph}"
    );
}
