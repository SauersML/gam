//! `scale_dimensions=True` must genuinely engage per-axis anisotropy for a
//! thin-plate (`bs="tp"`) smooth, not be a silent no-op (gam#1676).
//!
//! A canonical thin-plate regression spline carries a *single* curvature
//! penalty (the exact `∫|Dᵐf|²` RKHS Gram) with no per-axis structure, so the
//! `scale_dimensions` flag had nothing to engage and was silently dropped for
//! `bs="tp"` while it worked for `duchon()`/`matern()`. The fix rewrites a
//! multi-axis thin-plate term to its mathematically-equivalent anisotropic s=0
//! Duchon spline (the thin-plate kernel `r^{2m−d}` IS the s=0 Duchon kernel), so
//! the per-axis tension-ARD machinery (`Σ‖∇f‖²` → `d` directional
//! `Σ(∂f/∂x_a)²`, each its own REML `λ_a`) engages exactly as it does for
//! Duchon.
//!
//! These tests pin the contract at the `enable_scale_dimensions` rewrite layer
//! (the Rust core behind gamfit's `scale_dimensions` kwarg):
//!   * a multi-axis `tp` term becomes an s=0 Duchon term carrying
//!     `aniso_log_scales`, and the realized basis then emits one
//!     `OperatorRelevance{axis}` penalty per input axis (anisotropy ENGAGED);
//!   * a 1-D `tp` term is untouched (anisotropy is meaningless on one axis);
//!   * WITHOUT `scale_dimensions` a `tp` term stays canonical thin-plate.

use gam::basis::{
    CenterStrategy, DuchonNullspaceOrder, PenaltySource, SpatialIdentifiability,
    ThinPlateBasisSpec, build_duchon_basis,
};
use gam::estimate::FitOptions;
use gam::smooth::{
    ShapeConstraint, SmoothBasisSpec, SmoothTermSpec, TermCollectionSpec,
    fit_term_collection_forspec,
};
use gam::types::{InverseLink, LikelihoodSpec, ResponseFamily, StandardLink};
use ndarray::{Array1, Array2};
use rand::SeedableRng;
use rand::rngs::StdRng;
use rand_distr::{Distribution, Normal, Uniform};

fn gaussian_identity_likelihood() -> LikelihoodSpec {
    LikelihoodSpec::new(
        ResponseFamily::Gaussian,
        InverseLink::Standard(StandardLink::Identity),
    )
}

/// `FitOptions` with everything off, matching the minimal config the sibling
/// Duchon/Matérn `scale_dimensions` integration tests use.
fn minimal_fit_options() -> FitOptions {
    FitOptions {
        resource_policy: gam_runtime::resource::ResourcePolicy::default_library(),
        latent_cloglog: None,
        mixture_link: None,
        optimize_mixture: false,
        sas_link: None,
        optimize_sas: false,
        compute_inference: false,
        skip_rho_posterior_inference: false,
        max_iter: 24,
        tol: 1e-4,
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

fn synthetic_data(n: usize, d: usize, seed: u64) -> Array2<f64> {
    let mut rng = StdRng::seed_from_u64(seed);
    let dist = Uniform::new(-1.0_f64, 1.0).expect("uniform params valid");
    let mut data = Array2::<f64>::zeros((n, d));
    for i in 0..n {
        for j in 0..d {
            data[[i, j]] = dist.sample(&mut rng);
        }
    }
    data
}

fn thin_plate_term(d: usize, num_centers: usize) -> TermCollectionSpec {
    TermCollectionSpec {
        linear_terms: vec![],
        random_effect_terms: vec![],
        smooth_terms: vec![SmoothTermSpec {
            name: "tp".to_string(),
            basis: SmoothBasisSpec::ThinPlate {
                feature_cols: (0..d).collect(),
                spec: ThinPlateBasisSpec {
                    center_strategy: CenterStrategy::FarthestPoint { num_centers },
                    periodic: None,
                    length_scale: 0.0,
                    double_penalty: true,
                    identifiability: SpatialIdentifiability::None,
                    radial_reparam: None,
                },
                input_scales: None,
            },
            shape: ShapeConstraint::None,
            joint_null_rotation: None,
        }],
    }
}

#[test]
fn thin_plate_scale_dimensions_promotes_to_anisotropic_s0_duchon() {
    let mut spec = thin_plate_term(2, 25);

    // The Rust core behind gamfit's `scale_dimensions=True`.
    gam::term_builder::enable_scale_dimensions(&mut spec);

    // The thin-plate term is rewritten to the mathematically-equivalent s=0
    // Duchon spline (the thin-plate kernel `r^{2m−d}` is the s=0 Duchon kernel),
    // which CAN carry per-axis relevance, with the geometry seed sentinel set.
    let SmoothBasisSpec::Duchon { spec: duchon, .. } = &spec.smooth_terms[0].basis else {
        panic!(
            "scale_dimensions must rewrite a multi-axis thin-plate term to its \
             anisotropic Duchon twin, got {:?}",
            spec.smooth_terms[0].basis
        );
    };
    assert_eq!(duchon.power, 0.0, "s=0 keeps the exact thin-plate kernel");
    assert_eq!(
        duchon.length_scale, None,
        "pure scale-free Duchon (no kernel length scale): thin-plate is scale-free"
    );
    assert_eq!(
        duchon.nullspace_order,
        DuchonNullspaceOrder::Linear,
        "2-D thin-plate penalty order m=2 maps to the Linear Duchon null space"
    );
    assert_eq!(
        duchon.aniso_log_scales.as_deref(),
        Some(&[0.0, 0.0][..]),
        "the per-axis geometry seed sentinel must be set so anisotropy engages"
    );

    // And the realized basis genuinely engages anisotropy: the single isotropic
    // gradient penalty is REPLACED by one per-axis relevance penalty per axis.
    let data = synthetic_data(400, 2, 7);
    let srcs: Vec<PenaltySource> = build_duchon_basis(data.view(), duchon)
        .expect("anisotropic s=0 Duchon basis builds")
        .active_penalties
        .iter()
        .map(|penalty| penalty.info.source.clone())
        .collect();
    let relevance_axes: Vec<usize> = srcs
        .iter()
        .filter_map(|s| match s {
            PenaltySource::OperatorRelevance { axis } => Some(*axis),
            _ => None,
        })
        .collect();
    assert_eq!(
        relevance_axes,
        vec![0, 1],
        "thin-plate scale_dimensions must emit one per-axis relevance penalty \
         per input axis (anisotropy engaged), got {srcs:?}"
    );
    assert!(
        !srcs
            .iter()
            .any(|s| matches!(s, PenaltySource::OperatorTension)),
        "the isotropic gradient penalty must be replaced, not duplicated, got {srcs:?}"
    );
    // The exact thin-plate curvature RKHS Gram stays single / isotropic.
    assert!(
        srcs.iter().any(|s| matches!(s, PenaltySource::Primary)),
        "the thin-plate curvature penalty (Primary) must survive, got {srcs:?}"
    );
}

#[test]
fn thin_plate_scale_dimensions_is_a_noop_for_one_dimensional_term() {
    let mut spec = thin_plate_term(1, 15);
    gam::term_builder::enable_scale_dimensions(&mut spec);
    // Anisotropy is meaningless on a single axis (its Σ η = 0 contrast vector is
    // empty), so a 1-D thin-plate term is left exactly as it was.
    assert!(
        matches!(
            &spec.smooth_terms[0].basis,
            SmoothBasisSpec::ThinPlate { .. }
        ),
        "a 1-D thin-plate term must not be rewritten by scale_dimensions, got {:?}",
        spec.smooth_terms[0].basis
    );
}

#[test]
fn thin_plate_without_scale_dimensions_stays_canonical_thin_plate() {
    // The default path (scale_dimensions off) must be untouched: a multi-axis
    // thin-plate term that never sees `enable_scale_dimensions` stays a
    // canonical TPS term.
    let spec = thin_plate_term(2, 25);
    assert!(
        matches!(
            &spec.smooth_terms[0].basis,
            SmoothBasisSpec::ThinPlate { .. }
        ),
        "without scale_dimensions the term must stay canonical thin-plate"
    );
}

/// End-to-end fit regression for the issue's actual symptom. The issue repro
/// is a `gamfit.fit(..., scale_dimensions=True)` on a strongly anisotropic
/// surface (fast in `x1`, near-linear in `x2`) that came back *bit-for-bit
/// identical* to the default fit (`max|pred_on − pred_off| = 0.000e+00`) for
/// `thinplate(...)` while `duchon()`/`matern()` changed materially.
///
/// This test reproduces that at the Rust core behind the kwarg
/// (`enable_scale_dimensions` → `fit_term_collection_forspec`): the same 2-D
/// thin-plate term is fit once on the default path and once after
/// `enable_scale_dimensions`, and the two fits must now *differ* (the flag
/// genuinely engages) instead of being identical (the silent no-op).
#[test]
fn thin_plate_scale_dimensions_changes_the_fit_on_anisotropic_data() {
    let n = 600usize;
    // Strongly anisotropic surface mirroring the issue: a fast oscillation in
    // x1, a near-linear trend in x2. The default isotropic TPS must trade off a
    // single curvature scale across both axes; the anisotropic rewrite can
    // tension x2 down independently.
    let mut rng = StdRng::seed_from_u64(1);
    let unit = Uniform::new(0.0_f64, 1.0).expect("uniform params valid");
    let noise = Normal::new(0.0, 0.12).expect("normal params valid");
    let mut x = Array2::<f64>::zeros((n, 2));
    let mut y = Array1::<f64>::zeros(n);
    for i in 0..n {
        let x1 = unit.sample(&mut rng);
        let x2 = unit.sample(&mut rng);
        x[[i, 0]] = x1;
        x[[i, 1]] = x2;
        let truth = (2.0 * std::f64::consts::PI * 3.0 * x1).sin() + 0.4 * x2;
        y[i] = truth + noise.sample(&mut rng);
    }

    let weights = Array1::ones(n);
    let offset = Array1::zeros(n);
    let opts = minimal_fit_options();

    // A standalone fit has only an implicit intercept, so the spatial smooth
    // must be orthogonalized against it (the formula pipeline's default) or its
    // polynomial null-space constant aliases the intercept and the unpenalized
    // design is rank-deficient. The spec-rewrite tests above use the bare
    // `None` helper because they never fit.
    let thin_plate_fit_term = |num_centers: usize| {
        let mut spec = thin_plate_term(2, num_centers);
        if let SmoothBasisSpec::ThinPlate { spec: tp, .. } = &mut spec.smooth_terms[0].basis {
            tp.identifiability = SpatialIdentifiability::OrthogonalToParametric;
        }
        spec
    };

    // (1) Default path — canonical thin-plate, scale_dimensions off.
    let spec_off = thin_plate_fit_term(40);
    let fit_off = fit_term_collection_forspec(
        x.view(),
        y.view(),
        weights.view(),
        offset.view(),
        &spec_off,
        gaussian_identity_likelihood(),
        &opts,
    )
    .expect("default thin-plate fit completes");
    let pred_off = fit_off.design.design.to_dense().dot(&fit_off.fit.beta) + &offset;

    // (2) scale_dimensions=True — the same term after the rewrite.
    let mut spec_on = thin_plate_fit_term(40);
    gam::term_builder::enable_scale_dimensions(&mut spec_on);
    let fit_on = fit_term_collection_forspec(
        x.view(),
        y.view(),
        weights.view(),
        offset.view(),
        &spec_on,
        gaussian_identity_likelihood(),
        &opts,
    )
    .expect("scale_dimensions thin-plate fit completes");
    let pred_on = fit_on.design.design.to_dense().dot(&fit_on.fit.beta) + &offset;

    assert!(
        pred_on.iter().all(|v| v.is_finite()) && fit_on.fit.beta.iter().all(|v| v.is_finite()),
        "anisotropic thin-plate fit must be finite"
    );

    // The issue's core symptom: the two fits were bit-for-bit identical
    // (max|diff| = 0). `scale_dimensions` must genuinely change the fit now.
    let max_abs_diff = pred_on
        .iter()
        .zip(pred_off.iter())
        .map(|(a, b)| (a - b).abs())
        .fold(0.0_f64, f64::max);
    assert!(
        max_abs_diff > 1e-6,
        "scale_dimensions=True must change the thin-plate fit on anisotropic data \
         (issue #1676: was a silent no-op, max|diff| = 0); got max|diff| = {max_abs_diff:.3e}"
    );

    // And the anisotropic fit must still explain the surface (no degenerate
    // retreat to a near-constant fit): beat mean-only by a wide margin.
    let y_mean = y.mean().unwrap_or(0.0);
    let sse_on: f64 = pred_on
        .iter()
        .zip(y.iter())
        .map(|(p, t)| (p - t).powi(2))
        .sum();
    let sse_baseline: f64 = y.iter().map(|&v| (v - y_mean).powi(2)).sum();
    assert!(
        sse_on < 0.5 * sse_baseline,
        "anisotropic thin-plate fit must explain the anisotropic surface \
         (beat mean-only by ≥50%): sse_on={sse_on:.6e}, sse_baseline={sse_baseline:.6e}"
    );
}
