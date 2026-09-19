//! #938 Tier-0 marginal-smoothing adequacy diagnostic against a REAL fit artifact.
//!
//! The PSIS `ρ`-uncertainty adequacy diagnostic (`src/inference/rho_posterior.rs`) is
//! unit-tested against closed-form Gaussian / heavy-tail fixtures. This test
//! exercises the *objective-lifecycle seam*: a genuine penalized-smooth fit that
//! requests smoothing-parameter posterior inference
//! (`skip_rho_posterior_inference: false`) must produce the diagnostic from its
//! own converged REML objective and surface it on the fit artifact, so the tiers
//! that consume it (1-2) have a real entry point. It asserts the diagnostic is
//! present and structurally sound, and that it is deterministic across identical
//! fits. The diagnostic is not needed to build the returned fit, so a fit that
//! does not request it (the formula and Python default) does not pay for it.

use gam::estimate::FitOptions;
use gam::inference::rho_posterior::RhoPosteriorOutcome;
use gam::init_parallelism;
use gam::smooth::{
    ShapeConstraint, SmoothBasisSpec, SmoothTermSpec, TermCollectionSpec,
    fit_term_collection_forspec,
};
use gam::terms::basis::{BSplineBasisSpec, BSplineKnotSpec};
use gam::types::{InverseLink, LikelihoodSpec, ResponseFamily, StandardLink};
use ndarray::{Array1, Array2};
use rand::SeedableRng;
use rand::rngs::StdRng;
use rand_distr::{Distribution, Normal, Uniform};
use std::f64::consts::PI;

const N: usize = 300;
const SIGMA: f64 = 0.30;

/// Known smooth ground truth: the canonical `s(x)` design that genuinely
/// exercises the smoothing-parameter (ρ) machinery the diagnostic is about.
fn mu_true(x: f64) -> f64 {
    (2.0 * PI * x).sin()
}

fn build_data(seed: u64) -> (Array2<f64>, Array1<f64>) {
    let mut rng = StdRng::seed_from_u64(seed);
    let unif = Uniform::new(0.0, 1.0).expect("uniform");
    let noise = Normal::new(0.0, SIGMA).expect("normal");
    let mut x = Array2::<f64>::zeros((N, 1));
    let mut y = Array1::<f64>::zeros(N);
    for i in 0..N {
        let xi = unif.sample(&mut rng);
        x[[i, 0]] = xi;
        y[i] = mu_true(xi) + noise.sample(&mut rng);
    }
    (x, y)
}

fn smooth_spec() -> TermCollectionSpec {
    TermCollectionSpec {
        linear_terms: vec![],
        random_effect_terms: vec![],
        smooth_terms: vec![SmoothTermSpec {
            frozen_parametric_residualization: None,
            name: "s(x)".to_string(),
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
                    boundary: Default::default(),
                    boundary_conditions: Default::default(),
                },
            },
            shape: ShapeConstraint::None.into(),
            joint_null_rotation: None,
        }],
        level: Default::default(),
    }
}

fn fit_options() -> FitOptions {
    FitOptions {
        resource_policy: gam_runtime::resource::ResourcePolicy::default_library(),
        latent_cloglog: None,
        mixture_link: None,
        optimize_mixture: false,
        sas_link: None,
        optimize_sas: false,
        compute_inference: true,
        skip_rho_posterior_inference: false,
        max_iter: 30,
        tol: 1e-8,
        nullspace_dims: vec![],
        linear_constraints: None,
        firth_bias_reduction: false,
        rho_prior: Default::default(),
        persistent_warm_start_store: None,
    }
}

fn fit_and_take_adequacy(
    seed: u64,
) -> (f64, gam::inference::rho_posterior::RhoPosteriorAdequacy) {
    let (x, y) = build_data(seed);
    let weights = Array1::<f64>::ones(N);
    let offset = Array1::<f64>::zeros(N);
    let fitted = fit_term_collection_forspec(
        x.view(),
        y.view(),
        weights.view(),
        offset.view(),
        &smooth_spec(),
        LikelihoodSpec::new(
            ResponseFamily::Gaussian,
            InverseLink::Standard(StandardLink::Identity),
        ),
        &fit_options(),
    )
    .expect("gam fit");
    let fit = fitted.fit;
    let reml_score = fit
        .reml_score()
        .expect("the fit reports a REML/LAML criterion");
    let adequacy = match &fit.artifacts.rho_posterior {
        RhoPosteriorOutcome::Assessed(adequacy) => adequacy.clone(),
        other => panic!(
            "a smooth-term Gaussian GAM that requests rho-posterior inference has rho \
             parameters and an SPD outer Hessian, so the Tier-0 rho-posterior seam must grade \
             the real fit artifact, got {other:?}"
        ),
    };
    (reml_score, adequacy)
}

/// The seam delivers: a real fit carries an Assessed Tier-0 outcome with a finite
/// tail shape and a Kish effective sample size inside its bounds.
#[test]
fn real_gaussian_fit_carries_a_sound_tier0_adequacy_diagnostic() {
    init_parallelism();
    let (_reml_score, adequacy) = fit_and_take_adequacy(938_001);

    assert!(
        adequacy.k_hat.is_finite(),
        "the Pareto tail shape k̂ must be finite, got {}",
        adequacy.k_hat
    );
    assert!(adequacy.n_samples >= 2, "the diagnostic must draw proposals");

    // Kish's (Σw)²/Σw² over the M self-normalized weights lies in [1, M]: Σw = 1
    // and Cauchy–Schwarz give 1/M ≤ Σw² ≤ 1. Both edges carry the M-term
    // summations' relative rounding M·ε.
    let m = adequacy.n_samples as f64;
    let rounding = 1.0 + m * f64::EPSILON;
    assert!(
        adequacy.effective_sample_size * rounding >= 1.0
            && adequacy.effective_sample_size <= m * rounding,
        "ESS {} must lie in [1, M = {m}]",
        adequacy.effective_sample_size
    );
}

/// The diagnostic is deterministic: the fixed-seed proposal stream means two
/// fits of identical data yield bit-identical `k̂` (the lifecycle seam injects
/// the same live criterion both times).
#[test]
fn tier0_adequacy_is_deterministic_across_identical_fits() {
    init_parallelism();
    let (score_a, a) = fit_and_take_adequacy(938_002);
    let (score_b, b) = fit_and_take_adequacy(938_002);
    assert_eq!(
        score_a.to_bits(),
        score_b.to_bits(),
        "identical fits must reach the same REML score"
    );
    assert_eq!(
        a.k_hat.to_bits(),
        b.k_hat.to_bits(),
        "the fixed-seed diagnostic must give bit-identical k̂ across identical fits"
    );
    assert_eq!(a.n_samples, b.n_samples);
}
