//! Repro for #1066: the 2-D spatial isotropic-κ optimizer hard-fails
//! (`IntegrationError` / "isotropic analytic optimization did not converge
//! after 80 iterations") on a standard binomial-logit geo smooth, where mgcv
//! fits the same data in seconds.
//!
//! This is the production 2-D binomial analogue of #1053 (iso-1D Matérn). The
//! exact-gradient FD pins pass (#1053/#901), so the math is correct — the defect
//! is the robustness of the iso-κ outer optimizer (seed / ψ-axis step scaling /
//! line-search) on the stiff binomial landscape at higher basis k and larger n.
//!
//! In the fuzz harness both the `matern` and the `ps` (joint-PC → Duchon)
//! geo-disease scenarios fail. Both route through the SAME isotropic-κ radial
//! outer optimizer (`run_exact_joint_spatial_optimization`, isotropic kind), so
//! the regression exercises the Matérn arm (which unambiguously enrolls a single
//! κ axis) at the failing shape, reduced to a CI-tractable size: a smooth
//! binomial spatial field on two clustered geo coordinates (small r_min relative
//! to the diameter — the bad-seed-basin geometry of #1066), k=12 and k=24
//! centers. The fit must converge (no `IntegrationError`), not bail at the
//! 80-iter budget.

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
use rand::rngs::StdRng;
use rand::{RngExt, SeedableRng};

/// Geo-shaped 2-D binomial-logit fixture: a smooth spatial probability field on
/// two clustered coordinates (mimicking PC1/PC2 of a population-structure
/// embedding), Bernoulli outcomes. Deterministic via a fixed seed.
fn simulate_2d_binomial(n: usize) -> (Array2<f64>, Array1<f64>) {
    let mut rng = StdRng::seed_from_u64(0x1066_2026);
    let mut x = Array2::<f64>::zeros((n, 2));
    let mut y = Array1::<f64>::zeros(n);
    for i in 0..n {
        // Clustered geo coordinates: a couple of Gaussian blobs so the minimum
        // pairwise center spacing r_min is small relative to the diameter r_max
        // — exactly the geometry that pushes the ψ window's upper edge
        // ln(100/r_min) far out (the #1066 bad-seed-basin regime).
        let blob = rng.random_range(0.0..1.0);
        let (cx, cy) = if blob < 0.5 { (-1.0, -0.7) } else { (1.0, 0.8) };
        let p1 = cx + rng.random_range(-0.9..0.9);
        let p2 = cy + rng.random_range(-0.9..0.9);
        x[[i, 0]] = p1;
        x[[i, 1]] = p2;
        // Smooth spatial logit field.
        let eta = 0.9 * (0.8 * p1).sin() + 0.6 * (0.7 * p2).cos() - 0.3 * p1 * p2;
        let prob = 1.0 / (1.0 + (-eta).exp());
        y[i] = if rng.random_range(0.0..1.0) < prob {
            1.0
        } else {
            0.0
        };
    }
    (x, y)
}

fn matern_2d_spec(num_centers: usize) -> TermCollectionSpec {
    TermCollectionSpec {
        linear_terms: vec![],
        random_effect_terms: vec![],
        smooth_terms: vec![SmoothTermSpec {
            name: "geo".to_string(),
            basis: SmoothBasisSpec::Matern {
                feature_cols: vec![0, 1],
                spec: MaternBasisSpec {
                    center_strategy: CenterStrategy::FarthestPoint { num_centers },
                    periodic: None,
                    length_scale: 1.0,
                    nu: MaternNu::ThreeHalves,
                    include_intercept: false,
                    double_penalty: false,
                    identifiability: MaternIdentifiability::CenterSumToZero,
                    // Isotropic κ path (the #1066 subject): aniso_log_scales = None.
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
        resource_policy: gam_runtime::resource::ResourcePolicy::default_library(),
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
        persist_warm_start_disk: false,
    }
}

fn fit_geo_matern(n: usize, num_centers: usize) -> Result<Array1<f64>, String> {
    let (x, y) = simulate_2d_binomial(n);
    let weights = Array1::ones(n);
    let offset = Array1::zeros(n);
    let kappa_options = SpatialLengthScaleOptimizationOptions::default();
    let result = gam::fit_model(FitRequest::Standard(StandardFitRequest {
        data: gam::solver::fit_orchestration::StandardFitData::shared(x),
        y: std::sync::Arc::new(y),
        weights: std::sync::Arc::new(weights),
        offset: std::sync::Arc::new(offset),
        spec: matern_2d_spec(num_centers),
        family: LikelihoodSpec::new(
            ResponseFamily::Binomial,
            InverseLink::Standard(StandardLink::Logit),
        ),
        options: fit_options(),
        kappa_options,
        wiggle: None,
        coefficient_groups: Vec::new(),
        penalty_block_gamma_priors: Vec::new(),
        latent_coord: None,
        estimate_tweedie_p: false,
    }))
    .map_err(|e| format!("{e:?}"))?;
    match result {
        FitResult::Standard(s) => {
            if s.fit.beta.iter().all(|v: &f64| v.is_finite()) {
                Ok(s.fit.beta.clone())
            } else {
                Err("non-finite coefficients".to_string())
            }
        }
        _ => Err("expected Standard fit result".to_string()),
    }
}

#[test]
fn iso_kappa_2d_binomial_matern_k12_converges_1066() {
    gam::init_parallelism();
    let beta = fit_geo_matern(2000, 12).unwrap_or_else(|e| {
        panic!("#1066 matern iso-κ 2-D binomial k=12 fit hard-failed (should converge): {e}")
    });
    assert!(beta.iter().all(|v| v.is_finite()), "beta must be finite");
}

#[test]
fn iso_kappa_2d_binomial_matern_k24_converges_1066() {
    gam::init_parallelism();
    // k=24 is the worse-conditioned arm in the issue's evidence (case 2/11).
    let beta = fit_geo_matern(2000, 24).unwrap_or_else(|e| {
        panic!("#1066 matern iso-κ 2-D binomial k=24 fit hard-failed (should converge): {e}")
    });
    assert!(beta.iter().all(|v| v.is_finite()), "beta must be finite");
}
