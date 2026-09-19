//! #2728 — the two published coefficient covariances must not disagree by
//! orders of magnitude at a saturated smoothing direction.
//!
//! # What went wrong
//!
//! `beta_covariance_corrected()` (`Vp`) used to be assembled on this fixture by
//! a sigma-point cubature, `φ̂·E_ρ[H(ρ)⁻¹] + Cov_ρ[β̂(ρ)]`, with each node one
//! posterior sd out along a ρ-Hessian eigendirection. The step was taken from
//! the QUADRATIC model of the criterion, `σ_j^{-1/2}`, and never checked
//! against the criterion it was sampling.
//!
//! At a SATURATED smoothing direction that fails catastrophically. By the
//! exact reparameterisation identity `H_ρ = diag(λ)·H_λ·diag(λ) + diag(g_ρ)`,
//! the ρ-curvature at `λ = 7.2e-9` is ~0 because `λ²` multiplies it, not
//! because the profile is flat. So `σ⁻¹ = 9.5e4`, the step was **308 in
//! log-λ**, and the node landed at a criterion **3309 nats** above the optimum
//! while carrying weight ½. On the fixture below that inflated the reported SE
//! by 8.1× over the conditional `Vb` and 11.1× over the estimator's own
//! Monte-Carlo sampling spread.
//!
//! The cubature is gone: `Vp = Vb + J·V_ρ·Jᵀ` is the analytic first-order
//! correction for every smoothing dimension, with `V_ρ` the certified inverse
//! on the identified outer-Hessian subspace; no node is placed anywhere.
//!
//! # What is asserted here
//!
//! 1. **The method.** The fit publishes the first-order correction, and the
//!    retained first-order pair is the same matrix.
//! 2. **Calibration against the truth.** With `X` held fixed and only the
//!    Gaussian noise redrawn, the Monte-Carlo spread of `x'β̂` over refits is
//!    exactly what the covariance claims to be, with no misspecification in the
//!    comparison. `Vp` must be within a bounded factor of it.
//! 3. **Ordering.** `J·V_ρ·Jᵀ` is a Gram, so no corrected SE falls below the
//!    conditional one.

use gam::basis::{
    CenterStrategy, DuchonBasisSpec, DuchonNullspaceOrder, DuchonOperatorPenaltySpec,
    duchon_max_active_operator_derivative_order, resolve_duchon_orders,
};
use gam::estimate::{FitOptions, UnifiedFitResult};
use gam::smooth::{
    ShapeConstraint, SmoothBasisSpec, SmoothTermSpec, TermCollectionSpec,
    build_term_collection_design, fit_term_collection_forspec, freeze_term_collection_from_design,
};
use gam::solver::model_types::SmoothingCorrectionMethod;
use gam::types::{InverseLink, LikelihoodSpec, ResponseFamily, StandardLink};
use ndarray::{Array1, Array2, ArrayView2};
use rand::SeedableRng;
use rand::rngs::StdRng;
use rand_distr::{Distribution, Normal};

// ─── Fixture: the #2728 configuration ───────────────────────────────────────
//
// Hybrid anisotropic Duchon on 4 PC coordinates, `K = 80` farthest-point
// centers, Gaussian identity, `n = 4000`. This is the configuration the issue
// measured, reduced only in the number of Monte-Carlo refits.
const PC_DIM: usize = 4;
const K_CENTERS: usize = 80;
const N_TRAIN: usize = 4_000;
const N_EVAL: usize = 400;
const NOISE_SD: f64 = 0.30;
const HYBRID_LENGTH_SCALE: f64 = 1.0;
const SEED_DESIGN: u64 = 0xB10B_0001_0001_0001;
const SEED_EVAL: u64 = 0x0EFA_1000_0000_0001;
const SEED_NOISE: u64 = 0x51E5_0000_0000_0000;

/// Monte-Carlo refits. The asserted quantity is a ratio of two RMS values over
/// 400 evaluation points, so the estimate is far better determined than the
/// per-point sampling SD alone: 16 refits leaves the ratio's own noise well
/// inside the window below, while keeping the test's fits to ~16 x 1.7 s.
const N_REPLICATES: usize = 16;

/// Window on `RMS(se from Vp) / RMS(mc_sd)`.
///
/// A Bayesian `Vp` should be at least as wide as the frequentist sampling
/// spread, because it also carries the smoothing bias: `Vb − Vf = φ·H⁻¹SH⁻¹ ⪰
/// 0`. So the ratio is expected above 1 — `Vb` alone measures 1.37 here and
/// `Vp` 1.57. The lower bound catches a collapse of the correction, the upper
/// bound the inflation this issue is about (11.10 before the fix).
const SE_VS_MC_LO: f64 = 0.5;
const SE_VS_MC_HI: f64 = 4.0;

fn gaussian_identity_likelihood() -> LikelihoodSpec {
    LikelihoodSpec::new(
        ResponseFamily::Gaussian,
        InverseLink::Standard(StandardLink::Identity),
    )
}

/// Smooth ground truth on PC coordinates: a linear trend, a radial bump, and a
/// sinusoid on PC0. Mirrors the `large_scale_reml_stress` simulator the issue
/// measured against.
fn truth(row: &[f64]) -> f64 {
    let coefs = [0.55, -0.40, 0.30, 0.20, -0.15, 0.10];
    let mut linear = 0.0;
    for (j, &xj) in row.iter().enumerate() {
        if j < coefs.len() {
            linear += coefs[j] * xj;
        }
    }
    let mut dist2 = 0.0;
    for (j, &xj) in row.iter().enumerate() {
        let cj = match j {
            0 => 0.30,
            1 => -0.20,
            2 => 0.10,
            _ => 0.0,
        };
        dist2 += (xj - cj) * (xj - cj);
    }
    let radial_bump = (-dist2 / (2.0 * 0.8 * 0.8)).exp();
    let sinusoid = 0.4 * (std::f64::consts::PI * row[0]).sin();
    linear + radial_bump + sinusoid
}

fn simulate_design(n: usize, seed: u64) -> (Array2<f64>, Array1<f64>) {
    let mut rng = StdRng::seed_from_u64(seed);
    let normal = Normal::new(0.0, 1.0).expect("normal params must be valid");
    let mut x = Array2::<f64>::zeros((n, PC_DIM));
    let mut y_true = Array1::<f64>::zeros(n);
    for i in 0..n {
        let mut row = vec![0.0_f64; PC_DIM];
        for j in 0..PC_DIM {
            let v = normal.sample(&mut rng);
            x[[i, j]] = v;
            row[j] = v;
        }
        y_true[i] = truth(&row);
    }
    (x, y_true)
}

fn add_noise(y_true: &Array1<f64>, seed: u64) -> Array1<f64> {
    let mut rng = StdRng::seed_from_u64(seed);
    let noise = Normal::new(0.0, NOISE_SD).expect("noise params must be valid");
    y_true.mapv(|v| v + noise.sample(&mut rng))
}

fn duchon_aniso_pc_spec() -> TermCollectionSpec {
    let operator_penalties = DuchonOperatorPenaltySpec::default();
    let (nullspace_order, power) = resolve_duchon_orders(
        PC_DIM,
        DuchonNullspaceOrder::Linear,
        duchon_max_active_operator_derivative_order(&operator_penalties),
        Some(HYBRID_LENGTH_SCALE),
    );
    TermCollectionSpec {
        linear_terms: vec![],
        random_effect_terms: vec![],
        smooth_terms: vec![SmoothTermSpec {
            frozen_parametric_residualization: None,
            name: "duchon_pc_2728".to_string(),
            basis: SmoothBasisSpec::Duchon {
                feature_cols: (0..PC_DIM).collect(),
                spec: DuchonBasisSpec {
                    radial_reparam: None,
                    center_strategy: CenterStrategy::FarthestPoint {
                        num_centers: K_CENTERS,
                    },
                    length_scale: Some(HYBRID_LENGTH_SCALE),
                    power: power as f64,
                    nullspace_order,
                    identifiability: gam::basis::SpatialIdentifiability::default(),
                    aniso_log_scales: Some(vec![0.0; PC_DIM]),
                    operator_penalties,
                    periodic: None,
                    boundary: gam::basis::OneDimensionalBoundary::Open,
                },
                input_scale: None,
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
        compute_inference: true,
        skip_rho_posterior_inference: false,
        max_iter: 30,
        tol: 1e-5,
        nullspace_dims: vec![],
        linear_constraints: None,
        firth_bias_reduction: false,
        rho_prior: Default::default(),
        persistent_warm_start_store: None,
    }
}

fn row_se(design: ArrayView2<'_, f64>, covariance: &Array2<f64>) -> Array1<f64> {
    let mut se = Array1::<f64>::zeros(design.nrows());
    for (i, row) in design.outer_iter().enumerate() {
        se[i] = row.dot(&covariance.dot(&row)).max(0.0).sqrt();
    }
    se
}

fn rms(values: &Array1<f64>) -> f64 {
    (values.iter().map(|v| v * v).sum::<f64>() / values.len() as f64).sqrt()
}

fn trace(matrix: &Array2<f64>) -> f64 {
    matrix.diag().iter().sum()
}

#[test]
fn corrected_covariance_is_calibrated_at_a_saturated_direction_2728() {
    let spec = duchon_aniso_pc_spec();
    let (x_eval, _) = simulate_design(N_EVAL, SEED_EVAL);
    let (x_train, y_true) = simulate_design(N_TRAIN, SEED_DESIGN);
    let weights = Array1::ones(N_TRAIN);
    let offset = Array1::<f64>::zeros(N_TRAIN);

    let mut eval_design: Option<Array2<f64>> = None;
    let mut eta_draws: Vec<Array1<f64>> = Vec::with_capacity(N_REPLICATES);
    let mut first_fit: Option<UnifiedFitResult> = None;

    for replicate in 0..N_REPLICATES {
        let y = add_noise(&y_true, SEED_NOISE ^ replicate as u64);
        let fitted = fit_term_collection_forspec(
            x_train.view(),
            y.view(),
            weights.view(),
            offset.view(),
            &spec,
            gaussian_identity_likelihood(),
            &fit_options(),
        )
        .expect("Duchon-on-PC fit must succeed");
        if eval_design.is_none() {
            let frozen = freeze_term_collection_from_design(&spec, &fitted.design)
                .expect("freezing the trained spec must succeed");
            eval_design = Some(
                build_term_collection_design(x_eval.view(), &frozen)
                    .expect("held-out design build must succeed")
                    .design
                    .to_dense(),
            );
        }
        let design = eval_design.as_ref().expect("held-out design");
        eta_draws.push(design.dot(&fitted.fit.beta));
        if first_fit.is_none() {
            first_fit = Some(fitted.fit);
        }
    }

    let design = eval_design.expect("held-out design");
    let fit = first_fit.expect("at least one fit");

    // ── Angle 1: the analytic first-order correction is what ships ───────
    let method = fit
        .smoothing_correction_method()
        .expect("a fit with smoothing parameters must publish a correction method");
    let SmoothingCorrectionMethod::FirstOrderIdentifiedSubspace {
        active_rank,
        rho_dimension,
    } = method
    else {
        panic!("the corrected covariance must be the analytic first-order one; got {method:?}");
    };
    assert!(active_rank <= rho_dimension);
    let correction = fit
        .smoothing_correction()
        .expect("the first-order correction is published");
    let first_order = fit
        .smoothing_correction_first_order()
        .expect("the first-order correction is retained");
    assert_eq!(correction, first_order);
    assert!(
        trace(correction) >= 0.0,
        "J V_rho J^T is a Gram and carries non-negative variance: {:.6e}",
        trace(correction)
    );

    // ── Angle 2: calibration against the estimator's own sampling spread ──
    //
    // `X` was held fixed across replicates and only the noise redrawn, so the
    // basis, the centers, and the held-out design are identical in every refit
    // and the Monte-Carlo spread of `x'β̂` is exactly the quantity the
    // covariance claims to be. No misspecification enters the comparison.
    let vb = fit
        .beta_covariance()
        .expect("conditional covariance must be published");
    let vp = fit
        .beta_covariance_corrected()
        .expect("corrected covariance must be published");
    let rms_cond = rms(&row_se(design.view(), vb));
    let rms_corr = rms(&row_se(design.view(), vp));

    let replicates = eta_draws.len() as f64;
    let mut mc_sd = Array1::<f64>::zeros(design.nrows());
    for i in 0..design.nrows() {
        let mean: f64 = eta_draws.iter().map(|draw| draw[i]).sum::<f64>() / replicates;
        let var: f64 = eta_draws
            .iter()
            .map(|draw| (draw[i] - mean) * (draw[i] - mean))
            .sum::<f64>()
            / (replicates - 1.0);
        mc_sd[i] = var.sqrt();
    }
    let rms_mc = rms(&mc_sd);
    assert!(
        rms_mc > 0.0,
        "the Monte-Carlo spread must be non-degenerate: {rms_mc:.6e}"
    );
    let se_ratio = rms_corr / rms_mc;
    println!(
        "[#2728] active_rank={active_rank} rho_dimension={rho_dimension} \
         RMS(se_cond)={rms_cond:.5} RMS(se_corr)={rms_corr:.5} RMS(mc_sd)={rms_mc:.5} \
         se_corr/mc_sd={se_ratio:.4}"
    );
    assert!(
        (SE_VS_MC_LO..=SE_VS_MC_HI).contains(&se_ratio),
        "the corrected covariance must be calibrated against the estimator's own sampling \
         spread: RMS(se_corr)={rms_corr:.6e}, RMS(mc_sd)={rms_mc:.6e}, ratio={se_ratio:.4} \
         (#2728 measured 11.10 here; the conditional Vb alone measures \
         {:.4})",
        rms_cond / rms_mc,
    );

    // ── Angle 3: the correction only adds variance ───────────────────────
    let se_cond = row_se(design.view(), vb);
    let se_corr = row_se(design.view(), vp);
    for (i, (&corr, &cond)) in se_corr.iter().zip(se_cond.iter()).enumerate() {
        assert!(
            corr.is_finite() && corr >= cond * (1.0 - 1e-12),
            "row {i}: corrected SE {corr:.6e} must be finite and at least the conditional \
             SE {cond:.6e}"
        );
    }
}
