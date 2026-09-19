//! #2728 — the two published coefficient covariances must not disagree by
//! orders of magnitude, and the sigma-point nodes that build the corrected one
//! must sit where the posterior actually has mass.
//!
//! # What went wrong
//!
//! `beta_covariance_corrected()` (`Vp`) is assembled by the sigma-point
//! cubature branch as `φ̂·E_ρ[H(ρ)⁻¹] + Cov_ρ[β̂(ρ)]`, a two-point quadrature
//! for `ρ ~ N(ρ̂, V_ρ)` with each node one posterior sd out along a ρ-Hessian
//! eigendirection. The step was taken from the QUADRATIC model of the
//! criterion, `σ_j^{-1/2}`, and never checked against the criterion it was
//! sampling.
//!
//! At a SATURATED smoothing direction that check fails catastrophically. By the
//! exact reparameterisation identity `H_ρ = diag(λ)·H_λ·diag(λ) + diag(g_ρ)`,
//! the ρ-curvature at `λ = 7.2e-9` is ~0 because `λ²` multiplies it, not
//! because the profile is flat. So `σ⁻¹ = 9.5e4`, the step was **308 in
//! log-λ**, and the node landed at a criterion **3309 nats** above the optimum
//! — posterior weight `e^-3309` — while carrying weight ½. On the fixture
//! below that inflated the reported SE by 8.1× over the conditional `Vb` and
//! 11.1× over the estimator's own Monte-Carlo sampling spread.
//!
//! # What is asserted here
//!
//! Three independent angles on the same root cause, so a regression cannot slip
//! through by satisfying one of them:
//!
//! 1. **The node placement itself.** Every node the correction was built from
//!    sits at a criterion rise of order `PROFILE_SIGMA_RISE = 1/2`, which is
//!    the level a one-sigma node is *asserted* to occupy. This is exact, needs
//!    no Monte Carlo, and is the defect stated in its own terms: the number was
//!    3309.
//! 2. **The two published objects agree in magnitude.** The cubature
//!    correction is `E_ρ[φ̂·H(ρ)⁻¹] − φ̂·H(ρ̂)⁻¹ + Cov_ρ[β̂(ρ)]`, and the
//!    first-order `J·V_ρ·Jᵀ` is the leading term of `Cov_ρ[β̂(ρ)]`. The first
//!    difference is `½·A″:V_ρ` to the same order and carries no sign, so the
//!    cubature trace may be negative. A refinement cannot differ from what it
//!    refines by orders of MAGNITUDE, though: the measured ratio of traces was
//!    9993 before the fix.
//! 3. **Calibration against the truth.** With `X` held fixed and only the
//!    Gaussian noise redrawn, the Monte-Carlo spread of `x'β̂` over refits is
//!    exactly what the covariance claims to be, with no misspecification in the
//!    comparison. `Vp` must be within a bounded factor of it.

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

/// The criterion level a one-sigma sigma-point node is asserted to sit at,
/// mirrored from `crates/gam-solve/src/reml/eval.rs`. Under the quadratic model
/// that defines `V_ρ`, `V(ρ̂ + σ^{-1/2}u) − V(ρ̂) = ½·σ·(σ^{-1/2})² = 1/2`
/// exactly.
const PROFILE_SIGMA_RISE: f64 = 0.5;

/// Ceiling on the worst node's criterion rise, as a multiple of
/// `PROFILE_SIGMA_RISE`.
///
/// The production search accepts a node whose rise is within a factor of 1.5 of
/// the target, so a correctly calibrated node cannot exceed `1.5 × 1/2 = 0.75`.
/// The factor 4 here leaves room for the one case the search cannot fix — a
/// criterion so steep that even the smallest bracketed step overshoots — while
/// still failing by three orders of magnitude on the 3309 this issue is about.
const MAX_NODE_RISE_MULTIPLE: f64 = 4.0;

/// Window on `tr(cubature correction) / tr(first-order correction)`.
///
/// Both estimate `Cov_ρ[β̂]` plus, for the cubature, the second-order
/// `E_ρ[H⁻¹] − H(ρ̂)⁻¹` term. They differ by the curvature of `β̂(ρ)` over one
/// posterior sigma and by that second-order term — a factor, not an order of
/// magnitude. Measured on this fixture: 0.63 after the fix, 9993 before it.
const CORRECTION_TRACE_RATIO_LO: f64 = 1.0 / 16.0;
const CORRECTION_TRACE_RATIO_HI: f64 = 16.0;

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
fn corrected_covariance_nodes_are_criterion_calibrated_2728() {
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

    // ── Angle 1: the nodes sit where the posterior has mass ──────────────
    //
    // This is the defect stated in its own terms and it involves no estimate:
    // the number the fit reports here was 3309, against a target of 1/2.
    let method = fit
        .smoothing_correction_method()
        .expect("a fit with smoothing parameters must publish a correction method");
    let SmoothingCorrectionMethod::SigmaPointCubature {
        rank,
        n_points,
        max_node_criterion_rise,
    } = method
    else {
        panic!(
            "this fixture is the #2728 configuration and must exercise the sigma-point \
             cubature branch, so the calibration assertion below is not vacuous; got {method:?}"
        );
    };
    assert_eq!(
        n_points,
        2 * rank,
        "the rule places one (+, −) pair per upgraded direction"
    );
    assert!(
        max_node_criterion_rise.is_finite(),
        "the worst node's criterion rise must be measured, not absent: \
         {max_node_criterion_rise}"
    );
    assert!(
        max_node_criterion_rise <= MAX_NODE_RISE_MULTIPLE * PROFILE_SIGMA_RISE,
        "a sigma-point node sits at criterion rise {max_node_criterion_rise:.6e} against a \
         one-sigma level of {PROFILE_SIGMA_RISE}; its posterior weight is \
         exp(-{max_node_criterion_rise:.3e}) and it carries weight 1/2 in the quadrature \
         (#2728 measured 3.309e3 here)",
    );

    // ── Angle 2: the two published corrections agree in magnitude ────────
    let cubature = fit
        .smoothing_correction()
        .expect("the cubature branch publishes its correction");
    let first_order = fit
        .smoothing_correction_first_order()
        .expect("the first-order correction is retained alongside the cubature one");
    let (tr_cubature, tr_first_order) = (trace(cubature), trace(first_order));
    // `J·V_ρ·Jᵀ` is a Gram, so its trace is positive. The cubature adds
    // `E_ρ[φ̂·H(ρ)⁻¹] − φ̂·H(ρ̂)⁻¹`, which has no sign: on this fixture the node
    // traces average 9.915 against 9.994 at ρ̂, so that term is −0.079 and the
    // cubature trace is −0.0749 (sw4l lane probe 1255647 at 95115c8a1f). Only
    // the MAGNITUDE of the refinement is bounded against what it refines.
    assert!(
        tr_first_order > 0.0 && tr_cubature.is_finite(),
        "the first-order correction must carry positive variance and the cubature one must be \
         finite: cubature={tr_cubature:.6e}, first_order={tr_first_order:.6e}"
    );
    let trace_ratio = tr_cubature.abs() / tr_first_order;
    assert!(
        (CORRECTION_TRACE_RATIO_LO..=CORRECTION_TRACE_RATIO_HI).contains(&trace_ratio),
        "the cubature correction refines the first-order one, so their traces cannot differ \
         by orders of magnitude: tr(cubature)={tr_cubature:.6e}, \
         tr(first_order)={tr_first_order:.6e}, ratio={trace_ratio:.6e} \
         (#2728 measured 9.99e3 here)",
    );

    // ── Angle 3: calibration against the estimator's own sampling spread ──
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
        "[#2728] rank={rank} points={n_points} max_node_rise={max_node_criterion_rise:.4e} \
         tr(cub)/tr(fo)={trace_ratio:.4} RMS(se_cond)={rms_cond:.5} RMS(se_corr)={rms_corr:.5} \
         RMS(mc_sd)={rms_mc:.5} se_corr/mc_sd={se_ratio:.4}"
    );
    assert!(
        (SE_VS_MC_LO..=SE_VS_MC_HI).contains(&se_ratio),
        "the corrected covariance must be calibrated against the estimator's own sampling \
         spread: RMS(se_corr)={rms_corr:.6e}, RMS(mc_sd)={rms_mc:.6e}, ratio={se_ratio:.4} \
         (#2728 measured 11.10 here; the conditional Vb alone measures \
         {:.4})",
        rms_cond / rms_mc,
    );

    // The corrected covariance must remain a covariance. The cubature assembles
    // it as a mean of PSD inverse-Hessian blocks plus a sum of rank-one Grams,
    // so every quadratic form is non-negative by construction; a negative one
    // would mean the telescoping against `φ̂·H(ρ̂)⁻¹` had gone wrong.
    let se_corr = row_se(design.view(), vp);
    assert!(
        se_corr.iter().all(|v| v.is_finite() && *v >= 0.0),
        "every corrected standard error must be finite and non-negative"
    );
}
