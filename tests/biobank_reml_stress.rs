//! End-to-end stress test for the closed-form Duchon pipeline at
//! biobank-relevant scale.
//!
//! Runs in the default suite. The fit can take many minutes and use a
//! lot of memory — run under `--release` if iteration time matters:
//!
//! ```text
//! cargo test --release biobank_reml_stress
//! ```
//!
//! It exercises the full Duchon-on-PC GAM pipeline end-to-end:
//!
//!   * Pure-Rust deterministic biobank-style simulator producing
//!     `n` rows of `pc_dim` PC features sampled from N(0, I) and a
//!     continuous response `y = f_true(X) + ε`.
//!   * Hybrid anisotropic Duchon smooth (`length_scale = Some(...)`,
//!     `aniso_log_scales = Some(zeros)`) with `K` farthest-point centers.
//!   * REML/LAML outer loop must converge.
//!   * Held-out-grid relative L2 reconstruction error must be < 0.10.
//!   * Bias-corrected predictions must be available on `FitInference`
//!     and finite.
//!   * 95% prediction-interval coverage on held-out samples must
//!     exceed 0.85 across `N_COVERAGE_SIMS` independent simulations.
//!   * Wallclock budget per fit is documented and asserted.
//!
//! All randomness is seeded; failures are reproducible.

use gam::basis::{
    CenterStrategy, DuchonBasisSpec, DuchonNullspaceOrder, DuchonOperatorPenaltySpec,
};
use gam::estimate::FitOptions;
use gam::predict::{
    InferenceCovarianceMode, MeanIntervalMethod, PredictUncertaintyOptions, predict_gam,
    predict_gamwith_uncertainty,
};
use gam::smooth::{
    ShapeConstraint, SmoothBasisSpec, SmoothTermSpec, TermCollectionSpec,
    build_term_collection_design, fit_term_collection_forspec, freeze_term_collection_from_design,
};
use gam::types::LikelihoodFamily;
use ndarray::{Array1, Array2};
use rand::SeedableRng;
use rand::rngs::StdRng;
use rand_distr::{Distribution, Normal};
use std::time::Instant;

// ─── Test scale knobs ───────────────────────────────────────────────────
//
// `N_TRAIN`, `K_CENTERS`, and `PC_DIM` are deliberately moderate so the
// test is feasible at all in a default `--release` run on a developer
// box. The team-lead spec calls out `K∈{500,1000}` (start small) and
// `n` in the 50K-300K range; `n=50_000` and `K=500` are the lower end
// of that range. Crank up by editing these constants.
const N_TRAIN: usize = 50_000;
const N_HOLDOUT: usize = 4_000;
const PC_DIM: usize = 6;
const K_CENTERS: usize = 500;
const NOISE_SD: f64 = 0.30;
const SEED_BASE: u64 = 0xB10B_0001_0001_0001;
const N_COVERAGE_SIMS: usize = 20;
const N_COVERAGE_TRAIN: usize = 4_000;
const N_COVERAGE_HOLDOUT: usize = 400;
const K_COVERAGE: usize = 80;
const PC_DIM_COVERAGE: usize = 4;

// Wallclock ceilings (seconds). Generous on a developer box; documents
// intent so that a regression makes itself loud.
const WALLCLOCK_BUDGET_MAIN_SECS: f64 = 30.0 * 60.0; // 30 minutes
const WALLCLOCK_BUDGET_PER_COVERAGE_FIT_SECS: f64 = 120.0;

// ─── Synthetic biobank simulator ────────────────────────────────────────

/// Smooth ground-truth function on PC coordinates. Used both for
/// generating `y` and for evaluating reconstruction error.
///
/// The functional form mirrors the pipeline contract in
/// `production_pipeline_spec.md` and `biobank_sim.py`: a sum of a
/// linear PC trend, a radial bump centered near the origin, and a
/// sinusoid on PC0. It is smooth, bounded, and not separable into
/// per-axis pieces — all properties an anisotropic Duchon smooth
/// should be able to track.
fn truth(row: &[f64]) -> f64 {
    let mut linear = 0.0;
    let coefs = [0.55, -0.40, 0.30, 0.20, -0.15, 0.10];
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
        let d = xj - cj;
        dist2 += d * d;
    }
    let radial_bump = 1.0 * (-dist2 / (2.0 * 0.8 * 0.8)).exp();
    let sinusoid = 0.4 * (std::f64::consts::PI * row[0]).sin();
    linear + radial_bump + sinusoid
}

/// Generate `(X, y, y_true)` with PC coordinates sampled iid from
/// the standard normal and `y = truth(X) + N(0, NOISE_SD²)`.
fn simulate(n: usize, pc_dim: usize, seed: u64) -> (Array2<f64>, Array1<f64>, Array1<f64>) {
    let mut rng = StdRng::seed_from_u64(seed);
    let normal = Normal::new(0.0, 1.0).expect("normal params must be valid");
    let noise = Normal::new(0.0, NOISE_SD).expect("noise params must be valid");

    let mut x = Array2::<f64>::zeros((n, pc_dim));
    let mut y = Array1::<f64>::zeros(n);
    let mut y_true = Array1::<f64>::zeros(n);
    for i in 0..n {
        let mut row = vec![0.0_f64; pc_dim];
        for j in 0..pc_dim {
            let v = normal.sample(&mut rng);
            x[[i, j]] = v;
            row[j] = v;
        }
        let f = truth(&row);
        y_true[i] = f;
        y[i] = f + noise.sample(&mut rng);
    }
    (x, y, y_true)
}

/// Build the anisotropic-hybrid Duchon term spec used throughout the
/// test.
fn duchon_aniso_pc_spec(name: &str, pc_dim: usize, k_centers: usize) -> TermCollectionSpec {
    TermCollectionSpec {
        linear_terms: vec![],
        random_effect_terms: vec![],
        smooth_terms: vec![SmoothTermSpec {
            name: name.to_string(),
            basis: SmoothBasisSpec::Duchon {
                feature_cols: (0..pc_dim).collect(),
                spec: DuchonBasisSpec {
                    center_strategy: CenterStrategy::FarthestPoint {
                        num_centers: k_centers,
                    },
                    // Hybrid Duchon — required for aniso_log_scales.
                    length_scale: Some(1.0),
                    power: 1.0,
                    nullspace_order: DuchonNullspaceOrder::Linear,
                    identifiability: gam::basis::SpatialIdentifiability::default(),
                    aniso_log_scales: Some(vec![0.0; pc_dim]),
                    operator_penalties: DuchonOperatorPenaltySpec::default(),

                    periodic: None,
                    boundary: gam::basis::OneDimensionalBoundary::Open,
                },
                input_scales: None,
            },
            shape: ShapeConstraint::None,
        }],
    }
}

fn fit_options(max_iter: usize) -> FitOptions {
    FitOptions {
        latent_cloglog: None,
        mixture_link: None,
        optimize_mixture: false,
        sas_link: None,
        optimize_sas: false,
        compute_inference: true,
        max_iter,
        tol: 1e-5,
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

/// L2 relative error: ||pred - truth||₂ / ||truth - mean(truth)||₂.
fn relative_l2(pred: &Array1<f64>, truth: &Array1<f64>) -> f64 {
    let mean_t = truth.mean().unwrap_or(0.0);
    let mut num = 0.0;
    let mut den = 0.0;
    for (p, t) in pred.iter().zip(truth.iter()) {
        let dp = p - t;
        let dt = t - mean_t;
        num += dp * dp;
        den += dt * dt;
    }
    (num / den.max(1e-30)).sqrt()
}

// ─── Main stress test ───────────────────────────────────────────────────

#[test]
fn biobank_reml_stress_main() {
    let (x_train, y_train, _y_true_train) = simulate(N_TRAIN, PC_DIM, SEED_BASE);
    let (x_holdout, _y_holdout, y_true_holdout) =
        simulate(N_HOLDOUT, PC_DIM, SEED_BASE.wrapping_add(0xDEAD));

    let spec = duchon_aniso_pc_spec("duchon_pc_main", PC_DIM, K_CENTERS);
    let weights = Array1::ones(N_TRAIN);
    let offset = Array1::<f64>::zeros(N_TRAIN);

    let start = Instant::now();
    let fitted = fit_term_collection_forspec(
        x_train.view(),
        y_train.view(),
        weights.view(),
        offset.view(),
        &spec,
        LikelihoodFamily::GaussianIdentity,
        &fit_options(40),
    )
    .expect("biobank-scale Duchon-on-PC fit should succeed");
    let elapsed = start.elapsed();

    // (1) REML outer loop converged.
    assert!(
        fitted.fit.outer_converged,
        "REML outer optimization did not converge \
         (outer_iterations={}, outer_gradient_norm={:?})",
        fitted.fit.outer_iterations, fitted.fit.outer_gradient_norm,
    );
    assert!(
        fitted.fit.beta.iter().all(|v| v.is_finite()),
        "fitted coefficients must all be finite",
    );

    // (2) Held-out-grid reconstruction error: build the held-out design
    //     using the *fitted* term collection design (so centers, scaling,
    //     etc. match), then compute relative L2 against truth.
    let frozenspec = freeze_term_collection_from_design(&spec, &fitted.design)
        .expect("freezing trained spec must succeed");
    let holdout_design = build_term_collection_design(x_holdout.view(), &frozenspec)
        .expect("holdout design build must succeed");
    let holdout_dense = holdout_design.design.to_dense();
    let holdout_offset = Array1::<f64>::zeros(N_HOLDOUT);

    let pred = predict_gam(
        holdout_dense.view(),
        fitted.fit.beta.view(),
        holdout_offset.view(),
        LikelihoodFamily::GaussianIdentity,
    )
    .expect("predict on held-out grid should succeed");
    assert!(pred.mean.iter().all(|v| v.is_finite()));
    let rel_l2 = relative_l2(&pred.mean, &y_true_holdout);
    assert!(
        rel_l2 < 0.10,
        "held-out relative L2 reconstruction error too high: {rel_l2:.4} (>= 0.10)",
    );

    // (3) Bias-corrected predictions: FitInference must carry a finite
    //     bias-correction vector after a successful REML fit, and the
    //     uncertainty path must accept `apply_bias_correction = true`
    //     and produce finite η.
    let inference = fitted
        .fit
        .inference
        .as_ref()
        .expect("compute_inference=true must populate FitInference");
    let bc = inference
        .bias_correction_beta
        .as_ref()
        .expect("FitInference must carry bias_correction_beta");
    assert_eq!(bc.len(), fitted.fit.beta.len());
    assert!(
        bc.iter().all(|v| v.is_finite()),
        "bias_correction_beta must be entirely finite",
    );

    let pred_unc = predict_gamwith_uncertainty(
        holdout_dense.view(),
        fitted.fit.beta.view(),
        holdout_offset.view(),
        LikelihoodFamily::GaussianIdentity,
        &fitted.fit,
        &PredictUncertaintyOptions {
            confidence_level: 0.95,
            covariance_mode: InferenceCovarianceMode::ConditionalPlusSmoothingPreferred,
            mean_interval_method: MeanIntervalMethod::TransformEta,
            includeobservation_interval: true,
            apply_bias_correction: true,
            ..PredictUncertaintyOptions::default()
        },
    )
    .expect("bias-corrected uncertainty prediction must succeed");
    assert!(pred_unc.eta.iter().all(|v| v.is_finite()));
    assert!(pred_unc.mean.iter().all(|v| v.is_finite()));

    // (4) Wallclock budget.
    assert!(
        elapsed.as_secs_f64() < WALLCLOCK_BUDGET_MAIN_SECS,
        "main biobank stress fit exceeded wallclock budget: \
         {:.1}s >= {:.1}s",
        elapsed.as_secs_f64(),
        WALLCLOCK_BUDGET_MAIN_SECS,
    );

    eprintln!(
        "[biobank_reml_stress_main] n={N_TRAIN}, K={K_CENTERS}, pc_dim={PC_DIM} \
         | wall_clock={:.2}s, outer_iter={}, rel_l2_holdout={:.4}",
        elapsed.as_secs_f64(),
        fitted.fit.outer_iterations,
        rel_l2,
    );
}

// ─── Coverage simulation ────────────────────────────────────────────────

/// Repeatedly fit the same anisotropic Duchon model on freshly drawn
/// data, then check the empirical 95% coverage of the per-row mean
/// interval on held-out points. A correctly calibrated posterior
/// should produce at least 0.85 average coverage across simulations
/// (the slack accounts for finite-sample noise and the
/// well-known REML-conservativeness/anti-conservativeness drift at
/// this dimensionality).
#[test]
fn biobank_reml_stress_coverage() {
    let mut total_in = 0usize;
    let mut total_pts = 0usize;

    for sim_idx in 0..N_COVERAGE_SIMS {
        let train_seed = SEED_BASE.wrapping_add(0xC0DE_0000 + sim_idx as u64);
        let test_seed = SEED_BASE.wrapping_add(0xFADE_0000 + sim_idx as u64);

        let (x_tr, y_tr, _) = simulate(N_COVERAGE_TRAIN, PC_DIM_COVERAGE, train_seed);
        let (x_te, _y_te, y_true_te) = simulate(N_COVERAGE_HOLDOUT, PC_DIM_COVERAGE, test_seed);

        let spec = duchon_aniso_pc_spec(
            &format!("duchon_pc_cov_{sim_idx}"),
            PC_DIM_COVERAGE,
            K_COVERAGE,
        );
        let weights = Array1::ones(N_COVERAGE_TRAIN);
        let offset_tr = Array1::<f64>::zeros(N_COVERAGE_TRAIN);

        let start = Instant::now();
        let fitted = fit_term_collection_forspec(
            x_tr.view(),
            y_tr.view(),
            weights.view(),
            offset_tr.view(),
            &spec,
            LikelihoodFamily::GaussianIdentity,
            &fit_options(30),
        )
        .expect("coverage-sim Duchon-on-PC fit should succeed");
        let elapsed = start.elapsed();
        assert!(
            elapsed.as_secs_f64() < WALLCLOCK_BUDGET_PER_COVERAGE_FIT_SECS,
            "coverage-sim fit {sim_idx} exceeded per-fit budget: {:.1}s >= {:.1}s",
            elapsed.as_secs_f64(),
            WALLCLOCK_BUDGET_PER_COVERAGE_FIT_SECS,
        );
        assert!(
            fitted.fit.outer_converged,
            "coverage-sim {sim_idx}: REML did not converge",
        );

        let frozenspec = freeze_term_collection_from_design(&spec, &fitted.design)
            .expect("coverage-sim freeze spec must succeed");
        let holdout_design = build_term_collection_design(x_te.view(), &frozenspec)
            .expect("coverage-sim holdout design build must succeed");
        let holdout_dense = holdout_design.design.to_dense();
        let offset_te = Array1::<f64>::zeros(N_COVERAGE_HOLDOUT);

        let pred = predict_gamwith_uncertainty(
            holdout_dense.view(),
            fitted.fit.beta.view(),
            offset_te.view(),
            LikelihoodFamily::GaussianIdentity,
            &fitted.fit,
            &PredictUncertaintyOptions {
                confidence_level: 0.95,
                covariance_mode: InferenceCovarianceMode::ConditionalPlusSmoothingPreferred,
                mean_interval_method: MeanIntervalMethod::TransformEta,
                includeobservation_interval: false,
                apply_bias_correction: true,
                ..PredictUncertaintyOptions::default()
            },
        )
        .expect("coverage-sim uncertainty prediction must succeed");

        for i in 0..N_COVERAGE_HOLDOUT {
            let lo = pred.mean_lower[i];
            let hi = pred.mean_upper[i];
            let truth_i = y_true_te[i];
            if truth_i >= lo && truth_i <= hi {
                total_in += 1;
            }
            total_pts += 1;
        }
    }

    let coverage = total_in as f64 / total_pts.max(1) as f64;
    assert!(
        coverage > 0.85,
        "empirical 95% coverage too low: {coverage:.4} (expected > 0.85, \
         {total_in}/{total_pts})",
    );
    eprintln!(
        "[biobank_reml_stress_coverage] sims={N_COVERAGE_SIMS}, points={total_pts}, \
         coverage={coverage:.4}",
    );
}
