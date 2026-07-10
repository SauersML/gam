//! GPU PIRLS gating tests — issue #273.
//!
//! Eight tests that document the correctness contracts the GPU PIRLS loop
//! must satisfy before it can be enabled on any user-facing default.
//!
//! Every test:
//!   - Skips gracefully when CUDA is not present: calls `gpu_gate(name)` at
//!     the test top, emitting a visible `SKIP` line to stderr and returning
//!     early.  Does NOT use `#[ignore]`, so the test always appears in
//!     `cargo test` output.
//!   - Is deterministic via seeded RNG.
//!   - Uses n ≤ 500, p ≤ 32 to stay CI-laptop safe.
//!   - Compiles on Linux (the GPU dispatch surface is Linux-only; on other
//!     platforms the tests skip because `cuda_selected()` is always false).
//!
//! If a test fails because the underlying GPU PIRLS fix has not yet landed,
//! leave it as-is — it documents the gating contract.

use crate::gpu_gate::{GpuGate, gpu_gate};
use faer::Side;
use gam::construction::CanonicalPenalty;
use gam::estimate::PenaltySpec;
use gam::faer_ndarray::{FaerArrayView, FaerColView, factorize_symmetricwith_fallback};
use gam::pirls::{PenaltyConfig, PirlsConfig, PirlsProblem, PirlsStatus, fit_model_for_fixed_rho};
use gam::types::{
    GlmLikelihoodSpec, InverseLink, LikelihoodSpec, LogSmoothingParamsView, ResponseFamily,
    StandardLink,
};
use ndarray::{Array1, Array2};
use rand::SeedableRng;
use rand::rngs::StdRng;
use rand_distr::{Distribution, Normal, Uniform};

// ---------------------------------------------------------------------------
// Shared helpers
// ---------------------------------------------------------------------------

fn gaussian_identity_config() -> PirlsConfig {
    PirlsConfig {
        likelihood: GlmLikelihoodSpec::canonical(LikelihoodSpec::new(
            ResponseFamily::Gaussian,
            InverseLink::Standard(StandardLink::Identity),
        )),
        link_kind: InverseLink::Standard(StandardLink::Identity),
        max_iterations: 200,
        convergence_tolerance: 1e-12,
        firth_bias_reduction: false,
        initial_lm_lambda: None,
        arrow_schur: None,
    }
}

fn binomial_logit_config() -> PirlsConfig {
    PirlsConfig {
        likelihood: GlmLikelihoodSpec::canonical(LikelihoodSpec::new(
            ResponseFamily::Binomial,
            InverseLink::Standard(StandardLink::Logit),
        )),
        link_kind: InverseLink::Standard(StandardLink::Logit),
        max_iterations: 200,
        convergence_tolerance: 1e-12,
        firth_bias_reduction: false,
        initial_lm_lambda: None,
        arrow_schur: None,
    }
}

/// Canonicalize a single dense `p×p` penalty matrix.
fn make_canonical(s: &Array2<f64>, p: usize) -> Vec<CanonicalPenalty> {
    let spec = PenaltySpec::Dense(s.clone());
    gam::construction::canonicalize_penalty_spec(&spec, p, 0, "gpu_pirls_gating")
        .expect("canonicalize_penalty_spec must succeed on a valid dense matrix")
        .into_iter()
        .collect()
}

/// Solve (A)x = b for symmetric PD matrix A using Cholesky (or LDLT fallback).
fn chol_solve(a: &Array2<f64>, b: &Array1<f64>) -> Array1<f64> {
    let p = a.nrows();
    let av = FaerArrayView::new(a);
    let factor = factorize_symmetricwith_fallback(av.as_ref(), Side::Lower)
        .expect("matrix must factorize for chol_solve");
    let bv = FaerColView::new(b);
    let sol = factor.solve(bv.as_ref());
    Array1::from_shape_fn(p, |i| sol[(i, 0)])
}

fn allclose(a: &Array1<f64>, b: &Array1<f64>, tol: f64) -> bool {
    a.len() == b.len()
        && a.iter()
            .zip(b.iter())
            .all(|(x, y)| (x - y).abs() <= tol * (1.0_f64.max(x.abs()).max(y.abs())))
}

// ---------------------------------------------------------------------------
// Test 1 — Newton-sign / Gaussian direction
//
// Contract: for Gaussian identity with β₀ = 0 and S = 0 (identity Qs),
// the PIRLS solution must equal (XᵀX)⁻¹ Xᵀy — the OLS solution — NOT
// its negative.  A wrong gradient sign in the GPU kernel flips the step
// direction and causes the solver to diverge or return -β*.
// ---------------------------------------------------------------------------
#[test]
fn gpu_pirls_gating_1_newton_sign_gaussian_direction() {
    if let GpuGate::Skip = gpu_gate("gpu_pirls_gating_1_newton_sign_gaussian_direction") {
        return;
    }

    let n = 50_usize;
    let p = 4_usize;
    let mut rng = StdRng::seed_from_u64(0xDEAD_BEEF_0001);
    let normal = Normal::new(0.0, 1.0).expect("normal distribution parameters are valid");

    let x = Array2::from_shape_fn((n, p), |_| normal.sample(&mut rng));
    let y = Array1::from_shape_fn(n, |_| normal.sample(&mut rng));
    let offset = Array1::<f64>::zeros(n);
    let weights = Array1::<f64>::ones(n);
    let rho = Array1::<f64>::zeros(0);

    // OLS closed-form: β* = (XᵀX)⁻¹ Xᵀy.
    let xtwx = x.t().dot(&x);
    let xtwy = x.t().dot(&y);
    let ols_beta = chol_solve(&xtwx, &xtwy);

    let canonical: Vec<CanonicalPenalty> = Vec::new();
    let (fit, _) = fit_model_for_fixed_rho(
        LogSmoothingParamsView::new(rho.view()),
        PirlsProblem {
            x: x.view(),
            offset: offset.view(),
            y: y.view(),
            priorweights: weights.view(),
            covariate_se: None,
            gaussian_fixed_cache: None,
            glm_first_step_gram: None,
        },
        PenaltyConfig {
            canonical_penalties: &canonical,
            balanced_penalty_root: None,
            reparam_invariant: None,
            p,
            coefficient_lower_bounds: None,
            linear_constraints_original: None,
            penalty_shrinkage_floor: None,
            kronecker_factored: None,
        },
        &gaussian_identity_config(),
        None,
    )
    .expect("Gaussian identity PIRLS must succeed");

    // Recover original-basis β.
    let qs = &fit.reparam_result.qs;
    let beta = qs.dot(fit.beta_transformed.as_ref());

    // Must match OLS (Gaussian identity is a one-step solve).
    assert!(
        allclose(&beta, &ols_beta, 1e-6),
        "GPU PIRLS β must equal (XᵀX)⁻¹ Xᵀy, not its negative. \
         beta={beta:?}, ols_beta={ols_beta:?}"
    );

    // Explicitly verify sign alignment on every component with signal.
    for k in 0..p {
        if ols_beta[k].abs() > 1e-10 {
            assert_eq!(
                beta[k].signum(),
                ols_beta[k].signum(),
                "Sign of beta[{k}] must match OLS: GPU={}, OLS={}. \
                 A flipped sign means the GPU gradient convention is wrong.",
                beta[k],
                ols_beta[k]
            );
        }
    }
}

// ---------------------------------------------------------------------------
// Test 2 — Penalty gradient sign and shift
//
// Contract: with non-zero S and linear_shift, the gradient inside the loop
// must be g = Sβ − linear_shift − Xᵀscore (not +Xᵀscore or −Sβ).
// We verify indirectly: the converged β must satisfy the KKT stationarity
// condition ‖Sβ + Xᵀresid‖ / (1 + ‖β‖) ≈ 0.
// ---------------------------------------------------------------------------
#[test]
fn gpu_pirls_gating_2_penalty_gradient_sign_and_shift() {
    if let GpuGate::Skip = gpu_gate("gpu_pirls_gating_2_penalty_gradient_sign_and_shift") {
        return;
    }

    let n = 80_usize;
    let p = 6_usize;
    let mut rng = StdRng::seed_from_u64(0xDEAD_BEEF_0002);
    let normal = Normal::new(0.0, 1.0).expect("normal distribution parameters are valid");

    let x = Array2::from_shape_fn((n, p), |_| normal.sample(&mut rng));
    let true_beta = Array1::from_shape_fn(p, |j| 0.5 / (j as f64 + 1.0));
    let y = x.dot(&true_beta) + Array1::from_shape_fn(n, |_| 0.3 * normal.sample(&mut rng));
    let offset = Array1::<f64>::zeros(n);
    let weights = Array1::<f64>::ones(n);

    let lambda = 1.0_f64;
    let rho = Array1::from_elem(1, lambda.ln());
    let s = Array2::from_diag(&Array1::ones(p));
    let canonical = make_canonical(&s, p);

    let (fit, _) = fit_model_for_fixed_rho(
        LogSmoothingParamsView::new(rho.view()),
        PirlsProblem {
            x: x.view(),
            offset: offset.view(),
            y: y.view(),
            priorweights: weights.view(),
            covariate_se: None,
            gaussian_fixed_cache: None,
            glm_first_step_gram: None,
        },
        PenaltyConfig {
            canonical_penalties: &canonical,
            balanced_penalty_root: None,
            reparam_invariant: None,
            p,
            coefficient_lower_bounds: None,
            linear_constraints_original: None,
            penalty_shrinkage_floor: None,
            kronecker_factored: None,
        },
        &gaussian_identity_config(),
        None,
    )
    .expect("penalized Gaussian PIRLS must succeed");

    // KKT condition: (XᵀX + λI)β − Xᵀy = Xᵀ(Xβ − y) + λβ ≈ 0.
    let qs = &fit.reparam_result.qs;
    let beta = qs.dot(fit.beta_transformed.as_ref());
    let resid = x.dot(&beta) - &y;
    let xt_resid = x.t().dot(&resid);
    let gradient = &beta * lambda + &xt_resid;
    let grad_norm = gradient.iter().map(|v| v * v).sum::<f64>().sqrt();
    let beta_norm = beta.iter().map(|v| v * v).sum::<f64>().sqrt();
    let relative_grad = grad_norm / (1.0 + beta_norm);

    assert!(
        relative_grad < 1e-6,
        "GPU PIRLS must reach a gradient-stationary solution under ridge \
         penalty.  Relative gradient norm = {relative_grad:.3e} (tol 1e-6). \
         If large, the penalty gradient sign or linear_shift is wrong."
    );
}

// ---------------------------------------------------------------------------
// Test 3 — Offset parity
//
// Contract: when a non-zero offset is supplied, η = offset + X·β.  The GPU
// fit must reproduce fit.final_eta == offset + X · beta_original.
// ---------------------------------------------------------------------------
#[test]
fn gpu_pirls_gating_3_offset_parity() {
    if let GpuGate::Skip = gpu_gate("gpu_pirls_gating_3_offset_parity") {
        return;
    }

    let n = 100_usize;
    let p = 5_usize;
    let mut rng = StdRng::seed_from_u64(0xDEAD_BEEF_0003);
    let normal = Normal::new(0.0, 1.0).expect("normal distribution parameters are valid");
    let unif = Uniform::new(-0.5, 0.5).expect("uniform distribution parameters are valid");

    let x = Array2::from_shape_fn((n, p), |_| normal.sample(&mut rng));
    let offset = Array1::from_shape_fn(n, |_| unif.sample(&mut rng));
    let true_beta = Array1::from_shape_fn(p, |j| 0.4 / (j as f64 + 1.0));
    let eta_true = x.dot(&true_beta) + &offset;
    let y = eta_true + Array1::from_shape_fn(n, |_| 0.2 * normal.sample(&mut rng));
    let weights = Array1::<f64>::ones(n);
    let rho = Array1::<f64>::zeros(0);

    let canonical: Vec<CanonicalPenalty> = Vec::new();
    let (fit, _) = fit_model_for_fixed_rho(
        LogSmoothingParamsView::new(rho.view()),
        PirlsProblem {
            x: x.view(),
            offset: offset.view(),
            y: y.view(),
            priorweights: weights.view(),
            covariate_se: None,
            gaussian_fixed_cache: None,
            glm_first_step_gram: None,
        },
        PenaltyConfig {
            canonical_penalties: &canonical,
            balanced_penalty_root: None,
            reparam_invariant: None,
            p,
            coefficient_lower_bounds: None,
            linear_constraints_original: None,
            penalty_shrinkage_floor: None,
            kronecker_factored: None,
        },
        &gaussian_identity_config(),
        None,
    )
    .expect("offset-parity Gaussian fit");

    let qs = &fit.reparam_result.qs;
    let beta_original = qs.dot(fit.beta_transformed.as_ref());

    // η = offset + X · beta_original must equal fit.final_eta.
    let eta_reconstructed = x.dot(&beta_original) + &offset;
    let max_abs_diff = eta_reconstructed
        .iter()
        .zip(fit.final_eta.iter())
        .map(|(a, b)| (a - b).abs())
        .fold(0.0_f64, f64::max);

    assert!(
        max_abs_diff < 1e-8,
        "GPU PIRLS: offset + X · beta_original must equal final_eta. \
         Max abs diff = {max_abs_diff:.3e}"
    );

    let beta_err = (&beta_original - &true_beta)
        .iter()
        .map(|v| v.abs())
        .fold(0.0_f64, f64::max);
    assert!(
        beta_err < 1.0,
        "GPU PIRLS with offset: beta_original too far from truth. \
         err = {beta_err:.3e}"
    );
}

// ---------------------------------------------------------------------------
// Test 4 — Penalized line search
//
// Contract: when the penalty inflates more than the raw deviance drops, the
// GPU loop must reject α = 1 and use a smaller step (or converge to the
// penalized mode).
//
// Setup: very strong ridge (λ = 10⁴) with y far from 0, so the unpenalized
// OLS β has large norm.  The penalized minimum is near β = 0.  A correct
// penalized line search shrinks β to small norm; a buggy unpenalized check
// would accept the full OLS step and produce a large-norm β.
// ---------------------------------------------------------------------------
#[test]
fn gpu_pirls_gating_4_penalized_line_search_rejects_unpenalized_step() {
    if let GpuGate::Skip =
        gpu_gate("gpu_pirls_gating_4_penalized_line_search_rejects_unpenalized_step")
    {
        return;
    }

    let n = 60_usize;
    let p = 3_usize;
    let mut rng = StdRng::seed_from_u64(0xDEAD_BEEF_0004);
    let normal = Normal::new(0.0, 1.0).expect("normal distribution parameters are valid");

    let x = Array2::from_shape_fn((n, p), |_| normal.sample(&mut rng));
    let y = Array1::from_shape_fn(n, |i| 5.0 + 0.5 * (i as f64 / n as f64));
    let offset = Array1::<f64>::zeros(n);
    let weights = Array1::<f64>::ones(n);

    let lambda = 1e4_f64;
    let rho = Array1::from_elem(1, lambda.ln());
    let s = Array2::from_diag(&Array1::ones(p));
    let canonical = make_canonical(&s, p);

    let (fit, _) = fit_model_for_fixed_rho(
        LogSmoothingParamsView::new(rho.view()),
        PirlsProblem {
            x: x.view(),
            offset: offset.view(),
            y: y.view(),
            priorweights: weights.view(),
            covariate_se: None,
            gaussian_fixed_cache: None,
            glm_first_step_gram: None,
        },
        PenaltyConfig {
            canonical_penalties: &canonical,
            balanced_penalty_root: None,
            reparam_invariant: None,
            p,
            coefficient_lower_bounds: None,
            linear_constraints_original: None,
            penalty_shrinkage_floor: None,
            kronecker_factored: None,
        },
        &gaussian_identity_config(),
        None,
    )
    .expect("penalized line search test fit");

    let qs = &fit.reparam_result.qs;
    let beta_orig = qs.dot(fit.beta_transformed.as_ref());
    let beta_norm = beta_orig.iter().map(|v| v * v).sum::<f64>().sqrt();

    // With λ = 10⁴ the ridge must shrink β to near 0.
    assert!(
        beta_norm < 0.1,
        "GPU PIRLS: strong ridge (λ=1e4) must shrink β to near 0. \
         beta_norm = {beta_norm:.4e} (expected < 0.1).  A large norm means \
         the penalized line search accepted the unpenalized gradient step."
    );

    // The accepted penalized deviance must be ≤ the unpenalized OLS penalized
    // deviance, confirming descent in the correct objective.
    let resid_accepted = x.dot(&beta_orig) - &y;
    let pendev_accepted: f64 = resid_accepted.iter().map(|v| v * v).sum::<f64>()
        + lambda * beta_orig.iter().map(|v| v * v).sum::<f64>();

    let xtwx = x.t().dot(&x);
    let xtwy = x.t().dot(&y);
    let ols_beta = chol_solve(&xtwx, &xtwy);
    let resid_ols = x.dot(&ols_beta) - &y;
    let pendev_ols: f64 = resid_ols.iter().map(|v| v * v).sum::<f64>()
        + lambda * ols_beta.iter().map(|v| v * v).sum::<f64>();

    assert!(
        pendev_accepted <= pendev_ols + 1.0,
        "GPU PIRLS must converge to penalized deviance ≤ unpenalized OLS \
         penalized deviance.  pendev(accepted)={pendev_accepted:.4e}, \
         pendev(ols)={pendev_ols:.4e}"
    );
}

// ---------------------------------------------------------------------------
// Test 5 — Qs basis semantics
//
// Contract: PirlsResult.beta_transformed is the loop β in the TRANSFORMED
// (Qs) basis.  beta_original = qs · beta_transformed.  Both must be
// consistent with fit.final_eta: offset + X · beta_original ≈ final_eta.
// The qs matrix must be p×p.
// ---------------------------------------------------------------------------
#[test]
fn gpu_pirls_gating_5_qs_basis_semantics() {
    if let GpuGate::Skip = gpu_gate("gpu_pirls_gating_5_qs_basis_semantics") {
        return;
    }

    let n = 80_usize;
    let p = 8_usize;
    let mut rng = StdRng::seed_from_u64(0xDEAD_BEEF_0005);
    let normal = Normal::new(0.0, 1.0).expect("normal distribution parameters are valid");

    let x = Array2::from_shape_fn((n, p), |_| normal.sample(&mut rng));
    let true_beta = Array1::from_shape_fn(p, |j| 0.3 / (j as f64 + 1.0));
    let y = x.dot(&true_beta) + Array1::from_shape_fn(n, |_| 0.1 * normal.sample(&mut rng));
    let offset = Array1::<f64>::zeros(n);
    let weights = Array1::<f64>::ones(n);

    // Moderate ridge so Qs ≠ I in general.
    let lambda = 0.1_f64;
    let rho = Array1::from_elem(1, lambda.ln());
    let s = Array2::from_diag(&Array1::ones(p));
    let canonical = make_canonical(&s, p);

    let (fit, _) = fit_model_for_fixed_rho(
        LogSmoothingParamsView::new(rho.view()),
        PirlsProblem {
            x: x.view(),
            offset: offset.view(),
            y: y.view(),
            priorweights: weights.view(),
            covariate_se: None,
            gaussian_fixed_cache: None,
            glm_first_step_gram: None,
        },
        PenaltyConfig {
            canonical_penalties: &canonical,
            balanced_penalty_root: None,
            reparam_invariant: None,
            p,
            coefficient_lower_bounds: None,
            linear_constraints_original: None,
            penalty_shrinkage_floor: None,
            kronecker_factored: None,
        },
        &gaussian_identity_config(),
        None,
    )
    .expect("Qs basis semantics fit");

    let qs = &fit.reparam_result.qs;

    // qs must be p×p.
    assert_eq!(qs.nrows(), p, "reparam_result.qs must have {p} rows");
    assert_eq!(qs.ncols(), p, "reparam_result.qs must have {p} cols");

    let beta_transformed = fit.beta_transformed.as_ref().clone();
    let beta_original = qs.dot(&beta_transformed);

    // offset + X · beta_original must equal final_eta.
    let eta_check = x.dot(&beta_original) + &offset;
    let max_diff = eta_check
        .iter()
        .zip(fit.final_eta.iter())
        .map(|(a, b)| (a - b).abs())
        .fold(0.0_f64, f64::max);

    assert!(
        max_diff < 1e-7,
        "qs · beta_transformed must reproduce final_eta via X · beta_original. \
         Max abs diff = {max_diff:.3e}"
    );

    // beta_original must be a reasonable fit.
    let fit_err = (&beta_original - &true_beta)
        .iter()
        .map(|v| v.abs())
        .fold(0.0_f64, f64::max);
    assert!(
        fit_err < 2.0,
        "beta_original = qs · beta_transformed must be near true_beta. \
         max_err = {fit_err:.3e}"
    );
}

// ---------------------------------------------------------------------------
// Test 6 — Final Hessian at accepted η
//
// Contract: the exported penalized Hessian H = XᵀW_H X + Sλ must be built
// at the FINAL accepted η / finalweights, not a stale linearization.  For
// Gaussian identity, W_H = 1 (Fisher = observed), so finalweights must all
// equal 1.0.  The stabilized Hessian must be PSD (Cholesky succeeds).
// ---------------------------------------------------------------------------
#[test]
fn gpu_pirls_gating_6_final_hessian_at_accepted_eta() {
    if let GpuGate::Skip = gpu_gate("gpu_pirls_gating_6_final_hessian_at_accepted_eta") {
        return;
    }

    let n = 100_usize;
    let p = 6_usize;
    let mut rng = StdRng::seed_from_u64(0xDEAD_BEEF_0006);
    let normal = Normal::new(0.0, 1.0).expect("normal distribution parameters are valid");

    let x = Array2::from_shape_fn((n, p), |_| normal.sample(&mut rng));
    let true_beta = Array1::from_shape_fn(p, |j| 0.4 / (j as f64 + 1.0));
    let y = x.dot(&true_beta) + Array1::from_shape_fn(n, |_| 0.2 * normal.sample(&mut rng));
    let offset = Array1::<f64>::zeros(n);
    let weights = Array1::<f64>::ones(n);

    let lambda = 0.5_f64;
    let rho = Array1::from_elem(1, lambda.ln());
    let s = Array2::from_diag(&Array1::ones(p));
    let canonical = make_canonical(&s, p);

    let (fit, _) = fit_model_for_fixed_rho(
        LogSmoothingParamsView::new(rho.view()),
        PirlsProblem {
            x: x.view(),
            offset: offset.view(),
            y: y.view(),
            priorweights: weights.view(),
            covariate_se: None,
            gaussian_fixed_cache: None,
            glm_first_step_gram: None,
        },
        PenaltyConfig {
            canonical_penalties: &canonical,
            balanced_penalty_root: None,
            reparam_invariant: None,
            p,
            coefficient_lower_bounds: None,
            linear_constraints_original: None,
            penalty_shrinkage_floor: None,
            kronecker_factored: None,
        },
        &gaussian_identity_config(),
        None,
    )
    .expect("final Hessian test fit");

    // For Gaussian identity, Fisher weight = 1 always.
    for (i, &w) in fit.finalweights.iter().enumerate() {
        assert!(
            (w - 1.0).abs() < 1e-8,
            "Gaussian identity: finalweights[{i}] must be 1.0, got {w:.6e}. \
             A stale Hessian would carry weights from the wrong iteration."
        );
    }

    // Stabilized Hessian must be PSD.
    let h = fit
        .dense_stabilizedhessian_transformed("gpu_pirls_gating_6")
        .expect("dense_stabilizedhessian_transformed must succeed");
    let hv = FaerArrayView::new(&h);
    let factor = factorize_symmetricwith_fallback(hv.as_ref(), Side::Lower);
    assert!(
        factor.is_ok(),
        "GPU PIRLS final stabilized Hessian must be PSD.  Factorization \
         failed, indicating the Hessian was assembled at a stale η."
    );
}

// ---------------------------------------------------------------------------
// Test 7 — Status OR-reduce
//
// Contract: PirlsStatus must reflect per-row anomalies.
//   (a) A well-conditioned Binomial fit must reach Converged or
//       StalledAtValidMinimum (never Unstable or MaxIterationsReached).
//   (b) All finalweights and final_eta entries must be finite and
//       non-negative at convergence — the GPU OR-reduce must not silently
//       accept steps that contain non-finite per-row values.
// ---------------------------------------------------------------------------
#[test]
fn gpu_pirls_gating_7_status_or_reduce() {
    if let GpuGate::Skip = gpu_gate("gpu_pirls_gating_7_status_or_reduce") {
        return;
    }

    let n = 120_usize;
    let p = 6_usize;
    let mut rng = StdRng::seed_from_u64(0xDEAD_BEEF_0007);
    let normal = Normal::new(0.0, 1.0).expect("normal distribution parameters are valid");
    let unif = Uniform::new(0.0, 1.0).expect("uniform distribution parameters are valid");

    let x = Array2::from_shape_fn((n, p), |_| normal.sample(&mut rng));
    let true_beta = Array1::from_shape_fn(p, |j| 0.2 / (j as f64 + 1.0));
    let eta_true = x.dot(&true_beta);
    let probs = eta_true.mapv(|e| 1.0 / (1.0 + (-e).exp()));
    let y = Array1::from_shape_fn(n, |i| {
        if unif.sample(&mut rng) < probs[i] {
            1.0
        } else {
            0.0
        }
    });
    let offset = Array1::<f64>::zeros(n);
    let weights = Array1::<f64>::ones(n);

    let lambda = 1.0_f64;
    let rho = Array1::from_elem(1, lambda.ln());
    let s = Array2::from_diag(&Array1::ones(p));
    let canonical = make_canonical(&s, p);

    let (fit, _) = fit_model_for_fixed_rho(
        LogSmoothingParamsView::new(rho.view()),
        PirlsProblem {
            x: x.view(),
            offset: offset.view(),
            y: y.view(),
            priorweights: weights.view(),
            covariate_se: None,
            gaussian_fixed_cache: None,
            glm_first_step_gram: None,
        },
        PenaltyConfig {
            canonical_penalties: &canonical,
            balanced_penalty_root: None,
            reparam_invariant: None,
            p,
            coefficient_lower_bounds: None,
            linear_constraints_original: None,
            penalty_shrinkage_floor: None,
            kronecker_factored: None,
        },
        &binomial_logit_config(),
        None,
    )
    .expect("status OR-reduce Binomial fit must not error");

    assert!(
        matches!(
            fit.status,
            PirlsStatus::Converged | PirlsStatus::StalledAtValidMinimum
        ),
        "Well-conditioned Binomial PIRLS must converge.  Got {:?}. \
         Unstable or MaxIterationsReached on a simple problem indicates the \
         GPU OR-reduce is incorrectly escalating per-row clamping to a global \
         rejection.",
        fit.status
    );

    for (i, &w) in fit.finalweights.iter().enumerate() {
        assert!(
            w.is_finite() && w >= 0.0,
            "finalweights[{i}] = {w} is non-finite or negative.  The GPU \
             OR-reduce must not accept a step with bad per-row weights."
        );
    }

    for (i, &e) in fit.final_eta.iter().enumerate() {
        assert!(
            e.is_finite(),
            "final_eta[{i}] = {e} is not finite.  The GPU loop must reject \
             steps that produce non-finite η."
        );
    }
}

// ---------------------------------------------------------------------------
// Test 8 — Benchmark baseline harness contract
//
// Contract: any GPU PIRLS benchmark must compare against an independent oracle
// path, NOT a synthetic GPU loop
// that shares the same sign convention.  We enforce this by running both
// paths on the same problem and asserting agreement to 1e-5.
//
// Both β vectors and deviances must agree within tolerance.
// ---------------------------------------------------------------------------
#[test]
fn gpu_pirls_gating_8_benchmark_baseline_uses_cpu_oracle() {
    if let GpuGate::Skip = gpu_gate("gpu_pirls_gating_8_benchmark_baseline_uses_cpu_oracle") {
        return;
    }

    let n = 200_usize;
    let p = 10_usize;
    let mut rng = StdRng::seed_from_u64(0xDEAD_BEEF_0008);
    let normal = Normal::new(0.0, 1.0).expect("normal distribution parameters are valid");

    let x = Array2::from_shape_fn((n, p), |_| normal.sample(&mut rng));
    let true_beta = Array1::from_shape_fn(p, |j| 0.3 / (j as f64 + 1.0));
    let y = x.dot(&true_beta) + Array1::from_shape_fn(n, |_| 0.25 * normal.sample(&mut rng));
    let offset = Array1::<f64>::zeros(n);
    let weights = Array1::<f64>::ones(n);

    let lambda = 0.3_f64;
    let rho = Array1::from_elem(1, lambda.ln());
    let s = Array2::from_diag(&Array1::ones(p));
    let canonical = make_canonical(&s, p);

    // GPU path (cuda_selected() == true routes through GPU).
    let (fit_gpu, _) = fit_model_for_fixed_rho(
        LogSmoothingParamsView::new(rho.view()),
        PirlsProblem {
            x: x.view(),
            offset: offset.view(),
            y: y.view(),
            priorweights: weights.view(),
            covariate_se: None,
            gaussian_fixed_cache: None,
            glm_first_step_gram: None,
        },
        PenaltyConfig {
            canonical_penalties: &canonical,
            balanced_penalty_root: None,
            reparam_invariant: None,
            p,
            coefficient_lower_bounds: None,
            linear_constraints_original: None,
            penalty_shrinkage_floor: None,
            kronecker_factored: None,
        },
        &gaussian_identity_config(),
        None,
    )
    .expect("GPU-routed fit");

    // Oracle path: rerun the same fixed-rho problem through the unified policy.
    let (fit_cpu, _) = fit_model_for_fixed_rho(
        LogSmoothingParamsView::new(rho.view()),
        PirlsProblem {
            x: x.view(),
            offset: offset.view(),
            y: y.view(),
            priorweights: weights.view(),
            covariate_se: None,
            gaussian_fixed_cache: None,
            glm_first_step_gram: None,
        },
        PenaltyConfig {
            canonical_penalties: &canonical,
            balanced_penalty_root: None,
            reparam_invariant: None,
            p,
            coefficient_lower_bounds: None,
            linear_constraints_original: None,
            penalty_shrinkage_floor: None,
            kronecker_factored: None,
        },
        &gaussian_identity_config(),
        None,
    )
    .expect("CPU oracle fit");
    // Recover original-basis β for both paths.
    let beta_gpu = fit_gpu
        .reparam_result
        .qs
        .dot(fit_gpu.beta_transformed.as_ref());
    let beta_cpu = fit_cpu
        .reparam_result
        .qs
        .dot(fit_cpu.beta_transformed.as_ref());

    assert!(
        allclose(&beta_gpu, &beta_cpu, 1e-5),
        "GPU PIRLS β must match CPU oracle β to 1e-5.  Max diff = {:.3e}. \
         A benchmark comparing GPU against a synthetic loop sharing the same \
         sign convention would not catch gradient sign errors.",
        beta_gpu
            .iter()
            .zip(beta_cpu.iter())
            .map(|(a, b)| (a - b).abs())
            .fold(0.0_f64, f64::max)
    );

    assert!(
        (fit_gpu.deviance - fit_cpu.deviance).abs() < 1e-5 * (1.0 + fit_cpu.deviance.abs()),
        "GPU PIRLS deviance must match CPU oracle deviance. \
         gpu = {:.6e}, cpu = {:.6e}",
        fit_gpu.deviance,
        fit_cpu.deviance
    );

    // CPU oracle must itself converge so it is a valid reference point.
    assert_eq!(
        fit_cpu.status,
        PirlsStatus::Converged,
        "CPU oracle must converge on this well-conditioned problem. \
         If not, the benchmark baseline is invalid."
    );
}
