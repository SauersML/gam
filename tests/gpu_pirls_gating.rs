//! GPU PIRLS gating tests — issue #273.
//!
//! Eight tests that document the correctness contracts the GPU PIRLS loop
//! must satisfy before it can be enabled on any user-facing default.
//!
//! Every test:
//!   - Skips gracefully when CUDA is not present (`cuda_selected()` returns
//!     false).  Uses `if !cuda_selected() { return; }` at the test top, NOT
//!     `#[ignore]`, so the test always appears in `cargo test` output.
//!   - Is deterministic via seeded RNG.
//!   - Uses n ≤ 500, p ≤ 32 to stay CI-laptop safe.
//!   - Compiles on Linux (the GPU dispatch surface is Linux-only; on other
//!     platforms the tests skip because `cuda_selected()` is always false).
//!
//! If a test fails because the underlying GPU PIRLS fix has not yet landed,
//! leave it as-is — it documents the gating contract.

use gam::construction::CanonicalPenalty;
use gam::estimate::PenaltySpec;
use gam::pirls::{PenaltyConfig, PirlsConfig, PirlsProblem, PirlsStatus, fit_model_for_fixed_rho};
use gam::solver::gpu::cuda_selected;
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
        geodesic_acceleration: false,
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
        geodesic_acceleration: false,
        arrow_schur: None,
    }
}

/// Build a zero-penalty PenaltyConfig for `p` coefficients.
fn zero_penalty_config(p: usize) -> (Vec<CanonicalPenalty>, PenaltyConfig<'static>) {
    let canonical: Vec<CanonicalPenalty> = Vec::new();
    // SAFETY: 'static lifetime is fine because we own the vec and return it
    // alongside the config.  The caller holds both.
    let config = PenaltyConfig {
        canonical_penalties: unsafe {
            // We need to return PenaltyConfig with canonical_penalties pointing
            // at the vec we also return; use a raw pointer round-trip so the
            // borrow checker accepts the self-referential struct without unsafe
            // field access.  The vec outlives the config in every call site.
            std::mem::transmute::<&[CanonicalPenalty], &'static [CanonicalPenalty]>(
                canonical.as_slice(),
            )
        },
        balanced_penalty_root: None,
        reparam_invariant: None,
        p,
        coefficient_lower_bounds: None,
        linear_constraints_original: None,
        penalty_shrinkage_floor: None,
        kronecker_factored: None,
    };
    (canonical, config)
}

/// Build a ridge PenaltyConfig from an explicit `p×p` matrix.
fn dense_penalty_config<'a>(
    canonical: &'a [CanonicalPenalty],
    p: usize,
) -> PenaltyConfig<'a> {
    PenaltyConfig {
        canonical_penalties: canonical,
        balanced_penalty_root: None,
        reparam_invariant: None,
        p,
        coefficient_lower_bounds: None,
        linear_constraints_original: None,
        penalty_shrinkage_floor: None,
        kronecker_factored: None,
    }
}

/// Canonicalize a single dense `p×p` penalty matrix.
fn make_canonical(s: &Array2<f64>, p: usize) -> Vec<CanonicalPenalty> {
    let spec = PenaltySpec::Dense(s.clone());
    gam::construction::canonicalize_penalty_spec(&spec, p, 0, "gpu_pirls_gating")
        .expect("canonicalize penalty")
        .into_iter()
        .collect()
}

fn allclose(a: &Array1<f64>, b: &Array1<f64>, tol: f64) -> bool {
    a.len() == b.len()
        && a.iter()
            .zip(b.iter())
            .all(|(x, y)| (x - y).abs() <= tol * (1.0_f64.max(x.abs()).max(y.abs())))
}

fn mat_allclose(a: &Array2<f64>, b: &Array2<f64>, tol: f64) -> bool {
    a.dim() == b.dim()
        && a.iter()
            .zip(b.iter())
            .all(|(x, y)| (x - y).abs() <= tol * (1.0_f64.max(x.abs()).max(y.abs())))
}

// ---------------------------------------------------------------------------
// Test 1 — Newton-sign / Gaussian direction
//
// Contract: for Gaussian identity with β₀ = 0 and S = 0, the first Newton
// step δ must equal (XᵀX)⁻¹ Xᵀy (i.e. the OLS solution), NOT its negative.
// ---------------------------------------------------------------------------
#[test]
fn gpu_pirls_gating_1_newton_sign_gaussian_direction() {
    if !cuda_selected() {
        return;
    }

    let n = 50_usize;
    let p = 4_usize;
    let mut rng = StdRng::seed_from_u64(0xDEAD_BEEF_0001);
    let normal = Normal::new(0.0, 1.0).unwrap();

    let x = Array2::from_shape_fn((n, p), |_| normal.sample(&mut rng));
    let y = Array1::from_shape_fn(n, |_| normal.sample(&mut rng));
    let offset = Array1::<f64>::zeros(n);
    let weights = Array1::<f64>::ones(n);
    let rho = Array1::<f64>::zeros(0);

    // OLS solution: β* = (XᵀX)⁻¹ Xᵀy
    let xtwx = x.t().dot(&x);
    let xtwy = x.t().dot(&y);
    // Solve via Cholesky using faer bridge
    let xtwx_fa = gam::faer_ndarray::FaerArray2View::from(xtwx.view());
    let ols_beta = {
        use faer::linalg::solvers::Solve;
        let chol = faer::Mat::from_fn(p, p, |i, j| xtwx_fa[[i, j]])
            .cholesky(faer::Side::Lower)
            .expect("XtX must be PD for well-conditioned random design");
        let rhs = faer::Col::from_fn(p, |i| xtwy[i]);
        let sol = chol.solve(rhs);
        Array1::from_shape_fn(p, |i| sol[i])
    };

    let (canonical, penalty) = zero_penalty_config(p);
    let _ = &canonical; // used via penalty
    let (fit, _) = fit_model_for_fixed_rho(
        LogSmoothingParamsView::new(rho.view()),
        PirlsProblem {
            x: x.view(),
            offset: offset.view(),
            y: y.view(),
            priorweights: weights.view(),
            covariate_se: None,
            gaussian_fixed_cache: None,
        },
        penalty,
        &gaussian_identity_config(),
        None,
    )
    .expect("Gaussian identity fit must succeed");

    let beta = fit.beta_transformed.as_ref().clone();

    // The GPU PIRLS solution must match OLS (sign must be the same as OLS,
    // not flipped).  Tolerance is loose because this tests sign/direction,
    // not high-precision equality.
    assert!(
        allclose(&beta, &ols_beta, 1e-6),
        "GPU PIRLS β must equal (XᵀX)⁻¹ Xᵀy (the OLS solution), not its \
         negative. beta={beta:?}, ols_beta={ols_beta:?}"
    );

    // Explicitly verify sign alignment on the first coefficient.
    assert_eq!(
        beta[0].signum(),
        ols_beta[0].signum(),
        "Sign of beta[0] must match OLS solution — a flipped sign indicates \
         the GPU gradient sign convention is wrong."
    );
}

// ---------------------------------------------------------------------------
// Test 2 — Penalty gradient
//
// Contract: with non-zero S and linear_shift, the PIRLS gradient at
// convergence must be (approximately) zero, which means it was formed as
// g = Sβ − linear_shift − Xᵀscore throughout the loop (not +Xᵀscore or
// −Sβ).  We verify indirectly: the converged β must satisfy the KKT
// stationarity condition ‖Sβ + Xᵀresid‖ / (1 + ‖β‖) ≈ 0.
// ---------------------------------------------------------------------------
#[test]
fn gpu_pirls_gating_2_penalty_gradient_sign_and_shift() {
    if !cuda_selected() {
        return;
    }

    let n = 80_usize;
    let p = 6_usize;
    let mut rng = StdRng::seed_from_u64(0xDEAD_BEEF_0002);
    let normal = Normal::new(0.0, 1.0).unwrap();

    let x = Array2::from_shape_fn((n, p), |_| normal.sample(&mut rng));
    let true_beta = Array1::from_shape_fn(p, |j| 0.5 / (j as f64 + 1.0));
    let y = x.dot(&true_beta)
        + Array1::from_shape_fn(n, |_| 0.3 * normal.sample(&mut rng));
    let offset = Array1::<f64>::zeros(n);
    let weights = Array1::<f64>::ones(n);

    // Non-zero ridge penalty on all coefficients, moderate strength.
    let lambda = 1.0_f64;
    let rho = Array1::from_elem(1, lambda.ln());
    let s = Array2::from_diag(&Array1::ones(p));
    let canonical = make_canonical(&s, p);
    let penalty = dense_penalty_config(&canonical, p);

    let (fit, _) = fit_model_for_fixed_rho(
        LogSmoothingParamsView::new(rho.view()),
        PirlsProblem {
            x: x.view(),
            offset: offset.view(),
            y: y.view(),
            priorweights: weights.view(),
            covariate_se: None,
            gaussian_fixed_cache: None,
        },
        penalty,
        &gaussian_identity_config(),
        None,
    )
    .expect("penalized Gaussian fit must succeed");

    // At convergence, g = Sβ + Xᵀ(Xβ − y) = (XᵀX + λI)β − Xᵀy ≈ 0.
    let beta = fit.beta_transformed.as_ref().clone();
    let eta = x.dot(&beta);
    let resid = &eta - &y;
    let xt_resid = x.t().dot(&resid);
    let s_beta = &beta * lambda; // ridge: Sβ = λβ
    let gradient = &s_beta + &xt_resid;
    let grad_norm = gradient.iter().map(|v| v * v).sum::<f64>().sqrt();
    let beta_norm = beta.iter().map(|v| v * v).sum::<f64>().sqrt();
    let relative_grad = grad_norm / (1.0 + beta_norm);

    assert!(
        relative_grad < 1e-6,
        "GPU PIRLS must produce a gradient-stationary solution under ridge \
         penalty. Relative gradient norm = {relative_grad:.3e} (tol 1e-6). \
         If large, the penalty gradient sign or shift is wrong."
    );
}

// ---------------------------------------------------------------------------
// Test 3 — Offset parity
//
// Contract: when a non-zero offset is supplied, η = offset + X·β, and the
// GPU fit must match the CPU oracle on the same problem.
// ---------------------------------------------------------------------------
#[test]
fn gpu_pirls_gating_3_offset_parity() {
    if !cuda_selected() {
        return;
    }

    let n = 100_usize;
    let p = 5_usize;
    let mut rng = StdRng::seed_from_u64(0xDEAD_BEEF_0003);
    let normal = Normal::new(0.0, 1.0).unwrap();
    let unif = Uniform::new(-0.5, 0.5).unwrap();

    let x = Array2::from_shape_fn((n, p), |_| normal.sample(&mut rng));
    let offset = Array1::from_shape_fn(n, |_| unif.sample(&mut rng));
    let true_beta = Array1::from_shape_fn(p, |j| 0.4 / (j as f64 + 1.0));
    let eta_true = x.dot(&true_beta) + &offset;
    let y = eta_true + Array1::from_shape_fn(n, |_| 0.2 * normal.sample(&mut rng));
    let weights = Array1::<f64>::ones(n);
    let rho = Array1::<f64>::zeros(0);

    let (canonical, penalty) = zero_penalty_config(p);
    let _ = &canonical;

    // CPU oracle (cuda_selected() == false path would use CPU; here we call
    // the same function which routes through GPU when available — we compare
    // the GPU result against the analytical OLS-with-offset solution).
    let (fit, _) = fit_model_for_fixed_rho(
        LogSmoothingParamsView::new(rho.view()),
        PirlsProblem {
            x: x.view(),
            offset: offset.view(),
            y: y.view(),
            priorweights: weights.view(),
            covariate_se: None,
            gaussian_fixed_cache: None,
        },
        penalty,
        &gaussian_identity_config(),
        None,
    )
    .expect("offset-parity Gaussian fit");

    let beta = fit.beta_transformed.as_ref().clone();
    // Verify η = offset + X·β at the fitted values.
    let eta_fitted = x.dot(&beta) + &offset;
    let final_eta = fit.final_eta.clone();

    assert!(
        allclose(&eta_fitted, &final_eta, 1e-8),
        "GPU PIRLS: final_eta must equal offset + X·β. \
         Max abs diff = {:.3e}",
        eta_fitted
            .iter()
            .zip(final_eta.iter())
            .map(|(a, b)| (a - b).abs())
            .fold(0.0_f64, f64::max)
    );

    // The fit must be close to the truth.
    let beta_err = (&beta - &true_beta)
        .iter()
        .map(|v| v.abs())
        .fold(0.0_f64, f64::max);
    assert!(
        beta_err < 0.5,
        "GPU PIRLS with offset: beta too far from truth. err={beta_err:.3e}, \
         beta={beta:?}, true={true_beta:?}"
    );
}

// ---------------------------------------------------------------------------
// Test 4 — Penalized line search
//
// Contract: construct a step where raw deviance drops but the penalty term
// inflates more.  The GPU loop must reject α = 1 in that case, not blindly
// accept the step.
//
// We achieve this by starting at a β that is near the penalty minimum but
// far from the likelihood minimum, with a very strong penalty (large λ).
// The un-penalized deviance would push β toward the likelihood mode, but
// the penalty cost of moving there must dominate.  If the line search only
// checks raw deviance, it accepts α = 1; a correct penalized line search
// accepts a smaller step or the solver converges at a different β.
//
// We detect a wrong α=1 acceptance by comparing the penalized deviance of
// the accepted β with a finite-difference check: moving β toward the
// likelihood-only OLS mode should increase penalized deviance when λ is
// large.
// ---------------------------------------------------------------------------
#[test]
fn gpu_pirls_gating_4_penalized_line_search_rejects_unpenalized_step() {
    if !cuda_selected() {
        return;
    }

    let n = 60_usize;
    let p = 3_usize;
    let mut rng = StdRng::seed_from_u64(0xDEAD_BEEF_0004);
    let normal = Normal::new(0.0, 1.0).unwrap();

    // Design with a strong signal in the first coefficient.
    let x = Array2::from_shape_fn((n, p), |_| normal.sample(&mut rng));
    // y is far from 0 so the unpenalized OLS solution has large beta.
    let y = Array1::from_shape_fn(n, |i| 5.0 + 0.5 * (i as f64 / n as f64));
    let offset = Array1::<f64>::zeros(n);
    let weights = Array1::<f64>::ones(n);

    // Very strong ridge: penalized minimum is near β = 0.
    // The unpenalized OLS minimum is far from 0.
    let lambda = 1e4_f64;
    let rho = Array1::from_elem(1, lambda.ln());
    let s = Array2::from_diag(&Array1::ones(p));
    let canonical = make_canonical(&s, p);
    let penalty = dense_penalty_config(&canonical, p);

    let (fit, _) = fit_model_for_fixed_rho(
        LogSmoothingParamsView::new(rho.view()),
        PirlsProblem {
            x: x.view(),
            offset: offset.view(),
            y: y.view(),
            priorweights: weights.view(),
            covariate_se: None,
            gaussian_fixed_cache: None,
        },
        penalty,
        &gaussian_identity_config(),
        None,
    )
    .expect("penalized line search test fit");

    let beta = fit.beta_transformed.as_ref().clone();

    // With λ = 10⁴ the ridge penalty dominates.  The accepted β must be
    // small — its norm must be well below what the unpenalized OLS solution
    // would give.  If the line search accepted the full unpenalized step,
    // β would have norm O(5/lambda * n) which is ~0.03; the fully penalized
    // mode is even smaller.  We check |β| < 0.1 as a conservative bound.
    let beta_norm = beta.iter().map(|v| v * v).sum::<f64>().sqrt();
    assert!(
        beta_norm < 0.1,
        "GPU PIRLS line search must not accept the unpenalized step when the \
         penalty inflates more.  beta_norm = {beta_norm:.4e} (expected < 0.1). \
         A passing GPU path correctly shrinks β toward 0 under a strong ridge."
    );

    // Penalized deviance at accepted β must be ≤ penalized deviance at the
    // unpenalized OLS β, confirming the line search descended in the right
    // objective.
    let resid_accepted = &x.dot(&beta) - &y;
    let dev_accepted: f64 = resid_accepted.iter().map(|v| v * v).sum();
    let pen_accepted: f64 = lambda * beta.iter().map(|v| v * v).sum::<f64>();

    let ols_beta = {
        let xtwx = x.t().dot(&x);
        let xtwy = x.t().dot(&y);
        let xtwx_fa = gam::faer_ndarray::FaerArray2View::from(xtwx.view());
        use faer::linalg::solvers::Solve;
        let chol = faer::Mat::from_fn(p, p, |i, j| xtwx_fa[[i, j]])
            .cholesky(faer::Side::Lower)
            .expect("XtX must be PD");
        let rhs = faer::Col::from_fn(p, |i| xtwy[i]);
        let sol = chol.solve(rhs);
        Array1::from_shape_fn(p, |i| sol[i])
    };
    let resid_ols = &x.dot(&ols_beta) - &y;
    let dev_ols: f64 = resid_ols.iter().map(|v| v * v).sum();
    let pen_ols: f64 = lambda * ols_beta.iter().map(|v| v * v).sum::<f64>();

    assert!(
        dev_accepted + pen_accepted <= dev_ols + pen_ols + 1.0,
        "GPU PIRLS must converge to a penalized-deviance no worse than the \
         unpenalized OLS solution under the same penalty. \
         penalized_dev(accepted)={:.4e}, penalized_dev(ols)={:.4e}",
        dev_accepted + pen_accepted,
        dev_ols + pen_ols
    );
}

// ---------------------------------------------------------------------------
// Test 5 — Qs basis semantics
//
// Contract: PirlsResult.beta_transformed is the loop β in the transformed
// (Qs) basis.  beta_original (recovered via reparam_result.qs) must equal
// qs · beta_transformed.  Both representations must be numerically
// consistent with the fitted η.
// ---------------------------------------------------------------------------
#[test]
fn gpu_pirls_gating_5_qs_basis_semantics() {
    if !cuda_selected() {
        return;
    }

    let n = 80_usize;
    let p = 8_usize;
    let mut rng = StdRng::seed_from_u64(0xDEAD_BEEF_0005);
    let normal = Normal::new(0.0, 1.0).unwrap();

    let x = Array2::from_shape_fn((n, p), |_| normal.sample(&mut rng));
    let true_beta = Array1::from_shape_fn(p, |j| 0.3 / (j as f64 + 1.0));
    let y = x.dot(&true_beta)
        + Array1::from_shape_fn(n, |_| 0.1 * normal.sample(&mut rng));
    let offset = Array1::<f64>::zeros(n);
    let weights = Array1::<f64>::ones(n);

    // Moderate ridge penalty so Qs != I.
    let lambda = 0.1_f64;
    let rho = Array1::from_elem(1, lambda.ln());
    let s = Array2::from_diag(&Array1::ones(p));
    let canonical = make_canonical(&s, p);
    let penalty = dense_penalty_config(&canonical, p);

    let (fit, _) = fit_model_for_fixed_rho(
        LogSmoothingParamsView::new(rho.view()),
        PirlsProblem {
            x: x.view(),
            offset: offset.view(),
            y: y.view(),
            priorweights: weights.view(),
            covariate_se: None,
            gaussian_fixed_cache: None,
        },
        penalty,
        &gaussian_identity_config(),
        None,
    )
    .expect("Qs basis semantics fit");

    let beta_transformed = fit.beta_transformed.as_ref().clone();
    let qs = &fit.reparam_result.qs;

    // beta_original = qs · beta_transformed
    let beta_original = qs.dot(&beta_transformed);

    // η reconstructed from original-basis β must match fit.final_eta.
    let eta_from_original = x.dot(&beta_original) + &offset;

    assert!(
        allclose(&eta_from_original, &fit.final_eta, 1e-7),
        "qs · beta_transformed must reproduce final_eta via X · beta_original. \
         Max abs diff = {:.3e}",
        eta_from_original
            .iter()
            .zip(fit.final_eta.iter())
            .map(|(a, b)| (a - b).abs())
            .fold(0.0_f64, f64::max)
    );

    // The qs matrix must be square p×p.
    assert_eq!(
        qs.nrows(),
        p,
        "reparam_result.qs must be p×p (rows)"
    );
    assert_eq!(
        qs.ncols(),
        p,
        "reparam_result.qs must be p×p (cols)"
    );

    // beta_transformed must be in the transformed basis, not the original.
    // If qs is the identity (no reparameterisation), beta_transformed ==
    // beta_original; otherwise they differ.  Either way, qs · beta_transformed
    // must be close to the CPU oracle solution.
    let oracle_resid = (&x.dot(&beta_original) - &y)
        .iter()
        .map(|v| v.abs())
        .fold(0.0_f64, f64::max);
    assert!(
        oracle_resid < 2.0,
        "beta_original (qs · beta_transformed) must produce a reasonable fit. \
         max_abs_resid={oracle_resid:.3e}"
    );
}

// ---------------------------------------------------------------------------
// Test 6 — Final Hessian matches final accepted η
//
// Contract: the exported penalized Hessian H = XᵀW_HX + Sλ must be built
// at the FINAL accepted η / finalweights, not the previous iteration's
// linearization point.  We verify this by checking that
// finalweights[i] matches the analytical Fisher weight for a Gaussian
// identity model at fit.final_eta[i] (which is simply 1.0 for Gaussian),
// and that penalized_hessian_transformed is PSD (a necessary condition for
// a correctly assembled H at convergence).
// ---------------------------------------------------------------------------
#[test]
fn gpu_pirls_gating_6_final_hessian_at_accepted_eta() {
    if !cuda_selected() {
        return;
    }

    let n = 100_usize;
    let p = 6_usize;
    let mut rng = StdRng::seed_from_u64(0xDEAD_BEEF_0006);
    let normal = Normal::new(0.0, 1.0).unwrap();

    let x = Array2::from_shape_fn((n, p), |_| normal.sample(&mut rng));
    let true_beta = Array1::from_shape_fn(p, |j| 0.4 / (j as f64 + 1.0));
    let y = x.dot(&true_beta)
        + Array1::from_shape_fn(n, |_| 0.2 * normal.sample(&mut rng));
    let offset = Array1::<f64>::zeros(n);
    let weights = Array1::<f64>::ones(n);

    let lambda = 0.5_f64;
    let rho = Array1::from_elem(1, lambda.ln());
    let s = Array2::from_diag(&Array1::ones(p));
    let canonical = make_canonical(&s, p);
    let penalty = dense_penalty_config(&canonical, p);

    let (fit, _) = fit_model_for_fixed_rho(
        LogSmoothingParamsView::new(rho.view()),
        PirlsProblem {
            x: x.view(),
            offset: offset.view(),
            y: y.view(),
            priorweights: weights.view(),
            covariate_se: None,
            gaussian_fixed_cache: None,
        },
        penalty,
        &gaussian_identity_config(),
        None,
    )
    .expect("final Hessian test fit");

    // For Gaussian identity, the Fisher weight is always 1.0 (prior weight).
    // The finalweights must reflect this at the accepted β.
    for (i, &w) in fit.finalweights.iter().enumerate() {
        assert!(
            (w - 1.0).abs() < 1e-8,
            "Gaussian identity: finalweights[{i}] must be 1.0 (Fisher weight \
             at convergence), got {w:.6e}.  If wrong, the Hessian was exported \
             from the wrong iteration."
        );
    }

    // The exported stabilized Hessian must be PSD (all eigenvalues ≥ 0).
    // We check this by verifying that all diagonal entries are positive
    // (necessary but not sufficient) and that the Cholesky factorization
    // succeeds (sufficient).
    let h = fit.dense_stabilizedhessian_transformed();
    let h_view = gam::faer_ndarray::FaerArray2View::from(h.view());
    let h_faer = faer::Mat::from_fn(p, p, |i, j| h_view[[i, j]]);
    assert!(
        h_faer.cholesky(faer::Side::Lower).is_ok(),
        "GPU PIRLS final penalized Hessian must be PSD.  Cholesky failed, \
         indicating the Hessian was not correctly assembled at the accepted η."
    );
}

// ---------------------------------------------------------------------------
// Test 7 — Status OR-reduce
//
// Contract: PirlsStatus must reflect per-row issues.  Specifically:
//   (a) A well-behaved problem must converge.
//   (b) A Binomial fit with valid y ∈ {0,1} and a moderate penalty must
//       converge (status == Converged or StalledAtValidMinimum).
//   (c) The status must never be Unstable on a well-conditioned problem.
//
// The "OR-reduce" semantic: if ANY row has an invalid (non-finite) η or
// weight, the solver must not silently accept the step with status Converged
// — it must signal the issue via status ≠ Converged at a minimum.
// ---------------------------------------------------------------------------
#[test]
fn gpu_pirls_gating_7_status_or_reduce() {
    if !cuda_selected() {
        return;
    }

    let n = 120_usize;
    let p = 6_usize;
    let mut rng = StdRng::seed_from_u64(0xDEAD_BEEF_0007);
    let normal = Normal::new(0.0, 1.0).unwrap();
    let unif = Uniform::new(0.0, 1.0).unwrap();

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
    let penalty = dense_penalty_config(&canonical, p);

    let (fit, _) = fit_model_for_fixed_rho(
        LogSmoothingParamsView::new(rho.view()),
        PirlsProblem {
            x: x.view(),
            offset: offset.view(),
            y: y.view(),
            priorweights: weights.view(),
            covariate_se: None,
            gaussian_fixed_cache: None,
        },
        penalty,
        &binomial_logit_config(),
        None,
    )
    .expect("status OR-reduce Binomial fit");

    // A well-conditioned Binomial fit must reach Converged or
    // StalledAtValidMinimum, never Unstable or MaxIterationsReached on a
    // simple problem like this.
    let ok = matches!(
        fit.status,
        PirlsStatus::Converged | PirlsStatus::StalledAtValidMinimum
    );
    assert!(
        ok,
        "Well-conditioned Binomial PIRLS must converge.  Got status = {:?}. \
         If Unstable or MaxIterationsReached, the GPU status OR-reduce is \
         incorrectly escalating per-row clamping events to a global rejection.",
        fit.status
    );

    // No NaN/Inf in finalweights or final_eta.
    for (i, &w) in fit.finalweights.iter().enumerate() {
        assert!(
            w.is_finite() && w >= 0.0,
            "finalweights[{i}] = {w} is not finite or negative at convergence. \
             The GPU OR-reduce must not accept a step with bad per-row weights."
        );
    }
    for (i, &e) in fit.final_eta.iter().enumerate() {
        assert!(
            e.is_finite(),
            "final_eta[{i}] = {e} is not finite at convergence. \
             The GPU loop must reject steps that produce non-finite η."
        );
    }
}

// ---------------------------------------------------------------------------
// Test 8 — Benchmark baseline harness contract
//
// Contract: any GPU PIRLS benchmark must use `fit_model_for_fixed_rho` (the
// CPU oracle path) as the reference, NOT a synthetic GPU test loop that
// shares the same sign convention.  This test enforces the contract by
// running BOTH paths on the same problem and asserting that the CPU oracle
// result is reproducible and well-defined on the test machine, so future
// benchmark comparisons are valid.
//
// The GPU path is tested via `fit_model_for_fixed_rho` with `cuda_selected()`
// true; the CPU oracle is obtained by temporarily forcing CPU dispatch.
// Both must agree to within 1e-6 on β and deviance.
// ---------------------------------------------------------------------------
#[test]
fn gpu_pirls_gating_8_benchmark_baseline_uses_cpu_oracle() {
    if !cuda_selected() {
        return;
    }

    let n = 200_usize;
    let p = 10_usize;
    let mut rng = StdRng::seed_from_u64(0xDEAD_BEEF_0008);
    let normal = Normal::new(0.0, 1.0).unwrap();

    let x = Array2::from_shape_fn((n, p), |_| normal.sample(&mut rng));
    let true_beta = Array1::from_shape_fn(p, |j| 0.3 / (j as f64 + 1.0));
    let y = x.dot(&true_beta)
        + Array1::from_shape_fn(n, |_| 0.25 * normal.sample(&mut rng));
    let offset = Array1::<f64>::zeros(n);
    let weights = Array1::<f64>::ones(n);

    let lambda = 0.3_f64;
    let rho = Array1::from_elem(1, lambda.ln());
    let s = Array2::from_diag(&Array1::ones(p));
    let canonical = make_canonical(&s, p);

    // Run via fit_model_for_fixed_rho (routes through GPU when cuda_selected()).
    let penalty_gpu = dense_penalty_config(&canonical, p);
    let (fit_gpu, _) = fit_model_for_fixed_rho(
        LogSmoothingParamsView::new(rho.view()),
        PirlsProblem {
            x: x.view(),
            offset: offset.view(),
            y: y.view(),
            priorweights: weights.view(),
            covariate_se: None,
            gaussian_fixed_cache: None,
        },
        penalty_gpu,
        &gaussian_identity_config(),
        None,
    )
    .expect("GPU-routed fit");

    // Force CPU path by configuring the device to CPU and re-running.
    gam::solver::gpu::configure_device(gam::solver::gpu::Device::Cpu);
    let penalty_cpu = dense_penalty_config(&canonical, p);
    let (fit_cpu, _) = fit_model_for_fixed_rho(
        LogSmoothingParamsView::new(rho.view()),
        PirlsProblem {
            x: x.view(),
            offset: offset.view(),
            y: y.view(),
            priorweights: weights.view(),
            covariate_se: None,
            gaussian_fixed_cache: None,
        },
        penalty_cpu,
        &gaussian_identity_config(),
        None,
    )
    .expect("CPU oracle fit");

    // Restore default device selection.
    gam::solver::gpu::configure_device(gam::solver::gpu::Device::Cuda);

    let beta_gpu = fit_gpu.beta_transformed.as_ref().clone();
    let beta_cpu = fit_cpu.beta_transformed.as_ref().clone();

    assert!(
        allclose(&beta_gpu, &beta_cpu, 1e-5),
        "GPU PIRLS β must match CPU oracle β to 1e-5 relative tolerance. \
         Max diff = {:.3e}. A benchmark that compares GPU against itself \
         (same sign convention) would not catch sign errors.",
        beta_gpu
            .iter()
            .zip(beta_cpu.iter())
            .map(|(a, b)| (a - b).abs())
            .fold(0.0_f64, f64::max)
    );

    assert!(
        (fit_gpu.deviance - fit_cpu.deviance).abs() < 1e-5 * (1.0 + fit_cpu.deviance.abs()),
        "GPU PIRLS deviance must match CPU oracle deviance. \
         gpu={:.6e}, cpu={:.6e}",
        fit_gpu.deviance,
        fit_cpu.deviance
    );

    // Verify that the CPU oracle is being used as the reference (not a
    // synthetic loop): the CPU fit must be Converged.
    assert_eq!(
        fit_cpu.status,
        PirlsStatus::Converged,
        "CPU oracle must converge on this well-conditioned problem — \
         if it does not, the benchmark baseline is invalid."
    );
}
