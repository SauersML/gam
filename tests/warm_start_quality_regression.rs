//! Quality-regression guard for the warm-start machinery.
//!
//! Inner-PIRLS converges (in exact arithmetic) to the unique KKT point of
//! `min_β  D(β) + (1/2) Σ_k λ_k β' S_k β` regardless of the seed β. So a
//! warm-started fit at ρ must produce the SAME β as a cold-started fit at
//! the same ρ — modulo PIRLS convergence tolerance. This file pins that
//! invariant down with two integration tests:
//!
//! 1. `warm_start_with_nearby_rho_seed_converges_to_cold_beta` — seeds the
//!    new fit with β from a NEARBY ρ. The IFT predictor / tangent-line
//!    machinery in production tests this same path on every accepted
//!    outer iteration; here we confirm the cold-vs-warm ‖Δβ‖ is at
//!    PIRLS convergence-tolerance level.
//! 2. `warm_start_with_far_rho_seed_still_converges_to_cold_beta` — seeds
//!    with β from a far-away ρ. This exercises the case where the warm
//!    start is a poor predictor; PIRLS must still recover the true KKT
//!    β within tolerance, just with more inner iters.
//!
//! Failure of either test means a bandaid (or the warm-start machinery
//! itself) is causing inner-PIRLS to terminate at a non-KKT β —
//! exactly the kind of quality regression the mission forbids. These
//! tests run cheap (n=400, p=12) and gate every PR.

use gam::construction::CanonicalPenalty;
use gam::estimate::PenaltySpec;
use gam::pirls::{
    PenaltyConfig, PirlsConfig, PirlsProblem, PirlsStatus, fit_model_for_fixed_rho,
};
use gam::types::{
    Coefficients, GlmLikelihoodFamily, GlmLikelihoodSpec, InverseLink, LinkFunction,
    LogSmoothingParamsView,
};
use ndarray::{Array1, Array2, array};
use rand::rngs::StdRng;
use rand::{RngExt, SeedableRng};

const N: usize = 400;
const P: usize = 12;
const SEED: u64 = 0xDEAD_BEEF_5EED_5EED;

fn make_problem() -> (Array2<f64>, Array1<f64>, Array1<f64>, Vec<CanonicalPenalty>) {
    let mut rng = StdRng::seed_from_u64(SEED);
    let mut x = Array2::<f64>::zeros((N, P));
    for i in 0..N {
        x[[i, 0]] = 1.0; // intercept
        for j in 1..P {
            x[[i, j]] = rng.random_range(-1.0..1.0);
        }
    }
    let beta_true = Array1::from_shape_fn(P, |j| if j == 0 { -0.2 } else { 0.4 / j as f64 });
    let eta = x.dot(&beta_true);
    let y = eta.mapv(|e| {
        let prob = 1.0 / (1.0 + (-e).exp());
        if rng.random::<f64>() < prob { 1.0 } else { 0.0 }
    });
    let w = Array1::<f64>::ones(N);

    // Second-difference-style penalty on coefficients j>=1 (ridge-like
    // with mild structure). This is a single-block penalty so ρ is
    // 1-dimensional, simplifying the test.
    let mut s = Array2::<f64>::zeros((P, P));
    for j in 1..P {
        s[[j, j]] = 1.0;
    }
    let canonical = canonicalize(&[s]);
    (x, y, w, canonical)
}

fn canonicalize(s_list: &[Array2<f64>]) -> Vec<CanonicalPenalty> {
    let p = s_list[0].nrows();
    s_list
        .iter()
        .enumerate()
        .filter_map(|(idx, s)| {
            gam::construction::canonicalize_penalty_spec(
                &PenaltySpec::Dense(s.clone()),
                p,
                idx,
                "warm_start_quality_regression",
            )
            .expect("canonicalize")
        })
        .collect()
}

fn fit_at_rho(
    x: &Array2<f64>,
    y: &Array1<f64>,
    w: &Array1<f64>,
    penalties: &[CanonicalPenalty],
    rho: f64,
    warm: Option<&Coefficients>,
) -> (Coefficients, usize, PirlsStatus) {
    let (beta, iters, status, _final_lambda) =
        fit_at_rho_full(x, y, w, penalties, rho, warm, None);
    (beta, iters, status)
}

/// Parametric variant exposing both the LM-λ warm-start hint and the
/// converged final λ. Used by tests that exercise the
/// `initial_lm_lambda` plumbing wired in commit `ba4dc931`.
fn fit_at_rho_full(
    x: &Array2<f64>,
    y: &Array1<f64>,
    w: &Array1<f64>,
    penalties: &[CanonicalPenalty],
    rho: f64,
    warm: Option<&Coefficients>,
    initial_lm_lambda: Option<f64>,
) -> (Coefficients, usize, PirlsStatus, f64) {
    let p = x.ncols();
    let offset = Array1::<f64>::zeros(y.len());
    let cfg = PirlsConfig {
        likelihood: GlmLikelihoodSpec::canonical(GlmLikelihoodFamily::BinomialLogit),
        link_kind: InverseLink::Standard(LinkFunction::Logit),
        max_iterations: 200,
        // Tight enough that any drift between cold/warm shows up at the
        // 1e-5 relative-β level we assert below; loose enough that a
        // 200-iter cap is comfortable.
        convergence_tolerance: 1e-10,
        firth_bias_reduction: false,
        initial_lm_lambda,
    };
    let (result, _working) = fit_model_for_fixed_rho(
        LogSmoothingParamsView::new(array![rho].view()),
        PirlsProblem {
            x: x.view(),
            offset: offset.view(),
            y: y.view(),
            priorweights: w.view(),
            covariate_se: None,
        },
        PenaltyConfig {
            canonical_penalties: penalties,
            balanced_penalty_root: None,
            reparam_invariant: None,
            p,
            coefficient_lower_bounds: None,
            linear_constraints_original: None,
            penalty_shrinkage_floor: None,
            kronecker_factored: None,
        },
        &cfg,
        warm,
    )
    .expect("fit_model_for_fixed_rho");
    // Convert back to the ORIGINAL basis so that warm-vs-cold β are
    // directly comparable irrespective of which reparameterization
    // basis the solver elected to use.
    let beta_original = match result.coordinate_frame {
        gam::pirls::PirlsCoordinateFrame::OriginalSparseNative => {
            result.beta_transformed.as_ref().clone()
        }
        gam::pirls::PirlsCoordinateFrame::TransformedQs => {
            result.reparam_result.qs.dot(result.beta_transformed.as_ref())
        }
    };
    (
        Coefficients::new(beta_original),
        result.iteration,
        result.status,
        result.final_lm_lambda,
    )
}

fn relative_l2(a: &Coefficients, b: &Coefficients) -> f64 {
    let lhs = a.as_ref();
    let rhs = b.as_ref();
    let diff: f64 = lhs
        .iter()
        .zip(rhs.iter())
        .map(|(p, q)| (p - q) * (p - q))
        .sum::<f64>()
        .sqrt();
    let denom = rhs.iter().map(|v| v * v).sum::<f64>().sqrt().max(1.0);
    diff / denom
}

#[test]
fn warm_start_with_nearby_rho_seed_converges_to_cold_beta() {
    let (x, y, w, penalties) = make_problem();
    // Anchor fit at ρ_a, then probe at ρ_b that is one Newton-friendly
    // step away (Δρ = 0.5, which corresponds to λ_b ≈ 1.65 · λ_a).
    let rho_a = 0.0_f64;
    let rho_b = 0.5_f64;

    let (beta_a, _iter_a, status_a) = fit_at_rho(&x, &y, &w, &penalties, rho_a, None);
    assert!(
        matches!(
            status_a,
            PirlsStatus::Converged | PirlsStatus::StalledAtValidMinimum
        ),
        "cold fit at rho_a must converge, got {:?}",
        status_a
    );

    let (beta_b_cold, iter_b_cold, status_b_cold) =
        fit_at_rho(&x, &y, &w, &penalties, rho_b, None);
    let (beta_b_warm, iter_b_warm, status_b_warm) =
        fit_at_rho(&x, &y, &w, &penalties, rho_b, Some(&beta_a));
    assert!(matches!(
        status_b_cold,
        PirlsStatus::Converged | PirlsStatus::StalledAtValidMinimum
    ));
    assert!(matches!(
        status_b_warm,
        PirlsStatus::Converged | PirlsStatus::StalledAtValidMinimum
    ));

    let drift = relative_l2(&beta_b_warm, &beta_b_cold);
    eprintln!(
        "[warm-start regression] nearby Δρ=0.5: cold_iters={} warm_iters={} relative_drift={:.3e}",
        iter_b_cold, iter_b_warm, drift,
    );
    assert!(
        drift < 1e-5,
        "warm-start fit drifted from cold-start fit beyond convergence tolerance: {:.3e}",
        drift
    );
}

#[test]
fn warm_start_with_far_rho_seed_still_converges_to_cold_beta() {
    let (x, y, w, penalties) = make_problem();
    // Far jump: Δρ = 6.0 → λ_b ≈ 403 · λ_a. The seed β_a is a poor
    // initial guess, but the inner PIRLS must still find the
    // unique KKT β at ρ_b.
    let rho_a = -3.0_f64;
    let rho_b = 3.0_f64;

    let (beta_a, _iter_a, status_a) = fit_at_rho(&x, &y, &w, &penalties, rho_a, None);
    assert!(matches!(
        status_a,
        PirlsStatus::Converged | PirlsStatus::StalledAtValidMinimum
    ));

    let (beta_b_cold, iter_b_cold, status_b_cold) =
        fit_at_rho(&x, &y, &w, &penalties, rho_b, None);
    let (beta_b_warm, iter_b_warm, status_b_warm) =
        fit_at_rho(&x, &y, &w, &penalties, rho_b, Some(&beta_a));
    assert!(matches!(
        status_b_cold,
        PirlsStatus::Converged | PirlsStatus::StalledAtValidMinimum
    ));
    assert!(matches!(
        status_b_warm,
        PirlsStatus::Converged | PirlsStatus::StalledAtValidMinimum
    ));

    let drift = relative_l2(&beta_b_warm, &beta_b_cold);
    eprintln!(
        "[warm-start regression] far Δρ=6.0: cold_iters={} warm_iters={} relative_drift={:.3e}",
        iter_b_cold, iter_b_warm, drift,
    );
    assert!(
        drift < 1e-5,
        "warm-start fit (far seed) drifted from cold-start fit beyond convergence tolerance: {:.3e}",
        drift
    );
}

#[test]
fn pirls_result_exposes_finite_positive_final_lm_lambda() {
    // Cold fit must populate `final_lm_lambda` with a finite, positive
    // value so the REML runtime can persist it as the next-call hint.
    // Guards against regressions where the field is left at NaN or 0.
    let (x, y, w, penalties) = make_problem();
    let (_beta, iter, status, final_lambda) =
        fit_at_rho_full(&x, &y, &w, &penalties, 0.0, None, None);
    assert!(matches!(
        status,
        PirlsStatus::Converged | PirlsStatus::StalledAtValidMinimum
    ));
    eprintln!(
        "[lm-lambda regression] cold fit converged in {} iters, final_lm_lambda={:.3e}",
        iter, final_lambda
    );
    assert!(
        final_lambda.is_finite() && final_lambda > 0.0,
        "PirlsResult::final_lm_lambda must be finite and positive, got {:.3e}",
        final_lambda
    );
    // Sanity: the converged damping cannot have escaped the LM ceiling
    // (1e12) — that would mean the inner solver halted on a non-SPD
    // step, which contradicts the Converged / StalledAtValidMinimum
    // statuses asserted above.
    assert!(
        final_lambda < 1e12,
        "PirlsResult::final_lm_lambda escaped LM ceiling at converged state: {:.3e}",
        final_lambda
    );
}

#[test]
fn lm_lambda_warm_start_hint_preserves_kkt_beta_and_does_not_increase_iters() {
    // End-to-end test of the LM-λ persistence path (commit ba4dc931):
    // pull `final_lm_lambda` out of one fit, pass it as the
    // `initial_lm_lambda` hint of the next, and verify
    //   1. β converges to the same KKT point as the cold path,
    //   2. iter count is non-increasing (a hint that's clamped to the
    //      same value as the cold default at minimum produces equal
    //      iters; an informative hint produces fewer).
    let (x, y, w, penalties) = make_problem();
    let rho_a = 0.0_f64;
    let rho_b = 1.0_f64;

    // Anchor solve at ρ_a — captures the converged λ.
    let (_beta_a, _iter_a, status_a, lambda_a) =
        fit_at_rho_full(&x, &y, &w, &penalties, rho_a, None, None);
    assert!(matches!(
        status_a,
        PirlsStatus::Converged | PirlsStatus::StalledAtValidMinimum
    ));
    assert!(
        lambda_a.is_finite() && lambda_a > 0.0,
        "anchor fit must expose a usable lambda hint; got {:.3e}",
        lambda_a
    );

    // Cold solve at ρ_b: no λ hint, no β warm-start.
    let (beta_b_cold, iter_b_cold, status_b_cold, _lambda_b_cold) =
        fit_at_rho_full(&x, &y, &w, &penalties, rho_b, None, None);
    assert!(matches!(
        status_b_cold,
        PirlsStatus::Converged | PirlsStatus::StalledAtValidMinimum
    ));

    // Hinted solve at ρ_b: λ hint plumbed through PirlsConfig, no β
    // warm-start (so the hint's effect is isolated).
    let (beta_b_hinted, iter_b_hinted, status_b_hinted, _lambda_b_hinted) =
        fit_at_rho_full(&x, &y, &w, &penalties, rho_b, None, Some(lambda_a));
    assert!(matches!(
        status_b_hinted,
        PirlsStatus::Converged | PirlsStatus::StalledAtValidMinimum
    ));

    let drift = relative_l2(&beta_b_hinted, &beta_b_cold);
    eprintln!(
        "[lm-lambda regression] cold_iters={} hinted_iters={} (lambda_hint={:.3e}) drift={:.3e}",
        iter_b_cold, iter_b_hinted, lambda_a, drift,
    );
    assert!(
        drift < 1e-5,
        "lm-lambda hint caused beta drift beyond convergence tolerance: {:.3e}",
        drift
    );
    // Non-regression: the hint must not slow the solve down. When the
    // converged λ from ρ_a is at the cold-default floor the runtime
    // clamps it to 1e-6, so iter counts will match. When the geometry
    // genuinely needed damping the hint accelerates the next solve;
    // either way iter_b_hinted ≤ iter_b_cold.
    assert!(
        iter_b_hinted <= iter_b_cold,
        "lm-lambda hint regressed iter count: cold={} hinted={}",
        iter_b_cold,
        iter_b_hinted
    );
}
