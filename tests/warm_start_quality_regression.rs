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
        initial_lm_lambda: None,
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
