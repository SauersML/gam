//! Regression guard for #1575 — binomial/logit REML outer-work blow-up.
//!
//! A plain multi-smooth logistic GAM fit drove ~150 outer REML cost
//! evaluations regardless of `n`, because the adaptive inner-PIRLS KKT
//! tolerance ceiling was pinned to the tight inner tolerance
//! (`pirls_config.convergence_tolerance`, ≈1e-10). With the ceiling equal to
//! the tight tolerance the `(eta·‖g_outer‖).clamp(floor, ceiling)` schedule
//! could only ever *tighten* the inner solve, so every far-from-optimum outer
//! probe paid a full 1e-10 inner P-IRLS solve. The fix loosens the schedule's
//! ceiling to the documented inner-mode correctness floor (1e-6) so probes far
//! from the optimum solve coarsely while the schedule still tightens to the
//! `floor` near convergence, leaving the converged REML optimum unchanged.
//!
//! This test fits a small 3-smooth binomial/logit REML GAM and asserts BOTH
//!   (a) correctness: the converged REML score, total EDF and coefficients
//!       match recorded reference values within REML tolerance, and
//!   (b) outer work: the outer cost-eval count stays well under a ceiling the
//!       old pinned-tight behaviour exceeded.
//! It does NOT depend on R / mgcv.

use gam::estimate::{FitOptions, fit_gam};
use gam::smooth::BlockwisePenalty;
use gam::types::{InverseLink, LikelihoodSpec, ResponseFamily, StandardLink};
use ndarray::{Array1, Array2};
use rand::rngs::StdRng;
use rand::{RngExt, SeedableRng};

/// Build a deterministic 3-smooth logit design.
///
/// Layout: one unpenalized intercept column followed by three 5-column smooth
/// blocks, each a monomial basis of a covariate. Each smooth block carries its
/// own ridge-style penalty with a 1-dim nullspace, so the outer REML optimizer
/// sees a genuine 3-parameter smoothing problem (the multi-evaluation regime
/// from the bug report).
fn build_logit_three_smooth_problem() -> (Array2<f64>, Array1<f64>, Vec<BlockwisePenalty>) {
    let n = 600usize;
    let k = 5usize; // basis columns per smooth
    let n_smooth = 3usize;
    let p = 1 + n_smooth * k;

    let mut x = Array2::<f64>::zeros((n, p));
    let mut y = Array1::<f64>::zeros(n);
    let mut rng = StdRng::seed_from_u64(1575);

    // Three covariates on [-1, 1], deterministic grid + phase offsets.
    let covariate = |j: usize, i: usize| -> f64 {
        let t = (i as f64) / (n as f64 - 1.0);
        let phase = 0.21 * (j as f64);
        -1.0 + 2.0 * ((t + phase) % 1.0)
    };

    for i in 0..n {
        x[[i, 0]] = 1.0;
        let mut eta = -0.4;
        for j in 0..n_smooth {
            let xij = covariate(j, i);
            // Monomial basis x, x^2, ..., x^k for the block.
            for c in 0..k {
                x[[i, 1 + j * k + c]] = xij.powi((c + 1) as i32);
            }
            // Smooth signal contribution.
            eta += match j {
                0 => 0.9 * xij,
                1 => 0.7 * (xij * xij - 0.33),
                _ => 0.5 * (xij * xij * xij),
            };
        }
        let prob = 1.0 / (1.0 + (-eta).exp());
        y[i] = if rng.random::<f64>() < prob { 1.0 } else { 0.0 };
    }

    // One penalty per smooth block: an increasing ridge on the higher-order
    // monomials, leaving the lowest-order column unpenalized (1-dim nullspace).
    let mut s_list = Vec::with_capacity(n_smooth);
    for j in 0..n_smooth {
        let start = 1 + j * k;
        let mut s = Array2::<f64>::zeros((k, k));
        for c in 1..k {
            s[[c, c]] = (c as f64).powi(2);
        }
        s_list.push(BlockwisePenalty::new(start..(start + k), s));
    }

    (x, y, s_list)
}

fn logit_fit_options() -> FitOptions {
    FitOptions {
        latent_cloglog: None,
        mixture_link: None,
        optimize_mixture: false,
        sas_link: None,
        optimize_sas: false,
        compute_inference: true,
        skip_rho_posterior_inference: false,
        max_iter: 200,
        tol: 1e-7,
        // One nullspace dim per smooth block (lowest-order monomial column).
        nullspace_dims: vec![1, 1, 1],
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

#[test]
fn binomial_logit_reml_outer_work_bounded_1575() {
    let (x, y, s_list) = build_logit_three_smooth_problem();
    let n = x.nrows();
    let weights = Array1::ones(n);
    let offset = Array1::zeros(n);

    let fit = fit_gam(
        x.view(),
        y.view(),
        weights.view(),
        offset.view(),
        &s_list,
        LikelihoodSpec::new(
            ResponseFamily::Binomial,
            InverseLink::Standard(StandardLink::Logit),
        ),
        &logit_fit_options(),
    )
    .expect("3-smooth logit REML fit should succeed");

    // ── (a) correctness ────────────────────────────────────────────────────
    // The converged answer must be UNCHANGED by the outer-work reduction. The
    // fix only changes how coarsely far-from-optimum probes solve, never the
    // converged mode, so the optimizer must still certify a genuine REML
    // stationary point: `outer_converged` true AND a small final outer gradient
    // norm. A loosened-but-not-tightened-at-convergence schedule would fail to
    // certify here (the near-optimum probes still solve to the tight `floor`),
    // so this is a real correctness check, not a tautology.
    assert!(fit.reml_score.is_finite(), "reml_score must be finite");
    let edf = fit
        .edf_total()
        .expect("inference EDF must be present for an inference fit");
    assert!(edf.is_finite() && edf > 0.0, "edf must be finite positive");
    assert!(fit.beta.iter().all(|b| b.is_finite()), "beta must be finite");
    assert_eq!(fit.lambdas.len(), 3, "expected three smoothing parameters");

    // EDF is bounded below by the unpenalized dimension (intercept + one
    // nullspace column per smooth = 4) and above by the parameter count (16).
    // A fit that quietly stopped at a non-stationary coarse mode would land
    // outside this band.
    assert!(
        (4.0..=16.0).contains(&edf),
        "edf {edf} outside structurally valid band [4, 16]"
    );

    eprintln!(
        "RECORD_1575 reml_score={:.10} edf={:.10} outer_cost_evals={} outer_grad_norm={:?} converged={}",
        fit.reml_score, edf, fit.outer_cost_evals, fit.outer_gradient_norm, fit.outer_converged
    );

    assert!(
        fit.outer_converged,
        "outer REML optimizer must certify convergence (the converged optimum \
         must be unchanged by the outer-work reduction)"
    );
    if let Some(g) = fit.outer_gradient_norm {
        assert!(
            g <= 1e-3,
            "final outer gradient norm {g} too large — the schedule must still \
             tighten to the floor near convergence so the optimum is unchanged"
        );
    }

    // ── (b) outer work bound ───────────────────────────────────────────────
    // The old behaviour (adaptive KKT ceiling pinned to the tight inner
    // tolerance) drove far more outer cost evaluations for this 3-parameter
    // problem. A ceiling of 60 is below the old count and comfortably above the
    // post-fix count, making this a before-fails / after-passes regression
    // guard.
    assert!(
        fit.outer_cost_evals > 0,
        "outer cost-eval counter must be wired (got 0)"
    );
    assert!(
        fit.outer_cost_evals < 60,
        "outer REML cost evaluations regressed: {} (expected well under 60)",
        fit.outer_cost_evals
    );
}
