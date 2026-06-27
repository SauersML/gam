//! Regression guard for #1575 — the post-fit PSIS rho-uncertainty diagnostic
//! must stay OPT-IN (default off) so it does not silently re-impose ~33 surplus
//! full-n inner P-IRLS solves on every fit.
//!
//! ## What this pins
//!
//! A well-conditioned 3-smooth binomial/logit REML GAM is solved by the ARC
//! outer Newton in a handful of textbook-quadratic steps. Before #1575 the fit
//! then ran `compute_rho_uncertainty_diagnostic` unconditionally, evaluating the
//! exact profiled criterion at `sample_count + 1 = 33` Laplace-proposal draws
//! (each a full inner solve) to produce a `k_hat` heavy-tail flag that nothing
//! in the fit path consumes. That inflated the post-convergence outer work from
//! a couple dozen solves to ~57 — a large share of the binomial/Poisson REML
//! "~100× slower than mgcv" gap.
//!
//! With the diagnostic gated opt-in (default off) the SAME fit converges to the
//! BYTE-IDENTICAL optimum while doing far fewer inner solves. This test asserts:
//!   (a) the fit converges to a genuine stationary point (correctness gate), and
//!   (b) `outer_cost_evals` stays well below the diagnostic-on regime — a
//!       regression that re-enabled the 33-sample probe by default would push it
//!       back up and trip this bound.
//!
//! It lives in `gam-solve` (not the root `gam` crate) because the fit/optimizer
//! surface under test is here, and it needs no R / mgcv.

use gam_problem::{InverseLink, LikelihoodSpec, ResponseFamily, StandardLink};
use gam_solve::estimate::{FitOptions, fit_gam};
use gam_terms::smooth::BlockwisePenalty;
use ndarray::{Array1, Array2};
use rand::rngs::StdRng;
use rand::{RngExt, SeedableRng};

/// Build a deterministic, well-conditioned 3-smooth logit design: one
/// unpenalized intercept, then three smooth blocks, each a linear (null-space)
/// column plus four orthogonal-ish Fourier wiggle columns (penalized). The
/// near-orthogonal basis keeps the REML surface sharply identified, so the
/// outer ARC Newton converges cleanly — isolating the diagnostic overhead from
/// any flat-valley / multistart escalation.
fn build_logit_three_smooth_problem() -> (Array2<f64>, Array1<f64>, Vec<BlockwisePenalty>) {
    let n = 600usize;
    let k = 5usize; // columns per smooth block
    let n_smooth = 3usize;
    let p = 1 + n_smooth * k;

    let mut x = Array2::<f64>::zeros((n, p));
    let mut y = Array1::<f64>::zeros(n);
    let mut rng = StdRng::seed_from_u64(1575);

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
            // Column 0: linear null-space term. Columns 1..: Fourier wiggle.
            x[[i, 1 + j * k]] = xij;
            for c in 1..k {
                let freq = ((c + 1) / 2) as f64;
                let arg = std::f64::consts::PI * freq * xij;
                x[[i, 1 + j * k + c]] = if c % 2 == 1 { arg.sin() } else { arg.cos() };
            }
            eta += match j {
                0 => 0.9 * (std::f64::consts::PI * xij).sin(),
                1 => 0.7 * (2.0 * xij * xij - 1.0),
                _ => 0.5 * xij,
            };
        }
        let prob = 1.0 / (1.0 + (-eta).exp());
        y[i] = if rng.random::<f64>() < prob { 1.0 } else { 0.0 };
    }

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
        // Full inference (covariance, EDF) requested — the realistic gamfit.fit
        // path. The rho-uncertainty diagnostic is NOT inference and must stay off
        // regardless of this flag.
        compute_inference: true,
        skip_rho_posterior_inference: false,
        max_iter: 200,
        tol: 1e-7,
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
fn binomial_logit_reml_skips_unused_rho_uncertainty_diagnostic_1575() {
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
    .expect("well-conditioned 3-smooth logit REML fit should succeed");

    // (a) Correctness: a genuine, certified stationary optimum.
    assert!(fit.reml_score.is_finite(), "reml_score must be finite");
    assert!(
        fit.beta.iter().all(|b| b.is_finite()),
        "beta must be finite"
    );
    assert!(
        fit.outer_converged,
        "outer REML optimizer must certify convergence"
    );
    if let Some(g) = fit.outer_gradient_norm {
        assert!(
            g <= 1e-3 * (1.0 + fit.reml_score.abs()),
            "outer gradient {g} must clear the score-relative stationarity bound \
             (score={})",
            fit.reml_score
        );
    }
    let edf = fit
        .edf_total()
        .expect("inference EDF must be present for an inference fit");
    assert!(
        (4.0..=16.0).contains(&edf),
        "edf {edf} outside the structurally valid band [4, 16]"
    );

    // (b) The opt-in gate holds: the unused 33-sample PSIS rho-uncertainty probe
    // is NOT run, so the post-convergence outer work stays small. With the probe
    // on (the pre-#1575 default) this same fit spent ~57 `compute_cost` solves;
    // with it off it spends ~25. A bound of 40 sits firmly between the two
    // regimes: it passes with the diagnostic off and trips if a regression
    // re-enables the per-fit 33-sample probe by default.
    assert!(
        fit.outer_cost_evals > 0,
        "outer cost-eval counter must be wired (got 0)"
    );
    assert!(
        fit.outer_cost_evals < 40,
        "outer cost evals {} regressed toward the diagnostic-on regime (~57): the \
         unused PSIS rho-uncertainty probe appears to be running by default again \
         (#1575)",
        fit.outer_cost_evals
    );
}
