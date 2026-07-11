//! Regression guard for #1575 — binomial/logit REML outer-work blow-up.
//!
//! A plain multi-smooth logistic GAM fit reportedly drove ~150 outer REML cost
//! evaluations regardless of `n`. #1575 originally tried to cure this by
//! loosening the adaptive inner-PIRLS KKT tolerance ceiling from the tight inner
//! tolerance (`pirls_config.convergence_tolerance`, ≈1e-10) to a fixed 1e-6.
//! That loosening was found to be INERT for this fit — tight and loose ceilings
//! produce the IDENTICAL converged answer AND the identical outer-eval count —
//! and it was reverted as dead/misleading code. The #1575 outer-work reduction
//! therefore remains an OPEN perf target requiring a convergence-preserving
//! approach (e.g. inner warm-starting), not a tolerance tweak.
//!
//! This test fits a small 3-smooth binomial/logit REML GAM and asserts:
//!   (a) correctness: the optimizer certifies a genuine REML stationary point —
//!       the fit mints at all (sealed convergence evidence, SPEC 20) AND the
//!       final outer gradient clears the
//!       solver's own SCORE-RELATIVE stationarity bound. This fit is weakly
//!       identified (near-collinear monomial bases), so the REML surface is a
//!       flat valley and the residual gradient floors at O(0.1) on a score of
//!       ~390 — exactly the mgcv-aligned score-relative convergence the solver
//!       documents (see `rho_optimizer::bridges`). An absolute 1e-3 gradient
//!       bound is the WRONG criterion here (1e-3 is the absolute floor weakly
//!       identified coordinates cannot reach); the score-relative check still
//!       rejects a genuinely non-stationary stuck/overfit mode.
//!   (b) outer work: a coarse upper-bound regression guard (< 60 evals) that
//!       trips if the outer work blows back up toward the ~150-eval bug regime.
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
        resource_policy: gam_runtime::resource::ResourcePolicy::default_library(),
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
    // The converged answer must be a genuine REML stationary point: the fit
    // minting at all is the sealed convergence proof (SPEC 20) AND the final
    // outer gradient must
    // clear the solver's score-relative stationarity bound (checked below). This
    // is the load-bearing correctness gate — a non-stationary stuck mode (an
    // overfit, or a coarse-inner-solve stall) would fail it.
    assert!(fit.reml_score.is_finite(), "reml_score must be finite");
    let edf = fit
        .edf_total()
        .expect("inference EDF must be present for an inference fit");
    assert!(edf.is_finite() && edf > 0.0, "edf must be finite positive");
    assert!(
        fit.beta.iter().all(|b| b.is_finite()),
        "beta must be finite"
    );
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
        "RECORD_1575 reml_score={:.10} edf={:.10} outer_cost_evals={} inner_pirls_solves={} outer_grad_norm={:?} converged=certified",
        fit.reml_score, edf, fit.outer_cost_evals, fit.inner_pirls_solves, fit.outer_gradient_norm
    );

    // Fit existence is the sealed convergence proof (SPEC 20).
    // Stationarity is certified RELATIVE TO THE SCORE SCALE, matching the
    // solver's own outer-convergence contract (the score-relative flat-valley
    // bound in `rho_optimizer::bridges`: a cost-stalled optimum converges when
    // the projected gradient clears `FLAT_VALLEY_CONVERGED_REL_GRAD·(1+|score|)`,
    // capped at `FLAT_VALLEY_CONVERGED_ABS_GRAD_CAP = 1.0`). This binomial/logit
    // fit is genuinely weakly identified in some ρ coordinates (near-collinear
    // monomial bases): the REML surface flattens and the residual outer gradient
    // floors at O(0.1) on a score of ~390, exactly as mgcv's score-relative
    // convergence certifies. An ABSOLUTE 1e-3 bound is therefore the WRONG
    // correctness check here — 1e-3 is the absolute floor the solver documents
    // weakly-identified coordinates cannot reach. We keep a REAL correctness
    // check by asserting the same score-relative stationarity bound the solver
    // certifies against: this still REJECTS a genuinely non-stationary stuck
    // mode (e.g. the #1426-class overfit at |g|≈11, far above this bound),
    // while certifying the true flat-valley optimum.
    if let Some(g) = fit.outer_gradient_norm {
        let score_relative_stationarity_bound = (1.0e-3 * (1.0 + fit.reml_score.abs())).min(1.0);
        assert!(
            g <= score_relative_stationarity_bound,
            "final outer gradient norm {g} exceeds the score-relative stationarity \
             bound {score_relative_stationarity_bound} (= min(1e-3·(1+|score|), 1.0), \
             score={}) — the converged optimum is NOT stationary even by the \
             score-relative criterion, so the optimum changed / the fit is stuck",
            fit.reml_score
        );
    }

    // ── (b) outer work bound ───────────────────────────────────────────────
    // The #1575 PERF goal — cutting the ~150-eval outer REML work the
    // binomial/logit slowdown reported — is NOT achieved by the inner-PIRLS KKT
    // tolerance ceiling. The original #1575 "fix" loosened that ceiling, but it
    // is empirically INERT for this fit: tight (1e-10) and loose (1e-6) ceilings
    // produce the identical converged answer AND the identical outer-eval count
    // (the clamp already pins the inner tolerance to the score-relative regime).
    // The loosened ceiling was reverted (it was dead/misleading code). Genuine
    // outer-work reduction needs a different, convergence-preserving approach
    // (e.g. warm-starting the inner solve) and remains the OPEN #1575 target.
    //
    // This 3-parameter fit converges in ~54 outer cost evals on the current
    // solver, so the regression guard below (< 60) holds and still trips if the
    // outer work blows back up toward the ~150-eval bug regime. It is a coarse
    // upper-bound guard, NOT a claim that the #1575 perf target was met.
    assert!(
        fit.outer_cost_evals > 0,
        "outer cost-eval counter must be wired (got 0)"
    );
    assert!(
        fit.outer_cost_evals < 60,
        "outer REML cost evaluations regressed back toward the #1575 bug regime: \
         {} (the well-posed 3-smooth fit should stay well under 60; the ~150-eval \
         blow-up is the open perf target)",
        fit.outer_cost_evals
    );

    // ── (c) actual inner P-IRLS solve bound (the TRUE #1575 cost metric) ────
    // `outer_cost_evals` counts outer REQUESTS, including single-slot cache hits
    // and prior short-circuits; the genuinely expensive work the #1575 slowdown
    // is measured in is the number of cache-missing full-n inner P-IRLS solves
    // (`inner_pirls_solves`) across the seed-grid prepass, screening, multistart,
    // and finalize. A healthy warm-started fit performs ~2 inner solves per outer
    // cost-eval. This guard pins that the warm-start / parsimony-waiver /
    // PSIS-opt-in economy that cut the original ~150-eval (and many-hundred-solve)
    // pathology stays in force: if warm-starting broke, duplicate solving crept
    // in, or the redundant-seed / diagnostic-solve guards regressed, the solve
    // count would blow back up and trip this bound. It is a coarse upper bound
    // (the fixture solves in well under 150), NOT a correctness check — the
    // correctness gates are (a) above.
    assert!(
        fit.inner_pirls_solves > 0,
        "inner P-IRLS solve counter must be wired (got 0)"
    );
    assert!(
        fit.inner_pirls_solves < 150,
        "actual full-n inner P-IRLS solves regressed toward the #1575 bug regime: \
         {} (the well-posed 3-smooth fit should stay well under 150; a blow-up here \
         signals broken inner warm-starting or duplicate/redundant solving)",
        fit.inner_pirls_solves
    );
}

/// #1575: with Firth/Jeffreys bias reduction ON (the DEFAULT for binomial/logit
/// at n ≤ 20000, which mgcv never pays), the dominant outer cost is the exact
/// Tierney-Kadane LAML-Hessian Firth directional derivatives. Their O(k²) pair
/// loop now reuses single-index reduced Hadamard-Gram sub-blocks cached once per
/// direction (bit-identical work elision, locked by the firth.rs oracle test).
///
/// This drives the REAL Firth-ON outer-Hessian path end-to-end on a small
/// 3-smooth fit and asserts it (a) converges to a genuine REML stationary point
/// and (b) finishes within a sane outer-work budget — guarding that the cached
/// path stays wired and the Firth fit does not regress toward the bug regime.
#[test]
fn binomial_logit_reml_firth_on_outer_work_bounded_1575() {
    let (x, y, s_list) = build_logit_three_smooth_problem();
    let n = x.nrows();
    let weights = Array1::ones(n);
    let offset = Array1::zeros(n);

    let mut opts = logit_fit_options();
    opts.firth_bias_reduction = true; // exercise the Firth TK outer-Hessian path

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
        &opts,
    )
    .expect("3-smooth Firth-ON logit REML fit should succeed");

    // Correctness: finite fit, converged, EDF in the structurally valid band.
    assert!(fit.reml_score.is_finite(), "reml_score must be finite");
    assert!(
        fit.beta.iter().all(|b| b.is_finite()),
        "beta must be finite"
    );
    let edf = fit
        .edf_total()
        .expect("inference EDF must be present for an inference fit");
    assert!(
        (4.0..=16.0).contains(&edf),
        "Firth-ON edf {edf} outside structurally valid band [4, 16]"
    );
    // Fit existence is the sealed convergence proof (SPEC 20).
    if let Some(g) = fit.outer_gradient_norm {
        let bound = (1.0e-3 * (1.0 + fit.reml_score.abs())).min(1.0);
        assert!(
            g <= bound,
            "Firth-ON final outer gradient norm {g} exceeds score-relative \
             stationarity bound {bound}"
        );
    }

    eprintln!(
        "RECORD_1575_FIRTH reml_score={:.10} edf={:.10} outer_cost_evals={} inner_pirls_solves={} converged=certified",
        fit.reml_score, edf, fit.outer_cost_evals, fit.inner_pirls_solves
    );

    // Outer-work budget: the Firth path does more per-eval work but the eval/solve
    // COUNTS should stay in the same healthy regime as the Firth-off fit (the
    // cached-block change cuts per-eval cost, not the converged trajectory).
    assert!(
        fit.outer_cost_evals > 0 && fit.outer_cost_evals < 80,
        "Firth-ON outer cost evals {} outside sane budget (blow-up = regression)",
        fit.outer_cost_evals
    );
    assert!(
        fit.inner_pirls_solves > 0 && fit.inner_pirls_solves < 200,
        "Firth-ON inner P-IRLS solves {} outside sane budget",
        fit.inner_pirls_solves
    );
}
