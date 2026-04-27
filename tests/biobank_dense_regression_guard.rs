// Regression guard for the biobank-scale dense Duchon timeout fixes that
// landed in `src/solver/pirls.rs::pirls_soft_acceptance`,
// `src/solver/outer_strategy.rs::SEED_SCREENING_CASCADE_MULTIPLIERS`, and
// `OuterProblem::with_standard_gam_dimensions`.
//
// The four fixes act together so that a moderate-scale dense Bernoulli-logit
// GAM converges quickly and without grinding through 100+ inner P-IRLS
// iterations per outer evaluation. This test fits such a problem end-to-end
// and asserts:
//   (a) PIRLS reaches `Converged` or `StalledAtValidMinimum` (the recognised
//       valid-minimum exits — anything else means the soft-acceptance plateau
//       criteria silently regressed).
//   (b) The final inner P-IRLS iteration count is well below the 100-iter
//       budget — a regression that disabled per-iter soft-exit would push
//       this to the cap.
//   (c) Outer-iteration count is bounded — a regression in the
//       seed-screening cascade would force the planner to chew through full
//       inner solves at every cap stage and inflate this.
//   (d) Wall-clock stays under a generous ceiling that is still ~10x lower
//       than what an un-fixed run takes.
//   (e) Predicted probabilities track the true generating model on a held-
//       in metric (Brier score) — guards against a fit that "converges" to
//       the wrong place.
//
// The problem is intentionally smaller than a real biobank fit (k_smoothing
// = 2, below the BFGS-min-k cutoff, so the gradient-only routing in
// `with_standard_gam_dimensions` is not triggered here). This is a smoke
// test for the per-iter soft-acceptance and seed-screening fixes; the
// gradient-only routing has its own unit tests in `outer_strategy.rs`.

use gam::estimate::{FitOptions, fit_gam};
use gam::pirls::PirlsStatus;
use gam::predict::predict_gam;
use gam::smooth::BlockwisePenalty;
use gam::types::LikelihoodFamily;
use ndarray::{Array1, Array2};
use rand::rngs::StdRng;
use rand::{RngExt, SeedableRng};
use std::time::Instant;

/// Build a deterministic dense Bernoulli-logit problem with two smooth
/// blocks. Each block is a degree-(k-1) shifted-Legendre-style polynomial
/// basis on a covariate sampled uniformly from [-1, 1]; the penalty on
/// each block is the standard squared-second-difference penalty
/// `(D_2)' D_2`, which has a 2-dimensional null space (constant + linear).
///
/// Total layout: column 0 = intercept (unpenalized), columns 1..15 = smooth
/// of `x1` (k=14), columns 15..30 = smooth of `x2` (k=15). p = 30.
fn build_problem(
    n: usize,
    seed: u64,
) -> (
    Array2<f64>,
    Array1<f64>,
    Array1<f64>,
    Vec<BlockwisePenalty>,
    Array1<f64>,
) {
    let k1 = 14usize;
    let k2 = 15usize;
    let p = 1 + k1 + k2;
    assert_eq!(p, 30);

    let mut rng = StdRng::seed_from_u64(seed);
    let x1: Vec<f64> = (0..n).map(|_| rng.random_range(-1.0..1.0)).collect();
    let x2: Vec<f64> = (0..n).map(|_| rng.random_range(-1.0..1.0)).collect();

    // True generating mean is two smooth functions.
    let true_eta: Vec<f64> = x1
        .iter()
        .zip(x2.iter())
        .map(|(a, b)| -0.4 + 1.2 * (1.5 * a).sin() + 0.8 * b * b - 0.6 * b)
        .collect();

    let mut x = Array2::<f64>::zeros((n, p));
    for i in 0..n {
        x[[i, 0]] = 1.0;
        // Polynomial basis 1..k columns for x1.
        let mut acc = 1.0;
        for j in 0..k1 {
            acc *= x1[i];
            x[[i, 1 + j]] = acc;
        }
        let mut acc = 1.0;
        for j in 0..k2 {
            acc *= x2[i];
            x[[i, 1 + k1 + j]] = acc;
        }
    }

    let y = Array1::from_iter(true_eta.iter().map(|eta| {
        let p = 1.0 / (1.0 + (-eta).exp());
        if rng.random::<f64>() < p { 1.0 } else { 0.0 }
    }));
    let weights = Array1::ones(n);

    // Squared-second-difference penalty on each smooth block.
    let s1 = second_difference_penalty(k1);
    let s2 = second_difference_penalty(k2);
    let s_list = vec![
        BlockwisePenalty::new(1..(1 + k1), s1),
        BlockwisePenalty::new((1 + k1)..p, s2),
    ];

    let true_p = Array1::from_iter(true_eta.iter().map(|e| 1.0 / (1.0 + (-e).exp())));

    (x, y, weights, s_list, true_p)
}

fn second_difference_penalty(k: usize) -> Array2<f64> {
    // D_2 is (k-2) x k with rows [..., 1, -2, 1, ...].
    // S = D_2' D_2 is k x k pentadiagonal.
    let mut d = Array2::<f64>::zeros((k - 2, k));
    for i in 0..(k - 2) {
        d[[i, i]] = 1.0;
        d[[i, i + 1]] = -2.0;
        d[[i, i + 2]] = 1.0;
    }
    d.t().dot(&d)
}

#[test]
fn biobank_dense_logit_regression_guard() {
    let n = 2_000usize;
    let p = 30usize;
    let (x, y, weights, s_list, true_p) = build_problem(n, 0xB10B_A2C);
    let offset = Array1::zeros(n);

    let start = Instant::now();
    let fit = fit_gam(
        x.view(),
        y.view(),
        weights.view(),
        offset.view(),
        &s_list,
        LikelihoodFamily::BinomialLogit,
        &FitOptions {
            latent_cloglog: None,
            mixture_link: None,
            optimize_mixture: false,
            sas_link: None,
            optimize_sas: false,
            compute_inference: true,
            max_iter: 100,
            tol: 1e-6,
            nullspace_dims: vec![2, 2],
            linear_constraints: None,
            firth_bias_reduction: false,
            adaptive_regularization: None,
            penalty_shrinkage_floor: None,
            rho_prior: Default::default(),
            kronecker_penalty_system: None,
            kronecker_factored: None,
        },
    )
    .expect("biobank-scale dense Bernoulli-logit fit must succeed");
    let elapsed = start.elapsed();

    // (a) Inner solver landed at a valid minimum. Soft-acceptance regressions
    //     would surface as `MaxIterationsReached` or `LmStepSearchExhausted`.
    assert!(
        matches!(
            fit.pirls_status,
            PirlsStatus::Converged | PirlsStatus::StalledAtValidMinimum
        ),
        "PIRLS must terminate at a recognised valid-minimum status, got {:?}",
        fit.pirls_status
    );

    // (b) Final inner P-IRLS iteration count must be safely below the
    //     100-iter budget. A regression that re-disables per-iter
    //     soft-exit would push this to the cap.
    let pirls_iter = fit
        .artifacts
        .pirls
        .as_ref()
        .expect("compute_inference=true must populate FitArtifacts::pirls")
        .iteration;
    assert!(
        pirls_iter < 60,
        "PIRLS final inner iteration count regressed: got {} (>= 60 is the soft-exit regression band; cap is 100)",
        pirls_iter
    );

    // (c) Outer (smoothing) iteration count must be bounded.
    //     `n_params = 2` is well below the BFGS-min-k cutoff so the outer
    //     loop is short by construction; a 50-iter ceiling catches a
    //     wholesale planner regression.
    assert!(
        fit.outer_iterations <= 50,
        "outer iteration count regressed: {} > 50",
        fit.outer_iterations
    );
    assert!(
        fit.outer_converged,
        "outer optimization must converge (got non-converged with {} iterations)",
        fit.outer_iterations
    );

    // (d) Wall-clock ceiling. A pre-fix run on this problem can take many
    //     tens of seconds when the inner loop grinds. 30s is generous on
    //     CI but still much smaller than a regressed run.
    assert!(
        elapsed.as_secs_f64() < 30.0,
        "fit took {:.2}s — wall-clock regression (was expected to run in under a few seconds)",
        elapsed.as_secs_f64()
    );

    // (e) The fit must track the true generating model. Compare predicted
    //     probabilities against the true `p` from the data-generating
    //     process; a fit that "converges" to the wrong place will have a
    //     much larger Brier-style discrepancy.
    let pred = predict_gam(
        x.view(),
        fit.beta.view(),
        offset.view(),
        LikelihoodFamily::BinomialLogit,
    )
    .expect("predict on training X");
    assert_eq!(pred.mean.len(), n);
    let mse_vs_truth = (&pred.mean - &true_p)
        .mapv(|v| v * v)
        .mean()
        .unwrap_or(f64::INFINITY);
    assert!(
        mse_vs_truth < 0.02,
        "fit does not track the true generating probabilities: MSE(pred, true_p) = {mse_vs_truth:.4e}"
    );

    // Diagnostic landmark for future regression triage.
    eprintln!(
        "[biobank_dense_regression_guard] n={n}, p={p}, k_smoothing=2 \
         | wall_clock={:.3}s, outer_iter={}, pirls_iter={}, mse_vs_truth={mse_vs_truth:.4e}, status={:?}",
        elapsed.as_secs_f64(),
        fit.outer_iterations,
        pirls_iter,
        fit.pirls_status,
    );
}
