// Regression guard for the biobank-scale dense Bernoulli-logit convergence
// fixes. Four mechanisms must all stay healthy:
//
//   (1) `pirls_soft_acceptance` per-iteration early-exit on a 2-iter
//       plateau streak (`src/solver/pirls.rs`).
//   (2) `SEED_SCREENING_CASCADE_MULTIPLIERS = [1, 4, 16]`
//       (`src/solver/outer_strategy.rs`).
//   (3,4) `OuterProblem::with_standard_gam_dimensions` auto-routing to
//       gradient-only when truthful Hessian-assembly cost is large.
//
// This test fits a moderate dense problem (n=2000, single smooth, k=8,
// k_smoothing=1) and checks that:
//   - PIRLS terminates at a recognised valid-minimum status.
//   - Inner PIRLS iteration count is far below the 100-iter cap.
//   - Outer optimization converges in a small number of iterations.
//   - Wall-clock is under a generous ceiling.
//   - Predicted η correlates with the true η (>= 0.85) AND mean-abs
//     deviation is small (<= 0.20).
//
// k_smoothing = 1 keeps this below the `k >= 4` cutoff for the auto
// gradient-only routing in `with_standard_gam_dimensions`, so this test
// directly guards mechanisms (1) and (2). A regression in either causes
// either a status mismatch, an inner-iter blowup to ~100, or a wall-clock
// timeout — any of which fails the test.

use gam::estimate::{FitOptions, fit_gam};
use gam::pirls::PirlsStatus;
use gam::predict::predict_gam;
use gam::smooth::BlockwisePenalty;
use gam::types::LikelihoodFamily;
use ndarray::{Array1, Array2};
use rand::rngs::StdRng;
use rand::{RngExt, SeedableRng};
use std::time::Instant;

const N: usize = 2_000;
const K: usize = 8;
const SEED: u64 = 0xB10B_C0FFEE;

/// Cubic B-spline basis on [0,1] with `k` evenly-spaced interior knots,
/// evaluated at `x`. Returns an `n x k` design matrix.
fn bspline_basis(x: &[f64], k: usize) -> Array2<f64> {
    let degree = 3usize;
    // Open uniform knot vector on [0,1] with `k` basis functions.
    // Number of knots = k + degree + 1.
    let n_knots = k + degree + 1;
    let mut knots = vec![0.0f64; n_knots];
    let n_interior = k.saturating_sub(degree + 1);
    for i in 0..n_knots {
        if i <= degree {
            knots[i] = 0.0;
        } else if i >= k {
            knots[i] = 1.0;
        } else {
            let t = (i - degree) as f64 / (n_interior as f64 + 1.0);
            knots[i] = t;
        }
    }

    let n = x.len();
    let mut b = Array2::<f64>::zeros((n, k));
    for (row, &xi) in x.iter().enumerate() {
        // Cox–de Boor recursion. `nd[j]` is the basis function of order d at xi.
        // Order 1: indicator on [knots[j], knots[j+1]).
        let max_order = degree + 1; // 4 for cubic
        let span = k + degree;
        let mut prev = vec![0.0f64; span];
        for j in 0..span {
            let lo = knots[j];
            let hi = knots[j + 1];
            let inside = if (xi - 1.0).abs() < 1e-12 {
                hi >= 1.0 - 1e-12 && lo < 1.0 - 1e-12
            } else {
                xi >= lo && xi < hi
            };
            prev[j] = if inside { 1.0 } else { 0.0 };
        }
        for d in 2..=max_order {
            let mut next = vec![0.0f64; span];
            for j in 0..(span - d + 1) {
                let denom_l = knots[j + d - 1] - knots[j];
                let denom_r = knots[j + d] - knots[j + 1];
                let left = if denom_l > 0.0 {
                    (xi - knots[j]) / denom_l * prev[j]
                } else {
                    0.0
                };
                let right = if denom_r > 0.0 {
                    (knots[j + d] - xi) / denom_r * prev[j + 1]
                } else {
                    0.0
                };
                next[j] = left + right;
            }
            prev = next;
        }
        for j in 0..k {
            b[[row, j]] = prev[j];
        }
    }
    b
}

/// Squared-second-difference penalty `D_2' D_2`, k x k.
fn second_difference_penalty(k: usize) -> Array2<f64> {
    let mut d = Array2::<f64>::zeros((k - 2, k));
    for i in 0..(k - 2) {
        d[[i, i]] = 1.0;
        d[[i, i + 1]] = -2.0;
        d[[i, i + 2]] = 1.0;
    }
    d.t().dot(&d)
}

#[test]
fn biobank_convergence_regression() {
    let n = N;
    let k = K;
    let mut rng = StdRng::seed_from_u64(SEED);

    // Covariate uniform on [0,1].
    let x_raw: Vec<f64> = (0..n).map(|_| rng.random_range(0.0..1.0)).collect();
    let basis = bspline_basis(&x_raw, k);

    // True η: a non-trivial smooth oscillation in [-1.5, 1.5].
    let two_pi = std::f64::consts::TAU;
    let true_eta: Array1<f64> = Array1::from_iter(
        x_raw
            .iter()
            .map(|&t| (two_pi * t).sin() + 0.5 * (2.0 * two_pi * t).cos()),
    );

    // Bernoulli draws from logit^{-1}(η).
    let y = Array1::from_iter(true_eta.iter().map(|&eta| {
        let p = 1.0 / (1.0 + (-eta).exp());
        if rng.random::<f64>() < p { 1.0 } else { 0.0 }
    }));

    // Design = [intercept | basis]; p = 1 + k.
    let p = 1 + k;
    let mut x_design = Array2::<f64>::zeros((n, p));
    for i in 0..n {
        x_design[[i, 0]] = 1.0;
        for j in 0..k {
            x_design[[i, 1 + j]] = basis[[i, j]];
        }
    }

    // Deterministic non-uniform row weights exercise the fused weighted-residual
    // and weighted-design assembly paths while preserving an exact integer-row
    // weighting interpretation.
    let weights = Array1::from_iter((0..n).map(|i| if i % 5 == 0 { 2.0 } else { 1.0 }));
    let offset = Array1::<f64>::zeros(n);

    // Single smooth block — k_smoothing = 1.
    let s_block = second_difference_penalty(k);
    let s_list = vec![BlockwisePenalty::new(1..(1 + k), s_block)];

    let start = Instant::now();
    let fit = fit_gam(
        x_design.view(),
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
            // D_2 has a 2-d null space (constant + linear).
            nullspace_dims: vec![2],
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
    //     would surface as `MaxIterationsReached` or similar non-valid exits.
    assert!(
        matches!(
            fit.pirls_status,
            PirlsStatus::Converged | PirlsStatus::StalledAtValidMinimum
        ),
        "PIRLS must terminate at a recognised valid-minimum status, got {:?}",
        fit.pirls_status
    );

    // (b) Final inner P-IRLS iteration count must be safely below the 100-iter
    //     cap. A regression that disables the per-iter `pirls_soft_acceptance`
    //     plateau early-exit pushes this toward the cap.
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

    // (c) Outer (smoothing) iteration count must be bounded. A wholesale
    //     planner / seed-screening regression would inflate this.
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

    // (d) Wall-clock ceiling. A pre-fix run on this problem can grind tens of
    //     seconds when the inner loop is starved of soft-exit. 30s is generous
    //     on CI but still much smaller than a regressed run.
    assert!(
        elapsed.as_secs_f64() < 30.0,
        "fit took {:.2}s — wall-clock regression",
        elapsed.as_secs_f64()
    );

    // (e) Fit must track the true generating η. Two complementary checks:
    //     correlation guards against shape collapse; mean-abs deviation guards
    //     against systematic bias / wrong scale. Both must hold.
    let pred = predict_gam(
        x_design.view(),
        fit.beta.view(),
        offset.view(),
        LikelihoodFamily::BinomialLogit,
    )
    .expect("predict on training X");

    // `pred.mean` is on the response scale (probabilities); compare on η-scale
    // by converting back through logit so the threshold matches team-lead's
    // spec for "correlation between predicted η and true η".
    let pred_eta: Array1<f64> = pred.mean.mapv(|p| {
        let pc = p.clamp(1e-12, 1.0 - 1e-12);
        (pc / (1.0 - pc)).ln()
    });
    let mean_pred = pred_eta.mean().unwrap_or(0.0);
    let mean_true = true_eta.mean().unwrap_or(0.0);
    let mut num = 0.0f64;
    let mut den_p = 0.0f64;
    let mut den_t = 0.0f64;
    let mut sad = 0.0f64;
    for i in 0..n {
        let dp = pred_eta[i] - mean_pred;
        let dt = true_eta[i] - mean_true;
        num += dp * dt;
        den_p += dp * dp;
        den_t += dt * dt;
        sad += (pred_eta[i] - true_eta[i]).abs();
    }
    let corr = num / (den_p.sqrt() * den_t.sqrt()).max(1e-30);
    let mad = sad / n as f64;
    assert!(
        corr >= 0.85,
        "fit does not track true η: correlation = {corr:.4} (< 0.85)"
    );
    assert!(
        mad <= 0.20,
        "fit has too much η-scale deviation: mean-abs deviation = {mad:.4} (> 0.20)"
    );

    eprintln!(
        "[biobank_convergence_regression] n={n}, p={p}, k_smoothing=1 \
         | wall_clock={:.3}s, outer_iter={}, pirls_iter={}, corr_eta={:.4}, mad_eta={:.4}, status={:?}, weighted_rows=true",
        elapsed.as_secs_f64(),
        fit.outer_iterations,
        pirls_iter,
        corr,
        mad,
        fit.pirls_status,
    );
}
