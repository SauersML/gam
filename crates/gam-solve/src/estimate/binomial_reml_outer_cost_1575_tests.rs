//! #1575 measurement + regression: binomial/logit REML outer-loop cost.
//!
//! The issue reports a plain 3-smooth logistic GAM taking ~150 outer REML
//! cost/gradient/Hessian evaluations — an order of magnitude more outer work
//! than mgcv's REML Newton (~15) — with each outer eval paying a full n-sized
//! P-IRLS solve. The outer-eval count is n-independent, so this fixture uses a
//! deliberately small `n` (the outer overhead is fully visible there) to keep
//! the test cheap while still exercising the outer optimizer's convergence.

use super::*;
use gam_problem::{InverseLink, LikelihoodSpec, ResponseFamily, StandardLink};
use gam_terms::smooth::BlockwisePenalty;
use ndarray::{Array1, Array2};

const N: usize = 1000;
const K: usize = 10;
const N_SMOOTH: usize = 3;

struct Lcg {
    state: u64,
}
impl Lcg {
    fn new(seed: u64) -> Self {
        Self {
            state: seed.wrapping_add(0x9E37_79B9_7F4A_7C15),
        }
    }
    fn next_u64(&mut self) -> u64 {
        let mut z = self.state;
        self.state = self.state.wrapping_add(0x9E37_79B9_7F4A_7C15);
        z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
        z ^ (z >> 31)
    }
    fn unit(&mut self) -> f64 {
        (self.next_u64() >> 11) as f64 / (1u64 << 53) as f64
    }
}

/// Cox–de Boor cubic B-spline design on `[0,1]` with `n_basis` uniform-knot
/// bases (mgcv `bs="ps"` analogue).
fn cubic_bspline_design(xs: &[f64], n_basis: usize) -> Array2<f64> {
    let degree = 3usize;
    let n_internal = n_basis - (degree + 1);
    let n_knots = n_basis + degree + 1;
    let mut knots = vec![0.0f64; n_knots];
    for (k, slot) in knots.iter_mut().enumerate() {
        *slot = if k <= degree {
            0.0
        } else if k >= n_knots - degree - 1 {
            1.0
        } else {
            (k - degree) as f64 / (n_internal as f64 + 1.0)
        };
    }
    fn bspline(i: usize, p: usize, x: f64, knots: &[f64]) -> f64 {
        if p == 0 {
            return if (knots[i] <= x && x < knots[i + 1])
                || (x == 1.0 && knots[i + 1] == 1.0 && knots[i] < knots[i + 1])
            {
                1.0
            } else {
                0.0
            };
        }
        let mut left = 0.0;
        let d1 = knots[i + p] - knots[i];
        if d1 > 0.0 {
            left = (x - knots[i]) / d1 * bspline(i, p - 1, x, knots);
        }
        let mut right = 0.0;
        let d2 = knots[i + p + 1] - knots[i + 1];
        if d2 > 0.0 {
            right = (knots[i + p + 1] - x) / d2 * bspline(i + 1, p - 1, x, knots);
        }
        left + right
    }
    let mut b = Array2::<f64>::zeros((xs.len(), n_basis));
    for (r, &x) in xs.iter().enumerate() {
        for i in 0..n_basis {
            b[[r, i]] = bspline(i, degree, x, &knots);
        }
    }
    b
}

/// 2nd-order difference penalty `S = DᵀD` (nullspace dim 2).
fn second_difference_penalty(k: usize) -> Array2<f64> {
    let m = k - 2;
    let mut d = Array2::<f64>::zeros((m, k));
    for r in 0..m {
        d[[r, r]] = 1.0;
        d[[r, r + 1]] = -2.0;
        d[[r, r + 2]] = 1.0;
    }
    d.t().dot(&d)
}

fn build_fixture() -> (Array2<f64>, Array1<f64>, Vec<BlockwisePenalty>) {
    build_fixture_n(N, K)
}

/// Parameterised version of [`build_fixture`] so the outer-cost scaling harness
/// can measure the SAME 3-smooth logistic problem at several `n` in one process
/// (the outer-eval count is claimed n-independent in #1575; the harness checks
/// it empirically). `build_fixture()` is exactly `build_fixture_n(N, K)`.
fn build_fixture_n(n: usize, k: usize) -> (Array2<f64>, Array1<f64>, Vec<BlockwisePenalty>) {
    let p = 1 + N_SMOOTH * k;
    let mut rng = Lcg::new(100 + n as u64);

    let mut cov = vec![vec![0.0f64; n]; N_SMOOTH];
    for row in cov.iter_mut() {
        for v in row.iter_mut() {
            *v = rng.unit();
        }
    }

    let mut x = Array2::<f64>::zeros((n, p));
    for i in 0..n {
        x[[i, 0]] = 1.0;
    }
    let mut s_list = Vec::with_capacity(N_SMOOTH);
    for j in 0..N_SMOOTH {
        let block = cubic_bspline_design(&cov[j], k);
        for i in 0..n {
            for c in 0..k {
                x[[i, 1 + j * k + c]] = block[[i, c]];
            }
        }
        let start = 1 + j * k;
        s_list.push(BlockwisePenalty::new(
            start..(start + k),
            second_difference_penalty(k),
        ));
    }

    let mut y = Array1::<f64>::zeros(n);
    for i in 0..n {
        let (x1, x2, x3) = (cov[0][i], cov[1][i], cov[2][i]);
        let f = (2.0 * std::f64::consts::PI * x1).sin() * 1.5 + (x2 - 0.5).powi(2) * 6.0 - 1.0
            + (3.0 * std::f64::consts::PI * x3).cos();
        let prob = 1.0 / (1.0 + (-f).exp());
        y[i] = if rng.unit() < prob { 1.0 } else { 0.0 };
    }

    (x, y, s_list)
}

fn logit_options() -> FitOptions {
    FitOptions {
        compute_inference: true,
        max_iter: 300,
        tol: 1e-7,
        nullspace_dims: vec![2; N_SMOOTH],
        ..FitOptions::default()
    }
}

#[test]
fn binomial_logit_reml_outer_cost_is_bounded_1575() {
    let (x, y, s_list) = build_fixture();
    let weights = Array1::<f64>::ones(N);
    let offset = Array1::<f64>::zeros(N);

    let t0 = std::time::Instant::now();
    let fit = fit_gam(
        x.clone(),
        y.view(),
        weights.view(),
        offset.view(),
        &s_list,
        LikelihoodSpec::new(
            ResponseFamily::Binomial,
            InverseLink::Standard(StandardLink::Logit),
        ),
        &logit_options(),
    )
    .expect("binomial/logit P-spline REML fit should succeed");
    let dt = t0.elapsed();

    eprintln!(
        "#1575 binomial/logit REML: n={N} k={K} p={}  time={:.3}s  \
         outer_cost_evals={}  inner_pirls_solves={}  converged={}  \
         grad_norm={:?}  reml={:.6}  edf={:.4}  lambdas={:?}",
        1 + N_SMOOTH * K,
        dt.as_secs_f64(),
        fit.outer_cost_evals,
        fit.inner_pirls_solves,
        fit.outer_converged,
        fit.outer_gradient_norm,
        fit.reml_score,
        fit.edf_total().unwrap_or(f64::NAN),
        fit.lambdas
            .iter()
            .map(|l| format!("{l:.3e}"))
            .collect::<Vec<_>>(),
    );

    assert!(
        fit.outer_converged,
        "outer REML must certify convergence, not grind to the iter cap"
    );
    // mgcv's REML Newton converges in well under ~15 outer iterations on this
    // family of problem. Pin a generous multiple so the test fails loudly if
    // the outer loop regresses back to the ~150-eval grind reported in #1575.
    assert!(
        fit.outer_cost_evals <= 60,
        "outer REML cost evals = {} — expected the bounded outer loop, not the \
         ~150-eval grind (#1575)",
        fit.outer_cost_evals
    );
    // The genuinely expensive work is the count of cache-missing full-n inner
    // P-IRLS solves (`outer_cost_evals` counts outer *requests*, cache hits
    // included, and under-counts the real solve budget). The seed-grid prepass
    // dominates this count; memoizing its criterion-ranked coordinate-descent
    // probes (which re-draw the same ρ across sweeps) removed the redundant
    // re-solves bit-identically, taking this fixture from ~74 to ~65. Pin a
    // guard above the observed value so a regression that reintroduces the
    // duplicate grid probes (or a wider unbounded outer loop) fails loudly,
    // while leaving headroom for benign scheduling variation.
    assert!(
        fit.inner_pirls_solves <= 90,
        "cache-missing inner P-IRLS solves = {} — expected the deduplicated \
         seed-grid + outer loop, not redundant re-solves of identical ρ (#1575)",
        fit.inner_pirls_solves
    );
}

/// n-independence gate for the binomial/logit REML outer loop (#1575's central
/// claim). Fits the SAME 3-smooth logistic problem at two sizes and reports the
/// outer-eval / inner-solve / wall-clock table, then asserts the outer cost-eval
/// count does NOT scale with `n` — each fit stays within the same bounded outer
/// budget the single-`n` sibling pins. The outer overhead is fully visible at
/// small `n`, so the sweep is capped there to keep the gate cheap; a data-scaling
/// regression (the ~150-eval grind #1575 reported) would blow the bound loudly.
#[test]
fn binomial_logit_reml_outer_cost_is_n_independent_1575() {
    eprintln!(
        "{:>7} {:>4} {:>4} {:>8} {:>8} {:>10} {:>10} {:>9}",
        "n", "k", "p", "outer", "inner", "time_s", "time/inner", "conv"
    );
    let mut outer_by_n: Vec<(usize, usize)> = Vec::new();
    for &n in &[1000usize, 2000] {
        let k = K;
        let (x, y, s_list) = build_fixture_n(n, k);
        let weights = Array1::<f64>::ones(n);
        let offset = Array1::<f64>::zeros(n);
        let mut opts = logit_options();
        opts.nullspace_dims = vec![2; N_SMOOTH];
        let t0 = std::time::Instant::now();
        let fit = fit_gam(
            x,
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
        .expect("binomial/logit REML fit should succeed");
        let dt = t0.elapsed().as_secs_f64();
        eprintln!(
            "{:>7} {:>4} {:>4} {:>8} {:>8} {:>10.3} {:>10.4} {:>9}",
            n,
            k,
            1 + N_SMOOTH * k,
            fit.outer_cost_evals,
            fit.inner_pirls_solves,
            dt,
            dt / (fit.inner_pirls_solves.max(1) as f64),
            fit.outer_converged,
        );
        assert!(
            fit.outer_converged,
            "n={n}: outer REML must certify convergence, not grind to the iter cap"
        );
        // Same bounded outer budget the single-`n` sibling gate pins: the count
        // must stay bounded as `n` grows (that IS n-independence). mgcv's Newton
        // does this in ~15; the #1575 grind was ~150.
        assert!(
            fit.outer_cost_evals <= 60,
            "n={n}: outer REML cost evals = {} — expected the bounded, \
             n-independent outer loop, not the ~150-eval grind (#1575)",
            fit.outer_cost_evals
        );
        outer_by_n.push((n, fit.outer_cost_evals));
    }
    // Direct n-independence check: doubling `n` must not materially grow the
    // outer cost-eval count (a data-scaling regression would roughly track `n`).
    let (n_small, outer_small) = outer_by_n[0];
    let (n_large, outer_large) = outer_by_n[1];
    assert!(
        outer_large <= outer_small + 15,
        "outer cost-eval count scaled with n (#1575): n={n_small}→{outer_small}, \
         n={n_large}→{outer_large}; the outer loop must be n-independent"
    );
}
