//! Scaling-law probe for standard-GAM Bernoulli-probit at biobank shape.
//!
//! Times one full fit at each of several n values, then prints a summary
//! table that lets us extrapolate to the biobank n=320_000 target without
//! waiting on a 30-50 minute CI cycle.
//!
//! Run with:
//! ```text
//! cargo test --release --test standard_gam_scaling -- --ignored --nocapture standard_gam_scaling_law
//! ```
//!
//! The `[SCALING]` lines in the output are pivotable: parse them into a
//! (n, total_s, per_iter_s) triple per row and fit `total_s = a * n^α`.
//! Mission target: extrapolated total at n=320k must be ≤ 2400s (CI's
//! 40-min cmd timeout). With path #2/#3 + standard-GAM gate this is
//! expected to drop dramatically vs the pre-fix scaling.

use gam::estimate::{FitOptions, fit_gam};
use gam::pirls::PirlsStatus;
use gam::smooth::BlockwisePenalty;
use gam::types::LikelihoodFamily;
use ndarray::{Array1, Array2};
use rand::rngs::StdRng;
use rand::{RngExt, SeedableRng};
use std::time::Instant;

mod common;
use common::report_power_law as fit_and_report_power_law_inner;

const K: usize = 8;
const SEED: u64 = 0x5CA1_AB1E;

/// Cubic B-spline basis on [0,1] with `k` evenly-spaced interior knots.
fn bspline_basis(x: &[f64], k: usize) -> Array2<f64> {
    let degree = 3usize;
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
        let max_order = degree + 1;
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

fn second_difference_penalty(k: usize) -> Array2<f64> {
    let mut d = Array2::<f64>::zeros((k - 2, k));
    for i in 0..(k - 2) {
        d[[i, i]] = 1.0;
        d[[i, i + 1]] = -2.0;
        d[[i, i + 2]] = 1.0;
    }
    d.t().dot(&d)
}

fn run_fit(n: usize) -> (f64, usize, usize, bool) {
    run_fit_with_k(n, K)
}

fn run_fit_with_k(n: usize, k: usize) -> (f64, usize, usize, bool) {
    let mut rng = StdRng::seed_from_u64(SEED.wrapping_add(n as u64));
    let x_raw: Vec<f64> = (0..n).map(|_| rng.random_range(0.0..1.0)).collect();
    let basis = bspline_basis(&x_raw, k);
    let two_pi = std::f64::consts::TAU;
    let true_eta: Array1<f64> = Array1::from_iter(
        x_raw
            .iter()
            .map(|&t| (two_pi * t).sin() + 0.5 * (2.0 * two_pi * t).cos()),
    );
    let y = Array1::from_iter(true_eta.iter().map(|&eta| {
        let p = 1.0 / (1.0 + (-eta).exp());
        if rng.random::<f64>() < p { 1.0 } else { 0.0 }
    }));
    let p = 1 + k;
    let mut x_design = Array2::<f64>::zeros((n, p));
    for i in 0..n {
        x_design[[i, 0]] = 1.0;
        for j in 0..k {
            x_design[[i, 1 + j]] = basis[[i, j]];
        }
    }
    let weights = Array1::ones(n);
    let offset = Array1::<f64>::zeros(n);
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
    .expect("standard-GAM scaling probe must succeed");
    let elapsed = start.elapsed().as_secs_f64();

    let pirls_iter = fit
        .artifacts
        .pirls
        .as_ref()
        .expect("compute_inference=true must populate FitArtifacts::pirls")
        .iteration;

    let converged = matches!(
        fit.pirls_status,
        PirlsStatus::Converged | PirlsStatus::StalledAtValidMinimum
    ) && fit.outer_converged;

    (elapsed, fit.outer_iterations, pirls_iter, converged)
}

#[test]
fn standard_gam_scaling_law() {
    // Sweep doubling-ish to span a few orders of magnitude.
    let ns: Vec<usize> = vec![2_000, 5_000, 10_000, 25_000, 50_000, 100_000];

    eprintln!("\n[SCALING] header n total_s outer_iters pirls_iter per_outer_s converged");
    let mut rows: Vec<(usize, f64, usize, usize, bool)> = Vec::new();
    for &n in &ns {
        let (elapsed, outer_iters, pirls_iter, converged) = run_fit(n);
        let per_outer = elapsed / (outer_iters.max(1) as f64);
        eprintln!(
            "[SCALING] row n={n} total_s={elapsed:.3} outer_iters={outer_iters} pirls_iter={pirls_iter} per_outer_s={per_outer:.4} converged={converged}"
        );
        rows.push((n, elapsed, outer_iters, pirls_iter, converged));
    }

    // Fit a power law `total_s = a · n^α` via log-log least squares,
    // gated on rows that GENUINELY converged (outer_iters strictly less
    // than the max_iter cap of 100 — `outer_converged=true` from the
    // public API is unreliable when the inner solve hit its own cap).
    // Report R² and the largest log-residual so the verdict is honest:
    // a fit with R²<0.85 is unreliable for extrapolation across decades.
    fit_and_report_power_law(
        "[SCALING]",
        rows.iter()
            .filter_map(|(n, t, oi, _pi, _)| {
                // honest convergence gate: outer iters < cap AND time finite
                if *t > 0.0 && t.is_finite() && *oi < 100 {
                    Some((*n as f64, *t))
                } else {
                    None
                }
            })
            .collect(),
        &[("n=320k", 320_000.0), ("n=1M", 1_000_000.0)],
        2400.0,
    );
}

/// Thin wrapper preserving the v1 call site (`Vec<(f64, f64)>`,
/// fire-and-forget). The body lives in `tests/common/mod.rs` —
/// see `common::report_power_law` for the policy.
fn fit_and_report_power_law(
    tag: &str,
    points: Vec<(f64, f64)>,
    extrapolate: &[(&str, f64)],
    budget_y: f64,
) {
    let _ = fit_and_report_power_law_inner(tag, &points, extrapolate, budget_y);
}

#[test]
fn standard_gam_p_scaling_law() {
    // Fix n at 50k, sweep k (basis size). The CI bench
    // `rust_gam_jointpc_duchon60` uses 60 centers + 16-D PCs ≈ p~70-80.
    // Extrapolate the per-row cost vs k to see whether the standard-GAM
    // fit stays under 2400s at p=80, n=320k.
    let n: usize = 50_000;
    let ks: Vec<usize> = vec![8, 16, 32, 48, 64];
    eprintln!("\n[P-SCALING] header k total_s outer_iters per_outer_s converged");
    let mut rows: Vec<(usize, f64, usize, bool)> = Vec::new();
    for &k in &ks {
        let (elapsed, outer_iters, _pirls, converged) = run_fit_with_k(n, k);
        let per_outer = elapsed / (outer_iters.max(1) as f64);
        eprintln!(
            "[P-SCALING] row k={k} total_s={elapsed:.3} outer_iters={outer_iters} per_outer_s={per_outer:.4} converged={converged}"
        );
        rows.push((k, elapsed, outer_iters, converged));
    }
    fit_and_report_power_law(
        "[P-SCALING]",
        rows.iter()
            .filter_map(|(k, t, oi, _)| {
                if *t > 0.0 && t.is_finite() && *oi < 100 {
                    Some((*k as f64, *t))
                } else {
                    None
                }
            })
            .collect(),
        &[("k=42", 42.0), ("k=80", 80.0)],
        // P-scaling here is at fixed n=50k. The 2400s budget is at
        // biobank n=320k. Approx target at fixed n=50k for "fits at
        // biobank": 2400 / (320/50)^1 = 375s (assuming n^1 scaling).
        // Use this as the budget so verdicts at this n make sense.
        375.0,
    );
}
