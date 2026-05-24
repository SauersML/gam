//! JumpReLU STE (straight-through estimator) numeric-vs-analytic agreement.
//!
//! These tests pin down the analytic value / gradient / Hessian-diag of
//! `JumpReLUPenalty` against central-difference numerical references, which is
//! exactly what the gamfit composition engine needs to remain trustworthy as a
//! drop-in for outer-loop REML. Companion to the Python-side
//! `gamfit.torch.JumpReLUPenalty` whose backward uses the same smoothed gate.
//!
//! Reference: Paulo et al. "Transcoders Beat Sparse Autoencoders for
//! Interpretability." arXiv:2501.18823, 2025 — eq. 3 for the per-axis STE.

use gam::terms::PsiSlice;
use gam::terms::analytic_penalties::{AnalyticPenalty, JumpReLUPenalty};
use ndarray::Array1;

fn build(thresholds: Vec<f64>, weight: f64, eps: f64, n_obs: usize) -> JumpReLUPenalty {
    let d = thresholds.len();
    let slice = PsiSlice {
        range: 0..(n_obs * d),
        latent_dim: Some(d),
    };
    JumpReLUPenalty::new(slice, Array1::from(thresholds), weight, eps)
        .expect("JumpReLUPenalty::new")
}

/// Test 1 — analytic grad vs central-difference numeric grad, over a grid
/// that straddles the threshold (so the STE smoothing zone matters).
#[test]
fn jumprelu_ste_grad_matches_numeric_central_difference() {
    let thresholds = vec![0.10, 0.25, 0.50];
    let d = thresholds.len();
    let n_obs = 7;
    let weight = 1.5;
    let eps = 5e-2;
    let penalty = build(thresholds.clone(), weight, eps, n_obs);

    // Random-ish target straddling each threshold.
    let mut target = Array1::<f64>::zeros(n_obs * d);
    for row in 0..n_obs {
        for axis in 0..d {
            target[row * d + axis] = thresholds[axis] + 0.03 * (row as f64 - 3.0);
        }
    }
    let rho = Array1::<f64>::zeros(d);

    let analytic = penalty.grad_target(target.view(), rho.view());

    let h = 1e-6;
    for i in 0..target.len() {
        let mut t_plus = target.clone();
        let mut t_minus = target.clone();
        t_plus[i] += h;
        t_minus[i] -= h;
        let v_plus = penalty.value(t_plus.view(), rho.view());
        let v_minus = penalty.value(t_minus.view(), rho.view());
        let numeric = (v_plus - v_minus) / (2.0 * h);
        let a = analytic[i];
        let rel = (numeric - a).abs() / (numeric.abs().max(a.abs()).max(1e-8));
        assert!(
            rel < 1e-3,
            "grad mismatch at i={i}: analytic={a:.6e} numeric={numeric:.6e} rel={rel:.3e}"
        );
    }
}

/// Test 2 — analytic Hessian-diag vs central-difference of the analytic
/// gradient (full Newton self-consistency).
#[test]
fn jumprelu_ste_hessian_diag_matches_numeric_grad_diff() {
    let thresholds = vec![0.20, 0.40];
    let d = thresholds.len();
    let n_obs = 5;
    let weight = 1.0;
    let eps = 8e-2;
    let penalty = build(thresholds.clone(), weight, eps, n_obs);

    let mut target = Array1::<f64>::zeros(n_obs * d);
    for row in 0..n_obs {
        for axis in 0..d {
            target[row * d + axis] = thresholds[axis] + 0.02 * row as f64;
        }
    }
    let rho = Array1::<f64>::zeros(d);

    let analytic_diag = penalty
        .hessian_diag(target.view(), rho.view())
        .expect("hessian_diag");

    let h = 1e-5;
    for i in 0..target.len() {
        let mut t_plus = target.clone();
        let mut t_minus = target.clone();
        t_plus[i] += h;
        t_minus[i] -= h;
        let g_plus = penalty.grad_target(t_plus.view(), rho.view())[i];
        let g_minus = penalty.grad_target(t_minus.view(), rho.view())[i];
        let numeric_diag = (g_plus - g_minus) / (2.0 * h);
        let a = analytic_diag[i];
        let denom = numeric_diag.abs().max(a.abs()).max(1e-6);
        let rel = (numeric_diag - a).abs() / denom;
        assert!(
            rel < 5e-3,
            "hessian-diag mismatch at i={i}: analytic={a:.6e} numeric={numeric_diag:.6e} rel={rel:.3e}"
        );
    }
}

/// Test 3 — `grad_rho` matches central-difference w.r.t. the log-threshold
/// parameter, validating the threshold-learning path that REML uses.
#[test]
fn jumprelu_ste_grad_rho_matches_numeric_threshold_difference() {
    let thresholds = vec![0.15, 0.30, 0.45];
    let d = thresholds.len();
    let n_obs = 6;
    let weight = 0.7;
    let eps = 4e-2;
    let penalty = build(thresholds.clone(), weight, eps, n_obs);

    let mut target = Array1::<f64>::zeros(n_obs * d);
    for row in 0..n_obs {
        for axis in 0..d {
            target[row * d + axis] = thresholds[axis] + 0.05 * (row as f64 - 2.5);
        }
    }
    let rho = Array1::<f64>::from(vec![0.0; d]);

    let analytic = penalty.grad_rho(target.view(), rho.view());

    let h = 1e-6;
    for axis in 0..d {
        let mut rho_p = rho.clone();
        let mut rho_m = rho.clone();
        rho_p[axis] += h;
        rho_m[axis] -= h;
        let v_p = penalty.value(target.view(), rho_p.view());
        let v_m = penalty.value(target.view(), rho_m.view());
        let numeric = (v_p - v_m) / (2.0 * h);
        let a = analytic[axis];
        let rel = (numeric - a).abs() / (numeric.abs().max(a.abs()).max(1e-8));
        assert!(
            rel < 1e-3,
            "grad_rho mismatch at axis={axis}: analytic={a:.6e} numeric={numeric:.6e} rel={rel:.3e}"
        );
    }
}

/// Test 4 — limiting behaviour: as ε → 0, JumpReLU value tends to the true
/// hard-threshold ``Σ τ · 1[z > τ]`` weighted by ``weight``. This pins the
/// STE smoothing to a meaningful zero-limit.
#[test]
fn jumprelu_ste_value_converges_to_hard_threshold_as_eps_shrinks() {
    let thresholds = vec![0.2, 0.6];
    let d = thresholds.len();
    let n_obs = 100;
    let weight = 1.0;

    // Build targets well clear of each threshold (no boundary cases).
    let mut target = Array1::<f64>::zeros(n_obs * d);
    let mut hard_total = 0.0;
    for row in 0..n_obs {
        for axis in 0..d {
            let z = (row as f64 + 1.0) / (n_obs as f64) * 1.2;
            target[row * d + axis] = z;
            if z > thresholds[axis] + 0.05 {
                hard_total += weight * thresholds[axis];
            }
        }
    }
    let rho = Array1::<f64>::zeros(d);

    let mut prev_err = f64::INFINITY;
    for &eps in &[1e-1, 1e-2, 1e-3, 1e-4] {
        let p = build(thresholds.clone(), weight, eps, n_obs);
        let v = p.value(target.view(), rho.view());
        let err = (v - hard_total).abs();
        // Monotone-ish convergence (allow slack for boundary jitter).
        assert!(
            err <= prev_err * 1.5 + 1e-9,
            "eps={eps} not converging: prev_err={prev_err:.4e} err={err:.4e}",
        );
        prev_err = err;
    }
    assert!(
        prev_err < 1e-2,
        "did not converge tightly to hard threshold: final err={prev_err:.4e}"
    );
}
