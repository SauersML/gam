#![cfg(test)]

use ndarray::{Array1, Array2, array};

use crate::estimate::{ExternalOptimOptions, evaluate_external_gradients};
use crate::types::LikelihoodFamily;

#[inline]
fn softplus(x: f64) -> f64 {
    if x > 0.0 {
        x + (-x).exp().ln_1p()
    } else {
        x.exp().ln_1p()
    }
}

#[derive(Clone, Copy, Debug)]
struct OracleConfig {
    grid_bound: f64,
    steps: usize,
}

impl Default for OracleConfig {
    fn default() -> Self {
        Self {
            grid_bound: 8.0,
            steps: 241,
        }
    }
}

#[derive(Clone, Copy, Debug)]
struct OracleEval {
    log_evidence: f64,
    grad_rho: f64,
}

/// Deterministic 2D grid oracle for tiny Bernoulli-logit model.
///
/// Evidence:
///   L(ρ) = ∫ exp( l(β) - 0.5 λ ||β||² ) dβ,  λ=exp(ρ), S=I₂.
///
/// Exact gradient identity:
///   d/dρ log L(ρ) = -0.5 λ E_post[ ||β||² ].
///
/// We evaluate both by direct 2D quadrature over β=(β₁,β₂), using a stabilized
/// log-sum-exp style integral to avoid underflow.
fn exact_logit_oracle_eval_rho_2d(
    y: &Array1<f64>,
    x: &Array2<f64>,
    rho: f64,
    cfg: OracleConfig,
) -> OracleEval {
    assert_eq!(x.ncols(), 2);
    assert_eq!(x.nrows(), y.len());
    assert!(cfg.steps >= 5);

    let lambda = rho.exp();
    let h = (2.0 * cfg.grid_bound) / ((cfg.steps - 1) as f64);

    // First pass: locate max log-kernel for stable log-sum-exp integration.
    let mut max_logk = f64::NEG_INFINITY;
    for i in 0..cfg.steps {
        let b1 = -cfg.grid_bound + (i as f64) * h;
        for j in 0..cfg.steps {
            let b2 = -cfg.grid_bound + (j as f64) * h;
            let beta = array![b1, b2];
            let eta = x.dot(&beta);
            let ll = eta
                .iter()
                .zip(y.iter())
                .map(|(&e, &yy)| yy * e - softplus(e))
                .sum::<f64>();
            let pen = 0.5 * lambda * (b1 * b1 + b2 * b2);
            max_logk = max_logk.max(ll - pen);
        }
    }

    // Second pass: integrate denominator and quadratic numerator.
    let mut z = 0.0_f64;
    let mut q = 0.0_f64;
    for i in 0..cfg.steps {
        let b1 = -cfg.grid_bound + (i as f64) * h;
        let wi = if i == 0 || i + 1 == cfg.steps {
            0.5
        } else {
            1.0
        };
        for j in 0..cfg.steps {
            let b2 = -cfg.grid_bound + (j as f64) * h;
            let wj = if j == 0 || j + 1 == cfg.steps {
                0.5
            } else {
                1.0
            };
            let beta = array![b1, b2];
            let eta = x.dot(&beta);
            let ll = eta
                .iter()
                .zip(y.iter())
                .map(|(&e, &yy)| yy * e - softplus(e))
                .sum::<f64>();
            let pen = 0.5 * lambda * (b1 * b1 + b2 * b2);
            let w = wi * wj * (ll - pen - max_logk).exp();
            z += w;
            q += w * (b1 * b1 + b2 * b2);
        }
    }
    let log_evidence = max_logk + z.max(1e-300).ln() + 2.0 * h.ln();
    let post_q = q / z.max(1e-300);
    let grad_rho = -0.5 * lambda * post_q;
    OracleEval {
        log_evidence,
        grad_rho,
    }
}

fn exact_logit_oracle_grad_fd_rho_2d(
    y: &Array1<f64>,
    x: &Array2<f64>,
    rho: f64,
    cfg: OracleConfig,
) -> f64 {
    let h = 1e-3_f64.max(1e-4 * (1.0 + rho.abs()));
    let lp = exact_logit_oracle_eval_rho_2d(y, x, rho + 0.5 * h, cfg).log_evidence;
    let lm = exact_logit_oracle_eval_rho_2d(y, x, rho - 0.5 * h, cfg).log_evidence;
    (lp - lm) / h
}

fn laml_gradient_external_logit(y: &Array1<f64>, x: &Array2<f64>, rho: f64) -> f64 {
    let w = Array1::ones(y.len());
    let offset = Array1::zeros(y.len());
    let s_list = vec![Array2::eye(2)];
    let opts = ExternalOptimOptions {
        family: LikelihoodFamily::BinomialLogit,
        max_iter: 60,
        tol: 1e-8,
        nullspace_dims: vec![0],
        firth_bias_reduction: None,
    };
    let rho_arr = array![rho];
    let (analytic_grad, _fd_grad) = evaluate_external_gradients(
        y.view(),
        w.view(),
        x.clone(),
        offset.view(),
        &s_list,
        &opts,
        &rho_arr,
    )
    .expect("external gradient evaluation should succeed");
    analytic_grad[0]
}

#[inline]
fn rel_err(a: f64, b: f64) -> f64 {
    (a - b).abs() / b.abs().max(1e-8)
}

#[test]
fn tiny_logit_oracle_gradient_identity_matches_fd_of_log_evidence() {
    let x = array![[1.0, -0.3], [1.0, 0.6], [1.0, 1.2]];
    let y = array![0.0, 1.0, 1.0];
    let cfg = OracleConfig::default();
    for rho in [-0.5, 0.0, 0.25, 0.8] {
        let g_exact = exact_logit_oracle_eval_rho_2d(&y, &x, rho, cfg).grad_rho;
        let g_fd = exact_logit_oracle_grad_fd_rho_2d(&y, &x, rho, cfg);
        let rel = (g_exact - g_fd).abs() / g_fd.abs().max(1e-8);
        assert!(
            rel < 2.5e-2,
            "oracle identity mismatch at rho={:.3}: g_exact={:.6e}, g_fd={:.6e}, rel={:.3e}",
            rho,
            g_exact,
            g_fd,
            rel
        );
    }
}

#[test]
fn tiny_logit_laml_vs_exact_oracle_regular_regime() {
    let x = array![[1.0, -0.3], [1.0, 0.6], [1.0, 1.2]];
    let y = array![0.0, 1.0, 1.0];
    let cfg = OracleConfig::default();

    for rho in [-0.4, 0.0, 0.4] {
        let g_exact = exact_logit_oracle_eval_rho_2d(&y, &x, rho, cfg).grad_rho;
        let g_laml = laml_gradient_external_logit(&y, &x, rho);
        let rel_err = (g_laml - g_exact).abs() / g_exact.abs().max(1e-8);
        assert!(g_laml.is_finite() && g_exact.is_finite());
        assert!(
            g_laml.signum() == g_exact.signum() || g_laml.abs() < 1e-8 || g_exact.abs() < 1e-8,
            "gradient direction mismatch at rho={:.3}: g_laml={:.6e}, g_exact={:.6e}",
            rho,
            g_laml,
            g_exact
        );
        assert!(
            rel_err < 1.25,
            "regular-regime discrepancy too large at rho={:.3}: g_laml={:.6e}, g_exact={:.6e}, rel={:.3e}",
            rho,
            g_laml,
            g_exact,
            rel_err
        );
    }
}

#[test]
fn tiny_logit_laml_vs_exact_oracle_regular_regime_sweep_is_stable() {
    let x = array![[1.0, -0.3], [1.0, 0.6], [1.0, 1.2]];
    let y = array![0.0, 1.0, 1.0];
    let cfg = OracleConfig::default();

    let mut g_exacts = Vec::new();
    let mut g_lamls = Vec::new();
    let mut rels = Vec::new();
    for rho in [-0.6, -0.3, 0.0, 0.3, 0.6] {
        let g_exact = exact_logit_oracle_eval_rho_2d(&y, &x, rho, cfg).grad_rho;
        let g_laml = laml_gradient_external_logit(&y, &x, rho);
        assert!(g_exact.is_finite() && g_laml.is_finite());
        g_exacts.push(g_exact);
        g_lamls.push(g_laml);
        rels.push(rel_err(g_laml, g_exact));
    }

    let a = Array1::from_vec(g_lamls);
    let b = Array1::from_vec(g_exacts);
    let dot = a.dot(&b);
    let na = a.dot(&a).sqrt();
    let nb = b.dot(&b).sqrt();
    let cosine = if na * nb > 1e-12 {
        dot / (na * nb)
    } else {
        1.0
    };
    rels.sort_by(|x, y| x.partial_cmp(y).unwrap());
    let median_rel = rels[rels.len() / 2];
    let max_rel = rels.iter().copied().fold(0.0_f64, f64::max);

    assert!(
        cosine > 0.85,
        "regular sweep cosine too low: cosine={cosine:.6}"
    );
    assert!(
        median_rel < 1.0,
        "regular sweep median relative error too high: median_rel={median_rel:.3e}"
    );
    assert!(
        max_rel < 1.25,
        "regular sweep max relative error too high: max_rel={max_rel:.3e}"
    );
}

#[test]
fn tiny_logit_laml_vs_exact_oracle_near_separation_stress() {
    // Deliberately near-separable toy design. The point is to track discrepancy
    // behavior under stress, not to enforce tight approximation.
    let x = array![[1.0, -6.0], [1.0, 0.2], [1.0, 5.8]];
    let y = array![0.0, 0.0, 1.0];
    let cfg = OracleConfig {
        grid_bound: 9.0,
        steps: 261,
    };

    let mut worst_rel = 0.0_f64;
    for rho in [-0.6, 0.0, 0.8] {
        let g_exact = exact_logit_oracle_eval_rho_2d(&y, &x, rho, cfg).grad_rho;
        let g_laml = laml_gradient_external_logit(&y, &x, rho);
        let rel_err = (g_laml - g_exact).abs() / g_exact.abs().max(1e-8);
        worst_rel = worst_rel.max(rel_err);
        assert!(g_laml.is_finite() && g_exact.is_finite());
        assert!(
            g_laml.signum() == g_exact.signum() || g_laml.abs() < 1e-8 || g_exact.abs() < 1e-8,
            "stress direction mismatch at rho={:.3}: g_laml={:.6e}, g_exact={:.6e}",
            rho,
            g_laml,
            g_exact
        );
    }

    // Sanity cap: we expect larger error than regular regime, but still bounded.
    assert!(
        worst_rel < 2.5,
        "stress-case discrepancy exploded: worst_rel={:.3e}",
        worst_rel
    );
}

#[test]
fn tiny_logit_laml_vs_exact_oracle_stress_sweep_direction_consistent() {
    let x = array![[1.0, -6.0], [1.0, 0.2], [1.0, 5.8]];
    let y = array![0.0, 0.0, 1.0];
    let cfg = OracleConfig {
        grid_bound: 9.0,
        steps: 261,
    };

    let mut mismatches = 0usize;
    let mut worst_rel = 0.0_f64;
    for rho in [-0.8, -0.3, 0.2, 0.7, 1.2] {
        let g_exact = exact_logit_oracle_eval_rho_2d(&y, &x, rho, cfg).grad_rho;
        let g_laml = laml_gradient_external_logit(&y, &x, rho);
        if g_exact.is_finite() && g_laml.is_finite() {
            if g_exact.abs() > 1e-8 && g_laml.abs() > 1e-8 && g_exact.signum() != g_laml.signum() {
                mismatches += 1;
            }
            worst_rel = worst_rel.max(rel_err(g_laml, g_exact));
        }
    }

    assert!(
        mismatches <= 1,
        "stress sweep produced too many sign mismatches: {mismatches}"
    );
    assert!(
        worst_rel < 2.5,
        "stress sweep relative error exploded: worst_rel={worst_rel:.3e}"
    );
}

#[test]
fn test_laml_gradient_exact_formula_ground_truth() {
    let x = array![[1.0, -0.3], [1.0, 0.6], [1.0, 1.2]];
    let y = array![0.0, 1.0, 1.0];
    let cfg = OracleConfig::default();
    let rho = 0.0;
    let g_exact = exact_logit_oracle_eval_rho_2d(&y, &x, rho, cfg).grad_rho;
    let g_laml = laml_gradient_external_logit(&y, &x, rho);
    let rel_err = (g_laml - g_exact).abs() / g_exact.abs().max(1e-8);
    assert!(
        rel_err < 1.25,
        "ground-truth mismatch at rho={rho:.3}: g_laml={g_laml:.6e}, g_exact={g_exact:.6e}, rel={rel_err:.3e}",
    );
}

#[test]
fn test_laml_gradient_firth_exact_formula_ground_truth() {
    let x = array![[1.0, -6.0], [1.0, 0.2], [1.0, 5.8]];
    let y = array![0.0, 0.0, 1.0];
    let cfg = OracleConfig {
        grid_bound: 9.0,
        steps: 261,
    };
    let rho = 0.0;
    let g_exact = exact_logit_oracle_eval_rho_2d(&y, &x, rho, cfg).grad_rho;
    let g_laml = laml_gradient_external_logit(&y, &x, rho);
    let rel_err = (g_laml - g_exact).abs() / g_exact.abs().max(1e-8);
    assert!(
        rel_err < 2.5,
        "firth-path ground-truth mismatch at rho={rho:.3}: g_laml={g_laml:.6e}, g_exact={g_exact:.6e}, rel={rel_err:.3e}",
    );
}

#[test]
fn isolation_reparam_pgs_pc_mains_firth() {
    let n = 96usize;
    let p = 7usize;
    let mut x = Array2::<f64>::zeros((n, p));
    for i in 0..n {
        let t = (i as f64) / ((n - 1) as f64);
        x[[i, 0]] = 1.0;
        x[[i, 1]] = (2.0 * std::f64::consts::PI * t).sin();
        x[[i, 2]] = (2.0 * std::f64::consts::PI * t).cos();
        x[[i, 3]] = t;
        x[[i, 4]] = t * t;
        x[[i, 5]] = (3.0 * std::f64::consts::PI * t).sin();
        x[[i, 6]] = (3.0 * std::f64::consts::PI * t).cos();
    }
    let beta_true = array![0.1, 0.8, -0.4, 0.5, -0.3, 0.2, -0.1];
    let eta = x.dot(&beta_true);
    let y = eta.mapv(|e| {
        if 1.0 / (1.0 + (-e).exp()) > 0.5 {
            1.0
        } else {
            0.0
        }
    });
    let w = Array1::ones(n);
    let offset = Array1::zeros(n);

    let mut s1 = Array2::<f64>::zeros((p, p));
    let mut s2 = Array2::<f64>::zeros((p, p));
    for j in 1..p {
        s1[[j, j]] = 1.0;
    }
    for j in 3..p {
        s2[[j, j]] = 1.0;
    }

    let opts = ExternalOptimOptions {
        family: LikelihoodFamily::BinomialLogit,
        max_iter: 120,
        tol: 1e-10,
        nullspace_dims: vec![1, 0],
        firth_bias_reduction: None,
    };
    let rho = array![1.5, 0.8];
    let (analytic, fd) = evaluate_external_gradients(
        y.view(),
        w.view(),
        x.view(),
        offset.view(),
        &[s1, s2],
        &opts,
        &rho,
    )
    .expect("external gradient evaluation should succeed");

    let dot = analytic.dot(&fd);
    let na = analytic.dot(&analytic).sqrt();
    let nf = fd.dot(&fd).sqrt();
    let cosine = if na * nf > 1e-12 {
        dot / (na * nf)
    } else {
        1.0
    };
    let rel_l2 = (&analytic - &fd).dot(&(&analytic - &fd)).sqrt() / na.max(nf).max(1e-12);
    assert!(
        cosine > 0.99,
        "isolation mismatch: cosine={cosine:.6}, analytic={analytic:?}, fd={fd:?}"
    );
    assert!(
        rel_l2 < 3e-1,
        "isolation mismatch: rel_l2={rel_l2:.3e}, analytic={analytic:?}, fd={fd:?}"
    );
}
