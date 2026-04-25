// Component checks for the base REML/LAML gradient algebra.
//
// Strategy: decompose the non-TK LAML gradient into independent components and
// verify each against finite differences. TK-specific diagnostics live with the
// runtime TK tests because production TK is a frozen-curvature surrogate rather
// than the textbook full chain-rule objective. The base cost here is:
//
//   V(ρ) = -ℓ(β̂) + 0.5*β̂ᵀSβ̂          [penalised fit, term A]
//        + 0.5*log|H|                    [Hessian logdet, term B]
//        - 0.5*log|S|_+                  [penalty logdet, term C]
//        + optional normalizing constants       [runtime fixed-dispersion omits these]
//        + prior(ρ)                      [soft prior on ρ]
//
// The analytic gradient should be:
//   gₖ = 0.5*β̂ᵀAₖβ̂                    [A: envelope theorem]
//      + 0.5*tr(H⁻¹Hₖ)                  [B: with third-derivative contraction]
//      - 0.5*det1[k]                     [C: penalty logdet derivative]
//      + dprior/dρₖ                      [included]

use faer::Side;
use gam::estimate::{
    ExternalOptimOptions, evaluate_externalcost_andridge, evaluate_externalgradient,
};
use gam::faer_ndarray::{FaerCholesky, FaerEigh};
use gam::smooth::BlockwisePenalty;
use gam::types::LikelihoodFamily;
use ndarray::{Array1, Array2, Axis, array};

// ─────────────────────────────────────────────────────────────────────────────
// Shared test data and helpers
// ─────────────────────────────────────────────────────────────────────────────

/// Build a standard test design: n=60, p columns, with intercept + polynomial + trig.
fn test_design(n: usize, p: usize) -> (Array2<f64>, Array1<f64>) {
    let mut x = Array2::<f64>::zeros((n, p));
    for i in 0..n {
        let t = (i as f64) / ((n - 1) as f64);
        x[[i, 0]] = 1.0;
        if p > 1 {
            x[[i, 1]] = t;
        }
        if p > 2 {
            x[[i, 2]] = t * t;
        }
        if p > 3 {
            x[[i, 3]] = (2.0 * std::f64::consts::PI * t).sin();
        }
        if p > 4 {
            x[[i, 4]] = (4.0 * std::f64::consts::PI * t).cos();
        }
    }
    let y: Array1<f64> = (0..n)
        .map(|i| if i % 3 == 0 { 1.0 } else { 0.0 })
        .collect::<Vec<_>>()
        .into();
    (x, y)
}

fn blockwise_penalties(s_list: Vec<Array2<f64>>) -> Vec<BlockwisePenalty> {
    let p = s_list[0].ncols();
    s_list
        .into_iter()
        .map(|s| BlockwisePenalty::new(0..p, s))
        .collect()
}

/// Central FD of a scalar function.
fn fd_gradient(f: &dyn Fn(&Array1<f64>) -> f64, rho: &Array1<f64>, h: f64) -> Array1<f64> {
    let k = rho.len();
    let mut grad = Array1::<f64>::zeros(k);
    for i in 0..k {
        let mut rp = rho.clone();
        let mut rm = rho.clone();
        rp[i] += h;
        rm[i] -= h;
        grad[i] = (f(&rp) - f(&rm)) / (2.0 * h);
    }
    grad
}

/// Central FD of the external REML cost across rho using the public cost API.
fn fd_gradient_from_externalcost(
    y: &Array1<f64>,
    w: &Array1<f64>,
    x: &Array2<f64>,
    offset: &Array1<f64>,
    s_list: &[BlockwisePenalty],
    opts: &ExternalOptimOptions,
    rho: &Array1<f64>,
    h: f64,
) -> Array1<f64> {
    let k = rho.len();
    let mut grad = Array1::<f64>::zeros(k);
    for i in 0..k {
        let mut rp = rho.clone();
        let mut rm = rho.clone();
        rp[i] += h;
        rm[i] -= h;
        let cp = evaluate_externalcost_andridge(
            y.view(),
            w.view(),
            x.view(),
            offset.view(),
            s_list,
            opts,
            &rp,
        )
        .map(|(c, _)| c)
        .expect("cost+");
        let cm = evaluate_externalcost_andridge(
            y.view(),
            w.view(),
            x.view(),
            offset.view(),
            s_list,
            opts,
            &rm,
        )
        .map(|(c, _)| c)
        .expect("cost-");
        grad[i] = (cp - cm) / (2.0 * h);
    }
    grad
}

/// Relative L2 error between two vectors.
fn rel_l2(a: &Array1<f64>, b: &Array1<f64>) -> f64 {
    let na = a.iter().map(|v| v * v).sum::<f64>().sqrt();
    let nb = b.iter().map(|v| v * v).sum::<f64>().sqrt();
    (a - b).iter().map(|v| v * v).sum::<f64>().sqrt() / na.max(nb).max(1e-12)
}

/// Solve PIRLS for logit to convergence, returning (beta, eta, mu, w, H, log_lik).
fn solve_pirls_logit(
    y: &Array1<f64>,
    x: &Array2<f64>,
    s_total: &Array2<f64>,
) -> (
    Array1<f64>,
    Array1<f64>,
    Array1<f64>,
    Array1<f64>,
    Array2<f64>,
    f64,
) {
    let n = y.len();
    let p = x.ncols();
    let mut beta = Array1::<f64>::zeros(p);
    for _ in 0..300 {
        let eta = x.dot(&beta);
        let mu: Array1<f64> = eta.mapv(|e| 1.0 / (1.0 + (-e.clamp(-500.0, 500.0)).exp()));
        let w: Array1<f64> = mu.mapv(|m| (m * (1.0 - m)).max(1e-12));
        let z: Array1<f64> = &eta + &((y - &mu) / &w);
        let wx = x * &w.clone().insert_axis(Axis(1));
        let h = wx.t().dot(x) + s_total;
        let rhs = wx.t().dot(&z);
        let h_inv = invert_spd(&h);
        let beta_new = h_inv.dot(&rhs);
        let change = (&beta_new - &beta)
            .iter()
            .map(|v| v * v)
            .sum::<f64>()
            .sqrt();
        beta = beta_new;
        if change < 1e-13 {
            break;
        }
    }
    let eta = x.dot(&beta);
    let mu: Array1<f64> = eta.mapv(|e| 1.0 / (1.0 + (-e.clamp(-500.0, 500.0)).exp()));
    let w: Array1<f64> = mu.mapv(|m| (m * (1.0 - m)).max(1e-12));
    let wx = x * &w.clone().insert_axis(Axis(1));
    let h = wx.t().dot(x) + s_total;
    let log_lik: f64 = (0..n)
        .map(|i| {
            let ec = eta[i].clamp(-500.0, 500.0);
            y[i] * ec - (1.0 + ec.exp()).ln()
        })
        .sum();
    (beta, eta, mu, w, h, log_lik)
}

/// Invert SPD matrix via Cholesky.
fn invert_spd(a: &Array2<f64>) -> Array2<f64> {
    let chol = a.cholesky(Side::Lower).expect("Matrix should be SPD");
    let p = a.ncols();
    let mut inv = Array2::<f64>::zeros((p, p));
    for j in 0..p {
        let mut e = Array1::<f64>::zeros(p);
        e[j] = 1.0;
        inv.column_mut(j).assign(&chol.solvevec(&e));
    }
    inv
}

/// Build S_total = Σ λ_k S_k.
fn build_s_total(s_list: &[BlockwisePenalty], rho: &Array1<f64>) -> Array2<f64> {
    let p = s_list[0].local.ncols();
    let mut s_total = Array2::<f64>::zeros((p, p));
    for (k, s_k) in s_list.iter().enumerate() {
        s_total.scaled_add(rho[k].exp(), &s_k.local);
    }
    s_total
}

// ─────────────────────────────────────────────────────────────────────────────
// COMPONENT-LEVEL TESTS: decompose the LAML into pieces
// ─────────────────────────────────────────────────────────────────────────────

/// Component A: Penalised fit term. -ℓ(β̂(ρ)) + 0.5*β̂(ρ)ᵀS(ρ)β̂(ρ).
/// By the envelope theorem, dA/dρ_k = 0.5*β̂ᵀAₖβ̂ = 0.5*λ_k*β̂ᵀS_kβ̂.
/// This should hold for any link, as it follows from PIRLS stationarity.
#[test]
fn test_component_a_envelope_theorem() {
    let (x, y) = test_design(60, 5);
    let mut s1 = Array2::<f64>::zeros((5, 5));
    for j in 1..5 {
        s1[[j, j]] = 1.0;
    }
    let s_list = blockwise_penalties(vec![s1]);
    let rho = array![2.0];

    let cost_a = |rho: &Array1<f64>| -> f64 {
        let s_total = build_s_total(&s_list, rho);
        let (beta, _, _, _, _, log_lik) = solve_pirls_logit(&y, &x, &s_total);
        -log_lik + 0.5 * beta.dot(&s_total.dot(&beta))
    };

    let fd_a = fd_gradient(&cost_a, &rho, 1e-5);

    // Analytic via envelope theorem
    let s_total = build_s_total(&s_list, &rho);
    let (beta, _, _, _, _, _) = solve_pirls_logit(&y, &x, &s_total);
    let mut analytic_a = Array1::<f64>::zeros(rho.len());
    for k in 0..rho.len() {
        analytic_a[k] = 0.5 * rho[k].exp() * beta.dot(&s_list[k].local.dot(&beta));
    }

    let err = rel_l2(&analytic_a, &fd_a);
    assert!(
        err < 1e-4,
        "Envelope theorem failed: rel_l2={err:.3e}\nanalytic={analytic_a:?}\nfd={fd_a:?}"
    );
}

/// Component B: Hessian log-determinant. 0.5*log|H(ρ)|.
/// Its derivative should be 0.5*tr(H⁻¹ dH/dρ_k).
/// For non-Gaussian: dH/dρ_k = λ_k*S_k - X^T diag(c ⊙ Xv_k) X
/// where v_k = H⁻¹(λ_k*S_k*β̂).
#[test]
fn test_component_b_hessian_logdet() {
    let (x, y) = test_design(60, 5);
    let mut s1 = Array2::<f64>::zeros((5, 5));
    for j in 1..5 {
        s1[[j, j]] = 1.0;
    }
    let s_list = blockwise_penalties(vec![s1]);
    let rho = array![2.0];
    let n = y.len();
    let p = x.ncols();

    // Cost: 0.5 * log|H(ρ)|, where H = X^T W(β̂(ρ)) X + S(ρ)
    let cost_b = |rho: &Array1<f64>| -> f64 {
        let s_total = build_s_total(&s_list, rho);
        let (_, _, _, _, h, _) = solve_pirls_logit(&y, &x, &s_total);
        let (eigs, _) = h.eigh(Side::Lower).unwrap();
        0.5 * eigs
            .iter()
            .filter(|v| **v > 1e-14)
            .map(|v| v.ln())
            .sum::<f64>()
    };

    let fd_b = fd_gradient(&cost_b, &rho, 1e-5);

    // Analytic: 0.5 * tr(H⁻¹ H_k)
    let s_total = build_s_total(&s_list, &rho);
    let (beta, _, mu, _, h, _) = solve_pirls_logit(&y, &x, &s_total);
    let h_inv = invert_spd(&h);
    let c_vec: Array1<f64> = mu.mapv(|m| m * (1.0 - m) * (1.0 - 2.0 * m));

    // Leverage: h_i = x_i^T H^{-1} x_i
    let xh_inv: Array2<f64> = x.dot(&h_inv);
    let leverage: Array1<f64> = (0..n)
        .map(|i| xh_inv.row(i).dot(&x.row(i)))
        .collect::<Vec<_>>()
        .into();

    let mut analytic_b = Array1::<f64>::zeros(rho.len());
    for k in 0..rho.len() {
        let lambda_k = rho[k].exp();
        let s_k = &s_list[k];

        // tr(H⁻¹ λ_k S_k)
        let trace_h_inv_s_k: f64 = (0..p)
            .map(|i| {
                (0..p)
                    .map(|j| h_inv[[i, j]] * s_k.local[[j, i]])
                    .sum::<f64>()
            })
            .sum();

        // v_k = H⁻¹(λ_k S_k β̂)
        let v_k = h_inv.dot(&(s_k.local.dot(&beta) * lambda_k));
        let x_v_k = x.dot(&v_k);
        let tracethird: f64 = (0..n).map(|i| c_vec[i] * leverage[i] * x_v_k[i]).sum();

        analytic_b[k] = 0.5 * (lambda_k * trace_h_inv_s_k - tracethird);
    }

    let err = rel_l2(&analytic_b, &fd_b);
    assert!(
        err < 1e-3,
        "Hessian logdet derivative failed: rel_l2={err:.3e}\nanalytic={analytic_b:?}\nfd={fd_b:?}"
    );
}

/// Component B, sub-piece B1: tr(H⁻¹ S_k) without the third-derivative term.
/// This isolates the simple trace from the contraction.
#[test]
fn test_component_b1_simple_trace() {
    let (x, y) = test_design(60, 5);
    let mut s1 = Array2::<f64>::zeros((5, 5));
    for j in 1..5 {
        s1[[j, j]] = 1.0;
    }
    let s_list = blockwise_penalties(vec![s1]);
    let rho = array![2.0];
    let p = x.ncols();

    // With β held constant (no implicit differentiation), 0.5*log|X^T W X + S|
    // has derivative 0.5*λ_k*tr(H⁻¹ S_k) with no third-derivative term.
    let s_total_base = build_s_total(&s_list, &rho);
    let (_, _, mu_fixed, _, _, _) = solve_pirls_logit(&y, &x, &s_total_base);
    let w_fixed: Array1<f64> = mu_fixed.mapv(|m| (m * (1.0 - m)).max(1e-12));
    let wx_fixed = &x * &w_fixed.clone().insert_axis(Axis(1));

    // Cost at fixed β: 0.5*log|X^T W_fixed X + S(ρ)|
    let cost_b1 = |rho: &Array1<f64>| -> f64 {
        let s_total = build_s_total(&s_list, rho);
        let h = wx_fixed.t().dot(&x) + &s_total;
        let (eigs, _) = h.eigh(Side::Lower).unwrap();
        0.5 * eigs
            .iter()
            .filter(|v| **v > 1e-14)
            .map(|v| v.ln())
            .sum::<f64>()
    };

    let fd_b1 = fd_gradient(&cost_b1, &rho, 1e-5);

    // Analytic: 0.5 * λ_k * tr(H⁻¹ S_k)
    let h_base = wx_fixed.t().dot(&x) + &s_total_base;
    let h_inv = invert_spd(&h_base);
    let mut analytic_b1 = Array1::<f64>::zeros(rho.len());
    for k in 0..rho.len() {
        let lambda_k = rho[k].exp();
        let s_k = &s_list[k];
        let trace: f64 = (0..p)
            .map(|i| {
                (0..p)
                    .map(|j| h_inv[[i, j]] * s_k.local[[j, i]])
                    .sum::<f64>()
            })
            .sum();
        analytic_b1[k] = 0.5 * lambda_k * trace;
    }

    let err = rel_l2(&analytic_b1, &fd_b1);
    assert!(
        err < 1e-4,
        "Simple Hessian trace failed: rel_l2={err:.3e}\nanalytic={analytic_b1:?}\nfd={fd_b1:?}"
    );
}

/// Component C: Penalty log-determinant. -0.5*log|S(ρ)|_+.
/// Its derivative is -0.5*λ_k*tr(S_+⁻¹ S_k).
#[test]
fn test_component_c_penalty_logdet() {
    let (x, _) = test_design(60, 5);
    let p = x.ncols();
    let mut s1 = Array2::<f64>::zeros((p, p));
    for j in 1..p {
        s1[[j, j]] = 1.0;
    }
    let s_list = blockwise_penalties(vec![s1]);
    let nullspace_dims = vec![1usize];
    let rho = array![2.0];

    let total_nullspace: usize = nullspace_dims.iter().sum();
    let penalty_rank = p - total_nullspace;

    let cost_c = |rho: &Array1<f64>| -> f64 {
        let s_total = build_s_total(&s_list, rho);
        let (eigs, _) = s_total.eigh(Side::Lower).unwrap();
        let mut sorted: Vec<f64> = eigs.to_vec();
        sorted.sort_by(|a, b| b.partial_cmp(a).unwrap());
        -0.5 * sorted
            .iter()
            .take(penalty_rank)
            .map(|v| v.max(1e-14).ln())
            .sum::<f64>()
    };

    let fd_c = fd_gradient(&cost_c, &rho, 1e-5);

    // Analytic
    let s_total = build_s_total(&s_list, &rho);
    let (eigs_s, evecs_s) = s_total.eigh(Side::Lower).unwrap();
    let mut eig_pairs: Vec<(usize, f64)> = (0..p).map(|i| (i, eigs_s[i])).collect();
    eig_pairs.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());
    let mut analytic_c = Array1::<f64>::zeros(rho.len());
    for k in 0..rho.len() {
        let lambda_k = rho[k].exp();
        let s_k = &s_list[k];
        let mut det1_k = 0.0;
        for &(idx, ev_raw) in eig_pairs.iter().take(penalty_rank) {
            let ev = ev_raw.max(1e-14);
            let u = evecs_s.column(idx).to_owned();
            det1_k += u.dot(&s_k.local.dot(&u)) / ev;
        }
        analytic_c[k] = -0.5 * lambda_k * det1_k;
    }

    let err = rel_l2(&analytic_c, &fd_c);
    assert!(
        err < 1e-4,
        "Penalty logdet derivative failed: rel_l2={err:.3e}\nanalytic={analytic_c:?}\nfd={fd_c:?}"
    );
}

// ─────────────────────────────────────────────────────────────────────────────
// FULL STANDALONE vs LIBRARY: identify which piece the library gets wrong
// ─────────────────────────────────────────────────────────────────────────────

/// Full standalone LAML cost (textbook formula, no TK, no prior, correct rank).
fn standalone_laml_cost(
    y: &Array1<f64>,
    x: &Array2<f64>,
    s_list: &[BlockwisePenalty],
    rho: &Array1<f64>,
    nullspace_dims: &[usize],
) -> f64 {
    let p = x.ncols();
    let s_total = build_s_total(s_list, rho);
    let (beta, _, _, _, h, log_lik) = solve_pirls_logit(y, x, &s_total);

    let penalty = beta.dot(&s_total.dot(&beta));
    let (eigs_h, _) = h.eigh(Side::Lower).unwrap();
    let log_det_h: f64 = eigs_h.iter().filter(|v| **v > 1e-14).map(|v| v.ln()).sum();

    let total_null: usize = nullspace_dims.iter().sum();
    let penalty_rank = p - total_null;
    let (eigs_s, _) = s_total.eigh(Side::Lower).unwrap();
    let mut sorted: Vec<f64> = eigs_s.to_vec();
    sorted.sort_by(|a, b| b.partial_cmp(a).unwrap());
    let log_det_s: f64 = sorted
        .iter()
        .take(penalty_rank)
        .map(|v| v.max(1e-14).ln())
        .sum();
    let mp = total_null as f64;

    let laml = log_lik - 0.5 * penalty + 0.5 * log_det_s - 0.5 * log_det_h
        + (mp / 2.0) * (2.0 * std::f64::consts::PI).ln();
    -laml
}

// ─────────────────────────────────────────────────────────────────────────────
// STRUCTURAL RANK ISOLATION: single vs overlapping penalties
// ─────────────────────────────────────────────────────────────────────────────

/// Single penalty: R has rank(S1) rows, structural_rank = min(rank, p) should be correct.
#[test]
fn test_single_penalty_logit_gradient() {
    let (x, y) = test_design(60, 5);
    let n = y.len();
    let mut s1 = Array2::<f64>::zeros((5, 5));
    for j in 1..5 {
        s1[[j, j]] = 1.0;
    }
    let s_list = blockwise_penalties(vec![s1]);

    let opts = ExternalOptimOptions {
        latent_cloglog: None,
        mixture_link: None,
        optimize_mixture: false,
        sas_link: None,
        optimize_sas: false,
        family: LikelihoodFamily::BinomialLogit,
        compute_inference: true,
        max_iter: 200,
        tol: 1e-12,
        nullspace_dims: vec![1],
        linear_constraints: None,
        firth_bias_reduction: None,
        penalty_shrinkage_floor: None,
        rho_prior: Default::default(),
        kronecker_penalty_system: None,
        kronecker_factored: None,
    };
    let rho = array![2.0];
    let w = Array1::ones(n);
    let offset = Array1::zeros(n);
    let analytic = evaluate_externalgradient(
        y.view(),
        w.view(),
        x.view(),
        offset.view(),
        &s_list,
        &opts,
        &rho,
    )
    .unwrap();
    let fd = fd_gradient_from_externalcost(&y, &w, &x, &offset, &s_list, &opts, &rho, 1e-5);

    let err = rel_l2(&analytic, &fd);
    assert!(
        err < 0.05,
        "Single-penalty logit gradient wrong: rel_l2={err:.3e}\nanalytic={analytic:?}\nfd={fd:?}"
    );
}

/// Two OVERLAPPING penalties (S1 penalizes cols 1-4, S2 penalizes cols 3-4).
/// structural_rank = min(4+2, 5) = 5 but actual rank = 4.
#[test]
fn test_two_overlapping_penalties_logit_gradient() {
    let (x, y) = test_design(60, 5);
    let n = y.len();
    let mut s1 = Array2::<f64>::zeros((5, 5));
    let mut s2 = Array2::<f64>::zeros((5, 5));
    for j in 1..5 {
        s1[[j, j]] = 1.0;
    }
    for j in 3..5 {
        s2[[j, j]] = 1.0;
    }
    let s_list = blockwise_penalties(vec![s1, s2]);

    let opts = ExternalOptimOptions {
        latent_cloglog: None,
        mixture_link: None,
        optimize_mixture: false,
        sas_link: None,
        optimize_sas: false,
        family: LikelihoodFamily::BinomialLogit,
        compute_inference: true,
        max_iter: 200,
        tol: 1e-12,
        nullspace_dims: vec![1, 0],
        linear_constraints: None,
        firth_bias_reduction: None,
        penalty_shrinkage_floor: None,
        rho_prior: Default::default(),
        kronecker_penalty_system: None,
        kronecker_factored: None,
    };
    let rho = array![2.0, -1.0];
    let w = Array1::ones(n);
    let offset = Array1::zeros(n);
    let analytic = evaluate_externalgradient(
        y.view(),
        w.view(),
        x.view(),
        offset.view(),
        &s_list,
        &opts,
        &rho,
    )
    .unwrap();
    let fd = fd_gradient_from_externalcost(&y, &w, &x, &offset, &s_list, &opts, &rho, 1e-5);

    let err = rel_l2(&analytic, &fd);
    assert!(
        err < 0.05,
        "Overlapping-penalty logit gradient wrong: rel_l2={err:.3e}\nanalytic={analytic:?}\nfd={fd:?}"
    );
}

/// Two NON-OVERLAPPING penalties (S1 penalizes cols 1-2, S2 penalizes cols 3-4).
/// structural_rank = min(2+2, 5) = 4, actual rank = 4. Should be correct.
#[test]
fn test_two_nonoverlapping_penalties_logit_gradient() {
    let (x, y) = test_design(60, 5);
    let n = y.len();
    let mut s1 = Array2::<f64>::zeros((5, 5));
    let mut s2 = Array2::<f64>::zeros((5, 5));
    for j in 1..3 {
        s1[[j, j]] = 1.0;
    }
    for j in 3..5 {
        s2[[j, j]] = 1.0;
    }
    let s_list = blockwise_penalties(vec![s1, s2]);

    let opts = ExternalOptimOptions {
        latent_cloglog: None,
        mixture_link: None,
        optimize_mixture: false,
        sas_link: None,
        optimize_sas: false,
        family: LikelihoodFamily::BinomialLogit,
        compute_inference: true,
        max_iter: 200,
        tol: 1e-12,
        nullspace_dims: vec![1, 0], // S1 has nullspace of 1 (intercept)
        linear_constraints: None,
        firth_bias_reduction: None,
        penalty_shrinkage_floor: None,
        rho_prior: Default::default(),
        kronecker_penalty_system: None,
        kronecker_factored: None,
    };
    let rho = array![2.0, -1.0];
    let w = Array1::ones(n);
    let offset = Array1::zeros(n);
    let analytic = evaluate_externalgradient(
        y.view(),
        w.view(),
        x.view(),
        offset.view(),
        &s_list,
        &opts,
        &rho,
    )
    .unwrap();
    let fd = fd_gradient_from_externalcost(&y, &w, &x, &offset, &s_list, &opts, &rho, 1e-5);

    let err = rel_l2(&analytic, &fd);
    assert!(
        err < 0.05,
        "Non-overlapping-penalty logit gradient wrong: rel_l2={err:.3e}\nanalytic={analytic:?}\nfd={fd:?}"
    );
}

// ─────────────────────────────────────────────────────────────────────────────
// GAUSSIAN CONTROL: verify gradient is correct for Gaussian (no third-deriv)
// ─────────────────────────────────────────────────────────────────────────────

#[test]
fn test_gaussian_gradient_vs_fd() {
    let n = 60usize;
    let p = 5usize;
    let mut x = Array2::<f64>::zeros((n, p));
    let mut y = Array1::<f64>::zeros(n);
    for i in 0..n {
        let t = (i as f64) / ((n - 1) as f64);
        x[[i, 0]] = 1.0;
        x[[i, 1]] = t;
        x[[i, 2]] = t * t;
        x[[i, 3]] = (2.0 * std::f64::consts::PI * t).sin();
        x[[i, 4]] = (2.0 * std::f64::consts::PI * t).cos();
        y[i] = 0.5
            + 0.3 * t
            + 0.1 * (2.0 * std::f64::consts::PI * t).sin()
            + 0.05 * ((i as f64) * 0.1).sin();
    }
    let mut s1 = Array2::<f64>::zeros((p, p));
    let mut s2 = Array2::<f64>::zeros((p, p));
    for j in 1..p {
        s1[[j, j]] = 1.0;
    }
    for j in 3..p {
        s2[[j, j]] = 1.0;
    }
    let s_list = blockwise_penalties(vec![s1, s2]);

    let opts = ExternalOptimOptions {
        latent_cloglog: None,
        mixture_link: None,
        optimize_mixture: false,
        sas_link: None,
        optimize_sas: false,
        family: LikelihoodFamily::GaussianIdentity,
        compute_inference: true,
        max_iter: 120,
        tol: 1e-10,
        nullspace_dims: vec![1, 0],
        linear_constraints: None,
        firth_bias_reduction: None,
        penalty_shrinkage_floor: None,
        rho_prior: Default::default(),
        kronecker_penalty_system: None,
        kronecker_factored: None,
    };
    let rho = array![1.5, 0.8];
    let w = Array1::ones(n);
    let offset = Array1::zeros(n);
    let analytic = evaluate_externalgradient(
        y.view(),
        w.view(),
        x.view(),
        offset.view(),
        &s_list,
        &opts,
        &rho,
    )
    .unwrap();
    let fd = fd_gradient_from_externalcost(&y, &w, &x, &offset, &s_list, &opts, &rho, 1e-5);

    let err = rel_l2(&analytic, &fd);
    assert!(
        err < 1e-3,
        "Gaussian gradient wrong: rel_l2={err:.3e}\nanalytic={analytic:?}\nfd={fd:?}"
    );
}

// ─────────────────────────────────────────────────────────────────────────────
// PROBIT: does the same bug affect other non-Gaussian links?
// ─────────────────────────────────────────────────────────────────────────────

#[test]
fn test_probit_single_penalty_gradient() {
    let (x, y) = test_design(60, 5);
    let n = y.len();
    let mut s1 = Array2::<f64>::zeros((5, 5));
    for j in 1..5 {
        s1[[j, j]] = 1.0;
    }
    let s_list = blockwise_penalties(vec![s1]);

    let opts = ExternalOptimOptions {
        latent_cloglog: None,
        mixture_link: None,
        optimize_mixture: false,
        sas_link: None,
        optimize_sas: false,
        family: LikelihoodFamily::BinomialProbit,
        compute_inference: true,
        max_iter: 200,
        tol: 1e-12,
        nullspace_dims: vec![1],
        linear_constraints: None,
        firth_bias_reduction: None,
        penalty_shrinkage_floor: None,
        rho_prior: Default::default(),
        kronecker_penalty_system: None,
        kronecker_factored: None,
    };
    let rho = array![2.0];
    let w = Array1::ones(n);
    let offset = Array1::zeros(n);
    let analytic = evaluate_externalgradient(
        y.view(),
        w.view(),
        x.view(),
        offset.view(),
        &s_list,
        &opts,
        &rho,
    )
    .unwrap();
    let fd = fd_gradient_from_externalcost(&y, &w, &x, &offset, &s_list, &opts, &rho, 1e-5);

    let err = rel_l2(&analytic, &fd);
    assert!(
        err < 0.05,
        "Probit single-penalty gradient wrong: rel_l2={err:.3e}\nanalytic={analytic:?}\nfd={fd:?}"
    );
}

#[test]
fn test_probit_overlapping_penalties_gradient() {
    let (x, y) = test_design(60, 5);
    let n = y.len();
    let mut s1 = Array2::<f64>::zeros((5, 5));
    let mut s2 = Array2::<f64>::zeros((5, 5));
    for j in 1..5 {
        s1[[j, j]] = 1.0;
    }
    for j in 3..5 {
        s2[[j, j]] = 1.0;
    }
    let s_list = blockwise_penalties(vec![s1, s2]);

    let opts = ExternalOptimOptions {
        latent_cloglog: None,
        mixture_link: None,
        optimize_mixture: false,
        sas_link: None,
        optimize_sas: false,
        family: LikelihoodFamily::BinomialProbit,
        compute_inference: true,
        max_iter: 200,
        tol: 1e-12,
        nullspace_dims: vec![1, 0],
        linear_constraints: None,
        firth_bias_reduction: None,
        penalty_shrinkage_floor: None,
        rho_prior: Default::default(),
        kronecker_penalty_system: None,
        kronecker_factored: None,
    };
    let rho = array![2.0, -1.0];
    let w = Array1::ones(n);
    let offset = Array1::zeros(n);
    let analytic = evaluate_externalgradient(
        y.view(),
        w.view(),
        x.view(),
        offset.view(),
        &s_list,
        &opts,
        &rho,
    )
    .unwrap();
    let fd = fd_gradient_from_externalcost(&y, &w, &x, &offset, &s_list, &opts, &rho, 1e-5);

    let err = rel_l2(&analytic, &fd);
    assert!(
        err < 0.05,
        "Probit overlapping-penalty gradient wrong: rel_l2={err:.3e}\nanalytic={analytic:?}\nfd={fd:?}"
    );
}

// ─────────────────────────────────────────────────────────────────────────────
// STANDALONE FORMULA VALIDATION: verify our textbook formula is correct
// ─────────────────────────────────────────────────────────────────────────────

/// Standalone analytic gradient matches its own FD, proving the formula is correct.
#[test]
fn test_standalone_gradient_matches_own_fd() {
    let (x, y) = test_design(60, 5);
    let mut s1 = Array2::<f64>::zeros((5, 5));
    let mut s2 = Array2::<f64>::zeros((5, 5));
    for j in 1..5 {
        s1[[j, j]] = 1.0;
    }
    for j in 3..5 {
        s2[[j, j]] = 1.0;
    }
    let s_list = blockwise_penalties(vec![s1, s2]);
    let rho = array![2.0, -1.0];
    let p = x.ncols();
    let n = y.len();

    let standalone_fd = fd_gradient(
        &|rho: &Array1<f64>| standalone_laml_cost(&y, &x, &s_list, rho, &[1, 0]),
        &rho,
        1e-5,
    );

    // Analytic
    let s_total = build_s_total(&s_list, &rho);
    let (beta, _, mu, _, h, _) = solve_pirls_logit(&y, &x, &s_total);
    let h_inv = invert_spd(&h);
    let c_vec: Array1<f64> = mu.mapv(|m| m * (1.0 - m) * (1.0 - 2.0 * m));
    let xh_inv: Array2<f64> = x.dot(&h_inv);
    let leverage: Array1<f64> = (0..n)
        .map(|i| xh_inv.row(i).dot(&x.row(i)))
        .collect::<Vec<_>>()
        .into();

    let total_null = 1usize;
    let penalty_rank = p - total_null;
    let (eigs_s, evecs_s) = s_total.eigh(Side::Lower).unwrap();
    let mut eig_pairs: Vec<(usize, f64)> = (0..p).map(|i| (i, eigs_s[i])).collect();
    eig_pairs.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());

    let mut analytic = Array1::<f64>::zeros(rho.len());
    for k in 0..rho.len() {
        let lambda_k = rho[k].exp();
        let s_k = &s_list[k];

        // Term A
        let term_a = 0.5 * lambda_k * beta.dot(&s_k.local.dot(&beta));

        // Term B
        let trace_h_inv_s_k: f64 = (0..p)
            .map(|i| {
                (0..p)
                    .map(|j| h_inv[[i, j]] * s_k.local[[j, i]])
                    .sum::<f64>()
            })
            .sum();
        let v_k = h_inv.dot(&(s_k.local.dot(&beta) * lambda_k));
        let x_v_k = x.dot(&v_k);
        let tracethird: f64 = (0..n).map(|i| c_vec[i] * leverage[i] * x_v_k[i]).sum();
        let term_b = 0.5 * (lambda_k * trace_h_inv_s_k - tracethird);

        // Term C
        let mut det1_k = 0.0;
        for &(idx, ev_raw) in eig_pairs.iter().take(penalty_rank) {
            let ev = ev_raw.max(1e-14);
            let u = evecs_s.column(idx).to_owned();
            det1_k += u.dot(&s_k.local.dot(&u)) / ev;
        }
        let term_c = -0.5 * lambda_k * det1_k;

        analytic[k] = term_a + term_b + term_c;
    }

    let err = rel_l2(&analytic, &standalone_fd);
    assert!(
        err < 1e-4,
        "Standalone gradient formula is wrong: rel_l2={err:.3e}\nanalytic={analytic:?}\nstandalone_fd={standalone_fd:?}"
    );
}
