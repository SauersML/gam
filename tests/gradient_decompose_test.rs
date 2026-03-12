// Tests to find and prove math mistakes in the REML gradient code
use gam::estimate::{
    ExternalOptimOptions, evaluate_externalcost_andridge, evaluate_externalgradients,
};
use gam::types::LikelihoodFamily;
use ndarray::{s, Array1, Array2, Axis, Zip, array};
use ndarray_linalg::{Eigh, Inverse, UPLO};

// GAUSSIAN test: no third-derivative term, simpler gradient
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
    };
    let rho = array![1.5, 0.8];
    let (analytic, fd) = evaluate_externalgradients(
        y.view(),
        w.view(),
        x.view(),
        offset.view(),
        &[s1, s2],
        &opts,
        &rho,
    )
    .expect("Gaussian gradient evaluation should succeed");

    eprintln!("=== GAUSSIAN GRADIENT TEST ===");
    eprintln!("analytic = {:?}", analytic);
    eprintln!("fd       = {:?}", fd);
    let diff = &analytic - &fd;
    let rel = diff.mapv(|d| d.abs()) / fd.mapv(|f| f.abs().max(1e-12));
    eprintln!("diff     = {:?}", diff);
    eprintln!("rel_err  = {:?}", rel);

    for i in 0..analytic.len() {
        assert_eq!(
            analytic[i].signum(),
            fd[i].signum(),
            "Gaussian sign mismatch at i={i}: analytic={} fd={}",
            analytic[i],
            fd[i]
        );
    }
    let na = analytic.dot(&analytic).sqrt();
    let nf = fd.dot(&fd).sqrt();
    let cosine = analytic.dot(&fd) / (na * nf).max(1e-12);
    let rel_l2 = (&analytic - &fd).dot(&(&analytic - &fd)).sqrt() / na.max(nf).max(1e-12);
    assert!(cosine > 0.99, "Gaussian cosine={cosine:.6}");
    assert!(rel_l2 < 0.1, "Gaussian rel_l2={rel_l2:.3e}");
}

/// Standalone LAML cost for logit, computed from scratch using raw linear algebra.
/// No reparameterization, no ridge, no TK correction - just the textbook formula:
///   V(ρ) = -ℓ(β̂) + 0.5*β̂ᵀSβ̂ + 0.5*log|H| - 0.5*log|S|_+ + const
fn standalone_logit_laml_cost(
    y: &Array1<f64>,
    x: &Array2<f64>,
    s_list: &[Array2<f64>],
    rho: &Array1<f64>,
    nullspace_dims: &[usize],
) -> (f64, Array1<f64>) {
    let n = y.len();
    let p = x.ncols();
    let lambdas: Array1<f64> = rho.mapv(f64::exp);

    // Build total penalty S = Σ λ_k S_k
    let mut s_total = Array2::<f64>::zeros((p, p));
    for (k, s_k) in s_list.iter().enumerate() {
        s_total.scaled_add(lambdas[k], s_k);
    }

    // Solve PIRLS to convergence
    let mut beta = Array1::<f64>::zeros(p);
    for _iter in 0..200 {
        let eta = x.dot(&beta);
        let mu: Array1<f64> = eta.mapv(|e| {
            let ec = e.clamp(-500.0, 500.0);
            1.0 / (1.0 + (-ec).exp())
        });
        let w_diag: Array1<f64> = mu.mapv(|m| (m * (1.0 - m)).max(1e-12));
        let z: Array1<f64> = &eta + &(y - &mu) / &w_diag;

        // H = X^T W X + S
        let wx = &x * &w_diag.insert_axis(Axis(1));
        let h = wx.t().dot(x) + &s_total;
        // X^T W z
        let rhs = wx.t().dot(&z);

        let beta_new = h.inv().unwrap().dot(&rhs);
        let change = (&beta_new - &beta).dot(&(&beta_new - &beta)).sqrt();
        beta = beta_new;
        if change < 1e-12 {
            break;
        }
    }

    // Compute cost components at converged beta
    let eta = x.dot(&beta);
    let mu: Array1<f64> = eta.mapv(|e| {
        let ec = e.clamp(-500.0, 500.0);
        1.0 / (1.0 + (-ec).exp())
    });

    // Log-likelihood
    let log_lik: f64 = (0..n)
        .map(|i| {
            let ec = eta[i].clamp(-500.0, 500.0);
            y[i] * ec - (1.0 + ec.exp()).ln()
        })
        .sum();

    // Deviance = -2 * log_lik
    let deviance = -2.0 * log_lik;

    // Penalty term
    let penalty = beta.dot(&s_total.dot(&beta));

    // Working weights and Hessian
    let w_diag: Array1<f64> = mu.mapv(|m| (m * (1.0 - m)).max(1e-12));
    let wx = &x * &w_diag.insert_axis(Axis(1));
    let h = wx.t().dot(x) + &s_total;

    // log|H| via eigenvalues
    let (eigs_h, _) = h.eigh(UPLO::Lower).unwrap();
    let log_det_h: f64 = eigs_h.iter().filter(|&&v| v > 1e-14).map(|v| v.ln()).sum();

    // log|S|_+ (pseudo-determinant of penalty)
    let total_nullspace: usize = nullspace_dims.iter().sum();
    let penalty_rank = p - total_nullspace;
    let (eigs_s, _) = s_total.eigh(UPLO::Lower).unwrap();
    let mut eigs_sorted: Vec<f64> = eigs_s.to_vec();
    eigs_sorted.sort_by(|a, b| b.partial_cmp(a).unwrap());
    let log_det_s: f64 = eigs_sorted
        .iter()
        .take(penalty_rank)
        .map(|v| v.max(1e-14).ln())
        .sum();

    // Mp = number of unpenalized dimensions
    let mp = total_nullspace as f64;

    // LAML = ℓ - 0.5*penalty + 0.5*log|S| - 0.5*log|H| + (Mp/2)*log(2π)
    let laml = log_lik - 0.5 * penalty + 0.5 * log_det_s - 0.5 * log_det_h
        + (mp / 2.0) * (2.0 * std::f64::consts::PI).ln();

    // Cost = -laml
    let cost = -laml;

    // Now compute analytic gradient from scratch
    let h_inv = h.inv().unwrap();
    let mut grad = Array1::<f64>::zeros(rho.len());

    // Third derivative: c_i = dW_i/dη_i = μ(1-μ)(1-2μ) for logit
    let c_vec: Array1<f64> = mu.mapv(|m| m * (1.0 - m) * (1.0 - 2.0 * m));

    // Leverage h_i = x_i^T H^{-1} x_i
    let xh_inv = x.dot(&h_inv);
    let leverage: Array1<f64> = (0..n)
        .map(|i| xh_inv.row(i).dot(&x.row(i)))
        .collect::<Vec<_>>()
        .into();

    for k in 0..rho.len() {
        let s_k = &s_list[k];

        // Term 1: 0.5 * β̂^T A_k β̂ = 0.5 * λ_k * β̂^T S_k β̂
        let beta_term = 0.5 * lambdas[k] * beta.dot(&s_k.dot(&beta));

        // Term 2: 0.5 * tr(H^{-1} dH/dρ_k)
        // dH/dρ_k = λ_k S_k - λ_k X^T diag(c ⊙ Xv_k) X
        // where v_k = H^{-1}(S_k β̂)
        let s_k_beta = s_k.dot(&beta);
        let v_k = h_inv.dot(&s_k_beta);

        // tr(H^{-1} S_k)
        let trace_h_inv_s_k: f64 = (0..p)
            .map(|i| (0..p).map(|j| h_inv[[i, j]] * s_k[[j, i]]).sum::<f64>())
            .sum();

        // Σ_i c_i h_i (Xv_k)_i
        let x_v_k = x.dot(&v_k);
        let tracethird: f64 = (0..n).map(|i| c_vec[i] * leverage[i] * x_v_k[i]).sum();

        let trace_term = 0.5 * lambdas[k] * (trace_h_inv_s_k - tracethird);

        // Term 3: -0.5 * det1[k]
        // det1[k] = d/dρ_k log|S|_+ = λ_k * tr(S_+^{-1} S_k)
        // Using eigendecomposition of S_total
        let (eigs_s2, evecs_s) = s_total.eigh(UPLO::Lower).unwrap();
        let mut order: Vec<usize> = (0..p).collect();
        order.sort_by(|&a, &b| {
            eigs_s2[b]
                .partial_cmp(&eigs_s2[a])
                .unwrap_or(std::cmp::Ordering::Equal)
        });
        let mut det1_k = 0.0;
        for &idx in order.iter().take(penalty_rank) {
            let ev = eigs_s2[idx].max(1e-14);
            let u = evecs_s.column(idx).to_owned();
            let su = s_k.dot(&u);
            det1_k += u.dot(&su) / ev;
        }
        det1_k *= lambdas[k];
        let det1_term = -0.5 * det1_k;

        grad[k] = beta_term + trace_term + det1_term;
    }

    (cost, grad)
}

/// Test: standalone LAML computation matches the library's FD gradient.
/// This isolates math bugs in the REML gradient from infrastructure bugs.
#[test]
fn test_standalone_logit_laml_gradient_matches_fd() {
    let n = 40usize;
    let p = 4usize;
    let mut x = Array2::<f64>::zeros((n, p));
    for i in 0..n {
        let t = (i as f64) / ((n - 1) as f64);
        x[[i, 0]] = 1.0;
        x[[i, 1]] = t;
        x[[i, 2]] = (2.0 * std::f64::consts::PI * t).sin();
        x[[i, 3]] = (2.0 * std::f64::consts::PI * t).cos();
    }
    let beta_true = array![0.3, 1.0, -0.5, 0.2];
    let eta = x.dot(&beta_true);
    let y: Array1<f64> = eta.mapv(|e| {
        if 1.0 / (1.0 + (-e).exp()) > 0.5 {
            1.0
        } else {
            0.0
        }
    });

    let mut s1 = Array2::<f64>::zeros((p, p));
    for j in 1..p {
        s1[[j, j]] = 1.0;
    }
    let s_list = vec![s1];
    let nullspace_dims = vec![1usize];
    let rho = array![1.0];

    let (cost_base, grad_analytic) =
        standalone_logit_laml_cost(&y, &x, &s_list, &rho, &nullspace_dims);

    // FD gradient of the standalone cost
    let h = 1e-6;
    let mut fd_grad = Array1::<f64>::zeros(rho.len());
    for k in 0..rho.len() {
        let mut rho_p = rho.clone();
        let mut rho_m = rho.clone();
        rho_p[k] += h;
        rho_m[k] -= h;
        let (cost_p, _) = standalone_logit_laml_cost(&y, &x, &s_list, &rho_p, &nullspace_dims);
        let (cost_m, _) = standalone_logit_laml_cost(&y, &x, &s_list, &rho_m, &nullspace_dims);
        fd_grad[k] = (cost_p - cost_m) / (2.0 * h);
    }

    eprintln!("=== STANDALONE LAML GRADIENT TEST ===");
    eprintln!("cost_base     = {:.10}", cost_base);
    eprintln!("grad_analytic = {:?}", grad_analytic);
    eprintln!("fd_grad       = {:?}", fd_grad);
    let diff = &grad_analytic - &fd_grad;
    eprintln!("diff          = {:?}", diff);

    for k in 0..rho.len() {
        let rel_err = (grad_analytic[k] - fd_grad[k]).abs()
            / fd_grad[k].abs().max(grad_analytic[k].abs()).max(1e-12);
        assert!(
            rel_err < 1e-4,
            "Standalone LAML gradient mismatch at k={k}: analytic={:.8e} fd={:.8e} rel_err={:.3e}",
            grad_analytic[k],
            fd_grad[k],
            rel_err
        );
    }
}

/// Now compare the library's analytic gradient to the standalone analytic gradient.
/// Any discrepancy proves a bug in the library's implementation.
#[test]
fn test_library_gradient_vs_standalone_gradient() {
    let n = 40usize;
    let p = 4usize;
    let mut x = Array2::<f64>::zeros((n, p));
    for i in 0..n {
        let t = (i as f64) / ((n - 1) as f64);
        x[[i, 0]] = 1.0;
        x[[i, 1]] = t;
        x[[i, 2]] = (2.0 * std::f64::consts::PI * t).sin();
        x[[i, 3]] = (2.0 * std::f64::consts::PI * t).cos();
    }
    let beta_true = array![0.3, 1.0, -0.5, 0.2];
    let eta = x.dot(&beta_true);
    let y: Array1<f64> = eta.mapv(|e| {
        if 1.0 / (1.0 + (-e).exp()) > 0.5 {
            1.0
        } else {
            0.0
        }
    });

    let mut s1 = Array2::<f64>::zeros((p, p));
    for j in 1..p {
        s1[[j, j]] = 1.0;
    }

    let opts = ExternalOptimOptions {
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
    };
    let rho = array![1.0];

    // Library gradient
    let (lib_analytic, lib_fd) = evaluate_externalgradients(
        y.view(),
        Array1::ones(n).view(),
        x.view(),
        Array1::zeros(n).view(),
        &[s1.clone()],
        &opts,
        &rho,
    )
    .expect("should succeed");

    // Standalone gradient
    let (_, standalone_grad) =
        standalone_logit_laml_cost(&y, &x, &[s1.clone()], &rho, &[1]);

    // Also FD of the library cost
    let h = 1e-5;
    let mut manual_lib_fd = Array1::<f64>::zeros(rho.len());
    for k in 0..rho.len() {
        let mut rho_p = rho.clone();
        let mut rho_m = rho.clone();
        rho_p[k] += h;
        rho_m[k] -= h;
        let (cost_p, _) = evaluate_externalcost_andridge(
            y.view(),
            Array1::ones(n).view(),
            x.view(),
            Array1::zeros(n).view(),
            &[s1.clone()],
            &opts,
            &rho_p,
        )
        .unwrap();
        let (cost_m, _) = evaluate_externalcost_andridge(
            y.view(),
            Array1::ones(n).view(),
            x.view(),
            Array1::zeros(n).view(),
            &[s1.clone()],
            &opts,
            &rho_m,
        )
        .unwrap();
        manual_lib_fd[k] = (cost_p - cost_m) / (2.0 * h);
    }

    eprintln!("=== LIBRARY vs STANDALONE GRADIENT ===");
    eprintln!("library analytic = {:?}", lib_analytic);
    eprintln!("library FD       = {:?}", lib_fd);
    eprintln!("manual lib FD    = {:?}", manual_lib_fd);
    eprintln!("standalone grad  = {:?}", standalone_grad);
    eprintln!("lib_analytic - lib_fd       = {:?}", &lib_analytic - &lib_fd);
    eprintln!("standalone - lib_fd         = {:?}", &standalone_grad - &lib_fd);
    eprintln!("lib_analytic - standalone   = {:?}", &lib_analytic - &standalone_grad);

    // The key test: the standalone gradient (which we verified matches its own FD)
    // should also match the library's FD. If the library's analytic diverges from both,
    // that proves the bug is in the library's analytic gradient.
    for k in 0..rho.len() {
        let err_lib = (lib_analytic[k] - lib_fd[k]).abs();
        let err_standalone = (standalone_grad[k] - lib_fd[k]).abs();
        let scale = lib_fd[k].abs().max(1e-6);

        eprintln!(
            "coord {k}: |lib_analytic - lib_fd| = {:.3e}, |standalone - lib_fd| = {:.3e}, scale = {:.3e}",
            err_lib, err_standalone, scale
        );

        // If standalone matches FD well but library doesn't, the library has a bug
        // We expect standalone to match FD within ~1e-4 (FD accuracy)
        // The library analytic should also match, but if it doesn't, that's the bug
    }
}
