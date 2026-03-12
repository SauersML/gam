// Tests to find and prove math mistakes in the REML gradient code
use faer::Side;
use gam::estimate::{
    ExternalOptimOptions, evaluate_externalcost_andridge, evaluate_externalgradients,
};
use gam::faer_ndarray::{FaerCholesky, FaerEigh};
use gam::types::LikelihoodFamily;
use ndarray::{Array1, Array2, Axis, array};

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
    let na = analytic.iter().map(|v| v * v).sum::<f64>().sqrt();
    let nf = fd.iter().map(|v| v * v).sum::<f64>().sqrt();
    let cosine = analytic.dot(&fd) / (na * nf).max(1e-12);
    let rel_l2 =
        (&analytic - &fd).iter().map(|v| v * v).sum::<f64>().sqrt() / na.max(nf).max(1e-12);
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
    for _ in 0..200 {
        let eta = x.dot(&beta);
        let mu: Array1<f64> = eta.mapv(|e| {
            let ec = e.clamp(-500.0, 500.0);
            1.0 / (1.0 + (-ec).exp())
        });
        let w_diag: Array1<f64> = mu.mapv(|m| (m * (1.0 - m)).max(1e-12));
        let z: Array1<f64> = &eta + &((y - &mu) / &w_diag);

        // H = X^T W X + S
        let wx = x * &w_diag.clone().insert_axis(Axis(1));
        let h = wx.t().dot(x) + &s_total;
        // X^T W z
        let rhs = wx.t().dot(&z);

        let h_inv = invert_spd(&h);
        let beta_new = h_inv.dot(&rhs);
        let change = (&beta_new - &beta)
            .iter()
            .map(|v| v * v)
            .sum::<f64>()
            .sqrt();
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

    // Penalty term
    let penalty = beta.dot(&s_total.dot(&beta));

    // Working weights and Hessian
    let w_diag: Array1<f64> = mu.mapv(|m| (m * (1.0 - m)).max(1e-12));
    let wx = x * &w_diag.clone().insert_axis(Axis(1));
    let h = wx.t().dot(x) + &s_total;

    // log|H| via eigenvalues
    let (eigs_h, _) = h.eigh(Side::Lower).unwrap();
    let log_det_h: f64 = eigs_h.iter().filter(|v| **v > 1e-14).map(|v| v.ln()).sum();

    // log|S|_+ (pseudo-determinant of penalty)
    let total_nullspace: usize = nullspace_dims.iter().sum();
    let penalty_rank = p - total_nullspace;
    let (eigs_s, _) = s_total.eigh(Side::Lower).unwrap();
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
    let h_inv = invert_spd(&h);
    let mut grad = Array1::<f64>::zeros(rho.len());

    // Third derivative: c_i = dW_i/dη_i = μ(1-μ)(1-2μ) for logit
    let c_vec: Array1<f64> = mu.mapv(|m| m * (1.0 - m) * (1.0 - 2.0 * m));

    // Leverage h_i = x_i^T H^{-1} x_i
    let xh_inv: Array2<f64> = x.dot(&h_inv);
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
        // where v_k = H^{-1}(λ_k S_k β̂)
        let s_k_beta = s_k.dot(&beta);
        let v_k = h_inv.dot(&(&s_k_beta * lambdas[k]));

        // tr(H^{-1} λ_k S_k) = λ_k tr(H^{-1} S_k)
        let trace_h_inv_s_k: f64 = (0..p)
            .map(|i| (0..p).map(|j| h_inv[[i, j]] * s_k[[j, i]]).sum::<f64>())
            .sum();

        // tr(H^{-1} X^T diag(c ⊙ Xv_k) X) = Σ_i c_i h_i (Xv_k)_i
        let x_v_k = x.dot(&v_k);
        let tracethird: f64 = (0..n).map(|i| c_vec[i] * leverage[i] * x_v_k[i]).sum();

        // H_k = λ_k S_k - X^T diag(c ⊙ Xv_k) X  [note: v_k already includes λ_k]
        let trace_h_inv_h_k = lambdas[k] * trace_h_inv_s_k - tracethird;
        let hessian_term = 0.5 * trace_h_inv_h_k;

        // Term 3: -0.5 * det1[k]
        // det1[k] = λ_k * tr(S_+^{-1} S_k)
        let (eigs_s2, evecs_s) = s_total.eigh(Side::Lower).unwrap();
        let mut eig_pairs: Vec<(usize, f64)> = (0..p).map(|i| (i, eigs_s2[i])).collect();
        eig_pairs.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
        let mut det1_k = 0.0;
        for &(idx, ev_raw) in eig_pairs.iter().take(penalty_rank) {
            let ev = ev_raw.max(1e-14);
            let u = evecs_s.column(idx).to_owned();
            let su = s_k.dot(&u);
            det1_k += u.dot(&su) / ev;
        }
        det1_k *= lambdas[k];
        let det1_term = -0.5 * det1_k;

        grad[k] = beta_term + hessian_term + det1_term;
    }

    (cost, grad)
}

/// Helper: invert SPD matrix via Cholesky
fn invert_spd(a: &Array2<f64>) -> Array2<f64> {
    let chol = a.cholesky(Side::Lower).expect("Matrix should be SPD");
    let p = a.ncols();
    let mut inv = Array2::<f64>::zeros((p, p));
    for j in 0..p {
        let mut e = Array1::<f64>::zeros(p);
        e[j] = 1.0;
        let col = chol.solvevec(&e);
        inv.column_mut(j).assign(&col);
    }
    inv
}

/// Test: standalone LAML gradient matches its own FD.
/// This validates the textbook formula before comparing to the library.
#[test]
fn test_standalone_logit_laml_gradient_matches_own_fd() {
    let n = 60usize;
    let p = 5usize;
    let mut x = Array2::<f64>::zeros((n, p));
    // Use a deterministic but non-trivial test point
    for i in 0..n {
        let t = (i as f64) / ((n - 1) as f64);
        x[[i, 0]] = 1.0;
        x[[i, 1]] = t;
        x[[i, 2]] = t * t;
        x[[i, 3]] = (2.0 * std::f64::consts::PI * t).sin();
        x[[i, 4]] = (4.0 * std::f64::consts::PI * t).cos();
    }
    // y with a clear signal to avoid near-zero gradients
    let y: Array1<f64> = (0..n)
        .map(|i| if i % 3 == 0 { 1.0 } else { 0.0 })
        .collect::<Vec<_>>()
        .into();

    let mut s1 = Array2::<f64>::zeros((p, p));
    let mut s2 = Array2::<f64>::zeros((p, p));
    for j in 1..p {
        s1[[j, j]] = 1.0;
    }
    for j in 3..p {
        s2[[j, j]] = 1.0;
    }
    let s_list = vec![s1, s2];
    let nullspace_dims = vec![1usize, 0];
    // Use rho away from the optimum to get substantial gradients
    let rho = array![2.0, -1.0];

    let (cost_base, grad_analytic) =
        standalone_logit_laml_cost(&y, &x, &s_list, &rho, &nullspace_dims);

    // FD gradient of the standalone cost
    let h = 1e-5;
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
        eprintln!(
            "  coord {k}: analytic={:.8e} fd={:.8e} rel_err={:.3e}",
            grad_analytic[k], fd_grad[k], rel_err
        );
        assert!(
            rel_err < 1e-4,
            "Standalone LAML gradient mismatch at k={k}: analytic={:.8e} fd={:.8e} rel_err={:.3e}",
            grad_analytic[k],
            fd_grad[k],
            rel_err
        );
    }
}

/// Compare library analytic gradient to the library's own FD gradient for logit.
/// Any large discrepancy proves a bug in the library's analytic gradient.
#[test]
fn test_library_logit_gradient_vs_fd() {
    let n = 60usize;
    let p = 5usize;
    let mut x = Array2::<f64>::zeros((n, p));
    for i in 0..n {
        let t = (i as f64) / ((n - 1) as f64);
        x[[i, 0]] = 1.0;
        x[[i, 1]] = t;
        x[[i, 2]] = t * t;
        x[[i, 3]] = (2.0 * std::f64::consts::PI * t).sin();
        x[[i, 4]] = (4.0 * std::f64::consts::PI * t).cos();
    }
    let y: Array1<f64> = (0..n)
        .map(|i| if i % 3 == 0 { 1.0 } else { 0.0 })
        .collect::<Vec<_>>()
        .into();

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
        family: LikelihoodFamily::BinomialLogit,
        compute_inference: true,
        max_iter: 200,
        tol: 1e-12,
        nullspace_dims: vec![1, 0],
        linear_constraints: None,
        firth_bias_reduction: None,
    };
    let rho = array![2.0, -1.0];

    let (lib_analytic, lib_fd) = evaluate_externalgradients(
        y.view(),
        Array1::ones(n).view(),
        x.view(),
        Array1::zeros(n).view(),
        &[s1, s2],
        &opts,
        &rho,
    )
    .expect("should succeed");

    eprintln!("=== LIBRARY LOGIT GRADIENT VS FD ===");
    eprintln!("lib_analytic = {:?}", lib_analytic);
    eprintln!("lib_fd       = {:?}", lib_fd);
    let diff = &lib_analytic - &lib_fd;
    eprintln!("diff         = {:?}", diff);
    for k in 0..rho.len() {
        let rel_err = (lib_analytic[k] - lib_fd[k]).abs()
            / lib_fd[k].abs().max(lib_analytic[k].abs()).max(1e-12);
        eprintln!(
            "  coord {k}: analytic={:.8e} fd={:.8e} rel_err={:.3e}",
            lib_analytic[k], lib_fd[k], rel_err
        );
    }

    // This SHOULD pass if the analytic gradient is correct.
    // Currently expected to FAIL, proving the bug.
    let na = lib_analytic.iter().map(|v| v * v).sum::<f64>().sqrt();
    let nf = lib_fd.iter().map(|v| v * v).sum::<f64>().sqrt();
    let cosine = lib_analytic.dot(&lib_fd) / (na * nf).max(1e-12);
    let rel_l2 = (&lib_analytic - &lib_fd)
        .iter()
        .map(|v| v * v)
        .sum::<f64>()
        .sqrt()
        / na.max(nf).max(1e-12);
    eprintln!("cosine = {cosine:.6}, rel_l2 = {rel_l2:.3e}");
    assert!(
        cosine > 0.99,
        "Library logit gradient direction wrong: cosine={cosine:.6}"
    );
    assert!(
        rel_l2 < 0.05,
        "Library logit gradient magnitude wrong: rel_l2={rel_l2:.3e}"
    );
}

/// Compare library analytic gradient to standalone analytic gradient.
/// Also compare both to the library's FD gradient to triangulate which is wrong.
#[test]
fn test_library_gradient_vs_standalone_gradient() {
    let n = 60usize;
    let p = 5usize;
    let mut x = Array2::<f64>::zeros((n, p));
    for i in 0..n {
        let t = (i as f64) / ((n - 1) as f64);
        x[[i, 0]] = 1.0;
        x[[i, 1]] = t;
        x[[i, 2]] = t * t;
        x[[i, 3]] = (2.0 * std::f64::consts::PI * t).sin();
        x[[i, 4]] = (4.0 * std::f64::consts::PI * t).cos();
    }
    let y: Array1<f64> = (0..n)
        .map(|i| if i % 3 == 0 { 1.0 } else { 0.0 })
        .collect::<Vec<_>>()
        .into();

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
        family: LikelihoodFamily::BinomialLogit,
        compute_inference: true,
        max_iter: 200,
        tol: 1e-12,
        nullspace_dims: vec![1, 0],
        linear_constraints: None,
        firth_bias_reduction: None,
    };
    let rho = array![2.0, -1.0];

    // Library gradient
    let (lib_analytic, lib_fd) = evaluate_externalgradients(
        y.view(),
        Array1::ones(n).view(),
        x.view(),
        Array1::zeros(n).view(),
        &[s1.clone(), s2.clone()],
        &opts,
        &rho,
    )
    .expect("should succeed");

    // Standalone gradient
    let (standalone_cost, standalone_grad) =
        standalone_logit_laml_cost(&y, &x, &[s1.clone(), s2.clone()], &rho, &[1, 0]);

    // Also FD of the library cost for a clean comparison
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
            &[s1.clone(), s2.clone()],
            &opts,
            &rho_p,
        )
        .unwrap();
        let (cost_m, _) = evaluate_externalcost_andridge(
            y.view(),
            Array1::ones(n).view(),
            x.view(),
            Array1::zeros(n).view(),
            &[s1.clone(), s2.clone()],
            &opts,
            &rho_m,
        )
        .unwrap();
        manual_lib_fd[k] = (cost_p - cost_m) / (2.0 * h);
    }

    // Compute library cost at base point
    let (lib_cost_base, _) = evaluate_externalcost_andridge(
        y.view(),
        Array1::ones(n).view(),
        x.view(),
        Array1::zeros(n).view(),
        &[s1.clone(), s2.clone()],
        &opts,
        &rho,
    )
    .unwrap();

    eprintln!("=== LIBRARY vs STANDALONE GRADIENT ===");
    eprintln!("lib_cost_base    = {:.10}", lib_cost_base);
    eprintln!("standalone_cost  = {:.10}", standalone_cost);
    eprintln!(
        "cost_diff        = {:.10e}",
        lib_cost_base - standalone_cost
    );
    eprintln!("library analytic = {:?}", lib_analytic);
    eprintln!("library FD       = {:?}", lib_fd);
    eprintln!("manual lib FD    = {:?}", manual_lib_fd);
    eprintln!("standalone grad  = {:?}", standalone_grad);
    eprintln!(
        "lib_analytic - lib_fd       = {:?}",
        &lib_analytic - &lib_fd
    );
    eprintln!(
        "standalone - manual_lib_fd  = {:?}",
        &standalone_grad - &manual_lib_fd
    );
    eprintln!(
        "lib_analytic - standalone   = {:?}",
        &lib_analytic - &standalone_grad
    );

    for k in 0..rho.len() {
        let err_lib_vs_fd = (lib_analytic[k] - lib_fd[k]).abs() / lib_fd[k].abs().max(1e-6);
        let err_standalone_vs_fd =
            (standalone_grad[k] - manual_lib_fd[k]).abs() / manual_lib_fd[k].abs().max(1e-6);
        let err_lib_vs_standalone = (lib_analytic[k] - standalone_grad[k]).abs()
            / standalone_grad[k]
                .abs()
                .max(lib_analytic[k].abs())
                .max(1e-6);

        eprintln!(
            "coord {k}: lib_vs_fd={:.3e} standalone_vs_fd={:.3e} lib_vs_standalone={:.3e}",
            err_lib_vs_fd, err_standalone_vs_fd, err_lib_vs_standalone
        );
    }
}

/// Test logit gradient with a SINGLE penalty (no overlap).
/// If this passes but the two-penalty version fails, the bug is in structural_rank computation.
#[test]
fn test_logit_single_penalty_gradient_vs_fd() {
    let n = 60usize;
    let p = 5usize;
    let mut x = Array2::<f64>::zeros((n, p));
    for i in 0..n {
        let t = (i as f64) / ((n - 1) as f64);
        x[[i, 0]] = 1.0;
        x[[i, 1]] = t;
        x[[i, 2]] = t * t;
        x[[i, 3]] = (2.0 * std::f64::consts::PI * t).sin();
        x[[i, 4]] = (4.0 * std::f64::consts::PI * t).cos();
    }
    let y: Array1<f64> = (0..n)
        .map(|i| if i % 3 == 0 { 1.0 } else { 0.0 })
        .collect::<Vec<_>>()
        .into();

    // SINGLE penalty: penalizes cols 1-4, nullspace dim = 1
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
    let rho = array![2.0];

    let (lib_analytic, lib_fd) = evaluate_externalgradients(
        y.view(),
        Array1::ones(n).view(),
        x.view(),
        Array1::zeros(n).view(),
        &[s1],
        &opts,
        &rho,
    )
    .expect("should succeed");

    eprintln!("=== SINGLE PENALTY LOGIT GRADIENT TEST ===");
    eprintln!("lib_analytic = {:?}", lib_analytic);
    eprintln!("lib_fd       = {:?}", lib_fd);
    let diff = &lib_analytic - &lib_fd;
    eprintln!("diff         = {:?}", diff);
    for k in 0..rho.len() {
        let rel_err = (lib_analytic[k] - lib_fd[k]).abs()
            / lib_fd[k].abs().max(lib_analytic[k].abs()).max(1e-12);
        eprintln!(
            "  coord {k}: analytic={:.8e} fd={:.8e} rel_err={:.3e}",
            lib_analytic[k], lib_fd[k], rel_err
        );
    }

    let na = lib_analytic.iter().map(|v| v * v).sum::<f64>().sqrt();
    let nf = lib_fd.iter().map(|v| v * v).sum::<f64>().sqrt();
    let rel_l2 = (&lib_analytic - &lib_fd)
        .iter()
        .map(|v| v * v)
        .sum::<f64>()
        .sqrt()
        / na.max(nf).max(1e-12);
    eprintln!("rel_l2 = {rel_l2:.3e}");
    // If this passes, the bug is related to overlapping penalties / structural_rank
    assert!(
        rel_l2 < 0.05,
        "Single-penalty logit gradient should match FD: rel_l2={rel_l2:.3e}"
    );
}

/// Test probit gradient to prove the bug affects all non-Gaussian links, not just logit.
#[test]
fn test_probit_gradient_vs_fd() {
    let n = 60usize;
    let p = 5usize;
    let mut x = Array2::<f64>::zeros((n, p));
    for i in 0..n {
        let t = (i as f64) / ((n - 1) as f64);
        x[[i, 0]] = 1.0;
        x[[i, 1]] = t;
        x[[i, 2]] = t * t;
        x[[i, 3]] = (2.0 * std::f64::consts::PI * t).sin();
        x[[i, 4]] = (4.0 * std::f64::consts::PI * t).cos();
    }
    let y: Array1<f64> = (0..n)
        .map(|i| if i % 3 == 0 { 1.0 } else { 0.0 })
        .collect::<Vec<_>>()
        .into();

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
        family: LikelihoodFamily::BinomialProbit,
        compute_inference: true,
        max_iter: 200,
        tol: 1e-12,
        nullspace_dims: vec![1, 0],
        linear_constraints: None,
        firth_bias_reduction: None,
    };
    let rho = array![2.0, -1.0];

    let (analytic, fd) = evaluate_externalgradients(
        y.view(),
        Array1::ones(n).view(),
        x.view(),
        Array1::zeros(n).view(),
        &[s1, s2],
        &opts,
        &rho,
    )
    .expect("should succeed");

    eprintln!("=== PROBIT GRADIENT TEST ===");
    eprintln!("analytic = {:?}", analytic);
    eprintln!("fd       = {:?}", fd);
    let diff = &analytic - &fd;
    eprintln!("diff     = {:?}", diff);
    for k in 0..rho.len() {
        let rel_err = (analytic[k] - fd[k]).abs() / fd[k].abs().max(analytic[k].abs()).max(1e-12);
        eprintln!(
            "  coord {k}: analytic={:.8e} fd={:.8e} rel_err={:.3e}",
            analytic[k], fd[k], rel_err
        );
    }

    let na = analytic.iter().map(|v| v * v).sum::<f64>().sqrt();
    let nf = fd.iter().map(|v| v * v).sum::<f64>().sqrt();
    let cosine = analytic.dot(&fd) / (na * nf).max(1e-12);
    let rel_l2 =
        (&analytic - &fd).iter().map(|v| v * v).sum::<f64>().sqrt() / na.max(nf).max(1e-12);
    eprintln!("cosine = {cosine:.6}, rel_l2 = {rel_l2:.3e}");
    assert!(
        cosine > 0.99,
        "Probit gradient direction wrong: cosine={cosine:.6}"
    );
    assert!(
        rel_l2 < 0.05,
        "Probit gradient magnitude wrong: rel_l2={rel_l2:.3e}"
    );
}
