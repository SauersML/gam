use super::*;
use approx::assert_abs_diff_eq;
use ndarray::{array, s};

/// The isometry penalty is scale-invariant in decoder units: the per-row
/// pullback metric is normalized by `gbar = mean(trace(g))/d` before comparing
/// to the reference metric. Scaling every decoder Jacobian by a constant
/// therefore leaves value and normalized residuals unchanged.
#[test]
fn isometry_value_is_decoder_scale_invariant() {
    let n_obs = 2;
    let d = 2;
    let p = 3;
    let target = PsiSlice::full(n_obs * d, Some(d));
    let pen = IsometryPenalty::new_euclidean(target, p);
    let mut j = Array2::<f64>::zeros((n_obs, p * d));
    for n in 0..n_obs {
        for i in 0..p {
            for a in 0..d {
                j[[n, i * d + a]] = 0.4 + 0.2 * n as f64 - 0.1 * i as f64 + 0.3 * a as f64;
            }
        }
    }
    let mut j_scaled = j.clone();
    for value in j_scaled.iter_mut() {
        *value *= 17.0;
    }
    let rho = array![0.0_f64];
    let t = Array1::<f64>::zeros(n_obs * d);
    pen.set_jacobian_cache(Some(Arc::new(j)));
    let value = pen.value(t.view(), rho.view());
    pen.set_jacobian_cache(Some(Arc::new(j_scaled)));
    let scaled_value = pen.value(t.view(), rho.view());
    assert_abs_diff_eq!(value, scaled_value, epsilon = 1e-10);
}

/// #795 — the scale invariance the issue's fix (a) requires is NOT a
/// value-only property: the gradient `∂P/∂t`, the exact Hessian-vector product
/// `∇²P·v`, and the Gauss-Newton majorizer must ALL be invariant under a decoder
/// rescale, or the SAE joint Newton solve pairs a scale-free gradient with a
/// scale-dependent curvature and the proximal ridge saturates at 1e15.
///
/// A decoder rescale `B → λB` scales both the model Jacobian `J = (∂Φ/∂t)·B` and
/// its second jet `H = ∂J/∂t` linearly by λ. The pullback metric `g = JᵀJ ∝ λ²`
/// and the shared normalizer `gbar ∝ λ²`, so the normalized residual
/// `R = g/gbar − g_ref` — and therefore `P`, `∂P/∂t`, and `∇²P` — are invariant.
/// (`grad_jacobian = ∂P/∂J` instead scales ∝1/λ, since `P` is invariant while
/// `J` grew by λ; this is checked too.) The prior guard
/// `isometry_value_is_decoder_scale_invariant` covered only `value`.
#[test]
fn isometry_grad_hvp_majorizer_are_decoder_scale_invariant() {
    let (n_obs, p, d, j, h) = isometry_gn_fixture();
    let n = n_obs * d;
    let target = PsiSlice::full(n, Some(d));
    let rho = array![0.0_f64];
    let t = Array1::<f64>::zeros(n);
    let probes = [
        array![0.4_f64, -1.1, 0.7, 0.3, -0.5, 0.9],
        array![-2.3_f64, 0.6, -0.1, 1.4, 0.8, -1.7],
    ];

    let lambda = 6.5_f64;
    let j_scaled = Arc::new(&*j * lambda);
    let h_scaled = Arc::new(&*h * lambda);

    let base = IsometryPenalty::new_euclidean(target.clone(), p);
    base.refresh_caches(Some(j.clone()), Some(h.clone()));
    let scaled = IsometryPenalty::new_euclidean(target.clone(), p);
    scaled.refresh_caches(Some(j_scaled), Some(h_scaled));

    // value (re-pinned here alongside the rest, on the GN fixture).
    assert_abs_diff_eq!(
        base.value(t.view(), rho.view()),
        scaled.value(t.view(), rho.view()),
        epsilon = 1e-10
    );

    // grad_target (∂P/∂t): invariant.
    let g0 = base.grad_target(t.view(), rho.view());
    let g1 = scaled.grad_target(t.view(), rho.view());
    assert!(
        g0.iter().any(|x| x.abs() > 1e-9),
        "grad must be non-trivial"
    );
    for i in 0..n {
        assert_abs_diff_eq!(g0[i], g1[i], epsilon = 1e-9);
    }

    // hvp (∇²P·v) and psd_majorizer_hvp (GN block · v): both invariant.
    for v in &probes {
        let hv0 = base.hvp(t.view(), rho.view(), v.view());
        let hv1 = scaled.hvp(t.view(), rho.view(), v.view());
        let gn0 = base.psd_majorizer_hvp(t.view(), rho.view(), v.view());
        let gn1 = scaled.psd_majorizer_hvp(t.view(), rho.view(), v.view());
        assert!(
            gn0.iter().any(|x| x.abs() > 1e-9),
            "majorizer must be non-trivial"
        );
        for i in 0..n {
            assert_abs_diff_eq!(hv0[i], hv1[i], epsilon = 1e-8);
            assert_abs_diff_eq!(gn0[i], gn1[i], epsilon = 1e-8);
        }
    }

    // grad_jacobian (∂P/∂J): scales ∝1/λ (P invariant, J grew by λ).
    let gj0 = base.grad_jacobian(t.view(), rho.view());
    let gj1 = scaled.grad_jacobian(t.view(), rho.view());
    assert!(
        gj0.iter().any(|x| x.abs() > 1e-9),
        "grad_jacobian must be non-trivial"
    );
    for (a, b) in gj0.iter().zip(gj1.iter()) {
        assert_abs_diff_eq!(a / lambda, *b, epsilon = 1e-9);
    }
}

#[test]
fn ard_value_matches_quadratic_form() {
    let d = 2;
    let t = array![0.5_f64, 1.0, 2.0, -1.0, 0.0, 3.0];
    let target = PsiSlice::full(t.len(), Some(d));
    let ard = ARDPenalty::new(target, d);
    let rho = array![0.0_f64, 0.0]; // λ = 1 on both axes
    let v = ard.value(t.view(), rho.view());
    // Axis 0: 0.5² + 2.0² + 0.0² = 4.25 → ½·1·4.25
    // Axis 1: 1.0² + (-1)² + 3² = 11    → ½·1·11
    assert!((v - 0.5 * (4.25 + 11.0)).abs() < 1e-12);
}

#[test]
fn smoothed_l1_grad_smoothes_signum_at_zero() {
    let p = SparsityPenalty::smoothed_l1(PenaltyTier::Beta, 1e-3)
        .expect("positive eps builds smoothed L1 penalty");
    let t = array![0.0_f64, 1.0, -2.0];
    let rho = array![0.0_f64];
    let g = p.grad_target(t.view(), rho.view());
    // At x=0, grad = 0 / sqrt(0 + ε²) = 0 (not ±1).
    assert!(g[0].abs() < 1e-9);
    // At x=1, grad ≈ 1/sqrt(1 + ε²) ≈ 1.
    assert!((g[1] - 1.0).abs() < 1e-3);
    assert!((g[2] - (-1.0)).abs() < 1e-3);
}

#[test]
fn softmax_assignment_hvp_matches_gradient_directional_derivative() {
    let pen = SoftmaxAssignmentSparsityPenalty::new(3, 0.7);
    let t = array![0.4_f64, -0.8, 1.3, -0.2, 0.9, 0.1];
    let rho = array![1.4_f64.ln()];
    let v = array![0.2_f64, -0.5, 0.7, -0.3, 0.4, 0.6];

    // The analytic diagonal must equal hvp(.) probed by unit vectors e_k,
    // i.e. the i-th entry equals dot(hvp(t, rho, e_i), e_i).
    let h_diag = pen
        .hessian_diag(t.view(), rho.view())
        .expect("softmax entropy diagonal is analytic via row-dense HVP at e_k");
    for i in 0..t.len() {
        let mut e_i = Array1::<f64>::zeros(t.len());
        e_i[i] = 1.0;
        let hv_i = pen.hvp(t.view(), rho.view(), e_i.view());
        assert_abs_diff_eq!(h_diag[i], hv_i[i], epsilon = 1e-10);
    }

    let hv = pen.hvp(t.view(), rho.view(), v.view());
    let eps = 1e-6;
    let mut tp = t.clone();
    let mut tm = t.clone();
    for i in 0..t.len() {
        tp[i] += eps * v[i];
        tm[i] -= eps * v[i];
    }
    let gp = pen.grad_target(tp.view(), rho.view());
    let gm = pen.grad_target(tm.view(), rho.view());
    for i in 0..t.len() {
        let fd = (gp[i] - gm[i]) / (2.0 * eps);
        assert_abs_diff_eq!(hv[i], fd, epsilon = 1e-6);
    }
}

#[test]
fn softmax_row_dense_hessian_matches_hvp_and_diagonal() {
    // #1038: the exact dense per-row entropy Hessian must (a) reproduce the
    // analytic diagonal `hessian_diag`, (b) match the row-dense `hvp` action
    // on arbitrary directions, and (c) be gauge-null (`H·𝟙 = 0`).
    let pen = SoftmaxAssignmentSparsityPenalty::new(4, 0.7);
    let row = [0.4_f64, -0.8, 1.3, -0.2];
    let lambda = 1.4_f64;
    let rho = array![lambda.ln()];
    let inv_tau2 = (1.0 / 0.7_f64) * (1.0 / 0.7_f64);
    let scale = lambda * inv_tau2;
    let h = pen.row_dense_hessian(&row, scale);

    // (a) diagonal agreement.
    let full: Vec<f64> = row.to_vec();
    let diag = pen
        .hessian_diag(Array1::from_vec(full.clone()).view(), rho.view())
        .expect("diag");
    for k in 0..4 {
        assert_abs_diff_eq!(h[[k, k]], diag[k], epsilon = 1e-10);
    }
    // (b) symmetry + HVP agreement on a probe direction.
    for i in 0..4 {
        for j in 0..4 {
            assert_abs_diff_eq!(h[[i, j]], h[[j, i]], epsilon = 1e-12);
        }
    }
    let v = array![0.2_f64, -0.5, 0.7, -0.3];
    let hv = pen.hvp(Array1::from_vec(full.clone()).view(), rho.view(), v.view());
    for i in 0..4 {
        let acc: f64 = (0..4).map(|j| h[[i, j]] * v[j]).sum();
        assert_abs_diff_eq!(acc, hv[i], epsilon = 1e-9);
    }
    // (c) gauge null: H·𝟙 = 0 (softmax shift-invariance).
    for i in 0..4 {
        let row_sum: f64 = (0..4).map(|j| h[[i, j]]).sum();
        assert_abs_diff_eq!(row_sum, 0.0, epsilon = 1e-10);
    }
}

#[test]
fn softmax_row_dense_hessian_logit_derivative_matches_finite_difference() {
    // #1038: ∂H_{k,j}/∂z_w (the third-derivative tensor the θ-adjoint
    // contracts) must match a central finite difference of the dense block.
    let pen = SoftmaxAssignmentSparsityPenalty::new(4, 0.8);
    let row = [0.3_f64, -0.6, 0.9, 0.2];
    let scale = 1.1_f64 * (1.0 / 0.8_f64) * (1.0 / 0.8_f64);
    let eps = 1e-6;
    for w in 0..4 {
        let dh = pen.row_dense_hessian_logit_derivative(&row, scale, w);
        let mut rp = row;
        let mut rm = row;
        rp[w] += eps;
        rm[w] -= eps;
        let hp = pen.row_dense_hessian(&rp, scale);
        let hm = pen.row_dense_hessian(&rm, scale);
        for i in 0..4 {
            for j in 0..4 {
                let fd = (hp[[i, j]] - hm[[i, j]]) / (2.0 * eps);
                assert_abs_diff_eq!(dh[[i, j]], fd, epsilon = 1e-6);
            }
        }
    }
}

#[test]
fn softmax_row_fisher_metric_is_psd_gauge_null_and_derivative_matches_fd() {
    // #1190: the softmax Fisher metric `G = scale·(diag(a) − a aᵀ)` is the PSD
    // curvature operator the manifold-SAE evidence Hessian uses in place of the
    // indefinite entropy Hessian. It must be (a) symmetric PSD (all eigenvalues
    // >= 0), (b) gauge-null (`G·𝟙 = 0`, softmax shift-invariance), and (c) its
    // θ-derivative must match a central finite difference of `G` so the
    // assembly, log|H|, and adjoint stay on one exact branch.
    use gam_linalg::faer_ndarray::FaerEigh;
    let pen = SoftmaxAssignmentSparsityPenalty::new(4, 0.8);
    let row = [0.3_f64, -0.6, 0.9, 0.2];
    let scale = 1.1_f64 * (1.0 / 0.8_f64) * (1.0 / 0.8_f64);
    let g = pen.row_fisher_metric(&row, scale);
    // (a) symmetric.
    for i in 0..4 {
        for j in 0..4 {
            assert_abs_diff_eq!(g[[i, j]], g[[j, i]], epsilon = 1e-12);
        }
    }
    // (a) PSD: every eigenvalue >= -tiny round-off.
    let (evals, _) = g.eigh(faer::Side::Lower).expect("eigh");
    let max_abs = evals.iter().fold(0.0_f64, |a, &v| a.max(v.abs())).max(1.0);
    for &lambda in evals.iter() {
        assert!(
            lambda >= -1e-10 * max_abs,
            "Fisher metric must be PSD; got eigenvalue {lambda}"
        );
    }
    // (b) gauge null: G·𝟙 = 0.
    for i in 0..4 {
        let row_sum: f64 = (0..4).map(|j| g[[i, j]]).sum();
        assert_abs_diff_eq!(row_sum, 0.0, epsilon = 1e-12);
    }
    // (c) derivative matches central FD.
    let eps = 1e-6;
    for w in 0..4 {
        let dg = pen.row_fisher_metric_logit_derivative(&row, scale, w);
        let mut rp = row;
        let mut rm = row;
        rp[w] += eps;
        rm[w] -= eps;
        let gp = pen.row_fisher_metric(&rp, scale);
        let gm = pen.row_fisher_metric(&rm, scale);
        for i in 0..4 {
            for j in 0..4 {
                let fd = (gp[[i, j]] - gm[[i, j]]) / (2.0 * eps);
                assert_abs_diff_eq!(dg[[i, j]], fd, epsilon = 1e-6);
            }
        }
    }
}

#[test]
fn ibp_assignment_grad_target_matches_value_finite_difference() {
    let pen = IBPAssignmentPenalty::new(4, 6.0, 0.8, false);
    let t = array![
        0.2_f64, -0.3, 0.7, -0.5, 0.9, 0.4, -0.2, 0.1, -0.4, 0.8, 0.3, -0.1
    ];
    let rho = Array1::<f64>::zeros(0);
    let g = pen.grad_target(t.view(), rho.view());
    let eps = 1.0e-6;
    let fd = gam_test_support::fd_checker::numerical_gradient_central_diff(
        |tv| pen.value(tv.view(), rho.view()),
        &t,
        eps,
    );
    let mut max_err = 0.0_f64;
    for i in 0..t.len() {
        let err = (g[i] - fd[i]).abs();
        if err > max_err {
            max_err = err;
        }
        assert_abs_diff_eq!(g[i], fd[i], epsilon = 1.0e-7);
    }
    assert!(
        max_err < 1.0e-7,
        "IBP grad-FD max abs error = {max_err:.3e}"
    );
}

#[test]
fn ibp_cross_row_woodbury_d_matches_full_off_diagonal_hessian() {
    // #1038: the exact IBP Hessian couples DIFFERENT rows within a column
    // through the plug-in empirical mass `M_k = Σ_i z_ik`:
    //   ∂²(value)/∂ℓ_ik ∂ℓ_jk = w · s'_k · z'_ik · z'_jk   (the cross-row
    // rank-one block, including i=j). `cross_row_d[k] = w·s'_k` and
    // `z_jac[i*K+k] = z'_ik`, so the analytic product must reproduce the
    // central-difference second derivative of `value` for every (i≠j) pair.
    let pen = IBPAssignmentPenalty::new(3, 5.0, 0.85, false);
    // 4 rows × 3 columns; row-major (N, K).
    let t = array![
        0.3_f64, -0.2, 0.6, 0.5, 0.1, -0.4, -0.1, 0.7, 0.2, 0.4, -0.3, 0.8
    ];
    let rho = Array1::<f64>::zeros(0);
    let k = pen.k_max;
    let n = t.len() / k;
    let ch = pen.hessian_diag_logit_third_channels(t.view(), rho.view(), false);
    let eps = 1.0e-5;
    let mut max_err = 0.0_f64;
    // Mixed second derivative via 4-point central difference on `value`.
    let mixed_fd = |a: usize, b: usize| -> f64 {
        let bump = |sa: f64, sb: f64| -> Array1<f64> {
            let mut tt = t.clone();
            tt[a] += sa * eps;
            tt[b] += sb * eps;
            tt
        };
        (pen.value(bump(1.0, 1.0).view(), rho.view())
            - pen.value(bump(1.0, -1.0).view(), rho.view())
            - pen.value(bump(-1.0, 1.0).view(), rho.view())
            + pen.value(bump(-1.0, -1.0).view(), rho.view()))
            / (4.0 * eps * eps)
    };
    for col in 0..k {
        for i in 0..n {
            for j in 0..n {
                if i == j {
                    continue;
                }
                let analytic = ch.cross_row_d[col] * ch.z_jac[i * k + col] * ch.z_jac[j * k + col];
                let fd = mixed_fd(i * k + col, j * k + col);
                let err = (analytic - fd).abs();
                if err > max_err {
                    max_err = err;
                }
                assert_abs_diff_eq!(analytic, fd, epsilon = 5.0e-5);
            }
        }
    }
    // Distinct columns do NOT couple cross-row (independent stick-breaking
    // masses): the analytic model predicts zero, and the FD must agree.
    // Pick row 0, col 0 vs row 1, col 1 (flat indices 0 and k + 1).
    let mixed_distinct = mixed_fd(0, k + 1);
    assert!(
        mixed_distinct.abs() < 5.0e-5,
        "distinct-column cross-row coupling must vanish; got {mixed_distinct:.3e}"
    );
    assert!(
        max_err < 5.0e-5,
        "IBP cross-row Woodbury d·z'·z' vs FD max abs error = {max_err:.3e}"
    );
}

#[test]
fn ibp_cross_row_woodbury_dd_and_logit_curvature_match_finite_difference() {
    // #1416: the θ-adjoint differentiates the cross-row Woodbury block
    //   W_k = d_k·u_k u_kᵀ,  u_k[i] = J_ik,  d_k = w·s'_k(M_k).
    // Its θ-derivative needs two new exact channels:
    //   cross_row_dd[k] = ∂d_k/∂M_k = w·s''_k  (since ∂M_k/∂ℓ_mk = J_mk),
    //   logit_curvature[i*K+k] = ∂J_ik/∂ℓ_ik = c_ik.
    // Verify both against central differences of the base channels.
    let pen = IBPAssignmentPenalty::new(3, 5.0, 0.85, false);
    let t = array![
        0.3_f64, -0.2, 0.6, 0.5, 0.1, -0.4, -0.1, 0.7, 0.2, 0.4, -0.3, 0.8
    ];
    let rho = Array1::<f64>::zeros(0);
    let k = pen.k_max;
    let n = t.len() / k;
    let ch = pen.hessian_diag_logit_third_channels(t.view(), rho.view(), false);
    let eps = 1.0e-6;
    let bumped = |idx: usize, s: f64| {
        let mut tt = t.clone();
        tt[idx] += s * eps;
        pen.hessian_diag_logit_third_channels(tt.view(), rho.view(), false)
    };

    // logit_curvature[i*K+k] = d(z_jac[i*K+k])/dℓ_ik (only the same logit moves it).
    let mut max_c = 0.0_f64;
    for i in 0..n {
        for col in 0..k {
            let plus = bumped(i * k + col, 1.0);
            let minus = bumped(i * k + col, -1.0);
            let fd = (plus.z_jac[i * k + col] - minus.z_jac[i * k + col]) / (2.0 * eps);
            let err = (ch.logit_curvature[i * k + col] - fd).abs();
            max_c = max_c.max(err);
            assert_abs_diff_eq!(ch.logit_curvature[i * k + col], fd, epsilon = 1.0e-5);
        }
    }
    assert!(max_c < 1.0e-5, "logit_curvature FD max err = {max_c:.3e}");

    // cross_row_dd[k]·J_mk = d(cross_row_d[k])/dℓ_mk for any row m in column k.
    let mut max_dd = 0.0_f64;
    for col in 0..k {
        for m in 0..n {
            let plus = bumped(m * k + col, 1.0);
            let minus = bumped(m * k + col, -1.0);
            let fd = (plus.cross_row_d[col] - minus.cross_row_d[col]) / (2.0 * eps);
            let analytic = ch.cross_row_dd[col] * ch.z_jac[m * k + col];
            let err = (analytic - fd).abs();
            max_dd = max_dd.max(err);
            assert_abs_diff_eq!(analytic, fd, epsilon = 1.0e-5);
        }
    }
    assert!(
        max_dd < 1.0e-5,
        "cross_row_dd·J vs FD max err = {max_dd:.3e}"
    );
}

#[test]
fn ibp_majorized_channels_match_fd_of_psd_majorized_operator() {
    // gam#2144 consistency: with `majorize = true` every third channel must be the
    // exact derivative of the PSD Loewner-majorized column block
    //   D_ik = max(w·s'_k, 0)·J_ik² + max(w·s_k·c_ik, 0),   d_k = max(w·s'_k, 0),
    // NOT the raw indefinite IBP Hessian — so the low-rank-whitened θ-adjoint/ρ-trace
    // differentiate the SAME operator the majorized evidence log-det factors. Verify
    // (a) the clamps actually fire on this fixture (non-vacuous), (b) cross_row_d/dd
    // are the clamped coefficient and its gated mass-derivative, and (c)
    // m_channel/local_logit_third reproduce the TOTAL logit-derivative of `D_ik`
    // (`δ_iw·local_logit_third + m_channel·J_wk`) against a central difference.
    let pen = IBPAssignmentPenalty::new(3, 5.0, 0.85, false);
    let t = array![
        0.3_f64, -0.2, 0.6, 0.5, 0.1, -0.4, -0.1, 0.7, 0.2, 0.4, -0.3, 0.8
    ];
    let rho = Array1::<f64>::zeros(0);
    let k = pen.k_max;
    let n = t.len() / k;
    // Majorized column-block diagonal as a function of the logits, reconstructed
    // from the raw diagonal (`w·(s'·J² + s·c)`) + raw `cross_row_d` (`w·s'`) + `J`.
    let maj_diag = |tv: ArrayView1<'_, f64>| -> Array1<f64> {
        let raw = pen.hessian_diag_logit_third_channels(tv, rho.view(), false);
        let hdiag = pen.hessian_diag(tv, rho.view()).unwrap();
        let mut d = Array1::<f64>::zeros(tv.len());
        for i in 0..n {
            for col in 0..k {
                let idx = i * k + col;
                let jj = raw.z_jac[idx] * raw.z_jac[idx];
                let self_term = raw.cross_row_d[col] * jj; // w·s'·J²
                let sc = hdiag[idx] - self_term; // w·s·c
                d[idx] = raw.cross_row_d[col].max(0.0) * jj + sc.max(0.0);
            }
        }
        d
    };
    let ch = pen.hessian_diag_logit_third_channels(t.view(), rho.view(), true);
    let raw = pen.hessian_diag_logit_third_channels(t.view(), rho.view(), false);

    // (a) non-vacuity: at least one column's rank-one coefficient is clamped off.
    let clamped_col = (0..k).any(|c| raw.cross_row_d[c] < -1.0e-9 && ch.cross_row_d[c] == 0.0);
    assert!(
        clamped_col,
        "fixture must clamp at least one rank-one column (else the majorizer is vacuous)"
    );

    // (b) cross_row_d = max(w·s',0); cross_row_dd gated by the same clamp.
    for c in 0..k {
        assert_abs_diff_eq!(ch.cross_row_d[c], raw.cross_row_d[c].max(0.0), epsilon = 1.0e-12);
        if raw.cross_row_d[c] <= 0.0 {
            assert_eq!(ch.cross_row_dd[c], 0.0, "clamped column must gate cross_row_dd to 0");
        } else {
            assert_abs_diff_eq!(ch.cross_row_dd[c], raw.cross_row_dd[c], epsilon = 1.0e-12);
        }
    }

    // (c) m_channel/local_logit_third reproduce the total logit-FD of `D_ik`.
    let eps = 1.0e-6;
    let mut max_err = 0.0_f64;
    for w in 0..n {
        for col in 0..k {
            let widx = w * k + col;
            let mut tp = t.clone();
            let mut tm = t.clone();
            tp[widx] += eps;
            tm[widx] -= eps;
            let dp = maj_diag(tp.view());
            let dm = maj_diag(tm.view());
            for i in 0..n {
                let idx = i * k + col;
                let fd = (dp[idx] - dm[idx]) / (2.0 * eps);
                let local = if i == w { ch.local_logit_third[idx] } else { 0.0 };
                let analytic = local + ch.m_channel[idx] * ch.z_jac[widx];
                max_err = max_err.max((analytic - fd).abs());
            }
        }
    }
    assert!(
        max_err < 1.0e-5,
        "majorized channel total-derivative vs FD of D_ik max abs err = {max_err:.3e}"
    );
}

#[test]
fn ibp_cross_row_d_logalpha_matches_finite_difference() {
    // #1417 fix: the cross-row rank-one coefficient's logα-derivative used by the
    // LEARNABLE-α log-det ρ-gradient. For learnable α, `α(ρ₀)=α_base·e^{ρ₀}` so
    // `∂logα/∂ρ₀=1`, hence `∂(cross_row_d[k])/∂ρ₀ = ∂d_k/∂logα = cross_row_d_logalpha[k]`.
    // Central-difference the VALUE channel `cross_row_d` w.r.t. ρ₀ and compare.
    // (The pre-fix bug used the value `cross_row_d` itself in the off-diagonal of
    // the logα trace — inconsistent with the α-differentiated diagonal channel.)
    let pen = IBPAssignmentPenalty::new(3, 6.0, 0.8, true);
    let t = array![
        0.2_f64, -0.3, 0.7, -0.1, 0.4, 0.5, 0.6, -0.2, 0.3, 0.1, 0.8, -0.4
    ];
    let rho = array![0.15_f64];
    let k = pen.k_max;
    let ch = pen.hessian_diag_logit_third_channels(t.view(), rho.view(), false);
    let eps = 1.0e-6;
    let mut rp = rho.clone();
    let mut rm = rho.clone();
    rp[0] += eps;
    rm[0] -= eps;
    let plus = pen.hessian_diag_logit_third_channels(t.view(), rp.view(), false);
    let minus = pen.hessian_diag_logit_third_channels(t.view(), rm.view(), false);
    let mut max_err = 0.0_f64;
    let mut saw_nonzero = false;
    for col in 0..k {
        let fd = (plus.cross_row_d[col] - minus.cross_row_d[col]) / (2.0 * eps);
        let err = (ch.cross_row_d_logalpha[col] - fd).abs();
        max_err = max_err.max(err);
        if ch.cross_row_d_logalpha[col].abs() > 1.0e-9 {
            saw_nonzero = true;
        }
        assert_abs_diff_eq!(ch.cross_row_d_logalpha[col], fd, epsilon = 1.0e-6);
    }
    assert!(
        saw_nonzero,
        "fixture must exercise a nonzero logα cross-row coefficient (else vacuous)"
    );
    assert!(
        max_err < 1.0e-6,
        "cross_row_d_logalpha vs FD max abs err = {max_err:.3e}"
    );
    // Fixed-α leaves the channel at zero (the value `cross_row_d` is used there).
    let fixed = IBPAssignmentPenalty::new(3, 6.0, 0.8, false);
    let chf = fixed.hessian_diag_logit_third_channels(t.view(), Array1::<f64>::zeros(0).view(), false);
    assert!(
        chf.cross_row_d_logalpha.iter().all(|&v| v == 0.0),
        "fixed-α must leave cross_row_d_logalpha zero"
    );
}

#[test]
fn ibp_cross_row_dd_matches_mass_derivative_of_cross_row_d_2087() {
    // #2087/#1416 root cause guard: the IBP log-det θ-adjoint differentiates the
    // same cross-row rank-one coefficient `d_k = w·score'_k` that the Hessian
    // assembly puts into the Woodbury block. Its mass channel must therefore be
    // `∂d_k/∂M_k = w·score''_k`, not a hand-expanded surrogate with a missing
    // posterior-π Jacobian factor. Perturbing every row in one column by the same
    // concrete-probability amount gives a direct central difference of `d_k` with
    // respect to the empirical mass `M_k`.
    let pen = IBPAssignmentPenalty::new(3, 6.0, 0.8, false);
    let t = array![
        0.2_f64, -0.3, 0.7, -0.1, 0.4, 0.5, 0.6, -0.2, 0.3, 0.1, 0.8, -0.4
    ];
    let rho = Array1::<f64>::zeros(0);
    let raw = pen.hessian_diag_logit_third_channels(t.view(), rho.view(), false);
    let eps = 1.0e-6_f64;
    let tau = 0.8_f64;
    let z = t.mapv(|logit| 1.0 / (1.0 + (-logit / tau).exp()));
    let n = t.len() / pen.k_max;
    for col in 0..pen.k_max {
        let mut plus = t.clone();
        let mut minus = t.clone();
        for row in 0..n {
            let idx = row * pen.k_max + col;
            // Convert a probability-space perturbation `±eps` to the logit-space
            // perturbation that realizes it to first order: dz = J dℓ. The
            // fixture is comfortably interior, so every Jacobian is nonzero.
            let jac = z[idx] * (1.0 - z[idx]) / tau;
            plus[idx] += eps / jac;
            minus[idx] -= eps / jac;
        }
        let d_plus = pen
            .hessian_diag_logit_third_channels(plus.view(), rho.view(), false)
            .cross_row_d[col];
        let d_minus = pen
            .hessian_diag_logit_third_channels(minus.view(), rho.view(), false)
            .cross_row_d[col];
        let fd = (d_plus - d_minus) / (2.0 * eps * n as f64);
        assert!(
            (raw.cross_row_dd[col] - fd).abs() <= 2.0e-6,
            "column {col}: cross_row_dd={} must match ∂cross_row_d/∂M={fd}",
            raw.cross_row_dd[col]
        );
    }
}

#[test]
fn ibp_assignment_learnable_alpha_grad_rho_matches_value_finite_difference() {
    let pen = IBPAssignmentPenalty::new(3, 6.0, 0.8, true);
    let t = array![
        0.2_f64, -0.3, 0.7, -0.1, 0.4, 0.5, 0.6, -0.2, 0.3, 0.1, 0.8, -0.4
    ];
    let rho = array![0.2_f64];
    let grad = pen.grad_rho(t.view(), rho.view());
    let step = 1.0e-6_f64;
    let rho_plus = array![rho[0] + step];
    let rho_minus = array![rho[0] - step];
    let fd = (pen.value(t.view(), rho_plus.view()) - pen.value(t.view(), rho_minus.view()))
        / (2.0 * step);

    assert_abs_diff_eq!(grad[0], fd, epsilon = 2.0e-7);
}

#[test]
fn ibp_assignment_learnable_alpha_mixed_log_alpha_target_matches_fd() {
    let pen = IBPAssignmentPenalty::new(2, 2.0, 0.9, true);
    let t = array![0.2_f64, -0.3, 0.7, -0.1, 0.4, 0.5];
    let rho = array![0.15_f64];
    let analytic = pen.log_alpha_target_mixed_derivative(t.view(), rho.view());
    let step = 1.0e-6_f64;
    for i in 0..t.len() {
        let mut tp = t.clone();
        let mut tm = t.clone();
        tp[i] += step;
        tm[i] -= step;
        let gp = pen.grad_rho(tp.view(), rho.view())[0];
        let gm = pen.grad_rho(tm.view(), rho.view())[0];
        let fd = (gp - gm) / (2.0 * step);
        assert_abs_diff_eq!(analytic[i], fd, epsilon = 2.0e-7);
    }
}

#[test]
fn ibp_assignment_learnable_alpha_hdiag_log_alpha_derivative_matches_fd() {
    let pen = IBPAssignmentPenalty::new(2, 2.0, 0.9, true);
    let t = array![0.2_f64, -0.3, 0.7, -0.1, 0.4, 0.5];
    let rho = array![0.15_f64];
    let analytic = pen.hessian_diag_log_alpha_derivative(t.view(), rho.view());
    let step = 1.0e-6_f64;
    let rho_plus = array![rho[0] + step];
    let rho_minus = array![rho[0] - step];
    let hp = pen
        .hessian_diag(t.view(), rho_plus.view())
        .expect("IBP hessian diag exists");
    let hm = pen
        .hessian_diag(t.view(), rho_minus.view())
        .expect("IBP hessian diag exists");
    for i in 0..t.len() {
        let fd = (hp[i] - hm[i]) / (2.0 * step);
        assert_abs_diff_eq!(analytic[i], fd, epsilon = 2.0e-7);
    }
}

#[test]
fn ibp_assignment_extreme_logits_remain_finite() {
    let pen = IBPAssignmentPenalty::new(3, 1.5, 1.0e-3, false);
    let t = array![
        1000.0_f64, -1000.0, 500.0, -500.0, 750.0, -750.0, 250.0, -250.0, 0.0
    ];
    let rho = Array1::<f64>::zeros(0);

    let value = pen.value(t.view(), rho.view());
    assert!(
        value.is_finite(),
        "IBP value must remain finite for saturated concrete logits"
    );
    let grad = pen.grad_target(t.view(), rho.view());
    assert!(
        grad.iter().all(|entry| entry.is_finite()),
        "IBP gradient must remain finite for saturated concrete logits: {grad:?}"
    );
    let diag = pen
        .hessian_diag(t.view(), rho.view())
        .expect("IBP assignment exposes a diagonal Hessian");
    assert!(
        diag.iter().all(|entry| entry.is_finite()),
        "IBP Hessian diagonal must remain finite for saturated concrete logits: {diag:?}"
    );
}

#[test]
fn ibp_assignment_high_k_prior_keeps_positive_gradient_path() {
    let k = 400usize;
    let pen = IBPAssignmentPenalty::new(k, 0.1, 1.0, false);
    let t = Array1::<f64>::zeros(k);
    let rho = Array1::<f64>::zeros(0);

    let value = pen.value(t.view(), rho.view());
    assert!(value.is_finite(), "high-K IBP value must stay finite");
    let grad = pen.grad_target(t.view(), rho.view());
    assert_eq!(grad.len(), k);
    assert!(
        grad.iter().all(|entry| entry.is_finite()),
        "high-K IBP gradient must stay finite: {grad:?}"
    );
    assert!(
        grad.slice(s![320..]).iter().any(|entry| entry.abs() > 0.0),
        "late high-K atoms must retain a strictly positive gradient path"
    );
}

#[test]
fn learnable_weights_stay_finite_at_extreme_rho() {
    for rho in [1000.0_f64, -1000.0] {
        let resolved = resolve_learnable_weight(0.7, rho);
        assert!(
            resolved.is_finite() && resolved > 0.0,
            "resolved learnable weight must be finite-positive at rho={rho}: {resolved}"
        );
    }

    let softmax = SoftmaxAssignmentSparsityPenalty::new(3, 0.8);
    let logits = array![0.2_f64, -0.1, 0.4];
    for rho in [array![1000.0_f64], array![-1000.0_f64]] {
        let value = softmax.value(logits.view(), rho.view());
        let grad = softmax.grad_target(logits.view(), rho.view());
        let diag = softmax
            .hessian_diag(logits.view(), rho.view())
            .expect("softmax entropy exposes a diagonal Hessian");
        assert!(value.is_finite(), "softmax value non-finite at rho={rho:?}");
        assert!(grad.iter().all(|entry| entry.is_finite()));
        assert!(diag.iter().all(|entry| entry.is_finite()));
    }

    let jump = JumpReLUPenalty::new(PsiSlice::full(2, Some(1)), array![1.0_f64], 0.5, 0.1).unwrap();
    let jump_target = array![0.0_f64, 0.2];
    for rho in [array![1000.0_f64], array![-1000.0_f64]] {
        let value = jump.value(jump_target.view(), rho.view());
        let grad = jump.grad_target(jump_target.view(), rho.view());
        let diag = jump
            .hessian_diag(jump_target.view(), rho.view())
            .expect("JumpReLU exposes a diagonal Hessian");
        assert!(
            value.is_finite(),
            "JumpReLU value non-finite at rho={rho:?}"
        );
        assert!(grad.iter().all(|entry| entry.is_finite()));
        assert!(diag.iter().all(|entry| entry.is_finite()));
    }

    let target = PsiSlice {
        range: 0..4,
        latent_dim: Some(2),
    };
    let block_sizes = vec![1usize, 1usize];
    let p = 2usize;
    let coact = Array2::<f64>::ones((2, 2));
    let decoder = DecoderIncoherencePenalty::new(target, block_sizes, p, coact, 0.7, true).unwrap();
    let beta = Array1::<f64>::zeros(4);
    for rho in [array![1000.0_f64], array![-1000.0_f64]] {
        let value = decoder.value(beta.view(), rho.view());
        let grad = decoder.grad_target(beta.view(), rho.view());
        let hv = decoder.hvp(beta.view(), rho.view(), beta.view());
        assert!(
            value.is_finite(),
            "DecoderIncoherence value non-finite at rho={rho:?}"
        );
        assert!(grad.iter().all(|entry| entry.is_finite()));
        assert!(hv.iter().all(|entry| entry.is_finite()));
    }
}

#[test]
fn ard_grad_target_matches_lambda_t() {
    let d = 2;
    let t = array![0.5_f64, 1.0, 2.0, -1.0];
    let target = PsiSlice::full(t.len(), Some(d));
    let ard = ARDPenalty::new(target, d);
    // log-precisions: ρ0 = ln 2 (λ0 = 2), ρ1 = ln 3 (λ1 = 3).
    let rho = array![2.0_f64.ln(), 3.0_f64.ln()];
    let g = ard.grad_target(t.view(), rho.view());
    // Axis 0 entries (n*d + 0): indices 0, 2. λ0 · t at those slots.
    assert!((g[0] - 2.0 * 0.5).abs() < 1e-12);
    assert!((g[2] - 2.0 * 2.0).abs() < 1e-12);
    // Axis 1 entries (n*d + 1): indices 1, 3. λ1 · t.
    assert!((g[1] - 3.0 * 1.0).abs() < 1e-12);
    assert!((g[3] - -3.0).abs() < 1e-12);
}

#[test]
fn ard_hessian_diag_matches_lambda() {
    let d = 2;
    let t = array![0.5_f64, 1.0, 2.0, -1.0];
    let target = PsiSlice::full(t.len(), Some(d));
    let ard = ARDPenalty::new(target, d);
    let rho = array![2.0_f64.ln(), 3.0_f64.ln()];
    let h = ard
        .hessian_diag(t.view(), rho.view())
        .expect("ARD has a diagonal Hessian");
    assert!((h[0] - 2.0).abs() < 1e-12);
    assert!((h[2] - 2.0).abs() < 1e-12);
    assert!((h[1] - 3.0).abs() < 1e-12);
    assert!((h[3] - 3.0).abs() < 1e-12);
}

/// Deterministic `(n_obs=3, p=4, d=2)` first/second decoder jets for the
/// isometry Gauss-Newton majorizer tests. `J` is `(n_obs, p*d)` indexed
/// `J[n, i*d+a]`; `H` is `(n_obs, p*d*d)` indexed `H[n, (i*d+a)*d+c]`,
/// symmetric in `(a, c)` as a genuine second derivative must be.
fn isometry_gn_fixture() -> (usize, usize, usize, Arc<Array2<f64>>, Arc<Array2<f64>>) {
    let (n_obs, p, d) = (3usize, 4usize, 2usize);
    let mut j = Array2::<f64>::zeros((n_obs, p * d));
    for n in 0..n_obs {
        for i in 0..p {
            for a in 0..d {
                j[[n, i * d + a]] = 0.7 + 0.31 * (n as f64) - 0.23 * (i as f64)
                    + 0.17 * (a as f64)
                    + 0.05 * ((n * p + i) as f64);
            }
        }
    }
    let mut h = Array2::<f64>::zeros((n_obs, p * d * d));
    for n in 0..n_obs {
        for i in 0..p {
            for a in 0..d {
                for c in 0..d {
                    // Symmetric in (a, c): depends only on a+c and a·c.
                    let s = (a + c) as f64;
                    let pr = (a * c) as f64;
                    h[[n, (i * d + a) * d + c]] =
                        0.13 * (n as f64 + 1.0) + 0.09 * (i as f64) + 0.21 * s - 0.04 * pr;
                }
            }
        }
    }
    (n_obs, p, d, Arc::new(j), Arc::new(h))
}

/// The isometry penalty is a nonconvex least-squares objective, so its
/// curvature contribution to the Arrow-Schur inner solve is the
/// Gauss-Newton majorizer `B_GN`, not the indefinite exact Hessian. This
/// pins the two structural invariants the inner solve relies on: `B_GN` is
/// symmetric and positive-semidefinite, built from only the first and
/// second decoder jets (no third jet `K`).
#[test]
fn isometry_gn_majorizer_is_psd_and_symmetric() {
    let (n_obs, p, d, j, h) = isometry_gn_fixture();
    let n = n_obs * d;
    let target = PsiSlice::full(n, Some(d));
    let pen = IsometryPenalty::new_euclidean(target, p);
    pen.refresh_caches(Some(j), Some(h));
    // psd_majorizer_hvp reads the cached jets, so t only sets n_obs.
    let t = Array1::<f64>::zeros(n);
    let rho = array![0.0_f64];

    // Symmetry: assemble the dense operator column-by-column via unit probes.
    let mut bmat = Array2::<f64>::zeros((n, n));
    for k in 0..n {
        let mut e = Array1::<f64>::zeros(n);
        e[k] = 1.0;
        let col = pen.psd_majorizer_hvp(t.view(), rho.view(), e.view());
        for r in 0..n {
            bmat[[r, k]] = col[r];
        }
    }
    for r in 0..n {
        for c in 0..n {
            assert_abs_diff_eq!(bmat[[r, c]], bmat[[c, r]], epsilon = 1e-12);
        }
    }

    // PSD: vᵀ B v ≥ 0 for a spread of probe directions.
    let probes = [
        array![0.4_f64, -1.1, 0.7, 0.3, -0.5, 0.9],
        array![1.0_f64, 1.0, 1.0, 1.0, 1.0, 1.0],
        array![-2.3_f64, 0.6, -0.1, 1.4, 0.8, -1.7],
        array![0.0_f64, 0.0, 3.2, -0.4, 0.0, 0.5],
    ];
    for v in &probes {
        let bv = pen.psd_majorizer_hvp(t.view(), rho.view(), v.view());
        let quad = v.dot(&bv);
        assert!(
            quad >= -1e-9,
            "isometry GN majorizer must be PSD; got vᵀBv = {quad:.3e}"
        );
    }
}

/// As the normalized residual `g/gbar − g_ref → 0` the exact Hessian collapses onto its
/// Gauss-Newton block. Pinning the reference metric to the model's own
/// pullback metric drives the residual to exactly zero, so the exact `hvp`
/// (which here also has a — zero — third jet so its state assembles) must
/// agree bit-for-bit with the `B_GN` majorizer.
#[test]
fn isometry_gn_majorizer_matches_exact_hvp_at_zero_residual() {
    let (n_obs, p, d, j, h) = isometry_gn_fixture();
    let n = n_obs * d;
    let target = PsiSlice::full(n, Some(d));

    // Stage caches on a scratch penalty to read the pullback metric g(t),
    // then pin the reference to g/gbar (normalized residual ≡ 0).
    let scratch = IsometryPenalty::new_euclidean(target.clone(), p);
    scratch.refresh_caches(Some(j.clone()), Some(h.clone()));
    let mut g = scratch
        .pullback_metric(d)
        .expect("pullback metric available once J is cached");
    let mut trace_sum = 0.0_f64;
    for row in 0..g.nrows() {
        for axis in 0..d {
            trace_sum += g[[row, axis * d + axis]];
        }
    }
    let normalizer = trace_sum / (g.nrows() * d) as f64;
    for value in g.iter_mut() {
        *value /= normalizer;
    }

    // A zero third jet lets the exact hvp_state assemble (it requires a
    // third-derivative source) while contributing nothing to the result.
    let k_zero = Arc::new(ndarray::Array3::<f64>::zeros((n_obs, p, d * d * d)));
    let pen = IsometryPenalty::new_euclidean(target, p)
        .with_reference(IsometryReference::UserSupplied(Arc::new(g)))
        .with_third_decoder_derivative(k_zero);
    pen.refresh_caches(Some(j), Some(h));

    let t = Array1::<f64>::zeros(n);
    let rho = array![0.0_f64];
    let probes = [
        array![0.4_f64, -1.1, 0.7, 0.3, -0.5, 0.9],
        array![-2.3_f64, 0.6, -0.1, 1.4, 0.8, -1.7],
    ];
    for v in &probes {
        let exact = pen.hvp(t.view(), rho.view(), v.view());
        let gn = pen.psd_majorizer_hvp(t.view(), rho.view(), v.view());
        for i in 0..n {
            assert_abs_diff_eq!(exact[i], gn[i], epsilon = 1e-12);
        }
        // Guard against a vacuous all-zero "match".
        assert!(gn.iter().any(|x| x.abs() > 1e-9));
    }
}

/// Build the canonical JumpReLU sweep fixture: a logit grid that straddles
/// each per-axis scaled threshold so the gate `g = σ((z − τ)/ε)` sweeps both
/// sides of its inflection `g = ½`, where the true Hessian
/// `wτ·g(1−g)(1−2g)/ε²` changes sign.
fn jumprelu_sweep_fixture() -> (
    JumpReLUPenalty,
    Array1<f64>,
    Array1<f64>,
    [f64; 2],
    f64,
    f64,
) {
    let thresholds = array![0.25_f64, 0.8];
    let rho = array![0.0_f64, 1.5_f64.ln()];
    let eps = 0.04_f64;
    let weight = 1.3_f64;
    let scaled_thresholds = [thresholds[0] * rho[0].exp(), thresholds[1] * rho[1].exp()];
    let latent_dim = thresholds.len();
    let offsets = [-5.0_f64, -2.0, -0.5, -0.05, 0.0, 0.05, 0.5, 2.0, 5.0];
    let mut values = Vec::with_capacity(offsets.len() * latent_dim);
    for &offset in &offsets {
        values.push(scaled_thresholds[0] + offset);
        values.push(scaled_thresholds[1] + offset);
    }
    let target_values = Array1::from_vec(values);
    let slice = PsiSlice::full(target_values.len(), Some(latent_dim));
    let pen = JumpReLUPenalty::new(slice, thresholds, weight, eps).expect("valid JumpReLU penalty");
    (pen, target_values, rho, scaled_thresholds, eps, weight)
}

#[test]
fn jumprelu_hessian_diag_is_exact_true_second_derivative() {
    let (pen, target_values, rho, scaled_thresholds, eps, weight) = jumprelu_sweep_fixture();
    let latent_dim = scaled_thresholds.len();
    // `hessian_diag` must be the EXACT diagonal second derivative of the
    // smoothed jump penalty `P(z) = wτ·σ((z − τ)/ε)`:
    //   P''(z) = wτ·g(1 − g)(1 − 2g)/ε²,   g = σ((z − τ)/ε).
    // This is the true (indefinite) Hessian, not the PSD majorizer.
    let diag = pen
        .hessian_diag(target_values.view(), rho.view())
        .expect("JumpReLU exposes an analytic diagonal Hessian");

    let mut saw_negative = false;
    for (idx, &entry) in diag.iter().enumerate() {
        let axis = idx % latent_dim;
        let gate = pen.sigmoid_gate((target_values[idx] - scaled_thresholds[axis]) / eps);
        let expected = weight * scaled_thresholds[axis] * gate * (1.0 - gate) * (1.0 - 2.0 * gate)
            / (eps * eps);
        assert!(
            entry.is_finite(),
            "JumpReLU hessian_diag must be finite at index {idx}; entry={entry}"
        );
        assert_abs_diff_eq!(entry, expected, epsilon = 1e-12);
        if entry < 0.0 {
            saw_negative = true;
        }
    }
    // The true Hessian is genuinely indefinite past the gate inflection
    // (g > ½); the sweep must exercise that sign change so this test would
    // catch any regression back to the always-nonnegative PSD surrogate.
    assert!(
        saw_negative,
        "true JumpReLU hessian_diag must go negative once the gate passes g = ½"
    );
}

#[test]
fn jumprelu_hvp_diagonal_matches_hessian_diag() {
    let (pen, target_values, rho, _scaled_thresholds, _eps, _weight) = jumprelu_sweep_fixture();
    // `hvp` and `hessian_diag` are the SAME true operator; probing `hvp`
    // with unit vectors must reproduce `hessian_diag` exactly.
    let diag = pen
        .hessian_diag(target_values.view(), rho.view())
        .expect("JumpReLU exposes an analytic diagonal Hessian");
    for i in 0..target_values.len() {
        let mut e_i = Array1::<f64>::zeros(target_values.len());
        e_i[i] = 1.0;
        let hv_i = pen.hvp(target_values.view(), rho.view(), e_i.view());
        assert_abs_diff_eq!(diag[i], hv_i[i], epsilon = 1e-12);
    }
}

#[test]
fn jumprelu_psd_majorizer_diag_is_psd_over_logit_sweep() {
    let (pen, target_values, rho, scaled_thresholds, eps, weight) = jumprelu_sweep_fixture();
    let latent_dim = scaled_thresholds.len();
    // The PSD majorizer is a DISTINCT operator from the true Hessian. The
    // bare re-weighted-ℓ₂ surrogate wτ·[g(1−g)]²/ε² is ≥ 0 but does NOT
    // dominate the indefinite exact Hessian wτ·g(1−g)(1−2g)/ε² for gates
    // g < (3−√5)/2 (where the exact curvature is positive and larger). The
    // majorizer is therefore the elementwise max of that surrogate and the
    // absolute exact Hessian |h| = wτ·g(1−g)|1−2g|/ε²: PSD, finite, and a
    // genuine upper bound `B ⪰ ∂²P` everywhere. The Newton / PIRLS curvature
    // block consumes this, not `hessian_diag`.
    let diag = pen
        .psd_majorizer_diag(target_values.view(), rho.view())
        .expect("JumpReLU exposes a PSD diagonal majorizer");
    let exact = pen
        .hessian_diag(target_values.view(), rho.view())
        .expect("JumpReLU exposes a closed-form diagonal Hessian");

    for (idx, &entry) in diag.iter().enumerate() {
        let axis = idx % latent_dim;
        let gate = pen.sigmoid_gate((target_values[idx] - scaled_thresholds[axis]) / eps);
        let slope = gate * (1.0 - gate);
        let reweighted_l2 = slope * slope;
        let abs_exact = slope * (1.0 - 2.0 * gate).abs();
        let expected =
            weight * scaled_thresholds[axis] * reweighted_l2.max(abs_exact) / (eps * eps);
        assert!(
            entry.is_finite() && entry >= 0.0,
            "JumpReLU psd_majorizer_diag must be finite and PSD at index {idx}; entry={entry}"
        );
        assert_abs_diff_eq!(entry, expected, epsilon = 1e-12);
        // The defining contract: the majorizer dominates the exact Hessian.
        assert!(
            entry + 1e-12 >= exact[idx],
            "majorizer {entry} must dominate exact Hessian {} at index {idx}",
            exact[idx]
        );
    }
}

#[test]
fn log_sparsity_hessian_is_exact_true_second_derivative() {
    // Log sparsifier  P(x) = λ·log(1 + x²/δ²),  P'(x) = 2λx/(δ²+x²).
    // The EXACT second derivative is
    //   P''(x) = 2λ(δ² − x²)/(δ² + x²)²,
    // which is NEGATIVE for |x| > δ (Log is nonconvex). `hessian_diag`
    // and `hvp` must return this genuine (indefinite) Hessian — never the
    // positive IRLS majorizer 2λ/(δ²+x²). This guards against the operator
    // confusion in issue #444.
    let delta = 0.5_f64;
    let weight = 1.3_f64;
    let log_lambda = 0.2_f64;
    let lambda = weight * log_lambda.exp();
    let d2 = delta * delta;
    let pen = {
        let mut p = SparsityPenalty::log(PenaltyTier::Psi, delta).expect("valid log sparsity");
        p.weight = weight;
        p
    };
    // Sweep across |x| < δ (positive curvature) and |x| > δ (negative).
    let target = array![0.0_f64, 0.25, 0.5, 1.0, 2.0, -2.0, -0.1];
    let rho = array![log_lambda];

    let diag = pen
        .hessian_diag(target.view(), rho.view())
        .expect("log sparsity exposes an analytic diagonal Hessian");
    let mut saw_negative = false;
    for (i, &x) in target.iter().enumerate() {
        let denom = d2 + x * x;
        let expected = 2.0 * lambda * (d2 - x * x) / (denom * denom);
        assert_abs_diff_eq!(diag[i], expected, epsilon = 1e-12);
        // `hvp` probed with eᵢ must reproduce the same diagonal entry: the
        // two are the SAME true operator.
        let mut e_i = Array1::<f64>::zeros(target.len());
        e_i[i] = 1.0;
        let hv_i = pen.hvp(target.view(), rho.view(), e_i.view());
        assert_abs_diff_eq!(hv_i[i], expected, epsilon = 1e-12);
        if diag[i] < 0.0 {
            saw_negative = true;
        }
    }
    assert!(
        saw_negative,
        "true log-sparsity Hessian must go negative once |x| > δ"
    );
}

#[test]
fn log_sparsity_hessian_diag_matches_central_difference_of_gradient() {
    // The exact Hessian must EXACTLY differentiate `grad_target`. A tight
    // central difference of the analytic gradient pins the closed-form
    // diagonal (and so would catch any regression to the majorizer, whose
    // sign disagrees with the gradient's slope for |x| > δ).
    let delta = 0.7_f64;
    let weight = 0.9_f64;
    let log_lambda = -0.3_f64;
    let pen = {
        let mut p = SparsityPenalty::log(PenaltyTier::Psi, delta).expect("valid log sparsity");
        p.weight = weight;
        p
    };
    let target = array![0.0_f64, 0.3, 0.7, 1.5, -1.8];
    let rho = array![log_lambda];
    let diag = pen
        .hessian_diag(target.view(), rho.view())
        .expect("log sparsity exposes an analytic diagonal Hessian");
    let h = 1e-6_f64;
    for i in 0..target.len() {
        let mut tp = target.clone();
        let mut tm = target.clone();
        tp[i] += h;
        tm[i] -= h;
        let gp = pen.grad_target(tp.view(), rho.view());
        let gm = pen.grad_target(tm.view(), rho.view());
        let fd = (gp[i] - gm[i]) / (2.0 * h);
        assert_abs_diff_eq!(diag[i], fd, epsilon = 1e-5);
    }
}

#[test]
fn log_sparsity_psd_majorizer_diag_is_distinct_positive_operator() {
    // The PSD majorizer is a DIFFERENT operator from the exact Hessian:
    //   B(x) = 2λ/(δ²+x²) ⪰ 0,  agreeing with the exact Hessian only at
    //   x = 0 and strictly dominating it elsewhere. The Newton / PIRLS
    //   curvature block consumes this, not `hessian_diag`.
    let delta = 0.5_f64;
    let weight = 1.3_f64;
    let log_lambda = 0.2_f64;
    let lambda = weight * log_lambda.exp();
    let d2 = delta * delta;
    let pen = {
        let mut p = SparsityPenalty::log(PenaltyTier::Psi, delta).expect("valid log sparsity");
        p.weight = weight;
        p
    };
    let target = array![0.0_f64, 0.25, 0.5, 1.0, 2.0, -2.0, -0.1];
    let rho = array![log_lambda];
    let maj = pen
        .psd_majorizer_diag(target.view(), rho.view())
        .expect("log sparsity exposes a PSD diagonal majorizer");
    let exact = pen
        .hessian_diag(target.view(), rho.view())
        .expect("log sparsity exposes an analytic diagonal Hessian");
    for (i, &x) in target.iter().enumerate() {
        let expected = 2.0 * lambda / (d2 + x * x);
        assert_abs_diff_eq!(maj[i], expected, epsilon = 1e-12);
        assert!(
            maj[i] >= 0.0,
            "log-sparsity majorizer must be PSD at index {i}; entry={}",
            maj[i]
        );
        // Majorizer dominates the exact Hessian everywhere, with equality
        // only at x = 0.
        assert!(maj[i] + 1e-12 >= exact[i]);
        if x == 0.0 {
            assert_abs_diff_eq!(maj[i], exact[i], epsilon = 1e-12);
        }
    }
}

#[test]
fn ard_rho_grad_includes_occam_log_det_term() {
    let d = 2;
    let t = array![1.0_f64, 0.0, 0.0, 2.0];
    let n_obs = t.len() / d; // 2
    let target = PsiSlice::full(t.len(), Some(d));
    let ard = ARDPenalty::new(target, d);
    assert!((ard.n_eff - n_obs as f64).abs() < 1e-12);
    let rho = array![0.0_f64, 0.0];
    let dr = ard.grad_rho(t.view(), rho.view());
    // ∂P_j/∂ρ_j = ½ λ_j Σ t² − N_eff/2.
    // Axis 0: ½·1·(1+0) − ½·2 = −0.5.
    // Axis 1: ½·1·(0+4) − ½·2 =  1.0.
    assert!((dr[0] - (-0.5)).abs() < 1e-12);
    assert!((dr[1] - 1.0).abs() < 1e-12);
}

// ----- BlockOrthogonalityPenalty tests -----

fn block_ortho_test_target() -> Array1<f64> {
    // n_eff = 4, latent_dim = 4, row-major (n*d + a):
    // T = [[ 1, 0, 1,  0],
    //      [ 0, 1, 0,  1],
    //      [ 1, 1, 1, -1],
    //      [-1, 0, 1,  0]]
    // Groups = [[0,1], [2,3]]
    // Between-block cross Gram (T_L^T T_R) is 2x2:
    //   col0·col2 = 1*1 + 0*0 + 1*1 + (-1)*1 = 1
    //   col0·col3 = 1*0 + 0*1 + 1*(-1) + (-1)*0 = -1
    //   col1·col2 = 0*1 + 1*0 + 1*1 + 0*1 = 1
    //   col1·col3 = 0*0 + 1*1 + 1*(-1) + 0*0 = 0
    // ‖.‖_F² = 1 + 1 + 1 + 0 = 3
    array![
        1.0_f64, 0.0, 1.0, 0.0, 0.0, 1.0, 0.0, 1.0, 1.0, 1.0, 1.0, -1.0, -1.0, 0.0, 1.0, 0.0
    ]
}

#[test]
fn block_orthogonality_value_matches_offdiag_gram_frobenius() {
    let t = block_ortho_test_target();
    let target = PsiSlice::full(t.len(), Some(4));
    let pen =
        BlockOrthogonalityPenalty::new(target, vec![vec![0_usize, 1], vec![2, 3]], 2.5, 4, false)
            .expect("valid block orthogonality penalty");
    let rho = array![0.0_f64];
    let v = pen.value(t.view(), rho.view());
    // value = 0.5 · w · 3.0 = 0.5 · 2.5 · 3.0 = 3.75
    assert!(v.is_finite(), "block-orthogonality value must be finite");
    assert_abs_diff_eq!(v, 3.75, epsilon = 1e-12);
}

#[test]
fn block_orthogonality_grad_matches_finite_difference() {
    let t = block_ortho_test_target();
    let n = t.len();
    let target = PsiSlice::full(n, Some(4));
    let pen =
        BlockOrthogonalityPenalty::new(target, vec![vec![0_usize, 1], vec![2, 3]], 1.25, 4, false)
            .expect("valid block orthogonality penalty");
    let rho = array![0.0_f64];
    let g = pen.grad_target(t.view(), rho.view());
    let eps = 1e-6;
    let fd = gam_test_support::fd_checker::numerical_gradient_central_diff(
        |tv| pen.value(tv.view(), rho.view()),
        &t,
        eps,
    );
    let mut max_err = 0.0_f64;
    for i in 0..n {
        let err = (g[i] - fd[i]).abs();
        if err > max_err {
            max_err = err;
        }
        assert_abs_diff_eq!(g[i], fd[i], epsilon = 1e-6);
    }
    assert!(max_err < 1e-6, "grad-FD max abs error = {max_err:.3e}");
}

#[test]
fn block_orthogonality_hvp_matches_gradient_directional_derivative() {
    let t = block_ortho_test_target();
    let n = t.len();
    let target = PsiSlice::full(n, Some(4));
    let pen =
        BlockOrthogonalityPenalty::new(target, vec![vec![0_usize, 1], vec![2, 3]], 0.75, 4, false)
            .expect("valid block orthogonality penalty");
    let rho = array![0.0_f64];
    // Pick a non-trivial probe direction.
    let v: Array1<f64> = Array1::from_vec((0..n).map(|i| 0.3 * ((i as f64) + 1.0).sin()).collect());
    let hv = pen.hvp(t.view(), rho.view(), v.view());
    let eps = 1e-5;
    let mut tp = t.clone();
    let mut tm = t.clone();
    for i in 0..n {
        tp[i] += eps * v[i];
        tm[i] -= eps * v[i];
    }
    let gp = pen.grad_target(tp.view(), rho.view());
    let gm = pen.grad_target(tm.view(), rho.view());
    let mut max_err = 0.0_f64;
    for i in 0..n {
        let fd = (gp[i] - gm[i]) / (2.0 * eps);
        let err = (hv[i] - fd).abs();
        if err > max_err {
            max_err = err;
        }
        assert_abs_diff_eq!(hv[i], fd, epsilon = 1e-5);
    }
    assert!(max_err < 1e-5, "hvp-FD max abs error = {max_err:.3e}");
}

#[test]
fn block_orthogonality_hessian_diag_matches_finite_difference() {
    let t = block_ortho_test_target();
    let n = t.len();
    let target = PsiSlice::full(n, Some(4));
    let pen =
        BlockOrthogonalityPenalty::new(target, vec![vec![0_usize, 1], vec![2, 3]], 0.9, 4, false)
            .expect("valid block orthogonality penalty");
    let rho = array![0.0_f64];
    let diag = pen
        .hessian_diag(t.view(), rho.view())
        .expect("hessian_diag must be available");
    assert_eq!(diag.len(), n);
    let eps = 1e-5;
    for i in 0..n {
        let mut tp = t.clone();
        let mut tm = t.clone();
        tp[i] += eps;
        tm[i] -= eps;
        let gp = pen.grad_target(tp.view(), rho.view())[i];
        let gm = pen.grad_target(tm.view(), rho.view())[i];
        let fd = (gp - gm) / (2.0 * eps);
        assert_abs_diff_eq!(diag[i], fd, epsilon = 1e-5);
        assert!(
            diag[i] >= 0.0,
            "hessian_diag entry must be PSD; got {}",
            diag[i]
        );
    }
}

#[test]
fn block_orthogonality_rejects_groups_missing_an_axis() {
    // latent_dim derived as len/n_eff = 16/4 = 4; groups cover only axes
    // 0,1,2 → axis 3 is missing.
    let t = block_ortho_test_target();
    let target = PsiSlice::full(t.len(), Some(4));
    let err =
        BlockOrthogonalityPenalty::new(target, vec![vec![0_usize, 1], vec![2]], 1.0, 4, false)
            .expect_err("groups missing axis 3 must error");
    assert!(
        err.contains("must partition latent axes") && err.contains("missing axis 3"),
        "unexpected error message: {err}"
    );
}

// ----- MechanismSparsityPenalty tests -----

fn mech_sparsity_test_target() -> Array1<f64> {
    // latent_dim = 2, p_features = 3, row-major (latent*p + feature):
    // W = [[ 0.4, -0.3,  0.2],
    //      [-0.1,  0.6,  0.5]]
    array![0.4_f64, -0.3, 0.2, -0.1, 0.6, 0.5]
}

fn build_mech_sparsity(weight: f64) -> MechanismSparsityPenalty {
    let t = mech_sparsity_test_target();
    let target = PsiSlice::full(t.len(), Some(2));
    MechanismSparsityPenalty::new(
        target,
        vec![vec![0_usize, 1], vec![2]],
        weight,
        1e-2,
        4.0,
        false,
    )
    .expect("valid mechanism sparsity penalty")
}

#[test]
fn mechanism_sparsity_value_matches_group_norm_sum() {
    let pen = build_mech_sparsity(1.5);
    let t = mech_sparsity_test_target();
    let rho = array![0.0_f64];
    let v = pen.value(t.view(), rho.view());
    // For each latent row, sum_g sqrt(|g|) · sqrt(Σ x² + ε²)
    let eps2 = 1e-2_f64 * 1e-2_f64;
    let sqrt2 = 2.0_f64.sqrt();
    // Latent 0: group{0,1} → sqrt(2)·sqrt(0.16+0.09+eps²); group{2} → 1·sqrt(0.04+eps²)
    let l0 = sqrt2 * (0.16_f64 + 0.09 + eps2).sqrt() + (0.04_f64 + eps2).sqrt();
    // Latent 1: group{0,1} → sqrt(2)·sqrt(0.01+0.36+eps²); group{2} → 1·sqrt(0.25+eps²)
    let l1 = sqrt2 * (0.01_f64 + 0.36 + eps2).sqrt() + (0.25_f64 + eps2).sqrt();
    let expected = 1.5 * (l0 + l1);
    assert!(v.is_finite(), "mechanism-sparsity value must be finite");
    assert_abs_diff_eq!(v, expected, epsilon = 1e-12);
}

#[test]
fn mechanism_sparsity_grad_matches_finite_difference() {
    let pen = build_mech_sparsity(0.8);
    let t = mech_sparsity_test_target();
    let n = t.len();
    let rho = array![0.0_f64];
    let g = pen.grad_target(t.view(), rho.view());
    let eps = 1e-6;
    let fd = gam_test_support::fd_checker::numerical_gradient_central_diff(
        |tv| pen.value(tv.view(), rho.view()),
        &t,
        eps,
    );
    let mut max_err = 0.0_f64;
    for i in 0..n {
        let err = (g[i] - fd[i]).abs();
        if err > max_err {
            max_err = err;
        }
        assert_abs_diff_eq!(g[i], fd[i], epsilon = 1e-6);
    }
    assert!(max_err < 1e-6, "grad-FD max abs error = {max_err:.3e}");
}

#[test]
fn mechanism_sparsity_hvp_matches_gradient_directional_derivative() {
    let pen = build_mech_sparsity(0.5);
    let t = mech_sparsity_test_target();
    let n = t.len();
    let rho = array![0.0_f64];
    let v: Array1<f64> = Array1::from_vec((0..n).map(|i| 0.2 * ((i as f64) + 1.3).cos()).collect());
    let hv = pen.hvp(t.view(), rho.view(), v.view());
    let eps = 1e-5;
    let mut tp = t.clone();
    let mut tm = t.clone();
    for i in 0..n {
        tp[i] += eps * v[i];
        tm[i] -= eps * v[i];
    }
    let gp = pen.grad_target(tp.view(), rho.view());
    let gm = pen.grad_target(tm.view(), rho.view());
    let mut max_err = 0.0_f64;
    for i in 0..n {
        let fd = (gp[i] - gm[i]) / (2.0 * eps);
        let err = (hv[i] - fd).abs();
        if err > max_err {
            max_err = err;
        }
        assert_abs_diff_eq!(hv[i], fd, epsilon = 1e-5);
    }
    assert!(max_err < 1e-5, "hvp-FD max abs error = {max_err:.3e}");
}

#[test]
fn mechanism_sparsity_rejects_groups_missing_a_feature() {
    // groups cover features {0, 2} only → feature 1 missing.
    let t = mech_sparsity_test_target();
    let target = PsiSlice::full(t.len(), Some(2));
    let err =
        MechanismSparsityPenalty::new(target, vec![vec![0_usize], vec![2]], 1.0, 1e-2, 4.0, false)
            .expect_err("groups missing feature 1 must error");
    assert!(
        err.contains("must partition features") && err.contains("missing feature 1"),
        "unexpected error message: {err}"
    );
}

#[test]
fn mechanism_sparsity_rejects_overlapping_groups() {
    // Feature 1 appears in two groups → must error before reaching the
    // missing-feature path.
    let t = mech_sparsity_test_target();
    let target = PsiSlice::full(t.len(), Some(2));
    let err = MechanismSparsityPenalty::new(
        target,
        vec![vec![0_usize, 1], vec![1, 2]],
        1.0,
        1e-2,
        4.0,
        false,
    )
    .expect_err("overlapping feature must error");
    assert!(
        err.contains("feature 1 appears in more than one group"),
        "unexpected error message: {err}"
    );
}

// ----- NestedPrefixPenalty (Matryoshka SAE) tests -----

fn nested_prefix_test_target() -> (Array1<f64>, usize, usize) {
    // n_rows = 3, latent_dim = 4. Row-major (n*F + i).
    let t = array![
        1.0_f64, 2.0, 3.0, 4.0, // row 0
        -1.0, 0.5, 0.0, 2.0, // row 1
        0.1, -0.2, 0.3, -0.4, // row 2
    ];
    (t, 3, 4)
}

#[test]
fn nested_prefix_grad_matches_finite_difference() {
    let (t, _n, f) = nested_prefix_test_target();
    let target = PsiSlice::full(t.len(), Some(f));
    let pen = NestedPrefixPenalty::new(
        target,
        PenaltyTier::Psi,
        vec![1_usize, 2, 4],
        vec![0.7, 0.5, 0.3],
        1e-3,
    )
    .expect("valid nested-prefix penalty");
    let rho = array![0.0_f64, 0.0, 0.0];
    let g = pen.grad_target(t.view(), rho.view());
    let eps = 1e-6;
    let fd = gam_test_support::fd_checker::numerical_gradient_central_diff(
        |tv| pen.value(tv.view(), rho.view()),
        &t,
        eps,
    );
    let mut max_err = 0.0_f64;
    for i in 0..t.len() {
        let err = (g[i] - fd[i]).abs();
        if err > max_err {
            max_err = err;
        }
        assert_abs_diff_eq!(g[i], fd[i], epsilon = 1e-5);
    }
    assert!(max_err < 1e-5, "grad-FD max abs error = {max_err:.3e}");
}

#[test]
fn nested_prefix_hessian_diag_is_psd() {
    let (t, _n, f) = nested_prefix_test_target();
    let target = PsiSlice::full(t.len(), Some(f));
    let pen = NestedPrefixPenalty::new(
        target,
        PenaltyTier::Psi,
        vec![2_usize, 3, 4],
        vec![1.0, 0.5, 0.25],
        1e-3,
    )
    .expect("valid nested-prefix penalty");
    let rho = array![0.0_f64, 0.0, 0.0];
    let h = pen
        .hessian_diag(t.view(), rho.view())
        .expect("nested-prefix Hessian is diagonal");
    for &v in h.iter() {
        assert!(
            v >= 0.0 && v.is_finite(),
            "Hessian diag must be finite and PSD; got {v}"
        );
    }
    assert!(h[0] > 0.0);
}

#[test]
fn nested_prefix_mask_is_correct() {
    let (t, n_rows, f) = nested_prefix_test_target();
    let target = PsiSlice::full(t.len(), Some(f));
    let prefixes = vec![1_usize, 3, 4];
    let weights = vec![2.0_f64, 1.0, 0.5];
    let eps = 0.5;
    let pen =
        NestedPrefixPenalty::new(target, PenaltyTier::Psi, prefixes, weights, eps).expect("valid");
    let rho = Array1::<f64>::zeros(3);
    let v = pen.value(t.view(), rho.view());

    // W_i = Σ_{k: m_k > i} λ_k.
    //   axis 0: m_k ∈ {1,3,4} > 0 → 2+1+0.5 = 3.5
    //   axis 1: {3,4} > 1 → 1+0.5 = 1.5
    //   axis 2: {3,4} > 2 → 1+0.5 = 1.5
    //   axis 3: {4} > 3 → 0.5
    let w_axis = [3.5_f64, 1.5, 1.5, 0.5];
    let mut expected = 0.0;
    let eps2 = eps * eps;
    let src = t.as_slice().unwrap();
    for n in 0..n_rows {
        for i in 0..f {
            let x = src[n * f + i];
            expected += w_axis[i] * (x * x + eps2).sqrt();
        }
    }
    assert_abs_diff_eq!(v, expected, epsilon = 1e-10);
}

#[test]
fn nested_prefix_grad_rho_matches_finite_difference() {
    let (t, _n, f) = nested_prefix_test_target();
    let target = PsiSlice::full(t.len(), Some(f));
    let pen = NestedPrefixPenalty::new(
        target,
        PenaltyTier::Psi,
        vec![1_usize, 2, 4],
        vec![0.7, 0.5, 0.3],
        1e-3,
    )
    .expect("valid");
    let rho = array![0.1_f64, -0.2, 0.3];
    let dr = pen.grad_rho(t.view(), rho.view());
    let eps = 1e-6;
    for k in 0..3 {
        let mut rp = rho.clone();
        let mut rm = rho.clone();
        rp[k] += eps;
        rm[k] -= eps;
        let fd = (pen.value(t.view(), rp.view()) - pen.value(t.view(), rm.view())) / (2.0 * eps);
        assert_abs_diff_eq!(dr[k], fd, epsilon = 1e-5);
    }
}

/// Central-difference `(value(t+ε)−value(t−ε))/2ε` per coordinate against the
/// analytic `grad_target`: the single self-consistency check that pins "the
/// gradient is the gradient of the value". Returns the worst absolute error.
fn value_grad_fd_max_abs_error(
    pen: &dyn AnalyticPenalty,
    target: ArrayView1<'_, f64>,
    rho: ArrayView1<'_, f64>,
    epsilon: f64,
) -> f64 {
    let grad = pen.grad_target(target, rho);
    let mut worst = 0.0_f64;
    let mut tp = target.to_owned();
    let mut tm = target.to_owned();
    for i in 0..target.len() {
        let base = target[i];
        tp[i] = base + epsilon;
        tm[i] = base - epsilon;
        let fd = (pen.value(tp.view(), rho) - pen.value(tm.view(), rho)) / (2.0 * epsilon);
        tp[i] = base;
        tm[i] = base;
        let err = (grad[i] - fd).abs();
        if err > worst {
            worst = err;
        }
    }
    worst
}

#[test]
fn ard_value_grad_self_consistent_fd() {
    let d = 2;
    let target = PsiSlice::full(8, Some(d));
    let ard = ARDPenalty::new(target, d);
    let t = array![0.5_f64, 1.0, 2.0, -1.0, 0.3, -0.7, 1.4, -0.2];
    let rho = array![0.4_f64, -0.6];
    let worst = value_grad_fd_max_abs_error(&ard, t.view(), rho.view(), 1.0e-6);
    assert!(
        worst <= 1.0e-7,
        "ARD value↔grad FD max abs error = {worst:.3e}"
    );
}

#[test]
fn scadmcp_value_grad_self_consistent_fd() {
    // Targets straddle both MCP regimes (|t| ≤ γw active, > γw flat).
    let n_eff = 6usize;
    let target = PsiSlice::full(n_eff, Some(1));
    let pen = ScadMcpPenalty::new(
        target,
        0.5,
        n_eff,
        3.0,
        1.0e-4,
        PenaltyConcavity::Mcp,
        false,
    )
    .unwrap();
    let t = array![0.02_f64, 0.4, 0.9, 1.6, -1.1, -0.05];
    let rho = Array1::<f64>::zeros(0);
    // eps = 1e-4 dominates curvature near t≈0, so use a step well above eps.
    let worst = value_grad_fd_max_abs_error(&pen, t.view(), rho.view(), 1.0e-3);
    assert!(
        worst <= 1.0e-5,
        "ScadMcp value↔grad FD max abs error = {worst:.3e}"
    );
}

#[test]
fn nuclear_norm_value_grad_self_consistent_fd() {
    let n_eff = 4usize;
    let p = 3usize;
    let target = PsiSlice {
        range: 0..n_eff * p,
        latent_dim: Some(p),
    };
    let pen = NuclearNormPenalty::new(target, 0.8, n_eff, 1.0e-4, None, false).unwrap();
    let t = array![
        1.2_f64, -0.4, 0.3, 0.1, 0.9, -0.7, -0.5, 0.2, 1.1, 0.6, -0.3, 0.8
    ];
    let rho = Array1::<f64>::zeros(0);
    let worst = value_grad_fd_max_abs_error(&pen, t.view(), rho.view(), 1.0e-6);
    assert!(
        worst <= 1.0e-5,
        "NuclearNorm value↔grad FD max abs error = {worst:.3e}"
    );
}

#[test]
fn nuclear_norm_hvp_wide_matrix_max_rank_above_thin_rank_is_uncapped() {
    // n_eff < latent_dim gives a permanent right-nullspace in T^T T. A
    // max_rank above the thin SVD rank does not truncate the value/gradient,
    // so the HVP must be the full smoothed nuclear-norm derivative too. The
    // old right-Gram-width cap selected three of four right eigenvectors and
    // split the tied zero-eigenvalue nullspace.
    let n_eff = 2usize;
    let p = 4usize;
    let target = PsiSlice {
        range: 0..n_eff * p,
        latent_dim: Some(p),
    };
    let capped =
        NuclearNormPenalty::new(target.clone(), 0.7, n_eff, 1.0e-3, Some(3), false).unwrap();
    let uncapped = NuclearNormPenalty::new(target, 0.7, n_eff, 1.0e-3, None, false).unwrap();
    let t = array![2.0_f64, 0.0, 0.0, 0.0, 0.0, 1.5, 0.0, 0.0];
    let v = array![0.2_f64, -0.4, 0.6, -0.8, 0.3, -0.5, 0.7, -0.9];
    let rho = Array1::<f64>::zeros(0);

    let hv_capped = capped.hvp(t.view(), rho.view(), v.view());
    let hv_uncapped = uncapped.hvp(t.view(), rho.view(), v.view());
    for i in 0..t.len() {
        assert!(
            hv_capped[i].is_finite(),
            "wide NuclearNorm HVP must stay finite at index {i}"
        );
        assert_abs_diff_eq!(hv_capped[i], hv_uncapped[i], epsilon = 1.0e-10);
    }
}

#[test]
fn nuclear_norm_wide_block_max_rank_above_true_rank_value_grad_hvp_are_finite() {
    // Regression for #742: with a 3x10 block and max_rank=4, the active SVD
    // rank is still the thin rank 3. HVP must not use the right-Gram width
    // cutoff, which would split the seven-dimensional right nullspace.
    let n_eff = 3usize;
    let p = 10usize;
    let target = PsiSlice {
        range: 0..n_eff * p,
        latent_dim: Some(p),
    };
    let pen = NuclearNormPenalty::new(target, 0.9, n_eff, 1.0e-3, Some(4), false).unwrap();
    let t = array![
        2.0_f64, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, //
        0.0, 1.5, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, //
        0.0, 0.0, 1.2, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0
    ];
    let v = Array1::from_vec(
        (0..n_eff * p)
            .map(|i| 0.2 * ((i as f64) + 0.3).cos())
            .collect(),
    );
    let rho = Array1::<f64>::zeros(0);

    let value = pen.value(t.view(), rho.view());
    let grad = pen.grad_target(t.view(), rho.view());
    let hv = pen.hvp(t.view(), rho.view(), v.view());

    assert!(value.is_finite(), "wide NuclearNorm value must be finite");
    for i in 0..t.len() {
        assert!(
            grad[i].is_finite(),
            "wide NuclearNorm gradient must be finite at index {i}"
        );
        assert!(
            hv[i].is_finite(),
            "wide NuclearNorm HVP must be finite at index {i}"
        );
    }
}

#[test]
fn nuclear_norm_right_gram_divided_difference_uses_eigen_floor() {
    let n_eff = 2usize;
    let p = 2usize;
    let target = PsiSlice {
        range: 0..n_eff * p,
        latent_dim: Some(p),
    };
    let smoothing_eps = 1.0e-10_f64;
    let pen = NuclearNormPenalty::new(target, 0.9, n_eff, smoothing_eps, None, false).unwrap();
    let a = 1.0e-10_f64;
    let b = 2.0e-7_f64;
    let t = array![[a, 0.0_f64], [0.0, b]];
    let v = array![[0.0_f64, 1.0], [0.0, 0.0]];

    let (_right_filter, right_filter_derivative) = pen
        .right_spectral_inverse_sqrt_derivative(t.view(), v.view())
        .expect("right-Gram derivative");

    let eps2 = smoothing_eps * smoothing_eps;
    let eig_floor = eps2.max(1.0e-15);
    let lambda0 = (a * a + eps2).max(eig_floor);
    let lambda1 = (b * b + eps2).max(eig_floor);
    let f0 = lambda0.powf(-0.5);
    let f1 = lambda1.powf(-0.5);
    let expected = ((f0 - f1) / (lambda0 - lambda1)) * a;

    assert_abs_diff_eq!(
        right_filter_derivative[[0, 1]],
        expected,
        epsilon = expected.abs() * 1.0e-12
    );
}

/// The subspace fast path (joint-rowspace eigenproblem, used when the
/// block is wide: `d > 2m + 8`) must reproduce the dense `d×d` route
/// exactly — same regularized spectrum, same active window, same
/// divided-difference pair rules. Checked for the default no-cap window
/// (where the S⊥ zero class is ACTIVE and carries the constant f₀
/// filter) and for a biting `max_rank` (where S⊥ is inactive).
#[test]
fn nuclear_norm_wide_block_fast_path_matches_dense_oracle() {
    let n_eff = 3usize;
    let p = 40usize; // wide: p > 2*n_eff + 8 engages the subspace path
    for max_rank in [None, Some(2)] {
        let target = PsiSlice {
            range: 0..n_eff * p,
            latent_dim: Some(p),
        };
        let pen = NuclearNormPenalty::new(target, 0.8, n_eff, 1.0e-3, max_rank, false).unwrap();
        let t_flat = Array1::from_vec(
            (0..n_eff * p)
                .map(|i| (0.3 * (i as f64) + 0.11).sin() + 0.05 * (i as f64 % 7.0))
                .collect(),
        );
        let v_flat = Array1::from_vec(
            (0..n_eff * p)
                .map(|i| (0.17 * (i as f64) - 0.4).cos())
                .collect(),
        );
        let t = t_flat.view().into_shape_with_order((n_eff, p)).unwrap();
        let v = v_flat.view().into_shape_with_order((n_eff, p)).unwrap();

        let (fast_vr, fast_tdr) = pen
            .right_spectral_filters_applied(t.view(), v.view())
            .expect("fast path");
        let (rf, rfd) = pen
            .right_spectral_inverse_sqrt_derivative(t.view(), v.view())
            .expect("dense oracle");
        let dense_vr = v.dot(&rf);
        let dense_tdr = t.dot(&rfd);

        let scale = dense_vr
            .iter()
            .chain(dense_tdr.iter())
            .fold(0.0_f64, |a, &x| a.max(x.abs()))
            .max(1.0);
        // The fast subspace path and the dense oracle compute the SAME spectral
        // filter via different algorithms. For this near-degenerate wide block
        // the right inverse-sqrt filter R (and its derivative dR) amplify a
        // near-zero singular direction enormously — individual V·R / T·dR
        // entries reach ~1e3, i.e. the operation is conditioned at ~1e6. f64
        // rounding therefore leaves the two algorithms agreeing to ~eps·κ ≈ 2e-9
        // *relative to the amplified entry*, which a purely global-scale 1e-9
        // bound cannot see (1e-9·scale was ~1.1e-6 while the entry itself is
        // ~1126, so a 2.3e-6 = 2e-9-relative gap tripped it). Use a per-element
        // relative slack in addition to the global-scale term so the
        // well-conditioned bulk still holds the tight 1e-9 bar while the
        // condition-amplified entries are compared at their own scale. This is
        // not a blanket loosening: it is the correct relative-error model for an
        // inverse-sqrt-derivative comparison.
        let tol = |elem: f64| 1.0e-9 * scale + 1.0e-8 * elem.abs();
        for n in 0..n_eff {
            for a in 0..p {
                assert!(
                    (fast_vr[[n, a]] - dense_vr[[n, a]]).abs() <= tol(dense_vr[[n, a]]),
                    "V·R mismatch at ({n},{a}) max_rank={max_rank:?}: \
                         fast={} dense={}",
                    fast_vr[[n, a]],
                    dense_vr[[n, a]]
                );
                assert!(
                    (fast_tdr[[n, a]] - dense_tdr[[n, a]]).abs() <= tol(dense_tdr[[n, a]]),
                    "T·dR mismatch at ({n},{a}) max_rank={max_rank:?}: \
                         fast={} dense={}",
                    fast_tdr[[n, a]],
                    dense_tdr[[n, a]]
                );
            }
        }
    }
}

#[test]
fn nuclear_norm_wide_zero_joint_rowspace_rejects_biting_zero_tie() {
    let n_eff = 3usize;
    let p = 40usize; // wide: p > 2*n_eff + 8 engages the subspace path
    let target = PsiSlice {
        range: 0..n_eff * p,
        latent_dim: Some(p),
    };
    let pen = NuclearNormPenalty::new(target, 0.8, n_eff, 1.0e-3, Some(2), false).unwrap();
    let t = Array2::<f64>::zeros((n_eff, p));
    let v = Array2::<f64>::zeros((n_eff, p));

    let fast_err = pen
        .right_spectral_filters_applied(t.view(), v.view())
        .expect_err("fast path must reject a biting all-zero tied spectrum");
    let dense_err = pen
        .right_spectral_inverse_sqrt_derivative(t.view(), v.view())
        .expect_err("dense oracle rejects the same tied cutoff");

    assert!(
        fast_err.contains("splits a tied") && dense_err.contains("splits a tied"),
        "fast path error must preserve dense tie-guard semantics; \
             fast={fast_err}, dense={dense_err}"
    );
}

#[test]
fn nuclear_norm_hvp_truncated_rank_matches_gradient_directional_derivative() {
    let n_eff = 4usize;
    let p = 3usize;
    let target = PsiSlice {
        range: 0..n_eff * p,
        latent_dim: Some(p),
    };
    let pen = NuclearNormPenalty::new(target, 1.1, n_eff, 0.2, Some(2), false).unwrap();
    let t = array![
        2.0_f64, 0.1, -0.2, 0.3, 1.5, 0.4, -0.1, 0.2, 0.9, 0.5, -0.4, 0.7
    ];
    let v = Array1::from_vec(
        (0..t.len())
            .map(|i| 0.25 * ((i as f64) + 0.7).sin())
            .collect(),
    );
    let rho = Array1::<f64>::zeros(0);
    let hv = pen.hvp(t.view(), rho.view(), v.view());
    let eps = 1.0e-6;
    let mut tp = t.clone();
    let mut tm = t.clone();
    for i in 0..t.len() {
        tp[i] += eps * v[i];
        tm[i] -= eps * v[i];
    }
    let gp = pen.grad_target(tp.view(), rho.view());
    let gm = pen.grad_target(tm.view(), rho.view());
    let mut max_err = 0.0_f64;
    for i in 0..t.len() {
        let fd = (gp[i] - gm[i]) / (2.0 * eps);
        let err = (hv[i] - fd).abs();
        max_err = max_err.max(err);
        assert_abs_diff_eq!(hv[i], fd, epsilon = 1.0e-5);
    }
    assert!(
        max_err <= 1.0e-5,
        "truncated NuclearNorm HVP-FD max abs error = {max_err:.3e}"
    );
}

#[test]
fn decoder_incoherence_value_grad_self_consistent_fd() {
    let p = 3usize;
    let block_sizes = vec![2usize, 2usize];
    let total: usize = block_sizes.iter().map(|m| m * p).sum();
    let target = PsiSlice {
        range: 0..total,
        latent_dim: Some(total / p),
    };
    let mut coact = Array2::<f64>::from_elem((2, 2), 0.0);
    coact[[0, 1]] = 0.6;
    coact[[1, 0]] = 0.6;
    coact[[0, 0]] = 1.0;
    coact[[1, 1]] = 1.0;
    let pen = DecoderIncoherencePenalty::new(target, block_sizes, p, coact, 0.7, false).unwrap();
    let t = array![
        0.5_f64, -0.3, 0.2, 0.8, -0.1, 0.4, -0.6, 0.7, 0.1, -0.2, 0.9, 0.3
    ];
    let rho = Array1::<f64>::zeros(0);
    let worst = value_grad_fd_max_abs_error(&pen, t.view(), rho.view(), 1.0e-6);
    assert!(
        worst <= 1.0e-5,
        "DecoderIncoherence value↔grad FD max abs error = {worst:.3e}"
    );
}

#[test]
fn decoder_incoherence_heterogeneous_blocks_use_output_space_cross_gram() {
    let p = 3usize;
    let block_sizes = vec![2usize, 1usize];
    let total: usize = block_sizes.iter().map(|m| m * p).sum();
    let target = PsiSlice {
        range: 0..total,
        latent_dim: Some(total / p),
    };
    let mut coact = Array2::<f64>::zeros((2, 2));
    coact[[0, 1]] = 0.2;
    coact[[1, 0]] = 0.6;
    let pen = DecoderIncoherencePenalty::new(target, block_sizes, p, coact, 2.0, false).unwrap();
    // Stored decoder blocks are B0 = [[1,0,0], [0,2,0]] and
    // B1 = [[3,0,4]]. The output-space cross-Gram is B0·B1^T = [[3], [0]],
    // so ||B0·B1^T||_F^2 = 9. The symmetrized coactivation is 0.4.
    let beta = array![1.0_f64, 0.0, 0.0, 0.0, 2.0, 0.0, 3.0, 0.0, 4.0];
    let rho = Array1::<f64>::zeros(0);
    let value = pen.value(beta.view(), rho.view());
    assert_abs_diff_eq!(value, 3.6, epsilon = 1.0e-12);
}

#[test]
fn decoder_incoherence_rejects_negative_coactivation() {
    let p = 2usize;
    let block_sizes = vec![1usize, 1usize];
    let target = PsiSlice {
        range: 0..4,
        latent_dim: Some(2),
    };
    let mut coact = Array2::<f64>::zeros((2, 2));
    coact[[0, 1]] = -0.1;
    let err = DecoderIncoherencePenalty::new(target, block_sizes, p, coact, 1.0, false)
        .expect_err("negative coactivation must be rejected");
    assert_eq!(
        err,
        "DecoderIncoherencePenalty::new requires finite non-negative coactivation entries"
    );
}

#[test]
fn decoder_incoherence_separability_semantics() {
    // Two atoms, each a single decoder column in ℝ^{p_out=2}: atom 0 = first
    // column, atom 1 = second. β layout is [B_0[:,0]=t0,t1 ; B_1[:,0]=t2,t3].
    let p = 2usize;
    let block_sizes = vec![1usize, 1usize];
    let total: usize = block_sizes.iter().map(|m| m * p).sum();
    let target = PsiSlice {
        range: 0..total,
        latent_dim: Some(total / p),
    };
    let full_coact = || {
        let mut c = Array2::<f64>::zeros((2, 2));
        c[[0, 1]] = 1.0;
        c[[1, 0]] = 1.0;
        c
    };
    let rho = Array1::<f64>::zeros(0);

    // Orthogonal output-space decoder directions ⇒ B_0·B_1^T = 0 ⇒ P ≈ 0.
    let pen_ortho = DecoderIncoherencePenalty::new(
        target.clone(),
        block_sizes.clone(),
        p,
        full_coact(),
        1.0,
        false,
    )
    .unwrap();
    let t_ortho = array![1.0_f64, 0.0, 0.0, 1.0];
    let p_ortho = pen_ortho.value(t_ortho.view(), rho.view());
    assert!(
        p_ortho.abs() <= 1.0e-12,
        "orthogonal decoder blocks must give P≈0, got {p_ortho:.3e}"
    );

    // Coincident output-space decoder directions ⇒ B_0·B_1^T large ⇒ P large.
    let pen_coinc = DecoderIncoherencePenalty::new(
        target.clone(),
        block_sizes.clone(),
        p,
        full_coact(),
        1.0,
        false,
    )
    .unwrap();
    let t_coinc = array![1.0_f64, 0.0, 1.0, 0.0];
    let p_coinc = pen_coinc.value(t_coinc.view(), rho.view());
    // ½·w·W·‖B_0·B_1^T‖_F² = ½·1·1·(1)² = 0.5.
    assert!(
        (p_coinc - 0.5).abs() <= 1.0e-12,
        "coincident decoder blocks must give large P (=0.5 here), got {p_coinc:.3e}"
    );
    assert!(
        p_coinc > p_ortho + 1.0e-3,
        "coincident P must exceed orthogonal P"
    );

    // Co-activation weight 0 zeroes the pair's contribution even when the
    // blocks are coincident.
    let pen_zero = DecoderIncoherencePenalty::new(
        target,
        block_sizes,
        p,
        Array2::<f64>::zeros((2, 2)),
        1.0,
        false,
    )
    .unwrap();
    let p_zero = pen_zero.value(t_coinc.view(), rho.view());
    assert!(
        p_zero.abs() <= 1.0e-12,
        "zero co-activation must zero the pair contribution, got {p_zero:.3e}"
    );
    let g_zero = pen_zero.grad_target(t_coinc.view(), rho.view());
    assert!(
        g_zero.iter().all(|v| v.abs() <= 1.0e-12),
        "zero co-activation must zero the pair gradient"
    );
}

#[test]
fn decoder_incoherence_sparse_matches_dense_operator() {
    // #1026: the operator now stores SPARSE penalized pairs instead of a dense
    // K×K co-activation matrix. The sparse pairs with the same nonzero entries
    // must reproduce the dense operator's value / gradient / exact-Hessian-vector
    // product / PSD-majorizer-vector product to the last bit. We build the dense
    // operator (via `new`, fed a random symmetric non-negative K×K matrix) and a
    // sparse operator (via `new_sparse`, fed the matching `(j,k, ½(W_jk+W_kj))`
    // pairs) over a K=4 SAE decoder block layout, then compare every output.
    let p = 3usize;
    let block_sizes = vec![2usize, 1usize, 3usize, 2usize];
    let k = block_sizes.len();
    let total: usize = block_sizes.iter().map(|m| m * p).sum();
    let target = PsiSlice {
        range: 0..total,
        latent_dim: Some(total / p),
    };
    // Deterministic pseudo-random symmetric non-negative coactivation, with some
    // exact zeros (those pairs must be ABSENT from the sparse list).
    let mut coact = Array2::<f64>::zeros((k, k));
    let mut seed = 0x9E3779B97F4A7C15_u64;
    let mut next = || {
        seed ^= seed << 13;
        seed ^= seed >> 7;
        seed ^= seed << 17;
        (seed >> 11) as f64 / (1_u64 << 53) as f64
    };
    for j in 0..k {
        for kk in (j + 1)..k {
            // ~1/3 of the pairs are dropped to exactly 0 to exercise sparsity.
            let raw = next();
            let w = if raw < 0.33 { 0.0 } else { raw };
            coact[[j, kk]] = w;
            coact[[kk, j]] = w;
        }
    }
    // Matching sparse pairs: symmetrized weight, nonzero only.
    let mut pairs = Vec::new();
    for j in 0..k {
        for kk in (j + 1)..k {
            let w = 0.5 * (coact[[j, kk]] + coact[[kk, j]]);
            if w != 0.0 {
                pairs.push((j, kk, w));
            }
        }
    }

    let dense =
        DecoderIncoherencePenalty::new(target.clone(), block_sizes.clone(), p, coact, 0.7, false)
            .unwrap();
    let sparse =
        DecoderIncoherencePenalty::new_sparse(target, block_sizes, p, pairs, 0.7, false).unwrap();

    let rho = Array1::<f64>::zeros(0);
    // Deterministic pseudo-random decoder block and probe vector.
    let beta: Array1<f64> = (0..total).map(|_| next() - 0.5).collect();
    let v: Array1<f64> = (0..total).map(|_| next() - 0.5).collect();

    let vd = dense.value(beta.view(), rho.view());
    let vs = sparse.value(beta.view(), rho.view());
    assert!(
        (vd - vs).abs() <= 1.0e-12,
        "value mismatch dense={vd:.17e} sparse={vs:.17e}"
    );

    let gd = dense.grad_target(beta.view(), rho.view());
    let gs = sparse.grad_target(beta.view(), rho.view());
    let g_max = gd
        .iter()
        .zip(gs.iter())
        .map(|(a, b)| (a - b).abs())
        .fold(0.0_f64, f64::max);
    assert!(g_max <= 1.0e-12, "grad max abs mismatch = {g_max:.3e}");

    let hd = dense.hvp(beta.view(), rho.view(), v.view());
    let hs = sparse.hvp(beta.view(), rho.view(), v.view());
    let h_max = hd
        .iter()
        .zip(hs.iter())
        .map(|(a, b)| (a - b).abs())
        .fold(0.0_f64, f64::max);
    assert!(h_max <= 1.0e-12, "hvp max abs mismatch = {h_max:.3e}");

    let md = dense.psd_majorizer_hvp(beta.view(), rho.view(), v.view());
    let ms = sparse.psd_majorizer_hvp(beta.view(), rho.view(), v.view());
    let m_max = md
        .iter()
        .zip(ms.iter())
        .map(|(a, b)| (a - b).abs())
        .fold(0.0_f64, f64::max);
    assert!(
        m_max <= 1.0e-12,
        "psd_majorizer_hvp max abs mismatch = {m_max:.3e}"
    );
}

/// The direct dense scatter `accumulate_psd_majorizer_dense` must reproduce the
/// exact matrix built by the historical β-column unit-probe loop
/// (`hbb[:, j] += scale · psd_majorizer_hvp(e_j)`), bit-for-bit. That loop is the
/// `O(β·pairs)` high-K assembly cliff the SAE decoder-repulsion / incoherence
/// paths used to pay; the scatter is the `O(pairs·(M·p)²)` replacement, so it
/// must be numerically identical to remain a pure performance fix.
#[test]
fn accumulate_psd_majorizer_dense_matches_unit_probe_loop() {
    let p = 3usize;
    let block_sizes = vec![2usize, 1usize, 3usize, 2usize];
    let k = block_sizes.len();
    let total: usize = block_sizes.iter().map(|m| m * p).sum();
    let target = PsiSlice {
        range: 0..total,
        latent_dim: Some(total / p),
    };
    let mut seed = 0xD1B54A32D192ED03_u64;
    let mut next = || {
        seed ^= seed << 13;
        seed ^= seed >> 7;
        seed ^= seed << 17;
        (seed >> 11) as f64 / (1_u64 << 53) as f64
    };
    // Random nonzero symmetrized co-active pairs (some pairs dropped to zero).
    let mut pairs = Vec::new();
    for j in 0..k {
        for kk in (j + 1)..k {
            let raw = next();
            if raw >= 0.25 {
                pairs.push((j, kk, raw));
            }
        }
    }
    assert!(!pairs.is_empty(), "fixture must exercise at least one pair");
    let weight = 0.83_f64;
    let pen = DecoderIncoherencePenalty::new_sparse(target, block_sizes, p, pairs, weight, false)
        .unwrap();

    let rho = Array1::<f64>::zeros(0);
    let beta: Array1<f64> = (0..total).map(|_| next() - 0.5).collect();
    let scale = 0.7_f64;

    // Reference: the unit-probe column loop the fix replaces.
    let mut hbb_ref = Array2::<f64>::zeros((total, total));
    let mut probe = Array1::<f64>::zeros(total);
    for col in 0..total {
        probe.fill(0.0);
        probe[col] = 1.0;
        let hv = pen.psd_majorizer_hvp(beta.view(), rho.view(), probe.view());
        for row in 0..total {
            hbb_ref[[row, col]] += scale * hv[row];
        }
    }

    // Direct block scatter.
    let mut hbb = Array2::<f64>::zeros((total, total));
    pen.accumulate_psd_majorizer_dense(beta.view(), rho.view(), scale, &mut hbb);

    let max_abs = hbb
        .iter()
        .zip(hbb_ref.iter())
        .map(|(a, b)| (a - b).abs())
        .fold(0.0_f64, f64::max);
    assert!(
        max_abs <= 1.0e-12,
        "dense scatter vs unit-probe loop max abs mismatch = {max_abs:.3e}"
    );

    // The scattered curvature must be symmetric (it is a Gauss-Newton majorizer).
    let mut sym_max = 0.0_f64;
    for i in 0..total {
        for j in 0..total {
            sym_max = sym_max.max((hbb[[i, j]] - hbb[[j, i]]).abs());
        }
    }
    assert!(
        sym_max <= 1.0e-12,
        "scattered hbb not symmetric: {sym_max:.3e}"
    );
}

#[test]
fn nested_prefix_rejects_non_monotone_prefixes() {
    let target = PsiSlice::full(12, Some(4));
    let err = NestedPrefixPenalty::new(
        target,
        PenaltyTier::Psi,
        vec![2_usize, 2, 4],
        vec![1.0, 1.0, 1.0],
        1e-3,
    )
    .expect_err("non-strictly-increasing prefixes must error");
    assert!(err.contains("strictly increasing"), "got: {err}");
}

/// The harmonic-roughness penalty is the graduated diagonal periodic-Gram form:
/// `weight · Σ_r S[r mod K] · Σ_j target[r,j]²`, only on the harmonic rows.
/// Value, gradient, and Hessian diagonal must be mutually consistent (grad is
/// the exact derivative of value; the diagonal is the exact second derivative,
/// which is constant for a quadratic).
#[test]
fn harmonic_roughness_value_grad_hessian_are_consistent() {
    // K = 5 periodic basis rows: [DC, sin1, cos1, sin2, cos2]. Only h=2 (rows
    // 3, 4) is penalized with weight h⁴ = 16; DC and the fundamental are free.
    let row_weights = array![0.0, 0.0, 0.0, 16.0, 16.0];
    let n_eff = 10; // F = 2 atoms × K = 5 rows.
    let d = 3;
    let weight = 2.5;
    let penalty =
        HarmonicRoughnessPenalty::new(weight, n_eff, row_weights.clone(), false).unwrap();
    // Row-major (n_eff, d) target with distinct values per entry.
    let target: Array1<f64> = (0..n_eff * d).map(|i| 0.1 * (i as f64) - 0.7).collect();
    let rho = Array1::<f64>::zeros(0);

    // Value: only the two penalized rows of each atom contribute.
    let mut expected = 0.0;
    for r in 0..n_eff {
        let w = row_weights[r % row_weights.len()];
        for j in 0..d {
            expected += w * target[r * d + j] * target[r * d + j];
        }
    }
    expected *= weight;
    assert_abs_diff_eq!(penalty.value(target.view(), rho.view()), expected, epsilon = 1e-12);

    // Gradient matches a central finite difference of the value.
    let grad = penalty.grad_target(target.view(), rho.view());
    let h = 1e-6;
    for i in 0..target.len() {
        let mut tp = target.clone();
        let mut tm = target.clone();
        tp[i] += h;
        tm[i] -= h;
        let fd = (penalty.value(tp.view(), rho.view()) - penalty.value(tm.view(), rho.view()))
            / (2.0 * h);
        assert_abs_diff_eq!(grad[i], fd, epsilon = 1e-5);
    }

    // Hessian diagonal is the constant 2·weight·S[r mod K] on every column.
    let diag = penalty.hessian_diag(target.view(), rho.view()).unwrap();
    for r in 0..n_eff {
        let w = row_weights[r % row_weights.len()];
        for j in 0..d {
            assert_abs_diff_eq!(diag[r * d + j], 2.0 * weight * w, epsilon = 1e-12);
        }
    }
}

/// The evidence-optimal precision is the marginal-likelihood stationary point
/// `λ⋆ = N_pen / Σ_i S_ii b_i²` over the penalized coefficients only.
#[test]
fn harmonic_roughness_evidence_weight_matches_closed_form() {
    let row_weights = array![0.0, 0.0, 0.0, 16.0, 16.0];
    let n_eff = 10;
    let d = 3;
    let target: Array1<f64> = (0..n_eff * d).map(|i| 0.05 * (i as f64) + 0.3).collect();

    let mut energy = 0.0;
    let mut n_pen = 0.0;
    for r in 0..n_eff {
        let w = row_weights[r % row_weights.len()];
        if w > 0.0 {
            for j in 0..d {
                energy += w * target[r * d + j] * target[r * d + j];
                n_pen += 1.0;
            }
        }
    }
    let expected = n_pen / energy;
    let got = harmonic_roughness_evidence_weight(target.view(), n_eff, row_weights.view());
    assert_abs_diff_eq!(got, expected, epsilon = 1e-10);

    // An all-zero harmonic block has nothing to penalize → finite (floored) λ,
    // never a division by zero.
    let zeros = Array1::<f64>::zeros(n_eff * d);
    let floored = harmonic_roughness_evidence_weight(zeros.view(), n_eff, row_weights.view());
    assert!(floored.is_finite() && floored > 0.0, "got {floored}");
}
