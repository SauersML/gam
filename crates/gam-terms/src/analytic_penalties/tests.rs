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
    // The exact Hessian also reads the third decoder jet K = ∂H/∂t, which a decoder
    // rescale scales by λ too. It is symmetric in its derivative axes (a, c, e).
    let k = ndarray::Array3::<f64>::from_shape_fn((n_obs, p, d * d * d), |(row, output, axes)| {
        let (a, c, e) = (axes / (d * d), (axes / d) % d, axes % d);
        0.11 * (row as f64 + 1.0) - 0.07 * (output as f64) + 0.05 * ((a + c + e) as f64)
    });
    base.set_third_decoder_derivative(Some(Arc::new(k.clone())));
    scaled.set_third_decoder_derivative(Some(Arc::new(k * lambda)));

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
            hv0.iter().any(|x| x.abs() > 1e-9),
            "exact Hessian must be non-trivial"
        );
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
fn ordered_beta_bernoulli_assignment_grad_target_matches_value_finite_difference() {
    let pen = OrderedBetaBernoulliPenalty::new(4, 6.0, 0.8, false);
    let t = array![
        0.2_f64, -0.3, 0.7, -0.5, 0.9, 0.4, -0.2, 0.1, -0.4, 0.8, 0.3, -0.1
    ];
    let rho = Array1::<f64>::zeros(0);
    let g = pen.grad_target(t.view(), rho.view());
    let eps = 1.0e-6;
    let fd = gam_linalg_test_support::fd_checker::numerical_gradient_central_diff(
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
        "ordered Beta--Bernoulli grad-FD max abs error = {max_err:.3e}"
    );
}

#[test]
fn ordered_beta_bernoulli_value_is_exact_integrated_marginal() {
    let k = 2usize;
    let alpha = 1.7_f64;
    let tau = 0.8_f64;
    let pen = OrderedBetaBernoulliPenalty::new(k, alpha, tau, false);
    let target = array![0.2_f64, -0.3, 0.7, -0.1, 0.4, 0.5];
    let rho = Array1::<f64>::zeros(0);
    let n = target.len() / k;
    let mut mass = [0.0_f64; 2];
    for row in 0..n {
        for col in 0..k {
            mass[col] += 1.0 / (1.0 + (-target[row * k + col] / tau).exp());
        }
    }
    let ratio = alpha / (alpha + 1.0);
    let mut expected = 0.0;
    for col in 0..k {
        let mu = ratio.powi(col as i32 + 1);
        let a = mu / (1.0 - mu);
        expected += -a.ln()
            - statrs::function::gamma::ln_gamma(mass[col] + a)
            - statrs::function::gamma::ln_gamma(n as f64 - mass[col] + 1.0)
            + statrs::function::gamma::ln_gamma(n as f64 + a + 1.0);
    }
    assert_abs_diff_eq!(
        pen.value(target.view(), rho.view()),
        expected,
        epsilon = 1.0e-12
    );
}

#[test]
fn ordered_beta_bernoulli_majorized_channels_match_fd_of_psd_majorized_operator() {
    // The exact mass rank-one coefficient is strictly negative, so its zero
    // matrix is a PSD Loewner majorizer. Verify that the retained row-local
    // diagonal `max(weight·score·w_i·z_i'', 0)` and its same-row/shared-mass
    // derivative channels are one operator.
    let pen = OrderedBetaBernoulliPenalty::new(3, 5.0, 0.85, false);
    let t = array![
        0.3_f64, -0.2, 0.6, 0.5, 0.1, -0.4, -0.1, 0.7, 0.2, 0.4, -0.3, 0.8
    ];
    let rho = Array1::<f64>::zeros(0);
    let k = pen.k_max;
    let n = t.len() / k;
    let maj_diag = |tv: ArrayView1<'_, f64>| -> Array1<f64> {
        pen.psd_majorizer_logit_third_channels(tv, rho.view())
            .diagonal_term
            .mapv(|value| value.max(0.0))
    };
    let ch = pen.psd_majorizer_logit_third_channels(t.view(), rho.view());

    assert!(
        ch.mass_hessian_coefficient.iter().all(|value| *value < 0.0),
        "every exact integrated-marginal mass-Hessian coefficient must be negative"
    );

    // `m_channel` plus the local channel reproduces the total logit derivative.
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
                let local = if i == w {
                    ch.local_logit_third[idx]
                } else {
                    0.0
                };
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

/// The trait-level PSD majorizer of the ordered Beta--Bernoulli prior used to be
/// the trait default, i.e. the exact Hessian diagonal: it carries the negative
/// mass-coupled rank-one diagonal `s'_k·u_ik²` and the negative part of the
/// row-local term, so it was neither PSD nor a majorizer, and it disagreed with
/// the `max(diagonal_term, 0)` majorizer the Laplace path assembles. It must be
/// that majorizer: PSD, dominating the exact Hessian, and the frozen operator's
/// diagonal and log-determinant must be built from it.
#[test]
fn ordered_beta_bernoulli_trait_psd_majorizer_is_the_declared_loewner_majorizer() {
    let w = [1.6_f64, 0.4, 1.2, 0.8];
    let cases = [
        (
            OrderedBetaBernoulliPenalty::new(3, 5.0, 0.85, false),
            array![
                0.3_f64, -0.2, 0.6, 0.5, 0.1, -0.4, -0.1, 0.7, 0.2, 0.4, -0.3, 0.8
            ],
            Array1::<f64>::zeros(0),
        ),
        (
            OrderedBetaBernoulliPenalty::new(3, 1.7, 0.8, true).with_row_weights(Some(&w)),
            array![
                2.5_f64, -1.8, 0.6, 3.1, 0.1, -2.4, -0.1, 1.7, 0.2, 2.4, -0.3, 0.8
            ],
            array![0.15_f64],
        ),
    ];
    for (case, (pen, target, rho)) in cases.iter().enumerate() {
        let n = target.len();
        let mut hessian = Array2::<f64>::zeros((n, n));
        for j in 0..n {
            let mut e = Array1::<f64>::zeros(n);
            e[j] = 1.0;
            hessian
                .column_mut(j)
                .assign(&pen.hvp(target.view(), rho.view(), e.view()));
        }
        let declared = pen
            .psd_majorizer_logit_third_channels(target.view(), rho.view())
            .diagonal_term
            .mapv(|value| value.max(0.0));
        let diag = pen
            .psd_majorizer_diag(target.view(), rho.view())
            .expect("ordered Beta--Bernoulli majorizer diagonal");
        let exact_diag = pen
            .hessian_diag(target.view(), rho.view())
            .expect("ordered Beta--Bernoulli Hessian diagonal");
        let scale = hessian.iter().fold(0.0_f64, |acc, &v| acc.max(v.abs()));
        assert!(scale > 0.0, "case {case}: nonzero curvature");
        let tol = 1e-12 * scale;
        assert!(
            exact_diag.iter().any(|&v| v < -1e-3 * scale),
            "case {case}: the exact Hessian diagonal has negative entries"
        );
        let mut majorizer = Array2::<f64>::zeros((n, n));
        for i in 0..n {
            assert_abs_diff_eq!(diag[i], declared[i], epsilon = tol);
            assert!(diag[i] >= 0.0, "case {case}: B must be PSD");
            majorizer[[i, i]] = diag[i];
        }
        let min_eig = <Array2<f64> as PenaltyOp>::eigendecompose(&(&majorizer - &hessian))
            .expect("symmetric eigensolve")
            .0
            .iter()
            .fold(f64::INFINITY, |acc, &v| acc.min(v));
        assert!(
            min_eig >= -tol,
            "case {case}: B must dominate the exact Hessian; min eig(B - H) = {min_eig:.3e}"
        );

        let kind = AnalyticPenaltyKind::OrderedBetaBernoulli(Arc::new(pen.clone()));
        let op = FrozenAnalyticPenaltyOp::new(kind, target.clone(), rho.clone())
            .expect("frozen operator");
        let op_diag = op.diag();
        for i in 0..n {
            assert_abs_diff_eq!(op_diag[i], declared[i], epsilon = tol);
        }
        let lambda = 0.3;
        let expected: f64 = declared.iter().map(|&d| (d + lambda).ln()).sum();
        let log_det = op.log_det_plus_lambda_i(lambda).expect("frozen log det");
        assert_abs_diff_eq!(log_det, expected, epsilon = 1e-12 * expected.abs().max(1.0));
    }
}

#[test]
fn ordered_beta_bernoulli_assignment_learnable_alpha_grad_rho_matches_value_finite_difference() {
    let pen = OrderedBetaBernoulliPenalty::new(3, 6.0, 0.8, true);
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
fn ordered_beta_bernoulli_assignment_learnable_alpha_mixed_log_alpha_target_matches_fd() {
    let pen = OrderedBetaBernoulliPenalty::new(2, 2.0, 0.9, true);
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
fn ordered_beta_bernoulli_assignment_learnable_alpha_hdiag_log_alpha_derivative_matches_fd() {
    let pen = OrderedBetaBernoulliPenalty::new(2, 2.0, 0.9, true);
    let t = array![0.2_f64, -0.3, 0.7, -0.1, 0.4, 0.5];
    let rho = array![0.15_f64];
    let analytic = pen.hessian_diag_log_alpha_derivative(t.view(), rho.view());
    let step = 1.0e-6_f64;
    let rho_plus = array![rho[0] + step];
    let rho_minus = array![rho[0] - step];
    let hp = pen
        .hessian_diag(t.view(), rho_plus.view())
        .expect("ordered Beta--Bernoulli hessian diag exists");
    let hm = pen
        .hessian_diag(t.view(), rho_minus.view())
        .expect("ordered Beta--Bernoulli hessian diag exists");
    for i in 0..t.len() {
        let fd = (hp[i] - hm[i]) / (2.0 * step);
        assert_abs_diff_eq!(analytic[i], fd, epsilon = 2.0e-7);
    }
}

#[test]
fn ordered_beta_bernoulli_majorizer_log_alpha_derivative_matches_fd() {
    let pen = OrderedBetaBernoulliPenalty::new(2, 2.0, 0.9, true);
    let target = array![0.2_f64, -0.3, 0.7, -0.1, 0.4, 0.5];
    let rho = array![0.15_f64];
    let channels = pen.psd_majorizer_logit_third_channels(target.view(), rho.view());
    let raw_derivative = pen.hessian_diag_log_alpha_derivative(target.view(), rho.view());
    let mut analytic = Array1::<f64>::zeros(target.len());
    for index in 0..target.len() {
        if channels.diagonal_term[index] > 0.0 {
            let column = index % pen.k_max;
            analytic[index] = raw_derivative[index]
                - channels.mass_hessian_log_alpha_derivative[column]
                    * channels.z_jac[index]
                    * channels.z_jac[index];
        }
    }

    let step = 1.0e-6_f64;
    let plus = pen.psd_majorizer_logit_third_channels(target.view(), array![rho[0] + step].view());
    let minus = pen.psd_majorizer_logit_third_channels(target.view(), array![rho[0] - step].view());
    for index in 0..target.len() {
        let fd = (plus.diagonal_term[index].max(0.0) - minus.diagonal_term[index].max(0.0))
            / (2.0 * step);
        assert_abs_diff_eq!(analytic[index], fd, epsilon = 2.0e-6);
    }
}

#[test]
fn ordered_beta_bernoulli_assignment_extreme_logits_remain_finite() {
    let pen = OrderedBetaBernoulliPenalty::new(3, 1.5, 1.0e-3, false);
    let t = array![
        1000.0_f64, -1000.0, 500.0, -500.0, 750.0, -750.0, 250.0, -250.0, 0.0
    ];
    let rho = Array1::<f64>::zeros(0);

    let value = pen.value(t.view(), rho.view());
    assert!(
        value.is_finite(),
        "ordered Beta--Bernoulli value must remain finite for saturated concrete logits"
    );
    let grad = pen.grad_target(t.view(), rho.view());
    assert!(
        grad.iter().all(|entry| entry.is_finite()),
        "ordered Beta--Bernoulli gradient must remain finite for saturated concrete logits: {grad:?}"
    );
    let diag = pen
        .hessian_diag(t.view(), rho.view())
        .expect("ordered Beta--Bernoulli assignment exposes a diagonal Hessian");
    assert!(
        diag.iter().all(|entry| entry.is_finite()),
        "ordered Beta--Bernoulli Hessian diagonal must remain finite for saturated concrete logits: {diag:?}"
    );
}

#[test]
fn ordered_beta_bernoulli_assignment_high_k_prior_keeps_positive_gradient_path() {
    let k = 400usize;
    let pen = OrderedBetaBernoulliPenalty::new(k, 0.1, 1.0, false);
    let t = Array1::<f64>::zeros(k);
    let rho = Array1::<f64>::zeros(0);

    let value = pen.value(t.view(), rho.view());
    assert!(
        value.is_finite(),
        "high-K ordered Beta--Bernoulli value must stay finite"
    );
    let grad = pen.grad_target(t.view(), rho.view());
    assert_eq!(grad.len(), k);
    assert!(
        grad.iter().all(|entry| entry.is_finite()),
        "high-K ordered Beta--Bernoulli gradient must stay finite: {grad:?}"
    );
    assert!(
        grad.slice(s![320..]).iter().any(|entry| entry.abs() > 0.0),
        "late high-K atoms must retain a strictly positive gradient path"
    );
}

/// #991 design-honesty weighting of the ordered Beta--Bernoulli prior: every channel must be the
/// exact derivative of the ONE weighted integrated energy
/// `F_w = w·Σ_k[-log(a_k) - log B(M_k+a_k, N_eff-M_k+1)]`, where
/// `M_k=Σ_i w_i z_ik` and `N_eff=Σ_i w_i`. This is the CI gate for
/// value↔gradient↔Hessian↔ρ-channel consistency under nontrivial weights (the
/// repo's banned desync class), plus the `None ⇒ bit-for-bit` contract.
#[test]
fn ordered_beta_bernoulli_row_weighted_channels_are_one_operator_991() {
    let k = 3usize;
    // 4 rows, nontrivial mean-1 design weights.
    let w = [1.6_f64, 0.4, 1.2, 0.8];
    let t = array![
        0.3_f64, -0.2, 0.6, 0.5, 0.1, -0.4, -0.1, 0.7, 0.2, 0.4, -0.3, 0.8
    ];
    let pen = OrderedBetaBernoulliPenalty::new(k, 1.7, 0.8, true).with_row_weights(Some(&w));
    let rho = array![0.15_f64];
    let step = 1.0e-6_f64;

    // (a) gradient == FD of the weighted value.
    let g = pen.grad_target(t.view(), rho.view());
    for i in 0..t.len() {
        let mut tp = t.clone();
        let mut tm = t.clone();
        tp[i] += step;
        tm[i] -= step;
        let fd =
            (pen.value(tp.view(), rho.view()) - pen.value(tm.view(), rho.view())) / (2.0 * step);
        assert_abs_diff_eq!(g[i], fd, epsilon = 1.0e-6);
    }

    // (b) hvp along a fixed direction == FD directional derivative of the
    // weighted gradient (exercises the w_i·w_j cross-row rank-one blocks).
    let v = array![
        0.7_f64, -0.4, 0.2, 0.1, 0.9, -0.6, 0.3, -0.8, 0.5, -0.2, 0.6, 0.4
    ];
    let hv = pen.hvp(t.view(), rho.view(), v.view());
    let tp = &t + &(step * &v);
    let tm = &t - &(step * &v);
    let gp = pen.grad_target(tp.view(), rho.view());
    let gm = pen.grad_target(tm.view(), rho.view());
    for i in 0..t.len() {
        let fd = (gp[i] - gm[i]) / (2.0 * step);
        assert_abs_diff_eq!(hv[i], fd, epsilon = 1.0e-5);
    }
    // Isolate one row/column direction: the exact integrated marginal couples
    // it into every other row of the same column through M_k, while leaving
    // other columns exactly untouched.
    let mut one_site = Array1::<f64>::zeros(t.len());
    one_site[0] = 1.0;
    let cross_row = pen.hvp(t.view(), rho.view(), one_site.view());
    assert!(
        cross_row[k].abs() > 1.0e-6 && cross_row[2 * k].abs() > 1.0e-6,
        "same-column cross-row Hessian action must be nonzero: {cross_row:?}"
    );
    for row in 0..w.len() {
        for col in 1..k {
            assert_abs_diff_eq!(cross_row[row * k + col], 0.0, epsilon = 0.0);
        }
    }

    // (c) hessian_diag == the diagonal of the weighted operator (FD of grad).
    let hd = pen
        .hessian_diag(t.view(), rho.view())
        .expect("ordered Beta--Bernoulli hessian diag exists");
    for i in 0..t.len() {
        let mut tp = t.clone();
        let mut tm = t.clone();
        tp[i] += step;
        tm[i] -= step;
        let fd = (pen.grad_target(tp.view(), rho.view())[i]
            - pen.grad_target(tm.view(), rho.view())[i])
            / (2.0 * step);
        assert_abs_diff_eq!(hd[i], fd, epsilon = 1.0e-5);
    }

    // (d) learnable-α: grad_rho == FD of the weighted value in ρ, and the
    // α-diagonal derivative == FD of the weighted hessian_diag in ρ.
    let grho = pen.grad_rho(t.view(), rho.view());
    let rp = array![rho[0] + step];
    let rm = array![rho[0] - step];
    let fd_rho = (pen.value(t.view(), rp.view()) - pen.value(t.view(), rm.view())) / (2.0 * step);
    assert_abs_diff_eq!(grho[0], fd_rho, epsilon = 2.0e-6);
    let dh = pen.hessian_diag_log_alpha_derivative(t.view(), rho.view());
    let hp = pen.hessian_diag(t.view(), rp.view()).expect("hdiag");
    let hm = pen.hessian_diag(t.view(), rm.view()).expect("hdiag");
    for i in 0..t.len() {
        let fd = (hp[i] - hm[i]) / (2.0 * step);
        assert_abs_diff_eq!(dh[i], fd, epsilon = 2.0e-6);
    }

    // (e) mixed channel == FD of grad_rho in the logits (∂M/∂ℓ carries w_i).
    let mixed = pen.log_alpha_target_mixed_derivative(t.view(), rho.view());
    for i in 0..t.len() {
        let mut tp = t.clone();
        let mut tm = t.clone();
        tp[i] += step;
        tm[i] -= step;
        let fd = (pen.grad_rho(tp.view(), rho.view())[0] - pen.grad_rho(tm.view(), rho.view())[0])
            / (2.0 * step);
        assert_abs_diff_eq!(mixed[i], fd, epsilon = 2.0e-6);
    }

    // (f) None ⇒ bit-for-bit the unit-weight operator.
    let unit = [1.0_f64; 4];
    let pen_none = OrderedBetaBernoulliPenalty::new(k, 1.7, 0.8, true);
    let pen_unit =
        OrderedBetaBernoulliPenalty::new(k, 1.7, 0.8, true).with_row_weights(Some(&unit));
    assert_eq!(
        pen_none.value(t.view(), rho.view()),
        pen_unit.value(t.view(), rho.view())
    );
    let g_none = pen_none.grad_target(t.view(), rho.view());
    let g_unit = pen_unit.grad_target(t.view(), rho.view());
    for i in 0..t.len() {
        assert_eq!(g_none[i], g_unit[i]);
    }
}

#[test]
fn learnable_weight_effective_log_domain_is_exact_and_unsaturated() {
    assert!(
        resolve_learnable_weight(0.0, 0.0).is_err(),
        "a zero base would make a multiplicative rho coordinate dead"
    );
    let base = 1.7_f64;
    let (lower, upper) = learnable_weight_coordinate_domain(base)
        .unwrap()
        .expect("positive base has a coordinate domain");
    for (coordinate, expected_log_strength) in
        [(lower, LOG_STRENGTH_MIN), (upper, LOG_STRENGTH_MAX)]
    {
        let resolved = resolve_learnable_weight(base, coordinate).unwrap();
        assert!((resolved.ln() - expected_log_strength).abs() < 1.0e-12);
    }
    assert!(resolve_learnable_weight(base, lower - 1.0).is_err());
    assert!(resolve_learnable_weight(base, upper + 1.0).is_err());

    // `base > 1` is the subtle old plateau: raw rho=700 was accepted, while
    // ln(base)+rho was clipped. The legal upper face is shifted inward exactly.
    assert!(upper < LOG_STRENGTH_MAX);
    let penalty = OrderedBetaBernoulliPenalty::new(3, base, 0.8, true);
    penalty
        .validate_rho(array![upper].view())
        .expect("exact effective-strength endpoint is valid");
    assert!(penalty.validate_rho(array![upper + 1.0e-6].view()).is_err());
}

#[test]
fn registry_propagates_effective_strength_domain_and_freeze_refuses_invalid_rho() {
    let alpha = 1.7_f64;
    let penalty = OrderedBetaBernoulliPenalty::new(3, alpha, 0.8, true);
    let kind = AnalyticPenaltyKind::OrderedBetaBernoulli(Arc::new(penalty));
    let mut registry = AnalyticPenaltyRegistry::new();
    registry.push(kind.clone());

    let (expected_lower, expected_upper) = learnable_weight_coordinate_domain(alpha)
        .unwrap()
        .expect("positive alpha has a coordinate domain");
    let (lower, upper) = registry.rho_domain_bounds().unwrap();
    assert_eq!(lower.as_slice(), Some(&[expected_lower][..]));
    assert_eq!(upper.as_slice(), Some(&[expected_upper][..]));
    registry
        .validate_rho(array![expected_upper].view())
        .expect("closed upper face is legal");
    assert!(
        registry
            .validate_rho(array![expected_upper + 1.0e-6].view())
            .is_err()
    );

    let target = Array1::<f64>::zeros(3);
    assert!(FrozenAnalyticPenaltyOp::new(kind, target, array![expected_upper + 1.0e-6],).is_err());
}

#[test]
fn parametric_row_precision_domains_distinguish_log_strengths_from_raw_coordinates() {
    let target = PsiSlice::full(4, Some(2));
    let penalty = ParametricRowPrecisionPriorPenalty::new(
        target.clone(),
        array![[0.0_f64], [1.0]],
        array![0.0_f64, 2.0_f64.ln()],
        array![0.0_f64, -0.5],
        array![[0.0_f64], [0.5]],
        1.7,
        2,
        true,
    )
    .unwrap();

    let domains = penalty.rho_coordinate_domains().unwrap();
    assert_eq!(domains.len(), 7);
    assert_eq!(domains[0], (LOG_STRENGTH_MIN, LOG_STRENGTH_MAX));
    assert_eq!(
        domains[1],
        (
            LOG_STRENGTH_MIN - 2.0_f64.ln(),
            LOG_STRENGTH_MAX - 2.0_f64.ln(),
        )
    );
    for &(lower, upper) in &domains[2..6] {
        assert_eq!(lower, f64::NEG_INFINITY);
        assert_eq!(upper, f64::INFINITY);
    }
    assert_eq!(
        domains[6],
        learnable_weight_coordinate_domain(1.7)
            .unwrap()
            .expect("positive weight has a coordinate domain")
    );
    let mut registry = AnalyticPenaltyRegistry::new();
    registry.push(AnalyticPenaltyKind::ParametricRowPrecisionPrior(Arc::new(
        penalty.clone(),
    )));
    let (registry_lower, registry_upper) = registry.rho_domain_bounds().unwrap();
    for index in 2..6 {
        assert_eq!(registry_lower[index], f64::NEG_INFINITY);
        assert_eq!(registry_upper[index], f64::INFINITY);
    }

    // Raw beta and mu are finite additive coordinates, not log-strengths:
    // values beyond ±700 remain valid while a log-alpha beyond its effective
    // face is refused.
    let mut rho = Array1::<f64>::zeros(7);
    rho[2] = 701.0;
    rho[4] = -701.0;
    penalty
        .validate_rho(rho.view())
        .expect("finite raw-beta and mu coordinates are unbounded");
    registry
        .validate_rho(rho.view())
        .expect("registry preserves ordinary unbounded raw coordinates");
    rho[0] = LOG_STRENGTH_MAX + 1.0e-6;
    assert!(penalty.validate_rho(rho.view()).is_err());

    assert!(
        ParametricRowPrecisionPriorPenalty::new(
            target,
            array![[0.0_f64], [1.0]],
            array![LOG_STRENGTH_MAX + 1.0, 0.0],
            array![0.0_f64, -0.5],
            array![[0.0_f64], [0.5]],
            1.7,
            2,
            true,
        )
        .is_err()
    );
}

/// The conditional precision is `λ = α + β·r²` exactly. At a small `α` with an
/// underflowed `β` it is `α` itself, and the log-α coordinate of the ρ-gradient
/// at `t = 0` is `−½` per row, the derivative of `−½·ln λ` in `ln α`.
#[test]
fn parametric_row_precision_is_the_stated_precision_with_nothing_added() {
    let log_alpha = -40.0_f64;
    let weight = 1.7;
    let penalty = ParametricRowPrecisionPriorPenalty::new(
        PsiSlice::full(4, Some(2)),
        array![[0.0_f64], [1.0]],
        array![log_alpha, 0.0],
        array![-800.0_f64, -0.5],
        array![[0.0_f64], [0.5]],
        weight,
        2,
        false,
    )
    .unwrap();
    let rho = Array1::<f64>::zeros(6);
    let t = Array1::<f64>::zeros(4);

    let diag = penalty.diag_target(t.view(), rho.view());
    for n in 0..2 {
        assert_eq!(diag[n * 2], weight * log_alpha.exp());
    }

    let grad = penalty.grad_rho(t.view(), rho.view());
    assert!(
        (grad[0] + 1.0).abs() <= gam_linalg::roundoff::accumulation_growth(4),
        "log-alpha gradient at t = 0 must be -1 over two rows, got {}",
        grad[0]
    );
}

/// The parametric row-precision prior learns its whole conditional precision map:
/// `λ_k(u_n) = e^{log α_k} + softplus(raw β_k) ‖u_n − μ_k‖²`, plus a learnable
/// weight, for `2d + d·du + 1` outer coordinates in all. Before this test the only
/// check of `grad_rho` was the single log-α entry at `t = 0`, and nothing tied
/// `grad_target` or `hessian_diag` to the value. This pins value -> `grad_target`,
/// `grad_target` -> `hessian_diag` (the full FD Jacobian of the gradient, so the
/// Hessian is also checked to be exactly diagonal) and value -> every `grad_rho`
/// coordinate, all against central differences. It also checks the `as_dense`,
/// `log det(S + λI)` and frozen-operator channels against that diagonal.
///
/// Tolerances. The stencil error is `h²/6 |f'''| + ε_mach |f| / h` with `h = 1e-5`.
/// - The value is quadratic in `t`, so value -> grad has no truncation error.
///   Every summand is `O(1)`, which leaves a roundoff of about `1e-10`.
/// - The gradient is linear in `t`, so grad -> Hessian is also pure roundoff,
///   about `ε_mach · 3 / h ≈ 1e-10`.
/// - Along each ρ coordinate, `|f'''| ≲ 2` on this fixture (softplus, `e^ρ` and
///   the quadratic in μ are all `O(1)` here), so the truncation error is at most
///   about `3e-11`. The roundoff is `ε_mach · Σ|terms| / h ≈ 2e-10`.
/// A `1e-8` tolerance leaves at least 40× margin. The dense and log-det channels
/// re-evaluate the same `w·λ_k(u_n)`, so they agree to rounding: `1e-12` relative
/// on the entries, and one `ln` per entry for the eight-term log-det sum.
#[test]
fn parametric_row_precision_derivatives_match_central_differences() {
    let (n_eff, d) = (4usize, 2usize);
    let n = n_eff * d;
    let penalty = ParametricRowPrecisionPriorPenalty::new(
        PsiSlice::full(n, Some(d)),
        array![[-0.6_f64], [0.1], [0.45], [1.2]],
        array![-0.3_f64, 0.2],
        array![0.4_f64, -0.7],
        array![[0.2_f64], [-0.1]],
        0.8,
        n_eff,
        true,
    )
    .expect("parametric row-precision penalty");
    assert_eq!(penalty.rho_count(), 7);
    let rho = array![0.1_f64, -0.2, 0.3, 0.15, -0.25, 0.05, 0.2];
    penalty.validate_rho(rho.view()).expect("interior rho");
    let t = Array1::from_shape_fn(n, |i| 0.5 * (1.1 * i as f64 + 0.3).sin() + 0.1);
    let h = 1e-5;
    let tol = 1e-8;

    let g = penalty.grad_target(t.view(), rho.view());
    let diag = penalty
        .hessian_diag(t.view(), rho.view())
        .expect("row-precision Hessian is diagonal");
    for i in 0..n {
        let mut tp = t.clone();
        let mut tm = t.clone();
        tp[i] += h;
        tm[i] -= h;
        let fd = (penalty.value(tp.view(), rho.view()) - penalty.value(tm.view(), rho.view()))
            / (2.0 * h);
        assert_abs_diff_eq!(g[i], fd, epsilon = tol);
        let gp = penalty.grad_target(tp.view(), rho.view());
        let gm = penalty.grad_target(tm.view(), rho.view());
        for j in 0..n {
            let expected = if i == j { diag[i] } else { 0.0 };
            assert_abs_diff_eq!((gp[j] - gm[j]) / (2.0 * h), expected, epsilon = tol);
        }
    }

    let gr = penalty.grad_rho(t.view(), rho.view());
    assert_eq!(gr.len(), 7);
    for c in 0..7 {
        let mut rp = rho.clone();
        let mut rm = rho.clone();
        rp[c] += h;
        rm[c] -= h;
        let fd =
            (penalty.value(t.view(), rp.view()) - penalty.value(t.view(), rm.view())) / (2.0 * h);
        assert_abs_diff_eq!(gr[c], fd, epsilon = tol);
    }

    let scale = diag.iter().fold(0.0_f64, |acc, &x| acc.max(x.abs()));
    assert!(
        diag.iter().all(|&x| x > 0.0),
        "the conditional precision is positive"
    );
    let dense = penalty.as_dense(t.view(), rho.view());
    for i in 0..n {
        for j in 0..n {
            let expected = if i == j { diag[i] } else { 0.0 };
            assert_abs_diff_eq!(dense[[i, j]], expected, epsilon = 1e-12 * scale);
        }
    }
    let lambda = 0.3;
    let expected_log_det: f64 = diag.iter().map(|&x| (x + lambda).ln()).sum();
    let log_det = penalty
        .log_det_plus_lambda_i(rho.view(), lambda)
        .expect("row-precision log det");
    assert_abs_diff_eq!(log_det, expected_log_det, epsilon = 1e-12 * n as f64);

    let op = FrozenAnalyticPenaltyOp::new(
        AnalyticPenaltyKind::ParametricRowPrecisionPrior(Arc::new(penalty)),
        t.clone(),
        rho.clone(),
    )
    .expect("frozen row-precision operator");
    let frozen_diag = op.diag();
    for i in 0..n {
        assert_abs_diff_eq!(frozen_diag[i], diag[i], epsilon = 1e-12 * scale);
    }
    let frozen_log_det = op
        .log_det_plus_lambda_i(lambda)
        .expect("frozen row-precision log det");
    assert_abs_diff_eq!(frozen_log_det, expected_log_det, epsilon = 1e-12 * n as f64);
}

#[test]
fn sparsity_learnable_smoothing_has_one_structural_coordinate() {
    assert!(
        SparsityPenalty::hoyer(PenaltyTier::Psi)
            .with_learnable_smoothing()
            .is_err(),
        "Hoyer has no smoothing scale and must not acquire a dead rho axis"
    );

    let penalty = SparsityPenalty::smoothed_l1(PenaltyTier::Psi, 1.0e-3)
        .unwrap()
        .with_learnable_smoothing()
        .unwrap();
    assert_eq!(penalty.rho_count(), 2);
    let domains = penalty.rho_coordinate_domains().unwrap();
    assert_eq!(domains.len(), 2);
    assert_eq!(domains[1], (LOG_STRENGTH_MIN, LOG_STRENGTH_MAX));
    penalty
        .validate_rho(array![0.0, LOG_STRENGTH_MAX].view())
        .expect("closed smoothing face is valid");
    assert!(
        penalty
            .validate_rho(array![0.0, LOG_STRENGTH_MAX + 1.0e-6].view())
            .is_err()
    );
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

/// Regression for #2294's invariant boundary. Production evaluates a
/// registry isometry penalty only through an atom-owned clone that is retargeted
/// and refreshed first. If a caller violates that ownership contract, a live
/// `d=2` cache must never be reinterpreted at `d=1` or silently converted into
/// a zero penalty. The dimensional mismatch remains a hard shape invariant.
#[test]
#[should_panic(expected = "stale cross-atom Jacobian cache")]
fn isometry_pullback_metric_rejects_stale_cross_dimension_cache_2294() {
    let p = 4usize;
    let n_obs = 3usize;
    let target = PsiSlice::full(n_obs, Some(1));
    let pen = IsometryPenalty::new_euclidean(target, p);
    pen.refresh_caches(Some(Arc::new(Array2::<f64>::zeros((n_obs, p * 2)))), None);
    assert!(pen.pullback_metric(1).is_some());
}

/// #2627: an isometry penalty is a function of the decoder jets its owner installs.
/// Its evaluation precondition names each missing or misshapen jet by the order
/// that reads it, and a registry refuses with the penalty's own text. The routes
/// with no decoder jets (the latent-coordinate fits) refuse through the registry.
#[test]
fn isometry_precondition_names_each_missing_decoder_jet_2627() {
    let (n_obs, p, d, j, h) = isometry_gn_fixture();
    let n = n_obs * d;
    let pen = IsometryPenalty::new_euclidean(PsiSlice::full(n, Some(d)), p);
    let missing_j = pen
        .evaluation_state_precondition(IsometryEvaluationOrder::Value, n)
        .expect_err("a penalty with no installed jets has no value");
    assert!(missing_j.contains("decoder Jacobian J"), "{missing_j}");

    pen.refresh_caches(Some(j.clone()), None);
    assert_eq!(
        pen.evaluation_state_precondition(IsometryEvaluationOrder::Value, n),
        Ok(())
    );
    let missing_h = pen
        .evaluation_state_precondition(IsometryEvaluationOrder::Gradient, n)
        .expect_err("J alone has no target derivative");
    assert!(missing_h.contains("H = ∂J/∂t"), "{missing_h}");

    pen.refresh_caches(Some(j), Some(h));
    assert_eq!(
        pen.evaluation_state_precondition(IsometryEvaluationOrder::Gradient, n),
        Ok(())
    );
    let missing_k = pen
        .evaluation_state_precondition(IsometryEvaluationOrder::Hessian, n)
        .expect_err("J and H have no exact Hessian");
    assert!(missing_k.contains("K = ∂H/∂t"), "{missing_k}");
    pen.set_third_decoder_derivative(Some(Arc::new(ndarray::Array3::<f64>::zeros((
        n_obs,
        p,
        d * d * d,
    )))));
    assert_eq!(
        pen.evaluation_state_precondition(IsometryEvaluationOrder::Hessian, n),
        Ok(())
    );

    let misshapen = IsometryPenalty::new_euclidean(PsiSlice::full(n, Some(d)), p);
    misshapen.refresh_caches(Some(Arc::new(Array2::<f64>::zeros((n_obs - 1, p * d)))), None);
    let wrong_rows = misshapen
        .evaluation_state_precondition(IsometryEvaluationOrder::Value, n)
        .expect_err("a Jacobian for another row count does not fit this target");
    assert!(wrong_rows.contains("decoder Jacobian J has shape"), "{wrong_rows}");

    let mut registry = AnalyticPenaltyRegistry::new();
    registry.push(AnalyticPenaltyKind::Isometry(Arc::new(
        IsometryPenalty::new_euclidean(PsiSlice::full(n, Some(d)), p),
    )));
    assert_eq!(
        registry.isometry_evaluation_precondition(IsometryEvaluationOrder::Hessian, n),
        Err(missing_j)
    );
}

/// #2627: an evaluation that skips the precondition stops by name. It used to
/// return a zero gradient with a log warning, which silently dropped the gauge
/// from every objective that carried the penalty.
#[test]
#[should_panic(expected = "IsometryPenalty::grad_target: IsometryPenalty target derivatives read")]
fn isometry_grad_target_without_the_jacobian_motion_stops_instead_of_returning_zero_2627() {
    let (n_obs, p, d, j, ..) = isometry_gn_fixture();
    let n = n_obs * d;
    let pen = IsometryPenalty::new_euclidean(PsiSlice::full(n, Some(d)), p);
    pen.refresh_caches(Some(j), None);
    let grad = pen.grad_target(Array1::<f64>::zeros(n).view(), array![0.0_f64].view());
    assert!(
        grad.iter().any(|x| *x != 0.0),
        "grad_target without H returned {grad:?} instead of stopping"
    );
}

/// Build the canonical smooth-threshold sweep fixture: a logit grid that straddles
/// each per-axis scaled threshold so the gate `g = σ((z − τ)/ε)` sweeps both
/// sides of its inflection `g = ½`, where the true Hessian
/// `wτ·g(1−g)(1−2g)/ε²` changes sign.
fn smooth_threshold_sweep_fixture() -> (
    SmoothThresholdPenalty,
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
    let pen = SmoothThresholdPenalty::new(slice, thresholds, weight, eps)
        .expect("valid smooth-threshold penalty");
    (pen, target_values, rho, scaled_thresholds, eps, weight)
}

#[test]
fn smooth_threshold_hessian_diag_is_exact_true_second_derivative() {
    let (pen, target_values, rho, scaled_thresholds, eps, weight) =
        smooth_threshold_sweep_fixture();
    let latent_dim = scaled_thresholds.len();
    // `hessian_diag` must be the EXACT diagonal second derivative of the
    // smoothed jump penalty `P(z) = wτ·σ((z − τ)/ε)`:
    //   P''(z) = wτ·g(1 − g)(1 − 2g)/ε²,   g = σ((z − τ)/ε).
    // This is the true (indefinite) Hessian, not the PSD majorizer.
    let diag = pen
        .hessian_diag(target_values.view(), rho.view())
        .expect("smooth threshold exposes an analytic diagonal Hessian");

    let mut saw_negative = false;
    for (idx, &entry) in diag.iter().enumerate() {
        let axis = idx % latent_dim;
        let gate = pen.sigmoid_gate((target_values[idx] - scaled_thresholds[axis]) / eps);
        let expected = weight * scaled_thresholds[axis] * gate * (1.0 - gate) * (1.0 - 2.0 * gate)
            / (eps * eps);
        assert!(
            entry.is_finite(),
            "smooth-threshold hessian_diag must be finite at index {idx}; entry={entry}"
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
        "true smooth-threshold hessian_diag must go negative once the gate passes g = ½"
    );
}

#[test]
fn smooth_threshold_hvp_diagonal_matches_hessian_diag() {
    let (pen, target_values, rho, _scaled_thresholds, _eps, _weight) =
        smooth_threshold_sweep_fixture();
    // `hvp` and `hessian_diag` are the SAME true operator; probing `hvp`
    // with unit vectors must reproduce `hessian_diag` exactly.
    let diag = pen
        .hessian_diag(target_values.view(), rho.view())
        .expect("smooth threshold exposes an analytic diagonal Hessian");
    for i in 0..target_values.len() {
        let mut e_i = Array1::<f64>::zeros(target_values.len());
        e_i[i] = 1.0;
        let hv_i = pen.hvp(target_values.view(), rho.view(), e_i.view());
        assert_abs_diff_eq!(diag[i], hv_i[i], epsilon = 1e-12);
    }
}

#[test]
fn smooth_threshold_psd_majorizer_diag_is_psd_over_logit_sweep() {
    let (pen, target_values, rho, scaled_thresholds, eps, weight) =
        smooth_threshold_sweep_fixture();
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
        .expect("smooth threshold exposes a PSD diagonal majorizer");
    let exact = pen
        .hessian_diag(target_values.view(), rho.view())
        .expect("smooth threshold exposes a closed-form diagonal Hessian");

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
            "smooth-threshold psd_majorizer_diag must be finite and PSD at index {idx}; entry={entry}"
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

/// Checks `grad_target` against central differences of `value`, the Hessian
/// `diag` against central differences of `grad_target`, and `grad_rho`
/// against central differences of `value` in each ρ coordinate. The full FD
/// Jacobian of the gradient is compared with `diag`, so the Hessian is also
/// checked to be exactly diagonal.
fn assert_diagonal_penalty_matches_central_differences(
    pen: &dyn AnalyticPenalty,
    t: &Array1<f64>,
    rho: &Array1<f64>,
    diag: &Array1<f64>,
    tol: f64,
) {
    let h = 1e-5;
    let n = t.len();
    let worst = value_grad_fd_max_abs_error(pen, t.view(), rho.view(), h);
    assert!(
        worst <= tol,
        "{} value -> grad FD max abs error = {worst:.3e}",
        pen.name()
    );
    for i in 0..n {
        let mut tp = t.clone();
        let mut tm = t.clone();
        tp[i] += h;
        tm[i] -= h;
        let gp = pen.grad_target(tp.view(), rho.view());
        let gm = pen.grad_target(tm.view(), rho.view());
        for j in 0..n {
            let expected = if i == j { diag[i] } else { 0.0 };
            assert_abs_diff_eq!((gp[j] - gm[j]) / (2.0 * h), expected, epsilon = tol);
        }
    }
    let gr = pen.grad_rho(t.view(), rho.view());
    assert_eq!(gr.len(), rho.len());
    for c in 0..rho.len() {
        let mut rp = rho.clone();
        let mut rm = rho.clone();
        rp[c] += h;
        rm[c] -= h;
        let fd = (pen.value(t.view(), rp.view()) - pen.value(t.view(), rm.view())) / (2.0 * h);
        assert_abs_diff_eq!(gr[c], fd, epsilon = tol);
    }
}

/// Before this test the sparsity family's derivatives were checked only against
/// re-typed closed forms, or not at all. Nothing tied the learnable-smoothing
/// `grad_rho` coordinate (`log ε`, `log δ`), the dense Hoyer gradient and HVP,
/// the smooth-threshold gradient, Hessian and per-axis `grad_rho`, or the TopK
/// activation gradient and Hessian to the value. This pins all of them against
/// central differences with `h = 1e-5`.
///
/// Tolerances. The stencil error is `h²/6 |f'''| + ε_mach |f| / h`, and the
/// grad -> Hessian comparisons carry the largest truncation term.
/// - Smoothed L¹ with `ε = 0.3`, `λ = e^{0.2}`: the third derivative of
///   `λ x / sqrt(x² + ε²)` peaks at `3λ/ε³ ≈ 136` (at `x = 0`), which gives
///   about `2.3e-9`.
/// - Log with `δ = 0.5`, `λ = e^{−0.1}`: the third derivative of
///   `2λx/(δ² + x²)` peaks at `12λ/δ⁴ ≈ 174`, which gives about `2.9e-9`.
/// - Smooth threshold with `ε = 0.2`: the gradient is `wτ σ'(z)/ε` with
///   `z = (x − τ)/ε`. Its third x-derivative is `wτ σ''''(z)/ε⁴`, and
///   `|σ''''| ≤ 1/8`, `wτ ≤ 1.3 · 0.8 e^{−0.2}`, so it is at most about 66,
///   which gives about `1.1e-9`.
/// - Hoyer and TopK stay at least `0.05` from every kink and every magnitude
///   tie, so the stencil never crosses one. Hoyer's terms are `O(1)`
///   (about `2e-11`), and TopK's value is quadratic, which leaves pure
///   roundoff.
/// Every value/`grad_rho` term is `O(1)`, so its roundoff is at most about `1e-10`.
/// A `1e-7` tolerance leaves at least 30× margin.
#[test]
fn sparsity_threshold_and_topk_derivatives_match_central_differences() {
    let x = array![0.7_f64, -0.35, 0.05, -1.2, 0.4, -0.08];
    let tol = 1e-7;

    let smoothed_l1 = SparsityPenalty::smoothed_l1(PenaltyTier::Psi, 0.3)
        .expect("smoothed L1")
        .with_learnable_smoothing()
        .expect("learnable eps");
    let rho = array![0.2_f64, 0.3_f64.ln()];
    smoothed_l1.validate_rho(rho.view()).expect("interior rho");
    let diag = smoothed_l1
        .hessian_diag(x.view(), rho.view())
        .expect("smoothed L1 Hessian is diagonal");
    assert_diagonal_penalty_matches_central_differences(&smoothed_l1, &x, &rho, &diag, tol);

    let log = SparsityPenalty::log(PenaltyTier::Psi, 0.5)
        .expect("log sparsity")
        .with_learnable_smoothing()
        .expect("learnable delta");
    let rho = array![-0.1_f64, 0.5_f64.ln()];
    log.validate_rho(rho.view()).expect("interior rho");
    let diag = log
        .hessian_diag(x.view(), rho.view())
        .expect("log sparsity Hessian is diagonal");
    assert!(
        diag.iter().any(|&d| d < 0.0),
        "the fixture must reach the nonconvex region |x| > delta"
    );
    assert_diagonal_penalty_matches_central_differences(&log, &x, &rho, &diag, tol);

    let h = 1e-5;
    let hoyer = SparsityPenalty::hoyer(PenaltyTier::Psi);
    let rho = array![0.3_f64];
    let worst = value_grad_fd_max_abs_error(&hoyer, x.view(), rho.view(), h);
    assert!(
        worst <= tol,
        "Hoyer value -> grad FD max abs error = {worst:.3e}"
    );
    let v = Array1::from_shape_fn(x.len(), |i| 0.6 * (0.9 * i as f64 + 0.4).cos());
    let hv = hoyer.hvp(x.view(), rho.view(), v.view());
    let gp = hoyer.grad_target((&x + &(h * &v)).view(), rho.view());
    let gm = hoyer.grad_target((&x - &(h * &v)).view(), rho.view());
    for i in 0..x.len() {
        assert_abs_diff_eq!(hv[i], (gp[i] - gm[i]) / (2.0 * h), epsilon = tol);
    }
    let gr = hoyer.grad_rho(x.view(), rho.view());
    assert_eq!(gr.len(), 1);
    let fd_rho = (hoyer.value(x.view(), array![0.3 + h].view())
        - hoyer.value(x.view(), array![0.3 - h].view()))
        / (2.0 * h);
    assert_abs_diff_eq!(gr[0], fd_rho, epsilon = tol);

    let t = array![0.1_f64, 0.5, 0.3, 0.9, 0.45, 1.1];
    let threshold = SmoothThresholdPenalty::new(
        PsiSlice::full(t.len(), Some(2)),
        array![0.25_f64, 0.8],
        1.3,
        0.2,
    )
    .expect("smooth threshold");
    let rho = array![0.1_f64, -0.2];
    threshold.validate_rho(rho.view()).expect("interior rho");
    let diag = threshold
        .hessian_diag(t.view(), rho.view())
        .expect("smooth threshold Hessian is diagonal");
    assert!(
        diag.iter().any(|&d| d < 0.0) && diag.iter().any(|&d| d > 0.0),
        "the fixture must straddle the gate inflection"
    );
    assert_diagonal_penalty_matches_central_differences(&threshold, &t, &rho, &diag, tol);

    let t = array![0.9_f64, -0.2, 0.5, 0.1, -0.7, 0.4, 0.3, 0.6, -1.0];
    let topk = TopKActivationPenalty::new(PsiSlice::full(t.len(), Some(3)), 2, 0.9)
        .expect("topk activation");
    let rho = Array1::<f64>::zeros(0);
    let diag = topk
        .hessian_diag(t.view(), rho.view())
        .expect("topk Hessian is diagonal");
    assert_eq!(
        diag.iter().filter(|&&d| d > 0.0).count(),
        6,
        "two active axes per row"
    );
    assert_diagonal_penalty_matches_central_differences(&topk, &t, &rho, &diag, tol);
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

// ----- TotalVariationPenalty / ShapeMonotonicityPenalty tests -----

/// Row-major `(n_eff = 5, d = 2)` target whose edge differences run from
/// about 0.08 to about 0.92. That span covers both the curved core and the
/// near-linear tails of the smoothed-L¹ and softplus kernels.
fn tv_monotonicity_test_target() -> (usize, Array1<f64>, Array1<f64>) {
    let (n_eff, d) = (5usize, 2usize);
    let n = n_eff * d;
    let t = Array1::from_shape_fn(n, |i| {
        let x = i as f64;
        0.37 * (1.3 * x).sin() + 0.11 * x
    });
    let v = Array1::from_shape_fn(n, |i| {
        let x = i as f64;
        0.6 * (0.9 * x + 0.4).cos()
    });
    (n_eff, t, v)
}

/// Total variation had no Rust test of its analytic derivatives or of its
/// closed-form curvature channels. For both difference operators this pins:
/// value -> `grad_target`, `grad_target` -> `hvp` (directional),
/// value -> `grad_rho` (learnable log-weight), the `as_dense` columns against
/// `hvp` of the unit vectors, and `diag_target` against the dense diagonal.
/// It also pins the tridiagonal pivot recursion
/// `log_det_plus_lambda_i_forward_1d` against a dense eigendecomposition, and
/// the frozen operator's `diag` and `log det(S + λI)` against the dense matrix.
///
/// Tolerances. The FD errors come from the central stencil,
/// `h²/6 |f'''| + ε_mach |f| / h` with `h = 1e-5`. Here `w = 0.7 e^{0.2} ≈ 0.86`,
/// `ε = 0.5`, and a node has at most 3 edges. The kernel has
/// `|φ'''| ≤ 0.86/ε² ≈ 3.4` and `|φ''''| ≤ 3/ε³ = 24`.
/// - value -> grad: about `1.5e-10 + 2.2e-10` (with `|f| ≲ 10`).
/// - grad -> hvp: along `v` with `|Δv| ≤ 1.2`, about `3e-9`.
/// - value -> grad_rho: the value is `w₀ e^ρ A`, so its third ρ-derivative is
///   the value itself, and the error is about `2e-10`.
/// A `1e-7` tolerance leaves at least 30× margin. The dense and diagonal
/// channels evaluate the same per-edge curvature `w ε² / r³` as `hvp`, so they
/// agree to rounding (`1e-12` relative). The log-determinants run on a 10 × 10
/// SPD matrix with eigenvalues `≥ λ = 0.3` and norm `≲ 20`. Each `log`
/// therefore carries about `ε_mach · 20 / 0.3`, and `1e-10 · max(|log det|, 1)`
/// leaves several orders of margin.
#[test]
fn total_variation_derivatives_and_curvature_channels_match_central_differences() {
    let (n_eff, t, v) = tv_monotonicity_test_target();
    let n = t.len();
    let h = 1e-5;
    let tol = 1e-7;
    let lambda = 0.3;
    let rho = array![0.2_f64];
    let graph = vec![(0, 1), (1, 3), (3, 2), (4, 0), (2, 4), (1, 4)];
    for op_kind in [
        DifferenceOpKind::ForwardDiff1D,
        DifferenceOpKind::GraphEdges(graph),
    ] {
        let pen = TotalVariationPenalty::new(0.7, n_eff, op_kind.clone(), 0.5, true)
            .expect("total variation penalty");
        let label = format!("{op_kind:?}");

        let g = pen.grad_target(t.view(), rho.view());
        for i in 0..n {
            let mut tp = t.clone();
            let mut tm = t.clone();
            tp[i] += h;
            tm[i] -= h;
            let fd =
                (pen.value(tp.view(), rho.view()) - pen.value(tm.view(), rho.view())) / (2.0 * h);
            assert_abs_diff_eq!(g[i], fd, epsilon = tol);
        }

        let hv = pen.hvp(t.view(), rho.view(), v.view());
        let tp = &t + &(h * &v);
        let tm = &t - &(h * &v);
        let gp = pen.grad_target(tp.view(), rho.view());
        let gm = pen.grad_target(tm.view(), rho.view());
        for i in 0..n {
            assert_abs_diff_eq!(hv[i], (gp[i] - gm[i]) / (2.0 * h), epsilon = tol);
        }

        let gr = pen.grad_rho(t.view(), rho.view());
        assert_eq!(gr.len(), 1, "{label}: one learnable log-weight");
        let fd_rho = (pen.value(t.view(), array![0.2 + h].view())
            - pen.value(t.view(), array![0.2 - h].view()))
            / (2.0 * h);
        assert_abs_diff_eq!(gr[0], fd_rho, epsilon = tol);

        let dense = pen.as_dense(t.view(), rho.view());
        let scale = dense.iter().fold(0.0_f64, |acc, &x| acc.max(x.abs()));
        assert!(
            scale > 0.0,
            "{label}: curvature must be nonzero on this target"
        );
        let diag = pen.diag_target(t.view(), rho.view());
        for col in 0..n {
            let mut e = Array1::<f64>::zeros(n);
            e[col] = 1.0;
            let column = pen.hvp(t.view(), rho.view(), e.view());
            for row in 0..n {
                assert_abs_diff_eq!(dense[[row, col]], column[row], epsilon = 1e-12 * scale);
            }
            assert_abs_diff_eq!(diag[col], dense[[col, col]], epsilon = 1e-12 * scale);
        }

        let expected = <Array2<f64> as PenaltyOp>::log_det_plus_lambda_i(&dense, lambda)
            .expect("dense TV log det");
        let log_det_tol = 1e-10 * expected.abs().max(1.0);
        let forward = pen.log_det_plus_lambda_i_forward_1d(t.view(), rho.view(), lambda);
        match op_kind {
            DifferenceOpKind::ForwardDiff1D => {
                let forward = forward.expect("tridiagonal TV log det");
                assert_abs_diff_eq!(forward, expected, epsilon = log_det_tol);
            }
            DifferenceOpKind::GraphEdges(_) => {
                assert!(
                    forward.is_err(),
                    "{label}: the pivot recursion is path-only"
                );
            }
        }

        let op = FrozenAnalyticPenaltyOp::new(
            AnalyticPenaltyKind::TotalVariation(Arc::new(pen)),
            t.clone(),
            rho.clone(),
        )
        .expect("frozen TV operator");
        let frozen_diag = op.diag();
        for i in 0..n {
            assert_abs_diff_eq!(frozen_diag[i], dense[[i, i]], epsilon = 1e-12 * scale);
        }
        let frozen_log_det = op.log_det_plus_lambda_i(lambda).expect("frozen TV log det");
        assert_abs_diff_eq!(frozen_log_det, expected, epsilon = log_det_tol);
    }
}

/// Soft monotonicity had no Rust test of its analytic derivatives. For both
/// directions this pins value -> `grad_target`, `grad_target` -> `hvp`
/// (directional) and value -> `grad_rho`. It also checks that the Hessian
/// assembled from `hvp` of the unit vectors is symmetric and PSD, as the
/// convexity of softplus requires.
///
/// Tolerances. The per-edge value is `w ε softplus(z)` with `z = ∓Δ/ε`,
/// `w = 0.7 e^{0.2} ≈ 0.86` and `ε = 0.3`. Its slope derivatives are bounded
/// by `|softplus'''| ≤ 1/(6√3)` and `|softplus''''| ≤ 1/8`. So
/// `|f'''| ≤ 2 w · 0.096 / ε² ≈ 1.8` and `|f''''| ≤ 2 w · 0.125 / ε³ ≈ 8`
/// (at most 2 edges per node). With `h = 1e-5`, `|f| ≲ 5` and `|Δv| ≤ 1.2`:
/// - value -> grad: about `3e-11 + 1.1e-10`.
/// - grad -> hvp: about `2.4e-10`.
/// - value -> grad_rho: about `1e-10`.
/// A `1e-7` tolerance leaves at least 400× margin. Symmetry is exact up to the
/// shared edge factor (`1e-12` relative). The smallest eigenvalue of a PSD
/// matrix computed by `eigh` is off by about `ε_mach ‖H‖`, which is covered
/// by `-1e-12 · scale`.
#[test]
fn monotonicity_derivatives_match_central_differences_and_hessian_is_psd() {
    let (n_eff, t, v) = tv_monotonicity_test_target();
    let n = t.len();
    let h = 1e-5;
    let tol = 1e-7;
    let rho = array![0.2_f64];
    for direction in [1.0_f64, -1.0] {
        let pen = ShapeMonotonicityPenalty::new(0.7, n_eff, direction, 0.3, true)
            .expect("monotonicity penalty");

        let g = pen.grad_target(t.view(), rho.view());
        for i in 0..n {
            let mut tp = t.clone();
            let mut tm = t.clone();
            tp[i] += h;
            tm[i] -= h;
            let fd =
                (pen.value(tp.view(), rho.view()) - pen.value(tm.view(), rho.view())) / (2.0 * h);
            assert_abs_diff_eq!(g[i], fd, epsilon = tol);
        }

        let hv = pen.hvp(t.view(), rho.view(), v.view());
        let tp = &t + &(h * &v);
        let tm = &t - &(h * &v);
        let gp = pen.grad_target(tp.view(), rho.view());
        let gm = pen.grad_target(tm.view(), rho.view());
        for i in 0..n {
            assert_abs_diff_eq!(hv[i], (gp[i] - gm[i]) / (2.0 * h), epsilon = tol);
        }

        let gr = pen.grad_rho(t.view(), rho.view());
        assert_eq!(
            gr.len(),
            1,
            "direction {direction}: one learnable log-weight"
        );
        let fd_rho = (pen.value(t.view(), array![0.2 + h].view())
            - pen.value(t.view(), array![0.2 - h].view()))
            / (2.0 * h);
        assert_abs_diff_eq!(gr[0], fd_rho, epsilon = tol);

        let mut hessian = Array2::<f64>::zeros((n, n));
        for col in 0..n {
            let mut e = Array1::<f64>::zeros(n);
            e[col] = 1.0;
            let column = pen.hvp(t.view(), rho.view(), e.view());
            for row in 0..n {
                hessian[[row, col]] = column[row];
            }
        }
        let scale = hessian.iter().fold(0.0_f64, |acc, &x| acc.max(x.abs()));
        assert!(
            scale > 0.0,
            "direction {direction}: curvature must be nonzero"
        );
        for row in 0..n {
            for col in 0..n {
                assert_abs_diff_eq!(
                    hessian[[row, col]],
                    hessian[[col, row]],
                    epsilon = 1e-12 * scale
                );
            }
        }
        let (evals, _) = hessian
            .eigh(Side::Lower)
            .expect("monotonicity Hessian eigh");
        let min_eval = evals.iter().fold(f64::INFINITY, |acc, &x| acc.min(x));
        assert!(
            min_eval >= -1e-12 * scale,
            "direction {direction}: softplus curvature must be PSD; min eigenvalue {min_eval:.3e}"
        );
    }
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
fn block_orthogonality_value_is_half_weight_per_unordered_pair() {
    let t = block_ortho_test_target();
    let target = PsiSlice::full(t.len(), Some(4));
    let pen =
        BlockOrthogonalityPenalty::new(target, vec![vec![0_usize, 1], vec![2, 3]], 2.5, 4, false)
            .expect("valid block orthogonality penalty");
    let rho = array![0.0_f64];
    let v = pen.value(t.view(), rho.view());
    // Public normalization: 0.5 · w · Σ_{g<h} ||T_g^T T_h||²_F.
    // There is one unordered pair here, with squared Frobenius norm 3.
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
    let fd = gam_linalg_test_support::fd_checker::numerical_gradient_central_diff(
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
fn block_orthogonality_hessian_is_dense_so_it_exposes_no_diagonal_3863() {
    // The exact Hessian couples every row, so a `Some` diagonal would be read
    // as the curvature (default `psd_majorizer_hvp`, SAE `htt`, spatial
    // hyper-direction Hessian) and understate it. The exact curvature is the
    // HVP, whose unit-probe diagonal matches finite differences of the gradient.
    let t = block_ortho_test_target();
    let n = t.len();
    let target = PsiSlice::full(n, Some(4));
    let pen =
        BlockOrthogonalityPenalty::new(target, vec![vec![0_usize, 1], vec![2, 3]], 0.9, 4, false)
            .expect("valid block orthogonality penalty");
    let rho = array![0.0_f64];
    assert!(pen.hessian_diag(t.view(), rho.view()).is_none());
    assert!(pen.psd_majorizer_diag(t.view(), rho.view()).is_none());
    let eps = 1e-5;
    let mut e = Array1::<f64>::zeros(n);
    for i in 0..n {
        e[i] = 1.0;
        let h_ii = pen.hvp(t.view(), rho.view(), e.view())[i];
        e[i] = 0.0;
        let mut tp = t.clone();
        let mut tm = t.clone();
        tp[i] += eps;
        tm[i] -= eps;
        let gp = pen.grad_target(tp.view(), rho.view())[i];
        let gm = pen.grad_target(tm.view(), rho.view())[i];
        assert_abs_diff_eq!(h_ii, (gp - gm) / (2.0 * eps), epsilon = 1e-5);
    }
}

/// Dense matrix of `op(e_j)` over the standard basis.
fn dense_from_probes(n: usize, op: impl Fn(ArrayView1<'_, f64>) -> Array1<f64>) -> Array2<f64> {
    let mut dense = Array2::<f64>::zeros((n, n));
    let mut e = Array1::<f64>::zeros(n);
    for j in 0..n {
        e[j] = 1.0;
        dense.column_mut(j).assign(&op(e.view()));
        e[j] = 0.0;
    }
    dense
}

fn symmetric_eigenvalues(m: &Array2<f64>) -> Array1<f64> {
    let (evals, _) = m.eigh(Side::Lower).expect("symmetric eigendecomposition");
    evals
}

fn min_eigenvalue(m: &Array2<f64>) -> f64 {
    symmetric_eigenvalues(m)
        .iter()
        .copied()
        .fold(f64::INFINITY, f64::min)
}

fn spectral_radius(m: &Array2<f64>) -> f64 {
    symmetric_eigenvalues(m)
        .iter()
        .fold(0.0_f64, |acc, value| acc.max(value.abs()))
}

/// Deterministic, non-degenerate `n × d` latent block, row-major.
fn pseudo_random_block(n_obs: usize, d: usize, amplitude: f64) -> Array1<f64> {
    Array1::from_shape_fn(n_obs * d, |i| {
        amplitude * ((1.7 * i as f64 + 0.3).sin() + 0.5 * (0.9 * (i * i) as f64 + 1.1).cos())
    })
}

/// #3863 — BlockOrthogonality's PSD majorizer used to be the exact Hessian
/// diagonal, which is not `⪰ H` (the issue's FD table: λ_min(diag(H) − H) =
/// −14.46 and −11.78). The row-block majorizer `I_n ⊗ M` must dominate the
/// exact, indefinite Hessian, be PSD, and be row-block diagonal with the block
/// `psd_majorizer_row_block` reports (the shape the SAE `htt` probe assumes).
#[test]
fn block_orthogonality_psd_majorizer_dominates_exact_hessian_3863() {
    let cases: Vec<(Array1<f64>, usize, usize, Vec<Vec<usize>>, f64)> = vec![
        (block_ortho_test_target(), 4, 4, vec![vec![0, 1], vec![2, 3]], 0.9),
        (pseudo_random_block(4, 2, 1.0), 4, 2, vec![vec![0], vec![1]], 1.0),
        (pseudo_random_block(6, 3, 1.0), 6, 3, vec![vec![0], vec![1, 2]], 1.0),
        (pseudo_random_block(7, 5, 0.6), 7, 5, vec![vec![0, 3], vec![1], vec![2, 4]], 2.3),
    ];
    let rho = array![0.0_f64];
    for (t, n_obs, d, groups, weight) in cases {
        let dim = n_obs * d;
        let pen = BlockOrthogonalityPenalty::new(
            PsiSlice::full(dim, Some(d)),
            groups.clone(),
            weight,
            n_obs,
            false,
        )
        .expect("valid block orthogonality penalty");
        let exact = pen.as_dense(t.view(), rho.view());
        let majorizer =
            dense_from_probes(dim, |v| pen.psd_majorizer_hvp(t.view(), rho.view(), v));
        let block = pen
            .psd_majorizer_row_block(t.view(), rho.view())
            .expect("row-block majorizer");
        let scale = spectral_radius(&exact).max(spectral_radius(&majorizer));
        let tol = 1e-12 * scale;
        for i in 0..dim {
            for j in 0..dim {
                let expected = if i / d == j / d {
                    block[[i % d, j % d]]
                } else {
                    0.0
                };
                assert_eq!(majorizer[[i, j]], expected, "{groups:?}: B is not I_n ⊗ M");
            }
        }
        assert!(
            min_eigenvalue(&majorizer) >= -tol,
            "{groups:?}: majorizer is not PSD"
        );
        let gap = &majorizer - &exact;
        let gap_min = min_eigenvalue(&gap);
        assert!(
            gap_min >= -tol,
            "{groups:?}: λ_min(B − H) = {gap_min:.3e} < 0; B does not majorize H"
        );
        if n_obs != 4 || d != 4 {
            // The generic blocks reproduce the issue's defect: H is indefinite
            // and its own diagonal does not majorize it.
            assert!(min_eigenvalue(&exact) < -tol, "{groups:?}: H should be indefinite");
            let diag_gap = Array2::from_diag(&exact.diag()) - &exact;
            assert!(
                min_eigenvalue(&diag_gap) < -tol,
                "{groups:?}: diag(H) should fail to majorize H"
            );
        }
    }
}

/// #3863 — OrthogonalityPenalty had no majorizer, so `psd_majorizer_hvp` was
/// the exact, indefinite HVP. Replacing `G = TᵀT − I` by its PSD part must give
/// a PSD operator that dominates `H`, and where `G ⪰ 0` already (every
/// eigenvalue of `TᵀT` above one) it must coincide with `H` up to the
/// certified rounding lift.
#[test]
fn orthogonality_psd_majorizer_dominates_exact_hessian_3863() {
    let rho = array![0.0_f64];
    for (n_obs, d, amplitude, expect_indefinite) in
        [(5_usize, 2_usize, 0.3, true), (6, 3, 0.4, true), (6, 2, 3.0, false)]
    {
        let dim = n_obs * d;
        let pen = OrthogonalityPenalty::new(PsiSlice::full(dim, Some(d)), d, 1.7, n_obs, false)
            .expect("valid orthogonality penalty");
        let t = pseudo_random_block(n_obs, d, amplitude);
        let exact = dense_from_probes(dim, |v| pen.hvp(t.view(), rho.view(), v));
        let majorizer =
            dense_from_probes(dim, |v| pen.psd_majorizer_hvp(t.view(), rho.view(), v));
        let scale = spectral_radius(&exact).max(spectral_radius(&majorizer));
        let tol = 1e-12 * scale;
        assert_eq!(min_eigenvalue(&exact) < -tol, expect_indefinite);
        assert!(min_eigenvalue(&majorizer) >= -tol, "majorizer is not PSD");
        let gap = &majorizer - &exact;
        assert!(min_eigenvalue(&gap) >= -tol, "B does not majorize H");
        if !expect_indefinite {
            assert!(
                spectral_radius(&gap) <= 1e-10 * scale,
                "majorizer must equal H where TᵀT ⪰ I"
            );
        }
    }
}

/// #3863 — the frozen operator answered `matvec`/`diag` from the majorizer but
/// `as_dense`/`log det` from the exact Hessian (BlockOrthogonality), or refused
/// the log det outright (Orthogonality). All four faces now read one PSD
/// majorizer.
#[test]
fn frozen_orthogonality_operators_read_one_psd_majorizer_3863() {
    let n_obs = 6;
    let d = 3;
    let dim = n_obs * d;
    let t = pseudo_random_block(n_obs, d, 0.4);
    let kinds = vec![
        AnalyticPenaltyKind::BlockOrthogonality(Arc::new(
            BlockOrthogonalityPenalty::new(
                PsiSlice::full(dim, Some(d)),
                vec![vec![0], vec![1, 2]],
                1.3,
                n_obs,
                false,
            )
            .expect("block orthogonality"),
        )),
        AnalyticPenaltyKind::Orthogonality(Arc::new(
            OrthogonalityPenalty::new(PsiSlice::full(dim, Some(d)), d, 1.3, n_obs, false)
                .expect("orthogonality"),
        )),
    ];
    for kind in kinds {
        let rho = Array1::<f64>::zeros(kind.rho_count());
        let majorizer =
            dense_from_probes(dim, |v| kind.psd_majorizer_hvp(t.view(), rho.view(), v));
        let op = FrozenAnalyticPenaltyOp::new(kind, t.clone(), rho).expect("frozen operator");
        let scale = spectral_radius(&majorizer);
        let dense = op.as_dense();
        let diag = op.diag();
        let mut matvec_dense = Array2::<f64>::zeros((dim, dim));
        let mut e = Array1::<f64>::zeros(dim);
        for j in 0..dim {
            e[j] = 1.0;
            op.matvec(e.view(), matvec_dense.column_mut(j));
            e[j] = 0.0;
        }
        for i in 0..dim {
            assert_abs_diff_eq!(diag[i], majorizer[[i, i]], epsilon = 1e-12 * scale);
            for j in 0..dim {
                assert_abs_diff_eq!(dense[[i, j]], majorizer[[i, j]], epsilon = 1e-12 * scale);
                assert_abs_diff_eq!(matvec_dense[[i, j]], majorizer[[i, j]], epsilon = 1e-12 * scale);
            }
        }
        let lambda = 0.3;
        let expected = <Array2<f64> as PenaltyOp>::log_det_plus_lambda_i(&majorizer, lambda)
            .expect("dense log det");
        let log_det = op.log_det_plus_lambda_i(lambda).expect("frozen log det");
        assert_abs_diff_eq!(log_det, expected, epsilon = 1e-10 * expected.abs().max(1.0));
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

// ----- BlockSparsityPenalty tests -----

fn block_sparsity_test_target() -> Array1<f64> {
    // n_eff = 2 rows, latent_dim = 3, row-major (row*d + axis):
    // T = [[ 0.4, -0.3,  0.2],
    //      [-0.1,  0.6,  0.5]]
    // Group {0, 1} spans both rows (norm² 0.62), group {2} both rows (norm² 0.29).
    array![0.4_f64, -0.3, 0.2, -0.1, 0.6, 0.5]
}

fn build_block_sparsity(learnable_weight: bool) -> BlockSparsityPenalty {
    let t = block_sparsity_test_target();
    BlockSparsityPenalty::new(
        PsiSlice::full(t.len(), Some(3)),
        vec![vec![0_usize, 1], vec![2]],
        0.8,
        2,
        1e-2,
        learnable_weight,
    )
    .expect("valid block sparsity penalty")
}

/// Smallest smoothed group norm `s_g = sqrt(‖T_g‖² + ε²)` of the fixture, and the
/// largest `sqrt(|g|)` factor: the two numbers the finite-difference truncation bounds
/// below read.
fn block_sparsity_fixture_scales(penalty: &BlockSparsityPenalty, t: &Array1<f64>) -> (f64, f64) {
    let d = t.len() / penalty.n_eff;
    let mut min_norm = f64::INFINITY;
    let mut max_size_factor = 0.0_f64;
    for group in &penalty.groups {
        let mut norm2 = penalty.smoothing_eps * penalty.smoothing_eps;
        for row in 0..penalty.n_eff {
            for &axis in group {
                norm2 += t[row * d + axis] * t[row * d + axis];
            }
        }
        min_norm = min_norm.min(norm2.sqrt());
        max_size_factor = max_size_factor.max((group.len() as f64).sqrt());
    }
    (min_norm, max_size_factor)
}

/// The group-lasso gradient `w·sqrt(|g|)·t_i / s_g` is the derivative of the value.
///
/// Tolerance: a central difference with step `h` of a value `P` evaluated to relative
/// accuracy `k·ε` (k = 32 bounds the rounded operations in one value: the squares and
/// sums of a group norm, its square root, the `sqrt(|g|)` scaling and the group sum)
/// errs by at most `k·ε·|P|/h` from rounding plus `h²/6·max|∂³P/∂t_i³|` from
/// truncation. Along one coordinate `P` is `w·sqrt(|g|)·sqrt(x² + c)`, whose third
/// derivative `-3xc/(x² + c)^{5/2}` is bounded by `3/s²`; at `h ≪ s_min` the bound
/// holds at `s_min/2`, giving `h²/6·w·sqrt(|g|)·12/s_min²`.
#[test]
fn block_sparsity_grad_matches_finite_difference() {
    let penalty = build_block_sparsity(false);
    let t = block_sparsity_test_target();
    let rho = Array1::<f64>::zeros(0);
    let grad = penalty.grad_target(t.view(), rho.view());
    let h = 1e-6;
    let fd = gam_linalg_test_support::fd_checker::numerical_gradient_central_diff(
        |tv| penalty.value(tv.view(), rho.view()),
        &t,
        h,
    );
    let (s_min, size_factor) = block_sparsity_fixture_scales(&penalty, &t);
    let value = penalty.value(t.view(), rho.view());
    let tol = 32.0 * f64::EPSILON * value.abs() / h
        + h * h / 6.0 * penalty.weight * size_factor * 12.0 / (s_min * s_min);
    for i in 0..t.len() {
        assert!(
            (grad[i] - fd[i]).abs() <= tol,
            "grad[{i}] = {:.15e}, central difference {:.15e}, bound {tol:.3e}",
            grad[i],
            fd[i]
        );
    }
}

/// The closed-form Hessian-vector product `w·sqrt(|g|)·(v/s − t·(tᵀv)/s³)` is the
/// directional derivative of the gradient.
///
/// Tolerance: the gradient is formed to relative accuracy `k·ε` (k = 32 as above), so
/// the central difference of it along `v` rounds by at most `k·ε·‖∇P‖∞/h`. Its
/// truncation is `h²/6` times the fourth directional derivative of the smoothed norm,
/// bounded by `15·‖v‖³/s³` for `sqrt(‖x‖² + ε²)`; again evaluated at `s_min/2`.
#[test]
fn block_sparsity_hvp_matches_gradient_directional_derivative() {
    let penalty = build_block_sparsity(false);
    let t = block_sparsity_test_target();
    let n = t.len();
    let rho = Array1::<f64>::zeros(0);
    let v: Array1<f64> = Array1::from_shape_fn(n, |i| 0.2 * ((i as f64) + 1.3).cos());
    let hv = penalty.hvp(t.view(), rho.view(), v.view());
    let h = 1e-5;
    let tp = &t + &(&v * h);
    let tm = &t - &(&v * h);
    let gp = penalty.grad_target(tp.view(), rho.view());
    let gm = penalty.grad_target(tm.view(), rho.view());
    let grad = penalty.grad_target(t.view(), rho.view());
    let grad_scale = grad.iter().fold(0.0_f64, |acc, &g| acc.max(g.abs()));
    let v_norm = v.dot(&v).sqrt();
    let (s_min, size_factor) = block_sparsity_fixture_scales(&penalty, &t);
    let half = 0.5 * s_min;
    let tol = 32.0 * f64::EPSILON * grad_scale / h
        + h * h / 6.0 * penalty.weight * size_factor * 15.0 * v_norm.powi(3) / (half * half * half);
    for i in 0..n {
        let fd = (gp[i] - gm[i]) / (2.0 * h);
        assert!(
            (hv[i] - fd).abs() <= tol,
            "hvp[{i}] = {:.15e}, directional difference {fd:.15e}, bound {tol:.3e}",
            hv[i]
        );
    }
}

/// `as_dense` and `diag_target` feed the frozen operator's dense log-determinant and
/// its diagonal. Each column of `as_dense` is the finite-difference-checked `hvp` of
/// the matching unit vector, and `diag_target` is `as_dense`'s diagonal. The three
/// evaluate the same closed form entry by entry, so they agree to the rounding of
/// one entry, bounded by `32·ε` times the largest entry.
#[test]
fn block_sparsity_dense_hessian_and_diagonal_match_hvp() {
    let penalty = build_block_sparsity(true);
    let t = block_sparsity_test_target();
    let n = t.len();
    let rho = array![0.3_f64];
    let dense = penalty.as_dense(t.view(), rho.view());
    let diag = penalty.diag_target(t.view(), rho.view());
    let scale = dense.iter().fold(0.0_f64, |acc, &x| acc.max(x.abs()));
    assert!(scale > 0.0);
    let tol = 32.0 * f64::EPSILON * scale;
    for j in 0..n {
        let mut e = Array1::<f64>::zeros(n);
        e[j] = 1.0;
        let column = penalty.hvp(t.view(), rho.view(), e.view());
        for i in 0..n {
            assert!(
                (dense[[i, j]] - column[i]).abs() <= tol,
                "as_dense[{i}, {j}] = {:.15e}, hvp(e_{j})[{i}] = {:.15e}",
                dense[[i, j]],
                column[i]
            );
        }
        assert!(
            (diag[j] - dense[[j, j]]).abs() <= tol,
            "diag_target[{j}] = {:.15e}, as_dense[{j}, {j}] = {:.15e}",
            diag[j],
            dense[[j, j]]
        );
    }
}

/// With a learnable weight `w·exp(ρ)`, `grad_rho` is the ρ-derivative of the value.
///
/// Tolerance: `P(ρ) = P(0)·exp(ρ)`, so the central difference rounds by at most
/// `k·ε·|P|/h` and truncates by `h²/6·|∂³P/∂ρ³| = h²/6·|P|` (at `h ≪ 1`).
#[test]
fn block_sparsity_grad_rho_matches_finite_difference() {
    let penalty = build_block_sparsity(true);
    let t = block_sparsity_test_target();
    let rho = array![0.3_f64];
    let grad_rho = penalty.grad_rho(t.view(), rho.view());
    assert_eq!(grad_rho.len(), 1);
    let h = 1e-6;
    let value = penalty.value(t.view(), rho.view());
    let fd = (penalty.value(t.view(), array![0.3 + h].view())
        - penalty.value(t.view(), array![0.3 - h].view()))
        / (2.0 * h);
    let tol = 32.0 * f64::EPSILON * value.abs() / h + h * h / 6.0 * value.abs();
    assert!(
        (grad_rho[0] - fd).abs() <= tol,
        "grad_rho = {:.15e}, central difference {fd:.15e}, bound {tol:.3e}",
        grad_rho[0]
    );
}

/// `MechanismSparsityPenalty::as_dense` and `diag_target` feed the frozen operator
/// like the block penalty's; each column of `as_dense` is the finite-difference-checked
/// `hvp` of the matching unit vector and `diag_target` is its diagonal, to the rounding
/// of one entry (`32·ε` times the largest entry).
#[test]
fn mechanism_sparsity_dense_hessian_and_diagonal_match_hvp() {
    let penalty = build_mech_sparsity(0.5);
    let t = mech_sparsity_test_target();
    let n = t.len();
    let rho = Array1::<f64>::zeros(0);
    let dense = penalty.as_dense(t.view(), rho.view());
    let diag = penalty.diag_target(t.view(), rho.view());
    let scale = dense.iter().fold(0.0_f64, |acc, &x| acc.max(x.abs()));
    assert!(scale > 0.0);
    let tol = 32.0 * f64::EPSILON * scale;
    for j in 0..n {
        let mut e = Array1::<f64>::zeros(n);
        e[j] = 1.0;
        let column = penalty.hvp(t.view(), rho.view(), e.view());
        for i in 0..n {
            assert!(
                (dense[[i, j]] - column[i]).abs() <= tol,
                "as_dense[{i}, {j}] = {:.15e}, hvp(e_{j})[{i}] = {:.15e}",
                dense[[i, j]],
                column[i]
            );
        }
        assert!(
            (diag[j] - dense[[j, j]]).abs() <= tol,
            "diag_target[{j}] = {:.15e}, as_dense[{j}, {j}] = {:.15e}",
            diag[j],
            dense[[j, j]]
        );
    }
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
    let fd = gam_linalg_test_support::fd_checker::numerical_gradient_central_diff(
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
    let fd = gam_linalg_test_support::fd_checker::numerical_gradient_central_diff(
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

/// The frozen SCAD/MCP operator applies the PSD majorizer in `matvec`, but its
/// `diag` and `log_det_plus_lambda_i` used to read the exact Hessian diagonal,
/// which carries the concave constant `−1/γ` (MCP) / `−1/(γ−1)` (SCAD) across
/// the taper region: the operator disagreed with itself, and a small `λ`
/// shift made the log-determinant refuse. All three must be the majorizer.
#[test]
fn scadmcp_frozen_diag_and_log_det_are_the_psd_majorizer() {
    let n_eff = 6usize;
    let t = array![0.02_f64, 0.3, 0.9, 1.6, -1.1, -2.5];
    let rho = Array1::<f64>::zeros(0);
    for (variant, gamma) in [(PenaltyConcavity::Mcp, 3.0), (PenaltyConcavity::Scad, 3.7)] {
        let pen = ScadMcpPenalty::new(
            PsiSlice::full(n_eff, Some(1)),
            0.5,
            n_eff,
            gamma,
            1.0e-4,
            variant,
            false,
        )
        .unwrap();
        let exact = pen.hessian_diag(t.view(), rho.view()).unwrap();
        let majorizer = pen.psd_majorizer_diag(t.view(), rho.view()).unwrap();
        let lambda = 0.1;
        assert!(
            exact.iter().any(|&h| h + lambda < 0.0),
            "{variant:?}: the exact diagonal is negative past the λ shift in the taper"
        );
        let kind = AnalyticPenaltyKind::ScadMcp(Arc::new(pen.clone()));
        let op = FrozenAnalyticPenaltyOp::new(kind, t.clone(), rho.clone()).unwrap();
        let diag = op.diag();
        let mut e = Array1::<f64>::zeros(n_eff);
        let mut column = Array1::<f64>::zeros(n_eff);
        let mut expected = 0.0;
        for i in 0..n_eff {
            assert!(majorizer[i] >= 0.0, "{variant:?}: B must be PSD");
            assert!(
                majorizer[i] >= exact[i],
                "{variant:?}: B must dominate the exact Hessian"
            );
            e[i] = 1.0;
            op.matvec(e.view(), column.view_mut());
            e[i] = 0.0;
            assert_eq!(diag[i], majorizer[i], "{variant:?}: diag is the majorizer");
            assert_eq!(column[i], diag[i], "{variant:?}: diag agrees with matvec");
            expected += (majorizer[i] + lambda).ln();
        }
        let log_det = op
            .log_det_plus_lambda_i(lambda)
            .expect("the majorizer log-determinant is finite for every λ > 0");
        assert_abs_diff_eq!(log_det, expected, epsilon = 1e-12 * expected.abs().max(1.0));
    }
}

/// Both SCAD and MCP, with a learnable weight, checked against central
/// differences in every analytic channel: `grad_target` against `value`,
/// `hessian_diag` against `grad_target` (the penalty is coordinate-separable,
/// so the diagonal is the whole Hessian), and `grad_rho` against `value` in
/// the log-weight.
///
/// The effective weight is `w = 0.5·e^{0.2} ≈ 0.611` and `ε = 0.05`. The SCAD
/// knots (`γ = 3.7`) are at `r = w ≈ 0.611` and `r = γw ≈ 2.260`; the MCP knot
/// (`γ = 3`) is at `r = γw ≈ 1.832`. Every probe sits at least 0.16 from a knot,
/// so the `h = 1e-5` stencil never straddles one, and together the probes hit
/// every branch of both variants (SCAD linear / quadratic / flat, MCP
/// active / flat).
///
/// Tolerances come from the stencil error `h²/6·|f'''| + ε_mach·|f|/h`. The
/// largest third derivative is at `t = 0.03` (`r ≈ 0.058`, closest to the
/// smoothing scale):
/// * value→grad: `|f'''| = 3wε²|t|/r⁵ ≈ 2e2`, so the error is ≈ `3e-9`
///   (tolerance `1e-7`);
/// * grad→hessian: `|f''''| = 3wε²|r⁻⁵ − 5t²r⁻⁷| ≈ 2.2e3`, so the error is
///   ≈ `4e-8` (tolerance `1e-6`);
/// * value→grad_rho: on each branch `value = a·w + b·w² + c` in `w = w₀e^ρ`,
///   so `∂³/∂ρ³ = a·w + 8b·w²`, which sums to ≈ 16 over these probes. The
///   error is then ≈ `3e-10` (tolerance `1e-7`).
#[test]
fn scadmcp_learnable_hessian_and_grad_rho_match_central_differences() {
    let t = array![0.03_f64, -0.2, 0.9, -1.4, 2.0, 2.8, -3.1];
    let n_eff = t.len();
    let rho = array![0.2_f64];
    let h = 1.0e-5;
    for (variant, gamma) in [(PenaltyConcavity::Scad, 3.7), (PenaltyConcavity::Mcp, 3.0)] {
        let pen = ScadMcpPenalty::new(
            PsiSlice::full(n_eff, Some(1)),
            0.5,
            n_eff,
            gamma,
            0.05,
            variant,
            true,
        )
        .unwrap();
        assert_eq!(pen.rho_count(), 1);

        let worst_grad = value_grad_fd_max_abs_error(&pen, t.view(), rho.view(), h);
        assert!(
            worst_grad <= 1.0e-7,
            "{variant:?} value↔grad FD max abs error = {worst_grad:.3e}"
        );

        let hess = pen
            .hessian_diag(t.view(), rho.view())
            .expect("SCAD/MCP expose an exact Hessian diagonal");
        let mut tp = t.clone();
        let mut tm = t.clone();
        for i in 0..n_eff {
            tp[i] = t[i] + h;
            tm[i] = t[i] - h;
            let fd = (pen.grad_target(tp.view(), rho.view())[i]
                - pen.grad_target(tm.view(), rho.view())[i])
                / (2.0 * h);
            tp[i] = t[i];
            tm[i] = t[i];
            assert!(
                (hess[i] - fd).abs() <= 1.0e-6,
                "{variant:?} hessian_diag[{i}] = {} vs FD {fd} at t = {}",
                hess[i],
                t[i]
            );
        }

        let grad_rho = pen.grad_rho(t.view(), rho.view());
        assert_eq!(grad_rho.len(), 1);
        let fd_rho = (pen.value(t.view(), array![rho[0] + h].view())
            - pen.value(t.view(), array![rho[0] - h].view()))
            / (2.0 * h);
        assert!(
            (grad_rho[0] - fd_rho).abs() <= 1.0e-7,
            "{variant:?} grad_rho = {} vs FD {fd_rho}",
            grad_rho[0]
        );
    }
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
fn nuclear_norm_right_gram_divided_difference_uses_shared_eigen_shift() {
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

    // The eigen shift is the user's ε² — additive, never a clamp and no floor — so
    // the spectral value and its divided difference remain derivatives of the same
    // function even at an ε far below any absolute scale (here ε² = 1e-20).
    let eigen_shift = smoothing_eps * smoothing_eps;
    let lambda0 = a * a + eigen_shift;
    let lambda1 = b * b + eigen_shift;
    let f0 = lambda0.powf(-0.5);
    let f1 = lambda1.powf(-0.5);
    let expected = ((f0 - f1) / (lambda0 - lambda1)) * a;

    // The code evaluates the filter on the eigensolver's eigenvalues, not on a² and
    // b². A backward-stable eigh returns each within p·ε·λ_max of the exact value
    // (Weyl), and the small shifted eigenvalue λ₀ = 2e-20 sits far below
    // λ_max = 4e-14, so that band is a relative p·ε·λ_max/λ₀ ≈ 9e-10 of it. The
    // coefficient is dominated by f₀ = λ₀^{-1/2}, which moves by half that relative
    // amount. λ₁ moves it by at most p·ε·λ_max/λ₁, and the two evaluations' own
    // roundings (about ten operations each) add γ₂₀. Under the retired 1e-15 floor
    // the coefficient is 265× smaller, far outside this band, so the bar still pins
    // the shift the name promises.
    let eigen_band = p as f64 * f64::EPSILON * lambda1;
    let relative_band = eigen_band / (2.0 * lambda0)
        + eigen_band / lambda1
        + gam_linalg::roundoff::accumulation_growth(20);
    assert_abs_diff_eq!(
        right_filter_derivative[[0, 1]],
        expected,
        epsilon = expected.abs() * relative_band
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

/// #2469: the tie guard at a biting `max_rank` refuses only a gap the right
/// Gram's spectrum cannot resolve: twice the eigensolver band `p·ε·‖G‖₂` plus the
/// `m`-term formation band `γ_m·‖|T|ᵀ|T|‖_F`. `G = diag(1, 1 + 1e-12, 4)` with
/// `max_rank = 2` puts the cutoff between eigenvalues `1e-12` apart, about ninety
/// times twice that band. The `1e-12·(|λ₀| + |λ₁|)` guard it replaced refused it.
/// An exact tie `diag(1, 1, 4)` is still refused.
#[test]
fn nuclear_norm_tie_guard_refuses_only_an_unresolved_gap_2469() {
    let n_eff = 3usize;
    let p = 3usize;
    let pen = |max_rank| {
        let target = PsiSlice {
            range: 0..n_eff * p,
            latent_dim: Some(p),
        };
        NuclearNormPenalty::new(target, 0.8, n_eff, 1.0e-3, max_rank, false).unwrap()
    };
    let v = array![[0.1_f64, 0.2, 0.3], [0.4, 0.5, 0.6], [0.7, 0.8, 0.9]];
    let gap = 1.0e-12_f64;
    let near = array![[1.0_f64, 0.0, 0.0], [0.0, (1.0_f64 + gap).sqrt(), 0.0], [0.0, 0.0, 2.0]];
    let gram_absolute = near.mapv(f64::abs).t().dot(&near.mapv(f64::abs));
    let band = p as f64 * f64::EPSILON * 4.0
        + gam_linalg::roundoff::accumulation_growth(n_eff)
            * gram_absolute.iter().map(|value| value * value).sum::<f64>().sqrt();
    assert!(
        gap > 10.0 * 2.0 * band && gap <= 1.0e-12 * 2.0,
        "fixture premise: the gap {gap:.1e} is resolved (2·band = {:.3e}) and inside the \
         replaced 1e-12·(|λ₀| + |λ₁|) guard",
        2.0 * band
    );
    pen(Some(2))
        .right_spectral_inverse_sqrt_derivative(near.view(), v.view())
        .expect("a resolved gap at the cutoff is not a tie");
    let tied = array![[1.0_f64, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 2.0]];
    let err = pen(Some(2))
        .right_spectral_inverse_sqrt_derivative(tied.view(), v.view())
        .expect_err("an exact tie at the cutoff is refused");
    assert!(err.contains("splits a tied"), "got: {err}");
}

/// #2469: a row joins the wide-block joint row-space basis when its residual is
/// resolved, not above `1e-13·‖row‖`. In `d = 40`, `T = e₀` and `V = e₀ + 5e-14·e₁`
/// leave `V` a residual of `5e-14`. `e₀` is exact, so its direction carries no
/// error, and the residual only has to clear `V`'s own band for one direction
/// (`2·γ₄₄ ≈ 9.8e-15`), which is below the replaced cutoff. The basis has two
/// directions. A row exactly in the span (`V = 3·e₀`) adds none, and a zero row is
/// skipped.
#[test]
fn nuclear_norm_joint_row_space_basis_admits_a_residual_above_its_band_2469() {
    use crate::analytic_penalties::nuclear_norm::joint_row_space_basis;
    let d = 40usize;
    let gap = 5.0e-14_f64;
    let band = gam_math::roundoff::gram_schmidt_residual_band(
        gam_math::roundoff::GRAM_SCHMIDT_PASSES,
        1,
        d,
        1.0,
    );
    assert!(
        gap > band && gap < 1.0e-13,
        "fixture premise: the residual {gap:.1e} sits between the band {band:.3e} and the \
         replaced 1e-13 cutoff"
    );
    let mut t = Array2::<f64>::zeros((1, d));
    t[[0, 0]] = 1.0;
    let mut v = Array2::<f64>::zeros((2, d));
    v[[0, 0]] = 1.0;
    v[[0, 1]] = gap;
    let basis = joint_row_space_basis(t.view(), v.view());
    assert_eq!(basis.len(), 2, "the resolved residual is a second direction");
    assert!((basis[1][1].abs() - 1.0).abs() <= band);

    let mut in_span = Array2::<f64>::zeros((1, d));
    in_span[[0, 0]] = 3.0;
    assert_eq!(joint_row_space_basis(t.view(), in_span.view()).len(), 1);
}

/// #2469: the joint row space does not take a row whose residual only its own
/// band resolves. `T` holds six near-collinear unit rows of `(1, x, …, x⁷)` at
/// nodes `1, 1.05, …, 1.25` (#2600), and `V` their unit fourth difference, which
/// is in their span by construction. Its weights on the six sum to about 8000,
/// so the directions it is measured against are too coarse to resolve it, and
/// the basis keeps six directions.
#[test]
fn nuclear_norm_joint_row_space_skips_a_near_collinear_difference_2469() {
    use crate::analytic_penalties::nuclear_norm::joint_row_space_basis;
    let d = 8usize;
    let mut t = Array2::<f64>::zeros((6, d));
    for index in 0..6 {
        let node = 1.0 + 0.05 * index as f64;
        let mut power = 1.0_f64;
        for column in 0..d {
            t[[index, column]] = power;
            power *= node;
        }
        let norm = t.row(index).dot(&t.row(index)).sqrt();
        t.row_mut(index).mapv_inplace(|value| value / norm);
    }
    let mut difference = Array1::<f64>::zeros(d);
    for (index, weight) in [1.0_f64, -4.0, 6.0, -4.0, 1.0].iter().enumerate() {
        difference.scaled_add(*weight, &t.row(index));
    }
    let length = difference.dot(&difference).sqrt();
    let v = (&difference / length).insert_axis(ndarray::Axis(0));
    assert_eq!(joint_row_space_basis(t.view(), v.view()).len(), 6);
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
fn decoder_incoherence_exact_hvp_matches_fd_of_grad() {
    // #2343 — the exact quotient Hessian-vector product of the degree-0
    // normalized penalty must equal the finite difference of the analytic
    // gradient (the definition of Hv). Two atoms, M=2, p=3, off-collinear so
    // every quotient term (normalizer 1st/2nd derivs, residual C·V) is active.
    let p = 3usize;
    let block_sizes = vec![2usize, 2usize];
    let total: usize = block_sizes.iter().map(|m| m * p).sum();
    let target = PsiSlice {
        range: 0..total,
        latent_dim: Some(total / p),
    };
    let mut coact = Array2::<f64>::zeros((2, 2));
    coact[[0, 1]] = 0.6;
    coact[[1, 0]] = 0.6;
    let pen = DecoderIncoherencePenalty::new(target, block_sizes, p, coact, 0.7, false).unwrap();
    let t = array![
        0.5_f64, -0.3, 0.2, 0.8, -0.1, 0.4, -0.6, 0.7, 0.1, -0.2, 0.9, 0.3
    ];
    let v = array![
        0.2_f64, 0.5, -0.4, 0.3, 0.6, -0.1, 0.7, -0.2, -0.3, 0.15, -0.05, 0.25
    ];
    let rho = Array1::<f64>::zeros(0);
    let analytic = pen.hvp(t.view(), rho.view(), v.view());
    let h = 1e-6_f64;
    let gp = pen.grad_target((&t + &(&v * h)).view(), rho.view());
    let gm = pen.grad_target((&t - &(&v * h)).view(), rho.view());
    let fd = (&gp - &gm) / (2.0 * h);
    let worst = analytic
        .iter()
        .zip(fd.iter())
        .map(|(a, b)| (a - b).abs())
        .fold(0.0_f64, f64::max);
    assert!(
        worst <= 1.0e-5,
        "degree-0 exact hvp vs FD(grad) max abs error = {worst:.3e}"
    );
}

#[test]
fn decoder_incoherence_repulsion_is_radially_free_euler() {
    // #2343 — the degree-0 penalty is homogeneous degree-0 in each decoder
    // block, so by Euler's theorem the radial projection of its gradient onto
    // that block vanishes identically: Σ_{a,o} B_x[a,o]·∂P/∂B_x[a,o] = 0. This
    // is the property that makes the repulsion exert NO amplitude force (the
    // #2343 collapse cure). Assert it at machine precision on a generic config.
    let p = 3usize;
    let block_sizes = vec![2usize, 2usize];
    let total: usize = block_sizes.iter().map(|m| m * p).sum();
    let target = PsiSlice {
        range: 0..total,
        latent_dim: Some(total / p),
    };
    let mut coact = Array2::<f64>::zeros((2, 2));
    coact[[0, 1]] = 0.55;
    coact[[1, 0]] = 0.55;
    let pen = DecoderIncoherencePenalty::new(target, block_sizes, p, coact, 1.3, false).unwrap();
    let t = array![
        0.5_f64, -0.3, 0.2, 0.8, -0.1, 0.4, -0.6, 0.7, 0.1, -0.2, 0.9, 0.3
    ];
    let rho = Array1::<f64>::zeros(0);
    let g = pen.grad_target(t.view(), rho.view());
    // Block 0 = indices 0..6, block 1 = 6..12.
    let radial0: f64 = (0..6).map(|i| t[i] * g[i]).sum();
    let radial1: f64 = (6..12).map(|i| t[i] * g[i]).sum();
    let scale = pen.value(t.view(), rho.view()).abs().max(1.0);
    assert!(
        radial0.abs() <= 1.0e-12 * scale,
        "block-0 radial gradient = {radial0:.3e} (must be ~0 by Euler)"
    );
    assert!(
        radial1.abs() <= 1.0e-12 * scale,
        "block-1 radial gradient = {radial1:.3e} (must be ~0 by Euler)"
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
    // so ||B0·B1^T||_F^2 = 9. #2343: the coherence is now normalized by the LIVE
    // radii ||B0||_F^2 = 5 and ||B1||_F^2 = 25 (degree-0), so
    // value = 0.5·weight(2.0)·coact(0.4)·9/(5·25) = 3.6/125 = 0.0288.
    let beta = array![1.0_f64, 0.0, 0.0, 0.0, 2.0, 0.0, 3.0, 0.0, 4.0];
    let rho = Array1::<f64>::zeros(0);
    let value = pen.value(beta.view(), rho.view());
    assert_abs_diff_eq!(value, 0.0288, epsilon = 1.0e-12);
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
/// `½·weight · Σ_r S[r mod K] · Σ_j target[r,j]²`, only on the harmonic rows.
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
    let penalty = HarmonicRoughnessPenalty::new(weight, n_eff, row_weights.clone(), false).unwrap();
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
    expected *= 0.5 * weight;
    assert_abs_diff_eq!(
        penalty.value(target.view(), rho.view()),
        expected,
        epsilon = 1e-12
    );

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

    // Hessian diagonal is the constant weight·S[r mod K] on every column.
    let diag = penalty.hessian_diag(target.view(), rho.view()).unwrap();
    for r in 0..n_eff {
        let w = row_weights[r % row_weights.len()];
        for j in 0..d {
            assert_abs_diff_eq!(diag[r * d + j], weight * w, epsilon = 1e-12);
        }
    }
}

/// #2900 — `FrozenAnalyticPenaltyOp` used to answer `diag` with 32 Hutchinson
/// samples and `log det(S + λI)` with 16-probe SLQ above dimension 1024, so one
/// operator returned exact values on one side of a size window and estimates on the
/// other. Both are exact at every dimension now. One dimension past the old window,
/// on a group-sparsity Hessian whose within-group coupling puts off-diagonal mass in
/// every row (the mass a Hutchinson diagonal averages over), the diagonal equals the
/// frozen operator's own dense form, and the log-determinant equals that dense form's
/// eigensolve.
#[test]
fn frozen_penalty_diag_and_log_det_are_exact_past_dimension_1024_2900() {
    let n_eff = 205;
    let latent_dim = 5;
    let dim = n_eff * latent_dim;
    let penalty = BlockSparsityPenalty::new(
        PsiSlice::full(dim, Some(latent_dim)),
        vec![vec![0, 1], vec![2, 3, 4]],
        0.7,
        n_eff,
        1e-3,
        true,
    )
    .expect("block sparsity penalty");
    let kind = AnalyticPenaltyKind::BlockSparsity(Arc::new(penalty));
    let rho = Array1::<f64>::zeros(kind.rho_count());
    let target = Array1::from_shape_fn(dim, |i| ((i * 37 % 101) as f64 - 50.0) / 25.0);
    let op = FrozenAnalyticPenaltyOp::new(kind, target, rho).expect("frozen operator");
    assert_eq!(op.dim(), 1025);

    let dense = op.as_dense();
    let diag = op.diag();
    let off_diagonal_mass = dense
        .indexed_iter()
        .filter(|((i, j), _)| i != j)
        .fold(0.0_f64, |acc, (_, &v)| acc.max(v.abs()));
    let diagonal_scale = dense.diag().iter().fold(0.0_f64, |acc, &v| acc.max(v.abs()));
    assert!(diagonal_scale > 0.0 && off_diagonal_mass > 0.0);
    for i in 0..dim {
        assert_abs_diff_eq!(diag[i], dense[[i, i]], epsilon = 1e-12 * diagonal_scale);
    }

    let lambda = 0.3;
    let exact = <Array2<f64> as PenaltyOp>::log_det_plus_lambda_i(&dense, lambda)
        .expect("dense log det");
    let log_det = op.log_det_plus_lambda_i(lambda).expect("frozen log det");
    assert_abs_diff_eq!(log_det, exact, epsilon = 1e-9 * exact.abs().max(1.0));
}

/// Checks `grad_target`, `hvp` and `grad_rho` of a learnable-weight quadratic penalty
/// `P(t, ρ) = ½·w·tᵀHt − ½·len·ln w`, `w = w₀·exp(ρ)`, against central differences
/// of `value` and `grad_target`.
///
/// `unit_abs(x)` is `|H|·x` for `w = 1`: applied to `|t|` it bounds, entry by entry,
/// the sums of absolute terms a component of `H·t` is formed from, so `k·ε·w·unit_abs`
/// bounds the rounding of the analytic gradient (k = 32 covers the few rounded
/// operations per term). The value rounds by at most `k·ε·value_scale`, with
/// `value_scale(x, w) = ½·w·xᵀ·unit_abs(x) + ½·len·(|ln w| + 1)`.
///
/// * `P` is quadratic in `t`, so the central difference of the value along `e_i` has
///   no truncation error; it rounds by at most `k·ε·value_scale(|t| + h, w)/h`.
/// * `grad_target` is linear in `t`, so its central difference along `v` is exact in
///   exact arithmetic and rounds by at most `k·ε·w·unit_abs(|t| + h|v|)_i/h`; the
///   product itself rounds by `k·ε·w·unit_abs(|v|)_i`.
/// * Along `ρ`, `∂³P/∂ρ³ = ½·w·tᵀHt`, so the central difference in `ρ` truncates by at
///   most `h²/6·½·w(ρ + h)·|t|ᵀ·unit_abs(|t|)` and rounds by
///   `k·ε·value_scale(|t|, w(ρ ± h))/h`.
fn assert_learnable_quadratic_penalty_derivatives<P: AnalyticPenalty>(
    penalty: &P,
    base_weight: f64,
    t: &Array1<f64>,
    rho: f64,
    v: &Array1<f64>,
    unit_abs: impl Fn(&Array1<f64>) -> Array1<f64>,
) {
    let k = 32.0 * f64::EPSILON;
    let n = t.len();
    let len = n as f64;
    let value_scale =
        |x: &Array1<f64>, w: f64| 0.5 * w * x.dot(&unit_abs(x)) + 0.5 * len * (w.ln().abs() + 1.0);
    let rho_at = |r: f64| array![r];
    let rho_v = rho_at(rho);
    let w = base_weight * rho.exp();
    let t_abs = t.mapv(f64::abs);
    let v_abs = v.mapv(f64::abs);

    let h = 1e-3;
    let grad = penalty.grad_target(t.view(), rho_v.view());
    let grad_bound = unit_abs(&t_abs);
    let value_round = k * value_scale(&t_abs.mapv(|x| x + h), w) / h;
    for i in 0..n {
        let mut tp = t.clone();
        tp[i] += h;
        let mut tm = t.clone();
        tm[i] -= h;
        let fd = (penalty.value(tp.view(), rho_v.view()) - penalty.value(tm.view(), rho_v.view()))
            / (2.0 * h);
        let tol = value_round + k * w * grad_bound[i];
        assert!(
            (grad[i] - fd).abs() <= tol,
            "grad[{i}] = {:.15e}, central difference {fd:.15e}, bound {tol:.3e}",
            grad[i]
        );
    }

    let hv = penalty.hvp(t.view(), rho_v.view(), v.view());
    let gp = penalty.grad_target((t + &(v * h)).view(), rho_v.view());
    let gm = penalty.grad_target((t - &(v * h)).view(), rho_v.view());
    let fd_bound = unit_abs(&(&t_abs + &(&v_abs * h)));
    let hv_bound = unit_abs(&v_abs);
    for i in 0..n {
        let fd = (gp[i] - gm[i]) / (2.0 * h);
        let tol = k * (w * fd_bound[i] / h + w * hv_bound[i]);
        assert!(
            (hv[i] - fd).abs() <= tol,
            "hvp[{i}] = {:.15e}, directional difference {fd:.15e}, bound {tol:.3e}",
            hv[i]
        );
    }

    let h_rho = 1e-4;
    let grad_rho = penalty.grad_rho(t.view(), rho_v.view());
    assert_eq!(grad_rho.len(), 1);
    let fd = (penalty.value(t.view(), rho_at(rho + h_rho).view())
        - penalty.value(t.view(), rho_at(rho - h_rho).view()))
        / (2.0 * h_rho);
    let w_hi = base_weight * (rho + h_rho).exp();
    let w_lo = base_weight * (rho - h_rho).exp();
    let tol = k * value_scale(&t_abs, w_hi).max(value_scale(&t_abs, w_lo)) / h_rho
        + h_rho * h_rho / 6.0 * 0.5 * w_hi * t_abs.dot(&grad_bound);
    assert!(
        (grad_rho[0] - fd).abs() <= tol,
        "grad_rho = {:.15e}, central difference {fd:.15e}, bound {tol:.3e}",
        grad_rho[0]
    );
}

fn build_row_precision_fixture() -> RowPrecisionPriorPenalty {
    let lambda = array![
        [[2.0_f64, 0.3], [0.3, 1.0]],
        [[1.5, -0.4], [-0.4, 0.9]],
        [[0.8, 0.2], [0.2, 1.2]]
    ];
    RowPrecisionPriorPenalty::new(PsiSlice::full(6, Some(2)), lambda, 0.6, 3, true)
        .expect("valid row precision prior")
}

/// The row-precision prior's gradient `w·Λ_n t_n`, its curvature `w·Λ_n v_n` and its
/// strength derivative `½·w·Σ t_nᵀΛ_n t_n − ½·len` are the derivatives of its value.
#[test]
fn row_precision_prior_derivatives_match_finite_differences() {
    let penalty = build_row_precision_fixture();
    let t = array![0.4_f64, -0.3, 0.2, -0.1, 0.6, 0.5];
    let v = Array1::from_shape_fn(t.len(), |i| 0.2 * ((i as f64) + 1.3).cos());
    let lambda = penalty.lambda_per_row.clone();
    assert_learnable_quadratic_penalty_derivatives(&penalty, 0.6, &t, -0.2, &v, |x| {
        Array1::from_shape_fn(x.len(), |row| {
            let (n, i) = (row / 2, row % 2);
            (0..2)
                .map(|j| lambda[[n, i, j]].abs() * x[n * 2 + j])
                .sum::<f64>()
        })
    });
}

/// `as_dense` and `diag_target` of the row-precision prior feed the frozen operator's
/// exact log-determinant and its diagonal. Every entry of all three is the single
/// product `w·Λ_nij` (a column of `hvp` adds exact zeros to it), so they agree
/// exactly; the off-diagonal `Λ` makes the diagonal-only `hessian_diag` decline.
#[test]
fn row_precision_prior_dense_hessian_and_diagonal_match_hvp() {
    let penalty = build_row_precision_fixture();
    let t = array![0.4_f64, -0.3, 0.2, -0.1, 0.6, 0.5];
    let n = t.len();
    let rho = array![-0.2_f64];
    let dense = penalty.as_dense(t.view(), rho.view());
    let diag = penalty.diag_target(t.view(), rho.view());
    for j in 0..n {
        let mut e = Array1::<f64>::zeros(n);
        e[j] = 1.0;
        assert_eq!(
            dense.column(j).to_owned(),
            penalty.hvp(t.view(), rho.view(), e.view()),
            "as_dense column {j} differs from hvp(e_{j})"
        );
        assert_eq!(
            diag[j],
            dense[[j, j]],
            "diag_target[{j}] differs from as_dense"
        );
    }
    assert!(penalty.hessian_diag(t.view(), rho.view()).is_none());
}

/// The row-precision energy ½ tᵀΛt reads only the symmetric part of Λ, so a
/// penalty built from an asymmetric Λ is the penalty built from (Λ + Λᵀ)/2:
/// value, gradient, curvature and the log-determinant agree exactly. An input
/// whose symmetric part is singular is still refused as not positive definite,
/// whatever its skew part (#2469).
#[test]
fn row_precision_prior_reads_the_symmetric_part_of_its_precision_2469() {
    let target = PsiSlice::full(4, Some(2));
    let asymmetric = array![[[2.0_f64, 1.5], [0.5, 2.0]], [[3.0, -0.25], [0.75, 1.0]]];
    let symmetric = array![[[2.0_f64, 1.0], [1.0, 2.0]], [[3.0, 0.25], [0.25, 1.0]]];
    let from_asymmetric =
        RowPrecisionPriorPenalty::new(target.clone(), asymmetric, 1.3, 2, true).unwrap();
    let from_symmetric =
        RowPrecisionPriorPenalty::new(target.clone(), symmetric.clone(), 1.3, 2, true).unwrap();
    assert_eq!(from_asymmetric.lambda_per_row, symmetric);

    let t = array![0.7_f64, -1.1, 0.4, 2.3];
    let rho = array![0.2_f64];
    assert_eq!(
        from_asymmetric.value(t.view(), rho.view()),
        from_symmetric.value(t.view(), rho.view())
    );
    assert_eq!(
        from_asymmetric.grad_target(t.view(), rho.view()),
        from_symmetric.grad_target(t.view(), rho.view())
    );
    assert_eq!(
        from_asymmetric.as_dense(t.view(), rho.view()),
        from_symmetric.as_dense(t.view(), rho.view())
    );
    assert_eq!(
        from_asymmetric.log_det_plus_lambda_i(rho.view(), 0.5).unwrap(),
        from_symmetric.log_det_plus_lambda_i(rho.view(), 0.5).unwrap()
    );
    // The value is the quadratic form of the stored matrix plus the Gaussian
    // normalizer −½·len·ln μ of the learnable strength.
    let weight = 1.3 * 0.2_f64.exp();
    let mut energy = 0.0;
    for n in 0..2 {
        for i in 0..2 {
            for j in 0..2 {
                energy += t[2 * n + i] * symmetric[[n, i, j]] * t[2 * n + j];
            }
        }
    }
    assert_abs_diff_eq!(
        from_asymmetric.value(t.view(), rho.view()),
        0.5 * weight * energy - 0.5 * 4.0 * weight.ln(),
        epsilon = 1e-12
    );

    // [[1, 3], [−1, 1]] has symmetric part [[1, 1], [1, 1]], eigenvalues {0, 2}.
    let singular_part = array![[[1.0_f64, 3.0], [-1.0, 1.0]], [[1.0, 0.0], [0.0, 1.0]]];
    let refused = RowPrecisionPriorPenalty::new(target, singular_part, 1.3, 2, true);
    assert!(refused.unwrap_err().contains("must be positive definite"));
}

// ----- IvaeRidgeMeanGauge tests -----

/// The iVAE ridge gauge had no Rust test of `hvp`, `grad_rho`, `as_dense`,
/// `diag_target` or its frozen operator. The only check was a Python
/// value -> grad test with a fixed weight and `ε = 1e-6`. At that ε the ridge
/// hat matrix `P = U(UᵀU + εI)⁻¹Uᵀ` is almost idempotent, so it could not tell
/// `tᵀ(I - P)t` from `||t - Pt||²`. Here `ε = 0.5` separates the two by about
/// 15%. The value is pinned against the minimized ridge objective
/// `0.5 μ (||T - UB̂||² + ε||B̂||²)`, with `B̂` solved by an explicit 2 × 2
/// inverse that is independent of the penalty's eigendecomposition. The test
/// then checks value -> `grad_target`, `grad_target` -> `hvp`, value ->
/// `grad_rho` (learnable log-weight), the `as_dense` columns against `hvp`, and
/// `diag_target` and the frozen operator against the dense matrix. It also
/// checks that the curvature `μ(I - P) ⊗ I_d` is symmetric positive definite.
///
/// Tolerances. The value is quadratic in `t`, so the central stencil has no
/// truncation error in `t`. Only rounding remains, about
/// `ε_mach |f| / h ≈ 2.2e-16 · 2 / 1e-5 ≈ 4e-11`. In ρ the value is
/// `0.5 w₀ e^ρ A - 0.5 N ln(w₀ e^ρ)`. Its third ρ-derivative is
/// `0.5 w₀ e^ρ A ≲ 1`, so the truncation error is `h²/6 ≈ 2e-11`. A `1e-8`
/// tolerance leaves more than 100× margin. The dense, diagonal and `hvp`
/// channels all apply the same `P` entries, so they agree to rounding (`1e-12`
/// relative). The ridge-objective check is a reassociation of the same sums,
/// hence `1e-12` relative. The curvature eigenvalues lie in
/// `[μ ε / (σ_max² + ε), μ] ≈ [0.059, 0.86]`. With `λ = 0.3`, each `log` in
/// `log det` carries about `ε_mach · 1.2 / 0.3`, so `1e-10 · max(|log det|, 1)`
/// leaves several orders of margin.
#[test]
fn ivae_ridge_mean_gauge_is_the_ridge_objective_and_its_derivatives_match_central_differences() {
    let (n_eff, d, q) = (6usize, 2usize, 2usize);
    let n = n_eff * d;
    let ridge_eps = 0.5;
    let w0 = 0.7;
    let h = 1e-5;
    let tol = 1e-8;
    let lambda = 0.3;
    let aux = Array2::from_shape_fn((n_eff, q), |(row, col)| {
        if col == 0 {
            1.0
        } else {
            0.8 * (0.7 * row as f64 + 0.2).sin()
        }
    });
    let t = Array1::from_shape_fn(n, |i| {
        let x = i as f64;
        0.45 * (1.1 * x + 0.3).cos() + 0.07 * x
    });
    let v = Array1::from_shape_fn(n, |i| 0.5 * (0.8 * i as f64 + 1.0).sin());

    let fixed = IvaeRidgeMeanGauge::new(
        PsiSlice::full(n, Some(d)),
        aux.clone(),
        ridge_eps,
        w0,
        n_eff,
        false,
    )
    .expect("fixed-weight iVAE gauge");
    let gram = aux.t().dot(&aux) + ridge_eps * Array2::<f64>::eye(q);
    let det = gram[[0, 0]] * gram[[1, 1]] - gram[[0, 1]] * gram[[1, 0]];
    let gram_inv = array![
        [gram[[1, 1]] / det, -gram[[0, 1]] / det],
        [-gram[[1, 0]] / det, gram[[0, 0]] / det]
    ];
    let t_mat = t
        .clone()
        .into_shape_with_order((n_eff, d))
        .expect("row-major latent");
    let b_hat = gram_inv.dot(&aux.t().dot(&t_mat));
    let fit_residual = &t_mat - &aux.dot(&b_hat);
    let ridge_objective = 0.5
        * w0
        * (fit_residual.iter().map(|x| x * x).sum::<f64>()
            + ridge_eps * b_hat.iter().map(|x| x * x).sum::<f64>());
    let empty_rho = Array1::<f64>::zeros(0);
    let fixed_value = fixed.value(t.view(), empty_rho.view());
    assert_abs_diff_eq!(
        fixed_value,
        ridge_objective,
        epsilon = 1e-12 * ridge_objective.abs()
    );
    assert_eq!(fixed.grad_rho(t.view(), empty_rho.view()).len(), 0);

    let pen = IvaeRidgeMeanGauge::new(PsiSlice::full(n, Some(d)), aux, ridge_eps, w0, n_eff, true)
        .expect("learnable iVAE gauge");
    let rho = array![0.2_f64];

    let g = pen.grad_target(t.view(), rho.view());
    for i in 0..n {
        let mut tp = t.clone();
        let mut tm = t.clone();
        tp[i] += h;
        tm[i] -= h;
        let fd = (pen.value(tp.view(), rho.view()) - pen.value(tm.view(), rho.view())) / (2.0 * h);
        assert_abs_diff_eq!(g[i], fd, epsilon = tol);
    }

    let hv = pen.hvp(t.view(), rho.view(), v.view());
    let tp = &t + &(h * &v);
    let tm = &t - &(h * &v);
    let gp = pen.grad_target(tp.view(), rho.view());
    let gm = pen.grad_target(tm.view(), rho.view());
    for i in 0..n {
        assert_abs_diff_eq!(hv[i], (gp[i] - gm[i]) / (2.0 * h), epsilon = tol);
    }

    let gr = pen.grad_rho(t.view(), rho.view());
    assert_eq!(gr.len(), 1, "one learnable log-weight");
    let fd_rho = (pen.value(t.view(), array![0.2 + h].view())
        - pen.value(t.view(), array![0.2 - h].view()))
        / (2.0 * h);
    assert_abs_diff_eq!(gr[0], fd_rho, epsilon = tol);

    let dense = pen.as_dense(t.view(), rho.view());
    let scale = dense.iter().fold(0.0_f64, |acc, &x| acc.max(x.abs()));
    assert!(scale > 0.0, "the gauge curvature must be nonzero");
    let diag = pen.diag_target(t.view(), rho.view());
    for col in 0..n {
        let mut e = Array1::<f64>::zeros(n);
        e[col] = 1.0;
        let column = pen.hvp(t.view(), rho.view(), e.view());
        for row in 0..n {
            assert_abs_diff_eq!(dense[[row, col]], column[row], epsilon = 1e-12 * scale);
            assert_abs_diff_eq!(
                dense[[row, col]],
                dense[[col, row]],
                epsilon = 1e-12 * scale
            );
        }
        assert_abs_diff_eq!(diag[col], dense[[col, col]], epsilon = 1e-12 * scale);
    }
    let (evals, _) = dense.eigh(Side::Lower).expect("iVAE gauge curvature eigh");
    let min_eval = evals.iter().fold(f64::INFINITY, |acc, &x| acc.min(x));
    assert!(
        min_eval > 0.0,
        "μ(I - P) is positive definite for ε > 0; min eigenvalue {min_eval:.3e}"
    );

    let expected = <Array2<f64> as PenaltyOp>::log_det_plus_lambda_i(&dense, lambda)
        .expect("dense iVAE gauge log det");
    let op = FrozenAnalyticPenaltyOp::new(
        AnalyticPenaltyKind::IvaeRidgeMeanGauge(Arc::new(pen)),
        t.clone(),
        rho.clone(),
    )
    .expect("frozen iVAE gauge operator");
    let frozen_diag = op.diag();
    for i in 0..n {
        assert_abs_diff_eq!(frozen_diag[i], dense[[i, i]], epsilon = 1e-12 * scale);
    }
    let frozen_log_det = op
        .log_det_plus_lambda_i(lambda)
        .expect("frozen iVAE gauge log det");
    assert_abs_diff_eq!(
        frozen_log_det,
        expected,
        epsilon = 1e-10 * expected.abs().max(1.0)
    );
}

// ----- BlockSparsityPenalty tests -----

/// The group-lasso penalty's only Rust test compared the frozen operator's
/// diagonal and log-determinant with `as_dense`. Nothing checked `as_dense`,
/// `grad_target`, `hvp` or `grad_rho` against the value itself. This pins
/// value -> `grad_target`, `grad_target` -> `hvp` (directional) and
/// value -> `grad_rho` (learnable log-weight) against central differences.
/// It also checks the `as_dense` columns against `hvp` of the unit vectors,
/// `diag_target` against the dense diagonal, and that the Hessian
/// `w √|g| (I/s - x xᵀ/s³)` of the smoothed norm `s = √(‖x‖² + ε²)` is
/// positive definite: its smallest eigenvalue per group is `w √|g| ε²/s³ > 0`.
///
/// Tolerances. The smoothed norm's directional derivatives are bounded by
/// `|s'''| ≤ 0.86/ε²` and `|s''''| ≤ 3/ε³`. With `ε = 0.3`,
/// `w = 0.7 e^{0.2} ≈ 0.86` and `√|g| ≤ √2`, that gives
/// `|f'''| ≲ 12` and `|f''''| ≲ 140`. With `h = 1e-5`:
/// - value -> grad: `h²/6 · 12 + ε_mach · 3/h ≈ 3e-10`.
/// - grad -> hvp: along `v` with `‖v‖ ≈ 1.5`, `h²/6 · 140 · 1.5³ ≈ 8e-9`.
/// - value -> grad_rho: the value is `w₀ e^ρ A`, so its third ρ-derivative is
///   the value (`≈ 2.8`), which gives `≈ 5e-11`.
/// A `1e-7` tolerance leaves at least 10× margin. The dense, diagonal and
/// `hvp` channels evaluate the same per-group factors, so they agree to
/// rounding (`1e-12` relative).
#[test]
fn block_sparsity_derivatives_and_dense_hessian_match_central_differences() {
    let (n_eff, d) = (4usize, 3usize);
    let n = n_eff * d;
    let h = 1e-5;
    let tol = 1e-7;
    let t = Array1::from_shape_fn(n, |i| {
        let x = i as f64;
        0.37 * (1.3 * x).sin() + 0.11 * x - 0.4
    });
    let v = Array1::from_shape_fn(n, |i| 0.6 * (0.9 * i as f64 + 0.4).cos());
    let pen = BlockSparsityPenalty::new(
        PsiSlice::full(n, Some(d)),
        vec![vec![0, 2], vec![1]],
        0.7,
        n_eff,
        0.3,
        true,
    )
    .expect("block sparsity penalty");
    let rho = array![0.2_f64];

    let g = pen.grad_target(t.view(), rho.view());
    for i in 0..n {
        let mut tp = t.clone();
        let mut tm = t.clone();
        tp[i] += h;
        tm[i] -= h;
        let fd = (pen.value(tp.view(), rho.view()) - pen.value(tm.view(), rho.view())) / (2.0 * h);
        assert_abs_diff_eq!(g[i], fd, epsilon = tol);
    }

    let hv = pen.hvp(t.view(), rho.view(), v.view());
    let tp = &t + &(h * &v);
    let tm = &t - &(h * &v);
    let gp = pen.grad_target(tp.view(), rho.view());
    let gm = pen.grad_target(tm.view(), rho.view());
    for i in 0..n {
        assert_abs_diff_eq!(hv[i], (gp[i] - gm[i]) / (2.0 * h), epsilon = tol);
    }

    let gr = pen.grad_rho(t.view(), rho.view());
    assert_eq!(gr.len(), 1, "one learnable log-weight");
    let fd_rho = (pen.value(t.view(), array![0.2 + h].view())
        - pen.value(t.view(), array![0.2 - h].view()))
        / (2.0 * h);
    assert_abs_diff_eq!(gr[0], fd_rho, epsilon = tol);

    let dense = pen.as_dense(t.view(), rho.view());
    let scale = dense.iter().fold(0.0_f64, |acc, &x| acc.max(x.abs()));
    assert!(scale > 0.0, "group-lasso curvature must be nonzero");
    let diag = pen.diag_target(t.view(), rho.view());
    for col in 0..n {
        let mut e = Array1::<f64>::zeros(n);
        e[col] = 1.0;
        let column = pen.hvp(t.view(), rho.view(), e.view());
        for row in 0..n {
            assert_abs_diff_eq!(dense[[row, col]], column[row], epsilon = 1e-12 * scale);
        }
        assert_abs_diff_eq!(diag[col], dense[[col, col]], epsilon = 1e-12 * scale);
    }
    let (evals, _) = dense.eigh(Side::Lower).expect("group-lasso Hessian eigh");
    let min_eval = evals.iter().fold(f64::INFINITY, |acc, &x| acc.min(x));
    assert!(
        min_eval > 0.0,
        "smoothed group-lasso Hessian must be positive definite; min eigenvalue {min_eval:.3e}"
    );
}

/// `OrthogonalityPenalty`, `½ (w e^ρ / n_eff) ‖TᵀT − I‖²_F`, had no Rust test.
/// On a target with `TᵀT ≺ I` (Gram eigenvalues about 0.13 and 0.93), where the
/// exact Hessian is indefinite, this pins
/// value -> `grad_target`, `grad_target` -> `hvp` (directional),
/// value -> `grad_rho` (learnable log-weight), and the closed-form dense Hessian
/// `as_dense_with_precomputed_m` against `hvp` of the unit vectors. It also
/// checks that the dense Hessian is symmetric.
///
/// Tolerances. The stencil error is `h²/6 |f'''| + ε_mach |f| / h` with
/// `h = 1e-5` and `s = 0.9 e^{0.25} / 4 ≈ 0.29`.
/// - The value is quartic in `T`. Along a coordinate `|f'''| ≲ 1.8` on this
///   target and `|f| ≈ 0.11`, which gives about `3e-11`.
/// - The gradient `2 s T (TᵀT − I)` is cubic. Its third derivative along `V` is
///   bounded by `12 s ‖V‖³_F ≈ 6` (`‖V‖²_F ≈ 1.4`), and `|g| ≲ 1`, which gives
///   about `1e-10 + 2e-11`.
/// - The value is `e^ρ` times a constant, so its third ρ-derivative is the
///   value itself, and the error is about `2e-12`.
/// A `1e-8` tolerance leaves at least 80× margin. The dense form and `hvp`
/// evaluate the same bilinear terms in a different order, so they agree to
/// rounding (`1e-12` relative).
#[test]
fn orthogonality_derivatives_and_dense_hessian_match_central_differences() {
    let (n_obs, d) = (4usize, 2usize);
    let n = n_obs * d;
    let pen = OrthogonalityPenalty::new(PsiSlice::full(n, Some(d)), d, 0.9, n_obs, true)
        .expect("orthogonality penalty");
    assert_eq!(pen.rho_count(), 1);
    let rho = array![0.25_f64];
    let t = Array1::from_shape_fn(n, |i| {
        let x = i as f64;
        0.5 * (0.7 * x + 0.2).sin() + 0.15 * (1.9 * x).cos()
    });
    let v = Array1::from_shape_fn(n, |i| 0.6 * (0.9 * i as f64 + 0.4).cos());
    let h = 1e-5;
    let tol = 1e-8;

    let g = pen.grad_target(t.view(), rho.view());
    for i in 0..n {
        let mut tp = t.clone();
        let mut tm = t.clone();
        tp[i] += h;
        tm[i] -= h;
        let fd = (pen.value(tp.view(), rho.view()) - pen.value(tm.view(), rho.view())) / (2.0 * h);
        assert_abs_diff_eq!(g[i], fd, epsilon = tol);
    }

    let hv = pen.hvp(t.view(), rho.view(), v.view());
    let tp = &t + &(h * &v);
    let tm = &t - &(h * &v);
    let gp = pen.grad_target(tp.view(), rho.view());
    let gm = pen.grad_target(tm.view(), rho.view());
    for i in 0..n {
        assert_abs_diff_eq!(hv[i], (gp[i] - gm[i]) / (2.0 * h), epsilon = tol);
    }

    let gr = pen.grad_rho(t.view(), rho.view());
    assert_eq!(gr.len(), 1);
    let fd_rho = (pen.value(t.view(), array![0.25 + h].view())
        - pen.value(t.view(), array![0.25 - h].view()))
        / (2.0 * h);
    assert_abs_diff_eq!(gr[0], fd_rho, epsilon = tol);

    let t_mat = pen.target_matrix(t.view()).expect("n_obs x d target");
    let gram = OrthogonalityPenalty::gram_minus_identity(t_mat.view());
    let dense = pen.as_dense_with_precomputed_m(t_mat.view(), gram.view(), pen.scale(rho.view()));
    let scale = dense.iter().fold(0.0_f64, |acc, &x| acc.max(x.abs()));
    assert!(scale > 0.0, "curvature must be nonzero on this target");
    for col in 0..n {
        let mut e = Array1::<f64>::zeros(n);
        e[col] = 1.0;
        let column = pen.hvp(t.view(), rho.view(), e.view());
        for row in 0..n {
            assert_abs_diff_eq!(dense[[row, col]], column[row], epsilon = 1e-12 * scale);
            assert_abs_diff_eq!(
                dense[[row, col]],
                dense[[col, row]],
                epsilon = 1e-12 * scale
            );
        }
    }
}

/// The nested-prefix Hessian diagonal had only a sign check. This pins the full
/// FD Jacobian of `grad_target` against `diag(hessian_diag)`, so it also checks
/// that the Hessian is exactly diagonal. The fixture has three shells, so the
/// per-axis weights `W_i` take three distinct values, and it includes a zero entry.
///
/// Tolerance. The per-entry gradient is `W x / sqrt(x² + ε²)`. Its third
/// derivative is `−3 W ε² (r² − 5x²) / r⁷`, bounded by `3 W / ε³` (at `x = 0`).
/// With `ε = 0.3` and `W ≤ 0.7 e^{0.1} + 0.5 e^{−0.2} + 0.3 e^{0.3} ≈ 1.59`,
/// that is `≈ 176`, so the `h = 1e-5` stencil error is at most about `3e-9`.
/// The roundoff is `ε_mach · 1.6 / h ≈ 4e-11`. A `1e-7` tolerance leaves 30× margin.
#[test]
fn nested_prefix_hessian_diag_matches_finite_difference_of_gradient() {
    let (t, _n, f) = nested_prefix_test_target();
    let n = t.len();
    let pen = NestedPrefixPenalty::new(
        PsiSlice::full(n, Some(f)),
        PenaltyTier::Psi,
        vec![1_usize, 2, 4],
        vec![0.7, 0.5, 0.3],
        0.3,
    )
    .expect("valid nested-prefix penalty");
    let rho = array![0.1_f64, -0.2, 0.3];
    let diag = pen
        .hessian_diag(t.view(), rho.view())
        .expect("nested-prefix Hessian is diagonal");
    let h = 1e-5;
    for i in 0..n {
        let mut tp = t.clone();
        let mut tm = t.clone();
        tp[i] += h;
        tm[i] -= h;
        let gp = pen.grad_target(tp.view(), rho.view());
        let gm = pen.grad_target(tm.view(), rho.view());
        for j in 0..n {
            let expected = if i == j { diag[i] } else { 0.0 };
            assert_abs_diff_eq!((gp[j] - gm[j]) / (2.0 * h), expected, epsilon = 1e-7);
        }
    }
}



// ---------------------------------------------------------------------------
// Central-difference consistency for analytic derivatives that had no FD pin:
// smoothed total variation (path and graph operators), soft monotonicity, the
// iVAE ridge conditional-mean gauge, group-lasso block sparsity, the coupled
// row-precision prior, the Stiefel orthogonality gauge, and the normalized
// cross-Gram composition behind the decoder-incoherence and subspace-overlap
// priors.
//
// Tolerance derivation. A central difference with step h along a probe v has
// truncation error h²/6 · |∂³_v f| and roundoff ≈ u·|f|/h (u = 2.2e-16). The
// fixtures keep every coordinate and probe entry O(1) and every smoothing scale
// ≥ 0.2, so the largest third directional derivative that enters any check
// (the TV kernel's fourth derivative 3/ε³ ≈ 375 times |D v|³ ≤ 13) is below
// 5e3: at h = 1e-5 truncation is ≤ 1e-7 and roundoff ≤ 1e-10. The asserted
// 1e-6·(1 + |analytic|) bound therefore has an order of magnitude of margin,
// while any dropped, mis-signed or mis-scaled analytic term (an O(1) relative
// error) exceeds it by several orders.
// ---------------------------------------------------------------------------

const DERIV_FD_STEP: f64 = 1.0e-5;
const DERIV_FD_TOL: f64 = 1.0e-6;

fn assert_fd_close(label: &str, analytic: f64, fd: f64) {
    let bound = DERIV_FD_TOL * (1.0 + analytic.abs());
    let err = (analytic - fd).abs();
    assert!(
        err <= bound,
        "{label}: analytic {analytic:.12e} vs central difference {fd:.12e} (|Δ| = {err:.3e} > {bound:.3e})"
    );
}

/// Pins value→`grad_target`, `grad_target`→`hvp` along every probe, the convex
/// identity `psd_majorizer_hvp == hvp`, and value→`grad_rho`.
fn assert_penalty_derivatives_match_fd(
    label: &str,
    pen: &dyn AnalyticPenalty,
    target: ArrayView1<'_, f64>,
    rho: ArrayView1<'_, f64>,
    probes: &[Array1<f64>],
) {
    let h = DERIV_FD_STEP;
    let grad = pen.grad_target(target, rho);
    let mut tp = target.to_owned();
    let mut tm = target.to_owned();
    for i in 0..target.len() {
        tp[i] = target[i] + h;
        tm[i] = target[i] - h;
        let fd = (pen.value(tp.view(), rho) - pen.value(tm.view(), rho)) / (2.0 * h);
        tp[i] = target[i];
        tm[i] = target[i];
        assert_fd_close(&format!("{label} grad[{i}]"), grad[i], fd);
    }
    for (k, v) in probes.iter().enumerate() {
        let hv = pen.hvp(target, rho, v.view());
        let majorized = pen.psd_majorizer_hvp(target, rho, v.view());
        let step = v.mapv(|x| x * h);
        let gp = pen.grad_target((&target + &step).view(), rho);
        let gm = pen.grad_target((&target - &step).view(), rho);
        for i in 0..target.len() {
            let fd = (gp[i] - gm[i]) / (2.0 * h);
            assert_fd_close(&format!("{label} hvp[{k}][{i}]"), hv[i], fd);
            // Every penalty checked here is convex, so its PSD majorizer is the
            // exact Hessian and the two operators must agree to roundoff.
            assert_abs_diff_eq!(majorized[i], hv[i], epsilon = 1e-12 * (1.0 + hv[i].abs()));
        }
    }
    let grad_rho = pen.grad_rho(target, rho);
    assert_eq!(grad_rho.len(), pen.rho_count(), "{label} grad_rho length");
    for k in 0..rho.len() {
        let mut rp = rho.to_owned();
        let mut rm = rho.to_owned();
        rp[k] += h;
        rm[k] -= h;
        let fd = (pen.value(target, rp.view()) - pen.value(target, rm.view())) / (2.0 * h);
        assert_fd_close(&format!("{label} grad_rho[{k}]"), grad_rho[k], fd);
    }
}

/// The materialized Hessian must act exactly like the matrix-free `hvp`.
fn assert_dense_matches_hvp(
    label: &str,
    dense: &Array2<f64>,
    pen: &dyn AnalyticPenalty,
    target: ArrayView1<'_, f64>,
    rho: ArrayView1<'_, f64>,
    probes: &[Array1<f64>],
) {
    for (k, v) in probes.iter().enumerate() {
        let hv = pen.hvp(target, rho, v.view());
        let dv = dense.dot(v);
        for i in 0..v.len() {
            assert!(
                (dv[i] - hv[i]).abs() <= 1e-12 * (1.0 + hv[i].abs()),
                "{label} dense·v[{k}][{i}] = {:.12e} but hvp = {:.12e}",
                dv[i],
                hv[i]
            );
        }
    }
}

/// `ln det A` of a symmetric positive-definite matrix by an unpivoted Cholesky
/// factorization, as an independent dense reference.
fn cholesky_log_det(a: &Array2<f64>) -> f64 {
    let n = a.nrows();
    let mut l = Array2::<f64>::zeros((n, n));
    let mut log_det = 0.0;
    for j in 0..n {
        let mut pivot = a[[j, j]];
        for k in 0..j {
            pivot -= l[[j, k]] * l[[j, k]];
        }
        assert!(pivot > 0.0, "reference matrix is not positive definite");
        let ljj = pivot.sqrt();
        l[[j, j]] = ljj;
        log_det += 2.0 * ljj.ln();
        for i in j + 1..n {
            let mut s = a[[i, j]];
            for k in 0..j {
                s -= l[[i, k]] * l[[j, k]];
            }
            l[[i, j]] = s / ljj;
        }
    }
    log_det
}

/// Row-major `(5, 2)` latent block whose edge differences mix the quadratic
/// regime (|Δ| = 0.05 ≪ ε) with the L¹ regime (|Δ| up to 2.25 ≫ ε), plus two
/// dense probes.
fn five_by_two_fixture() -> (Array1<f64>, Vec<Array1<f64>>) {
    let t = array![
        0.10_f64, -0.40, 0.15, 0.90, -0.60, 0.95, -0.55, -1.30, 0.40, -1.20
    ];
    let probes = vec![
        array![1.0_f64, -0.5, 0.3, 0.8, -1.1, 0.2, 0.6, -0.4, 0.9, -0.7],
        array![-0.2_f64, 0.7, 1.0, -0.3, 0.5, 0.4, -0.8, 1.2, -0.6, 0.1],
    ];
    (t, probes)
}

#[test]
fn total_variation_forward_1d_derivatives_match_fd() {
    let (t, probes) = five_by_two_fixture();
    let rho = array![0.3_f64];
    let pen = TotalVariationPenalty::new(0.7, 5, DifferenceOpKind::ForwardDiff1D, 0.2, true)
        .expect("valid forward-difference TV");
    assert_penalty_derivatives_match_fd("TV forward", &pen, t.view(), rho.view(), &probes);

    let dense = pen.as_dense(t.view(), rho.view());
    assert_dense_matches_hvp("TV forward", &dense, &pen, t.view(), rho.view(), &probes);
    let diag = pen.diag_target(t.view(), rho.view());
    for i in 0..t.len() {
        assert_abs_diff_eq!(
            diag[i],
            dense[[i, i]],
            epsilon = 1e-12 * (1.0 + dense[[i, i]])
        );
    }
    // The tridiagonal pivot recurrence behind the frozen operator's log-det must
    // equal the dense log-det of the same Hessian, including a λ small enough
    // that the path Laplacian's constant null direction dominates.
    for &lambda in &[1.0e-3_f64, 0.5, 4.0] {
        let tridiagonal = pen
            .log_det_plus_lambda_i_forward_1d(t.view(), rho.view(), lambda)
            .expect("positive pivots");
        let mut shifted = dense.clone();
        for i in 0..t.len() {
            shifted[[i, i]] += lambda;
        }
        let reference = cholesky_log_det(&shifted);
        assert_abs_diff_eq!(
            tridiagonal,
            reference,
            epsilon = 1e-10 * (1.0 + reference.abs())
        );
    }
}

#[test]
fn total_variation_graph_edges_derivatives_match_fd() {
    let (t, probes) = five_by_two_fixture();
    let rho = array![-0.4_f64];
    let edges = vec![(0, 2), (2, 1), (3, 1), (0, 4), (4, 3)];
    let pen = TotalVariationPenalty::new(1.3, 5, DifferenceOpKind::GraphEdges(edges), 0.25, true)
        .expect("valid graph TV");
    assert_penalty_derivatives_match_fd("TV graph", &pen, t.view(), rho.view(), &probes);

    let dense = pen.as_dense(t.view(), rho.view());
    assert_dense_matches_hvp("TV graph", &dense, &pen, t.view(), rho.view(), &probes);
    let diag = pen.diag_target(t.view(), rho.view());
    for i in 0..t.len() {
        assert_abs_diff_eq!(
            diag[i],
            dense[[i, i]],
            epsilon = 1e-12 * (1.0 + dense[[i, i]])
        );
    }
}

#[test]
fn shape_monotonicity_derivatives_match_fd() {
    let (t, probes) = five_by_two_fixture();
    let rho = array![0.2_f64];
    for direction in [1.0_f64, -1.0] {
        let pen = ShapeMonotonicityPenalty::new(0.6, 5, direction, 0.25, true)
            .expect("valid monotonicity penalty");
        assert_penalty_derivatives_match_fd(
            &format!("monotonicity dir {direction}"),
            &pen,
            t.view(),
            rho.view(),
            &probes,
        );
    }
}

#[test]
fn ivae_ridge_mean_gauge_derivatives_match_fd() {
    let (t, probes) = five_by_two_fixture();
    let rho = array![0.25_f64];
    let aux = array![
        [1.0_f64, 0.2],
        [0.4, -0.7],
        [-0.3, 0.5],
        [0.8, 1.1],
        [-1.2, 0.1]
    ];
    let pen = IvaeRidgeMeanGauge::new(PsiSlice::full(10, Some(2)), aux, 0.1, 0.8, 5, true)
        .expect("valid iVAE gauge");
    // grad_rho includes the Gaussian normalizer −½·len·ln μ, so the ρ check
    // pins the normalizer's derivative as well as the quadratic's.
    assert_penalty_derivatives_match_fd("iVAE gauge", &pen, t.view(), rho.view(), &probes);

    let dense = pen.as_dense(t.view(), rho.view());
    assert_dense_matches_hvp("iVAE gauge", &dense, &pen, t.view(), rho.view(), &probes);
    let diag = pen.diag_target(t.view(), rho.view());
    for i in 0..t.len() {
        assert_abs_diff_eq!(
            diag[i],
            dense[[i, i]],
            epsilon = 1e-12 * (1.0 + dense[[i, i]].abs())
        );
    }
}

fn ncg_split(
    theta: ArrayView1<'_, f64>,
    left_rows: usize,
    right_rows: usize,
    cols: usize,
) -> (Array2<f64>, Array2<f64>) {
    let split = left_rows * cols;
    let left = Array2::from_shape_vec((left_rows, cols), theta.slice(s![..split]).to_vec())
        .expect("left block shape");
    let right = Array2::from_shape_vec((right_rows, cols), theta.slice(s![split..]).to_vec())
        .expect("right block shape");
    (left, right)
}

/// `u^p v^p = ∂φ/∂E`, recomputed from the `GramNormalization` definitions.
fn ncg_normalizer(
    left: &Array2<f64>,
    right: &Array2<f64>,
    normalization: normalized_gram::GramNormalization,
) -> f64 {
    let sum_sq = |m: &Array2<f64>| m.iter().map(|x| x * x).sum::<f64>();
    match normalization {
        normalized_gram::GramNormalization::DecoderNorm => 1.0 / (sum_sq(left) * sum_sq(right)),
        normalized_gram::GramNormalization::SelfGramNorm => {
            let left_gram = left.dot(&left.t());
            let right_gram = right.dot(&right.t());
            1.0 / (sum_sq(&left_gram).sqrt() * sum_sq(&right_gram).sqrt())
        }
    }
}

fn assert_normalized_cross_gram_derivatives_match_fd(
    normalization: normalized_gram::GramNormalization,
) {
    let (left_rows, right_rows, cols) = (2, 3, 3);
    let theta = array![
        0.9_f64, -0.3, 0.5, 0.2, 1.1, -0.6, 0.4, 0.8, -0.2, -0.7, 0.3, 0.9, 0.6, -0.5, 1.0
    ];
    let l_dir = array![
        0.3_f64, 1.0, -0.4, 0.7, -0.2, 0.5, -0.9, 0.1, 0.6, 0.4, -0.8, 0.2, -0.3, 0.9, 0.5
    ];
    let r_dir = array![
        -0.6_f64, 0.2, 0.8, -0.1, 0.4, 1.1, 0.3, -0.7, 0.5, 0.9, 0.2, -0.4, 0.7, 0.1, -0.5
    ];
    let dim = theta.len();
    let h = DERIV_FD_STEP;
    let label = format!("{normalization:?}");
    let at = |t: &Array1<f64>| {
        let (left, right) = ncg_split(t.view(), left_rows, right_rows, cols);
        normalized_gram::NormalizedCrossGram::new(left.view(), right.view(), normalization)
            .expect("positive normalizers")
    };
    let value = |t: &Array1<f64>| {
        let (left, right) = ncg_split(t.view(), left_rows, right_rows, cols);
        let cross = left.dot(&right.t());
        cross.iter().map(|x| x * x).sum::<f64>() * ncg_normalizer(&left, &right, normalization)
    };
    // Frozen-direction Gauss–Newton form 2·u^p v^p·⟨C_l, C_r⟩ with
    // C_l = l_X Yᵀ + X l_Yᵀ; its θ-gradient is what the GN bilinear returns.
    let gn_form = |t: &Array1<f64>| {
        let (x, y) = ncg_split(t.view(), left_rows, right_rows, cols);
        let (lx, ly) = ncg_split(l_dir.view(), left_rows, right_rows, cols);
        let (rx, ry) = ncg_split(r_dir.view(), left_rows, right_rows, cols);
        let cl = lx.dot(&y.t()) + x.dot(&ly.t());
        let cr = rx.dot(&y.t()) + x.dot(&ry.t());
        2.0 * ncg_normalizer(&x, &y, normalization) * (&cl * &cr).sum()
    };
    let bilinear_hessian = |t: &Array1<f64>| l_dir.dot(&at(t).hessian_action(r_dir.view()));
    let coordinate_fd = |f: &dyn Fn(&Array1<f64>) -> f64, i: usize| {
        let mut tp = theta.clone();
        let mut tm = theta.clone();
        tp[i] += h;
        tm[i] -= h;
        (f(&tp) - f(&tm)) / (2.0 * h)
    };

    let base = at(&theta);
    let grad = base.gradient();
    let third = base.third_bilinear(l_dir.view(), r_dir.view());
    let gn = base.gauss_newton_bilinear_gradient(l_dir.view(), r_dir.view());
    let diag = base.diagonal();
    for i in 0..dim {
        assert_fd_close(
            &format!("{label} gradient[{i}]"),
            grad[i],
            coordinate_fd(&value, i),
        );
        assert_fd_close(
            &format!("{label} third_bilinear[{i}]"),
            third[i],
            coordinate_fd(&bilinear_hessian, i),
        );
        assert_fd_close(
            &format!("{label} gauss_newton_bilinear_gradient[{i}]"),
            gn[i],
            coordinate_fd(&gn_form, i),
        );
        let mut e = Array1::<f64>::zeros(dim);
        e[i] = 1.0;
        let column = base.hessian_action(e.view());
        assert_abs_diff_eq!(
            diag[i],
            column[i],
            epsilon = 1e-12 * (1.0 + column[i].abs())
        );
    }
    for (k, v) in [l_dir.view(), r_dir.view()].into_iter().enumerate() {
        let hv = base.hessian_action(v);
        let step = v.mapv(|x| x * h);
        let gp = at(&(&theta + &step)).gradient();
        let gm = at(&(&theta - &step)).gradient();
        for i in 0..dim {
            assert_fd_close(
                &format!("{label} hessian_action[{k}][{i}]"),
                hv[i],
                (gp[i] - gm[i]) / (2.0 * h),
            );
        }
    }
}

#[test]
fn normalized_cross_gram_decoder_norm_derivatives_match_fd() {
    assert_normalized_cross_gram_derivatives_match_fd(
        normalized_gram::GramNormalization::DecoderNorm,
    );
}

#[test]
fn normalized_cross_gram_self_gram_norm_derivatives_match_fd() {
    assert_normalized_cross_gram_derivatives_match_fd(
        normalized_gram::GramNormalization::SelfGramNorm,
    );
}

#[test]
fn block_sparsity_derivatives_match_fd() {
    // Row-major (4, 3) block. Group {0, 2} sits in the L² regime (norm ≫ ε),
    // where the rank-one −x xᵀ/s³ correction nearly cancels I/s along x;
    // group {1} has entries ≪ ε, so its smoothed norm stays in the quadratic
    // regime s ≈ ε.
    let t = array![
        0.8_f64, 0.02, -0.5, -0.3, -0.03, 1.1, 0.6, 0.01, 0.2, -0.9, 0.04, -0.4
    ];
    let probes = vec![
        array![0.5_f64, -1.0, 0.3, 0.7, 0.4, -0.6, -0.2, 0.9, 1.1, 0.3, -0.5, 0.8],
        array![-0.7_f64, 0.2, 1.0, -0.4, 0.6, 0.1, 0.9, -0.3, -0.8, 0.5, 0.7, -0.2],
    ];
    let rho = array![0.15_f64];
    let pen = BlockSparsityPenalty::new(
        PsiSlice::full(12, Some(3)),
        vec![vec![0, 2], vec![1]],
        0.9,
        4,
        0.3,
        true,
    )
    .expect("valid block sparsity");
    assert_penalty_derivatives_match_fd("block sparsity", &pen, t.view(), rho.view(), &probes);

    let dense = pen.as_dense(t.view(), rho.view());
    assert_dense_matches_hvp("block sparsity", &dense, &pen, t.view(), rho.view(), &probes);
    let diag = pen.diag_target(t.view(), rho.view());
    for i in 0..t.len() {
        assert_abs_diff_eq!(
            diag[i],
            dense[[i, i]],
            epsilon = 1e-12 * (1.0 + dense[[i, i]].abs())
        );
    }
}

#[test]
fn row_precision_prior_derivatives_match_fd() {
    let (t, probes) = five_by_two_fixture();
    let rho = array![-0.35_f64];
    // One symmetric positive-definite precision per row, with off-diagonal
    // coupling so the Hessian is not diagonal.
    let lambda = array![
        [[2.0_f64, 0.6], [0.6, 1.0]],
        [[1.5, -0.4], [-0.4, 0.9]],
        [[0.7, 0.2], [0.2, 2.2]],
        [[3.0, 1.1], [1.1, 1.4]],
        [[1.2, -0.5], [-0.5, 0.8]]
    ];
    let pen = RowPrecisionPriorPenalty::new(PsiSlice::full(10, Some(2)), lambda, 1.4, 5, true)
        .expect("valid row-precision prior");
    // grad_rho carries the normalizer derivative −½·len alongside ½μ·tᵀΛt.
    assert_penalty_derivatives_match_fd("row precision", &pen, t.view(), rho.view(), &probes);
    assert!(
        pen.hessian_diag(t.view(), rho.view()).is_none(),
        "a coupled per-row precision has no diagonal Hessian"
    );
    let dense = pen.as_dense(t.view(), rho.view());
    assert_dense_matches_hvp("row precision", &dense, &pen, t.view(), rho.view(), &probes);
}

/// ½·(μ/n)·‖TᵀT − I‖²_F is quartic and nonconvex, so it is checked outside the
/// convex helper: value→`grad_target`, `grad_target`→`hvp`, the materialized
/// Hessian against `hvp`, and value→`grad_rho`.
#[test]
fn orthogonality_derivatives_match_fd() {
    let (t, probes) = five_by_two_fixture();
    let rho = array![-0.2_f64];
    let pen = OrthogonalityPenalty::new(PsiSlice::full(10, Some(2)), 2, 1.1, 5, true)
        .expect("valid orthogonality penalty");
    let h = DERIV_FD_STEP;
    let grad = pen.grad_target(t.view(), rho.view());
    for i in 0..t.len() {
        let mut tp = t.clone();
        let mut tm = t.clone();
        tp[i] += h;
        tm[i] -= h;
        let fd = (pen.value(tp.view(), rho.view()) - pen.value(tm.view(), rho.view())) / (2.0 * h);
        assert_fd_close(&format!("orthogonality grad[{i}]"), grad[i], fd);
    }
    for (k, v) in probes.iter().enumerate() {
        let hv = pen.hvp(t.view(), rho.view(), v.view());
        let step = v.mapv(|x| x * h);
        let gp = pen.grad_target((&t + &step).view(), rho.view());
        let gm = pen.grad_target((&t - &step).view(), rho.view());
        for i in 0..t.len() {
            assert_fd_close(
                &format!("orthogonality hvp[{k}][{i}]"),
                hv[i],
                (gp[i] - gm[i]) / (2.0 * h),
            );
        }
    }
    let t_mat = pen.target_matrix(t.view()).expect("(5, 2) target");
    let m = OrthogonalityPenalty::gram_minus_identity(t_mat);
    let dense = pen.as_dense_with_precomputed_m(t_mat, m.view(), pen.scale(rho.view()));
    assert_dense_matches_hvp("orthogonality", &dense, &pen, t.view(), rho.view(), &probes);
    let grad_rho = pen.grad_rho(t.view(), rho.view());
    let mut rp = rho.clone();
    let mut rm = rho.clone();
    rp[0] += h;
    rm[0] -= h;
    let fd = (pen.value(t.view(), rp.view()) - pen.value(t.view(), rm.view())) / (2.0 * h);
    assert_fd_close("orthogonality grad_rho", grad_rho[0], fd);
}

/// `FrozenAnalyticPenaltyOp` is one PSD curvature operator: `matvec` applies the
/// penalty's PSD majorizer, so `diag`, `as_dense` and `log det(S + λI)` must read
/// that same operator. Isometry is nonconvex and its majorizer is the Gauss-Newton
/// block `B_GN`, while its cached HVP state builds the exact Hessian, which also
/// carries the residual and third-jet terms and is indefinite. The fixture installs
/// the third jet so the exact Hessian is available and differs from `B_GN`; every
/// frozen query must still return `B_GN`, the matrix whose columns are `matvec`
/// of the unit vectors.
///
/// Tolerances: `diag` and `as_dense` probe the same `psd_majorizer_hvp` as
/// `matvec` on the same inputs, so they agree to rounding (a relative
/// `1e-12` of the largest entry). The log-determinant is an eigensolve of a
/// 6 × 6 symmetric matrix with `λ = 0.3` added. Its eigenvalues are therefore at
/// least 0.3, each `log` term carries a relative error of about `n·ε / 0.3`, and
/// `1e-10` of `max(|log det|, 1)` leaves several orders of margin.
#[test]
fn frozen_isometry_operator_reads_the_gauss_newton_majorizer_everywhere() {
    let (n_obs, p, d, j, h) = isometry_gn_fixture();
    let n = n_obs * d;
    let pen = IsometryPenalty::new_euclidean(PsiSlice::full(n, Some(d)), p);
    pen.refresh_caches(Some(j), Some(h));
    let k = ndarray::Array3::<f64>::from_shape_fn((n_obs, p, d * d * d), |(row, output, axes)| {
        let (a, c, e) = (axes / (d * d), (axes / d) % d, axes % d);
        0.11 * (row as f64 + 1.0) - 0.07 * (output as f64) + 0.05 * ((a + c + e) as f64)
    });
    pen.set_third_decoder_derivative(Some(Arc::new(k)));
    let kind = AnalyticPenaltyKind::Isometry(Arc::new(pen));
    let rho = array![0.0_f64];
    let t = Array1::<f64>::zeros(n);

    let mut majorizer = Array2::<f64>::zeros((n, n));
    let mut exact = Array2::<f64>::zeros((n, n));
    for col in 0..n {
        let mut e = Array1::<f64>::zeros(n);
        e[col] = 1.0;
        let b = kind.psd_majorizer_hvp(t.view(), rho.view(), e.view());
        let hv = kind.hvp(t.view(), rho.view(), e.view());
        for row in 0..n {
            majorizer[[row, col]] = b[row];
            exact[[row, col]] = hv[row];
        }
    }
    let scale = majorizer.iter().fold(0.0_f64, |acc, &v| acc.max(v.abs()));
    let gap = (&exact - &majorizer).iter().fold(0.0_f64, |acc, &v| acc.max(v.abs()));
    assert!(scale > 0.0, "the Gauss-Newton majorizer must be nonzero on this fixture");
    assert!(
        gap > 1e-6 * scale,
        "the fixture must separate the exact Hessian from B_GN; max gap {gap:.3e}"
    );

    let op = FrozenAnalyticPenaltyOp::new(kind, t, rho).expect("frozen isometry operator");
    let mut applied = Array2::<f64>::zeros((n, n));
    for col in 0..n {
        let mut e = Array1::<f64>::zeros(n);
        e[col] = 1.0;
        let mut out = Array1::<f64>::zeros(n);
        op.matvec(e.view(), out.view_mut());
        for row in 0..n {
            applied[[row, col]] = out[row];
        }
    }
    let dense = op.as_dense();
    let diag = op.diag();
    for row in 0..n {
        assert_abs_diff_eq!(diag[row], majorizer[[row, row]], epsilon = 1e-12 * scale);
        for col in 0..n {
            assert_abs_diff_eq!(applied[[row, col]], majorizer[[row, col]], epsilon = 1e-12 * scale);
            assert_abs_diff_eq!(dense[[row, col]], majorizer[[row, col]], epsilon = 1e-12 * scale);
        }
    }

    let lambda = 0.3;
    let expected = <Array2<f64> as PenaltyOp>::log_det_plus_lambda_i(&majorizer, lambda)
        .expect("log det of the Gauss-Newton majorizer");
    let log_det = op.log_det_plus_lambda_i(lambda).expect("frozen isometry log det");
    assert_abs_diff_eq!(log_det, expected, epsilon = 1e-10 * expected.abs().max(1.0));
}

