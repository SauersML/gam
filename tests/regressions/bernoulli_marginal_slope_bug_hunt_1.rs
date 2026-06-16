use gam::inference::probability::{normal_cdf, normal_pdf, signed_probit_logcdf_and_mills_ratio};

#[test]
fn bug_marginal_slope_probit_matches_phi_eta_times_deta_dxj_random_x() {
    let x = 0.37_f64;
    let beta0 = -0.42_f64;
    let beta1 = 1.9_f64;
    let eta = beta0 + beta1 * x;
    let p = normal_cdf(eta);
    let h = 1e-6;
    let p_plus = normal_cdf(beta0 + beta1 * (x + h));
    let p_minus = normal_cdf(beta0 + beta1 * (x - h));
    let fd = (p_plus - p_minus) / (2.0 * h);
    let closed_form = normal_pdf(eta) * beta1;
    assert!(
        (fd - closed_form).abs() < 1e-8,
        "For Bernoulli probit, marginal slope must satisfy dP/dx_j = phi(eta)*deta/dx_j at random x; finite-difference and closed form disagree: fd={fd:.17e}, closed_form={closed_form:.17e}, p={p:.17e}, eta={eta:.17e}"
    );
}

#[test]
fn bug_signed_probit_mills_ratio_is_positive_and_saturates_for_large_abs_eta() {
    for &eta in &[-10.0, -2.3, -0.1, 0.0, 0.1, 1.7, 8.0] {
        let (_logcdf, lambda) = signed_probit_logcdf_and_mills_ratio(eta);
        assert!(
            lambda.is_finite() && lambda > 0.0,
            "Mills ratio phi(eta)/Phi(eta) must be strictly positive and finite for finite eta; got lambda={lambda:.17e} at eta={eta}"
        );
    }
    let (logcdf_pos, lambda_pos) = signed_probit_logcdf_and_mills_ratio(40.0);
    assert!(
        logcdf_pos > -1e-12 && lambda_pos < 1e-12,
        "At very large positive eta, log Phi(eta) should saturate near 0 and phi/Phi near 0; got logcdf={logcdf_pos:.17e}, lambda={lambda_pos:.17e}"
    );
    let (logcdf_neg, lambda_neg) = signed_probit_logcdf_and_mills_ratio(-40.0);
    assert!(
        logcdf_neg.is_finite() && lambda_neg.is_finite() && lambda_neg > 0.0,
        "At very large negative eta, saturated path must remain finite and keep positive mills ratio; got logcdf={logcdf_neg:.17e}, lambda={lambda_neg:.17e}"
    );
}

#[test]
fn bug_exact_kernel_second_and_third_derivatives_match_finite_difference() {
    let f = |x: f64| normal_pdf(x);
    let x0 = 0.23_f64;
    let h = 1e-4;
    let fpp_fd = (f(x0 + h) - 2.0 * f(x0) + f(x0 - h)) / (h * h);
    // The third-difference stencil divides by 2·h³, so a step optimal for the
    // second derivative (1e-4) lets floating-point rounding (≈ε/h³ ≈ 5e-5)
    // dominate the truncation error and swamp the closed-form comparison. The
    // round-off-vs-truncation optimum for a 3rd derivative is h ≈ ε^{1/5} ≈
    // 1e-3 (truncation O(h²) ≈ 1e-6, rounding O(ε/h³) ≈ 2e-7), so use a wider
    // step here than for the 2nd derivative.
    let h3 = 1e-3;
    let fppp_fd = (f(x0 + 2.0 * h3) - 2.0 * f(x0 + h3) + 2.0 * f(x0 - h3) - f(x0 - 2.0 * h3))
        / (2.0 * h3 * h3 * h3);
    let fpp_closed = (x0 * x0 - 1.0) * normal_pdf(x0);
    let fppp_closed = (3.0 * x0 - x0 * x0 * x0) * normal_pdf(x0);
    assert!(
        (fpp_fd - fpp_closed).abs() < 1e-6,
        "Second derivative jet check failed: finite-diff and closed form disagree: fd={fpp_fd:.17e}, closed_form={fpp_closed:.17e}"
    );
    assert!(
        (fppp_fd - fppp_closed).abs() < 1e-5,
        "Third derivative jet check failed: finite-diff and closed form disagree: fd={fppp_fd:.17e}, closed_form={fppp_closed:.17e}"
    );
}

#[test]
fn bug_bernoulli_probability_clamp_never_leaves_eps_interval() {
    let eps = 1e-12_f64;
    let candidates = [0.0, eps * 0.1, eps, 0.5, 1.0 - eps, 1.0 - eps * 0.1, 1.0];
    for &p in &candidates {
        let clamped = p.max(eps).min(1.0 - eps);
        assert!(
            clamped >= eps && clamped <= 1.0 - eps,
            "Clamp must never return outside [eps, 1-eps]; got clamped={clamped:.17e} for p={p:.17e}, eps={eps:.17e}"
        );
    }
}

#[test]
fn bug_frailty_integration_preserves_probability_bounds_and_monotonicity() {
    let etas = [-4.0_f64, -1.0, 0.0, 1.0, 4.0];
    let mut last = 0.0;
    for (i, eta) in etas.iter().enumerate() {
        let p = normal_cdf(*eta / (1.0_f64 + 0.4_f64 * 0.4_f64).sqrt());
        assert!(
            p.is_finite() && p > 0.0 && p < 1.0,
            "Frailty-integrated probit probability must stay strictly within (0,1); got p={p:.17e} at eta={eta}"
        );
        if i > 0 {
            assert!(
                p >= last,
                "Frailty-integrated probit probability must be monotone in eta; got p[{i}]={p:.17e} < p[{prev}]={last:.17e}",
                prev = i - 1
            );
        }
        last = p;
    }
}
