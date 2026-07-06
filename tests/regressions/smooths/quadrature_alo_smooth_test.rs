use gam::inference::alo::{AloInput, compute_alo_from_input};
use gam::inference::quadrature::{
    QuadratureContext, cloglog_ghq_value, integrated_family_moments_jet,
};
use gam::inference::smooth_test::{SmoothTestInput, SmoothTestScale, wood_smooth_test};
use gam::matrix::{PsdWeightsView, SignedWeightsView};
use gam::types::{
    InverseLink, LikelihoodScaleMetadata, LikelihoodSpec, LinkFunction, ResponseFamily,
    StandardLink,
};
use ndarray::{Array1, Array2, array};
use rand::{RngExt, SeedableRng, rngs::StdRng};
use statrs::distribution::{ContinuousCDF, FisherSnedecor};

#[test]
fn integrated_family_moments_jet_matches_lognormal_mean_for_poisson_log_link() {
    let mut rng = StdRng::seed_from_u64(42);
    let ctx = QuadratureContext::new();
    let spec = LikelihoodSpec::new(
        ResponseFamily::Poisson,
        InverseLink::Standard(StandardLink::Log),
    );

    for _ in 0..32 {
        let eta = rng.random_range(-2.0..2.0);
        let sigma = rng.random_range(0.01..1.2);
        let got = integrated_family_moments_jet(
            &ctx,
            &spec,
            LikelihoodScaleMetadata::FixedDispersion { phi: 1.0 },
            eta,
            sigma,
        )
        .expect("integrated family moments should evaluate for finite eta and sigma")
        .mean;
        let expected = (eta + 0.5 * sigma * sigma).exp();
        let rel_err = ((got - expected) / expected).abs();
        assert!(
            rel_err < 2e-8,
            "Poisson-log integrated mean should match exp(eta + sigma^2/2) within GHQ error; got rel_err={rel_err:.3e}, eta={eta:.4}, sigma={sigma:.4}"
        );
    }
}

/// Independent brute-force reference for `L(η,σ) = ∫ (1 − exp(−exp(η+σz))) φ(z) dz`
/// via composite Simpson on a fine grid over [−12, 12] (φ tails below 1e-30
/// beyond). This does NOT share GHQ nodes/weights with the routine under test,
/// so matching it certifies genuine resolution — unlike an order-doubling check
/// the routine can trivially satisfy by clamping both requests to one order
/// (the #2063 anti-pattern).
fn cloglog_convolution_reference(eta: f64, sigma: f64) -> f64 {
    let n = 200_000usize;
    let (lo, hi) = (-12.0_f64, 12.0_f64);
    let h = (hi - lo) / n as f64;
    let inv_sqrt_2pi = 1.0 / (2.0 * std::f64::consts::PI).sqrt();
    let mut sum = 0.0_f64;
    for i in 0..=n {
        let z = lo + i as f64 * h;
        let g = 1.0 - (-(eta + sigma * z).exp()).exp();
        let phi = (-0.5 * z * z).exp() * inv_sqrt_2pi;
        let w = if i == 0 || i == n {
            1.0
        } else if i % 2 == 1 {
            4.0
        } else {
            2.0
        };
        sum += w * g * phi;
    }
    (sum * h / 3.0).clamp(0.0, 1.0)
}

#[test]
fn quadrature_order_doubling_stabilizes_cloglog_integral() {
    let mut rng = StdRng::seed_from_u64(7);
    let ctx = QuadratureContext::new();

    for _ in 0..32 {
        let eta = rng.random_range(-1.5..1.5);
        let sigma = rng.random_range(0.1..1.0);
        let q = cloglog_ghq_value(&ctx, eta, sigma, 15);
        let q2 = cloglog_ghq_value(&ctx, eta, sigma, 31);

        // (a) Order-doubling stability: two different requested orders must land
        // on the same value (each is internally escalated to convergence).
        let diff = (q2 - q).abs();
        assert!(
            diff < 1e-8,
            "Doubling GHQ order should stabilize cloglog integral; |I_31-I_15|={diff:.3e}, eta={eta:.4}, sigma={sigma:.4}"
        );

        // (b) Non-vacuous certification (#2063): the converged value must match
        // an INDEPENDENT high-resolution reference, proving genuine resolution
        // rather than two requests being clamped to one internal order.
        let reference = cloglog_convolution_reference(eta, sigma);
        let ref_err = (q - reference).abs();
        assert!(
            ref_err < 1e-7,
            "cloglog GHQ value must match the brute-force convolution reference; \
             |I_15-ref|={ref_err:.3e}, I_15={q:.12}, ref={reference:.12}, eta={eta:.4}, sigma={sigma:.4}"
        );
    }
}

#[test]
fn alo_residual_matches_closed_form_identity_link() {
    let x = array![[1.0, 0.0], [1.0, 1.0], [1.0, 2.0], [1.0, 3.0]];
    let y = array![0.9, 2.1, 2.8, 4.2];
    let eta = array![1.0, 2.0, 3.0, 4.0];
    let w = Array1::from_elem(4, 1.0);
    let z = y.clone();
    let offset = Array1::zeros(4);
    let h = x.t().dot(&x);

    let input = AloInput {
        design: &x,
        penalized_hessian: &h,
        hessian_weights: SignedWeightsView::from_array(&w),
        score_weights: PsdWeightsView::try_from_array(&w).expect("psd weights"),
        working_response: &z,
        eta: &eta,
        offset: &offset,
        link: LinkFunction::Identity,
        phi: 1.0,
        penalty_root: None,
        ridge: 0.0,
        score_curvature: None,
    };

    let out = compute_alo_from_input(&input)
        .expect("ALO should compute for full-rank identity-link setup");
    for i in 0..y.len() {
        let h_ii = out.leverage[i];
        let expected = (y[i] - eta[i]) / (1.0 - h_ii);
        let got = y[i] - out.eta_tilde[i];
        assert!(
            (got - expected).abs() < 1e-10,
            "ALO residual should match (y-mu)/(1-h_ii) for identity link; row={i}, got={got:.12}, expected={expected:.12}"
        );
    }
}

#[test]
fn alo_respects_family_and_robust_weights() {
    let x = array![[1.0], [1.0], [1.0], [1.0]];
    let y = array![1.0, 2.0, 3.0, 4.0];
    let eta = array![1.5, 1.5, 3.5, 3.5];
    let w_h = array![1.0, 2.0, 0.5, 3.0];
    let w_s = array![0.8, 1.5, 0.4, 2.5];
    let z = y.clone();
    let offset = Array1::zeros(4);
    let h = array![[w_h.sum()]];

    let input = AloInput {
        design: &x,
        penalized_hessian: &h,
        hessian_weights: SignedWeightsView::from_array(&w_h),
        score_weights: PsdWeightsView::try_from_array(&w_s).expect("psd weights"),
        working_response: &z,
        eta: &eta,
        offset: &offset,
        link: LinkFunction::Identity,
        phi: 1.0,
        penalty_root: None,
        ridge: 0.0,
        score_curvature: None,
    };
    let out = compute_alo_from_input(&input)
        .expect("ALO should compute when family and robust weights are finite and positive");

    for i in 0..4 {
        let x_hinv_x = 1.0 / w_h.sum();
        let h_ii = w_h[i] * x_hinv_x;
        let expected = eta[i] + x_hinv_x * w_s[i] * (eta[i] - z[i]) / (1.0 - h_ii);
        assert!(
            (out.eta_tilde[i] - expected).abs() < 1e-10,
            "ALO eta_tilde should use hessian weights in leverage and score weights in correction; row={i}, got={:.12}, expected={expected:.12}",
            out.eta_tilde[i],
        );
    }
}

#[test]
fn smooth_test_estimated_scale_pvalue_matches_f_distribution_formula() {
    let beta = array![0.0, 0.4, -0.3, 0.25];
    let covariance = Array2::from_diag(&Array1::from(vec![1.0, 0.8, 0.6, 1.2]));
    let residual_df = 120.0;

    let out = wood_smooth_test(SmoothTestInput {
        beta: beta.view(),
        covariance: &covariance,
        influence_matrix: None,
        whitening_gram: None,
        coeff_range: 1..4,
        edf: 3.0,
        nullspace_dim: 0,
        residual_df,
        scale: SmoothTestScale::Estimated,
    })
    .expect("Wood smooth test should return a finite statistic and p-value for positive-definite covariance");

    // `covariance` is scale-included, so `wood_smooth_test` returns the proper
    // Wald χ² (dispersion already divided out by the covariance scale). The
    // Estimated-scale F-statistic is therefore `T / ref_df` with no further
    // `φ̂` factor.
    let f_stat = out.statistic / out.ref_df;
    let dist = FisherSnedecor::new(out.ref_df, residual_df)
        .expect("F distribution parameters should be valid for positive reference and residual df");
    let expected_p = 1.0 - dist.cdf(f_stat);
    assert!(
        (out.p_value - expected_p).abs() < 1e-12,
        "Estimated-scale Wood p-value should match F-tail formula; got={:.15}, expected={expected_p:.15}",
        out.p_value,
    );
}

#[test]
fn smooth_test_scale_has_consistent_alpha_rejection_ordering() {
    let beta = array![0.0, 0.7, -0.4, 0.3];
    let covariance = Array2::from_diag(&Array1::from(vec![1.0, 0.5, 0.7, 0.9]));

    let out = wood_smooth_test(SmoothTestInput {
        beta: beta.view(),
        covariance: &covariance,
        influence_matrix: None,
        whitening_gram: None,
        coeff_range: 1..4,
        edf: 3.0,
        nullspace_dim: 0,
        residual_df: 80.0,
        scale: SmoothTestScale::Estimated,
    })
    .expect("Wood smooth test should produce a p-value for valid estimated-scale inputs");

    let reject_at_001 = out.p_value < 0.01;
    let reject_at_005 = out.p_value < 0.05;
    let reject_at_010 = out.p_value < 0.10;

    assert!(
        (!reject_at_001 || reject_at_005) && (!reject_at_005 || reject_at_010),
        "Rejection regions must be nested across alpha levels: reject@0.01 implies reject@0.05 implies reject@0.10"
    );
}
