use gam::inference::quadrature::{QuadratureContext, integrated_family_moments_jet};
use gam::inference::smooth_test::{SmoothTestInput, SmoothTestScale, wood_smooth_test};
use gam::types::{GlmLikelihoodSpec, InverseLink, LikelihoodSpec, ResponseFamily, StandardLink};
use ndarray::{Array1, Array2, array};
use rand::{RngExt, SeedableRng, rngs::StdRng};
use statrs::distribution::{ContinuousCDF, FisherSnedecor};

#[test]
fn integrated_family_moments_jet_matches_lognormal_mean_for_poisson_log_link() {
    let mut rng = StdRng::seed_from_u64(42);
    let ctx = QuadratureContext::new();
    let likelihood = GlmLikelihoodSpec::canonical(LikelihoodSpec::new(
        ResponseFamily::Poisson,
        InverseLink::Standard(StandardLink::Log),
    ));

    for _ in 0..32 {
        let eta = rng.random_range(-2.0..2.0);
        let sigma = rng.random_range(0.01..1.2);
        let got = integrated_family_moments_jet(&ctx, &likelihood, eta, sigma)
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
        residual_df: Some(residual_df),
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
        residual_df: Some(80.0),
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
