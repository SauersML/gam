use gam::solver::pirls::{calculate_deviance, update_glmvectors_by_family};
use gam::types::{
    GlmLikelihoodSpec, InverseLink, LikelihoodScaleMetadata, LikelihoodSpec, ResponseFamily,
};
use ndarray::array;

#[test]
fn gaussian_fixed_dispersion_should_scale_working_weight_and_deviance() {
    let likelihood = GlmLikelihoodSpec {
        spec: LikelihoodSpec {
            response: ResponseFamily::Gaussian,
            link: InverseLink::Standard(gam::types::StandardLink::Identity),
        },
        scale: LikelihoodScaleMetadata::FixedDispersion { phi: 4.0 },
    };

    let y = array![3.0];
    let eta = array![1.0];
    let prior = array![2.0];
    let mut mu = array![0.0];
    let mut w = array![0.0];
    let mut z = array![0.0];

    update_glmvectors_by_family(
        y.view(),
        &eta,
        &likelihood,
        prior.view(),
        &mut mu,
        &mut w,
        &mut z,
    )
    .expect("Gaussian IRLS update should succeed");

    let expected_weight = prior[0] / 4.0;
    assert!(
        (w[0] - expected_weight).abs() < 1e-12,
        "Gaussian working weight must include fixed dispersion phi (expected {}, got {})",
        expected_weight,
        w[0]
    );

    let dev = calculate_deviance(y.view(), &mu, &likelihood, prior.view());
    let expected_dev = prior[0] * (y[0] - mu[0]).powi(2) / 4.0;
    assert!(
        (dev - expected_dev).abs() < 1e-12,
        "Gaussian deviance must divide weighted RSS by fixed dispersion phi (expected {}, got {})",
        expected_dev,
        dev
    );
}
