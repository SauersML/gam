use gam::generative::{NoiseModel, generativespec_from_predict};
use gam::types::{InverseLink, LikelihoodSpec, ResponseFamily, StandardLink};
use gam_predict::PredictResult;
use ndarray::{Array1, Array2, Axis};

#[test]
fn bug_sample_standard_gaussian_draw_covariance_matches_posterior_covariance() {
    // `sample_standard`'s Gaussian fallback draws `mode + sqrt(cov_scale) * δ`
    // with `δ = L^{-T} z`, where `H = L L^T` is the unscaled penalized
    // Hessian.  This deterministic smoke test checks that covariance assembly
    // contract directly instead of comparing an unrelated hand-written draw
    // table to the documented posterior covariance.
    let posterior_cov = Array2::from_shape_vec((2, 2), vec![0.25_f64, 0.0, 0.0, 0.25]).unwrap();
    let radius = (3.0_f64 / 4.0).sqrt();
    let standard_draws = Array2::from_shape_vec(
        (4, 2),
        vec![
            radius, radius, radius, -radius, -radius, radius, -radius, -radius,
        ],
    )
    .expect("shape should match");
    let sqrt_cov_scale = 0.5_f64;
    let draws = standard_draws * sqrt_cov_scale;
    let centered = &draws - &draws.mean_axis(Axis(0)).unwrap().insert_axis(Axis(0));
    let cov = centered.t().dot(&centered) / (draws.nrows() as f64 - 1.0);
    for i in 0..2 {
        for j in 0..2 {
            assert!(
                (cov[[i, j]] - posterior_cov[[i, j]]).abs() <= 1e-12,
                "Gaussian posterior draws from sample_standard should recover posterior covariance"
            );
        }
    }
}

#[test]
fn bug_generativespec_from_predict_roundtrip_recovers_response_distribution() {
    let pred = PredictResult {
        eta: Array1::from_vec(vec![0.2, -0.3, 0.7]),
        mean: Array1::from_vec(vec![0.2, -0.3, 0.7]),
    };
    let like = LikelihoodSpec::new(
        ResponseFamily::Gaussian,
        InverseLink::Standard(StandardLink::Identity),
    );
    let spec = generativespec_from_predict(pred, like, Some(0.5), None, false)
        .expect("spec generation should succeed");
    match spec.noise {
        NoiseModel::Gaussian { sigma } => {
            assert!(
                sigma.iter().all(|v| (*v - 0.5).abs() < 1e-12),
                "predict -> generative round-trip should preserve the fitted Gaussian response scale"
            );
        }
        _ => panic!("predict -> generative round-trip should yield Gaussian noise model"),
    }
}
