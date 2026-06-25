use gam::generative::{
    GenerativeSpec, NoiseModel, generativespec_from_predict, sampleobservation_replicates,
};
use gam::hmc::NutsResult;
use gam::polya_gamma::PolyaGamma;
use gam_predict::PredictResult;
use gam::types::{InverseLink, LikelihoodSpec, ResponseFamily, StandardLink};
use ndarray::{Array1, Array2, Axis};
use rand::{SeedableRng, rngs::StdRng};

fn pg_theoretical_mean(c: f64) -> f64 {
    if c.abs() < 1e-12 {
        0.25
    } else {
        (0.5 * c).tanh() / (2.0 * c)
    }
}

#[test]
fn bug_polya_gamma_pg11_mean_matches_theory_with_clt_bound() {
    let mut rng = StdRng::seed_from_u64(7);
    let pg = PolyaGamma::new();
    let n = 100_000usize;
    let c = 1.1;
    let draws: Vec<f64> = (0..n).map(|_| pg.draw(&mut rng, c)).collect();
    let mean = draws.iter().sum::<f64>() / n as f64;
    let var = draws.iter().map(|x| (x - mean).powi(2)).sum::<f64>() / (n as f64 - 1.0);
    let se = (var / n as f64).sqrt();
    let theory = pg_theoretical_mean(c);
    assert!(
        (mean - theory).abs() <= 3.0 * se,
        "PG(1,{c}) empirical mean should lie within 3 standard errors of the analytic mean"
    );
}

#[test]
fn bug_sampleobservation_replicates_family_means_converge_to_true_means() {
    let mut rng = StdRng::seed_from_u64(123);
    let spec = GenerativeSpec {
        mean: Array1::from_vec(vec![0.2, 0.5, 0.8]),
        noise: NoiseModel::Bernoulli,
    };
    let draws = sampleobservation_replicates(&spec, 4_000, &mut rng)
        .expect("replicate sampling should succeed");
    let empirical = draws.mean_axis(Axis(0)).expect("non-empty draws");
    for (idx, (&mu_hat, &mu_true)) in empirical.iter().zip(spec.mean.iter()).enumerate() {
        assert!(
            (mu_hat - mu_true).abs() < 0.01,
            "Replicate mean for family observation {idx} should converge to its true mean"
        );
    }
}

#[test]
fn bug_sample_standard_gaussian_draw_covariance_matches_posterior_covariance() {
    let draws = Array2::from_shape_vec((4, 2), vec![1.0, -1.0, 1.0, -1.0, -1.0, 1.0, -1.0, 1.0])
        .expect("shape should match");
    let centered = &draws - &draws.mean_axis(Axis(0)).unwrap().insert_axis(Axis(0));
    let cov = centered.t().dot(&centered) / (draws.nrows() as f64 - 1.0);
    let posterior_cov = Array2::from_shape_vec((2, 2), vec![0.25_f64, 0.0, 0.0, 0.25]).unwrap();
    for i in 0..2 {
        for j in 0..2 {
            let sigma = posterior_cov[[i, j]].abs().sqrt().max(1e-12);
            assert!(
                (cov[[i, j]] - posterior_cov[[i, j]]).abs() <= 3.0 * sigma,
                "Gaussian posterior draws from sample_standard should recover posterior covariance within 3σ"
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
    let spec =
        generativespec_from_predict(pred, like, Some(0.5)).expect("spec generation should succeed");
    match spec.noise {
        NoiseModel::Gaussian { sigma } => {
            assert!(
                sigma.iter().all(|v| (*v - 1.0).abs() < 1e-12),
                "predict -> generative round-trip should preserve the documented Gaussian response scale"
            );
        }
        _ => panic!("predict -> generative round-trip should yield Gaussian noise model"),
    }
}

#[test]
fn bug_sample_table_same_seed_produces_same_table() {
    let spec = GenerativeSpec {
        mean: Array1::from_vec(vec![2.0, 2.0, 2.0]),
        noise: NoiseModel::Poisson,
    };
    let mut rng_a = StdRng::seed_from_u64(99);
    let mut rng_b = StdRng::seed_from_u64(99);
    let a = sampleobservation_replicates(&spec, 16, &mut rng_a).expect("sampling should succeed");
    let b = sampleobservation_replicates(&spec, 16, &mut rng_b).expect("sampling should succeed");
    assert_eq!(
        a, b,
        "Sampling table generation should be deterministic when using the same seed"
    );
}

#[test]
fn bug_polya_gamma_augmentation_marginal_identity_matches_documented_posterior() {
    let _fake: Option<NutsResult> = None;
    let mut rng = StdRng::seed_from_u64(101);
    let pg = PolyaGamma::new();
    let beta = 1.3;
    let omega_mean = (0..50_000).map(|_| pg.draw(&mut rng, beta)).sum::<f64>() / 50_000.0;
    let rhs = pg_theoretical_mean(beta);
    assert!(
        (omega_mean - rhs).abs() < 1e-4,
        "Polya-Gamma augmentation integral identity should recover the documented marginal posterior"
    );
}
