//! Estimated-theta negative-binomial GAM (gam#3349): the joint (theta, rho)
//! certificate judges the theta block against the resolution the rho block's
//! own certificate leaves in the mode, not a literal outer tolerance. On a
//! lognormal-overdispersed count fixture whose second smoothing coordinate is a
//! flat, shrunk-out direction, the alternation reaches an accurate joint point
//! and the fit certifies it.

use csv::StringRecord;
use gam_data::{EncodedDataset, encode_recordswith_inferred_schema};
use gam_models::fit_orchestration::FitConfig;
use gam_models::inference::model_payload_builders::fit_formula_to_payload;
use rand::{SeedableRng, rngs::StdRng};
use rand_distr::{Distribution, Normal, Poisson, Uniform};

const ROWS: usize = 400;

/// `y ~ Poisson(exp(1 + sin(2 pi x) + 0.3 z))`, `x ~ U(0, 1)`, `z ~ N(0, 1)`:
/// counts overdispersed by a lognormal frailty.
fn count_fixture(seed: u64) -> EncodedDataset {
    let mut rng = StdRng::seed_from_u64(seed);
    let normal = Normal::new(0.0, 1.0).expect("unit normal");
    let unit = Uniform::new(0.0, 1.0).expect("unit interval");
    let headers = ["y", "x"].iter().map(|name| name.to_string()).collect();
    let records = (0..ROWS)
        .map(|_| {
            let x: f64 = unit.sample(&mut rng);
            let z: f64 = normal.sample(&mut rng);
            let mean = (1.0 + (2.0 * std::f64::consts::PI * x).sin() + 0.3 * z).exp();
            let y: f64 = Poisson::new(mean).expect("positive mean").sample(&mut rng);
            StringRecord::from(vec![format!("{y}"), format!("{x:.17e}")])
        })
        .collect();
    encode_recordswith_inferred_schema(headers, records).expect("encode the fixture")
}

#[test]
fn estimated_theta_negative_binomial_certifies_its_accurate_joint_point() {
    let data = count_fixture(0x3002_5001);
    let config = FitConfig {
        family: Some("negative-binomial".to_string()),
        ..FitConfig::default()
    };
    let payload = fit_formula_to_payload("y ~ s(x)".to_string(), &data, &config)
        .unwrap_or_else(|error| panic!("the estimated-theta fit must certify: {error}"));
    let fit = payload.fit_result.as_ref().expect("the payload carries its fit");
    assert!(
        fit.reml_score().is_some_and(f64::is_finite),
        "the certified fit has a finite criterion"
    );
}
