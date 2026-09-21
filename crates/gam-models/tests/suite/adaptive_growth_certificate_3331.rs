//! #3331: adaptive basis growth compares two fits' REML/LAML evidence only
//! through the error bar each fit certifies for its own criterion value. A
//! fit whose outer certificate carries no such bar can never grow, so every
//! standard family the formula workflow grows must certify one at its
//! converged optimum.

use csv::StringRecord;
use gam_data::{EncodedDataset, encode_recordswith_inferred_schema};
use gam_models::fit_orchestration::{FitConfig, FitResult, fit_from_formula};
use rand::SeedableRng;
use rand::rngs::StdRng;
use rand_distr::{Distribution, Normal, Poisson, Uniform};
use std::f64::consts::PI;

fn dataset(x: &[f64], y: &[f64]) -> EncodedDataset {
    let headers = vec!["y".to_string(), "x".to_string()];
    let records: Vec<StringRecord> = x
        .iter()
        .zip(y)
        .map(|(x, y)| StringRecord::from(vec![y.to_string(), x.to_string()]))
        .collect();
    encode_recordswith_inferred_schema(headers, records).expect("encode")
}

fn certified_error(family: &str, data: &EncodedDataset) {
    let config = FitConfig {
        family: Some(family.to_string()),
        ..FitConfig::default()
    };
    let FitResult::Standard(fit) = fit_from_formula("y ~ s(x)", data, &config)
        .unwrap_or_else(|e| panic!("{family} s(x) must fit, got: {e}"))
    else {
        panic!("expected a standard fit for {family}");
    };
    let certificate = fit
        .fit
        .artifacts
        .criterion_certificate
        .as_ref()
        .unwrap_or_else(|| panic!("{family}: the converged fit carries no outer certificate"));
    let error = certificate.criterion_error.unwrap_or_else(|| {
        panic!("{family}: the outer certificate carries no criterion error bar, so growth is impossible")
    });
    assert!(
        error.decrease_left.is_finite() && error.decrease_left >= 0.0,
        "{family}: decrease bound {error:?}"
    );
    assert!(
        error.value_band.is_finite() && error.value_band >= 0.0,
        "{family}: value band {error:?}"
    );
}

fn covariate(n: usize, rng: &mut StdRng) -> Vec<f64> {
    let unit = Uniform::new(0.0, 1.0).expect("unit");
    (0..n).map(|_| unit.sample(rng)).collect()
}

fn signal(x: f64) -> f64 {
    (2.0 * PI * 2.0 * x).sin()
}

#[test]
fn gaussian_fit_certifies_its_criterion_error() {
    let mut rng = StdRng::seed_from_u64(3331);
    let x = covariate(400, &mut rng);
    let noise = Normal::new(0.0, 0.3).expect("noise");
    let y: Vec<f64> = x.iter().map(|&x| signal(x) + noise.sample(&mut rng)).collect();
    certified_error("gaussian", &dataset(&x, &y));
}

#[test]
fn binomial_fit_certifies_its_criterion_error() {
    let mut rng = StdRng::seed_from_u64(3332);
    let x = covariate(600, &mut rng);
    let unit = Uniform::new(0.0, 1.0).expect("unit");
    let y: Vec<f64> = x
        .iter()
        .map(|&x| {
            let p = 1.0 / (1.0 + (-2.0 * signal(x)).exp());
            f64::from(u8::from(unit.sample(&mut rng) < p))
        })
        .collect();
    certified_error("binomial", &dataset(&x, &y));
}

#[test]
fn poisson_fit_certifies_its_criterion_error() {
    let mut rng = StdRng::seed_from_u64(3333);
    let x = covariate(400, &mut rng);
    let y: Vec<f64> = x
        .iter()
        .map(|&x| {
            Poisson::new((1.0 + signal(x)).exp())
                .expect("rate")
                .sample(&mut rng)
        })
        .collect();
    certified_error("poisson", &dataset(&x, &y));
}
