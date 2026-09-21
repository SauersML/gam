//! gam#3236 item 2: a 2-D isotropic Matérn smooth `y ~ matern(x, z)` must fit
//! a Gamma surface that `te(x, z)` fits, ending on a certified outer optimum.
//!
//! The fixture holds the rows of the Python contract
//! `test_matern_2d_smooth_is_fittable_under_gamma_family`:
//! `x, z ~ U(0, 1)`, `μ = exp(0.8·sin(3x) + 0.5·z)`, `y ~ Gamma(shape 5, mean μ)`,
//! n = 600, drawn by NumPy `default_rng(0)`. It is stored as CSV because the
//! generator is not reproducible from Rust. The report at main `890f20cb3e`
//! ended the iso-kappa joint REML search on
//! `trust_region_reject_floor(... after 14 consecutive rejections)` with
//! `|Pg| = 1.593` against a bound of `3.682e-2`.
//!
//! A fit is only minted from a converged optimization, so a returned payload is
//! itself the certificate. The recovery check reads the terminal inner solve's
//! fitted mean on the training rows and compares it with the true mean.

use csv::StringRecord;
use gam_data::{EncodedDataset, encode_recordswith_inferred_schema};
use gam_models::fit_orchestration::FitConfig;
use gam_models::inference::model_payload_builders::fit_formula_to_payload;

const GAMMA_SURFACE_SEED_0: &str = include_str!("fixtures/matern_gamma_surface_3236_seed0.csv");

fn dataset() -> EncodedDataset {
    let mut reader = csv::Reader::from_reader(GAMMA_SURFACE_SEED_0.as_bytes());
    let headers: Vec<String> = reader
        .headers()
        .expect("fixture header")
        .iter()
        .map(str::to_string)
        .collect();
    let rows: Vec<StringRecord> = reader
        .records()
        .map(|record| record.expect("fixture row"))
        .collect();
    encode_recordswith_inferred_schema(headers, rows).expect("encode the fixture")
}

fn true_mean() -> Vec<f64> {
    let mut reader = csv::Reader::from_reader(GAMMA_SURFACE_SEED_0.as_bytes());
    reader
        .records()
        .map(|record| {
            let record = record.expect("fixture row");
            let x: f64 = record[1].parse().expect("x");
            let z: f64 = record[2].parse().expect("z");
            (0.8 * (3.0 * x).sin() + 0.5 * z).exp()
        })
        .collect()
}

fn correlation(a: &[f64], b: &[f64]) -> f64 {
    let n = a.len() as f64;
    let ma = a.iter().sum::<f64>() / n;
    let mb = b.iter().sum::<f64>() / n;
    let (mut sab, mut saa, mut sbb) = (0.0, 0.0, 0.0);
    for (x, y) in a.iter().zip(b) {
        sab += (x - ma) * (y - mb);
        saa += (x - ma) * (x - ma);
        sbb += (y - mb) * (y - mb);
    }
    sab / (saa * sbb).sqrt()
}

/// Fits `formula` under the Gamma family and returns the correlation of the
/// fitted mean with the true mean on the training rows.
fn certified_recovery(formula: &str) -> f64 {
    let data = dataset();
    let config = FitConfig {
        family: Some("gamma".to_string()),
        ..FitConfig::default()
    };
    let payload = fit_formula_to_payload(formula.to_string(), &data, &config)
        .unwrap_or_else(|error| panic!("{formula}: the outer search must certify: {error}"));
    let fit = payload.fit_result.as_ref().expect("the payload carries its fit");
    assert!(
        fit.reml_score().is_some_and(f64::is_finite),
        "{formula}: the certified fit has a finite criterion"
    );
    let fitted_mean: Vec<f64> = fit
        .artifacts
        .pirls
        .as_ref()
        .expect("a single-predictor GLM fit carries its terminal inner solve")
        .finalmu
        .to_vec();
    assert!(
        fitted_mean.iter().all(|value| value.is_finite()),
        "{formula}: the fitted mean is finite"
    );
    correlation(&fitted_mean, &true_mean())
}

#[test]
fn tensor_gamma_surface_control_3236() {
    let corr = certified_recovery("y ~ te(x, z)");
    assert!(corr > 0.4, "te(x, z) recovers the surface: corr = {corr:.4}");
}

#[test]
fn matern_gamma_surface_certifies_and_recovers_3236() {
    let corr = certified_recovery("y ~ matern(x, z)");
    assert!(corr > 0.5, "matern(x, z) recovers the surface: corr = {corr:.4}");
}
