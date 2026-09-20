//! #3059: a learnable Gaussian-shift frailty σ no longer pins the Bernoulli
//! marginal-slope fit to the standard-normal closed form. The rigid kernel
//! differentiates σ on whichever law the row anchors on, so a skewed score
//! whose estimated law beats the closed form is fitted on that law instead of
//! being recorded as `gaussian-uncertified`.
//!
//! The probit likelihood reads σ only through the observed slope `s(σ)·g(x)`,
//! so σ is identified only when the fixed part of `g` (pilot baseline + slope
//! offset) lies outside the slope design's span. An intercept slope (`"1"`)
//! absorbs every σ move and is refused as an input error; a slope without its
//! own intercept (`"0 + x"`) leaves σ identified and fits.
use csv::StringRecord;
use gam_data::encode_recordswith_inferred_schema;
use gam_linalg::utils::splitmix64;
use gam_math::probability::normal_cdf;
use gam_models::fit_orchestration::FitConfig;
use gam_models::inference::model_payload_builders::fit_formula_to_payload;
use gam_models::survival::lognormal_kernel::{FrailtyScale, FrailtySpec};

fn next_unit(state: &mut u64) -> f64 {
    (splitmix64(state) >> 11) as f64 / (1u64 << 53) as f64
}

fn next_gauss(state: &mut u64) -> f64 {
    let u1 = next_unit(state).max(f64::MIN_POSITIVE);
    let u2 = next_unit(state);
    (-2.0 * u1.ln()).sqrt() * (std::f64::consts::TAU * u2).cos()
}

/// The issue's reproducer: a standardised Gamma(2, 1) score (skewness √2).
fn skewed_score_dataset() -> gam_data::EncodedDataset {
    let headers = ["y", "z", "x"].iter().map(|s| s.to_string()).collect();
    let mut state = 0x3059u64;
    let rows = (0..3000)
        .map(|_| {
            let x = next_gauss(&mut state);
            let u1 = next_unit(&mut state).max(f64::MIN_POSITIVE);
            let u2 = next_unit(&mut state).max(f64::MIN_POSITIVE);
            let z = (-u1.ln() - u2.ln() - 2.0) / std::f64::consts::SQRT_2;
            let y = u8::from(next_unit(&mut state) < normal_cdf(-0.3 + 0.5 * x + z));
            StringRecord::from(vec![
                y.to_string(),
                format!("{z:.17e}"),
                format!("{x:.17e}"),
            ])
        })
        .collect();
    encode_recordswith_inferred_schema(headers, rows).expect("encode dataset")
}

fn learned_sigma_config(slope_formula: &str) -> FitConfig {
    FitConfig {
        family: Some("bernoulli-marginal-slope".to_string()),
        z_column: Some("z".to_string()),
        slope_formula: Some(slope_formula.to_string()),
        frailty: FrailtySpec::GaussianShift {
            scale: FrailtyScale::Learned { initial_sigma: 0.5 },
        },
        ..FitConfig::default()
    }
}

#[test]
fn learned_sigma_under_an_intercept_slope_is_refused_as_unidentified_3059() {
    let err = fit_formula_to_payload(
        "y ~ x".to_string(),
        &skewed_score_dataset(),
        &learned_sigma_config("1"),
    )
    .err()
    .expect("σ is matched exactly by rescaling the slope intercept and must be refused");
    let message = err.to_string();
    assert!(
        message.contains("not identified"),
        "the refusal names the identifiability failure: {message}"
    );
}

#[test]
fn identified_learned_sigma_fits_a_skewed_score_on_its_finite_law_3059() {
    let payload = fit_formula_to_payload(
        "y ~ x".to_string(),
        &skewed_score_dataset(),
        &learned_sigma_config("0 + x"),
    )
    .unwrap_or_else(|e| panic!("identified learned-σ fit: {e}"));
    let consumed = payload
        .latent_law_consumed
        .as_ref()
        .expect("BMS payload records the latent law it consumed");
    assert!(
        !matches!(
            consumed.label(),
            "gaussian-uncertified" | "estimated-gaussian-adequate"
        ),
        "a skewed score under a learned σ is fitted on its estimated law, not the closed form: {}",
        consumed.label()
    );
    assert!(
        payload.fit_result.is_some(),
        "the finite-law learned-σ fit converged"
    );
}
