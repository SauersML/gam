//! `warm_start_from`: a fit resumes from a saved model's certified outer point
//! through the outer cache seam, recertifies it, and lands where the cold fit
//! did. A model it cannot resume is refused by name, never fitted cold.

use csv::StringRecord;
use gam_data::{EncodedDataset, encode_recordswith_inferred_schema};
use gam_models::fit_orchestration::{FitConfig, OuterWarmStart};
use gam_models::inference::model::FittedModelPayload;
use gam_models::inference::model_payload_builders::fit_formula_to_payload;
use rand::{SeedableRng, rngs::StdRng};
use rand_distr::{Distribution, Normal, Uniform};

const ROWS: usize = 400;
const FORMULA: &str = "event ~ s(age0)";

/// A binary marginal-slope fixture: the score's slope drifts with age.
fn fixture() -> EncodedDataset {
    let mut rng = StdRng::seed_from_u64(0x57A2_7F20);
    let normal = Normal::new(0.0, 1.0).expect("unit normal");
    let ages = Uniform::new(40.0, 70.0).expect("age range");
    let headers = ["event", "z", "age0"]
        .iter()
        .map(|name| name.to_string())
        .collect();
    let records = (0..ROWS)
        .map(|_| {
            let age: f64 = ages.sample(&mut rng);
            let z: f64 = normal.sample(&mut rng);
            let noise: f64 = normal.sample(&mut rng);
            let latent = 0.03 * (age - 55.0) + 0.6 * (1.0 + 0.01 * (age - 55.0)) * z + noise;
            StringRecord::from(vec![
                if latent > 0.2 {
                    "1".to_string()
                } else {
                    "0".to_string()
                },
                format!("{z:.17e}"),
                format!("{age:.17e}"),
            ])
        })
        .collect();
    encode_recordswith_inferred_schema(headers, records).expect("encode the fixture")
}

fn config(slope_formula: &str, warm_start: Option<OuterWarmStart>) -> FitConfig {
    FitConfig {
        link: Some("probit".to_string()),
        z_column: Some("z".to_string()),
        slope_formula: Some(slope_formula.to_string()),
        outer_warm_start: warm_start,
        ..FitConfig::default()
    }
}

fn certified_point(payload: &FittedModelPayload) -> (Vec<f64>, Vec<f64>) {
    let record = payload
        .fit_result
        .as_ref()
        .and_then(|fit| fit.artifacts.outer_warm_start.clone())
        .expect("a custom-family fit records its certified outer point");
    (record.rho, record.beta)
}

#[test]
fn a_warm_refit_resumes_at_the_cold_point_and_a_model_it_cannot_resume_is_refused() {
    let data = fixture();
    let cold = fit_formula_to_payload(FORMULA.to_string(), &data, &config("age0", None))
        .expect("the cold fit certifies");
    let (cold_rho, cold_beta) = certified_point(&cold);

    let scratch = tempfile::tempdir().expect("a scratch directory");
    let warm_start = OuterWarmStart::from_model(&cold, FORMULA, scratch.path().to_path_buf())
        .expect("a model of the same formula resumes");
    let warm = fit_formula_to_payload(
        FORMULA.to_string(),
        &data,
        &config("age0", Some(warm_start.clone())),
    )
    .expect("the warm fit certifies");
    assert!(
        warm_start.consumed(),
        "the fit resumed from the model's point"
    );
    let (warm_rho, warm_beta) = certified_point(&warm);
    let drift = cold_rho
        .iter()
        .zip(&warm_rho)
        .fold(0.0_f64, |worst, (cold, warm)| {
            worst.max((cold - warm).abs())
        });
    assert!(
        drift <= 1e-4,
        "the warm fit certifies where the cold one did: max |Δρ| = {drift:.3e}"
    );
    assert_eq!(
        cold_beta.len(),
        warm_beta.len(),
        "the coefficient layout is unchanged"
    );
    let railed = |payload: &FittedModelPayload| {
        payload
            .fit_result
            .as_ref()
            .and_then(|fit| fit.artifacts.criterion_certificate.as_ref())
            .map(|certificate| certificate.lambdas_railed.clone())
    };
    assert_eq!(
        railed(&cold),
        railed(&warm),
        "the same coordinates are railed"
    );
    let iterations =
        |payload: &FittedModelPayload| payload.fit_result.as_ref().map(|fit| fit.outer_iterations);
    assert!(
        iterations(&warm) <= iterations(&cold),
        "resuming never costs outer iterations: warm {:?}, cold {:?}",
        iterations(&warm),
        iterations(&cold)
    );

    let other_formula =
        OuterWarmStart::from_model(&cold, "event ~ s(age0) + z", scratch.path().to_path_buf())
            .expect_err("a model of another formula does not resume");
    assert!(other_formula.contains("warm_start_from"), "{other_formula}");

    let other_width = OuterWarmStart::from_model(&cold, FORMULA, scratch.path().to_path_buf())
        .expect("the point stages");
    let refused = fit_formula_to_payload(
        FORMULA.to_string(),
        &data,
        &config("s(age0)", Some(other_width)),
    )
    .err()
    .expect("a point of another design width is refused");
    assert!(
        refused.to_string().contains("warm_start_from"),
        "the refusal names the warm start: {refused}"
    );
}
