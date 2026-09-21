//! `warm_start_from` (gam#3002): on the parent's own inputs a fit resumes the
//! saved certified point with no outer iteration and lands on it; on other data
//! the point joins the marginal-slope multistart as one more seed and cannot
//! raise the published criterion; a point of another outer dimension leaves the
//! fit bit for bit cold. A model of another formula, or one that records no
//! point, is refused by name.

use csv::StringRecord;
use gam_data::{EncodedDataset, encode_recordswith_inferred_schema};
use gam_models::fit_orchestration::{
    FitConfig, WarmStartRefusal, WorkflowError, resolve_warm_start,
};
use gam_models::inference::model::FittedModelPayload;
use gam_models::inference::model_payload_builders::fit_formula_to_payload;
use rand::{SeedableRng, rngs::StdRng};
use rand_distr::{Distribution, Normal, Uniform};

const ROWS: usize = 400;
const FORMULA: &str = "event ~ s(age0)";

/// A binary marginal-slope fixture: the score's slope drifts with age.
fn fixture(seed: u64) -> EncodedDataset {
    let mut rng = StdRng::seed_from_u64(seed);
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

fn config(slope_formula: &str) -> FitConfig {
    FitConfig {
        link: Some("probit".to_string()),
        z_column: Some("z".to_string()),
        slope_formula: Some(slope_formula.to_string()),
        ..FitConfig::default()
    }
}

fn fit(data: &EncodedDataset, config: &FitConfig) -> FittedModelPayload {
    fit_formula_to_payload(FORMULA.to_string(), data, config).expect("the fit certifies")
}

/// A fit from `parent` on `data`, with the warm start it resolves to.
fn warm_fit(
    parent: &FittedModelPayload,
    data: &EncodedDataset,
    slope_formula: &str,
) -> (
    Result<FittedModelPayload, WorkflowError>,
    gam_model_api::WarmStart,
) {
    let cold = config(slope_formula);
    let warm_start = resolve_warm_start(parent, FORMULA, data, &cold)
        .expect("a model of the same formula resolves");
    let warm = FitConfig {
        warm_start: Some(warm_start.clone()),
        ..cold
    };
    (
        fit_formula_to_payload(FORMULA.to_string(), data, &warm),
        warm_start,
    )
}

fn certified_theta(payload: &FittedModelPayload) -> Vec<f64> {
    payload
        .fit_result
        .as_ref()
        .and_then(|fit| fit.artifacts.outer_warm_start.as_ref())
        .expect("a custom-family fit records its certified outer point")
        .theta
        .clone()
}

fn criterion(payload: &FittedModelPayload) -> f64 {
    payload
        .fit_result
        .as_ref()
        .and_then(|fit| fit.reml_score())
        .expect("the fit has a finite criterion")
}

fn outer_iterations(payload: &FittedModelPayload) -> usize {
    payload
        .fit_result
        .as_ref()
        .map(|fit| fit.outer_iterations)
        .expect("the payload carries its fit")
}

#[test]
fn a_warm_refit_on_the_parents_inputs_resumes_its_point_with_no_outer_iteration() {
    let data = fixture(0x57A2_7F20);
    let cold = fit(&data, &config("age0"));
    let (warm, warm_start) = warm_fit(&cold, &data, "age0");
    let warm = warm.expect("the resume certifies");
    assert!(
        warm_start.same_inputs,
        "the parent's own inputs match its fingerprint"
    );
    assert_eq!(
        warm_start.recorded(),
        Some(gam_model_api::WarmStartOutcome::Resumed)
    );
    assert_eq!(
        outer_iterations(&warm),
        0,
        "a certified point is still stationary on its own inputs"
    );
    let bits = |theta: Vec<f64>| {
        theta
            .iter()
            .map(|value| value.to_bits())
            .collect::<Vec<_>>()
    };
    assert_eq!(
        bits(certified_theta(&warm)),
        bits(certified_theta(&cold)),
        "the resume publishes the parent's certified point"
    );
    assert!(
        warm.informational_notes
            .iter()
            .any(|note| note.starts_with("warm_start_from: resumed")),
        "the model says it resumed: {:?}",
        warm.informational_notes
    );
}

#[test]
fn a_warm_start_from_other_data_joins_the_multistart_and_never_raises_the_criterion() {
    let parent = fit(&fixture(0x57A2_7F20), &config("age0"));
    let data = fixture(0x3002_0002);
    let cold = fit(&data, &config("age0"));
    let (warm, warm_start) = warm_fit(&parent, &data, "age0");
    let warm = warm.expect("the warm fit certifies");
    assert!(
        !warm_start.same_inputs,
        "other data has another fingerprint"
    );
    assert_eq!(
        warm_start.recorded(),
        Some(gam_model_api::WarmStartOutcome::JoinedMultistart),
        "the marginal-slope search takes the argmin over its seeds, so the point joins them"
    );
    let (cold_v, warm_v) = (criterion(&cold), criterion(&warm));
    let envelope = gam_solve::rho_optimizer::outer_value_agreement_bound(cold_v, warm_v);
    assert!(
        warm_v <= cold_v + envelope,
        "the argmin over a superset of the cold seeds: warm V={warm_v:.12e} cold V={cold_v:.12e}"
    );
}

#[test]
fn a_model_the_fit_cannot_resume_is_refused_by_name() {
    let data = fixture(0x57A2_7F20);
    let cold = fit(&data, &config("age0"));

    let other_formula = resolve_warm_start(&cold, "event ~ s(age0) + z", &data, &config("age0"))
        .expect_err("a model of another formula does not resume");
    assert!(matches!(
        other_formula,
        WorkflowError::WarmStartRefused {
            refusal: WarmStartRefusal::FormulaDiffers { .. }
        }
    ));

    let mut unrecorded = cold.clone();
    if let Some(fit) = unrecorded.fit_result.as_mut() {
        fit.artifacts.outer_warm_start = None;
    }
    let unrecorded_route = resolve_warm_start(&unrecorded, FORMULA, &data, &config("age0"))
        .expect_err("a model whose route records no point does not resume");
    assert!(matches!(
        unrecorded_route,
        WorkflowError::WarmStartRefused {
            refusal: WarmStartRefusal::NoRecordedPoint
        }
    ));
}

/// A point of another outer dimension belongs to another search: the fit runs
/// exactly as it runs cold, and says it did not use the point.
#[test]
fn a_point_of_another_outer_dimension_leaves_the_fit_cold() {
    let data = fixture(0x57A2_7F20);
    let parent = fit(&data, &config("age0"));
    let cold = fit(&data, &config("s(age0)"));
    let (warm, warm_start) = warm_fit(&parent, &data, "s(age0)");
    let warm = warm.expect("the fit runs cold");
    assert!(matches!(
        warm_start.recorded(),
        Some(gam_model_api::WarmStartOutcome::NotUsed(_))
    ));
    let bits = |theta: Vec<f64>| {
        theta
            .iter()
            .map(|value| value.to_bits())
            .collect::<Vec<_>>()
    };
    assert_eq!(
        bits(certified_theta(&warm)),
        bits(certified_theta(&cold)),
        "an unused point leaves the fit bit for bit cold"
    );
    assert!(
        warm.inference_notes
            .iter()
            .any(|note| note
                .starts_with("warm_start_from: the model's certified point was not used")),
        "the model says the point was not used: {:?}",
        warm.inference_notes
    );
}
