//! #3217: a latent score that separates the outcomes gives the pooled probit
//! of y on z — the Bernoulli marginal-slope fit's rigid pilot — no finite
//! mode. The unarmed pilot used to stop its Newton where the likelihood had
//! underflowed to zero and read that as a mode: the pilot slope was a point
//! far along the separating ray, every row's IRLS metric underflowed, and the
//! confound audit refused the fit as fully confounded. The unarmed pilot now
//! certifies the separation exactly, the certificate is typed Jeffreys arming
//! evidence, and the armed pilot is the Jeffreys-penalized mode, which is
//! finite.
use csv::StringRecord;
use gam_data::encode_recordswith_inferred_schema;
use gam_linalg::utils::splitmix64;
use gam_models::fit_orchestration::FitConfig;
use gam_models::inference::model_payload_builders::fit_formula_to_payload;
use gam_problem::jeffreys_arming::JeffreysArmingEvidence;

fn next_unit(state: &mut u64) -> f64 {
    (splitmix64(state) >> 11) as f64 / (1u64 << 53) as f64
}

fn next_gauss(state: &mut u64) -> f64 {
    let u1 = next_unit(state).max(f64::MIN_POSITIVE);
    let u2 = next_unit(state);
    (-2.0 * u1.ln()).sqrt() * (std::f64::consts::TAU * u2).cos()
}

#[test]
fn separating_latent_score_arms_the_pooled_pilot_3217() {
    let headers = ["y", "z", "x"].iter().map(|s| s.to_string()).collect();
    let mut state = 0x3217u64;
    let rows = (0..400)
        .map(|_| {
            let x = next_gauss(&mut state);
            let z = next_gauss(&mut state);
            let y = u8::from(z > 0.0);
            StringRecord::from(vec![
                y.to_string(),
                format!("{z:.17e}"),
                format!("{x:.17e}"),
            ])
        })
        .collect();
    let data = encode_recordswith_inferred_schema(headers, rows).expect("encode dataset");
    let config = FitConfig {
        family: Some("bernoulli-marginal-slope".to_string()),
        z_column: Some("z".to_string()),
        slope_formula: Some("1".to_string()),
        ..FitConfig::default()
    };
    let payload = fit_formula_to_payload("y ~ x".to_string(), &data, &config)
        .unwrap_or_else(|e| panic!("separated latent score fit: {e}"));
    let evidence = payload
        .fit_result
        .expect("BMS payload carries its fit")
        .artifacts
        .jeffreys_arming_evidence;
    match evidence {
        Some(JeffreysArmingEvidence::PrefitLatentScoreSeparation {
            threshold,
            positive_above_threshold,
        }) => {
            assert!(positive_above_threshold, "successes lie above the threshold");
            assert!(threshold.is_finite());
        }
        other => panic!("expected latent-score separation evidence, got {other:?}"),
    }
}
