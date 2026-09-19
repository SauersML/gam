//! #3164: the Bernoulli marginal-slope fit solves one objective whichever
//! route it takes. Both the fast path (no κ / fixed frailty) and the exact
//! joint route (learned frailty σ) fit unarmed first and arm the Jeffreys
//! term only on the unarmed fit's own typed evidence (#979 ruling (b)); the
//! armed fit publishes that evidence on `artifacts.jeffreys_arming_evidence`.
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

/// `separated`: y = 1[z > 0] exactly, so the latent slope has an improper
/// likelihood ray. Otherwise y is drawn from a moderate probit model.
fn dataset(separated: bool) -> gam_data::EncodedDataset {
    let headers = ["y", "z", "x"].iter().map(|s| s.to_string()).collect();
    let mut state = 0x3164u64;
    let rows = (0..400)
        .map(|_| {
            let x = next_gauss(&mut state);
            let z = next_gauss(&mut state);
            let y = if separated {
                u8::from(z > 0.0)
            } else {
                u8::from(next_unit(&mut state) < normal_cdf(-0.2 + 0.4 * x + 0.7 * z))
            };
            StringRecord::from(vec![
                y.to_string(),
                format!("{z:.17e}"),
                format!("{x:.17e}"),
            ])
        })
        .collect();
    encode_recordswith_inferred_schema(headers, rows).expect("encode dataset")
}

fn config(learned_sigma: bool) -> FitConfig {
    FitConfig {
        family: Some("bernoulli-marginal-slope".to_string()),
        z_column: Some("z".to_string()),
        slope_formula: Some("1".to_string()),
        frailty: if learned_sigma {
            FrailtySpec::GaussianShift {
                scale: FrailtyScale::Learned { initial_sigma: 0.5 },
            }
        } else {
            FrailtySpec::None
        },
        ..FitConfig::default()
    }
}

fn armed(separated: bool, learned_sigma: bool) -> bool {
    let payload = fit_formula_to_payload(
        "y ~ x".to_string(),
        &dataset(separated),
        &config(learned_sigma),
    )
    .unwrap_or_else(|e| panic!("fit (separated={separated}, learned σ={learned_sigma}): {e}"));
    payload
        .fit_result
        .expect("BMS payload carries its fit")
        .artifacts
        .jeffreys_arming_evidence
        .is_some()
}

#[test]
fn non_separating_fit_stays_unarmed_on_the_fast_path_3164() {
    assert!(!armed(false, false));
}

#[test]
fn non_separating_fit_stays_unarmed_on_the_exact_joint_route_3164() {
    // Pins the removed `jeffreys_armed: true` literal: the exact route used to
    // solve the Jeffreys-penalized objective on every fit.
    assert!(!armed(false, true));
}

#[test]
fn separated_fit_arms_on_the_fast_path_3164() {
    assert!(armed(true, false));
}

#[test]
fn separated_fit_arms_on_the_exact_joint_route_3164() {
    assert!(armed(true, true));
}
