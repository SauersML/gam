//! gam#2943: the marginal-slope Murphy–Topel generated-regressor correction
//! reached only the top-level covariance matrices. A fit whose latent-z
//! conditional calibration fired kept the uncorrected matrix and standard
//! errors in its inference block, so its published standard errors disagreed
//! with its published covariance and every load of the saved model refused it
//! ("inference conditional covariance must match top-level
//! covariance_conditional"). That ended 6 of 9 fits of the gnomon#2335 All of
//! Us benchmark.
//!
//! The fixture is a scaled-down copy of that benchmark's nonlinear synthetic
//! generator: six ancestry groups with admixture continua in six PCs, and a
//! score whose cluster means and spreads no linear PC model removes, with
//! gamma(2) skew and t(3) tails. The benchmark's linear PC adjustment of the
//! score is left out; the cluster structure it cannot remove is what fires the
//! calibration, and the tails send the second-stage measure to global-empirical.
//! A small one-dimensional smooth retains smoothing coordinates and the full
//! generated-regressor covariance path without a six-dimensional benchmark fit.
//! Calibration, empirical measure and both covariance pairs are asserted below.

use csv::StringRecord;
use gam_data::{EncodedDataset, encode_recordswith_inferred_schema};
use gam_models::bms::LatentMeasureKind;
use gam_models::fit_orchestration::FitConfig;
use gam_models::inference::model::{FittedModel, FittedModelPayload};
use gam_models::inference::model_payload_builders::fit_formula_to_payload;
use rand::{RngExt as _, SeedableRng, rngs::StdRng};
use rand_distr::{Beta, Distribution, Gamma, Normal, StudentT};

const ROWS: usize = 256;
/// EUR, AFR, AMR, EAS, SAS, MID.
const SHARE: [f64; 6] = [0.50, 0.22, 0.19, 0.05, 0.025, 0.015];
/// Continental poles in six PCs: EUR, AFR, NAT, EAS, SAS, MID.
const POLES: [[f64; 6]; 6] = [
    [0.00, 0.00, 0.00, 0.00, 0.00, 0.00],
    [1.00, 0.10, 0.05, 0.00, 0.00, 0.00],
    [0.15, -0.90, 0.20, 0.10, 0.00, 0.00],
    [0.10, -0.70, 0.80, 0.00, 0.10, 0.00],
    [0.05, -0.15, 0.35, 0.50, 0.00, 0.10],
    [0.02, 0.00, 0.10, 0.20, 0.30, 0.20],
];
const NOISE: [f64; 6] = [0.020, 0.015, 0.012, 0.010, 0.008, 0.006];
const PREVALENCE: f64 = 0.45;

fn nonlinear_score_dataset(seed: u64) -> EncodedDataset {
    let mut rng = StdRng::seed_from_u64(seed);
    let normal = Normal::new(0.0, 1.0).expect("unit normal");
    let gamma2 = Gamma::new(2.0, 1.0).expect("gamma(2)");
    let t3 = StudentT::new(3.0).expect("t(3)");
    let afr_share = Beta::new(8.0, 2.0).expect("beta(8, 2)");
    let nat_share = Beta::new(2.0, 3.0).expect("beta(2, 3)");
    let amr_afr_share = Beta::new(1.0, 4.0).expect("beta(1, 4)");

    let mut groups = Vec::with_capacity(ROWS);
    let mut pcs = Vec::with_capacity(ROWS);
    let mut ages = Vec::with_capacity(ROWS);
    let mut sexes = Vec::with_capacity(ROWS);
    let mut scores = Vec::with_capacity(ROWS);
    let mut slopes = Vec::with_capacity(ROWS);
    let mut afrs = Vec::with_capacity(ROWS);
    for _ in 0..ROWS {
        let draw = rng.random::<f64>();
        let mut cumulative = 0.0;
        let mut group = SHARE.len() - 1;
        for (index, share) in SHARE.iter().enumerate() {
            cumulative += share;
            if draw < cumulative {
                group = index;
                break;
            }
        }
        let mut mix = [0.0_f64; 6];
        match group {
            0 => mix[0] = 1.0,
            1 => {
                let afr = afr_share.sample(&mut rng);
                mix[0] = 1.0 - afr;
                mix[1] = afr;
            }
            2 => {
                let nat = nat_share.sample(&mut rng);
                let afr = 0.15 * amr_afr_share.sample(&mut rng);
                mix[0] = 1.0 - nat - afr;
                mix[1] = afr;
                mix[2] = nat;
            }
            pole => mix[pole] = 1.0,
        }
        let mut pc = [0.0_f64; 6];
        for (column, value) in pc.iter_mut().enumerate() {
            *value = (0..6).map(|k| mix[k] * POLES[k][column]).sum::<f64>()
                + NOISE[column] * normal.sample(&mut rng);
        }
        let (afr, nat, eas, sas, mid) = (mix[1], mix[2], mix[3], mix[4], mix[5]);
        let spread = 1.0 + 0.8 * afr - 0.4 * eas;
        let skew = (gamma2.sample(&mut rng) - 2.0) / 2.0_f64.sqrt();
        let score = -2.5 * afr - 0.8 * nat + 3.0 * eas - 1.5 * sas
            + 2.0 * mid
            + 2.0 * afr * (1.0 - afr)
            + spread * (0.7 * skew + 0.5 * t3.sample(&mut rng));
        groups.push(group);
        pcs.push(pc);
        sexes.push(if rng.random::<f64>() < 0.5 { 0.0 } else { 1.0 });
        ages.push((55.0 + 16.0 * normal.sample(&mut rng)).clamp(18.0, 95.0));
        scores.push(score);
        slopes.push(0.35 - 0.25 * afr - 0.15 * nat);
        afrs.push(afr);
    }

    // Liability uses the score standardised within its ancestry group, as the
    // benchmark generator does; the event is its top 45 % with unit noise.
    let mut within = vec![0.0_f64; ROWS];
    for group in 0..SHARE.len() {
        let members: Vec<usize> = (0..ROWS).filter(|&row| groups[row] == group).collect();
        if members.len() < 2 {
            continue;
        }
        let mean = members.iter().map(|&row| scores[row]).sum::<f64>() / members.len() as f64;
        let sd = (members
            .iter()
            .map(|&row| (scores[row] - mean).powi(2))
            .sum::<f64>()
            / members.len() as f64)
            .sqrt();
        for &row in &members {
            within[row] = (scores[row] - mean) / sd;
        }
    }
    let noisy: Vec<f64> = (0..ROWS)
        .map(|row| {
            0.02 * (ages[row] - 55.0) + 0.1 * sexes[row] - 0.2 * afrs[row]
                + slopes[row] * within[row]
                + normal.sample(&mut rng)
        })
        .collect();
    let mut sorted = noisy.clone();
    sorted.sort_by(f64::total_cmp);
    let threshold = sorted[((1.0 - PREVALENCE) * ROWS as f64) as usize];

    let mean = scores.iter().sum::<f64>() / ROWS as f64;
    let sd = (scores.iter().map(|s| (s - mean).powi(2)).sum::<f64>() / ROWS as f64).sqrt();
    let headers = [
        "event", "z", "age0", "sex", "PC1", "PC2", "PC3", "PC4", "PC5", "PC6",
    ]
    .iter()
    .map(|name| name.to_string())
    .collect();
    let records = (0..ROWS)
        .map(|row| {
            let mut fields = vec![
                if noisy[row] > threshold {
                    "1".to_string()
                } else {
                    "0".to_string()
                },
                format!("{:.17e}", (scores[row] - mean) / sd),
                format!("{:.17e}", ages[row]),
                if sexes[row] > 0.5 {
                    "1".to_string()
                } else {
                    "0".to_string()
                },
            ];
            fields.extend(pcs[row].iter().map(|v| format!("{v:.17e}")));
            StringRecord::from(fields)
        })
        .collect();
    encode_recordswith_inferred_schema(headers, records).expect("encode the nonlinear score frame")
}

fn fit_payload(data: &EncodedDataset, formula: &str, slope_formula: &str) -> FittedModelPayload {
    let config = FitConfig {
        link: Some("probit".to_string()),
        z_column: Some("z".to_string()),
        slope_formula: Some(slope_formula.to_string()),
        ..FitConfig::default()
    };
    fit_formula_to_payload(formula.to_string(), data, &config).expect("the fixture fits")
}

/// The corrected fit publishes one covariance and the standard errors of that
/// covariance, and its saved model loads.
fn assert_corrected_fit_is_consistent_and_loads(payload: FittedModelPayload, label: &str) {
    assert!(
        matches!(
            payload.latent_measure,
            Some(LatentMeasureKind::GlobalEmpirical { .. })
        ),
        "{label}: the fixture must reach the global-empirical measure; got {}",
        payload
            .latent_measure
            .as_ref()
            .map(|measure| format!("{measure:?}").chars().take(80).collect::<String>())
            .unwrap_or_else(|| "none".to_string())
    );
    // #2943 is about a fit whose latent-z conditional calibration FIRED. A
    // fixture on which it no longer fires never reaches the defective path, and
    // the load below would pass for a reason unrelated to the fix.
    assert!(
        payload.latent_z_conditional_calibration.is_some(),
        "{label}: the fixture must fire the latent-z conditional calibration \
         (#2943's precondition); the payload carries none"
    );
    let fit = payload
        .fit_result
        .as_ref()
        .expect("the payload carries its fit result");
    assert!(
        !fit.log_lambdas.is_empty(),
        "{label}: the fixture must fit smoothing coordinates"
    );
    assert!(
        fit.artifacts.covariance_declined.is_none(),
        "{label}: the generated-regressor correction must be computed on this path, not withheld: {:?}",
        fit.artifacts.covariance_declined
    );
    for (pair, standard_errors, covariance) in [
        (
            "conditional",
            fit.beta_standard_errors(),
            fit.beta_covariance(),
        ),
        (
            "corrected",
            fit.beta_standard_errors_corrected(),
            fit.beta_covariance_corrected(),
        ),
    ] {
        // Both pairs are required: this fit has smoothing coordinates and did
        // not decline its covariance, so a missing pair is a regression, not a
        // case to skip.
        let standard_errors = standard_errors
            .unwrap_or_else(|| panic!("{label}: the {pair} standard errors must be published"));
        let covariance = covariance.unwrap_or_else(|| {
            panic!("{label}: {pair} standard errors are published without their covariance")
        });
        // `zip` stops at the shorter side, so a length mismatch would silently
        // compare only a prefix.
        assert_eq!(
            standard_errors.len(),
            covariance.nrows(),
            "{label}: {pair} publishes {} standard errors for a {}x{} covariance",
            standard_errors.len(),
            covariance.nrows(),
            covariance.ncols()
        );
        assert_eq!(
            covariance.nrows(),
            covariance.ncols(),
            "{label}: {pair} covariance must be square"
        );
        assert_eq!(
            standard_errors.len(),
            fit.beta.len(),
            "{label}: {pair} must cover every coefficient"
        );
        assert!(
            covariance.iter().all(|value| value.is_finite()),
            "{label}: {pair} covariance must be finite"
        );
        for (i, (se, diagonal)) in standard_errors
            .iter()
            .zip(covariance.diag().iter())
            .enumerate()
        {
            assert!(
                se.is_finite() && *se >= 0.0 && *diagonal >= 0.0,
                "{label}: {pair} invalid variance at {i}"
            );
            assert!(
                (se * se - diagonal).abs() <= 1.0e-12 * diagonal.abs().max(1.0),
                "{label}: {pair} standard error {i} squares to {:.17e} against the published \
                 covariance diagonal {diagonal:.17e}",
                se * se
            );
        }
    }
    // What `gamfit.fit` hands `compile_model`: the saved model's bytes, parsed
    // and validated exactly as a load does.
    let bytes =
        serde_json::to_vec(&FittedModel::from_payload(payload)).expect("serialize the model");
    let loaded: FittedModel = serde_json::from_slice(&bytes).expect("parse the saved model");
    loaded
        .validate_for_persistence()
        .unwrap_or_else(|err| panic!("{label}: the saved model must load: {err:?}"));
    loaded
        .validate_numeric_finiteness()
        .unwrap_or_else(|err| panic!("{label}: the saved model must be finite: {err:?}"));
}

#[test]
fn a_fit_whose_calibration_fired_saves_a_model_that_loads_2943() {
    let data = nonlinear_score_dataset(2335);
    let payload = fit_payload(&data, "event ~ s(PC1, k=4) + PC2 + sex", "1");
    assert_corrected_fit_is_consistent_and_loads(payload, "gnomon#2335 nonlinear score");
}
