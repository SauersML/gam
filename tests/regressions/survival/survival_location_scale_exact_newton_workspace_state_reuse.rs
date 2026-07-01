use csv::StringRecord;
use gam::{FitConfig, FitResult, encode_recordswith_inferred_schema, fit_from_formula};

fn tiny_dataset(rows: &[[f64; 4]]) -> gam::inference::data::EncodedDataset {
    let headers = ["entry", "exit", "event", "x"]
        .into_iter()
        .map(str::to_string)
        .collect::<Vec<_>>();
    let records = rows
        .iter()
        .map(|row| StringRecord::from(row.map(|v| v.to_string()).to_vec()))
        .collect::<Vec<_>>();
    encode_recordswith_inferred_schema(headers, records).expect("encode tiny dataset")
}

fn fit_theta(data: &gam::inference::data::EncodedDataset) -> Vec<f64> {
    let cfg = FitConfig {
        survival_likelihood: "location-scale".to_string(),
        noise_formula: Some("1".to_string()),
        ..FitConfig::default()
    };
    let FitResult::SurvivalLocationScale(fit) =
        fit_from_formula("Surv(entry, exit, event) ~ x", data, &cfg).expect("fit should succeed")
    else {
        panic!("expected survival location-scale result")
    };
    let u = &fit.fit.fit;
    u.beta_time()
        .iter()
        .chain(u.beta_threshold().iter())
        .chain(u.beta_log_sigma().iter())
        .copied()
        .collect()
}

#[test]
fn workspace_state_reuse_does_not_change_repeated_fit_output_after_different_call() {
    let data_a = tiny_dataset(&[
        [1.0, 2.0, 1.0, -1.0],
        [1.0, 3.0, 0.0, -0.7],
        [1.0, 4.0, 1.0, -0.4],
        [1.0, 5.0, 0.0, -0.1],
        [1.0, 6.0, 1.0, 0.2],
        [1.0, 7.0, 0.0, 0.5],
        [1.0, 8.0, 1.0, 0.8],
        [1.0, 9.0, 0.0, 1.1],
        [1.0, 10.0, 1.0, 1.4],
        [1.0, 11.0, 0.0, 1.7],
        [1.0, 12.0, 1.0, 2.0],
        [1.0, 13.0, 0.0, 2.3],
    ]);
    let data_b = tiny_dataset(&[
        [1.0, 2.5, 0.0, -0.8],
        [1.0, 3.5, 0.0, -0.5],
        [1.0, 4.5, 1.0, -0.2],
        [1.0, 5.5, 0.0, 0.1],
        [1.0, 6.5, 0.0, 0.4],
        [1.0, 7.5, 0.0, 0.7],
        [1.0, 8.5, 1.0, 1.0],
        [1.0, 9.5, 0.0, 1.3],
        [1.0, 10.5, 0.0, 1.6],
        [1.0, 11.5, 0.0, 1.9],
        [1.0, 12.5, 1.0, 2.2],
        [1.0, 13.5, 0.0, 2.5],
    ]);

    let theta_first = fit_theta(&data_a);
    fit_theta(&data_b);
    let theta_second = fit_theta(&data_a);

    let max_abs = theta_first
        .iter()
        .zip(theta_second.iter())
        .map(|(a, b)| (a - b).abs())
        .fold(0.0_f64, f64::max);

    assert!(
        max_abs <= 0.0,
        "Calling the exact Newton survival workspace twice with identical inputs should return identical coefficients"
    );
}
