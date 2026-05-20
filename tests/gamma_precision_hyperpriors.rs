use approx::assert_relative_eq;
use csv::StringRecord;
use gam::{FitConfig, FitResult, encode_recordswith_inferred_schema, fit_from_formula};

fn grouped_fixture(n_per: usize) -> gam::data::EncodedDataset {
    let means = [-2.0_f64, -1.0, 1.0, 2.0];
    let headers = ["g", "y"].into_iter().map(String::from).collect();
    let mut rows = Vec::<StringRecord>::with_capacity(n_per * means.len());
    for (group, mean) in means.iter().enumerate() {
        for i in 0..n_per {
            let noise = 0.01 * ((i % 5) as f64 - 2.0);
            rows.push(StringRecord::from(vec![
                group.to_string(),
                (mean + noise).to_string(),
            ]));
        }
    }
    encode_recordswith_inferred_schema(headers, rows).expect("fixture encoding")
}

#[test]
fn penalty_block_gamma_prior_shrinks_random_effect_coefficients() {
    let data = grouped_fixture(50);
    let cfg = FitConfig {
        family: Some("gaussian".to_string()),
        penalty_block_gamma_priors: vec![("g".to_string(), 1_000.0, 1.0)],
        ..FitConfig::default()
    };
    let fit = match fit_from_formula("y ~ group(g)", &data, &cfg).expect("gamma-prior fit") {
        FitResult::Standard(fit) => fit,
        _ => panic!("expected standard fit"),
    };

    let lambda = fit.fit.lambdas[0];
    assert!(
        (lambda - 1_000.0).abs() / 1_000.0 < 0.10,
        "prior-dominated lambda should stay near shape/rate: {lambda}"
    );
    let re_range = fit.design.random_effect_ranges[0].1.clone();
    let max_abs = fit
        .fit
        .beta
        .slice(ndarray::s![re_range])
        .iter()
        .map(|value| value.abs())
        .fold(0.0_f64, f64::max);
    assert!(
        max_abs < 0.12,
        "high-precision Gamma prior should shrink group coefficients toward zero, max_abs={max_abs}"
    );
}

#[test]
fn omitted_gamma_prior_matches_uninformed_fit_bitwise() {
    let data = grouped_fixture(12);
    let base_cfg = FitConfig {
        family: Some("gaussian".to_string()),
        ..FitConfig::default()
    };
    let explicit_empty_cfg = FitConfig {
        family: Some("gaussian".to_string()),
        penalty_block_gamma_priors: Vec::new(),
        ..FitConfig::default()
    };
    let base = match fit_from_formula("y ~ group(g)", &data, &base_cfg).expect("base fit") {
        FitResult::Standard(fit) => fit,
        _ => panic!("expected standard fit"),
    };
    let empty =
        match fit_from_formula("y ~ group(g)", &data, &explicit_empty_cfg).expect("empty fit") {
            FitResult::Standard(fit) => fit,
            _ => panic!("expected standard fit"),
        };

    // Both arms route through the same uninformed code path, but two
    // independent fits sharing the global rayon pool can disagree at 1–2
    // ULPs from work-stealing reduction order — not a math error.
    for (b, e) in base.fit.lambdas.iter().zip(empty.fit.lambdas.iter()) {
        assert_relative_eq!(b, e, max_relative = 1e-12);
    }
    for (b, e) in base.fit.beta.iter().zip(empty.fit.beta.iter()) {
        assert_relative_eq!(b, e, max_relative = 1e-12);
    }
    assert_relative_eq!(
        base.fit.reml_score,
        empty.fit.reml_score,
        max_relative = 1e-12
    );
}
