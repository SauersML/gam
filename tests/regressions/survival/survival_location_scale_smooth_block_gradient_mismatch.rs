//! Regression for #512: default survival location-scale fits used to route
//! `Surv(entry, exit, event) ~ s(x)`'s smooth columns into the log-σ block when
//! no `noise_formula` was supplied. The solver's canonical-gauge audit then
//! reduced that duplicated scale block to width zero, while
//! `SurvivalLocationScaleFamily` still emitted a width-`s(x)` scale gradient.
//! REML startup aborted with
//! "SurvivalLocationScaleFamily joint gradient length mismatch for block 2".
//!
//! The same deterministic right-censored data and smooth formula already fit
//! through the transformation and Weibull survival modes, so this pins the
//! default location-scale wiring against the same ordinary penalized-smooth
//! workload: all three modes must fit and return finite, non-degenerate
//! coefficients.

use csv::StringRecord;
use gam::{FitConfig, FitResult, encode_recordswith_inferred_schema, fit_from_formula};

fn build_dataset(n: usize) -> gam::inference::data::EncodedDataset {
    let headers = ["entry", "exit", "event", "x"]
        .into_iter()
        .map(str::to_string)
        .collect::<Vec<_>>();
    let mut rows = Vec::with_capacity(n);
    for i in 0..n {
        let x = -1.0 + 2.0 * (i as f64) / ((n - 1) as f64);
        // Deterministic right-censored exponential survival data. Mean exit
        // time E[T|x] = exp(0.5 + 0.6·x); we sample T = -ln(u) * mean using
        // a deterministic u_i in (0, 1) derived from a hashed integer index
        // so the test never touches an RNG. Every 7th row is right-censored
        // at half its draw (event=0); the rest are observed (event=1).
        let u = (((i as u64).wrapping_mul(1_103_515_245).wrapping_add(12345) >> 7) % 9999 + 1)
            as f64
            / 10000.0;
        let mean = (0.5 + 0.6 * x).exp();
        let draw = -u.ln() * mean;
        let censored = i % 7 == 0;
        let exit = if censored { 0.5 * draw } else { draw };
        let event = if censored { 0.0 } else { 1.0 };
        rows.push(StringRecord::from(vec![
            "0".to_string(),
            format!("{exit}"),
            format!("{event}"),
            format!("{x}"),
        ]));
    }
    encode_recordswith_inferred_schema(headers, rows).expect("encode dataset")
}

fn assert_finite_non_degenerate_coefficients(label: &str, fit: &FitResult) {
    let blocks = match fit {
        FitResult::SurvivalLocationScale(result) => &result.fit.fit.blocks,
        FitResult::SurvivalTransformation(result) => &result.fit.blocks,
        other => panic!(
            "{label}: unexpected fit result variant: {}",
            fit_result_kind(other)
        ),
    };

    let coefficient_count = blocks.iter().map(|block| block.beta.len()).sum::<usize>();
    assert!(
        coefficient_count > 0,
        "{label}: fit returned no coefficients"
    );

    let finite_count = blocks
        .iter()
        .flat_map(|block| block.beta.iter())
        .filter(|value| value.is_finite())
        .count();
    assert_eq!(
        finite_count, coefficient_count,
        "{label}: fit returned non-finite coefficients"
    );

    let max_abs = blocks
        .iter()
        .flat_map(|block| block.beta.iter())
        .map(|value| value.abs())
        .fold(0.0_f64, f64::max);
    assert!(
        max_abs > 1e-10,
        "{label}: fit returned degenerate all-zero coefficients"
    );
}

fn fit_result_kind(fit: &FitResult) -> &'static str {
    match fit {
        FitResult::Standard(_) => "Standard",
        FitResult::GaussianLocationScale(_) => "GaussianLocationScale",
        FitResult::BinomialLocationScale(_) => "BinomialLocationScale",
        FitResult::SurvivalLocationScale(_) => "SurvivalLocationScale",
        FitResult::SurvivalTransformation(_) => "SurvivalTransformation",
        FitResult::BernoulliMarginalSlope(_) => "BernoulliMarginalSlope",
        FitResult::SurvivalMarginalSlope(_) => "SurvivalMarginalSlope",
        FitResult::LatentSurvival(_) => "LatentSurvival",
        FitResult::LatentBinary(_) => "LatentBinary",
        FitResult::TransformationNormal(_) => "TransformationNormal",
        FitResult::DispersionLocationScale(_) => "DispersionLocationScale",
        FitResult::SplineScan(_) => "SplineScan",
        FitResult::ResidualCascade(_) => "ResidualCascade",
    }
}

fn fit_survival_smooth(label: &str, survival_likelihood: &str) -> FitResult {
    let data = build_dataset(600);
    let config = FitConfig {
        survival_likelihood: Some(survival_likelihood.to_string()),
        ..FitConfig::default()
    };
    fit_from_formula("Surv(entry, exit, event) ~ s(x, k=10)", &data, &config)
        .unwrap_or_else(|err| panic!("{label}: fit failed: {err}"))
}

#[test]
fn surv_smooth_fits_in_transformation_weibull_and_default_location_scale_modes() {
    for (label, survival_likelihood) in [
        ("transformation reference", "transformation"),
        ("weibull reference", "weibull"),
        ("default location-scale", "location-scale"),
    ] {
        let fit = fit_survival_smooth(label, survival_likelihood);
        assert_finite_non_degenerate_coefficients(label, &fit);
    }
}
