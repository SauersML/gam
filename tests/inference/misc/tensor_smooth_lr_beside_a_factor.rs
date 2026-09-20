//! A tensor-product smooth beside a categorical main effect gets its
//! likelihood-ratio p-value, in finite time.
//!
//! `y ~ factor(g) + te(x, z)` is a two-scale term, so its LR reference replays
//! the multi-scale λ̂-selection on every null draw. That selection used to sweep
//! one scale at a time along a valley that couples them, and on this fit a
//! single draw took more than sixty sweeps of `~140 ms` each: the inference
//! never returned. The term must come back with a replayed selection and a
//! p-value in `[0, 1]`.

use csv::StringRecord;
use gam::smooth::{SmoothLrPValue, smooth_term_lr_inference_forspec};
use gam::{FitConfig, FitRequest, encode_recordswith_inferred_schema, materialize};
use rand::SeedableRng;
use rand::rngs::StdRng;
use rand_distr::{Distribution, Normal, Uniform};
use std::f64::consts::PI;

#[test]
fn a_tensor_smooth_beside_a_factor_gets_its_lr_pvalue() {
    let n = 500;
    let mut rng = StdRng::seed_from_u64(31);
    let unit = Uniform::new(0.0_f64, 1.0).expect("unit interval");
    let noise = Normal::new(0.0_f64, 0.3).expect("noise law");
    let rows = (0..n)
        .map(|index| {
            let x = unit.sample(&mut rng);
            let z = unit.sample(&mut rng);
            let y = (2.0 * PI * x).sin() + noise.sample(&mut rng);
            StringRecord::from(vec![
                format!("{x:.17e}"),
                format!("{z:.17e}"),
                format!("g{}", index % 5),
                format!("{y:.17e}"),
            ])
        })
        .collect();
    let data = encode_recordswith_inferred_schema(
        ["x", "z", "g", "y"].into_iter().map(String::from).collect(),
        rows,
    )
    .expect("encode dataset");
    let config = FitConfig {
        family: Some("gaussian".to_string()),
        ..FitConfig::default()
    };
    let materialized = materialize("y ~ factor(g) + te(x, z)", &data, &config).expect("materialize");
    let FitRequest::Standard(request) = materialized.request else {
        panic!("expected a standard fit request");
    };
    let reports = smooth_term_lr_inference_forspec(
        request.data.view(),
        request.y.view(),
        request.weights.view(),
        request.offset.view(),
        &request.spec,
        request.family,
        &request.options,
    )
    .expect("smooth-term LR inference");
    let report = reports
        .into_iter()
        .find(|report| report.name.starts_with("te("))
        .expect("an LR report for te(x, z)");
    let inference = report
        .outcome
        .unwrap_or_else(|reason| panic!("te(x, z) has no LR p-value: {reason}"));
    assert!(
        inference.ref_df_provenance.selection.replay().is_some(),
        "te(x, z) did not replay its λ̂-selection"
    );
    let p_value = match inference.p_value {
        SmoothLrPValue::Resolved(value) | SmoothLrPValue::UpperBound(value) => value,
    };
    assert!(
        (0.0..=1.0).contains(&p_value),
        "te(x, z) LR p-value {p_value} is not a probability"
    );
}
