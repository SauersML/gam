//! The per-term LR null model is the full model with one block at zero, at the
//! full fit's `λ̂`, driven through the real `smooth_term_lr_inference_forspec`.
//!
//! The reduced model used to be refitted from scratch, re-running REML for every
//! surviving smoothing parameter. When the tested term sits at its penalty null
//! space the two REML optima differ only by the outer search's tolerance, and
//! that tolerance became the statistic: `W ~ 1e-6 … 1e-4` against a null law
//! whose mean is `~1e-8`, published as a vanishing p-value for a term with no
//! effect. Seed 20 of the fixture below published `p < 2.3e-28` for the inert
//! `s(x2)` from `W = 2.9e-4`. The nested fit at fixed `ρ̂` makes the only
//! difference between the two optima the constraint.
//!
//! The fixture is the pyGAM audit's binomial inference cell (`n = 400`,
//! `η = 1.5 sin 2πx₁ + 0.6 cos 2πx₃`, `x₂` inert) drawn from a splitmix64
//! stream so the same data can be regenerated outside Rust.
#![cfg(test)]

use super::entry::materialize;
use super::request::{FitConfig, FitRequest};
use csv::StringRecord;
use gam_data::encode_recordswith_inferred_schema;

struct SplitMix64(u64);

impl SplitMix64 {
    fn next_unit(&mut self) -> f64 {
        self.0 = self.0.wrapping_add(0x9E37_79B9_7F4A_7C15);
        let mut z = self.0;
        z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
        z ^= z >> 31;
        (z >> 11) as f64 * (-53f64).exp2()
    }
}

fn binomial_cell_reports(seed: u64) -> Vec<super::drivers::SmoothTermLrReport> {
    let mut rng = SplitMix64(seed);
    let two_pi = 2.0 * std::f64::consts::PI;
    let rows: Vec<StringRecord> = (0..400)
        .map(|_| {
            let (x1, x2, x3, u) = (
                rng.next_unit(),
                rng.next_unit(),
                rng.next_unit(),
                rng.next_unit(),
            );
            let eta = 1.5 * (two_pi * x1).sin() + 0.6 * (two_pi * x3).cos();
            let y = if u < 1.0 / (1.0 + (-eta).exp()) { 1.0 } else { 0.0 };
            StringRecord::from(vec![
                x1.to_string(),
                x2.to_string(),
                x3.to_string(),
                f64::to_string(&y),
            ])
        })
        .collect();
    let headers = ["x1", "x2", "x3", "y"].map(String::from).to_vec();
    let ds = encode_recordswith_inferred_schema(headers, rows).expect("encode dataset");
    let cfg = FitConfig {
        family: Some("binomial".to_string()),
        ..FitConfig::default()
    };
    let formula = "y ~ s(x1) + s(x2) + s(x3)";
    let request = match materialize(formula, &ds, &cfg).expect("materialize").request {
        FitRequest::Standard(request) => request,
        _ => panic!("expected a Standard fit request for {formula}"),
    };
    super::drivers::smooth_term_lr_inference_forspec(
        request.data.view(),
        request.y.view(),
        request.weights.view(),
        request.offset.view(),
        &request.spec,
        request.family.clone(),
        &request.options,
    )
    .expect("smooth-term LR inference")
}

#[test]
fn a_null_term_is_not_scored_on_the_outer_search_tolerance() {
    let reports = binomial_cell_reports(20);
    let names: Vec<&str> = reports.iter().map(|r| r.name.as_str()).collect();
    assert_eq!(names, ["s(x1)", "s(x2)", "s(x3)"]);

    let null = reports[1]
        .inference()
        .unwrap_or_else(|| panic!("s(x2): {:?}", reports[1].outcome));
    // A calibrated null p-value is uniform; 1e-3 separates the defect (tens of
    // orders below it) from the bottom 0.1% of a correct law.
    let p = null.p_value.value().unwrap_or_else(|| {
        panic!(
            "s(x2) (no effect) published p < {:?} from W = {:.3e}",
            null.p_value.upper_bound(),
            null.statistic_lr
        )
    });
    assert!(
        p > 1e-3 && p <= 1.0,
        "s(x2) (no effect): p = {p:.3e} from W = {:.3e}, ref_df = {:.3e}",
        null.statistic_lr,
        null.ref_df
    );

    // The strong term is still detected: the fix moves the nested fit, not the
    // power.
    let strong = reports[0]
        .inference()
        .unwrap_or_else(|| panic!("s(x1): {:?}", reports[0].outcome));
    let ceiling = strong
        .p_value
        .value()
        .or(strong.p_value.upper_bound())
        .expect("a typed p-value");
    assert!(ceiling < 1e-6, "s(x1) not detected: {:?}", strong.p_value);
}
