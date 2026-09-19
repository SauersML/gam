//! A tensor-product smooth beside a categorical main effect gets its summary
//! Wald p-value, in finite time.
//!
//! `y ~ factor(g) + te(x, z)` is a two-scale term, so its Wald row replays the
//! multi-scale λ̂-selection on every null draw. That selection used to sweep
//! one scale at a time along a valley that couples them, and on this fit a
//! single draw took more than sixty sweeps of `~140 ms` each: the summary never
//! returned. The row must come back with a p-value in `[0, 1]`.

use csv::StringRecord;
use gam::{FitConfig, FitResult, encode_recordswith_inferred_schema, fit_from_formula};
use gam_solve::estimate::smooth_term_summary_rows;
use rand::SeedableRng;
use rand::rngs::StdRng;
use rand_distr::{Distribution, Normal, Uniform};
use std::f64::consts::PI;

#[test]
fn a_tensor_smooth_beside_a_factor_gets_its_wald_pvalue() {
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
    let result = fit_from_formula("y ~ factor(g) + te(x, z)", &data, &config).expect("fit");
    let FitResult::Standard(fit) = result else {
        panic!("expected a standard fit");
    };
    let rows = smooth_term_summary_rows(&fit.design, &fit.resolvedspec, &fit.fit);
    let row = rows
        .iter()
        .find(|row| row.name.starts_with("te("))
        .unwrap_or_else(|| {
            let names: Vec<&str> = rows.iter().map(|row| row.name.as_str()).collect();
            panic!("no summary row for te(x, z) among {names:?}")
        });
    let p_value = row
        .pvalue
        .unwrap_or_else(|| panic!("te(x, z) has no Wald p-value: {:?}", row.pvalue_unavailable));
    assert!(
        (0.0..=1.0).contains(&p_value),
        "te(x, z) Wald p-value {p_value} is not a probability"
    );
}
