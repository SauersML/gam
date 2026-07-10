//! Regression for GitHub issue #157: at least one mgcv idiom for a plain
//! categorical / random-intercept term must be accepted by the formula DSL:
//!   - `factor(group)`  →  random-intercept / categorical effect on `group`
//!   - `s(group, bs='re')`  →  random intercept on the factor `group`
//!
//! Before the fix, `factor(...)` raised "unknown term function" and the
//! 1-variable `s(g, bs='re')` form was rejected because the factor-smooth
//! path demanded a numeric companion.

use csv::StringRecord;
use gam::inference::formula_dsl::{ParsedTerm, parse_formula};
use gam::pirls::PirlsStatus;
use gam::{
    FitConfig, FitResult, encode_recordswith_inferred_schema, fit_from_formula, init_parallelism,
};
use rand::SeedableRng;
use rand::rngs::StdRng;
use rand_distr::{Distribution, Normal, Uniform};

fn random_effect_name(formula: &str) -> String {
    let parsed = parse_formula(formula).unwrap_or_else(|e| panic!("{formula}: {e}"));
    parsed
        .terms
        .into_iter()
        .find_map(|term| match term {
            ParsedTerm::RandomEffect { name, .. } => Some(name),
            _ => None,
        })
        .unwrap_or_else(|| panic!("{formula} did not parse to a RandomEffect"))
}

#[test]
fn factor_and_re_idioms_parse_to_random_effect() {
    // The headline mgcv idiom for a categorical / random-intercept on a single
    // factor. Both spellings must reach the same RandomEffect lowering.
    assert_eq!(random_effect_name("y ~ s(x) + factor(group)"), "group");
    assert_eq!(random_effect_name("y ~ s(x) + s(group, bs='re')"), "group");
    assert_eq!(
        random_effect_name("y ~ s(x) + s(group, bs=\"re\")"),
        "group"
    );
    // Pre-existing aliases still work and produce the same lowering.
    assert_eq!(random_effect_name("y ~ s(x) + re(group)"), "group");
    assert_eq!(random_effect_name("y ~ s(x) + group(group)"), "group");
}

#[test]
fn factor_idiom_rejects_multivariable_call() {
    // factor()/re()/group() are single-variable terms; multi-var must error
    // (and not silently fall through to factor-smooth, which is a different
    // construct).
    let err = parse_formula("y ~ factor(a, b)").expect_err("multi-var factor must error");
    let msg = format!("{err}");
    assert!(
        msg.contains("exactly one variable"),
        "unexpected error: {msg}"
    );
}

fn synth_categorical_dataset(n: usize, seed: u64) -> gam::inference::data::EncodedDataset {
    let mut rng = StdRng::seed_from_u64(seed);
    let ux = Uniform::new(0.0_f64, 1.0_f64).expect("uniform");
    let noise = Normal::new(0.0_f64, 0.2_f64).expect("noise");
    let group_effects = [-1.0_f64, -0.3, 0.0, 0.5, 1.5];
    let n_groups = group_effects.len();

    let headers = ["x", "group", "y"]
        .into_iter()
        .map(String::from)
        .collect::<Vec<_>>();
    let mut rows: Vec<StringRecord> = Vec::with_capacity(n);
    for i in 0..n {
        let x = ux.sample(&mut rng);
        let g = i % n_groups;
        // Use string labels so the column is inferred as Categorical.
        let group = format!("g{g}");
        let y = (2.0 * std::f64::consts::PI * x).sin() + group_effects[g] + noise.sample(&mut rng);
        rows.push(StringRecord::from(vec![
            x.to_string(),
            group,
            y.to_string(),
        ]));
    }
    encode_recordswith_inferred_schema(headers, rows).expect("encode categorical dataset")
}

#[test]
fn fit_factor_group_recovers_group_means() {
    init_parallelism();
    let data = synth_categorical_dataset(200, 0);

    let cfg = FitConfig {
        family: Some("gaussian".to_string()),
        ..FitConfig::default()
    };

    // End-to-end: `factor(group)` must reach a Standard fit (no panic, no
    // "unknown term function") and PIRLS must converge.
    let result =
        fit_from_formula("y ~ s(x) + factor(group)", &data, &cfg).expect("factor(group) must fit");
    let FitResult::Standard(fit) = result else {
        panic!("expected standard fit");
    };
    assert_eq!(
        fit.fit.convergence_evidence().inner_status(),
        PirlsStatus::Converged
    );

    // `s(group, bs='re')` shorthand must take the same code path successfully.
    let result_re = fit_from_formula("y ~ s(x) + s(group, bs='re')", &data, &cfg)
        .expect("s(group, bs='re') must fit");
    let FitResult::Standard(fit_re) = result_re else {
        panic!("expected standard fit for s(group, bs='re')");
    };
    assert_eq!(
        fit_re.fit.convergence_evidence().inner_status(),
        PirlsStatus::Converged
    );
}
