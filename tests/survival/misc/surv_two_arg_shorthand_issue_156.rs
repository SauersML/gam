//! Regression for GitHub issue #156: the 2-arg `Surv(time, event)` right-
//! censored shorthand must be accepted by the formula DSL (matching the R
//! `survival::Surv(time, event)` default) and lower to a survival fit with a
//! synthetic zero entry column. Before the fix, the response parser rejected
//! 2-arg `Surv(...)` and demanded `Surv(entry, exit, event)` even for plain
//! right-censored data.

use csv::StringRecord;
use gam::inference::formula_dsl::parse_surv_response;
use gam::pirls::PirlsStatus;
use gam::{
    FitConfig, FitResult, encode_recordswith_inferred_schema, fit_from_formula, init_parallelism,
};
use rand::SeedableRng;
use rand::rngs::StdRng;
use rand_distr::{Distribution, Exp, Normal, Uniform, Weibull};

#[test]
fn surv_two_arg_response_parses_to_implicit_zero_entry() {
    // Parser-level: Surv(time, event) ≡ entry=None, exit="time", event="event".
    let parsed = parse_surv_response("Surv(time, event)")
        .expect("Surv(time, event) parse")
        .expect("Surv(time, event) is a survival response");
    assert!(
        parsed.0.is_none(),
        "Surv(time, event) shorthand must yield entry=None (synthetic zero entry); got {:?}",
        parsed.0
    );
    assert_eq!(parsed.1, "time");
    assert_eq!(parsed.2, "event");

    // Three-arg form is unchanged.
    let parsed3 = parse_surv_response("Surv(entry, exit, event)")
        .unwrap()
        .unwrap();
    assert_eq!(parsed3.0.as_deref(), Some("entry"));
    assert_eq!(parsed3.1, "exit");
    assert_eq!(parsed3.2, "event");

    // 1-arg or 4-arg Surv(...) is still rejected, with a message that mentions
    // both supported forms.
    let err = parse_surv_response("Surv(t)")
        .err()
        .expect("1-arg rejected");
    let msg = format!("{err}");
    assert!(msg.contains("Surv(time, event)"));
    assert!(msg.contains("Surv(entry, exit, event)"));
}

fn synth_right_censored_dataset(n: usize, seed: u64) -> gam::inference::data::EncodedDataset {
    let mut rng = StdRng::seed_from_u64(seed);
    let ux = Uniform::new(0.0_f64, 1.0_f64).expect("uniform");
    let weib = Weibull::new(2.0_f64, 1.5_f64).expect("weibull");
    let cens = Exp::new(0.4_f64).expect("exp censoring");
    // Add a small linear x effect so the fit actually has signal.
    let noise = Normal::new(0.0_f64, 0.05_f64).expect("noise");

    let headers = ["time", "event", "x"]
        .into_iter()
        .map(String::from)
        .collect::<Vec<_>>();
    let mut rows: Vec<StringRecord> = Vec::with_capacity(n);
    for _ in 0..n {
        let x = ux.sample(&mut rng);
        let t = weib.sample(&mut rng) * (-(0.8 * x + noise.sample(&mut rng))).exp();
        let c = cens.sample(&mut rng) + 0.1;
        let time = t.min(c);
        let event = if t <= c { 1.0_f64 } else { 0.0_f64 };
        rows.push(StringRecord::from(vec![
            time.to_string(),
            event.to_string(),
            x.to_string(),
        ]));
    }
    encode_recordswith_inferred_schema(headers, rows).expect("encode synth survival data")
}

#[test]
fn fit_surv_two_arg_shorthand_lowers_to_location_scale_with_zero_entry() {
    init_parallelism();

    let data = synth_right_censored_dataset(200, 0);
    let config = FitConfig {
        survival_likelihood: Some("location-scale".to_string()),
        noise_formula: Some("1".to_string()),
        ..FitConfig::default()
    };

    // The 2-arg Surv(time, event) form must be accepted end-to-end (parse →
    // schema resolution → survival likelihood dispatch → PIRLS) without any
    // user-provided entry column. The fit need not converge to high accuracy
    // on a small synthetic sample, but PIRLS must complete and reach a
    // converged status — i.e. the dispatch is real.
    let result = fit_from_formula("Surv(time, event) ~ x", &data, &config)
        .expect("Surv(time, event) shorthand must fit end-to-end");
    let FitResult::SurvivalLocationScale(fit) = result else {
        panic!("expected survival location-scale fit result");
    };
    assert_eq!(
        fit.fit.fit.convergence_evidence().inner_status(),
        PirlsStatus::Converged
    );
}
