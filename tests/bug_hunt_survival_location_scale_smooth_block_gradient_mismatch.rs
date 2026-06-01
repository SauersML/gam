//! Regression for the specific error in #512: when no `noise_formula` is
//! supplied, `materialize_survival` used to fall through to
//! `log_sigmaspec = termspec.clone()` for the survival location-scale mode,
//! duplicating every threshold term onto the log-σ block. For a smooth
//! `s(x)` on the mean that was structurally fatal — the canonical-gauge
//! identifiability audit attributed/dropped every log-σ column (per the
//! descending priorities time=200 > threshold=150 > log_sigma=120, #366),
//! leaving the solver's `ParameterBlockSpec` design at width 0 while the
//! family kept the un-audited `x_log_sigma` at the smooth's width.
//! `SurvivalLocationScaleFamily::exact_newton_joint_gradient_evaluation`
//! then errored "joint gradient length mismatch for block 2: got <smooth
//! width>, expected 0" on every REML startup seed and refused the fit.
//!
//! The fix routes the no-`noise_formula` default through the same empty
//! `TermCollectionSpec` every other survival mode uses, so the
//! `infer_non_intercept_start_design`/`design_column_tail` contract yields a
//! zero-column `x_log_sigma` that matches the spec by construction. This
//! test pins the regression: `Surv(entry, exit, event) ~ s(x)` with the
//! default location-scale likelihood must NOT produce the
//! "joint gradient length mismatch" error. (A second, unrelated
//! inner-Newton convergence defect that surfaces after this fix on the
//! same input is tracked separately; this test scopes to #512.)

use csv::StringRecord;
use gam::{FitConfig, encode_recordswith_inferred_schema, fit_from_formula};

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

#[test]
fn surv_smooth_location_scale_default_does_not_trip_log_sigma_block_gradient_mismatch() {
    let data = build_dataset(600);
    let config = FitConfig {
        survival_likelihood: "location-scale".to_string(),
        // Crucially, leave `noise_formula = None` — this is the documented
        // default mode the issue tracks (`gamfit.fit(df, "Surv(...) ~
        // s(x)")`). The pre-fix code path defaulted to
        // `log_sigmaspec = termspec.clone()` here, which produced the
        // structural mismatch the regression test below pins against.
        ..FitConfig::default()
    };
    let outcome = fit_from_formula("Surv(entry, exit, event) ~ s(x, k=10)", &data, &config);
    if let Err(err) = outcome {
        let message = err.to_string();
        assert!(
            !message.contains("joint gradient length mismatch"),
            "the #512 regression surfaced: {message}"
        );
        assert!(
            !message.contains("SurvivalLocationScaleFamily joint gradient"),
            "the #512 regression surfaced (legacy phrasing): {message}"
        );
    }
}
