//! Rust ↔ `gamfit` parity for the Gaussian linear formula path.
//!
//! Both halves must call the SAME public entry point. `gamfit.fit` reaches the
//! estimator through `fit_materialized_standard_with_notes`, whose in-process
//! twin is `fit_from_formula` — not `fit_model`, which is the estimator switch
//! one level BELOW the dispatch that owns the exact-fit / spline-scan /
//! residual-cascade fast paths. Comparing `fit_model` against `gamfit` compares
//! two different routes and blames the difference on the FFI: that is exactly
//! what #2595 was, and the noiseless fixture below is the case that separates
//! them (`fit_model` reported `-641.1167425547742`, the formula path `0`,
//! neither of which was the fit's criterion).

use csv::StringRecord;
use gam::inference::data::{EncodedDataset, encode_recordswith_inferred_schema};
use gam::solver::fit_orchestration::{FitConfig, FitResult, StandardFitResult, fit_from_formula};
use serde_json::Value;
use std::process::Command;

const N: usize = 80;

/// `y = 0.5 + 1.25·x` (+ optional deterministic wiggle), materialized exactly as
/// the Python half below builds it so the two halves fit the same numbers.
fn fixture(noise: f64) -> EncodedDataset {
    let mut rows: Vec<StringRecord> = Vec::with_capacity(N);
    for i in 0..N {
        let x = -1.0 + 2.0 * (i as f64) / ((N - 1) as f64);
        let y = 0.5 + 1.25 * x + noise * ((i % 7) as f64 - 3.0);
        rows.push(StringRecord::from(vec![x.to_string(), y.to_string()]));
    }
    let headers = vec!["x".to_string(), "y".to_string()];
    encode_recordswith_inferred_schema(headers, rows).expect("encode")
}

fn rust_fit(noise: f64) -> StandardFitResult {
    let data = fixture(noise);
    let cfg = FitConfig {
        family: Some("gaussian".to_string()),
        ..FitConfig::default()
    };
    match fit_from_formula("y ~ x", &data, &cfg).expect("fit") {
        FitResult::Standard(fit) => fit,
        _ => panic!("expected standard fit"),
    }
}

/// Run the Python half over the identical fixture and return its summary JSON.
fn python_summary(noise: f64) -> Value {
    let py = format!(
        r#"
import json
import gamfit
noise = {noise}
rows=[]
n={N}
for i in range(n):
    x=-1.0+2.0*i/(n-1)
    y=0.5+1.25*x + noise*((i % 7) - 3.0)
    rows.append({{'x':x,'y':y}})
m=gamfit.fit(rows, 'y ~ x', family='gaussian')
s=m.summary()
out={{
 'beta': [c['estimate'] for c in s['coefficients']],
 'deviance': float(s['deviance']),
 # `reml_score` / `raw_reml_score` are `None` when the fit has NO criterion --
 # emitted as JSON null so the Rust half compares presence, not a stand-in.
 'reml': s['reml_score'],
 'raw_reml': s['raw_reml_score'],
 'reml_score_unavailable': s.get('reml_score_unavailable'),
 'null_dim': s['null_dim'],
 'null_space_logdet': s['null_space_logdet'],
}}
print(json.dumps(out))
"#
    );
    let out = Command::new("python3")
        .arg("-c")
        .arg(&py)
        // Run from a directory that contains no `./gamfit`. From the repo root
        // the source package shadows the installed wheel on `sys.path` (cwd is
        // searched first) and has no compiled `_rust`, so the import dies and
        // the parity assertions below never execute. CI works around this by
        // copying the built extension into the source tree; not depending on
        // that makes the test give the same verdict wherever it is run.
        .current_dir(std::env::temp_dir())
        .output()
        .expect("run python");
    assert!(
        out.status.success(),
        "python failed: {}",
        String::from_utf8_lossy(&out.stderr)
    );
    serde_json::from_slice(&out.stdout).expect("json")
}

fn assert_beta_parity(beta_rust: &[f64], value: &Value) {
    let beta_py: Vec<f64> = value["beta"]
        .as_array()
        .expect("summary coefficients")
        .iter()
        .map(|entry| entry.as_f64().expect("finite coefficient"))
        .collect();
    assert_eq!(beta_rust.len(), beta_py.len());
    for (index, (a, b)) in beta_rust.iter().zip(beta_py.iter()).enumerate() {
        assert!((a - b).abs() <= 1e-9, "beta[{index}] rust={a} py={b}");
    }
}

/// The ordinary case: a criterion exists on both sides and they agree, and the
/// summary headline is the documented normalization of the raw score.
#[test]
fn python_rust_ffi_parity_gaussian_linear_case() {
    let sf = rust_fit(0.01);
    let beta_rust: Vec<f64> = sf.fit.beta.to_vec();
    let ll_rust = sf.fit.log_likelihood;
    let reml_rust = sf
        .fit
        .reml_score()
        .expect("a noisy Gaussian linear fit has a REML criterion");

    let value = python_summary(0.01);
    assert_beta_parity(&beta_rust, &value);

    // `UnifiedFitResult::reml_score` is the outer optimizer's own criterion
    // value, whose Python counterpart is `raw_reml_score`. The summary's
    // headline `reml_score` is the CROSS-MODEL comparable score: raw plus the
    // rank-aware Tierney-Kadane normalizer over the penalty null space. The two
    // coincide only when that null space is empty, which a formula fit is not
    // required to produce.
    let raw_reml_py = value["raw_reml"]
        .as_f64()
        .expect("the noisy fit reports a raw criterion");
    let reml_py = value["reml"]
        .as_f64()
        .expect("the noisy fit reports a comparable criterion");
    assert!(
        (reml_rust - raw_reml_py).abs() <= 1e-9,
        "raw reml rust={reml_rust} py={raw_reml_py}"
    );
    // Pin the headline's documented relationship to the raw score as well, so a
    // normalizer that silently changes definition (or stops being applied) is a
    // failure rather than an invisible shift in what `Summary.reml_score` means.
    // A standard formula fit publishes its penalty null space, and a comparable
    // criterion exists only with it: a fit without that metadata publishes no
    // `reml_score`, never its raw criterion under the comparable name (#2627).
    let null_dim = value["null_dim"]
        .as_f64()
        .expect("a standard formula fit publishes its penalty null-space dimension");
    let expected_headline = gam::solver::topology_selector::tk_normalized_score(
        raw_reml_py,
        null_dim,
        value["null_space_logdet"].as_f64(),
        1,
        gam::solver::evidence::TopologyScoreScale::PerObservation,
    )
    .expect("summary null-space metadata must admit the TK normalizer");
    assert!(
        (reml_py - expected_headline).abs() <= 1e-9,
        "comparable reml py={reml_py} expected={expected_headline} \
         (raw={raw_reml_py}, null_dim={null_dim}, null_space_logdet={:?})",
        value["null_space_logdet"].as_f64()
    );
    assert!(
        ll_rust.is_finite(),
        "rust gaussian log-likelihood must be finite: {ll_rust}"
    );
}

/// The degenerate case #2595 was filed on: `y` is an EXACT affine function of
/// `x`, so `φ̂ = 0` and the profiled restricted likelihood is unbounded. Both
/// halves must say the same thing — that the fit has no criterion — rather than
/// one reporting a roundoff-determined number and the other a placeholder zero.
#[test]
fn python_rust_ffi_parity_gaussian_exact_fit_has_no_criterion() {
    let sf = rust_fit(0.0);
    let beta_rust: Vec<f64> = sf.fit.beta.to_vec();
    assert_eq!(
        sf.fit.reml_score(),
        None,
        "an exactly-interpolating Gaussian fit must report NO criterion, not a value"
    );
    assert_eq!(
        sf.fit.penalized_objective(),
        None,
        "the objective must be absent together with the criterion"
    );
    // The fit itself is real and exact: coefficients, zero deviance, zero scale.
    assert!(sf.fit.deviance == 0.0, "deviance={}", sf.fit.deviance);
    assert!(
        sf.fit.standard_deviation == 0.0,
        "sigma={}",
        sf.fit.standard_deviation
    );

    let value = python_summary(0.0);
    assert_beta_parity(&beta_rust, &value);
    assert!(
        value["raw_reml"].is_null(),
        "Summary.raw_reml_score must be null on an exact fit, got {}",
        value["raw_reml"]
    );
    assert!(
        value["reml"].is_null(),
        "Summary.reml_score must be null on an exact fit, got {}",
        value["reml"]
    );
    // The absence never travels without its reason.
    let reason = value["reml_score_unavailable"]
        .as_str()
        .expect("an absent criterion must carry its explanation");
    assert!(
        reason.contains("no REML/LAML criterion"),
        "unexpected explanation: {reason}"
    );
}
