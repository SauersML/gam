//! End-to-end determinism contract for the public Rust fitting API.
//!
//! Rayon owns one process-global pool, so the parent test launches this test
//! binary afresh for every thread count.  The child prints the exact IEEE-754
//! words of the user-visible fitted coefficients, smoothing parameters, and
//! log-likelihood; comparing text therefore compares bits, not a tolerance.

use std::process::Command;

use csv::StringRecord;
use gam::{FitConfig, FitResult, encode_recordswith_inferred_schema, fit_from_formula};

fn data(kind: &str) -> gam::data::EncodedDataset {
    let headers = if kind == "survival" {
        vec!["time", "event", "x"]
    } else {
        vec!["y", "x"]
    }
    .into_iter()
    .map(str::to_owned)
    .collect();
    let rows = (0..240)
        .map(|i| {
            let x = -2.4 + 4.8 * i as f64 / 239.0;
            let wave = (2.1 * x).sin() + 0.15 * (5.3 * x).cos();
            let values = match kind {
                "gaussian" => vec![wave + 0.07 * ((i * 37 % 17) as f64 - 8.0), x],
                "binomial" => vec![
                    ((i * 53 % 101) as f64 / 101.0 < 1.0 / (1.0 + (-wave).exp())) as u8 as f64,
                    x,
                ],
                "survival" => {
                    let latent = (1.3 - 0.45 * x + 0.12 * wave).exp();
                    let censor = 5.0 + (i * 29 % 31) as f64 / 5.0;
                    vec![latent.min(censor), (latent <= censor) as u8 as f64, x]
                }
                _ => unreachable!(),
            };
            StringRecord::from(
                values
                    .into_iter()
                    .map(|v| format!("{v:.17e}"))
                    .collect::<Vec<_>>(),
            )
        })
        .collect();
    encode_recordswith_inferred_schema(headers, rows).expect("encode deterministic fixture")
}

fn words(values: impl IntoIterator<Item = f64>) -> String {
    values
        .into_iter()
        .map(|v| format!("{:016x}", v.to_bits()))
        .collect::<Vec<_>>()
        .join(",")
}

/// Fit one fixture and print the exact bit patterns of its user-visible
/// results. Each fixture is its own `#[test]` so the parent can address it by
/// name with `--exact`; run directly, it is simply a fit that must converge.
fn print_child_result(kind: &str) {
    gam::init_parallelism();
    let mut config = FitConfig::default();
    let formula = match kind {
        "gaussian" => {
            config.family = Some("gaussian".into());
            "y ~ s(x, k=10)"
        }
        "binomial" => {
            config.family = Some("binomial".into());
            "y ~ s(x, k=10)"
        }
        "survival" => {
            config.survival_likelihood = Some("transformation".into());
            config.time_basis = "ispline".into();
            config.time_num_internal_knots = 2;
            "Surv(time, event) ~ s(x, k=8) + survmodel(spec=net)"
        }
        _ => panic!("unknown child fixture"),
    };
    let result = fit_from_formula(formula, &data(kind), &config).expect("fit must converge");
    let fit = match &result {
        FitResult::Standard(result) => &result.fit,
        FitResult::SurvivalTransformation(result) => &result.fit,
        _ => panic!("unexpected fit result"),
    };
    let coefficients = fit
        .blocks
        .iter()
        .flat_map(|block| block.beta.iter().copied());
    println!(
        "RESULT coefficients={} lambdas={} log_likelihood={:016x}",
        words(coefficients),
        words(fit.lambdas.iter().copied()),
        fit.log_likelihood.to_bits()
    );
}

#[test]
fn thread_count_fit_child_gaussian() {
    print_child_result("gaussian");
}

#[test]
fn thread_count_fit_child_binomial() {
    print_child_result("binomial");
}

#[test]
fn thread_count_fit_child_survival() {
    print_child_result("survival");
}

#[test]
fn fits_are_bit_identical_across_rayon_thread_counts_and_runs() {
    let exe = std::env::current_exe().expect("current test binary");
    for kind in ["gaussian", "binomial", "survival"] {
        let mut baseline = None;
        for threads in [1, 2, 8] {
            for run in 0..2 {
                let output = Command::new(&exe)
                    .args(["--exact", &format!("thread_count_fit_child_{kind}"), "--nocapture"])
                    .env("RAYON_NUM_THREADS", threads.to_string())
                    .output()
                    .expect("launch isolated fit process");
                assert!(
                    output.status.success(),
                    "{kind}, threads={threads}, run={run}: {}",
                    String::from_utf8_lossy(&output.stderr)
                );
                let stdout = String::from_utf8(output.stdout).expect("UTF-8 child output");
                let result = stdout
                    .lines()
                    .find(|line| line.starts_with("RESULT "))
                    .expect("child result");
                match &baseline {
                    None => baseline = Some(result.to_owned()),
                    Some(expected) => assert_eq!(
                        result, expected,
                        "{kind} fit changed at threads={threads}, run={run}"
                    ),
                }
            }
        }
    }
}
