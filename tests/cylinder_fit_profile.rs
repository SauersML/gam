//! Profile the cylinder GAM fit at N=10K to find the actual perf
//! bottleneck. Times each stage of fit_from_formula → materialize →
//! build_term_collection_design → PIRLS → REML so we can see which
//! segment is slow and target it specifically.

use csv::StringRecord;
use gam::{
    FitConfig, FitResult, encode_recordswith_inferred_schema, fit_from_formula, init_parallelism,
    materialize,
};
use std::f64::consts::TAU;
use std::time::Instant;

fn dataset(n: usize) -> gam::data::EncodedDataset {
    let headers = vec!["theta".to_string(), "h".to_string(), "y".to_string()];
    let rows: Vec<StringRecord> = (0..n)
        .map(|i| {
            let theta = TAU * (i as f64) / (n as f64);
            let h = -1.0 + 2.0 * ((i % 16) as f64) / 15.0;
            let y = 1.0 + 0.55 * theta.cos() - 0.25 * (2.0 * theta).sin() + 0.3 * h;
            StringRecord::from(vec![theta.to_string(), h.to_string(), y.to_string()])
        })
        .collect();
    encode_recordswith_inferred_schema(headers, rows).expect("encode")
}

fn time<T>(label: &str, f: impl FnOnce() -> T) -> T {
    let t = Instant::now();
    let r = f();
    eprintln!("[stage] {label}: {:.3} ms", t.elapsed().as_secs_f64() * 1e3);
    r
}

#[test]
fn cylinder_fit_n_10k_stages() {
    init_parallelism();
    let formula = "y ~ te(theta, h, periodic=[0], period=[6.283185307179586, None])";
    let n = 10_000;
    let data = time("dataset(10k)", || dataset(n));
    let cfg = FitConfig {
        family: Some("gaussian".to_string()),
        ..FitConfig::default()
    };

    // Stage 1: parse + materialize (formula → FitRequest)
    let mat = time("materialize", || {
        materialize(formula, &data, &cfg).expect("materialize")
    });
    let _ = mat;

    // Stage 2: full fit
    let total = Instant::now();
    let res = fit_from_formula(formula, &data, &cfg).expect("cylinder fit");
    eprintln!(
        "[stage] total fit_from_formula N={n}: {:.3} ms",
        total.elapsed().as_secs_f64() * 1e3
    );

    match res {
        FitResult::Standard(fit) => {
            eprintln!("[info] p (ncols)={}", fit.fit.beta.len());
            eprintln!(
                "[info] outer_iters={} pirls_status={:?}",
                fit.fit.outer_iterations, fit.fit.pirls_status
            );
            eprintln!(
                "[info] coef[0..5]={:?}",
                &fit.fit.beta.as_slice().unwrap()[..5.min(fit.fit.beta.len())]
            );
        }
        _ => panic!("expected standard fit"),
    }
}

#[test]
fn cylinder_fit_n_10k_repeated_for_warmup_amortization() {
    // Measure 5 sequential fits at N=10K to see if there's any one-time
    // setup cost that the first fit pays for (rayon pool, BLAS init, etc.)
    init_parallelism();
    let formula = "y ~ te(theta, h, periodic=[0], period=[6.283185307179586, None])";
    let data = dataset(10_000);
    let cfg = FitConfig {
        family: Some("gaussian".to_string()),
        ..FitConfig::default()
    };
    for i in 0..5 {
        let t = Instant::now();
        let _ = fit_from_formula(formula, &data, &cfg).expect("fit");
        eprintln!(
            "[scale] cylinder te N=10000 iter {}: {:.3} ms",
            i,
            t.elapsed().as_secs_f64() * 1e3
        );
    }
}

#[test]
fn cylinder_fit_scaling_curve() {
    // Time the same fit at N = 1k, 3k, 10k, 30k, 100k to see how it scales.
    init_parallelism();
    let formula = "y ~ te(theta, h, periodic=[0], period=[6.283185307179586, None])";
    for &n in &[1_000_usize, 3_000, 10_000, 30_000, 100_000] {
        let data = dataset(n);
        let cfg = FitConfig {
            family: Some("gaussian".to_string()),
            ..FitConfig::default()
        };
        let t = Instant::now();
        let res = fit_from_formula(formula, &data, &cfg);
        let ms = t.elapsed().as_secs_f64() * 1e3;
        let p = res
            .ok()
            .map(|r| match r {
                FitResult::Standard(f) => f.fit.beta.len(),
                _ => 0,
            })
            .unwrap_or(0);
        eprintln!("[scale] cylinder te N={n} p={p}: {ms:.3} ms");
    }
}
