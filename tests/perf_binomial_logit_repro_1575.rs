//! Measurement repro for the binomial-logit REML slowdown (#1575/#1727/#1762).
//!
//! #1575: a 3-smooth binomial-logit GAM runs ~150 outer cost evals, each a full
//! n-sized inner P-IRLS to 1e-10, making the fit ~100-160x slower than mgcv.
//! #1762: a near-perfect-separation single smooth stalls the ARC outer optimizer
//! in a flat valley and reports NON-CONVERGED after ~117s.
//!
//! These are diagnostic measurement guards (printing the #1575 cost counters
//! surfaced on the fit) plus loose regression assertions on outer work.

use gam::{FitConfig, FitResult, encode_recordswith_inferred_schema, fit_from_formula, init_parallelism};
use csv::StringRecord;
use rand::SeedableRng;
use rand::rngs::StdRng;
use rand_distr::{Distribution, Uniform};
use std::f64::consts::PI;
use std::time::Instant;

fn logistic(f: f64) -> f64 {
    1.0 / (1.0 + (-f).exp())
}

/// #1575 dataset: y ~ s(x1)+s(x2)+s(x3), binomial logit, mild signal.
fn binomial_three_smooth_records(n: usize, seed: u64) -> (Vec<String>, Vec<StringRecord>) {
    let mut rng = StdRng::seed_from_u64(seed);
    let u = Uniform::new(0.0_f64, 1.0).expect("uniform");
    let rows: Vec<StringRecord> = (0..n)
        .map(|_| {
            let x1 = u.sample(&mut rng);
            let x2 = u.sample(&mut rng);
            let x3 = u.sample(&mut rng);
            let f = (2.0 * PI * x1).sin() * 1.5 + (x2 - 0.5).powi(2) * 6.0 - 1.0
                + (3.0 * PI * x3).cos();
            let p = logistic(f);
            let y = if u.sample(&mut rng) < p { 1.0 } else { 0.0 };
            StringRecord::from(vec![
                y.to_string(),
                x1.to_string(),
                x2.to_string(),
                x3.to_string(),
            ])
        })
        .collect();
    let headers = ["y", "x1", "x2", "x3"].into_iter().map(String::from).collect();
    (headers, rows)
}

/// #1762 dataset: near-perfect separation, y ~ s(x), eta = 12 x on x in U(-1,1).
fn binomial_separation_records(n: usize, seed: u64) -> (Vec<String>, Vec<StringRecord>) {
    let mut rng = StdRng::seed_from_u64(seed);
    let u = Uniform::new(-1.0_f64, 1.0).expect("uniform");
    let p01 = Uniform::new(0.0_f64, 1.0).expect("uniform01");
    let rows: Vec<StringRecord> = (0..n)
        .map(|_| {
            let x = u.sample(&mut rng);
            let p = logistic(12.0 * x);
            let y = if p01.sample(&mut rng) < p { 1.0 } else { 0.0 };
            StringRecord::from(vec![y.to_string(), x.to_string()])
        })
        .collect();
    let headers = ["y", "x"].into_iter().map(String::from).collect();
    (headers, rows)
}

fn binomial_cfg() -> FitConfig {
    FitConfig {
        family: Some("binomial".to_string()),
        ..FitConfig::default()
    }
}

#[test]
fn binomial_three_smooth_outer_work_1575() {
    init_parallelism();
    let (headers, rows) = binomial_three_smooth_records(2000, 1575);
    let ds = encode_recordswith_inferred_schema(headers, rows).expect("encode binomial dataset");
    let cfg = binomial_cfg();

    let t0 = Instant::now();
    let result = fit_from_formula("y ~ s(x1) + s(x2) + s(x3)", &ds, &cfg).expect("gam binomial fit");
    let elapsed = t0.elapsed();
    let FitResult::Standard(fit) = result else {
        panic!("expected a standard GAM fit for binomial s(x1)+s(x2)+s(x3)");
    };

    eprintln!(
        "[#1575 repro] n=2000 outer_iterations={} outer_cost_evals={} inner_pirls_solves={} \
         outer_converged={} grad_norm={} wall={:.2}s",
        fit.fit.outer_iterations,
        fit.fit.outer_cost_evals,
        fit.fit.inner_pirls_solves,
        fit.fit.outer_converged,
        fit.fit.outer_gradient_norm.map_or("none".to_string(), |g| format!("{g:.3e}")),
        elapsed.as_secs_f64(),
    );
    assert!(fit.fit.outer_converged, "#1575: 3-smooth binomial must converge");
}

#[test]
fn binomial_separation_flat_valley_1762() {
    init_parallelism();
    let (headers, rows) = binomial_separation_records(3200, 7);
    let ds = encode_recordswith_inferred_schema(headers, rows).expect("encode separation dataset");
    let cfg = binomial_cfg();

    let t0 = Instant::now();
    let result = fit_from_formula("y ~ s(x)", &ds, &cfg).expect("gam binomial separation fit");
    let elapsed = t0.elapsed();
    let FitResult::Standard(fit) = result else {
        panic!("expected a standard GAM fit for binomial s(x)");
    };

    eprintln!(
        "[#1762 repro] n=3200 separation outer_iterations={} outer_cost_evals={} \
         inner_pirls_solves={} outer_converged={} grad_norm={} wall={:.2}s",
        fit.fit.outer_iterations,
        fit.fit.outer_cost_evals,
        fit.fit.inner_pirls_solves,
        fit.fit.outer_converged,
        fit.fit.outer_gradient_norm.map_or("none".to_string(), |g| format!("{g:.3e}")),
        elapsed.as_secs_f64(),
    );
}
