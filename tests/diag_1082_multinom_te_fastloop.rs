//! FAST-ITERATION repro for the #1082 multinomial te-tensor separable grind
//! (`quality_vs_synthetic_multinomial_deviance_identity`). Standalone single-file
//! test target so an edit recompiles only the lib + this file, NOT the 167-module
//! `quality` crate root. Mirrors the quality test's deterministic DGP + formula +
//! fit, printing outer/inner counts, accuracy/log-loss, and wall time. NOT a
//! correctness gate — a diagnostic harness. Delete before merge.

use gam::families::multinomial::{fit_penalized_multinomial_formula, predict_multinomial_formula};
use gam::{FitConfig, EncodedDataset, encode_recordswith_inferred_schema, init_parallelism};
use csv::StringRecord;
use ndarray::Array2;
use std::f64::consts::PI;
use std::time::Instant;

struct Obs {
    x1: f64,
    x2: f64,
    label: String,
}

fn make_observations(n: usize) -> Vec<Obs> {
    let stride1 = (2.0_f64).sqrt().fract();
    let stride2 = (3.0_f64).sqrt().fract();
    let mut u1 = 0.12_f64;
    let mut u2 = 0.37_f64;
    let mut obs = Vec::with_capacity(n);
    for _ in 0..n {
        u1 = (u1 + stride1).fract();
        u2 = (u2 + stride2).fract();
        let a = 2.0 * PI * u1;
        let b = -3.0 + 6.0 * u2;
        let l0 = 1.5 * a.sin();
        let l1 = -0.8 * a.cos() * b;
        let l2 = 0.0;
        let label = if l0 >= l1 && l0 >= l2 {
            "A"
        } else if l1 >= l0 && l1 >= l2 {
            "B"
        } else {
            "C"
        };
        obs.push(Obs { x1: a, x2: b, label: label.to_string() });
    }
    obs
}

fn encode(obs: &[Obs]) -> EncodedDataset {
    let headers = ["x1", "x2", "y"].into_iter().map(String::from).collect();
    let rows: Vec<StringRecord> = obs
        .iter()
        .map(|o| {
            StringRecord::from(vec![
                format!("{:.17e}", o.x1),
                format!("{:.17e}", o.x2),
                o.label.clone(),
            ])
        })
        .collect();
    encode_recordswith_inferred_schema(headers, rows).expect("encode")
}

fn accuracy(probs: &Array2<f64>, idx: &[usize]) -> f64 {
    let mut ok = 0usize;
    for (i, &c) in idx.iter().enumerate() {
        let mut best = 0usize;
        for k in 1..probs.ncols() {
            if probs[[i, k]] > probs[[i, best]] {
                best = k;
            }
        }
        if best == c {
            ok += 1;
        }
    }
    ok as f64 / idx.len() as f64
}

fn log_loss(probs: &Array2<f64>, idx: &[usize]) -> f64 {
    let mut s = 0.0;
    for (i, &c) in idx.iter().enumerate() {
        s += -(probs[[i, c]].max(1e-12)).ln();
    }
    s / idx.len() as f64
}

#[test]
fn diag_multinom_te_fastloop() {
    init_parallelism();
    let n = 400;
    let obs = make_observations(n);
    let mut train = Vec::new();
    let mut test = Vec::new();
    for (i, o) in obs.into_iter().enumerate() {
        if i % 10 < 3 {
            test.push(o);
        } else {
            train.push(o);
        }
    }
    let ds_train = encode(&train);
    let ds_test = encode(&test);
    let formula = "y ~ s(x1, bs='cc', k=8) + s(x2, bs='tp', k=5) + te(x1, x2, bs=c('cc','tp'))";
    let cfg = FitConfig::default();
    let t0 = Instant::now();
    let model = fit_penalized_multinomial_formula(&ds_train, formula, &cfg, 1.0, 50, 1e-7)
        .expect("fit");
    let secs = t0.elapsed().as_secs_f64();
    let ci = |lbl: &str| model.class_levels.iter().position(|l| l == lbl).unwrap();
    let probs = predict_multinomial_formula(&model, &ds_test).expect("predict");
    let idx: Vec<usize> = test.iter().map(|o| ci(&o.label)).collect();
    eprintln!(
        "DIAGFAST fit_secs={:.1} converged={} iters={} acc={:.4} logloss={:.4}",
        secs,
        model.converged,
        model.iterations,
        accuracy(&probs, &idx),
        log_loss(&probs, &idx)
    );
}
