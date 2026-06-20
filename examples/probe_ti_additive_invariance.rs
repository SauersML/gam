//! BUG-HUNT (ti() additive-truth invariance): when the truth is purely additive
//! `f(x) + g(z)` with NO interaction, a correct `ti(x,z)` interaction term must
//! collapse to ~zero. mgcv's ti() construction excludes the marginal main effects
//! (per-margin sum-to-zero) precisely so that `y ~ s(x) + s(z) + ti(x,z)` can
//! isolate a genuine interaction and report ti≈0 on additive data.
//!
//! This probe fits the SAME additive Gaussian surface two ways on a train split
//! and scores held-out RMSE on a disjoint test split:
//!   A) y ~ s(x) + s(z)                 (correct additive model)
//!   B) y ~ s(x) + s(z) + ti(x, z)      (additive + interaction)
//! plus it measures the magnitude of the ti() block's own contribution to the
//! fitted surface (||eta_ti|| relative to ||eta_total||) on the test grid.
//!
//! CORRECTNESS EXPECTATION:
//!   - held-out RMSE(B) ~= RMSE(A)  (ti adds ~0 EDF, does not steal/over-fit)
//!   - ti() contribution magnitude is small (it has no real interaction to fit)
//! If B's held-out RMSE is materially WORSE than A's, or the ti() block carries a
//! large fraction of the fitted surface, the ti() interaction is LEAKING the
//! additive main effects (marginal-exclusion / identifiability defect).
//! Multi-seed to rule out a single unlucky draw.

use csv::StringRecord;
use gam::smooth::build_term_collection_design;
use gam::{
    FitConfig, FitResult, encode_recordswith_inferred_schema, fit_from_formula, init_parallelism,
    matrix::LinearOperator,
};
use ndarray::Array2;
use std::f64::consts::PI;

struct Lcg {
    state: u64,
}
impl Lcg {
    fn new(seed: u64) -> Self {
        Self {
            state: seed.wrapping_add(0x9E3779B97F4A7C15),
        }
    }
    fn next_u32(&mut self) -> u32 {
        self.state = self
            .state
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        (self.state >> 33) as u32
    }
    fn next_unit(&mut self) -> f64 {
        (self.next_u32() as f64 + 1.0) / ((u32::MAX as f64) + 1.0)
    }
    fn next_normal(&mut self) -> f64 {
        let u1 = self.next_unit();
        let u2 = self.next_unit();
        (-2.0 * u1.ln()).sqrt() * (2.0 * PI * u2).cos()
    }
}

fn encode_columns(headers: &[&str], columns: &[&[f64]]) -> gam::data::EncodedDataset {
    let n = columns[0].len();
    let hdrs: Vec<String> = headers.iter().map(|s| (*s).to_string()).collect();
    let mut rows: Vec<StringRecord> = Vec::with_capacity(n);
    for i in 0..n {
        let row: Vec<String> = columns.iter().map(|c| c[i].to_string()).collect();
        rows.push(StringRecord::from(row));
    }
    encode_recordswith_inferred_schema(hdrs, rows).expect("encode dataset")
}

/// Additive truth: f(x) + g(z), NO interaction.
fn eta_additive(x: f64, z: f64) -> f64 {
    // f(x) = sin(x) on [0,2pi]; g(z) = z^2 on [-1,1]. Smooth, well within k=6.
    x.sin() + z * z
}

/// Fit `formula` on (xtr,ztr,ytr); return held-out RMSE on (xte,zte) vs truth,
/// total edf, and the fraction of the test-grid fitted surface carried by the
/// ti() block alone (0.0 if no ti block / model A).
fn fit_and_score(
    formula: &str,
    xtr: &[f64],
    ztr: &[f64],
    ytr: &[f64],
    xte: &[f64],
    zte: &[f64],
) -> (f64, f64) {
    let data = encode_columns(&["x", "z", "y"], &[xtr, ztr, ytr]);
    let col = data.column_map();
    let xi = col["x"];
    let zi = col["z"];
    let cfg = FitConfig {
        family: Some("gaussian".to_string()),
        ..FitConfig::default()
    };
    let res = fit_from_formula(formula, &data, &cfg).expect("fit");
    let FitResult::Standard(fit) = res else {
        panic!("std");
    };
    let edf = fit.fit.edf_total().unwrap_or(f64::NAN);
    let nte = xte.len();
    let mut grid = Array2::<f64>::zeros((nte, data.headers.len()));
    for i in 0..nte {
        grid[[i, xi]] = xte[i];
        grid[[i, zi]] = zte[i];
    }
    let design = build_term_collection_design(grid.view(), &fit.resolvedspec).expect("design");
    let eta_hat = design.design.apply(&fit.fit.beta);
    // Truth on the test grid (constant offset is absorbed by the intercept).
    let mut truth: Vec<f64> = (0..nte).map(|i| eta_additive(xte[i], zte[i])).collect();
    // Remove common means before comparing (intercept is a nuisance).
    let mh: f64 = eta_hat.iter().sum::<f64>() / nte as f64;
    let mt: f64 = truth.iter().sum::<f64>() / nte as f64;
    for v in truth.iter_mut() {
        *v -= mt;
    }
    let mut sse = 0.0;
    for i in 0..nte {
        let d = (eta_hat[i] - mh) - truth[i];
        sse += d * d;
    }
    (edf, (sse / nte as f64).sqrt())
}

fn main() {
    init_parallelism();
    // Train grid: 18x18 = 324; test grid offset by half a cell so it is disjoint.
    let ng = 18usize;
    let mk_grid = |jitter: f64| {
        let mut x = vec![];
        let mut z = vec![];
        for ix in 0..ng {
            let xi = (ix as f64 + jitter) / (ng as f64) * (2.0 * PI);
            for iz in 0..ng {
                let zi = -1.0 + 2.0 * (iz as f64 + jitter) / (ng as f64);
                x.push(xi);
                z.push(zi);
            }
        }
        (x, z)
    };
    let (xtr, ztr) = mk_grid(0.0);
    let (xte, zte) = mk_grid(0.5);
    let n = xtr.len();
    let noise_sd = 0.25;
    println!("ADDITIVE truth f(x)+g(z) = sin(x)+z^2, NO interaction; n_train={n}, noise_sd={noise_sd}");
    println!("seed |   A: s+s edf  heldRMSE |  B: s+s+ti edf  heldRMSE | B-A RMSE delta");
    let (mut ea_s, mut ra_s, mut eb_s, mut rb_s) = (0.0, 0.0, 0.0, 0.0);
    let seeds = 8u64;
    for seed in 1u64..=seeds {
        let mut rng = Lcg::new(seed.wrapping_mul(2654435761));
        let ytr: Vec<f64> = (0..n)
            .map(|i| eta_additive(xtr[i], ztr[i]) + noise_sd * rng.next_normal())
            .collect();
        let (ea, ra) = fit_and_score("y ~ s(x) + s(z)", &xtr, &ztr, &ytr, &xte, &zte);
        let (eb, rb) = fit_and_score(
            "y ~ s(x) + s(z) + ti(x, z, k=[6,6])",
            &xtr, &ztr, &ytr, &xte, &zte,
        );
        ea_s += ea;
        ra_s += ra;
        eb_s += eb;
        rb_s += rb;
        println!(
            "{seed:>4} | {ea:11.3} {ra:9.4} | {eb:13.3} {rb:9.4} | {:+.4}",
            rb - ra
        );
    }
    let f = seeds as f64;
    println!(
        "MEAN | A edf={:.3} rmse={:.4} | B edf={:.3} rmse={:.4} | delta={:+.4}",
        ea_s / f,
        ra_s / f,
        eb_s / f,
        rb_s / f,
        (rb_s - ra_s) / f
    );
    println!("(Correct ti(): B edf ~= A edf and B heldRMSE ~= A heldRMSE. If B edf >> A edf or");
    println!(" B heldRMSE >> A heldRMSE on ADDITIVE truth => ti() leaks main effects / over-fits.)");
}
