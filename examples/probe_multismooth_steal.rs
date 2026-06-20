//! BUG-HUNT (axis 5: multi-smooth coupling / signal-stealing): in
//! `y ~ s(x) + s(z)` where the smooth signal lives ENTIRELY in x and z is a
//! confounder correlated with x but carrying NO independent signal, does the
//! JOINT REML λ-selection:
//!   (a) correctly attribute the signal to s(x) and shrink s(z) toward its null
//!       space (~1 EDF), OR
//!   (b) STEAL the signal into the noise covariate s(z) — collapsing the true
//!       s(x) to a flat line (EDF 1) while loading s(z) with many EDF?
//!
//! Crucially we DISTINGUISH a benign relabel from a real wrong-fit by scoring
//! HELD-OUT predictive RMSE against the TRUE surface f(x) = 1.3·sin(2πx):
//!   - benign: even if labels flip, fitted ŷ tracks the truth on test data
//!     (low RMSE) because z≈x carries the same information;
//!   - BUG: s(z) at high EDF over-fits the TRAINING noise through z and the
//!     prediction degrades on held-out data (RMSE blows up) — an
//!     anti-conservative selection that ships a non-generalizing fit.
//!
//! (x, z) are drawn jointly Gaussian with target correlation rho via
//! z = rho·x_std + sqrt(1-rho²)·e, then mapped to [0,1]. The signal is a
//! function of x ONLY, so any predictive degradation as rho rises is a coupling
//! defect, not information loss.

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

fn truth(x: f64) -> f64 {
    1.3 * (2.0 * PI * x).sin()
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

fn term_edf(fit: &gam::StandardFitResult, needle: &str) -> f64 {
    let design = &fit.design;
    let unified = &fit.fit;
    let mut penalty_cursor = 0usize;
    for _ in &design.random_effect_ranges {
        penalty_cursor += 1;
    }
    for term in &design.smooth.terms {
        let k = term.penalties_local.len();
        if term.name.contains(needle) {
            return unified.per_term_edf(term.coeff_range.clone(), penalty_cursor, k);
        }
        penalty_cursor += k;
    }
    f64::NAN
}

/// Standard normal CDF (Abramowitz–Stegun 7.1.26 erf) → map a normal to U(0,1).
fn norm_cdf(v: f64) -> f64 {
    // erf via A&S; phi(v) = 0.5*(1+erf(v/sqrt2)).
    let z = v / std::f64::consts::SQRT_2;
    let t = 1.0 / (1.0 + 0.3275911 * z.abs());
    let y = 1.0
        - (((((1.061405429 * t - 1.453152027) * t) + 1.421413741) * t - 0.284496736) * t
            + 0.254829592)
            * t
            * (-z * z).exp();
    let erf = if z >= 0.0 { y } else { -y };
    0.5 * (1.0 + erf)
}

fn main() {
    init_parallelism();
    let n = 400usize;
    let n_test = 500usize;
    let cfg = FitConfig {
        family: Some("gaussian".to_string()),
        ..FitConfig::default()
    };

    // Fixed test grid in x (uniform on [0,1]); z on test set is irrelevant to the
    // truth, so set z_test = x_test (the strongest case for "z is a valid proxy").
    let x_test: Vec<f64> = (0..n_test)
        .map(|i| (i as f64 + 0.5) / n_test as f64)
        .collect();
    let f_test: Vec<f64> = x_test.iter().map(|&x| truth(x)).collect();

    for &rho in &[0.0_f64, 0.6, 0.9, 0.98] {
        println!("=== x-z correlation rho = {rho} ===");
        println!(
            "seed | edf_s(x) | edf_s(z) | train_edf | TEST_RMSE  (truth lives in x; z carries no independent signal)"
        );
        let mut steals = 0usize;
        let mut test_rmses = vec![];
        for seed in 1u64..=16 {
            let mut rng = Lcg::new(seed);
            let mut x = vec![0.0; n];
            let mut z = vec![0.0; n];
            for i in 0..n {
                // Jointly-Gaussian (x_raw, z_raw) with correlation rho, then
                // probability-integral-transform each to U(0,1) so both covariates
                // share the SAME marginal as the test grid.
                let ex = rng.next_normal();
                let e = rng.next_normal();
                let zr = rho * ex + (1.0 - rho * rho).sqrt() * e;
                x[i] = norm_cdf(ex);
                z[i] = norm_cdf(zr);
            }
            let sigma = 0.5;
            let y: Vec<f64> = (0..n)
                .map(|i| truth(x[i]) + sigma * rng.next_normal())
                .collect();
            let data = encode_columns(&["x", "z", "y"], &[&x, &z, &y]);
            let res = fit_from_formula("y ~ s(x) + s(z)", &data, &cfg).expect("fit");
            let FitResult::Standard(fit) = res else {
                panic!("std");
            };
            let edf = fit.fit.edf_total().unwrap_or(f64::NAN);
            let ex_edf = term_edf(&fit, "x");
            let ez_edf = term_edf(&fit, "z");

            // Predict on the held-out grid with z_test = x_test (z is a perfect
            // proxy on the test set, so a benign relabel still predicts well).
            let col = data.column_map();
            let xi = col["x"];
            let zi = col["z"];
            let mut anchor = Array2::<f64>::zeros((n_test, data.headers.len()));
            for r in 0..n_test {
                anchor[[r, xi]] = x_test[r];
                anchor[[r, zi]] = x_test[r];
            }
            let adesign = build_term_collection_design(anchor.view(), &fit.resolvedspec)
                .expect("anchor design");
            let pred: Vec<f64> = adesign.design.apply(&fit.fit.beta).to_vec();
            // Shape RMSE: remove the common mean offset (intercept).
            let mp: f64 = pred.iter().sum::<f64>() / n_test as f64;
            let mf: f64 = f_test.iter().sum::<f64>() / n_test as f64;
            let mut sse = 0.0;
            for r in 0..n_test {
                let d = (pred[r] - mp) - (f_test[r] - mf);
                sse += d * d;
            }
            let test_rmse = (sse / n_test as f64).sqrt();
            test_rmses.push(test_rmse);
            let stole = ex_edf < 2.0 && ez_edf > 3.0;
            if stole {
                steals += 1;
            }
            let flag = if stole { " <-- STEAL" } else { "" };
            println!("{seed:>4} | {ex_edf:8.3} | {ez_edf:8.3} | {edf:9.3} | {test_rmse:9.4}{flag}");
        }
        let mean_rmse = test_rmses.iter().sum::<f64>() / test_rmses.len() as f64;
        let max_rmse = test_rmses.iter().cloned().fold(0.0_f64, f64::max);
        println!(
            "  -> steals={steals}/16  TEST_RMSE mean={mean_rmse:.4} max={max_rmse:.4}  (a benign relabel keeps RMSE low; over-fit steal blows it up)\n"
        );
    }
}
