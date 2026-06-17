//! #1263 repro: binomial/poisson cylinder rate-surface recovery diagnostic.
//! Run: cargo run --release --example cyl_family_repro

use csv::StringRecord;
use gam::matrix::LinearOperator;
use gam::smooth::build_term_collection_design;
use gam::{
    FitConfig, FitResult, encode_recordswith_inferred_schema, fit_from_formula, init_parallelism,
};
use ndarray::{Array1, Array2};
use std::f64::consts::TAU;

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

fn predict_matrix(n_cols: usize, columns_in_order: &[&[f64]]) -> Array2<f64> {
    let n_rows = columns_in_order[0].len();
    let mut m = Array2::<f64>::zeros((n_rows, n_cols));
    for (j, col) in columns_in_order.iter().enumerate() {
        for i in 0..n_rows {
            m[[i, j]] = col[i];
        }
    }
    m
}

fn fit_and_predict_eta(
    formula: &str,
    data: &gam::data::EncodedDataset,
    cfg: &FitConfig,
    test_rows: &Array2<f64>,
) -> Array1<f64> {
    let result = fit_from_formula(formula, data, cfg).expect("fit succeeded");
    let FitResult::Standard(fit) = result else {
        panic!("expected standard fit");
    };
    eprintln!(
        "  beta(len={}) min={:.4} max={:.4} mean={:.4}",
        fit.fit.beta.len(),
        fit.fit.beta.iter().cloned().fold(f64::INFINITY, f64::min),
        fit.fit
            .beta
            .iter()
            .cloned()
            .fold(f64::NEG_INFINITY, f64::max),
        fit.fit.beta.iter().sum::<f64>() / fit.fit.beta.len() as f64,
    );
    eprintln!("  lambdas={:?}", fit.fit.lambdas);
    let test_design = build_term_collection_design(test_rows.view(), &fit.resolvedspec)
        .expect("rebuild prediction design");
    test_design.design.apply(&fit.fit.beta)
}

fn logistic(x: f64) -> f64 {
    1.0 / (1.0 + (-x).exp())
}
fn mse(a: &Array1<f64>, b: &Array1<f64>) -> f64 {
    a.iter()
        .zip(b.iter())
        .map(|(p, t)| (p - t).powi(2))
        .sum::<f64>()
        / a.len() as f64
}

fn main() {
    init_parallelism();
    // ---- binomial cylinder ----
    let n_theta = 30usize;
    let n_h = 20usize;
    let n = n_theta * n_h;
    let mut rng = Lcg::new(0xB1B1_C0C0_u64);
    let (mut theta, mut h, mut p_true, mut y) = (vec![], vec![], vec![], vec![]);
    for i in 0..n_theta {
        let t = TAU * (i as f64) / (n_theta as f64);
        for j in 0..n_h {
            let hh = -1.0 + 2.0 * (j as f64) / ((n_h - 1) as f64);
            let eta = 0.55 * t.cos() - 0.25 * (2.0 * t).sin() + 0.3 * hh;
            let p = logistic(eta);
            let u = rng.next_unit();
            theta.push(t);
            h.push(hh);
            p_true.push(p);
            y.push(if u < p { 1.0 } else { 0.0 });
        }
    }
    let data = encode_columns(&["theta", "h", "y"], &[&theta, &h, &y]);
    let test = predict_matrix(3, &[&theta, &h, &vec![0.0; n]]);
    let p_true_arr = Array1::from(p_true.clone());
    let bcfg = FitConfig {
        family: Some("binomial".to_string()),
        ..FitConfig::default()
    };
    eprintln!("== BINOMIAL te k=4 ==");
    let eta_pred = fit_and_predict_eta(
        "y ~ te(theta, h, periodic=[0], period=[6.283185307179586, None], k=4)",
        &data,
        &bcfg,
        &test,
    );
    let p_pred = eta_pred.mapv(logistic);
    eprintln!(
        "  eta_pred min={:.3} max={:.3} | MSE(p)={:.4e} (tol 0.02)",
        eta_pred.iter().cloned().fold(f64::INFINITY, f64::min),
        eta_pred.iter().cloned().fold(f64::NEG_INFINITY, f64::max),
        mse(&p_pred, &p_true_arr)
    );

    // control: same data, GAUSSIAN family fit to y (just to see surface shape)
    eprintln!("== BINOMIAL-data GAUSSIAN te k=4 (control) ==");
    let gcfg = FitConfig {
        family: Some("gaussian".to_string()),
        ..FitConfig::default()
    };
    let _ = fit_and_predict_eta(
        "y ~ te(theta, h, periodic=[0], period=[6.283185307179586, None], k=4)",
        &data,
        &gcfg,
        &test,
    );

    // control: binomial k=8 (more capacity)
    eprintln!("== BINOMIAL te k=8 (more capacity) ==");
    let eta8 = fit_and_predict_eta(
        "y ~ te(theta, h, periodic=[0], period=[6.283185307179586, None], k=8)",
        &data,
        &bcfg,
        &test,
    );
    eprintln!(
        "  MSE(p) k=8 = {:.4e}",
        mse(&eta8.mapv(logistic), &p_true_arr)
    );

    // ---- poisson cylinder ----
    let mut rng = Lcg::new(0xDEADC0DE_u64);
    let (mut theta, mut h, mut rate_true, mut y) = (vec![], vec![], vec![], vec![]);
    for i in 0..n_theta {
        let t = TAU * (i as f64) / (n_theta as f64);
        for j in 0..n_h {
            let hh = -1.0 + 2.0 * (j as f64) / ((n_h - 1) as f64);
            let eta = 1.5 + 0.4 * t.cos() + 0.2 * hh;
            let lam = eta.exp();
            let mut k = 0u32;
            let mut s = 0.0_f64;
            loop {
                s += -rng.next_unit().ln();
                if s > lam {
                    break;
                }
                k += 1;
                if k > 100 {
                    break;
                }
            }
            theta.push(t);
            h.push(hh);
            rate_true.push(lam);
            y.push(k as f64);
        }
    }
    let data = encode_columns(&["theta", "h", "y"], &[&theta, &h, &y]);
    let test = predict_matrix(3, &[&theta, &h, &vec![0.0; n]]);
    let rate_true_arr = Array1::from(rate_true.clone());
    let pcfg = FitConfig {
        family: Some("poisson".to_string()),
        ..FitConfig::default()
    };
    eprintln!("== POISSON te k=4 ==");
    let eta_pred = fit_and_predict_eta(
        "y ~ te(theta, h, periodic=[0], period=[6.283185307179586, None], k=4)",
        &data,
        &pcfg,
        &test,
    );
    let rate_pred = eta_pred.mapv(f64::exp);
    eprintln!(
        "  eta_pred min={:.3} max={:.3} | MSE(rate)={:.4e} (tol 0.5) | mean_rate_true={:.3}",
        eta_pred.iter().cloned().fold(f64::INFINITY, f64::min),
        eta_pred.iter().cloned().fold(f64::NEG_INFINITY, f64::max),
        mse(&rate_pred, &rate_true_arr),
        rate_true.iter().sum::<f64>() / n as f64
    );
}
