//! BUG-HUNT: Poisson log-link offset correctness probe (known truth).
//!
//! True model: y_i ~ Poisson(mu_i), log(mu_i) = log(E_i) + b0 + b1*x_i,
//! where E_i ("exposure") is a known per-row offset on the log scale.
//! A correct GLM with offset must recover (b0, b1) regardless of E_i, because
//! the offset enters eta additively and is NOT a free parameter.
//!
//! We fit the SAME data three ways and compare the recovered LINEAR (parametric)
//! coefficients:
//!   (A) offset supplied via offset_column = log(E)        -> must recover (b0,b1)
//!   (B) no offset, but data generated with E_i == 1        -> reference truth
//!   (C) offset folded into the response by NOT supplying it (wrong on purpose)
//!
//! If (A) does not recover (b0,b1) to within Monte-Carlo error, offset handling
//! in the Poisson IRLS / design path is broken.

use csv::StringRecord;
use gam::{
    FitConfig, FitResult, encode_recordswith_inferred_schema, fit_from_formula, init_parallelism,
};

struct Lcg {
    state: u64,
}
impl Lcg {
    fn new(seed: u64) -> Self {
        Self { state: seed.wrapping_add(0x9E3779B97F4A7C15) }
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
    fn poisson(&mut self, lam: f64) -> f64 {
        // Knuth
        let mut k = 0u32;
        let mut s = 0.0_f64;
        loop {
            s += -self.next_unit().ln();
            if s > lam {
                break;
            }
            k += 1;
            if k > 100_000 {
                break;
            }
        }
        k as f64
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

fn linear_coeffs(formula: &str, data: &gam::data::EncodedDataset, cfg: &FitConfig) -> (f64, f64) {
    let result = fit_from_formula(formula, data, cfg).expect("fit");
    let FitResult::Standard(fit) = result else {
        panic!("expected standard fit");
    };
    let beta = &fit.fit.beta;
    eprintln!("  beta = {:?}", beta.to_vec());
    // Parametric design: intercept + x => beta[0], beta[1]
    (beta[0], beta[1])
}

fn main() {
    init_parallelism();
    let n = 4000usize;
    let b0_true = 0.3_f64;
    let b1_true = 0.8_f64;

    let mut rng = Lcg::new(0xA11CE_u64);
    let mut x = Vec::with_capacity(n);
    let mut log_e = Vec::with_capacity(n);
    let mut y_with_offset = Vec::with_capacity(n);
    let mut y_no_offset = Vec::with_capacity(n);

    // separate rng stream for the no-offset reference so noise differs but the
    // estimator targets the same (b0,b1)
    let mut rng2 = Lcg::new(0xB0B_u64);

    for i in 0..n {
        let xi = -1.0 + 2.0 * (i as f64) / ((n - 1) as f64);
        // Known offset: exposure varies a lot across rows (log E in [-1.5, 1.5]).
        let le = -1.5 + 3.0 * rng.next_unit();
        let eta = b0_true + b1_true * xi;
        let mu_off = (le + eta).exp(); // with exposure
        let mu_ref = eta.exp(); // E == 1
        x.push(xi);
        log_e.push(le);
        y_with_offset.push(rng.poisson(mu_off));
        y_no_offset.push(rng2.poisson(mu_ref));
    }

    let pcfg = FitConfig { family: Some("poisson".to_string()), ..FitConfig::default() };

    // (B) reference: no offset, E==1 data
    eprintln!("== (B) reference no-offset (E==1) ==");
    let data_ref = encode_columns(&["x", "y"], &[&x, &y_no_offset]);
    let (b0_ref, b1_ref) = linear_coeffs("y ~ x", &data_ref, &pcfg);

    // (A) offset supplied: must recover (b0,b1)
    eprintln!("== (A) offset supplied (offset_column=log_e) ==");
    let data_off = encode_columns(&["x", "log_e", "y"], &[&x, &log_e, &y_with_offset]);
    let acfg = FitConfig {
        family: Some("poisson".to_string()),
        offset_column: Some("log_e".to_string()),
        ..FitConfig::default()
    };
    let (b0_a, b1_a) = linear_coeffs("y ~ x", &data_off, &acfg);

    // (C) WRONG: same exposure data but offset NOT supplied -> intercept absorbs
    // mean(log_e); slope may still be ~ b1 but intercept must be biased.
    eprintln!("== (C) offset OMITTED (control, should be biased) ==");
    let data_c = encode_columns(&["x", "y"], &[&x, &y_with_offset]);
    let (b0_c, b1_c) = linear_coeffs("y ~ x", &data_c, &pcfg);

    eprintln!("\n==== RESULTS (true b0={b0_true}, b1={b1_true}) ====");
    eprintln!("(B) ref     : b0={b0_ref:+.4}  b1={b1_ref:+.4}");
    eprintln!("(A) offset  : b0={b0_a:+.4}  b1={b1_a:+.4}   <-- MUST match truth");
    eprintln!("(C) omitted : b0={b0_c:+.4}  b1={b1_c:+.4}   (intercept ~ b0 + mean(log_e))");
    eprintln!(
        "    mean(log_e)={:.4}",
        log_e.iter().sum::<f64>() / n as f64
    );

    // SE-scale tolerance: Poisson b1 SE ~ 1/sqrt(n * Var(x) * mean_mu) is tiny here.
    // Allow 0.05 absolute (very generous for n=4000).
    let tol = 0.05_f64;
    let ok_a = (b0_a - b0_true).abs() < tol && (b1_a - b1_true).abs() < tol;
    eprintln!(
        "\nVERDICT (A offset recovers truth within {tol}): {}",
        if ok_a { "PASS" } else { "FAIL <-- BUG" }
    );
    assert!(
        ok_a,
        "offset recovery FAILED: (A) supplied-offset fit did not recover truth \
         (b0={b0_a:+.4}, b1={b1_a:+.4}; true b0={b0_true}, b1={b1_true}) within tol={tol}"
    );
}
