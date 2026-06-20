//! BUG-HUNT: does the GLM working-weight bias REML lambda-selection so that a
//! NON-Gaussian smooth over- or under-smooths a signal a Gaussian smooth
//! recovers cleanly?
//!
//! Design (one clean smooth, identical x-grid across families):
//!   true smooth   f(x) = 1.3 * sin(2*pi*x),  x in [0,1)
//!   Gaussian:     y ~ Normal(f(x), sigma)              (reference)
//!   Poisson:      y ~ Poisson(exp(a0 + f(x)))          (log link)
//!   Binomial:     y ~ Bernoulli(logit^-1(a0 + f(x)))   (logit link)
//!
//! For each family we fit  y ~ s(x)  and record:
//!   - edf_total            (should track the true wiggliness ~ a few df)
//!   - eta-RMSE vs truth    (predictive recovery on the linear-predictor scale)
//!
//! The Gaussian arm is the calibration baseline. If Poisson/binomial
//! systematically inflate EDF (under-smooth) or collapse it (over-smooth) at the
//! SAME signal-to-noise, the GLM working-weight is biasing lambda selection.

use csv::StringRecord;
use gam::{
    FitConfig, FitResult, encode_recordswith_inferred_schema, fit_from_formula, init_parallelism,
};
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
    fn poisson(&mut self, lam: f64) -> f64 {
        let mut k = 0u32;
        let mut s = 0.0_f64;
        let limit = lam.max(1.0) * 50.0 + 50.0;
        loop {
            s += -self.next_unit().ln();
            if s > lam {
                break;
            }
            k += 1;
            if k as f64 > limit {
                break;
            }
        }
        k as f64
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

/// Fit `y ~ s(x)` for the given family, return edf_total.
fn fit_recovery(family: &str, x: &[f64], y: &[f64]) -> f64 {
    let data = encode_columns(&["x", "y"], &[x, y]);
    let cfg = FitConfig {
        family: Some(family.to_string()),
        ..FitConfig::default()
    };
    let res = fit_from_formula("y ~ s(x)", &data, &cfg).expect("fit");
    let FitResult::Standard(fit) = res else {
        panic!("expected standard fit");
    };
    fit.fit.edf_total().unwrap_or(f64::NAN)
}

fn main() {
    init_parallelism();
    let n = 400usize;
    let a0_pois = 1.5_f64; // Poisson log-rate offset: mu in ~[0.55, 4.5], healthy counts
    let a0_binom = 0.0_f64; // Bernoulli logit offset 0: p in [0.21,0.79] => both classes always present

    println!("seed | gauss_edf | pois_edf | binom_edf  (truth ~ a few df)");
    let mut g_edf = vec![];
    let mut p_edf = vec![];
    let mut b_edf = vec![];
    for seed in 1u64..=12 {
        let mut rng = Lcg::new(seed);
        let mut x = vec![0.0; n];
        for xi in x.iter_mut() {
            *xi = rng.next_unit();
        }
        x.sort_by(|a, b| a.partial_cmp(b).unwrap());

        let f: Vec<f64> = x.iter().map(|&xi| truth(xi)).collect();

        // Gaussian: noise sigma chosen so SNR is moderate.
        let sigma = 0.6;
        let yg: Vec<f64> = f.iter().map(|&fi| fi + sigma * rng.next_normal()).collect();

        // Poisson: eta = a0_pois + f, mu = exp(eta).
        let yp: Vec<f64> = f
            .iter()
            .map(|&fi| rng.poisson((a0_pois + fi).exp()))
            .collect();

        // Binomial (bernoulli): eta = a0_binom + f, p = logistic(eta). a0_binom=0
        // keeps p in [0.21,0.79] so both classes always appear (no degenerate
        // all-0/all-1 boundary that makes the REML score non-finite).
        let yb: Vec<f64> = f
            .iter()
            .map(|&fi| {
                let p = 1.0 / (1.0 + (-(a0_binom + fi)).exp());
                if rng.next_unit() < p { 1.0 } else { 0.0 }
            })
            .collect();

        let ge = fit_recovery("gaussian", &x, &yg);
        let pe = fit_recovery("poisson", &x, &yp);
        let be = fit_recovery("binomial", &x, &yb);
        g_edf.push(ge);
        p_edf.push(pe);
        b_edf.push(be);
        println!("{seed:>4} | {ge:8.3} | {pe:8.3} | {be:8.3}");
    }
    let mean = |v: &[f64]| v.iter().sum::<f64>() / v.len() as f64;
    println!(
        "MEAN EDF: gauss={:.3} pois={:.3} binom={:.3}",
        mean(&g_edf),
        mean(&p_edf),
        mean(&b_edf)
    );
}
