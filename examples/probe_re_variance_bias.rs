//! BUG-HUNT (random-effect variance-component selection): with FEW groups
//! (G = 5..8), does REML over- or under-estimate the random-intercept variance
//! component? The few-groups regime is exactly where REML variance estimation is
//! least stable; an over-shrink (variance collapses to 0 when it is genuinely
//! nonzero) or an over-fit (variance inflated, intercepts un-pooled) is a
//! selection pathology analogous to the #1371/#1271 corners but in the
//! variance-component (raw physical-lambda) parameterization rather than a
//! difference-penalty null space.
//!
//! True model: y_ij = mu + b_i + eps_ij,  b_i ~ N(0, tau2_true),
//! eps ~ N(0, sigma2). We fit `y ~ group(g)` (the random-intercept form) and
//! recover gam's estimate of tau2 as the sample variance of the predicted
//! per-group deviations (the standard BLUP read, same as the lme4 quality test).
//! We sweep tau2_true across {0, small, moderate, large} and report the ratio
//! tau2_hat / tau2_true (or the raw tau2_hat at tau2_true = 0).

use csv::StringRecord;
use gam::smooth::build_term_collection_design;
use gam::{
    FitConfig, FitResult, encode_recordswith_inferred_schema, fit_from_formula, init_parallelism,
    matrix::LinearOperator,
};
use ndarray::Array2;

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
        (-2.0 * u1.ln()).sqrt() * (2.0 * std::f64::consts::PI * u2).cos()
    }
}

/// Fit `y ~ group(g)` on grouped data; return gam's tau2_hat (sample variance of
/// predicted per-group deviations).
fn fit_tau2(
    n_groups: usize,
    per_group: usize,
    tau2_true: f64,
    sigma: f64,
    seed: u64,
) -> f64 {
    let mut rng = Lcg::new(seed);
    let tau = tau2_true.sqrt();
    let mut b = vec![0.0; n_groups];
    for bi in b.iter_mut() {
        *bi = tau * rng.next_normal();
    }
    let mu = 3.0;
    let mut rows: Vec<StringRecord> = Vec::new();
    let mut g_codes: Vec<f64> = Vec::new();
    for grp in 0..n_groups {
        for _ in 0..per_group {
            let y = mu + b[grp] + sigma * rng.next_normal();
            rows.push(StringRecord::from(vec![format!("g{grp}"), format!("{y}")]));
            g_codes.push(grp as f64);
        }
    }
    let headers = vec!["g".to_string(), "y".to_string()];
    let ds = encode_recordswith_inferred_schema(headers, rows).expect("encode");
    let col = ds.column_map();
    let g_idx = col["g"];
    let cfg = FitConfig {
        family: Some("gaussian".to_string()),
        ..FitConfig::default()
    };
    let res = fit_from_formula("y ~ group(g)", &ds, &cfg).expect("fit");
    let FitResult::Standard(fit) = res else {
        panic!("std");
    };
    // Predict per-group intercept at a common reference (only g varies).
    let mut anchor = Array2::<f64>::zeros((n_groups, ds.headers.len()));
    for grp in 0..n_groups {
        anchor[[grp, g_idx]] = grp as f64;
    }
    let anchor_design =
        build_term_collection_design(anchor.view(), &fit.resolvedspec).expect("anchor design");
    let pred: Vec<f64> = anchor_design.design.apply(&fit.fit.beta).to_vec();
    let mean = pred.iter().sum::<f64>() / n_groups as f64;
    let var = pred.iter().map(|v| (v - mean).powi(2)).sum::<f64>() / (n_groups as f64 - 1.0);
    var
}

fn main() {
    init_parallelism();
    let sigma = 1.0;
    let per_group = 20;
    for &n_groups in &[5usize, 6, 8] {
        for &tau2_true in &[0.0_f64, 0.25, 1.0, 4.0] {
            let mut ratios = vec![];
            for seed in 1u64..=16 {
                let t2 = fit_tau2(n_groups, per_group, tau2_true, sigma, seed);
                ratios.push(t2);
            }
            let mean = ratios.iter().sum::<f64>() / ratios.len() as f64;
            if tau2_true == 0.0 {
                println!(
                    "G={n_groups} tau2_true=0    -> tau2_hat mean={mean:.4} (should be ~0; >0 = spurious group variance)"
                );
            } else {
                println!(
                    "G={n_groups} tau2_true={tau2_true:<4} -> tau2_hat mean={mean:.4} ratio={:.3} (1.0=unbiased; <<1 over-shrink, >>1 over-fit)",
                    mean / tau2_true
                );
            }
        }
    }
}
