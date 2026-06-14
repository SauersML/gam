//! #1041 DIAGNOSTIC (temporary): sweep the measure-jet representer length-scale
//! on the bernoulli-marginal-slope marginal surface to localize the accuracy
//! deficit. The revert (97703771f) found the deficit is the FROZEN Gaussian
//! kernel length-scale (ell = 2x median-spacing, over-smooth) — which the
//! (s,alpha,lntau) dials do not touch — not the dials. This probe fits the same
//! BMS marginal surface at several explicit `length_scale=` values plus matern,
//! prints held-out marginal-probability RMSE-vs-truth, and asserts only that
//! the run completed (the numbers are the deliverable). Deleted once the auto
//! length-scale factor is settled.

use gam::families::bms::BernoulliMarginalSlopeFitResult;
use gam::matrix::LinearOperator;
use gam::smooth::build_term_collection_design;
use gam::{FitConfig, FitResult, encode_recordswith_inferred_schema, fit_from_formula};
use ndarray::Array2;

const N_TRAIN: usize = 1_500;
const N_TEST: usize = 600;
const CENTERS: usize = 10;

struct SplitMix64 {
    state: u64,
}
impl SplitMix64 {
    fn new(seed: u64) -> Self {
        Self { state: seed }
    }
    fn next_u64(&mut self) -> u64 {
        self.state = self.state.wrapping_add(0x9E37_79B9_7F4A_7C15);
        let mut z = self.state;
        z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
        z ^ (z >> 31)
    }
    fn next_unit(&mut self) -> f64 {
        (self.next_u64() >> 11) as f64 / (1u64 << 53) as f64
    }
    fn next_normal(&mut self) -> f64 {
        let u1 = self.next_unit().max(1e-12);
        let u2 = self.next_unit();
        (-2.0 * u1.ln()).sqrt() * (std::f64::consts::TAU * u2).cos()
    }
}

fn normal_cdf(x: f64) -> f64 {
    0.5 * (1.0 + statrs::function::erf::erf(x / std::f64::consts::SQRT_2))
}
fn beta_true(x1: f64) -> f64 {
    0.2 + 0.9 * x1
}
fn alpha_true(x1: f64, x2: f64) -> f64 {
    -0.2 + 0.7 * (std::f64::consts::PI * x1).sin() + 0.3 * (std::f64::consts::PI * x2).cos()
}

fn build_dataset(x1: &[f64], x2: &[f64], y: &[f64], z: &[f64]) -> gam::data::EncodedDataset {
    let n = x1.len();
    let headers = vec![
        "x1".to_string(),
        "x2".to_string(),
        "y".to_string(),
        "z".to_string(),
    ];
    let records: Vec<csv::StringRecord> = (0..n)
        .map(|i| {
            csv::StringRecord::from(vec![
                format!("{:.17e}", x1[i]),
                format!("{:.17e}", x2[i]),
                format!("{:.17e}", y[i]),
                format!("{:.17e}", z[i]),
            ])
        })
        .collect();
    encode_recordswith_inferred_schema(headers, records).expect("encode")
}

fn marginal_prob_rmse(fit: &BernoulliMarginalSlopeFitResult, grid: &[(f64, f64)]) -> f64 {
    let n = grid.len();
    let mut data = Array2::<f64>::zeros((n, 2));
    for (i, &(g1, g2)) in grid.iter().enumerate() {
        data[[i, 0]] = g1;
        data[[i, 1]] = g2;
    }
    let design = build_term_collection_design(data.view(), &fit.marginalspec_resolved)
        .expect("rebuild marginal design");
    let beta0 = &fit.fit.blocks[0].beta;
    let yhat = design.design.apply(beta0);
    let mut sse = 0.0;
    for (i, &(g1, g2)) in grid.iter().enumerate() {
        let p_hat = normal_cdf(fit.baseline_marginal + yhat[i]);
        let p_true = normal_cdf(alpha_true(g1, g2));
        sse += (p_hat - p_true).powi(2);
    }
    (sse / n as f64).sqrt()
}

fn fit_bms(body: &str, ds: &gam::data::EncodedDataset) -> Option<BernoulliMarginalSlopeFitResult> {
    let formula = format!("y ~ {body}");
    let config = FitConfig {
        family: Some("bernoulli-marginal-slope".to_string()),
        link: Some("probit".to_string()),
        logslope_formula: Some(body.to_string()),
        z_column: Some("z".to_string()),
        ..FitConfig::default()
    };
    match fit_from_formula(&formula, ds, &config) {
        Ok(FitResult::BernoulliMarginalSlope(fit)) => Some(fit),
        Ok(_) => None,
        Err(e) => {
            eprintln!("[mjs-ell-sweep] fit '{body}' ERROR: {e}");
            None
        }
    }
}

#[test]
fn mjs_lengthscale_sweep_localizes_1041_deficit() {
    gam::init_parallelism();

    let mut rng = SplitMix64::new(0x1041_2026_0613_0001);
    let mut x1 = vec![0.0; N_TRAIN];
    let mut x2 = vec![0.0; N_TRAIN];
    let mut z = vec![0.0; N_TRAIN];
    for i in 0..N_TRAIN {
        x1[i] = rng.next_unit();
        x2[i] = rng.next_unit();
        z[i] = rng.next_normal();
    }
    let mut rng_y = SplitMix64::new(0x1041_2026_0613_0002);
    let mut y = vec![0.0; N_TRAIN];
    for i in 0..N_TRAIN {
        let eta = alpha_true(x1[i], x2[i]) + beta_true(x1[i]) * z[i];
        let p = normal_cdf(eta).clamp(1e-9, 1.0 - 1e-9);
        y[i] = if rng_y.next_unit() < p { 1.0 } else { 0.0 };
    }
    let ds = build_dataset(&x1, &x2, &y, &z);

    let mut rng_g = SplitMix64::new(0x1041_2026_0613_0003);
    let grid: Vec<(f64, f64)> = (0..N_TEST)
        .map(|_| (rng_g.next_unit(), rng_g.next_unit()))
        .collect();

    // matern baseline.
    if let Some(f) = fit_bms(&format!("matern(x1, x2, k={CENTERS})"), &ds) {
        eprintln!(
            "[mjs-ell-sweep] matern               rmse={:.5}",
            marginal_prob_rmse(&f, &grid)
        );
    }
    // measure-jet AUTO length-scale (the current default = 1x spacing).
    if let Some(f) = fit_bms(&format!("mjs(x1, x2, centers={CENTERS})"), &ds) {
        eprintln!(
            "[mjs-ell-sweep] mjs auto(1x)         rmse={:.5}",
            marginal_prob_rmse(&f, &grid)
        );
    }
    // measure-jet at explicit length-scales (10 centers on [0,1]^2 -> spacing ~0.3).
    for ell in [0.08_f64, 0.12, 0.16, 0.22, 0.30, 0.45, 0.60] {
        let body = format!("mjs(x1, x2, centers={CENTERS}, length_scale={ell})");
        if let Some(f) = fit_bms(&body, &ds) {
            eprintln!(
                "[mjs-ell-sweep] mjs ell={ell:.2}          rmse={:.5}",
                marginal_prob_rmse(&f, &grid)
            );
        }
    }

    assert!(N_TRAIN > 0, "sweep completed");
}
