//! #1041 accuracy regression gate: on a `bernoulli-marginal-slope` (probit)
//! fit, the measure-jet basis must be **accuracy-competitive** with the
//! Matern and Duchon bases — it was consistently the LEAST accurate (worst in
//! 6-7/8 datasets on held-out truth-RMSE) because SIMPLE mode froze the
//! measure-jet `(s, alpha, ln tau)` dials while Matern/Duchon got their
//! length-scale REML-learned. The fix (770f825eb) enrolls those dials in
//! SIMPLE mode; this test pins the consequence: head-to-head, measure-jet's
//! held-out RMSE against the KNOWN generative probability surface must
//! match-or-beat the better of Matern/Duchon (within a small CI flake guard),
//! never sit systematically worst.
//!
//! Truth is self-constructed (not a reference tool): a single principled
//! probit Bernoulli draw per row from `eta = alpha(x1,x2) + beta(x1)*z`, and
//! the scored metric is RMSE of the fitted marginal probability `Phi(eta_hat)`
//! at `z = 0` (the marginal surface) against the planted `Phi(alpha_true)` on
//! a held-out latent grid. All three bases see the SAME data and the SAME
//! held-out grid; the assertion is relative (measure-jet vs the baseline
//! min), so it is a basis-conditioning gate, not an absolute-accuracy bar.

use gam::matrix::LinearOperator;
use gam::smooth::build_term_collection_design;
use gam::{
    BernoulliMarginalSlopeFitResult, FitConfig, FitResult, encode_recordswith_inferred_schema,
    fit_from_formula,
};
use ndarray::Array2;

const N_TRAIN: usize = 1_500;
const N_TEST: usize = 600;
const CENTERS: usize = 10;

/// SplitMix64 — same data law as `measure_jet_bms_backend.rs` so the two
/// tests share one generative construction.
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

/// Planted logslope truth: monotone in x1, flat in x2.
fn beta_true(x1: f64) -> f64 {
    0.2 + 0.9 * x1
}

/// Planted marginal surface: smooth in both ambient coordinates.
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
    encode_recordswith_inferred_schema(headers, records).expect("encode BMS dataset")
}

/// Held-out RMSE of the fitted MARGINAL probability surface `Phi(alpha_hat)`
/// at `z = 0` against the planted `Phi(alpha_true)` on a fresh latent grid.
/// The marginal surface is reconstructed exactly as the family stores it:
/// block 0 = marginal, `alpha_hat(x) = baseline_marginal + design(x).beta0`.
fn marginal_prob_rmse(
    fit: &BernoulliMarginalSlopeFitResult,
    grid: &[(f64, f64)],
    what: &str,
) -> f64 {
    let n = grid.len();
    let mut data = Array2::<f64>::zeros((n, 2));
    for (i, &(g1, g2)) in grid.iter().enumerate() {
        data[[i, 0]] = g1;
        data[[i, 1]] = g2;
    }
    let design = build_term_collection_design(data.view(), &fit.marginalspec_resolved)
        .unwrap_or_else(|e| panic!("{what}: rebuild marginal design: {e}"));
    let beta0 = &fit.fit.blocks[0].beta;
    let yhat = design.design.apply(beta0);
    let mut sse = 0.0;
    for (i, &(g1, g2)) in grid.iter().enumerate() {
        let eta_hat = fit.baseline_marginal + yhat[i];
        let p_hat = normal_cdf(eta_hat);
        let p_true = normal_cdf(alpha_true(g1, g2));
        let d = p_hat - p_true;
        sse += d * d;
    }
    (sse / n as f64).sqrt()
}

fn fit_bms(body: &str, ds: &gam::data::EncodedDataset) -> BernoulliMarginalSlopeFitResult {
    let formula = format!("y ~ {body}");
    let config = FitConfig {
        family: Some("bernoulli-marginal-slope".to_string()),
        link: Some("probit".to_string()),
        logslope_formula: Some(body.to_string()),
        z_column: Some("z".to_string()),
        ..FitConfig::default()
    };
    let result = fit_from_formula(&formula, ds, &config)
        .unwrap_or_else(|e| panic!("gam bms fit '{formula}': {e}"));
    match result {
        FitResult::BernoulliMarginalSlope(fit) => *fit,
        other => panic!("expected BernoulliMarginalSlope fit for '{body}', got {other:?}"),
    }
}

#[test]
fn measure_jet_bms_accuracy_is_competitive_with_matern_and_duchon() {
    gam::init_parallelism();

    // One shared dataset: principled probit Bernoulli draws from
    // eta = alpha(x1,x2) + beta(x1)*z.
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

    // Held-out latent grid shared by all three bases.
    let mut rng_g = SplitMix64::new(0x1041_2026_0613_0003);
    let grid: Vec<(f64, f64)> = (0..N_TEST)
        .map(|_| (rng_g.next_unit(), rng_g.next_unit()))
        .collect();

    let mjs_body = format!("mjs(x1, x2, centers={CENTERS})");
    let matern_body = format!("matern(x1, x2, k={CENTERS})");
    let duchon_body = format!("duchon(x1, x2, k={CENTERS})");

    let mjs_fit = fit_bms(&mjs_body, &ds);
    let matern_fit = fit_bms(&matern_body, &ds);
    let duchon_fit = fit_bms(&duchon_body, &ds);

    let mjs_rmse = marginal_prob_rmse(&mjs_fit, &grid, "mjs");
    let matern_rmse = marginal_prob_rmse(&matern_fit, &grid, "matern");
    let duchon_rmse = marginal_prob_rmse(&duchon_fit, &grid, "duchon");
    println!(
        "[#1041 bms-accuracy] mjs={mjs_rmse:.5} matern={matern_rmse:.5} duchon={duchon_rmse:.5}"
    );

    // Match-or-beat the better of the two simpler bases, within a small CI
    // flake guard. Before the dial-enrollment fix measure-jet sat ~1.68x the
    // matern truth-RMSE on the parity fixture (and was worst in 6-7/8 #1041
    // datasets); this bound fails if the dials revert to frozen.
    let baseline = matern_rmse.min(duchon_rmse);
    assert!(
        mjs_rmse <= 1.10 * baseline,
        "#1041: measure-jet BMS marginal accuracy not competitive: mjs={mjs_rmse:.5} \
         matern={matern_rmse:.5} duchon={duchon_rmse:.5} baseline={baseline:.5} \
         (ratio {:.3} > 1.10)",
        mjs_rmse / baseline
    );
}
