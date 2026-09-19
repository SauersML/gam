#![cfg(test)]
//! Posterior-predictive prediction through the anchor (gam#2927).
//!
//! The Bernoulli marginal-slope linear predictor is anchored:
//! `η = c(b)·q + s·b·z` under the standard-normal latent law, `η = a(q, b) + s·b·z`
//! with `a` the root of the calibration equation under a declared empirical
//! law. A coefficient draw `θ ~ N(θ̂, V)` moves `q`, `b` AND the anchor, so the
//! posterior-predictive probability `E_θ[Φ(η(θ))]` is not `Φ(η̂)` (plug-in), not
//! `Φ(η̂/√(1 + v))` with the anchor frozen, and not `Φ(η̂/√(1 + gᵀVg))` with
//! `η(θ)` linearised at `θ̂` — that last one is what the posterior-mean pass
//! reported before this test existed.
//!
//! The reference here is Monte Carlo over the FULL coefficient posterior:
//! 10⁵ draws of `θ`, each mapped through the production `final_eta_from_theta`
//! (which re-solves the anchor per draw). Every named integration is measured
//! against it on a fitted synthetic model at several contexts and latent
//! scores; only the exact one is required to agree within the Monte Carlo
//! resolution, the others are reported.

use crate::bernoulli_marginal_slope::{
    AnchoredPosteriorIntegration, bernoulli_marginal_slope_posterior_mean,
};
use crate::test_support::init_parallelism;
use crate::{FittedModelPredictExt, PosteriorMeanOptions, PredictInput, PredictableModel};
use gam_data::encode_recordswith_inferred_schema;
use gam_linalg::faer_ndarray::FaerEigh;
use gam_math::probability::normal_cdf;
use gam_models::bms::{EmpiricalZGrid, LatentMeasureKind};
use gam_models::fit_orchestration::FitConfig;
use gam_models::inference::model::FittedModel;
use gam_models::inference::model_payload_builders::fit_formula_to_payload;
use gam_models::inference::predict_input::build_predict_input_for_model;
use gam_models::inference::predict_io::BernoulliMarginalSlopePredictor;
use gam_solve::model_types::UnifiedFitResult;
use ndarray::{Array1, Array2};
use std::collections::HashMap;

const MC_DRAWS: usize = 100_000;

fn splitmix(state: &mut u64) -> u64 {
    gam_linalg::utils::splitmix64(state)
}

pub(crate) fn uniform(state: &mut u64) -> f64 {
    (splitmix(state) >> 11) as f64 / (1u64 << 53) as f64
}

pub(crate) fn gaussian(state: &mut u64) -> f64 {
    let u1 = uniform(state).max(1e-300);
    let u2 = uniform(state);
    (-2.0 * u1.ln()).sqrt() * (std::f64::consts::TAU * u2).cos()
}

/// Planted marginal probit index and slope surfaces.
fn planted_q(x: f64) -> f64 {
    -0.7 + 0.9 * (2.0 * x).sin()
}

fn planted_b(x: f64) -> f64 {
    0.45 + 0.35 * x
}

/// A synthetic Bernoulli marginal-slope dataset simulated from the family's
/// own model: `P(y = 1 | x, z) = Φ(c(b(x))·q(x) + b(x)·z)`, `z ~ N(0, 1)`, so the
/// marginal law is `Φ(q(x))` and the conditional slope is `b(x)`.
fn simulate_dataset(n: usize, seed: u64) -> gam_data::EncodedDataset {
    let mut state = seed;
    let headers = vec!["y".to_string(), "x".to_string(), "z".to_string()];
    let mut records = Vec::with_capacity(n);
    for _ in 0..n {
        let x = -1.0 + 2.0 * uniform(&mut state);
        let z = gaussian(&mut state);
        let b = planted_b(x);
        let eta = (1.0 + b * b).sqrt() * planted_q(x) + b * z;
        let y = if uniform(&mut state) < normal_cdf(eta) {
            1.0
        } else {
            0.0
        };
        records.push(csv::StringRecord::from(vec![
            format!("{y}"),
            format!("{x:.17e}"),
            format!("{z:.17e}"),
        ]));
    }
    encode_recordswith_inferred_schema(headers, records).expect("encode marginal-slope dataset")
}

fn fit_marginal_slope_model(n: usize, seed: u64) -> FittedModel {
    let ds = simulate_dataset(n, seed);
    let cfg = FitConfig {
        family: Some("bernoulli-marginal-slope".to_string()),
        slope_formula: Some("s(x, k=5)".to_string()),
        z_column: Some("z".to_string()),
        // The simulated score is standard normal by construction; freezing it
        // pins the rigid latent law so the anchor is the closed-form
        // `c(b)·q`, the branch whose curvature this test is about.
        frozen_score: true,
        precompute_conformal: Some(false),
        ..FitConfig::default()
    };
    let payload = fit_formula_to_payload("y ~ s(x, k=6)".to_string(), &ds, &cfg)
        .expect("bernoulli marginal-slope fit");
    FittedModel::from_payload(payload)
}

/// Prediction rows: a grid of contexts `x` crossed with latent scores `z`.
fn prediction_frame() -> (Array2<f64>, HashMap<String, usize>, Vec<(f64, f64)>) {
    let contexts = [-0.8, -0.3, 0.2, 0.7];
    let scores = [-2.0, -1.0, 0.0, 1.0, 2.0];
    let mut cells = Vec::new();
    for &x in &contexts {
        for &z in &scores {
            cells.push((x, z));
        }
    }
    let mut data = Array2::<f64>::zeros((cells.len(), 3));
    for (row, &(x, z)) in cells.iter().enumerate() {
        data[[row, 1]] = x;
        data[[row, 2]] = z;
    }
    let col_map = HashMap::from([
        ("y".to_string(), 0usize),
        ("x".to_string(), 1usize),
        ("z".to_string(), 2usize),
    ]);
    (data, col_map, cells)
}

fn predict_input_for(
    model: &FittedModel,
    data: &Array2<f64>,
    col_map: &HashMap<String, usize>,
) -> PredictInput {
    let zero = Array1::<f64>::zeros(data.nrows());
    build_predict_input_for_model(
        model,
        data.view(),
        col_map,
        model.training_headers.as_ref(),
        &zero,
        &zero,
        false,
    )
    .expect("marginal-slope predict input")
}

/// A symmetric factor `L` with `L Lᵀ = V` for a positive semidefinite `V`, via
/// its eigendecomposition (negative round-off eigenvalues are clipped to 0).
pub(crate) fn psd_factor(cov: &Array2<f64>) -> Array2<f64> {
    let (eigenvalues, eigenvectors) = cov
        .eigh(faer::Side::Lower)
        .expect("coefficient covariance eigendecomposition");
    let p = cov.nrows();
    let mut factor = Array2::<f64>::zeros((p, p));
    for j in 0..p {
        let scale = eigenvalues[j].max(0.0).sqrt();
        for i in 0..p {
            factor[[i, j]] = eigenvectors[[i, j]] * scale;
        }
    }
    factor
}

struct MonteCarloReference {
    mean: Array1<f64>,
    standard_error: Array1<f64>,
}

/// How a Monte Carlo draw is mapped to its linear predictor.
#[derive(Clone, Copy)]
enum DrawKernel {
    /// The production `final_eta_from_theta`: independent of every accessor the
    /// exact integration uses.
    Production,
    /// `anchored_primaries` + `AnchoredRowKernel::eta`, the accessors the exact
    /// integration uses, with the tail-tolerant root acceptance. Needed for a
    /// declared empirical law under a wide posterior: draws several standard
    /// deviations into the tail of `q` land where the fit-time root solve
    /// refuses its own roundoff floor, and the production path inherits that
    /// refusal. The two agree at `θ̂` to floating-point resolution
    /// (`kernels_agree_with_the_production_kernel`).
    AnchoredKernels,
}

/// `E_θ[Φ(η(θ))]` by Monte Carlo over `θ ~ N(θ̂, V)`, every draw mapped through
/// the anchored kernel (anchor re-solved per draw), with the standard error of
/// each row's estimate.
fn monte_carlo_reference(
    predictor: &BernoulliMarginalSlopePredictor,
    input: &PredictInput,
    covariance: &Array2<f64>,
    draws: usize,
    seed: u64,
    draw_kernel: DrawKernel,
) -> MonteCarloReference {
    use rayon::iter::{IntoParallelIterator, ParallelIterator};
    let theta_hat = predictor.theta();
    let factor = psd_factor(covariance);
    let p = theta_hat.len();
    let n = input.design.nrows();
    let kernels = predictor
        .anchored_row_kernels(input)
        .expect("anchored row kernels");
    let chunks = 64usize;
    let per_chunk = draws.div_ceil(chunks);
    let (sum, sum_sq): (Array1<f64>, Array1<f64>) = (0..chunks)
        .into_par_iter()
        .map(|chunk| {
            let mut state = seed ^ (chunk as u64 + 1).wrapping_mul(0x9E37_79B9_7F4A_7C15);
            let mut sum = Array1::<f64>::zeros(n);
            let mut sum_sq = Array1::<f64>::zeros(n);
            let mut standard = Array1::<f64>::zeros(p);
            let first = chunk * per_chunk;
            let last = ((chunk + 1) * per_chunk).min(draws);
            for _ in first..last {
                standard.mapv_inplace(|_| gaussian(&mut state));
                let theta = &theta_hat + &factor.dot(&standard);
                let eta = match draw_kernel {
                    DrawKernel::Production => predictor
                        .final_eta_from_theta(input, &theta)
                        .expect("posterior draw maps through the production kernel"),
                    DrawKernel::AnchoredKernels => {
                        let (q, b) = predictor
                            .anchored_primaries(input, &theta)
                            .expect("posterior draw projects onto the primaries");
                        Array1::from_iter((0..n).map(|row| {
                            kernels[row]
                                .eta(q[row], b[row])
                                .expect("posterior draw maps through the anchored row kernel")
                        }))
                    }
                };
                for row in 0..n {
                    let probability = normal_cdf(eta[row]);
                    sum[row] += probability;
                    sum_sq[row] += probability * probability;
                }
            }
            (sum, sum_sq)
        })
        .reduce(
            || (Array1::zeros(n), Array1::zeros(n)),
            |(a, a2), (b, b2)| (a + b, a2 + b2),
        );
    let count = draws as f64;
    let mean = sum.mapv(|s| s / count);
    let standard_error = Array1::from_iter((0..n).map(|row| {
        let variance = (sum_sq[row] / count - mean[row] * mean[row]).max(0.0);
        (variance / count).sqrt()
    }));
    MonteCarloReference {
        mean,
        standard_error,
    }
}

/// At `θ̂` the anchored row kernels must reproduce the production linear
/// predictor exactly: same primaries, same anchor, same latent score.
fn kernels_agree_with_the_production_kernel(
    label: &str,
    predictor: &BernoulliMarginalSlopePredictor,
    input: &PredictInput,
) {
    let theta = predictor.theta();
    let production = predictor
        .final_eta_from_theta(input, &theta)
        .expect("production kernel at theta_hat");
    let (q, b) = predictor
        .anchored_primaries(input, &theta)
        .expect("primaries at theta_hat");
    let kernels = predictor
        .anchored_row_kernels(input)
        .expect("anchored row kernels");
    for row in 0..production.len() {
        let anchored = kernels[row].eta(q[row], b[row]).expect("anchored kernel");
        assert!(
            (anchored - production[row]).abs() <= 1e-10 * (1.0 + production[row].abs()),
            "[{label}] row {row}: anchored kernel η {anchored} != production η {}",
            production[row]
        );
    }
}

const METHODS: [(AnchoredPosteriorIntegration, &str); 4] = [
    (AnchoredPosteriorIntegration::PlugIn, "plug-in"),
    (
        AnchoredPosteriorIntegration::FrozenAnchorGaussian,
        "frozen-anchor",
    ),
    (
        AnchoredPosteriorIntegration::LinearisedAnchorGaussian,
        "linearised",
    ),
    (AnchoredPosteriorIntegration::ExactAnchor, "exact"),
];

/// Runs every named integration on `(predictor, fit)` against the Monte Carlo
/// reference, prints the per-row table and the max |Δp| per method, and
/// requires the exact integration to sit inside the Monte Carlo resolution at
/// every row. Returns the max |Δp| per method in `METHODS` order.
fn measure(
    label: &str,
    predictor: &BernoulliMarginalSlopePredictor,
    input: &PredictInput,
    fit: &UnifiedFitResult,
    cells: &[(f64, f64)],
    seed: u64,
    draw_kernel: DrawKernel,
) -> [f64; 4] {
    let covariance = fit
        .beta_covariance()
        .expect("marginal-slope fit carries a conditional covariance");
    kernels_agree_with_the_production_kernel(label, predictor, input);
    let reference =
        monte_carlo_reference(predictor, input, covariance, MC_DRAWS, seed, draw_kernel);
    let estimates: Vec<Array1<f64>> = METHODS
        .iter()
        .map(|(integration, name)| {
            bernoulli_marginal_slope_posterior_mean(predictor, input, fit, *integration)
                .unwrap_or_else(|e| panic!("[{label}] {name} integration: {e}"))
        })
        .collect();
    eprintln!(
        "[{label}] {:>6} {:>5} | {:>9} {:>8} | {:>10} {:>10} {:>10} {:>10}",
        "x", "z", "mc", "mc_se", "plug-in", "frozen", "linear", "exact"
    );
    let mut max_abs = [0.0_f64; 4];
    let mut max_se = 0.0_f64;
    for (row, &(x, z)) in cells.iter().enumerate() {
        let deltas: Vec<f64> = estimates
            .iter()
            .map(|estimate| estimate[row] - reference.mean[row])
            .collect();
        eprintln!(
            "[{label}] {x:>6.2} {z:>5.1} | {:>9.5} {:>8.1e} | {:>+10.2e} {:>+10.2e} {:>+10.2e} {:>+10.2e}",
            reference.mean[row],
            reference.standard_error[row],
            deltas[0],
            deltas[1],
            deltas[2],
            deltas[3],
        );
        for (slot, delta) in deltas.iter().enumerate() {
            max_abs[slot] = max_abs[slot].max(delta.abs());
        }
        max_se = max_se.max(reference.standard_error[row]);
        // The exact integration and the Monte Carlo reference estimate the same
        // integral; 4 standard errors is the widest gap sampling alone explains.
        let tolerance = 4.0 * reference.standard_error[row] + 1e-6;
        assert!(
            deltas[3].abs() <= tolerance,
            "[{label}] exact anchored integration at x={x}, z={z} is {:+.3e} from the Monte Carlo \
             reference (tolerance {tolerance:.2e})",
            deltas[3]
        );
    }
    eprintln!(
        "[{label}] max|Δp| vs MC ({MC_DRAWS} draws, max MC se {max_se:.1e}): plug-in {:.2e}, \
         frozen-anchor {:.2e}, linearised {:.2e}, exact {:.2e}",
        max_abs[0], max_abs[1], max_abs[2], max_abs[3]
    );
    max_abs
}

/// A declared empirical latent law that is NOT standard normal: a 41-node
/// discretisation of a scaled Student-t-like density (heavier tails, unit
/// variance to first order), so the anchor `a(q, b)` is a genuine root solve.
pub(crate) fn heavy_tailed_grid() -> EmpiricalZGrid {
    let nodes: Vec<f64> = (0..41).map(|i| -4.0 + 0.2 * i as f64).collect();
    let raw: Vec<f64> = nodes
        .iter()
        .map(|&z| (1.0 + z * z / 4.0).powf(-3.0))
        .collect();
    let total: f64 = raw.iter().sum();
    let weights = raw.iter().map(|w| w / total).collect();
    EmpiricalZGrid::new(nodes, weights, "heavy-tailed test grid").expect("valid grid")
}

#[test]
fn anchored_posterior_predictive_matches_monte_carlo_over_the_coefficient_posterior() {
    init_parallelism();
    let model = fit_marginal_slope_model(400, 20260916);
    let predictor = model
        .bernoulli_marginal_slope_predictor()
        .expect("fitted marginal-slope predictor");
    assert!(
        matches!(predictor.latent_measure, LatentMeasureKind::StandardNormal),
        "a frozen score fits the rigid standard-normal law"
    );
    assert!(!predictor.has_flexible_runtime());
    let fit = model
        .fit_result
        .clone()
        .expect("fitted model carries its fit result");
    let (data, col_map, cells) = prediction_frame();
    let input = predict_input_for(&model, &data, &col_map);

    // 1. Fitted covariance, rigid law: the production configuration.
    let fitted = measure(
        "rigid/fitted-V",
        &predictor,
        &input,
        &fit,
        &cells,
        1,
        DrawKernel::Production,
    );

    // 2. The same fit with its covariance inflated 9× (SEs ×3): the curvature
    //    the first-order shortcut drops grows with the posterior spread, so this
    //    is where a linearised anchor separates from the exact one.
    let mut wide_fit = fit.clone();
    let wide_cov = fit.beta_covariance().expect("covariance").mapv(|v| 9.0 * v);
    wide_fit.covariance_conditional = Some(wide_cov.clone());
    let mut wide_predictor = model
        .bernoulli_marginal_slope_predictor()
        .expect("fitted marginal-slope predictor");
    wide_predictor.covariance = Some(wide_cov);
    let wide = measure(
        "rigid/9×V",
        &wide_predictor,
        &input,
        &wide_fit,
        &cells,
        2,
        DrawKernel::Production,
    );

    // 3. Declared empirical law (root-solved anchor) on the fitted coefficients
    //    and the inflated covariance.
    let mut empirical_predictor = model
        .bernoulli_marginal_slope_predictor()
        .expect("fitted marginal-slope predictor");
    empirical_predictor.latent_measure = LatentMeasureKind::GlobalEmpirical {
        grid: heavy_tailed_grid(),
    };
    empirical_predictor.covariance = wide_fit.covariance_conditional.clone();
    let empirical = measure(
        "empirical/9×V",
        &empirical_predictor,
        &input,
        &wide_fit,
        &cells,
        3,
        DrawKernel::AnchoredKernels,
    );

    // The pass the surfaces publish is the exact integration.
    let published = predictor
        .predict_posterior_mean(&input, &fit, &PosteriorMeanOptions::point_only())
        .expect("posterior-mean pass");
    let exact = bernoulli_marginal_slope_posterior_mean(
        &predictor,
        &input,
        &fit,
        AnchoredPosteriorIntegration::ExactAnchor,
    )
    .expect("exact integration");
    for row in 0..cells.len() {
        assert_eq!(
            published.mean[row].to_bits(),
            exact[row].to_bits(),
            "the posterior-mean pass publishes the exact anchored integration at row {row}"
        );
    }
    assert_eq!(
        AnchoredPosteriorIntegration::default_for(&predictor),
        AnchoredPosteriorIntegration::ExactAnchor
    );

    // Under the inflated covariance the linearised anchor must be visibly
    // wrong where the exact one is not: otherwise this test would not
    // distinguish the two and could not catch a regression to the shortcut.
    for (label, max_abs) in [("rigid/9×V", wide), ("empirical/9×V", empirical)] {
        assert!(
            max_abs[2] > 4.0 * max_abs[3] && max_abs[2] > 2e-3,
            "[{label}] the linearised-anchor shortcut (max|Δp| {:.2e}) is not separated from \
             the exact integration (max|Δp| {:.2e}) at this posterior spread",
            max_abs[2],
            max_abs[3]
        );
    }
    eprintln!(
        "[rigid/fitted-V] linearised max|Δp| {:.2e} vs exact {:.2e}",
        fitted[2], fitted[3]
    );
}
