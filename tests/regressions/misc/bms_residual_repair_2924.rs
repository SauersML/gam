//! gam#2924 — residual genetic repair: a shrunk block of centred genetic
//! residual features inside the anchored marginal-slope likelihood.
//!
//! # The generator
//!
//! ```text
//!   x ~ U(−1, 1),   z | x ~ N(0, 1),   r | x, z ~ N(0, Σ_rr),   Σ_rr known (K = 2)
//!   q(x) = q₀ + q₁ x,   b(x) = b₀ + b₁ x,   β known
//!   Y | x, z, r ~ Bernoulli(Φ(c(x) q(x) + b(x) z + βᵀ r)),   c(x) = √(1 + b(x)² + βᵀ Σ_rr β)
//! ```
//!
//! so `E[Y | x] = Φ(q(x))` exactly: the anchor holds in the generator, and the
//! score-only optimum `f₀ = E[Y | x, z] = Φ((c q + b z)/τ)`, `τ = √(1 + κ)`,
//! `κ = βᵀΣ_rrβ`, lies inside the single-score marginal-slope family with the
//! same marginal index `q` and slope `b/τ`.
//!
//! # The oracles, in closed form
//!
//! With `m = c q + b z` and `r ~ N(0, Σ_rr)`, Stein's lemma gives
//! `E[r Φ(m + βᵀr)] = Σ_rr β · φ(m/τ)/τ`, and integrating `z` out,
//! `E_z[φ(m/τ)] = φ(q)·τ/c`. So
//!
//! ```text
//!   c_r = E[rY] = Σ_rr β · E_x[φ(q(x))/c(x)],
//!   c_rᵀ Σ_rr⁺ c_r = κ · (E_x[φ(q(x))/c(x)])²          (residual_repair_law)
//! ```
//!
//! is the value a LINEAR read of the residual features removes from squared
//! risk relative to `f₀`. The exact nested information gain
//! `E[(p₁ − p₀)²]` (`nested_information_gain`) is the improvement the fitted
//! probit model converges to; it dominates the linear repair value, and
//! `E_r[(Φ(m + βᵀr) − Φ(m/τ))²] = E_u[Φ(m + u)²] − Φ(m/τ)²` with
//! `u ~ N(0, κ)`. Both are one- and three-dimensional integrals over known
//! densities, evaluated here by composite Simpson quadrature to far below the
//! sampling resolution of the comparison.
//!
//! # What is gated
//!
//! * `β̂` recovers `β` to sampling tolerance at every `n`;
//! * the held-out Brier improvement of the residual fit over the score-only fit
//!   approaches the nested gain as `n` grows and exceeds the linear oracle;
//! * `E[p̂ | x]` stays within tolerance of `Φ(q(x))` on held-out contexts, with
//!   and without the block;
//! * under `r ⟂ Y`, `β̂` shrinks to ~0, the ridge spends a nonzero trace (block
//!   EDF below `K`), and against `p_true` the fit loses no more than an
//!   unpenalised block costs in expectation.

use csv::StringRecord;
use gam::families::survival::predict::fit_result_from_saved_model_for_prediction;
use gam::inference::model::FittedModel;
use gam::inference::model_payload_builders::fit_formula_to_payload;
use gam::predict::input::build_predict_input_for_model;
use gam::predict::interval_policy::{PredictionRequest, resolve_prediction_request};
use gam::predict::{FittedModelPredictExt, InferenceCovarianceMode};
use gam::probability::{normal_cdf, normal_pdf};
use gam::utils::splitmix64;
use gam::{FitConfig, encode_recordswith_inferred_schema, init_parallelism};
use ndarray::Array1;

const Q0: f64 = -0.6;
const Q1: f64 = 0.5;
const B0: f64 = 0.5;
const B1: f64 = 0.25;
const SIGMA_RR: [[f64; 2]; 2] = [[1.0, 0.3], [0.3, 0.6]];
const BETA: [f64; 2] = [0.45, -0.35];
const N_TEST: usize = 200_000;

fn next_unit(state: &mut u64) -> f64 {
    (splitmix64(state) >> 11) as f64 / (1u64 << 53) as f64
}

fn next_gauss(state: &mut u64) -> f64 {
    let u1 = next_unit(state).max(f64::MIN_POSITIVE);
    let u2 = next_unit(state);
    (-2.0 * u1.ln()).sqrt() * (std::f64::consts::TAU * u2).cos()
}

fn kappa(beta: &[f64; 2]) -> f64 {
    let mut out = 0.0;
    for i in 0..2 {
        for j in 0..2 {
            out += beta[i] * SIGMA_RR[i][j] * beta[j];
        }
    }
    out
}

fn q_of(x: f64) -> f64 {
    Q0 + Q1 * x
}

fn b_of(x: f64) -> f64 {
    B0 + B1 * x
}

fn c_of(x: f64, kappa: f64) -> f64 {
    (1.0 + b_of(x) * b_of(x) + kappa).sqrt()
}

/// Composite Simpson over `[a, b]` with an even number of panels.
fn simpson(a: f64, b: f64, panels: usize, f: impl Fn(f64) -> f64) -> f64 {
    let n = if panels % 2 == 0 { panels } else { panels + 1 };
    let h = (b - a) / n as f64;
    let mut sum = f(a) + f(b);
    for i in 1..n {
        let weight = if i % 2 == 1 { 4.0 } else { 2.0 };
        sum += weight * f(a + i as f64 * h);
    }
    sum * h / 3.0
}

/// `c_rᵀ Σ_rr⁺ c_r = κ · (E_x[φ(q(x))/c(x)])²`.
fn linear_repair_oracle(beta: &[f64; 2]) -> f64 {
    let kappa = kappa(beta);
    let mean = 0.5 * simpson(-1.0, 1.0, 2000, |x| normal_pdf(q_of(x)) / c_of(x, kappa));
    kappa * mean * mean
}

/// `E[(p₁ − p₀)²]` with `p₁ = Φ(m + βᵀr)`, `p₀ = E[p₁ | x, z] = Φ(m/τ)`.
fn nested_gain_oracle(beta: &[f64; 2]) -> f64 {
    let kappa = kappa(beta);
    let tau = (1.0 + kappa).sqrt();
    let sd_u = kappa.sqrt();
    let inner = |m: f64| -> f64 {
        // E_u[Φ(m + u)²] − Φ(m/τ)², u ~ N(0, κ).
        let second = simpson(-8.0, 8.0, 800, |t| {
            let p = normal_cdf(m + sd_u * t);
            p * p * normal_pdf(t)
        });
        let p0 = normal_cdf(m / tau);
        second - p0 * p0
    };
    0.5 * simpson(-1.0, 1.0, 200, |x| {
        let c = c_of(x, kappa);
        let q = q_of(x);
        let b = b_of(x);
        simpson(-8.0, 8.0, 400, |z| inner(c * q + b * z) * normal_pdf(z))
    })
}

struct Sample {
    dataset: gam::inference::data::EncodedDataset,
    x: Vec<f64>,
    y: Vec<f64>,
    /// The generator's `E[Y | x, z, r]` at every row.
    p_true: Vec<f64>,
    /// The generator's probit index at every row, `p_true = Φ(eta_true)`.
    eta_true: Vec<f64>,
}

/// Draw `n` rows from the generator with coefficient block `beta` (all-zero for
/// the `r ⟂ Y` arm), seeded so each arm is reproducible.
fn draw(n: usize, beta: &[f64; 2], seed: u64) -> Sample {
    let headers = ["y", "x", "z", "r1", "r2"]
        .iter()
        .map(|s| s.to_string())
        .collect::<Vec<_>>();
    let kappa = kappa(beta);
    let l11 = SIGMA_RR[0][0].sqrt();
    let l21 = SIGMA_RR[1][0] / l11;
    let l22 = (SIGMA_RR[1][1] - l21 * l21).sqrt();
    let mut state = seed;
    let mut rows = Vec::with_capacity(n);
    let mut x = Vec::with_capacity(n);
    let mut y = Vec::with_capacity(n);
    let mut p_true = Vec::with_capacity(n);
    let mut eta_true = Vec::with_capacity(n);
    for _ in 0..n {
        let xi = 2.0 * next_unit(&mut state) - 1.0;
        let z = next_gauss(&mut state);
        let e1 = next_gauss(&mut state);
        let e2 = next_gauss(&mut state);
        let r1 = l11 * e1;
        let r2 = l21 * e1 + l22 * e2;
        let eta = c_of(xi, kappa) * q_of(xi) + b_of(xi) * z + beta[0] * r1 + beta[1] * r2;
        let p = normal_cdf(eta);
        let yi = u8::from(next_unit(&mut state) < p);
        rows.push(StringRecord::from(vec![
            yi.to_string(),
            xi.to_string(),
            z.to_string(),
            r1.to_string(),
            r2.to_string(),
        ]));
        x.push(xi);
        y.push(f64::from(yi));
        p_true.push(p);
        eta_true.push(eta);
    }
    Sample {
        dataset: encode_recordswith_inferred_schema(headers, rows).expect("encode #2924 sample"),
        x,
        y,
        p_true,
        eta_true,
    }
}

struct Fitted {
    model: FittedModel,
    /// `β̂` when the block was present.
    beta_residual: Option<Vec<f64>>,
    /// The residual ridge's fitted `log λ`.
    residual_log_lambda: Option<f64>,
    /// The residual block's effective degrees of freedom (sum of its leverages).
    residual_edf: Option<f64>,
}

fn fit(sample: &Sample, with_block: bool) -> Fitted {
    fit_with(sample, with_block, "y ~ s(x)", "s(x)", None)
}

fn fit_with(
    sample: &Sample,
    with_block: bool,
    formula: &str,
    slope_formula: &str,
    latent_measure: Option<&str>,
) -> Fitted {
    let config = FitConfig {
        family: Some("bernoulli-marginal-slope".to_string()),
        z_column: Some("z".to_string()),
        slope_formula: Some(slope_formula.to_string()),
        latent_measure: latent_measure.map(str::to_string),
        residual_columns: if with_block {
            vec!["r1".to_string(), "r2".to_string()]
        } else {
            Vec::new()
        },
        ..FitConfig::default()
    };
    let payload = fit_formula_to_payload(formula.to_string(), &sample.dataset, &config)
        .unwrap_or_else(|e| panic!("bernoulli marginal-slope fit (block={with_block}): {e}"));
    let unified = payload
        .unified
        .as_ref()
        .expect("marginal-slope payload carries the unified fit");
    let beta_residual = with_block.then(|| unified.blocks[2].beta.to_vec());
    let residual_log_lambda = with_block.then(|| {
        let rho = &unified.log_lambdas;
        rho[rho.len() - 1]
    });
    let residual_edf = with_block.then(|| unified.blocks[2].edf);
    if with_block {
        let geometry = payload
            .residual_repair
            .as_ref()
            .expect("the residual fit persists its geometry");
        assert_eq!(geometry.columns, vec!["r1".to_string(), "r2".to_string()]);
        assert_eq!(geometry.pooled_covariance.len(), 3);
        assert_eq!(unified.blocks.len(), 3, "marginal, slope, residual");
    } else {
        assert!(payload.residual_repair.is_none());
        assert_eq!(unified.blocks.len(), 2);
    }
    Fitted {
        model: FittedModel::from_payload(payload),
        beta_residual,
        residual_log_lambda,
        residual_edf,
    }
}

struct Predictions {
    plugin: Array1<f64>,
    posterior_mean: Array1<f64>,
}

fn predict(fitted: &Fitted, sample: &Sample) -> Predictions {
    let model = &fitted.model;
    let col_map = sample.dataset.column_map();
    let n = sample.dataset.values.nrows();
    let zeros = Array1::<f64>::zeros(n);
    let input = build_predict_input_for_model(
        model,
        sample.dataset.values.view(),
        &col_map,
        model.training_headers.as_ref(),
        &zeros,
        &zeros,
        false,
    )
    .expect("predict input for the held-out sample");
    let predictor = model
        .bernoulli_marginal_slope_predictor()
        .unwrap_or_else(|e| panic!("marginal-slope predictor: {e}"));
    let predict_fit =
        fit_result_from_saved_model_for_prediction(model).expect("saved fit for prediction");
    let resolved = resolve_prediction_request(
        &predictor,
        &input,
        &predict_fit,
        true,
        &PredictionRequest {
            interval: None,
            covariance_mode: InferenceCovarianceMode::Conditional,
            observation_interval: false,
            observation_prior_weights: None,
            extrapolation_variance: None,
        },
    )
    .expect("plug-in and posterior-mean prediction");
    Predictions {
        plugin: resolved.mean_plugin,
        posterior_mean: resolved
            .posterior_mean
            .expect("the curved link reports a posterior mean"),
    }
}

fn brier(y: &[f64], p: &Array1<f64>) -> f64 {
    y.iter()
        .zip(p.iter())
        .map(|(&y, &p)| (y - p) * (y - p))
        .sum::<f64>()
        / y.len() as f64
}

fn mean_sq_gap(a: &Array1<f64>, b: &[f64]) -> f64 {
    a.iter()
        .zip(b.iter())
        .map(|(&a, &b)| (a - b) * (a - b))
        .sum::<f64>()
        / b.len() as f64
}

/// What estimating the residual coefficients by unpenalised maximum likelihood
/// costs in expected squared accuracy against `p_true` when they are null:
/// `tr(E_test[φ(η)² r rᵀ] · (Σ_train w(η) r rᵀ)⁻¹)`, with
/// `w = φ(η)² / (Φ(η)(1 − Φ(η)))` the probit working weight. Also returns, for
/// comparison only, the shortcut `K·E_test[φ(η)²] / (n·E_train[w])` that
/// assumes the block's columns independent of `η`.
fn null_ml_cost_floors(train: &Sample, test: &Sample) -> (f64, f64) {
    let moment = |sample: &Sample, weight: &dyn Fn(f64) -> f64| {
        let columns = sample.dataset.column_map();
        let (c1, c2) = (columns["r1"], columns["r2"]);
        let mut m = [[0.0; 2]; 2];
        for (row, &eta) in sample.eta_true.iter().enumerate() {
            let r = [sample.dataset.values[[row, c1]], sample.dataset.values[[row, c2]]];
            let w = weight(eta);
            for a in 0..2 {
                for b in 0..2 {
                    m[a][b] += w * r[a] * r[b];
                }
            }
        }
        m
    };
    let working_weight = |eta: f64| {
        let p = normal_cdf(eta);
        normal_pdf(eta).powi(2) / (p * (1.0 - p))
    };
    let density_sq = |eta: f64| normal_pdf(eta).powi(2);
    let information = moment(train, &working_weight);
    let det = information[0][0] * information[1][1] - information[0][1] * information[1][0];
    let covariance = [
        [information[1][1] / det, -information[0][1] / det],
        [-information[1][0] / det, information[0][0] / det],
    ];
    let spread = moment(test, &density_sq);
    let mut floor = 0.0;
    for a in 0..2 {
        for b in 0..2 {
            floor += spread[a][b] * covariance[b][a];
        }
    }
    let rows_test = test.eta_true.len() as f64;
    let rows_train = train.eta_true.len() as f64;
    let mean_density_sq = test.eta_true.iter().map(|&eta| density_sq(eta)).sum::<f64>() / rows_test;
    let mean_weight = train.eta_true.iter().map(|&eta| working_weight(eta)).sum::<f64>() / rows_train;
    let independent = 2.0 * mean_density_sq / (rows_train * mean_weight);
    (floor / rows_test, independent)
}

/// Max over context bins of `|mean p̂ − mean Φ(q(x))|`.
fn calibration_gap(x: &[f64], p_hat: &Array1<f64>, bins: usize) -> f64 {
    let mut sum_hat = vec![0.0; bins];
    let mut sum_true = vec![0.0; bins];
    let mut count = vec![0usize; bins];
    for (i, &xi) in x.iter().enumerate() {
        let bin = (((xi + 1.0) / 2.0 * bins as f64) as usize).min(bins - 1);
        sum_hat[bin] += p_hat[i];
        sum_true[bin] += normal_cdf(q_of(xi));
        count[bin] += 1;
    }
    (0..bins)
        .map(|b| ((sum_hat[b] - sum_true[b]) / count[b].max(1) as f64).abs())
        .fold(0.0, f64::max)
}

#[test]
fn residual_block_recovers_beta_and_approaches_the_oracle_repair_value() {
    init_parallelism();
    #[cfg(target_os = "macos")]
    gam::gpu::configure_global_policy(gam::gpu::GpuPolicy::Off);

    let linear_oracle = linear_repair_oracle(&BETA);
    let nested_oracle = nested_gain_oracle(&BETA);
    eprintln!(
        "[2924] kappa={:.5} linear repair oracle c_r' S+ c_r={linear_oracle:.5} nested gain \
         E[(p1-p0)^2]={nested_oracle:.5}",
        kappa(&BETA)
    );
    assert!(linear_oracle > 0.0 && nested_oracle >= linear_oracle);

    let test = draw(N_TEST, &BETA, 0x2924_7E57_0000_0001);
    let bins = 8;
    let mut last_gain = 0.0;
    for (n, seed) in [
        (2_000usize, 0x2924_0001_u64),
        (8_000, 0x2924_0002),
        (32_000, 0x2924_0003),
    ] {
        let train = draw(n, &BETA, seed);
        let score_only = fit(&train, false);
        let repaired = fit(&train, true);
        let beta_hat = repaired.beta_residual.clone().expect("β̂");
        let pred0 = predict(&score_only, &test);
        let pred1 = predict(&repaired, &test);
        let brier0 = brier(&test.y, &pred0.plugin);
        let brier1 = brier(&test.y, &pred1.plugin);
        let gain = brier0 - brier1;
        let gain_posterior =
            brier(&test.y, &pred0.posterior_mean) - brier(&test.y, &pred1.posterior_mean);
        let model_error1 = mean_sq_gap(&pred1.plugin, &test.p_true);
        let cal0 = calibration_gap(&test.x, &pred0.plugin, bins);
        let cal1 = calibration_gap(&test.x, &pred1.plugin, bins);
        eprintln!(
            "[2924] n={n}: beta_hat=({:.4}, {:.4}) truth=({:.4}, {:.4}) log_lambda_r={:.3} | \
             held-out Brier score-only={brier0:.5} repaired={brier1:.5} gain={gain:.5} \
             (posterior-mean gain {gain_posterior:.5}) | E[(p_hat1 - p_true)^2]={model_error1:.2e} | \
             calibration gap score-only={cal0:.4} repaired={cal1:.4}",
            beta_hat[0],
            beta_hat[1],
            BETA[0],
            BETA[1],
            repaired.residual_log_lambda.unwrap_or(f64::NAN),
        );
        // (a) β recovery: the sampling band at n = 2000 is ≈ 0.04 per coordinate.
        let tolerance = 0.16 / (n as f64 / 2_000.0).sqrt() + 0.02;
        for k in 0..2 {
            assert!(
                (beta_hat[k] - BETA[k]).abs() < tolerance,
                "n={n}: beta_hat[{k}]={} vs {} (tolerance {tolerance:.3})",
                beta_hat[k],
                BETA[k]
            );
        }
        // (c) the baseline surface stays marginal with and without the block:
        // the bin mean of a fitted surface carries the training sample's own
        // Bernoulli noise, ≈ √(p(1−p)·8/n) per bin, so the band shrinks with n.
        let calibration_tolerance = 0.012 + 0.7 / (n as f64).sqrt();
        assert!(
            cal0 < calibration_tolerance,
            "n={n}: score-only calibration gap {cal0} (tolerance {calibration_tolerance:.4})"
        );
        assert!(
            cal1 < calibration_tolerance,
            "n={n}: repaired calibration gap {cal1} (tolerance {calibration_tolerance:.4})"
        );
        // (b) the improvement approaches the nested gain and exceeds the
        // linear oracle less the paired sampling band of the held-out Brier
        // difference (≈ 3e-4 at N_TEST = 200k) plus the finite-n model error.
        assert!(
            gain > linear_oracle - 3.0e-3,
            "n={n}: held-out gain {gain} is below the linear repair oracle {linear_oracle}"
        );
        assert!(
            (gain - nested_oracle).abs() < 0.25 * nested_oracle + 1.0e-3,
            "n={n}: held-out gain {gain} is not near the nested gain {nested_oracle}"
        );
        last_gain = gain;
    }
    assert!(
        (last_gain - nested_oracle).abs() < 0.12 * nested_oracle + 1.0e-3,
        "n=32000: held-out gain {last_gain} vs nested gain {nested_oracle}"
    );
}

/// The skew of the declared-law arm's score: `z = (e^{0.6u} − μ)/σ`, `u ~ N(0, 1)`,
/// standardised in population.
const SKEW: f64 = 0.6;
const LEVELS: usize = 16;

fn skewed_score(u: f64) -> f64 {
    let mean = (0.5 * SKEW * SKEW).exp();
    let sd = (((SKEW * SKEW).exp() - 1.0) * (SKEW * SKEW).exp()).sqrt();
    ((SKEW * u).exp() - mean) / sd
}

/// The population law of the skewed score: Simpson nodes over `u ∈ [−8, 8]`
/// pushed through [`skewed_score`], weighted by `φ(u)`.
fn skewed_law() -> Vec<(f64, f64)> {
    let panels = 1600usize;
    let h = 16.0 / panels as f64;
    (0..=panels)
        .map(|i| {
            let u = -8.0 + i as f64 * h;
            let simpson = if i == 0 || i == panels {
                1.0
            } else if i % 2 == 1 {
                4.0
            } else {
                2.0
            };
            (skewed_score(u), simpson * h / 3.0 * normal_pdf(u))
        })
        .collect()
}

fn level_x(level: usize) -> f64 {
    -1.0 + 2.0 * (level as f64 + 0.5) / LEVELS as f64
}

/// The true intercept of each context level under the skewed law. Given `z` the
/// drive `b z + βᵀr` is `N(b z, κ)`, so `E[Φ(α + drive)] = Σ_j w_j Φ((α + b z_j)/τ)`,
/// `τ = √(1 + κ)`, and `α` is τ times the root of `Σ_j w_j Φ(ã + (b/τ) z_j) = Φ(q)`,
/// solved here by bisection.
fn skewed_true_intercepts(beta: &[f64; 2], law: &[(f64, f64)]) -> Vec<f64> {
    let tau = (1.0 + kappa(beta)).sqrt();
    (0..LEVELS)
        .map(|level| {
            let x = level_x(level);
            let (target, slope) = (normal_cdf(q_of(x)), b_of(x) / tau);
            let mass = |a: f64| law.iter().map(|&(z, w)| w * normal_cdf(a + slope * z)).sum::<f64>();
            let (mut lo, mut hi) = (-30.0, 30.0);
            for _ in 0..200 {
                let mid = 0.5 * (lo + hi);
                if mass(mid) < target {
                    lo = mid;
                } else {
                    hi = mid;
                }
            }
            tau * 0.5 * (lo + hi)
        })
        .collect()
}

/// `E[(p₁ − p₀)²]` on the skewed arm, `p₀ = Φ((α + b z)/τ)`.
fn skewed_nested_gain(beta: &[f64; 2], law: &[(f64, f64)], alpha: &[f64]) -> f64 {
    let kappa = kappa(beta);
    let tau = (1.0 + kappa).sqrt();
    let sd_u = kappa.sqrt();
    (0..LEVELS)
        .map(|level| {
            let b = b_of(level_x(level));
            law.iter()
                .map(|&(z, w)| {
                    let m = alpha[level] + b * z;
                    let second = simpson(-8.0, 8.0, 400, |t| {
                        let p = normal_cdf(m + sd_u * t);
                        p * p * normal_pdf(t)
                    });
                    let p0 = normal_cdf(m / tau);
                    w * (second - p0 * p0)
                })
                .sum::<f64>()
        })
        .sum::<f64>()
        / LEVELS as f64
}

fn draw_skewed(n: usize, beta: &[f64; 2], alpha: &[f64], seed: u64) -> Sample {
    let headers = ["y", "x", "z", "r1", "r2"]
        .iter()
        .map(|s| s.to_string())
        .collect::<Vec<_>>();
    let l11 = SIGMA_RR[0][0].sqrt();
    let l21 = SIGMA_RR[1][0] / l11;
    let l22 = (SIGMA_RR[1][1] - l21 * l21).sqrt();
    let mut state = seed;
    let mut rows = Vec::with_capacity(n);
    let (mut x, mut y, mut p_true) = (Vec::with_capacity(n), Vec::with_capacity(n), Vec::with_capacity(n));
    let mut eta_true = Vec::with_capacity(n);
    for _ in 0..n {
        let level = ((next_unit(&mut state) * LEVELS as f64) as usize).min(LEVELS - 1);
        let xi = level_x(level);
        let z = skewed_score(next_gauss(&mut state));
        let e1 = next_gauss(&mut state);
        let e2 = next_gauss(&mut state);
        let r1 = l11 * e1;
        let r2 = l21 * e1 + l22 * e2;
        let eta = alpha[level] + b_of(xi) * z + beta[0] * r1 + beta[1] * r2;
        let p = normal_cdf(eta);
        let yi = u8::from(next_unit(&mut state) < p);
        rows.push(StringRecord::from(vec![
            yi.to_string(),
            xi.to_string(),
            z.to_string(),
            r1.to_string(),
            r2.to_string(),
        ]));
        x.push(xi);
        y.push(f64::from(yi));
        p_true.push(p);
        eta_true.push(eta);
    }
    Sample {
        dataset: encode_recordswith_inferred_schema(headers, rows).expect("encode #2924 skewed sample"),
        x,
        y,
        p_true,
        eta_true,
    }
}

#[test]
fn residual_block_anchors_on_a_declared_skewed_score_law() {
    init_parallelism();
    #[cfg(target_os = "macos")]
    gam::gpu::configure_global_policy(gam::gpu::GpuPolicy::Off);

    // The score's law is skewed, so the fit declares its finite law and the
    // block enters through the finite mixture of Gaussian drives.
    let law = skewed_law();
    let alpha = skewed_true_intercepts(&BETA, &law);
    let nested_oracle = skewed_nested_gain(&BETA, &law, &alpha);
    let test = draw_skewed(N_TEST, &BETA, &alpha, 0x2924_7E57_5E3D_0001);
    let n = 32_000usize;
    let train = draw_skewed(n, &BETA, &alpha, 0x2924_5E3D_0003);
    let score_only = fit_with(&train, false, "y ~ x", "x", Some("global-empirical"));
    let repaired = fit_with(&train, true, "y ~ x", "x", Some("global-empirical"));
    let beta_hat = repaired.beta_residual.clone().expect("β̂");
    let pred0 = predict(&score_only, &test);
    let pred1 = predict(&repaired, &test);
    let gain = brier(&test.y, &pred0.plugin) - brier(&test.y, &pred1.plugin);
    let bins = 8;
    let cal0 = calibration_gap(&test.x, &pred0.plugin, bins);
    let cal1 = calibration_gap(&test.x, &pred1.plugin, bins);
    // The same block anchored on the standard-normal law this score does not
    // have: what the declared law is for.
    let gaussian = fit_with(&train, true, "y ~ x", "x", Some("standard-normal"));
    let cal_gaussian = calibration_gap(&test.x, &predict(&gaussian, &test).plugin, bins);
    eprintln!(
        "[2924 skewed] n={n}: beta_hat=({:.4}, {:.4}) truth=({:.4}, {:.4}) log_lambda_r={:.3} | \
         held-out Brier gain={gain:.5} nested gain={nested_oracle:.5} | calibration gap \
         score-only={cal0:.4} repaired={cal1:.4} repaired-on-standard-normal={cal_gaussian:.4}",
        beta_hat[0],
        beta_hat[1],
        BETA[0],
        BETA[1],
        repaired.residual_log_lambda.unwrap_or(f64::NAN),
    );
    let tolerance = 0.16 / (n as f64 / 2_000.0).sqrt() + 0.02;
    for k in 0..2 {
        assert!(
            (beta_hat[k] - BETA[k]).abs() < tolerance,
            "skewed law: beta_hat[{k}]={} vs {} (tolerance {tolerance:.3})",
            beta_hat[k],
            BETA[k]
        );
    }
    let calibration_tolerance = 0.012 + 0.7 / (n as f64).sqrt();
    assert!(
        cal0 < calibration_tolerance,
        "skewed law: score-only calibration gap {cal0} (tolerance {calibration_tolerance:.4})"
    );
    assert!(
        cal1 < calibration_tolerance,
        "skewed law: repaired calibration gap {cal1} (tolerance {calibration_tolerance:.4})"
    );
    assert!(
        (gain - nested_oracle).abs() < 0.25 * nested_oracle + 1.0e-3,
        "skewed law: held-out gain {gain} is not near the nested gain {nested_oracle}"
    );
}

#[test]
fn residual_block_shrinks_to_zero_when_the_features_carry_no_outcome_information() {
    init_parallelism();
    #[cfg(target_os = "macos")]
    gam::gpu::configure_global_policy(gam::gpu::GpuPolicy::Off);

    let null_beta = [0.0, 0.0];
    let train = draw(8_000, &null_beta, 0x2924_0000_00FF);
    let test = draw(50_000, &null_beta, 0x2924_7E57_00FF);
    let score_only = fit(&train, false);
    let repaired = fit(&train, true);
    let beta_hat = repaired.beta_residual.clone().expect("β̂");
    let pred0 = predict(&score_only, &test);
    let pred1 = predict(&repaired, &test);
    let max_gap = pred0
        .plugin
        .iter()
        .zip(pred1.plugin.iter())
        .map(|(a, b)| (a - b).abs())
        .fold(0.0, f64::max);
    let mean_gap = mean_sq_gap(&pred1.plugin, pred0.plugin.as_slice().expect("contiguous")).sqrt();
    let gain = brier(&test.y, &pred0.plugin) - brier(&test.y, &pred1.plugin);
    // Against p_true the block may cost at most what estimating its null
    // coefficients by unpenalised maximum likelihood costs in expectation: the
    // trace tr(E_test[φ(η)² r rᵀ] · (Σ_train w(η) r rᵀ)⁻¹), with
    // w = φ(η)² / (Φ(η)(1 − Φ(η))) the probit working weight. The REML ridge
    // shrinks below it. This is an accuracy bound, not a ridge guard: the bound
    // is the expected cost of an unpenalised block, so such a block exceeds it
    // on only some draws. The gap max|p1 − p0| is reported, not asserted: it
    // moves with smoothing parameters the criterion leaves flat, in either
    // direction of accuracy.
    let (floor, independent_floor) = null_ml_cost_floors(&train, &test);
    let excess = mean_sq_gap(&pred1.plugin, &test.p_true) - mean_sq_gap(&pred0.plugin, &test.p_true);
    eprintln!(
        "[2924 null] beta_hat=({:.4}, {:.4}) log_lambda_r={:.3} edf_r={:.6} (K - edf_r = {:.3e}) | \
         max|p1-p0|={max_gap:.2e} rms|p1-p0|={mean_gap:.2e} | held-out Brier gain={gain:.2e} | \
         E[(p1-p_true)^2]-E[(p0-p_true)^2]={excess:.3e} floor={floor:.3e} \
         independent_floor={independent_floor:.3e}",
        beta_hat[0],
        beta_hat[1],
        repaired.residual_log_lambda.unwrap_or(f64::NAN),
        repaired.residual_edf.unwrap_or(f64::NAN),
        beta_hat.len() as f64 - repaired.residual_edf.unwrap_or(f64::NAN)
    );
    for k in 0..2 {
        assert!(
            beta_hat[k].abs() < 0.05,
            "beta_hat[{k}]={} did not shrink to ~0 under r ⟂ Y",
            beta_hat[k]
        );
    }
    // The ridge guard. The block's EDF is its width K minus the trace its ridge
    // spends, λ·tr(H⁻¹S). With S = 0 that trace is exactly zero and the EDF reads
    // exactly K; any active ridge spends a positive trace. K − K·ε separates that
    // exact zero from an active ridge; it is not a rounding bound on the trace.
    // It guards a missing ridge, not a weakened one: a tiny λ reads under K.
    let width = beta_hat.len() as f64;
    let edf = repaired.residual_edf.expect("the residual fit reports its block EDF");
    let exact_zero_margin = width * f64::EPSILON;
    assert!(
        edf < width - exact_zero_margin,
        "the residual block's EDF {edf} is within {exact_zero_margin:.3e} of its width \
         {width}: the ridge spends nothing"
    );
    assert!(
        excess <= floor,
        "the residual fit lost {excess:.3e} of squared accuracy against p_true, above the \
         {floor:.3e} that estimating {} null coefficients costs",
        beta_hat.len()
    );
    assert!(
        gain.abs() < 1.0e-3,
        "held-out Brier gain {gain} under r ⟂ Y should be at the noise floor"
    );
}
