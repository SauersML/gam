#![cfg(test)]
//! Survival marginal-slope posterior-mean surface against Monte Carlo (gam#2927,
//! gam#2931).
//!
//! The survival marginal-slope family is anchored the same way as the Bernoulli
//! one — `S(t | x, z) = Φ(−η(t))` with `η(t) = c(b)·q(t) + s·b·z` under the
//! standard-normal law and `η(t) = a(q, b) + s·b·z` under a declared empirical
//! law — and `gam_models::survival::predict::predict_survival` under
//! `SurvivalPredictEstimand::PosteriorMean` publishes the posterior-predictive
//! law: `E_θ[S(t; θ)]`, and the hazard of that law, `E_θ[f(t; θ)]/E_θ[S(t; θ)]`.
//! Both integrations it can run are measured here against Monte Carlo over the
//! full coefficient posterior, every draw run through the production plug-in
//! prediction (anchor re-solved per draw): the `2·rank` sigma-point rule, exact
//! for cubic functionals of `θ` only, and the exact integration over the
//! bivariate Gaussian law of `(q(t), b(t))` (both affine in `θ`) with the anchor
//! re-solved at every node, which is what the pass publishes. The event density
//! `f(t) = S(t)·h(t)` and the predictive hazard are measured the same way,
//! because the exact rule reaches them through the conditional law of the
//! tangents `(q′(t), b′(t))` rather than by replaying draws. The Monte Carlo
//! hazard is `mean(f)/mean(S)` over the draws; the table also prints how far
//! `mean(f/S)` sits from it.
//!
//! Measured on this fixture before the exact rule existed (gam#2927, 5 latent
//! scores × 3 horizons, 20,000 draws): the sigma-point surface was within
//! 4.1e-4 of the Monte Carlo integral at the fitted covariance where the
//! plug-in was off by 1.4e-3, and within 2.0e-3 at 9× the covariance (worst
//! cell 45 MC standard errors) where the plug-in was off by 1.3e-2.

use crate::anchored_posterior_predictive_tests::{
    gaussian, heavy_tailed_grid, psd_factor, uniform,
};
use crate::test_support::init_parallelism;
use gam_data::encode_recordswith_inferred_schema;
use gam_math::probability::normal_cdf;
use gam_models::bms::LatentMeasureKind;
use gam_models::fit_orchestration::FitConfig;
use gam_models::inference::model::FittedModel;
use gam_models::inference::model_payload_builders::fit_formula_to_payload;
use gam_models::survival::predict::{
    SurvivalPosteriorIntegration, SurvivalPredictEstimand, SurvivalPredictRequest,
    SurvivalPredictionCovarianceMode, predict_survival, predict_survival_posterior_mean_with,
};
use ndarray::{Array1, Array2};
use std::collections::HashMap;

/// Planted survival marginal-slope model, the #2765 acceptance fixture's: a
/// slope that attenuates along follow-up, `b(t) = 0.85 − 0.32·log t`, over the
/// Weibull-shaped marginal index `q(t) = −1.15 + 0.95·log t`.
const SURVIVAL_LOCATION_LEVEL: f64 = -1.15;
const SURVIVAL_LOCATION_TREND: f64 = 0.95;
const SURVIVAL_SLOPE_LEVEL: f64 = 0.85;
const SURVIVAL_SLOPE_TREND: f64 = -0.32;
const SURVIVAL_SLOPE_TIME_K: usize = 4;
const SURVIVAL_SLOPE_TIME_DEGREE: usize = 2;

fn survival_planted_eta(time: f64, z: f64) -> f64 {
    let slope = SURVIVAL_SLOPE_LEVEL + SURVIVAL_SLOPE_TREND * time.ln();
    let location = SURVIVAL_LOCATION_LEVEL + SURVIVAL_LOCATION_TREND * time.ln();
    location * (1.0 + slope * slope).sqrt() + slope * z
}

/// Standard-normal quantile by bisection, independent of the crate under test.
fn normal_quantile(p: f64) -> f64 {
    let (mut low, mut high) = (-12.0_f64, 12.0_f64);
    for _ in 0..200 {
        let mid = 0.5 * (low + high);
        if normal_cdf(mid) < p {
            low = mid;
        } else {
            high = mid;
        }
    }
    0.5 * (low + high)
}

/// Event times drawn by inverting `Φ(−η(T)) = u` exactly, by bisection on
/// `log T` (`η` is increasing in `t` over the fixture's support).
fn survival_planted_event_time(u: f64, z: f64) -> f64 {
    let target = -normal_quantile(u);
    let (mut low, mut high) = (-6.0_f64, 6.0_f64);
    for _ in 0..200 {
        let mid = 0.5 * (low + high);
        if survival_planted_eta(mid.exp(), z) < target {
            low = mid;
        } else {
            high = mid;
        }
    }
    (0.5 * (low + high)).exp()
}

fn simulate_survival_dataset(n: usize, seed: u64) -> gam_data::EncodedDataset {
    let mut state = seed;
    let headers = ["time", "event", "z"]
        .iter()
        .map(|s| s.to_string())
        .collect::<Vec<_>>();
    let mut records = Vec::with_capacity(n);
    for _ in 0..n {
        let z = gaussian(&mut state);
        let u = uniform(&mut state).clamp(1e-6, 1.0 - 1e-6);
        let event_time = survival_planted_event_time(u, z);
        let censor_time = 0.35 + 5.0 * uniform(&mut state);
        let (time, event) = if event_time <= censor_time {
            (event_time, 1u8)
        } else {
            (censor_time, 0u8)
        };
        let time = time.clamp(1e-3, 1e3);
        records.push(csv::StringRecord::from(vec![
            format!("{time:.17e}"),
            event.to_string(),
            format!("{z:.17e}"),
        ]));
    }
    encode_recordswith_inferred_schema(headers, records).expect("encode survival dataset")
}

/// Fits the survival marginal-slope fixture with the #2765 acceptance test's
/// configuration (intercept-only surfaces, a quadratic follow-up margin on
/// the slope, Weibull baseline). A covariate-bearing variant of this fixture —
/// `Surv(time, event) ~ x` / `s(x, k=5)` with a constant slope — is refused at
/// seed validation on current `main` ("changed profile objective between value
/// screening and derivative assembly"), which is a fit-side defect outside this
/// test's scope; the contexts here therefore collapse and only the latent
/// score and the horizon vary.
fn fit_survival_marginal_slope_model(n: usize, seed: u64) -> FittedModel {
    let ds = simulate_survival_dataset(n, seed);
    let cfg = FitConfig {
        survival_likelihood: Some("marginal-slope".to_string()),
        slope_formula: Some("1".to_string()),
        z_column: Some("z".to_string()),
        frozen_score: true,
        slope_time_k: Some(SURVIVAL_SLOPE_TIME_K),
        slope_time_degree: SURVIVAL_SLOPE_TIME_DEGREE,
        baseline_target: "weibull".to_string(),
        time_num_internal_knots: 3,
        precompute_conformal: Some(false),
        ..FitConfig::default()
    };
    let payload = fit_formula_to_payload("Surv(time, event) ~ 1".to_string(), &ds, &cfg)
        .expect("survival marginal-slope fit");
    FittedModel::from_payload(payload)
}

/// Prediction rows for the survival fixture: the latent scores of the
/// Bernoulli grid (its contexts collapse, see
/// `fit_survival_marginal_slope_model`), with the exit-time column present (a
/// time grid overrides it) and no event.
fn survival_prediction_frame() -> (Array2<f64>, HashMap<String, usize>, Vec<f64>) {
    let scores = vec![-2.0, -1.0, 0.0, 1.0, 2.0];
    let mut data = Array2::<f64>::zeros((scores.len(), 3));
    for (row, &z) in scores.iter().enumerate() {
        data[[row, 0]] = 1.0;
        data[[row, 2]] = z;
    }
    let col_map = HashMap::from([
        ("time".to_string(), 0usize),
        ("event".to_string(), 1usize),
        ("z".to_string(), 2usize),
    ]);
    (data, col_map, scores)
}

/// Which surface a prediction publishes.
#[derive(Clone, Copy, Debug)]
enum Surface {
    PlugIn,
    /// `predict_survival` under its default posterior-mean estimand.
    Published,
    /// The posterior mean under a named integration.
    Posterior(SurvivalPosteriorIntegration),
}

/// One prediction's survival `S(t)`, event density `f(t) = S(t)·h(t)` and
/// hazard `h(t)` over `times`, row × time.
struct Surfaces {
    survival: Array2<f64>,
    density: Array2<f64>,
    hazard: Array2<f64>,
}

fn survival_surfaces(
    model: &FittedModel,
    data: &Array2<f64>,
    col_map: &HashMap<String, usize>,
    times: &[f64],
    surface: Surface,
) -> Surfaces {
    let zero = Array1::<f64>::zeros(data.nrows());
    let request = SurvivalPredictRequest {
        model,
        data: data.view(),
        col_map,
        training_headers: model.training_headers.as_ref(),
        primary_offset: &zero,
        noise_offset: &zero,
        time_grid: Some(times),
        with_uncertainty: false,
        estimand: SurvivalPredictEstimand::Plugin,
    };
    let mode = SurvivalPredictionCovarianceMode::Conditional;
    let result = match surface {
        Surface::PlugIn => predict_survival(request, mode),
        Surface::Published => predict_survival(
            SurvivalPredictRequest {
                estimand: SurvivalPredictEstimand::PosteriorMean,
                ..request
            },
            mode,
        ),
        Surface::Posterior(integration) => {
            predict_survival_posterior_mean_with(request, mode, integration)
        }
    }
    .unwrap_or_else(|e| panic!("survival {surface:?} prediction: {e}"));
    Surfaces {
        density: &result.survival * &result.hazard,
        survival: result.survival,
        hazard: result.hazard,
    }
}

/// Write a coefficient draw into a saved survival model, the way the
/// posterior quadrature does for each of its nodes.
fn assign_survival_coefficients(model: &mut FittedModel, draw: &Array1<f64>) {
    assert!(
        model.beta_baseline_timewiggle.is_none() && model.survival_beta_time.is_none(),
        "this fixture carries its coefficients in the fit result only"
    );
    let fit = model
        .fit_result
        .as_mut()
        .expect("survival model carries its fit result");
    fit.beta.assign(draw);
    let mut cursor = 0usize;
    for block in &mut fit.blocks {
        let end = cursor + block.beta.len();
        block.beta.assign(&draw.slice(ndarray::s![cursor..end]));
        cursor = end;
    }
    assert_eq!(cursor, draw.len());
}

/// The posterior-predictive surfaces by Monte Carlo over the survival model's
/// own posterior, each with the standard error of its estimate: survival
/// `mean(S)`, event density `mean(f)`, and the hazard of the predictive law
/// `mean(f)/mean(S)` (delta-method standard error of a ratio of means). The
/// posterior mean of the per-draw hazard, `mean(f/S)`, is kept only to show how
/// far it sits from the predictive hazard.
struct MonteCarloReference {
    survival: Array2<f64>,
    survival_se: Array2<f64>,
    density: Array2<f64>,
    density_se: Array2<f64>,
    hazard: Array2<f64>,
    hazard_se: Array2<f64>,
    mean_draw_hazard: Array2<f64>,
}

/// Every draw is run through the production plug-in prediction, anchor
/// re-solved per draw.
fn survival_monte_carlo_reference(
    model: &FittedModel,
    data: &Array2<f64>,
    col_map: &HashMap<String, usize>,
    times: &[f64],
    draws: usize,
    seed: u64,
) -> MonteCarloReference {
    use rayon::iter::{IntoParallelIterator, ParallelIterator};
    let fit = model.fit_result.as_ref().expect("fit result");
    let covariance = fit
        .covariance_conditional
        .as_ref()
        .expect("survival fit carries a conditional covariance");
    let p = fit.beta.len();
    assert_eq!(covariance.dim(), (p, p));
    assert_eq!(
        model
            .saved_prediction_runtime()
            .expect("prediction runtime")
            .influence_absorber_width
            .unwrap_or(0),
        0,
        "a frozen score carries no influence absorber; every coefficient is active"
    );
    let theta_hat = fit.beta.clone();
    let factor = psd_factor(covariance);
    let shape = (data.nrows(), times.len());
    // Running sums of S, S², f, f², f·S and h over the draws.
    let zeros = || -> [Array2<f64>; 6] { std::array::from_fn(|_| Array2::<f64>::zeros(shape)) };
    let chunks = 48usize;
    let per_chunk = draws.div_ceil(chunks);
    let sums = (0..chunks)
        .into_par_iter()
        .map(|chunk| {
            let mut state = seed ^ (chunk as u64 + 1).wrapping_mul(0x9E37_79B9_7F4A_7C15);
            let mut draw_model = model.clone();
            let mut sums = zeros();
            let mut standard = Array1::<f64>::zeros(p);
            let first = chunk * per_chunk;
            let last = ((chunk + 1) * per_chunk).min(draws);
            for _ in first..last {
                standard.mapv_inplace(|_| gaussian(&mut state));
                let theta = &theta_hat + &factor.dot(&standard);
                assign_survival_coefficients(&mut draw_model, &theta);
                let draw = survival_surfaces(&draw_model, data, col_map, times, Surface::PlugIn);
                sums[0] += &draw.survival;
                sums[1] += &draw.survival.mapv(|s| s * s);
                sums[2] += &draw.density;
                sums[3] += &draw.density.mapv(|f| f * f);
                sums[4] += &(&draw.density * &draw.survival);
                sums[5] += &draw.hazard;
            }
            sums
        })
        .reduce(zeros, |mut a, b| {
            for (slot, value) in a.iter_mut().zip(b) {
                *slot += &value;
            }
            a
        });
    let count = draws as f64;
    let [sum_s, sum_s2, sum_f, sum_f2, sum_fs, sum_h] = sums;
    let survival = sum_s.mapv(|v| v / count);
    let density = sum_f.mapv(|v| v / count);
    let variance = |second: &Array2<f64>, mean: &Array2<f64>| {
        Array2::from_shape_fn(shape, |cell| {
            (second[cell] / count - mean[cell].powi(2)).max(0.0)
        })
    };
    let var_s = variance(&sum_s2, &survival);
    let var_f = variance(&sum_f2, &density);
    let cov_fs = Array2::from_shape_fn(shape, |cell| {
        sum_fs[cell] / count - density[cell] * survival[cell]
    });
    let hazard = &density / &survival;
    let hazard_se = Array2::from_shape_fn(shape, |cell| {
        let (s, f) = (survival[cell], density[cell]);
        let ratio_variance = var_f[cell] / (s * s) - 2.0 * f * cov_fs[cell] / (s * s * s)
            + f * f * var_s[cell] / (s * s * s * s);
        (ratio_variance.max(0.0) / count).sqrt()
    });
    MonteCarloReference {
        survival_se: var_s.mapv(|v| (v / count).sqrt()),
        density_se: var_f.mapv(|v| (v / count).sqrt()),
        survival,
        density,
        hazard,
        hazard_se,
        mean_draw_hazard: sum_h.mapv(|v| v / count),
    }
}

/// max |Δ| against the Monte Carlo reference over every cell, and the
/// sigma-point rule's worst survival gap in Monte Carlo standard errors.
#[derive(Clone, Copy)]
struct Measured {
    plugin: f64,
    sigma_point: f64,
    exact: f64,
    sigma_point_density: f64,
    exact_density: f64,
    sigma_point_hazard: f64,
    exact_hazard: f64,
    sigma_point_gap_se: f64,
}

/// Measures the plug-in, sigma-point and exact survival surfaces (and the two
/// posterior densities and predictive hazards) of `model` against Monte Carlo,
/// prints the per-cell table, and requires the exact integration to sit inside
/// the Monte Carlo resolution at every cell and to be what the posterior-mean
/// pass publishes.
fn measure(
    label: &str,
    model: &FittedModel,
    data: &Array2<f64>,
    col_map: &HashMap<String, usize>,
    times: &[f64],
    scores: &[f64],
    draws: usize,
    seed: u64,
) -> Measured {
    assert_eq!(
        SurvivalPosteriorIntegration::default_for(model).expect("default integration"),
        SurvivalPosteriorIntegration::ExactAnchor,
        "[{label}] a rigid survival marginal-slope model publishes the exact integration"
    );
    let plugin = survival_surfaces(model, data, col_map, times, Surface::PlugIn);
    let sigma_point = survival_surfaces(
        model,
        data,
        col_map,
        times,
        Surface::Posterior(SurvivalPosteriorIntegration::SigmaPoint),
    );
    let exact = survival_surfaces(
        model,
        data,
        col_map,
        times,
        Surface::Posterior(SurvivalPosteriorIntegration::ExactAnchor),
    );
    let published = survival_surfaces(model, data, col_map, times, Surface::Published);
    for (cell, &value) in published.survival.indexed_iter() {
        assert_eq!(
            (value.to_bits(), published.hazard[cell].to_bits()),
            (exact.survival[cell].to_bits(), exact.hazard[cell].to_bits()),
            "[{label}] the posterior-mean pass publishes the exact integration at cell {cell:?}"
        );
    }
    let reference = survival_monte_carlo_reference(model, data, col_map, times, draws, seed);
    eprintln!(
        "[{label}] {:>5} {:>5} | {:>9} {:>8} | {:>10} {:>10} {:>10} | {:>9} {:>8} | {:>10} {:>10} \
         | {:>9} {:>8} | {:>10} {:>10} {:>10}",
        "z",
        "t",
        "S mc",
        "mc_se",
        "plug-in",
        "sigma-pt",
        "exact",
        "f mc",
        "mc_se",
        "sigma-pt",
        "exact",
        "h mc",
        "mc_se",
        "sigma-pt",
        "exact",
        "E[f/S]"
    );
    let mut measured = Measured {
        plugin: 0.0,
        sigma_point: 0.0,
        exact: 0.0,
        sigma_point_density: 0.0,
        exact_density: 0.0,
        sigma_point_hazard: 0.0,
        exact_hazard: 0.0,
        sigma_point_gap_se: 0.0,
    };
    let (mut max_se, mut worst_ratio) = (0.0_f64, 0.0_f64);
    for (row, &z) in scores.iter().enumerate() {
        for (column, &t) in times.iter().enumerate() {
            let cell = [row, column];
            let (s_mc, s_se) = (reference.survival[cell], reference.survival_se[cell]);
            let (f_mc, f_se) = (reference.density[cell], reference.density_se[cell]);
            let (h_mc, h_se) = (reference.hazard[cell], reference.hazard_se[cell]);
            let deltas = [
                plugin.survival[cell] - s_mc,
                sigma_point.survival[cell] - s_mc,
                exact.survival[cell] - s_mc,
                sigma_point.density[cell] - f_mc,
                exact.density[cell] - f_mc,
                sigma_point.hazard[cell] - h_mc,
                exact.hazard[cell] - h_mc,
                reference.mean_draw_hazard[cell] - h_mc,
            ];
            eprintln!(
                "[{label}] {z:>5.1} {t:>5.2} | {s_mc:>9.5} {s_se:>8.1e} | {:>+10.2e} {:>+10.2e} {:>+10.2e} \
                 | {f_mc:>9.5} {f_se:>8.1e} | {:>+10.2e} {:>+10.2e} \
                 | {h_mc:>9.5} {h_se:>8.1e} | {:>+10.2e} {:>+10.2e} {:>+10.2e}",
                deltas[0],
                deltas[1],
                deltas[2],
                deltas[3],
                deltas[4],
                deltas[5],
                deltas[6],
                deltas[7]
            );
            measured.plugin = measured.plugin.max(deltas[0].abs());
            measured.sigma_point = measured.sigma_point.max(deltas[1].abs());
            measured.exact = measured.exact.max(deltas[2].abs());
            measured.sigma_point_density = measured.sigma_point_density.max(deltas[3].abs());
            measured.exact_density = measured.exact_density.max(deltas[4].abs());
            measured.sigma_point_hazard = measured.sigma_point_hazard.max(deltas[5].abs());
            measured.exact_hazard = measured.exact_hazard.max(deltas[6].abs());
            max_se = max_se.max(s_se);
            worst_ratio = worst_ratio.max(deltas[2].abs() / (s_se + 1e-12));
            measured.sigma_point_gap_se = measured
                .sigma_point_gap_se
                .max(deltas[1].abs() / (s_se + 1e-12));
            // The exact integration and the Monte Carlo reference estimate the
            // same integrals; 4 standard errors is the widest gap sampling alone
            // explains.
            for (name, delta, se) in [
                ("survival", deltas[2], s_se),
                ("event density", deltas[4], f_se),
                ("predictive hazard", deltas[6], h_se),
            ] {
                assert!(
                    delta.abs() <= 4.0 * se + 1e-6,
                    "[{label}] exact {name} at z={z}, t={t} is {delta:+.3e} from Monte Carlo (se {se:.2e})"
                );
            }
        }
    }
    eprintln!(
        "[{label}] max|Δ| vs MC ({draws} draws, max S se {max_se:.1e}): S plug-in {:.2e}, \
         sigma-point {:.2e} (worst gap {:.1} MC se), exact {:.2e} (worst exact gap \
         {worst_ratio:.1} MC se); f sigma-point {:.2e}, exact {:.2e}; h sigma-point {:.2e}, \
         exact {:.2e}",
        measured.plugin,
        measured.sigma_point,
        measured.sigma_point_gap_se,
        measured.exact,
        measured.sigma_point_density,
        measured.exact_density,
        measured.sigma_point_hazard,
        measured.exact_hazard
    );
    measured
}

#[test]
fn survival_marginal_slope_posterior_mean_against_monte_carlo() {
    init_parallelism();
    let model = fit_survival_marginal_slope_model(800, 20260917);
    let (data, col_map, scores) = survival_prediction_frame();
    let times = [0.75, 1.5, 3.0];
    // Each Monte Carlo draw is a full survival prediction, so the reference
    // uses fewer draws than the Bernoulli test; the standard error is reported
    // and the comparison is stated in units of it.
    let draws = 20_000;

    // The same fit with its covariance inflated 9× (SEs ×3), where the cubature
    // error of the sigma-point rule grows well past the Monte Carlo resolution.
    let mut wide_model = model.clone();
    {
        let fit = wide_model.fit_result.as_mut().expect("fit result");
        let wide = fit
            .covariance_conditional
            .as_ref()
            .expect("covariance")
            .mapv(|v| 9.0 * v);
        fit.covariance_conditional = Some(wide);
    }
    // A declared empirical latent law that is not standard normal, so the anchor
    // `a(q, b)` is a root solve at every node and every draw.
    let declared_law = |base: &FittedModel| {
        let mut declared = base.clone();
        declared.latent_measure = Some(LatentMeasureKind::GlobalEmpirical {
            grid: heavy_tailed_grid(),
        });
        declared
    };

    // (label, model, Monte Carlo seed, covariance inflated)
    let configurations = [
        ("rigid/fitted-V", model.clone(), 11, false),
        ("rigid/9×V", wide_model.clone(), 12, true),
        ("declared-law/fitted-V", declared_law(&model), 13, false),
        ("declared-law/9×V", declared_law(&wide_model), 14, true),
    ];
    let measured: Vec<(&str, Measured, bool)> = configurations
        .iter()
        .map(|(label, configured, seed, inflated)| {
            (
                *label,
                measure(
                    label, configured, &data, &col_map, &times, &scores, draws, *seed,
                ),
                *inflated,
            )
        })
        .collect();

    for (label, m, _) in &measured {
        assert!(
            m.exact <= m.plugin + 1e-12,
            "[{label}] the exact posterior-mean survival surface (max|ΔS| {:.2e}) is further from \
             Monte Carlo than the plug-in ({:.2e})",
            m.exact,
            m.plugin
        );
    }
    // Under the inflated covariance the sigma-point rule must be visibly off
    // where the exact integration is not: otherwise this test could not tell
    // the two apart and would not catch a regression to the cubature. Both are
    // judged in Monte Carlo standard errors, cell by cell: every exact cell sits
    // within 4 of them (asserted in `measure`), the widest gap sampling alone
    // explains, so the sigma-point rule is told apart exactly when some cell
    // misses by more than that same bar. A ratio of the two max|ΔS| cannot
    // make this call, because the exact rule's max|ΔS| is the Monte Carlo noise
    // of the widest cell, not an integration error.
    for (label, m, _) in measured.iter().filter(|(_, _, inflated)| *inflated) {
        assert!(
            m.sigma_point_gap_se > 4.0,
            "[{label}] the sigma-point rule's worst survival cell is {:.1} MC standard errors \
             from Monte Carlo (max|ΔS| {:.2e}); it is not separated from the exact integration \
             (max|ΔS| {:.2e}) at this posterior spread",
            m.sigma_point_gap_se,
            m.sigma_point,
            m.exact
        );
    }
}
