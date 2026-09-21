//! gam#2948 acceptance: survival marginal-slope fits with flex blocks anchor on
//! a finite latent law.
//!
//! These configurations were refused by name before gam#2948, because the flex
//! row program integrated the marginal identity against the standard normal
//! only. Two fixtures:
//!
//! 1. **A declared skewed law** (the gam#2923 fixture, simulated from the rigid
//!    anchored model) with a link deviation. The saved flex model, replayed at
//!    every node `u_k` of the declared law for a training row's exit time, has
//!    law-weighted survival `Σ_k w_k Ŝ(t | u_k) = Φ(−q̂(t))`, and it is calibrated
//!    in context.
//! 2. **gnomon-2370's known-truth calibrate fixture** (gnomon#2370): a probit
//!    Weibull baseline, delayed entry, a standardized Gamma(4) score law and a
//!    planted cubic score warp and link deviation, fitted with gnomon's request
//!    document (`linkwiggle()` in the survival and slope formulas). The flex arm
//!    fits on the requested global-empirical law and on the default law, and
//!    each fit's survival at exit is measured against the planted truth.

use csv::StringRecord;
use gam_data::encode_recordswith_inferred_schema;
use gam_models::fit_orchestration::{DeclaredLatentLaw, FitConfig, FitResult, fit_from_formula};
use gam_models::inference::model::FittedModel;
use gam_models::inference::model_payload_builders::fit_formula_to_payload;
use gam_models::survival::{
    SurvivalPredictEstimand, SurvivalPredictRequest, SurvivalPredictionCovarianceMode,
    predict_survival,
};
use ndarray::Array1;
use std::collections::HashMap;

use gam_linalg::utils::splitmix64;

fn normal_cdf(x: f64) -> f64 {
    gam_math::probability::normal_cdf(x)
}

fn normal_pdf(x: f64) -> f64 {
    gam_math::probability::normal_pdf(x)
}

fn encode(headers: &[&str], rows: Vec<Vec<f64>>) -> gam_data::EncodedDataset {
    let headers = headers.iter().map(|s| s.to_string()).collect::<Vec<_>>();
    let records = rows
        .into_iter()
        .map(|row| StringRecord::from(row.iter().map(f64::to_string).collect::<Vec<_>>()))
        .collect();
    encode_recordswith_inferred_schema(headers, records).expect("encode the #2948 fixture")
}

/// The saved model's survival at each row of `data`, at the row's own time.
fn replay_survival(
    model: &FittedModel,
    data: &gam_data::EncodedDataset,
    training: &gam_data::EncodedDataset,
) -> Vec<f64> {
    let col_map: HashMap<String, usize> = data
        .headers
        .iter()
        .enumerate()
        .map(|(index, name)| (name.clone(), index))
        .collect();
    let zeros = Array1::<f64>::zeros(data.values.nrows());
    let prediction = predict_survival(
        SurvivalPredictRequest {
            model,
            data: data.values.view(),
            col_map: &col_map,
            training_headers: Some(&training.headers),
            primary_offset: &zeros,
            noise_offset: &zeros,
            time_grid: None,
            with_uncertainty: false,
            estimand: SurvivalPredictEstimand::Plugin,
        },
        SurvivalPredictionCovarianceMode::Conditional,
    )
    .expect("saved survival marginal-slope prediction");
    (0..data.values.nrows())
        .map(|row| prediction.survival[[row, 0]])
        .collect()
}

mod declared_skewed_law {
    use super::*;

    const N: usize = 2_400;
    const SLOPE: f64 = 1.6;
    const LOCATION_LEVEL: f64 = -1.15;
    const LOCATION_TREND: f64 = 0.95;
    const FORMULA: &str = "Surv(time, event) ~ 1 + linkwiggle(degree=3, internal_knots=2)";

    fn next_unit(state: &mut u64) -> f64 {
        (splitmix64(state) >> 11) as f64 / (1u64 << 53) as f64
    }

    /// Standard-normal quantile by bisection on `Φ`.
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

    /// The gam#2923 skewed two-component law on 41 nodes, standardised to zero
    /// mean and unit variance.
    struct Law {
        nodes: Vec<f64>,
        weights: Vec<f64>,
    }

    impl Law {
        fn skewed() -> Self {
            let raw_nodes: Vec<f64> = (0..41).map(|k| -2.5 + 0.15 * k as f64).collect();
            let raw: Vec<f64> = raw_nodes
                .iter()
                .map(|&u| {
                    (-0.5 * ((u + 0.9) / 0.5).powi(2)).exp()
                        + 0.20 * (-0.5 * ((u - 2.8) / 0.5).powi(2)).exp()
                })
                .collect();
            let total: f64 = raw.iter().sum();
            let weights: Vec<f64> = raw.into_iter().map(|w| w / total).collect();
            let mean: f64 = raw_nodes.iter().zip(&weights).map(|(u, w)| u * w).sum();
            let var: f64 = raw_nodes
                .iter()
                .zip(&weights)
                .map(|(u, w)| (u - mean).powi(2) * w)
                .sum();
            let nodes = raw_nodes.iter().map(|u| (u - mean) / var.sqrt()).collect();
            Self { nodes, weights }
        }

        fn marginal_survival(&self, alpha: f64, slope: f64) -> f64 {
            self.nodes
                .iter()
                .zip(&self.weights)
                .map(|(&u, &w)| w * normal_cdf(-(alpha + slope * u)))
                .sum()
        }

        /// The anchor `α(q, b)` on this law, by bisection.
        fn anchor(&self, q: f64, slope: f64) -> f64 {
            let target = normal_cdf(-q);
            let (mut low, mut high) = (-40.0_f64, 40.0_f64);
            for _ in 0..200 {
                let mid = 0.5 * (low + high);
                if self.marginal_survival(mid, slope) > target {
                    low = mid;
                } else {
                    high = mid;
                }
            }
            0.5 * (low + high)
        }

        fn draw(&self, u: f64) -> f64 {
            let mut cumulative = 0.0;
            for (&node, &weight) in self.nodes.iter().zip(&self.weights) {
                cumulative += weight;
                if u < cumulative {
                    return node;
                }
            }
            *self.nodes.last().expect("non-empty law")
        }

        fn declared(&self) -> DeclaredLatentLaw {
            DeclaredLatentLaw {
                nodes: self.nodes.clone(),
                weights: self.weights.clone(),
            }
        }
    }

    fn planted_index(time: f64) -> f64 {
        LOCATION_LEVEL + LOCATION_TREND * time.ln()
    }

    /// Simulate from the anchored model on `law`: `S(t | z) = Φ(−(α(q(t), b) + b·z))`.
    fn build_dataset(law: &Law, seed: u64) -> (gam_data::EncodedDataset, Vec<f64>, Vec<f64>) {
        let mut state = seed;
        let draws: Vec<f64> = (0..N).map(|_| law.draw(next_unit(&mut state))).collect();
        let mut rows = Vec::with_capacity(N);
        let mut times = Vec::with_capacity(N);
        for &z in &draws {
            let u = next_unit(&mut state).clamp(1e-6, 1.0 - 1e-6);
            let censor = 0.35 + 5.0 * next_unit(&mut state);
            let target = -normal_quantile(u) - SLOPE * z;
            let (mut low, mut high) = (-6.0_f64, 6.0_f64);
            for _ in 0..200 {
                let mid = 0.5 * (low + high);
                if law.anchor(planted_index(mid.exp()), SLOPE) < target {
                    low = mid;
                } else {
                    high = mid;
                }
            }
            let event_time = (0.5 * (low + high)).exp();
            let (time, event) = if event_time <= censor {
                (event_time, 1.0)
            } else {
                (censor, 0.0)
            };
            let time = time.clamp(1e-3, 1e3);
            times.push(time);
            rows.push(vec![time, event, z]);
        }
        (encode(&["time", "event", "z"], rows), times, draws)
    }

    #[test]
    fn flex_fit_on_a_declared_law_is_anchored_and_calibrated_2948() {
        crate::initialize_cpu_fitting();
        gam_runtime::test_support::install_diagnostic_logger();
        #[cfg(target_os = "macos")]
        gam_gpu::configure_global_policy(gam_gpu::GpuPolicy::Off);

        let law = Law::skewed();
        let (data, times, scores) = build_dataset(&law, 0x2948_0000_0001);
        let config = FitConfig {
            survival_likelihood: Some("marginal-slope".to_string()),
            z_column: Some("z".to_string()),
            slope_formula: Some("1".to_string()),
            time_num_internal_knots: 3,
            declared_latent_law: Some(law.declared()),
            ..FitConfig::default()
        };

        let result = fit_from_formula(FORMULA, &data, &config)
            .expect("a link-deviation fit on a declared latent law");
        let FitResult::SurvivalMarginalSlope(fit) = result else {
            panic!("expected a SurvivalMarginalSlope fit result");
        };
        assert!(
            matches!(
                fit.latent_measure,
                gam_models::bms::LatentMeasureKind::GlobalEmpirical { .. }
            ),
            "the declared law must be the flex fit's measure; got {:?}",
            fit.latent_measure
        );
        let payload = fit_formula_to_payload(FORMULA.to_string(), &data, &config)
            .expect("fit to a saved payload");
        let model = FittedModel::from_payload(payload);

        // Claim 1: replay a spread of training rows at every node of the law.
        let replay_rows: Vec<usize> = (0..N).step_by(80).collect();
        let mut node_rows = Vec::with_capacity(replay_rows.len() * law.nodes.len());
        for &row in &replay_rows {
            for &u in &law.nodes {
                node_rows.push(vec![times[row], 0.0, u]);
            }
        }
        let node_survival =
            replay_survival(&model, &encode(&["time", "event", "z"], node_rows), &data);
        let mut worst_anchor_miss = 0.0_f64;
        for (index, &row) in replay_rows.iter().enumerate() {
            let start = index * law.nodes.len();
            let marginal: f64 = node_survival[start..start + law.nodes.len()]
                .iter()
                .zip(&law.weights)
                .map(|(survival, weight)| weight * survival)
                .sum();
            worst_anchor_miss =
                worst_anchor_miss.max((marginal - normal_cdf(-fit.fitted_exit_index[row])).abs());
        }

        // Claim 2: each subject's survival at its own exit time against the truth.
        let survival = replay_survival(&model, &data, &data);
        let conditional_error = times
            .iter()
            .zip(&scores)
            .zip(&survival)
            .map(|((&time, &z), &predicted)| {
                let truth = normal_cdf(-(law.anchor(planted_index(time), SLOPE) + SLOPE * z));
                (predicted - truth).abs()
            })
            .sum::<f64>()
            / N as f64;
        eprintln!(
            "[2948 flex declared law] n={N} | max |Σ_k w_k Ŝ(t|u_k) − Φ(−q̂(t))| over {} rows = \
             {worst_anchor_miss:.3e} | mean |Ŝ(t,z) − S(t,z)| = {conditional_error:.4}",
            replay_rows.len()
        );
        // The predictor solves each node calibration to the empirical intercept
        // tolerance; the gam#2923 replay pins the anchored index to 1e-6.
        assert!(
            worst_anchor_miss < 1e-6,
            "the saved flex model's law-weighted survival must be the marginal survival \
             Φ(−q̂); worst miss {worst_anchor_miss:.3e}"
        );
        assert!(
            conditional_error < 0.02,
            "the anchored flex fit must be calibrated in context; mean |Ŝ − S| = \
             {conditional_error:.4}"
        );
    }
}

/// A flex block over K ≥ 2 scores stays refused by name: the flex row program
/// anchors each timepoint on the scalar law of one score, and a joint law of the
/// score vector has none.
mod several_scores {
    use super::*;

    #[test]
    fn flex_block_over_several_scores_is_refused_by_name_2948() {
        crate::initialize_cpu_fitting();
        gam_runtime::test_support::install_diagnostic_logger();
        #[cfg(target_os = "macos")]
        gam_gpu::configure_global_policy(gam_gpu::GpuPolicy::Off);

        let n = 400;
        let mut state = 0x2948_0000_0002_u64;
        let mut unit = || ((splitmix64(&mut state) >> 11) as f64 + 0.5) / (1u64 << 53) as f64;
        let rows: Vec<Vec<f64>> = (0..n)
            .map(|_| {
                let z0 = 2.0 * unit() - 1.0;
                let z1 = 2.0 * unit() - 1.0;
                let time = 0.2 + 3.0 * unit();
                let event = if unit() < 0.6 { 1.0 } else { 0.0 };
                vec![time, event, z0, z1]
            })
            .collect();
        let data = encode(&["time", "event", "z0", "z1"], rows);
        let config = FitConfig {
            survival_likelihood: Some("marginal-slope".to_string()),
            z_column: Some("z0".to_string()),
            slope_formula: Some("slope(z0, 1) + slope(z1, 1)".to_string()),
            latent_measure: Some("global-empirical".to_string()),
            ..FitConfig::default()
        };
        let started = std::time::Instant::now();
        let error = match fit_from_formula(
            "Surv(time, event) ~ 1 + linkwiggle(degree=3, internal_knots=2)",
            &data,
            &config,
        ) {
            Ok(_) => panic!("a link deviation over K = 2 scores on a finite law must refuse, and it fitted"),
            Err(error) => error.to_string(),
        };
        eprintln!(
            "[2948 several scores] refused in {:.3}s: {error}",
            started.elapsed().as_secs_f64()
        );
        assert!(
            error.contains("K ≥ 2 scores") && error.contains("link-deviation flex block"),
            "a flex block over several scores must refuse by the flex program's own reason; got {error}"
        );
    }
}

/// gnomon-2370's known-truth calibrate survival fixture, flex arm (gnomon#2370).
mod known_truth_flex_arm {
    use super::*;

    const ROWS: usize = 2000;
    const THETA0: f64 = -6.99;
    const THETA1: f64 = 1.706;
    const GAMMA_SEX: f64 = 0.25;
    const SLOPE: f64 = 0.8;
    const WARP: f64 = 0.05;
    const LINK: f64 = 0.02;
    const SEED: u64 = 0x2370_2941;
    const FORMULA: &str = "Surv(age_entry, age_exit, event_target) ~ sex + linkwiggle()";

    struct SplitMix(u64);

    impl SplitMix {
        fn next_u64(&mut self) -> u64 {
            self.0 = self.0.wrapping_add(0x9e37_79b9_7f4a_7c15);
            let mut z = self.0;
            z = (z ^ (z >> 30)).wrapping_mul(0xbf58_476d_1ce4_e5b9);
            z = (z ^ (z >> 27)).wrapping_mul(0x94d0_49bb_1331_11eb);
            z ^ (z >> 31)
        }

        fn uniform(&mut self) -> f64 {
            ((self.next_u64() >> 11) as f64 + 0.5) / (1u64 << 53) as f64
        }
    }

    fn gamma4_quantile(p: f64) -> f64 {
        let cdf = |x: f64| {
            if x <= 0.0 {
                0.0
            } else {
                1.0 - (-x).exp() * (1.0 + x + x * x / 2.0 + x * x * x / 6.0)
            }
        };
        let (mut low, mut high) = (0.0_f64, 80.0_f64);
        for _ in 0..200 {
            let middle = 0.5 * (low + high);
            if cdf(middle) < p {
                low = middle;
            } else {
                high = middle;
            }
        }
        0.5 * (low + high)
    }

    /// One standardized Gamma(4) mid-quantile per row (skewness 1, excess kurtosis 1.5).
    fn score_atoms() -> Vec<f64> {
        let raw: Vec<f64> = (0..ROWS)
            .map(|i| gamma4_quantile((i as f64 + 0.5) / ROWS as f64))
            .collect();
        let mean = raw.iter().sum::<f64>() / ROWS as f64;
        let sd = (raw.iter().map(|value| (value - mean).powi(2)).sum::<f64>() / ROWS as f64).sqrt();
        raw.iter().map(|value| (value - mean) / sd).collect()
    }

    fn index(alpha: f64, z: f64) -> f64 {
        let rigid = alpha + SLOPE * z;
        rigid + SLOPE * WARP * (z * z * z - 3.0 * z) + LINK * (rigid * rigid * rigid - 3.0 * rigid)
    }

    fn marginal_index(age: f64, sex: f64) -> f64 {
        THETA0 + THETA1 * age.ln() + GAMMA_SEX * sex
    }

    /// The planted anchor `α` solving `mean_k Φ(−η(α, u_k)) = Φ(−q)` over the score
    /// law, tabulated on the fixture's index range and read by linear interpolation.
    struct Truth {
        q_start: f64,
        q_step: f64,
        alpha: Vec<f64>,
    }

    impl Truth {
        fn new(atoms: &[f64]) -> Self {
            let solve = |q: f64| -> f64 {
                let target = normal_cdf(-q);
                let (mut low, mut high) = (-30.0_f64, 30.0_f64);
                let mut alpha = q;
                for _ in 0..100 {
                    let mut value = -target;
                    let mut derivative = 0.0;
                    for &u in atoms {
                        let eta = index(alpha, u);
                        let rigid = alpha + SLOPE * u;
                        value += normal_cdf(-eta) / atoms.len() as f64;
                        derivative -= normal_pdf(eta) * (1.0 + LINK * (3.0 * rigid * rigid - 3.0))
                            / atoms.len() as f64;
                    }
                    if value > 0.0 {
                        low = alpha;
                    } else {
                        high = alpha;
                    }
                    let newton = alpha - value / derivative;
                    let next = if derivative < 0.0 && newton > low && newton < high {
                        newton
                    } else {
                        0.5 * (low + high)
                    };
                    if (next - alpha).abs() < 1e-13 {
                        return next;
                    }
                    alpha = next;
                }
                alpha
            };
            let q_start = marginal_index(19.0, 0.0);
            let q_end = marginal_index(53.0, 1.0);
            let points = 2401;
            let q_step = (q_end - q_start) / (points - 1) as f64;
            let alpha = (0..points)
                .map(|i| solve(q_start + i as f64 * q_step))
                .collect();
            Self {
                q_start,
                q_step,
                alpha,
            }
        }

        fn survival(&self, age: f64, sex: f64, z: f64) -> f64 {
            let position = (marginal_index(age, sex) - self.q_start) / self.q_step;
            let lower = (position.floor().max(0.0) as usize).min(self.alpha.len() - 2);
            let fraction = position - lower as f64;
            let alpha = self.alpha[lower] * (1.0 - fraction) + self.alpha[lower + 1] * fraction;
            normal_cdf(-index(alpha, z))
        }
    }

    /// Entry ages `20 + (i % 5)`, administrative censoring at `40 + (i % 13)`,
    /// `sex = i % 2`, unit weights, and the score atoms shuffled across rows.
    /// Returns the dataset and each row's planted survival at its exit age.
    fn generate(truth: &Truth) -> (gam_data::EncodedDataset, Vec<f64>) {
        let mut rng = SplitMix(SEED);
        let mut z = score_atoms();
        for i in (1..ROWS).rev() {
            let j = (rng.next_u64() % (i as u64 + 1)) as usize;
            z.swap(i, j);
        }
        let mut rows = Vec::with_capacity(ROWS);
        let mut truth_exit = Vec::with_capacity(ROWS);
        for i in 0..ROWS {
            let entry = 20.0 + (i % 5) as f64;
            let censor = 40.0 + (i % 13) as f64;
            let sex = (i % 2) as f64;
            let s_entry = truth.survival(entry, sex, z[i]);
            let u = rng.uniform();
            let (exit, event) = if truth.survival(censor, sex, z[i]) / s_entry >= u {
                (censor, 0.0)
            } else {
                let (mut low, mut high) = (entry, censor);
                for _ in 0..80 {
                    let middle = 0.5 * (low + high);
                    if truth.survival(middle, sex, z[i]) / s_entry > u {
                        low = middle;
                    } else {
                        high = middle;
                    }
                }
                (0.5 * (low + high), 1.0)
            };
            truth_exit.push(truth.survival(exit, sex, z[i]));
            rows.push(vec![z[i], sex, entry, exit, event, 1.0]);
        }
        (
            encode(
                &["score", "sex", "age_entry", "age_exit", "event_target", "weight"],
                rows,
            ),
            truth_exit,
        )
    }

    /// gnomon-2370's request document, with the latent measure of the arm.
    fn request(data: &gam_data::EncodedDataset, latent_measure: Option<&str>) -> FitConfig {
        let exit_column = data
            .headers
            .iter()
            .position(|name| name == "age_exit")
            .expect("age_exit column");
        let baseline_scale = data.values.column(exit_column).sum() / data.values.nrows() as f64;
        FitConfig {
            survival_likelihood: Some("marginal-slope".to_string()),
            slope_formula: Some("1 + linkwiggle()".to_string()),
            z_column: Some("score".to_string()),
            latent_measure: latent_measure.map(str::to_string),
            time_basis: "ispline".to_string(),
            time_degree: 3,
            time_num_internal_knots: 4,
            baseline_target: "weibull".to_string(),
            baseline_scale: Some(baseline_scale),
            baseline_shape: Some(1.0),
            weight_column: Some("weight".to_string()),
            ..FitConfig::default()
        }
    }

    fn arm(label: &str, latent_measure: Option<&str>) {
        crate::initialize_cpu_fitting();
        gam_runtime::test_support::install_diagnostic_logger();
        #[cfg(target_os = "macos")]
        gam_gpu::configure_global_policy(gam_gpu::GpuPolicy::Off);

        let truth = Truth::new(&score_atoms());
        let (data, truth_exit) = generate(&truth);
        let started = std::time::Instant::now();
        let payload = fit_formula_to_payload(FORMULA.to_string(), &data, &request(&data, latent_measure))
            .unwrap_or_else(|error| {
                panic!("the flex arm ({label}) must fit on gnomon-2370's request; refused: {error}")
            });
        let seconds = started.elapsed().as_secs_f64();
        let measure_is_empirical = matches!(
            payload.latent_measure,
            Some(gam_models::bms::LatentMeasureKind::GlobalEmpirical { .. })
        );
        eprintln!(
            "[2948 known truth {label}] fit_seconds {seconds:.1} latent_measure_empirical \
             {measure_is_empirical}"
        );
        let model = FittedModel::from_payload(payload);
        let survival = replay_survival(&model, &data, &data);
        let errors: Vec<f64> = survival
            .iter()
            .zip(&truth_exit)
            .map(|(predicted, planted)| (predicted - planted).abs())
            .collect();
        let max_error = errors.iter().copied().fold(0.0_f64, f64::max);
        let mean_error = errors.iter().sum::<f64>() / ROWS as f64;
        eprintln!(
            "[2948 known truth {label}] max |Ŝ(exit) − S(exit)| {max_error:.4e} mean {mean_error:.4e}"
        );
        assert!(
            survival.iter().all(|s| s.is_finite() && (0.0..=1.0).contains(s)),
            "the flex arm's replayed survival must be a probability at every row"
        );
    }

    #[test]
    fn known_truth_flex_arm_fits_on_the_requested_global_empirical_law_2948() {
        arm("global-empirical", Some("global-empirical"));
    }

    #[test]
    fn known_truth_flex_arm_fits_on_the_standard_normal_closed_form_2948() {
        arm("standard-normal", Some("standard-normal"));
    }
}
