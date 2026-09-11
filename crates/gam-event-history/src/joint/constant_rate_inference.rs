//! Exact coefficient integration for constant counting-process rates.
//! The prior is the existing final-function law T*rate ~ Exp(lambda), with
//! the SAME shared lambda and follow-up scale as JointFunctionPriors.
use super::*;
use ndarray::Array1;

struct GammaRate {
    shape: f64,
    log_rate: f64,
}

/// Independent Gamma rate posteriors at the learned shared baseline-level
/// strength, or their exact point mass at zero. This is conditional on the
/// empirical-Bayes strength; it does not integrate strength uncertainty.
pub struct ConstantRatePosterior {
    gamma: Option<Vec<GammaRate>>,
    rate_mean: Vec<f64>,
    rate_variance: Vec<f64>,
    log_rate_mean: Option<Vec<f64>>,
    log_rate_variance: Option<Vec<f64>>,
    log_strengths: Option<[f64; 1]>,
    log_evidence: f64,
    gradient: f64,
    iterations: usize,
}

impl ConstantRatePosterior {
    pub fn is_zero_rate(&self) -> bool {
        self.gamma.is_none()
    }
    pub fn rate_mean(&self) -> &[f64] {
        &self.rate_mean
    }
    pub fn rate_variance(&self) -> &[f64] {
        &self.rate_variance
    }
    pub fn log_rate_mean(&self) -> Option<&[f64]> {
        self.log_rate_mean.as_deref()
    }
    pub fn log_rate_variance(&self) -> Option<&[f64]> {
        self.log_rate_variance.as_deref()
    }
    pub fn log_strengths(&self) -> Option<&[f64]> {
        self.log_strengths.as_ref().map(|v| v.as_slice())
    }
    pub fn log_evidence(&self) -> f64 {
        self.log_evidence
    }
    pub fn gradient(&self) -> f64 {
        self.gradient
    }
    pub fn iterations(&self) -> usize {
        self.iterations
    }

    /// Posterior E[exp(-sum_d exposure_d * rate_d)]. This averages rates
    /// under their joint posterior, rather than inserting mean coefficients.
    /// Exposures must be finite and nonnegative, one per declared mark.
    pub fn no_event_probability(&self, exposure: &[f64]) -> Result<f64, EventHistoryError> {
        if exposure.len() != self.rate_mean.len()
            || exposure.iter().any(|&x| !x.is_finite() || x < 0.0)
        {
            return Err(invalid(
                "constant-rate prediction requires a finite nonnegative exposure per mark",
            ));
        }
        let Some(gamma) = &self.gamma else {
            return Ok(1.0);
        };
        let log_survival = sum(gamma.iter().zip(exposure).map(|(g, &e)| {
            if e == 0.0 {
                0.0
            } else {
                -g.shape * emission::softplus(&(e.ln() - g.log_rate))
            }
        }));
        if log_survival.is_nan() || log_survival > 0.0 {
            return Err(numerical(
                "constant-rate posterior survival is unrepresentable",
            ));
        }
        Ok(log_survival.exp())
    }
}

struct RateStatistic {
    count: f64,
    log_exposure: f64,
    /// log Gamma(count+1), digamma(count+1), trigamma(count+1).
    gamma: [f64; 3],
}

fn log_add(a: f64, b: f64) -> f64 {
    if a == f64::NEG_INFINITY {
        return b;
    }
    if b == f64::NEG_INFINITY {
        return a;
    }
    a.max(b) + (-(a - b).abs()).exp().ln_1p()
}

/// Log evidence and its first two analytic log-c derivatives, c=lambda*T.
/// Unexposed zero-count marks integrate to one and contribute no derivative.
fn evidence(statistics: &[RateStatistic], log_c: f64) -> (f64, f64, f64) {
    let mut terms = Vec::with_capacity(statistics.len());
    let mut scores = Vec::with_capacity(statistics.len());
    let mut curvature = Vec::with_capacity(statistics.len());
    for s in statistics {
        if s.log_exposure == f64::NEG_INFINITY {
            continue;
        }
        let (p, q) = if log_c >= s.log_exposure {
            let e = (s.log_exposure - log_c).exp();
            (1.0 / (1.0 + e), e / (1.0 + e))
        } else {
            let e = (log_c - s.log_exposure).exp();
            (e / (1.0 + e), 1.0 / (1.0 + e))
        };
        let log_rate = log_add(s.log_exposure, log_c);
        terms.push(s.gamma[0] - s.count * log_rate - emission::softplus(&(s.log_exposure - log_c)));
        scores.push(q - s.count * p);
        curvature.push(-(s.count + 1.0) * p * q);
    }
    (
        sum(terms.into_iter()),
        sum(scores.into_iter()),
        sum(curvature.into_iter()),
    )
}

impl JointCohortIntegration<'_, '_> {
    pub(super) fn infer_constant_rates(
        &self,
        priors: &JointFunctionPriors<'_>,
        options: &CoefficientInferenceOptions,
    ) -> Result<ConstantRatePosterior, EventHistoryError> {
        let d = self.model.spec.marks.len();
        if d.checked_mul(256)
            .is_none_or(|n| n > options.pilot.memory_limit_bytes)
        {
            return Err(invalid(
                "constant-rate inference exceeds its statistics/posterior memory budget",
            ));
        }
        let mut log_exposure = vec![f64::NEG_INFINITY; d];
        let mut counts = vec![0usize; d];
        let mut genetic_density = 0.0;
        let mut genetic_correction = 0.0;
        for subject in self.subjects {
            let h = subject.history;
            let mut risk = h.initially_at_risk.clone();
            let mut exposure = vec![0.0; d];
            let mut correction = vec![0.0; d];
            for n in 0..h.times.len() {
                for mark in 0..d {
                    if risk[mark] {
                        let contribution = h.exposure[n] - correction[mark];
                        let next = exposure[mark] + contribution;
                        correction[mark] = (next - exposure[mark]) - contribution;
                        exposure[mark] = next;
                    }
                }
                if let Some(mark) = h.events[n] {
                    counts[mark] = counts[mark]
                        .checked_add(1)
                        .ok_or_else(|| numerical("constant-rate event count overflow"))?;
                    if self.model.spec.marks[mark] == MarkKind::Once {
                        risk[mark] = false;
                    }
                }
            }
            for mark in 0..d {
                log_exposure[mark] = log_add(log_exposure[mark], exposure[mark].ln());
            }
            let observed = subject
                .analytic_observed_genetic_log_density()
                .ok_or_else(|| {
                    invalid("constant-rate inference requires analytic genetic integration")
                })?;
            let contribution = observed - genetic_correction;
            let next = genetic_density + contribution;
            genetic_correction = (next - genetic_density) - contribution;
            genetic_density = next;
        }
        if !genetic_density.is_finite()
            || log_exposure
                .iter()
                .any(|v| v.is_nan() || *v == f64::INFINITY)
        {
            return Err(numerical(
                "constant-rate sufficient statistics are unrepresentable",
            ));
        }
        if log_exposure.iter().all(|&x| x == f64::NEG_INFINITY) {
            return Err(numerical(
                "baseline-level evidence is unidentified without risk exposure",
            ));
        }
        if counts
            .iter()
            .zip(&log_exposure)
            .any(|(&n, &e)| n > 0 && e == f64::NEG_INFINITY)
        {
            return Err(invalid(
                "an observed event with zero risk exposure has no finite baseline-level evidence optimum",
            ));
        }
        if counts.iter().all(|&n| n == 0) {
            // Every exposed term c/(E+c) is strictly increasing in c.
            // Its global supremum is one at c=infinity; this is a proper
            // point mass on zero rates, not a large finite intercept penalty.
            return Ok(ConstantRatePosterior {
                gamma: None,
                rate_mean: vec![0.0; d],
                rate_variance: vec![0.0; d],
                log_rate_mean: None,
                log_rate_variance: None,
                log_strengths: None,
                log_evidence: genetic_density,
                gradient: 0.0,
                iterations: 0,
            });
        }
        let statistics: Vec<_> = counts
            .iter()
            .zip(&log_exposure)
            .map(|(&count, &e)| {
                let count = count as f64;
                RateStatistic {
                    count,
                    log_exposure: e,
                    gamma: gam_math::jet_tower::ln_gamma_derivative_stack_order2(count + 1.0),
                }
            })
            .collect();
        let total_count = sum(statistics.iter().map(|s| s.count));
        let exposed: Vec<_> = log_exposure
            .iter()
            .copied()
            .filter(|x| x.is_finite())
            .collect();
        // c = sum exposure / sum events is a data-derived starting scale.
        // Equal exposed durations make this the exact stationary solution.
        let initial = log_sum_exp(&exposed) - total_count.ln();
        let equal_exposures = exposed.iter().all(|&e| e == exposed[0]);
        let (log_c, iterations) = if equal_exposures {
            (initial, 0)
        } else {
            let objective = opt::FusedObjective::new(|x: &Array1<f64>| {
                let (value, score, _) = evidence(&statistics, x[0]);
                if !value.is_finite() || !score.is_finite() {
                    return Err(opt::ObjectiveEvalError::recoverable_from(numerical(
                        "non-finite constant-rate evidence",
                    )));
                }
                Ok(opt::FirstOrderSample {
                    value: -value,
                    gradient: Array1::from_vec(vec![-score]),
                })
            });
            let solution = opt::Bfgs::new(Array1::from_vec(vec![initial]), objective)
                .without_relative_stall()
                .with_tolerance(
                    opt::Tolerance::new(options.strengths.stationarity_tolerance)
                        .map_err(|e| invalid(e.to_string()))?,
                )
                .with_gradient_tolerance(opt::GradientTolerance::absolute(
                    options.strengths.stationarity_tolerance,
                ))
                .with_max_iterations(
                    opt::MaxIterations::new(options.strengths.maximum_iterations)
                        .map_err(|e| invalid(e.to_string()))?,
                )
                .run()
                .map_err(|e| {
                    numerical(format!("constant-rate strength optimization failed: {e}"))
                })?;
            (solution.final_point[0], solution.iterations)
        };
        let (value, gradient, curvature) = evidence(&statistics, log_c);
        if !value.is_finite()
            || !gradient.is_finite()
            || !curvature.is_finite()
            || gradient.abs() > options.strengths.stationarity_tolerance
            || curvature >= 0.0
        {
            return Err(numerical(
                "constant-rate evidence did not reach its identified stationary maximum",
            ));
        }
        let mut gamma = Vec::with_capacity(d);
        let mut rate_mean = Vec::with_capacity(d);
        let mut rate_variance = Vec::with_capacity(d);
        let mut log_rate_mean = Vec::with_capacity(d);
        let mut log_rate_variance = Vec::with_capacity(d);
        for s in statistics {
            let log_rate = log_add(s.log_exposure, log_c);
            let shape = s.count + 1.0;
            rate_mean.push((shape.ln() - log_rate).exp());
            rate_variance.push((shape.ln() - 2.0 * log_rate).exp());
            log_rate_mean.push(s.gamma[1] - log_rate);
            log_rate_variance.push(s.gamma[2]);
            gamma.push(GammaRate { shape, log_rate });
        }
        if rate_mean
            .iter()
            .chain(&rate_variance)
            .chain(&log_rate_mean)
            .chain(&log_rate_variance)
            .any(|v| !v.is_finite())
        {
            return Err(numerical(
                "constant-rate posterior moments are unrepresentable",
            ));
        }
        let log_evidence = genetic_density + value;
        if !log_evidence.is_finite() {
            return Err(numerical(
                "constant-rate integrated evidence is unrepresentable",
            ));
        }
        Ok(ConstantRatePosterior {
            gamma: Some(gamma),
            rate_mean,
            rate_variance,
            log_rate_mean: Some(log_rate_mean),
            log_rate_variance: Some(log_rate_variance),
            log_strengths: Some([log_c - priors.log_followup_scale()]),
            log_evidence,
            gradient,
            iterations,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use rand::{SeedableRng, rngs::SmallRng};

    #[test]
    fn constant_rate_evidence_has_analytic_total_curvature_and_stable_limits() {
        let stats = [
            RateStatistic {
                count: 1.0,
                log_exposure: 0.0,
                gamma: gam_math::jet_tower::ln_gamma_derivative_stack_order2(2.0),
            },
            RateStatistic {
                count: 0.0,
                log_exposure: 3.0_f64.ln(),
                gamma: gam_math::jet_tower::ln_gamma_derivative_stack_order2(1.0),
            },
        ];
        for x in [-4.0, -1.0, 0.0, 2.0, 4.0] {
            let (value, first, second) = evidence(&stats, x);
            let delta = 1e-4;
            let left = evidence(&stats, x - delta);
            let right = evidence(&stats, x + delta);
            assert!((first - (right.0 - left.0) / (2.0 * delta)).abs() < 1e-8);
            assert!((second - (right.1 - left.1) / (2.0 * delta)).abs() < 1e-8);
            let c = x.exp();
            let direct = (c * c / ((1.0 + c).powi(2) * (3.0 + c))).ln();
            assert!((value - direct).abs() < 1e-12);
            assert!(second < 0.0);
        }
        assert_eq!(evidence(&stats, -1000.0).1, 2.0);
        assert_eq!(evidence(&stats, 1000.0).1, -1.0);
        assert!(evidence(&stats, -1000.0).0.is_finite());
        assert!(evidence(&stats, 1000.0).0.is_finite());
        assert!(evidence(&stats, 3.0_f64.ln()).1.abs() < 1e-15);
    }

    #[test]
    fn automatic_constant_rates_cover_risk_sets_genetics_time_units_and_the_null_boundary() {
        // First mark: once event at time one, exposure one. Second mark:
        // recurrent, no events, exposure three. Third mark: prevalent once,
        // no risk exposure. The shared c optimum solves c^2-c-6=0, so c=3.
        for (case, scale) in [
            (0, 1.0),
            (0, 1e100),
            (0, 1e-100),
            (1, 1.0),
            (2, 1.0),
            (3, 1.0),
        ] {
            let model = JointLikelihood::new(JointSpecification {
                signatures: 0,
                marks: if case == 2 {
                    vec![MarkKind::Once; 3]
                } else {
                    vec![MarkKind::Once, MarkKind::Recurrent, MarkKind::Once]
                },
                baseline_columns: 1,
                drive_columns: 1,
                entry_columns: 0,
                measurements: vec![],
                genetic_mean: vec![0.0; 2],
                genetic_precision: ndarray::array![
                    [4.0 / 3.0, -2.0 / 3.0],
                    [-2.0 / 3.0, 4.0 / 3.0]
                ],
            })
            .unwrap();
            let history = JointHistory {
                times: [0.0, 0.5, 1.0, 2.0, 3.0]
                    .iter()
                    .map(|v| v * scale)
                    .collect(),
                exposure: if case == 3 {
                    vec![0.0, 0.0, 0.0, 3.0 * scale, 0.0]
                } else {
                    vec![0.0, scale, 0.0, 2.0 * scale, 0.0]
                },
                events: if case == 0 || case == 3 {
                    vec![None, None, Some(0), None, None]
                } else {
                    vec![None; 5]
                },
                initially_at_risk: if case == 2 {
                    vec![false; 3]
                } else {
                    vec![true, true, false]
                },
                baseline_design: Array2::ones((5, 1)),
                drive_design: Array2::ones((4, 1)),
                entry_design: vec![],
                genetics: vec![Some(1.2), None],
                measurements: vec![],
            };
            let profile = JointReferenceProfile {
                times: vec![0.0, 3.0 * scale],
                baseline_design: Array2::ones((2, 1)),
                drive_design: Array2::ones((1, 1)),
                entry_design: vec![],
                genetics: vec![None; 2],
            };
            let mut rng = SmallRng::seed_from_u64(91271);
            let theta = vec![0.0; 3];
            let references = [model
                .resolve_reference(
                    &theta,
                    &profile,
                    &ReferenceResolutionOptions {
                        replicates: 4,
                        initial_particles: 4,
                        maximum_particles: 16,
                        minimum_risk_effective_samples: 2.0,
                        ..ReferenceResolutionOptions::default()
                    },
                    &mut rng,
                )
                .unwrap()
                .0];
            let subjects = [model
                .integration(
                    &theta,
                    &history,
                    &[0.0; 15],
                    None,
                    &IntegrationOptions::default(),
                    &mut rng,
                )
                .unwrap()];
            let cohort = model
                .cohort_integration(&subjects, &references, &[0])
                .unwrap();
            let priors = model.function_priors(&[&history], 1 << 20).unwrap();
            let accuracy = IntegrationAccuracy::default();
            let tolerance = CohortScoreTolerance {
                log_error: 1e-10,
                coefficient_score_error: vec![1e-10; 3],
                standard_error_multiplier: 3.0,
            };
            let options = CoefficientInferenceOptions::default();
            let result = cohort.infer_coefficients(
                &priors,
                &theta,
                &[0.0],
                &accuracy,
                &tolerance,
                &options,
                &mut rng,
            );
            if case == 2 || case == 3 {
                let error = result.err().unwrap().to_string();
                assert!(
                    error.contains(if case == 2 {
                        "unidentified"
                    } else {
                        "zero risk exposure"
                    }),
                    "{error}"
                );
                continue;
            }
            let result = result.unwrap();
            let posterior = result.constant_rates().unwrap();
            let observed_genetics = -0.5 * (2.0 * std::f64::consts::PI).ln() - 0.72;
            assert!(result.draws().is_none());
            assert!(result.sampled_evidence().is_none());
            assert!(result.rounds().is_empty());
            if case == 1 {
                assert!(posterior.is_zero_rate());
                assert_eq!(posterior.rate_mean(), &[0.0; 3]);
                assert_eq!(posterior.rate_variance(), &[0.0; 3]);
                assert!(result.coefficient_mean().is_none());
                assert!(result.coefficient_variance().is_none());
                assert!(result.log_strengths().is_none());
                assert!((result.log_evidence() - observed_genetics).abs() < 1e-12);
                assert_eq!(posterior.no_event_probability(&[1e300; 3]).unwrap(), 1.0);
                assert_eq!(posterior.iterations(), 0);
            } else {
                assert!(!posterior.is_zero_rate());
                assert!(posterior.iterations() > 0);
                assert!(posterior.gradient().abs() <= options.strengths.stationarity_tolerance);
                assert!(result.log_strengths().unwrap()[0].abs() < 2e-6);
                for (&mean, expected) in
                    posterior
                        .rate_mean()
                        .iter()
                        .zip([0.5, 1.0 / 6.0, 1.0 / 3.0])
                {
                    assert!((mean * scale - expected).abs() < 1e-6);
                }
                assert!(
                    (result.log_evidence()
                        - (observed_genetics + (3.0_f64 / 32.0).ln() - scale.ln()))
                    .abs()
                        < 1e-10
                );
                let survival = posterior
                    .no_event_probability(&[2.0 * scale, scale, 0.0])
                    .unwrap();
                assert!((survival - 8.0 / 21.0).abs() < 1e-6);
                assert!(
                    result
                        .coefficient_variance()
                        .unwrap()
                        .iter()
                        .all(|v| *v > 0.0)
                );
                assert_eq!(posterior.no_event_probability(&[0.0; 3]).unwrap(), 1.0);
            }
            assert!(posterior.no_event_probability(&[1.0]).is_err());
            assert!(posterior.no_event_probability(&[0.0, -1.0, 0.0]).is_err());
            assert!(
                posterior
                    .no_event_probability(&[0.0, f64::NAN, 0.0])
                    .is_err()
            );
        }
    }
}
