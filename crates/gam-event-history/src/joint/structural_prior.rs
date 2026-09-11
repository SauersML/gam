//! Normalized priors on physical dynamics and observation-shape functions.
//! Chart Jacobians and strength normalizers are part of the same density.
use super::*;

pub(super) enum StructuralFunction {
    /// T*r = 2*T*integral_0^infinity (d exp(-r*t)/dt)^2 dt.
    TemporalVariation {
        coordinate: usize,
        log_time: f64,
    },
    /// Inverse squared residual scale, with a Gamma(3, lambda) law.
    MeasurementPrecision {
        coordinate: usize,
    },
    /// V/scale^2-1 for Student-t (multiplier 2), or (V-mu)/mu^2
    /// for counts (multiplier 1). The denominator is softplus(coordinate).
    InverseSoftplus {
        coordinate: usize,
        log_multiplier: f64,
    },
    CountMean {
        coordinate: usize,
    },
}

pub(super) struct ScalarPriorEvaluation {
    pub coordinate: usize,
    pub log_density: f64,
    pub first: f64,
    pub second: f64,
    pub mixed: f64,
    pub strength_first: f64,
    pub strength_second: f64,
}

/// t = d log(softplus(q))/dq and d log(t)/dq. The latter is small
/// at the negative end, where subtracting complement - t loses curvature.
fn softplus_shape(q: f64) -> (f64, f64, f64, f64, f64) {
    let log_s = emission::log_softplus(&q);
    let log_sigmoid = -emission::softplus(&(-q));
    let sigmoid = log_sigmoid.exp();
    let complement = (-emission::softplus(&q)).exp();
    let log_t = log_sigmoid - log_s;
    let t = log_t.exp();
    let log_t_derivative = if q <= 0.0 {
        let x = q.exp();
        // delta = (x-log1p(x))/x, evaluated without the cancellation or
        // the x^2 underflow of the direct numerator. The alternating series
        // converges geometrically for x<=1/2; its next term bounds the error.
        let delta = if x <= 0.5 {
            let mut power = x;
            let mut sum = 0.5 * x;
            let mut order = 2.0;
            loop {
                power *= -x;
                let term = power / (order + 1.0);
                if term.abs() <= f64::EPSILON * sum.abs() {
                    break;
                }
                sum += term;
                order += 1.0;
            }
            sum
        } else {
            1.0 - x.ln_1p() / x
        };
        -complement * delta / (1.0 - delta)
    } else {
        complement - t
    };
    (log_s, log_t, t, sigmoid, log_t_derivative)
}

impl StructuralFunction {
    pub(super) fn evaluate(&self, theta: &[f64], rho: f64) -> ScalarPriorEvaluation {
        match *self {
            Self::CountMean { coordinate } => {
                let log_weight = rho + theta[coordinate];
                let weight = log_weight.exp();
                ScalarPriorEvaluation {
                    coordinate,
                    log_density: log_weight - weight,
                    first: 1.0 - weight,
                    second: -weight,
                    mixed: -weight,
                    strength_first: 1.0 - weight,
                    strength_second: -weight,
                }
            }
            Self::TemporalVariation {
                coordinate,
                log_time,
            } => {
                let q = theta[coordinate];
                let (log_s, log_t, _, sigmoid, _) = softplus_shape(q);
                let complement = (-emission::softplus(&q)).exp();
                // Add large opposite rho/q before the small time scale.
                let log_weight = if q < 0.0 {
                    (rho + q) + log_time + (log_s - q)
                } else {
                    rho + log_time + log_s
                };
                let weight = log_weight.exp();
                let weighted_slope = (log_weight + log_t).exp();
                ScalarPriorEvaluation {
                    coordinate,
                    log_density: log_weight + log_t - weight,
                    first: complement - weighted_slope,
                    second: -sigmoid * complement - weighted_slope * complement,
                    mixed: -weighted_slope,
                    strength_first: 1.0 - weight,
                    strength_second: -weight,
                }
            }
            Self::MeasurementPrecision { coordinate } => {
                // Gamma shape three is the smallest integer shape for which
                // sigma^4 has a finite prior mean. Gamma(3)=2 cancels the
                // Jacobian multiplier |d exp(-2q)/dq| / exp(-2q) = 2.
                let log_weight = 2.0 * (0.5 * rho - theta[coordinate]);
                let weight = log_weight.exp();
                ScalarPriorEvaluation {
                    coordinate,
                    log_density: 3.0 * log_weight - weight,
                    first: -6.0 + 2.0 * weight,
                    second: -4.0 * weight,
                    mixed: 2.0 * weight,
                    strength_first: 3.0 - weight,
                    strength_second: -weight,
                }
            }
            Self::InverseSoftplus {
                coordinate,
                log_multiplier,
            } => {
                // Shape five is the smallest integer Gamma shape giving
                // four finite inverse-functional moments. Thus large count
                // sizes and Student-t degrees of freedom have finite prior
                // fourth moments, rather than an infinite coordinate mean.
                let q = theta[coordinate];
                let (log_s, log_t, t, sigmoid, log_t_derivative) = softplus_shape(q);
                let complement = (-emission::softplus(&q)).exp();
                let log_weight = if q < 0.0 {
                    (rho - q) + log_multiplier - (log_s - q)
                } else {
                    rho + log_multiplier - log_s
                };
                let weight = log_weight.exp();
                let weighted_slope = (log_weight + log_t).exp();
                ScalarPriorEvaluation {
                    coordinate,
                    log_density: 5.0 * log_weight - 24.0_f64.ln() + log_t - weight,
                    first: complement - 6.0 * t + weighted_slope,
                    second: -sigmoid * complement - 6.0 * t * log_t_derivative
                        + weighted_slope * (complement - 2.0 * t),
                    mixed: weighted_slope,
                    strength_first: 5.0 - weight,
                    strength_second: -weight,
                }
            }
        }
    }
}

impl JointFunctionPriors<'_> {
    pub(super) fn add_structural_priors(
        &mut self,
        histories: &[&JointHistory],
    ) -> Result<(), EventHistoryError> {
        let log_spans: Vec<_> = histories
            .iter()
            .map(|h| (h.times[h.times.len() - 1] - h.times[0]).ln())
            .collect();
        if log_spans.iter().any(|v| !v.is_finite()) {
            return Err(invalid(
                "structural priors require finite positive follow-up spans",
            ));
        }
        let log_time = log_sum_exp(&log_spans) - (histories.len() as f64).ln();
        self.log_followup_scale = log_time;
        for h in histories {
            let total = h.exposure.iter().sum::<f64>();
            for n in 0..h.times.len() {
                for j in 0..self.baseline_mean_design.len() {
                    self.baseline_mean_design[j] +=
                        h.exposure[n] / total / histories.len() as f64 * h.baseline_design[[n, j]];
                }
            }
        }
        // Its unit intercept is an exact coordinate direction, irrespective
        // of accumulated floating-point error in normalized exposure weights.
        self.baseline_mean_design[0] = 1.0;
        self.penalties.push(FunctionPenalty::BaselineLevel);
        if self.model.spec.signatures > 0 {
            self.penalties.push(FunctionPenalty::TemporalVariation);
            self.structural.push(
                self.model
                    .layout
                    .rates
                    .clone()
                    .map(|coordinate| StructuralFunction::TemporalVariation {
                        coordinate,
                        log_time,
                    })
                    .collect(),
            );
        }
        for (channel, family) in self.model.spec.measurements.iter().enumerate() {
            let shape = &self.model.layout.measurement_shape[channel];
            match family {
                MeasurementFamily::StudentT => {
                    self.penalties
                        .push(FunctionPenalty::MeasurementPrecision { channel });
                    self.structural
                        .push(vec![StructuralFunction::MeasurementPrecision {
                            coordinate: shape.start,
                        }]);
                    self.penalties
                        .push(FunctionPenalty::TailVarianceInflation { channel });
                    self.structural
                        .push(vec![StructuralFunction::InverseSoftplus {
                            coordinate: shape.start + 1,
                            log_multiplier: 2.0_f64.ln(),
                        }]);
                }
                MeasurementFamily::NegativeBinomial => {
                    self.penalties.push(FunctionPenalty::CountMean { channel });
                    self.structural.push(vec![StructuralFunction::CountMean {
                        coordinate: self.model.layout.measurement_location[channel].start,
                    }]);
                    self.penalties
                        .push(FunctionPenalty::CountOverdispersion { channel });
                    self.structural
                        .push(vec![StructuralFunction::InverseSoftplus {
                            coordinate: shape.start,
                            log_multiplier: 0.0,
                        }]);
                }
                MeasurementFamily::BinaryProbit | MeasurementFamily::OrdinalProbit { .. } => {}
            }
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::scalar::Mixed;

    use super::super::test_support::structural_oracle as oracle;

    #[test]
    fn physical_shape_priors_include_exact_normalizers_and_chart_curvature() {
        let cases = [
            StructuralFunction::CountMean { coordinate: 0 },
            StructuralFunction::TemporalVariation {
                coordinate: 0,
                log_time: 3.0_f64.ln(),
            },
            StructuralFunction::MeasurementPrecision { coordinate: 0 },
            StructuralFunction::InverseSoftplus {
                coordinate: 0,
                log_multiplier: 2.0_f64.ln(),
            },
            StructuralFunction::InverseSoftplus {
                coordinate: 0,
                log_multiplier: 0.0,
            },
        ];
        for function in &cases {
            for q in [-8.0, -1.0, 0.0, 3.0, 20.0] {
                for rho in [-3.0, 0.5, 4.0] {
                    let value = function.evaluate(&[q], rho);
                    for i in 0..2 {
                        for j in 0..2 {
                            let jet = oracle(
                                function,
                                &[Mixed::seed(q, f64::from(i == 0), f64::from(j == 0))],
                                &Mixed::seed(rho, f64::from(i == 1), f64::from(j == 1)),
                            );
                            let g = if i == 0 {
                                value.first
                            } else {
                                value.strength_first
                            };
                            let h = if i != j {
                                value.mixed
                            } else if i == 0 {
                                value.second
                            } else {
                                value.strength_second
                            };
                            for (a, b) in [(value.log_density, jet.base), (g, jet.u), (h, jet.uv)] {
                                assert!(
                                    (a - b).abs() < 1e-10 * (1.0 + b.abs()),
                                    "q={q}, rho={rho}, {i},{j}: {a} vs {b}"
                                );
                            }
                        }
                    }
                }
            }
        }
    }

    #[test]
    fn physical_priors_normalize_and_have_the_declared_finite_moments() {
        let (nodes, weights) = gam_math::special::gauss_legendre(129);
        let cases = [
            (StructuralFunction::CountMean { coordinate: 0 }, 1.0),
            (
                StructuralFunction::TemporalVariation {
                    coordinate: 0,
                    log_time: 3.0_f64.ln(),
                },
                1.0,
            ),
            (
                StructuralFunction::MeasurementPrecision { coordinate: 0 },
                3.0,
            ),
            (
                StructuralFunction::InverseSoftplus {
                    coordinate: 0,
                    log_multiplier: 2.0_f64.ln(),
                },
                5.0,
            ),
            (
                StructuralFunction::InverseSoftplus {
                    coordinate: 0,
                    log_multiplier: 0.0,
                },
                5.0,
            ),
        ];
        let inverse_softplus = |value: f64| value + (-(-value).exp_m1()).ln();
        for (function, shape) in cases {
            for rho in [-2.0_f64, 0.0, 2.0] {
                let lambda = rho.exp();
                let mut mass = 0.0;
                let mut mean = 0.0;
                let mut score = 0.0;
                let mut inverse_moment = 0.0;
                for (&node, &weight) in nodes.iter().zip(&weights) {
                    // Integrate in u=lambda*f, independently of the log charts.
                    // The omitted Gamma(shape<=5) tail above 64 is below 1e-21.
                    let physical = 32.0 * (node + 1.0) / lambda;
                    let (q, log_jacobian) = match function {
                        StructuralFunction::CountMean { .. } => (physical.ln(), physical.ln()),
                        StructuralFunction::TemporalVariation { log_time, .. } => {
                            let q = inverse_softplus(physical / 3.0);
                            (q, log_time - emission::softplus(&(-q)))
                        }
                        StructuralFunction::MeasurementPrecision { .. } => {
                            (-0.5 * physical.ln(), (2.0 * physical).ln())
                        }
                        StructuralFunction::InverseSoftplus { log_multiplier, .. } => {
                            let s = log_multiplier.exp() / physical;
                            let q = inverse_softplus(s);
                            (q, log_multiplier - emission::softplus(&(-q)) - 2.0 * s.ln())
                        }
                    };
                    let evaluated = function.evaluate(&[q], rho);
                    let density =
                        (evaluated.log_density - log_jacobian).exp() * 32.0 * weight / lambda;
                    mass += density;
                    mean += density * physical;
                    score += density * evaluated.strength_first;
                    if shape == 3.0 {
                        inverse_moment += density / physical.powi(2);
                    }
                    if shape == 5.0 {
                        inverse_moment += density / physical.powi(4);
                    }
                }
                assert!(
                    (mass - 1.0).abs() < 1e-12,
                    "shape {shape}, rho {rho}: mass {mass}"
                );
                assert!((mean * lambda / shape - 1.0).abs() < 1e-12);
                assert!(score.abs() < 1e-12);
                if shape == 3.0 {
                    assert!((inverse_moment / (lambda * lambda / 2.0) - 1.0).abs() < 1e-12);
                }
                if shape == 5.0 {
                    assert!((inverse_moment / (lambda.powi(4) / 24.0) - 1.0).abs() < 1e-12);
                }
            }
        }
    }

    #[test]
    fn scalar_prior_curvature_survives_saturated_softplus_and_large_cancelling_scales() {
        let inverse = StructuralFunction::InverseSoftplus {
            coordinate: 0,
            log_multiplier: 2.0_f64.ln(),
        };
        let tail = inverse.evaluate(&[-40.0], -800.0);
        // With a negligible gamma killing term, log density has curvature
        // (shape-1)*exp(q)/2 at the negative chart tail.
        assert!((tail.second / (2.0 * (-40.0_f64).exp()) - 1.0).abs() < 1e-12);
        let rate = StructuralFunction::TemporalVariation {
            coordinate: 0,
            log_time: 3.0_f64.ln(),
        };
        for q in [-40.0, 40.0] {
            let value = rate.evaluate(&[q], -800.0);
            assert!((value.second / (-q.abs()).exp() + 1.0).abs() < 1e-12);
        }
        for large in [800.0, 1e200] {
            let value = rate.evaluate(&[-large], large);
            assert!((value.strength_first + 2.0).abs() < 1e-12);
            assert!((value.log_density - (3.0_f64.ln() - 3.0)).abs() < 1e-12);
            let value = inverse.evaluate(&[-large], -large);
            assert!((value.strength_first - 3.0).abs() < 1e-12);
            assert!((value.log_density - (5.0 * 2.0_f64.ln() - 24.0_f64.ln() - 2.0)).abs() < 1e-12);
            let noise = StructuralFunction::MeasurementPrecision { coordinate: 0 };
            let value = noise.evaluate(&[large], 2.0 * large);
            assert_eq!(value.log_density, -1.0);
            assert_eq!(value.first, -4.0);
        }
    }
}
