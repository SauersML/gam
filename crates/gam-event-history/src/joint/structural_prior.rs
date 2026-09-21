//! Normalized priors on physical dynamics and observation-shape functions.
//! Chart Jacobians and strength normalizers are part of the same density. Every
//! law is exponential in its physical quantity: the maximum-entropy law on
//! [0, inf) given its mean, whose one parameter is the learned strength.
use super::emission;
use super::function_prior::FunctionPenalty;
use super::law::{JointLikelihood, MeasurementFamily, invalid};
use crate::EventHistoryError;
use crate::scalar::{add_real, div, exp, ln};
use gam_math::nested_dual::JetField;

pub(super) enum StructuralFunction {
    /// T*r = 2*T*integral_0^infinity (d exp(-r*t)/dt)^2 dt, with T the frozen mean follow-up
    /// span, supplied at evaluation.
    TemporalVariation { coordinate: usize },
    /// The inverse squared residual scale sigma^-2 = exp(-2q).
    MeasurementPrecision { coordinate: usize },
    /// V/scale^2-1 for Student-t (multiplier 2), or (V-mu)/mu^2
    /// for counts (multiplier 1). The denominator is softplus(coordinate).
    InverseSoftplus { coordinate: usize, multiplier: f64 },
    CountMean { coordinate: usize },
}

/// Derivatives of one scalar law's log density in its coordinate `q` and
/// its log strength `rho`, through the third order LAML needs.
pub(super) struct ScalarPriorEvaluation<S> {
    pub coordinate: usize,
    pub log_density: S,
    /// d/dq, d^2/dq^2 and d^3/dq^3.
    pub first: S,
    pub second: S,
    pub third: S,
    /// d^2/dq drho and d^3/dq^2 drho.
    pub mixed: S,
    pub strength_mixed_second: S,
    pub strength_first: S,
    pub strength_second: S,
}

/// log(1 + x) with its exact derivative stack.
pub(super) fn ln_1p<S: JetField>(x: &S) -> S {
    let v = x.value();
    let r = 1.0 / (1.0 + v);
    x.compose_unary([v.ln_1p(), r, -r * r, 2.0 * r * r * r, -6.0 * r * r * r * r])
}

/// `[log softplus(q), log t, t, sigmoid(q), d log t/dq]` with
/// t = d log(softplus(q))/dq. The last is small at the negative end, where
/// subtracting complement - t loses curvature.
fn softplus_shape<S: JetField>(q: &S) -> [S; 5] {
    let log_s = emission::log_softplus(q);
    let log_sigmoid = emission::softplus(&q.neg()).neg();
    let sigmoid = exp(&log_sigmoid);
    let complement = exp(&emission::softplus(q).neg());
    let log_t = log_sigmoid.sub(&log_s);
    let t = exp(&log_t);
    let log_t_derivative = if q.value() <= 0.0 {
        let x = exp(q);
        // delta = (x-log1p(x))/x, evaluated without the cancellation or
        // the x^2 underflow of the direct numerator. The alternating series
        // converges geometrically for x<=1/2; its next term bounds the error.
        let delta = if x.value() <= 0.5 {
            let mut power = x.clone();
            let mut sum = x.scale(0.5);
            let mut order = 2.0;
            loop {
                power = power.mul(&x).neg();
                let term = div(&power, &power.constant_like(order + 1.0));
                if term.value().abs() <= f64::EPSILON * sum.value().abs() {
                    break;
                }
                sum = sum.add(&term);
                order += 1.0;
            }
            sum
        } else {
            add_real(&div(&ln_1p(&x), &x).neg(), 1.0)
        };
        div(
            &complement.mul(&delta).neg(),
            &add_real(&delta.neg(), 1.0),
        )
    } else {
        complement.sub(&t)
    };
    [log_s, log_t, t, sigmoid, log_t_derivative]
}

impl StructuralFunction {
    /// The law's derivatives at `theta` and log strength `rho`. `log_time` is the log of the
    /// frozen mean follow-up span, formed over S; only TemporalVariation reads it.
    pub(super) fn evaluate<S: JetField>(
        &self,
        theta: &[S],
        rho: &S,
        log_time: &S,
    ) -> ScalarPriorEvaluation<S> {
        match *self {
            Self::CountMean { coordinate } => {
                let log_weight = rho.add(&theta[coordinate]);
                let weight = exp(&log_weight);
                ScalarPriorEvaluation {
                    coordinate,
                    log_density: log_weight.sub(&weight),
                    first: add_real(&weight.neg(), 1.0),
                    second: weight.neg(),
                    third: weight.neg(),
                    mixed: weight.neg(),
                    strength_mixed_second: weight.neg(),
                    strength_first: add_real(&weight.neg(), 1.0),
                    strength_second: weight.neg(),
                }
            }
            Self::TemporalVariation { coordinate } => {
                let q = &theta[coordinate];
                let [log_s, log_t, _, sigmoid, _] = softplus_shape(q);
                let complement = exp(&emission::softplus(q).neg());
                // Add large opposite rho/q before the small time scale.
                let log_weight = if q.value() < 0.0 {
                    rho.add(q).add(log_time).add(&log_s.sub(q))
                } else {
                    rho.add(log_time).add(&log_s)
                };
                let weight = exp(&log_weight);
                let weighted_slope = exp(&log_weight.add(&log_t));
                // The density is rho + log T + log sigmoid(q) - lambda T softplus(q),
                // so every q-derivative past the first shares sigmoid(1-sigmoid).
                let second = sigmoid
                    .mul(&complement)
                    .neg()
                    .sub(&weighted_slope.mul(&complement));
                let third = second.mul(&complement.sub(&sigmoid));
                ScalarPriorEvaluation {
                    coordinate,
                    log_density: log_weight.add(&log_t).sub(&weight),
                    first: complement.sub(&weighted_slope),
                    second,
                    third,
                    mixed: weighted_slope.neg(),
                    strength_mixed_second: weighted_slope.mul(&complement).neg(),
                    strength_first: add_real(&weight.neg(), 1.0),
                    strength_second: weight.neg(),
                }
            }
            Self::MeasurementPrecision { coordinate } => {
                // On the precision exp(-2q): log p(q) = rho - 2q + log 2 - exp(rho - 2q), where
                // log 2 is the Jacobian |d exp(-2q)/dq| / exp(-2q).
                let log_weight = rho.scale(0.5).sub(&theta[coordinate]).scale(2.0);
                let weight = exp(&log_weight);
                ScalarPriorEvaluation {
                    coordinate,
                    log_density: log_weight.add(&ln(&rho.constant_like(2.0))).sub(&weight),
                    first: add_real(&weight.scale(2.0), -2.0),
                    second: weight.scale(-4.0),
                    third: weight.scale(8.0),
                    mixed: weight.scale(2.0),
                    strength_mixed_second: weight.scale(-4.0),
                    strength_first: add_real(&weight.neg(), 1.0),
                    strength_second: weight.neg(),
                }
            }
            Self::InverseSoftplus {
                coordinate,
                multiplier,
            } => {
                // On f = multiplier/softplus(q): log p(q) = log(lambda f) + log t - lambda f, with
                // t = d log softplus/dq and log t = log|df/dq| - log f.
                let q = &theta[coordinate];
                let [log_s, log_t, t, sigmoid, u] = softplus_shape(q);
                let complement = exp(&emission::softplus(q).neg());
                let log_multiplier = ln(&rho.constant_like(multiplier));
                let log_weight = if q.value() < 0.0 {
                    rho.sub(q).add(&log_multiplier).sub(&log_s.sub(q))
                } else {
                    rho.add(&log_multiplier).sub(&log_s)
                };
                let weight = exp(&log_weight);
                let weighted_slope = exp(&log_weight.add(&log_t));
                // With u = d log t/dq, t' = t u and u' = -sigmoid(1-sigmoid) - t u; the derivatives
                // of -t + u + w t follow.
                let curvature = sigmoid.mul(&complement);
                let gap = complement.sub(&t.scale(2.0));
                let square = u.mul(&u);
                let product = t.mul(&u);
                ScalarPriorEvaluation {
                    coordinate,
                    log_density: log_weight.add(&log_t).sub(&weight),
                    first: complement.sub(&t.scale(2.0)).add(&weighted_slope),
                    second: curvature
                        .neg()
                        .sub(&product.scale(2.0))
                        .add(&weighted_slope.mul(&gap)),
                    third: curvature
                        .mul(&complement.sub(&sigmoid))
                        .neg()
                        .sub(&t.mul(&square.sub(&product).sub(&curvature)).scale(2.0))
                        .add(&weighted_slope.mul(
                            &square
                                .sub(&product.scale(4.0))
                                .add(&t.mul(&t))
                                .sub(&curvature),
                        )),
                    mixed: weighted_slope.clone(),
                    strength_mixed_second: weighted_slope.mul(&gap),
                    strength_first: add_real(&weight.neg(), 1.0),
                    strength_second: weight.neg(),
                }
            }
        }
    }

    /// `d log Z / d tau` at `tau = 1/lambda = 0` for a shape law whose strength limit is a
    /// reduced observation family: TailVarianceInflation (Student-t to Gaussian) and
    /// CountOverdispersion (negative binomial to Poisson), the InverseSoftplus laws. The law puts
    /// Exp(lambda) on x, so E[x] = tau and E[x^2] = 2 tau^2, and the derivative is `sum_i s_i`,
    /// where `s_i` is the channel's per-record derivative in x of the log likelihood at x = 0,
    /// evaluated at the reduced fit and including its log-det motion. The verdict is
    /// `ZeroEffectDecision::from_score` on this score evaluated at `Running`.
    pub(super) fn boundary_score<S: JetField>(&self, record_scores: &[S]) -> Result<S, EventHistoryError> {
        if !matches!(self, Self::InverseSoftplus { .. }) {
            return Err(invalid(
                "only a tail or overdispersion law has a reduced observation family at its strength limit",
            ));
        }
        let Some(first) = record_scores.first() else {
            return Err(invalid("a boundary score needs the channel's record scores"));
        };
        if record_scores.iter().any(|s| !s.value().is_finite()) {
            return Err(invalid("record boundary scores must be finite"));
        }
        Ok(record_scores
            .iter()
            .fold(first.constant_like(0.0), |sum, s| sum.add(s)))
    }
}

/// Physical dynamics and observation-shape laws, in strength order. Each
/// entry shares one learned strength across its functions. TemporalVariation
/// reads the frozen mean follow-up span shared with the baseline-level prior at
/// evaluation. Probit thresholds have the normalized simplex measure of
/// `category_prior`, with no strength.
pub(super) fn structural_priors(model: &JointLikelihood) -> Vec<(FunctionPenalty, Vec<StructuralFunction>)> {
    let mut out = Vec::new();
    if model.spec.signatures > 0 {
        out.push((
            FunctionPenalty::TemporalVariation,
            model
                .layout
                .rates
                .clone()
                .map(|coordinate| StructuralFunction::TemporalVariation { coordinate })
                .collect(),
        ));
    }
    for (channel, family) in model.spec.measurements.iter().enumerate() {
        let shape = &model.layout.measurement_shape[channel];
        match family {
            MeasurementFamily::StudentT => {
                out.push((
                    FunctionPenalty::MeasurementPrecision { channel },
                    vec![StructuralFunction::MeasurementPrecision {
                        coordinate: shape.start,
                    }],
                ));
                out.push((
                    FunctionPenalty::TailVarianceInflation { channel },
                    vec![StructuralFunction::InverseSoftplus {
                        coordinate: shape.start + 1,
                        multiplier: 2.0,
                    }],
                ));
            }
            MeasurementFamily::NegativeBinomial => {
                // The count mean's level: column zero of the intercept function.
                out.push((
                    FunctionPenalty::CountMean { channel },
                    vec![StructuralFunction::CountMean {
                        coordinate: model.layout.measurement_location[channel].start,
                    }],
                ));
                out.push((
                    FunctionPenalty::CountOverdispersion { channel },
                    vec![StructuralFunction::InverseSoftplus {
                        coordinate: shape.start,
                        multiplier: 1.0,
                    }],
                ));
            }
            MeasurementFamily::Probit { .. } => {}
        }
    }
    out
}

#[cfg(test)]
mod tests {
    use super::super::function_prior_tests::{
        channel_agrees, compensated_sum, rule, structural_oracle as oracle,
    };
    use super::*;
    use crate::scalar::Rows;
    use crate::test_support::{Bound, agrees};

    fn cases() -> [StructuralFunction; 5] {
        [
            StructuralFunction::CountMean { coordinate: 0 },
            StructuralFunction::TemporalVariation { coordinate: 0 },
            StructuralFunction::MeasurementPrecision { coordinate: 0 },
            StructuralFunction::InverseSoftplus {
                coordinate: 0,
                multiplier: 2.0,
            },
            StructuralFunction::InverseSoftplus {
                coordinate: 0,
                multiplier: 1.0,
            },
        ]
    }

    fn constant(value: f64) -> Bound {
        Bound::exact(0.0).constant_like(value)
    }

    /// The follow-up span T = 3, formed over Bound.
    fn log_time() -> Bound {
        ln(&constant(3.0))
    }

    #[test]
    fn physical_shape_priors_include_exact_normalizers_and_chart_curvature() {
        // Measured bar: both routes pass softplus, log-softplus and ln_1p through
        // compose_unary, whose one-ulp charge has no cited accuracy.
        let log_time = log_time();
        for function in &cases() {
            for q in [-8.0, -1.0, 0.0, 3.0, 20.0] {
                for rho in [-3.0, 0.5, 4.0] {
                    let value = function.evaluate(&[Bound::exact(q)], &Bound::exact(rho), &log_time);
                    for i in 0..2 {
                        for j in 0..2 {
                            let jet = oracle(
                                function,
                                &[Rows::seed(
                                    Rows::seed(Bound::exact(q), [f64::from(i == 0)]),
                                    [f64::from(j == 0)],
                                )],
                                &Rows::seed(
                                    Rows::seed(Bound::exact(rho), [f64::from(i == 1)]),
                                    [f64::from(j == 1)],
                                ),
                                &Rows::seed(Rows::seed(log_time, [0.0]), [0.0]),
                            );
                            let g = if i == 0 {
                                &value.first
                            } else {
                                &value.strength_first
                            };
                            let h = if i != j {
                                &value.mixed
                            } else if i == 0 {
                                &value.second
                            } else {
                                &value.strength_second
                            };
                            let name = format!("q={q}, rho={rho}, {i},{j}");
                            agrees(&value.log_density, &jet.base.base, &name);
                            channel_agrees(g, &jet.base.rows[0], &name);
                            channel_agrees(h, &jet.rows[0].rows[0], &name);
                        }
                    }
                    // Third order: an inner directional jet under two outer unit directions.
                    for (strength, expected) in
                        [(false, &value.third), (true, &value.strength_mixed_second)]
                    {
                        let jet = oracle(
                            function,
                            &[Rows::seed(
                                Rows::seed(Rows::seed(Bound::exact(q), [f64::from(!strength)]), [1.0]),
                                [1.0],
                            )],
                            &Rows::seed(
                                Rows::seed(Rows::seed(Bound::exact(rho), [f64::from(strength)]), [0.0]),
                                [0.0],
                            ),
                            &Rows::seed(Rows::seed(Rows::seed(log_time, [0.0]), [0.0]), [0.0]),
                        );
                        let channel = &jet.rows[0].rows[0].rows[0];
                        let name = format!("q={q}, rho={rho}, strength {strength}");
                        if !strength
                            && q == 0.0
                            && matches!(function, StructuralFunction::TemporalVariation { .. })
                        {
                            // d^3 log p/dq^3 = -sigmoid(1-sigmoid)(1-2 sigmoid)(1+lambda T)
                            // vanishes exactly at q = 0. Production forms the sigmoid and its
                            // complement by the same computation there, so its zero is exact.
                            assert_eq!(expected.value, 0.0, "{name}: the symmetric zero");
                        } else {
                            agrees(expected, channel, &name);
                        }
                    }
                }
            }
        }
    }

    #[test]
    fn physical_priors_normalize_with_their_declared_mean_and_zero_score() {
        // Measured bar: the laws and their charts pass softplus, log-softplus, expm1 and ln_1p
        // through compose_unary, whose one-ulp charge has no cited accuracy.
        let log_time = log_time();
        // Integrate z = lambda f ~ Exp(1) over [0, range] with the certified Gauss-Legendre rule.
        let range = 64.0_f64;
        for function in &cases() {
            for rho in [-2.0_f64, 0.0, 2.0] {
                let lambda = exp(&Bound::exact(rho));
                let integrate = |nodes_count: usize| {
                    let r = rule(nodes_count, 0.0, range);
                    let mut values: [Vec<f64>; 3] = Default::default();
                    let mut errors = [0.0; 3];
                    for (point, weight) in r.points.iter().zip(&r.weights) {
                        let physical = div(point, &lambda);
                        // Chart points and Jacobians, in the chart's own arithmetic.
                        let inverse_softplus =
                            |v: &Bound| v.add(&ln(&emission::expm1(&v.neg()).neg()));
                        let (q, log_jacobian) = match function {
                            StructuralFunction::CountMean { .. } => {
                                (ln(&physical), ln(&physical))
                            }
                            StructuralFunction::TemporalVariation { .. } => {
                                let q = inverse_softplus(&div(&physical, &constant(3.0)));
                                let jacobian = emission::softplus(&q.neg()).neg().add(&log_time);
                                (q, jacobian)
                            }
                            StructuralFunction::MeasurementPrecision { .. } => {
                                (ln(&physical).scale(-0.5), ln(&physical.scale(2.0)))
                            }
                            StructuralFunction::InverseSoftplus { multiplier, .. } => {
                                let s = div(&constant(*multiplier), &physical);
                                let q = inverse_softplus(&s);
                                let jacobian = emission::softplus(&q.neg())
                                    .neg()
                                    .sub(&ln(&s).scale(2.0))
                                    .add(&ln(&constant(*multiplier)));
                                (q, jacobian)
                            }
                        };
                        let evaluated =
                            function.evaluate(&[Bound::exact(q.value)], &Bound::exact(rho), &log_time);
                        let density = exp(&evaluated.log_density.sub(&log_jacobian))
                            .mul(&div(weight, &lambda));
                        // The evaluation point's log z moves by the chart point's rounding
                        // through |d log z/dq| <= 2, and by the node displacement, relative to z.
                        let log_shift = 2.0 * q.rounding() + r.point_error / point.value;
                        // A node's relative error: its own account, its weight's certified error,
                        // and log_shift through every row's |d log(row)/d log z|, below
                        // 1 + range from the density and 1 from z.
                        let relative = density.rounding() / density.value
                            + r.weight_relative_error
                            + (range + 2.0) * log_shift;
                        let rows = [
                            density,
                            density.mul(&physical),
                            density.mul(&evaluated.strength_first),
                        ];
                        for (k, row) in rows.iter().enumerate() {
                            values[k].push(row.value);
                            errors[k] += row.rounding() + row.value.abs() * relative;
                        }
                        // The score 1 - z moves by at most z <= range per unit log z.
                        errors[2] += density.value * range * log_shift;
                    }
                    let mut out = [(0.0, 0.0); 3];
                    for k in 0..3 {
                        let (total, summation) = compensated_sum(&values[k]);
                        out[k] = (total, errors[k] + summation);
                    }
                    out
                };
                let coarse = integrate(129);
                let fine = integrate(257);
                // Exp(1) beyond range: mass e^-range, first moment (1 + range) e^-range, and the
                // score 1 - z below (2 + range) e^-range.
                let tail = (-range).exp();
                let exact = [
                    constant(1.0),
                    div(&constant(1.0), &lambda),
                    constant(0.0),
                ];
                let tails = [tail, (1.0 + range) * tail / lambda.value, (2.0 + range) * tail];
                for k in 0..3 {
                    let bar = (fine[k].0 - coarse[k].0).abs() + tails[k] + fine[k].1 + exact[k].rounding();
                    assert!(
                        (fine[k].0 - exact[k].value).abs() <= bar,
                        "rho {rho}, row {k}: {} vs {} within {bar}",
                        fine[k].0,
                        exact[k].value
                    );
                }
            }
        }
    }

    #[test]
    fn a_shape_law_boundary_score_is_the_first_moment_derivative_of_its_exponential_face() {
        // For a log likelihood linear in x, s x, log Z(tau) = log E exp(s x) under x ~ Exp with
        // mean tau, which is -log(1 - s tau) for s tau < 1; its derivative at tau = 0, taken by a
        // jet, is s.
        let tail = StructuralFunction::InverseSoftplus {
            coordinate: 0,
            multiplier: 2.0,
        };
        let scores = [0.75, -1.5, 0.125];
        let production = tail
            .boundary_score(&scores.map(Bound::exact))
            .unwrap();
        let tau = Rows::seed(Bound::exact(0.0), [1.0]);
        let total = scores
            .iter()
            .fold(tau.constant_like(0.0), |sum, &s| sum.add(&tau.constant_like(s)));
        let jet = ln(&add_real(&total.mul(&tau).neg(), 1.0)).neg();
        agrees(&production, &jet.rows[0], "tail boundary derivative");
        assert!(
            StructuralFunction::MeasurementPrecision { coordinate: 0 }
                .boundary_score(&[Bound::exact(1.0)])
                .is_err()
        );
        assert!(tail.boundary_score::<Bound>(&[]).is_err());
        assert!(tail.boundary_score(&[Bound::exact(f64::NAN)]).is_err());
    }

    #[test]
    fn scalar_prior_curvature_survives_saturated_softplus_and_large_cancelling_scales() {
        // Measured bar: the laws pass softplus and log-softplus through compose_unary, whose
        // one-ulp charge has no cited accuracy.
        let log_time = log_time();
        let rate = StructuralFunction::TemporalVariation { coordinate: 0 };
        for q in [-40.0_f64, 40.0] {
            let value = rate.evaluate(&[Bound::exact(q)], &Bound::exact(-800.0), &log_time);
            // -sigmoid(1-sigmoid)(1 + lambda T) with lambda T = 3 exp(-800), which underflows:
            // exp(-|q|)/(1+exp(-|q|))^2 against exp(-|q|) has relative remainder below
            // 2 exp(-|q|).
            let expected = exp(&Bound::exact(-q.abs())).neg();
            assert!(
                (value.second.value - expected.value).abs()
                    <= expected.value.abs() * 2.0 * (-q.abs()).exp()
                        + value.second.rounding()
                        + expected.rounding()
            );
        }
        // q = -large, rho = large: rho + q = 0 exactly and log softplus(q) - q = 0 exactly, so
        // the weight is exp(log 3); log sigmoid(q) = -large = log softplus(q), so log t = 0.
        // For the inverse law rho - q = 0 exactly, so its weight is exp(log 2). At large = 800
        // the running bounds resolve these values; every larger scale repeats the same exact
        // cancellations bitwise.
        let inverse = StructuralFunction::InverseSoftplus {
            coordinate: 0,
            multiplier: 2.0,
        };
        let value = rate.evaluate(&[Bound::exact(-800.0)], &Bound::exact(800.0), &log_time);
        let weight = exp(&log_time);
        agrees(&value.strength_first, &add_real(&weight.neg(), 1.0), "rate strength score");
        agrees(&value.log_density, &log_time.sub(&weight), "rate log density");
        let value = inverse.evaluate(&[Bound::exact(-800.0)], &Bound::exact(-800.0), &log_time);
        let two = ln(&constant(2.0));
        agrees(&value.strength_first, &add_real(&exp(&two).neg(), 1.0), "inverse strength score");
        agrees(&value.log_density, &two.sub(&exp(&two)), "inverse log density");
        let three = 3.0_f64.ln();
        let rate_reference = rate.evaluate(&[-800.0], &800.0, &three);
        let inverse_reference = inverse.evaluate(&[-800.0], &-800.0, &three);
        for large in [800.0_f64, 1e200] {
            let value = rate.evaluate(&[-large], &large, &three);
            assert_eq!(value.strength_first, rate_reference.strength_first);
            assert_eq!(value.log_density, rate_reference.log_density);
            let value = inverse.evaluate(&[-large], &-large, &three);
            assert_eq!(value.strength_first, inverse_reference.strength_first);
            assert_eq!(value.log_density, inverse_reference.log_density);
            // 0.5 rho - q = 0 exactly, so the precision law is exactly (0 + log 2) - exp(0) and its
            // first derivative exactly 2 exp(0) - 2.
            let noise = StructuralFunction::MeasurementPrecision { coordinate: 0 };
            let value = noise.evaluate(&[large], &(2.0 * large), &three);
            assert_eq!(value.log_density, 2.0_f64.ln() - 1.0);
            assert_eq!(value.first, 0.0);
        }
    }
}
