use super::{MeasurementFamily, invalid, numerical};
use crate::EventHistoryError;
use crate::scalar::{add_real, div, exp, ln};
use gam_math::nested_dual::JetField;

pub(super) fn softplus<S: JetField>(x: &S) -> S {
    let u = x.value();
    let value = if u > 0.0 {
        u + (-u).exp().ln_1p()
    } else {
        u.exp().ln_1p()
    };
    let s = if u >= 0.0 {
        1.0 / (1.0 + (-u).exp())
    } else {
        let e = u.exp();
        e / (1.0 + e)
    };
    let d = s * (1.0 - s);
    x.compose_unary([value, s, d, d * (1.0 - 2.0 * s), d * (1.0 - 6.0 * d)])
}

pub(super) fn expm1<S: JetField>(x: &S) -> S {
    let e = x.value().exp();
    x.compose_unary([x.value().exp_m1(), e, e, e, e])
}

fn log1p<S: JetField>(x: &S) -> S {
    let inv = 1.0 / (1.0 + x.value());
    x.compose_unary([
        x.value().ln_1p(),
        inv,
        -inv * inv,
        2.0 * inv.powi(3),
        -6.0 * inv.powi(4),
    ])
}

pub(super) fn log_softplus<S: JetField>(x: &S) -> S {
    if x.value() < -18.0 {
        // log(log1p(z)/z), z=exp(x). The omitted term is O(z^5),
        // below 2e-40 at this boundary. No overflow/underflow cancellation.
        let z = exp(x);
        let series = add_real(&z.scale(0.2), -0.25);
        let series = add_real(&z.mul(&series), 1.0 / 3.0);
        let series = add_real(&z.mul(&series), -0.5);
        x.add(&log1p(&z.mul(&series)))
    } else {
        ln(&softplus(x))
    }
}

fn logcdf<S: JetField>(x: &S) -> S {
    x.compose_unary(gam_math::probability::normal_logcdf_derivatives(x.value()))
}

fn lgamma<S: JetField>(x: &S) -> S {
    x.compose_unary(gam_math::jet_tower::ln_gamma_derivative_stack(x.value()))
}

pub(super) fn validate_value(family: &MeasurementFamily, y: f64) -> Result<(), EventHistoryError> {
    let valid = y.is_finite()
        && match family {
            MeasurementFamily::StudentT => true,
            MeasurementFamily::BinaryProbit => y == 0.0 || y == 1.0,
            MeasurementFamily::OrdinalProbit { categories } => {
                y >= 0.0 && y.fract() == 0.0 && y < *categories as f64
            }
            MeasurementFamily::NegativeBinomial => y >= 0.0 && y.fract() == 0.0,
        };
    if valid {
        Ok(())
    } else {
        Err(invalid("measurement is outside its declared support"))
    }
}

/// Analytic first and second location derivatives. The structured latent
/// solve needs these two numbers, not coefficient/shape derivative towers.
/// In particular, Student-t updates do not evaluate gamma functions here.
pub(super) fn location_derivatives(
    family: &MeasurementFamily,
    y: f64,
    eta: f64,
    shape: &[f64],
) -> Result<(f64, f64), EventHistoryError> {
    let probit = |x: f64, sign: f64| {
        let d = gam_math::probability::normal_logcdf_derivatives(x);
        (sign * d[1], d[2])
    };
    let (score, curvature) = match family {
        MeasurementFamily::BinaryProbit => {
            let sign = if y == 1.0 { 1.0 } else { -1.0 };
            probit(sign * eta, sign)
        }
        MeasurementFamily::OrdinalProbit { categories } => {
            let category = y as usize;
            let mut lower = 0.0;
            let mut upper = 0.0;
            let mut threshold = 0.0;
            for (j, gap) in shape.iter().enumerate() {
                let next = threshold + softplus(gap);
                if next <= threshold || !next.is_finite() {
                    return Err(numerical("ordinal threshold gap is unresolved"));
                }
                threshold = next;
                if j + 1 == category.saturating_sub(1) {
                    lower = threshold;
                }
                if j + 1 == category {
                    upper = threshold;
                }
            }
            if category == 0 {
                probit(-eta, -1.0)
            } else if category == categories - 1 {
                probit(eta - lower, 1.0)
            } else {
                // log p = L + log(1-exp(S-L)), choosing the CDF or survival
                // tail so neither endpoint probability rounds to one.
                let (large, small, sign) = if lower > eta {
                    (eta - lower, eta - upper, 1.0)
                } else {
                    (upper - eta, lower - eta, -1.0)
                };
                let l = gam_math::probability::normal_logcdf_derivatives(large);
                let s = gam_math::probability::normal_logcdf_derivatives(small);
                let separation = l[0] - s[0];
                if separation <= 0.0 || !separation.is_finite() {
                    return Err(numerical("ordinal interval probability is unresolved"));
                }
                let weight = 1.0 / separation.exp_m1();
                let first = sign * (s[1] - l[1]);
                let second = s[2] - l[2];
                (
                    sign * l[1] - weight * first,
                    l[2] - weight * second - (weight * first) * ((1.0 + weight) * first),
                )
            }
        }
        MeasurementFamily::StudentT => {
            let df = 2.0 + softplus(&shape[1]);
            if df <= 2.0 || !df.is_finite() {
                return Err(numerical("Student-t degrees of freedom are unresolved"));
            }
            let residual = y - eta;
            // D = nu*sigma^2 + residual^2. Neither term is squared in f64;
            // the tail remains resolvable when residual^2 would overflow.
            let log_square = 2.0 * residual.abs().ln();
            let log_scale_square = df.ln() + 2.0 * shape[0];
            let log_d = crate::chain::log_sum_exp(&[log_scale_square, log_square]);
            let log_coefficient = df.ln_1p() - log_d;
            let score = if residual == 0.0 {
                0.0
            } else {
                residual.signum() * (log_coefficient + residual.abs().ln()).exp()
            };
            let tail_fraction = (log_square - log_d).exp();
            let contrast = 2.0 * tail_fraction - 1.0;
            let curvature = if contrast == 0.0 {
                0.0
            } else {
                contrast.signum() * (log_coefficient + contrast.abs().ln()).exp()
            };
            (score, curvature)
        }
        MeasurementFamily::NegativeBinomial => {
            let size = softplus(&shape[0]);
            if size <= 0.0 || !size.is_finite() {
                return Err(numerical("count dispersion is unresolved"));
            }
            let contrast = eta - size.ln();
            let p = (-softplus(&(-contrast))).exp();
            let q = (-softplus(&contrast)).exp();
            (y * q - size * p, -(y * p) * q - (size * q) * p)
        }
    };
    if !score.is_finite() || !curvature.is_finite() {
        return Err(numerical("non-finite measurement location derivatives"));
    }
    Ok((score, curvature))
}

/// Analytic scores in location and the declared raw shape coordinates.
pub(super) fn parameter_scores(
    family: &MeasurementFamily,
    y: f64,
    eta: f64,
    shape: &[f64],
) -> Result<(f64, Vec<f64>), EventHistoryError> {
    let location = location_derivatives(family, y, eta, shape)?.0;
    let sigmoid = |x: f64| (-softplus(&(-x))).exp();
    let mut scores = vec![0.0; shape.len()];
    match family {
        MeasurementFamily::BinaryProbit => {}
        MeasurementFamily::OrdinalProbit { categories } => {
            let category = y as usize;
            for j in 0..category.saturating_sub(1) {
                scores[j] = -location * sigmoid(shape[j]);
            }
            if category > 0 && category < categories - 1 {
                let upper: f64 = shape[..category].iter().map(softplus).sum();
                let z = upper - eta;
                let log_probability = log_density(family, y, &eta, shape)?;
                scores[category - 1] = (-0.5 * z * z
                    - 0.5 * (2.0 * std::f64::consts::PI).ln()
                    - log_probability
                    - softplus(&(-shape[category - 1])))
                .exp();
            }
        }
        MeasurementFamily::StudentT => {
            let df = 2.0 + softplus(&shape[1]);
            let contrast = 2.0 * ((y - eta).abs().ln() - shape[0]) - df.ln();
            let fraction = sigmoid(contrast);
            scores[0] = (df + 1.0) * fraction - 1.0;
            scores[1] = 0.5
                * sigmoid(shape[1])
                * (gam_math::jet_tower::digamma(0.5 * (df + 1.0))
                    - gam_math::jet_tower::digamma(0.5 * df)
                    - 1.0 / df
                    - softplus(&contrast)
                    + (1.0 + 1.0 / df) * fraction);
        }
        MeasurementFamily::NegativeBinomial => {
            let size = softplus(&shape[0]);
            let log_size = size.ln();
            let log_derivative = -softplus(&(-shape[0]));
            let derivative = log_derivative.exp();
            let total = crate::chain::log_sum_exp(&[log_size, eta]);
            // psi(size+y)-psi(size) = 1/size + psi(size+y)-psi(size+1)
            // for a positive count. Multiplying by the softplus derivative
            // before exposing 1/size preserves a finite score at small size.
            let gamma_score = if y == 0.0 {
                0.0
            } else {
                (log_derivative - log_size).exp()
                    * (1.0
                        + size
                            * (gam_math::jet_tower::digamma(size + y)
                                - gam_math::jet_tower::digamma(size + 1.0)))
            };
            let count_term = if y == 0.0 {
                0.0
            } else {
                (y.ln() - total + log_derivative).exp()
            };
            scores[0] =
                gamma_score + derivative * (log_size - total + (eta - total).exp()) - count_term;
        }
    }
    if scores.iter().any(|v| !v.is_finite()) {
        return Err(numerical("non-finite measurement shape score"));
    }
    Ok((location, scores))
}

pub(super) fn log_density<S: JetField>(
    family: &MeasurementFamily,
    y: f64,
    eta: &S,
    shape: &[S],
) -> Result<S, EventHistoryError> {
    let out = match family {
        MeasurementFamily::BinaryProbit => logcdf(&eta.scale(if y == 1.0 { 1.0 } else { -1.0 })),
        MeasurementFamily::OrdinalProbit { categories } => {
            let category = y as usize;
            let mut cutpoints = vec![eta.constant_like(0.0)];
            for gap in shape {
                let next = cutpoints
                    .last()
                    .expect("first threshold is fixed")
                    .add(&softplus(gap));
                if next.value() <= cutpoints.last().expect("first threshold is fixed").value() {
                    return Err(numerical("ordinal threshold gap is unresolved"));
                }
                cutpoints.push(next);
            }
            if category == 0 {
                logcdf(&cutpoints[0].sub(eta))
            } else if category == categories - 1 {
                logcdf(&eta.sub(&cutpoints[category - 1]))
            } else {
                let lower = cutpoints[category - 1].sub(eta);
                let upper = cutpoints[category].sub(eta);
                let (large, small) = if lower.value() > 0.0 {
                    (logcdf(&lower.neg()), logcdf(&upper.neg()))
                } else {
                    (logcdf(&upper), logcdf(&lower))
                };
                large.add(&ln(&expm1(&small.sub(&large)).neg()))
            }
        }
        MeasurementFamily::StudentT => {
            let log_scale = &shape[0];
            let df = add_real(&softplus(&shape[1]), 2.0);
            if df.value() <= 2.0 {
                return Err(numerical("Student-t degrees of freedom are unresolved"));
            }
            let residual = eta.constant_like(y).sub(eta);
            let log_magnitude = residual.value().abs().ln() - log_scale.value();
            let tail = if log_scale.value().abs() < 300.0 && log_magnitude < 300.0 {
                let standard = residual.mul(&exp(&log_scale.neg()));
                log1p(&div(&standard.mul(&standard), &df))
            } else if residual.value() == 0.0 {
                return Err(numerical(
                    "Student-t scale cannot resolve the density curvature",
                ));
            } else {
                let abs = residual.scale(residual.value().signum());
                softplus(&ln(&abs).scale(2.0).sub(&log_scale.scale(2.0)).sub(&ln(&df)))
            };
            lgamma(&add_real(&df, 1.0).scale(0.5))
                .sub(&lgamma(&df.scale(0.5)))
                .sub(&add_real(&ln(&df), std::f64::consts::PI.ln()).scale(0.5))
                .sub(log_scale)
                .sub(&add_real(&df, 1.0).mul(&tail).scale(0.5))
        }
        MeasurementFamily::NegativeBinomial => {
            let size = softplus(&shape[0]);
            if size.value() <= 0.0 {
                return Err(numerical("count dispersion is unresolved"));
            }
            let log_size = ln(&size);
            let total = crate::chain::log_sum_exp(&[log_size.clone(), eta.clone()]);
            lgamma(&add_real(&size, y))
                .sub(&lgamma(&size))
                .sub(&lgamma(&eta.constant_like(y + 1.0)))
                .add(&size.mul(&log_size.sub(&total)))
                .add(&eta.sub(&total).scale(y))
        }
    };
    if !out.value().is_finite() {
        return Err(numerical("non-finite measurement log likelihood"));
    }
    Ok(out)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::scalar::Mixed;

    fn compare(family: &MeasurementFamily, y: f64, eta: f64, shape: &[f64]) {
        let analytic = location_derivatives(family, y, eta, shape).unwrap();
        let seeded: Vec<_> = shape.iter().map(|&v| Mixed::seed(v, 0.0, 0.0)).collect();
        let jet = log_density(family, y, &Mixed::seed(eta, 1.0, 1.0), &seeded).unwrap();
        for (a, b) in [(analytic.0, jet.u), (analytic.1, jet.uv)] {
            assert!(
                (a - b).abs() <= 2e-12 + 2e-9 * b.abs(),
                "{family:?}: y={y}, eta={eta}, shape={shape:?}: analytic {a}, AD {b}"
            );
        }
    }

    #[test]
    fn analytic_measurement_locations_agree_with_density_derivatives() {
        for eta in [-40.0, -12.0, -2.0, 0.0, 0.7, 3.0, 12.0, 40.0] {
            for y in [0.0, 1.0] {
                compare(&MeasurementFamily::BinaryProbit, y, eta, &[]);
            }
            for y in [0.0, 1.0, 2.0, 3.0] {
                for gap in [-3.0, 0.5, 3.0] {
                    compare(
                        &MeasurementFamily::OrdinalProbit { categories: 4 },
                        y,
                        eta,
                        &[gap, -0.3],
                    );
                }
            }
            for y in [-1e100, -1.7, 0.0, 0.9, 1e100] {
                for log_scale in [-4.0, 0.0, 4.0] {
                    compare(&MeasurementFamily::StudentT, y, eta, &[log_scale, 0.2]);
                }
            }
            for y in [0.0, 1.0, 100.0] {
                for shape in [-5.0, 0.0, 5.0] {
                    compare(&MeasurementFamily::NegativeBinomial, y, eta, &[shape]);
                }
            }
        }
    }

    #[test]
    fn analytic_measurement_tails_preserve_representable_scores() {
        let (score, curvature) =
            location_derivatives(&MeasurementFamily::StudentT, 1e200, 0.0, &[0.0, 0.0]).unwrap();
        let df = 2.0 + 2.0_f64.ln();
        assert!((score * 1e200 - (df + 1.0)).abs() < 1e-12);
        assert_eq!(curvature, 0.0); // Its true order is 1e-400.
        let center =
            location_derivatives(&MeasurementFamily::StudentT, 0.0, 0.0, &[0.0, 0.0]).unwrap();
        assert_eq!(center.0, 0.0);
        assert!((center.1 + (df + 1.0) / df).abs() < 1e-14);
        // The curvature prefactor overflows, but its product with the small
        // contrast near the inflection point is representable.
        let inflection = location_derivatives(
            &MeasurementFamily::StudentT,
            (-358.0_f64).exp(),
            0.0,
            &[-358.0 - 0.5 * df.ln() + 1e-12, 0.0],
        )
        .unwrap();
        assert!(inflection.0.is_finite() && inflection.1.is_finite());
        assert!(inflection.1 < 0.0);
        let size = 2.0_f64.ln();
        let left = location_derivatives(&MeasurementFamily::NegativeBinomial, 7.0, -800.0, &[0.0])
            .unwrap();
        let right =
            location_derivatives(&MeasurementFamily::NegativeBinomial, 7.0, 800.0, &[0.0]).unwrap();
        assert_eq!(left, (7.0, -0.0));
        assert_eq!(right, (-size, -0.0));
        assert!(
            location_derivatives(&MeasurementFamily::StudentT, 0.0, 0.0, &[-800.0, 0.0]).is_err()
        );
    }

    #[test]
    fn measurement_location_derivative_speed() {
        use std::hint::black_box;
        use std::time::Instant;
        let cases = [
            (MeasurementFamily::StudentT, 0.7, vec![0.1, 0.2]),
            (MeasurementFamily::BinaryProbit, 1.0, vec![]),
            (
                MeasurementFamily::OrdinalProbit { categories: 4 },
                1.0,
                vec![0.4, 0.2],
            ),
            (MeasurementFamily::NegativeBinomial, 3.0, vec![0.2]),
        ];
        for (family, y, shape) in cases {
            let seeded: Vec<_> = shape.iter().map(|&v| Mixed::seed(v, 0.0, 0.0)).collect();
            let mut analytic_time = f64::INFINITY;
            let mut ad_time = f64::INFINITY;
            for _ in 0..3 {
                let start = Instant::now();
                for n in 0..20000 {
                    let eta = black_box((n % 127) as f64 / 16.0 - 4.0);
                    black_box(
                        location_derivatives(
                            black_box(&family),
                            black_box(y),
                            eta,
                            black_box(&shape),
                        )
                        .unwrap(),
                    );
                }
                analytic_time = analytic_time.min(start.elapsed().as_secs_f64());
                let start = Instant::now();
                for n in 0..20000 {
                    let eta = black_box((n % 127) as f64 / 16.0 - 4.0);
                    let result = log_density(
                        black_box(&family),
                        black_box(y),
                        &Mixed::seed(eta, 1.0, 1.0),
                        black_box(&seeded),
                    )
                    .unwrap();
                    black_box((result.u, result.uv));
                }
                ad_time = ad_time.min(start.elapsed().as_secs_f64());
            }
            eprintln!(
                "measurement location {family:?}: analytic {analytic_time:.6}s, AD {ad_time:.6}s, speedup {:.3}x (20000 calls, best of three)",
                ad_time / analytic_time
            );
        }
    }
}
