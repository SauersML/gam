use super::{MeasurementFamily, invalid, numerical};
use crate::EventHistoryError;
use crate::scalar::{add_real, div, exp, ln};
use gam_math::nested_dual::JetField;

pub(super) fn softplus<S: JetField>(x: &S) -> S {
    let u = x.value();
    let value = if u > 0.0 { u + (-u).exp().ln_1p() } else { u.exp().ln_1p() };
    let s = if u >= 0.0 { 1.0 / (1.0 + (-u).exp()) } else { let e = u.exp(); e / (1.0 + e) };
    let d = s * (1.0 - s);
    x.compose_unary([value, s, d, d * (1.0 - 2.0 * s), d * (1.0 - 6.0 * d)])
}

pub(super) fn expm1<S: JetField>(x: &S) -> S {
    let e = x.value().exp();
    x.compose_unary([x.value().exp_m1(), e, e, e, e])
}

fn log1p<S: JetField>(x: &S) -> S {
    let inv = 1.0 / (1.0 + x.value());
    x.compose_unary([x.value().ln_1p(), inv, -inv * inv,
        2.0 * inv.powi(3), -6.0 * inv.powi(4)])
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
    } else { ln(&softplus(x)) }
}

fn logcdf<S: JetField>(x: &S) -> S {
    x.compose_unary(gam_math::probability::normal_logcdf_derivatives(x.value()))
}

fn lgamma<S: JetField>(x: &S) -> S {
    x.compose_unary(gam_math::jet_tower::ln_gamma_derivative_stack(x.value()))
}

pub(super) fn validate_value(family: &MeasurementFamily, y: f64) -> Result<(), EventHistoryError> {
    let valid = y.is_finite() && match family {
        MeasurementFamily::StudentT => true,
        MeasurementFamily::BinaryProbit => y == 0.0 || y == 1.0,
        MeasurementFamily::OrdinalProbit { categories } => y >= 0.0 && y.fract() == 0.0 && y < *categories as f64,
        MeasurementFamily::NegativeBinomial => y >= 0.0 && y.fract() == 0.0,
    };
    if valid { Ok(()) } else { Err(invalid("measurement is outside its declared support")) }
}

pub(super) fn log_density<S: JetField>(family: &MeasurementFamily, y: f64, eta: &S, shape: &[S])
    -> Result<S, EventHistoryError> {
    let out = match family {
        MeasurementFamily::BinaryProbit => logcdf(&eta.scale(if y == 1.0 { 1.0 } else { -1.0 })),
        MeasurementFamily::OrdinalProbit { categories } => {
            let category = y as usize;
            let mut cutpoints = vec![eta.constant_like(0.0)];
            for gap in shape {
                let next = cutpoints.last().expect("first threshold is fixed").add(&softplus(gap));
                if next.value() <= cutpoints.last().expect("first threshold is fixed").value() {
                    return Err(numerical("ordinal threshold gap is unresolved"));
                }
                cutpoints.push(next);
            }
            if category == 0 { logcdf(&cutpoints[0].sub(eta)) }
            else if category == categories - 1 { logcdf(&eta.sub(&cutpoints[category - 1])) }
            else {
                let lower = cutpoints[category - 1].sub(eta);
                let upper = cutpoints[category].sub(eta);
                let (large, small) = if lower.value() > 0.0 {
                    (logcdf(&lower.neg()), logcdf(&upper.neg()))
                } else { (logcdf(&upper), logcdf(&lower)) };
                large.add(&ln(&expm1(&small.sub(&large)).neg()))
            }
        }
        MeasurementFamily::StudentT => {
            let log_scale = &shape[0];
            let df = add_real(&softplus(&shape[1]), 2.0);
            if df.value() <= 2.0 { return Err(numerical("Student-t degrees of freedom are unresolved")); }
            let residual = eta.constant_like(y).sub(eta);
            let log_magnitude = residual.value().abs().ln() - log_scale.value();
            let tail = if log_scale.value().abs() < 300.0 && log_magnitude < 300.0 {
                let standard = residual.mul(&exp(&log_scale.neg()));
                log1p(&div(&standard.mul(&standard), &df))
            } else if residual.value() == 0.0 {
                return Err(numerical("Student-t scale cannot resolve the density curvature"));
            } else {
                let abs = residual.scale(residual.value().signum());
                softplus(&ln(&abs).scale(2.0).sub(&log_scale.scale(2.0)).sub(&ln(&df)))
            };
            lgamma(&add_real(&df, 1.0).scale(0.5)).sub(&lgamma(&df.scale(0.5)))
                .sub(&add_real(&ln(&df), std::f64::consts::PI.ln()).scale(0.5))
                .sub(log_scale).sub(&add_real(&df, 1.0).mul(&tail).scale(0.5))
        }
        MeasurementFamily::NegativeBinomial => {
            let size = softplus(&shape[0]);
            if size.value() <= 0.0 { return Err(numerical("count dispersion is unresolved")); }
            let log_size = ln(&size);
            let total = crate::chain::log_sum_exp(&[log_size.clone(), eta.clone()]);
            lgamma(&add_real(&size, y)).sub(&lgamma(&size))
                .sub(&lgamma(&eta.constant_like(y + 1.0)))
                .add(&size.mul(&log_size.sub(&total)))
                .add(&eta.sub(&total).scale(y))
        }
    };
    if !out.value().is_finite() { return Err(numerical("non-finite measurement log likelihood")); }
    Ok(out)
}
