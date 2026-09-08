//! Exact fifth derivatives for the six-primary follow-up slope frame.
//!
//! Entry and exit index terms reuse the static kernel's mixed Bell polynomial.
//! The event Jacobian also depends on the slope rate; its log derivative is a
//! finite Taylor composition in (q, qdot, g, gdot), including the q*gdot term.

use super::information_third::{FACTORIAL, mixed_fifth};
use super::*;

// Base-six indexing makes multiplication an index addition when total degree
// stays <= 5. Only 126 of the 1296 slots are visited.
const POLYNOMIAL_SIZE: usize = 6 * 6 * 6 * 6;
const POWERS: [[usize; 4]; 126] = {
    let mut powers = [[0; 4]; 126];
    let mut at = 0;
    let mut q = 0;
    while q <= 5 {
        let mut qd = 0;
        while qd <= 5 - q {
            let mut g = 0;
            while g <= 5 - q - qd {
                let mut gd = 0;
                while gd <= 5 - q - qd - g {
                    powers[at] = [q, qd, g, gd];
                    at += 1;
                    gd += 1;
                }
                g += 1;
            }
            qd += 1;
        }
        q += 1;
    }
    powers
};

fn index(powers: [usize; 4]) -> usize {
    ((powers[0] * 6 + powers[1]) * 6 + powers[2]) * 6 + powers[3]
}

/// Coefficients of c(g+t), where c²=1+a(g+t)². One extra order gives
/// c'(g+t) through fifth order for the event's time-derivative factor.
fn scale_coefficients(g: f64, a: f64) -> [f64; 7] {
    let mut c = [0.0; 7];
    c[0] = (1.0 + a * g * g).sqrt();
    for n in 1..=6 {
        let rhs = match n {
            1 => 2.0 * a * g,
            2 => a,
            _ => 0.0,
        };
        c[n] = (rhs - (1..n).map(|j| c[j] * c[n - j]).sum::<f64>()) / (2.0 * c[0]);
    }
    c
}

/// Taylor coefficients of log(c(g)*qdot + (q*c'(g)+b)*gdot).
/// The finite log-composition polynomial differentiates analytically; there
/// is no step size or numerical differentiation in the production path.
fn log_derivative_coefficients(
    q: f64,
    qd: f64,
    g: f64,
    gd: f64,
    a: f64,
    b: f64,
) -> [f64; POLYNOMIAL_SIZE] {
    let c = scale_coefficients(g, a);
    let mut relative = [0.0; POLYNOMIAL_SIZE];
    for order in 0..=5 {
        let c_prime = (order + 1) as f64 * c[order + 1];
        let affine = q * c_prime + if order == 0 { b } else { 0.0 };
        relative[index([0, 0, order, 0])] = qd * c[order] + gd * affine;
        if order < 5 {
            relative[index([1, 0, order, 0])] = gd * c_prime;
            relative[index([0, 1, order, 0])] = c[order];
            relative[index([0, 0, order, 1])] = affine;
        }
        if order < 4 {
            relative[index([1, 0, order, 1])] = c_prime;
        }
    }
    let value = relative[0];
    relative[0] = 0.0;
    for coefficient in &mut relative {
        *coefficient /= value;
    }
    let mut power = [0.0; POLYNOMIAL_SIZE];
    power[0] = 1.0;
    let mut logarithm = [0.0; POLYNOMIAL_SIZE];
    for order in 1..=5 {
        let mut next = [0.0; POLYNOMIAL_SIZE];
        for left in POWERS {
            let li = index(left);
            if power[li] == 0.0 {
                continue;
            }
            let degree = left.iter().sum::<usize>();
            for right in POWERS {
                if degree + right.iter().sum::<usize>() > 5 {
                    continue;
                }
                let ri = index(right);
                if relative[ri] != 0.0 {
                    next[li + ri] += power[li] * relative[ri];
                }
            }
        }
        let sign = if order % 2 == 1 { 1.0 } else { -1.0 };
        for powers in POWERS {
            let at = index(powers);
            logarithm[at] += sign / order as f64 * next[at];
        }
        power = next;
    }
    logarithm
}

fn dynamic_row_fifth(
    primaries: &[f64; DYNAMIC_SLOPE_PRIMARIES],
    inputs: &RigidRowInputs,
) -> Result<[[[[[f64; 6]; 6]; 6]; 6]; 6], String> {
    // The same admission/value program as the coefficient line search guards
    // the branch whose fifth derivatives are extracted below.
    rigid_row_value::<6, DynamicSlopeGeometry>(primaries, inputs)?;
    if inputs.wi == 0.0 {
        return Ok([[[[[0.0; 6]; 6]; 6]; 6]; 6]);
    }
    let a = inputs.probit_scale.powi(2) * inputs.covariance_ones;
    let b = inputs.probit_scale * inputs.z_sum;
    let scale0 = scale_coefficients(primaries[PRIMARY_SLOPE], a);
    let scale1 = scale_coefficients(primaries[PRIMARY_SLOPE_EXIT], a);
    let c0: [f64; 6] = std::array::from_fn(|j| scale0[j]);
    let c1: [f64; 6] = std::array::from_fn(|j| scale1[j]);
    let eta0 = primaries[PRIMARY_Q0] * c0[0] + b * primaries[PRIMARY_SLOPE];
    let eta1 = primaries[PRIMARY_Q1] * c1[0] + b * primaries[PRIMARY_SLOPE_EXIT];
    let entry = gam_math::probability::normal_logcdf_derivatives_through_fifth(-eta0)
        .map(|value| inputs.wi * value);
    let exit = gam_math::probability::normal_logcdf_derivatives_through_fifth(-eta1)
        .map(|value| -inputs.wi * (1.0 - inputs.di) * value);
    let q0 = mixed_fifth(primaries[PRIMARY_Q0], &c0.map(|x| -x), -b, &entry);
    let mut q1 = mixed_fifth(primaries[PRIMARY_Q1], &c1.map(|x| -x), -b, &exit);
    let event_weight = inputs.wi * inputs.di;
    let density = [
        0.5 * event_weight * eta1 * eta1,
        event_weight * eta1,
        event_weight,
        0.0,
        0.0,
        0.0,
    ];
    let density_fifth = mixed_fifth(primaries[PRIMARY_Q1], &c1, b, &density);
    for m in 0..=5 {
        q1[m] += density_fifth[m];
    }
    let log_derivative = if event_weight == 0.0 {
        [0.0; POLYNOMIAL_SIZE]
    } else {
        log_derivative_coefficients(
            primaries[PRIMARY_Q1],
            primaries[PRIMARY_QD1],
            primaries[PRIMARY_SLOPE_EXIT],
            primaries[PRIMARY_SLOPE_RATE],
            a,
            b,
        )
    };
    Ok(std::array::from_fn(|i| {
        std::array::from_fn(|j| {
            std::array::from_fn(|k| {
                std::array::from_fn(|l| {
                    std::array::from_fn(|m| {
                        let mut counts = [0usize; 6];
                        for axis in [i, j, k, l, m] {
                            counts[axis] += 1;
                        }
                        if counts[PRIMARY_Q0] + counts[PRIMARY_SLOPE] == 5 {
                            return q0[counts[PRIMARY_Q0]];
                        }
                        if counts[PRIMARY_Q0] + counts[PRIMARY_SLOPE] != 0 {
                            return 0.0;
                        }
                        let powers = [
                            counts[PRIMARY_Q1],
                            counts[PRIMARY_QD1],
                            counts[PRIMARY_SLOPE_EXIT],
                            counts[PRIMARY_SLOPE_RATE],
                        ];
                        let mut value = -event_weight
                            * log_derivative[index(powers)]
                            * powers.iter().map(|&n| FACTORIAL[n]).product::<f64>();
                        if counts[PRIMARY_Q1] + counts[PRIMARY_SLOPE_EXIT] == 5 {
                            value += q1[counts[PRIMARY_Q1]];
                        }
                        value
                    })
                })
            })
        })
    }))
}

impl SurvivalMarginalSlopeRowKernel<DYNAMIC_SLOPE_PRIMARIES, DynamicSlopeGeometry> {
    pub(crate) fn third_information_all_axes(
        &self,
        u: &[f64],
        v: &[f64],
    ) -> Result<Vec<Array2<f64>>, String> {
        self.third_information_all_axes_from(u, v, dynamic_row_fifth)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use gam_math::jet_scalar::JetScalar;

    #[test]
    fn follow_up_fifth_matches_differentiated_fourth_for_events_and_censoring_2765() {
        for event in [0.0, 1.0] {
            for rate in [-0.3, 0.0, 0.4] {
                let inputs = RigidRowInputs {
                    row: 0,
                    wi: 1.7,
                    di: event,
                    z_sum: 0.7,
                    covariance_ones: 1.2,
                    probit_scale: 0.9,
                    qd1_lower: 1e-8,
                };
                let point = [-0.9, 0.4, 1.1, -0.7, 0.8, rate];
                let exact = dynamic_row_fifth(&point, &inputs).expect("admitted dynamic row");
                for axis in 0..6 {
                    let step = 1e-5;
                    let tower = |sign: f64| {
                        let mut shifted = point;
                        shifted[axis] += sign * step;
                        let vars: [SparseTower4<6, RIGID_LINEAR_MASK>; 6] =
                            std::array::from_fn(|a| SparseTower4::variable(shifted[a], a));
                        rigid_row_nll::<6, DynamicSlopeGeometry, _>(&vars, &inputs)
                            .expect("perturbed dynamic row")
                    };
                    let plus = tower(1.0);
                    let minus = tower(-1.0);
                    for a in 0..6 {
                        for b in 0..6 {
                            for c in 0..6 {
                                for d in 0..6 {
                                    let fd =
                                        (plus.t4[a][b][c][d] - minus.t4[a][b][c][d]) / (2.0 * step);
                                    let actual = exact[a][b][c][d][axis];
                                    assert!(
                                        (actual - fd).abs() <= 3e-6 * (1.0 + fd.abs()),
                                        "event={event} rate={rate} axes={a},{b},{c},{d},{axis}: exact={actual} FD={fd}"
                                    );
                                }
                            }
                        }
                    }
                }
            }
        }
    }
}
