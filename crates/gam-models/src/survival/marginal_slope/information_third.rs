//! Exact fifth likelihood derivatives for rigid, time-constant slopes.
//!
//! The static row separates into functions of (q0,g), (q1,g), and qd1.
//! Differentiate each unary composition analytically in q first, then use the
//! finite Bell polynomial in g. This computes the eleven potentially nonzero
//! mixed fifth derivatives once per row, before any coefficient pullback.

use super::*;
use crate::row_kernel::RowKernel;

pub(super) const FACTORIAL: [f64; 6] = [1.0, 1.0, 2.0, 6.0, 24.0, 120.0];

fn product(a: &[f64; 6], b: &[f64; 6]) -> [f64; 6] {
    std::array::from_fn(|n| (0..=n).map(|j| a[j] * b[n - j]).sum())
}

/// Mixed derivatives d_q^m d_g^(5-m) F(q*c(g)+b*g), m=0,...,5.
/// The leaf stack is with respect to its scalar index; c holds Taylor
/// coefficients, so factorials enter only at the final coefficient extraction.
pub(super) fn mixed_fifth(q: f64, c: &[f64; 6], b: f64, leaf: &[f64; 6]) -> [f64; 6] {
    let mut delta = c.map(|value| q * value);
    delta[0] = 0.0;
    delta[1] += b;
    let mut powers = [[0.0; 6]; 6];
    powers[0][0] = 1.0;
    for j in 1..=5 {
        powers[j] = product(&powers[j - 1], &delta);
    }
    let mut c_power = [0.0; 6];
    c_power[0] = 1.0;
    let mut out = [0.0; 6];
    for m in 0..=5 {
        let k = 5 - m;
        let mut composed = [0.0; 6];
        for j in 0..=k {
            for n in 0..=k {
                composed[n] += leaf[m + j] * powers[j][n] / FACTORIAL[j];
            }
        }
        out[m] = product(&c_power, &composed)[k] * FACTORIAL[k];
        if m < 5 {
            c_power = product(&c_power, c);
        }
    }
    out
}

fn static_row_fifth(
    primaries: &[f64; STATIC_SLOPE_PRIMARIES],
    inputs: &RigidRowInputs,
) -> Result<[[[[[f64; 4]; 4]; 4]; 4]; 4], String> {
    if inputs.wi == 0.0 {
        return Ok([[[[[0.0; 4]; 4]; 4]; 4]; 4]);
    }
    let [neg_eta0, neg_eta1, derivative] =
        rigid_row_admission_witnesses::<4, StaticSlopeGeometry>(primaries, inputs);
    validate_rigid_row_admission::<4, StaticSlopeGeometry>(
        primaries[PRIMARY_QD1],
        inputs,
        neg_eta0,
        neg_eta1,
        derivative,
    )?;
    let g = primaries[PRIMARY_SLOPE];
    let a = inputs.probit_scale.powi(2) * inputs.covariance_ones;
    let b = inputs.probit_scale * inputs.z_sum;
    // c(g+t)^2 = 1+a(g+t)^2 gives this exact triangular recurrence.
    let mut c = [0.0; 6];
    c[0] = (1.0 + a * g * g).sqrt();
    for n in 1..=5 {
        let rhs = match n {
            1 => 2.0 * a * g,
            2 => a,
            _ => 0.0,
        };
        c[n] = (rhs - (1..n).map(|j| c[j] * c[n - j]).sum::<f64>()) / (2.0 * c[0]);
    }
    let neg_c = c.map(|value| -value);
    let entry = gam_math::probability::normal_logcdf_derivatives_through_fifth(neg_eta0)
        .map(|value| inputs.wi * value);
    let exit = gam_math::probability::normal_logcdf_derivatives_through_fifth(neg_eta1)
        .map(|value| -inputs.wi * (1.0 - inputs.di) * value);
    // The negative-index leaf's q derivative contributes (-c)^m.
    let q0 = mixed_fifth(primaries[PRIMARY_Q0], &neg_c, -b, &entry);
    let mut q1 = mixed_fifth(primaries[PRIMARY_Q1], &neg_c, -b, &exit);
    let event_weight = inputs.wi * inputs.di;
    let eta1 = -neg_eta1;
    let density = [
        0.5 * event_weight * eta1 * eta1,
        event_weight * eta1,
        event_weight,
        0.0,
        0.0,
        0.0,
    ];
    let density_fifth = mixed_fifth(primaries[PRIMARY_Q1], &c, b, &density);
    for m in 0..=5 {
        q1[m] += density_fifth[m];
    }
    // -w*d*log(qd1*c) separates exactly. The fifth log(c) coefficient
    // follows from log(c)=log(1+a*g^2)/2 and its finite Taylor series.
    let mut relative = [0.0; 6];
    relative[1] = 2.0 * a * g / c[0].powi(2);
    relative[2] = a / c[0].powi(2);
    let mut power = [0.0; 6];
    power[0] = 1.0;
    let mut log_c_fifth = 0.0;
    for j in 1..=5 {
        power = product(&power, &relative);
        let sign = if j % 2 == 1 { 1.0 } else { -1.0 };
        log_c_fifth += 0.5 * sign * power[5] * FACTORIAL[5] / j as f64;
    }
    let slope_fifth = q0[0] + q1[0] - event_weight * log_c_fifth;
    let derivative_fifth = -event_weight * 24.0 / primaries[PRIMARY_QD1].powi(5);
    Ok(std::array::from_fn(|i| {
        std::array::from_fn(|j| {
            std::array::from_fn(|k| {
                std::array::from_fn(|l| {
                    std::array::from_fn(|m| {
                        let axes = [i, j, k, l, m];
                        let count = |axis| axes.iter().filter(|&&value| value == axis).count();
                        let n0 = count(PRIMARY_Q0);
                        let n1 = count(PRIMARY_Q1);
                        let nd = count(PRIMARY_QD1);
                        if n0 + n1 + nd == 0 {
                            slope_fifth
                        } else if n0 > 0 && n1 + nd == 0 {
                            q0[n0]
                        } else if n1 > 0 && n0 + nd == 0 {
                            q1[n1]
                        } else if nd == 5 {
                            derivative_fifth
                        } else {
                            0.0
                        }
                    })
                })
            })
        })
    }))
}

impl SurvivalMarginalSlopeRowKernel<STATIC_SLOPE_PRIMARIES, StaticSlopeGeometry> {
    pub(crate) fn third_information_all_axes(
        &self,
        u: &[f64],
        v: &[f64],
    ) -> Result<Vec<Array2<f64>>, String> {
        self.third_information_all_axes_from(u, v, static_row_fifth)
    }
}

impl<const P: usize, G: SlopeRowGeometry<P>> SurvivalMarginalSlopeRowKernel<P, G> {
    pub(super) fn third_information_all_axes_from(
        &self,
        u: &[f64],
        v: &[f64],
        fifth: impl Fn(&[f64; P], &RigidRowInputs) -> Result<[[[[[f64; P]; P]; P]; P]; P], String>,
    ) -> Result<Vec<Array2<f64>>, String> {
        let p = self.n_coefficients();
        if u.len() != p || v.len() != p || u.iter().chain(v).any(|x| !x.is_finite()) {
            return Err(format!(
                "survival third information derivative requires finite directions of length {p}"
            ));
        }
        let identity = Array2::<f64>::eye(p);
        let jacobians = self
            .jacobian_action_matrix(identity.view())
            .ok_or("survival third information derivative requires row Jacobians")?;
        if jacobians.dim() != (self.family.n, P * p) {
            return Err("survival third information row Jacobian shape mismatch".into());
        }
        let mut axes = vec![Array2::<f64>::zeros((p, p)); p];
        for row in 0..self.family.n {
            let inputs = rigid_row_inputs(
                &self.family,
                &self.block_states,
                row,
                "third information derivative",
            )?;
            let primaries =
                rigid_row_kernel_primaries::<P, G>(&self.family, &self.block_states, row)?;
            let fifth = fifth(&primaries, &inputs)?;
            let du = self.jacobian_action(row, u);
            let dv = self.jacobian_action(row, v);
            let mut tensor = [[[0.0; P]; P]; P];
            for a in 0..P {
                for b in 0..P {
                    for c in 0..P {
                        for d in 0..P {
                            for e in 0..P {
                                tensor[a][b][c] += fifth[a][b][c][d][e] * du[d] * dv[e];
                            }
                        }
                    }
                }
            }
            for axis in 0..p {
                let mut primary_hessian = [[0.0; P]; P];
                for a in 0..P {
                    for b in 0..P {
                        for c in 0..P {
                            primary_hessian[a][b] +=
                                tensor[a][b][c] * jacobians[[row, c * p + axis]];
                        }
                    }
                }
                self.add_pullback_hessian(row, &primary_hessian, &mut axes[axis]);
            }
        }
        Ok(axes)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use gam_math::jet_scalar::JetScalar;

    #[test]
    fn static_fifth_matches_differentiated_fourth_for_events_and_censoring() {
        for event in [0.0, 1.0] {
            for slope in [-1.3, 0.0, 0.8] {
                let inputs = RigidRowInputs {
                    row: 0,
                    wi: 1.7,
                    di: event,
                    z_sum: 0.7,
                    covariance_ones: 1.2,
                    probit_scale: 0.9,
                    qd1_lower: 1e-8,
                };
                let point = [-0.9, 0.4, 1.1, slope];
                let exact = static_row_fifth(&point, &inputs).expect("admitted row");
                for axis in 0..4 {
                    let step = 1e-5;
                    let tower = |sign: f64| {
                        let mut shifted = point;
                        shifted[axis] += sign * step;
                        let vars: [SparseTower4<4, RIGID_LINEAR_MASK>; 4] =
                            std::array::from_fn(|a| SparseTower4::variable(shifted[a], a));
                        rigid_row_nll::<4, StaticSlopeGeometry, _>(&vars, &inputs)
                            .expect("perturbed row")
                    };
                    let plus = tower(1.0);
                    let minus = tower(-1.0);
                    for a in 0..4 {
                        for b in 0..4 {
                            for c in 0..4 {
                                for d in 0..4 {
                                    let fd =
                                        (plus.t4[a][b][c][d] - minus.t4[a][b][c][d]) / (2.0 * step);
                                    let actual = exact[a][b][c][d][axis];
                                    assert!(
                                        (actual - fd).abs() <= 2e-6 * (1.0 + fd.abs()),
                                        "event={event} slope={slope} axes={a},{b},{c},{d},{axis}: exact={actual} FD={fd}"
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
