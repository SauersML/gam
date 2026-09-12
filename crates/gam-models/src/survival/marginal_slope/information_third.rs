//! Exact fifth and sixth likelihood derivatives for rigid, time-constant slopes.
//!
//! The static row separates into functions of (q0,g), (q1,g), and qd1.
//! Differentiate each unary composition analytically in q first, then use the
//! finite Bell polynomial in g. This computes the eleven potentially nonzero
//! mixed fifth derivatives once per row, before any coefficient pullback. The
//! sixth is the same construction one order higher; the fourth information
//! derivative consumes it (gam#2894).

use super::*;
use crate::row_kernel::RowKernel;

pub(super) const FACTORIAL: [f64; 7] = [1.0, 1.0, 2.0, 6.0, 24.0, 120.0, 720.0];

fn product<const N: usize>(a: &[f64; N], b: &[f64; N]) -> [f64; N] {
    std::array::from_fn(|n| (0..=n).map(|j| a[j] * b[n - j]).sum())
}

/// Mixed derivatives d_q^m d_g^(K-m) F(q*c(g)+b*g), m=0,...,K, with K = N-1.
/// The leaf stack is with respect to its scalar index; c holds Taylor
/// coefficients, so factorials enter only at the final coefficient extraction.
fn mixed_order<const N: usize>(q: f64, c: &[f64; N], b: f64, leaf: &[f64; N]) -> [f64; N] {
    let order = N - 1;
    let mut delta = c.map(|value| q * value);
    delta[0] = 0.0;
    delta[1] += b;
    let mut powers = [[0.0; N]; N];
    powers[0][0] = 1.0;
    for j in 1..=order {
        powers[j] = product(&powers[j - 1], &delta);
    }
    let mut c_power = [0.0; N];
    c_power[0] = 1.0;
    let mut out = [0.0; N];
    for m in 0..=order {
        let k = order - m;
        let mut composed = [0.0; N];
        for j in 0..=k {
            for n in 0..=k {
                composed[n] += leaf[m + j] * powers[j][n] / FACTORIAL[j];
            }
        }
        out[m] = product(&c_power, &composed)[k] * FACTORIAL[k];
        if m < order {
            c_power = product(&c_power, c);
        }
    }
    out
}

/// [`mixed_order`] at order five, the frame the follow-up slope kernel shares.
pub(super) fn mixed_fifth(q: f64, c: &[f64; 6], b: f64, leaf: &[f64; 6]) -> [f64; 6] {
    mixed_order(q, c, b, leaf)
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

fn static_row_sixth(
    primaries: &[f64; STATIC_SLOPE_PRIMARIES],
    inputs: &RigidRowInputs,
) -> Result<[[[[[[f64; 4]; 4]; 4]; 4]; 4]; 4], String> {
    if inputs.wi == 0.0 {
        return Ok([[[[[[0.0; 4]; 4]; 4]; 4]; 4]; 4]);
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
    let neg_c = c.map(|value| -value);
    let entry = gam_math::probability::normal_logcdf_derivatives_through_sixth(neg_eta0)
        .map(|value| inputs.wi * value);
    let exit = gam_math::probability::normal_logcdf_derivatives_through_sixth(neg_eta1)
        .map(|value| -inputs.wi * (1.0 - inputs.di) * value);
    // The negative-index leaf's q derivative contributes (-c)^m.
    let q0 = mixed_order(primaries[PRIMARY_Q0], &neg_c, -b, &entry);
    let mut q1 = mixed_order(primaries[PRIMARY_Q1], &neg_c, -b, &exit);
    let event_weight = inputs.wi * inputs.di;
    let eta1 = -neg_eta1;
    let density = [
        0.5 * event_weight * eta1 * eta1,
        event_weight * eta1,
        event_weight,
        0.0,
        0.0,
        0.0,
        0.0,
    ];
    let density_sixth = mixed_order(primaries[PRIMARY_Q1], &c, b, &density);
    for m in 0..=6 {
        q1[m] += density_sixth[m];
    }
    // -w*d*log(qd1*c) separates exactly. The sixth log(c) coefficient
    // follows from log(c)=log(1+a*g^2)/2 and its finite Taylor series.
    let mut relative = [0.0; 7];
    relative[1] = 2.0 * a * g / c[0].powi(2);
    relative[2] = a / c[0].powi(2);
    let mut power = [0.0; 7];
    power[0] = 1.0;
    let mut log_c_sixth = 0.0;
    for j in 1..=6 {
        power = product(&power, &relative);
        let sign = if j % 2 == 1 { 1.0 } else { -1.0 };
        log_c_sixth += 0.5 * sign * power[6] * FACTORIAL[6] / j as f64;
    }
    let slope_sixth = q0[0] + q1[0] - event_weight * log_c_sixth;
    // d⁶(−log qd1)/dqd1⁶ = 120/qd1⁶.
    let derivative_sixth = event_weight * 120.0 / primaries[PRIMARY_QD1].powi(6);
    Ok(std::array::from_fn(|i| {
        std::array::from_fn(|j| {
            std::array::from_fn(|k| {
                std::array::from_fn(|l| {
                    std::array::from_fn(|m| {
                        std::array::from_fn(|n| {
                            let axes = [i, j, k, l, m, n];
                            let count =
                                |axis| axes.iter().filter(|&&value| value == axis).count();
                            let n0 = count(PRIMARY_Q0);
                            let n1 = count(PRIMARY_Q1);
                            let nd = count(PRIMARY_QD1);
                            if n0 + n1 + nd == 0 {
                                slope_sixth
                            } else if n0 > 0 && n1 + nd == 0 {
                                q0[n0]
                            } else if n1 > 0 && n0 + nd == 0 {
                                q1[n1]
                            } else if nd == 6 {
                                derivative_sixth
                            } else {
                                0.0
                            }
                        })
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

    /// `⟨W, H³[u, e_a, e_b]⟩` for every axis pair: the fifth likelihood derivatives
    /// contracted with the row-projected trace weight and one direction, pulled back as a
    /// Hessian in one row pass (gam#2894). Linear in the symmetric weight `W`.
    pub(crate) fn contracted_trace_hessian_directional(
        &self,
        weight: &Array2<f64>,
        u: &[f64],
    ) -> Result<Array2<f64>, String> {
        let p = self.n_coefficients();
        if weight.dim() != (p, p) || u.len() != p || u.iter().any(|x| !x.is_finite()) {
            return Err(format!(
                "survival contracted trace Hessian derivative requires a ({p}, {p}) weight and a finite direction of length {p}"
            ));
        }
        self.chunked_pullback_reduce(p, |row, acc| -> Result<(), String> {
            let inputs = rigid_row_inputs(
                &self.family,
                &self.block_states,
                row,
                "contracted trace Hessian derivative",
            )?;
            let primaries = rigid_row_kernel_primaries::<
                STATIC_SLOPE_PRIMARIES,
                StaticSlopeGeometry,
            >(&self.family, &self.block_states, row)?;
            let fifth = static_row_fifth(&primaries, &inputs)?;
            let w_row = self.primary_trace_weight(row, weight)?;
            let du = self.jacobian_action(row, u);
            let mut coeff = [[0.0_f64; STATIC_SLOPE_PRIMARIES]; STATIC_SLOPE_PRIMARIES];
            for c in 0..STATIC_SLOPE_PRIMARIES {
                for d in 0..STATIC_SLOPE_PRIMARIES {
                    let mut sum = 0.0;
                    for a in 0..STATIC_SLOPE_PRIMARIES {
                        for b in 0..STATIC_SLOPE_PRIMARIES {
                            for e in 0..STATIC_SLOPE_PRIMARIES {
                                sum += w_row[a][b] * fifth[a][b][c][d][e] * du[e];
                            }
                        }
                    }
                    coeff[c][d] = sum;
                }
            }
            self.add_pullback_hessian(row, &coeff, acc);
            Ok(())
        })
    }

    /// `⟨W, H⁴[u, w, e_a, e_b]⟩` for every axis pair: the sixth likelihood derivatives
    /// contracted with the row-projected trace weight and two directions, pulled back as a
    /// Hessian in one row pass (gam#2894).
    pub(crate) fn contracted_trace_hessian_second_directional(
        &self,
        weight: &Array2<f64>,
        u: &[f64],
        w: &[f64],
    ) -> Result<Array2<f64>, String> {
        let p = self.n_coefficients();
        if weight.dim() != (p, p)
            || u.len() != p
            || w.len() != p
            || u.iter().chain(w).any(|x| !x.is_finite())
        {
            return Err(format!(
                "survival contracted trace Hessian second derivative requires a ({p}, {p}) weight and finite directions of length {p}"
            ));
        }
        self.chunked_pullback_reduce(p, |row, acc| -> Result<(), String> {
            let inputs = rigid_row_inputs(
                &self.family,
                &self.block_states,
                row,
                "contracted trace Hessian second derivative",
            )?;
            let primaries = rigid_row_kernel_primaries::<
                STATIC_SLOPE_PRIMARIES,
                StaticSlopeGeometry,
            >(&self.family, &self.block_states, row)?;
            let sixth = static_row_sixth(&primaries, &inputs)?;
            let w_row = self.primary_trace_weight(row, weight)?;
            let du = self.jacobian_action(row, u);
            let dw = self.jacobian_action(row, w);
            let mut coeff = [[0.0_f64; STATIC_SLOPE_PRIMARIES]; STATIC_SLOPE_PRIMARIES];
            for c in 0..STATIC_SLOPE_PRIMARIES {
                for d in 0..STATIC_SLOPE_PRIMARIES {
                    let mut sum = 0.0;
                    for a in 0..STATIC_SLOPE_PRIMARIES {
                        for b in 0..STATIC_SLOPE_PRIMARIES {
                            for e in 0..STATIC_SLOPE_PRIMARIES {
                                for f in 0..STATIC_SLOPE_PRIMARIES {
                                    sum += w_row[a][b] * sixth[a][b][c][d][e][f] * du[e] * dw[f];
                                }
                            }
                        }
                    }
                    coeff[c][d] = sum;
                }
            }
            self.add_pullback_hessian(row, &coeff, acc);
            Ok(())
        })
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
        let mut tensors = Vec::with_capacity(self.family.n);
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
            tensors.push(tensor);
        }
        self.all_axes_primary_tensor_pullback(&tensors)
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

    #[test]
    fn static_sixth_matches_differentiated_fifth_for_events_and_censoring_2894() {
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
                let exact = static_row_sixth(&point, &inputs).expect("admitted row");
                for axis in 0..4 {
                    let step = 1e-5;
                    let fifth = |sign: f64| {
                        let mut shifted = point;
                        shifted[axis] += sign * step;
                        static_row_fifth(&shifted, &inputs).expect("perturbed row")
                    };
                    let plus = fifth(1.0);
                    let minus = fifth(-1.0);
                    for a in 0..4 {
                        for b in 0..4 {
                            for c in 0..4 {
                                for d in 0..4 {
                                    for e in 0..4 {
                                        let fd = (plus[a][b][c][d][e] - minus[a][b][c][d][e])
                                            / (2.0 * step);
                                        let actual = exact[a][b][c][d][e][axis];
                                        assert!(
                                            (actual - fd).abs() <= 2e-6 * (1.0 + fd.abs()),
                                            "event={event} slope={slope} axes={a},{b},{c},{d},{e},{axis}: exact={actual} FD={fd}"
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
}
