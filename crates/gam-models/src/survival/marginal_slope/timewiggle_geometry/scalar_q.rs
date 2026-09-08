//! Scalar time-wiggle coefficient map shared by production and its isolated MSI benchmark.

use gam_math::nested_dual::{Dual2, JetField};
use ndarray::ArrayView1;

/// One row of each time-wiggle basis derivative from `B` through `B'''''`.
///
/// Keeping the six rows separate avoids allocating a per-row array of stacks
/// in the hot path. The generic scalar constructor consumes `[B..B'''']` for
/// `q` and the shifted `[B'..B''''']` stack for `dq/dh`.
pub(crate) struct TimewiggleBasisDerivativeRows<'a> {
    basis: ArrayView1<'a, f64>,
    basis_d1: ArrayView1<'a, f64>,
    basis_d2: ArrayView1<'a, f64>,
    basis_d3: ArrayView1<'a, f64>,
    basis_d4: ArrayView1<'a, f64>,
    basis_d5: ArrayView1<'a, f64>,
}

impl<'a> TimewiggleBasisDerivativeRows<'a> {
    pub(crate) fn new(
        basis: ArrayView1<'a, f64>,
        basis_d1: ArrayView1<'a, f64>,
        basis_d2: ArrayView1<'a, f64>,
        basis_d3: ArrayView1<'a, f64>,
        basis_d4: ArrayView1<'a, f64>,
        basis_d5: ArrayView1<'a, f64>,
    ) -> Self {
        Self {
            basis,
            basis_d1,
            basis_d2,
            basis_d3,
            basis_d4,
            basis_d5,
        }
    }

    fn validate_width(&self, width: usize, endpoint: &str) -> Result<(), String> {
        let widths = [
            self.basis.len(),
            self.basis_d1.len(),
            self.basis_d2.len(),
            self.basis_d3.len(),
            self.basis_d4.len(),
            self.basis_d5.len(),
        ];
        if widths.iter().any(|&actual| actual != width) {
            return Err(format!(
                "survival marginal-slope {endpoint} timewiggle B..B5 widths {widths:?} must all equal scalar coefficient width {width}"
            ));
        }
        Ok(())
    }

    #[inline]
    fn basis_stack(&self, coefficient: usize) -> [f64; 5] {
        [
            self.basis[coefficient],
            self.basis_d1[coefficient],
            self.basis_d2[coefficient],
            self.basis_d3[coefficient],
            self.basis_d4[coefficient],
        ]
    }

    #[inline]
    fn derivative_stack(&self, coefficient: usize) -> [f64; 5] {
        [
            self.basis_d1[coefficient],
            self.basis_d2[coefficient],
            self.basis_d3[coefficient],
            self.basis_d4[coefficient],
            self.basis_d5[coefficient],
        ]
    }
}

/// Bitwise current values anchoring the generic time-wiggle scalar program.
/// Replacing only the real value channels preserves the existing f64
/// dot-product result exactly while leaving all derivative channels intact.
#[derive(Clone, Copy)]
pub(crate) struct TimewiggleQBaseValues {
    pub(crate) q0: f64,
    pub(crate) q1: f64,
    pub(crate) dq1_dh1: f64,
}

/// Generic time-wiggle row coordinates.
pub(crate) struct TimewiggleScalarQ<J> {
    pub(crate) q0: J,
    pub(crate) q1: J,
    pub(crate) qd1: J,
}

/// Wiggle coefficients are coordinates held fixed under family differentiation.
/// Their coefficient-space jets remain live, so mixed coefficient/family
/// derivatives enter through the product with the composed basis.
pub(crate) trait TimewiggleScalar: JetField {
    type Coefficient;
    fn weighted_basis(&self, coefficient: &Self::Coefficient, derivatives: [f64; 5]) -> Self;
}

impl TimewiggleScalar for f64 {
    type Coefficient = f64;

    #[inline]
    fn weighted_basis(&self, coefficient: &f64, derivatives: [f64; 5]) -> Self {
        coefficient * derivatives[0]
    }
}

impl<S: JetField> TimewiggleScalar for Dual2<S> {
    type Coefficient = S;

    #[inline]
    fn weighted_basis(&self, coefficient: &S, derivatives: [f64; 5]) -> Self {
        let basis = self.compose_unary(derivatives);
        Self {
            v: coefficient.mul(&basis.v),
            g: coefficient.mul(&basis.g),
            h: coefficient.mul(&basis.h),
        }
    }
}

/// Add the identity endpoint to the weighted basis with live coefficient jets.
fn timewiggle_q_at_endpoint<J: TimewiggleScalar>(
    h: &J,
    beta_w: &[J::Coefficient],
    derivative_stack: impl Fn(usize) -> [f64; 5],
) -> J {
    beta_w
        .iter()
        .enumerate()
        .fold(h.clone(), |sum, (column, beta)| {
            sum.add(&h.weighted_basis(beta, derivative_stack(column)))
        })
}

/// Evaluate `q0 = h0 + B(h0) beta_w`, `q1 = h1 + B(h1) beta_w`, and
/// `qd1 = (1 + B'(h1) beta_w) d_raw` over one scalar algebra.
///
/// `Dual2<Order2<_>>` and `Dual2<OneSeed<_>>` carry baseline-family directions;
/// their inner scalar carries every beta-wiggle primary channel. The supplied
/// base values are the current f64 path's already-computed results; centering
/// makes those values bit-identical without changing any derivative.
pub(crate) fn timewiggle_q_from_basis_derivative_rows<J: TimewiggleScalar>(
    h0: &J,
    h1: &J,
    d_raw: &J,
    beta_w: &[J::Coefficient],
    entry_basis: &TimewiggleBasisDerivativeRows<'_>,
    exit_basis: &TimewiggleBasisDerivativeRows<'_>,
    base_values: TimewiggleQBaseValues,
) -> Result<TimewiggleScalarQ<J>, String> {
    entry_basis.validate_width(beta_w.len(), "entry")?;
    exit_basis.validate_width(beta_w.len(), "exit")?;

    let q0 = timewiggle_q_at_endpoint(h0, beta_w, |column| entry_basis.basis_stack(column))
        .with_value(base_values.q0);
    let mut q1 = h1.clone();
    let mut dq1_dh1 = h1.constant_like(1.0);
    for (column, beta) in beta_w.iter().enumerate() {
        let value_stack = exit_basis.basis_stack(column);
        let slope_stack = exit_basis.derivative_stack(column);
        let value_term = h1.weighted_basis(beta, value_stack);
        // Equal supplied stacks are the same local polynomial over the same
        // input jet. Reuse the complete product, including live beta channels.
        let slope_term = if slope_stack == value_stack {
            value_term.clone()
        } else {
            h1.weighted_basis(beta, slope_stack)
        };
        q1 = q1.add(&value_term);
        dq1_dh1 = dq1_dh1.add(&slope_term);
    }
    let q1 = q1.with_value(base_values.q1);
    let dq1_dh1 = dq1_dh1.with_value(base_values.dq1_dh1);

    Ok(TimewiggleScalarQ {
        q0,
        q1,
        qd1: dq1_dh1.mul(d_raw),
    })
}

#[cfg(test)]
mod generic_scalar_q_tests {
    use super::*;
    use gam_math::jet_scalar::{JetScalar, OneSeed, Order2};
    use gam_math::nested_dual::Dual2;
    use ndarray::Array1;

    // Two analytic basis functions, B0(h)=h^2 and B1(h)=exp(h), represented by
    // their exact B..B5 rows. This isolates the scalar algebra from spline
    // construction and contains no finite-difference oracle.
    fn analytic_basis_derivatives(h: f64) -> [Array1<f64>; 6] {
        let exponential = h.exp();
        [
            Array1::from_vec(vec![h * h, exponential]),
            Array1::from_vec(vec![2.0 * h, exponential]),
            Array1::from_vec(vec![2.0, exponential]),
            Array1::from_vec(vec![0.0, exponential]),
            Array1::from_vec(vec![0.0, exponential]),
            Array1::from_vec(vec![0.0, exponential]),
        ]
    }

    fn derivative_rows(derivatives: &[Array1<f64>; 6]) -> TimewiggleBasisDerivativeRows<'_> {
        TimewiggleBasisDerivativeRows::new(
            derivatives[0].view(),
            derivatives[1].view(),
            derivatives[2].view(),
            derivatives[3].view(),
            derivatives[4].view(),
            derivatives[5].view(),
        )
    }

    fn analytic_base_values(h0: f64, h1: f64, beta: [f64; 2]) -> TimewiggleQBaseValues {
        TimewiggleQBaseValues {
            q0: h0 + beta[0] * h0 * h0 + beta[1] * h0.exp(),
            q1: h1 + beta[0] * h1 * h1 + beta[1] * h1.exp(),
            dq1_dh1: 1.0 + 2.0 * beta[0] * h1 + beta[1] * h1.exp(),
        }
    }

    fn analytic_q<const K: usize, J: JetScalar<K>>(
        h0: &J,
        h1: &J,
        d_raw: &J,
        beta: &[J; 2],
    ) -> TimewiggleScalarQ<J> {
        let entry_exponential = h0.exp();
        let exit_exponential = h1.exp();
        let q0 = h0
            .add(&beta[0].mul(&h0.mul(h0)))
            .add(&beta[1].mul(&entry_exponential));
        let q1 = h1
            .add(&beta[0].mul(&h1.mul(h1)))
            .add(&beta[1].mul(&exit_exponential));
        let dq1_dh1 = J::constant(1.0)
            .add(&beta[0].mul(&h1.scale(2.0)))
            .add(&beta[1].mul(&exit_exponential));
        TimewiggleScalarQ {
            q0,
            q1,
            qd1: dq1_dh1.mul(d_raw),
        }
    }

    fn assert_close(actual: f64, expected: f64, channel: &str) {
        let tolerance = 1024.0 * f64::EPSILON * (1.0 + actual.abs().max(expected.abs()));
        assert!(
            (actual - expected).abs() <= tolerance,
            "{channel}: actual={actual:.17e}, expected={expected:.17e}, tolerance={tolerance:.3e}"
        );
    }

    fn assert_order2<const K: usize>(actual: &Order2<K>, expected: &Order2<K>, prefix: &str) {
        assert_close(actual.value(), expected.value(), &format!("{prefix}.value"));
        for a in 0..K {
            assert_close(actual.g()[a], expected.g()[a], &format!("{prefix}.g[{a}]"));
            for b in 0..K {
                assert_close(
                    actual.h()[a][b],
                    expected.h()[a][b],
                    &format!("{prefix}.h[{a}][{b}]"),
                );
            }
        }
    }

    fn assert_dual_order2<const K: usize>(
        actual: &Dual2<Order2<K>>,
        expected: &Dual2<Order2<K>>,
        prefix: &str,
    ) {
        assert_order2(&actual.v, &expected.v, &format!("{prefix}.v"));
        assert_order2(&actual.g, &expected.g, &format!("{prefix}.g"));
        assert_order2(&actual.h, &expected.h, &format!("{prefix}.h"));
    }

    fn assert_oneseed<const K: usize>(actual: &OneSeed<K>, expected: &OneSeed<K>, prefix: &str) {
        assert_order2(&actual.base, &expected.base, &format!("{prefix}.base"));
        assert_order2(&actual.eps, &expected.eps, &format!("{prefix}.eps"));
    }

    fn assert_dual_oneseed<const K: usize>(
        actual: &Dual2<OneSeed<K>>,
        expected: &Dual2<OneSeed<K>>,
        prefix: &str,
    ) {
        assert_oneseed(&actual.v, &expected.v, &format!("{prefix}.v"));
        assert_oneseed(&actual.g, &expected.g, &format!("{prefix}.g"));
        assert_oneseed(&actual.h, &expected.h, &format!("{prefix}.h"));
    }

    fn order2_family_variable<const K: usize>(
        value: f64,
        axis: usize,
        family_first: f64,
        family_second: f64,
    ) -> Dual2<Order2<K>> {
        Dual2 {
            v: Order2::variable(value, axis),
            g: Order2::constant(family_first),
            h: Order2::constant(family_second),
        }
    }

    fn oneseed_family_variable<const K: usize>(
        value: f64,
        axis: usize,
        direction: f64,
        family_first: f64,
        family_second: f64,
    ) -> Dual2<OneSeed<K>> {
        Dual2 {
            v: OneSeed::seed_direction(value, axis, direction),
            g: OneSeed::constant(family_first),
            h: OneSeed::constant(family_second),
        }
    }

    #[test]
    fn f64_timewiggle_q_preserves_current_value_bits() {
        let values = [0.35, -0.4, 1.2, 0.18, 0.07];
        let beta = [values[3], values[4]];
        let entry_derivatives = analytic_basis_derivatives(values[0]);
        let exit_derivatives = analytic_basis_derivatives(values[1]);
        let entry_rows = derivative_rows(&entry_derivatives);
        let exit_rows = derivative_rows(&exit_derivatives);
        let base_values = analytic_base_values(values[0], values[1], beta);

        let actual = timewiggle_q_from_basis_derivative_rows(
            &values[0],
            &values[1],
            &values[2],
            &beta,
            &entry_rows,
            &exit_rows,
            base_values,
        )
        .expect("analytic B..B5 rows have matching widths");

        assert_eq!(actual.q0.to_bits(), base_values.q0.to_bits());
        assert_eq!(actual.q1.to_bits(), base_values.q1.to_bits());
        assert_eq!(
            actual.qd1.to_bits(),
            (base_values.dq1_dh1 * values[2]).to_bits()
        );
    }

    #[test]
    fn dual2_order2_timewiggle_q_matches_analytic_scalar_program() {
        const K: usize = 5;
        let values = [0.35, -0.4, 1.2, 0.18, 0.07];
        let h0 = order2_family_variable(values[0], 0, 0.11, -0.03);
        let h1 = order2_family_variable(values[1], 1, -0.08, 0.02);
        let d_raw = order2_family_variable(values[2], 2, 0.05, -0.01);
        let beta = [
            <Dual2<Order2<K>> as JetScalar<K>>::variable(values[3], 3),
            <Dual2<Order2<K>> as JetScalar<K>>::variable(values[4], 4),
        ];
        let entry_derivatives = analytic_basis_derivatives(values[0]);
        let exit_derivatives = analytic_basis_derivatives(values[1]);
        let entry_rows = derivative_rows(&entry_derivatives);
        let exit_rows = derivative_rows(&exit_derivatives);

        let actual = timewiggle_q_from_basis_derivative_rows(
            &h0,
            &h1,
            &d_raw,
            &beta.map(|coefficient| coefficient.v),
            &entry_rows,
            &exit_rows,
            analytic_base_values(values[0], values[1], [values[3], values[4]]),
        )
        .expect("analytic B..B5 rows have matching widths");
        let expected = analytic_q(&h0, &h1, &d_raw, &beta);

        assert_dual_order2(&actual.q0, &expected.q0, "q0");
        assert_dual_order2(&actual.q1, &expected.q1, "q1");
        assert_dual_order2(&actual.qd1, &expected.qd1, "qd1");
    }

    #[test]
    fn dual2_oneseed_timewiggle_q_matches_analytic_family_hessian_drift() {
        const K: usize = 5;
        let values = [0.35, -0.4, 1.2, 0.18, 0.07];
        let direction = [0.3, -0.2, 0.15, -0.4, 0.25];
        let h0 = oneseed_family_variable::<K>(values[0], 0, direction[0], 0.11, -0.03);
        let h1 = oneseed_family_variable::<K>(values[1], 1, direction[1], -0.08, 0.02);
        let d_raw = oneseed_family_variable::<K>(values[2], 2, direction[2], 0.05, -0.01);
        let beta = [
            oneseed_family_variable::<K>(values[3], 3, direction[3], 0.0, 0.0),
            oneseed_family_variable::<K>(values[4], 4, direction[4], 0.0, 0.0),
        ];
        let entry_derivatives = analytic_basis_derivatives(values[0]);
        let exit_derivatives = analytic_basis_derivatives(values[1]);
        let entry_rows = derivative_rows(&entry_derivatives);
        let exit_rows = derivative_rows(&exit_derivatives);

        let actual = timewiggle_q_from_basis_derivative_rows(
            &h0,
            &h1,
            &d_raw,
            &beta.map(|coefficient| coefficient.v),
            &entry_rows,
            &exit_rows,
            analytic_base_values(values[0], values[1], [values[3], values[4]]),
        )
        .expect("analytic B..B5 rows have matching widths");
        let expected = analytic_q(&h0, &h1, &d_raw, &beta);

        assert_dual_oneseed(&actual.q0, &expected.q0, "q0");
        assert_dual_oneseed(&actual.q1, &expected.q1, "q1");
        assert_dual_oneseed(&actual.qd1, &expected.qd1, "qd1");
    }

    fn measure_analytic_basis_program<S: JetScalar<5>>(
        make: impl Fn(f64, usize, f64, f64, f64) -> Dual2<S>,
        compare: impl Fn(&Dual2<S>, &Dual2<S>, &str),
        sum: impl Fn(&S) -> f64,
    ) -> gam_math::paired_timing::PairedTiming {
        let rows: Vec<_> = (0..64)
            .map(|row| {
                let t = row as f64 / 63.0;
                let values = [
                    0.15 + 0.4 * t,
                    -0.6 + 0.35 * t,
                    0.9 + 0.5 * t,
                    -0.12 + 0.3 * t,
                    0.04 + 0.09 * t,
                ];
                let directions = [0.3, -0.2, 0.15, -0.4, 0.25];
                let first = [0.11, -0.08, 0.05, 0.0, 0.0];
                let second = [-0.03, 0.02, -0.01, 0.0, 0.0];
                let inputs: [Dual2<S>; 5] = std::array::from_fn(|axis| {
                    make(
                        values[axis],
                        axis,
                        directions[axis],
                        first[axis],
                        second[axis],
                    )
                });
                (
                    inputs,
                    analytic_basis_derivatives(values[0]),
                    analytic_basis_derivatives(values[1]),
                    analytic_base_values(values[0], values[1], [values[3], values[4]]),
                )
            })
            .collect();
        let production = |row: usize| {
            let (inputs, entry, exit, base) = std::hint::black_box(&rows[row]);
            timewiggle_q_from_basis_derivative_rows(
                &inputs[0],
                &inputs[1],
                &inputs[2],
                &[inputs[3].v, inputs[4].v],
                &derivative_rows(entry),
                &derivative_rows(exit),
                *base,
            )
            .expect("valid timewiggle benchmark row")
        };
        let reference = |row: usize| {
            let (inputs, _, _, _) = std::hint::black_box(&rows[row]);
            analytic_q(&inputs[0], &inputs[1], &inputs[2], &[inputs[3], inputs[4]])
        };
        for row in 0..rows.len() {
            let actual = production(row);
            let expected = reference(row);
            compare(&actual.q0, &expected.q0, "q0");
            compare(&actual.q1, &expected.q1, "q1");
            compare(&actual.qd1, &expected.qd1, "qd1");
        }
        let consume = |q: TimewiggleScalarQ<Dual2<S>>| {
            [&q.q0, &q.q1, &q.qd1]
                .into_iter()
                .map(|jet| sum(&jet.v) + sum(&jet.g) + sum(&jet.h))
                .sum::<f64>()
        };
        gam_math::paired_timing::paired_interleaved(
            15,
            128,
            0x9320_71AE,
            |nudge| {
                let offset = nudge.to_bits() as usize;
                (0usize..64)
                    .map(|row| consume(production(row.wrapping_add(offset) & 63)))
                    .sum()
            },
            |nudge| {
                let offset = nudge.to_bits() as usize;
                (0usize..64)
                    .map(|row| consume(reference(row.wrapping_add(offset) & 63)))
                    .sum()
            },
        )
    }

    fn sum_order2(jet: &Order2<5>) -> f64 {
        jet.value() + jet.g().iter().sum::<f64>() + jet.h().iter().flatten().sum::<f64>()
    }

    #[test]
    fn timewiggle_live_coefficient_channels_match_polynomials_at_all_widths_932() {
        fn dense(value: f64) -> Order2<5> {
            Order2(gam_math::jet_tower::Tower2 {
                v: value,
                g: std::array::from_fn(|i| 0.03 * (i + 1) as f64 + 0.02 * value),
                h: std::array::from_fn(|i| {
                    std::array::from_fn(|j| 0.01 / (i + j + 1) as f64 + 0.003 * value)
                }),
            })
        }
        let seed = |value| OneSeed {
            base: dense(value),
            eps: dense(0.07 + 0.03 * value),
        };
        let family_seed = |value| Dual2 {
            v: seed(value),
            g: seed(0.1 - 0.02 * value),
            h: seed(-0.03 + 0.01 * value),
        };
        let h0 = family_seed(0.35);
        let h1 = family_seed(-0.4);
        let raw = family_seed(1.2);
        let polynomial = |h: &Dual2<OneSeed<5>>, coefficients: &[f64]| {
            coefficients
                .iter()
                .rev()
                .fold(h.constant_like(0.0), |value, &coefficient| {
                    value.mul(h).add(&h.constant_like(coefficient))
                })
        };
        for width in [0, 1, 2, 7, 32] {
            let coefficients: Vec<[f64; 5]> = (0..width)
                .map(|column| {
                    std::array::from_fn(|degree| {
                        let sign = if (column + degree) % 2 == 0 {
                            1.0
                        } else {
                            -1.0
                        };
                        sign * 0.03 * (degree + 1) as f64 / (column + 1) as f64
                    })
                })
                .collect();
            let basis = |h: f64| -> [Array1<f64>; 6] {
                std::array::from_fn(|order| {
                    Array1::from_iter(coefficients.iter().map(|c| {
                        (order..5)
                            .map(|degree| {
                                let factor =
                                    (0..order).map(|j| (degree - j) as f64).product::<f64>();
                                c[degree] * factor * h.powi((degree - order) as i32)
                            })
                            .sum()
                    }))
                })
            };
            let beta: Vec<_> = (0..width).map(|j| seed(0.1 / (j + 1) as f64)).collect();
            let mut q0 = h0;
            let mut q1 = h1;
            let mut slope = h1.constant_like(1.0);
            for (coefficient, c) in beta.iter().zip(&coefficients) {
                let outer_constant = Dual2 {
                    v: *coefficient,
                    g: coefficient.constant_like(0.0),
                    h: coefficient.constant_like(0.0),
                };
                q0 = q0.add(&outer_constant.mul(&polynomial(&h0, c)));
                q1 = q1.add(&outer_constant.mul(&polynomial(&h1, c)));
                let derivative: [f64; 4] = std::array::from_fn(|i| (i + 1) as f64 * c[i + 1]);
                slope = slope.add(&outer_constant.mul(&polynomial(&h1, &derivative)));
            }
            let entry = basis(h0.value());
            let exit = basis(h1.value());
            let actual = timewiggle_q_from_basis_derivative_rows(
                &h0,
                &h1,
                &raw,
                &beta,
                &derivative_rows(&entry),
                &derivative_rows(&exit),
                TimewiggleQBaseValues {
                    q0: q0.value(),
                    q1: q1.value(),
                    dq1_dh1: slope.value(),
                },
            )
            .expect("matching polynomial basis widths");
            assert_dual_oneseed(&actual.q0, &q0, &format!("width={width} q0"));
            assert_dual_oneseed(&actual.q1, &q1, &format!("width={width} q1"));
            assert_dual_oneseed(&actual.qd1, &slope.mul(&raw), &format!("width={width} qd1"));
        }
    }

    #[test]
    fn release_timewiggle_q_vs_analytic_basis_program_932() {
        if cfg!(debug_assertions) {
            return;
        }
        // This is the analytic polynomial/exp opponent from the July
        // reopening. It is distinct from the additional requirement for a
        // fully hand-expanded runtime-width spline derivative schedule.
        let mut gate = gam_math::paired_timing::SpeedGate::open("TIMEWIGGLE-Q-932");
        let second = measure_analytic_basis_program(
            |value, axis, _, first, second| order2_family_variable(value, axis, first, second),
            assert_dual_order2,
            sum_order2,
        );
        gate.faster(
            "Dual2<Order2<5>> width=2",
            &second,
            "production",
            "analytic_basis",
        );
        let third =
            measure_analytic_basis_program(oneseed_family_variable, assert_dual_oneseed, |jet| {
                sum_order2(&jet.base) + sum_order2(&jet.eps)
            });
        gate.faster(
            "Dual2<OneSeed<5>> width=2",
            &third,
            "production",
            "analytic_basis",
        );
        gate.finish();
    }
}
