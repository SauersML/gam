//! Exact reference law without latent signatures. Baseline designs are linear
//! within each declared profile interval, so cumulative hazards are analytic.
use super::*;

/// log integral_0^1 exp(x u) du and its first four derivatives, x<=0.
/// Derivatives are cumulants of the exponentially tilted uniform law.
fn log_relative_exponential(x: f64) -> [f64; 5] {
    let a = -x;
    let mut integrals = [0.0; 5];
    let endpoint = x.exp();
    if a <= 2.0 * (integrals.len() - 1) as f64 {
        // Positive series: exp(-a) k! sum_j a^j/(k+j+1)!.
        // Beyond twice the largest moment order, the recurrence below
        // contracts by at least a factor two and avoids cancellation.
        for (k, integral) in integrals.iter_mut().enumerate() {
            let mut term = 1.0 / (k + 1) as f64;
            let mut sum = term;
            let mut j = 1;
            loop {
                term *= a / (k + j + 1) as f64;
                let next = sum + term;
                if next == sum {
                    break;
                }
                sum = next;
                j += 1;
            }
            *integral = endpoint * sum;
        }
        let mass = integrals[0];
        for integral in &mut integrals {
            *integral /= mass;
        }
    } else {
        // Recur on NORMALIZED moments. Raw integrals can underflow even
        // when their ratio (and hence a log-integral derivative) is finite.
        integrals[0] = 1.0;
        let edge = endpoint / -x.exp_m1();
        for k in 1..5 {
            integrals[k] = k as f64 * integrals[k - 1] / a - edge;
        }
    }
    let mean = integrals[1];
    let second = integrals[2];
    let third = integrals[3];
    let fourth = integrals[4];
    [
        gam_math::special::log_exprel(x),
        mean,
        second - mean * mean,
        third - 3.0 * mean * second + 2.0 * mean.powi(3),
        fourth - 4.0 * mean * third - 3.0 * second * second + 12.0 * mean * mean * second
            - 6.0 * mean.powi(4),
    ]
}

fn integrated_rate<S: JetField>(left: &S, right: &S, duration: f64) -> S {
    let (anchor, difference) = if right.value() >= left.value() {
        (right, left.sub(right))
    } else {
        (left, right.sub(left))
    };
    exp(&add_real(
        &anchor.add(&difference.compose_unary(log_relative_exponential(difference.value()))),
        duration.ln(),
    ))
}

impl JointReferenceBank<'_> {
    fn rank_zero_design(&self) -> (Vec<f64>, Array2<f64>) {
        let rows = 2 * self.profile.times.len() - 1;
        let times = (0..rows)
            .map(|i| {
                if i % 2 == 0 {
                    self.profile.times[i / 2]
                } else {
                    self.profile.times[i / 2]
                        + 0.5 * (self.profile.times[i / 2 + 1] - self.profile.times[i / 2])
                }
            })
            .collect();
        let design = Array2::from_shape_fn((rows, self.model.spec.baseline_columns), |(i, b)| {
            if i % 2 == 0 {
                self.profile.baseline_design[[i / 2, b]]
            } else {
                0.5 * self.profile.baseline_design[[i / 2, b]]
                    + 0.5 * self.profile.baseline_design[[i / 2 + 1, b]]
            }
        });
        (times, design)
    }

    pub(super) fn rank_zero_evolution<S: JetField>(
        &self,
        theta: &[S],
    ) -> Result<JointReferenceEvolution<S>, EventHistoryError> {
        let (times, design) = self.rank_zero_design();
        let marks = self.model.spec.marks.len();
        let columns = self.model.spec.baseline_columns;
        let zero = theta[0].constant_like(0.0);
        let mut cumulative = vec![zero.clone(); marks];
        let mut log_risk_mass = vec![zero.clone(); times.len() * marks];
        for n in 1..times.len() {
            for d in 0..marks {
                let mut left = zero.clone();
                let mut right = zero.clone();
                for b in 0..columns {
                    let coefficient = &theta[self.model.layout.baseline.start + d * columns + b];
                    left = left.add(&coefficient.scale(design[[n - 1, b]]));
                    right = right.add(&coefficient.scale(design[[n, b]]));
                }
                cumulative[d] =
                    cumulative[d].add(&integrated_rate(&left, &right, times[n] - times[n - 1]));
            }
            let mut terminal = zero.clone();
            for (d, kind) in self.model.spec.marks.iter().enumerate() {
                if *kind == MarkKind::Terminal {
                    terminal = terminal.add(&cumulative[d]);
                }
            }
            for d in 0..marks {
                log_risk_mass[n * marks + d] = if self.model.spec.marks[d] == MarkKind::Once {
                    terminal.add(&cumulative[d]).neg()
                } else {
                    terminal.neg()
                };
            }
        }
        if log_risk_mass.iter().any(|v| !v.value().is_finite()) {
            return Err(numerical(
                "analytic reference cumulative hazard is not representable",
            ));
        }
        Ok(JointReferenceEvolution {
            theta: theta.to_vec(),
            log_moments: vec![zero; times.len() * marks],
            times,
            log_risk_mass,
            marks,
            diagnostics: ReferenceDiagnostics {
                maximum_log_moment_standard_error: 0.0,
                minimum_risk_effective_samples: self.particles as f64,
                maximum_step_hazard: 0.0,
            },
        })
    }

    pub(super) fn rank_zero_mass_jacobian(
        &self,
        theta: &[f64],
    ) -> Result<Array2<f64>, EventHistoryError> {
        let (times, design) = self.rank_zero_design();
        let marks = self.model.spec.marks.len();
        let columns = self.model.spec.baseline_columns;
        let mut cumulative = Array2::<f64>::zeros((marks, columns));
        let mut jacobian = Array2::zeros((times.len() * marks, theta.len()));
        for n in 1..times.len() {
            for d in 0..marks {
                let mut left = 0.0;
                let mut right = 0.0;
                for b in 0..columns {
                    let coefficient = theta[self.model.layout.baseline.start + d * columns + b];
                    left += coefficient * design[[n - 1, b]];
                    right += coefficient * design[[n, b]];
                }
                let integral = integrated_rate(&left, &right, times[n] - times[n - 1]);
                let tilted = log_relative_exponential(-(right - left).abs())[1];
                let mean = if right >= left { 1.0 - tilted } else { tilted };
                for b in 0..columns {
                    cumulative[[d, b]] +=
                        integral * ((1.0 - mean) * design[[n - 1, b]] + mean * design[[n, b]]);
                }
            }
            for d in 0..marks {
                for e in 0..marks {
                    if self.model.spec.marks[e] == MarkKind::Terminal
                        || (e == d && self.model.spec.marks[d] == MarkKind::Once)
                    {
                        for b in 0..columns {
                            jacobian[[
                                n * marks + d,
                                self.model.layout.baseline.start + e * columns + b,
                            ]] = -cumulative[[e, b]];
                        }
                    }
                }
            }
        }
        if jacobian.iter().any(|v| !v.is_finite()) {
            return Err(numerical(
                "analytic reference mass derivative is not representable",
            ));
        }
        Ok(jacobian)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::scalar::Mixed;
    use rand::{SeedableRng, rngs::SmallRng};

    #[test]
    fn analytic_rate_integrals_match_independent_quadrature_through_four_derivatives() {
        let (nodes, weights) = gam_math::special::gauss_legendre(129);
        for slope in [
            -100.0, -30.0, -8.00001, -8.0, -0.5, -1e-9, 0.0, 0.5, 8.0, 100.0,
        ] {
            let beta = Mixed::seed(Mixed::seed(slope, 1.0, 1.0), 1.0, 1.0);
            let zero = beta.constant_like(0.0);
            let actual = integrated_rate(&zero, &beta, 1.0);
            let oracle = nodes.iter().zip(&weights).fold(zero, |sum, (&x, &w)| {
                sum.add(&exp(&beta.scale(0.5 * (x + 1.0))).scale(0.5 * w))
            });
            for (a, b) in [
                (actual.base.base, oracle.base.base),
                (actual.base.u, oracle.base.u),
                (actual.base.uv, oracle.base.uv),
                (actual.u.uv, oracle.u.uv),
                (actual.uv.uv, oracle.uv.uv),
            ] {
                assert!(
                    (a - b).abs() < 1e-10 * b.abs().max(1e-12),
                    "slope {slope}: {a} vs {b}"
                );
            }
        }
        let actual = integrated_rate(&-800.0, &800.0, 1e-200);
        let exact = (800.0 + 1e-200_f64.ln() - 1600.0_f64.ln()).exp();
        assert!((actual / exact - 1.0).abs() < 1e-12);
        assert_eq!(log_relative_exponential(-1e200)[1], 1e-200);
    }

    #[test]
    fn rank_zero_reference_preserves_competing_risk_survival_and_total_derivatives() {
        let model = JointLikelihood::new(JointSpecification {
            signatures: 0,
            marks: vec![
                MarkKind::Once,
                MarkKind::Terminal,
                MarkKind::Terminal,
                MarkKind::Recurrent,
            ],
            baseline_columns: 2,
            drive_columns: 1,
            entry_columns: 0,
            measurements: vec![],
            genetic_mean: vec![],
            genetic_precision: Array2::zeros((0, 0)),
        })
        .unwrap();
        let theta = [-1.2, 0.3, -2.5, -0.2, -3.0, 0.1, -0.5, 0.0];
        let profile = JointReferenceProfile {
            times: vec![0.0, 6.0],
            baseline_design: ndarray::arr2(&[[1.0, 0.0], [1.0, 6.0]]),
            drive_design: Array2::ones((1, 1)),
            entry_design: vec![],
            genetics: vec![],
        };
        let mut rng = SmallRng::seed_from_u64(8271);
        let options = ReferenceOptions {
            particles: 4,
            ..ReferenceOptions::default()
        };
        let accuracy = ReferenceAccuracy {
            minimum_risk_effective_samples: 2.0,
            maximum_step_hazard: 1e-6,
            ..ReferenceAccuracy::default()
        };
        let bank = model
            .reference_bank(&theta, &profile, &options, &accuracy, &mut rng)
            .unwrap();
        let curve = bank.evolve(&theta, &accuracy).unwrap();
        let fine = model
            .reference_bank(
                &theta,
                &profile.refined().unwrap(),
                &options,
                &accuracy,
                &mut rng,
            )
            .unwrap()
            .evolve(&theta, &accuracy)
            .unwrap();
        let (nodes, weights) = gam_math::special::gauss_legendre(129);
        for (n, &time) in curve.times.iter().enumerate() {
            let hazards: Vec<f64> = (0..4)
                .map(|d| {
                    nodes
                        .iter()
                        .zip(&weights)
                        .map(|(&x, &w)| {
                            0.5 * time
                                * w
                                * (theta[2 * d] + theta[2 * d + 1] * 0.5 * time * (x + 1.0)).exp()
                        })
                        .sum()
                })
                .collect();
            for d in 0..4 {
                let expected = -hazards[1] - hazards[2] - if d == 0 { hazards[0] } else { 0.0 };
                assert!((curve.log_risk_mass[n * 4 + d] - expected).abs() < 1e-12);
                assert!(
                    (curve.log_risk_mass[n * 4 + d] - fine.log_risk_mass[2 * n * 4 + d]).abs()
                        < 1e-12
                );
                assert_eq!(curve.log_moments[n * 4 + d], 0.0);
            }
        }
        let sensitivity = bank.sensitivity(&theta, &accuracy, 32 << 20).unwrap();
        for q in 0..theta.len() {
            let seeded: Vec<Mixed<f64>> = theta
                .iter()
                .enumerate()
                .map(|(j, &v)| Mixed::seed(v, f64::from(q == j), f64::from(q == j)))
                .collect();
            let jet = bank.evolve(&seeded, &accuracy).unwrap();
            for row in 0..curve.log_risk_mass.len() {
                assert!(
                    (sensitivity.log_risk_mass_jacobian()[[row, q]] - jet.log_risk_mass[row].u)
                        .abs()
                        < 1e-11
                );
            }
        }
    }
}
