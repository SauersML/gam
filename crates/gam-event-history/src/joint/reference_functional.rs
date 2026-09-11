//! Error assessment for a downstream function of an entire shared reference
//! population. Delete-one-replicate evaluations retain dependence between all
//! subjects using that population; particles within a population are not iid
//! replicates. All meshes and random draws remain fixed during this assessment.
use super::*;
#[path = "reference_value.rs"]
mod value;

pub(in crate::joint) struct FunctionalValue {
    pub values: Vec<f64>,
    pub conditional_standard_error: Vec<f64>,
}

pub(in crate::joint) struct FunctionalPoint {
    pub value: FunctionalValue,
    pub reference_standard_error: Vec<f64>,
    pub jackknife_bias: Vec<f64>,
}

pub(in crate::joint) struct FunctionalAssessment {
    /// Coarse time, fine time, and fine time with more particles.
    pub points: [FunctionalPoint; 3],
}

impl FunctionalValue {
    fn validate(&self, width: usize) -> Result<(), EventHistoryError> {
        if width == 0
            || self.values.len() != width
            || self.conditional_standard_error.len() != width
            || self.values.iter().any(|v| !v.is_finite())
            || self
                .conditional_standard_error
                .iter()
                .any(|&v| !v.is_finite() || v < 0.0)
        {
            return Err(numerical(
                "invalid shared-reference functional value or conditional error",
            ));
        }
        Ok(())
    }
}

fn curve_bytes(curve: &JointReferenceEvolution<f64>) -> Result<usize, EventHistoryError> {
    curve
        .theta
        .len()
        .checked_add(curve.times.len())
        .and_then(|n| n.checked_add(curve.log_moments.len()))
        .and_then(|n| n.checked_add(curve.log_risk_mass.len()))
        .and_then(|n| n.checked_mul(8))
        .and_then(|n| n.checked_add(std::mem::size_of::<JointReferenceEvolution<f64>>()))
        .ok_or_else(|| invalid("reference functional workspace size overflow"))
}

fn sensitivity_bytes(value: &JointReferenceSensitivity) -> Result<usize, EventHistoryError> {
    value
        .log_moment_jacobian()
        .len()
        .checked_add(value.log_risk_mass_jacobian().len())
        .and_then(|n| n.checked_mul(8))
        .and_then(|n| n.checked_add(curve_bytes(value.reference()).ok()?))
        .ok_or_else(|| invalid("reference functional Jacobian workspace size overflow"))
}

/// Subtract one independent population from the pooled ratio. Replaying the
/// remaining R-1 populations for every deletion would cost R^2 evaluations.
/// This uses a second sequential pass over R sensitivities, with no R x N x P
/// Jacobian storage. A numerically dominant replicate makes deletion unresolved.
fn leave_one_out(
    pooled: &JointReferenceSensitivity,
    removed: &JointReferenceSensitivity,
    replicates: usize,
    workspace_limit: usize,
) -> Result<JointReferenceSensitivity, EventHistoryError> {
    let full = pooled.reference();
    let part = removed.reference();
    if replicates < 2
        || full.times != part.times
        || full.theta != part.theta
        || full.marks != part.marks
        || pooled.log_moment_jacobian().dim() != removed.log_moment_jacobian().dim()
    {
        return Err(invalid(
            "reference deletion requires matching population states and grids",
        ));
    }
    if sensitivity_bytes(pooled)? > workspace_limit {
        return Err(numerical(
            "reference deletion exceeds its derivative workspace budget",
        ));
    }
    let count = (replicates as f64).ln();
    let remaining_count = ((replicates - 1) as f64).ln();
    let mut reference = JointReferenceEvolution {
        theta: full.theta.clone(),
        times: full.times.clone(),
        marks: full.marks,
        log_moments: Vec::with_capacity(full.log_moments.len()),
        log_risk_mass: Vec::with_capacity(full.log_risk_mass.len()),
        // These are conservative within-population diagnostics from the
        // original ensemble, not a successful resolution report for a deletion.
        diagnostics: full.diagnostics.clone(),
    };
    let mut moment = Array2::zeros(pooled.log_moment_jacobian().dim());
    let mut mass = Array2::zeros(pooled.log_risk_mass_jacobian().dim());
    for n in 0..full.log_moments.len() {
        let lm = part.log_risk_mass[n] - full.log_risk_mass[n] - count;
        let la = lm + part.log_moments[n] - full.log_moments[n];
        let mass_remainder = -lm.exp_m1();
        let activity_remainder = -la.exp_m1();
        if !(mass_remainder > 0.0 && activity_remainder > 0.0)
            || !mass_remainder.is_finite()
            || !activity_remainder.is_finite()
        {
            return Err(numerical(
                "shared-reference jackknife unresolved: one replicate dominates the pooled risk mass or activity",
            ));
        }
        let log_mass_remainder = mass_remainder.ln();
        let log_activity_remainder = activity_remainder.ln();
        reference
            .log_risk_mass
            .push(full.log_risk_mass[n] + count - remaining_count + log_mass_remainder);
        reference
            .log_moments
            .push(full.log_moments[n] + log_activity_remainder - log_mass_remainder);
        let mass_ratio = (lm - log_mass_remainder).exp();
        let activity_ratio = (la - log_activity_remainder).exp();
        for q in 0..moment.ncols() {
            let dm =
                pooled.log_risk_mass_jacobian()[[n, q]] - removed.log_risk_mass_jacobian()[[n, q]];
            let da =
                dm + pooled.log_moment_jacobian()[[n, q]] - removed.log_moment_jacobian()[[n, q]];
            mass[[n, q]] = pooled.log_risk_mass_jacobian()[[n, q]] + mass_ratio * dm;
            moment[[n, q]] =
                pooled.log_moment_jacobian()[[n, q]] + activity_ratio * da - mass_ratio * dm;
        }
    }
    if reference
        .log_moments
        .iter()
        .chain(&reference.log_risk_mass)
        .chain(moment.iter())
        .chain(mass.iter())
        .any(|v| !v.is_finite())
    {
        return Err(numerical(
            "non-finite shared-reference deletion value or sensitivity",
        ));
    }
    Ok(JointReferenceSensitivity::pooled(
        reference,
        moment,
        mass,
        workspace_limit,
    ))
}

/// Streaming centered sums of squares stored as Euclidean norms, so finite
/// standard errors are not lost by overflowing an intermediate variance.
struct JackknifeMoments {
    count: usize,
    mean: Vec<f64>,
    centered_norm: Vec<f64>,
}

impl JackknifeMoments {
    fn new(width: usize) -> Self {
        Self {
            count: 0,
            mean: vec![0.0; width],
            centered_norm: vec![0.0; width],
        }
    }
    fn add(&mut self, values: &[f64]) {
        self.count += 1;
        let count = self.count as f64;
        for (j, &value) in values.iter().enumerate() {
            let delta = value - self.mean[j];
            self.mean[j] += delta / count;
            self.centered_norm[j] =
                self.centered_norm[j].hypot(delta * ((count - 1.0) / count).sqrt());
        }
    }
    fn finish(self, full: FunctionalValue) -> Result<FunctionalPoint, EventHistoryError> {
        let factor = ((self.count - 1) as f64 / self.count as f64).sqrt();
        let reference_standard_error: Vec<_> =
            self.centered_norm.iter().map(|v| v * factor).collect();
        let jackknife_bias: Vec<_> = self
            .mean
            .iter()
            .zip(&full.values)
            .map(|(m, v)| (self.count - 1) as f64 * (m - v))
            .collect();
        if reference_standard_error
            .iter()
            .chain(&jackknife_bias)
            .any(|v| !v.is_finite())
        {
            return Err(numerical(
                "shared-reference functional jackknife is not representable",
            ));
        }
        Ok(FunctionalPoint {
            value: full,
            reference_standard_error,
            jackknife_bias,
        })
    }
}

impl ResolvedReference<'_> {
    fn functional_point<F>(
        &self,
        theta: &[f64],
        banks: &[JointReferenceBank<'_>],
        evaluate: &mut F,
    ) -> Result<FunctionalPoint, EventHistoryError>
    where
        F: FnMut(&JointReferenceSensitivity) -> Result<FunctionalValue, EventHistoryError>,
    {
        let first = banks[0].evolve_for_resolution(theta, &self.step_limits)?;
        // Aggregate temporarily copies moment/mass arrays for each replicate.
        let curve_workspace = banks
            .len()
            .checked_mul(3)
            .and_then(|n| n.checked_add(4))
            .and_then(|n| n.checked_mul(curve_bytes(&first).ok()?))
            .ok_or_else(|| invalid("reference ensemble workspace size overflow"))?;
        if curve_workspace > self.options.memory_limit_bytes {
            return Err(numerical(
                "reference functional ensemble exceeds its workspace budget",
            ));
        }
        let mut curves = vec![first];
        for bank in &banks[1..] {
            curves.push(bank.evolve_for_resolution(theta, &self.step_limits)?);
        }
        let reference = aggregate(&curves, curves[0].times())?.reference;
        drop(curves);
        let pooled = self.pooled_sensitivity(theta, banks, reference)?;
        let value = evaluate(&pooled)?;
        let width = value.values.len();
        value.validate(width)?;
        let mut jackknife = JackknifeMoments::new(width);
        let held_bytes = sensitivity_bytes(&pooled)?;
        let remainder_budget = self
            .options
            .memory_limit_bytes
            .checked_sub(held_bytes)
            .and_then(|n| n.checked_sub(held_bytes))
            .ok_or_else(|| {
                numerical("reference functional deletions exceed the derivative workspace budget")
            })?;
        for bank in banks {
            let removed =
                bank.sensitivity_for_resolution(theta, &self.step_limits, remainder_budget)?;
            let deletion_budget = self
                .options
                .memory_limit_bytes
                .checked_sub(held_bytes)
                .and_then(|n| n.checked_sub(sensitivity_bytes(&removed).ok()?))
                .ok_or_else(|| {
                    numerical("reference deletion exceeds the derivative workspace budget")
                })?;
            let deleted = leave_one_out(&pooled, &removed, banks.len(), deletion_budget)?;
            drop(removed);
            let output = evaluate(&deleted)?;
            output.validate(width)?;
            jackknife.add(&output.values);
        }
        jackknife.finish(value)
    }

    pub(in crate::joint) fn assess_functional<F>(
        &self,
        theta: &[f64],
        mut evaluate: F,
    ) -> Result<FunctionalAssessment, EventHistoryError>
    where
        F: FnMut(&JointReferenceSensitivity) -> Result<FunctionalValue, EventHistoryError>,
    {
        Ok(FunctionalAssessment {
            points: [
                self.functional_point(theta, &self.coarse, &mut evaluate)?,
                self.functional_point(theta, &self.fine, &mut evaluate)?,
                self.functional_point(theta, &self.large, &mut evaluate)?,
            ],
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::scalar::Mixed;

    #[test]
    fn population_deletion_matches_direct_repooling_including_sensitivities() {
        let curves: Vec<_> = (0..5)
            .map(|r| JointReferenceEvolution {
                theta: vec![Mixed::seed(0.0, 1.0, 0.0)],
                times: vec![0.0, 1.0],
                marks: 1,
                log_moments: vec![
                    Mixed::seed(0.0, 0.0, 0.0),
                    Mixed::seed((1.0 + r as f64).ln(), 0.2 * r as f64, 0.0),
                ],
                log_risk_mass: vec![
                    Mixed::seed(0.0, 0.0, 0.0),
                    Mixed::seed(-0.3 * r as f64, -0.4 * r as f64, 0.0),
                ],
                diagnostics: ReferenceDiagnostics {
                    maximum_log_moment_standard_error: 0.0,
                    minimum_risk_effective_samples: 1000.0,
                    maximum_step_hazard: 0.01,
                },
            })
            .collect();
        let convert = |curve: JointReferenceEvolution<Mixed<f64>>| {
            let moment =
                Array2::from_shape_vec((2, 1), curve.log_moments.iter().map(|v| v.u).collect())
                    .unwrap();
            let mass =
                Array2::from_shape_vec((2, 1), curve.log_risk_mass.iter().map(|v| v.u).collect())
                    .unwrap();
            JointReferenceSensitivity::pooled(
                JointReferenceEvolution {
                    theta: vec![0.0],
                    times: curve.times,
                    marks: 1,
                    log_moments: curve.log_moments.iter().map(|v| v.base).collect(),
                    log_risk_mass: curve.log_risk_mass.iter().map(|v| v.base).collect(),
                    diagnostics: curve.diagnostics,
                },
                moment,
                mass,
                1 << 20,
            )
        };
        let combined = convert(aggregate(&curves, &[0.0, 1.0]).unwrap().reference);
        for r in 0..curves.len() {
            let curve = &curves[r];
            let single = convert(JointReferenceEvolution {
                theta: curve.theta.clone(),
                times: curve.times.clone(),
                marks: curve.marks,
                log_moments: curve.log_moments.clone(),
                log_risk_mass: curve.log_risk_mass.clone(),
                diagnostics: curve.diagnostics.clone(),
            });
            let deleted = leave_one_out(&combined, &single, curves.len(), 1 << 20).unwrap();
            let values =
                value::delete_population(combined.reference(), single.reference(), curves.len())
                    .unwrap();
            assert_eq!(values.log_moments, deleted.reference().log_moments);
            assert_eq!(values.log_risk_mass, deleted.reference().log_risk_mass);
            let retained: Vec<_> = curves
                .iter()
                .enumerate()
                .filter(|(j, _)| *j != r)
                .map(|(_, v)| JointReferenceEvolution {
                    theta: v.theta.clone(),
                    times: v.times.clone(),
                    marks: v.marks,
                    log_moments: v.log_moments.clone(),
                    log_risk_mass: v.log_risk_mass.clone(),
                    diagnostics: v.diagnostics.clone(),
                })
                .collect();
            let expected = convert(aggregate(&retained, &[0.0, 1.0]).unwrap().reference);
            for (a, b) in deleted
                .reference()
                .log_moments()
                .iter()
                .zip(expected.reference().log_moments())
                .chain(
                    deleted
                        .reference()
                        .log_risk_mass()
                        .iter()
                        .zip(expected.reference().log_risk_mass()),
                )
                .chain(
                    deleted
                        .log_moment_jacobian()
                        .iter()
                        .zip(expected.log_moment_jacobian()),
                )
                .chain(
                    deleted
                        .log_risk_mass_jacobian()
                        .iter()
                        .zip(expected.log_risk_mass_jacobian()),
                )
            {
                assert!((a - b).abs() < 1e-13, "{a} vs {b}");
            }
            assert!(leave_one_out(&combined, &single, curves.len(), 1).is_err());
        }
    }

    #[test]
    fn shared_reference_jackknife_scales_with_cohort_size_and_keeps_large_errors() {
        let values = [1.0, 2.0, 4.0, 8.0];
        let mean = values.iter().sum::<f64>() / values.len() as f64;
        for multiplier in [1.0, 100.0, 1e200] {
            let mut moments = JackknifeMoments::new(1);
            for value in &values {
                moments.add(&[multiplier * (4.0 * mean - value) / 3.0]);
            }
            let result = moments
                .finish(FunctionalValue {
                    values: vec![multiplier * mean],
                    conditional_standard_error: vec![0.0],
                })
                .unwrap();
            let expected = (values.iter().map(|v| (v - mean).powi(2)).sum::<f64>() / 12.0).sqrt();
            assert!((result.reference_standard_error[0] / multiplier - expected).abs() < 1e-13);
            assert!(result.jackknife_bias[0].abs() / multiplier < 1e-13);
        }
    }
}
