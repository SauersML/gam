//! Error assessment for a downstream function of an entire shared reference
//! population. Delete-one-replicate evaluations retain dependence between all
//! subjects using that population; particles within a population are not iid
//! replicates. All meshes and random draws remain fixed during this assessment.
//!
//! A function is evaluated from a curve and that curve's pullback, so its
//! coefficient scores differentiate the remaining ratio of risk-weighted
//! activity to risk mass after each deletion instead of freezing the
//! normaliser. Every deletion pulls back through the R - 1 remaining
//! populations: R^2 reverse sweeps, none carrying a coefficient-sized tangent,
//! where forward pooled Jacobians would carry p tangents through each of 2R
//! sweeps.
use super::*;
use crate::scalar::{div, sqrt};
#[path = "reference_value.rs"]
mod value;

/// The coefficient gradient of `g' m + h' M` for one curve's log moments `m`
/// and log risk masses `M`, adjoints given on that curve's rows.
pub(in crate::joint) type ReferencePullback<'a> =
    dyn Fn(&[f64], &[f64]) -> Result<Vec<f64>, EventHistoryError> + 'a;

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
        .and_then(|n| n.checked_mul(std::mem::size_of::<f64>()))
        .and_then(|n| n.checked_add(std::mem::size_of::<JointReferenceEvolution<f64>>()))
        .ok_or_else(|| invalid("reference functional workspace size overflow"))
}

/// `sqrt(a^2 + b^2)` over the power of two at or above the larger magnitude, so
/// a finite norm never overflows through its squares and both scalings are
/// exact.
fn hypot<S: JetField>(a: &S, b: &S) -> S {
    let larger = a.value().abs().max(b.value().abs());
    if larger == 0.0 {
        return a.add(b);
    }
    // Clamped so the power and its reciprocal are both finite normals.
    let exponent = (larger.log2().ceil() as i32).clamp(f64::MIN_EXP - 1, f64::MAX_EXP - 1);
    let power = 2f64.powi(exponent);
    let (x, y) = (a.scale(power.recip()), b.scale(power.recip()));
    sqrt(&x.mul(&x).add(&y.mul(&y))).scale(power)
}

/// Streaming centered sums of squares stored as Euclidean norms, so finite
/// standard errors are not lost by overflowing an intermediate variance.
struct JackknifeMoments<S> {
    count: usize,
    mean: Vec<S>,
    centered_norm: Vec<S>,
}

impl<S: JetField> JackknifeMoments<S> {
    fn new(width: usize, like: &S) -> Self {
        Self {
            count: 0,
            mean: vec![like.constant_like(0.0); width],
            centered_norm: vec![like.constant_like(0.0); width],
        }
    }
    /// Welford's update with its factors formed over `S` from the exact count,
    /// so their rounding is charged. The first value is the mean and adds no
    /// centred square.
    fn add(&mut self, values: &[S]) {
        self.count += 1;
        if self.count == 1 {
            for (slot, value) in self.mean.iter_mut().zip(values) {
                *slot = value.clone();
            }
            return;
        }
        let count = self.count as f64;
        for (j, value) in values.iter().enumerate() {
            let delta = value.sub(&self.mean[j]);
            let total = delta.constant_like(count);
            self.mean[j] = self.mean[j].add(&div(&delta, &total));
            let shrink = sqrt(&div(&delta.constant_like(count - 1.0), &total));
            self.centered_norm[j] = hypot(&self.centered_norm[j], &delta.mul(&shrink));
        }
    }
    /// Replicate standard errors `((R-1)/R sum_r (F_r - mean)^2)^(1/2)` and
    /// biases `(R - 1)(mean - F)` for the full values `F`.
    fn deletions(&self, full: &[S]) -> (Vec<S>, Vec<S>) {
        let like = &full[0];
        let factor = sqrt(&div(
            &like.constant_like((self.count - 1) as f64),
            &like.constant_like(self.count as f64),
        ));
        (
            self.centered_norm.iter().map(|v| v.mul(&factor)).collect(),
            self.mean
                .iter()
                .zip(full)
                .map(|(m, v)| m.sub(v).scale((self.count - 1) as f64))
                .collect(),
        )
    }
}

impl JackknifeMoments<f64> {
    fn finish(self, full: FunctionalValue) -> Result<FunctionalPoint, EventHistoryError> {
        let (reference_standard_error, jackknife_bias) = self.deletions(&full.values);
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

impl ResolvedReference {
    fn functional_point<F>(
        &self,
        theta: &[f64],
        banks: &[JointReferenceBank],
        evaluate: &mut F,
    ) -> Result<FunctionalPoint, EventHistoryError>
    where
        F: FnMut(
            &JointReferenceEvolution<f64>,
            &ReferencePullback<'_>,
        ) -> Result<FunctionalValue, EventHistoryError>,
    {
        let first = banks[0].evolve(&self.model, theta)?;
        // Live curve-sized arrays: R replicate curves, the pooled and deleted
        // curves, and aggregate's 2R interpolated copies.
        let workspace = banks
            .len()
            .checked_mul(3)
            .and_then(|n| n.checked_add(2))
            .and_then(|n| n.checked_mul(curve_bytes(&first).ok()?))
            .ok_or_else(|| invalid("reference ensemble workspace size overflow"))?;
        let budget = materialization_budget();
        if workspace > budget {
            return Err(numerical(format!(
                "reference functional ensemble needs {workspace} bytes, above this machine's {budget}-byte materialization budget"
            )));
        }
        let mut curves = vec![first];
        for bank in &banks[1..] {
            curves.push(bank.evolve(&self.model, theta)?);
        }
        let times = curves[0].times.clone();
        let pooled = aggregate(&curves, &times, &curves[0].row_times)?.reference;
        let full_pullback =
            |g: &[f64], h: &[f64]| self.pooled_pullback(theta, banks, &curves, None, g, h);
        let full = evaluate(&pooled, &full_pullback)?;
        let width = full.values.len();
        full.validate(width)?;
        let mut jackknife = JackknifeMoments::new(width, &0.0);
        for removed in 0..banks.len() {
            let deleted = value::delete_population(&pooled, &curves[removed], banks.len())?;
            let deleted_pullback = |g: &[f64], h: &[f64]| {
                self.pooled_pullback(theta, banks, &curves, Some(removed), g, h)
            };
            let output = evaluate(&deleted, &deleted_pullback)?;
            output.validate(width)?;
            jackknife.add(&output.values);
        }
        jackknife.finish(full)
    }

    pub(in crate::joint) fn assess_functional<F>(
        &self,
        theta: &[f64],
        mut evaluate: F,
    ) -> Result<FunctionalAssessment, EventHistoryError>
    where
        F: FnMut(
            &JointReferenceEvolution<f64>,
            &ReferencePullback<'_>,
        ) -> Result<FunctionalValue, EventHistoryError>,
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
    use crate::joint::reference_tests::derivative_agrees;
    use crate::test_support::{Bound, agrees};
    use crate::scalar::Rows;

    /// A deletion's values equal direct re-pooling of the remaining
    /// populations within both routes' rounding, and its pullback equals jets
    /// through that re-pooling on every coefficient, including selection
    /// through the risk mass.
    #[test]
    fn population_deletion_values_and_pullbacks_match_direct_repooling() {
        let model = Arc::new(
            JointLikelihood::new(JointSpecification {
                signatures: 1,
                marks: vec![MarkKind::Recurrent, MarkKind::Once],
                baseline_columns: 1,
                drive_columns: 1,
                entry_columns: 0,
                measurements: vec![],
                genetic_mean: vec![0.0],
                genetic_precision: Array2::eye(1),
            })
            .unwrap(),
        );
        let jump = model.layout.jumps[0].as_ref().unwrap().start;
        let mut theta = vec![0.0; model.layout.width];
        theta[model.layout.baseline.start] = -1.0;
        theta[model.layout.baseline.start + 1] = -1.5;
        theta[model.layout.entry.start + 1] = 0.4;
        theta[jump] = 1.0;
        let profile = JointReferenceProfile {
            times: (0..=4).map(|n| n as f64 / 4.0).collect(),
            baseline_design: Array2::ones((5, 1)),
            drive_design: Array2::ones((4, 1)),
            entry_design: vec![],
            genetics: vec![None],
        };
        // One accepted round at declared resolution: no stopping tolerance.
        let options = ReferenceResolutionOptions {
            replicates: 5,
            initial_particles: 32,
            log_moment_tolerance: f64::MAX,
            risk_mass_tolerance: f64::MAX,
            sampling_error_rate: 1e-3,
        };
        let (resolved, _) =
            ResolvedReference::resolve(&model, &theta, &profile, &options, 41).unwrap();
        let banks = &resolved.large;
        let exact = |values: &[f64]| values.iter().map(|&v| Bound::exact(v)).collect::<Vec<_>>();
        let bounded = exact(&theta);
        let curves: Vec<_> = banks
            .iter()
            .map(|b| b.evolve(&model, &bounded).unwrap())
            .collect();
        let times = curves[0].times.clone();
        let pooled = aggregate(&curves, &times, &curves[0].row_times).unwrap().reference;
        let rows = pooled.log_moments.len();
        let g: Vec<f64> = (0..rows).map(|i| (0.41 * i as f64).sin()).collect();
        let h: Vec<f64> = (0..rows).map(|i| (0.23 * i as f64).cos()).collect();
        for removed in 0..banks.len() {
            let deleted = value::delete_population(&pooled, &curves[removed], banks.len()).unwrap();
            let retained: Vec<_> = banks
                .iter()
                .enumerate()
                .filter(|(r, _)| *r != removed)
                .map(|(_, b)| b.evolve(&model, &bounded).unwrap())
                .collect();
            let direct = aggregate(&retained, &times, &retained[0].row_times).unwrap().reference;
            for i in 0..rows {
                for (production, oracle, name) in [
                    (&deleted.log_moments[i], &direct.log_moments[i], "log moment"),
                    (&deleted.log_risk_mass[i], &direct.log_risk_mass[i], "log mass"),
                ] {
                    let name = format!("deletion {removed}, row {i}, {name}");
                    if oracle.value == 0.0 {
                        // The origin's log mass: re-pooling cancels exactly, and
                        // the deletion must land within the two routes' bounds.
                        assert!(
                            production.value.abs() <= production.bar(oracle),
                            "{name}: {}",
                            production.value
                        );
                    } else {
                        agrees(production, oracle, &name);
                    }
                }
            }
            let hand = resolved
                .pooled_pullback(&bounded, banks, &curves, Some(removed), &exact(&g), &exact(&h))
                .unwrap();
            for q in 0..theta.len() {
                let seeds: Vec<Rows<Bound, 1>> = theta
                    .iter()
                    .enumerate()
                    .map(|(j, &v)| Rows::seed(Bound::exact(v), [f64::from(j == q)]))
                    .collect();
                let retained: Vec<_> = banks
                    .iter()
                    .enumerate()
                    .filter(|(r, _)| *r != removed)
                    .map(|(_, b)| b.evolve(&model, &seeds).unwrap())
                    .collect();
                let jet = aggregate(&retained, &times, &retained[0].row_times).unwrap().reference;
                let directional = (0..rows).fold(Bound::exact(0.0), |acc, i| {
                    acc.add(&jet.log_moments[i].rows[0].scale(g[i]))
                        .add(&jet.log_risk_mass[i].rows[0].scale(h[i]))
                });
                derivative_agrees(
                    &hand[q],
                    &directional,
                    &format!("deletion {removed}, coefficient {q}"),
                );
            }
        }
    }

    /// The jackknife of a mean is its sample standard error with zero bias. The
    /// streaming route agrees with that closed form within both routes'
    /// rounding, and a power-of-two scale whose squares would overflow scales
    /// the standard error exactly, so a finite error is never lost.
    #[test]
    fn shared_reference_jackknife_matches_the_mean_closed_form_at_any_scale() {
        let values = [1.0, 2.0, 4.0, 8.0];
        let replicates = values.len() as f64;
        // Every factor is an exact integer, divided over the bounded scalar so
        // the rounding of a quotient such as 1/3 is charged.
        let run = |multiplier: f64| {
            let inputs: Vec<Bound> = values.iter().map(|&v| Bound::exact(v * multiplier)).collect();
            let full = div(
                &inputs.iter().fold(Bound::exact(0.0), |acc, v| acc.add(v)),
                &Bound::exact(replicates),
            );
            let mut moments = JackknifeMoments::new(1, &Bound::exact(0.0));
            for v in &inputs {
                // A deletion value of the mean: (R F - F_r) / (R - 1).
                let deleted = div(&full.scale(replicates).sub(v), &Bound::exact(replicates - 1.0));
                moments.add(std::slice::from_ref(&deleted));
            }
            let (se, bias) = moments.deletions(std::slice::from_ref(&full));
            (inputs, full, se[0], bias[0])
        };
        let (inputs, full, se, bias) = run(1.0);
        let squares = inputs.iter().fold(Bound::exact(0.0), |acc, v| {
            let d = v.sub(&full);
            acc.add(&d.mul(&d))
        });
        let expected = sqrt(&div(&squares, &Bound::exact(replicates * (replicates - 1.0))));
        agrees(&se, &expected, "jackknife standard error of a mean");
        assert!(
            bias.value.abs() <= bias.rounding(),
            "the jackknife bias of a mean is zero within its bound: {}",
            bias.value
        );
        let multiplier = 2.0_f64.powi(1000);
        let (_, _, large, _) = run(multiplier);
        assert_eq!(
            large.value,
            se.value * multiplier,
            "a finite standard error must survive squares that overflow"
        );
    }
}

