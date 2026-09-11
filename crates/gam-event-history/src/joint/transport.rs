//! Fixed innovation coordinates for coefficient-dependent OU paths. The OU
//! density and transport Jacobian cancel analytically, avoiding subtraction
//! of large rate scores at nearly static transitions. Genetics stay fixed:
//! their specified joint Gaussian law has no fitted coefficient coordinates.
use super::score::regression_score;
use super::*;

fn genes<S: JetField>(h: &JointHistory, coordinates: &[S], zero: &S) -> (Vec<S>, usize) {
    let mut missing = 0;
    let genes = h
        .genetics
        .iter()
        .map(|value| match value {
            Some(value) => zero.constant_like(*value),
            None => {
                let value = coordinates[missing].clone();
                missing += 1;
                value
            }
        })
        .collect();
    (genes, missing)
}

#[cfg(test)]
mod tests {
    use super::*;
    use rand::{SeedableRng, rngs::SmallRng};

    #[test]
    fn posterior_moments_follow_the_current_entry_and_drive_with_the_same_innovations() {
        let model = JointLikelihood::new(JointSpecification {
            signatures: 1,
            marks: vec![MarkKind::Once],
            baseline_columns: 1,
            drive_columns: 1,
            entry_columns: 0,
            measurements: vec![MeasurementFamily::BinaryProbit],
            genetic_mean: vec![],
            genetic_precision: Array2::zeros((0, 0)),
        })
        .unwrap();
        let history = JointHistory {
            times: vec![0.0, 0.5, 1.0],
            exposure: vec![0.0, 1.0, 0.0],
            events: vec![None; 3],
            initially_at_risk: vec![false],
            baseline_design: Array2::ones((3, 1)),
            drive_design: Array2::ones((2, 1)),
            entry_design: vec![],
            genetics: vec![],
            measurements: vec![MeasurementRecord {
                node: 1,
                channel: 0,
                value: Some(1.0),
                after_event: false,
            }],
        };
        // An observed binary channel keeps the general integration path.
        // Its zero slope makes the exact conditional state law N(0, OU).
        let mut theta = vec![0.0; model.layout.width];
        let mut rng = SmallRng::seed_from_u64(80219);
        let bank = model
            .integration(
                &theta,
                &history,
                &[0.0; 3],
                None,
                &IntegrationOptions {
                    samples: 4096,
                    ..IntegrationOptions::default()
                },
                &mut rng,
            )
            .unwrap();
        let accuracy = IntegrationAccuracy {
            log_standard_error: 0.05,
            moment_standard_error: 0.1,
            ..IntegrationAccuracy::default()
        };
        let anchor = bank.posterior(&theta, &[0.0; 3], &accuracy).unwrap();
        theta[model.layout.entry.start] = 20.0;
        theta[model.layout.drive.start] = 20.0;
        let moved = bank.posterior(&theta, &[0.0; 3], &accuracy).unwrap();
        assert_eq!(
            anchor.likelihood.log_marginal,
            moved.likelihood.log_marginal
        );
        assert_eq!(
            anchor.likelihood.effective_samples,
            moved.likelihood.effective_samples
        );
        for n in 0..3 {
            assert!((moved.mean[n] - anchor.mean[n] - 20.0).abs() < 2e-13);
            assert!(
                (moved.state_covariance[n][[0, 0]] - anchor.state_covariance[n][[0, 0]]).abs()
                    < 2e-13
            );
            assert!(anchor.mean[n].abs() < 4.0 * anchor.mean_standard_error[n]);
        }
    }
}

impl JointLikelihood {
    /// Invert the triangular OU map at the proposal anchor. Return its
    /// conditional path log density so q(path)/p(path|genes) can be retained
    /// as a fixed importance ratio in innovation coordinates.
    pub(super) fn path_innovations(
        &self,
        theta: &[f64],
        h: &JointHistory,
        path: &[f64],
    ) -> Result<(Vec<f64>, f64), EventHistoryError> {
        let (genes, missing) = genes(h, path, &0.0);
        let k = self.spec.signatures;
        let mut innovations = path.to_vec();
        let mut density = 0.0;
        let log_tau = (2.0 * std::f64::consts::PI).ln();
        let entry = self.entry_features(h);
        for axis in 0..k {
            let value = path[missing + axis]
                - self.mean(theta, self.layout.entry.start, &entry, &genes, axis);
            innovations[missing + axis] = value;
            density -= 0.5 * (value * value + log_tau);
        }
        for n in 1..h.times.len() {
            let dt = h.times[n] - h.times[n - 1];
            let columns = h.drive_design.row(n - 1).to_vec();
            for axis in 0..k {
                let decay = -emission::softplus(&theta[self.layout.rates.start + axis]) * dt;
                let phi = decay.exp();
                let weight = -decay.exp_m1();
                let variance = -(2.0 * decay).exp_m1();
                if variance <= 0.0 || !variance.is_finite() {
                    return Err(numerical("joint OU innovation variance is unresolved"));
                }
                let after =
                    path[missing + (n - 1) * k + axis] + self.jump(theta, h.events[n - 1], axis);
                let drive = self.mean(theta, self.layout.drive.start, &columns, &genes, axis);
                let value = (path[missing + n * k + axis] - (phi * after + weight * drive))
                    / variance.sqrt();
                innovations[missing + n * k + axis] = value;
                density -= 0.5 * (value * value + variance.ln() + log_tau);
            }
        }
        if !density.is_finite() || innovations.iter().any(|v| !v.is_finite()) {
            return Err(numerical(
                "joint innovation proposal is not numerically resolved",
            ));
        }
        Ok((innovations, density))
    }

    pub(super) fn transport_path<S: JetField>(
        &self,
        theta: &[S],
        h: &JointHistory,
        innovations: &[f64],
    ) -> Result<Vec<S>, EventHistoryError> {
        let mut path: Vec<_> = innovations
            .iter()
            .map(|&v| theta[0].constant_like(v))
            .collect();
        let (genes, missing) = genes(h, &path, &theta[0]);
        let k = self.spec.signatures;
        let entry = self.entry_features(h);
        for axis in 0..k {
            path[missing + axis] = add_real(
                &self.mean(theta, self.layout.entry.start, &entry, &genes, axis),
                innovations[missing + axis],
            );
        }
        for n in 1..h.times.len() {
            let dt = h.times[n] - h.times[n - 1];
            let columns = h.drive_design.row(n - 1).to_vec();
            for axis in 0..k {
                let decay = emission::softplus(&theta[self.layout.rates.start + axis]).scale(-dt);
                let phi = exp(&decay);
                let weight = emission::expm1(&decay).neg();
                let variance = emission::expm1(&decay.scale(2.0)).neg();
                if variance.value() <= 0.0 || !variance.value().is_finite() {
                    return Err(numerical("joint OU innovation variance is unresolved"));
                }
                let after = path[missing + (n - 1) * k + axis].add(&self.jump(
                    theta,
                    h.events[n - 1],
                    axis,
                ));
                let drive = self.mean(theta, self.layout.drive.start, &columns, &genes, axis);
                path[missing + n * k + axis] = phi.mul(&after).add(&weight.mul(&drive)).add(
                    &exp(&ln(&variance).scale(0.5)).scale(innovations[missing + n * k + axis]),
                );
            }
        }
        Ok(path)
    }

    /// Reverse the triangular state map in O(nodes * signatures) state
    /// storage. No nodes-by-coefficients Jacobian or autodiff tape is built.
    pub(super) fn innovation_score(
        &self,
        theta: &[f64],
        h: &JointHistory,
        innovations: &[f64],
        reference: &[f64],
    ) -> Result<JointPathScore, EventHistoryError> {
        let path = self.transport_path(theta, h, innovations)?;
        let (mut score, mut adjoint) =
            self.path_density_score(theta, h, &path, reference, false)?;
        let (genes, missing) = genes(h, &path, &0.0);
        let k = self.spec.signatures;
        for n in (1..h.times.len()).rev() {
            let dt = h.times[n] - h.times[n - 1];
            let columns = h.drive_design.row(n - 1).to_vec();
            for axis in 0..k {
                let raw = theta[self.layout.rates.start + axis];
                let decay = -emission::softplus(&raw) * dt;
                let phi = decay.exp();
                let weight = -decay.exp_m1();
                let sd = (-(2.0 * decay).exp_m1()).sqrt();
                let dphi = -dt * phi * (-emission::softplus(&(-raw))).exp();
                let after =
                    path[missing + (n - 1) * k + axis] + self.jump(theta, h.events[n - 1], axis);
                let drive = self.mean(theta, self.layout.drive.start, &columns, &genes, axis);
                let a = adjoint[n * k + axis];
                regression_score(
                    &mut score.coefficients,
                    self.layout.drive.start,
                    &columns,
                    &genes,
                    axis,
                    a * weight,
                );
                score.coefficients[self.layout.rates.start + axis] +=
                    a * dphi * (after - drive - phi * innovations[missing + n * k + axis] / sd);
                if let Some(range) = h.events[n - 1].and_then(|d| self.layout.jumps[d].as_ref()) {
                    score.coefficients[range.start + axis] += a * phi;
                }
                adjoint[(n - 1) * k + axis] += a * phi;
            }
        }
        let entry = self.entry_features(h);
        for axis in 0..k {
            regression_score(
                &mut score.coefficients,
                self.layout.entry.start,
                &entry,
                &genes,
                axis,
                adjoint[axis],
            );
        }
        if score.coefficients.iter().any(|v| !v.is_finite()) {
            return Err(numerical("non-finite innovation coefficient score"));
        }
        Ok(score)
    }
}
