//! Analytic complete-path coefficient scores, with a separate reference
//! adjoint so the reference evolution's parameter dependence is never lost.
use super::*;
use ndarray::ArrayView2;

pub struct JointPathScore {
    pub log_density: f64,
    pub(super) coefficients: Vec<f64>,
    pub(super) reference: Vec<f64>,
}

impl JointPathScore {
    /// Apply the reference Jacobian (node-major marks by coefficients).
    /// This must be the Jacobian of the moments supplied to log_density_score,
    /// evaluated at the same coefficients. Zero is valid only for a reference
    /// that is mathematically independent of those parameters.
    pub fn pullback(
        &self,
        reference_jacobian: ArrayView2<'_, f64>,
    ) -> Result<Vec<f64>, EventHistoryError> {
        if reference_jacobian.dim() != (self.reference.len(), self.coefficients.len())
            || reference_jacobian.iter().any(|v| !v.is_finite())
        {
            return Err(invalid(
                "reference Jacobian has invalid dimensions or non-finite entries",
            ));
        }
        let mut score = self.coefficients.clone();
        for (row, &adjoint) in reference_jacobian.rows().into_iter().zip(&self.reference) {
            for (out, &derivative) in score.iter_mut().zip(row) {
                *out += adjoint * derivative;
            }
        }
        if score.iter().any(|v| !v.is_finite()) {
            return Err(numerical("non-finite normalized coefficient score"));
        }
        Ok(score)
    }
}

pub(super) fn regression_score(
    out: &mut [f64],
    start: usize,
    columns: &[f64],
    genes: &[f64],
    axis: usize,
    score: f64,
) {
    for (j, &feature) in columns.iter().enumerate() {
        let base = start + (axis * columns.len() + j) * (genes.len() + 1);
        out[base] += score * feature;
        for (g, &gene) in genes.iter().enumerate() {
            out[base + g + 1] += score * feature * gene;
        }
    }
}

impl JointLikelihood {
    /// Value and analytic coefficient/reference scores at a fixed latent
    /// path. Latent integration uses these as complete-data scores. This is
    /// not a score obtained by holding a parameter-dependent reference fixed:
    /// the returned object requires its reference Jacobian for a pullback.
    pub fn log_density_score(
        &self,
        theta: &[f64],
        h: &JointHistory,
        path: &[f64],
        reference: &[f64],
    ) -> Result<JointPathScore, EventHistoryError> {
        self.path_density_score(theta, h, path, reference, true)
            .map(|v| v.0)
    }

    /// The optional state adjoints exclude the OU density, whose normalized
    /// innovation law cancels its transport Jacobian analytically.
    pub(super) fn path_density_score(
        &self,
        theta: &[f64],
        h: &JointHistory,
        path: &[f64],
        reference: &[f64],
        dynamics: bool,
    ) -> Result<(JointPathScore, Vec<f64>), EventHistoryError> {
        let log_density = self.path_density(theta, h, path, reference, dynamics)?;
        let k = self.spec.signatures;
        let marks = self.spec.marks.len();
        let mut missing = 0;
        let genes: Vec<_> = h
            .genetics
            .iter()
            .map(|gene| match gene {
                Some(g) => *g,
                None => {
                    let g = path[missing];
                    missing += 1;
                    g
                }
            })
            .collect();
        let state = |n: usize| &path[missing + n * k..missing + (n + 1) * k];
        let mut coefficients = vec![0.0; theta.len()];
        let mut reference_score = vec![0.0; reference.len()];
        let mut state_score = if dynamics {
            vec![]
        } else {
            vec![0.0; h.times.len() * k]
        };
        if dynamics {
            let entry = self.entry_features(h);
            for axis in 0..k {
                let mean = self.mean(theta, self.layout.entry.start, &entry, &genes, axis);
                regression_score(
                    &mut coefficients,
                    self.layout.entry.start,
                    &entry,
                    &genes,
                    axis,
                    state(0)[axis] - mean,
                );
            }
            for n in 1..h.times.len() {
                let dt = h.times[n] - h.times[n - 1];
                let columns = h.drive_design.row(n - 1).to_vec();
                for axis in 0..k {
                    let raw = theta[self.layout.rates.start + axis];
                    let rate = emission::softplus(&raw);
                    let phi = (-rate * dt).exp();
                    let weight = -(-rate * dt).exp_m1();
                    let variance = -(-2.0 * rate * dt).exp_m1();
                    let drive = self.mean(theta, self.layout.drive.start, &columns, &genes, axis);
                    let after = state(n - 1)[axis] + self.jump(theta, h.events[n - 1], axis);
                    let residual = state(n)[axis] - (phi * after + weight * drive);
                    let mean_score = residual / variance;
                    regression_score(
                        &mut coefficients,
                        self.layout.drive.start,
                        &columns,
                        &genes,
                        axis,
                        mean_score * weight,
                    );
                    let dphi = -dt * phi * (-emission::softplus(&(-raw))).exp();
                    coefficients[self.layout.rates.start + axis] += dphi
                        * ((after - drive) * mean_score
                            - phi * (residual * mean_score - 1.0) / variance);
                    if let Some(range) = h.events[n - 1].and_then(|d| self.layout.jumps[d].as_ref())
                    {
                        coefficients[range.start + axis] += mean_score * phi;
                    }
                }
            }
        }
        let decoder = super::decoder::PreparedDecoder::new(self, theta);
        let mut risk = h.initially_at_risk.clone();
        for n in 0..h.times.len() {
            for d in 0..marks {
                if !risk[d] {
                    continue;
                }
                let start = self.layout.decoder.start + d * k;
                let weights = decoder.weights(d);
                let log_activity = decoder.activity(d, state(n));
                let baseline: f64 = (0..self.spec.baseline_columns)
                    .map(|b| {
                        theta[self.layout.baseline.start + d * self.spec.baseline_columns + b]
                            * h.baseline_design[[n, b]]
                    })
                    .sum();
                let log_rate = baseline + log_activity - reference[n * marks + d];
                let exposure = if h.exposure[n] == 0.0 {
                    0.0
                } else {
                    h.exposure[n] * log_rate.exp()
                };
                let residual = f64::from(h.events[n] == Some(d)) - exposure;
                reference_score[n * marks + d] = -residual;
                for b in 0..self.spec.baseline_columns {
                    coefficients
                        [self.layout.baseline.start + d * self.spec.baseline_columns + b] +=
                        residual * h.baseline_design[[n, b]];
                }
                for axis in 0..k {
                    coefficients[start + axis] += residual
                        * ((weights[axis + 1] + emission::log_softplus(&state(n)[axis])
                            - log_activity)
                            .exp()
                            - weights[axis + 1].exp());
                    if !dynamics {
                        state_score[n * k + axis] += residual
                            * (weights[axis + 1]
                                - emission::softplus(&(-state(n)[axis]))
                                - log_activity)
                                .exp();
                    }
                }
            }
            if let Some(d) = h.events[n] {
                if self.spec.marks[d] == MarkKind::Once {
                    risk[d] = false;
                }
            }
        }
        for record in &h.measurements {
            let Some(y) = record.value else {
                continue;
            };
            let location = self.layout.measurement_location[record.channel].clone();
            let x: Vec<_> = (0..k)
                .map(|axis| {
                    state(record.node)[axis]
                        + if record.after_event {
                            self.jump(theta, h.events[record.node], axis)
                        } else {
                            0.0
                        }
                })
                .collect();
            let mut eta = theta[location.start];
            for axis in 0..k {
                eta += theta[location.start + axis + 1] * x[axis];
            }
            let shape = self.layout.measurement_shape[record.channel].clone();
            let (score, shape_score) = emission::parameter_scores(
                &self.spec.measurements[record.channel],
                y,
                eta,
                &theta[shape.clone()],
            )?;
            coefficients[location.start] += score;
            for axis in 0..k {
                coefficients[location.start + axis + 1] += score * x[axis];
                if !dynamics {
                    state_score[record.node * k + axis] += score * theta[location.start + axis + 1];
                }
                if record.after_event {
                    if let Some(range) =
                        h.events[record.node].and_then(|d| self.layout.jumps[d].as_ref())
                    {
                        coefficients[range.start + axis] +=
                            score * theta[location.start + axis + 1];
                    }
                }
            }
            for (index, score) in shape.zip(shape_score) {
                coefficients[index] += score;
            }
        }
        if coefficients
            .iter()
            .chain(&reference_score)
            .chain(&state_score)
            .any(|v| !v.is_finite())
        {
            return Err(numerical("non-finite complete-path coefficient score"));
        }
        Ok((
            JointPathScore {
                log_density,
                coefficients,
                reference: reference_score,
            },
            state_score,
        ))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use gam_math::jet_scalar::Order1;
    use rand::{SeedableRng, rngs::SmallRng};

    fn fixture() -> (
        JointLikelihood,
        JointHistory,
        Vec<f64>,
        Vec<f64>,
        Vec<f64>,
        Array2<f64>,
    ) {
        let model = JointLikelihood::new(JointSpecification {
            signatures: 2,
            marks: vec![MarkKind::Recurrent, MarkKind::Once, MarkKind::Terminal],
            baseline_columns: 2,
            drive_columns: 2,
            entry_columns: 1,
            measurements: vec![
                MeasurementFamily::StudentT,
                MeasurementFamily::BinaryProbit,
                MeasurementFamily::OrdinalProbit { categories: 4 },
                MeasurementFamily::NegativeBinomial,
            ],
            genetic_mean: vec![0.1, -0.2],
            genetic_precision: Array2::eye(2),
        })
        .unwrap();
        let times = vec![0.0, 0.2, 0.3, 0.5, 0.65, 0.8, 1.0];
        let h = JointHistory {
            baseline_design: Array2::from_shape_fn(
                (7, 2),
                |(n, b)| if b == 0 { 1.0 } else { times[n] },
            ),
            drive_design: Array2::from_shape_fn(
                (6, 2),
                |(n, b)| if b == 0 { 1.0 } else { times[n] },
            ),
            times,
            exposure: vec![0.0, 0.25, 0.0, 0.4, 0.0, 0.35, 0.0],
            events: vec![None, None, Some(0), None, Some(1), None, Some(2)],
            initially_at_risk: vec![true; 3],
            entry_design: vec![0.6],
            genetics: vec![None, Some(0.6)],
            measurements: (0..4)
                .flat_map(|channel| {
                    [false, true].map(move |after_event| MeasurementRecord {
                        node: 2,
                        channel,
                        value: Some(if channel == 2 { 2.0 } else { 1.0 }),
                        after_event,
                    })
                })
                .collect(),
        };
        let mut theta: Vec<_> = (0..model.layout.width)
            .map(|j| 0.03 * (j as f64).sin())
            .collect();
        for d in 0..3 {
            theta[2 * d] = -1.2;
        }
        let path: Vec<_> = (0..model.latent_dimension(&h).unwrap())
            .map(|j| 0.2 * (j as f64).cos())
            .collect();
        let reference: Vec<_> = (0..21).map(|j| 0.02 * (j as f64).cos()).collect();
        let jacobian =
            Array2::from_shape_fn((21, theta.len()), |(n, p)| 0.01 * ((n + p) as f64).sin());
        (model, h, theta, path, reference, jacobian)
    }

    fn batched_score(
        model: &JointLikelihood,
        h: &JointHistory,
        theta: &[f64],
        path: &[f64],
        reference: &[f64],
        jacobian: &Array2<f64>,
    ) -> Vec<f64> {
        let mut gradient = vec![0.0; theta.len()];
        let paths: Vec<_> = path
            .iter()
            .map(|&v| Order1::<8> { v, g: [0.0; 8] })
            .collect();
        for start in (0..theta.len()).step_by(8) {
            let coefficients: Vec<_> = theta
                .iter()
                .enumerate()
                .map(|(j, &v)| {
                    let mut g = [0.0; 8];
                    if j >= start && j < start + 8 {
                        g[j - start] = 1.0;
                    }
                    Order1::<8> { v, g }
                })
                .collect();
            let moments: Vec<_> = reference
                .iter()
                .enumerate()
                .map(|(n, &v)| {
                    let mut g = [0.0; 8];
                    for axis in 0..8 {
                        if start + axis < theta.len() {
                            g[axis] = jacobian[[n, start + axis]];
                        }
                    }
                    Order1::<8> { v, g }
                })
                .collect();
            let result = model
                .log_density(&coefficients, h, &paths, &moments)
                .unwrap();
            for j in start..(start + 8).min(theta.len()) {
                gradient[j] = result.g[j - start];
            }
        }
        gradient
    }

    #[test]
    fn innovation_transport_preserves_anchor_weights_and_total_scores_away_from_the_anchor() {
        let (model, h, anchor, path, reference, jacobian) = fixture();
        let (innovations, dynamics) = model.path_innovations(&anchor, &h, &path).unwrap();
        let restored = model.transport_path(&anchor, &h, &innovations).unwrap();
        for (a, b) in restored.iter().zip(&path) {
            assert!((a - b).abs() < 1e-13);
        }
        let complete = model.log_density(&anchor, &h, &path, &reference).unwrap();
        let transformed = model
            .path_density(&anchor, &h, &restored, &reference, false)
            .unwrap()
            + dynamics;
        assert!((complete - transformed).abs() < 1e-12);
        for static_limit in [false, true] {
            let mut theta: Vec<_> = anchor
                .iter()
                .enumerate()
                .map(|(i, v)| v + 0.4 * (i as f64).cos())
                .collect();
            if static_limit {
                for index in model.layout.rates.clone() {
                    theta[index] = -35.0;
                }
            }
            let score = model
                .innovation_score(&theta, &h, &innovations, &reference)
                .unwrap()
                .pullback(jacobian.view())
                .unwrap();
            for j in 0..theta.len() {
                let coefficients: Vec<_> = theta
                    .iter()
                    .enumerate()
                    .map(|(i, &v)| Order1::<1> {
                        v,
                        g: [f64::from(i == j)],
                    })
                    .collect();
                let moments: Vec<_> = reference
                    .iter()
                    .enumerate()
                    .map(|(i, &v)| Order1::<1> {
                        v,
                        g: [jacobian[[i, j]]],
                    })
                    .collect();
                let current_path = model
                    .transport_path(&coefficients, &h, &innovations)
                    .unwrap();
                let jet = model
                    .path_density(&coefficients, &h, &current_path, &moments, false)
                    .unwrap();
                assert!(
                    (score[j] - jet.g[0]).abs() < 2e-10 * (1.0 + score[j].abs()),
                    "coordinate {j}, static={static_limit}: {} vs {}",
                    score[j],
                    jet.g[0]
                );
            }
        }
    }

    #[test]
    fn full_path_score_includes_shapes_genetic_drive_jumps_and_reference() {
        let (model, h, theta, path, reference, jacobian) = fixture();
        let result = model
            .log_density_score(&theta, &h, &path, &reference)
            .unwrap();
        assert_eq!(
            result.log_density,
            model.log_density(&theta, &h, &path, &reference).unwrap()
        );
        let actual = result.pullback(jacobian.view()).unwrap();
        let expected = batched_score(&model, &h, &theta, &path, &reference, &jacobian);
        for j in 0..theta.len() {
            assert!(
                (actual[j] - expected[j]).abs() < 2e-10 * (1.0 + expected[j].abs()),
                "coordinate {j}: {} vs {}",
                actual[j],
                expected[j]
            );
        }
        assert!(result.pullback(Array2::zeros((1, 1)).view()).is_err());
        let mut invalid = jacobian.clone();
        invalid[[0, 0]] = f64::NAN;
        assert!(result.pullback(invalid.view()).is_err());
        for (family, y, eta, shape) in [
            (MeasurementFamily::StudentT, 1e150, 0.0, vec![0.0, 0.2]),
            (MeasurementFamily::NegativeBinomial, 3.0, 1.0, vec![-700.0]),
            (
                MeasurementFamily::OrdinalProbit { categories: 4 },
                3.0,
                -40.0,
                vec![0.4, -0.5],
            ),
        ] {
            let (_, score) = emission::parameter_scores(&family, y, eta, &shape).unwrap();
            for j in 0..shape.len() {
                let shapes: Vec<_> = shape
                    .iter()
                    .enumerate()
                    .map(|(q, &v)| Order1::<1> {
                        v,
                        g: [f64::from(q == j)],
                    })
                    .collect();
                let exact =
                    emission::log_density(&family, y, &Order1::<1> { v: eta, g: [0.0] }, &shapes)
                        .unwrap();
                assert!(
                    (score[j] - exact.g[0]).abs() < 2e-9 * (1.0 + score[j].abs()),
                    "{family:?}: {} vs {}",
                    score[j],
                    exact.g[0]
                );
            }
        }
    }

    #[test]
    fn integrated_analytic_score_matches_the_fixed_importance_objective() {
        let (model, h, theta, _, reference, jacobian) = fixture();
        let mut rng = SmallRng::seed_from_u64(941);
        let bank = model
            .integration(
                &theta,
                &h,
                &reference,
                None,
                &IntegrationOptions {
                    samples: 1024,
                    ..IntegrationOptions::default()
                },
                &mut rng,
            )
            .unwrap();
        let accuracy = IntegrationAccuracy {
            log_standard_error: 0.2,
            minimum_effective_samples: 16.0,
            ..IntegrationAccuracy::default()
        };
        assert_eq!(
            bank.reference_difference_error(&theta, &reference, &reference, &accuracy)
                .unwrap(),
            0.0
        );
        let shifted: Vec<_> = reference.iter().map(|v| v + 0.01).collect();
        let paired = bank
            .reference_difference_error(&theta, &reference, &shifted, &accuracy)
            .unwrap();
        let separate = bank
            .log_marginal(&theta, &reference, &accuracy)
            .unwrap()
            .log_standard_error
            + bank
                .log_marginal(&theta, &shifted, &accuracy)
                .unwrap()
                .log_standard_error;
        assert!(paired > 0.0 && paired < separate);
        let out = bank
            .log_marginal_score(&theta, &reference, jacobian.view(), &accuracy)
            .unwrap();
        assert_eq!(
            out.likelihood.log_marginal,
            bank.log_marginal(&theta, &reference, &accuracy)
                .unwrap()
                .log_marginal
        );
        for q in [
            0,
            model.layout.rates.start,
            model.layout.entry.start + 1,
            model.layout.jumps[0].as_ref().unwrap().start,
            model.layout.measurement_shape[0].start,
            model.layout.measurement_shape[3].start,
        ] {
            let coefficients: Vec<_> = theta
                .iter()
                .enumerate()
                .map(|(j, &v)| Order1::<1> {
                    v,
                    g: [f64::from(q == j)],
                })
                .collect();
            let moments: Vec<_> = reference
                .iter()
                .enumerate()
                .map(|(n, &v)| Order1::<1> {
                    v,
                    g: [jacobian[[n, q]]],
                })
                .collect();
            let expected = bank
                .log_marginal(&coefficients, &moments, &accuracy)
                .unwrap();
            assert!(
                (out.gradient[q] - expected.log_marginal.g[0]).abs() < 1e-10,
                "coordinate {q}"
            );
        }
        assert!(
            out.standard_error
                .iter()
                .all(|&v| v.is_finite() && v >= 0.0)
        );
        assert!(out.standard_error.iter().any(|&v| v > 0.01));
    }

    #[test]
    fn full_path_coefficient_score_speed() {
        use std::{hint::black_box, time::Instant};
        let (model, h, theta, path, reference, jacobian) = fixture();
        let (mut hand, mut ad) = (f64::INFINITY, f64::INFINITY);
        for _ in 0..3 {
            let start = Instant::now();
            for _ in 0..1000 {
                black_box(
                    model
                        .log_density_score(black_box(&theta), &h, &path, &reference)
                        .unwrap()
                        .pullback(jacobian.view())
                        .unwrap(),
                );
            }
            hand = hand.min(start.elapsed().as_secs_f64());
            let start = Instant::now();
            for _ in 0..1000 {
                black_box(batched_score(
                    &model,
                    &h,
                    black_box(&theta),
                    &path,
                    &reference,
                    &jacobian,
                ));
            }
            ad = ad.min(start.elapsed().as_secs_f64());
        }
        eprintln!(
            "full path {} coefficients: analytic {hand:.6}s, forward AD batches of eight {ad:.6}s, {:.3}x (1000 calls, best of three)",
            theta.len(),
            ad / hand
        );
    }
}
