//! Analytic observation scores of the complete-path density at a fixed latent
//! path, with a separate reference adjoint so the reference evolution's
//! parameter dependence is never lost.
//!
//! The principal route integrates in innovation coordinates. There the entry
//! and OU densities are the standardized law −½|e|², and their coefficients
//! act through the transport (`transport.rs`). This score therefore covers the
//! observation factors: event intensities at each node's anchor with tied
//! counts, the compensator over each cell's quadrature points, decoder weights,
//! disease jumps seen by after-event measurements, and measurement channels. It
//! also returns the adjoint in the node states, which the transport's reverse
//! recursion carries to the entry, drive, rate and jump coefficients. Nothing
//! here divides by an innovation variance, so static signatures stay finite.
//!
//! The score is written once over any [`JetField`] scalar. Seeded along one
//! coefficient direction it returns that direction's Hessian column of the
//! observation density in one pass; no derivative is assembled by replaying
//! coefficient pairs (#2965). Evaluated at the running-error scalar
//! `law::numerical::Running`, the same pass returns every value's first-order
//! rounding bound. The fit's decision bands read that bound only at convergence
//! and classification checks, where one extra pass costs about one f64 pass;
//! the search evaluates at f64.
use super::decoder::PreparedDecoder;
use super::emission;
use super::law::{JointHistory, JointLikelihood, numerical};
use crate::scalar::exp;
use crate::{EventHistoryError, MarkKind};
use gam_math::nested_dual::JetField;

/// Value and analytic scores of the observation density at a fixed latent
/// path. `reference` is the adjoint of the supplied reference log moments
/// (point-major marks). The reference evolution pulls it back onto the
/// coefficients; holding the normalizer fixed is a different estimator and
/// never substitutes for that total score.
pub(super) struct JointPathScore<S> {
    pub(super) log_density: S,
    pub(super) coefficients: Vec<S>,
    pub(super) reference: Vec<S>,
}

pub(super) fn regression_score<S: JetField>(
    out: &mut [S],
    start: usize,
    columns: &[f64],
    genes: &[S],
    axis: usize,
    score: &S,
) {
    for (j, &feature) in columns.iter().enumerate() {
        let base = start + (axis * columns.len() + j) * (genes.len() + 1);
        let weighted = score.scale(feature);
        out[base] = out[base].add(&weighted);
        for (g, gene) in genes.iter().enumerate() {
            out[base + g + 1] = out[base + g + 1].add(&weighted.mul(gene));
        }
    }
}

fn accumulate<S: JetField>(out: &mut [S], index: usize, value: &S) {
    out[index] = out[index].add(value);
}

impl JointLikelihood {
    /// `path_density(.., dynamics = false)` and its analytic scores in the
    /// coefficients, the reference log moments and the node states.
    pub(super) fn observation_score<S: JetField>(
        &self,
        theta: &[S],
        h: &JointHistory,
        path: &[S],
        reference: &[S],
    ) -> Result<(JointPathScore<S>, Vec<S>), EventHistoryError> {
        let log_density = self.path_density(theta, h, path, reference, false)?;
        let k = self.spec.signatures;
        let marks = self.spec.marks.len();
        let columns = self.spec.population_columns;
        let zero = theta[0].constant_like(0.0);
        let missing = h.genetics.iter().filter(|gene| gene.is_none()).count();
        let state = |n: usize| &path[missing + n * k..missing + (n + 1) * k];
        let fired = |node: usize, mark: usize| h.events[node].iter().filter(|&&e| e == mark).count();
        let mut coefficients = vec![zero.clone(); theta.len()];
        let mut reference_score = vec![zero.clone(); reference.len()];
        let mut state_score = vec![zero.clone(); h.times.len() * k];
        let decoder = PreparedDecoder::new(self, theta, h.population_design.view())?;
        let mut risk = h.initially_at_risk.clone();
        let mut log_weight_adjoint = vec![zero.clone(); k + 1];
        let mut point = 0;
        for n in 0..h.times.len() {
            let anchor = point;
            let cell_end = anchor
                + 1
                + h.points[anchor + 1..]
                    .iter()
                    .take_while(|p| p.node == n)
                    .count();
            point = cell_end;
            for d in 0..marks {
                if !risk[d] {
                    continue;
                }
                let count = fired(n, d) as f64;
                // The anchor carries the node's events, each cell point its
                // exposure; every point reads the node's pre-jump state.
                for j in anchor..cell_end {
                    if j == anchor && count == 0.0 {
                        continue;
                    }
                    let weights = decoder.weights(j, d);
                    let log_activity = decoder.activity(j, d, state(n));
                    let log_rate = self
                        .baseline_log_rate(theta, h.baseline_design.row(j), d)
                        .add(&log_activity)
                        .sub(&reference[j * marks + d]);
                    let adjoint = if j == anchor {
                        zero.constant_like(count)
                    } else {
                        exp(&log_rate).scale(h.points[j].weight).neg()
                    };
                    reference_score[j * marks + d] = adjoint.neg();
                    self.baseline_pullback(h.baseline_design.row(j), d, &adjoint, &mut coefficients);
                    // d log A / d log pi_j = pi_j s_j / A with s_0 = 1, and
                    // d log A / d x_j = pi_j sigmoid(x_j) / A.
                    log_weight_adjoint[0] = adjoint.mul(&exp(&weights[0].sub(&log_activity)));
                    for axis in 0..k {
                        let x = &state(n)[axis];
                        log_weight_adjoint[axis + 1] = adjoint.mul(&exp(&weights[axis + 1]
                            .add(&emission::log_softplus(x))
                            .sub(&log_activity)));
                        let slope = exp(&weights[axis + 1]
                            .sub(&emission::softplus(&x.neg()))
                            .sub(&log_activity));
                        state_score[n * k + axis] =
                            state_score[n * k + axis].add(&adjoint.mul(&slope));
                    }
                    let row = h.population_design.row(j);
                    decoder.weights_pullback(j, row, d, &log_weight_adjoint, &mut coefficients);
                }
            }
            for &d in &h.events[n] {
                if self.spec.marks[d] == MarkKind::Once {
                    risk[d] = false;
                }
            }
        }
        for record in &h.measurements {
            let Some(y) = record.value else {
                continue;
            };
            let family = &self.spec.measurements[record.channel];
            let location = self.measurement_location(theta, h, record, state(record.node));
            let shape_range = self.layout.measurement_shape[record.channel].clone();
            let shape = &theta[shape_range.clone()];
            let (score, shape_score) = emission::parameter_scores(family, y, &location.eta, shape)?;
            // A measurement reads the population row of its node's anchor.
            let anchor = h.points.partition_point(|p| p.node < record.node);
            let row = h.population_design.row(anchor);
            let start = self.layout.measurement_location[record.channel].start;
            for (p, &feature) in row.iter().enumerate() {
                accumulate(&mut coefficients, start + p, &score.scale(feature));
            }
            for axis in 0..k {
                let x = if record.after_event {
                    state(record.node)[axis].add(&self.jump(theta, &h.events[record.node], axis))
                } else {
                    state(record.node)[axis].clone()
                };
                let loading_score = score.mul(&x);
                let offset = start + (axis + 1) * columns;
                for (p, &feature) in row.iter().enumerate() {
                    accumulate(&mut coefficients, offset + p, &loading_score.scale(feature));
                }
                let along = score.mul(&location.loadings[axis]);
                state_score[record.node * k + axis] = state_score[record.node * k + axis].add(&along);
                if record.after_event {
                    for (d, range) in self.layout.jumps.iter().enumerate() {
                        let count = fired(record.node, d);
                        if let Some(range) = range.as_ref().filter(|_| count > 0) {
                            accumulate(&mut coefficients, range.start + axis, &along.scale(count as f64));
                        }
                    }
                }
            }
            for (index, value) in shape_range.zip(shape_score) {
                accumulate(&mut coefficients, index, &value);
            }
        }
        if coefficients
            .iter()
            .chain(&reference_score)
            .chain(&state_score)
            .any(|v| !v.value().is_finite())
        {
            return Err(numerical("non-finite observation coefficient score"));
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
    use super::super::law::numerical::Running;
    use super::super::law::{
        CompensatorPoint, JointSpecification, MeasurementFamily, MeasurementRecord,
    };
    use super::*;
    use crate::scalar::Rows;
    use crate::test_support::Bound;
    use ndarray::Array2;

    fn exact(values: &[f64]) -> Vec<Bound> {
        values.iter().map(|&v| Bound::exact(v)).collect()
    }

    fn running(values: &[f64]) -> Vec<Running> {
        values.iter().map(|&v| Running::exact(v)).collect()
    }

    /// `v` with tangent `u` along one direction.
    fn directional(v: f64, u: f64) -> Rows<Bound, 1> {
        Rows::seed(Bound::exact(v), [u])
    }

    /// `v` with tangent `u` along an inner and `w` along an outer direction:
    /// `.rows[0].rows[0]` is the mixed second derivative.
    fn paired(v: f64, u: f64, w: f64) -> Rows<Rows<Bound, 1>, 1> {
        Rows::seed(Rows::seed(Bound::exact(v), [u]), [w])
    }

    /// (node, weight, time) of every compensator point: each node's zero-weight
    /// anchor, then its cell points. Node 3's cell is split into two points.
    const POINTS: [(usize, f64, f64); 14] = [
        (0, 0.0, 0.0),
        (1, 0.0, 0.2),
        (1, 0.2, 0.1),
        (2, 0.0, 0.3),
        (2, 0.1, 0.25),
        (3, 0.0, 0.5),
        (3, 0.1, 0.35),
        (3, 0.1, 0.45),
        (4, 0.0, 0.65),
        (4, 0.15, 0.575),
        (5, 0.0, 0.8),
        (5, 0.15, 0.725),
        (6, 0.0, 1.0),
        (6, 0.2, 0.9),
    ];

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
            population_columns: 2,
            drive_columns: 2,
            entry_columns: 1,
            baseline_penalties: vec![],
            drive_penalties: vec![],
            population_penalties: vec![],
            measurements: vec![
                MeasurementFamily::StudentT,
                MeasurementFamily::Probit { categories: 2 },
                MeasurementFamily::Probit { categories: 4 },
                MeasurementFamily::NegativeBinomial,
            ],
            genetic_mean: vec![0.1, -0.2],
            genetic_precision: Array2::eye(2),
        })
        .unwrap();
        let times = vec![0.0, 0.2, 0.3, 0.5, 0.65, 0.8, 1.0];
        let point_design =
            Array2::from_shape_fn((POINTS.len(), 2), |(j, b)| if b == 0 { 1.0 } else { POINTS[j].2 });
        let h = JointHistory {
            baseline_design: point_design.clone(),
            population_design: point_design,
            drive_design: Array2::from_shape_fn((6, 2), |(n, b)| if b == 0 { 1.0 } else { times[n] }),
            times,
            points: POINTS
                .iter()
                .map(|&(node, weight, time)| CompensatorPoint { node, time, weight })
                .collect(),
            // A recurrent mark tied with itself, a once-only mark tied with the
            // recurrent one, and a terminal event at exit.
            events: vec![vec![], vec![], vec![0, 0], vec![], vec![1, 0], vec![], vec![2]],
            initially_at_risk: vec![true; 3],
            entry_design: vec![0.6],
            genetics: vec![None, Some(0.6)],
            measurements: (0..4)
                .flat_map(|channel| {
                    [false, true].map(move |after_event| MeasurementRecord {
                        node: 2,
                        channel,
                        value: Some(if channel == 2 { 2.0 } else { 1.0 }),
                        exposure: (channel == 3).then_some(1.3),
                        after_event,
                    })
                })
                .collect(),
        };
        let mut theta: Vec<_> = (0..model.layout.width)
            .map(|j| 0.03 * (j as f64).sin())
            .collect();
        for d in 0..3 {
            theta[model.layout.baseline.start + 2 * d] = -1.2;
        }
        let path: Vec<_> = (0..model.latent_dimension(&h).unwrap())
            .map(|j| 0.2 * (j as f64).cos())
            .collect();
        let rows = POINTS.len() * 3;
        let reference: Vec<_> = (0..rows).map(|j| 0.02 * (j as f64).cos()).collect();
        let jacobian =
            Array2::from_shape_fn((rows, theta.len()), |(n, p)| 0.01 * ((n + p) as f64).sin());
        (model, h, theta, path, reference, jacobian)
    }

    /// Total score through a fixed linear reference map m = m0 + J theta.
    fn total<S: JetField>(score: &JointPathScore<S>, jacobian: &Array2<f64>) -> Vec<S> {
        let mut out = score.coefficients.clone();
        for (row, adjoint) in jacobian.rows().into_iter().zip(&score.reference) {
            for (value, &derivative) in out.iter_mut().zip(row) {
                *value = value.add(&adjoint.scale(derivative));
            }
        }
        out
    }

    #[test]
    fn regression_score_is_the_features_times_the_genetic_channels() {
        // Axis 1 of a two-axis block with features (1, t) and scores g = (0.5, -2).
        let mut out = vec![0.0; 12];
        regression_score(&mut out, 0, &[1.0, 0.25], &[0.5, -2.0], 1, &3.0);
        assert_eq!(out[..6], [0.0; 6]);
        assert_eq!(out[6..], [3.0, 1.5, -6.0, 0.75, 0.375, -1.5]);
    }

    #[test]
    fn observation_score_includes_ties_cells_decoder_jumps_shapes_and_reference() {
        // The production band is Running's first-order bound from one pass of the
        // same generic route, certified against the separate Bound oracle. Its
        // charges: sqrt and division derived, exp and ln cited from glibc 2.28, and
        // softplus as composed, log Φ (probit channels) and lgamma/digamma (the
        // count channel) measured. Coordinates reading those channels carry
        // measured bands.
        let (model, h, theta, path, reference, jacobian) = fixture();
        let (result, _) = model
            .observation_score(&theta, &h, &path, &reference)
            .unwrap();
        let actual = total(&result, &jacobian);
        let (bounded, _) = model
            .observation_score(&exact(&theta), &h, &exact(&path), &exact(&reference))
            .unwrap();
        let (banded, _) = model
            .observation_score(&running(&theta), &h, &running(&path), &running(&reference))
            .unwrap();
        // The f64 value is the density's own, and Running's band covers the bounded route.
        let value_bar = banded.log_density.rounding() + bounded.log_density.rounding();
        assert!((result.log_density - bounded.log_density.value).abs() <= value_bar);
        let bounded_total = total(&bounded, &jacobian);
        let banded_total = total(&banded, &jacobian);
        let states: Vec<_> = path.iter().map(|&v| directional(v, 0.0)).collect();
        let mut resolved = vec![false; theta.len()];
        for j in 0..theta.len() {
            let coefficients: Vec<_> = theta
                .iter()
                .enumerate()
                .map(|(q, &v)| directional(v, f64::from(q == j)))
                .collect();
            let moments: Vec<_> = reference
                .iter()
                .enumerate()
                .map(|(n, &v)| directional(v, jacobian[[n, j]]))
                .collect();
            let oracle = model
                .path_density(&coefficients, &h, &states, &moments, false)
                .unwrap()
                .rows[0];
            // Two routes to one derivative agree within their own running bounds.
            let bar = bounded_total[j].bar(&oracle);
            assert!(
                (bounded_total[j].value - oracle.value).abs() <= bar,
                "coordinate {j}: {} vs {}, bar {bar}",
                bounded_total[j].value,
                oracle.value
            );
            // The production band: the f64 score within Running's bound and the oracle's.
            let band = banded_total[j].rounding() + oracle.rounding();
            assert!(
                (actual[j] - oracle.value).abs() <= band,
                "coordinate {j}: f64 score {} vs {}, band {band}",
                actual[j],
                oracle.value
            );
            resolved[j] = oracle.value.abs() > bar.max(band);
        }
        // Every observation block is exercised above its bar: cells and ties in
        // the baseline, the decoder, both tied jumps, and every channel's
        // location and shape.
        let blocks = [model.layout.baseline.clone(), model.layout.decoder.clone()]
            .into_iter()
            .chain(model.layout.jumps.iter().flatten().cloned())
            .chain(model.layout.measurement_location.iter().cloned())
            .chain(model.layout.measurement_shape.iter().filter(|r| !r.is_empty()).cloned());
        for block in blocks {
            assert!(
                resolved[block.clone()].iter().any(|&r| r),
                "block {block:?} has no coordinate above its bar"
            );
        }
        // The observation density does not see the entry, drive or rates.
        for index in model
            .layout
            .entry
            .clone()
            .chain(model.layout.drive.clone())
            .chain(model.layout.rates.clone())
        {
            assert_eq!(result.coefficients[index], 0.0);
        }
    }

    #[test]
    fn one_seeded_score_pass_is_a_hessian_column_of_the_observation_density() {
        // A column costs one pass of the score, never a replay per coefficient
        // pair. Check it against the second-order jet of the density itself,
        // including second derivatives transported through the reference map.
        let (model, h, theta, path, reference, jacobian) = fixture();
        let columns = model.spec.population_columns;
        let states: Vec<_> = path.iter().map(|&v| directional(v, 0.0)).collect();
        let pair_states: Vec<_> = path.iter().map(|&v| paired(v, 0.0, 0.0)).collect();
        for column in [
            model.layout.baseline.start + 3,
            model.layout.decoder.start + 1,
            model.layout.decoder.start + 2 * columns + 1,
            model.layout.jumps[0].as_ref().unwrap().start + 1,
            model.layout.jumps[1].as_ref().unwrap().start,
            model.layout.measurement_location[0].start + columns + 1,
            model.layout.measurement_shape[0].start + 1,
            model.layout.measurement_shape[2].start,
            model.layout.measurement_shape[3].start,
        ] {
            let seeded: Vec<_> = theta
                .iter()
                .enumerate()
                .map(|(j, &v)| directional(v, f64::from(j == column)))
                .collect();
            let moments: Vec<_> = reference
                .iter()
                .enumerate()
                .map(|(n, &v)| directional(v, jacobian[[n, column]]))
                .collect();
            let (score, _) = model
                .observation_score(&seeded, &h, &states, &moments)
                .unwrap();
            let hessian_column = total(&score, &jacobian);
            let mut resolved = 0;
            for j in 0..theta.len() {
                let pair: Vec<_> = theta
                    .iter()
                    .enumerate()
                    .map(|(q, &v)| paired(v, f64::from(q == column), f64::from(q == j)))
                    .collect();
                let pair_moments: Vec<_> = reference
                    .iter()
                    .enumerate()
                    .map(|(n, &v)| paired(v, jacobian[[n, column]], jacobian[[n, j]]))
                    .collect();
                let oracle = model
                    .path_density(&pair, &h, &pair_states, &pair_moments, false)
                    .unwrap()
                    .rows[0]
                    .rows[0];
                let production = &hessian_column[j].rows[0];
                let bar = production.bar(&oracle);
                assert!(
                    (production.value - oracle.value).abs() <= bar,
                    "column {column}, row {j}: {} vs {}, bar {bar}",
                    production.value,
                    oracle.value
                );
                resolved += usize::from(oracle.value.abs() > bar);
            }
            assert!(resolved > 0, "column {column} has no entry above its bar");
        }
    }

    #[test]
    fn state_adjoint_is_the_observation_score_in_the_latent_path() {
        let (model, h, theta, path, reference, _) = fixture();
        let (score, adjoint) = model
            .observation_score(&exact(&theta), &h, &exact(&path), &exact(&reference))
            .unwrap();
        let coefficients: Vec<_> = theta.iter().map(|&v| directional(v, 0.0)).collect();
        let moments: Vec<_> = reference.iter().map(|&v| directional(v, 0.0)).collect();
        let missing = h.genetics.iter().filter(|g| g.is_none()).count();
        assert_eq!(adjoint.len(), path.len() - missing);
        let mut resolved = 0;
        for (node_axis, production) in adjoint.iter().enumerate() {
            let states: Vec<_> = path
                .iter()
                .enumerate()
                .map(|(q, &v)| directional(v, f64::from(q == missing + node_axis)))
                .collect();
            let oracle = model
                .path_density(&coefficients, &h, &states, &moments, false)
                .unwrap();
            let value_bar = score.log_density.bar(&oracle.base);
            assert!((score.log_density.value - oracle.base.value).abs() <= value_bar);
            let bar = production.bar(&oracle.rows[0]);
            assert!(
                (production.value - oracle.rows[0].value).abs() <= bar,
                "state {node_axis}: {} vs {}, bar {bar}",
                production.value,
                oracle.rows[0].value
            );
            resolved += usize::from(oracle.rows[0].value.abs() > bar);
        }
        assert!(resolved > 0, "no state adjoint is above its bar");
    }
}
