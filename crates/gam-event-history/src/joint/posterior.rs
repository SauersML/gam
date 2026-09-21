//! Conditional latent-mode inference and its Laplace integral.
//!
//! Latent coordinates of one history are the missing genetic scores and the
//! standardized OU innovations of `transport.rs`, so paths follow the
//! coefficients through the affine state law. The mode is found by Newton
//! steps whose linear systems are the structured Gaussian problems of
//! `precision.rs`. No Cartesian latent grid and no dense trajectory precision
//! is formed.
//!
//! The returned Gaussian is an approximation of the posterior. `log_marginal`
//! is a Laplace approximation of the latent integral, not an exact evidence.
//!
//! Observation factors are node-local. A node's events read the pre-jump state
//! at its anchor point, with their tied counts. Its compensator integrates the
//! cell points `(t_(n-1), t_n]` at the same state, each at its own baseline and
//! population rows and reference moment. Measurements read the population row
//! of their node's anchor. The observation routes are generic over the scalar,
//! so a rounding-bound tracker instruments the same arithmetic the mode
//! evaluates in `f64`.
use super::decoder::PreparedDecoder;
use super::emission;
use super::law::{JointHistory, JointLikelihood, invalid, numerical};
use super::precision::{Factorization, Solution, StateLaw};
use crate::scalar::{add_real, exp};
use crate::{EventHistoryError, MarkKind};
use gam_math::nested_dual::JetField;
use ndarray::{Array1, Array2};
use std::ops::Range;

/// A converged posterior mode with the Gaussian approximation at that mode.
#[derive(Clone, Debug)]
pub struct LaplacePosterior {
    /// Missing genetic coordinates of the mode, in history order.
    pub genes: Vec<f64>,
    /// Node-major states of the mode path.
    pub states: Array2<f64>,
    /// Standardized OU innovations of the mode path. Under other coefficients
    /// the same innovations follow the new state law, which makes them a warm
    /// start there.
    pub innovations: Array2<f64>,
    pub state_covariance: Vec<Array2<f64>>,
    /// `lag_covariance[n - 1] = Cov(x_n, x_(n-1))`.
    pub lag_covariance: Vec<Array2<f64>>,
    pub genetic_covariance: Array2<f64>,
    pub state_genetic_covariance: Vec<Array2<f64>>,
    /// Laplace approximation of the latent integral; not an exact evidence.
    pub log_marginal: f64,
    /// `log det` of the posterior precision in genetic and standardized
    /// innovation coordinates, and the magnitude its pivots sum.
    pub log_determinant: f64,
    pub log_determinant_magnitude: f64,
    /// Newton decrement at the returned mode.
    pub newton_decrement: f64,
    pub iterations: usize,
}

/// Node-local observation factors at one path: node-major gradients, and per
/// node row-major `K x K` curvatures.
pub(super) struct Observation<S> {
    pub(super) gradient: Vec<S>,
    /// Observed `-grad^2 h_n`.
    pub(super) curvature: Vec<Vec<S>>,
    /// The positive-semidefinite part of the curvature: the observed one without
    /// the negative convexity of an observed event's softplus activity and
    /// without the negative curvature of a measurement in its tails.
    pub(super) metric: Vec<Vec<S>>,
    /// The part the metric drops, `metric - curvature`, accumulated from its own
    /// summands: an observed event's `count pi_k softplus''(x_k) / R` on the
    /// diagonal and a measurement's `max(d^2 log p / d eta^2, 0) l l'`. Each summand
    /// is a nonnegative multiple of a nonnegative diagonal or of an outer
    /// product. The multipliers are nonnegative as the values they are computed
    /// to be: an event's integer count times `share * complement`, a product of
    /// two exponentials, and a measurement's second derivative selected only when
    /// its computed value is positive. The excess is defined by those computed
    /// multipliers and vectors, so it is positive semidefinite exactly; the solve
    /// certificate subtracts it.
    pub(super) excess: Vec<Vec<S>>,
    /// Sum of the magnitudes of the evaluated factors, which sets the
    /// resolution of the objective in floating point.
    pub(super) magnitude: f64,
}

/// The positive decoder of one mark at one point and state.
struct EventGeometry<S> {
    log_rate: S,
    /// `q_k = d log R / d x_k`.
    share: Vec<S>,
    /// `pi_k softplus''(x_k) / R`.
    convexity: Vec<S>,
}

/// One subject's posterior mode at fixed coefficients: the state law, the
/// factorization of the posterior precision at the mode, and the mode itself.
pub(super) struct Mode {
    pub(super) law: StateLaw,
    pub(super) factor: Factorization,
    /// Genes, states and innovations are the evaluated mode; adjoints come
    /// from the Newton solve there.
    pub(super) point: Solution,
    /// `f(mode) + (m/2) log(2 pi) - log det Q / 2`, a Laplace approximation.
    pub(super) log_marginal: f64,
    pub(super) decrement: f64,
    pub(super) iterations: usize,
}

/// Each node's anchor point and its cell points, for a validated history.
fn node_points(h: &JointHistory) -> Vec<(usize, Range<usize>)> {
    let mut out = Vec::with_capacity(h.times.len());
    let mut anchor = 0;
    for n in 0..h.times.len() {
        let end = anchor + 1 + h.points[anchor + 1..].iter().take_while(|p| p.node == n).count();
        out.push((anchor, anchor + 1..end));
        anchor = end;
    }
    out
}

impl JointLikelihood {
    fn event_geometry<S: JetField>(
        &self,
        theta: &[S],
        decoder: &PreparedDecoder<S>,
        h: &JointHistory,
        point: usize,
        mark: usize,
        x: &[S],
        reference: &[S],
    ) -> EventGeometry<S> {
        let k = self.spec.signatures;
        let weights = decoder.weights(point, mark);
        let log_activity = decoder.activity(point, mark, x);
        let log_rate = self
            .baseline_log_rate(theta, h.baseline_design.row(point), mark)
            .add(&log_activity)
            .sub(&reference[point * self.spec.marks.len() + mark]);
        let mut geometry = EventGeometry {
            log_rate,
            share: Vec::with_capacity(k),
            convexity: Vec::with_capacity(k),
        };
        for axis in 0..k {
            let share = exp(&weights[axis + 1]
                .sub(&emission::softplus(&x[axis].neg()))
                .sub(&log_activity));
            let complement = exp(&emission::softplus(&x[axis]).neg());
            geometry.convexity.push(share.mul(&complement));
            geometry.share.push(share);
        }
        geometry
    }

    /// Gradients, curvatures and metrics of every node's observation factors
    /// at node-major `states`, with reference moments per point and mark.
    pub(super) fn observation<S: JetField>(
        &self,
        theta: &[S],
        h: &JointHistory,
        states: &[S],
        reference: &[S],
    ) -> Result<Observation<S>, EventHistoryError> {
        let k = self.spec.signatures;
        let nodes = h.times.len();
        let marks = self.spec.marks.len();
        if states.len() != nodes * k || reference.len() != h.points.len() * marks {
            return Err(invalid(
                "joint observation states or reference moments do not match the history",
            ));
        }
        let zero = theta[0].constant_like(0.0);
        let decoder = PreparedDecoder::new(self, theta, h.population_design.view())?;
        let mut observation = Observation {
            gradient: vec![zero.clone(); nodes * k],
            curvature: vec![vec![zero.clone(); k * k]; nodes],
            metric: vec![vec![zero.clone(); k * k]; nodes],
            excess: vec![vec![zero.clone(); k * k]; nodes],
            magnitude: 0.0,
        };
        let mut risk = h.initially_at_risk.clone();
        for (n, (anchor, cell)) in node_points(h).into_iter().enumerate() {
            let x = &states[n * k..(n + 1) * k];
            for d in 0..marks {
                if !risk[d] {
                    continue;
                }
                let count = h.events[n].iter().filter(|&&e| e == d).count() as f64;
                let points = std::iter::once((anchor, count, 0.0))
                    .chain(cell.clone().map(|j| (j, 0.0, h.points[j].weight)));
                for (j, events, weight) in points {
                    if events == 0.0 && weight == 0.0 {
                        continue;
                    }
                    let e = self.event_geometry(theta, &decoder, h, j, d, x, reference);
                    let compensator = exp(&e.log_rate).scale(weight);
                    observation.magnitude +=
                        (events * e.log_rate.value()).abs() + compensator.value().abs();
                    let residual = add_real(&compensator.neg(), events);
                    for i in 0..k {
                        let index = n * k + i;
                        observation.gradient[index] =
                            observation.gradient[index].add(&residual.mul(&e.share[i]));
                        observation.curvature[n][i * k + i] =
                            observation.curvature[n][i * k + i].sub(&residual.mul(&e.convexity[i]));
                        observation.metric[n][i * k + i] =
                            observation.metric[n][i * k + i].add(&compensator.mul(&e.convexity[i]));
                        if events > 0.0 {
                            // metric - curvature = (compensator + residual) convexity.
                            observation.excess[n][i * k + i] =
                                observation.excess[n][i * k + i].add(&e.convexity[i].scale(events));
                        }
                        for jj in 0..k {
                            let outer = e.share[i].mul(&e.share[jj]).scale(events);
                            observation.curvature[n][i * k + jj] =
                                observation.curvature[n][i * k + jj].add(&outer);
                            observation.metric[n][i * k + jj] =
                                observation.metric[n][i * k + jj].add(&outer);
                        }
                    }
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
            let n = record.node;
            let location = self.measurement_location(theta, h, record, &states[n * k..(n + 1) * k]);
            let shape = &theta[self.layout.measurement_shape[record.channel].clone()];
            let family = &self.spec.measurements[record.channel];
            let (score, second) = emission::location_derivatives(family, y, &location.eta, shape)?;
            observation.magnitude += emission::log_density(family, y, &location.eta, shape)?
                .value()
                .abs();
            let tail = if second.value() < 0.0 {
                second.neg()
            } else {
                zero.clone()
            };
            // metric - curvature = (max(-second, 0) + second) l l' = max(second, 0) l l'.
            let dropped = if second.value() > 0.0 {
                second.clone()
            } else {
                zero.clone()
            };
            for i in 0..k {
                let index = n * k + i;
                observation.gradient[index] =
                    observation.gradient[index].add(&score.mul(&location.loadings[i]));
                for jj in 0..k {
                    let outer = location.loadings[i].mul(&location.loadings[jj]);
                    observation.curvature[n][i * k + jj] =
                        observation.curvature[n][i * k + jj].sub(&second.mul(&outer));
                    observation.metric[n][i * k + jj] =
                        observation.metric[n][i * k + jj].add(&tail.mul(&outer));
                    observation.excess[n][i * k + jj] =
                        observation.excess[n][i * k + jj].add(&dropped.mul(&outer));
                }
            }
        }
        if observation
            .gradient
            .iter()
            .chain(observation.curvature.iter().flatten())
            .chain(observation.metric.iter().flatten())
            .chain(observation.excess.iter().flatten())
            .any(|v| !v.value().is_finite())
            || !observation.magnitude.is_finite()
        {
            return Err(numerical("non-finite joint observation curvature"));
        }
        Ok(observation)
    }

    /// Log posterior density in genetic and standardized innovation
    /// coordinates, without the `-(N K / 2) log(2 pi)` innovation normalizer.
    fn latent_objective(
        &self,
        theta: &[f64],
        h: &JointHistory,
        genes: &Array1<f64>,
        states: &Array2<f64>,
        innovations: &Array2<f64>,
        reference: &[f64],
    ) -> Result<f64, EventHistoryError> {
        let path: Vec<f64> = genes.iter().chain(states.iter()).copied().collect();
        Ok(self.path_density(theta, h, &path, reference, false)?
            - 0.5 * innovations.iter().map(|e| e * e).sum::<f64>())
    }

    /// Newton iteration to the posterior mode. The observed curvature gives the
    /// step whenever the conditional precision it defines is positive definite.
    /// Otherwise the metric of [`Observation`] does, which is always definite
    /// with the Gaussian state law. A step is accepted when it realizes at
    /// least half of the gain its quadratic model predicts,
    /// `(t - t^2/2) delta^2`, halving `t` otherwise. The iteration stops when
    /// the predicted gain `delta^2/2` of an observed-curvature step is below
    /// the floating-point resolution of the objective. That last step is then
    /// taken, and the mode is evaluated there. There is no iteration budget: a
    /// step that no longer changes the coordinates is refused, as is a
    /// stationary point whose observed curvature is not definite, where the
    /// Laplace approximation is undefined.
    pub(super) fn mode(
        &self,
        theta: &[f64],
        h: &JointHistory,
        reference: &[f64],
        initial: Option<&LaplacePosterior>,
    ) -> Result<Mode, EventHistoryError> {
        let law = self.state_law(theta, h)?;
        let k = law.signatures();
        let nodes = law.nodes();
        let m = law.genes();
        // Retained factorization plus the observation curvature, metric and
        // gradient of one iteration, from the actual allocation sizes.
        let observation_bytes = k
            .checked_mul(k)
            .and_then(|square| square.checked_mul(2))
            .and_then(|blocks| blocks.checked_add(k))
            .and_then(|per_node| per_node.checked_mul(nodes))
            .and_then(|entries| entries.checked_mul(std::mem::size_of::<f64>()));
        let bytes = Factorization::working_bytes(nodes, k, m)
            .zip(observation_bytes)
            .and_then(|(factor, observation)| factor.checked_add(observation))
            .ok_or_else(|| invalid("joint posterior storage size overflows"))?;
        let budget = gam_runtime::resource::ResourcePolicy::default_library()
            .max_single_materialization_bytes;
        if bytes as f64 > budget as f64 {
            return Err(invalid(format!(
                "a joint posterior over {nodes} nodes, {k} signatures and {m} missing genetic scores needs {bytes} bytes, above this machine's {budget}-byte materialisation budget",
            )));
        }
        if reference.len() != h.points.len() * self.spec.marks.len()
            || reference.iter().any(|v| !v.is_finite())
        {
            return Err(invalid(
                "joint reference moments have invalid dimensions or non-finite values",
            ));
        }
        let (mut genes, mut innovations) = match initial {
            Some(start)
                if start.genes.len() == m
                    && start.innovations.dim() == (nodes, k)
                    && start
                        .genes
                        .iter()
                        .chain(start.innovations.iter())
                        .all(|v| v.is_finite()) =>
            {
                (Array1::from(start.genes.clone()), start.innovations.clone())
            }
            Some(_) => {
                return Err(invalid(
                    "joint posterior warm start has invalid dimensions or values",
                ));
            }
            None => (law.gene_mean.clone(), Array2::zeros((nodes, k))),
        };
        let mut iterations = 0;
        let mut converged = false;
        loop {
            let states = law.transport(genes.view(), innovations.view(), true);
            let value = self.latent_objective(theta, h, &genes, &states, &innovations, reference)?;
            let observation = self.observation(
                theta,
                h,
                states.as_slice().ok_or_else(|| numerical("joint states are not contiguous"))?,
                reference,
            )?;
            let blocks = |rows: &[Vec<f64>]| -> Result<Vec<Array2<f64>>, EventHistoryError> {
                rows.iter()
                    .map(|block| {
                        Array2::from_shape_vec((k, k), block.clone())
                            .map_err(|error| numerical(format!("joint curvature block: {error}")))
                    })
                    .collect()
            };
            let observed = blocks(&observation.curvature)?;
            let (factor, curvature, exact) = match Factorization::new(&law, &observed)? {
                Some(factor) => (factor, observed, true),
                None if converged => {
                    return Err(numerical(
                        "joint posterior mode is not a strict local maximum: its curvature is not positive definite",
                    ));
                }
                None => {
                    let metric = blocks(&observation.metric)?;
                    let factor = Factorization::new(&law, &metric)?.ok_or_else(|| {
                        numerical("joint posterior metric is not positive definite")
                    })?;
                    (factor, metric, false)
                }
            };
            let gradient = Array2::from_shape_vec((nodes, k), observation.gradient.clone())
                .map_err(|error| numerical(format!("joint observation gradient: {error}")))?;
            let mut information = gradient.clone();
            for (n, w) in curvature.iter().enumerate() {
                let anchored = w.dot(&states.row(n));
                let mut row = information.row_mut(n);
                row += &anchored;
            }
            let solution = factor.solve(&law, information.view(), true)?;
            let (gene_force, innovation_force) = law.transport_adjoint(gradient.view());
            let gene_step = &solution.genes - &genes;
            let innovation_step = &solution.innovations - &innovations;
            let gene_gradient = gene_force - law.gene_precision.dot(&(&genes - &law.gene_mean));
            let innovation_gradient = innovation_force - &innovations;
            let decrement =
                gene_step.dot(&gene_gradient) + (&innovation_step * &innovation_gradient).sum();
            if !decrement.is_finite() {
                return Err(numerical("joint Newton decrement is unresolved"));
            }
            if converged {
                let log_marginal = value + 0.5 * m as f64 * (2.0 * std::f64::consts::PI).ln()
                    - 0.5 * factor.log_determinant;
                if !log_marginal.is_finite() {
                    return Err(numerical("non-finite joint Laplace integral"));
                }
                let point = Solution {
                    genes,
                    states,
                    adjoints: solution.adjoints,
                    innovations,
                };
                return Ok(Mode {
                    law,
                    factor,
                    point,
                    log_marginal,
                    decrement: decrement.max(0.0).sqrt(),
                    iterations,
                });
            }
            let resolution = f64::EPSILON
                * (value.abs()
                    + observation.magnitude
                    + innovations.iter().map(|e| e * e).sum::<f64>());
            iterations += 1;
            if 0.5 * decrement <= resolution {
                if !exact {
                    return Err(numerical(
                        "joint posterior mode is not a strict local maximum: its observed curvature is not positive definite",
                    ));
                }
                genes = solution.genes;
                innovations = solution.innovations;
                converged = true;
                continue;
            }
            let mut fraction = 1.0;
            loop {
                let trial_genes = &genes + &(&gene_step * fraction);
                let trial_innovations = &innovations + &(&innovation_step * fraction);
                if trial_genes == genes && trial_innovations == innovations {
                    return Err(numerical(
                        "joint posterior mode has no resolved ascent step",
                    ));
                }
                let trial_states = law.transport(trial_genes.view(), trial_innovations.view(), true);
                // The quadratic model predicts the gain (t - t^2/2) delta^2. Requiring half
                // of it is the conventional sufficient-increase fraction; any fraction
                // below one accepts the full step near the mode, where the model is exact
                // to third order in the step.
                let model = fraction * (1.0 - 0.5 * fraction) * decrement;
                match self.latent_objective(
                    theta,
                    h,
                    &trial_genes,
                    &trial_states,
                    &trial_innovations,
                    reference,
                ) {
                    Ok(candidate) if candidate - value >= 0.5 * model => {
                        genes = trial_genes;
                        innovations = trial_innovations;
                        break;
                    }
                    Ok(_) | Err(EventHistoryError::NumericalFailure { .. }) => (),
                    Err(error) => return Err(error),
                }
                fraction *= 0.5;
            }
        }
    }

    /// The Laplace approximation of the latent integral at fixed coefficients
    /// and reference moments, one per point and mark.
    pub(super) fn laplace_posterior(
        &self,
        theta: &[f64],
        h: &JointHistory,
        reference: &[f64],
        initial: Option<&LaplacePosterior>,
    ) -> Result<LaplacePosterior, EventHistoryError> {
        let mode = self.mode(theta, h, reference, initial)?;
        let moments = mode.factor.moments;
        Ok(LaplacePosterior {
            genes: mode.point.genes.to_vec(),
            states: mode.point.states,
            innovations: mode.point.innovations,
            state_covariance: moments.states,
            lag_covariance: moments.lags,
            genetic_covariance: moments.genes,
            state_genetic_covariance: moments.state_genes,
            log_marginal: mode.log_marginal,
            log_determinant: mode.factor.log_determinant,
            log_determinant_magnitude: mode.factor.log_determinant_magnitude,
            newton_decrement: mode.decrement,
            iterations: mode.iterations,
        })
    }

    /// [`InnovationScore`] at missing genetic scores `genes` and standardized
    /// innovations `innovations`, generic over the scalar: at the production
    /// running-error scalar each value carries its rounding bound for a decision
    /// band, and the f64 hot path is the same arithmetic. That bound rests on the
    /// OU factors' `softplus` charge, which is measured, not derived or cited
    /// (see `JointLikelihood::transition`).
    pub(super) fn innovation_score<S: JetField>(
        &self,
        theta: &[S],
        h: &JointHistory,
        genes: &[S],
        innovations: &[S],
        reference: &[S],
    ) -> Result<InnovationScore<S>, EventHistoryError> {
        let states = self.transport_states(theta, h, genes, innovations, true)?;
        let path: Vec<S> = genes.iter().chain(&states).cloned().collect();
        let (score, forces) = self.observation_score(theta, h, &path, reference)?;
        let transported =
            self.transport_coefficient_adjoint(theta, h, genes, innovations, &forces, true)?;
        let squares = innovations
            .iter()
            .fold(theta[0].constant_like(0.0), |sum, e| sum.add(&e.mul(e)));
        Ok(InnovationScore {
            value: score.log_density.sub(&squares.scale(0.5)),
            coefficients: score
                .coefficients
                .iter()
                .zip(&transported)
                .map(|(direct, carried)| direct.add(carried))
                .collect(),
            reference: score.reference,
        })
    }
}

/// The Laplace profile `path_density(dynamics = false) - |e|^2 / 2` at fixed
/// latent coordinates, its coefficient score and the adjoint of the reference
/// moments.
pub(super) struct InnovationScore<S> {
    pub(super) value: S,
    /// The total coefficient score at fixed `(g, e)`: the observation score
    /// plus the state adjoint carried back through the transport. At a
    /// converged mode the latent gradient vanishes, so this is the derivative of
    /// the profile along the mode (envelope).
    pub(super) coefficients: Vec<S>,
    /// Adjoint of the reference log moments, point-major marks.
    pub(super) reference: Vec<S>,
}

#[cfg(test)]
mod tests {
    use super::super::law::numerical::Running;
    use super::super::law::{CompensatorPoint, JointSpecification, MeasurementFamily, MeasurementRecord};
    use super::*;
    use crate::scalar::Rows;
    use crate::test_support::Bound;

    type Two = Rows<Rows<Bound, 1>, 1>;

    fn two(value: f64, u: f64, v: f64) -> Two {
        Rows::seed(Rows::seed(Bound::exact(value), [u]), [v])
    }

    /// Anchors at every node and `cells` equal cell points per gap, with
    /// baseline and population rows `[1, t]` at each point's time.
    fn quadrature(times: &[f64], cells: usize) -> (Vec<CompensatorPoint>, Array2<f64>) {
        let mut points = vec![CompensatorPoint {
            node: 0,
            time: times[0],
            weight: 0.0,
        }];
        for n in 1..times.len() {
            points.push(CompensatorPoint {
                node: n,
                time: times[n],
                weight: 0.0,
            });
            let width = (times[n] - times[n - 1]) / cells as f64;
            for c in 0..cells {
                points.push(CompensatorPoint {
                    node: n,
                    time: times[n - 1] + width * (c as f64 + 0.5),
                    weight: width,
                });
            }
        }
        let rows = Array2::from_shape_fn((points.len(), 2), |(j, p)| if p == 0 { 1.0 } else { points[j].time });
        (points, rows)
    }

    fn fixture() -> (JointLikelihood, JointHistory, Vec<f64>, Vec<f64>) {
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
            genetic_precision: ndarray::arr2(&[[1.5, -0.4], [-0.4, 1.2]]),
        })
        .unwrap();
        let times = vec![0.0, 0.2, 0.3, 0.5, 0.65, 0.8, 1.0];
        let (points, rows) = quadrature(&times, 2);
        let h = JointHistory {
            drive_design: Array2::from_shape_fn((6, 2), |(n, b)| if b == 0 { 1.0 } else { times[n] }),
            times,
            points,
            events: vec![vec![], vec![], vec![0], vec![], vec![1, 0], vec![], vec![2]],
            initially_at_risk: vec![true; 3],
            baseline_design: rows.clone(),
            population_design: rows,
            entry_design: vec![0.6],
            genetics: vec![None, Some(0.6)],
            measurements: (0..4)
                .flat_map(|channel| {
                    [false, true].map(move |after_event| MeasurementRecord {
                        node: 2,
                        channel,
                        value: Some(if channel == 2 { 2.0 } else { 1.0 }),
                        exposure: (channel == 3).then_some(1.5),
                        after_event,
                    })
                })
                .collect(),
        };
        let mut theta: Vec<f64> = (0..model.layout.width)
            .map(|j| 0.3 * (j as f64).sin())
            .collect();
        for d in 0..3 {
            theta[model.layout.baseline.start + 2 * d] = -0.4;
        }
        let reference = (0..h.points.len() * 3).map(|j| 0.02 * (j as f64).cos()).collect();
        (model, h, theta, reference)
    }

    /// One entry within the two routes' bounds; true when the oracle entry is
    /// above its bar. Some entries are exactly zero by algebra and leave only a
    /// roundoff residual on the jet route, so the magnitude floor is asserted
    /// over a node's block, never per entry. Node 1 of the fixture carries
    /// compensator points only. There each mark's rate is
    /// `exp(baseline + activity) = e^baseline (pi_0 + sum_k pi_k softplus(x_k))`
    /// (`PreparedDecoder::activity` is the log of that sum), a sum of one-axis
    /// terms, so every cross-axis second derivative `d^2 rate / dx_0 dx_1`
    /// vanishes identically and the node's cross-axis curvature is exactly 0.
    fn agree_within(production: &Bound, oracle: &Bound, name: &str) -> bool {
        let bar = production.bar(oracle);
        assert!(
            (production.value - oracle.value).abs() <= bar,
            "{name}: production {}, oracle {}, bar {bar}",
            production.value,
            oracle.value
        );
        oracle.value.abs() > bar
    }

    #[test]
    fn observation_forces_and_curvatures_are_derivatives_of_the_complete_density() {
        let (model, h, theta, reference) = fixture();
        let nodes = h.times.len();
        let k = 2;
        let genes = [0.35];
        let states: Vec<f64> = (0..nodes * k)
            .map(|i| 0.4 * ((2 * (i / k) + 3 * (i % k)) as f64).cos())
            .collect();
        let exact: Vec<Bound> = theta.iter().map(|&v| Bound::exact(v)).collect();
        let bounded_states: Vec<Bound> = states.iter().map(|&v| Bound::exact(v)).collect();
        let bounded_reference: Vec<Bound> = reference.iter().map(|&v| Bound::exact(v)).collect();
        let observation = model
            .observation(&exact, &h, &bounded_states, &bounded_reference)
            .unwrap();
        let path: Vec<f64> = genes.iter().chain(states.iter()).copied().collect();
        let coefficients: Vec<Two> = theta.iter().map(|&v| two(v, 0.0, 0.0)).collect();
        let moments: Vec<Two> = reference.iter().map(|&v| two(v, 0.0, 0.0)).collect();
        let jet = |i: usize, j: usize| {
            let seeded: Vec<Two> = path
                .iter()
                .enumerate()
                .map(|(p, &v)| two(v, f64::from(p == i), f64::from(p == j)))
                .collect();
            model
                .path_density(&coefficients, &h, &seeded, &moments, false)
                .unwrap()
        };
        let index = |n: usize, a: usize| genes.len() + n * k + a;
        for n in 0..nodes {
            // A node with no observation factor is exactly zero on production and
            // carries no floor.
            let mut resolved = observation.curvature[n].iter().all(|c| c.value == 0.0)
                && (0..k).all(|a| observation.gradient[n * k + a].value == 0.0);
            for a in 0..k {
                for m in 0..nodes {
                    for b in 0..k {
                        let value = jet(index(n, a), index(m, b));
                        let second = &value.rows[0].rows[0];
                        if m != n {
                            // No factor couples two nodes.
                            assert_eq!(second.value, 0.0, "coupling {n},{a},{m},{b}");
                            continue;
                        }
                        resolved |= agree_within(
                            &observation.curvature[n][a * k + b].neg(),
                            second,
                            &format!("curvature {n},{a},{b}"),
                        );
                        if b == a {
                            resolved |= agree_within(
                                &observation.gradient[n * k + a],
                                &value.base.rows[0],
                                &format!("gradient {n},{a}"),
                            );
                        }
                    }
                }
            }
            assert!(resolved, "node {n}: every observation entry is below its bar");
            // The metric sums positive-semidefinite terms: its diagonal sums
            // nonnegative numbers, and its determinant can only round below zero.
            let metric = &observation.metric[n];
            assert!(metric[0].value >= 0.0 && metric[3].value >= 0.0);
            let determinant = metric[0].mul(&metric[3]).sub(&metric[1].mul(&metric[2]));
            assert!(
                determinant.value >= -determinant.rounding(),
                "metric determinant {} below its bound {}",
                determinant.value,
                determinant.rounding()
            );
            // The excess, accumulated from its own summands, is what the metric
            // drops from the observed curvature.
            for index in 0..k * k {
                let dropped = observation.metric[n][index].sub(&observation.curvature[n][index]);
                agree_within(&observation.excess[n][index], &dropped, &format!("excess {n},{index}"));
            }
        }
    }

    #[test]
    fn innovation_score_differentiates_the_profile_at_fixed_latent_coordinates() {
        // The oracle differentiates path_density(dynamics = false) - |e|^2 / 2
        // along each coefficient and reference moment by jets, through the jet
        // transport route that forms every rate factor from the jet. The raw
        // rates run from an ordinary value to both representable edges: a decay
        // leaving phi between 1e-261 and 1e-130 over the fixture's gaps, and a
        // rate near 4e-18 leaving phi next to 1. The production route is
        // instrumented by Bound; at the production running-error scalar, its own
        // bound must cover its distance to the oracle.
        type One = Rows<Bound, 1>;
        let (model, h, mut theta, reference) = fixture();
        let k = 2;
        let nodes = h.times.len();
        let genes = [0.35];
        let innovations: Vec<f64> = (0..nodes * k)
            .map(|i| 0.6 * ((3 * (i / k) + 2 * (i % k)) as f64).sin())
            .collect();
        let exact = |values: &[f64]| -> Vec<Bound> { values.iter().map(|&v| Bound::exact(v)).collect() };
        let running = |values: &[f64]| -> Vec<Running> { values.iter().map(|&v| Running::exact(v)).collect() };
        let seeded = |values: &[f64], at: Option<usize>| -> Vec<One> {
            values
                .iter()
                .enumerate()
                .map(|(i, &v)| Rows::seed(Bound::exact(v), [f64::from(at == Some(i))]))
                .collect()
        };
        let covered = |banded: &Running, oracle: &Bound, name: &str| {
            let bar = banded.rounding() + oracle.rounding();
            assert!(
                (banded.value - oracle.value).abs() <= bar,
                "{name}: running {}, oracle {}, bar {bar}",
                banded.value,
                oracle.value
            );
        };
        for raw in [0.4, 3000.0, -40.0] {
            for axis in 0..k {
                theta[model.layout.rates.start + axis] = raw + 0.3 * axis as f64;
            }
            let production = model
                .innovation_score(
                    &exact(&theta),
                    &h,
                    &exact(&genes),
                    &exact(&innovations),
                    &exact(&reference),
                )
                .unwrap();
            let banded = model
                .innovation_score(
                    &running(&theta),
                    &h,
                    &running(&genes),
                    &running(&innovations),
                    &running(&reference),
                )
                .unwrap();
            let profile = |coefficients: &[One], moments: &[One]| -> One {
                let latent_genes = seeded(&genes, None);
                let latent_innovations = seeded(&innovations, None);
                let states = model.transport_path(coefficients, &h, &latent_genes, &latent_innovations);
                let path: Vec<One> = latent_genes.iter().chain(&states).cloned().collect();
                let squares = latent_innovations
                    .iter()
                    .fold(coefficients[0].constant_like(0.0), |sum, e| sum.add(&e.mul(e)));
                model
                    .path_density(coefficients, &h, &path, moments, false)
                    .unwrap()
                    .sub(&squares.scale(0.5))
            };
            let base = profile(&seeded(&theta, None), &seeded(&reference, None));
            assert!(
                agree_within(&production.value, &base.base, &format!("raw {raw} profile value")),
                "raw {raw}: the profile value is below its bar"
            );
            covered(&banded.value, &base.base, &format!("raw {raw} running profile value"));
            let mut resolved = false;
            for j in 0..theta.len() {
                let oracle = profile(&seeded(&theta, Some(j)), &seeded(&reference, None));
                let name = format!("raw {raw} coefficient {j}");
                resolved |= agree_within(&production.coefficients[j], &oracle.rows[0], &name);
                covered(&banded.coefficients[j], &oracle.rows[0], &name);
            }
            assert!(resolved, "raw {raw}: every coefficient score entry is below its bar");
            let mut resolved = false;
            for i in 0..reference.len() {
                let oracle = profile(&seeded(&theta, None), &seeded(&reference, Some(i)));
                let name = format!("raw {raw} reference {i}");
                resolved |= agree_within(&production.reference[i], &oracle.rows[0], &name);
                covered(&banded.reference[i], &oracle.rows[0], &name);
            }
            assert!(resolved, "raw {raw}: every reference adjoint entry is below its bar");
        }
    }

    /// A K = 1 Laplace integral and its independent product-grid reference.
    /// Each replicate adds four probit outcomes at every node, three in one
    /// category and one in the other. The node likelihood then keeps an
    /// interior mode and its curvature grows with the replicates; equal outcomes
    /// would instead push the mode toward saturation, where `log Phi` flattens.
    /// Returns the grid integral, the grid's order-refinement error, the Laplace
    /// integral, and the second-order Laplace term.
    fn k1_laplace_against_the_product_grid(replicates: usize) -> (f64, f64, f64, f64) {
        use crate::chain::{AtomTransition, GaussHermite, Grid};
        use crate::marginal::{NodeLikelihood, filter_start, filter_step};
        type Four = Rows<Rows<Rows<Rows<f64, 1>, 1>, 1>, 1>;
        let four = |value: f64, a: f64, b: f64, c: f64, d: f64| -> Four {
            Rows::seed(Rows::seed(Rows::seed(Rows::seed(value, [a]), [b]), [c]), [d])
        };
        let model = JointLikelihood::new(JointSpecification {
            signatures: 1,
            marks: vec![MarkKind::Recurrent],
            baseline_columns: 1,
            population_columns: 1,
            drive_columns: 1,
            entry_columns: 0,
            baseline_penalties: vec![],
            drive_penalties: vec![],
            population_penalties: vec![],
            measurements: vec![MeasurementFamily::Probit { categories: 2 }],
            genetic_mean: vec![],
            genetic_precision: Array2::zeros((0, 0)),
        })
        .unwrap();
        let nodes = 6;
        let times: Vec<f64> = (0..nodes).map(|n| 0.4 * n as f64).collect();
        let (points, rows) = quadrature(&times, 1);
        let point_count = points.len();
        let h = JointHistory {
            times,
            points,
            events: vec![vec![], vec![], vec![0], vec![], vec![], vec![]],
            initially_at_risk: vec![true],
            baseline_design: rows.column(0).to_owned().insert_axis(ndarray::Axis(1)),
            population_design: rows.column(0).to_owned().insert_axis(ndarray::Axis(1)),
            drive_design: Array2::ones((nodes - 1, 1)),
            entry_design: vec![],
            genetics: vec![],
            measurements: (0..4 * replicates * nodes)
                .map(|index| {
                    let node = index % nodes;
                    let majority = index / nodes % 4 != 3;
                    MeasurementRecord {
                        node,
                        channel: 0,
                        value: Some(f64::from(majority == (node % 2 == 1))),
                        exposure: None,
                        after_event: false,
                    }
                })
                .collect(),
        };
        // Zero entry mean, drive and jump: the state law is the unit OU atom of chain.rs.
        let mut theta = vec![0.0; model.layout.width];
        theta[model.layout.baseline.start] = 0.2;
        theta[model.layout.decoder.start] = 0.8;
        theta[model.layout.rates.start] = 0.3;
        theta[model.layout.measurement_location[0].start] = 0.2;
        theta[model.layout.measurement_location[0].start + 1] = 0.8;
        let reference = vec![0.1; point_count];
        let posterior = model.laplace_posterior(&theta, &h, &reference, None).unwrap();

        // Node-separable observation factors by telescoping the law's density.
        let base = model
            .path_density(&theta, &h, &vec![0.0; nodes], &reference, false)
            .unwrap();
        let terms = |n: usize, grid: &Grid<f64>| {
            let ell: Vec<f64> = (0..grid.size())
                .map(|i| {
                    let mut path = vec![0.0; nodes];
                    path[n] = *grid.coordinate(i, 0);
                    model.path_density(&theta, &h, &path, &reference, false).unwrap() - base
                })
                .collect();
            let shift = ell.iter().copied().fold(f64::NEG_INFINITY, f64::max);
            NodeLikelihood {
                score: vec![],
                curvature: vec![],
                informative: vec![false],
                ell,
                shift,
            }
        };
        let grid_integral = |order: usize| {
            let gh = GaussHermite::new(order).unwrap();
            let like = 0.0_f64;
            let start = |grid: &Grid<f64>, derivatives: bool| {
                assert!(!derivatives);
                terms(0, grid)
            };
            let mut node = filter_start(&gh, &like, 1, &start, false, "k1 reference").unwrap();
            let mut total = node.normaliser.ln() + node.likelihood.shift;
            for n in 1..nodes {
                let kappa = emission::softplus(&theta[model.layout.rates.start])
                    * (h.times[n] - h.times[n - 1]);
                let step = |grid: &Grid<f64>, derivatives: bool| {
                    assert!(!derivatives);
                    terms(n, grid)
                };
                node = filter_step(
                    &gh,
                    &like,
                    &node.grid,
                    &node.alpha,
                    vec![AtomTransition::new(&kappa)],
                    0,
                    &step,
                    false,
                    "k1 reference",
                )
                .unwrap();
                total += node.normaliser.ln() + node.likelihood.shift;
            }
            total + base
        };
        let coarse = grid_integral(24);
        let exact = grid_integral(40);
        let grid_error = (coarse - exact).abs();

        // Dense Hessian and node-local third and fourth derivatives at the mode.
        let mode: Vec<f64> = posterior.states.column(0).to_vec();
        let coefficients: Vec<Four> = theta.iter().map(|&v| four(v, 0.0, 0.0, 0.0, 0.0)).collect();
        let moments: Vec<Four> = reference.iter().map(|&v| four(v, 0.0, 0.0, 0.0, 0.0)).collect();
        let jet = |i: usize, j: usize, inner: usize| {
            let seeded: Vec<Four> = mode
                .iter()
                .enumerate()
                .map(|(p, &v)| {
                    let at = f64::from(p == inner);
                    four(v, at, at, f64::from(p == i), f64::from(p == j))
                })
                .collect();
            model
                .log_density(&coefficients, &h, &seeded, &moments)
                .unwrap()
        };
        let mut precision = Array2::<f64>::zeros((nodes, nodes));
        let mut third = vec![0.0; nodes];
        let mut fourth = vec![0.0; nodes];
        for i in 0..nodes {
            for j in 0..nodes {
                precision[[i, j]] = -jet(i, j, nodes).rows[0].rows[0].base.base;
            }
            let local = jet(i, i, i);
            third[i] = local.rows[0].rows[0].base.rows[0];
            fourth[i] = local.rows[0].rows[0].rows[0].rows[0];
        }
        let covariance = super::super::precision::Cholesky::new(&precision)
            .unwrap()
            .inverse();
        let mut correction = 0.0;
        for n in 0..nodes {
            correction += fourth[n] * covariance[[n, n]].powi(2) / 8.0;
            for m in 0..nodes {
                correction += third[n] * third[m] * covariance[[n, m]]
                    * (covariance[[n, n]] * covariance[[m, m]] / 8.0
                        + covariance[[n, m]].powi(2) / 12.0);
            }
        }
        (exact, grid_error, posterior.log_marginal, correction)
    }

    #[test]
    fn k1_laplace_agrees_with_the_product_grid_integral_to_its_second_order_term() {
        let mut previous = f64::INFINITY;
        for replicates in [1, 8] {
            let (exact, grid_error, laplace, correction) =
                k1_laplace_against_the_product_grid(replicates);
            let error = exact - laplace;
            assert!(exact.abs() > 1.0, "integral magnitude {exact}");
            assert!(
                grid_error < correction.abs(),
                "the product grid does not resolve the second-order term: refinement error {grid_error}, term {correction}"
            );
            assert!(
                error.abs() > grid_error,
                "the Laplace error {error} is below the grid resolution {grid_error}"
            );
            // The asymptotic error estimate is the second-order term itself: the
            // remainder beyond it is bounded by its own magnitude plus the grid's
            // measured refinement error, with no scale factor.
            assert!(
                (error - correction).abs() <= correction.abs() + grid_error,
                "replicates {replicates}: grid {exact}, Laplace {laplace}, error {error}, second-order term {correction}, grid error {grid_error}"
            );
            // The estimate is in its asymptotic regime: the term shrinks as the
            // likelihood sharpens.
            assert!(
                correction.abs() < previous,
                "the second-order term does not shrink as the likelihood sharpens: {correction} after {previous}"
            );
            previous = correction.abs();
        }
    }
}
