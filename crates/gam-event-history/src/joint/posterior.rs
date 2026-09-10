//! Conditional latent-mode inference and the corresponding Laplace integral.
//! Temporal Markov blocks and a genetic Schur complement replace the Cartesian
//! latent grid. The returned Gaussian is explicitly an approximation.
use super::precision::{Factorization, Precision};
use super::*;

#[derive(Clone, Debug)]
pub struct PosteriorOptions {
    /// Maximum Newton decrement, in local posterior standard deviations.
    pub mode_tolerance: f64,
    pub max_iterations: usize,
    pub memory_limit_bytes: usize,
}

impl Default for PosteriorOptions {
    fn default() -> Self {
        Self {
            mode_tolerance: 1e-7,
            max_iterations: 50,
            memory_limit_bytes: 512 * 1024 * 1024,
        }
    }
}

#[derive(Clone, Debug)]
pub struct LaplacePosterior {
    /// Missing genetic coordinates followed by node-major latent states.
    pub mode: Vec<f64>,
    pub state_covariance: Vec<Array2<f64>>,
    pub genetic_covariance: Array2<f64>,
    pub state_genetic_covariance: Vec<Array2<f64>>,
    /// Laplace approximation of the latent integral; not an exact evidence.
    pub log_marginal: f64,
    pub newton_decrement: f64,
    pub iterations: usize,
    /// Scalar entries in the undamped precision representation.
    pub precision_entries: usize,
}

pub(super) struct GaussianPrior {
    pub precision: Precision,
    pub information: Vec<f64>,
}

impl GaussianPrior {
    fn factor(
        &mut self,
        indices: &[usize],
        coefficients: &[f64],
        offset: f64,
        variance: f64,
    ) -> Result<(), EventHistoryError> {
        if !variance.is_finite()
            || variance <= 0.0
            || !offset.is_finite()
            || coefficients.iter().any(|c| !c.is_finite())
        {
            return Err(numerical("joint Gaussian factor is not finite and proper"));
        }
        for (a, &i) in indices.iter().enumerate() {
            self.information[i] -= offset * coefficients[a] / variance;
            for (b, &j) in indices.iter().enumerate() {
                self.precision
                    .add(i, j, coefficients[a] * coefficients[b] / variance)?;
            }
        }
        Ok(())
    }
}

impl JointLikelihood {
    pub(super) fn prior(
        &self,
        theta: &[f64],
        h: &JointHistory,
    ) -> Result<GaussianPrior, EventHistoryError> {
        let k = self.spec.signatures;
        let missing: Vec<usize> = h
            .genetics
            .iter()
            .enumerate()
            .filter_map(|(g, v)| v.is_none().then_some(g))
            .collect();
        let m = missing.len();
        let precision = Precision::new(h.times.len(), k, m);
        let mut prior = GaussianPrior {
            information: vec![0.0; precision.dimension()],
            precision,
        };
        // L L' is the full genetic precision, so L'(g-mu) has unit precision.
        // Observed scores enter the offset; missing scores retain their joint
        // conditional dependence through the dense border.
        for row in 0..h.genetics.len() {
            let offset: f64 = (0..h.genetics.len())
                .map(|g| {
                    self.genetic_factor[[g, row]]
                        * (h.genetics[g].unwrap_or(0.0) - self.spec.genetic_mean[g])
                })
                .sum();
            let coefficients: Vec<f64> = missing
                .iter()
                .map(|&g| self.genetic_factor[[g, row]])
                .collect();
            prior.factor(&(0..m).collect::<Vec<_>>(), &coefficients, offset, 1.0)?;
        }
        let entry = self.entry_features(h);
        let weights = |start: usize, columns: &[f64], axis: usize| -> (f64, Vec<f64>) {
            let g = h.genetics.len();
            let mut slopes = vec![0.0; g + 1];
            for (b, &feature) in columns.iter().enumerate() {
                for j in 0..=g {
                    slopes[j] += feature * theta[start + (axis * columns.len() + b) * (g + 1) + j];
                }
            }
            let offset = slopes[0]
                + h.genetics
                    .iter()
                    .enumerate()
                    .map(|(j, v)| slopes[j + 1] * v.unwrap_or(0.0))
                    .sum::<f64>();
            (offset, missing.iter().map(|&g| slopes[g + 1]).collect())
        };
        for axis in 0..k {
            let (offset, slopes) = weights(self.layout.entry.start, &entry, axis);
            let mut indices: Vec<usize> = (0..m).collect();
            indices.push(m + axis);
            let mut coefficients: Vec<f64> = slopes.iter().map(|s| -s).collect();
            coefficients.push(1.0);
            prior.factor(&indices, &coefficients, -offset, 1.0)?;
        }
        for n in 1..h.times.len() {
            let dt = h.times[n] - h.times[n - 1];
            let columns = h.drive_design.row(n - 1).to_vec();
            for axis in 0..k {
                let rate = emission::softplus(&theta[self.layout.rates.start + axis]);
                let phi = (-rate * dt).exp();
                let weight = -(-rate * dt).exp_m1();
                let variance = -(-2.0 * rate * dt).exp_m1();
                let (offset, slopes) = weights(self.layout.drive.start, &columns, axis);
                let jump = self.jump(theta, h.events[n - 1], axis);
                let mut indices: Vec<usize> = (0..m).collect();
                indices.extend([m + (n - 1) * k + axis, m + n * k + axis]);
                let mut coefficients: Vec<f64> = slopes.iter().map(|s| -weight * s).collect();
                coefficients.extend([-phi, 1.0]);
                prior.factor(
                    &indices,
                    &coefficients,
                    -weight * offset - phi * jump,
                    variance,
                )?;
            }
        }
        Ok(prior)
    }

    /// The non-Gaussian observation factors are node-local. Event curvature
    /// has a diagonal compensator term plus one rank-one event term, because
    /// the intensity is a positive sum of softplus activities.
    pub(super) fn observation_curvature(
        &self,
        theta: &[f64],
        h: &JointHistory,
        path: &[f64],
        reference: &[f64],
        gradient: &mut [f64],
        precision: &mut Precision,
    ) -> Result<(), EventHistoryError> {
        let k = self.spec.signatures;
        let m = precision.genes;
        let mut risk = h.initially_at_risk.clone();
        for n in 0..h.times.len() {
            let base = m + n * k;
            let x = &path[base..base + k];
            for d in 0..self.spec.marks.len() {
                if !risk[d] {
                    continue;
                }
                let weights = &theta
                    [self.layout.decoder.start + d * k..self.layout.decoder.start + (d + 1) * k];
                let mut numerator = vec![0.0];
                let mut denominator = vec![0.0];
                for axis in 0..k {
                    numerator.push(weights[axis] + emission::log_softplus(&x[axis]));
                    denominator.push(weights[axis]);
                }
                let log_numerator = log_sum_exp(&numerator);
                let baseline: f64 = (0..self.spec.baseline_columns)
                    .map(|b| {
                        theta[self.layout.baseline.start + d * self.spec.baseline_columns + b]
                            * h.baseline_design[[n, b]]
                    })
                    .sum();
                let log_rate = baseline + log_numerator
                    - log_sum_exp(&denominator)
                    - reference[n * self.spec.marks.len() + d];
                let compensator = if h.exposure[n] == 0.0 {
                    0.0
                } else {
                    log_rate.exp() * h.exposure[n]
                };
                let event = f64::from(h.events[n] == Some(d));
                let residual = event - compensator;
                let u: Vec<f64> = (0..k)
                    .map(|axis| {
                        (weights[axis] - emission::softplus(&(-x[axis])) - log_numerator).exp()
                    })
                    .collect();
                for axis in 0..k {
                    gradient[base + axis] += residual * u[axis];
                    precision.diagonal[n][[axis, axis]] -=
                        residual * u[axis] * (-emission::softplus(&x[axis])).exp();
                    if event != 0.0 {
                        for other in 0..k {
                            precision.diagonal[n][[axis, other]] += u[axis] * u[other];
                        }
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
            let base = m + record.node * k;
            let location = self.layout.measurement_location[record.channel].clone();
            let loadings = &theta[location.start + 1..location.end];
            let mut eta = theta[location.start];
            for axis in 0..k {
                let jump = if record.after_event {
                    self.jump(theta, h.events[record.node], axis)
                } else {
                    0.0
                };
                eta += loadings[axis] * (path[base + axis] + jump);
            }
            let (score, curvature) = emission::location_derivatives(
                &self.spec.measurements[record.channel],
                y,
                eta,
                &theta[self.layout.measurement_shape[record.channel].clone()],
            )?;
            for axis in 0..k {
                gradient[base + axis] += score * loadings[axis];
                for other in 0..k {
                    precision.diagonal[record.node][[axis, other]] -=
                        curvature * loadings[axis] * loadings[other];
                }
            }
        }
        if gradient.iter().any(|v| !v.is_finite())
            || precision
                .diagonal
                .iter()
                .flat_map(|d| d.iter())
                .any(|v| !v.is_finite())
        {
            return Err(numerical("non-finite joint observation curvature"));
        }
        Ok(())
    }

    /// Approximate the latent integral at fixed model parameters and reference
    /// evolution. Every step is evaluated against the complete joint density.
    /// Damping is a step device only: covariance and log determinant always
    /// use the undamped Hessian of the density at a stationary mode.
    pub fn laplace_posterior(
        &self,
        theta: &[f64],
        h: &JointHistory,
        reference: &[f64],
        initial: Option<&[f64]>,
        options: &PosteriorOptions,
    ) -> Result<LaplacePosterior, EventHistoryError> {
        self.validate_parameters(theta)?;
        let dimension = self.latent_dimension(h)?;
        if !options.mode_tolerance.is_finite()
            || options.mode_tolerance <= 0.0
            || options.max_iterations == 0
        {
            return Err(invalid(
                "joint posterior needs positive finite tolerance and an iteration bound",
            ));
        }
        let k = self.spec.signatures;
        let m = h.genetics.iter().filter(|g| g.is_none()).count();
        let row_width = k
            .checked_mul(2)
            .and_then(|v| v.checked_add(m))
            .ok_or_else(|| invalid("joint posterior row dimension overflow"))?;
        let entries = h
            .times
            .len()
            .checked_mul(k)
            .and_then(|v| v.checked_mul(row_width))
            .and_then(|v| m.checked_mul(m).and_then(|corner| v.checked_add(corner)))
            .ok_or_else(|| invalid("joint posterior storage overflow"))?;
        let bytes = entries
            .checked_mul(12 * std::mem::size_of::<f64>())
            .and_then(|v| {
                dimension
                    .checked_mul(10 * std::mem::size_of::<f64>())
                    .and_then(|vectors| v.checked_add(vectors))
            })
            .ok_or_else(|| invalid("joint posterior work storage overflow"))?;
        if bytes > options.memory_limit_bytes {
            return Err(numerical(format!(
                "joint posterior needs approximately {bytes} working bytes, above its {}-byte work limit",
                options.memory_limit_bytes
            )));
        }
        let prior = self.prior(theta, h)?;
        let mut mode = match initial {
            Some(x) if x.len() == dimension && x.iter().all(|v| v.is_finite()) => x.to_vec(),
            Some(_) => {
                return Err(invalid(
                    "joint posterior initial path has invalid dimensions or values",
                ));
            }
            None => Factorization::new(&prior.precision)?.solve(&prior.information),
        };
        let mut value = self.log_density(theta, h, &mode, reference)?;
        for iteration in 0..options.max_iterations {
            let applied = prior.precision.apply(&mode);
            let mut gradient: Vec<f64> = prior
                .information
                .iter()
                .zip(applied)
                .map(|(a, b)| a - b)
                .collect();
            let mut precision = prior.precision.clone();
            self.observation_curvature(theta, h, &mode, reference, &mut gradient, &mut precision)?;
            let scale = precision.damping_scale();
            let mut damping = 0.0;
            let mut accepted = false;
            for _ in 0..20 {
                let mut trial_precision = precision.clone();
                trial_precision.add_damping(damping);
                if let Ok(factor) = Factorization::new(&trial_precision) {
                    let step = factor.solve(&gradient);
                    let decrement: f64 = gradient.iter().zip(&step).map(|(a, b)| a * b).sum();
                    if !decrement.is_finite() || decrement < 0.0 {
                        return Err(numerical("joint Newton decrement is unresolved"));
                    }
                    if damping == 0.0 && decrement.sqrt() <= options.mode_tolerance {
                        let (state_covariance, genetic_covariance, state_genetic_covariance) =
                            factor.covariance();
                        return Ok(LaplacePosterior {
                            mode,
                            state_covariance,
                            genetic_covariance,
                            state_genetic_covariance,
                            log_marginal: value
                                + 0.5 * dimension as f64 * (2.0 * std::f64::consts::PI).ln()
                                - 0.5 * factor.log_determinant,
                            newton_decrement: decrement.sqrt(),
                            iterations: iteration,
                            precision_entries: precision.stored_entries(),
                        });
                    }
                    let mut fraction = 1.0;
                    for _ in 0..40 {
                        let proposal: Vec<f64> = mode
                            .iter()
                            .zip(&step)
                            .map(|(x, d)| x + fraction * d)
                            .collect();
                        match self.log_density(theta, h, &proposal, reference) {
                            Ok(candidate)
                                if candidate
                                    >= value + 1e-4 * fraction * decrement
                                        - 16.0 * f64::EPSILON * (1.0 + value.abs()) =>
                            {
                                mode = proposal;
                                value = candidate;
                                accepted = true;
                                break;
                            }
                            Ok(_) | Err(EventHistoryError::NumericalFailure { .. }) => (),
                            Err(error) => return Err(error),
                        }
                        fraction *= 0.5;
                    }
                }
                if accepted {
                    break;
                }
                damping = if damping == 0.0 {
                    1e-6 * scale
                } else {
                    damping * 10.0
                };
            }
            if !accepted {
                return Err(numerical(
                    "joint posterior mode has no resolved ascent step",
                ));
            }
        }
        Err(numerical(
            "joint posterior mode did not reach its Newton-decrement tolerance",
        ))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::scalar::Mixed;

    #[test]
    fn structured_precision_is_the_hessian_of_the_complete_density() {
        let model = JointLikelihood::new(JointSpecification {
            signatures: 1,
            marks: vec![MarkKind::Recurrent],
            baseline_columns: 1,
            drive_columns: 1,
            entry_columns: 0,
            genetic_mean: vec![0.2],
            genetic_precision: Array2::ones((1, 1)),
            measurements: vec![
                MeasurementFamily::BinaryProbit,
                MeasurementFamily::StudentT,
                MeasurementFamily::OrdinalProbit { categories: 4 },
                MeasurementFamily::NegativeBinomial,
            ],
        })
        .unwrap();
        let h = JointHistory {
            times: vec![0.0, 0.3, 0.6, 1.0],
            exposure: vec![0.0, 0.6, 0.0, 0.4],
            events: vec![None, None, Some(0), None],
            initially_at_risk: vec![true],
            baseline_design: Array2::ones((4, 1)),
            drive_design: Array2::ones((3, 1)),
            entry_design: vec![],
            genetics: vec![None],
            measurements: (0..4)
                .map(|channel| MeasurementRecord {
                    node: 2,
                    channel,
                    value: Some(1.0),
                    after_event: true,
                })
                .collect(),
        };
        let theta: Vec<f64> = (0..model.layout.width).map(|i| 0.04 * i as f64).collect();
        let path: Vec<f64> = (0..model.latent_dimension(&h).unwrap())
            .map(|i| 0.1 * i as f64)
            .collect();
        let prior = model.prior(&theta, &h).unwrap();
        let mut gradient: Vec<f64> = prior
            .information
            .iter()
            .zip(prior.precision.apply(&path))
            .map(|(a, b)| a - b)
            .collect();
        let mut precision = prior.precision.clone();
        model
            .observation_curvature(&theta, &h, &path, &[0.0; 4], &mut gradient, &mut precision)
            .unwrap();
        let coefficients: Vec<Mixed<f64>> =
            theta.iter().map(|&v| Mixed::seed(v, 0.0, 0.0)).collect();
        for i in 0..path.len() {
            let mut unit = vec![0.0; path.len()];
            unit[i] = 1.0;
            let column = precision.apply(&unit);
            for j in 0..path.len() {
                let states: Vec<Mixed<f64>> = path
                    .iter()
                    .enumerate()
                    .map(|(q, &v)| Mixed::seed(v, f64::from(q == i), f64::from(q == j)))
                    .collect();
                let value = model
                    .log_density(
                        &coefficients,
                        &h,
                        &states,
                        &vec![Mixed::seed(0.0, 0.0, 0.0); 4],
                    )
                    .unwrap();
                assert!((value.u - gradient[i]).abs() < 1e-12, "gradient {i}");
                assert!((value.uv + column[j]).abs() < 1e-12, "precision {i},{j}");
            }
        }
    }
}
