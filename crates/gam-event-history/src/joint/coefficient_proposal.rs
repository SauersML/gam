//! Normalized coefficient proposals from the declared function priors. Flat
//! Student-t location coordinates receive proper conditional Cauchy proposals;
//! this changes the proposal measure, never the fitted prior or likelihood.
use super::*;
use rand::Rng;
use rand::distr::Open01;
use rand_distr::{Cauchy, Distribution, Gamma, StandardNormal};

struct LocationProposal {
    coordinate: usize,
    scale_coordinate: usize,
    center: f64,
    half_log_count: f64,
}

pub struct PriorCoefficientProposal<'p, 'm> {
    priors: &'p JointFunctionPriors<'m>,
    log_strengths: Vec<f64>,
    root_inverses: Vec<Array2<f64>>,
    gaussian_roots: Vec<usize>,
    locations: Vec<LocationProposal>,
    decoder_background: Vec<Gamma<f64>>,
    gamma_three: Gamma<f64>,
    gamma_five: Gamma<f64>,
    standard_cauchy: Cauchy<f64>,
    workspace_bytes: usize,
}

fn exponential<R: Rng + ?Sized>(rng: &mut R) -> f64 {
    let uniform: f64 = Open01.sample(rng);
    -uniform.ln()
}

/// Inverse softplus from log of its positive argument. Retains finite raw
/// coordinates when exp(log_value) would underflow, without clipping them.
fn inverse_log_softplus(log_value: f64) -> Result<f64, EventHistoryError> {
    let x = log_value.exp();
    let value = if x == 0.0 {
        log_value
    } else if x <= 1.0 {
        log_value + (x.exp_m1() / x).ln()
    } else {
        x + (-(-x).exp_m1()).ln()
    };
    if !value.is_finite() {
        return Err(numerical(
            "positive-function proposal coordinate is not representable",
        ));
    }
    Ok(value)
}

impl<'m> JointFunctionPriors<'m> {
    /// A normalized starting/defensive proposal. Locations use all observed
    /// measurements only to center a conditional Cauchy density with scale
    /// sigma/sqrt(count). No posterior coverage or efficiency claim follows
    /// from starting at the prior; data-adaptive proposals need pilot work.
    pub fn coefficient_proposal<'p>(
        &'p self,
        histories: &[&JointHistory],
        log_strengths: &[f64],
    ) -> Result<PriorCoefficientProposal<'p, 'm>, EventHistoryError> {
        if histories.is_empty()
            || log_strengths.len() != self.penalties.len()
            || log_strengths.iter().any(|v| !v.is_finite())
        {
            return Err(invalid(
                "coefficient proposal requires histories and one finite strength per prior function",
            ));
        }
        for history in histories {
            self.model.validate_history(history)?;
        }
        let mut unique: Vec<&Arc<FunctionRoot>> = Vec::new();
        let mut gaussian_roots = Vec::with_capacity(self.gaussian.len());
        let mut entries = 0usize;
        let mut largest = 0usize;
        for function in &self.gaussian {
            let index =
                if let Some(index) = unique.iter().position(|r| Arc::ptr_eq(r, &function.root)) {
                    index
                } else {
                    let size = function.root.root.len();
                    entries = entries
                        .checked_add(size)
                        .ok_or_else(|| invalid("proposal root size overflow"))?;
                    largest = largest.max(size);
                    unique.push(&function.root);
                    unique.len() - 1
                };
            gaussian_roots.push(index);
        }
        // Retained roots/inverses, the largest factorization's workspace and
        // scalar/coefficient buffers, before building any dense factor.
        let workspace_bytes = entries
            .checked_mul(2)
            .and_then(|n| n.checked_add(largest.checked_mul(8)?))
            .and_then(|n| n.checked_add(self.model.layout.width.checked_mul(32)?))
            .and_then(|n| n.checked_add(self.penalties.len().checked_mul(8)?))
            .and_then(|n| n.checked_mul(8))
            .ok_or_else(|| invalid("coefficient proposal workspace overflow"))?;
        if workspace_bytes > self.memory_limit_bytes {
            return Err(invalid(
                "coefficient proposal exceeds its function-prior memory budget",
            ));
        }
        let mut root_inverses = Vec::with_capacity(unique.len());
        for root in unique {
            use faer::prelude::*;
            let n = root.root.nrows();
            let matrix = faer::Mat::from_fn(n, n, |i, j| root.root[[i, j]]);
            let identity = faer::Mat::from_fn(n, n, |i, j| f64::from(i == j));
            let solved = matrix.as_ref().col_piv_qr().solve_lstsq(identity.as_ref());
            let inverse = Array2::from_shape_fn((n, n), |(i, j)| solved[(i, j)]);
            if inverse.iter().any(|v| !v.is_finite()) {
                return Err(numerical(
                    "function-prior root inverse is not representable",
                ));
            }
            root_inverses.push(inverse);
        }
        let mut locations = Vec::new();
        for (channel, family) in self.model.spec.measurements.iter().enumerate() {
            if !matches!(family, MeasurementFamily::StudentT) {
                continue;
            }
            let mut count = 0usize;
            let mut center = 0.0_f64;
            for value in histories
                .iter()
                .flat_map(|h| &h.measurements)
                .filter(|m| m.channel == channel)
                .filter_map(|m| m.value)
            {
                count = count
                    .checked_add(1)
                    .ok_or_else(|| invalid("measurement count overflow"))?;
                let weight = 1.0 / count as f64;
                center = if center.signum() == value.signum() {
                    center + (value - center) * weight
                } else {
                    center * (1.0 - weight) + value * weight
                };
            }
            if count == 0 {
                return Err(invalid(format!(
                    "Student-t channel {channel} has no observations; its flat location measure is not integrable"
                )));
            }
            locations.push(LocationProposal {
                coordinate: self.model.layout.measurement_location[channel].start,
                scale_coordinate: self.model.layout.measurement_shape[channel].start,
                center,
                half_log_count: 0.5 * (count as f64).ln(),
            });
        }
        let decoder_background = log_strengths[..self.decoder_strengths]
            .iter()
            .map(|rho| {
                let shape = 1.0 + rho.exp();
                if !shape.is_finite() {
                    return Err(numerical(
                        "decoder proposal Gamma shape is not representable",
                    ));
                }
                Gamma::new(shape, 1.0).map_err(|e| {
                    numerical(format!(
                        "decoder proposal Gamma shape is not representable: {e}"
                    ))
                })
            })
            .collect::<Result<Vec<_>, _>>()?;
        Ok(PriorCoefficientProposal {
            priors: self,
            log_strengths: log_strengths.to_vec(),
            root_inverses,
            gaussian_roots,
            locations,
            decoder_background,
            gamma_three: Gamma::new(3.0, 1.0).map_err(|e| invalid(e.to_string()))?,
            gamma_five: Gamma::new(5.0, 1.0).map_err(|e| invalid(e.to_string()))?,
            standard_cauchy: Cauchy::new(0.0, 1.0).map_err(|e| invalid(e.to_string()))?,
            workspace_bytes,
        })
    }
}

impl PriorCoefficientProposal<'_, '_> {
    pub(in crate::joint) fn workspace_bytes(&self) -> usize {
        self.workspace_bytes
    }
    pub fn log_strengths(&self) -> &[f64] {
        &self.log_strengths
    }

    /// Full normalized density in the same coefficient chart as the prior.
    /// Conditional Cauchy normalizers include the sampled residual scale.
    pub fn log_density(&self, theta: &[f64]) -> Result<f64, EventHistoryError> {
        let mut log_density = self
            .priors
            .evaluate(theta, &self.log_strengths)?
            .log_density();
        for location in &self.locations {
            let log_scale = theta[location.scale_coordinate] - location.half_log_count;
            let residual = theta[location.coordinate] - location.center;
            let log_relative = if residual == 0.0 {
                f64::NEG_INFINITY
            } else if residual.is_finite() {
                residual.abs().ln() - log_scale
            } else {
                log_sum_exp(&[
                    theta[location.coordinate].abs().ln(),
                    location.center.abs().ln(),
                ]) - log_scale
            };
            log_density -=
                std::f64::consts::PI.ln() + log_scale + emission::softplus(&(2.0 * log_relative));
        }
        if !log_density.is_finite() {
            return Err(numerical(
                "coefficient proposal density is not representable",
            ));
        }
        Ok(log_density)
    }

    pub fn draw<R: Rng + ?Sized>(
        &self,
        rng: &mut R,
    ) -> Result<CoefficientImportanceDraw, EventHistoryError> {
        let prior = self.priors;
        let model = prior.model;
        let mut theta = vec![0.0; model.layout.width];
        let k = model.spec.signatures;
        for (d, background) in self.decoder_background.iter().enumerate() {
            let log_background = background.sample(rng).ln();
            for axis in 0..k {
                theta[model.layout.decoder.start + d * k + axis] =
                    exponential(rng).ln() - log_background;
            }
        }
        // Structural scales precede Gaussian blocks that are conditional on
        // those scales. Unit-rate Gamma draws are converted in the log domain.
        let structural_start = prior.decoder_strengths + prior.gaussian.len() + 1;
        for (index, group) in prior.structural.iter().enumerate() {
            let rho = self.log_strengths[structural_start + index];
            for function in group {
                match *function {
                    StructuralFunction::CountMean { coordinate } => {
                        theta[coordinate] = exponential(rng).ln() - rho
                    }
                    StructuralFunction::TemporalVariation {
                        coordinate,
                        log_time,
                    } => {
                        theta[coordinate] =
                            inverse_log_softplus(exponential(rng).ln() - rho - log_time)?;
                    }
                    StructuralFunction::MeasurementPrecision { coordinate } => {
                        theta[coordinate] = 0.5 * (rho - self.gamma_three.sample(rng).ln());
                    }
                    StructuralFunction::InverseSoftplus {
                        coordinate,
                        log_multiplier,
                    } => {
                        theta[coordinate] = inverse_log_softplus(
                            log_multiplier + rho - self.gamma_five.sample(rng).ln(),
                        )?;
                    }
                }
            }
        }
        for (channel, family) in model.spec.measurements.iter().enumerate() {
            if !matches!(
                family,
                MeasurementFamily::BinaryProbit | MeasurementFamily::OrdinalProbit { .. }
            ) {
                continue;
            }
            let gaps = &model.layout.measurement_shape[channel];
            let mut masses: Vec<_> = (0..gaps.len() + 2).map(|_| exponential(rng)).collect();
            let total = masses.iter().sum::<f64>();
            for p in &mut masses {
                *p /= total;
            }
            let mut tails = vec![0.0; masses.len()];
            let mut tail = 0.0;
            for j in (0..masses.len()).rev() {
                tail += masses[j];
                tails[j] = tail;
            }
            let mut lower = 0.0;
            let mut previous = 0.0;
            for j in 0..masses.len() - 1 {
                lower += masses[j];
                // Use the smaller tail: forming 1-p would lose tiny final
                // category masses and manufacture infinite probit thresholds.
                let z = if lower <= 0.5 {
                    gam_math::probability::standard_normal_quantile(lower)
                } else {
                    gam_math::probability::standard_normal_quantile(tails[j + 1]).map(|v| -v)
                }
                .map_err(|e| numerical(e.to_string()))?;
                if j == 0 {
                    theta[model.layout.measurement_location[channel].start] = -z;
                } else {
                    theta[gaps.start + j - 1] = inverse_log_softplus((z - previous).ln())?;
                }
                previous = z;
            }
        }
        for location in &self.locations {
            let log_scale = theta[location.scale_coordinate] - location.half_log_count;
            theta[location.coordinate] =
                location.center + scaled(self.standard_cauchy.sample(rng), log_scale);
        }
        for (index, function) in prior.gaussian.iter().enumerate() {
            let inverse = &self.root_inverses[self.gaussian_roots[index]];
            let log_scale = function.log_scale.map_or(0.0, |q| theta[q])
                - 0.5 * self.log_strengths[prior.decoder_strengths + index];
            for coordinates in &function.coordinates {
                let z: Vec<f64> = (0..coordinates.len())
                    .map(|_| StandardNormal.sample(rng))
                    .collect();
                for i in 0..coordinates.len() {
                    let value = (0..z.len()).map(|j| inverse[[i, j]] * z[j]).sum();
                    theta[coordinates.start + i] = scaled(value, log_scale);
                }
            }
        }
        let rho = self.log_strengths[prior.decoder_strengths + prior.gaussian.len()];
        for d in 0..model.spec.marks.len() {
            let start = model.layout.baseline.start + d * model.spec.baseline_columns;
            let mean = (1..model.spec.baseline_columns)
                .map(|j| prior.baseline_mean_design[j] * theta[start + j])
                .sum::<f64>();
            theta[start] = exponential(rng).ln() - rho - prior.log_followup_scale - mean;
        }
        let log_proposal_density = self.log_density(&theta)?;
        Ok(CoefficientImportanceDraw {
            coefficients: theta,
            log_proposal_density,
        })
    }

    /// Independent draws with a checked allocation. Failed numerical draws
    /// are errors; they are never discarded/replaced by resampling, which
    /// would silently truncate the declared proposal law.
    pub fn draws<R: Rng + ?Sized>(
        &self,
        count: usize,
        rng: &mut R,
    ) -> Result<Vec<CoefficientImportanceDraw>, EventHistoryError> {
        let bytes = count
            .checked_mul(
                self.priors
                    .model
                    .layout
                    .width
                    .checked_mul(8)
                    .and_then(|n| n.checked_add(std::mem::size_of::<CoefficientImportanceDraw>()))
                    .ok_or_else(|| invalid("proposal draw size overflow"))?,
            )
            .and_then(|n| n.checked_add(self.workspace_bytes));
        if bytes.is_none_or(|n| n > self.priors.memory_limit_bytes) {
            return Err(invalid(
                "coefficient draws exceed their proposal memory budget",
            ));
        }
        (0..count).map(|_| self.draw(rng)).collect()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use rand::{SeedableRng, rngs::SmallRng};

    #[test]
    fn normalized_coefficient_draws_follow_all_declared_function_measures() {
        let (model, histories, _) = super::super::tests::fixture();
        let refs: Vec<_> = histories.iter().collect();
        let prior = model.function_priors(&refs, 64 << 20).unwrap();
        let rho: Vec<_> = (0..prior.penalties.len())
            .map(|j| 0.15 * (j as f64).sin())
            .collect();
        let proposal = prior.coefficient_proposal(&refs, &rho).unwrap();
        let mut rng = SmallRng::seed_from_u64(2011);
        let samples = 8192;
        let mut score_sum = vec![0.0; rho.len()];
        let mut score_square = vec![0.0; rho.len()];
        let coordinates: usize = prior
            .gaussian
            .iter()
            .map(|g| g.coordinates.iter().map(|r| r.len()).sum::<usize>())
            .sum();
        let mut gaussian_mean = vec![0.0; coordinates];
        let mut gaussian_square = vec![0.0; coordinates];
        let mut gaussian_cross = 0.0;
        let mut decoder_mean = vec![0.0; model.spec.marks.len()];
        let mut probability_mean = vec![0.0; 5];
        let mut location_pit = 0.0;
        for _ in 0..samples {
            let draw = proposal.draw(&mut rng).unwrap();
            let theta = &draw.coefficients;
            assert_eq!(
                draw.log_proposal_density,
                proposal.log_density(theta).unwrap()
            );
            let evaluation = prior.evaluate(theta, &rho).unwrap();
            for (j, &g) in evaluation.log_strength_gradient().iter().enumerate() {
                score_sum[j] += g;
                score_square[j] += g * g;
            }
            let mut index = 0;
            let mut cross_recorded = false;
            for (group, g) in prior.gaussian.iter().enumerate() {
                let log_scale = 0.5 * rho[prior.decoder_strengths + group]
                    - g.log_scale.map_or(0.0, |q| theta[q]);
                for r in &g.coordinates {
                    let white: Vec<f64> = g
                        .root
                        .root
                        .rows()
                        .into_iter()
                        .map(|row| {
                            scaled(
                                row.iter().zip(&theta[r.clone()]).map(|(a, b)| a * b).sum(),
                                log_scale,
                            )
                        })
                        .collect();
                    if !cross_recorded && white.len() > 1 {
                        gaussian_cross += white[0] * white[1];
                        cross_recorded = true;
                    }
                    for z in white {
                        gaussian_mean[index] += z;
                        gaussian_square[index] += z * z;
                        index += 1;
                    }
                }
            }
            assert!(cross_recorded);
            for (d, mean) in decoder_mean.iter_mut().enumerate() {
                let start = model.layout.decoder.start + d * model.spec.signatures;
                let logits: Vec<_> = std::iter::once(0.0)
                    .chain(theta[start..start + model.spec.signatures].iter().copied())
                    .collect();
                *mean += (-log_sum_exp(&logits)).exp();
            }
            probability_mean[0] += gam_math::probability::normal_cdf(
                theta[model.layout.measurement_location[1].start],
            );
            let mut previous = 0.0;
            let mut cut = 0.0;
            for j in 0..3 {
                if j > 0 {
                    cut +=
                        emission::softplus(&theta[model.layout.measurement_shape[2].start + j - 1]);
                }
                let cumulative = gam_math::probability::normal_cdf(
                    cut - theta[model.layout.measurement_location[2].start],
                );
                probability_mean[j + 1] += cumulative - previous;
                previous = cumulative;
            }
            probability_mean[4] += 1.0 - previous;
            let location = &proposal.locations[0];
            let standardized = scaled(
                theta[location.coordinate] - location.center,
                location.half_log_count - theta[location.scale_coordinate],
            );
            location_pit += 0.5 + standardized.atan() / std::f64::consts::PI;
        }
        let n = samples as f64;
        for j in 0..rho.len() {
            let mean = score_sum[j] / n;
            let se = ((score_square[j] / n - mean * mean) / (n - 1.0)).sqrt();
            assert!(
                mean.abs() < 6.0 * se,
                "prior score {:?}: {mean}, SE {se}",
                prior.penalties[j]
            );
        }
        for j in 0..coordinates {
            assert!((gaussian_mean[j] / n).abs() < 6.0 / n.sqrt());
            assert!((gaussian_square[j] / n - 1.0).abs() < 6.0 * (2.0 / n).sqrt());
        }
        assert!((gaussian_cross / n).abs() < 6.0 / n.sqrt());
        for (d, mean) in decoder_mean.iter().enumerate() {
            let expected =
                (1.0 + rho[d].exp()) / (1.0 + model.spec.signatures as f64 + rho[d].exp());
            assert!((mean / n - expected).abs() < 3.0 / n.sqrt());
        }
        for (j, mean) in probability_mean.iter().enumerate() {
            let expected = if j == 0 { 0.5 } else { 0.25 };
            assert!((mean / n - expected).abs() < 3.0 / n.sqrt());
        }
        assert!((location_pit / n - 0.5).abs() < 6.0 / (12.0 * n).sqrt());
    }

    #[test]
    fn coefficient_proposals_preserve_charts_reproducibility_and_error_contracts() {
        for log_value in [-800.0, -40.0, -1.0, 0.0, 4.0, 600.0] {
            let raw = inverse_log_softplus(log_value).unwrap();
            assert!((emission::log_softplus(&raw) - log_value).abs() < 1e-12);
        }
        assert!(inverse_log_softplus(800.0).is_err());
        let (model, mut histories, _) = super::super::tests::fixture();
        let prior = model
            .function_priors(&histories.iter().collect::<Vec<_>>(), 64 << 20)
            .unwrap();
        let rho = vec![0.0; prior.penalties.len()];
        let proposal = prior
            .coefficient_proposal(&histories.iter().collect::<Vec<_>>(), &rho)
            .unwrap();
        let left = proposal
            .draws(8, &mut SmallRng::seed_from_u64(2027))
            .unwrap();
        let right = proposal
            .draws(8, &mut SmallRng::seed_from_u64(2027))
            .unwrap();
        for (a, b) in left.iter().zip(&right) {
            assert_eq!(a.coefficients, b.coefficients);
            assert_eq!(a.log_proposal_density, b.log_proposal_density);
        }
        assert!(
            proposal
                .draws(usize::MAX, &mut SmallRng::seed_from_u64(1))
                .is_err()
        );
        assert!(proposal.log_density(&[0.0]).is_err());
        let mut wide = rho.clone();
        wide[0] = 800.0;
        assert!(
            prior
                .coefficient_proposal(&histories.iter().collect::<Vec<_>>(), &wide)
                .is_err()
        );
        for h in &mut histories {
            for m in &mut h.measurements {
                if m.channel == 0 {
                    m.value = None;
                }
            }
        }
        assert!(
            prior
                .coefficient_proposal(&histories.iter().collect::<Vec<_>>(), &rho)
                .err()
                .unwrap()
                .to_string()
                .contains("flat location")
        );
    }
}
