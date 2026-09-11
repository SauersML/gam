//! Uniform simplex measure on binary/ordinal baseline probabilities. In the
//! probit intercept/threshold chart this includes the complete CDF and gap
//! Jacobian. This is a normalized base prior, with no extra strength knob.
use super::*;

pub(super) struct CategoryPriors {
    pub(super) channels: Vec<(usize, Range<usize>)>,
}

struct ChannelEvaluation {
    intercept: usize,
    gaps: Range<usize>,
    z: Vec<f64>,
    log_slopes: Vec<f64>,
    log_complements: Vec<f64>,
}

pub(super) struct CategoryEvaluation {
    log_density: f64,
    channels: Vec<ChannelEvaluation>,
}

fn scaled_pair(a: f64, b: f64, log_scale: f64) -> f64 {
    if a == 0.0 || b == 0.0 {
        0.0
    } else {
        (a.abs().ln() + b.abs().ln() + log_scale).exp() * a.signum() * b.signum()
    }
}

impl CategoryPriors {
    pub(super) fn new(model: &JointLikelihood) -> Self {
        Self {
            channels: model
                .spec
                .measurements
                .iter()
                .enumerate()
                .filter_map(|(j, f)| {
                    matches!(
                        f,
                        MeasurementFamily::BinaryProbit | MeasurementFamily::OrdinalProbit { .. }
                    )
                    .then(|| {
                        (
                            model.layout.measurement_location[j].start,
                            model.layout.measurement_shape[j].clone(),
                        )
                    })
                })
                .collect(),
        }
    }
    pub(super) fn evaluate(&self, theta: &[f64], gradient: &mut [f64]) -> CategoryEvaluation {
        let mut out = CategoryEvaluation {
            log_density: 0.0,
            channels: Vec::with_capacity(self.channels.len()),
        };
        for (intercept, gaps) in &self.channels {
            let cuts = gaps.len() + 1;
            let mut value = ChannelEvaluation {
                intercept: *intercept,
                gaps: gaps.clone(),
                z: vec![-theta[*intercept]],
                log_slopes: Vec::with_capacity(gaps.len()),
                log_complements: Vec::with_capacity(gaps.len()),
            };
            let mut cut = 0.0;
            for q in gaps.clone() {
                cut += emission::softplus(&theta[q]);
                value.z.push(cut - theta[*intercept]);
                value.log_slopes.push(-emission::softplus(&(-theta[q])));
                value.log_complements.push(-emission::softplus(&theta[q]));
            }
            let norm = value.z.iter().fold(0.0_f64, |a, &b| a.hypot(b));
            out.log_density += (1..=cuts).map(|j| (j as f64).ln()).sum::<f64>()
                - 0.5 * cuts as f64 * (2.0 * std::f64::consts::PI).ln()
                - 0.5 * norm * norm
                + value.log_slopes.iter().sum::<f64>();
            gradient[*intercept] += value.z.iter().sum::<f64>();
            let mut suffix = 0.0;
            for j in (0..gaps.len()).rev() {
                suffix += value.z[j + 1];
                gradient[gaps.start + j] +=
                    value.log_complements[j].exp() - scaled(suffix, value.log_slopes[j]);
            }
            out.channels.push(value);
        }
        out
    }
}

impl CategoryEvaluation {
    pub(super) fn log_density(&self) -> f64 {
        self.log_density
    }
    pub(super) fn add_negative_hessian(&self, direction: &[f64], out: &mut [f64]) {
        for channel in &self.channels {
            let mut dz = vec![-direction[channel.intercept]];
            let mut change = -direction[channel.intercept];
            for (j, q) in channel.gaps.clone().enumerate() {
                change += scaled(direction[q], channel.log_slopes[j]);
                dz.push(change);
            }
            out[channel.intercept] -= dz.iter().sum::<f64>();
            let mut suffix = 0.0;
            let mut suffix_change = 0.0;
            for j in (0..channel.gaps.len()).rev() {
                suffix += channel.z[j + 1];
                suffix_change += dz[j + 1];
                let q = channel.gaps.start + j;
                out[q] += scaled(suffix_change, channel.log_slopes[j])
                    + scaled_pair(
                        direction[q],
                        suffix + 1.0,
                        channel.log_slopes[j] + channel.log_complements[j],
                    );
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn probit_category_base_measure_is_uniform_on_its_probability_simplex() {
        use gam_math::probability::standard_normal_quantile;
        let (nodes, weights) = gam_math::special::gauss_legendre(64);
        let prior = CategoryPriors {
            channels: vec![(0, 1..2)],
        };
        let mut mass = 0.0;
        let mut mean = [0.0; 3];
        for (&x, &wx) in nodes.iter().zip(&weights) {
            let p0 = 0.5 * (x + 1.0);
            for (&y, &wy) in nodes.iter().zip(&weights) {
                let p1 = 0.5 * (y + 1.0) * (1.0 - p0);
                let p2 = 1.0 - p0 - p1;
                let z0 = standard_normal_quantile(p0).unwrap();
                let z1 = standard_normal_quantile(p0 + p1).unwrap();
                let gap = z1 - z0;
                let q = gap + (-(-gap).exp_m1()).ln();
                let mut gradient = [0.0; 2];
                let value = prior.evaluate(&[-z0, q], &mut gradient);
                // Remove the independently computed CDF/chart Jacobian,
                // then include the triangle map from (x,y) to (p0,p1).
                let log_jacobian = -0.5 * (z0 * z0 + z1 * z1)
                    - (2.0 * std::f64::consts::PI).ln()
                    - emission::softplus(&(-q));
                let weight =
                    0.25 * wx * wy * (1.0 - p0) * (value.log_density() - log_jacobian).exp();
                mass += weight;
                for (j, p) in [p0, p1, p2].iter().enumerate() {
                    mean[j] += weight * p;
                }
            }
        }
        assert!((mass - 1.0).abs() < 1e-12);
        for value in mean {
            assert!((value - 1.0 / 3.0).abs() < 1e-12);
        }
    }

    #[test]
    fn category_chart_retains_curvature_from_an_underflowing_gap_slope() {
        let prior = CategoryPriors {
            channels: vec![(0, 1..2)],
        };
        let mut gradient = [0.0; 2];
        let value = prior.evaluate(&[1e150, -800.0], &mut gradient);
        assert!(value.log_density().is_finite());
        assert!(gradient.iter().all(|v| v.is_finite()));
        let mut product = [0.0; 2];
        value.add_negative_hessian(&[0.0, 1.0], &mut product);
        let expected = -(1e150_f64.ln() - 800.0).exp();
        assert!((product[1] / expected - 1.0).abs() < 1e-12);
    }
}
