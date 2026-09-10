//! A normalized prior on the decoder's final function shape. At fixed time
//! write F(x)=c[pi_0+sum pi_k softplus(x_k)]. Its lower asymptote divided by
//! that asymptote plus its upper-axis slopes is pi_0, independent of c.
//! Consequently -log(pi_0) is a function penalty that removes latent disease
//! dependence without penalizing the marginal baseline rate.
use super::*;

/// Dirichlet(1+lambda,1,...,1) on each mark's simplex of function weights,
/// including its exact normalizer and the Jacobian to decoder logit coordinates.
/// This is a prior evaluation, not a posterior estimate or marginal evidence.
pub struct DecoderPriorEvaluation {
    log_density: f64,
    gradient: Vec<f64>,
    decoder_start: usize,
    weights: Array2<f64>,
    scaled_weights: Array2<f64>,
    background_weights: Vec<f64>,
    log_scaled_background_weights: Vec<f64>,
    log_strengths: Vec<f64>,
    strength_gradient: Vec<f64>,
    strength_second: Vec<f64>,
    coefficient_strength_cross: Array2<f64>,
    weighted_penalty: Vec<f64>,
    mean_weights: Array2<f64>,
}

impl DecoderPriorEvaluation {
    pub fn log_density(&self) -> f64 {
        self.log_density
    }
    /// Full coefficient order; non-decoder coordinates are exactly zero.
    pub fn gradient(&self) -> &[f64] {
        &self.gradient
    }
    /// Apply the negative log-density Hessian in linear storage. Each mark's
    /// block is diagonal minus rank one; no marks x signatures^2 allocation.
    pub fn negative_hessian_product(
        &self,
        direction: &[f64],
    ) -> Result<Vec<f64>, EventHistoryError> {
        if direction.len() != self.gradient.len() || direction.iter().any(|v| !v.is_finite()) {
            return Err(invalid(
                "decoder prior Hessian direction has invalid dimensions or values",
            ));
        }
        let mut out = vec![0.0; direction.len()];
        let k = self.weights.ncols();
        for d in 0..self.weights.nrows() {
            let start = self.decoder_start + d * k;
            // Center on the most probable category before taking the mean.
            // In particular, if one signature probability rounds to one,
            // the background still contributes c*pi_0 rather than being
            // lost in `v - sum(pi*v)`. Scale before summation so a large
            // strength can rescue a representable tiny-probability term.
            let mut pivot_weight = self.background_weights[d];
            let mut pivot = 0.0;
            for j in 0..k {
                if self.weights[[d, j]] > pivot_weight {
                    pivot_weight = self.weights[[d, j]];
                    pivot = direction[start + j];
                }
            }
            let mut mean = if pivot == 0.0 {
                0.0
            } else {
                -(self.log_scaled_background_weights[d] + pivot.abs().ln()).exp() * pivot.signum()
            };
            let mut correction = 0.0;
            for j in 0..k {
                let contribution =
                    self.scaled_weights[[d, j]] * (direction[start + j] - pivot) - correction;
                let next = mean + contribution;
                correction = (next - mean) - contribution;
                mean = next;
            }
            for j in 0..k {
                out[start + j] = self.scaled_weights[[d, j]] * (direction[start + j] - pivot)
                    - self.weights[[d, j]] * mean;
            }
        }
        if out.iter().any(|v| !v.is_finite()) {
            return Err(numerical(
                "decoder prior Hessian product is not representable",
            ));
        }
        Ok(out)
    }
    pub fn log_strengths(&self) -> &[f64] {
        &self.log_strengths
    }
    /// Partial derivatives of log prior, not derivatives of marginal evidence.
    pub fn log_strength_gradient(&self) -> &[f64] {
        &self.strength_gradient
    }
    pub fn log_strength_second_derivative(&self) -> &[f64] {
        &self.strength_second
    }
    /// Mixed derivatives of log prior in each mark's logits and log strength.
    pub fn coefficient_strength_cross(&self) -> &Array2<f64> {
        &self.coefficient_strength_cross
    }
    pub fn weighted_function_penalty(&self) -> &[f64] {
        &self.weighted_penalty
    }
    /// Exact prior means, with the background in column zero. These are not
    /// posterior weights inferred from a dataset.
    pub fn mean_weights(&self) -> &Array2<f64> {
        &self.mean_weights
    }
}

impl JointLikelihood {
    /// Evaluate the decoder's function prior at theta and one log strength per
    /// mark. At K=0 the law is a point mass on a constant function, requiring
    /// no strength coordinates. Strength learning must integrate coefficients
    /// by REML/LAML; maximizing this joint density is not that calculation.
    pub fn decoder_prior(
        &self,
        theta: &[f64],
        log_strengths: &[f64],
    ) -> Result<DecoderPriorEvaluation, EventHistoryError> {
        self.validate_parameters(theta)?;
        let k = self.spec.signatures;
        let marks = self.spec.marks.len();
        if log_strengths.len() != if k == 0 { 0 } else { marks }
            || log_strengths.iter().any(|v| !v.is_finite())
        {
            return Err(invalid(
                "decoder prior requires one finite log strength per mark, or none at zero signatures",
            ));
        }
        let mut result = DecoderPriorEvaluation {
            log_density: 0.0,
            gradient: vec![0.0; theta.len()],
            decoder_start: self.layout.decoder.start,
            weights: Array2::zeros((marks, k)),
            scaled_weights: Array2::zeros((marks, k)),
            background_weights: vec![0.0; marks],
            log_scaled_background_weights: vec![0.0; marks],
            log_strengths: log_strengths.to_vec(),
            strength_gradient: vec![0.0; log_strengths.len()],
            strength_second: vec![0.0; log_strengths.len()],
            coefficient_strength_cross: Array2::zeros((marks, k)),
            weighted_penalty: vec![0.0; marks],
            mean_weights: Array2::zeros((marks, k + 1)),
        };
        if k == 0 {
            for d in 0..marks {
                result.mean_weights[[d, 0]] = 1.0;
            }
            return Ok(result);
        }
        for (d, &rho) in log_strengths.iter().enumerate() {
            let start = self.layout.decoder.start + d * k;
            let w = &theta[start..start + k];
            // Use shifted logits for individual simplex probabilities. Adding
            // log(K) to a very large common logit can lose that normalization.
            let shift = w.iter().copied().fold(0.0_f64, f64::max);
            let relative: Vec<_> = std::iter::once(-shift)
                .chain(w.iter().map(|&x| x - shift))
                .collect();
            let log_relative_sum = log_sum_exp(&relative);
            let log_pi_zero = -shift - log_relative_sum;
            result.background_weights[d] = log_pi_zero.exp();
            result.log_scaled_background_weights[d] =
                log_sum_exp(&[rho, ((k + 1) as f64).ln()]) - shift - log_relative_sum;
            let log_pi: Vec<_> = relative[1..].iter().map(|r| r - log_relative_sum).collect();
            let pi: Vec<_> = log_pi.iter().map(|v| v.exp()).collect();
            let log_activity_sum = log_sum_exp(w);
            let weighted = if log_activity_sum < 0.0 {
                // lambda*log1p(S) = sum exp(rho+w_k)*log1p(S)/S.
                // Shift rho into each term before summing: rho+log(S) would
                // lose log(K) when rho and the logits are large opposites.
                let ratio = emission::log_softplus(&log_activity_sum) - log_activity_sum;
                let terms: Vec<_> = w.iter().map(|&x| rho + x + ratio).collect();
                log_sum_exp(&terms).exp()
            } else {
                (rho + emission::log_softplus(&log_activity_sum)).exp()
            };
            let lambda_pi: Vec<_> = relative[1..]
                .iter()
                .map(|&r| (rho + r - log_relative_sum).exp())
                .collect();
            let scaled_pi: Vec<_> = lambda_pi
                .iter()
                .zip(&pi)
                .map(|(a, p)| a + (k + 1) as f64 * p)
                .collect();
            // C(lambda)=prod_{j=1}^K(lambda+j). Pair each normalizer term
            // with its log simplex weight before summing to avoid huge
            // cancelling totals in the no-latent-effect limit.
            let mut log_density = log_pi_zero - weighted;
            for axis in 0..k {
                let log_j = ((axis + 1) as f64).ln();
                log_density += log_sum_exp(&[rho, log_j]) + log_pi[axis];
                let p = (-emission::softplus(&(log_j - rho))).exp();
                result.strength_gradient[d] += p;
                result.strength_second[d] += p * (1.0 - p);
                result.gradient[start + axis] = 1.0 - scaled_pi[axis];
                result.coefficient_strength_cross[[d, axis]] = -lambda_pi[axis];
                result.weights[[d, axis]] = pi[axis];
                result.scaled_weights[[d, axis]] = scaled_pi[axis];
            }
            result.log_density += log_density;
            result.weighted_penalty[d] = weighted;
            result.strength_gradient[d] -= weighted;
            result.strength_second[d] -= weighted;
            let log_total = log_sum_exp(&[rho, ((k + 1) as f64).ln()]);
            result.mean_weights[[d, 0]] = (log_sum_exp(&[rho, 0.0]) - log_total).exp();
            for axis in 0..k {
                result.mean_weights[[d, axis + 1]] = (-log_total).exp();
            }
        }
        if !result.log_density.is_finite()
            || result
                .gradient
                .iter()
                .chain(&result.strength_gradient)
                .chain(&result.strength_second)
                .chain(result.coefficient_strength_cross.iter())
                .chain(result.scaled_weights.iter())
                .any(|v| !v.is_finite())
        {
            return Err(numerical(
                "decoder function prior or its derivatives are not representable",
            ));
        }
        Ok(result)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::scalar::Mixed;

    fn make_model(k: usize) -> JointLikelihood {
        JointLikelihood::new(JointSpecification {
            signatures: k,
            marks: vec![MarkKind::Recurrent],
            baseline_columns: 1,
            drive_columns: 1,
            entry_columns: 0,
            measurements: vec![],
            genetic_mean: vec![],
            genetic_precision: Array2::zeros((0, 0)),
        })
        .unwrap()
    }

    fn oracle<S: JetField>(w: &[S], rho: &S) -> S {
        let lambda = exp(rho);
        let mut numerator = rho.constant_like(0.0);
        let mut logits = vec![rho.constant_like(0.0)];
        for (j, x) in w.iter().enumerate() {
            numerator = numerator
                .add(&ln(&add_real(&lambda, (j + 1) as f64)))
                .add(x);
            logits.push(x.clone());
        }
        numerator.sub(&add_real(&lambda, (w.len() + 1) as f64).mul(&log_sum_exp(&logits)))
    }

    #[test]
    fn function_prior_includes_the_chart_jacobian_and_all_derivatives() {
        let model = make_model(3);
        let mut theta = vec![0.0; model.layout.width];
        let start = model.layout.decoder.start;
        theta[start..start + 3].copy_from_slice(&[-2.0, 0.2, 1.3]);
        for rho in [-5.0, 0.4, 7.0] {
            let out = model.decoder_prior(&theta, &[rho]).unwrap();
            for i in 0..4 {
                for j in 0..4 {
                    let w: Vec<_> = theta[start..start + 3]
                        .iter()
                        .enumerate()
                        .map(|(q, &v)| Mixed::seed(v, f64::from(q == i), f64::from(q == j)))
                        .collect();
                    let jet = oracle(&w, &Mixed::seed(rho, f64::from(i == 3), f64::from(j == 3)));
                    let g = if i == 3 {
                        out.log_strength_gradient()[0]
                    } else {
                        out.gradient()[start + i]
                    };
                    let h = if i == 3 && j == 3 {
                        out.log_strength_second_derivative()[0]
                    } else if i == 3 {
                        out.coefficient_strength_cross()[[0, j]]
                    } else if j == 3 {
                        out.coefficient_strength_cross()[[0, i]]
                    } else {
                        let mut direction = vec![0.0; theta.len()];
                        direction[start + j] = 1.0;
                        -out.negative_hessian_product(&direction).unwrap()[start + i]
                    };
                    for (a, b) in [(out.log_density(), jet.base), (g, jet.u), (h, jet.uv)] {
                        assert!(
                            (a - b).abs() < 1e-10 * (1.0 + b.abs()),
                            "rho={rho}, ({i},{j}): {a} vs {b}"
                        );
                    }
                }
            }
        }
    }

    #[test]
    fn function_prior_normalization_and_means_match_simplex_integration() {
        let model = make_model(2);
        let mut theta = vec![0.0; model.layout.width];
        let rho = 3.0_f64.ln();
        let (nodes, weights) = gam_math::special::gauss_legendre(64);
        let mut mass = 0.0;
        let mut mean = [0.0; 3];
        let mut strength_score = 0.0;
        for (&x, &wx) in nodes.iter().zip(&weights) {
            let p1 = (x + 1.0) * 0.5;
            for (&y, &wy) in nodes.iter().zip(&weights) {
                let p2 = (y + 1.0) * 0.5 * (1.0 - p1);
                let p0 = 1.0 - p1 - p2;
                theta[model.layout.decoder.start] = p1.ln() - p0.ln();
                theta[model.layout.decoder.start + 1] = p2.ln() - p0.ln();
                let out = model.decoder_prior(&theta, &[rho]).unwrap();
                // Remove the logit-chart Jacobian to integrate in the
                // physical simplex; then include this triangle's Jacobian.
                let weight = 0.25
                    * wx
                    * wy
                    * (1.0 - p1)
                    * (out.log_density() - p0.ln() - p1.ln() - p2.ln()).exp();
                mass += weight;
                for (j, p) in [p0, p1, p2].iter().enumerate() {
                    mean[j] += weight * p;
                }
                strength_score += weight * out.log_strength_gradient()[0];
            }
        }
        let prior = model.decoder_prior(&theta, &[rho]).unwrap();
        assert!((mass - 1.0).abs() < 1e-12);
        for j in 0..3 {
            assert!((mean[j] - prior.mean_weights()[[0, j]]).abs() < 1e-12);
        }
        assert!((mean[0] - 4.0 / 6.0).abs() < 1e-12);
        assert!((mean[1] - 1.0 / 6.0).abs() < 1e-12);
        assert!(strength_score.abs() < 1e-9);
    }

    #[test]
    fn function_penalty_is_invariant_to_intensity_scale_and_leaves_baselines_free() {
        let model = make_model(2);
        let mut theta = vec![0.0; model.layout.width];
        theta[model.layout.decoder.start] = -0.3;
        theta[model.layout.decoder.start + 1] = 0.5;
        let rho = 0.7;
        let prior = model.decoder_prior(&theta, &[rho]).unwrap();
        for scale in [1e-30, 0.2, 7.0, 1e30] {
            let low = scale
                * model
                    .log_relative_activity(&theta, 0, &[-40.0, -40.0])
                    .unwrap()
                    .exp();
            let mut total = low;
            for axis in 0..2 {
                let mut x = [-40.0, -40.0];
                x[axis] = 40.0;
                let left = scale * model.log_relative_activity(&theta, 0, &x).unwrap().exp();
                x[axis] = 50.0;
                let right = scale * model.log_relative_activity(&theta, 0, &x).unwrap().exp();
                total += (right - left) / 10.0;
            }
            let penalty = -rho.exp() * (low / total).ln();
            assert!((penalty - prior.weighted_function_penalty()[0]).abs() < 1e-12);
        }
        theta[0] = 14.0;
        let shifted = model.decoder_prior(&theta, &[rho]).unwrap();
        assert_eq!(prior.log_density(), shifted.log_density());
        assert_eq!(prior.gradient()[0], 0.0);
    }

    #[test]
    fn function_prior_hessian_retains_saturated_simplex_curvature() {
        let model = make_model(1);
        let mut theta = vec![0.0; model.layout.width];
        let coordinate = model.layout.decoder.start;
        let mut direction = vec![0.0; theta.len()];
        direction[coordinate] = 1.0;
        for (rho, logit, log_expected) in [(0.0, 40.0, 3.0_f64.ln() - 40.0), (700.0, 800.0, -100.0)]
        {
            theta[coordinate] = logit;
            let prior = model.decoder_prior(&theta, &[rho]).unwrap();
            let product = prior.negative_hessian_product(&direction).unwrap();
            // Closed form: (lambda+2)*exp(-w)/(1+exp(-w))^2.
            // Both examples round the signature's probability to one; in
            // the second exp(-w) itself underflows but the curvature does not.
            assert!((product[coordinate] / log_expected.exp() - 1.0).abs() < 1e-13);
        }
    }

    #[test]
    fn function_prior_preserves_strong_shrinkage_and_weak_strength_extremes() {
        let model = make_model(3);
        let mut theta = vec![0.0; model.layout.width];
        let start = model.layout.decoder.start;
        for rho in [800.0, 1e200] {
            theta[start..start + 3].fill(-rho);
            let out = model.decoder_prior(&theta, &[rho]).unwrap();
            assert!((out.log_density() + 3.0).abs() < 1e-12);
            assert!(out.log_strength_gradient()[0].abs() < 1e-12);
            assert!((out.log_strength_second_derivative()[0] + 3.0).abs() < 1e-12);
            assert!((out.weighted_function_penalty()[0] - 3.0).abs() < 1e-12);
            let mut direction = vec![0.0; theta.len()];
            direction[start] = 1.0;
            let product = out.negative_hessian_product(&direction).unwrap();
            assert!((product[start] - 1.0).abs() < 1e-12);
            assert!(out.gradient().iter().all(|v| v.abs() < 1e-12));
            assert_eq!(out.mean_weights()[[0, 0]], 1.0);
        }
        theta[start..start + 3].fill(1e200);
        let out = model.decoder_prior(&theta, &[-800.0]).unwrap();
        assert!(out.log_density().is_finite());
        assert!(out.log_strength_gradient()[0] < 0.0);
        assert!(out.log_strength_gradient()[0] > -1e-140);
        for j in 0..3 {
            assert!((out.gradient()[start + j] + 1.0 / 3.0).abs() < 1e-12);
        }
        assert!(model.decoder_prior(&theta, &[]).is_err());
        assert!(model.decoder_prior(&theta, &[f64::NAN]).is_err());
        let null = make_model(0);
        let coefficients = vec![0.0; null.layout.width];
        let prior = null.decoder_prior(&coefficients, &[]).unwrap();
        assert_eq!(prior.log_density(), 0.0);
        assert_eq!(prior.mean_weights()[[0, 0]], 1.0);
        assert!(prior.log_strengths().is_empty());
        assert!(null.decoder_prior(&coefficients, &[0.0]).is_err());
    }
}
