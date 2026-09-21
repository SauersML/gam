//! A normalized prior on the decoder's constant-function weights. At fixed time
//! and context write F(x)=c[pi_0+sum pi_k softplus(x_k)]. Its lower asymptote
//! divided by that asymptote plus its upper-axis slopes is pi_0, independent of
//! c. Consequently -log(pi_0) is a function penalty that removes latent disease
//! dependence without penalizing the marginal baseline rate. The law acts on
//! each logit function's constant coefficient, column zero of the population
//! basis. Variation over the other columns has its own frozen penalty, whose
//! strength limit leaves weights constant under this law. A structure may drop
//! connections: its support restricts each mark's law to the present categories,
//! which is the face conditional of the larger law at `pi_dk = 0`.
use super::emission;
use super::law::{JointLikelihood, invalid, numerical};
use crate::EventHistoryError;
use crate::chain::log_sum_exp;
use crate::scalar::{add_real, exp, ln};
use gam_math::nested_dual::JetField;

/// Dirichlet(1+lambda,1,...,1) on each mark's simplex of constant-function
/// weights over its present categories, including its exact normalizer and the
/// Jacobian to decoder logit coordinates. This is a prior evaluation, not a
/// posterior estimate or marginal evidence. Per-signature quantities are stored
/// mark-major; absent categories carry exact zeros.
pub struct DecoderPriorEvaluation<S> {
    log_density: S,
    gradient: Vec<S>,
    decoder_start: usize,
    signatures: usize,
    columns: usize,
    support: Vec<Vec<bool>>,
    weights: Vec<S>,
    scaled_weights: Vec<S>,
    lambda_weights: Vec<S>,
    background_weights: Vec<S>,
    log_scaled_background_weights: Vec<S>,
    log_lambda_background_weights: Vec<S>,
    log_strengths: Vec<S>,
    strength_gradient: Vec<S>,
    strength_second: Vec<S>,
    coefficient_strength_cross: Vec<S>,
    weighted_penalty: Vec<S>,
    mean_weights: Vec<S>,
}

fn sum_compensated<S: JetField>(zero: &S, values: impl Iterator<Item = S>) -> S {
    let mut total = zero.clone();
    let mut correction = zero.clone();
    for value in values {
        let contribution = value.sub(&correction);
        let next = total.add(&contribution);
        correction = next.sub(&total).sub(&contribution);
        total = next;
    }
    total
}

fn finite<S: JetField>(out: Vec<S>, reason: &str) -> Result<Vec<S>, EventHistoryError> {
    if out.iter().any(|v| !v.value().is_finite()) {
        return Err(numerical(reason));
    }
    Ok(out)
}

impl<S: JetField> DecoderPriorEvaluation<S> {
    pub fn log_density(&self) -> &S {
        &self.log_density
    }
    /// Full coefficient order; every coordinate but the present logit levels is exactly zero.
    pub fn gradient(&self) -> &[S] {
        &self.gradient
    }
    fn validate_direction(&self, direction: &[S]) -> Result<(), EventHistoryError> {
        if direction.len() != self.gradient.len() || direction.iter().any(|v| !v.value().is_finite())
        {
            return Err(invalid(
                "decoder prior direction has invalid dimensions or values",
            ));
        }
        Ok(())
    }
    fn marks(&self) -> usize {
        self.background_weights.len()
    }
    /// The level coefficient of mark `d`'s logit function for `axis`.
    fn coordinate(&self, d: usize, axis: usize) -> usize {
        self.decoder_start + (d * self.signatures + axis) * self.columns
    }
    /// Mark `d`'s present categories.
    fn present(&self, d: usize) -> Vec<usize> {
        (0..self.signatures).filter(|&axis| self.support[d][axis]).collect()
    }
    /// The most probable category of a mark: `None` is the background.
    fn pivot(&self, d: usize) -> Option<usize> {
        let k = self.signatures;
        let mut pivot_weight = self.background_weights[d].value();
        let mut pivot = None;
        for j in self.present(d) {
            if self.weights[d * k + j].value() > pivot_weight {
                pivot_weight = self.weights[d * k + j].value();
                pivot = Some(j);
            }
        }
        pivot
    }
    /// `c (diag(pi) - pi pi') v` on one mark's present logit levels, with `c pi`
    /// supplied as `scaled` and `log(c pi_0)` as `log_scaled_background`.
    fn simplex_product(
        &self,
        d: usize,
        scaled: &[S],
        log_scaled_background: &S,
        direction: &[S],
        out: &mut [S],
    ) {
        let k = self.signatures;
        let present = self.present(d);
        let zero = self.log_density.constant_like(0.0);
        // Center on the most probable category before taking the mean.
        // In particular, if one signature probability rounds to one,
        // the background still contributes c*pi_0 rather than being
        // lost in `v - sum(pi*v)`. Scale before summation so a large
        // strength can rescue a representable tiny-probability term.
        let pivot = self
            .pivot(d)
            .map_or_else(|| zero.clone(), |j| direction[self.coordinate(d, j)].clone());
        let mean = if pivot.value() == 0.0 {
            zero.clone()
        } else {
            let sign = pivot.value().signum();
            exp(&log_scaled_background.add(&ln(&pivot.scale(sign)))).scale(-sign)
        };
        let mean = sum_compensated(
            &zero,
            std::iter::once(mean).chain(
                present
                    .iter()
                    .map(|&j| scaled[d * k + j].mul(&direction[self.coordinate(d, j)].sub(&pivot))),
            ),
        );
        for &j in &present {
            out[self.coordinate(d, j)] = scaled[d * k + j]
                .mul(&direction[self.coordinate(d, j)].sub(&pivot))
                .sub(&self.weights[d * k + j].mul(&mean));
        }
    }
    /// Apply the negative log-density Hessian in linear storage. Each mark's
    /// block is diagonal minus rank one; no marks x signatures^2 allocation.
    pub fn negative_hessian_product(&self, direction: &[S]) -> Result<Vec<S>, EventHistoryError> {
        self.validate_direction(direction)?;
        let mut out = vec![self.log_density.constant_like(0.0); direction.len()];
        for d in 0..self.marks() {
            self.simplex_product(
                d,
                &self.scaled_weights,
                &self.log_scaled_background_weights[d],
                direction,
                &mut out,
            );
        }
        finite(out, "decoder prior Hessian product is not representable")
    }
    /// d/d rho of one mark's negative Hessian product: lambda (diag(pi) - pi pi') v.
    pub fn strength_hessian_product(
        &self,
        mark: usize,
        direction: &[S],
    ) -> Result<Vec<S>, EventHistoryError> {
        self.validate_direction(direction)?;
        if mark >= self.log_strengths.len() {
            return Err(invalid("decoder prior strength index is outside its marks"));
        }
        let mut out = vec![self.log_density.constant_like(0.0); direction.len()];
        self.simplex_product(
            mark,
            &self.lambda_weights,
            &self.log_lambda_background_weights[mark],
            direction,
            &mut out,
        );
        finite(out, "decoder prior strength curvature product is not representable")
    }
    /// `D(H v)[delta]` with `H` the negative log-density Hessian. Per mark this
    /// is (lambda+C+1) times the third derivative of log(1 + sum exp(w)) over the
    /// C present categories:
    /// `pi_i [(v_i - v_bar)(delta_i - delta_bar) - Cov_pi(v, delta)]`, where the
    /// background carries value zero. Centering on the most probable category
    /// keeps the small centered values exact when that probability rounds to one.
    pub fn third_derivative_product(
        &self,
        delta: &[S],
        direction: &[S],
    ) -> Result<Vec<S>, EventHistoryError> {
        self.validate_direction(delta)?;
        self.validate_direction(direction)?;
        let zero = self.log_density.constant_like(0.0);
        let mut out = vec![zero.clone(); direction.len()];
        let k = self.signatures;
        for d in 0..self.marks() {
            let present = self.present(d);
            let pivot = self.pivot(d);
            let centered = |values: &[S]| {
                let p = pivot.map_or_else(|| zero.clone(), |j| values[self.coordinate(d, j)].clone());
                let shift = sum_compensated(
                    &zero,
                    present
                        .iter()
                        .map(|&j| self.weights[d * k + j].mul(&values[self.coordinate(d, j)].sub(&p)))
                        .chain(std::iter::once(self.background_weights[d].mul(&p).neg())),
                );
                let signatures: Vec<S> = present
                    .iter()
                    .map(|&j| values[self.coordinate(d, j)].sub(&p).sub(&shift))
                    .collect();
                (p.neg().sub(&shift), signatures)
            };
            let (background_v, v) = centered(direction);
            let (background_delta, e) = centered(delta);
            let covariance = sum_compensated(
                &zero,
                present
                    .iter()
                    .enumerate()
                    .map(|(r, &j)| self.weights[d * k + j].mul(&v[r]).mul(&e[r]))
                    .chain(std::iter::once(
                        self.background_weights[d]
                            .mul(&background_v)
                            .mul(&background_delta),
                    )),
            );
            for (r, &j) in present.iter().enumerate() {
                out[self.coordinate(d, j)] =
                    self.scaled_weights[d * k + j].mul(&v[r].mul(&e[r]).sub(&covariance));
            }
        }
        finite(out, "decoder prior third derivative product is not representable")
    }
    pub fn log_strengths(&self) -> &[S] {
        &self.log_strengths
    }
    /// Partial derivatives of log prior, not derivatives of marginal evidence.
    pub fn log_strength_gradient(&self) -> &[S] {
        &self.strength_gradient
    }
    pub fn log_strength_second_derivative(&self) -> &[S] {
        &self.strength_second
    }
    /// Mixed derivatives of log prior in each mark's logit levels and log
    /// strength, mark-major.
    pub fn coefficient_strength_cross(&self) -> &[S] {
        &self.coefficient_strength_cross
    }
    pub fn weighted_function_penalty(&self) -> &[S] {
        &self.weighted_penalty
    }
    /// Exact prior means, mark-major with the background first in each mark.
    /// These are not posterior weights inferred from a dataset.
    pub fn mean_weights(&self) -> &[S] {
        &self.mean_weights
    }
}

/// `log C_C(lambda) = sum_{j<=C} log(lambda + j)`, the Dirichlet(1+lambda,1,...,1) normalizer of
/// a mark with C present categories, over S. The face density of a connection in the larger law is
/// `C_C(lambda) / C_(C-1)(lambda) = C + lambda`, with C counting the connected categories
/// including it.
pub(super) fn log_decoder_normalizer<S: JetField>(categories: usize, log_strength: &S) -> S {
    (1..=categories).fold(log_strength.constant_like(0.0), |sum, j| {
        sum.add(&log_sum_exp(&[
            log_strength.clone(),
            ln(&log_strength.constant_like(j as f64)),
        ]))
    })
}

impl JointLikelihood {
    /// Evaluate the decoder's function prior at theta and one log strength per
    /// mark, with every connection present. At K=0 the law is a point mass on a
    /// constant function, requiring no strength coordinates. Strength learning
    /// must integrate coefficients by REML/LAML; maximizing this joint density is
    /// not that calculation.
    pub fn decoder_prior<S: JetField>(
        &self,
        theta: &[S],
        log_strengths: &[S],
    ) -> Result<DecoderPriorEvaluation<S>, EventHistoryError> {
        let support = vec![vec![true; self.spec.signatures]; self.spec.marks.len()];
        self.supported_decoder_prior(theta, log_strengths, &support)
    }

    /// The decoder prior of a structure whose connections `support[mark][signature]`
    /// may be absent. Each mark's law is Dirichlet(1+lambda,1,...,1) over its background
    /// and its C present categories, with normalizer C_C(lambda) = prod_{j<=C} (lambda+j).
    /// That is the conditional of the larger law on the face `pi_dk = 0`: the larger law's
    /// marginal of `pi_dk` is Beta(1, C+lambda) with C its present count including k,
    /// whose density at zero is C_C(lambda)/C_(C-1)(lambda) = C + lambda. Absent logits are
    /// not coordinates of this prior, and the fit holds them out of integration. Every mark
    /// needs a present category: an empty row is a strength limit, not a support.
    pub fn supported_decoder_prior<S: JetField>(
        &self,
        theta: &[S],
        log_strengths: &[S],
        support: &[Vec<bool>],
    ) -> Result<DecoderPriorEvaluation<S>, EventHistoryError> {
        self.validate_parameters(theta)?;
        let k = self.spec.signatures;
        let marks = self.spec.marks.len();
        if log_strengths.len() != if k == 0 { 0 } else { marks }
            || log_strengths.iter().any(|v| !v.value().is_finite())
        {
            return Err(invalid(
                "decoder prior requires one finite log strength per mark, or none at zero signatures",
            ));
        }
        if support.len() != marks
            || support
                .iter()
                .any(|row| row.len() != k || (k > 0 && !row.iter().any(|&present| present)))
        {
            return Err(invalid(
                "a decoder support needs one row per mark over every signature, each with a present connection",
            ));
        }
        let zero = theta[0].constant_like(0.0);
        let mut result = DecoderPriorEvaluation {
            log_density: zero.clone(),
            gradient: vec![zero.clone(); theta.len()],
            decoder_start: self.layout.decoder.start,
            signatures: k,
            columns: self.spec.population_columns,
            support: support.to_vec(),
            weights: vec![zero.clone(); marks * k],
            scaled_weights: vec![zero.clone(); marks * k],
            lambda_weights: vec![zero.clone(); marks * k],
            background_weights: vec![zero.clone(); marks],
            log_scaled_background_weights: vec![zero.clone(); marks],
            log_lambda_background_weights: vec![zero.clone(); marks],
            log_strengths: log_strengths.to_vec(),
            strength_gradient: vec![zero.clone(); log_strengths.len()],
            strength_second: vec![zero.clone(); log_strengths.len()],
            coefficient_strength_cross: vec![zero.clone(); marks * k],
            weighted_penalty: vec![zero.clone(); marks],
            mean_weights: vec![zero.clone(); marks * (k + 1)],
        };
        if k == 0 {
            for d in 0..marks {
                result.mean_weights[d] = zero.constant_like(1.0);
            }
            return Ok(result);
        }
        for (d, rho) in log_strengths.iter().enumerate() {
            let present = result.present(d);
            let categories = present.len();
            let w: Vec<S> = present
                .iter()
                .map(|&axis| theta[result.coordinate(d, axis)].clone())
                .collect();
            // Use shifted logits for individual simplex probabilities. Adding
            // log(C) to a very large common logit can lose that normalization.
            let shift = w.iter().fold(zero.clone(), |best, x| {
                if x.value() > best.value() {
                    x.clone()
                } else {
                    best
                }
            });
            let relative: Vec<S> = std::iter::once(shift.neg())
                .chain(w.iter().map(|x| x.sub(&shift)))
                .collect();
            let log_relative_sum = log_sum_exp(&relative);
            let log_pi_zero = shift.neg().sub(&log_relative_sum);
            result.background_weights[d] = exp(&log_pi_zero);
            let log_total = log_sum_exp(&[
                rho.clone(),
                ln(&zero.constant_like((categories + 1) as f64)),
            ]);
            result.log_scaled_background_weights[d] = log_total.sub(&shift).sub(&log_relative_sum);
            result.log_lambda_background_weights[d] = rho.add(&log_pi_zero);
            let log_pi: Vec<S> = relative[1..]
                .iter()
                .map(|r| r.sub(&log_relative_sum))
                .collect();
            let pi: Vec<S> = log_pi.iter().map(|v| exp(v)).collect();
            let log_activity_sum = log_sum_exp(&w);
            let weighted = if log_activity_sum.value() < 0.0 {
                // lambda*log1p(S) = sum exp(rho+w_k)*log1p(S)/S.
                // Shift rho into each term before summing: rho+log(S) would
                // lose log(C) when rho and the logits are large opposites.
                let ratio = emission::log_softplus(&log_activity_sum).sub(&log_activity_sum);
                let terms: Vec<S> = w.iter().map(|x| rho.add(x).add(&ratio)).collect();
                exp(&log_sum_exp(&terms))
            } else {
                exp(&rho.add(&emission::log_softplus(&log_activity_sum)))
            };
            let lambda_pi: Vec<S> = relative[1..]
                .iter()
                .map(|r| exp(&rho.add(r).sub(&log_relative_sum)))
                .collect();
            let scaled_pi: Vec<S> = lambda_pi
                .iter()
                .zip(&pi)
                .map(|(a, p)| a.add(&p.scale((categories + 1) as f64)))
                .collect();
            // C_C(lambda)=prod_{j=1}^C(lambda+j). Pair each normalizer term
            // with its log simplex weight before summing to avoid huge
            // cancelling totals in the no-latent-effect limit.
            let mut log_density = log_pi_zero.sub(&weighted);
            for (r, &axis) in present.iter().enumerate() {
                let log_j = ln(&zero.constant_like((r + 1) as f64));
                log_density = log_density
                    .add(&log_sum_exp(&[rho.clone(), log_j.clone()]).add(&log_pi[r]));
                let p = exp(&emission::softplus(&log_j.sub(rho)).neg());
                result.strength_gradient[d] = result.strength_gradient[d].add(&p);
                result.strength_second[d] =
                    result.strength_second[d].add(&p.mul(&add_real(&p.neg(), 1.0)));
                let index = d * k + axis;
                let coordinate = result.coordinate(d, axis);
                result.gradient[coordinate] = add_real(&scaled_pi[r].neg(), 1.0);
                result.coefficient_strength_cross[index] = lambda_pi[r].neg();
                result.weights[index] = pi[r].clone();
                result.scaled_weights[index] = scaled_pi[r].clone();
                result.lambda_weights[index] = lambda_pi[r].clone();
            }
            result.log_density = result.log_density.add(&log_density);
            result.strength_gradient[d] = result.strength_gradient[d].sub(&weighted);
            result.strength_second[d] = result.strength_second[d].sub(&weighted);
            result.weighted_penalty[d] = weighted;
            result.mean_weights[d * (k + 1)] =
                exp(&log_sum_exp(&[rho.clone(), zero.clone()]).sub(&log_total));
            for &axis in &present {
                result.mean_weights[d * (k + 1) + axis + 1] = exp(&log_total.neg());
            }
        }
        if !result.log_density.value().is_finite()
            || result
                .gradient
                .iter()
                .chain(&result.strength_gradient)
                .chain(&result.strength_second)
                .chain(&result.coefficient_strength_cross)
                .chain(&result.scaled_weights)
                .any(|v| !v.value().is_finite())
        {
            return Err(numerical(
                "decoder function prior or its derivatives are not representable",
            ));
        }
        Ok(result)
    }

    /// `d log Z / d tau` at `tau = 1/lambda_d = 0` for mark `d`'s decoder row, the Decoder{d}
    /// strength limit. In the chart `y_k = lambda pi_dk` the row's Dirichlet(1+lambda,1,...,1)
    /// density `C_C(lambda) lambda^-C (1 - sum y/lambda)^lambda` tends to i.i.d. Exp(1), with
    /// `E[pi_dk] = tau/(1 + (C+1) tau)` and every second moment O(tau^2). So the derivative is
    /// `sum_k g_k` over the row's present categories, where `g_k` is the fit's derivative in
    /// `pi_dk` of the reduced log posterior, including its log-det motion, at the empty row.
    /// The verdict is `ZeroEffectDecision::from_score` on this score evaluated at `Running`.
    pub fn decoder_boundary_score<S: JetField>(
        &self,
        support: &[bool],
        score: &[S],
    ) -> Result<S, EventHistoryError> {
        let k = self.spec.signatures;
        if support.len() != k
            || score.len() != k
            || !support.iter().any(|&present| present)
            || score.iter().any(|g| !g.value().is_finite())
        {
            return Err(invalid(
                "a decoder boundary score needs one finite derivative per signature and a present connection",
            ));
        }
        Ok(score
            .iter()
            .zip(support)
            .filter(|(_, present)| **present)
            .fold(score[0].constant_like(0.0), |sum, (g, _)| sum.add(g)))
    }
}

#[cfg(test)]
mod tests {
    use super::super::decoder::PreparedDecoder;
    use super::super::function_prior_tests::{compensated_sum, exact, rule};
    use super::super::law::{BasisPenalty, JointSpecification};
    use super::*;
    use crate::MarkKind;
    use crate::scalar::{Rows, div};
    use crate::test_support::{Bound, agrees};
    use ndarray::Array2;

    /// Two population columns: every logit function has a level and one
    /// variation coefficient, which this prior must never read.
    fn make_model(k: usize) -> JointLikelihood {
        JointLikelihood::new(JointSpecification {
            signatures: k,
            marks: vec![MarkKind::Recurrent],
            baseline_columns: 1,
            population_columns: 2,
            drive_columns: 1,
            entry_columns: 0,
            baseline_penalties: vec![],
            drive_penalties: vec![],
            population_penalties: vec![BasisPenalty {
                columns: 1..2,
                local: ndarray::array![[1.0]],
                rank: 1,
            }],
            measurements: vec![],
            genetic_mean: vec![],
            genetic_precision: Array2::zeros((0, 0)),
        })
        .unwrap()
    }

    /// The level coefficient of mark 0's logit function for `axis`.
    fn logit(model: &JointLikelihood, axis: usize) -> usize {
        model.layout.decoder.start + axis * model.spec.population_columns
    }

    /// Logit levels, with nonzero variation coefficients a prior reading them would move on.
    fn coefficients(model: &JointLikelihood, levels: &[f64]) -> Vec<f64> {
        let mut values = vec![0.0; model.layout.width];
        for (axis, &level) in levels.iter().enumerate() {
            values[logit(model, axis)] = level;
            values[logit(model, axis) + 1] = 5.0 - axis as f64;
        }
        values
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

    fn constant(value: f64) -> Bound {
        Bound::exact(0.0).constant_like(value)
    }

    fn unit(width: usize, index: usize) -> Vec<Bound> {
        (0..width).map(|q| Bound::exact(f64::from(q == index))).collect()
    }

    #[test]
    fn function_prior_includes_the_chart_jacobian_and_all_derivatives() {
        // Measured bar: the log density's weighted penalty passes log-softplus through
        // compose_unary, whose one-ulp charge has no cited accuracy.
        let model = make_model(3);
        let levels = [-2.0, 0.2, 1.3];
        let values = coefficients(&model, &levels);
        let theta = exact(&values);
        for rho in [-5.0, 0.4, 7.0] {
            let out = model.decoder_prior(&theta, &[Bound::exact(rho)]).unwrap();
            for i in 0..4 {
                let product = (i < 3).then(|| {
                    out.negative_hessian_product(&unit(values.len(), logit(&model, i)))
                        .unwrap()
                });
                for j in 0..4 {
                    let w: Vec<_> = (0..3)
                        .map(|q| {
                            Rows::seed(
                                Rows::seed(Bound::exact(levels[q]), [f64::from(q == i)]),
                                [f64::from(q == j)],
                            )
                        })
                        .collect();
                    let jet = oracle(
                        &w,
                        &Rows::seed(
                            Rows::seed(Bound::exact(rho), [f64::from(i == 3)]),
                            [f64::from(j == 3)],
                        ),
                    );
                    let g = if i == 3 {
                        out.log_strength_gradient()[0]
                    } else {
                        out.gradient()[logit(&model, i)]
                    };
                    let h = if i == 3 && j == 3 {
                        out.log_strength_second_derivative()[0]
                    } else if i == 3 {
                        out.coefficient_strength_cross()[j]
                    } else if j == 3 {
                        out.coefficient_strength_cross()[i]
                    } else {
                        product.as_ref().unwrap()[logit(&model, j)].neg()
                    };
                    let name = format!("rho={rho}, ({i},{j})");
                    agrees(out.log_density(), &jet.base.base, &name);
                    agrees(&g, &jet.base.rows[0], &name);
                    agrees(&h, &jet.rows[0].rows[0], &name);
                }
                if let Some(product) = &product {
                    for axis in 0..3 {
                        assert_eq!(product[logit(&model, axis) + 1].value, 0.0);
                    }
                }
            }
            for axis in 0..3 {
                assert_eq!(out.gradient()[logit(&model, axis) + 1].value, 0.0);
            }
        }
    }

    #[test]
    fn a_supported_decoder_prior_is_the_dirichlet_over_present_categories() {
        // Measured bar: the log density's weighted penalty passes log-softplus through
        // compose_unary, whose one-ulp charge has no cited accuracy.
        let model = make_model(3);
        let levels = [-0.7, 0.9, 0.4];
        let values = coefficients(&model, &levels);
        let theta = exact(&values);
        let support = vec![vec![true, false, true]];
        for rho in [-2.0, 0.6, 5.0] {
            let out = model
                .supported_decoder_prior(&theta, &[Bound::exact(rho)], &support)
                .unwrap();
            // The oracle reads only the present levels, axes 0 and 2.
            let present = [0_usize, 2];
            for (r, &axis) in present.iter().enumerate() {
                let product = out
                    .negative_hessian_product(&unit(values.len(), logit(&model, axis)))
                    .unwrap();
                for (s, &other) in present.iter().enumerate() {
                    let w: Vec<_> = present
                        .iter()
                        .enumerate()
                        .map(|(q, &a)| {
                            Rows::seed(
                                Rows::seed(Bound::exact(levels[a]), [f64::from(q == r)]),
                                [f64::from(q == s)],
                            )
                        })
                        .collect();
                    let jet = oracle(&w, &Rows::seed(Rows::seed(Bound::exact(rho), [0.0]), [0.0]));
                    let name = format!("rho={rho}, present ({axis},{other})");
                    agrees(out.log_density(), &jet.base.base, &name);
                    agrees(&out.gradient()[logit(&model, axis)], &jet.base.rows[0], &name);
                    agrees(&product[logit(&model, other)].neg(), &jet.rows[0].rows[0], &name);
                }
                // The absent category is not a coordinate of this prior.
                assert_eq!(product[logit(&model, 1)].value, 0.0);
            }
            assert_eq!(out.gradient()[logit(&model, 1)].value, 0.0);
            let mut direction = vec![0.0; values.len()];
            direction[logit(&model, 1)] = 1.0;
            assert!(
                out.negative_hessian_product(&exact(&direction))
                    .unwrap()
                    .iter()
                    .all(|v| v.value == 0.0)
            );
            let lambda = exp(&Bound::exact(rho));
            // Prior means over the background and two present categories: (1+lambda)/(3+lambda)
            // and 1/(3+lambda); the absent mean is zero.
            let total = add_real(&lambda, 3.0);
            agrees(
                &out.mean_weights()[0],
                &div(&add_real(&lambda, 1.0), &total),
                "background mean",
            );
            agrees(&out.mean_weights()[1], &div(&constant(1.0), &total), "present mean");
            assert_eq!(out.mean_weights()[2].value, 0.0);
        }
        // The face normalizer: removing a present category divides C_C(lambda) by C + lambda,
        // with C the larger count including it. Production's log density less the chart part
        // sum w - (lambda + C + 1) log(1 + sum exp w) is log C_C(lambda).
        let rho = Bound::exact(0.6);
        let lambda = exp(&rho);
        let normalizer = |support: &[Vec<bool>]| {
            let out = model
                .supported_decoder_prior(&theta, &[rho], support)
                .unwrap();
            let w: Vec<Bound> = (0..3)
                .filter(|&axis| support[0][axis])
                .map(|axis| Bound::exact(levels[axis]))
                .collect();
            let logits: Vec<Bound> = std::iter::once(Bound::exact(0.0)).chain(w.iter().copied()).collect();
            let chart = w
                .iter()
                .fold(Bound::exact(0.0), |sum, x| sum.add(x))
                .sub(&add_real(&lambda, (w.len() + 1) as f64).mul(&log_sum_exp(&logits)));
            out.log_density().sub(&chart)
        };
        let larger = normalizer(&[vec![true, true, true]]);
        let smaller = normalizer(&[vec![true, false, true]]);
        // The exposed normalizer is the one production's density carries. The density arms tie that
        // density to sum_j log(lambda + j), so the exposed face density
        // log C_3 - log C_2 = log(3 + lambda) follows.
        agrees(&log_decoder_normalizer(3, &rho), &larger, "log C_3(lambda)");
        agrees(&log_decoder_normalizer(2, &rho), &smaller, "log C_2(lambda)");
        assert!(model.supported_decoder_prior(&theta, &[rho], &[vec![false, false, false]]).is_err());
    }

    #[test]
    fn decoder_boundary_score_is_the_first_moment_derivative_of_the_empty_row() {
        // For a log likelihood linear in the row, g' pi, log Z(tau) = log E exp(g' pi) under
        // Dirichlet(1 + 1/tau, 1, ..., 1) over C present categories. Its moments are
        // E pi_k = tau/(1 + (C+1) tau) and E pi_k pi_j = (1 + [k=j]) tau^2/((1 + (C+1) tau)(1 + (C+2) tau)),
        // and every higher moment is O(tau^3), so log(1 + g'E pi + ½ E (g' pi)^2) has the exact
        // log Z's derivative at tau = 0, taken here by a jet.
        let model = make_model(3);
        let support = [true, false, true];
        let score = [1.25, -7.0, -0.5];
        let production = model
            .decoder_boundary_score(&support, &exact(&score))
            .unwrap();
        let tau = Rows::seed(Bound::exact(0.0), [1.0]);
        let categories = 2.0;
        let one = |v: f64| tau.constant_like(v);
        let first = div(&tau, &add_real(&tau.scale(categories + 1.0), 1.0));
        let second = div(
            &tau.mul(&tau),
            &add_real(&tau.scale(categories + 1.0), 1.0).mul(&add_real(&tau.scale(categories + 2.0), 1.0)),
        );
        let present: Vec<f64> = score
            .iter()
            .zip(&support)
            .filter(|(_, present)| **present)
            .map(|(g, _)| *g)
            .collect();
        let linear = present.iter().fold(one(0.0), |sum, &g| sum.add(&first.mul(&one(g))));
        let quadratic = present.iter().enumerate().fold(one(0.0), |sum, (a, &ga)| {
            present.iter().enumerate().fold(sum, |sum, (b, &gb)| {
                let multiplicity = if a == b { 2.0 } else { 1.0 };
                sum.add(&second.mul(&one(ga)).mul(&one(gb)).scale(multiplicity))
            })
        });
        let jet = ln(&add_real(&linear.add(&quadratic.scale(0.5)), 1.0));
        agrees(&production, &jet.rows[0], "empty-row boundary derivative");
        assert!(model.decoder_boundary_score(&[false, false, false], &exact(&score)).is_err());
        assert!(model.decoder_boundary_score(&support, &exact(&score[1..])).is_err());
    }

    #[test]
    fn third_derivative_and_strength_curvature_products_match_nested_jets() {
        let model = make_model(3);
        let levels = [-2.0, 0.2, 1.3];
        let values = coefficients(&model, &levels);
        let mut direction = vec![0.0; values.len()];
        let mut delta = vec![0.0; values.len()];
        for (axis, (&v, &e)) in [0.3, -0.7, 1.1].iter().zip(&[-0.4, 0.9, 0.2]).enumerate() {
            direction[logit(&model, axis)] = v;
            delta[logit(&model, axis)] = e;
            // Variation directions the prior must ignore.
            direction[logit(&model, axis) + 1] = 0.6;
            delta[logit(&model, axis) + 1] = -0.8;
        }
        let theta = exact(&values);
        for rho in [-5.0, 0.4, 7.0] {
            let out = model.decoder_prior(&theta, &[Bound::exact(rho)]).unwrap();
            let third = out
                .third_derivative_product(&exact(&delta), &exact(&direction))
                .unwrap();
            let strength = out.strength_hessian_product(0, &exact(&direction)).unwrap();
            for i in 0..3 {
                // Outer directions (v, delta), inner coordinate i: d^3 log p.
                let w: Vec<_> = (0..3)
                    .map(|q| {
                        Rows::seed(
                            Rows::seed(
                                Rows::seed(Bound::exact(levels[q]), [f64::from(q == i)]),
                                [direction[logit(&model, q)]],
                            ),
                            [delta[logit(&model, q)]],
                        )
                    })
                    .collect();
                let jet = oracle(
                    &w,
                    &Rows::seed(Rows::seed(Rows::seed(Bound::exact(rho), [0.0]), [0.0]), [0.0]),
                );
                agrees(
                    &third[logit(&model, i)].neg(),
                    &jet.rows[0].rows[0].rows[0],
                    &format!("third, rho={rho}, i={i}"),
                );
                // Outer directions (v, e_i), inner strength: d rho d v d e_i log p.
                let w: Vec<_> = (0..3)
                    .map(|q| {
                        Rows::seed(
                            Rows::seed(Rows::seed(Bound::exact(levels[q]), [0.0]), [direction[logit(&model, q)]]),
                            [f64::from(q == i)],
                        )
                    })
                    .collect();
                let jet = oracle(
                    &w,
                    &Rows::seed(Rows::seed(Rows::seed(Bound::exact(rho), [1.0]), [0.0]), [0.0]),
                );
                agrees(
                    &strength[logit(&model, i)].neg(),
                    &jet.rows[0].rows[0].rows[0],
                    &format!("strength, rho={rho}, i={i}"),
                );
                assert_eq!(third[logit(&model, i) + 1].value, 0.0);
                assert_eq!(strength[logit(&model, i) + 1].value, 0.0);
            }
        }
        let out = model.decoder_prior(&values, &[0.4]).unwrap();
        assert!(out.third_derivative_product(&delta[1..], &direction).is_err());
    }

    #[test]
    fn function_prior_normalization_and_means_match_simplex_integration() {
        // Measured bar: the log density's weighted penalty passes log-softplus through
        // compose_unary, whose one-ulp charge has no cited accuracy.
        let model = make_model(2);
        let lambda = 3.0_f64;
        let rho = Bound::exact(lambda.ln());
        // Dirichlet(4,1,1) times the triangle map's Jacobian (1-p1) is 20(1-p1)^4 (1-q)^3 over
        // (p1, q) in the unit square, a polynomial of degree 4 in each variable, so the
        // 64-point product rule is exact and only the rule's certified errors and rounding
        // remain.
        let r = rule(64, 0.0, 1.0);
        let mut mass = Vec::new();
        let mut means = [Vec::new(), Vec::new(), Vec::new()];
        let mut score = Vec::new();
        let mut errors = [0.0; 5];
        for (p1, wx) in r.points.iter().zip(&r.weights) {
            let complement = add_real(&p1.neg(), 1.0);
            for (q, wy) in r.points.iter().zip(&r.weights) {
                let p2 = q.mul(&complement);
                let p0 = complement.mul(&add_real(&q.neg(), 1.0));
                let logs = [ln(&p0), ln(p1), ln(&p2)];
                let logits = [logs[1].sub(&logs[0]), logs[2].sub(&logs[0])];
                let values = coefficients(&model, &[logits[0].value, logits[1].value]);
                let out = model.decoder_prior(&exact(&values), &[rho]).unwrap();
                let weight = exp(&out.log_density().sub(&logs[0]).sub(&logs[1]).sub(&logs[2]))
                    .mul(&complement)
                    .mul(wx)
                    .mul(wy);
                // A node displaced by the rule's point error moves every row's log by at most
                // (5/(1-p1) + 1/p1) in p1 and (4/(1-q) + 1/q) in q: the integrand's exponents
                // plus the largest mean factor's.
                let displacement = r.point_error
                    * (5.0 / (1.0 - p1.value) + 1.0 / p1.value + 4.0 / (1.0 - q.value) + 1.0 / q.value);
                // Relative error of this node: its own account, both weights' certified errors,
                // the displacement, and the logits' rounding through the density gradient.
                let relative = weight.rounding() / weight.value
                    + 2.0 * r.weight_relative_error
                    + displacement
                    + (0..2)
                        .map(|k| out.gradient()[logit(&model, k)].value.abs() * logits[k].rounding())
                        .sum::<f64>();
                mass.push(weight.value);
                errors[0] += weight.value * relative;
                for (j, p) in [p0, *p1, p2].iter().enumerate() {
                    let moment = weight.mul(p);
                    means[j].push(moment.value);
                    errors[1 + j] += moment.value * relative + moment.rounding();
                }
                let g = out.log_strength_gradient()[0];
                let product = weight.mul(&g);
                score.push(product.value);
                // The score c + lambda log p0 moves by lambda (1/(1-p1) + 1/(1-q)) per unit
                // displacement.
                errors[4] += product.value.abs() * relative
                    + product.rounding()
                    + weight.value
                        * lambda
                        * r.point_error
                        * (1.0 / (1.0 - p1.value) + 1.0 / (1.0 - q.value));
            }
        }
        let (total, summation) = compensated_sum(&mass);
        assert!(
            (total - 1.0).abs() <= errors[0] + summation,
            "mass {total} within {}",
            errors[0] + summation
        );
        // Exact means (1+lambda)/(K+1+lambda) = 4/6 and 1/(K+1+lambda) = 1/6.
        let prior = model
            .decoder_prior(&exact(&vec![0.0; model.layout.width]), &[rho])
            .unwrap();
        let exact_means = [
            div(&constant(4.0), &constant(6.0)),
            div(&constant(1.0), &constant(6.0)),
            div(&constant(1.0), &constant(6.0)),
        ];
        for j in 0..3 {
            let (mean, summation) = compensated_sum(&means[j]);
            assert!(
                (mean - exact_means[j].value).abs()
                    <= errors[1 + j] + summation + exact_means[j].rounding(),
                "mean {j}: {mean}"
            );
            agrees(&prior.mean_weights()[j], &exact_means[j], &format!("prior mean {j}"));
        }
        let (strength_score, summation) = compensated_sum(&score);
        assert!(strength_score.abs() <= errors[4] + summation);
    }

    #[test]
    fn function_penalty_is_invariant_to_intensity_scale_and_leaves_baselines_free() {
        // Measured bar: the activity and the weighted penalty pass softplus and log-softplus
        // through compose_unary, whose one-ulp charge has no cited accuracy.
        let model = make_model(2);
        let mut values = coefficients(&model, &[-0.3, 0.5]);
        let theta = exact(&values);
        let rho = Bound::exact(0.7);
        let prior = model.decoder_prior(&theta, &[rho]).unwrap();
        let pi_zero = prior.background_weights[0].value;
        let lambda = rho.value.exp();
        // Remainder: softplus(-40) activity on the low asymptote and log1p(exp(-40)) slope
        // corrections move log(low/total) by at most exp(-40) (1/pi_0 + 2).
        let remainder = lambda * (-40.0_f64).exp() * (1.0 / pi_zero + 2.0);
        // A population row whose variation feature vanishes, so the logits are their levels.
        let row = ndarray::array![[1.0, 0.0]];
        let decoder = PreparedDecoder::new(&model, &theta, row.view()).unwrap();
        let activity = |x: [f64; 2]| exp(&decoder.activity(0, 0, &exact(&x)));
        // The asymptote ratio from the production activity. A common intensity scale c cancels
        // from it by construction; the scale enters production only through the baseline, and
        // the shift arm below checks that it leaves this prior unchanged.
        let low = activity([-40.0, -40.0]);
        let mut total = low;
        for axis in 0..2 {
            let mut x = [-40.0, -40.0];
            x[axis] = 40.0;
            let left = activity(x);
            x[axis] = 50.0;
            let right = activity(x);
            total = total.add(&right.sub(&left).scale(0.1));
        }
        let penalty = ln(&div(&low, &total)).mul(&exp(&rho)).neg();
        let bar = remainder + penalty.rounding() + prior.weighted_function_penalty()[0].rounding();
        assert!(
            (penalty.value - prior.weighted_function_penalty()[0].value).abs() <= bar,
            "{} vs {} within {bar}",
            penalty.value,
            prior.weighted_function_penalty()[0].value
        );
        values[model.layout.baseline.start] = 14.0;
        let shifted = model.decoder_prior(&exact(&values), &[rho]).unwrap();
        assert_eq!(prior.log_density().value, shifted.log_density().value);
        assert_eq!(prior.gradient()[model.layout.baseline.start].value, 0.0);
    }

    #[test]
    fn function_prior_hessian_retains_saturated_simplex_curvature() {
        let model = make_model(1);
        let coordinate = logit(&model, 0);
        for (rho, level) in [(0.0_f64, 40.0_f64), (700.0, 800.0)] {
            let values = coefficients(&model, &[level]);
            let prior = model
                .decoder_prior(&exact(&values), &[Bound::exact(rho)])
                .unwrap();
            let product = &prior
                .negative_hessian_product(&unit(values.len(), coordinate))
                .unwrap()[coordinate];
            // Closed form (lambda+2) exp(-w)/(1+exp(-w))^2 with representable log
            // log(lambda+2) - w: exactly log 3 - w at rho = 0, and rho - w otherwise. Dropping
            // (1+exp(-w))^-2 and 2/lambda leaves a relative remainder below
            // 2 exp(-w) + 2 exp(-rho) (the second term only when rho > 0).
            let (log_expected, remainder) = if rho == 0.0 {
                (ln(&constant(3.0)).sub(&Bound::exact(level)), 2.0 * (-level).exp())
            } else {
                (
                    Bound::exact(rho).sub(&Bound::exact(level)),
                    2.0 * (-level).exp() + 2.0 * (-rho).exp(),
                )
            };
            let expected = exp(&log_expected);
            let bar = expected.value * remainder + product.rounding() + expected.rounding();
            assert!(
                (product.value - expected.value).abs() <= bar,
                "rho={rho}, w={level}: {} vs {} within {bar}",
                product.value,
                expected.value
            );
        }
    }

    #[test]
    fn function_prior_preserves_strong_shrinkage_and_weak_strength_extremes() {
        // Measured bar: the weighted penalty passes log-softplus through compose_unary, whose
        // one-ulp charge has no cited accuracy.
        let model = make_model(3);
        // At w_k = -rho with rho >= 800, every exp(-rho) underflows. The logits' normalizer is
        // exactly 0, log pi_k = -rho, log(lambda+j) = rho, and each pair cancels exactly;
        // sum_j lambda/(lambda+j) = 3 exactly. So the log density is exactly -W, the strength
        // score exactly 3 - W and its curvature exactly -W, with W = lambda log(1 + sum exp w)
        // = exp(log 3). The dropped remainder, below 23 exp(-rho), underflows. Every larger rho
        // repeats the same exact cancellations bitwise.
        let at = |rho: f64| {
            model
                .decoder_prior(&coefficients(&model, &[-rho; 3]), &[rho])
                .unwrap()
        };
        let moderate = model
            .decoder_prior(&exact(&coefficients(&model, &[-800.0; 3])), &[Bound::exact(800.0)])
            .unwrap();
        let weighted = moderate.weighted_function_penalty()[0];
        agrees(&weighted, &exp(&ln(&constant(3.0))), "shrinkage penalty");
        assert_eq!(moderate.log_density().value, -weighted.value);
        assert_eq!(moderate.log_strength_gradient()[0].value, 3.0 - weighted.value);
        assert_eq!(moderate.log_strength_second_derivative()[0].value, -weighted.value);
        let reference = at(800.0);
        for rho in [800.0, 1e200] {
            let out = at(rho);
            assert_eq!(out.log_density(), reference.log_density());
            assert_eq!(out.log_strength_gradient(), reference.log_strength_gradient());
            assert_eq!(
                out.log_strength_second_derivative(),
                reference.log_strength_second_derivative()
            );
            assert_eq!(
                out.weighted_function_penalty(),
                reference.weighted_function_penalty()
            );
            // lambda pi_k = exp(0) and (K+1) pi_k underflows, so the product is an exact 1 and
            // every gradient 1 - (lambda+K+1) pi_k is an exact 0.
            let mut direction = vec![0.0; model.layout.width];
            direction[logit(&model, 0)] = 1.0;
            let product = out.negative_hessian_product(&direction).unwrap();
            assert_eq!(product[logit(&model, 0)], 1.0);
            assert!(out.gradient().iter().all(|&v| v == 0.0));
            assert_eq!(out.mean_weights()[0], 1.0);
        }
        let values = coefficients(&model, &[1e200; 3]);
        let rho = Bound::exact(-800.0);
        let out = model.decoder_prior(&exact(&values), &[rho]).unwrap();
        // log(1 + 3 exp(1e200)) = 1e200 exactly, so the strength score is
        // -exp(rho + log 1e200). Its positive part sum_j lambda/(lambda+j), below 3 exp(rho),
        // underflows.
        let expected = exp(&rho.add(&ln(&Bound::exact(1e200)))).neg();
        agrees(&out.log_strength_gradient()[0], &expected, "weak strength score");
        // pi_k = exp(-log 3) and lambda pi_k underflows: the gradient 1 - 4 pi_k.
        let expected = add_real(&exp(&ln(&constant(3.0)).neg()).scale(-4.0), 1.0);
        for j in 0..3 {
            agrees(&out.gradient()[logit(&model, j)], &expected, &format!("weak gradient {j}"));
        }
        assert!(model.decoder_prior(&values, &[]).is_err());
        let null = make_model(0);
        let coefficients = vec![0.0; null.layout.width];
        let prior = null.decoder_prior(&coefficients, &[]).unwrap();
        assert_eq!(*prior.log_density(), 0.0);
        assert_eq!(prior.mean_weights()[0], 1.0);
        assert!(prior.log_strengths().is_empty());
        assert!(null.decoder_prior(&coefficients, &[0.0]).is_err());
    }
}
