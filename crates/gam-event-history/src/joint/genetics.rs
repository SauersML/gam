//! Genetic scores in the joint latent-signature law.
//!
//! Scores shape the shared trajectory through the state drive,
//!
//! ```text
//!     dx = Kappa [m(t, g, c) - x] dt + L dW + ...,     m(t, g, c) = u(t, c) + B(t, c) g,
//! ```
//!
//! and a person's scores can be missing. A missing score is an integration
//! coordinate of the joint law, never a filled value, so the law needs an explicit
//! JOINT distribution of the score vector conditional on the reference context
//! (for example genetic principal components):
//!
//! ```text
//!     g | c ~ N(A c, Sigma).
//! ```
//!
//! Separate per-score transforms give each score a marginal law and say nothing
//! about their dependence; the law carries the full covariance. It is declared by
//! the model: fitted once from the training score sample by its own observed-data
//! likelihood under missingness at random given `c`, persisted with the fit, and
//! replayed by state inference, reference evolution and serving.
//!
//! With precision `Lambda = Sigma^-1`, the missing block `m` given the observed
//! block `o` is Gaussian with precision `Lambda_mm` and mean
//! `mu_m - Lambda_mm^-1 Lambda_mo (g_o - mu_o)`, and the joint density factors
//! exactly as `p(g_o | c) p(g_m | g_o, c)`. `Lambda_mm` is the corner of the state
//! posterior's dense genetic border; the reference bank transports standard
//! normal draws through the same conditional law.
//!
//! The survival marginal-slope declared law (gam#2929) is a finite node set
//! transported per context. Conditioning a finite law on partially observed
//! coordinates conditions on an event of probability zero, so that law can serve
//! only an all-missing score vector. For either law, adequacy is judged on the
//! functional that consumes it (reference moments, forecasts), not on moment
//! agreement alone.
//!
//! `(A, Sigma)` enter at their mode as a declared input law. The fit returns only
//! a certified stationary point: EM from any start, then Newton on the
//! observed-data likelihood until every score entry in `(A, vech Sigma)` lies
//! within its own first-order roundoff bound. Averaging over their posterior
//! instead would change a consumed expectation by O(d/n) for `d` law coefficients
//! fitted on `n` score vectors: each coefficient's posterior spread is O(n^-1/2),
//! the consumed expectation is smooth in them, and the delta method's leading
//! term is the trace of its curvature against that O(1/n) covariance. The latent
//! spread a masked subject carries, `Lambda_mm^-1`, is O(1) and does not shrink
//! with `n`. The measured size of the difference at small `n` is pending.
//!
//! Context columns beyond the intercept carry a normalized Gaussian prior on the
//! function `c -> A c` (SPEC 12/14): `vec(A_ctx) ~ N(0, G^-1 ⊗ Sigma / lambda)`, with
//! `G` the centered context Gram under the equal-subject measure. Its strength is
//! learned by Laplace evidence through `opt`, and the no-context model, this evidence
//! family's `lambda -> infinity` limit, is compared as its own structure. The family's exact
//! boundary score at that limit decides whether the evidence rises from it; when it does
//! not, the fit returns the no-context law, so the default recovers no context dependence.

use crate::EventHistoryError;
use crate::scalar::{div, ln, sqrt};
use gam_linalg::triangular::{
    CholeskyGuard, cholesky_factor_in_place, cholesky_solve_matrix, cholesky_solve_vector,
};
use gam_math::nested_dual::JetField;
use ndarray::{Array1, Array2, ArrayView2, Axis};
use serde::{Deserialize, Serialize};
use std::collections::BTreeMap;
use std::f64::consts::TAU;
use std::ops::Range;

fn invalid(reason: impl Into<String>) -> EventHistoryError {
    EventHistoryError::InvalidInput {
        reason: reason.into(),
    }
}

fn numerical(reason: impl Into<String>) -> EventHistoryError {
    EventHistoryError::NumericalFailure {
        reason: reason.into(),
    }
}

fn lower_factor(matrix: &Array2<f64>, what: &str) -> Result<Array2<f64>, EventHistoryError> {
    cholesky_factor_in_place(matrix.view(), CholeskyGuard::FiniteStrict)
        .ok_or_else(|| numerical(format!("{what} is not positive definite")))
}

fn factor_log_determinant(factor: &Array2<f64>) -> f64 {
    2.0 * factor.diag().iter().map(|v| v.ln()).sum::<f64>()
}

fn submatrix<T: Clone>(matrix: &Array2<T>, rows: &[usize], columns: &[usize]) -> Array2<T> {
    Array2::from_shape_fn((rows.len(), columns.len()), |(a, b)| {
        matrix[[rows[a], columns[b]]].clone()
    })
}

/// The lower Cholesky factor of a symmetric positive-definite matrix over `S`, reading its lower
/// triangle (Cholesky–Banachiewicz), refusing a non-positive pivot.
fn cholesky_over<S: JetField>(
    zero: &S,
    matrix: &Array2<S>,
    what: &str,
) -> Result<Array2<S>, EventHistoryError> {
    cholesky_resolved(|_: &S| 0.0, zero, matrix, what)
}

/// `cholesky_over`, refusing any pivot that is not positive beyond `pivot_rounding` of it. At
/// `Running` that is the pivot's own first-order rounding, below which the square root's and
/// logarithm's first-order charges no longer bound it.
fn cholesky_resolved<S: JetField>(
    pivot_rounding: impl Fn(&S) -> f64,
    zero: &S,
    matrix: &Array2<S>,
    what: &str,
) -> Result<Array2<S>, EventHistoryError> {
    let n = matrix.nrows();
    let mut factor = Array2::from_shape_fn((n, n), |_| zero.constant_like(0.0));
    for i in 0..n {
        for j in 0..=i {
            let mut sum = matrix[[i, j]].clone();
            for k in 0..j {
                sum = sum.sub(&factor[[i, k]].mul(&factor[[j, k]]));
            }
            if i == j {
                let rounding = pivot_rounding(&sum);
                if !(sum.value().is_finite() && sum.value() > rounding) {
                    return Err(numerical(format!(
                        "{what} has pivot {i} = {:.3e}, not positive beyond its rounding {rounding:.3e}",
                        sum.value()
                    )));
                }
                factor[[i, i]] = sqrt(&sum);
            } else {
                factor[[i, j]] = div(&sum, &factor[[j, j]]);
            }
        }
    }
    Ok(factor)
}

/// `A^-1 b` over `S` through the lower Cholesky factor of `A`: forward, then back substitution.
fn cholesky_solve_over<S: JetField>(zero: &S, factor: &Array2<S>, rhs: &[S]) -> Vec<S> {
    let n = rhs.len();
    let mut forward: Vec<S> = Vec::with_capacity(n);
    for i in 0..n {
        let mut sum = rhs[i].clone();
        for k in 0..i {
            sum = sum.sub(&factor[[i, k]].mul(&forward[k]));
        }
        forward.push(div(&sum, &factor[[i, i]]));
    }
    let mut solution = vec![zero.constant_like(0.0); n];
    for i in (0..n).rev() {
        let mut sum = forward[i].clone();
        for k in i + 1..n {
            sum = sum.sub(&factor[[k, i]].mul(&solution[k]));
        }
        solution[i] = div(&sum, &factor[[i, i]]);
    }
    solution
}

/// `log det A = 2 sum_i ln L_ii` over `S` from the lower Cholesky factor of `A`.
fn log_determinant_over<S: JetField>(zero: &S, factor: &Array2<S>) -> S {
    (0..factor.nrows())
        .fold(zero.constant_like(0.0), |sum, i| sum.add(&ln(&factor[[i, i]])))
        .scale(2.0)
}

// This file's matrix-level routes carry inline f64 error matrices by `numerical::Running`'s rules
// (brief 19:15, rules doc in law.rs's `numerical`), and solves inject errors only through its
// `solve_forward_error` certificate.
use super::law::numerical::{
    Running, RunningSum, contraction_with_error, product_error, product_with_error,
    solve_forward_error,
};

/// A symmetric positive-definite matrix's inverse through its Cholesky factor, with the entrywise
/// first-order errors of `solve_forward_error`'s certificate.
fn inverse_with_error(
    matrix: &Array2<f64>,
    assembly: &Array2<f64>,
    what: &str,
) -> Result<(Array2<f64>, Array2<f64>), EventHistoryError> {
    let n = matrix.nrows();
    let factor = lower_factor(matrix, what)?;
    let inverse = cholesky_solve_matrix(&factor, &Array2::<f64>::eye(n));
    let certificate = solve_forward_error(matrix, &inverse, assembly, what)?;
    Ok((inverse, certificate.inverse_error()))
}

/// A computed value whose inline first-order error is `error` (an absolute error, `eps mu`), as a
/// `Running` scalar: `mu = error / eps`, so that `rounding()` returns `error`.
fn running_with_error(value: f64, error: f64) -> Running {
    Running {
        value,
        mu: error / f64::EPSILON,
    }
}

/// The declared joint law of the score vector conditional on reference context.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct GeneticScoreLaw {
    /// Score coordinates `G`.
    pub scores: usize,
    /// Columns of the reference-context row `c`; the caller supplies the intercept.
    pub context_columns: usize,
    /// `A`, `G x context_columns`, row-major.
    pub mean_coefficients: Vec<f64>,
    /// Lower Cholesky factor of `Sigma`, `G x G`, row-major.
    pub covariance_factor: Vec<f64>,
}

impl GeneticScoreLaw {
    pub fn validate(&self) -> Result<(), EventHistoryError> {
        let g = self.scores;
        if g > 0 && self.context_columns == 0 {
            return Err(invalid(
                "a genetic score law needs at least one reference-context column",
            ));
        }
        let means = g
            .checked_mul(self.context_columns)
            .ok_or_else(|| invalid("genetic score law dimension overflow"))?;
        let factor = g
            .checked_mul(g)
            .ok_or_else(|| invalid("genetic score law dimension overflow"))?;
        if self.mean_coefficients.len() != means
            || self.covariance_factor.len() != factor
            || self
                .mean_coefficients
                .iter()
                .chain(&self.covariance_factor)
                .any(|v| !v.is_finite())
        {
            return Err(invalid(
                "genetic score law coefficients have invalid dimensions or non-finite values",
            ));
        }
        for i in 0..g {
            if !(self.covariance_factor[i * g + i] > 0.0)
                || self.covariance_factor[i * g + i + 1..(i + 1) * g]
                    .iter()
                    .any(|v| *v != 0.0)
            {
                return Err(invalid(
                    "genetic score covariance factor must be lower triangular with a positive diagonal",
                ));
            }
        }
        Ok(())
    }

    fn mean_at(&self, context: &[f64]) -> Vec<f64> {
        let p = self.context_columns;
        (0..self.scores)
            .map(|j| {
                self.mean_coefficients[j * p..(j + 1) * p]
                    .iter()
                    .zip(context)
                    .map(|(a, c)| a * c)
                    .sum()
            })
            .collect()
    }

    fn factor(&self) -> Array2<f64> {
        let g = self.scores;
        Array2::from_shape_fn((g, g), |(i, j)| self.covariance_factor[i * g + j])
    }

    /// The law of the missing scores given the observed ones and the context, in f64.
    pub fn conditional(
        &self,
        context: &[f64],
        observed: &[Option<f64>],
    ) -> Result<ConditionalScores, EventHistoryError> {
        self.conditional_over(&0.0, context, observed)
    }

    /// The law of the missing scores given the observed ones and the context over `S`, computed
    /// once per subject with the persisted law, the context and the observed scores as exact data
    /// and every derived quantity formed over `S`: `mu = A c` and `Sigma = L L'`; the observed
    /// block's Cholesky factor and `log p(g_o | c)`; the conditional mean
    /// `mu_m + Sigma_mo Sigma_oo^-1 (g_o - mu_o)` and covariance
    /// `C = Sigma_mm - Sigma_mo Sigma_oo^-1 Sigma_om`; `Lambda_mm = C^-1` with its Cholesky factor;
    /// and `log det Lambda_mm = -log det C`. Its Cholesky factors refuse only non-positive pivots,
    /// not pivots inside their own rounding: no production decision evaluates it at `Running`, and
    /// its `Running` values are certified by the Bound-oracle test.
    pub fn conditional_over<S: JetField>(
        &self,
        zero: &S,
        context: &[f64],
        observed: &[Option<f64>],
    ) -> Result<ConditionalScores<S>, EventHistoryError> {
        self.validate()?;
        let g = self.scores;
        let p = self.context_columns;
        if context.len() != p || observed.len() != g {
            return Err(invalid(
                "score context or observed scores do not match the genetic score law",
            ));
        }
        if context
            .iter()
            .chain(observed.iter().flatten())
            .any(|v| !v.is_finite())
        {
            return Err(invalid("score context and observed scores must be finite"));
        }
        if g > 0 && context.first() != Some(&1.0) {
            return Err(invalid(
                "score context column 0 must be the intercept, exactly one",
            ));
        }
        let exact = |value: f64| zero.constant_like(value);
        let mean: Vec<S> = (0..g)
            .map(|j| {
                (0..p).fold(exact(0.0), |sum, b| {
                    sum.add(&exact(self.mean_coefficients[j * p + b]).mul(&exact(context[b])))
                })
            })
            .collect();
        let covariance = Array2::from_shape_fn((g, g), |(i, j)| {
            (0..g).fold(exact(0.0), |sum, k| {
                sum.add(
                    &exact(self.covariance_factor[i * g + k])
                        .mul(&exact(self.covariance_factor[j * g + k])),
                )
            })
        });
        let (missing, present): (Vec<usize>, Vec<usize>) =
            (0..g).partition(|&j| observed[j].is_none());
        let residual: Vec<S> = observed
            .iter()
            .zip(&mean)
            .filter_map(|(value, mu)| value.map(|v| exact(v).sub(mu)))
            .collect();
        let observed_factor = cholesky_over(
            zero,
            &submatrix(&covariance, &present, &present),
            "the observed score covariance block",
        )?;
        let whitened = cholesky_solve_over(zero, &observed_factor, &residual);
        let observed_log_density = if present.is_empty() {
            exact(0.0)
        } else {
            let quadratic = whitened
                .iter()
                .zip(&residual)
                .fold(exact(0.0), |sum, (a, b)| sum.add(&a.mul(b)));
            ln(&exact(TAU))
                .scale(present.len() as f64)
                .add(&log_determinant_over(zero, &observed_factor))
                .add(&quadratic)
                .scale(-0.5)
        };
        let m = missing.len();
        // Sigma_oo^-1 Sigma_om, one missing score per column.
        let coupling: Vec<Vec<S>> = missing
            .iter()
            .map(|&k| {
                let column: Vec<S> = present.iter().map(|&j| covariance[[j, k]].clone()).collect();
                cholesky_solve_over(zero, &observed_factor, &column)
            })
            .collect();
        let conditional_covariance = Array2::from_shape_fn((m, m), |(a, b)| {
            let explained = present
                .iter()
                .zip(&coupling[b])
                .fold(exact(0.0), |sum, (&j, w)| sum.add(&covariance[[missing[a], j]].mul(w)));
            covariance[[missing[a], missing[b]]].sub(&explained)
        });
        let covariance_factor = cholesky_over(
            zero,
            &conditional_covariance,
            "the conditional covariance of missing scores",
        )?;
        let precision_columns: Vec<Vec<S>> = (0..m)
            .map(|b| {
                let unit: Vec<S> = (0..m).map(|a| exact(f64::from(u8::from(a == b)))).collect();
                cholesky_solve_over(zero, &covariance_factor, &unit)
            })
            .collect();
        let precision = Array2::from_shape_fn((m, m), |(a, b)| precision_columns[b][a].clone());
        let precision_factor =
            cholesky_over(zero, &precision, "the conditional precision of missing scores")?;
        let log_determinant = log_determinant_over(zero, &covariance_factor).neg();
        let conditional_mean: Vec<S> = missing
            .iter()
            .map(|&k| {
                present
                    .iter()
                    .zip(&whitened)
                    .fold(mean[k].clone(), |sum, (&j, w)| sum.add(&covariance[[k, j]].mul(w)))
            })
            .collect();
        if conditional_mean.iter().any(|v| !v.value().is_finite())
            || !observed_log_density.value().is_finite()
        {
            return Err(numerical("conditional genetic score law is not finite"));
        }
        Ok(ConditionalScores {
            missing,
            mean: conditional_mean,
            precision,
            precision_factor,
            log_determinant,
            observed_log_density,
            observed: observed.to_vec(),
        })
    }

    /// `E_c[(1, g)(1, g)']` under this law over a weighted context sample: the
    /// weighted mean of `(1, A c)(1, A c)'` plus `diag(0, Sigma)`. The final-function
    /// roughness of the drive averages its `(1 + G)` interactions against it.
    pub fn interaction_second_moment(
        &self,
        context: ArrayView2<'_, f64>,
        weights: &[f64],
    ) -> Result<Array2<f64>, EventHistoryError> {
        self.validate()?;
        let g = self.scores;
        if context.ncols() != self.context_columns
            || weights.len() != context.nrows()
            || context.iter().chain(weights).any(|v| !v.is_finite())
            || weights.iter().any(|w| *w < 0.0)
        {
            return Err(invalid(
                "context sample or weights do not match the genetic score law",
            ));
        }
        if g > 0 && context.column(0).iter().any(|v| *v != 1.0) {
            return Err(invalid(
                "score context column 0 must be the intercept, exactly one",
            ));
        }
        let total: f64 = weights.iter().sum();
        if !(total > 0.0) {
            return Err(invalid(
                "an interaction second moment needs a positive total weight",
            ));
        }
        let mut moment = Array2::<f64>::zeros((g + 1, g + 1));
        for (row, &w) in context.rows().into_iter().zip(weights) {
            let augmented: Vec<f64> = std::iter::once(1.0)
                .chain(self.mean_at(&row.to_vec()))
                .collect();
            for i in 0..=g {
                for j in 0..=g {
                    moment[[i, j]] += w * augmented[i] * augmented[j] / total;
                }
            }
        }
        let factor = self.factor();
        let covariance = factor.dot(&factor.t());
        for i in 0..g {
            for j in 0..g {
                moment[[i + 1, j + 1]] += covariance[[i, j]];
            }
        }
        Ok(moment)
    }

    /// `g = A c + L eps`: the declared law's transport of a whitened residual to a
    /// context.
    pub fn transport(
        &self,
        context: &[f64],
        residual: &[f64],
        genome: &mut [f64],
    ) -> Result<(), EventHistoryError> {
        self.validate()?;
        let g = self.scores;
        if context.len() != self.context_columns
            || residual.len() != g
            || genome.len() != g
            || context.iter().chain(residual).any(|v| !v.is_finite())
        {
            return Err(invalid(
                "context, residual or score buffer do not match the genetic score law",
            ));
        }
        if g > 0 && context.first() != Some(&1.0) {
            return Err(invalid(
                "score context column 0 must be the intercept, exactly one",
            ));
        }
        let mean = self.mean_at(context);
        for j in 0..g {
            genome[j] = mean[j]
                + (0..=j)
                    .map(|k| self.covariance_factor[j * g + k] * residual[k])
                    .sum::<f64>();
        }
        Ok(())
    }

    /// `eps_i = L^-1 (g_i - A c_i)` over the rows observing every score.
    pub fn whitened_residuals(
        &self,
        scores: &[Vec<Option<f64>>],
        context: ArrayView2<'_, f64>,
    ) -> Result<Vec<Vec<f64>>, EventHistoryError> {
        self.validate()?;
        let g = self.scores;
        if scores.len() != context.nrows()
            || context.ncols() != self.context_columns
            || scores.iter().any(|row| row.len() != g)
        {
            return Err(invalid(
                "score rows or context do not match the genetic score law",
            ));
        }
        let mut residuals = Vec::new();
        for (row, observed) in scores.iter().enumerate() {
            let Some(values) = observed.iter().copied().collect::<Option<Vec<f64>>>() else {
                continue;
            };
            let c = context.row(row).to_vec();
            if c.iter().chain(&values).any(|v| !v.is_finite()) || (g > 0 && c.first() != Some(&1.0))
            {
                return Err(invalid(
                    "scores and score context must be finite, with the intercept, exactly one, in column 0",
                ));
            }
            let mean = self.mean_at(&c);
            let mut residual = vec![0.0; g];
            for j in 0..g {
                let tail: f64 = (0..j)
                    .map(|k| self.covariance_factor[j * g + k] * residual[k])
                    .sum();
                residual[j] = (values[j] - mean[j] - tail) / self.covariance_factor[j * g + j];
            }
            residuals.push(residual);
        }
        Ok(residuals)
    }

    /// The observed-data log-likelihood of a score sample under this law, with its
    /// analytic gradient in the mean coefficients and the covariance factor.
    pub fn log_likelihood(
        &self,
        scores: &[Vec<Option<f64>>],
        context: ArrayView2<'_, f64>,
        weights: &[f64],
    ) -> Result<ScoreLawLikelihood, EventHistoryError> {
        self.validate()?;
        let p = context.ncols();
        let (g, patterns) = pattern_statistics(scores, context, weights)?;
        if g != self.scores || p != self.context_columns {
            return Err(invalid(
                "score sample does not match the genetic score law's dimensions",
            ));
        }
        let mean = Array2::from_shape_fn((g, p), |(j, b)| self.mean_coefficients[j * p + b]);
        let factor = self.factor();
        let covariance = factor.dot(&factor.t());
        let value = observed_log_likelihood(&patterns, &mean, &covariance)?;
        let mut mean_gradient = Array2::<f64>::zeros((g, p));
        let mut covariance_gradient = Array2::<f64>::zeros((g, g));
        for stats in &patterns {
            let o = &stats.present;
            let block = lower_factor(
                &submatrix(&covariance, o, o),
                "an observed score covariance block",
            )?;
            let observed_mean = mean.select(Axis(0), o);
            let fitted = observed_mean.dot(&stats.context);
            // sum w r c' and sum w r r' with r = g_o - A_o c.
            let residual_context = &stats.score_context - &fitted;
            let cross = observed_mean.dot(&stats.score_context.t());
            let residual = &stats.score - &cross - &cross.t() + fitted.dot(&observed_mean.t());
            let inverse = cholesky_solve_matrix(&block, &Array2::<f64>::eye(o.len()));
            // d/dA_o = S^-1 sum w r c';  d/dSigma_oo = (S^-1 (sum w r r') S^-1 - (sum w) S^-1) / 2.
            let mean_part = inverse.dot(&residual_context);
            let covariance_part =
                (inverse.dot(&residual).dot(&inverse) - &inverse * stats.weight) * 0.5;
            for (i, &j) in o.iter().enumerate() {
                for b in 0..p {
                    mean_gradient[[j, b]] += mean_part[[i, b]];
                }
                for (k, &l) in o.iter().enumerate() {
                    covariance_gradient[[j, l]] += covariance_part[[i, k]];
                }
            }
        }
        // Sigma = L L' gives d/dL = 2 (d/dSigma) L, read on the lower triangle.
        let factor_full = covariance_gradient.dot(&factor) * 2.0;
        let factor_gradient = (0..g * g)
            .map(|index| {
                let (i, j) = (index / g, index % g);
                if j <= i { factor_full[[i, j]] } else { 0.0 }
            })
            .collect();
        Ok(ScoreLawLikelihood {
            value,
            mean_gradient: mean_gradient.iter().copied().collect(),
            factor_gradient,
        })
    }

    /// The law of `g | c` from observed data with scores missing at random given the context.
    /// Each missingness pattern keeps only its weighted sufficient statistics, so an iteration
    /// costs nothing per row. An intercept-only context gets the certified maximum likelihood;
    /// otherwise the context columns enter under their learned prior, and the returned
    /// `ContextDecision` reports how they were decided.
    pub fn fit(
        scores: &[Vec<Option<f64>>],
        context: ArrayView2<'_, f64>,
        weights: &[f64],
    ) -> Result<ScoreLawFit, EventHistoryError> {
        let p = context.ncols();
        let (g, patterns) = pattern_statistics(scores, context, weights)?;
        if g == 0 {
            return Ok(ScoreLawFit {
                law: Self {
                    scores: 0,
                    context_columns: p,
                    mean_coefficients: Vec::new(),
                    covariance_factor: Vec::new(),
                },
                context: ContextDecision::InterceptOnly,
            });
        }
        let Some(prior) = ContextPrior::new(context, weights)? else {
            let (mean, covariance) = initial_law(&patterns, g, p)?;
            let (mean, covariance) = certified_fit(&patterns, mean, covariance)?;
            return Ok(ScoreLawFit {
                law: Self::from_parts(&mean, &covariance)?,
                context: ContextDecision::InterceptOnly,
            });
        };
        prior.learn(&patterns, g, p)
    }

    fn from_parts(mean: &Array2<f64>, covariance: &Array2<f64>) -> Result<Self, EventHistoryError> {
        let (g, p) = mean.dim();
        let factor = lower_factor(covariance, "the fitted genetic score covariance")?;
        let law = Self {
            scores: g,
            context_columns: p,
            mean_coefficients: mean.iter().copied().collect(),
            covariance_factor: factor.iter().copied().collect(),
        };
        law.validate()?;
        Ok(law)
    }
}

/// Per-score weighted least squares on the observed rows with a diagonal covariance:
/// EM's start. Refuses unidentified context spans and score pairs.
fn initial_law(
    patterns: &[PatternStatistics],
    g: usize,
    p: usize,
) -> Result<(Array2<f64>, Array2<f64>), EventHistoryError> {
        for k in 0..g {
            for l in 0..k {
                if !patterns
                    .iter()
                    .any(|stats| stats.present.contains(&k) && stats.present.contains(&l))
                {
                    return Err(invalid(format!(
                        "genetic scores {l} and {k} are never observed together, so their covariance is not identified"
                    )));
                }
            }
        }
        let mut mean = Array2::<f64>::zeros((g, p));
        let mut covariance = Array2::<f64>::zeros((g, g));
        for j in 0..g {
            let mut gram = Array2::<f64>::zeros((p, p));
            let mut cross = Array1::<f64>::zeros(p);
            let mut square = 0.0;
            let mut weight = 0.0;
            for stats in patterns {
                if let Some(i) = stats.present.iter().position(|&k| k == j) {
                    gram += &stats.context;
                    cross += &stats.score_context.row(i);
                    square += stats.score[[i, i]];
                    weight += stats.weight;
                }
            }
            let gram_factor = cholesky_factor_in_place(gram.view(), CholeskyGuard::FiniteStrict)
                .ok_or_else(|| {
                    invalid(format!(
                        "genetic score {j}: its observed rows do not identify the reference-context span"
                    ))
                })?;
            let coefficients = cholesky_solve_vector(&gram_factor, &cross);
            let variance = (square - coefficients.dot(&cross)) / weight;
            if !(variance > 0.0) {
                return Err(numerical(format!(
                    "genetic score {j} is an exact function of the reference context"
                )));
            }
            mean.row_mut(j).assign(&coefficients);
            covariance[[j, j]] = variance;
        }
        Ok((mean, covariance))
}

/// The fitted score law and how its context columns were decided.
#[derive(Clone, Debug, PartialEq)]
pub struct ScoreLawFit {
    pub law: GeneticScoreLaw,
    pub context: ContextDecision,
}

/// How the context columns entered the fitted law.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub enum ContextDecision {
    /// No context prior applies: the declared context is the intercept alone, or no score is
    /// declared.
    InterceptOnly,
    /// The family's boundary score lies below minus its bound: the evidence does not rise from
    /// the no-context limit.
    NoContext { score: f64, bound: f64 },
    /// The boundary score lies within its bound, a first-order tie; the no-context law is
    /// returned and the tie is reported.
    FirstOrderTie { score: f64, bound: f64 },
    /// The evidence rises from the limit to its optimum at `log_strength`, above the limit's
    /// evidence beyond their rounding band.
    Optimum {
        log_strength: f64,
        evidence: f64,
        limit_evidence: f64,
    },
    /// The evidence rises from the limit, but its optimum lies below the limit's evidence beyond
    /// their rounding band.
    LimitDominates {
        log_strength: f64,
        evidence: f64,
        limit_evidence: f64,
    },
    /// The evidence rises from the limit, and its optimum's evidence and the limit's agree within
    /// their rounding band; the no-context law is returned and the tie is reported.
    EvidenceTie { difference: f64, bound: f64 },
}

/// A score sample's observed-data log-likelihood and its gradient.
#[derive(Clone, Debug)]
pub struct ScoreLawLikelihood {
    pub value: f64,
    /// `d value / d A`, row-major like `mean_coefficients`.
    pub mean_gradient: Vec<f64>,
    /// `d value / d L` on the lower triangle, row-major like `covariance_factor`;
    /// the strict upper triangle is zero.
    pub factor_gradient: Vec<f64>,
}

/// Weighted sufficient statistics per missingness pattern. Rows observing no
/// score, or of zero weight, contribute a likelihood of one and are left out.
fn pattern_statistics(
    scores: &[Vec<Option<f64>>],
    context: ArrayView2<'_, f64>,
    weights: &[f64],
) -> Result<(usize, Vec<PatternStatistics>), EventHistoryError> {
    let (n, p) = context.dim();
    let g = scores.first().map_or(0, Vec::len);
    if scores.len() != n || weights.len() != n || scores.iter().any(|row| row.len() != g) {
        return Err(invalid(
            "score rows, context rows and weights need one entry per subject and a common score width",
        ));
    }
    if context
        .iter()
        .chain(scores.iter().flatten().flatten())
        .chain(weights)
        .any(|v| !v.is_finite())
        || weights.iter().any(|w| *w < 0.0)
    {
        return Err(invalid(
            "scores, score context and weights must be finite, with nonnegative weights",
        ));
    }
    if g > 0 && p == 0 {
        return Err(invalid(
            "a genetic score law needs at least one reference-context column",
        ));
    }
    if g > 0 && context.column(0).iter().any(|v| *v != 1.0) {
        return Err(invalid(
            "score context column 0 must be the intercept, exactly one",
        ));
    }
    let mut grouped: BTreeMap<Vec<bool>, PatternStatistics> = BTreeMap::new();
    for (row, observed) in scores.iter().enumerate() {
        let w = weights[row];
        if w == 0.0 || observed.iter().all(Option::is_none) {
            continue;
        }
        let key: Vec<bool> = observed.iter().map(Option::is_some).collect();
        let stats = grouped.entry(key).or_insert_with(|| {
            let (present, missing): (Vec<usize>, Vec<usize>) =
                (0..g).partition(|&j| observed[j].is_some());
            PatternStatistics {
                weight: 0.0,
                weight_error: 0.0,
                context: Array2::zeros((p, p)),
                context_error: Array2::zeros((p, p)),
                score_context: Array2::zeros((present.len(), p)),
                score_context_error: Array2::zeros((present.len(), p)),
                score: Array2::zeros((present.len(), present.len())),
                score_error: Array2::zeros((present.len(), present.len())),
                present,
                missing,
            }
        });
        let c = context.row(row);
        let values: Vec<f64> = stats
            .present
            .iter()
            .map(|&j| observed[j].unwrap_or(0.0))
            .collect();
        // Each row adds exact inputs through two products, so a term rounds by at most
        // 2 eps |term| and its addition by eps times the partial sum.
        stats.weight += w;
        stats.weight_error += f64::EPSILON * stats.weight.abs();
        for a in 0..p {
            for b in 0..p {
                let term = w * c[a] * c[b];
                stats.context[[a, b]] += term;
                stats.context_error[[a, b]] +=
                    f64::EPSILON * (2.0 * term.abs() + stats.context[[a, b]].abs());
            }
        }
        for (i, &v) in values.iter().enumerate() {
            for b in 0..p {
                let term = w * v * c[b];
                stats.score_context[[i, b]] += term;
                stats.score_context_error[[i, b]] +=
                    f64::EPSILON * (2.0 * term.abs() + stats.score_context[[i, b]].abs());
            }
            for (k, &u) in values.iter().enumerate() {
                let term = w * v * u;
                stats.score[[i, k]] += term;
                stats.score_error[[i, k]] +=
                    f64::EPSILON * (2.0 * term.abs() + stats.score[[i, k]].abs());
            }
        }
    }
    Ok((g, grouped.into_values().collect()))
}

/// Weighted sufficient statistics of the rows sharing one missingness pattern.
#[derive(Clone)]
struct PatternStatistics {
    present: Vec<usize>,
    missing: Vec<usize>,
    weight: f64,
    /// `sum w c c'`.
    context: Array2<f64>,
    /// `sum w g_o c'`.
    score_context: Array2<f64>,
    /// `sum w g_o g_o'`.
    score: Array2<f64>,
    /// First-order rounding errors of the accumulated sums above.
    weight_error: f64,
    context_error: Array2<f64>,
    score_context_error: Array2<f64>,
    score_error: Array2<f64>,
}

fn observed_log_likelihood(
    patterns: &[PatternStatistics],
    mean: &Array2<f64>,
    covariance: &Array2<f64>,
) -> Result<f64, EventHistoryError> {
    let mut total = 0.0;
    for stats in patterns {
        let o = &stats.present;
        let block = lower_factor(
            &submatrix(covariance, o, o),
            "an observed score covariance block",
        )?;
        let observed_mean = mean.select(Axis(0), o);
        let cross = observed_mean.dot(&stats.score_context.t());
        let residual = &stats.score - &cross - &cross.t()
            + observed_mean.dot(&stats.context).dot(&observed_mean.t());
        let solved = cholesky_solve_matrix(&block, &residual);
        total -= 0.5
            * (stats.weight * (o.len() as f64 * TAU.ln() + factor_log_determinant(&block))
                + solved.diag().sum());
    }
    if !total.is_finite() {
        return Err(numerical("genetic score log-likelihood is not finite"));
    }
    Ok(total)
}

/// One EM step: each pattern's missing scores enter through their conditional
/// expectation and covariance given its observed scores and context.
fn expectation_maximization(
    patterns: &[PatternStatistics],
    mean: &Array2<f64>,
    covariance: &Array2<f64>,
) -> Result<(Array2<f64>, Array2<f64>), EventHistoryError> {
    let (g, p) = mean.dim();
    let mut context = Array2::<f64>::zeros((p, p));
    let mut score_context = Array2::<f64>::zeros((g, p));
    let mut score = Array2::<f64>::zeros((g, g));
    let mut weight = 0.0;
    for stats in patterns {
        let (o, m) = (&stats.present, &stats.missing);
        context += &stats.context;
        weight += stats.weight;
        for (i, &j) in o.iter().enumerate() {
            for b in 0..p {
                score_context[[j, b]] += stats.score_context[[i, b]];
            }
            for (k, &l) in o.iter().enumerate() {
                score[[j, l]] += stats.score[[i, k]];
            }
        }
        if m.is_empty() {
            continue;
        }
        let block = lower_factor(
            &submatrix(covariance, o, o),
            "an observed score covariance block",
        )?;
        let observed_missing = submatrix(covariance, o, m);
        // E[g_m | g_o, c] = D c + R g_o with R = Sigma_mo Sigma_oo^-1, D = A_m - R A_o.
        let regression = cholesky_solve_matrix(&block, &observed_missing)
            .t()
            .to_owned();
        let conditional = submatrix(covariance, m, m) - regression.dot(&observed_missing);
        let offset = mean.select(Axis(0), m) - regression.dot(&mean.select(Axis(0), o));
        let missing_context =
            offset.dot(&stats.context) + regression.dot(&stats.score_context);
        let missing_observed =
            offset.dot(&stats.score_context.t()) + regression.dot(&stats.score);
        let missing_missing = &conditional * stats.weight
            + offset.dot(&stats.context).dot(&offset.t())
            + offset
                .dot(&stats.score_context.t())
                .dot(&regression.t())
            + regression.dot(&stats.score_context).dot(&offset.t())
            + regression.dot(&stats.score).dot(&regression.t());
        for (a, &j) in m.iter().enumerate() {
            for b in 0..p {
                score_context[[j, b]] += missing_context[[a, b]];
            }
            for (i, &l) in o.iter().enumerate() {
                score[[j, l]] += missing_observed[[a, i]];
                score[[l, j]] += missing_observed[[a, i]];
            }
            for (b, &l) in m.iter().enumerate() {
                score[[j, l]] += missing_missing[[a, b]];
            }
        }
    }
    let context_factor = lower_factor(&context, "the reference-context Gram matrix")?;
    let next_mean = cholesky_solve_matrix(&context_factor, score_context.t())
        .t()
        .to_owned();
    let residual = (score - next_mean.dot(&score_context.t())) / weight;
    let next_covariance = (&residual + &residual.t()) * 0.5;
    Ok((next_mean, next_covariance))
}

/// EM to its monotone stall, the first step that no longer raises the computed
/// likelihood. From any start this reaches the mode's neighbourhood; it is not a
/// convergence certificate.
fn expectation_maximization_limit(
    patterns: &[PatternStatistics],
    mut mean: Array2<f64>,
    mut covariance: Array2<f64>,
) -> Result<(Array2<f64>, Array2<f64>), EventHistoryError> {
    let mut log_likelihood = observed_log_likelihood(patterns, &mean, &covariance)?;
    // EM never decreases the observed-data likelihood, which is bounded above while
    // the covariance stays positive definite; a strictly increasing sequence of
    // doubles below a bound is finite.
    loop {
        let (next_mean, next_covariance) = expectation_maximization(patterns, &mean, &covariance)?;
        let next = observed_log_likelihood(patterns, &next_mean, &next_covariance)?;
        if !(next > log_likelihood) {
            return Ok((mean, covariance));
        }
        mean = next_mean;
        covariance = next_covariance;
        log_likelihood = next;
    }
}

/// The observed-data score and Hessian in `(A, vech Sigma)`, with first-order bounds on
/// each score entry's and each Hessian entry's rounding.
struct ScoreCurvature {
    gradient: Vec<f64>,
    band: Vec<f64>,
    hessian: Array2<f64>,
    hessian_error: Array2<f64>,
}

impl ScoreCurvature {
    /// The largest `|score| / roundoff bound` over the parameters.
    fn largest_ratio(&self) -> f64 {
        self.gradient
            .iter()
            .zip(&self.band)
            .map(|(score, band)| score.abs() / band)
            .fold(0.0_f64, f64::max)
    }
}

/// Position of `Sigma_kl` in `(A, vech Sigma)`, lower triangle row-major after `A`.
fn covariance_index(g: usize, p: usize, k: usize, l: usize) -> usize {
    let (high, low) = if k >= l { (k, l) } else { (l, k) };
    g * p + high * (high + 1) / 2 + low
}

/// Per pattern, with `P = Sigma_oo^-1`, `E = sum w r c'` and `R = sum w r r'`:
///
/// ```text
///     d/dA_o = P E,     d/dSigma_oo = P (R - n Sigma_oo) P / 2,
///     D2[(dA_1, dS_1), (dA_2, dS_2)] = - tr(S_cc dA_1' P dA_2)
///         - tr(E' P dS_1 P dA_2) - tr(E' P dS_2 P dA_1) - tr(M dS_1 P dS_2),
///     M = P R P - n P / 2.
/// ```
///
/// `R - n Sigma_oo` is formed on the statistics, so its cancellation at the mode
/// happens before any solve and the solves only scale a small matrix.
fn score_curvature(
    patterns: &[PatternStatistics],
    mean: &Array2<f64>,
    covariance: &Array2<f64>,
) -> Result<ScoreCurvature, EventHistoryError> {
    let (g, p) = mean.dim();
    let dimension = g * p + g * (g + 1) / 2;
    let mut gradient = vec![0.0; dimension];
    let mut band = vec![0.0; dimension];
    let mut hessian = Array2::<f64>::zeros((dimension, dimension));
    let mut hessian_error = Array2::<f64>::zeros((dimension, dimension));
    // Every piece carries its first-order rounding error: the statistics' sums, the observed
    // block's inverse in the theorem form, and each product and sum by its running error. The
    // band is the accumulated score's error.
    for stats in patterns {
        let o = &stats.present;
        let pieces = pattern_pieces(stats, mean, covariance)?;
        let inverse = (&pieces.inverse, &pieces.inverse_error);
        let (weighted, weighted_error) = product_with_error(
            inverse.0,
            inverse.1,
            &pieces.residual_context,
            &pieces.residual_context_error,
        );
        let (inverse_excess, inverse_excess_error) =
            product_with_error(inverse.0, inverse.1, &pieces.excess, &pieces.excess_error);
        let (scaled_excess, scaled_excess_error) =
            product_with_error(&inverse_excess, &inverse_excess_error, inverse.0, inverse.1);
        // P R P - n P / 2 = P (R - n S) P + n P / 2.
        let half_weight = 0.5 * stats.weight;
        let scaled_inverse = inverse.0 * half_weight;
        let scaled_inverse_error = Array2::from_shape_fn(inverse.0.dim(), |(r, c)| {
            product_error(inverse.0[[r, c]], inverse.1[[r, c]], half_weight, 0.5 * stats.weight_error)
        });
        let (moment, moment_error) = sum_with_error(
            &scaled_inverse,
            &scaled_inverse_error,
            &scaled_excess,
            &scaled_excess_error,
        );

        for (a, &j) in o.iter().enumerate() {
            for b in 0..p {
                let index = j * p + b;
                gradient[index] += weighted[[a, b]];
                band[index] += weighted_error[[a, b]] + f64::EPSILON * gradient[index].abs();
            }
            for (c, &l) in o.iter().enumerate().take(a + 1) {
                // The spread P (R - n S) P / 2 and the multiplicity are exact scalings.
                let multiplicity = if a == c { 0.5 } else { 1.0 };
                let index = covariance_index(g, p, j, l);
                gradient[index] += multiplicity * scaled_excess[[a, c]];
                band[index] += multiplicity * scaled_excess_error[[a, c]]
                    + f64::EPSILON * gradient[index].abs();
            }
        }
        assemble(
            o,
            g,
            p,
            &CurvatureTerms {
                inverse: Some(inverse),
                weighted: (&weighted, &weighted_error),
                moment: (&moment, &moment_error),
            },
            inverse,
            (&stats.context, &stats.context_error),
            (&mut hessian, Some(&mut hessian_error)),
        );
    }
    if gradient
        .iter()
        .chain(&band)
        .chain(hessian.iter())
        .chain(hessian_error.iter())
        .any(|v| !v.is_finite())
    {
        return Err(numerical("genetic score law curvature is not finite"));
    }
    Ok(ScoreCurvature {
        gradient,
        band,
        hessian,
        hessian_error,
    })
}

fn not_stationary(ratio: f64, reason: &str) -> EventHistoryError {
    EventHistoryError::Fit {
        reason: format!(
            "genetic score law fit is not stationary: its largest score is {ratio:.3e} times its roundoff bound, and {reason}"
        ),
    }
}

/// Newton on the observed-data likelihood from the EM limit. The fit is returned
/// only where every score entry lies within its roundoff bound.
fn newton_finish(
    patterns: &[PatternStatistics],
    mut mean: Array2<f64>,
    mut covariance: Array2<f64>,
) -> Result<(Array2<f64>, Array2<f64>), EventHistoryError> {
    let (g, p) = mean.dim();
    let mut curvature = score_curvature(patterns, &mean, &covariance)?;
    let mut ratio = curvature.largest_ratio();
    // Each accepted step strictly lowers the ratio, and a strictly decreasing
    // sequence of positive doubles is finite: this ends at the certificate or at a
    // typed refusal.
    while ratio > 1.0 {
        let information = curvature.hessian.mapv(|v| -v);
        let factor = cholesky_factor_in_place(information.view(), CholeskyGuard::FiniteStrict)
            .ok_or_else(|| {
                not_stationary(ratio, "the observed information is not positive definite")
            })?;
        let step = cholesky_solve_vector(&factor, &Array1::from(curvature.gradient.clone()));
        let next_mean = Array2::from_shape_fn((g, p), |(j, b)| mean[[j, b]] + step[j * p + b]);
        let next_covariance = Array2::from_shape_fn((g, g), |(k, l)| {
            covariance[[k, l]] + step[covariance_index(g, p, k, l)]
        });
        let next = score_curvature(patterns, &next_mean, &next_covariance).map_err(|error| {
            not_stationary(ratio, &format!("the Newton step leaves the law: {error}"))
        })?;
        let next_ratio = next.largest_ratio();
        if !(next_ratio < ratio) {
            return Err(not_stationary(
                ratio,
                "a Newton step no longer reduces the score",
            ));
        }
        mean = next_mean;
        covariance = next_covariance;
        curvature = next;
        ratio = next_ratio;
    }
    Ok((mean, covariance))
}

/// EM to its stall, then the certified Newton finish.
fn certified_fit(
    patterns: &[PatternStatistics],
    mean: Array2<f64>,
    covariance: Array2<f64>,
) -> Result<(Array2<f64>, Array2<f64>), EventHistoryError> {
    let (mean, covariance) = expectation_maximization_limit(patterns, mean, covariance)?;
    newton_finish(patterns, mean, covariance)
}

/// One pattern's pieces at a law, with the first-order errors of the computed ones. The law's
/// parameters are exact operands.
struct PatternPieces {
    /// `Sigma_oo`.
    covariance: Array2<f64>,
    /// `P = Sigma_oo^-1`.
    inverse: Array2<f64>,
    inverse_error: Array2<f64>,
    /// `E = sum w r c'`.
    residual_context: Array2<f64>,
    residual_context_error: Array2<f64>,
    /// `R - n Sigma_oo`, formed on the statistics.
    excess: Array2<f64>,
    excess_error: Array2<f64>,
}

/// `a + b` entrywise with first-order errors.
fn sum_with_error(
    a: &Array2<f64>,
    a_error: &Array2<f64>,
    b: &Array2<f64>,
    b_error: &Array2<f64>,
) -> (Array2<f64>, Array2<f64>) {
    let value = a + b;
    let error = a_error + b_error + &value.mapv(|v| f64::EPSILON * v.abs());
    (value, error)
}

fn pattern_pieces(
    stats: &PatternStatistics,
    mean: &Array2<f64>,
    covariance: &Array2<f64>,
) -> Result<PatternPieces, EventHistoryError> {
    let o = &stats.present;
    let observed_covariance = submatrix(covariance, o, o);
    let (inverse, inverse_error) = inverse_with_error(
        &observed_covariance,
        &Array2::<f64>::zeros((o.len(), o.len())),
        "an observed score covariance block",
    )?;
    let observed_mean = mean.select(Axis(0), o);
    let mean_error = Array2::<f64>::zeros(observed_mean.dim());
    let (fitted, fitted_error) =
        product_with_error(&observed_mean, &mean_error, &stats.context, &stats.context_error);
    let (residual_context, residual_context_error) = sum_with_error(
        &stats.score_context,
        &stats.score_context_error,
        &fitted.mapv(|v| -v),
        &fitted_error,
    );
    let (cross, cross_error) = product_with_error(
        &observed_mean,
        &mean_error,
        &stats.score_context.t().to_owned(),
        &stats.score_context_error.t().to_owned(),
    );
    let (explained, explained_error) = product_with_error(
        &fitted,
        &fitted_error,
        &observed_mean.t().to_owned(),
        &mean_error.t().to_owned(),
    );
    let scaled = &observed_covariance * stats.weight;
    let scaled_error = observed_covariance
        .mapv(|v| v.abs() * stats.weight_error + f64::EPSILON * (v * stats.weight).abs());
    let (excess, excess_error) =
        sum_with_error(&stats.score, &stats.score_error, &cross.mapv(|v| -v), &cross_error);
    let (excess, excess_error) = sum_with_error(
        &excess,
        &excess_error,
        &cross.t().mapv(|v| -v),
        &cross_error.t().to_owned(),
    );
    let (excess, excess_error) = sum_with_error(&excess, &excess_error, &explained, &explained_error);
    let (excess, excess_error) =
        sum_with_error(&excess, &excess_error, &scaled.mapv(|v| -v), &scaled_error);
    Ok(PatternPieces {
        covariance: observed_covariance,
        inverse,
        inverse_error,
        residual_context,
        residual_context_error,
        excess,
        excess_error,
    })
}

/// The factors of one pattern's Hessian that `assemble` pairs with a second inverse, each as
/// `(value, first-order error)`.
struct CurvatureTerms<'a> {
    inverse: Option<(&'a Array2<f64>, &'a Array2<f64>)>,
    weighted: (&'a Array2<f64>, &'a Array2<f64>),
    moment: (&'a Array2<f64>, &'a Array2<f64>),
}

/// Adds `-P_1 ⊗ S_cc` on the A block, `-(PE)_1 P_2` across and `-M_1 P_2` on the
/// Sigma block, over both orientations of each covariance pair. The Hessian is
/// `assemble(P, PE, M; P)`; its directional derivative is
/// `assemble(dP, dPE, dM; P) + assemble(-, PE, M; dP)`. Every operand is
/// `(value, first-order error)`, and `target` accumulates the Hessian, with its errors when an
/// error matrix is given.
fn assemble(
    o: &[usize],
    g: usize,
    p: usize,
    first: &CurvatureTerms<'_>,
    second_inverse: (&Array2<f64>, &Array2<f64>),
    context: (&Array2<f64>, &Array2<f64>),
    target: (&mut Array2<f64>, Option<&mut Array2<f64>>),
) {
    let (hessian, mut error) = target;
    let mut subtract = |entry: [usize; 2], term: f64, term_error: f64| {
        hessian[entry] -= term;
        if let Some(error) = error.as_deref_mut() {
            error[entry] += term_error + f64::EPSILON * hessian[entry].abs();
        }
    };
    if let Some((inverse, inverse_error)) = first.inverse {
        for (a, &j) in o.iter().enumerate() {
            for (c, &l) in o.iter().enumerate() {
                for b in 0..p {
                    for d in 0..p {
                        subtract(
                            [j * p + b, l * p + d],
                            inverse[[a, c]] * context.0[[b, d]],
                            product_error(
                                inverse[[a, c]],
                                inverse_error[[a, c]],
                                context.0[[b, d]],
                                context.1[[b, d]],
                            ),
                        );
                    }
                }
            }
        }
    }
    let oriented = |a: usize, c: usize| -> Vec<(usize, usize)> {
        if a == c { vec![(a, c)] } else { vec![(a, c), (c, a)] }
    };
    let pairs: Vec<(usize, usize)> = (0..o.len())
        .flat_map(|a| (0..=a).map(move |c| (a, c)))
        .collect();
    for &(a, c) in &pairs {
        let row = covariance_index(g, p, o[a], o[c]);
        for (e, &j) in o.iter().enumerate() {
            for b in 0..p {
                let mut value = RunningSum::default();
                for &(alpha, beta) in &oriented(a, c) {
                    value.add(
                        first.weighted.0[[alpha, b]] * second_inverse.0[[beta, e]],
                        product_error(
                            first.weighted.0[[alpha, b]],
                            first.weighted.1[[alpha, b]],
                            second_inverse.0[[beta, e]],
                            second_inverse.1[[beta, e]],
                        ),
                    );
                }
                subtract([row, j * p + b], value.value, value.error);
                subtract([j * p + b, row], value.value, value.error);
            }
        }
        for &(e, f) in &pairs {
            let column = covariance_index(g, p, o[e], o[f]);
            let mut value = RunningSum::default();
            for &(alpha, beta) in &oriented(a, c) {
                let mut inner = RunningSum::default();
                for &(x, y) in &oriented(e, f) {
                    inner.add(
                        first.moment.0[[y, alpha]] * second_inverse.0[[beta, x]],
                        product_error(
                            first.moment.0[[y, alpha]],
                            first.moment.1[[y, alpha]],
                            second_inverse.0[[beta, x]],
                            second_inverse.1[[beta, x]],
                        ),
                    );
                }
                value.add(inner.value, inner.error);
            }
            subtract([row, column], value.value, value.error);
        }
    }
}

/// The directional derivative of `score_curvature`'s Hessian along `direction` in
/// `(A, vech Sigma)`: `dP = -P dS P`, `dE = -dA S_cc`, `dR = -dA E' - E dA'`,
/// `dM = dP R P + P dR P + P R dP - n dP / 2`.
fn hessian_direction(
    patterns: &[PatternStatistics],
    mean: &Array2<f64>,
    covariance: &Array2<f64>,
    direction: &[f64],
    direction_error: &[f64],
) -> Result<(Array2<f64>, Array2<f64>), EventHistoryError> {
    let (g, p) = mean.dim();
    let dimension = direction.len();
    let d_mean = Array2::from_shape_fn((g, p), |(j, b)| direction[j * p + b]);
    let d_mean_error = Array2::from_shape_fn((g, p), |(j, b)| direction_error[j * p + b]);
    let d_covariance =
        Array2::from_shape_fn((g, g), |(k, l)| direction[covariance_index(g, p, k, l)]);
    let d_covariance_error =
        Array2::from_shape_fn((g, g), |(k, l)| direction_error[covariance_index(g, p, k, l)]);
    let mut result = Array2::<f64>::zeros((dimension, dimension));
    let mut result_error = Array2::<f64>::zeros((dimension, dimension));
    let negated = |(value, error): (Array2<f64>, Array2<f64>)| (value.mapv(|v| -v), error);
    for stats in patterns {
        let o = &stats.present;
        let pieces = pattern_pieces(stats, mean, covariance)?;
        let inverse = (&pieces.inverse, &pieces.inverse_error);
        let half_weight = 0.5 * stats.weight;
        let scaled = |(value, error): (&Array2<f64>, &Array2<f64>)| {
            (
                value * half_weight,
                Array2::from_shape_fn(value.dim(), |(r, c)| {
                    product_error(value[[r, c]], error[[r, c]], half_weight, 0.5 * stats.weight_error)
                }),
            )
        };
        let (inverse_direction, inverse_direction_error) = product_with_error(
            inverse.0,
            inverse.1,
            &submatrix(&d_covariance, o, o),
            &submatrix(&d_covariance_error, o, o),
        );
        let (d_inverse, d_inverse_error) = negated(product_with_error(
            &inverse_direction,
            &inverse_direction_error,
            inverse.0,
            inverse.1,
        ));
        let d_observed_mean = d_mean.select(Axis(0), o);
        let d_observed_mean_error = d_mean_error.select(Axis(0), o);
        let (d_residual_context, d_residual_context_error) = negated(product_with_error(
            &d_observed_mean,
            &d_observed_mean_error,
            &stats.context,
            &stats.context_error,
        ));
        let (weighted, weighted_error) = product_with_error(
            inverse.0,
            inverse.1,
            &pieces.residual_context,
            &pieces.residual_context_error,
        );
        let (moved_inverse, moved_inverse_error) = product_with_error(
            &d_inverse,
            &d_inverse_error,
            &pieces.residual_context,
            &pieces.residual_context_error,
        );
        let (moved_residual, moved_residual_error) =
            product_with_error(inverse.0, inverse.1, &d_residual_context, &d_residual_context_error);
        let (d_weighted, d_weighted_error) = sum_with_error(
            &moved_inverse,
            &moved_inverse_error,
            &moved_residual,
            &moved_residual_error,
        );
        let weighted_covariance = &pieces.covariance * stats.weight;
        let weighted_covariance_error = pieces
            .covariance
            .mapv(|v| v.abs() * stats.weight_error + f64::EPSILON * (v * stats.weight).abs());
        let (residual, residual_error) = sum_with_error(
            &pieces.excess,
            &pieces.excess_error,
            &weighted_covariance,
            &weighted_covariance_error,
        );
        // dR = -dA E' - E dA'.
        let (left, left_error) = negated(product_with_error(
            &d_observed_mean,
            &d_observed_mean_error,
            &pieces.residual_context.t().to_owned(),
            &pieces.residual_context_error.t().to_owned(),
        ));
        let (right, right_error) = product_with_error(
            &pieces.residual_context,
            &pieces.residual_context_error,
            &d_observed_mean.t().to_owned(),
            &d_observed_mean_error.t().to_owned(),
        );
        let (d_residual, d_residual_error) =
            sum_with_error(&left, &left_error, &right.mapv(|v| -v), &right_error);
        // M = P R P - n P / 2.
        let (inverse_residual, inverse_residual_error) =
            product_with_error(inverse.0, inverse.1, &residual, &residual_error);
        let (sandwich, sandwich_error) =
            product_with_error(&inverse_residual, &inverse_residual_error, inverse.0, inverse.1);
        let (scaled_inverse, scaled_inverse_error) = scaled(inverse);
        let (moment, moment_error) = sum_with_error(
            &sandwich,
            &sandwich_error,
            &scaled_inverse.mapv(|v| -v),
            &scaled_inverse_error,
        );
        // dM = dP R P + P dR P + P R dP - n dP / 2.
        let (d_inverse_residual, d_inverse_residual_error) =
            product_with_error(&d_inverse, &d_inverse_error, &residual, &residual_error);
        let (first_term, first_term_error) = product_with_error(
            &d_inverse_residual,
            &d_inverse_residual_error,
            inverse.0,
            inverse.1,
        );
        let (inverse_d_residual, inverse_d_residual_error) =
            product_with_error(inverse.0, inverse.1, &d_residual, &d_residual_error);
        let (second_term, second_term_error) = product_with_error(
            &inverse_d_residual,
            &inverse_d_residual_error,
            inverse.0,
            inverse.1,
        );
        let (third_term, third_term_error) = product_with_error(
            &inverse_residual,
            &inverse_residual_error,
            &d_inverse,
            &d_inverse_error,
        );
        let (scaled_d_inverse, scaled_d_inverse_error) = scaled((&d_inverse, &d_inverse_error));
        let (d_moment, d_moment_error) =
            sum_with_error(&first_term, &first_term_error, &second_term, &second_term_error);
        let (d_moment, d_moment_error) =
            sum_with_error(&d_moment, &d_moment_error, &third_term, &third_term_error);
        let (d_moment, d_moment_error) = sum_with_error(
            &d_moment,
            &d_moment_error,
            &scaled_d_inverse.mapv(|v| -v),
            &scaled_d_inverse_error,
        );
        assemble(
            o,
            g,
            p,
            &CurvatureTerms {
                inverse: Some((&d_inverse, &d_inverse_error)),
                weighted: (&d_weighted, &d_weighted_error),
                moment: (&d_moment, &d_moment_error),
            },
            inverse,
            (&stats.context, &stats.context_error),
            (&mut result, Some(&mut result_error)),
        );
        assemble(
            o,
            g,
            p,
            &CurvatureTerms {
                inverse: None,
                weighted: (&weighted, &weighted_error),
                moment: (&moment, &moment_error),
            },
            (&d_inverse, &d_inverse_error),
            (&stats.context, &stats.context_error),
            (&mut result, Some(&mut result_error)),
        );
    }
    Ok((result, result_error))
}

/// The Laplace evidence of the score law at a certified mode over `(A, vech Sigma)`,
/// and with a context prior its exact derivative in `rho = log lambda`:
///
/// ```text
///     log Z = J + (G P_c / 2) rho + (G / 2) log det G_ctx + (d / 2) log 2pi - log det(-H) / 2,
///     d log Z / d rho = -tr(P R_prior) / 2 + G P_c / 2
///         + tr(V dH/drho) / 2 + tr(V D_theta H[V dg/drho]) / 2,
/// ```
///
/// `J` the likelihood with the prior's pseudo-pattern, `V = (-H)^-1`, and the last
/// term the mode's movement with the strength.
fn laplace_evidence(
    data: &[PatternStatistics],
    prior: Option<(&ContextPrior, f64)>,
    mean: &Array2<f64>,
    covariance: &Array2<f64>,
) -> Result<(f64, f64), EventHistoryError> {
    let (g, p) = mean.dim();
    let mut patterns = data.to_vec();
    if let Some((context_prior, rho)) = prior {
        patterns.push(context_prior.pattern(g, rho.exp()));
    }
    let value = observed_log_likelihood(&patterns, mean, covariance)?;
    let curvature = score_curvature(&patterns, mean, covariance)?;
    let dimension = curvature.gradient.len();
    let information = curvature.hessian.mapv(|v| -v);
    let information_factor = lower_factor(&information, "the score law's observed information")?;
    let posterior = cholesky_solve_matrix(&information_factor, &Array2::<f64>::eye(dimension));
    let mut log_evidence = value + 0.5 * dimension as f64 * TAU.ln()
        - 0.5 * factor_log_determinant(&information_factor);
    let Some((context_prior, rho)) = prior else {
        return Ok((log_evidence, 0.0));
    };
    let prior_dimension = (g * context_prior.columns) as f64;
    log_evidence += 0.5 * prior_dimension * rho + 0.5 * g as f64 * context_prior.log_det_gram;
    let pseudo = context_prior.pattern(g, rho.exp());
    let pieces = pattern_pieces(&pseudo, mean, covariance)?;
    let inverse = &pieces.inverse;
    let penalty = &pieces.excess + &(&pieces.covariance * pseudo.weight);
    let weighted = inverse.dot(&pieces.residual_context);
    let scaled_penalty = inverse.dot(&penalty).dot(inverse);
    let envelope = -0.5 * (inverse * &penalty).sum();
    // This value and its derivative in rho are the optimizer's objective, and their rounding is not
    // carried: the operand errors below are placeholders and nothing accumulates, so opt::Bfgs's
    // convergence on this derivative is not resolution-checked. The context decision compares
    // evidences through `evidence_with_error` instead.
    let placeholder = |matrix: &Array2<f64>| Array2::<f64>::zeros(matrix.dim());
    let mut force = Array1::<f64>::zeros(dimension);
    for j in 0..g {
        for b in 0..p {
            force[j * p + b] = weighted[[j, b]];
        }
        for l in 0..=j {
            let multiplicity = if j == l { 1.0 } else { 2.0 };
            force[covariance_index(g, p, j, l)] = multiplicity * 0.5 * scaled_penalty[[j, l]];
        }
    }
    let mut explicit = Array2::<f64>::zeros((dimension, dimension));
    assemble(
        &pseudo.present,
        g,
        p,
        &CurvatureTerms {
            inverse: Some((inverse, &placeholder(inverse))),
            weighted: (&weighted, &placeholder(&weighted)),
            moment: (&scaled_penalty, &placeholder(&scaled_penalty)),
        },
        (inverse, &placeholder(inverse)),
        (&pseudo.context, &placeholder(&pseudo.context)),
        (&mut explicit, None),
    );
    let movement = posterior.dot(&force).to_vec();
    let implicit =
        hessian_direction(&patterns, mean, covariance, &movement, &vec![0.0; movement.len()])?.0;
    let derivative = envelope
        + 0.5 * prior_dimension
        + 0.5 * (&posterior * &(explicit + implicit)).sum();
    if !log_evidence.is_finite() || !derivative.is_finite() {
        return Err(numerical("genetic score law evidence is not finite"));
    }
    Ok((log_evidence, derivative))
}

/// The Laplace evidence at a certified mode along a route that carries its first-order error, for
/// the context decision's comparison of evidences: `(value, error)`. First order throughout.
///
/// - The likelihood, per pattern `-(n (o log 2pi + log det Sigma_oo) + tr(Sigma_oo^-1 R)) / 2` with
///   `R = excess + n Sigma_oo`: the trace through `pattern_pieces`' inline errors and the observed
///   block's certified inverse; each log determinant, of the observed blocks, of the information
///   and of the centered Gram, by `cholesky_resolved` at `Running` from entries carrying their
///   inline errors (`running_with_error`), refusing any pivot not positive beyond its own rounding;
///   `log 2pi` over `Running` from `TAU`; `rho` the optimizer's exact input. The information's
///   inline errors form their only solves, each pattern's `Sigma_oo^-1`, through
///   `inverse_with_error`'s certificate in `pattern_pieces`, and the Gram's form none, so seeding
///   `Running` from them carries errors accumulated by Running's rules and injects nothing. The
///   mode's covariance entries enter as exact operands for two reasons: they are the evaluation
///   point `theta^`, whose distance to the exact mode the mode channel below charges, and `Sigma`
///   is itself a coordinate (`vech Sigma`), not formed from a factor.
/// - A `covariance_weight` `w` adds `w (g log 2pi + log det Sigma) / 2`, the no-context limit's
///   normalizer remnant.
/// - The certified mode's error. To first order the exact mode `theta*` and the computed `theta^`
///   differ by `delta = V r`, with `r` the exact score at `theta^` and `|r| <= |s^| + band`, so
///   `|delta| <= |V| (|s^| + band)`, the mode box. The likelihood there changes by
///   `J(theta^) - J(theta*) = delta' H delta / 2 = -r' V r / 2`, at most `|r|' |V| |r| / 2`, charged as
///   `|r|' |V| |r|`. `-log det I / 2` changes by `tr(V dH[delta]) / 2`, at most
///   `sum_k |tr(V dH_k)| |delta_k| / 2`, with one `hessian_direction` per coordinate at the decision
///   only. The remnant `w log det Sigma / 2` changes by `w tr(Sigma^-1 dSigma[delta]) / 2`, at most
///   `w sum_k multiplicity_k |Sigma^-1_kl| |delta_k| / 2` over the vech coordinates.
fn evidence_with_error(
    data: &[PatternStatistics],
    prior: Option<(&ContextPrior, f64)>,
    covariance_weight: usize,
    mean: &Array2<f64>,
    covariance: &Array2<f64>,
) -> Result<(f64, f64), EventHistoryError> {
    let (g, p) = mean.dim();
    let mut patterns = data.to_vec();
    if let Some((context_prior, rho)) = prior {
        patterns.push(context_prior.pattern(g, rho.exp()));
    }
    let zero = Running::exact(0.0);
    let ln_two_pi = ln(&zero.constant_like(TAU));
    let mut evidence = zero;
    for stats in &patterns {
        let pieces = pattern_pieces(stats, mean, covariance)?;
        let log_det_block = log_determinant_over(
            &zero,
            &cholesky_resolved(
                Running::rounding,
                &zero,
                &pieces.covariance.mapv(Running::exact),
                "an observed score covariance block",
            )?,
        );
        let weighted_covariance = &pieces.covariance * stats.weight;
        let weighted_covariance_error = pieces
            .covariance
            .mapv(|v| v.abs() * stats.weight_error + f64::EPSILON * (v * stats.weight).abs());
        let (residual, residual_error) = sum_with_error(
            &pieces.excess,
            &pieces.excess_error,
            &weighted_covariance,
            &weighted_covariance_error,
        );
        let (trace, trace_error) =
            contraction_with_error(&pieces.inverse, &pieces.inverse_error, &residual, &residual_error);
        let normalizer = running_with_error(stats.weight, stats.weight_error)
            .mul(&ln_two_pi.scale(stats.present.len() as f64).add(&log_det_block));
        evidence = evidence.add(&normalizer.add(&running_with_error(trace, trace_error)).scale(-0.5));
    }
    let curvature = score_curvature(&patterns, mean, covariance)?;
    let dimension = curvature.gradient.len();
    let information = curvature.hessian.mapv(|v| -v);
    let information_entries = Array2::from_shape_fn((dimension, dimension), |(i, j)| {
        running_with_error(information[[i, j]], curvature.hessian_error[[i, j]])
    });
    let log_det_information = log_determinant_over(
        &zero,
        &cholesky_resolved(Running::rounding, &zero, &information_entries, "the score law's observed information")?,
    );
    evidence = evidence
        .add(&ln_two_pi.scale(0.5 * dimension as f64))
        .sub(&log_det_information.scale(0.5));
    if let Some((context_prior, rho)) = prior {
        let columns = context_prior.columns;
        let gram = Array2::from_shape_fn((columns, columns), |(a, b)| {
            running_with_error(
                context_prior.gram[[a + 1, b + 1]],
                context_prior.gram_error[[a + 1, b + 1]],
            )
        });
        let log_det_gram =
            log_determinant_over(&zero, &cholesky_resolved(Running::rounding, &zero, &gram, "the centered context Gram")?);
        evidence = evidence
            .add(&Running::exact(rho).scale(0.5 * (g * columns) as f64))
            .add(&log_det_gram.scale(0.5 * g as f64));
    }
    let covariance_inverse = if covariance_weight > 0 {
        let log_det_covariance = log_determinant_over(
            &zero,
            &cholesky_resolved(
                Running::rounding,
                &zero,
                &covariance.mapv(Running::exact),
                "the no-context limit's covariance",
            )?,
        );
        evidence = evidence.add(
            &ln_two_pi
                .scale((covariance_weight * g) as f64)
                .add(&log_det_covariance.scale(covariance_weight as f64))
                .scale(0.5),
        );
        Some(
            inverse_with_error(
                covariance,
                &Array2::<f64>::zeros((g, g)),
                "the no-context limit's covariance",
            )?
            .0,
        )
    } else {
        None
    };
    let posterior =
        inverse_with_error(&information, &curvature.hessian_error, "the score law's observed information")?
            .0;
    let slack: Vec<f64> = curvature
        .gradient
        .iter()
        .zip(&curvature.band)
        .map(|(score, band)| score.abs() + band)
        .collect();
    let mode_box = posterior.mapv(f64::abs).dot(&Array1::from(slack.clone()));
    let vech: Vec<(usize, usize)> =
        (0..g).flat_map(|high| (0..=high).map(move |low| (high, low))).collect();
    let no_direction = vec![0.0; dimension];
    let mut mode_channel = 0.0;
    for k in 0..dimension {
        let mut direction = no_direction.clone();
        direction[k] = 1.0;
        let derivative = hessian_direction(&patterns, mean, covariance, &direction, &no_direction)?.0;
        let information_trace = (&posterior * &derivative).sum();
        let covariance_trace = match &covariance_inverse {
            Some(inverse) if k >= g * p => {
                let (high, low) = vech[k - g * p];
                let multiplicity = if high == low { 1.0 } else { 2.0 };
                0.5 * covariance_weight as f64 * multiplicity * inverse[[high, low]]
            }
            _ => 0.0,
        };
        mode_channel +=
            (slack[k] + 0.5 * information_trace.abs() + covariance_trace.abs()) * mode_box[k];
    }
    let error = evidence.rounding() + mode_channel;
    if !evidence.value.is_finite() || !error.is_finite() {
        return Err(numerical("genetic score law evidence is not finite"));
    }
    Ok((evidence.value, error))
}

/// The context decision between an optimum's evidence and the limit's across their rounding band.
/// With `D = evidence - limit_evidence` and `B` both evidences' errors plus the subtraction's own
/// rounding, `D > B` is `Optimum`, `D < -B` is `LimitDominates`, and otherwise the evidences tie.
fn classify_evidence(
    log_strength: f64,
    evidence: f64,
    evidence_error: f64,
    limit_evidence: f64,
    limit_error: f64,
) -> ContextDecision {
    let difference = evidence - limit_evidence;
    let bound = evidence_error + limit_error + f64::EPSILON * difference.abs();
    if difference > bound {
        ContextDecision::Optimum {
            log_strength,
            evidence,
            limit_evidence,
        }
    } else if difference < -bound {
        ContextDecision::LimitDominates {
            log_strength,
            evidence,
            limit_evidence,
        }
    } else {
        ContextDecision::EvidenceTie { difference, bound }
    }
}

/// The family's exact boundary score with its running-error bound, the fixed-nuisance score
/// test it extends, and the context information's scale.
struct BoundaryScore {
    value: f64,
    bound: f64,
    /// The part of `bound` charged to the certified limit mode's error.
    mode_channel: f64,
    score_test: f64,
    kappa: f64,
}

/// The context information's scale per unit of prior precision, typed. With observed scores
/// the context block `sum_o P_o ⊗ S_cc,o` is positive semidefinite against a positive-definite
/// prior metric, so a non-positive or non-finite trace is a defect of that information and a
/// refusal, never the no-context law. (With no observed score there is no law to fit, and the
/// fit refuses earlier.)
fn context_scale(kappa: f64) -> Result<f64, EventHistoryError> {
    if kappa > 0.0 && kappa.is_finite() {
        Ok(kappa)
    } else {
        Err(EventHistoryError::Fit {
            reason: format!(
                "the context block's information sum_o P_o ⊗ S_cc,o has no positive finite trace in the prior's metric (kappa = {kappa}) although scores are observed"
            ),
        })
    }
}

/// The normalized Gaussian prior on the context columns of `A` (SPEC 12/14):
/// `vec(A_ctx) ~ N(0, G^-1 ⊗ Sigma / lambda)`, `G` the centered Gram of the
/// non-intercept context columns under the equal-subject measure, one strength
/// across scores, the intercept unpenalized. It is exactly `P_c` pseudo-observations
/// of every score with `sum c c' = lambda G` and no score moments, so the EM step,
/// the certified Newton finish and the closed-form Hessian carry it unchanged.
struct ContextPrior {
    /// `G` with a zero intercept row and column.
    gram: Array2<f64>,
    /// `G`'s first-order rounding errors.
    gram_error: Array2<f64>,
    log_det_gram: f64,
    columns: usize,
}

impl ContextPrior {
    /// `None` for an intercept-only context.
    fn new(context: ArrayView2<'_, f64>, weights: &[f64]) -> Result<Option<Self>, EventHistoryError> {
        let p = context.ncols();
        if p <= 1 {
            return Ok(None);
        }
        let mut total = RunningSum::default();
        for &w in weights {
            total.add(w, 0.0);
        }
        if !(total.value > 0.0) {
            return Err(invalid(
                "a context prior needs a positive total subject weight",
            ));
        }
        // A quotient x / t is the product of x with recip(t), which charges |1/t| + mu_t / t^2, so
        // its error is e_x / |t| + |x / t| (e_t / |t| + 2 eps).
        let quotient_error = |x: f64, x_error: f64| -> f64 {
            x_error / total.value
                + (x / total.value).abs() * (total.error / total.value + 2.0 * f64::EPSILON)
        };
        let mut centre = vec![RunningSum::default(); p];
        for (row, &w) in context.rows().into_iter().zip(weights) {
            for b in 1..p {
                let product = w * row[b];
                centre[b].add(product / total.value, quotient_error(product, f64::EPSILON * product.abs()));
            }
        }
        let mut gram = Array2::<f64>::zeros((p, p));
        let mut gram_error = Array2::<f64>::zeros((p, p));
        for (row, &w) in context.rows().into_iter().zip(weights) {
            for a in 1..p {
                let deviation_a = row[a] - centre[a].value;
                let deviation_a_error = centre[a].error + f64::EPSILON * deviation_a.abs();
                let weighted = w * deviation_a;
                let weighted_error = w.abs() * deviation_a_error + f64::EPSILON * weighted.abs();
                for b in 1..p {
                    let deviation_b = row[b] - centre[b].value;
                    let deviation_b_error = centre[b].error + f64::EPSILON * deviation_b.abs();
                    let product = weighted * deviation_b;
                    let product_rounding =
                        product_error(weighted, weighted_error, deviation_b, deviation_b_error);
                    gram[[a, b]] += product / total.value;
                    gram_error[[a, b]] += quotient_error(product, product_rounding)
                        + f64::EPSILON * gram[[a, b]].abs();
                }
            }
        }
        let columns: Vec<usize> = (1..p).collect();
        let block = lower_factor(&submatrix(&gram, &columns, &columns), "the centered context Gram")?;
        Ok(Some(Self {
            gram,
            gram_error,
            log_det_gram: factor_log_determinant(&block),
            columns: p - 1,
        }))
    }

    /// The prior at strength `lambda` as a complete pseudo-pattern. `lambda` is charged one
    /// rounding of its own (the runtime libm's `exp`, cited below one ulp).
    fn pattern(&self, g: usize, lambda: f64) -> PatternStatistics {
        let p = self.gram.nrows();
        let context = &self.gram * lambda;
        let context_error = &self.gram_error * lambda.abs() + context.mapv(|v| 2.0 * f64::EPSILON * v.abs());
        PatternStatistics {
            present: (0..g).collect(),
            missing: Vec::new(),
            weight: self.columns as f64,
            weight_error: 0.0,
            context,
            context_error,
            score_context: Array2::zeros((g, p)),
            score_context_error: Array2::zeros((g, p)),
            score: Array2::zeros((g, g)),
            score_error: Array2::zeros((g, g)),
        }
    }

    /// The certified penalized mode at `rho` from a warm start, with its evidence
    /// and the evidence's derivative in `rho`.
    fn penalized(
        &self,
        patterns: &[PatternStatistics],
        rho: f64,
        mean: Array2<f64>,
        covariance: Array2<f64>,
    ) -> Result<(Array2<f64>, Array2<f64>, f64, f64), EventHistoryError> {
        let g = mean.nrows();
        let mut all = patterns.to_vec();
        all.push(self.pattern(g, rho.exp()));
        let (mean, covariance) = certified_fit(&all, mean, covariance)?;
        let (value, derivative) = laplace_evidence(patterns, Some((self, rho)), &mean, &covariance)?;
        Ok((mean, covariance, value, derivative))
    }

    /// The data patterns restricted to the intercept column.
    fn reduce(patterns: &[PatternStatistics]) -> Vec<PatternStatistics> {
        patterns
            .iter()
            .map(|stats| PatternStatistics {
                present: stats.present.clone(),
                missing: stats.missing.clone(),
                weight: stats.weight,
                weight_error: stats.weight_error,
                context: stats.context.select(Axis(0), &[0]).select(Axis(1), &[0]),
                context_error: stats.context_error.select(Axis(0), &[0]).select(Axis(1), &[0]),
                score_context: stats.score_context.select(Axis(1), &[0]),
                score_context_error: stats.score_context_error.select(Axis(1), &[0]),
                score: stats.score.clone(),
                score_error: stats.score_error.clone(),
            })
            .collect()
    }

    /// The no-context model's law: the intercept-only certified fit on the same score
    /// sample, embedded with zero context coefficients.
    fn null_law(
        reduced: &[PatternStatistics],
        g: usize,
        p: usize,
    ) -> Result<(Array2<f64>, Array2<f64>), EventHistoryError> {
        let (mean, covariance) = initial_law(reduced, g, 1)?;
        let (mean, covariance) = certified_fit(reduced, mean, covariance)?;
        let embedded =
            Array2::from_shape_fn((g, p), |(j, b)| if b == 0 { mean[[j, 0]] } else { 0.0 });
        Ok((embedded, covariance))
    }

    /// This family's no-context limit patterns, the intercept-only data patterns with the prior's
    /// `P_c` pseudo-observations of every score and no context moments, and their certified mode.
    fn limit_mode(
        &self,
        reduced: &[PatternStatistics],
        g: usize,
    ) -> Result<(Vec<PatternStatistics>, Array2<f64>, Array2<f64>), EventHistoryError> {
        let mut limit = reduced.to_vec();
        limit.push(PatternStatistics {
            present: (0..g).collect(),
            missing: Vec::new(),
            weight: self.columns as f64,
            weight_error: 0.0,
            context: Array2::zeros((1, 1)),
            context_error: Array2::zeros((1, 1)),
            score_context: Array2::zeros((g, 1)),
            score_context_error: Array2::zeros((g, 1)),
            score: Array2::zeros((g, g)),
            score_error: Array2::zeros((g, g)),
        });
        let (mean, covariance) = initial_law(reduced, g, 1)?;
        let (mean, covariance) = certified_fit(&limit, mean, covariance)?;
        Ok((limit, mean, covariance))
    }

    /// The exact boundary score of this evidence family, `S = d log Z / d tau` at
    /// `tau = 1 / lambda = 0`, in closed form at the limit mode, with its running-error bound.
    ///
    /// With `z = sqrt(lambda) x` on the context coefficients the family's objective is exactly
    /// `J = L(n) + sqrt(tau) g(n)'z - z'(K(Sigma) + tau I_cc) z / 2` (the likelihood is quadratic
    /// in the context coefficients at fixed nuisance `n`), and its Laplace evidence expands as
    ///
    /// ```text
    ///     S = T/2 + P_c tr(Sigma^-1 nu_Sigma)/2 + tr(V_n X)/2 + tr(V_n B'QB)/2,
    ///     T = s'Qs - tr(Q I_cc),   Q = K^-1 = Sigma ⊗ G^-1,   w = Q s,
    ///     nu = V_n f,   f = G_n'w - w' dK w / 2,
    ///     X = D(grad^2 L)[nu] + D_x H_nn[w] - w' d^2K w / 2,   B = G_n - dK w,
    /// ```
    ///
    /// `s` and `I_cc` the context score and information, `G_n` the context-nuisance cross
    /// Hessian of the likelihood, `V_n` the limit's nuisance posterior, all at the limit mode.
    /// `T/2` alone is the fixed-nuisance score test; the other terms are the nuisance's Laplace
    /// correction. On one complete score it is `n((n - 3) r^2 - 1)/2`.
    fn boundary_score(
        &self,
        patterns: &[PatternStatistics],
        reduced: &[PatternStatistics],
        g: usize,
        p: usize,
    ) -> Result<BoundaryScore, EventHistoryError> {
        let (limit, limit_mean, covariance) = self.limit_mode(reduced, g)?;
        self.boundary_score_at(patterns, &limit, &limit_mean, &covariance, p)
    }

    /// The boundary score at a given no-context limit mode: in production the certified one
    /// `boundary_score` computes.
    fn boundary_score_at(
        &self,
        patterns: &[PatternStatistics],
        limit: &[PatternStatistics],
        limit_mean: &Array2<f64>,
        covariance: &Array2<f64>,
        p: usize,
    ) -> Result<BoundaryScore, EventHistoryError> {
        let g = limit_mean.nrows();
        let columns = self.columns;
        let m = g * columns;
        let pairs = g * (g + 1) / 2;
        let limit_curvature = score_curvature(&limit, &limit_mean, &covariance)?;
        let limit_information = limit_curvature.hessian.mapv(|v| -v);
        let (nuisance_posterior, nuisance_posterior_error) = inverse_with_error(
            &limit_information,
            &limit_curvature.hessian_error,
            "the no-context limit's information",
        )?;
        let embedded =
            Array2::from_shape_fn((g, p), |(j, b)| if b == 0 { limit_mean[[j, 0]] } else { 0.0 });
        let full = score_curvature(patterns, &embedded, &covariance)?;
        let context_index: Vec<usize> =
            (0..g).flat_map(|j| (1..p).map(move |b| j * p + b)).collect();
        let nuisance_index: Vec<usize> =
            (0..g).map(|j| j * p).chain((0..pairs).map(|k| g * p + k)).collect();
        let information = submatrix(&full.hessian, &context_index, &context_index).mapv(|v| -v);
        let information_error = submatrix(&full.hessian_error, &context_index, &context_index);
        let score = Array2::from_shape_fn((m, 1), |(i, _)| full.gradient[context_index[i]]);
        let score_error = Array2::from_shape_fn((m, 1), |(i, _)| full.band[context_index[i]]);
        let cross = submatrix(&full.hessian, &context_index, &nuisance_index);
        let cross_error = submatrix(&full.hessian_error, &context_index, &nuisance_index);
        let context_columns: Vec<usize> = (1..p).collect();
        let gram = submatrix(&self.gram, &context_columns, &context_columns);
        let gram_error = submatrix(&self.gram_error, &context_columns, &context_columns);
        let (gram_inverse, gram_inverse_error) =
            inverse_with_error(&gram, &gram_error, "the centered context Gram")?;
        // Q = Sigma ⊗ G^-1, with the limit mode's Sigma an exact operand.
        let unit = Array2::from_shape_fn((m, m), |(r, c)| {
            covariance[[r / columns, c / columns]] * gram_inverse[[r % columns, c % columns]]
        });
        let unit_error = Array2::from_shape_fn((m, m), |(r, c)| {
            product_error(
                covariance[[r / columns, c / columns]],
                0.0,
                gram_inverse[[r % columns, c % columns]],
                gram_inverse_error[[r % columns, c % columns]],
            )
        });
        let (w, w_error) = product_with_error(&unit, &unit_error, &score, &score_error);
        let (quadratic, quadratic_error) = contraction_with_error(&score, &score_error, &w, &w_error);
        let (trace, trace_error) =
            contraction_with_error(&unit, &unit_error, &information, &information_error);
        let score_test = 0.5 * (quadratic - trace);
        let score_test_error =
            0.5 * (quadratic_error + trace_error + f64::EPSILON * (quadratic - trace).abs());
        let kappa = context_scale(trace / m as f64)?;

        let exact = |rows: usize, columns: usize| Array2::<f64>::zeros((rows, columns));
        let (precision, precision_error) =
            inverse_with_error(&covariance, &exact(g, g), "the no-context limit's covariance")?;
        // X_w = unvec(w) (scores by columns) and X_w G X_w'.
        let moment = Array2::from_shape_fn((g, columns), |(j, b)| w[[j * columns + b, 0]]);
        let moment_error = Array2::from_shape_fn((g, columns), |(j, b)| w_error[[j * columns + b, 0]]);
        let (moment_gram, moment_gram_error) =
            product_with_error(&moment, &moment_error, &gram, &gram_error);
        let (spread, spread_error) = product_with_error(
            &moment_gram,
            &moment_gram_error,
            &moment.t().to_owned(),
            &moment_error.t().to_owned(),
        );
        let vech: Vec<(usize, usize)> = (0..g).flat_map(|k| (0..=k).map(move |l| (k, l))).collect();
        let symmetric_unit = |(k, l): (usize, usize)| -> Array2<f64> {
            let mut e = Array2::<f64>::zeros((g, g));
            e[[k, l]] = 1.0;
            e[[l, k]] = 1.0;
            e
        };
        // dK / d sigma_k = -(P E_k P) ⊗ G, so -w' dK_k w = tr(P E_k P X_w G X_w').
        let d_precision: Vec<(Array2<f64>, Array2<f64>)> = vech
            .iter()
            .map(|&pair| {
                let (left, left_error) =
                    product_with_error(&precision, &precision_error, &symmetric_unit(pair), &exact(g, g));
                product_with_error(&left, &left_error, &precision, &precision_error)
            })
            .collect();
        let (mut force, mut force_error) = product_with_error(
            &cross.t().to_owned(),
            &cross_error.t().to_owned(),
            &w,
            &w_error,
        );
        for (k, (d, d_error)) in d_precision.iter().enumerate() {
            let (contracted, contracted_error) = contraction_with_error(d, d_error, &spread, &spread_error);
            force[[g + k, 0]] += 0.5 * contracted;
            force_error[[g + k, 0]] += 0.5 * contracted_error + f64::EPSILON * force[[g + k, 0]].abs();
        }
        let (movement, movement_error) =
            product_with_error(&nuisance_posterior, &nuisance_posterior_error, &force, &force_error);
        let vech_position = |k: usize, l: usize| -> usize {
            let (high, low) = if k >= l { (k, l) } else { (l, k) };
            g + high * (high + 1) / 2 + low
        };
        let movement_covariance =
            Array2::from_shape_fn((g, g), |(k, l)| movement[[vech_position(k, l), 0]]);
        let movement_covariance_error =
            Array2::from_shape_fn((g, g), |(k, l)| movement_error[[vech_position(k, l), 0]]);
        let (normalizer_trace, normalizer_trace_error) = contraction_with_error(
            &precision,
            &precision_error,
            &movement_covariance,
            &movement_covariance_error,
        );
        let normalizer = 0.5 * columns as f64 * normalizer_trace;
        let normalizer_error =
            0.5 * columns as f64 * normalizer_trace_error + f64::EPSILON * normalizer.abs();

        let mut context_direction = vec![0.0; full.gradient.len()];
        let mut context_direction_error = vec![0.0; full.gradient.len()];
        for (local, &index) in context_index.iter().enumerate() {
            context_direction[index] = w[[local, 0]];
            context_direction_error[index] = w_error[[local, 0]];
        }
        let (full_direction, full_direction_error) = hessian_direction(
            patterns,
            &embedded,
            &covariance,
            &context_direction,
            &context_direction_error,
        )?;
        let (limit_direction, limit_direction_error) = hessian_direction(
            &limit,
            &limit_mean,
            &covariance,
            &movement.column(0).to_vec(),
            &movement_error.column(0).to_vec(),
        )?;
        let (mut curvature_change, mut curvature_change_error) = sum_with_error(
            &limit_direction,
            &limit_direction_error,
            &submatrix(&full_direction, &nuisance_index, &nuisance_index),
            &submatrix(&full_direction_error, &nuisance_index, &nuisance_index),
        );
        for (a, (da, da_error)) in d_precision.iter().enumerate() {
            for (c, (dc, dc_error)) in d_precision.iter().enumerate() {
                // w' d^2K w for (sigma_a, sigma_c) = tr((P E_a P E_c P + P E_c P E_a P) X_w G X_w').
                let (left, left_error) =
                    product_with_error(da, da_error, &symmetric_unit(vech[c]), &exact(g, g));
                let (first, first_error) =
                    product_with_error(&left, &left_error, &precision, &precision_error);
                let (right, right_error) =
                    product_with_error(dc, dc_error, &symmetric_unit(vech[a]), &exact(g, g));
                let (second, second_error) =
                    product_with_error(&right, &right_error, &precision, &precision_error);
                let (both, both_error) = sum_with_error(&first, &first_error, &second, &second_error);
                let (contracted, contracted_error) =
                    contraction_with_error(&both, &both_error, &spread, &spread_error);
                let entry = [g + a, g + c];
                curvature_change[entry] -= 0.5 * contracted;
                curvature_change_error[entry] +=
                    0.5 * contracted_error + f64::EPSILON * curvature_change[entry].abs();
            }
        }
        let (curvature_trace, curvature_trace_error) = contraction_with_error(
            &nuisance_posterior,
            &nuisance_posterior_error,
            &curvature_change,
            &curvature_change_error,
        );
        let term_curvature = 0.5 * curvature_trace;
        let term_curvature_error = 0.5 * curvature_trace_error;
        // B = G_n - dK w: its Sigma columns add vec(P E_k P X_w G).
        let mut border = cross.clone();
        let mut border_error = cross_error.clone();
        for (k, (d, d_error)) in d_precision.iter().enumerate() {
            let (left, left_error) = product_with_error(d, d_error, &moment, &moment_error);
            let (column, column_error) = product_with_error(&left, &left_error, &gram, &gram_error);
            for r in 0..m {
                border[[r, g + k]] += column[[r / columns, r % columns]];
                border_error[[r, g + k]] += column_error[[r / columns, r % columns]]
                    + f64::EPSILON * border[[r, g + k]].abs();
            }
        }
        let (border_unit, border_unit_error) = product_with_error(
            &border.t().to_owned(),
            &border_error.t().to_owned(),
            &unit,
            &unit_error,
        );
        let (bordered, bordered_error) =
            product_with_error(&border_unit, &border_unit_error, &border, &border_error);
        let (border_trace, border_trace_error) = contraction_with_error(
            &nuisance_posterior,
            &nuisance_posterior_error,
            &bordered,
            &bordered_error,
        );
        let term_border = 0.5 * border_trace;
        let term_border_error = 0.5 * border_trace_error;
        let mut value = RunningSum::default();
        value.add(score_test, score_test_error);
        value.add(normalizer, normalizer_error);
        value.add(term_curvature, term_curvature_error);
        value.add(term_border, term_border_error);

        // The certified limit mode's error. Its computed score lies within its band, so the exact
        // score there lies within |s^| + band, and the mode moves by at most |V_n| (|s^| + band) to
        // first order; that reaches S through the context score s = G_n dtheta and T/2's w.
        let nuisance_error = nuisance_posterior.mapv(f64::abs).dot(&Array1::from_iter(
            limit_curvature
                .gradient
                .iter()
                .zip(&limit_curvature.band)
                .map(|(score, band)| score.abs() + band),
        ));
        let mode_channel = w
            .column(0)
            .mapv(f64::abs)
            .dot(&cross.mapv(f64::abs).dot(&nuisance_error));
        let bound = value.error + mode_channel;
        if !value.value.is_finite() || !bound.is_finite() {
            return Err(numerical("the genetic score law's boundary score is not finite"));
        }
        Ok(BoundaryScore {
            value: value.value,
            bound,
            mode_channel,
            score_test,
            kappa,
        })
    }

    /// This evidence family's no-context limit, `lambda -> infinity`. The context block's
    /// log-determinant cancels the prior normalizer, the prior's `P_c` pseudo-observations
    /// stay in the covariance block, and the value is
    /// `l(0, n) + (d_n / 2) log 2pi - log det(I_nn + N) / 2` at their joint mode, with the
    /// prior's normalizer remnant `P_c (g log 2pi + log det Sigma) / 2`; `(value, error)` through
    /// `evidence_with_error`.
    fn limit_evidence(
        &self,
        reduced: &[PatternStatistics],
        g: usize,
    ) -> Result<(f64, f64), EventHistoryError> {
        let (limit, mean, covariance) = self.limit_mode(reduced, g)?;
        evidence_with_error(&limit, None, self.columns, &mean, &covariance)
    }

    /// The context strength by Laplace evidence in `rho` through `opt`, compared with the
    /// no-context model as its own structure (this family's `lambda -> infinity` limit).
    fn learn(
        &self,
        patterns: &[PatternStatistics],
        g: usize,
        p: usize,
    ) -> Result<ScoreLawFit, EventHistoryError> {
        let reduced = Self::reduce(patterns);
        let (null_mean, null_covariance) = Self::null_law(&reduced, g, p)?;
        let (null_evidence, null_error) = self.limit_evidence(&reduced, g)?;
        // The family's exact boundary score decides whether its evidence rises from the
        // no-context limit. Below minus its running-error bound it does not, and the fit returns
        // the no-context law. Within its bound the evidence is flat to first order: the fit
        // returns the no-context law and reports the first-order tie.
        let boundary = self.boundary_score(patterns, &reduced, g, p)?;
        log::debug!(
            "genetic score law context boundary: family score {:.6e} ± {:.2e} (limit-mode channel {:.2e}), fixed-nuisance score test {:.6e}, kappa {:.4e}",
            boundary.value,
            boundary.bound,
            boundary.mode_channel,
            boundary.score_test,
            boundary.kappa
        );
        if boundary.value < -boundary.bound {
            return Ok(ScoreLawFit {
                law: GeneticScoreLaw::from_parts(&null_mean, &null_covariance)?,
                context: ContextDecision::NoContext {
                    score: boundary.value,
                    bound: boundary.bound,
                },
            });
        }
        if !(boundary.value > boundary.bound) {
            return Ok(ScoreLawFit {
                law: GeneticScoreLaw::from_parts(&null_mean, &null_covariance)?,
                context: ContextDecision::FirstOrderTie {
                    score: boundary.value,
                    bound: boundary.bound,
                },
            });
        }
        // S > 0: the evidence rises from the limit to an interior maximum. The isotropic Gaussian
        // evidence with this boundary score peaks at lambda = dim kappa^2 / (2 S), the start; that
        // is the one use of kappa's scale.
        let dimension = (g * self.columns) as f64;
        let start = (dimension * boundary.kappa * boundary.kappa / (2.0 * boundary.value)).ln();
        let mut warm = (null_mean.clone(), null_covariance.clone());
        let objective = opt::FusedObjective::new(
            |x: &Array1<f64>| -> Result<opt::FirstOrderSample, opt::ObjectiveEvalError> {
                let (mean, covariance, value, derivative) = self
                    .penalized(patterns, x[0], warm.0.clone(), warm.1.clone())
                    .map_err(|error| opt::ObjectiveEvalError::fatal(error.to_string()))?;
                warm = (mean, covariance);
                Ok(opt::FirstOrderSample {
                    value: -value,
                    gradient: Array1::from(vec![-derivative]),
                })
            },
        );
        let solution = opt::Bfgs::new(Array1::from(vec![start]), objective)
            .run()
            .map_err(|error| EventHistoryError::Fit {
                reason: format!(
                    "the genetic score law's context strength did not converge: {error}"
                ),
            })?;
        let log_strength = solution.final_point[0];
        let fitted = self.penalized(patterns, log_strength, warm.0, warm.1)?;
        // The optimum's evidence against the limit's, both along the route that carries their
        // first-order errors. Beyond the band either side decides; within it the evidences tie, and
        // the fit returns the no-context law and reports the tie.
        let (evidence, evidence_error) =
            evidence_with_error(patterns, Some((self, log_strength)), 0, &fitted.0, &fitted.1)?;
        let context =
            classify_evidence(log_strength, evidence, evidence_error, null_evidence, null_error);
        let law = if matches!(context, ContextDecision::Optimum { .. }) {
            GeneticScoreLaw::from_parts(&fitted.0, &fitted.1)?
        } else {
            GeneticScoreLaw::from_parts(&null_mean, &null_covariance)?
        };
        Ok(ScoreLawFit { law, context })
    }
}

/// The missing scores of one subject given its observed scores and context, over the scalar `S`
/// the law was conditioned at by `GeneticScoreLaw::conditional_over`.
#[derive(Clone, Debug)]
pub struct ConditionalScores<S = f64> {
    /// Unobserved score indices in increasing order: the first latent coordinates.
    pub missing: Vec<usize>,
    /// Conditional mean of the missing scores.
    pub mean: Vec<S>,
    /// Conditional precision `Lambda_mm`.
    pub precision: Array2<S>,
    /// Lower Cholesky factor of `Lambda_mm`.
    pub precision_factor: Array2<S>,
    /// `log det Lambda_mm`.
    pub log_determinant: S,
    /// `log p(g_o | c)`; zero when nothing is observed.
    pub observed_log_density: S,
    observed: Vec<Option<f64>>,
}

impl<S: JetField> ConditionalScores<S> {
    pub fn observed(&self) -> &[Option<f64>] {
        &self.observed
    }

    /// `log p(g_m | g_o, c) = (log det Lambda_mm - m log 2pi - d' Lambda_mm d) / 2` at the missing
    /// scores, `d` their deviation from the conditional mean, over the law's own scalar.
    pub fn log_density(&self, missing_scores: &[S]) -> Result<S, EventHistoryError> {
        let m = self.missing.len();
        if missing_scores.len() != m {
            return Err(invalid("missing score count does not match the conditional law"));
        }
        let deviation: Vec<S> = missing_scores
            .iter()
            .zip(&self.mean)
            .map(|(x, mu)| x.sub(mu))
            .collect();
        let exact = |value: f64| self.log_determinant.constant_like(value);
        let mut quadratic = exact(0.0);
        for a in 0..m {
            for b in 0..m {
                quadratic =
                    quadratic.add(&deviation[a].mul(&self.precision[[a, b]]).mul(&deviation[b]));
            }
        }
        Ok(self
            .log_determinant
            .sub(&ln(&exact(TAU)).scale(m as f64))
            .sub(&quadratic)
            .scale(0.5))
    }

    /// The full score vector with the missing coordinates taken from the latent path.
    pub fn genome<T: JetField>(
        &self,
        zero: &T,
        missing_scores: &[T],
    ) -> Result<Vec<T>, EventHistoryError> {
        if missing_scores.len() != self.missing.len() {
            return Err(invalid("missing score count does not match the conditional law"));
        }
        let mut latent = missing_scores.iter();
        self.observed
            .iter()
            .map(|value| match value {
                Some(value) => Ok(zero.constant_like(*value)),
                None => latent
                    .next()
                    .cloned()
                    .ok_or_else(|| invalid("missing score count does not match the conditional law")),
            })
            .collect()
    }
}

impl ConditionalScores<f64> {
    /// Writes the full score vector: observed scores, and missing scores
    /// transported from standard normal coordinates by `mean + F^-T z`, where
    /// `F F' = Lambda_mm`.
    pub fn draw_into(
        &self,
        standard_normal: &[f64],
        genome: &mut [f64],
    ) -> Result<(), EventHistoryError> {
        let m = self.missing.len();
        if standard_normal.len() != m
            || genome.len() != self.observed.len()
            || standard_normal.iter().any(|z| !z.is_finite())
        {
            return Err(invalid(
                "standard normal coordinates or score buffer do not match the conditional law",
            ));
        }
        let mut draw = standard_normal.to_vec();
        for i in (0..m).rev() {
            let tail: f64 = (i + 1..m)
                .map(|k| self.precision_factor[[k, i]] * draw[k])
                .sum();
            draw[i] = (draw[i] - tail) / self.precision_factor[[i, i]];
        }
        for (slot, value) in genome.iter_mut().zip(&self.observed) {
            if let Some(value) = value {
                *slot = *value;
            }
        }
        for ((&j, mu), x) in self.missing.iter().zip(&self.mean).zip(&draw) {
            genome[j] = mu + x;
        }
        Ok(())
    }
}

/// One function of a drive block with its contiguous coefficients.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct DriveFunction {
    pub axis: usize,
    /// `0` for `u_k`, `g + 1` for `B_kg`.
    pub interaction: usize,
    pub coefficients: Range<usize>,
}

fn basis_value<S: JetField>(theta: &[S], begin: usize, columns: &[f64]) -> S {
    let mut value = theta[begin].constant_like(0.0);
    for (b, &feature) in columns.iter().enumerate() {
        value = value.add(&theta[begin + b].scale(feature));
    }
    value
}

/// One drive-shaped coefficient block of the state law. Each signature axis has a
/// function `u_k` and one function `B_kg` per score its structure keeps, all over
/// the same supplied basis row (time, and context where the basis carries it). An
/// omitted `B_kg` is the explicit zero-effect model: no coefficients and no
/// contribution. Coefficients are axis-major, then `u` before the kept scores in
/// order, so each function is contiguous and its function prior attaches to one
/// range.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct DriveBlock {
    start: usize,
    columns: usize,
    scores: usize,
    /// `begins[k][i]`: the first coefficient of interaction `i` on axis `k`, when kept.
    begins: Vec<Vec<Option<usize>>>,
    width: usize,
}

impl DriveBlock {
    /// `kept[k][g]` keeps `B_kg`; every `u_k` is kept.
    pub fn new(
        start: usize,
        columns: usize,
        scores: usize,
        kept: &[Vec<bool>],
    ) -> Result<Self, EventHistoryError> {
        if columns == 0 {
            return Err(invalid("a drive block needs at least one basis column"));
        }
        if kept.iter().any(|row| row.len() != scores) {
            return Err(invalid(
                "drive structure must name every score on every signature",
            ));
        }
        let mut cursor = start;
        let mut begins = Vec::with_capacity(kept.len());
        for row in kept {
            let mut axis = Vec::with_capacity(scores + 1);
            for interaction in 0..=scores {
                if interaction == 0 || row[interaction - 1] {
                    axis.push(Some(cursor));
                    cursor = cursor
                        .checked_add(columns)
                        .ok_or_else(|| invalid("drive block dimension overflow"))?;
                } else {
                    axis.push(None);
                }
            }
            begins.push(axis);
        }
        Ok(Self {
            start,
            columns,
            scores,
            begins,
            width: cursor - start,
        })
    }

    pub fn start(&self) -> usize {
        self.start
    }

    pub fn width(&self) -> usize {
        self.width
    }

    /// Every kept function with its coefficients.
    pub fn functions(&self) -> Vec<DriveFunction> {
        let columns = self.columns;
        self.begins
            .iter()
            .enumerate()
            .flat_map(|(axis, row)| {
                row.iter()
                    .enumerate()
                    .filter_map(move |(interaction, begin)| {
                        begin.map(|begin| DriveFunction {
                            axis,
                            interaction,
                            coefficients: begin..begin + columns,
                        })
                    })
            })
            .collect()
    }

    fn check(
        &self,
        coefficients: usize,
        columns: &[f64],
        scores: usize,
        axis: usize,
    ) -> Result<(), EventHistoryError> {
        if coefficients < self.start + self.width
            || columns.len() != self.columns
            || scores != self.scores
            || axis >= self.begins.len()
            || columns.iter().any(|v| !v.is_finite())
        {
            return Err(invalid(
                "drive basis row, scores or coefficients do not match the drive block",
            ));
        }
        Ok(())
    }

    /// `m_k = u_k + sum_g B_kg g_g` at one basis row, over the kept `B_kg`.
    pub fn mean<S: JetField>(
        &self,
        theta: &[S],
        columns: &[f64],
        genes: &[S],
        axis: usize,
    ) -> Result<S, EventHistoryError> {
        self.check(theta.len(), columns, genes.len(), axis)?;
        let mut value = theta[self.start].constant_like(0.0);
        for (interaction, begin) in self.begins[axis].iter().enumerate() {
            let Some(begin) = begin else {
                continue;
            };
            let function = basis_value(theta, *begin, columns);
            value = value.add(&if interaction == 0 {
                function
            } else {
                function.mul(&genes[interaction - 1])
            });
        }
        Ok(value)
    }

    /// The drive as an affine function of the missing scores: the offset from `u`
    /// and the observed scores, and one slope per missing score (zero for an
    /// omitted `B_kg`).
    pub fn slopes<S: JetField>(
        &self,
        theta: &[S],
        columns: &[f64],
        axis: usize,
        scores: &ConditionalScores,
    ) -> Result<(S, Vec<S>), EventHistoryError> {
        self.check(theta.len(), columns, scores.observed.len(), axis)?;
        let zero = theta[self.start].constant_like(0.0);
        let function = |interaction: usize| -> S {
            match self.begins[axis][interaction] {
                Some(begin) => basis_value(theta, begin, columns),
                None => zero.clone(),
            }
        };
        let mut offset = function(0);
        for (s, value) in scores.observed.iter().enumerate() {
            if let Some(value) = value {
                offset = offset.add(&function(s + 1).scale(*value));
            }
        }
        let slopes = scores.missing.iter().map(|&s| function(s + 1)).collect();
        Ok((offset, slopes))
    }

    /// Adds `adjoint * d m_k / d theta` at one basis row into `theta_adjoint`.
    pub fn pullback<S: JetField>(
        &self,
        columns: &[f64],
        genes: &[S],
        axis: usize,
        adjoint: &S,
        theta_adjoint: &mut [S],
    ) -> Result<(), EventHistoryError> {
        self.check(theta_adjoint.len(), columns, genes.len(), axis)?;
        for (interaction, begin) in self.begins[axis].iter().enumerate() {
            let Some(begin) = begin else {
                continue;
            };
            let weight = if interaction == 0 {
                adjoint.clone()
            } else {
                adjoint.mul(&genes[interaction - 1])
            };
            for (b, &feature) in columns.iter().enumerate() {
                theta_adjoint[begin + b] = theta_adjoint[begin + b].add(&weight.scale(feature));
            }
        }
        Ok(())
    }

    /// The transpose of `slopes` at one basis row: adds
    /// `offset_force * d offset / d theta + sum_a slope_forces[a] * d slopes[a] / d theta`
    /// into `theta_adjoint`.
    pub fn slopes_pullback(
        &self,
        columns: &[f64],
        axis: usize,
        scores: &ConditionalScores,
        offset_force: f64,
        slope_forces: &[f64],
        theta_adjoint: &mut [f64],
    ) -> Result<(), EventHistoryError> {
        self.check(theta_adjoint.len(), columns, scores.observed.len(), axis)?;
        if slope_forces.len() != scores.missing.len()
            || !offset_force.is_finite()
            || slope_forces.iter().any(|v| !v.is_finite())
        {
            return Err(invalid(
                "offset and slope forces must be finite, one slope force per missing score",
            ));
        }
        let mut weights = vec![0.0; self.scores + 1];
        weights[0] = offset_force;
        for (s, value) in scores.observed.iter().enumerate() {
            if let Some(value) = value {
                weights[s + 1] = offset_force * value;
            }
        }
        for (&s, force) in scores.missing.iter().zip(slope_forces) {
            weights[s + 1] = *force;
        }
        for (interaction, begin) in self.begins[axis].iter().enumerate() {
            let Some(begin) = begin else {
                continue;
            };
            for (b, &feature) in columns.iter().enumerate() {
                theta_adjoint[begin + b] += weights[interaction] * feature;
            }
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::f64::consts::{PI, SQRT_2};

    fn law() -> GeneticScoreLaw {
        GeneticScoreLaw {
            scores: 3,
            context_columns: 2,
            mean_coefficients: vec![0.3, -0.5, 1.1, 0.2, -0.4, 0.9],
            covariance_factor: vec![1.2, 0.0, 0.0, 0.5, 0.9, 0.0, -0.3, 0.4, 0.7],
        }
    }

    fn covariance_of(law: &GeneticScoreLaw) -> [[f64; 3]; 3] {
        let l = &law.covariance_factor;
        let mut sigma = [[0.0; 3]; 3];
        for i in 0..3 {
            for j in 0..3 {
                for k in 0..3 {
                    sigma[i][j] += l[i * 3 + k] * l[j * 3 + k];
                }
            }
        }
        sigma
    }

    fn mean_of(law: &GeneticScoreLaw, context: [f64; 2]) -> [f64; 3] {
        let a = &law.mean_coefficients;
        [0, 1, 2].map(|j| a[2 * j] * context[0] + a[2 * j + 1] * context[1])
    }

    /// Inverse and determinant of a 3x3 matrix by cofactors, independent of the
    /// triangular route the law takes.
    fn cofactor_inverse(s: [[f64; 3]; 3]) -> ([[f64; 3]; 3], f64) {
        let cofactor = |i: usize, j: usize| {
            let (r0, r1) = ((i + 1) % 3, (i + 2) % 3);
            let (c0, c1) = ((j + 1) % 3, (j + 2) % 3);
            s[r0][c0] * s[r1][c1] - s[r0][c1] * s[r1][c0]
        };
        let det = (0..3).map(|j| s[0][j] * cofactor(0, j)).sum::<f64>();
        let mut inverse = [[0.0; 3]; 3];
        for i in 0..3 {
            for j in 0..3 {
                inverse[j][i] = cofactor(i, j) / det;
            }
        }
        (inverse, det)
    }

    /// First-order rounding bound of a floating-point route: eps times the count of summed
    /// products it forms, times the largest magnitude among them, times the conditioning
    /// of any solve on the route.
    fn rounding(terms: usize, magnitude: f64, condition: f64) -> f64 {
        f64::EPSILON * terms as f64 * magnitude * condition
    }

    /// The deepest route on a 3x3 fixture nests at most three 3-term sums.
    const ROUTE_TERMS: usize = 27;

    fn largest_abs(values: &[f64]) -> f64 {
        values.iter().fold(0.0_f64, |m, v| m.max(v.abs()))
    }

    /// `||M||_inf ||M^-1||_inf` of a symmetric positive-definite matrix after Jacobi
    /// equilibration `D^-1/2 M D^-1/2`. A Cholesky route's forward error is governed by the
    /// equilibrated condition (van der Sluis), so coordinates in different units do not
    /// inflate it.
    fn condition_of(matrix: &Array2<f64>) -> f64 {
        let n = matrix.nrows();
        let scale: Vec<f64> = (0..n).map(|i| matrix[[i, i]].sqrt()).collect();
        let equilibrated =
            Array2::from_shape_fn((n, n), |(i, j)| matrix[[i, j]] / (scale[i] * scale[j]));
        let factor = lower_factor(&equilibrated, "an equilibrated information").unwrap();
        let inverse = cholesky_solve_matrix(&factor, &Array2::<f64>::eye(n));
        infinity_norm(&equilibrated) * infinity_norm(&inverse)
    }

    fn infinity_norm(matrix: &Array2<f64>) -> f64 {
        matrix
            .rows()
            .into_iter()
            .map(|row| row.iter().map(|v| v.abs()).sum::<f64>())
            .fold(0.0_f64, f64::max)
    }

    /// The fixture law's covariance conditioning and the largest operand magnitude of its
    /// 3x3 routes (covariance entries times precision entries).
    fn law_scale(law: &GeneticScoreLaw) -> (f64, f64) {
        let sigma = covariance_of(law);
        let matrix = Array2::from_shape_fn((3, 3), |(i, j)| sigma[i][j]);
        let inverse = cofactor_inverse(sigma).0;
        (
            condition_of(&matrix),
            largest_abs(&sigma.concat()) * largest_abs(&inverse.concat()),
        )
    }

    /// The step balancing an O(h^2) central difference's truncation against its eps / h
    /// rounding, at a parameter's scale.
    fn central_step(scale: f64) -> f64 {
        f64::EPSILON.cbrt() * (1.0 + scale.abs())
    }

    /// A central difference with an error bound. With `d1` at step h and `d2` at 2h, the
    /// O(h^2) rule gives `d1 - f' = T + R1` and `d2 - f' = 4T + R2`, so
    /// `|d1 - f'| <= |d2 - d1| / 3 + (4 |R1| + |R2|) / 3` with `|R1| <= r / h` and
    /// `|R2| <= r / (2h)` for a per-evaluation rounding `r`: `|d2 - d1| / 3 + 1.5 r / h`.
    fn central_difference(f: impl Fn(f64) -> f64, h: f64, rounding_of_value: f64) -> (f64, f64) {
        let narrow = (f(h) - f(-h)) / (2.0 * h);
        let wide = (f(2.0 * h) - f(-2.0 * h)) / (4.0 * h);
        (narrow, (wide - narrow).abs() / 3.0 + 1.5 * rounding_of_value / h)
    }

    /// The rounding account of one observed-data log-likelihood evaluation: its absolute
    /// summands (the weighted log terms and the trace of P against the residual moment's
    /// absolute bound), over (p + g)^2 products per pattern, at the worst observed block's
    /// conditioning.
    fn likelihood_account(
        patterns: &[PatternStatistics],
        mean: &Array2<f64>,
        covariance: &Array2<f64>,
    ) -> f64 {
        let (g, p) = mean.dim();
        let mut magnitude = 0.0;
        let mut condition = 1.0_f64;
        for stats in patterns {
            let o = &stats.present;
            let block = submatrix(covariance, o, o);
            condition = condition.max(condition_of(&block));
            let factor = lower_factor(&block, "an observed block").unwrap();
            let inverse_abs =
                cholesky_solve_matrix(&factor, &Array2::<f64>::eye(o.len())).mapv(f64::abs);
            let mean_abs = mean.select(Axis(0), o).mapv(f64::abs);
            let moment = stats.score.mapv(f64::abs)
                + mean_abs.dot(&stats.score_context.t().mapv(f64::abs)) * 2.0
                + mean_abs
                    .dot(&stats.context.mapv(f64::abs))
                    .dot(&mean_abs.t());
            magnitude += 0.5
                * (stats.weight * (o.len() as f64 * TAU.ln() + factor_log_determinant(&factor).abs())
                    + (&inverse_abs * &moment).sum());
        }
        rounding((p + g) * (p + g) * patterns.len(), magnitude, condition)
    }

    #[test]
    fn conditioning_on_observed_scores_is_the_dense_gaussian_conditional() {
        let law = law();
        let context = [1.0, 0.6];
        let mu = mean_of(&law, context);
        let sigma = covariance_of(&law);
        let (sigma_inverse, sigma_det) = cofactor_inverse(sigma);
        let (condition, magnitude) = law_scale(&law);
        let operands = largest_abs(&[mu[0], mu[1], mu[2], 0.8, 0.1, -0.7, -0.2, 1.4]);
        let bar = rounding(ROUTE_TERMS, magnitude * (1.0 + operands).powi(2), condition);

        let g0 = 0.8;
        let out = law.conditional(&context, &[Some(g0), None, None]).unwrap();
        assert_eq!(out.missing, vec![1, 2]);
        let covariance = |a: usize, b: usize| sigma[a][b] - sigma[a][0] * sigma[0][b] / sigma[0][0];
        for (a, i) in [1usize, 2].into_iter().enumerate() {
            let expected = mu[i] + sigma[i][0] * (g0 - mu[0]) / sigma[0][0];
            assert!(
                (out.mean[a] - expected).abs() <= bar,
                "conditional mean {a}: {} vs {expected}",
                out.mean[a]
            );
            for b in 0..2 {
                let product: f64 = (0..2)
                    .map(|k| out.precision[[a, k]] * covariance([1, 2][k], [1, 2][b]))
                    .sum();
                assert!(
                    (product - f64::from(a == b)).abs() <= bar,
                    "precision times covariance-route conditional covariance at ({a},{b}): {product}"
                );
            }
        }
        let point = [g0, 0.1, -0.7];
        let deviation = [0, 1, 2].map(|j| point[j] - mu[j]);
        let quadratic: f64 = (0..3)
            .map(|i| (0..3).map(|j| deviation[i] * sigma_inverse[i][j] * deviation[j]).sum::<f64>())
            .sum();
        let joint = -0.5 * (3.0 * TAU.ln() + sigma_det.ln() + quadratic);
        let factored = out.observed_log_density + out.log_density(&point[1..]).unwrap();
        assert!(
            (factored - joint).abs() <= bar,
            "p(g_o) p(g_m | g_o) = {factored}, joint density {joint}"
        );

        let (g1, g2) = (-0.2, 1.4);
        let out = law.conditional(&context, &[None, Some(g1), Some(g2)]).unwrap();
        assert_eq!(out.missing, vec![0]);
        let det = sigma[1][1] * sigma[2][2] - sigma[1][2] * sigma[2][1];
        let inverse = [
            [sigma[2][2] / det, -sigma[1][2] / det],
            [-sigma[2][1] / det, sigma[1][1] / det],
        ];
        let (r1, r2) = (g1 - mu[1], g2 - mu[2]);
        let weights = [
            sigma[0][1] * inverse[0][0] + sigma[0][2] * inverse[1][0],
            sigma[0][1] * inverse[0][1] + sigma[0][2] * inverse[1][1],
        ];
        let expected_mean = mu[0] + weights[0] * r1 + weights[1] * r2;
        let expected_variance = sigma[0][0] - weights[0] * sigma[1][0] - weights[1] * sigma[2][0];
        assert!((out.mean[0] - expected_mean).abs() <= bar);
        assert!((out.precision[[0, 0]] * expected_variance - 1.0).abs() <= bar);
        assert!((out.precision[[0, 0]] - sigma_inverse[0][0]).abs() <= bar);
    }

    #[test]
    fn masked_scores_integrate_a_drive_functional_to_its_exact_gaussian_value() {
        let law = law();
        let context = [1.0, -0.4];
        let mu = mean_of(&law, context);
        let sigma = covariance_of(&law);
        let loadings = [0.7, -0.5, 0.9];
        let g0 = 0.35;

        // E[exp(a'g) | g_0, c] = exp(a'E[g | g_0, c] + a_m' C a_m / 2), from the
        // covariance-route conditional mean and covariance.
        let conditional_mean = [1, 2].map(|i| mu[i] + sigma[i][0] * (g0 - mu[0]) / sigma[0][0]);
        let covariance = |a: usize, b: usize| sigma[a][b] - sigma[a][0] * sigma[0][b] / sigma[0][0];
        let linear =
            loadings[0] * g0 + loadings[1] * conditional_mean[0] + loadings[2] * conditional_mean[1];
        let variance: f64 = [1usize, 2]
            .into_iter()
            .map(|a| [1usize, 2].into_iter().map(|b| loadings[a] * covariance(a, b) * loadings[b]).sum::<f64>())
            .sum();
        let exact = (linear + 0.5 * variance).exp();

        let masked = law.conditional(&context, &[Some(g0), None, None]).unwrap();
        // A tensor Hermite rule's truncation on this entire integrand shrinks with order, so
        // the 40-node rule's error is at most its difference from the 20-node rule; the sum
        // of order^2 positive terms, each formed on a 3x3 route, adds its rounding.
        let (condition, magnitude) = law_scale(&law);
        let integrate = |order: usize| -> f64 {
            let rule = gam_math::quadrature::gauss_hermite_rule(order).unwrap();
            let mut total = 0.0;
            let mut nodes = [0.0; 3];
            for (&x0, &w0) in rule.nodes.iter().zip(&rule.weights) {
                for (&x1, &w1) in rule.nodes.iter().zip(&rule.weights) {
                    masked
                        .draw_into(&[SQRT_2 * x0, SQRT_2 * x1], &mut nodes)
                        .unwrap();
                    let exponent: f64 = nodes.iter().zip(&loadings).map(|(g, a)| g * a).sum();
                    total += w0 * w1 / PI * exponent.exp();
                }
            }
            total
        };
        let integral = integrate(40);
        let refinement = (integral - integrate(20)).abs() / exact;
        let quadrature_rounding = rounding(40 * 40 * ROUTE_TERMS, magnitude, condition);
        assert!(
            (integral / exact - 1.0).abs() <= refinement + quadrature_rounding,
            "integrated over the masked scores {integral}, exact {exact}, refinement {refinement:.3e}"
        );
        let mut genome = [0.0; 3];
        masked.draw_into(&[0.3, -1.1], &mut genome).unwrap();
        assert_eq!(genome[0], g0);

        // Control: a point imputation at the conditional mean misses the variance
        // factor, so this fixture separates integration from imputation.
        masked.draw_into(&[0.0, 0.0], &mut genome).unwrap();
        let exponent: f64 = genome.iter().zip(&loadings).map(|(g, a)| g * a).sum();
        let imputed = exponent.exp();
        let identity_bar = rounding(ROUTE_TERMS, magnitude, condition);
        assert!((imputed / exact - (-0.5 * variance).exp()).abs() <= identity_bar);
        // This law's conditional covariance at g_0 is ((0.81, 0.36), (0.36, 0.65)), so the
        // loadings (-0.5, 0.9) carry the variance 0.2025 - 0.324 + 0.5265 = 0.405 that the
        // imputation misses.
        assert!(
            (variance - 0.405).abs() <= identity_bar,
            "masked-score variance {variance}"
        );

        let observed = law
            .conditional(&context, &[Some(g0), Some(0.2), Some(-1.0)])
            .unwrap();
        assert!(observed.missing.is_empty());
        observed.draw_into(&[], &mut genome).unwrap();
        assert_eq!(genome, [g0, 0.2, -1.0]);
        assert_eq!(observed.log_density(&[]).unwrap(), 0.0);
    }

    fn weighted_least_squares(design: &[Vec<f64>], response: &[f64], weights: &[f64]) -> (Vec<f64>, f64) {
        let p = design[0].len();
        let mut system = vec![vec![0.0; p + 1]; p];
        for ((row, y), w) in design.iter().zip(response).zip(weights) {
            for a in 0..p {
                for b in 0..p {
                    system[a][b] += w * row[a] * row[b];
                }
                system[a][p] += w * row[a] * y;
            }
        }
        for column in 0..p {
            let pivot = (column..p)
                .max_by(|&i, &j| system[i][column].abs().total_cmp(&system[j][column].abs()))
                .unwrap();
            system.swap(column, pivot);
            for r in 0..p {
                if r != column {
                    let ratio = system[r][column] / system[column][column];
                    for c in column..=p {
                        let value = system[column][c];
                        system[r][c] -= ratio * value;
                    }
                }
            }
        }
        let beta: Vec<f64> = (0..p).map(|a| system[a][p] / system[a][a]).collect();
        let total: f64 = weights.iter().sum();
        let residual: f64 = design
            .iter()
            .zip(response)
            .zip(weights)
            .map(|((row, y), w)| {
                let fitted: f64 = row.iter().zip(&beta).map(|(x, b)| x * b).sum();
                w * (y - fitted).powi(2)
            })
            .sum();
        (beta, residual / total)
    }

    #[test]
    fn a_fit_with_scores_missing_at_random_is_the_monotone_closed_form_maximum() {
        let n = 240;
        let total = n + 4;
        let mut context = Array2::<f64>::zeros((total, 2));
        let mut scores = Vec::with_capacity(total);
        let mut weights = Vec::with_capacity(total);
        let mut complete = (Vec::new(), Vec::new(), Vec::new());
        let mut first = (Vec::new(), Vec::new(), Vec::new());
        for i in 0..n {
            let x = 2.0 * (i as f64 + 0.5) / n as f64 - 1.0;
            let e1 = (i as f64 * 0.754_877_666_246_692_7).fract() - 0.5;
            let e2 = (i as f64 * 0.569_840_290_998_053_2).fract() - 0.5;
            let g1 = 0.4 + 0.8 * x + e1;
            let g2 = -0.3 + 0.5 * x + 0.6 * e1 + e2;
            let w = 1.0 + 0.5 * (i % 3) as f64;
            context[[i, 0]] = 1.0;
            context[[i, 1]] = x;
            // Score 2 is absent whenever score 1 is high: missing at random given
            // the observed score, not completely at random.
            let g2_observed = (g1 <= 0.6).then_some(g2);
            scores.push(vec![Some(g1), g2_observed]);
            weights.push(w);
            first.0.push(vec![1.0, x]);
            first.1.push(g1);
            first.2.push(w);
            if g2_observed.is_some() {
                complete.0.push(vec![1.0, x, g1]);
                complete.1.push(g2);
                complete.2.push(w);
            }
        }
        for k in 0..4 {
            context[[n + k, 0]] = 1.0;
            context[[n + k, 1]] = 0.5 * k as f64;
            scores.push(vec![None, None]);
            weights.push(2.0);
        }

        // Anderson's factorisation of a monotone pattern is the exact maximum:
        // score 1 on the context over every row observing it, then score 2 on
        // (context, score 1) over the complete rows.
        let (a1, s11) = weighted_least_squares(&first.0, &first.1, &first.2);
        let (gamma, tau) = weighted_least_squares(&complete.0, &complete.1, &complete.2);
        let beta = gamma[2];
        let a2 = [gamma[0] + beta * a1[0], gamma[1] + beta * a1[1]];
        let oracle_mean = [a1[0], a1[1], a2[0], a2[1]];
        let oracle_covariance = [s11, beta * s11, tau + beta * beta * s11];

        let fit = maximum_likelihood(&scores, context.view(), &weights);
        let l = &fit.covariance_factor;
        let fitted_covariance = [l[0] * l[0], l[2] * l[0], l[2] * l[2] + l[3] * l[3]];

        // The fit is certified: every score entry lies within its roundoff bound, so each
        // parameter lies within |V| band of the exact maximum, V the inverse observed
        // information. The oracle's two weighted normal-equation solves round by n row sums
        // of 9 products at their compounded conditioning, and the covariance's rebuild from
        // its factor by 2 products at Sigma's scale.
        let (g, patterns) = pattern_statistics(&scores, context.view(), &weights).unwrap();
        let (start_mean, start_covariance) = initial_law(&patterns, g, 2).unwrap();
        let (certified_mean, certified_covariance) =
            certified_fit(&patterns, start_mean, start_covariance).unwrap();
        let certified = score_curvature(&patterns, &certified_mean, &certified_covariance).unwrap();
        let information = certified.hessian.mapv(|v| -v);
        let posterior = cholesky_solve_matrix(
            &lower_factor(&information, "the monotone fit's information").unwrap(),
            &Array2::<f64>::eye(information.nrows()),
        )
        .mapv(f64::abs);
        let parameter_bound = posterior.dot(&Array1::from(certified.band.clone()));
        let normal = |design: &[Vec<f64>], row_weights: &[f64]| {
            let width = design[0].len();
            Array2::from_shape_fn((width, width), |(a, b)| {
                design
                    .iter()
                    .zip(row_weights)
                    .map(|(row, w)| w * row[a] * row[b])
                    .sum::<f64>()
            })
        };
        let normal_rounding = rounding(
            n * 9,
            1.0 + largest_abs(&oracle_mean) + largest_abs(&oracle_covariance),
            condition_of(&normal(&first.0, &first.2)) * condition_of(&normal(&complete.0, &complete.2)),
        );
        let rebuild_rounding = rounding(2, largest_abs(&oracle_covariance), 1.0);
        for (index, (fitted, oracle)) in fit.mean_coefficients.iter().zip(&oracle_mean).enumerate() {
            let bar = parameter_bound[index] + normal_rounding;
            assert!(
                (fitted - oracle).abs() <= bar,
                "mean coefficient {index}: {fitted} vs monotone maximum {oracle}, bound {bar:.3e}"
            );
        }
        for ((fitted, oracle), &(k, l)) in fitted_covariance
            .iter()
            .zip(&oracle_covariance)
            .zip(&[(0usize, 0usize), (1, 0), (1, 1)])
        {
            let bar = parameter_bound[covariance_index(g, 2, k, l)] + normal_rounding + rebuild_rounding;
            assert!(
                (fitted - oracle).abs() <= bar,
                "covariance ({k},{l}): {fitted} vs monotone maximum {oracle}, bound {bar:.3e}"
            );
        }
        let largest_bar =
            largest_abs(parameter_bound.as_slice().unwrap()) + normal_rounding + rebuild_rounding;

        // Control: the complete-case regression of score 2 on the context is the
        // wrong answer under this masking, by far more than the bar.
        let complete_design: Vec<Vec<f64>> = complete.0.iter().map(|row| row[..2].to_vec()).collect();
        let complete_case = weighted_least_squares(&complete_design, &complete.1, &complete.2).0;
        let miss = (complete_case[0] - a2[0]).abs().max((complete_case[1] - a2[1]).abs());
        assert!(
            miss > largest_bar,
            "complete-case mutant misses by only {miss}, bound {largest_bar:.3e}"
        );

        // The fitted law conditions a masked subject on its observed score.
        let masked = fit.conditional(&[1.0, 0.2], &[Some(0.9), None]).unwrap();
        let expected = a2[0] + 0.2 * a2[1] + beta * (0.9 - a1[0] - 0.2 * a1[1]);
        // First-order propagation of the parameter bounds through
        // mu = A_2 c + (Sigma_21 / Sigma_11)(0.9 - A_1 c) with c = (1, 0.2), and through
        // tau = Sigma_22 - Sigma_21^2 / Sigma_11.
        let bounds = parameter_bound.as_slice().unwrap();
        let a_bound = largest_abs(&bounds[..4]) + normal_rounding;
        let s_bound = largest_abs(&bounds[4..]) + normal_rounding + rebuild_rounding;
        let residual = (0.9 - a1[0] - 0.2 * a1[1]).abs();
        let mean_bar = 1.2 * (1.0 + beta.abs()) * a_bound
            + residual * (1.0 + beta.abs()) * s_bound / s11
            + rounding(ROUTE_TERMS, 1.0 + expected.abs(), 1.0);
        assert!(
            (masked.mean[0] - expected).abs() <= mean_bar,
            "masked conditional mean {} vs {expected}, bound {mean_bar:.3e}",
            masked.mean[0]
        );
        let precision_bar = (1.0 + beta.abs()).powi(2) * s_bound / tau + rounding(ROUTE_TERMS, 1.0, 1.0);
        assert!(
            (masked.precision[[0, 0]] * tau - 1.0).abs() <= precision_bar,
            "masked conditional precision times tau {}, bound {precision_bar:.3e}",
            masked.precision[[0, 0]] * tau
        );
    }

    fn patterned_sample(n: usize) -> (Vec<Vec<Option<f64>>>, Array2<f64>, Vec<f64>) {
        // Every pattern cycles through the whole context range, including
        // non-monotone ones and a row observing nothing.
        let patterns = [
            [true, true, true],
            [true, false, true],
            [false, true, true],
            [true, true, false],
            [false, false, true],
            [false, false, false],
        ];
        let mut context = Array2::<f64>::zeros((n, 2));
        let mut scores = Vec::with_capacity(n);
        let mut weights = Vec::with_capacity(n);
        for i in 0..n {
            let x = 2.0 * (i as f64 + 0.5) / n as f64 - 1.0;
            let e = [
                0.754_877_666_246_692_7,
                0.569_840_290_998_053_2,
                0.362_646_470_928_318_3,
            ]
            .map(|alpha| (i as f64 * alpha).fract() - 0.5);
            let g = [
                0.2 + 0.9 * x + e[0],
                -0.4 + 0.3 * x + 0.5 * e[0] + e[1],
                0.1 - 0.6 * x - 0.4 * e[1] + 0.8 * e[2],
            ];
            context[[i, 0]] = 1.0;
            context[[i, 1]] = x;
            let pattern = patterns[i % patterns.len()];
            scores.push((0..3).map(|j| pattern[j].then_some(g[j])).collect());
            weights.push(1.0 + 0.5 * (i % 4) as f64);
        }
        (scores, context, weights)
    }

    #[test]
    fn the_observed_data_likelihood_gradient_matches_central_differences_on_every_pattern() {
        let (scores, context, weights) = patterned_sample(36);
        let law = law();
        let analytic = law.log_likelihood(&scores, context.view(), &weights).unwrap();
        let value_at = |candidate: &GeneticScoreLaw| {
            candidate
                .log_likelihood(&scores, context.view(), &weights)
                .unwrap()
                .value
        };
        // Each value rounds by the likelihood's own account; the analytic entries round by
        // the score's bands (the A block directly, the factor block through dl/dL = 2 G L).
        let (g, patterns) = pattern_statistics(&scores, context.view(), &weights).unwrap();
        let p = law.context_columns;
        let mean = Array2::from_shape_fn((g, p), |(j, b)| law.mean_coefficients[j * p + b]);
        let factor = law.factor();
        let covariance = factor.dot(&factor.t());
        let value_rounding = likelihood_account(&patterns, &mean, &covariance);
        let score = score_curvature(&patterns, &mean, &covariance).unwrap();
        let mut largest_bar = 0.0_f64;
        for index in 0..law.mean_coefficients.len() {
            let (central, bound) = central_difference(
                |step| {
                    let mut shifted = law.clone();
                    shifted.mean_coefficients[index] += step;
                    value_at(&shifted)
                },
                central_step(law.mean_coefficients[index]),
                value_rounding,
            );
            let bar = bound + score.band[index];
            largest_bar = largest_bar.max(bar);
            assert!(
                (central - analytic.mean_gradient[index]).abs() <= bar,
                "mean gradient {index}: analytic {} vs central {central}, bound {bar:.3e}",
                analytic.mean_gradient[index]
            );
        }
        let sigma_band =
            Array2::from_shape_fn((g, g), |(k, l)| score.band[covariance_index(g, p, k, l)]);
        let factor_band = sigma_band.dot(&factor.mapv(f64::abs)) * 2.0;
        for i in 0..3 {
            for j in 0..3 {
                let index = 3 * i + j;
                if j > i {
                    assert_eq!(analytic.factor_gradient[index], 0.0);
                    continue;
                }
                let (central, bound) = central_difference(
                    |step| {
                        let mut shifted = law.clone();
                        shifted.covariance_factor[index] += step;
                        value_at(&shifted)
                    },
                    central_step(law.covariance_factor[index]),
                    value_rounding,
                );
                let bar = bound + factor_band[[i, j]];
                largest_bar = largest_bar.max(bar);
                assert!(
                    (central - analytic.factor_gradient[index]).abs() <= bar,
                    "factor gradient ({i},{j}): analytic {} vs central {central}, bound {bar:.3e}",
                    analytic.factor_gradient[index]
                );
            }
        }
        // Control: this law is not the sample's maximum, so some gradient entry is resolved
        // beyond every entry's error bound.
        assert!(
            analytic
                .mean_gradient
                .iter()
                .chain(&analytic.factor_gradient)
                .any(|d| d.abs() > largest_bar),
            "no gradient entry exceeds the largest bound {largest_bar:.3e}"
        );
    }

    #[test]
    fn a_fit_over_non_monotone_patterns_is_a_stationary_point_of_the_observed_likelihood() {
        let (scores, context, weights) = patterned_sample(180);
        let (g, patterns) = pattern_statistics(&scores, context.view(), &weights).unwrap();
        let (mean, covariance) = initial_law(&patterns, g, 2).unwrap();
        let (mean, covariance) = certified_fit(&patterns, mean, covariance).unwrap();
        // At the returned point every score entry lies within its first-order roundoff bound.
        let at_fit = score_curvature(&patterns, &mean, &covariance).unwrap();
        assert!(
            at_fit.largest_ratio() <= 1.0,
            "score/roundoff ratio at the fit {}",
            at_fit.largest_ratio()
        );
        // Control: moving each mean coefficient by its own posterior standard deviation puts
        // the score far outside its bound and lowers the likelihood.
        let information = at_fit.hessian.mapv(|v| -v);
        let posterior = cholesky_solve_matrix(
            &lower_factor(&information, "the fit's information").unwrap(),
            &Array2::<f64>::eye(information.nrows()),
        );
        let displaced_mean = Array2::from_shape_fn((g, 2), |(j, b)| {
            mean[[j, b]] + posterior[[j * 2 + b, j * 2 + b]].sqrt()
        });
        let displaced = score_curvature(&patterns, &displaced_mean, &covariance).unwrap();
        assert!(
            displaced.largest_ratio() > 1.0,
            "displaced score/roundoff ratio {}",
            displaced.largest_ratio()
        );
        assert!(
            observed_log_likelihood(&patterns, &displaced_mean, &covariance).unwrap()
                < observed_log_likelihood(&patterns, &mean, &covariance).unwrap()
        );
    }

    #[test]
    fn the_corner_and_affine_border_integrate_masked_scores_into_the_exact_entry_state_law() {
        let law = law();
        let context = [1.0, 0.3];
        let mu = mean_of(&law, context);
        let sigma = covariance_of(&law);
        let g0 = 0.6;
        let scores = law.conditional(&context, &[Some(g0), None, None]).unwrap();
        let entry = DriveBlock::new(0, 2, 3, &[vec![true; 3]]).unwrap();
        assert_eq!(entry.width(), 8);
        // u, then B_0, B_1, B_2, each over two basis columns. At features (1, 0.4)
        // the missing scores' slopes are 1.0 and -0.48. This law's conditional
        // covariance at g_0 is ((0.81, 0.36), (0.36, 0.65)), so they carry the
        // spread 0.81 - 0.3456 + 0.14976 = 0.61416 into the entry state.
        let theta = vec![0.2, -0.1, 0.4, 0.2, 0.8, 0.5, -0.6, 0.3];
        let features = [1.0, 0.4];
        let (offset, slopes) = entry.slopes(&theta, &features, 0, &scores).unwrap();
        let slope_rounding = rounding(2, largest_abs(&theta), 1.0);
        assert!((slopes[0] - 1.0).abs() <= slope_rounding);
        assert!((slopes[1] + 0.48).abs() <= slope_rounding);

        // The state posterior's Gaussian prior over (g_1, g_2, x_0): the
        // conditional corner and information, then the entry factor
        // x_0 - slopes'g_m - offset ~ N(0, 1).
        let mut precision = [[0.0; 3]; 3];
        let mut information = [0.0; 3];
        for a in 0..2 {
            for b in 0..2 {
                precision[a][b] = scores.precision[[a, b]];
                information[a] += scores.precision[[a, b]] * scores.mean[b];
            }
        }
        let coefficients = [-slopes[0], -slopes[1], 1.0];
        for a in 0..3 {
            information[a] += coefficients[a] * offset;
            for b in 0..3 {
                precision[a][b] += coefficients[a] * coefficients[b];
            }
        }
        let covariance = cofactor_inverse(precision).0;
        let state_mean: f64 = (0..3).map(|b| covariance[2][b] * information[b]).sum();
        let state_variance = covariance[2][2];
        let (law_condition, law_magnitude) = law_scale(&law);
        let prior_matrix = Array2::from_shape_fn((3, 3), |(i, j)| precision[i][j]);
        let state_bar = rounding(
            ROUTE_TERMS,
            law_magnitude
                * (1.0 + largest_abs(&information) + largest_abs(&[offset, slopes[0], slopes[1]]))
                    .powi(2),
            law_condition * condition_of(&prior_matrix),
        );

        // The exact law from the covariance route: x_0 = offset + slopes'g_m + N(0, 1)
        // with g_m | g_0, c Gaussian.
        let conditional_mean = [1, 2].map(|i| mu[i] + sigma[i][0] * (g0 - mu[0]) / sigma[0][0]);
        let conditional = |a: usize, b: usize| sigma[a][b] - sigma[a][0] * sigma[0][b] / sigma[0][0];
        let exact_mean = offset + slopes[0] * conditional_mean[0] + slopes[1] * conditional_mean[1];
        let spread: f64 = (0..2)
            .map(|a| {
                (0..2)
                    .map(|b| slopes[a] * conditional(a + 1, b + 1) * slopes[b])
                    .sum::<f64>()
            })
            .sum();
        assert!(
            (state_mean - exact_mean).abs() <= state_bar,
            "entry state mean {state_mean} vs exact {exact_mean}"
        );
        assert!(
            (state_variance - (1.0 + spread)).abs() <= state_bar,
            "entry state variance {state_variance} vs exact {}",
            1.0 + spread
        );
        let genome = [g0, conditional_mean[0], conditional_mean[1]];
        let drive = entry.mean(&theta, &features, &genome, 0).unwrap();
        assert!((drive - exact_mean).abs() <= state_bar);
        // Control: filling the masked scores at their conditional mean gives the
        // entry state variance one and misses exactly this spread.
        assert!(
            (spread - 0.61416).abs() <= state_bar,
            "masked-score spread {spread}"
        );
    }

    #[test]
    fn the_closed_form_hessian_is_the_derivative_of_the_score_on_every_pattern() {
        let (scores, context, weights) = patterned_sample(36);
        let (g, patterns) = pattern_statistics(&scores, context.view(), &weights).unwrap();
        let law = law();
        let p = law.context_columns;
        let mean = Array2::from_shape_fn((g, p), |(j, b)| law.mean_coefficients[j * p + b]);
        let factor = law.factor();
        let covariance = factor.dot(&factor.t());
        let curvature = score_curvature(&patterns, &mean, &covariance).unwrap();
        let dimension = curvature.gradient.len();
        assert_eq!(dimension, 3 * 2 + 6);
        let shifted = |parameter: usize, step: f64| {
            let mut shifted_mean = mean.clone();
            let mut shifted_covariance = covariance.clone();
            if parameter < g * p {
                shifted_mean[[parameter / p, parameter % p]] += step;
            } else {
                for k in 0..g {
                    for l in 0..g {
                        if covariance_index(g, p, k, l) == parameter {
                            shifted_covariance[[k, l]] += step;
                        }
                    }
                }
            }
            (shifted_mean, shifted_covariance)
        };
        // Values round by the likelihood's account and score entries by their bands; the
        // closed-form Hessian rounds by d^2 products at its own magnitude. Each central
        // difference adds its truncation measured against the doubled step.
        let value_rounding = likelihood_account(&patterns, &mean, &covariance);
        let hessian_rounding = rounding(
            dimension * dimension,
            largest_abs(curvature.hessian.as_slice().unwrap()),
            1.0,
        );
        for column in 0..dimension {
            let parameter = (0..g)
                .flat_map(|k| (0..g).map(move |l| (k, l)))
                .find(|&(k, l)| covariance_index(g, p, k, l) == column)
                .map_or_else(|| mean[[column / p, column % p]], |(k, l)| covariance[[k, l]]);
            let h = central_step(parameter);
            let (central_value, value_bound) = central_difference(
                |step| {
                    let (m, c) = shifted(column, step);
                    observed_log_likelihood(&patterns, &m, &c).unwrap()
                },
                h,
                value_rounding,
            );
            let bar = value_bound + curvature.band[column];
            assert!(
                (central_value - curvature.gradient[column]).abs() <= bar,
                "score {column}: closed form {} vs central {central_value}, bound {bar:.3e}",
                curvature.gradient[column]
            );
            for row in 0..dimension {
                let (central, bound) = central_difference(
                    |step| {
                        let (m, c) = shifted(column, step);
                        score_curvature(&patterns, &m, &c).unwrap().gradient[row]
                    },
                    h,
                    curvature.band[row],
                );
                let bar = bound + hessian_rounding;
                assert!(
                    (central - curvature.hessian[[row, column]]).abs() <= bar,
                    "hessian ({row},{column}): closed form {} vs central {central}, bound {bar:.3e}",
                    curvature.hessian[[row, column]]
                );
            }
        }
    }

    #[test]
    fn the_stationarity_certificate_refuses_an_em_stall_and_passes_after_newton() {
        let (scores, context, weights) = patterned_sample(180);
        let (g, patterns) = pattern_statistics(&scores, context.view(), &weights).unwrap();
        let start = law();
        let p = start.context_columns;
        let mean = Array2::from_shape_fn((g, p), |(j, b)| start.mean_coefficients[j * p + b]);
        let factor = start.factor();
        let (em_mean, em_covariance) =
            expectation_maximization_limit(&patterns, mean, factor.dot(&factor.t())).unwrap();
        let em = score_curvature(&patterns, &em_mean, &em_covariance).unwrap();
        // Positive control: EM's monotone stall locates the mode only to about
        // sqrt(eps)-relative parameters, so its score is far outside a roundoff
        // bound. A certificate this stall passed would be vacuous.
        assert!(
            em.largest_ratio() > 1.0,
            "the EM stall passed the certificate: ratio {}",
            em.largest_ratio()
        );
        let (newton_mean, newton_covariance) =
            newton_finish(&patterns, em_mean, em_covariance).unwrap();
        let newton = score_curvature(&patterns, &newton_mean, &newton_covariance).unwrap();
        assert!(
            newton.largest_ratio() <= 1.0,
            "after Newton the ratio is {}",
            newton.largest_ratio()
        );
        eprintln!(
            "score/roundoff ratio: EM stall {:.3e}, after Newton {:.3e}",
            em.largest_ratio(),
            newton.largest_ratio()
        );
        // Two certified maxima from different starts agree within |V| (band_1 + band_2).
        let (other_mean, other_covariance) = initial_law(&patterns, g, p).unwrap();
        let (other_mean, other_covariance) =
            certified_fit(&patterns, other_mean, other_covariance).unwrap();
        let other = score_curvature(&patterns, &other_mean, &other_covariance).unwrap();
        let information = newton.hessian.mapv(|v| -v);
        let posterior = cholesky_solve_matrix(
            &lower_factor(&information, "the certified information").unwrap(),
            &Array2::<f64>::eye(information.nrows()),
        )
        .mapv(f64::abs);
        let bands = Array1::from(newton.band.clone()) + &Array1::from(other.band.clone());
        let bound = posterior.dot(&bands);
        for (index, (fitted, finished)) in other_mean.iter().zip(newton_mean.iter()).enumerate() {
            assert!(
                (fitted - finished).abs() <= bound[index],
                "fits from two starts disagree at {index}: {fitted} vs {finished}, bound {:.3e}",
                bound[index]
            );
        }
    }

    #[test]
    fn at_small_n_the_plug_in_score_law_shifts_a_masked_rate_by_its_measured_parameter_spread() {
        let n = 200;
        let p = 10;
        let alphas = [
            0.754_877_666_246_692_7,
            0.569_840_290_998_053_2,
            0.362_646_470_928_318_3,
        ];
        let mut context = Array2::<f64>::zeros((n, p));
        let mut scores = Vec::with_capacity(n);
        for i in 0..n {
            let x = 2.0 * (i as f64 + 0.5) / n as f64 - 1.0;
            let row = [
                1.0,
                x,
                x * x - 1.0 / 3.0,
                (3.0 * x).sin(),
                (2.0 * x).cos(),
                (5.0 * x).sin(),
                x * x * x,
                (7.0 * x).cos(),
                x * (1.5 * x).sin(),
                0.5 * (4.0 * x).cos(),
            ];
            for (b, value) in row.iter().enumerate() {
                context[[i, b]] = *value;
            }
            // Unit-variance uniform residuals.
            let e = alphas.map(|alpha| 3.0_f64.sqrt() * (2.0 * (i as f64 * alpha).fract() - 1.0));
            let g = [
                0.3 + 0.5 * x + e[0],
                -0.2 + 0.4 * row[3] + 0.6 * e[0] + 0.8 * e[1],
                0.1 - 0.3 * row[2] - 0.5 * e[1] + 0.7 * e[2],
            ];
            // Heavy missingness: two rows in five observe nothing, one in five
            // observes everything, and the rest miss one or two scores.
            let pattern = match i % 5 {
                0 | 1 => [false, false, false],
                2 => [true, true, true],
                3 => [true, i % 2 == 0, true],
                _ => [i % 3 != 0, true, i % 2 == 1],
            };
            scores.push((0..3).map(|j| pattern[j].then_some(g[j])).collect::<Vec<_>>());
        }
        let weights = vec![1.0; n];
        let fit = maximum_likelihood(&scores, context.view(), &weights);
        let (g, patterns) = pattern_statistics(&scores, context.view(), &weights).unwrap();
        let mean = Array2::from_shape_fn((g, p), |(j, b)| fit.mean_coefficients[j * p + b]);
        let factor = fit.factor();
        let covariance = factor.dot(&factor.t());
        let curvature = score_curvature(&patterns, &mean, &covariance).unwrap();
        let dimension = curvature.gradient.len();
        let information = curvature.hessian.mapv(|v| -v);
        let information_factor =
            cholesky_factor_in_place(information.view(), CholeskyGuard::FiniteStrict).unwrap();
        let posterior = cholesky_solve_matrix(&information_factor, &Array2::<f64>::eye(dimension));

        let subject = context.row(57).to_vec();
        let observed = [Some(0.4), None, None];
        let loadings = [0.6, -0.4, 0.5];
        // log E[exp(a'g) | g_0, c*]: the log-rate a masked subject's drive carries.
        let log_rate = |law: &GeneticScoreLaw| -> f64 {
            let masked = law.conditional(&subject, &observed).unwrap();
            let tail = Array1::from(vec![loadings[1], loadings[2]]);
            let spread = cholesky_solve_vector(&masked.precision_factor, &tail).dot(&tail);
            loadings[0] * 0.4 + loadings[1] * masked.mean[0] + loadings[2] * masked.mean[1]
                + 0.5 * spread
        };
        let law_at = |theta: &[f64]| -> GeneticScoreLaw {
            let shifted = Array2::from_shape_fn((g, g), |(k, l)| theta[covariance_index(g, p, k, l)]);
            let shifted_factor =
                cholesky_factor_in_place(shifted.view(), CholeskyGuard::FiniteStrict).unwrap();
            GeneticScoreLaw {
                scores: g,
                context_columns: p,
                mean_coefficients: theta[..g * p].to_vec(),
                covariance_factor: shifted_factor.iter().copied().collect(),
            }
        };
        let mut theta = fit.mean_coefficients.clone();
        theta.resize(dimension, 0.0);
        for k in 0..g {
            for l in 0..=k {
                theta[covariance_index(g, p, k, l)] = covariance[[k, l]];
            }
        }
        // Rebuilding the law from (A, vech Sigma) re-factors Sigma: the log-rate moves by at
        // most that route's rounding at Sigma's conditioning times the log-rate's summand
        // scale (the A c terms it sums and its own value).
        let summand_scale = (0..g)
            .map(|l| {
                (0..p)
                    .map(|d| (fit.mean_coefficients[l * p + d] * subject[d]).abs())
                    .sum::<f64>()
            })
            .sum::<f64>()
            + log_rate(&fit).abs();
        let rebuild_difference = (log_rate(&law_at(&theta)) - log_rate(&fit)).abs();
        let rebuild_bar = rounding(ROUTE_TERMS, summand_scale, condition_of(&covariance));

        // At fixed Sigma the log-rate is affine in A with gradient b ⊗ c*, where b is
        // a_m on the missing rows and -R'a_m on the observed row. Central differences of
        // an affine function carry no truncation term at any step, so the check steps by
        // one and its whole error is the two evaluations' rounding.
        let b = [
            -(loadings[1] * covariance[[1, 0]] + loadings[2] * covariance[[2, 0]]) / covariance[[0, 0]],
            loadings[1],
            loadings[2],
        ];
        let rate_at = |shifts: &[(usize, f64)]| {
            let mut shifted = theta.clone();
            for &(index, step) in shifts {
                shifted[index] += step;
            }
            log_rate(&law_at(&shifted))
        };
        // Rounding account of one log-rate evaluation. It forms the p products A_lb c*_b per
        // score (plus the observed score in the residual), weights them by Lambda = Sigma^-1
        // into the information, solves on Lambda_mm (amplification at most ||C||_inf with
        // C = Lambda_mm^-1), loads the result with a_m, and adds the observed and spread
        // terms. A k-term sum rounds by at most k eps sum|terms| to first order, and every
        // sum here has at most p + 2g terms.
        let fitted_conditional = fit.conditional(&subject, &observed).unwrap();
        let tail_loadings = Array1::from(vec![loadings[1], loadings[2]]);
        let fitted_spread = cholesky_solve_vector(&fitted_conditional.precision_factor, &tail_loadings)
            .dot(&tail_loadings);
        let conditional_covariance =
            cholesky_solve_matrix(&fitted_conditional.precision_factor, &Array2::<f64>::eye(2));
        let solve_norm = (0..2)
            .map(|a| (0..2).map(|c| conditional_covariance[[a, c]].abs()).sum::<f64>())
            .fold(0.0_f64, f64::max);
        let sigma_inverse = cofactor_inverse([
            [covariance[[0, 0]], covariance[[0, 1]], covariance[[0, 2]]],
            [covariance[[1, 0]], covariance[[1, 1]], covariance[[1, 2]]],
            [covariance[[2, 0]], covariance[[2, 1]], covariance[[2, 2]]],
        ])
        .0;
        let loading_weight = loadings[1].abs() + loadings[2].abs();
        let terms = (p + 2 * g) as f64;
        // The information's magnitude with coefficient `index` shifted by `step`
        // (`usize::MAX` shifts nothing).
        let information_magnitude = |index: usize, step: f64| -> f64 {
            let row = |l: usize| -> f64 {
                (0..p)
                    .map(|d| {
                        let shift = if l * p + d == index { step } else { 0.0 };
                        ((theta[l * p + d] + shift) * subject[d]).abs()
                    })
                    .sum::<f64>()
                    + if l == 0 { 0.4 } else { 0.0 }
            };
            [1usize, 2]
                .into_iter()
                .map(|a| (0..g).map(|l| sigma_inverse[a][l].abs() * row(l)).sum::<f64>())
                .sum()
        };
        let evaluation_rounding = |index: usize, step: f64| -> f64 {
            f64::EPSILON
                * terms
                * (loading_weight * solve_norm * information_magnitude(index, step)
                    + (loadings[0] * 0.4).abs()
                    + fitted_spread)
        };
        // The A-part of the plug-in shift is exact: the conditional Laplace posterior
        // of A given Sigma is Gaussian and the log-rate is affine in A, so its
        // posterior average is exp(s^2 / 2) with s^2 = J' (-H_AA)^-1 J.
        let information_mean = Array2::from_shape_fn((g * p, g * p), |(r, c)| information[[r, c]]);
        let mean_factor =
            cholesky_factor_in_place(information_mean.view(), CholeskyGuard::FiniteStrict).unwrap();
        let direction = Array1::from_shape_fn(g * p, |index| b[index / p] * subject[index % p]);
        let mean_spread = cholesky_solve_vector(&mean_factor, &direction).dot(&direction);
        // s^2 and both bounds are quadratic forms on (g p)^2 products at the information's
        // conditioning.
        let mean_posterior = cholesky_solve_matrix(&mean_factor, &Array2::<f64>::eye(g * p));
        let loewner_rounding = rounding(
            g * p * g * p,
            largest_abs(direction.as_slice().unwrap()).powi(2) * infinity_norm(&mean_posterior),
            condition_of(&information_mean),
        );
        // Dropping every pattern but the complete one lowers the information in the
        // Loewner order, so s^2 <= (b' Sigma b)(c*' S_complete^-1 c*).
        let complete = patterns
            .iter()
            .find(|stats| stats.present.len() == g)
            .unwrap();
        let complete_factor =
            cholesky_factor_in_place(complete.context.view(), CholeskyGuard::FiniteStrict).unwrap();
        let subject_row = Array1::from(subject.clone());
        let leverage = cholesky_solve_vector(&complete_factor, &subject_row).dot(&subject_row);
        let b_sigma_b: f64 = (0..g)
            .map(|j| (0..g).map(|l| b[j] * covariance[[j, l]] * b[l]).sum::<f64>())
            .sum();
        // The joint (A, vech Sigma) Laplace average to second order,
        // E[exp F] = exp(F^)(1 + (grad F' V grad F + tr(hess F V)) / 2), with the log-rate's
        // derivatives in closed form, not by differences of a log-rate that sums large A c terms.
        // With score 0 observed at g_0 = 0.4, q = a_m' Sigma_m0, S = Sigma_00, r0 = g_0 - A_0 c*
        // and K = a_m' Sigma_mm a_m:
        //     F = a_0 g_0 + a_m' A_m c* + q r0 / S + K / 2 - q^2 / (2 S),
        // affine in A, with second derivatives only in (A_0, Sigma_00, Sigma_10, Sigma_20).
        let (a1, a2) = (loadings[1], loadings[2]);
        let big_s = covariance[[0, 0]];
        let q = a1 * covariance[[1, 0]] + a2 * covariance[[2, 0]];
        let r0 = 0.4 - (0..p).map(|d| fit.mean_coefficients[d] * subject[d]).sum::<f64>();
        let closed_form = loadings[0] * 0.4
            + (0..p)
                .map(|d| {
                    (a1 * fit.mean_coefficients[p + d] + a2 * fit.mean_coefficients[2 * p + d])
                        * subject[d]
                })
                .sum::<f64>()
            + q * r0 / big_s
            + 0.5
                * (a1 * a1 * covariance[[1, 1]]
                    + 2.0 * a1 * a2 * covariance[[2, 1]]
                    + a2 * a2 * covariance[[2, 2]])
            - 0.5 * q * q / big_s;
        let (s00, s10, s20) = (
            covariance_index(g, p, 0, 0),
            covariance_index(g, p, 1, 0),
            covariance_index(g, p, 2, 0),
        );
        let (s11, s21, s22) = (
            covariance_index(g, p, 1, 1),
            covariance_index(g, p, 2, 1),
            covariance_index(g, p, 2, 2),
        );
        let closed_derivatives = |parameters: &[f64]| -> (Vec<f64>, Array2<f64>) {
            let sigma = |k: usize, l: usize| parameters[covariance_index(g, p, k, l)];
            let big_s = sigma(0, 0);
            let q = a1 * sigma(1, 0) + a2 * sigma(2, 0);
            let r0 = 0.4 - (0..p).map(|d| parameters[d] * subject[d]).sum::<f64>();
            let mut gradient = vec![0.0; dimension];
            let mut hessian = Array2::<f64>::zeros((dimension, dimension));
            for d in 0..p {
                gradient[d] = -(q / big_s) * subject[d];
                gradient[p + d] = a1 * subject[d];
                gradient[2 * p + d] = a2 * subject[d];
                for (index, value) in [
                    (s10, -a1 / big_s),
                    (s20, -a2 / big_s),
                    (s00, q / (big_s * big_s)),
                ] {
                    hessian[[d, index]] = value * subject[d];
                    hessian[[index, d]] = value * subject[d];
                }
            }
            gradient[s10] = a1 * (r0 - q) / big_s;
            gradient[s20] = a2 * (r0 - q) / big_s;
            gradient[s00] = (0.5 * q * q - q * r0) / (big_s * big_s);
            gradient[s11] = 0.5 * a1 * a1;
            gradient[s21] = a1 * a2;
            gradient[s22] = 0.5 * a2 * a2;
            for (i, j, value) in [
                (s10, s10, -a1 * a1 / big_s),
                (s20, s20, -a2 * a2 / big_s),
                (s10, s20, -a1 * a2 / big_s),
                (s10, s00, -a1 * (r0 - q) / (big_s * big_s)),
                (s20, s00, -a2 * (r0 - q) / (big_s * big_s)),
                (s00, s00, (2.0 * q * r0 - q * q) / big_s.powi(3)),
            ] {
                hessian[[i, j]] = value;
                hessian[[j, i]] = value;
            }
            (gradient, hessian)
        };
        let (gradient, hessian) = closed_derivatives(&theta);

        // The closed forms against the law itself. Each Sigma-coordinate gradient entry is
        // checked against central differences of the log-rate, each Sigma-block Hessian entry
        // against central differences of the closed-form gradient. Every bar is the difference's
        // Richardson truncation plus its evaluation's rounding over h, plus the closed form's own
        // rounding; the gradient entries, all nonzero by construction, must be resolved beyond it.
        // The A coordinates are pinned by the affine check above.
        let value_rounding = evaluation_rounding(usize::MAX, 0.0);
        let gradient_rounding = rounding(ROUTE_TERMS, largest_abs(&gradient), 1.0);
        let hessian_rounding = rounding(ROUTE_TERMS, largest_abs(hessian.as_slice().unwrap()), 1.0);
        let mut second_order = 0.0;
        let mut magnitude = 0.0;
        for i in 0..dimension {
            for j in 0..dimension {
                let term = posterior[[i, j]] * (hessian[[i, j]] + gradient[i] * gradient[j]);
                second_order += term;
                magnitude += term.abs();
            }
        }
        let joint_shift = 0.5 * second_order;
        // The expansion's own truncation. For F's quadratic model the Laplace average is exact,
        // det(I - V H)^(-1/2) exp(g' (V^-1 - H)^-1 g / 2) - 1, so its difference from the
        // second-order shift is that model's truncation. It exists only while V^-1 - H is
        // positive definite.
        let model_information = &information - &hessian;
        let model_factor = cholesky_factor_in_place(model_information.view(), CholeskyGuard::FiniteStrict)
            .expect("the quadratic model's Laplace average diverges: V^-1 - H is not positive definite");
        let gradient_vector = Array1::from(gradient.clone());
        let quadratic_model = (0.5
            * (factor_log_determinant(&information_factor) - factor_log_determinant(&model_factor))
            + 0.5 * cholesky_solve_vector(&model_factor, &gradient_vector).dot(&gradient_vector))
        .exp_m1();
        let quadratic_truncation = (quadratic_model - joint_shift).abs();

        // F departs from its quadratic model only through U / S, with U = q r0 - q^2 / 2 quadratic
        // and r0 = 0.4 - t, t = c*'A_0. Its third derivatives therefore live on the four reduced
        // coordinates (t, Sigma_00, Sigma_10, Sigma_20): -6 U / S^4, 2 U_x / S^3 and -U_xy / S^2
        // by how many indices name S = Sigma_00, and zero when none does.
        let big_u = q * r0 - 0.5 * q * q;
        let first_u = |x: usize| -> f64 {
            match x {
                0 => -q,
                2 => a1 * (r0 - q),
                3 => a2 * (r0 - q),
                _ => 0.0,
            }
        };
        let second_u = |x: usize, y: usize| -> f64 {
            match (x.min(y), x.max(y)) {
                (0, 2) => -a1,
                (0, 3) => -a2,
                (2, 2) => -a1 * a1,
                (2, 3) => -a1 * a2,
                (3, 3) => -a2 * a2,
                _ => 0.0,
            }
        };
        let third = |x: usize, y: usize, z: usize| -> f64 {
            let others: Vec<usize> = [x, y, z].into_iter().filter(|&r| r != 1).collect();
            match others.len() {
                0 => -6.0 * big_u / big_s.powi(4),
                1 => 2.0 * first_u(others[0]) / big_s.powi(3),
                2 => -second_u(others[0], others[1]) / (big_s * big_s),
                _ => 0.0,
            }
        };
        // Each reduced coordinate as a combination of (A, vech Sigma), and their posterior
        // covariance.
        let combination = |x: usize| -> Vec<(usize, f64)> {
            match x {
                0 => (0..p).map(|d| (d, subject[d])).collect(),
                1 => vec![(s00, 1.0)],
                2 => vec![(s10, 1.0)],
                _ => vec![(s20, 1.0)],
            }
        };
        let reduced_covariance = Array2::from_shape_fn((4, 4), |(x, y)| {
            combination(x)
                .iter()
                .map(|&(i, u)| {
                    combination(y)
                        .iter()
                        .map(|&(j, v)| u * posterior[[i, j]] * v)
                        .sum::<f64>()
                })
                .sum::<f64>()
        });
        // E[R3^2] for R3 = F[d, d, d] / 6 with d ~ N(0, V), exactly by Isserlis on the reduced
        // coordinates: a sixth moment sums its factors' covariances over the fifteen pairings.
        let mut pairings: Vec<[(usize, usize); 3]> = Vec::with_capacity(15);
        for partner in 1..6 {
            let rest: Vec<usize> = (1..6).filter(|&position| position != partner).collect();
            for (one, two, three) in [(1, 2, 3), (2, 1, 3), (3, 1, 2)] {
                pairings.push([(0, partner), (rest[0], rest[one]), (rest[two], rest[three])]);
            }
        }
        let mut reduced_moment = 0.0;
        for tuple in 0..4_usize.pow(6) {
            let index: [usize; 6] =
                std::array::from_fn(|position| tuple / 4_usize.pow(position as u32) % 4);
            let coefficient = third(index[0], index[1], index[2]) * third(index[3], index[4], index[5]);
            if coefficient != 0.0 {
                let moment: f64 = pairings
                    .iter()
                    .map(|pairing| {
                        pairing
                            .iter()
                            .map(|&(x, y)| reduced_covariance[[index[x], index[y]]])
                            .product::<f64>()
                    })
                    .sum();
                reduced_moment += coefficient * moment;
            }
        }
        let cubic_second_moment = reduced_moment / 36.0;
        // For comparison only, the coordinatewise Hoelder sum sum |F_ijk| sd_i sd_j sd_k over
        // (A, vech Sigma), which depends on the context basis.
        let reduced_of = |index: usize| -> Option<(usize, f64)> {
            if index < p {
                Some((0, subject[index]))
            } else if index == s00 {
                Some((1, 1.0))
            } else if index == s10 {
                Some((2, 1.0))
            } else if index == s20 {
                Some((3, 1.0))
            } else {
                None
            }
        };
        let deviations: Vec<f64> = (0..dimension).map(|i| posterior[[i, i]].sqrt()).collect();
        let relevant: Vec<(usize, usize, f64)> = (0..dimension)
            .filter_map(|i| reduced_of(i).map(|(x, weight)| (i, x, weight)))
            .collect();
        let mut coordinate_sum = 0.0;
        let mut nonzero_triples = 0_usize;
        let mut largest_term = (0.0_f64, [0_usize; 3], 0.0_f64, 0.0_f64);
        for &(i, x, u) in &relevant {
            for &(j, y, v) in &relevant {
                for &(k, z, weight) in &relevant {
                    let derivative = third(x, y, z) * u * v * weight;
                    if derivative != 0.0 {
                        nonzero_triples += 1;
                        let spread = deviations[i] * deviations[j] * deviations[k];
                        let term = derivative.abs() * spread;
                        coordinate_sum += term;
                        if term > largest_term.0 {
                            largest_term = (term, [i, j, k], derivative, spread);
                        }
                    }
                }
            }
        }
        // A leading-order estimate of the cubic remainder, not a bound: it takes F's third
        // derivatives at their mode values and the remainder to first order. Cauchy-Schwarz gives
        // E[e^(F_q - F^) R3] <= sqrt(E e^(2 (F_q - F^))) sqrt(E R3^2), and E e^(2 (F_q - F^)) is the
        // quadratic model's average at (2g, 2H), which exists only while V^-1 - 2H is positive
        // definite.
        let doubled_information = &information - &(&hessian * 2.0);
        let doubled_factor = cholesky_factor_in_place(doubled_information.view(), CholeskyGuard::FiniteStrict)
            .expect("E e^(2 (F_q - F^)) diverges: V^-1 - 2H is not positive definite, so the Cauchy-Schwarz estimate does not exist");
        let doubled_moment = (0.5
            * (factor_log_determinant(&information_factor) - factor_log_determinant(&doubled_factor))
            + 2.0 * cholesky_solve_vector(&doubled_factor, &gradient_vector).dot(&gradient_vector))
        .exp();
        let cubic_remainder = doubled_moment.sqrt() * cubic_second_moment.sqrt();
        let coordinate_estimate = doubled_moment.sqrt() * 15.0_f64.sqrt() / 6.0 * coordinate_sum;
        // A d^2-term sum at the information's equilibrated conditioning, plus both truncation terms.
        let shift_rounding =
            0.5 * rounding(dimension * dimension, magnitude, condition_of(&information));
        let shift_error = shift_rounding + quadratic_truncation + cubic_remainder;
        let mean_shift = (0.5 * mean_spread).exp_m1();
        let masked = fit.conditional(&subject, &observed).unwrap();
        let tail = Array1::from(vec![loadings[1], loadings[2]]);
        let latent_spread = cholesky_solve_vector(&masked.precision_factor, &tail)
            .dot(&tail)
            .exp_m1()
            .sqrt();
        // Floor: every pattern's embedded Sigma_oo^-1 lies below Sigma^-1 in the
        // Loewner order, so -H_AA <= Sigma^-1 ⊗ S_all and
        // s^2 >= (b' Sigma b)(c*' S_all^-1 c*) > 0. The plug-in shift is resolvably
        // nonzero, so the comparison below is not met by a vanishing left side.
        let all_context = patterns
            .iter()
            .fold(Array2::<f64>::zeros((p, p)), |sum, stats| sum + &stats.context);
        let all_factor =
            cholesky_factor_in_place(all_context.view(), CholeskyGuard::FiniteStrict).unwrap();
        let all_leverage = cholesky_solve_vector(&all_factor, &subject_row).dot(&subject_row);
        let floor = b_sigma_b * all_leverage;
        eprintln!(
            "plug-in cubic diagnostics: S {big_s:.4e}, U {big_u:.4e}, sd Sigma_00 {:.3e}, sd Sigma_10 {:.3e}, sd Sigma_20 {:.3e}, sd t {:.3e}, largest A_0 sd {:.3e}, information equilibrated condition {:.3e}, largest posterior entry {:.3e}, nonzero triples {nonzero_triples}, largest term {:.3e} at {:?} (derivative {:.3e}, sd product {:.3e}), coordinatewise sum {coordinate_sum:.3e}, coordinatewise estimate {coordinate_estimate:.2e}, reduced second moment {cubic_second_moment:.3e}, doubled moment {doubled_moment:.4e}",
            deviations[s00],
            deviations[s10],
            deviations[s20],
            reduced_covariance[[0, 0]].sqrt(),
            largest_abs(&deviations[..p]),
            condition_of(&information),
            largest_abs(posterior.as_slice().unwrap()),
            largest_term.0,
            largest_term.1,
            largest_term.2,
            largest_term.3
        );
        eprintln!(
            "plug-in measurement n={n} p={p} d={dimension}: rate shift A-part exact {mean_shift:.4e} ± {:.2e} (all-rows floor {:.4e}, complete-rows bound {:.4e}), joint second order {joint_shift:.4e} ± {shift_error:.2e} (equilibrated rounding {shift_rounding:.2e}, quadratic model {quadratic_model:.4e}, quadratic truncation {quadratic_truncation:.2e}, cubic remainder {cubic_remainder:.2e}); latent relative spread {latent_spread:.4e}; ratio {:.4e}",
            0.5 * (0.5 * mean_spread).exp() * loewner_rounding,
            (0.5 * floor).exp_m1(),
            (0.5 * b_sigma_b * leverage).exp_m1(),
            joint_shift.abs() / latent_spread
        );

        // Every check follows the measurement lines, so they print whatever fails.
        assert!(
            rebuild_difference <= rebuild_bar,
            "rebuilding the law from (A, vech Sigma) moves the log-rate by {rebuild_difference:.3e}, bound {rebuild_bar:.3e}"
        );
        // The affine check: central differences at step one against b ⊗ c*.
        for j in 0..g {
            for column in 0..p {
                let index = j * p + column;
                let central = (rate_at(&[(index, 1.0)]) - rate_at(&[(index, -1.0)])) / 2.0;
                let exact = b[j] * subject[column];
                let bar = 0.5 * (evaluation_rounding(index, 1.0) + evaluation_rounding(index, -1.0));
                assert!(
                    (central - exact).abs() <= bar,
                    "d log-rate / dA[{j},{column}]: central {central} vs b ⊗ c* {exact}, rounding bound {bar:.3e}"
                );
            }
        }
        assert!(
            mean_spread > 0.0 && mean_spread <= b_sigma_b * leverage + loewner_rounding,
            "A-uncertainty spread {mean_spread} vs complete-rows bound {}",
            b_sigma_b * leverage
        );
        assert!(
            (closed_form - log_rate(&fit)).abs()
                <= rounding(ROUTE_TERMS, summand_scale, condition_of(&covariance)),
            "closed-form log-rate {closed_form} vs the law's {}",
            log_rate(&fit)
        );
        let sigma_coordinates = [s00, s10, s20, s11, s21, s22];
        for &k in &sigma_coordinates {
            let h = central_step(theta[k]);
            let (central, bound) = central_difference(|step| rate_at(&[(k, step)]), h, value_rounding);
            let bar = bound + gradient_rounding;
            assert!(
                (central - gradient[k]).abs() <= bar,
                "closed-form gradient at {k}: {} vs central {central}, bound {bar:.3e}",
                gradient[k]
            );
            assert!(
                gradient[k].abs() > bar,
                "gradient entry {k} ({}) is not resolved beyond its bound {bar:.3e}",
                gradient[k]
            );
            // Every Hessian column in a Sigma coordinate, including the A_0 x Sigma cross entries
            // built from c*; the entries nonzero by construction (the (Sigma_00, Sigma_10,
            // Sigma_20) block and their A_0 crosses where c*_d != 0) must be resolved beyond it.
            for l in 0..dimension {
                let (central, bound) = central_difference(
                    |step| {
                        let mut shifted = theta.clone();
                        shifted[k] += step;
                        closed_derivatives(&shifted).0[l]
                    },
                    h,
                    gradient_rounding,
                );
                let bar = bound + hessian_rounding;
                assert!(
                    (central - hessian[[l, k]]).abs() <= bar,
                    "closed-form hessian ({l},{k}): {} vs central {central}, bound {bar:.3e}",
                    hessian[[l, k]]
                );
                let curved = [s00, s10, s20];
                let structural = curved.contains(&k)
                    && (curved.contains(&l) || (l < p && subject[l] != 0.0));
                assert!(
                    !structural || hessian[[l, k]].abs() > bar,
                    "hessian entry ({l},{k}) ({}) is not resolved beyond its bound {bar:.3e}",
                    hessian[[l, k]]
                );
            }
        }
        assert!(
            mean_spread >= floor - loewner_rounding,
            "A-uncertainty spread {mean_spread} below the all-rows floor {floor}"
        );
        // Verdict (gam-44 ruling, SPEC 3): the plug-in law's rate differs from its
        // Laplace posterior average by less than the functional's own latent spread,
        // so the plug-in stays documented with this measured bound.
        // Scope: the plug-in measured is the certified unpenalized maximum. Production's `fit`
        // plugs in exactly that law when the context is intercept-only (the default). With
        // context columns it plugs in the evidence-selected penalized mode. That mode's shift
        // tr(V (grad^2 F + grad F grad F'))/2 is not monotone in V, since the matrix is
        // indefinite here, so this verdict says nothing about that plug-in; it is re-measured at
        // the penalized mode when a context-dependent caller lands.
        assert!(
            joint_shift.abs() + shift_error <= latent_spread,
            "plug-in rate shift {joint_shift} ± {shift_error} exceeds the latent spread {latent_spread}"
        );
        // Control: under one row's information (V scaled by n) the A-part shift alone is
        // exactly expm1(n s^2 / 2) for any V, since the log-rate is affine in A given Sigma. It
        // exceeds the latent spread, so the verdict is an inequality a real plug-in shift fails.
        let one_row_shift = (0.5 * n as f64 * mean_spread).exp_m1();
        assert!(
            one_row_shift > latent_spread,
            "one-row A-part shift {one_row_shift} does not exceed the latent spread {latent_spread}"
        );
    }

    #[test]
    fn the_interaction_second_moment_averages_the_augmented_score_outer_product_over_context() {
        let law = law();
        let context = Array2::from_shape_vec((2, 2), vec![1.0, 0.6, 1.0, -0.4]).unwrap();
        let weights = [1.0, 3.0];
        let moment = law.interaction_second_moment(context.view(), &weights).unwrap();
        let sigma = covariance_of(&law);
        let (condition, magnitude) = law_scale(&law);
        let moment_bar = rounding(
            ROUTE_TERMS,
            magnitude * (1.0 + 2.0 * largest_abs(&law.mean_coefficients)).powi(2),
            condition,
        );
        let mut expected = [[0.0; 4]; 4];
        for (row, w) in [[1.0, 0.6], [1.0, -0.4]].into_iter().zip(weights) {
            let mu = mean_of(&law, row);
            let augmented = [1.0, mu[0], mu[1], mu[2]];
            for i in 0..4 {
                for j in 0..4 {
                    expected[i][j] += w * augmented[i] * augmented[j] / 4.0;
                }
            }
        }
        for i in 0..3 {
            for j in 0..3 {
                expected[i + 1][j + 1] += sigma[i][j];
            }
        }
        for i in 0..4 {
            for j in 0..4 {
                assert!((moment[[i, j]] - expected[i][j]).abs() <= moment_bar);
            }
        }
        // Column 0 of the score context is the intercept, exactly one.
        assert!(law.conditional(&[0.9, 0.6], &[None, None, None]).is_err());
        let shifted = Array2::from_shape_vec((1, 2), vec![2.0, 0.0]).unwrap();
        assert!(law.interaction_second_moment(shifted.view(), &[1.0]).is_err());
    }

    /// The unpenalized certified maximum, for tests of the likelihood machinery itself.
    fn maximum_likelihood(
        scores: &[Vec<Option<f64>>],
        context: ArrayView2<'_, f64>,
        weights: &[f64],
    ) -> GeneticScoreLaw {
        let (g, patterns) = pattern_statistics(scores, context, weights).unwrap();
        let (mean, covariance) = initial_law(&patterns, g, context.ncols()).unwrap();
        let (mean, covariance) = certified_fit(&patterns, mean, covariance).unwrap();
        GeneticScoreLaw::from_parts(&mean, &covariance).unwrap()
    }

    #[test]
    fn the_context_prior_is_the_ridge_pseudo_pattern_on_complete_scores() {
        let n = 30;
        let lambda = 2.5;
        let mut context = Array2::<f64>::zeros((n, 3));
        let mut scores = Vec::with_capacity(n);
        let weights: Vec<f64> = (0..n).map(|i| 1.0 + 0.25 * (i % 3) as f64).collect();
        for i in 0..n {
            let x = 2.0 * (i as f64 + 0.5) / n as f64 - 1.0;
            let z = (i as f64 * 0.754_877_666_246_692_7).fract() - 0.5;
            let e = (i as f64 * 0.569_840_290_998_053_2).fract() - 0.5;
            context[[i, 0]] = 1.0;
            context[[i, 1]] = x;
            context[[i, 2]] = z;
            scores.push(vec![
                Some(0.2 + 0.7 * x + e),
                Some(-0.4 + 0.3 * z - 0.5 * e + 0.2 * x * x),
            ]);
        }
        let (g, patterns) = pattern_statistics(&scores, context.view(), &weights).unwrap();
        let prior = ContextPrior::new(context.view(), &weights).unwrap().unwrap();
        let mut penalized = patterns.clone();
        penalized.push(prior.pattern(g, lambda));
        let (mean, covariance) = initial_law(&patterns, g, 3).unwrap();
        let (mean, covariance) = certified_fit(&penalized, mean, covariance).unwrap();

        // Closed form: A = T_gc (T_cc + lambda G)^-1 and Sigma = (T_gg - A T_gc') / (W + P_c),
        // from row sums and the centered Gram, inverted by cofactors.
        let total: f64 = weights.iter().sum();
        let mut centre = [0.0; 3];
        let mut t_cc = [[0.0; 3]; 3];
        let mut t_gc = [[0.0; 3]; 2];
        let mut t_gg = [[0.0; 2]; 2];
        for i in 0..n {
            let c = [context[[i, 0]], context[[i, 1]], context[[i, 2]]];
            let s = [
                scores[i][0].unwrap_or(f64::NAN),
                scores[i][1].unwrap_or(f64::NAN),
            ];
            let w = weights[i];
            for a in 0..3 {
                centre[a] += w * c[a] / total;
                for b in 0..3 {
                    t_cc[a][b] += w * c[a] * c[b];
                }
            }
            for j in 0..2 {
                for b in 0..3 {
                    t_gc[j][b] += w * s[j] * c[b];
                }
                for l in 0..2 {
                    t_gg[j][l] += w * s[j] * s[l];
                }
            }
        }
        let mut shrunk = t_cc;
        for i in 0..n {
            for a in 1..3 {
                for b in 1..3 {
                    shrunk[a][b] += lambda
                        * weights[i]
                        * (context[[i, a]] - centre[a])
                        * (context[[i, b]] - centre[b])
                        / total;
                }
            }
        }
        let inverse = cofactor_inverse(shrunk).0;
        let mut expected_mean = [[0.0; 3]; 2];
        for j in 0..2 {
            for b in 0..3 {
                expected_mean[j][b] = (0..3).map(|a| t_gc[j][a] * inverse[a][b]).sum();
            }
        }
        // The penalized fit is certified, so each parameter lies within |V| band of the exact
        // maximum; the closed form's n row sums of 27 products and its cofactor inverse add
        // their rounding at the shrunk Gram's conditioning.
        let curvature = score_curvature(&penalized, &mean, &covariance).unwrap();
        let information = curvature.hessian.mapv(|v| -v);
        let posterior = cholesky_solve_matrix(
            &lower_factor(&information, "the penalized information").unwrap(),
            &Array2::<f64>::eye(information.nrows()),
        )
        .mapv(f64::abs);
        let parameter_bound = posterior.dot(&Array1::from(curvature.band.clone()));
        let shrunk_matrix = Array2::from_shape_fn((3, 3), |(a, b)| shrunk[a][b]);
        let oracle_rounding = rounding(
            n * ROUTE_TERMS,
            (1.0 + largest_abs(&expected_mean.concat())) * (1.0 + largest_abs(&t_gg.concat())),
            condition_of(&shrunk_matrix),
        );
        for j in 0..2 {
            for b in 0..3 {
                let bar = parameter_bound[j * 3 + b] + oracle_rounding;
                assert!(
                    (mean[[j, b]] - expected_mean[j][b]).abs() <= bar,
                    "ridge mean ({j},{b}): {} vs {}, bound {bar:.3e}",
                    mean[[j, b]],
                    expected_mean[j][b]
                );
            }
        }
        for j in 0..2 {
            for l in 0..2 {
                let explained: f64 = (0..3).map(|b| expected_mean[j][b] * t_gc[l][b]).sum();
                let expected = (t_gg[j][l] - explained) / (total + 2.0);
                let bar = parameter_bound[covariance_index(2, 3, j, l)] + oracle_rounding;
                assert!(
                    (covariance[[j, l]] - expected).abs() <= bar,
                    "ridge covariance ({j},{l}): {} vs {expected}, bound {bar:.3e}",
                    covariance[[j, l]]
                );
            }
        }
    }

    #[test]
    fn the_context_strength_evidence_derivative_matches_refit_differences() {
        let (scores, context, weights) = patterned_sample(90);
        let (g, patterns) = pattern_statistics(&scores, context.view(), &weights).unwrap();
        let prior = ContextPrior::new(context.view(), &weights).unwrap().unwrap();
        let (mean, covariance) = initial_law(&patterns, g, 2).unwrap();
        let evidence_at = |rho: f64| {
            prior
                .penalized(&patterns, rho, mean.clone(), covariance.clone())
                .unwrap()
        };
        let rho = 0.4;
        let at = evidence_at(rho);
        // One evidence rounds by its likelihood account (with the prior's pattern) plus a
        // log-determinant whose pivots each carry the information's conditioning; the analytic
        // derivative's traces round the same way. The central difference adds its truncation
        // measured against the doubled step.
        let mut all = patterns.clone();
        all.push(prior.pattern(g, rho.exp()));
        let information = score_curvature(&all, &at.0, &at.1)
            .unwrap()
            .hessian
            .mapv(|v| -v);
        let evidence_rounding = likelihood_account(&all, &at.0, &at.1)
            + rounding(information.nrows(), 1.0, condition_of(&information));
        let (central, bound) = central_difference(
            |step| evidence_at(rho + step).2,
            central_step(rho),
            evidence_rounding,
        );
        assert!(
            (central - at.3).abs() <= bound + evidence_rounding,
            "d log Z / d rho: analytic {} vs central {central}, bound {:.3e}",
            at.3,
            bound + evidence_rounding
        );
    }

    fn paired_sample(effect: f64) -> (Vec<Vec<Option<f64>>>, Array2<f64>, Vec<f64>) {
        let pairs = 30;
        let mut context = Array2::<f64>::zeros((2 * pairs, 2));
        let mut scores = Vec::with_capacity(2 * pairs);
        for i in 0..pairs {
            let x = (i as f64 + 0.5) / pairs as f64;
            let e1 = (i as f64 * 0.754_877_666_246_692_7).fract() - 0.5;
            let e2 = (i as f64 * 0.569_840_290_998_053_2).fract() - 0.5;
            for (row, sign) in [(2 * i, 1.0), (2 * i + 1, -1.0)] {
                context[[row, 0]] = 1.0;
                context[[row, 1]] = sign * x;
                scores.push(vec![
                    Some(0.3 + effect * sign * x + e1),
                    (i % 3 != 0).then_some(-0.1 + 0.5 * e1 + e2),
                ]);
            }
        }
        (scores, context, vec![1.0; 2 * pairs])
    }

    #[test]
    fn a_context_free_score_sample_recovers_the_intercept_only_law_and_a_driven_one_keeps_its_effect() {
        // Rows pair +x with -x at identical scores, so the unpenalized context effect is
        // zero to roundoff: its evidence is maximized in the no-context limit.
        let (scores, context, weights) = paired_sample(0.0);
        let fitted = GeneticScoreLaw::fit(&scores, context.view(), &weights).unwrap();
        let law = &fitted.law;
        assert!(
            matches!(fitted.context, ContextDecision::NoContext { .. }),
            "zero-effect decision {:?}",
            fitted.context
        );
        assert_eq!(law.context_columns, 2);
        assert_eq!(law.mean_coefficients[1], 0.0);
        assert_eq!(law.mean_coefficients[3], 0.0);
        // Control: an effect far above the residual spread keeps a context effect, shrunk
        // toward zero from its unpenalized maximum.
        let (scores, context, weights) = paired_sample(1.5);
        // The decision's pieces at the no-context point. The score statistic in the prior's
        // metric, T = s'Qs - tr(Q I_cc) with I_cc the A-block information, is positive iff the
        // evidence falls to the no-context limit from an interior maximum (at leading order in
        // the context block); then the rho-score read where `learn` reads it is negative, and
        // the evidence at the moment strength dim kappa^2 / T exceeds the limit (the isotropic
        // value -dim(log W' - W' + 1)/2 > 0 with W' = s'Qs / tr(Q I_cc) > 1).
        let (g, patterns) = pattern_statistics(&scores, context.view(), &weights).unwrap();
        let prior = ContextPrior::new(context.view(), &weights).unwrap().unwrap();
        let reduced = ContextPrior::reduce(&patterns);
        let (null_mean, null_covariance) = ContextPrior::null_law(&reduced, g, 2).unwrap();
        let limit = prior.limit_evidence(&reduced, g).unwrap().0;
        let at_null = score_curvature(&patterns, &null_mean, &null_covariance).unwrap();
        let information = at_null.hessian.mapv(|v| -v);
        let context_index: Vec<usize> = (0..g).map(|j| j * 2 + 1).collect();
        let efficient = submatrix(&information, &context_index, &context_index);
        let unit = Array2::from_shape_fn((g, g), |(j, l)| null_covariance[[j, l]] / prior.gram[[1, 1]]);
        // Negative control (the defect this guard exists for): the old scale took the Schur
        // complement of the observed information at the no-context point, whose cross blocks
        // carry E = sum w r c'. At this driven fixture that complement is not positive definite,
        // and the typed guard must refuse it rather than return the no-context law.
        let nuisance_index: Vec<usize> = (0..information.nrows())
            .filter(|i| !context_index.contains(i))
            .collect();
        let cross = submatrix(&information, &nuisance_index, &context_index);
        let nuisance_factor = lower_factor(
            &submatrix(&information, &nuisance_index, &nuisance_index),
            "the no-context information",
        )
        .unwrap();
        let schur = &efficient - &cross.t().dot(&cholesky_solve_matrix(&nuisance_factor, &cross));
        let schur_kappa = (&schur * &unit).sum() / g as f64;
        assert!(
            !(schur_kappa > 0.0),
            "the Schur scale {schur_kappa} is positive at this fixture, so the control is vacuous"
        );
        assert!(
            matches!(context_scale(schur_kappa), Err(EventHistoryError::Fit { .. })),
            "the Schur scale {schur_kappa} did not hit the typed refusal"
        );
        // The decision's pieces: this driven effect's family boundary score exceeds its bound,
        // and the evidence at the start strength dim kappa^2 / (2 S) exceeds the no-context limit
        // (the isotropic value -dim(log W' - W' + 1)/2 > 0 at leading order).
        let boundary = prior.boundary_score(&patterns, &reduced, g, 2).unwrap();
        eprintln!(
            "driven 2-score boundary: fixed-nuisance score test T/2 {:.6e}, family boundary score S {:.6e} ± {:.2e}, kappa {:.4e}",
            boundary.score_test,
            boundary.value,
            boundary.bound,
            boundary.kappa
        );
        assert!(
            boundary.value > boundary.bound,
            "the driven effect's family boundary score {} does not exceed its bound {}",
            boundary.value,
            boundary.bound
        );
        let start = (g as f64 * boundary.kappa * boundary.kappa / (2.0 * boundary.value)).ln();
        let at_start = prior
            .penalized(&patterns, start, null_mean.clone(), null_covariance.clone())
            .map_err(|error| format!("penalized refit at the start rho {start}: {error}"))
            .unwrap();
        assert!(
            at_start.2 > limit,
            "evidence at the start {} vs the no-context limit {limit} (family score {}, rho-score there {})",
            at_start.2,
            boundary.value,
            at_start.3
        );
        let fitted = GeneticScoreLaw::fit(&scores, context.view(), &weights).unwrap();
        let law = &fitted.law;
        assert!(
            matches!(fitted.context, ContextDecision::Optimum { .. }),
            "driven 2-score decision {:?}",
            fitted.context
        );
        let unpenalized = maximum_likelihood(&scores, context.view(), &weights);
        assert!(
            law.mean_coefficients[1] > 0.0
                && law.mean_coefficients[1] < unpenalized.mean_coefficients[1],
            "the driven context effect {:?} vs unpenalized {:?}",
            law.mean_coefficients,
            unpenalized.mean_coefficients
        );
    }

    /// One complete score on a +x/-x paired context whose sample correlation with x is
    /// `r_squared` exactly: the pairs share their residual, so `S_xy = beta S_xx` and
    /// `S_yy = beta^2 S_xx + S_ee`.
    /// The single-score family's closed forms `S = n((n - 3) r^2 - 1)/2` and
    /// `T/2 = n((n + 1) r^2 - 1)/2`, from the sample's rows over the running-error tracker.
    fn boundary_oracles(
        scores: &[Vec<Option<f64>>],
        context: &Array2<f64>,
    ) -> (crate::test_support::Bound, crate::test_support::Bound) {
        use crate::scalar::recip;
        use crate::test_support::Bound;
        let rows = scores.len() as f64;
        let n = Bound::exact(rows);
        let inverse_n = recip(&n);
        let response = |i: usize| Bound::exact(scores[i][0].unwrap_or(f64::NAN));
        let covariate = |i: usize| Bound::exact(context[[i, 1]]);
        let (mut y_sum, mut x_sum) = (Bound::exact(0.0), Bound::exact(0.0));
        for i in 0..scores.len() {
            y_sum = y_sum.add(&response(i));
            x_sum = x_sum.add(&covariate(i));
        }
        let (y_mean, x_mean) = (y_sum.mul(&inverse_n), x_sum.mul(&inverse_n));
        let (mut s_xx, mut s_xy, mut s_yy) = (Bound::exact(0.0), Bound::exact(0.0), Bound::exact(0.0));
        for i in 0..scores.len() {
            let dx = covariate(i).sub(&x_mean);
            let dy = response(i).sub(&y_mean);
            s_xx = s_xx.add(&dx.mul(&dx));
            s_xy = s_xy.add(&dx.mul(&dy));
            s_yy = s_yy.add(&dy.mul(&dy));
        }
        let r_squared = s_xy.mul(&s_xy).mul(&recip(&s_xx.mul(&s_yy)));
        let closed = |offset: f64| {
            n.mul(&r_squared.scale(rows + offset).sub(&Bound::exact(1.0)))
                .scale(0.5)
        };
        (closed(-3.0), closed(1.0))
    }

    fn single_score_sample(r_squared: f64) -> (Vec<Vec<Option<f64>>>, Array2<f64>, Vec<f64>) {
        let pairs = 30;
        let xs: Vec<f64> = (0..pairs).map(|i| (i as f64 + 0.5) / pairs as f64).collect();
        let es: Vec<f64> = (0..pairs)
            .map(|i| (i as f64 * 0.754_877_666_246_692_7).fract() - 0.5)
            .collect();
        let e_mean = es.iter().sum::<f64>() / pairs as f64;
        let s_ee = 2.0 * es.iter().map(|e| (e - e_mean).powi(2)).sum::<f64>();
        let s_xx = 2.0 * xs.iter().map(|x| x * x).sum::<f64>();
        let beta = (r_squared * s_ee / ((1.0 - r_squared) * s_xx)).sqrt();
        let mut context = Array2::<f64>::zeros((2 * pairs, 2));
        let mut scores = Vec::with_capacity(2 * pairs);
        for i in 0..pairs {
            for (row, sign) in [(2 * i, 1.0), (2 * i + 1, -1.0)] {
                context[[row, 0]] = 1.0;
                context[[row, 1]] = sign * xs[i];
                scores.push(vec![Some(0.3 + beta * sign * xs[i] + es[i])]);
            }
        }
        (scores, context, vec![1.0; 2 * pairs])
    }

    #[test]
    fn a_weak_context_effect_above_its_dimension_returns_the_no_context_law_and_a_driven_one_selects_context() {
        // On one complete score with a centered context column the joint Laplace family is
        // closed form: with u = 1 / (1 + lambda / n) and r^2 the sample correlation,
        //     log Z(lambda) - log Z_inf = -(n - 3)/2 log(1 - r^2 u) + log(1 - u) / 2,
        // so the evidence rises toward the no-context limit iff (n - 3) r^2 < 1, whatever the
        // Wald statistic W = n r^2 / (1 - r^2) says.
        let n = 60.0;
        let oracle = |scores: &[Vec<Option<f64>>], context: &Array2<f64>| {
            let rows = scores.len();
            let y: Vec<f64> = scores.iter().map(|row| row[0].unwrap_or(f64::NAN)).collect();
            let y_mean = y.iter().sum::<f64>() / rows as f64;
            let x_mean = (0..rows).map(|i| context[[i, 1]]).sum::<f64>() / rows as f64;
            let (mut s_xx, mut s_xy, mut s_yy) = (0.0, 0.0, 0.0);
            for (i, value) in y.iter().enumerate() {
                let (dx, dy) = (context[[i, 1]] - x_mean, value - y_mean);
                s_xx += dx * dx;
                s_xy += dx * dy;
                s_yy += dy * dy;
            }
            [
                s_xy * s_xy / (s_xx * s_yy),
                s_xy / s_xx,
                y_mean,
                s_yy / rows as f64,
                s_xx,
                s_yy,
                y.iter().map(|value| value * value).sum::<f64>(),
            ]
        };
        // First-order rounding account of one evidence on this design: eps times the products
        // it forms (n accumulated rows plus the d^2 log-determinant products) times the
        // magnitudes of its summands (the n-weighted log terms, the residual trace from the
        // uncentered score moment, the log-determinant and the prior's terms), amplified by
        // the conditioning of its diagonal information.
        let account = |sigma_squared: f64, informations: &[f64], extra: f64, s_gg: f64| -> f64 {
            let log_tau = TAU.ln();
            let magnitude = 0.5 * n * (log_tau + sigma_squared.ln().abs())
                + 0.5 * s_gg / sigma_squared
                + 0.5 * informations.len() as f64 * log_tau
                + 0.5 * informations.iter().map(|i| i.ln().abs()).sum::<f64>()
                + extra;
            let largest = informations.iter().copied().fold(0.0_f64, f64::max);
            let smallest = informations.iter().copied().fold(f64::INFINITY, f64::min);
            f64::EPSILON
                * (n + (informations.len() * informations.len()) as f64)
                * (largest / smallest)
                * magnitude
        };

        // Weak: r^2 halfway between the score threshold 1/n and the family's 1/(n - 3).
        let weak = 0.5 * (1.0 / n + 1.0 / (n - 3.0));
        let (scores, context, weights) = single_score_sample(weak);
        let weak_statistics = oracle(&scores, &context);
        let (r_squared, intercept, variance) =
            (weak_statistics[0], weak_statistics[2], weak_statistics[3]);
        let (s_xx, s_yy, s_gg) = (weak_statistics[4], weak_statistics[5], weak_statistics[6]);
        assert!(n * r_squared / (1.0 - r_squared) > 1.0, "the weak effect must sit above its dimension");
        assert!((n - 3.0) * r_squared < 1.0);
        let (g, patterns) = pattern_statistics(&scores, context.view(), &weights).unwrap();
        let prior = ContextPrior::new(context.view(), &weights).unwrap().unwrap();
        let limit = prior.limit_evidence(&ContextPrior::reduce(&patterns), g).unwrap().0;
        let (start_mean, start_covariance) =
            ContextPrior::null_law(&ContextPrior::reduce(&patterns), g, 2).unwrap();
        // Control that the rule reads the family's evidence, not the fixed-nuisance score test:
        // in this band T/2 = n((n + 1) r^2 - 1)/2 > 0 at the limit mode while the family's exact
        // boundary score S = n((n - 3) r^2 - 1)/2 < 0, and the fit returns the no-context law.
        // The closed form's rounding is n-row sums times its n-scaled factors.
        let boundary = prior
            .boundary_score(&patterns, &ContextPrior::reduce(&patterns), g, 2)
            .unwrap();
        // The production band certified against a Bound oracle of the same score: the closed
        // forms from the sample's rows over the running-error tracker.
        let (score_oracle, score_test_oracle) = boundary_oracles(&scores, &context);
        let score_bar = boundary.bound + score_oracle.rounding();
        eprintln!(
            "disagreement band: fixed-nuisance score test T/2 {:.6e}, family boundary score S {:.6e} ± {:.2e}, closed form {:.6e} ± {:.2e}",
            boundary.score_test,
            boundary.value,
            boundary.bound,
            score_oracle.value,
            score_oracle.rounding()
        );
        assert!(
            (boundary.value - score_oracle.value).abs() <= score_bar,
            "family boundary score {} vs closed form {}, bound {score_bar:.3e}",
            boundary.value,
            score_oracle.value
        );
        assert!(
            score_oracle.value.abs() > score_bar,
            "the closed form {} is not resolved beyond its bound {score_bar:.3e}",
            score_oracle.value
        );
        assert!(
            boundary.score_test > 0.0 && boundary.value < -boundary.bound,
            "the band needs T/2 {} > 0 > S {} beyond its bound {}",
            boundary.score_test,
            boundary.value,
            boundary.bound
        );
        // The fixed-nuisance score test's own route, against its closed form n((n + 1) r^2 - 1)/2;
        // its rounding is one part of the band's.
        let score_test_bar = boundary.bound + score_test_oracle.rounding();
        assert!(
            (boundary.score_test - score_test_oracle.value).abs() <= score_test_bar,
            "fixed-nuisance score test {} vs closed form {}, bound {score_test_bar:.3e}",
            boundary.score_test,
            score_test_oracle.value
        );
        for rho in [-1.0, 2.0, 5.0] {
            let evidence = prior
                .penalized(&patterns, rho, start_mean.clone(), start_covariance.clone())
                .unwrap()
                .2;
            let u = 1.0 / (1.0 + f64::exp(rho) / n);
            let expected = -0.5 * (n - 3.0) * (1.0 - r_squared * u).ln() + 0.5 * (1.0 - u).ln();
            // The evidence's mode has sigma^2 = s_yy (1 - r^2 u) / (n + 1) and diagonal
            // information (n / sigma^2, s_xx / (u sigma^2), (n + 1) / (2 sigma^4)); the limit's
            // has sigma^2 = s_yy / (n + 1) and (n / sigma^2, (n + 1) / (2 sigma^4)).
            let sigma_squared = s_yy * (1.0 - r_squared * u) / (n + 1.0);
            let evidence_rounding = account(
                sigma_squared,
                &[
                    n / sigma_squared,
                    s_xx / (u * sigma_squared),
                    (n + 1.0) / (2.0 * sigma_squared * sigma_squared),
                ],
                0.5 * (rho.abs() + (s_xx / n).ln().abs() + TAU.ln() + sigma_squared.ln().abs()),
                s_gg,
            );
            let limit_sigma = s_yy / (n + 1.0);
            let limit_rounding = account(
                limit_sigma,
                &[n / limit_sigma, (n + 1.0) / (2.0 * limit_sigma * limit_sigma)],
                TAU.ln() + limit_sigma.ln().abs(),
                s_gg,
            );
            // The closed form sums (n - 3)/2 log(1 - r^2 u) with r^2 from n-row sums.
            let closed_form_rounding = f64::EPSILON
                * n
                * ((n - 3.0) * (1.0 - r_squared * u).ln().abs() + (1.0 - u).ln().abs() + 1.0);
            let bar = evidence_rounding + limit_rounding + closed_form_rounding;
            assert!(
                (evidence - limit - expected).abs() <= bar,
                "rho {rho}: family evidence above the limit {} vs closed form {expected}, rounding bound {bar:.3e}",
                evidence - limit
            );
            assert!(evidence < limit, "the weak effect's evidence must stay below its limit");
        }
        let fitted = GeneticScoreLaw::fit(&scores, context.view(), &weights).unwrap();
        let law = &fitted.law;
        assert!(
            matches!(fitted.context, ContextDecision::NoContext { .. }),
            "disagreement-band decision {:?}",
            fitted.context
        );
        assert_eq!(law.mean_coefficients[1], 0.0);
        // The returned law is the certified intercept-only maximum: within |V| band of the
        // exact one, against the n-row sums' rounding.
        let reduced = ContextPrior::reduce(&patterns);
        let (null_mean, null_covariance) = ContextPrior::null_law(&reduced, g, 1).unwrap();
        let null_curvature = score_curvature(&reduced, &null_mean, &null_covariance).unwrap();
        let null_information = null_curvature.hessian.mapv(|v| -v);
        let null_posterior = cholesky_solve_matrix(
            &lower_factor(&null_information, "the intercept-only information").unwrap(),
            &Array2::<f64>::eye(null_information.nrows()),
        )
        .mapv(f64::abs);
        let null_bound = null_posterior.dot(&Array1::from(null_curvature.band.clone()));
        let sums_rounding = rounding(n as usize, s_gg + intercept.abs(), 1.0);
        assert!(
            (law.mean_coefficients[0] - intercept).abs() <= null_bound[0] + sums_rounding
        );
        assert!(
            (law.covariance_factor[0].powi(2) - variance).abs()
                <= null_bound[1] + sums_rounding + rounding(2, variance, 1.0)
        );

        // Driven: (n - 3) r^2 > 1, so an interior maximum above the limit exists and the fit
        // keeps a context effect shrunk below its least-squares slope.
        let (scores, context, weights) = single_score_sample(0.3);
        let driven_statistics = oracle(&scores, &context);
        let (r_squared, slope) = (driven_statistics[0], driven_statistics[1]);
        assert!((n - 3.0) * r_squared > 1.0);
        let (g, patterns) = pattern_statistics(&scores, context.view(), &weights).unwrap();
        let prior = ContextPrior::new(context.view(), &weights).unwrap().unwrap();
        let boundary = prior
            .boundary_score(&patterns, &ContextPrior::reduce(&patterns), g, 2)
            .unwrap();
        let (score_oracle, score_test_oracle) = boundary_oracles(&scores, &context);
        let score_bar = boundary.bound + score_oracle.rounding();
        assert!(
            (boundary.value - score_oracle.value).abs() <= score_bar,
            "driven family boundary score {} vs closed form {}, bound {score_bar:.3e}",
            boundary.value,
            score_oracle.value
        );
        assert!(
            score_oracle.value.abs() > score_bar,
            "the driven closed form {} is not resolved beyond its bound {score_bar:.3e}",
            score_oracle.value
        );
        let score_test_bar = boundary.bound + score_test_oracle.rounding();
        assert!(
            (boundary.score_test - score_test_oracle.value).abs() <= score_test_bar,
            "driven fixed-nuisance score test {} vs closed form {}, bound {score_test_bar:.3e}",
            boundary.score_test,
            score_test_oracle.value
        );
        assert!(boundary.value > boundary.bound);
        let fitted = GeneticScoreLaw::fit(&scores, context.view(), &weights).unwrap();
        let law = &fitted.law;
        assert!(
            matches!(fitted.context, ContextDecision::Optimum { .. }),
            "driven single-score decision {:?}",
            fitted.context
        );
        assert!(
            law.mean_coefficients[1] > 0.0 && law.mean_coefficients[1] < slope,
            "driven context coefficient {} vs least-squares slope {slope}",
            law.mean_coefficients[1]
        );
    }

    #[test]
    fn a_boundary_score_within_its_bound_returns_the_no_context_law_as_a_reported_tie() {
        // On one complete score the family's boundary score is n((n - 3) r^2 - 1)/2, zero at
        // r^2 = 1/(n - 3): a constructed first-order tie. The sample's own r^2 differs from it
        // only by the representation of its rows, far inside the score's rounding account.
        let n = 60.0;
        let (scores, context, weights) = single_score_sample(1.0 / (n - 3.0));
        let (g, patterns) = pattern_statistics(&scores, context.view(), &weights).unwrap();
        let prior = ContextPrior::new(context.view(), &weights).unwrap().unwrap();
        let boundary = prior
            .boundary_score(&patterns, &ContextPrior::reduce(&patterns), g, 2)
            .unwrap();
        // The premise: the sample's closed form is zero within its own rounding.
        let (score_oracle, score_test_oracle) = boundary_oracles(&scores, &context);
        assert!(
            score_oracle.value.abs() <= score_oracle.rounding(),
            "the constructed sample's closed form {} is not a tie within its rounding {:.3e}",
            score_oracle.value,
            score_oracle.rounding()
        );
        assert!(
            score_test_oracle.value > score_test_oracle.rounding(),
            "at the tie the fixed-nuisance score test {} must still be positive",
            score_test_oracle.value
        );
        assert!(
            boundary.value.abs() <= boundary.bound,
            "constructed tie: boundary score {} vs its bound {}",
            boundary.value,
            boundary.bound
        );
        let fitted = GeneticScoreLaw::fit(&scores, context.view(), &weights).unwrap();
        assert!(
            matches!(fitted.context, ContextDecision::FirstOrderTie { .. }),
            "tie decision {:?}",
            fitted.context
        );
        assert_eq!(fitted.law.mean_coefficients[1], 0.0);
    }

    #[test]
    fn the_evidence_decision_classifies_an_exact_difference_against_its_band() {
        // Exactly representable inputs: limit evidence 2 and both errors 2^-32, so the band is
        // B = 2^-31 + eps |D|. The driven single-score and 2-score fits reach this classifier on the
        // clear Optimum side.
        let error = 2.0_f64.powi(-32);
        let beyond = 2.0_f64.powi(-30);
        // D = 2^-32 lies inside B: a tie. Dropping the tie branch fails here, and so does removing the
        // band (D > B replaced by D > 0), which classifies this difference as Optimum.
        let tie = classify_evidence(1.5, 2.0 + error, error, 2.0, error);
        assert!(matches!(tie, ContextDecision::EvidenceTie { .. }), "inside the band: {tie:?}");
        // D = 2^-30 lies beyond B: Optimum. Flipping the upper inequality fails here.
        let optimum = classify_evidence(1.5, 2.0 + beyond, error, 2.0, error);
        assert!(matches!(optimum, ContextDecision::Optimum { .. }), "above the band: {optimum:?}");
        // D = -2^-30 lies below -B: LimitDominates. Flipping the lower inequality fails here.
        let dominated = classify_evidence(1.5, 2.0 - beyond, error, 2.0, error);
        assert!(
            matches!(dominated, ContextDecision::LimitDominates { .. }),
            "below the band: {dominated:?}"
        );
    }

    #[test]
    fn a_running_scalar_seeded_from_an_inline_error_reports_that_error() {
        // mu = error / eps scales by 2^52 and rounding() = eps mu by 2^-52: power-of-two scalings of
        // normal values, both exact, so the error reads back bit for bit. Seeding mu with the error
        // itself misses by 2^52, and a near-identity seeding such as mu (1 + eps) changes the bits.
        for error in [3.0e-16, 1.25e-9, 0.5] {
            let seeded = running_with_error(1.75, error);
            assert_eq!(seeded.value, 1.75);
            assert_eq!(
                seeded.rounding().to_bits(),
                error.to_bits(),
                "seeded error {error:e} reads back as {:e}",
                seeded.rounding()
            );
        }
    }

    #[test]
    fn a_running_cholesky_refuses_a_pivot_inside_its_own_rounding() {
        let zero = Running::exact(0.0);
        // A pivot resolved beyond its rounding factors.
        let resolved = Array2::from_elem((1, 1), running_with_error(4.0, 1.0e-12));
        let factor = cholesky_resolved(Running::rounding, &zero, &resolved, "a resolved block").unwrap();
        assert_eq!(factor[[0, 0]].value, 2.0);
        // The second pivot, 1e-10 with a declared error of 1e-8, is not positive beyond its own
        // rounding: the factor refuses and names it.
        let unresolved = Array2::from_shape_fn((2, 2), |(i, j)| match (i, j) {
            (0, 0) => running_with_error(1.0, 0.0),
            (1, 1) => running_with_error(1.0e-10, 1.0e-8),
            _ => running_with_error(0.0, 0.0),
        });
        let refusal = cholesky_resolved(Running::rounding, &zero, &unresolved, "an unresolved block");
        assert!(
            matches!(&refusal, Err(EventHistoryError::NumericalFailure { reason }) if reason.contains("pivot 1")),
            "unresolved pivot: {refusal:?}"
        );
        // The same values at f64 carry no rounding and factor: the refusal is the pivot's rounding.
        let values = unresolved.mapv(|entry| entry.value);
        assert!(cholesky_over(&0.0, &values, "the same block at f64").is_ok());
    }

    #[test]
    fn the_conditional_law_over_running_is_certified_by_a_covariance_route_bound_oracle() {
        use crate::joint::law::numerical::Running;
        use crate::scalar::recip;
        use crate::test_support::Bound;
        let law = law();
        let context = [1.0, 0.6];
        let observed = [Some(0.8), None, None];
        let point = [0.1, -0.7];
        let production = law.conditional(&context, &observed).unwrap();
        let running = law.conditional_over(&Running::exact(0.0), &context, &observed).unwrap();

        // An independent route over the tracker: the covariance route with the observed score's
        // 1x1 pivot, C = Sigma_mm - Sigma_m0 Sigma_0m / Sigma_00,
        // E[g_m | g_0] = mu_m + Sigma_m0 (g_0 - mu_0) / Sigma_00, Lambda_mm = C^-1 by cofactors and
        // log det Lambda_mm = -ln det C.
        let exact = Bound::exact;
        let mean: Vec<Bound> = (0..3)
            .map(|j| {
                (0..2).fold(exact(0.0), |sum, b| {
                    sum.add(&exact(law.mean_coefficients[2 * j + b]).mul(&exact(context[b])))
                })
            })
            .collect();
        let sigma = |i: usize, j: usize| {
            (0..3).fold(exact(0.0), |sum, k| {
                sum.add(&exact(law.covariance_factor[3 * i + k]).mul(&exact(law.covariance_factor[3 * j + k])))
            })
        };
        let pivot = recip(&sigma(0, 0));
        let residual = exact(0.8).sub(&mean[0]);
        let conditional_mean = [1, 2].map(|i| mean[i].add(&sigma(i, 0).mul(&residual).mul(&pivot)));
        let c = |a: usize, b: usize| sigma(a, b).sub(&sigma(a, 0).mul(&sigma(0, b)).mul(&pivot));
        let determinant = c(1, 1).mul(&c(2, 2)).sub(&c(1, 2).mul(&c(2, 1)));
        let inverse_determinant = recip(&determinant);
        let precision = [
            [c(2, 2).mul(&inverse_determinant), c(1, 2).neg().mul(&inverse_determinant)],
            [c(2, 1).neg().mul(&inverse_determinant), c(1, 1).mul(&inverse_determinant)],
        ];
        let log_determinant = ln(&determinant).neg();
        let deviation = [
            exact(point[0]).sub(&conditional_mean[0]),
            exact(point[1]).sub(&conditional_mean[1]),
        ];
        let quadratic = (0..2).fold(exact(0.0), |sum, a| {
            (0..2).fold(sum, |sum, b| sum.add(&deviation[a].mul(&precision[a][b]).mul(&deviation[b])))
        });
        let density = log_determinant
            .sub(&ln(&exact(TAU)).scale(2.0))
            .sub(&quadratic)
            .scale(0.5);

        // Each production value agrees with the oracle within the production route's own Running
        // bound plus the oracle's, and the oracle is resolved beyond that bar.
        let agree = |name: &str, value: f64, bound: f64, oracle: &Bound| {
            let bar = bound + oracle.rounding();
            assert!(
                oracle.value.abs() > bar,
                "{name}: oracle {} is below its bar {bar:.3e}",
                oracle.value
            );
            assert!(
                (value - oracle.value).abs() <= bar,
                "{name}: production {value}, oracle {}, bar {bar:.3e}",
                oracle.value
            );
        };
        for a in 0..2 {
            agree("conditional mean", production.mean[a], running.mean[a].rounding(), &conditional_mean[a]);
            for b in 0..2 {
                agree(
                    "conditional precision",
                    production.precision[[a, b]],
                    running.precision[[a, b]].rounding(),
                    &precision[a][b],
                );
            }
        }
        agree(
            "conditional log determinant",
            production.log_determinant,
            running.log_determinant.rounding(),
            &log_determinant,
        );
        let production_density = production.log_density(&point).unwrap();
        let running_density = running.log_density(&point.map(Running::exact)).unwrap();
        agree("conditional log density", production_density, running_density.rounding(), &density);
        // The f64 instance and the Running instance are one route: identical values.
        assert_eq!(running_density.value, production_density);
    }

    #[test]
    fn the_boundary_band_covers_the_certified_limit_modes_first_order_error() {
        // The band charges the certified limit mode's error through the context score only,
        // |w|'|G_n| e with the mode box e = |V_n| (|s^_n| + band_n). This measures the full
        // first-order change of S over that box, sum_k |dS/dtheta_k| e_k, by central differences
        // at shifted limit modes with their Richardson truncation and the band as each
        // evaluation's rounding, and asserts the channel covers it.
        let n = 60.0;
        for r_squared in [0.5 * (1.0 / n + 1.0 / (n - 3.0)), 0.3] {
            let (scores, context, weights) = single_score_sample(r_squared);
            let (g, patterns) = pattern_statistics(&scores, context.view(), &weights).unwrap();
            let prior = ContextPrior::new(context.view(), &weights).unwrap().unwrap();
            let (limit, limit_mean, covariance) =
                prior.limit_mode(&ContextPrior::reduce(&patterns), g).unwrap();
            let at_mode = prior
                .boundary_score_at(&patterns, &limit, &limit_mean, &covariance, 2)
                .unwrap();
            let limit_curvature = score_curvature(&limit, &limit_mean, &covariance).unwrap();
            let posterior = inverse_with_error(
                &limit_curvature.hessian.mapv(|v| -v),
                &limit_curvature.hessian_error,
                "the no-context limit's information",
            )
            .unwrap()
            .0;
            let mode_box = posterior.mapv(f64::abs).dot(&Array1::from_iter(
                limit_curvature
                    .gradient
                    .iter()
                    .zip(&limit_curvature.band)
                    .map(|(score, band)| score.abs() + band),
            ));
            // The limit's coordinates: the intercepts, then vech Sigma row-major.
            let theta: Vec<f64> = (0..g)
                .map(|j| limit_mean[[j, 0]])
                .chain(
                    (0..g)
                        .flat_map(|high| (0..=high).map(move |low| (high, low)))
                        .map(|(high, low)| covariance[[high, low]]),
                )
                .collect();
            let shifted = |k: usize, step: f64| -> f64 {
                let mut mean = limit_mean.clone();
                let mut shifted_covariance = covariance.clone();
                if k < g {
                    mean[[k, 0]] += step;
                } else {
                    for high in 0..g {
                        for low in 0..=high {
                            if covariance_index(g, 1, high, low) == k {
                                shifted_covariance[[high, low]] += step;
                                if high != low {
                                    shifted_covariance[[low, high]] += step;
                                }
                            }
                        }
                    }
                }
                prior
                    .boundary_score_at(&patterns, &limit, &mean, &shifted_covariance, 2)
                    .unwrap()
                    .value
            };
            let mut first_order = 0.0;
            for k in 0..theta.len() {
                let (central, truncation) =
                    central_difference(|step| shifted(k, step), central_step(theta[k]), at_mode.bound);
                first_order += (central.abs() + truncation) * mode_box[k];
            }
            eprintln!(
                "limit-mode box at r^2 {r_squared:.4e}: full first-order change {first_order:.3e}, band's mode channel {:.3e}, band {:.3e}",
                at_mode.mode_channel,
                at_mode.bound
            );
            assert!(
                first_order <= at_mode.mode_channel,
                "the limit mode's first-order change {first_order:.3e} exceeds the band's mode channel {:.3e}",
                at_mode.mode_channel
            );
        }
    }

    #[test]
    fn transport_inverts_whitening_on_complete_rows() {
        let law = law();
        let (condition, magnitude) = law_scale(&law);
        let transport_bar = rounding(
            ROUTE_TERMS,
            magnitude * (1.0 + largest_abs(&law.mean_coefficients) + 1.2).powi(2),
            condition,
        );
        let context =
            Array2::from_shape_vec((3, 2), vec![1.0, 0.25, 1.0, -0.8, 1.0, 0.1]).unwrap();
        let scores = vec![
            vec![Some(0.4), Some(-1.1), Some(0.9)],
            vec![Some(-0.3), None, Some(0.2)],
            vec![Some(1.2), Some(0.5), Some(-0.6)],
        ];
        let residuals = law.whitened_residuals(&scores, context.view()).unwrap();
        assert_eq!(residuals.len(), 2);
        let mut genome = [0.0; 3];
        for (residual, row) in residuals.iter().zip([0usize, 2]) {
            law.transport(&context.row(row).to_vec(), residual, &mut genome)
                .unwrap();
            for j in 0..3 {
                assert!((genome[j] - scores[row][j].unwrap_or(f64::NAN)).abs() <= transport_bar);
            }
        }
    }

    #[test]
    fn the_drive_is_affine_in_scores_and_coefficients_with_exact_split_and_gradient() {
        // Axis 1 omits B_1, its zero-effect model.
        let drive = DriveBlock::new(3, 2, 3, &[vec![true; 3], vec![true, false, true]]).unwrap();
        assert_eq!(drive.width(), 2 * 4 + 2 * 3);
        let functions = drive.functions();
        assert_eq!(functions.len(), 7);
        assert!(!functions.iter().any(|f| f.axis == 1 && f.interaction == 2));
        let mut covered = vec![false; 3 + drive.width()];
        for function in &functions {
            assert_eq!(function.coefficients.len(), 2);
            for index in function.coefficients.clone() {
                assert!(!covered[index], "functions overlap at {index}");
                covered[index] = true;
            }
        }
        assert!(covered[3..].iter().all(|c| *c) && covered[..3].iter().all(|c| !*c));

        let theta: Vec<f64> = (0..3 + drive.width()).map(|i| 0.1 * i as f64 - 0.7).collect();
        let columns = [1.0, 0.25];
        let genes = [0.5, -1.2, 0.8];
        let axis = 1;
        // A drive sums at most (G + 1) functions of `columns` products each, at the operands'
        // magnitudes.
        let drive_bar = rounding(
            functions.len() * columns.len(),
            largest_abs(&theta) * (1.0 + largest_abs(&genes)) * largest_abs(&columns),
            1.0,
        );
        let function = |interaction: usize| -> f64 {
            functions
                .iter()
                .find(|f| f.axis == axis && f.interaction == interaction)
                .map_or(0.0, |f| {
                    f.coefficients
                        .clone()
                        .zip(columns)
                        .map(|(i, feature)| theta[i] * feature)
                        .sum::<f64>()
                })
        };
        let expected = function(0) + (0..3).map(|s| function(s + 1) * genes[s]).sum::<f64>();
        let mean = drive.mean(&theta, &columns, &genes, axis).unwrap();
        assert!((mean - expected).abs() <= drive_bar);
        // The omitted B_1 reads no score: moving that score leaves the drive's bits.
        let moved = drive.mean(&theta, &columns, &[0.5, 7.0, 0.8], axis).unwrap();
        assert_eq!(moved, mean);

        let scores = law()
            .conditional(&[1.0, 0.0], &[Some(genes[0]), None, Some(genes[2])])
            .unwrap();
        let (offset, slopes) = drive.slopes(&theta, &columns, axis, &scores).unwrap();
        assert_eq!(slopes, vec![0.0]);
        assert!((offset - mean).abs() <= drive_bar);
        let (offset, slopes) = drive.slopes(&theta, &columns, 0, &scores).unwrap();
        let full = drive.mean(&theta, &columns, &genes, 0).unwrap();
        assert_eq!(slopes.len(), 1);
        assert!((offset + slopes[0] * genes[1] - full).abs() <= drive_bar);

        // The drive and its affine split are linear in the coefficients, so a
        // pullback contracted against the coefficients reproduces them (Euler).
        let mut adjoint = vec![0.0; theta.len()];
        drive
            .pullback(&columns, &genes, axis, &2.5, &mut adjoint)
            .unwrap();
        let reproduced: f64 = theta.iter().zip(&adjoint).map(|(t, a)| t * a).sum();
        assert!(
            (reproduced - 2.5 * mean).abs()
                <= 2.5 * drive_bar
                    + rounding(theta.len(), largest_abs(&theta) * largest_abs(&adjoint), 1.0)
        );
        // Only this axis's kept functions (u, B_0, B_2) receive adjoint.
        assert_eq!(adjoint.iter().filter(|a| **a != 0.0).count(), 3 * 2);
        let (offset, slopes) = drive.slopes(&theta, &columns, 0, &scores).unwrap();
        let mut forces = vec![0.0; theta.len()];
        drive
            .slopes_pullback(&columns, 0, &scores, 0.7, &[-1.3], &mut forces)
            .unwrap();
        let contracted: f64 = theta.iter().zip(&forces).map(|(t, f)| t * f).sum();
        assert!(
            (contracted - (0.7 * offset - 1.3 * slopes[0])).abs()
                <= 2.0 * drive_bar
                    + rounding(theta.len(), largest_abs(&theta) * largest_abs(&forces), 1.0)
        );
        assert!(drive.mean(&theta, &columns, &genes[..2], axis).is_err());
        assert!(DriveBlock::new(0, 2, 3, &[vec![true; 2]]).is_err());
    }
}
