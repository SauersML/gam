//! Score test of a random-effect variance component on the boundary.
//!
//! # The question this answers
//!
//! A `group(g)` term is `η = Xβ + X_R b` with `b ~ N(0, σ²_b Σ_b)` and a
//! ridge penalty on `b`. "Does this term matter?" is `H₀: σ²_b = 0`, and that
//! null sits on the BOUNDARY of the parameter space. None of the reference laws
//! the smooth table uses are valid there:
//!
//! * the coefficient Wald statistic `b̂ᵀV_b⁻b̂` is computed from a `b̂` that the
//!   penalty shrinks toward zero by an amount REML chose from the same data, so
//!   its `χ²_edf` reference is neither the right shape nor the right scale;
//! * the likelihood-ratio statistic has the chi-bar-square law
//!   `½χ²₀ + ½χ²₁` only for a SINGLE variance parameter with a scalar-shaped
//!   information, and even then its finite-sample law for a Gaussian model is a
//!   point mass at zero far larger than one half (Crainiceanu & Ruppert, 2004).
//!
//! # The statistic
//!
//! The score for `σ²_b` at `σ²_b = 0` (Lin, 1997) is, up to a constant,
//!
//! ```text
//! T = uᵀΣ_b u,        u = X̃_Rᵀ v,
//! ```
//!
//! where `v` is the working residual of the model WITHOUT the random effect and
//! `X̃_R = X_R − X_O A`, `A = (X_OᵀW_H X_O)⁻ X_OᵀW_H X_R`, is the random-effect
//! design with every OTHER column of the fit (intercept, linear terms, smooths,
//! other random effects) projected out in the fit's own curvature metric. The
//! variance component is the whole alternative, so a single scalar direction is
//! tested and no `b̂` enters — the statistic cannot be shrunk by the penalty it
//! is testing. The group penalties this crate builds are identity ridges, so
//! `Σ_b = I` and `T = ‖u‖²`.
//!
//! `v` is read off the full fit rather than a refit: with `s = W_F ⊙ (z − η̂)`
//! the fit's working score,
//!
//! ```text
//! v = s + W_H ⊙ (X_R b̂)
//! ```
//!
//! adds the random effect's own contribution back into the residual, and
//! projecting `X_O` out of `X_R` in the `W_H` metric makes `u` blind to where
//! `β̂_O` landed: `X̃_RᵀW_H X_O = 0`, so any shift of `β̂_O` — including the
//! shrinkage bias of every penalized smooth — leaves `u` unchanged. For a
//! Gaussian identity fit this is exact, `u = X̃_RᵀW y`, whatever `β̂` the
//! optimizer returned. For a GLM it is the one-step score at `b = 0`.
//!
//! # The reference law
//!
//! Under `H₀`, `u ~ N(0, φV)` with `V = X̃_RᵀW_F X̃_R`, so
//!
//! ```text
//! T/φ ~ Σ_j μ_j χ²₁,      μ = eig(V),
//! ```
//!
//! a positive weighted sum of independent `χ²₁` with the weights KNOWN from the
//! design. This is the exact finite-sample spectral reference of Wood (2013,
//! "A simple test for random effects in regression models", Biometrika 100),
//! evaluated here by inversion rather than approximated by moments. There is no
//! point mass at `p = 1`: `T > 0` almost surely.
//!
//! When the scale is estimated the statistic is the ratio `T/D'`, with `D'` the
//! weighted residual sum of squares of the UNPENALIZED full-model fit and
//! `ν = n⁺ − rank(X)` its degrees of freedom (`n⁺` the rows with positive
//! weight). `X̃_R` lies in `span(X)` and the unpenalized residual is `W`-orthogonal
//! to it, so for a Gaussian model `u` and `D'` are independent and
//! `D'/φ ~ χ²_ν`. The p-value is then
//!
//! ```text
//! P(T/D' > t) = P(Σ_j μ_j χ²₁ − t·χ²_ν > 0),
//! ```
//!
//! a signed weighted chi-square tail at zero — exact again, and exactly the
//! one-way ANOVA `F` on a balanced design. `D'` is the unpenalized residual and
//! not the fit's `φ̂` because `φ̂` is computed from a residual the penalty shrank,
//! whose law under `H₀` depends on the smoothing parameters REML chose.


use std::ops::Range;

use faer::Side;
use gam_linalg::faer_ndarray::strict_symmetric_eigh;
use gam_linalg::matrix::DesignMatrix;
use gam_math::probability::{WeightedChiSquareTerm, signed_weighted_chi_square_sf};
use ndarray::{Array1, Array2, ArrayView1, Axis, s};
use serde::{Deserialize, Serialize};

/// Rows streamed per chunk. The design is read twice per fit (once for the
/// Grams, once for every tested term together), never materialized whole.
const ROW_BLOCK: usize = 4096;

/// How the dispersion enters the reference law.
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum RandomEffectTestScale {
    /// `φ` is fixed by the family (`1` for binomial and Poisson), or pinned.
    Known { dispersion: f64 },
    /// `φ` was estimated from the data, so the statistic is a ratio against the
    /// unpenalized residual sum of squares.
    Estimated,
}

/// Why a random-effect term has no p-value.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum RandomEffectTestUnavailable {
    /// The fit did not retain the IRLS row state (weights, working response,
    /// linear predictor) the score needs.
    NoIrlsRowState,
    /// The design could not be read, or it or the row state held a non-finite
    /// value.
    DesignUnavailable,
    /// Every direction of the term lies inside the span of the model's other
    /// columns, so the data carry no information about it.
    NoEstimableDirection,
    /// The scale is estimated, and the unpenalized model leaves no residual
    /// degrees of freedom (or no residual variation) to estimate it from.
    NoResidualDegreesOfFreedom,
    /// The reference tail could not be resolved to any accuracy.
    TailUnresolved,
    /// The scale is known, but the fit publishes no finite positive dispersion
    /// to scale the score's variance by.
    KnownScaleUnavailable,
    /// The fitted mean composes the term's predictor with a link wiggle, so it
    /// is not linear in the term's coefficients and the linear score has no
    /// row state to read.
    MeanNotLinearInTerm,
}

impl RandomEffectTestUnavailable {
    /// Serialized label carried into the model payload and the Python surface.
    pub fn label(self) -> &'static str {
        match self {
            Self::NoIrlsRowState => "random_effect_no_irls_row_state",
            Self::DesignUnavailable => "random_effect_design_unavailable",
            Self::NoEstimableDirection => "random_effect_no_estimable_direction",
            Self::NoResidualDegreesOfFreedom => "random_effect_no_residual_degrees_of_freedom",
            Self::TailUnresolved => "random_effect_tail_unresolved",
            Self::KnownScaleUnavailable => "random_effect_known_scale_unavailable",
            Self::MeanNotLinearInTerm => "random_effect_mean_not_linear_in_term",
        }
    }

    /// One-line reason printed beside the term in a summary table.
    pub fn explanation(self) -> &'static str {
        match self {
            Self::NoIrlsRowState => {
                "the fit did not retain the IRLS row state the variance-component score test needs"
            }
            Self::DesignUnavailable => {
                "the design or IRLS row state could not be read as finite values"
            }
            Self::NoEstimableDirection => {
                "every direction of this term is spanned by the model's other terms"
            }
            Self::NoResidualDegreesOfFreedom => {
                "the scale is estimated but the unpenalized model leaves no residual degrees of freedom"
            }
            Self::TailUnresolved => "the reference tail probability could not be resolved",
            Self::KnownScaleUnavailable => {
                "the scale is known but the fit publishes no finite positive dispersion"
            }
            Self::MeanNotLinearInTerm => {
                "the mean composes this term with a link wiggle, so it is not linear in the term's coefficients"
            }
        }
    }
}

/// A computed random-effect test.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct RandomEffectTest {
    /// Reported on a chi-square-like scale with mean `reference_df` under `H₀`:
    /// `(T/φ̂)·reference_df/Σμ`, with `φ̂ = φ` for a known scale and `D'/ν` for
    /// an estimated one.
    pub statistic: f64,
    /// The effective degrees of freedom `(Σμ)²/Σμ²` of the spectral reference —
    /// `rank` exactly when the design is balanced.
    pub reference_df: f64,
    /// Number of estimable directions of the term (eigenvalues of `V` above its
    /// rounding floor).
    pub rank: usize,
    /// `ν`, when the scale is estimated.
    pub residual_df: Option<f64>,
    /// The exact tail of the statistic's reference law (see the module docs).
    pub p_value: f64,
    /// Bound on `|p_value − P|/P` from the tail evaluation; `0` for a closed form.
    pub p_value_relative_error: f64,
}

/// Outcome for one term: a test, or a typed reason there is none.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case", tag = "status")]
pub enum RandomEffectTestOutcome {
    Tested(RandomEffectTest),
    Unavailable { reason: RandomEffectTestUnavailable },
}

impl From<Result<RandomEffectTest, RandomEffectTestUnavailable>> for RandomEffectTestOutcome {
    fn from(result: Result<RandomEffectTest, RandomEffectTestUnavailable>) -> Self {
        match result {
            Ok(test) => Self::Tested(test),
            Err(reason) => Self::Unavailable { reason },
        }
    }
}

/// One random-effect term's test, keyed the way the summary walks the design.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct RandomEffectTestRecord {
    /// Random-effect term name (matches the summary row).
    pub term: String,
    /// The term's GLOBAL coefficient range.
    pub coefficient_range: Range<usize>,
    pub outcome: RandomEffectTestOutcome,
}

/// One term to test.
#[derive(Clone, Debug)]
pub struct RandomEffectTermRequest {
    /// GLOBAL coefficient range of the term's block.
    pub range: Range<usize>,
}

/// The fit's row state, in the fit's own row and coefficient layout.
pub struct RandomEffectTestInput<'a> {
    pub design: &'a DesignMatrix,
    /// `β̂`, full coefficient vector.
    pub beta: ArrayView1<'a, f64>,
    /// `W_H` — curvature weights the penalized Hessian was assembled from. Used
    /// only for the projection.
    pub hessian_weights: ArrayView1<'a, f64>,
    /// `W_F` — Fisher/score weights, `Var(s) = φ·W_F`.
    pub score_weights: ArrayView1<'a, f64>,
    /// `s = W_F ⊙ (z − η̂)` — the working score the fit solved to stationarity.
    pub score: ArrayView1<'a, f64>,
    pub scale: RandomEffectTestScale,
}

/// Per-fit quantities shared by every tested term: the design Gram in the `W_H`
/// metric and, for an estimated scale, the unpenalized residual.
pub struct RandomEffectTestBasis<'a> {
    input: RandomEffectTestInput<'a>,
    hessian_gram: Array2<f64>,
    scale: ResolvedScale,
}

#[derive(Clone, Copy)]
enum ResolvedScale {
    Known(f64),
    Estimated {
        residual_sum_of_squares: f64,
        residual_df: f64,
    },
}

impl<'a> RandomEffectTestBasis<'a> {
    /// First pass over the design: `G_H = XᵀW_H X`, and for an estimated scale
    /// `G_F = XᵀW_F X`, `Xᵀs` and `Σ s²/W_F`, from which
    /// `D' = Σ s²/W_F − (Xᵀs)ᵀG_F⁻(Xᵀs)` is the weighted residual sum of squares
    /// of the unpenalized fit: its residual is `e − X G_F⁻Xᵀs` with
    /// `e = z − η̂`, since the unpenalized solution is one Newton step from `β̂`.
    pub fn new(input: RandomEffectTestInput<'a>) -> Result<Self, RandomEffectTestUnavailable> {
        let n = input.design.nrows();
        let p = input.design.ncols();
        if n == 0
            || p == 0
            || input.beta.len() != p
            || input.hessian_weights.len() != n
            || input.score_weights.len() != n
            || input.score.len() != n
        {
            return Err(RandomEffectTestUnavailable::DesignUnavailable);
        }
        let rows_finite = input.beta.iter().all(|v| v.is_finite())
            && input.hessian_weights.iter().all(|v| v.is_finite())
            && input.score_weights.iter().all(|v| v.is_finite() && *v >= 0.0)
            && input.score.iter().all(|v| v.is_finite());
        if !rows_finite {
            return Err(RandomEffectTestUnavailable::DesignUnavailable);
        }
        let estimated = matches!(input.scale, RandomEffectTestScale::Estimated);
        if let RandomEffectTestScale::Known { dispersion } = input.scale
            && !(dispersion.is_finite() && dispersion > 0.0)
        {
            return Err(RandomEffectTestUnavailable::DesignUnavailable);
        }

        let mut hessian_gram = Array2::<f64>::zeros((p, p));
        let mut fisher_gram = Array2::<f64>::zeros(if estimated { (p, p) } else { (0, 0) });
        let mut design_score = Array1::<f64>::zeros(p);
        let mut score_energy = 0.0_f64;
        let mut positive_rows = 0usize;
        let mut start = 0usize;
        while start < n {
            let stop = (start + ROW_BLOCK).min(n);
            let block = input
                .design
                .try_row_chunk(start..stop)
                .map_err(|_| RandomEffectTestUnavailable::DesignUnavailable)?;
            if block.iter().any(|v| !v.is_finite()) {
                return Err(RandomEffectTestUnavailable::DesignUnavailable);
            }
            hessian_gram += &weighted_cross(&block, input.hessian_weights.slice(s![start..stop]));
            if estimated {
                let score_weights = input.score_weights.slice(s![start..stop]);
                let score = input.score.slice(s![start..stop]);
                fisher_gram += &weighted_cross(&block, score_weights);
                design_score += &block.t().dot(&score);
                for (&weight, &value) in score_weights.iter().zip(score.iter()) {
                    if weight > 0.0 {
                        score_energy += value * value / weight;
                        positive_rows += 1;
                    } else if value != 0.0 {
                        // `s = W_F·(z − η̂)` vanishes wherever `W_F` does; a
                        // nonzero score on a zero-weight row is not a fit's state.
                        return Err(RandomEffectTestUnavailable::DesignUnavailable);
                    }
                }
            }
            start = stop;
        }

        let scale = match input.scale {
            RandomEffectTestScale::Known { dispersion } => ResolvedScale::Known(dispersion),
            RandomEffectTestScale::Estimated => {
                let pinv = equilibrated_pseudo_inverse(&fisher_gram)
                    .ok_or(RandomEffectTestUnavailable::DesignUnavailable)?;
                let explained = design_score.dot(&pinv.inverse.dot(&design_score));
                ResolvedScale::Estimated {
                    residual_sum_of_squares: score_energy - explained,
                    residual_df: positive_rows as f64 - pinv.rank as f64,
                }
            }
        };
        Ok(Self {
            input,
            hessian_gram,
            scale,
        })
    }

    /// Test every requested term, reading the design once more for all of them.
    ///
    /// The result is aligned with `terms`.
    pub fn test_terms(
        &self,
        terms: &[RandomEffectTermRequest],
    ) -> Vec<Result<RandomEffectTest, RandomEffectTestUnavailable>> {
        let p = self.input.design.ncols();
        let mut prepared: Vec<Result<PreparedTerm, RandomEffectTestUnavailable>> = terms
            .iter()
            .map(|request| self.prepare_term(request, p))
            .collect();
        if prepared.iter().any(|term| term.is_ok())
            && let Err(reason) = self.accumulate(&mut prepared)
        {
            for term in prepared.iter_mut() {
                if term.is_ok() {
                    *term = Err(reason);
                }
            }
        }
        prepared
            .into_iter()
            .map(|term| term.and_then(|term| self.finish_term(term)))
            .collect()
    }

    fn prepare_term(
        &self,
        request: &RandomEffectTermRequest,
        p: usize,
    ) -> Result<PreparedTerm, RandomEffectTestUnavailable> {
        let range = request.range.clone();
        if range.is_empty() || range.end > p {
            return Err(RandomEffectTestUnavailable::NoEstimableDirection);
        }
        let tested: Vec<usize> = range.clone().collect();
        let other: Vec<usize> = (0..p).filter(|j| !range.contains(j)).collect();
        let projection = if other.is_empty() {
            Array2::<f64>::zeros((0, tested.len()))
        } else {
            let other_gram = self
                .hessian_gram
                .select(Axis(0), &other)
                .select(Axis(1), &other);
            let cross = self
                .hessian_gram
                .select(Axis(0), &other)
                .select(Axis(1), &tested);
            let pinv = equilibrated_pseudo_inverse(&other_gram)
                .ok_or(RandomEffectTestUnavailable::DesignUnavailable)?;
            pinv.inverse.dot(&cross)
        };
        if projection.iter().any(|v| !v.is_finite()) {
            return Err(RandomEffectTestUnavailable::DesignUnavailable);
        }
        let q = tested.len();
        Ok(PreparedTerm {
            beta_tested: self.input.beta.slice(s![range]).to_owned(),
            tested,
            other,
            projection,
            fisher_projected: Array2::zeros((q, q)),
            score: Array1::zeros(q),
            unprojected_trace: 0.0,
        })
    }

    /// Second pass: `V = X̃_RᵀW_F X̃_R` and `u = X̃_Rᵀv` for every prepared term.
    fn accumulate(
        &self,
        prepared: &mut [Result<PreparedTerm, RandomEffectTestUnavailable>],
    ) -> Result<(), RandomEffectTestUnavailable> {
        let n = self.input.design.nrows();
        let mut start = 0usize;
        while start < n {
            let stop = (start + ROW_BLOCK).min(n);
            let block = self
                .input
                .design
                .try_row_chunk(start..stop)
                .map_err(|_| RandomEffectTestUnavailable::DesignUnavailable)?;
            let hessian_weights = self.input.hessian_weights.slice(s![start..stop]);
            let score_weights = self.input.score_weights.slice(s![start..stop]);
            let score = self.input.score.slice(s![start..stop]);
            for term in prepared.iter_mut().flatten() {
                let tested_block = block.select(Axis(1), &term.tested);
                let mut residual = score.to_owned();
                residual += &(&hessian_weights * &tested_block.dot(&term.beta_tested));
                term.unprojected_trace += tested_block
                    .axis_iter(Axis(0))
                    .zip(score_weights.iter())
                    .map(|(row, &weight)| weight * row.dot(&row))
                    .sum::<f64>();
                let projected = if term.other.is_empty() {
                    tested_block
                } else {
                    tested_block - block.select(Axis(1), &term.other).dot(&term.projection)
                };
                term.fisher_projected += &weighted_cross(&projected, score_weights);
                term.score += &projected.t().dot(&residual);
            }
            start = stop;
        }
        Ok(())
    }

    fn finish_term(&self, term: PreparedTerm) -> Result<RandomEffectTest, RandomEffectTestUnavailable> {
        let p = self.input.design.ncols();
        if term.fisher_projected.iter().any(|v| !v.is_finite())
            || term.score.iter().any(|v| !v.is_finite())
        {
            return Err(RandomEffectTestUnavailable::DesignUnavailable);
        }
        let symmetric = 0.5 * (&term.fisher_projected + &term.fisher_projected.t());
        let (eigenvalues, _) = strict_symmetric_eigh(&symmetric, Side::Lower)
            .map_err(|_| RandomEffectTestUnavailable::DesignUnavailable)?;
        // `X̃_R` is a difference of two quantities of the size of `X_R`, so an
        // eigenvalue of `V` is resolved only above the rounding of that
        // difference: `p·ε` relative to `‖X_RᵀW_F X_R‖`, whose trace bounds it.
        let floor = (p as f64) * f64::EPSILON * term.unprojected_trace;
        let kept: Vec<usize> = (0..eigenvalues.len())
            .filter(|&j| eigenvalues[j] > floor)
            .collect();
        if kept.is_empty() {
            return Err(RandomEffectTestUnavailable::NoEstimableDirection);
        }
        let rank = kept.len();
        let residual_df = match self.scale {
            ResolvedScale::Known(_) => None,
            ResolvedScale::Estimated {
                residual_sum_of_squares,
                residual_df,
            } => {
                if !(residual_df >= 1.0 && residual_sum_of_squares > 0.0) {
                    return Err(RandomEffectTestUnavailable::NoResidualDegreesOfFreedom);
                }
                Some(residual_df)
            }
        };

        let statistic = term.score.dot(&term.score);
        let weights: Vec<f64> = kept.iter().map(|&j| eigenvalues[j]).collect();
        let weight_sum: f64 = weights.iter().sum();
        let weight_square_sum: f64 = weights.iter().map(|w| w * w).sum();
        let effective_df = weight_sum * weight_sum / weight_square_sum;
        let mut terms: Vec<WeightedChiSquareTerm> = weights
            .iter()
            .map(|&weight| WeightedChiSquareTerm {
                weight,
                degrees_of_freedom: 1.0,
            })
            .collect();
        let (tail, dispersion) = match self.scale {
            ResolvedScale::Known(dispersion) => (
                signed_weighted_chi_square_sf(&terms, statistic / dispersion),
                dispersion,
            ),
            ResolvedScale::Estimated {
                residual_sum_of_squares,
                residual_df,
            } => {
                terms.push(WeightedChiSquareTerm {
                    weight: -statistic / residual_sum_of_squares,
                    degrees_of_freedom: residual_df,
                });
                (
                    signed_weighted_chi_square_sf(&terms, 0.0),
                    residual_sum_of_squares / residual_df,
                )
            }
        };
        let (p_value, p_value_relative_error) = resolved_tail(tail.probability, tail.relative_error)?;
        Ok(RandomEffectTest {
            statistic: statistic / dispersion * effective_df / weight_sum,
            reference_df: effective_df,
            rank,
            residual_df,
            p_value,
            p_value_relative_error,
        })
    }
}

struct PreparedTerm {
    tested: Vec<usize>,
    other: Vec<usize>,
    beta_tested: Array1<f64>,
    /// `A = G_OO⁻G_OR` in the `W_H` metric.
    projection: Array2<f64>,
    /// `V = X̃_RᵀW_F X̃_R`.
    fisher_projected: Array2<f64>,
    /// `u = X̃_Rᵀv`.
    score: Array1<f64>,
    /// `tr(X_RᵀW_F X_R)`, the scale `V`'s rounding floor is relative to.
    unprojected_trace: f64,
}

/// Read the weighted chi-square tail through its own error contract: a
/// relative error of one or more carries no information, EXCEPT for a
/// probability of exactly zero, which the evaluator reports that way when the
/// tail lies below the subnormal range — a resolved "smaller than any
/// representable p-value".
fn resolved_tail(probability: f64, relative_error: f64) -> Result<(f64, f64), RandomEffectTestUnavailable> {
    if probability.is_nan() || relative_error.is_nan() {
        return Err(RandomEffectTestUnavailable::TailUnresolved);
    }
    if relative_error >= 1.0 && probability > 0.0 {
        return Err(RandomEffectTestUnavailable::TailUnresolved);
    }
    Ok((probability, relative_error))
}

/// `Bᵀ diag(w) B`.
fn weighted_cross(block: &Array2<f64>, weights: ArrayView1<'_, f64>) -> Array2<f64> {
    let mut weighted = block.clone();
    for (mut row, &weight) in weighted.axis_iter_mut(Axis(0)).zip(weights.iter()) {
        row.iter_mut().for_each(|v| *v *= weight);
    }
    block.t().dot(&weighted)
}

struct PseudoInverse {
    inverse: Array2<f64>,
    rank: usize,
}

/// Moore-Penrose inverse of a symmetric positive semi-definite Gram, taken on
/// its Jacobi-equilibrated form `C = D^{-1/2} G D^{-1/2}` (`D = diag(G)`) so the
/// rank decision is made on a matrix whose diagonal is one: a design mixing a
/// raw intercept with a smooth basis of a very different column scale would
/// otherwise have its small-but-real directions judged against the largest
/// column's norm. A zero-diagonal column is a column with no weight on any row;
/// it is left unscaled and falls below the floor. The floor is the rounding of
/// the eigen-solve, `dim·ε·λ_max(C)`.
///
/// `D^{-1/2} C⁺ D^{-1/2}` is a generalized inverse of `G` (not its Moore-Penrose
/// inverse when `G` is singular), which is all a projection and a residual sum
/// of squares require: `G G⁻ G = G` makes both invariant to the choice.
fn equilibrated_pseudo_inverse(gram: &Array2<f64>) -> Option<PseudoInverse> {
    let dim = gram.nrows();
    if dim == 0 || gram.ncols() != dim || gram.iter().any(|v| !v.is_finite()) {
        return None;
    }
    let inverse_root: Array1<f64> = gram
        .diag()
        .iter()
        .map(|&d| if d > 0.0 { 1.0 / d.sqrt() } else { 1.0 })
        .collect();
    let mut equilibrated = gram.clone();
    for i in 0..dim {
        for j in 0..dim {
            equilibrated[[i, j]] *= inverse_root[i] * inverse_root[j];
        }
    }
    let symmetric = 0.5 * (&equilibrated + &equilibrated.t());
    let (eigenvalues, eigenvectors) = strict_symmetric_eigh(&symmetric, Side::Lower).ok()?;
    let largest = eigenvalues.iter().cloned().fold(0.0_f64, f64::max);
    if !(largest > 0.0) {
        return Some(PseudoInverse {
            inverse: Array2::zeros((dim, dim)),
            rank: 0,
        });
    }
    let floor = (dim as f64) * f64::EPSILON * largest;
    let mut scaled = eigenvectors.clone();
    let mut rank = 0usize;
    for (index, &eigenvalue) in eigenvalues.iter().enumerate() {
        let factor = if eigenvalue > floor {
            rank += 1;
            1.0 / eigenvalue
        } else {
            0.0
        };
        scaled.column_mut(index).iter_mut().for_each(|v| *v *= factor);
    }
    let mut inverse = scaled.dot(&eigenvectors.t());
    for i in 0..dim {
        for j in 0..dim {
            inverse[[i, j]] *= inverse_root[i] * inverse_root[j];
        }
    }
    inverse
        .iter()
        .all(|v| v.is_finite())
        .then_some(PseudoInverse { inverse, rank })
}

#[cfg(test)]
mod tests {
    use super::*;
    use gam_linalg::matrix::{DenseDesignMatrix, DesignMatrix};
    use gam_math::probability::fisher_snedecor_sf;

    struct Lcg(u64);

    impl Lcg {
        fn next_uniform(&mut self) -> f64 {
            self.0 = self
                .0
                .wrapping_mul(6_364_136_223_846_793_005)
                .wrapping_add(1_442_695_040_888_963_407);
            ((self.0 >> 11) as f64) / ((1u64 << 53) as f64)
        }

        fn next_normal(&mut self) -> f64 {
            let u1 = loop {
                let u = self.next_uniform();
                if u > 0.0 {
                    break u;
                }
            };
            let u2 = self.next_uniform();
            (-2.0 * u1.ln()).sqrt() * (std::f64::consts::TAU * u2).cos()
        }
    }

    /// `[1 | x | indicator(g)]` for group labels `groups`, with the indicator
    /// block at `2..2+levels`.
    fn one_way_design(groups: &[usize], levels: usize, x: &[f64]) -> Array2<f64> {
        let n = groups.len();
        let mut design = Array2::<f64>::zeros((n, 2 + levels));
        for i in 0..n {
            design[[i, 0]] = 1.0;
            design[[i, 1]] = x[i];
            design[[i, 2 + groups[i]]] = 1.0;
        }
        design
    }

    /// Gaussian identity row state for an arbitrary `β` — the statistic must not
    /// depend on it.
    fn gaussian_test(
        design: &Array2<f64>,
        y: &Array1<f64>,
        beta: &Array1<f64>,
        range: Range<usize>,
        scale: RandomEffectTestScale,
    ) -> Result<RandomEffectTest, RandomEffectTestUnavailable> {
        let n = design.nrows();
        let backing = DesignMatrix::Dense(DenseDesignMatrix::from(design.clone()));
        let weights = Array1::<f64>::ones(n);
        let score = y - &design.dot(beta);
        let basis = RandomEffectTestBasis::new(RandomEffectTestInput {
            design: &backing,
            beta: beta.view(),
            hessian_weights: weights.view(),
            score_weights: weights.view(),
            score: score.view(),
            scale,
        })?;
        basis
            .test_terms(&[RandomEffectTermRequest { range }])
            .pop()
            .expect("one term requested")
    }

    /// Classical one-way ANOVA `F` for the group factor after the intercept.
    fn anova_f(groups: &[usize], levels: usize, y: &Array1<f64>) -> (f64, f64, f64) {
        let n = groups.len();
        let grand = y.mean().unwrap();
        let mut sums = vec![0.0; levels];
        let mut counts = vec![0.0; levels];
        for i in 0..n {
            sums[groups[i]] += y[i];
            counts[groups[i]] += 1.0;
        }
        let means: Vec<f64> = (0..levels).map(|l| sums[l] / counts[l]).collect();
        let between: f64 = (0..levels)
            .map(|l| counts[l] * (means[l] - grand).powi(2))
            .sum();
        let within: f64 = (0..n).map(|i| (y[i] - means[groups[i]]).powi(2)).sum();
        let df1 = (levels - 1) as f64;
        let df2 = (n - levels) as f64;
        ((between / df1) / (within / df2), df1, df2)
    }

    fn intercept_and_groups(groups: &[usize], levels: usize) -> Array2<f64> {
        let n = groups.len();
        let mut design = Array2::<f64>::zeros((n, 1 + levels));
        for i in 0..n {
            design[[i, 0]] = 1.0;
            design[[i, 1 + groups[i]]] = 1.0;
        }
        design
    }

    #[test]
    fn balanced_estimated_scale_variance_component_is_the_one_way_anova_f() {
        let levels = 6;
        let groups: Vec<usize> = (0..60).map(|i| i % levels).collect();
        let mut rng = Lcg(7);
        let y: Array1<f64> = groups
            .iter()
            .map(|&g| 0.3 * g as f64 + rng.next_normal())
            .collect();
        let design = intercept_and_groups(&groups, levels);
        let beta = Array1::<f64>::zeros(design.ncols());
        let test = gaussian_test(
            &design,
            &y,
            &beta,
            1..1 + levels,
            RandomEffectTestScale::Estimated,
        )
        .expect("test runs");
        let (f, df1, df2) = anova_f(&groups, levels, &y);
        let expected = fisher_snedecor_sf(f, df1, df2);
        assert_eq!(test.rank, levels - 1);
        assert!((test.reference_df - df1).abs() < 1e-9, "{test:?}");
        assert_eq!(test.residual_df, Some(df2));
        assert!((test.statistic - f * df1).abs() < 1e-8 * f * df1, "{test:?} vs F={f}");
        assert!(
            (test.p_value - expected).abs() <= 1e-8 * expected + 1e-14,
            "{} vs {expected}",
            test.p_value
        );
    }

    #[test]
    fn statistic_does_not_depend_on_where_the_fit_left_beta() {
        let levels = 8;
        let mut rng = Lcg(3);
        let n = 120;
        let x: Vec<f64> = (0..n).map(|_| rng.next_uniform()).collect();
        let groups: Vec<usize> = (0..n).map(|_| (rng.next_uniform() * levels as f64) as usize).collect();
        let design = one_way_design(&groups, levels, &x);
        let y: Array1<f64> = (0..n).map(|i| 2.0 * x[i] + rng.next_normal()).collect();
        let reference = gaussian_test(
            &design,
            &y,
            &Array1::zeros(design.ncols()),
            2..2 + levels,
            RandomEffectTestScale::Estimated,
        )
        .expect("test runs");
        // Any β — a heavily shrunk random effect, a biased slope — gives the same
        // score: `u = X̃_RᵀWy` exactly for a Gaussian identity fit.
        let shifted: Array1<f64> = (0..design.ncols()).map(|j| 0.1 * j as f64 - 0.4).collect();
        let moved = gaussian_test(
            &design,
            &y,
            &shifted,
            2..2 + levels,
            RandomEffectTestScale::Estimated,
        )
        .expect("test runs");
        assert!((moved.statistic - reference.statistic).abs() < 1e-9 * reference.statistic);
        assert!((moved.p_value - reference.p_value).abs() < 1e-9);
    }

    #[test]
    fn null_p_values_are_uniform_for_known_and_estimated_scale() {
        let levels = 7;
        let n = 70;
        let reps = 2000;
        let mut rng = Lcg(2024);
        let x: Vec<f64> = (0..n).map(|_| rng.next_uniform()).collect();
        // Unbalanced: level `l` holds the rows with `(i/n)² ∈ [l/L, (l+1)/L)`,
        // 27 rows in the first level down to 5 in the last.
        let groups: Vec<usize> = (0..n)
            .map(|i| ((i as f64 / n as f64).powi(2) * levels as f64) as usize)
            .collect();
        let design = one_way_design(&groups, levels, &x);
        let beta = Array1::<f64>::zeros(design.ncols());
        for scale in [
            RandomEffectTestScale::Known { dispersion: 1.0 },
            RandomEffectTestScale::Estimated,
        ] {
            let mut rejections_05 = 0usize;
            let mut rejections_10 = 0usize;
            let mut sum = 0.0;
            for _ in 0..reps {
                let y: Array1<f64> = (0..n).map(|i| 1.0 + x[i] + rng.next_normal()).collect();
                let test = gaussian_test(&design, &y, &beta, 2..2 + levels, scale)
                    .expect("test runs");
                sum += test.p_value;
                rejections_05 += usize::from(test.p_value < 0.05);
                rejections_10 += usize::from(test.p_value < 0.10);
            }
            let m = reps as f64;
            for (alpha, count) in [(0.05, rejections_05), (0.10, rejections_10)] {
                let rate = count as f64 / m;
                let mcse = (alpha * (1.0 - alpha) / m).sqrt();
                assert!(
                    (rate - alpha).abs() <= 3.0 * mcse,
                    "{scale:?}: size {rate} at {alpha}"
                );
            }
            let mean = sum / m;
            assert!((mean - 0.5).abs() <= 3.0 * (1.0 / 12.0 / m).sqrt(), "{scale:?}: mean {mean}");
        }
    }

    #[test]
    fn a_real_group_effect_is_detected() {
        let levels = 10;
        let n = 200;
        let mut rng = Lcg(5);
        let effects: Vec<f64> = (0..levels).map(|_| 0.6 * rng.next_normal()).collect();
        let x: Vec<f64> = (0..n).map(|_| rng.next_uniform()).collect();
        let groups: Vec<usize> = (0..n).map(|i| i % levels).collect();
        let design = one_way_design(&groups, levels, &x);
        let y: Array1<f64> = (0..n).map(|i| x[i] + effects[groups[i]] + rng.next_normal()).collect();
        let test = gaussian_test(
            &design,
            &y,
            &Array1::zeros(design.ncols()),
            2..2 + levels,
            RandomEffectTestScale::Known { dispersion: 1.0 },
        )
        .expect("test runs");
        assert!(test.p_value < 1e-6, "{test:?}");
    }

    #[test]
    fn a_term_spanned_by_the_other_columns_has_no_estimable_direction() {
        let groups: Vec<usize> = (0..40).map(|i| i % 4).collect();
        let mut design = intercept_and_groups(&groups, 4);
        // Duplicate the group block: the second copy is spanned by the first.
        let duplicate = design.slice(s![.., 1..5]).to_owned();
        design = ndarray::concatenate![Axis(1), design, duplicate];
        let y: Array1<f64> = (0..40).map(|i| i as f64 * 0.01).collect();
        let reason = gaussian_test(
            &design,
            &y,
            &Array1::zeros(design.ncols()),
            5..9,
            RandomEffectTestScale::Known { dispersion: 1.0 },
        )
        .expect_err("no direction survives");
        assert_eq!(reason, RandomEffectTestUnavailable::NoEstimableDirection);
    }

    #[test]
    fn a_saturated_design_has_no_residual_degrees_of_freedom() {
        let groups: Vec<usize> = (0..6).collect();
        let design = intercept_and_groups(&groups, 6);
        let y: Array1<f64> = (0..6).map(|i| i as f64).collect();
        let reason = gaussian_test(
            &design,
            &y,
            &Array1::zeros(design.ncols()),
            1..7,
            RandomEffectTestScale::Estimated,
        )
        .expect_err("no residual d.f.");
        assert_eq!(reason, RandomEffectTestUnavailable::NoResidualDegreesOfFreedom);
    }

    #[test]
    fn records_serialize_with_a_typed_status() {
        let record = RandomEffectTestRecord {
            term: "g".into(),
            coefficient_range: 1..4,
            outcome: RandomEffectTestOutcome::Unavailable {
                reason: RandomEffectTestUnavailable::NoEstimableDirection,
            },
        };
        let json = serde_json::to_string(&record).unwrap();
        assert!(json.contains("\"status\":\"unavailable\""), "{json}");
        assert!(json.contains("\"reason\":\"no_estimable_direction\""), "{json}");
        let back: RandomEffectTestRecord = serde_json::from_str(&json).unwrap();
        assert_eq!(back, record);
    }
}
