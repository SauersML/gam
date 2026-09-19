//! Score test of a penalized coefficient block's variance components on the
//! boundary: `group(g)`/`re(g)` random effects, and smooths whose penalties
//! jointly penalize every coefficient direction (double-penalty smooths,
//! shrinkage smooths, tensor smooths with a null-space penalty).
//!
//! # The question this answers
//!
//! A penalized block is `η = Xβ + X_R b` with a quadratic penalty
//! `Σ_j λ_j bᵀS_j b`, i.e. the Gaussian prior whose precision is `Σ_j λ_j S_j`.
//! When the `S_j` jointly penalize every direction of `b`, "does this term
//! matter?" is `H₀: every variance τ_j = 1/λ_j is 0`, and that null sits on
//! the BOUNDARY of the parameter space. None of the reference laws the Wald
//! smooth table uses are valid there:
//!
//! * the coefficient Wald statistic `b̂ᵀV_b⁻b̂` is computed from a `b̂` that the
//!   penalty shrinks toward zero by an amount REML chose from the same data, so
//!   its `χ²_edf` reference is neither the right shape nor the right scale. A
//!   double-penalty smooth that REML shrinks to `edf ≈ 0` has `T ≈ 0` and
//!   `p ≈ 1` in a large fraction of null fits: a point mass that makes the test
//!   conservative;
//! * the likelihood-ratio statistic has the chi-bar-square law
//!   `½χ²₀ + ½χ²₁` only for a SINGLE variance parameter with a scalar-shaped
//!   information, and even then its finite-sample law for a Gaussian model is a
//!   point mass at zero far larger than one half (Crainiceanu & Ruppert, 2004).
//!
//! # The statistic
//!
//! The score for a variance component `τ` of prior covariance `τΣ` at `τ = 0`
//! (Lin, 1997) is, up to a constant,
//!
//! ```text
//! T = uᵀΣ u,        u = X̃_Rᵀ v,
//! ```
//!
//! where `v` is the working residual of the model WITHOUT the block and
//! `X̃_R = X_R − X_O A`, `A = (X_OᵀW_H X_O)⁻ X_OᵀW_H X_R`, is the block's
//! design with every OTHER column of the fit (intercept, linear terms, other
//! smooths, other random effects) projected out in the fit's own curvature
//! metric. No `b̂` enters, so the statistic cannot be shrunk by the penalty it
//! is testing.
//!
//! A block with several penalties has one variance per penalty, and the test is
//! the score in ONE fixed direction of that variance cone, the one that gives
//! every component the same null mean:
//!
//! ```text
//! Σ₀ = Σ_j K_j / tr(K_j V),      T = uᵀΣ₀u = Σ_j uᵀK_ju / tr(K_j V),
//! ```
//!
//! so `E₀[uᵀK_ju] = φ·tr(K_jV)` makes every component's share of `T/φ` a
//! unit-mean quantity under `H₀`.
//!
//! `K_j` is penalty `j`'s share of the prior covariance. At the base precision
//! `P = Σ_j a_j S_j`, `a_j = 1/tr(G_u⁻S_j)` (each penalty on the scale of the
//! data it penalizes, `G_u = X_RᵀW_F X_R`),
//!
//! ```text
//! K_j = P⁻¹ (a_j S_j) P⁻¹,        Σ_j K_j = P⁻¹,
//! ```
//!
//! the derivative of the prior covariance `(Σ_k λ_k S_k)⁻¹` along `−log λ_j`.
//! It is NOT the pseudo-inverse `S_j⁺`: a double penalty (wiggliness plus a
//! ridge on its null space) has ranges that are complementary but, once the
//! basis factory has reparametrized, identifiability-constrained and
//! centred the block, no longer Euclidean-orthogonal, and `S_j⁺` of a
//! reparametrized penalty is not the reparametrized `S_j⁺`. A direction built
//! from `S_j⁺` then scores a mixture of the components that depends on the
//! chart and can all but miss a purely linear effect. `K_j` moves covariantly
//! under `X_R → X_R Z`, `S_j → ZᵀS_jZ` (`a_j` is invariant, `P → ZᵀPZ`,
//! `K_j → Z⁻¹K_jZ⁻ᵀ`), so `T` and its reference law do not depend on the chart,
//! and for complementary ranges `K_j` is proportional to the prior covariance
//! of penalty `j` alone in every chart, whatever the base; the base only
//! shapes the direction for genuinely overlapping ranges (a tensor product's
//! marginal penalties), where `Σ₀` is still a fixed positive definite
//! weighting of the block.
//!
//! The direction depends only on the design, the weights and the penalty
//! STRUCTURE — never on `y`, `b̂` or the fitted `λ̂` — so the reference law below
//! is exact for it, and it is invariant to rescaling any `S_j`, so the p-value
//! does not depend on how a basis factory normalized its penalties.
//! Standardizing by the null means keeps a low-dimensional component (the
//! linear null space of a smoothing penalty) from being swamped by a
//! high-dimensional one: each carries unit null mean, so a purely linear effect
//! and a purely wiggly one are both seen. A `group()` block's identity ridge
//! gives `Σ₀ ∝ I` and the Lin statistic `‖u‖²`.
//!
//! `v` is read off the full fit rather than a refit: with `s = W_F ⊙ (z − η̂)`
//! the fit's working score,
//!
//! ```text
//! v = s + W_H ⊙ (X_R b̂)
//! ```
//!
//! adds the block's own contribution back into the residual, and projecting
//! `X_O` out of `X_R` in the `W_H` metric makes `u` blind to where `β̂_O`
//! landed: `X̃_RᵀW_H X_O = 0`, so any shift of `β̂_O` — including the shrinkage
//! bias of every other penalized smooth — leaves `u` unchanged. For a Gaussian
//! identity fit this is exact, `u = X̃_RᵀW y`, whatever `β̂` the optimizer
//! returned. For a GLM it is the one-step score at `b = 0`.
//!
//! # The reference law
//!
//! Under `H₀`, `u ~ N(0, φV)` with `V = X̃_RᵀW_F X̃_R`, so with `Σ₀ = LLᵀ`
//!
//! ```text
//! T/φ ~ Σ_i μ_i χ²₁,      μ = eig(LᵀVL),
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
//! P(T/D' > t) = P(Σ_i μ_i χ²₁ − t·χ²_ν > 0),
//! ```
//!
//! a signed weighted chi-square tail at zero — exact again, and exactly the
//! one-way ANOVA `F` on a balanced design. `D'` is the unpenalized residual and
//! not the fit's `φ̂` because `φ̂` is computed from a residual the penalty shrank,
//! whose law under `H₀` depends on the smoothing parameters REML chose.
//!
//! # Scope
//!
//! The test needs the penalties to cover the whole block: a direction no
//! penalty touches is a fixed effect, `b = 0` along it is an interior null,
//! and a zero variance component would not mean a zero effect. Such a block is
//! refused ([`VarianceComponentTestUnavailable::UnpenalizedDirections`]), as
//! is a block that carries no penalty at all; the smooth summary keeps its
//! Wald test for those terms.

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
pub enum VarianceComponentTestScale {
    /// `φ` is fixed by the family (`1` for binomial and Poisson), or pinned.
    Known { dispersion: f64 },
    /// `φ` was estimated from the data, so the statistic is a ratio against the
    /// unpenalized residual sum of squares.
    Estimated,
}

/// Why a block has no p-value.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum VarianceComponentTestUnavailable {
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
    /// The block's penalties leave some coefficient directions unpenalized, so
    /// a zero variance component would not mean a zero effect, or a penalty
    /// was not a finite square matrix of the block's size.
    UnpenalizedDirections,
    /// The reference tail could not be resolved to any accuracy.
    TailUnresolved,
}

impl VarianceComponentTestUnavailable {
    /// Serialized label carried into the model payload and the Python surface.
    pub fn label(self) -> &'static str {
        match self {
            Self::NoIrlsRowState => "variance_component_no_irls_row_state",
            Self::DesignUnavailable => "variance_component_design_unavailable",
            Self::NoEstimableDirection => "variance_component_no_estimable_direction",
            Self::NoResidualDegreesOfFreedom => "variance_component_no_residual_degrees_of_freedom",
            Self::UnpenalizedDirections => "variance_component_unpenalized_directions",
            Self::TailUnresolved => "variance_component_tail_unresolved",
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
            Self::UnpenalizedDirections => {
                "the term's penalties do not cover every coefficient direction"
            }
            Self::TailUnresolved => "the reference tail probability could not be resolved",
        }
    }
}

/// A computed variance-component test.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct VarianceComponentTest {
    /// Reported on a chi-square-like scale with mean `reference_df` under `H₀`:
    /// `(T/φ̂)·reference_df/Σμ`, with `φ̂ = φ` for a known scale and `D'/ν` for
    /// an estimated one.
    pub statistic: f64,
    /// The effective degrees of freedom `(Σμ)²/Σμ²` of the spectral reference
    /// — `rank` exactly when the design is balanced.
    pub reference_df: f64,
    /// Number of estimable directions of the term (reference weights above
    /// their rounding floor).
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
pub enum VarianceComponentTestOutcome {
    Tested(VarianceComponentTest),
    Unavailable { reason: VarianceComponentTestUnavailable },
}

impl From<Result<VarianceComponentTest, VarianceComponentTestUnavailable>> for VarianceComponentTestOutcome {
    fn from(result: Result<VarianceComponentTest, VarianceComponentTestUnavailable>) -> Self {
        match result {
            Ok(test) => Self::Tested(test),
            Err(reason) => Self::Unavailable { reason },
        }
    }
}

/// One term's test, keyed the way the summary walks the design.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct VarianceComponentTestRecord {
    /// Term name (matches the summary row).
    pub term: String,
    /// The term's GLOBAL coefficient range.
    pub coefficient_range: Range<usize>,
    pub outcome: VarianceComponentTestOutcome,
}

/// One term to test.
#[derive(Clone, Debug)]
pub struct VarianceComponentTermRequest {
    /// GLOBAL coefficient range of the term's block.
    pub range: Range<usize>,
    /// The block's LOCAL penalties `S_j` (each `q × q`, `q = range.len()`, in
    /// the block's coefficient basis), which must jointly penalize every
    /// direction. Tested at the boundary null "every variance is zero".
    pub penalties: Vec<Array2<f64>>,
}

/// The fit's row state, in the fit's own row and coefficient layout.
pub struct VarianceComponentTestInput<'a> {
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
    pub scale: VarianceComponentTestScale,
}

/// Per-fit quantities shared by every tested term: the design Gram in the `W_H`
/// metric and, for an estimated scale, the unpenalized residual.
pub struct VarianceComponentTestBasis<'a> {
    input: VarianceComponentTestInput<'a>,
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

impl<'a> VarianceComponentTestBasis<'a> {
    /// First pass over the design: `G_H = XᵀW_H X`, and for an estimated scale
    /// `G_F = XᵀW_F X`, `Xᵀs` and `Σ s²/W_F`, from which
    /// `D' = Σ s²/W_F − (Xᵀs)ᵀG_F⁻(Xᵀs)` is the weighted residual sum of squares
    /// of the unpenalized fit: its residual is `e − X G_F⁻Xᵀs` with
    /// `e = z − η̂`, since the unpenalized solution is one Newton step from `β̂`.
    pub fn new(input: VarianceComponentTestInput<'a>) -> Result<Self, VarianceComponentTestUnavailable> {
        let n = input.design.nrows();
        let p = input.design.ncols();
        if n == 0
            || p == 0
            || input.beta.len() != p
            || input.hessian_weights.len() != n
            || input.score_weights.len() != n
            || input.score.len() != n
        {
            return Err(VarianceComponentTestUnavailable::DesignUnavailable);
        }
        let rows_finite = input.beta.iter().all(|v| v.is_finite())
            && input.hessian_weights.iter().all(|v| v.is_finite())
            && input.score_weights.iter().all(|v| v.is_finite() && *v >= 0.0)
            && input.score.iter().all(|v| v.is_finite());
        if !rows_finite {
            return Err(VarianceComponentTestUnavailable::DesignUnavailable);
        }
        let estimated = matches!(input.scale, VarianceComponentTestScale::Estimated);
        if let VarianceComponentTestScale::Known { dispersion } = input.scale
            && !(dispersion.is_finite() && dispersion > 0.0)
        {
            return Err(VarianceComponentTestUnavailable::DesignUnavailable);
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
                .map_err(|_| VarianceComponentTestUnavailable::DesignUnavailable)?;
            if block.iter().any(|v| !v.is_finite()) {
                return Err(VarianceComponentTestUnavailable::DesignUnavailable);
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
                        return Err(VarianceComponentTestUnavailable::DesignUnavailable);
                    }
                }
            }
            start = stop;
        }

        let scale = match input.scale {
            VarianceComponentTestScale::Known { dispersion } => ResolvedScale::Known(dispersion),
            VarianceComponentTestScale::Estimated => {
                let pinv = equilibrated_pseudo_inverse(&fisher_gram)
                    .ok_or(VarianceComponentTestUnavailable::DesignUnavailable)?;
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
        terms: &[VarianceComponentTermRequest],
    ) -> Vec<Result<VarianceComponentTest, VarianceComponentTestUnavailable>> {
        let p = self.input.design.ncols();
        let mut prepared: Vec<Result<PreparedTerm, VarianceComponentTestUnavailable>> = terms
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
        request: &VarianceComponentTermRequest,
        p: usize,
    ) -> Result<PreparedTerm, VarianceComponentTestUnavailable> {
        let range = request.range.clone();
        if range.is_empty() || range.end > p {
            return Err(VarianceComponentTestUnavailable::NoEstimableDirection);
        }
        let q = range.len();
        let penalties = covering_penalties(&request.penalties, q)?;
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
                .ok_or(VarianceComponentTestUnavailable::DesignUnavailable)?;
            pinv.inverse.dot(&cross)
        };
        if projection.iter().any(|v| !v.is_finite()) {
            return Err(VarianceComponentTestUnavailable::DesignUnavailable);
        }
        Ok(PreparedTerm {
            penalties,
            beta_tested: self.input.beta.slice(s![range]).to_owned(),
            tested,
            other,
            projection,
            fisher_projected: Array2::zeros((q, q)),
            fisher_unprojected: Array2::zeros((q, q)),
            score: Array1::zeros(q),
        })
    }

    /// Second pass: `V = X̃_RᵀW_F X̃_R`, `G_u = X_RᵀW_F X_R` and `u = X̃_Rᵀv` for
    /// every prepared term.
    fn accumulate(
        &self,
        prepared: &mut [Result<PreparedTerm, VarianceComponentTestUnavailable>],
    ) -> Result<(), VarianceComponentTestUnavailable> {
        let n = self.input.design.nrows();
        let mut start = 0usize;
        while start < n {
            let stop = (start + ROW_BLOCK).min(n);
            let block = self
                .input
                .design
                .try_row_chunk(start..stop)
                .map_err(|_| VarianceComponentTestUnavailable::DesignUnavailable)?;
            let hessian_weights = self.input.hessian_weights.slice(s![start..stop]);
            let score_weights = self.input.score_weights.slice(s![start..stop]);
            let score = self.input.score.slice(s![start..stop]);
            for term in prepared.iter_mut().flatten() {
                let tested_block = block.select(Axis(1), &term.tested);
                let mut residual = score.to_owned();
                residual += &(&hessian_weights * &tested_block.dot(&term.beta_tested));
                term.fisher_unprojected += &weighted_cross(&tested_block, score_weights);
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

    fn finish_term(
        &self,
        term: PreparedTerm,
    ) -> Result<VarianceComponentTest, VarianceComponentTestUnavailable> {
        let p = self.input.design.ncols();
        if term.fisher_projected.iter().any(|v| !v.is_finite())
            || term.fisher_unprojected.iter().any(|v| !v.is_finite())
            || term.score.iter().any(|v| !v.is_finite())
        {
            return Err(VarianceComponentTestUnavailable::DesignUnavailable);
        }
        let residual_df = match self.scale {
            ResolvedScale::Known(_) => None,
            ResolvedScale::Estimated {
                residual_sum_of_squares,
                residual_df,
            } => {
                if !(residual_df >= 1.0 && residual_sum_of_squares > 0.0) {
                    return Err(VarianceComponentTestUnavailable::NoResidualDegreesOfFreedom);
                }
                Some(residual_df)
            }
        };
        // `X̃_R` is a difference of two quantities of the size of `X_R`, so a
        // quadratic form `aᵀVa` is resolved only above the rounding of that
        // difference: `p·ε` relative to the same form in the unprojected Gram,
        // `aᵀG_u a`, which bounds it.
        let rounding = (p as f64) * f64::EPSILON;
        self.variance_component(
            &term.penalties,
            &term.fisher_projected,
            &term.fisher_unprojected,
            &term.score,
            rounding,
            residual_df,
        )
    }

    fn variance_component(
        &self,
        penalties: &[Array2<f64>],
        fisher_projected: &Array2<f64>,
        fisher_unprojected: &Array2<f64>,
        score: &Array1<f64>,
        rounding: f64,
        residual_df: Option<f64>,
    ) -> Result<VarianceComponentTest, VarianceComponentTestUnavailable> {
        let q = score.len();
        let components = prior_covariance_components(penalties, fisher_unprojected)?;
        // `Σ₀ = Σ_j K_j/tr(K_j V)` over the components the projected design
        // resolves. A component whose whole null mean is rounding is a prior
        // direction the other columns already span; it carries no information
        // and is left out of the direction rather than divided by noise.
        let mut direction = Array2::<f64>::zeros((q, q));
        for component in &components {
            let null_mean = trace_product(component, fisher_projected);
            let resolution = rounding * trace_product(component, fisher_unprojected);
            if null_mean > resolution {
                direction.scaled_add(1.0 / null_mean, component);
            }
        }
        let (direction_values, direction_vectors) = symmetric_eigh(&direction)?;
        let direction_floor = crate::basis::spectral_tolerance(&direction_values);
        let root_columns: Vec<usize> = (0..q)
            .filter(|&j| direction_values[j] > direction_floor)
            .collect();
        if root_columns.is_empty() {
            return Err(VarianceComponentTestUnavailable::NoEstimableDirection);
        }
        // `L` with `LLᵀ = Σ₀`; `T = ‖Lᵀu‖²` and `T/φ ~ Σ eig(LᵀVL)·χ²₁`.
        let mut root = direction_vectors.select(Axis(1), &root_columns);
        for (mut column, &j) in root.axis_iter_mut(Axis(1)).zip(root_columns.iter()) {
            let factor = direction_values[j].sqrt();
            column.iter_mut().for_each(|v| *v *= factor);
        }
        let reference = root.t().dot(&fisher_projected.dot(&root));
        let (reference_values, _) = symmetric_eigh(&reference)?;
        let floor = rounding * trace_product(&root.dot(&root.t()), fisher_unprojected);
        let weights: Vec<f64> = reference_values
            .iter()
            .copied()
            .filter(|&value| value > floor)
            .collect();
        if weights.is_empty() {
            return Err(VarianceComponentTestUnavailable::NoEstimableDirection);
        }
        let rooted_score = root.t().dot(score);
        let statistic = rooted_score.dot(&rooted_score);
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
        let (p_value, p_value_relative_error) =
            resolved_tail(tail.probability, tail.relative_error)?;
        Ok(VarianceComponentTest {
            statistic: statistic / dispersion * effective_df / weight_sum,
            reference_df: effective_df,
            rank: weights.len(),
            residual_df,
            p_value,
            p_value_relative_error,
        })
    }
}

struct PreparedTerm {
    /// The covering penalties `S_j`.
    penalties: Vec<Array2<f64>>,
    tested: Vec<usize>,
    other: Vec<usize>,
    beta_tested: Array1<f64>,
    /// `A = G_OO⁻G_OR` in the `W_H` metric.
    projection: Array2<f64>,
    /// `V = X̃_RᵀW_F X̃_R`.
    fisher_projected: Array2<f64>,
    /// `G_u = X_RᵀW_F X_R`, the scale `V`'s rounding floor is relative to.
    fisher_unprojected: Array2<f64>,
    /// `u = X̃_Rᵀv`.
    score: Array1<f64>,
}

/// Every penalty that penalizes something, symmetrized and read at the crate's
/// one penalty-spectrum rank cutoff, after checking that together they cover
/// the block: `Σ_j P_j`, the sum of the orthogonal projectors onto the ranges,
/// is nonsingular exactly when no direction lies in every null space.
fn covering_penalties(
    penalties: &[Array2<f64>],
    q: usize,
) -> Result<Vec<Array2<f64>>, VarianceComponentTestUnavailable> {
    let mut kept = Vec::with_capacity(penalties.len());
    let mut coverage = Array2::<f64>::zeros((q, q));
    for penalty in penalties {
        if penalty.nrows() != q || penalty.ncols() != q || penalty.iter().any(|v| !v.is_finite())
        {
            return Err(VarianceComponentTestUnavailable::UnpenalizedDirections);
        }
        let symmetric = 0.5 * (penalty + &penalty.t());
        let (values, vectors) = symmetric_eigh(&symmetric)?;
        let cutoff = crate::basis::spectral_tolerance(&values);
        let range: Vec<usize> = (0..q).filter(|&j| values[j] > cutoff).collect();
        if range.is_empty() {
            continue;
        }
        let basis = vectors.select(Axis(1), &range);
        coverage += &basis.dot(&basis.t());
        kept.push(symmetric);
    }
    let (coverage_values, _) = symmetric_eigh(&coverage)?;
    let cutoff = crate::basis::spectral_tolerance(&coverage_values);
    if kept.is_empty() || coverage_values.iter().any(|&value| value <= cutoff) {
        return Err(VarianceComponentTestUnavailable::UnpenalizedDirections);
    }
    Ok(kept)
}

/// The prior covariance components `K_j = P⁻¹(a_j S_j)P⁻¹` of the block at the
/// base precision `P = Σ_j a_j S_j`, `a_j = 1/tr(G_u⁻ S_j)`.
///
/// `K_j` is the share of the prior covariance `P⁻¹ = Σ_j K_j` that penalty `j`
/// carries: `−∂(Σ_k λ_k S_k)⁻¹/∂ log λ_j` at `λ = a`. Every ingredient moves
/// covariantly under a reparametrization `X_R → X_R Z`, `S_j → ZᵀS_jZ` of the
/// block (`a_j` is invariant, `P → ZᵀPZ`, `K_j → Z⁻¹K_jZ⁻ᵀ`), so `uᵀK_ju` and
/// `tr(K_jV)` — and the test — do not depend on the chart the basis factory
/// realized. The base `a_j` puts every penalty on the scale of the data it
/// penalizes, so the result is invariant to rescaling any `S_j` too. When the
/// ranges are complementary (a double penalty: a wiggliness penalty plus a
/// ridge on its null space, in ANY chart) `K_j ∝` the prior covariance of
/// penalty `j` alone, whatever the base; the base only matters for genuinely
/// overlapping penalties.
fn prior_covariance_components(
    penalties: &[Array2<f64>],
    fisher_unprojected: &Array2<f64>,
) -> Result<Vec<Array2<f64>>, VarianceComponentTestUnavailable> {
    let q = fisher_unprojected.nrows();
    let generalized = equilibrated_pseudo_inverse(fisher_unprojected)
        .ok_or(VarianceComponentTestUnavailable::DesignUnavailable)?;
    let mut scaled = Vec::with_capacity(penalties.len());
    let mut precision = Array2::<f64>::zeros((q, q));
    for penalty in penalties {
        let data_scale = trace_product(&generalized.inverse, penalty);
        if !(data_scale.is_finite() && data_scale > 0.0) {
            return Err(VarianceComponentTestUnavailable::NoEstimableDirection);
        }
        let component = penalty / data_scale;
        precision += &component;
        scaled.push(component);
    }
    let (values, vectors) = symmetric_eigh(&precision)?;
    let cutoff = crate::basis::spectral_tolerance(&values);
    if values.iter().any(|&value| value <= cutoff) {
        return Err(VarianceComponentTestUnavailable::UnpenalizedDirections);
    }
    let mut inverse_vectors = vectors.clone();
    for (mut column, &value) in inverse_vectors.axis_iter_mut(Axis(1)).zip(values.iter()) {
        column.iter_mut().for_each(|v| *v /= value);
    }
    let covariance = inverse_vectors.dot(&vectors.t());
    Ok(scaled
        .iter()
        .map(|component| {
            let share = covariance.dot(&component.dot(&covariance));
            0.5 * (&share + &share.t())
        })
        .collect())
}

fn symmetric_eigh(
    matrix: &Array2<f64>,
) -> Result<(Array1<f64>, Array2<f64>), VarianceComponentTestUnavailable> {
    let symmetric = 0.5 * (matrix + &matrix.t());
    strict_symmetric_eigh(&symmetric, Side::Lower)
        .map_err(|_| VarianceComponentTestUnavailable::DesignUnavailable)
}

/// `tr(AB)` for symmetric `A`, `B`.
fn trace_product(a: &Array2<f64>, b: &Array2<f64>) -> f64 {
    (a * b).sum()
}

/// Read the weighted chi-square tail through its own error contract: a
/// relative error of one or more carries no information, EXCEPT for a
/// probability of exactly zero, which the evaluator reports that way when the
/// tail lies below the subnormal range — a resolved "smaller than any
/// representable p-value".
fn resolved_tail(probability: f64, relative_error: f64) -> Result<(f64, f64), VarianceComponentTestUnavailable> {
    if probability.is_nan() || relative_error.is_nan() {
        return Err(VarianceComponentTestUnavailable::TailUnresolved);
    }
    if relative_error >= 1.0 && probability > 0.0 {
        return Err(VarianceComponentTestUnavailable::TailUnresolved);
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
        penalties: Vec<Array2<f64>>,
        scale: VarianceComponentTestScale,
    ) -> Result<VarianceComponentTest, VarianceComponentTestUnavailable> {
        let n = design.nrows();
        let backing = DesignMatrix::Dense(DenseDesignMatrix::from(design.clone()));
        let weights = Array1::<f64>::ones(n);
        let score = y - &design.dot(beta);
        let basis = VarianceComponentTestBasis::new(VarianceComponentTestInput {
            design: &backing,
            beta: beta.view(),
            hessian_weights: weights.view(),
            score_weights: weights.view(),
            score: score.view(),
            scale,
        })?;
        basis
            .test_terms(&[VarianceComponentTermRequest { range, penalties }])
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

    fn ridge(levels: usize) -> Vec<Array2<f64>> {
        vec![Array2::eye(levels)]
    }

    /// A null p-value sample is U(0, 1) in BOTH tails: size within 3 MCSE of
    /// `α` at .10/.05/.01 (an undersized test fails exactly as an oversized one
    /// does), and the Kolmogorov-Smirnov distance below its 1% critical value
    /// `1.628/√m`, which also rules out a point mass at `p = 1`.
    fn assert_uniform(label: &str, mut p_values: Vec<f64>) {
        let m = p_values.len() as f64;
        for alpha in [0.10, 0.05, 0.01] {
            let rate = p_values.iter().filter(|&&p| p <= alpha).count() as f64 / m;
            let mcse = (alpha * (1.0 - alpha) / m).sqrt();
            assert!(
                (rate - alpha).abs() <= 3.0 * mcse,
                "{label}: size {rate} at {alpha} (MCSE {mcse})"
            );
        }
        p_values.sort_by(f64::total_cmp);
        let ks = p_values
            .iter()
            .enumerate()
            .map(|(i, &p)| ((i + 1) as f64 / m - p).max(p - i as f64 / m))
            .fold(0.0_f64, f64::max);
        assert!(ks < 1.628 / m.sqrt(), "{label}: KS distance {ks}");
    }

    /// Six centered Gaussian bumps on `[0, 1]`: a small smooth basis whose
    /// coefficient-index polynomials are the null space of a second-difference
    /// penalty.
    fn bump_basis(x: &[f64]) -> Array2<f64> {
        let k = 6;
        let mut basis = Array2::<f64>::zeros((x.len(), k));
        for (i, &xi) in x.iter().enumerate() {
            for j in 0..k {
                let centre = j as f64 / (k - 1) as f64;
                basis[[i, j]] = (-(xi - centre).powi(2) / (2.0 * 0.15 * 0.15)).exp();
            }
        }
        let means = basis.mean_axis(Axis(0)).unwrap();
        basis - &means
    }

    /// Double penalty on a six-coefficient block: the second-difference
    /// penalty `DᵀD` (rank 4) and the orthogonal projector onto its null space
    /// (rank 2), scaled by `wiggle` and `null`.
    fn double_penalty(wiggle: f64, null: f64) -> Vec<Array2<f64>> {
        let k = 6;
        let mut difference = Array2::<f64>::zeros((k - 2, k));
        for r in 0..k - 2 {
            difference[[r, r]] = 1.0;
            difference[[r, r + 1]] = -2.0;
            difference[[r, r + 2]] = 1.0;
        }
        let bending = difference.t().dot(&difference);
        let index: Vec<f64> = (0..k).map(|j| j as f64 - 2.5).collect();
        let constant = Array1::from_elem(k, 1.0 / (k as f64).sqrt());
        let norm = index.iter().map(|v| v * v).sum::<f64>().sqrt();
        let slope: Array1<f64> = index.iter().map(|v| v / norm).collect();
        let mut null_projector = Array2::<f64>::zeros((k, k));
        for a in 0..k {
            for b in 0..k {
                null_projector[[a, b]] = constant[a] * constant[b] + slope[a] * slope[b];
            }
        }
        vec![wiggle * bending, null * null_projector]
    }

    /// `[1 | z | bump_basis(x)]`, the smooth block at `2..8`.
    fn smooth_design(x: &[f64], z: &[f64]) -> Array2<f64> {
        let n = x.len();
        let mut design = Array2::<f64>::zeros((n, 8));
        design.column_mut(0).fill(1.0);
        for i in 0..n {
            design[[i, 1]] = z[i];
        }
        design.slice_mut(s![.., 2..8]).assign(&bump_basis(x));
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
            ridge(levels),
            VarianceComponentTestScale::Estimated,
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
            ridge(levels),
            VarianceComponentTestScale::Estimated,
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
            ridge(levels),
            VarianceComponentTestScale::Estimated,
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
            VarianceComponentTestScale::Known { dispersion: 1.0 },
            VarianceComponentTestScale::Estimated,
        ] {
            let p_values: Vec<f64> = (0..reps)
                .map(|_| {
                    let y: Array1<f64> = (0..n).map(|i| 1.0 + x[i] + rng.next_normal()).collect();
                    gaussian_test(&design, &y, &beta, 2..2 + levels, ridge(levels), scale)
                        .expect("test runs")
                        .p_value
                })
                .collect();
            assert_uniform(&format!("{scale:?}"), p_values);
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
            ridge(levels),
            VarianceComponentTestScale::Known { dispersion: 1.0 },
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
            ridge(4),
            VarianceComponentTestScale::Known { dispersion: 1.0 },
        )
        .expect_err("no direction survives");
        assert_eq!(reason, VarianceComponentTestUnavailable::NoEstimableDirection);
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
            ridge(6),
            VarianceComponentTestScale::Estimated,
        )
        .expect_err("no residual d.f.");
        assert_eq!(reason, VarianceComponentTestUnavailable::NoResidualDegreesOfFreedom);
    }

    #[test]
    fn double_penalty_null_p_values_are_uniform_for_known_and_estimated_scale() {
        let n = 80;
        let reps = 3000;
        let mut rng = Lcg(91);
        let x: Vec<f64> = (0..n).map(|_| rng.next_uniform().powi(2)).collect();
        let z: Vec<f64> = (0..n).map(|_| rng.next_normal()).collect();
        let design = smooth_design(&x, &z);
        let beta = Array1::<f64>::zeros(design.ncols());
        for scale in [
            VarianceComponentTestScale::Known { dispersion: 1.0 },
            VarianceComponentTestScale::Estimated,
        ] {
            let p_values: Vec<f64> = (0..reps)
                .map(|_| {
                    let y: Array1<f64> = (0..n).map(|i| 0.5 - z[i] + rng.next_normal()).collect();
                    gaussian_test(&design, &y, &beta, 2..8, double_penalty(1.0, 1.0), scale)
                        .expect("test runs")
                        .p_value
                })
                .collect();
            assert_uniform(&format!("double penalty, {scale:?}"), p_values);
        }
    }

    #[test]
    fn rescaling_a_penalty_does_not_move_the_p_value() {
        let n = 60;
        let mut rng = Lcg(17);
        let x: Vec<f64> = (0..n).map(|_| rng.next_uniform()).collect();
        let z: Vec<f64> = (0..n).map(|_| rng.next_normal()).collect();
        let design = smooth_design(&x, &z);
        let y: Array1<f64> = (0..n)
            .map(|i| z[i] + 0.4 * (std::f64::consts::TAU * x[i]).sin() + rng.next_normal())
            .collect();
        let beta = Array1::<f64>::zeros(design.ncols());
        let scale = VarianceComponentTestScale::Estimated;
        let reference =
            gaussian_test(&design, &y, &beta, 2..8, double_penalty(1.0, 1.0), scale).unwrap();
        let rescaled =
            gaussian_test(&design, &y, &beta, 2..8, double_penalty(3.0e4, 2.0e-3), scale).unwrap();
        assert!(
            (rescaled.p_value - reference.p_value).abs() <= 1e-9 * reference.p_value,
            "{reference:?} vs {rescaled:?}"
        );
        assert!((rescaled.statistic - reference.statistic).abs() <= 1e-9 * reference.statistic);
    }

    /// A basis factory hands the test its block after reparametrizing,
    /// constraining and centring it: `X_R Z`, `ZᵀS_jZ` for a non-orthogonal
    /// `Z`, where the double penalty's ranges are still complementary but no
    /// longer Euclidean-orthogonal. The same model in that chart must give the
    /// same statistic and p-value, and a purely linear effect must still be
    /// found there.
    #[test]
    fn reparametrizing_the_block_does_not_move_the_p_value() {
        let n = 200;
        let mut rng = Lcg(31);
        let x: Vec<f64> = (0..n).map(|_| rng.next_uniform()).collect();
        let z: Vec<f64> = (0..n).map(|_| rng.next_normal()).collect();
        let design = smooth_design(&x, &z);
        let y: Array1<f64> = (0..n).map(|i| z[i] + 1.2 * x[i] + rng.next_normal()).collect();
        let beta = Array1::<f64>::zeros(design.ncols());
        let k = 6;
        let mut chart = Array2::<f64>::eye(k);
        for a in 0..k {
            for b in 0..k {
                if a != b {
                    chart[[a, b]] = 0.35 * ((a * k + b) as f64).sin() + if b > a { 0.8 } else { 0.0 };
                }
            }
        }
        let mut reparametrized = design.clone();
        let block = design.slice(s![.., 2..8]).dot(&chart);
        reparametrized.slice_mut(s![.., 2..8]).assign(&block);
        let moved: Vec<Array2<f64>> = double_penalty(1.0, 1.0)
            .iter()
            .map(|penalty| chart.t().dot(&penalty.dot(&chart)))
            .collect();
        for scale in [
            VarianceComponentTestScale::Known { dispersion: 1.0 },
            VarianceComponentTestScale::Estimated,
        ] {
            let reference =
                gaussian_test(&design, &y, &beta, 2..8, double_penalty(1.0, 1.0), scale).unwrap();
            let charted =
                gaussian_test(&reparametrized, &y, &beta, 2..8, moved.clone(), scale).unwrap();
            assert!(
                (charted.p_value - reference.p_value).abs() <= 1e-8 * reference.p_value,
                "{reference:?} vs {charted:?}"
            );
            assert!(
                (charted.statistic - reference.statistic).abs() <= 1e-8 * reference.statistic,
                "{reference:?} vs {charted:?}"
            );
            assert!(charted.p_value < 1e-3, "linear effect missed: {charted:?}");
        }
    }

    #[test]
    fn linear_and_wiggly_effects_are_both_detected() {
        let n = 200;
        let mut rng = Lcg(29);
        let x: Vec<f64> = (0..n).map(|_| rng.next_uniform()).collect();
        let z: Vec<f64> = (0..n).map(|_| rng.next_normal()).collect();
        let design = smooth_design(&x, &z);
        let beta = Array1::<f64>::zeros(design.ncols());
        let noise: Vec<f64> = (0..n).map(|_| rng.next_normal()).collect();
        for (label, effect) in [
            ("linear", Box::new(|x: f64| 1.2 * x) as Box<dyn Fn(f64) -> f64>),
            ("wiggly", Box::new(|x: f64| 0.5 * (std::f64::consts::TAU * x).sin())),
        ] {
            let y: Array1<f64> = (0..n).map(|i| z[i] + effect(x[i]) + noise[i]).collect();
            let test = gaussian_test(
                &design,
                &y,
                &beta,
                2..8,
                double_penalty(1.0, 1.0),
                VarianceComponentTestScale::Estimated,
            )
            .unwrap();
            assert!(test.p_value < 1e-3, "{label}: {test:?}");
        }
    }

    #[test]
    fn a_block_with_unpenalized_directions_is_refused() {
        let n = 50;
        let mut rng = Lcg(41);
        let x: Vec<f64> = (0..n).map(|_| rng.next_uniform()).collect();
        let z: Vec<f64> = (0..n).map(|_| rng.next_normal()).collect();
        let design = smooth_design(&x, &z);
        let y: Array1<f64> = (0..n).map(|_| rng.next_normal()).collect();
        let bending = double_penalty(1.0, 1.0).swap_remove(0);
        // The bending penalty alone leaves the linear part unpenalized, and a
        // block with no penalty at all leaves every direction unpenalized.
        for (label, penalties) in [("bending only", vec![bending]), ("no penalty", Vec::new())] {
            let reason = gaussian_test(
                &design,
                &y,
                &Array1::zeros(design.ncols()),
                2..8,
                penalties,
                VarianceComponentTestScale::Known { dispersion: 1.0 },
            )
            .expect_err(label);
            assert_eq!(
                reason,
                VarianceComponentTestUnavailable::UnpenalizedDirections,
                "{label}"
            );
        }
    }

    #[test]
    fn records_serialize_with_a_typed_status() {
        let record = VarianceComponentTestRecord {
            term: "g".into(),
            coefficient_range: 1..4,
            outcome: VarianceComponentTestOutcome::Unavailable {
                reason: VarianceComponentTestUnavailable::NoEstimableDirection,
            },
        };
        let json = serde_json::to_string(&record).unwrap();
        assert!(json.contains("\"status\":\"unavailable\""), "{json}");
        assert!(json.contains("\"reason\":\"no_estimable_direction\""), "{json}");
        let back: VarianceComponentTestRecord = serde_json::from_str(&json).unwrap();
        assert_eq!(back, record);
    }
}
