//! The CONDITIONAL score covariance `Σ(a) = Var(z | a)` (gam#2766).
//!
//! # What this fixes
//!
//! The marginal-slope probit identity, transcribed from
//! `crate::bms::gradient_paths`, is
//!
//! ```text
//!   z | a ~ N(0, Σ(a)),   η = c(a)·q(t, a) + r(a)ᵀ z
//!   E_z[Φ(−η) | a] = Φ(−q(t, a))    ⟺    c(a) = √(1 + r(a)ᵀ Σ(a) r(a))
//! ```
//!
//! `Σ(a)`, conditional on the marginal-index span. Until this module existed the
//! families supplied it from `marginal_slope_covariance_from_scores` — ONE
//! weighted empirical covariance pooled over every row. Substituting a constant
//! `c̄ = √(1 + rᵀΣ̄r)` into the exact integral leaves
//!
//! ```text
//!   E_z[Φ(−η) | a] = Φ(−q · c̄ / √(1 + rᵀΣ(a)r))
//! ```
//!
//! so the realised marginal index is `q · c̄/c(a)`: a multiplicative,
//! covariate-dependent distortion of the one estimand this family exists to
//! deliver. It is the same failure the `homoskedastic_var` field doc records for
//! the K=1 diagonal (`Φ(q√(1+b²)/√(1+b²v))`), one dimension up.
//!
//! gam#2768 removed the per-coordinate half of it: after that gate every
//! coordinate satisfies `E[ζ_j|a] = 0` and `Var(ζ_j|a) = 1`. What no
//! per-coordinate location-scale map can reach is the OFF-DIAGONAL —
//! `Cov(ζ_j, ζ_k | a)` — and that is what this module models.
//!
//! # The parameterisation, and why this one
//!
//! A covariance-valued regression has to return a positive-definite matrix at
//! every `a`, including rows the fit never saw. Modelling the entries of `Σ(a)`
//! directly does not: nothing keeps a fitted `[[1, ρ(a)], [ρ(a), 1]]` inside
//! `|ρ| < 1`, and one row over the edge makes `c(a)` the square root of a
//! negative number.
//!
//! This module uses Pourahmadi's **modified Cholesky decomposition** (MCD),
//! whose parameters are *unconstrained* — every real value of every parameter
//! yields a positive-definite `Σ(a)`, so extrapolation cannot manufacture an
//! inadmissible covariance:
//!
//! ```text
//!   T(a) Σ(a) T(a)ᵀ = D(a),
//!   T(a) unit lower triangular with  T[j][k] = −φ_{jk}(a)  (k < j),
//!   D(a) = diag(d_0(a), …, d_{K−1}(a)),   log d_j(a) = γ_jᵀ A(a).
//! ```
//!
//! Read forwards this is a triangular system of ordinary regressions, which is
//! exactly why it is the right object here — it is the SAME shape as the
//! machinery gam#2768 already ships, applied one coordinate at a time:
//!
//! ```text
//!   ζ_j = Σ_{k<j} φ_{jk}(a)·ζ_k + ε_j,     Var(ε_j | a) = d_j(a)
//! ```
//!
//! the `φ` stage being a weighted ridge (like the conditional MEAN stage) and
//! the `d` stage a log-linear variance fit (the conditional VARIANCE stage). The
//! reconstruction `Σ(a) = T(a)⁻¹ D(a) T(a)⁻ᵀ` is one forward substitution, and
//! `L(a) = T(a)⁻¹ D(a)^{1/2}` is its exact Cholesky factor — which is precisely
//! the `Σ = L Lᵀ` shape [`MarginalSlopeCovariance::low_rank`] already admits, so
//! the row program's quadratic forms stay exact sums of squares and no runtime
//! eigendecomposition or PSD tolerance appears anywhere on this path.
//!
//! `log` d rather than a linear `d` with a floor (the shape gam#2768 used for
//! the K=1 variance) because a floor is a non-differentiable clamp that can and
//! does bind, whereas `exp` is positive by construction on the whole real line.
//!
//! # What triggers it
//!
//! One robust Rao score test per score PAIR, on the same centred marginal-index
//! span the gam#2768 gate uses and at the same level: `u_i = ζ_ij·ζ_ik − mean`
//! against `ã(C)`. That statistic tests exactly the sentence this issue is
//! titled with — "the covariance between two scores varies" — and nothing else.
//! If no pair fires, this module returns `None` and the caller keeps the pooled
//! `Σ̄` object it already built, byte for byte.
//!
//! Once a pair HAS fired, every stage of the MCD is fitted honestly with its own
//! gate: a `φ_{jk}` becomes `a`-varying only if its own interaction test fires,
//! a `log d_j` becomes `a`-varying only if its own Breusch-Pagan test fires.
//! That ordering — escalate on the thing the issue names, then fit the escalated
//! model without further hedging — is the one gam#2768 already uses (a
//! pure-variance trigger there still fits the conditional mean).
//!
//! # Extrapolation
//!
//! Each fitted linear predictor (`φ_{jk}(a)` and `log d_j(a)`) is clamped at
//! evaluation to the range it took over the TRAINING rows. The bound is the
//! data's, not a constant: a linear predictor is identified only on the range
//! the sample explored, and holding the boundary value beyond it is the same
//! monotone-extrapolation contract [`crate::bms::LatentZRankIntCalibration`]
//! already states for out-of-support scores. Without it a predict row far
//! outside the training hull could return an arbitrarily large `Σ(a)` from a
//! model that had no evidence there.
//!
//! # What `Σ(a)` is a moment OF, and why the ordering with gam#2768 matters
//!
//! The estimated object is the conditional SECOND CENTRAL MOMENT about the
//! weighted GLOBAL score mean, `E[(z − z̄)(z − z̄)ᵀ | a]` — which is what makes
//! the no-escalation limit of this model exactly the pooled
//! `marginal_slope_covariance_from_scores` object it refines, rather than
//! something merely close to it.
//!
//! That equals `Var(z | a)` precisely when `E[z | a]` is constant. It is, by the
//! time this runs: the gam#2768 per-coordinate gate is sequenced FIRST, and it
//! either removes a detected conditional mean (`ζ = (z − m(a))/√v(a)`) or
//! certifies at the same `α` that there is none to remove. Running the two in
//! the other order would be wrong in both directions — this model would absorb
//! mean structure into a "covariance", and the mean gate would then be
//! correcting an axis whose scale had already moved.
//!
//! # Numerical edge cases
//!
//! * **Degenerate (collinear) scores.** A rank-deficient score geometry is a
//!   real, expected input — [`MarginalSlopeCovariance::full`] says so and admits
//!   the exactly-singular matrix it produces. Here it lands as a zero innovation
//!   variance, which `log d` cannot represent, so the innovation is floored at
//!   the SAME band that admission clamps a numerically-zero eigenvalue inside
//!   (`128·K·ε·max weighted second moment`). The result is positive definite
//!   rather than singular, which is strictly better for the `√(1 + rᵀΣr)` that
//!   consumes it.
//! * **Extreme magnitudes.** Every stage checks finiteness of what it produces
//!   and refuses with the offending value rather than propagating a `NaN`: a
//!   score scale that overflows `ε²` overflows the pooled estimator too, and
//!   this path says so at the point it happens.
//! * **An ill-conditioned conditioning span.** A penalised-spline marginal index
//!   is routinely rank-deficient. Every regression here — the couplings and the
//!   Fisher-scoring step for the innovation alike — goes through the same
//!   relative, column-scaled Tikhonov primitive the gam#2768 stages use, so the
//!   two degrade identically instead of one of them having its own normal
//!   equations.
//! * **Parallel and GPU row lanes.** `Σ(a_i)` is a per-row CONSTANT, not a
//!   function of `β`, so every existing derivative formula holds verbatim and
//!   nothing about the chain rule changes. Each rayon chunk binds its own row
//!   workspace, and the survival GPU row kernel already carried `cov_ones` per
//!   row, so the conditional lane needed no new device state.
//!
//! # `K = 1` is deliberately out of scope
//!
//! At `K = 1` there is no off-diagonal, and `Var(z|a)` is gam#2768's
//! per-coordinate branch. Running a second, differently parameterised
//! conditional-variance model on top of the score that branch has already
//! standardised would double-correct it. [`ConditionalScoreCovariance::fit`]
//! therefore returns `None` at `K < 2`, and the K=1 path is bit-for-bit
//! unchanged.

use super::{
    AUTO_Z_CONDITIONAL_RAO_ALPHA, AUTO_Z_CONDITIONAL_RIDGE_REL, MarginalSlopeCovariance,
    build_intercept_basis, robust_conditional_score_pvalue,
};
use ndarray::{Array1, Array2, ArrayView1, ArrayView2};
use serde::{Deserialize, Serialize};

/// Damped Fisher-scoring iterations allowed for one log-linear innovation
/// variance. The objective is strictly concave and bounded above and every
/// accepted step strictly ascends it, so the loop terminates on its own; this
/// cap exists to bound it, not to select an answer, and a run that reaches it is
/// a refusal rather than a truncation. Measured on the module's own fixtures the
/// hardest fit converges in 22 accepted steps, with the gain shrinking by about
/// 4.5x per step — from which 64 is roughly 40 orders of magnitude of headroom.
const LOG_INNOVATION_MAX_ITERATIONS: usize = 64;

/// One coordinate of the modified Cholesky decomposition.
///
/// `autoregression[k]` (for `k < j`) holds the coefficients of `φ_{jk}(a)` and
/// `log_innovation` the coefficients of `log d_j(a)`, both over the
/// intercept-augmented basis `A = [1 | a]`. A coefficient vector of length `1`
/// is a CONSTANT — the stage's own gate did not fire — and one of length
/// `1 + basis_ncols` is `a`-varying. The two lengths are the only encoding of
/// "this stage varies"; there is no separate flag that could disagree with the
/// coefficients.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct ConditionalScoreCoordinate {
    /// `φ_{jk}(a)` for `k = 0 … j−1`, in `k` order. Empty for `j = 0`.
    pub autoregression: Vec<Vec<f64>>,
    /// `log d_j(a)`.
    pub log_innovation: Vec<f64>,
    /// Training range `[min, max]` of each `φ_{jk}(a)` linear predictor, in the
    /// same order as `autoregression`. Length-1 (constant) stages still carry
    /// their degenerate range so evaluation has one code path.
    pub autoregression_range: Vec<[f64; 2]>,
    /// Training range `[min, max]` of the `log d_j(a)` linear predictor.
    pub log_innovation_range: [f64; 2],
}

/// The fitted conditional score covariance `Σ(a) = Var(z | a)`.
///
/// Evaluation is [`Self::factor_into`], which writes the exact lower-triangular
/// Cholesky factor `L(a)` with `Σ(a) = L(a)·L(a)ᵀ`.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct ConditionalScoreCovariance {
    /// Number of latent-score coordinates `K`. Always `≥ 2`.
    pub score_dim: usize,
    /// Number of marginal-design columns in `a` (EXCLUDING the leading
    /// intercept). A predict-time conditioning block must present exactly this
    /// many columns.
    pub basis_ncols: usize,
    /// Per-coordinate MCD blocks, in coordinate order. Length `score_dim`.
    pub coordinates: Vec<ConditionalScoreCoordinate>,
    /// Weighted mean of each raw score column at fit time. `Σ(a)` is the second
    /// CENTRAL moment about this vector, which is what makes the no-fire limit
    /// of this model the pooled `marginal_slope_covariance_from_scores` object
    /// it replaces.
    pub score_mean: Vec<f64>,
    /// The pair-wise Rao p-values that decided the escalation, `(j, k, p)` with
    /// `j < k`. Diagnostic; carried so a fit can say WHY it escalated.
    pub pair_pvalues: Vec<(usize, usize, f64)>,
}

/// The score-covariance geometry a marginal-slope fit consumes, ROW BY ROW.
///
/// One object with two states, because every consumer wants the same thing —
/// "the covariance at this row" — and only the fit's own gate decides which
/// state it is in:
///
/// * `Pooled` — the single weighted empirical `Σ̄` from
///   `marginal_slope_covariance_from_scores`. Every row returns the same
///   object, so a fit that does not escalate is bit-for-bit what it was before
///   gam#2766.
/// * conditional — a materialised `Σ(a_i)` per row, produced by
///   [`ConditionalScoreCovariance::row_covariances`]. The pooled object is
///   RETAINED alongside it: it is still the fit's summary statistic (it is what
///   the on-disk contract carries and what a diagnostic reports), and keeping it
///   means no caller has to decide between "the covariance" and "the covariance
///   here".
///
/// The row lane is an index, not a branch on a model: `at_row` is one match on
/// an `Option` and one slice index, so the conditional path costs the hot loop
/// nothing beyond the indirection.
#[derive(Clone)]
pub struct ScoreCovarianceField {
    pooled: MarginalSlopeCovariance,
    per_row: Option<std::sync::Arc<Vec<MarginalSlopeCovariance>>>,
    model: Option<std::sync::Arc<ConditionalScoreCovariance>>,
}

impl std::fmt::Debug for ScoreCovarianceField {
    /// Summarises rather than dumps. A derived `Debug` on a conditional field
    /// prints one `K × K` matrix PER ROW, which on a biobank-scale fit is a
    /// multi-gigabyte line the first time anything formats a family.
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        formatter
            .debug_struct("ScoreCovarianceField")
            .field("pooled", &self.pooled)
            .field("materialised_rows", &self.materialised_rows())
            .field("conditional_model", &self.model)
            .finish()
    }
}

impl PartialEq for ScoreCovarianceField {
    /// Compares the geometry, not the materialised stack: two fields built from
    /// the same pooled covariance and the same conditional model ARE the same
    /// field, and the per-row stack is a pure function of the two.
    fn eq(&self, other: &Self) -> bool {
        self.pooled == other.pooled && self.model == other.model
    }
}

impl From<MarginalSlopeCovariance> for ScoreCovarianceField {
    fn from(pooled: MarginalSlopeCovariance) -> Self {
        Self {
            pooled,
            per_row: None,
            model: None,
        }
    }
}

impl ScoreCovarianceField {
    /// The pooled, row-invariant field. This is the pre-gam#2766 object.
    pub fn pooled(pooled: MarginalSlopeCovariance) -> Self {
        Self::from(pooled)
    }

    /// Materialise `Σ(a_i)` for every row of `a_block`, keeping `pooled` as the
    /// fit's summary. Refuses a model whose score dimension disagrees with the
    /// pooled covariance's, because the two are the same `K` by construction and
    /// a mismatch means the caller paired a field with the wrong fit.
    pub fn conditional(
        pooled: MarginalSlopeCovariance,
        model: ConditionalScoreCovariance,
        a_block: ArrayView2<'_, f64>,
    ) -> Result<Self, String> {
        if model.score_dim != pooled.dim() {
            return Err(format!(
                "conditional score covariance is K={} but the pooled covariance is K={}",
                model.score_dim,
                pooled.dim()
            ));
        }
        let per_row = model.row_covariances(a_block)?;
        Ok(Self {
            pooled,
            per_row: Some(std::sync::Arc::new(per_row)),
            model: Some(std::sync::Arc::new(model)),
        })
    }

    /// The covariance at `row`.
    #[inline(always)]
    pub fn at_row(&self, row: usize) -> &MarginalSlopeCovariance {
        match &self.per_row {
            None => &self.pooled,
            Some(stack) => &stack[row],
        }
    }

    /// The fit's pooled summary covariance, whatever the field's state.
    #[inline]
    pub fn pooled_covariance(&self) -> &MarginalSlopeCovariance {
        &self.pooled
    }

    /// `K`.
    #[inline]
    pub fn dim(&self) -> usize {
        self.pooled.dim()
    }

    /// Whether the covariance varies by row.
    #[inline]
    pub fn is_conditional(&self) -> bool {
        self.per_row.is_some()
    }

    /// The fitted conditional model, when the field carries one. This is the
    /// object persistence and prediction need; the materialised stack is a
    /// training-row cache and is never saved.
    #[inline]
    pub fn model(&self) -> Option<&ConditionalScoreCovariance> {
        self.model.as_deref()
    }

    /// Rows the field was materialised for, when it is conditional. A caller
    /// that indexes past this has mixed two samples.
    #[inline]
    pub fn materialised_rows(&self) -> Option<usize> {
        self.per_row.as_ref().map(|stack| stack.len())
    }
}

/// Affine evaluation `coeffs·[1 | a]`, clamped to the training range of that
/// linear predictor. A length-1 coefficient vector is the constant stage.
#[inline]
fn clamped_affine(coeffs: &[f64], range: &[f64; 2], a_row: ArrayView1<'_, f64>) -> f64 {
    let mut acc = coeffs[0];
    for (coefficient, &value) in coeffs[1..].iter().zip(a_row.iter()) {
        acc += coefficient * value;
    }
    acc.clamp(range[0], range[1])
}

impl ConditionalScoreCovariance {
    /// The lower-triangular Cholesky factor `L(a)` of `Σ(a)`, written into
    /// `factor` (shape `K × K`, fully overwritten including the strict upper
    /// triangle).
    ///
    /// `T(a)` is unit lower triangular with `T[j][k] = −φ_{jk}(a)`, so its
    /// inverse `U = T⁻¹` is unit lower triangular and obtained by one forward
    /// substitution:
    ///
    /// ```text
    ///   U[j][j] = 1,   U[j][k] = φ_{jk}(a) + Σ_{m=k+1}^{j−1} φ_{jm}(a)·U[m][k]
    /// ```
    ///
    /// and `L = U·D^{1/2}`, i.e. `L[j][k] = U[j][k]·√d_k(a)`. Positive
    /// definiteness needs only `d_k(a) > 0`, which `exp` guarantees.
    pub fn factor_into(
        &self,
        a_row: ArrayView1<'_, f64>,
        factor: &mut Array2<f64>,
    ) -> Result<(), String> {
        let k = self.score_dim;
        if a_row.len() != self.basis_ncols {
            return Err(format!(
                "conditional score covariance expects {} basis columns, got {}",
                self.basis_ncols,
                a_row.len()
            ));
        }
        if factor.dim() != (k, k) {
            return Err(format!(
                "conditional score covariance factor must be {k}x{k}, got {}x{}",
                factor.nrows(),
                factor.ncols()
            ));
        }
        factor.fill(0.0);
        for j in 0..k {
            let block = &self.coordinates[j];
            // U[j][*] by forward substitution, written in place into row j.
            factor[[j, j]] = 1.0;
            for k_index in 0..j {
                let phi = clamped_affine(
                    &block.autoregression[k_index],
                    &block.autoregression_range[k_index],
                    a_row,
                );
                if !phi.is_finite() {
                    return Err(format!(
                        "conditional score covariance autoregression ({j},{k_index}) is not finite"
                    ));
                }
                let mut value = phi;
                for m in (k_index + 1)..j {
                    let phi_jm = clamped_affine(
                        &block.autoregression[m],
                        &block.autoregression_range[m],
                        a_row,
                    );
                    value += phi_jm * factor[[m, k_index]];
                }
                factor[[j, k_index]] = value;
            }
        }
        // Scale column `k` by √d_k(a).
        for column in 0..k {
            let block = &self.coordinates[column];
            let log_d = clamped_affine(
                &block.log_innovation,
                &block.log_innovation_range,
                a_row,
            );
            let scale = (0.5 * log_d).exp();
            if !(scale.is_finite() && scale > 0.0) {
                return Err(format!(
                    "conditional score covariance innovation {column} evaluated to log d = {log_d}"
                ));
            }
            for row in column..k {
                factor[[row, column]] *= scale;
            }
        }
        Ok(())
    }

    /// One admitted [`MarginalSlopeCovariance`] per row of `a_block`, in the
    /// `Σ = L Lᵀ` low-rank representation whose quadratic forms are exact sums
    /// of squares.
    ///
    /// Materialising the whole stack once is deliberate. The row program
    /// evaluates `rᵀΣ(a_i)r` inside the inner Newton loop, many times per row
    /// per outer step; re-running the forward substitution there would put an
    /// `O(K³)` triangular solve in the hot path to save `K²` doubles of storage.
    /// The stack costs `K` times the score matrix `z` the family already holds.
    pub fn row_covariances(
        &self,
        a_block: ArrayView2<'_, f64>,
    ) -> Result<Vec<MarginalSlopeCovariance>, String> {
        if a_block.ncols() != self.basis_ncols {
            return Err(format!(
                "conditional score covariance expects {} basis columns, got {}",
                self.basis_ncols,
                a_block.ncols()
            ));
        }
        let mut factor = Array2::<f64>::zeros((self.score_dim, self.score_dim));
        let mut out = Vec::with_capacity(a_block.nrows());
        for row in 0..a_block.nrows() {
            self.factor_into(a_block.row(row), &mut factor)?;
            out.push(MarginalSlopeCovariance::low_rank(factor.clone())?);
        }
        Ok(out)
    }

    /// Fit the conditional covariance, or decide there is nothing to fit.
    ///
    /// `scores` are the latent scores the row kernel will actually see — i.e.
    /// AFTER the gam#2768 per-coordinate calibration, because `Σ` is the
    /// covariance of the axis the likelihood integrates over. Returns `None`
    /// when `K < 2`, when the basis is degenerate, or when no score pair's
    /// covariance is significantly non-constant on `a_block`; in every one of
    /// those the caller must keep the pooled object unchanged.
    pub fn fit(
        scores: ArrayView2<'_, f64>,
        weights: ArrayView1<'_, f64>,
        a_block: ArrayView2<'_, f64>,
    ) -> Result<Option<Self>, String> {
        let (n, k) = scores.dim();
        let p = a_block.ncols();
        if k < 2 || p == 0 || n == 0 {
            return Ok(None);
        }
        if weights.len() != n || a_block.nrows() != n {
            return Err(format!(
                "conditional score covariance length mismatch: rows={n}, weights={}, basis rows={}",
                weights.len(),
                a_block.nrows()
            ));
        }
        let total_weight = weights.iter().copied().sum::<f64>();
        if !(total_weight.is_finite() && total_weight > 0.0) {
            return Ok(None);
        }
        if scores.iter().chain(a_block.iter()).any(|v| !v.is_finite()) {
            return Ok(None);
        }

        // Centre on the weighted score mean, so the no-escalation limit of this
        // model is exactly `marginal_slope_covariance_from_scores`.
        let mut score_mean = vec![0.0_f64; k];
        for coordinate in 0..k {
            score_mean[coordinate] = scores
                .column(coordinate)
                .iter()
                .zip(weights.iter())
                .map(|(&value, &weight)| weight * value)
                .sum::<f64>()
                / total_weight;
        }
        let mut centered = Array2::<f64>::zeros((n, k));
        for row in 0..n {
            for coordinate in 0..k {
                centered[[row, coordinate]] = scores[[row, coordinate]] - score_mean[coordinate];
            }
        }

        // Centre the basis columns: the score tests are about conditional
        // structure BEYOND the global level, and a constant column collapses to
        // ~0 and is dropped by the statistic's own pseudo-inverse rank.
        let mut a_centered = a_block.to_owned();
        for column in 0..p {
            let column_mean = a_block
                .column(column)
                .iter()
                .zip(weights.iter())
                .map(|(&value, &weight)| weight * value)
                .sum::<f64>()
                / total_weight;
            a_centered
                .column_mut(column)
                .mapv_inplace(|value| value - column_mean);
        }

        // The escalation gate: one robust Rao score test per PAIR, on the
        // cross-product residual. This is the statistic for the sentence the
        // issue is titled with and nothing wider.
        let mut pair_pvalues = Vec::new();
        let mut any_pair_fires = false;
        for left in 0..k {
            for right in (left + 1)..k {
                let products: Vec<f64> = (0..n)
                    .map(|row| centered[[row, left]] * centered[[row, right]])
                    .collect();
                let mean = products
                    .iter()
                    .zip(weights.iter())
                    .map(|(&value, &weight)| weight * value)
                    .sum::<f64>()
                    / total_weight;
                let residual: Vec<f64> = products.iter().map(|&value| value - mean).collect();
                let p_value =
                    robust_conditional_score_pvalue(a_centered.view(), &residual, weights)?;
                if let Some(p_value) = p_value {
                    pair_pvalues.push((left, right, p_value));
                    any_pair_fires |= p_value < AUTO_Z_CONDITIONAL_RAO_ALPHA;
                }
            }
        }
        if !any_pair_fires {
            return Ok(None);
        }

        let basis = build_intercept_basis(a_block);
        // The band inside which an innovation variance is indistinguishable
        // from zero. Same form and same coefficient as the PSD band
        // `MarginalSlopeCovariance::full` admits an eigenvalue inside
        // (`128·k·ε·max|λ̂|`), against the largest weighted second moment of the
        // centred scores -- the scale a variance of this sample can have. It is
        // derived from the sample and the floating-point type, and it exists
        // because `log 0` is not a number, not because a number had to be
        // picked.
        let score_scale = (0..k)
            .map(|coordinate| {
                (0..n)
                    .map(|row| {
                        weights[row] * centered[[row, coordinate]] * centered[[row, coordinate]]
                    })
                    .sum::<f64>()
                    / total_weight
            })
            .fold(0.0_f64, f64::max);
        let innovation_floor =
            (128.0 * k as f64 * f64::EPSILON * score_scale).max(f64::MIN_POSITIVE);
        // One MCD block per coordinate, in order: the triangular structure means
        // coordinate `j` regresses on the CENTRED scores below it, not on their
        // innovations, so no state carries between iterations.
        let mut coordinates = Vec::with_capacity(k);
        for j in 0..k {
            let (autoregression, autoregression_range, residual) =
                fit_autoregression(&centered, j, basis.view(), a_centered.view(), weights)?;
            let (log_innovation, log_innovation_range) = fit_log_innovation(
                &residual,
                basis.view(),
                a_centered.view(),
                weights,
                total_weight,
                innovation_floor,
            )?;
            coordinates.push(ConditionalScoreCoordinate {
                autoregression,
                log_innovation,
                autoregression_range,
                log_innovation_range,
            });
        }

        Ok(Some(Self {
            score_dim: k,
            basis_ncols: p,
            coordinates,
            score_mean,
            pair_pvalues,
        }))
    }
}

/// Fit `ζ_j = Σ_{k<j} φ_{jk}(a)·ζ_k + ε_j`, returning the coefficient blocks,
/// their realised training ranges, and the residual `ε_j`.
///
/// Each `φ_{jk}` starts constant. A robust Rao score test of the constant-fit
/// residual against `ã ⊙ ζ_k` — the exact LM statistic for "the coefficient of
/// `ζ_k` depends on `a`" — decides whether that one coefficient is promoted to
/// the full basis. Promotion is per `(j, k)`, so a model with one varying
/// coupling does not spend `p` parameters on the couplings that are constant.
fn fit_autoregression(
    centered: &Array2<f64>,
    j: usize,
    basis: ArrayView2<'_, f64>,
    a_centered: ArrayView2<'_, f64>,
    weights: ArrayView1<'_, f64>,
) -> Result<(Vec<Vec<f64>>, Vec<[f64; 2]>, Vec<f64>), String> {
    let n = centered.nrows();
    let response: Vec<f64> = (0..n).map(|row| centered[[row, j]]).collect();
    if j == 0 {
        return Ok((Vec::new(), Vec::new(), response));
    }

    // Stage 1 — constant couplings.
    let mut constant_design = Array2::<f64>::zeros((n, j));
    for row in 0..n {
        for k_index in 0..j {
            constant_design[[row, k_index]] = centered[[row, k_index]];
        }
    }
    let (constant_coeffs, constant_fitted) =
        weighted_ridge_columns(constant_design.view(), &response, weights)?;
    let constant_residual: Vec<f64> = (0..n)
        .map(|row| response[row] - constant_fitted[row])
        .collect();

    // Stage 2 — per-coupling interaction tests on that residual.
    let mut varying = vec![false; j];
    for k_index in 0..j {
        let interaction: Vec<f64> = (0..n)
            .map(|row| constant_residual[row] * centered[[row, k_index]])
            .collect();
        if let Some(p_value) =
            robust_conditional_score_pvalue(a_centered, &interaction, weights)?
        {
            varying[k_index] = p_value < AUTO_Z_CONDITIONAL_RAO_ALPHA;
        }
    }
    if varying.iter().all(|fires| !fires) {
        let coeffs: Vec<Vec<f64>> = (0..j).map(|k| vec![constant_coeffs[k]]).collect();
        let ranges: Vec<[f64; 2]> = (0..j)
            .map(|k| [constant_coeffs[k], constant_coeffs[k]])
            .collect();
        return Ok((coeffs, ranges, constant_residual));
    }

    // Stage 3 — refit with the promoted couplings expanded over `[1 | a]`.
    let basis_width = basis.ncols();
    let mut widths = Vec::with_capacity(j);
    let mut total_columns = 0usize;
    for &fires in varying.iter() {
        let width = if fires { basis_width } else { 1 };
        widths.push(width);
        total_columns += width;
    }
    let mut design = Array2::<f64>::zeros((n, total_columns));
    let mut offset = 0usize;
    for (k_index, &width) in widths.iter().enumerate() {
        for row in 0..n {
            let score = centered[[row, k_index]];
            if width == 1 {
                design[[row, offset]] = score;
            } else {
                for column in 0..basis_width {
                    design[[row, offset + column]] = score * basis[[row, column]];
                }
            }
        }
        offset += width;
    }
    let (coeffs, fitted) = weighted_ridge_columns(design.view(), &response, weights)?;
    let residual: Vec<f64> = (0..n).map(|row| response[row] - fitted[row]).collect();

    // Unpack into per-coupling blocks and record each one's realised range.
    let mut blocks = Vec::with_capacity(j);
    let mut ranges = Vec::with_capacity(j);
    let mut offset = 0usize;
    for &width in widths.iter() {
        let block: Vec<f64> = coeffs[offset..offset + width].to_vec();
        let range = linear_predictor_range(&block, basis, weights);
        blocks.push(block);
        ranges.push(range);
        offset += width;
    }
    Ok((blocks, ranges, residual))
}

/// Fit `log d(a) = γᵀ[1 | a]` for the innovation `ε` by exact Fisher scoring of
/// the Gaussian log-linear variance model, gated by a Breusch-Pagan score test.
///
/// With `ε_i ~ N(0, d_i)`, `d_i = exp(A_iᵀγ)`, the weighted log-likelihood score
/// and Fisher information are
///
/// ```text
///   s(γ) = ½ Σ_i w_i A_i (ε_i²/d_i − 1),     I(γ) = ½ Σ_i w_i A_i A_iᵀ
/// ```
///
/// so the scoring step is `Δγ = (Σ w A Aᵀ)⁻¹ Σ w A (ε²/d − 1)`: the halves
/// cancel and the information does not depend on the response. That is why this
/// is a short exact loop rather than the log-of-squares linear regression
/// (Harvey's two-step), whose intercept carries the `E[log χ²₁] = −1.2704` bias
/// and whose slopes are inefficient.
///
/// `Σ w A Aᵀ` is the EXPECTED information, not the observed one — the observed
/// Hessian carries an `ε²/d` weight — so the undamped step overshoots and the
/// iteration is monotone in the log-likelihood but not in any norm of the step.
/// The loop is therefore a line-searched ascent, not a bare Newton iteration;
/// see the body for the measurement that forced it.
fn fit_log_innovation(
    residual: &[f64],
    basis: ArrayView2<'_, f64>,
    a_centered: ArrayView2<'_, f64>,
    weights: ArrayView1<'_, f64>,
    total_weight: f64,
    innovation_floor: f64,
) -> Result<(Vec<f64>, [f64; 2]), String> {
    let n = residual.len();
    let raw = residual
        .iter()
        .zip(weights.iter())
        .map(|(&value, &weight)| weight * value * value)
        .sum::<f64>()
        / total_weight;
    if !raw.is_finite() {
        return Err(format!(
            "conditional score covariance innovation variance is {raw}, which no log-linear \
             variance model can represent"
        ));
    }
    let homoskedastic = raw.max(innovation_floor);
    let constant = vec![homoskedastic.ln()];
    let constant_range = [constant[0], constant[0]];

    // A direction the sample does not distinguish from a point. Collinear
    // scores are a REAL and expected input -- `MarginalSlopeCovariance::full`
    // says so, and admits an eigenvalue anywhere inside its own solver band as
    // an exact zero -- so this cannot be a refusal. It is instead the one place
    // the log parameterisation needs a floor, because `log 0` is not a number:
    // the innovation is held at the same band `full` clamps inside, which
    // contributes no spread at the working precision and keeps `Σ(a)` positive
    // definite rather than singular. Below it there is nothing to model, so the
    // Breusch-Pagan stage is skipped as well.
    if raw <= innovation_floor {
        return Ok((constant, constant_range));
    }

    // Breusch-Pagan: does the innovation variance depend on `a` at all?
    let bp_residual: Vec<f64> = residual
        .iter()
        .map(|&value| value * value - homoskedastic)
        .collect();
    let fires = robust_conditional_score_pvalue(a_centered, &bp_residual, weights)?
        .is_some_and(|p_value| p_value < AUTO_Z_CONDITIONAL_RAO_ALPHA);
    if !fires {
        return Ok((constant, constant_range));
    }

    let width = basis.ncols();
    let mut gamma = vec![0.0_f64; width];
    gamma[0] = homoskedastic.ln();

    // The Gaussian log-likelihood of the log-linear variance model, up to a
    // constant: `ℓ(γ) = −½ Σ w (A_iᵀγ + ε_i²·exp(−A_iᵀγ))`. Non-finite is a
    // legitimate answer — an iterate that drives `exp(−A_iᵀγ)` past the
    // representable range is simply not an ascent step, and the line search
    // below halves past it rather than the whole fit refusing.
    let log_likelihood = |coefficients: &[f64]| -> Option<f64> {
        let mut total = 0.0_f64;
        for row in 0..n {
            let weight = weights[row];
            if !(weight > 0.0) {
                continue;
            }
            let mut linear = 0.0;
            for column in 0..width {
                linear += coefficients[column] * basis[[row, column]];
            }
            total -= 0.5 * weight * (linear + residual[row] * residual[row] * (-linear).exp());
        }
        total.is_finite().then_some(total)
    };

    // Fisher scoring with a step-halving line search.
    //
    // The scoring step `Δγ = (Σ w A Aᵀ)⁻¹ Σ w A (ε²/d − 1)` is literally a
    // weighted least-squares fit of `u_i = ε_i²/d_i − 1` on `A`, so it is taken
    // with the same regularised primitive the coupling stage uses; the two then
    // degrade identically on a rank-deficient span instead of one of them having
    // its own hand-rolled normal equations.
    //
    // The line search is not optional. `Σ w A Aᵀ` is the EXPECTED information;
    // the observed one carries an `ε²/d` weight, so the undamped step
    // systematically overshoots and the iteration is not monotone in any norm of
    // the step. Measured on a cubic span: the raw step's `max|A·Δγ|` went
    // 3.21 → 1.04 → 1.66, and an earlier "stop when the step stops shrinking"
    // rule took that third reading as convergence and returned a `log d` short of
    // the optimum by 4.5 nats. Damping restores monotone ASCENT, which is the
    // property a strictly concave objective actually supports, and the same run
    // then converges in 22 accepted steps.
    let mut current = log_likelihood(&gamma).ok_or_else(|| {
        format!(
            "conditional score covariance log-variance seed log d = {} is not evaluable",
            gamma[0]
        )
    })?;
    let mut converged = false;
    for _ in 0..LOG_INNOVATION_MAX_ITERATIONS {
        let mut deviation = Array1::<f64>::zeros(n);
        for row in 0..n {
            let mut linear = 0.0;
            for column in 0..width {
                linear += gamma[column] * basis[[row, column]];
            }
            deviation[row] = residual[row] * residual[row] * (-linear).exp() - 1.0;
        }
        if !deviation.iter().all(|value| value.is_finite()) {
            return Err(
                "conditional score covariance log-variance iterate left the representable range"
                    .to_string(),
            );
        }
        let (step, _) = weighted_ridge_columns(
            basis,
            deviation.as_slice().expect("deviation is standard layout"),
            weights,
        )?;
        // Halve until the step ascends. Two derived exits and no chosen
        // tolerance: a trial that no longer moves `γ` at all in floating point
        // cannot ascend, and a gain below the log-likelihood's own last place is
        // not a gain.
        let mut trial = 1.0_f64;
        let mut accepted: Option<(Vec<f64>, f64)> = None;
        while trial > 0.0 {
            let candidate: Vec<f64> = gamma
                .iter()
                .zip(step.iter())
                .map(|(value, direction)| value + trial * direction)
                .collect();
            if candidate
                .iter()
                .zip(gamma.iter())
                .all(|(new, old)| new.to_bits() == old.to_bits())
            {
                break;
            }
            if let Some(value) = log_likelihood(&candidate)
                && value > current
            {
                accepted = Some((candidate, value));
                break;
            }
            trial *= 0.5;
        }
        let Some((candidate, value)) = accepted else {
            // No representable step ascends: this IS the optimum to round-off.
            converged = true;
            break;
        };
        let gain = value - current;
        gamma = candidate;
        current = value;
        if gain <= f64::EPSILON * (1.0 + current.abs()) {
            converged = true;
            break;
        }
    }
    if !converged {
        return Err(format!(
            "conditional score covariance log-variance scoring did not converge in \
             {LOG_INNOVATION_MAX_ITERATIONS} damped Fisher steps"
        ));
    }
    let range = linear_predictor_range(&gamma, basis, weights);
    Ok((gamma, range))
}

/// `[min, max]` of `coeffs·[1 | a]` over the training rows that carry weight.
/// A constant stage returns its own degenerate range, so evaluation has exactly
/// one code path.
fn linear_predictor_range(
    coeffs: &[f64],
    basis: ArrayView2<'_, f64>,
    weights: ArrayView1<'_, f64>,
) -> [f64; 2] {
    if coeffs.len() == 1 {
        return [coeffs[0], coeffs[0]];
    }
    let mut low = f64::INFINITY;
    let mut high = f64::NEG_INFINITY;
    for row in 0..basis.nrows() {
        if !(weights[row] > 0.0) {
            continue;
        }
        let mut linear = 0.0;
        for column in 0..coeffs.len() {
            linear += coeffs[column] * basis[[row, column]];
        }
        low = low.min(linear);
        high = high.max(linear);
    }
    if !(low.is_finite() && high.is_finite() && low <= high) {
        return [coeffs[0], coeffs[0]];
    }
    [low, high]
}

/// Weighted ridge of `response` on `design` with the same relative,
/// column-scaled Tikhonov penalty the conditional location-scale stages use.
/// Returns the coefficients and the fitted values.
fn weighted_ridge_columns(
    design: ArrayView2<'_, f64>,
    response: &[f64],
    weights: ArrayView1<'_, f64>,
) -> Result<(Vec<f64>, Vec<f64>), String> {
    let width = design.ncols();
    let mut penalty = Array2::<f64>::zeros((width, width));
    for column in 0..width {
        let diagonal = design
            .column(column)
            .iter()
            .zip(weights.iter())
            .map(|(&value, &weight)| weight * value * value)
            .sum::<f64>()
            .max(f64::MIN_POSITIVE);
        penalty[[column, column]] = diagonal;
    }
    let response_array = Array1::from_vec(response.to_vec());
    let response_column = response_array.view().insert_axis(ndarray::Axis(1));
    let (coeffs, fitted) = gam_linalg::utils::gaussian_weighted_ridge(
        design,
        response_column,
        penalty.view(),
        weights,
        AUTO_Z_CONDITIONAL_RIDGE_REL,
    )?;
    Ok((
        coeffs.column(0).to_vec(),
        fitted.column(0).to_vec(),
    ))
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Deterministic standard normals (Box–Muller over splitmix64).
    fn gaussians(n: usize, seed: u64) -> Vec<f64> {
        let mut state = seed;
        let mut unit = || {
            state = state.wrapping_add(0x9E37_79B9_7F4A_7C15);
            let mut z = state;
            z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
            z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
            z ^= z >> 31;
            ((z >> 11) as f64 + 0.5) / (1u64 << 53) as f64
        };
        let mut out = Vec::with_capacity(n + 1);
        while out.len() < n {
            let u1 = unit().max(1e-12);
            let u2 = unit();
            let r = (-2.0 * u1.ln()).sqrt();
            out.push(r * (std::f64::consts::TAU * u2).cos());
            out.push(r * (std::f64::consts::TAU * u2).sin());
        }
        out.truncate(n);
        out
    }

    fn standardized(mut v: Vec<f64>) -> Vec<f64> {
        let n = v.len() as f64;
        let mean = v.iter().sum::<f64>() / n;
        let sd = (v.iter().map(|x| (x - mean) * (x - mean)).sum::<f64>() / n)
            .sqrt()
            .max(1e-12);
        for value in v.iter_mut() {
            *value = (*value - mean) / sd;
        }
        v
    }

    /// A `K = 2` sample drawn from EXACTLY the model this module fits:
    /// `ζ₀ = √d₀·e₀`, `ζ₁ = φ(x)·ζ₀ + √d₁(x)·e₁` with `φ(x) = φ₀ + φ₁x` and
    /// `log d₁(x) = γ₀ + γ₁x`. Recovery is then a statement about the estimator
    /// and not about approximation error.
    fn in_class_fixture(
        n: usize,
        phi: [f64; 2],
        gamma: [f64; 2],
        log_d0: f64,
    ) -> (Array2<f64>, Array1<f64>, Array2<f64>) {
        let x = standardized(gaussians(n, 0x2766_D4));
        let e0 = standardized(gaussians(n, 0x2766_E5));
        let e1 = standardized(gaussians(n, 0x2766_F6));
        let mut scores = Array2::<f64>::zeros((n, 2));
        let mut a_block = Array2::<f64>::zeros((n, 1));
        let sd0 = (0.5 * log_d0).exp();
        for row in 0..n {
            a_block[[row, 0]] = x[row];
            let z0 = sd0 * e0[row];
            let sd1 = (0.5 * (gamma[0] + gamma[1] * x[row])).exp();
            scores[[row, 0]] = z0;
            scores[[row, 1]] = (phi[0] + phi[1] * x[row]) * z0 + sd1 * e1[row];
        }
        (scores, Array1::<f64>::ones(n), a_block)
    }

    /// The estimator must recover a truth drawn from its own model class:
    /// the coupling slope, the innovation-variance slope, and hence `Σ(a)`
    /// itself across the covariate range.
    #[test]
    fn recovers_an_in_class_conditional_covariance() {
        let n = 40_000;
        let phi = [0.3_f64, 0.5];
        let gamma = [(0.75_f64).ln(), -0.4];
        let log_d0 = 0.0;
        let (scores, weights, a_block) = in_class_fixture(n, phi, gamma, log_d0);
        let fitted = ConditionalScoreCovariance::fit(scores.view(), weights.view(), a_block.view())
            .expect("fit")
            .expect("a varying cross-score covariance must escalate");

        assert_eq!(fitted.score_dim, 2);
        assert_eq!(fitted.basis_ncols, 1);
        // Coordinate 1's coupling must be the planted affine function.
        let coupling = &fitted.coordinates[1].autoregression[0];
        assert_eq!(
            coupling.len(),
            2,
            "the coupling depends on a, so its own interaction gate must promote it"
        );
        assert!(
            (coupling[0] - phi[0]).abs() < 0.02 && (coupling[1] - phi[1]).abs() < 0.02,
            "coupling {coupling:?} against planted {phi:?}"
        );
        let innovation = &fitted.coordinates[1].log_innovation;
        assert_eq!(innovation.len(), 2, "the innovation variance depends on a");
        assert!(
            (innovation[0] - gamma[0]).abs() < 0.03 && (innovation[1] - gamma[1]).abs() < 0.03,
            "log innovation {innovation:?} against planted {gamma:?}"
        );

        // And the assembled Σ(a) itself, against the closed-form truth.
        for &probe in &[-1.5_f64, -0.5, 0.0, 0.5, 1.5] {
            let a_row = Array1::from_vec(vec![probe]);
            let sigma = fitted.dense_at(a_row.view()).expect("Σ(a)");
            let phi_true = phi[0] + phi[1] * probe;
            let d0_true = log_d0.exp();
            let d1_true = (gamma[0] + gamma[1] * probe).exp();
            let truth = [
                d0_true,
                phi_true * d0_true,
                phi_true * phi_true * d0_true + d1_true,
            ];
            let got = [sigma[[0, 0]], sigma[[0, 1]], sigma[[1, 1]]];
            for (index, (&want, &have)) in truth.iter().zip(got.iter()).enumerate() {
                assert!(
                    (want - have).abs() < 0.05 * (1.0 + want.abs()),
                    "Σ(a={probe}) entry {index}: got {have}, want {want}"
                );
            }
            assert_eq!(sigma[[0, 1]], sigma[[1, 0]], "Σ(a) must be symmetric");
        }
        // The gate's own evidence must name the pair it fired on.
        assert!(
            fitted
                .pair_pvalues
                .iter()
                .any(|&(left, right, p)| left == 0 && right == 1 && p < 1.0e-3),
            "the (0,1) pair must be the recorded trigger: {:?}",
            fitted.pair_pvalues
        );
    }

    /// The escalation gate must hold its SIZE. A trigger-happy gate would
    /// install a fitted covariance field on every multi-score fit, replacing an
    /// exactly-correct pooled object with an estimated one — a worse trade than
    /// the defect it exists to fix.
    ///
    /// Size, not one sample. The gate is a level-`AUTO_Z_CONDITIONAL_RAO_ALPHA`
    /// hypothesis test, so "it did not fire on this null sample" is an assertion
    /// about a `1 − α` event and a single unlucky draw would make it red for a
    /// reason that is not a defect. (It does happen: at `n = 40000` the first
    /// seed tried here drew `|Z| = 3.31`, `p = 9.2e-4`, just inside `α = 1e-3`.)
    /// What is actually claimed is the escalation RATE over a bank of null
    /// replicates, and the bound is derived rather than chosen: with
    /// `R = REPLICATES` draws at level `α`, the escalation count is
    /// `Binomial(R, α)`, and `MAX_NULL_ESCALATIONS` is the smallest `k` for
    /// which `P(Binomial(R, α) > k) < α` — i.e. the smallest bound that makes
    /// THIS test's own false-alarm rate no worse than the gate's. At `R = 32`,
    /// `α = 1e-3`: `P(X ≥ 1) = 3.2e-2` (too loose), `P(X ≥ 2) = 4.9e-4 < α`, so
    /// `k = 1`.
    #[test]
    fn the_escalation_gate_holds_its_size_on_null_replicates() {
        const REPLICATES: usize = 32;
        const MAX_NULL_ESCALATIONS: usize = 1;
        let n = 4_000;
        let mut escalations = Vec::new();
        for replicate in 0..REPLICATES {
            let (scores, weights, a_block) = null_fixture(n, replicate as u64);
            if ConditionalScoreCovariance::fit(scores.view(), weights.view(), a_block.view())
                .expect("fit")
                .is_some()
            {
                escalations.push(replicate);
            }
        }
        assert!(
            escalations.len() <= MAX_NULL_ESCALATIONS,
            "the pair gate escalated on {} of {REPLICATES} null replicates (bound \
             {MAX_NULL_ESCALATIONS}): {escalations:?}",
            escalations.len()
        );
    }

    /// The same fixture shape as [`in_class_fixture`] with every `a`-varying
    /// coefficient set to zero, so `Cov(z₀, z₁ | a)` is constant by
    /// construction, seeded per replicate.
    fn null_fixture(n: usize, replicate: u64) -> (Array2<f64>, Array1<f64>, Array2<f64>) {
        let base = 0x2766_0000_u64 + replicate * 3;
        let x = standardized(gaussians(n, base));
        let e0 = standardized(gaussians(n, base + 1));
        let e1 = standardized(gaussians(n, base + 2));
        let mut scores = Array2::<f64>::zeros((n, 2));
        let mut a_block = Array2::<f64>::zeros((n, 1));
        for row in 0..n {
            a_block[[row, 0]] = x[row];
            scores[[row, 0]] = e0[row];
            scores[[row, 1]] = 0.4 * e0[row] + (0.75_f64).sqrt() * e1[row];
        }
        (scores, Array1::<f64>::ones(n), a_block)
    }

    /// `K = 1` has no off-diagonal, and its conditional variance is gam#2768's
    /// per-coordinate branch. Escalating here would double-correct it.
    #[test]
    fn a_single_score_is_out_of_scope() {
        let n = 4_000;
        let x = standardized(gaussians(n, 0x2766_D4));
        let e = standardized(gaussians(n, 0x2766_E5));
        let mut scores = Array2::<f64>::zeros((n, 1));
        let mut a_block = Array2::<f64>::zeros((n, 1));
        for row in 0..n {
            a_block[[row, 0]] = x[row];
            scores[[row, 0]] = (0.5 * x[row]).exp() * e[row];
        }
        let weights = Array1::<f64>::ones(n);
        assert!(
            ConditionalScoreCovariance::fit(scores.view(), weights.view(), a_block.view())
                .expect("fit")
                .is_none(),
            "K = 1 is gam#2768's branch, not this one"
        );
    }

    /// Positive definiteness is a property of the PARAMETERISATION, so it must
    /// survive rows the fit never saw — including rows far outside the training
    /// hull, where a model on the entries of `Σ` would return `|ρ| > 1`.
    #[test]
    fn the_factor_is_positive_definite_off_the_training_hull() {
        let n = 20_000;
        let (scores, weights, a_block) =
            in_class_fixture(n, [0.3, 0.9], [(0.5_f64).ln(), -0.8], 0.0);
        let fitted = ConditionalScoreCovariance::fit(scores.view(), weights.view(), a_block.view())
            .expect("fit")
            .expect("escalates");
        for &probe in &[-1.0e6_f64, -50.0, -5.0, 0.0, 5.0, 50.0, 1.0e6] {
            let a_row = Array1::from_vec(vec![probe]);
            let mut factor = Array2::<f64>::zeros((2, 2));
            fitted.factor_into(a_row.view(), &mut factor).expect("L(a)");
            assert!(
                factor[[0, 1]] == 0.0,
                "L(a) must be lower triangular; got {factor:?}"
            );
            assert!(
                factor[[0, 0]] > 0.0 && factor[[1, 1]] > 0.0,
                "L(a) must have a strictly positive diagonal at a={probe}: {factor:?}"
            );
            let sigma = fitted.dense_at(a_row.view()).expect("Σ(a)");
            let determinant = sigma[[0, 0]] * sigma[[1, 1]] - sigma[[0, 1]] * sigma[[1, 0]];
            assert!(
                sigma[[0, 0]] > 0.0 && determinant > 0.0,
                "Σ(a={probe}) must be positive definite: {sigma:?} (det {determinant})"
            );
            let correlation = sigma[[0, 1]] / (sigma[[0, 0]] * sigma[[1, 1]]).sqrt();
            assert!(
                correlation.abs() < 1.0,
                "an admissible Σ cannot have |corr| >= 1; got {correlation} at a={probe}"
            );
        }
    }

    /// The materialised per-row stack must be the same object `dense_at`
    /// describes, in the `Σ = L Lᵀ` representation whose quadratic forms the row
    /// program evaluates as exact sums of squares.
    #[test]
    fn the_row_stack_agrees_with_the_dense_evaluation() {
        let n = 4_000;
        let (scores, weights, a_block) =
            in_class_fixture(n, [0.2, 0.6], [(0.9_f64).ln(), 0.3], 0.1);
        let fitted = ConditionalScoreCovariance::fit(scores.view(), weights.view(), a_block.view())
            .expect("fit")
            .expect("escalates");
        let stack = fitted.row_covariances(a_block.view()).expect("row stack");
        assert_eq!(stack.len(), n);
        for &row in &[0usize, 1, 17, n / 2, n - 1] {
            let dense = fitted.dense_at(a_block.row(row)).expect("Σ(a)");
            let from_stack = stack[row].to_dense();
            for left in 0..2 {
                for right in 0..2 {
                    assert!(
                        (dense[[left, right]] - from_stack[[left, right]]).abs() < 1.0e-12,
                        "row {row} entry ({left},{right}): {} vs {}",
                        dense[[left, right]],
                        from_stack[[left, right]]
                    );
                }
            }
            // `1ᵀΣ1` is the cached scalar the SHARED slope lane consumes; it
            // must be the same number the dense matrix implies.
            let ones_form: f64 = dense.iter().sum();
            assert!(
                (stack[row].ones_quadratic_form() - ones_form).abs()
                    < 1.0e-10 * (1.0 + ones_form.abs()),
                "row {row}: cached 1'Σ1 {} vs dense {ones_form}",
                stack[row].ones_quadratic_form()
            );
        }
    }

    /// Collinear scores are a real and expected input — `MarginalSlopeCovariance`
    /// says so in its own admission doc, and admits the exactly-singular pooled
    /// matrix they produce. The log parameterisation cannot represent a zero
    /// innovation variance, so this is the one place it needs a floor, and the
    /// floor must not turn a valid input into a refusal.
    ///
    /// The fixture makes the pair gate fire on a perfectly collinear pair:
    /// `z₁ = z₀` with `Var(z₀|a)` moving, so `Cov(z₀, z₁|a) = Var(z₀|a)` moves
    /// too. Before the floor this reached `fit_log_innovation` with an exactly
    /// zero residual variance and errored the whole fit.
    #[test]
    fn collinear_scores_are_admitted_rather_than_refused() {
        let n = 20_000;
        let x = standardized(gaussians(n, 0x2766_5E));
        let e0 = standardized(gaussians(n, 0x2766_6F));
        let mut scores = Array2::<f64>::zeros((n, 2));
        let mut a_block = Array2::<f64>::zeros((n, 1));
        for row in 0..n {
            a_block[[row, 0]] = x[row];
            let z0 = (0.5 * (0.6 * x[row])).exp() * e0[row];
            scores[[row, 0]] = z0;
            scores[[row, 1]] = z0;
        }
        let weights = Array1::<f64>::ones(n);
        let fitted = ConditionalScoreCovariance::fit(scores.view(), weights.view(), a_block.view())
            .expect("a collinear score pair must be admitted, not refused")
            .expect("a moving Var(z₀|a) makes Cov(z₀,z₁|a) move, so the pair gate fires");
        for &probe in &[-1.5_f64, 0.0, 1.5] {
            let a_row = Array1::from_vec(vec![probe]);
            let sigma = fitted.dense_at(a_row.view()).expect("Σ(a)");
            let determinant = sigma[[0, 0]] * sigma[[1, 1]] - sigma[[0, 1]] * sigma[[1, 0]];
            assert!(
                sigma[[0, 0]] > 0.0 && determinant > 0.0,
                "Σ(a={probe}) must stay positive definite on a collinear pair: {sigma:?}"
            );
            let correlation = sigma[[0, 1]] / (sigma[[0, 0]] * sigma[[1, 1]]).sqrt();
            assert!(
                (correlation - 1.0).abs() < 1.0e-6,
                "a collinear pair must read as correlation 1; got {correlation} at a={probe}"
            );
            // And the variance itself must track the planted `exp(0.6·x)`.
            let planted = (0.6 * probe).exp();
            assert!(
                (sigma[[0, 0]] - planted).abs() < 0.1 * (1.0 + planted),
                "Σ₀₀(a={probe}) = {} against planted {planted}",
                sigma[[0, 0]]
            );
        }
    }

    /// Three scores, so the triangular reconstruction is exercised past the
    /// `K = 2` special case where `T⁻¹` has no accumulated term.
    #[test]
    fn three_scores_reconstruct_through_the_forward_substitution() {
        let n = 30_000;
        let x = standardized(gaussians(n, 0x2766_1A));
        let e0 = standardized(gaussians(n, 0x2766_2B));
        let e1 = standardized(gaussians(n, 0x2766_3C));
        let e2 = standardized(gaussians(n, 0x2766_4D));
        let mut scores = Array2::<f64>::zeros((n, 3));
        let mut a_block = Array2::<f64>::zeros((n, 1));
        // φ₁₀(x) = 0.4 + 0.3x,  φ₂₀ = 0.2 (constant),  φ₂₁(x) = −0.1 + 0.45x.
        for row in 0..n {
            let xi = x[row];
            a_block[[row, 0]] = xi;
            let z0 = e0[row];
            let z1 = (0.4 + 0.3 * xi) * z0 + 0.8 * e1[row];
            let z2 = 0.2 * z0 + (-0.1 + 0.45 * xi) * z1 + 0.7 * e2[row];
            scores[[row, 0]] = z0;
            scores[[row, 1]] = z1;
            scores[[row, 2]] = z2;
        }
        let weights = Array1::<f64>::ones(n);
        let fitted = ConditionalScoreCovariance::fit(scores.view(), weights.view(), a_block.view())
            .expect("fit")
            .expect("escalates");
        assert_eq!(fitted.coordinates.len(), 3);
        assert_eq!(fitted.coordinates[2].autoregression.len(), 2);
        for &probe in &[-1.0_f64, 0.0, 1.0] {
            let a_row = Array1::from_vec(vec![probe]);
            let sigma = fitted.dense_at(a_row.view()).expect("Σ(a)");
            // Closed-form truth by the same forward recursion.
            let phi10 = 0.4 + 0.3 * probe;
            let phi20 = 0.2_f64;
            let phi21 = -0.1 + 0.45 * probe;
            let (d0, d1, d2) = (1.0_f64, 0.64_f64, 0.49_f64);
            let s00 = d0;
            let s01 = phi10 * d0;
            let s11 = phi10 * phi10 * d0 + d1;
            let s02 = phi20 * d0 + phi21 * s01;
            let s12 = phi20 * s01 + phi21 * s11;
            let s22 = phi20 * s02 + phi21 * s12 + d2;
            let truth = [s00, s01, s02, s11, s12, s22];
            let got = [
                sigma[[0, 0]],
                sigma[[0, 1]],
                sigma[[0, 2]],
                sigma[[1, 1]],
                sigma[[1, 2]],
                sigma[[2, 2]],
            ];
            for (index, (&want, &have)) in truth.iter().zip(got.iter()).enumerate() {
                assert!(
                    (want - have).abs() < 0.06 * (1.0 + want.abs()),
                    "K=3 Σ(a={probe}) entry {index}: got {have}, want {want}"
                );
            }
        }
    }

    /// An INDEPENDENT oracle: the fitted `Σ(a)` against a nonparametric local
    /// estimate.
    ///
    /// Every other recovery test here compares the fit to the coefficients it
    /// was generated from, which is a comparison inside the model's own
    /// parameterisation. This one is not: it bins the rows by the conditioning
    /// covariate, takes the ordinary empirical second moments inside each bin,
    /// and asks the fitted surface to reproduce them. It would catch a
    /// parameterisation that recovers its own planted coefficients and still
    /// assembles the wrong matrix — a forward-substitution transposition, a
    /// column scaled by `d` instead of `√d`, a coordinate order swapped.
    ///
    /// The truth is IN the model class here, deliberately: the bar is each bin's
    /// own sampling band, so any approximation error would be read as a defect.
    /// How the model behaves when the truth is out of class is a different
    /// question and is measured where it belongs — on the end-to-end identity,
    /// in `survival_multi_z_conditional_covariance_2766`, against a planted
    /// sigmoid.
    #[test]
    fn the_fitted_surface_matches_a_nonparametric_local_covariance() {
        const BINS: usize = 8;
        let n = 40_000;
        let phi = [0.15_f64, 0.42];
        let gamma = [(0.7_f64).ln(), -0.3];
        let (scores, weights, a_block) = in_class_fixture(n, phi, gamma, 0.0);
        let fitted = ConditionalScoreCovariance::fit(scores.view(), weights.view(), a_block.view())
            .expect("fit")
            .expect("escalates");

        // Bin on the conditioning covariate's own rank, so every bin holds the
        // same number of rows and the band below is the same in each.
        let mut order: Vec<usize> = (0..n).collect();
        order.sort_by(|&left, &right| {
            a_block[[left, 0]]
                .partial_cmp(&a_block[[right, 0]])
                .expect("the fixture covariate is finite, so the ordering is total")
        });
        let mut bin = vec![0usize; n];
        for (rank, &row) in order.iter().enumerate() {
            bin[row] = (rank * BINS / n).min(BINS - 1);
        }

        let mut counts = vec![0usize; BINS];
        let mut empirical = vec![[0.0_f64; 3]; BINS];
        let mut modelled = vec![[0.0_f64; 3]; BINS];
        for row in 0..n {
            let index = bin[row];
            counts[index] += 1;
            let (left, right) = (scores[[row, 0]], scores[[row, 1]]);
            empirical[index][0] += left * left;
            empirical[index][1] += left * right;
            empirical[index][2] += right * right;
            let sigma = fitted.dense_at(a_block.row(row)).expect("Σ(a)");
            modelled[index][0] += sigma[[0, 0]];
            modelled[index][1] += sigma[[0, 1]];
            modelled[index][2] += sigma[[1, 1]];
        }
        let mut worst = 0.0_f64;
        let mut worst_label = String::new();
        for index in 0..BINS {
            let scale = counts[index] as f64;
            // A second moment of Gaussians has sampling standard error
            // `moment·√(2/n_bin)`; three of those is the band a correct surface
            // sits inside, and it is derived from the bin rather than chosen.
            for entry in 0..3 {
                let want = empirical[index][entry] / scale;
                let have = modelled[index][entry] / scale;
                let reference = (empirical[index][0] / scale).max(empirical[index][2] / scale);
                let band = 3.0 * reference * (2.0 / scale).sqrt();
                let miss = (want - have).abs() / band;
                if miss > worst {
                    worst = miss;
                    worst_label = format!(
                        "bin {index} (n={}) entry {entry}: empirical {want:.4} against fitted \
                         {have:.4}, band {band:.4}",
                        counts[index]
                    );
                }
            }
        }
        assert!(
            worst <= 1.0,
            "the fitted surface must sit inside each bin's own sampling band; worst {worst:.2}x \
             at {worst_label}"
        );
    }

    /// A fitted field has to survive the on-disk round trip unchanged: the
    /// predict path rebuilds `Σ(a)` from exactly these coefficients, so a
    /// serialisation that dropped a stage would silently evaluate a different
    /// model.
    #[test]
    fn the_fitted_field_round_trips_through_serde() {
        let n = 8_000;
        let (scores, weights, a_block) =
            in_class_fixture(n, [0.25, 0.55], [(0.8_f64).ln(), -0.35], 0.0);
        let fitted = ConditionalScoreCovariance::fit(scores.view(), weights.view(), a_block.view())
            .expect("fit")
            .expect("escalates");
        let encoded = serde_json::to_string(&fitted).expect("encode");
        let decoded: ConditionalScoreCovariance = serde_json::from_str(&encoded).expect("decode");
        assert_eq!(decoded, fitted);
        let a_row = Array1::from_vec(vec![0.37]);
        let before = fitted.dense_at(a_row.view()).expect("Σ before");
        let after = decoded.dense_at(a_row.view()).expect("Σ after");
        assert_eq!(before, after);
    }

    /// Against the object it replaces: averaged over the training rows, the
    /// conditional field must reproduce the pooled covariance. `Σ̄ = E[Σ(a)] +
    /// Var(E[z|a])`, and the conditional mean is zero here, so the two agree —
    /// which is the sense in which this is a refinement of the pooled estimator
    /// and not a different quantity.
    #[test]
    fn the_row_average_reproduces_the_pooled_covariance() {
        let n = 40_000;
        let (scores, weights, a_block) =
            in_class_fixture(n, [0.3, 0.5], [(0.75_f64).ln(), -0.4], 0.0);
        let fitted = ConditionalScoreCovariance::fit(scores.view(), weights.view(), a_block.view())
            .expect("fit")
            .expect("escalates");
        let pooled = super::super::marginal_slope_covariance_from_scores(scores.view(), &weights)
            .expect("pooled Σ")
            .to_dense();
        let stack = fitted.row_covariances(a_block.view()).expect("stack");
        let mut average = Array2::<f64>::zeros((2, 2));
        for covariance in &stack {
            average += &covariance.to_dense();
        }
        average /= n as f64;
        for left in 0..2 {
            for right in 0..2 {
                let want = pooled[[left, right]];
                let have = average[[left, right]];
                assert!(
                    (want - have).abs() < 0.05 * (1.0 + want.abs()),
                    "row-averaged Σ({left},{right}) = {have} against pooled {want}"
                );
            }
        }
    }
}
