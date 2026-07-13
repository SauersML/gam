//! Vector-valued response support.
//!
//! Many smooths sharing one latent: the shape function in the latent-variable
//! engine maps to a reduced activation vector (tens-to-hundreds of dimensions,
//! after a random-matrix noise cut). This module defines the response-side
//! types, the Gaussian vector likelihood, and the connector trait the inner
//! solver consumes.
//!
//! Conventions:
//! - `Y` is shape `(N, M)`: `N` rows, `M` output dimensions.
//! - `eta` is shape `(N, M)`: the linear predictor with one column per output.
//! - For Gaussian identity-link, mean(η) = η, so the likelihood depends only
//!   on `eta` and `Y`.
//!
//! The Hessian is block-structured: per-row (N independent blocks for the
//! Gaussian case), each of size `(M, M)`. For a Gaussian likelihood with
//! Diagonal/Isotropic noise this per-row block is itself diagonal — exactly
//! what the arrow Schur elimination in `solver/arrow_schur.rs` consumes.

use crate::model_types::EstimationError;
use crate::multinomial_reml::{MultinomialLogitRowProgram, multinomial_logit_probabilities_into};
use ndarray::{Array1, Array2, Array3, ArrayView2};

/// Per-output noise model for a vector response.
///
/// `LowRank` stores the symmetric structured precision
/// `W = diag(diag) + U Uᵀ`, with `factor` holding `U`. The vector likelihood
/// consumes the owned arrays directly; PIRLS low-rank Gram assembly is handled
/// by `gam_linalg::low_rank_weight::LowRankWeight` and
/// `gam_solve::pirls`.
#[derive(Clone, Debug)]
pub enum VectorNoise {
    /// Shared σ across all M outputs: Σ = σ² I_M.
    Isotropic(f64),
    /// Per-output σ_m: Σ = diag(σ_m²).
    Diagonal(Array1<f64>),
    /// Symmetric structured form `W = diag(diag) + factor · factorᵀ`.
    LowRank {
        diag: Array1<f64>,
        factor: Array2<f64>,
    },
}

impl VectorNoise {
    /// Per-output precision vector (1/σ_m²) for the Isotropic / Diagonal cases.
    /// LowRank returns the diagonal piece only; the low-rank correction is
    /// applied separately by the Piece 5 weight code.
    pub fn diag_precision(&self, m: usize) -> Result<Array1<f64>, EstimationError> {
        match self {
            Self::Isotropic(sigma) => {
                if !sigma.is_finite() || *sigma <= 0.0 {
                    crate::bail_invalid_estim!(
                        "VectorNoise::Isotropic: σ must be > 0 and finite (got {sigma})",
                    );
                }
                let p = 1.0 / (sigma * sigma);
                Ok(Array1::from_elem(m, p))
            }
            Self::Diagonal(sigma) => {
                if sigma.len() != m {
                    crate::bail_invalid_estim!(
                        "VectorNoise::Diagonal: σ length {} ≠ M={m}",
                        sigma.len()
                    );
                }
                let mut out = Array1::<f64>::zeros(m);
                for j in 0..m {
                    let s = sigma[j];
                    if !s.is_finite() || s <= 0.0 {
                        crate::bail_invalid_estim!(
                            "VectorNoise::Diagonal: σ[{j}] must be > 0 and finite (got {s})",
                        );
                    }
                    out[j] = 1.0 / (s * s);
                }
                Ok(out)
            }
            Self::LowRank { diag, .. } => {
                if diag.len() != m {
                    crate::bail_invalid_estim!(
                        "VectorNoise::LowRank: diag length {} ≠ M={m}",
                        diag.len()
                    );
                }
                let mut out = Array1::<f64>::zeros(m);
                for j in 0..m {
                    let d = diag[j];
                    if !d.is_finite() || d <= 0.0 {
                        crate::bail_invalid_estim!(
                            "VectorNoise::LowRank: diag[{j}] must be > 0 (got {d})",
                        );
                    }
                    // `diag` is the PRECISION diagonal (W = diag(d) + F·Fᵀ).
                    // Pass it through unchanged.
                    out[j] = d;
                }
                Ok(out)
            }
        }
    }
}

/// Vector-valued response target.
///
/// `y` is `(N, M)`; `row_weights` (if present) is length `N` and scales the
/// per-row contribution to the likelihood (e.g. observation weights from a
/// re-sampling or inverse-probability scheme).
#[derive(Clone, Debug)]
pub struct VectorResponseTarget {
    /// shape (N, M) — N rows × M output dimensions.
    pub y: Array2<f64>,
    /// per-output noise (or shared scalar).
    pub noise: VectorNoise,
    /// optional row weights (N,).
    pub row_weights: Option<Array1<f64>>,
}

impl VectorResponseTarget {
    pub fn new(y: Array2<f64>, noise: VectorNoise) -> Self {
        Self {
            y,
            noise,
            row_weights: None,
        }
    }

    pub fn with_row_weights(mut self, w: Array1<f64>) -> Result<Self, EstimationError> {
        validate_row_weights(&w, self.y.nrows())?;
        self.row_weights = Some(w);
        Ok(self)
    }

    pub fn n(&self) -> usize {
        self.y.nrows()
    }
    pub fn m(&self) -> usize {
        self.y.ncols()
    }
}

/// Relative tolerance on the per-row simplex constraint `Σ_c y_{n,c} = 1`.
///
/// The multinomial-logit log-likelihood `ℓ = Σ_c y_c log p_c` has the
/// canonical residual gradient `y_a − p_a` and Fisher block
/// `p_a δ_{ab} − p_a p_b` **only** when each target row is a probability
/// vector (`y_c ≥ 0`, `Σ_c y_c = 1`). For a general row mass `s = Σ_c y_c`
/// the true derivatives are `y_a − s p_a` and `s (p_a δ_{ab} − p_a p_b)`, so
/// any row whose mass deviates from 1 makes the implemented gradient/Hessian
/// disagree with the implemented objective. We therefore require simplex rows
/// at every construction boundary and reject anything else, rather than
/// silently fitting with inconsistent curvature. The tolerance absorbs only
/// floating-point round-off in an otherwise-exact one-hot / label-smoothed
/// row (e.g. a sum of `K` rationals), not genuine count or proportional data.
pub(crate) const MULTINOMIAL_SIMPLEX_TOL: f64 = 1.0e-9;

/// Validate that every row of a multinomial target `y ∈ ℝ^{N×K}` is a point on
/// the probability simplex: `y_{n,c} ≥ 0` for all entries and
/// `Σ_c y_{n,c} = 1` for every row (up to [`MULTINOMIAL_SIMPLEX_TOL`]). This
/// is the precondition under which [`MultinomialLogitLikelihood`]'s residual
/// gradient and Fisher block are the exact derivatives of its log-likelihood;
/// see the constant's docs. Finiteness is checked first so the message points
/// at the offending entry rather than at a NaN-poisoned row sum.
pub(crate) fn validate_multinomial_simplex(
    y: ArrayView2<f64>,
    context: &str,
) -> Result<(), EstimationError> {
    let (n, k) = y.dim();
    for row in 0..n {
        let mut row_sum = 0.0_f64;
        for c in 0..k {
            let v = y[[row, c]];
            if !v.is_finite() {
                crate::bail_invalid_estim!("{context}: y[{row},{c}] must be finite (got {v})");
            }
            if v < 0.0 {
                crate::bail_invalid_estim!(
                    "{context}: multinomial target must be a probability vector \
                     (y_c ≥ 0); got y[{row},{c}] = {v}"
                );
            }
            row_sum += v;
        }
        if (row_sum - 1.0).abs() > MULTINOMIAL_SIMPLEX_TOL {
            crate::bail_invalid_estim!(
                "{context}: multinomial target rows must sum to 1 (one-hot for \
                 hard labels, or a label-smoothed probability vector); row {row} \
                 sums to {row_sum}. The softmax residual gradient y_a − p_a and \
                 Fisher block p_a δ_ab − p_a p_b are the derivatives of \
                 Σ_c y_c log p_c only when the row mass is 1."
            );
        }
    }
    Ok(())
}

fn validate_row_weights(weights: &Array1<f64>, n: usize) -> Result<(), EstimationError> {
    if weights.len() != n {
        crate::bail_invalid_estim!("row_weights length {} ≠ N={n}", weights.len());
    }
    for (idx, weight) in weights.iter().copied().enumerate() {
        if !(weight.is_finite() && weight >= 0.0) {
            crate::bail_invalid_estim!(
                "row_weights[{idx}] must be finite and non-negative (got {weight})"
            );
        }
    }
    Ok(())
}

/// Connector trait the inner solver (Piece 1) plugs into.
///
/// `eta` is the `(N, M)` linear predictor; `y` is the `(N, M)` target. The
/// implementation is responsible for any link inversion. The `hess_diag`
/// return is the per-element diagonal of the per-row Hessian block; for a
/// Diagonal-noise Gaussian this is exactly `(N, M)` of per-output precisions.
pub trait VectorLikelihood {
    /// log p(Y | η).
    fn log_lik(&self, eta: ArrayView2<f64>, y: ArrayView2<f64>) -> Result<f64, EstimationError>;

    /// ∂ log p(Y | η) / ∂ η, shape (N, M).
    fn grad_eta(
        &self,
        eta: ArrayView2<f64>,
        y: ArrayView2<f64>,
    ) -> Result<Array2<f64>, EstimationError>;

    /// Diagonal of the per-row Hessian −∂² log p / ∂ η ∂ η, shape (N, M).
    /// This is the per-row block consumed by `solver/arrow_schur.rs`.
    fn hess_diag(
        &self,
        eta: ArrayView2<f64>,
        y: ArrayView2<f64>,
    ) -> Result<Array2<f64>, EstimationError>;

    /// Per-row dense Hessian block −∂² log p / ∂η_a ∂η_b, shape (N, M, M).
    ///
    /// Default implementation lifts [`Self::hess_diag`] onto the per-row
    /// diagonal, valid only when the per-row Hessian is genuinely diagonal
    /// across outputs (e.g. Gaussian with Isotropic/Diagonal noise).
    /// Likelihoods with off-diagonal output coupling must override this:
    /// [`GaussianVectorLikelihood`] with a low-rank precision factor `F`
    /// (block `w·(diag(precision) + F·Fᵀ)`, off-diagonals `w·Σ_k F[a,k]·F[b,k]`)
    /// and multinomial-logit (per-row Fisher block `p_a (δ_ab − p_b)`).
    ///
    /// The returned array is consumed by
    /// [`gam_solve::pirls::dense_block_xtwx`] /
    /// [`gam_solve::pirls::dense_block_xtwy`] to build `XᵀWX` and `XᵀWy`
    /// for vector-response IRLS in output-major coefficient ordering.
    fn hess_block(
        &self,
        eta: ArrayView2<f64>,
        y: ArrayView2<f64>,
    ) -> Result<Array3<f64>, EstimationError> {
        let diag = self.hess_diag(eta, y)?;
        let (n, m) = diag.dim();
        let mut out = Array3::<f64>::zeros((n, m, m));
        for row in 0..n {
            for j in 0..m {
                out[[row, j, j]] = diag[[row, j]];
            }
        }
        Ok(out)
    }
}

pub(crate) fn validate_vector_likelihood_inputs(
    context: &str,
    eta: ArrayView2<'_, f64>,
    y: ArrayView2<'_, f64>,
    expected_columns: Option<usize>,
) -> Result<(), EstimationError> {
    if eta.dim() != y.dim() {
        crate::bail_invalid_estim!(
            "{context}: eta shape {:?} does not match response shape {:?}",
            eta.dim(),
            y.dim()
        );
    }
    if let Some(expected) = expected_columns
        && eta.ncols() != expected
    {
        crate::bail_invalid_estim!(
            "{context}: eta has {} columns; expected {expected}",
            eta.ncols()
        );
    }
    if let Some(((row, column), value)) = eta.indexed_iter().find(|(_, value)| !value.is_finite()) {
        crate::bail_invalid_estim!("{context}: eta[{row},{column}] must be finite, got {value}");
    }
    if let Some(((row, column), value)) = y.indexed_iter().find(|(_, value)| !value.is_finite()) {
        crate::bail_invalid_estim!(
            "{context}: response[{row},{column}] must be finite, got {value}"
        );
    }
    Ok(())
}

/// Gaussian vector likelihood with identity link.
///
/// `log p(Y|η) = −½ Σ_n w_n · rᵀ W r` where `r = Y_n − η_n` and `W` is the
/// per-output **precision** matrix. For Isotropic / Diagonal `W = diag(prec)`;
/// for `LowRank` it is `W = diag(prec) + F · Fᵀ`, with `F` carried alongside
/// the diagonal here.
///
/// (Up to the constant log-determinant of the noise covariance, dropped here
/// because it does not depend on β or the latent t; the determinant is
/// accounted for in the REML score, not the inner likelihood.)
#[derive(Clone, Debug)]
pub struct GaussianVectorLikelihood {
    /// Per-output diagonal precision (length M). For Isotropic / Diagonal /
    /// LowRank this is the diagonal piece of the precision matrix
    /// (`1/σ_m²` for Diagonal/Isotropic; `diag` for LowRank).
    pub precision: Array1<f64>,
    /// Optional dense rank-r factor `F` of size `(M, r)` such that the full
    /// per-row precision is `diag(precision) + F · Fᵀ`. `None` for the
    /// Isotropic / Diagonal cases.
    pub factor: Option<Array2<f64>>,
    /// Optional row weights (length N), or None for uniform.
    pub row_weights: Option<Array1<f64>>,
}

impl GaussianVectorLikelihood {
    pub fn from_target(target: &VectorResponseTarget) -> Result<Self, EstimationError> {
        if let Some(weights) = target.row_weights.as_ref() {
            validate_row_weights(weights, target.n())?;
        }
        let precision = target.noise.diag_precision(target.m())?;
        let factor = match &target.noise {
            VectorNoise::LowRank { factor, .. } => {
                if factor.nrows() != target.m() {
                    crate::bail_invalid_estim!(
                        "VectorNoise::LowRank: factor has {} rows but M={}",
                        factor.nrows(),
                        target.m()
                    );
                }
                for ((row, col), value) in factor.indexed_iter() {
                    if !value.is_finite() {
                        crate::bail_invalid_estim!(
                            "VectorNoise::LowRank: factor[{row},{col}] must be finite (got {value})"
                        );
                    }
                }
                Some(factor.clone())
            }
            _ => None,
        };
        Ok(Self {
            precision,
            factor,
            row_weights: target.row_weights.clone(),
        })
    }

    #[inline]
    fn row_weight(&self, n: usize) -> f64 {
        self.row_weights.as_ref().map_or(1.0, |w| w[n])
    }
}

impl VectorLikelihood for GaussianVectorLikelihood {
    fn log_lik(&self, eta: ArrayView2<f64>, y: ArrayView2<f64>) -> Result<f64, EstimationError> {
        validate_vector_likelihood_inputs(
            "GaussianVectorLikelihood::log_lik",
            eta,
            y,
            Some(self.precision.len()),
        )?;
        let m = eta.ncols();
        let rank = self.factor.as_ref().map_or(0, |f| f.ncols());
        let mut acc = 0.0;
        // Scratch buffer for Fᵀ r (length rank), reused across rows.
        let mut ftr = vec![0.0f64; rank];
        for n in 0..eta.nrows() {
            let w = self.row_weight(n);
            // Diagonal part: Σ_m d_m r_m²
            let mut row_acc = 0.0;
            for j in 0..m {
                let r = y[[n, j]] - eta[[n, j]];
                row_acc += self.precision[j] * r * r;
            }
            // Low-rank part: ||Fᵀ r||²
            if let Some(f) = self.factor.as_ref() {
                for k in 0..rank {
                    ftr[k] = 0.0;
                }
                for j in 0..m {
                    let r = y[[n, j]] - eta[[n, j]];
                    for k in 0..rank {
                        ftr[k] += f[[j, k]] * r;
                    }
                }
                for k in 0..rank {
                    row_acc += ftr[k] * ftr[k];
                }
            }
            acc += w * row_acc;
        }
        Ok(-0.5 * acc)
    }

    fn grad_eta(
        &self,
        eta: ArrayView2<f64>,
        y: ArrayView2<f64>,
    ) -> Result<Array2<f64>, EstimationError> {
        validate_vector_likelihood_inputs(
            "GaussianVectorLikelihood::grad_eta",
            eta,
            y,
            Some(self.precision.len()),
        )?;
        let (n_rows, n_cols) = eta.dim();
        let rank = self.factor.as_ref().map_or(0, |f| f.ncols());
        let mut out = Array2::<f64>::zeros((n_rows, n_cols));
        let mut ftr = vec![0.0f64; rank];
        for n in 0..n_rows {
            let w = self.row_weight(n);
            // Diagonal part: w · d_m · (y − η)_m
            for j in 0..n_cols {
                out[[n, j]] = w * self.precision[j] * (y[[n, j]] - eta[[n, j]]);
            }
            // Low-rank part: + w · F (Fᵀ r) for r = y − η
            if let Some(f) = self.factor.as_ref() {
                for k in 0..rank {
                    ftr[k] = 0.0;
                }
                for j in 0..n_cols {
                    let r = y[[n, j]] - eta[[n, j]];
                    for k in 0..rank {
                        ftr[k] += f[[j, k]] * r;
                    }
                }
                for j in 0..n_cols {
                    let mut s = 0.0;
                    for k in 0..rank {
                        s += f[[j, k]] * ftr[k];
                    }
                    out[[n, j]] += w * s;
                }
            }
        }
        Ok(out)
    }

    fn hess_diag(
        &self,
        eta: ArrayView2<f64>,
        y: ArrayView2<f64>,
    ) -> Result<Array2<f64>, EstimationError> {
        validate_vector_likelihood_inputs(
            "GaussianVectorLikelihood::hess_diag",
            eta,
            y,
            Some(self.precision.len()),
        )?;
        // Diagonal of −∂² log p / ∂η² = w · diag(diag(d) + F·Fᵀ); the diagonal
        // of (F·Fᵀ) at output m is Σ_k F[m, k]². This is the diagonal
        // *preconditioner* only — the off-diagonal cross terms F[a, k]·F[b, k]
        // are carried by the full per-row block in [`Self::hess_block`] (which
        // this type overrides whenever `factor` is present). Callers that need
        // the true Hessian must use `hess_block`, not this diagonal.
        let (n_rows, n_cols) = eta.dim();
        let mut out = Array2::<f64>::zeros((n_rows, n_cols));
        // Pre-compute Σ_k F[m, k]² per output m (independent of n).
        let f_row_sqsum: Option<Array1<f64>> = self.factor.as_ref().map(|f| {
            let m = f.nrows();
            let r = f.ncols();
            let mut s = Array1::<f64>::zeros(m);
            for j in 0..m {
                let mut acc = 0.0;
                for k in 0..r {
                    let v = f[[j, k]];
                    acc += v * v;
                }
                s[j] = acc;
            }
            s
        });
        for n in 0..n_rows {
            let w = self.row_weight(n);
            for j in 0..n_cols {
                let mut d = self.precision[j];
                if let Some(s) = f_row_sqsum.as_ref() {
                    d += s[j];
                }
                out[[n, j]] = w * d;
            }
        }
        Ok(out)
    }

    fn hess_block(
        &self,
        eta: ArrayView2<f64>,
        y: ArrayView2<f64>,
    ) -> Result<Array3<f64>, EstimationError> {
        // Per-row dense block −∂² log p / ∂η_a ∂η_b. With log-likelihood
        //     ℓ = −½ Σ_n w_n · rₙᵀ W rₙ,   r = y − η,   W = diag(precision) + F·Fᵀ,
        // the gradient is wₙ · W rₙ and the negative Hessian block is exactly
        //     H_{n,a,b} = w_n · ( precision_a · δ_ab + Σ_k F[a,k] · F[b,k] ).
        // This is the true second derivative of `log_lik` (it differentiates
        // `grad_eta` exactly); the diagonal-only trait default would drop the
        // F·Fᵀ cross terms F[a,k]·F[b,k] for a ≠ b, so it must be overridden
        // whenever a low-rank factor is present.
        validate_vector_likelihood_inputs(
            "GaussianVectorLikelihood::hess_block",
            eta,
            y,
            Some(self.precision.len()),
        )?;
        let (n_rows, m) = eta.dim();
        let rank = self.factor.as_ref().map_or(0, |f| f.ncols());

        // Per-output Gram of the low-rank factor, G_{a,b} = Σ_k F[a,k]·F[b,k].
        // Independent of the row n, so assemble once and scale by w_n.
        let gram: Option<Array2<f64>> = self.factor.as_ref().map(|f| {
            let mut g = Array2::<f64>::zeros((m, m));
            for a in 0..m {
                for b in a..m {
                    let mut acc = 0.0;
                    for k in 0..rank {
                        acc += f[[a, k]] * f[[b, k]];
                    }
                    g[[a, b]] = acc;
                    g[[b, a]] = acc;
                }
            }
            g
        });

        let mut out = Array3::<f64>::zeros((n_rows, m, m));
        for n in 0..n_rows {
            let w = self.row_weight(n);
            for a in 0..m {
                for b in 0..m {
                    let mut val = if a == b { self.precision[a] } else { 0.0 };
                    if let Some(g) = gram.as_ref() {
                        val += g[[a, b]];
                    }
                    out[[n, a, b]] = w * val;
                }
            }
        }
        Ok(out)
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Piece 5 / Piece 1 row-block support
// ─────────────────────────────────────────────────────────────────────────────

/// Multinomial-logit (softmax) likelihood with explicit reference class.
///
/// Conventions:
/// - `K` is the total number of classes; the linear predictor has `M = K - 1`
///   columns corresponding to the *active* classes. Class `K - 1` is the
///   reference class with η_{K-1} ≡ 0 (so the gauge is fixed by construction
///   and no additional sum-to-zero projection is required at the η level).
/// - `y` is the categorical response with shape `(N, K)`. Each row must be a
///   point on the probability simplex (`y_c ≥ 0`, `Σ_c y_c = 1`): a one-hot
///   indicator for hard-label classification, or a label-smoothed probability
///   vector. The row *weight* `w_n` scales the whole row's likelihood
///   contribution and is independent of the row mass — it is **not** the row
///   sum. Callers enforce the simplex precondition via
///   [`validate_multinomial_simplex`] at every construction boundary; under it
///   the residual gradient `y_a − p_a` and Fisher block `p_a δ_ab − p_a p_b`
///   below are the exact derivatives of the log-likelihood `Σ_c y_c log p_c`.
/// - `eta` is the active linear predictor with shape `(N, M = K - 1)`.
///
/// Softmax with baseline:
/// ```text
///     p_a   = exp(η_a) / (1 + Σ_b exp(η_b))           for a ∈ [0, K-1)
///     p_{K-1} = 1 / (1 + Σ_b exp(η_b))
/// ```
///
/// Log-likelihood (rows with weight `w_n`, default 1.0):
/// ```text
///     log L = Σ_n w_n · ( Σ_{a < K-1} y_{n,a} · η_{n,a} − log(1 + Σ_b exp(η_{n,b})) )
///           = Σ_n w_n · Σ_{c ∈ [0, K)} y_{n,c} · log p_{n,c}
/// ```
///
/// Per-row gradient w.r.t. the active η is the canonical Bernoulli/softmax
/// residual:
/// ```text
///     ∂ log L / ∂η_{n,a} = w_n · (y_{n,a} − p_{n,a})       for a ∈ [0, K-1)
/// ```
///
/// Per-row Fisher (= observed, since logit is canonical for the multinomial)
/// information block, shape `(M, M)`:
/// ```text
///     H_{n,a,b} = w_n · ( p_{n,a} · δ_{ab} − p_{n,a} · p_{n,b} )
/// ```
///
/// This is the standard reference-coded multinomial-logit GLM. The dense
/// per-row block flows through [`VectorLikelihood::hess_block`] into
/// [`gam_solve::pirls::dense_block_xtwx`], which builds the stacked
/// `XᵀWX` in output-major coefficient ordering `β = [β_0; β_1; …; β_{K-2}]`
/// with each per-class block of size `(P, P)`.
#[derive(Clone, Debug)]
pub struct MultinomialLogitLikelihood {
    /// Number of active classes `M = K − 1`. Cached for shape checks.
    pub active_classes: usize,
    /// Optional row weights (length N), or `None` for uniform 1.0.
    pub row_weights: Option<Array1<f64>>,
}

impl MultinomialLogitLikelihood {
    /// Construct from the total number of classes `K ≥ 2`.
    pub fn with_classes(total_classes: usize) -> Result<Self, EstimationError> {
        if total_classes < 2 {
            crate::bail_invalid_estim!(
                "MultinomialLogitLikelihood requires K ≥ 2 classes (got {total_classes})"
            );
        }
        Ok(Self {
            active_classes: total_classes - 1,
            row_weights: None,
        })
    }

    /// Attach per-row weights (length N, finite and non-negative).
    pub fn with_row_weights(mut self, w: Array1<f64>) -> Result<Self, EstimationError> {
        validate_row_weights(&w, w.len())?;
        self.row_weights = Some(w);
        Ok(self)
    }

    /// Total class count `K = M + 1`.
    #[inline]
    pub fn total_classes(&self) -> usize {
        self.active_classes + 1
    }

    #[inline]
    fn row_weight(&self, n: usize) -> f64 {
        self.row_weights.as_ref().map_or(1.0, |w| w[n])
    }

    /// Numerically-stable softmax with implicit reference column (η_{K-1} = 0).
    ///
    /// Writes `K` probabilities into `out` (length `M + 1`). The shift uses
    /// `max(0, max(eta_active))` so the reference class is included in the
    /// max and the denominator stays bounded. This is the canonical
    /// reference implementation; the FFI surface and any direct
    /// matrix-free callers route through this method rather than carrying
    /// their own softmax.
    pub fn softmax_with_baseline(eta_active: &[f64], out: &mut [f64]) {
        multinomial_logit_probabilities_into(eta_active, out);
    }

    /// Convenience: compute the full (N, K) probability matrix from
    /// (N, K-1) active linear predictor. This is the multinomial inverse
    /// link used by prediction.
    pub fn probabilities(&self, eta: ArrayView2<f64>) -> Array2<f64> {
        let n = eta.nrows();
        let m = self.active_classes;
        assert_eq!(eta.ncols(), m, "η must have K-1 columns");
        let k = self.total_classes();
        let mut probs = Array2::<f64>::zeros((n, k));
        let mut eta_row = vec![0.0_f64; m];
        let mut probs_row = vec![0.0_f64; k];
        for row in 0..n {
            for j in 0..m {
                eta_row[j] = eta[[row, j]];
            }
            Self::softmax_with_baseline(&eta_row, &mut probs_row);
            for j in 0..k {
                probs[[row, j]] = probs_row[j];
            }
        }
        probs
    }

    #[inline]
    fn row_program<'row>(
        &self,
        row: usize,
        eta: &'row [f64],
        response: &'row [f64],
    ) -> Result<MultinomialLogitRowProgram<'row>, EstimationError> {
        MultinomialLogitRowProgram::new(eta, response, self.row_weight(row)).map_err(|error| {
            EstimationError::InvalidInput(format!("invalid multinomial row {row}: {error}"))
        })
    }

    /// Fused live value/gradient/Hessian evaluation. This is the one production
    /// batch entry used by the joint REML adapter, so it performs one stable
    /// normalization per row rather than three independent likelihood passes.
    pub(crate) fn value_gradient_hessian(
        &self,
        eta: ArrayView2<f64>,
        y: ArrayView2<f64>,
    ) -> Result<(f64, Array2<f64>, Array3<f64>), EstimationError> {
        let n = eta.nrows();
        let m = self.active_classes;
        let k = self.total_classes();
        if y.dim() != (n, k) {
            crate::bail_invalid_estim!(
                "MultinomialLogitLikelihood::value_gradient_hessian: response shape {:?} must be ({n}, {k})",
                y.dim()
            );
        }
        validate_vector_likelihood_inputs(
            "MultinomialLogitLikelihood::value_gradient_hessian active response",
            eta,
            y.slice(ndarray::s![.., ..m]),
            Some(m),
        )?;
        let mut gradient_log_likelihood = Array2::<f64>::zeros((n, m));
        let mut hessian = Array3::<f64>::zeros((n, m, m));
        let mut eta_row = vec![0.0_f64; m];
        let mut response_row = vec![0.0_f64; k];
        let mut probabilities = vec![0.0_f64; k];
        let mut gradient_nll = vec![0.0_f64; m];
        let mut hessian_row = vec![0.0_f64; m * m];
        let mut negative_log_likelihood = 0.0_f64;
        for row in 0..n {
            for axis in 0..m {
                eta_row[axis] = eta[[row, axis]];
            }
            for class in 0..k {
                response_row[class] = y[[row, class]];
            }
            let program = self.row_program(row, &eta_row, &response_row)?;
            negative_log_likelihood += program.value_gradient_hessian_into(
                &mut probabilities,
                &mut gradient_nll,
                &mut hessian_row,
            );
            for axis in 0..m {
                gradient_log_likelihood[[row, axis]] = -gradient_nll[axis];
                for other in 0..m {
                    hessian[[row, axis, other]] = hessian_row[axis * m + other];
                }
            }
        }
        Ok((-negative_log_likelihood, gradient_log_likelihood, hessian))
    }

    /// Fused value/gradient entry for callers that do not consume curvature.
    pub(crate) fn value_gradient(
        &self,
        eta: ArrayView2<f64>,
        y: ArrayView2<f64>,
    ) -> Result<(f64, Array2<f64>), EstimationError> {
        let n = eta.nrows();
        let m = self.active_classes;
        let k = self.total_classes();
        if y.dim() != (n, k) {
            crate::bail_invalid_estim!(
                "MultinomialLogitLikelihood::value_gradient: response shape {:?} must be ({n}, {k})",
                y.dim()
            );
        }
        validate_vector_likelihood_inputs(
            "MultinomialLogitLikelihood::value_gradient active response",
            eta,
            y.slice(ndarray::s![.., ..m]),
            Some(m),
        )?;
        let mut gradient_log_likelihood = Array2::<f64>::zeros((n, m));
        let mut eta_row = vec![0.0_f64; m];
        let mut response_row = vec![0.0_f64; k];
        let mut probabilities = vec![0.0_f64; k];
        let mut gradient_nll = vec![0.0_f64; m];
        let mut negative_log_likelihood = 0.0_f64;
        for row in 0..n {
            for axis in 0..m {
                eta_row[axis] = eta[[row, axis]];
            }
            for class in 0..k {
                response_row[class] = y[[row, class]];
            }
            let program = self.row_program(row, &eta_row, &response_row)?;
            negative_log_likelihood +=
                program.value_gradient_into(&mut probabilities, &mut gradient_nll);
            for axis in 0..m {
                gradient_log_likelihood[[row, axis]] = -gradient_nll[axis];
            }
        }
        Ok((-negative_log_likelihood, gradient_log_likelihood))
    }
}

impl VectorLikelihood for MultinomialLogitLikelihood {
    fn log_lik(&self, eta: ArrayView2<f64>, y: ArrayView2<f64>) -> Result<f64, EstimationError> {
        let n = eta.nrows();
        let m = self.active_classes;
        let k = self.total_classes();
        if y.dim() != (n, k) {
            crate::bail_invalid_estim!(
                "MultinomialLogitLikelihood::log_lik: response shape {:?} must be ({n}, {k})",
                y.dim()
            );
        }
        validate_vector_likelihood_inputs(
            "MultinomialLogitLikelihood::log_lik active response",
            eta,
            y.slice(ndarray::s![.., ..m]),
            Some(m),
        )?;
        let mut eta_row = vec![0.0_f64; m];
        let mut response_row = vec![0.0_f64; k];
        let mut negative_log_likelihood = 0.0_f64;
        for row in 0..n {
            for axis in 0..m {
                eta_row[axis] = eta[[row, axis]];
            }
            for class in 0..k {
                response_row[class] = y[[row, class]];
            }
            negative_log_likelihood += self
                .row_program(row, &eta_row, &response_row)?
                .negative_log_likelihood();
        }
        Ok(-negative_log_likelihood)
    }

    fn grad_eta(
        &self,
        eta: ArrayView2<f64>,
        y: ArrayView2<f64>,
    ) -> Result<Array2<f64>, EstimationError> {
        Ok(self.value_gradient(eta, y)?.1)
    }

    fn hess_diag(
        &self,
        eta: ArrayView2<f64>,
        y: ArrayView2<f64>,
    ) -> Result<Array2<f64>, EstimationError> {
        // Per-row diagonal of the (M, M) Fisher block:
        //     H_{n,a,a} = w_n · p_{n,a} · (1 − p_{n,a})
        // Provided for callers that explicitly want the diagonal-only
        // preconditioner; the joint dense block ships through `hess_block`.
        let n = eta.nrows();
        let m = self.active_classes;
        let k = self.total_classes();
        if y.dim() != (n, k) {
            crate::bail_invalid_estim!(
                "MultinomialLogitLikelihood::hess_diag: response shape {:?} must be ({n}, {k})",
                y.dim()
            );
        }
        validate_vector_likelihood_inputs(
            "MultinomialLogitLikelihood::hess_diag active response",
            eta,
            y.slice(ndarray::s![.., ..m]),
            Some(m),
        )?;
        let mut out = Array2::<f64>::zeros((n, m));
        let mut eta_row = vec![0.0_f64; m];
        let mut response_row = vec![0.0_f64; k];
        let mut probabilities = vec![0.0_f64; k];
        let mut diagonal = vec![0.0_f64; m];
        for row in 0..n {
            for axis in 0..m {
                eta_row[axis] = eta[[row, axis]];
            }
            for class in 0..k {
                response_row[class] = y[[row, class]];
            }
            self.row_program(row, &eta_row, &response_row)?
                .hessian_diagonal_into(&mut probabilities, &mut diagonal);
            for axis in 0..m {
                out[[row, axis]] = diagonal[axis];
            }
        }
        Ok(out)
    }

    fn hess_block(
        &self,
        eta: ArrayView2<f64>,
        y: ArrayView2<f64>,
    ) -> Result<Array3<f64>, EstimationError> {
        Ok(self.value_gradient_hessian(eta, y)?.2)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::{Array1, Array2};

    // Macro (not fn) so the assertion / panic tokens are inlined into each
    // caller's test body, satisfying the build.rs scanner that looks for
    // `assert!(` / `panic!(` directly in the `#[test]` function.
    macro_rules! expect_invalid_input {
        ($result:expr, $needle:expr $(,)?) => {{
            let needle: &str = $needle;
            match $result {
                Ok(_) => {
                    panic!("expected EstimationError::InvalidInput containing `{needle}`, got Ok")
                }
                Err(EstimationError::InvalidInput(msg)) => {
                    assert!(
                        msg.contains(needle),
                        "InvalidInput message `{msg}` does not contain `{needle}`"
                    );
                    msg
                }
                Err(other) => panic!(
                    "expected EstimationError::InvalidInput containing `{needle}`, got {other:?}"
                ),
            }
        }};
    }

    fn dummy_target(n: usize, m: usize) -> VectorResponseTarget {
        VectorResponseTarget::new(Array2::<f64>::zeros((n, m)), VectorNoise::Isotropic(1.0))
    }

    #[test]
    fn with_row_weights_rejects_wrong_length() {
        let target = dummy_target(4, 2);
        let weights = Array1::from(vec![1.0, 1.0, 1.0]);
        expect_invalid_input!(target.with_row_weights(weights), "row_weights length");
    }

    #[test]
    fn with_row_weights_rejects_negative_entry() {
        let target = dummy_target(3, 2);
        let weights = Array1::from(vec![1.0, -0.5, 2.0]);
        expect_invalid_input!(
            target.with_row_weights(weights),
            "must be finite and non-negative",
        );
    }

    #[test]
    fn with_row_weights_rejects_nan_entry() {
        let target = dummy_target(3, 2);
        let weights = Array1::from(vec![1.0, f64::NAN, 2.0]);
        expect_invalid_input!(
            target.with_row_weights(weights),
            "must be finite and non-negative",
        );
    }

    #[test]
    fn with_row_weights_rejects_infinite_entry() {
        let target = dummy_target(3, 2);
        let weights = Array1::from(vec![1.0, f64::INFINITY, 2.0]);
        expect_invalid_input!(
            target.with_row_weights(weights),
            "must be finite and non-negative",
        );
    }

    #[test]
    fn with_row_weights_accepts_zero_and_positive() {
        let target = dummy_target(3, 2);
        let weights = Array1::from(vec![0.0, 1.5, 3.0]);
        let weighted = target
            .with_row_weights(weights)
            .expect("zero / positive weights should be accepted");
        assert!(weighted.row_weights.is_some());
    }

    #[test]
    fn from_target_rejects_low_rank_factor_with_wrong_row_count() {
        let n = 4;
        let m = 3;
        // factor has 2 rows instead of M = 3.
        let factor = Array2::from_shape_vec((2, 2), vec![0.1, 0.2, 0.3, 0.4]).unwrap();
        let target = VectorResponseTarget::new(
            Array2::<f64>::zeros((n, m)),
            VectorNoise::LowRank {
                diag: Array1::from(vec![1.0; m]),
                factor,
            },
        );
        expect_invalid_input!(GaussianVectorLikelihood::from_target(&target), "factor has",);
    }

    #[test]
    fn from_target_rejects_non_finite_low_rank_factor_entry() {
        let n = 4;
        let m = 3;
        let mut factor = Array2::<f64>::zeros((m, 2));
        factor[[1, 0]] = f64::NAN;
        let target = VectorResponseTarget::new(
            Array2::<f64>::zeros((n, m)),
            VectorNoise::LowRank {
                diag: Array1::from(vec![1.0; m]),
                factor,
            },
        );
        expect_invalid_input!(
            GaussianVectorLikelihood::from_target(&target),
            "must be finite",
        );
    }

    #[test]
    fn from_target_accepts_well_formed_low_rank_factor() {
        let n = 2;
        let m = 3;
        let factor = Array2::from_shape_vec((m, 2), vec![0.1, 0.2, 0.3, 0.4, 0.5, 0.6]).unwrap();
        let target = VectorResponseTarget::new(
            Array2::<f64>::zeros((n, m)),
            VectorNoise::LowRank {
                diag: Array1::from(vec![1.0; m]),
                factor: factor.clone(),
            },
        );
        let lik = GaussianVectorLikelihood::from_target(&target)
            .expect("well-formed low-rank factor should be accepted");
        let stored = lik.factor.expect("low-rank factor should be carried");
        assert_eq!(stored.dim(), (m, 2));
        for ((i, j), v) in stored.indexed_iter() {
            assert_eq!(*v, factor[[i, j]]);
        }
        // `GaussianVectorLikelihood::precision` is the per-output diagonal
        // of length `M`, populated from `target.noise.diag_precision(M)`
        // — not a per-row precision of length `N`. The historical
        // `assert_eq!(n, lik.precision.len().max(n))` reduces to
        // `precision.len() ≤ n`, which is the opposite of the contract
        // (and false for any `M > N`, the typical multivariate-response
        // shape).
        assert_eq!(m, lik.precision.len());
    }

    #[test]
    fn from_target_propagates_row_weight_length_mismatch() {
        let n = 3;
        let m = 2;
        let target = VectorResponseTarget {
            y: Array2::<f64>::zeros((n, m)),
            noise: VectorNoise::Isotropic(1.0),
            row_weights: Some(Array1::from(vec![1.0, 1.0])),
        };
        expect_invalid_input!(
            GaussianVectorLikelihood::from_target(&target),
            "row_weights length",
        );
    }

    #[test]
    fn vector_likelihood_rejects_nonfinite_optimizer_state_without_panicking_932() {
        let target = dummy_target(1, 2);
        let likelihood =
            GaussianVectorLikelihood::from_target(&target).expect("finite Gaussian vector target");
        let eta = Array2::from_shape_vec((1, 2), vec![0.0, f64::NAN]).expect("eta shape");
        expect_invalid_input!(
            likelihood.log_lik(eta.view(), target.y.view()),
            "eta[0,1] must be finite",
        );
    }

    #[test]
    fn multinomial_row_validation_propagates_as_typed_likelihood_error_932() {
        let likelihood = MultinomialLogitLikelihood::with_classes(3)
            .expect("three-class reference-coded likelihood");
        let eta =
            Array2::from_shape_vec((1, 2), vec![f64::INFINITY, 0.0]).expect("active eta shape");
        let response =
            Array2::from_shape_vec((1, 3), vec![1.0, 0.0, 0.0]).expect("simplex response shape");
        expect_invalid_input!(
            likelihood.log_lik(eta.view(), response.view()),
            "eta[0,0] must be finite",
        );
    }
}
