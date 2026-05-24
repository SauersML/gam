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

use crate::estimate::EstimationError;
use ndarray::{Array1, Array2, ArrayView2};

/// Per-output noise model for a vector response.
///
/// `LowRank` stores the symmetric structured precision
/// `W = diag(diag) + U Uᵀ`, with `factor` holding `U`. The vector likelihood
/// consumes the owned arrays directly; PIRLS low-rank Gram assembly is handled
/// by `crate::linalg::low_rank_weight::LowRankWeight` and
/// `crate::solver::pirls`.
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
                    return Err(EstimationError::InvalidInput(format!(
                        "VectorNoise::Isotropic: σ must be > 0 and finite (got {sigma})",
                    )));
                }
                let p = 1.0 / (sigma * sigma);
                Ok(Array1::from_elem(m, p))
            }
            Self::Diagonal(sigma) => {
                if sigma.len() != m {
                    return Err(EstimationError::InvalidInput(format!(
                        "VectorNoise::Diagonal: σ length {} ≠ M={m}",
                        sigma.len()
                    )));
                }
                let mut out = Array1::<f64>::zeros(m);
                for j in 0..m {
                    let s = sigma[j];
                    if !s.is_finite() || s <= 0.0 {
                        return Err(EstimationError::InvalidInput(format!(
                            "VectorNoise::Diagonal: σ[{j}] must be > 0 and finite (got {s})",
                        )));
                    }
                    out[j] = 1.0 / (s * s);
                }
                Ok(out)
            }
            Self::LowRank { diag, .. } => {
                if diag.len() != m {
                    return Err(EstimationError::InvalidInput(format!(
                        "VectorNoise::LowRank: diag length {} ≠ M={m}",
                        diag.len()
                    )));
                }
                let mut out = Array1::<f64>::zeros(m);
                for j in 0..m {
                    let d = diag[j];
                    if !d.is_finite() || d <= 0.0 {
                        return Err(EstimationError::InvalidInput(format!(
                            "VectorNoise::LowRank: diag[{j}] must be > 0 (got {d})",
                        )));
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
        // TODO(coverage): add test exercising non-finite and negative row-weight rejection
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

fn validate_row_weights(weights: &Array1<f64>, n: usize) -> Result<(), EstimationError> {
    if weights.len() != n {
        return Err(EstimationError::InvalidInput(format!(
            "row_weights length {} ≠ N={n}",
            weights.len()
        )));
    }
    for (idx, weight) in weights.iter().copied().enumerate() {
        if !(weight.is_finite() && weight >= 0.0) {
            return Err(EstimationError::InvalidInput(format!(
                "row_weights[{idx}] must be finite and non-negative (got {weight})"
            )));
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
    fn log_lik(&self, eta: ArrayView2<f64>, y: ArrayView2<f64>) -> f64;

    /// ∂ log p(Y | η) / ∂ η, shape (N, M).
    fn grad_eta(&self, eta: ArrayView2<f64>, y: ArrayView2<f64>) -> Array2<f64>;

    /// Diagonal of the per-row Hessian −∂² log p / ∂ η ∂ η, shape (N, M).
    /// This is the per-row block consumed by `solver/arrow_schur.rs`.
    fn hess_diag(&self, eta: ArrayView2<f64>, y: ArrayView2<f64>) -> Array2<f64>;
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
        // TODO(coverage): add test exercising low-rank factor shape and non-finite entries
        if let Some(weights) = target.row_weights.as_ref() {
            validate_row_weights(weights, target.n())?;
        }
        let precision = target.noise.diag_precision(target.m())?;
        let factor = match &target.noise {
            VectorNoise::LowRank { factor, .. } => {
                if factor.nrows() != target.m() {
                    return Err(EstimationError::InvalidInput(format!(
                        "VectorNoise::LowRank: factor has {} rows but M={}",
                        factor.nrows(),
                        target.m()
                    )));
                }
                for ((row, col), value) in factor.indexed_iter() {
                    if !value.is_finite() {
                        return Err(EstimationError::InvalidInput(format!(
                            "VectorNoise::LowRank: factor[{row},{col}] must be finite (got {value})"
                        )));
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
    fn log_lik(&self, eta: ArrayView2<f64>, y: ArrayView2<f64>) -> f64 {
        debug_assert_eq!(eta.dim(), y.dim());
        debug_assert_eq!(eta.ncols(), self.precision.len());
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
        -0.5 * acc
    }

    fn grad_eta(&self, eta: ArrayView2<f64>, y: ArrayView2<f64>) -> Array2<f64> {
        debug_assert_eq!(eta.dim(), y.dim());
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
        out
    }

    fn hess_diag(&self, eta: ArrayView2<f64>, y: ArrayView2<f64>) -> Array2<f64> {
        std::hint::black_box(y);
        // −∂² log p / ∂η² = w · (diag(d) + F·Fᵀ); the diagonal of (F·Fᵀ)
        // at output m is Σ_k F[m, k]². Cross terms F[m, k]·F[m', k] live in
        // the dense rank-r correction returned by `hess_full` (only exposed
        // via the `GaussianVectorLikelihood` API; the `VectorLikelihood`
        // trait sees the diagonal preconditioner).
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
        out
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Piece 5 / Piece 1 row-block support
// ─────────────────────────────────────────────────────────────────────────────
