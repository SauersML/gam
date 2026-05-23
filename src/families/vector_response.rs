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
/// `LowRank` is a placeholder that hands off to Piece 5's low-rank weight
/// machinery; the inner `factor` is the `L` in `Σ = diag(diag) + L L^T`.
#[derive(Clone, Debug)]
pub enum VectorNoise {
    /// Shared σ across all M outputs: Σ = σ² I_M.
    Isotropic(f64),
    /// Per-output σ_m: Σ = diag(σ_m²).
    Diagonal(Array1<f64>),
    /// Σ = diag(diag) + factor · factor^T.
    /// Routed through Piece 5's `LowRankWeight` once that lands.
    // TODO(piece-5): switch to the real `LowRankWeight` type from solver/.
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
                    out[j] = 1.0 / d;
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
        if w.len() != self.y.nrows() {
            return Err(EstimationError::InvalidInput(format!(
                "row_weights length {} ≠ N={}",
                w.len(),
                self.y.nrows()
            )));
        }
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
/// `log p(Y|η) = −½ Σ_n Σ_m w_n · prec_m · (Y_{n,m} − η_{n,m})²` (up to the
/// constant log-determinant of the noise covariance, dropped here because it
/// does not depend on β or the latent t; the determinant is accounted for in
/// the REML score, not the inner likelihood).
#[derive(Clone, Debug)]
pub struct GaussianVectorLikelihood {
    /// Per-output diagonal precision (length M). For Isotropic/Diagonal/LowRank
    /// this is `1/σ_m²` (LowRank's off-diagonal correction is applied through
    /// Piece 5's `LowRankWeight` outside this struct).
    pub precision: Array1<f64>,
    /// Optional row weights (length N), or None for uniform.
    pub row_weights: Option<Array1<f64>>,
}

impl GaussianVectorLikelihood {
    pub fn from_target(target: &VectorResponseTarget) -> Result<Self, EstimationError> {
        let precision = target.noise.diag_precision(target.m())?;
        Ok(Self {
            precision,
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
        let mut acc = 0.0;
        for n in 0..eta.nrows() {
            let w = self.row_weight(n);
            let mut row_acc = 0.0;
            for m in 0..eta.ncols() {
                let r = y[[n, m]] - eta[[n, m]];
                row_acc += self.precision[m] * r * r;
            }
            acc += w * row_acc;
        }
        -0.5 * acc
    }

    fn grad_eta(&self, eta: ArrayView2<f64>, y: ArrayView2<f64>) -> Array2<f64> {
        debug_assert_eq!(eta.dim(), y.dim());
        let (n_rows, n_cols) = eta.dim();
        let mut out = Array2::<f64>::zeros((n_rows, n_cols));
        for n in 0..n_rows {
            let w = self.row_weight(n);
            for m in 0..n_cols {
                // ∂/∂η_{n,m} of −½ w · prec_m · (y − η)² = w · prec_m · (y − η)
                out[[n, m]] = w * self.precision[m] * (y[[n, m]] - eta[[n, m]]);
            }
        }
        out
    }

    fn hess_diag(&self, eta: ArrayView2<f64>, _y: ArrayView2<f64>) -> Array2<f64> {
        // −∂² log p / ∂η² = w · prec_m, independent of η for Gaussian-identity.
        let (n_rows, n_cols) = eta.dim();
        let mut out = Array2::<f64>::zeros((n_rows, n_cols));
        for n in 0..n_rows {
            let w = self.row_weight(n);
            for m in 0..n_cols {
                out[[n, m]] = w * self.precision[m];
            }
        }
        out
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Piece 5 / Piece 1 integration hooks
// ─────────────────────────────────────────────────────────────────────────────

/// Placeholder for Piece 5's low-rank weight type. Once Piece 5 lands, this
/// alias is replaced with the real `crate::solver::low_rank_weight::LowRankWeight`.
// TODO(piece-5): replace this scaffold with the canonical re-export.
#[derive(Clone, Debug)]
pub struct LowRankWeightPlaceholder {
    pub diag: Array1<f64>,
    pub factor: Array2<f64>,
}

impl LowRankWeightPlaceholder {
    pub fn from_noise(noise: &VectorNoise) -> Option<Self> {
        match noise {
            VectorNoise::LowRank { diag, factor } => Some(Self {
                diag: diag.clone(),
                factor: factor.clone(),
            }),
            _ => None,
        }
    }
}

/// Per-row Hessian block consumed by the arrow Schur elimination in
/// `solver/arrow_schur.rs`.
///
/// For the Gaussian-identity case the block is diagonal, so we expose just the
/// `(M,)` diagonal here. When non-Gaussian likelihoods land (Bernoulli vector,
/// Poisson vector) this will grow a dense-block variant.
// per-row block consumed by solver/arrow_schur.rs
#[derive(Clone, Debug)]
pub enum PerRowHessianBlock {
    /// Diagonal block of length M.
    Diagonal(Array1<f64>),
    /// Dense (M, M) block (reserved for non-Gaussian vector likelihoods).
    Dense(Array2<f64>),
}

/// Build the per-row Hessian blocks for the arrow solver.
///
/// Returns a `Vec<PerRowHessianBlock>` of length N. For Gaussian likelihoods
/// each block is `Diagonal(M)`; the arrow Schur path can read these directly
/// without ever materialising an `(N·M, N·M)` Hessian.
pub fn per_row_hessian_blocks(
    likelihood: &dyn VectorLikelihood,
    eta: ArrayView2<f64>,
    y: ArrayView2<f64>,
) -> Vec<PerRowHessianBlock> {
    let diag = likelihood.hess_diag(eta, y);
    let n = diag.nrows();
    let mut out = Vec::with_capacity(n);
    for row in diag.outer_iter() {
        out.push(PerRowHessianBlock::Diagonal(row.to_owned()));
    }
    out
}

// ─────────────────────────────────────────────────────────────────────────────
// TODO: non-Gaussian vector likelihoods
// ─────────────────────────────────────────────────────────────────────────────
//
// Bernoulli vector: η_{n,m} = logit(p_{n,m}), Y ∈ {0,1}^{N×M}. log_lik is the
// usual logistic sum across (n, m); grad_eta = (y − sigmoid(η)); hess_diag =
// sigmoid(η)·(1−sigmoid(η)). Per-row block is diagonal (independence across
// outputs at fixed η, with the latent coupling handled by the arrow form).
//
// Poisson vector: η_{n,m} = log(λ_{n,m}). grad = y − exp(η);
// hess_diag = exp(η).
//
// These are deferred; the trait is already general enough — only the impl
// blocks need writing when the upstream link-inversion / quadrature plumbing
// is exposed to vector targets.
