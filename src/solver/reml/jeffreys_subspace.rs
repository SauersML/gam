//! Shared under-identified-subspace selector for the universal Jeffreys/Firth
//! robustness machinery.
//!
//! The Jeffreys penalty `Phi = 1/2 log|I(beta)|` is only ever applied to the
//! directions that are identified by NEITHER the data nor a proper prior — the
//! "under-identified span". Penalized smooth directions already carry a proper
//! wiggliness prior (their `S_lambda` curvature), so applying Jeffreys there
//! would double-regularize and bias the smooth fit. This module produces the
//! orthonormal basis `Z_J` of that span for one parameter block.
//!
//! For a block with aggregate penalty `S = sum_k S_k`, the under-identified
//! span is exactly `ker(S)` — the penalty null space, which always contains the
//! parametric (unpenalized) part and the structural null space of every smooth
//! penalty (the polynomial/affine basis a difference/curvature penalty cannot
//! see). A block with no penalties at all (a pure parametric block) is entirely
//! under-identified, so `Z_J = I`.
//!
//! Both tiers of the robustness machinery consume the SAME `Z_J`:
//!   * Tier A (single-eta GLM via `FirthDenseOperator`) scopes the Fisher
//!     information to `X * Z_J`.
//!   * Tier B (coupled multi-predictor custom-family joint Newton, e.g. BMS)
//!     restricts the joint-Hessian Jeffreys term `Phi_J = 1/2 log|Z_J^T H Z_J|`
//!     to the same span.
//!
//! Everything here is pure linear algebra on the block's penalty matrices and
//! is gated upstream by `RobustConfig` (default OFF), so it never runs in the
//! released solver until a caller opts in.

use crate::linalg::faer_ndarray::FaerEigh;
use faer::Side;
use ndarray::{Array2, ArrayView2};

/// Relative threshold (against the largest aggregate-penalty eigenvalue) below
/// which an eigen-direction counts as a penalty-null (under-identified)
/// direction. Mirrors the conservative ratio the penalty pseudo-logdet uses.
const PENALTY_NULL_RELATIVE_TOL: f64 = 1e-9;

/// Orthonormal basis of one block's under-identified span.
///
/// `columns` is `p x m` with orthonormal columns spanning `ker(S_aggregate)`
/// (the parametric + smooth-null directions). `m == 0` means the block is fully
/// penalized in every direction and gets no Jeffreys term.
#[derive(Debug, Clone)]
pub struct JeffreysSubspace {
    /// `p x m` orthonormal basis of the under-identified span (m <= p).
    pub columns: Array2<f64>,
}

impl JeffreysSubspace {
    /// Block dimension `p` (rows of the basis).
    #[inline]
    pub fn block_dim(&self) -> usize {
        self.columns.nrows()
    }

    /// Dimension `m` of the under-identified span (columns of the basis).
    #[inline]
    pub fn span_dim(&self) -> usize {
        self.columns.ncols()
    }

    /// True when the block has no under-identified direction (fully penalized).
    #[inline]
    pub fn is_empty(&self) -> bool {
        self.columns.ncols() == 0
    }
}

/// Build `Z_J` for a block from its aggregate (unit-weight) penalty matrix.
///
/// `aggregate_penalty` is `p x p` and PSD: pass `sum_k S_k` (any positive
/// weighting is fine — only the null space matters, and scaling preserves it).
/// `structural_nullity`, when `Some(m0)`, pins the null-space dimension exactly
/// (from `ParameterBlockSpec::nullspace_dims` intersection), bypassing the
/// numerical eigenvalue threshold. When `None`, a relative eigenvalue cutoff is
/// used.
///
/// Returns the bottom-`m0` eigenvectors of the aggregate penalty (its null
/// space) as an orthonormal `p x m0` basis. For an all-zero penalty (pure
/// parametric block) this is the full identity, as required.
pub fn jeffreys_subspace_from_penalty(
    aggregate_penalty: ArrayView2<'_, f64>,
    structural_nullity: Option<usize>,
) -> Result<JeffreysSubspace, String> {
    let p = aggregate_penalty.nrows();
    if aggregate_penalty.ncols() != p {
        return Err(format!(
            "jeffreys_subspace: aggregate penalty must be square, got {}x{}",
            aggregate_penalty.nrows(),
            aggregate_penalty.ncols()
        ));
    }
    if p == 0 {
        return Ok(JeffreysSubspace {
            columns: Array2::zeros((0, 0)),
        });
    }
    let frobenius = aggregate_penalty.iter().map(|v| v * v).sum::<f64>().sqrt();
    if frobenius == 0.0 {
        // Pure parametric block: the entire coefficient space is
        // under-identified by the penalty, so Z_J is the identity.
        return Ok(JeffreysSubspace {
            columns: Array2::eye(p),
        });
    }
    let owned = aggregate_penalty.to_owned();
    let (evals, evecs) = owned
        .eigh(Side::Lower)
        .map_err(|e| format!("jeffreys_subspace: penalty eigendecomposition failed: {e}"))?;
    // faer returns ascending eigenvalues, so the bottom `m0` columns of `evecs`
    // span the penalty null space (the under-identified directions).
    let lambda_max = evals.iter().cloned().fold(0.0_f64, f64::max);
    let m0 = match structural_nullity {
        Some(m0) => m0.min(p),
        None => {
            let cutoff = PENALTY_NULL_RELATIVE_TOL * lambda_max.max(f64::MIN_POSITIVE);
            evals.iter().filter(|&&e| e <= cutoff).count()
        }
    };
    if m0 == 0 {
        return Ok(JeffreysSubspace {
            columns: Array2::zeros((p, 0)),
        });
    }
    let columns = evecs.slice(ndarray::s![.., 0..m0]).to_owned();
    Ok(JeffreysSubspace { columns })
}
