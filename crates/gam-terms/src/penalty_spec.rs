//! Penalty specification for the public estimate API.
//!
//! `PenaltySpec` is a *penalty spec* keyed entirely on `gam-terms` penalty
//! types (`PenaltyStructureHint`, `PenaltyOp`, `BlockwisePenalty`) plus the
//! neutral `gam_problem::CoefficientPriorMean`. It therefore lives in
//! `gam-terms` (the layer that owns those penalty primitives); the solver
//! consumes it from above via `gam_terms::PenaltySpec`.
//!
//! Moved byte-identically from the monolith `src/model_types.rs` during the
//! `#1521` carve, with the only changes being the crate-local module paths
//! (`crate::terms::smooth::*` -> `crate::smooth::*`,
//! `crate::terms::analytic_penalties::*` -> `crate::analytic_penalties::*`)
//! and `EstimationError` sourced from `gam_problem`.

use std::ops::Range;

use ndarray::{Array2, s};

use crate::smooth::{BlockwisePenalty, PenaltyStructureHint};

/// Programmatic prior mean for a coefficient penalty block.
///
/// This type lives in the neutral `gam-problem` crate (with its inherent
/// `evaluate` returning `gam_problem::PriorMeanError`); re-exported here so all
/// existing `PenaltySpec`-adjacent references keep resolving. Solver-side
/// callers map `PriorMeanError` into `EstimationError::InvalidInput`.
pub use gam_problem::CoefficientPriorMean;
pub use gam_problem::EstimationError;

/// A penalty specification for the public estimate API.
///
/// `Block` stores only the active sub-block and its column range, avoiding
/// the O(p^2) cost of embedding into a full penalty matrix.
/// `Dense` stores a full `p x p` penalty matrix for callers that already
/// have one.
#[derive(Clone)]
pub enum PenaltySpec {
    /// Block-local penalty: `local` is `block_dim x block_dim`,
    /// applied to columns `col_range` of the coefficient vector.
    Block {
        local: Array2<f64>,
        col_range: Range<usize>,
        prior_mean: CoefficientPriorMean,
        /// Optional structural hint for fast-path spectral decomposition.
        structure_hint: Option<PenaltyStructureHint>,
        /// Optional operator-form handle bit-equivalent to `local`.
        op: Option<std::sync::Arc<dyn crate::analytic_penalties::PenaltyOp>>,
    },
    /// Full dense penalty matrix (`p x p`).
    Dense(Array2<f64>),
    /// Full dense penalty matrix with a programmatic prior mean in the same
    /// global coefficient basis.
    DenseWithMean {
        matrix: Array2<f64>,
        prior_mean: CoefficientPriorMean,
    },
}

impl std::fmt::Debug for PenaltySpec {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            PenaltySpec::Block {
                local,
                col_range,
                prior_mean,
                structure_hint,
                op,
            } => f
                .debug_struct("Block")
                .field(
                    "local",
                    &format_args!("{}×{}", local.nrows(), local.ncols()),
                )
                .field("col_range", col_range)
                .field("prior_mean", prior_mean)
                .field("structure_hint", structure_hint)
                .field("op", &op.as_ref().map(|o| o.dim()))
                .finish(),
            PenaltySpec::Dense(m) => f
                .debug_tuple("Dense")
                .field(&format_args!("{}×{}", m.nrows(), m.ncols()))
                .finish(),
            PenaltySpec::DenseWithMean { matrix, prior_mean } => f
                .debug_struct("DenseWithMean")
                .field(
                    "matrix",
                    &format_args!("{}×{}", matrix.nrows(), matrix.ncols()),
                )
                .field("prior_mean", prior_mean)
                .finish(),
        }
    }
}

impl PenaltySpec {
    /// The column range this penalty covers.
    /// For `Dense`, this is `0..p` where `p = m.ncols()`.
    pub fn col_range(&self, p: usize) -> Range<usize> {
        match self {
            PenaltySpec::Block { col_range, .. } => col_range.clone(),
            PenaltySpec::Dense(m) => {
                assert_eq!(m.ncols(), p);
                0..p
            }
            PenaltySpec::DenseWithMean { matrix, .. } => {
                assert_eq!(matrix.ncols(), p);
                0..p
            }
        }
    }

    /// Op-form handle when present (only for `Block`; `Dense` always returns `None`).
    pub fn op(&self) -> Option<&std::sync::Arc<dyn crate::analytic_penalties::PenaltyOp>> {
        match self {
            PenaltySpec::Block { op, .. } => op.as_ref(),
            PenaltySpec::Dense(_) | PenaltySpec::DenseWithMean { .. } => None,
        }
    }

    /// Convert from a `BlockwisePenalty`, preserving the structure hint and op.
    pub fn from_blockwise(bp: BlockwisePenalty) -> Self {
        PenaltySpec::Block {
            local: bp.local,
            col_range: bp.col_range,
            prior_mean: bp.prior_mean,
            structure_hint: bp.structure_hint,
            op: bp.op,
        }
    }

    pub fn from_blockwise_ref(bp: &BlockwisePenalty) -> Self {
        PenaltySpec::Block {
            local: bp.local.clone(),
            col_range: bp.col_range.clone(),
            prior_mean: bp.prior_mean.clone(),
            structure_hint: bp.structure_hint.clone(),
            op: bp.op.clone(),
        }
    }

    /// Materialize the full `p x p` dense penalty matrix.
    /// For `Dense`, this is a clone.  For `Block`, this embeds `local` into a
    /// zero matrix at the given `col_range`.
    pub fn to_dense(&self) -> Array2<f64> {
        match self {
            PenaltySpec::Dense(m) => m.clone(),
            PenaltySpec::DenseWithMean { matrix, .. } => matrix.clone(),
            PenaltySpec::Block {
                local, col_range, ..
            } => {
                let p = col_range.end.max(local.nrows());
                // Caller should supply p externally when the total dim is larger;
                // this is the best we can do without it.
                let mut out = Array2::zeros((p, p));
                out.slice_mut(s![col_range.clone(), col_range.clone()])
                    .assign(local);
                out
            }
        }
    }

    /// Materialize the full `p_total x p_total` dense penalty matrix.
    /// For `Dense`, this is a clone (asserts that it matches `p_total`).
    /// For `Block`, this embeds `local` into a `p_total x p_total` zero matrix.
    pub fn to_global(&self, p_total: usize) -> Array2<f64> {
        match self {
            PenaltySpec::Dense(m) => {
                assert_eq!(m.nrows(), p_total);
                m.clone()
            }
            PenaltySpec::DenseWithMean { matrix, .. } => {
                assert_eq!(matrix.nrows(), p_total);
                matrix.clone()
            }
            PenaltySpec::Block {
                local, col_range, ..
            } => {
                let mut out = Array2::zeros((p_total, p_total));
                out.slice_mut(s![col_range.clone(), col_range.clone()])
                    .assign(local);
                out
            }
        }
    }
}

/// Pure shape validation for a [`PenaltySpec`] against a coefficient dimension.
///
/// Neutral term-side check (no solver state): it only inspects block/dense
/// dimensions and returns [`EstimationError`]. Moved byte-identically from the
/// solver's `estimate::external_options` during the `#1521` carve so it can be
/// single-sourced alongside the type it validates; the solver re-exports it via
/// `gam_terms::validate_penalty_spec_shape`.
pub fn validate_penalty_spec_shape(
    idx: usize,
    spec: &PenaltySpec,
    p: usize,
    context: &str,
) -> Result<(), EstimationError> {
    match spec {
        PenaltySpec::Block {
            local, col_range, ..
        } => {
            let bd = col_range.len();
            if local.nrows() != bd || local.ncols() != bd {
                crate::bail_invalid_estim!(
                    "{context}: block penalty {idx} local matrix must be {bd}x{bd}, got {}x{}",
                    local.nrows(),
                    local.ncols()
                );
            }
            if col_range.end > p {
                crate::bail_invalid_estim!(
                    "{context}: block penalty {idx} col_range {}..{} exceeds p={p}",
                    col_range.start,
                    col_range.end
                );
            }
        }
        PenaltySpec::Dense(m) => {
            if m.nrows() != p || m.ncols() != p {
                crate::bail_invalid_estim!(
                    "{context}: dense penalty {idx} must be {p}x{p}, got {}x{}",
                    m.nrows(),
                    m.ncols()
                );
            }
        }
        PenaltySpec::DenseWithMean { matrix, .. } => {
            if matrix.nrows() != p || matrix.ncols() != p {
                crate::bail_invalid_estim!(
                    "{context}: dense penalty {idx} must be {p}x{p}, got {}x{}",
                    matrix.nrows(),
                    matrix.ncols()
                );
            }
        }
    }
    Ok(())
}
