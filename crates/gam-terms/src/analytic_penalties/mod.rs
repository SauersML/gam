//! Analytic penalty primitives for the three-tier (beta / ext-coord / rho) engine.
//!
//! This module implements the structured
//! penalties identified as the minimal identifiability tools needed by an
//! SAE / principal-manifold / latent-coordinate workflow:
//!
//!   * [`IsometryPenalty`] — pulls the decoder pullback metric
//!     `J(t)^T W J(t)` toward a reference metric on the latent manifold. Lives
//!     on the extension-coordinate tier (specifically on a
//!     [`crate::latent::LatentCoordValues`] slice). Breaks the
//!     diffeomorphism gauge so the inner Hessian on `t` is full-rank and the
//!     IFT is well-defined.
//!   * [`SparsityPenalty`] — smoothed L¹ (`sqrt(x² + ε²)`), Hoyer, or Log
//!     sparsifier. Applied to a `β` slice (SAE codes) or extension-coordinate
//!     slice (soft atom
//!     amplitudes). Differentiable everywhere; the smoothing parameter `ε` may
//!     itself live in `ρ` so REML shrinks it.
//!   * [`IBPAssignmentPenalty`] — deterministic continuous-relaxation
//!     Beta-Bernoulli/IBP prior over per-row SAE-manifold active sets.
//!   * [`ARDPenalty`] — one penalty parameter per latent axis. The marginal
//!     likelihood's Occam factor sends unused axes' precision to infinity,
//!     discovering intrinsic dimension only after a separate gauge fix
//!     (`AuxPrior` or `Isometry`) pins rotations / reparameterisations.
//!   * [`TotalVariationPenalty`] — smoothed L¹ on first differences of a
//!     latent coefficient block. This is coordinatewise/anisotropic TV: each
//!     latent axis is penalized independently on every edge. Promotes
//!     piecewise-constant atom maps.
//!   * [`NuclearNormPenalty`] — smoothed L¹ on singular values of a matrix
//!     latent block, `Σ_i (sqrt(σ_i² + ε²) - ε)`. Promotes low intrinsic rank
//!     without choosing a canonical axis basis; in SAE wiring this is the
//!     decoder-embedding rank-selection lever.
//!   * [`BlockSparsityPenalty`] — group-lasso smoothed L¹ over predefined
//!     latent-axis blocks. Unlike per-element L¹ or per-axis L² ARD, it
//!     shrinks whole semantic groups together; pair with
//!     `LatentIdMode::AuxPriorDimSelection` when aux classes define the active
//!     group subset.
//!   * [`RowPrecisionPriorPenalty`] — zero-mean Gaussian row-precision
//!     prior on latent rows. This fixed-precomputed variant accepts one
//!     precision matrix per row. It is not an iVAE conditional-mean gauge;
//!     use `LatentIdMode::AuxPrior` for the ridge/linear projection residual.
//!   * [`IvaeRidgeMeanGauge`] — iVAE-style conditional-mean gauge fixing:
//!     penalizes the component of the latent field not explained by auxiliary
//!     covariates via the ridge projection `U(UᵀU + εI)⁻¹Uᵀ`.
//!   * [`ParametricRowPrecisionPriorPenalty`] — zero-mean Gaussian
//!     row-precision prior with a learnable distance-kernel map from auxiliary
//!     rows to diagonal per-row precision. It changes shrinkage strength, not
//!     the conditional mean.
//!   * [`OrthogonalityPenalty`] — fixes the rotation gauge inside a latent
//!     block by penalizing cross-axis correlations. Pair with ARD when
//!     intrinsic dimension should be identifiable.
//!   * [`BlockOrthogonalityPenalty`] — penalizes only between-block
//!     cross-products of latent axes, leaving within-block structure free.
//!   * [`ScadMcpPenalty`] — elementwise nonconvex SCAD/MCP sparsity on
//!     extension-coordinate latent blocks. Tapers the shrinkage derivative to
//!     zero beyond the SCAD/MCP cutoff so large coefficients are not L¹-biased.
//!   * [`DecoderIncoherencePenalty`] — β-tier SAE decoder penalty
//!     `½·w·Σ_{j<k} W[j,k]·‖B_j B_k^T‖²_F` for stored decoder blocks
//!     `B_k ∈ R^{M_k×p_out}`, with `W[j,k]` coming from co-activation.
//!     Pushes co-firing atom decoder column spaces apart.
//!
//! All shipped primitives are **analytic**: no autograd, no finite differencing. Each
//! exposes:
//!
//!   * `value(target, rho) -> f64`
//!   * `grad_target(target, rho) -> Array1<f64>`
//!   * `hessian_diag(target, rho) -> Array1<f64>` (when block-diagonal) or
//!     `hvp(target, rho, v) -> Array1<f64>` (when not)
//!   * `grad_rho(target, rho) -> Array1<f64>` (one entry per ρ-axis owned)
//!
//! The signatures are deliberately uniform with the existing smoothness path:
//! the quadratic ARD penalty produces a [`crate::smooth::BlockwisePenalty`]
//! that slots directly into the canonical-penalty pipeline, while the
//! non-quadratic Sparsity, TV, NuclearNorm, SCAD/MCP, Orthogonality,
//! DecoderIncoherence, and Isometry
//! penalties produce [`AnalyticPenaltyOp`] handles that downstream PIRLS / REML consumers query
//! through the same `value / gradient / hvp` interface they already use for
//! smoothness.
//!
//! ## Registration with REML
//!
//! Each penalty owns a (possibly empty) sub-range of the global `ρ` vector.
//! See [`AnalyticPenaltyKind::rho_count`]. The outer REML loop concatenates
//! these onto the existing per-smooth `ρ`s, exactly the way anisotropic
//! kernel-shape paths append ext-coords. The IsometryPenalty owns one `ρ`; the
//! SparsityPenalty owns either zero (`ε` fixed) or one (`ε` REML-selected) plus
//! one strength; the ARDPenalty owns `d` (one per latent axis);
//! NuclearNorm, BlockSparsity, BlockOrthogonality, ScadMcp,
//! DecoderIncoherence, RowPrecisionPrior, and Orthogonality each own one
//! strength only when their weight is learnable.
//! IvaeRidgeMeanGauge owns one strength only when its weight is learnable.
//! ParametricRowPrecisionPrior owns its log-baseline precision, raw distance
//! sensitivity, and reference point coordinates, plus one strength axis when
//! requested.
//!
//! ## Three-tier landings
//!
//! | Penalty   | Target tier | ρ-axes owned         |
//! |-----------|-------------|----------------------|
//! | Isometry  | ext-coord (latent t) | 1 (log μ_iso)        |
//! | Sparsity  | β or ext-coord       | 1 (strength) [+1 ε]  |
//! | IBP       | ext-coord (logits)   | 0 or 1 (log α)       |
//! | ARD       | ext-coord (latent t) | d (one per axis)     |
//! | TV        | ext-coord (latent t) | 0 or 1 (log μ_tv)    |
//! | NuclearNorm | ext-coord (latent t) | 0 or 1 (log μ_nuc)  |
//! | BlockSparsity | ext-coord (latent t) | 0 or 1 (log μ_group) |
//! | MechanismSparsity | β (decoder W) | 0 or 1 (log μ_mech) |
//! | ScadMcp | ext-coord (latent t) | 0 or 1 (log μ_scad_mcp) |
//! | DecoderIncoherence | β (SAE decoder blocks) | 0 or 1 (log μ_decoder_incoh) |
//! | RowPrecisionPrior | ext-coord (latent t) | 0 or 1 (log μ_aux) |
//! | IvaeRidgeMeanGauge | ext-coord (latent t) | 0 or 1 (log μ_ivae_mean) |
//! | ParametricRowPrecisionPrior | ext-coord (latent t) | d + d + d·du [+1 log μ_aux] |
//! | Orthogonality | ext-coord (latent t) | 0 or 1 (log μ_orth) |
//! | BlockOrthogonality | ext-coord (latent t) | 0 or 1 (log μ_block_orth) |

// Re-exported so every concern submodule can pull the shared external imports
// through `use super::*;` without re-listing them.
pub(crate) use faer::Side;
pub(crate) use ndarray::{
    Array1, Array2, Array3, ArrayView1, ArrayView2, ArrayViewMut1, CowArray, Ix2, Ix3,
};
pub(crate) use std::sync::{Arc, RwLock};

pub use self::manifest::PenaltyManifest;
pub use self::op::{PenaltyOp, ScaledPenaltyOp};
pub use self::sheaf::{EdgeRestriction, SheafConsistencyPenalty};
pub(crate) use gam_linalg::faer_ndarray::{FaerEigh, FaerSvd};
pub(crate) use gam_linalg::lanczos::{
    SymmetricLanczosOptions, symmetric_lanczos_eigenpairs, symmetric_lanczos_log_quadrature,
};
pub(crate) use crate::basis::{
    BasisError, DuchonNullspaceOrder, radial_basis_cartesian_derivative,
};
pub(crate) use crate::terms::sae::manifold::{GumbelTemperatureSchedule, ScheduleKind};
pub(crate) use crate::smooth::BlockwisePenalty;

#[macro_use]
mod penalty_trait;

mod ard;
mod block_sparsity;
pub mod equivariant_penalty;
mod ibp;
mod isometry;
mod manifest;
mod nested_prefix;
mod nuclear_norm;
mod op;
mod orthogonality;
mod registry;
mod row_precision;
mod scad_mcp;
mod sheaf;
mod sparsity;
mod total_variation;

pub use ard::*;
pub use block_sparsity::*;
pub use ibp::*;
pub use isometry::*;
pub use nested_prefix::*;
pub use nuclear_norm::*;
pub use orthogonality::*;
pub use penalty_trait::*;
pub use registry::*;
pub use row_precision::*;
pub use scad_mcp::*;
pub use sparsity::*;
pub use total_variation::*;

pub(crate) fn flatten_matrix(m: &Array2<f64>) -> Array1<f64> {
    let n_obs = m.nrows();
    let d = m.ncols();
    let mut out = Array1::<f64>::zeros(n_obs * d);
    for n in 0..n_obs {
        for a in 0..d {
            out[n * d + a] = m[[n, a]];
        }
    }
    out
}

#[cfg(test)]
mod tests;
