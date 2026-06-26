//! Exact-Newton joint-ψ term carriers and the joint-ψ workspace trait.
//!
//! These ψ-hyperparameter term types and the [`ExactNewtonJointPsiWorkspace`]
//! trait are neutral carriers in the criterion contract: they reference only
//! [`HyperOperator`] / [`DriftDerivResult`] (defined in this crate) and ndarray
//! arrays, so they live in `gam-problem` and are re-exported by the
//! `custom_family` layer for backward compatibility.

use crate::{DriftDerivResult, HyperOperator};
use ndarray::{Array1, Array2};
use std::sync::Arc;

#[derive(Clone)]
pub struct ExactNewtonJointPsiTerms {
    pub objective_psi: f64,
    pub score_psi: Array1<f64>,
    pub hessian_psi: Array2<f64>,
    pub hessian_psi_operator: Option<Arc<dyn HyperOperator>>,
}

impl std::fmt::Debug for ExactNewtonJointPsiTerms {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("ExactNewtonJointPsiTerms")
            .field("objective_psi", &self.objective_psi)
            .field("score_psi", &self.score_psi)
            .field("hessian_psi", &self.hessian_psi)
            .field(
                "hessian_psi_operator",
                &self.hessian_psi_operator.as_ref().map(|_| "<operator>"),
            )
            .finish()
    }
}

impl ExactNewtonJointPsiTerms {
    pub fn zeros(total: usize) -> Self {
        Self {
            objective_psi: 0.0,
            score_psi: Array1::zeros(total),
            hessian_psi: Array2::zeros((total, total)),
            hessian_psi_operator: None,
        }
    }
}

pub struct ExactNewtonJointPsiSecondOrderTerms {
    pub objective_psi_psi: f64,
    pub score_psi_psi: Array1<f64>,
    pub hessian_psi_psi: Array2<f64>,
    pub hessian_psi_psi_operator: Option<Box<dyn HyperOperator>>,
}

/// Direction-contracted second-order ψ terms for the profiled θ-HVP (#740).
///
/// The per-pair [`ExactNewtonJointPsiSecondOrderTerms`] are the `(ψ_i, ψ_j)`
/// entries of the joint hyper-Hessian; assembling the full outer Hessian from
/// them costs one O(n) family row pass per pair, i.e. `K²·n`. A matrix-free
/// profiled θ-HVP never needs the individual pairs — it needs, for one applied
/// outer direction with ψ-weights `α_ψ`, the `α`-contraction of those pairs
/// against the combined ψ-direction `ψ(α) = Σ_j α_j ψ_j`:
///
/// ```text
///   objective[i] = Σ_j α_j V_{ψ_i ψ_j}
///   score[i]     = Σ_j α_j g_{ψ_i ψ_j}          (a p-vector per output row i)
///   hessian[i]   = Σ_j α_j D²_β H_L[ψ_i, ψ_j]
///                = D²_β H_L[ψ_i, ψ(α)]            (bilinearity)
/// ```
///
/// All `psi_dim` output rows share the SAME contracted second leg `ψ(α)`, so a
/// family that streams its rows once over `ψ(α)` (carrying every fixed first
/// leg `ψ_i` as a batched factor column) produces every row in a SINGLE n-pass.
/// That is the cost the profiled θ-HVP turns into `K·n`-to-densify /
/// `m·n`-in-CG instead of the dense path's `K²·n`.
///
/// Indexing is over the flattened ψ coordinates in the same order as
/// [`ExactNewtonJointPsiWorkspace::second_order_terms`]; `hessian[i]` carries
/// the `D²_β H_L[ψ_i, ψ(α)]` drift as a [`DriftDerivResult`] (dense or
/// operator-backed) plus any block-local `S_{ψ_i ψ_j}` penalty motion folded by
/// the family, exactly mirroring the per-pair `hessian_psi_psi(_operator)`.
pub struct ExactNewtonJointPsiSecondOrderContracted {
    /// `objective[i] = Σ_j α_j V_{ψ_i ψ_j}`, one scalar per ψ output row.
    pub objective: Array1<f64>,
    /// `score[i] = Σ_j α_j g_{ψ_i ψ_j}`, the `psi_dim × total` matrix whose
    /// row `i` is the contracted fixed-β score derivative for output row `i`.
    pub score: Array2<f64>,
    /// `hessian[i] = D²_β H_L[ψ_i, ψ(α)]` for each ψ output row `i`.
    pub hessian: Vec<DriftDerivResult>,
}

pub trait ExactNewtonJointPsiWorkspace: Send + Sync {
    fn first_order_terms(&self, _: usize) -> Result<Option<ExactNewtonJointPsiTerms>, String> {
        // Default implementation ignores this parameter.
        Ok(None)
    }

    fn first_order_terms_all(&self) -> Result<Option<Vec<ExactNewtonJointPsiTerms>>, String> {
        Ok(None)
    }

    fn second_order_terms(
        &self,
        psi_i: usize,
        psi_j: usize,
    ) -> Result<Option<ExactNewtonJointPsiSecondOrderTerms>, String>;

    /// Direction-contracted second-order ψ terms for the profiled θ-HVP (#740).
    ///
    /// Given the ψ-block weights `alpha_psi` (length `psi_dim`, the ψ slice of
    /// one applied outer direction α), return the `α`-contraction of every
    /// `(ψ_i, ψ_j)` second-order term against the combined ψ-direction
    /// `ψ(α) = Σ_j alpha_psi[j] · ψ_j`, as
    /// [`ExactNewtonJointPsiSecondOrderContracted`]. A family that can stream
    /// its rows once over `ψ(α)` overrides this so the profiled outer-Hessian
    /// operator applies one combined-direction n-pass per matvec instead of the
    /// dense path's `K²` per-pair [`Self::second_order_terms`] passes.
    ///
    /// Default returns `None`: the profiled θ-HVP operator is then not built and
    /// the evaluator keeps the exact per-pair assembly (dense
    /// `compute_outer_hessian` / `build_outer_hessian_operator`). Overriding
    /// this method is purely a representation/cost choice — it must produce the
    /// exact same contraction the per-pair terms would, which the
    /// `profiled_theta_hvp_outer_hessian_fd` finite-difference cross-check
    /// guards.
    fn second_order_terms_contracted(
        &self,
        _: &[f64],
    ) -> Result<Option<ExactNewtonJointPsiSecondOrderContracted>, String> {
        // Default implementation ignores this parameter.
        Ok(None)
    }

    fn hessian_directional_derivative(
        &self,
        psi_index: usize,
        d_beta_flat: &Array1<f64>,
    ) -> Result<Option<DriftDerivResult>, String>;
}
