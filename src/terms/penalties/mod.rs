//! Analytic penalty registry manifests.
//!
//! Add a primitive by adding one module here, implementing
//! [`PenaltyManifest`] for its concrete penalty type, and registering it in
//! [`analytic_penalty_registry`].

use crate::terms::analytic_penalties::{AnalyticPenalty, PenaltyTier};

pub mod ard;
pub mod block_orthogonality;
pub mod block_sparsity;
pub mod ibp_assignment;
pub mod isometry;
pub mod ivae_ridge_mean_gauge;
pub mod jump_relu;
pub mod mechanism_sparsity;
pub mod nuclear_norm;
pub mod orthogonality;
pub mod parametric_row_precision_prior;
pub mod row_precision_prior;
pub mod scad_mcp;
pub mod softmax_assignment_sparsity;
pub mod sparsity;
pub mod topk_activation;
pub mod total_variation;

pub trait PenaltyManifest: AnalyticPenalty {
    const KIND_TAG: &'static str;
    const PYTHON_WRAPPER: &'static str;
    const ROW_BLOCK_DIAGONAL: bool;

    fn dispatch_tier(&self) -> PenaltyTier {
        self.tier()
    }
}

#[macro_export]
macro_rules! analytic_penalty_registry {
    ($macro:ident) => {
        $macro! {
            register!(Isometry, IsometryPenalty);
            register!(Sparsity, SparsityPenalty);
            register!(SoftmaxAssignmentSparsity, SoftmaxAssignmentSparsityPenalty);
            register!(IBPAssignment, IBPAssignmentPenalty);
            register!(Ard, ARDPenalty);
            register!(TopKActivation, TopKActivationPenalty);
            register!(JumpReLU, JumpReLUPenalty);
            register!(TotalVariation, TotalVariationPenalty);
            register!(NuclearNorm, NuclearNormPenalty);
            register!(BlockSparsity, BlockSparsityPenalty);
            register!(MechanismSparsity, MechanismSparsityPenalty);
            register!(RowPrecisionPrior, RowPrecisionPriorPenalty);
            register!(IvaeRidgeMeanGauge, IvaeRidgeMeanGauge);
            register!(ParametricRowPrecisionPrior, ParametricRowPrecisionPriorPenalty);
            register!(ScadMcp, ScadMcpPenalty);
            register!(BlockOrthogonality, BlockOrthogonalityPenalty);
            register!(Orthogonality, OrthogonalityPenalty);
        }
    };
}
