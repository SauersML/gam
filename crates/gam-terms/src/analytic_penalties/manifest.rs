//! Analytic penalty registry manifests.
//!
//! Add a primitive by implementing [`PenaltyManifest`] for its concrete
//! penalty type here and registering it in [`analytic_penalty_registry`].
//!
//! Inlined into `gam-terms` during the #1521 crate carve. This module was the
//! last `analytic_penalties` submodule still bridged to the old monolith via a
//! `#[path = "../../../../src/terms/analytic_penalties/manifest.rs"]` shim; when
//! `83dd4411c` deleted the monolith `src/terms/` tree the shim's target vanished
//! and the crate stopped compiling (the trait + every `impl` + the
//! `analytic_penalty_registry!` macro consumed by `registry.rs` live here). The
//! content is unchanged from the monolith version except the explicit
//! `crate::terms::analytic_penalties::{…}` import list is replaced by the
//! crate-idiomatic `use super::*;` (every sibling submodule pulls the shared
//! penalty types this way — see the `mod.rs` re-export block).

use super::*;

pub trait PenaltyManifest: AnalyticPenalty {
    const KIND_TAG: &'static str;
    const PYTHON_WRAPPER: &'static str;
    const ROW_BLOCK_DIAGONAL: bool;

    fn dispatch_tier(&self) -> PenaltyTier {
        self.tier()
    }
}

impl PenaltyManifest for ARDPenalty {
    const KIND_TAG: &'static str = "ard";
    const PYTHON_WRAPPER: &'static str = "ARDPenalty";
    const ROW_BLOCK_DIAGONAL: bool = true;
}

impl PenaltyManifest for BlockOrthogonalityPenalty {
    const KIND_TAG: &'static str = "block_orthogonality";
    const PYTHON_WRAPPER: &'static str = "BlockOrthogonalityPenalty";
    const ROW_BLOCK_DIAGONAL: bool = false;
}

impl PenaltyManifest for BlockSparsityPenalty {
    const KIND_TAG: &'static str = "block_sparsity";
    const PYTHON_WRAPPER: &'static str = "BlockSparsityPenalty";
    const ROW_BLOCK_DIAGONAL: bool = false;
}

impl PenaltyManifest for DecoderIncoherencePenalty {
    const KIND_TAG: &'static str = "decoder_incoherence";
    const PYTHON_WRAPPER: &'static str = "DecoderIncoherencePenalty";
    const ROW_BLOCK_DIAGONAL: bool = false;
}

impl PenaltyManifest for IBPAssignmentPenalty {
    const KIND_TAG: &'static str = "ibp_assignment";
    const PYTHON_WRAPPER: &'static str = "IBPAssignmentPenalty";
    const ROW_BLOCK_DIAGONAL: bool = true;
}

impl PenaltyManifest for IsometryPenalty {
    const KIND_TAG: &'static str = "isometry";
    const PYTHON_WRAPPER: &'static str = "IsometryPenalty";
    const ROW_BLOCK_DIAGONAL: bool = false;
}

impl PenaltyManifest for IvaeRidgeMeanGauge {
    const KIND_TAG: &'static str = "ivae_ridge_mean_gauge";
    const PYTHON_WRAPPER: &'static str = "IvaeRidgeMeanGauge";
    const ROW_BLOCK_DIAGONAL: bool = false;
}

impl PenaltyManifest for JumpReLUPenalty {
    const KIND_TAG: &'static str = "jumprelu";
    const PYTHON_WRAPPER: &'static str = "JumpReLUPenalty";
    const ROW_BLOCK_DIAGONAL: bool = true;
}

impl PenaltyManifest for MechanismSparsityPenalty {
    const KIND_TAG: &'static str = "mechanism_sparsity";
    const PYTHON_WRAPPER: &'static str = "MechanismSparsityPenalty";
    const ROW_BLOCK_DIAGONAL: bool = false;
}

impl PenaltyManifest for ShapeMonotonicityPenalty {
    const KIND_TAG: &'static str = "monotonicity";
    const PYTHON_WRAPPER: &'static str = "MonotonicityPenalty";
    const ROW_BLOCK_DIAGONAL: bool = false;
}

impl PenaltyManifest for NestedPrefixPenalty {
    const KIND_TAG: &'static str = "nested_prefix";
    const PYTHON_WRAPPER: &'static str = "NestedPrefixPenalty";
    const ROW_BLOCK_DIAGONAL: bool = true;
}

impl PenaltyManifest for NuclearNormPenalty {
    const KIND_TAG: &'static str = "nuclear_norm";
    const PYTHON_WRAPPER: &'static str = "NuclearNormPenalty";
    const ROW_BLOCK_DIAGONAL: bool = false;
}

impl PenaltyManifest for OrthogonalityPenalty {
    const KIND_TAG: &'static str = "orthogonality";
    const PYTHON_WRAPPER: &'static str = "OrthogonalityPenalty";
    const ROW_BLOCK_DIAGONAL: bool = false;
}

impl PenaltyManifest for ParametricRowPrecisionPriorPenalty {
    const KIND_TAG: &'static str = "parametric_row_precision_prior";
    const PYTHON_WRAPPER: &'static str = "ParametricAuxConditionalPriorPenalty";
    const ROW_BLOCK_DIAGONAL: bool = true;
}

impl PenaltyManifest for RowPrecisionPriorPenalty {
    const KIND_TAG: &'static str = "row_precision_prior";
    const PYTHON_WRAPPER: &'static str = "AuxConditionalPriorPenalty";
    const ROW_BLOCK_DIAGONAL: bool = true;
}

impl PenaltyManifest for ScadMcpPenalty {
    const KIND_TAG: &'static str = "scad_mcp";
    const PYTHON_WRAPPER: &'static str = "ScadMcpPenalty";
    const ROW_BLOCK_DIAGONAL: bool = true;
}

impl PenaltyManifest for SheafConsistencyPenalty {
    const KIND_TAG: &'static str = "sheaf_consistency";
    const PYTHON_WRAPPER: &'static str = "SheafConsistencyPenalty";
    const ROW_BLOCK_DIAGONAL: bool = false;
}

impl PenaltyManifest for SoftmaxAssignmentSparsityPenalty {
    const KIND_TAG: &'static str = "softmax_assignment_sparsity";
    const PYTHON_WRAPPER: &'static str = "SoftmaxAssignmentSparsityPenalty";
    const ROW_BLOCK_DIAGONAL: bool = true;
}

impl PenaltyManifest for SparsityPenalty {
    const KIND_TAG: &'static str = "sparsity";
    const PYTHON_WRAPPER: &'static str = "SparsityPenalty";
    const ROW_BLOCK_DIAGONAL: bool = true;
}

impl PenaltyManifest for TopKActivationPenalty {
    const KIND_TAG: &'static str = "topk_activation";
    const PYTHON_WRAPPER: &'static str = "TopKActivationPenalty";
    const ROW_BLOCK_DIAGONAL: bool = true;
}

impl PenaltyManifest for TotalVariationPenalty {
    const KIND_TAG: &'static str = "total_variation";
    const PYTHON_WRAPPER: &'static str = "TotalVariationPenalty";
    const ROW_BLOCK_DIAGONAL: bool = false;
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
            register!(Monotonicity, ShapeMonotonicityPenalty);
            register!(NestedPrefix, NestedPrefixPenalty);
            register!(RowPrecisionPrior, RowPrecisionPriorPenalty);
            register!(IvaeRidgeMeanGauge, IvaeRidgeMeanGauge);
            register!(ParametricRowPrecisionPrior, ParametricRowPrecisionPriorPenalty);
            register!(ScadMcp, ScadMcpPenalty);
            register!(BlockOrthogonality, BlockOrthogonalityPenalty);
            register!(DecoderIncoherence, DecoderIncoherencePenalty);
            register!(Orthogonality, OrthogonalityPenalty);
            register!(SheafConsistency, SheafConsistencyPenalty);
        }
    };
}
