#[macro_export]
macro_rules! bail_invalid_basis {
    ($fmt:literal $(, $($arg:tt)*)?) => {
        return Err($crate::basis::BasisError::InvalidInput(format!($fmt $(, $($arg)*)?)))
    };
    ($msg:expr $(,)?) => {
        return Err($crate::basis::BasisError::InvalidInput($msg))
    };
}

#[macro_export]
macro_rules! bail_dim_basis {
    ($fmt:literal $(, $($arg:tt)*)?) => {
        return Err($crate::basis::BasisError::DimensionMismatch(format!($fmt $(, $($arg)*)?)))
    };
    ($msg:expr $(,)?) => {
        return Err($crate::basis::BasisError::DimensionMismatch($msg))
    };
}

#[macro_export]
macro_rules! bail_invalid_estim {
    ($fmt:literal $(, $($arg:tt)*)?) => {
        return Err($crate::EstimationError::InvalidInput(format!($fmt $(, $($arg)*)?)))
    };
    ($msg:expr $(,)?) => {
        return Err($crate::EstimationError::InvalidInput($msg))
    };
}

#[macro_export]
macro_rules! gpu_bail {
    ($($arg:tt)*) => {
        return ::std::result::Result::Err(gam_gpu::gpu_error::GpuError::DriverCallFailed {
            reason: ::std::format!($($arg)*),
        })
    };
}

pub mod analytic_penalties;
pub mod basis;
pub mod chunked_kernel_design;
pub mod construction;
pub mod decoders;
pub mod dictionary;
pub mod geometry;
pub mod inference;
pub mod kronecker;
pub mod latent;
pub mod penalty_spec;
pub mod smooth;
pub mod smooth_overrides;
pub mod structure;
pub mod term_builder;

pub mod terms {
    pub use crate::*;
}

/// Re-export of the neutral estimation error so crate-local macros
/// (`bail_invalid_estim!`) and call sites can reference `crate::EstimationError`.
pub use gam_problem::EstimationError;
pub use penalty_spec::{PenaltySpec, validate_penalty_spec_shape};

pub use analytic_penalties::{
    ARDPenalty, AnalyticPenalty, AnalyticPenaltyKind, AnalyticPenaltyOp, AnalyticPenaltyRegistry,
    BlockOrthogonalityPenalty, BlockSparsityPenalty, DecoderIncoherencePenalty, DifferenceOpKind,
    EdgeRestriction, FrozenAnalyticPenaltyOp, IBPAssignmentPenalty, IbpHessianDiagThirdChannels,
    IsometryDuchonRadialSource, IsometryPenalty, IsometryReference, IvaeRidgeMeanGauge,
    JumpReLUPenalty, MechanismSparsityPenalty, NestedPrefixPenalty, NuclearNormPenalty,
    OrthogonalityPenalty, ParametricRowPrecisionPriorPenalty, PenaltyConcavity, PenaltyTier,
    PsiSlice, RowPrecisionPriorPenalty, ScadMcpPenalty, ScalarWeightSchedule,
    ShapeMonotonicityPenalty, SheafConsistencyPenalty, SoftmaxAssignmentSparsityPenalty,
    SparsityKind, SparsityPenalty, TopKActivationPenalty, TotalVariationPenalty, WeightField,
};
