pub mod analytic_penalties;
pub mod atom_codes;
pub mod atom_selection;
pub mod basis;
pub mod closed_form_operator;
pub mod construction;
pub mod hull;
pub mod input_loc_derivatives;
pub mod latent_coord;
pub mod layout;
pub mod penalty_op;
pub mod sae_manifold;
pub mod smooth;
pub mod term_builder;

pub use analytic_penalties::{
    ARDPenalty, AnalyticPenalty, AnalyticPenaltyKind, AnalyticPenaltyOp,
    AnalyticPenaltyRegistry, DifferenceOpKind, FrozenAnalyticPenaltyOp, IBPAssignmentPenalty,
    IsometryPenalty, IsometryDuchonRadialSource, IsometryReference, OrthogonalityPenalty,
    PenaltyTier, PsiSlice, SoftmaxAssignmentSparsityPenalty, SparsityKind, SparsityPenalty,
    TotalVariationPenalty, WeightField,
};
pub use atom_selection::{
    AssignmentSparsityCoupling, AtomLibrary, AtomRecord, AtomSelectionStrategy, EntropicSoftmax,
    L1Relaxed, ShapeRef, TopK,
};
pub use latent_coord::{
    AuxPriorFamily, AuxPriorStrength, InputLocationDerivative, LatentCoordValues, LatentIdMode,
    LatentManifold,
};
pub use sae_manifold::{
    AssignmentMode, GumbelTemperatureSchedule, SaeAssignment, SaeAtomBasisKind, SaeManifoldAtom,
    SaeManifoldLoss, SaeManifoldRho, SaeManifoldTerm, ScheduleKind,
};
