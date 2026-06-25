pub mod analytic_penalties;
pub mod basis;
pub mod construction;
pub mod decoders;
pub mod dictionary;
pub mod geometry;
pub mod latent;
pub mod sae;
pub mod smooth;
pub mod smooth_overrides;
pub mod structure;
pub mod term_builder;
#[path = "smooth/torch_dispatch.rs"]
pub mod torch_dispatch;

pub use analytic_penalties::{
    ARDPenalty, AnalyticPenalty, AnalyticPenaltyKind, AnalyticPenaltyOp, AnalyticPenaltyRegistry,
    BlockOrthogonalityPenalty, BlockSparsityPenalty, DecoderIncoherencePenalty, DifferenceOpKind,
    EdgeRestriction, FrozenAnalyticPenaltyOp, IBPAssignmentPenalty, IsometryDuchonRadialSource,
    IsometryPenalty, IsometryReference, IvaeRidgeMeanGauge, JumpReLUPenalty,
    MechanismSparsityPenalty, NestedPrefixPenalty, NuclearNormPenalty, OrthogonalityPenalty,
    ParametricRowPrecisionPriorPenalty, PenaltyConcavity, PenaltyTier, PsiSlice,
    RowPrecisionPriorPenalty, ScadMcpPenalty, ScalarWeightSchedule, ShapeMonotonicityPenalty,
    SheafConsistencyPenalty, SoftmaxAssignmentSparsityPenalty, SparsityKind, SparsityPenalty,
    TopKActivationPenalty, TotalVariationPenalty, WeightField,
};
pub use basis::matern_gradient::{
    MaternBasisGradientTarget, StreamingMaternBasisGradientEvaluator,
};
pub use decoders::gated_decoder::GatedSAEDecoder;
pub use decoders::interchange_decoder::{
    InterchangeDecodeBackward, InterchangeDecodeForward, InterchangeSwapBackward,
    InterchangeSwapForward, interchange_decode_backward, interchange_decode_forward,
    interchange_swap_backward, interchange_swap_forward,
};
pub use dictionary::{
    LinearDictionaryAssignment, LinearDictionaryConfig, LinearDictionaryFit, fit_linear_dictionary,
};
pub use geometry::PeeledHull;
pub use latent::{
    AuxPriorFamily, AuxPriorStrength, InputLocationDerivative, LatentCoordValues, LatentIdMode,
    LatentManifold,
};
pub use sae::atom_selection::{
    AssignmentSparsityCoupling, AtomLibrary, AtomRecord, AtomSelectionStrategy, EntropicSoftmax,
    L1Relaxed, ShapeRef, TopK,
};
pub use sae::certificates::{
    CriterionCertificate, DirectionalSamples, certificate_from_samples,
    deterministic_probe_direction, probe_step,
};
pub use sae::criterion_atoms::{SaeCriterion, SaeCriterionAtom};
pub use sae::encode::{
    AtlasConfig, AtomEncodeAtlas, BasisHessianLipschitz, CertifiedChart, ChartRegion, EncodeAtlas,
    EncodeResult, KANTOROVICH_THRESHOLD, RowCertificate, row_certificate,
};
pub use sae::manifold::{
    ArdSharing, AssignmentMode, CertificateInputs, CurvatureBifurcation, CurvatureWalkReport,
    GumbelTemperatureSchedule, PeriodicHarmonicEvaluator, SaeAssignment, SaeAtomBasisKind,
    SaeBasisEvaluator, SaeManifoldAtom, SaeManifoldLoss, SaeManifoldOuterObjective, SaeManifoldRho,
    SaeManifoldTerm, SaeOuterRhoGradientComponents, ScheduleKind, SphereChartEvaluator,
    TorusHarmonicEvaluator, dictionary_incoherence_report,
    dictionary_incoherence_report_with_dispersion,
};
pub use sae::row_jet_program::{AtomRowBasisJet, RowGate, SaeReconstructionRowProgram};
