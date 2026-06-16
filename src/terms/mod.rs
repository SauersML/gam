pub mod analytic_penalties;
pub mod basis;
pub(crate) mod coefficient_group_resolver;
pub mod construction;
pub mod decoders;
pub mod dictionary;
pub mod geometry;
pub mod latent;
pub mod penalties;
pub mod sae;
pub mod smooth;
pub mod smooth_overrides;
pub mod structure;
pub mod term_builder;
pub mod torch_dispatch;

pub use analytic_penalties::{
    ARDPenalty, AnalyticPenalty, AnalyticPenaltyKind, AnalyticPenaltyOp, AnalyticPenaltyRegistry,
    BlockOrthogonalityPenalty, BlockSparsityPenalty, DecoderIncoherencePenalty, DifferenceOpKind,
    FrozenAnalyticPenaltyOp, IBPAssignmentPenalty, IsometryDuchonRadialSource, IsometryPenalty,
    IsometryReference, IvaeRidgeMeanGauge, JumpReLUPenalty, MechanismSparsityPenalty,
    NestedPrefixPenalty, NuclearNormPenalty, OrthogonalityPenalty, ShapeMonotonicityPenalty,
    ParametricRowPrecisionPriorPenalty, PenaltyConcavity, PenaltyTier, PsiSlice,
    RowPrecisionPriorPenalty, ScadMcpPenalty, ScalarWeightSchedule,
    SoftmaxAssignmentSparsityPenalty, SparsityKind, SparsityPenalty, TopKActivationPenalty,
    TotalVariationPenalty, WeightField,
};
pub use sae::atom_selection::{
    AssignmentSparsityCoupling, AtomLibrary, AtomRecord, AtomSelectionStrategy, EntropicSoftmax,
    L1Relaxed, ShapeRef, TopK,
};
pub use decoders::gated_decoder::GatedSAEDecoder;
pub use decoders::interchange_decoder::{
    InterchangeDecodeBackward, InterchangeDecodeForward, InterchangeSwapBackward,
    InterchangeSwapForward, interchange_decode_backward, interchange_decode_forward,
    interchange_swap_backward, interchange_swap_forward,
};
pub use latent::{
    AuxPriorFamily, AuxPriorStrength, InputLocationDerivative, LatentCoordValues, LatentIdMode,
    LatentManifold,
};
pub use dictionary::linear::{
    LinearDictionaryAssignment, LinearDictionaryConfig, LinearDictionaryFit, fit_linear_dictionary,
};
pub use basis::matern_gradient::{MaternBasisGradientTarget, StreamingMaternBasisGradientEvaluator};
pub use sae::criterion_atoms::{SaeCriterion, SaeCriterionAtom};
pub use sae::encode_atlas::{
    AtlasConfig, AtomEncodeAtlas, BasisHessianLipschitz, CertifiedChart, ChartRegion, EncodeAtlas,
    EncodeResult, KANTOROVICH_THRESHOLD, RowCertificate, row_certificate,
};
pub use sae::manifold::{
    AssignmentMode, CertificateInputs, CurvatureBifurcation, CurvatureWalkReport,
    GumbelTemperatureSchedule, PeriodicHarmonicEvaluator, SaeAssignment, SaeAtomBasisKind,
    SaeBasisEvaluator, SaeManifoldAtom, SaeManifoldLoss, SaeManifoldOuterObjective, SaeManifoldRho,
    SaeManifoldTerm, SaeOuterRhoGradientComponents, ScheduleKind, SphereChartEvaluator,
    TorusHarmonicEvaluator, dictionary_incoherence_report,
    dictionary_incoherence_report_with_dispersion,
};
pub use sae::optimality_certificate::{
    CriterionCertificate, DirectionalSamples, certificate_from_samples,
    deterministic_probe_direction, probe_step,
};
pub use sae::row_jet_program::{AtomRowBasisJet, RowGate, SaeReconstructionRowProgram};
pub use penalties::sheaf::{EdgeRestriction, SheafConsistencyPenalty};
