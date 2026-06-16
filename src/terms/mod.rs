pub mod analytic_penalties;
pub mod anova_atom;
pub mod atom_codes;
pub mod atom_selection;
pub mod basis;
pub mod behavioral_head;
pub mod closed_form_operator;
pub(crate) mod coefficient_group_resolver;
pub mod construction;
pub mod decoders;
pub mod equivariant_penalty;
pub mod gated_decoder;
pub mod hull;
pub mod input_loc_derivatives;
pub mod latent_coord;
pub mod linear_dictionary;
pub mod penalty_op;
pub mod sae;
pub mod sae_corpus;
pub mod sae_encode_atlas;
pub mod sae_manifold;
pub mod sheaf;
pub mod smooth;
pub mod smooth_overrides;
pub mod sphere_gpu;
pub mod term_builder;
pub mod torch_dispatch;

pub use analytic_penalties::{
    ARDPenalty, AnalyticPenalty, AnalyticPenaltyKind, AnalyticPenaltyOp, AnalyticPenaltyRegistry,
    BlockOrthogonalityPenalty, BlockSparsityPenalty, DecoderIncoherencePenalty, DifferenceOpKind,
    FrozenAnalyticPenaltyOp, IBPAssignmentPenalty, IsometryDuchonRadialSource, IsometryPenalty,
    IsometryReference, IvaeRidgeMeanGauge, JumpReLUPenalty, MechanismSparsityPenalty,
    MonotonicityPenalty, NestedPrefixPenalty, NuclearNormPenalty, OrthogonalityPenalty,
    ParametricRowPrecisionPriorPenalty, PenaltyConcavity, PenaltyTier, PsiSlice,
    RowPrecisionPriorPenalty, ScadMcpPenalty, ScalarWeightSchedule,
    SoftmaxAssignmentSparsityPenalty, SparsityKind, SparsityPenalty, TopKActivationPenalty,
    TotalVariationPenalty, WeightField,
};
pub use atom_selection::{
    AssignmentSparsityCoupling, AtomLibrary, AtomRecord, AtomSelectionStrategy, EntropicSoftmax,
    L1Relaxed, ShapeRef, TopK,
};
pub use gated_decoder::GatedSAEDecoder;
pub use decoders::interchange_decoder::{
    InterchangeDecodeBackward, InterchangeDecodeForward, InterchangeSwapBackward,
    InterchangeSwapForward, interchange_decode_backward, interchange_decode_forward,
    interchange_swap_backward, interchange_swap_forward,
};
pub use latent_coord::{
    AuxPriorFamily, AuxPriorStrength, InputLocationDerivative, LatentCoordValues, LatentIdMode,
    LatentManifold,
};
pub use linear_dictionary::{
    LinearDictionaryAssignment, LinearDictionaryConfig, LinearDictionaryFit, fit_linear_dictionary,
};
pub use basis::matern_gradient::{MaternBasisGradientTarget, StreamingMaternBasisGradientEvaluator};
pub use sae::criterion_atoms::{SaeCriterion, SaeCriterionAtom};
pub use sae_encode_atlas::{
    AtlasConfig, AtomEncodeAtlas, BasisHessianLipschitz, CertifiedChart, ChartRegion, EncodeAtlas,
    EncodeResult, KANTOROVICH_THRESHOLD, RowCertificate, row_certificate,
};
pub use sae_manifold::{
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
pub use sheaf::{EdgeRestriction, SheafConsistencyPenalty};
