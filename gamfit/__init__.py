"""Formula-first generalized additive models with a high-performance Rust core.

Fit Gaussian, binomial, Poisson, and Gamma GLMs with smooth terms, random
effects, location-scale extensions, survival likelihoods, and learnable
links. Smoothing parameters are selected by REML or LAML; posterior
sampling uses NUTS. Geometric / manifold smooths (cyclic 1-D, cylinder
/ torus tensor, intrinsic sphere, boundary-conditioned B-splines) make
predictor spaces that wrap or close first-class.

The public surface also includes latent-coordinate and SAE-manifold tools:
analytic penalties such as ``ScadMcpPenalty`` and ``NuclearNormPenalty``;
assignment-family descriptors for softmax / ordered Beta--Bernoulli / top-k / smooth threshold
SAE gates; topology selection helpers; and manifold-SAE result objects with
per-row ``assignments`` plus per-atom decoder covariance / posterior shape
bands when produced by the Rust fit.

Quick start::

    import gamfit

    model = gamfit.fit(train, "y ~ s(x)")
    pred = model.predict(test, interval=0.95)
    posterior = model.sample(train)          # NUTS draws over coefficients
    print(model.summary())
    print(posterior)                         # one-line convergence summary
    model.save("model.gam")

For multi-smooth fits with per-smooth λ (the mgcv default), use the formula
API ``gamfit.fit(df, 'y ~ s(x1) + s(x2)')``.

See https://github.com/SauersML/gam for the full guide.
"""

from importlib import metadata as _metadata
from typing import TYPE_CHECKING as _TYPE_CHECKING

from ._api import (
    SUPPORT_SAE_SCHEMA,
    CtnStage1,
    SharedPrecisionGroup,
    bspline_basis,
    bspline_basis_derivative,
    build_info,
    conditional_prior_ivae,
    cross_fit_shared_precision_groups,
    cuda_subprocess_env,
    cuda_subprocess_library_dirs,
    cuda_diagnostics,
    derive_ivae_aux_scale,
    duchon_basis,
    duchon_function_norm_penalty,
    matern_basis,
    explain_error,
    fit,
    format_cuda_diagnostics,
    fit_array,
    gaussian_reml_fit,
    gaussian_reml_fit_backward,
    gaussian_reml_fit_batched,
    gaussian_reml_fit_batched_backward,
    gaussian_reml_fit_blocks_backward,
    gaussian_reml_fit_blocks_forward,
    gaussian_reml_fit_formula,
    gaussian_reml_fit_latent,
    gaussian_reml_fit_latent_backward,
    gaussian_reml_optimize_latent,
    glm_reml_fit_latent,
    glm_reml_fit_latent_backward,
    gaussian_reml_fit_positions,
    gaussian_reml_fit_positions_backward,
    gaussian_reml_fit_positions_batched,
    gaussian_reml_fit_positions_batched_backward,
    gaussian_reml_fit_with_constraints_backward,
    gaussian_reml_fit_with_constraints_forward,
    gaussian_weighted_ridge,
    gaussian_weighted_ridge_batch,
    load,
    loads,
    model_from_dict,
    save,
    mechanism_sparsity_jacobian,
    periodic_spline_curve_basis,
    smoothness_penalty,
    sphere_basis,
    sphere_basis_jet,
    validate_formula,
)
from ._binding import RustExtensionUnavailableError
from ._warnings import GamInferenceWarning, emit_inference_warnings
from ._rust import (  # native topology-census instruments
    adjudicate_atom_shape,
    shape_matched_control,
    shape_matched_control_f32,
)
from ._shape_census import (
    LabelShuffleMarginNull,
    ShapeControlledCensus,
    run_label_shuffle_margin_null,
    run_shape_controlled_census,
)
from ._compare import compare_models
from ._event_history import EventHistoryModel, fit_event_history
from ._linear_dictionary import LinearDictionaryFit, linear_dictionary_fit
from ._joint_events import JointEventModel, fit_joint_event_model, load_joint_event_model
from ._sparse_dictionary import (
    BlockSparseDictStream,
    BlockSparseDictionaryConvergence,
    BlockSparseDictionaryFit,
    BlockSparseStreamArtifact,
    BlockSparseStreamConvergence,
    SparseDictStream,
    SparseDictStreamArtifact,
    SparseDictionaryConvergence,
    SparseDictionaryFit,
    block_sparse_dictionary_fit,
    block_sparse_dictionary_fit_begin,
    fixed_budget_block_sparse_dictionary_fit,
    sparse_dictionary_fit,
    sparse_dictionary_fit_begin,
)
from ._sae_spectral import (
    AtlasNerveDiagram,
    AtomRetentionEvidence,
    BlockCoordinateReport,
    ChartInterpNullCalibration,
    ChartInterpNullCalibrationReport,
    ChartInterpNullProtocol,
    ChartInterpReadout,
    ChartInterpReport,
    ChartInterpStatisticValue,
    ComposedContract,
    ConditionalCoactivationInfluence,
    CoordinatePosterior,
    CouplingRobustnessCertificate,
    DoseResponseCalibrationReport,
    DualCertificateReport,
    FisherEffectEvidence,
    HolonomyReport,
    RoutabilityAudit,
    RoutabilityFloor,
    SpectrometerReport,
    SpikeRecovery,
    StageContainment,
    VarianceChargeEvidence,
    WholeSetContainment,
    atlas_nerve_diagram,
    audit_sae,
    block_firing_coordinates,
    chart_interp_score,
    compose_contracts,
    conditional_coactivation_influence,
    coordinate_posterior_from_precision,
    coupling_robustness_certificate,
    dimension_spectrometer,
    dose_response_calibration,
    effect_weighted_retention,
    loop_holonomy,
    recover_spikes,
    routability_audit,
    routability_floor,
    separation_limit,
    sparse_dict_dual_certificate,
    whole_set_containment,
)
from ._penalties import (
    ARDPenalty,
    AnalyticPenaltyKind,
    AuxConditionalPriorPenalty,
    BlockOrthogonalityPenalty,
    BlockSparsityPenalty,
    GatedSAEDecoder,
    OrderedBetaBernoulliPenalty,
    IsometryPenalty,
    IvaeRidgeMeanGauge,
    SmoothThresholdPenalty,
    MechanismSparsityPenalty,
    NuclearNormPenalty,
    OrthogonalityPenalty,
    ParametricAuxConditionalPriorPenalty,
    Penalty,
    PENALTY_MANIFEST,
    ScalarWeightSchedule,
    ScadMcpPenalty,
    SoftmaxAssignmentSparsityPenalty,
    SparsityPenalty,
    TopKActivationPenalty,
    TotalVariationPenalty,
)
from ._sheaf import SheafConsistencyPenalty
from .topology import (
    Circle,
    Cylinder,
    EuclideanPatch,
    Sphere as TopologySphere,
    Torus,
)
from ._select_topology import (
    BasisSpec,
    ScoreKind,
    ScoreScale,
    SelectTopologyResult,
    TopologyAutoSelector,
    TopologyAutoSelectorRank,
    TopologyAutoSelectorResult,
    TopologyCandidateFailure,
    TopologySelectionError,
    TopologyStack,
    select_topology,
    stack_topologies,
)
from ._diagnostics import Diagnostics
from . import diagnostics
from . import identifiability
from .identifiability import (
    IdentifiabilityReport,
    IdentifiabilityTheoremResult,
    IdentifiableFactorFitResult,
    check as identifiability_check,
    identifiable_factor_fit,
)
from ._equivariant import (
    GaugeCompanion,
    gauge_companion,
    rho_so2,
    rho_so2_jvp,
    rho_so3,
    rho_so3_jvp,
)
from .smooth import (
    BSpline,
    Categorical,
    Duchon,
    LatentCoord,
    Matern,
    MeasureJet,
    Pca,
    PeriodicSplineCurve,
    ShapeConstraintLiteral,
    Smooth as SmoothSpec,
    Sphere,
    TensorBSpline,
)
from ._protocol import BasisDescriptor, ManifoldDescriptor, PenaltyDescriptor
from . import manifolds  # noqa: F401  expose gamfit.manifolds.Circle, …
from . import kernels  # noqa: F401  expose gamfit.kernels.sinkhorn_barycenter, …
from ._basis_descriptors import PeriodicHarmonic
from ._composite_penalty import CompositePenalty
from ._smooth import (
    Smooth,
    SmoothSum,
)  # compositional Smooth(latent=..., basis=..., penalty=...)
from . import examples, topology
from .examples import (
    PartialSupervisionExample,
    PartialSupervisionFit,
    SaeSupervisedFit,
    partial_supervision,
    sae_supervised,
)
from ._exceptions import (
    AloError,
    ArrowSchurError,
    BasisError,
    CacheStoreError,
    CalibratorError,
    ColumnNotFoundError,
    CorrectedCovarianceError,
    CubicCellKernelError,
    CustomFamilyError,
    DataError,
    DeviationRuntimeError,
    DictionaryConvergenceError,
    EigendecompositionError,
    FitConvergenceError,
    FitError,
    FitInputError,
    FitInvariantError,
    FitNumericalError,
    FitSeedError,
    FittedModelError,
    FormulaError,
    GamError,
    GamlssError,
    GeometryError,
    GpuError,
    GradientUnavailableError,
    HessianNotPositiveDefiniteError,
    HmcError,
    IdentifiabilityCompilerError,
    IllConditionedError,
    InnerModeConvergenceError,
    IntegrationError,
    InvalidConfigurationError,
    InvalidInputError,
    InvalidSpecificationError,
    JointPenaltyError,
    LatentSurvivalError,
    LayoutError,
    LinearAlgebraError,
    LinearSystemSolveError,
    LognormalKernelError,
    MapUniquenessError,
    MatrixError,
    MatrixMaterializationError,
    MissingDependencyError,
    ModelOverparameterizedError,
    MonotoneRootError,
    OuterStrategyError,
    ParameterConstraintError,
    PenaltySpectrumError,
    PerfectSeparationError,
    PirlsConvergenceError,
    PredictInputError,
    PredictionError,
    RemlConvergenceError,
    ScaleDesignError,
    SchemaMismatchError,
    SmoothError,
    SurvivalConstructionError,
    SurvivalError,
    SurvivalLocationScaleError,
    SurvivalMarginalSlopeError,
    SurvivalPredictError,
    TermBuilderError,
    TransformationNormalError,
    UnsupportedLinkError,
)
from ._model import (
    AffineDesign,
    CompetingRisksCIF,
    CompetingRisksPrediction,
    Model,
    MultinomialModel,
    MultinomialPrediction,
    SurvivalPrediction,
    TermBlock,
    competing_risks_cif,
)
from ._response_geometry import (
    ResponseGeometryModel,
    alr,
    closure,
    clr,
    simplex_frechet_mean,
    sphere_frechet_mean,
)
from ._sampling import (
    CumulativeIncidenceDraws,
    PairedPosteriorSamples,
    PosteriorPredictive,
    PosteriorSamples,
    SamplingConfig,
)
from ._tables import PredictionResult
from ._sae_manifold import (
    GumbelTemperatureSchedule,
    ManifoldSAE,
    flat_block_assignment,
    gumbel_geometric_schedule,
    gumbel_linear_schedule,
    gumbel_reciprocal_iter_schedule,
    plot,
    sae_manifold_certify_external,
    sae_manifold_fit,
)
from ._sae_viz import plot_atom, plot_fit
from ._sae_trust import atom_trust_scores, sae_trust_diagnostics
from ._schema import SchemaCheck, SchemaIssue
from ._summary import Summary
from ._validation import FormulaValidation
from .structure_discovery import (
    atom_birth_gate,
    e_bh_dictionary_certificate,
    expected_resolution_budget,
    log_e_from_p_value,
    plan_probe_for_contested_claim,
    select_probe_by_expected_evidence,
    split_likelihood_log_e,
)
from .bartlett import lawley_bartlett_factor, lawley_bartlett_factor_estimated_lambda
from .full_conformal import glm_full_conformal
from .layer_transport import (
    certify_chart_transfer,
    chart_transfer_operator,
    fit_transport,
    layer_transport_fit,
    layer_transport_ladder,
)
from .manifold_crosscoder import sae_crosscoder_fit
from .manifold_behavior import sae_behavior_fit
from .checkpoint_dynamics import sae_checkpoint_dynamics
from .intervention_calibration import ChartCalibration, fit_chart_calibration
from .parameter_decomposition import ParameterDecompositionReport, run_parameter_decomposition
from ._sae_spectral import audit_sae
from .geometry import (
    CircleManifold,
    EuclideanManifold,
    GrassmannManifold,
    ProductManifold,
    SpdManifold,
    SphereManifold,
    StiefelManifold,
    TorusManifold,
)

try:
    __version__ = _metadata.version("gamfit")
except _metadata.PackageNotFoundError:
    __version__ = "0.0.0+unknown"

# Names whose implementation lives behind the optional ``torch`` extra. They
# are loaded lazily while keeping the cold-start import path torch-free.
_LAZY_TORCH_ATTRS: dict[str, tuple[str, str]] = {
    "PoincareAtoms": ("gamfit.torch.hyperbolic", "PoincareAtoms"),
    "InterchangeSwapDecoder": ("gamfit.torch.interchange", "InterchangeSwapDecoder"),
}


if _TYPE_CHECKING:
    from .torch.hyperbolic import PoincareAtoms as PoincareAtoms
    from .torch.interchange import InterchangeSwapDecoder as InterchangeSwapDecoder


def __getattr__(name: str) -> object:
    """Lazy attribute hook for optional-extra primitives exposed at the top level.

    A missing optional dependency (typically ``torch``) is surfaced as
    ``AttributeError`` chained from the underlying ``ModuleNotFoundError``.
    This preserves the Python contract that ``hasattr`` only ever returns a
    bool and that ``from gamfit import *`` does not blow up on torch-less
    installs while torch-specific modules remain under ``gamfit.torch``.
    """
    target = _LAZY_TORCH_ATTRS.get(name)
    if target is not None:
        module_path, attr = target
        from importlib import import_module

        try:
            module = import_module(module_path)
        except ModuleNotFoundError as exc:
            missing = exc.name
            if missing is None or (missing != "torch" and not module_path.startswith(missing)):
                raise
            raise AttributeError(
                f"gamfit.{name} requires an optional dependency that is not "
                f"installed ({exc.name!r}). Install it with: "
                f"pip install torch."
            ) from exc
        return getattr(module, attr)
    raise AttributeError(f"module 'gamfit' has no attribute {name!r}")


# Static so type checkers see every re-export as explicit (PEP 484 re-export
# rules); ``tests/test_public_all_matches_exported_names.py`` pins it to the set of
# public names this module actually binds. Lazy torch attributes stay out so
# ``from gamfit import *`` works on torch-less installs (issue #303).
__all__ = [
    "ARDPenalty",
    "AffineDesign",
    "AloError",
    "AnalyticPenaltyKind",
    "ArrowSchurError",
    "AtlasNerveDiagram",
    "AtomRetentionEvidence",
    "AuxConditionalPriorPenalty",
    "BSpline",
    "BasisDescriptor",
    "BasisError",
    "BasisSpec",
    "BlockCoordinateReport",
    "BlockOrthogonalityPenalty",
    "BlockSparseDictStream",
    "BlockSparseDictionaryConvergence",
    "BlockSparseDictionaryFit",
    "BlockSparseStreamArtifact",
    "BlockSparseStreamConvergence",
    "BlockSparsityPenalty",
    "CacheStoreError",
    "CalibratorError",
    "Categorical",
    "ChartCalibration",
    "ChartInterpNullCalibration",
    "ChartInterpNullCalibrationReport",
    "ChartInterpNullProtocol",
    "ChartInterpReadout",
    "ChartInterpReport",
    "ChartInterpStatisticValue",
    "Circle",
    "CircleManifold",
    "ColumnNotFoundError",
    "CompetingRisksCIF",
    "CompetingRisksPrediction",
    "ComposedContract",
    "CompositePenalty",
    "ConditionalCoactivationInfluence",
    "CoordinatePosterior",
    "CorrectedCovarianceError",
    "CouplingRobustnessCertificate",
    "CtnStage1",
    "CubicCellKernelError",
    "CumulativeIncidenceDraws",
    "CustomFamilyError",
    "Cylinder",
    "DataError",
    "DeviationRuntimeError",
    "Diagnostics",
    "DictionaryConvergenceError",
    "DoseResponseCalibrationReport",
    "DualCertificateReport",
    "Duchon",
    "EigendecompositionError",
    "EuclideanManifold",
    "EuclideanPatch",
    "EventHistoryModel",
    "FisherEffectEvidence",
    "FitConvergenceError",
    "FitError",
    "FitInputError",
    "FitInvariantError",
    "FitNumericalError",
    "FitSeedError",
    "FittedModelError",
    "FormulaError",
    "FormulaValidation",
    "GamError",
    "GamInferenceWarning",
    "GamlssError",
    "GatedSAEDecoder",
    "GaugeCompanion",
    "GeometryError",
    "GpuError",
    "GradientUnavailableError",
    "GrassmannManifold",
    "GumbelTemperatureSchedule",
    "HessianNotPositiveDefiniteError",
    "HmcError",
    "HolonomyReport",
    "IdentifiabilityCompilerError",
    "IdentifiabilityReport",
    "IdentifiabilityTheoremResult",
    "IdentifiableFactorFitResult",
    "IllConditionedError",
    "InnerModeConvergenceError",
    "IntegrationError",
    "InvalidConfigurationError",
    "InvalidInputError",
    "InvalidSpecificationError",
    "IsometryPenalty",
    "IvaeRidgeMeanGauge",
    "JointEventModel",
    "JointPenaltyError",
    "LabelShuffleMarginNull",
    "LatentCoord",
    "LatentSurvivalError",
    "LayoutError",
    "LinearAlgebraError",
    "LinearDictionaryFit",
    "LinearSystemSolveError",
    "LognormalKernelError",
    "ManifoldDescriptor",
    "ManifoldSAE",
    "MapUniquenessError",
    "Matern",
    "MatrixError",
    "MatrixMaterializationError",
    "MeasureJet",
    "MechanismSparsityPenalty",
    "MissingDependencyError",
    "Model",
    "ModelOverparameterizedError",
    "MonotoneRootError",
    "MultinomialModel",
    "MultinomialPrediction",
    "NuclearNormPenalty",
    "OrderedBetaBernoulliPenalty",
    "OrthogonalityPenalty",
    "OuterStrategyError",
    "PENALTY_MANIFEST",
    "PairedPosteriorSamples",
    "ParameterConstraintError",
    "ParameterDecompositionReport",
    "ParametricAuxConditionalPriorPenalty",
    "PartialSupervisionExample",
    "PartialSupervisionFit",
    "Pca",
    "Penalty",
    "PenaltyDescriptor",
    "PenaltySpectrumError",
    "PerfectSeparationError",
    "PeriodicHarmonic",
    "PeriodicSplineCurve",
    "PirlsConvergenceError",
    "PosteriorPredictive",
    "PosteriorSamples",
    "PredictInputError",
    "PredictionError",
    "PredictionResult",
    "ProductManifold",
    "RemlConvergenceError",
    "ResponseGeometryModel",
    "RoutabilityAudit",
    "RoutabilityFloor",
    "RustExtensionUnavailableError",
    "SUPPORT_SAE_SCHEMA",
    "SaeSupervisedFit",
    "SamplingConfig",
    "ScadMcpPenalty",
    "ScalarWeightSchedule",
    "ScaleDesignError",
    "SchemaCheck",
    "SchemaIssue",
    "SchemaMismatchError",
    "ScoreKind",
    "ScoreScale",
    "SelectTopologyResult",
    "ShapeConstraintLiteral",
    "ShapeControlledCensus",
    "SharedPrecisionGroup",
    "SheafConsistencyPenalty",
    "Smooth",
    "SmoothError",
    "SmoothSpec",
    "SmoothSum",
    "SmoothThresholdPenalty",
    "SoftmaxAssignmentSparsityPenalty",
    "SparseDictStream",
    "SparseDictStreamArtifact",
    "SparseDictionaryConvergence",
    "SparseDictionaryFit",
    "SparsityPenalty",
    "SpdManifold",
    "SpectrometerReport",
    "Sphere",
    "SphereManifold",
    "SpikeRecovery",
    "StageContainment",
    "StiefelManifold",
    "Summary",
    "SurvivalConstructionError",
    "SurvivalError",
    "SurvivalLocationScaleError",
    "SurvivalMarginalSlopeError",
    "SurvivalPredictError",
    "SurvivalPrediction",
    "TensorBSpline",
    "TermBlock",
    "TermBuilderError",
    "TopKActivationPenalty",
    "TopologyAutoSelector",
    "TopologyAutoSelectorRank",
    "TopologyAutoSelectorResult",
    "TopologyCandidateFailure",
    "TopologySelectionError",
    "TopologySphere",
    "TopologyStack",
    "Torus",
    "TorusManifold",
    "TotalVariationPenalty",
    "TransformationNormalError",
    "UnsupportedLinkError",
    "VarianceChargeEvidence",
    "WholeSetContainment",
    "__version__",
    "adjudicate_atom_shape",
    "alr",
    "atlas_nerve_diagram",
    "atom_birth_gate",
    "atom_trust_scores",
    "audit_sae",
    "block_firing_coordinates",
    "block_sparse_dictionary_fit",
    "block_sparse_dictionary_fit_begin",
    "bspline_basis",
    "bspline_basis_derivative",
    "build_info",
    "certify_chart_transfer",
    "chart_interp_score",
    "chart_transfer_operator",
    "closure",
    "clr",
    "compare_models",
    "competing_risks_cif",
    "compose_contracts",
    "conditional_coactivation_influence",
    "conditional_prior_ivae",
    "coordinate_posterior_from_precision",
    "coupling_robustness_certificate",
    "cross_fit_shared_precision_groups",
    "cuda_diagnostics",
    "cuda_subprocess_env",
    "cuda_subprocess_library_dirs",
    "derive_ivae_aux_scale",
    "diagnostics",
    "dimension_spectrometer",
    "dose_response_calibration",
    "duchon_basis",
    "duchon_function_norm_penalty",
    "e_bh_dictionary_certificate",
    "effect_weighted_retention",
    "emit_inference_warnings",
    "examples",
    "expected_resolution_budget",
    "explain_error",
    "fit",
    "fit_array",
    "fit_chart_calibration",
    "fit_event_history",
    "fit_joint_event_model",
    "fit_transport",
    "fixed_budget_block_sparse_dictionary_fit",
    "flat_block_assignment",
    "format_cuda_diagnostics",
    "gauge_companion",
    "gaussian_reml_fit",
    "gaussian_reml_fit_backward",
    "gaussian_reml_fit_batched",
    "gaussian_reml_fit_batched_backward",
    "gaussian_reml_fit_blocks_backward",
    "gaussian_reml_fit_blocks_forward",
    "gaussian_reml_fit_formula",
    "gaussian_reml_fit_latent",
    "gaussian_reml_fit_latent_backward",
    "gaussian_reml_fit_positions",
    "gaussian_reml_fit_positions_backward",
    "gaussian_reml_fit_positions_batched",
    "gaussian_reml_fit_positions_batched_backward",
    "gaussian_reml_fit_with_constraints_backward",
    "gaussian_reml_fit_with_constraints_forward",
    "gaussian_reml_optimize_latent",
    "gaussian_weighted_ridge",
    "gaussian_weighted_ridge_batch",
    "glm_full_conformal",
    "glm_reml_fit_latent",
    "glm_reml_fit_latent_backward",
    "gumbel_geometric_schedule",
    "gumbel_linear_schedule",
    "gumbel_reciprocal_iter_schedule",
    "identifiability",
    "identifiability_check",
    "identifiable_factor_fit",
    "kernels",
    "lawley_bartlett_factor",
    "lawley_bartlett_factor_estimated_lambda",
    "layer_transport_fit",
    "layer_transport_ladder",
    "linear_dictionary_fit",
    "load",
    "load_joint_event_model",
    "loads",
    "log_e_from_p_value",
    "loop_holonomy",
    "manifolds",
    "matern_basis",
    "mechanism_sparsity_jacobian",
    "model_from_dict",
    "partial_supervision",
    "periodic_spline_curve_basis",
    "plan_probe_for_contested_claim",
    "plot",
    "plot_atom",
    "plot_fit",
    "recover_spikes",
    "rho_so2",
    "rho_so2_jvp",
    "rho_so3",
    "rho_so3_jvp",
    "routability_audit",
    "routability_floor",
    "run_label_shuffle_margin_null",
    "run_parameter_decomposition",
    "run_shape_controlled_census",
    "sae_behavior_fit",
    "sae_checkpoint_dynamics",
    "sae_crosscoder_fit",
    "sae_manifold_certify_external",
    "sae_manifold_fit",
    "sae_supervised",
    "sae_trust_diagnostics",
    "save",
    "select_probe_by_expected_evidence",
    "select_topology",
    "separation_limit",
    "shape_matched_control",
    "shape_matched_control_f32",
    "simplex_frechet_mean",
    "smoothness_penalty",
    "sparse_dict_dual_certificate",
    "sparse_dictionary_fit",
    "sparse_dictionary_fit_begin",
    "sphere_basis",
    "sphere_basis_jet",
    "sphere_frechet_mean",
    "split_likelihood_log_e",
    "stack_topologies",
    "topology",
    "validate_formula",
    "whole_set_containment",
]
