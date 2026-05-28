"""Formula-first generalized additive models with a high-performance Rust core.

Fit Gaussian, binomial, Poisson, and Gamma GLMs with smooth terms, random
effects, location-scale extensions, survival likelihoods, and learnable
links. Smoothing parameters are selected by REML or LAML; posterior
sampling uses NUTS. Geometric / manifold smooths (cyclic 1-D, cylinder
/ torus tensor, intrinsic sphere, boundary-conditioned B-splines) make
predictor spaces that wrap or close first-class — no seams, no pole
artefacts.

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

# Source-vs-wheel skew note:
# This Python source expects the Rust extension from gam-pyffi >= 0.1.124.
# Wheels published before that extension version may lack these pyfunctions:
# - sae_manifold_reconstruction_r2
# - sae_manifold_assignment_summary
# - topology_dispatch_key
# - assemble_candidate_formula
# Rebuild or reinstall the local extension if importing from a source tree with
# an older compiled .so.

from importlib import metadata as _metadata
from pathlib import Path

from ._api import (
    SharedPrecisionGroup,
    bspline_basis,
    bspline_basis_derivative,
    build_info,
    conditional_prior_ivae,
    cross_fit_shared_precision_groups,
    cuda_subprocess_env,
    cuda_subprocess_library_dirs,
    cuda_diagnostics,
    duchon_basis,
    duchon_function_norm_penalty,
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
    save,
    mechanism_sparsity_jacobian,
    periodic_spline_curve_basis,
    smoothness_penalty,
    sphere_basis,
    validate_formula,
)
from ._binding import RustExtensionUnavailableError
from ._compare import compare_models
from ._penalties import (
    ARDPenalty,
    AnalyticPenaltyKind,
    AuxConditionalPriorPenalty,
    BlockOrthogonalityPenalty,
    BlockSparsityPenalty,
    GatedSAEDecoder,
    IBPAssignmentPenalty,
    IsometryPenalty,
    IvaeRidgeMeanGauge,
    JumpReLUPenalty,
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
    select_topology,
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
    EquivariantPenalty,
    GaugeCompanion,
    LieAtom,
    equivariant_smooth,
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
from ._basis_descriptors import Fourier, PeriodicHarmonic
from ._composite_penalty import CompositePenalty
from ._smooth import Smooth, SmoothSum  # compositional Smooth(latent=..., basis=..., penalty=...)
ComposedSmooth = Smooth  # explicit alias for callers preferring the descriptive name
from ._penalty_descriptors import (
    ARDPenalty as _ARDPenaltyDescriptor,
    BlockOrthogonalityDescriptor,
    IBPPenalty,
    MechanismSparsityDescriptor,
)

# Promote the torch-aware descriptor classes to the top-level penalty names so
# `gamfit.ARDPenalty(0.1) + gamfit.IBPPenalty(1.0)` works uniformly through
# the new BasisDescriptor/PenaltyDescriptor protocol. The original
# Rust-pyclass descriptors used by the formula pipeline remain reachable as
# `gamfit._penalties.ARDPenalty`, `gamfit._penalties.BlockOrthogonalityPenalty`,
# etc., and continue to drive the REML core.
ARDPenalty = _ARDPenaltyDescriptor
from . import recipes, topology
from .recipes import (
    PartialSupervisionFit,
    PartialSupervisionRecipe,
    SaeSupervisedFit,
    partial_supervision,
    sae_supervised,
)
from ._exceptions import FormulaError, GamError, PredictionError, SchemaMismatchError
from ._model import (
    CompetingRisksCIF,
    CompetingRisksPrediction,
    Model,
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
from ._sae_manifold import (
    GumbelTemperatureSchedule,
    ManifoldSAE,
    SaeManifoldAtomFit,
    SaeManifoldFitResult,
    gumbel_geometric_schedule,
    gumbel_linear_schedule,
    gumbel_reciprocal_iter_schedule,
    sae_manifold_fit,
)
from ._schema import SchemaCheck, SchemaIssue
from .sindy import SINDyAtoms
from ._summary import Summary
from ._validation import FormulaValidation
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

# Names whose implementation lives behind the optional ``torch`` extra. They are
# advertised at the top level so users can write ``gamfit.AdaptiveTopK(...)``
# without an explicit ``gamfit.torch`` import, while keeping the cold-start
# import path torch-free.
_LAZY_TORCH_ATTRS: dict[str, tuple[str, str]] = {
    "AdaptiveTopK":          ("gamfit.torch.modules",     "AdaptiveTopK"),
    "Crosscoder":            ("gamfit.crosscoder",        "Crosscoder"),
    "PoincareAtoms":         ("gamfit.torch.hyperbolic",  "PoincareAtoms"),
    "InterchangeSwapDecoder":("gamfit.torch.interchange", "InterchangeSwapDecoder"),
}


def __getattr__(name: str):
    """Lazy attribute hook for optional-extra primitives exposed at the top level.

    A missing optional dependency (typically ``torch``) is surfaced as
    ``AttributeError`` chained from the underlying ``ModuleNotFoundError``.
    This preserves the Python contract that ``hasattr`` only ever returns a
    bool and that ``from gamfit import *`` does not blow up on torch-less
    installs, while ``gamfit.AdaptiveTopK`` (direct access) still produces a
    clear actionable message pointing at the ``gamfit[torch]`` extra.
    """
    target = _LAZY_TORCH_ATTRS.get(name)
    if target is not None:
        module_path, attr = target
        from importlib import import_module
        try:
            module = import_module(module_path)
        except ModuleNotFoundError as exc:
            raise AttributeError(
                f"gamfit.{name} requires an optional dependency that is not "
                f"installed ({exc.name!r}). Install the torch extra: "
                f"pip install 'gamfit[torch]'."
            ) from exc
        return getattr(module, attr)
    raise AttributeError(f"module 'gamfit' has no attribute {name!r}")


def load_posterior(path: str | Path) -> PosteriorSamples:
    """Load a :class:`PosteriorSamples` archive from disk.

    Thin wrapper around :meth:`PosteriorSamples.load` provided for symmetry
    with :func:`gamfit.load` / :func:`gamfit.fit` at module level.

    Parameters
    ----------
    path : str or pathlib.Path
        Filesystem path to an ``.npz`` archive previously written by
        :meth:`PosteriorSamples.save`.

    Returns
    -------
    PosteriorSamples
        Reconstructed posterior draws and metadata.

    Examples
    --------
    >>> draws = gamfit.load_posterior("posterior.npz")
    >>> draws.beta.shape
    (1000, 42)
    """
    return PosteriorSamples.load(path)


__all__ = [
    # Note: the lazy-torch primitives (AdaptiveTopK, Crosscoder,
    # InterchangeSwapDecoder, PoincareAtoms) are intentionally NOT in __all__.
    # They remain reachable as ``gamfit.AdaptiveTopK`` via ``__getattr__``;
    # excluding them here keeps ``from gamfit import *`` working on torch-less
    # installs where attribute access would otherwise raise AttributeError
    # mid-iteration (see issue #303).
    "Diagnostics",
    "diagnostics",
    "EquivariantPenalty",
    "GaugeCompanion",
    "LieAtom",
    "PartialSupervisionFit",
    "PartialSupervisionRecipe",
    "SaeSupervisedFit",
    "partial_supervision",
    "sae_supervised",
    "recipes",
    "equivariant_smooth",
    "gauge_companion",
    "rho_so2",
    "rho_so2_jvp",
    "rho_so3",
    "rho_so3_jvp",
    "FormulaError",
    "FormulaValidation",
    "GamError",
    "GumbelTemperatureSchedule",
    "ManifoldSAE",
    "CompetingRisksCIF",
    "CompetingRisksPrediction",
    "Model",
    "CumulativeIncidenceDraws",
    "PairedPosteriorSamples",
    "PosteriorPredictive",
    "PosteriorSamples",
    "PredictionError",
    "ResponseGeometryModel",
    "RustExtensionUnavailableError",
    "SamplingConfig",
    "SaeManifoldAtomFit",
    "SaeManifoldFitResult",
    "ScalarWeightSchedule",
    "SINDyAtoms",
    "SchemaCheck",
    "SchemaIssue",
    "SchemaMismatchError",
    "SharedPrecisionGroup",
    "CircleManifold",
    "EuclideanManifold",
    "GrassmannManifold",
    "ProductManifold",
    "SpdManifold",
    "SphereManifold",
    "StiefelManifold",
    "TorusManifold",
    "BasisSpec",
    "ARDPenalty",
    "AnalyticPenaltyKind",
    "AuxConditionalPriorPenalty",
    "BlockOrthogonalityPenalty",
    "SheafConsistencyPenalty",
    "BlockSparsityPenalty",
    "GatedSAEDecoder",
    "IBPAssignmentPenalty",
    "IdentifiabilityReport",
    "IdentifiabilityTheoremResult",
    "IdentifiableFactorFitResult",
    "identifiability",
    "identifiability_check",
    "identifiable_factor_fit",
    "IsometryPenalty",
    "IvaeRidgeMeanGauge",
    "MechanismSparsityPenalty",
    "NuclearNormPenalty",
    "OrthogonalityPenalty",
    "ParametricAuxConditionalPriorPenalty",
    "Penalty",
    "PENALTY_MANIFEST",
    "Pca",
    "ScadMcpPenalty",
    "SoftmaxAssignmentSparsityPenalty",
    "SparsityPenalty",
    "TopKActivationPenalty",
    "JumpReLUPenalty",
    "TotalVariationPenalty",
    "Summary",
    "ScoreKind",
    "ScoreScale",
    "SelectTopologyResult",
    "SurvivalPrediction",
    "TermBlock",
    "__version__",
    "alr",
    "build_info",
    "bspline_basis",
    "bspline_basis_derivative",
    "closure",
    "clr",
    "compare_models",
    "conditional_prior_ivae",
    "competing_risks_cif",
    "cross_fit_shared_precision_groups",
    "cuda_diagnostics",
    "cuda_subprocess_env",
    "cuda_subprocess_library_dirs",
    "BSpline",
    "Categorical",
    "Duchon",
    "LatentCoord",
    "Matern",
    "PeriodicSplineCurve",
    "ShapeConstraintLiteral",
    "Smooth",
    "Sphere",
    "TensorBSpline",
    "topology",
    "Circle",
    "Cylinder",
    "EuclideanPatch",
    "TopologySphere",
    "TopologyAutoSelector",
    "TopologyAutoSelectorRank",
    "TopologyAutoSelectorResult",
    "Torus",
    "duchon_basis",
    "duchon_function_norm_penalty",
    "explain_error",
    "fit",
    "format_cuda_diagnostics",
    "fit_array",
    "gaussian_reml_fit",
    "gaussian_reml_fit_backward",
    "gaussian_reml_fit_batched",
    "gaussian_reml_fit_batched_backward",
    "gaussian_reml_fit_blocks_backward",
    "gaussian_reml_fit_blocks_forward",
    "gaussian_reml_fit_formula",
    "gaussian_reml_fit_latent",
    "gaussian_reml_fit_latent_backward",
    "glm_reml_fit_latent",
    "glm_reml_fit_latent_backward",
    "gaussian_reml_fit_positions",
    "gaussian_reml_fit_positions_backward",
    "gaussian_reml_fit_positions_batched",
    "gaussian_reml_fit_positions_batched_backward",
    "gaussian_reml_fit_with_constraints_backward",
    "gaussian_reml_fit_with_constraints_forward",
    "gaussian_weighted_ridge",
    "gaussian_weighted_ridge_batch",
    "gumbel_geometric_schedule",
    "gumbel_linear_schedule",
    "gumbel_reciprocal_iter_schedule",
    "load",
    "load_posterior",
    "loads",
    "save",
    "mechanism_sparsity_jacobian",
    "periodic_spline_curve_basis",
    "sphere_basis",
    "simplex_frechet_mean",
    "smoothness_penalty",
    "sphere_frechet_mean",
    "sae_manifold_fit",
    "select_topology",
    "validate_formula",
]
