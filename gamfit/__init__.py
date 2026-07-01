"""Formula-first generalized additive models with a high-performance Rust core.

Fit Gaussian, binomial, Poisson, and Gamma GLMs with smooth terms, random
effects, location-scale extensions, survival likelihoods, and learnable
links. Smoothing parameters are selected by REML or LAML; posterior
sampling uses NUTS. Geometric / manifold smooths (cyclic 1-D, cylinder
/ torus tensor, intrinsic sphere, boundary-conditioned B-splines) make
predictor spaces that wrap or close first-class.

The public surface also includes latent-coordinate and SAE-manifold tools:
analytic penalties such as ``ScadMcpPenalty`` and ``NuclearNormPenalty``;
assignment-family descriptors for softmax / finite-IBP / top-k / JumpReLU
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

# Source-vs-wheel skew note:
# This Python source expects the Rust extension from gam-pyffi >= 0.1.124.
# Wheels published before that extension version may lack these pyfunctions:
# - sae_manifold_reconstruction_r2
# - sae_manifold_assignment_summary
# - topology_dispatch_key
# - assemble_candidate_formula
# - rank_topology_candidates
# Rebuild or reinstall the local extension if importing from a source tree with
# an older compiled .so.

from importlib import metadata as _metadata
from pathlib import Path

from ._api import (
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
    save,
    mechanism_sparsity_jacobian,
    periodic_spline_curve_basis,
    smoothness_penalty,
    sphere_basis,
    validate_formula,
)
from ._binding import RustExtensionUnavailableError
from ._warnings import GamInferenceWarning, emit_inference_warnings
from ._rust import adjudicate_atom_shape  # cross-class atom-shape adjudicator (Rust)
from ._compare import compare_models
from ._linear_dictionary import LinearDictionaryFit, linear_dictionary_fit
from ._sparse_dictionary import SparseDictionaryFit, sparse_dictionary_fit
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
    TopologyStack,
    select_topology,
    stack_topologies,
)
from ._diagnostics import Diagnostics
from . import diagnostics
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
from ._basis_descriptors import Fourier, PeriodicHarmonic
from ._composite_penalty import CompositePenalty
from ._smooth import Smooth, SmoothSum  # compositional Smooth(latent=..., basis=..., penalty=...)
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
from . import topology
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
    EigendecompositionError,
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
from .layer_transport import fit_transport, layer_transport_fit, layer_transport_ladder
from .checkpoint_dynamics import sae_checkpoint_dynamics
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
_LAZY_TORCH_ATTRS: dict[str, tuple[str, str]] = {}


_EXCLUDE_FROM_ALL = {"Path"}


def __getattr__(name: str):
    """Lazy attribute hook for optional-extra primitives exposed at the top level.

    A missing optional dependency (typically ``torch``) is surfaced as
    ``AttributeError`` chained from the underlying ``ModuleNotFoundError``.
    This preserves the Python contract that ``hasattr`` only ever returns a
    bool and that ``from gamfit import *`` does not blow up on torch-less
    installs, while ``gamfit.AdaptiveTopK`` (direct access) still produces a
    clear actionable message pointing at ``pip install torch``.
    """
    target = _LAZY_TORCH_ATTRS.get(name)
    if target is not None:
        module_path, attr = target
        from importlib import import_module
        try:
            module = import_module(module_path)
        except ModuleNotFoundError as exc:
            if exc.name != "torch" and not module_path.startswith(exc.name):
                raise
            raise AttributeError(
                f"gamfit.{name} requires an optional dependency that is not "
                f"installed ({exc.name!r}). Install it with: "
                f"pip install torch."
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


def _build_public_api() -> list[str]:
    """Derive ``__all__`` from what is actually importable at this install.

    Hand-maintaining ``__all__`` is brittle: every new top-level export has to
    be added in two places, and lazy-torch names placed in ``__all__`` blow
    up ``from gamfit import *`` on torch-less installs because Python's
    star-import iterates ``__all__`` and runs ``getattr`` on every entry
    (issue #303).

    Instead, derive the public API at import time:
      1. every non-underscore name currently bound in the module globals
         (the heavy ``from ._x import ...`` blocks above), minus submodules
         and private re-imports;
      2. an explicit allowlist of submodule attributes (``diagnostics``,
         ``examples``, ``topology``, ``identifiability``, ``manifolds``,
         ``kernels``) that are part of the public API even though they are
         module objects.
    """
    from types import ModuleType

    public_submodules = {
        "diagnostics",
        "topology",
        "manifolds",
        "kernels",
    }
    api: set[str] = set()
    for name, value in globals().items():
        if name.startswith("_"):
            continue
        if name in _EXCLUDE_FROM_ALL:
            continue
        # Skip module objects unless they are explicitly in the public
        # submodule allowlist: ``importlib``, ``Path``, etc. were imported
        # for internal use and should not leak into star imports.
        if isinstance(value, ModuleType) and name not in public_submodules:
            continue
        api.add(name)
    api.add("__version__")
    return sorted(api)


__all__ = _build_public_api()
