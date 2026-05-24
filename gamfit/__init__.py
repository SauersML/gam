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

from importlib import metadata as _metadata
from pathlib import Path

from ._api import (
    SharedPrecisionGroup,
    bspline_basis,
    bspline_basis_derivative,
    build_info,
    cross_fit_shared_precision_groups,
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
    periodic_spline_curve_basis,
    smoothness_penalty,
    sphere_basis,
    validate_formula,
)
from ._binding import RustExtensionUnavailableError
from ._compare import compare_models
from ._penalties import (
    ARDPenalty,
    AuxConditionalPriorPenalty,
    BlockSparsityPenalty,
    IBPAssignmentPenalty,
    IsometryPenalty,
    NuclearNormPenalty,
    OrthogonalityPenalty,
    Penalty,
    SoftmaxAssignmentSparsityPenalty,
    SparsityPenalty,
    TotalVariationPenalty,
)
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
    select_topology,
)
from ._diagnostics import Diagnostics
from .smooth import (
    BSpline,
    Categorical,
    Duchon,
    LatentCoord,
    Matern,
    PeriodicSplineCurve,
    ShapeConstraintLiteral,
    Smooth,
    Sphere,
    TensorBSpline,
)
from . import topology
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
    SaeManifoldAtomFit,
    SaeManifoldFitResult,
    gumbel_geometric_schedule,
    gumbel_linear_schedule,
    gumbel_reciprocal_iter_schedule,
    sae_manifold_fit,
)
from ._schema import SchemaCheck, SchemaIssue
from ._summary import Summary
from ._validation import FormulaValidation

try:
    __version__ = _metadata.version("gamfit")
except _metadata.PackageNotFoundError:
    __version__ = "0.0.0+unknown"

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
    "Diagnostics",
    "FormulaError",
    "FormulaValidation",
    "GamError",
    "GumbelTemperatureSchedule",
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
    "SchemaCheck",
    "SchemaIssue",
    "SchemaMismatchError",
    "SharedPrecisionGroup",
    "BasisSpec",
    "ARDPenalty",
    "AuxConditionalPriorPenalty",
    "BlockSparsityPenalty",
    "IBPAssignmentPenalty",
    "IsometryPenalty",
    "NuclearNormPenalty",
    "OrthogonalityPenalty",
    "Penalty",
    "SoftmaxAssignmentSparsityPenalty",
    "SparsityPenalty",
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
    "competing_risks_cif",
    "cross_fit_shared_precision_groups",
    "cuda_diagnostics",
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
    "periodic_spline_curve_basis",
    "sphere_basis",
    "simplex_frechet_mean",
    "smoothness_penalty",
    "sphere_frechet_mean",
    "sae_manifold_fit",
    "select_topology",
    "validate_formula",
]
