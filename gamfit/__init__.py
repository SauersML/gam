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

See https://github.com/SauersML/gam for the full guide.
"""

from importlib import metadata as _metadata
from pathlib import Path

from ._api import (
    bspline_basis,
    bspline_basis_derivative,
    build_info,
    cuda_diagnostics,
    duchon_basis_1d,
    duchon_basis_1d_derivative,
    explain_error,
    fit,
    format_cuda_diagnostics,
    fit_array,
    gaussian_reml_fit,
    gaussian_reml_fit_backward,
    gaussian_reml_fit_batched,
    gaussian_reml_fit_batched_backward,
    gaussian_reml_fit_formula,
    gaussian_reml_fit_positions,
    gaussian_reml_fit_positions_backward,
    gaussian_reml_fit_positions_batched,
    gaussian_reml_fit_positions_batched_backward,
    gaussian_weighted_ridge,
    gaussian_weighted_ridge_batch,
    load,
    loads,
    smoothness_penalty,
    validate_formula,
)
from ._binding import RustExtensionUnavailableError
from ._diagnostics import Diagnostics
from ._exceptions import FormulaError, GamError, PredictionError, SchemaMismatchError
from ._model import CompetingRisksCIF, Model, SurvivalPrediction, competing_risks_cif
from ._response_geometry import (
    ResponseGeometryModel,
    alr,
    closure,
    clr,
    simplex_frechet_mean,
    sphere_frechet_mean,
)
from ._sampling import PosteriorPredictive, PosteriorSamples, SamplingConfig
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
    "CompetingRisksCIF",
    "Model",
    "PosteriorPredictive",
    "PosteriorSamples",
    "PredictionError",
    "ResponseGeometryModel",
    "RustExtensionUnavailableError",
    "SamplingConfig",
    "SchemaCheck",
    "SchemaIssue",
    "SchemaMismatchError",
    "Summary",
    "SurvivalPrediction",
    "__version__",
    "alr",
    "build_info",
    "bspline_basis",
    "bspline_basis_derivative",
    "closure",
    "clr",
    "competing_risks_cif",
    "cuda_diagnostics",
    "duchon_basis_1d",
    "duchon_basis_1d_derivative",
    "explain_error",
    "fit",
    "format_cuda_diagnostics",
    "fit_array",
    "gaussian_reml_fit",
    "gaussian_reml_fit_backward",
    "gaussian_reml_fit_batched",
    "gaussian_reml_fit_batched_backward",
    "gaussian_reml_fit_formula",
    "gaussian_reml_fit_positions",
    "gaussian_reml_fit_positions_backward",
    "gaussian_reml_fit_positions_batched",
    "gaussian_reml_fit_positions_batched_backward",
    "gaussian_weighted_ridge",
    "gaussian_weighted_ridge_batch",
    "load",
    "load_posterior",
    "loads",
    "simplex_frechet_mean",
    "smoothness_penalty",
    "sphere_frechet_mean",
    "validate_formula",
]
