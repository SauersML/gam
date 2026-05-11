"""Formula-first generalized additive models with a high-performance Rust core.

Fit Gaussian, binomial, Poisson, and Gamma GLMs with smooth terms, random
effects, location-scale extensions, survival likelihoods, and learnable
links. Smoothing parameters are selected by REML or LAML; posterior
sampling uses NUTS.

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

from ._api import build_info, explain_error, fit, load, loads, validate_formula
from ._binding import RustExtensionUnavailableError
from ._diagnostics import Diagnostics
from ._exceptions import FormulaError, GamError, PredictionError, SchemaMismatchError
from ._model import Model, SurvivalPrediction
from ._sampling import PosteriorPredictive, PosteriorSamples, SamplingConfig
from ._schema import SchemaCheck, SchemaIssue
from ._summary import Summary
from ._validation import FormulaValidation

try:
    __version__ = _metadata.version("gamfit")
except _metadata.PackageNotFoundError:
    __version__ = "0.0.0+unknown"

def load_posterior(path: object) -> PosteriorSamples:
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
    return PosteriorSamples.load(path)  # type: ignore[arg-type]


__all__ = [
    "Diagnostics",
    "FormulaError",
    "FormulaValidation",
    "GamError",
    "Model",
    "PosteriorPredictive",
    "PosteriorSamples",
    "PredictionError",
    "RustExtensionUnavailableError",
    "SamplingConfig",
    "SchemaCheck",
    "SchemaIssue",
    "SchemaMismatchError",
    "Summary",
    "SurvivalPrediction",
    "__version__",
    "build_info",
    "explain_error",
    "fit",
    "load",
    "load_posterior",
    "loads",
    "validate_formula",
]
