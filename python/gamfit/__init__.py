"""Formula-first generalized additive models with a high-performance Rust core.

Fit Gaussian, binomial, Poisson, and Gamma GLMs with smooth terms, random
effects, location-scale extensions, survival likelihoods, and learnable
links. Smoothing parameters are selected by REML or LAML; posterior
sampling uses NUTS.

Quick start::

    import gamfit

    model = gamfit.fit(train, "y ~ s(x)")
    pred = model.predict(test, interval=0.95)
    print(model.summary())
    model.save("model.gam")

See https://github.com/SauersML/gam for the full guide.
"""

from importlib import metadata as _metadata

from . import pgs
from ._api import build_info, explain_error, fit, load, loads, validate_formula
from ._binding import RustExtensionUnavailableError
from ._diagnostics import Diagnostics
from ._exceptions import FormulaError, GamError, PredictionError, SchemaMismatchError
from ._model import Model, SurvivalPrediction
from ._schema import SchemaCheck, SchemaIssue
from ._summary import Summary
from ._validation import FormulaValidation

try:
    __version__ = _metadata.version("gamfit")
except _metadata.PackageNotFoundError:
    __version__ = "0.0.0+unknown"

__all__ = [
    "Diagnostics",
    "FormulaError",
    "FormulaValidation",
    "GamError",
    "Model",
    "PredictionError",
    "RustExtensionUnavailableError",
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
    "loads",
    "pgs",
    "validate_formula",
]
