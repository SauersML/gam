"""Formula-first generalized additive models with a high-performance Rust core.

Fit Gaussian, binomial, Poisson, and Gamma GLMs with smooth terms, random
effects, location-scale extensions, survival likelihoods, and learnable
links. Smoothing parameters are selected by REML or LAML; posterior
sampling uses NUTS. Geometric / manifold smooths (cyclic 1-D, cylinder
/ torus tensor, intrinsic sphere, boundary-conditioned B-splines) make
predictor spaces that wrap or close first-class.

The top level holds the fit / load entry points and the fitted-model
classes. Everything else lives in a public submodule, loaded on first
attribute access:

- ``gamfit.errors`` -- the exception hierarchy (``gamfit.errors.GamError``, ...)
- ``gamfit.results`` -- result, prediction, and posterior-sample types
- ``gamfit.plot`` -- matplotlib plotting (optional ``gamfit[plot]`` extra)
- ``gamfit.smooth`` / ``gamfit.basis`` / ``gamfit.penalties`` -- term
  specifications, raw basis builders, and analytic penalties
- ``gamfit.reml`` -- array-level REML / ridge primitives
- ``gamfit.topology`` / ``gamfit.manifolds`` / ``gamfit.geometry`` -- latent
  topologies, topology selection, and manifold descriptors
- ``gamfit.sae`` -- sparse-dictionary and SAE-manifold tools
- ``gamfit.identifiability``, ``gamfit.inference``, ``gamfit.response_geometry``,
  ``gamfit.diagnostics``, ``gamfit.kernels``, ``gamfit.cuda``,
  ``gamfit.examples``, ``gamfit.sklearn``, ``gamfit.torch``

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

from importlib import import_module as _import_module
from importlib import metadata as _metadata

from ._api import (
    CtnStage1,
    build_info,
    explain_error,
    fit,
    fit_array,
    load,
    loads,
    validate_formula,
)
from ._compare import compare_models
from ._event_history import EventHistoryModel, fit_event_history
from ._joint_events import JointEventModel, fit_joint_event_model, load_joint_event_model
from ._model import Model, MultinomialModel, competing_risks_cif
from ._response_geometry import ResponseGeometryModel

try:
    __version__ = _metadata.version("gamfit")
except _metadata.PackageNotFoundError:
    __version__ = "0.0.0+unknown"

__all__ = [
    "CtnStage1",
    "EventHistoryModel",
    "JointEventModel",
    "Model",
    "MultinomialModel",
    "ResponseGeometryModel",
    "__version__",
    "build_info",
    "compare_models",
    "competing_risks_cif",
    "explain_error",
    "fit",
    "fit_array",
    "fit_event_history",
    "fit_joint_event_model",
    "load",
    "load_joint_event_model",
    "loads",
    "validate_formula",
]

# Public submodules. They are imported on first attribute access so that
# ``import gamfit`` stays cheap and never pulls in optional dependencies.
# They are not in ``__all__``: ``from gamfit import *`` must not import torch.
_SUBMODULES = frozenset(
    {
        "basis",
        "cuda",
        "diagnostics",
        "errors",
        "examples",
        "geometry",
        "identifiability",
        "inference",
        "kernels",
        "kernels_jax",
        "kernels_torch",
        "manifolds",
        "penalties",
        "plot",
        "reml",
        "response_geometry",
        "results",
        "sae",
        "sklearn",
        "smooth",
        "topology",
        "torch",
    }
)


def __getattr__(name: str):
    """Import a public submodule on first access.

    A submodule whose optional dependency (e.g. ``torch``) is missing raises
    ``AttributeError`` chained from the ``ModuleNotFoundError``, so
    ``hasattr(gamfit, "torch")`` returns a bool instead of raising.
    """
    if name in _SUBMODULES:
        try:
            return _import_module(f"{__name__}.{name}")
        except ModuleNotFoundError as exc:
            if exc.name is None or exc.name.startswith(__name__):
                raise
            raise AttributeError(
                f"gamfit.{name} requires the optional dependency {exc.name!r}, "
                "which is not installed"
            ) from exc
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def __dir__() -> list[str]:
    return sorted(set(__all__) | _SUBMODULES)
