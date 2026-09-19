"""Exceptions and warnings raised by gamfit.

Every exception is a Rust-defined type re-exported here under one public
name. The engine classifies every failure into one ``ErrorCategory``, and the
class raised sits under that category's base: :class:`FormulaError`,
:class:`DataError`, :class:`ConvergenceError`, :class:`NotFittedError` or
:class:`InternalError`. :class:`GamfitError` is the umbrella (it is not a
``ValueError``): ``except gamfit.errors.GamfitError`` catches every error the
engine raises. See ``docs/exceptions.md`` for the hierarchy.
"""

from __future__ import annotations

from ._binding import (
    RustExtensionUnavailableError,
)
from ._warnings import (
    GamInferenceWarning,
)
from ._exceptions import (
    BasisError,
    CalibratorError,
    ColumnNotFoundError,
    ConvergenceError,
    DataError,
    DictionaryConvergenceError,
    EigendecompositionError,
    FitConvergenceError,
    FitInputError,
    FitInvariantError,
    FitNumericalError,
    FitSeedError,
    FormulaError,
    GamfitError,
    GeometryError,
    GradientUnavailableError,
    HessianNotPositiveDefiniteError,
    IllConditionedError,
    InnerModeConvergenceError,
    IntegrationError,
    InternalError,
    InvalidConfigurationError,
    InvalidInputError,
    InvalidSpecificationError,
    LayoutError,
    LinearSystemSolveError,
    MissingDependencyError,
    ModelOverparameterizedError,
    MonotoneRootError,
    NotFittedError,
    ParameterConstraintError,
    PenaltySpectrumError,
    PerfectSeparationError,
    PirlsConvergenceError,
    PredictInputError,
    PredictionError,
    RemlConvergenceError,
    SchemaMismatchError,
)

__all__ = [
    "BasisError",
    "CalibratorError",
    "ColumnNotFoundError",
    "ConvergenceError",
    "DataError",
    "DictionaryConvergenceError",
    "EigendecompositionError",
    "FitConvergenceError",
    "FitInputError",
    "FitInvariantError",
    "FitNumericalError",
    "FitSeedError",
    "FormulaError",
    "GamInferenceWarning",
    "GamfitError",
    "GeometryError",
    "GradientUnavailableError",
    "HessianNotPositiveDefiniteError",
    "IllConditionedError",
    "InnerModeConvergenceError",
    "IntegrationError",
    "InternalError",
    "InvalidConfigurationError",
    "InvalidInputError",
    "InvalidSpecificationError",
    "LayoutError",
    "LinearSystemSolveError",
    "MissingDependencyError",
    "ModelOverparameterizedError",
    "MonotoneRootError",
    "NotFittedError",
    "ParameterConstraintError",
    "PenaltySpectrumError",
    "PerfectSeparationError",
    "PirlsConvergenceError",
    "PredictInputError",
    "PredictionError",
    "RemlConvergenceError",
    "RustExtensionUnavailableError",
    "SchemaMismatchError",
]
