"""Public exception hierarchy for gamfit.

Every gamfit exception is defined in Rust via ``pyo3::create_exception!``
(see ``crates/gam-pyffi/src/lib.rs``) and re-exported here under its public
``gamfit.*`` name. The class identity caught by user code with
``except gamfit.errors.RemlConvergenceError`` is exactly the same Python type
object that the Rust extension constructs via ``RemlConvergenceError::new_err``;
there is no parallel Python-defined class shadowing the Rust one.

Architecture (issue #343):

* The Rust engine has rich ``thiserror``-typed error enums.
  ``EstimationError`` (in ``src/solver/estimate.rs``) has ~20 variants;
  ``crates/gam-pyffi/src/lib.rs::estimation_error_to_pyerr`` dispatches
  each variant to its corresponding subclass below. No ``err.to_string()``
  flattening, no message-regex reclassification.
* ``GamError`` inherits from :class:`ValueError`, so callers can catch either
  the package umbrella or Python's standard value-contract exception.
* The remaining ``Result<_, String>`` error paths at the FFI boundary
  (formula validation, schema-mismatch during predict, basis builders
  not wrapped in ``EstimationError``, etc.) still flow through
  :func:`map_exception`. They will be migrated to typed dispatch one enum at a
  time; until then,
  NEW error variants for already-typed enums MUST extend the Rust-side
  dispatcher, never the regex classifier.
"""

from __future__ import annotations

from ._binding import RustExtensionUnavailableError, rust_module

# Pull every gamfit exception class out of the Rust extension. This
# happens at import time so the public ``gamfit.errors.GamError`` name is the
# same type object as ``gam._rust.GamError``.
_rust = rust_module()

GamError = _rust.GamError
FormulaError = _rust.FormulaError
# `ColumnNotFoundError` subclasses `FormulaError` (referencing a missing
# column is a formula authoring error). Instances carry structured
# attributes set by the Rust FFI boundary at raise time — `column` (str),
# `role` (Optional[str]), `available` (list[str]), `similar` (list[str]),
# `tsv_hint` (bool) — so `explain_error(...)` and any other consumer can
# read the failure context without parsing the formatted message.
ColumnNotFoundError = _rust.ColumnNotFoundError
SchemaMismatchError = _rust.SchemaMismatchError
PredictionError = _rust.PredictionError

# EstimationError variant subclasses. Each one corresponds to exactly
# one variant of ``gam::estimate::EstimationError``; the Rust side
# selects the right class via ``estimation_error_to_pyerr``.
BasisError = _rust.BasisError
LinearSystemSolveError = _rust.LinearSystemSolveError
EigendecompositionError = _rust.EigendecompositionError
PenaltySpectrumError = _rust.PenaltySpectrumError
ParameterConstraintError = _rust.ParameterConstraintError
PirlsConvergenceError = _rust.PirlsConvergenceError
PerfectSeparationError = _rust.PerfectSeparationError
HessianNotPositiveDefiniteError = _rust.HessianNotPositiveDefiniteError
RemlConvergenceError = _rust.RemlConvergenceError
DictionaryConvergenceError = _rust.DictionaryConvergenceError
GradientUnavailableError = _rust.GradientUnavailableError
LayoutError = _rust.LayoutError
ModelOverparameterizedError = _rust.ModelOverparameterizedError
IllConditionedError = _rust.IllConditionedError
InvalidInputError = _rust.InvalidInputError
MonotoneRootError = _rust.MonotoneRootError
CalibratorError = _rust.CalibratorError
InvalidSpecificationError = _rust.InvalidSpecificationError

# Remaining engine error enum subclasses (issue #343 follow-up). Each one
# corresponds to a `pub enum *Error` in `src/`; the Rust side selects the
# right class via the per-enum `*_error_to_pyerr` dispatcher.
GeometryError = _rust.GeometryError
MatrixMaterializationError = _rust.MatrixMaterializationError
GpuError = _rust.GpuError
LinearAlgebraError = _rust.LinearAlgebraError
MatrixError = _rust.MatrixError
CacheStoreError = _rust.CacheStoreError
SmoothError = _rust.SmoothError
ArrowSchurError = _rust.ArrowSchurError
OuterStrategyError = _rust.OuterStrategyError
TermBuilderError = _rust.TermBuilderError
CorrectedCovarianceError = _rust.CorrectedCovarianceError
PredictInputError = _rust.PredictInputError
HmcError = _rust.HmcError
AloError = _rust.AloError
SurvivalError = _rust.SurvivalError
CubicCellKernelError = _rust.CubicCellKernelError
SurvivalConstructionError = _rust.SurvivalConstructionError
TransformationNormalError = _rust.TransformationNormalError
CustomFamilyError = _rust.CustomFamilyError
GamlssError = _rust.GamlssError
SurvivalMarginalSlopeError = _rust.SurvivalMarginalSlopeError
LatentSurvivalError = _rust.LatentSurvivalError
SurvivalPredictError = _rust.SurvivalPredictError
DeviationRuntimeError = _rust.DeviationRuntimeError
DataError = _rust.DataError
FittedModelError = _rust.FittedModelError
LognormalKernelError = _rust.LognormalKernelError
ScaleDesignError = _rust.ScaleDesignError
IdentifiabilityCompilerError = _rust.IdentifiabilityCompilerError
JointPenaltyError = _rust.JointPenaltyError
SurvivalLocationScaleError = _rust.SurvivalLocationScaleError
MapUniquenessError = _rust.MapUniquenessError
UnsupportedLinkError = _rust.UnsupportedLinkError
InvalidConfigurationError = _rust.InvalidConfigurationError
MissingDependencyError = _rust.MissingDependencyError

# Fit-failure categories (#2937). A failure of a fit's solve raises the class of
# its category; instances carry `variant`, `category`, `causes` and `fields`. `FitError`
# itself is a failure with no category to claim. `PirlsConvergenceError`,
# `RemlConvergenceError` and `InnerModeConvergenceError` (gam#2943) are
# `FitConvergenceError` subclasses, and `IntegrationError` is now raised only for
# genuine quadrature failures.
FitError = _rust.FitError
FitConvergenceError = _rust.FitConvergenceError
InnerModeConvergenceError = _rust.InnerModeConvergenceError
FitSeedError = _rust.FitSeedError
FitInvariantError = _rust.FitInvariantError
FitInputError = _rust.FitInputError
FitNumericalError = _rust.FitNumericalError
IntegrationError = _rust.IntegrationError


def map_exception(exc: BaseException) -> BaseException:
    """Normalize an exception caught at the gamfit Python boundary.

    Typed errors raised by the Rust extension (any subclass of
    :class:`GamError`) pass through unchanged — the FFI boundary already
    selected the correct subclass via variant dispatch
    (``estimation_error_to_pyerr``, ``workflow_error_to_pyerr``,
    ``geometry_error_to_pyerr``, etc. in ``crates/gam-pyffi/src/lib.rs``),
    so there is nothing to reclassify. The message-regex classifier is gone
    (issue #343); structured engine errors no longer
    round-trip through stringly-typed text.

    ``TypeError`` / ``LookupError`` / ``ArithmeticError`` describe
    Python-native contract violations rather than gamfit engine errors,
    so they pass through unwrapped. Every other ``ValueError`` is
    promoted to :class:`GamError` to preserve the documented
    ``except gamfit.errors.GamError`` umbrella from issue #330 — this is a
    type-hierarchy widening, never a narrowing, because ``GamError``
    inherits from ``ValueError``.
    """
    if isinstance(exc, RustExtensionUnavailableError):
        return exc
    if isinstance(exc, GamError):
        return exc
    if isinstance(exc, (TypeError, LookupError, ArithmeticError)):
        return exc
    if isinstance(exc, ValueError):
        return GamError(str(exc))
    return exc
