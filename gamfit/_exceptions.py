"""Public exception hierarchy for gamfit.

Every gamfit exception is defined in Rust via ``pyo3::create_exception!``
(see ``crates/gam-pyffi/src/lib.rs``) and re-exported here under its public
``gamfit.*`` name. The class identity caught by user code with
``except gamfit.RemlConvergenceError`` is exactly the same Python type
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
# happens at import time so the public ``gamfit.GamError`` name is the
# same type object as ``gam._rust.GamError``.
_rust = rust_module()

GamError: type = _rust.GamError
FormulaError: type = _rust.FormulaError
# `ColumnNotFoundError` subclasses `FormulaError` (referencing a missing
# column is a formula authoring error). Instances carry structured
# attributes set by the Rust FFI boundary at raise time — `column` (str),
# `role` (Optional[str]), `available` (list[str]), `similar` (list[str]),
# `tsv_hint` (bool) — so `explain_error(...)` and any other consumer can
# read the failure context without parsing the formatted message.
ColumnNotFoundError: type = _rust.ColumnNotFoundError
SchemaMismatchError: type = _rust.SchemaMismatchError
PredictionError: type = _rust.PredictionError

# EstimationError variant subclasses. Each one corresponds to exactly
# one variant of ``gam::estimate::EstimationError``; the Rust side
# selects the right class via ``estimation_error_to_pyerr``.
BasisError: type = _rust.BasisError
LinearSystemSolveError: type = _rust.LinearSystemSolveError
EigendecompositionError: type = _rust.EigendecompositionError
PenaltySpectrumError: type = _rust.PenaltySpectrumError
ParameterConstraintError: type = _rust.ParameterConstraintError
PirlsConvergenceError: type = _rust.PirlsConvergenceError
PerfectSeparationError: type = _rust.PerfectSeparationError
HessianNotPositiveDefiniteError: type = _rust.HessianNotPositiveDefiniteError
RemlConvergenceError: type = _rust.RemlConvergenceError
DictionaryConvergenceError: type = _rust.DictionaryConvergenceError
GradientUnavailableError: type = _rust.GradientUnavailableError
LayoutError: type = _rust.LayoutError
ModelOverparameterizedError: type = _rust.ModelOverparameterizedError
IllConditionedError: type = _rust.IllConditionedError
InvalidInputError: type = _rust.InvalidInputError
MonotoneRootError: type = _rust.MonotoneRootError
CalibratorError: type = _rust.CalibratorError
InvalidSpecificationError: type = _rust.InvalidSpecificationError

# Remaining engine error enum subclasses (issue #343 follow-up). Each one
# corresponds to a `pub enum *Error` in `src/`; the Rust side selects the
# right class via the per-enum `*_error_to_pyerr` dispatcher.
GeometryError: type = _rust.GeometryError
MatrixMaterializationError: type = _rust.MatrixMaterializationError
GpuError: type = _rust.GpuError
LinearAlgebraError: type = _rust.LinearAlgebraError
MatrixError: type = _rust.MatrixError
CacheStoreError: type = _rust.CacheStoreError
SmoothError: type = _rust.SmoothError
ArrowSchurError: type = _rust.ArrowSchurError
OuterStrategyError: type = _rust.OuterStrategyError
TermBuilderError: type = _rust.TermBuilderError
CorrectedCovarianceError: type = _rust.CorrectedCovarianceError
PredictInputError: type = _rust.PredictInputError
HmcError: type = _rust.HmcError
AloError: type = _rust.AloError
SurvivalError: type = _rust.SurvivalError
CubicCellKernelError: type = _rust.CubicCellKernelError
SurvivalConstructionError: type = _rust.SurvivalConstructionError
TransformationNormalError: type = _rust.TransformationNormalError
CustomFamilyError: type = _rust.CustomFamilyError
GamlssError: type = _rust.GamlssError
SurvivalMarginalSlopeError: type = _rust.SurvivalMarginalSlopeError
LatentSurvivalError: type = _rust.LatentSurvivalError
SurvivalPredictError: type = _rust.SurvivalPredictError
DeviationRuntimeError: type = _rust.DeviationRuntimeError
DataError: type = _rust.DataError
FittedModelError: type = _rust.FittedModelError
LognormalKernelError: type = _rust.LognormalKernelError
ScaleDesignError: type = _rust.ScaleDesignError
IdentifiabilityCompilerError: type = _rust.IdentifiabilityCompilerError
JointPenaltyError: type = _rust.JointPenaltyError
SurvivalLocationScaleError: type = _rust.SurvivalLocationScaleError
MapUniquenessError: type = _rust.MapUniquenessError
UnsupportedLinkError: type = _rust.UnsupportedLinkError
InvalidConfigurationError: type = _rust.InvalidConfigurationError
MissingDependencyError: type = _rust.MissingDependencyError
IntegrationError: type = _rust.IntegrationError


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
    ``except gamfit.GamError`` umbrella from issue #330 — this is a
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
