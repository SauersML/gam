"""Public exception hierarchy for gamfit.

Every gamfit exception is defined in Rust (``crates/gam-pyffi/src/ffi/ffi_errors.rs``)
and re-exported by ``gamfit.errors`` under its public name. The class caught by
``except gamfit.errors.RemlConvergenceError`` is the same type object the Rust extension
constructs; there is no parallel Python-defined class.

The engine classifies every failure into one ``ErrorCategory``
(``crates/gam-spec/src/error_category.rs``). The FFI boundary raises a class
under the category's base, and the CLI exits with the category's code, so both
front ends classify a failure identically without reading its message::

    GamfitError(Exception)
    ├── FormulaError(GamfitError, ValueError)      the request: formula, option, column name
    ├── DataError(GamfitError, ValueError)         the data cannot support the request
    ├── ConvergenceError(GamfitError, RuntimeError) a valid problem whose solve did not finish
    ├── NotFittedError(GamfitError, ValueError, AttributeError)
    └── InternalError(GamfitError, RuntimeError)   an engine defect; please report it

``NotFittedError`` has exactly the bases of ``sklearn.exceptions.NotFittedError``,
so ``except ValueError`` / ``except AttributeError`` handlers written for scikit-learn
catch it, without gamfit importing scikit-learn.
"""

from __future__ import annotations

from ._binding import RustExtensionUnavailableError, rust_module

_rust = rust_module()

# The five category bases.
GamfitError = _rust.GamfitError
FormulaError = _rust.FormulaError
DataError = _rust.DataError
ConvergenceError = _rust.ConvergenceError
NotFittedError = _rust.NotFittedError
InternalError = _rust.InternalError

# FormulaError subclasses. `ColumnNotFoundError` instances carry `column`,
# `role`, `available`, `similar` and `tsv_hint` attributes set at raise time.
ColumnNotFoundError = _rust.ColumnNotFoundError
InvalidSpecificationError = _rust.InvalidSpecificationError
InvalidConfigurationError = _rust.InvalidConfigurationError
BasisError = _rust.BasisError
MissingDependencyError = _rust.MissingDependencyError

# DataError subclasses.
SchemaMismatchError = _rust.SchemaMismatchError
PredictionError = _rust.PredictionError
PredictInputError = _rust.PredictInputError
PerfectSeparationError = _rust.PerfectSeparationError
ModelOverparameterizedError = _rust.ModelOverparameterizedError
IllConditionedError = _rust.IllConditionedError
InvalidInputError = _rust.InvalidInputError
GeometryError = _rust.GeometryError
FitInputError = _rust.FitInputError

# ConvergenceError subclasses. A fit's solve failure raises the class of its
# fit category (#2937); instances carry `variant`, `category`, `error_category`,
# `causes` and `fields`.
FitConvergenceError = _rust.FitConvergenceError
PirlsConvergenceError = _rust.PirlsConvergenceError
RemlConvergenceError = _rust.RemlConvergenceError
InnerModeConvergenceError = _rust.InnerModeConvergenceError
FitSeedError = _rust.FitSeedError
FitNumericalError = _rust.FitNumericalError
LinearSystemSolveError = _rust.LinearSystemSolveError
EigendecompositionError = _rust.EigendecompositionError
PenaltySpectrumError = _rust.PenaltySpectrumError
ParameterConstraintError = _rust.ParameterConstraintError
HessianNotPositiveDefiniteError = _rust.HessianNotPositiveDefiniteError
MonotoneRootError = _rust.MonotoneRootError
IntegrationError = _rust.IntegrationError
CalibratorError = _rust.CalibratorError
DictionaryConvergenceError = _rust.DictionaryConvergenceError

# InternalError subclasses.
FitInvariantError = _rust.FitInvariantError
GradientUnavailableError = _rust.GradientUnavailableError
LayoutError = _rust.LayoutError


def map_exception(exc: BaseException) -> BaseException:
    """Normalize an exception caught at the gamfit Python boundary.

    Errors raised by the Rust extension (any :class:`GamfitError`) pass through
    unchanged: the FFI boundary already chose the class from the engine's
    ``ErrorCategory``, so there is nothing to reclassify.

    ``TypeError`` / ``LookupError`` / ``ArithmeticError`` describe Python-native
    contract violations and pass through unwrapped. A remaining ``ValueError``
    comes from argument validation in the Python layer, which is a request the
    caller got wrong, so it becomes :class:`FormulaError` (still a
    ``ValueError``, so existing handlers keep catching it).
    """
    if isinstance(exc, (RustExtensionUnavailableError, GamfitError)):
        return exc
    if isinstance(exc, (TypeError, LookupError, ArithmeticError)):
        return exc
    if isinstance(exc, ValueError):
        return FormulaError(str(exc))
    return exc
