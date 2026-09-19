"""Public exception hierarchy for gamfit.

Every gamfit exception is defined in Rust (``crates/gam-pyffi/src/ffi/ffi_errors.rs``)
and re-exported here under its public ``gamfit.*`` name. The class caught by
``except gamfit.RemlConvergenceError`` is the same type object the Rust extension
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
GamfitError: type = _rust.GamfitError
FormulaError: type = _rust.FormulaError
DataError: type = _rust.DataError
ConvergenceError: type = _rust.ConvergenceError
NotFittedError: type = _rust.NotFittedError
InternalError: type = _rust.InternalError

# FormulaError subclasses. `ColumnNotFoundError` instances carry `column`,
# `role`, `available`, `similar` and `tsv_hint` attributes set at raise time.
ColumnNotFoundError: type = _rust.ColumnNotFoundError
InvalidSpecificationError: type = _rust.InvalidSpecificationError
InvalidConfigurationError: type = _rust.InvalidConfigurationError
BasisError: type = _rust.BasisError
MissingDependencyError: type = _rust.MissingDependencyError

# DataError subclasses.
SchemaMismatchError: type = _rust.SchemaMismatchError
PredictionError: type = _rust.PredictionError
PerfectSeparationError: type = _rust.PerfectSeparationError
ModelOverparameterizedError: type = _rust.ModelOverparameterizedError
IllConditionedError: type = _rust.IllConditionedError
InvalidInputError: type = _rust.InvalidInputError
GeometryError: type = _rust.GeometryError
FitInputError: type = _rust.FitInputError

# ConvergenceError subclasses. A fit's solve failure raises the class of its
# fit category (#2937); instances carry `variant`, `category`, `error_category`,
# `causes` and `fields`.
FitConvergenceError: type = _rust.FitConvergenceError
PirlsConvergenceError: type = _rust.PirlsConvergenceError
RemlConvergenceError: type = _rust.RemlConvergenceError
InnerModeConvergenceError: type = _rust.InnerModeConvergenceError
FitSeedError: type = _rust.FitSeedError
FitNumericalError: type = _rust.FitNumericalError
LinearSystemSolveError: type = _rust.LinearSystemSolveError
EigendecompositionError: type = _rust.EigendecompositionError
PenaltySpectrumError: type = _rust.PenaltySpectrumError
ParameterConstraintError: type = _rust.ParameterConstraintError
HessianNotPositiveDefiniteError: type = _rust.HessianNotPositiveDefiniteError
MonotoneRootError: type = _rust.MonotoneRootError
IntegrationError: type = _rust.IntegrationError
CalibratorError: type = _rust.CalibratorError
DictionaryConvergenceError: type = _rust.DictionaryConvergenceError

# InternalError subclasses.
FitInvariantError: type = _rust.FitInvariantError
GradientUnavailableError: type = _rust.GradientUnavailableError
LayoutError: type = _rust.LayoutError


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
