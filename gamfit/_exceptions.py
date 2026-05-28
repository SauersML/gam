class GamError(ValueError):
    """Base class for Python-facing GAM errors.

    All gamfit-specific exceptions raised by the Python binding inherit from
    ``GamError``, so catching this class is the broadest way to handle a
    failure originating from the Rust engine or the binding layer.

    ``GamError`` is a subclass of :class:`ValueError`. Engine-side failures
    (REML convergence, ill-conditioning, degenerate inner iterates, response
    geometry domain violations) historically surfaced as bare ``ValueError``
    because the Rust extension raises ``PyValueError`` for every domain-style
    error. Inheriting from ``ValueError`` keeps that contract intact —
    ``except ValueError`` still catches them — while ``except GamError`` now
    works as the documented broad catch for any gamfit engine error.

    Examples
    --------
    >>> try:
    ...     gamfit.fit(df, "y ~ s(x)")
    ... except gamfit.GamError as exc:
    ...     print(gamfit.explain_error(exc))
    """


class FormulaError(GamError):
    """The formula is invalid or unsupported.

    Raised when the Wilkinson-style formula string cannot be parsed, references
    columns missing from the input table, or describes a model the engine does
    not support.

    Examples
    --------
    >>> try:
    ...     gamfit.fit(df, "y ~ s(nope)")
    ... except gamfit.FormulaError as exc:
    ...     print(exc)
    """


class SchemaMismatchError(GamError):
    """Prediction input does not match the training schema.

    Raised when the table passed to :meth:`Model.predict` or related methods
    lacks columns that were present at fit time, has incompatible dtypes, or
    introduces unknown categorical levels.

    Examples
    --------
    >>> try:
    ...     model.predict(serving_df)
    ... except gamfit.SchemaMismatchError as exc:
    ...     print(model.check(serving_df))
    """


class PredictionError(GamError):
    """Prediction failed.

    Raised for runtime failures during prediction that are not pure schema
    problems (numerical issues, unsupported prediction modes for the fitted
    model, etc.).

    Examples
    --------
    >>> try:
    ...     model.predict(test_df)
    ... except gamfit.PredictionError as exc:
    ...     print(gamfit.explain_error(exc))
    """


def map_exception(exc: BaseException) -> BaseException:
    from ._binding import RustExtensionUnavailableError, rust_module

    if isinstance(exc, RustExtensionUnavailableError):
        return exc
    if isinstance(exc, GamError):
        return exc
    message = str(exc)
    kind = rust_module().classify_exception_message(message)
    if kind == "formula":
        return FormulaError(message)
    if kind == "schema_mismatch":
        return SchemaMismatchError(message)
    if kind == "prediction":
        return PredictionError(message)
    # No formula/schema/prediction classification matched. The Rust extension
    # raises ``PyValueError`` for both engine-side failures (REML convergence,
    # numerical breakdown, degenerate inner iterates) and caller-input domain
    # errors (response-geometry antipodal sphere log, non-positive simplex
    # mass, mismatched base-point dimensions). Both belong under the documented
    # ``except gamfit.GamError`` umbrella — see issue #330 — so we promote
    # ``ValueError`` to ``GamError`` here. Because ``GamError`` inherits from
    # ``ValueError``, callers using ``except ValueError`` continue to catch
    # these errors unchanged: the type-hierarchy widens, it does not narrow.
    #
    # ``TypeError`` / ``LookupError`` / ``ArithmeticError`` describe genuine
    # Python-native contract violations (shape/dtype mismatch, missing key,
    # division-by-zero) that are not gamfit engine errors, so they pass
    # through untouched.
    if isinstance(exc, ValueError):
        return GamError(message)
    if isinstance(exc, (TypeError, LookupError, ArithmeticError)):
        return exc
    return GamError(message)
