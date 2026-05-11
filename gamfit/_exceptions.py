class GamError(Exception):
    """Base class for Python-facing GAM errors.

    All gamfit-specific exceptions raised by the Python binding inherit from
    ``GamError``, so catching this class is the broadest way to handle a
    failure originating from the Rust engine or the binding layer.

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
    from ._binding import RustExtensionUnavailableError

    if isinstance(exc, RustExtensionUnavailableError):
        return exc
    message = str(exc)
    lower = message.lower()
    if "formula" in lower or "parse" in lower:
        return FormulaError(message)
    if "schema" in lower or "missing required column" in lower or "unknown column" in lower:
        return SchemaMismatchError(message)
    if "prediction" in lower or "predict" in lower:
        return PredictionError(message)
    return GamError(message)
