class GamError(Exception):
    """Base class for Python-facing GAM errors."""


class FormulaError(GamError):
    """The formula is invalid or unsupported."""


class SchemaMismatchError(GamError):
    """Prediction input does not match the training schema."""


class PredictionError(GamError):
    """Prediction failed."""


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
