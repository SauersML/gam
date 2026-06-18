from __future__ import annotations

import importlib
from collections.abc import Mapping, Sequence
from typing import Any, cast

SUPPORTED_OUTPUT_KINDS = {"dict", "numpy", "pandas", "polars", "pyarrow"}


class PredictionResult(dict[str, list[Any]]):
    """Dict-shaped prediction table with attribute access to columns.

    ``Model.predict(..., return_type="dict")`` and dict-shaped tabular
    defaults return this class. It behaves like a normal ``dict`` for
    subscription and iteration, while also allowing field access such as
    ``pred.mean``, ``pred.std_error``, and ``pred.mean_lower``.
    """

    _ALIASES = {
        "lower": "mean_lower",
        "upper": "mean_upper",
        "se_mean": "std_error",
    }

    def __getattr__(self, name: str) -> list[Any]:
        key = self._ALIASES.get(name, name)
        try:
            return self[key]
        except KeyError as exc:
            raise AttributeError(
                f"{type(self).__name__!s} has no prediction column {name!r}"
            ) from exc


class PreNormalizedTable:
    """A table already normalized to ``(headers, rows, kind)`` form.

    ``normalize_table`` stringifies every cell of the input — an
    ``O(n_rows * n_cols)`` pass that is invariant to the formula. Callers that
    fit the *same* table many times (e.g. topology AUTO selection, which refits
    one table across a budget cascade and several candidate topologies, #869)
    can normalize once and wrap the result here; ``normalize_table`` then
    short-circuits and returns the cached ``(headers, rows, kind)`` verbatim
    instead of re-coercing and re-stringifying. ``kind`` is preserved so the
    output-table restoration still reflects the original input library.
    """

    __slots__ = ("headers", "rows", "kind")

    def __init__(self, headers: list[str], rows: list[list[str]], kind: str) -> None:
        self.headers = headers
        self.rows = rows
        self.kind = kind


def _try_import(name: str) -> Any | None:
    try:
        return importlib.import_module(name)
    except ImportError:
        return None


def normalize_table(data: Any) -> tuple[list[str], list[list[str]], str]:
    if isinstance(data, PreNormalizedTable):
        return data.headers, data.rows, data.kind
    columns, kind = table_columns(data)
    headers = list(columns)
    if not headers:
        raise ValueError("table must have at least one column")
    row_count = len(columns[headers[0]])
    if row_count == 0:
        raise ValueError("table data cannot be empty")
    rows = [
        [stringify_cell(columns[header][row_index]) for header in headers]
        for row_index in range(row_count)
    ]
    return headers, rows, kind


def table_columns(data: Any) -> tuple[dict[str, list[Any]], str]:
    kind = detect_table_kind(data)
    if kind == "pandas":
        names = [str(column) for column in data.columns]
        reject_duplicate_column_names(names, kind)
        return (
            {name: data.iloc[:, index].tolist() for index, name in enumerate(names)},
            kind,
        )
    if kind == "polars":
        names = [str(column) for column in data.columns]
        reject_duplicate_column_names(names, kind)
        return (
            {name: data[name].to_list() for name in names},
            kind,
        )
    if kind == "pyarrow":
        names = [str(name) for name in data.column_names]
        reject_duplicate_column_names(names, kind)
        return (
            {name: data.column(index).to_pylist() for index, name in enumerate(names)},
            kind,
        )
    if kind == "numpy":
        return numpy_table_columns(data), kind
    if isinstance(data, Mapping):
        return mapping_table_columns(data), "mapping"
    if isinstance(data, Sequence) and not isinstance(data, (str, bytes, bytearray)):
        rows_like = list(data)
        if not rows_like:
            raise ValueError("table data cannot be empty")
        if isinstance(rows_like[0], Mapping):
            return record_table_columns(cast("list[Mapping[str, Any]]", rows_like)), "records"
        if isinstance(rows_like[0], Sequence) and not isinstance(
            rows_like[0],
            (str, bytes, bytearray),
        ):
            return sequence_table_columns(cast("Sequence[Sequence[Any]]", rows_like)), "rows"
    raise TypeError(
        "unsupported table input; use pandas, pyarrow, numpy, a mapping, a list of records, or a 2D row sequence"
    )


def restore_output_table(
    columns: dict[str, list[Any]],
    *,
    requested: str | None,
    input_kind: str,
    training_kind: str | None,
) -> Any:
    target = requested or preferred_output_kind(input_kind, training_kind)
    if target not in SUPPORTED_OUTPUT_KINDS:
        allowed = ", ".join(sorted(SUPPORTED_OUTPUT_KINDS))
        raise ValueError(
            f"unsupported return_type '{target}'; use one of: {allowed}"
        )
    if target == "dict":
        return PredictionResult(columns)
    if target == "pandas":
        import pandas as pd

        return pd.DataFrame(columns)
    if target == "polars":
        pl = _try_import("polars")
        if pl is None:
            raise ImportError("return_type 'polars' requires the polars package")

        return pl.DataFrame(columns)
    if target == "numpy":
        import numpy as np

        ordered = list(columns)
        return np.column_stack([columns[name] for name in ordered])
    if target == "pyarrow":
        pa = _try_import("pyarrow")
        if pa is None:
            raise ImportError("return_type 'pyarrow' requires the pyarrow package")

        return pa.table(columns)
    raise ValueError(f"unsupported return_type '{target}'")


def preferred_output_kind(input_kind: str, training_kind: str | None) -> str:
    if input_kind in {"pandas", "polars", "numpy", "pyarrow"}:
        return input_kind
    if training_kind in {"pandas", "polars", "numpy", "pyarrow"}:
        return training_kind
    return "dict"


def detect_table_kind(data: Any) -> str:
    if data is None:
        return "unknown"
    pd = _try_import("pandas")
    if pd is not None and isinstance(data, pd.DataFrame):
        return "pandas"

    pl = _try_import("polars")
    if pl is not None and isinstance(data, pl.DataFrame):
        return "polars"

    pa = _try_import("pyarrow")
    if pa is not None and isinstance(data, pa.Table):
        return "pyarrow"

    np = _try_import("numpy")
    if np is not None and isinstance(data, np.ndarray):
        return "numpy"

    return "unknown"


def response_column_name(formula: str) -> str | None:
    if "~" not in formula:
        return None
    candidate = formula.split("~", 1)[0].strip()
    if not candidate or candidate.startswith("Surv("):
        return None
    if candidate.replace("_", "").isalnum():
        return candidate
    return None


def mapping_table_columns(data: Mapping[Any, Any]) -> dict[str, list[Any]]:
    columns = {str(key): vector_values(value) for key, value in data.items()}
    validate_column_lengths(columns)
    return columns


def record_table_columns(rows: list[Mapping[str, Any]]) -> dict[str, list[Any]]:
    headers = collect_record_headers(rows)
    columns: dict[str, list[Any]] = {header: [] for header in headers}
    for row in rows:
        for header in headers:
            if header not in row:
                raise ValueError(f"row is missing key '{header}'")
            columns[header].append(row[header])
    return columns


def sequence_table_columns(rows: Sequence[Sequence[Any]]) -> dict[str, list[Any]]:
    width = len(rows[0])
    if width == 0:
        raise ValueError("row sequences must have at least one column")
    for index, row in enumerate(rows):
        if len(row) != width:
            raise ValueError(
                f"row {index + 1} has width {len(row)} but expected {width}"
            )
    headers = [f"x{index}" for index in range(width)]
    columns: dict[str, list[Any]] = {header: [] for header in headers}
    for row in rows:
        for index, value in enumerate(row):
            columns[headers[index]].append(value)
    return columns


def numpy_table_columns(array: Any) -> dict[str, list[Any]]:
    import numpy as np

    values = np.asarray(array)
    if values.ndim == 1:
        return {"x0": values.tolist()}
    if values.ndim != 2:
        raise ValueError("numpy input must be 1D or 2D")
    headers = [f"x{index}" for index in range(values.shape[1])]
    return {header: values[:, index].tolist() for index, header in enumerate(headers)}


def reject_duplicate_column_names(names: Sequence[str], kind: str) -> None:
    seen: dict[str, int] = {}
    duplicates: list[str] = []
    for name in names:
        seen[name] = seen.get(name, 0) + 1
        if seen[name] == 2:
            duplicates.append(name)
    if duplicates:
        listed = ", ".join(repr(name) for name in duplicates)
        raise ValueError(
            f"{kind} input has duplicate column names ({listed}); "
            "every column must have a unique name"
        )


def validate_column_lengths(columns: Mapping[str, Sequence[Any]]) -> None:
    lengths = {len(values) for values in columns.values()}
    if len(lengths) > 1:
        raise ValueError("all columns must have the same length")


def collect_record_headers(rows: list[Mapping[str, Any]]) -> list[str]:
    headers: list[str] = []
    seen = set()
    for row in rows:
        for key in row.keys():
            key_str = str(key)
            if key_str not in seen:
                seen.add(key_str)
                headers.append(key_str)
    return headers


def stringify_cell(value: Any) -> str:
    if isinstance(value, bool):
        return "1" if value else "0"
    if value is None:
        raise ValueError("table cells cannot be None")
    if isinstance(value, int):
        # bool is handled above; covers Python int and any int subclass
        # (e.g. numpy integers) so the rendered text is always a bare integer.
        return repr(int(value))
    if isinstance(value, float):
        if value != value:
            raise ValueError("table cells cannot be NaN")
        # float subclasses (e.g. numpy.float64) render via repr() with the
        # type name in NumPy 2.x ("np.float64(-3.0)"), which the Rust core
        # cannot parse and so misclassifies the column as categorical.
        # Normalize through the canonical Python float first.
        return repr(float(value))
    text = str(value)
    if not text:
        raise ValueError("table cells cannot be empty strings")
    return text


def coerce_numeric_vector(values: Sequence[Any], *, label: str) -> list[float]:
    numeric: list[float] = []
    for index, value in enumerate(values):
        try:
            numeric_value = float(value)
        except (TypeError, ValueError) as exc:
            raise ValueError(
                f"{label} contains a non-numeric value at position {index + 1}: {value!r}"
            ) from exc
        if numeric_value != numeric_value:
            raise ValueError(f"{label} contains NaN at position {index + 1}")
        numeric.append(numeric_value)
    return numeric


def attach_target(
    data: Any,
    y: Any,
    *,
    target_name: str = "y",
) -> tuple[dict[str, list[Any]], str]:
    columns, kind = table_columns(data)
    if target_name in columns:
        raise ValueError(
            f"target column '{target_name}' already exists in the feature table"
        )
    if isinstance(y, str):
        raise TypeError("string targets must refer to an existing column on the input table")
    target_values = vector_values(y)
    if columns:
        expected = len(next(iter(columns.values())))
        if len(target_values) != expected:
            raise ValueError(
                f"target vector has length {len(target_values)} but expected {expected}"
            )
    columns[target_name] = target_values
    return columns, kind


def vector_values(values: Any) -> list[Any]:
    kind = detect_table_kind(values)
    if kind == "numpy":
        import numpy as np

        array = np.asarray(values)
        if array.ndim != 1:
            raise ValueError("target arrays must be 1D")
        result: list[Any] = array.tolist()
        return result
    if isinstance(values, Mapping):
        raise TypeError("target values must be a vector, not a mapping")
    # pandas/polars/pyarrow columns (and other array-likes) expose tolist /
    # to_list / to_pylist; prefer those so we get native Python scalars rather
    # than library-specific scalar objects whose repr is not numeric-parseable.
    for method_name in ("tolist", "to_list", "to_pylist"):
        method = getattr(values, method_name, None)
        if callable(method) and not isinstance(values, (str, bytes, bytearray)):
            converted = method()
            if isinstance(converted, list):
                return converted
    if isinstance(values, Sequence) and not isinstance(values, (str, bytes, bytearray)):
        return list(values)
    raise TypeError("target values must be a 1D array-like sequence")
