from __future__ import annotations

import importlib
import math
import sys
from collections.abc import Mapping, Sequence
from typing import TYPE_CHECKING, Any, cast

if TYPE_CHECKING:
    from ._rust import _EncodedTable

SUPPORTED_OUTPUT_KINDS = {"dict", "numpy", "pandas", "polars", "pyarrow"}


class PredictionResult(dict[str, Any]):
    """Dict-shaped prediction table with attribute access to columns.

    ``Model.predict(..., return_type="dict")`` and dict-shaped tabular
    defaults return this class. It behaves like a normal ``dict`` for
    subscription and iteration, while also allowing field access such as
    ``pred.posterior_mean``, ``pred.posterior_mean_standard_error``, and
    ``pred.posterior_mean_lower``.
    Model-based interval results also carry the scalar metadata field
    ``pred.covariance_source`` / ``pred["covariance_source"]``.
    """

    def __getattr__(self, name: str) -> Any:
        try:
            return self[name]
        except KeyError as exc:
            raise AttributeError(
                f"{type(self).__name__!s} has no prediction column {name!r}"
            ) from exc


class PreNormalizedTable:
    """A table already normalized to ``(headers, native_table, kind)`` form.

    The native table is the Rust-owned, typed encoding produced by
    :func:`normalize_table`.  Reusing it across topology candidates avoids both
    reparsing and another dense numeric copy; categorical labels are retained
    only once in the Rust schema rather than expanded into a Python string per
    cell. ``kind`` is preserved so output restoration still reflects the input
    library.
    """

    __slots__ = ("headers", "rows", "kind")

    def __init__(self, headers: list[str], rows: _EncodedTable, kind: str) -> None:
        self.headers = headers
        self.rows = rows
        self.kind = kind


# Mirror of gam-data's `CATEGORICAL_CELL_SENTINEL`: a rendered row of an encoded
# table prefixes each categorical cell with it, so a level whose label parses as
# a number ("0", "1") keeps its categorical source intent when the row is
# re-read by the string-row entry points (#1317). A leading NUL never appears in
# a numeric literal and is stripped before any level matching.
CATEGORICAL_CELL_SENTINEL = "\x00"


def normalize_table(
    data: Any, *, required_columns: Sequence[str] | None = None
) -> tuple[list[str], _EncodedTable, str]:
    """Encode ``data`` as a Rust-owned typed table.

    This is a transport adapter only. Polars and PyArrow tables hand Rust an
    Arrow C stream. Every other input crosses column by column in the layout its
    source declares: a NumPy numeric vector as ``float64``, a pandas categorical
    as codes plus levels, and anything else as the raw cell values. gam-data
    decides what every column means, so all input libraries share one set of
    inference rules and one set of ``DataError`` messages. pandas needs no
    pyarrow for any of this.
    """
    if isinstance(data, PreNormalizedTable):
        return data.headers, data.rows, data.kind
    columns, kind = _table_column_views(data)
    if required_columns is not None:
        names = list(required_columns)
        missing = set(names) - set(columns)
        if missing:
            raise ValueError(f"missing required columns: {sorted(missing)}")
        columns = {name: columns[name] for name in names}
        if kind in _ARROW_TABLE_KINDS:
            data = data.select(names)
    headers = list(columns)
    if not headers:
        from ._exceptions import DataError
        raise DataError("column '<table>' has no columns")
    reject_duplicate_column_names(headers, kind)
    validate_column_lengths(columns)
    if len(columns[headers[0]]) == 0:
        from ._exceptions import DataError
        raise DataError("column '<table>' has no observations")

    from ._binding import rust_module

    if kind in _ARROW_TABLE_KINDS:
        return headers, rust_module().encoded_table_from_arrow(headers, data), kind
    native = rust_module().encoded_table_from_columns(
        headers, [_column_payload(header, columns[header]) for header in headers]
    )
    return headers, native, kind


# Table kinds whose frames export the Arrow C stream themselves (Polars natively,
# PyArrow by definition), so Rust reads their buffers without a Python copy.
_ARROW_TABLE_KINDS = frozenset({"polars", "pyarrow"})

# NumPy dtype kinds that are numbers: bool, signed and unsigned int, float.
_NUMERIC_DTYPE_KINDS = "biuf"
# NumPy dtype kinds whose cells cross as raw values: object, str, bytes.
_CELL_DTYPE_KINDS = "OUS"


def _column_payload(name: str, values: Any) -> Any:
    """One column in the layout ``encoded_table_from_columns`` accepts.

    Returns a ``float64`` ndarray for a declared numeric column, a
    ``(codes, levels)`` tuple for a pandas categorical, or a list or object
    array of raw values for everything else. Reads declared dtypes only; a
    declared dtype that is neither numbers nor cells (datetime, timedelta,
    complex) is refused for the whole column, as the Arrow reader refuses an
    unsupported Arrow type, because its values would cast to meaningless floats.
    """
    import numpy as np

    pd = sys.modules.get("pandas")
    if pd is not None and isinstance(values, pd.Series):
        dtype = values.dtype
        if isinstance(dtype, pd.CategoricalDtype):
            return (
                np.asarray(values.cat.codes, dtype=np.int64),
                list(dtype.categories),
            )
        _reject_unsupported_dtype(name, dtype)
        if dtype.kind in _NUMERIC_DTYPE_KINDS:
            # Nullable extension dtypes (Int64, boolean, Float64) map NA to NaN.
            return values.to_numpy(dtype=np.float64, na_value=np.nan)
        return values.to_numpy(dtype=object, na_value=None)
    if isinstance(values, np.ndarray) or (
        hasattr(values, "__array__") and not isinstance(values, (list, tuple))
    ):
        array = np.asarray(values)
        _reject_unsupported_dtype(name, array.dtype)
        if array.dtype.kind in _NUMERIC_DTYPE_KINDS:
            return np.ascontiguousarray(array, dtype=np.float64)
        return array.astype(object, copy=False)
    # A plain sequence is passed as a list, never through ``np.asarray``, which
    # would render a NaN beside a string as the text 'nan'.
    return list(values)


def _reject_unsupported_dtype(name: str, dtype: Any) -> None:
    if dtype.kind not in _NUMERIC_DTYPE_KINDS + _CELL_DTYPE_KINDS:
        from ._exceptions import DataError

        raise DataError(
            f"unsupported column dtype {dtype} for column '{name}'; "
            "columns must hold numbers, strings or categories"
        )


def _table_column_views(data: Any) -> tuple[dict[str, Any], str]:
    """Return zero-copy/lazy column views for the primary Rust table boundary."""
    kind = detect_table_kind(data)
    if kind == "pandas":
        names = [str(column) for column in data.columns]
        reject_duplicate_column_names(names, kind)
        return {name: data.iloc[:, index] for index, name in enumerate(names)}, kind
    if kind == "polars":
        names = [str(column) for column in data.columns]
        reject_duplicate_column_names(names, kind)
        return {name: data[name] for name in names}, kind
    if kind == "pyarrow":
        names = [str(name) for name in data.column_names]
        reject_duplicate_column_names(names, kind)
        return {name: data.column(index) for index, name in enumerate(names)}, kind
    if kind == "numpy":
        import numpy as np

        values = np.asarray(data)
        if values.ndim == 1:
            return {"x0": values}, kind
        if values.ndim != 2:
            raise ValueError("numpy input must be 1D or 2D")
        return {
            f"x{index}": values[:, index] for index in range(values.shape[1])
        }, kind
    if isinstance(data, Mapping):
        columns: dict[str, Any] = {}
        for key, value in data.items():
            name = str(key)
            if name in columns:
                raise ValueError(
                    f"key collision: original key {key!r} normalizes to {name!r}, "
                    "which is already used"
                )
            columns[name] = _vector_view(value)
        validate_column_lengths(columns)
        return columns, "mapping"
    # Record and row inputs are Python objects already; their existing
    # columnisation is the minimal representation.
    return table_columns(data)


def _vector_view(values: Any) -> Any:
    if isinstance(values, Mapping):
        raise TypeError("target values must be a vector, not a mapping")
    if isinstance(values, (str, bytes, bytearray)):
        raise TypeError("target values must be a 1D array-like sequence")
    ndim = getattr(values, "ndim", None)
    if ndim is not None:
        if int(ndim) != 1:
            raise ValueError("target arrays must be 1D")
        return values
    if isinstance(values, Sequence):
        return values
    # pandas/Polars/Arrow vector objects are sized and indexable without being
    # registered as collections.abc.Sequence.
    if hasattr(values, "__len__") and hasattr(values, "__getitem__"):
        return values
    raise TypeError("target values must be a 1D array-like sequence")


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
    training_kind: str,
) -> Any:
    target = requested or preferred_output_kind(input_kind, training_kind)
    if target not in SUPPORTED_OUTPUT_KINDS:
        allowed = ", ".join(sorted(SUPPORTED_OUTPUT_KINDS))
        raise ValueError(
            f"unsupported return_type '{target}'; use one of: {allowed}"
        )
    if target == "dict":
        return PredictionResult(columns)
    if target == "numpy":
        import numpy as np

        ordered = list(columns)
        return np.column_stack([columns[name] for name in ordered])
    library = _import_output_library(target)
    if target == "pyarrow":
        return library.table(columns)
    return library.DataFrame(columns)


def _import_output_library(name: str) -> Any:
    """Import the table library an output kind names; only an explicit output
    request (or an input already of that kind) reaches here."""
    try:
        return importlib.import_module(name)
    except ModuleNotFoundError as exc:
        if exc.name != name:
            raise
        raise ImportError(f"return_type '{name}' requires the {name} package") from exc


def preferred_output_kind(input_kind: str, training_kind: str) -> str:
    if input_kind in {"pandas", "polars", "numpy", "pyarrow"}:
        return input_kind
    if training_kind in {"pandas", "polars", "numpy", "pyarrow"}:
        return training_kind
    return "dict"


# Table kind (the library's module name) -> its table type. A value can only be
# an instance of a library's table type once that library is imported, so
# detection reads ``sys.modules`` and never imports anything.
_TABLE_TYPES = (
    ("pandas", "DataFrame"),
    ("polars", "DataFrame"),
    ("pyarrow", "Table"),
    ("numpy", "ndarray"),
)


def detect_table_kind(data: Any) -> str:
    for kind, type_name in _TABLE_TYPES:
        table_type = getattr(sys.modules.get(kind), type_name, None)
        if isinstance(table_type, type) and isinstance(data, table_type):
            return kind
    return "unknown"


def mapping_table_columns(data: Mapping[Any, Any]) -> dict[str, list[Any]]:
    columns: dict[str, list[Any]] = {}
    for key, value in data.items():
        key_str = str(key)
        if key_str in columns:
            raise ValueError(f"key collision: original key {key!r} normalizes to {key_str!r}, which is already used")
        columns[key_str] = vector_values(value)
    validate_column_lengths(columns)
    return columns


def record_table_columns(rows: list[Mapping[str, Any]]) -> dict[str, list[Any]]:
    headers, key_map = collect_record_headers(rows)
    columns: dict[str, list[Any]] = {header: [] for header in headers}
    for row_idx, row in enumerate(rows):
        for original_key, header in key_map.items():
            if original_key not in row:
                raise ValueError(f"row {row_idx + 1} is missing key {original_key!r} (normalized to '{header}')")
            columns[header].append(row[original_key])
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
        from ._exceptions import DataError
        raise DataError(f"column {duplicates[0]!r} has a duplicate name")


def validate_column_lengths(columns: Mapping[str, Sequence[Any]]) -> None:
    lengths = {len(values) for values in columns.values()}
    if len(lengths) > 1:
        raise ValueError("all columns must have the same length")


def collect_record_headers(rows: list[Mapping[str, Any]]) -> tuple[list[str], dict[Any, str]]:
    headers: list[str] = []
    key_map: dict[Any, str] = {}
    seen_str = set()
    for row in rows:
        for key in row.keys():
            if key not in key_map:
                key_str = str(key)
                if key_str in seen_str:
                    raise ValueError(f"key collision: original key {key!r} normalizes to {key_str!r}, which is already used")
                seen_str.add(key_str)
                headers.append(key_str)
                key_map[key] = key_str
    return headers, key_map


def coerce_numeric_vector(values: Sequence[Any], *, label: str) -> list[float]:
    numeric: list[float] = []
    for index, value in enumerate(values):
        try:
            numeric_value = float(value)
        except (TypeError, ValueError) as exc:
            raise ValueError(
                f"{label} contains a non-numeric value at position {index + 1}: {value!r}"
            ) from exc
        if not math.isfinite(numeric_value):
            raise ValueError(f"{label} contains non-finite value at position {index + 1}")
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
