from __future__ import annotations

import importlib
import math
import numbers
from collections.abc import Mapping, Sequence
from decimal import Decimal
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
    """A table already normalized to ``(headers, native_table, kind)`` form.

    The native table is the Rust-owned, typed encoding produced by
    :func:`normalize_table`.  Reusing it across topology candidates avoids both
    reparsing and another dense numeric copy; categorical labels are retained
    only once in the Rust schema rather than expanded into a Python string per
    cell. ``kind`` is preserved so output restoration still reflects the input
    library.
    """

    __slots__ = ("headers", "rows", "kind")

    def __init__(self, headers: list[str], rows: Any, kind: str) -> None:
        self.headers = headers
        self.rows = rows
        self.kind = kind


def _try_import(name: str) -> Any | None:
    try:
        return importlib.import_module(name)
    except ImportError as e:
        if getattr(e, "name", None) != name:
            raise
        return None


# Sentinel prefix marking a cell that originates from a genuinely-categorical
# (string / object / categorical-dtype) source column. The Rust column-kind
# inference (`infer_and_encode_column_major`) treats any sentinel-prefixed cell
# as categorical regardless of whether its text parses as a number, then strips
# the sentinel before recording the level. This preserves the dtype intent of a
# pandas/polars/pyarrow column whose level labels happen to be numeric strings
# ("0", "1", "2"): without it such a column is silently lowered to a numeric
# by-variable, annihilating the "0" level and forcing amplitude ∝ label (#1317).
# A leading NUL never appears in a numeric literal and is stripped on the Rust
# side before any level matching, so predict frames (which re-run this same
# stringification) match the clean training levels.
CATEGORICAL_CELL_SENTINEL = "\x00"


def normalize_table(data: Any) -> tuple[list[str], Any, str]:
    if isinstance(data, PreNormalizedTable):
        return data.headers, data.rows, data.kind
    columns, kind = _table_column_views(data)
    headers = list(columns)
    if not headers:
        raise ValueError("table must have at least one column")
    reject_duplicate_column_names(headers, kind)
    validate_column_lengths(columns)
    row_count = len(columns[headers[0]])
    if row_count == 0:
        raise ValueError("table data cannot be empty")

    # Arrow-capable providers hand ownership of a fresh C stream capsule to
    # arrow-rs.  Numeric buffers are then decoded directly from Arrow memory and
    # only genuine string/dictionary-string columns allocate labels.  This is
    # the preferred path for PyArrow and Polars, and for pandas versions that
    # expose the Arrow PyCapsule protocol.
    if kind in {"pandas", "polars", "pyarrow"} and hasattr(
        data, "__arrow_c_stream__"
    ):
        from ._binding import rust_module

        return headers, rust_module().encoded_table_from_arrow(headers, data), kind

    categorical = categorical_dtype_columns(data, kind, columns=columns)
    numeric_positions = [
        index for index, header in enumerate(headers) if header not in categorical
    ]
    categorical_positions = [
        index for index, header in enumerate(headers) if header in categorical
    ]

    # rust-numpy borrows this one typed block at the boundary.  `column_stack`
    # may make one homogeneous numeric copy for a mixed-dtype frame, but it
    # never creates Python scalar objects and the Rust engine's final N×P f64
    # matrix is the only retained dense representation.
    import numpy as np

    if numeric_positions:
        numeric_values = np.column_stack(
            [
                np.asarray(columns[headers[index]], dtype=np.float64)
                for index in numeric_positions
            ]
        )
        numeric_values = np.ascontiguousarray(numeric_values, dtype=np.float64)
    else:
        numeric_values = np.empty((row_count, 0), dtype=np.float64)

    # Strings are the one column kind for which Python objects are intrinsic.
    # Convert those columns only, column-major, and let Rust infer/canonicalize
    # their levels without ever constructing a row-major Python table.
    categorical_values = [
        [
            stringify_cell(
                _external_scalar(columns[headers[index]][row]),
                column=headers[index],
                row=row,
            )
            for row in range(row_count)
        ]
        for index in categorical_positions
    ]

    from ._binding import rust_module

    native = rust_module().encoded_table_from_columns(
        headers,
        numeric_values,
        numeric_positions,
        categorical_values,
        categorical_positions,
    )
    return headers, native, kind


def _external_scalar(value: Any) -> Any:
    """Unwrap a scalar owned by an external table library.

    Arrow scalar wrappers expose ``as_py``; converting only categorical cells
    here avoids the former all-column ``to_pylist`` expansion.
    """
    as_py = getattr(value, "as_py", None)
    return as_py() if callable(as_py) else value


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
    # columnisation is the minimal representation and still avoids the former
    # second N×P string matrix.
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


def _stringify_marked(value: Any, is_categorical: bool, *, column: str | None = None, row: int | None = None) -> str:
    text = stringify_cell(value, column=column, row=row)
    if is_categorical:
        return CATEGORICAL_CELL_SENTINEL + text
    return text


def _infer_categorical_from_values(columns: dict[str, list[Any]]) -> frozenset[str]:
    # Mirror what pandas infers for a Python list / object column: a column is
    # categorical iff its values are NOT a uniform numeric (or bool) vector —
    # i.e. it carries at least one Python ``str``. The presence of ANY string
    # makes the whole column object-dtype-like, exactly as `pd.Series(...).dtype`
    # reports `object` (→ categorical) for a list that mixes strings with
    # anything else.
    #
    # Earlier this used an "ALL non-null values must be str" rule, which under-
    # detected MIXED string+numeric columns: `["a", 1, "b"]` is `object` dtype in
    # pandas (→ categorical) but the all-str rule saw the `1` and lowered the
    # column to a NUMERIC covariate — the same typed-vs-untyped parity gap as
    # #1467/#1468/#1469, one boundary case further out. A single string anywhere
    # is enough (and is correct): a column that cannot be parsed as uniformly
    # numeric is categorical, matching the pandas/polars/pyarrow branches.
    #
    # `bool` is NOT a `str`, so a pure-bool column (`[True, False]`, pandas `bool`
    # dtype) stays numeric, while a `bool`+`str` mix (`[True, "a"]`, pandas
    # `object`) becomes categorical via the string. `None`/`NaN` are nulls and
    # never make a column categorical on their own (an all-null column carries no
    # string and stays numeric, as pandas reports `float64` for an all-`NaN`
    # list). `numpy.str_` is a `str` subclass, so it is covered too.
    out = set()
    for name, values in columns.items():
        if any(isinstance(value, str) for value in values):
            out.add(name)
    return frozenset(out)


def categorical_dtype_columns(
    data: Any, kind: str, *, columns: dict[str, list[Any]] | None = None
) -> frozenset[str]:
    """Names of columns whose *source dtype* is non-numeric (string / object /
    categorical), independent of whether the rendered cell text parses as a
    number.

    Typed table libraries (pandas/polars/pyarrow) carry this signal in their
    declared schema, so the dtype is read directly. Untyped inputs (mappings,
    record/row sequences, numpy, plain row sequences) have no declared dtype, so
    the signal is inferred from the *values*: a column is categorical iff it
    carries at least one Python ``str`` (``numpy.str_`` is a ``str`` subclass, so
    it is covered too), mirroring how pandas assigns `object` dtype to any list
    that is not a uniform numeric/bool vector. A pure ``int``/``float``/``bool``/
    numpy-number column stays numeric (a genuinely-numeric ``by=`` covariate is
    preserved); a column that mixes a string with numerics (``["a", 1]``) is
    `object` in pandas and is therefore categorical here too. ``None`` and
    ``NaN`` are nulls and never make a column categorical on their own. This
    closes the dict/records/numpy gap (#1467/#1468/#1469) where a string column
    with numeric-looking labels ("0"/"1") was lowered to a numeric covariate, and
    its mixed-column corollary (a string+numeric column was likewise mis-lowered).
    """
    try:
        if kind == "pandas":
            import pandas as pd

            out = set()
            for index, name in enumerate(str(c) for c in data.columns):
                dtype = data.iloc[:, index].dtype
                if isinstance(dtype, pd.CategoricalDtype) or not (
                    pd.api.types.is_numeric_dtype(dtype)
                    or pd.api.types.is_bool_dtype(dtype)
                ):
                    out.add(name)
            return frozenset(out)
        if kind == "polars":
            import polars as pl

            out = set()
            for name in (str(c) for c in data.columns):
                dtype = data.schema[name]
                if dtype in (pl.Utf8, pl.Categorical, pl.Enum) or dtype == pl.Object:
                    out.add(name)
            return frozenset(out)
        if kind == "pyarrow":
            import pyarrow as pa

            out = set()
            for field in data.schema:
                t = field.type
                if (
                    pa.types.is_string(t)
                    or pa.types.is_large_string(t)
                    or pa.types.is_dictionary(t)
                ):
                    out.add(str(field.name))
            return frozenset(out)
    except (ImportError, AttributeError, TypeError, KeyError, ValueError):
        # Dtype introspection is a best-effort enhancement; if a library's
        # introspection API shifts, fall back to value-based inference rather
        # than fail the fit.
        pass

    # Untyped inputs (mappings, records, numpy, row sequences) have no
    # declared dtype, or we hit an introspection error above: infer
    # categoricality from the column values.
    if columns is None:
        columns, _ = table_columns(data)
    return _infer_categorical_from_values(columns)


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
        listed = ", ".join(repr(name) for name in duplicates)
        raise ValueError(
            f"{kind} input has duplicate column names ({listed}); "
            "every column must have a unique name"
        )


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


def stringify_cell(value: Any, *, column: str | None = None, row: int | None = None) -> str:
    if isinstance(value, bool) or type(value).__name__ == "bool_":
        return "1" if value else "0"
    if value is None:
        if column and row is not None:
            raise ValueError(f"column '{column}' has None at row {row + 1}")
        elif column:
            raise ValueError(f"column '{column}' contains None")
        else:
            raise ValueError("table cells cannot be None")
    if isinstance(value, (numbers.Real, Decimal)):
        numeric_value = float(value)
        if not math.isfinite(numeric_value):
            if column and row is not None:
                raise ValueError(f"column '{column}' has non-finite numeric value at row {row + 1}")
            elif column:
                raise ValueError(f"column '{column}' contains non-finite numeric value")
            else:
                raise ValueError("table cells cannot be non-finite numeric value")
        if isinstance(value, numbers.Integral):
            return repr(int(value))
        return repr(numeric_value)
    text = str(value)
    if not text:
        if column and row is not None:
            raise ValueError(f"column '{column}' has empty string at row {row + 1}")
        elif column:
            raise ValueError(f"column '{column}' contains empty string")
        else:
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
