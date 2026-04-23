from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

SUPPORTED_OUTPUT_KINDS = {"dict", "numpy", "pandas", "polars", "pyarrow"}


def normalize_table(data: Any) -> tuple[list[str], list[list[str]], str]:
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
        return (
            {str(column): data[column].tolist() for column in data.columns},
            kind,
        )
    if kind == "polars":
        return (
            {str(column): data[column].to_list() for column in data.columns},
            kind,
        )
    if kind == "pyarrow":
        return (
            {str(name): data.column(name).to_pylist() for name in data.column_names},
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
            return record_table_columns(rows_like), "records"
        if isinstance(rows_like[0], Sequence) and not isinstance(
            rows_like[0],
            (str, bytes, bytearray),
        ):
            return sequence_table_columns(rows_like), "rows"
    raise TypeError(
        "unsupported table input; use pandas, pyarrow, numpy, a mapping, a list of records, or a 2D row sequence"
    )


def restore_output_table(
    columns: dict[str, list[float]],
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
        return columns
    if target == "pandas":
        import pandas as pd

        return pd.DataFrame(columns)
    if target == "polars":
        import polars as pl

        return pl.DataFrame(columns)
    if target == "numpy":
        import numpy as np

        ordered = list(columns)
        return np.column_stack([columns[name] for name in ordered])
    if target == "pyarrow":
        import pyarrow as pa

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
    module = type(data).__module__
    name = type(data).__name__
    if module.startswith("pandas.") and name == "DataFrame":
        return "pandas"
    if module.startswith("polars.") and name == "DataFrame":
        return "polars"
    if module.startswith("pyarrow.") and name == "Table":
        return "pyarrow"
    if (module == "numpy" or module.startswith("numpy.")) and name == "ndarray":
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
    columns = {str(key): list(value) for key, value in data.items()}
    validate_column_lengths(columns)
    return columns


def record_table_columns(rows: list[Mapping[str, Any]]) -> dict[str, list[Any]]:
    headers = collect_record_headers(rows)
    columns = {header: [] for header in headers}
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
    columns = {header: [] for header in headers}
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
    if isinstance(value, (int, float)):
        if value != value:
            raise ValueError("table cells cannot be NaN")
        return repr(value)
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
        return array.tolist()
    if isinstance(values, Mapping):
        raise TypeError("target values must be a vector, not a mapping")
    if isinstance(values, Sequence) and not isinstance(values, (str, bytes, bytearray)):
        return list(values)
    raise TypeError("target values must be a 1D array-like sequence")
