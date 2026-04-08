# RFC 0001: Python API v1 Contract

Status: Draft

## Purpose

Define the first stable Python-facing contract for `gam` before adding PyO3 bindings.

This RFC freezes the user-facing shape of `fit`, `predict`, `summary`, `save`, `load`, and `check`. It does not freeze the full Rust library surface. Rust remains the compute engine. Python is an ergonomic layer on top.

## Decision

The project will expose Python in three layers:

1. `gam-core`: a small Rust kernel API for fit, predict, diagnostics, report, save, and load.
2. `gam-pyffi`: a thin PyO3 boundary that passes Arrow tables, JSON options, HTML, and opaque model bytes.
3. `python/gam`: a pure-Python package that handles pandas, Polars, PyArrow, plotting, notebook display, sklearn wrappers, and friendly exceptions.

The Python ABI will not bind internal Rust structs directly. The boundary will use:

- Arrow tables for tabular input and output
- JSON for options and summaries
- opaque model bytes for persistence and in-memory transport

## Goals

- Keep Rust as the only implementation of fitting, prediction, sampling, generation, and schema validation.
- Preserve one formula language across CLI, Rust, and Python.
- Make the Python API formula-first and notebook-friendly.
- Keep the FFI boundary small enough that internal Rust refactors do not break Python.

## Non-goals for v1

- Binding every current Rust module or request/result type
- Python-authored custom families
- Exposing every experimental knob
- Maintaining a compatibility shim for an older Python surface
- Replacing the CLI or reimplementing compute in Python

## Python top-level API

```python
import gam

model = gam.fit(
    data=df,
    formula="y ~ s(age) + s(bmi) + group(site)",
    family="auto",
    config=None,
)

pred = model.predict(df_test, interval=0.95)
summary = model.summary()
check = model.check(df_test)
model.report("report.html")
model.save("risk_model.gam")

loaded = gam.load("risk_model.gam")
```

### `gam.fit`

```python
gam.fit(
    data,
    formula,
    *,
    family="auto",
    predict=None,
    weights=None,
    offset=None,
    config=None,
    return_type=None,
) -> Model
```

Contract:

- `data` accepts pandas `DataFrame`, Polars `DataFrame`, PyArrow `Table`, or a low-level Arrow-compatible table object.
- `formula` uses the exact formula language already supported by the CLI.
- `family="auto"` means Python passes no explicit family override and Rust resolves it.
- `predict`, `weights`, and `offset` are column names, not array side channels.
- `config` is a Python dict mapped to a versioned JSON config payload.
- `return_type` is optional. By default, prediction output matches the input table family when practical.

Validation:

- Formula parsing happens in Rust.
- Schema inference and column-kind detection happen in Rust.
- Python validates only obvious user-shape errors before calling FFI.

### `gam.load`

```python
gam.load(path) -> Model
gam.loads(model_bytes) -> Model
```

Contract:

- Loaded models preserve training formula, family state, and saved schema metadata from the Rust model payload.
- Python may attach extra presentation metadata, but core model compatibility is determined in Rust.

## `Model` contract

```python
class Model:
    def predict(self, data, *, interval=None, type=None, return_type=None):
        ...

    def summary(self):
        ...

    def check(self, data):
        ...

    def report(self, path=None, *, data=None):
        ...

    def save(self, path):
        ...
```

### `Model.predict`

```python
model.predict(
    data,
    *,
    interval=None,
    type=None,
    return_type=None,
)
```

Contract:

- Prediction executes entirely in Rust.
- `interval` requests uncertainty output as a central interval in `(0, 1)`, for example `0.95`.
- `type=None` returns the model-default prediction table for the fitted model kind.
- Output is tabular. Column names come from the Rust prediction payload and remain aligned with CLI naming where possible.
- For pandas input, default output is pandas. For Polars input, default output is Polars. Otherwise default output is PyArrow.

Prediction must fail on schema mismatch. There is no permissive fallback path that silently drops, reorders, or coerces incompatible serving columns.

### `Model.summary`

Returns a rich Python object backed by a Rust JSON summary payload.

Contract:

- No plain-text parsing contract.
- The object must support notebook display and `str(summary)` for terminals.
- The JSON payload is versioned separately from the Python class layout.

### `Model.check`

```python
model.check(data) -> SchemaCheck
```

Contract:

- Runs schema compatibility checks without computing full predictions.
- Reports missing columns, unknown columns, incompatible column kinds, and invalid categorical levels when applicable.
- Returns a structured object; it does not print directly.

### `Model.report`

```python
model.report(path=None, *, data=None) -> str
```

Contract:

- Produces HTML from Rust.
- If `path` is provided, Python writes the HTML to disk and returns the path.
- If `path` is omitted, Python returns the HTML string for notebooks and web contexts.

### `Model.save`

```python
model.save(path)
model.dumps() -> bytes
```

Contract:

- Persistence format is defined by Rust and remains opaque to Python.
- Python does not reinterpret or rewrite the internal model payload.

## Rust kernel boundary

The Python layer depends on a curated façade, not on the full `src/lib.rs` re-export graph.

The Rust kernel must expose only a narrow surface equivalent to:

- `fit_table(data, formula, config_json) -> model_bytes`
- `predict_table(model_bytes, data, options_json) -> arrow_table`
- `summary_json(model_bytes) -> json_string`
- `check_json(model_bytes, data) -> json_string`
- `report_html(model_bytes, data, options_json) -> html_string`
- `load_model(model_bytes) -> validated_model_handle`
- `save_model(model_handle) -> model_bytes`

The exact Rust function names may differ, but the boundary types may not expand beyond Arrow, JSON, HTML, and opaque model bytes without a new RFC.

## Error model

Rust errors must cross the FFI boundary as structured error codes plus message text. Python maps them into typed exceptions.

Initial Python exception set:

- `GamError`
- `FormulaError`
- `SchemaMismatchError`
- `SeparationError`
- `ConditioningError`
- `FitError`
- `PredictionError`

The exception taxonomy is part of the Python contract. Internal Rust error enums are not.

## Scope for v1

Required in v1:

- pandas input
- formula-based fit
- predict
- summary
- save and load
- schema check
- notebook HTML repr for `Model`

Explicitly deferred:

- sklearn estimators
- Polars-specific optimizations
- direct sampling and generation APIs
- full advanced family surface
- Python-side plotting beyond a first simple helper

## Compatibility rule

Compatibility is defined at the Python API level and at the saved-model payload level only.

The current broad Rust library exports are not the Python compatibility contract and may be narrowed or reorganized as needed to support this RFC.
