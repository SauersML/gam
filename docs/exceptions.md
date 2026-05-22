# Exceptions

`gamfit` raises a small set of dedicated exception types. They share a
common base.

## Hierarchy

```
Exception
└── GamError
    ├── FormulaError
    ├── SchemaMismatchError
    └── PredictionError

ImportError
└── RustExtensionUnavailableError
```

`map_exception` (in `gamfit._exceptions`) inspects the message of an
exception raised by the Rust binding and returns the most specific class.
The classification rules are keyword-based on the lower-cased message:

| Keyword in message | Mapped class |
| --- | --- |
| `formula` or `parse` | `FormulaError` |
| `schema`, `missing required column`, `unknown column` | `SchemaMismatchError` |
| `prediction` or `predict` | `PredictionError` |
| (any other Rust error) | `GamError` |

`RustExtensionUnavailableError` is returned unchanged.

## When each is raised

### `GamError`

Base class. Raised for engine errors that do not match a more specific
keyword. Catch this to handle any gamfit-side failure.

### `FormulaError`

The formula is invalid or unsupported. Common causes:

- Syntax error (missing `~`, unbalanced parentheses).
- Identifier not present in the data.
- An unknown smooth option (e.g. `s(x, kk=10)`).
- A combination the engine cannot fit.

```python
try:
    gamfit.fit(df, "y ~ s(x, kk=10)")
except gamfit.FormulaError as e:
    print(gamfit.explain_error(e))
```

### `SchemaMismatchError`

The data passed to `predict()` (or similar) is missing a column the model
needs, has incompatible dtypes, or introduces unknown categorical levels.
`Model.check(data)` reports the exact issues without raising:

```python
check = model.check(test_df)
if not check.ok:
    for issue in check.issues:
        print(issue.kind, issue.column, issue.message)
```

### `PredictionError`

Prediction failed for a reason other than a schema mismatch — numerical
issues, an unsupported prediction mode for the fitted model class, etc.

### `RustExtensionUnavailableError`

The compiled extension `gamfit._rust` failed to load. Occurs when
installing from source without a Rust toolchain.

```python
try:
    gamfit.build_info()
except gamfit.RustExtensionUnavailableError as e:
    print(gamfit.explain_error(e))
```

## `explain_error`

`gamfit.explain_error(exc)` returns a short human-readable hint for any
exception:

| Exception | Hint |
| --- | --- |
| `RustExtensionUnavailableError` | "Build the extension with maturin before calling Rust-backed APIs." |
| `FormulaError` | "Check the formula syntax and confirm every referenced column exists." |
| `SchemaMismatchError` | "Compare the serving data with the training schema using model.check(...)." |
| `PredictionError` | "Prediction failed. Validate the new data and confirm the fitted model is supported by the Python binding." |
| Other `GamError` | "The Rust engine returned an error. Inspect the exception message for the underlying failure detail." |
| Other | "Unexpected error. Inspect the full traceback and the original exception message." |

Use it as a one-line user-facing message:

```python
try:
    model.predict(bad_df)
except Exception as e:
    raise SystemExit(f"{type(e).__name__}: {gamfit.explain_error(e)}")
```

## Patterns

### Defensive predict

```python
def safe_predict(model, data):
    check = model.check(data)
    if not check.ok:
        check.raise_for_error()
    return model.predict(data)
```

### Catch every gamfit-side error

```python
try:
    model = gamfit.fit(df, formula)
except gamfit.GamError as e:
    log.error("gamfit failed: %s — %s", type(e).__name__, gamfit.explain_error(e))
    raise
```
