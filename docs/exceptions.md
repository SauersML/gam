# Exceptions

`gamfit` raises a small set of dedicated exception types. All of them
inherit from a common base so you can catch them generically.

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

## When each is raised

### `GamError`

Base class. Raised when the Rust engine returns an error that doesn't map
to one of the more specific types below. Catch this if you want a single
`except` for "something gamfit-side went wrong".

### `FormulaError`

The formula is invalid or unsupported. Common causes:

- Syntax error (missing `~`, bad parentheses).
- An identifier that isn't in the data.
- A smooth option that doesn't exist (e.g. `s(x, kk=10)`).
- A combination the engine can't fit (e.g. `linkwiggle` on `sas`).

```python
try:
    gamfit.fit(df, "y ~ s(x, kk=10)")
except gamfit.FormulaError as e:
    print(gamfit.explain_error(e))
    # "Check the formula syntax and confirm every referenced column exists."
```

### `SchemaMismatchError`

You called `predict()` (or similar) with data missing a column the model
needs, or with incompatible types. Use `Model.check(data)` to surface the
exact issues without raising:

```python
check = model.check(test_df)
if not check.ok:
    for issue in check.issues:
        print(issue.kind, issue.column, issue.message)
```

### `PredictionError`

Prediction failed for a reason that isn't a schema mismatch — for example,
the fitted model class isn't yet supported by the Python predict path. The
error message names what's supported.

### `RustExtensionUnavailableError`

The compiled extension `gamfit._rust` failed to load. This is rare on the
prebuilt wheels but can happen if you installed from source without a Rust
toolchain.

```python
try:
    gamfit.build_info()
except gamfit.RustExtensionUnavailableError as e:
    print(gamfit.explain_error(e))
    # "Build the extension with maturin before calling Rust-backed APIs."
```

## explain_error

`gamfit.explain_error(exc)` takes any exception and returns a short
human-readable hint:

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

## Practical patterns

### Defensive predict

```python
def safe_predict(model, data):
    check = model.check(data)
    if not check.ok:
        check.raise_for_error()
    return model.predict(data)
```

### Catch everything gamfit-side

```python
try:
    model = gamfit.fit(df, formula)
except gamfit.GamError as e:
    log.error("gamfit failed: %s — %s", type(e).__name__, gamfit.explain_error(e))
    raise
```
