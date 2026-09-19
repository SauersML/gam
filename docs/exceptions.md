# Exceptions

`gamfit` raises Rust-defined Python exception classes and re-exports them
from `gamfit.errors`. Every engine error is a `GamfitError`; the Rust extension
unavailable case remains an `ImportError`.

## Hierarchy

The engine classifies every failure into one `ErrorCategory`
(`crates/gam-spec/src/error_category.rs`): formula, data, convergence,
not-fitted or internal. The classes are defined in Rust
(`crates/gam-pyffi/src/ffi/ffi_errors.rs`) and re-exported by
`gamfit.errors`. The FFI boundary raises a class under the base of the
failure's category, chosen from the typed engine error and never from its
message. The CLI exits with the same category's code, so both front ends
classify a failure identically.

The tree below is exhaustive:

```
Exception
├── GamfitError
│   ├── FormulaError(GamfitError, ValueError)
│   │   ├── ColumnNotFoundError
│   │   ├── InvalidSpecificationError
│   │   │   └── InvalidConfigurationError
│   │   ├── BasisError
│   │   └── MissingDependencyError
│   ├── DataError(GamfitError, ValueError)
│   │   ├── SchemaMismatchError
│   │   ├── PredictionError
│   │   ├── PerfectSeparationError
│   │   ├── ModelOverparameterizedError
│   │   ├── IllConditionedError
│   │   ├── InvalidInputError
│   │   ├── GeometryError
│   │   └── FitInputError
│   ├── ConvergenceError(GamfitError, RuntimeError)
│   │   ├── FitConvergenceError
│   │   │   ├── PirlsConvergenceError
│   │   │   ├── RemlConvergenceError
│   │   │   └── InnerModeConvergenceError
│   │   ├── FitSeedError
│   │   ├── FitNumericalError
│   │   │   ├── LinearSystemSolveError
│   │   │   ├── EigendecompositionError
│   │   │   ├── PenaltySpectrumError
│   │   │   ├── ParameterConstraintError
│   │   │   ├── HessianNotPositiveDefiniteError
│   │   │   └── MonotoneRootError
│   │   ├── IntegrationError
│   │   ├── CalibratorError
│   │   └── DictionaryConvergenceError
│   ├── NotFittedError(GamfitError, ValueError, AttributeError)
│   └── InternalError(GamfitError, RuntimeError)
│       ├── FitInvariantError
│       ├── GradientUnavailableError
│       └── LayoutError
└── ImportError
    └── RustExtensionUnavailableError
```

| Category | Python base | CLI exit code | Meaning |
| --- | --- | --- | --- |
| formula | `FormulaError` | 2 | The request is wrong: formula syntax, an unknown term or option, a missing column, a term applied to a column of the wrong kind. Fix the call. |
| data | `DataError` | 3 | The request is well formed but the data cannot support it: separation, rank deficiency, non-finite values, a schema mismatch at predict time. Fix the data or simplify the model. |
| convergence | `ConvergenceError` | 4 | A valid problem whose numerical solve did not finish. Loosen tolerances, reparameterize or simplify. |
| not fitted | `NotFittedError` | 5 | A fitted-model method was called on an estimator that has not been fitted. |
| internal | `InternalError` | 70 | The engine broke its own contract. This is a bug; please report it. |

`FormulaError` and `DataError` are `ValueError`s and `ConvergenceError` is a
`RuntimeError`, so generic handlers keep working. `NotFittedError` has the
bases of `sklearn.exceptions.NotFittedError` (`ValueError` and
`AttributeError`), without gamfit importing scikit-learn.

Catch `GamfitError` for a stable package-level umbrella, a category base to
branch on what the caller should do, or a specific subclass when recovery
depends on the exact failure (for example, retry with looser tolerances on
`RemlConvergenceError`, or suggest more data on `ModelOverparameterizedError`).

`map_exception` (in `gamfit._exceptions`) returns every `GamfitError` and
`RustExtensionUnavailableError` unchanged. Python-native `TypeError`,
`LookupError`, and `ArithmeticError` pass through; a remaining `ValueError`
from argument validation in the Python layer becomes `FormulaError`.

## When each is raised

### `FormulaError`

The formula or an option is invalid or unsupported. Common causes:

- Syntax error (missing `~`, unbalanced parentheses).
- Parser-level unknown identifiers, such as unsupported term or link names.
- Missing formula columns, reported as `ColumnNotFoundError`, a
  `FormulaError` subclass with structured attributes such as `column`,
  `available`, and `similar` when available.
- A numeric term on a categorical column: `s(g)`, `te(x, g)` or `linear(g)`
  where `g` holds strings. The message names the column, reports the first
  non-numeric value and its row, and lists the terms that take a factor
  (`factor(g)`, `group(g)`, `s(x, by=g)`, `fs(x, g)`, `s(g, bs="re")`).

```python
try:
    gamfit.fit(df, "y ~ s(x, k=10")
except gamfit.errors.FormulaError as e:
    print(gamfit.explain_error(e))
```

### `SchemaMismatchError`

The data passed to `predict()` (or similar) is missing a column the model
needs, violates the saved schema, or introduces unseen categorical levels.
`Model.check(data)` reports missing columns directly and returns schema
encoder failures as issues without raising:

```python
check = model.check(test_df)
if not check.ok:
    for issue in check.issues:
        print(issue.kind, issue.column, issue.message)
```

### Fit failures

A model fit's solve failure raises the class of its fit category, chosen from
the typed engine error that stopped the fit, never from its message:

| Class | Category | Raised when |
| --- | --- | --- |
| `FitConvergenceError` | `convergence` | An outer smoothing search or an inner coefficient solve ended without its convergence certificate. `PirlsConvergenceError`, `RemlConvergenceError` and `InnerModeConvergenceError` are its subclasses. |
| `InnerModeConvergenceError` | `convergence` | The fit ended holding an inner coefficient solve that never certified its mode, with no outer search left to step away from it (gam#2943). `fields` and plain attributes carry `carrying_block` (the block holding the largest unresolved KKT residual), `cycles`, `cycle_budget`, `kkt_residual`, `kkt_tol` and `terminal_reason`; each optional one is `None` where the solve did not record it. |
| `FitSeedError` | `startup_seeds` | Outer startup validation refused every candidate seed, so no outer solver started. |
| `FitNumericalError` | `numerical` | A factorization, eigendecomposition or root solve failed, or a row quantity could not be represented in float64. |
| `IntegrationError` | `integration` | A quadrature or numerical integration did not reach its tolerance. |
| `ConvergenceError` itself | `unclassified` | The failure reached the Python boundary as prose, so no finer category can be claimed. |
| `FitInputError` | `input` | The solve refused the configuration, the data or the problem's size (separation, rank deficiency, an unsupported option). A `DataError`. |
| `FitInvariantError` | `invariant` | The engine's own consistency contract was violated, e.g. an inference covariance disagreeing with the top-level covariance (gam#1789). An `InternalError`; please report it. |

Every instance carries five attributes, and the message ends with the variant
and category after the unchanged engine message:

- `variant`: the typed engine variant that decided the failure, e.g.
  `EstimationError::StartupSeedsRefused` or
  `SurvivalMarginalSlopeError::RootSolveFailed`.
- `category`: the fit category label in the table above.
- `error_category`: the engine-wide category (`formula`, `data`,
  `convergence`, `not_fitted`, `internal`) that picked the base class.
- `causes`: the message chain, outermost first: each orchestration layer's
  context (such as `CTN fold 2 failed`), then the engine message.
- `fields`: a dict of the typed evidence the variant exposes, by field name, so a
  caller reads numbers without parsing the message. It is empty for a variant
  that exposes none.

```python
try:
    model = gamfit.fit(df, "y ~ s(x)")
except gamfit.errors.FitSeedError as e:
    print("no admissible start:", e.variant, e.causes[-1])
except gamfit.errors.FitConvergenceError as e:
    print("did not converge:", e.variant)
except gamfit.errors.ConvergenceError as e:
    print(e.category, e.variant, str(e))
```

### `PredictionError`

Prediction failed for a reason other than a schema mismatch — numerical
issues, an unsupported prediction mode for the fitted model class, or a
prediction-time input error.

### `NotFittedError`

`predict`, `score` or another fitted-model method was called on a
`gamfit.sklearn` estimator before `fit`. It is also a `ValueError` and an
`AttributeError`, like scikit-learn's own `NotFittedError`.

### `RustExtensionUnavailableError`

The compiled extension `gamfit._rust` failed to load. Occurs when
installing from source without a Rust toolchain.

```python
try:
    gamfit.fit(df, "y ~ s(x)")
except gamfit.errors.RustExtensionUnavailableError as e:
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
| Other `GamfitError` | "The Rust engine returned an error. Inspect the exception message for the underlying failure detail." |
| Other | "Unexpected error. Inspect the full traceback and the original exception message." |

Use it as a one-line user-facing message:

```python no-exec
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

### Catch every mapped gamfit error

```python
import logging

log = logging.getLogger("my_app")
formula = "y ~ s(x)"

try:
    model = gamfit.fit(df, formula)
except gamfit.errors.GamfitError as e:
    log.error("gamfit failed: %s — %s", type(e).__name__, gamfit.explain_error(e))
    raise
```
