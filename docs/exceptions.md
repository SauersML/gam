# Exceptions

`gamfit` raises Rust-defined Python exception classes and re-exports them
from `gamfit`. Engine errors share the `GamError` base, which inherits
from `ValueError`; the Rust extension unavailable case remains an
`ImportError`.

## Hierarchy

The classes are defined in Rust (`crates/gam-pyffi/src/ffi_errors.rs`,
via `pyo3::create_exception!`) and re-exported from `gamfit` by
`gamfit/_exceptions.py`. `GamError` is the umbrella base for every
engine error and itself inherits from `ValueError`; the only exception
outside that tree is `RustExtensionUnavailableError`, which inherits
from `ImportError`.

The tree below is exhaustive — it lists every exception class the
package exposes. The four classes that have their own children
(`FormulaError`, `PredictionError`, `InvalidSpecificationError`, and the
`GamError` root) show those children indented beneath them; every other
class is a direct `GamError` subclass.

```
Exception
├── ValueError
│   └── GamError
│       ├── FormulaError
│       │   ├── ColumnNotFoundError
│       │   └── TermBuilderError
│       ├── SchemaMismatchError
│       ├── PredictionError
│       │   ├── PredictInputError
│       │   └── SurvivalPredictError
│       ├── InvalidSpecificationError
│       │   ├── UnsupportedLinkError
│       │   └── InvalidConfigurationError
│       ├── BasisError
│       ├── LinearSystemSolveError
│       ├── EigendecompositionError
│       ├── PenaltySpectrumError
│       ├── ParameterConstraintError
│       ├── PirlsConvergenceError
│       ├── PerfectSeparationError
│       ├── HessianNotPositiveDefiniteError
│       ├── RemlConvergenceError
│       ├── GradientUnavailableError
│       ├── LayoutError
│       ├── ModelOverparameterizedError
│       ├── IllConditionedError
│       ├── InvalidInputError
│       ├── MonotoneRootError
│       ├── CalibratorError
│       ├── GeometryError
│       ├── MatrixMaterializationError
│       ├── GpuError
│       ├── LinearAlgebraError
│       ├── MatrixError
│       ├── CacheStoreError
│       ├── SmoothError
│       ├── ArrowSchurError
│       ├── OuterStrategyError
│       ├── CorrectedCovarianceError
│       ├── HmcError
│       ├── AloError
│       ├── SurvivalError
│       ├── CubicCellKernelError
│       ├── SurvivalConstructionError
│       ├── TransformationNormalError
│       ├── CustomFamilyError
│       ├── GamlssError
│       ├── SurvivalMarginalSlopeError
│       ├── LatentSurvivalError
│       ├── DeviationRuntimeError
│       ├── DataError
│       ├── FittedModelError
│       ├── LognormalKernelError
│       ├── ScaleDesignError
│       ├── IdentifiabilityCompilerError
│       ├── JointPenaltyError
│       ├── SurvivalLocationScaleError
│       ├── MapUniquenessError
│       ├── MissingDependencyError
│       └── IntegrationError
└── ImportError
    └── RustExtensionUnavailableError
```

Catch `GamError` for a stable package-level umbrella, or catch a
specific subclass when recovery depends on the failure mode (for
example, retry with looser tolerances on `RemlConvergenceError`, or
suggest more data on `ModelOverparameterizedError`). Each subclass maps
one-to-one to a Rust engine-error variant, selected at the FFI boundary
by variant dispatch rather than by message parsing.

`map_exception` (in `gamfit._exceptions`) preserves typed Rust
`GamError` subclasses unchanged. `RustExtensionUnavailableError` is also
returned unchanged. Python-native `TypeError`, `LookupError`, and
`ArithmeticError` pass through; other `ValueError`s from remaining string
FFI paths are promoted to `GamError`.

## When each is raised

### `GamError`

Base class for engine errors after the extension has loaded. It is also a
`ValueError`, so existing value-contract handlers continue to catch it.

### `FormulaError`

The formula is invalid or unsupported. Common causes:

- Syntax error (missing `~`, unbalanced parentheses).
- Parser-level unknown identifiers, such as unsupported term or link names.
- Missing formula columns, reported as `ColumnNotFoundError`, a
  `FormulaError` subclass with structured attributes such as `column`,
  `available`, and `similar` when available.

```python
try:
    gamfit.fit(df, "y ~ s(x, k=10")
except gamfit.FormulaError as e:
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

### `PredictionError`

Prediction failed for a reason other than a schema mismatch — numerical
issues, an unsupported prediction mode for the fitted model class, or a
prediction-time input error. Some prediction families raise subclasses
such as `PredictInputError` or `SurvivalPredictError`.

### `RustExtensionUnavailableError`

The compiled extension `gamfit._rust` failed to load. Occurs when
installing from source without a Rust toolchain.

```python
try:
    gamfit.fit(df, "y ~ s(x)")
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

### Catch every mapped gamfit error

```python
try:
    model = gamfit.fit(df, formula)
except gamfit.GamError as e:
    log.error("gamfit failed: %s — %s", type(e).__name__, gamfit.explain_error(e))
    raise
```
