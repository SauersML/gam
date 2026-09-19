# Exceptions

`gamfit` raises Rust-defined Python exception classes and re-exports them
from `gamfit`. Engine errors share the `GamError` base, which inherits
from `ValueError`; the Rust extension unavailable case remains an
`ImportError`.

## Hierarchy

The classes are defined in Rust (`crates/gam-pyffi/src/ffi/ffi_errors.rs`,
with a few geometry classes in `crates/gam-pyffi/src/manifold/geometry_ffi.rs`,
via `pyo3::create_exception!`) and re-exported from `gamfit` by
`gamfit/_exceptions.py`. `GamError` is the umbrella base for every
engine error and itself inherits from `ValueError`; the only exception
outside that tree is `RustExtensionUnavailableError`, which inherits
from `ImportError`.

The tree below is exhaustive — it lists every exception class the
package exposes. The classes that have their own children
(`FormulaError`, `PredictionError`, `InvalidSpecificationError`, `FitError`,
`FitConvergenceError`, and the `GamError` root) show those children indented
beneath them; every other class is a direct `GamError` subclass.

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
│       ├── FitError
│       │   ├── FitConvergenceError
│       │   │   ├── PirlsConvergenceError
│       │   │   ├── RemlConvergenceError
│       │   │   └── InnerModeConvergenceError
│       │   ├── FitSeedError
│       │   ├── FitInvariantError
│       │   ├── FitInputError
│       │   ├── FitNumericalError
│       │   └── IntegrationError
│       ├── PerfectSeparationError
│       ├── HessianNotPositiveDefiniteError
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
│       └── MissingDependencyError
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

### `FitError` and its subclasses

A model fit's solve failed. The class names the failure's category, chosen
from the typed engine error that stopped the fit, never from its message:

| Class | Category | Raised when |
| --- | --- | --- |
| `FitConvergenceError` | `convergence` | An outer smoothing search or an inner coefficient solve ended without its convergence certificate. `PirlsConvergenceError`, `RemlConvergenceError` and `InnerModeConvergenceError` are its subclasses. |
| `InnerModeConvergenceError` | `convergence` | The fit ended holding an inner coefficient solve that never certified its mode, with no outer search left to step away from it (gam#2943). `fields` and plain attributes carry `carrying_block` (the block holding the largest unresolved KKT residual), `cycles`, `cycle_budget`, `kkt_residual`, `kkt_tol` and `terminal_reason`; each optional one is `None` where the solve did not record it. |
| `FitSeedError` | `startup_seeds` | Outer startup validation refused every candidate seed, so no outer solver started. |
| `FitInvariantError` | `invariant` | The engine's own consistency contract was violated, e.g. an inference covariance disagreeing with the top-level covariance (gam#1789). An engine defect; please report it. |
| `FitInputError` | `input` | The solve refused the configuration, the data or the problem's size (separation, rank deficiency, an unsupported option). |
| `FitNumericalError` | `numerical` | A factorization, eigendecomposition or root solve failed, or a row quantity could not be represented in float64. |
| `IntegrationError` | `integration` | A quadrature or numerical integration did not reach its tolerance. |
| `FitError` itself | `unclassified` | The failure reached the Python boundary as prose, so no category can be claimed. |

Every instance carries four attributes, and the message ends with the variant
and category after the unchanged engine message:

- `variant`: the typed engine variant that decided the failure, e.g.
  `EstimationError::StartupSeedsRefused` or
  `SurvivalMarginalSlopeError::RootSolveFailed`.
- `category`: the label in the table above.
- `causes`: the message chain, outermost first: each orchestration layer's
  context (such as `CTN fold 2 failed`), then the engine message.
- `fields`: a dict of the typed evidence the variant exposes, by field name, so a
  caller reads numbers without parsing the message. It is empty for a variant
  that exposes none.

```python
try:
    model = gamfit.fit(df, "y ~ s(x)")
except gamfit.FitSeedError as e:
    print("no admissible start:", e.variant, e.causes[-1])
except gamfit.FitConvergenceError as e:
    print("did not converge:", e.variant)
except gamfit.FitError as e:
    print(e.category, e.variant, str(e))
```

Before gam#2937 every fit failure raised `IntegrationError`. Code that
caught `IntegrationError` to handle any fit failure should catch `FitError`
instead; `IntegrationError` is now raised only for genuine integration
failures.

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
except gamfit.GamError as e:
    log.error("gamfit failed: %s — %s", type(e).__name__, gamfit.explain_error(e))
    raise
```
