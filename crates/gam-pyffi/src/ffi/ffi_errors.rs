//! Typed engine-error → Python-exception boundary.
//!
//! This module owns the canonical gamfit exception class hierarchy (defined
//! via `pyo3::create_exception!`) and every typed engine→Python adaptor that
//! converts a `gam` error enum into the matching exception subclass without a
//! message-regex classifier (issue #343). Concentrating both the class
//! identities and the variant-dispatch converters here keeps the error
//! contract in one place: `gamfit/_exceptions.py` re-exports the classes under
//! their public `gamfit.*` names, and every FFI submodule reaches the
//! converters/classes through the crate-root re-export.
//!
//! The classes live here (not in any fit/predict module) so the Rust
//! extension owns the canonical type identity; `gamfit/_exceptions.py`
//! re-exports them so the public names remain `gamfit.GamError`,
//! `gamfit.FormulaError`, etc.
//!
//! Inheritance: every gamfit exception is a subclass of `GamError`, and
//! `GamError` itself is a subclass of Python's built-in `ValueError`.
//! That preserves the historical contract that `except ValueError`
//! catches every engine-side failure (the Rust extension previously
//! raised bare `PyValueError` for everything), while `except GamError`
//! becomes the documented broad catch — see issue #330.
//!
//! Adding a new engine error variant: extend `estimation_error_to_pyerr`
//! (or the per-enum analogue) with the new variant; do NOT add new
//! patterns to a message-regex classifier.

use crate::ffi_prelude::*;

use pyo3::create_exception;

create_exception!(
    _rust,
    GamError,
    PyValueError,
    "Base class for Python-facing gamfit engine errors.\n\
     \n\
     All gamfit-specific exceptions raised by the Rust extension inherit\n\
     from `GamError`, which itself inherits from `ValueError` to preserve\n\
     the historical `except ValueError` contract."
);

create_exception!(
    _rust,
    FormulaError,
    GamError,
    "The Wilkinson-style formula could not be parsed or references columns \
     missing from the input table."
);

create_exception!(
    _rust,
    ColumnNotFoundError,
    FormulaError,
    "A formula referenced a column that does not exist in the input data.\n\
     \n\
     Instances carry structured attributes — `column` (str), `role` \
     (Optional[str]), `available` (list[str]), `similar` (list[str]), \
     and `tsv_hint` (bool) — set by the FFI boundary at raise time, so \
     callers can inspect the failure without parsing the message text. \
     `column` is the missing name as written, `available` is every header \
     present in the input, `similar` is a cheap shortlist of close matches, \
     and `tsv_hint` is True when the file is almost certainly a TSV mis-\
     extensioned as CSV (sole header contains literal tab characters). \
     Subclass of `FormulaError` so `except gamfit.FormulaError` still \
     catches it."
);

create_exception!(
    _rust,
    SchemaMismatchError,
    GamError,
    "Prediction input does not match the training schema."
);

create_exception!(
    _rust,
    PredictionError,
    GamError,
    "Prediction failed for a reason that is not a pure schema mismatch."
);

// EstimationError variant subclasses.
//
// Each subclass corresponds to exactly one variant of
// `gam::solver::estimate::EstimationError`. Catching the specific subclass lets
// callers branch on the exact failure mode (e.g. retry with looser
// tolerances on `RemlConvergenceError`, suggest more data on
// `ModelOverparameterizedError`).

create_exception!(
    _rust,
    BasisError,
    GamError,
    "Underlying basis function generation failed."
);

create_exception!(
    _rust,
    LinearSystemSolveError,
    GamError,
    "A linear system solve failed; the penalized Hessian may be singular."
);

create_exception!(
    _rust,
    EigendecompositionError,
    GamError,
    "Eigendecomposition failed."
);

create_exception!(
    _rust,
    PenaltySpectrumError,
    GamError,
    "Penalty spectrum check failed (non-finite or indefinite eigenvalue)."
);

create_exception!(
    _rust,
    ParameterConstraintError,
    GamError,
    "Parameter constraint violation."
);

create_exception!(
    _rust,
    PirlsConvergenceError,
    GamError,
    "The P-IRLS inner loop did not converge within its iteration budget."
);

create_exception!(
    _rust,
    PerfectSeparationError,
    GamError,
    "Perfect or quasi-perfect separation detected during model fitting."
);

create_exception!(
    _rust,
    HessianNotPositiveDefiniteError,
    GamError,
    "Hessian matrix is not positive definite at the converged iterate."
);

create_exception!(
    _rust,
    RemlConvergenceError,
    GamError,
    "REML smoothing optimization failed to converge."
);

create_exception!(
    _rust,
    GradientUnavailableError,
    GamError,
    "The unified evaluator returned no gradient in the requested mode."
);

create_exception!(
    _rust,
    LayoutError,
    GamError,
    "An internal error occurred during model layout or coefficient mapping."
);

create_exception!(
    _rust,
    ModelOverparameterizedError,
    GamError,
    "Model is over-parameterized: more coefficients than samples."
);

create_exception!(
    _rust,
    IllConditionedError,
    GamError,
    "Model is ill-conditioned (large condition number)."
);

create_exception!(
    _rust,
    InvalidInputError,
    GamError,
    "Invalid input to the engine (shape/dtype/range violation)."
);

create_exception!(
    _rust,
    MonotoneRootError,
    GamError,
    "Monotone-root solve failed."
);

create_exception!(
    _rust,
    CalibratorError,
    GamError,
    "Calibrator training failed."
);

create_exception!(
    _rust,
    InvalidSpecificationError,
    GamError,
    "Invalid specification supplied to the engine."
);

// -------------------------------------------------------------------------
// Remaining engine error enum subclasses (issue #343 follow-up).
//
// Each `pub enum *Error` in `src/` gets a corresponding subclass below, so
// every engine error path is variant-typed at the FFI boundary and no
// longer flows through the message-regex classifier. Inheritance is
// chosen by semantic relationship: builder-layer errors that arise from
// formula authoring (e.g. `TermBuilderError`) inherit from
// `FormulaError`; prediction-time input errors inherit from
// `PredictionError`; everything else inherits from `GamError`.
// -------------------------------------------------------------------------

create_exception!(
    _rust,
    GeometryError,
    GamError,
    "Riemannian-geometry / manifold-primitive operation failed \
     (dimension mismatch, invalid point, singular tangent space)."
);

create_exception!(
    _rust,
    MatrixMaterializationError,
    GamError,
    "Lazy design-matrix materialization failed (size cap exceeded, \
     forbidden by policy, or row-block evaluation failure)."
);

create_exception!(
    _rust,
    GpuError,
    GamError,
    "GPU offload path failed (driver unavailable, kernel launch error, \
     calibration failure, or feature not yet implemented on this device)."
);

create_exception!(
    _rust,
    LinearAlgebraError,
    GamError,
    "Dense linear-algebra primitive failed (factorization, SVD, or \
     eigendecomposition reported non-convergence or non-finite input)."
);

create_exception!(
    _rust,
    MatrixError,
    GamError,
    "Matrix-level invariant violated (dimension mismatch, refused \
     densification, or related shape contract failure)."
);

create_exception!(
    _rust,
    CacheStoreError,
    GamError,
    "Persistent on-disk model cache I/O or serialization failure."
);

create_exception!(
    _rust,
    SmoothError,
    GamError,
    "Smooth-term construction failed (invalid configuration for the \
     requested basis or penalty)."
);

create_exception!(
    _rust,
    ArrowSchurError,
    GamError,
    "Arrow-Schur block solver failed (per-row factor failure, ill-\
     conditioning, PCG non-convergence, or adaptive-correction failure)."
);

create_exception!(
    _rust,
    OuterStrategyError,
    GamError,
    "Outer smoothing-strategy contract violated (operator-shape \
     mismatch, non-finite Hessian, or rho-block shape error)."
);

create_exception!(
    _rust,
    TermBuilderError,
    FormulaError,
    "A formula term could not be built from the input data \
     (missing column, incompatible options, degenerate data, etc.). \
     Subclass of `FormulaError` so existing `except FormulaError` \
     handlers still catch it."
);

create_exception!(
    _rust,
    CorrectedCovarianceError,
    GamError,
    "Corrected posterior covariance construction failed \
     (shape mismatch, eigendecomposition failure, or indefinite outer \
     Hessian)."
);

create_exception!(
    _rust,
    PredictInputError,
    PredictionError,
    "Prediction input is invalid or incompatible with the fitted model \
     (shape mismatch, missing metadata, or malformed payload). \
     Subclass of `PredictionError`."
);

create_exception!(
    _rust,
    HmcError,
    GamError,
    "Hamiltonian Monte Carlo sampler failed (non-finite state, invalid \
     configuration, unsupported family / link, or sampling divergence)."
);

create_exception!(
    _rust,
    AloError,
    GamError,
    "Approximate leave-one-out computation failed (invalid input, \
     degenerate design, or influence-matrix factorization failure)."
);

create_exception!(
    _rust,
    SurvivalError,
    GamError,
    "Survival kernel invariant violated (dimension mismatch, non-finite \
     input, invalid time grid, non-monotone cumulative hazard, etc.)."
);

create_exception!(
    _rust,
    CubicCellKernelError,
    GamError,
    "Cubic-cell-moment kernel rejected an input (degenerate interval, \
     invalid cell shape, insufficient moments, or out-of-domain \
     bivariate-normal evaluation)."
);

create_exception!(
    _rust,
    SurvivalConstructionError,
    GamError,
    "Survival model construction failed (invalid config, missing column, \
     dimension mismatch, data validation, or unsupported distribution)."
);

create_exception!(
    _rust,
    TransformationNormalError,
    GamError,
    "Transformation-normal family rejected the design / response \
     (degenerate design, non-finite input, or monotonicity violation)."
);

create_exception!(
    _rust,
    CustomFamilyError,
    GamError,
    "Custom family contract violated (invalid input, optimization \
     failure, numerical failure, or identifiability violation)."
);

create_exception!(
    _rust,
    GamlssError,
    GamError,
    "GAMLSS location-scale family rejected the input (dimension \
     mismatch, non-finite, unsupported configuration, or constraint \
     violation)."
);

create_exception!(
    _rust,
    SurvivalMarginalSlopeError,
    GamError,
    "Survival marginal-slope family failed (invalid input, \
     monotonicity violation, integration failure, or unsupported \
     configuration)."
);

create_exception!(
    _rust,
    LatentSurvivalError,
    GamError,
    "Latent-survival family rejected the dataset (invalid frailty, \
     invalid dataset, block mismatch, or numerical failure)."
);

create_exception!(
    _rust,
    SurvivalPredictError,
    PredictionError,
    "Survival prediction failed (invalid input, missing fit metadata, \
     incompatible schema, or numerical failure). Subclass of \
     `PredictionError`."
);

create_exception!(
    _rust,
    DeviationRuntimeError,
    GamError,
    "Marginal-slope deviation runtime rejected the input (invalid \
     input, dimension mismatch, or numerical failure)."
);

create_exception!(
    _rust,
    DataError,
    GamError,
    "Input dataset failed schema / encoding validation (parse error, \
     empty input, invalid value, missing column)."
);

create_exception!(
    _rust,
    FittedModelError,
    GamError,
    "Saved fitted-model payload is incompatible (schema mismatch, \
     corrupt payload, missing field, or incompatible config)."
);

create_exception!(
    _rust,
    LognormalKernelError,
    GamError,
    "Lognormal kernel configuration is invalid."
);

create_exception!(
    _rust,
    ScaleDesignError,
    GamError,
    "Scale-design construction failed (invalid weights, dimension \
     mismatch, non-finite input, degenerate design, or SVD failure)."
);

create_exception!(
    _rust,
    IdentifiabilityCompilerError,
    GamError,
    "Identifiability compiler rejected the block layout (dimension \
     mismatch, fully aliased block, or linear-algebra failure)."
);

create_exception!(
    _rust,
    JointPenaltyError,
    GamError,
    "Joint penalty matrix rejected (not square, not symmetric, \
     non-finite entry, or nullspace too large)."
);

create_exception!(
    _rust,
    SurvivalLocationScaleError,
    GamError,
    "Survival location-scale family rejected the input (dimension \
     mismatch, invalid configuration, constraint violation, or \
     numerical failure)."
);

create_exception!(
    _rust,
    MapUniquenessError,
    GamError,
    "MAP-uniqueness identifiability audit detected duplicate or \
     overlapping posterior modes."
);

create_exception!(
    _rust,
    UnsupportedLinkError,
    InvalidSpecificationError,
    "An inverse-link / link transform was requested that the engine \
     does not support for the chosen family. Subclass of \
     `InvalidSpecificationError`."
);

create_exception!(
    _rust,
    InvalidConfigurationError,
    InvalidSpecificationError,
    "Fit configuration is internally inconsistent or selects an \
     unsupported combination (conflicting family/link, unsupported \
     link placement, frailty for an incompatible family, duplicate or \
     out-of-range hyperpriors). Subclass of `InvalidSpecificationError`."
);

create_exception!(
    _rust,
    MissingDependencyError,
    GamError,
    "A required input column, frailty parameter, baseline target, or \
     cause count is missing for the requested fit mode."
);

create_exception!(
    _rust,
    IntegrationError,
    GamError,
    "An underlying numerical step (PIRLS / smoothing-parameter \
     optimizer / profile-cost evaluation) failed to converge or \
     produced a non-finite value."
);

// -------------------------------------------------------------------------
// (Removed) Legacy message-regex classifier — issue #343.

/// Variant-dispatch: convert a typed engine error into the matching
/// Python exception subclass. This is the single chokepoint where
/// `EstimationError`'s typing is preserved across the FFI boundary —
/// no `err.to_string()` flattening, no Python-side regex reclassification.
pub(crate) fn estimation_error_to_pyerr(err: EstimationError) -> PyErr {
    let message = err.to_string();
    match err {
        EstimationError::BasisError(_) => BasisError::new_err(message),
        EstimationError::LinearSystemSolveFailed(_) => LinearSystemSolveError::new_err(message),
        EstimationError::EigendecompositionFailed(_) => EigendecompositionError::new_err(message),
        EstimationError::PenaltySpectrumNonFinite { .. } => PenaltySpectrumError::new_err(message),
        EstimationError::PenaltySpectrumIndefinite { .. } => PenaltySpectrumError::new_err(message),
        EstimationError::ParameterConstraintViolation(_) => {
            ParameterConstraintError::new_err(message)
        }
        EstimationError::PirlsDidNotConverge { .. } => PirlsConvergenceError::new_err(message),
        EstimationError::PerfectSeparationDetected { .. } => {
            PerfectSeparationError::new_err(message)
        }
        EstimationError::PrefitPerfectSeparationDetected { .. } => {
            PerfectSeparationError::new_err(message)
        }
        EstimationError::PrefitLinearSeparationDetected { .. } => {
            PerfectSeparationError::new_err(message)
        }
        EstimationError::MultinomialSeparationDetected { .. } => {
            PerfectSeparationError::new_err(message)
        }
        EstimationError::HessianNotPositiveDefinite { .. } => {
            HessianNotPositiveDefiniteError::new_err(message)
        }
        EstimationError::RemlOptimizationFailed(_) => RemlConvergenceError::new_err(message),
        EstimationError::GradientUnavailable { .. } => GradientUnavailableError::new_err(message),
        EstimationError::LayoutError(_) => LayoutError::new_err(message),
        EstimationError::ModelOverparameterized { .. } => {
            ModelOverparameterizedError::new_err(message)
        }
        EstimationError::PrefitRankDeficientDesignDetected { .. } => {
            ModelOverparameterizedError::new_err(message)
        }
        EstimationError::PrefitNearDegenerateDesignDetected { .. } => {
            IllConditionedError::new_err(message)
        }
        EstimationError::ModelIsIllConditioned { .. } => IllConditionedError::new_err(message),
        EstimationError::InvalidInput(_) => InvalidInputError::new_err(message),
        EstimationError::MonotoneRoot(_) => MonotoneRootError::new_err(message),
        EstimationError::CalibratorTrainingFailed(_) => CalibratorError::new_err(message),
        EstimationError::InvalidSpecification(_) => InvalidSpecificationError::new_err(message),
        EstimationError::PredictionError => PredictionError::new_err(message),
        EstimationError::CustomFamily(_) => CustomFamilyError::new_err(message),
    }
}

pub(crate) fn py_value_error(message: String) -> PyErr {
    // Engine errors funneled here are gamfit-specific failures, so they must
    // carry GamError identity (a ValueError subclass) — preserving the
    // historical `except ValueError` contract while making `except
    // gamfit.GamError` reliable for engine errors (issue #330).
    GamError::new_err(message)
}

fn panic_payload_message(payload: Box<dyn std::any::Any + Send>) -> String {
    if let Some(message) = payload.downcast_ref::<&'static str>() {
        (*message).to_string()
    } else if let Some(message) = payload.downcast_ref::<String>() {
        message.clone()
    } else {
        "non-string panic payload".to_string()
    }
}

fn py_panic_error(context: &'static str, payload: Box<dyn std::any::Any + Send>) -> PyErr {
    py_value_error(format!(
        "{context} panicked inside Rust boundary: {}",
        panic_payload_message(payload)
    ))
}

pub(crate) fn detach_py_result<T, F>(py: Python<'_>, context: &'static str, f: F) -> PyResult<T>
where
    T: Send + 'static,
    F: FnOnce() -> Result<T, String> + Send + 'static,
{
    match py.detach(move || catch_unwind(AssertUnwindSafe(f))) {
        Ok(Ok(value)) => Ok(value),
        Ok(Err(message)) => Err(py_value_error(message)),
        Err(payload) => Err(py_panic_error(context, payload)),
    }
}

/// A predict-path error that remembers whether it is a *schema mismatch*
/// (the caller's frame is missing a column the fitted model requires) or an
/// ordinary failure. The prediction FFI historically flattened every failure
/// to a bare `String`, so a missing-required-column rejection surfaced as the
/// generic `GamError` instead of the documented `SchemaMismatchError`
/// (issue #343's typed-error contract). Keeping the two cases distinct lets
/// [`detach_predict_result`] pick the right Python class without string
/// sniffing, while any non-schema `?` inside the predict impl still converts
/// straight through `From<String>`.
pub(crate) enum PredictError {
    /// The frame does not carry a column the model needs → `SchemaMismatchError`.
    SchemaMismatch(String),
    /// Any other predict failure → `GamError` (a `ValueError` subclass), the
    /// same class the bare-`String` path produced before.
    Other(String),
}

impl From<String> for PredictError {
    fn from(message: String) -> Self {
        PredictError::Other(message)
    }
}

impl From<PredictError> for String {
    fn from(err: PredictError) -> Self {
        match err {
            PredictError::SchemaMismatch(message) | PredictError::Other(message) => message,
        }
    }
}

/// Predict-path twin of [`detach_py_result`]: releases the GIL, runs the
/// closure, and maps a [`PredictError`] onto the *typed* Python exception —
/// `SchemaMismatch` → `SchemaMismatchError`, everything else → `GamError`.
/// Panics are still surfaced as the context-tagged panic error.
pub(crate) fn detach_predict_result<T, F>(
    py: Python<'_>,
    context: &'static str,
    f: F,
) -> PyResult<T>
where
    T: Send + 'static,
    F: FnOnce() -> Result<T, PredictError> + Send + 'static,
{
    match py.detach(move || catch_unwind(AssertUnwindSafe(f))) {
        Ok(Ok(value)) => Ok(value),
        Ok(Err(PredictError::SchemaMismatch(message))) => {
            Err(SchemaMismatchError::new_err(message))
        }
        Ok(Err(PredictError::Other(message))) => Err(py_value_error(message)),
        Err(payload) => Err(py_panic_error(context, payload)),
    }
}

/// Detach the GIL, run a closure that has already produced a typed
/// `PyResult<T>` (with the engine→Python class selection baked in), and
/// preserve panics as `py_panic_error`. This is the chokepoint for call
/// sites that need typed-variant dispatch where the engine error
/// originates inside the closure: convert with the matching
/// `*_error_to_pyerr` helper before returning. Replaces the
/// `.map_err(|e| e.to_string())?` flattening to typed-class loss (issue
/// #343).
pub(crate) fn detach_pyresult<T, F>(py: Python<'_>, context: &'static str, f: F) -> PyResult<T>
where
    T: Send + 'static,
    F: FnOnce() -> PyResult<T> + Send + 'static,
{
    match py.detach(move || catch_unwind(AssertUnwindSafe(f))) {
        Ok(Ok(value)) => Ok(value),
        Ok(Err(err)) => Err(err),
        Err(payload) => Err(py_panic_error(context, payload)),
    }
}

/// Detach the GIL, run a closure returning a typed `EstimationError`, and
/// preserve the variant across the Python boundary via
/// `estimation_error_to_pyerr`. This is the principled engine→Python
/// adaptor: no `err.to_string()` flattening, no message-regex
/// reclassification on the Python side. Each `EstimationError` variant
/// surfaces as a specific `gamfit.GamError` subclass (see issue #343).
pub(crate) fn detach_estimation_result<T, F>(
    py: Python<'_>,
    context: &'static str,
    f: F,
) -> PyResult<T>
where
    T: Send + 'static,
    F: FnOnce() -> Result<T, EstimationError> + Send + 'static,
{
    match py.detach(move || catch_unwind(AssertUnwindSafe(f))) {
        Ok(Ok(value)) => Ok(value),
        Ok(Err(err)) => Err(estimation_error_to_pyerr(err)),
        Err(payload) => Err(py_panic_error(context, payload)),
    }
}

/// Variant-dispatch the engine's top-level `WorkflowError` into the matching
/// Python exception class. The key entry is `WorkflowError::ColumnNotFound`,
/// which surfaces as `gamfit.ColumnNotFoundError` with the structured
/// fields attached as Python attributes (`column`, `role`, `available`,
/// `similar`, `tsv_hint`) — issue #305 / #343. Other variants degrade to
/// the most appropriate existing gamfit exception type; new variants can
/// be added to this single chokepoint as their dispatch is needed,
/// without ever growing a message-regex classifier.
pub(crate) fn workflow_error_to_pyerr(py: Python<'_>, err: WorkflowError) -> PyErr {
    match err {
        WorkflowError::ColumnNotFound {
            name,
            role,
            available,
            similar,
            tsv_hint,
        } => {
            // Build the canonical human-readable message from a `Display`
            // reconstruction so its text is anchored at the typed source,
            // never re-parsed downstream.
            let display = WorkflowError::ColumnNotFound {
                name: name.clone(),
                role: role.clone(),
                available: available.clone(),
                similar: similar.clone(),
                tsv_hint,
            }
            .to_string();
            let exc = ColumnNotFoundError::new_err(display);
            // Attach the structured payload as Python-side attributes so
            // `explain_error(exc)` and downstream code can read them via
            // `exc.column`, `exc.available`, etc. — no regex on the
            // formatted prose. PyO3 5 exposes `Bound<'_, PyAny>` from the
            // PyErr instance; setattr on it persists on the exception.
            let bound = exc.value(py);
            // Best-effort attribute attachment. If any setattr fails (e.g.
            // because a future PyO3 release tightens exception-instance
            // attribute access), we still raise the typed class with the
            // canonical message — the typed branch in `explain_error`
            // remains correct, only the per-instance enrichment is lost.
            // Errors are surfaced as Python unraisable warnings rather
            // than escalating, since the typed exception class itself is
            // the primary contract.
            let attach_result: PyResult<()> = (|| {
                bound.setattr("column", name.as_str())?;
                match role.as_deref() {
                    Some(r) => bound.setattr("role", r)?,
                    None => bound.setattr("role", py.None())?,
                }
                bound.setattr("available", available)?;
                bound.setattr("similar", similar)?;
                bound.setattr("tsv_hint", tsv_hint)?;
                Ok(())
            })();
            if let Err(attach_err) = attach_result {
                attach_err.write_unraisable(py, Some(&bound));
            }
            exc
        }
        // Variant-typed dispatch (issue #343). The four flavours that
        // previously flattened to bare `py_value_error(reason)` now each
        // carry a distinct typed subclass, so callers can branch on
        // `except InvalidConfigurationError` / `SchemaMismatchError` /
        // `MissingDependencyError` / `IntegrationError` without parsing
        // the prose. All four still inherit from `GamError` (and
        // therefore `ValueError`), so legacy `except ValueError` /
        // `except GamError` handlers keep catching them.
        WorkflowError::InvalidConfig { reason } => InvalidConfigurationError::new_err(reason),
        WorkflowError::SchemaMismatch { reason } => SchemaMismatchError::new_err(reason),
        WorkflowError::MissingDependency { reason } => MissingDependencyError::new_err(reason),
        WorkflowError::IntegrationFailed { reason } => IntegrationError::new_err(reason),
        WorkflowError::FormulaDsl { .. } => FormulaError::new_err(err.to_string()),
    }
}

pub(crate) fn detach_workflow_result<T, F>(
    py: Python<'_>,
    context: &'static str,
    f: F,
) -> PyResult<T>
where
    T: Send + 'static,
    F: FnOnce() -> Result<T, WorkflowError> + Send + 'static,
{
    match py.detach(move || catch_unwind(AssertUnwindSafe(f))) {
        Ok(Ok(value)) => Ok(value),
        Ok(Err(err)) => Err(workflow_error_to_pyerr(py, err)),
        Err(payload) => Err(py_panic_error(context, payload)),
    }
}

/// Variant-dispatch the engine's `GeometryError` into the typed Python
/// `gamfit.GeometryError`. All three variants — `DimensionMismatch`,
/// `InvalidPoint`, `Singular` — share the same Python class because the
/// distinction matters only in the message text; the typed class makes
/// `except gamfit.GeometryError` actionable without parsing the prose.
pub(crate) fn geometry_error_to_pyerr(err: EngineGeometryError) -> PyErr {
    GeometryError::new_err(err.to_string())
}

/// Detach the GIL, run a closure returning a typed `GeometryError`, and
/// preserve the variant across the Python boundary via
/// `geometry_error_to_pyerr` — the principled engine→Python adaptor for
/// every Poincaré / Lorentz / manifold primitive. Replaces the
/// `.map_err(|e| e.to_string())` flattening that used to surface as a
/// bare `PyValueError` (issue #343).
pub(crate) fn detach_geometry_result<T, F>(
    py: Python<'_>,
    context: &'static str,
    f: F,
) -> PyResult<T>
where
    T: Send + 'static,
    F: FnOnce() -> Result<T, EngineGeometryError> + Send + 'static,
{
    match py.detach(move || catch_unwind(AssertUnwindSafe(f))) {
        Ok(Ok(value)) => Ok(value),
        Ok(Err(err)) => Err(geometry_error_to_pyerr(err)),
        Err(payload) => Err(py_panic_error(context, payload)),
    }
}

// -------------------------------------------------------------------------
// Engine error → typed `PyErr` adaptors (issue #343).
//
// One trivial converter per typed engine→Python boundary actually used.
// Each helper preserves the typed-class identity so `except gamfit.SurvivalError`
// (etc.) is actionable without the user parsing the prose. A call site that does
// `.map_err(|e| e.to_string())?` against a `Result<_, EngineError>` in a
// `PyResult<_>` function should swap to `.map_err(<engine>_error_to_pyerr)?` —
// the message text is identical, only the Python class type widens from
// `ValueError` to the typed subclass. The orphan rule prevents a blanket
// `impl From<E> for PyErr`, so each converter is emitted explicitly via the
// `error_to_pyerr!` macro below.
// -------------------------------------------------------------------------

// Declarative converter generator: one orphan-rule-safe line per typed
// engine error → Python exception boundary actually needed at a call site.
// The body is invariably `<Pyo3Exc>::new_err(err.to_string())`; only the fn
// name, the source engine error type, and the PyO3 exception class vary. A
// blanket `impl From<E> for PyErr` is blocked by the orphan rule, so this
// macro is the canonical single source of truth for the trivial converters.
// Add a new typed converter by adding one `error_to_pyerr!(...)` invocation
// at the point a `.map_err(...)` site needs it.
macro_rules! error_to_pyerr {
    ($fn_name:ident, $src:ty, $exc:ident) => {
        pub(crate) fn $fn_name(err: $src) -> PyErr {
            $exc::new_err(err.to_string())
        }
    };
}

error_to_pyerr!(
    survival_error_to_pyerr,
    gam::families::survival::SurvivalError,
    SurvivalError
);
error_to_pyerr!(
    basis_error_to_pyerr,
    gam::terms::basis::BasisError,
    GamError
);
error_to_pyerr!(shape_error_to_pyerr, ndarray::ShapeError, GamError);
error_to_pyerr!(serde_json_error_to_pyerr, serde_json::Error, GamError);
