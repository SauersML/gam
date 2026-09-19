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
// `ModelOverparameterizedError`, which `PrefitRankDeficientDesignDetected` raises).

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

// Fit-failure categories (#2937). Every failure of a fit's solve used to reach
// Python as `IntegrationError`; each now raises the class of its category,
// selected from the typed engine error (`FailureCategory`), never from text.

create_exception!(
    _rust,
    FitError,
    GamError,
    "A model fit's solve failed. Subclasses name the failure's category; an \
     instance of `FitError` itself is a failure that reached the boundary as \
     prose, with no category to claim. Instances carry `variant` (str, the typed \
     engine variant, e.g. `EstimationError::StartupSeedsRefused`), `category` \
     (str), `causes` (list[str], the message chain, outermost first) and \
     `fields` (dict, the typed evidence the variant exposes by field name; empty \
     when it exposes none)."
);

create_exception!(
    _rust,
    FitConvergenceError,
    FitError,
    "An outer smoothing search or an inner coefficient solve ended without its \
     convergence certificate."
);

create_exception!(
    _rust,
    FitSeedError,
    FitError,
    "Outer startup validation refused every candidate seed, so no outer solver \
     started."
);

create_exception!(
    _rust,
    FitInvariantError,
    FitError,
    "A fit result or intermediate state violated the engine's own consistency \
     contract. An engine defect, not a property of the data; please report it."
);

create_exception!(
    _rust,
    FitInputError,
    FitError,
    "The fit's solve refused the configuration, the data or the problem's size \
     (separation, rank deficiency, an unsupported option)."
);

create_exception!(
    _rust,
    FitNumericalError,
    FitError,
    "A numerical step of the fit failed: a factorization, an eigendecomposition, \
     a root solve, or a row quantity that float64 cannot represent."
);

create_exception!(
    _rust,
    PirlsConvergenceError,
    FitConvergenceError,
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
    FitConvergenceError,
    "REML smoothing optimization failed to converge."
);

create_exception!(
    _rust,
    InnerModeConvergenceError,
    FitConvergenceError,
    "The fit ended holding an inner coefficient solve that never certified its \
     mode, with no outer search left to step away from it (gam#2943). `fields` \
     and plain attributes carry the terminal solve's evidence: `carrying_block`, \
     `cycles`, `cycle_budget`, `kkt_residual`, `kkt_tol` and `terminal_reason`, \
     each optional one None where the solve did not record it."
);

create_exception!(
    _rust,
    DictionaryConvergenceError,
    GamError,
    "A dictionary optimizer failed to reach its certified fixed point. Instances \
     carry the solver's structured residual evidence; no partial fit is returned."
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
    FitError,
    "A quadrature or numerical integration did not reach its tolerance. \
     Every fit failure used to raise this class; since gam#2937 other failures \
     raise their own `FitError` subclass."
);

// -------------------------------------------------------------------------
// (Removed) Legacy message-regex classifier — issue #343.

/// Variant-dispatch: convert a typed engine error into the matching
/// Python exception subclass. This is the single chokepoint where
/// `EstimationError`'s typing is preserved across the FFI boundary —
/// no `err.to_string()` flattening, no Python-side regex reclassification.
pub(crate) fn estimation_error_to_pyerr(err: EstimationError) -> PyErr {
    estimation_error_to_pyerr_with_message(&err, message_with_advice(&err, err.advice()))
}

/// The exception message: the rendered error plus, when the typed error
/// declares one, the same `help:` remediation the CLI prints. Both front ends
/// read it from the error's own `advice()`, so they cannot disagree (#2470).
fn message_with_advice(err: &dyn std::fmt::Display, advice: Option<String>) -> String {
    match advice {
        Some(advice) => format!("{err}\nhelp: {advice}"),
        None => err.to_string(),
    }
}

/// Select the Python exception from the innermost typed cause while retaining
/// the complete outer message. `OuterObjectiveEvaluationFailed` records which
/// orchestration boundary observed a fatal evaluator failure; it does not
/// replace the source error's public class identity.
fn estimation_error_to_pyerr_with_message(err: &EstimationError, message: String) -> PyErr {
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
        // The fixed-lambda Newton lanes (multinomial softmax, vector GLM,
        // Firth refit) are the inner conditional solve, exactly the PIRLS
        // role, so their typed exhaustion shares the PIRLS exception class.
        EstimationError::FixedLambdaNewtonDidNotConverge { .. } => {
            PirlsConvergenceError::new_err(message)
        }
        // Outer smoothing-parameter searches that ended without a stationarity
        // certificate all carry REML-convergence identity: the caller's remedy
        // (resume from the carried checkpoint / loosen the outer tolerance) is
        // the same across these lanes.
        EstimationError::RemlDidNotConverge { .. } => RemlConvergenceError::new_err(message),
        EstimationError::DominatedCertifiedPlateau { .. } => RemlConvergenceError::new_err(message),
        EstimationError::BlockOrthogonalRemlDidNotConverge { .. } => {
            RemlConvergenceError::new_err(message)
        }
        EstimationError::NegativeBinomialAlternationDidNotConverge { .. } => {
            RemlConvergenceError::new_err(message)
        }
        // The post-REML (beta, phi) alternation is an inner mean re-solve at a
        // refreshed precision; its failure is a P-IRLS certification failure.
        EstimationError::BetaPrecisionRefinementDidNotConverge { .. } => {
            PirlsConvergenceError::new_err(message)
        }
        EstimationError::FitDidNotConverge { .. } => RemlConvergenceError::new_err(message),
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
        EstimationError::LaplacePrecisionIndefinite { .. } => {
            HessianNotPositiveDefiniteError::new_err(message)
        }
        EstimationError::RemlOptimizationFailed(_) => RemlConvergenceError::new_err(message),
        EstimationError::StartupSeedsRefused(_) => FitSeedError::new_err(message),
        // The outer certificate at the fitted point describes a criterion whose
        // identified rank can change inside its own Newton step, so what the
        // caller holds is an uncertified outer optimum.
        EstimationError::IdentifiedRankNotLocallyConstant { .. } => {
            RemlConvergenceError::new_err(message)
        }
        // A penalty trace outside its rank beyond its solve's band says the
        // Hessian and the penalty were not one operator: an engine consistency
        // failure, neither a convergence budget nor an integration.
        EstimationError::EdfTraceOutsideRank { .. } => FitInvariantError::new_err(message),
        // A trial-point refusal only reaches Python when the outer smoothing
        // search never found a rho it could evaluate — so what the caller is
        // holding is an outer non-convergence, and the remedy (reseed, widen
        // the window, loosen the outer tolerance) is the REML one.
        EstimationError::TrialPointRefused { .. } => RemlConvergenceError::new_err(message),
        // A rho-local stage of the #784 block quadrature correction reaches Python only when the
        // outer search found no rho it could evaluate, which is an outer non-convergence like
        // `TrialPointRefused`. A corrector that breaks its moment contract is an engine invariant.
        EstimationError::BlockQuadratureCorrectionRefused { stage } => {
            if stage.is_trial_point_local() {
                RemlConvergenceError::new_err(message)
            } else {
                FitInvariantError::new_err(message)
            }
        }
        EstimationError::OuterObjectiveEvaluationFailed { source, .. } => {
            if let Some(source) = source.estimation_error() {
                estimation_error_to_pyerr_with_message(source, message)
            } else {
                // An opt client may attach a typed error that is not an engine
                // `EstimationError`. It is still an evaluator-construction
                // failure, not evidence that REML merely exhausted its
                // convergence budget, and nothing about it says integration.
                // Preserve the complete boundary message under the `FitError`
                // base without guessing a category from that source's prose.
                FitError::new_err(message)
            }
        }
        EstimationError::GradientUnavailable { .. } => GradientUnavailableError::new_err(message),
        EstimationError::LayoutError(_) => LayoutError::new_err(message),
        EstimationError::PrefitRankDeficientDesignDetected { .. } => {
            ModelOverparameterizedError::new_err(message)
        }
        EstimationError::PrefitNearDegenerateDesignDetected { .. } => {
            IllConditionedError::new_err(message)
        }
        EstimationError::ModelIsIllConditioned { .. } => IllConditionedError::new_err(message),
        EstimationError::InvalidInput(_) | EstimationError::ProfiledResidualUnresolved { .. } => {
            InvalidInputError::new_err(message)
        }
        EstimationError::FitResultInvariantViolated(_) => FitInvariantError::new_err(message),
        EstimationError::InverseLinkDomainViolation { .. }
        | EstimationError::PirlsRowGeometryUnrepresentable { .. }
        | EstimationError::LogStrengthDomainViolation { .. } => InvalidInputError::new_err(message),
        EstimationError::MonotoneRoot(_) => MonotoneRootError::new_err(message),
        EstimationError::CalibratorTrainingFailed(_) => CalibratorError::new_err(message),
        EstimationError::InvalidSpecification(_) => InvalidSpecificationError::new_err(message),
        EstimationError::PredictionError => PredictionError::new_err(message),
        // The fitted posterior's second-order model cannot publish a spread; the
        // prediction request is what declines, not the fit (#1082).
        EstimationError::PredictiveIntervalsDeclined { .. } => PredictionError::new_err(message),
        // A fit that ended holding an uncertified inner solve (gam#2943). Only
        // that variant carries terminal inner-mode evidence. It precedes the
        // catch-all below, whose class is not a `FitError`.
        EstimationError::CustomFamily(family_error)
            if family_error.terminal_inner_mode_evidence().is_some() =>
        {
            InnerModeConvergenceError::new_err(message)
        }
        EstimationError::CustomFamily(_) => CustomFamilyError::new_err(message),
        // Invalid stabilization metadata is a model/solver specification
        // defect, not a data problem.
        EstimationError::InvalidStabilization(_) => InvalidSpecificationError::new_err(message),
        // The exact Tweedie series refusing its term budget is a typed
        // convergence-class refusal of the fit's likelihood evaluation; users
        // catch the same class as other did-not-converge outcomes.
        EstimationError::ExactTweedieSeriesWorkLimit { .. } => {
            RemlConvergenceError::new_err(message)
        }
        // A dense copy the process cannot hold is a statement about the size of
        // the data the caller supplied, so it carries invalid-input identity.
        EstimationError::DenseMaterializationRefused { .. } => InvalidInputError::new_err(message),
    }
}

pub(crate) fn py_value_error(message: String) -> PyErr {
    // Engine errors funneled here are gamfit-specific failures, so they must
    // carry GamError identity (a ValueError subclass) — preserving the
    // historical `except ValueError` contract while making `except
    // gamfit.GamError` reliable for engine errors (issue #330).
    GamError::new_err(message)
}

fn py_panic_error(context: &'static str, payload: Box<dyn std::any::Any + Send>) -> PyErr {
    py_value_error(format!(
        "{context} panicked inside Rust boundary: {}",
        gam_runtime::panic_payload_message(payload)
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

/// Panic-safe GIL-detached execution for an engine result whose error enum has a
/// typed Python dispatcher. Unlike [`detach_py_result`], this preserves the enum
/// until the closure rejoins the GIL, so variant identity and structured evidence
/// are not flattened through `String`.
pub(crate) fn detach_typed_py_result<T, E, F, M>(
    py: Python<'_>,
    context: &'static str,
    f: F,
    map_error: M,
) -> PyResult<T>
where
    T: Send + 'static,
    E: Send + 'static,
    F: FnOnce() -> Result<T, E> + Send + 'static,
    M: FnOnce(Python<'_>, E) -> PyErr,
{
    match py.detach(move || catch_unwind(AssertUnwindSafe(f))) {
        Ok(Ok(value)) => Ok(value),
        Ok(Err(error)) => Err(map_error(py, error)),
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
        WorkflowError::SchemaMismatch { .. } => {
            let advice = err.advice();
            SchemaMismatchError::new_err(message_with_advice(&err, advice))
        }
        WorkflowError::MissingDependency { reason } => MissingDependencyError::new_err(reason),
        WorkflowError::Fit(failure) => fit_failure_to_pyerr(
            py,
            FitFailureReport {
                message: failure.to_string(),
                variant: failure.variant_name(),
                category: failure.category(),
                causes: failure.causes(),
                estimation_error: failure.estimation_error(),
                terminal_inner_mode: failure.terminal_inner_mode_evidence(),
            },
        ),
        WorkflowError::InvalidData { column, problem } => {
            DataError::new_err(format!("column '{column}' {problem}"))
        }
        // The certification refit's failure, when one failed, decides the
        // category; otherwise the resolution search itself did not converge.
        WorkflowError::SpatialUnderresolved {
            ref refit_failure, ..
        } => {
            let refit = match refit_failure.as_deref() {
                Some(WorkflowError::Fit(failure)) => Some(failure),
                _ => None,
            };
            fit_failure_to_pyerr(
                py,
                FitFailureReport {
                    message: err.to_string(),
                    variant: err.variant_name(),
                    category: err.failure_category(),
                    causes: vec![err.to_string()],
                    estimation_error: refit.and_then(|failure| failure.estimation_error()),
                    terminal_inner_mode: refit
                        .and_then(|failure| failure.terminal_inner_mode_evidence()),
                },
            )
        }
        WorkflowError::FormulaDsl { .. } => FormulaError::new_err(err.to_string()),
        WorkflowError::MarginalSlopeLink { .. } => InvalidConfigurationError::new_err(err.to_string()),
        WorkflowError::TransformationNormalConflict { .. } => {
            InvalidConfigurationError::new_err(err.to_string())
        }
    }
}

/// What the boundary reads off a fit failure to raise it.
struct FitFailureReport<'a> {
    message: String,
    variant: &'static str,
    category: gam::FailureCategory,
    causes: Vec<String>,
    estimation_error: Option<&'a EstimationError>,
    /// The terminal inner solve's facts, when the fit ended without a
    /// certified inner mode (gam#2943).
    terminal_inner_mode: Option<gam::TerminalInnerModeEvidence<'a>>,
}

/// The exception class of a fit failure's category. `Unclassified` raises the
/// `FitError` base: that failure reached the boundary as prose, so no subclass
/// can be claimed for it.
fn fit_category_error(category: gam::FailureCategory, message: String) -> PyErr {
    use gam::FailureCategory as Category;
    match category {
        Category::Convergence => FitConvergenceError::new_err(message),
        Category::StartupSeeds => FitSeedError::new_err(message),
        Category::Invariant => FitInvariantError::new_err(message),
        Category::Input => FitInputError::new_err(message),
        Category::Numerical => FitNumericalError::new_err(message),
        Category::Integration => IntegrationError::new_err(message),
        Category::Unclassified => FitError::new_err(message),
    }
}

/// Raise a fit's solve failure as the class of its category (#2937).
///
/// The message is the failure's complete rendered chain, unchanged, followed by
/// the typed variant and category, and by the `help:` line when the engine
/// error declares one. When the failure ends in an `EstimationError` whose own
/// class already names a fit category (`RemlConvergenceError` is a
/// `FitConvergenceError`), that more specific class is raised, so a fit and a
/// direct estimation entry point raise the same class for the same variant.
/// The variant, category and message chain are also set as attributes, so a
/// caller branches on them without parsing the message.
fn fit_failure_to_pyerr(py: Python<'_>, report: FitFailureReport<'_>) -> PyErr {
    let FitFailureReport {
        message,
        variant,
        category,
        causes,
        estimation_error,
        terminal_inner_mode,
    } = report;
    let mut message = format!(
        "{message}\nvariant: {variant}\ncategory: {}",
        category.label()
    );
    if let Some(advice) = estimation_error.and_then(EstimationError::advice) {
        message.push_str("\nhelp: ");
        message.push_str(&advice);
    }
    // A fit that ended without a certified inner mode raises its own class
    // whichever leaf carried it; the fit boundary mints it on a custom-family
    // leaf, which carries no estimation error (gam#2943).
    let specific = match terminal_inner_mode {
        Some(_) => Some(InnerModeConvergenceError::new_err(message.clone())),
        None => estimation_error
            .map(|source| estimation_error_to_pyerr_with_message(source, message.clone()))
            .filter(|candidate| candidate.is_instance_of::<FitError>(py)),
    };
    let exc = specific.unwrap_or_else(|| fit_category_error(category, message));
    let bound = exc.value(py);
    // As for `ColumnNotFoundError`: the class is the contract, the attributes
    // are enrichment, so an attribute that cannot be set is reported as
    // unraisable rather than replacing the typed exception.
    let attach_result: PyResult<()> = (|| {
        bound.setattr("variant", variant)?;
        bound.setattr("category", category.label())?;
        bound.setattr("causes", causes)?;
        // The typed evidence a variant exposes, by field name, each also set as
        // a plain attribute. Only a fit that ended without a certified inner
        // mode exposes any yet (gam#2943); every other variant gets an empty
        // dict rather than a missing attribute.
        let fields = pyo3::types::PyDict::new(py);
        if let Some(evidence) = terminal_inner_mode {
            fields.set_item("carrying_block", evidence.carrying_block)?;
            fields.set_item("cycles", evidence.cycles)?;
            fields.set_item("cycle_budget", evidence.cycle_budget)?;
            fields.set_item("kkt_residual", evidence.kkt_residual)?;
            fields.set_item("kkt_tol", evidence.kkt_tol)?;
            fields.set_item("terminal_reason", evidence.terminal_reason)?;
        }
        for (name, value) in fields.iter() {
            bound.setattr(name.str()?, value)?;
        }
        bound.setattr("fields", fields)?;
        Ok(())
    })();
    if let Err(attach_err) = attach_result {
        attach_err.write_unraisable(py, Some(&bound));
    }
    exc
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

#[cfg(test)]
mod fit_failure_dispatch_tests {
    use super::*;
    use gam::FailureCategory;
    use gam::families::fit_orchestration::FitFailure;

    fn raise(failure: FitFailure) -> PyErr {
        Python::attach(|py| workflow_error_to_pyerr(py, WorkflowError::Fit(failure)))
    }

    #[test]
    fn each_fit_failure_category_raises_its_own_class_2937() {
        Python::attach(|py| {
            let seeds = raise(FitFailure::from(EstimationError::StartupSeedsRefused(
                "no candidate seeds passed outer startup validation (custom family):".to_string(),
            )));
            assert!(seeds.is_instance_of::<FitSeedError>(py));

            let reml = raise(FitFailure::from(EstimationError::RemlOptimizationFailed(
                "stalled".to_string(),
            )));
            assert!(reml.is_instance_of::<RemlConvergenceError>(py));
            assert!(reml.is_instance_of::<FitConvergenceError>(py));

            let laws = raise(FitFailure::raised(
                FailureCategory::Convergence,
                "expectile LAWS exhausted its safety cap",
            ));
            assert!(laws.is_instance_of::<FitConvergenceError>(py));
            assert!(!laws.is_instance_of::<RemlConvergenceError>(py));

            let invariant = raise(FitFailure::from(EstimationError::FitResultInvariantViolated(
                "UnifiedFitResult inference conditional covariance must match top-level \
                 covariance_conditional"
                    .to_string(),
            )));
            assert!(invariant.is_instance_of::<FitInvariantError>(py));

            let input = raise(FitFailure::raised(
                FailureCategory::Input,
                "gaussian location-scale fit: the response has no finite positive spread",
            ));
            assert!(input.is_instance_of::<FitInputError>(py));

            let numerical = raise(FitFailure::raised(
                FailureCategory::Numerical,
                "survival marginal-slope intercept solve failed",
            ));
            assert!(numerical.is_instance_of::<FitNumericalError>(py));

            let integration = raise(FitFailure::raised(
                FailureCategory::Integration,
                "quadrature missed its tolerance",
            ));
            assert!(integration.is_instance_of::<IntegrationError>(py));

            let prose = raise(FitFailure::unclassified("a helper's prose"));
            assert!(prose.is_instance_of::<FitError>(py));
            for subclass_check in [
                prose.is_instance_of::<FitConvergenceError>(py),
                prose.is_instance_of::<FitSeedError>(py),
                prose.is_instance_of::<FitInvariantError>(py),
                prose.is_instance_of::<FitInputError>(py),
                prose.is_instance_of::<FitNumericalError>(py),
                prose.is_instance_of::<IntegrationError>(py),
            ] {
                assert!(!subclass_check, "prose must not claim a category");
            }

            for err in [&seeds, &reml, &laws, &invariant, &input, &numerical, &prose] {
                assert!(err.is_instance_of::<FitError>(py));
                assert!(err.is_instance_of::<GamError>(py));
                assert!(
                    !err.is_instance_of::<IntegrationError>(py),
                    "only a genuine integration failure is an IntegrationError"
                );
            }
        });
    }

    #[test]
    fn a_fit_exception_carries_the_variant_category_and_full_chain_2937() {
        Python::attach(|py| {
            let seeds = EstimationError::StartupSeedsRefused(
                "no candidate seeds passed outer startup validation (custom family):".to_string(),
            );
            let engine_text = seeds.to_string();
            let err = raise(FitFailure::from(seeds).context("CTN fold 1 failed"));
            let value = err.value(py);
            assert_eq!(
                value.to_string(),
                format!(
                    "CTN fold 1 failed: {engine_text}\nvariant: \
                     EstimationError::StartupSeedsRefused\ncategory: startup_seeds"
                )
            );
            let variant: String = value.getattr("variant").unwrap().extract().unwrap();
            assert_eq!(variant, "EstimationError::StartupSeedsRefused");
            let category: String = value.getattr("category").unwrap().extract().unwrap();
            assert_eq!(category, "startup_seeds");
            let causes: Vec<String> = value.getattr("causes").unwrap().extract().unwrap();
            assert_eq!(causes, vec!["CTN fold 1 failed".to_string(), engine_text]);
            let fields = value.getattr("fields").unwrap();
            let fields = fields.cast::<pyo3::types::PyDict>().unwrap();
            assert!(fields.is_empty(), "this variant exposes no typed evidence");
        });
    }

    #[test]
    fn a_direct_estimation_invariant_raises_the_invariant_class_2937() {
        Python::attach(|py| {
            let err = estimation_error_to_pyerr(EstimationError::FitResultInvariantViolated(
                "UnifiedFitResult inference conditional covariance must match top-level \
                 covariance_conditional"
                    .to_string(),
            ));
            assert!(err.is_instance_of::<FitInvariantError>(py));
            assert!(!err.is_instance_of::<InvalidInputError>(py));

            let trace = estimation_error_to_pyerr(EstimationError::EdfTraceOutsideRank {
                block: 2,
                trace: 6.09e4,
                rank: 22,
                band: 1.0e-6,
            });
            assert!(trace.is_instance_of::<FitInvariantError>(py));
            assert!(
                !trace.is_instance_of::<IntegrationError>(py),
                "only a genuine integration failure is an IntegrationError"
            );
        });
    }

    fn uncertified_inner_solve() -> gam::families::custom_family::CustomFamilyError {
        gam::families::custom_family::CustomFamilyError::InnerSolveNotConverged {
            cycles: 8,
            terminal: None,
            kkt_residual: Some(1.081e3),
            kkt_tol: Some(5.352e-2),
            theta_dim: 5,
            rho_dim: 3,
            psi_dim: 2,
            cycle_budget: Some(8),
            carrying_block: Some("slope_surface".to_string()),
        }
    }

    #[test]
    fn a_fit_ending_without_a_certified_inner_mode_raises_its_class_with_the_evidence_2943() {
        use gam::families::custom_family::CustomFamilyError as EngineCustomFamilyError;
        Python::attach(|py| {
            let terminal = EngineCustomFamilyError::fit_ended_without_certified_inner_mode(
                uncertified_inner_solve(),
            );
            // The fit boundary mints the variant on a custom-family leaf, under
            // the context of the layers above it.
            let err = raise(FitFailure::from(terminal.clone()).context("CTN fold 1 failed"));
            assert!(err.is_instance_of::<InnerModeConvergenceError>(py));
            assert!(err.is_instance_of::<FitConvergenceError>(py));
            assert!(!err.is_instance_of::<RemlConvergenceError>(py));
            let value = err.value(py);
            let variant: String = value
                .getattr("variant")
                .expect("variant attribute")
                .extract()
                .expect("variant is a str");
            assert_eq!(variant, "CustomFamilyError::FitEndedWithoutCertifiedInnerMode");
            let category: String = value
                .getattr("category")
                .expect("category attribute")
                .extract()
                .expect("category is a str");
            assert_eq!(category, "convergence");

            let cycles: usize = value
                .getattr("cycles")
                .expect("cycles attribute")
                .extract()
                .expect("cycles is an int");
            assert_eq!(cycles, 8);
            let cycle_budget: Option<usize> = value
                .getattr("cycle_budget")
                .expect("cycle_budget attribute")
                .extract()
                .expect("cycle_budget is an int or None");
            assert_eq!(cycle_budget, Some(8));
            let carrying_block: Option<String> = value
                .getattr("carrying_block")
                .expect("carrying_block attribute")
                .extract()
                .expect("carrying_block is a str or None");
            assert_eq!(carrying_block.as_deref(), Some("slope_surface"));
            let kkt_residual: Option<f64> = value
                .getattr("kkt_residual")
                .expect("kkt_residual attribute")
                .extract()
                .expect("kkt_residual is a float or None");
            assert_eq!(kkt_residual, Some(1.081e3));
            let kkt_tol: Option<f64> = value
                .getattr("kkt_tol")
                .expect("kkt_tol attribute")
                .extract()
                .expect("kkt_tol is a float or None");
            assert_eq!(kkt_tol, Some(5.352e-2));
            assert!(
                value
                    .getattr("terminal_reason")
                    .expect("terminal_reason attribute")
                    .is_none(),
                "a refusal with no terminal verdict reports None"
            );

            let fields = value.getattr("fields").expect("fields attribute");
            let fields = fields
                .cast::<pyo3::types::PyDict>()
                .expect("fields is a dict");
            assert_eq!(fields.len(), 6);
            for name in [
                "carrying_block",
                "cycles",
                "cycle_budget",
                "kkt_residual",
                "kkt_tol",
                "terminal_reason",
            ] {
                let from_fields = fields
                    .get_item(name)
                    .expect("dict lookup")
                    .expect("every evidence key is in fields");
                let attribute = value.getattr(name).expect("every evidence key is an attribute");
                assert!(
                    from_fields.eq(attribute).expect("evidence values compare"),
                    "fields[{name}] and the {name} attribute disagree"
                );
            }

            // The same variant carried as an estimation error, through a fit or
            // a direct estimation entry point, raises the same class.
            let estimation =
                raise(FitFailure::from(EstimationError::CustomFamily(terminal.clone())));
            assert!(estimation.is_instance_of::<InnerModeConvergenceError>(py));
            let direct = estimation_error_to_pyerr(EstimationError::CustomFamily(terminal));
            assert!(direct.is_instance_of::<InnerModeConvergenceError>(py));

            // A refusal that never ended a fit keeps the category class and
            // exposes no evidence.
            let trial = raise(FitFailure::from(uncertified_inner_solve()));
            assert!(trial.is_instance_of::<FitConvergenceError>(py));
            assert!(!trial.is_instance_of::<InnerModeConvergenceError>(py));
            let trial_fields = trial.value(py).getattr("fields").expect("fields attribute");
            let trial_fields = trial_fields
                .cast::<pyo3::types::PyDict>()
                .expect("fields is a dict");
            assert!(trial_fields.is_empty(), "only the fit-ending variant exposes evidence");
        });
    }
}
