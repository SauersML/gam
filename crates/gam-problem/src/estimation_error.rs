use gam_linalg::LinalgError;
use gam_linalg::faer_ndarray::FaerLinalgError;

use crate::{BasisError, CustomFamilyError, MonotoneRootError};

/// A comprehensive error type for the model estimation process.
#[derive(thiserror::Error)]
pub enum EstimationError {
    #[error("Underlying basis function generation failed: {0}")]
    BasisError(#[from] BasisError),

    #[error("Custom-family fit failed: {0}")]
    CustomFamily(#[from] CustomFamilyError),

    #[error("A linear system solve failed. The penalized Hessian may be singular. Error: {0}")]
    LinearSystemSolveFailed(FaerLinalgError),

    #[error("Eigendecomposition failed: {0}")]
    EigendecompositionFailed(FaerLinalgError),

    #[error(
        "Penalty spectrum check failed in '{context}': non-finite eigenvalue {value:?} at index {index}"
    )]
    PenaltySpectrumNonFinite {
        context: String,
        index: usize,
        value: f64,
    },

    #[error(
        "Penalty spectrum check failed in '{context}': indefinite eigenvalue {value:.3e} at index {index} (tolerance {tolerance:.3e}, scale {scale:.3e})"
    )]
    PenaltySpectrumIndefinite {
        context: String,
        index: usize,
        value: f64,
        tolerance: f64,
        scale: f64,
    },

    #[error("Parameter constraint violation: {0}")]
    ParameterConstraintViolation(String),

    #[error(
        "The P-IRLS inner loop did not converge within {max_iterations} iterations. Last gradient norm was {last_change:.6e}."
    )]
    PirlsDidNotConverge {
        max_iterations: usize,
        last_change: f64,
    },

    #[error(
        "Perfect or quasi-perfect separation detected during model fitting at iteration {iteration}. \
        The model cannot converge because a predictor perfectly separates the binary outcomes. \
        (Diagnostic: max|eta| = {max_abs_eta:.2e})."
    )]
    PerfectSeparationDetected { iteration: usize, max_abs_eta: f64 },

    #[error(
        "Pre-fit perfect separation detected in the realized binomial inverse-link design: column {column_index} \
        has a threshold {threshold:.6e} that separates the binary outcomes \
        (positive_above_threshold={positive_above_threshold}). The unpenalized MLE is not finite; \
        enable Firth/Jeffreys bias reduction or remove/reparameterize the separating column."
    )]
    PrefitPerfectSeparationDetected {
        column_index: usize,
        threshold: f64,
        positive_above_threshold: bool,
    },

    #[error(
        "Pre-fit linear separation detected in the realized binomial inverse-link design: \
        {num_unpenalized_columns} effectively unpenalized columns admit a separating direction \
        with minimum signed margin {min_signed_margin:.6e} (columns {column_indices:?}). \
        The unpenalized MLE is not finite; enable Firth/Jeffreys bias reduction or \
        remove/reparameterize/penalize the separating columns."
    )]
    PrefitLinearSeparationDetected {
        min_signed_margin: f64,
        num_unpenalized_columns: usize,
        column_indices: Vec<usize>,
    },

    #[error(
        "Pre-fit rank deficiency detected in the realized unpenalized design: rank {rank} < {num_unpenalized_columns} \
        unpenalized columns (min eigenvalue {min_eigenvalue:.3e}, tolerance {tolerance:.3e}, columns {column_indices:?}). \
        Remove/reparameterize the aliased columns or add an explicit penalty/constraint before fitting."
    )]
    PrefitRankDeficientDesignDetected {
        rank: usize,
        num_unpenalized_columns: usize,
        min_eigenvalue: f64,
        tolerance: f64,
        column_indices: Vec<usize>,
    },

    #[error(
        "Pre-fit near-degeneracy detected in the realized unpenalized design: the {num_unpenalized_columns} \
        unpenalized columns span a numerically rank-degenerate direction (Gram condition number {condition_number:.3e} \
        exceeds tolerance {tolerance:.3e}; min eigenvalue {min_eigenvalue:.3e}, max eigenvalue {max_eigenvalue:.3e}, \
        columns {column_indices:?}). The unpenalized normal equations are effectively singular along this direction, \
        so the fit would grind/diverge. Remove/reparameterize the near-aliased columns or add an explicit \
        penalty/constraint before fitting."
    )]
    PrefitNearDegenerateDesignDetected {
        num_unpenalized_columns: usize,
        condition_number: f64,
        min_eigenvalue: f64,
        max_eigenvalue: f64,
        tolerance: f64,
        column_indices: Vec<usize>,
    },

    #[error(
        "Perfect or quasi-perfect separation detected during multinomial fitting at iteration {iteration}. \
        The active class-{active_class_index} logit against the reference class is saturated at training row {row_index}, \
        so the unpenalized softmax MLE is not finite in that direction. \
        (Diagnostic: max|eta| = {max_abs_eta:.2e})."
    )]
    MultinomialSeparationDetected {
        iteration: usize,
        max_abs_eta: f64,
        active_class_index: usize,
        row_index: usize,
    },

    #[error(
        "Hessian matrix is not positive definite (minimum eigenvalue: {min_eigenvalue:.4e}). This indicates a numerical instability."
    )]
    HessianNotPositiveDefinite { min_eigenvalue: f64 },

    #[error("REML smoothing optimization failed to converge: {0}")]
    RemlOptimizationFailed(String),

    #[error("{context}: unified evaluator returned no gradient in {mode} mode")]
    GradientUnavailable {
        context: &'static str,
        mode: &'static str,
    },

    #[error("An internal error occurred during model layout or coefficient mapping: {0}")]
    LayoutError(String),

    #[error(
        "Model is over-parameterized: {num_coeffs} coefficients for {num_samples} samples.\n\n\
        Coefficient Breakdown:\n\
          - Intercept:                     {intercept_coeffs}\n\
          - Binary Main Effects:           {binary_main_coeffs}\n\
          - Primary Smooth Effects:        {primary_smooth_coeffs}\n\
          - Binary×Primary Interactions:   {binary_primary_interaction_coeffs}\n\
          - Auxiliary Main Effects:        {aux_main_coeffs}\n\
          - Auxiliary Interactions:        {aux_interaction_coeffs}"
    )]
    ModelOverparameterized {
        num_coeffs: usize,
        num_samples: usize,
        intercept_coeffs: usize,
        binary_main_coeffs: usize,
        primary_smooth_coeffs: usize,
        aux_main_coeffs: usize,
        binary_primary_interaction_coeffs: usize,
        aux_interaction_coeffs: usize,
    },

    #[error(
        "Model is ill-conditioned with condition number {condition_number:.2e}. This typically occurs when the model is over-parameterized (too many knots relative to data points). Consider reducing the number of knots or increasing regularization."
    )]
    ModelIsIllConditioned { condition_number: f64 },

    #[error("Invalid input: {0}")]
    InvalidInput(String),

    #[error("monotone root solve: {0}")]
    MonotoneRoot(#[from] MonotoneRootError),

    #[error("Calibrator training failed: {0}")]
    CalibratorTrainingFailed(String),

    #[error("Invalid specification: {0}")]
    InvalidSpecification(String),

    #[error("Prediction error")]
    PredictionError,
}

// Ensure Debug prints with actual line breaks by delegating to Display
impl core::fmt::Debug for EstimationError {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        write!(f, "{}", self)
    }
}

impl EstimationError {
    /// Classifies inner-solve failures that the outer REML loop should
    /// treat as a soft retreat (return +inf cost / infeasible outer-eval)
    /// rather than propagate as a hard error.
    ///
    /// Why: when the penalised Hessian becomes effectively singular at the
    /// current rho, when P-IRLS hits a perfect-separation diagnostic, or when
    /// it exhausts its iteration budget, the outer optimiser's correct
    /// response is to back away from this rho — not to terminate the fit.
    /// All three variants encode "the inner problem at this rho is too hard
    /// to evaluate, try a different rho".
    pub fn is_inner_solve_retreat(&self) -> bool {
        matches!(
            self,
            EstimationError::ModelIsIllConditioned { .. }
                | EstimationError::PerfectSeparationDetected { .. }
                | EstimationError::MultinomialSeparationDetected { .. }
                | EstimationError::PirlsDidNotConverge { .. }
        )
    }
}

impl From<LinalgError> for EstimationError {
    fn from(error: LinalgError) -> Self {
        match error {
            LinalgError::InvalidInput(message) => EstimationError::InvalidInput(message),
            LinalgError::HessianNotPositiveDefinite { min_eigenvalue } => {
                EstimationError::HessianNotPositiveDefinite { min_eigenvalue }
            }
            LinalgError::ModelIsIllConditioned { condition_number } => {
                EstimationError::ModelIsIllConditioned { condition_number }
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    // ── is_inner_solve_retreat ────────────────────────────────────────────────

    #[test]
    fn model_ill_conditioned_is_retreat() {
        assert!(EstimationError::ModelIsIllConditioned { condition_number: 1e15 }
            .is_inner_solve_retreat());
    }

    #[test]
    fn perfect_separation_is_retreat() {
        assert!(EstimationError::PerfectSeparationDetected {
            iteration: 3,
            max_abs_eta: 50.0
        }
        .is_inner_solve_retreat());
    }

    #[test]
    fn multinomial_separation_is_retreat() {
        assert!(EstimationError::MultinomialSeparationDetected {
            iteration: 1,
            max_abs_eta: 100.0,
            active_class_index: 2,
            row_index: 7
        }
        .is_inner_solve_retreat());
    }

    #[test]
    fn pirls_did_not_converge_is_retreat() {
        assert!(EstimationError::PirlsDidNotConverge {
            max_iterations: 100,
            last_change: 1e-3
        }
        .is_inner_solve_retreat());
    }

    #[test]
    fn invalid_input_is_not_retreat() {
        assert!(
            !EstimationError::InvalidInput("bad".to_string()).is_inner_solve_retreat()
        );
    }

    #[test]
    fn reml_optimization_failed_is_not_retreat() {
        assert!(
            !EstimationError::RemlOptimizationFailed("outer fail".to_string())
                .is_inner_solve_retreat()
        );
    }

    // ── error message content ─────────────────────────────────────────────────

    #[test]
    fn invalid_input_message_appears_in_display() {
        let err = EstimationError::InvalidInput("test_message".to_string());
        assert!(err.to_string().contains("test_message"));
    }

    #[test]
    fn pirls_did_not_converge_mentions_max_iterations() {
        let err = EstimationError::PirlsDidNotConverge {
            max_iterations: 42,
            last_change: 0.001,
        };
        assert!(err.to_string().contains("42"));
    }

    // ── From<LinalgError> ─────────────────────────────────────────────────────

    #[test]
    fn from_linalg_invalid_input_maps_to_invalid_input() {
        let linalg_err = LinalgError::InvalidInput("linalg msg".to_string());
        let err = EstimationError::from(linalg_err);
        assert!(matches!(err, EstimationError::InvalidInput(_)));
        assert!(err.to_string().contains("linalg msg"));
    }

    #[test]
    fn from_linalg_hessian_not_spd_maps_correctly() {
        let linalg_err = LinalgError::HessianNotPositiveDefinite { min_eigenvalue: -1.0 };
        let err = EstimationError::from(linalg_err);
        assert!(matches!(
            err,
            EstimationError::HessianNotPositiveDefinite { .. }
        ));
    }
}
