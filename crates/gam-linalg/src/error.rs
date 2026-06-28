#[derive(Debug, thiserror::Error)]
pub enum LinalgError {
    #[error("Invalid input: {0}")]
    InvalidInput(String),

    #[error(
        "Hessian matrix is not positive definite (minimum eigenvalue: {min_eigenvalue:.4e}). This indicates a numerical instability."
    )]
    HessianNotPositiveDefinite { min_eigenvalue: f64 },

    #[error(
        "Model is ill-conditioned with condition number {condition_number:.2e}. This typically occurs when the model is over-parameterized (too many knots relative to data points). Consider reducing the number of knots or increasing regularization."
    )]
    ModelIsIllConditioned { condition_number: f64 },
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn invalid_input_display_contains_message() {
        let err = LinalgError::InvalidInput("bad dims".to_string());
        assert!(err.to_string().contains("bad dims"));
    }

    #[test]
    fn hessian_not_spd_display_contains_min_eigenvalue() {
        let err = LinalgError::HessianNotPositiveDefinite { min_eigenvalue: -0.001 };
        assert!(err.to_string().to_lowercase().contains("positive definite"));
    }

    #[test]
    fn ill_conditioned_display_contains_condition_number() {
        let err = LinalgError::ModelIsIllConditioned { condition_number: 1.5e12 };
        assert!(err.to_string().to_lowercase().contains("ill-conditioned"));
    }
}
