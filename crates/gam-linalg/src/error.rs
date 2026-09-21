#[derive(Debug, thiserror::Error)]
pub enum LinalgError {
    #[error("Invalid input: {0}")]
    InvalidInput(String),

    #[error(
        "Hessian matrix is not positive definite (minimum eigenvalue: {min_eigenvalue:.4e}). This indicates a numerical instability."
    )]
    HessianNotPositiveDefinite { min_eigenvalue: f64 },

    /// A factorization lost rank exactly at one coordinate, with the pivot the
    /// factorization read there. The caller decides whether the operator it
    /// handed in carried a penalty (#4468).
    #[error(
        "The {context} factor is singular at coordinate {index}: its pivot is {pivot:.3e}."
    )]
    SingularFactorPivot {
        context: &'static str,
        index: usize,
        pivot: f64,
    },

    /// A pivot test failed on an operator the caller declares PENALIZED, so a
    /// heavier penalty can make the same factorization succeed (#4468).
    #[error(
        "The {context} did not factor at this smoothing strength: a pivot did not clear \
         the factorization's own accumulated rounding."
    )]
    PenalizedPivotUnresolvedAtRho { context: &'static str },

    /// A fill-reducing ordering or a symbolic Cholesky failed. Both read the
    /// sparsity pattern and no numerical value, so neither is a statement
    /// about conditioning (#4468).
    #[error("The sparse factorization's {stage} stage failed on the sparsity pattern.")]
    SymbolicFactorizationFailed { stage: &'static str },
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
        let err = LinalgError::HessianNotPositiveDefinite {
            min_eigenvalue: -0.001,
        };
        assert!(err.to_string().to_lowercase().contains("positive definite"));
    }

    /// Each variant names the thing that failed rather than a condition number
    /// no producer computes (#4468).
    #[test]
    fn each_factorization_failure_names_what_failed() {
        let pivot = LinalgError::SingularFactorPivot {
            context: "sparse exact Cholesky",
            index: 7,
            pivot: 0.0,
        };
        let text = pivot.to_string();
        assert!(text.contains("coordinate 7"), "{text}");
        assert!(text.contains("sparse exact Cholesky"), "{text}");

        let penalized = LinalgError::PenalizedPivotUnresolvedAtRho {
            context: "sparse exact penalized Cholesky",
        };
        assert!(
            penalized.to_string().contains("smoothing strength"),
            "{penalized}"
        );

        let symbolic = LinalgError::SymbolicFactorizationFailed {
            stage: "fill-reducing ordering",
        };
        let text = symbolic.to_string();
        assert!(text.contains("fill-reducing ordering"), "{text}");
        assert!(text.contains("sparsity pattern"), "{text}");
    }
}
