//! Neutral programmatic prior-mean type for a coefficient penalty block.
//!
//! Lives in `gam-problem` so the penalty contract can carry a centering vector
//! without depending on `solver`'s `EstimationError`. Evaluation failures are
//! reported through the neutral [`PriorMeanError`]; callers map this into their
//! own error flow (e.g. `EstimationError::InvalidInput`).

use std::sync::Arc;

use ndarray::Array1;

/// Neutral error for prior-mean evaluation failures.
///
/// Carries the human-readable message; callers in the solver crate map this
/// into `EstimationError::InvalidInput` to preserve end-to-end behavior.
#[derive(Debug, Clone)]
pub struct PriorMeanError(pub String);

impl std::fmt::Display for PriorMeanError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(&self.0)
    }
}

impl std::error::Error for PriorMeanError {}

/// Programmatic prior mean for a coefficient penalty block.
///
/// The mean is evaluated once during penalty canonicalization and then enters
/// the solver as the centering vector in `(beta - mean)' S (beta - mean)`.
#[derive(Clone, Default)]
pub enum CoefficientPriorMean {
    #[default]
    Zero,
    Scalar(f64),
    Constant(Array1<f64>),
    Functional {
        metadata: Array1<f64>,
        evaluator: Arc<dyn Fn(&Array1<f64>) -> Array1<f64> + Send + Sync>,
    },
    /// Covariate-functional mean `mu(a) = amplitude * K(a)` for a coefficient block.
    ///
    /// Formula-level coefficient groups pass their row/covariate metadata as
    /// `covariates`; the user-supplied kernel returns the block-sized basis
    /// vector `K(a)` and the scalar amplitude supplies `alpha`.
    KernelBasis {
        covariates: Array1<f64>,
        amplitude: f64,
        kernel: Arc<dyn Fn(&Array1<f64>) -> Array1<f64> + Send + Sync>,
    },
}

impl std::fmt::Debug for CoefficientPriorMean {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Zero => f.write_str("Zero"),
            Self::Scalar(value) => f.debug_tuple("Scalar").field(value).finish(),
            Self::Constant(values) => f
                .debug_tuple("Constant")
                .field(&format_args!("len={}", values.len()))
                .finish(),
            Self::Functional { metadata, .. } => f
                .debug_struct("Functional")
                .field("metadata_len", &metadata.len())
                .finish_non_exhaustive(),
            Self::KernelBasis {
                covariates,
                amplitude,
                ..
            } => f
                .debug_struct("KernelBasis")
                .field("covariate_len", &covariates.len())
                .field("amplitude", amplitude)
                .finish_non_exhaustive(),
        }
    }
}

impl CoefficientPriorMean {
    pub const fn scalar(value: f64) -> Self {
        Self::Scalar(value)
    }

    pub fn constant(values: Array1<f64>) -> Self {
        Self::Constant(values)
    }

    pub fn functional(
        metadata: Array1<f64>,
        evaluator: Arc<dyn Fn(&Array1<f64>) -> Array1<f64> + Send + Sync>,
    ) -> Self {
        Self::Functional {
            metadata,
            evaluator,
        }
    }

    pub fn kernel_basis(
        covariates: Array1<f64>,
        amplitude: f64,
        kernel: Arc<dyn Fn(&Array1<f64>) -> Array1<f64> + Send + Sync>,
    ) -> Self {
        Self::KernelBasis {
            covariates,
            amplitude,
            kernel,
        }
    }

    pub fn evaluate(&self, block_dim: usize, context: &str) -> Result<Array1<f64>, PriorMeanError> {
        let values = match self {
            Self::Zero => Array1::zeros(block_dim),
            Self::Scalar(value) => {
                if !value.is_finite() {
                    return Err(PriorMeanError(format!(
                        "{context}: coefficient prior mean scalar must be finite, got {value}"
                    )));
                }
                Array1::from_elem(block_dim, *value)
            }
            Self::Constant(values) => values.clone(),
            Self::Functional {
                metadata,
                evaluator,
            } => evaluator(metadata),
            Self::KernelBasis {
                covariates,
                amplitude,
                kernel,
            } => {
                if !amplitude.is_finite() {
                    return Err(PriorMeanError(format!(
                        "{context}: coefficient prior mean amplitude must be finite, got {amplitude}"
                    )));
                }
                let mut values = kernel(covariates);
                values *= *amplitude;
                values
            }
        };
        if values.len() != block_dim {
            return Err(PriorMeanError(format!(
                "{context}: coefficient prior mean length must be {block_dim}, got {}",
                values.len()
            )));
        }
        if values.iter().any(|&value| !value.is_finite()) {
            return Err(PriorMeanError(format!(
                "{context}: coefficient prior mean contains non-finite values"
            )));
        }
        Ok(values)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::array;

    #[test]
    fn zero_variant_returns_zeros_of_requested_len() {
        let m = CoefficientPriorMean::Zero;
        let v = m.evaluate(4, "ctx").unwrap();
        assert_eq!(v.len(), 4);
        assert!(v.iter().all(|&x| x == 0.0));
    }

    #[test]
    fn scalar_fills_vector_with_constant() {
        let m = CoefficientPriorMean::scalar(3.0);
        let v = m.evaluate(3, "ctx").unwrap();
        assert_eq!(v.len(), 3);
        assert!(v.iter().all(|&x| x == 3.0));
    }

    #[test]
    fn scalar_nan_returns_error() {
        let m = CoefficientPriorMean::scalar(f64::NAN);
        assert!(m.evaluate(2, "ctx").is_err());
    }

    #[test]
    fn scalar_infinite_returns_error() {
        let m = CoefficientPriorMean::scalar(f64::INFINITY);
        assert!(m.evaluate(2, "ctx").is_err());
    }

    #[test]
    fn constant_variant_clones_vector() {
        let arr = array![1.0_f64, 2.0, 3.0];
        let m = CoefficientPriorMean::constant(arr.clone());
        let v = m.evaluate(3, "ctx").unwrap();
        assert_eq!(v, arr);
    }

    #[test]
    fn constant_dimension_mismatch_returns_error() {
        let arr = array![1.0_f64, 2.0];
        let m = CoefficientPriorMean::constant(arr);
        assert!(m.evaluate(5, "ctx").is_err());
    }

    #[test]
    fn functional_variant_calls_evaluator() {
        let meta = array![0.0_f64];
        let m = CoefficientPriorMean::functional(meta, Arc::new(|_| array![7.0_f64, 8.0]));
        let v = m.evaluate(2, "ctx").unwrap();
        assert_eq!(v[0], 7.0);
        assert_eq!(v[1], 8.0);
    }

    #[test]
    fn kernel_basis_scales_kernel_output() {
        let covs = array![0.0_f64];
        let m = CoefficientPriorMean::kernel_basis(covs, 2.0, Arc::new(|_| array![1.0_f64, 3.0]));
        let v = m.evaluate(2, "ctx").unwrap();
        assert!((v[0] - 2.0).abs() < 1e-14);
        assert!((v[1] - 6.0).abs() < 1e-14);
    }

    #[test]
    fn kernel_basis_nan_amplitude_returns_error() {
        let covs = array![0.0_f64];
        let m = CoefficientPriorMean::kernel_basis(covs, f64::NAN, Arc::new(|_| array![1.0_f64]));
        assert!(m.evaluate(1, "ctx").is_err());
    }

    #[test]
    fn default_is_zero_variant() {
        let m = CoefficientPriorMean::default();
        let v = m.evaluate(5, "ctx").unwrap();
        assert!(v.iter().all(|&x| x == 0.0));
    }

    #[test]
    fn error_message_includes_context() {
        let m = CoefficientPriorMean::scalar(f64::NAN);
        let err = m.evaluate(1, "myctx").unwrap_err();
        let msg = err.to_string();
        assert!(msg.contains("myctx"), "error should mention context: {msg}");
    }
}
