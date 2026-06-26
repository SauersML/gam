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
