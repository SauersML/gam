//! Identifiability: one coherent home for every identifiability concern in the
//! crate.
//!
//! - [`kernel`] — the low-level rank / null-space kernels.
//! - [`precondition`] — cheap pre-fit precondition checks.
//! - [`audit`] — the joint cross-block identifiability audit.
//! - [`canonical`] — canonicalization of specs for identifiability.
//! - [`families`] — the family-agnostic block compiler, its GPU paths, and the
//!   per-family row-Hessian implementations.
//! - [`marginal_slope`] — survival marginal-slope identifiability.

pub mod audit;
pub mod canonical;
pub mod families;
pub mod kernel;
pub mod marginal_slope;
pub mod precondition;

#[cfg(test)]
pub(crate) mod tests_common {
    pub(crate) fn spec_from_dense(
        name: &str,
        design: ndarray::Array2<f64>,
    ) -> crate::families::custom_family::ParameterBlockSpec {
        let n = design.nrows();
        crate::families::custom_family::ParameterBlockSpec {
            name: name.to_string(),
            design: crate::linalg::matrix::DesignMatrix::Dense(
                crate::linalg::matrix::DenseDesignMatrix::from(design),
            ),
            offset: ndarray::Array1::<f64>::zeros(n),
            penalties: Vec::new(),
            nullspace_dims: Vec::new(),
            initial_log_lambdas: ndarray::Array1::<f64>::zeros(0),
            initial_beta: None,
            gauge_priority: 100,
            jacobian_callback: None,
            stacked_design: None,
            stacked_offset: None,
        }
    }

    pub(crate) fn linspace(n: usize) -> ndarray::Array1<f64> {
        if n <= 1 {
            return ndarray::Array1::<f64>::zeros(n.max(1));
        }
        let step = 2.0 / (n as f64 - 1.0);
        ndarray::Array1::from_iter((0..n).map(|i| -1.0 + step * i as f64))
    }
}
