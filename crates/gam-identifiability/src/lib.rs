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

#[macro_export]
macro_rules! bail_dim_custom {
    ($fmt:literal $(, $($arg:tt)*)?) => {
        return Err(gam_problem::CustomFamilyError::DimensionMismatch {
            reason: format!($fmt $(, $($arg)*)?),
        })
    };
    ($msg:expr $(,)?) => {
        return Err(gam_problem::CustomFamilyError::DimensionMismatch { reason: $msg })
    };
}

pub mod audit;
pub mod canonical;
pub mod families;
pub mod kernel;
pub mod marginal_slope;
pub mod precondition;
