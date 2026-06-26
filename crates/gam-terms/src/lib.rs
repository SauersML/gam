#[macro_export]
macro_rules! bail_invalid_basis {
    ($fmt:literal $(, $($arg:tt)*)?) => {
        return Err($crate::basis::BasisError::InvalidInput(format!($fmt $(, $($arg)*)?)))
    };
    ($msg:expr $(,)?) => {
        return Err($crate::basis::BasisError::InvalidInput($msg))
    };
}

#[macro_export]
macro_rules! bail_dim_basis {
    ($fmt:literal $(, $($arg:tt)*)?) => {
        return Err($crate::basis::BasisError::DimensionMismatch(format!($fmt $(, $($arg)*)?)))
    };
    ($msg:expr $(,)?) => {
        return Err($crate::basis::BasisError::DimensionMismatch($msg))
    };
}

pub mod analytic_penalties;
pub mod basis;
pub mod chunked_kernel_design;
pub mod dictionary;
pub mod geometry;
pub mod kronecker;
pub mod latent;

#[macro_export]
macro_rules! gpu_bail {
    ($($arg:tt)*) => {
        return ::std::result::Result::Err(gam_gpu::gpu_error::GpuError::DriverCallFailed {
            reason: ::std::format!($($arg)*),
        })
    };
}

pub mod construction {
    pub use crate::kronecker::{KroneckerInvariantStructure, kronecker_product};
}

pub mod terms {
    pub use crate::*;
}
