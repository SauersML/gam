//! Linear algebra helpers for `gam`: faer/ndarray bridges, matrix operators,
//! sparse solves, iterative solvers, and numerical stability utilities.

#[macro_export]
macro_rules! impl_reason_error_boilerplate {
    ($type:ident { $($variant:ident),+ $(,)? }) => {
        impl ::std::fmt::Display for $type {
            fn fmt(&self, f: &mut ::std::fmt::Formatter<'_>) -> ::std::fmt::Result {
                match self {
                    $(Self::$variant { reason })|+ => f.write_str(reason),
                }
            }
        }

        impl ::std::error::Error for $type {}

        impl From<$type> for String {
            fn from(err: $type) -> String {
                err.to_string()
            }
        }
    };
}

pub mod decision;
pub mod dense;
mod error;
pub mod faer_ndarray;
pub mod gaussian_weighted_ridge_backward;
pub mod gpu_hook;
pub mod lanczos;
pub mod low_rank_weight;
pub mod matrix;
pub mod pairwise_reduce;
pub mod parallel;
pub mod pcg;
pub mod psd_trust_region;
pub mod sparse_exact;
pub mod test_support;
pub mod triangular;
pub mod types;
pub mod utils;

pub use error::LinalgError;
pub use types::{RidgeDeterminantMode, RidgePolicy};
