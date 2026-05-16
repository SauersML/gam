//! Public dispatch entry points for hot dense linalg kernels.

pub use super::blas::{
    try_fast_ab, try_fast_atb, try_fast_atv, try_fast_av, try_fast_xt_diag_x, try_fast_xt_diag_y,
};
