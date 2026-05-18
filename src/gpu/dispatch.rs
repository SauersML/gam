//! Public dispatch entry points for hot dense linalg kernels.

pub use super::blas::{
    try_fast_ab, try_fast_ab_strided_batched, try_fast_atb, try_fast_atb_strided_batched,
    try_fast_atv, try_fast_av, try_fast_xt_diag_x, try_fast_xt_diag_y,
    try_solve_lower_triangular_matrix, try_solve_upper_triangular_matrix,
};
