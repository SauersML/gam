mod error;
pub mod faer_ndarray;
pub(crate) mod lanczos;
pub mod low_rank_weight;
pub mod matrix;
pub mod pairwise_reduce;
pub(crate) mod pcg;
pub mod sparse_exact;
pub mod triangular;
pub mod utils;

pub use error::LinalgError;
