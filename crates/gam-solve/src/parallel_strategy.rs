//! Workload-aware parallel scheduling for row reductions.
//!
//! This module deliberately does not size or replace Rayon's global pool.
//! The pool is process infrastructure; each numerical kernel still owns the
//! decision of how much parallelism is useful for its arithmetic shape.

pub use gam_linalg::parallel::{row_reduction_chunk_count, row_reduction_chunk_rows};
