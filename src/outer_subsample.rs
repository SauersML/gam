//! Outer-loop row subsampling primitive — re-exported from `gam-problem`.
//!
//! The canonical definitions of [`OuterScoreSubsample`], [`WeightedOuterRow`],
//! [`RowSet`], and the deterministic row-tiling helpers (`ARROW_ROW_CHUNK`,
//! `arrow_row_chunk_count`) now live in the neutral `gam-problem` crate so the
//! `CustomFamily` trait layer can depend on them downward without duplication.
//! This module is a stable re-export so existing `gam::outer_subsample::*` and
//! `crate::outer_subsample::*` paths keep resolving.

pub use gam_problem::outer_subsample::*;
