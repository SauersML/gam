//! Outer-score row subsampling — re-exported from `gam-problem`.
//!
//! The canonical definitions of [`OuterScoreSubsample`], [`RowSet`], and
//! [`WeightedOuterRow`] are neutral low-layer primitives that live in the
//! `gam-problem` crate so the `CustomFamily` trait layer (and the model
//! families above it) can depend on them downward without duplication. This
//! module is a stable re-export so existing `gam_models::outer_subsample::*`
//! and `crate::outer_subsample::*` paths keep resolving.

pub use gam_problem::outer_subsample::*;
