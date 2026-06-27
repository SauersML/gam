//! Quantile / order-statistic helpers.
//!
//! These are single-sourced in the `gam-math` crate (the neutral numeric
//! layer shared by the sibling crates that consume them). This module
//! re-exports them so existing `crate::util::quantile::*` paths keep resolving
//! without a second, drift-prone copy of the definitions.
pub use gam_math::quantile::*;
