//! Joint (cross-block) penalty specifications — re-exported from `gam-problem`.
//!
//! The canonical definitions of [`JointPenaltySpec`], [`JointPenaltyBundle`],
//! and [`JointPenaltyError`] now live in the neutral `gam-problem` crate so the
//! `CustomFamily` trait layer can depend on them downward without duplication.
//! This module is a stable re-export so existing `gam_models::joint_penalty::*`
//! and `crate::joint_penalty::*` paths keep resolving.

pub use gam_problem::joint_penalty::*;
