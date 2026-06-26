//! The `PenaltyMatrix` carrier (dense / Kronecker / scaled) used by every
//! custom-family block, plus its constructors and the `Array2` conversion.
//!
//! The type itself now lives in the neutral `gam-problem` contract crate (it is
//! a pure ndarray carrier with no upward dependencies, the lone exception being
//! the Kronecker materialization which is satisfied by a `gam-problem`-local
//! helper). It is re-exported here so every in-crate path keeps resolving
//! `crate::custom_family::PenaltyMatrix` unchanged.

pub use gam_problem::PenaltyMatrix;
