//! Basis-facing analytic penalty operator surface.
//!
//! The full analytic-penalty registry still depends on higher SAE/smooth wiring.
//! `gam-terms` exposes the lower operator abstraction needed by basis assembly
//! without compiling those upper modules into the lower crate.

mod op;

pub use self::op::{PenaltyOp, ScaledPenaltyOp};
