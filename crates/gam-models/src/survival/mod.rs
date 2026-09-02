//! Survival families and survival-specific support code.
//!
//! Each submodule owns one concern; survival code is addressed through
//! `crate::survival::{submodule}` or through this module root for the
//! intentionally flattened survival-family surface.
//!
//! - [`base`]              — the core survival family (formerly `survival.rs`).
//! - [`construction`]      — survival time-basis and baseline construction.
//! - [`location_scale`]    — location-scale survival family.
//! - [`marginal_slope`]    — survival marginal-slope family.
//! - [`predict`]           — library-side survival prediction.
//! - [`risk_calibration`]  — risk score -> absolute survival surface.
//! - [`latent`]            — latent survival and binary deployment families.
//! - [`royston_parmar`]    — Royston-Parmar survival helpers.

pub mod base;
pub mod construction;
pub use construction::*;
pub mod latent;
pub mod location_scale;
pub mod lognormal_kernel;
pub mod marginal_slope;
pub mod predict;
pub mod risk_calibration;
pub mod royston_parmar;
pub mod surface;
pub(crate) mod time_constraints;

pub use base::*;
pub use latent::*;
pub use location_scale::*;
pub use lognormal_kernel::*;
pub use marginal_slope::*;
pub use predict::*;
pub use risk_calibration::*;
pub use royston_parmar::*;
pub use surface::*;
