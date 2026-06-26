//! Span / breakpoint selection helper.
//!
//! Single-sourced in the `gam-runtime` crate (the runtime layer shared by the
//! sibling crates that consume it). This module re-exports it so existing
//! `crate::util::span::*` paths keep resolving without a second, drift-prone
//! copy of the definition.
pub(crate) use gam_runtime::span::*;
