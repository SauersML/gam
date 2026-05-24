//! CPU-safe placeholder module for the GPU backend.
//!
//! Concrete CUDA resources are introduced behind operation-level dispatch so
//! CPU-only builds keep identical behavior and do not require CUDA libraries.

/// Describes whether a value is host-resident or intended for device residency.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum Residency {
    Host,
    Device,
}
