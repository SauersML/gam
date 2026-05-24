//! GPU kernel implementations. Each submodule provides a single fused
//! kernel that the dispatch layer can route from CPU-equivalent code.
//! `BackendPhaseMarker` is retained as the public marker the HAL surface
//! uses to identify the kernel-set version.

pub mod cell_moments;
pub mod fused_xtwx;
pub mod hutchpp;
pub mod irls_link;
pub mod reductions;
pub mod row_scale;
pub mod spatial;

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct BackendPhaseMarker;
