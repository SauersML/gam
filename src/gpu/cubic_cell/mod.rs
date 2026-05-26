//! GPU substrate for de-nested cubic-cell **derivative moments**.
//!
//! This module is the shared GPU evaluator for the de-nested cubic transport
//! kernel that currently lives in `src/families/cubic_cell_kernel.rs`. For
//! each partition cell `(left, right, c_0, c_1, c_2, c_3)` it computes the
//! derivative-moment vector
//!
//! ```text
//!   M_k = ∫_{left}^{right} z^k · φ(c(z)) · φ(z) dz,   k = 0..=max_degree
//! ```
//!
//! where `c(z) = c_0 + c_1·z + c_2·z² + c_3·z³` is the cell's cubic
//! correction. Three CPU branches feed into the same device API:
//!
//! * **Affine** (`c_2 = c_3 = 0`): closed-form via the `T_n(a,b)` recurrence
//!   used by `affine_anchor_moment_vector_into`.
//! * **Non-affine finite**: fixed 384-point Gauss–Legendre on the cell. This
//!   matches the high-degree CPU parity reference and is intentionally NOT
//!   the `reduce_quartic_moments` / `reduce_sextic_moments` recurrence, which
//!   amplifies cancellation at high `max_degree`.
//! * **Affine tail**: closed-form on a semi-infinite interval.
//!
//! This is **distinct** from `src/gpu/cubic_bspline_moments.rs`, which
//! computes tensor B-spline cell moments. The two modules share neither math
//! nor data layout: do not conflate them.
//!
//! Consumers (BMS flex `src/gpu/bms_flex.rs`, survival flex
//! `src/gpu/survival_flex.rs`) pick a `CubicCellMomentMode` and choose where
//! results materialize (`Host`, `Device`, `Both`). `Both` is only emitted
//! once a real consumer asks for it.

use crate::gpu::error::GpuError;

/// Maximum derivative-moment degree the substrate is built to evaluate.
///
/// Consumers and their high-water marks:
/// * Bernoulli flex Hessian: 9
/// * BMS outer higher-derivative reuse: 21
/// * Survival flex Hessian (with `D_uv` cross terms): 24
pub(crate) const MAX_SUPPORTED_DEGREE: usize = 24;

/// A single de-nested cubic-cell payload in the layout the device kernels
/// consume. Matches the CPU layout in `cubic_cell_kernel.rs`: the cubic
/// correction `c(z) = c_0 + c_1·z + c_2·z² + c_3·z³` evaluated over
/// `[left, right]`.
#[derive(Clone, Copy, Debug)]
pub(crate) struct GpuDenestedCubicCell {
    pub left: f64,
    pub right: f64,
    pub c0: f64,
    pub c1: f64,
    pub c2: f64,
    pub c3: f64,
}

/// Branch classification for a single cell, as decided by the CPU
/// classifier. The device dispatcher buckets cells by tag and launches one
/// specialized kernel per branch to avoid warp divergence.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(crate) enum GpuCellBranchTag {
    /// `c_2 = c_3 = 0` and the interval is finite — closed-form `T_n`
    /// recurrence at the affine anchor.
    Affine,
    /// Finite interval with at least one of `c_2`, `c_3` non-zero — fixed
    /// 384-point Gauss–Legendre on the cell.
    NonAffineFinite,
    /// Semi-infinite affine tail (one endpoint at ±∞, `c_2 = c_3 = 0`) —
    /// closed-form on the tail interval.
    AffineTail,
}

/// What the caller wants the substrate to compute per cell.
///
/// `ValueAndDerivative` is the survival path (Stage 5) and is not wired
/// yet; the Stage 1 substrate only commits to `DerivativeOnly`.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(crate) enum CubicCellMomentMode {
    DerivativeOnly,
    ValueAndDerivative,
}

/// Where the caller wants results materialized.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(crate) enum CubicCellMomentResidency {
    Host,
    Device,
}

/// Host-side input view for `try_build_cubic_cell_derivative_moments`.
///
/// The substrate borrows cell data from the caller; it does not own the
/// CPU partition. `branches` is parallel to `cells` and is produced by the
/// CPU classifier (see Stage 1 commit 2).
pub(crate) struct CubicCellDerivativeMomentHostView<'a> {
    pub cells: &'a [GpuDenestedCubicCell],
    pub branches: &'a [GpuCellBranchTag],
    pub max_degree: usize,
    pub mode: CubicCellMomentMode,
    pub residency: CubicCellMomentResidency,
}

/// Output of `try_build_cubic_cell_derivative_moments`. The variants line
/// up with `CubicCellMomentResidency` so the substrate does not silently
/// materialize a host copy the caller did not ask for.
pub(crate) enum CubicCellDerivativeMomentOutput {
    /// Row-major `[n_cells, max_degree + 1]` host buffer.
    Host(Vec<f64>),
    /// Opaque device handle for Stage 3+ BMS flex / Stage 5 survival flex.
    /// Concrete device buffer wiring lands with the NVRTC kernels.
    Device(DeviceCubicCellMomentBatch),
}

/// Opaque handle to a device-resident moment batch. Populated by the
/// device kernel landings in commits 3–6.
pub(crate) struct DeviceCubicCellMomentBatch {
    pub n_cells: usize,
    pub moment_stride: usize,
}

/// Workspace for the substrate. Holds pinned host staging + persistent
/// device buffers that the caller can reuse across launches. Until the
/// device kernels land, the workspace is a placeholder.
pub(crate) struct CubicCellMomentWorkspace {
    /// Largest `n_cells` seen so far — used to size persistent buffers.
    pub capacity_cells: usize,
    /// Largest `max_degree + 1` seen so far.
    pub capacity_stride: usize,
}

impl CubicCellMomentWorkspace {
    pub(crate) const fn new() -> Self {
        Self {
            capacity_cells: 0,
            capacity_stride: 0,
        }
    }
}

/// Try to build derivative moments on the GPU.
///
/// Returns `Ok(None)` when the substrate decides the workload is too small
/// or the device is unavailable for this launch (caller must fall back to
/// CPU). Returns `Err(GpuError::NotYetImplemented)` until the NVRTC kernels
/// in commits 3–6 land.
pub(crate) fn try_build_cubic_cell_derivative_moments(
    input: CubicCellDerivativeMomentHostView<'_>,
    _workspace: Option<&mut CubicCellMomentWorkspace>,
) -> Result<Option<CubicCellDerivativeMomentOutput>, GpuError> {
    if input.cells.len() != input.branches.len() {
        return Err(GpuError::NotYetImplemented {
            reason: format!(
                "gpu cubic-cell substrate: cells.len()={} != branches.len()={}",
                input.cells.len(),
                input.branches.len()
            ),
        });
    }
    if input.max_degree > MAX_SUPPORTED_DEGREE {
        return Err(GpuError::NotYetImplemented {
            reason: format!(
                "gpu cubic-cell substrate: max_degree={} exceeds MAX_SUPPORTED_DEGREE={}",
                input.max_degree, MAX_SUPPORTED_DEGREE
            ),
        });
    }
    if matches!(input.mode, CubicCellMomentMode::ValueAndDerivative) {
        return Err(GpuError::NotYetImplemented {
            reason: "gpu cubic-cell substrate: ValueAndDerivative mode lands with survival flex \
                     Stage 5"
                .to_string(),
        });
    }
    Err(GpuError::NotYetImplemented {
        reason: "gpu cubic-cell derivative-moment substrate: kernels land in Stage 1 commits 3-6"
            .to_string(),
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn stub_reports_not_yet_implemented() {
        let cells = [GpuDenestedCubicCell {
            left: -1.0,
            right: 1.0,
            c0: 0.0,
            c1: 0.0,
            c2: 0.0,
            c3: 0.0,
        }];
        let branches = [GpuCellBranchTag::Affine];
        let view = CubicCellDerivativeMomentHostView {
            cells: &cells,
            branches: &branches,
            max_degree: 9,
            mode: CubicCellMomentMode::DerivativeOnly,
            residency: CubicCellMomentResidency::Host,
        };
        let err = try_build_cubic_cell_derivative_moments(view, None).unwrap_err();
        match err {
            GpuError::NotYetImplemented { reason } => {
                assert!(
                    reason.contains("Stage 1"),
                    "expected Stage 1 milestone hint, got: {reason}"
                );
            }
            other => panic!("expected NotYetImplemented, got {other:?}"),
        }
    }

    #[test]
    fn stub_rejects_mismatched_lengths() {
        let cells = [GpuDenestedCubicCell {
            left: 0.0,
            right: 1.0,
            c0: 0.0,
            c1: 0.0,
            c2: 0.0,
            c3: 0.0,
        }];
        let branches: [GpuCellBranchTag; 0] = [];
        let view = CubicCellDerivativeMomentHostView {
            cells: &cells,
            branches: &branches,
            max_degree: 9,
            mode: CubicCellMomentMode::DerivativeOnly,
            residency: CubicCellMomentResidency::Host,
        };
        let err = try_build_cubic_cell_derivative_moments(view, None).unwrap_err();
        let msg = err.to_string();
        assert!(msg.contains("cells.len()"), "got: {msg}");
        assert!(msg.contains("branches.len()"), "got: {msg}");
    }

    #[test]
    fn stub_rejects_degree_above_supported_max() {
        let cells = [GpuDenestedCubicCell {
            left: 0.0,
            right: 1.0,
            c0: 0.0,
            c1: 0.0,
            c2: 0.0,
            c3: 0.0,
        }];
        let branches = [GpuCellBranchTag::Affine];
        let view = CubicCellDerivativeMomentHostView {
            cells: &cells,
            branches: &branches,
            max_degree: MAX_SUPPORTED_DEGREE + 1,
            mode: CubicCellMomentMode::DerivativeOnly,
            residency: CubicCellMomentResidency::Host,
        };
        let err = try_build_cubic_cell_derivative_moments(view, None).unwrap_err();
        assert!(err.to_string().contains("MAX_SUPPORTED_DEGREE"));
    }

    #[test]
    fn workspace_new_starts_at_zero_capacity() {
        let ws = CubicCellMomentWorkspace::new();
        assert_eq!(ws.capacity_cells, 0);
        assert_eq!(ws.capacity_stride, 0);
    }
}
