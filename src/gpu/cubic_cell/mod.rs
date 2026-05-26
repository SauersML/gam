//! GPU substrate for de-nested cubic-cell **derivative moments**.
//!
//! This module is the shared GPU evaluator for the de-nested cubic transport
//! kernel that currently lives in `src/families/cubic_cell_kernel.rs`. For
//! each partition cell `(left, right, c_0, c_1, c_2, c_3)` it computes the
//! derivative-moment vector
//!
//! ```text
//!   M_k = ∫_{left}^{right} z^k · exp(-q(z)) dz,   k = 0..=max_degree,
//!   q(z) = 0.5 · (z² + η(z)²),
//!   η(z) = c_0 + c_1·z + c_2·z² + c_3·z³.
//! ```
//!
//! Three CPU branches feed into the same device API:
//!
//! * **Affine** (`c_2 = c_3 = 0`, finite interval): closed-form via the
//!   `T_n(a,b)` recurrence used by `affine_anchor_moment_vector_into`.
//! * **Non-affine finite**: fixed 384-point Gauss–Legendre on the cell. This
//!   matches the high-degree CPU parity reference and is intentionally NOT
//!   the `reduce_quartic_moments` / `reduce_sextic_moments` recurrence, which
//!   amplifies cancellation at high `max_degree`.
//! * **Affine tail**: closed-form on a semi-infinite (or whole-line) interval.
//!
//! This is **distinct** from `src/gpu/cubic_bspline_moments.rs`, which
//! computes tensor B-spline cell moments. The two modules share neither math
//! nor data layout: do not conflate them.
//!
//! Consumers (`src/gpu/bms_flex.rs`, `src/gpu/survival_flex.rs`) pick a
//! `CubicCellMomentMode` and choose where results materialize via
//! `CubicCellMomentResidency`. Host residency works on every platform today
//! by routing through the CPU evaluator that the device kernel is byte-for-
//! byte parity-tested against (see `host_substrate`). Device residency lands
//! with the NVRTC kernel on Linux+CUDA targets.

pub(crate) mod branch;
pub(crate) mod host_substrate;
pub(crate) mod kernel_src;

use crate::gpu::error::GpuError;

pub(crate) use branch::{classify_cell_for_gpu, classify_cells_for_gpu};
pub(crate) use host_substrate::{HostMomentBatch, build_host_moments};

/// Maximum derivative-moment degree the substrate is built to evaluate.
///
/// Consumers and their high-water marks:
/// * Bernoulli flex Hessian: 9
/// * BMS outer higher-derivative reuse: 21
/// * Survival flex Hessian (with `D_uv` cross terms): 24
pub(crate) const MAX_SUPPORTED_DEGREE: usize = 24;

/// A single de-nested cubic-cell payload in the layout the device kernels
/// consume. Matches the CPU layout in `cubic_cell_kernel.rs`: the cubic
/// correction `η(z) = c_0 + c_1·z + c_2·z² + c_3·z³` evaluated over
/// `[left, right]`.
#[derive(Clone, Copy, Debug, PartialEq)]
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
    /// Semi-infinite (or whole-line) affine tail with `c_2 = c_3 = 0` —
    /// closed-form on the tail interval.
    AffineTail,
}

/// What the caller wants the substrate to compute per cell.
///
/// `ValueAndDerivative` is the survival path (Stage 5) and is not wired
/// yet; the current substrate commits to `DerivativeOnly`.
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

/// Per-cell status code written by the substrate. `Ok` means the
/// corresponding row of `moments` is the requested degree's full moment
/// vector. Any other variant means the row is zeroed and the caller must
/// react (fall back to CPU, surface the failure, etc).
///
/// The numeric values match the device kernel's status code emission so a
/// future GPU landing can fill the same `Vec<u8>` without translation.
#[repr(u8)]
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(crate) enum CubicCellMomentStatus {
    Ok = 0,
    /// Finite cell with `right <= left`, mismatched caller branch tag, or
    /// CPU classifier rejected the cell.
    InvalidInterval = 1,
    /// Semi-infinite cell with material `c_2` or `c_3`.
    NonAffineInfiniteInterval = 2,
    /// At least one of `c_0..c_3` was NaN/Inf.
    NonFiniteCoefficient = 3,
    /// Evaluator produced a non-finite moment (q overflow on a pathological
    /// cell). The row is zeroed; this is the GPU-side counterpart to a CPU
    /// `Err`.
    NonFiniteEvaluation = 4,
}

impl CubicCellMomentStatus {
    #[inline]
    pub(crate) fn from_byte(byte: u8) -> Option<Self> {
        match byte {
            0 => Some(Self::Ok),
            1 => Some(Self::InvalidInterval),
            2 => Some(Self::NonAffineInfiniteInterval),
            3 => Some(Self::NonFiniteCoefficient),
            4 => Some(Self::NonFiniteEvaluation),
            _ => None,
        }
    }
}

/// Host-side input view for `try_build_cubic_cell_derivative_moments`.
///
/// The substrate borrows cell data from the caller; it does not own the
/// CPU partition. `branches` is parallel to `cells` and is produced by the
/// CPU classifier via [`classify_cells_for_gpu`] — callers that already
/// have an `ExactCellBranch` from `cubic_cell_kernel` can map it directly
/// (`Affine -> Affine` for finite, `Affine -> AffineTail` for semi-
/// infinite, `Quartic | Sextic -> NonAffineFinite`).
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
    /// Row-major `[n_cells, max_degree + 1]` host buffer + per-cell status
    /// codes. The row for cell `i` is `moments[i * stride ..][..stride]`
    /// where `stride = max_degree + 1`. Rows for non-OK cells are zeroed.
    Host {
        moments: Vec<f64>,
        status: Vec<u8>,
        stride: usize,
    },
    /// Opaque device handle for BMS flex / survival flex. Concrete device
    /// buffer wiring lands with the NVRTC kernels.
    Device(DeviceCubicCellMomentBatch),
}

/// Opaque handle to a device-resident moment batch. Populated by the
/// device kernel landings.
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

    fn note(&mut self, n_cells: usize, stride: usize) {
        self.capacity_cells = self.capacity_cells.max(n_cells);
        self.capacity_stride = self.capacity_stride.max(stride);
    }
}

/// Try to build derivative moments via the substrate.
///
/// On `Host` residency: routes through the CPU evaluator (parity reference
/// for the device kernel) and returns real moments + per-cell status. The
/// host path works on every platform.
///
/// On `Device` residency: returns `Err(GpuError::NotYetImplemented)` until
/// the NVRTC kernel ships on Linux+CUDA. Until then, callers that strictly
/// require device residency must surface the error and fall back to host.
///
/// Returns `Ok(None)` when the workload is empty (`cells.is_empty()`).
pub(crate) fn try_build_cubic_cell_derivative_moments(
    input: CubicCellDerivativeMomentHostView<'_>,
    workspace: Option<&mut CubicCellMomentWorkspace>,
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
    if input.cells.is_empty() {
        return Ok(None);
    }

    let stride = input.max_degree + 1;
    if let Some(ws) = workspace {
        ws.note(input.cells.len(), stride);
    }

    match input.residency {
        CubicCellMomentResidency::Host => match build_host_moments(&input) {
            Ok(batch) => Ok(Some(CubicCellDerivativeMomentOutput::Host {
                moments: batch.moments,
                status: batch.status,
                stride: batch.stride,
            })),
            Err(reason) => Err(GpuError::NotYetImplemented { reason }),
        },
        CubicCellMomentResidency::Device => Err(GpuError::NotYetImplemented {
            reason: "gpu cubic-cell substrate: device residency lands with the NVRTC kernel \
                     (see src/gpu/cubic_cell/kernel_src.rs)"
                .to_string(),
        }),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn host_residency_returns_real_moments() {
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
        let out = try_build_cubic_cell_derivative_moments(view, None)
            .expect("host residency works")
            .expect("non-empty input produces output");
        match out {
            CubicCellDerivativeMomentOutput::Host {
                moments,
                status,
                stride,
            } => {
                assert_eq!(stride, 10);
                assert_eq!(status, vec![CubicCellMomentStatus::Ok as u8]);
                assert_eq!(moments.len(), stride);
                // M_0 for η ≡ 0 over [-1,1] is ∫_{-1}^{1} exp(-z²/2) dz =
                // sqrt(2π)·(Φ(1)-Φ(-1)) ≈ 1.7112488348667447.
                assert!((moments[0] - 1.7112488348667447).abs() < 1e-12);
                assert!(moments[1].abs() < 1e-12);
                assert!(moments[3].abs() < 1e-12);
            }
            CubicCellDerivativeMomentOutput::Device(_) => panic!("expected host output"),
        }
    }

    #[test]
    fn device_residency_returns_not_yet_implemented() {
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
            residency: CubicCellMomentResidency::Device,
        };
        let err = try_build_cubic_cell_derivative_moments(view, None).unwrap_err();
        let msg = err.to_string();
        assert!(
            msg.contains("device residency"),
            "expected device-residency hint, got: {msg}"
        );
    }

    #[test]
    fn empty_input_returns_ok_none() {
        let view = CubicCellDerivativeMomentHostView {
            cells: &[],
            branches: &[],
            max_degree: 9,
            mode: CubicCellMomentMode::DerivativeOnly,
            residency: CubicCellMomentResidency::Host,
        };
        let out = try_build_cubic_cell_derivative_moments(view, None).expect("ok");
        assert!(out.is_none());
    }

    #[test]
    fn rejects_mismatched_lengths() {
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
    fn rejects_degree_above_supported_max() {
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
    fn workspace_grows_to_largest_capacity_seen() {
        let mut ws = CubicCellMomentWorkspace::new();
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
            max_degree: 21,
            mode: CubicCellMomentMode::DerivativeOnly,
            residency: CubicCellMomentResidency::Host,
        };
        try_build_cubic_cell_derivative_moments(view, Some(&mut ws)).expect("ok");
        assert!(ws.capacity_cells >= 1);
        assert!(ws.capacity_stride >= 22);
    }

    #[test]
    fn workspace_new_starts_at_zero_capacity() {
        let ws = CubicCellMomentWorkspace::new();
        assert_eq!(ws.capacity_cells, 0);
        assert_eq!(ws.capacity_stride, 0);
    }

    #[test]
    fn status_byte_roundtrip() {
        for s in [
            CubicCellMomentStatus::Ok,
            CubicCellMomentStatus::InvalidInterval,
            CubicCellMomentStatus::NonAffineInfiniteInterval,
            CubicCellMomentStatus::NonFiniteCoefficient,
            CubicCellMomentStatus::NonFiniteEvaluation,
        ] {
            assert_eq!(CubicCellMomentStatus::from_byte(s as u8), Some(s));
        }
        assert_eq!(CubicCellMomentStatus::from_byte(99), None);
    }
}
