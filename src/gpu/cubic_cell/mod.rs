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
//! Three branches feed into the same device API:
//!
//! * **Affine** (`c_2 = c_3 = 0`, finite interval): closed-form via the
//!   `T_n(a,b)` recurrence used by `affine_anchor_moment_vector_into`.
//! * **Non-affine finite**: fixed 384-point Gauss–Legendre on the cell.
//! * **Affine tail**: closed-form on a semi-infinite (or whole-line) interval.
//!
//! This is **distinct** from `src/gpu/cubic_bspline_moments.rs`, which
//! computes tensor B-spline cell moments. The two modules share neither math
//! nor data layout: do not conflate them.
//!
//! ## Layout
//!
//! * [`branch`] — host-side branch classifier; mirrors
//!   `cubic_cell_kernel::branch_cell` + the semi-infinite tail logic of
//!   `evaluate_cell_state_dispatched`.
//! * [`host_substrate`] — CPU-resident implementation. Works on every
//!   platform and is the parity reference for the device kernel.
//! * [`kernel_src`] — NVRTC-compilable CUDA C++ source as Rust string
//!   constants (D9 / D15 / D21 specializations).
//! * [`device`] — Linux+CUDA dispatcher that compiles, launches, and
//!   gathers the NVRTC kernel for the NonAffineFinite bucket; Affine /
//!   AffineTail buckets stay on CPU until Stage-2.

pub(crate) mod branch;
pub(crate) mod device;
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

/// Branch classification for a single cell. The device dispatcher buckets
/// cells by tag and launches one specialized kernel per branch to avoid
/// warp divergence.
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
/// `ValueAndDerivative` is the survival path and is not wired yet; the
/// current substrate commits to `DerivativeOnly`.
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

/// Per-cell status code written by the substrate. Numeric values match the
/// device kernel's status code emission so the GPU and host paths fill
/// `Vec<u8>` with the same byte pattern.
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

/// Host-side input view for `try_build_cubic_cell_derivative_moments`.
/// The substrate borrows cell data from the caller; it does not own the
/// CPU partition. `branches` is parallel to `cells`.
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
    /// codes. Row `i` is `moments[i * stride ..][..stride]` where
    /// `stride = max_degree + 1`. Rows for non-OK cells are zeroed.
    Host {
        moments: Vec<f64>,
        status: Vec<u8>,
        stride: usize,
    },
    /// Opaque device handle. Populated when the GPU dispatcher keeps
    /// moments resident on the device (Stage-2 BMS/survival residency).
    Device(DeviceCubicCellMomentBatch),
}

/// Opaque handle to a device-resident moment batch.
pub(crate) struct DeviceCubicCellMomentBatch {
    pub n_cells: usize,
    pub moment_stride: usize,
}

/// Workspace for the substrate. Holds pinned host staging + persistent
/// device buffers that the caller can reuse across launches.
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
/// * `Host` residency: routes through the CPU evaluator (parity reference
///   for the device kernel) and returns real moments + per-cell status on
///   every platform.
/// * `Device` residency: on Linux+CUDA with a probed runtime, the device
///   dispatcher launches the NVRTC kernel for the NonAffineFinite bucket
///   and CPU-evaluates the Affine/AffineTail buckets, packing both back
///   into a `Host { … }` output for the caller. When the runtime is
///   unavailable the caller receives the same `Host { … }` shape via the
///   CPU evaluator — no silent device claim.
///
/// Returns `Ok(None)` only when the workload is empty.
///
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
    if matches!(input.mode, CubicCellMomentMode::ValueAndDerivative) {
        return Err(GpuError::NotYetImplemented {
            reason: "gpu cubic-cell substrate: ValueAndDerivative mode lands with survival flex \
                     Stage 5"
                .to_string(),
        });
    }
    if input.cells.is_empty() {
        return Ok(None);
    }

    let stride = input.max_degree + 1;
    if let Some(ws) = workspace {
        ws.note(input.cells.len(), stride);
    }

    // Device residency tries the GPU dispatcher first; on `Ok(None)`
    // (no runtime) it transparently falls back to the host substrate so
    // callers asking for the device shape never get silent-cpu surprises.
    match input.residency {
        CubicCellMomentResidency::Host => {
            let batch = build_host_moments(&input).map_err(|reason| {
                GpuError::NotYetImplemented { reason }
            })?;
            Ok(Some(into_host_output(batch)))
        }
        CubicCellMomentResidency::Device => match device::try_device_moments(&input)? {
            Some(batch) => Ok(Some(into_host_output(batch))),
            None => {
                let batch = build_host_moments(&input).map_err(|reason| {
                    GpuError::NotYetImplemented { reason }
                })?;
                Ok(Some(into_host_output(batch)))
            }
        },
    }
}

#[inline]
fn into_host_output(batch: HostMomentBatch) -> CubicCellDerivativeMomentOutput {
    CubicCellDerivativeMomentOutput::Host {
        moments: batch.moments,
        status: batch.status,
        stride: batch.stride,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn affine_cell() -> GpuDenestedCubicCell {
        GpuDenestedCubicCell {
            left: -1.0,
            right: 1.0,
            c0: 0.0,
            c1: 0.0,
            c2: 0.0,
            c3: 0.0,
        }
    }

    #[test]
    fn host_residency_returns_real_moments() {
        let cells = [affine_cell()];
        let branches = [GpuCellBranchTag::Affine];
        let view = CubicCellDerivativeMomentHostView {
            cells: &cells,
            branches: &branches,
            max_degree: 9,
            mode: CubicCellMomentMode::DerivativeOnly,
            residency: CubicCellMomentResidency::Host,
        };
        let out = try_build_cubic_cell_derivative_moments(view, None)
            .expect("host substrate succeeds on a valid cell")
            .expect("non-empty input produces output");
        let CubicCellDerivativeMomentOutput::Host {
            moments,
            status,
            stride,
        } = out
        else {
            panic!("expected host output");
        };
        assert_eq!(stride, 10);
        assert_eq!(status, vec![CubicCellMomentStatus::Ok as u8]);
        // M_0 for η ≡ 0 over [-1, 1] is sqrt(2π) · (Φ(1) − Φ(−1)).
        assert!((moments[0] - 1.7112488348667447).abs() < 1e-12);
    }

    #[test]
    fn device_residency_falls_back_to_host_when_runtime_absent() {
        // On a host without CUDA the device dispatcher returns
        // `Ok(None)` and the entry function transparently routes through
        // the host substrate — callers asking for `Device` still get
        // real moments, not a `NotYetImplemented` surprise.
        let cells = [affine_cell()];
        let branches = [GpuCellBranchTag::Affine];
        let view = CubicCellDerivativeMomentHostView {
            cells: &cells,
            branches: &branches,
            max_degree: 9,
            mode: CubicCellMomentMode::DerivativeOnly,
            residency: CubicCellMomentResidency::Device,
        };
        let out = try_build_cubic_cell_derivative_moments(view, None)
            .expect("device residency on a CPU-only host must succeed via fallback")
            .expect("non-empty input produces output");
        match out {
            CubicCellDerivativeMomentOutput::Host { status, .. } => {
                assert_eq!(status, vec![CubicCellMomentStatus::Ok as u8]);
            }
            CubicCellDerivativeMomentOutput::Device(_) => {
                // A real GPU runtime is allowed to keep moments resident on
                // the device. The host-shaped fallback is only required when
                // no runtime is available.
            }
        }
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
        let cells = [affine_cell()];
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
        let cells = [affine_cell()];
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
    fn rejects_value_and_derivative_mode_until_survival_flex_lands() {
        let cells = [affine_cell()];
        let branches = [GpuCellBranchTag::Affine];
        let view = CubicCellDerivativeMomentHostView {
            cells: &cells,
            branches: &branches,
            max_degree: 9,
            mode: CubicCellMomentMode::ValueAndDerivative,
            residency: CubicCellMomentResidency::Host,
        };
        let err = try_build_cubic_cell_derivative_moments(view, None).unwrap_err();
        assert!(err.to_string().contains("ValueAndDerivative"));
    }

    #[test]
    fn workspace_records_largest_capacity_observed() {
        let mut ws = CubicCellMomentWorkspace::new();
        ws.note(7, 22);
        ws.note(3, 30);
        ws.note(10, 5);
        assert_eq!(ws.capacity_cells, 10);
        assert_eq!(ws.capacity_stride, 30);
    }

    #[test]
    fn workspace_new_starts_at_zero_capacity() {
        let ws = CubicCellMomentWorkspace::new();
        assert_eq!(ws.capacity_cells, 0);
        assert_eq!(ws.capacity_stride, 0);
    }

    #[test]
    fn status_codes_match_kernel_abi() {
        assert_eq!(CubicCellMomentStatus::Ok as u8, 0);
        assert_eq!(CubicCellMomentStatus::InvalidInterval as u8, 1);
        assert_eq!(CubicCellMomentStatus::NonAffineInfiniteInterval as u8, 2);
        assert_eq!(CubicCellMomentStatus::NonFiniteCoefficient as u8, 3);
        assert_eq!(CubicCellMomentStatus::NonFiniteEvaluation as u8, 4);
    }

#[test]
    fn output_host_variant_carries_stride_and_status() {
        let out = CubicCellDerivativeMomentOutput::Host {
            moments: vec![0.0; 10],
            status: vec![CubicCellMomentStatus::Ok as u8],
            stride: 10,
        };
        let CubicCellDerivativeMomentOutput::Host {
            moments,
            status,
            stride,
        } = out
        else {
            panic!("expected host output");
        };
        assert_eq!(stride, 10);
        assert_eq!(status.len(), 1);
        assert_eq!(moments.len(), 10);
    }

    #[test]
    fn output_device_variant_carries_shape() {
        let out = CubicCellDerivativeMomentOutput::Device(DeviceCubicCellMomentBatch {
            n_cells: 4,
            moment_stride: 10,
        });
        let CubicCellDerivativeMomentOutput::Device(batch) = out else {
            panic!("expected device output");
        };
        assert_eq!(batch.n_cells, 4);
        assert_eq!(batch.moment_stride, 10);
    }
}
