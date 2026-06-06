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

/// Where the caller wants results materialized.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(crate) enum CubicCellMomentResidency {
    /// Materialize moments into a host `Vec<f64>` (parity reference; works on
    /// every platform).
    Host,
    /// Materialize moments into a device-resident `CudaSlice<f64>` on the
    /// shared cubic-cell context. Linux+CUDA only; on other platforms this
    /// variant degrades to `Host`-shaped output through the host substrate
    /// (no silent device claim).
    #[cfg(target_os = "linux")]
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
    pub residency: CubicCellMomentResidency,
}

/// Output of `try_build_cubic_cell_derivative_moments`.
#[derive(Debug)]
pub(crate) enum CubicCellDerivativeMomentOutput {
    /// Row-major `[n_cells, max_degree + 1]` host buffer + per-cell status
    /// codes. Row `i` is `moments[i * stride ..][..stride]` where
    /// `stride = max_degree + 1`. Rows for non-OK cells are zeroed.
    Host {
        moments: Vec<f64>,
        status: Vec<u8>,
        stride: usize,
    },
    /// Device-resident moments on the cubic-cell backend's shared CUDA
    /// context. Linux-only — non-Linux callers see the `Host` variant even
    /// when they request `Device` residency. Layout matches `Host` so
    /// `d_moments` is a row-major `[n_cells, stride]` `CudaSlice<f64>`. The
    /// host-side `status` vector mirrors the per-cell device status so
    /// downstream branching decisions never have to round-trip from the
    /// device.
    #[cfg(target_os = "linux")]
    Device {
        d_moments: cudarc::driver::CudaSlice<f64>,
        status: Vec<u8>,
        stride: usize,
        n_cells: usize,
    },
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
) -> Result<Option<CubicCellDerivativeMomentOutput>, GpuError> {
    if input.cells.len() != input.branches.len() {
        crate::gpu_bail!(
            "gpu cubic-cell substrate: cells.len()={} != branches.len()={}",
            input.cells.len(),
            input.branches.len()
        );
    }
    if input.max_degree > MAX_SUPPORTED_DEGREE {
        crate::gpu_bail!(
            "gpu cubic-cell substrate: max_degree={} exceeds MAX_SUPPORTED_DEGREE={}",
            input.max_degree,
            MAX_SUPPORTED_DEGREE
        );
    }
    if input.cells.is_empty() {
        return Ok(None);
    }

    match input.residency {
        CubicCellMomentResidency::Host => {
            let batch = build_host_moments(&input)
                .map_err(|reason| GpuError::DriverCallFailed { reason })?;
            Ok(Some(into_host_output(batch)))
        }
        #[cfg(target_os = "linux")]
        CubicCellMomentResidency::Device => {
            if let Some(device_batch) = device::try_device_moments_resident(&input)? {
                return Ok(Some(device_batch));
            }
            // Non-Linux, or no usable runtime: fall back to the host shape so
            // the caller has a parity-shaped result instead of a phantom
            // device claim.
            let batch = build_host_moments(&input)
                .map_err(|reason| GpuError::DriverCallFailed { reason })?;
            Ok(Some(into_host_output(batch)))
        }
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

    fn host_view<'a>(
        cells: &'a [GpuDenestedCubicCell],
        branches: &'a [GpuCellBranchTag],
        max_degree: usize,
    ) -> CubicCellDerivativeMomentHostView<'a> {
        CubicCellDerivativeMomentHostView {
            cells,
            branches,
            max_degree,
            residency: CubicCellMomentResidency::Host,
        }
    }

    #[test]
    fn host_residency_returns_real_moments() {
        let cells = [affine_cell()];
        let branches = [GpuCellBranchTag::Affine];
        let out = try_build_cubic_cell_derivative_moments(host_view(&cells, &branches, 9))
            .expect("host substrate succeeds on a valid cell")
            .expect("non-empty input produces output");
        let (moments, status, stride) = match out {
            CubicCellDerivativeMomentOutput::Host {
                moments,
                status,
                stride,
            } => (moments, status, stride),
            #[cfg(target_os = "linux")]
            CubicCellDerivativeMomentOutput::Device { .. } => {
                panic!("host residency request must not yield Device output")
            }
        };
        assert_eq!(stride, 10);
        assert_eq!(status, vec![CubicCellMomentStatus::Ok as u8]);
        // M_0(η ≡ 0, [-1, 1]) = ∫_{-1}^{1} exp(-z²/2) dz
        //                     = √(2π) · (Φ(1) − Φ(−1))
        //                     = √(2π) · erf(1/√2)
        //                     ≈ 2.5066282746310002 · 0.6826894921370859
        //                     ≈ 1.7112487837842974
        assert!(
            (moments[0] - 1.711_248_783_784_297_4).abs() < 1e-13,
            "M_0 should match the closed-form √(2π)·erf(1/√2): got {}",
            moments[0]
        );
    }

    #[test]
    fn empty_input_returns_ok_none() {
        let out = try_build_cubic_cell_derivative_moments(host_view(&[], &[], 9)).expect("ok");
        assert!(out.is_none());
    }

    #[test]
    fn rejects_mismatched_lengths() {
        let cells = [affine_cell()];
        let branches: [GpuCellBranchTag; 0] = [];
        let err =
            try_build_cubic_cell_derivative_moments(host_view(&cells, &branches, 9)).unwrap_err();
        let msg = err.to_string();
        assert!(msg.contains("cells.len()"), "got: {msg}");
        assert!(msg.contains("branches.len()"), "got: {msg}");
    }

    #[test]
    fn rejects_degree_above_supported_max() {
        let cells = [affine_cell()];
        let branches = [GpuCellBranchTag::Affine];
        let err = try_build_cubic_cell_derivative_moments(host_view(
            &cells,
            &branches,
            MAX_SUPPORTED_DEGREE + 1,
        ))
        .unwrap_err();
        assert!(err.to_string().contains("MAX_SUPPORTED_DEGREE"));
    }

    #[test]
    fn status_codes_match_kernel_abi() {
        assert_eq!(CubicCellMomentStatus::Ok as u8, 0);
        assert_eq!(CubicCellMomentStatus::InvalidInterval as u8, 1);
        assert_eq!(CubicCellMomentStatus::NonAffineInfiniteInterval as u8, 2);
        assert_eq!(CubicCellMomentStatus::NonFiniteCoefficient as u8, 3);
        assert_eq!(CubicCellMomentStatus::NonFiniteEvaluation as u8, 4);
    }

    // Phase 4 device-residency parity test lives next to the device backend
    // at `crate::gpu::cubic_cell::device::tests::cubic_cell_device_residency_matches_cpu_all_branches`
    // so it can use the in-mod `download_moments` helper directly.
}
