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
//! * `tests_host_substrate` — test-only CPU oracle for device-kernel parity.
//! * [`kernel_src`] — NVRTC-compilable CUDA C++ source as Rust string
//!   constants (D9 / D15 / D21 specializations).
//! * [`device`] — Linux+CUDA dispatcher that compiles, launches, and
//!   gathers the NVRTC kernel for the NonAffineFinite bucket; Affine /
//!   AffineTail buckets stay on CPU until Stage-2.

pub(crate) mod branch;
#[cfg(target_os = "linux")]
pub(crate) mod device;
#[cfg(test)]
#[path = "host_substrate.rs"]
mod tests_host_substrate;
pub(crate) mod kernel_src;

#[cfg(target_os = "linux")]
use gam_gpu::gpu_error::GpuError;

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

#[cfg(target_os = "linux")]
impl CubicCellMomentStatus {
    fn from_device_code(code: u8) -> Result<Self, GpuError> {
        match code {
            0 => Ok(Self::Ok),
            1 => Ok(Self::InvalidInterval),
            2 => Ok(Self::NonAffineInfiniteInterval),
            3 => Ok(Self::NonFiniteCoefficient),
            4 => Ok(Self::NonFiniteEvaluation),
            _ => Err(GpuError::DriverCallFailed {
                reason: format!("gpu cubic-cell kernel emitted unknown status code {code}"),
            }),
        }
    }
}

/// Host-side input view for `try_build_cubic_cell_derivative_moments`.
/// The substrate borrows cell data from the caller; it does not own the
/// CPU partition. `branches` is parallel to `cells`.
pub(crate) struct CubicCellDerivativeMomentHostView<'a> {
    pub cells: &'a [GpuDenestedCubicCell],
    pub branches: &'a [GpuCellBranchTag],
    pub max_degree: usize,
}

/// Device-resident output of `try_build_cubic_cell_derivative_moments`.
#[cfg(target_os = "linux")]
#[derive(Debug)]
pub(crate) struct CubicCellDerivativeMomentOutput {
    pub(crate) d_moments: cudarc::driver::CudaSlice<f64>,
    pub(crate) status: Vec<CubicCellMomentStatus>,
    pub(crate) stride: usize,
    pub(crate) n_cells: usize,
}

/// Try to build derivative moments via the substrate.
///
/// On Linux+CUDA, the dispatcher launches the NVRTC kernel for the
/// NonAffineFinite bucket and CPU-classifies the Affine/AffineTail buckets,
/// packing all accepted rows into one device-resident output. Runtime absence
/// at this already-selected device boundary is a typed error, never host
/// substitution.
///
/// Returns `Ok(None)` only when the workload is empty.
///
#[cfg(target_os = "linux")]
pub(crate) fn try_build_cubic_cell_derivative_moments(
    input: CubicCellDerivativeMomentHostView<'_>,
) -> Result<Option<CubicCellDerivativeMomentOutput>, GpuError> {
    if input.cells.len() != input.branches.len() {
        gam_gpu::gpu_bail!(
            "gpu cubic-cell substrate: cells.len()={} != branches.len()={}",
            input.cells.len(),
            input.branches.len()
        );
    }
    if input.max_degree > MAX_SUPPORTED_DEGREE {
        gam_gpu::gpu_bail!(
            "gpu cubic-cell substrate: max_degree={} exceeds MAX_SUPPORTED_DEGREE={}",
            input.max_degree,
            MAX_SUPPORTED_DEGREE
        );
    }
    if input.cells.is_empty() {
        return Ok(None);
    }

    device::build_device_moments_resident(&input).map(Some)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn status_codes_match_kernel_abi() {
        assert_eq!(CubicCellMomentStatus::Ok as u8, 0);
        assert_eq!(CubicCellMomentStatus::InvalidInterval as u8, 1);
        assert_eq!(CubicCellMomentStatus::NonAffineInfiniteInterval as u8, 2);
        assert_eq!(CubicCellMomentStatus::NonFiniteCoefficient as u8, 3);
        assert_eq!(CubicCellMomentStatus::NonFiniteEvaluation as u8, 4);
    }

    // Phase 4 device-residency parity test lives next to the device backend
    // at `crate::gpu_kernels::cubic_cell::device::tests::cubic_cell_device_residency_matches_cpu_all_branches`
    // so it can use the in-mod `download_moments` helper directly.
}
