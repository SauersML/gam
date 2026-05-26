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
    /// Device-resident moments + status on the cubic-cell backend's shared
    /// CUDA context. Linux-only — non-Linux callers see the `Host` variant
    /// even when they request `Device` residency. Layout matches `Host` so
    /// `d_moments` is a row-major `[n_cells, stride]` `CudaSlice<f64>` and
    /// `d_status` is a `CudaSlice<u8>` of length `n_cells`. The host-side
    /// `status` vector mirrors `d_status` for branching decisions the caller
    /// would otherwise have to round-trip from the device.
    #[cfg(target_os = "linux")]
    Device {
        d_moments: cudarc::driver::CudaSlice<f64>,
        d_status: cudarc::driver::CudaSlice<u8>,
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

    match input.residency {
        CubicCellMomentResidency::Host => {
            if let Some(batch) = device::try_device_moments(&input)? {
                return Ok(Some(into_host_output(batch)));
            }
            let batch = build_host_moments(&input)
                .map_err(|reason| GpuError::NotYetImplemented { reason })?;
            Ok(Some(into_host_output(batch)))
        }
        CubicCellMomentResidency::Device => {
            #[cfg(target_os = "linux")]
            {
                if let Some(device_batch) = device::try_device_moments_resident(&input)? {
                    return Ok(Some(device_batch));
                }
            }
            // Non-Linux, or no usable runtime: fall back to the host shape so
            // the caller has a parity-shaped result instead of a phantom
            // device claim.
            let batch = build_host_moments(&input)
                .map_err(|reason| GpuError::NotYetImplemented { reason })?;
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
        let CubicCellDerivativeMomentOutput::Host {
            moments,
            status,
            stride,
        } = out;
        assert_eq!(stride, 10);
        assert_eq!(status, vec![CubicCellMomentStatus::Ok as u8]);
        // M_0 for η ≡ 0 over [-1, 1] is sqrt(2π) · (Φ(1) − Φ(−1)).
        assert!((moments[0] - 1.7112488348667447).abs() < 1e-12);
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
        let err = try_build_cubic_cell_derivative_moments(host_view(&cells, &branches, 9))
            .unwrap_err();
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

    /// Phase 4 parity test: device-resident moments must match the CPU
    /// `evaluate_cell_derivative_moments_uncached` reference across all
    /// three branches (`Affine`, `NonAffineFinite`, `AffineTail`) at the
    /// production high-water-mark degrees (9, 15, 21).
    ///
    /// Skipped silently on hosts without a usable CUDA runtime so the test
    /// passes on the Mac builder. On V100 it runs the device pipeline,
    /// downloads the moments for verification, and compares elementwise
    /// against the CPU evaluator at `abs <= 1e-12 OR rel <= 1e-11`.
    #[cfg(target_os = "linux")]
    #[test]
    fn cubic_cell_device_residency_matches_cpu_all_branches() {
        use crate::families::cubic_cell_kernel::{
            DenestedCubicCell, evaluate_cell_derivative_moments_uncached,
        };
        use crate::gpu::runtime::GpuRuntime;
        if GpuRuntime::global().is_none() {
            eprintln!(
                "[cubic_cell device-residency parity] no CUDA runtime — skipping"
            );
            return;
        }
        // One cell per branch, plus a sextic NonAffineFinite stressor.
        let cpu_cells = vec![
            // Pure Affine.
            DenestedCubicCell {
                left: -1.0,
                right: 1.0,
                c0: 0.2,
                c1: 0.7,
                c2: 0.0,
                c3: 0.0,
            },
            // Quartic NonAffineFinite.
            DenestedCubicCell {
                left: -1.25,
                right: -0.2,
                c0: -0.35,
                c1: 0.85,
                c2: 0.4,
                c3: 0.0,
            },
            // Sextic NonAffineFinite.
            DenestedCubicCell {
                left: -0.5,
                right: 1.7,
                c0: 0.2,
                c1: -0.6,
                c2: 0.25,
                c3: 0.18,
            },
            // AffineTail (left-infinite).
            DenestedCubicCell {
                left: f64::NEG_INFINITY,
                right: -0.7,
                c0: 0.1,
                c1: 0.5,
                c2: 0.0,
                c3: 0.0,
            },
            // AffineTail (right-infinite).
            DenestedCubicCell {
                left: 1.2,
                right: f64::INFINITY,
                c0: -0.05,
                c1: 0.3,
                c2: 0.0,
                c3: 0.0,
            },
            // Whole-line affine.
            DenestedCubicCell {
                left: f64::NEG_INFINITY,
                right: f64::INFINITY,
                c0: 0.0,
                c1: 0.0,
                c2: 0.0,
                c3: 0.0,
            },
        ];
        let cells_gpu: Vec<GpuDenestedCubicCell> = cpu_cells
            .iter()
            .map(|c| GpuDenestedCubicCell {
                left: c.left,
                right: c.right,
                c0: c.c0,
                c1: c.c1,
                c2: c.c2,
                c3: c.c3,
            })
            .collect();
        let branches: Vec<GpuCellBranchTag> = cpu_cells
            .iter()
            .map(|c| {
                if !c.left.is_finite() || !c.right.is_finite() {
                    GpuCellBranchTag::AffineTail
                } else if c.c2 == 0.0 && c.c3 == 0.0 {
                    GpuCellBranchTag::Affine
                } else {
                    GpuCellBranchTag::NonAffineFinite
                }
            })
            .collect();

        for &max_degree in &[9_usize, 15, 21] {
            let view = CubicCellDerivativeMomentHostView {
                cells: &cells_gpu,
                branches: &branches,
                max_degree,
                residency: CubicCellMomentResidency::Device,
            };
            let out = try_build_cubic_cell_derivative_moments(view)
                .expect("device-residency dispatch must succeed with CUDA")
                .expect("non-empty input must yield output");
            let (d_moments, status, stride, n_cells) = match out {
                CubicCellDerivativeMomentOutput::Device {
                    d_moments,
                    d_status: _,
                    status,
                    stride,
                    n_cells,
                } => (d_moments, status, stride, n_cells),
                CubicCellDerivativeMomentOutput::Host { .. } => panic!(
                    "device residency must produce CubicCellDerivativeMomentOutput::Device on a CUDA host"
                ),
            };
            assert_eq!(stride, max_degree + 1);
            assert_eq!(n_cells, cpu_cells.len());
            assert_eq!(status.len(), cpu_cells.len());
            // Download for verification.
            let backend =
                crate::gpu::cubic_cell::device::CubicCellGpuBackend::probe()
                    .expect("backend probe");
            let host_moments = backend
                .test_only_download_moments(&d_moments)
                .expect("DtoH download for parity check");
            for (i, &cpu_cell) in cpu_cells.iter().enumerate() {
                assert_eq!(
                    status[i],
                    CubicCellMomentStatus::Ok as u8,
                    "cell {i} must classify Ok (status={})",
                    status[i]
                );
                let row = &host_moments[i * stride..(i + 1) * stride];
                let cpu_state =
                    evaluate_cell_derivative_moments_uncached(cpu_cell, max_degree)
                        .expect("cpu reference");
                for (k, (&got, &want)) in row.iter().zip(cpu_state.moments.iter()).enumerate() {
                    let abs = (got - want).abs();
                    let denom = want.abs().max(1.0);
                    let rel = abs / denom;
                    assert!(
                        abs <= 1e-12 || rel <= 1e-11,
                        "device parity drift at degree={max_degree} cell={i} k={k} \
                         gpu={got:.17e} cpu={want:.17e} abs={abs:.3e} rel={rel:.3e}"
                    );
                }
            }
        }
    }
}
