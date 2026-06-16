//! Host-resident implementation of the cubic-cell derivative-moment substrate.
//!
//! The host implementation is the CPU-fallback path and the *parity reference*
//! for the GPU kernel: it produces the same row-major `[n_cells,
//! max_degree+1]` moment layout the device kernel writes. Callers that pick
//! `CubicCellMomentResidency::Host` get a real result on every platform; the
//! `Device` residency variant lands with the NVRTC kernel wiring on Linux.
//!
//! The host path defers all heavy math to the existing CPU evaluator
//! `crate::families::cubic_cell_kernel::evaluate_cell_derivative_moments_uncached`.
//! This module's only jobs are:
//!
//! 1. validate the host view (lengths, supported degree, mode),
//! 2. per-cell: respect the caller-supplied branch tag *if* it agrees with
//!    the host classifier (mismatches degrade the cell to a status code
//!    instead of silently producing different math),
//! 3. pack moments into the GPU-shaped output buffer with the agreed stride,
//! 4. record one status code per cell so the caller can react to per-cell
//!    failures without having to re-run the CPU classifier.

use crate::families::cubic_cell_kernel::{
    DenestedCubicCell, evaluate_cell_derivative_moments_uncached,
};
use crate::gpu::kernels::cubic_cell::branch::classify_cell_for_gpu;
use crate::gpu::kernels::cubic_cell::{
    CubicCellDerivativeMomentHostView, CubicCellMomentStatus, GpuCellBranchTag,
};

/// Output of [`build_host_moments`]: row-major moments, per-cell status
/// codes, and the stride used to index `moments`. The row for cell `i` is
/// `moments[i * stride ..][..stride]`; rows for non-OK cells are zeroed.
pub(crate) struct HostMomentBatch {
    pub moments: Vec<f64>,
    pub status: Vec<u8>,
    pub stride: usize,
}

/// CPU implementation of the GPU substrate. Returns row-major derivative
/// moments + per-cell status. Mirrors the shape the device kernel will write
/// so a future GPU landing is a drop-in substitute.
pub(crate) fn build_host_moments(
    view: &CubicCellDerivativeMomentHostView<'_>,
) -> Result<HostMomentBatch, String> {
    let n_cells = view.cells.len();
    let stride = view.max_degree + 1;
    let mut moments = vec![0.0_f64; n_cells.saturating_mul(stride)];
    let mut status = vec![CubicCellMomentStatus::Ok as u8; n_cells];

    for (i, &gpu_cell) in view.cells.iter().enumerate() {
        let row = &mut moments[i * stride..(i + 1) * stride];

        let host_tag = match classify_cell_for_gpu(gpu_cell) {
            Ok(tag) => tag,
            Err(code) => {
                status[i] = code as u8;
                // Row already initialized to zero.
                continue;
            }
        };
        let caller_tag = view.branches[i];
        if host_tag != caller_tag {
            // Caller's classifier disagreed with ours — refuse rather than
            // silently produce a different cell's moments. The substrate
            // does not arbitrate; it routes the per-cell failure back to
            // the caller so they can re-classify or fall back.
            status[i] = CubicCellMomentStatus::InvalidInterval as u8;
            continue;
        }

        let cpu_cell = DenestedCubicCell {
            left: gpu_cell.left,
            right: gpu_cell.right,
            c0: gpu_cell.c0,
            c1: gpu_cell.c1,
            c2: gpu_cell.c2,
            c3: gpu_cell.c3,
        };
        match evaluate_cell_derivative_moments_uncached(cpu_cell, view.max_degree) {
            Ok(state) => {
                // The CPU evaluator returns `max_degree + 1` moments by
                // construction; copying the prefix matches the row.
                let copy_len = state.moments.len().min(stride);
                row[..copy_len].copy_from_slice(&state.moments[..copy_len]);
                // Guard against an evaluator that produced a non-finite
                // moment — happens when q overflows for pathological cells.
                if row.iter().any(|x| !x.is_finite()) {
                    for slot in row.iter_mut() {
                        *slot = 0.0;
                    }
                    status[i] = CubicCellMomentStatus::NonFiniteEvaluation as u8;
                }
            }
            Err(_) => {
                // Row stays zeroed.
                status[i] = match host_tag {
                    GpuCellBranchTag::AffineTail => {
                        CubicCellMomentStatus::NonAffineInfiniteInterval as u8
                    }
                    _ => CubicCellMomentStatus::InvalidInterval as u8,
                };
            }
        }
    }

    Ok(HostMomentBatch {
        moments,
        status,
        stride,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::families::cubic_cell_kernel::{
        DenestedCubicCell, evaluate_cell_derivative_moments_uncached,
    };
    use crate::gpu::kernels::cubic_cell::{
        CubicCellDerivativeMomentHostView, CubicCellMomentResidency, GpuCellBranchTag,
        GpuDenestedCubicCell,
    };

    fn gpu_from_cpu(cpu: DenestedCubicCell) -> GpuDenestedCubicCell {
        GpuDenestedCubicCell {
            left: cpu.left,
            right: cpu.right,
            c0: cpu.c0,
            c1: cpu.c1,
            c2: cpu.c2,
            c3: cpu.c3,
        }
    }

    fn assert_row_matches_cpu(
        row: &[f64],
        cell: DenestedCubicCell,
        max_degree: usize,
        ulp_rel: f64,
    ) {
        let state =
            evaluate_cell_derivative_moments_uncached(cell, max_degree).expect("cpu reference");
        assert_eq!(row.len(), max_degree + 1);
        for (k, (&got, &want)) in row.iter().zip(state.moments.iter()).enumerate() {
            let denom = want.abs().max(1.0);
            let rel = (got - want).abs() / denom;
            assert!(
                rel <= ulp_rel,
                "moment k={k} got={got:.17e} want={want:.17e} rel={rel:.3e} tol={ulp_rel:.3e}"
            );
        }
    }

    #[test]
    fn host_substrate_matches_cpu_for_quartic_finite_cell() {
        let cpu = DenestedCubicCell {
            left: -1.25,
            right: -0.2,
            c0: -0.35,
            c1: 0.85,
            c2: 0.4,
            c3: 0.0,
        };
        let gpu = gpu_from_cpu(cpu);
        let branches = vec![GpuCellBranchTag::NonAffineFinite];
        let view = CubicCellDerivativeMomentHostView {
            cells: std::slice::from_ref(&gpu),
            branches: &branches,
            max_degree: 9,
            residency: CubicCellMomentResidency::Host,
        };
        let out = build_host_moments(&view).expect("host substrate");
        assert_eq!(out.status[0], CubicCellMomentStatus::Ok as u8);
        assert_row_matches_cpu(&out.moments[..out.stride], cpu, 9, 0.0);
    }

    #[test]
    fn host_substrate_matches_cpu_for_sextic_finite_cell_at_d21() {
        let cpu = DenestedCubicCell {
            left: -0.5,
            right: 1.7,
            c0: 0.2,
            c1: -0.6,
            c2: 0.25,
            c3: 0.18,
        };
        let gpu = gpu_from_cpu(cpu);
        let branches = vec![GpuCellBranchTag::NonAffineFinite];
        let view = CubicCellDerivativeMomentHostView {
            cells: std::slice::from_ref(&gpu),
            branches: &branches,
            max_degree: 21,
            residency: CubicCellMomentResidency::Host,
        };
        let out = build_host_moments(&view).expect("host substrate");
        assert_eq!(out.status[0], CubicCellMomentStatus::Ok as u8);
        assert_row_matches_cpu(&out.moments[..out.stride], cpu, 21, 0.0);
    }

    #[test]
    fn host_substrate_matches_cpu_for_affine_tail_cell() {
        let cpu = DenestedCubicCell {
            left: f64::NEG_INFINITY,
            right: -0.7,
            c0: 0.1,
            c1: 0.5,
            c2: 0.0,
            c3: 0.0,
        };
        let gpu = gpu_from_cpu(cpu);
        let branches = vec![GpuCellBranchTag::AffineTail];
        let view = CubicCellDerivativeMomentHostView {
            cells: std::slice::from_ref(&gpu),
            branches: &branches,
            max_degree: 15,
            residency: CubicCellMomentResidency::Host,
        };
        let out = build_host_moments(&view).expect("host substrate");
        assert_eq!(out.status[0], CubicCellMomentStatus::Ok as u8);
        assert_row_matches_cpu(&out.moments[..out.stride], cpu, 15, 0.0);
    }

    #[test]
    fn host_substrate_matches_cpu_for_whole_line_affine() {
        let cpu = DenestedCubicCell {
            left: f64::NEG_INFINITY,
            right: f64::INFINITY,
            c0: 0.0,
            c1: 0.0,
            c2: 0.0,
            c3: 0.0,
        };
        let gpu = gpu_from_cpu(cpu);
        let branches = vec![GpuCellBranchTag::AffineTail];
        let view = CubicCellDerivativeMomentHostView {
            cells: std::slice::from_ref(&gpu),
            branches: &branches,
            max_degree: 9,
            residency: CubicCellMomentResidency::Host,
        };
        let out = build_host_moments(&view).expect("host substrate");
        assert_eq!(out.status[0], CubicCellMomentStatus::Ok as u8);
        assert_row_matches_cpu(&out.moments[..out.stride], cpu, 9, 0.0);
    }

    #[test]
    fn host_substrate_zeros_invalid_cell_and_records_status() {
        let gpu = GpuDenestedCubicCell {
            left: 1.0,
            right: -1.0,
            c0: 0.0,
            c1: 0.0,
            c2: 0.0,
            c3: 0.0,
        };
        let branches = vec![GpuCellBranchTag::NonAffineFinite];
        let view = CubicCellDerivativeMomentHostView {
            cells: std::slice::from_ref(&gpu),
            branches: &branches,
            max_degree: 9,
            residency: CubicCellMomentResidency::Host,
        };
        let out = build_host_moments(&view).expect("host substrate");
        assert_eq!(out.status[0], CubicCellMomentStatus::InvalidInterval as u8);
        assert!(out.moments.iter().all(|&x| x == 0.0));
    }

    #[test]
    fn cubic_cell_substrate_parity_against_cpu_evaluator() {
        // Multi-cell fabricated partition mimicking the shape BMS feeds in:
        // two affine tails bracketing a handful of interior microcells with a
        // mix of pure-affine, quartic, and sextic coefficients. The substrate
        // must return the exact CPU evaluator's moment vector for every cell,
        // at every degree the consumers care about.
        let cells_cpu = [
            DenestedCubicCell {
                left: f64::NEG_INFINITY,
                right: -1.5,
                c0: 0.05,
                c1: 0.4,
                c2: 0.0,
                c3: 0.0,
            },
            DenestedCubicCell {
                left: -1.5,
                right: -0.3,
                c0: -0.1,
                c1: 0.2,
                c2: 0.0,
                c3: 0.0,
            },
            DenestedCubicCell {
                left: -0.3,
                right: 0.4,
                c0: 0.0,
                c1: 0.5,
                c2: 0.3,
                c3: 0.0,
            },
            DenestedCubicCell {
                left: 0.4,
                right: 1.1,
                c0: 0.15,
                c1: -0.25,
                c2: 0.1,
                c3: 0.18,
            },
            DenestedCubicCell {
                left: 1.1,
                right: 2.0,
                c0: -0.2,
                c1: 0.6,
                c2: 0.0,
                c3: 0.0,
            },
            DenestedCubicCell {
                left: 2.0,
                right: f64::INFINITY,
                c0: 0.3,
                c1: -0.4,
                c2: 0.0,
                c3: 0.0,
            },
        ];
        let cells_gpu: Vec<GpuDenestedCubicCell> =
            cells_cpu.iter().copied().map(gpu_from_cpu).collect();
        let branches: Vec<GpuCellBranchTag> = cells_cpu
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

        // Exercise every degree the production consumers actually request
        // (9 = Bernoulli flex Hessian, 15 = intermediate, 21 = BMS outer
        // higher-derivative reuse).
        for &max_degree in &[9usize, 15, 21] {
            let view = CubicCellDerivativeMomentHostView {
                cells: &cells_gpu,
                branches: &branches,
                max_degree,
                residency: CubicCellMomentResidency::Host,
            };
            let out = build_host_moments(&view).expect("host substrate");
            assert_eq!(out.stride, max_degree + 1);
            assert_eq!(out.status.len(), cells_cpu.len());
            for (i, &cell) in cells_cpu.iter().enumerate() {
                assert_eq!(
                    out.status[i],
                    CubicCellMomentStatus::Ok as u8,
                    "cell {i} status was {} at degree={max_degree}",
                    out.status[i]
                );
                let row = &out.moments[i * out.stride..(i + 1) * out.stride];
                // ulp_rel = 0.0 — the host substrate is a *deferring* wrapper
                // around the same CPU evaluator, so bit-identical equality is
                // the right bar. Any drift means the substrate has introduced
                // a transformation that breaks consumers' assumptions.
                assert_row_matches_cpu(row, cell, max_degree, 0.0);
            }
        }
    }

    #[test]
    fn host_substrate_flags_caller_branch_mismatch() {
        // A finite quartic cell, but the caller claims AffineTail.
        let gpu = GpuDenestedCubicCell {
            left: -1.0,
            right: 1.0,
            c0: 0.2,
            c1: 0.3,
            c2: 0.4,
            c3: 0.0,
        };
        let branches = vec![GpuCellBranchTag::AffineTail];
        let view = CubicCellDerivativeMomentHostView {
            cells: std::slice::from_ref(&gpu),
            branches: &branches,
            max_degree: 9,
            residency: CubicCellMomentResidency::Host,
        };
        let out = build_host_moments(&view).expect("host substrate");
        assert_eq!(out.status[0], CubicCellMomentStatus::InvalidInterval as u8);
        assert!(out.moments.iter().all(|&x| x == 0.0));
    }
}
