//! Host-resident implementation of the cubic-cell derivative-moment substrate.
//!
//! The host implementation is the CPU-fallback path and the *parity reference*
//! for the GPU kernel: it produces the same row-major `[n_cells,
//! max_degree+1]` moment layout the device kernel writes. Callers that pick
//! `CubicCellMomentResidency::Host` get a real result on every platform; the
//! `Device` residency variant lands with the NVRTC kernel wiring on Linux.
//!
//! The host path defers all heavy math to the existing CPU evaluator
//! `crate::cubic_cell_kernel::evaluate_cell_derivative_moments_uncached`.
//! This module's only jobs are:
//!
//! 1. validate the host view (lengths, supported degree, mode),
//! 2. per-cell: respect the caller-supplied branch tag *if* it agrees with
//!    the host classifier (mismatches degrade the cell to a status code
//!    instead of silently producing different math),
//! 3. pack moments into the GPU-shaped output buffer with the agreed stride,
//! 4. record one status code per cell so the caller can react to per-cell
//!    failures without having to re-run the CPU classifier.
//!
//! The production substrate's host path returns only the per-cell status
//! codes (see [`build_host_cell_status`]); production consumers (BMS
//! row-primary Hessian, survival-flex row evaluator) read the verdict but
//! never the moments themselves. The moment-emitting reference path
//! (`build_host_moments` + `HostMomentBatch`) lives in the test module below
//! as a comparison oracle for the device kernel's row-major moment buffer.

use crate::cubic_cell_kernel::{
    DenestedCubicCell, evaluate_cell_derivative_moments_uncached,
};
use crate::gpu_kernels::cubic_cell::branch::classify_cell_for_gpu;
use crate::gpu_kernels::cubic_cell::{
    CubicCellDerivativeMomentHostView, CubicCellMomentStatus, GpuCellBranchTag,
};

/// Classify each cell with the host classifier and return the per-cell
/// status vector — the only payload production callers (the substrate's
/// `Host` residency output, and the survival-flex row evaluator) consume on
/// the CPU path.
pub(crate) fn build_host_cell_status(
    view: &CubicCellDerivativeMomentHostView<'_>,
) -> Result<Vec<u8>, String> {
    let n_cells = view.cells.len();
    let mut status = vec![CubicCellMomentStatus::Ok as u8; n_cells];

    for (i, &gpu_cell) in view.cells.iter().enumerate() {
        let host_tag = match classify_cell_for_gpu(gpu_cell) {
            Ok(tag) => tag,
            Err(code) => {
                status[i] = code as u8;
                continue;
            }
        };
        let caller_tag = view.branches[i];
        if host_tag != caller_tag {
            // Caller's classifier disagreed with ours — refuse the cell
            // rather than silently producing a different cell's status.
            status[i] = CubicCellMomentStatus::InvalidInterval as u8;
            continue;
        }

        // The production substrate path does not need the moments
        // themselves; it consumes only the per-cell classifier verdict.
        // We still simulate the evaluator's failure modes here so the
        // status vector mirrors what `build_host_moments` would have
        // recorded (NonFiniteEvaluation, etc.) for parity with the
        // moment-emitting path.
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
                if state.moments.iter().any(|x| !x.is_finite()) {
                    status[i] = CubicCellMomentStatus::NonFiniteEvaluation as u8;
                }
            }
            Err(_) => {
                status[i] = match host_tag {
                    GpuCellBranchTag::AffineTail => {
                        CubicCellMomentStatus::NonAffineInfiniteInterval as u8
                    }
                    _ => CubicCellMomentStatus::InvalidInterval as u8,
                };
            }
        }
    }

    Ok(status)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::cubic_cell_kernel::{
        DenestedCubicCell, evaluate_cell_derivative_moments_uncached,
    };
    use crate::gpu_kernels::cubic_cell::{
        CubicCellDerivativeMomentHostView, CubicCellMomentResidency, GpuCellBranchTag,
        GpuDenestedCubicCell,
    };

    /// Row-major moments + per-cell status, the same layout the device
    /// kernel writes. Production callers consume only the status (see
    /// [`build_host_cell_status`]); this batch shape is the parity oracle
    /// the moment-emitting unit tests compare to the CPU evaluator.
    pub(super) struct HostMomentBatch {
        pub moments: Vec<f64>,
        pub status: Vec<u8>,
        pub stride: usize,
    }

    /// Moment-emitting analog of [`super::build_host_cell_status`] — runs
    /// the CPU evaluator on every Ok cell so the test suite can check
    /// substrate moments against the CPU reference without the production
    /// substrate having to materialize a Vec<f64> nobody reads.
    pub(super) fn build_host_moments(
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
                    continue;
                }
            };
            let caller_tag = view.branches[i];
            if host_tag != caller_tag {
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
                    let copy_len = state.moments.len().min(stride);
                    row[..copy_len].copy_from_slice(&state.moments[..copy_len]);
                    if row.iter().any(|x| !x.is_finite()) {
                        for slot in row.iter_mut() {
                            *slot = 0.0;
                        }
                        status[i] = CubicCellMomentStatus::NonFiniteEvaluation as u8;
                    }
                }
                Err(_) => {
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
