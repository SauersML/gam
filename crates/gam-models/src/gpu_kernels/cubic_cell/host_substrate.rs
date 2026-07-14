//! Test oracle for the cubic-cell derivative-moment substrate.
//!
//! This module is compiled only by the parent module's test configuration. It
//! produces the same row-major `[n_cells, max_degree+1]` moment layout as the
//! device kernel, without creating a production CPU fallback for a selected
//! device route.
//!
//! The host path defers all heavy math to the existing CPU evaluator
//! `crate::cubic_cell_kernel::evaluate_cell_derivative_moments_uncached`.
//! This module's only jobs are:
//!
//! 1. validate the host view (lengths and supported degree),
//! 2. per-cell: respect the caller-supplied branch tag *if* it agrees with
//!    the host classifier (mismatches degrade the cell to a status code
//!    instead of silently producing different math),
//! 3. pack moments into the GPU-shaped output buffer with the agreed stride,
//! 4. record one status code per cell so the caller can react to per-cell
//!    failures without having to re-run the CPU classifier.
//!
//! The moment-emitting path below is the numerical parity oracle.

use crate::gpu_kernels::cubic_cell::{
    CubicCellDerivativeMomentHostView, MAX_SUPPORTED_DEGREE,
};

fn validate_host_view(view: &CubicCellDerivativeMomentHostView<'_>) -> Result<(), String> {
    if view.cells.len() != view.branches.len() {
        return Err(format!(
            "host cubic-cell oracle: cells.len()={} != branches.len()={}",
            view.cells.len(),
            view.branches.len()
        ));
    }
    if view.max_degree > MAX_SUPPORTED_DEGREE {
        return Err(format!(
            "host cubic-cell oracle: max_degree={} exceeds MAX_SUPPORTED_DEGREE={}",
            view.max_degree, MAX_SUPPORTED_DEGREE
        ));
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::validate_host_view;
    use crate::cubic_cell_kernel::{DenestedCubicCell, evaluate_cell_derivative_moments_uncached};
    use crate::gpu_kernels::cubic_cell::branch::classify_cell_for_gpu;
    use crate::gpu_kernels::cubic_cell::{
        CubicCellDerivativeMomentHostView, CubicCellMomentStatus, GpuCellBranchTag,
        GpuDenestedCubicCell, MAX_SUPPORTED_DEGREE,
    };

    /// Row-major moments + per-cell status, matching the layout the device
    /// kernel writes. This batch is the private parity oracle compared with the
    /// independent CPU evaluator.
    struct HostMomentBatch {
        pub moments: Vec<f64>,
        pub status: Vec<CubicCellMomentStatus>,
        pub stride: usize,
    }

    /// Run the CPU evaluator on every accepted cell so the test suite can check
    /// substrate moments against an independent host reference.
    fn build_host_moments(
        view: &CubicCellDerivativeMomentHostView<'_>,
    ) -> Result<HostMomentBatch, String> {
        validate_host_view(view)?;
        let n_cells = view.cells.len();
        let stride = view.max_degree + 1;
        let mut moments = vec![0.0_f64; n_cells.saturating_mul(stride)];
        let mut status = vec![CubicCellMomentStatus::Ok; n_cells];

        for (i, &gpu_cell) in view.cells.iter().enumerate() {
            let row = &mut moments[i * stride..(i + 1) * stride];

            let host_tag = match classify_cell_for_gpu(gpu_cell) {
                Ok(tag) => tag,
                Err(code) => {
                    status[i] = code;
                    continue;
                }
            };
            let caller_tag = view.branches[i];
            if host_tag != caller_tag {
                status[i] = CubicCellMomentStatus::InvalidInterval;
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
                        status[i] = CubicCellMomentStatus::NonFiniteEvaluation;
                    }
                }
                Err(_) => {
                    status[i] = match host_tag {
                        GpuCellBranchTag::AffineTail => {
                            CubicCellMomentStatus::NonAffineInfiniteInterval
                        }
                        _ => CubicCellMomentStatus::InvalidInterval,
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
    fn host_oracle_accepts_empty_workload() {
        let view = CubicCellDerivativeMomentHostView {
            cells: &[],
            branches: &[],
            max_degree: 9,
        };
        let out = build_host_moments(&view).expect("empty oracle workload is valid");
        assert!(out.moments.is_empty());
        assert!(out.status.is_empty());
        assert_eq!(out.stride, 10);
    }

    #[test]
    fn host_oracle_rejects_mismatched_cell_and_branch_counts() {
        let cells = [GpuDenestedCubicCell {
            left: -1.0,
            right: 1.0,
            c0: 0.0,
            c1: 0.0,
            c2: 0.0,
            c3: 0.0,
        }];
        let view = CubicCellDerivativeMomentHostView {
            cells: &cells,
            branches: &[],
            max_degree: 9,
        };
        let error = build_host_moments(&view)
            .err()
            .expect("mismatched parallel arrays must fail");
        assert!(error.contains("cells.len()"), "got: {error}");
        assert!(error.contains("branches.len()"), "got: {error}");
    }

    #[test]
    fn host_oracle_rejects_unsupported_degree() {
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
            max_degree: MAX_SUPPORTED_DEGREE + 1,
        };
        let error = build_host_moments(&view)
            .err()
            .expect("unsupported degree must fail");
        assert!(error.contains("MAX_SUPPORTED_DEGREE"), "got: {error}");
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
        };
        let out = build_host_moments(&view).expect("host substrate");
        assert_eq!(out.status[0], CubicCellMomentStatus::Ok);
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
        };
        let out = build_host_moments(&view).expect("host substrate");
        assert_eq!(out.status[0], CubicCellMomentStatus::Ok);
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
        };
        let out = build_host_moments(&view).expect("host substrate");
        assert_eq!(out.status[0], CubicCellMomentStatus::Ok);
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
        };
        let out = build_host_moments(&view).expect("host substrate");
        assert_eq!(out.status[0], CubicCellMomentStatus::Ok);
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
        };
        let out = build_host_moments(&view).expect("host substrate");
        assert_eq!(out.status[0], CubicCellMomentStatus::InvalidInterval);
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
            };
            let out = build_host_moments(&view).expect("host substrate");
            assert_eq!(out.stride, max_degree + 1);
            assert_eq!(out.status.len(), cells_cpu.len());
            for (i, &cell) in cells_cpu.iter().enumerate() {
                assert_eq!(
                    out.status[i],
                    CubicCellMomentStatus::Ok,
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
        };
        let out = build_host_moments(&view).expect("host substrate");
        assert_eq!(out.status[0], CubicCellMomentStatus::InvalidInterval);
        assert!(out.moments.iter().all(|&x| x == 0.0));
    }
}
