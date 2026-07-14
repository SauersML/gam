//! Host-side branch classifier for the GPU cubic-cell substrate.
//!
//! The classifier mirrors `crate::cubic_cell_kernel::branch_cell`
//! plus the semi-infinite-tail handling baked into
//! `evaluate_cell_state_dispatched`. Tagging happens once at the substrate
//! boundary so the all-branch GPU kernel consumes a canonical branch code
//! instead of reimplementing the tolerance predicate.
//!
//! The host classifier and the device kernel use the *same* CPU functions to
//! decide cell branch, so a future kernel landing cannot drift on tolerance.

use crate::cubic_cell_kernel::{DenestedCubicCell, ExactCellBranch, branch_cell};
use crate::gpu_kernels::cubic_cell::{
    CubicCellMomentStatus, GpuCellBranchTag, GpuDenestedCubicCell,
};

/// Result of classifying a single cell for GPU dispatch.
///
/// `Ok(tag)` is the branch the GPU dispatcher should send the cell to.
/// `Err(status)` indicates a cell-shape failure that the CPU classifier would
/// also reject; the substrate writes a zeroed moment row and the recorded
/// status code at the corresponding output position.
pub(crate) fn classify_cell_for_gpu(
    cell: GpuDenestedCubicCell,
) -> Result<GpuCellBranchTag, CubicCellMomentStatus> {
    if !is_finite_coefficient_block(cell) {
        return Err(CubicCellMomentStatus::NonFiniteCoefficient);
    }
    let left_inf = !cell.left.is_finite();
    let right_inf = !cell.right.is_finite();
    if left_inf && right_inf {
        if !is_affine_quadcubic_zero(cell) {
            return Err(CubicCellMomentStatus::NonAffineInfiniteInterval);
        }
        return Ok(GpuCellBranchTag::AffineTail);
    }
    if left_inf || right_inf {
        if !is_affine_quadcubic_zero(cell) {
            return Err(CubicCellMomentStatus::NonAffineInfiniteInterval);
        }
        return Ok(GpuCellBranchTag::AffineTail);
    }
    if cell.right <= cell.left {
        return Err(CubicCellMomentStatus::InvalidInterval);
    }
    let cpu_cell = DenestedCubicCell {
        left: cell.left,
        right: cell.right,
        c0: cell.c0,
        c1: cell.c1,
        c2: cell.c2,
        c3: cell.c3,
    };
    match branch_cell(cpu_cell) {
        Ok(ExactCellBranch::Affine) => Ok(GpuCellBranchTag::Affine),
        Ok(ExactCellBranch::Quartic) | Ok(ExactCellBranch::Sextic) => {
            Ok(GpuCellBranchTag::NonAffineFinite)
        }
        Err(_) => Err(CubicCellMomentStatus::InvalidInterval),
    }
}

#[inline]
fn is_finite_coefficient_block(cell: GpuDenestedCubicCell) -> bool {
    cell.c0.is_finite() && cell.c1.is_finite() && cell.c2.is_finite() && cell.c3.is_finite()
}

#[inline]
fn is_affine_quadcubic_zero(cell: GpuDenestedCubicCell) -> bool {
    // Match the CPU dispatcher's tolerance for "semi-infinite cell must be
    // affine" exactly so host and device agree byte-for-byte on which tails
    // are accepted.
    use crate::cubic_cell_kernel::NORMALIZED_CELL_BRANCH_TOL;
    cell.c2.abs() <= NORMALIZED_CELL_BRANCH_TOL && cell.c3.abs() <= NORMALIZED_CELL_BRANCH_TOL
}

#[cfg(test)]
mod tests {
    use super::*;

    fn cell(left: f64, right: f64, c0: f64, c1: f64, c2: f64, c3: f64) -> GpuDenestedCubicCell {
        GpuDenestedCubicCell {
            left,
            right,
            c0,
            c1,
            c2,
            c3,
        }
    }

    #[test]
    fn whole_line_affine_tail_is_affine_tail_when_quadcubic_zero() {
        let c = cell(f64::NEG_INFINITY, f64::INFINITY, 0.3, -0.4, 0.0, 0.0);
        assert_eq!(classify_cell_for_gpu(c), Ok(GpuCellBranchTag::AffineTail));
    }

    #[test]
    fn semi_infinite_with_curvature_is_rejected() {
        let c = cell(f64::NEG_INFINITY, 0.0, 0.1, 0.2, 0.5, 0.0);
        assert_eq!(
            classify_cell_for_gpu(c),
            Err(CubicCellMomentStatus::NonAffineInfiniteInterval)
        );
    }

    #[test]
    fn finite_affine_when_curvature_below_branch_tol() {
        // c2, c3 well below the normalized branch tolerance.
        let c = cell(-1.0, 1.0, 0.2, 0.3, 1e-14, 1e-14);
        assert_eq!(classify_cell_for_gpu(c), Ok(GpuCellBranchTag::Affine));
    }

    #[test]
    fn finite_quartic_routes_to_non_affine_finite() {
        let c = cell(-1.0, 1.0, 0.2, 0.3, 0.4, 0.0);
        assert_eq!(
            classify_cell_for_gpu(c),
            Ok(GpuCellBranchTag::NonAffineFinite)
        );
    }

    #[test]
    fn finite_sextic_routes_to_non_affine_finite() {
        let c = cell(-1.0, 1.0, 0.2, 0.3, 0.4, 0.5);
        assert_eq!(
            classify_cell_for_gpu(c),
            Ok(GpuCellBranchTag::NonAffineFinite)
        );
    }

    #[test]
    fn finite_with_reversed_bounds_is_invalid() {
        let c = cell(1.0, -1.0, 0.0, 0.0, 0.0, 0.0);
        assert_eq!(
            classify_cell_for_gpu(c),
            Err(CubicCellMomentStatus::InvalidInterval)
        );
    }

    #[test]
    fn nan_coefficient_is_rejected() {
        let c = cell(-1.0, 1.0, f64::NAN, 0.0, 0.0, 0.0);
        assert_eq!(
            classify_cell_for_gpu(c),
            Err(CubicCellMomentStatus::NonFiniteCoefficient)
        );
    }
}
