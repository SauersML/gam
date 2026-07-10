#![cfg(test)]

//! Shared moving-boundary flux helper for the survival timepoint hand oracles.

use super::*;

pub(super) fn moving_density_boundary_flux(
    axis: usize,
    primary: &FlexPrimarySlices,
    a_u: &Array1<f64>,
    entry: &CachedCellEntry,
    poly: &[f64],
    b: f64,
    include_intercept: bool,
) -> f64 {
    if b == 0.0 {
        return 0.0;
    }
    let cell = entry.partition_cell.cell;
    let edge_velocity = |edge: crate::cubic_cell_kernel::PartitionEdge, z: f64| -> f64 {
        match edge {
            crate::cubic_cell_kernel::PartitionEdge::Crossing { .. } => {
                let direct_g = if axis == primary.g { z } else { 0.0 };
                let intercept = if include_intercept { a_u[axis] } else { 0.0 };
                -(intercept + direct_g) / b
            }
            crate::cubic_cell_kernel::PartitionEdge::Fixed(_) => 0.0,
        }
    };
    let v_r = edge_velocity(entry.partition_cell.right_edge, cell.right);
    let v_l = edge_velocity(entry.partition_cell.left_edge, cell.left);
    let right = if v_r != 0.0 {
        v_r * crate::cubic_cell_kernel::cell_density_boundary_integrand(cell, poly, cell.right)
    } else {
        0.0
    };
    let left = if v_l != 0.0 {
        v_l * crate::cubic_cell_kernel::cell_density_boundary_integrand(cell, poly, cell.left)
    } else {
        0.0
    };
    right - left
}
