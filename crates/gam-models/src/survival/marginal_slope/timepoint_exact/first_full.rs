//! The shared moving-boundary flux helper for the test-only hand oracle.
//!
//! Both the grad-only flex path and the full value/grad/Hessian + contracted
//! base/extensions are now jet-sourced through the ONE `flex_jet` builder
//! (`compute_survival_timepoint_first_order_exact` at [`Jet1`], the value/grad/
//! Hessian pack at `Jet2`). The hand first-order and full second-order packs
//! (`compute_survival_timepoint_exact_from_cached` and its D-path builders) survive
//! only as the FD/parity ORACLE in `first_full_exact_oracle_tests`.
//! `moving_density_boundary_flux` is the moving-edge flux those oracle assemblers
//! (and the directional/bidirectional oracle tests) share, so it stays here.

use super::*;

#[cfg(test)]
pub(super) fn moving_density_boundary_flux(
    axis: usize,
    primary: &FlexPrimarySlices,
    a_u: &Array1<f64>,
    entry: &CachedCellEntry,
    poly: &[f64],
    b: f64,
    include_intercept: bool,
) -> f64 {
    // The link crossing `z = (τ − a)/b` moves with `θ` both directly (through
    // `b = g`) and through the intercept response `a(θ)` at velocity `a_u`. Two
    // distinct quantities need this flux with two distinct velocities:
    //
    // * `include_intercept = true` — `∂z/∂θ_axis = −(a_u[axis] + direct_g)/b`,
    //   the genuine TOTAL z-motion, used by the first-derivative `d_u` (which
    //   also carries the `chi·a_u` intercept chain in its interior, so it is a
    //   consistent total derivative and is FD-verified correct).
    // * `include_intercept = false` — `∂z/∂θ_axis|_a = −direct_g/b`, the PARTIAL
    //   z-motion at fixed intercept, used by the IFT partials `f_uv`/`f_au`
    //   (whose interiors use partial cell coefficients, a held fixed). The
    //   intercept-chain contribution to `a_uv` is supplied separately by the
    //   explicit `f_au·a_u + f_aa·a_u²` terms in the IFT recovery; feeding the
    //   total velocity here double-counts the intercept motion (gam#1454).
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
