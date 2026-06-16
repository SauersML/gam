//! Cached piecewise-cubic partition build.
//!
//! Builds the cells + per-cell moment states + fixed partials once per
//! `(a, b, β_h, β_w)` so the first-order / full / directional / bidirectional
//! integration passes (F, D, D_uv) all share one partition. The cell-table and
//! per-cell fixed-partials assemblies route through the GPU-shaped `try_device_*`
//! seams, falling back to the CPU implementation on decline.

use super::*;

impl SurvivalMarginalSlopeFamily {
    /// Build a cached partition: cells + moment states + fixed partials,
    /// computed once per (a, b, β_h, β_w) and reused across the three
    /// integration passes (F, D, D_uv).
    ///
    /// The cell-table assembly and the per-cell primary-fixed-partials
    /// assembly route through the GPU-shaped `try_device_*` seams in
    /// [`crate::families::survival::marginal_slope::gpu_prep`].  Until the matching NVRTC kernels
    /// land, both seams return `Ok(None)` and the call site falls back to
    /// the existing CPU implementation, so behavior is preserved.
    pub(crate) fn build_cached_partition_with_moment_order(
        &self,
        primary: &FlexPrimarySlices,
        a: f64,
        b: f64,
        beta_h: Option<&Array1<f64>>,
        beta_w: Option<&Array1<f64>>,
        moment_order: usize,
    ) -> Result<CachedPartitionCells, String> {
        // ── 1. partition cells via the device seam, CPU fallback on decline ──
        let raw_cells = {
            let row_input =
                crate::families::survival::marginal_slope::gpu_prep::PartitionCellsRowInputs {
                    a,
                    b,
                    beta_h: beta_h.and_then(|b| b.as_slice()),
                    beta_w: beta_w.and_then(|b| b.as_slice()),
                };
            let dev =
                crate::families::survival::marginal_slope::gpu_prep::try_device_partition_cells(
                    std::slice::from_ref(&row_input),
                )
                .map_err(|e| e.to_string())?;
            match dev {
                Some(mut by_row) if by_row.len() == 1 => by_row.remove(0),
                _ => self.denested_partition_cells(a, b, beta_h, beta_w)?,
            }
        };

        // ── 2. per-cell prelude (neg_cell, z_mid, u_mid, moment state) ──
        let n = raw_cells.len();
        let mut neg_cells = Vec::with_capacity(n);
        let mut z_mids = Vec::with_capacity(n);
        let mut u_mids = Vec::with_capacity(n);
        let mut states = Vec::with_capacity(n);
        let mut fp_inputs = Vec::<
            crate::families::survival::marginal_slope::gpu_prep::CellPrimaryFixedPartialsCellInputs,
        >::with_capacity(n);
        for partition_cell in &raw_cells {
            let cell = partition_cell.cell;
            let neg_cell = exact_kernel::DenestedCubicCell {
                left: cell.left,
                right: cell.right,
                c0: -cell.c0,
                c1: -cell.c1,
                c2: -cell.c2,
                c3: -cell.c3,
            };
            let z_mid = exact_kernel::interval_probe_point(cell.left, cell.right)?;
            let u_mid = a + b * z_mid;
            let state = exact_kernel::evaluate_cell_moments(cell, moment_order)?;
            neg_cells.push(neg_cell);
            z_mids.push(z_mid);
            u_mids.push(u_mid);
            states.push(state);
            fp_inputs.push(
                crate::families::survival::marginal_slope::gpu_prep::CellPrimaryFixedPartialsCellInputs {
                    score_span: partition_cell.score_span,
                    link_span: partition_cell.link_span,
                },
            );
        }

        // ── 3. per-cell fixed partials via the device seam, CPU fallback ──
        let layout = crate::families::survival::marginal_slope::gpu_prep::FlexPrimaryLayout {
            r: u32::try_from(primary.total).map_err(|_| {
                format!(
                    "build_cached_partition_with_moment_order: primary.total={} exceeds u32",
                    primary.total
                )
            })?,
            g_slot: u32::try_from(primary.g).map_err(|_| {
                format!(
                    "build_cached_partition_with_moment_order: primary.g={} exceeds u32",
                    primary.g
                )
            })?,
        };
        let row_fp_input =
            crate::families::survival::marginal_slope::gpu_prep::CellPrimaryFixedPartialsRowInputs {
                cells: &fp_inputs,
                layout,
            };
        let dev_fixed = crate::families::survival::marginal_slope::gpu_prep::try_device_cell_primary_fixed_partials(
            std::slice::from_ref(&row_fp_input),
        )
        .map_err(|e| e.to_string())?;
        // When the device path returns flat-packed partials, reconstruct
        // the per-cell `DenestedCellPrimaryFixedPartials` from the device
        // buffer via the `from_flat_slice` shim — byte-identical to what
        // the CPU per-cell helper would produce for the supported
        // (trivial-span) shape.  Any decline drops through to the CPU
        // per-cell loop below.
        if let Some(out) = dev_fixed.as_ref()
            && out.partials.len() == 1
            && out.partials[0].len() == n
        {
            let mut cells = Vec::with_capacity(n);
            for (idx, partition_cell) in raw_cells.into_iter().enumerate() {
                let flat = &out.partials[0][idx];
                let fixed = DenestedCellPrimaryFixedPartials::from_flat_slice(
                    flat.as_slice(),
                    primary.total,
                )?;
                cells.push(CachedCellEntry {
                    partition_cell,
                    neg_cell: neg_cells[idx],
                    state: states[idx].clone(),
                    fixed,
                });
            }
            return Ok(CachedPartitionCells { cells });
        }
        let mut cells = Vec::with_capacity(n);
        for (idx, partition_cell) in raw_cells.into_iter().enumerate() {
            let fixed = self.denested_cell_primary_fixed_partials(
                primary,
                a,
                b,
                partition_cell.score_span,
                partition_cell.link_span,
                z_mids[idx],
                u_mids[idx],
            )?;
            cells.push(CachedCellEntry {
                partition_cell,
                neg_cell: neg_cells[idx],
                state: states[idx].clone(),
                fixed,
            });
        }
        Ok(CachedPartitionCells { cells })
    }

    pub(crate) fn build_cached_partition(
        &self,
        primary: &FlexPrimarySlices,
        a: f64,
        b: f64,
        beta_h: Option<&Array1<f64>>,
        beta_w: Option<&Array1<f64>>,
    ) -> Result<CachedPartitionCells, String> {
        self.build_cached_partition_with_moment_order(primary, a, b, beta_h, beta_w, 24)
    }
}
