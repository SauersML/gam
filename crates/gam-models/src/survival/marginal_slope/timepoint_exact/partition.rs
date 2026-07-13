//! Cached piecewise-cubic partition build.
//!
//! Builds the cells + per-cell moment states + fixed partials once per
//! `(a, b, β_h, β_w)` so the first-order / full / directional / bidirectional
//! integration passes (F, D, D_uv) all share one canonical partition.

use super::*;

impl SurvivalMarginalSlopeFamily {
    /// Build a cached partition: cells + moment states + fixed partials,
    /// computed once per (a, b, β_h, β_w) and reused across the three
    /// integration passes (F, D, D_uv).
    pub(crate) fn build_cached_partition_with_moment_order(
        &self,
        primary: &FlexPrimarySlices,
        a: f64,
        b: f64,
        beta_h: Option<&Array1<f64>>,
        beta_w: Option<&Array1<f64>>,
        moment_order: usize,
    ) -> Result<CachedPartitionCells, String> {
        let raw_cells = self.denested_partition_cells(a, b, beta_h, beta_w)?;

        // Per-cell prelude (z_mid, u_mid, moment state).
        let n = raw_cells.len();
        let mut z_mids = Vec::with_capacity(n);
        let mut u_mids = Vec::with_capacity(n);
        let mut states = Vec::with_capacity(n);
        for partition_cell in &raw_cells {
            let cell = partition_cell.cell;
            let z_mid = exact_kernel::interval_probe_point(cell.left, cell.right)?;
            let u_mid = a + b * z_mid;
            let state = exact_kernel::evaluate_cell_moments(cell, moment_order)?;
            z_mids.push(z_mid);
            u_mids.push(u_mid);
            states.push(state);
        }

        // Canonical per-cell fixed partials.
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
        // The flex moment closure expands `S(z)=e^{−Δq}=Σ_{k≤4}(−Δq)^k/k!`; with η
        // cubic, `−Δq=½(η²−η₀²)` is degree-6 in z, so the fourth-order `(−Δq)⁴` term
        // reaches `M_{n+24}` (= `M_28` for the `n≤4` base moments). A moment order of
        // 27 silently dropped `M_28`, truncating the contracted-fourth (Jet4) channel.
        // Build to 32 so every order of the e^{−Δq} closure has its full moment dot.
        self.build_cached_partition_with_moment_order(primary, a, b, beta_h, beta_w, 32)
    }
}
