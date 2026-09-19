//! Cached piecewise-cubic partition build.
//!
//! Builds the cells + per-cell moment states + fixed partials once per
//! `(a, b, β_h, β_w)` so the first-order / full / directional / bidirectional
//! integration passes (F, D, D_uv) all share one canonical partition. A family
//! anchored on a declared finite law caches that law's nodes instead (gam#2948).

use super::*;

impl SurvivalMarginalSlopeFamily {
    /// Build a cached partition: cells + moment states + fixed partials,
    /// computed once per (a, b, β_h, β_w) and reused across the three
    /// integration passes (F, D, D_uv). On a declared finite law it is row
    /// `row`'s nodes.
    pub(crate) fn build_cached_partition_with_moment_order(
        &self,
        row: usize,
        primary: &FlexPrimarySlices,
        a: f64,
        b: f64,
        beta_h: Option<&Array1<f64>>,
        beta_w: Option<&Array1<f64>>,
        moment_order: usize,
    ) -> Result<CachedPartitionCells, String> {
        if let Some(grid) = self.flex_law_grid(Some(row))? {
            return Ok(CachedPartitionCells::Law {
                nodes: self.build_law_nodes(grid, primary, a, b, beta_h, beta_w)?,
                scale: self.probit_frailty_scale(),
            });
        }
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
        Ok(CachedPartitionCells::Gaussian(cells))
    }

    pub(crate) fn build_cached_partition(
        &self,
        row: usize,
        primary: &FlexPrimarySlices,
        a: f64,
        b: f64,
        beta_h: Option<&Array1<f64>>,
        beta_w: Option<&Array1<f64>>,
    ) -> Result<CachedPartitionCells, String> {
        self.build_cached_partition_with_moment_order(
            row,
            primary,
            a,
            b,
            beta_h,
            beta_w,
            super::flex_jet::FLEX_ORDER_FOUR_MOMENT_DEGREE,
        )
    }

    /// A declared law's nodes at one timepoint (gam#2948): the score warp and
    /// its basis at every node `u_k`, and the link deviation and its basis at
    /// `U = a + b·u_k`, each as the derivative stack of its span's cubic. A basis
    /// function that is zero there reaches no channel and is not kept.
    fn build_law_nodes(
        &self,
        grid: AnchorGrid<'_>,
        primary: &FlexPrimarySlices,
        a: f64,
        b: f64,
        beta_h: Option<&Array1<f64>>,
        beta_w: Option<&Array1<f64>>,
    ) -> Result<Vec<CachedLawNode>, String> {
        let mut nodes = Vec::with_capacity(grid.len());
        for (&node, &weight) in grid.nodes.iter().zip(grid.weights) {
            let mut score_value = 0.0;
            let mut score_basis = Vec::new();
            if let (Some(runtime), Some(range)) = (self.score_warp.as_ref(), primary.h.as_ref()) {
                if let Some(beta) = beta_h {
                    score_value = runtime.local_cubic_at(beta.view(), node)?.evaluate(node);
                }
                runtime.for_each_basis_cubic_at(node, |basis, span| {
                    let value = span.evaluate(node);
                    if value != 0.0 {
                        score_basis.push((range.start + basis, value));
                    }
                    Ok(())
                })?;
            }
            let u = a + b * node;
            let mut link_stack = [0.0; 4];
            let mut link_basis = Vec::new();
            if let (Some(runtime), Some(range)) = (self.link_dev.as_ref(), primary.w.as_ref()) {
                if let Some(beta) = beta_w {
                    link_stack = span_derivative_stack(runtime.local_cubic_at(beta.view(), u)?, u);
                }
                runtime.for_each_basis_cubic_at(u, |basis, span| {
                    let stack = span_derivative_stack(span, u);
                    if stack.iter().any(|&entry| entry != 0.0) {
                        link_basis.push((range.start + basis, stack));
                    }
                    Ok(())
                })?;
            }
            nodes.push(CachedLawNode {
                node,
                weight,
                score_value,
                score_basis,
                link_stack,
                link_basis,
            });
        }
        Ok(nodes)
    }
}

/// `[c(x), c′(x), c″(x), c‴(x)]` of a span's cubic at `x`.
fn span_derivative_stack(span: exact_kernel::LocalSpanCubic, x: f64) -> [f64; 4] {
    [
        span.evaluate(x),
        span.first_derivative(x),
        span.second_derivative(x),
        6.0 * span.c3,
    ]
}
