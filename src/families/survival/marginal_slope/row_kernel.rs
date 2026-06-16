//! The rigid per-row `RowKernel<4>` implementation and its Jacobian-action
//! assembly helpers: the memory-efficient row-at-a-time kernel used on the
//! no-flex hot path.

use super::*;

// ── RowKernel<4> implementation ───────────────────────────────────────

pub(crate) struct SurvivalMarginalSlopeRowKernel {
    pub(crate) family: SurvivalMarginalSlopeFamily,
    pub(crate) block_states: Vec<ParameterBlockState>,
    pub(crate) slices: BlockSlices,
}

impl SurvivalMarginalSlopeRowKernel {
    pub(crate) fn new(
        family: SurvivalMarginalSlopeFamily,
        block_states: Vec<ParameterBlockState>,
    ) -> Self {
        let slices = block_slices(&family, &block_states);
        Self {
            family,
            block_states,
            slices,
        }
    }
}

impl RowKernel<4> for SurvivalMarginalSlopeRowKernel {
    fn n_rows(&self) -> usize {
        self.family.n
    }
    fn n_coefficients(&self) -> usize {
        self.slices.total
    }

    fn row_kernel(&self, row: usize) -> Result<(f64, [f64; 4], [[f64; 4]; 4]), String> {
        let beta_time = &self.block_states[0].beta;
        let q0 = self.family.design_entry.dot_row(row, beta_time)
            + self.family.offset_entry[row]
            + self.block_states[1].eta[row];
        let q1 = self.family.design_exit.dot_row(row, beta_time)
            + self.family.offset_exit[row]
            + self.block_states[1].eta[row];
        let qd1 = self.family.design_derivative_exit.dot_row(row, beta_time)
            + self.family.derivative_offset_exit[row];
        self.family.row_primary_closed_form_rigid(
            row,
            q0,
            q1,
            qd1,
            &self.block_states,
            self.family.probit_frailty_scale(),
        )
    }

    fn jacobian_action(&self, row: usize, d_beta: &[f64]) -> [f64; 4] {
        let d_beta = ndarray::ArrayView1::from(d_beta);
        let d_time = d_beta.slice(s![self.slices.time.clone()]);
        let d_marginal = d_beta.slice(s![self.slices.marginal.clone()]);
        let d_logslope = d_beta.slice(s![self.slices.logslope.clone()]);
        [
            self.family.design_entry.dot_row_view(row, d_time)
                + self.family.marginal_design.dot_row_view(row, d_marginal),
            self.family.design_exit.dot_row_view(row, d_time)
                + self.family.marginal_design.dot_row_view(row, d_marginal),
            self.family.design_derivative_exit.dot_row_view(row, d_time),
            self.family.logslope_design.dot_row_view(row, d_logslope),
        ]
    }

    fn jacobian_action_matrix(&self, factor: ArrayView2<'_, f64>) -> Option<Array2<f64>> {
        if factor.nrows() != self.slices.total {
            return None;
        }
        let n_rows = self.family.n;
        // Whole-projection build: each axis uses the batched design matvec
        // (`fast_ab` on dense, one operator `dot` per column on operator-backed
        // designs).
        Some(self.assemble_jf(factor, n_rows, |design, factor_block| {
            survival_axis_jf_via_design(design, factor_block, n_rows)
        }))
    }

    fn jacobian_action_matrix_rows(
        &self,
        factor: ArrayView2<'_, f64>,
        start: usize,
        end: usize,
    ) -> Array2<f64> {
        if factor.nrows() != self.slices.total {
            // Shape contract broken (the tiled trace always passes the
            // coefficient-width factor, so this is defensive only): fall back
            // to the exact generic per-row build over the range.
            return crate::families::row_kernel::row_kernel_jacobian_action_matrix_generic_rows(
                self, factor, start, end,
            );
        }
        // Block-tiled build for one row-tile: dense designs slice to a
        // contiguous row block and GEMM (`fast_ab`), operator/sparse designs
        // fall to a row-local dot over the range. Bounds peak memory to the
        // tile while keeping BLAS-3 on the materialized designs.
        let b = end.saturating_sub(start);
        self.assemble_jf(factor, b, |design, factor_block| {
            survival_axis_jf_via_design_rows(design, factor_block, start, end)
        })
    }

    fn jacobian_transpose_action(&self, row: usize, v: &[f64; 4], out: &mut [f64]) {
        {
            let mut time = ndarray::ArrayViewMut1::from(&mut out[self.slices.time.clone()]);
            self.family
                .design_entry
                .axpy_row_into(row, v[0], &mut time)
                .expect("time entry axpy dim mismatch");
            self.family
                .design_exit
                .axpy_row_into(row, v[1], &mut time)
                .expect("time exit axpy dim mismatch");
            self.family
                .design_derivative_exit
                .axpy_row_into(row, v[2], &mut time)
                .expect("time deriv axpy dim mismatch");
        }
        {
            let mut marginal = ndarray::ArrayViewMut1::from(&mut out[self.slices.marginal.clone()]);
            self.family
                .marginal_design
                .axpy_row_into(row, v[0] + v[1], &mut marginal)
                .expect("marginal axpy dim mismatch");
        }
        {
            let mut logslope = ndarray::ArrayViewMut1::from(&mut out[self.slices.logslope.clone()]);
            self.family
                .logslope_design
                .axpy_row_into(row, v[3], &mut logslope)
                .expect("logslope axpy dim mismatch");
        }
    }

    fn add_pullback_hessian(&self, row: usize, h: &[[f64; 4]; 4], target: &mut Array2<f64>) {
        let mut h_arr = Array2::<f64>::zeros((4, 4));
        for a in 0..4 {
            for b in 0..4 {
                h_arr[[a, b]] = h[a][b];
            }
        }
        self.family
            .add_pullback_primary_hessian(target, row, &self.slices, &h_arr);
    }

    fn add_diagonal_quadratic(&self, row: usize, h: &[[f64; 4]; 4], diag: &mut [f64]) {
        let designs: [(usize, &DesignMatrix); 3] = [
            (0, &self.family.design_entry),
            (1, &self.family.design_exit),
            (2, &self.family.design_derivative_exit),
        ];
        for &(pi, des) in &designs {
            {
                let mut td = ndarray::ArrayViewMut1::from(&mut diag[self.slices.time.clone()]);
                des.squared_axpy_row_into(row, h[pi][pi], &mut td)
                    .expect("time squared_axpy dim mismatch");
            }
            for &(pj, des_j) in &designs {
                if pj <= pi {
                    continue;
                }
                let mut td = ndarray::ArrayViewMut1::from(&mut diag[self.slices.time.clone()]);
                des.crossdiag_axpy_row_into(row, des_j, 2.0 * h[pi][pj], &mut td)
                    .expect("time crossdiag dim mismatch");
            }
        }
        {
            let alpha = h[0][0] + 2.0 * h[0][1] + h[1][1];
            let mut md = ndarray::ArrayViewMut1::from(&mut diag[self.slices.marginal.clone()]);
            self.family
                .marginal_design
                .squared_axpy_row_into(row, alpha, &mut md)
                .expect("marginal squared_axpy dim mismatch");
        }
        {
            let mut gd = ndarray::ArrayViewMut1::from(&mut diag[self.slices.logslope.clone()]);
            self.family
                .logslope_design
                .squared_axpy_row_into(row, h[3][3], &mut gd)
                .expect("logslope squared_axpy dim mismatch");
        }
    }

    fn row_third_contracted(&self, row: usize, dir: &[f64; 4]) -> Result<[[f64; 4]; 4], String> {
        // Batched path: one k=5 MultiDirJet [e_0..e_3, dir] covers all 6
        // off-diagonal (a,b) entries via mask reads; 4 small k=3 jets handle
        // diagonal ∂²_{e_a} ∂_{dir} entries. Replaces the legacy 10 separate
        // calls into row_neglog_directional_refs.
        let dir_view = ndarray::aview1(&dir[..]);
        self.family
            .row_primary_third_contracted_batched(row, &self.block_states, dir_view)
    }

    fn row_fourth_contracted(
        &self,
        row: usize,
        dir_u: &[f64; 4],
        dir_v: &[f64; 4],
    ) -> Result<[[f64; 4]; 4], String> {
        // Batched path: one k=6 MultiDirJet [e_0..e_3, dir_u, dir_v] covers
        // the 6 off-diagonal (a,b) entries; 4 k=4 jets handle the diagonal
        // ∂²_{e_a} ∂_{dir_u} ∂_{dir_v} entries.
        let u_view = ndarray::aview1(&dir_u[..]);
        let v_view = ndarray::aview1(&dir_v[..]);
        self.family
            .row_primary_fourth_contracted_batched(row, &self.block_states, u_view, v_view)
    }
}

impl SurvivalMarginalSlopeRowKernel {
    /// Assemble the `(n_out × 4·rank)` joint Jacobian-action projection `Jᵢ · F`
    /// from the four primary axes — `[entry+marginal | exit+marginal |
    /// derivative | logslope]` — given a per-axis builder `axis(design,
    /// factor_block)` that produces that design's `n_out × rank` contribution.
    /// The whole-projection path passes the batched builder; the block-tiled
    /// path passes the row-range builder. Either way at most one axis transient
    /// is alive at a time: the marginal block feeds both the entry and exit
    /// axes, so it is built once and dropped, and every other axis is a
    /// statement-scoped temporary — keeping the assembly peak at
    /// `output + one n_out×rank block` rather than five blocks at once.
    pub(crate) fn assemble_jf<F>(
        &self,
        factor: ArrayView2<'_, f64>,
        n_out: usize,
        axis: F,
    ) -> Array2<f64>
    where
        F: Fn(&DesignMatrix, ArrayView2<'_, f64>) -> Array2<f64>,
    {
        let rank = factor.ncols();
        if rank == 0 {
            return Array2::<f64>::zeros((n_out, 0));
        }
        let f_time = factor.slice(s![self.slices.time.clone(), ..]);
        let f_marginal = factor.slice(s![self.slices.marginal.clone(), ..]);
        let f_logslope = factor.slice(s![self.slices.logslope.clone(), ..]);

        let mut jf = Array2::<f64>::zeros((n_out, 4 * rank));
        {
            let jf_marginal = axis(&self.family.marginal_design, f_marginal);
            {
                let mut axis0 = jf.slice_mut(s![.., 0..rank]);
                axis0.assign(&axis(&self.family.design_entry, f_time));
                axis0 += &jf_marginal;
            }
            {
                let mut axis1 = jf.slice_mut(s![.., rank..2 * rank]);
                axis1.assign(&axis(&self.family.design_exit, f_time));
                axis1 += &jf_marginal;
            }
        }
        jf.slice_mut(s![.., 2 * rank..3 * rank])
            .assign(&axis(&self.family.design_derivative_exit, f_time));
        jf.slice_mut(s![.., 3 * rank..4 * rank])
            .assign(&axis(&self.family.logslope_design, f_logslope));
        jf
    }
}

pub(crate) fn survival_axis_jf_via_design(
    design: &DesignMatrix,
    factor_block: ArrayView2<'_, f64>,
    n_rows: usize,
) -> Array2<f64> {
    let rank = factor_block.ncols();
    if rank == 0 {
        return Array2::<f64>::zeros((n_rows, 0));
    }
    let factor = factor_block.as_standard_layout().into_owned();
    match design.as_dense_ref() {
        Some(dense) => fast_ab(dense, &factor),
        None => {
            let mut out = Array2::<f64>::zeros((n_rows, rank));
            for c in 0..rank {
                let result = design.dot(&factor.column(c).to_owned());
                out.column_mut(c).assign(&result);
            }
            out
        }
    }
}

/// Row-range analogue of [`survival_axis_jf_via_design`]: one design's
/// `(end-start) × rank` Jacobian-action block over rows `[start, end)`. Dense
/// designs slice to a contiguous row block and GEMM via `fast_ab` (BLAS-3,
/// zero-copy slice); sparse/operator designs fall to a row-local
/// `dot_row_view` over the range (each entry touches only its own row, so
/// tiling never re-walks the full operator). Used by the block-tiled trace to
/// hold one row-tile at a time instead of the whole `n × rank` projection.
pub(crate) fn survival_axis_jf_via_design_rows(
    design: &DesignMatrix,
    factor_block: ArrayView2<'_, f64>,
    start: usize,
    end: usize,
) -> Array2<f64> {
    let b = end.saturating_sub(start);
    let rank = factor_block.ncols();
    if rank == 0 {
        return Array2::<f64>::zeros((b, 0));
    }
    let factor = factor_block.as_standard_layout().into_owned();
    match design.as_dense_ref() {
        Some(dense) => {
            let block = dense.slice(s![start..end, ..]);
            fast_ab(&block, &factor)
        }
        None => {
            let mut out = Array2::<f64>::zeros((b, rank));
            for (i, row) in (start..end).enumerate() {
                for c in 0..rank {
                    out[[i, c]] = design.dot_row_view(row, factor.column(c));
                }
            }
            out
        }
    }
}
