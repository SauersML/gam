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

pub(crate) fn rigid_row_kernel_primaries(
    family: &SurvivalMarginalSlopeFamily,
    block_states: &[ParameterBlockState],
    row: usize,
) -> Result<[f64; 4], String> {
    let q_geom = family.row_dynamic_q_values(row, block_states)?;
    Ok([q_geom.q0, q_geom.q1, q_geom.qd1, block_states[2].eta[row]])
}

pub(crate) fn rigid_row_kernel_nll_tower(
    family: &SurvivalMarginalSlopeFamily,
    block_states: &[ParameterBlockState],
    row: usize,
    p: &[crate::families::jet_tower::Tower4<4>; 4],
    context: &str,
) -> Result<crate::families::jet_tower::Tower4<4>, String> {
    use crate::families::jet_tower::Tower4;

    let wi = family.weights[row];
    let di = family.event[row];
    let (z_sum, covariance_ones) = family.exact_shared_score_summary(row, block_states, context)?;
    let probit_scale = family.probit_frailty_scale();

    let q0 = p[0];
    let q1 = p[1];
    let qd1 = p[2];
    let g = p[3];

    let observed_g = g * probit_scale;
    let one_plus_b2 = observed_g * observed_g * covariance_ones + 1.0;
    let c = one_plus_b2.compose_unary(unary_derivatives_sqrt(one_plus_b2.v));

    let eta0 = q0 * c + observed_g * z_sum;
    let eta1 = q1 * c + observed_g * z_sum;
    let ad1 = qd1 * c;

    let qd1_lower = family.time_derivative_lower_bound();
    if survival_derivative_guard_violated(qd1.v, qd1_lower) {
        return Err(SurvivalMarginalSlopeError::MonotonicityViolation {
            reason: format!(
                "survival marginal-slope monotonicity violated at row {row}: raw time derivative={:.3e} must be at least derivative_guard={:.3e}; transformed time derivative={:.3e}",
                qd1.v, qd1_lower, ad1.v
            ),
        }
        .into());
    }

    // Mirror the exact closed-form contract
    // (`signed_probit_neglog_derivatives_up_to_fourth`): the saturated `+∞`
    // tail is the legitimate zero-survival limit, but `-∞`/NaN signed margins
    // are domain failures that must surface as an error rather than being
    // masked into a NaN/∞-laden derivative stack by `unary_derivatives_neglog_phi`.
    // The guard respects zero weight (those terms drop out entirely).
    let reject_nonfinite_margin = |margin: f64, weight: f64| -> Result<(), String> {
        if weight != 0.0 && margin != f64::INFINITY && !margin.is_finite() {
            Err(SurvivalMarginalSlopeError::NumericalFailure {
                reason: format!(
                    "non-finite signed margin in rigid survival marginal-slope row tower at row {row}: {margin}"
                ),
            }
            .into())
        } else {
            Ok(())
        }
    };

    let neg_eta0 = -eta0;
    reject_nonfinite_margin(neg_eta0.v, wi)?;
    let entry = neg_eta0
        .compose_unary(unary_derivatives_neglog_phi(neg_eta0.v, wi))
        .scale(-1.0);

    let neg_eta1 = -eta1;
    reject_nonfinite_margin(neg_eta1.v, wi * (1.0 - di))?;
    let exit = neg_eta1.compose_unary(unary_derivatives_neglog_phi(neg_eta1.v, wi * (1.0 - di)));

    let event_density = if di > 0.0 {
        eta1.compose_unary(unary_derivatives_log_normal_pdf(eta1.v))
            .scale(-wi * di)
    } else {
        Tower4::<4>::zero()
    };

    let time_deriv = if di > 0.0 {
        ad1.compose_unary(unary_derivatives_log(ad1.v))
            .scale(-wi * di)
    } else {
        Tower4::<4>::zero()
    };

    Ok(exit + entry + event_density + time_deriv)
}

impl crate::families::jet_tower::RowNllProgram<4> for SurvivalMarginalSlopeRowKernel {
    fn n_rows(&self) -> usize {
        self.family.n
    }

    fn primaries(&self, row: usize) -> Result<[f64; 4], String> {
        rigid_row_kernel_primaries(&self.family, &self.block_states, row)
    }

    fn row_nll(
        &self,
        row: usize,
        p: &[crate::families::jet_tower::Tower4<4>; 4],
    ) -> Result<crate::families::jet_tower::Tower4<4>, String> {
        rigid_row_kernel_nll_tower(
            &self.family,
            &self.block_states,
            row,
            p,
            "survival marginal-slope rigid row tower",
        )
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
        crate::families::jet_tower::derived_row_kernel(self, row)
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
            crate::families::row_kernel::row_kernel_design_jf(design, factor_block, n_rows)
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
            crate::families::row_kernel::row_kernel_design_jf_rows(design, factor_block, start, end)
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
        crate::families::jet_tower::derived_third_contracted(self, row, dir)
    }

    fn row_fourth_contracted(
        &self,
        row: usize,
        dir_u: &[f64; 4],
        dir_v: &[f64; 4],
    ) -> Result<[[f64; 4]; 4], String> {
        crate::families::jet_tower::derived_fourth_contracted(self, row, dir_u, dir_v)
    }

    /// Batched all-axes FIRST directional derivative of the joint Hessian for
    /// the rigid survival marginal-slope kernel (gam#979).
    ///
    /// The generic per-axis fall-back (`row_kernel_directional_derivative_all_axes`)
    /// asks for `Hdot[e_a]` `p` separate times, and EACH per-axis sweep rebuilds
    /// the per-row fourth-order `Tower4<4>` for every row inside
    /// `row_third_contracted` (`evaluate_program` → full t3/t4 build) — `n·p`
    /// tower evaluations per all-axes call. For survival the tower is the
    /// expensive object (closed-form probit/log-pdf composition over four
    /// primaries), so this is the #979 inner-Newton Jeffreys/Firth hot path.
    ///
    /// This override builds each row's `t3` tensor ONCE (the swept axis enters
    /// only through the cheap primary projection `dir_a = Jᵢ·e_a` and the linear
    /// `t3.third_contracted(dir_a)`), then closes every axis off that single
    /// build. Crucially it reuses the kernel's OWN `jacobian_action`,
    /// `Tower4::third_contracted`, and `add_pullback_hessian` in the EXACT SAME
    /// `ARROW_ROW_CHUNK`-chunked reduction order as the generic per-axis path
    /// (`par_try_reduce_fold(RowSet::All)`): the cached `t3[row]` is bit-for-bit
    /// the tensor a fresh `evaluate_program(row)` would produce (a deterministic
    /// pure function of the row), and every float op downstream is identical, so
    /// axis `a` matches `row_kernel_directional_derivative(self, All, e_a)`
    /// bit-for-bit. Only the redundant `(p−1)·n` tower rebuilds are removed.
    ///
    /// Claims only the full-data unit-weight `RowSet::All` case; otherwise
    /// returns `None` so the generic per-axis Horvitz-Thompson sweep runs.
    fn directional_derivative_all_axes_dense_override(
        &self,
        rows: &crate::families::row_kernel::RowSet,
        p: usize,
    ) -> Option<Result<Vec<Array2<f64>>, String>> {
        if p != self.n_coefficients() {
            return Some(Err(format!(
                "survival marginal-slope directional_derivative_all_axes_dense_override: \
                 axis count {p} disagrees with n_coefficients() {}",
                self.n_coefficients(),
            )));
        }
        if !matches!(rows, crate::families::row_kernel::RowSet::All) {
            return None;
        }
        Some(self.directional_derivative_all_axes_build_once(p))
    }

    /// Batched all-axes SECOND directional derivative of the joint Hessian for
    /// the rigid survival marginal-slope kernel (gam#979): the outer-REML
    /// Jeffreys `H_Φ` drift analogue of the first-order override above.
    ///
    /// With `d_beta_u` fixed and the second direction sweeping every canonical
    /// axis, the generic per-axis path runs `p` full-data sweeps each rebuilding
    /// the per-row `Tower4<4>` (`row_fourth_contracted` → `evaluate_program`).
    /// This override builds each row's `t4` tensor and the fixed-direction
    /// projection `dir_u = Jᵢ·u` ONCE, then closes every axis with the cheap
    /// linear `t4.fourth_contracted(dir_u, dir_a)` and the kernel's own
    /// `add_pullback_hessian`, in the SAME chunked reduction order as
    /// `row_kernel_second_directional_derivative(self, All, u, e_a)` — bit-for-bit
    /// identical, only the redundant tower rebuilds removed.
    ///
    /// Claims only the full-data unit-weight `RowSet::All` case; otherwise `None`.
    fn second_directional_derivative_all_axes_dense_override(
        &self,
        rows: &crate::families::row_kernel::RowSet,
        d_beta_u: &[f64],
    ) -> Option<Result<Vec<Array2<f64>>, String>> {
        if d_beta_u.len() != self.n_coefficients() {
            return Some(Err(format!(
                "survival marginal-slope second_directional_derivative_all_axes_dense_override: \
                 fixed direction has {} entries, expected {}",
                d_beta_u.len(),
                self.n_coefficients(),
            )));
        }
        if !matches!(rows, crate::families::row_kernel::RowSet::All) {
            return None;
        }
        Some(self.second_directional_derivative_all_axes_build_once(d_beta_u))
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

        let jf_marginal = axis(&self.family.marginal_design, f_marginal);
        let mut axis0 = axis(&self.family.design_entry, f_time);
        axis0 += &jf_marginal;
        let mut axis1 = axis(&self.family.design_exit, f_time);
        axis1 += &jf_marginal;
        let axis2 = axis(&self.family.design_derivative_exit, f_time);
        let axis3 = axis(&self.family.logslope_design, f_logslope);

        crate::families::row_kernel::row_kernel_pack_jf_axes::<4>(
            n_out,
            rank,
            [(0, axis0), (1, axis1), (2, axis2), (3, axis3)],
        )
    }
}

impl SurvivalMarginalSlopeRowKernel {
    /// Build every row's full fourth-order primary tower ONCE.
    ///
    /// `evaluate_program(self, row)` is a deterministic pure function of the
    /// row's primaries, so the cached tower's `t3`/`t4` channels are bit-for-bit
    /// what a fresh per-axis `row_third_contracted` / `row_fourth_contracted`
    /// rebuild would produce — the build-once batched overrides below contract
    /// against these cached towers without changing any downstream arithmetic.
    fn build_row_towers(&self) -> Result<Vec<crate::families::jet_tower::Tower4<4>>, String> {
        let n = <Self as RowKernel<4>>::n_rows(self);
        (0..n)
            .into_par_iter()
            .map(|row| crate::families::jet_tower::evaluate_program::<4, Self>(self, row))
            .collect()
    }

    /// Deterministic `ARROW_ROW_CHUNK`-chunked reduction matching
    /// `par_try_reduce_fold(RowSet::All)`: rows fold in index order inside each
    /// fixed 256-row chunk, chunks reduce in chunk-index order on the caller
    /// thread. `per_row(row, &mut acc)` accumulates one row's pullback into the
    /// `p×p` accumulator exactly as the generic per-axis fold does.
    fn chunked_pullback_reduce<F>(&self, p: usize, per_row: F) -> Result<Array2<f64>, String>
    where
        F: Fn(usize, &mut Array2<f64>) -> Result<(), String> + Sync,
    {
        let n = <Self as RowKernel<4>>::n_rows(self);
        let chunk = crate::outer_subsample::ARROW_ROW_CHUNK;
        let n_chunks = crate::outer_subsample::arrow_row_chunk_count(n);
        let chunk_accumulators: Vec<Result<Array2<f64>, String>> = (0..n_chunks)
            .into_par_iter()
            .map(|chunk_idx| {
                let start = chunk_idx * chunk;
                let end = (start + chunk).min(n);
                let mut acc = Array2::<f64>::zeros((p, p));
                for row in start..end {
                    per_row(row, &mut acc)?;
                }
                Ok(acc)
            })
            .collect();
        let mut total = Array2::<f64>::zeros((p, p));
        for acc in chunk_accumulators {
            total += &acc?;
        }
        Ok(total)
    }

    /// gam#979 build-once all-axes FIRST directional derivative — see the trait
    /// override docstring. Builds the per-row `t3` towers once, then for each
    /// canonical axis runs the identical chunked pullback reduction the generic
    /// per-axis sweep runs, reusing the cached tower instead of rebuilding it.
    fn directional_derivative_all_axes_build_once(
        &self,
        p: usize,
    ) -> Result<Vec<Array2<f64>>, String> {
        let towers = self.build_row_towers()?;
        (0..p)
            .into_par_iter()
            .map(|a| {
                let mut axis = vec![0.0_f64; p];
                axis[a] = 1.0;
                crate::linalg::faer_ndarray::with_nested_parallel(|| {
                    self.chunked_pullback_reduce(p, |row, acc| {
                        let dir = self.jacobian_action(row, &axis);
                        let third = towers[row].third_contracted(&dir);
                        self.add_pullback_hessian(row, &third, acc);
                        Ok(())
                    })
                })
            })
            .collect()
    }

    /// gam#979 build-once all-axes SECOND directional derivative — see the trait
    /// override docstring. Builds the per-row `t4` towers and the fixed-direction
    /// projection once, then closes every axis from that single build in the
    /// generic per-axis sweep's reduction order.
    fn second_directional_derivative_all_axes_build_once(
        &self,
        d_beta_u: &[f64],
    ) -> Result<Vec<Array2<f64>>, String> {
        let p = self.n_coefficients();
        let towers = self.build_row_towers()?;
        (0..p)
            .into_par_iter()
            .map(|a| {
                let mut axis = vec![0.0_f64; p];
                axis[a] = 1.0;
                crate::linalg::faer_ndarray::with_nested_parallel(|| {
                    self.chunked_pullback_reduce(p, |row, acc| {
                        let dir_u = self.jacobian_action(row, d_beta_u);
                        let dir_v = self.jacobian_action(row, &axis);
                        let fourth = towers[row].fourth_contracted(&dir_u, &dir_v);
                        self.add_pullback_hessian(row, &fourth, acc);
                        Ok(())
                    })
                })
            })
            .collect()
    }
}
