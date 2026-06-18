//! Flex primary third/fourth contracted exact tensors.
//!
//! Builds the third- and fourth-order directional contractions of the
//! primary-space NLL Hessian on top of the exact timepoint evaluations, and
//! routes the general (flex vs rigid) entry points to either the exact flex
//! path or the rigid fallback.

use super::*;

impl SurvivalMarginalSlopeFamily {
    /// Exact third-order directional contraction for the flexible survival
    /// path.  Returns D_dir H[u,v] where H is the primary-space NLL Hessian.
    pub(crate) fn row_flex_primary_third_contracted_exact(
        &self,
        row: usize,
        block_states: &[ParameterBlockState],
        dir: &Array1<f64>,
    ) -> Result<Array2<f64>, String> {
        self.ensure_scalar_flex_exact_score_geometry("row_flex_primary_third_contracted_exact")?;
        let primary = flex_primary_slices(self);
        let p = primary.total;
        if dir.len() != p {
            return Err(SurvivalMarginalSlopeError::IncompatibleDimensions {
                reason: format!(
                    "survival third contracted: dir length {} != primary dimension {p}",
                    dir.len()
                ),
            }
            .into());
        }
        if dir.iter().all(|v| v.abs() == 0.0) {
            return Ok(Array2::<f64>::zeros((p, p)));
        }

        let q_geom = self.row_dynamic_q_geometry(row, block_states)?;
        let q0 = q_geom.q0;
        let q1 = q_geom.q1;
        let qd1 = q_geom.qd1;
        let g = block_states[2].eta[row];
        let beta_h = self.flex_score_beta(block_states)?;
        let beta_w = self.flex_link_beta(block_states)?;
        let o_infl = self.influence_index_offset(row, block_states)?;

        if survival_derivative_guard_violated(qd1, self.derivative_guard) {
            return Err(SurvivalMarginalSlopeError::MonotonicityViolation {
                reason: format!(
                    "survival third contracted monotonicity violated at row {row}: qd1={qd1:.3e}"
                ),
            }
            .into());
        }

        let (a0, d0) = self.solve_row_survival_intercept_with_slot(
            q0,
            g,
            beta_h,
            beta_w,
            Some((row, SurvivalInterceptSlotKind::Entry)),
        )?;
        let (a1, d1) = self.solve_row_survival_intercept_with_slot(
            q1,
            g,
            beta_h,
            beta_w,
            Some((row, SurvivalInterceptSlotKind::Exit)),
        )?;

        let entry_cached = self.build_cached_partition(&primary, a0, g, beta_h, beta_w)?;
        let exit_cached = self.build_cached_partition(&primary, a1, g, beta_h, beta_w)?;

        let entry = self.compute_survival_timepoint_exact_from_cached(
            row,
            &primary,
            q0,
            primary.q0,
            a0,
            g,
            d0,
            beta_h,
            beta_w,
            o_infl,
            false,
            &entry_cached,
        )?;
        let exit = self.compute_survival_timepoint_exact_from_cached(
            row,
            &primary,
            q1,
            primary.q1,
            a1,
            g,
            d1,
            beta_h,
            beta_w,
            o_infl,
            true,
            &exit_cached,
        )?;

        if !exit.chi.is_finite() || exit.chi <= 0.0 {
            return Err(SurvivalMarginalSlopeError::NumericalFailure {
                reason: format!(
                    "survival third contracted row {row}: non-positive chi1={:.3e}",
                    exit.chi,
                ),
            }
            .into());
        }

        let entry_ext = self.compute_survival_timepoint_directional_exact_from_cached(
            row,
            &primary,
            q0,
            primary.q0,
            a0,
            g,
            beta_h,
            beta_w,
            &entry_cached,
            dir,
            false,
        )?;
        let exit_ext = self.compute_survival_timepoint_directional_exact_from_cached(
            row,
            &primary,
            q1,
            primary.q1,
            a1,
            g,
            beta_h,
            beta_w,
            &exit_cached,
            dir,
            true,
        )?;

        // Delegate the per-(u, v) assembly to the Block 10 GPU-substrate
        // pure assembler in `crate::families::survival::marginal_slope::gpu`.  This is the
        // single source of truth for the third-contraction inner loop —
        // shared with the GPU dispatch path so CPU/GPU cannot drift.
        let entry_b = block10_pack_base(&entry);
        let exit_b = block10_pack_base(&exit);
        let entry_d = block10_pack_dir(&entry_ext);
        let exit_d = block10_pack_dir(&exit_ext);
        let dir_vec: Vec<f64> = dir.to_vec();
        let inputs =
            crate::families::survival::marginal_slope::gpu::SurvivalFlexBlock10ThirdInputs {
                p,
                qd1_index: primary.qd1,
                qd1,
                w: self.weights[row],
                d: self.event[row],
                dir: &dir_vec,
                entry_base: &entry_b,
                exit_base: &exit_b,
                entry_ext: &entry_d,
                exit_ext: &exit_d,
            };
        let flat =
            crate::families::survival::marginal_slope::gpu::cpu_oracle_third_contraction(&inputs)?;
        Ok(Array2::<f64>::from_shape_vec((p, p), flat).map_err(|e| e.to_string())?)
    }

    /// Fourth-order directional contraction for the flexible survival path.
    ///
    /// The mixed second-directional timepoint transport is carried exactly
    /// through the implicit intercept solve, the observed-point eta/chi jets,
    /// and the cellwise density-normalization integrand.
    pub(crate) fn row_flex_primary_fourth_contracted_exact(
        &self,
        row: usize,
        block_states: &[ParameterBlockState],
        dir_u: &Array1<f64>,
        dir_v: &Array1<f64>,
    ) -> Result<Array2<f64>, String> {
        self.ensure_scalar_flex_exact_score_geometry("row_flex_primary_fourth_contracted_exact")?;
        let primary = flex_primary_slices(self);
        let p = primary.total;
        if dir_u.len() != p || dir_v.len() != p {
            return Err(SurvivalMarginalSlopeError::IncompatibleDimensions {
                reason: format!(
                    "survival fourth contracted: dir lengths ({},{}) != {p}",
                    dir_u.len(),
                    dir_v.len(),
                ),
            }
            .into());
        }
        if dir_u.iter().all(|v| v.abs() == 0.0) || dir_v.iter().all(|v| v.abs() == 0.0) {
            return Ok(Array2::<f64>::zeros((p, p)));
        }

        let q_geom = self.row_dynamic_q_geometry(row, block_states)?;
        let q0 = q_geom.q0;
        let q1 = q_geom.q1;
        let qd1 = q_geom.qd1;
        let g = block_states[2].eta[row];
        let beta_h = self.flex_score_beta(block_states)?;
        let beta_w = self.flex_link_beta(block_states)?;
        let o_infl = self.influence_index_offset(row, block_states)?;

        if survival_derivative_guard_violated(qd1, self.derivative_guard) {
            return Err(SurvivalMarginalSlopeError::MonotonicityViolation {
                reason: format!(
                    "survival fourth contracted monotonicity violated at row {row}: qd1={qd1:.3e}"
                ),
            }
            .into());
        }

        let (a0, d0) = self.solve_row_survival_intercept_with_slot(
            q0,
            g,
            beta_h,
            beta_w,
            Some((row, SurvivalInterceptSlotKind::Entry)),
        )?;
        let (a1, d1) = self.solve_row_survival_intercept_with_slot(
            q1,
            g,
            beta_h,
            beta_w,
            Some((row, SurvivalInterceptSlotKind::Exit)),
        )?;

        let entry_cached = self.build_cached_partition(&primary, a0, g, beta_h, beta_w)?;
        let exit_cached = self.build_cached_partition(&primary, a1, g, beta_h, beta_w)?;

        let entry_base = self.compute_survival_timepoint_exact_from_cached(
            row,
            &primary,
            q0,
            primary.q0,
            a0,
            g,
            d0,
            beta_h,
            beta_w,
            o_infl,
            false,
            &entry_cached,
        )?;
        let exit_base = self.compute_survival_timepoint_exact_from_cached(
            row,
            &primary,
            q1,
            primary.q1,
            a1,
            g,
            d1,
            beta_h,
            beta_w,
            o_infl,
            true,
            &exit_cached,
        )?;

        if !exit_base.chi.is_finite() || exit_base.chi <= 0.0 {
            return Err(SurvivalMarginalSlopeError::NumericalFailure {
                reason: format!(
                    "survival fourth contracted row {row}: non-positive chi1={:.3e}",
                    exit_base.chi,
                ),
            }
            .into());
        }

        let entry_ext_u = self.compute_survival_timepoint_directional_exact_from_cached(
            row,
            &primary,
            q0,
            primary.q0,
            a0,
            g,
            beta_h,
            beta_w,
            &entry_cached,
            dir_u,
            false,
        )?;
        let entry_ext_v = self.compute_survival_timepoint_directional_exact_from_cached(
            row,
            &primary,
            q0,
            primary.q0,
            a0,
            g,
            beta_h,
            beta_w,
            &entry_cached,
            dir_v,
            false,
        )?;
        let exit_ext_u = self.compute_survival_timepoint_directional_exact_from_cached(
            row,
            &primary,
            q1,
            primary.q1,
            a1,
            g,
            beta_h,
            beta_w,
            &exit_cached,
            dir_u,
            true,
        )?;
        let exit_ext_v = self.compute_survival_timepoint_directional_exact_from_cached(
            row,
            &primary,
            q1,
            primary.q1,
            a1,
            g,
            beta_h,
            beta_w,
            &exit_cached,
            dir_v,
            true,
        )?;

        // Bidirectional extensions D_{d1} D_{d2} (η_uv, χ_uv, D_uv) via exact
        // IFT second-order recursion through the cell kernel.
        let entry_bi = self.compute_survival_timepoint_bidirectional_exact_from_cached(
            row,
            &primary,
            q0,
            primary.q0,
            a0,
            g,
            beta_h,
            beta_w,
            &entry_cached,
            dir_u,
            dir_v,
        )?;
        let exit_bi = self.compute_survival_timepoint_bidirectional_exact_from_cached(
            row,
            &primary,
            q1,
            primary.q1,
            a1,
            g,
            beta_h,
            beta_w,
            &exit_cached,
            dir_u,
            dir_v,
        )?;

        // Delegate the per-(u, v) assembly + averaged-ordered
        // symmetrization to the Block 10 GPU-substrate pure assembler.
        // Single source of truth shared with the GPU dispatch path.
        let entry_b = block10_pack_base(&entry_base);
        let exit_b = block10_pack_base(&exit_base);
        let entry_d1 = block10_pack_dir(&entry_ext_u);
        let entry_d2 = block10_pack_dir(&entry_ext_v);
        let exit_d1 = block10_pack_dir(&exit_ext_u);
        let exit_d2 = block10_pack_dir(&exit_ext_v);
        let entry_bi_p = block10_pack_bi(&entry_bi);
        let exit_bi_p = block10_pack_bi(&exit_bi);
        let dir_u_vec: Vec<f64> = dir_u.to_vec();
        let dir_v_vec: Vec<f64> = dir_v.to_vec();
        let inputs =
            crate::families::survival::marginal_slope::gpu::SurvivalFlexBlock10FourthInputs {
                p,
                qd1_index: primary.qd1,
                qd1,
                w: self.weights[row],
                d: self.event[row],
                dir_u: &dir_u_vec,
                dir_v: &dir_v_vec,
                entry_base: &entry_b,
                exit_base: &exit_b,
                entry_ext_u: &entry_d1,
                entry_ext_v: &entry_d2,
                exit_ext_u: &exit_d1,
                exit_ext_v: &exit_d2,
                entry_bi: &entry_bi_p,
                exit_bi: &exit_bi_p,
            };
        let flat =
            crate::families::survival::marginal_slope::gpu::cpu_oracle_fourth_contraction(&inputs)
                .map_err(|e| format!("block10 fourth contraction: {e}"))?;
        let mut out = Array2::<f64>::zeros((p, p));
        for u in 0..p {
            for v in 0..p {
                out[[u, v]] = flat[u * p + v];
            }
        }
        Ok(out)
    }

    pub(crate) fn row_primary_third_contracted_general(
        &self,
        row: usize,
        block_states: &[ParameterBlockState],
        dir: &Array1<f64>,
    ) -> Result<Array2<f64>, String> {
        if self.effective_flex_active(block_states)? {
            self.row_flex_primary_third_contracted_exact(row, block_states, dir)
        } else {
            self.row_primary_third_contracted(row, block_states, dir.view())
        }
    }

    pub(crate) fn row_primary_fourth_contracted(
        &self,
        row: usize,
        block_states: &[ParameterBlockState],
        dir_u: ArrayView1<'_, f64>,
        dir_v: ArrayView1<'_, f64>,
    ) -> Result<Array2<f64>, String> {
        // Batched path delegating to the shared k=6 jet helper.
        let r = self.row_primary_fourth_contracted_batched(row, block_states, dir_u, dir_v)?;
        let mut out = Array2::<f64>::zeros((N_PRIMARY, N_PRIMARY));
        for a in 0..N_PRIMARY {
            for b in 0..N_PRIMARY {
                out[[a, b]] = r[a][b];
            }
        }
        Ok(out)
    }

    pub(crate) fn row_primary_fourth_contracted_general(
        &self,
        row: usize,
        block_states: &[ParameterBlockState],
        dir_u: &Array1<f64>,
        dir_v: &Array1<f64>,
    ) -> Result<Array2<f64>, String> {
        if self.effective_flex_active(block_states)? {
            self.row_flex_primary_fourth_contracted_exact(row, block_states, dir_u, dir_v)
        } else {
            self.row_primary_fourth_contracted(row, block_states, dir_u.view(), dir_v.view())
        }
    }

    // ── Pullback through design matrices ──────────────────────────────
}
