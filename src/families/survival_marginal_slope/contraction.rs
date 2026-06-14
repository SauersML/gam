//! Directional jet contraction of the primary kernel: the batched
//! directional NLL jet and the third/fourth contracted primary tensors
//! used to assemble higher-order curvature.

use super::*;

impl SurvivalMarginalSlopeFamily {
    /// Batched variant of the per-row NLL multi-directional jet, returning the full
    /// `MultiDirJet` for the per-row NLL composed against an arbitrary
    /// ordered list of `dirs` (one direction per jet bit).
    ///
    /// Callers reading higher-order contracted tensors (third/fourth)
    /// previously looped the 10 upper-triangular (a, b) pairs and called
    /// `row_neglog_directional_refs` ten times, each rebuilding the same
    /// q-geometry, the same primary jets, and rerunning the same composed
    /// unary derivatives. By packing the basis directions e_0..e_3 alongside
    /// the user-supplied `dir` (and optional `dir_v`) into one larger jet, the
    /// per-row q-geometry + jet-construction work is shared across all 10
    /// upper-triangular cells. Off-diagonal entries are then a single mask
    /// read; diagonal entries fall back to a small per-axis jet that still
    /// reuses no work between cells.
    ///
    /// `dirs.len()` must be in `1..=8` (MAX_DIRS in `jet_partitions`).
    pub(crate) fn row_neglog_directional_jet_batched(
        &self,
        row: usize,
        block_states: &[ParameterBlockState],
        dirs: &[ArrayView1<'_, f64>],
    ) -> Result<MultiDirJet, String> {
        crate::families::jet_partitions::ROW_NEGLOG_CALLS
            .fetch_add(1, std::sync::atomic::Ordering::Relaxed);
        let k = dirs.len();
        if k == 0 || k > 8 {
            return Err(SurvivalMarginalSlopeError::InvalidInput {
                reason: format!(
                    "survival marginal-slope row directional jet expects 1..=8 directions, got {k}"
                ),
            }
            .into());
        }
        let wi = self.weights[row];
        let di = self.event[row];
        let (z_sum, covariance_ones) = self.exact_shared_score_summary(
            row,
            block_states,
            "row_neglog_directional_jet_batched",
        )?;

        // Primary scalar jets: q0, q1, qd1, g. Fixed-capacity stack buffers
        // (k ≤ 8) avoid heap allocation in the per-row hot loop.
        let mut q0_first_buf = [0.0f64; 8];
        let mut q1_first_buf = [0.0f64; 8];
        let mut qd1_first_buf = [0.0f64; 8];
        let mut g_first_buf = [0.0f64; 8];
        for (i, dir) in dirs.iter().enumerate() {
            q0_first_buf[i] = dir[0];
            q1_first_buf[i] = dir[1];
            qd1_first_buf[i] = dir[2];
            g_first_buf[i] = dir[3];
        }
        let q0_first: &[f64] = &q0_first_buf[..k];
        let q1_first: &[f64] = &q1_first_buf[..k];
        let qd1_first: &[f64] = &qd1_first_buf[..k];
        let g_first: &[f64] = &g_first_buf[..k];

        // Reuse the realized q-geometry (timewiggle / frailty manifold).
        let q_geom = self.row_dynamic_q_values(row, block_states)?;
        let q0_val = q_geom.q0;
        let q1_val = q_geom.q1;
        let qd1_val = q_geom.qd1;
        let g_val = block_states[2].eta[row];

        let q0_jet = MultiDirJet::linear(k, q0_val, q0_first);
        let q1_jet = MultiDirJet::linear(k, q1_val, q1_first);
        let qd1_jet = MultiDirJet::linear(k, qd1_val, qd1_first);
        let g_jet = MultiDirJet::linear(k, g_val, g_first);

        // observed_g = frailty_scale * g
        let observed_g_jet = g_jet.scale(self.probit_frailty_scale());
        // c = sqrt(1 + observed_g^2 * 1' Sigma 1) for a shared K-vector slope.
        let one_plus_b2 = MultiDirJet::constant(k, 1.0)
            .add(&observed_g_jet.mul(&observed_g_jet).scale(covariance_ones));
        let c_jet = one_plus_b2.compose_unary(unary_derivatives_sqrt(one_plus_b2.coeff(0)));

        let a0_jet = q0_jet.mul(&c_jet);
        let a1_jet = q1_jet.mul(&c_jet);
        let ad1_jet = qd1_jet.mul(&c_jet);

        let z_jet = MultiDirJet::constant(k, z_sum);
        let eta0_jet = a0_jet.add(&observed_g_jet.mul(&z_jet));
        let eta1_jet = a1_jet.add(&observed_g_jet.mul(&z_jet));

        let neg_eta0 = eta0_jet.scale(-1.0);
        let entry_term = neg_eta0
            .compose_unary(unary_derivatives_neglog_phi(neg_eta0.coeff(0), wi))
            .scale(-1.0);

        let neg_eta1 = eta1_jet.scale(-1.0);
        let exit_term = neg_eta1.compose_unary(unary_derivatives_neglog_phi(
            neg_eta1.coeff(0),
            wi * (1.0 - di),
        ));

        let event_density_term = if di > 0.0 {
            eta1_jet
                .compose_unary(unary_derivatives_log_normal_pdf(eta1_jet.coeff(0)))
                .scale(-wi * di)
        } else {
            MultiDirJet::zero(k)
        };

        let qd1_lower = self.time_derivative_lower_bound();
        let qd1_val_chk = qd1_jet.coeff(0);
        let ad1_val = ad1_jet.coeff(0);
        if survival_derivative_guard_violated(qd1_val_chk, qd1_lower) {
            return Err(SurvivalMarginalSlopeError::MonotonicityViolation {
                reason: format!(
                    "survival marginal-slope monotonicity violated at row {row}: raw time derivative={qd1_val_chk:.3e} must be at least derivative_guard={qd1_lower:.3e}; transformed time derivative={ad1_val:.3e}"
                ),
            }
            .into());
        }
        let time_deriv_term = if di > 0.0 {
            ad1_jet
                .compose_unary(unary_derivatives_log(ad1_val))
                .scale(-wi * di)
        } else {
            MultiDirJet::zero(k)
        };

        let total = exit_term
            .add(&entry_term)
            .add(&event_density_term)
            .add(&time_deriv_term);

        Ok(total)
    }

    /// Build the row's third-order contracted tensor `T[a][b] = ∂_{e_a} ∂_{e_b} ∂_{dir}` of NLL_i
    /// against the four primary axes (a, b ∈ {0..N_PRIMARY}). Six off-diagonal
    /// entries are read from a single batched k=5 jet over [e_0..e_3, dir];
    /// four diagonal entries (a == b) need order-3 in a single basis axis and
    /// fall back to a k=3 jet [e_a, e_a, dir]. Result is symmetric.
    pub(crate) fn row_primary_third_contracted_batched(
        &self,
        row: usize,
        block_states: &[ParameterBlockState],
        dir: ArrayView1<'_, f64>,
    ) -> Result<[[f64; 4]; 4], String> {
        let e0 = unit_primary_direction_ref(0).view();
        let e1 = unit_primary_direction_ref(1).view();
        let e2 = unit_primary_direction_ref(2).view();
        let e3 = unit_primary_direction_ref(3).view();
        // One k=5 jet covers all 6 off-diagonal (a < b) entries via masks
        // `(1<<a) | (1<<b) | (1<<4)`.
        let batched =
            self.row_neglog_directional_jet_batched(row, block_states, &[e0, e1, e2, e3, dir])?;
        let dir_bit = 1usize << 4;
        let mut r = [[0.0_f64; 4]; 4];
        for a in 0..N_PRIMARY {
            for b in (a + 1)..N_PRIMARY {
                let mask = (1usize << a) | (1usize << b) | dir_bit;
                let value = batched.coeff(mask);
                r[a][b] = value;
                r[b][a] = value;
            }
        }
        // Diagonal entries: ∂²_{e_a} ∂_{dir} require two copies of e_a in
        // distinct bit positions. A k=3 jet [e_a, e_a, dir] read at full
        // mask (7) supplies that.
        for a in 0..N_PRIMARY {
            let ea = unit_primary_direction_ref(a).view();
            let diag_jet =
                self.row_neglog_directional_jet_batched(row, block_states, &[ea, ea, dir])?;
            r[a][a] = diag_jet.coeff(diag_jet.full_mask());
        }
        Ok(r)
    }

    /// Build the row's fourth-order contracted tensor
    /// `T[a][b] = ∂_{e_a} ∂_{e_b} ∂_{dir_u} ∂_{dir_v}` of NLL_i, mirrored.
    pub(crate) fn row_primary_fourth_contracted_batched(
        &self,
        row: usize,
        block_states: &[ParameterBlockState],
        dir_u: ArrayView1<'_, f64>,
        dir_v: ArrayView1<'_, f64>,
    ) -> Result<[[f64; 4]; 4], String> {
        let e0 = unit_primary_direction_ref(0).view();
        let e1 = unit_primary_direction_ref(1).view();
        let e2 = unit_primary_direction_ref(2).view();
        let e3 = unit_primary_direction_ref(3).view();
        // One k=6 jet covers all 6 off-diagonal (a < b) entries via masks
        // `(1<<a) | (1<<b) | (1<<4) | (1<<5)`.
        let batched = self.row_neglog_directional_jet_batched(
            row,
            block_states,
            &[e0, e1, e2, e3, dir_u, dir_v],
        )?;
        let u_bit = 1usize << 4;
        let v_bit = 1usize << 5;
        let mut r = [[0.0_f64; 4]; 4];
        for a in 0..N_PRIMARY {
            for b in (a + 1)..N_PRIMARY {
                let mask = (1usize << a) | (1usize << b) | u_bit | v_bit;
                let value = batched.coeff(mask);
                r[a][b] = value;
                r[b][a] = value;
            }
        }
        for a in 0..N_PRIMARY {
            let ea = unit_primary_direction_ref(a).view();
            let diag_jet = self.row_neglog_directional_jet_batched(
                row,
                block_states,
                &[ea, ea, dir_u, dir_v],
            )?;
            r[a][a] = diag_jet.coeff(diag_jet.full_mask());
        }
        Ok(r)
    }

    /// Compute per-row primary gradient and Hessian using the closed-form
    /// scalar kernel.  The hot inner computation uses stack arrays only;
    /// conversion to Array1/Array2 happens once at the boundary for API
    /// compatibility with outer-derivative paths.
    pub(crate) fn compute_row_primary_gradient_hessian_uncached(
        &self,
        row: usize,
        block_states: &[ParameterBlockState],
    ) -> Result<(f64, Array1<f64>, Array2<f64>), String> {
        let q_geom = self.row_dynamic_q_values(row, block_states)?;
        let (nll, grad_arr, hess_arr) = self.row_primary_closed_form_rigid(
            row,
            q_geom.q0,
            q_geom.q1,
            q_geom.qd1,
            block_states,
            self.probit_frailty_scale(),
        )?;
        // Convert stack arrays to ndarray types at the boundary.
        let grad = Array1::from_vec(grad_arr.to_vec());
        let mut hess = Array2::zeros((N_PRIMARY, N_PRIMARY));
        for i in 0..N_PRIMARY {
            for j in 0..N_PRIMARY {
                hess[[i, j]] = hess_arr[i][j];
            }
        }
        Ok((nll, grad, hess))
    }

    pub(crate) fn build_eval_cache(&self, block_states: &[ParameterBlockState]) -> Result<EvalCache, String> {
        let row_bases = (0..self.n)
            .into_par_iter()
            .map(|row| {
                let (_, gradient, hessian) =
                    self.compute_row_primary_gradient_hessian_uncached(row, block_states)?;
                Ok(RowPrimaryBase { gradient, hessian })
            })
            .collect::<Result<Vec<_>, String>>()?;
        Ok(EvalCache { row_bases })
    }

    pub(crate) fn row_primary_gradient_hessian<'a>(
        &self,
        row: usize,
        cache: &'a EvalCache,
    ) -> (&'a Array1<f64>, &'a Array2<f64>) {
        let base = &cache.row_bases[row];
        (&base.gradient, &base.hessian)
    }

    pub(crate) fn offset_channel_geometry(
        &self,
        block_states: &[ParameterBlockState],
    ) -> Result<(OffsetChannelResiduals, OffsetChannelCurvatures), String> {
        let flex_active = self.effective_flex_active(block_states)?;
        let primary = flex_active.then(|| flex_primary_slices(self));
        let rows = (0..self.n)
            .into_par_iter()
            .map(
                |row| -> Result<(usize, f64, f64, f64, [[f64; 3]; 3]), String> {
                    if self.weights[row] <= 0.0 {
                        return Ok((row, 0.0, 0.0, 0.0, [[0.0; 3]; 3]));
                    }
                    let q_geom = self.row_dynamic_q_geometry(row, block_states)?;
                    let (gradient, hessian) = if let Some(primary) = primary.as_ref() {
                        let (_, gradient, hessian) = self
                            .compute_row_flex_primary_gradient_hessian_exact(
                                row,
                                block_states,
                                &q_geom,
                                primary,
                            )?;
                        (gradient, hessian)
                    } else {
                        let (_, gradient, hessian) =
                            self.compute_row_primary_gradient_hessian_uncached(row, block_states)?;
                        (gradient, hessian)
                    };
                    let channel = [0usize, 1usize, 2usize];
                    let mut curvature = [[0.0; 3]; 3];
                    for a in 0..3 {
                        for b in 0..3 {
                            curvature[a][b] = hessian[[channel[a], channel[b]]];
                        }
                    }
                    Ok((row, gradient[1], gradient[0], gradient[2], curvature))
                },
            )
            .collect::<Result<Vec<_>, String>>()?;
        let mut exit = Array1::<f64>::zeros(self.n);
        let mut entry = Array1::<f64>::zeros(self.n);
        let mut derivative = Array1::<f64>::zeros(self.n);
        let mut curvatures = vec![[[0.0; 3]; 3]; self.n];
        for (row, r_exit, r_entry, r_derivative, curvature) in rows {
            exit[row] = r_exit;
            entry[row] = r_entry;
            derivative[row] = r_derivative;
            curvatures[row] = curvature;
        }
        Ok((
            OffsetChannelResiduals {
                exit,
                entry,
                derivative,
                // Marginal-slope has no interval upper-bound channel.
                right: Array1::<f64>::zeros(self.n),
            },
            OffsetChannelCurvatures { rows: curvatures },
        ))
    }

}

impl SurvivalMarginalSlopeFamily {
    pub(crate) fn row_primary_third_contracted(
        &self,
        row: usize,
        block_states: &[ParameterBlockState],
        dir: ArrayView1<'_, f64>,
    ) -> Result<Array2<f64>, String> {
        // Batched path delegating to the shared k=5 jet helper. The stack
        // tensor is then copied once into Array2 at the API boundary.
        let r = self.row_primary_third_contracted_batched(row, block_states, dir)?;
        let mut out = Array2::<f64>::zeros((N_PRIMARY, N_PRIMARY));
        for a in 0..N_PRIMARY {
            for b in 0..N_PRIMARY {
                out[[a, b]] = r[a][b];
            }
        }
        Ok(out)
    }

    /// Second-order cross-coefficient for parameter pair (u, v) from the
    /// b-family.  In survival, the only b-coupling is through g (the slope
    /// parameter), so this is nonzero only when u==g or v==g.
    pub(crate) fn cell_pair_second_coeff(
        &self,
        primary: &FlexPrimarySlices,
        coeff_bu: &[[f64; 4]],
        u: usize,
        v: usize,
    ) -> [f64; 4] {
        if u == primary.g {
            coeff_bu[v]
        } else if v == primary.g {
            coeff_bu[u]
        } else {
            [0.0; 4]
        }
    }

    /// Third-order a-cross-coefficient for parameter pair (u, v) from the
    /// ab-family. Nonzero only when u==g or v==g.
    pub(crate) fn cell_pair_third_coeff_a(
        &self,
        primary: &FlexPrimarySlices,
        coeff_abu: &[[f64; 4]],
        u: usize,
        v: usize,
    ) -> [f64; 4] {
        if u == primary.g {
            coeff_abu[v]
        } else if v == primary.g {
            coeff_abu[u]
        } else {
            [0.0; 4]
        }
    }

    /// Directional derivative of the fixed eta second partial r_uv w.r.t.
    /// a contraction direction.  Only (g,g), (g,h), (g,w) entries are nonzero.
    pub(crate) fn observed_fixed_eta_second_partial_dir(
        &self,
        primary: &FlexPrimarySlices,
        obs: &ObservedDenestedCellPartials,
        u: usize,
        v: usize,
        z_obs: f64,
        u_obs: f64,
        a: f64,
        b: f64,
        a_dir: f64,
        dir: &Array1<f64>,
        beta_w: Option<&Array1<f64>>,
    ) -> Result<f64, String> {
        let scale = self.probit_frailty_scale();
        if u == primary.g && v == primary.g {
            let dc_dbbb_val = {
                let zero_span = exact_kernel::LocalSpanCubic {
                    left: 0.0,
                    right: 1.0,
                    c0: 0.0,
                    c1: 0.0,
                    c2: 0.0,
                    c3: 0.0,
                };
                let link_span_obs = if let (Some(rt), Some(bw)) = (self.link_dev.as_ref(), beta_w) {
                    rt.local_cubic_at(bw, u_obs)?
                } else {
                    zero_span
                };
                let (_, _, _, dc_dbbb_raw) =
                    exact_kernel::denested_cell_third_partials(link_span_obs);
                eval_coeff4_at(&scale_coeff4(dc_dbbb_raw, scale), z_obs)
            };
            return Ok(eval_coeff4_at(&obs.dc_dabb, z_obs) * a_dir + dc_dbbb_val * dir[primary.g]);
        }
        if u == primary.g || v == primary.g {
            let other = if u == primary.g { v } else { u };
            if let Some(h_range) = primary.h.as_ref()
                && other >= h_range.start
                && other < h_range.end
            {
                return Ok(0.0);
            }
            if let Some(w_range) = primary.w.as_ref()
                && other >= w_range.start
                && other < w_range.end
            {
                let local_idx = other - w_range.start;
                let runtime = self
                    .link_dev
                    .as_ref()
                    .ok_or_else(|| "missing link runtime".to_string())?;
                let basis_span = runtime.basis_cubic_at(local_idx, u_obs)?;
                let (_, dc_abw, dc_bbw) =
                    exact_kernel::link_basis_cell_second_partials(basis_span, a, b);
                return Ok(eval_coeff4_at(&scale_coeff4(dc_abw, scale), z_obs) * a_dir
                    + eval_coeff4_at(&scale_coeff4(dc_bbw, scale), z_obs) * dir[primary.g]);
            }
        }
        Ok(0.0)
    }

    /// Directional derivative of the fixed chi second partial.
    pub(crate) fn observed_fixed_chi_second_partial_dir(
        &self,
        primary: &FlexPrimarySlices,
        u: usize,
        v: usize,
        z_obs: f64,
        u_obs: f64,
        a_dir: f64,
        dir: &Array1<f64>,
    ) -> Result<f64, String> {
        let scale = self.probit_frailty_scale();
        if u == primary.g && v == primary.g {
            return Ok(0.0);
        }
        if u == primary.g || v == primary.g {
            let other = if u == primary.g { v } else { u };
            if let Some(w_range) = primary.w.as_ref()
                && other >= w_range.start
                && other < w_range.end
            {
                let local_idx = other - w_range.start;
                let runtime = self
                    .link_dev
                    .as_ref()
                    .ok_or_else(|| "missing link runtime".to_string())?;
                let basis_span = runtime.basis_cubic_at(local_idx, u_obs)?;
                let (_, dc_aabw, dc_abbw, _) =
                    exact_kernel::link_basis_cell_third_partials(basis_span);
                return Ok(eval_coeff4_at(&scale_coeff4(dc_aabw, scale), z_obs) * a_dir
                    + eval_coeff4_at(&scale_coeff4(dc_abbw, scale), z_obs) * dir[primary.g]);
            }
        }
        Ok(0.0)
    }

}

