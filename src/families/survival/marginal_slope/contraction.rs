//! Rigid primary-tower contractions and cache helpers used to assemble
//! higher-order curvature.

use super::*;

struct SurvivalMarginalSlopeRigidRowProgram<'a> {
    family: &'a SurvivalMarginalSlopeFamily,
    block_states: &'a [ParameterBlockState],
}

impl crate::families::jet_tower::RowNllProgram<4> for SurvivalMarginalSlopeRigidRowProgram<'_> {
    fn n_rows(&self) -> usize {
        self.family.n
    }

    fn primaries(&self, row: usize) -> Result<[f64; 4], String> {
        rigid_row_kernel_primaries(self.family, self.block_states, row)
    }

    fn row_nll(
        &self,
        row: usize,
        p: &[crate::families::jet_tower::Tower4<4>; 4],
    ) -> Result<crate::families::jet_tower::Tower4<4>, String> {
        rigid_row_kernel_nll_tower(
            self.family,
            self.block_states,
            row,
            p,
            "survival marginal-slope rigid row helper tower",
        )
    }
}

impl SurvivalMarginalSlopeFamily {
    /// Build the row's third-order contracted tensor
    /// `T[a][b] = d_ea d_eb d_dir NLL_i` from the single rigid
    /// `RowNllProgram` tower.
    pub(crate) fn row_primary_third_contracted_tower(
        &self,
        row: usize,
        block_states: &[ParameterBlockState],
        dir: ArrayView1<'_, f64>,
    ) -> Result<[[f64; 4]; 4], String> {
        if dir.len() != N_PRIMARY {
            return Err(SurvivalMarginalSlopeError::IncompatibleDimensions {
                reason: format!(
                    "survival rigid third contracted: dir length {} != primary dimension {N_PRIMARY}",
                    dir.len()
                ),
            }
            .into());
        }
        let program = SurvivalMarginalSlopeRigidRowProgram {
            family: self,
            block_states,
        };
        let mut dir_arr = [0.0_f64; N_PRIMARY];
        dir_arr.copy_from_slice(dir.as_slice().ok_or_else(|| {
            "survival rigid third contracted: non-contiguous direction".to_string()
        })?);
        crate::families::jet_tower::derived_third_contracted(&program, row, &dir_arr)
    }

    /// Build the row's fourth-order contracted tensor
    /// `T[a][b] = d_ea d_eb d_dir_u d_dir_v NLL_i` from the single rigid
    /// `RowNllProgram` tower.
    pub(crate) fn row_primary_fourth_contracted_tower(
        &self,
        row: usize,
        block_states: &[ParameterBlockState],
        dir_u: ArrayView1<'_, f64>,
        dir_v: ArrayView1<'_, f64>,
    ) -> Result<[[f64; 4]; 4], String> {
        if dir_u.len() != N_PRIMARY || dir_v.len() != N_PRIMARY {
            return Err(SurvivalMarginalSlopeError::IncompatibleDimensions {
                reason: format!(
                    "survival rigid fourth contracted: dir lengths ({},{}) != primary dimension {N_PRIMARY}",
                    dir_u.len(),
                    dir_v.len()
                ),
            }
            .into());
        }
        let program = SurvivalMarginalSlopeRigidRowProgram {
            family: self,
            block_states,
        };
        let mut u_arr = [0.0_f64; N_PRIMARY];
        u_arr.copy_from_slice(dir_u.as_slice().ok_or_else(|| {
            "survival rigid fourth contracted: non-contiguous u direction".to_string()
        })?);
        let mut v_arr = [0.0_f64; N_PRIMARY];
        v_arr.copy_from_slice(dir_v.as_slice().ok_or_else(|| {
            "survival rigid fourth contracted: non-contiguous v direction".to_string()
        })?);
        crate::families::jet_tower::derived_fourth_contracted(&program, row, &u_arr, &v_arr)
    }

    /// Compute per-row primary gradient and Hessian using the rigid
    /// `RowNllProgram` tower.  The hot inner computation uses stack arrays only;
    /// conversion to Array1/Array2 happens once at the boundary for API
    /// compatibility with outer-derivative paths.
    pub(crate) fn compute_row_primary_gradient_hessian_uncached(
        &self,
        row: usize,
        block_states: &[ParameterBlockState],
    ) -> Result<(f64, Array1<f64>, Array2<f64>), String> {
        let program = SurvivalMarginalSlopeRigidRowProgram {
            family: self,
            block_states,
        };
        let (nll, grad_arr, hess_arr) =
            crate::families::jet_tower::derived_row_kernel(&program, row)?;
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

    pub(crate) fn build_eval_cache(
        &self,
        block_states: &[ParameterBlockState],
    ) -> Result<EvalCache, String> {
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
        let r = self.row_primary_third_contracted_tower(row, block_states, dir)?;
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

    /// Third cell-coefficient cross `∂³c/∂u∂v∂(dir)` contracted with a
    /// direction `dir`, accumulated into `out` (scaled by `sign`).
    ///
    /// The denested cell coefficient `c` is an exact cubic in the latent `z`
    /// whose coefficients depend on the intercept `a`, the rigid slope `b≡g`,
    /// and the score/link basis parameters. Differentiating w.r.t. three
    /// *parameter* axes leaves a nonzero coefficient only when at least two of
    /// the three axes are the slope `g` (= the `b` argument): the basis
    /// channels enter `c` linearly, so any triple with two or more distinct
    /// basis indices vanishes. Hence `∂³c/∂u∂v∂c'` is carried by the `bbu`
    /// family (`∂²/∂b² ∂param`):
    ///   - `u==g, v==g`  →  `Σ_{c'} coeff_bbu[c']·dir[c']`   (incl. `∂³c/∂g³`),
    ///   - `u==g, v≠g`   →  `coeff_bbu[v]·dir[g]`,
    ///   - `v==g, u≠g`   →  `coeff_bbu[u]·dir[g]`,
    ///   - otherwise     →  `0`.
    ///
    /// This is the `f_uv` analogue of `cell_pair_third_coeff_a`'s role for the
    /// `f_au` jets: omitting it drops the `∂²/∂g²` curvature of a basis
    /// coefficient from every `D_g f_uv[g, ·]` cell integral (gam#1195).
    pub(crate) fn add_cell_pair_third_coeff_dir(
        &self,
        primary: &FlexPrimarySlices,
        coeff_bbu: &[[f64; 4]],
        u: usize,
        v: usize,
        dir: &Array1<f64>,
        sign: f64,
        out: &mut [f64; 4],
    ) {
        let g = primary.g;
        if u == g && v == g {
            for (c, &dir_c) in dir.iter().enumerate() {
                if dir_c == 0.0 {
                    continue;
                }
                for k in 0..4 {
                    out[k] += sign * coeff_bbu[c][k] * dir_c;
                }
            }
        } else if u == g {
            let dir_g = dir[g];
            if dir_g != 0.0 {
                for k in 0..4 {
                    out[k] += sign * coeff_bbu[v][k] * dir_g;
                }
            }
        } else if v == g {
            let dir_g = dir[g];
            if dir_g != 0.0 {
                for k in 0..4 {
                    out[k] += sign * coeff_bbu[u][k] * dir_g;
                }
            }
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
