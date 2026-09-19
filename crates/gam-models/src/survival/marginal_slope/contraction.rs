//! Rigid primary-tower contractions and cache helpers used to assemble
//! higher-order curvature.

use super::*;

use gam_math::jet_scalar::{JetScalar, OneSeed, TwoSeed};

impl SurvivalMarginalSlopeFamily {
    /// Resolve one rigid row's inputs and primaries, then evaluate its primary
    /// tower into `out` through [`Self::write_primary_tower`].
    pub(crate) fn write_row_primary_tower<
        const P: usize,
        G: SlopeRowGeometry<P>,
        S: JetScalar<P>,
    >(
        &self,
        row: usize,
        block_states: &[ParameterBlockState],
        context: &str,
        out: &mut S,
    ) -> Result<(), String> {
        let inputs = rigid_row_inputs(self, block_states, row, context)?;
        let primaries = rigid_row_kernel_primaries::<P, G>(self, block_states, row)?;
        Self::write_primary_tower::<P, G, S>(&primaries, &inputs, out)
    }

    /// Evaluate one rigid row's primary tower, at the order `S` carries, into `out`.
    ///
    /// The one place a primary tower is seeded and evaluated (build.rs refuses a
    /// tower seed anywhere else in this module's production code). The row
    /// program keeps all of [`rigid_row_nll`]'s jet intermediates in this
    /// function's own frame: 64 KiB for the four-primary frames and 276 KiB for
    /// the six-primary follow-up frame. Inlined into a Rayon closure, that frame
    /// can be inlined on into the recursive split helper and reserved again at
    /// every split level, which is how a 2 MiB default worker overflowed
    /// (gam#2967). Out of line it exists once per thread, at the leaf, and the
    /// tower leaves through `out`, so the caller holds a pointer rather than a
    /// tower-sized return slot.
    #[inline(never)]
    pub(crate) fn write_primary_tower<const P: usize, G: SlopeRowGeometry<P>, S: JetScalar<P>>(
        primaries: &[f64; P],
        inputs: &RigidRowInputs,
        out: &mut S,
    ) -> Result<(), String> {
        let vars: [S; P] = std::array::from_fn(|a| S::variable(primaries[a], a));
        *out = rigid_row_nll::<P, G, _>(&vars, inputs)?;
        Ok(())
    }

    /// Evaluate the single-source rigid row program once through order three.
    ///
    /// Three of the four rigid primaries are affine, so static sparsity retains
    /// only the ten potentially nonzero third-order channels. The resulting
    /// tower is independent of the directions later contracted into it.
    pub(crate) fn build_row_primary_third_tower<const P: usize, G: SlopeRowGeometry<P>>(
        &self,
        row: usize,
        block_states: &[ParameterBlockState],
    ) -> Result<G::Tower3, String> {
        let mut tower = G::Tower3::constant(0.0);
        self.write_row_primary_tower::<P, G, _>(
            row,
            block_states,
            "survival marginal-slope rigid row helper third",
            &mut tower,
        )?;
        Ok(tower)
    }

    /// Contract a previously evaluated sparse row tower with one direction,
    /// preserving the dense tower's exact accumulation order.
    pub(crate) fn contract_row_primary_third_tower<const P: usize, G: SlopeRowGeometry<P>>(
        tower: &G::Tower3,
        dir: &Array1<f64>,
    ) -> Result<[[f64; P]; P], String> {
        if dir.len() != P {
            return Err(SurvivalMarginalSlopeError::IncompatibleDimensions {
                reason: format!(
                    "survival rigid third contracted: dir length {} != primary dimension {P}",
                    dir.len()
                ),
            }
            .into());
        }
        let mut dir_arr = [0.0_f64; P];
        dir_arr.copy_from_slice(dir.as_slice().ok_or_else(|| {
            "survival rigid third contracted: non-contiguous direction".to_string()
        })?);
        Ok(tower3_third_contracted(tower.t3(), &dir_arr))
    }

    /// Build one rigid row's third-order directional contraction without
    /// materializing the full third tensor. Single-axis callers retain this
    /// cheaper directional scalar; only multi-axis callers build a shared tower.
    pub(crate) fn row_primary_third_contracted_tower<const P: usize, G: SlopeRowGeometry<P>>(
        &self,
        row: usize,
        block_states: &[ParameterBlockState],
        dir: ArrayView1<'_, f64>,
    ) -> Result<[[f64; P]; P], String> {
        if dir.len() != P {
            return Err(SurvivalMarginalSlopeError::IncompatibleDimensions {
                reason: format!(
                    "survival rigid third contracted: dir length {} != primary dimension {P}",
                    dir.len()
                ),
            }
            .into());
        }
        let mut dir_arr = [0.0_f64; P];
        dir_arr.copy_from_slice(dir.as_slice().ok_or_else(|| {
            "survival rigid third contracted: non-contiguous direction".to_string()
        })?);
        let inputs = rigid_row_inputs(
            self,
            block_states,
            row,
            "survival marginal-slope rigid row helper third",
        )?;
        let p = rigid_row_kernel_primaries::<P, G>(self, block_states, row)?;
        let vars: [OneSeed<P>; P] =
            std::array::from_fn(|a| OneSeed::seed_direction(p[a], a, dir_arr[a]));
        Ok(rigid_row_nll::<P, G, _>(&vars, &inputs)?.contracted_third())
    }

    /// Build the row's fourth-order contracted tensor
    /// `T[a][b] = d_ea d_eb d_dir_u d_dir_v NLL_i` from the single-source rigid
    /// row NLL at the packed `TwoSeed<4>` bidirectional scalar (no dense `t4`).
    pub(crate) fn row_primary_fourth_contracted_tower<const P: usize, G: SlopeRowGeometry<P>>(
        &self,
        row: usize,
        block_states: &[ParameterBlockState],
        dir_u: ArrayView1<'_, f64>,
        dir_v: ArrayView1<'_, f64>,
    ) -> Result<[[f64; P]; P], String> {
        if dir_u.len() != P || dir_v.len() != P {
            return Err(SurvivalMarginalSlopeError::IncompatibleDimensions {
                reason: format!(
                    "survival rigid fourth contracted: dir lengths ({},{}) != primary dimension {P}",
                    dir_u.len(),
                    dir_v.len()
                ),
            }
            .into());
        }
        let mut u_arr = [0.0_f64; P];
        u_arr.copy_from_slice(dir_u.as_slice().ok_or_else(|| {
            "survival rigid fourth contracted: non-contiguous u direction".to_string()
        })?);
        let mut v_arr = [0.0_f64; P];
        v_arr.copy_from_slice(dir_v.as_slice().ok_or_else(|| {
            "survival rigid fourth contracted: non-contiguous v direction".to_string()
        })?);
        let inputs = rigid_row_inputs(
            self,
            block_states,
            row,
            "survival marginal-slope rigid row helper fourth",
        )?;
        let p = rigid_row_kernel_primaries::<P, G>(self, block_states, row)?;
        let vars: [TwoSeed<P>; P] =
            std::array::from_fn(|a| TwoSeed::seed(p[a], a, u_arr[a], v_arr[a]));
        Ok(rigid_row_nll::<P, G, _>(&vars, &inputs)?.contracted_fourth())
    }

    /// Compute per-row primary gradient and Hessian from the direct symbolic
    /// lowering of the single-source rigid row program. The hot inner
    /// computation uses stack arrays only; conversion to Array1/Array2 happens
    /// once at the boundary for API compatibility with outer-derivative paths.
    pub(crate) fn compute_row_primary_gradient_hessian_uncached(
        &self,
        row: usize,
        block_states: &[ParameterBlockState],
    ) -> Result<(f64, Array1<f64>, Array2<f64>), String> {
        if self.per_z_slope_active() {
            // The shared-slope frames read `Σ_k z_k` against one slope; a
            // per-score family's row program is the per-score vector frame,
            // closed form or anchored on its joint latent law (gam#2929).
            return Err(SurvivalMarginalSlopeError::UnsupportedConfiguration {
                reason: format!(
                    "survival marginal-slope row {row}: the shared-slope primary frame does not \
                     serve a per-score slope over K={} scores",
                    self.score_dim()
                ),
            }
            .into());
        }
        in_slope_frame!(self, P, Frame, {
            self.row_primary_gradient_hessian_in_frame::<P, Frame>(row, block_states)
        })
    }

    fn row_primary_gradient_hessian_in_frame<const P: usize, G: SlopeRowGeometry<P>>(
        &self,
        row: usize,
        block_states: &[ParameterBlockState],
    ) -> Result<(f64, Array1<f64>, Array2<f64>), String> {
        let inputs = rigid_row_inputs(
            self,
            block_states,
            row,
            "survival marginal-slope rigid row helper kernel",
        )?;
        let p = rigid_row_kernel_primaries::<P, G>(self, block_states, row)?;
        let (nll, grad_arr, hess_arr) = rigid_row_order2::<P, G>(&p, &inputs)?;
        // Convert stack arrays to ndarray types at the boundary.
        let grad = Array1::from_vec(grad_arr.to_vec());
        let mut hess = Array2::zeros((P, P));
        for i in 0..P {
            for j in 0..P {
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
        if self.per_z_slope_active() {
            return self.offset_channel_geometry_per_z(block_states);
        }
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
        Ok(self.assemble_offset_channels(rows))
    }

    /// The offset channels of a per-score family (gam#2929): the row program's
    /// derivatives in `(q₀, q₁, q̇₁)` are the leading primaries of the per-score
    /// vector frame, closed form or anchored on the joint latent law.
    fn offset_channel_geometry_per_z(
        &self,
        block_states: &[ParameterBlockState],
    ) -> Result<(OffsetChannelResiduals, OffsetChannelCurvatures), String> {
        let beta_time = &block_states[0].beta;
        let probit_scale = self.probit_frailty_scale();
        let rows = (0..self.n)
            .into_par_iter()
            .map_init(
                || {
                    (
                        VectorRowWorkspace::for_family(self),
                        self.slope_row_workspace(),
                    )
                },
                |(row_workspace, slope_workspace),
                 row|
                 -> Result<(usize, f64, f64, f64, [[f64; 3]; 3]), String> {
                    if self.weights[row] <= 0.0 {
                        return Ok((row, 0.0, 0.0, 0.0, [[0.0; 3]; 3]));
                    }
                    let row_workspace = row_workspace.as_mut().map_err(|error| error.clone())?;
                    let slope_workspace =
                        slope_workspace.as_mut().map_err(|error| error.clone())?;
                    let q0 = self.design_entry.dot_row(row, beta_time)
                        + self.offset_entry[row]
                        + block_states[1].eta[row];
                    let q1 = self.design_exit.dot_row(row, beta_time)
                        + self.offset_exit[row]
                        + block_states[1].eta[row];
                    let qd1 = self.design_derivative_exit.dot_row(row, beta_time)
                        + self.derivative_offset_exit[row];
                    self.fill_slope_values_for_row(row, block_states, slope_workspace)?;
                    let z_row = self.z.row(row);
                    let z = z_row.as_slice().ok_or_else(|| {
                        "per-score offset-channel score row must be contiguous".to_string()
                    })?;
                    row_workspace.evaluate_row(
                        row,
                        q0,
                        q1,
                        qd1,
                        slope_workspace.values(),
                        z,
                        self.weights[row],
                        self.entry_weight(row),
                        self.event[row],
                        self.derivative_guard,
                        probit_scale,
                    )?;
                    let (gradient, hessian) = row_workspace.derivatives();
                    let mut curvature = [[0.0; 3]; 3];
                    for a in 0..3 {
                        for b in 0..3 {
                            curvature[a][b] = hessian[[a, b]];
                        }
                    }
                    Ok((row, gradient[1], gradient[0], gradient[2], curvature))
                },
            )
            .collect::<Result<Vec<_>, String>>()?;
        Ok(self.assemble_offset_channels(rows))
    }

    fn assemble_offset_channels(
        &self,
        rows: Vec<(usize, f64, f64, f64, [[f64; 3]; 3])>,
    ) -> (OffsetChannelResiduals, OffsetChannelCurvatures) {
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
        (
            OffsetChannelResiduals {
                exit,
                entry,
                derivative,
                // Marginal-slope has no interval upper-bound channel.
                right: Array1::<f64>::zeros(self.n),
            },
            OffsetChannelCurvatures { rows: curvatures },
        )
    }
}

impl SurvivalMarginalSlopeFamily {
    pub(crate) fn row_primary_third_contracted(
        &self,
        row: usize,
        block_states: &[ParameterBlockState],
        dir: ArrayView1<'_, f64>,
    ) -> Result<Array2<f64>, String> {
        // Batched path delegating to the shared jet helper. The stack tensor is
        // then copied once into Array2 at the API boundary.
        fn to_array<const P: usize>(r: [[f64; P]; P]) -> Array2<f64> {
            let mut out = Array2::<f64>::zeros((P, P));
            for a in 0..P {
                for b in 0..P {
                    out[[a, b]] = r[a][b];
                }
            }
            out
        }
        in_slope_frame!(self, P, Frame, {
            Ok(to_array(
                self.row_primary_third_contracted_tower::<P, Frame>(row, block_states, dir)?,
            ))
        })
    }
}
