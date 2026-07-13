//! Rigid primary-tower contractions and cache helpers used to assemble
//! higher-order curvature.

use super::*;

use gam_math::jet_scalar::{JetScalar, OneSeed, TwoSeed};
use gam_math::nested_dual::JetField;

impl SurvivalMarginalSlopeFamily {
    /// Build the row's third-order contracted tensor
    /// `T[a][b] = d_ea d_eb d_dir NLL_i` from the single-source rigid row NLL
    /// at the packed `OneSeed<4>` directional scalar (no dense `t3`).
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
        let mut dir_arr = [0.0_f64; N_PRIMARY];
        dir_arr.copy_from_slice(dir.as_slice().ok_or_else(|| {
            "survival rigid third contracted: non-contiguous direction".to_string()
        })?);
        let inputs = rigid_row_inputs(
            self,
            block_states,
            row,
            "survival marginal-slope rigid row helper third",
        )?;
        let p = rigid_row_kernel_primaries(self, block_states, row)?;
        let vars: [OneSeed<4>; 4] =
            std::array::from_fn(|a| OneSeed::seed_direction(p[a], a, dir_arr[a]));
        Ok(rigid_row_nll(&vars, &inputs)?.contracted_third())
    }

    /// Build the row's fourth-order contracted tensor
    /// `T[a][b] = d_ea d_eb d_dir_u d_dir_v NLL_i` from the single-source rigid
    /// row NLL at the packed `TwoSeed<4>` bidirectional scalar (no dense `t4`).
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
        let mut u_arr = [0.0_f64; N_PRIMARY];
        u_arr.copy_from_slice(dir_u.as_slice().ok_or_else(|| {
            "survival rigid fourth contracted: non-contiguous u direction".to_string()
        })?);
        let mut v_arr = [0.0_f64; N_PRIMARY];
        v_arr.copy_from_slice(dir_v.as_slice().ok_or_else(|| {
            "survival rigid fourth contracted: non-contiguous v direction".to_string()
        })?);
        let inputs = rigid_row_inputs(
            self,
            block_states,
            row,
            "survival marginal-slope rigid row helper fourth",
        )?;
        let p = rigid_row_kernel_primaries(self, block_states, row)?;
        let vars: [TwoSeed<4>; 4] =
            std::array::from_fn(|a| TwoSeed::seed(p[a], a, u_arr[a], v_arr[a]));
        Ok(rigid_row_nll(&vars, &inputs)?.contracted_fourth())
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
        let inputs = rigid_row_inputs(
            self,
            block_states,
            row,
            "survival marginal-slope rigid row helper kernel",
        )?;
        let p = rigid_row_kernel_primaries(self, block_states, row)?;
        let (nll, grad_arr, hess_arr) = rigid_row_order2(&p, &inputs)?;
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
}
