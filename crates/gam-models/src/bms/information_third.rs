//! Third information derivatives with moving design rows. These are fifth
//! likelihood derivatives, including every derivative of the coefficient
//! pullback itself when a spatial hyperparameter moves the design.

use super::family::*;
use super::gradient_paths::*;
use super::hessian_paths::*;
use super::*;

impl BernoulliMarginalSlopeFamily {
    pub(super) fn rigid_hyper_information_third_axes(
        &self,
        states: &[ParameterBlockState],
        layout: &crate::custom_family::CustomFamilyHyperLayout,
        psi_i: usize,
        psi_j: Option<usize>,
        beta_direction: Option<&Array1<f64>>,
        options: &BlockwiseFitOptions,
    ) -> Result<Option<Vec<Array2<f64>>>, String> {
        if self.effective_flex_active(states)? {
            return Ok(None);
        }
        let locate = |index| match layout.axis(index) {
            Some(crate::custom_family::CustomFamilyHyperAxis::DesignPenalty { .. }) => {
                psi_derivative_location(layout.design_derivative_blocks(), index).ok_or_else(|| {
                    format!("BMS third information cannot locate design axis {index}")
                })
            }
            _ => Err(format!(
                "BMS third information requires a design hyperparameter, got axis {index}"
            )),
        };
        let (block_i, local_i) = locate(psi_i)?;
        let axis_i =
            self.resolve_psi_axis_spec(layout.design_derivative_blocks(), block_i, local_i)?;
        let axis_j = psi_j
            .map(|j| {
                let (block, local) = locate(j)?;
                self.resolve_psi_axis_spec(layout.design_derivative_blocks(), block, local)
                    .map(|axis| (axis, local))
            })
            .transpose()?;
        let slices = block_slices(self);
        let p = slices.total;
        let pm = slices.marginal.len();
        if beta_direction.is_some_and(|u| u.len() != p)
            || (psi_j.is_some() == beta_direction.is_some())
        {
            return Err(
                "BMS third information needs exactly one second hyperparameter or beta direction"
                    .to_string(),
            );
        }
        let n = self.y.len();
        let map_ij = if let Some((j, local_j)) = axis_j.as_ref() {
            if j.block_idx == block_i {
                let blocks = layout.design_derivative_blocks();
                Some(
                    gam_custom_family::resolve_custom_family_x_psi_psi_map(
                        &blocks[block_i][local_i],
                        &blocks[block_i][*local_j],
                        *local_j,
                        n,
                        axis_i.psi_map.ncols(),
                        0..n,
                        "BMS third information mixed design",
                        &self.policy,
                    )
                    .map_err(|e| e.to_string())?,
                )
            } else {
                None
            }
        } else {
            None
        };
        let mut row_weights = vec![0.0; n];
        for row in outer_weighted_rows(options, n) {
            row_weights[row.index] = row.weight;
        }
        // Accumulate one ordered coefficient triple per row. Expanding the
        // equal permutations after summation avoids repeated cubic ndarray
        // indexing and writes in every fifth-derivative row contraction.
        let mut out = vec![vec![0.0; p * p]; p];
        let mut x = vec![0.0; p];
        let mut xu = vec![0.0; p];
        let mut xv = vec![0.0; p];
        let mut xuv = vec![0.0; p];
        let primary = |a: usize| usize::from(a >= pm);
        for start in (0..n).step_by(4096) {
            let end = (start + 4096).min(n);
            let xm = self
                .marginal_design
                .try_row_chunk(start..end)
                .map_err(|e| e.to_string())?;
            let xg = self
                .slope_design
                .try_row_chunk(start..end)
                .map_err(|e| e.to_string())?;
            for row in start..end {
                let ht_weight = row_weights[row];
                if ht_weight == 0.0 {
                    continue;
                }
                for a in 0..p {
                    x[a] = if a < pm {
                        xm[[row - start, a]]
                    } else {
                        xg[[row - start, a - pm]]
                    };
                }
                xu.fill(0.0);
                xv.fill(0.0);
                xuv.fill(0.0);
                let mut u = [0.0; 2];
                let mut v = [0.0; 2];
                let mut uv = [0.0; 2];
                let ri = axis_i.psi_map.row_vector(row).map_err(|e| e.to_string())?;
                let offset_i = if block_i == 0 { 0 } else { pm };
                for (a, value) in ri.iter().enumerate() {
                    xu[offset_i + a] = *value;
                }
                u[block_i] = ri.dot(&states[block_i].beta);
                if let Some((axis, _)) = axis_j.as_ref() {
                    let rj = axis.psi_map.row_vector(row).map_err(|e| e.to_string())?;
                    let offset_j = if axis.block_idx == 0 { 0 } else { pm };
                    for (a, value) in rj.iter().enumerate() {
                        xv[offset_j + a] = *value;
                    }
                    v[axis.block_idx] = rj.dot(&states[axis.block_idx].beta);
                    if let Some(map) = map_ij.as_ref() {
                        let rij = map.row_vector(row).map_err(|e| e.to_string())?;
                        for (a, value) in rij.iter().enumerate() {
                            xuv[offset_i + a] = *value;
                        }
                        uv[block_i] = rij.dot(&states[block_i].beta);
                    }
                } else if let Some(direction) = beta_direction {
                    for a in 0..p {
                        v[primary(a)] += x[a] * direction[a];
                        uv[primary(a)] += xu[a] * direction[a];
                    }
                }
                let marginal = self.marginal_link_map(states[0].eta[row])?;
                let slope = states[1].eta[row];
                let t3 = self.rigid_row_third_full(row, marginal, slope)?;
                let t4 = self.rigid_row_fourth_full(row, marginal, slope)?;
                let t5 = match self.latent_measure.empirical_grid_for_training_row(row)? {
                    None => rigid_standard_normal_fifth_full(
                        marginal,
                        slope,
                        self.z[row],
                        self.y[row],
                        self.weights[row],
                        self.probit_frailty_scale(),
                    )?,
                    Some(grid) => self.empirical_rigid_row_fifth_full(
                        row,
                        marginal,
                        slope,
                        &grid.nodes,
                        &grid.weights,
                    )?,
                };
                // Contract in the two-dimensional primary space once. The
                // coefficient-width scatter must not repeat these contractions
                // for each of its p^3 entries.
                let response: [[[[f64; 3]; 2]; 2]; 2] = std::array::from_fn(|a| {
                    std::array::from_fn(|b| {
                        std::array::from_fn(|c| {
                            let mut r = [0.0; 3];
                            for d in 0..2 {
                                r[0] += t4[a][b][c][d] * uv[d];
                                r[1] += t4[a][b][c][d] * u[d];
                                r[2] += t4[a][b][c][d] * v[d];
                                for e in 0..2 {
                                    r[0] += t5[a][b][c][d][e] * u[d] * v[e];
                                }
                            }
                            r
                        })
                    })
                });
                // The three beta axes cannot share a chain-rule partition:
                // eta is linear in beta. Thus only T3, T4 and T5 occur, and
                // every nonzero mixed design term is displayed below.
                for c in 0..p {
                    for a in 0..=c {
                        for b in 0..=a {
                            let (ia, ib, ic) = (primary(a), primary(b), primary(c));
                            let [fifth_and_fourth_uv, fourth_u, fourth_v] = response[ia][ib][ic];
                            let first_u =
                                xu[a] * x[b] * x[c] + x[a] * xu[b] * x[c] + x[a] * x[b] * xu[c];
                            let first_v =
                                xv[a] * x[b] * x[c] + x[a] * xv[b] * x[c] + x[a] * x[b] * xv[c];
                            let mixed = xuv[a] * x[b] * x[c]
                                + x[a] * xuv[b] * x[c]
                                + x[a] * x[b] * xuv[c]
                                + xu[a] * xv[b] * x[c]
                                + xu[a] * x[b] * xv[c]
                                + xv[a] * xu[b] * x[c]
                                + x[a] * xu[b] * xv[c]
                                + xv[a] * x[b] * xu[c]
                                + x[a] * xv[b] * xu[c];
                            let value = ht_weight
                                * (x[a] * x[b] * x[c] * fifth_and_fourth_uv
                                    + first_u * fourth_v
                                    + first_v * fourth_u
                                    + mixed * t3[ia][ib][ic]);
                            out[c][a * p + b] += value;
                        }
                    }
                }
            }
        }
        for c in 0..p {
            for a in 0..=c {
                for b in 0..=a {
                    let value = out[c][a * p + b];
                    out[c][b * p + a] = value;
                    out[a][c * p + b] = value;
                    out[a][b * p + c] = value;
                    out[b][a * p + c] = value;
                    out[b][c * p + a] = value;
                }
            }
        }
        out.into_iter()
            .map(|data| Array2::from_shape_vec((p, p), data).map_err(|e| e.to_string()))
            .collect::<Result<Vec<_>, _>>()
            .map(Some)
    }
}
