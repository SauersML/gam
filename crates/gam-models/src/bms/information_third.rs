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
        let offsets = [0, pm];
        let widths = [pm, p - pm];
        let axes = HyperThirdInformationAxes {
            axis_i: &axis_i,
            block_i,
            axis_j: axis_j.as_ref().map(|(axis, _)| axis),
            map_ij: map_ij.as_ref(),
            beta_direction,
        };
        let mut out: Vec<Array2<f64>> = vec![Array2::<f64>::zeros((p, p)); p];
        for start in (0..n).step_by(4096) {
            let end = (start + 4096).min(n);
            let rows =
                self.rigid_hyper_third_information_rows(states, &axes, &row_weights, start..end)?;
            rigid_hyper_third_information_grams(&rows, &offsets, &widths, &axes, &mut out);
        }
        Ok(Some(out))
    }

    /// One row chunk of `rigid_hyper_information_third_axes`: the design rows `x`,
    /// `∂X/∂ψ_i` (`xu`), `∂X/∂ψ_j` (`xv`) and `∂²X/∂ψ_i∂ψ_j` (`xuv`), and each row's
    /// primary coefficients `[t4·(u,v) + t5·(u,v), t4·u, t4·v, t3]` for the eight
    /// block triples, with the row's measure weight folded in.
    pub(super) fn rigid_hyper_third_information_rows(
        &self,
        states: &[ParameterBlockState],
        axes: &HyperThirdInformationAxes<'_>,
        row_weights: &[f64],
        range: std::ops::Range<usize>,
    ) -> Result<HyperThirdInformationRows, String> {
        let slices = block_slices(self);
        let p = slices.total;
        let pm = slices.marginal.len();
        let (start, end) = (range.start, range.end);
        let primary = |a: usize| usize::from(a >= pm);
        let xm = self
            .marginal_design
            .try_row_chunk(start..end)
            .map_err(|e| e.to_string())?;
        let xg = self
            .slope_design
            .try_row_chunk(start..end)
            .map_err(|e| e.to_string())?;
        let rows = end - start;
        let mut chunk = HyperThirdInformationRows {
            x: Array2::zeros((rows, p)),
            xu: Array2::zeros((rows, p)),
            xv: Array2::zeros((rows, p)),
            xuv: Array2::zeros((rows, p)),
            coefficients: vec![[[[0.0; 4]; 2]; 2]; 2]; rows],
        };
        let offset_i = if axes.block_i == 0 { 0 } else { pm };
        for row in start..end {
            let local = row - start;
            let ht_weight = row_weights[row];
            if ht_weight == 0.0 {
                continue;
            }
            for a in 0..p {
                chunk.x[[local, a]] = if a < pm {
                    xm[[local, a]]
                } else {
                    xg[[local, a - pm]]
                };
            }
            let mut u = [0.0; 2];
            let mut v = [0.0; 2];
            let mut uv = [0.0; 2];
            let ri = axes
                .axis_i
                .psi_map
                .row_vector(row)
                .map_err(|e| e.to_string())?;
            for (a, value) in ri.iter().enumerate() {
                chunk.xu[[local, offset_i + a]] = *value;
            }
            u[axes.block_i] = ri.dot(&states[axes.block_i].beta);
            if let Some(axis) = axes.axis_j {
                let rj = axis.psi_map.row_vector(row).map_err(|e| e.to_string())?;
                let offset_j = if axis.block_idx == 0 { 0 } else { pm };
                for (a, value) in rj.iter().enumerate() {
                    chunk.xv[[local, offset_j + a]] = *value;
                }
                v[axis.block_idx] = rj.dot(&states[axis.block_idx].beta);
                if let Some(map) = axes.map_ij {
                    let rij = map.row_vector(row).map_err(|e| e.to_string())?;
                    for (a, value) in rij.iter().enumerate() {
                        chunk.xuv[[local, offset_i + a]] = *value;
                    }
                    uv[axes.block_i] = rij.dot(&states[axes.block_i].beta);
                }
            } else if let Some(direction) = axes.beta_direction {
                for a in 0..p {
                    v[primary(a)] += chunk.x[[local, a]] * direction[a];
                    uv[primary(a)] += chunk.xu[[local, a]] * direction[a];
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
            // Contract in the two-dimensional primary space once per row.
            for ia in 0..2 {
                for ib in 0..2 {
                    for ic in 0..2 {
                        let mut fifth_and_fourth_uv = 0.0;
                        let mut fourth_u = 0.0;
                        let mut fourth_v = 0.0;
                        for d in 0..2 {
                            fifth_and_fourth_uv += t4[ia][ib][ic][d] * uv[d];
                            fourth_u += t4[ia][ib][ic][d] * u[d];
                            fourth_v += t4[ia][ib][ic][d] * v[d];
                            for e in 0..2 {
                                fifth_and_fourth_uv += t5[ia][ib][ic][d][e] * u[d] * v[e];
                            }
                        }
                        chunk.coefficients[local][ia][ib][ic] = [
                            ht_weight * fifth_and_fourth_uv,
                            ht_weight * fourth_u,
                            ht_weight * fourth_v,
                            ht_weight * t3[ia][ib][ic],
                        ];
                    }
                }
            }
        }
        Ok(chunk)
    }
}

/// The moving design axes behind one ψ third-information request.
pub(super) struct HyperThirdInformationAxes<'a> {
    pub(super) axis_i: &'a PsiAxisSpec,
    pub(super) block_i: usize,
    pub(super) axis_j: Option<&'a PsiAxisSpec>,
    pub(super) map_ij: Option<&'a gam_custom_family::PsiDesignMap>,
    pub(super) beta_direction: Option<&'a Array1<f64>>,
}

/// One chunk of rows; see [`BernoulliMarginalSlopeFamily::rigid_hyper_third_information_rows`].
pub(super) struct HyperThirdInformationRows {
    pub(super) x: Array2<f64>,
    pub(super) xu: Array2<f64>,
    pub(super) xv: Array2<f64>,
    pub(super) xuv: Array2<f64>,
    pub(super) coefficients: Vec<[[[[f64; 4]; 2]; 2]; 2]>,
}

/// Add one chunk's `{D³H[ψ_i, ψ_j or δβ, e_c]}` to `out`.
///
/// Every coefficient triple `(a, b, c)` contributes a symmetric sum of design-row
/// products times the row coefficient of its block triple `(P(a), P(b), P(c))`.
/// Grouping the terms by the design rows at the first two positions leaves, for
/// output axis `c` and ordered block pair `(A, B)`, five weighted Grams:
/// `Xᵀ diag(x_c·R₀ + xu_c·R_v + xv_c·R_u + xuv_c·T₃) X`,
/// `XUᵀ diag(x_c·R_v + xv_c·T₃) X`, `XVᵀ diag(x_c·R_u + xu_c·T₃) X`,
/// `XUVᵀ diag(x_c·T₃) X` and `XUᵀ diag(x_c·T₃) XV`, the last four each with the
/// transpose of its partner position. That replaces the per-row scalar walk over
/// every ordered coefficient triple, which was a seventh of the binary
/// marginal-slope fit at 1500 rows (#979).
pub(super) fn rigid_hyper_third_information_grams(
    rows: &HyperThirdInformationRows,
    offsets: &[usize; 2],
    widths: &[usize; 2],
    axes: &HyperThirdInformationAxes<'_>,
    out: &mut [Array2<f64>],
) {
    let n_rows = rows.coefficients.len();
    let p = rows.x.ncols();
    let xv_block = axes.axis_j.map(|axis| axis.block_idx);
    let xuv_active = axes.map_ij.is_some();
    let mut weights = Array1::<f64>::zeros(n_rows);
    for c in 0..p {
        let block_c = usize::from(c >= offsets[1]);
        let xc = rows.x.column(c);
        let xuc = rows.xu.column(c);
        let xvc = rows.xv.column(c);
        let xuvc = rows.xuv.column(c);
        for block_a in 0..2 {
            for block_b in 0..2 {
                if widths[block_a] == 0 || widths[block_b] == 0 {
                    continue;
                }
                let range_a = offsets[block_a]..offsets[block_a] + widths[block_a];
                let range_b = offsets[block_b]..offsets[block_b] + widths[block_b];
                let coefficient = |local: usize| &rows.coefficients[local][block_a][block_b][block_c];
                for (local, weight) in weights.iter_mut().enumerate() {
                    let r = coefficient(local);
                    *weight = xc[local] * r[0] + xuc[local] * r[2] + xvc[local] * r[1] + xuvc[local] * r[3];
                }
                add_weighted_gram(&mut out[c], &rows.x, range_a.clone(), &rows.x, range_b.clone(), &weights, false);
                if block_a == axes.block_i {
                    for (local, weight) in weights.iter_mut().enumerate() {
                        let r = coefficient(local);
                        *weight = xc[local] * r[2] + xvc[local] * r[3];
                    }
                    add_weighted_gram(&mut out[c], &rows.xu, range_a.clone(), &rows.x, range_b.clone(), &weights, true);
                }
                if xv_block == Some(block_a) {
                    for (local, weight) in weights.iter_mut().enumerate() {
                        let r = coefficient(local);
                        *weight = xc[local] * r[1] + xuc[local] * r[3];
                    }
                    add_weighted_gram(&mut out[c], &rows.xv, range_a.clone(), &rows.x, range_b.clone(), &weights, true);
                }
                if xuv_active && block_a == axes.block_i {
                    for (local, weight) in weights.iter_mut().enumerate() {
                        *weight = xc[local] * coefficient(local)[3];
                    }
                    add_weighted_gram(&mut out[c], &rows.xuv, range_a.clone(), &rows.x, range_b.clone(), &weights, true);
                }
                if block_a == axes.block_i && xv_block == Some(block_b) {
                    for (local, weight) in weights.iter_mut().enumerate() {
                        *weight = xc[local] * coefficient(local)[3];
                    }
                    add_weighted_gram(&mut out[c], &rows.xu, range_a, &rows.xv, range_b, &weights, true);
                }
            }
        }
    }
}

/// `target[L, R] += Σ_rows w·left_L ⊗ right_R`, and the transpose into `target[R, L]`.
fn add_weighted_gram(
    target: &mut Array2<f64>,
    left: &Array2<f64>,
    left_columns: std::ops::Range<usize>,
    right: &Array2<f64>,
    right_columns: std::ops::Range<usize>,
    weights: &Array1<f64>,
    with_transpose: bool,
) {
    let left_block = left.slice(ndarray::s![.., left_columns.clone()]);
    let right_block = right.slice(ndarray::s![.., right_columns.clone()]);
    let gram = gam_linalg::faer_ndarray::fast_xt_diag_y(&left_block, weights, &right_block);
    target
        .slice_mut(ndarray::s![left_columns.clone(), right_columns.clone()])
        .scaled_add(1.0, &gram);
    if with_transpose {
        target
            .slice_mut(ndarray::s![right_columns, left_columns])
            .scaled_add(1.0, &gram.t());
    }
}
