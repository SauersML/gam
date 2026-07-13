//! Block-structured Hessian assembly: `BlockHessianAccumulator` (per-block
//! pullbacks, rank-1 psi crosses, q-geometry second pullbacks, time-wiggle
//! cross terms) and the `BlockHessianOperator` that exposes the assembled
//! curvature to the outer hyperparameter solver as a `HyperOperator`.

use super::*;

pub(crate) struct BlockHessianAccumulator {
    pub(crate) h_tt: Array2<f64>,
    pub(crate) h_mm: Array2<f64>,
    pub(crate) h_gg: Array2<f64>,
    pub(crate) h_hh: Array2<f64>,
    pub(crate) h_ww: Array2<f64>,
    /// Absorbed-influence diagonal block (#461), `p_i × p_i` (`p_i = Z̃.ncols()`).
    pub(crate) h_ii: Array2<f64>,
    pub(crate) h_tm: Array2<f64>,
    pub(crate) h_tg: Array2<f64>,
    pub(crate) h_th: Array2<f64>,
    pub(crate) h_tw: Array2<f64>,
    /// time × influence cross-block.
    pub(crate) h_ti: Array2<f64>,
    pub(crate) h_mg: Array2<f64>,
    pub(crate) h_mh: Array2<f64>,
    pub(crate) h_mw: Array2<f64>,
    /// marginal × influence cross-block.
    pub(crate) h_mi: Array2<f64>,
    pub(crate) h_gh: Array2<f64>,
    pub(crate) h_gw: Array2<f64>,
    /// logslope × influence cross-block.
    pub(crate) h_gi: Array2<f64>,
    pub(crate) h_hw: Array2<f64>,
    /// score_warp × influence cross-block.
    pub(crate) h_hi: Array2<f64>,
    /// link_dev × influence cross-block.
    pub(crate) h_wi: Array2<f64>,
}

pub(crate) const PULLBACK_PARALLEL_MIN_CELLS: usize = 16_384;

pub(crate) const PULLBACK_PARALLEL_TARGET_CELLS: usize = 65_536;

impl BlockHessianAccumulator {
    pub(crate) fn new(
        p_t: usize,
        p_m: usize,
        p_g: usize,
        p_h: usize,
        p_w: usize,
        p_i: usize,
    ) -> Self {
        Self {
            h_tt: Array2::zeros((p_t, p_t)),
            h_mm: Array2::zeros((p_m, p_m)),
            h_gg: Array2::zeros((p_g, p_g)),
            h_hh: Array2::zeros((p_h, p_h)),
            h_ww: Array2::zeros((p_w, p_w)),
            h_ii: Array2::zeros((p_i, p_i)),
            h_tm: Array2::zeros((p_t, p_m)),
            h_tg: Array2::zeros((p_t, p_g)),
            h_th: Array2::zeros((p_t, p_h)),
            h_tw: Array2::zeros((p_t, p_w)),
            h_ti: Array2::zeros((p_t, p_i)),
            h_mg: Array2::zeros((p_m, p_g)),
            h_mh: Array2::zeros((p_m, p_h)),
            h_mw: Array2::zeros((p_m, p_w)),
            h_mi: Array2::zeros((p_m, p_i)),
            h_gh: Array2::zeros((p_g, p_h)),
            h_gw: Array2::zeros((p_g, p_w)),
            h_gi: Array2::zeros((p_g, p_i)),
            h_hw: Array2::zeros((p_h, p_w)),
            h_hi: Array2::zeros((p_h, p_i)),
            h_wi: Array2::zeros((p_w, p_i)),
        }
    }

    pub(crate) fn block_dims(&self) -> (usize, usize, usize, usize, usize, usize) {
        (
            self.h_tt.nrows(),
            self.h_mm.nrows(),
            self.h_gg.nrows(),
            self.h_hh.nrows(),
            self.h_ww.nrows(),
            self.h_ii.nrows(),
        )
    }

    pub(crate) fn deterministic_lhs_chunks(
        lhs_len: usize,
        rhs_len: usize,
    ) -> Vec<std::ops::Range<usize>> {
        let cells = lhs_len.saturating_mul(rhs_len);
        if cells < PULLBACK_PARALLEL_MIN_CELLS || lhs_len <= 1 || rayon::current_num_threads() <= 1
        {
            return std::iter::once(0..lhs_len).collect();
        }
        let chunk_len = (PULLBACK_PARALLEL_TARGET_CELLS / rhs_len.max(1))
            .max(1)
            .min(lhs_len);
        (0..lhs_len)
            .step_by(chunk_len)
            .map(|start| start..(start + chunk_len).min(lhs_len))
            .collect()
    }

    pub(crate) fn add_ordered_lhs_partials(
        target: &mut Array2<f64>,
        partials: Vec<Result<(std::ops::Range<usize>, Array2<f64>), String>>,
    ) -> Result<(), String> {
        for partial in partials {
            let (range, block) = partial?;
            target.slice_mut(s![range, ..]).scaled_add(1.0, &block);
        }
        Ok(())
    }

    /// Accumulate a primary-space Hessian into block-local matrices.
    /// Equivalent to `add_pullback_primary_hessian` but avoids the p×p target.
    pub(crate) fn add_pullback(
        &mut self,
        family: &SurvivalMarginalSlopeFamily,
        row: usize,
        primary_hessian: &Array2<f64>,
    ) -> Result<(), String> {
        // Time×time block: 3×3 design cross-products
        let time_designs = [
            &family.design_entry,
            &family.design_exit,
            &family.design_derivative_exit,
        ];
        let (p_t, _, _, _, _, _) = self.block_dims();
        let tt_chunks = Self::deterministic_lhs_chunks(p_t, p_t);
        if tt_chunks.len() == 1 {
            for a in 0..3 {
                for b in 0..3 {
                    time_designs[a]
                        .row_outer_into(
                            row,
                            time_designs[b],
                            primary_hessian[[a, b]],
                            &mut self.h_tt,
                        )
                        .map_err(|e| format!("add_pullback time row_outer_into: {e}"))?;
                }
            }
        } else {
            let time_rows: Vec<Array1<f64>> = time_designs
                .iter()
                .map(|des| {
                    let chunk = des
                        .try_row_chunk(row..row + 1)
                        .map_err(|e| format!("add_pullback time design try_row_chunk: {e}"))?;
                    Ok(chunk.row(0).to_owned())
                })
                .collect::<Result<_, String>>()?;
            let time_partials: Vec<Result<(std::ops::Range<usize>, Array2<f64>), String>> =
                tt_chunks
                    .into_par_iter()
                    .map(|chunk| {
                        let mut local = Array2::zeros((chunk.len(), p_t));
                        for (local_a, coeff_a) in chunk.clone().enumerate() {
                            for coeff_b in 0..p_t {
                                let mut value = 0.0;
                                for a in 0..3 {
                                    for b in 0..3 {
                                        value += primary_hessian[[a, b]]
                                            * time_rows[a][coeff_a]
                                            * time_rows[b][coeff_b];
                                    }
                                }
                                local[[local_a, coeff_b]] = value;
                            }
                        }
                        Ok((chunk, local))
                    })
                    .collect();
            Self::add_ordered_lhs_partials(&mut self.h_tt, time_partials)?;
        }

        // Marginal×marginal: single rank-1 with combined weight
        let mm_weight = primary_hessian[[0, 0]]
            + primary_hessian[[0, 1]]
            + primary_hessian[[1, 0]]
            + primary_hessian[[1, 1]];
        family
            .marginal_design
            .syr_row_into(row, mm_weight, &mut self.h_mm)
            .map_err(|e| format!("add_pullback marginal syr_row_into: {e}"))?;

        // Logslope×logslope: single rank-1
        family
            .logslope_layout.coefficient_design()
            .syr_row_into(row, primary_hessian[[3, 3]], &mut self.h_gg)
            .map_err(|e| format!("add_pullback logslope syr_row_into: {e}"))?;

        // Marginal×logslope cross-block
        let mg_weight = primary_hessian[[0, 3]] + primary_hessian[[1, 3]];
        if mg_weight != 0.0 {
            let m_chunk = family
                .marginal_design
                .try_row_chunk(row..row + 1)
                .map_err(|e| format!("add_pullback marginal_design try_row_chunk: {e}"))?;
            let m_row = m_chunk.row(0);
            let g_chunk = family
                .logslope_layout.coefficient_design()
                .try_row_chunk(row..row + 1)
                .map_err(|e| format!("add_pullback logslope_design try_row_chunk: {e}"))?;
            let g_row = g_chunk.row(0);
            ndarray::linalg::general_mat_mul(
                mg_weight,
                &m_row.view().insert_axis(Axis(1)),
                &g_row.view().insert_axis(Axis(0)),
                1.0,
                &mut self.h_mg,
            );
        }

        // Time×logslope cross-block
        let tg_weights = [
            primary_hessian[[0, 3]],
            primary_hessian[[1, 3]],
            primary_hessian[[2, 3]],
        ];
        let g_chunk = family
            .logslope_layout.coefficient_design()
            .try_row_chunk(row..row + 1)
            .map_err(|e| format!("add_pullback logslope_design try_row_chunk: {e}"))?;
        let g_row = g_chunk.row(0);
        for (des, alpha) in time_designs.iter().zip(tg_weights.iter()) {
            if *alpha == 0.0 {
                continue;
            }
            let t_chunk = des
                .try_row_chunk(row..row + 1)
                .map_err(|e| format!("add_pullback time design try_row_chunk: {e}"))?;
            let t_row = t_chunk.row(0);
            ndarray::linalg::general_mat_mul(
                *alpha,
                &t_row.view().insert_axis(Axis(1)),
                &g_row.view().insert_axis(Axis(0)),
                1.0,
                &mut self.h_tg,
            );
        }

        // Time×marginal cross-block
        let tm_weights = [
            primary_hessian[[0, 0]] + primary_hessian[[0, 1]],
            primary_hessian[[1, 0]] + primary_hessian[[1, 1]],
            primary_hessian[[2, 0]] + primary_hessian[[2, 1]],
        ];
        let m_chunk = family
            .marginal_design
            .try_row_chunk(row..row + 1)
            .map_err(|e| format!("add_pullback marginal_design try_row_chunk: {e}"))?;
        let m_row = m_chunk.row(0);
        for (des, alpha) in time_designs.iter().zip(tm_weights.iter()) {
            if *alpha == 0.0 {
                continue;
            }
            let t_chunk = des
                .try_row_chunk(row..row + 1)
                .map_err(|e| format!("add_pullback time design try_row_chunk: {e}"))?;
            let t_row = t_chunk.row(0);
            ndarray::linalg::general_mat_mul(
                *alpha,
                &t_row.view().insert_axis(Axis(1)),
                &m_row.view().insert_axis(Axis(0)),
                1.0,
                &mut self.h_tm,
            );
        }

        let primary = flex_primary_slices(family);
        if let Some(h_range) = primary.h.as_ref() {
            for local_idx in 0..h_range.len() {
                let idx = h_range.start + local_idx;
                let th_weights = [
                    primary_hessian[[0, idx]],
                    primary_hessian[[1, idx]],
                    primary_hessian[[2, idx]],
                ];
                for (des, alpha) in time_designs.iter().zip(th_weights.iter()) {
                    if *alpha == 0.0 {
                        continue;
                    }
                    let t_chunk = des
                        .try_row_chunk(row..row + 1)
                        .map_err(|e| format!("add_pullback time design try_row_chunk: {e}"))?;
                    let t_row = t_chunk.row(0);
                    for coeff_idx in 0..t_row.len() {
                        self.h_th[[coeff_idx, local_idx]] += *alpha * t_row[coeff_idx];
                    }
                }

                let mh_weight = primary_hessian[[0, idx]] + primary_hessian[[1, idx]];
                if mh_weight != 0.0 {
                    let m_chunk = family
                        .marginal_design
                        .try_row_chunk(row..row + 1)
                        .map_err(|e| format!("add_pullback marginal_design try_row_chunk: {e}"))?;
                    let m_row = m_chunk.row(0);
                    for coeff_idx in 0..m_row.len() {
                        self.h_mh[[coeff_idx, local_idx]] += mh_weight * m_row[coeff_idx];
                    }
                }

                let gh_weight = primary_hessian[[3, idx]];
                if gh_weight != 0.0 {
                    let g_chunk = family
                        .logslope_layout.coefficient_design()
                        .try_row_chunk(row..row + 1)
                        .map_err(|e| format!("add_pullback logslope_design try_row_chunk: {e}"))?;
                    let g_row = g_chunk.row(0);
                    for coeff_idx in 0..g_row.len() {
                        self.h_gh[[coeff_idx, local_idx]] += gh_weight * g_row[coeff_idx];
                    }
                }
            }

            for left_local in 0..h_range.len() {
                for right_local in 0..h_range.len() {
                    self.h_hh[[left_local, right_local]] +=
                        primary_hessian[[h_range.start + left_local, h_range.start + right_local]];
                }
            }
        }

        if let Some(w_range) = primary.w.as_ref() {
            for local_idx in 0..w_range.len() {
                let idx = w_range.start + local_idx;
                let tw_weights = [
                    primary_hessian[[0, idx]],
                    primary_hessian[[1, idx]],
                    primary_hessian[[2, idx]],
                ];
                for (des, alpha) in time_designs.iter().zip(tw_weights.iter()) {
                    if *alpha == 0.0 {
                        continue;
                    }
                    let t_chunk = des
                        .try_row_chunk(row..row + 1)
                        .map_err(|e| format!("add_pullback time design try_row_chunk: {e}"))?;
                    let t_row = t_chunk.row(0);
                    for coeff_idx in 0..t_row.len() {
                        self.h_tw[[coeff_idx, local_idx]] += *alpha * t_row[coeff_idx];
                    }
                }

                let mw_weight = primary_hessian[[0, idx]] + primary_hessian[[1, idx]];
                if mw_weight != 0.0 {
                    let m_chunk = family
                        .marginal_design
                        .try_row_chunk(row..row + 1)
                        .map_err(|e| format!("add_pullback marginal_design try_row_chunk: {e}"))?;
                    let m_row = m_chunk.row(0);
                    for coeff_idx in 0..m_row.len() {
                        self.h_mw[[coeff_idx, local_idx]] += mw_weight * m_row[coeff_idx];
                    }
                }

                let gw_weight = primary_hessian[[3, idx]];
                if gw_weight != 0.0 {
                    let g_chunk = family
                        .logslope_layout.coefficient_design()
                        .try_row_chunk(row..row + 1)
                        .map_err(|e| format!("add_pullback logslope_design try_row_chunk: {e}"))?;
                    let g_row = g_chunk.row(0);
                    for coeff_idx in 0..g_row.len() {
                        self.h_gw[[coeff_idx, local_idx]] += gw_weight * g_row[coeff_idx];
                    }
                }
            }

            for left_local in 0..w_range.len() {
                for right_local in 0..w_range.len() {
                    self.h_ww[[left_local, right_local]] +=
                        primary_hessian[[w_range.start + left_local, w_range.start + right_local]];
                }
            }
        }

        if let (Some(h_range), Some(w_range)) = (primary.h.as_ref(), primary.w.as_ref()) {
            for h_local in 0..h_range.len() {
                for w_local in 0..w_range.len() {
                    self.h_hw[[h_local, w_local]] +=
                        primary_hessian[[h_range.start + h_local, w_range.start + w_local]];
                }
            }
        }

        // Absorbed-influence block (#461). The absorber contributes a single
        // primary scalar `o_infl` at index `primary.infl`, whose design row is
        // `Z̃_infl[row,:]` (the residualized leakage columns). It projects exactly
        // like the single-scalar logslope `g` index (primary 3 → logslope_design)
        // but through `Z̃` and crossed with every block, plus the diagonal
        // `h_ii = primary_hessian[[infl, infl]]·Z̃Z̃ᵀ`. `o_infl` is an additive η₁
        // shift (∂η₁/∂o_infl = 1), so the per-block primary weights mirror the
        // existing channels: time uses {q0,q1,qd1}, marginal {q0,q1}, logslope
        // {g}, score_warp/link_dev their own primary ranges.
        if let Some(infl_idx) = primary.infl {
            let z_tilde = family.influence_absorber.as_ref().ok_or_else(|| {
                "add_pullback: influence primary index present but no Z̃ design".to_string()
            })?;
            let z_row = z_tilde.row(row);
            let p_i = z_row.len();

            // Influence × influence diagonal: primary_hessian[[infl,infl]]·Z̃Z̃ᵀ.
            let ii_weight = primary_hessian[[infl_idx, infl_idx]];
            if ii_weight != 0.0 {
                let z_col = z_row.view().insert_axis(Axis(1));
                ndarray::linalg::general_mat_mul(
                    ii_weight,
                    &z_col,
                    &z_row.view().insert_axis(Axis(0)),
                    1.0,
                    &mut self.h_ii,
                );
            }

            // Time × influence: each time sub-design (entry/exit/deriv) crossed
            // with Z̃ at the matching primary weight.
            let ti_weights = [
                primary_hessian[[0, infl_idx]],
                primary_hessian[[1, infl_idx]],
                primary_hessian[[2, infl_idx]],
            ];
            for (des, alpha) in time_designs.iter().zip(ti_weights.iter()) {
                if *alpha == 0.0 {
                    continue;
                }
                let t_chunk = des
                    .try_row_chunk(row..row + 1)
                    .map_err(|e| format!("add_pullback time design try_row_chunk: {e}"))?;
                let t_row = t_chunk.row(0);
                for (t_coeff, &t_val) in t_row.iter().enumerate() {
                    for i_coeff in 0..p_i {
                        self.h_ti[[t_coeff, i_coeff]] += *alpha * t_val * z_row[i_coeff];
                    }
                }
            }

            // Marginal × influence (marginal uses q0 + q1).
            let mi_weight = primary_hessian[[0, infl_idx]] + primary_hessian[[1, infl_idx]];
            if mi_weight != 0.0 {
                let m_chunk = family
                    .marginal_design
                    .try_row_chunk(row..row + 1)
                    .map_err(|e| format!("add_pullback marginal_design try_row_chunk: {e}"))?;
                let m_row = m_chunk.row(0);
                for (m_coeff, &m_val) in m_row.iter().enumerate() {
                    for i_coeff in 0..p_i {
                        self.h_mi[[m_coeff, i_coeff]] += mi_weight * m_val * z_row[i_coeff];
                    }
                }
            }

            // Logslope × influence (logslope uses g, primary index 3).
            let gi_weight = primary_hessian[[3, infl_idx]];
            if gi_weight != 0.0 {
                let g_chunk = family
                    .logslope_layout.coefficient_design()
                    .try_row_chunk(row..row + 1)
                    .map_err(|e| format!("add_pullback logslope_design try_row_chunk: {e}"))?;
                let g_row = g_chunk.row(0);
                for (g_coeff, &g_val) in g_row.iter().enumerate() {
                    for i_coeff in 0..p_i {
                        self.h_gi[[g_coeff, i_coeff]] += gi_weight * g_val * z_row[i_coeff];
                    }
                }
            }

            // Score-warp × influence (score_warp basis is itself in the primary
            // vector, so its row design is the identity on `h_range`).
            if let Some(h_range) = primary.h.as_ref() {
                for h_local in 0..h_range.len() {
                    let weight = primary_hessian[[h_range.start + h_local, infl_idx]];
                    if weight == 0.0 {
                        continue;
                    }
                    for i_coeff in 0..p_i {
                        self.h_hi[[h_local, i_coeff]] += weight * z_row[i_coeff];
                    }
                }
            }

            // Link-dev × influence (link_dev basis is in the primary vector too).
            if let Some(w_range) = primary.w.as_ref() {
                for w_local in 0..w_range.len() {
                    let weight = primary_hessian[[w_range.start + w_local, infl_idx]];
                    if weight == 0.0 {
                        continue;
                    }
                    for i_coeff in 0..p_i {
                        self.h_wi[[w_local, i_coeff]] += weight * z_row[i_coeff];
                    }
                }
            }
        }

        Ok(())
    }

    /// Add a rank-1 update from psi_row (in the psi block) crossed with the
    /// pullback of a primary-space vector. Adds both left⊗right and right⊗left.
    pub(crate) fn add_rank1_psi_cross(
        &mut self,
        family: &SurvivalMarginalSlopeFamily,
        row: usize,
        psi_block_idx: usize,
        psi_row: &Array1<f64>,
        right_primary: &Array1<f64>,
    ) -> Result<(), String> {
        // right_primary components mapped to blocks:
        // time:     entry*rp[0] + exit*rp[1] + deriv*rp[2]
        // marginal: marginal*(rp[0] + rp[1])
        // logslope: logslope*rp[3]
        let psi_col = psi_row.view().insert_axis(Axis(1));

        // Block (psi, time): psi_row ⊗ right_time
        // Block (time, psi): right_time ⊗ psi_row  (= transpose of above)
        let time_designs = [
            (&family.design_entry, right_primary[0]),
            (&family.design_exit, right_primary[1]),
            (&family.design_derivative_exit, right_primary[2]),
        ];
        for (des, alpha) in &time_designs {
            if *alpha == 0.0 {
                continue;
            }
            let t_chunk = des
                .try_row_chunk(row..row + 1)
                .map_err(|e| format!("add_rank1_psi_cross time design try_row_chunk: {e}"))?;
            let t_row = t_chunk.row(0);
            let t_col = t_row.view().insert_axis(Axis(1));
            match psi_block_idx {
                1 => {
                    // psi=marginal: (time, marginal) block = h_tm
                    // right⊗left: right_time ⊗ psi_row → h_tm
                    ndarray::linalg::general_mat_mul(
                        *alpha,
                        &t_col,
                        &psi_row.view().insert_axis(Axis(0)),
                        1.0,
                        &mut self.h_tm,
                    );
                    // left⊗right: psi_row ⊗ right_time → h_tm^T (handled by symmetry)
                }
                2 => {
                    // psi=logslope: (time, logslope) block = h_tg
                    ndarray::linalg::general_mat_mul(
                        *alpha,
                        &t_col,
                        &psi_row.view().insert_axis(Axis(0)),
                        1.0,
                        &mut self.h_tg,
                    );
                }
                _ => {}
            }
        }

        // Block (psi, marginal) or (marginal, psi)
        let m_alpha = right_primary[0] + right_primary[1];
        if m_alpha != 0.0 {
            let m_chunk = family
                .marginal_design
                .try_row_chunk(row..row + 1)
                .map_err(|e| format!("add_rank1_psi_cross marginal_design try_row_chunk: {e}"))?;
            let m_row = m_chunk.row(0);
            match psi_block_idx {
                1 => {
                    // psi=marginal: (marginal, marginal) = h_mm, symmetric rank-2
                    ndarray::linalg::general_mat_mul(
                        m_alpha,
                        &psi_col,
                        &m_row.view().insert_axis(Axis(0)),
                        1.0,
                        &mut self.h_mm,
                    );
                    ndarray::linalg::general_mat_mul(
                        m_alpha,
                        &m_row.view().insert_axis(Axis(1)),
                        &psi_row.view().insert_axis(Axis(0)),
                        1.0,
                        &mut self.h_mm,
                    );
                }
                2 => {
                    // psi=logslope: (marginal, logslope) block = h_mg
                    // left⊗right: psi_row(logslope) ⊗ m_row → goes to h_mg^T
                    // right⊗left: m_row ⊗ psi_row(logslope) → goes to h_mg
                    ndarray::linalg::general_mat_mul(
                        m_alpha,
                        &m_row.view().insert_axis(Axis(1)),
                        &psi_row.view().insert_axis(Axis(0)),
                        1.0,
                        &mut self.h_mg,
                    );
                }
                _ => {}
            }
        }

        // Block (psi, logslope) or (logslope, psi)
        if right_primary[3] != 0.0 {
            let g_chunk = family
                .logslope_layout.coefficient_design()
                .try_row_chunk(row..row + 1)
                .map_err(|e| format!("add_rank1_psi_cross logslope_design try_row_chunk: {e}"))?;
            let g_row = g_chunk.row(0);
            match psi_block_idx {
                1 => {
                    // psi=marginal: (marginal, logslope) = h_mg
                    ndarray::linalg::general_mat_mul(
                        right_primary[3],
                        &psi_col,
                        &g_row.view().insert_axis(Axis(0)),
                        1.0,
                        &mut self.h_mg,
                    );
                }
                2 => {
                    // psi=logslope: (logslope, logslope) = h_gg, symmetric rank-2
                    ndarray::linalg::general_mat_mul(
                        right_primary[3],
                        &psi_col,
                        &g_row.view().insert_axis(Axis(0)),
                        1.0,
                        &mut self.h_gg,
                    );
                    ndarray::linalg::general_mat_mul(
                        right_primary[3],
                        &g_row.view().insert_axis(Axis(1)),
                        &psi_row.view().insert_axis(Axis(0)),
                        1.0,
                        &mut self.h_gg,
                    );
                }
                _ => {}
            }
        }

        let primary = flex_primary_slices(family);
        if let Some(h_range) = primary.h.as_ref() {
            for local_idx in 0..h_range.len() {
                let alpha = right_primary[h_range.start + local_idx];
                if alpha == 0.0 {
                    continue;
                }
                match psi_block_idx {
                    1 => {
                        for coeff_idx in 0..psi_row.len() {
                            self.h_mh[[coeff_idx, local_idx]] += alpha * psi_row[coeff_idx];
                        }
                    }
                    2 => {
                        for coeff_idx in 0..psi_row.len() {
                            self.h_gh[[coeff_idx, local_idx]] += alpha * psi_row[coeff_idx];
                        }
                    }
                    _ => {}
                }
            }
        }
        if let Some(w_range) = primary.w.as_ref() {
            for local_idx in 0..w_range.len() {
                let alpha = right_primary[w_range.start + local_idx];
                if alpha == 0.0 {
                    continue;
                }
                match psi_block_idx {
                    1 => {
                        for coeff_idx in 0..psi_row.len() {
                            self.h_mw[[coeff_idx, local_idx]] += alpha * psi_row[coeff_idx];
                        }
                    }
                    2 => {
                        for coeff_idx in 0..psi_row.len() {
                            self.h_gw[[coeff_idx, local_idx]] += alpha * psi_row[coeff_idx];
                        }
                    }
                    _ => {}
                }
            }
        }
        Ok(())
    }

    /// Add outer product of two psi block-local rows (possibly in different blocks).
    /// Adds both α·(a ⊗ b) and α·(b ⊗ a) to maintain symmetry.
    ///
    /// The full p×p symmetric Hessian has blocks:
    ///   (block_i, block_j) += α · psi_row_i ⊗ psi_row_j
    ///   (block_j, block_i) += α · psi_row_j ⊗ psi_row_i   (= transpose)
    /// Our off-diagonal storage convention (h_mg = marginal×logslope) handles
    /// the transpose automatically in to_dense/operator assembly.
    pub(crate) fn add_psi_psi_outer(
        &mut self,
        block_i: usize,
        psi_row_i: &Array1<f64>,
        block_j: usize,
        psi_row_j: &Array1<f64>,
        alpha: f64,
    ) {
        add_two_surface_psi_outer(
            block_i,
            psi_row_i,
            block_j,
            psi_row_j,
            alpha,
            1,
            2,
            &mut self.h_mm,
            &mut self.h_gg,
            &mut self.h_mg,
        );
    }

    /// Returns the `(row, col)` block of the symmetric joint Hessian as a view
    /// into the stored half, transposing on the fly when the requested pair is
    /// below the diagonal.
    ///
    /// This is the single source of truth for the block layout: it is the only
    /// place that maps a block pair onto one of the fifteen stored matrices and
    /// decides whether a transpose is needed. The match is exhaustive over all
    /// 5×5 pairs, so adding a block produces a hard compile error here rather
    /// than a silent gap in one assembler. Diagonal blocks are symmetric by
    /// construction and returned untransposed.
    #[inline]
    pub(crate) fn block_view(&self, row: HessBlock, col: HessBlock) -> ArrayView2<'_, f64> {
        use HessBlock::*;
        match (row, col) {
            (Time, Time) => self.h_tt.view(),
            (Marginal, Marginal) => self.h_mm.view(),
            (Logslope, Logslope) => self.h_gg.view(),
            (ScoreWarp, ScoreWarp) => self.h_hh.view(),
            (LinkDev, LinkDev) => self.h_ww.view(),

            (Time, Marginal) => self.h_tm.view(),
            (Marginal, Time) => self.h_tm.t(),
            (Time, Logslope) => self.h_tg.view(),
            (Logslope, Time) => self.h_tg.t(),
            (Time, ScoreWarp) => self.h_th.view(),
            (ScoreWarp, Time) => self.h_th.t(),
            (Time, LinkDev) => self.h_tw.view(),
            (LinkDev, Time) => self.h_tw.t(),

            (Marginal, Logslope) => self.h_mg.view(),
            (Logslope, Marginal) => self.h_mg.t(),
            (Marginal, ScoreWarp) => self.h_mh.view(),
            (ScoreWarp, Marginal) => self.h_mh.t(),
            (Marginal, LinkDev) => self.h_mw.view(),
            (LinkDev, Marginal) => self.h_mw.t(),

            (Logslope, ScoreWarp) => self.h_gh.view(),
            (ScoreWarp, Logslope) => self.h_gh.t(),
            (Logslope, LinkDev) => self.h_gw.view(),
            (LinkDev, Logslope) => self.h_gw.t(),

            (ScoreWarp, LinkDev) => self.h_hw.view(),
            (LinkDev, ScoreWarp) => self.h_hw.t(),

            (Influence, Influence) => self.h_ii.view(),
            (Time, Influence) => self.h_ti.view(),
            (Influence, Time) => self.h_ti.t(),
            (Marginal, Influence) => self.h_mi.view(),
            (Influence, Marginal) => self.h_mi.t(),
            (Logslope, Influence) => self.h_gi.view(),
            (Influence, Logslope) => self.h_gi.t(),
            (ScoreWarp, Influence) => self.h_hi.view(),
            (Influence, ScoreWarp) => self.h_hi.t(),
            (LinkDev, Influence) => self.h_wi.view(),
            (Influence, LinkDev) => self.h_wi.t(),
        }
    }

    /// Visits the present off-diagonal block pairs `(lo, hi)` in column-major
    /// upper-triangle order (`hi` outermost), skipping pairs whose blocks are
    /// inactive. This fixes the off-diagonal traversal order shared by
    /// [`Self::to_dense`] and [`BlockHessianOperator::bilinear`].
    #[inline]
    pub(crate) fn for_each_offdiagonal_pair(
        slices: &BlockSlices,
        mut visit: impl FnMut(HessBlock, std::ops::Range<usize>, HessBlock, std::ops::Range<usize>),
    ) {
        for hi_idx in 1..HessBlock::ALL.len() {
            let hi = HessBlock::ALL[hi_idx];
            let Some(r_hi) = slices.range_of(hi) else {
                continue;
            };
            for &lo in &HessBlock::ALL[..hi_idx] {
                let Some(r_lo) = slices.range_of(lo) else {
                    continue;
                };
                visit(lo, r_lo, hi, r_hi.clone());
            }
        }
    }

    /// Assemble into a dense p×p matrix.
    pub(crate) fn to_dense(&self, slices: &BlockSlices) -> Array2<f64> {
        let mut out = Array2::zeros((slices.total, slices.total));
        for block in HessBlock::ALL {
            let Some(range) = slices.range_of(block) else {
                continue;
            };
            out.slice_mut(s![range.clone(), range])
                .assign(&self.block_view(block, block));
        }
        Self::for_each_offdiagonal_pair(slices, |lo, r_lo, hi, r_hi| {
            out.slice_mut(s![r_lo.clone(), r_hi.clone()])
                .assign(&self.block_view(lo, hi));
            out.slice_mut(s![r_hi, r_lo])
                .assign(&self.block_view(hi, lo));
        });
        out
    }

    pub(crate) fn into_operator(self, slices: BlockSlices) -> BlockHessianOperator {
        BlockHessianOperator {
            blocks: self,
            slices,
        }
    }

    pub(crate) fn add(&mut self, other: &BlockHessianAccumulator) {
        self.h_tt += &other.h_tt;
        self.h_mm += &other.h_mm;
        self.h_gg += &other.h_gg;
        self.h_hh += &other.h_hh;
        self.h_ww += &other.h_ww;
        self.h_ii += &other.h_ii;
        self.h_tm += &other.h_tm;
        self.h_tg += &other.h_tg;
        self.h_th += &other.h_th;
        self.h_tw += &other.h_tw;
        self.h_ti += &other.h_ti;
        self.h_mg += &other.h_mg;
        self.h_mh += &other.h_mh;
        self.h_mw += &other.h_mw;
        self.h_mi += &other.h_mi;
        self.h_gh += &other.h_gh;
        self.h_gw += &other.h_gw;
        self.h_gi += &other.h_gi;
        self.h_hw += &other.h_hw;
        self.h_hi += &other.h_hi;
        self.h_wi += &other.h_wi;
    }

    pub(crate) fn diagonal(&self, slices: &BlockSlices) -> Array1<f64> {
        let mut out = Array1::zeros(slices.total);
        for block in HessBlock::ALL {
            let Some(range) = slices.range_of(block) else {
                continue;
            };
            out.slice_mut(s![range])
                .assign(&self.block_view(block, block).diag());
        }
        out
    }
}

impl std::ops::AddAssign<&BlockHessianAccumulator> for BlockHessianAccumulator {
    fn add_assign(&mut self, other: &BlockHessianAccumulator) {
        self.add(other);
    }
}

impl BlockHessianAccumulator {
    /// Lifted pullback: J^T H J + Σ_a f_a K_a using actual Jacobians
    pub(crate) fn add_pullback_with_q_geometry(
        &mut self,
        family: &SurvivalMarginalSlopeFamily,
        row: usize,
        qg: &SurvivalMarginalSlopeDynamicRow,
        fg: &Array1<f64>,
        ph: &Array2<f64>,
    ) -> Result<(), String> {
        let jt = [&qg.dq0_time, &qg.dq1_time, &qg.dqd1_time];
        let jm = [&qg.dq0_marginal, &qg.dq1_marginal, &qg.dqd1_marginal];
        let ktt = [&qg.d2q0_time_time, &qg.d2q1_time_time, &qg.d2qd1_time_time];
        let ktm = [
            &qg.d2q0_time_marginal,
            &qg.d2q1_time_marginal,
            &qg.d2qd1_time_marginal,
        ];
        let kmm = [
            &qg.d2q0_marginal_marginal,
            &qg.d2q1_marginal_marginal,
            &qg.d2qd1_marginal_marginal,
        ];
        let pt = jt[0].len();
        let pm = jm[0].len();
        // Serial accumulation directly into self.h_tt / h_mm / h_tm.
        // The outer (over rows) parallelism is what saturates threads at
        // large-scale N; nesting inner par_iter here adds work-stealing
        // overhead, per-chunk Array2::zeros allocations, and risks
        // OnceLock + nested rayon deadlock. Row order, accumulation
        // order, and per-row arithmetic remain bit-identical to the
        // previous serialised partial-merge pass.
        for a in 0..pt {
            for b in 0..pt {
                let mut v = 0.0;
                for u in 0..3 {
                    for w in 0..3 {
                        v += ph[[u, w]] * jt[u][a] * jt[w][b];
                    }
                }
                for u in 0..3 {
                    v += fg[u] * ktt[u][[a, b]];
                }
                self.h_tt[[a, b]] += v;
            }
        }

        for a in 0..pm {
            for b in 0..pm {
                let mut v = 0.0;
                for u in 0..3 {
                    for w in 0..3 {
                        v += ph[[u, w]] * jm[u][a] * jm[w][b];
                    }
                }
                for u in 0..3 {
                    v += fg[u] * kmm[u][[a, b]];
                }
                self.h_mm[[a, b]] += v;
            }
        }
        family
            .logslope_layout.coefficient_design()
            .syr_row_into(row, ph[[3, 3]], &mut self.h_gg)
            .map_err(|e| format!("add_pullback_with_q_geometry gg syr: {e}"))?;
        for a in 0..pt {
            for b in 0..pm {
                let mut v = 0.0;
                for u in 0..3 {
                    for w in 0..3 {
                        v += ph[[u, w]] * jt[u][a] * jm[w][b];
                    }
                }
                for u in 0..3 {
                    v += fg[u] * ktm[u][[a, b]];
                }
                self.h_tm[[a, b]] += v;
            }
        }
        let gc = family
            .logslope_layout.coefficient_design()
            .try_row_chunk(row..row + 1)
            .map_err(|e| format!("add_pullback_with_q_geometry logslope try_row_chunk: {e}"))?;
        let gr = gc.row(0);
        for a in 0..pt {
            let mut w = 0.0;
            for u in 0..3 {
                w += ph[[u, 3]] * jt[u][a];
            }
            if w != 0.0 {
                for b in 0..gr.len() {
                    self.h_tg[[a, b]] += w * gr[b];
                }
            }
        }
        for a in 0..pm {
            let mut w = 0.0;
            for u in 0..3 {
                w += ph[[u, 3]] * jm[u][a];
            }
            if w != 0.0 {
                for b in 0..gr.len() {
                    self.h_mg[[a, b]] += w * gr[b];
                }
            }
        }
        let pl = flex_primary_slices(family);
        if let Some(hr) = pl.h.as_ref() {
            for li in 0..hr.len() {
                let ix = hr.start + li;
                for a in 0..pt {
                    let mut w = 0.0;
                    for u in 0..3 {
                        w += ph[[u, ix]] * jt[u][a];
                    }
                    self.h_th[[a, li]] += w;
                }
                for a in 0..pm {
                    let mut w = 0.0;
                    for u in 0..3 {
                        w += ph[[u, ix]] * jm[u][a];
                    }
                    self.h_mh[[a, li]] += w;
                }
                let gw = ph[[3, ix]];
                if gw != 0.0 {
                    for b in 0..gr.len() {
                        self.h_gh[[b, li]] += gw * gr[b];
                    }
                }
            }
            for l in 0..hr.len() {
                for r in 0..hr.len() {
                    self.h_hh[[l, r]] += ph[[hr.start + l, hr.start + r]];
                }
            }
        }
        if let Some(wr) = pl.w.as_ref() {
            for li in 0..wr.len() {
                let ix = wr.start + li;
                for a in 0..pt {
                    let mut w = 0.0;
                    for u in 0..3 {
                        w += ph[[u, ix]] * jt[u][a];
                    }
                    self.h_tw[[a, li]] += w;
                }
                for a in 0..pm {
                    let mut w = 0.0;
                    for u in 0..3 {
                        w += ph[[u, ix]] * jm[u][a];
                    }
                    self.h_mw[[a, li]] += w;
                }
                let gw = ph[[3, ix]];
                if gw != 0.0 {
                    for b in 0..gr.len() {
                        self.h_gw[[b, li]] += gw * gr[b];
                    }
                }
            }
            for l in 0..wr.len() {
                for r in 0..wr.len() {
                    self.h_ww[[l, r]] += ph[[wr.start + l, wr.start + r]];
                }
            }
        }
        if let (Some(hr), Some(wr)) = (pl.h.as_ref(), pl.w.as_ref()) {
            for hl in 0..hr.len() {
                for wl in 0..wr.len() {
                    self.h_hw[[hl, wl]] += ph[[hr.start + hl, wr.start + wl]];
                }
            }
        }
        Ok(())
    }

    /// U^α cross terms (eq 47, terms 1+2)
    pub(crate) fn add_timewiggle_psi_u_cross(
        &mut self,
        family: &SurvivalMarginalSlopeFamily,
        row: usize,
        qg: &SurvivalMarginalSlopeDynamicRow,
        lift: &TimewiggleMarginalPsiRowLift,
        ph: &Array2<f64>,
    ) -> Result<(), String> {
        let jt = [&qg.dq0_time, &qg.dq1_time, &qg.dqd1_time];
        let jm = [&qg.dq0_marginal, &qg.dq1_marginal, &qg.dqd1_marginal];
        let ut = [&lift.u_q0_time, &lift.u_q1_time, &lift.u_qd1_time];
        let um = [
            &lift.u_q0_marginal,
            &lift.u_q1_marginal,
            &lift.u_qd1_marginal,
        ];
        let pt = jt[0].len();
        let pm = jm[0].len();
        for a in 0..pt {
            for b in 0..pt {
                let mut v = 0.0;
                for u in 0..3 {
                    for w in 0..3 {
                        v += ph[[u, w]] * (ut[u][a] * jt[w][b] + jt[u][a] * ut[w][b]);
                    }
                }
                self.h_tt[[a, b]] += v;
            }
        }
        for a in 0..pm {
            for b in 0..pm {
                let mut v = 0.0;
                for u in 0..3 {
                    for w in 0..3 {
                        v += ph[[u, w]] * (um[u][a] * jm[w][b] + jm[u][a] * um[w][b]);
                    }
                }
                self.h_mm[[a, b]] += v;
            }
        }
        for a in 0..pt {
            for b in 0..pm {
                let mut v = 0.0;
                for u in 0..3 {
                    for w in 0..3 {
                        v += ph[[u, w]] * (ut[u][a] * jm[w][b] + jt[u][a] * um[w][b]);
                    }
                }
                self.h_tm[[a, b]] += v;
            }
        }
        let gc = family
            .logslope_layout.coefficient_design()
            .try_row_chunk(row..row + 1)
            .map_err(|e| format!("add_timewiggle_psi_u_cross logslope try_row_chunk: {e}"))?;
        let gr = gc.row(0);
        for a in 0..pt {
            let mut wt = 0.0;
            for u in 0..3 {
                wt += ph[[u, 3]] * ut[u][a];
            }
            if wt != 0.0 {
                for b in 0..gr.len() {
                    self.h_tg[[a, b]] += wt * gr[b];
                }
            }
        }
        for a in 0..pm {
            let mut wt = 0.0;
            for u in 0..3 {
                wt += ph[[u, 3]] * um[u][a];
            }
            if wt != 0.0 {
                for b in 0..gr.len() {
                    self.h_mg[[a, b]] += wt * gr[b];
                }
            }
        }
        let pl = flex_primary_slices(family);
        if let Some(hr) = pl.h.as_ref() {
            for li in 0..hr.len() {
                let ix = hr.start + li;
                for a in 0..pt {
                    let mut w = 0.0;
                    for u in 0..3 {
                        w += ph[[u, ix]] * ut[u][a];
                    }
                    self.h_th[[a, li]] += w;
                }
                for a in 0..pm {
                    let mut w = 0.0;
                    for u in 0..3 {
                        w += ph[[u, ix]] * um[u][a];
                    }
                    self.h_mh[[a, li]] += w;
                }
            }
        }
        if let Some(wr) = pl.w.as_ref() {
            for li in 0..wr.len() {
                let ix = wr.start + li;
                for a in 0..pt {
                    let mut w = 0.0;
                    for u in 0..3 {
                        w += ph[[u, ix]] * ut[u][a];
                    }
                    self.h_tw[[a, li]] += w;
                }
                for a in 0..pm {
                    let mut w = 0.0;
                    for u in 0..3 {
                        w += ph[[u, ix]] * um[u][a];
                    }
                    self.h_mw[[a, li]] += w;
                }
            }
        }
        Ok(())
    }

    /// K^{BC,α} terms (eq 47, term 5)
    pub(crate) fn add_timewiggle_psi_kappa_alpha(
        &mut self,
        family: &SurvivalMarginalSlopeFamily,
        lift: &TimewiggleMarginalPsiRowLift,
        fg: &Array1<f64>,
    ) {
        let tw = family.time_wiggle_range();
        let pb = lift.x_entry_base.len();
        let pm = lift.marginal_row.len();
        let mu = lift.mu;
        let fq0 = fg[0];
        if fq0 != 0.0 {
            let (m3, m2) = (lift.entry_m3, lift.entry_m2);
            for i in 0..pb {
                for j in 0..pb {
                    self.h_tt[[i, j]] +=
                        fq0 * m3 * mu * lift.x_entry_base[i] * lift.x_entry_base[j];
                }
            }
            for loc in 0..tw.len() {
                let ti = tw.start + loc;
                for i in 0..pb {
                    let v = fq0 * lift.entry_basis_d2[loc] * mu * lift.x_entry_base[i];
                    self.h_tt[[i, ti]] += v;
                    self.h_tt[[ti, i]] += v;
                }
            }
            for i in 0..pb {
                for j in 0..pm {
                    self.h_tm[[i, j]] += fq0
                        * (m3 * mu * lift.x_entry_base[i] * lift.marginal_row[j]
                            + m2 * lift.x_entry_base[i] * lift.psi_row[j]);
                }
            }
            for loc in 0..tw.len() {
                let ti = tw.start + loc;
                for j in 0..pm {
                    self.h_tm[[ti, j]] += fq0
                        * (lift.entry_basis_d2[loc] * mu * lift.marginal_row[j]
                            + lift.entry_basis_d1[loc] * lift.psi_row[j]);
                }
            }
            for i in 0..pm {
                for j in 0..pm {
                    self.h_mm[[i, j]] += fq0
                        * (m3 * mu * lift.marginal_row[i] * lift.marginal_row[j]
                            + m2 * (lift.psi_row[i] * lift.marginal_row[j]
                                + lift.marginal_row[i] * lift.psi_row[j]));
                }
            }
        }
        let fq1 = fg[1];
        if fq1 != 0.0 {
            let (m3, m2) = (lift.exit_m3, lift.exit_m2);
            for i in 0..pb {
                for j in 0..pb {
                    self.h_tt[[i, j]] += fq1 * m3 * mu * lift.x_exit_base[i] * lift.x_exit_base[j];
                }
            }
            for loc in 0..tw.len() {
                let ti = tw.start + loc;
                for i in 0..pb {
                    let v = fq1 * lift.exit_basis_d2[loc] * mu * lift.x_exit_base[i];
                    self.h_tt[[i, ti]] += v;
                    self.h_tt[[ti, i]] += v;
                }
            }
            for i in 0..pb {
                for j in 0..pm {
                    self.h_tm[[i, j]] += fq1
                        * (m3 * mu * lift.x_exit_base[i] * lift.marginal_row[j]
                            + m2 * lift.x_exit_base[i] * lift.psi_row[j]);
                }
            }
            for loc in 0..tw.len() {
                let ti = tw.start + loc;
                for j in 0..pm {
                    self.h_tm[[ti, j]] += fq1
                        * (lift.exit_basis_d2[loc] * mu * lift.marginal_row[j]
                            + lift.exit_basis_d1[loc] * lift.psi_row[j]);
                }
            }
            for i in 0..pm {
                for j in 0..pm {
                    self.h_mm[[i, j]] += fq1
                        * (m3 * mu * lift.marginal_row[i] * lift.marginal_row[j]
                            + m2 * (lift.psi_row[i] * lift.marginal_row[j]
                                + lift.marginal_row[i] * lift.psi_row[j]));
                }
            }
        }
        let fqd = fg[2];
        if fqd != 0.0 {
            let (m4, m3, m2, dr) = (lift.exit_m4, lift.exit_m3, lift.exit_m2, lift.d_raw);
            for i in 0..pb {
                for j in 0..pb {
                    self.h_tt[[i, j]] += fqd
                        * (m4 * mu * dr * lift.x_exit_base[i] * lift.x_exit_base[j]
                            + m3 * mu
                                * (lift.x_exit_base[i] * lift.x_deriv_base[j]
                                    + lift.x_deriv_base[i] * lift.x_exit_base[j]));
                }
            }
            for loc in 0..tw.len() {
                let ti = tw.start + loc;
                for i in 0..pb {
                    let v = fqd
                        * (lift.exit_basis_d3[loc] * mu * dr * lift.x_exit_base[i]
                            + lift.exit_basis_d2[loc] * mu * lift.x_deriv_base[i]);
                    self.h_tt[[i, ti]] += v;
                    self.h_tt[[ti, i]] += v;
                }
            }
            for i in 0..pb {
                for j in 0..pm {
                    self.h_tm[[i, j]] += fqd
                        * (m4 * mu * dr * lift.x_exit_base[i] * lift.marginal_row[j]
                            + m3 * dr * lift.x_exit_base[i] * lift.psi_row[j]
                            + m3 * mu * lift.x_deriv_base[i] * lift.marginal_row[j]
                            + m2 * lift.x_deriv_base[i] * lift.psi_row[j]);
                }
            }
            for loc in 0..tw.len() {
                let ti = tw.start + loc;
                for j in 0..pm {
                    self.h_tm[[ti, j]] += fqd
                        * (lift.exit_basis_d3[loc] * mu * dr * lift.marginal_row[j]
                            + lift.exit_basis_d2[loc] * dr * lift.psi_row[j]);
                }
            }
            for i in 0..pm {
                for j in 0..pm {
                    self.h_mm[[i, j]] += fqd
                        * (m4 * mu * dr * lift.marginal_row[i] * lift.marginal_row[j]
                            + m3 * dr
                                * (lift.psi_row[i] * lift.marginal_row[j]
                                    + lift.marginal_row[i] * lift.psi_row[j]));
                }
            }
        }
    }

    /// Σ_a w_a K_a^{BC}: second-pullback weighted by arbitrary vector (eq 47, term 4)
    pub(crate) fn add_second_pullback_weighted(
        &mut self,
        qg: &SurvivalMarginalSlopeDynamicRow,
        w: &Array1<f64>,
    ) {
        let ktt = [&qg.d2q0_time_time, &qg.d2q1_time_time, &qg.d2qd1_time_time];
        let ktm = [
            &qg.d2q0_time_marginal,
            &qg.d2q1_time_marginal,
            &qg.d2qd1_time_marginal,
        ];
        let kmm = [
            &qg.d2q0_marginal_marginal,
            &qg.d2q1_marginal_marginal,
            &qg.d2qd1_marginal_marginal,
        ];
        let pt = ktt[0].nrows();
        let pm = kmm[0].nrows();
        for q in 0..3 {
            let wq = w[q];
            if wq == 0.0 {
                continue;
            }
            for a in 0..pt {
                for b in 0..pt {
                    self.h_tt[[a, b]] += wq * ktt[q][[a, b]];
                }
            }
            for a in 0..pt {
                for b in 0..pm {
                    self.h_tm[[a, b]] += wq * ktm[q][[a, b]];
                }
            }
            for a in 0..pm {
                for b in 0..pm {
                    self.h_mm[[a, b]] += wq * kmm[q][[a, b]];
                }
            }
        }
    }

    /// Rank-1 psi cross with actual Jacobians from q-geometry
    pub(crate) fn add_rank1_psi_cross_with_q_geometry(
        &mut self,
        family: &SurvivalMarginalSlopeFamily,
        row: usize,
        qg: &SurvivalMarginalSlopeDynamicRow,
        psi_block: usize,
        psi_row: &Array1<f64>,
        rp: &Array1<f64>,
    ) -> Result<(), String> {
        let jt = [&qg.dq0_time, &qg.dq1_time, &qg.dqd1_time];
        let jm = [&qg.dq0_marginal, &qg.dq1_marginal, &qg.dqd1_marginal];
        let pt = jt[0].len();
        let pm = jm[0].len();
        for a in 0..pt {
            let mut w = 0.0;
            for q in 0..3 {
                w += rp[q] * jt[q][a];
            }
            if w == 0.0 {
                continue;
            }
            let tgt = match psi_block {
                1 => &mut self.h_tm,
                2 => &mut self.h_tg,
                _ => continue,
            };
            for b in 0..psi_row.len() {
                tgt[[a, b]] += w * psi_row[b];
            }
        }
        for a in 0..pm {
            let mut w = 0.0;
            for q in 0..3 {
                w += rp[q] * jm[q][a];
            }
            if w == 0.0 {
                continue;
            }
            match psi_block {
                1 => {
                    for b in 0..psi_row.len() {
                        self.h_mm[[a, b]] += w * psi_row[b];
                        self.h_mm[[b, a]] += w * psi_row[b];
                    }
                }
                2 => {
                    for b in 0..psi_row.len() {
                        self.h_mg[[a, b]] += w * psi_row[b];
                    }
                }
                _ => {}
            }
        }
        let gc = family
            .logslope_layout.coefficient_design()
            .try_row_chunk(row..row + 1)
            .map_err(|e| {
                format!("add_rank1_psi_cross_with_q_geometry logslope try_row_chunk: {e}")
            })?;
        let gr = gc.row(0);
        let gw = rp[3];
        if gw != 0.0 {
            match psi_block {
                1 => {
                    for a in 0..gr.len() {
                        for b in 0..psi_row.len() {
                            self.h_mg[[b, a]] += gw * gr[a] * psi_row[b];
                        }
                    }
                }
                2 => {
                    for a in 0..gr.len() {
                        for b in 0..psi_row.len() {
                            self.h_gg[[a, b]] += gw * gr[a] * psi_row[b];
                            self.h_gg[[b, a]] += gw * gr[a] * psi_row[b];
                        }
                    }
                }
                _ => {}
            }
        }
        let pl = flex_primary_slices(family);
        if let Some(hr) = pl.h.as_ref() {
            for li in 0..hr.len() {
                let hw = rp[hr.start + li];
                if hw == 0.0 {
                    continue;
                }
                match psi_block {
                    1 => {
                        for b in 0..psi_row.len() {
                            self.h_mh[[b, li]] += hw * psi_row[b];
                        }
                    }
                    2 => {
                        for b in 0..psi_row.len() {
                            self.h_gh[[b, li]] += hw * psi_row[b];
                        }
                    }
                    _ => {}
                }
            }
        }
        if let Some(wr) = pl.w.as_ref() {
            for li in 0..wr.len() {
                let ww = rp[wr.start + li];
                if ww == 0.0 {
                    continue;
                }
                match psi_block {
                    1 => {
                        for b in 0..psi_row.len() {
                            self.h_mw[[b, li]] += ww * psi_row[b];
                        }
                    }
                    2 => {
                        for b in 0..psi_row.len() {
                            self.h_gw[[b, li]] += ww * psi_row[b];
                        }
                    }
                    _ => {}
                }
            }
        }
        Ok(())
    }
}

/// Block-structured HyperOperator for survival marginal-slope psi Hessians.
/// Stores the full 5-block exact joint Hessian layout and performs matvec
/// blockwise instead of materializing dense p×p structure in the outer path.
/// `HyperOperator` view over a populated [`BlockHessianAccumulator`].
///
/// The operator owns the *same* block storage as the dense path — there is
/// no second copy of the 15 cross-block matrices and no second block-layout
/// implementation. The matvec/bilinear here and `to_dense` (which simply
/// scatters via [`BlockHessianAccumulator::to_dense`]) are guaranteed to be
/// consistent because they read one accumulator through one `BlockSlices`.
pub(crate) struct BlockHessianOperator {
    pub(crate) blocks: BlockHessianAccumulator,
    pub(crate) slices: BlockSlices,
}

impl HyperOperator for BlockHessianOperator {
    fn dim(&self) -> usize {
        self.slices.total
    }

    fn mul_vec(&self, v: &Array1<f64>) -> Array1<f64> {
        let mut out = Array1::zeros(self.slices.total);
        self.mul_vec_into(v.view(), out.view_mut());
        out
    }

    fn mul_vec_view(&self, v: ArrayView1<'_, f64>) -> Array1<f64> {
        let mut out = Array1::zeros(self.slices.total);
        self.mul_vec_into(v, out.view_mut());
        out
    }

    fn mul_vec_into(&self, v: ArrayView1<'_, f64>, mut out: ArrayViewMut1<'_, f64>) {
        let b = &self.blocks;
        let slices = &self.slices;
        out.fill(0.0);
        // Full symmetric block matvec, organised row-block by row-block. The
        // block layout and transpose handling live entirely in `block_view`;
        // the `HessBlock::ALL` order fixes the per-output accumulation order so
        // the result is bit-identical to the previous hand-written scatter.
        for row in HessBlock::ALL {
            let Some(r_row) = slices.range_of(row) else {
                continue;
            };
            let mut o_row = out.slice_mut(s![r_row]);
            for col in HessBlock::ALL {
                let Some(r_col) = slices.range_of(col) else {
                    continue;
                };
                o_row += &b.block_view(row, col).dot(&v.slice(s![r_col]));
            }
        }
    }

    fn bilinear(&self, v: &Array1<f64>, u: &Array1<f64>) -> f64 {
        let b = &self.blocks;
        let slices = &self.slices;
        // vᵀ H u over the same single-source block layout as `mul_vec_into`.
        // Diagonal blocks first, then each off-diagonal pair contributes both
        // vₗₒ·Hₗₒ,ₕᵢ·uₕᵢ and vₕᵢ·Hₕᵢ,ₗₒ·uₗₒ — preserving the previous summation
        // order so the scalar is bit-identical.
        let mut total = 0.0;
        for block in HessBlock::ALL {
            let Some(range) = slices.range_of(block) else {
                continue;
            };
            let v_i = v.slice(s![range.clone()]);
            let u_i = u.slice(s![range]);
            total += v_i.dot(&b.block_view(block, block).dot(&u_i));
        }
        BlockHessianAccumulator::for_each_offdiagonal_pair(slices, |lo, r_lo, hi, r_hi| {
            let v_lo = v.slice(s![r_lo.clone()]);
            let u_lo = u.slice(s![r_lo]);
            let v_hi = v.slice(s![r_hi.clone()]);
            let u_hi = u.slice(s![r_hi]);
            total += v_lo.dot(&b.block_view(lo, hi).dot(&u_hi));
            total += v_hi.dot(&b.block_view(hi, lo).dot(&u_lo));
        });
        total
    }

    fn to_dense(&self) -> Array2<f64> {
        // Single block-layout source of truth: the operator densifies through
        // exactly the same scatter the dense path uses, so the dense and
        // operator Hessians are bit-identical by construction.
        self.blocks.to_dense(&self.slices)
    }

    fn is_implicit(&self) -> bool {
        false
    }
}
