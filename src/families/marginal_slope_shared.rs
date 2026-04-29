use crate::custom_family::CustomFamilyBlockPsiDerivative;
use crate::families::cubic_cell_kernel::{self, DenestedPartitionCell, LocalSpanCubic};
use crate::families::jet_partitions::MultiDirJet;
use ndarray::{Array1, Array2, Axis};
use std::ops::Range;

#[inline]
pub fn eval_coeff4_at(coefficients: &[f64; 4], z: f64) -> f64 {
    ((coefficients[3] * z + coefficients[2]) * z + coefficients[1]) * z + coefficients[0]
}

#[inline]
pub fn add_scaled_coeff4(target: &mut [f64; 4], source: &[f64; 4], scale: f64) {
    for j in 0..4 {
        target[j] += scale * source[j];
    }
}

#[inline]
pub fn scale_coeff4(source: [f64; 4], scale: f64) -> [f64; 4] {
    [
        source[0] * scale,
        source[1] * scale,
        source[2] * scale,
        source[3] * scale,
    ]
}

pub fn probit_frailty_scale(gaussian_frailty_sd: Option<f64>) -> f64 {
    let sigma = gaussian_frailty_sd.unwrap_or(0.0);
    if sigma <= 0.0 {
        1.0
    } else {
        crate::families::lognormal_kernel::ProbitFrailtyScaleJet::from_log_sigma(sigma.ln()).s
    }
}

pub(crate) fn probit_frailty_scale_multi_dir_jet(
    gaussian_frailty_sd: Option<f64>,
    missing_sigma_message: &str,
    n_dirs: usize,
    first_masks: &[usize],
    second_masks: &[usize],
) -> Result<MultiDirJet, String> {
    let sigma = gaussian_frailty_sd.ok_or_else(|| missing_sigma_message.to_string())?;
    let jet = crate::families::lognormal_kernel::ProbitFrailtyScaleJet::from_log_sigma(sigma.ln());
    let mut coeffs = Vec::with_capacity(1 + first_masks.len() + second_masks.len());
    coeffs.push((0usize, jet.s));
    coeffs.extend(first_masks.iter().copied().map(|mask| (mask, jet.ds)));
    coeffs.extend(second_masks.iter().copied().map(|mask| (mask, jet.d2s)));
    Ok(MultiDirJet::with_coeffs(n_dirs, &coeffs))
}

fn zero_local_span_cubic() -> LocalSpanCubic {
    LocalSpanCubic {
        left: 0.0,
        right: 1.0,
        c0: 0.0,
        c1: 0.0,
        c2: 0.0,
        c3: 0.0,
    }
}

pub(crate) fn build_denested_partition_cells(
    a: f64,
    b: f64,
    score_warp: Option<&crate::families::bernoulli_marginal_slope::DeviationRuntime>,
    beta_h: Option<&Array1<f64>>,
    link_dev: Option<&crate::families::bernoulli_marginal_slope::DeviationRuntime>,
    beta_w: Option<&Array1<f64>>,
    scale: f64,
) -> Result<Vec<DenestedPartitionCell>, String> {
    let score_breaks = score_warp
        .map(|runtime| runtime.breakpoints().to_vec())
        .unwrap_or_default();
    let link_breaks = link_dev
        .map(|runtime| runtime.breakpoints().to_vec())
        .unwrap_or_default();

    let mut cells = cubic_cell_kernel::build_denested_partition_cells_with_tails(
        a,
        b,
        &score_breaks,
        &link_breaks,
        |z| {
            if let (Some(runtime), Some(beta)) = (score_warp, beta_h) {
                runtime.local_cubic_at(beta, z)
            } else {
                Ok(zero_local_span_cubic())
            }
        },
        |u| {
            if let (Some(runtime), Some(beta)) = (link_dev, beta_w) {
                runtime.local_cubic_at(beta, u)
            } else {
                Ok(zero_local_span_cubic())
            }
        },
    )?;
    if scale != 1.0 {
        for partition_cell in &mut cells {
            partition_cell.cell.c0 *= scale;
            partition_cell.cell.c1 *= scale;
            partition_cell.cell.c2 *= scale;
            partition_cell.cell.c3 *= scale;
        }
    }
    Ok(cells)
}

pub(crate) struct ObservedDenestedCellPartials {
    pub(crate) coeff: [f64; 4],
    pub(crate) dc_da: [f64; 4],
    pub(crate) dc_db: [f64; 4],
    pub(crate) dc_daa: [f64; 4],
    pub(crate) dc_dab: [f64; 4],
    pub(crate) dc_dbb: [f64; 4],
    pub(crate) dc_daaa: [f64; 4],
    pub(crate) dc_daab: [f64; 4],
    pub(crate) dc_dabb: [f64; 4],
    pub(crate) dc_dbbb: [f64; 4],
}

pub(crate) fn observed_denested_cell_partials(
    z_obs: f64,
    a: f64,
    b: f64,
    score_warp: Option<&crate::families::bernoulli_marginal_slope::DeviationRuntime>,
    beta_h: Option<&Array1<f64>>,
    link_dev: Option<&crate::families::bernoulli_marginal_slope::DeviationRuntime>,
    beta_w: Option<&Array1<f64>>,
    scale: f64,
) -> Result<ObservedDenestedCellPartials, String> {
    let zero_score_span = zero_local_span_cubic();
    let zero_link_span = zero_local_span_cubic();
    let u_obs = a + b * z_obs;
    let score_span_obs = if let (Some(runtime), Some(beta_h)) = (score_warp, beta_h) {
        runtime.local_cubic_at(beta_h, z_obs)?
    } else {
        zero_score_span
    };
    let link_span_obs = if let (Some(runtime), Some(beta_w)) = (link_dev, beta_w) {
        runtime.local_cubic_at(beta_w, u_obs)?
    } else {
        zero_link_span
    };
    let coeff = scale_coeff4(
        cubic_cell_kernel::denested_cell_coefficients(score_span_obs, link_span_obs, a, b),
        scale,
    );
    let (dc_da_raw, dc_db_raw) =
        cubic_cell_kernel::denested_cell_coefficient_partials(score_span_obs, link_span_obs, a, b);
    let (dc_daa_raw, dc_dab_raw, dc_dbb_raw) =
        cubic_cell_kernel::denested_cell_second_partials(score_span_obs, link_span_obs, a, b);
    let (dc_daaa, dc_daab, dc_dabb, dc_dbbb) =
        cubic_cell_kernel::denested_cell_third_partials(link_span_obs);
    Ok(ObservedDenestedCellPartials {
        coeff,
        dc_da: scale_coeff4(dc_da_raw, scale),
        dc_db: scale_coeff4(dc_db_raw, scale),
        dc_daa: scale_coeff4(dc_daa_raw, scale),
        dc_dab: scale_coeff4(dc_dab_raw, scale),
        dc_dbb: scale_coeff4(dc_dbb_raw, scale),
        dc_daaa: scale_coeff4(dc_daaa, scale),
        dc_daab: scale_coeff4(dc_daab, scale),
        dc_dabb: scale_coeff4(dc_dabb, scale),
        dc_dbbb: scale_coeff4(dc_dbbb, scale),
    })
}

pub(crate) fn add_two_surface_psi_outer(
    block_i: usize,
    psi_row_i: &Array1<f64>,
    block_j: usize,
    psi_row_j: &Array1<f64>,
    alpha: f64,
    marginal_block: usize,
    logslope_block: usize,
    h_mm: &mut Array2<f64>,
    h_gg: &mut Array2<f64>,
    h_mg: &mut Array2<f64>,
) {
    if alpha == 0.0 {
        return;
    }
    let col_i = psi_row_i.view().insert_axis(Axis(1));
    let row_j = psi_row_j.view().insert_axis(Axis(0));

    if block_i == block_j {
        let col_j = psi_row_j.view().insert_axis(Axis(1));
        let row_i = psi_row_i.view().insert_axis(Axis(0));
        let target = match block_i {
            b if b == marginal_block => h_mm,
            b if b == logslope_block => h_gg,
            _ => return,
        };
        ndarray::linalg::general_mat_mul(alpha, &col_i, &row_j, 1.0, target);
        ndarray::linalg::general_mat_mul(alpha, &col_j, &row_i, 1.0, target);
    } else {
        let (marginal_row, logslope_row) = if block_i == marginal_block {
            (psi_row_i, psi_row_j)
        } else {
            (psi_row_j, psi_row_i)
        };
        let m_col = marginal_row.view().insert_axis(Axis(1));
        let g_row = logslope_row.view().insert_axis(Axis(0));
        ndarray::linalg::general_mat_mul(alpha, &m_col, &g_row, 1.0, h_mg);
    }
}

pub(crate) fn psi_derivative_location(
    derivative_blocks: &[Vec<CustomFamilyBlockPsiDerivative>],
    psi_index: usize,
) -> Option<(usize, usize)> {
    let mut cursor = 0usize;
    for (block_idx, block) in derivative_blocks.iter().enumerate() {
        if psi_index < cursor + block.len() {
            return Some((block_idx, psi_index - cursor));
        }
        cursor += block.len();
    }
    None
}

pub(crate) fn is_sigma_aux_index(
    gaussian_frailty_sd: Option<f64>,
    derivative_blocks: &[Vec<CustomFamilyBlockPsiDerivative>],
    psi_index: usize,
) -> bool {
    let total = derivative_blocks.iter().map(Vec::len).sum::<usize>();
    if gaussian_frailty_sd.is_none() || total == 0 || psi_index != total - 1 {
        return false;
    }
    let Some((block_idx, local_idx)) = psi_derivative_location(derivative_blocks, psi_index) else {
        return false;
    };
    let deriv = &derivative_blocks[block_idx][local_idx];
    deriv.penalty_index.is_none()
        && deriv.x_psi.is_empty()
        && deriv.s_psi.is_empty()
        && deriv.s_psi_components.is_none()
        && deriv.x_psi_psi.is_none()
        && deriv.s_psi_psi.is_none()
}

#[derive(Clone, Copy)]
pub(crate) struct CoeffSupport {
    pub(crate) include_primary: bool,
    pub(crate) include_h: bool,
    pub(crate) include_w: bool,
}

impl CoeffSupport {
    #[inline]
    pub(crate) fn without_primary(self) -> Self {
        Self {
            include_primary: false,
            ..self
        }
    }
}

pub(crate) struct SparsePrimaryCoeffJetView<'a> {
    primary_index: usize,
    h_range: Option<Range<usize>>,
    w_range: Option<Range<usize>>,
    pub(crate) first: &'a [[f64; 4]],
    pub(crate) a_first: &'a [[f64; 4]],
    pub(crate) b_first: &'a [[f64; 4]],
    pub(crate) aa_first: &'a [[f64; 4]],
    pub(crate) ab_first: &'a [[f64; 4]],
    pub(crate) bb_first: &'a [[f64; 4]],
    pub(crate) aaa_first: &'a [[f64; 4]],
    pub(crate) aab_first: &'a [[f64; 4]],
    pub(crate) abb_first: &'a [[f64; 4]],
    pub(crate) bbb_first: &'a [[f64; 4]],
}

impl<'a> SparsePrimaryCoeffJetView<'a> {
    pub(crate) fn new(
        primary_index: usize,
        h_range: Option<&Range<usize>>,
        w_range: Option<&Range<usize>>,
        first: &'a [[f64; 4]],
        a_first: &'a [[f64; 4]],
        b_first: &'a [[f64; 4]],
        aa_first: &'a [[f64; 4]],
        ab_first: &'a [[f64; 4]],
        bb_first: &'a [[f64; 4]],
        aaa_first: &'a [[f64; 4]],
        aab_first: &'a [[f64; 4]],
        abb_first: &'a [[f64; 4]],
        bbb_first: &'a [[f64; 4]],
    ) -> Self {
        Self {
            primary_index,
            h_range: h_range.cloned(),
            w_range: w_range.cloned(),
            first,
            a_first,
            b_first,
            aa_first,
            ab_first,
            bb_first,
            aaa_first,
            aab_first,
            abb_first,
            bbb_first,
        }
    }

    #[inline]
    fn in_h_range(&self, idx: usize) -> bool {
        self.h_range
            .as_ref()
            .map(|range| range.contains(&idx))
            .unwrap_or(false)
    }

    #[inline]
    fn in_w_range(&self, idx: usize) -> bool {
        self.w_range
            .as_ref()
            .map(|range| range.contains(&idx))
            .unwrap_or(false)
    }

    #[inline]
    fn param_supported(&self, idx: usize, support: CoeffSupport) -> bool {
        (support.include_primary && idx == self.primary_index)
            || (support.include_h && self.in_h_range(idx))
            || (support.include_w && self.in_w_range(idx))
    }

    pub(crate) fn directional_family(
        &self,
        family: &[[f64; 4]],
        dir: &Array1<f64>,
        support: CoeffSupport,
    ) -> [f64; 4] {
        let mut out = [0.0; 4];
        if support.include_primary {
            add_scaled_coeff4(
                &mut out,
                &family[self.primary_index],
                dir[self.primary_index],
            );
        }
        if support.include_h {
            if let Some(h_range) = self.h_range.as_ref() {
                for idx in h_range.clone() {
                    add_scaled_coeff4(&mut out, &family[idx], dir[idx]);
                }
            }
        }
        if support.include_w {
            if let Some(w_range) = self.w_range.as_ref() {
                for idx in w_range.clone() {
                    add_scaled_coeff4(&mut out, &family[idx], dir[idx]);
                }
            }
        }
        out
    }

    pub(crate) fn mixed_directional_from_b_family(
        &self,
        family: &[[f64; 4]],
        dir_u: &Array1<f64>,
        dir_v: &Array1<f64>,
        support: CoeffSupport,
    ) -> [f64; 4] {
        let mut out = [0.0; 4];
        let dir_u_primary = dir_u[self.primary_index];
        let dir_v_primary = dir_v[self.primary_index];
        if support.include_primary {
            add_scaled_coeff4(
                &mut out,
                &family[self.primary_index],
                dir_u_primary * dir_v_primary,
            );
        }
        if support.include_h {
            if let Some(h_range) = self.h_range.as_ref() {
                for idx in h_range.clone() {
                    add_scaled_coeff4(
                        &mut out,
                        &family[idx],
                        dir_u_primary * dir_v[idx] + dir_v_primary * dir_u[idx],
                    );
                }
            }
        }
        if support.include_w {
            if let Some(w_range) = self.w_range.as_ref() {
                for idx in w_range.clone() {
                    add_scaled_coeff4(
                        &mut out,
                        &family[idx],
                        dir_u_primary * dir_v[idx] + dir_v_primary * dir_u[idx],
                    );
                }
            }
        }
        out
    }

    pub(crate) fn param_directional_from_b_family(
        &self,
        family: &[[f64; 4]],
        param: usize,
        dir: &Array1<f64>,
        support: CoeffSupport,
    ) -> [f64; 4] {
        if param == self.primary_index {
            return self.directional_family(family, dir, support);
        }
        if self.param_supported(param, support.without_primary()) {
            let mut out = [0.0; 4];
            add_scaled_coeff4(&mut out, &family[param], dir[self.primary_index]);
            return out;
        }
        [0.0; 4]
    }

    pub(crate) fn param_mixed_from_bb_family(
        &self,
        family: &[[f64; 4]],
        param: usize,
        dir_u: &Array1<f64>,
        dir_v: &Array1<f64>,
        support: CoeffSupport,
    ) -> [f64; 4] {
        if param == self.primary_index {
            return self.mixed_directional_from_b_family(family, dir_u, dir_v, support);
        }
        if self.param_supported(param, support.without_primary()) {
            let mut out = [0.0; 4];
            add_scaled_coeff4(
                &mut out,
                &family[param],
                dir_u[self.primary_index] * dir_v[self.primary_index],
            );
            return out;
        }
        [0.0; 4]
    }

    pub(crate) fn pair_from_b_family(
        &self,
        family: &[[f64; 4]],
        u: usize,
        v: usize,
        support: CoeffSupport,
    ) -> [f64; 4] {
        if u == self.primary_index && v == self.primary_index {
            if support.include_primary {
                return family[self.primary_index];
            }
            return [0.0; 4];
        }
        if u == self.primary_index && self.param_supported(v, support.without_primary()) {
            return family[v];
        }
        if v == self.primary_index && self.param_supported(u, support.without_primary()) {
            return family[u];
        }
        [0.0; 4]
    }

    pub(crate) fn pair_directional_from_bb_family(
        &self,
        family: &[[f64; 4]],
        u: usize,
        v: usize,
        dir: &Array1<f64>,
        support: CoeffSupport,
    ) -> [f64; 4] {
        if u == self.primary_index && v == self.primary_index {
            return self.directional_family(family, dir, support);
        }
        if u == self.primary_index && self.param_supported(v, support.without_primary()) {
            let mut out = [0.0; 4];
            add_scaled_coeff4(&mut out, &family[v], dir[self.primary_index]);
            return out;
        }
        if v == self.primary_index && self.param_supported(u, support.without_primary()) {
            let mut out = [0.0; 4];
            add_scaled_coeff4(&mut out, &family[u], dir[self.primary_index]);
            return out;
        }
        [0.0; 4]
    }

    pub(crate) fn pair_mixed_from_bbb_family(
        &self,
        family: &[[f64; 4]],
        u: usize,
        v: usize,
        dir_u: &Array1<f64>,
        dir_v: &Array1<f64>,
        support: CoeffSupport,
    ) -> [f64; 4] {
        if u == self.primary_index && v == self.primary_index {
            return self.mixed_directional_from_b_family(family, dir_u, dir_v, support);
        }
        if u == self.primary_index && self.param_supported(v, support.without_primary()) {
            let mut out = [0.0; 4];
            add_scaled_coeff4(
                &mut out,
                &family[v],
                dir_u[self.primary_index] * dir_v[self.primary_index],
            );
            return out;
        }
        if v == self.primary_index && self.param_supported(u, support.without_primary()) {
            let mut out = [0.0; 4];
            add_scaled_coeff4(
                &mut out,
                &family[u],
                dir_u[self.primary_index] * dir_v[self.primary_index],
            );
            return out;
        }
        [0.0; 4]
    }
}
