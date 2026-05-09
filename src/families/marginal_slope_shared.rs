use crate::custom_family::CustomFamilyBlockPsiDerivative;
use crate::families::cubic_cell_kernel::{self, DenestedPartitionCell, LocalSpanCubic};
use crate::families::jet_partitions::MultiDirJet;
use ndarray::{Array1, Array2, Axis};
use std::ops::Range;
use std::sync::Arc;

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

pub(crate) fn add_optional_vector(left: &mut Option<Array1<f64>>, right: &Option<Array1<f64>>) {
    if let (Some(left), Some(right)) = (left.as_mut(), right.as_ref()) {
        *left += right;
    }
}

pub(crate) fn add_optional_matrix(left: &mut Option<Array2<f64>>, right: &Option<Array2<f64>>) {
    if let (Some(left), Some(right)) = (left.as_mut(), right.as_ref()) {
        *left += right;
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

// ---------------------------------------------------------------------------
// Outer-only stratified row subsample (Phase 1 scaffolding).
//
// The biobank-scale outer-loop score/gradient passes do O(n) work per outer
// evaluation, which dominates wall-clock once n grows past ~10^5. To keep
// outer-loop iterations tractable while leaving the inner PIRLS solve and the
// final covariance assembly untouched, outer-only hot loops can be redirected
// to iterate over a small stratified subsample with a constant rescaling
// factor, sampled once per fit and shared via `Arc`. The subsample is
// stratified by event/outcome × z-deciles (≤ 200 strata) so that the rescaled
// estimator inherits the same support coverage as the full-data estimator.
//
// This module defines only the types and helpers; Phase 2 wires them into
// per-row hot loops. Default state (`outer_score_subsample = None`) keeps the
// legacy full-data behavior bit-for-bit.

/// Stratified row index subsample shared across outer-loop evaluations.
///
/// `mask` is sorted, deduplicated, and never empty in practice (enforced by
/// `build_outer_score_subsample`).
///
/// Per-row inverse-inclusion weights `w_i = N_h / k_h` (where `h` is the row's
/// stratum) are stored alongside the mask in `rows`. The Horvitz–Thompson
/// estimator for any linear-in-row functional T = Σ_i f_i is
///   T̂ = Σ_{i ∈ mask} w_i · f_i,
/// which is unbiased even when per-stratum sampling fractions differ
/// (the `ceil(k * N_h / n).max(1)` rule in the stratified builder makes
/// rare strata oversample relative to the bulk, so a single global rescale
/// `n_full / |mask|` is biased in those strata).
///
/// `weight_scale` is retained as a *diagnostic* (mean of `w_i` across the
/// mask). It equals the legacy `n_full / |mask|` when all rows share a
/// uniform weight (the common case for caller-supplied masks via
/// [`OuterScoreSubsample::new`]); it can drift from that value under the
/// stratified builder's rare-stratum boost. It is not the per-row scaling
/// factor — consumers must read `rows[i].weight` for HT correctness.
#[derive(Debug, Clone)]
pub struct OuterScoreSubsample {
    pub mask: Arc<Vec<usize>>,
    pub rows: Arc<Vec<WeightedOuterRow>>,
    pub n_full: usize,
    pub weight_scale: f64,
    pub seed: u64,
}

impl OuterScoreSubsample {
    /// Wrap a precomputed mask with the legacy uniform `n_full / m` weight
    /// per row. The caller is responsible for sortedness and uniqueness;
    /// `build_outer_score_subsample` is the canonical (per-stratum HT)
    /// builder.
    pub fn new(mask: Vec<usize>, n_full: usize, seed: u64) -> Self {
        let m = mask.len();
        let w = if m == 0 {
            1.0
        } else {
            n_full as f64 / m as f64
        };
        Self::with_uniform_weight(mask, n_full, seed, w)
    }

    /// Wrap a precomputed mask with an explicit uniform per-row weight.
    /// Useful for tests that need the unrescaled (`weight = 1.0`) sum over a
    /// custom mask, and for callers that already know the desired
    /// rescaling factor and don't want the constructor to derive it from
    /// `n_full / |mask|`.
    pub fn with_uniform_weight(mask: Vec<usize>, n_full: usize, seed: u64, weight: f64) -> Self {
        let rows: Vec<WeightedOuterRow> = mask
            .iter()
            .map(|&index| WeightedOuterRow {
                index,
                weight,
                stratum: 0,
            })
            .collect();
        let weight_scale = if rows.is_empty() { 1.0 } else { weight };
        Self {
            mask: Arc::new(mask),
            rows: Arc::new(rows),
            n_full,
            weight_scale,
            seed,
        }
    }

    /// Wrap a vector of `(index, weight, stratum)` triples. The mask is
    /// derived as the sorted/dedup'd index list. Used by the stratified
    /// builder to install per-row HT weights.
    pub fn from_weighted_rows(
        mut rows: Vec<WeightedOuterRow>,
        n_full: usize,
        seed: u64,
    ) -> Self {
        rows.sort_by_key(|r| r.index);
        rows.dedup_by_key(|r| r.index);
        let mask: Vec<usize> = rows.iter().map(|r| r.index).collect();
        let weight_scale = if rows.is_empty() {
            1.0
        } else {
            rows.iter().map(|r| r.weight).sum::<f64>() / rows.len() as f64
        };
        Self {
            mask: Arc::new(mask),
            rows: Arc::new(rows),
            n_full,
            weight_scale,
            seed,
        }
    }

    #[inline]
    pub fn len(&self) -> usize {
        self.mask.len()
    }

    #[inline]
    pub fn is_empty(&self) -> bool {
        self.mask.is_empty()
    }

    /// True when at least two retained rows have different per-row weights.
    /// Consumers that previously applied a single post-sum scalar must
    /// switch to per-row weighting whenever this returns true.
    pub fn has_variable_weights(&self) -> bool {
        let mut iter = self.rows.iter();
        let Some(first) = iter.next() else {
            return false;
        };
        iter.any(|r| (r.weight - first.weight).abs() > 0.0)
    }
}

#[derive(Debug, Clone, Copy)]
pub struct WeightedOuterRow {
    pub index: usize,
    pub weight: f64,
    /// Stratum identifier the row was drawn from. Pure diagnostic — consumers
    /// must use `weight` for any aggregation.
    pub stratum: u32,
}

/// Splitmix64: deterministic single-u64 expansion. Local copy so this module
/// stays self-contained; matches the constants used elsewhere in the crate.
#[inline]
fn splitmix64(state: &mut u64) -> u64 {
    *state = state.wrapping_add(0x9E3779B97F4A7C15);
    let mut z = *state;
    z = (z ^ (z >> 30)).wrapping_mul(0xBF58476D1CE4E5B9);
    z = (z ^ (z >> 27)).wrapping_mul(0x94D049BB133111EB);
    z ^ (z >> 31)
}

/// Build a deterministic stratified row subsample of size ≥ `k` from
/// `(z, stratum_secondary)`.
///
/// Stratification: 100 z-deciles × distinct values of `stratum_secondary`
/// (typically the {0,1} event/outcome indicator, giving ≤ 200 strata).
/// Each non-empty stratum contributes `ceil(k * stratum_size / n)` rows
/// drawn via a splitmix64-keyed Fisher-Yates partial shuffle so the result
/// is reproducible from `(seed, stratum_id)`.
///
/// The returned mask is sorted, deduplicated, and never empty when `n > 0`.
/// Per-row weights `w_i = N_h / k_h` (Horvitz-Thompson inverse-inclusion
/// weights for the stratum the row came from) are assigned to
/// `OuterScoreSubsample::rows`, and `weight_scale` is reported as the mean
/// of those weights for diagnostics only.
///
/// Panics if `z.len() != stratum_secondary.len()`.
pub fn build_outer_score_subsample(
    z: &[f64],
    stratum_secondary: &[u8],
    k: usize,
    seed: u64,
) -> OuterScoreSubsample {
    let n = z.len();
    assert_eq!(
        n,
        stratum_secondary.len(),
        "build_outer_score_subsample: z and stratum_secondary must have equal length",
    );

    if n == 0 {
        return OuterScoreSubsample::with_uniform_weight(Vec::new(), 0, seed, 1.0);
    }

    // If the requested subsample covers the full dataset (or more), short-
    // circuit to the full row set with weight 1.0 — this is a no-op
    // relative to the legacy full-data path.
    if k >= n {
        let mask: Vec<usize> = (0..n).collect();
        return OuterScoreSubsample::with_uniform_weight(mask, n, seed, 1.0);
    }

    // Q = 100 z-deciles. Sort indices by z and split into Q ~equal chunks.
    const Q: usize = 100;
    let mut z_order: Vec<usize> = (0..n).collect();
    z_order.sort_by(|&a, &b| z[a].partial_cmp(&z[b]).unwrap_or(std::cmp::Ordering::Equal));
    // decile[i] = bin index in 0..Q for row i
    let mut decile = vec![0u16; n];
    for (rank, &row) in z_order.iter().enumerate() {
        // Map rank in 0..n to bin in 0..Q. Using floor((rank * Q) / n)
        // keeps bin sizes within ±1 row of n/Q.
        let bin = (rank * Q) / n;
        let bin = bin.min(Q - 1);
        decile[row] = bin as u16;
    }

    // Distinct secondary values (the canonical use case is {0,1}, but the
    // general u8 alphabet is supported transparently).
    let mut distinct_secondary: Vec<u8> = stratum_secondary.to_vec();
    distinct_secondary.sort_unstable();
    distinct_secondary.dedup();
    // stratum index = secondary_rank * Q + decile, where secondary_rank is
    // the position of the row's secondary value in `distinct_secondary`.
    let mut secondary_rank = vec![0u16; 256];
    for (rank, &val) in distinct_secondary.iter().enumerate() {
        secondary_rank[val as usize] = rank as u16;
    }
    let n_strata = distinct_secondary.len() * Q;

    // Bucket rows by stratum.
    let mut strata: Vec<Vec<usize>> = vec![Vec::new(); n_strata];
    for i in 0..n {
        let s = secondary_rank[stratum_secondary[i] as usize] as usize * Q + decile[i] as usize;
        strata[s].push(i);
    }

    // For each non-empty stratum, draw ceil(k * stratum_size / n) rows and
    // tag each retained row with its HT weight w_h = N_h / k_h.
    let mut picked: Vec<WeightedOuterRow> = Vec::with_capacity(k + n_strata);
    for (stratum_id, rows) in strata.iter().enumerate() {
        if rows.is_empty() {
            continue;
        }
        let take = ((k as u128 * rows.len() as u128 + n as u128 - 1) / n as u128) as usize;
        let take = take.max(1).min(rows.len());
        // HT inverse-inclusion weight for this stratum: w_h = N_h / k_h.
        // Identical for every row drawn from `stratum_id`.
        let w_h = rows.len() as f64 / take as f64;
        let stratum_tag = stratum_id as u32;

        // Deterministic key from (seed, stratum_id).
        let mut state = seed ^ (stratum_id as u64).wrapping_mul(0x9E3779B97F4A7C15);
        // Mix once so even seed=0, stratum_id=0 produces a non-trivial state.
        let _ = splitmix64(&mut state);

        if take == rows.len() {
            for &index in rows.iter() {
                picked.push(WeightedOuterRow {
                    index,
                    weight: w_h,
                    stratum: stratum_tag,
                });
            }
        } else {
            // Fisher-Yates partial shuffle: produce `take` distinct rows.
            let mut buf: Vec<usize> = rows.clone();
            let m = buf.len();
            for i in 0..take {
                let r = splitmix64(&mut state);
                let j = i + (r as usize) % (m - i);
                buf.swap(i, j);
            }
            for &index in &buf[..take] {
                picked.push(WeightedOuterRow {
                    index,
                    weight: w_h,
                    stratum: stratum_tag,
                });
            }
        }
    }

    // `from_weighted_rows` sorts + dedups by index. Strata are disjoint by
    // construction so dedup is a no-op, but we route through the constructor
    // so the OuterScoreSubsample contract stays in one place.
    OuterScoreSubsample::from_weighted_rows(picked, n, seed)
}

// ---------------------------------------------------------------------------
// Outer-row iteration helpers.
//
// These wrap the choice between "iterate 0..n" (default) and "iterate
// `subsample.mask`" so per-row hot loops in Phase 2 can call a single helper
// rather than branch by hand. We expose both an enum that callers can match
// on directly (cheap path: a `Range` plus a `Arc<Vec<usize>>`) and a
// `Vec<usize>`-returning convenience that satisfies
// `IntoParallelIterator<Item = usize>` via `Vec`'s rayon impl.

/// Row-index iteration choice for outer-only score/gradient passes.
#[derive(Debug, Clone)]
pub enum OuterRowIter {
    /// Full data: iterate `0..n`.
    All { n: usize },
    /// Subsample: iterate `subsample.mask`.
    Subset { mask: Arc<Vec<usize>> },
}

impl OuterRowIter {
    /// Number of rows this iterator covers.
    #[inline]
    pub fn len(&self) -> usize {
        match self {
            OuterRowIter::All { n } => *n,
            OuterRowIter::Subset { mask } => mask.len(),
        }
    }

    #[inline]
    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    /// Materialize the row indices as a `Vec<usize>`. Useful for callers
    /// that want a `IntoParallelIterator<Item = usize>` source — `Vec<usize>`
    /// satisfies that trait via rayon's blanket impl.
    pub fn to_vec(&self) -> Vec<usize> {
        match self {
            OuterRowIter::All { n } => (0..*n).collect(),
            OuterRowIter::Subset { mask } => mask.as_ref().clone(),
        }
    }
}

/// Choose the row-iteration strategy for an outer-only pass. When
/// `opts.outer_score_subsample` is `Some`, returns the subsample mask;
/// otherwise returns the full range `0..n`.
///
/// Callers using this helper iterate over row indices and must additionally
/// consult [`outer_row_weights_by_index`] (or [`outer_weighted_rows`]) for
/// per-row HT weights — a single global rescale is biased under stratified
/// sampling and is no longer exposed.
pub fn outer_row_indices(
    opts: &crate::custom_family::BlockwiseFitOptions,
    n: usize,
) -> OuterRowIter {
    match opts.outer_score_subsample.as_ref() {
        Some(s) => OuterRowIter::Subset {
            mask: Arc::clone(&s.mask),
        },
        None => OuterRowIter::All { n },
    }
}

/// Diagnostic: mean per-row HT weight across the subsample (or 1.0 in the
/// no-subsample / full-data path). Equals the legacy `n_full / |mask|`
/// global rescale when all retained rows share a uniform weight; differs
/// from it under the stratified builder's rare-stratum boost.
///
/// Consumers must not use this as a per-row scaling factor when
/// `OuterScoreSubsample::has_variable_weights()` is true — the unbiased
/// estimator multiplies each row by `outer_row_weights_by_index(opts, n)[i]`,
/// not by this scalar. Retained for backward-compatible tests over uniformly
/// weighted masks (e.g. callers that build `OuterScoreSubsample::new`).
#[inline]
pub fn outer_score_scale(opts: &crate::custom_family::BlockwiseFitOptions, _n: usize) -> f64 {
    match opts.outer_score_subsample.as_ref() {
        Some(s) => s.weight_scale,
        None => 1.0,
    }
}

/// Per-row HT-weighted iteration: returns one `WeightedOuterRow` per
/// retained row when a subsample is active; otherwise returns
/// `(index, weight = 1.0, stratum = 0)` for every row in `0..n`.
pub fn outer_weighted_rows(
    opts: &crate::custom_family::BlockwiseFitOptions,
    n: usize,
) -> Vec<WeightedOuterRow> {
    match opts.outer_score_subsample.as_ref() {
        Some(s) => s.rows.as_ref().clone(),
        None => (0..n)
            .map(|index| WeightedOuterRow {
                index,
                weight: 1.0,
                stratum: 0,
            })
            .collect(),
    }
}

/// Dense-by-row HT weights of length `n`. Masked rows carry their HT
/// weight; unmasked rows default to 1.0 so that callers who index by row
/// regardless of subsampling still get a valid scalar (the consumer is
/// expected to iterate only over `outer_row_indices`).
pub fn outer_row_weights_by_index(
    opts: &crate::custom_family::BlockwiseFitOptions,
    n: usize,
) -> Vec<f64> {
    match opts.outer_score_subsample.as_ref() {
        Some(s) => {
            let mut weights = vec![1.0; n];
            for r in s.rows.iter() {
                if r.index < n {
                    weights[r.index] = r.weight;
                }
            }
            weights
        }
        None => vec![1.0; n],
    }
}

/// Phase-2 placeholder. The intent is to debug-assert (in callers that are
/// known to be _outer-only_ score/gradient passes) that no inner-PIRLS
/// codepath consults the subsample. For Phase 1 this is a no-op so the
/// signature can be threaded into Phase-2 hot loops without touching call
/// sites later.
#[inline]
pub fn assert_outer_only(_opts: &crate::custom_family::BlockwiseFitOptions, _context: &str) {
    // intentionally empty in Phase 1
}

/// Deterministic-order parallel reduction over a row-index slice.
///
/// Splits `rows` into a fixed number of contiguous chunks
/// (`TARGET_CHUNK_COUNT`), processes each chunk sequentially in parallel
/// via `process_row`, and combines the per-chunk accumulators in
/// chunk-index order via `combine` on the calling thread. The chunk size
/// is a pure function of `rows.len()`, so the reduction tree is fixed
/// across calls regardless of rayon's thread-pool size or work-stealing
/// decisions.
///
/// `try_fold/try_reduce` over `rows.into_par_iter()` does **not** have
/// this property: rayon's adaptive splitter sets chunk boundaries based
/// on `current_num_threads()` and runtime work-stealing, so two calls
/// with identical inputs can return ULP-different floating-point sums
/// when the rayon pool has different concurrent activity. Tests that
/// compare two reductions and rely on bit-for-bit equality flake under
/// load with that pattern. This primitive is the per-family deterministic
/// row-reduction that the bernoulli / survival sigma-ψ paths funnel
/// through; their per-row contributions are the dominant non-deterministic
/// source in the marginal-slope outer-loop score / Hessian sums.
pub(crate) fn chunked_row_reduction<Item, Acc, Init, Process, Combine>(
    rows: &[Item],
    init: Init,
    process_row: Process,
    mut combine: Combine,
) -> Result<Acc, String>
where
    Item: Sync + Copy,
    Acc: Send,
    Init: Fn() -> Acc + Sync,
    Process: Fn(Item, &mut Acc) -> Result<(), String> + Sync,
    Combine: FnMut(&mut Acc, Acc),
{
    use rayon::iter::{IntoParallelIterator, ParallelIterator};
    const TARGET_CHUNK_COUNT: usize = 32;
    let n = rows.len();
    if n == 0 {
        return Ok(init());
    }
    let chunk_size = n.div_ceil(TARGET_CHUNK_COUNT).max(1);
    let n_chunks = n.div_ceil(chunk_size);
    // `(0..n_chunks).into_par_iter()` is `IndexedParallelIterator`, so the
    // `.collect::<Vec<_>>()` below preserves chunk-index order regardless
    // of work-stealing. That ordered `Vec` is what makes the sequential
    // `combine` deterministic.
    let chunk_states: Vec<Acc> = (0..n_chunks)
        .into_par_iter()
        .map(|chunk_idx| -> Result<Acc, String> {
            let start = chunk_idx * chunk_size;
            let end = (start + chunk_size).min(n);
            let mut acc = init();
            for &item in &rows[start..end] {
                process_row(item, &mut acc)?;
            }
            Ok(acc)
        })
        .collect::<Result<Vec<Acc>, String>>()?;
    let mut total = init();
    for chunk in chunk_states {
        combine(&mut total, chunk);
    }
    Ok(total)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn subsample_full_n_equals_no_subsample() {
        // mask = (0..n) — the all-rows subsample should have weight_scale 1.0
        // and outer_row_indices should yield the same sorted set in both
        // Some(mask=full) and None modes.
        let n: usize = 1024;
        let z: Vec<f64> = (0..n).map(|i| i as f64).collect();
        let secondary: Vec<u8> = (0..n).map(|i| (i % 2) as u8).collect();
        let s = build_outer_score_subsample(&z, &secondary, n, 0xDEADBEEF);
        assert_eq!(s.len(), n);
        assert!((s.weight_scale - 1.0).abs() < 1e-12);

        let mut full = crate::custom_family::BlockwiseFitOptions::default();
        let from_none = outer_row_indices(&full, n).to_vec();
        full.outer_score_subsample = Some(Arc::new(s));
        let from_some = outer_row_indices(&full, n).to_vec();

        let mut a = from_none.clone();
        let mut b = from_some.clone();
        a.sort_unstable();
        b.sort_unstable();
        assert_eq!(a, b);
        assert_eq!(a, (0..n).collect::<Vec<_>>());
    }

    #[test]
    fn stratification_covers_all_strata() {
        // Synthetic with 2 secondary classes × 100 z-deciles. Every
        // non-empty (secondary, decile) stratum must contribute ≥ 1 row.
        let n: usize = 20_000;
        let z: Vec<f64> = (0..n).map(|i| (i as f64) * 0.001).collect();
        let secondary: Vec<u8> = (0..n).map(|i| (i % 2) as u8).collect();
        let k = 2_000;
        let s = build_outer_score_subsample(&z, &secondary, k, 12345);
        assert!(s.len() >= k, "subsample size {} < k {}", s.len(), k);

        // Recompute deciles to label rows.
        let mut order: Vec<usize> = (0..n).collect();
        order.sort_by(|&a, &b| z[a].partial_cmp(&z[b]).unwrap());
        let mut decile = vec![0usize; n];
        for (rank, &row) in order.iter().enumerate() {
            decile[row] = ((rank * 100) / n).min(99);
        }
        // For each (sec, dec), is there at least one row in mask?
        let mut covered = vec![false; 200];
        for &row in s.mask.iter() {
            let stratum = secondary[row] as usize * 100 + decile[row];
            covered[stratum] = true;
        }
        // All 200 strata are non-empty in this synthetic, so all must be
        // covered.
        for (stratum, &c) in covered.iter().enumerate() {
            assert!(c, "stratum {} uncovered", stratum);
        }
    }

    #[test]
    fn deterministic_seed() {
        // Same inputs + seed must produce identical masks; different seeds
        // produce different masks (with overwhelming probability for these
        // sizes).
        let n: usize = 5_000;
        let z: Vec<f64> = (0..n).map(|i| (i as f64).sin()).collect();
        let secondary: Vec<u8> = (0..n).map(|i| (i % 2) as u8).collect();
        let k = 800;
        let a = build_outer_score_subsample(&z, &secondary, k, 0xABCDEF);
        let b = build_outer_score_subsample(&z, &secondary, k, 0xABCDEF);
        let c = build_outer_score_subsample(&z, &secondary, k, 0xFEDCBA);
        assert_eq!(a.mask.as_ref(), b.mask.as_ref());
        assert_ne!(a.mask.as_ref(), c.mask.as_ref());
    }

    #[test]
    fn weight_scale_correct() {
        // n=10000, k=2000 → weight_scale ≈ 5.0 (allow small overshoot from
        // ceil(k * stratum_size / n) summed across strata).
        let n: usize = 10_000;
        let z: Vec<f64> = (0..n).map(|i| i as f64).collect();
        let secondary: Vec<u8> = (0..n).map(|i| (i % 2) as u8).collect();
        let k = 2_000;
        let s = build_outer_score_subsample(&z, &secondary, k, 7);
        assert!(s.len() >= k);
        // overshoot bounded by number of strata (one extra row per stratum
        // from the ceil); for 2 × 100 = 200 strata, overshoot ≤ 200.
        assert!(
            s.len() <= k + 200,
            "subsample {} much larger than expected",
            s.len()
        );
        let scale = s.weight_scale;
        // expected ≈ 5.0; allow ±10% for the ceiling overshoot.
        assert!(
            (scale - 5.0).abs() < 0.5,
            "weight_scale {} not near 5.0",
            scale
        );
    }

}
