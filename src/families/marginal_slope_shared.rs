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
/// `build_outer_score_subsample`). `weight_scale = n_full / mask.len()` is the
/// constant rescaling factor that outer-only passes apply per-row so that
/// linear-in-row sums approximate full-data sums in expectation.
#[derive(Debug, Clone)]
pub struct OuterScoreSubsample {
    pub mask: Arc<Vec<usize>>,
    pub n_full: usize,
    pub weight_scale: f64,
    pub seed: u64,
}

impl OuterScoreSubsample {
    /// Wrap a precomputed mask. The caller is responsible for sortedness and
    /// uniqueness; `build_outer_score_subsample` is the canonical builder.
    pub fn new(mask: Vec<usize>, n_full: usize, seed: u64) -> Self {
        let m = mask.len();
        let weight_scale = if m == 0 {
            1.0
        } else {
            n_full as f64 / m as f64
        };
        Self {
            mask: Arc::new(mask),
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
/// `weight_scale = n_full / mask.len()` (computed by `OuterScoreSubsample::new`).
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
        return OuterScoreSubsample::new(Vec::new(), 0, seed);
    }

    // If the requested subsample covers the full dataset (or more), short-
    // circuit to the full row set. weight_scale = 1.0 and this becomes a
    // no-op compared to the legacy full-data path.
    if k >= n {
        let mask: Vec<usize> = (0..n).collect();
        return OuterScoreSubsample::new(mask, n, seed);
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

    // For each non-empty stratum, draw ceil(k * stratum_size / n) rows.
    let mut picked: Vec<usize> = Vec::with_capacity(k + n_strata);
    for (stratum_id, rows) in strata.iter().enumerate() {
        if rows.is_empty() {
            continue;
        }
        let take = ((k as u128 * rows.len() as u128 + n as u128 - 1) / n as u128) as usize;
        let take = take.max(1).min(rows.len());

        // Deterministic key from (seed, stratum_id).
        let mut state = seed ^ (stratum_id as u64).wrapping_mul(0x9E3779B97F4A7C15);
        // Mix once so even seed=0, stratum_id=0 produces a non-trivial state.
        let _ = splitmix64(&mut state);

        if take == rows.len() {
            picked.extend_from_slice(rows);
        } else {
            // Fisher-Yates partial shuffle: produce `take` distinct rows.
            let mut buf: Vec<usize> = rows.clone();
            let m = buf.len();
            for i in 0..take {
                let r = splitmix64(&mut state);
                let j = i + (r as usize) % (m - i);
                buf.swap(i, j);
            }
            picked.extend_from_slice(&buf[..take]);
        }
    }

    // Sort + dedup. Strata are disjoint by construction so dedup is a no-op,
    // but we sort regardless to satisfy the OuterScoreSubsample contract.
    picked.sort_unstable();
    picked.dedup();

    OuterScoreSubsample::new(picked, n, seed)
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

/// Per-row rescaling factor for outer-only sums. Returns
/// `subsample.weight_scale` when a subsample is active and `1.0` otherwise.
#[inline]
pub fn outer_score_scale(opts: &crate::custom_family::BlockwiseFitOptions, _n: usize) -> f64 {
    match opts.outer_score_subsample.as_ref() {
        Some(s) => s.weight_scale,
        None => 1.0,
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

// ---------------------------------------------------------------------------
// Phase 4: biobank-scale outer-score subsample auto-injection.
//
// At biobank scale (n > BIOBANK_OUTER_SUBSAMPLE_THRESHOLD) the
// marginal-slope outer optimization burns most of its time in O(n) per-row
// score / Hessian sweeps. `inject_biobank_outer_subsample` constructs a
// stratified subsample (100 z-deciles × distinct secondary classes) and
// installs it on the supplied `BlockwiseFitOptions` so outer-only sweeps
// downscale to ~K rows. Inner-PIRLS and final-covariance passes still run
// on the full data because they don't consult `outer_score_subsample`.
//
// The auto-enable gate fires only when `outer_score_subsample.is_none()`
// so any caller that already supplied a subsample (for example a test or
// a future CLI flag) is preserved verbatim.
// ---------------------------------------------------------------------------

/// Auto-enable threshold for the biobank-scale outer-score subsample.
pub const BIOBANK_OUTER_SUBSAMPLE_THRESHOLD: usize = 50_000;
/// Lower bound on the auto-derived subsample size — below this the
/// stratified estimator becomes too noisy for the outer optimizer.
pub const BIOBANK_OUTER_SUBSAMPLE_K_MIN: usize = 4_000;
/// Upper bound on the auto-derived subsample size — above this the
/// per-iter savings from subsampling diminish (and memory cost grows).
pub const BIOBANK_OUTER_SUBSAMPLE_K_MAX: usize = 40_000;
/// Default subsample size targeted at biobank scale; mgcv::bam(discrete=TRUE)
/// regime, ample for stable score / Hessian sums. Retained as a public
/// reference point and as the value used by tests; the runtime gate uses
/// [`auto_outer_subsample_k`] instead.
pub const BIOBANK_OUTER_SUBSAMPLE_K: usize = 20_000;
/// Deterministic seed for the auto-enabled subsample. Tests rely on this
/// being stable across runs.
pub const BIOBANK_OUTER_SUBSAMPLE_SEED: u64 = 0xC0FFEE_5EED;

/// Auto-derive the outer-score subsample size from the data row count.
///
/// "Magic by default": no CLI flag, no env var. The runtime picks a K that
/// targets ≈6% of n (so subsample work is ≈16× cheaper than full-data work)
/// with a floor of 4_000 (statistical adequacy for stratified score sums)
/// and a ceiling of 40_000 (diminishing returns + memory).
///
/// At the canonical anchor points:
/// - n = 50_000 (threshold) → K = 4_000   (8% of n; floor binds)
/// - n = 100_000             → K = 6_250
/// - n = 320_000 (biobank)   → K = 20_000  (matches prior hardcoded default)
/// - n = 1_000_000           → K = 40_000  (ceiling binds; ≈4% of n)
///
/// The 1/16 ratio is the sweet spot per Wood's `mgcv::bam(discrete=TRUE)`
/// experiments — the outer-score gradient SE is dominated by the constant
/// stratification contribution rather than the subsample variance, so K
/// can grow sub-linearly with n without losing precision.
pub fn auto_outer_subsample_k(n: usize) -> usize {
    (n / 16)
        .max(BIOBANK_OUTER_SUBSAMPLE_K_MIN)
        .min(BIOBANK_OUTER_SUBSAMPLE_K_MAX)
}

/// Install a stratified outer-score subsample on `opts` when `n` exceeds the
/// biobank-scale threshold and no subsample is already configured. Returns
/// `true` when a subsample was installed, `false` otherwise.
///
/// `z` is the linearized PGS axis the marginal-slope family stratifies on.
/// `secondary` is the distinct-class indicator (binary y for bernoulli,
/// event indicator for survival) used as the secondary stratification key.
///
/// This is a thin, separately-testable wrapper around
/// [`build_outer_score_subsample`] so the wiring layer (workflow.rs) can be
/// exercised without spinning up a full fit.
pub fn inject_biobank_outer_subsample(
    opts: &mut crate::custom_family::BlockwiseFitOptions,
    z: &[f64],
    secondary: &[u8],
) -> bool {
    let n = z.len();
    if n <= BIOBANK_OUTER_SUBSAMPLE_THRESHOLD {
        return false;
    }
    if opts.outer_score_subsample.is_some() {
        return false;
    }
    if z.len() != secondary.len() {
        // Defensive: mismatched lengths would panic inside the builder; bail
        // out instead so the legacy full-data path keeps running.
        return false;
    }
    let k = auto_outer_subsample_k(n);
    let subsample = build_outer_score_subsample(z, secondary, k, BIOBANK_OUTER_SUBSAMPLE_SEED);
    log::info!(
        "[biobank-scale] constructed outer-score subsample: n={} k={} weight_scale={:.3} seed={:#x}",
        n,
        subsample.len(),
        subsample.weight_scale,
        BIOBANK_OUTER_SUBSAMPLE_SEED,
    );
    opts.outer_score_subsample = Some(Arc::new(subsample));
    true
}

/// Convenience wrapper that converts an `Array1<f64>` event/y indicator
/// (the canonical marginal-slope storage) into the `&[u8]` form the
/// stratifier expects, then delegates to [`inject_biobank_outer_subsample`].
///
/// Non-zero entries map to `1u8`, exact zeros to `0u8`. NaNs are bucketed
/// with non-zero (treated as a third class only if any are present, but
/// `build_outer_score_subsample` handles arbitrary u8 alphabets transparently).
pub fn inject_biobank_outer_subsample_from_arrays(
    opts: &mut crate::custom_family::BlockwiseFitOptions,
    z: &[f64],
    secondary_f64: &[f64],
) -> bool {
    if z.len() != secondary_f64.len() {
        return false;
    }
    let secondary: Vec<u8> = secondary_f64
        .iter()
        .map(|&v| if v != 0.0 { 1u8 } else { 0u8 })
        .collect();
    inject_biobank_outer_subsample(opts, z, &secondary)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn auto_outer_subsample_k_anchor_points() {
        // The anchors documented in the docstring should match the
        // implementation. If the heuristic changes, update both.
        assert_eq!(auto_outer_subsample_k(50_000), 4_000);
        assert_eq!(auto_outer_subsample_k(100_000), 6_250);
        assert_eq!(auto_outer_subsample_k(320_000), 20_000);
        assert_eq!(auto_outer_subsample_k(1_000_000), 40_000);
    }

    #[test]
    fn auto_outer_subsample_k_floor_binds_below_threshold_x16() {
        // Below n = K_MIN * 16 = 64_000, the n/16 term is below the floor.
        assert_eq!(auto_outer_subsample_k(0), BIOBANK_OUTER_SUBSAMPLE_K_MIN);
        assert_eq!(
            auto_outer_subsample_k(60_000),
            BIOBANK_OUTER_SUBSAMPLE_K_MIN
        );
        assert_eq!(
            auto_outer_subsample_k(64_000),
            BIOBANK_OUTER_SUBSAMPLE_K_MIN
        );
    }

    #[test]
    fn auto_outer_subsample_k_ceiling_binds_above_threshold_x16() {
        // Above n = K_MAX * 16 = 640_000, the n/16 term is above the ceiling.
        assert_eq!(
            auto_outer_subsample_k(640_000),
            BIOBANK_OUTER_SUBSAMPLE_K_MAX
        );
        assert_eq!(
            auto_outer_subsample_k(10_000_000),
            BIOBANK_OUTER_SUBSAMPLE_K_MAX
        );
    }

    #[test]
    fn auto_outer_subsample_k_is_monotone_non_decreasing() {
        // Sanity: more rows never asks for fewer subsample rows.
        let mut prev = 0usize;
        for n in (0..2_000_000).step_by(7919) {
            let k = auto_outer_subsample_k(n);
            assert!(
                k >= prev,
                "auto_outer_subsample_k regressed at n={n}: prev={prev} k={k}"
            );
            prev = k;
        }
    }

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

    // ── Phase 4 wiring: biobank-scale auto-injection ───────────────────────

    #[test]
    fn inject_biobank_outer_subsample_fires_at_biobank_scale() {
        // n=100k > 50k threshold → subsample should be installed and the
        // mask length should be roughly K (within stratification overshoot).
        let n: usize = 100_000;
        let z: Vec<f64> = (0..n).map(|i| (i as f64) * 1e-3).collect();
        let secondary: Vec<u8> = (0..n).map(|i| (i % 2) as u8).collect();
        let mut opts = crate::custom_family::BlockwiseFitOptions::default();
        assert!(opts.outer_score_subsample.is_none());
        let installed = inject_biobank_outer_subsample(&mut opts, &z, &secondary);
        assert!(installed, "subsample should be installed at n={}", n);
        let s = opts
            .outer_score_subsample
            .as_ref()
            .expect("subsample present");
        // K with 200-stratum overshoot bound.
        assert!(
            s.len() >= BIOBANK_OUTER_SUBSAMPLE_K,
            "mask len {} below K {}",
            s.len(),
            BIOBANK_OUTER_SUBSAMPLE_K
        );
        assert!(
            s.len() <= BIOBANK_OUTER_SUBSAMPLE_K + 200,
            "mask len {} much larger than K {} + 200",
            s.len(),
            BIOBANK_OUTER_SUBSAMPLE_K
        );
        assert_eq!(s.n_full, n);
        // weight_scale ≈ n / mask.len() ≈ 5.
        assert!((s.weight_scale - 5.0).abs() < 0.5);
        assert_eq!(s.seed, BIOBANK_OUTER_SUBSAMPLE_SEED);
    }

    #[test]
    fn inject_biobank_outer_subsample_skips_below_threshold() {
        // n=10k ≤ 50k threshold → subsample stays None.
        let n: usize = 10_000;
        let z: Vec<f64> = (0..n).map(|i| (i as f64) * 1e-3).collect();
        let secondary: Vec<u8> = (0..n).map(|i| (i % 2) as u8).collect();
        let mut opts = crate::custom_family::BlockwiseFitOptions::default();
        let installed = inject_biobank_outer_subsample(&mut opts, &z, &secondary);
        assert!(!installed, "subsample must not fire below threshold");
        assert!(opts.outer_score_subsample.is_none());
    }

    #[test]
    fn inject_biobank_outer_subsample_preserves_existing() {
        // Pre-existing subsample must not be overwritten.
        let n: usize = 100_000;
        let z: Vec<f64> = (0..n).map(|i| (i as f64) * 1e-3).collect();
        let secondary: Vec<u8> = (0..n).map(|i| (i % 2) as u8).collect();
        let preset = build_outer_score_subsample(&z, &secondary, 1_000, 0xABCDEF);
        let preset_seed = preset.seed;
        let preset_len = preset.len();
        let mut opts = crate::custom_family::BlockwiseFitOptions::default();
        opts.outer_score_subsample = Some(Arc::new(preset));
        let installed = inject_biobank_outer_subsample(&mut opts, &z, &secondary);
        assert!(!installed, "existing subsample must be preserved");
        let s = opts
            .outer_score_subsample
            .as_ref()
            .expect("subsample present");
        assert_eq!(s.seed, preset_seed);
        assert_eq!(s.len(), preset_len);
    }

    #[test]
    fn inject_biobank_outer_subsample_from_arrays_maps_nonzero_to_one() {
        // f64-secondary convenience: any non-zero entry should map to 1u8.
        let n: usize = 100_000;
        let z: Vec<f64> = (0..n).map(|i| (i as f64) * 1e-3).collect();
        // Mix of 0.0 and 1.0 — exercises the standard binary case.
        let secondary_f64: Vec<f64> = (0..n).map(|i| (i % 2) as f64).collect();
        let mut opts = crate::custom_family::BlockwiseFitOptions::default();
        let installed = inject_biobank_outer_subsample_from_arrays(&mut opts, &z, &secondary_f64);
        assert!(installed);
        let s = opts
            .outer_score_subsample
            .as_ref()
            .expect("subsample present");
        assert!(s.len() >= BIOBANK_OUTER_SUBSAMPLE_K);
    }

    #[test]
    fn inject_biobank_outer_subsample_rejects_mismatched_lengths() {
        let z: Vec<f64> = (0..100_000).map(|i| i as f64).collect();
        let secondary: Vec<u8> = (0..50_000).map(|i| (i % 2) as u8).collect();
        let mut opts = crate::custom_family::BlockwiseFitOptions::default();
        let installed = inject_biobank_outer_subsample(&mut opts, &z, &secondary);
        assert!(!installed);
        assert!(opts.outer_score_subsample.is_none());
    }
}
