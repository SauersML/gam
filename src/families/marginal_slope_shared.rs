//! Shared kernels and outer-evaluation infrastructure for the
//! marginal-slope family of GAMs (BMS, survival, latent survival).
//!
//! # Outer-row subsampling
//!
//! At large scale (n ≥ tens of thousands) the outer rho-gradient is
//! a sum-over-rows trace whose per-row cost is dominated by the cubic
//! cell-moment kernel. The pieces here — [`AutoOuterSubsampleOptions`],
//! [`auto_outer_score_subsample`], [`maybe_install_auto_outer_subsample`],
//! and [`build_outer_score_subsample`] — implement a stratified
//! Horvitz–Thompson estimator that replaces the full row sum with an
//! unbiased sample, gated by
//! [`crate::custom_family::BlockwiseFitOptions::auto_outer_subsample`]
//! and enabled by default for large marginal-slope fits.
//!
//! `maybe_install_auto_outer_subsample` is the entry point family
//! impls call: it consults the per-family phase counter and the
//! per-family last-ρ mutex (used to detect distinct outer steps),
//! installs a stratified mask for the first
//! `BMS_AUTO_SUBSAMPLE_PHASE1_BUDGET` (or family analog) outer
//! evaluations, and reverts to full data afterward so the BFGS/ARC
//! convergence target `outer_tol` is reached on exact gradients
//! rather than chasing the stochastic noise floor.
//!
//! This subsampling is **complementary** to the trace-estimator tier
//! system documented at the top of `solver::reml::unified` (exact /
//! Hutchinson multi-target / Hutch++ single-target). They operate on
//! orthogonal axes — the trace estimators reduce work *within* the
//! Hessian structure for a fixed row set; subsampling reduces the row
//! set itself for the family-specific row-trace path.

use crate::custom_family::{CustomFamilyBlockPsiDerivative, ParameterBlockSpec};
use crate::solver::outer_subsample::{OuterScoreSubsample, WeightedOuterRow};
use crate::families::cubic_cell_kernel::{self, DenestedPartitionCell, LocalSpanCubic};
use crate::families::jet_partitions::MultiDirJet;
use ndarray::{Array1, Array2, Axis};
use std::ops::Range;
use std::sync::Arc;

/// Canonical inner-cache `beta_seed` validator passed to the generic
/// outer-engine (`optimize_spatial_length_scale_exact_joint`).
///
/// The outer solver hands back the converged inner `beta` at each accepted
/// ρ-step so the next inner solve can warm-start from it. This guards that
/// cached vector for non-finite entries (which would poison the warm start)
/// and, when clean, stashes it into the caller's `pending` cell.
///
/// This is the single source of truth for the seed callback: every family
/// that wires up the exact-joint outer engine (survival location-scale,
/// bernoulli marginal-slope, survival marginal-slope) routes through here
/// instead of re-deriving the identical closure, which previously drifted in
/// error construction (`EstimationError::InvalidInput(...)` vs
/// `bail_invalid_estim!`).
pub fn make_beta_seed_validator(
    pending: &std::cell::RefCell<Option<Array1<f64>>>,
) -> impl FnMut(
    &Array1<f64>,
) -> Result<
    crate::solver::rho_optimizer::SeedOutcome,
    crate::solver::estimate::EstimationError,
> + '_ {
    move |beta: &Array1<f64>| {
        bail_if_cached_beta_non_finite(beta)?;
        // Stage the seed for promotion at the next eval, where the freshly
        // built per-block widths are known. A width mismatch is reconciled
        // there (the eval's `from_cached_beta` logs and falls back to a cold
        // β for that step) — never an error that aborts the fit. Staging a
        // finite β always succeeds, so the contract reply is `Installed`.
        pending.replace(Some(beta.clone()));
        Ok(crate::solver::rho_optimizer::SeedOutcome::Installed)
    }
}

/// Canonical non-finite guard on a cached inner `beta`.
///
/// Single source of truth for the `"cached inner beta contains non-finite
/// entries"` check + error: the full seed closure
/// ([`make_beta_seed_validator`]) and the bare warm-start length-then-finite
/// guards in `custom_family` all route through this so the predicate and the
/// error construction (`EstimationError::InvalidInput`) never drift apart.
#[inline]
pub fn bail_if_cached_beta_non_finite(
    beta: &Array1<f64>,
) -> Result<(), crate::solver::estimate::EstimationError> {
    if beta.iter().any(|v| !v.is_finite()) {
        crate::bail_invalid_estim!("cached inner beta contains non-finite entries");
    }
    Ok(())
}

/// Positive floor applied to the time-derivative slope `a1 = q'(t)·c` before
/// taking `ln(a1)` or `1/a1` in the survival marginal-slope likelihood. The
/// monotonicity guard already enforces `a1 > derivative_guard`, but this is the
/// last line of defence against a `ln(0)`/divide-by-zero blowing up the NLL or
/// score when `a1` underflows to a denormal; set at the smallest power-of-ten
/// double that keeps `ln` finite and the reciprocal representable.
pub const SURVIVAL_SLOPE_LOG_DIVIDE_FLOOR: f64 = 1e-300;

#[inline]
pub const fn eval_coeff4_at(coefficients: &[f64; 4], z: f64) -> f64 {
    ((coefficients[3] * z + coefficients[2]) * z + coefficients[1]) * z + coefficients[0]
}

#[inline]
pub fn add_scaled_coeff4(target: &mut [f64; 4], source: &[f64; 4], scale: f64) {
    for j in 0..4 {
        target[j] += scale * source[j];
    }
}

#[inline]
fn coeff4_dot(left: &[f64; 4], right: &[f64; 4]) -> f64 {
    left[0] * right[0] + left[1] * right[1] + left[2] * right[2] + left[3] * right[3]
}

#[inline]
pub const fn scale_coeff4(source: [f64; 4], scale: f64) -> [f64; 4] {
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

/// Per-sweep scale jets for the shared directional obj/grad/hess kernel.
///
/// Every marginal-slope family forms its exact-Newton primary terms by
/// differentiating the same row negative-log directional jet
/// (`row_neglog_directional_with_scale_jet`) along unit primary directions.
/// The sweep appends one unit direction for the gradient pass and two for the
/// Hessian pass on top of a fixed *leading* prefix of directions, scaling the
/// frailty kernel with an order-matched [`MultiDirJet`] each time. The `obj`
/// slot is `Some` only when the caller also wants the zeroth-order objective
/// (the prefix-only evaluation); psi-Hessian directional sweeps leave it `None`.
#[derive(Clone)]
pub(crate) struct DirectionalScaleJets {
    pub(crate) obj: Option<MultiDirJet>,
    pub(crate) grad: MultiDirJet,
    pub(crate) hess: MultiDirJet,
}

/// Output of [`directional_obj_grad_hess`]: the (optional) zeroth-order
/// objective, the full primary gradient, and the symmetric primary Hessian.
pub(crate) struct DirectionalPrimaryTerms {
    pub(crate) objective: f64,
    pub(crate) grad: Array1<f64>,
    pub(crate) hess: Array2<f64>,
}

/// Shared exact-Newton primary directional sweep for the marginal-slope
/// families (Bernoulli, survival, latent survival).
///
/// Given a fixed `leading` prefix of directions and a family-specific row jet
/// evaluator `eval`, this builds the objective (when requested), the gradient
/// `g_a = D[leading, e_a] φ`, and the symmetric Hessian
/// `H_ab = D[leading, e_a, e_b] φ`, where `e_a` is the `a`-th unit primary
/// direction (length `primary_dim`) and `D[..]` is the mixed directional
/// derivative the row jet returns. `eval(dirs, scale)` must return the highest
/// mixed-partial coefficient of the row negative-log jet for the supplied
/// directions and scale jet — exactly what each family's
/// `row_neglog_directional_with_scale_jet` produces.
///
/// Centralizing the sweep removes the per-family duplication of the
/// obj/grad/hess loop nest, which is the single most drift-prone piece of the
/// exact-Newton stack: a stray index or a missing symmetric assignment in one
/// copy silently destabilizes only that family's optimizer.
pub(crate) fn directional_obj_grad_hess<Eval>(
    primary_dim: usize,
    leading: &[&Array1<f64>],
    scales: &DirectionalScaleJets,
    eval: Eval,
) -> Result<DirectionalPrimaryTerms, String>
where
    Eval: Fn(&[&Array1<f64>], &MultiDirJet) -> Result<f64, String>,
{
    let objective = if let Some(scale_obj) = scales.obj.as_ref() {
        eval(leading, scale_obj)?
    } else {
        0.0
    };

    let unit = |a: usize| -> Array1<f64> {
        let mut da = Array1::<f64>::zeros(primary_dim);
        da[a] = 1.0;
        da
    };

    let units: Vec<Array1<f64>> = (0..primary_dim).map(unit).collect();

    let mut grad = Array1::<f64>::zeros(primary_dim);
    let mut dirs: Vec<&Array1<f64>> = Vec::with_capacity(leading.len() + 2);
    for a in 0..primary_dim {
        dirs.clear();
        dirs.extend_from_slice(leading);
        dirs.push(&units[a]);
        grad[a] = eval(&dirs, &scales.grad)?;
    }

    let mut hess = Array2::<f64>::zeros((primary_dim, primary_dim));
    for a in 0..primary_dim {
        for b in a..primary_dim {
            dirs.clear();
            dirs.extend_from_slice(leading);
            dirs.push(&units[a]);
            dirs.push(&units[b]);
            let value = eval(&dirs, &scales.hess)?;
            hess[[a, b]] = value;
            hess[[b, a]] = value;
        }
    }

    Ok(DirectionalPrimaryTerms {
        objective,
        grad,
        hess,
    })
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
    score_warp: Option<&crate::families::bms::DeviationRuntime>,
    beta_h: Option<&Array1<f64>>,
    link_dev: Option<&crate::families::bms::DeviationRuntime>,
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
    score_warp: Option<&crate::families::bms::DeviationRuntime>,
    beta_h: Option<&Array1<f64>>,
    link_dev: Option<&crate::families::bms::DeviationRuntime>,
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

/// Predicate used by every marginal-slope family's persistent-warm-start
/// fingerprint guard: the caller's parameter blocks must each have row count
/// matching the family's `n`, and the list must be non-empty.
#[inline]
pub(crate) fn parameter_block_specs_match_rows(
    specs: &[ParameterBlockSpec],
    expected_n: usize,
) -> bool {
    !specs.is_empty()
        && specs
            .iter()
            .all(|spec| spec.design.nrows() == expected_n && spec.offset.len() == expected_n)
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
        if support.include_h
            && let Some(h_range) = self.h_range.as_ref()
        {
            for idx in h_range.clone() {
                add_scaled_coeff4(&mut out, &family[idx], dir[idx]);
            }
        }
        if support.include_w
            && let Some(w_range) = self.w_range.as_ref()
        {
            for idx in w_range.clone() {
                add_scaled_coeff4(&mut out, &family[idx], dir[idx]);
            }
        }
        out
    }

    pub(crate) fn add_directional_family_adjoint(
        &self,
        family: &[[f64; 4]],
        coeff_adjoint: &[f64; 4],
        support: CoeffSupport,
        direction_adjoint: &mut [f64],
    ) {
        assert!(direction_adjoint.len() > self.primary_index);
        if support.include_primary {
            direction_adjoint[self.primary_index] +=
                coeff4_dot(coeff_adjoint, &family[self.primary_index]);
        }
        if support.include_h
            && let Some(h_range) = self.h_range.as_ref()
        {
            for idx in h_range.clone() {
                direction_adjoint[idx] += coeff4_dot(coeff_adjoint, &family[idx]);
            }
        }
        if support.include_w
            && let Some(w_range) = self.w_range.as_ref()
        {
            for idx in w_range.clone() {
                direction_adjoint[idx] += coeff4_dot(coeff_adjoint, &family[idx]);
            }
        }
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
        if support.include_h
            && let Some(h_range) = self.h_range.as_ref()
        {
            for idx in h_range.clone() {
                add_scaled_coeff4(
                    &mut out,
                    &family[idx],
                    dir_u_primary * dir_v[idx] + dir_v_primary * dir_u[idx],
                );
            }
        }
        if support.include_w
            && let Some(w_range) = self.w_range.as_ref()
        {
            for idx in w_range.clone() {
                add_scaled_coeff4(
                    &mut out,
                    &family[idx],
                    dir_u_primary * dir_v[idx] + dir_v_primary * dir_u[idx],
                );
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

    pub(crate) fn add_param_directional_from_b_family_adjoint(
        &self,
        family: &[[f64; 4]],
        param: usize,
        coeff_adjoint: &[f64; 4],
        support: CoeffSupport,
        direction_adjoint: &mut [f64],
    ) {
        assert!(direction_adjoint.len() > self.primary_index);
        if param == self.primary_index {
            self.add_directional_family_adjoint(family, coeff_adjoint, support, direction_adjoint);
        } else if self.param_supported(param, support.without_primary()) {
            direction_adjoint[self.primary_index] += coeff4_dot(coeff_adjoint, &family[param]);
        }
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

    pub(crate) fn add_pair_directional_from_bb_family_adjoint(
        &self,
        family: &[[f64; 4]],
        u: usize,
        v: usize,
        coeff_adjoint: &[f64; 4],
        support: CoeffSupport,
        direction_adjoint: &mut [f64],
    ) {
        assert!(direction_adjoint.len() > self.primary_index);
        if u == self.primary_index && v == self.primary_index {
            self.add_directional_family_adjoint(family, coeff_adjoint, support, direction_adjoint);
        } else if u == self.primary_index && self.param_supported(v, support.without_primary()) {
            direction_adjoint[self.primary_index] += coeff4_dot(coeff_adjoint, &family[v]);
        } else if v == self.primary_index && self.param_supported(u, support.without_primary()) {
            direction_adjoint[self.primary_index] += coeff4_dot(coeff_adjoint, &family[u]);
        }
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
// The large-scale outer-loop score/gradient passes do O(n) work per outer
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

/// Splitmix64: deterministic single-u64 expansion. Thin wrapper over the
/// canonical implementation in [`crate::linalg::utils::splitmix64`].
#[inline]
const fn splitmix64(state: &mut u64) -> u64 {
    crate::linalg::utils::splitmix64(state)
}

/// Configuration for the automatic outer-score subsampler.
///
/// At large scale (n ≥ tens of thousands) the marginal-slope outer
/// rho-gradient computes a sum-over-rows trace
/// `tr(F Fᵀ M_k) = Σ_i row_i(k)` whose per-row work is dominated by
/// the cell-moment kernel. Stratified Horvitz–Thompson subsampling
/// replaces the full sum with an unbiased estimator using `K` of `N`
/// rows; the trace cost drops from `O(N · cell_work)` to
/// `O(K · cell_work)`.
///
/// # Math
///
/// Estimator `T̂ = Σ_{i∈S} w_i · row_i` with HT weights
/// `w_i = N_h / K_h` (per-stratum) is unbiased: `E[T̂] = T`.
///
/// Variance under stratified SRS without replacement:
/// `Var(T̂) = Σ_h N_h² (1 − K_h/N_h) S_h² / K_h`
/// where `S_h²` is the within-stratum variance of per-row contributions.
/// With proportional allocation `K_h = K · N_h/N`, the standard deviation
/// of `T̂` relative to `T` is roughly
/// `σ(T̂)/T ≈ (1/√K) · √(1 − K/N) · cv_within`
/// where `cv_within` is the within-stratum coefficient of variation.
///
/// The defaults are tuned so that the relative gradient-noise σ stays
/// below ≈ 1 % across realistic `n` ∈ [30 000, 300 000+], assuming
/// `cv_within ≲ 1` (which holds for marginal-slope contributions
/// because the z-decile stratification absorbs the dominant
/// inhomogeneity).
#[derive(Clone, Debug)]
pub struct AutoOuterSubsampleOptions {
    /// Below this `n`, the auto-subsampler always returns `None` (use
    /// full data). Default 30 000.
    pub min_n_for_auto: usize,
    /// Floor on `K`, so the relative gradient noise stays bounded
    /// even when the target fraction would round to a smaller `K`.
    /// `K = max(min_k, round(n · target_fraction))`. Default 10 000
    /// gives `σ/T ≤ 1 %` for cv_within ≤ 1 and any `n ≥ min_n_for_auto`.
    pub min_k: usize,
    /// Target ratio `K / n` once `n ≫ min_k`. Default 0.10.
    pub target_fraction: f64,
    /// RNG seed for stratified mask construction. Default
    /// `0xA075_8AMP_LE_5UB5` (deterministic across runs at the same
    /// `n`, so CRN holds across BFGS iterations).
    pub seed: u64,
    /// Family-supplied **per-unit-of-K** outer-derivative work cost.
    ///
    /// Despite the historical name, this is *not* a per-row quantity.
    /// It is `predicted_outer_gradient_work / K` evaluated at the
    /// family's reference operating point — i.e. how many work units
    /// each additional row in the K-subsample contributes summed over
    /// all n. The auto schedule caps `K` by
    /// `K_work = AUTO_OUTER_WORK_BUDGET / outer_work_per_k_unit`,
    /// guaranteeing a single outer evaluation never exceeds
    /// [`AUTO_OUTER_WORK_BUDGET`] work units regardless of the
    /// noise-only target. Default `1` (no effective work cap beyond
    /// `K ≤ n`); families with measurable per-K cost (survival
    /// marginal-slope, BMS) overwrite at the call site.
    ///
    /// Calibration recipe: from a profiled run,
    ///     outer_work_per_k_unit = predicted_gradient_work / K.
    /// For the large-scale survival marginal-slope reference
    /// (predicted_gradient_work ≈ 4.33×10⁹ at K=19_661), this gives
    /// ~220_000; we use 250_000 as a conservative upper bound. With
    /// `AUTO_OUTER_WORK_BUDGET = 5×10⁸` that caps K at ~2_000.
    pub outer_work_per_k_unit: u64,
}

/// Half-billion outer-derivative work units per evaluation. Picked so the
/// rigid survival marginal-slope pilot Newton cycle (which previously ran
/// ~57 min at n≈2e5 with `K=19_661`) finishes in a minute or two on
/// commodity hardware once `K` is capped by this budget.
pub const AUTO_OUTER_WORK_BUDGET: u64 = 500_000_000;

/// Absolute floor on `K` chosen by the auto schedule. Even when the work
/// budget would drive `K` to a handful of rows the stratified mask cannot
/// usefully shrink below `MIN_K_FLOOR` without collapsing entire deciles
/// of `z`-strata. Set so the resulting gradient noise (~3 %) is still
/// usable for BFGS Phase 1 progress when the family is very expensive.
pub const AUTO_OUTER_MIN_K_FLOOR: usize = 1_000;

/// L2 distance below which two outer ρ keys are treated as the *same* outer
/// step (a line-search retry, not a fresh outer iteration). Well below any
/// meaningful BFGS step on log-scale ρ, well above float-noise from cloning
/// the ρ vector. Used to keep the phase-1 budget counting outer iterations
/// rather than per-step function evaluations.
const AUTO_OUTER_DISTINCT_STEP_L2_TOL: f64 = 1e-10;

/// Reason the auto schedule chose the reported `K`. Used by the
/// `[family auto-subsample]` log line so operators can tell whether the
/// noise model, the work budget, the `MIN_K_FLOOR`, or `n` itself
/// determined the subsample size.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum AutoOuterCapReason {
    Noise,
    Work,
    Floor,
    NFull,
}

impl AutoOuterCapReason {
    pub fn as_str(self) -> &'static str {
        match self {
            AutoOuterCapReason::Noise => "noise",
            AutoOuterCapReason::Work => "work",
            AutoOuterCapReason::Floor => "floor",
            AutoOuterCapReason::NFull => "n",
        }
    }
}

impl Default for AutoOuterSubsampleOptions {
    fn default() -> Self {
        Self {
            min_n_for_auto: 30_000,
            min_k: 10_000,
            target_fraction: 0.10,
            seed: 0xA075_8A8B_1ED5_5B5C,
            outer_work_per_k_unit: 1,
        }
    }
}

/// Outcome of [`AutoOuterSubsampleOptions::target_k_detailed`]: the
/// chosen `K`, the underlying noise-only choice, the work-budget cap,
/// and which constraint won.
#[derive(Clone, Copy, Debug)]
pub struct AutoOuterKChoice {
    pub k: usize,
    pub k_noise: usize,
    pub k_work: usize,
    pub cap_reason: AutoOuterCapReason,
}

impl AutoOuterSubsampleOptions {
    /// Compute the K that this configuration would pick for a given n.
    /// Returns `None` if `n < min_n_for_auto` (caller should not subsample).
    pub fn target_k(&self, n: usize) -> Option<usize> {
        self.target_k_detailed(n).map(|choice| choice.k)
    }

    /// Same as [`target_k`] but also reports the noise-only `K`, the
    /// work-budget cap, and which constraint set the final value. Used by
    /// [`maybe_install_auto_outer_subsample`] to surface a `cap_reason`
    /// in the auto-subsample log line.
    pub fn target_k_detailed(&self, n: usize) -> Option<AutoOuterKChoice> {
        if n < self.min_n_for_auto {
            return None;
        }
        let k_noise_raw = ((n as f64) * self.target_fraction).round() as usize;
        let k_noise = k_noise_raw.max(self.min_k);
        // Work-budget cap. `outer_work_per_k_unit == 1` is the
        // default-1-work-unit signal that the family has not measured
        // its per-K cost, in which case the work cap is `WORK_BUDGET`
        // and typically dominated by `n`.
        let work_per_k = self.outer_work_per_k_unit.max(1);
        let k_work_u64 = AUTO_OUTER_WORK_BUDGET / work_per_k;
        let k_work = usize::try_from(k_work_u64).unwrap_or(usize::MAX);
        // Combine noise + work + n + floor in a single comparison so we
        // can attribute the binding constraint exactly once.
        let mut k = k_noise.min(k_work);
        let mut cap_reason = if k_work < k_noise {
            AutoOuterCapReason::Work
        } else {
            AutoOuterCapReason::Noise
        };
        if k < AUTO_OUTER_MIN_K_FLOOR {
            k = AUTO_OUTER_MIN_K_FLOOR;
            cap_reason = AutoOuterCapReason::Floor;
        }
        if k > n {
            k = n;
            cap_reason = AutoOuterCapReason::NFull;
        }
        if k >= n {
            // Borderline: the auto schedule would cover the whole
            // dataset. Subsampling buys nothing.
            return None;
        }
        Some(AutoOuterKChoice {
            k,
            k_noise,
            k_work,
            cap_reason,
        })
    }
}

/// Build a stratified outer-score subsample automatically from problem
/// characteristics. Returns `None` for problems too small to benefit
/// (the caller should fall back to the full-data path).
///
/// Stratification matches `build_outer_score_subsample`: 100 z-deciles
/// × the supplied secondary stratum (typically the {0, 1} response
/// indicator). When `stratum_secondary` is `None` the secondary
/// dimension collapses to a single bin.
///
/// The returned mask carries proper Horvitz–Thompson weights so that
/// `Σ_{i ∈ mask} weight_i · row_i` is an unbiased estimate of the
/// full row sum.
pub fn auto_outer_score_subsample(
    z: &[f64],
    stratum_secondary: Option<&[u8]>,
    options: &AutoOuterSubsampleOptions,
) -> Option<OuterScoreSubsample> {
    let n = z.len();
    let k = options.target_k(n)?;
    let secondary_storage;
    let secondary: &[u8] = if let Some(s) = stratum_secondary {
        if s.len() != n {
            // Caller error; fall through to no-subsample rather than panic.
            return None;
        }
        s
    } else {
        secondary_storage = vec![0u8; n];
        &secondary_storage
    };
    Some(build_outer_score_subsample(z, secondary, k, options.seed))
}

/// Two-phase auto-subsample guard shared across marginal-slope families.
///
/// Returns `Some(cloned_options)` carrying a freshly stratified
/// Horvitz-Thompson mask when `options.auto_outer_subsample` is enabled, the
/// caller has not already supplied a mask, and
/// the per-family phase counter is below `phase1_budget`. Returns `None`
/// when the caller's options should be used unchanged (either subsample
/// is disabled / pre-installed, the budget is exhausted, or the problem
/// is too small for `auto_outer_score_subsample` to find a benefit).
///
/// The `phase_counter` and `last_rho` pair together implement
/// distinct-step detection: line searches re-call the family at the
/// same ρ during step-size retries, but the budget is meant to count
/// outer iterations, not function evaluations. The counter only ticks
/// when the incoming ρ differs from the last observed ρ in L2 by
/// > 1e-10 — well below any meaningful BFGS step on log-scale ρ, well
/// > above float-noise from cloning. The mutex around `last_rho` is the
/// > minimal coordination needed: `(counter, last_rho)` must update
/// > together so two threads cannot both decide "new ρ" and double-bump.
///
/// The transition at `phase_idx == phase1_budget` is logged exactly
/// once via `log::info!` with the supplied `family_label`. Each phase-1
/// install also logs the planned mask size and predicted gradient
/// noise. Callers running with auto-subsample disabled see no logging.
pub fn maybe_install_auto_outer_subsample(
    options: &crate::custom_family::BlockwiseFitOptions,
    z: &[f64],
    stratum_secondary: Option<&[u8]>,
    outer_rho_key: &[f64],
    phase_counter: &Arc<std::sync::atomic::AtomicUsize>,
    last_rho: &Arc<std::sync::Mutex<Option<Array1<f64>>>>,
    phase1_budget: usize,
    family_label: &'static str,
    outer_work_per_k_unit: u64,
) -> Option<crate::custom_family::BlockwiseFitOptions> {
    if options.outer_score_subsample.is_some() || !options.auto_outer_subsample {
        return None;
    }
    let phase_idx = {
        let mut guard = last_rho
            .lock()
            .expect("auto_subsample_last_rho mutex poisoned");
        let new_step = match guard.as_ref() {
            None => true,
            Some(prev) if prev.len() != outer_rho_key.len() => true,
            Some(prev) => {
                let mut sq = 0.0_f64;
                for (a, b) in outer_rho_key.iter().zip(prev.iter()) {
                    let d = a - b;
                    sq += d * d;
                }
                sq.sqrt() > AUTO_OUTER_DISTINCT_STEP_L2_TOL
            }
        };
        if new_step {
            *guard = Some(Array1::from(outer_rho_key.to_vec()));
            phase_counter.fetch_add(1, std::sync::atomic::Ordering::SeqCst)
        } else {
            phase_counter
                .load(std::sync::atomic::Ordering::SeqCst)
                .saturating_sub(1)
        }
    };
    if phase_idx >= phase1_budget {
        if phase_idx == phase1_budget {
            log::info!(
                "[{family_label} auto-subsample] Phase 1 budget exhausted after {} evals; \
                 Phase 2 (full data) for remaining iterations",
                phase1_budget
            );
        }
        return None;
    }
    // Honour the family's per-K-unit work cost. Constructing
    // `AutoOuterSubsampleOptions::default()` here would silently reset
    // `outer_work_per_k_unit` to 1, making the work-budget cap
    // (`K_work = AUTO_OUTER_WORK_BUDGET / outer_work_per_k_unit`)
    // never bind, and letting the noise-only rule pick K ≈ 0.10·n —
    // which at large-scale n=195_780 is K≈19_578 instead of the survival
    // family's intended K≈2_000. That ~9× inflation drove the
    // documented 8h large-scale hang (exit 137 from resource exhaustion).
    let auto_options = AutoOuterSubsampleOptions {
        outer_work_per_k_unit: outer_work_per_k_unit.max(1),
        ..AutoOuterSubsampleOptions::default()
    };
    // Compute the K choice up-front so the log surfaces both the
    // noise-only target and the work-cap target even when stratification
    // ceil overshoots the picked K.
    let choice = auto_options.target_k_detailed(z.len())?;
    let mask = auto_outer_score_subsample(z, stratum_secondary, &auto_options)?;
    let n_full = mask.n_full;
    let k = mask.len();
    log::info!(
        "[{family_label} auto-subsample] phase=1 eval={}/{} n={} K={} fraction={:.3} expected_grad_noise={:.2}% work_per_k_unit={} k_noise={} k_work={} cap_reason={}",
        phase_idx + 1,
        phase1_budget,
        n_full,
        k,
        k as f64 / n_full.max(1) as f64,
        100.0 * (1.0 / (k as f64).sqrt()) * (1.0 - k as f64 / n_full.max(1) as f64).sqrt(),
        outer_work_per_k_unit,
        choice.k_noise,
        choice.k_work,
        choice.cap_reason.as_str(),
    );
    let mut cloned = options.clone();
    cloned.outer_score_subsample = Some(Arc::new(mask));
    Some(cloned)
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
        let take = (k as u128 * rows.len() as u128).div_ceil(n as u128) as usize;
        let take = take.max(1).min(rows.len());
        // HT inverse-inclusion weight for this stratum: w_h = N_h / k_h.
        // Identical for every row drawn from `stratum_id`.
        let w_h = rows.len() as f64 / take as f64;
        let stratum_tag = stratum_id as u32;

        // Deterministic key from (seed, stratum_id).
        let mut state = seed ^ (stratum_id as u64).wrapping_mul(0x9E3779B97F4A7C15);
        // Mix once so even seed=0, stratum_id=0 produces a non-trivial state.
        splitmix64(&mut state);

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
pub fn outer_score_scale(opts: &crate::custom_family::BlockwiseFitOptions, n: usize) -> f64 {
    match opts.outer_score_subsample.as_ref() {
        Some(s) => {
            assert_eq!(
                s.n_full, n,
                "outer-score subsample full-data length must match caller length"
            );
            s.weight_scale
        }
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

/// Shared monotonicity line-search safeguard for time-block linear inequality
/// constraints `A·beta >= b`.
///
/// Both survival families (location-scale and marginal-slope) clamp a Newton
/// step `beta + alpha·delta` to the largest feasible fraction `alpha ∈ [0, 1]`
/// such that no constraint row is driven below its bound, then back off by the
/// fixed `0.995` safeguard whenever the boundary is reached. The slack/drift
/// arithmetic and the `0.995` factor live here once; each family supplies only
/// its own error type by mapping the dimension-mismatch and constraint-violation
/// conditions into `E` via the two closures (preserving family-specific message
/// text and error variants).
///
/// `map_dim_err` is called with `(beta_len, delta_len, expected_ncols)` when the
/// step dimensions disagree with the constraint matrix. `map_violation_err` is
/// called with `(row, slack)` when the current `beta` already violates a
/// constraint row (slack below `-1e-10`).
pub fn feasible_step_fraction<E>(
    constraints: &crate::solver::active_set::LinearInequalityConstraints,
    beta: &Array1<f64>,
    direction: &Array1<f64>,
    map_dim_err: impl Fn(usize, usize, usize) -> E,
    map_violation_err: impl Fn(usize, f64) -> E,
) -> Result<f64, E> {
    if beta.len() != constraints.a.ncols() || direction.len() != constraints.a.ncols() {
        return Err(map_dim_err(
            beta.len(),
            direction.len(),
            constraints.a.ncols(),
        ));
    }
    // Feasibility-violation tolerance for the *current* iterate, kept consistent
    // with the QP entry gate `check_linear_feasibility` (called at 1e-8) and the
    // residual left by `project_onto_linear_constraints` (per-row violation <= 1e-10
    // on the working vector, accumulating up to O(1e-9) on the final beta through
    // its sequential Dykstra corrections). Rejecting at -1e-10 here re-classified a
    // beta the QP had already accepted as feasible as a hard error (gam#797: the
    // projected survival time-block seed lands at slack ~ -1.1e-9 on a binding
    // derivative-guard row, so every trust-region attempt errored out before any
    // step). A slack within this band is numerically AT the boundary; treat it as
    // active (slack = 0) rather than a violation.
    const FEASIBLE_STEP_VIOLATION_TOL: f64 = 1e-8;
    // Multiplicative backoff applied when the step is clipped by a binding
    // constraint, keeping the new iterate strictly interior (slack > 0) so the
    // next iteration's feasibility gate cannot reject a point that landed
    // exactly on the boundary through round-off.
    const FEASIBLE_STEP_BOUNDARY_BACKOFF: f64 = 0.995;
    let mut alpha = 1.0f64;
    for row in 0..constraints.a.nrows() {
        let a_row = constraints.a.row(row);
        let raw_slack = a_row.dot(beta) - constraints.b[row];
        if raw_slack < -FEASIBLE_STEP_VIOLATION_TOL {
            return Err(map_violation_err(row, raw_slack));
        }
        // Clamp boundary round-off to the boundary so a tiny negative slack cannot
        // produce a spurious negative/zero step fraction below.
        let slack = raw_slack.max(0.0);
        let drift = a_row.dot(direction);
        if drift < 0.0 {
            alpha = alpha.min((slack / -drift).clamp(0.0, 1.0));
        }
    }
    if alpha >= 1.0 {
        Ok(1.0)
    } else {
        Ok((FEASIBLE_STEP_BOUNDARY_BACKOFF * alpha).clamp(0.0, 1.0))
    }
}

/// Family-specific ψ-calculus hooks for the shared exact-Newton joint-ψ
/// workspace.
///
/// The two marginal-slope families (Bernoulli marginal-slope and survival
/// marginal-slope) build an [`ExactNewtonJointPsiWorkspace`] whose four methods
/// share a single skeleton: a σ-auxiliary (log-σ frailty) dispatch branch on
/// top of a family-specific non-σ row pass. The skeleton lives once in
/// [`MarginalSlopeExactNewtonPsiWorkspace`]; each family supplies only the
/// resolved per-call operations here, holding its own block states, specs,
/// derivative blocks, cache and outer-subsample options internally.
///
/// Implementors own all workspace state, so every hook takes only the ψ index /
/// pair / direction. The two genuine per-family policy differences in the
/// second-order σ-aux branch are encoded as
/// [`both_sigma_aux_second_order`](Self::both_sigma_aux_second_order) (which
/// pure-σ pairs are admissible) and
/// [`mixed_sigma_aux_second_order`](Self::mixed_sigma_aux_second_order) (how a
/// mixed σ / non-σ pair is handled) rather than being harmonized away.
pub trait MarginalSlopePsiFamily: Send + Sync {
    /// True when ψ index `psi_index` addresses the log-σ frailty auxiliary
    /// parameter rather than a spatial / spline derivative axis.
    fn is_sigma_aux(&self, psi_index: usize) -> bool;

    /// First-order joint-ψ terms for the σ-auxiliary parameter.
    fn sigma_first_order_terms(
        &self,
    ) -> Result<Option<crate::custom_family::ExactNewtonJointPsiTerms>, String>;

    /// First-order joint-ψ terms for a non-σ derivative axis `psi_index`.
    fn psi_first_order_terms(
        &self,
        psi_index: usize,
    ) -> Result<Option<crate::custom_family::ExactNewtonJointPsiTerms>, String>;

    /// Batched first-order joint-ψ terms over all derivative axes (used by the
    /// outer score sweep). Returns `Ok(None)` when the batched fast path is
    /// unavailable for the current configuration so the caller falls back to
    /// per-axis evaluation.
    fn psi_first_order_terms_all(
        &self,
    ) -> Result<Option<Vec<crate::custom_family::ExactNewtonJointPsiTerms>>, String>;

    /// Whether the σ-aux second-order branch should treat `(psi_i, psi_j)` as a
    /// pure-σ pair (dispatching to [`sigma_second_order_terms`](Self::sigma_second_order_terms)).
    /// Any σ-touching pair that is not pure-σ routes through
    /// [`mixed_sigma_aux_second_order`](Self::mixed_sigma_aux_second_order).
    fn both_sigma_aux_second_order(&self, psi_i: usize, psi_j: usize) -> bool;

    /// Second-order joint-ψ terms for a pure σ / σ pair.
    fn sigma_second_order_terms(
        &self,
    ) -> Result<Option<crate::custom_family::ExactNewtonJointPsiSecondOrderTerms>, String>;

    /// Per-family policy for a mixed σ / non-σ second-order pair: one family
    /// rejects it (no cross auxiliary terms available), the other returns
    /// `Ok(None)`.
    fn mixed_sigma_aux_second_order(
        &self,
    ) -> Result<Option<crate::custom_family::ExactNewtonJointPsiSecondOrderTerms>, String>;

    /// Second-order joint-ψ terms for a non-σ derivative-axis pair.
    fn psi_second_order_terms(
        &self,
        psi_i: usize,
        psi_j: usize,
    ) -> Result<Option<crate::custom_family::ExactNewtonJointPsiSecondOrderTerms>, String>;

    /// Direction-contracted second-order ψ terms over the non-σ derivative
    /// axes (#740). `alpha_psi` is the full ψ-block weight vector; the
    /// contraction is against the combined non-σ direction
    /// `ψ(α) = Σ_j alpha_psi[j] · ψ_j`, streaming the family's rows ONCE so the
    /// profiled θ-HVP operator applies one combined-direction n-pass per matvec
    /// instead of `K²` per-pair [`Self::psi_second_order_terms`] passes.
    ///
    /// Default `None` keeps the family on the exact per-pair path. The generic
    /// workspace only calls this when no σ-auxiliary axis carries weight (a σ
    /// term routes the whole direction back to the per-pair fallback), so an
    /// override only handles the pure non-σ derivative axes — the same domain
    /// as [`Self::psi_second_order_terms`].
    fn psi_second_order_terms_contracted(
        &self,
        alpha_psi: &[f64],
    ) -> Result<Option<crate::custom_family::ExactNewtonJointPsiSecondOrderContracted>, String>
    {
        assert!(alpha_psi.len() < usize::MAX);
        Ok(None)
    }

    /// Hessian directional derivative for the σ-auxiliary parameter, returned
    /// as a dense matrix (the generic wraps it into
    /// [`DriftDerivResult::Dense`](crate::solver::estimate::reml::unified::DriftDerivResult::Dense)).
    fn sigma_hessian_directional_derivative(
        &self,
        d_beta_flat: &Array1<f64>,
    ) -> Result<Option<Array2<f64>>, String>;

    /// Hessian directional derivative for a non-σ derivative axis, returned as
    /// a hyper-operator (the generic wraps it into
    /// [`DriftDerivResult::Operator`](crate::solver::estimate::reml::unified::DriftDerivResult::Operator)).
    fn psi_hessian_directional_derivative(
        &self,
        psi_index: usize,
        d_beta_flat: &Array1<f64>,
    ) -> Result<Option<Arc<dyn crate::solver::estimate::reml::unified::HyperOperator>>, String>;
}

/// Generic exact-Newton joint-ψ workspace shared by the marginal-slope
/// families. Owns the σ-auxiliary dispatch skeleton and delegates every
/// family-specific operation to its [`MarginalSlopePsiFamily`] impl.
pub struct MarginalSlopeExactNewtonPsiWorkspace<F: MarginalSlopePsiFamily> {
    family: F,
}

impl<F: MarginalSlopePsiFamily> MarginalSlopeExactNewtonPsiWorkspace<F> {
    pub fn new(family: F) -> Self {
        Self { family }
    }
}

impl<F: MarginalSlopePsiFamily> crate::custom_family::ExactNewtonJointPsiWorkspace
    for MarginalSlopeExactNewtonPsiWorkspace<F>
{
    fn first_order_terms(
        &self,
        psi_index: usize,
    ) -> Result<Option<crate::custom_family::ExactNewtonJointPsiTerms>, String> {
        if self.family.is_sigma_aux(psi_index) {
            return self.family.sigma_first_order_terms();
        }
        self.family.psi_first_order_terms(psi_index)
    }

    fn first_order_terms_all(
        &self,
    ) -> Result<Option<Vec<crate::custom_family::ExactNewtonJointPsiTerms>>, String> {
        self.family.psi_first_order_terms_all()
    }

    fn second_order_terms(
        &self,
        psi_i: usize,
        psi_j: usize,
    ) -> Result<Option<crate::custom_family::ExactNewtonJointPsiSecondOrderTerms>, String> {
        if self.family.is_sigma_aux(psi_i) || self.family.is_sigma_aux(psi_j) {
            if self.family.both_sigma_aux_second_order(psi_i, psi_j) {
                return self.family.sigma_second_order_terms();
            }
            return self.family.mixed_sigma_aux_second_order();
        }
        self.family.psi_second_order_terms(psi_i, psi_j)
    }

    fn second_order_terms_contracted(
        &self,
        alpha_psi: &[f64],
    ) -> Result<Option<crate::custom_family::ExactNewtonJointPsiSecondOrderContracted>, String>
    {
        // The σ-auxiliary axes do not participate in the family's combined
        // non-σ row stream (their second-order terms come from a separate
        // σ/σ and mixed-σ path with no directional row kernel). If any
        // σ-aux axis carries weight in this applied direction, decline the
        // contracted fast path entirely so the caller keeps the exact
        // per-pair assembly — the contracted hook is a representation/cost
        // choice, never an approximation, so falling back is the correct
        // behaviour rather than dropping the σ contribution.
        for (j, &weight) in alpha_psi.iter().enumerate() {
            if weight != 0.0 && self.family.is_sigma_aux(j) {
                return Ok(None);
            }
        }
        self.family.psi_second_order_terms_contracted(alpha_psi)
    }

    fn hessian_directional_derivative(
        &self,
        psi_index: usize,
        d_beta_flat: &Array1<f64>,
    ) -> Result<Option<crate::solver::estimate::reml::unified::DriftDerivResult>, String> {
        if self.family.is_sigma_aux(psi_index) {
            return self
                .family
                .sigma_hessian_directional_derivative(d_beta_flat)
                .map(|result| {
                    result.map(crate::solver::estimate::reml::unified::DriftDerivResult::Dense)
                });
        }
        self.family
            .psi_hessian_directional_derivative(psi_index, d_beta_flat)
            .map(|result| {
                result.map(crate::solver::estimate::reml::unified::DriftDerivResult::Operator)
            })
    }
}

/// Deterministic-order parallel reduction over a row-index slice.
///
/// Splits `rows` into contiguous chunks sized to saturate the rayon pool
/// (several chunks per worker, floored so small `n` stays coarse), processes
/// each chunk sequentially in parallel via `process_row`, and combines the
/// per-chunk accumulators in chunk-index order via `combine` on the calling
/// thread. The chunk count is a pure function of `(rows.len(), worker_count)`;
/// the worker count is fixed for the process (one global pool), so the
/// reduction tree is fixed across calls regardless of rayon's work-stealing
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
    let n = rows.len();
    if n == 0 {
        return Ok(init());
    }
    // The chunk count is sized so the heavy reduction phases actually saturate
    // the rayon pool: a fixed `32` left half of a 64-core box idle whenever the
    // pool had more than 32 workers, capping utilization at ~50% on the biobank
    // coord-corrections / row-stream phases. Targeting several chunks per worker
    // keeps load balanced across an uneven row-cost tail (work-stealing still
    // moves whole chunks, never partial sums) without flooding the sequential
    // `combine` with tiny partials. The count is a pure function of
    // `(rows.len(), worker_count)`; the worker count is fixed for the process
    // (one global pool, gam owns its threads), so chunk boundaries are stable
    // across calls and the ordered `Vec` collect + sequential `combine` keep the
    // reduction bit-for-bit deterministic regardless of work-stealing.
    const CHUNKS_PER_WORKER: usize = 4;
    const MIN_CHUNK_COUNT: usize = 32;
    const MIN_ROWS_PER_CHUNK: usize = 64;
    let workers = rayon::current_num_threads().max(1);
    let target_chunk_count = workers
        .saturating_mul(CHUNKS_PER_WORKER)
        .max(MIN_CHUNK_COUNT);
    // Never carve chunks below `MIN_ROWS_PER_CHUNK` rows: for small `n` the
    // scheduler/partial-accumulator overhead would dominate the row arithmetic.
    let chunk_count = target_chunk_count
        .min(n.div_ceil(MIN_ROWS_PER_CHUNK))
        .max(1);
    let chunk_size = n.div_ceil(chunk_count).max(1);
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

    // ---------------------------------------------------------------------
    // Parity guard for the shared exact-Newton directional sweep.
    //
    // `directional_obj_grad_hess` is the single engine that both the
    // Bernoulli and survival marginal-slope families now route their
    // exact-Newton obj/grad/hess and psi-Hessian directional sweeps
    // through (replacing three hand-rolled, drift-prone loop nests). The
    // test below reconstructs the *reference* loop nest those families used
    // to carry inline and asserts the shared engine reproduces it
    // bit-for-bit on randomized fixtures across both sweep shapes
    // (objective present / suppressed, leading-prefix lengths 1 and 2).
    //
    // The synthetic `eval` mirrors the contract of every family's
    // `row_neglog_directional_with_scale_jet`: it builds one linear
    // `MultiDirJet` per supplied direction at a distinct base, multiplies
    // them together, scales by the per-sweep scale jet, composes through a
    // smooth nonlinearity, and returns the highest mixed-partial
    // coefficient. That makes the appended unit directions genuinely
    // interact (so a transposed Hessian index or a missing symmetric
    // assignment is caught) and makes the scale jet load-bearing (so a
    // mis-wired obj/grad/hess scale is caught).

    use crate::families::jet_partitions::MultiDirJet;

    /// Deterministic LCG so the fixture is reproducible without pulling in
    /// an RNG dependency.
    struct Lcg(u64);
    impl Lcg {
        fn next_f64(&mut self) -> f64 {
            self.0 = self.0.wrapping_mul(6364136223846793005).wrapping_add(1);
            ((self.0 >> 11) as f64) / ((1u64 << 53) as f64) * 2.0 - 1.0
        }
    }

    /// Synthetic row-jet evaluator with the exact contract
    /// `directional_obj_grad_hess` expects: given `dirs` (the leading prefix
    /// plus appended unit directions) and a `scale` jet of matching order,
    /// return the top mixed-partial coefficient of a smooth multilinear
    /// functional of the directions.
    fn synthetic_row_eval(
        bases: &[f64],
        weight: f64,
        dirs: &[&Array1<f64>],
        scale: &MultiDirJet,
    ) -> Result<f64, String> {
        let k = dirs.len();
        if k > 4 {
            return Err(format!("synthetic eval expects 0..=4 directions, got {k}"));
        }
        if scale.coeffs.len() != (1usize << k) {
            return Err(format!(
                "synthetic eval scale jet dimension mismatch: coeffs={}, dirs={k}",
                scale.coeffs.len()
            ));
        }
        let primary_dim = bases.len();
        // One linear jet per direction, each at a distinct base coordinate,
        // mixing the direction's components across the primary dimensions so
        // every Hessian entry is exercised.
        let first = |dir: &Array1<f64>| -> Vec<f64> {
            (0..k).map(|j| dir[j % primary_dim]).collect::<Vec<f64>>()
        };
        let mut product = MultiDirJet::constant(k, 1.0);
        for (slot, dir) in dirs.iter().enumerate() {
            let base = bases[slot % primary_dim] + 0.25 * slot as f64;
            let comps: Vec<f64> = (0..primary_dim)
                .map(|p| dir[p] * (1.0 + 0.5 * p as f64))
                .collect();
            let lin = MultiDirJet::linear(k, base, &first(&Array1::from(comps)));
            product = product.mul(&lin);
        }
        let scaled = product.mul(scale);
        // Smooth nonlinearity φ(x) = weight·ln(1 + x²) composed through the
        // jet — derivs[0..=4] evaluated at the zeroth-order coefficient.
        let x = scaled.coeff(0);
        let denom = 1.0 + x * x;
        let d1 = weight * (2.0 * x) / denom;
        let d2 = weight * (2.0 * (1.0 - x * x)) / (denom * denom);
        let d3 = weight * (-4.0 * x * (3.0 - x * x)) / (denom * denom * denom);
        let d4 = weight * (-12.0 * (1.0 - 6.0 * x * x + x * x * x * x))
            / (denom * denom * denom * denom);
        let phi = weight * denom.ln();
        Ok(scaled
            .compose_unary([phi, d1, d2, d3, d4])
            .coeff((1usize << k) - 1))
    }

    /// Hand-rolled reference sweep — the exact obj/grad/hess loop nest the
    /// families carried inline before the unification onto
    /// `directional_obj_grad_hess`. Kept here purely as the parity oracle.
    fn reference_obj_grad_hess<Eval>(
        primary_dim: usize,
        leading: &[&Array1<f64>],
        scales: &DirectionalScaleJets,
        eval: Eval,
    ) -> Result<(f64, Array1<f64>, Array2<f64>), String>
    where
        Eval: Fn(&[&Array1<f64>], &MultiDirJet) -> Result<f64, String>,
    {
        let unit = |a: usize| -> Array1<f64> {
            let mut da = Array1::<f64>::zeros(primary_dim);
            da[a] = 1.0;
            da
        };
        let objective = if let Some(scale_obj) = scales.obj.as_ref() {
            eval(leading, scale_obj)?
        } else {
            0.0
        };
        let mut grad = Array1::<f64>::zeros(primary_dim);
        for a in 0..primary_dim {
            let da = unit(a);
            let mut dirs: Vec<&Array1<f64>> = leading.to_vec();
            dirs.push(&da);
            grad[a] = eval(&dirs, &scales.grad)?;
        }
        let mut hess = Array2::<f64>::zeros((primary_dim, primary_dim));
        for a in 0..primary_dim {
            let da = unit(a);
            for b in a..primary_dim {
                let db = unit(b);
                let mut dirs: Vec<&Array1<f64>> = leading.to_vec();
                dirs.push(&da);
                dirs.push(&db);
                let value = eval(&dirs, &scales.hess)?;
                hess[[a, b]] = value;
                hess[[b, a]] = value;
            }
        }
        Ok((objective, grad, hess))
    }

    /// Build a scale jet of the requested order with random first/second
    /// mixed coefficients on the supplied masks — mirrors the structure of
    /// each family's `sigma_scale_jet` (a base plus first-order entries on
    /// the leading log-sigma slots and a second-order entry on the pair).
    fn random_scale_jet(
        rng: &mut Lcg,
        n_dirs: usize,
        first_masks: &[usize],
        second_masks: &[usize],
    ) -> MultiDirJet {
        let mut coeffs: Vec<(usize, f64)> = vec![(0usize, 1.0 + 0.1 * rng.next_f64())];
        for &m in first_masks {
            coeffs.push((1usize << m, rng.next_f64()));
        }
        for &m in second_masks {
            coeffs.push(((1usize << m) | 1usize, rng.next_f64()));
        }
        MultiDirJet::with_coeffs(n_dirs, &coeffs)
    }

    #[test]
    fn directional_obj_grad_hess_matches_reference_loop_nest() {
        let primary_dim = 4usize;
        let mut rng = Lcg(0x5EED_1234_ABCD_0001);
        // Sweep both family shapes: first-order log-sigma (leading=[zero],
        // obj present), second-order (leading=[zero,zero], obj present), and
        // the psi-Hessian directional (leading=[zero,row_dir], obj absent).
        for trial in 0..32 {
            let bases: Vec<f64> = (0..primary_dim).map(|_| rng.next_f64()).collect();
            let weight = 0.5 + 0.5 * (rng.next_f64() + 1.0);
            let eval = |dirs: &[&Array1<f64>], scale: &MultiDirJet| {
                synthetic_row_eval(&bases, weight, dirs, scale)
            };

            let zero = Array1::<f64>::zeros(primary_dim);
            let row_dir: Array1<f64> =
                Array1::from((0..primary_dim).map(|_| rng.next_f64()).collect::<Vec<_>>());

            let cases: Vec<(Vec<&Array1<f64>>, DirectionalScaleJets)> = vec![
                (
                    vec![&zero],
                    DirectionalScaleJets {
                        obj: Some(random_scale_jet(&mut rng, 1, &[], &[])),
                        grad: random_scale_jet(&mut rng, 2, &[0], &[]),
                        hess: random_scale_jet(&mut rng, 3, &[0], &[]),
                    },
                ),
                (
                    vec![&zero, &zero],
                    DirectionalScaleJets {
                        obj: Some(random_scale_jet(&mut rng, 2, &[0, 1], &[])),
                        grad: random_scale_jet(&mut rng, 3, &[0, 1], &[]),
                        hess: random_scale_jet(&mut rng, 4, &[0, 1], &[]),
                    },
                ),
                (
                    vec![&zero, &row_dir],
                    DirectionalScaleJets {
                        obj: None,
                        grad: random_scale_jet(&mut rng, 3, &[0], &[]),
                        hess: random_scale_jet(&mut rng, 4, &[0], &[]),
                    },
                ),
            ];

            for (leading, scales) in &cases {
                let shared =
                    directional_obj_grad_hess(primary_dim, leading, scales, eval).expect("shared");
                let (ref_obj, ref_grad, ref_hess) =
                    reference_obj_grad_hess(primary_dim, leading, scales, eval).expect("reference");

                assert_eq!(
                    shared.objective, ref_obj,
                    "trial {trial}: objective drift {} vs {}",
                    shared.objective, ref_obj
                );
                for a in 0..primary_dim {
                    assert_eq!(
                        shared.grad[a], ref_grad[a],
                        "trial {trial}: grad[{a}] drift {} vs {}",
                        shared.grad[a], ref_grad[a]
                    );
                    for b in 0..primary_dim {
                        assert_eq!(
                            shared.hess[[a, b]],
                            ref_hess[[a, b]],
                            "trial {trial}: hess[{a},{b}] drift {} vs {}",
                            shared.hess[[a, b]],
                            ref_hess[[a, b]]
                        );
                    }
                }
                // The Hessian the engine returns must be exactly symmetric —
                // a transposed write in the upper-triangle loop is the classic
                // exact-Newton drift bug.
                for a in 0..primary_dim {
                    for b in 0..primary_dim {
                        assert_eq!(
                            shared.hess[[a, b]],
                            shared.hess[[b, a]],
                            "trial {trial}: hess asymmetric at ({a},{b})"
                        );
                    }
                }
            }
        }
    }

    #[test]
    fn auto_outer_score_subsample_skips_small_problems() {
        let n = 1000;
        let z: Vec<f64> = (0..n).map(|i| i as f64).collect();
        let opts = AutoOuterSubsampleOptions::default();
        assert!(
            auto_outer_score_subsample(&z, None, &opts).is_none(),
            "n={n} below default min_n_for_auto=30000 should not subsample"
        );
    }

    #[test]
    fn auto_outer_score_subsample_returns_target_k_above_threshold() {
        let n = 60_000;
        let z: Vec<f64> = (0..n).map(|i| (i as f64).sin()).collect();
        let opts = AutoOuterSubsampleOptions::default();
        let mask = auto_outer_score_subsample(&z, None, &opts)
            .expect("n=60000 should auto-subsample with default options");
        // Default target_fraction=0.10 and min_k=10000 → K = max(10000, 6000) = 10000.
        assert_eq!(mask.n_full, n);
        assert!(
            mask.len() >= 9_900 && mask.len() <= 10_200,
            "expected K≈10_000, got {}",
            mask.len()
        );
        // HT weights should reconstruct n_full in expectation: sum of
        // per-row weights ≈ n_full (allowing for small allocation rounding).
        let weight_sum: f64 = mask.rows.iter().map(|r| r.weight).sum();
        let rel_err = (weight_sum - n as f64).abs() / n as f64;
        assert!(
            rel_err < 0.02,
            "HT weight sum {weight_sum:.3} should ≈ n_full={n}, rel_err={rel_err:.4}"
        );
    }

    #[test]
    fn auto_outer_score_subsample_horvitz_thompson_unbiased() {
        // On a synthetic per-row contribution `t_i = z_i² + 1`, verify
        // the HT-weighted sum over the auto-mask matches the full sum
        // within 3 standard deviations of the predicted estimator
        // variance. This guards against silent regressions in either
        // the stratified mask construction or the weight assignment.
        let n = 50_000;
        let z: Vec<f64> = (0..n)
            .map(|i| ((i as f64) / n as f64) * 2.0 - 1.0)
            .collect();
        let stratum: Vec<u8> = (0..n).map(|i| if i % 3 == 0 { 1 } else { 0 }).collect();
        let opts = AutoOuterSubsampleOptions {
            seed: 0xC0FFEE,
            ..AutoOuterSubsampleOptions::default()
        };
        let t: Vec<f64> = z.iter().map(|zi| zi * zi + 1.0).collect();
        let exact: f64 = t.iter().sum();
        let mask = auto_outer_score_subsample(&z, Some(&stratum), &opts)
            .expect("n=50000 should auto-subsample");
        let estimate: f64 = mask.rows.iter().map(|r| r.weight * t[r.index]).sum();
        // Predicted standard error: σ ≈ (1/√K) · √(1 − K/N) · cv · |T|.
        // For t_i ∈ [1, 2], cv ≲ 0.4. Be generous (factor 5) to keep
        // the test robust against PRNG-dependent allocation jitter.
        let k = mask.len();
        let predicted_se =
            exact * 0.4 * (1.0 / (k as f64).sqrt()) * (1.0 - k as f64 / n as f64).sqrt();
        let observed_err = (estimate - exact).abs();
        assert!(
            observed_err < 5.0 * predicted_se.max(1.0),
            "HT estimate {estimate:.3} vs exact {exact:.3}: err={observed_err:.3} exceeds 5×predicted_se={:.3}",
            predicted_se
        );
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
        let mut covered = [false; 200];
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
