//! The width-independent residual repair row kernel (gam#2924).
//!
//! # Five primaries, whatever the width
//!
//! The row likelihood of [`residual_row_nll`] reads the residual coefficients
//! `β` only through the row-varying read `t_i = βᵀr_i` and the anchor's two
//! moments of the drive, `u_i = βᵀγ(a_i)` and `v_i = βᵀΣ_rr(a_i)β`, with
//! `γ = Σ_r0` the first column of the declared joint covariance (the same at
//! every row under the pooled law). So the row program has the five primaries
//! `p_i = (η_m, g, t, u, v)` for any block width `K`, and its jets never grow
//! with `K`.
//!
//! # The chain rule through one quadratic map
//!
//! `t` and `u` are linear in `β` — a design row `r_i` and a coefficient row
//! `γ(a_i)` — and `v` is quadratic, with the constant second derivative
//! `D²v[d, e] = 2·d_rᵀΣ_rr e_r` and nothing above it. The coefficient-space
//! channels are therefore the generic pullback through the tangent Jacobian
//! `J_i = ∂p_i/∂θ`, whose `v` row is `2Σ_rr(a_i)β`, plus the terms in which the
//! curvature of `v` absorbs directions. Faà di Bruno with every block of size at
//! most two, writing `μ_d = 2Σ_rr d_r`, `h`/`T³`/`T⁴` the row's primary-space
//! derivatives and `h[v,·]` a row of `h`:
//!
//! ```text
//!   H        = Σ_i J_iᵀ h J_i + h_v·2Σ_rr
//!   H′[d]    = Σ_i J_iᵀ T³[Jd] J_i + μ_d ⊗ J_iᵀh[v,·] + J_iᵀh[v,·] ⊗ μ_d
//!              + (h·Jd)_v·2Σ_rr
//!   H″[d, e] = Σ_i J_iᵀ T⁴[Jd, Je] J_i + (d_rᵀμ_e)·J_iᵀ T³[e_v] J_i
//!              + μ_d ⊗ J_iᵀT³[Je][v,·] + its transpose + μ_e ⊗ J_iᵀT³[Jd][v,·] + its transpose
//!              + (T³[Jd]·Je)_v·2Σ_rr + (d_rᵀμ_e)·h_vv·2Σ_rr + h_vv·(μ_d ⊗ μ_e + μ_e ⊗ μ_d)
//! ```
//!
//! The residual block of a pullback is `O(K²)` per row and each row runs a fixed
//! number of five-primary jets, so a biobank block score with `26` or more block
//! columns costs what the rigid kernel costs plus `O(nK²)` assembly. There is no
//! width ceiling.
//!
//! # One curvature under the pooled law
//!
//! When the declared joint covariance is the same at every row, `2Σ_rr` and
//! each `μ_d` are row-invariant. Every term above that carries them is then a
//! row sum of scalars or coefficient vectors times one fixed block, so the
//! assemblies accumulate those sums and apply `2Σ_rr` and the rank-two updates
//! once per sweep instead of once per row.

use super::family::*;
use super::hessian_paths::BlockSlices;
use super::residual_repair::{ResidualBlockRuntime, ResidualDrive, ResidualRowState, residual_row_nll};
use super::*;
use crate::row_kernel::{RowKernel, RowKernelCache, RowSet};
use gam_math::jet_scalar::{JetScalar, SymmetricQuadraticCoefficients};
use gam_math::jet_tower::RowProgram;
use rayon::prelude::*;
use std::borrow::Cow;
use std::ops::Range;

/// Primary index of the anchor's second drive moment `v = βᵀΣ_rrβ`.
const V: usize = 4;

/// The anchor's moments of the residual drive under one row's declared joint
/// covariance: `γ = Σ_r0`, `Σ_rrβ`, `u = γᵀβ` and `v = βᵀΣ_rrβ`.
#[derive(Clone, Debug)]
pub(super) struct DriveMoments {
    gamma: Vec<f64>,
    sigma_beta: Vec<f64>,
    u: f64,
    v: f64,
}

impl DriveMoments {
    pub(super) fn at(covariance: &MarginalSlopeCovariance, beta: &[f64]) -> Self {
        let (sigma_beta, u, v) = Self::drive_products(covariance, beta);
        Self {
            gamma: (0..beta.len()).map(|j| covariance.coefficient(0, j + 1)).collect(),
            sigma_beta,
            u,
            v,
        }
    }

    /// The `β`-dependent moments alone, `(Σ_rrβ, u, v)`, from one product with
    /// the joint covariance; [`Self::at`] adds `γ`, which does not depend on `β`.
    fn drive_products(covariance: &MarginalSlopeCovariance, beta: &[f64]) -> (Vec<f64>, f64, f64) {
        let k = beta.len();
        let mut lifted = vec![0.0_f64; k + 1];
        lifted[1..].copy_from_slice(beta);
        let mut image = vec![0.0_f64; k + 1];
        covariance.multiply(&lifted, &mut image);
        let sigma_beta = image[1..].to_vec();
        let v = beta.iter().zip(&sigma_beta).map(|(b, x)| b * x).sum();
        (sigma_beta, image[0], v)
    }
}

/// What the coefficient-space assembly needs beyond [`RowKernel`]: where the
/// residual coordinates sit, the residual rows of each row Jacobian, and the
/// curvature `D²v = 2Σ_rr(a_i)` of the one quadratic primary.
pub(super) trait ResidualDriveRows: RowKernel<5> {
    /// The residual coefficient range inside the flat coefficient vector.
    fn residual_range(&self) -> Range<usize>;

    /// The residual rows `(r_i, γ(a_i), 2Σ_rr(a_i)β)` of `J_i`: the `t`, `u` and
    /// `v` rows restricted to the residual coordinates.
    fn residual_jacobian_rows(&self, row: usize) -> [Vec<f64>; 3];

    /// `μ = 2Σ_rr(a_i)·d_r` for the residual part of a coefficient direction.
    fn curvature_action(&self, row: usize, d_r: &[f64]) -> Vec<f64>;

    /// `target[res, res] += scale·2Σ_rr(a_i)`.
    fn add_curvature(&self, row: usize, scale: f64, target: &mut Array2<f64>);

    /// `2Σ_rr` as a dense row-major `K×K` block when the declared joint
    /// covariance is the same at every row, so an assembly can apply it once
    /// from row sums; `None` when it varies by row.
    fn row_invariant_curvature(&self) -> Option<&[f64]> {
        None
    }
}

/// `2Σ_rr` of a joint covariance over `(z, r)` as a dense row-major `K×K` block:
/// the residual rows and columns, read once through the covariance's own
/// representation.
fn doubled_residual_block(covariance: &MarginalSlopeCovariance) -> Vec<f64> {
    let k = covariance.dim() - 1;
    let mut block = vec![0.0_f64; k * k];
    match covariance.representation() {
        MarginalSlopeCovarianceRef::Diagonal(diagonal) => {
            for j in 0..k {
                block[j * k + j] = 2.0 * diagonal[j + 1];
            }
        }
        MarginalSlopeCovarianceRef::Full(matrix) => {
            for (j, row) in block.chunks_exact_mut(k).enumerate() {
                for (entry, &value) in row.iter_mut().zip(matrix.row(j + 1).iter().skip(1)) {
                    *entry = 2.0 * value;
                }
            }
        }
        MarginalSlopeCovarianceRef::LowRank(factor) => {
            for (j, row) in block.chunks_exact_mut(k).enumerate() {
                let left = factor.row(j + 1);
                for (l, entry) in row.iter_mut().enumerate() {
                    let right = factor.row(l + 1);
                    *entry = 2.0 * left.iter().zip(right.iter()).map(|(a, b)| a * b).sum::<f64>();
                }
            }
        }
    }
    block
}

/// `2Σ_rr·d_r` from the dense row-major block `curvature = 2Σ_rr`.
fn invariant_curvature_action(curvature: &[f64], d_r: &[f64]) -> Vec<f64> {
    curvature.chunks_exact(d_r.len()).map(|row| dot(row, d_r)).collect()
}

/// `target[res, res] += scale·curvature` for the dense row-major block `curvature`.
fn add_invariant_curvature(target: &mut Array2<f64>, residual: &Range<usize>, curvature: &[f64], scale: f64) {
    if scale == 0.0 {
        return;
    }
    let mut block = target.slice_mut(s![residual.clone(), residual.clone()]);
    for (mut row, source) in block.rows_mut().into_iter().zip(curvature.chunks_exact(residual.len())) {
        for (entry, &value) in row.iter_mut().zip(source) {
            *entry += scale * value;
        }
    }
}

/// `target[res, res] += scale·(μ_d ⊗ μ_e + μ_e ⊗ μ_d)`.
fn add_symmetric_outer(target: &mut Array2<f64>, residual: &Range<usize>, mu_d: &[f64], mu_e: &[f64], scale: f64) {
    if scale == 0.0 {
        return;
    }
    for (j, (&dj, &ej)) in mu_d.iter().zip(mu_e).enumerate() {
        for (l, (&dl, &el)) in mu_d.iter().zip(mu_e).enumerate() {
            target[[residual.start + j, residual.start + l]] += scale * (dj * el + ej * dl);
        }
    }
}

/// `out += scale·J_iᵀ·v`.
fn add_transpose_vector(
    kern: &(impl ResidualDriveRows + ?Sized),
    row: usize,
    v: &[f64; 5],
    scale: f64,
    out: &mut [f64],
) {
    let scaled_v: [f64; 5] = std::array::from_fn(|a| scale * v[a]);
    kern.jacobian_transpose_action(row, &scaled_v, out);
}

/// Elementwise `a + b` for two row-sum vectors of one length, either possibly
/// empty because the sweep did not need it.
fn add_row_sums(mut a: Vec<f64>, b: Vec<f64>) -> Vec<f64> {
    for (x, y) in a.iter_mut().zip(&b) {
        *x += y;
    }
    a
}

#[inline]
fn dot(a: &[f64], b: &[f64]) -> f64 {
    a.iter().zip(b).map(|(x, y)| x * y).sum()
}

#[inline]
fn scaled(h: &[[f64; 5]; 5], w: f64) -> [[f64; 5]; 5] {
    if w == 1.0 {
        return *h;
    }
    std::array::from_fn(|a| std::array::from_fn(|b| w * h[a][b]))
}

/// The residual parts `(rᵀd_r, γᵀd_r, 2(Σ_rrβ)ᵀd_r)` of `J_i·d`.
#[inline]
fn residual_action(rho: &[Vec<f64>; 3], d_r: &[f64]) -> [f64; 3] {
    [dot(&rho[0], d_r), dot(&rho[1], d_r), dot(&rho[2], d_r)]
}

/// `out_r += v_t·r_i + v_u·γ + v_v·2Σ_rrβ`.
#[inline]
fn residual_transpose_action(rho: &[Vec<f64>; 3], v: &[f64; 3], out_r: &mut [f64]) {
    for (j, out) in out_r.iter_mut().enumerate() {
        *out += v[0] * rho[0][j] + v[1] * rho[1][j] + v[2] * rho[2][j];
    }
}

/// The residual coordinates of `J_iᵀ·h·J_i`: the residual block is added to
/// `target_rr`; the residual columns of the two surface rows come back as the
/// K-vectors `c_q = Σ_b h[q][2+b]·ρ_b` and `c_g = Σ_b h[g][2+b]·ρ_b`, which the
/// caller scatters through its designs.
fn pullback_residual_block(
    h: &[[f64; 5]; 5],
    rho: &[Vec<f64>; 3],
    mut target_rr: ndarray::ArrayViewMut2<'_, f64>,
) -> [Vec<f64>; 2] {
    let k = rho[0].len();
    let mut projected = [vec![0.0_f64; k], vec![0.0_f64; k], vec![0.0_f64; k]];
    for (a, image) in projected.iter_mut().enumerate() {
        for b in 0..3 {
            let coefficient = h[2 + a][2 + b];
            if coefficient != 0.0 {
                for (x, r) in image.iter_mut().zip(&rho[b]) {
                    *x += coefficient * r;
                }
            }
        }
    }
    for j in 0..k {
        let (r0, r1, r2) = (rho[0][j], rho[1][j], rho[2][j]);
        let mut row = target_rr.row_mut(j);
        for l in 0..k {
            row[l] += r0 * projected[0][l] + r1 * projected[1][l] + r2 * projected[2][l];
        }
    }
    let surface = |primary: usize| -> Vec<f64> {
        (0..k)
            .map(|j| h[primary][2] * rho[0][j] + h[primary][3] * rho[1][j] + h[primary][4] * rho[2][j])
            .collect()
    };
    [surface(0), surface(1)]
}

/// `acc[res+j, c] += w·μ_j·g_c` and `acc[c, res+j] += w·g_c·μ_j`.
fn add_rank_two(acc: &mut Array2<f64>, residual_start: usize, mu: &[f64], g: &[f64], w: f64) {
    for (j, &m) in mu.iter().enumerate() {
        let scale = w * m;
        if scale == 0.0 {
            continue;
        }
        for (c, &gc) in g.iter().enumerate() {
            let value = scale * gc;
            acc[[residual_start + j, c]] += value;
            acc[[c, residual_start + j]] += value;
        }
    }
}

/// `J_iᵀ·v` as a flat coefficient vector.
fn transpose_vector(kern: &(impl ResidualDriveRows + ?Sized), row: usize, v: &[f64; 5]) -> Vec<f64> {
    let mut out = vec![0.0_f64; kern.n_coefficients()];
    kern.jacobian_transpose_action(row, v, &mut out);
    out
}

/// The negative log-likelihood Hessian in coefficient space from a row cache:
/// `Σ_i J_iᵀ h_i J_i + h_{i,v}·2Σ_rr(a_i)`.
pub(super) fn residual_hessian_dense(
    kern: &(impl ResidualDriveRows + ?Sized),
    cache: &RowKernelCache<5>,
) -> Result<Array2<f64>, String> {
    let n = kern.n_rows();
    let p = kern.n_coefficients();
    let residual = kern.residual_range();
    let invariant = kern.row_invariant_curvature();
    let (mut hessian, curvature_weight) = RowSet::All.par_try_reduce_fold(
        n,
        || (Array2::<f64>::zeros((p, p)), 0.0_f64),
        |(mut acc, mut curvature_weight), row, w| -> Result<_, String> {
            kern.add_pullback_hessian(row, &scaled(&cache.hessians[row], w), &mut acc);
            let scale = w * cache.gradients[row][V];
            if invariant.is_some() {
                curvature_weight += scale;
            } else {
                kern.add_curvature(row, scale, &mut acc);
            }
            Ok((acc, curvature_weight))
        },
        |(a, wa), (b, wb)| Ok((a + b, wa + wb)),
    )?;
    if let Some(curvature) = invariant {
        add_invariant_curvature(&mut hessian, &residual, curvature, curvature_weight);
    }
    Ok(hessian)
}

/// `H′[d] = ∂H/∂θ[d]` in coefficient space (see the module docs).
pub(super) fn residual_hessian_directional_derivative(
    kern: &(impl ResidualDriveRows + ?Sized),
    d_beta: &[f64],
) -> Result<Array2<f64>, String> {
    let n = kern.n_rows();
    let p = kern.n_coefficients();
    if d_beta.len() != p {
        return Err(format!(
            "residual repair directional derivative: direction has {} entries, expected {p}",
            d_beta.len()
        ));
    }
    let residual = kern.residual_range();
    let invariant = kern.row_invariant_curvature();
    let invariant_mu_d = invariant.map(|curvature| invariant_curvature_action(curvature, &d_beta[residual.clone()]));
    let sums_len = if invariant.is_some() { p } else { 0 };
    let (mut derivative, h_v_sum, curvature_weight) = RowSet::All.par_try_reduce_fold(
        n,
        || (Array2::<f64>::zeros((p, p)), vec![0.0_f64; sums_len], 0.0_f64),
        |(mut acc, mut h_v_sum, mut curvature_weight), row, w| -> Result<_, String> {
            let jd = kern.jacobian_action(row, d_beta);
            let (_, _, h) = kern.row_kernel(row)?;
            let third = kern.row_third_contracted(row, &jd)?;
            kern.add_pullback_hessian(row, &scaled(&third, w), &mut acc);
            let scale = w * dot(&h[V], &jd);
            if invariant.is_some() {
                add_transpose_vector(kern, row, &h[V], w, &mut h_v_sum);
                curvature_weight += scale;
            } else {
                let mu_d = kern.curvature_action(row, &d_beta[residual.clone()]);
                let h_v = transpose_vector(kern, row, &h[V]);
                add_rank_two(&mut acc, residual.start, &mu_d, &h_v, w);
                kern.add_curvature(row, scale, &mut acc);
            }
            Ok((acc, h_v_sum, curvature_weight))
        },
        |(a, ha, wa), (b, hb, wb)| Ok((a + b, add_row_sums(ha, hb), wa + wb)),
    )?;
    if let (Some(curvature), Some(mu_d)) = (invariant, invariant_mu_d.as_deref()) {
        add_rank_two(&mut derivative, residual.start, mu_d, &h_v_sum, 1.0);
        add_invariant_curvature(&mut derivative, &residual, curvature, curvature_weight);
    }
    Ok(derivative)
}

/// `H″[d, e] = ∂²H/∂θ²[d, e]` in coefficient space (see the module docs).
pub(super) fn residual_hessian_second_directional_derivative(
    kern: &(impl ResidualDriveRows + ?Sized),
    d_beta: &[f64],
    e_beta: &[f64],
) -> Result<Array2<f64>, String> {
    let n = kern.n_rows();
    let p = kern.n_coefficients();
    if d_beta.len() != p || e_beta.len() != p {
        return Err(format!(
            "residual repair second directional derivative: directions have {} / {} entries, \
             expected {p}",
            d_beta.len(),
            e_beta.len()
        ));
    }
    let residual = kern.residual_range();
    let unit_v: [f64; 5] = std::array::from_fn(|a| if a == V { 1.0 } else { 0.0 });
    let invariant = kern.row_invariant_curvature();
    let invariant_mu = invariant.map(|curvature| {
        (
            invariant_curvature_action(curvature, &d_beta[residual.clone()]),
            invariant_curvature_action(curvature, &e_beta[residual.clone()]),
        )
    });
    let sums_len = if invariant.is_some() { p } else { 0 };
    // Under a row-invariant covariance the fold carries, besides the matrix,
    // `Σ w·J_iᵀT³[Je][v,·]`, `Σ w·J_iᵀT³[Jd][v,·]`, the curvature weight of
    // `2Σ_rr` and `Σ w·h_vv`.
    let (mut derivative, third_e_sum, third_d_sum, curvature_weight, h_vv_sum) = RowSet::All.par_try_reduce_fold(
        n,
        || {
            (
                Array2::<f64>::zeros((p, p)),
                vec![0.0_f64; sums_len],
                vec![0.0_f64; sums_len],
                0.0_f64,
                0.0_f64,
            )
        },
        |(mut acc, mut third_e_sum, mut third_d_sum, mut curvature_weight, mut h_vv_sum), row, w| -> Result<_, String> {
            let jd = kern.jacobian_action(row, d_beta);
            let je = kern.jacobian_action(row, e_beta);
            let (_, _, h) = kern.row_kernel(row)?;
            let fourth = kern.row_fourth_contracted(row, &jd, &je)?;
            kern.add_pullback_hessian(row, &scaled(&fourth, w), &mut acc);
            let third_d = kern.row_third_contracted(row, &jd)?;
            let third_e = kern.row_third_contracted(row, &je)?;
            let (mu_d, mu_e) = match &invariant_mu {
                Some((mu_d, mu_e)) => (Cow::Borrowed(mu_d), Cow::Borrowed(mu_e)),
                None => (
                    Cow::Owned(kern.curvature_action(row, &d_beta[residual.clone()])),
                    Cow::Owned(kern.curvature_action(row, &e_beta[residual.clone()])),
                ),
            };
            let kappa = dot(&d_beta[residual.clone()], &mu_e);
            if kappa != 0.0 {
                let third_v = kern.row_third_contracted(row, &unit_v)?;
                kern.add_pullback_hessian(row, &scaled(&third_v, w * kappa), &mut acc);
            }
            let scale = w * (dot(&third_d[V], &je) + kappa * h[V][V]);
            let h_vv = w * h[V][V];
            if invariant.is_some() {
                add_transpose_vector(kern, row, &third_e[V], w, &mut third_e_sum);
                add_transpose_vector(kern, row, &third_d[V], w, &mut third_d_sum);
                curvature_weight += scale;
                h_vv_sum += h_vv;
            } else {
                add_rank_two(&mut acc, residual.start, &mu_d, &transpose_vector(kern, row, &third_e[V]), w);
                add_rank_two(&mut acc, residual.start, &mu_e, &transpose_vector(kern, row, &third_d[V]), w);
                kern.add_curvature(row, scale, &mut acc);
                add_symmetric_outer(&mut acc, &residual, &mu_d, &mu_e, h_vv);
            }
            Ok((acc, third_e_sum, third_d_sum, curvature_weight, h_vv_sum))
        },
        |(a, ea, da, wa, va), (b, eb, db, wb, vb)| {
            Ok((a + b, add_row_sums(ea, eb), add_row_sums(da, db), wa + wb, va + vb))
        },
    )?;
    if let (Some(curvature), Some((mu_d, mu_e))) = (invariant, invariant_mu.as_ref()) {
        add_rank_two(&mut derivative, residual.start, mu_d, &third_e_sum, 1.0);
        add_rank_two(&mut derivative, residual.start, mu_e, &third_d_sum, 1.0);
        add_invariant_curvature(&mut derivative, &residual, curvature, curvature_weight);
        add_symmetric_outer(&mut derivative, &residual, mu_d, mu_e, h_vv_sum);
    }
    Ok(derivative)
}

/// `{H′[e_a]}_{a=0..p}`: one independent full-data sweep per coefficient axis.
pub(super) fn residual_hessian_directional_derivative_all_axes(
    kern: &(impl ResidualDriveRows + ?Sized + Sync),
) -> Result<Vec<Array2<f64>>, String> {
    let p = kern.n_coefficients();
    (0..p)
        .into_par_iter()
        .map(|a| {
            let mut axis = vec![0.0_f64; p];
            axis[a] = 1.0;
            gam_problem::with_nested_parallel(|| residual_hessian_directional_derivative(kern, &axis))
        })
        .collect()
}

/// `{H″[u, e_a]}_{a=0..p}`: one independent full-data sweep per coefficient axis.
pub(super) fn residual_hessian_second_directional_derivative_all_axes(
    kern: &(impl ResidualDriveRows + ?Sized + Sync),
    d_beta_u: &[f64],
) -> Result<Vec<Array2<f64>>, String> {
    let p = kern.n_coefficients();
    (0..p)
        .into_par_iter()
        .map(|a| {
            let mut axis = vec![0.0_f64; p];
            axis[a] = 1.0;
            gam_problem::with_nested_parallel(|| {
                residual_hessian_second_directional_derivative(kern, d_beta_u, &axis)
            })
        })
        .collect()
}

/// One row's local geometry in the coordinates `(η_m, g, β_1, …, β_K)`: the
/// five-primary channels pulled back through `J̃_i` (identity on the two
/// surfaces, the residual rows on `β`) plus the curvature of `v`. This is what
/// the saved-H ALO replay reads as a row's score and observed Hessian.
pub(super) fn residual_row_geometry(
    kern: &(impl ResidualDriveRows + ?Sized),
    row: usize,
) -> Result<(f64, Array1<f64>, Array2<f64>), String> {
    let (nll, g, h) = kern.row_kernel(row)?;
    let rho = kern.residual_jacobian_rows(row);
    let k = rho[0].len();
    let width = 2 + k;
    let mut score = Array1::<f64>::zeros(width);
    score[0] = g[0];
    score[1] = g[1];
    residual_transpose_action(
        &rho,
        &[g[2], g[3], g[4]],
        &mut score.as_slice_mut().ok_or("residual row geometry: non-contiguous score")?[2..],
    );
    let mut hessian = Array2::<f64>::zeros((width, width));
    hessian[[0, 0]] = h[0][0];
    hessian[[0, 1]] = h[0][1];
    hessian[[1, 0]] = h[1][0];
    hessian[[1, 1]] = h[1][1];
    let [c_q, c_g] = pullback_residual_block(&h, &rho, hessian.slice_mut(s![2.., 2..]));
    for j in 0..k {
        hessian[[0, 2 + j]] = c_q[j];
        hessian[[2 + j, 0]] = c_q[j];
        hessian[[1, 2 + j]] = c_g[j];
        hessian[[2 + j, 1]] = c_g[j];
    }
    // `add_curvature` addresses the kernel's flat coefficient layout; lift the
    // residual block out of it and back.
    let residual = kern.residual_range();
    let mut curvature = Array2::<f64>::zeros((residual.end, residual.end));
    kern.add_curvature(row, g[V], &mut curvature);
    hessian
        .slice_mut(s![2.., 2..])
        .scaled_add(1.0, &curvature.slice(s![residual.clone(), residual]));
    Ok((nll, score, hessian))
}

/// Each row's `γ(a_i)` and lower-triangular `2Σ_rr(a_i)` under a row-varying law.
/// Neither depends on `β`, so they are built once per fit and shared by every
/// kernel over it, instead of re-derived from the row's factor on every call.
/// The stored values are the per-call values themselves
/// ([`DriveMoments::at`]'s `γ`, [`doubled_residual_block`]), so every result is
/// bitwise the per-call result.
pub(crate) struct RowCovariance {
    width: usize,
    /// Row-major `n × K`: `γ(a_i)`.
    gamma: Vec<f64>,
    /// Row-major `n × K(K+1)/2`: the lower triangle of `2Σ_rr(a_i)` by rows.
    curvature: Vec<f64>,
    /// The governor's charge for the storage above, held while it is live.
    _reservation: gam_runtime::resource::MemoryReservation,
}

impl RowCovariance {
    /// `f64` entries per row: `K` for `γ` and `K(K+1)/2` for the curvature.
    fn entries_per_row(width: usize) -> usize {
        width + width * (width + 1) / 2
    }

    /// `None` when some row's `2Σ_rr` is not bitwise symmetric, so its lower
    /// triangle would not reproduce the upper one. Every row's is for the
    /// low-rank rows a conditional law stores, whose entries are sums of the same
    /// products in the same order.
    fn build(
        field: &super::conditional_score_covariance::ScoreCovarianceField,
        width: usize,
        rows: usize,
        reservation: gam_runtime::resource::MemoryReservation,
    ) -> Option<Self> {
        let k = width;
        let packed = k * (k + 1) / 2;
        let mut gamma = vec![0.0_f64; rows * k];
        let mut curvature = vec![0.0_f64; rows * packed];
        let symmetric = gamma
            .par_chunks_mut(k)
            .zip(curvature.par_chunks_mut(packed))
            .enumerate()
            .map(|(row, (gamma_row, curvature_row))| {
                let covariance = field.at_row(row);
                for (j, entry) in gamma_row.iter_mut().enumerate() {
                    *entry = covariance.coefficient(0, j + 1);
                }
                let block = doubled_residual_block(covariance);
                let mut at = 0;
                for j in 0..k {
                    curvature_row[at..at + j + 1].copy_from_slice(&block[j * k..j * k + j + 1]);
                    at += j + 1;
                }
                (0..k).all(|j| (0..j).all(|l| block[j * k + l].to_bits() == block[l * k + j].to_bits()))
            })
            .reduce(|| true, |a, b| a && b);
        symmetric.then_some(Self {
            width: k,
            gamma,
            curvature,
            _reservation: reservation,
        })
    }

    #[inline]
    fn gamma_row(&self, row: usize) -> &[f64] {
        &self.gamma[row * self.width..(row + 1) * self.width]
    }

    #[inline]
    fn curvature_row(&self, row: usize) -> &[f64] {
        let packed = self.width * (self.width + 1) / 2;
        &self.curvature[row * packed..(row + 1) * packed]
    }

    /// `2Σ_rr(a_i)[j][l]` from a row's lower triangle.
    #[inline]
    fn packed_entry(row: &[f64], j: usize, l: usize) -> f64 {
        let (high, low) = if j >= l { (j, l) } else { (l, j) };
        row[high * (high + 1) / 2 + low]
    }
}

/// Where a [`ResidualBlockRuntime`] keeps its [`RowCovariance`]: built on the
/// first kernel over the fit, then shared.
#[derive(Default)]
pub(crate) struct RowCovarianceSlot(std::sync::OnceLock<Option<RowCovariance>>);

impl RowCovarianceSlot {
    /// The fit's row covariance, built on first use under `reserve`'s grant of
    /// its `rows × columns` `f64` storage; `None` when refused or not bitwise
    /// symmetric, which every later kernel over the fit then shares.
    fn get_or_build(
        &self,
        field: &super::conditional_score_covariance::ScoreCovarianceField,
        width: usize,
        rows: usize,
        reserve: &dyn Fn(usize, usize) -> Option<gam_runtime::resource::MemoryReservation>,
    ) -> Option<&RowCovariance> {
        self.0
            .get_or_init(|| {
                reserve(rows, RowCovariance::entries_per_row(width))
                    .and_then(|reservation| RowCovariance::build(field, width, rows, reservation))
            })
            .as_ref()
    }
}

/// Each row's `Σ_rr(a_i)β`, `u_i` and `v_i` for one kernel's `β`, computed once
/// with [`DriveMoments::at`]'s own arithmetic.
struct RowMoments {
    width: usize,
    /// Row-major `n × K`.
    sigma_beta: Vec<f64>,
    u: Vec<f64>,
    v: Vec<f64>,
    _reservation: gam_runtime::resource::MemoryReservation,
}

impl RowMoments {
    /// `f64` entries per row: `K` for `Σ_rrβ`, one each for `u` and `v`.
    fn entries_per_row(width: usize) -> usize {
        width + 2
    }

    fn build(
        field: &super::conditional_score_covariance::ScoreCovarianceField,
        beta: &[f64],
        rows: usize,
        reservation: gam_runtime::resource::MemoryReservation,
    ) -> Self {
        let k = beta.len();
        let mut sigma_beta = vec![0.0_f64; rows * k];
        let mut u = vec![0.0_f64; rows];
        let mut v = vec![0.0_f64; rows];
        sigma_beta
            .par_chunks_mut(k)
            .zip(u.par_iter_mut().zip(v.par_iter_mut()))
            .enumerate()
            .for_each(|(row, (sigma_beta_row, (u_row, v_row)))| {
                // `γ` is the fit cache's; only the product with `β` is this kernel's.
                let (product, u_i, v_i) = DriveMoments::drive_products(field.at_row(row), beta);
                sigma_beta_row.copy_from_slice(&product);
                *u_row = u_i;
                *v_row = v_i;
            });
        Self {
            width: k,
            sigma_beta,
            u,
            v,
            _reservation: reservation,
        }
    }
}

/// The residual repair kernel over the family's training rows.
pub(super) struct ResidualDriveKernel {
    family: BernoulliMarginalSlopeFamily,
    block_states: Vec<ParameterBlockState>,
    slices: BlockSlices,
    residual: Range<usize>,
    runtime: Arc<ResidualBlockRuntime>,
    /// The drive moments of the pooled law, computed once; `None` when the
    /// joint covariance varies by row.
    pooled: Option<DriveMoments>,
    /// `2Σ_rr` of the pooled law as a dense row-major `K×K` block, computed
    /// once; `None` when the joint covariance varies by row.
    pooled_curvature: Option<Vec<f64>>,
    /// Whether the fit's [`RowCovariance`] (in the runtime) holds every row's
    /// `γ` and `2Σ_rr`; `false` under the pooled law, or when refused, in which
    /// case every call derives them from the row's factor.
    row_covariance: bool,
    /// Each row's `Σ_rrβ`, `u`, `v` for this kernel's `β` under a row-varying
    /// law, when the governor granted their storage; `None` otherwise.
    row_moments: Option<RowMoments>,
}

impl ResidualDriveKernel {
    pub(super) fn new(
        family: BernoulliMarginalSlopeFamily,
        block_states: Vec<ParameterBlockState>,
    ) -> Result<Self, String> {
        Self::with_row_cache_reservation(family, block_states, &|rows, columns| {
            // A refusal is not an error: the kernel keeps the per-call path.
            gam_runtime::resource::MemoryGovernor::global()
                .try_reserve_dense_f64(rows, columns, "residual repair row covariance cache")
                .ok()
        })
    }

    /// `reserve(rows, columns)` grants or refuses each row cache's
    /// `rows × columns` `f64` storage before any of it is allocated.
    fn with_row_cache_reservation(
        family: BernoulliMarginalSlopeFamily,
        block_states: Vec<ParameterBlockState>,
        reserve: &dyn Fn(usize, usize) -> Option<gam_runtime::resource::MemoryReservation>,
    ) -> Result<Self, String> {
        let runtime = family.residual.clone().ok_or_else(|| {
            "residual row kernel constructed on a family without a residual block".to_string()
        })?;
        if family.flex_active() {
            return Err(super::residual_repair::ResidualRepairRefusal::FlexBlocksUnsupported.to_string());
        }
        family.validate_exact_block_state_shapes(&block_states)?;
        let slices = super::hessian_paths::block_slices(&family);
        let residual = slices
            .residual
            .clone()
            .ok_or("residual row kernel: block slices carry no residual range")?;
        if residual.len() != runtime.width() || block_states[2].beta.len() != runtime.width() {
            return Err(format!(
                "residual row kernel width mismatch: range {}, coefficients {}, features {}",
                residual.len(),
                block_states[2].beta.len(),
                runtime.width()
            ));
        }
        let pooled = if runtime.field.is_conditional() || family.y.is_empty() {
            None
        } else {
            let beta = block_states[2]
                .beta
                .as_slice()
                .ok_or("residual beta not contiguous")?;
            Some(DriveMoments::at(runtime.field.at_row(0), beta))
        };
        let pooled_curvature = pooled
            .is_some()
            .then(|| doubled_residual_block(runtime.field.at_row(0)));
        let rows = family.y.len();
        let (row_covariance, row_moments) = if runtime.field.is_conditional() && rows > 0 {
            let beta = block_states[2]
                .beta
                .as_slice()
                .ok_or("residual beta not contiguous")?;
            let row_covariance = runtime
                .row_covariance
                .get_or_build(&runtime.field, beta.len(), rows, reserve)
                .is_some();
            let row_moments = reserve(rows, RowMoments::entries_per_row(beta.len()))
                .map(|reservation| RowMoments::build(&runtime.field, beta, rows, reservation));
            (row_covariance, row_moments)
        } else {
            (false, None)
        };
        Ok(Self {
            family,
            block_states,
            slices,
            residual,
            runtime,
            pooled,
            pooled_curvature,
            row_covariance,
            row_moments,
        })
    }

    /// The fit's row covariance, when it holds every row.
    #[inline]
    fn row_covariance(&self) -> Option<&RowCovariance> {
        if self.row_covariance {
            self.runtime.row_covariance.0.get().and_then(Option::as_ref)
        } else {
            None
        }
    }

    #[inline]
    fn beta(&self) -> &[f64] {
        self.block_states[2]
            .beta
            .as_slice()
            .expect("residual beta is validated contiguous at construction")
    }

    /// Calls `f(γ, Σ_rrβ, u, v)` with row `row`'s drive moments: borrowed from the
    /// pooled law or the row cache, and derived from the row's factor for this
    /// call when neither holds them.
    #[inline]
    fn with_moments<T>(&self, row: usize, f: impl FnOnce(&[f64], &[f64], f64, f64) -> T) -> T {
        if let Some(moments) = &self.pooled {
            return f(&moments.gamma, &moments.sigma_beta, moments.u, moments.v);
        }
        if let (Some(covariance), Some(moments)) = (self.row_covariance(), &self.row_moments) {
            let k = moments.width;
            return f(
                covariance.gamma_row(row),
                &moments.sigma_beta[row * k..(row + 1) * k],
                moments.u[row],
                moments.v[row],
            );
        }
        let moments = DriveMoments::at(self.runtime.field.at_row(row), self.beta());
        f(&moments.gamma, &moments.sigma_beta, moments.u, moments.v)
    }
}

impl RowProgram<5> for ResidualDriveKernel {
    fn n_rows(&self) -> usize {
        self.family.y.len()
    }

    fn primaries(&self, row: usize) -> Result<[f64; 5], String> {
        if row >= self.family.y.len() {
            return Err(format!("ResidualDriveKernel: row {row} out of range"));
        }
        let features = self.runtime.features.row(row);
        let t = features
            .iter()
            .zip(self.beta())
            .map(|(r, b)| r * b)
            .sum::<f64>();
        let (u, v) = self.with_moments(row, |_, _, u, v| (u, v));
        Ok([self.block_states[0].eta[row], self.block_states[1].eta[row], t, u, v])
    }

    fn eval<S: JetScalar<5>>(&self, row: usize, p: &[S; 5]) -> Result<S, String> {
        if row >= self.family.y.len() {
            return Err(format!("ResidualDriveKernel: row {row} out of range"));
        }
        let marginal = self
            .family
            .marginal_link_map(self.block_states[0].eta[row])?;
        let grid = self
            .family
            .latent_measure
            .empirical_grid_for_training_row(row)?;
        let state = ResidualRowState {
            marginal,
            z: self.family.z[row],
            y: self.family.y[row],
            w: self.family.weights[row],
            probit_scale: self.family.probit_frailty_scale(),
            grid: grid.as_deref(),
        };
        let drive = ResidualDrive {
            t: p[2],
            u: p[3],
            v: p[V],
        };
        residual_row_nll(&state, &p[0], &p[1], &drive).map_err(|e| format!("row {row}: {e}"))
    }
}

impl RowKernel<5> for ResidualDriveKernel {
    fn n_coefficients(&self) -> usize {
        self.slices.total
    }

    fn jacobian_action(&self, row: usize, d_beta: &[f64]) -> [f64; 5] {
        let d_view = ndarray::ArrayView1::from(d_beta);
        let rho = self.residual_jacobian_rows(row);
        let [t, u, v] = residual_action(&rho, &d_beta[self.residual.clone()]);
        [
            self.family
                .marginal_design
                .dot_row_view(row, d_view.slice(s![self.slices.marginal.clone()])),
            self.family
                .slope_design
                .dot_row_view(row, d_view.slice(s![self.slices.slope.clone()])),
            t,
            u,
            v,
        ]
    }

    fn jacobian_transpose_action(&self, row: usize, v: &[f64; 5], out: &mut [f64]) {
        {
            let mut m = ndarray::ArrayViewMut1::from(&mut out[self.slices.marginal.clone()]);
            self.family
                .marginal_design
                .axpy_row_into(row, v[0], &mut m)
                .expect("marginal axpy dim mismatch");
        }
        {
            let mut g = ndarray::ArrayViewMut1::from(&mut out[self.slices.slope.clone()]);
            self.family
                .slope_design
                .axpy_row_into(row, v[1], &mut g)
                .expect("slope axpy dim mismatch");
        }
        let rho = self.residual_jacobian_rows(row);
        residual_transpose_action(&rho, &[v[2], v[3], v[4]], &mut out[self.residual.clone()]);
    }

    fn add_pullback_hessian(&self, row: usize, h: &[[f64; 5]; 5], target: &mut Array2<f64>) {
        let marginal = self.slices.marginal.clone();
        let slope = self.slices.slope.clone();
        let residual = self.residual.clone();
        self.family
            .marginal_design
            .syr_row_into_view(row, h[0][0], target.slice_mut(s![marginal.clone(), marginal.clone()]))
            .expect("marginal syr dim mismatch");
        if h[0][1] != 0.0 {
            self.family
                .marginal_design
                .row_outer_into_view(
                    row,
                    &self.family.slope_design,
                    h[0][1],
                    target.slice_mut(s![marginal.clone(), slope.clone()]),
                )
                .expect("marginal-slope outer dim mismatch");
            self.family
                .slope_design
                .row_outer_into_view(
                    row,
                    &self.family.marginal_design,
                    h[0][1],
                    target.slice_mut(s![slope.clone(), marginal.clone()]),
                )
                .expect("slope-marginal outer dim mismatch");
        }
        self.family
            .slope_design
            .syr_row_into_view(row, h[1][1], target.slice_mut(s![slope.clone(), slope.clone()]))
            .expect("slope syr dim mismatch");
        let rho = self.residual_jacobian_rows(row);
        let [c_q, c_g] =
            pullback_residual_block(h, &rho, target.slice_mut(s![residual.clone(), residual.clone()]));
        for (j, (&q, &g)) in c_q.iter().zip(&c_g).enumerate() {
            let col = residual.start + j;
            if q != 0.0 {
                self.family
                    .marginal_design
                    .axpy_row_into(row, q, &mut target.slice_mut(s![marginal.clone(), col]))
                    .expect("marginal-residual axpy dim mismatch");
                self.family
                    .marginal_design
                    .axpy_row_into(row, q, &mut target.slice_mut(s![col, marginal.clone()]))
                    .expect("residual-marginal axpy dim mismatch");
            }
            if g != 0.0 {
                self.family
                    .slope_design
                    .axpy_row_into(row, g, &mut target.slice_mut(s![slope.clone(), col]))
                    .expect("slope-residual axpy dim mismatch");
                self.family
                    .slope_design
                    .axpy_row_into(row, g, &mut target.slice_mut(s![col, slope.clone()]))
                    .expect("residual-slope axpy dim mismatch");
            }
        }
    }

    fn add_diagonal_quadratic(&self, row: usize, h: &[[f64; 5]; 5], diag: &mut [f64]) {
        {
            let mut md = ndarray::ArrayViewMut1::from(&mut diag[self.slices.marginal.clone()]);
            self.family
                .marginal_design
                .squared_axpy_row_into(row, h[0][0], &mut md)
                .expect("marginal squared_axpy dim mismatch");
        }
        {
            let mut gd = ndarray::ArrayViewMut1::from(&mut diag[self.slices.slope.clone()]);
            self.family
                .slope_design
                .squared_axpy_row_into(row, h[1][1], &mut gd)
                .expect("slope squared_axpy dim mismatch");
        }
        let rho = self.residual_jacobian_rows(row);
        for (j, out) in diag[self.residual.clone()].iter_mut().enumerate() {
            for a in 0..3 {
                for b in 0..3 {
                    *out += h[2 + a][2 + b] * rho[a][j] * rho[b][j];
                }
            }
        }
    }
}

impl ResidualDriveRows for ResidualDriveKernel {
    fn residual_range(&self) -> Range<usize> {
        self.residual.clone()
    }

    fn residual_jacobian_rows(&self, row: usize) -> [Vec<f64>; 3] {
        self.with_moments(row, |gamma, sigma_beta, _, _| {
            [
                self.runtime.features.row(row).to_vec(),
                gamma.to_vec(),
                sigma_beta.iter().map(|x| 2.0 * x).collect(),
            ]
        })
    }

    fn curvature_action(&self, row: usize, d_r: &[f64]) -> Vec<f64> {
        if let Some(curvature) = &self.pooled_curvature {
            return invariant_curvature_action(curvature, d_r);
        }
        if let Some(cache) = self.row_covariance() {
            // The per-call products and sum, in the per-call order.
            let packed = cache.curvature_row(row);
            return (0..d_r.len())
                .map(|j| {
                    (0..d_r.len())
                        .map(|l| RowCovariance::packed_entry(packed, j, l) * d_r[l])
                        .sum::<f64>()
                })
                .collect();
        }
        invariant_curvature_action(&doubled_residual_block(self.runtime.field.at_row(row)), d_r)
    }

    fn add_curvature(&self, row: usize, scale: f64, target: &mut Array2<f64>) {
        if scale == 0.0 {
            return;
        }
        if let Some(curvature) = &self.pooled_curvature {
            add_invariant_curvature(target, &self.residual, curvature, scale);
            return;
        }
        if let Some(cache) = self.row_covariance() {
            let packed = cache.curvature_row(row);
            let mut block = target.slice_mut(s![self.residual.clone(), self.residual.clone()]);
            for (j, mut target_row) in block.rows_mut().into_iter().enumerate() {
                for (l, entry) in target_row.iter_mut().enumerate() {
                    *entry += scale * RowCovariance::packed_entry(packed, j, l);
                }
            }
            return;
        }
        add_invariant_curvature(
            target,
            &self.residual,
            &doubled_residual_block(self.runtime.field.at_row(row)),
            scale,
        );
    }

    fn row_invariant_curvature(&self) -> Option<&[f64]> {
        self.pooled_curvature.as_deref()
    }
}

#[cfg(test)]
mod residual_drive_kernel_tests {
    //! The five-primary kernel against the coefficient-primary row program it
    //! replaces: the same row statement lowered with `(η_m, g, β_1, …, β_K)` as
    //! the jet primaries (exact by construction, `O(K⁴)` per row) must agree
    //! with the chain-rule assembly channel for channel; at widths past any
    //! ladder the assembly is gated by finite differences.

    use super::super::residual_repair::declare_unit_score_variance;
    use super::super::tests_residual_repair_laws::{ResidualBlock, covariance, skewed_grid};
    use super::*;

    struct RowFixture {
        y: f64,
        w: f64,
        z: f64,
        s: f64,
        r: Vec<f64>,
        cov: MarginalSlopeCovariance,
        grid: Option<EmpiricalZGrid>,
    }

    impl RowFixture {
        fn new(k: usize, law: Option<EmpiricalZGrid>, seed: u64) -> Self {
            let cov = MarginalSlopeCovariance::full(
                declare_unit_score_variance(&covariance(k, seed).to_dense()).unwrap(),
            )
            .unwrap();
            Self {
                y: 1.0,
                w: 1.2,
                z: 0.7,
                s: 0.9,
                r: (0..k).map(|j| ((j as f64) * 0.37 + 0.2).sin()).collect(),
                cov,
                grid: law,
            }
        }

        fn state(&self, eta_m: f64) -> ResidualRowState<'_> {
            let link = InverseLink::Standard(StandardLink::Probit);
            ResidualRowState {
                marginal: bernoulli_marginal_link_map(&link, eta_m).unwrap(),
                z: self.z,
                y: self.y,
                w: self.w,
                probit_scale: self.s,
                grid: self.grid.as_ref(),
            }
        }
    }

    /// The five-primary kernel on rows that share one covariance and law, with
    /// unit surface designs: `θ = (η_m, g, β)`.
    struct DriveRow<'a> {
        rows: Vec<&'a RowFixture>,
        theta: Vec<f64>,
        /// `2Σ_rr` when the rows report their covariance as row-invariant, so
        /// the assemblies take their pooled-law path.
        invariant: Option<Vec<f64>>,
    }

    impl<'a> DriveRow<'a> {
        fn new(fixture: &'a RowFixture, theta: Vec<f64>, invariant: bool) -> Self {
            Self::over(vec![fixture], theta, invariant)
        }

        /// Rows that must share the first row's covariance.
        fn over(rows: Vec<&'a RowFixture>, theta: Vec<f64>, invariant: bool) -> Self {
            assert!(rows.iter().all(|row| row.cov == rows[0].cov), "rows share one covariance");
            let invariant = invariant.then(|| doubled_residual_block(&rows[0].cov));
            Self { rows, theta, invariant }
        }
    }

    impl DriveRow<'_> {
        fn cov(&self) -> &MarginalSlopeCovariance {
            &self.rows[0].cov
        }

        fn moments(&self) -> DriveMoments {
            DriveMoments::at(self.cov(), &self.theta[2..])
        }
    }

    impl RowProgram<5> for DriveRow<'_> {
        fn n_rows(&self) -> usize {
            self.rows.len()
        }
        fn primaries(&self, row: usize) -> Result<[f64; 5], String> {
            let m = self.moments();
            Ok([
                self.theta[0],
                self.theta[1],
                dot(&self.rows[row].r, &self.theta[2..]),
                m.u,
                m.v,
            ])
        }
        fn eval<S: JetScalar<5>>(&self, row: usize, p: &[S; 5]) -> Result<S, String> {
            let state = self.rows[row].state(self.theta[0]);
            let drive = ResidualDrive {
                t: p[2],
                u: p[3],
                v: p[4],
            };
            residual_row_nll(&state, &p[0], &p[1], &drive)
        }
    }

    impl RowKernel<5> for DriveRow<'_> {
        fn n_coefficients(&self) -> usize {
            self.theta.len()
        }
        fn jacobian_action(&self, row: usize, d: &[f64]) -> [f64; 5] {
            let [t, u, v] = residual_action(&self.residual_jacobian_rows(row), &d[2..]);
            [d[0], d[1], t, u, v]
        }
        fn jacobian_transpose_action(&self, row: usize, v: &[f64; 5], out: &mut [f64]) {
            out[0] += v[0];
            out[1] += v[1];
            residual_transpose_action(&self.residual_jacobian_rows(row), &[v[2], v[3], v[4]], &mut out[2..]);
        }
        fn add_pullback_hessian(&self, row: usize, h: &[[f64; 5]; 5], target: &mut Array2<f64>) {
            for a in 0..2 {
                for b in 0..2 {
                    target[[a, b]] += h[a][b];
                }
            }
            let rho = self.residual_jacobian_rows(row);
            let [c_q, c_g] = pullback_residual_block(h, &rho, target.slice_mut(s![2.., 2..]));
            for j in 0..c_q.len() {
                target[[0, 2 + j]] += c_q[j];
                target[[2 + j, 0]] += c_q[j];
                target[[1, 2 + j]] += c_g[j];
                target[[2 + j, 1]] += c_g[j];
            }
        }
        fn add_diagonal_quadratic(&self, row: usize, h: &[[f64; 5]; 5], diag: &mut [f64]) {
            let mut dense = Array2::<f64>::zeros((self.theta.len(), self.theta.len()));
            self.add_pullback_hessian(row, h, &mut dense);
            for (j, out) in diag.iter_mut().enumerate() {
                *out += dense[[j, j]];
            }
        }
    }

    impl ResidualDriveRows for DriveRow<'_> {
        fn residual_range(&self) -> Range<usize> {
            2..self.theta.len()
        }
        fn residual_jacobian_rows(&self, row: usize) -> [Vec<f64>; 3] {
            let m = self.moments();
            [
                self.rows[row].r.clone(),
                m.gamma,
                m.sigma_beta.iter().map(|x| 2.0 * x).collect(),
            ]
        }
        fn curvature_action(&self, row: usize, d_r: &[f64]) -> Vec<f64> {
            let mut out = vec![0.0; d_r.len()];
            ResidualBlock(&self.rows[row].cov).multiply(d_r, &mut out);
            out.iter().map(|x| 2.0 * x).collect()
        }
        fn add_curvature(&self, row: usize, scale: f64, target: &mut Array2<f64>) {
            let k = self.theta.len() - 2;
            for j in 0..k {
                for l in 0..k {
                    target[[2 + j, 2 + l]] += 2.0 * scale * self.rows[row].cov.coefficient(j + 1, l + 1);
                }
            }
        }
        fn row_invariant_curvature(&self) -> Option<&[f64]> {
            self.invariant.as_deref()
        }
    }

    /// The same row with `(η_m, g, β_1, …, β_K)` as the jet primaries.
    struct CoefficientRow<'a, const P: usize> {
        fixture: &'a RowFixture,
        theta: [f64; P],
    }

    impl<const P: usize> RowProgram<P> for CoefficientRow<'_, P> {
        fn n_rows(&self) -> usize {
            1
        }
        fn primaries(&self, row: usize) -> Result<[f64; P], String> {
            assert_eq!(row, 0, "one-row fixture");
            Ok(self.theta)
        }
        fn eval<S: JetScalar<P>>(&self, row: usize, p: &[S; P]) -> Result<S, String> {
            assert_eq!(row, 0, "one-row fixture");
            let state = self.fixture.state(self.theta[0]);
            let mut gamma = vec![0.0; P - 2];
            for (j, entry) in gamma.iter_mut().enumerate() {
                *entry = self.fixture.cov.coefficient(0, j + 1);
            }
            let beta = &p[2..];
            let drive = ResidualDrive {
                t: S::linear_combination(beta, &self.fixture.r),
                u: S::linear_combination(beta, &gamma),
                v: S::symmetric_quadratic_form(beta, &ResidualBlock(&self.fixture.cov)),
            };
            residual_row_nll(&state, &p[0], &p[1], &drive)
        }
    }

    fn theta(k: usize) -> Vec<f64> {
        let mut theta = vec![-0.45, 0.55];
        theta.extend((0..k).map(|j| 0.3 * ((j as f64) * 0.71 + 0.4).cos() / (k as f64).sqrt()));
        theta
    }

    fn direction(p: usize, phase: f64) -> Vec<f64> {
        (0..p).map(|j| ((j as f64) * 0.53 + phase).sin()).collect()
    }

    fn close(what: &str, exact: f64, reference: f64, tolerance: f64) {
        assert!(
            (exact - reference).abs() <= tolerance * (1.0 + reference.abs()),
            "{what}: assembled {exact} vs reference {reference}"
        );
    }

    fn cache_of(row: &DriveRow<'_>) -> RowKernelCache<5> {
        crate::row_kernel::build_row_kernel_cache(row, &RowSet::All).unwrap()
    }

    /// `invariant` selects the assemblies' pooled-law path, which applies `2Σ_rr`
    /// once from row sums, instead of the per-row path.
    fn equivalence<const P: usize>(law: Option<EmpiricalZGrid>, invariant: bool) {
        let k = P - 2;
        let fixture = RowFixture::new(k, law, 17 + k as u64);
        let base = theta(k);
        let drive = DriveRow::new(&fixture, base.clone(), invariant);
        let reference = CoefficientRow::<P> {
            fixture: &fixture,
            theta: std::array::from_fn(|a| base[a]),
        };
        let law_label = if fixture.grid.is_some() { "declared law" } else { "normal law" };
        let label = format!("{law_label}, {} path", if invariant { "pooled" } else { "per-row" });
        let (nll_ref, grad_ref, hess_ref) = gam_math::jet_tower::program_row_kernel(&reference, 0).unwrap();
        let cache = cache_of(&drive);
        close(&format!("{label} K={k} value"), cache.nll[0], nll_ref, 1e-12);
        let grad = transpose_vector(&drive, 0, &cache.gradients[0]);
        let hess = residual_hessian_dense(&drive, &cache).unwrap();
        let d = direction(P, 0.3);
        let e = direction(P, 1.9);
        let third = residual_hessian_directional_derivative(&drive, &d).unwrap();
        let fourth = residual_hessian_second_directional_derivative(&drive, &d, &e).unwrap();
        let d_arr: [f64; P] = std::array::from_fn(|a| d[a]);
        let e_arr: [f64; P] = std::array::from_fn(|a| e[a]);
        let third_ref = gam_math::jet_tower::program_third_contracted(&reference, 0, &d_arr).unwrap();
        let fourth_ref =
            gam_math::jet_tower::program_fourth_contracted(&reference, 0, &d_arr, &e_arr).unwrap();
        let (_, score, observed) = residual_row_geometry(&drive, 0).unwrap();
        for a in 0..P {
            close(&format!("{label} K={k} grad[{a}]"), grad[a], grad_ref[a], 1e-11);
            close(&format!("{label} K={k} ALO score[{a}]"), score[a], grad_ref[a], 1e-11);
            for b in 0..P {
                close(&format!("{label} K={k} hess[{a}][{b}]"), hess[[a, b]], hess_ref[a][b], 1e-10);
                close(&format!("{label} K={k} ALO hess[{a}][{b}]"), observed[[a, b]], hess_ref[a][b], 1e-10);
                close(&format!("{label} K={k} H'[{a}][{b}]"), third[[a, b]], third_ref[a][b], 1e-9);
                close(&format!("{label} K={k} H''[{a}][{b}]"), fourth[[a, b]], fourth_ref[a][b], 1e-9);
            }
        }
    }

    #[test]
    fn drive_kernel_equals_the_coefficient_primary_program_through_order_four() {
        for law in [None, Some(skewed_grid())] {
            for invariant in [false, true] {
                equivalence::<3>(law.clone(), invariant);
                equivalence::<5>(law.clone(), invariant);
                equivalence::<14>(law.clone(), invariant);
            }
        }
    }

    fn finite_difference_gate(k: usize, law: Option<EmpiricalZGrid>, invariant: bool) {
        let fixture = RowFixture::new(k, law, 101 + k as u64);
        let base = theta(k);
        let p = base.len();
        let d = direction(p, 0.8);
        let e = direction(p, 2.6);
        let at = |theta: &[f64]| DriveRow::new(&fixture, theta.to_vec(), invariant);
        let displaced = |dir: &[f64], scale: f64| -> Vec<f64> {
            base.iter().zip(dir).map(|(x, v)| x + scale * v).collect()
        };
        let gradient = |theta: &[f64]| {
            let row = at(theta);
            let cache = cache_of(&row);
            transpose_vector(&row, 0, &cache.gradients[0])
        };
        let hessian = |theta: &[f64]| {
            let row = at(theta);
            residual_hessian_dense(&row, &cache_of(&row)).unwrap()
        };
        let h = 1.0e-5;
        let law_label = if fixture.grid.is_some() { "declared law" } else { "normal law" };
        let label = format!("{law_label}, {} path", if invariant { "pooled" } else { "per-row" });
        let hess = hessian(&base);
        let third = residual_hessian_directional_derivative(&at(&base), &d).unwrap();
        let fourth = residual_hessian_second_directional_derivative(&at(&base), &d, &e).unwrap();
        let (gp, gm) = (gradient(&displaced(&d, h)), gradient(&displaced(&d, -h)));
        let (hp, hm) = (hessian(&displaced(&d, h)), hessian(&displaced(&d, -h)));
        let tp = residual_hessian_directional_derivative(&at(&displaced(&e, h)), &d).unwrap();
        let tm = residual_hessian_directional_derivative(&at(&displaced(&e, -h)), &d).unwrap();
        for a in 0..p {
            let hd: f64 = (0..p).map(|b| hess[[a, b]] * d[b]).sum();
            close(&format!("{label} K={k} H·d[{a}]"), hd, (gp[a] - gm[a]) / (2.0 * h), 2e-6);
            for b in 0..p {
                close(
                    &format!("{label} K={k} H'[{a}][{b}]"),
                    third[[a, b]],
                    (hp[[a, b]] - hm[[a, b]]) / (2.0 * h),
                    2e-6,
                );
                close(
                    &format!("{label} K={k} H''[{a}][{b}]"),
                    fourth[[a, b]],
                    (tp[[a, b]] - tm[[a, b]]) / (2.0 * h),
                    2e-6,
                );
            }
        }
    }

    #[test]
    fn drive_kernel_is_finite_difference_tight_at_block_widths_26_and_32() {
        for law in [None, Some(skewed_grid())] {
            for invariant in [false, true] {
                finite_difference_gate(26, law.clone(), invariant);
                finite_difference_gate(32, law.clone(), invariant);
            }
        }
    }

    /// `γ_m = m·u/(1 − m·u)` with `u = ε/2`: the relative forward-error bound of
    /// `m` rounded operations (Higham, *Accuracy and Stability of Numerical
    /// Algorithms*, §3.1).
    fn gamma(m: usize) -> f64 {
        let mu = m as f64 * f64::EPSILON / 2.0;
        mu / (1.0 - mu)
    }

    /// `n` rows that share `fixture`'s covariance and law and differ in `y`,
    /// `w`, `z` and `r`.
    fn row_variants(fixture: &RowFixture, n: usize) -> Vec<RowFixture> {
        (0..n)
            .map(|i| {
                let phase = 0.61 * i as f64;
                RowFixture {
                    y: if i % 3 == 0 { 0.0 } else { 1.0 },
                    w: 0.6 + 0.1 * (i % 7) as f64,
                    z: 1.4 * (phase + 0.3).sin(),
                    s: fixture.s,
                    r: (0..fixture.r.len()).map(|j| ((j as f64) * 0.37 + phase).sin()).collect(),
                    cov: fixture.cov.clone(),
                    grid: fixture.grid.clone(),
                }
            })
            .collect()
    }

    /// Entrywise Neumaier-compensated sum.
    fn compensated_sum(parts: &[Array2<f64>]) -> Array2<f64> {
        let mut sum = Array2::<f64>::zeros(parts[0].raw_dim());
        let mut carry = Array2::<f64>::zeros(parts[0].raw_dim());
        for part in parts {
            ndarray::Zip::from(&mut sum).and(&mut carry).and(part).for_each(|s, c, &x| {
                let t = *s + x;
                *c += if s.abs() >= x.abs() { (*s - t) + x } else { (x - t) + *s };
                *s = t;
            });
        }
        sum + carry
    }

    /// Adds `|one row's pullback of h|` to `target`.
    fn pullback_magnitude(target: &mut Array2<f64>, one: &DriveRow<'_>, h: &[[f64; 5]; 5]) {
        let mut piece = Array2::<f64>::zeros(target.raw_dim());
        one.add_pullback_hessian(0, h, &mut piece);
        *target += &piece.mapv(f64::abs);
    }

    /// Adds the magnitudes of the rank-two term `μ ⊗ v + v ⊗ μ`.
    fn rank_two_magnitude(target: &mut Array2<f64>, mu: &[f64], v: &[f64]) {
        for (j, &m) in mu.iter().enumerate() {
            for (c, &g) in v.iter().enumerate() {
                target[[2 + j, c]] += (m * g).abs();
                target[[c, 2 + j]] += (m * g).abs();
            }
        }
    }

    /// Adds `|scale·2Σ_rr|` on the residual block.
    fn curvature_magnitude(target: &mut Array2<f64>, curvature: &[f64], scale: f64) {
        let k = target.nrows() - 2;
        for j in 0..k {
            for l in 0..k {
                target[[2 + j, 2 + l]] += (scale * curvature[j * k + l]).abs();
            }
        }
    }

    /// Adds `|scale·μ_d ⊗ μ_e| + |scale·μ_e ⊗ μ_d|` on the residual block.
    fn outer_magnitude(target: &mut Array2<f64>, mu_d: &[f64], mu_e: &[f64], scale: f64) {
        for (j, (&dj, &ej)) in mu_d.iter().zip(mu_e).enumerate() {
            for (l, (&dl, &el)) in mu_d.iter().zip(mu_e).enumerate() {
                target[[2 + j, 2 + l]] += (scale * dj * el).abs() + (scale * ej * dl).abs();
            }
        }
    }

    /// The pooled path reorders the per-row path's sums: it adds up the row
    /// scalars and coefficient vectors that multiply `2Σ_rr` and each `μ`, and
    /// applies them once. Any evaluation order of a sum of `N` terms, each a
    /// product of at most `c` rounded factors, lies within `γ_{N+c}·Σ|terms|` of
    /// the exact sum (Higham §3.1 and §4.2). So the two paths agree within
    /// `2γ_{N+c}·Σ|terms|` entrywise, with `Σ|terms|` accumulated here from each
    /// row's own terms.
    ///
    /// The accuracy reference is the compensated sum of the rows' single-row
    /// assemblies. Those carry the same within-row rounding in both paths, so
    /// the reference resolves either path's error to `γ_c·Σ|terms| + u·|H|`,
    /// and the pooled path must be no less accurate than the per-row path to
    /// within twice that resolution.
    fn reordering_gate(k: usize, law: Option<EmpiricalZGrid>) {
        let n = 48;
        let fixture = RowFixture::new(k, law, 211 + k as u64);
        let variants = row_variants(&fixture, n);
        let rows: Vec<&RowFixture> = variants.iter().collect();
        let base = theta(k);
        let p = base.len();
        let d = direction(p, 0.8);
        let e = direction(p, 2.6);
        let per_row = DriveRow::over(rows.clone(), base.clone(), false);
        let pooled = DriveRow::over(rows.clone(), base.clone(), true);
        let curvature = pooled.invariant.clone().expect("the pooled rows report their curvature");
        let mu_e = invariant_curvature_action(&curvature, &e[2..]);
        let kappa = dot(&d[2..], &mu_e);
        // Magnitudes of the curvature actions as the per-row path forms them, a
        // K-term product each: |2Σ_rr|·|d_r| and |d_r|ᵀ|2Σ_rr||e_r|.
        let abs_of = |v: &[f64]| -> Vec<f64> { v.iter().map(|x| x.abs()).collect() };
        let abs_curvature = abs_of(&curvature);
        let mu_d_abs = invariant_curvature_action(&abs_curvature, &abs_of(&d[2..]));
        let mu_e_abs = invariant_curvature_action(&abs_curvature, &abs_of(&e[2..]));
        let kappa_abs = dot(&abs_of(&d[2..]), &mu_e_abs);
        let unit_v: [f64; 5] = std::array::from_fn(|a| if a == V { 1.0 } else { 0.0 });
        let mut parts: [Vec<Array2<f64>>; 3] = Default::default();
        let mut magnitude: [Array2<f64>; 3] = std::array::from_fn(|_| Array2::<f64>::zeros((p, p)));
        for &row in &rows {
            let one = DriveRow::new(row, base.clone(), false);
            parts[0].push(residual_hessian_dense(&one, &cache_of(&one)).unwrap());
            parts[1].push(residual_hessian_directional_derivative(&one, &d).unwrap());
            parts[2].push(residual_hessian_second_directional_derivative(&one, &d, &e).unwrap());
            let (_, g, h) = one.row_kernel(0).unwrap();
            let jd = one.jacobian_action(0, &d);
            let je = one.jacobian_action(0, &e);
            let third_d = one.row_third_contracted(0, &jd).unwrap();
            let third_e = one.row_third_contracted(0, &je).unwrap();
            pullback_magnitude(&mut magnitude[0], &one, &h);
            curvature_magnitude(&mut magnitude[0], &curvature, g[V]);
            pullback_magnitude(&mut magnitude[1], &one, &third_d);
            rank_two_magnitude(&mut magnitude[1], &mu_d_abs, &transpose_vector(&one, 0, &h[V]));
            curvature_magnitude(&mut magnitude[1], &curvature, dot(&h[V], &jd));
            pullback_magnitude(&mut magnitude[2], &one, &one.row_fourth_contracted(0, &jd, &je).unwrap());
            if kappa != 0.0 {
                let third_v = one.row_third_contracted(0, &unit_v).unwrap();
                pullback_magnitude(&mut magnitude[2], &one, &scaled(&third_v, kappa_abs));
            }
            rank_two_magnitude(&mut magnitude[2], &mu_d_abs, &transpose_vector(&one, 0, &third_e[V]));
            rank_two_magnitude(&mut magnitude[2], &mu_e_abs, &transpose_vector(&one, 0, &third_d[V]));
            curvature_magnitude(
                &mut magnitude[2],
                &curvature,
                dot(&third_d[V], &je).abs() + kappa_abs * h[V][V].abs(),
            );
            outer_magnitude(&mut magnitude[2], &mu_d_abs, &mu_e_abs, h[V][V]);
        }
        let assembled = [
            (
                residual_hessian_dense(&per_row, &cache_of(&per_row)).unwrap(),
                residual_hessian_dense(&pooled, &cache_of(&pooled)).unwrap(),
            ),
            (
                residual_hessian_directional_derivative(&per_row, &d).unwrap(),
                residual_hessian_directional_derivative(&pooled, &d).unwrap(),
            ),
            (
                residual_hessian_second_directional_derivative(&per_row, &d, &e).unwrap(),
                residual_hessian_second_directional_derivative(&pooled, &d, &e).unwrap(),
            ),
        ];
        // An H″ residual entry takes the most terms per row: two pullbacks, four
        // rank-two halves, the curvature and the outer product. No term takes
        // more than a K-term curvature action and six further rounded operations.
        let (terms_per_row, factors) = (8, k + 6);
        let sweep = gamma(n * terms_per_row + factors);
        let resolution = gamma(terms_per_row + factors);
        let law_label = if fixture.grid.is_some() { "declared law" } else { "normal law" };
        for (c, name) in ["H", "H'", "H''"].iter().enumerate() {
            let reference = compensated_sum(&parts[c]);
            let (per_row_value, pooled_value) = &assembled[c];
            let mut worst = [0.0_f64; 3];
            let mut squared_error = [0.0_f64; 2];
            for ((idx, &a), &b) in per_row_value.indexed_iter().zip(pooled_value.iter()) {
                let m = magnitude[c][idx];
                let exact = reference[idx];
                let (error_a, error_b) = ((a - exact).abs(), (b - exact).abs());
                squared_error[0] += error_a * error_a;
                squared_error[1] += error_b * error_b;
                assert!(
                    (a - b).abs() <= 2.0 * sweep * m,
                    "{law_label} K={k} {name}{idx:?}: per-row {a:e} and pooled {b:e} differ by {:e}, above \
                     the reordering bound 2γ_N·Σ|terms| = {:e}",
                    (a - b).abs(),
                    2.0 * sweep * m
                );
                let slack = 2.0 * (resolution * m + f64::EPSILON / 2.0 * exact.abs());
                assert!(
                    error_b <= error_a + slack,
                    "{law_label} K={k} {name}{idx:?}: pooled error {error_b:e} exceeds per-row error \
                     {error_a:e} by more than the reference resolution {slack:e}"
                );
                if m > 0.0 {
                    worst[0] = worst[0].max((a - b).abs() / m);
                    worst[1] = worst[1].max(error_a / m);
                    worst[2] = worst[2].max(error_b / m);
                }
            }
            eprintln!(
                "[reorder] {law_label} K={k} {name}: max |per-row - pooled|/Σ|terms| = {:.2e} (bound {:.2e}); \
                 max error/Σ|terms| per-row {:.2e}, pooled {:.2e}; Frobenius error per-row {:.3e}, pooled {:.3e}",
                worst[0],
                2.0 * sweep,
                worst[1],
                worst[2],
                squared_error[0].sqrt(),
                squared_error[1].sqrt()
            );
        }
    }

    #[test]
    fn pooled_assembly_agrees_with_the_per_row_assembly_within_its_reordering_bound() {
        for law in [None, Some(skewed_grid())] {
            reordering_gate(26, law.clone());
            reordering_gate(32, law);
        }
    }
}

#[cfg(test)]
mod row_covariance_cache_tests {
    //! The row covariance cache stores the per-call values themselves, so a
    //! kernel that holds it and one the governor refused must agree bit for bit,
    //! and the cached kernel must agree with itself at every pool width.

    use super::super::conditional_score_covariance::{ConditionalScoreCovariance, ScoreCovarianceField};
    use super::super::hessian_paths::{new_cell_moment_cache_stats, new_cell_moment_lru_cache};
    use super::super::residual_repair::ResidualRepairGeometry;
    use super::*;
    use crate::row_kernel::{build_row_kernel_cache, row_kernel_gradient};
    use gam_linalg::matrix::{DenseDesignMatrix, DesignMatrix};
    use gam_problem::{InverseLink, StandardLink};
    use ndarray::{Array1, Array2};
    use std::sync::{Arc, Mutex};

    /// A deterministic draw in `(0, 1)`.
    fn unit(i: usize, salt: u64) -> f64 {
        let mut state = (i as u64).wrapping_mul(0x9E37_79B9_7F4A_7C15) ^ salt;
        let draw = gam_linalg::utils::splitmix64(&mut state);
        ((draw >> 11) as f64 + 0.5) / (1u64 << 53) as f64
    }

    fn gauss(i: usize, salt: u64) -> f64 {
        (-2.0 * unit(i, salt).ln()).sqrt() * (std::f64::consts::TAU * unit(i, salt ^ 0xA5A5)).cos()
    }

    /// `n` rows whose `(z, r1, r2)` covariance moves with a context `a`, so the
    /// gam#2766 conditional law escalates, and a residual family over them.
    fn conditional_fixture(n: usize, k: usize) -> (BernoulliMarginalSlopeFamily, Vec<ParameterBlockState>) {
        let a = Array2::from_shape_fn((n, 1), |(i, _)| 2.0 * unit(i, 1) - 1.0);
        let mut scores = Array2::<f64>::zeros((n, k + 1));
        for i in 0..n {
            let rho = 0.8 * a[[i, 0]];
            let z = gauss(i, 2);
            scores[[i, 0]] = z;
            let mut previous = gauss(i, 3);
            scores[[i, 1]] = rho * z + (1.0 - rho * rho).sqrt() * previous;
            for j in 2..=k {
                let fresh = gauss(i, 3 + j as u64);
                scores[[i, j]] = 0.5 * previous + 0.8 * fresh;
                previous = fresh;
            }
        }
        let weights = Array1::<f64>::ones(n);
        let model = ConditionalScoreCovariance::fit(scores.view(), weights.view(), a.view())
            .expect("conditional covariance fit")
            .expect("a covariance that moves with a escalates");
        let pooled = gradient_paths::marginal_slope_covariance_from_scores(scores.view(), &weights)
            .expect("pooled covariance");
        let pooled_dense: Vec<Vec<f64>> = (0..=k)
            .map(|j| (0..=k).map(|l| pooled.coefficient(j, l)).collect())
            .collect();
        let field = ScoreCovarianceField::conditional(pooled, model.clone(), a.view()).expect("field");
        let features = scores.slice(s![.., 1..]).to_owned();
        let runtime = ResidualBlockRuntime {
            features: features.clone(),
            field,
            geometry: ResidualRepairGeometry {
                columns: (1..=k).map(|j| format!("r{j}")).collect(),
                pooled_covariance: pooled_dense,
                conditional_covariance: Some(model),
                centring_pvalues: vec![1.0; k],
            },
            row_covariance: Default::default(),
        };
        let (pm, pg) = (3usize, 2usize);
        let marginal_x = Array2::from_shape_fn((n, pm), |(i, j)| if j == 0 { 1.0 } else { a[[i, 0]].powi(j as i32) });
        // Outside the marginal span, so a generic identifiability pass keeps both
        // slope columns rather than aliasing them onto the marginal ones.
        let slope_x = Array2::from_shape_fn((n, pg), |(i, j)| {
            if j == 0 { 0.5 + 0.3 * (2.1 * a[[i, 0]]).cos() } else { (3.3 * a[[i, 0]]).sin() }
        });
        let policy = gam_runtime::resource::ResourcePolicy::default_library();
        let family = BernoulliMarginalSlopeFamily {
            jeffreys_armed: false,
            residual: Some(Arc::new(runtime)),
            search: None,
            y: Arc::new(Array1::from_shape_fn(n, |i| if unit(i, 5) < 0.4 { 1.0 } else { 0.0 })),
            weights: Arc::new(Array1::from_shape_fn(n, |i| 0.6 + 0.4 * unit(i, 6))),
            z: Arc::new(scores.column(0).to_owned()),
            latent_measure: LatentMeasureKind::StandardNormal,
            gaussian_frailty_sd: None,
            base_link: InverseLink::Standard(StandardLink::Probit),
            marginal_design: DesignMatrix::Dense(DenseDesignMatrix::from(marginal_x.clone())),
            slope_design: DesignMatrix::Dense(DenseDesignMatrix::from(slope_x.clone())),
            score_warp: None,
            link_dev: None,
            policy: policy.clone(),
            cell_moment_lru: new_cell_moment_lru_cache(&policy),
            cell_moment_cache_stats: new_cell_moment_cache_stats(),
            intercept_warm_starts: None,
            auto_subsample_phase_counter: Arc::new(std::sync::atomic::AtomicUsize::new(0)),
            auto_subsample_last_rho: Arc::new(Mutex::new(None)),
        };
        let marginal_beta = Array1::from_vec(vec![-0.3, 0.2, -0.1]);
        let slope_beta = Array1::from_vec(vec![0.5, -0.2]);
        let residual_beta = Array1::from_shape_fn(k, |j| 0.35 * (0.9 * j as f64 + 0.4).cos() / (k as f64).sqrt());
        let states = vec![
            ParameterBlockState {
                eta: marginal_x.dot(&marginal_beta),
                beta: marginal_beta,
            },
            ParameterBlockState {
                eta: slope_x.dot(&slope_beta),
                beta: slope_beta,
            },
            ParameterBlockState {
                eta: features.dot(&residual_beta),
                beta: residual_beta,
            },
        ];
        (family, states)
    }

    /// Every channel the fit reads, as one bit stream.
    fn channel_bits(kern: &ResidualDriveKernel) -> Vec<u64> {
        let p = kern.n_coefficients();
        let d: Vec<f64> = (0..p).map(|j| ((j as f64) * 0.53 + 0.8).sin()).collect();
        let e: Vec<f64> = (0..p).map(|j| ((j as f64) * 0.29 + 2.6).cos()).collect();
        let cache = build_row_kernel_cache(kern, &RowSet::All).expect("row cache");
        let mut bits: Vec<u64> = cache.nll.iter().map(|v| v.to_bits()).collect();
        bits.extend(row_kernel_gradient(kern, &cache, &RowSet::All).iter().map(|v| v.to_bits()));
        bits.extend(residual_hessian_dense(kern, &cache).expect("H").iter().map(|v| v.to_bits()));
        bits.extend(residual_hessian_directional_derivative(kern, &d).expect("H'").iter().map(|v| v.to_bits()));
        bits.extend(
            residual_hessian_second_directional_derivative(kern, &d, &e)
                .expect("H''")
                .iter()
                .map(|v| v.to_bits()),
        );
        for axis in residual_hessian_directional_derivative_all_axes(kern).expect("H' all axes") {
            bits.extend(axis.iter().map(|v| v.to_bits()));
        }
        for row in [0, 7, 41] {
            let (nll, score, observed) = residual_row_geometry(kern, row).expect("ALO row geometry");
            bits.push(nll.to_bits());
            bits.extend(score.iter().map(|v| v.to_bits()));
            bits.extend(observed.iter().map(|v| v.to_bits()));
        }
        bits
    }

    #[test]
    fn row_covariance_cache_is_bitwise_the_per_call_kernel_at_every_pool_width() {
        // Two fixtures, so the refused kernel's runtime never holds a cache the
        // cached kernel's built: the fit-level cache lives in the runtime.
        let (refused_family, refused_states) = conditional_fixture(600, 2);
        let refused =
            ResidualDriveKernel::with_row_cache_reservation(refused_family, refused_states, &|_, _| None)
                .expect("refused kernel");
        assert!(
            !refused.row_covariance && refused.row_moments.is_none(),
            "a refused reservation keeps the per-call path"
        );
        let (family, states) = conditional_fixture(600, 2);
        let cached = ResidualDriveKernel::new(family.clone(), states.clone()).expect("cached kernel");
        assert!(
            cached.row_covariance && cached.row_moments.is_some(),
            "the governor grants a 600-row cache"
        );
        let second = ResidualDriveKernel::new(family, states).expect("second kernel over the fit");
        assert!(second.row_covariance, "a later kernel shares the fit's row covariance");
        let width = cached.residual.len();
        eprintln!(
            "[2924 row cache] {} bytes per row per fit and {} per kernel at K = {width}",
            8 * RowCovariance::entries_per_row(width),
            8 * RowMoments::entries_per_row(width)
        );
        let pool = |workers: usize| {
            rayon::ThreadPoolBuilder::new()
                .num_threads(workers)
                .build()
                .expect("test worker pool")
        };
        let reference = pool(1).install(|| channel_bits(&refused));
        for workers in [1usize, 4, 12] {
            let observed = pool(workers).install(|| channel_bits(&cached));
            assert_eq!(observed.len(), reference.len(), "channel count at {workers} workers");
            let first_difference = observed.iter().zip(&reference).position(|(a, b)| a != b);
            assert_eq!(
                first_difference, None,
                "the cached kernel at {workers} workers differs from the per-call kernel"
            );
        }
    }

    /// The fit-level cache is built from `Σ(a)` alone, which no stage of a fit
    /// re-estimates. Run an escalated K = 26 fit to convergence through the
    /// custom-family engine, which constructs a kernel for every assembly, and
    /// at the end compare every row's cached `γ` and `2Σ_rr` with a fresh
    /// derivation from the row's factor, bit for bit.
    #[test]
    fn row_covariance_cache_is_the_fresh_recompute_after_an_escalated_fit() {
        let k = 26;
        let (family, states) = conditional_fixture(1_500, k);
        let runtime = family.residual.clone().expect("the fixture carries a residual block");
        assert!(runtime.field.is_conditional(), "the K = {k} fixture escalates to Σ(a)");
        let dense_block = |name: &str, design: &DesignMatrix, beta: &Array1<f64>, priority: u8| ParameterBlockSpec {
            name: name.to_string(),
            design: design.clone(),
            offset: Array1::zeros(design.nrows()),
            penalties: vec![PenaltyMatrix::Diagonal(Array1::from_elem(beta.len(), 1.0))],
            nullspace_dims: vec![0],
            initial_log_lambdas: Array1::zeros(1),
            initial_beta: Some(beta.clone()),
            gauge_priority: priority,
            jacobian_callback: None,
            stacked_design: None,
            stacked_offset: None,
        };
        let specs = vec![
            dense_block("marginal_surface", &family.marginal_design, &states[0].beta, 150),
            dense_block("slope_surface", &family.slope_design, &states[1].beta, 120),
            runtime
                .block_spec(Array1::zeros(1), Some(states[2].beta.clone()))
                .expect("residual block spec"),
        ];
        let fit = crate::custom_family::fit_custom_family(
            &family,
            &specs,
            &crate::custom_family::BlockwiseFitOptions::default(),
        )
        .expect("the escalated K = 26 fit converges");
        let cache = runtime
            .row_covariance
            .0
            .get()
            .and_then(Option::as_ref)
            .expect("the fit's kernels built the row covariance");
        let packed = k * (k + 1) / 2;
        for row in 0..family.y.len() {
            let covariance = runtime.field.at_row(row);
            let gamma: Vec<u64> = (0..k).map(|j| covariance.coefficient(0, j + 1).to_bits()).collect();
            let cached_gamma: Vec<u64> = cache.gamma_row(row).iter().map(|v| v.to_bits()).collect();
            assert_eq!(cached_gamma, gamma, "row {row}: cached γ differs from the recompute");
            let block = doubled_residual_block(covariance);
            let cached = cache.curvature_row(row);
            assert_eq!(cached.len(), packed);
            for j in 0..k {
                for l in 0..k {
                    assert_eq!(
                        RowCovariance::packed_entry(cached, j, l).to_bits(),
                        block[j * k + l].to_bits(),
                        "row {row}: cached 2Σ_rr[{j}][{l}] differs from the recompute"
                    );
                }
            }
        }
        eprintln!(
            "[2924 row cache] escalated K = {k} fit: {} outer iterations, residual EDF {:.3}; the \
             cache matches a fresh recompute on all {} rows",
            fit.outer_iterations,
            fit.blocks[2].edf,
            family.y.len()
        );
    }
}
