//! Variance-component score test for one smooth term.
//!
//! A penalized smooth `f_j(x) = X_j·β_j` is the random effect
//! `β_j ~ N(0, τ·S_j⁻)`, and "no effect" is the boundary null `τ = 0`. The
//! score for `τ` at `τ = 0` is a quadratic form in the term's score vector
//! (Lin 1997; Zhang & Lin 2003), with `K` the fixed covariance direction below,
//!
//! ```text
//! s = X_jᵀ W (z − X_o β̃_o),        Q = sᵀ K s / φ̂,
//! ```
//!
//! where `o` is every other coefficient, `z` the working response and `β̃_o`
//! the fit of `z` on the other columns at their own penalties. `s` is linear in
//! `z`, so under the null it is exactly Gaussian for a Gaussian response, with
//! covariance `φ·C`, and `Q` is exactly `Σ w_i χ²₁` with `w = eig(K^½ C K^½)`.
//! When `φ` is estimated, `Q ≥ q` is `Σ w_i χ²₁ ≥ q·χ²_ρ/ρ` and the tail is the
//! signed weighted chi-square `P(Σ w_i χ²₁ − (q/ρ)·χ²_ρ > 0)`.
//!
//! Everything is read off the one full fit. `b = Hβ̂` equals `XᵀWz` at
//! convergence (the penalized score vanishes, so `XᵀW(z − Xβ̂) = S(λ)β̂`), so
//! `s = b_j − G_jo·H_oo⁻¹·b_o` with `G = XᵀWX` and `H = G + S(λ)`, and
//!
//! ```text
//! C = G_jj − G_jo A − Aᵀ G_oj + Aᵀ G_oo A,        A = H_oo⁻¹ G_oj,
//! ```
//!
//! Why a score test rather than a Wald test on `β̂_j`: the Wald statistic is a
//! function of the term's OWN fitted smoothing parameter. Under the null that
//! parameter goes to its boundary in a large share of replicates, which puts an
//! atom of `β̂_j ≈ 0` (and `p ≈ 1`) in the statistic's null law, and in the rest
//! the data-driven `λ_j` selects the directions the statistic is spread over.
//! No reference law of fixed shape is right under both. The score test never
//! looks at `λ_j`: `s` fits the other terms only, and `K` is the term's
//! structural penalty, fixed by the basis. Its reference law is therefore exact
//! at the fitted `λ_o`, across the whole unit interval.
//!
//! `K` is the covariance direction of the linear variance-component model
//! `β_j = Σ_l β_l`, one independent component `β_l ~ N(0, φ·τ_l·K_l)` per
//! structural penalty `S_l` of the term. The score for component `l` is
//! `U_l = sᵀ K_l s / φ`, with null mean `t_l = tr(K_l C)`, and the test
//! combines the components with equal weight once each is put on its own null
//! scale:
//!
//! ```text
//! K = Σ_l K_l / t_l.
//! ```
//!
//! `K_l` is read off the penalties themselves. Let `N_l` be an orthonormal
//! basis of `null(Σ_{k≠l} S_k)`, the directions that `S_l` alone charges. When
//! these subspaces together form a basis of the block, every `S_k` vanishes on
//! `N_l` for `k ≠ l`, so the prior `βᵀ(Σ λ_l S_l)β` splits into one quadratic
//! form per component at every `λ`, and the component covariances are exact:
//!
//! ```text
//! K_l = N_l (N_lᵀ S_l N_l)⁻¹ N_lᵀ.
//! ```
//!
//! A smooth with its null-space penalty is this case whatever functional that
//! penalty charges: the null-space penalty alone charges the wiggliness null
//! space, and the wiggliness alone charges what the null-space penalty leaves
//! free. `K_l` is then the covariance of the null functions themselves (a
//! level-specific or varying-coefficient slope), and it transforms with any
//! change of coefficient chart `β = T·γ`, so the test does not depend on the
//! chart. The Euclidean pseudo-inverse `S_l⁺` is the same thing only when the
//! ranges are orthogonal; for a ridge that charges a functional of the curve
//! (the B-spline ridge charges the mean slope through the end values) it points
//! along the functional's row, a curve concentrated at the ends, and leaves a
//! linear effect nearly invisible. When the subspaces do not form a basis (a
//! tensor product whose marginal penalties overlap) no exact split exists, and
//! `K_l = S_l⁺` is the additive working model.
//!
//! That choice does not depend on the scale any penalty was built at
//! (rescaling `S_l` by `c` divides both `K_l` and `t_l` by `c`), and it gives
//! the null space the same standing as the wiggly part: an alternative that is
//! linear in `x` enters through the null-space component with the same weight
//! as a curve enters through the wiggly one. Summing the penalties first and
//! inverting the sum does not: it weights directions by the reciprocal of the
//! summed curvature, so the few smoothest wiggly directions dominate `K` and a
//! null-space effect is nearly invisible to the test. Any fixed `K` gives an
//! exact null law, so the choice decides power, never size.
//!
//! The test needs the penalties' ranges to cover the block: a direction no
//! penalty touches is a fixed effect, not part of the variance component, and
//! `τ = 0` does not set it to zero. A component whose null mean `t_l` is lost
//! in rounding against `tr(K_l G_jj)` lies inside the span of the other terms;
//! it carries no score and is left out of `K`.

use gam_linalg::faer_ndarray::{FaerCholesky, FaerEigh};
use gam_math::probability::{WeightedChiSquareTerm, signed_weighted_chi_square_sf};
use ndarray::{Array1, Array2, ArrayView1};
use std::ops::Range;

use crate::inference::smooth_test::{SmoothTestResult, SmoothTestScale};

/// Inputs to [`smooth_score_test`]. `beta`, `penalized_hessian` (`H = G + S(λ)`)
/// and `weighted_gram` (`G = XᵀWX`) are the full fit in one coefficient layout;
/// the term is `coeff_range`. `structural_penalties` are the term-local
/// structural penalties `S_l`, each `m × m`, at any scale. `covariance_scale` is the `φ̂` that turns `H⁻¹` into the
/// coefficient covariance (the profiled `σ̂²` for a Gaussian, `1` for a family
/// whose working weight carries the dispersion). `residual_df` is `ρ` for the
/// estimated-scale reference.
#[derive(Debug, Clone)]
pub struct SmoothScoreTestInput<'a> {
    pub beta: ArrayView1<'a, f64>,
    pub penalized_hessian: &'a Array2<f64>,
    pub weighted_gram: &'a Array2<f64>,
    pub coeff_range: Range<usize>,
    pub structural_penalties: &'a [Array2<f64>],
    pub covariance_scale: f64,
    pub residual_df: Option<f64>,
    pub scale: SmoothTestScale,
}

/// Why no score test exists for a term.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SmoothScoreTestRefusal {
    /// The inputs are not one finite fit in one layout: a dimension disagrees,
    /// an entry is not finite, or the covariance scale is not positive.
    InconsistentFit,
    /// The penalties' ranges do not cover the block (or there is no penalty),
    /// so the term has directions that no penalty shrinks and that `τ = 0` does
    /// not remove.
    UnpenalizedDirection,
    /// The other coefficients' penalized Hessian is not positive definite, or
    /// the term's score has no variance left once they are fitted: the term is
    /// not identified apart from the rest of the model.
    NotIdentified,
    /// The likelihood curvature `G` leaves the term's score an indefinite
    /// covariance, so the score has no variance law to refer it to.
    IndefiniteCurvature,
    /// The scale is estimated but the fit has no residual degrees of freedom.
    ResidualDfUnavailable,
}

/// The variance-component score test of one smooth term; see the module docs.
///
/// The returned `statistic` is `Q` rescaled so that its null mean equals the
/// reported `ref_df = (Σw)²/Σw²`, the two-moment effective degrees of freedom
/// of the null law. The rescaling is a unit choice for display and does not
/// change the p-value, which is the exact tail of `Σ w_i χ²₁` (over `χ²_ρ/ρ`
/// when the scale is estimated).
pub fn smooth_score_test(
    input: SmoothScoreTestInput<'_>,
) -> Result<SmoothTestResult, SmoothScoreTestRefusal> {
    if input.penalized_hessian.dim() != (input.beta.len(), input.beta.len()) {
        return Err(SmoothScoreTestRefusal::InconsistentFit);
    }
    let working_score = input.penalized_hessian.dot(&input.beta);
    smooth_score_test_at_working_score(input, working_score.view())
}

/// [`smooth_score_test`] with the working score `b = XᵀWz` supplied rather
/// than read off the fit as `H·β̂`.
///
/// `H·β̂ = XᵀWz` is the penalized stationarity condition, so the two agree at
/// an exact mode of `ℓ − ½βᵀS(λ)β`. A fit whose published mode maximizes a
/// different objective (a Jeffreys/Firth prior added to the penalized
/// likelihood) is not stationary for that one, and there the identity is off
/// by the prior's gradient. A caller that holds the likelihood gradient `∇ℓ(β̂)`
/// forms `b = Gβ̂ + ∇ℓ(β̂)` exactly, whatever objective produced `β̂`; that is
/// the linearization the score's reference law is derived at.
pub fn smooth_score_test_at_working_score(
    input: SmoothScoreTestInput<'_>,
    working_score: ArrayView1<'_, f64>,
) -> Result<SmoothTestResult, SmoothScoreTestRefusal> {
    let p = input.beta.len();
    if working_score.len() != p || working_score.iter().any(|v| !v.is_finite()) {
        return Err(SmoothScoreTestRefusal::InconsistentFit);
    }
    let range = input.coeff_range.clone();
    let m = range.len();
    let h = input.penalized_hessian;
    let g = input.weighted_gram;
    let penalties = input.structural_penalties;
    if m == 0
        || range.end > p
        || h.dim() != (p, p)
        || g.dim() != (p, p)
        || penalties.iter().any(|penalty| penalty.dim() != (m, m))
        || !(input.covariance_scale.is_finite() && input.covariance_scale > 0.0)
        || input
            .beta
            .iter()
            .chain(h.iter())
            .chain(g.iter())
            .chain(penalties.iter().flat_map(|penalty| penalty.iter()))
            .any(|v| !v.is_finite())
    {
        return Err(SmoothScoreTestRefusal::InconsistentFit);
    }

    let other: Vec<usize> = (0..p).filter(|i| !range.contains(i)).collect();
    let term: Vec<usize> = range.collect();
    let b = working_score.to_owned();
    let b_j = select_vector(&b, &term);
    let g_jj = select(g, &term, &term);
    let (score, score_cov) = if other.is_empty() {
        (b_j, g_jj.clone())
    } else {
        let h_oo = select(h, &other, &other);
        let g_oo = select(g, &other, &other);
        let g_oj = select(g, &other, &term);
        let factor = h_oo
            .cholesky(faer::Side::Lower)
            .map_err(|_| SmoothScoreTestRefusal::NotIdentified)?;
        // A = H_oo⁻¹ G_oj; s = b_j − A'b_o; C = G_jj − G_jo A − A'G_oj + A'G_oo A.
        let a = factor.solve_mat(&g_oj);
        let b_o = select_vector(&b, &other);
        let score = &b_j - &a.t().dot(&b_o);
        let cross = g_oj.t().dot(&a);
        let cov = &g_jj - &cross - &cross.t() + &a.t().dot(&g_oo.dot(&a));
        (score, symmetrized(&cov))
    };
    // `C = MᵀGM` is a covariance whenever `G` is; a likelihood curvature that is
    // not (an observed information away from a likelihood maximum) gives the
    // score no variance law, and testing on the positive part of `C` would
    // test a different statistic.
    let (score_cov_evals, _) = score_cov
        .eigh(faer::Side::Lower)
        .map_err(|_| SmoothScoreTestRefusal::InconsistentFit)?;
    let score_cov_tolerance = crate::basis::spectral_noise_tolerance(&score_cov_evals);
    if score_cov_evals.iter().any(|&e| e < -score_cov_tolerance) {
        return Err(SmoothScoreTestRefusal::IndefiniteCurvature);
    }

    // K = Σ_l K_l / tr(K_l C) over the components the other terms leave
    // identified; the ranges of all S_l must cover the block.
    let mut ranges = Vec::with_capacity(penalties.len());
    let mut coverage = Array2::<f64>::zeros((m, m));
    for penalty in penalties {
        let range = PenaltyRange::of(penalty)?;
        if range.values.is_empty() {
            continue;
        }
        coverage += &range.vectors.dot(&range.vectors.t());
        ranges.push(range);
    }
    let (covered, _) = coverage
        .eigh(faer::Side::Lower)
        .map_err(|_| SmoothScoreTestRefusal::InconsistentFit)?;
    let coverage_tolerance = crate::basis::spectral_tolerance(&covered);
    if ranges.is_empty() || covered.iter().any(|&e| !(e > coverage_tolerance)) {
        return Err(SmoothScoreTestRefusal::UnpenalizedDirection);
    }
    let mut kernel = Array2::<f64>::zeros((m, m));
    let mut identified = false;
    for component in component_covariances(&ranges, m)? {
        let null_mean = (&component * &score_cov).sum();
        let scale = (&component * &g_jj).sum();
        if null_mean > crate::basis::spectral_tolerance_for_dim(m, &Array1::from_elem(1, scale)) {
            kernel.scaled_add(1.0 / null_mean, &component);
            identified = true;
        }
    }
    if !identified {
        return Err(SmoothScoreTestRefusal::NotIdentified);
    }

    // K^½: L = V·diag(k^½) on the range of K, so K = L·Lᵀ.
    let (kernel_evals, kernel_evecs) = symmetrized(&kernel)
        .eigh(faer::Side::Lower)
        .map_err(|_| SmoothScoreTestRefusal::InconsistentFit)?;
    let kernel_tolerance = crate::basis::spectral_tolerance(&kernel_evals);
    let kept: Vec<usize> = (0..m).filter(|&i| kernel_evals[i] > kernel_tolerance).collect();
    let mut root = kernel_evecs.select(ndarray::Axis(1), &kept);
    for (mut column, &i) in root.columns_mut().into_iter().zip(kept.iter()) {
        column.mapv_inplace(|v| v * kernel_evals[i].sqrt());
    }

    let whitened_score = root.t().dot(&score);
    let q = whitened_score.dot(&whitened_score) / input.covariance_scale;
    let whitened_cov = symmetrized(&root.t().dot(&score_cov.dot(&root)));
    let (weights, _) = whitened_cov
        .eigh(faer::Side::Lower)
        .map_err(|_| SmoothScoreTestRefusal::InconsistentFit)?;
    let weight_tolerance = crate::basis::spectral_tolerance(&weights);
    let weights: Vec<f64> = weights.iter().copied().filter(|&w| w > weight_tolerance).collect();
    if weights.is_empty() {
        return Err(SmoothScoreTestRefusal::NotIdentified);
    }
    let total: f64 = weights.iter().sum();
    let total_sq: f64 = weights.iter().map(|w| w * w).sum();
    let ref_df = total * total / total_sq;
    let unit = ref_df / total;
    let statistic = q * unit;

    let mut terms: Vec<WeightedChiSquareTerm> = weights
        .iter()
        .map(|&w| WeightedChiSquareTerm { weight: w * unit, degrees_of_freedom: 1.0 })
        .collect();
    let tail = match input.scale {
        SmoothTestScale::Known => signed_weighted_chi_square_sf(&terms, statistic),
        SmoothTestScale::Estimated => {
            let rho = input
                .residual_df
                .filter(|rho| rho.is_finite() && *rho > 0.0)
                .ok_or(SmoothScoreTestRefusal::ResidualDfUnavailable)?;
            terms.push(WeightedChiSquareTerm { weight: -statistic / rho, degrees_of_freedom: rho });
            signed_weighted_chi_square_sf(&terms, 0.0)
        }
    };
    if !tail.probability.is_finite() {
        return Err(SmoothScoreTestRefusal::InconsistentFit);
    }
    Ok(SmoothTestResult { statistic, ref_df, p_value: tail.probability.clamp(0.0, 1.0) })
}

/// One structural penalty `S = V·diag(d)·Vᵀ` with `V` the orthonormal
/// eigenvectors whose eigenvalues `d` clear the spectral tolerance.
struct PenaltyRange {
    matrix: Array2<f64>,
    vectors: Array2<f64>,
    values: Vec<f64>,
}

impl PenaltyRange {
    fn of(penalty: &Array2<f64>) -> Result<Self, SmoothScoreTestRefusal> {
        let matrix = symmetrized(penalty);
        let (evals, evecs) = matrix
            .eigh(faer::Side::Lower)
            .map_err(|_| SmoothScoreTestRefusal::InconsistentFit)?;
        let tolerance = crate::basis::spectral_tolerance(&evals);
        let kept: Vec<usize> = (0..evals.len()).filter(|&i| evals[i] > tolerance).collect();
        Ok(Self {
            vectors: evecs.select(ndarray::Axis(1), &kept),
            values: kept.iter().map(|&i| evals[i]).collect(),
            matrix,
        })
    }

    /// `S / d_max`: every penalty at unit top curvature, so the rank decision
    /// on a sum of penalties does not depend on the scale each was built at.
    fn normalized(&self) -> Array2<f64> {
        let top = self.values.iter().copied().fold(0.0_f64, f64::max);
        &self.matrix / top
    }

    fn pseudo_inverse(&self) -> Array2<f64> {
        let mut scaled = self.vectors.clone();
        for (mut column, &value) in scaled.columns_mut().into_iter().zip(self.values.iter()) {
            column.mapv_inplace(|v| v / value);
        }
        scaled.dot(&self.vectors.t())
    }
}

/// The covariance direction `K_l` of each variance component; see the module
/// docs. When the directions each penalty alone charges make up the block,
/// the penalty separates on them and `K_l` is `S_l` inverted on its own
/// subspace, which is exact and does not depend on the coefficient chart.
/// Otherwise the penalties overlap and `K_l = S_l⁺` is the additive working
/// model; the two agree whenever the ranges are orthogonal.
fn component_covariances(
    ranges: &[PenaltyRange],
    m: usize,
) -> Result<Vec<Array2<f64>>, SmoothScoreTestRefusal> {
    match exclusive_subspaces(ranges, m)? {
        Some(subspaces) => subspaces
            .iter()
            .zip(ranges)
            .map(|(basis, range)| inverse_on_subspace(&range.matrix, basis))
            .collect(),
        None => Ok(ranges.iter().map(PenaltyRange::pseudo_inverse).collect()),
    }
}

/// For each penalty, an orthonormal basis `N_l` of the directions no other
/// penalty charges, `null(Σ_{k≠l} S_k)`, when these subspaces together are a
/// basis of the block; `None` when they are not.
///
/// On such a basis every `S_k` vanishes on `N_l` for `k ≠ l`, so
/// `βᵀ(Σ λ_k S_k)β = Σ λ_l a_lᵀ(N_lᵀ S_l N_l)a_l` for `β = Σ N_l a_l` at every
/// `λ`: the components are independent under the prior and their covariances
/// add. A double-penalty smooth is this case whatever functional its ridge
/// charges: the ridge alone charges the wiggliness null space, and the
/// wiggliness alone charges the ridge's null space.
fn exclusive_subspaces(
    ranges: &[PenaltyRange],
    m: usize,
) -> Result<Option<Vec<Array2<f64>>>, SmoothScoreTestRefusal> {
    let mut subspaces = Vec::with_capacity(ranges.len());
    for l in 0..ranges.len() {
        let mut others = Array2::<f64>::zeros((m, m));
        for (k, range) in ranges.iter().enumerate() {
            if k != l {
                others += &range.normalized();
            }
        }
        let (evals, evecs) = others
            .eigh(faer::Side::Lower)
            .map_err(|_| SmoothScoreTestRefusal::InconsistentFit)?;
        let tolerance = crate::basis::spectral_tolerance(&evals);
        let null: Vec<usize> = (0..m).filter(|&i| evals[i] <= tolerance).collect();
        subspaces.push(evecs.select(ndarray::Axis(1), &null));
    }
    if subspaces.iter().map(|basis| basis.ncols()).sum::<usize>() != m {
        return Ok(None);
    }
    let views: Vec<_> = subspaces.iter().map(|basis| basis.view()).collect();
    let stacked = ndarray::concatenate(ndarray::Axis(1), &views)
        .map_err(|_| SmoothScoreTestRefusal::InconsistentFit)?;
    let (singular_sq, _) = stacked
        .t()
        .dot(&stacked)
        .eigh(faer::Side::Lower)
        .map_err(|_| SmoothScoreTestRefusal::InconsistentFit)?;
    let tolerance = crate::basis::spectral_tolerance(&singular_sq);
    Ok(singular_sq.iter().all(|&e| e > tolerance).then_some(subspaces))
}

/// `N (Nᵀ S N)⁻¹ Nᵀ`, the prior covariance direction of the component that
/// lives on `span N` when `S` is its only penalty there.
fn inverse_on_subspace(
    penalty: &Array2<f64>,
    basis: &Array2<f64>,
) -> Result<Array2<f64>, SmoothScoreTestRefusal> {
    let restricted = PenaltyRange::of(&basis.t().dot(&penalty.dot(basis)))?;
    if restricted.values.len() != basis.ncols() {
        return Err(SmoothScoreTestRefusal::UnpenalizedDirection);
    }
    Ok(basis.dot(&restricted.pseudo_inverse().dot(&basis.t())))
}

fn symmetrized(matrix: &Array2<f64>) -> Array2<f64> {
    (matrix + &matrix.t()) * 0.5
}

fn select(matrix: &Array2<f64>, rows: &[usize], cols: &[usize]) -> Array2<f64> {
    Array2::from_shape_fn((rows.len(), cols.len()), |(i, j)| matrix[[rows[i], cols[j]]])
}

fn select_vector(vector: &Array1<f64>, rows: &[usize]) -> Array1<f64> {
    Array1::from_shape_fn(rows.len(), |i| vector[rows[i]])
}

#[cfg(test)]
mod tests {
    use super::*;
    use rand::rngs::StdRng;
    use rand::{RngExt, SeedableRng};

    fn standard_normal(rng: &mut StdRng) -> f64 {
        let u1 = 1.0 - rng.random::<f64>();
        let u2 = rng.random::<f64>();
        (-2.0 * u1.ln()).sqrt() * (2.0 * std::f64::consts::PI * u2).cos()
    }

    /// A penalized regression with an intercept, an unpenalized linear column
    /// and a penalized smooth of `x1` (the other terms), and the tested smooth of
    /// `x2` in columns `4..14`. The tested term carries two penalties, as a
    /// double-penalty smooth does: a second-difference penalty on its
    /// coefficients and the projector onto that penalty's null space
    /// `span{1, k}`, so together they shrink every direction.
    struct Design {
        x: Array2<f64>,
        weights: Array1<f64>,
        penalty: Array2<f64>,
        wiggle: Array2<f64>,
        null_space: Array2<f64>,
        term: Range<usize>,
    }

    fn design(n: usize) -> Design {
        let x1 = Array1::from_shape_fn(n, |i| i as f64 / (n - 1) as f64);
        // A full-period permutation of the grid, so x2 is not a function of x1.
        let x2 = Array1::from_shape_fn(n, |i| ((i * 37) % n) as f64 / (n - 1) as f64);
        let term = 4..14;
        let m = term.len();
        let mut x = Array2::<f64>::zeros((n, 4 + m));
        for i in 0..n {
            x[[i, 0]] = 1.0;
            x[[i, 1]] = x1[i];
            x[[i, 2]] = (std::f64::consts::PI * x1[i]).sin();
            x[[i, 3]] = (2.0 * std::f64::consts::PI * x1[i]).cos();
            for k in 0..m {
                let t = x2[i];
                x[[i, 4 + k]] = (std::f64::consts::PI * (k + 1) as f64 * t).cos() + 0.3 * t.powi(k as i32 + 1);
            }
        }
        // Center the tested columns so the term is identified apart from the
        // intercept.
        for k in term.clone() {
            let mean = x.column(k).mean().unwrap();
            x.column_mut(k).mapv_inplace(|v| v - mean);
        }
        let mut wiggle = Array2::<f64>::zeros((m, m));
        for r in 0..m - 2 {
            let d = [1.0, -2.0, 1.0];
            for a in 0..3 {
                for b in 0..3 {
                    wiggle[[r + a, r + b]] += d[a] * d[b];
                }
            }
        }
        let null_space = {
            let (u, v) = null_space_basis(m);
            Array2::from_shape_fn((m, m), |(a, b)| u[a] * u[b] + v[a] * v[b])
        };
        let mut penalty = Array2::<f64>::zeros((4 + m, 4 + m));
        penalty[[2, 2]] = 3.0;
        penalty[[3, 3]] = 3.0;
        for a in 0..m {
            for b in 0..m {
                penalty[[4 + a, 4 + b]] = 20.0 * wiggle[[a, b]] + null_space[[a, b]];
            }
        }
        let weights = Array1::from_shape_fn(n, |i| 1.0 + (i % 3) as f64);
        Design { x, weights, penalty, wiggle, null_space, term }
    }

    /// An orthonormal basis `(constant, centered index)` of the
    /// second-difference null space in `m` coefficients.
    fn null_space_basis(m: usize) -> (Array1<f64>, Array1<f64>) {
        let constant = Array1::from_elem(m, 1.0 / (m as f64).sqrt());
        let mut slope = Array1::from_shape_fn(m, |k| k as f64 - (m - 1) as f64 / 2.0);
        let norm = slope.dot(&slope).sqrt();
        slope.mapv_inplace(|v| v / norm);
        (constant, slope)
    }

    struct Fit {
        beta: Array1<f64>,
        hessian: Array2<f64>,
        gram: Array2<f64>,
        sigma2_hat: f64,
        residual_df: f64,
    }

    /// The weighted penalized least-squares fit at the design's fixed λ.
    fn fit(design: &Design, y: &Array1<f64>) -> Fit {
        let wx = &design.x * &design.weights.view().insert_axis(ndarray::Axis(1));
        let gram = design.x.t().dot(&wx);
        let hessian = &gram + &design.penalty;
        let factor = hessian.cholesky(faer::Side::Lower).expect("H is positive definite");
        let beta = factor.solvevec(&wx.t().dot(y));
        let influence = factor.solve_mat(&gram);
        let edf = influence.diag().sum();
        let residual = y - &design.x.dot(&beta);
        let rss = residual.iter().zip(design.weights.iter()).map(|(r, w)| w * r * r).sum::<f64>();
        let residual_df = y.len() as f64 - edf;
        Fit { beta, hessian, gram, sigma2_hat: rss / residual_df, residual_df }
    }

    fn test(design: &Design, fit: &Fit, scale: SmoothTestScale) -> SmoothTestResult {
        test_with(design, fit, scale, &[design.wiggle.clone(), design.null_space.clone()])
            .expect("the score test is defined")
    }

    fn test_with(
        design: &Design,
        fit: &Fit,
        scale: SmoothTestScale,
        structural_penalties: &[Array2<f64>],
    ) -> Result<SmoothTestResult, SmoothScoreTestRefusal> {
        let covariance_scale = match scale {
            SmoothTestScale::Known => 1.0,
            SmoothTestScale::Estimated => fit.sigma2_hat,
        };
        smooth_score_test(SmoothScoreTestInput {
            beta: fit.beta.view(),
            penalized_hessian: &fit.hessian,
            weighted_gram: &fit.gram,
            coeff_range: design.term.clone(),
            structural_penalties,
            covariance_scale,
            residual_df: Some(fit.residual_df),
            scale,
        })
    }

    /// `y = 1 + 2·x1 + ε/√w` with unit dispersion: the other terms' truth is in
    /// their unpenalized span, so the residualized score has mean zero exactly
    /// and its null law is the stated one.
    fn null_response(design: &Design, rng: &mut StdRng) -> Array1<f64> {
        Array1::from_shape_fn(design.x.nrows(), |i| {
            1.0 + 2.0 * design.x[[i, 1]] + standard_normal(rng) / design.weights[i].sqrt()
        })
    }

    /// Two-sided calibration of the whole null law. The p-values must pass a
    /// Kolmogorov–Smirnov test of uniformity at level 0.001, and the size at
    /// 0.10, 0.05 and 0.01 must sit within three Monte-Carlo standard errors of
    /// the level on both sides: a conservative test fails exactly as an
    /// anti-conservative one does. Both reference laws are checked, the known
    /// scale `Σ w χ²₁` and the estimated scale over `χ²_ρ/ρ`.
    #[test]
    fn the_score_test_is_uniform_under_the_null() {
        let design = design(120);
        let reps = 2000;
        for scale in [SmoothTestScale::Known, SmoothTestScale::Estimated] {
            let mut rng = StdRng::seed_from_u64(0x5c07e);
            let mut p_values: Vec<f64> = (0..reps)
                .map(|_| test(&design, &fit(&design, &null_response(&design, &mut rng)), scale).p_value)
                .collect();
            p_values.sort_by(f64::total_cmp);
            let ks = p_values
                .iter()
                .enumerate()
                .map(|(i, &p)| ((i + 1) as f64 / reps as f64 - p).max(p - i as f64 / reps as f64))
                .fold(0.0_f64, f64::max);
            let ks_level: f64 = 0.001;
            let ks_critical = (-0.5 * (ks_level / 2.0).ln()).sqrt() / (reps as f64).sqrt();
            assert!(ks <= ks_critical, "{scale:?}: KS distance {ks} above {ks_critical}");
            for level in [0.10, 0.05, 0.01] {
                let size = p_values.iter().filter(|&&p| p <= level).count() as f64 / reps as f64;
                let mcse = (level * (1.0 - level) / reps as f64).sqrt();
                assert!(
                    (size - level).abs() <= 3.0 * mcse,
                    "{scale:?}: size {size} at level {level} (MCSE {mcse})"
                );
            }
        }
    }

    /// A real smooth effect of `x2` is detected.
    #[test]
    fn the_score_test_detects_a_smooth_effect() {
        let design = design(120);
        let reps = 200;
        let mut rng = StdRng::seed_from_u64(0xe77ec7);
        let mut rejections = 0;
        for _ in 0..reps {
            let mut y = null_response(&design, &mut rng);
            for i in 0..y.len() {
                y[i] += 0.8 * design.x[[i, 4]];
            }
            let fitted = fit(&design, &y);
            rejections += usize::from(test(&design, &fitted, SmoothTestScale::Estimated).p_value <= 0.05);
        }
        assert!(rejections as f64 / reps as f64 >= 0.9, "power {rejections}/{reps}");
    }

    /// A real effect in the null space of the wiggliness penalty — the term's
    /// linear part, as in a varying coefficient `z·x` — is detected. This is the
    /// direction that inverting the summed penalty `(Σ_l S_l/‖S_l‖_F)⁻¹` all but
    /// ignores: there the smoothest wiggly directions outweigh the null space by
    /// orders of magnitude, and at this amplitude that kernel rejects 12 of 200
    /// at 0.05 where this one rejects all 200.
    #[test]
    fn the_score_test_detects_an_effect_in_the_penalty_null_space() {
        let design = design(120);
        let (_, slope) = null_space_basis(design.term.len());
        let effect = design.x.slice(ndarray::s![.., design.term.clone()]).dot(&slope);
        let reps = 200;
        let mut rng = StdRng::seed_from_u64(0x9a11);
        let mut rejections = 0;
        for _ in 0..reps {
            let y = null_response(&design, &mut rng) + &(&effect * 0.6);
            let fitted = fit(&design, &y);
            rejections += usize::from(test(&design, &fitted, SmoothTestScale::Estimated).p_value <= 0.05);
        }
        assert!(rejections as f64 / reps as f64 >= 0.9, "power {rejections}/{reps}");
    }

    /// Rescaling any one structural penalty rescales its pseudo-inverse and its
    /// null mean together, so neither the p-value nor the reported statistic
    /// moves, whatever scale each penalty was built at.
    #[test]
    fn the_test_does_not_depend_on_the_penalty_scales() {
        let design = design(80);
        let mut rng = StdRng::seed_from_u64(7);
        let mut y = null_response(&design, &mut rng);
        for i in 0..y.len() {
            y[i] += 0.2 * design.x[[i, 6]];
        }
        let fitted = fit(&design, &y);
        let base = test(&design, &fitted, SmoothTestScale::Estimated);
        let scaled = test_with(
            &design,
            &fitted,
            SmoothTestScale::Estimated,
            &[&design.wiggle * 37.0, &design.null_space * 0.02],
        )
        .expect("the score test is defined");
        assert!((base.p_value - scaled.p_value).abs() <= 1e-10, "{base:?} vs {scaled:?}");
        assert!((base.statistic - scaled.statistic).abs() <= 1e-9 * base.statistic, "{base:?} vs {scaled:?}");
        assert!((base.ref_df - scaled.ref_df).abs() <= 1e-9 * base.ref_df, "{base:?} vs {scaled:?}");
    }

    /// Two centerings of one smooth are one model: `X₂ = X₁ + 1·cᵀ` on the term
    /// columns, with the intercept absorbing the constant. The weighted fit's
    /// residual is orthogonal to the intercept in the weighted metric, so the
    /// score moves by nothing and the test is the same in both gauges. This is
    /// the frequency-weights versus duplicated-rows case, where the sum-to-zero
    /// constraint is taken over different row multisets.
    #[test]
    fn the_test_does_not_depend_on_the_smooth_centering() {
        let first = design(90);
        let mut second = design(90);
        let c = [0.3, -0.2, 0.15, 0.05, -0.4, 0.1, 0.25, -0.05, 0.2, -0.3];
        for (k, shift) in second.term.clone().zip(c) {
            second.x.column_mut(k).mapv_inplace(|v| v + shift);
        }
        let mut rng = StdRng::seed_from_u64(11);
        let mut y = null_response(&first, &mut rng);
        for i in 0..y.len() {
            y[i] += 0.3 * first.x[[i, 5]];
        }
        let a = test(&first, &fit(&first, &y), SmoothTestScale::Estimated);
        let b = test(&second, &fit(&second, &y), SmoothTestScale::Estimated);
        assert!((a.p_value - b.p_value).abs() <= 1e-9, "{a:?} vs {b:?}");
        assert!((a.statistic - b.statistic).abs() <= 1e-8 * a.statistic, "{a:?} vs {b:?}");
    }

    /// The same smooth in a second coefficient chart, `β = T·γ`: term columns
    /// `X·T` and every penalty `TᵀST`.
    fn in_chart(source: &Design, chart: &Array2<f64>) -> Design {
        let mut moved = design(source.x.nrows());
        let term = source.term.clone();
        let columns = source.x.slice(ndarray::s![.., term.clone()]).dot(chart);
        moved.x.slice_mut(ndarray::s![.., term.clone()]).assign(&columns);
        let congruent = |penalty: &Array2<f64>| chart.t().dot(&penalty.dot(chart));
        moved.wiggle = congruent(&source.wiggle);
        moved.null_space = congruent(&source.null_space);
        let block = &(&moved.wiggle * 20.0) + &moved.null_space;
        moved.penalty.slice_mut(ndarray::s![term.clone(), term]).assign(&block);
        moved
    }

    /// A coefficient chart is not a property of the smooth: `β = T·γ` fits
    /// the same curve with the same penalty, so the test must return the same
    /// p-value and statistic. The Euclidean pseudo-inverse `S_l⁺` does not
    /// transform with the chart once the penalty ranges are not orthogonal in
    /// it; on this chart it moved the statistic from 3.39 on 3.28 reference df
    /// to 2.49 on 2.57. The components of a penalty that separates the block do
    /// transform with it.
    #[test]
    fn the_test_does_not_depend_on_the_coefficient_chart() {
        let first = design(100);
        let m = first.term.len();
        let chart = Array2::from_shape_fn((m, m), |(a, b)| match a.cmp(&b) {
            std::cmp::Ordering::Equal => 1.0 + 0.1 * a as f64,
            std::cmp::Ordering::Less => 0.4 / (b - a) as f64,
            std::cmp::Ordering::Greater => 0.0,
        });
        let second = in_chart(&first, &chart);
        let (_, slope) = null_space_basis(m);
        let linear = first.x.slice(ndarray::s![.., first.term.clone()]).dot(&slope);
        let mut rng = StdRng::seed_from_u64(0xc4a27);
        let mut y = null_response(&first, &mut rng);
        for i in 0..y.len() {
            y[i] += 0.25 * linear[i] + 0.2 * first.x[[i, 6]];
        }
        let a = test(&first, &fit(&first, &y), SmoothTestScale::Estimated);
        let b = test(&second, &fit(&second, &y), SmoothTestScale::Estimated);
        assert!((a.p_value - b.p_value).abs() <= 1e-9, "{a:?} vs {b:?}");
        assert!((a.statistic - b.statistic).abs() <= 1e-8 * a.statistic, "{a:?} vs {b:?}");
    }

    /// A null-space ridge charges functionals of the curve, and the row of a
    /// functional need not be the null function it pins down: the B-spline
    /// ridge charges the mean slope through the end coefficients. Here each row
    /// is a null function plus one of the roughest wiggly directions. The ridge
    /// still shrinks exactly the wiggliness null space, so the null-space
    /// component still covaries along the null functions, and the linear effect
    /// below is found. Its pseudo-inverse instead points along the rows, mostly
    /// wiggly, and found the effect in 62 of 200 fits at 0.05.
    #[test]
    fn a_ridge_along_a_functional_still_tests_the_null_space() {
        let mut design = design(120);
        let m = design.term.len();
        let (constant, slope) = null_space_basis(m);
        let (_, wiggly) = design.wiggle.eigh(faer::Side::Lower).expect("the wiggliness penalty is symmetric");
        let rows = [&constant + &(&wiggly.column(m - 1) * 3.0), &slope + &(&wiggly.column(m - 2) * 3.0)];
        design.null_space = Array2::from_shape_fn((m, m), |(a, b)| rows.iter().map(|row| row[a] * row[b]).sum());
        let block = &(&design.wiggle * 20.0) + &design.null_space;
        let term = design.term.clone();
        design.penalty.slice_mut(ndarray::s![term.clone(), term]).assign(&block);
        let effect = design.x.slice(ndarray::s![.., design.term.clone()]).dot(&slope);
        let reps = 200;
        let mut rng = StdRng::seed_from_u64(0xe4d5);
        let mut rejections = 0;
        for _ in 0..reps {
            let y = null_response(&design, &mut rng) + &(&effect * 0.6);
            let fitted = fit(&design, &y);
            rejections += usize::from(test(&design, &fitted, SmoothTestScale::Estimated).p_value <= 0.05);
        }
        assert!(rejections as f64 / reps as f64 >= 0.9, "power {rejections}/{reps}");
    }

    /// A direction that no penalty shrinks is a fixed effect of the term, which
    /// `τ = 0` does not remove: without the null-space penalty the test is
    /// refused, not computed on the penalized part.
    #[test]
    fn a_direction_no_penalty_shrinks_is_refused() {
        let design = design(60);
        let mut rng = StdRng::seed_from_u64(3);
        let fitted = fit(&design, &null_response(&design, &mut rng));
        let refused = test_with(&design, &fitted, SmoothTestScale::Estimated, &[design.wiggle.clone()]);
        assert_eq!(refused.unwrap_err(), SmoothScoreTestRefusal::UnpenalizedDirection);
        let refused = test_with(&design, &fitted, SmoothTestScale::Estimated, &[]);
        assert_eq!(refused.unwrap_err(), SmoothScoreTestRefusal::UnpenalizedDirection);
    }

    /// An observed-information curvature need not be positive semi-definite —
    /// a location-scale fit's joint `H − S(λ)` is indefinite away from a
    /// log-concave likelihood. Where it leaves the term's score covariance
    /// with a negative eigenvalue the score has no variance law, and the test
    /// is refused rather than computed on the positive part of the spectrum.
    #[test]
    fn an_indefinite_score_covariance_is_refused() {
        let design = design(60);
        let mut rng = StdRng::seed_from_u64(13);
        let mut fitted = fit(&design, &null_response(&design, &mut rng));
        let term = design.term.clone();
        let shift = 100.0 * fitted.gram.diag().sum();
        for k in term {
            fitted.gram[[k, k]] -= shift;
        }
        let refused = test_with(
            &design,
            &fitted,
            SmoothTestScale::Estimated,
            &[design.wiggle.clone(), design.null_space.clone()],
        );
        assert_eq!(refused.unwrap_err(), SmoothScoreTestRefusal::IndefiniteCurvature);
    }

    /// The estimated-scale reference needs its denominator degrees of freedom.
    #[test]
    fn an_estimated_scale_without_residual_df_is_refused() {
        let design = design(60);
        let mut rng = StdRng::seed_from_u64(5);
        let fitted = fit(&design, &null_response(&design, &mut rng));
        let refused = smooth_score_test(SmoothScoreTestInput {
            beta: fitted.beta.view(),
            penalized_hessian: &fitted.hessian,
            weighted_gram: &fitted.gram,
            coeff_range: design.term.clone(),
            structural_penalties: &[design.wiggle.clone(), design.null_space.clone()],
            covariance_scale: fitted.sigma2_hat,
            residual_df: None,
            scale: SmoothTestScale::Estimated,
        });
        assert_eq!(refused.unwrap_err(), SmoothScoreTestRefusal::ResidualDfUnavailable);
    }
}
