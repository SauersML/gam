//! The λ-selection domain, derived per coordinate from a penalized term's own
//! design-relative penalty spectrum (#2812).
//!
//! On a penalty's range each direction carries data curvature `γ_j` (a
//! generalized eigenvalue of the block's design Gram against the penalty,
//! quotiented by the penalty's null space) and penalty curvature `λ = e^ρ` in
//! the same units. What the outer search consumes is the criterion's gradient
//! in `ρ`, and in each direction that gradient is the direction's effective
//! degrees of freedom `γ_j / (γ_j + λ)` carried through an inverse of the
//! penalized Hessian, whose conditioning in that direction is `λ / γ_j` once
//! the penalty dominates and `γ_j / λ` once the data does. A quantity carried
//! through an inverse of condition `κ` holds relative error `ε κ`; the
//! direction's contribution and the error in it cross at `λ / γ_j = 1 / √ε`.
//! Above `ln(γ_max / √ε)` every direction's gradient is under its own
//! round-off: the term is switched off to the gradient's resolution. Below
//! `ln(√ε γ_min)` the same holds for the penalty's side: the term is
//! unpenalized to the gradient's resolution. Between the two the criterion's
//! gradient resolves the term. A search reaching either edge has found a
//! structural result, not a wall.

use crate::estimate::reml::reml_outer_engine::positive_eigenvalue_threshold;
use faer::Side;
use gam_linalg::faer_ndarray::FaerEigh;
use ndarray::{Array1, Array2};

/// Generalized eigenvalues `γ_j` of the Gram `G = XᵀX` (or `XᵀWX`) against the
/// penalty `S` on the penalty's range space, quotiented by `ker(S)`: with
/// `S = U D Uᵀ` and `A = UᵀGU` partitioned into null (`0`) and range (`r`)
/// parts, the eigenvalues of `D_r^{-1/2} (A_rr − A_r0 A_00⁺ A_0r) D_r^{-1/2}`.
/// The Schur complement matters whenever `G` couples the penalized range to
/// the null space. Returns `None` when the pair carries no usable geometry
/// (no positive penalty eigenvalue, or shapes that do not agree).
pub fn penalty_range_gammas_from_gram(gram: &Array2<f64>, s_dense: &Array2<f64>) -> Option<Vec<f64>> {
    penalty_range_gammas_with_shared_nullspace(gram, s_dense, s_dense)
}

/// Project a component against only the null space shared by ALL penalties.
/// A direction penalized by another component cannot absorb this component's
/// data curvature freely. Profiling it out as if it were unpenalized can erase
/// the entire spectrum for overlapping smooths. The resulting spectrum bounds
/// the component's data curvature before the other penalized directions are
/// profiled; it supplies a conservative domain as their strengths vary.
pub fn penalty_range_gammas_with_shared_nullspace(
    gram: &Array2<f64>,
    s_dense: &Array2<f64>,
    aggregate_penalty: &Array2<f64>,
) -> Option<Vec<f64>> {
    let p = s_dense.nrows();
    if p == 0 || s_dense.ncols() != p || gram.dim() != (p, p)
        || aggregate_penalty.dim() != (p, p) {
        return None;
    }
    let (s_evals, s_evecs) = s_dense.eigh(Side::Lower).ok()?;
    let s_max = s_evals.iter().fold(0.0_f64, |a, &b| a.max(b.abs()));
    if !(s_max > 0.0) {
        return None;
    }
    let s_thresh = positive_eigenvalue_threshold(s_evals.as_slice()?);
    let mut range_cols: Vec<usize> = Vec::new();
    let mut inv_sqrt_d: Vec<f64> = Vec::new();
    for (j, &dj) in s_evals.iter().enumerate() {
        if dj > s_thresh {
            range_cols.push(j);
            inv_sqrt_d.push(1.0 / dj.sqrt());
        }
    }
    let r = range_cols.len();
    if r == 0 {
        return None;
    }
    let mut y = Array2::<f64>::zeros((p, r));
    for (col, (&src, &w)) in range_cols.iter().zip(inv_sqrt_d.iter()).enumerate() {
        let u = s_evecs.column(src);
        for row in 0..p {
            y[(row, col)] = u[row] * w;
        }
    }
    let mut b = y.t().dot(gram).dot(&y);
    let (aggregate_evals, aggregate_evecs) = aggregate_penalty.eigh(Side::Lower).ok()?;
    let aggregate_threshold = positive_eigenvalue_threshold(aggregate_evals.as_slice()?);
    let null_cols: Vec<usize> = aggregate_evals.iter().enumerate()
        .filter_map(|(j, &value)| (value <= aggregate_threshold).then_some(j)).collect();
    if !null_cols.is_empty() {
        let r0 = null_cols.len();
        let mut u0 = Array2::<f64>::zeros((p, r0));
        for (col, &src) in null_cols.iter().enumerate() {
            let u = aggregate_evecs.column(src);
            for row in 0..p {
                u0[(row, col)] = u[row];
            }
        }
        let g00 = u0.t().dot(gram).dot(&u0);
        let g_r0 = y.t().dot(gram).dot(&u0);
        let mut g00_sym = g00.clone();
        for i in 0..r0 {
            for j in (i + 1)..r0 {
                let avg = 0.5 * (g00_sym[(i, j)] + g00_sym[(j, i)]);
                g00_sym[(i, j)] = avg;
                g00_sym[(j, i)] = avg;
            }
        }
        let (e0, v0) = g00_sym.eigh(Side::Lower).ok()?;
        let tol0 = positive_eigenvalue_threshold(e0.as_slice()?);
        for k in 0..r0 {
            if e0[k] <= tol0 {
                continue;
            }
            let inv_e = 1.0 / e0[k];
            let w_k = g_r0.dot(&v0.column(k));
            for i in 0..r {
                for j in 0..r {
                    b[(i, j)] -= inv_e * w_k[i] * w_k[j];
                }
            }
        }
    }
    let mut b_sym = b.clone();
    for i in 0..r {
        for j in (i + 1)..r {
            let avg = 0.5 * (b_sym[(i, j)] + b_sym[(j, i)]);
            b_sym[(i, j)] = avg;
            b_sym[(j, i)] = avg;
        }
    }
    let (b_evals, _) = b_sym.eigh(Side::Lower).ok()?;
    let gammas: Vec<f64> = b_evals
        .iter()
        .map(|&gj| if gj.is_finite() && gj > 0.0 { gj } else { 0.0 })
        .collect();
    if gammas.is_empty() {
        return None;
    }
    Some(gammas)
}

pub use gam_problem::{log_gradient_resolution, precision_box};

/// The resolvability interval `[ln(√ε γ_min), ln(γ_max / √ε)]` of one term over
/// the positive `γ_j`; `None` when no direction carries curvature.
pub fn resolvability_interval(gammas: &[f64]) -> Option<(f64, f64)> {
    let mut gamma_min = f64::INFINITY;
    let mut gamma_max = 0.0_f64;
    for &gamma in gammas {
        if gamma.is_finite() && gamma > 0.0 {
            gamma_min = gamma_min.min(gamma);
            gamma_max = gamma_max.max(gamma);
        }
    }
    if !(gamma_max > 0.0) {
        return None;
    }
    let lower = log_gradient_resolution() + gamma_min.ln();
    let upper = gamma_max.ln() - log_gradient_resolution();
    (lower < upper).then_some((lower, upper))
}

/// One coordinate's domain from its resolvability interval: intersected with
/// the representable log-strength range, and with a family-declared floor when
/// one is given (the multinomial's derived minimum strength).
pub fn coordinate_domain(interval: Option<(f64, f64)>, family_floor: Option<f64>) -> (f64, f64) {
    let (lo, hi) = interval.unwrap_or_else(precision_box);
    let mut lo = lo.max(gam_problem::LOG_STRENGTH_MIN);
    let hi = hi.min(gam_problem::LOG_STRENGTH_MAX);
    if let Some(floor) = family_floor {
        lo = lo.max(floor);
    }
    (lo, hi)
}

/// The per-coordinate domain of a penalized design given as one Gram over
/// ALL columns and one penalty block per ρ coordinate, each a local matrix on
/// a contiguous column range. A coordinate whose block cannot be projected
/// keeps the precision box.
pub fn resolvability_domain_from_gram_blocks<'a>(
    gram: &Array2<f64>,
    blocks: impl IntoIterator<Item = (std::ops::Range<usize>, &'a Array2<f64>)>,
    rho_dim: usize,
) -> (Array1<f64>, Array1<f64>) {
    let (box_lo, box_hi) = coordinate_domain(None, None);
    let mut lower = Array1::<f64>::from_elem(rho_dim, box_lo);
    let mut upper = Array1::<f64>::from_elem(rho_dim, box_hi);
    for (k, (range, local)) in blocks.into_iter().take(rho_dim).enumerate() {
        if range.end > gram.nrows() || range.start >= range.end {
            continue;
        }
        let block_gram = gram
            .slice(ndarray::s![range.start..range.end, range.start..range.end])
            .to_owned();
        let interval = penalty_range_gammas_from_gram(&block_gram, local)
            .as_deref()
            .and_then(resolvability_interval);
        if let Some(interval) = interval {
            let (lo, hi) = coordinate_domain(Some(interval), None);
            lower[k] = lo;
            upper[k] = hi;
        }
    }
    (lower, upper)
}
