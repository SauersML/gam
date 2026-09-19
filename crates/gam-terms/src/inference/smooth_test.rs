//! Wood-style smooth-component Wald tests.
//!
//! The test follows the rank-truncated covariance inverse used by Wood (2013):
//! the term's coefficient block is mapped into fitted-value space by the
//! design-whitening `R` (`RᵀR = X'WX`) and tested with a spectral
//! pseudo-inverse of the whitened covariance `R·V·Rᵀ` truncated at the
//! fractional rank `edf1 = Σ_block (2F − F·F)_ii`, `F = H⁻¹ X'WX`. The whitening
//! is essential — truncating the raw coefficient covariance keeps the
//! largest-variance (heavily-penalized, signal-free) directions and discards
//! the fitted function; whitening restores the generalized `(V, X'WX)`
//! eigenbasis whose leading directions are the least-penalized modes that carry
//! the fit (issue #2142). A fractional rank carries its share of the next
//! direction through Wood's 2×2 block, averaged over the block's two sign
//! orientations, and the statistic is referred to that construction's exact
//! null law `χ²_k + ν·χ²₁` rather than to a χ² at a separately estimated
//! reference d.f.: a rank-`round(edf)` statistic judged against
//! `χ²_{max(tr(F)²/tr(F²), rank)}` compares a quadratic of one dimension with a
//! law of another, which is what made interaction and `by` terms conservative.
//!
//! Bartlett and Lawley mean corrections are likelihood-ratio corrections, so
//! they are not applied here. In the ordinary unpenalized Gaussian model the
//! Wald statistic satisfies `T / q ~ F(q, ν)` exactly, while under a ridge
//! penalty even the one-parameter statistic becomes `(n / (n + λ))χ²₁` rather
//! than a central χ²/F reference target.

use gam_linalg::faer_ndarray::FaerEigh;
use gam_math::fractional_rank::fractional_rank_sf;
use gam_math::probability::{chi_square_sf, fisher_snedecor_sf};
use ndarray::{Array1, Array2, ArrayView1, s};
use std::ops::Range;

/// Whether the residual dispersion `φ` is known or estimated from the
/// fit.  Selects the reference distribution for the Wald p-value: `Known`
/// → `χ²_{ref_df}` (e.g. binomial/Poisson), `Estimated` → `F_{ref_df,
/// residual_df}` (e.g. Gaussian where `φ̂` carries its own sampling
/// variability).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SmoothTestScale {
    Known,
    Estimated,
}

/// Inputs to `wood_smooth_test`. `beta` is the full coefficient vector;
/// the term block being tested is `beta[coeff_range]`. `covariance` is the
/// matching posterior covariance Σ̂ (full p×p; the diagonal block is sliced
/// out). **`covariance` must be the scale-included posterior covariance**
/// (mgcv `Vb`/`Vp`, i.e. `H⁻¹` already multiplied by the dispersion `φ̂`),
/// so the Wald statistic `T = β̂'·Σ̂⁻·β̂` is dimensionless — the residual
/// dispersion has already been divided out and the F-statistic is `T/ref_df`
/// with *no* further `φ̂` factor. `influence_matrix` is the optional
/// coefficient-space influence `F = H⁻¹ X'WX` (full `p×p`); with the Gram it
/// gives the fractional test rank `edf1 = Σ_block (2F − F·F)_ii`, and without
/// it `tr(F_jj)² / tr(F_jj²)` is the legacy reference d.f.
/// `whitening_gram` is the optional term-block-aligned weighted design Gram
/// `G = X'WX` (`H − S(λ)`, full `p×p`, same coefficient layout as
/// `covariance`); when present the covariance is mapped into the Wood (2013)
/// *fitted-value* space `R·V·Rᵀ` (`RᵀR = G`) before the rank-`r` truncation, so
/// the pseudo-inverse keeps the directions that carry the estimated function
/// rather than the raw largest-variance (heavily-penalized) coefficient
/// directions. When absent the raw coefficient covariance is truncated directly
/// — a graceful fallback for persisted models whose Gram was not serialized.
/// `edf` is the smooth's effective d.f., the test rank when the Gram is supplied
/// without an influence matrix and the legacy truncation rank otherwise;
/// `nullspace_dim` is the fixed-effect (unpenalized) leading dimension within
/// the block, which the legacy raw-covariance split tests at full rank (the
/// whitened test needs no such floor: unpenalized directions carry the largest
/// whitened variance and lead the spectrum). `residual_df` is the denominator
/// d.f. for the `Estimated`-scale F branch. It is `None` when that inference
/// geometry is unavailable; the estimated-scale test then returns `None`
/// instead of inventing denominator degrees of freedom. The known-scale branch
/// does not consume it.
#[derive(Debug, Clone)]
pub struct SmoothTestInput<'a> {
    pub beta: ArrayView1<'a, f64>,
    pub covariance: &'a Array2<f64>,
    pub influence_matrix: Option<&'a Array2<f64>>,
    pub whitening_gram: Option<&'a Array2<f64>>,
    pub coeff_range: Range<usize>,
    pub edf: f64,
    pub nullspace_dim: usize,
    pub residual_df: Option<f64>,
    pub scale: SmoothTestScale,
}

/// Output of `wood_smooth_test`: the Wald statistic
/// `T = f̂ᵀ·Vf⁻ᵣ·f̂` (rank-`r` truncated pseudo-inverse of the design-whitened
/// covariance `Vf = R·V·Rᵀ`, fractional `r` included), the reference d.f. — the
/// rank `r` of the reference law on the whitened path, a χ²/F d.f. on the
/// legacy one — and the resulting direct-tail `p_value`.
#[derive(Debug, Clone)]
pub struct SmoothTestResult {
    pub statistic: f64,
    pub ref_df: f64,
    pub p_value: f64,
    /// The statistic `p_value` is the tail of: `T` itself against `χ²_{ref_df}`
    /// for a known scale, `F = T/ref_df` against `F_{ref_df, residual_df}` for
    /// an estimated one.
    pub reference_statistic: f64,
}

/// Wood (2013) rank-truncated Wald smooth-component test.
///
/// Maps the term block `beta[coeff_range]` (and its posterior covariance
/// subblock) into the fitted-value space `f = R·β` — where `RᵀR = G` is the
/// term's weighted design Gram `G = X'WX` supplied in `whitening_gram` — and
/// tests it with the rank-`r` spectral pseudo-inverse of the whitened
/// covariance `Vf = R·V·Rᵀ` at the fractional rank `r = edf1` (`edf` when no
/// influence matrix is supplied). The statistic is compared against the exact
/// null law of the rank-`r` construction, `χ²_k + ν·χ²₁` (`k = ⌊r⌋`,
/// `ν = r − k`), when the
/// scale is `Known`, and against that law over an independent `χ²_ρ/ρ`,
/// `ρ = residual_df`, when `Estimated`; at integer `r` these are `χ²_r` and
/// `r·F_{r,ρ}`. Without the Gram the legacy raw-covariance split is referred
/// to `χ²_{ref_df}` or `F_{ref_df, residual_df}` at the trace-corrected d.f.
///
/// The whitening is the crux of Wood (2013): the raw coefficient covariance `V`
/// orders its eigen-directions by *coefficient* variance, which for a genuinely
/// wiggly smooth places the estimated signal in the small-variance
/// best-determined directions — so truncating `V` directly and keeping its
/// *largest* eigenvalues discards exactly the fitted function and reports a
/// dominant term as non-significant (issue #2142). Whitening by the design Gram
/// restores the generalized eigenbasis of `(V, G)`, in which the largest
/// whitened-variance directions are the least-penalized modes that carry the
/// fit; the rank-`r` truncation then keeps the signal. The statistic is
/// invariant to any uniform rescaling of `G`, so whether the Gram carries the
/// dispersion `φ̂` is irrelevant. When `whitening_gram` is `None` the raw
/// covariance is truncated unchanged (graceful fallback for persisted models
/// whose Gram was dropped).
///
/// Because `covariance` is the scale-included posterior covariance, `T`
/// already has the dispersion `φ̂` divided out (it is a proper Wald χ²);
/// the estimated-scale F-statistic is therefore `T/ref_df` with no extra
/// `φ̂` factor. Dividing by `φ̂` a second time — the historical defect
/// fixed in issue #675 — makes the p-value scale as `1/φ̂` and so depend on
/// the units of the response. Returns `None` on degenerate inputs (empty
/// block, non-finite EDF, non-finite stat, or non-positive residual d.f.
/// in the F branch).
pub fn wood_smooth_test(input: SmoothTestInput<'_>) -> Option<SmoothTestResult> {
    let start = input.coeff_range.start;
    let end = input.coeff_range.end;
    if start >= end
        || end > input.beta.len()
        || end > input.covariance.nrows()
        || end > input.covariance.ncols()
        || !input.edf.is_finite()
        || input.edf <= 0.0
    {
        return None;
    }
    let k = end - start;
    let beta = input.beta.slice(s![start..end]).to_owned();
    let cov = block(input.covariance, start, end)?;
    let null_dim = input.nullspace_dim.min(k);

    // Two regimes, selected by whether the design Gram is supplied:
    //
    //   * With `whitening_gram` (the `summary()` paths): the genuine Wood (2013)
    //     test. Map `(β, V)` into fitted-value space `(R·β, R·V·Rᵀ)`
    //     (`RᵀR = X'WX`) and test it at the term's fractional rank `edf1`
    //     against that rank's exact reference law.
    //   * Without it (persisted models, ANOVA-binding / multinomial callers whose
    //     covariance is already in a projected frame): the legacy null/penalized
    //     split on the raw covariance — a full-rank quadratic over the leading
    //     `null_dim` unpenalized coordinates plus a rank-`round(edf − null_dim)`
    //     truncation of the trailing penalized block. Preserved byte-for-byte so
    //     no non-summary caller shifts.
    if let Some((beta_w, cov_w)) = input
        .whitening_gram
        .and_then(|g| block(g, start, end))
        .and_then(|g| whiten_to_fitted_space(&beta, &cov, &g))
    {
        let rank = match input.influence_matrix {
            Some(influence) => alternative_edf(influence, start, end)?,
            None => input.edf,
        };
        let residual_df = match input.scale {
            SmoothTestScale::Known => None,
            SmoothTestScale::Estimated => Some(
                input
                    .residual_df
                    .filter(|value| value.is_finite() && *value > 0.0)?,
            ),
        };
        return fractional_rank_test(&beta_w, &cov_w, rank, residual_df);
    }

    // `rank_used` is the number of covariance directions actually summed; it can
    // fall below the requested rank on a rank-deficient block. The χ²/F
    // reference d.f. is floored at it so a boundary-shrunk term (whose Wood
    // influence-trace d.f. collapses toward 0) is never judged against a
    // degenerate ~0-d.f. reference — the mechanism that turned a *zero* Wald
    // statistic into p≈0 for a term the fit removed (#1360).
    let (statistic, rank_used) = legacy_split_quadratic(&beta, &cov, null_dim, input.edf)?;

    if rank_used == 0 {
        // No estimable direction in the block (every covariance eigenmode is
        // numerically null): the term carries no testable signal.
        return None;
    }
    // Wood (2013) influence-trace participation d.f. when available, but never
    // below `rank_used`. The historical fallback to `edf` collapsed to ~0 for a
    // shrunk term, making `χ²_{ref_df→0}` degenerate.
    let ref_df = match reference_df(input.influence_matrix, start, end) {
        Some(rd) if rd.is_finite() && rd > 0.0 => rd.max(rank_used as f64),
        _ => rank_used as f64,
    };
    if !statistic.is_finite() || statistic < 0.0 || !ref_df.is_finite() || ref_df <= 0.0 {
        return None;
    }
    let (reference_statistic, p_value) = match input.scale {
        SmoothTestScale::Known => (statistic, chi_square_sf(statistic, ref_df)),
        SmoothTestScale::Estimated => {
            let residual_df = input
                .residual_df
                .filter(|value| value.is_finite() && *value > 0.0)?;
            // `statistic` is already a dispersion-free Wald χ² (the covariance
            // is scale-included), so the estimated-scale F-statistic is the
            // χ² divided by its reference d.f. only — mgcv's `Tr/rank`. Dividing
            // by `φ̂` again would re-introduce a response-unit dependence (#675).
            let f_stat = statistic / ref_df;
            (f_stat, fisher_snedecor_sf(f_stat, ref_df, residual_df))
        }
    };
    if !p_value.is_finite() {
        return None;
    }
    Some(SmoothTestResult {
        statistic,
        ref_df,
        p_value,
        reference_statistic,
    })
}

fn block(matrix: &Array2<f64>, start: usize, end: usize) -> Option<Array2<f64>> {
    if start >= end || end > matrix.nrows() || end > matrix.ncols() {
        return None;
    }
    Some(matrix.slice(s![start..end, start..end]).to_owned())
}

/// Wood's alternative effective degrees of freedom of the block,
/// `edf1 = Σ_{i∈block} (2F − F·F)_ii` over the FULL influence matrix
/// `F = H⁻¹X'WX`, so the cross-term leverages `F_ij F_ji` with coefficients of
/// other terms enter each diagonal. `edf1` accounts for the smoothing parameter
/// having been estimated, and sits between `edf` and the block dimension.
fn alternative_edf(influence: &Array2<f64>, start: usize, end: usize) -> Option<f64> {
    let p = influence.nrows();
    if influence.ncols() != p || end > p {
        return None;
    }
    let edf1 = (start..end)
        .map(|i| 2.0 * influence[[i, i]] - influence.row(i).dot(&influence.column(i)))
        .sum::<f64>();
    edf1.is_finite().then_some(edf1)
}

/// The resolved eigen-spectrum of a whitened covariance, largest first, as
/// `(λ_i, z_i)` pairs with `z_i = v_iᵀ·f̂` the fitted block's coordinate on the
/// eigenvector. An eigenvalue inside the eigensolve's backward-error band
/// `n·ε·max|λ|` cannot be told from zero and is left out. `None` when nothing
/// is resolved.
fn resolved_spectrum(beta: &Array1<f64>, cov: &Array2<f64>) -> Option<Vec<(f64, f64)>> {
    let (evals, evecs) = cov.to_owned().eigh(faer::Side::Lower).ok()?;
    let max_abs_eigenvalue = evals
        .iter()
        .copied()
        .fold(0.0_f64, |acc, v| acc.max(v.abs()));
    let tol = evals.len() as f64 * f64::EPSILON * max_abs_eigenvalue;
    let mut spectrum: Vec<(f64, f64)> = (0..evals.len())
        .filter(|&i| evals[i] > tol)
        .map(|i| (evals[i], beta.dot(&evecs.column(i))))
        .collect();
    spectrum.sort_by(|a, b| b.0.total_cmp(&a.0));
    (!spectrum.is_empty() && spectrum.iter().all(|(l, z)| l.is_finite() && z.is_finite()))
        .then_some(spectrum)
}

/// Wood (2013) test of a whitened block at a fractional rank `r`.
///
/// With `u_i = z_i/√λ_i` the standardized coordinates on the covariance's
/// eigen-directions, largest variance first, `k = ⌊r⌋` and `ν = r − k`, the
/// statistic is
///
/// `T = Σ_{i≤k} u_i² + ν·u_{k+1}²`,
///
/// whose exact null law is `χ²_k + ν·χ²₁` (see [`gam_math::fractional_rank`]):
/// its mean is `r`, it moves continuously with `r`, and it is `χ²_r` at integer
/// `r`. Wood's rank-`r` pseudo-inverse carries the share `ν` of the
/// `(k+1)`-st direction through the 2×2 block `[[1, ±b₁₂], [±b₁₂, ν]]`,
/// `b₁₂ = √(ν(1 − ν)/2)`; the sign is fixed by an arbitrary eigenvector
/// orientation, and `T` is the mean of the two orientations' statistics, the
/// cross term `±2b₁₂u_k u_{k+1}` cancelling. Averaging the two *statistics*
/// keeps the reference law exact; averaging their two tail probabilities, as
/// mgcv does, does not, and is conservative. At `r < 1` (`k = 0`) the
/// statistic is `ν·u₁²` on `ν·χ²₁`, the leading direction's one-degree test.
///
/// A rank above the number of resolved eigen-directions is lowered to it,
/// where the rank becomes an integer; a non-positive rank carries no testable
/// direction and gives `None`. `residual_df` selects the estimated-scale law,
/// in which the statistic is divided by an independent `χ²_ρ/ρ`.
fn fractional_rank_test(
    beta: &Array1<f64>,
    cov: &Array2<f64>,
    rank: f64,
    residual_df: Option<f64>,
) -> Option<SmoothTestResult> {
    if !(rank.is_finite() && rank > 0.0) {
        return None;
    }
    let spectrum = resolved_spectrum(beta, cov)?;
    let rank = rank.min(spectrum.len() as f64);
    let k = rank.floor() as usize;
    let nu = rank - k as f64;
    let standardized_square = |i: usize| spectrum[i].1 * spectrum[i].1 / spectrum[i].0;
    let whole = (0..k).map(standardized_square).sum::<f64>();
    let statistic = if nu > 0.0 {
        whole + nu * standardized_square(k)
    } else {
        whole
    };
    let p_value = fractional_rank_sf(statistic, rank, residual_df).probability;
    // With an estimated scale the law is `rank·F` at an integer rank, so the
    // F-scale statistic `p_value` is the tail of is `T/rank`.
    let reference_statistic = match residual_df {
        Some(_) => statistic / rank,
        None => statistic,
    };
    (statistic.is_finite() && p_value.is_finite()).then_some(SmoothTestResult {
        statistic,
        ref_df: rank,
        p_value,
        reference_statistic,
    })
}

/// Legacy raw-covariance smooth test used when no design Gram is available:
/// a full-rank quadratic over the leading `null_dim` unpenalized coordinates
/// plus a rank-`round(edf − null_dim)` truncation of the trailing penalized
/// block, both on the raw coefficient covariance. Returns the summed statistic
/// and the total number of covariance directions actually used. This is a
/// reparameterization-*dependent* approximation of Wood (2013) — the whitened
/// path supersedes it — but it is retained bit-for-bit for the ANOVA-binding,
/// multinomial and persisted-model callers that never carry `X'WX`.
fn legacy_split_quadratic(
    beta: &Array1<f64>,
    cov: &Array2<f64>,
    null_dim: usize,
    edf: f64,
) -> Option<(f64, usize)> {
    let k = beta.len();
    let null_dim = null_dim.min(k);
    let pen_dim = k.saturating_sub(null_dim);
    let mut statistic = 0.0;
    let mut rank_used = 0usize;
    if null_dim > 0 {
        let beta_null = beta.slice(s![0..null_dim]).to_owned();
        let cov_null = cov.slice(s![0..null_dim, 0..null_dim]).to_owned();
        let (q, used) = truncated_quadratic(&beta_null, &cov_null, null_dim)?;
        statistic += q;
        rank_used += used;
    }
    if pen_dim > 0 {
        let beta_pen = beta.slice(s![null_dim..k]).to_owned();
        let cov_pen = cov.slice(s![null_dim..k, null_dim..k]).to_owned();
        let rank = truncated_rank(edf - null_dim as f64, pen_dim);
        if rank > 0 {
            let (q, used) = truncated_quadratic(&beta_pen, &cov_pen, rank)?;
            statistic += q;
            rank_used += used;
        }
    }
    Some((statistic, rank_used))
}

fn truncated_rank(edf_pen: f64, pen_dim: usize) -> usize {
    if pen_dim == 0 || !edf_pen.is_finite() || edf_pen <= 0.0 {
        return 0;
    }
    (edf_pen.round() as usize).clamp(1, pen_dim)
}

/// Map a term's coefficient-space `(β, V)` into its Wood (2013) fitted-value
/// space using the weighted design Gram `G = X'WX` (`G = RᵀR`). Returns
/// `(R·β, R·V·Rᵀ)`, where `R` has one row `√μ_i · u_iᵀ` per eigenpair
/// `(μ_i, u_i)` of `G` whose eigenvalue clears a relative tolerance. A
/// rank-deficient Gram (degenerate design, collinear tensor margins) therefore
/// yields a lower-dimensional fitted space rather than a failure; `None` only
/// when `G` has no positive eigenvalue (no estimable fitted direction) or the
/// shapes disagree. `R·V·Rᵀ` is symmetrized to absorb round-off so the
/// downstream eigendecomposition sees an exactly symmetric matrix.
fn whiten_to_fitted_space(
    beta: &Array1<f64>,
    cov: &Array2<f64>,
    gram: &Array2<f64>,
) -> Option<(Array1<f64>, Array2<f64>)> {
    let k = beta.len();
    if gram.nrows() != k || gram.ncols() != k || cov.nrows() != k || cov.ncols() != k {
        return None;
    }
    let (evals, evecs) = gram.to_owned().eigh(faer::Side::Lower).ok()?;
    let max_ev = evals
        .iter()
        .copied()
        .fold(0.0_f64, |acc, v| acc.max(v.abs()));
    if max_ev <= 0.0 {
        return None;
    }
    // A Gram eigenvalue inside the symmetric eigensolve's backward-error band
    // `k·ε·max|μ|` cannot be told from zero; those modes are the null space.
    let tol = k as f64 * f64::EPSILON * max_ev;
    let rows: Vec<usize> = (0..evals.len()).filter(|&i| evals[i] > tol).collect();
    if rows.is_empty() {
        return None;
    }
    // R (r×k): row i = √μ_i · u_iᵀ, so RᵀR = Σ μ_i u_i u_iᵀ = G (up to the
    // dropped near-null modes) and R maps coefficients to fitted-value coords.
    let mut r_mat = Array2::<f64>::zeros((rows.len(), k));
    for (ri, &i) in rows.iter().enumerate() {
        let scale = evals[i].sqrt();
        let u = evecs.column(i);
        for j in 0..k {
            r_mat[[ri, j]] = scale * u[j];
        }
    }
    let beta_w = r_mat.dot(beta);
    let mut cov_w = r_mat.dot(cov).dot(&r_mat.t());
    gam_linalg::matrix::symmetrize_in_place(&mut cov_w);
    Some((beta_w, cov_w))
}

/// Returns the rank-`rank` truncated Wald quadratic together with the number of
/// covariance directions (eigenmodes above the eigensolver's rounding band)
/// that were actually summed into it. The `used` count is the *effective rank of
/// the statistic*: it can fall below `rank` when the covariance subblock is
/// itself rank-deficient. Callers fold it into the χ² reference degrees of
/// freedom so the tail probability is never evaluated against a degenerate
/// ~0 d.f.
fn truncated_quadratic(beta: &Array1<f64>, cov: &Array2<f64>, rank: usize) -> Option<(f64, usize)> {
    if beta.is_empty() || cov.nrows() != beta.len() || cov.ncols() != beta.len() || rank == 0 {
        return None;
    }
    let (evals, evecs) = cov.to_owned().eigh(faer::Side::Lower).ok()?;
    let mut order: Vec<usize> = (0..evals.len()).collect();
    order.sort_by(|&a, &b| evals[b].total_cmp(&evals[a]));
    // A covariance eigenvalue inside the eigensolve's backward-error band
    // `k·ε·max|λ|` cannot be told from zero, so it has no inverse to sum.
    let max_abs_eigenvalue = evals
        .iter()
        .copied()
        .fold(0.0_f64, |acc, v| acc.max(v.abs()));
    let tol = evals.len() as f64 * f64::EPSILON * max_abs_eigenvalue;
    let mut q = 0.0;
    let mut used = 0usize;
    for idx in order {
        let lambda = evals[idx];
        if lambda <= tol {
            continue;
        }
        let v = evecs.column(idx);
        let proj = beta.dot(&v);
        q += proj * proj / lambda;
        used += 1;
        if used >= rank {
            break;
        }
    }
    (used > 0 && q.is_finite()).then_some((q.max(0.0), used))
}

fn reference_df(influence: Option<&Array2<f64>>, start: usize, end: usize) -> Option<f64> {
    let f = influence?;
    let f_block = block(f, start, end)?;
    let tr = (0..f_block.nrows()).map(|i| f_block[[i, i]]).sum::<f64>();
    let tr2 = f_block.dot(&f_block).diag().sum();
    if tr.is_finite() && tr2.is_finite() && tr > 0.0 && tr2 > 0.0 {
        Some(tr * tr / tr2)
    } else {
        None
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::array;
    use rand::rngs::StdRng;
    use rand::{RngExt, SeedableRng};
    use statrs::function::gamma::gamma_ur;

    #[test]
    fn reference_df_uses_trace_correction() {
        let beta = array![1.0, 2.0];
        let cov = array![[2.0, 0.0], [0.0, 3.0]];
        let f = array![[0.5, 0.0], [0.0, 0.25]];
        let out = wood_smooth_test(SmoothTestInput {
            beta: beta.view(),
            covariance: &cov,
            influence_matrix: Some(&f),
            whitening_gram: None,
            coeff_range: 0..2,
            edf: 1.0,
            nullspace_dim: 0,
            residual_df: None,
            scale: SmoothTestScale::Known,
        })
        .expect("smooth test");
        assert!((out.ref_df - 1.8).abs() < 1e-12);
        assert!(out.statistic > 0.0);
        assert!((0.0..=1.0).contains(&out.p_value));
    }

    #[test]
    fn known_scale_branch_reports_plain_wald_chi_square() {
        let beta = array![1.0, 2.0];
        let cov = array![[2.0, 0.0], [0.0, 3.0]];
        let f = array![[0.5, 0.0], [0.0, 0.25]];
        let out = wood_smooth_test(SmoothTestInput {
            beta: beta.view(),
            covariance: &cov,
            influence_matrix: Some(&f),
            whitening_gram: None,
            coeff_range: 0..2,
            edf: 1.0,
            nullspace_dim: 0,
            residual_df: None,
            scale: SmoothTestScale::Known,
        })
        .expect("smooth test");

        let expected = gamma_ur(0.5 * out.ref_df, 0.5 * out.statistic);
        assert!((out.p_value - expected).abs() < 1e-15);
    }

    #[test]
    fn estimated_scale_refuses_missing_residual_degrees_of_freedom() {
        let beta = array![1.0, 2.0];
        let covariance = array![[2.0, 0.0], [0.0, 3.0]];
        let result = wood_smooth_test(SmoothTestInput {
            beta: beta.view(),
            covariance: &covariance,
            influence_matrix: None,
            whitening_gram: None,
            coeff_range: 0..2,
            edf: 1.0,
            nullspace_dim: 0,
            residual_df: None,
            scale: SmoothTestScale::Estimated,
        });
        assert!(
            result.is_none(),
            "estimated-scale inference must be omitted when denominator d.f. is unavailable"
        );
    }

    /// Rescaling the response by `c` is `β → c·β`, `Σ → c²·Σ` (the covariance
    /// is scale-included). The Wald statistic `T = β'Σ⁻β` is then invariant,
    /// and — because the estimated-scale F-statistic is `T/ref_df` with no
    /// further `φ̂` factor — so is the p-value. This is the unit-level guard
    /// for issue #675: the historical `T/(ref_df·φ̂)` made the p-value scale
    /// as `1/c²` even though `T` did not move.
    #[test]
    fn estimated_scale_pvalue_is_response_unit_invariant() {
        let beta = array![2.5, -3.5, 1.8];
        let cov = array![[2.0, 0.3, 0.0], [0.3, 1.5, 0.1], [0.0, 0.1, 0.9]];
        let f = array![[0.7, 0.0, 0.0], [0.0, 0.6, 0.0], [0.0, 0.0, 0.4]];

        let run = |c: f64| {
            let beta_c = &beta * c;
            let cov_c = &cov * (c * c);
            wood_smooth_test(SmoothTestInput {
                beta: beta_c.view(),
                covariance: &cov_c,
                influence_matrix: Some(&f),
                whitening_gram: None,
                coeff_range: 0..3,
                edf: 2.0,
                nullspace_dim: 0,
                residual_df: Some(50.0),
                scale: SmoothTestScale::Estimated,
            })
            .expect("smooth test")
        };

        let base = run(1.0);
        assert!(base.statistic > 0.0);
        // A non-trivial, clearly-significant p-value so the invariance check is
        // not vacuously comparing two values pinned at a boundary.
        assert!(base.p_value > 0.0 && base.p_value < 0.05);
        for c in [1e-3, 0.1, 10.0, 1e3, 1e6] {
            let scaled = run(c);
            let rel_stat = (scaled.statistic - base.statistic).abs() / base.statistic;
            assert!(
                rel_stat < 1e-9,
                "Wald statistic not scale-invariant at c={c}: {} vs {}",
                scaled.statistic,
                base.statistic
            );
            let rel_p = (scaled.p_value - base.p_value).abs() / base.p_value;
            assert!(
                rel_p < 1e-9,
                "estimated-scale p-value not scale-invariant at c={c}: {} vs {}",
                scaled.p_value,
                base.p_value
            );
        }
    }

    /// A term the fit drove to the penalty boundary (coefficients ≈ 0, EDF → 0)
    /// must read as *not* significant. The defect (#1360): the reference d.f.
    /// fell back to `edf` and collapsed toward 0, so the χ² tail of a *zero*
    /// statistic evaluated at ~0 d.f. degenerated to p ≈ 0 — an overwhelming
    /// false positive for a term that was removed. The reference d.f. is now
    /// floored at the rank actually summed (≥ 1), so a zero statistic returns
    /// p ≈ 1.
    #[test]
    fn boundary_shrunk_term_is_not_significant() {
        // Near-zero coefficients with a well-conditioned (non-degenerate)
        // covariance: the Wald statistic is ~0 regardless of how the reference
        // d.f. is formed.
        let beta = array![1e-9, -2e-9, 5e-10];
        let cov = array![[0.04, 0.0, 0.0], [0.0, 0.05, 0.0], [0.0, 0.0, 0.06]];
        // A degenerate influence block (sign-flipped near-zero leverages) so the
        // Wood trace correction is unavailable and the fallback is exercised.
        let f = array![[1e-9, 0.0, 0.0], [0.0, -1e-9, 0.0], [0.0, 0.0, 1e-12]];
        for scale in [SmoothTestScale::Known, SmoothTestScale::Estimated] {
            let out = wood_smooth_test(SmoothTestInput {
                beta: beta.view(),
                covariance: &cov,
                influence_matrix: Some(&f),
                whitening_gram: None,
                coeff_range: 0..3,
                edf: 1e-6,
                nullspace_dim: 0,
                residual_df: Some(500.0),
                scale,
            })
            .expect("boundary term still produces a result");
            assert!(
                out.ref_df >= 1.0,
                "reference d.f. must not collapse below the tested rank: {}",
                out.ref_df
            );
            assert!(
                out.statistic < 1e-6,
                "boundary statistic should be ~0: {}",
                out.statistic
            );
            assert!(
                out.p_value > 0.5,
                "shrunk boundary term must not be significant (p={}, scale={:?})",
                out.p_value,
                scale
            );
        }
    }

    /// Flooring the reference d.f. at the tested rank must not weaken a genuinely
    /// significant term: a large statistic with a healthy influence block keeps
    /// its small p-value (the floor only raises a *degenerate* sub-1 d.f.).
    #[test]
    fn floor_does_not_blunt_a_real_signal() {
        let beta = array![6.0, -5.0];
        let cov = array![[1.0, 0.0], [0.0, 1.0]];
        let f = array![[0.9, 0.0], [0.0, 0.9]];
        let out = wood_smooth_test(SmoothTestInput {
            beta: beta.view(),
            covariance: &cov,
            influence_matrix: Some(&f),
            whitening_gram: None,
            coeff_range: 0..2,
            edf: 2.0,
            nullspace_dim: 2,
            residual_df: None,
            scale: SmoothTestScale::Known,
        })
        .expect("smooth test");
        assert!(out.statistic > 40.0, "statistic={}", out.statistic);
        assert!(
            out.p_value < 1e-6,
            "a strong term must stay significant: p={}",
            out.p_value
        );
    }

    /// The #2142 root cause, isolated: a dominant smooth whose fitted signal
    /// lives in the *best-determined* (small raw-variance) coefficient direction
    /// while an orthogonal, signal-free direction carries all the raw variance.
    /// Truncating the raw covariance to rank 1 keeps the large-variance
    /// direction — projecting the signal onto ~0 and reporting p ≈ 1 — whereas
    /// the design-whitened truncation keeps the least-penalized (large
    /// whitened-variance) direction that actually holds the fit, recovering a
    /// tiny p. Same `(β, V)`; the only difference is whether the weighted Gram
    /// is supplied.
    #[test]
    fn whitening_recovers_signal_the_raw_truncation_discards() {
        // Direction 0 (e0) is tightly determined (small posterior variance) and
        // holds all the coefficient signal; direction 1 (e1) is loose and empty.
        let beta = array![5.0, 0.0];
        let cov = array![[0.01, 0.0], [0.0, 1.0]];
        // Weighted Gram: e0 carries far more Fisher information (X'WX_00 ≫ _11),
        // which is precisely *why* its posterior variance is small. Whitening by
        // it makes the whitened variance of e0 (g0·V00 = 4) exceed that of e1
        // (g1·V11 = 1), so the rank-1 cut keeps e0.
        let gram = array![[400.0, 0.0], [0.0, 1.0]];

        let raw = wood_smooth_test(SmoothTestInput {
            beta: beta.view(),
            covariance: &cov,
            influence_matrix: None,
            whitening_gram: None,
            coeff_range: 0..2,
            edf: 1.0,
            nullspace_dim: 0,
            residual_df: None,
            scale: SmoothTestScale::Known,
        })
        .expect("raw smooth test");
        assert!(
            raw.statistic < 1e-6 && raw.p_value > 0.5,
            "raw truncation must keep the empty large-variance direction (the bug): stat={}, p={}",
            raw.statistic,
            raw.p_value
        );

        let whitened = wood_smooth_test(SmoothTestInput {
            beta: beta.view(),
            covariance: &cov,
            influence_matrix: None,
            whitening_gram: Some(&gram),
            coeff_range: 0..2,
            edf: 1.0,
            nullspace_dim: 0,
            residual_df: None,
            scale: SmoothTestScale::Known,
        })
        .expect("whitened smooth test");
        assert!(
            whitened.statistic > 100.0 && whitened.p_value < 1e-6,
            "whitened truncation must keep the signal direction: stat={}, p={}",
            whitened.statistic,
            whitened.p_value
        );
    }

    /// The Wald statistic is invariant to any uniform rescaling `G → c·G` of the
    /// whitening Gram: `R → √c·R` scales `R·β` by `√c` and `R·V·Rᵀ` by `c`, and
    /// the two factors cancel in `(R·β)ᵀ (R·V·Rᵀ)⁻ (R·β)`. This is why passing
    /// the raw `X'WX` (no `φ̂`) is correct even though the covariance is
    /// scale-included.
    #[test]
    fn whitening_statistic_is_invariant_to_gram_scaling() {
        let beta = array![2.0, -1.5, 0.7];
        let cov = array![[0.02, 0.0, 0.0], [0.0, 0.3, 0.0], [0.0, 0.0, 0.9]];
        let gram_base = array![[50.0, 1.0, 0.0], [1.0, 8.0, 0.5], [0.0, 0.5, 2.0]];
        let run = |c: f64| {
            let g = &gram_base * c;
            wood_smooth_test(SmoothTestInput {
                beta: beta.view(),
                covariance: &cov,
                influence_matrix: None,
                whitening_gram: Some(&g),
                coeff_range: 0..3,
                edf: 2.0,
                nullspace_dim: 0,
                residual_df: None,
                scale: SmoothTestScale::Known,
            })
            .expect("whitened smooth test")
        };
        let base = run(1.0);
        assert!(base.statistic > 0.0);
        for c in [1e-6, 1e-2, 7.0, 1e3, 1e6] {
            let scaled = run(c);
            let rel = (scaled.statistic - base.statistic).abs() / base.statistic;
            assert!(
                rel < 1e-9,
                "statistic not Gram-scale-invariant at c={c}: {} vs {}",
                scaled.statistic,
                base.statistic
            );
        }
    }

    /// A rank-deficient whitening Gram (e.g. a collinear/degenerate term design)
    /// must degrade gracefully to a lower-dimensional fitted space rather than
    /// error: the surviving direction is still tested and yields a finite result.
    #[test]
    fn whitening_tolerates_rank_deficient_gram() {
        let beta = array![3.0, 1.0];
        let cov = array![[0.05, 0.0], [0.0, 0.4]];
        // Rank-1 Gram: only the e0 fitted direction is estimable.
        let gram = array![[9.0, 0.0], [0.0, 0.0]];
        let out = wood_smooth_test(SmoothTestInput {
            beta: beta.view(),
            covariance: &cov,
            influence_matrix: None,
            whitening_gram: Some(&gram),
            coeff_range: 0..2,
            edf: 2.0,
            nullspace_dim: 0,
            residual_df: None,
            scale: SmoothTestScale::Known,
        })
        .expect("rank-deficient Gram still yields a result");
        // Only one fitted direction survives, so the reference d.f. is 1.
        assert!((out.ref_df - 1.0).abs() < 1e-9, "ref_df={}", out.ref_df);
        assert!(out.statistic.is_finite() && out.statistic > 0.0);
        assert!((0.0..=1.0).contains(&out.p_value));
    }

    /// A Gram mode far below `1e-10` of the largest but above the eigensolver's
    /// rounding band is a real fitted direction and stays in the whitened frame.
    #[test]
    fn whitening_keeps_a_resolved_small_gram_mode() {
        let beta = array![1.0, 1.0];
        let cov = array![[0.5, 0.0], [0.0, 0.5]];
        let gram = array![[1.0, 0.0], [0.0, 1e-12]];
        let (_, cov_w) = whiten_to_fitted_space(&beta, &cov, &gram).expect("whitened frame");
        assert_eq!(cov_w.nrows(), 2, "the 1e-12 Gram mode is resolved");
    }

    /// A covariance mode between the eigensolver's rounding band and `1e-10` of
    /// the largest is resolved, so its projection enters the Wald quadratic.
    #[test]
    fn truncated_quadratic_sums_a_resolved_small_covariance_mode() {
        let beta = array![0.0, 1e-6];
        let cov = array![[1.0, 0.0], [0.0, 1e-12]];
        let (q, used) = truncated_quadratic(&beta, &cov, 2).expect("wald quadratic");
        assert_eq!(used, 2, "the 1e-12 covariance mode is resolved");
        // The eigensolve returns each eigenvalue within its backward-error band
        // `n·ε·max|λ|`, so the small mode is known only to `band / 1e-12` relative
        // and `(1e-6)² / λ̂` carries that error on top of rounding the two literals,
        // the square and the division. Dropping the mode would give q = 0 instead.
        let small_mode = cov[[1, 1]];
        let band = cov.nrows() as f64 * f64::EPSILON * cov[[0, 0]];
        assert!(
            (q - 1.0).abs()
                <= band / (small_mode - band) + gam_linalg::roundoff::accumulation_growth(5),
            "q={q} band={band:e}"
        );
    }

    /// `edf1` sums `2F_ii − (F·F)_ii` over the block against the FULL influence
    /// matrix, so the cross leverages with coefficients outside the block enter.
    #[test]
    fn alternative_edf_uses_the_full_influence_rows() {
        let f = array![[0.5, 0.2], [0.3, 0.4]];
        let first = alternative_edf(&f, 0, 1).expect("edf1");
        assert!((first - (1.0 - (0.25 + 0.06))).abs() < 1e-15, "{first}");
        let both = alternative_edf(&f, 0, 2).expect("edf1");
        assert!((both - (0.69 + 0.8 - (0.06 + 0.16))).abs() < 1e-15, "{both}");
        assert!(alternative_edf(&f, 0, 3).is_none());
    }

    /// At fractional rank `r = 1.5` the statistic is `u₁² + ½·u₂²` on the
    /// standardized whitened coordinates, and its tail is the `χ²₁ + ½·χ²₁` law.
    #[test]
    fn fractional_rank_statistic_carries_the_share_of_the_next_direction() {
        let beta = array![2.0, 3.0];
        let cov = array![[4.0, 0.0], [0.0, 1.0]];
        let gram = Array2::<f64>::eye(2);
        // `2f − f² = 1 − (1 − f)²` is 1 at f = 1 and ½ at f = 1 − √½.
        let f = array![[1.0, 0.0], [0.0, 1.0 - 0.5_f64.sqrt()]];
        let out = wood_smooth_test(SmoothTestInput {
            beta: beta.view(),
            covariance: &cov,
            influence_matrix: Some(&f),
            whitening_gram: Some(&gram),
            coeff_range: 0..2,
            edf: 1.2,
            nullspace_dim: 0,
            residual_df: None,
            scale: SmoothTestScale::Known,
        })
        .expect("smooth test");
        assert!((out.ref_df - 1.5).abs() < 1e-14, "ref_df={}", out.ref_df);
        assert!((out.statistic - 5.5).abs() < 1e-12, "stat={}", out.statistic);
        let expected = fractional_rank_sf(out.statistic, 1.5, None).probability;
        assert!((out.p_value - expected).abs() < 1e-15);
    }

    /// Below rank one only the leading direction is tested, and the scaled
    /// statistic `ν·u₁²` on `ν·χ²₁` is its ordinary one-degree test.
    #[test]
    fn fractional_rank_below_one_is_the_leading_direction_test() {
        let beta = array![3.0, 1.0];
        let cov = array![[4.0, 0.0], [0.0, 1.0]];
        let out = fractional_rank_test(&beta, &cov, 0.4, None).expect("smooth test");
        assert!((out.statistic - 0.4 * 9.0 / 4.0).abs() < 1e-12);
        assert!((out.p_value - chi_square_sf(9.0 / 4.0, 1.0)).abs() < 1e-12);
        assert!(fractional_rank_test(&beta, &cov, 0.0, None).is_none());
        assert!(fractional_rank_test(&beta, &cov, -0.3, None).is_none());
    }

    fn standard_normal(rng: &mut StdRng) -> f64 {
        let u1 = 1.0 - rng.random::<f64>();
        let u2 = rng.random::<f64>();
        (-2.0 * u1.ln()).sqrt() * (2.0 * std::f64::consts::PI * u2).cos()
    }

    /// Seeded null calibration of the whitened test at a fractional rank. The
    /// block is drawn from its null law `β ~ N(0, φV)` and, on the estimated
    /// scale, the covariance carries an independent `φ̂/φ ~ χ²_ρ/ρ`. The shrunk
    /// influence spectrum puts `edf = 2.4` and `edf1 = 3.34` apart from the
    /// block dimension 5, which is where the former rank-`round(edf)` statistic
    /// judged against `χ²_{max(tr(F)²/tr(F²), rank)}` rejected 0.0075 at 0.05 and
    /// 0.021 at 0.10 on the known scale.
    /// The rejection rate must sit within three Monte-Carlo standard errors of
    /// the nominal level at 0.10, 0.05 and 0.01.
    #[test]
    fn fractional_rank_test_holds_its_size_under_the_null() {
        let variances: [f64; 5] = [4.0, 2.0, 1.0, 0.5, 0.25];
        let leverages: [f64; 5] = [0.9, 0.6, 0.4, 0.3, 0.2];
        let p = variances.len();
        let cov = Array2::from_diag(&Array1::from(variances.to_vec()));
        let f = Array2::from_diag(&Array1::from(leverages.to_vec()));
        let gram = Array2::<f64>::eye(p);
        let edf = leverages.iter().sum::<f64>();
        let residual_df = 20.0;
        let reps = 4000;
        let levels = [0.10, 0.05, 0.01];
        for scale in [SmoothTestScale::Known, SmoothTestScale::Estimated] {
            let mut rng = StdRng::seed_from_u64(0x5eed_2025);
            let mut rejections = [0usize; 3];
            for _ in 0..reps {
                let beta = Array1::from_iter(
                    variances.iter().map(|v| v.sqrt() * standard_normal(&mut rng)),
                );
                let dispersion_ratio = match scale {
                    SmoothTestScale::Known => 1.0,
                    SmoothTestScale::Estimated => {
                        (0..residual_df as usize)
                            .map(|_| standard_normal(&mut rng).powi(2))
                            .sum::<f64>()
                            / residual_df
                    }
                };
                let cov_hat = &cov * dispersion_ratio;
                let out = wood_smooth_test(SmoothTestInput {
                    beta: beta.view(),
                    covariance: &cov_hat,
                    influence_matrix: Some(&f),
                    whitening_gram: Some(&gram),
                    coeff_range: 0..p,
                    edf,
                    nullspace_dim: 0,
                    residual_df: Some(residual_df),
                    scale,
                })
                .expect("smooth test");
                assert!((out.ref_df - 3.34).abs() < 1e-12, "ref_df={}", out.ref_df);
                for (count, level) in rejections.iter_mut().zip(levels) {
                    *count += usize::from(out.p_value <= level);
                }
            }
            for (count, level) in rejections.iter().zip(levels) {
                let size = *count as f64 / reps as f64;
                let mcse = (level * (1.0 - level) / reps as f64).sqrt();
                assert!(
                    (size - level).abs() <= 3.0 * mcse,
                    "{scale:?}: size {size} at level {level} (MCSE {mcse})"
                );
            }
        }
    }
}
