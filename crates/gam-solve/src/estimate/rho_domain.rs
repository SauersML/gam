//! The λ-selection domain, derived per coordinate from a penalized term's own
//! design-relative penalty spectrum (#2812).
//!
//! On a penalty's range each direction carries data curvature `γ_j` (a
//! generalized eigenvalue of the block's design Gram against the penalty,
//! quotiented by the penalty's null space, and also by only the null space it
//! shares with the other penalties on its columns when it has companions) and
//! penalty curvature `λ = e^ρ` in
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
use gam_linalg::matrix::DesignMatrix;
use gam_terms::construction::CanonicalPenalty;
use ndarray::{Array1, Array2, ArrayView1, Axis, s};

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

/// Whether a term's unpenalized fit is identified by its own data (#2954): every
/// generalized eigenvalue `γ_j` of its `p`-column Gram against its penalty, on
/// the penalty's range, clears the band `p·ε·‖B‖_F` those eigenvalues carry.
/// `B = D_r^(−1/2)·(A_rr − A_r0·A_00⁺·A_0r)·D_r^(−1/2)` is formed from `p`-term inner
/// products, so its entries hold `p·ε·‖B‖` of rounding, and a symmetric
/// eigensolver's backward error is of the same order; `‖B‖_F = √(Σγ_j²)` bounds
/// both. Then `H_β` restricted to the term stays nonsingular as its `λ → 0`, and
/// the lower resolvability edge is the unpenalized fit's limit model. Otherwise
/// the penalty is what makes `H_β` positive definite there (a basis wider than
/// the data support), and that edge is a representability face.
pub fn unpenalized_fit_is_identified(gammas: &[f64], columns: usize) -> bool {
    // Scaled by the largest `γ_j`, so `Σγ_j²` cannot overflow where every `γ_j`
    // is finite.
    let scale = gammas
        .iter()
        .fold(0.0_f64, |scale, gamma| scale.max(gamma.abs()));
    let frobenius = scale
        * gammas
            .iter()
            .map(|gamma| (gamma / scale).powi(2))
            .sum::<f64>()
            .sqrt();
    let band = columns as f64 * f64::EPSILON * frobenius;
    frobenius.is_finite() && frobenius > 0.0 && gammas.iter().all(|&gamma| gamma > band)
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

/// The sum of every OTHER penalty on exactly `range`, skipping the penalty at
/// index `skip`, or `None` when that penalty sits alone there. A double penalty
/// ships its bending block and its null-space ridge as two coordinates on one
/// column range.
fn shared_columns_companions<'a, S: ndarray::Data<Elem = f64> + 'a>(
    penalties: impl IntoIterator<Item = (&'a std::ops::Range<usize>, &'a ndarray::ArrayBase<S, ndarray::Ix2>)>,
    skip: usize,
    range: &std::ops::Range<usize>,
    dim: (usize, usize),
) -> Option<Array2<f64>> {
    let mut companions: Option<Array2<f64>> = None;
    for (index, (other_range, other)) in penalties.into_iter().enumerate() {
        if index != skip && other_range == range && other.dim() == dim {
            match companions.as_mut() {
                Some(sum) => *sum += other,
                None => companions = Some(other.to_owned()),
            }
        }
    }
    companions
}

/// Orthonormal frame of the null space of a symmetric PSD matrix, classified at
/// the pseudo-determinant's positive-eigenvalue threshold. `None` when the
/// eigendecomposition fails.
pub(crate) fn psd_null_frame(matrix: &Array2<f64>) -> Option<Array2<f64>> {
    let (evals, evecs) = matrix.eigh(Side::Lower).ok()?;
    let threshold = positive_eigenvalue_threshold(evals.as_slice()?);
    let null_cols: Vec<usize> = evals
        .iter()
        .enumerate()
        .filter_map(|(j, &value)| (value <= threshold).then_some(j))
        .collect();
    let mut frame = Array2::<f64>::zeros((matrix.nrows(), null_cols.len()));
    for (col, &src) in null_cols.iter().enumerate() {
        frame.column_mut(col).assign(&evecs.column(src));
    }
    Some(frame)
}

/// A penalty range's weighted Gram. Columns that never share a row (a factor's
/// indicator columns, or one random slope per level) have a diagonal Gram,
/// held as that diagonal: the range's `q × q` matrix is never formed.
enum RangeGram {
    Dense(Array2<f64>),
    Diagonal(Array1<f64>),
}

impl RangeGram {
    fn dim(&self) -> usize {
        match self {
            Self::Dense(gram) => gram.nrows(),
            Self::Diagonal(diagonal) => diagonal.len(),
        }
    }

    fn to_dense(&self) -> Array2<f64> {
        match self {
            Self::Dense(gram) => gram.clone(),
            Self::Diagonal(diagonal) => Array2::from_diag(diagonal),
        }
    }
}

/// The diagonal of `matrix` when every off-diagonal entry is exactly zero.
fn exact_diagonal<S: ndarray::Data<Elem = f64>>(
    matrix: &ndarray::ArrayBase<S, ndarray::Ix2>,
) -> Option<Array1<f64>> {
    gam_terms::construction::is_diagonal(matrix.view()).then(|| matrix.diag().to_owned())
}

/// [`penalty_range_gammas_with_shared_nullspace`] for a diagonal Gram, penalty
/// and aggregate penalty, in closed form. The eigenvalues of a diagonal matrix
/// are its entries and its eigenvectors coordinate vectors (up to a rotation
/// inside a repeated eigenvalue, which leaves every spectrum below unchanged),
/// so the general reduction decouples per coordinate: `B = diag(g_j / s_j)` on
/// the penalty's range, and profiling out an aggregate-null coordinate `m`
/// with `g_m` above the Gram's threshold removes `g_m² / (s_m g_m)` from `B_mm`
/// alone, leaving exactly zero. The classification thresholds are the general path's, taken over the
/// same eigenvalues.
fn diagonal_penalty_range_gammas(
    gram: &Array1<f64>,
    s: &Array1<f64>,
    aggregate: &Array1<f64>,
) -> Option<Vec<f64>> {
    let p = s.len();
    if p == 0 || gram.len() != p || aggregate.len() != p {
        return None;
    }
    let s_max = s.iter().fold(0.0_f64, |a, &b| a.max(b.abs()));
    if !(s_max > 0.0) {
        return None;
    }
    let s_thresh = positive_eigenvalue_threshold(s.as_slice()?);
    let aggregate_threshold = positive_eigenvalue_threshold(aggregate.as_slice()?);
    let null_gram: Vec<f64> = (0..p)
        .filter(|&j| aggregate[j] <= aggregate_threshold)
        .map(|j| gram[j])
        .collect();
    let null_gram_threshold = positive_eigenvalue_threshold(&null_gram);
    let gammas: Vec<f64> = (0..p)
        .filter(|&j| s[j] > s_thresh)
        .map(|j| {
            let profiled = aggregate[j] <= aggregate_threshold && gram[j] > null_gram_threshold;
            let gamma = if profiled { 0.0 } else { gram[j] / s[j] };
            if gamma.is_finite() && gamma > 0.0 { gamma } else { 0.0 }
        })
        .collect();
    (!gammas.is_empty()).then_some(gammas)
}

/// One penalty's resolvability among the penalties on its columns, with the
/// penalty's own spectrum against the range Gram
/// (`penalty_range_gammas_from_gram`) that the interval was read from.
struct SharedColumnsResolvability {
    interval: Option<(f64, f64)>,
    own_gammas: Option<Vec<f64>>,
}

/// The resolvability interval of one penalty among the penalties on its columns.
///
/// Alone on its columns, a penalty's null space is quotiented out as
/// unpenalized. Beside companions that null space is penalized by them: a double
/// penalty's ridge has the whole bending range as its kernel. Profiling that
/// kernel out as free reads only the curvature left once the companion's
/// directions have absorbed the data, which is small whenever they can
/// represent the null function closely, and then the upper edge sits below the
/// strengths that switch the term off. So the curvature is also read quotiented
/// only by the null space every penalty on the columns shares
/// ([`penalty_range_gammas_with_shared_nullspace`]).
///
/// Neither read sees the strengths at which the companions dominate. There the
/// directions they penalize are pinned, the free directions are `N = null(C)`
/// for the companions' sum `C`, and the coordinate switches the term off along
/// `N` at the curvature of `NᵀGN` against `NᵀSN`. When `range(S) = null(C)`
/// (the complementary ridge `N M Nᵀ`, #2372) that is the shared read. A ridge
/// charged along another direction (the mean-slope ridge `m vvᵀ`, #1561)
/// charges the null function `n̂` only `(vᵀn̂)²` of its strength, so its
/// switch-off curvature is `n̂ᵀGn̂ / (vᵀn̂)²`, far above `vᵀGv` along its own
/// range. The coordinate's domain spans all three intervals: it stays free
/// wherever the term is resolvable at some strength of its companions. A lone
/// penalty keeps its own interval.
///
/// A diagonal Gram beside diagonal penalties is read in closed form
/// ([`diagonal_penalty_range_gammas`]); any other pair takes the general
/// reduction on the dense Gram.
fn shared_columns_resolvability<S: ndarray::Data<Elem = f64>>(
    gram: &RangeGram,
    local: &ndarray::ArrayBase<S, ndarray::Ix2>,
    companions: Option<&Array2<f64>>,
) -> SharedColumnsResolvability {
    let combine = |own: Option<&Vec<f64>>, others: [Option<(f64, f64)>; 2]| {
        [own.and_then(|gammas| resolvability_interval(gammas))]
            .into_iter()
            .chain(others)
            .flatten()
            .reduce(|left, right| (left.0.min(right.0), left.1.max(right.1)))
    };
    if let RangeGram::Diagonal(g) = gram
        && let Some(s) = exact_diagonal(local)
        && let Some(companions) = match companions {
            None => Some(None),
            Some(matrix) => exact_diagonal(matrix).map(Some),
        }
    {
        let own_gammas = diagonal_penalty_range_gammas(g, &s, &s);
        let (shared, companions_off) = match companions {
            None => (None, None),
            Some(c) => {
                let aggregate = &c + &s;
                let shared = diagonal_penalty_range_gammas(g, &s, &aggregate)
                    .as_deref()
                    .and_then(resolvability_interval);
                let free_threshold = c.as_slice().map(positive_eigenvalue_threshold);
                let free: Vec<usize> = free_threshold
                    .map(|t| (0..c.len()).filter(|&j| c[j] <= t).collect())
                    .unwrap_or_default();
                let companions_off = (!free.is_empty())
                    .then(|| {
                        let free_gram = free.iter().map(|&j| g[j]).collect::<Array1<f64>>();
                        let free_penalty = free.iter().map(|&j| s[j]).collect::<Array1<f64>>();
                        diagonal_penalty_range_gammas(&free_gram, &free_penalty, &free_penalty)
                    })
                    .flatten()
                    .as_deref()
                    .and_then(resolvability_interval);
                (shared, companions_off)
            }
        };
        let interval = combine(own_gammas.as_ref(), [shared, companions_off]);
        return SharedColumnsResolvability {
            interval,
            own_gammas,
        };
    }
    let gram = gram.to_dense();
    let local = &local.to_owned();
    let own_gammas = penalty_range_gammas_from_gram(&gram, local);
    let Some(companions) = companions else {
        return SharedColumnsResolvability {
            interval: combine(own_gammas.as_ref(), [None, None]),
            own_gammas,
        };
    };
    let aggregate = companions + local;
    let shared = penalty_range_gammas_with_shared_nullspace(&gram, local, &aggregate)
        .as_deref()
        .and_then(resolvability_interval);
    let companions_off = psd_null_frame(companions)
        .filter(|frame| frame.ncols() > 0)
        .and_then(|frame| {
            let free_gram = frame.t().dot(&gram).dot(&frame);
            let free_penalty = frame.t().dot(local).dot(&frame);
            penalty_range_gammas_from_gram(&free_gram, &free_penalty)
        })
        .as_deref()
        .and_then(resolvability_interval);
    SharedColumnsResolvability {
        interval: combine(own_gammas.as_ref(), [shared, companions_off]),
        own_gammas,
    }
}

/// The per-coordinate domain of a penalized design given as one Gram over
/// ALL columns and one penalty block per ρ coordinate, each a local matrix on
/// a contiguous column range. Blocks on the same range are read together
/// (`shared_columns_resolvability`). A coordinate whose block cannot
/// be projected keeps the precision box.
pub fn resolvability_domain_from_gram_blocks<'a>(
    gram: &Array2<f64>,
    blocks: impl IntoIterator<Item = (std::ops::Range<usize>, &'a Array2<f64>)>,
    rho_dim: usize,
) -> (Array1<f64>, Array1<f64>) {
    let (box_lo, box_hi) = coordinate_domain(None, None);
    let mut lower = Array1::<f64>::from_elem(rho_dim, box_lo);
    let mut upper = Array1::<f64>::from_elem(rho_dim, box_hi);
    let blocks: Vec<(std::ops::Range<usize>, &'a Array2<f64>)> =
        blocks.into_iter().take(rho_dim).collect();
    for (k, (range, local)) in blocks.iter().enumerate() {
        if range.end > gram.nrows() || range.start >= range.end {
            continue;
        }
        let block_gram = gram
            .slice(ndarray::s![range.start..range.end, range.start..range.end])
            .to_owned();
        let companions = shared_columns_companions(
            blocks.iter().map(|(other_range, other)| (other_range, *other)),
            k,
            range,
            local.dim(),
        );
        let interval =
            shared_columns_resolvability(&RangeGram::Dense(block_gram), local, companions.as_ref())
                .interval;
        if let Some(interval) = interval {
            let (lo, hi) = coordinate_domain(Some(interval), None);
            lower[k] = lo;
            upper[k] = hi;
        }
    }
    (lower, upper)
}

/// The per-coordinate domain of a penalized design given one canonical penalty
/// per ρ coordinate, from the weighted Gram of each penalty's own columns. The
/// Grams are accumulated by streaming the design in the library's byte-balanced
/// row chunks (`byte_balanced_row_chunk`), so no `p × p` matrix is formed and
/// sparse or lazy designs are read through the same stream. A coordinate whose
/// block cannot be projected keeps the precision box.
pub(crate) fn resolvability_domain_from_design(
    weights: ArrayView1<'_, f64>,
    design: &DesignMatrix,
    penalties: &[CanonicalPenalty],
) -> Result<(Array1<f64>, Array1<f64>), String> {
    resolvability_domain_and_limit_faces_from_design(weights, design, penalties)
        .map(|domain| (domain.lower, domain.upper))
}

/// The #2812 resolvability domain with the kind of each of its faces (#2954).
///
/// A coordinate's edge is its limit model when it comes from the term's own
/// resolvability interval: past `ln(γ_max / √ε)` every direction's effective
/// degrees of freedom `γ_j / (γ_j + λ)` are under the gradient's resolution, so
/// the term is at its null-space fit, and below `ln(√ε γ_min)` it is at its
/// unpenalized fit where its data identify that fit
/// ([`unpenalized_fit_is_identified`]). An edge that is the precision box a term without penalty
/// geometry falls back to, or that the representable log-strength range cut, is
/// a literal and is not.
pub(crate) struct ResolvabilityDomain {
    pub(crate) lower: Array1<f64>,
    pub(crate) upper: Array1<f64>,
    pub(crate) lower_is_limit: Vec<bool>,
    pub(crate) upper_is_limit: Vec<bool>,
}

/// The weighted Gram `X_rᵀ W X_r` of each range's columns `r`. A sparse design
/// is read column by column: a range whose columns share no row gets only its
/// diagonal `Σ w x²`, any other range the exact products of its column pairs.
/// A dense or lazy design is streamed in the library's byte-balanced row chunks
/// (`byte_balanced_row_chunk`), so no `p × p` matrix is formed, and a range
/// Gram that comes out exactly diagonal is kept as its diagonal.
fn range_grams(
    weights: ArrayView1<'_, f64>,
    design: &DesignMatrix,
    ranges: &[std::ops::Range<usize>],
) -> Result<Vec<RangeGram>, String> {
    let n = design.nrows();
    let p = design.ncols();
    if let Some(sparse) = design.as_sparse() {
        let symbolic = sparse.symbolic();
        let col_ptr = symbolic.col_ptr();
        let row_idx = symbolic.row_idx();
        let values = sparse.val();
        let entries = |col: usize| (col_ptr[col]..col_ptr[col + 1]).filter(|&e| values[e] != 0.0);
        // `owner[row]` is the index of the last range that saw a nonzero in
        // `row`, so a second nonzero in one range's row is caught in one pass.
        let mut owner = vec![usize::MAX; n];
        let mut scattered = vec![0.0_f64; n];
        let mut grams = Vec::with_capacity(ranges.len());
        for (index, range) in ranges.iter().enumerate() {
            let disjoint = range.clone().all(|col| {
                entries(col).all(|e| {
                    let row = row_idx[e];
                    let fresh = owner[row] != index;
                    owner[row] = index;
                    fresh
                })
            });
            if disjoint {
                let diagonal = range
                    .clone()
                    .map(|col| {
                        entries(col)
                            .map(|e| values[e] * (values[e] * weights[row_idx[e]]))
                            .sum::<f64>()
                    })
                    .collect::<Array1<f64>>();
                grams.push(RangeGram::Diagonal(diagonal));
                continue;
            }
            let mut gram = Array2::<f64>::zeros((range.len(), range.len()));
            for a in range.clone() {
                for e in entries(a) {
                    scattered[row_idx[e]] = values[e] * weights[row_idx[e]];
                }
                for b in a..range.end {
                    let value: f64 = entries(b)
                        .map(|e| values[e] * scattered[row_idx[e]])
                        .sum();
                    gram[[a - range.start, b - range.start]] = value;
                    gram[[b - range.start, a - range.start]] = value;
                }
                for e in entries(a) {
                    scattered[row_idx[e]] = 0.0;
                }
            }
            grams.push(RangeGram::Dense(gram));
        }
        return Ok(grams);
    }
    let mut grams: Vec<Array2<f64>> = ranges
        .iter()
        .map(|range| Array2::<f64>::zeros((range.len(), range.len())))
        .collect();
    let chunk_rows = gam_runtime::resource::byte_balanced_row_chunk(p, n);
    let mut chunk = Array2::<f64>::zeros((chunk_rows, p));
    for start in (0..n).step_by(chunk_rows) {
        let end = (start + chunk_rows).min(n);
        let rows = end - start;
        design
            .row_chunk_into(start..end, chunk.slice_mut(s![0..rows, ..]))
            .map_err(|err| format!("ρ-domain Gram failed to stream design rows: {err}"))?;
        let row_weights = weights.slice(s![start..end]).insert_axis(Axis(1));
        for (range, gram) in ranges.iter().zip(grams.iter_mut()) {
            let block = chunk.slice(s![0..rows, range.clone()]);
            let weighted = &block * &row_weights;
            *gram += &block.t().dot(&weighted);
        }
    }
    Ok(grams
        .into_iter()
        .map(|gram| match exact_diagonal(&gram) {
            Some(diagonal) => RangeGram::Diagonal(diagonal),
            None => RangeGram::Dense(gram),
        })
        .collect())
}

pub(crate) fn resolvability_domain_and_limit_faces_from_design(
    weights: ArrayView1<'_, f64>,
    design: &DesignMatrix,
    penalties: &[CanonicalPenalty],
) -> Result<ResolvabilityDomain, String> {
    let n = design.nrows();
    let p = design.ncols();
    if weights.len() != n {
        return Err(format!(
            "ρ-domain Gram: {} weights for a design with {n} rows",
            weights.len()
        ));
    }
    let mut ranges: Vec<std::ops::Range<usize>> = Vec::new();
    for penalty in penalties {
        let range = &penalty.col_range;
        if range.start < range.end && range.end <= p && !ranges.contains(range) {
            ranges.push(range.clone());
        }
    }
    let grams = range_grams(weights, design, &ranges)?;
    let (box_lo, box_hi) = coordinate_domain(None, None);
    let mut lower = Array1::<f64>::from_elem(penalties.len(), box_lo);
    let mut upper = Array1::<f64>::from_elem(penalties.len(), box_hi);
    let mut lower_is_limit = vec![false; penalties.len()];
    let mut upper_is_limit = vec![false; penalties.len()];
    for (k, penalty) in penalties.iter().enumerate() {
        let Some(index) = ranges.iter().position(|range| *range == penalty.col_range) else {
            continue;
        };
        let companions = shared_columns_companions(
            penalties
                .iter()
                .map(|other| (&other.col_range, &other.local)),
            k,
            &penalty.col_range,
            penalty.local.dim(),
        );
        let resolvability =
            shared_columns_resolvability(&grams[index], &penalty.local, companions.as_ref());
        if let Some(interval) = resolvability.interval {
            let (lo, hi) = coordinate_domain(Some(interval), None);
            lower[k] = lo;
            upper[k] = hi;
            // λ → ∞ is the null-space fit whatever the data. λ → 0 is a limit
            // only where the unpenalized fit exists: the term's own data must
            // identify its penalized range.
            let columns = grams[index].dim();
            let identified = resolvability
                .own_gammas
                .as_deref()
                .is_some_and(|gammas| unpenalized_fit_is_identified(gammas, columns));
            lower_is_limit[k] = lo == interval.0 && identified;
            upper_is_limit[k] = hi == interval.1;
        }
    }
    Ok(ResolvabilityDomain {
        lower,
        upper,
        lower_is_limit,
        upper_is_limit,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::{Array2, array};

    /// A rank-one ridge charged along `v` rather than along `null(S)` (#2668 row
    /// 23). With the bending companion dominant only `n = e₃` is free, and the
    /// ridge charges it `(vᵀn)²` of its strength, so the strength that switches
    /// the term off is `nᵀGn / (vᵀn)²`. The ridge coordinate's upper edge must
    /// reach it; the shared read along `range(R)` alone must not, or this
    /// fixture would not exercise the companions-off read.
    #[test]
    fn oblique_null_ridge_upper_edge_reaches_the_companions_off_curvature() {
        let bend: Array2<f64> = array![[1.0, 0.0, 0.0], [0.0, 2.0, 0.0], [0.0, 0.0, 0.0]];
        let v = array![6.0, 0.0, 1.0] / 37.0_f64.sqrt();
        let ridge = Array2::from_shape_fn((3, 3), |(i, j)| v[i] * v[j]);
        let gram: Array2<f64> = array![[4.0, 0.5, 0.2], [0.5, 3.0, 0.1], [0.2, 0.1, 5.0]];
        let switch_off = (gram[[2, 2]] / (v[2] * v[2])).ln() - log_gradient_resolution();

        let (lower, upper) =
            resolvability_domain_from_gram_blocks(&gram, [(0..3, &bend), (0..3, &ridge)], 2);
        assert!(
            upper[1] >= switch_off * (1.0 - 64.0 * f64::EPSILON),
            "ridge upper edge {} must reach the companions-off switch-off strength {switch_off}",
            upper[1]
        );
        assert!(lower[1] < upper[1]);

        let aggregate = &bend + &ridge;
        let shared_only = penalty_range_gammas_with_shared_nullspace(&gram, &ridge, &aggregate)
            .as_deref()
            .and_then(resolvability_interval)
            .expect("the shared read resolves the ridge");
        assert!(
            shared_only.1 < switch_off,
            "positive control: the shared read's edge {} must fall short of {switch_off}",
            shared_only.1
        );
    }

    /// A complementary ridge, `range(R) = null(S)`: the companions-off read is
    /// the shared read `nᵀGn`, so the edge is what a1fadc5d1 derived.
    #[test]
    fn complementary_null_ridge_keeps_the_shared_upper_edge() {
        let bend: Array2<f64> = array![[1.0, 0.0, 0.0], [0.0, 2.0, 0.0], [0.0, 0.0, 0.0]];
        let ridge: Array2<f64> = array![[0.0, 0.0, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, 1.0]];
        let gram: Array2<f64> = array![[4.0, 0.5, 0.2], [0.5, 3.0, 0.1], [0.2, 0.1, 5.0]];
        let expected = gram[[2, 2]].ln() - log_gradient_resolution();

        let (lower, upper) =
            resolvability_domain_from_gram_blocks(&gram, [(0..3, &bend), (0..3, &ridge)], 2);
        assert!(lower[1] < upper[1]);
        assert!(
            (upper[1] - expected).abs() <= 64.0 * f64::EPSILON * expected.abs(),
            "complementary ridge upper edge {} must stay the shared edge {expected}",
            upper[1]
        );
    }

    /// A many-level factor's range Gram is read column by column from the sparse
    /// design as its diagonal, and a diagonal penalty against it in closed form.
    /// Neither may move the domain: the sparse Grams must equal the streamed
    /// dense ones, and every closed-form read (own, shared with a companion, and
    /// with the companion dominant) must equal the general reduction on the
    /// dense Gram. The fixture has an empty level (zero Gram entry), a level the
    /// factor penalty leaves unpenalized, and a companion that leaves levels free
    /// (the companions-off read).
    #[test]
    fn diagonal_factor_domain_matches_the_dense_reduction() {
        use faer::sparse::{SparseColMat, Triplet};

        let levels = 9;
        let smooth = 2;
        let rows = 60;
        let columns = 1 + smooth + levels;
        let mut dense = Array2::<f64>::zeros((rows, columns));
        for row in 0..rows {
            let t = (row as f64 + 0.5) / rows as f64;
            dense[[row, 0]] = 1.0;
            dense[[row, 1]] = t;
            dense[[row, 2]] = t * t - 0.3;
            // Level 4 gets no rows; the others get unequal counts and values.
            let level = [0, 1, 2, 3, 5, 6, 7, 8][row % 8];
            dense[[row, 1 + smooth + level]] = 1.0 + 0.1 * (row % 3) as f64;
        }
        let weights = Array1::from_shape_fn(rows, |row| 0.5 + ((row * 7) % 5) as f64 / 4.0);
        let triplets: Vec<Triplet<usize, usize, f64>> = dense
            .indexed_iter()
            .filter(|&(_, &value)| value != 0.0)
            .map(|((row, col), &value)| Triplet::new(row, col, value))
            .collect();
        let sparse = DesignMatrix::from(
            SparseColMat::try_new_from_triplets(rows, columns, &triplets).expect("valid design"),
        );
        let ranges = vec![1..1 + smooth, 1 + smooth..columns];
        let sparse_grams = range_grams(weights.view(), &sparse, &ranges).expect("sparse Grams");
        let dense_grams =
            range_grams(weights.view(), &DesignMatrix::from(dense.clone()), &ranges).expect("Grams");
        assert!(matches!(sparse_grams[0], RangeGram::Dense(_)));
        assert!(matches!(sparse_grams[1], RangeGram::Diagonal(_)));
        assert!(matches!(dense_grams[1], RangeGram::Diagonal(_)));
        for (sparse_gram, dense_gram) in sparse_grams.iter().zip(&dense_grams) {
            let (a, b) = (sparse_gram.to_dense(), dense_gram.to_dense());
            let scale = b.iter().fold(0.0_f64, |m, v| m.max(v.abs()));
            let gap = (&a - &b).iter().fold(0.0_f64, |m, v| m.max(v.abs()));
            assert!(gap <= 64.0 * f64::EPSILON * scale, "Gram gap {gap} at scale {scale}");
        }

        // Level 2 is left unpenalized by the factor penalty; the companion
        // penalizes only levels 0..5, leaving 5..9 free when it dominates.
        let factor = Array2::from_diag(&Array1::from_shape_fn(levels, |j| {
            if j == 2 { 0.0 } else { 1.0 + 0.25 * j as f64 }
        }));
        let companion = Array2::from_diag(&Array1::from_shape_fn(levels, |j| {
            if j < 5 { 2.0 } else { 0.0 }
        }));
        let gram = &sparse_grams[1];
        let dense_gram = RangeGram::Dense(gram.to_dense());
        let sorted = |gammas: Option<Vec<f64>>| {
            let mut gammas = gammas.expect("the factor penalty has a range");
            gammas.sort_by(f64::total_cmp);
            gammas
        };
        for (local, companions) in [
            (&factor, None),
            (&factor, Some(&companion)),
            (&companion, Some(&factor)),
        ] {
            let closed = shared_columns_resolvability(gram, local, companions);
            let general = shared_columns_resolvability(&dense_gram, local, companions);
            let (closed_own, general_own) = (sorted(closed.own_gammas), sorted(general.own_gammas));
            assert_eq!(closed_own.len(), general_own.len());
            for (c, g) in closed_own.iter().zip(&general_own) {
                assert!((c - g).abs() <= 64.0 * f64::EPSILON * g.abs(), "γ {c} vs {g}");
            }
            let (closed, general) = (
                closed.interval.expect("closed-form interval"),
                general.interval.expect("general interval"),
            );
            for (c, g) in [(closed.0, general.0), (closed.1, general.1)] {
                assert!(
                    (c - g).abs() <= 64.0 * f64::EPSILON * g.abs().max(1.0),
                    "interval edge {c} vs {g}"
                );
            }
        }
    }

    /// #2954: λ → 0 is the unpenalized fit's limit only where the term's own data
    /// identify its penalized range. A Gram with no curvature along one penalized
    /// direction (a basis wider than the data support) leaves `γ = 0` there, so the
    /// unpenalized fit is not identified and the lower edge is a representability
    /// face; the full-rank Gram identifies it.
    #[test]
    fn a_rank_deficient_block_has_no_unpenalized_limit_2954() {
        let penalty = array![[0.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]];
        let full = array![[2.0, 0.3, 0.1], [0.3, 1.5, 0.2], [0.1, 0.2, 1.0]];
        let deficient = array![[2.0, 0.3, 0.0], [0.3, 1.5, 0.0], [0.0, 0.0, 0.0]];
        let identified = penalty_range_gammas_from_gram(&full, &penalty).expect("spectrum");
        assert!(
            unpenalized_fit_is_identified(&identified, 3),
            "full rank: {identified:?}"
        );
        let unidentified = penalty_range_gammas_from_gram(&deficient, &penalty).expect("spectrum");
        assert!(
            !unpenalized_fit_is_identified(&unidentified, 3),
            "rank deficient: {unidentified:?}"
        );
    }
}
