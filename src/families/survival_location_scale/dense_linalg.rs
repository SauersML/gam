use super::*;

pub(crate) fn safe_fast_xt_diag_x(x: &Array2<f64>, weights: &Array1<f64>) -> Array2<f64> {
    let sanitized = sanitize_survival_weight_vector(weights);
    fast_xt_diag_x(x, &sanitized)
}

pub(crate) fn safe_fast_xt_diag_x_with_parallelism(
    x: &Array2<f64>,
    weights: &Array1<f64>,
    par: faer::Par,
) -> Array2<f64> {
    let sanitized = sanitize_survival_weight_vector(weights);
    fast_xt_diag_x_with_parallelism(x, &sanitized, par)
}

/// Horvitz-Thompson outer-subsample row mask. When `mask` is `None` this
/// returns `weighted_crossprod_dense(left, weights, right)` verbatim, which is
/// the byte-identical pre-refactor expression. When `mask` is `Some(m)`, the
/// per-row weight `weights[i]` is replaced by `weights[i] * m[i]` before the
/// cross product. The math invariant is that each survival-LS assembly site
/// is row-additive — `Σ_i x_i y_iᵀ · w_i` — so per-row HT-masking is unbiased.
#[inline]
pub(crate) fn mxtwx(
    left: &Array2<f64>,
    weights: &Array1<f64>,
    right: &Array2<f64>,
    mask: Option<&Array1<f64>>,
) -> Result<Array2<f64>, String> {
    match mask {
        Some(m) => weighted_crossprod_dense(left, &(weights * m), right),
        None => weighted_crossprod_dense(left, weights, right),
    }
}

#[inline]
pub(crate) fn mxtwxd(x: &Array2<f64>, weights: &Array1<f64>, mask: Option<&Array1<f64>>) -> Array2<f64> {
    match mask {
        Some(m) => safe_fast_xt_diag_x(x, &(weights * m)),
        None => safe_fast_xt_diag_x(x, weights),
    }
}

/// Multiply a per-row weight by the HT mask. The `None` branch returns the
/// caller's array unmodified (zero-copy borrow), so any downstream
/// `X.t().dot(&out)` / `out.sum()` / `out.dot(&other)` aggregate is
/// byte-identical to the pre-refactor path. The `Some` branch produces an
/// owned masked copy.
#[inline]
pub(crate) fn mask_row_vec<'a>(
    weights: &'a Array1<f64>,
    mask: Option<&Array1<f64>>,
) -> std::borrow::Cow<'a, Array1<f64>> {
    match mask {
        Some(m) => std::borrow::Cow::Owned(weights * m),
        None => std::borrow::Cow::Borrowed(weights),
    }
}

/// HT-mask-aware variant of [`weighted_crossprod_psi_maps`]. `None` is
/// byte-identical to the pre-refactor call. `Some(m)` multiplies the
/// per-row weight view by `m` before the cross product.
#[inline]
pub(crate) fn mxtwx_psi(
    left: crate::families::custom_family::CustomFamilyPsiLinearMapRef<'_>,
    weights: ArrayView1<'_, f64>,
    right: crate::families::custom_family::CustomFamilyPsiLinearMapRef<'_>,
    mask: Option<&Array1<f64>>,
) -> Result<Array2<f64>, String> {
    match mask {
        Some(m) => {
            let masked = &weights * m;
            weighted_crossprod_psi_maps(left, masked.view(), right)
        }
        None => weighted_crossprod_psi_maps(left, weights, right),
    }
}

#[inline]
pub(crate) fn should_use_survival_rayon(work_items: u64) -> bool {
    rayon::current_num_threads() > 1
        && rayon::current_thread_index().is_none()
        && work_items >= DENSE_WEIGHTED_CROSSPROD_PARALLEL_FLOP_THRESHOLD
}

#[inline]
pub(crate) fn dense_row_chunk_count(nrows: usize) -> usize {
    let max_chunks = rayon::current_num_threads()
        .saturating_mul(DENSE_ROW_CHUNKS_PER_THREAD)
        .max(1);
    nrows.min(max_chunks).max(1)
}

pub(crate) fn accumulate_weighted_crossprod_dense_stable_rows(
    out: &mut Array2<f64>,
    left: &Array2<f64>,
    weights: &Array1<f64>,
    right: &Array2<f64>,
    rows: std::ops::Range<usize>,
) {
    for i in rows {
        let wi = weights[i];
        if wi == 0.0 {
            continue;
        }
        for j in 0..left.ncols() {
            let lij = left[[i, j]];
            if lij == 0.0 {
                continue;
            }
            for k in 0..right.ncols() {
                let rijk = right[[i, k]];
                if rijk == 0.0 {
                    continue;
                }
                let contrib = safe_product3(wi, lij, rijk);
                out[[j, k]] = safe_sum2(out[[j, k]], contrib);
            }
        }
    }
}

pub(crate) fn accumulate_weighted_crossprod_dense_rows(
    out: &mut Array2<f64>,
    left: &Array2<f64>,
    weights: &Array1<f64>,
    right: &Array2<f64>,
    rows: std::ops::Range<usize>,
) -> bool {
    for i in rows {
        let wi = weights[i];
        if wi == 0.0 {
            continue;
        }
        for j in 0..left.ncols() {
            let lij = left[[i, j]];
            if lij == 0.0 {
                continue;
            }
            let weighted_lij = wi * lij;
            if !weighted_lij.is_finite() {
                return false;
            }
            for k in 0..right.ncols() {
                let rijk = right[[i, k]];
                if rijk == 0.0 {
                    continue;
                }
                let contrib = weighted_lij * rijk;
                let updated = out[[j, k]] + contrib;
                if !contrib.is_finite() || !updated.is_finite() {
                    return false;
                }
                out[[j, k]] = updated;
            }
        }
    }
    true
}

pub(crate) fn weighted_crossprod_dense_stable(
    left: &Array2<f64>,
    weights: &Array1<f64>,
    right: &Array2<f64>,
) -> Result<Array2<f64>, String> {
    if left.nrows() != weights.len() || right.nrows() != weights.len() {
        return Err(SurvivalLocationScaleError::DimensionMismatch { reason: format!(
            "weighted_crossprod_dense stable row mismatch: left is {}x{}, weights has {}, right is {}x{}",
            left.nrows(),
            left.ncols(),
            weights.len(),
            right.nrows(),
            right.ncols()
        ) }.into());
    }

    let nrows = weights.len();
    let out_dim = (left.ncols(), right.ncols());
    let work = (nrows as u64)
        .saturating_mul(left.ncols() as u64)
        .saturating_mul(right.ncols() as u64);

    let out = if nrows > 1 && should_use_survival_rayon(work) {
        use rayon::prelude::*;

        let chunk_count = dense_row_chunk_count(nrows);
        let chunk_rows = nrows.div_ceil(chunk_count);
        let partials: Vec<Array2<f64>> = (0..chunk_count)
            .into_par_iter()
            .map(|chunk_idx| {
                let start = chunk_idx * chunk_rows;
                let end = (start + chunk_rows).min(nrows);
                let mut local = Array2::<f64>::zeros(out_dim);
                if start < end {
                    accumulate_weighted_crossprod_dense_stable_rows(
                        &mut local,
                        left,
                        weights,
                        right,
                        start..end,
                    );
                }
                local
            })
            .collect();

        let mut reduced = Array2::<f64>::zeros(out_dim);
        for local in partials {
            for (dst, src) in reduced.iter_mut().zip(local.iter()) {
                *dst = safe_sum2(*dst, *src);
            }
        }
        reduced
    } else {
        let mut serial = Array2::<f64>::zeros(out_dim);
        accumulate_weighted_crossprod_dense_stable_rows(
            &mut serial,
            left,
            weights,
            right,
            0..nrows,
        );
        serial
    };

    if out.iter().any(|value| !value.is_finite()) {
        return Err(SurvivalLocationScaleError::InvalidConfiguration {
            reason: "weighted_crossprod_dense stable accumulation produced non-finite values"
                .to_string(),
        }
        .into());
    }
    Ok(out)
}

pub(crate) fn weighted_crossprod_dense(
    left: &Array2<f64>,
    weights: &Array1<f64>,
    right: &Array2<f64>,
) -> Result<Array2<f64>, String> {
    weighted_crossprod_dense_with_parallelism(left, weights, right, faer::get_global_parallelism())
}

pub(crate) fn weighted_crossprod_dense_with_parallelism(
    left: &Array2<f64>,
    weights: &Array1<f64>,
    right: &Array2<f64>,
    par: faer::Par,
) -> Result<Array2<f64>, String> {
    if left.nrows() != weights.len() || right.nrows() != weights.len() {
        return Err(SurvivalLocationScaleError::DimensionMismatch { reason: format!(
            "weighted_crossprod_dense row mismatch: left is {}x{}, weights has {}, right is {}x{}",
            left.nrows(),
            left.ncols(),
            weights.len(),
            right.nrows(),
            right.ncols()
        ) }.into());
    }
    if left.iter().any(|value| !value.is_finite()) || right.iter().any(|value| !value.is_finite()) {
        return Err(SurvivalLocationScaleError::InvalidConfiguration {
            reason: "weighted_crossprod_dense inputs contain non-finite design values".to_string(),
        }
        .into());
    }

    let nrows = weights.len();
    let sanitized_weights = sanitize_survival_weight_vector(weights);
    let work = (nrows as u64)
        .saturating_mul(left.ncols() as u64)
        .saturating_mul(right.ncols() as u64);

    if nrows > 1 && should_use_survival_rayon(work) {
        use rayon::prelude::*;

        let out_dim = (left.ncols(), right.ncols());
        let chunk_count = dense_row_chunk_count(nrows);
        let chunk_rows = nrows.div_ceil(chunk_count);
        let partials: Vec<Option<Array2<f64>>> = (0..chunk_count)
            .into_par_iter()
            .map(|chunk_idx| {
                let start = chunk_idx * chunk_rows;
                let end = (start + chunk_rows).min(nrows);
                let mut local = Array2::<f64>::zeros(out_dim);
                if start < end
                    && !accumulate_weighted_crossprod_dense_rows(
                        &mut local,
                        left,
                        &sanitized_weights,
                        right,
                        start..end,
                    )
                {
                    return None;
                }
                Some(local)
            })
            .collect();

        if partials.iter().all(Option::is_some) {
            let mut out = Array2::<f64>::zeros(out_dim);
            let mut fast_path_ok = true;
            'reduce: for local in partials.into_iter().flatten() {
                for (dst, src) in out.iter_mut().zip(local.iter()) {
                    let updated = *dst + *src;
                    if !updated.is_finite() {
                        fast_path_ok = false;
                        break 'reduce;
                    }
                    *dst = updated;
                }
            }
            if fast_path_ok {
                return Ok(out);
            }
        }
    } else {
        let mut weighted_right = right.clone();
        let mut fast_path_ok = true;
        'outer: for i in 0..weighted_right.nrows() {
            let wi = sanitized_weights[i];
            if wi == 0.0 {
                weighted_right.row_mut(i).fill(0.0);
                continue;
            }
            if wi == 1.0 {
                continue;
            }
            for j in 0..weighted_right.ncols() {
                let scaled = wi * weighted_right[[i, j]];
                if !scaled.is_finite() {
                    fast_path_ok = false;
                    break 'outer;
                }
                weighted_right[[i, j]] = scaled;
            }
        }
        if fast_path_ok {
            let out = fast_atb_with_parallelism(left, &weighted_right, par);
            if out.iter().all(|value| value.is_finite()) {
                return Ok(out);
            }
        }
    }

    weighted_crossprod_dense_stable(left, &sanitized_weights, right)
}

pub(crate) fn scale_dense_rows(
    mat: &Array2<f64>,
    coeffs: &Array1<f64>,
) -> Result<Array2<f64>, SurvivalLocationScaleError> {
    if mat.nrows() != coeffs.len() {
        crate::bail_dim_sls!(
            "row scaling dimension mismatch: matrix has {} rows but coeffs have {} entries",
            mat.nrows(),
            coeffs.len()
        );
    }
    let sanitized_coeffs = sanitize_survival_weight_vector(coeffs);
    let work = mat.nrows().saturating_mul(mat.ncols());
    let mut out = mat.clone();

    if mat.nrows() > 1
        && rayon::current_num_threads() > 1
        && rayon::current_thread_index().is_none()
        && work >= DENSE_ROW_SCALE_PARALLEL_ELEM_THRESHOLD
    {
        use rayon::prelude::*;

        let chunk_count = dense_row_chunk_count(mat.nrows());
        let chunk_rows = mat.nrows().div_ceil(chunk_count);
        out.axis_chunks_iter_mut(Axis(0), chunk_rows)
            .into_par_iter()
            .enumerate()
            .for_each(|(chunk_idx, mut rows)| {
                let start = chunk_idx * chunk_rows;
                for (local_i, mut row) in rows.rows_mut().into_iter().enumerate() {
                    let coeff = sanitized_coeffs[start + local_i];
                    row.mapv_inplace(|value| safe_product(value, coeff));
                }
            });
    } else {
        for i in 0..out.nrows() {
            let coeff = sanitized_coeffs[i];
            out.row_mut(i)
                .mapv_inplace(|value| safe_product(value, coeff));
        }
    }

    if out.iter().any(|value| value.is_nan()) {
        return Err(SurvivalLocationScaleError::NumericalFailure {
            reason: "row scaling produced NaN values".to_string(),
        });
    }
    Ok(out)
}

pub(crate) fn embed_tail_columns(
    local: &Array2<f64>,
    total_cols: usize,
    tail_range: std::ops::Range<usize>,
) -> Result<Array2<f64>, String> {
    if tail_range.end > total_cols || tail_range.len() != local.ncols() {
        return Err(SurvivalLocationScaleError::DimensionMismatch {
            reason: format!(
                "tail embedding mismatch: local_cols={}, total_cols={}, tail={:?}",
                local.ncols(),
                total_cols,
                tail_range
            ),
        }
        .into());
    }
    let mut out = Array2::<f64>::zeros((local.nrows(), total_cols));
    out.slice_mut(s![.., tail_range]).assign(local);
    Ok(out)
}

pub(crate) fn assign_block(target: &mut Array2<f64>, row_start: usize, col_start: usize, block: &Array2<f64>) {
    let row_end = row_start + block.nrows();
    let col_end = col_start + block.ncols();
    target
        .slice_mut(s![row_start..row_end, col_start..col_end])
        .assign(block);
}

pub(crate) fn assign_symmetric_block(
    target: &mut Array2<f64>,
    row_start: usize,
    col_start: usize,
    block: &Array2<f64>,
) {
    assign_block(target, row_start, col_start, block);
    if row_start != col_start || block.nrows() != block.ncols() {
        assign_block(target, col_start, row_start, &block.t().to_owned());
    }
}
