use super::*;

#[inline]
pub(crate) fn should_use_survival_rayon(work_items: u64) -> bool {
    rayon::current_num_threads() > 1
        && gam_runtime::parallel::at_top_level()
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
        let partials: Vec<Array2<f64>> = gam_runtime::parallel::fan_out(|| {
            (0..chunk_count)
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
                .collect()
        });

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

/// `Xᵀ·diag(w)·X` for the SAME design on both sides, computed so the result is
/// BITWISE symmetric (gam#1561).
///
/// [`weighted_crossprod_dense_with_parallelism`] is a general `Lᵀ·diag(w)·R`.
/// Handed the same matrix twice it is still a general GEMM: entry `(i, j)` and
/// entry `(j, i)` are separate accumulations, and a blocked kernel is free to
/// sum them in different orders. IEEE addition is not associative, so the two
/// triangles come back differing in the last bits.
///
/// That is not a tolerance question here. `aft_absolute_newton_direction`
/// declares [`SymmetricAssembly::Mirrored`], whose band is EXACTLY ZERO because
/// a mirrored matrix has no legitimate disagreement at all, and
/// `strict_symmetric_eigh` then refuses the Hessian. At `4057627f4b` that
/// refusal was live in eight tests across four crates, every one at index
/// `(1, 0)` and every one by a single ulp: defects of 3.469e-18, 1.110e-16,
/// 2.220e-16 and 4.441e-16 against a band of 0.
///
/// `fast_xt_diag_x_with_parallelism` is one of the two routines the
/// `Mirrored` documentation names as producing a bitwise-symmetric result, and
/// it earns that: it accumulates the LOWER TRIANGLE ONLY
/// (`CrossprodStructure::SymmetricLower`) and mirrors it once into the upper.
/// One rounded value per off-diagonal pair, which is what the declaration
/// asserts.
///
/// The finiteness and weight validation is the same as the general routine's,
/// so a caller swapping to this one loses no check.
pub(crate) fn weighted_selfcrossprod_dense_mirrored(
    x: &Array2<f64>,
    weights: &Array1<f64>,
    par: faer::Par,
) -> Result<Array2<f64>, String> {
    if x.nrows() != weights.len() {
        return Err(SurvivalLocationScaleError::DimensionMismatch {
            reason: format!(
                "weighted_selfcrossprod_dense row mismatch: x is {}x{}, weights has {}",
                x.nrows(),
                x.ncols(),
                weights.len()
            ),
        }
        .into());
    }
    if x.iter().any(|value| !value.is_finite()) {
        return Err(SurvivalLocationScaleError::InvalidConfiguration {
            reason: "weighted_selfcrossprod_dense inputs contain non-finite design values"
                .to_string(),
        }
        .into());
    }
    require_finite_row_weights(weights, "weighted_selfcrossprod_dense")?;
    let out = gam_linalg::faer_ndarray::fast_xt_diag_x_with_parallelism(x, weights, par);
    if out.iter().any(|value| !value.is_finite()) {
        return Err(SurvivalLocationScaleError::InvalidConfiguration {
            reason: "weighted_selfcrossprod_dense accumulation produced non-finite values"
                .to_string(),
        }
        .into());
    }
    Ok(out)
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
    require_finite_row_weights(weights, "weighted_crossprod_dense")?;
    let work = (nrows as u64)
        .saturating_mul(left.ncols() as u64)
        .saturating_mul(right.ncols() as u64);

    if nrows > 1 && should_use_survival_rayon(work) {
        use rayon::prelude::*;

        let out_dim = (left.ncols(), right.ncols());
        let chunk_count = dense_row_chunk_count(nrows);
        let chunk_rows = nrows.div_ceil(chunk_count);
        let partials: Vec<Option<Array2<f64>>> = gam_runtime::parallel::fan_out(|| {
            (0..chunk_count)
                .into_par_iter()
                .map(|chunk_idx| {
                    let start = chunk_idx * chunk_rows;
                    let end = (start + chunk_rows).min(nrows);
                    let mut local = Array2::<f64>::zeros(out_dim);
                    if start < end
                        && !accumulate_weighted_crossprod_dense_rows(
                            &mut local,
                            left,
                            weights,
                            right,
                            start..end,
                        )
                    {
                        return None;
                    }
                    Some(local)
                })
                .collect()
        });

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
            let wi = weights[i];
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

    weighted_crossprod_dense_stable(left, weights, right)
}

pub(crate) fn scale_dense_rows(
    mat: &Array2<f64>,
    coeffs: &Array1<f64>,
) -> Result<Array2<f64>, SurvivalLocationScaleError> {
    if mat.nrows() != coeffs.len() {
        bail_dim_sls!(
            "row scaling dimension mismatch: matrix has {} rows but coeffs have {} entries",
            mat.nrows(),
            coeffs.len()
        );
    }
    require_finite_row_weights(coeffs, "scale_dense_rows")?;
    let work = mat.nrows().saturating_mul(mat.ncols());
    let mut out = mat.clone();

    if mat.nrows() > 1
        && rayon::current_num_threads() > 1
        && gam_runtime::parallel::at_top_level()
        && work >= DENSE_ROW_SCALE_PARALLEL_ELEM_THRESHOLD
    {
        use rayon::prelude::*;

        let chunk_count = dense_row_chunk_count(mat.nrows());
        let chunk_rows = mat.nrows().div_ceil(chunk_count);
        gam_runtime::parallel::fan_out(|| {
            out.axis_chunks_iter_mut(Axis(0), chunk_rows)
                .into_par_iter()
                .enumerate()
                .for_each(|(chunk_idx, mut rows)| {
                    let start = chunk_idx * chunk_rows;
                    for (local_i, mut row) in rows.rows_mut().into_iter().enumerate() {
                        let coeff = coeffs[start + local_i];
                        row.mapv_inplace(|value| safe_product(value, coeff));
                    }
                })
        });
    } else {
        for i in 0..out.nrows() {
            let coeff = coeffs[i];
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

#[cfg(test)]
mod mirrored_selfcrossprod_tests {
    use super::*;
    use ndarray::{Array1, Array2};

    /// #1561: a same-channel survival-LS Hessian group must come back BITWISE
    /// symmetric, and the general crossprod does not promise that.
    ///
    /// `aft_absolute_newton_direction` hands the assembled Hessian to
    /// `strict_symmetric_eigh` declaring `SymmetricAssembly::Mirrored`, whose
    /// band is EXACTLY ZERO: the declaration says every off-diagonal pair was
    /// rounded once and written to both triangles, so any disagreement at all
    /// is a construction defect rather than rounding. One ulp refuses the fit.
    ///
    /// The necessity of this routine is evidenced by the eight tests that
    /// carried that refusal at `4057627f4b`, across `gam-cli`, `gam-models`
    /// and `gam::regressions`, every one at index `(1, 0)` and every one a
    /// single ulp: 3.469e-18, 1.110e-16, 2.220e-16, 4.441e-16 against a band
    /// of 0. It is NOT re-derived here with a synthetic fixture, because
    /// whether a general GEMM reassociates on any given shape depends on its
    /// blocking and would make this gate's meaning depend on a tile size. The
    /// eight real failures are the better evidence, and this test pins the
    /// CONTRACT instead: bitwise symmetry, and the same value as before.
    #[test]
    fn the_self_crossprod_is_bitwise_symmetric_and_keeps_the_general_value() {
        let n = 4096usize;
        let p = 7usize;
        let mut x = Array2::<f64>::zeros((n, p));
        for row in 0..n {
            for col in 0..p {
                // Deterministic and deliberately not dyadic: entries whose
                // products need the whole mantissa are what make a
                // reassociated sum differ in its last bit.
                let t = (row as f64 + 1.0) * (col as f64 + 1.7);
                x[[row, col]] = (t * 0.739_085_133_215_160_7).sin() * (1.0 + 0.5 * col as f64);
            }
        }
        let weights = Array1::from_shape_fn(n, |row| 0.25 + ((row % 13) as f64) / 17.0);

        let mirrored = weighted_selfcrossprod_dense_mirrored(&x, &weights, faer::Par::Seq)
            .expect("mirrored self-crossprod");
        for i in 0..p {
            for j in 0..p {
                assert_eq!(
                    mirrored[[i, j]].to_bits(),
                    mirrored[[j, i]].to_bits(),
                    "({i}, {j}) is not bitwise symmetric: {:e} versus {:e}",
                    mirrored[[i, j]],
                    mirrored[[j, i]]
                );
            }
        }

        // The routine must not move the VALUE. The bar is the band a
        // reassociated sum of these summands may legitimately leave, read off
        // this fixture's own summands, not a chosen closeness.
        let general = weighted_crossprod_dense_with_parallelism(&x, &weights, &x, faer::Par::Seq)
            .expect("general crossprod");
        for i in 0..p {
            for j in 0..p {
                let absolute_sum: f64 = (0..n)
                    .map(|row| (x[[row, i]] * weights[row] * x[[row, j]]).abs())
                    .sum();
                let band = gam_linalg::roundoff::accumulation_band(n, absolute_sum);
                let gap = (mirrored[[i, j]] - general[[i, j]]).abs();
                assert!(
                    gap <= band,
                    "({i}, {j}) moved by {gap:e}, past this entry's own accumulation band \
                     {band:e}; the mirrored routine changed the value, not just its symmetry"
                );
            }
        }
    }
}
