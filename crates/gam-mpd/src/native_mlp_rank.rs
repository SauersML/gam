//! Necessary rank floors on the measured Local family, independent of input law.
//! These are proposal diagnostics, not acceptance or neural-forward certificates.
use gam_linalg::decompose::svd;
use ndarray::Array2;
use serde::Serialize;

#[derive(Clone, Debug, Serialize)]
pub struct ResidualFloor {
    /// Computed residual proxy (singular tail or writer projection), with the
    /// comparison error below; not the maximum-row disagreement.
    pub frobenius_residual: f64,
    /// Centering, decomposition and comparison arithmetic uncertainty.
    pub comparison_error: f64,
    /// Necessary maximum-row / RMS-native-row error, using the upper native norm.
    pub necessary_local_lower: Option<f64>,
}
#[derive(Clone, Debug, Serialize)]
pub struct RankFloor {
    pub rank: usize,
    pub rows: usize,
    pub output_width: usize,
    pub native_frobenius_norm: f64,
    pub native_norm_error: f64,
    pub centering_error: f64,
    pub singular_value_band: f64,
    pub svd_validation: SvdValidation,
    pub writer_svd_validation: Option<SvdValidation>,
    pub affine_rank: ResidualFloor,
    /// Unrestricted affine offset, fixed writer span; valid even when the actual
    /// rule's intercept is restricted to that span (then the bound is weaker).
    pub fixed_writer_span: Option<ResidualFloor>,
    pub writer_orthogonality_defect_upper: Option<f64>,
}
/// Necessary correction ranks conditional on one fixed base prediction. The
/// base can have full output rank; only its additive correction is restricted.
#[derive(Clone, Debug, Serialize)]
pub struct FixedBaseRankCurve {
    pub rows: usize,
    pub output_width: usize,
    pub native_frobenius_norm: f64,
    pub native_norm_error: f64,
    pub residual_frobenius_norm: f64,
    pub subtraction_error: f64,
    pub centering_error: f64,
    pub singular_values: Vec<f64>,
    pub svd_validation: SvdValidation,
    /// Same order as the requested ranks. One SVD serves the whole curve.
    pub corrections: Vec<CorrectionRankFloor>,
}
#[derive(Clone, Debug, Serialize)]
pub struct CorrectionRankFloor {
    pub rank: usize,
    pub affine_correction: ResidualFloor,
}
/// Jointly optimizes an unrestricted affine base and rank-limited correction
/// on the supplied rows. This lower bound relaxes the correction's input law.
#[derive(Clone, Debug, Serialize)]
pub struct AffineCorrectionRankCurve {
    pub rows: usize,
    pub input_width: usize,
    pub output_width: usize,
    pub native_frobenius_norm: f64,
    pub native_norm_error: f64,
    pub design_smallest_singular_lower: f64,
    pub design_svd_validation: SvdValidation,
    pub projection_operator_error_upper: f64,
    pub residual_arithmetic_error_upper: f64,
    pub residual_comparison_error_upper: f64,
    pub residual_svd_validation: SvdValidation,
    pub singular_values: Vec<f64>,
    pub corrections: Vec<CorrectionRankFloor>,
}
/// A posteriori validation, independent of the decomposition's resolution cutoff.
#[derive(Clone, Debug, Serialize)]
pub struct SvdValidation {
    pub reconstruction_frobenius_error_upper: f64,
    pub left_orthogonality_defect_upper: f64,
    pub right_orthogonality_defect_upper: f64,
    pub singular_value_enclosure: f64,
}
fn mul_up(a: f64, b: f64) -> f64 {
    (a * b).next_up()
}
fn add_up(a: f64, b: f64) -> f64 {
    (a + b).next_up()
}
fn dot_error(depth: usize, a: f64, b: f64, entries: usize) -> Result<f64, String> {
    Ok(add_up(
        mul_up(gamma(depth + 2)?, mul_up(a, b)),
        mul_up(
            (depth + 2) as f64,
            mul_up(entries as f64, f64::MIN_POSITIVE),
        ),
    ))
}
fn subtract_error(a: f64, b: f64, entries: usize) -> f64 {
    add_up(
        mul_up(f64::EPSILON, add_up(a, b)),
        mul_up(entries as f64, f64::MIN_POSITIVE),
    )
}
fn validate_svd(a: &Array2<f64>, s: &gam_linalg::decompose::Svd) -> Result<SvdValidation, String> {
    let r = a.nrows().min(a.ncols());
    if s.u.dim() != (a.nrows(), r) || s.vt.dim() != (r, a.ncols()) || s.singular_values.len() != r {
        return Err("thin SVD factor dimensions disagree".into());
    }
    if s.singular_values.iter().any(|v| !v.is_finite() || *v < 0.0)
        || s.singular_values
            .iter()
            .zip(s.singular_values.iter().skip(1))
            .any(|(a, b)| a < b)
    {
        return Err("invalid singular values".into());
    }
    let u_norm = upper(norm(s.u.iter().copied())?);
    let v_norm = upper(norm(s.vt.iter().copied())?);
    let identity = Array2::<f64>::eye(r);
    let identity_norm = upper(norm(identity.iter().copied())?);
    let ug = s.u.t().dot(&s.u);
    let vg = s.vt.dot(&s.vt.t());
    let ug_norm = upper(norm(ug.iter().copied())?);
    let vg_norm = upper(norm(vg.iter().copied())?);
    let left = add_up(
        upper(norm((&ug - &identity).iter().copied())?),
        add_up(
            dot_error(a.nrows(), u_norm, u_norm, r * r)?,
            subtract_error(ug_norm, identity_norm, r * r),
        ),
    );
    let right = add_up(
        upper(norm((&vg - &identity).iter().copied())?),
        add_up(
            dot_error(a.ncols(), v_norm, v_norm, r * r)?,
            subtract_error(vg_norm, identity_norm, r * r),
        ),
    );
    if !left.is_finite() || !right.is_finite() || left >= 1.0 || right >= 1.0 {
        return Err("SVD factor orthogonality cannot be validated".into());
    }
    let largest = s.singular_values.iter().copied().fold(0.0_f64, f64::max);
    let mut scaled = s.u.clone();
    for (index, mut col) in scaled.columns_mut().into_iter().enumerate() {
        col.mapv_inplace(|v| v * s.singular_values[index]);
    }
    let scaled_norm = upper(norm(scaled.iter().copied())?);
    let scaled_error = add_up(
        mul_up(f64::EPSILON, mul_up(u_norm, largest)),
        mul_up(scaled.len() as f64, f64::MIN_POSITIVE),
    );
    let reconstruction = scaled.dot(&s.vt);
    let reconstructed_norm = upper(norm(reconstruction.iter().copied())?);
    let a_norm = upper(norm(a.iter().copied())?);
    let reconstruction_error = add_up(
        upper(norm((a - &reconstruction).iter().copied())?),
        add_up(
            mul_up(scaled_error, v_norm),
            add_up(
                dot_error(r, scaled_norm, v_norm, a.len())?,
                subtract_error(a_norm, reconstructed_norm, a.len()),
            ),
        ),
    );
    // Polar factors Uhat,Vhat exist: ||U-Uhat||2 <= left and
    // ||V-Vhat||2 <= right when their Gram defects are below one.
    // ||V||2 <= sqrt(1+right), with outward rounding. Therefore a
    // nearby exact orthonormal Uhat*diag(sigma)*Vhat differs from A by
    // reconstruction_error + sigma_max*(left*sqrt(1+right)+right).
    // Weyl then encloses every singular value. No use of Svd.band occurs.
    let v_spectral = add_up(1.0, right).sqrt().next_up();
    let enclosure = add_up(
        reconstruction_error,
        mul_up(largest, add_up(mul_up(left, v_spectral), right)),
    );
    if !enclosure.is_finite() {
        return Err("SVD reconstruction enclosure overflow".into());
    }
    Ok(SvdValidation {
        reconstruction_frobenius_error_upper: reconstruction_error,
        left_orthogonality_defect_upper: left,
        right_orthogonality_defect_upper: right,
        singular_value_enclosure: enclosure,
    })
}

fn gamma(n: usize) -> Result<f64, String> {
    let e = n as f64 * f64::EPSILON;
    if e >= 0.5 {
        return Err("comparison arithmetic depth is unresolved".into());
    }
    Ok((e / (1.0 - e)).next_up())
}
fn norm(values: impl IntoIterator<Item = f64>) -> Result<(f64, f64), String> {
    let values: Vec<f64> = values.into_iter().collect();
    if values.iter().any(|v| !v.is_finite()) {
        return Err("nonfinite diagnostic value".into());
    }
    let scale = values.iter().fold(0.0_f64, |m, v| m.max(v.abs()));
    if scale == 0.0 {
        return Ok((0.0, 0.0));
    }
    let (mut lo, mut hi) = (0.0_f64, 0.0_f64);
    for x in values {
        let quotient = x.abs() / scale;
        let qlo = quotient.next_down().max(0.0);
        let qhi = quotient.next_up();
        lo = (lo + (qlo * qlo).next_down().max(0.0)).next_down().max(0.0);
        hi = (hi + (qhi * qhi).next_up()).next_up();
    }
    let lower = (lo.sqrt().next_down().max(0.0) * scale)
        .next_down()
        .max(0.0);
    let upper = (hi.sqrt().next_up() * scale).next_up();
    if !upper.is_finite() {
        return Err("diagnostic norm overflow".into());
    }
    let value = lower + (upper - lower) * 0.5;
    Ok((value, (value - lower).max(upper - value).next_up()))
}
fn upper(pair: (f64, f64)) -> f64 {
    (pair.0 + pair.1).next_up()
}
fn floor(value: f64, error: f64, native_upper: f64) -> Result<ResidualFloor, String> {
    if !value.is_finite() || !error.is_finite() || error < 0.0 {
        return Err("unresolved residual diagnostic".into());
    }
    let lower = if native_upper > 0.0 {
        Some(
            ((value - error).next_down().max(0.0) / native_upper)
                .next_down()
                .max(0.0),
        )
    } else {
        None
    };
    Ok(ResidualFloor {
        frobenius_residual: value,
        comparison_error: error,
        necessary_local_lower: lower,
    })
}

fn centered_writes(values: &Array2<f64>) -> Result<(Array2<f64>, f64), String> {
    let (n, d) = values.dim();
    let mut centered = values.clone();
    let mut centering_errors = Vec::with_capacity(values.len());
    for col in 0..d {
        let mut sum = 0.0;
        let mut absolute = 0.0;
        for &v in values.column(col) {
            sum += v;
            absolute = add_up(absolute, v.abs());
        }
        if !sum.is_finite() || !absolute.is_finite() {
            return Err("centering accumulation overflow".into());
        }
        let mean = sum / n as f64;
        let mean_error = add_up(
            mul_up(gamma(n + 2)?, (absolute / n as f64).next_up()),
            mul_up(f64::MIN_POSITIVE, (n + 2) as f64),
        );
        for row in 0..n {
            centered[[row, col]] -= mean;
            centering_errors.push(add_up(
                mean_error,
                add_up(
                    mul_up(f64::EPSILON, add_up(values[[row, col]].abs(), mean.abs())),
                    f64::MIN_POSITIVE,
                ),
            ));
        }
    }
    Ok((centered, upper(norm(centering_errors)?)))
}

/// Fix finite arrays Y (native writes) and B (base predictions). For any
/// correction whose centered output matrix has rank at most K, Eckart–Young
/// bounds ||Y - B - correction||F below by the singular tail of centered Y-B.
/// Dividing by ||Y||F gives a necessary maximum-row / RMS-native-row error.
/// This denominator is the native norm, never the residual norm.
///
/// The subtraction, centering and SVD comparison errors are included. Centering
/// is an orthogonal projection, so it cannot amplify subtraction error. All
/// values here are the supplied binary64 arrays regarded as exact numbers:
/// neural execution and correction evaluation roundoff are not certified.
/// This is conditional on this fixed B, not a lower bound over alternative bases.
/// A constant correction offset is unrestricted and removed by centering.
pub fn measured_fixed_base_rank_curve(
    native: &Array2<f64>,
    base: &Array2<f64>,
    ranks: &[usize],
) -> Result<FixedBaseRankCurve, String> {
    let (rows, output_width) = native.dim();
    if rows == 0 || output_width == 0 || base.dim() != native.dim() {
        return Err("matching nonempty native and base write families required".into());
    }
    let native_norm = norm(native.iter().copied())?;
    let base_norm = norm(base.iter().copied())?;
    let native_upper = if native_norm.0 == 0.0 {
        0.0
    } else {
        mul_up(upper(native_norm), add_up(1.0, gamma(native.len() + 4)?))
    };
    let residual = native - base;
    let residual_norm = norm(residual.iter().copied())?;
    let subtraction_error = subtract_error(upper(native_norm), upper(base_norm), native.len());
    let (centered, centering_error) = centered_writes(&residual)?;
    let decomposition = svd(centered.view(), false).map_err(|e| e.to_string())?;
    let validation = validate_svd(&centered, &decomposition)?;
    let mut corrections = Vec::with_capacity(ranks.len());
    for &rank in ranks {
        let tail = norm(decomposition.singular_values.iter().skip(rank).copied())?;
        let count = decomposition.singular_values.len().saturating_sub(rank);
        let spectral_error = mul_up(
            validation.singular_value_enclosure,
            (count.max(1) as f64).sqrt().next_up(),
        );
        let error = add_up(
            add_up(tail.1, spectral_error),
            add_up(centering_error, subtraction_error),
        );
        corrections.push(CorrectionRankFloor {
            rank,
            affine_correction: floor(tail.0, error, native_upper)?,
        });
    }
    Ok(FixedBaseRankCurve {
        rows,
        output_width,
        native_frobenius_norm: native_norm.0,
        native_norm_error: (native_upper - native_norm.0).max(native_norm.1).next_up(),
        residual_frobenius_norm: residual_norm.0,
        subtraction_error,
        centering_error,
        singular_values: decomposition.singular_values.to_vec(),
        svd_validation: validation,
        corrections,
    })
}

/// Let D=[inputs,1], P be the orthogonal projector onto its column space, and
/// Y be native writes. In exact arithmetic,
///
/// min_{Theta, rank(C)<=K} ||Y-D*Theta-C||F = tail_K((I-P)*Y).
///
/// Projecting any candidate error by I-P gives the lower bound: it eliminates
/// D*Theta and cannot increase rank(C) or Frobenius norm. Least squares plus
/// the truncated residual SVD attains it. Dividing by ||Y||F lower-bounds the
/// maximum-row error normalized by native-row RMS. This is not an upper bound
/// on that maximum, and the relaxed correction need not have a cheap input law.
///
/// This implementation requires *validated full column rank* of D. It never
/// discards unresolved small singular directions: with unrestricted Theta even
/// a tiny nonzero direction can matter. Rank-unresolved designs return an error.
/// All statements concern these fixed finite arrays and exact affine/rank
/// membership, not neural execution roundoff, other inputs, or Run fidelity.
pub fn measured_affine_correction_rank_curve(
    inputs: &Array2<f64>,
    native: &Array2<f64>,
    ranks: &[usize],
) -> Result<AffineCorrectionRankCurve, String> {
    let (rows, input_width) = inputs.dim();
    let output_width = native.ncols();
    if rows == 0 || input_width == 0 || output_width == 0 || native.nrows() != rows {
        return Err("aligned nonempty input and native-write families required".into());
    }
    let columns = input_width
        .checked_add(1)
        .ok_or("affine design width overflow")?;
    if rows < columns {
        return Err("affine design cannot have full column rank on these rows".into());
    }
    let native_norm = norm(native.iter().copied())?;
    let native_upper = if native_norm.0 == 0.0 {
        0.0
    } else {
        mul_up(upper(native_norm), add_up(1.0, gamma(native.len() + 4)?))
    };
    norm(inputs.iter().copied())?;
    let design = Array2::from_shape_fn((rows, columns), |(i, j)| {
        if j == input_width {
            1.0
        } else {
            inputs[[i, j]]
        }
    });
    let ds = svd(design.view(), false).map_err(|e| e.to_string())?;
    let dv = validate_svd(&design, &ds)?;
    let smallest = (*ds.singular_values.last().ok_or("empty affine design SVD")?
        - dv.singular_value_enclosure)
        .next_down();
    if smallest <= 0.0 || !smallest.is_finite() {
        return Err("affine design full column rank is numerically unresolved".into());
    }
    // A nearby orthonormal Uhat spans an exact matrix Dhat with ||D-Dhat||2
    // <= eta. Equal ranks give ||P_D-P_Uhat||2 <= eta/sigma_min(D).
    // Also ||U*U^T-P_Uhat||2 <= ||U^T*U-I||2, validated in dv.
    let projection_error = add_up(
        dv.left_orthogonality_defect_upper,
        (dv.singular_value_enclosure / smallest).next_up(),
    );
    let u_norm = upper(norm(ds.u.iter().copied())?);
    let y_norm = upper(native_norm);
    let amplitudes = ds.u.t().dot(native);
    let a_norm = upper(norm(amplitudes.iter().copied())?);
    let projected = ds.u.dot(&amplitudes);
    let p_norm = upper(norm(projected.iter().copied())?);
    let residual = native - &projected;
    let first_error = dot_error(rows, u_norm, y_norm, amplitudes.len())?;
    let arithmetic = add_up(
        mul_up(u_norm, first_error),
        add_up(
            dot_error(columns, u_norm, a_norm, projected.len())?,
            subtract_error(y_norm, p_norm, native.len()),
        ),
    );
    let residual_error = add_up(arithmetic, mul_up(projection_error, y_norm));
    if !residual_error.is_finite() {
        return Err("affine residual comparison overflow".into());
    }
    let rs = svd(residual.view(), false).map_err(|e| e.to_string())?;
    let rv = validate_svd(&residual, &rs)?;
    let mut corrections = Vec::with_capacity(ranks.len());
    for &rank in ranks {
        let tail = norm(rs.singular_values.iter().skip(rank).copied())?;
        let count = rs.singular_values.len().saturating_sub(rank);
        let spectral_error = mul_up(
            rv.singular_value_enclosure,
            (count.max(1) as f64).sqrt().next_up(),
        );
        corrections.push(CorrectionRankFloor {
            rank,
            affine_correction: floor(
                tail.0,
                add_up(tail.1, add_up(spectral_error, residual_error)),
                native_upper,
            )?,
        });
    }
    Ok(AffineCorrectionRankCurve {
        rows,
        input_width,
        output_width,
        native_frobenius_norm: native_norm.0,
        native_norm_error: (native_upper - native_norm.0).max(native_norm.1).next_up(),
        design_smallest_singular_lower: smallest,
        design_svd_validation: dv,
        projection_operator_error_upper: projection_error,
        residual_arithmetic_error_upper: arithmetic,
        residual_comparison_error_upper: residual_error,
        residual_svd_validation: rv,
        singular_values: rs.singular_values.to_vec(),
        corrections,
    })
}
/// For any predictions in a K-dimensional affine output space, centering makes
/// their matrix rank at most K. Eckart–Young bounds Frobenius error below by the
/// singular-value tail of centered native writes. Since max-row >= Frobenius/√n
/// and Local's denominator is ||native||F/√n, the necessary ratio is tail/||native||F.
///
/// Validated reconstruction/orthogonality enclosures concern these finite input
/// values and comparison arithmetic (the SVD resolution cutoff is not used);
/// they do not bound the neural forward calculation that supplied native writes
/// or a candidate writer's execution roundoff. Necessity assumes exact affine-space
/// membership; out-of-span exceptions/additions invalidate that assumption.
/// A writer matrix has K rows. No input-law assumption or marginal search enters.
pub fn measured_rank_floor(
    native: &Array2<f64>,
    rank: usize,
    writers: Option<&Array2<f64>>,
) -> Result<RankFloor, String> {
    let (n, d) = native.dim();
    if n == 0 || d == 0 {
        return Err("nonempty native-write family required".into());
    }
    if native.iter().any(|v| !v.is_finite()) {
        return Err("nonfinite native write".into());
    }
    let native_norm = norm(native.iter().copied())?;
    let native_upper = if native_norm.0 == 0.0 {
        0.0
    } else {
        mul_up(upper(native_norm), add_up(1.0, gamma(native.len() + 4)?))
    };
    let (centered, centering_error) = centered_writes(native)?;
    let decomposition = svd(centered.view(), false).map_err(|e| e.to_string())?;
    let validation = validate_svd(&centered, &decomposition)?;
    let tail = norm(decomposition.singular_values.iter().skip(rank).copied())?;
    // Weyl plus centering uncertainty in Frobenius norm. Keep every tail value;
    // unresolved values are covered by error, never silently treated as exact zero.
    let tail_count = decomposition.singular_values.len().saturating_sub(rank);
    let spectral_error = mul_up(
        validation.singular_value_enclosure,
        (tail_count.max(1) as f64).sqrt().next_up(),
    );
    let rank_error = ((tail.1 + spectral_error + centering_error) * (1.0 + gamma(8)?)).next_up();
    let affine_rank = floor(tail.0, rank_error, native_upper)?;
    let (mut fixed_writer_span, mut defect, mut writer_validation) = (None, None, None);
    if let Some(b) = writers {
        if b.nrows() != rank || b.ncols() != d || rank == 0 || rank > d {
            return Err(
                "writer rows must equal positive rank and columns native output width".into(),
            );
        }
        let bs = svd(b.view(), false).map_err(|e| e.to_string())?;
        let b_validation = validate_svd(b, &bs)?;
        let b_band = b_validation.singular_value_enclosure;
        if bs.singular_values.iter().any(|&s| s <= b_band) {
            return Err("writer span has unresolved full row rank".into());
        }
        let mut orthogonal_error = 0.0_f64;
        for &s in &bs.singular_values {
            let lo = (s - b_band).next_down().max(0.0);
            let hi = (s + b_band).next_up();
            orthogonal_error = orthogonal_error
                .max((1.0 - (lo * lo).next_down()).max(((hi * hi).next_up() - 1.0).max(0.0)));
        }
        let z_norm = upper(norm(centered.iter().copied())?);
        let b_norm = upper(norm(b.iter().copied())?);
        let amplitudes = centered.dot(&b.t());
        let a_norm = upper(norm(amplitudes.iter().copied())?);
        let projected = amplitudes.dot(b);
        let p_norm = upper(norm(projected.iter().copied())?);
        let residual = &centered - &projected;
        let r_norm = norm(residual.iter().copied())?;
        // P=Bᵀ(BBᵀ)^-1B, while computed projection uses BᵀB. Their exact
        // spectral difference is max|σ(B)^2-1| for full-row-rank B.
        let first_error = dot_error(d, z_norm, b_norm, amplitudes.len())?;
        let arithmetic = add_up(
            mul_up(first_error, b_norm),
            add_up(
                dot_error(rank, a_norm, b_norm, projected.len())?,
                subtract_error(z_norm, p_norm, native.len()),
            ),
        );
        let error = ((r_norm.1 + arithmetic + z_norm * orthogonal_error + centering_error)
            * (1.0 + gamma(24)?))
        .next_up();
        fixed_writer_span = Some(floor(r_norm.0, error, native_upper)?);
        defect = Some(orthogonal_error);
        writer_validation = Some(b_validation);
    }
    Ok(RankFloor {
        rank,
        rows: n,
        output_width: d,
        native_frobenius_norm: native_norm.0,
        native_norm_error: (native_upper - native_norm.0).max(native_norm.1).next_up(),
        centering_error,
        singular_value_band: validation.singular_value_enclosure,
        svd_validation: validation,
        writer_svd_validation: writer_validation,
        affine_rank,
        fixed_writer_span,
        writer_orthogonality_defect_upper: defect,
    })
}
#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::array;
    #[test]
    fn joint_affine_floor_is_attained_by_orthogonal_residual_truncation() {
        let x = array![[-1.0], [-1.0], [1.0], [1.0]];
        let u = [1.0, -1.0, 1.0, -1.0];
        let v = [1.0, -1.0, -1.0, 1.0];
        let y = Array2::from_shape_fn((4, 2), |(i, j)| {
            if j == 0 {
                10.0 * x[[i, 0]] + 5.0 + 2.0 * u[i]
            } else {
                20.0 * x[[i, 0]] - 2.0 + v[i]
            }
        });
        let curve = measured_affine_correction_rank_curve(&x, &y, &[0, 1, 2]).unwrap();
        let denom = 2136.0_f64.sqrt();
        for (point, tail) in curve.corrections.iter().zip([20.0_f64.sqrt(), 2.0, 0.0]) {
            let lower = point.affine_correction.necessary_local_lower.unwrap();
            assert!(lower <= tail / denom + 1e-15);
            assert!((lower - tail / denom).abs() < 1e-9);
        }
        // Rank-one correction C=[2u,0] leaves v in the second coordinate.
        // Each residual row has norm1, so max-row/RMS attains the rank1 floor.
        assert!(
            (curve.corrections[1]
                .affine_correction
                .necessary_local_lower
                .unwrap()
                - 1.0 / (denom / 2.0))
                .abs()
                < 1e-9
        );
        let fixed_zero = measured_fixed_base_rank_curve(&y, &Array2::zeros(y.dim()), &[0]).unwrap();
        assert!(
            fixed_zero.corrections[0]
                .affine_correction
                .necessary_local_lower
                .unwrap()
                > 0.9
        );
        assert!(
            curve.corrections[0]
                .affine_correction
                .necessary_local_lower
                .unwrap()
                < 0.1
        );
    }
    #[test]
    fn joint_affine_floor_is_invariant_to_invertible_input_coordinates() {
        let x = array![[-1.0], [-1.0], [1.0], [1.0]];
        let y = array![[1.0, 1.0], [-1.0, -1.0], [1.0, -1.0], [-1.0, 1.0]];
        let a = measured_affine_correction_rank_curve(&x, &y, &[0, 1]).unwrap();
        let b = measured_affine_correction_rank_curve(&x.mapv(|v| 3.0 + 10.0 * v), &y, &[0, 1])
            .unwrap();
        for (a, b) in a.corrections.iter().zip(&b.corrections) {
            assert!(
                (a.affine_correction.necessary_local_lower.unwrap()
                    - b.affine_correction.necessary_local_lower.unwrap())
                .abs()
                    < 1e-9
            );
        }
    }
    #[test]
    fn joint_affine_floor_rejects_unresolved_design_rank() {
        let y = array![[1.0], [2.0], [3.0], [4.0]];
        // Constant input duplicates the automatically included intercept.
        assert!(
            measured_affine_correction_rank_curve(&Array2::ones((4, 1)), &y, &[0])
                .unwrap_err()
                .contains("rank")
        );
        let x = array![[1e-30], [-1e-30], [1e-30], [-1e-30]];
        // This direction cannot be dropped: unbounded affine coefficients could
        // amplify it. Return unresolved rather than a positive error floor.
        assert!(
            measured_affine_correction_rank_curve(&x, &y, &[0])
                .unwrap_err()
                .contains("rank")
        );
        assert!(
            measured_affine_correction_rank_curve(&Array2::ones((1, 2)), &array![[1.0]], &[0])
                .is_err()
        );
    }
    #[test]
    fn joint_affine_floor_vanishes_for_exact_affine_outputs() {
        let x = array![[-2.0], [-1.0], [1.0], [2.0]];
        let y = Array2::from_shape_fn((4, 2), |(i, j)| (j + 1) as f64 * x[[i, 0]] + 3.0);
        let curve = measured_affine_correction_rank_curve(&x, &y, &[0]).unwrap();
        assert_eq!(
            curve.corrections[0].affine_correction.necessary_local_lower,
            Some(0.0)
        );
        let zeros =
            measured_affine_correction_rank_curve(&x, &Array2::zeros((4, 2)), &[0]).unwrap();
        assert_eq!(
            zeros.corrections[0].affine_correction.necessary_local_lower,
            None
        );
    }
    #[test]
    fn fixed_full_rank_base_escapes_low_rank_output_restriction() {
        let native = array![[1.0, 0.0], [-1.0, 0.0], [0.0, 1.0], [0.0, -1.0]];
        assert!(
            measured_rank_floor(&native, 0, None)
                .unwrap()
                .affine_rank
                .necessary_local_lower
                .unwrap()
                > 0.99
        );
        let curve = measured_fixed_base_rank_curve(&native, &native, &[0, 1, 2]).unwrap();
        for correction in curve.corrections {
            assert_eq!(
                correction.affine_correction.necessary_local_lower,
                Some(0.0)
            );
        }
    }
    #[test]
    fn correction_floor_uses_native_denominator_and_unrestricted_offset() {
        let native = array![[10.0, 0.0], [-10.0, 0.0], [0.0, 10.0], [0.0, -10.0]];
        // Residual is 0.1*native plus a constant. The latter is free under the
        // affine-rank relaxation; it must not inflate the rank-zero floor.
        let base = native.mapv(|x| 0.9 * x - 100.0);
        let curve = measured_fixed_base_rank_curve(&native, &base, &[0, 1, 2, 99]).unwrap();
        let expected = [0.1, 0.1 / 2.0_f64.sqrt(), 0.0, 0.0];
        for (point, expected) in curve.corrections.iter().zip(expected) {
            let lower = point.affine_correction.necessary_local_lower.unwrap();
            assert!(lower <= expected + 1e-15);
            assert!((lower - expected).abs() < 1e-9);
        }
        // Changing the fixed base is a substantive change to the premise.
        let zero = Array2::zeros(native.dim());
        let other = measured_fixed_base_rank_curve(&native, &zero, &[0]).unwrap();
        assert!(
            other.corrections[0]
                .affine_correction
                .necessary_local_lower
                .unwrap()
                > 0.99
        );
    }
    #[test]
    fn fixed_base_curve_covers_rounded_subtraction_and_zero_native_norm() {
        let native = array![[1.0, 0.0], [-1.0, 0.0]];
        let base = array![[f64::EPSILON / 4.0, 0.0], [0.0, 0.0]];
        assert_eq!((native[[0, 0]] - base[[0, 0]]).to_bits(), 1.0_f64.to_bits());
        let curve = measured_fixed_base_rank_curve(&native, &base, &[0]).unwrap();
        assert!(curve.subtraction_error >= f64::EPSILON / 4.0);
        assert!(
            curve.corrections[0]
                .affine_correction
                .necessary_local_lower
                .unwrap()
                < 1.0
        );
        let zeros = Array2::zeros((2, 2));
        let zero_curve = measured_fixed_base_rank_curve(&zeros, &native, &[0]).unwrap();
        assert!(
            zero_curve.corrections[0]
                .affine_correction
                .necessary_local_lower
                .is_none()
        );
    }
    #[test]
    fn fixed_base_curve_rejects_mismatched_and_nonfinite_arrays() {
        let native = array![[1.0, 2.0]];
        assert!(measured_fixed_base_rank_curve(&native, &Array2::zeros((2, 1)), &[0]).is_err());
        assert!(measured_fixed_base_rank_curve(&native, &array![[f64::NAN, 0.0]], &[0]).is_err());
        assert!(
            measured_fixed_base_rank_curve(&Array2::zeros((0, 2)), &Array2::zeros((0, 2)), &[0])
                .is_err()
        );
    }
    #[test]
    fn enclosure_validates_actual_factors_and_ignores_claimed_cutoff() {
        let a = array![[3.0, 0.0], [0.0, 1.0]];
        let mut decomposition = svd(a.view(), false).unwrap();
        decomposition.band = 0.0;
        decomposition.singular_values[0] += 0.1;
        let validation = validate_svd(&a, &decomposition).unwrap();
        assert!(validation.reconstruction_frobenius_error_upper >= 0.1);
        assert!(validation.singular_value_enclosure >= 0.1);
        assert!(validation.singular_value_enclosure < 0.101);
        // Orthogonality cannot be assumed from an SVD-shaped struct or band.
        decomposition.u.column_mut(0).mapv_inplace(|v| 2.0 * v);
        assert!(validate_svd(&a, &decomposition).is_err());
    }
    #[test]
    fn enclosure_covers_nonorthogonal_but_nearby_polar_factors() {
        let a = array![[3.0, 0.0], [0.0, 1.0]];
        let mut decomposition = svd(a.view(), false).unwrap();
        decomposition.band = 0.0;
        // Keep reconstruction exact while moving both factors off orthonormal.
        decomposition.u.column_mut(0).mapv_inplace(|v| 1.001 * v);
        decomposition.singular_values[0] /= 1.001;
        let actual_error = 3.0 - decomposition.singular_values[0];
        let validation = validate_svd(&a, &decomposition).unwrap();
        assert!(validation.left_orthogonality_defect_upper > 0.002);
        assert!(validation.singular_value_enclosure >= actual_error);
    }
    #[test]
    fn unavoidable_rank_loss_is_a_real_local_lower_bound() {
        let y = array![[1.0, 0.0], [-1.0, 0.0], [0.0, 1.0], [0.0, -1.0]];
        let bound = measured_rank_floor(&y, 1, Some(&array![[1.0, 0.0]])).unwrap();
        let exact = (0.5_f64).sqrt();
        let lower = bound.affine_rank.necessary_local_lower.unwrap();
        assert!(lower > 0.70 && lower <= exact);
        assert!(bound.affine_rank.comparison_error > 0.0);
        let actual_maxrow = 1.0;
        let rms = 1.0;
        assert!(lower <= actual_maxrow / rms);
        let fixed = bound.fixed_writer_span.unwrap();
        assert!(fixed.necessary_local_lower.unwrap() > 0.70);
    }
    #[test]
    fn floor_holds_for_different_affine_prediction_spaces_and_max_rows() {
        let y = array![[3.0, 0.0], [-3.0, 0.0], [0.0, 1.0], [0.0, -1.0]];
        let bound = measured_rank_floor(&y, 1, None).unwrap();
        let lower = bound.affine_rank.necessary_local_lower.unwrap();
        let rms = (20.0_f64 / 4.0).sqrt();
        for angle in [0.0_f64, 0.2, 0.8, 1.5, 2.3] {
            let direction = [angle.cos(), angle.sin()];
            let max_row = y
                .outer_iter()
                .map(|row| {
                    let amplitude = row[0] * direction[0] + row[1] * direction[1];
                    (row[0] - amplitude * direction[0]).hypot(row[1] - amplitude * direction[1])
                })
                .fold(0.0_f64, f64::max);
            assert!(lower <= max_row / rms);
        }
    }
    #[test]
    fn unresolved_nonzero_tail_is_not_claimed_exactly_zero() {
        let y = array![[1.0, 0.0], [-1.0, 0.0], [0.0, 1e-20], [0.0, -1e-20]];
        let bound = measured_rank_floor(&y, 1, None).unwrap();
        assert!(bound.affine_rank.frobenius_residual > 0.0);
        assert!(bound.affine_rank.comparison_error > bound.affine_rank.frobenius_residual);
        assert_eq!(bound.affine_rank.necessary_local_lower, Some(0.0));
    }
    #[test]
    fn affine_offset_and_fixed_span_are_distinguished() {
        let y = array![[100.0, -2.0], [100.0, 2.0]];
        let bound = measured_rank_floor(&y, 1, Some(&array![[1.0, 0.0]])).unwrap();
        assert_eq!(bound.affine_rank.necessary_local_lower, Some(0.0));
        assert!(bound.affine_rank.comparison_error > 0.0);
        assert!(
            bound
                .fixed_writer_span
                .unwrap()
                .necessary_local_lower
                .unwrap()
                > 0.0199
        );
        let good = measured_rank_floor(&y, 1, Some(&array![[0.0, 1.0]])).unwrap();
        assert_eq!(
            good.fixed_writer_span.unwrap().necessary_local_lower,
            Some(0.0)
        );
    }
    #[test]
    fn finite_nonorthogonal_basis_and_unresolved_inputs_are_handled() {
        let y = array![[1.0, 0.0], [-1.0, 0.0], [0.0, 1.0], [0.0, -1.0]];
        let approximate = measured_rank_floor(&y, 1, Some(&array![[1.00001, 0.0]])).unwrap();
        assert!(
            approximate
                .fixed_writer_span
                .unwrap()
                .necessary_local_lower
                .unwrap()
                > 0.70
        );
        assert!(measured_rank_floor(&y, 1, Some(&array![[0.0, 0.0]])).is_err());
        assert!(measured_rank_floor(&array![[f64::NAN]], 0, None).is_err());
        assert_eq!(
            measured_rank_floor(&array![[0.0]], 0, None)
                .unwrap()
                .affine_rank
                .necessary_local_lower,
            None
        );
    }
}
