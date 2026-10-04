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
    let mut centered = native.clone();
    let mut centering_errors = Vec::with_capacity(native.len());
    for col in 0..d {
        let mut sum = 0.0;
        let mut absolute = 0.0;
        for &v in native.column(col) {
            sum += v;
            absolute += v.abs();
        }
        if !sum.is_finite() || !absolute.is_finite() {
            return Err("centering accumulation overflow".into());
        }
        let mean = sum / n as f64;
        let mean_error =
            (gamma(n + 2)? * absolute / n as f64 + f64::MIN_POSITIVE * n as f64).next_up();
        for row in 0..n {
            centered[[row, col]] -= mean;
            centering_errors.push(
                (mean_error
                    + f64::EPSILON * (native[[row, col]].abs() + mean.abs())
                    + f64::MIN_POSITIVE)
                    .next_up(),
            );
        }
    }
    let centering_error = upper(norm(centering_errors)?);
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
