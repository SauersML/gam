use ndarray::{Array2, ArrayView1, ArrayView2};

use crate::normalize_weights;

pub fn validate_simplex_array(points: ArrayView2<'_, f64>) -> Result<(), String> {
    let (n, d) = points.dim();
    if n == 0 || d < 2 {
        return Err(
            "simplex values must have at least one row and at least two columns".to_string(),
        );
    }
    if let Some(((row, col), value)) = points.indexed_iter().find(|(_, v)| !v.is_finite()) {
        return Err(format!(
            "simplex values must contain only finite values; got {value} at ({row}, {col})"
        ));
    }
    Ok(())
}

pub fn closure(points: ArrayView2<'_, f64>) -> Result<Array2<f64>, String> {
    validate_simplex_array(points)?;
    let (n, d) = points.dim();
    let mut out = Array2::<f64>::zeros((n, d));
    for row in 0..n {
        let mut total = 0.0_f64;
        for col in 0..d {
            let v = points[[row, col]];
            if v < 0.0 {
                return Err("simplex values must be non-negative".to_string());
            }
            total += v;
        }
        if total <= 0.0 {
            return Err("simplex rows must have positive total mass".to_string());
        }
        for col in 0..d {
            out[[row, col]] = points[[row, col]] / total;
        }
    }
    Ok(out)
}

fn require_positive(comp: ArrayView2<'_, f64>, label: &str) -> Result<(), String> {
    for value in comp.iter() {
        if *value <= 0.0 {
            return Err(format!("{label} require strictly positive simplex values"));
        }
    }
    Ok(())
}

pub fn simplex_frechet_mean(
    points: ArrayView2<'_, f64>,
    weights: Option<ArrayView1<'_, f64>>,
) -> Result<Vec<f64>, String> {
    let comp = closure(points)?;
    require_positive(comp.view(), "simplex Fr\u{e9}chet mean")?;
    let (n, d) = comp.dim();
    let w = normalize_weights(n, weights)?;
    let mut mean_log = vec![0.0_f64; d];
    for row in 0..n {
        for col in 0..d {
            mean_log[col] += w[row] * comp[[row, col]].ln();
        }
    }
    let mut max_v = f64::NEG_INFINITY;
    for &v in mean_log.iter() {
        if v > max_v {
            max_v = v;
        }
    }
    let mut total = 0.0_f64;
    let mut out = vec![0.0_f64; d];
    for col in 0..d {
        let e = (mean_log[col] - max_v).exp();
        out[col] = e;
        total += e;
    }
    for value in out.iter_mut() {
        *value /= total;
    }
    Ok(out)
}

/// Coordinate system for simplex (Aitchison) log/exp maps: centered log-ratio
/// (`Clr`, `d`-dim, sum-zero) or additive log-ratio (`Alr`, `(d-1)`-dim relative
/// to a reference part).
#[derive(Copy, Clone, Debug, PartialEq, Eq)]
pub enum SimplexCoord {
    Clr,
    Alr,
}

/// Parse a simplex coordinate label. `"simplex"`/`"clr"` → CLR, `"alr"` → ALR.
pub fn parse_simplex_coord(coordinates: &str) -> Result<SimplexCoord, String> {
    match coordinates.to_ascii_lowercase().as_str() {
        "simplex" | "clr" => Ok(SimplexCoord::Clr),
        "alr" => Ok(SimplexCoord::Alr),
        other => Err(format!(
            "simplex coordinates must be 'clr' or 'alr'; got {other:?}"
        )),
    }
}

/// Wrap a (possibly negative) reference index into `0..d`.
fn resolve_reference(reference: isize, d: usize) -> usize {
    let d_i = d as isize;
    let mut r = reference % d_i;
    if r < 0 {
        r += d_i;
    }
    r as usize
}

/// Centered log-ratio coordinates: `clr(x)_j = ln x_j - mean_k ln x_k` after
/// closure. Requires strictly positive compositions.
pub fn clr(values: ArrayView2<'_, f64>) -> Result<Array2<f64>, String> {
    let comp = closure(values)?;
    require_positive(comp.view(), "CLR coordinates")?;
    let (n, d) = comp.dim();
    let mut out = Array2::<f64>::zeros((n, d));
    for row in 0..n {
        let mut sum_log = 0.0_f64;
        for col in 0..d {
            let lg = comp[[row, col]].ln();
            out[[row, col]] = lg;
            sum_log += lg;
        }
        let mean = sum_log / (d as f64);
        for col in 0..d {
            out[[row, col]] -= mean;
        }
    }
    Ok(out)
}

/// Additive log-ratio coordinates relative to `reference`: `alr(x)_j = ln x_j -
/// ln x_ref` for `j != ref`, yielding `(d-1)` columns. Requires strictly
/// positive compositions.
pub fn alr(values: ArrayView2<'_, f64>, reference: isize) -> Result<Array2<f64>, String> {
    let comp = closure(values)?;
    require_positive(comp.view(), "ALR coordinates")?;
    let (n, d) = comp.dim();
    let ref_idx = resolve_reference(reference, d);
    let mut out = Array2::<f64>::zeros((n, d - 1));
    for row in 0..n {
        let log_ref = comp[[row, ref_idx]].ln();
        let mut k = 0usize;
        for col in 0..d {
            if col == ref_idx {
                continue;
            }
            out[[row, k]] = comp[[row, col]].ln() - log_ref;
            k += 1;
        }
    }
    Ok(out)
}

/// Inverse additive log-ratio: map `(d-1)` ALR coordinates back to the simplex
/// via a numerically stable softmax with the reference logit pinned to zero.
pub fn inverse_alr(coords: ArrayView2<'_, f64>, reference: isize) -> Result<Array2<f64>, String> {
    let (n, dm1) = coords.dim();
    if !coords.iter().all(|v| v.is_finite()) {
        return Err("ALR coordinates must contain only finite values".to_string());
    }
    let d = dm1 + 1;
    let ref_idx = resolve_reference(reference, d);
    let mut out = Array2::<f64>::zeros((n, d));
    for row in 0..n {
        let mut max_v = f64::NEG_INFINITY;
        let mut k = 0usize;
        for col in 0..d {
            let v = if col == ref_idx {
                0.0
            } else {
                let val = coords[[row, k]];
                k += 1;
                val
            };
            out[[row, col]] = v;
            if v > max_v {
                max_v = v;
            }
        }
        let mut total = 0.0_f64;
        for col in 0..d {
            let e = (out[[row, col]] - max_v).exp();
            out[[row, col]] = e;
            total += e;
        }
        for col in 0..d {
            out[[row, col]] /= total;
        }
    }
    Ok(out)
}

/// Riemannian log map at an intrinsic simplex base point, expressed in the
/// chosen coordinate system: the difference of the values' and base's CLR/ALR
/// coordinates.
pub fn simplex_log_map(
    values: ArrayView2<'_, f64>,
    base: ArrayView1<'_, f64>,
    coord: SimplexCoord,
    reference: isize,
) -> Result<Array2<f64>, String> {
    let comp = closure(values)?;
    let base2 = Array2::from_shape_fn((1, base.len()), |(_, j)| base[j]);
    let base_comp = closure(base2.view())?;
    if comp.ncols() != base_comp.ncols() {
        return Err("simplex values and base point have different dimensions".to_string());
    }
    require_positive(comp.view(), "simplex log map")?;
    require_positive(base_comp.view(), "simplex log map")?;
    match coord {
        SimplexCoord::Clr => {
            let values_clr = clr(values)?;
            let base_clr = clr(base2.view())?;
            let (n, d) = values_clr.dim();
            let mut out = Array2::<f64>::zeros((n, d));
            for row in 0..n {
                for col in 0..d {
                    out[[row, col]] = values_clr[[row, col]] - base_clr[[0, col]];
                }
            }
            Ok(out)
        }
        SimplexCoord::Alr => {
            let values_alr = alr(values, reference)?;
            let base_alr = alr(base2.view(), reference)?;
            let (n, dm1) = values_alr.dim();
            let mut out = Array2::<f64>::zeros((n, dm1));
            for row in 0..n {
                for col in 0..dm1 {
                    out[[row, col]] = values_alr[[row, col]] - base_alr[[0, col]];
                }
            }
            Ok(out)
        }
    }
}

/// Riemannian exp map from tangent coordinates back to the simplex at `base`,
/// inverting [`simplex_log_map`] for the matching coordinate system.
pub fn simplex_exp_map(
    tangent: ArrayView2<'_, f64>,
    base: ArrayView1<'_, f64>,
    coord: SimplexCoord,
    reference: isize,
) -> Result<Array2<f64>, String> {
    let base2 = Array2::from_shape_fn((1, base.len()), |(_, j)| base[j]);
    let base_comp = closure(base2.view())?;
    let d = base_comp.ncols();
    match coord {
        SimplexCoord::Clr => {
            if tangent.ncols() != d {
                return Err("CLR tangent dimension must equal simplex dimension".to_string());
            }
            require_positive(base_comp.view(), "simplex exp map")?;
            let n = tangent.nrows();
            let mut out = Array2::<f64>::zeros((n, d));
            for row in 0..n {
                let mut max_v = f64::NEG_INFINITY;
                for col in 0..d {
                    let lg = base_comp[[0, col]].ln() + tangent[[row, col]];
                    out[[row, col]] = lg;
                    if lg > max_v {
                        max_v = lg;
                    }
                }
                let mut total = 0.0_f64;
                for col in 0..d {
                    let e = (out[[row, col]] - max_v).exp();
                    out[[row, col]] = e;
                    total += e;
                }
                for col in 0..d {
                    out[[row, col]] /= total;
                }
            }
            Ok(out)
        }
        SimplexCoord::Alr => {
            if tangent.ncols() + 1 != d {
                return Err("ALR tangent dimension must be simplex dimension minus one".to_string());
            }
            let base_alr = alr(base2.view(), reference)?;
            let n = tangent.nrows();
            let dm1 = d - 1;
            let mut shifted = Array2::<f64>::zeros((n, dm1));
            for row in 0..n {
                for col in 0..dm1 {
                    shifted[[row, col]] = base_alr[[0, col]] + tangent[[row, col]];
                }
            }
            inverse_alr(shifted.view(), reference)
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::{Array1, array};

    // ── parse_simplex_coord ───────────────────────────────────────────────────

    #[test]
    fn parse_simplex_coord_simplex_and_clr_map_to_clr() {
        assert_eq!(parse_simplex_coord("simplex").unwrap(), SimplexCoord::Clr);
        assert_eq!(parse_simplex_coord("clr").unwrap(), SimplexCoord::Clr);
    }

    #[test]
    fn parse_simplex_coord_alr_maps_to_alr() {
        assert_eq!(parse_simplex_coord("alr").unwrap(), SimplexCoord::Alr);
    }

    #[test]
    fn parse_simplex_coord_case_insensitive() {
        assert_eq!(parse_simplex_coord("CLR").unwrap(), SimplexCoord::Clr);
        assert_eq!(parse_simplex_coord("ALR").unwrap(), SimplexCoord::Alr);
        assert_eq!(parse_simplex_coord("Simplex").unwrap(), SimplexCoord::Clr);
    }

    #[test]
    fn parse_simplex_coord_unknown_is_error() {
        assert!(parse_simplex_coord("pca").is_err());
        assert!(parse_simplex_coord("").is_err());
    }

    // ── validate_simplex_array ────────────────────────────────────────────────

    #[test]
    fn validate_simplex_array_valid_input_passes() {
        let m = array![[0.5_f64, 0.5]];
        assert!(validate_simplex_array(m.view()).is_ok());
    }

    #[test]
    fn validate_simplex_array_no_rows_is_error() {
        use ndarray::Array2;
        let m: Array2<f64> = Array2::zeros((0, 3));
        assert!(validate_simplex_array(m.view()).is_err());
    }

    #[test]
    fn validate_simplex_array_single_column_is_error() {
        let m = array![[0.5_f64]];
        assert!(validate_simplex_array(m.view()).is_err());
    }

    #[test]
    fn validate_simplex_array_non_finite_is_error() {
        let m = array![[0.5_f64, f64::NAN]];
        let err = validate_simplex_array(m.view()).unwrap_err();
        assert!(err.contains("finite"), "error should mention finite, got: {err}");
    }

    // ── closure ───────────────────────────────────────────────────────────────

    #[test]
    fn closure_normalizes_rows_to_sum_one() {
        let m = array![[1.0_f64, 2.0, 3.0], [4.0, 4.0, 4.0]];
        let c = closure(m.view()).unwrap();
        assert!((c.row(0).sum() - 1.0).abs() < 1e-14, "row 0 sum: {}", c.row(0).sum());
        assert!((c.row(1).sum() - 1.0).abs() < 1e-14, "row 1 sum: {}", c.row(1).sum());
    }

    #[test]
    fn closure_equal_weights_gives_uniform_composition() {
        let m = array![[2.0_f64, 2.0]];
        let c = closure(m.view()).unwrap();
        assert!((c[[0, 0]] - 0.5).abs() < 1e-14);
        assert!((c[[0, 1]] - 0.5).abs() < 1e-14);
    }

    #[test]
    fn closure_negative_value_is_error() {
        let m = array![[1.0_f64, -0.5]];
        assert!(closure(m.view()).is_err());
    }

    #[test]
    fn closure_zero_total_mass_is_error() {
        let m = array![[0.0_f64, 0.0]];
        let err = closure(m.view()).unwrap_err();
        assert!(err.contains("total mass") || err.contains("positive"), "got: {err}");
    }

    // ── resolve_reference ─────────────────────────────────────────────────────

    #[test]
    fn resolve_reference_positive_index() {
        assert_eq!(resolve_reference(1, 3), 1);
        assert_eq!(resolve_reference(2, 3), 2);
    }

    #[test]
    fn resolve_reference_negative_index_wraps() {
        // -1 → last element (d-1)
        assert_eq!(resolve_reference(-1, 3), 2);
        // -2 → second-to-last
        assert_eq!(resolve_reference(-2, 3), 1);
        // -3 → first (same as 0)
        assert_eq!(resolve_reference(-3, 3), 0);
    }

    // ── clr known values ──────────────────────────────────────────────────────

    #[test]
    fn clr_of_uniform_composition_is_zero() {
        // clr([1/3, 1/3, 1/3]) = [0, 0, 0]
        let m = array![[1.0_f64, 1.0, 1.0]];
        let c = clr(m.view()).unwrap();
        for v in c.iter() {
            assert!(v.abs() < 1e-14, "clr of uniform should be 0, got {v}");
        }
    }

    #[test]
    fn clr_sum_is_zero_per_row() {
        let m = array![[1.0_f64, 2.0, 3.0], [4.0, 1.0, 1.0]];
        let c = clr(m.view()).unwrap();
        for row in c.rows() {
            assert!(row.sum().abs() < 1e-12, "clr row must sum to zero, got {}", row.sum());
        }
    }

    // ── alr / inverse_alr round-trip ─────────────────────────────────────────

    #[test]
    fn alr_inverse_alr_round_trip() {
        let m = array![[0.2_f64, 0.5, 0.3]];
        let coords = alr(m.view(), -1).unwrap(); // reference = last
        let recovered = inverse_alr(coords.view(), -1).unwrap();
        for col in 0..3 {
            assert!(
                (recovered[[0, col]] - m[[0, col]]).abs() < 1e-12,
                "col {col}: {} vs {}",
                recovered[[0, col]],
                m[[0, col]]
            );
        }
    }

    /// CLR exp map at a strictly-interior base with a finite tangent succeeds
    /// and lands in the open simplex (all components strictly positive, summing
    /// to one).
    #[test]
    fn clr_exp_map_interior_base_lands_in_open_simplex() {
        let base: Array1<f64> = array![0.2, 0.5, 0.3];
        let tangent = array![[0.4_f64, -0.1, -0.3]];
        let out = simplex_exp_map(tangent.view(), base.view(), SimplexCoord::Clr, 0)
            .expect("interior base with finite tangent must succeed");
        let sum: f64 = out.row(0).sum();
        assert!((sum - 1.0).abs() < 1e-12, "components must sum to one");
        for v in out.iter() {
            assert!(*v > 0.0, "components must be strictly positive; got {v}");
        }
    }

    /// CLR exp map at a boundary base (a zero component, on the closed simplex
    /// but off the Aitchison manifold) must error rather than produce NaN.
    #[test]
    fn clr_exp_map_boundary_base_errors() {
        let base: Array1<f64> = array![1.0, 0.0, 0.0];
        let tangent = array![[0.1_f64, -0.05, -0.05]];
        let err = simplex_exp_map(tangent.view(), base.view(), SimplexCoord::Clr, 0)
            .expect_err("boundary base must be rejected, not yield NaN");
        assert!(
            err.contains("strictly positive"),
            "error must explain the positivity domain; got {err}"
        );
    }

    /// CLR log map followed by exp map at the same interior base recovers the
    /// original interior point.
    #[test]
    fn clr_log_exp_round_trip_recovers_interior_point() {
        let base: Array1<f64> = array![0.25, 0.45, 0.30];
        let point = array![[0.1_f64, 0.6, 0.3]];
        let tangent = simplex_log_map(point.view(), base.view(), SimplexCoord::Clr, 0)
            .expect("log map at interior base must succeed");
        let recovered = simplex_exp_map(tangent.view(), base.view(), SimplexCoord::Clr, 0)
            .expect("exp map at interior base must succeed");
        for col in 0..3 {
            assert!(
                (recovered[[0, col]] - point[[0, col]]).abs() < 1e-12,
                "round-trip must recover input at column {col}: {} vs {}",
                recovered[[0, col]],
                point[[0, col]]
            );
        }
    }
}
