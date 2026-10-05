//! Fixed-design coefficient fitting around an explicit reference, using a thin SVD.
//! This solves a numerical subproblem; it does not discover the design or certify
//! generalization. No normal-equation inverse or silent f32 quantization is used.
use gam_linalg::{
    decompose::svd,
    faer_ndarray::{FaerArrayView, HouseholderQr, fast_ab, fast_atb},
};
use ndarray::{Array2, Axis, s};
use serde::{Deserialize, Serialize};
use std::ops::Range;

#[derive(Clone, Debug, Default, Serialize, Deserialize)]
#[serde(default, deny_unknown_fields)]
pub struct Settings {
    /// Relative singular-value cutoff. None uses eps * max(rows, columns).
    /// The effective absolute cutoff is at least the SVD's rounding band.
    pub relative_rank_cutoff: Option<f64>,
    /// Lambda in ||X B - Y||_F^2 + lambda ||B - B0||_F^2. No row averaging.
    pub ridge: f64,
}

#[derive(Clone, Debug, Serialize)]
pub struct Diagnostics {
    pub rows: usize,
    pub features: usize,
    pub outputs: usize,
    pub retained_rank: usize,
    pub singular_values: Vec<f64>,
    pub relative_rank_cutoff: f64,
    pub absolute_rank_cutoff: f64,
    pub svd_rounding_band: f64,
    pub retained_condition_number: Option<f64>,
    pub ridge: f64,
    pub initial_residual_frobenius: f64,
    pub final_residual_frobenius: f64,
    pub coefficient_change_frobenius: f64,
    /// ||R - U_retained U_retained^T R||_F for R = Y - X B0.
    /// Numerical residual floor with updates restricted to retained right modes;
    /// truncation means this need not be the unrestricted least-squares minimum.
    pub retained_subspace_residual_floor: f64,
    pub method: &'static str,
    pub row_tiles: usize,
    /// Numerical rank guard accumulated across QR reductions, not a certified
    /// roundoff bound. Zero for the direct SVD route.
    pub qr_rounding_guard: f64,
    pub scope: &'static str,
}

pub struct Fit {
    /// p by k, predicting X[n,p] * coefficients[p,k].
    pub coefficients: Array2<f64>,
    pub diagnostics: Diagnostics,
}

fn norm(values: &Array2<f64>) -> Result<f64, String> {
    let value = values.iter().fold(0f64, |n, v| n.hypot(*v));
    if value.is_finite() {
        Ok(value)
    } else {
        Err("nonfinite coefficient-fit norm".into())
    }
}

/// Fit B = B0 + delta, restricting delta to retained right singular modes.
/// At lambda=0 this is the minimum-change least-squares solution in that subspace.
/// At lambda>0 it is the unique ridge update in that subspace. Discarded and null
/// directions retain B0, rather than being silently set to zero.
pub fn fit(
    design: &Array2<f64>,
    targets: &Array2<f64>,
    reference: &Array2<f64>,
    settings: Settings,
) -> Result<Fit, String> {
    if design.nrows() > design.ncols().saturating_mul(2)
        && targets.nrows() == design.nrows()
        && reference.dim() == (design.ncols(), targets.ncols())
    {
        let tile_rows = design.ncols().max(256).min(4096);
        return fit_residual_streaming(design.nrows(), reference, settings, tile_rows, |rows| {
            let x = design.slice(s![rows.clone(), ..]).to_owned();
            let residual = targets.slice(s![rows, ..]).to_owned() - fast_ab(&x, reference);
            Ok((x, residual))
        });
    }
    fit_dense(design, targets, reference, settings)
}

fn fit_dense(
    design: &Array2<f64>,
    targets: &Array2<f64>,
    reference: &Array2<f64>,
    settings: Settings,
) -> Result<Fit, String> {
    let (n, p) = design.dim();
    let k = targets.ncols();
    if n == 0
        || p == 0
        || k == 0
        || targets.nrows() != n
        || reference.dim() != (p, k)
        || design
            .iter()
            .chain(targets.iter())
            .chain(reference.iter())
            .any(|v| !v.is_finite())
    {
        return Err("coefficient fit needs finite nonempty X[n,p], Y[n,k], B0[p,k]".into());
    }
    let relative = settings
        .relative_rank_cutoff
        .unwrap_or(f64::EPSILON * n.max(p) as f64);
    if !relative.is_finite()
        || !(0. ..1.).contains(&relative)
        || !settings.ridge.is_finite()
        || settings.ridge < 0.
    {
        return Err("relative rank cutoff must be in [0,1), ridge finite and nonnegative".into());
    }
    let factor = svd(design.view(), false).map_err(|e| e.to_string())?;
    if !factor.band.is_finite()
        || factor.band < 0.
        || factor
            .singular_values
            .iter()
            .any(|v| !v.is_finite() || *v < 0.)
        || factor
            .u
            .iter()
            .chain(factor.vt.iter())
            .any(|v| !v.is_finite())
    {
        return Err("nonfinite coefficient-fit SVD".into());
    }
    let singular_values = factor.singular_values.to_vec();
    let largest = singular_values[0];
    let cutoff = (relative * largest).max(factor.band);
    let retained = singular_values
        .iter()
        .enumerate()
        .filter_map(|(i, s)| (*s > cutoff).then_some(i))
        .collect::<Vec<_>>();
    let residual = targets - &fast_ab(design, reference);
    let initial_residual_frobenius = norm(&residual)?;
    let mut coefficients = reference.clone();
    let mut floor = residual.clone();
    if !retained.is_empty() {
        let u = factor.u.select(Axis(1), &retained);
        let vt = factor.vt.select(Axis(0), &retained);
        let mut projected = fast_atb(&u, &residual);
        floor -= &fast_ab(&u, &projected);
        let root_ridge = settings.ridge.sqrt();
        for (row, &index) in retained.iter().enumerate() {
            let sigma = singular_values[index];
            // sigma/(sigma^2+lambda) without squaring large singular values.
            let denominator = sigma.hypot(root_ridge);
            let gain = (sigma / denominator) / denominator;
            if !gain.is_finite() {
                return Err("unresolved coefficient-fit reciprocal singular value".into());
            }
            projected.row_mut(row).mapv_inplace(|v| v * gain);
        }
        coefficients += &fast_atb(&vt, &projected);
    }
    if coefficients.iter().any(|v| !v.is_finite()) {
        return Err("nonfinite fitted coefficients".into());
    }
    let final_residual_frobenius = norm(&(targets - &fast_ab(design, &coefficients)))?;
    let coefficient_change_frobenius = norm(&(&coefficients - reference))?;
    let retained_condition_number = retained.last().and_then(|&i| {
        let condition = largest / singular_values[i];
        condition.is_finite().then_some(condition)
    });
    Ok(Fit {
        coefficients,
        diagnostics: Diagnostics {
            rows: n,
            features: p,
            outputs: k,
            retained_rank: retained.len(),
            singular_values,
            relative_rank_cutoff: relative,
            absolute_rank_cutoff: cutoff,
            svd_rounding_band: factor.band,
            retained_condition_number,
            ridge: settings.ridge,
            initial_residual_frobenius,
            final_residual_frobenius,
            coefficient_change_frobenius,
            retained_subspace_residual_floor: norm(&floor)?,
            method: "direct_svd",
            row_tiles: 1,
            qr_rounding_guard: 0.,
            scope: "Fixed-design f64 SVD solve around supplied reference. Rank and residuals are numerical diagnostics, not exact rank or roundoff certificates. Ridge objective has no row averaging. Updates are restricted to retained singular modes; all discarded reference directions are preserved up to arithmetic rounding.",
        },
    })
}

/// Stream every row twice. `read` supplies X and the reference residual Y-X B0,
/// with unchanged row order/content on both passes. Neither X, Q, U nor residuals
/// of full panel height are retained. Householder reductions keep [R, Qᵀ residual]
/// plus the robust norm of eliminated RHS rows; SVD acts only on the small R.
/// The second pass measures the actual residual of the returned coefficients.
pub fn fit_residual_streaming(
    rows: usize,
    reference: &Array2<f64>,
    settings: Settings,
    tile_rows: usize,
    mut read: impl FnMut(Range<usize>) -> Result<(Array2<f64>, Array2<f64>), String>,
) -> Result<Fit, String> {
    let (p, k) = reference.dim();
    let relative = settings
        .relative_rank_cutoff
        .unwrap_or(f64::EPSILON * rows.max(p) as f64);
    if rows == 0
        || p == 0
        || k == 0
        || tile_rows == 0
        || reference.iter().any(|v| !v.is_finite())
        || !relative.is_finite()
        || !(0. ..1.).contains(&relative)
        || !settings.ridge.is_finite()
        || settings.ridge < 0.
    {
        return Err("streaming coefficient fit needs positive dimensions/tiles, finite reference and valid rank/ridge settings".into());
    }
    let check = |x: &Array2<f64>, residual: &Array2<f64>, count: usize| {
        if x.dim() != (count, p)
            || residual.dim() != (count, k)
            || x.iter().chain(residual).any(|v| !v.is_finite())
        {
            Err(
                "streaming design/reference-residual tile has wrong shape or nonfinite values"
                    .to_string(),
            )
        } else {
            Ok(())
        }
    };
    let mut reduced = Array2::<f64>::zeros((0, p));
    let mut rhs = Array2::<f64>::zeros((0, k));
    let mut initial_norm = 0f64;
    let mut eliminated_norm = 0f64;
    let mut qr_dimension_sum = 0f64;
    let mut tiles = 0usize;
    for start in (0..rows).step_by(tile_rows) {
        let end = start.saturating_add(tile_rows).min(rows);
        let (x, residual) = read(start..end)?;
        check(&x, &residual, end - start)?;
        initial_norm = initial_norm.hypot(norm(&residual)?);
        let held = reduced.nrows();
        let height = held
            .checked_add(end - start)
            .ok_or("QR row count overflow")?;
        let mut stack = Array2::<f64>::zeros((height, p));
        stack.slice_mut(s![..held, ..]).assign(&reduced);
        stack.slice_mut(s![held.., ..]).assign(&x);
        let mut transformed = faer::Mat::from_fn(height, k, |i, j| {
            if i < held {
                rhs[[i, j]]
            } else {
                residual[[i - held, j]]
            }
        });
        let view = FaerArrayView::new(&stack);
        let qr = HouseholderQr::new(view.as_ref());
        qr.apply_transpose_on_the_left(transformed.as_mut());
        let retained_rows = height.min(p);
        reduced = Array2::from_shape_fn((retained_rows, p), |(i, j)| qr.r()[(i, j)]);
        rhs = Array2::from_shape_fn((retained_rows, k), |(i, j)| transformed[(i, j)]);
        for i in retained_rows..height {
            for j in 0..k {
                eliminated_norm = eliminated_norm.hypot(transformed[(i, j)]);
            }
        }
        if reduced.iter().chain(rhs.iter()).any(|v| !v.is_finite()) || !eliminated_norm.is_finite()
        {
            return Err("nonfinite streaming Householder reduction".into());
        }
        qr_dimension_sum += height.max(p) as f64;
        tiles += 1;
    }
    let factor = svd(reduced.view(), false).map_err(|e| e.to_string())?;
    if !factor.band.is_finite()
        || factor.band < 0.
        || factor
            .singular_values
            .iter()
            .any(|v| !v.is_finite() || *v < 0.)
        || factor
            .u
            .iter()
            .chain(factor.vt.iter())
            .any(|v| !v.is_finite())
    {
        return Err("nonfinite reduced coefficient-fit SVD".into());
    }
    let singular_values = factor.singular_values.to_vec();
    let largest = singular_values[0];
    // Include all reduction dimensions rather than pretending the small R's
    // SVD band describes the tall pipeline. This remains a numerical guard,
    // not a certificate for QR accumulation or numerical rank.
    let qr_guard = f64::EPSILON * qr_dimension_sum * largest;
    let cutoff = (relative * largest).max(factor.band + qr_guard);
    if !cutoff.is_finite() || !initial_norm.is_finite() {
        return Err("nonfinite streaming rank guard or initial residual".into());
    }
    let retained = singular_values
        .iter()
        .enumerate()
        .filter_map(|(i, sigma)| (*sigma > cutoff).then_some(i))
        .collect::<Vec<_>>();
    let mut coefficients = reference.clone();
    let mut floor = rhs.clone();
    if !retained.is_empty() {
        let u = factor.u.select(Axis(1), &retained);
        let vt = factor.vt.select(Axis(0), &retained);
        let mut projected = fast_atb(&u, &rhs);
        floor -= &fast_ab(&u, &projected);
        for (row, &index) in retained.iter().enumerate() {
            let sigma = singular_values[index];
            let denominator = sigma.hypot(settings.ridge.sqrt());
            let gain = (sigma / denominator) / denominator;
            if !gain.is_finite() {
                return Err("unresolved reduced reciprocal singular value".into());
            }
            projected.row_mut(row).mapv_inplace(|v| v * gain);
        }
        coefficients += &fast_atb(&vt, &projected);
    }
    if coefficients.iter().any(|v| !v.is_finite()) {
        return Err("nonfinite streaming fitted coefficients".into());
    }
    let change = &coefficients - reference;
    let mut final_norm = 0f64;
    for start in (0..rows).step_by(tile_rows) {
        let end = start.saturating_add(tile_rows).min(rows);
        let (x, residual) = read(start..end)?;
        check(&x, &residual, end - start)?;
        final_norm = final_norm.hypot(norm(&(residual - fast_ab(&x, &change)))?);
    }
    let condition = retained.last().and_then(|&i| {
        let value = largest / singular_values[i];
        value.is_finite().then_some(value)
    });
    Ok(Fit {
        coefficients,
        diagnostics: Diagnostics {
            rows,
            features: p,
            outputs: k,
            retained_rank: retained.len(),
            singular_values,
            relative_rank_cutoff: relative,
            absolute_rank_cutoff: cutoff,
            svd_rounding_band: factor.band,
            retained_condition_number: condition,
            ridge: settings.ridge,
            initial_residual_frobenius: initial_norm,
            final_residual_frobenius: final_norm,
            coefficient_change_frobenius: norm(&change)?,
            retained_subspace_residual_floor: eliminated_norm.hypot(norm(&floor)?),
            method: "streaming_householder_qr_reduced_svd",
            row_tiles: tiles,
            qr_rounding_guard: qr_guard,
            scope: "All-row f64 Householder reductions and reduced SVD around supplied reference; no normal equations, row sampling, or full-height singular vectors. Original row count sets default relative cutoff; an accumulated QR dimension guard augments the reduced SVD band. Rank/guards/floor are numerical diagnostics, not certificates, and rank near cutoff may differ from direct SVD. Eliminated RHS tail norm is accumulated directly, never by subtracting norm squares. Actual returned-coefficient residual is measured in a second all-row tiled pass. Ridge has no row averaging; discarded/null reference directions are preserved up to rounding.",
        },
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::array;
    fn close(a: &Array2<f64>, b: &Array2<f64>) {
        assert_eq!(a.dim(), b.dim());
        assert!(
            a.iter().zip(b).all(|(x, y)| (x - y).abs() < 1e-11),
            "{a:?} versus {b:?}"
        );
    }
    #[test]
    fn streaming_tall_rank_deficient_ridge_and_residual_match_direct_svd() {
        let x = Array2::from_shape_fn((97, 4), |(r, c)| {
            let a = (r as f64 * 0.13).sin();
            let b = (r as f64 * 0.21).cos();
            match c {
                0 => a,
                1 => b,
                2 => a + 2. * b,
                _ => 0.,
            }
        });
        let reference = array![[3., -1.], [5., 2.], [-2., 4.], [7., 8.]];
        let expected = array![[1., 3.], [-2., 1.], [0.5, -1.], [20., 30.]];
        let y = fast_ab(&x, &expected)
            + Array2::from_shape_fn((97, 2), |(r, c)| 0.03 * ((r * 7 + c) as f64).sin());
        for ridge in [0., 0.7] {
            let settings = Settings {
                relative_rank_cutoff: Some(1e-10),
                ridge,
            };
            let dense = fit_dense(&x, &y, &reference, settings.clone()).unwrap();
            for tile in [1, 7, 32] {
                let mut visits = vec![0usize; 97];
                let streamed =
                    fit_residual_streaming(97, &reference, settings.clone(), tile, |rows| {
                        for r in rows.clone() {
                            visits[r] += 1;
                        }
                        let design = x.slice(s![rows.clone(), ..]).to_owned();
                        let residual =
                            y.slice(s![rows, ..]).to_owned() - fast_ab(&design, &reference);
                        Ok((design, residual))
                    })
                    .unwrap();
                close(&streamed.coefficients, &dense.coefficients);
                assert!(
                    visits.iter().all(|v| *v == 2),
                    "every row must be reduced and replayed"
                );
                assert_eq!(streamed.diagnostics.retained_rank, 2);
                assert!(streamed.diagnostics.qr_rounding_guard > 0.);
                assert!(
                    (streamed.diagnostics.final_residual_frobenius
                        - dense.diagnostics.final_residual_frobenius)
                        .abs()
                        < 1e-10
                );
                assert!(
                    (streamed.diagnostics.retained_subspace_residual_floor
                        - dense.diagnostics.retained_subspace_residual_floor)
                        .abs()
                        < 1e-10
                );
                assert!(streamed.diagnostics.retained_subspace_residual_floor > 0.1);
                assert_eq!(
                    streamed.coefficients.row(3),
                    reference.row(3),
                    "unobserved native reference direction"
                );
            }
        }
    }

    #[test]
    fn streaming_cutoff_uses_original_rows_and_zero_design_keeps_reference() {
        let reference = array![[3., 7.], [5., 9.]];
        let solved = fit_residual_streaming(37, &reference, Settings::default(), 5, |rows| {
            Ok((
                Array2::zeros((rows.len(), 2)),
                Array2::ones((rows.len(), 2)),
            ))
        })
        .unwrap();
        assert_eq!(solved.coefficients, reference);
        assert_eq!(solved.diagnostics.relative_rank_cutoff, f64::EPSILON * 37.);
        assert_eq!(solved.diagnostics.retained_rank, 0);
        assert!((solved.diagnostics.final_residual_frobenius - 74f64.sqrt()).abs() < 1e-12);
        assert!((solved.diagnostics.retained_subspace_residual_floor - 74f64.sqrt()).abs() < 1e-12);
        assert!(
            fit_residual_streaming(10, &reference, Settings::default(), 3, |rows| Ok((
                Array2::zeros((rows.len(), 1)),
                Array2::ones((rows.len(), 2))
            )))
            .is_err()
        );
    }

    #[test]
    fn public_tall_fit_dispatches_without_full_height_singular_vectors() {
        let x = Array2::from_shape_fn(
            (1025, 2),
            |(r, c)| if c == 0 { 1. } else { r as f64 / 1025. },
        );
        let y = fast_ab(&x, &array![[2.], [-3.]]);
        let solved = fit(&x, &y, &array![[7.], [8.]], Settings::default()).unwrap();
        assert_eq!(
            solved.diagnostics.method,
            "streaming_householder_qr_reduced_svd"
        );
        assert!(solved.diagnostics.row_tiles > 1);
        close(&solved.coefficients, &array![[2.], [-3.]]);
    }
    #[test]
    fn exact_multiple_output_recovery() {
        let x = array![[1., 0.], [0., 2.], [1., 1.]];
        let expected = array![[2., -1.], [3., 4.]];
        let y = fast_ab(&x, &expected);
        let solved = fit(&x, &y, &array![[7., 8.], [-3., 2.]], Settings::default()).expect("fit");
        close(&solved.coefficients, &expected);
        assert_eq!(solved.diagnostics.retained_rank, 2);
        assert!(solved.diagnostics.final_residual_frobenius < 1e-11);
    }
    #[test]
    fn reference_nullspace_survives_rank_deficiency() {
        let x = array![[1., 1., 0.], [2., 2., 0.]];
        let solved = fit(
            &x,
            &array![[6.], [12.]],
            &array![[3.], [5.], [7.]],
            Settings::default(),
        )
        .expect("fit");
        close(&solved.coefficients, &array![[2.], [4.], [7.]]);
        assert_eq!(solved.diagnostics.retained_rank, 1);
        assert!((solved.coefficients[(0, 0)] - solved.coefficients[(1, 0)] + 2.).abs() < 1e-12);
        assert_eq!(solved.coefficients[(2, 0)], 7.);
    }
    #[test]
    fn inconsistent_targets_expose_residual_floor() {
        let solved = fit(
            &array![[1.], [1.]],
            &array![[1.], [3.]],
            &array![[7.]],
            Settings::default(),
        )
        .expect("fit");
        close(&solved.coefficients, &array![[2.]]);
        assert!((solved.diagnostics.final_residual_frobenius - 2f64.sqrt()).abs() < 1e-12);
        assert!((solved.diagnostics.retained_subspace_residual_floor - 2f64.sqrt()).abs() < 1e-12);
    }
    #[test]
    fn ridge_is_centered_on_reference_not_zero() {
        let solved = fit(
            &array![[2., 0.], [0., 3.]],
            &array![[8.], [18.]],
            &array![[1.], [2.]],
            Settings {
                relative_rank_cutoff: None,
                ridge: 4.,
            },
        )
        .expect("ridge");
        close(&solved.coefficients, &array![[2.5], [2. + 36. / 13.]]);
        assert!(solved.diagnostics.final_residual_frobenius > 0.);
        assert!(solved.diagnostics.retained_subspace_residual_floor < 1e-12);
    }
    #[test]
    fn discarded_modes_and_zero_design_keep_reference() {
        let solved = fit(
            &array![[1., 0.], [0., 1e-8]],
            &array![[3.], [9e-8]],
            &array![[1.], [7.]],
            Settings {
                relative_rank_cutoff: Some(1e-4),
                ridge: 0.,
            },
        )
        .expect("truncated");
        close(&solved.coefficients, &array![[3.], [7.]]);
        assert_eq!(solved.diagnostics.retained_rank, 1);
        assert!((solved.diagnostics.retained_subspace_residual_floor - 2e-8).abs() < 1e-15);
        let reference = array![[3., 7.], [5., 9.]];
        let zero = fit(
            &Array2::zeros((3, 2)),
            &Array2::ones((3, 2)),
            &reference,
            Settings::default(),
        )
        .expect("zero design");
        assert_eq!(zero.coefficients, reference);
        assert_eq!(zero.diagnostics.retained_rank, 0);
        assert_eq!(zero.diagnostics.retained_condition_number, None);
    }
    #[test]
    fn invalid_inputs_and_regularization_reject() {
        let x = array![[1.]];
        assert!(
            fit(
                &x,
                &x,
                &x,
                Settings {
                    relative_rank_cutoff: Some(1.),
                    ridge: 0.
                }
            )
            .is_err()
        );
        assert!(
            fit(
                &x,
                &x,
                &x,
                Settings {
                    relative_rank_cutoff: None,
                    ridge: -1.
                }
            )
            .is_err()
        );
        assert!(fit(&x, &array![[f64::NAN]], &x, Settings::default()).is_err());
        assert!(fit(&x, &x, &Array2::zeros((2, 1)), Settings::default()).is_err());
    }
}
