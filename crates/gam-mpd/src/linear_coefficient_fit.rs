//! Fixed-design coefficient fitting around an explicit reference, using a thin SVD.
//! This solves a numerical subproblem; it does not discover the design or certify
//! generalization. No normal-equation inverse or silent f32 quantization is used.
use gam_linalg::{
    decompose::svd,
    faer_ndarray::{fast_ab, fast_atb},
};
use ndarray::{Array2, Axis};
use serde::{Deserialize, Serialize};

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
            scope: "Fixed-design f64 SVD solve around supplied reference. Rank and residuals are numerical diagnostics, not exact rank or roundoff certificates. Ridge objective has no row averaging. Updates are restricted to retained singular modes; all discarded reference directions are preserved up to arithmetic rounding.",
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
