//! Streaming closed-form gradients for Matérn radial basis values.
//!
//! This is the lightweight public primitive for composition-engine callers
//! that need `dK/dtheta` without finite differences or a full smooth-term
//! build. It streams row chunks over `(data, centers)` and supports the global
//! log-kappa coordinate plus per-axis anisotropic log-scale coordinates.

use gam_math::special::stable_polynomial_times_exp_neg;
use ndarray::{Array2, ArrayView2, s};
use rayon::prelude::*;

use crate::basis::duchon_kernel_math::{
    centered_aniso_log_scale_mean, centered_aniso_metric_weights,
};
use crate::basis::{BasisError, MaternNu};

/// Default row-chunk size for streaming the `(data × centers)` distance scan.
/// Chosen so a chunk's working set (`chunk × k_centers` f64) stays in L2 for
/// typical center counts while keeping rayon task granularity coarse.
const DEFAULT_ROW_CHUNK: usize = 2048;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum MaternBasisGradientTarget {
    LogKappa,
    /// Derivative through the centered, clamped metric coordinates. At an
    /// exact centered contrast of ±50 the metric is nondifferentiable; use
    /// its saturated-side derivative (zero for that metric coordinate).
    AnisoLogScale(usize),
}

#[derive(Debug, Clone)]
pub struct StreamingMaternBasisGradientEvaluator {
    centers: Array2<f64>,
    length_scale: f64,
    nu: MaternNu,
    metric_axis_scales: Vec<f64>,
    metric_axis_derivatives: Vec<bool>,
    chunk_size: usize,
}

impl StreamingMaternBasisGradientEvaluator {
    pub fn new(
        centers: ArrayView2<'_, f64>,
        length_scale: f64,
        nu: MaternNu,
        aniso_log_scales: Option<&[f64]>,
        chunk_size: Option<usize>,
    ) -> Result<Self, BasisError> {
        if centers.ncols() == 0 {
            crate::bail_invalid_basis!(
                "StreamingMaternBasisGradientEvaluator requires centers with at least one column"
                    .to_string(),
            );
        }
        if centers.iter().any(|v| !v.is_finite()) {
            crate::bail_invalid_basis!(
                "StreamingMaternBasisGradientEvaluator centers must be finite"
            );
        }
        if !(length_scale.is_finite() && length_scale > 0.0) {
            crate::bail_invalid_basis!(
                "StreamingMaternBasisGradientEvaluator length_scale must be finite and positive; got {length_scale}"
            );
        }
        let (metric_weights, metric_axis_derivatives) = match aniso_log_scales {
            Some(eta) => {
                if eta.len() != centers.ncols() {
                    crate::bail_dim_basis!(
                        "aniso_log_scales length {} != center dimension {}",
                        eta.len(),
                        centers.ncols()
                    );
                }
                for (axis, value) in eta.iter().enumerate() {
                    if !value.is_finite() {
                        return Err(BasisError::InvalidInput(format!(
                            "aniso_log_scales[{axis}] must be finite"
                        )));
                    }
                }
                let mean = centered_aniso_log_scale_mean(eta);
                (
                    centered_aniso_metric_weights(eta),
                    eta.iter()
                        .map(|&value| {
                            let centered = value - mean;
                            centered > -50.0 && centered < 50.0
                        })
                        .collect(),
                )
            }
            None => (vec![1.0; centers.ncols()], vec![true; centers.ncols()]),
        };
        Ok(Self {
            centers: centers.as_standard_layout().to_owned(),
            length_scale,
            nu,
            metric_axis_scales: metric_weights.into_iter().map(f64::sqrt).collect(),
            metric_axis_derivatives,
            chunk_size: chunk_size.unwrap_or(DEFAULT_ROW_CHUNK).max(1),
        })
    }

    pub fn n_centers(&self) -> usize {
        self.centers.nrows()
    }

    pub fn dimension(&self) -> usize {
        self.centers.ncols()
    }

    pub fn row_chunk_gradient(
        &self,
        data: ArrayView2<'_, f64>,
        start: usize,
        end: usize,
        target: MaternBasisGradientTarget,
    ) -> Result<Array2<f64>, BasisError> {
        self.validate_data(data)?;
        if start > end || end > data.nrows() {
            crate::bail_invalid_basis!(
                "Matérn gradient row chunk {start}..{end} is outside data with {} rows",
                data.nrows()
            );
        }
        if let MaternBasisGradientTarget::AnisoLogScale(axis) = target
            && axis >= self.dimension()
        {
            crate::bail_invalid_basis!(
                "Matérn anisotropic gradient axis {axis} out of bounds for dimension {}",
                self.dimension()
            );
        }

        let chunk_n = end - start;
        let k = self.n_centers();
        let dim = self.dimension();
        let all_axes_active = self.metric_axis_derivatives.iter().all(|&active| active);
        // The shared metric deliberately leaves a one-axis eta uncentered.
        let mean_derivative = if dim > 1 { 1.0 / dim as f64 } else { 0.0 };
        if chunk_n == 0 || k == 0 {
            return Ok(Array2::zeros((chunk_n, k)));
        }
        let centers = self
            .centers
            .as_slice()
            .expect("standard-layout Matérn gradient centers");
        let mut values = vec![0.0_f64; chunk_n * k];
        values
            .par_chunks_mut(k)
            .enumerate()
            .for_each(|(local, row)| {
                let global = start + local;
                for center_idx in 0..k {
                    let c = &centers[center_idx * dim..(center_idx + 1) * dim];
                    let mut distance = 0.0_f64;
                    let mut active_distance = 0.0_f64;
                    let mut axis_component = 0.0;
                    for axis in 0..dim {
                        let component = dimensionless_metric_displacement(
                            data[[global, axis]],
                            c[axis],
                            self.length_scale,
                            self.metric_axis_scales[axis],
                        );
                        distance = distance.hypot(component);
                        if !all_axes_active && self.metric_axis_derivatives[axis] {
                            active_distance = active_distance.hypot(component);
                        }
                        if target == MaternBasisGradientTarget::AnisoLogScale(axis) {
                            axis_component = component;
                        }
                    }
                    let d_log_kappa = matern_log_kappa_derivative(distance, self.nu);
                    row[center_idx] = match target {
                        MaternBasisGradientTarget::LogKappa => d_log_kappa,
                        MaternBasisGradientTarget::AnisoLogScale(axis) => {
                            if d_log_kappa == 0.0 {
                                0.0
                            } else {
                                let axis_fraction = if self.metric_axis_derivatives[axis] {
                                    (axis_component / distance).powi(2)
                                } else {
                                    0.0
                                };
                                let active_fraction = if all_axes_active {
                                    1.0
                                } else {
                                    (active_distance / distance).powi(2)
                                };
                                d_log_kappa * (axis_fraction - mean_derivative * active_fraction)
                            }
                        }
                    };
                }
            });
        Array2::from_shape_vec((chunk_n, k), values).map_err(|err| {
            BasisError::InvalidInput(format!("Matérn gradient chunk shape failed: {err}"))
        })
    }

    pub fn evaluate(
        &self,
        data: ArrayView2<'_, f64>,
        target: MaternBasisGradientTarget,
    ) -> Result<Array2<f64>, BasisError> {
        self.validate_data(data)?;
        let mut out = Array2::<f64>::zeros((data.nrows(), self.n_centers()));
        for start in (0..data.nrows()).step_by(self.chunk_size) {
            let end = (start + self.chunk_size).min(data.nrows());
            let chunk = self.row_chunk_gradient(data, start, end, target)?;
            out.slice_mut(s![start..end, ..]).assign(&chunk);
        }
        Ok(out)
    }

    fn validate_data(&self, data: ArrayView2<'_, f64>) -> Result<(), BasisError> {
        if data.ncols() != self.dimension() {
            crate::bail_dim_basis!(
                "Matérn gradient data dimension {} != center dimension {}",
                data.ncols(),
                self.dimension()
            );
        }
        if data.iter().any(|v| !v.is_finite()) {
            crate::bail_invalid_basis!("Matérn gradient data must be finite");
        }
        Ok::<(), _>(())
    }
}

/// Form the dimensionless displacement before any squares. The two product
/// orders cover ratios outside the float range that the metric scale brings
/// back inside it; halving protects opposite-sign coordinate subtraction.
fn dimensionless_metric_displacement(
    value: f64,
    center: f64,
    length: f64,
    axis_scale: f64,
) -> f64 {
    let delta = value - center;
    let (delta, axis_scale) = if delta.is_finite() {
        (delta, axis_scale)
    } else {
        (0.5 * value - 0.5 * center, 2.0 * axis_scale)
    };
    let ratio = delta / length;
    if ratio.is_normal() {
        ratio * axis_scale
    } else {
        (delta * axis_scale) / length
    }
}

fn matern_log_kappa_derivative(x: f64, nu: MaternNu) -> f64 {
    match nu {
        MaternNu::Half => stable_polynomial_times_exp_neg(x, &[0.0, -1.0]),
        MaternNu::ThreeHalves => {
            stable_polynomial_times_exp_neg(3.0_f64.sqrt() * x, &[0.0, 0.0, -1.0])
        }
        MaternNu::FiveHalves => stable_polynomial_times_exp_neg(
            5.0_f64.sqrt() * x,
            &[0.0, 0.0, -1.0 / 3.0, -1.0 / 3.0],
        ),
        MaternNu::SevenHalves => stable_polynomial_times_exp_neg(
            7.0_f64.sqrt() * x,
            &[0.0, 0.0, -1.0 / 5.0, -1.0 / 5.0, -1.0 / 15.0],
        ),
        MaternNu::NineHalves => stable_polynomial_times_exp_neg(
            9.0_f64.sqrt() * x,
            &[0.0, 0.0, -1.0 / 7.0, -1.0 / 7.0, -2.0 / 35.0, -1.0 / 105.0],
        ),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::array;

    fn matern_value_from_distance(r: f64, length_scale: f64, nu: MaternNu) -> f64 {
        let x = r / length_scale;
        match nu {
            MaternNu::Half => stable_polynomial_times_exp_neg(x, &[1.0]),
            MaternNu::ThreeHalves => {
                stable_polynomial_times_exp_neg(3.0_f64.sqrt() * x, &[1.0, 1.0])
            }
            MaternNu::FiveHalves => {
                stable_polynomial_times_exp_neg(5.0_f64.sqrt() * x, &[1.0, 1.0, 1.0 / 3.0])
            }
            MaternNu::SevenHalves => stable_polynomial_times_exp_neg(
                7.0_f64.sqrt() * x,
                &[1.0, 1.0, 2.0 / 5.0, 1.0 / 15.0],
            ),
            MaternNu::NineHalves => stable_polynomial_times_exp_neg(
                9.0_f64.sqrt() * x,
                &[1.0, 1.0, 3.0 / 7.0, 2.0 / 21.0, 1.0 / 105.0],
            ),
        }
    }

    #[test]
    fn log_kappa_gradient_matches_finite_difference() {
        let data = array![[0.1, 0.2], [1.0, -0.3], [0.4, 0.8]];
        let centers = array![[0.0, 0.0], [0.8, 0.5]];
        let length_scale = 1.3;
        let eval = StreamingMaternBasisGradientEvaluator::new(
            centers.view(),
            length_scale,
            MaternNu::FiveHalves,
            None,
            Some(2),
        )
        .unwrap();
        let analytic = eval
            .evaluate(data.view(), MaternBasisGradientTarget::LogKappa)
            .unwrap();
        let h: f64 = 1.0e-5;
        for i in 0..data.nrows() {
            for j in 0..centers.nrows() {
                let r = ((0..data.ncols())
                    .map(|axis| {
                        let d = data[[i, axis]] - centers[[j, axis]];
                        d * d
                    })
                    .sum::<f64>())
                .sqrt();
                let plus =
                    matern_value_from_distance(r, length_scale * (-h).exp(), MaternNu::FiveHalves);
                let minus =
                    matern_value_from_distance(r, length_scale * h.exp(), MaternNu::FiveHalves);
                let fd = (plus - minus) / (2.0 * h);
                assert!((analytic[[i, j]] - fd).abs() < 1.0e-8);
            }
        }
    }

    #[test]
    fn anisotropic_axis_gradient_matches_finite_difference() {
        let data = array![[0.2, -0.1], [1.1, 0.7]];
        let centers = array![[0.0, 0.0], [0.6, 0.4], [1.0, -0.2]];
        let eta = [0.2_f64, -0.2];
        let eval = StreamingMaternBasisGradientEvaluator::new(
            centers.view(),
            0.9,
            MaternNu::ThreeHalves,
            Some(&eta),
            Some(1),
        )
        .unwrap();
        let analytic = eval
            .evaluate(data.view(), MaternBasisGradientTarget::AnisoLogScale(1))
            .unwrap();
        let h = 1.0e-5;
        for i in 0..data.nrows() {
            for j in 0..centers.nrows() {
                let value_at = |axis_eta: f64| {
                    let eta_trial = [eta[0], axis_eta];
                    let weights = centered_aniso_metric_weights(&eta_trial);
                    let r = ((0..2)
                        .map(|axis| {
                            let d = data[[i, axis]] - centers[[j, axis]];
                            weights[axis] * d * d
                        })
                        .sum::<f64>())
                    .sqrt();
                    matern_value_from_distance(r, 0.9, MaternNu::ThreeHalves)
                };
                let fd = (value_at(eta[1] + h) - value_at(eta[1] - h)) / (2.0 * h);
                assert!((analytic[[i, j]] - fd).abs() < 1.0e-8);
            }
        }
    }

    #[test]
    fn matern_gradient_retains_polynomial_weighted_exponential_tails() {
        let a = 746.0_f64;
        let derivative = matern_log_kappa_derivative(a / 3.0, MaternNu::NineHalves);
        let expected_log = 5.0 * a.ln() - a - 105.0_f64.ln()
            + (6.0 / a + 15.0 / a.powi(2) + 15.0 / a.powi(3)).ln_1p();
        assert!(derivative < 0.0);
        assert!(((-derivative).ln() - expected_log).abs() < 1e-8);
    }

    #[test]
    fn streaming_matern_gradients_are_invariant_to_coordinate_units() {
        let data = array![[0.2, -0.1], [1.1, 0.7]];
        let centers = array![[0.0, 0.0], [0.6, 0.4]];
        let evaluate = |scale: f64, target| {
            let centers = &centers * scale;
            let data = &data * scale;
            StreamingMaternBasisGradientEvaluator::new(
                centers.view(),
                0.9 * scale,
                MaternNu::ThreeHalves,
                Some(&[0.2, -0.2]),
                Some(1),
            )
            .expect("scaled evaluator")
            .evaluate(data.view(), target)
            .expect("scaled gradient")
        };
        for target in [
            MaternBasisGradientTarget::LogKappa,
            MaternBasisGradientTarget::AnisoLogScale(1),
        ] {
            let expected = evaluate(1.0, target);
            for scale in [1e-200, 1e200] {
                let actual = evaluate(scale, target);
                for (&actual, &expected) in actual.iter().zip(&expected) {
                    assert!((actual - expected).abs() < 1e-14);
                }
            }
        }
    }

    #[test]
    fn dimensionless_distance_handles_extreme_intermediate_ratios() {
        assert!(
            (dimensionless_metric_displacement(1e-125, 0.0, 1e200, 1e20) / 1e-305 - 1.0).abs()
                < 1e-14
        );
        assert!(
            (dimensionless_metric_displacement(1e300, 0.0, 1e-20, 1e-22) / 1e298 - 1.0).abs()
                < 1e-14
        );
        assert_eq!(
            dimensionless_metric_displacement(f64::MAX, -f64::MAX, f64::MAX, 1.0),
            2.0
        );
    }

    #[test]
    fn anisotropic_gradient_differentiates_the_clamped_metric() {
        let eta = [80.0_f64, 0.0, 0.0];
        let data = array![[(-50.0_f64).exp(), (80.0_f64 / 3.0).exp(), 0.0]];
        let centers = array![[0.0, 0.0, 0.0]];
        let evaluator = StreamingMaternBasisGradientEvaluator::new(
            centers.view(), 1.0, MaternNu::ThreeHalves, Some(&eta), None,
        )
        .unwrap();
        let h = 1e-5;
        for axis in 0..3 {
            let value_at = |step| {
                let mut trial = eta;
                trial[axis] += step;
                let weights = centered_aniso_metric_weights(&trial);
                let distance = (0..3)
                    .map(|j| weights[j] * data[[0, j]] * data[[0, j]])
                    .sum::<f64>()
                    .sqrt();
                matern_value_from_distance(distance, 1.0, MaternNu::ThreeHalves)
            };
            let fd = (value_at(h) - value_at(-h)) / (2.0 * h);
            let actual = evaluator
                .evaluate(data.view(), MaternBasisGradientTarget::AnisoLogScale(axis))
                .unwrap()[[0, 0]];
            assert!((actual - fd).abs() < 1e-8, "axis {axis}: {actual} vs {fd}");
        }
    }

    #[test]
    fn one_axis_anisotropic_gradient_matches_the_uncentered_metric() {
        let data = array![[0.7]];
        let centers = array![[0.0]];
        let evaluator = StreamingMaternBasisGradientEvaluator::new(
            centers.view(), 1.0, MaternNu::Half, Some(&[0.2]), None,
        )
        .unwrap();
        let actual = evaluator
            .evaluate(data.view(), MaternBasisGradientTarget::AnisoLogScale(0))
            .unwrap()[[0, 0]];
        let distance = 0.7 * 0.2_f64.exp();
        let expected = -distance * (-distance).exp();
        assert!((actual - expected).abs() < 1e-14);
    }

}
