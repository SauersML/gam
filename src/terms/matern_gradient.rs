//! Streaming closed-form gradients for Matérn radial basis values.
//!
//! This is the lightweight public primitive for composition-engine callers
//! that need `dK/dtheta` without finite differences or a full smooth-term
//! build. It streams row chunks over `(data, centers)` and supports the global
//! log-kappa coordinate plus per-axis anisotropic log-scale coordinates.

use ndarray::{Array1, Array2, ArrayView1, ArrayView2, s};
use rayon::prelude::*;

use crate::terms::basis::{BasisError, MaternNu};

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum MaternBasisGradientTarget {
    LogKappa,
    AnisoLogScale(usize),
}

#[derive(Debug, Clone)]
pub struct StreamingMaternBasisGradientEvaluator {
    centers: Array2<f64>,
    length_scale: f64,
    nu: MaternNu,
    metric_weights: Vec<f64>,
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
            return Err(BasisError::InvalidInput(
                "StreamingMaternBasisGradientEvaluator requires centers with at least one column"
                    .to_string(),
            ));
        }
        if centers.iter().any(|v| !v.is_finite()) {
            return Err(BasisError::InvalidInput(
                "StreamingMaternBasisGradientEvaluator centers must be finite".to_string(),
            ));
        }
        if !(length_scale.is_finite() && length_scale > 0.0) {
            return Err(BasisError::InvalidInput(format!(
                "StreamingMaternBasisGradientEvaluator length_scale must be finite and positive; got {length_scale}"
            )));
        }
        let metric_weights = match aniso_log_scales {
            Some(eta) => {
                if eta.len() != centers.ncols() {
                    return Err(BasisError::DimensionMismatch(format!(
                        "aniso_log_scales length {} != center dimension {}",
                        eta.len(),
                        centers.ncols()
                    )));
                }
                eta.iter()
                    .enumerate()
                    .map(|(axis, value)| {
                        if !value.is_finite() {
                            Err(BasisError::InvalidInput(format!(
                                "aniso_log_scales[{axis}] must be finite"
                            )))
                        } else {
                            Ok((2.0 * value).exp())
                        }
                    })
                    .collect::<Result<Vec<_>, _>>()?
            }
            None => vec![1.0; centers.ncols()],
        };
        Ok(Self {
            centers: centers.as_standard_layout().to_owned(),
            length_scale,
            nu,
            metric_weights,
            chunk_size: chunk_size.unwrap_or(2048).max(1),
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
            return Err(BasisError::InvalidInput(format!(
                "Matérn gradient row chunk {start}..{end} is outside data with {} rows",
                data.nrows()
            )));
        }
        if let MaternBasisGradientTarget::AnisoLogScale(axis) = target
            && axis >= self.dimension()
        {
            return Err(BasisError::InvalidInput(format!(
                "Matérn anisotropic gradient axis {axis} out of bounds for dimension {}",
                self.dimension()
            )));
        }

        let chunk_n = end - start;
        let k = self.n_centers();
        let dim = self.dimension();
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
                    let mut r2 = 0.0;
                    let mut axis_component = 0.0;
                    for axis in 0..dim {
                        let h = data[[global, axis]] - c[axis];
                        let component = self.metric_weights[axis] * h * h;
                        r2 += component;
                        if target == MaternBasisGradientTarget::AnisoLogScale(axis) {
                            axis_component = component;
                        }
                    }
                    let d_log_kappa =
                        matern_log_kappa_derivative(r2.sqrt(), self.length_scale, self.nu);
                    row[center_idx] = match target {
                        MaternBasisGradientTarget::LogKappa => d_log_kappa,
                        MaternBasisGradientTarget::AnisoLogScale(_) => {
                            if r2 == 0.0 {
                                0.0
                            } else {
                                d_log_kappa * axis_component / r2
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

    pub fn forward_mul(
        &self,
        data: ArrayView2<'_, f64>,
        target: MaternBasisGradientTarget,
        coeffs: ArrayView1<'_, f64>,
    ) -> Result<Array1<f64>, BasisError> {
        self.validate_data(data)?;
        if coeffs.len() != self.n_centers() {
            return Err(BasisError::DimensionMismatch(format!(
                "Matérn gradient coeff length {} != centers {}",
                coeffs.len(),
                self.n_centers()
            )));
        }
        let mut out = Array1::<f64>::zeros(data.nrows());
        for start in (0..data.nrows()).step_by(self.chunk_size) {
            let end = (start + self.chunk_size).min(data.nrows());
            let chunk = self.row_chunk_gradient(data, start, end, target)?;
            out.slice_mut(s![start..end]).assign(&chunk.dot(&coeffs));
        }
        Ok(out)
    }

    fn validate_data(&self, data: ArrayView2<'_, f64>) -> Result<(), BasisError> {
        if data.ncols() != self.dimension() {
            return Err(BasisError::DimensionMismatch(format!(
                "Matérn gradient data dimension {} != center dimension {}",
                data.ncols(),
                self.dimension()
            )));
        }
        if data.iter().any(|v| !v.is_finite()) {
            return Err(BasisError::InvalidInput(
                "Matérn gradient data must be finite".to_string(),
            ));
        }
        Ok(())
    }
}

fn stable_poly_exp(a: f64, coeffs: &[f64]) -> f64 {
    if a > 745.0 {
        return 0.0;
    }
    let mut poly = 0.0;
    for &coeff in coeffs.iter().rev() {
        poly = poly * a + coeff;
    }
    poly * (-a).exp()
}

fn matern_log_kappa_derivative(r: f64, length_scale: f64, nu: MaternNu) -> f64 {
    let x = r / length_scale;
    match nu {
        MaternNu::Half => stable_poly_exp(x, &[0.0, -1.0]),
        MaternNu::ThreeHalves => stable_poly_exp(3.0_f64.sqrt() * x, &[0.0, 0.0, -1.0]),
        MaternNu::FiveHalves => {
            stable_poly_exp(5.0_f64.sqrt() * x, &[0.0, 0.0, -1.0 / 3.0, -1.0 / 3.0])
        }
        MaternNu::SevenHalves => stable_poly_exp(
            7.0_f64.sqrt() * x,
            &[0.0, 0.0, -1.0 / 5.0, -1.0 / 5.0, -1.0 / 15.0],
        ),
        MaternNu::NineHalves => stable_poly_exp(
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
            MaternNu::Half => stable_poly_exp(x, &[1.0]),
            MaternNu::ThreeHalves => stable_poly_exp(3.0_f64.sqrt() * x, &[1.0, 1.0]),
            MaternNu::FiveHalves => stable_poly_exp(5.0_f64.sqrt() * x, &[1.0, 1.0, 1.0 / 3.0]),
            MaternNu::SevenHalves => {
                stable_poly_exp(7.0_f64.sqrt() * x, &[1.0, 1.0, 2.0 / 5.0, 1.0 / 15.0])
            }
            MaternNu::NineHalves => stable_poly_exp(
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
                    let weights = [(2.0 * eta[0]).exp(), (2.0 * axis_eta).exp()];
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
    fn forward_mul_matches_materialized_dot() {
        let data = array![[0.1], [0.3], [0.8]];
        let centers = array![[0.0], [0.5]];
        let coeffs = array![2.0, -0.25];
        let eval = StreamingMaternBasisGradientEvaluator::new(
            centers.view(),
            1.1,
            MaternNu::NineHalves,
            None,
            Some(2),
        )
        .unwrap();
        let dense = eval
            .evaluate(data.view(), MaternBasisGradientTarget::LogKappa)
            .unwrap();
        let streaming = eval
            .forward_mul(
                data.view(),
                MaternBasisGradientTarget::LogKappa,
                coeffs.view(),
            )
            .unwrap();
        assert_eq!(streaming, dense.dot(&coeffs));
    }
}
