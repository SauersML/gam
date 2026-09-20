use ndarray::ArrayView2;
use serde::{Deserialize, Serialize};

use gam_math::quantile::quantile_from_sorted;

type MatrixBands = (Vec<f64>, Vec<f64>, Vec<f64>);
pub(crate) type PosteriorBands = (Vec<f64>, Vec<f64>, Vec<f64>, Vec<f64>, Vec<f64>, Vec<f64>);

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct PosteriorPredictBandsPayload {
    pub linear_predictor: Vec<f64>,
    pub linear_predictor_lower: Vec<f64>,
    pub linear_predictor_upper: Vec<f64>,
    pub mean: Vec<f64>,
    pub mean_lower: Vec<f64>,
    pub mean_upper: Vec<f64>,
    pub n_rows: usize,
    pub n_draws: usize,
    pub model_class: String,
    pub family_kind: String,
}

fn validate_band_request(
    draws: ArrayView2<'_, f64>,
    level: f64,
    scale: &'static str,
    allow_infinite: bool,
) -> Result<(), String> {
    if !(level > 0.0 && level < 1.0) {
        return Err(format!("interval level must lie in (0, 1); got {level}"));
    }
    if draws.nrows() == 0 {
        return Err("posterior bands unavailable: zero draws".to_string());
    }
    if let Some(((draw, row), value)) = draws.indexed_iter().find(|(_, value)| {
        value.is_nan() || (value.is_infinite() && (!allow_infinite || value.is_sign_negative()))
    }) {
        return Err(format!(
            "posterior {scale} draw is invalid at draw {draw}, row {row}: {value}"
        ));
    }
    Ok(())
}

fn matrix_bands(
    draws: ArrayView2<'_, f64>,
    level: f64,
    scale: &'static str,
    allow_infinite: bool,
) -> Result<MatrixBands, String> {
    validate_band_request(draws, level, scale, allow_infinite)?;
    let alpha = (1.0 - level) / 2.0;
    let n_draws = draws.nrows();
    let n_rows = draws.ncols();
    let inv_n = 1.0 / n_draws as f64;
    let mut centre = vec![0.0_f64; n_rows];
    let mut lower = vec![0.0_f64; n_rows];
    let mut upper = vec![0.0_f64; n_rows];
    let mut column = vec![0.0_f64; n_draws];
    for row in 0..n_rows {
        for draw in 0..n_draws {
            column[draw] = draws[[draw, row]];
        }
        centre[row] = column.iter().sum::<f64>() * inv_n;
        // `validate_band_request` has already rejected NaN draws, so the IEEE
        // total order coincides with the numeric order here.
        column.sort_by(f64::total_cmp);
        lower[row] = quantile_from_sorted(&column, alpha);
        upper[row] = quantile_from_sorted(&column, 1.0 - alpha);
    }
    Ok((centre, lower, upper))
}

/// Collapse canonical-predictor and response-scale posterior draw matrices to
/// per-row posterior means and credible bands.
///
/// Model-specific posterior prediction must supply both matrices.  A generic
/// inverse-link push-through is not valid for location-scale or
/// transformation-normal models, whose response mean is produced by their
/// typed prediction kernel rather than by `g^-1(eta)`.  Keeping this reducer
/// agnostic to model class makes those kernels the sole owners of response
/// semantics while still giving every frontend one interval implementation.
pub fn draw_bands_from_matrices(
    eta: ArrayView2<'_, f64>,
    mean: ArrayView2<'_, f64>,
    level: f64,
) -> Result<PosteriorBands, String> {
    if eta.dim() != mean.dim() {
        return Err(format!(
            "posterior draw matrix shape mismatch: eta is {:?}, mean is {:?}",
            eta.dim(),
            mean.dim()
        ));
    }
    let (eta_mean, eta_lower, eta_upper) = matrix_bands(eta, level, "linear-predictor", false)?;
    // Exact response transforms (notably exp) may correctly overflow to +inf;
    // preserve that IEEE result while still rejecting unordered NaNs.
    let (mean, mean_lower, mean_upper) = matrix_bands(mean, level, "response", true)?;
    Ok((eta_mean, eta_lower, eta_upper, mean, mean_lower, mean_upper))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn model_specific_mean_draws_are_summarized_without_inverse_link_reconstruction() {
        let eta = Array2::from_shape_vec((3, 2), vec![-2.0, 4.0, 0.0, 2.0, 3.0, -1.0])
            .expect("eta shape");
        // Deliberately not a scalar transform of eta: this is the contract used
        // by location-scale and transformation-normal prediction kernels.
        let mean = Array2::from_shape_vec((3, 2), vec![10.0, 1.0, 14.0, 8.0, 30.0, 3.0])
            .expect("mean shape");
        let (eta_centre, _, _, mean_centre, mean_lower, mean_upper) =
            draw_bands_from_matrices(eta.view(), mean.view(), 0.5).expect("draw bands");
        let inv_three = 1.0 / 3.0;
        assert_eq!(eta_centre, vec![inv_three, 5.0 * inv_three]);
        assert_eq!(mean_centre, vec![18.0, 4.0]);
        assert!(mean_lower[0] >= 10.0 && mean_upper[0] <= 30.0);
        assert!(mean_lower[1] >= 1.0 && mean_upper[1] <= 8.0);
    }

    #[test]
    fn posterior_draw_matrices_must_have_identical_shapes() {
        let eta = Array2::<f64>::zeros((2, 3));
        let mean = Array2::<f64>::zeros((2, 2));
        let error = draw_bands_from_matrices(eta.view(), mean.view(), 0.95)
            .expect_err("shape mismatch must fail");
        assert!(error.contains("shape mismatch"));
    }

    #[test]
    fn response_positive_infinity_is_preserved_without_nan_quantiles() {
        let eta = Array2::from_shape_vec((2, 1), vec![700.0, 710.0]).expect("eta shape");
        let mean = Array2::from_shape_vec((2, 1), vec![700.0_f64.exp(), f64::INFINITY])
            .expect("mean shape");
        let (_, _, _, centre, lower, upper) =
            draw_bands_from_matrices(eta.view(), mean.view(), 0.5).expect("draw bands");
        assert_eq!(centre, vec![f64::INFINITY]);
        assert_eq!(lower, vec![f64::INFINITY]);
        assert_eq!(upper, vec![f64::INFINITY]);
    }

    #[test]
    fn invalid_extended_draw_values_are_rejected_by_scale() {
        let finite = Array2::from_shape_vec((1, 1), vec![0.0]).expect("finite shape");
        for invalid_eta in [f64::INFINITY, f64::NEG_INFINITY, f64::NAN] {
            let eta = Array2::from_shape_vec((1, 1), vec![invalid_eta]).expect("eta shape");
            assert!(draw_bands_from_matrices(eta.view(), finite.view(), 0.95).is_err());
        }
        for invalid_mean in [f64::NEG_INFINITY, f64::NAN] {
            let mean = Array2::from_shape_vec((1, 1), vec![invalid_mean]).expect("mean shape");
            assert!(draw_bands_from_matrices(finite.view(), mean.view(), 0.95).is_err());
        }
    }
    use ndarray::Array2;

}
