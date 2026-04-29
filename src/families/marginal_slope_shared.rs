#[inline]
pub(crate) fn eval_coeff4_at(coefficients: &[f64; 4], z: f64) -> f64 {
    ((coefficients[3] * z + coefficients[2]) * z + coefficients[1]) * z + coefficients[0]
}

#[inline]
pub(crate) fn add_scaled_coeff4(target: &mut [f64; 4], source: &[f64; 4], scale: f64) {
    for j in 0..4 {
        target[j] += scale * source[j];
    }
}

#[inline]
pub(crate) fn scale_coeff4(source: [f64; 4], scale: f64) -> [f64; 4] {
    [
        source[0] * scale,
        source[1] * scale,
        source[2] * scale,
        source[3] * scale,
    ]
}

pub(crate) fn probit_frailty_scale(gaussian_frailty_sd: Option<f64>) -> f64 {
    let sigma = gaussian_frailty_sd.unwrap_or(0.0);
    if sigma <= 0.0 {
        1.0
    } else {
        crate::families::lognormal_kernel::ProbitFrailtyScaleJet::from_log_sigma(sigma.ln()).s
    }
}
