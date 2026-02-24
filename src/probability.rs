/// Standard normal PDF φ(x).
#[inline]
pub fn normal_pdf(x: f64) -> f64 {
    const INV_SQRT_2PI: f64 = 0.398_942_280_401_432_7;
    INV_SQRT_2PI * (-0.5 * x * x).exp()
}

/// Standard normal CDF Φ(x) using a stable Abramowitz-Stegun-style approximation.
#[inline]
pub fn normal_cdf_approx(x: f64) -> f64 {
    let z = x.abs().clamp(0.0, 30.0);
    let t = 1.0 / (1.0 + 0.231_641_9 * z);
    let poly = (((((1.330_274_429 * t - 1.821_255_978) * t) + 1.781_477_937) * t
        - 0.356_563_782)
        * t
        + 0.319_381_530)
        * t;
    let cdf_pos = 1.0 - normal_pdf(z) * poly;
    if x >= 0.0 { cdf_pos } else { 1.0 - cdf_pos }
}
