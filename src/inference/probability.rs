use crate::types::LikelihoodFamily;
use ndarray::{Array1, ArrayView1};

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
    let poly = (((((1.330_274_429 * t - 1.821_255_978) * t) + 1.781_477_937) * t - 0.356_563_782)
        * t
        + 0.319_381_530)
        * t;
    let cdf_pos = 1.0 - normal_pdf(z) * poly;
    if x >= 0.0 { cdf_pos } else { 1.0 - cdf_pos }
}

/// Standard normal quantile Φ⁻¹(p) using Acklam's rational approximation.
#[inline]
pub fn standard_normal_quantile(p: f64) -> Result<f64, String> {
    if !(p.is_finite() && p > 0.0 && p < 1.0) {
        return Err(format!("normal quantile requires p in (0,1), got {p}"));
    }

    const A: [f64; 6] = [
        -3.969_683_028_665_376e1,
        2.209_460_984_245_205e2,
        -2.759_285_104_469_687e2,
        1.383_577_518_672_69e2,
        -3.066_479_806_614_716e1,
        2.506_628_277_459_239,
    ];
    const B: [f64; 5] = [
        -5.447_609_879_822_406e1,
        1.615_858_368_580_409e2,
        -1.556_989_798_598_866e2,
        6.680_131_188_771_972e1,
        -1.328_068_155_288_572e1,
    ];
    const C: [f64; 6] = [
        -7.784_894_002_430_293e-3,
        -3.223_964_580_411_365e-1,
        -2.400_758_277_161_838,
        -2.549_732_539_343_734,
        4.374_664_141_464_968,
        2.938_163_982_698_783,
    ];
    const D: [f64; 4] = [
        7.784_695_709_041_462e-3,
        3.224_671_290_700_398e-1,
        2.445_134_137_142_996,
        3.754_408_661_907_416,
    ];
    const P_LOW: f64 = 0.02425;
    const P_HIGH: f64 = 1.0 - P_LOW;

    let x = if p < P_LOW {
        let q = (-2.0 * p.ln()).sqrt();
        (((((C[0] * q + C[1]) * q + C[2]) * q + C[3]) * q + C[4]) * q + C[5])
            / ((((D[0] * q + D[1]) * q + D[2]) * q + D[3]) * q + 1.0)
    } else if p <= P_HIGH {
        let q = p - 0.5;
        let r = q * q;
        (((((A[0] * r + A[1]) * r + A[2]) * r + A[3]) * r + A[4]) * r + A[5]) * q
            / (((((B[0] * r + B[1]) * r + B[2]) * r + B[3]) * r + B[4]) * r + 1.0)
    } else {
        let q = (-2.0 * (1.0 - p).ln()).sqrt();
        -(((((C[0] * q + C[1]) * q + C[2]) * q + C[3]) * q + C[4]) * q + C[5])
            / ((((D[0] * q + D[1]) * q + D[2]) * q + D[3]) * q + 1.0)
    };
    Ok(x)
}

/// Inverse-link transform per likelihood family.
#[inline]
pub fn inverse_link_array(family: LikelihoodFamily, eta: ArrayView1<'_, f64>) -> Array1<f64> {
    match family {
        LikelihoodFamily::GaussianIdentity => eta.to_owned(),
        LikelihoodFamily::BinomialLogit => eta.mapv(|v| {
            let z = v.clamp(-30.0, 30.0);
            1.0 / (1.0 + (-z).exp())
        }),
        LikelihoodFamily::BinomialProbit => eta.mapv(normal_cdf_approx),
        LikelihoodFamily::BinomialCLogLog => eta.mapv(|v| {
            let z = v.clamp(-30.0, 30.0);
            1.0 - (-(z.exp())).exp()
        }),
        LikelihoodFamily::RoystonParmar => eta.to_owned(),
    }
}
