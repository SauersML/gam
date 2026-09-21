use ndarray::{Array1, ArrayView1};

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct SigmaJet1 {
    pub sigma: f64,
    pub d1: f64,
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub(crate) struct SigmaJet3 {
    pub sigma: f64,
    pub d1: f64,
    pub d2: f64,
    pub d3: f64,
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub(crate) struct SigmaJet4 {
    pub sigma: f64,
    pub d1: f64,
    pub d2: f64,
    pub d3: f64,
    pub d4: f64,
}

/// Exact exponential link on the native `f64` range.
///
/// This matches `exp(eta)` itself: values remain finite throughout the true
/// representable range, overflow to `+inf` only when `f64::exp` overflows, and
/// underflow to `0.0` only when `f64::exp` underflows.
#[inline]
pub fn safe_exp(eta: f64) -> f64 {
    eta.exp()
}

#[inline]
pub fn exp_sigma_jet1_scalar(eta: f64) -> SigmaJet1 {
    let sigma = safe_exp(eta);
    SigmaJet1 { sigma, d1: sigma }
}

#[inline]
pub fn exp_sigma_from_eta_scalar(eta: f64) -> f64 {
    safe_exp(eta)
}

/// Largest exponent argument whose `exp` is still finite in binary64.
///
/// `ln(f64::MAX) ≈ 709.782712893384`; this constant sits ~1e-11 below it so
/// `exp(EXP_SATURATION_MAX_ARG)` is guaranteed to round to a finite value
/// (≈ `f64::MAX · (1 − 1.3e-11)`). The inverse σ-link saturates only here —
/// at the representability boundary of the number format itself — so the
/// implemented link equals the mathematical `exp(-η)` for every argument
/// whose value is representable in `f64`, and the saturated value differs
/// from the true value's rounding by less than one part in 1e11.
///
/// (A former cap at +500 silently rewrote *finite* models: `exp(600)` ≈
/// 3.8e260 is perfectly representable but was returned as `exp(500)` ≈
/// 1.4e217, desynchronizing the likelihood value from the uncapped
/// gradient/Hessian algebra over the entire exponent band (500, 709.78].)
pub const EXP_SATURATION_MAX_ARG: f64 = 709.78271289338;

/// Overflow-safe `exp(-x)`: exact wherever `exp(-x)` is representable.
///
/// Saturates at `exp(EXP_SATURATION_MAX_ARG) ≈ f64::MAX` instead of
/// overflowing to `+inf` (which would poison downstream products with NaN
/// via `inf · 0`), and allows natural IEEE 754 underflow to `0.0` when `x`
/// is very positive because that is the mathematically correct limit.
///
/// The one-sided guard is critical: for `x = 701` the correct value is
/// `exp(-701) ≈ 5e-305` (essentially zero); a two-sided clamp would destroy
/// far-tail exact derivatives.
#[inline]
pub(crate) fn exp_neg_stable(x: f64) -> f64 {
    (-x).min(EXP_SATURATION_MAX_ARG).exp()
}

/// Inverse exp-link `1/σ = exp(-η)` with the one-sided representability
/// guard from `exp_neg_stable`: exact for every η whose `exp(-η)` fits in
/// `f64`, saturating near `f64::MAX` only past that boundary. Required by
/// every solver path that forms products like `t · exp(-η_ls)` — without the
/// guard, very negative η_ls produces `+inf`, which propagates as `NaN`
/// through subsequent multiplications and breaks the monotonicity floor /
/// penalty chain.
#[inline]
pub fn exp_sigma_inverse_from_eta_scalar(eta: f64) -> f64 {
    exp_neg_stable(eta)
}

/// Standardized survival threshold q0 = -eta_t · exp(-eta_ls) with log-space
/// overflow detection.
///
/// log|q0| = ln|eta_t| + (-eta_ls) is formed exactly (no argument cap), so
/// the result equals the mathematical product for every representable
/// magnitude; saturation to ±MAX happens only when |q0| genuinely exceeds
/// `f64::MAX` — the representability boundary of the number format, not an
/// arbitrary ceiling. When `exp(-eta_ls)` alone is unrepresentable but the
/// product is finite (|eta_t| tiny), or the inverse scale underflows while
/// the product remains representable (|eta_t| large), the magnitude is
/// evaluated in the log domain instead of through the rounded factor.
#[inline]
pub fn survival_q0_from_eta(eta_t: f64, eta_ls: f64) -> f64 {
    if eta_t == 0.0 {
        return 0.0;
    }
    let log_abs = eta_t.abs().ln() - eta_ls;
    if log_abs > EXP_SATURATION_MAX_ARG {
        return if eta_t > 0.0 { -f64::MAX } else { f64::MAX };
    }
    let inverse_scale = exp_sigma_inverse_from_eta_scalar(eta_ls);
    if -eta_ls > EXP_SATURATION_MAX_ARG || inverse_scale < f64::MIN_POSITIVE {
        let mag = log_abs.exp();
        return if eta_t > 0.0 { -mag } else { mag };
    }
    let q = -eta_t * inverse_scale;
    if q.is_finite() {
        q
    } else {
        // Roundoff at the very edge of the representable band can push the
        // direct product to ±inf even though log_abs cleared the check.
        if eta_t > 0.0 { -f64::MAX } else { f64::MAX }
    }
}

#[inline]
pub fn exp_sigma_eta_for_sigma_scalar(sigma: f64) -> f64 {
    assert!(
        sigma.is_finite(),
        "exp sigma inverse link requires finite sigma: sigma={sigma}"
    );
    assert!(
        sigma > 0.0,
        "exp sigma inverse link: sigma must be positive (got sigma={sigma})"
    );
    sigma.ln()
}

#[inline]
pub(crate) fn exp_sigma_jet3_scalar(eta: f64) -> SigmaJet3 {
    let jet = exp_sigma_jet4_scalar(eta);
    SigmaJet3 {
        sigma: jet.sigma,
        d1: jet.d1,
        d2: jet.d2,
        d3: jet.d3,
    }
}

pub fn exp_sigma_derivs_up_to_third(
    eta: ArrayView1<'_, f64>,
) -> (Array1<f64>, Array1<f64>, Array1<f64>, Array1<f64>) {
    let n = eta.len();
    let mut sigma = Array1::<f64>::uninit(n);
    let mut d1 = Array1::<f64>::uninit(n);
    let mut d2 = Array1::<f64>::uninit(n);
    let mut d3 = Array1::<f64>::uninit(n);
    for i in 0..n {
        let jet = exp_sigma_jet3_scalar(eta[i]);
        sigma[i].write(jet.sigma);
        d1[i].write(jet.d1);
        d2[i].write(jet.d2);
        d3[i].write(jet.d3);
    }
    // SAFETY: every slot in each length-`n` output is written exactly once by
    // the loop over `0..n` before `assume_init`.
    unsafe {
        (
            sigma.assume_init(),
            d1.assume_init(),
            d2.assume_init(),
            d3.assume_init(),
        )
    }
}

#[inline]
pub(crate) fn exp_sigma_jet4_scalar(eta: f64) -> SigmaJet4 {
    let sigma = safe_exp(eta);
    SigmaJet4 {
        sigma,
        d1: sigma,
        d2: sigma,
        d3: sigma,
        d4: sigma,
    }
}

#[inline]
pub fn exp_sigma_derivs_up_to_fourth_scalar(eta: f64) -> (f64, f64, f64, f64, f64) {
    let jet = exp_sigma_jet4_scalar(eta);
    (jet.sigma, jet.d1, jet.d2, jet.d3, jet.d4)
}

/// The location-scale noise link is σ = b + exp(η), with the lower bound `b`
/// derived from the response's measurement resolution.
///
/// # Why there is a bound at all
///
/// The Gaussian location-scale log-likelihood
///
///   ℓ = −½ Σ w_i (y_i−μ_i)²/σ_i² − Σ w_i log σ_i
///
/// has no maximum when σ can reach 0. If the mean model fits a subset of rows
/// exactly (tied responses in a small group, an interpolating smooth), shrinking
/// σ on those rows sends −log σ to +∞. With σ ≥ b > 0, each row's log-likelihood
/// is at most −log b, so the penalized objective is bounded for any finite data
/// and the working weight 1/σ² is at most 1/b².
///
/// # Where `b` comes from
///
/// A recorded response is only known to within its measurement resolution δ,
/// the grid the values were recorded on. Model the recorded value as the latent
/// value plus a rounding error that is uniform on (−δ/2, δ/2). That error
/// carries Sheppard's variance δ²/12 on top of any modelled noise, whatever the
/// covariates. So no standard deviation below δ/√12 is supported by the data:
///
///   b = δ / √12.
///
/// δ is estimated as the smallest positive gap between distinct recorded
/// responses among the rows that enter the likelihood
/// ([`gaussian_resolution_sigma_floor`]). The grid spacing divides every gap
/// between grid values, so this is the finest difference the data resolve:
///
/// * Rounded data (e.g. to 0.1) with any two neighbours one step apart give
///   δ equal to that step.
/// * Continuous data give a δ of order range/n², so the bound is effectively
///   absent: κ = dlogσ/dη = exp(η)/σ ≈ 1 wherever the data inform σ, and the fit
///   is the log-link fit.
///
/// Why this bound rarely binds: for a correctly specified model on data rounded
/// to δ, the residual variance is at least the quantization variance δ²/12, so
/// the maximum-likelihood σ exceeds b and exp(η) stays in the interior. It binds
/// only where the fit would otherwise claim more precision than the recording
/// resolution allows, which is exactly the degenerate direction the bound
/// exists to close.
///
/// # Scale equivariance
///
/// `fit_gaussian_location_scale_model` fits on y / s with s = sample_std(y), and
/// computes `b` from that standardized response. Under y → c·y, both δ and s
/// scale by c, so the dimensionless `b` is unchanged and σ̂_{c·y} = c·σ̂_y
/// exactly. Mapping back to raw units shifts the log-σ intercept by +ln(s),
/// which scales only the exp(η) term. The floor is therefore reconstructed
/// explicitly at s·b ([`logb_sigma_from_eta_scalar`] with `floor = s·b`), which
/// is δ_raw/√12 in the response's own units.
#[inline]
pub fn logb_sigma_jet1_scalar(floor: f64, eta: f64) -> SigmaJet1 {
    let s = safe_exp(eta);
    SigmaJet1 {
        sigma: floor + s,
        d1: s,
    }
}

/// σ = floor + exp(η) for the logb noise link.
///
/// `floor` is in the same units as σ: the fit-time floor b on the standardized
/// response, or s·b for raw response units (see [`logb_sigma_jet1_scalar`]).
#[inline]
pub fn logb_sigma_from_eta_scalar(floor: f64, eta: f64) -> f64 {
    floor + safe_exp(eta)
}

/// Lower bound b = δ/√12 on the location-scale σ, where δ is the smallest
/// positive gap between distinct responses on rows with positive weight.
///
/// The derivation is on [`logb_sigma_jet1_scalar`]. The result is in the units
/// of `y`, so the location-scale fit calls this on its standardized response.
/// A response with fewer than two distinct values on positively weighted rows
/// has no resolution to derive a bound from, and is refused.
pub fn gaussian_resolution_sigma_floor(
    y: ArrayView1<'_, f64>,
    weights: ArrayView1<'_, f64>,
) -> Result<f64, String> {
    if y.len() != weights.len() {
        return Err(format!(
            "Gaussian location-scale σ floor needs one weight per response: {} responses, {} weights",
            y.len(),
            weights.len()
        ));
    }
    let mut values: Vec<f64> = Vec::with_capacity(y.len());
    for (i, (&yi, &wi)) in y.iter().zip(weights.iter()).enumerate() {
        if !yi.is_finite() || !wi.is_finite() || wi < 0.0 {
            return Err(format!(
                "Gaussian location-scale σ floor needs finite responses and finite non-negative weights; row {i} has y={yi}, weight={wi}"
            ));
        }
        if wi > 0.0 {
            values.push(yi);
        }
    }
    values.sort_unstable_by(f64::total_cmp);
    let resolution = values
        .windows(2)
        .map(|pair| pair[1] - pair[0])
        .filter(|gap| *gap > 0.0)
        .min_by(f64::total_cmp)
        .ok_or_else(|| {
            "Gaussian location-scale σ floor: the positively weighted responses take fewer than two distinct values, so they carry no measurement resolution to bound σ by".to_string()
        })?;
    Ok(resolution / 12.0_f64.sqrt())
}

/// Posterior mean `E[σ]` of `σ = floor + exp(η)` when the log-σ predictor has
/// the Gaussian posterior `η ~ N(mean, variance)`: the lognormal first moment
/// gives exactly `floor + exp(mean + variance/2)`. At `variance = 0` this is
/// the plug-in `logb_sigma_from_eta_with_floor_scalar`.
#[inline]
pub fn logb_sigma_posterior_mean_with_floor_scalar(floor: f64, mean: f64, variance: f64) -> f64 {
    floor + safe_exp(mean + 0.5 * variance)
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::fs;
    use std::path::Path;

    #[test]
    fn survival_threshold_preserves_product_when_inverse_scale_underflows() {
        for eta_ls in [740.0_f64, 800.0] {
            let eta_t = 700.0_f64.exp();
            let expected = (700.0 - eta_ls).exp();
            let threshold = survival_q0_from_eta(eta_t, eta_ls);
            assert!(threshold < 0.0);
            assert!((threshold / -expected - 1.0).abs() < 1.0e-12);
            assert_eq!(survival_q0_from_eta(-eta_t, eta_ls), -threshold);
        }
    }

    fn collect_rs_files(dir: &Path, out: &mut Vec<std::path::PathBuf>) {
        let Ok(entries) = fs::read_dir(dir) else {
            return;
        };
        for entry in entries.flatten() {
            let path = entry.path();
            if path.is_dir() {
                collect_rs_files(&path, out);
                continue;
            }
            if path.extension().and_then(|e| e.to_str()) == Some("rs") {
                out.push(path);
            }
        }
    }

    fn stripwhitespace(s: &str) -> String {
        s.chars().filter(|c| !c.is_whitespace()).collect()
    }

    #[test]
    fn forbid_bounded_sigma_link_pattern_in_source() {
        let root = Path::new(env!("CARGO_MANIFEST_DIR")).join("src");
        let mut files = Vec::new();
        collect_rs_files(&root, &mut files);

        // This module (`sigma_link.rs`) is the canonical home of the σ-link
        // implementation and the guard itself: the forbidden strings appear
        // here verbatim in `bad_patterns`, so scanning our own source would
        // always self-trip. Skip exactly this file — every *other* file under
        // `src/` is still checked.
        let self_file = root.join("sigma_link.rs");

        let bad_patterns = [
            "bounded_sigma",
            "model.sigma_min",
            "model.sigma_max",
            "payload.sigma_min",
            "payload.sigma_max",
            "survival_sigma_min",
            "survival_sigma_max",
            "fnsafe_sigma_from_eta(",
            "fnsigma_and_deriv_from_eta(",
            "fnsigma_from_eta_scalar(",
        ];

        for file in files {
            if file == self_file {
                continue;
            }
            let Ok(content) = fs::read_to_string(&file) else {
                continue;
            };
            let compact = stripwhitespace(&content);
            for pat in bad_patterns {
                assert!(
                    !compact.contains(pat),
                    "forbidden sigma link pattern '{pat}' found in {}",
                    file.display()
                );
            }
        }
    }

    #[test]
    fn exp_sigma_inverse_accepts_positive_sigma() {
        let eta = exp_sigma_eta_for_sigma_scalar(2.5);
        assert!(eta.is_finite());
        assert!((eta - 2.5_f64.ln()).abs() < 1e-12);
    }

    #[test]
    #[should_panic(expected = "sigma must be positive")]
    fn exp_sigma_inverse_rejects_non_positive_sigma() {
        exp_sigma_eta_for_sigma_scalar(0.0);
    }

    #[test]
    fn safe_exp_matches_native_exp_semantics() {
        assert!(safe_exp(0.0).is_finite());
        assert!(safe_exp(700.0).is_finite());
        assert!(safe_exp(-700.0).is_finite());
        assert!(safe_exp(1000.0).is_infinite());
        assert_eq!(safe_exp(-1000.0), 0.0);
        assert!(safe_exp(f64::MAX).is_infinite());
        assert_eq!(safe_exp(f64::MIN), 0.0);
        assert!((safe_exp(1.0) - 1.0_f64.exp()).abs() < 1e-15);
        assert!((safe_exp(-5.0) - (-5.0_f64).exp()).abs() < 1e-15);
    }

    #[test]
    fn exp_sigma_derivatives_match_exact_exp_in_far_tails() {
        for &eta in &[709.0, -745.0] {
            let (sigma, d1, d2, d3, d4) = exp_sigma_derivs_up_to_fourth_scalar(eta);
            assert_eq!(sigma, eta.exp());
            assert_eq!(d1, sigma);
            assert_eq!(d2, sigma);
            assert_eq!(d3, sigma);
            assert_eq!(d4, sigma);
        }
    }

    #[test]
    fn logb_sigma_floor_bounds_below_for_arbitrarily_negative_eta() {
        let floor = 0.25;
        for &eta in &[-1000.0, -100.0, -50.0, -10.0] {
            let sigma = logb_sigma_from_eta_scalar(floor, eta);
            assert!(sigma >= floor);
            assert!(sigma.is_finite());
            let inv_s2 = (sigma * sigma).recip();
            assert!(inv_s2 <= floor.powi(-2) + 1e-12);
        }
    }

    #[test]
    fn resolution_sigma_floor_is_sheppard_bound_of_the_recording_grid() {
        // Values recorded to a 0.5 grid, in shuffled order, with ties, and one
        // zero-weight row off the grid that must not enter the resolution.
        let y = ndarray::array![2.0, 0.5, 3.5, 2.0, 1.0, 0.5, 1.2345];
        let w = ndarray::array![1.0, 2.0, 1.0, 1.0, 0.5, 1.0, 0.0];
        let floor = gaussian_resolution_sigma_floor(y.view(), w.view()).expect("resolution");
        assert!((floor - 0.5 / 12.0_f64.sqrt()).abs() <= 1e-15);
    }

    #[test]
    fn resolution_sigma_floor_scales_with_the_response() {
        let y = ndarray::array![0.3, 1.7, 0.9, 2.4, 1.1];
        let w = ndarray::Array1::<f64>::ones(y.len());
        let base = gaussian_resolution_sigma_floor(y.view(), w.view()).expect("resolution");
        for &c in &[1e-4, 1e4] {
            let scaled = y.mapv(|v| c * v);
            let floor = gaussian_resolution_sigma_floor(scaled.view(), w.view()).expect("resolution");
            assert!((floor / (c * base) - 1.0).abs() <= 1e-12, "c={c}: {floor} vs {}", c * base);
        }
    }

    #[test]
    fn resolution_sigma_floor_refuses_a_response_without_two_distinct_values() {
        let y = ndarray::array![1.0, 1.0, 1.0, 4.0];
        let w = ndarray::array![1.0, 2.0, 1.0, 0.0];
        let err = gaussian_resolution_sigma_floor(y.view(), w.view()).expect_err("no resolution");
        assert!(err.contains("fewer than two distinct values"), "{err}");
    }

    #[test]
    fn logb_sigma_recovers_exp_link_in_upper_regime() {
        for &eta in &[3.0, 5.0, 10.0] {
            let logb = logb_sigma_from_eta_scalar(0.1, eta);
            let pure_exp = exp_sigma_from_eta_scalar(eta);
            let rel_err = (logb - pure_exp).abs() / pure_exp;
            assert!(rel_err < 1e-2);
        }
    }

}
