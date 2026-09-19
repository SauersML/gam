//! Truncated spectral coefficient builders and Legendre-recurrence evaluation
//! for intrinsic S² (sphere) Wahba smooths.
//!
//! These routines are pure scalar math (no dependency on the rest of the
//! basis machinery): one builds the truncated per-degree coefficient array the
//! `s2_wahba_legendre_colmajor` GPU kernel uploads, and one evaluates the
//! corresponding zonal kernel `Σ_ℓ c_ℓ P_ℓ(cos γ)` via the same Legendre
//! 3-term recurrence the kernel runs, so CPU and GPU paths stay bit-aligned.

/// Build the truncated Sobolev coefficient array
/// `c_0 = 0`, `c_ℓ = (2ℓ+1) / (4π · [ℓ(ℓ+1)]^m)` for `ℓ = 1..=lmax`.
/// Returned vector has length `lmax + 1` with `result[ℓ] = c_ℓ`. The
/// GPU `s2_wahba_legendre_colmajor` kernel uploads exactly this array.
pub(crate) fn sobolev_s2_truncated_coefficients(lmax: usize, m: usize) -> Vec<f64> {
    let four_pi = 4.0 * std::f64::consts::PI;
    let mut coeffs = vec![0.0_f64; lmax + 1];
    let mi = m as i32;
    for ell in 1..=lmax {
        let l = ell as f64;
        let eigen = (l * (l + 1.0)).powi(mi);
        coeffs[ell] = (2.0 * l + 1.0) / (four_pi * eigen);
    }
    coeffs
}

/// Evaluate `Σ_{ℓ=0..=lmax} c_ℓ · P_ℓ(cos γ)` via the same Legendre
/// 3-term recurrence the GPU kernel runs. `coeffs[ℓ] = c_ℓ` with
/// `coeffs.len() = lmax + 1`. The recurrence is
/// `p_{ℓ+1} = ((2ℓ+1)·t·p_ℓ − ℓ·p_{ℓ−1}) / (ℓ + 1)`.
#[inline]
pub(crate) fn sphere_truncated_spectral_eval(cos_gamma: f64, coeffs: &[f64]) -> f64 {
    let t = cos_gamma.clamp(-1.0, 1.0);
    let lmax = coeffs.len().saturating_sub(1);
    if lmax == 0 {
        return coeffs.first().copied().unwrap_or(0.0);
    }
    // p_{ℓ-1}, p_ℓ
    let mut p_prev = 1.0_f64; // P_0(t)
    let mut p_curr = t; // P_1(t)
    let mut acc = coeffs[0] * p_prev + coeffs[1] * p_curr;
    for ell in 1..lmax {
        let lf = ell as f64;
        let p_next = ((2.0 * lf + 1.0) * t * p_curr - lf * p_prev) / (lf + 1.0);
        acc += coeffs[ell + 1] * p_next;
        p_prev = p_curr;
        p_curr = p_next;
    }
    acc
}

/// Exact derivative `d/d(cos γ) [ Σ_ℓ c_ℓ P_ℓ(cos γ) ]` of
/// [`sphere_truncated_spectral_eval`]. Uses the Legendre derivative identity
/// `(1 − x²) P_ℓ'(x) = ℓ (P_{ℓ−1}(x) − x P_ℓ(x))`, advancing `P_ℓ` with the same
/// 3-term recurrence, with the closed-form pole limit `P_ℓ'(±1) = ±^(ℓ-1)
/// ℓ(ℓ+1)/2` substituted near `|cos γ| = 1` where the `(1 − x²)` denominator
/// would otherwise lose precision.
pub(crate) fn sphere_truncated_spectral_derivative_eval(cos_gamma: f64, coeffs: &[f64]) -> f64 {
    let x = cos_gamma.clamp(-1.0, 1.0);
    let lmax = coeffs.len().saturating_sub(1);
    if lmax == 0 {
        return 0.0;
    }
    // ONE sweep over the closed interval, poles included. `P'_ℓ` comes from the
    // recurrence that has no pole,
    //
    // ```text
    //   P'_ℓ(x) = (2ℓ - 1)·P_{ℓ-1}(x) + P'_{ℓ-2}(x),    P'_0 = 0, P'_1 = 1
    // ```
    //
    // replacing `P'_ℓ = ℓ(P_{ℓ-1} - x·P_ℓ)/(1 - x²)`, which is a removable
    // `0/0` at `x = ±1` and therefore needed both an `f64::EPSILON` floor on
    // the denominator and a separate `|x| > 1 - 1e-10` pole branch. Its
    // numerator subtracts two `O(1)` Legendre values to leave `O((ℓ+1)(1-x))`,
    // so it decays like `ε/(1 - x²)` and had already fallen to eight digits by
    // the time the threshold caught it — measured on this kernel at `lmax=64`,
    // `m=2` against a 40-digit reference: `7.4e-16` at `|x| = 0.99`, `7.0e-10`
    // at `1 - 1e-8`, `6.5e-8` at `1 - 1e-10`, where the recurrence below holds
    // `2e-15` throughout and reproduces the closed pole value
    // `P'_ℓ(±1) = (±1)^{ℓ+1}·ℓ(ℓ+1)/2` to `5e-16` at the poles themselves.
    //
    // The deleted pole branch was not wrong — this is a FINITE sum, so summing
    // `Σ c_ℓ P'_ℓ(±1)` in closed form is exact. It was a second route to a
    // number this one already produces, and its existence is what let the
    // primary route's accuracy go unmeasured.
    let mut p_prev = 1.0_f64; // P_{ℓ-2}, seeded at P_0
    let mut p_curr = x; // P_{ℓ-1}, seeded at P_1
    let mut d_prev = 0.0_f64; // P'_{ℓ-2}, seeded at P'_0
    let mut d_curr = 1.0_f64; // P'_{ℓ-1}, seeded at P'_1
    let mut acc = coeffs[1] * d_curr;
    for ell in 2..=lmax {
        let lf = ell as f64;
        let two_l_minus_1 = 2.0 * lf - 1.0;
        let d_next = two_l_minus_1 * p_curr + d_prev;
        let p_next = (two_l_minus_1 * x * p_curr - (lf - 1.0) * p_prev) / lf;
        acc += coeffs[ell] * d_next;
        p_prev = p_curr;
        p_curr = p_next;
        d_prev = d_curr;
        d_curr = d_next;
    }
    acc
}

/// Exact second derivative `d²/d(cos γ)² [ Σ_ℓ c_ℓ P_ℓ(cos γ) ]` of
/// [`sphere_truncated_spectral_eval`], from the recurrence obtained by
/// differentiating the `P'` recurrence of
/// [`sphere_truncated_spectral_derivative_eval`] once more,
///
/// ```text
///   P''_ℓ(x) = (2ℓ - 1)·P'_{ℓ-1}(x) + P''_{ℓ-2}(x),    P''_0 = P''_1 = 0
/// ```
///
/// so one sweep carries `P`, `P'` and `P''` over the closed interval, poles
/// included.
pub(crate) fn sphere_truncated_spectral_second_derivative_eval(cos_gamma: f64, coeffs: &[f64]) -> f64 {
    let x = cos_gamma.clamp(-1.0, 1.0);
    let lmax = coeffs.len().saturating_sub(1);
    let mut p_prev = 1.0_f64; // P_{ℓ-2}, seeded at P_0
    let mut p_curr = x; // P_{ℓ-1}, seeded at P_1
    let mut d_prev = 0.0_f64; // P'_{ℓ-2}, seeded at P'_0
    let mut d_curr = 1.0_f64; // P'_{ℓ-1}, seeded at P'_1
    let mut s_prev = 0.0_f64; // P''_{ℓ-2}, seeded at P''_0
    let mut s_curr = 0.0_f64; // P''_{ℓ-1}, seeded at P''_1
    let mut acc = 0.0_f64;
    for ell in 2..=lmax {
        let lf = ell as f64;
        let two_l_minus_1 = 2.0 * lf - 1.0;
        let s_next = two_l_minus_1 * d_curr + s_prev;
        let d_next = two_l_minus_1 * p_curr + d_prev;
        let p_next = (two_l_minus_1 * x * p_curr - (lf - 1.0) * p_prev) / lf;
        acc += coeffs[ell] * s_next;
        p_prev = p_curr;
        p_curr = p_next;
        d_prev = d_curr;
        d_curr = d_next;
        s_prev = s_curr;
        s_curr = s_next;
    }
    acc
}
