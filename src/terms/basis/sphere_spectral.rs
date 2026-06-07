//! Truncated spectral coefficient builders and Legendre-recurrence evaluation
//! for intrinsic S² (sphere) Wahba/pseudo-spline smooths.
//!
//! These three routines are pure scalar math (no dependency on the rest of the
//! basis machinery): two build the truncated per-degree coefficient array the
//! `s2_wahba_legendre_colmajor` GPU kernel uploads, and one evaluates the
//! corresponding zonal kernel `Σ_ℓ c_ℓ P_ℓ(cos γ)` via the same Legendre
//! 3-term recurrence the kernel runs, so CPU and GPU paths stay bit-aligned.

/// Build the truncated Sobolev coefficient array
/// `c_0 = 0`, `c_ℓ = (2ℓ+1) / (4π · [ℓ(ℓ+1)]^m)` for `ℓ = 1..=lmax`.
/// Returned vector has length `lmax + 1` with `result[ℓ] = c_ℓ`. The
/// GPU `s2_wahba_legendre_colmajor` kernel uploads exactly this array.
pub fn sobolev_s2_truncated_coefficients(lmax: usize, m: usize) -> Vec<f64> {
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

/// Build the truncated pseudo-spline coefficient array
/// `c_0 = 0`, `c_ℓ = 2 / (4π · Π_{k=1..m+1}(ℓ + k))` for `ℓ = 1..=lmax`.
pub fn pseudo_s2_truncated_coefficients(lmax: usize, m: usize) -> Vec<f64> {
    let four_pi = 4.0 * std::f64::consts::PI;
    let mut coeffs = vec![0.0_f64; lmax + 1];
    for ell in 1..=lmax {
        let l = ell as f64;
        let mut denom = 1.0_f64;
        for k in 1..=(m + 1) {
            denom *= l + k as f64;
        }
        coeffs[ell] = 2.0 / (four_pi * denom);
    }
    coeffs
}

/// Evaluate `Σ_{ℓ=0..=lmax} c_ℓ · P_ℓ(cos γ)` via the same Legendre
/// 3-term recurrence the GPU kernel runs. `coeffs[ℓ] = c_ℓ` with
/// `coeffs.len() = lmax + 1`. The recurrence is
/// `p_{ℓ+1} = ((2ℓ+1)·t·p_ℓ − ℓ·p_{ℓ−1}) / (ℓ + 1)`.
#[inline]
pub fn sphere_truncated_spectral_eval(cos_gamma: f64, coeffs: &[f64]) -> f64 {
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
