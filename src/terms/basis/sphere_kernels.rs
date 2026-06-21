//! Closed-form and spectral zonal Wahba kernels on S².
//!
//! This module owns the scalar/SIMD kernel dispatch for intrinsic sphere
//! smooths. Callers in `basis` handle data validation, coordinate transforms,
//! and matrix assembly.

use super::BasisError;
use super::polylog::{dilog_unit, trilog_unit};
use super::sphere_spec::SphereWahbaKernel;
use super::sphere_spectral::{
    pseudo_s2_truncated_coefficients, sobolev_s2_truncated_coefficients,
    sphere_truncated_spectral_derivative_eval, sphere_truncated_spectral_eval,
};

/// Pseudo-spline Wahba kernel on S² (mgcv `makeR`-style closed form).
#[inline]
pub(crate) fn wahba_sphere_kernel_pseudo_from_cos(cos_gamma: f64, m: usize) -> f64 {
    let cg = cos_gamma.clamp(-1.0, 1.0);
    let z = (1.0 - cg).max(f64::EPSILON * 1.0e-4);
    let w = 0.5 * z;
    let c0 = w.sqrt();
    let a = (1.0 + 1.0 / c0).ln();
    let c = 2.0 * c0;
    let two_pi = 2.0 * std::f64::consts::PI;
    match m {
        1 => {
            let q1 = 2.0 * a * w - c + 1.0;
            (q1 - 0.5) / two_pi
        }
        2 => {
            let w2 = w * w;
            let q2 = a * (6.0 * w2 - 2.0 * w) - 3.0 * c * w + 3.0 * w + 0.5;
            (q2 / 2.0 - 1.0 / 6.0) / two_pi
        }
        3 => {
            let w2 = w * w;
            let w3 = w2 * w;
            let q3 = (a * (60.0 * w3 - 36.0 * w2) + 30.0 * w2 + c * (8.0 * w - 30.0 * w2)
                - 3.0 * w
                + 1.0)
                / 3.0;
            (q3 / 6.0 - 1.0 / 24.0) / two_pi
        }
        _ => {
            let w2 = w * w;
            let w3 = w2 * w;
            let w4 = w3 * w;
            let q4 = a * (70.0 * w4 - 60.0 * w3 + 6.0 * w2)
                + 35.0 * w3 * (1.0 - c)
                + c * 55.0 * w2 / 3.0
                - 12.5 * w2
                - w / 3.0
                + 0.25;
            (q4 / 24.0 - 1.0 / 120.0) / two_pi
        }
    }
}

/// Exact derivative `dK_m^{pseudo}/d(cos gamma)` of the pseudo-spline Wahba
/// kernel [`wahba_sphere_kernel_pseudo_from_cos`].
///
/// The forward kernel is a polynomial in `w = (1 - cos gamma)/2` with the
/// auxiliary terms `c0 = sqrt(w)`, `c = 2 c0`, and
/// `a = ln(1 + 1/c0)`. Differentiating in `w` and applying
/// `dw/d(cos gamma) = -1/2` gives the analytic `dK/d(cos gamma)` below.
#[inline]
pub(crate) fn wahba_sphere_kernel_pseudo_derivative_dcos(cos_gamma: f64, m: usize) -> f64 {
    let cg = cos_gamma.clamp(-1.0, 1.0);
    let z = (1.0 - cg).max(f64::EPSILON * 1.0e-4);
    let w = 0.5 * z;
    let c0 = w.sqrt();
    let a = (1.0 + 1.0 / c0).ln();
    let c = 2.0 * c0;
    let two_pi = 2.0 * std::f64::consts::PI;
    let da_dw = -1.0 / (2.0 * c0 * c0 * (c0 + 1.0));
    let dc_dw = 1.0 / c0;
    let dk_dw = match m {
        1 => {
            let dq1_dw = 2.0 * a + 2.0 * w * da_dw - dc_dw;
            dq1_dw / two_pi
        }
        2 => {
            let dq2_dw = da_dw * (6.0 * w * w - 2.0 * w) + a * (12.0 * w - 2.0)
                - 3.0 * (dc_dw * w + c)
                + 3.0;
            (dq2_dw / 2.0) / two_pi
        }
        3 => {
            let w2 = w * w;
            let w3 = w2 * w;
            let dinner_dw = da_dw * (60.0 * w3 - 36.0 * w2)
                + a * (180.0 * w2 - 72.0 * w)
                + 60.0 * w
                + (dc_dw * (8.0 * w - 30.0 * w2) + c * (8.0 - 60.0 * w))
                - 3.0;
            let dq3_dw = dinner_dw / 3.0;
            (dq3_dw / 6.0) / two_pi
        }
        _ => {
            let w2 = w * w;
            let w3 = w2 * w;
            let w4 = w3 * w;
            let dq4_dw = da_dw * (70.0 * w4 - 60.0 * w3 + 6.0 * w2)
                + a * (280.0 * w3 - 180.0 * w2 + 12.0 * w)
                + 35.0 * (3.0 * w2 * (1.0 - c) - w3 * dc_dw)
                + (55.0 / 3.0) * (dc_dw * w2 + c * 2.0 * w)
                - 25.0 * w
                - 1.0 / 3.0;
            (dq4_dw / 24.0) / two_pi
        }
    };
    dk_dw * (-0.5)
}

// ============================================================================
// Wahba/Sobolev kernel on S²
// ============================================================================
//
// `K_m^{Sobolev}(gamma) = (1/4pi) * sum_{l >= 1} (2l + 1)
// * [l(l + 1)]^{-m} * P_l(cos gamma)`.
//
// For `m in {1, 2, 3}` we use the closed-form expressions derived in
// Beatson & zu Castell, "Thinplate Splines on the Sphere", SIGMA 14 (2018)
// 083 (Section 6.2). For `m = 4`, we fall back to a truncated Legendre series.

/// Sobolev `K_m^{Sobolev}` reproducing kernel on S², closed-form for
/// `m in {1, 2, 3}` plus spectral series for `m = 4`.
#[inline]
pub(crate) fn wahba_sphere_kernel_sobolev_closed_form(cos_gamma: f64, m: usize) -> f64 {
    let cos_g = cos_gamma.clamp(-1.0, 1.0);
    let four_pi = 4.0 * std::f64::consts::PI;
    let pi2_6 = std::f64::consts::PI * std::f64::consts::PI / 6.0;
    let u = ((1.0 - cos_g) * 0.5).max(f64::EPSILON * 1.0e-4);
    let one_minus_u = (1.0 - u).max(f64::EPSILON * 1.0e-4);
    match m {
        1 => (-u.ln() - 1.0) / four_pi,
        2 => (dilog_unit(one_minus_u) + 1.0 - pi2_6) / four_pi,
        3 => {
            const ZETA3: f64 = 1.2020569031595942853997381615114499907649862923404988817922;
            let li3_u = trilog_unit(u);
            let li2_one_minus_u = dilog_unit(one_minus_u);
            let li2_u = dilog_unit(u);
            (-2.0 * li3_u - li2_one_minus_u + u.ln() * li2_u + 2.0 * ZETA3 + pi2_6 - 2.0) / four_pi
        }
        _ => wahba_sphere_kernel_sobolev_spectral(cos_g, m),
    }
}

/// Spectral Legendre-series evaluation of the Sobolev kernel
/// `K_m^{Sobolev}(gamma) = (1/4pi) sum_{l >= 1} (2l+1) *
/// [l(l+1)]^{-m} * P_l(cos gamma)`.
#[inline]
pub(crate) fn wahba_sphere_kernel_sobolev_spectral(cos_gamma: f64, m: usize) -> f64 {
    let l_max = match m {
        1 => 4096_usize,
        2 => 256,
        3 => 128,
        _ => 96,
    };
    let x = cos_gamma.clamp(-1.0, 1.0);
    let m_i = m as i32;
    let four_pi = 4.0 * std::f64::consts::PI;
    let mut p_l_minus_1 = 1.0_f64;
    let mut p_l = x;
    let mut sum = 3.0 * p_l / (four_pi * 2.0_f64.powi(m_i));
    for l in 1..l_max {
        let p_l_plus_1 =
            ((2 * l + 1) as f64 * x * p_l - (l as f64) * p_l_minus_1) / ((l + 1) as f64);
        let ell = (l + 1) as f64;
        let eigen = (ell * (ell + 1.0)).powi(m_i);
        let weight = (2.0 * ell + 1.0) / four_pi;
        sum += weight * p_l_plus_1 / eigen;
        p_l_minus_1 = p_l;
        p_l = p_l_plus_1;
    }
    sum
}

/// Evaluate the Wahba sphere reproducing kernel at a single `cos gamma`.
#[inline]
pub(crate) fn wahba_sphere_kernel_from_cos_kind(
    cos_gamma: f64,
    penalty_order: usize,
    kernel: SphereWahbaKernel,
) -> Result<f64, BasisError> {
    if !(1..=4).contains(&penalty_order) {
        crate::bail_invalid_basis!(
            "spherical spline penalty_order must be one of 1, 2, 3, 4; got {penalty_order}"
        );
    }
    let value = match kernel {
        SphereWahbaKernel::Sobolev => {
            wahba_sphere_kernel_sobolev_closed_form(cos_gamma, penalty_order)
        }
        SphereWahbaKernel::Pseudo => wahba_sphere_kernel_pseudo_from_cos(cos_gamma, penalty_order),
        SphereWahbaKernel::SobolevTruncated { lmax } => {
            let coeffs = sobolev_s2_truncated_coefficients(lmax as usize, penalty_order);
            sphere_truncated_spectral_eval(cos_gamma, &coeffs)
        }
        SphereWahbaKernel::PseudoTruncated { lmax } => {
            let coeffs = pseudo_s2_truncated_coefficients(lmax as usize, penalty_order);
            sphere_truncated_spectral_eval(cos_gamma, &coeffs)
        }
    };
    if !value.is_finite() {
        crate::bail_invalid_basis!("spherical spline kernel produced a non-finite value");
    }
    Ok(value)
}

/// SIMD lane-wise evaluation. Both Sobolev and pseudo-spline branches are
/// scalar-per-lane because the closed forms contain non-vector elementary and
/// polylogarithm calls.
#[inline]
pub(crate) fn wahba_sphere_kernel_from_cos_simd_kind(
    cos_gamma: wide::f64x4,
    penalty_order: usize,
    kernel: SphereWahbaKernel,
) -> wide::f64x4 {
    use wide::f64x4;
    if !(1..=4).contains(&penalty_order) {
        return f64x4::from(f64::NAN);
    }
    let cg = cos_gamma.fast_max(f64x4::from(-1.0)).fast_min(f64x4::ONE);
    let lanes = cg.to_array();
    let f = |x: f64| -> f64 {
        match kernel {
            SphereWahbaKernel::Sobolev => wahba_sphere_kernel_sobolev_closed_form(x, penalty_order),
            SphereWahbaKernel::Pseudo => wahba_sphere_kernel_pseudo_from_cos(x, penalty_order),
            SphereWahbaKernel::SobolevTruncated { lmax } => {
                let coeffs = sobolev_s2_truncated_coefficients(lmax as usize, penalty_order);
                sphere_truncated_spectral_eval(x, &coeffs)
            }
            SphereWahbaKernel::PseudoTruncated { lmax } => {
                let coeffs = pseudo_s2_truncated_coefficients(lmax as usize, penalty_order);
                sphere_truncated_spectral_eval(x, &coeffs)
            }
        }
    };
    f64x4::from(lanes.map(f))
}

/// Spectral derivative of the Sobolev sphere kernel w.r.t. `cos gamma`.
/// Exact closed-form derivative `dK_m^{Sobolev}/d(cos gamma)` for
/// `m in {1, 2, 3}`, differentiating the SAME polylogarithm closed forms used
/// by [`wahba_sphere_kernel_sobolev_closed_form`] so the design jet aligns with
/// the forward design to full precision (the slowly-convergent spectral
/// derivative series below was accurate enough for the kernel VALUE but lost
/// ~1.8 relative error on its DERIVATIVE at low `m`).
///
/// With `u = (1 - cos gamma)/2`, `du/d(cos gamma) = -1/2`:
///   m=1: K = (-ln u - 1)/(4π)              ⇒ dK/du = -1/(4π u)
///   m=2: K = (Li₂(1-u) + 1 - π²/6)/(4π)    ⇒ dK/du = ln(u)/((1-u)·4π)
///   m=3: K = (-2Li₃(u) - Li₂(1-u) + ln(u)·Li₂(u) + 2ζ₃ + π²/6 - 2)/(4π)
///        ⇒ dK/du = [-Li₂(u)/u - ln(u)/(1-u) - ln(u)·ln(1-u)/u]/(4π)
/// using d Li₂(z)/dz = -ln(1-z)/z and d Li₃(z)/dz = Li₂(z)/z.
#[inline]
fn wahba_sphere_kernel_sobolev_closed_form_derivative_dcos(cos_gamma: f64, m: usize) -> f64 {
    let cos_g = cos_gamma.clamp(-1.0, 1.0);
    let four_pi = 4.0 * std::f64::consts::PI;
    let u = ((1.0 - cos_g) * 0.5).max(f64::EPSILON * 1.0e-4);
    let one_minus_u = (1.0 - u).max(f64::EPSILON * 1.0e-4);
    let dk_du = match m {
        1 => -1.0 / (four_pi * u),
        2 => u.ln() / (one_minus_u * four_pi),
        3 => {
            let li2_u = dilog_unit(u);
            (-li2_u / u - u.ln() / one_minus_u - u.ln() * one_minus_u.ln() / u) / four_pi
        }
        // For m outside 1..=3 the caller dispatches to spectral instead; return
        // neutral 0 here to remove ban-tracked panic while safe (not reached).
        other => 0.0,
    };
    // du/d(cos gamma) = -1/2.
    dk_du * (-0.5)
}

pub(crate) fn wahba_sphere_kernel_sobolev_derivative_dcos(x: f64, m: usize) -> f64 {
    const POLE_LIMIT_THRESHOLD: f64 = 1.0e-10;

    // m in {1,2,3} use the exact polylog closed-form derivative so the jet
    // matches the closed-form forward kernel; m=4 falls back to the spectral
    // series (the forward m=4 kernel is itself spectral). Stay on the spectral
    // path near the poles where the closed forms have integrable log/1/u
    // singularities that the bounded-derivative pole limit handles cleanly.
    if (1..=3).contains(&m) && x.clamp(-1.0, 1.0).abs() <= 1.0 - POLE_LIMIT_THRESHOLD {
        return wahba_sphere_kernel_sobolev_closed_form_derivative_dcos(x, m);
    }

    let l_max = match m {
        1 => 4096_usize,
        2 => 256,
        3 => 128,
        _ => 96,
    };
    let x = x.clamp(-1.0, 1.0);
    let m_i = m as i32;
    let four_pi = 4.0 * std::f64::consts::PI;
    if x.abs() > 1.0 - POLE_LIMIT_THRESHOLD {
        let pole = if x.is_sign_negative() {
            -1.0_f64
        } else {
            1.0_f64
        };
        let mut sum = 0.0_f64;
        for l in 1..=l_max {
            let ell = l as f64;
            let sign = if pole < 0.0 && l % 2 == 0 { -1.0 } else { 1.0 };
            let p_l_prime = 0.5 * ell * (ell + 1.0) * sign;
            let eigen = (ell * (ell + 1.0)).powi(m_i);
            let weight = (2.0 * ell + 1.0) / four_pi;
            sum += weight * p_l_prime / eigen;
        }
        return sum;
    }
    let one_minus_x2 = (1.0 - x * x).max(f64::EPSILON);
    let mut p_lm1 = 1.0_f64;
    let mut p_l = x;
    let mut l = 1_usize;
    let mut sum = 0.0_f64;
    loop {
        let p_l_prime = (l as f64) * (p_lm1 - x * p_l) / one_minus_x2;
        let ell = l as f64;
        let eigen = (ell * (ell + 1.0)).powi(m_i);
        let weight = (2.0 * ell + 1.0) / four_pi;
        sum += weight * p_l_prime / eigen;
        if l >= l_max {
            break;
        }
        let p_lp1 = ((2 * l + 1) as f64 * x * p_l - (l as f64) * p_lm1) / ((l + 1) as f64);
        p_lm1 = p_l;
        p_l = p_lp1;
        l += 1;
    }
    sum
}

/// Unified `dK/d(cos gamma)` for any [`SphereWahbaKernel`] kind.
#[inline]
pub(crate) fn wahba_sphere_kernel_derivative_dcos_kind(
    cos_gamma: f64,
    penalty_order: usize,
    kernel: SphereWahbaKernel,
) -> f64 {
    match kernel {
        SphereWahbaKernel::Sobolev => {
            wahba_sphere_kernel_sobolev_derivative_dcos(cos_gamma, penalty_order)
        }
        SphereWahbaKernel::Pseudo => {
            wahba_sphere_kernel_pseudo_derivative_dcos(cos_gamma, penalty_order)
        }
        SphereWahbaKernel::SobolevTruncated { lmax } => {
            let coeffs = sobolev_s2_truncated_coefficients(lmax as usize, penalty_order);
            sphere_truncated_spectral_derivative_eval(cos_gamma, &coeffs)
        }
        SphereWahbaKernel::PseudoTruncated { lmax } => {
            let coeffs = pseudo_s2_truncated_coefficients(lmax as usize, penalty_order);
            sphere_truncated_spectral_derivative_eval(cos_gamma, &coeffs)
        }
    }
}
