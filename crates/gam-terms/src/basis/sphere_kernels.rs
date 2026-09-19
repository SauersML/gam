//! Closed-form and spectral zonal Wahba kernels on S².
//!
//! This module owns the scalar/SIMD kernel dispatch for intrinsic sphere
//! smooths. Callers in `basis` handle data validation, coordinate transforms,
//! and matrix assembly.

use super::BasisError;
use super::polylog::{dilog_of_complement, dilog_unit, trilog_unit};
use super::sphere_half_angle::HalfAngleSeparation;
use super::sphere_spec::SphereWahbaKernel;
use super::sphere_spectral::{
    sobolev_s2_truncated_coefficients, sphere_truncated_spectral_derivative_eval,
    sphere_truncated_spectral_eval, sphere_truncated_spectral_second_derivative_eval,
};

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
/// `m in {1, 2, 3}` plus spectral series for `m = 4`, as a function of the
/// half-angle separation.
///
/// Every closed form here is a function of `u = sin²(γ/2)` and
/// `v = cos²(γ/2) = 1 − u`, which is why the separation is carried as the pair:
/// `m = 1` needs `−ln u` (singular at coincidence) and `m = 2` needs `Li₂(v)`
/// (whose argument vanishes at the antipode). Taking `v` as `1.0 - u` instead
/// destroys the antipodal end — at `cos γ = −1 + 1e-16`, `u` rounds to `1.0`
/// and `1.0 - u` is `0`, reporting an exact antipode for a pair that is not one.
#[inline]
pub(crate) fn wahba_sphere_kernel_sobolev(sep: HalfAngleSeparation, m: usize) -> f64 {
    let four_pi = 4.0 * std::f64::consts::PI;
    let pi2_6 = std::f64::consts::PI * std::f64::consts::PI / 6.0;
    // No `f64::EPSILON * 1.0e-4` floor on either half. Both polylogarithms
    // already carry their endpoints exactly (`Li₂(0) = Li₃(0) = 0`,
    // `Li₂(1) = π²/6`, `Li₃(1) = ζ₃`), so the only thing the floor did was to
    // keep `u.ln()` off `-∞` — and it did that by evaluating the kernel at a
    // separation of `1.5e-10` rad instead of at `0`, which is a choice of
    // resolution, not a limit (#2469, #2475). The two places `ln u` appears are
    // handled on their own terms below.
    let u = sep.u;
    let one_minus_u = sep.v;
    match m {
        // `K_1 = (−ln u − 1)/4π` is genuinely `+∞` at coincidence: the m = 1
        // Sobolev Gram diagonal does not exist, which is exactly why
        // `validate_spherical_wahba_gram_request` refuses to build one. Letting
        // the infinity through is the honest report — every public entry point
        // checks `is_finite` — where the floor returned `45.0/4π` and looked
        // like an answer.
        1 => (-u.ln() - 1.0) / four_pi,
        // `Li₂(v)` is taken through the exact small half of the pair: at
        // coincidence `u = 0` exactly but `v = 1` only up to rounding, and
        // `dilog_unit(v)` would give every Gram diagonal entry its own last bits.
        2 => (dilog_of_complement(u, one_minus_u) + 1.0 - pi2_6) / four_pi,
        3 => {
            const ZETA3: f64 = 1.2020569031595942853997381615114499907649862923404988817922;
            let li3_u = trilog_unit(u);
            let li2_one_minus_u = dilog_of_complement(u, one_minus_u);
            // `ln(u)·Li₂(u)` is `−∞ · 0` at coincidence with the REMOVABLE
            // limit `0`, since `Li₂(u) = u + u²/4 + … = O(u)` and `u ln u → 0`.
            // Resolving it analytically is what makes `K_3(0) = (2ζ₃ − 2)/4π`
            // exact rather than floor-dependent.
            let cross = if u <= 0.0 {
                0.0
            } else {
                u.ln() * dilog_unit(u)
            };
            (-2.0 * li3_u - li2_one_minus_u + cross + 2.0 * ZETA3 + pi2_6 - 2.0) / four_pi
        }
        _ => wahba_sphere_kernel_sobolev_spectral(sep.cos_gamma(), m),
    }
}

/// Truncation degree of the Sobolev spectral series, shared by the kernel value
/// and its first and second derivatives so all three sum the same partial sum.
fn sobolev_spectral_l_max(m: usize) -> usize {
    match m {
        1 => 4096,
        2 => 256,
        3 => 128,
        _ => 96,
    }
}

/// Spectral Legendre-series evaluation of the Sobolev kernel
/// `K_m^{Sobolev}(gamma) = (1/4pi) sum_{l >= 1} (2l+1) *
/// [l(l+1)]^{-m} * P_l(cos gamma)`.
#[inline]
pub(crate) fn wahba_sphere_kernel_sobolev_spectral(cos_gamma: f64, m: usize) -> f64 {
    let l_max = sobolev_spectral_l_max(m);
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

/// Evaluate the Wahba sphere reproducing kernel at a single half-angle
/// separation.
#[inline]
pub(crate) fn wahba_sphere_kernel_kind(
    sep: HalfAngleSeparation,
    penalty_order: usize,
    kernel: SphereWahbaKernel,
) -> Result<f64, BasisError> {
    if !(1..=4).contains(&penalty_order) {
        crate::bail_invalid_basis!(
            "spherical spline penalty_order must be one of 1, 2, 3, 4; got {penalty_order}"
        );
    }
    let value = wahba_sphere_kernel_kind_unchecked(sep, penalty_order, kernel);
    if !value.is_finite() {
        crate::bail_invalid_basis!("spherical spline kernel produced a non-finite value");
    }
    Ok(value)
}

/// The kernel dispatch itself, with `penalty_order` and finiteness already
/// established (or, on the SIMD path, established once for the whole vector).
///
/// This is the single place the two [`SphereWahbaKernel`] variants are mapped
/// to their evaluators; the scalar and SIMD entry points differ only in how they
/// loop over it.
#[inline]
fn wahba_sphere_kernel_kind_unchecked(
    sep: HalfAngleSeparation,
    penalty_order: usize,
    kernel: SphereWahbaKernel,
) -> f64 {
    match kernel {
        SphereWahbaKernel::Sobolev => wahba_sphere_kernel_sobolev(sep, penalty_order),
        SphereWahbaKernel::SobolevTruncated { lmax } => {
            let coeffs = sobolev_s2_truncated_coefficients(lmax as usize, penalty_order);
            sphere_truncated_spectral_eval(sep.cos_gamma(), &coeffs)
        }
    }
}

/// SIMD lane-wise evaluation over four half-angle separations. Both branches
/// are scalar-per-lane because the closed forms contain non-vector elementary and polylogarithm calls; what the vector form buys is
/// the *separation* arithmetic (see
/// [`super::sphere_half_angle::half_angle_separation`]), which is pure `+ − ×`.
#[inline]
pub(crate) fn wahba_sphere_kernel_simd_kind(
    u: wide::f64x4,
    v: wide::f64x4,
    penalty_order: usize,
    kernel: SphereWahbaKernel,
) -> wide::f64x4 {
    use wide::f64x4;
    if !(1..=4).contains(&penalty_order) {
        return f64x4::from(f64::NAN);
    }
    let zero = f64x4::ZERO;
    let u_lanes = u.fast_max(zero).fast_min(f64x4::ONE).to_array();
    let v_lanes = v.fast_max(zero).fast_min(f64x4::ONE).to_array();
    let mut out = [0.0_f64; 4];
    for lane in 0..4 {
        let sep = HalfAngleSeparation {
            u: u_lanes[lane],
            v: v_lanes[lane],
        };
        out[lane] = wahba_sphere_kernel_kind_unchecked(sep, penalty_order, kernel);
    }
    f64x4::from(out)
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
fn wahba_sphere_kernel_sobolev_closed_form_derivative_dhav(
    sep: HalfAngleSeparation,
    m: usize,
) -> f64 {
    let four_pi = 4.0 * std::f64::consts::PI;
    // No floor on `u` or `v`. The sole caller only reaches this closed form
    // below the COINCIDENT pole, which bounds `u = sin²(γ/2)` away from `0`
    // by `5e-11` — nine orders above the `f64::EPSILON * 1.0e-4` floor these
    // lines used to carry (a factor of 2.3e9), so it could never bind.
    // Flooring here was dead arithmetic that read as if the singularities were
    // being handled (#2469, #2475 site 4).
    //
    // `1 - u` is carried as the separation's own `v = cos²(γ/2)` rather than as
    // `1.0 - u`, because `v` is the quantity that vanishes at the ANTIPODE and
    // only this form resolves it. Going through `u` instead destroys the
    // antipodal end outright — at `cos γ = -1 + 1e-16`, `1 - cos γ` rounds to
    // `2.0`, so `u` rounds to `1.0` and `1.0 - u` is `0`, reporting an exact
    // antipode for a pair that is not one. Both halves reach here already taken
    // from the side where they are small and exact (#2489).
    let u = sep.u;
    let v = sep.v;
    // assert!, not debug_assert!: the ban-scanner forbids debug_assert (silent
    // in release → debug/release divergence). An O(1) comparison in front of a
    // dilogarithm is free.
    //
    // Only `u > 0` is asserted. `v == 0` is the antipode, which is an ordinary
    // interior point of all three closed forms — every one of them is finite
    // and smooth there — and is handled by the removable-singularity arms
    // below rather than excluded.
    assert!(
        u > 0.0,
        "closed-form Sobolev derivative called at the coincident pole \
         (u = sin²(γ/2) = {u}); the caller's POLE_LIMIT_THRESHOLD guard is \
         supposed to make this unreachable"
    );
    // `ln u`, taken from whichever of the two exact halves is the small one.
    // `ln_1p(-v)` keeps full relative accuracy as `v → 0` (where `ln u → 0` and
    // is about to be divided by `v`); `u.ln()` keeps it as `u → 0`, where
    // `ln_1p` would have to reconstitute a tiny `1 - v` and lose the digits.
    let ln_u = if v <= 0.5 { (-v).ln_1p() } else { u.ln() };
    // `ln(1-u) = ln v`, by the same rule mirrored. Taking it as `v.ln()` near
    // the COINCIDENT end costs relative accuracy for exactly the reason `ln u`
    // costs it near the antipodal end: at `u = 5e-11` the true `ln v` is
    // `-5e-11`, but `v` can only carry `1 - 5e-11` to an absolute `1.1e-16`, so
    // the answer arrives with `2.2e-6` relative error. That error was reaching
    // the m=3 derivative — measured 2.4e-6 against a 40-digit reference at
    // `cos γ = 1 - 1e-10`, against `< 2e-14` everywhere else on the branch.
    let ln_v = if u <= 0.5 { (-u).ln_1p() } else { v.ln() };
    // `ln(u)/(1-u)` is `0/0` at the antipode with the finite limit `-1`:
    // `ln(1-v)/v = -1 - v/2 - v²/3 - …`. This is the factor that carries the
    // antipodal limit of BOTH the m=2 and the m=3 form.
    let ln_u_over_v = if v == 0.0 { -1.0 } else { ln_u / v };
    let dk_du = match m {
        1 => -1.0 / (four_pi * u),
        2 => ln_u_over_v / four_pi,
        3 => {
            let li2_u = dilog_unit(u);
            // `ln(u)·ln(1-u)/u` is `0·(-∞)` at the antipode and vanishes there
            // like `v·ln v`; nothing cancels in it for `v > 0`.
            let cross = if v == 0.0 { 0.0 } else { ln_u * ln_v / u };
            (-li2_u / u - ln_u_over_v - cross) / four_pi
        }
        // SAFETY: the sole caller
        // `wahba_sphere_kernel_sobolev_derivative_dhav` dispatches to this
        // closed form only inside `(1..=3).contains(&m)`, so
        // any other `m` is a caller-contract violation (a programming error,
        // not runtime data), and panicking surfaces it instead of returning a
        // silently-wrong derivative.
        other => {
            panic!("closed-form Sobolev derivative only defined for m in {{1,2,3}}; got m={other}")
        }
    };
    dk_du
}

/// `dK_m^{Sobolev}/du` with respect to the half-angle separation
/// `u = sin²(γ/2)`.
pub(crate) fn wahba_sphere_kernel_sobolev_derivative_dhav(
    sep: HalfAngleSeparation,
    m: usize,
) -> f64 {
    const POLE_LIMIT_THRESHOLD: f64 = 1.0e-10;
    // `u = (1 − cos γ)/2`, so the historical `cos γ ≤ 1 − POLE_LIMIT_THRESHOLD`
    // guard is exactly `u ≥ POLE_LIMIT_THRESHOLD/2`. Stating it in `u` keeps the
    // boundary where it was while letting the caller hand in a `u` that is
    // accurate below it (#2489) instead of one quantized to multiples of `ε/4`.
    const POLE_LIMIT_U: f64 = 0.5 * POLE_LIMIT_THRESHOLD;

    // m in {1,2,3} use the exact polylog closed-form derivative so the jet
    // matches the closed-form forward kernel; m=4 falls back to the spectral
    // series (the forward m=4 kernel is itself spectral). Leave the closed form
    // near the COINCIDENT pole, where `dK/du` carries the genuine `1/u` (m=1)
    // and `ln u` (m=2, m=3) singularities of the Sobolev kernel.
    //
    // The guard is ONE-SIDED, and used not to be. Only `cos γ → +1` is a pole
    // of these kernels; `cos γ → -1` is an ordinary interior point where all
    // three derivatives are finite, smooth, and elementary:
    //
    // ```text
    //   m=1, m=2:  dK/d(cos γ)|_{cos γ = -1} = 1/(8π)          = 3.9788735772973834e-2
    //   m=3:       dK/d(cos γ)|_{cos γ = -1} = (π²/6 - 1)/(8π) = 2.5661111176813525e-2
    // ```
    //
    // Routing the antipode to the spectral branch was not a conservative
    // choice, it was wrong, because term-by-term differentiation of a Legendre
    // series need not converge where the series itself does. At `m = 1` the
    // differentiated terms `(2ℓ+1)(-1)^{ℓ+1}/8π` GROW, so the branch summed a
    // divergent alternating series and returned its `l_max`-th partial sum:
    // `Σ_{ℓ≤L}(-1)^{ℓ+1}(2ℓ+1) = -L` exactly, i.e. `-4096/(8π) = -162.9747`
    // where the answer is `+0.0397887`. Wrong sign, 4096x magnitude, and a pure
    // function of the truncation constant — doubling `l_max` doubles it.
    //
    // The region is not measure-zero either: `|cos γ| > 1 - 1e-10` at the
    // antipodal end is every pair within 1.4e-5 rad (~3 arcsec) of antipodal,
    // and the previous behaviour stepped from `+0.0398` to `-162.97` across
    // that boundary. Antipodal pairs are what farthest-point centre selection
    // actively seeks out, which is the same reason the `Li₃` accuracy near
    // `z = 1` mattered (see `polylog`'s module docs).
    //
    // m=4 keeps both poles: its differentiated terms decay like `ℓ^-4`, so the
    // spectral limit converges there and is the only form available.
    if (1..=3).contains(&m) && sep.u >= POLE_LIMIT_U {
        return wahba_sphere_kernel_sobolev_closed_form_derivative_dhav(sep, m);
    }

    let l_max = sobolev_spectral_l_max(m);
    let x = sep.cos_gamma();
    let m_i = m as i32;
    let four_pi = 4.0 * std::f64::consts::PI;
    // ONE sweep, valid on the closed interval including both poles. There is no
    // pole special-case and no `1 - x²` floor, because the derivative is taken
    // from the recurrence that has no pole:
    //
    // ```text
    //   P'_ℓ(x) = (2ℓ - 1)·P_{ℓ-1}(x) + P'_{ℓ-2}(x),    P'_0 = 0, P'_1 = 1
    // ```
    //
    // The form this replaces, `P'_ℓ = ℓ(P_{ℓ-1} - x·P_ℓ)/(1 - x²)`, is a
    // removable `0/0` at `x = ±1` — which is why it needed a floor AND a
    // separate pole branch — and it is already losing digits well before it
    // gets there: its numerator subtracts two `O(1)` Legendre values to leave
    // `O((ℓ+1)(1-x))`, so the relative error grows like `ε/(1-x²)`. Measured on
    // a truncated Sobolev kernel (`lmax=64`, `m=2`) against a 40-digit
    // reference:
    //
    // ```text
    //   |x|          0.99      0.9999    1-1e-8    1-1e-9    1-1e-10
    //   quotient    7.4e-16    1.5e-13   7.0e-10   3.3e-8    6.5e-8
    //   recurrence  4.3e-16    1.3e-17   1.8e-15   3.0e-15   2.2e-15
    // ```
    //
    // so the old code was handing the pole branch a value that had already
    // decayed to eight digits by the time the threshold caught it. The
    // recurrence needs no catching: at `x = ±1` it reproduces the closed pole
    // formula `P'_ℓ(±1) = (±1)^{ℓ+1}·ℓ(ℓ+1)/2` to 5e-16, so the branch it
    // replaces was computing the same number by a second route.
    let mut p_prev = 1.0_f64; // P_{ℓ-2}, seeded at P_0
    let mut p_curr = x; // P_{ℓ-1}, seeded at P_1
    let mut d_prev = 0.0_f64; // P'_{ℓ-2}, seeded at P'_0
    let mut d_curr = 1.0_f64; // P'_{ℓ-1}, seeded at P'_1
    let mut sum = 3.0 * d_curr / (four_pi * 2.0_f64.powi(m_i));
    for l in 2..=l_max {
        let ell = l as f64;
        let two_l_minus_1 = 2.0 * ell - 1.0;
        let d_next = two_l_minus_1 * p_curr + d_prev;
        let p_next = (two_l_minus_1 * x * p_curr - (ell - 1.0) * p_prev) / ell;
        let eigen = (ell * (ell + 1.0)).powi(m_i);
        let weight = (2.0 * ell + 1.0) / four_pi;
        sum += weight * d_next / eigen;
        p_prev = p_curr;
        p_curr = p_next;
        d_prev = d_curr;
        d_curr = d_next;
    }
    // The spectral sweep produces `dK/d(cos γ)`; `d(cos γ)/du = −2`.
    -2.0 * sum
}

/// Unified `dK/du` for any [`SphereWahbaKernel`] kind, against the half-angle
/// separation `u = sin²(γ/2)`.
///
/// This is the form the design jet wants: paired with
/// [`super::sphere_half_angle::half_angle_partials`] it computes the `|γ|` cusp
/// gradient as a product of two finite factors, where the `cos γ` chain has to
/// recover it from `∞ · 0` (#2489).
#[inline]
pub(crate) fn wahba_sphere_kernel_derivative_dhav_kind(
    sep: HalfAngleSeparation,
    penalty_order: usize,
    kernel: SphereWahbaKernel,
) -> f64 {
    match kernel {
        SphereWahbaKernel::Sobolev => {
            wahba_sphere_kernel_sobolev_derivative_dhav(sep, penalty_order)
        }
        SphereWahbaKernel::SobolevTruncated { lmax } => {
            let coeffs = sobolev_s2_truncated_coefficients(lmax as usize, penalty_order);
            -2.0 * sphere_truncated_spectral_derivative_eval(sep.cos_gamma(), &coeffs)
        }
    }
}

/// `d²K_m^{Sobolev}/du²`. For `m ∈ {1, 2, 3}` this differentiates the closed
/// forms of [`wahba_sphere_kernel_sobolev_closed_form_derivative_dhav`] once more
/// (`dv/du = −1`, `d Li₂(u)/du = −ln v / u`):
///
/// ```text
///   m=1:  1/(4π u²)
///   m=2:  (v/u + ln u) / (4π v²)
///   m=3:  [ (Li₂(u) + u·ln u / v + ln u · ln v)/u² − (v/u + ln u)/v² ] / (4π)
/// ```
///
/// all three divergent at coincidence, where the caller never asks for them.
/// `m = 4` differentiates the same truncated spectral series as the first
/// derivative, which converges there.
fn wahba_sphere_kernel_sobolev_second_derivative_dhav(sep: HalfAngleSeparation, m: usize) -> f64 {
    if !(1..=3).contains(&m) {
        let coeffs = sobolev_s2_truncated_coefficients(sobolev_spectral_l_max(m), m);
        // The sweep produces `d²K/d(cos γ)²`, and `(d(cos γ)/du)² = 4`.
        return 4.0 * sphere_truncated_spectral_second_derivative_eval(sep.cos_gamma(), &coeffs);
    }
    let four_pi = 4.0 * std::f64::consts::PI;
    let u = sep.u;
    let v = sep.v;
    assert!(
        u > 0.0,
        "Sobolev m={m} second derivative diverges at coincidence (u = sin²(γ/2) = {u}); \
         the caller resolves coincidence without it"
    );
    // `ln u` and `ln v`, each from whichever exact half is small, as in the
    // first derivative.
    let ln_u = if v <= 0.5 { (-v).ln_1p() } else { u.ln() };
    let ln_v = if u <= 0.5 { (-u).ln_1p() } else { v.ln() };
    // `(v/u + ln u)/v²` is `0/0` at the antipode with the finite limit `½`, since
    // `v/u + ln u = v²/2 + 2v³/3 + …`. Away from it the cancellation costs
    // `O(ε/v)` relative, and the `(∂u)² = O(v)` factor it multiplies in the
    // Hessian brings that back to `O(ε)`.
    let antipode = if v == 0.0 { 0.5 } else { (v / u + ln_u) / (v * v) };
    match m {
        1 => 1.0 / (four_pi * u * u),
        2 => antipode / four_pi,
        _ => {
            let ln_u_over_v = if v == 0.0 { -1.0 } else { ln_u / v };
            let cross = if v == 0.0 { 0.0 } else { ln_u * ln_v };
            ((dilog_unit(u) + u * ln_u_over_v + cross) / (u * u) - antipode) / four_pi
        }
    }
}

/// Unified `d²K/du²` for any [`SphereWahbaKernel`] kind, the second-order
/// companion of [`wahba_sphere_kernel_derivative_dhav_kind`]. The Sobolev
/// `m ≤ 3` arms are only defined for `u > 0`.
pub(crate) fn wahba_sphere_kernel_second_derivative_dhav_kind(
    sep: HalfAngleSeparation,
    penalty_order: usize,
    kernel: SphereWahbaKernel,
) -> f64 {
    match kernel {
        SphereWahbaKernel::Sobolev => {
            wahba_sphere_kernel_sobolev_second_derivative_dhav(sep, penalty_order)
        }
        SphereWahbaKernel::SobolevTruncated { lmax } => {
            let coeffs = sobolev_s2_truncated_coefficients(lmax as usize, penalty_order);
            4.0 * sphere_truncated_spectral_second_derivative_eval(sep.cos_gamma(), &coeffs)
        }
    }
}

/// Whether the kernel's input-location Hessian exists where an evaluation point
/// coincides with a center, which is exactly when `dK/du` is finite at `u = 0`:
/// there `∂u = 0` and the Hessian is `K'(0)·∂²u`. The Sobolev spectral sum for
/// `K'(0)` goes like `Σ ℓ^{3−2m}`, so it needs `m ≥ 3`. Every truncated kernel
/// is a polynomial in `cos γ` and smooth.
pub(crate) fn wahba_sphere_kernel_hessian_exists_at_coincidence(
    penalty_order: usize,
    kernel: SphereWahbaKernel,
) -> bool {
    match kernel {
        SphereWahbaKernel::Sobolev => penalty_order >= 3,
        SphereWahbaKernel::SobolevTruncated { .. } => true,
    }
}
