//! Unit / integration tests for the `basis` module.
//!
//! The test bodies live in the shared `tests/src_modules/` fixtures and are
//! pulled in verbatim; every item they reference resolves through the parent
//! module's re-exports (`use super::*`).

include!("../../../../tests/src_modules/smooths/basis_radial_periodic_thinplate_tests.rs");
include!("../../../../tests/src_modules/smooths/basis_duchon_matern_jet_derivative_tests.rs");

// Test oracles for the closed-form kernels. Production evaluates these kernels
// through the chart-dispatched builders in `closed_form_penalty`; the fixtures
// above compare against these direct formulas.

/// Riesz kernel R_j^d(r) = F^{-1}{|ρ|^{-2j}}(r) for r > 0.
///
/// Non-log case (j > 0, j ∉ d/2 + ℕ₀):
///   R_j^d(r) = Γ(d/2 - j) / (4^j π^{d/2} Γ(j)) · r^{2j - d}.
/// Log case (j = d/2 + n, n ∈ ℕ₀):
///   R_j^d(r) = c_n · r^{2n} · (log r + A_n),
///   c_n = (-1)^{n+1} / (2^{2j-1} π^{d/2} Γ(j) n!).
///
/// The finite-part constant `A_n` is chosen so the distributional
/// recurrence `Δ R_j^d = -R_{j-1}^d` holds exactly away from the
/// origin. This removes the previous null-space polynomial residue in
/// log-Riesz regimes and keeps the anisotropic `(-Δ_B)^q` path analytic.
fn riesz_kernel_value(d: usize, j: f64, r: f64) -> f64 {
    assert!(d >= 1, "riesz_kernel_value: d must be ≥ 1");
    assert!(
        j.is_finite() && j >= 1.0,
        "riesz_kernel_value: j must be ≥ 1, got {j}"
    );
    assert!(r > 0.0, "riesz_kernel_value: r must be > 0");

    // Detect log case: 2j is a non-negative even integer offset of `d`.
    // For integer `j` this is exact; for fractional `j` it never fires
    // because `2j − d` won't be an even integer to within `LOG_EPS`.
    let two_j = 2.0 * j;
    const LOG_EPS: f64 = 1e-12;
    let offset = two_j - d as f64;
    if offset >= -LOG_EPS && (offset.round() - offset).abs() < LOG_EPS {
        let n_f64 = (offset / 2.0).round();
        if n_f64 >= 0.0 && (n_f64 * 2.0 - offset).abs() < LOG_EPS {
            let n = n_f64 as usize;
            let two_j_i = (two_j.round()) as i32;
            let sign = if n.is_multiple_of(2) { -1.0 } else { 1.0 }; // (−1)^{n+1}
            let denom = 2.0_f64.powi(two_j_i - 1)
                * std::f64::consts::PI.powf(d as f64 / 2.0)
                * statrs::function::gamma::gamma(j)
                * super::closed_form_penalty::factorial_f64(n);
            return sign / denom
                * r.powi((2 * n) as i32)
                * (r.ln() + super::closed_form_penalty::log_riesz_finite_part_shift(d, n));
        }
    }

    // Non-log case (admits fractional `j`).
    let half_d = d as f64 / 2.0;
    let num = statrs::function::gamma::gamma(half_d - j);
    let denom = 4.0_f64.powf(j) * std::f64::consts::PI.powf(half_d) * statrs::function::gamma::gamma(j);
    num / denom * r.powf(2.0 * j - d as f64)
}

/// Matérn building block M_ℓ^d(r; κ) = F^{-1}{(|ρ|² + κ²)^{-ℓ}}(r) for r > 0, κ > 0.
///
/// M_ℓ^d(r; κ) = κ^{d/2 - ℓ} / ((2π)^{d/2} · 2^{ℓ-1} · Γ(ℓ)) · r^{ℓ - d/2} · K_{ℓ - d/2}(κr).
///
/// For r > 0, K_ν is evaluated by the Temme/Steed order-reduced algorithm used
/// by `bessel_k`, with the half-integer closed form retained where applicable.
///
/// For r → 0, returns the small-arg limit using K_ν(x) ~ Γ(|ν|)/2 · (x/2)^{-|ν|}
/// (ν ≠ 0) or the log limit (ν = 0).
fn matern_kernel_value(d: usize, ell: usize, kappa: f64, r: f64) -> f64 {
    assert!(d >= 1, "matern_kernel_value: d must be ≥ 1");
    assert!(ell >= 1, "matern_kernel_value: ell must be ≥ 1");
    if !(kappa > 0.0) {
        return f64::NAN;
    }
    assert!(r >= 0.0, "matern_kernel_value: r must be ≥ 0");

    let nu = ell as f64 - d as f64 / 2.0;
    let ln_pref = (d as f64 / 2.0 - ell as f64) * kappa.ln()
        - (d as f64 / 2.0) * (2.0 * std::f64::consts::PI).ln()
        - (ell as f64 - 1.0) * std::f64::consts::LN_2
        - statrs::function::gamma::ln_gamma(ell as f64);
    let pref = ln_pref.exp();

    if r == 0.0 {
        // M(0) = pref · lim_{r→0} r^{ℓ - d/2} K_{ℓ - d/2}(κr)
        // For ν > 0: K_ν(x) ~ Γ(ν)/2 (x/2)^{-ν}, so r^ν K_ν(κr) → Γ(ν)/2 (κ/2)^{-ν}.
        // For ν < 0 (i.e. ℓ < d/2): r^ν · K_{-|ν|}(κr) ~ r^ν · Γ(|ν|)/2 (κr/2)^{-|ν|}
        //   → Γ(|ν|)/2 (κ/2)^{-|ν|} · r^{ν - |ν|} = ∞ (singular). Return ∞.
        // For ν = 0: K_0(x) ~ -log(x/2) - γ; r^0·K_0(κr) → ∞. Return ∞.
        if nu > 0.0 {
            let lim = 0.5 * statrs::function::gamma::gamma(nu) * (0.5 * kappa).powf(-nu);
            return pref * lim;
        } else {
            return f64::INFINITY;
        }
    }

    let kr = kappa * r;
    let kv = super::closed_form_penalty::bessel_k(nu, kr);
    pref * r.powf(nu) * kv
}

/// Returns the anisotropic Duchon penalty quantity (without
/// the J prefactor on `g_q` — caller multiplies by J) through the
/// analytic radial `(-Δ_B)^q` chain. No numerical quadrature is used.
///
/// For `R = 0` the radial chain may be singular. The finite spectral
/// self-pair is evaluated first by `schoenberg_self_pair_bundle` using
/// the closed Gamma/Beta diagonal; smooth odd-dimensional hybrid cases
/// use the Taylor limit. Remaining non-convergent diagonals are rejected
/// rather than approximated by a heat quadrature.
fn anisotropic_duchon_penalty_radial(
    q: usize,
    m: usize,
    s: f64,
    kappa: f64,
    eta: &[f64],
    r: &[f64],
) -> f64 {
    assert_eq!(
        eta.len(),
        r.len(),
        "anisotropic_duchon_penalty_radial: eta and r dimension mismatch"
    );
    assert!(
        !r.is_empty(),
        "anisotropic_duchon_penalty_radial: empty input"
    );
    assert!(
        q <= 2,
        "anisotropic_duchon_penalty_radial: q must be in {{0,1,2}}"
    );

    let powers = super::closed_form_penalty::AnisoMetricPowers::new(eta);
    super::closed_form_penalty::anisotropic_duchon_penalty_radial_with_powers(q, m, s, kappa, eta, &powers, r)
}

/// Bundled value + first/second derivatives of the radial-form
/// anisotropic pair-block `J · g_q`.
///
/// Uses analytic chain rules on `(R, s_1, s_2, u_1, u_2)` for regular
/// non-log regimes. Finite spectral self-pairs use the closed
/// Schoenberg Gamma/Beta diagonal with exact η/κ derivatives; smooth
/// odd-dimensional hybrid self-pairs use the Taylor limit; other
/// singular/log-Riesz self-pairs use the analytic Schoenberg derivative
/// bundle for the same distributional diagonal used by the value path.
fn pair_block_radial_with_j_second_derivatives(
    q: usize,
    m: usize,
    s: usize,
    kappa: f64,
    eta: &[f64],
    r: &[f64],
) -> super::closed_form_penalty::PairBlockBundle {
    let powers = super::closed_form_penalty::AnisoMetricPowers::new(eta);
    super::closed_form_penalty::pair_block_radial_with_j_second_derivatives_with_powers(
        q,
        m,
        s,
        kappa,
        eta,
        &powers,
        r,
        super::closed_form_penalty::PairOrigin::Full,
    )
}
