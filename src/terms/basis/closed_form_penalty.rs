//! Closed-form analytic Duchon/thin-plate penalty kernels.
//!
//! Self-contained math for the closed-form (collision-free and collided)
//! partial-fraction radial penalty kernels and their `κ`/`η` derivative
//! bundles. Extracted verbatim from `basis` along its module seam; the two
//! `super::` references resolve to the parent `basis` module's Duchon
//! partial-fraction helpers.

use crate::linalg::utils::KahanSum;
use crate::probability::binomial_coefficient_f64 as binomial_f64;
use statrs::function::gamma::{gamma as gamma_fn, ln_gamma};
use std::sync::OnceLock;

/// Gauss-Legendre nodes and weights on `[-1, 1]` for `n` points,
/// computed via Newton iteration on Legendre polynomial roots
/// (Bonnet's recurrence). Returns `(nodes, weights)` ascending.
pub(crate) fn compute_gauss_legendre(n: usize) -> (Vec<f64>, Vec<f64>) {
    let mut tmp: Vec<(f64, f64)> = Vec::with_capacity(n);
    let half = n.div_ceil(2);
    for i in 0..half {
        let mut z = (std::f64::consts::PI * (i as f64 + 0.75) / (n as f64 + 0.5)).cos();
        let mut pp = 0.0_f64;
        for _ in 0..200 {
            let mut p1 = 1.0_f64;
            let mut p2 = 0.0_f64;
            for j in 0..n {
                let p3 = p2;
                p2 = p1;
                p1 = ((2.0 * j as f64 + 1.0) * z * p2 - j as f64 * p3) / (j as f64 + 1.0);
            }
            pp = n as f64 * (z * p1 - p2) / (z * z - 1.0);
            let z_prev = z;
            z = z_prev - p1 / pp;
            if (z - z_prev).abs() < 1e-15 {
                break;
            }
        }
        let w = 2.0 / ((1.0 - z * z) * pp * pp);
        // For odd n the central node is at z = 0; record once.
        if !n.is_multiple_of(2) && i == half - 1 {
            tmp.push((0.0, w));
        } else {
            tmp.push((-z.abs(), w));
            tmp.push((z.abs(), w));
        }
    }
    tmp.sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap_or(std::cmp::Ordering::Equal));
    let mut nodes = Vec::with_capacity(n);
    let mut weights = Vec::with_capacity(n);
    for (z, w) in tmp.into_iter().take(n) {
        nodes.push(z);
        weights.push(w);
    }
    (nodes, weights)
}

pub(crate) fn gauss_legendre_64() -> &'static (Vec<f64>, Vec<f64>) {
    static CACHE: OnceLock<(Vec<f64>, Vec<f64>)> = OnceLock::new();
    CACHE.get_or_init(|| compute_gauss_legendre(64))
}

/// True when the Beta-form Schwinger integral
///
/// ```text
/// f(R) = (1/B(2m, 2s)) ∫_0^1 t^{2m-1}(1-t)^{2s-1} M_{2(m+s)}^d(√(1-t)·κ, R) dt
/// ```
///
/// for the hybrid Duchon kernel
/// `1/(|w|^{4m}(κ²+|w|²)^{2s})` is integrable everywhere on `[0,1]`. The
/// only issue is the Matérn factor at `t→1` (Riesz limit κ_t → 0). For
/// Matérn order `n = 2(m+s)` in dimension `d`:
///   * `2n > d`: M ~ κ_t^{d-2n}, the (1-t)^{2s-1} factor must absorb the
///     exponent `(d-2n)/2`. Integrability at t=1 requires
///     `(2s - 1) + (d - 2n)/2 > -1`, i.e. `d > 4m`.
///   * `2n ≤ d`: M stays bounded; integrability is automatic.
/// Combining, the Schwinger integrand is everywhere bounded iff `d > 4m`
/// or `4(m+s) ≤ d`. The latter never coincides with the convergent
/// regime of canonical TPS, so we use `d > 4m` as the dispatch gate.
#[inline]
pub(crate) fn schwinger_radial_is_convergent(d: usize, m: usize) -> bool {
    d > 4 * m
}

/// Numerically-stable evaluation of `[f, f', …, f^{(max_order)}](R)` for
/// the hybrid Duchon kernel
///
/// ```text
/// f̂(w) = 1 / (|w|^{4m} · (κ² + |w|²)^{2s})
/// ```
///
/// matching the codebase's PF convention (Riesz factor exponent `4m` is
/// `2·a` with `a = 2m`, Matérn factor exponent `2s` is the `b = 2s` from
/// `isotropic_duchon_penalty`). Computed via Schwinger parametrization
///
/// ```text
/// 1 / (A^p B^q) = (1/B(p,q)) · ∫_0^1 t^{p-1}(1-t)^{q-1} (tA + (1-t)B)^{-(p+q)} dt
/// ```
///
/// with `A = |w|²`, `B = κ² + |w|²`, `p = 2m`, `q = 2s`. The bracket
/// simplifies to `|w|² + (1-t)κ²`, so each integrand point is the Matérn
/// spectrum of order `2(m+s)` with effective inverse-length
/// `κ_t = √(1-t)·κ`. Inverse Fourier termwise:
///
/// ```text
/// f(R) = (1/B(2m, 2s)) · ∫_0^1 t^{2m-1}(1-t)^{2s-1} · M_{2(m+s)}^d(√(1-t)·κ, R) dt
/// ```
///
/// Replacing the partial-fraction expansion with this representation
/// eliminates the catastrophic cancellation between alternating Riesz
/// and Matérn blocks at high `d`: every quadrature contribution carries
/// its full IEEE-754 precision because the integrand is smooth and
/// bounded for `R > 0` (precondition `d > 4m` ensures integrability at
/// `t→1`; Gauss-Legendre nodes are open on `[0,1]` so `t = 1` is never
/// evaluated).
///
/// Differentiation under the integral sign moves into the Matérn factor;
/// `M^{(k)}_{2(m+s)}^d(κ_t, R)` is computed by `matern_block_radial_derivatives`.
pub(crate) fn stable_hybrid_duchon_radial(
    d: usize,
    m: usize,
    s: usize,
    kappa: f64,
    r: f64,
    max_order: usize,
) -> Vec<f64> {
    assert!(m >= 1, "stable_hybrid_duchon_radial: m ≥ 1");
    assert!(s >= 1, "stable_hybrid_duchon_radial: s ≥ 1");
    assert!(kappa > 0.0, "stable_hybrid_duchon_radial: κ > 0");
    assert!(r > 0.0, "stable_hybrid_duchon_radial: r > 0");
    assert!(
        max_order <= 6,
        "stable_hybrid_duchon_radial requires max_order <= 6: max_order={max_order}"
    );
    assert!(
        schwinger_radial_is_convergent(d, m),
        "stable_hybrid_duchon_radial: requires d > 4m"
    );

    // Substitute `t = 1 - u²` to make the integrand analytic at the
    // Riesz endpoint. With `p = 2m`, `q = 2s`, `n = 2(m+s)`:
    //
    //   ∫_0^1 t^{p-1}(1-t)^{q-1} M_n^d(√(1-t)·κ, R) dt
    //     = 2 ∫_0^1 (1-u²)^{p-1} · u^{2q-1} · M_n^d(u·κ, R) du
    //
    // At `u→0`, `M_n^d(uκ, R) ∝ (uκ)^{d-2n}`, so the combined
    // u-exponent of the integrand is `(2q-1) + (d-2n) = d - 4m - 1`.
    // Under the convergence gate `d > 4m` this is a non-negative
    // integer (`d`, `m` integers), so the integrand is analytic on
    // `[0,1]`, restoring spectral Gauss-Legendre convergence (~14-16
    // digits at 64 points). Without this substitution the original
    // `(1-t)^{q-1}` factor combines with the diverging Matérn into
    // `(1-t)^{(d-1)/2 - m + s}`, half-integer for odd d, which kills
    // Gauss-Legendre's spectral rate down to 4-5 digits.
    let (nodes, weights) = gauss_legendre_64();
    let p_eff = 2 * m;
    let q_eff = 2 * s;
    let matern_order = p_eff + q_eff;
    let log_beta =
        ln_gamma(p_eff as f64) + ln_gamma(q_eff as f64) - ln_gamma((p_eff + q_eff) as f64);
    let inv_beta = (-log_beta).exp();

    let mut accum = vec![KahanSum::default(); max_order + 1];
    for (xi, wi) in nodes.iter().zip(weights.iter()) {
        // Map [-1, 1] -> [0, 1] via u = (1 + ξ)/2; Jacobian du/dξ = 1/2.
        let u = 0.5 * (1.0 + xi);
        if u <= 0.0 || u >= 1.0 {
            continue;
        }
        let kappa_u = u * kappa; // sqrt(1-t)·κ with 1-t = u²
        let one_minus_u2 = 1.0 - u * u;
        let one_minus_u2_pow = one_minus_u2.powi((p_eff - 1) as i32);
        let u_pow = u.powi((2 * q_eff - 1) as i32);
        // The factor of 2 from the t = 1-u² Jacobian, combined with
        // 1/2 from the [-1,1]→[0,1] map, leaves the unit prefactor wi.
        let weight = wi * one_minus_u2_pow * u_pow;
        let matern_derivs = matern_block_radial_derivatives(d, matern_order, kappa_u, r, max_order);
        for (k, v) in matern_derivs.iter().enumerate() {
            accum[k].add(weight * v);
        }
    }
    accum.iter().map(|acc| inv_beta * acc.sum()).collect()
}

pub(crate) fn factorial_f64(n: usize) -> f64 {
    let mut acc = 1.0_f64;
    for k in 2..=n {
        acc *= k as f64;
    }
    acc
}

/// Modified Bessel function of the second kind, K_ν(x), for x > 0.
///
/// Algorithm:
/// - Half-integer ν = n + 1/2 (n ≥ 0): closed-form polynomial × √(π/(2x))·e^{-x}.
///   K_{-ν}(x) = K_ν(x) handles the negative half-integer case.
/// - Otherwise: reduce the order to μ ∈ [-1/2, 1/2], evaluate K_μ and K_{μ+1}
///   by Temme's small-x series or Steed's CF2 large-x continued fraction,
///   then use the stable upward recurrence to return K_ν.
pub fn bessel_k(nu: f64, x: f64) -> f64 {
    assert!(x > 0.0 && x.is_finite(), "bessel_k requires finite x > 0");
    assert!(nu.is_finite(), "bessel_k requires finite ν");
    let nu_abs = nu.abs(); // K_{-ν} = K_ν

    // Half-integer fast path.
    let two_nu = 2.0 * nu_abs;
    let n_round = two_nu.round();
    if (two_nu - n_round).abs() < 1e-12 && (n_round as i64) % 2 == 1 {
        let n = ((n_round as i64 - 1) / 2) as usize; // ν = n + 1/2
        return bessel_k_half_integer(n, x);
    }

    bessel_k_bessik(nu_abs, x)
}

pub(crate) const BESSEL_K_EPS: f64 = 1.0e-15;
pub(crate) const BESSEL_K_MAX_ITER: usize = 10_000;
pub(crate) const BESSEL_K_CHEB_C1: [f64; 7] = [
    -1.142_022_680_371_168,
    6.516_511_267_073_7e-3,
    3.087_090_173_086e-4,
    -3.470_626_964_9e-6,
    6.943_766_4e-9,
    3.677_95e-11,
    -1.356e-13,
];
pub(crate) const BESSEL_K_CHEB_C2: [f64; 8] = [
    1.843_740_587_300_905,
    -7.685_284_084_478_67e-2,
    1.271_927_136_654_6e-3,
    -4.971_736_704_2e-6,
    -3.312_611_98e-8,
    2.423_096e-10,
    -1.702e-13,
    -1.49e-15,
];

pub(crate) fn bessel_k_bessik(nu: f64, x: f64) -> f64 {
    let nl = (nu + 0.5).floor() as usize;
    let mu = nu - nl as f64;
    let (mut rkmu, mut rk1) = if x <= 2.0 {
        bessel_k_temme(mu, x)
    } else {
        bessel_k_steed_cf2(mu, x)
    };

    for j in 1..=nl {
        let order = mu + j as f64;
        let rk_next = rk1 * (2.0 * order) / x + rkmu;
        rkmu = rk1;
        rk1 = rk_next;
    }
    rkmu
}

/// K_{n+1/2}(x) = √(π/(2x)) · e^{-x} · Σ_{k=0}^{n} (n+k)! / (k!(n-k)!) · (2x)^{-k}.
pub(crate) fn bessel_k_half_integer(n: usize, x: f64) -> f64 {
    let pref = (std::f64::consts::PI / (2.0 * x)).sqrt() * (-x).exp();
    let mut sum = 0.0_f64;
    let two_x = 2.0 * x;
    for k in 0..=n {
        // (n+k)! / (k! (n-k)!)
        let num = factorial_f64(n + k);
        let den = factorial_f64(k) * factorial_f64(n - k);
        sum += num / den / two_x.powi(k as i32);
    }
    pref * sum
}

pub(crate) fn bessel_k_temme(mu: f64, x: f64) -> (f64, f64) {
    let half_x = 0.5 * x;
    let mu2 = mu * mu;
    let pimu = std::f64::consts::PI * mu;
    let fact = if pimu.abs() < BESSEL_K_EPS {
        1.0
    } else {
        pimu / pimu.sin()
    };
    let dlog = -half_x.ln();
    let sigma = mu * dlog;
    let fact2 = if sigma.abs() < BESSEL_K_EPS {
        1.0
    } else {
        sigma.sinh() / sigma
    };
    let (gam1, gam2, gampl, gammi) = bessel_k_beschb(mu);
    let mut ff = fact * (gam1 * sigma.cosh() + gam2 * fact2 * dlog);
    let mut sum = ff;
    let exp_sigma = sigma.exp();
    let mut p = 0.5 * exp_sigma / gampl;
    let mut q = 0.5 / (exp_sigma * gammi);
    let mut c = 1.0_f64;
    let d = half_x * half_x;
    let mut sum1 = p;

    for i in 1..=BESSEL_K_MAX_ITER {
        let i_f = i as f64;
        ff = (i_f * ff + p + q) / (i_f * i_f - mu2);
        c *= d / i_f;
        p /= i_f - mu;
        q /= i_f + mu;
        let del = c * ff;
        sum += del;
        let del1 = c * (p - i_f * ff);
        sum1 += del1;
        if del.abs() < BESSEL_K_EPS * sum.abs() {
            return (sum, sum1 * 2.0 / x);
        }
    }
    // SAFETY: Temme's series converges geometrically for |μ| ≤ 1/2 and
    // 0 < x ≤ 2 (the reduced-order, small-x branch entered by
    // `bessel_k_bessik`). Public entry `bessel_k` asserts finite ν and
    // 0 < x finite, and `nl = (|ν| + 1/2).floor()` produces μ = |ν| − nl
    // in [−1/2, 1/2). With BESSEL_K_EPS = 1e-15 the term ratio
    // |del_{i+1}/del_i| ~ (x/2)² / i² drops below ε within ~40
    // iterations for x ≤ 2; BESSEL_K_MAX_ITER = 10_000 is an
    // overdetermined defensive cap whose only reachable trigger would
    // be invariant violation upstream. Emitting any finite substitute
    // here would silently corrupt the penalty matrix.
    panic!("bessel_k Temme series failed to converge for mu={mu} x={x}");
}

pub(crate) fn bessel_k_steed_cf2(mu: f64, x: f64) -> (f64, f64) {
    let mut b = 2.0 * (1.0 + x);
    let mut d = 1.0 / b;
    let mut delh = d;
    let mut h = delh;
    let mut q1 = 0.0_f64;
    let mut q2 = 1.0_f64;
    let a1 = 0.25 - mu * mu;
    let mut q = a1;
    let mut c = a1;
    let mut a = -a1;
    let mut s = 1.0 + q * delh;

    for i in 2..=BESSEL_K_MAX_ITER {
        let i_f = i as f64;
        a -= 2.0 * (i_f - 1.0);
        c = -a * c / i_f;
        let qnew = (q1 - b * q2) / a;
        q1 = q2;
        q2 = qnew;
        q += c * qnew;
        b += 2.0;
        d = 1.0 / (b + a * d);
        delh *= b * d - 1.0;
        h += delh;
        let dels = q * delh;
        s += dels;
        if dels.abs() < BESSEL_K_EPS * s.abs() {
            h *= a1;
            let rkmu = (std::f64::consts::PI / (2.0 * x)).sqrt() * (-x).exp() / s;
            let rk1 = rkmu * (mu + x + 0.5 - h) / x;
            return (rkmu, rk1);
        }
    }
    // SAFETY: Steed's CF2 (NR §6.7) converges uniformly for |μ| ≤ 1/2 at
    // x > 2 — the reduced-order, large-x branch routed in by
    // `bessel_k_bessik`. With x > 2, the modified Lentz tail shrinks as
    // ~(x+i)^{-1}, reaching BESSEL_K_EPS = 1e-15 in well under 100
    // iterations for x up to 100. BESSEL_K_MAX_ITER = 10_000 is a defensive
    // cap whose only reachable trigger would be invariant violation upstream
    // (public `bessel_k` asserts finite ν and 0 < x finite). Emitting any
    // finite substitute here would silently corrupt the penalty matrix.
    panic!("bessel_k Steed CF2 failed to converge for mu={mu} x={x}");
}

pub(crate) fn bessel_k_beschb(mu: f64) -> (f64, f64, f64, f64) {
    let xx = 8.0 * mu * mu - 1.0;
    let gam1 = chebyshev_eval_minus1_to_1(&BESSEL_K_CHEB_C1, xx);
    let gam2 = chebyshev_eval_minus1_to_1(&BESSEL_K_CHEB_C2, xx);
    let gampl = gam2 - mu * gam1;
    let gammi = gam2 + mu * gam1;
    (gam1, gam2, gampl, gammi)
}

pub(crate) fn chebyshev_eval_minus1_to_1(coeffs: &[f64], x: f64) -> f64 {
    let mut d = 0.0_f64;
    let mut dd = 0.0_f64;
    let y2 = 2.0 * x;
    for j in (1..coeffs.len()).rev() {
        let previous_d = d;
        d = y2 * d - dd + coeffs[j];
        dd = previous_d;
    }
    x * d - dd + 0.5 * coeffs[0]
}

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
pub fn riesz_kernel_value(d: usize, j: f64, r: f64) -> f64 {
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
                * gamma_fn(j)
                * factorial_f64(n);
            return sign / denom
                * r.powi((2 * n) as i32)
                * (r.ln() + log_riesz_finite_part_shift(d, n));
        }
    }

    // Non-log case (admits fractional `j`).
    let half_d = d as f64 / 2.0;
    let num = gamma_fn(half_d - j);
    let denom = 4.0_f64.powf(j) * std::f64::consts::PI.powf(half_d) * gamma_fn(j);
    num / denom * r.powf(2.0 * j - d as f64)
}

/// Canonical log-Riesz finite-part constant for
/// `R_{d/2+n}^d(r) = c_n r^{2n}(log r + A_n)`.
///
/// Applying the radial Laplacian gives
///
/// ```text
/// Δ[r^{2n}(log r + A_n)]
///   = 4n(n+d/2-1) r^{2n-2}
///     · (log r + A_n + (4n+d-2)/(4n(n+d/2-1))).
/// ```
///
/// Since the constants satisfy
/// `c_n 4n(n+d/2-1) = -c_{n-1}`, exact distributional recurrence
/// `Δ R_{d/2+n}^d = -R_{d/2+n-1}^d` requires
/// `A_n = A_{n-1} - (4n+d-2)/(4n(n+d/2-1))`, with `A_0 = 0`.
pub(crate) fn log_riesz_finite_part_shift(d: usize, n: usize) -> f64 {
    let half_d = 0.5 * d as f64;
    let mut shift = 0.0_f64;
    for t in 1..=n {
        let tf = t as f64;
        shift -= (4.0 * tf + d as f64 - 2.0) / (4.0 * tf * (tf + half_d - 1.0));
    }
    shift
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
pub fn matern_kernel_value(d: usize, ell: usize, kappa: f64, r: f64) -> f64 {
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
        - ln_gamma(ell as f64);
    let pref = ln_pref.exp();

    if r == 0.0 {
        // M(0) = pref · lim_{r→0} r^{ℓ - d/2} K_{ℓ - d/2}(κr)
        // For ν > 0: K_ν(x) ~ Γ(ν)/2 (x/2)^{-ν}, so r^ν K_ν(κr) → Γ(ν)/2 (κ/2)^{-ν}.
        // For ν < 0 (i.e. ℓ < d/2): r^ν · K_{-|ν|}(κr) ~ r^ν · Γ(|ν|)/2 (κr/2)^{-|ν|}
        //   → Γ(|ν|)/2 (κ/2)^{-|ν|} · r^{ν - |ν|} = ∞ (singular). Return ∞.
        // For ν = 0: K_0(x) ~ -log(x/2) - γ; r^0·K_0(κr) → ∞. Return ∞.
        if nu > 0.0 {
            let lim = 0.5 * gamma_fn(nu) * (0.5 * kappa).powf(-nu);
            return pref * lim;
        } else {
            return f64::INFINITY;
        }
    }

    let kr = kappa * r;
    let kv = bessel_k(nu, kr);
    pref * r.powf(nu) * kv
}

pub(crate) const DUCHON_SMALL_CHI_SERIES_MAX: f64 = 0.125;
const DUCHON_SMALL_CHI_SERIES_MAX_TERMS: usize = 96;
pub(crate) const DUCHON_SMALL_CHI_SERIES_REL_TOL: f64 = 4.0e-16;

#[inline]
pub(crate) fn use_duchon_small_chi_riesz_series(kappa: f64, r: f64) -> bool {
    kappa > 0.0
        && kappa.is_finite()
        && r > 0.0
        && r.is_finite()
        && (kappa * r).abs() <= DUCHON_SMALL_CHI_SERIES_MAX
}

/// Small-χ Riesz-series chart for
/// `F^{-1}{ρ^{-2a}(κ²+ρ²)^{-b}}`.
///
/// Expanding at high frequency gives
///
/// ```text
/// Σ_n (-1)^n C(b+n-1,n) κ^{2n} R_{a+b+n}^d(R).
/// ```
///
/// When `d > 2(a+b)`, the low-frequency mass is uniformly integrable and
/// this is the true pointwise positive-κ kernel for small χ. In singular
/// regimes this same chart is the constrained Duchon finite-part
/// representative after quotienting the polynomial nullspace. Either way,
/// this avoids the catastrophic Riesz/Matérn partial-fraction cancellation
/// that appears as κR→0.
///
/// This helper returns radial R-derivatives of that same series and,
/// with `kappa_derivative_order` set to 1 or 2, the corresponding
/// analytic κ partials. It is the shared value/η/κ source for the
/// cancellation basin; production never differentiates it numerically.
pub(crate) fn duchon_small_chi_riesz_series_radial_derivatives(
    d: usize,
    a: usize,
    b: usize,
    kappa: f64,
    r: f64,
    max_order: usize,
    kappa_derivative_order: usize,
) -> Vec<f64> {
    assert!(b >= 1);
    assert!(
        kappa > 0.0,
        "matern kernel derivative requires kappa > 0: kappa={kappa}"
    );
    assert!(r > 0.0, "matern kernel derivative requires r > 0: r={r}");
    assert!(
        kappa_derivative_order <= 2,
        "matern kernel derivative supports kappa_derivative_order <= 2: order={kappa_derivative_order}"
    );

    let mut total = vec![KahanSum::default(); max_order + 1];
    let mut coeff = 1.0_f64;
    let kappa_sq = kappa * kappa;
    let base = a + b;

    let mut prev_term_norm = f64::INFINITY;
    let mut saw_nonzero_term = false;
    for n in 0..DUCHON_SMALL_CHI_SERIES_MAX_TERMS {
        // Closed-form k-th derivative w.r.t. kappa of the term
        // (kappa^2)^n in the Riesz small-chi series. We only ever invoke
        // this function with kappa_derivative_order ∈ {0, 1, 2} (asserted
        // above); the formula generalizes uniformly so a future caller
        // requesting order k computes (2n)·(2n−1)·…·(2n−k+1) / kappa^k.
        // For k=2 this collapses to p·(p−1)/kappa², matching the
        // pre-refactor explicit branch (kappa_sq = kappa·kappa).
        let kappa_factor = if kappa_derivative_order == 0 {
            1.0
        } else if n == 0 {
            0.0
        } else {
            let p = 2.0 * n as f64;
            let mut numerator = 1.0_f64;
            for j in 0..kappa_derivative_order {
                numerator *= p - j as f64;
            }
            let denom = kappa.powi(kappa_derivative_order as i32);
            numerator / denom
        };

        let scale = coeff * kappa_factor;
        let block = if kappa_factor == 0.0 {
            None
        } else {
            Some(riesz_block_radial_derivatives(
                d,
                (base + n) as f64,
                r,
                max_order,
            ))
        };
        let term_norm = block
            .as_ref()
            .map(|values| {
                values
                    .iter()
                    .map(|&value| (scale * value).abs())
                    .fold(0.0_f64, f64::max)
            })
            .unwrap_or(0.0);

        if term_norm > 0.0 {
            if saw_nonzero_term && term_norm > prev_term_norm {
                break;
            }
            saw_nonzero_term = true;
            prev_term_norm = term_norm;
        }

        if let Some(block) = block {
            for (order, value) in block.into_iter().enumerate() {
                total[order].add(scale * value);
            }
        }

        let total_norm = total
            .iter()
            .map(|acc| acc.sum().abs())
            .fold(0.0_f64, f64::max);
        if n >= 4 && term_norm <= DUCHON_SMALL_CHI_SERIES_REL_TOL * total_norm.max(1.0) {
            break;
        }

        coeff *= -((b + n) as f64) * kappa_sq / ((n + 1) as f64);
    }

    total.iter().map(|acc| acc.sum()).collect()
}

pub(crate) fn duchon_small_chi_riesz_series_value(d: usize, a: usize, b: usize, kappa: f64, r: f64) -> f64 {
    duchon_small_chi_riesz_series_radial_derivatives(d, a, b, kappa, r, 0, 0)[0]
}

/// Hybrid isotropic Duchon penalty
/// g_q^iso(R; m, s, κ) = F^{-1}{1/(ρ^{2(2m-q)} (κ² + ρ²)^{2s})}(R).
///
/// This returns the canonical constrained Duchon representative: polynomial
/// nullspace components are quotiented out, and the small-κR chart evaluates
/// the matching finite-part Riesz series directly. The ordinary
/// partial-fraction Green's function and this representative differ by
/// nullspace terms in low-dimensional singular regimes, but the constrained
/// fit only sees this representative. Value, radial derivatives, and κ
/// partials all use the same chart switch, so production never mixes a
/// stable value formula with cancelled derivative formulas.
///
/// Edge cases:
/// - s = 0: g_q^iso(R) = R_{2m-q}^d(R) (no Matérn factor).
/// - κ = 0, s ≥ 1: g_q^iso(R) = R_{2m+2s-q}^d(R) (Riesz pure).
/// - General: small-κR finite-part Riesz series or, outside that chart,
///   partial-fraction decomposition with a = 2m - q, b = 2s.
///
/// Requires a := 2m - q ≥ 1.
pub fn isotropic_duchon_penalty(q: usize, d: usize, m: usize, s: f64, kappa: f64, r: f64) -> f64 {
    assert!(2 * m >= q + 1, "isotropic_duchon_penalty: need 2m - q ≥ 1");
    assert!(
        s.is_finite() && s >= 0.0,
        "isotropic_duchon_penalty: s must be finite and ≥ 0, got {s}"
    );
    let a = 2 * m - q;

    if s == 0.0 {
        return riesz_kernel_value(d, a as f64, r);
    }
    if kappa == 0.0 {
        // Pure-Riesz scale-free: fractional `s` rides directly into
        // the kernel via `j = a + 2s`. No partial-fraction expansion
        // needed (that path is for the hybrid Matérn-blend regime
        // below).
        return riesz_kernel_value(d, a as f64 + 2.0 * s, r);
    }

    // Hybrid Matérn-blend (κ > 0) uses the partial-fraction
    // expansion with integer `b = 2s`. Fractional `s` is not yet
    // supported on this branch — its expansion uses integer
    // binomials and powers of `κ²` that have no clean fractional
    // generalisation. Reject up front rather than silently
    // truncating.
    assert!(
        s.fract() == 0.0,
        "isotropic_duchon_penalty: hybrid Matérn (κ > 0) requires integer s, got {s}"
    );
    let s_int = s as usize;
    let b = 2 * s_int;
    if use_duchon_small_chi_riesz_series(kappa, r) {
        return duchon_small_chi_riesz_series_value(d, a, b, kappa, r);
    }

    let kappa_sq = kappa * kappa;

    // A_j = (-1)^{a-j} · C(a+b-j-1, a-j) · κ^{-2(a+b-j)}, j = 1..a
    let mut sum = KahanSum::default();
    for j in 1..=a {
        let sign = if (a - j).is_multiple_of(2) { 1.0 } else { -1.0 };
        let binom = binomial_f64(a + b - j - 1, a - j);
        let coeff = sign * binom * kappa_sq.powi(-((a + b - j) as i32));
        let term = coeff * riesz_kernel_value(d, j as f64, r);
        sum.add(term);
    }

    // B_ℓ = (-1)^a · C(a+b-ℓ-1, b-ℓ) · κ^{-2(a+b-ℓ)}, ℓ = 1..b
    let sign_a = if a.is_multiple_of(2) { 1.0 } else { -1.0 };
    for ell in 1..=b {
        let binom = binomial_f64(a + b - ell - 1, b - ell);
        let coeff = sign_a * binom * kappa_sq.powi(-((a + b - ell) as i32));
        let term = coeff * matern_kernel_value(d, ell, kappa, r);
        sum.add(term);
    }

    sum.sum()
}

/// Analytic anisotropic Duchon pair-block kernel.
///
/// The historical implementation evaluated the Schoenberg heat integral
/// numerically. Production now uses the exact radial identity
/// `g_q(z) = (-Δ_B)^q f(|z|)` with closed-form radial derivatives of the
/// isotropic Riesz/Matérn hybrid kernel.
pub fn anisotropic_duchon_penalty(
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
        "anisotropic_duchon_penalty: eta and r dimension mismatch"
    );
    assert!(!r.is_empty(), "anisotropic_duchon_penalty: empty input");
    assert!(q <= 2, "anisotropic_duchon_penalty: q must be in {{0,1,2}}");
    anisotropic_duchon_penalty_radial(q, m, s, kappa, eta, r)
}

/// Bundled value + first/second derivatives of the anisotropic pair-block
/// kernel `J · g_q(z; η, m, s, κ)` with `J = exp(Σ η_k)`.
///
/// All entries already include the `J` prefactor and its derivatives, so
/// callers should consume them directly. Mixed second derivatives are
/// stored in a square `d_eta.len() × d_eta.len()` matrix and are
/// symmetric by construction.
#[derive(Debug, Clone)]
pub struct PairBlockBundle {
    pub value: f64,
    pub d_eta: Vec<f64>,
    pub d_kappa: f64,
    pub d2_eta: Vec<Vec<f64>>,
    pub d2_eta_kappa: Vec<f64>,
    pub d2_kappa: f64,
}

/// Per-axis powers of the anisotropic metric `B = diag(exp(-2η))`.
///
/// The pair-block hot loops need `b`, `b²`, and `b³` for every pair.
/// Precomputing them once per penalty/operator removes one `exp()` per
/// axis per pair without changing the analytic radial formula.
#[derive(Debug, Clone)]
pub(crate) struct AnisoMetricPowers {
    pub(crate) b: Vec<f64>,
    pub(crate) b2: Vec<f64>,
    pub(crate) b3: Vec<f64>,
}

impl AnisoMetricPowers {
    pub(crate) fn new(eta: &[f64]) -> Self {
        let mut b = Vec::with_capacity(eta.len());
        let mut b2 = Vec::with_capacity(eta.len());
        let mut b3 = Vec::with_capacity(eta.len());
        for &e in eta {
            let v = (-2.0 * e).exp();
            let v2 = v * v;
            b.push(v);
            b2.push(v2);
            b3.push(v2 * v);
        }
        Self { b, b2, b3 }
    }

    #[inline(always)]
    pub(crate) fn assert_dim(&self, dim: usize) {
        assert_eq!(self.b.len(), dim);
        assert_eq!(self.b2.len(), dim);
        assert_eq!(self.b3.len(), dim);
    }
}

/// Exact `R = 0` self-pair for the convergent Schoenberg/spectral kernel.
///
/// At zero lag the heat kernel contributes `(4πτ)^(-d/2)` and the
/// anisotropic operator contributes only traces of `B = diag(exp(-2η))`:
///
/// ```text
/// C_0(B) = 1
/// C_1(B) = tr(B) / 2
/// C_2(B) = (tr(B)^2 + 2 tr(B^2)) / 4
/// ```
///
/// The remaining one-dimensional integral is
///
/// ```text
/// C_q(B) (4π)^(-d/2) / Γ(2s)
///   ∫_0^∞ τ^(2(m+s)-d/2-q-1) exp(-κ²τ) Γ(2s, as density) dτ
/// = C_q(B) (4π)^(-d/2)
///   Γ(λ) Γ(μ) / (Γ(2s) Γ(d/2+q)) κ^(-2λ),
/// ```
///
/// with `λ = 2(m+s) - d/2 - q` and `μ = d/2 + q - 2m`.
/// The branch is valid exactly when `λ > 0` (UV convergence) and
/// `μ > 0` (IR convergence). The returned bundle includes the external
/// `J = exp(Ση)` prefactor and its product-rule derivatives, so callers
/// can use it in the same slots as `pair_block_radial_with_j_second_derivatives`.
pub(crate) fn schoenberg_self_pair_bundle(
    q: usize,
    m: usize,
    s: usize,
    kappa: f64,
    eta: &[f64],
) -> Option<PairBlockBundle> {
    let d = eta.len();
    if q > 2 || s == 0 || !(kappa > 0.0) || !kappa.is_finite() {
        return None;
    }

    let half_d = 0.5 * d as f64;
    let order = 2 * (m + s);
    let s_total = 2 * s;
    let lambda = order as f64 - half_d - q as f64;
    let mu = s_total as f64 - lambda;
    if !(lambda > 0.0 && mu > 0.0) {
        return None;
    }

    let mut s1 = 0.0_f64;
    let mut s2 = 0.0_f64;
    let mut b = vec![0.0_f64; d];
    for (axis, &e) in eta.iter().enumerate() {
        let bb = (-2.0 * e).exp();
        b[axis] = bb;
        s1 += bb;
        s2 += bb * bb;
    }

    let (c_eta, c_eta_grad, c_eta_hess) = match q {
        0 => (1.0, vec![0.0_f64; d], vec![vec![0.0_f64; d]; d]),
        1 => {
            let mut grad = vec![0.0_f64; d];
            let mut hess = vec![vec![0.0_f64; d]; d];
            for k in 0..d {
                grad[k] = -b[k];
                hess[k][k] = 2.0 * b[k];
            }
            (0.5 * s1, grad, hess)
        }
        2 => {
            let mut grad = vec![0.0_f64; d];
            let mut hess = vec![vec![0.0_f64; d]; d];
            for k in 0..d {
                grad[k] = -s1 * b[k] - 2.0 * b[k] * b[k];
                for l in 0..d {
                    hess[k][l] = if k == l {
                        2.0 * s1 * b[k] + 10.0 * b[k] * b[k]
                    } else {
                        2.0 * b[k] * b[l]
                    };
                }
            }
            (0.25 * (s1 * s1 + 2.0 * s2), grad, hess)
        }
        // `q > 2` is rejected by the early guard above; this arm
        // therefore cannot run. Returning `None` keeps the typed
        // contract intact rather than panicking.
        _ => return None,
    };
    if c_eta == 0.0 || !c_eta.is_finite() {
        return None;
    }

    let log_base = -half_d * (4.0 * std::f64::consts::PI).ln() + ln_gamma(lambda) + ln_gamma(mu)
        - ln_gamma(s_total as f64)
        - ln_gamma(half_d + q as f64)
        - 2.0 * lambda * kappa.ln();
    let base = log_base.exp();
    if !base.is_finite() {
        return None;
    }

    let exponent = -2.0 * lambda;
    let g = base * c_eta;
    let g_kappa = exponent * g / kappa;
    let g_kappa2 = exponent * (exponent - 1.0) * g / (kappa * kappa);
    let mut g_eta = vec![0.0_f64; d];
    let mut g_eta2 = vec![vec![0.0_f64; d]; d];
    let mut g_eta_kappa = vec![0.0_f64; d];
    for k in 0..d {
        g_eta[k] = base * c_eta_grad[k];
        g_eta_kappa[k] = exponent * g_eta[k] / kappa;
        for l in 0..d {
            g_eta2[k][l] = base * c_eta_hess[k][l];
        }
    }

    let big_j = eta.iter().sum::<f64>().exp();
    let value = big_j * g;
    let d_eta = (0..d).map(|k| big_j * (g + g_eta[k])).collect();
    let d_kappa = big_j * g_kappa;
    let d2_eta = (0..d)
        .map(|k| {
            (0..d)
                .map(|l| big_j * (g + g_eta[k] + g_eta[l] + g_eta2[k][l]))
                .collect::<Vec<f64>>()
        })
        .collect();
    let d2_eta_kappa = (0..d).map(|k| big_j * (g_kappa + g_eta_kappa[k])).collect();
    let d2_kappa = big_j * g_kappa2;

    Some(PairBlockBundle {
        value,
        d_eta,
        d_kappa,
        d2_eta,
        d2_eta_kappa,
        d2_kappa,
    })
}

pub(crate) fn hybrid_self_pair_radial_derivative_with_kappa_derivs_odd_d(
    q: usize,
    m: usize,
    s: usize,
    d: usize,
    kappa: f64,
) -> Option<(f64, f64, f64)> {
    if d % 2 != 1 || q > 2 || !(kappa > 0.0) || !kappa.is_finite() {
        return None;
    }
    let smoothness_order = 2 * (m + s);
    let required = d + 2 * q;
    if smoothness_order <= required {
        return None;
    }

    let length_scale = 1.0 / kappa;
    let coeffs = super::duchon_partial_fraction_coeffs(m, s, kappa);
    let f = super::duchon_phi_even_derivative_collision(length_scale, m, s, d, &coeffs, q).ok()?;
    if !f.is_finite() {
        return None;
    }

    // In odd dimension, every half-integer Matérn Taylor block and every
    // contributing Riesz block carries the same κ power after partial
    // fraction assembly:
    //   f^(2q)(0; κ) = C · κ^{d + 2q - 2(m+s)}.
    let exponent = d as f64 + 2.0 * q as f64 - 2.0 * (m + s) as f64;
    let f_kappa = exponent * f / kappa;
    let f_kappa2 = exponent * (exponent - 1.0) * f / (kappa * kappa);
    Some((f, f_kappa, f_kappa2))
}

pub(crate) fn hybrid_self_pair_bundle_odd_d(
    q: usize,
    m: usize,
    s: usize,
    kappa: f64,
    eta: &[f64],
) -> Option<PairBlockBundle> {
    let d = eta.len();
    let (f, f_kappa, f_kappa2) =
        hybrid_self_pair_radial_derivative_with_kappa_derivs_odd_d(q, m, s, d, kappa)?;

    let mut s1 = 0.0_f64;
    let mut s2 = 0.0_f64;
    let mut b = vec![0.0_f64; d];
    for (axis, &e) in eta.iter().enumerate() {
        let bb = (-2.0 * e).exp();
        b[axis] = bb;
        s1 += bb;
        s2 += bb * bb;
    }

    let (g, g_kappa, g_kappa2, g_eta, g_eta2, g_eta_kappa) = match q {
        0 => (
            f,
            f_kappa,
            f_kappa2,
            vec![0.0_f64; d],
            vec![vec![0.0_f64; d]; d],
            vec![0.0_f64; d],
        ),
        1 => {
            let mut g_eta = vec![0.0_f64; d];
            let mut g_eta2 = vec![vec![0.0_f64; d]; d];
            let mut g_eta_kappa = vec![0.0_f64; d];
            for k in 0..d {
                g_eta[k] = 2.0 * b[k] * f;
                g_eta2[k][k] = -4.0 * b[k] * f;
                g_eta_kappa[k] = 2.0 * b[k] * f_kappa;
            }
            (
                -s1 * f,
                -s1 * f_kappa,
                -s1 * f_kappa2,
                g_eta,
                g_eta2,
                g_eta_kappa,
            )
        }
        2 => {
            let a = s1 * s1 + 2.0 * s2;
            let mut g_eta = vec![0.0_f64; d];
            let mut g_eta2 = vec![vec![0.0_f64; d]; d];
            let mut g_eta_kappa = vec![0.0_f64; d];
            for k in 0..d {
                let a_k = -4.0 * s1 * b[k] - 8.0 * b[k] * b[k];
                g_eta[k] = f * a_k / 3.0;
                g_eta_kappa[k] = f_kappa * a_k / 3.0;
                for l in 0..d {
                    let a_kl = if k == l {
                        8.0 * s1 * b[k] + 40.0 * b[k] * b[k]
                    } else {
                        8.0 * b[k] * b[l]
                    };
                    g_eta2[k][l] = f * a_kl / 3.0;
                }
            }
            (
                f * a / 3.0,
                f_kappa * a / 3.0,
                f_kappa2 * a / 3.0,
                g_eta,
                g_eta2,
                g_eta_kappa,
            )
        }
        _ => return None,
    };

    let big_j = eta.iter().sum::<f64>().exp();
    let value = big_j * g;
    let d_eta = (0..d).map(|k| big_j * (g + g_eta[k])).collect();
    let d_kappa = big_j * g_kappa;
    let d2_eta = (0..d)
        .map(|k| {
            (0..d)
                .map(|l| big_j * (g + g_eta[k] + g_eta[l] + g_eta2[k][l]))
                .collect::<Vec<f64>>()
        })
        .collect();
    let d2_eta_kappa = (0..d).map(|k| big_j * (g_kappa + g_eta_kappa[k])).collect();
    let d2_kappa = big_j * g_kappa2;

    Some(PairBlockBundle {
        value,
        d_eta,
        d_kappa,
        d2_eta,
        d2_eta_kappa,
        d2_kappa,
    })
}

/// Exact zero-lag bundle for every analytic self-pair regime currently
/// supported by the closed-form Duchon path. This is the single
/// production entry point for diagonal values and diagonal η/κ
/// derivatives: callers should not hand-code the Schoenberg or odd-d
/// collision branches separately.
pub(crate) fn analytic_self_pair_bundle(
    q: usize,
    m: usize,
    s: usize,
    kappa: f64,
    eta: &[f64],
) -> Option<PairBlockBundle> {
    schoenberg_self_pair_bundle(q, m, s, kappa, eta)
        .or_else(|| hybrid_self_pair_bundle_odd_d(q, m, s, kappa, eta))
}

// ============================================================
//  Radial-derivative anisotropic penalty
// ------------------------------------------------------------
//  The pair-block kernel can be written as
//      g_q(z; η, m, s, κ) = (-Δ_B)^q f(R)
//  with R = |z|, B = diag(b_k), b_k = exp(-2 η_k),
//  Δ_B = Σ_k b_k ∂²/∂z_k² (anisotropic Laplacian) and
//  f(R) = isotropic_duchon_penalty(0, d, m, s as f64, κ, R).
//  Matrix entries multiply this bare kernel by J = exp(Σ η_k).
//
//  Acting on a radial function we get the structured forms:
//      Δ_B f(R) = f''(R)·u_1/R²  +  f'(R)·(s_1/R - u_1/R³)
//  with s_p = Σ b_k^p, u_p = Σ b_k^p z_k².  Applying Δ_B again
//  yields (derivation in the inline comments below):
//      Δ_B² f = u_1²·[ f''''/R⁴ - 6 f'''/R⁵ + 15 f''/R⁶ - 15 f'/R⁷ ]
//             + s_1 u_1·[ 2 f'''/R³ - 6 f''/R⁴ + 6 f'/R⁵ ]
//             + s_1² ·[ f''/R² - f'/R³ ]
//             + u_2  ·[ 4 f'''/R³ - 12 f''/R⁴ + 12 f'/R⁵ ]
//             + s_2  ·[ 2 f''/R² - 2 f'/R³ ]
//  In the isotropic limit (b_k = 1) this collapses to the standard
//      Δ²f = f'''' + 2(d-1)/R · f''' + (d-1)(d-3)/R² · f'' - (d-1)(d-3)/R³ · f'
//  as a sanity check.
//
//  Compared to the former heat-integral quadrature, this radial form is
//   * finite pointwise for any R > 0 in any (m, s, d) regime
//     (no IR divergence requirement),
//   * a few hundred FLOP per pair,
//   * exact up to roundoff in the radial-derivative table.
// ============================================================

/// Half-width (in Bessel-order shifts) supported by the cached
/// `bessel_k_table_around` lookup. Each successive ∂_R application of
/// the recurrence `K_b' = -(K_{b-1} + K_{b+1})/2` widens the required
/// Bessel-order range by 1; supporting `f^{(max_order)}` therefore
/// requires `max_order` shifts on either side of ν.
pub(crate) const BESSEL_TABLE_HALF_WIDTH: i32 = 6;

/// Bundle of values `K_{ν+i}(x)` for `−H ≤ i ≤ H`, where
/// `H = BESSEL_TABLE_HALF_WIDTH`. Returns a length-`(2H + 1)` vector
/// indexed so that `out[i + H] = K_{ν+i}(x)`. Used to combine into
/// derivatives of `r^a · K_b(κr)` via the recurrence
/// `K_b' = -(K_{b-1} + K_{b+1})/2`; supporting `f^{(k)}` requires
/// `H ≥ k`.
pub(crate) fn bessel_k_table_around(nu: f64, x: f64) -> Vec<f64> {
    let h = BESSEL_TABLE_HALF_WIDTH;
    let len = (2 * h + 1) as usize;
    let mut out = vec![0.0_f64; len];
    for i in -h..=h {
        let order = nu + i as f64;
        out[(i + h) as usize] = bessel_k(order, x);
    }
    out
}

/// Radial derivatives of a single Matérn block
///   m_ℓ(r) = pref(ℓ, κ, d) · r^ν · K_ν(κr),    ν = ℓ - d/2,
///   pref = κ^{d/2 - ℓ} / ((2π)^{d/2} · 2^{ℓ-1} · Γ(ℓ)).
///
/// Returns `[m, m', m'', m''', m^{(4)}]` at the supplied `r > 0`.
///
/// Algorithm:
/// We expand the Matérn block as a linear combination of
/// `c · r^a · K_b(κr)` terms.  Differentiation is closed:
///   d/dr[c r^a K_b(κr)] = (c·a) r^{a-1} K_b(κr)
///                       + (-c κ/2) r^a K_{b-1}(κr)
///                       + (-c κ/2) r^a K_{b+1}(κr).
/// Each step triples the term count.  For the 4 derivatives we
/// need (max order = 4) the term list grows to at most 3^4 = 81
/// entries while staying allocation-free per pair.
pub(crate) fn matern_block_radial_derivatives(
    d: usize,
    ell: usize,
    kappa: f64,
    r: f64,
    max_order: usize,
) -> Vec<f64> {
    assert!(d >= 1, "matern block requires dimension >= 1: d={d}");
    assert!(ell >= 1, "matern block requires ell >= 1: ell={ell}");
    assert!(
        kappa > 0.0,
        "matern block requires kappa > 0: kappa={kappa}"
    );
    assert!(r > 0.0, "matern block requires r > 0: r={r}");
    assert!(
        max_order <= 6,
        "matern_block_radial_derivatives: max_order ≤ 6"
    );

    let nu = ell as f64 - d as f64 / 2.0;
    // Prefactor κ^{d/2-ℓ} / ((2π)^{d/2} 2^{ℓ-1} Γ(ℓ))
    let ln_pref = (d as f64 / 2.0 - ell as f64) * kappa.ln()
        - (d as f64 / 2.0) * (2.0 * std::f64::consts::PI).ln()
        - (ell as f64 - 1.0) * std::f64::consts::LN_2
        - ln_gamma(ell as f64);
    let pref = ln_pref.exp();

    // Symbolic term list: each entry (coef, a, b_offset) means
    //   coef · r^a · K_{ν + b_offset}(κr)
    // Initially: pref · r^ν · K_ν(κr) → (pref, ν, 0).
    let mut terms: Vec<(f64, f64, i32)> = vec![(pref, nu, 0)];
    let mut out = Vec::with_capacity(max_order + 1);

    // Cache bessel table at this r once we know which orders we need.
    // Each radial-derivative step widens the required Bessel-order
    // range by 1, so for `max_order ≤ H = BESSEL_TABLE_HALF_WIDTH`
    // shifts ±H around ν are sufficient.
    let kr = kappa * r;
    let bessel_table = bessel_k_table_around(nu, kr);
    let half_width = BESSEL_TABLE_HALF_WIDTH;
    let bessel = |b_off: i32| -> f64 {
        // b_off ∈ [-H, H] for the supported max_order range.
        let idx = (b_off + half_width) as usize;
        bessel_table[idx]
    };

    // Evaluate current term list at scalar r. When d is even, ν is
    // an integer and every successive derivative leaves `a` integer-
    // valued; we then use `r.powi` (≈5–10× faster than `powf`).
    // For odd d (half-integer ν), fall back to `powf`.
    let a_is_integer = d.is_multiple_of(2);
    let evaluate = |terms: &Vec<(f64, f64, i32)>| -> f64 {
        let mut sum = 0.0_f64;
        if a_is_integer {
            for &(c, a, b) in terms {
                if c == 0.0 {
                    continue;
                }
                assert!(
                    a.fract() == 0.0,
                    "matern_block_radial_derivatives: expected integer exponent for even d"
                );
                sum += c * r.powi(a as i32) * bessel(b);
            }
        } else {
            for &(c, a, b) in terms {
                if c == 0.0 {
                    continue;
                }
                sum += c * r.powf(a) * bessel(b);
            }
        }
        sum
    };

    out.push(evaluate(&terms));

    for _ in 0..max_order {
        let mut next: Vec<(f64, f64, i32)> = Vec::with_capacity(terms.len() * 3);
        for &(c, a, b) in &terms {
            if c == 0.0 {
                continue;
            }
            // term1: derivative of r^a (a · r^{a-1}) · K_b
            if a != 0.0 {
                next.push((c * a, a - 1.0, b));
            }
            // term2: derivative of K_b(κr) = κ · K_b'(κr)
            //        = κ · -(K_{b-1} + K_{b+1})/2
            let coef = -c * kappa * 0.5;
            next.push((coef, a, b - 1));
            next.push((coef, a, b + 1));
        }
        terms = compress_terms(next);
        out.push(evaluate(&terms));
    }
    out
}

/// Merge terms with equal `(a, b)` exponents to keep the list short.
pub(crate) fn compress_terms(mut terms: Vec<(f64, f64, i32)>) -> Vec<(f64, f64, i32)> {
    terms.sort_by(|x, y| {
        x.2.cmp(&y.2)
            .then_with(|| x.1.partial_cmp(&y.1).unwrap_or(std::cmp::Ordering::Equal))
    });
    let mut out: Vec<(f64, f64, i32)> = Vec::with_capacity(terms.len());
    for (c, a, b) in terms {
        if let Some(last) = out.last_mut()
            && last.2 == b
            && (last.1 - a).abs() < 1e-15
        {
            last.0 += c;
            continue;
        }
        out.push((c, a, b));
    }
    out
}

/// Radial derivatives `[R^{(0)}, …, R^{(max_order)}]` of a single
/// Riesz block `R_j^d(r) = c · r^{2j-d}` (non-log) or
/// `c · r^{2n} · ln r` (log case `2j = d + 2n`).
pub(crate) fn riesz_block_radial_derivatives(d: usize, j: f64, r: f64, max_order: usize) -> Vec<f64> {
    assert!(d >= 1, "riesz block requires dimension >= 1: d={d}");
    assert!(
        j.is_finite() && j >= 1.0,
        "riesz_block: need j ≥ 1, got {j}"
    );
    assert!(r > 0.0, "riesz block requires r > 0: r={r}");

    let two_j = 2.0 * j;
    let half_d = d as f64 / 2.0;
    let mut out = Vec::with_capacity(max_order + 1);

    // Log case detection: `2j − d` is a non-negative even integer
    // (within ε). For fractional `j` this never fires.
    const LOG_EPS: f64 = 1e-12;
    let offset = two_j - d as f64;
    let log_case = offset >= -LOG_EPS && {
        let n_f = (offset / 2.0).round();
        n_f >= 0.0 && (n_f * 2.0 - offset).abs() < LOG_EPS
    };
    if log_case {
        // R_j^d(r) = c · r^{2n} · (ln r + A_n).
        let n_f = (offset / 2.0).round();
        let n = n_f as usize;
        let two_j_i = two_j.round() as i32;
        let sign = if n.is_multiple_of(2) { -1.0 } else { 1.0 };
        let denom = 2.0_f64.powi(two_j_i - 1)
            * std::f64::consts::PI.powf(d as f64 / 2.0)
            * gamma_fn(j)
            * factorial_f64(n);
        let c = sign / denom;
        let two_n = 2 * n;
        let shift = log_riesz_finite_part_shift(d, n);
        for k in 0..=max_order {
            let mut sum = 0.0_f64;
            for i in 0..=k {
                if i > two_n {
                    continue;
                }
                let binom = binomial_f64(k, i);
                let mut ff = 1.0_f64;
                for q in 0..i {
                    ff *= (two_n - q) as f64;
                }
                let r_pow = r.powi((two_n - i) as i32);
                let log_part = if k - i == 0 {
                    r.ln() + shift
                } else {
                    let m = (k - i) as i32;
                    let sign2 = if (m as usize) % 2 == 1 { 1.0 } else { -1.0 };
                    sign2 * factorial_f64((k - i) - 1) * r.powi(-m)
                };
                sum += binom * ff * r_pow * log_part;
            }
            out.push(c * sum);
        }
        return out;
    }

    // Non-log case: c · r^p with p = 2j − d (real-valued for fractional
    // j). Successive derivatives multiply by the current exponent and
    // decrement it by 1; with non-integer p we use `powf` throughout.
    let c =
        gamma_fn(half_d - j) / (4.0_f64.powf(j) * std::f64::consts::PI.powf(half_d) * gamma_fn(j));
    let mut coef = c;
    let mut exp = 2.0 * j - d as f64;
    out.push(coef * r.powf(exp));
    for _ in 0..max_order {
        // d/dr[c · r^p] = c·p · r^{p−1}
        coef *= exp;
        exp -= 1.0;
        out.push(coef * r.powf(exp));
    }
    out
}

/// Radial derivatives `[f, f', f'', …, f^{(max_order)}]`(R) of the
/// isotropic Duchon kernel `f(R) = isotropic_duchon_penalty(0, d, m, s as f64, κ, R)`.
///
/// Used by the radial-derivative anisotropic form.  Requires `R > 0`.
pub fn radial_derivatives_of_isotropic_duchon(
    d: usize,
    m: usize,
    s: f64,
    kappa: f64,
    r: f64,
    max_order: usize,
) -> Vec<f64> {
    assert!(
        r > 0.0,
        "radial_derivatives_of_isotropic_duchon: r must be > 0"
    );
    assert!(
        2 * m >= 1,
        "radial_derivatives_of_isotropic_duchon: need m ≥ 1"
    );
    assert!(
        max_order <= 6,
        "radial_derivatives_of_isotropic_duchon: max_order ≤ 6"
    );
    assert!(
        s.is_finite() && s >= 0.0,
        "radial_derivatives_of_isotropic_duchon: s must be finite and ≥ 0, got {s}"
    );

    // a = 2m (we differentiate f = isotropic Duchon at q=0; the q is
    // applied externally via Δ_B^q in g_q).  Match isotropic_duchon_penalty(q=0).
    let a = 2 * m;

    if s == 0.0 {
        // Pure Riesz: f = R_{a}^d(r)
        return riesz_block_radial_derivatives(d, a as f64, r, max_order);
    }
    if kappa == 0.0 {
        // Scale-free Duchon with fractional s rides directly into the
        // Riesz block-derivatives at the real-valued block order
        // `j = a + 2s` — `riesz_block_radial_derivatives` already
        // accepts fractional `j` via `r.powf(2j − d)` and the
        // log-case detector ε-bounds.
        return riesz_block_radial_derivatives(d, a as f64 + 2.0 * s, r, max_order);
    }

    // Hybrid Matérn-blend (κ > 0) still uses the integer partial-fraction
    // chain below; reject fractional s up front rather than truncating
    // silently.
    assert!(
        s.fract() == 0.0,
        "radial_derivatives_of_isotropic_duchon: hybrid Matérn (κ > 0) requires integer s, got {s}"
    );
    let s = s as usize;

    // Hybrid case (s ≥ 1, κ > 0). Three charts in priority order:
    //
    // 1. Small-χ Riesz series (`κr ≤ DUCHON_SMALL_CHI_SERIES_MAX`).
    //    Exact analytic finite-part representative at small κ. Converges
    //    spectrally in `κ²r²` so the tails decay geometrically; carries
    //    full f64 precision when applicable. Several test fixtures
    //    (`test_small_kappa_finite_part_chart_is_shared_by_value_radial_and_kappa_partials`)
    //    pin the production code to this chart at the boundary
    //    κ=0.01, r=1.3 because the value, radial-derivative, and
    //    κ-partial code paths must all use the *same* finite-part
    //    representative there. Dispatching to Schwinger first at high
    //    d would re-evaluate the integrand at a near-singular Matérn
    //    limit (κ_t = √(1-t)κ → 0) and produce wildly different
    //    numerics. So check small-χ first.
    //
    // 2. Schwinger Beta-form (`d > 4m`, fallback when small-χ does not
    //    apply). For high-d cases the alternating Riesz/Matérn
    //    partial-fraction expansion loses most of its precision to
    //    catastrophic cancellation: individual terms have magnitudes
    //    that scale with high powers of `1/r` (Riesz factors `r^{2j-d}`
    //    for j ≪ d/2) while the combined kernel is moderate, so the
    //    cumulative sum is dominated by IEEE-754 noise. The Beta-form
    //    Schwinger integral
    //      1/(|w|^{4m}(κ²+|w|²)^{2s})
    //        = (1/B(2m, 2s)) ∫_0^1 t^{2m-1}(1-t)^{2s-1}
    //                              (|w|² + (1-t)κ²)^{-2(m+s)} dt
    //    termwise-IFT'd is a Beta-weighted average of Matérn kernels
    //    with strictly non-negative integrand. The convergence gate
    //    `d > 4m` ensures the t→1 (Riesz limit) endpoint is integrable.
    //
    // 3. PF (low-d, no small-χ): the original alternating Riesz/Matérn
    //    expansion. Cancellation is mild for `d ≤ 4m`.
    let b = 2 * s;
    if use_duchon_small_chi_riesz_series(kappa, r) {
        return duchon_small_chi_riesz_series_radial_derivatives(d, a, b, kappa, r, max_order, 0);
    }
    if schwinger_radial_is_convergent(d, m) {
        return stable_hybrid_duchon_radial(d, m, s, kappa, r, max_order);
    }

    let kappa_sq = kappa * kappa;
    let mut total_acc = vec![KahanSum::default(); max_order + 1];
    for j in 1..=a {
        let sign = if (a - j).is_multiple_of(2) { 1.0 } else { -1.0 };
        let binom = binomial_f64(a + b - j - 1, a - j);
        let coeff = sign * binom * kappa_sq.powi(-((a + b - j) as i32));
        let block = riesz_block_radial_derivatives(d, (j) as f64, r, max_order);
        for (k, v) in block.into_iter().enumerate() {
            let term = coeff * v;
            total_acc[k].add(term);
        }
    }
    let sign_a = if a.is_multiple_of(2) { 1.0 } else { -1.0 };
    for ell in 1..=b {
        let binom = binomial_f64(a + b - ell - 1, b - ell);
        let coeff = sign_a * binom * kappa_sq.powi(-((a + b - ell) as i32));
        let block = matern_block_radial_derivatives(d, ell, kappa, r, max_order);
        for (k, v) in block.into_iter().enumerate() {
            let term = coeff * v;
            total_acc[k].add(term);
        }
    }
    total_acc.iter().map(|acc| acc.sum()).collect()
}

/// Radial-form bare anisotropic Duchon pair-block penalty
///   g_q(z; η, m, s, κ) = (-Δ_B)^q f(R)
/// implemented via analytic radial derivatives of f. Matrix assembly
/// multiplies this value by `J = exp(Ση)`.
///
/// Closed-form pure-Duchon self-pair value `g_q(0; η, m, s, κ=0)`,
/// implementing the math team's Letter A §3 finite-part formulas:
///
///   h_1(0) = −s_1 · F_{2,0},
///   h_2(0) =  (F_{4,0}/3) · (s_1² + 2 s_2),
///
/// where `F_{2q,0} = f^{(2q)}(0)` is the (2q)-th radial derivative of
/// the q = 0 kernel `f(R) = R_{2(m+s)}^d(R) = c · R^p` with
/// `p = 4(m+s) − d`. For non-log cases (`p` not a non-negative even
/// integer):
///   * `p > 2q`:  F_{2q,0} = 0, so `h_q(0) = 0`.
///   * `p = 2q`:  F_{2q,0} = c · p!/(p − 2q)! = c · (2q)!.
///   * `p < 2q`:  divergent — Hadamard finite-part needed; not
///                 handled here (caller falls back to ε-reg).
/// Log cases (`p` non-negative even integer) are also returned as
/// `None`.
///
/// Returns `None` if the analytic limit is unavailable for the given
/// (q, d, m, s); caller should keep its ε-regularization path.
pub fn pure_duchon_self_pair_value(
    q: usize,
    d: usize,
    m: usize,
    s: usize,
    eta: &[f64],
) -> Option<f64> {
    if q != 1 && q != 2 {
        return None;
    }
    if eta.len() != d {
        return None;
    }
    let mm = 2 * (m + s); // Riesz block index for q=0 pure-Duchon kernel
    // Detect log case: 2·mm == d + 2n for some n ≥ 0 ⇔ p = 2 mm − d
    // is a non-negative even integer.
    let two_mm = 2 * mm;
    if two_mm >= d && (two_mm - d).is_multiple_of(2) {
        return None; // log regime — Hadamard not implemented here
    }
    let p_int = two_mm as isize - d as isize; // exponent of R in f(R) = c·R^p

    let two_q = 2 * q as isize;
    if p_int < two_q {
        return None; // divergent — Hadamard not implemented here
    }

    // s_1, s_2 (R=0 reduces u_1 = u_2 = 0).
    let mut s_1 = 0.0_f64;
    let mut s_2 = 0.0_f64;
    for &e in eta {
        let bb = (-2.0 * e).exp();
        s_1 += bb;
        s_2 += bb * bb;
    }

    // F_{2q,0}: 0 if p > 2q, else c · (2q)! at p = 2q (per math team §3).
    let f_2q_0 = if p_int > two_q {
        0.0
    } else {
        // p == two_q
        let c = riesz_kernel_coefficient_nonlog(d, mm);
        c * factorial_f64(two_q as usize)
    };

    let value = match q {
        1 => -s_1 * f_2q_0,
        2 => (f_2q_0 / 3.0) * (s_1 * s_1 + 2.0 * s_2),
        // `q ∉ {1, 2}` is rejected by the early guard above; this arm
        // therefore cannot run. Returning `None` keeps the typed
        // contract intact rather than panicking.
        _ => return None,
    };
    Some(value)
}

/// Riesz kernel coefficient `c_j^d` for the non-log case
/// (`R_j^d(R) = c_j^d · R^{2j − d}`):
///   c_j^d = Γ(d/2 − j) / (4^j · π^{d/2} · Γ(j)).
pub(crate) fn riesz_kernel_coefficient_nonlog(d: usize, j: usize) -> f64 {
    let half_d = d as f64 / 2.0;
    let num = gamma_fn(half_d - j as f64);
    let denom = 4.0_f64.powi(j as i32) * std::f64::consts::PI.powf(half_d) * gamma_fn(j as f64);
    num / denom
}

/// Returns the same quantity as `anisotropic_duchon_penalty` (without
/// the J prefactor on `g_q` — caller multiplies by J) through the
/// analytic radial `(-Δ_B)^q` chain. No numerical quadrature is used.
///
/// For `R = 0` the radial chain may be singular. The finite spectral
/// self-pair is evaluated first by `schoenberg_self_pair_bundle` using
/// the closed Gamma/Beta diagonal; smooth odd-dimensional hybrid cases
/// use the Taylor limit. Remaining non-convergent diagonals are rejected
/// rather than approximated by a heat quadrature.
pub fn anisotropic_duchon_penalty_radial(
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

    let powers = AnisoMetricPowers::new(eta);
    anisotropic_duchon_penalty_radial_with_powers(q, m, s, kappa, eta, &powers, r)
}

pub(crate) fn anisotropic_duchon_penalty_radial_with_powers(
    q: usize,
    m: usize,
    s: f64,
    kappa: f64,
    eta: &[f64],
    powers: &AnisoMetricPowers,
    r: &[f64],
) -> f64 {
    assert_eq!(
        eta.len(),
        r.len(),
        "anisotropic_duchon_penalty_radial_with_powers: eta and r dimension mismatch"
    );
    assert!(
        !r.is_empty(),
        "anisotropic_duchon_penalty_radial_with_powers: empty input"
    );
    assert!(
        q <= 2,
        "anisotropic_duchon_penalty_radial_with_powers: q must be in {{0,1,2}}"
    );
    assert!(
        s.is_finite() && s >= 0.0,
        "anisotropic_duchon_penalty_radial_with_powers: s must be finite and ≥ 0, got {s}"
    );
    powers.assert_dim(r.len());

    let d = r.len();
    // Integer-only helpers gate the self-pair / partial-fraction
    // branches; fractional `s` routes around them via the
    // SAFETY: zero-lag self-pair requires validated nullspace-order condition (m > d/2 + s).
    // already-fractional `radial_derivatives_of_isotropic_duchon`
    // path below.
    let s_int = if s.fract() == 0.0 {
        Some(s as usize)
    } else {
        None
    };
    if is_zero_lag(r) {
        if let Some(si) = s_int
            && let Some(bundle) = analytic_self_pair_bundle(q, m, si, kappa, eta)
        {
            return bundle.value / eta.iter().sum::<f64>().exp();
        }
        // SAFETY: zero-lag self-pair requires validated nullspace-order condition (m > d/2 + s).
        panic!(
            "anisotropic_duchon_penalty_radial: zero lag has no finite analytic self-pair for q={q} d={d} m={m} s={s}"
        );
    }

    if let Some(common_eta) = uniform_eta_value(eta) {
        let euclidean_r2 = squared_norm(r);
        if let Some(value) =
            uniform_metric_radial_duchon_penalty(q, m, s, kappa, d, common_eta, euclidean_r2)
        {
            return value;
        }
    }

    // Build invariants R, s_1, s_2, u_1, u_2 only for genuinely
    // anisotropic metrics. For a uniform metric B=bI,
    //
    //   (-Δ_B)^q f(√b |r|) = b^q · g_q^iso(√b |r|),
    //
    // so the q-specific isotropic kernel is the exact analytic chart.
    // Routing uniform metrics through the general radial-derivative chain
    // differentiates the q=0 partial-fraction expansion and can lose many
    // digits in the small-κ cancellation basin.
    let (big_r, s1, s2, u1, u2) = aniso_invariants_with_powers(powers, r);

    // Request the same radial-derivative depth used by the derivative
    // bundle for q > 0. The radial ladder itself chooses the stable chart:
    // partial fractions away from cancellation and the constrained
    // finite-part Riesz series when κR is small. That keeps q>0 values and
    // η/κ derivatives on one analytic representative.
    let max_order = if q == 0 { 0 } else { (2 * q + 2).min(6) };
    let fr = radial_derivatives_of_isotropic_duchon(d, m, s, kappa, big_r, max_order);

    // `q ∈ {0, 1, 2}` is enforced by the `assert!(q <= 2)` at the top of
    // this function; the explicit arms below cover all admissible values
    // without a wildcard.
    if q == 0 {
        fr[0]
    } else if q == 1 {
        -anisotropic_laplacian_of_radial_first(big_r, s1, u1, &fr)
    } else {
        anisotropic_laplacian_of_radial_second(big_r, s1, s2, u1, u2, &fr)
    }
}

pub(crate) fn uniform_metric_radial_duchon_penalty(
    q: usize,
    m: usize,
    s: f64,
    kappa: f64,
    d: usize,
    common_eta: f64,
    euclidean_r2: f64,
) -> Option<f64> {
    let b = (-2.0 * common_eta).exp();
    if !(b.is_finite() && b > 0.0) {
        return None;
    }

    if euclidean_r2 == 0.0 {
        return None;
    }

    let big_r = (b * euclidean_r2).sqrt();
    let max_order = if q == 0 { 0 } else { 2 * q };
    let fr = radial_derivatives_of_isotropic_duchon(d, m, s, kappa, big_r, max_order);
    let big_r2 = big_r * big_r;
    let s1 = (d as f64) * b;
    let s2 = (d as f64) * b * b;
    let u1 = b * big_r2;
    let u2 = b * b * big_r2;
    // `q > 2` has no closed-form analytic chart in this routine; signal
    // that to callers via `None` rather than panicking.
    let value = if q == 0 {
        fr[0]
    } else if q == 1 {
        -anisotropic_laplacian_of_radial_first(big_r, s1, u1, &fr)
    } else if q == 2 {
        anisotropic_laplacian_of_radial_second(big_r, s1, s2, u1, u2, &fr)
    } else {
        return None;
    };
    Some(value)
}

pub(crate) fn uniform_eta_value(eta: &[f64]) -> Option<f64> {
    let (&first, rest) = eta.split_first()?;
    (first.is_finite() && rest.iter().all(|&value| value == first)).then_some(first)
}

pub(crate) fn squared_norm(x: &[f64]) -> f64 {
    x.iter().map(|&value| value * value).sum()
}

pub(crate) fn is_zero_lag(r: &[f64]) -> bool {
    r.iter().all(|&value| value == 0.0)
}

/// Δ_B f(R) with f given by its radial derivatives `[f, f', f'', …]`.
pub(crate) fn anisotropic_laplacian_of_radial_first(big_r: f64, s1: f64, u1: f64, fr: &[f64]) -> f64 {
    // Δ_B f = f''·u_1/R² + f'·(s_1/R - u_1/R³)
    let r2 = big_r * big_r;
    let r3 = r2 * big_r;
    fr[2] * u1 / r2 + fr[1] * (s1 / big_r - u1 / r3)
}

/// Δ_B² f(R) with f given by its radial derivatives `[f, f', …, f^{(4)}]`.
pub(crate) fn anisotropic_laplacian_of_radial_second(
    big_r: f64,
    s1: f64,
    s2: f64,
    u1: f64,
    u2: f64,
    fr: &[f64],
) -> f64 {
    let r = big_r;
    let r2 = r * r;
    let r3 = r2 * r;
    let r4 = r2 * r2;
    let r5 = r4 * r;
    let r6 = r4 * r2;
    let r7 = r6 * r;

    // u_1² · [f''''/R⁴ - 6 f'''/R⁵ + 15 f''/R⁶ - 15 f'/R⁷]
    let part_u1sq =
        u1 * u1 * (fr[4] / r4 - 6.0 * fr[3] / r5 + 15.0 * fr[2] / r6 - 15.0 * fr[1] / r7);
    // s_1·u_1 · [2 f'''/R³ - 6 f''/R⁴ + 6 f'/R⁵]
    let part_s1u1 = s1 * u1 * (2.0 * fr[3] / r3 - 6.0 * fr[2] / r4 + 6.0 * fr[1] / r5);
    // s_1² · [f''/R² - f'/R³]
    let part_s1sq = s1 * s1 * (fr[2] / r2 - fr[1] / r3);
    // u_2 · [4 f'''/R³ - 12 f''/R⁴ + 12 f'/R⁵]
    let part_u2 = u2 * (4.0 * fr[3] / r3 - 12.0 * fr[2] / r4 + 12.0 * fr[1] / r5);
    // s_2 · [2 f''/R² - 2 f'/R³]
    let part_s2 = s2 * (2.0 * fr[2] / r2 - 2.0 * fr[1] / r3);

    part_u1sq + part_s1u1 + part_s1sq + part_u2 + part_s2
}

/// Anisotropic invariants used by the radial form:
///   R  = √(Σ b_k r_k²)         (length of axis-rescaled lag)
///   s_p = Σ b_k^p, p ∈ {1, 2}   (anisotropy traces)
///   u_p = Σ b_k^{p+1} r_k², p ∈ {1, 2}
/// where b_k = exp(-2 η_k).
pub(crate) fn aniso_invariants(eta: &[f64], r: &[f64]) -> (f64, f64, f64, f64, f64) {
    let powers = AnisoMetricPowers::new(eta);
    aniso_invariants_with_powers(&powers, r)
}

pub(crate) fn aniso_invariants_with_powers(
    powers: &AnisoMetricPowers,
    r: &[f64],
) -> (f64, f64, f64, f64, f64) {
    // SIMD-vectorized over the d-axis using `wide::f64x4` (stable, AVX2 on
    // x86_64; portable scalar fallback elsewhere). Five accumulators share
    // the same iteration pattern, so we widen each into a 4-lane vector
    // accumulator and horizontally reduce at the end. The expensive
    // `exp(-2η)` powers are precomputed by `AnisoMetricPowers`; the hot
    // pair loop only performs multiplies/adds.
    use wide::f64x4;

    let d = r.len();
    powers.assert_dim(d);
    let mut s1_v = f64x4::ZERO;
    let mut s2_v = f64x4::ZERO;
    let mut r2_v = f64x4::ZERO;
    let mut u1_v = f64x4::ZERO;
    let mut u2_v = f64x4::ZERO;

    let chunks = d / 4;
    let tail = d % 4;
    for c in 0..chunks {
        let base = c * 4;
        let b_arr = [
            powers.b[base],
            powers.b[base + 1],
            powers.b[base + 2],
            powers.b[base + 3],
        ];
        let b2_arr = [
            powers.b2[base],
            powers.b2[base + 1],
            powers.b2[base + 2],
            powers.b2[base + 3],
        ];
        let b3_arr = [
            powers.b3[base],
            powers.b3[base + 1],
            powers.b3[base + 2],
            powers.b3[base + 3],
        ];
        let rk_arr = [r[base], r[base + 1], r[base + 2], r[base + 3]];
        let b = f64x4::from(b_arr);
        let b2 = f64x4::from(b2_arr);
        let b3 = f64x4::from(b3_arr);
        let rk = f64x4::from(rk_arr);
        let rk2 = rk * rk;
        s1_v += b;
        s2_v += b2;
        r2_v += b * rk2;
        u1_v += b2 * rk2;
        u2_v += b3 * rk2;
    }

    // Horizontal reduction of the 4-lane accumulators.
    let s1_a = s1_v.to_array();
    let s2_a = s2_v.to_array();
    let r2_a = r2_v.to_array();
    let u1_a = u1_v.to_array();
    let u2_a = u2_v.to_array();
    let mut s1 = s1_a[0] + s1_a[1] + s1_a[2] + s1_a[3];
    let mut s2 = s2_a[0] + s2_a[1] + s2_a[2] + s2_a[3];
    let mut r2 = r2_a[0] + r2_a[1] + r2_a[2] + r2_a[3];
    let mut u1 = u1_a[0] + u1_a[1] + u1_a[2] + u1_a[3];
    let mut u2 = u2_a[0] + u2_a[1] + u2_a[2] + u2_a[3];

    // Scalar tail (0..3 elements).
    let tail_start = chunks * 4;
    for k in 0..tail {
        let idx = tail_start + k;
        let rk2 = r[idx] * r[idx];
        let b = powers.b[idx];
        let b2 = powers.b2[idx];
        s1 += b;
        s2 += b2;
        r2 += b * rk2;
        u1 += b2 * rk2;
        u2 += powers.b3[idx] * rk2;
    }

    (r2.sqrt(), s1, s2, u1, u2)
}

/// Scalar reference implementation of `aniso_invariants` used as a
/// baseline in the SIMD pair-block benchmark. Returns (R, s_1, s_2,
/// u_1, u_2) with no `wide::f64x4` lane parallelism. Numerically
/// identical to the SIMD path up to floating-point summation order
/// (lane reductions vs sequential).
pub fn aniso_invariants_scalar(eta: &[f64], r: &[f64]) -> (f64, f64, f64, f64, f64) {
    assert_eq!(eta.len(), r.len());
    let mut s1 = 0.0_f64;
    let mut s2 = 0.0_f64;
    let mut r2 = 0.0_f64;
    let mut u1 = 0.0_f64;
    let mut u2 = 0.0_f64;
    for k in 0..eta.len() {
        let b = (-2.0 * eta[k]).exp();
        let b2 = b * b;
        let rk2 = r[k] * r[k];
        s1 += b;
        s2 += b2;
        r2 += b * rk2;
        u1 += b2 * rk2;
        u2 += b2 * b * rk2;
    }
    (r2.sqrt(), s1, s2, u1, u2)
}

/// SIMD path of `aniso_invariants` exposed under a stable name for the
/// pair-block SIMD-vs-scalar benchmark. Forwards to the private
/// implementation used in production hot paths.
pub fn aniso_invariants_simd(eta: &[f64], r: &[f64]) -> (f64, f64, f64, f64, f64) {
    aniso_invariants(eta, r)
}

/// κ-partial of `radial_derivatives_of_isotropic_duchon`: returns
/// `[∂_κ f, ∂_κ f', …, ∂_κ f^{(max_order)}]`(R).
///
/// Only Matérn blocks (and their κ-dependent partial-fraction
/// coefficients) contribute: pure Riesz blocks are κ-independent.
///
/// Analytic chain-rule derivation. With `a = 2m`, `b = 2s`,
/// `n_j = a + b − j`, the partial-fraction expansion of f at fixed
/// (d, m, s) is
///
///   f^{(k)}(R; κ) = Σ_{j=1}^{a} A_j(κ) · [R_j^d]^{(k)}(R)
///                 + Σ_{ℓ=1}^{b} B_ℓ(κ) · [M_ℓ^d(·; κ)]^{(k)}(R),
///
/// with A_j(κ) = (−1)^{a−j} · C(n_j+j−1, a−j) · κ^{−2 n_j} and
/// B_ℓ(κ) = (−1)^a · C(n_ℓ+ℓ−1, b−ℓ) · κ^{−2 n_ℓ}. Differentiating in κ
/// gives A_j'(κ) = −(2 n_j / κ) · A_j(κ) and B_ℓ'(κ) = −(2 n_ℓ / κ) ·
/// B_ℓ(κ), and the Matérn-block κ-derivative satisfies
/// `∂_κ M_ℓ^d(R; κ) = −2 ℓ κ · M_{ℓ+1}^d(R; κ)` (and the same identity
/// commutes with the radial derivative ∂^k_R since M depends on R only
/// through κR alongside κ). The result is
///
///   ∂_κ f^{(k)} = Σ_j A_j'(κ) · [R_j^d]^{(k)}
///               + Σ_ℓ ( B_ℓ'(κ) · [M_ℓ^d]^{(k)}
///                       − 2 ℓ κ · B_ℓ(κ) · [M_{ℓ+1}^d]^{(k)} ).
///
/// Fully analytic κ partial for the radial derivative ladder.
pub fn radial_derivatives_of_isotropic_duchon_kappa_partial(
    d: usize,
    m: usize,
    s: usize,
    kappa: f64,
    r: f64,
    max_order: usize,
) -> Vec<f64> {
    assert!(
        r > 0.0,
        "Duchon kappa partial requires positive radius: r={r}, d={d}, m={m}, s={s}, kappa={kappa}"
    );
    assert!(
        max_order <= 6,
        "Duchon kappa partial supports max_order <= 6: max_order={max_order}, d={d}, m={m}, s={s}"
    );
    if s == 0 || kappa == 0.0 {
        // No Matérn or κ → 0 limit (Riesz pure) — both κ-independent.
        return vec![0.0_f64; max_order + 1];
    }

    let a = 2 * m;
    let b = 2 * s;
    if use_duchon_small_chi_riesz_series(kappa, r) {
        return duchon_small_chi_riesz_series_radial_derivatives(d, a, b, kappa, r, max_order, 1);
    }

    let kappa_sq = kappa * kappa;
    let mut total = vec![KahanSum::default(); max_order + 1];

    // Riesz piece: A_j'(κ) = -(2 n_j / κ) · A_j(κ).
    for j in 1..=a {
        let n_j = a + b - j;
        let sign = if (a - j).is_multiple_of(2) { 1.0 } else { -1.0 };
        let binom = binomial_f64(a + b - j - 1, a - j);
        let a_j = sign * binom * kappa_sq.powi(-(n_j as i32));
        let a_j_prime = -(2.0 * n_j as f64 / kappa) * a_j;
        let block = riesz_block_radial_derivatives(d, (j) as f64, r, max_order);
        for (k, v) in block.into_iter().enumerate() {
            total[k].add(a_j_prime * v);
        }
    }

    // Matérn pieces: B_ℓ'·M_ℓ + B_ℓ·∂_κ M_ℓ, with ∂_κ M_ℓ = -2ℓκ·M_{ℓ+1}.
    let sign_a = if a.is_multiple_of(2) { 1.0 } else { -1.0 };
    for ell in 1..=b {
        let n_ell = a + b - ell;
        let binom = binomial_f64(a + b - ell - 1, b - ell);
        let b_ell = sign_a * binom * kappa_sq.powi(-(n_ell as i32));
        let b_ell_prime = -(2.0 * n_ell as f64 / kappa) * b_ell;

        let m_ell = matern_block_radial_derivatives(d, ell, kappa, r, max_order);
        let m_ell_p1 = matern_block_radial_derivatives(d, ell + 1, kappa, r, max_order);
        let kappa_factor = -2.0 * ell as f64 * kappa;
        for k in 0..=max_order {
            total[k].add(b_ell_prime * m_ell[k] + b_ell * kappa_factor * m_ell_p1[k]);
        }
    }

    total.iter().map(|acc| acc.sum()).collect()
}

/// Second κ-partial of `radial_derivatives_of_isotropic_duchon`: returns
/// `[∂²_κ f, ∂²_κ f', …, ∂²_κ f^{(max_order)}]`(R).
///
/// Differentiating the partial-fraction expansion twice in κ:
///   A_j''(κ) = (2 n_j (2 n_j + 1) / κ²) · A_j(κ)
///   B_ℓ''(κ) = (2 n_ℓ (2 n_ℓ + 1) / κ²) · B_ℓ(κ)
///   ∂_κ  M_ℓ = -2 ℓ κ · M_{ℓ+1}
///   ∂²_κ M_ℓ = -2 ℓ · M_{ℓ+1} + 4 ℓ (ℓ+1) κ² · M_{ℓ+2}
/// Composition (treat each Matérn term as a product B_ℓ(κ) · M_ℓ(R; κ)):
///   ∂²_κ (B_ℓ · M_ℓ) = B_ℓ'' M_ℓ + 2 B_ℓ' (-2 ℓ κ M_{ℓ+1})
///                   + B_ℓ (-2 ℓ M_{ℓ+1} + 4 ℓ (ℓ+1) κ² M_{ℓ+2}).
pub fn radial_derivatives_of_isotropic_duchon_kappa_partial2(
    d: usize,
    m: usize,
    s: usize,
    kappa: f64,
    r: f64,
    max_order: usize,
) -> Vec<f64> {
    assert!(
        r > 0.0,
        "Duchon second kappa partial requires positive radius: r={r}, d={d}, m={m}, s={s}, kappa={kappa}"
    );
    assert!(
        max_order <= 6,
        "Duchon second kappa partial supports max_order <= 6: max_order={max_order}, d={d}, m={m}, s={s}"
    );
    if s == 0 || kappa == 0.0 {
        return vec![0.0_f64; max_order + 1];
    }

    let a = 2 * m;
    let b = 2 * s;
    if use_duchon_small_chi_riesz_series(kappa, r) {
        return duchon_small_chi_riesz_series_radial_derivatives(d, a, b, kappa, r, max_order, 2);
    }

    let kappa_sq = kappa * kappa;
    let mut total = vec![KahanSum::default(); max_order + 1];

    for j in 1..=a {
        let n_j = a + b - j;
        let sign = if (a - j).is_multiple_of(2) { 1.0 } else { -1.0 };
        let binom = binomial_f64(a + b - j - 1, a - j);
        let a_j = sign * binom * kappa_sq.powi(-(n_j as i32));
        let nj_f = n_j as f64;
        let a_j_dd = (2.0 * nj_f * (2.0 * nj_f + 1.0) / kappa_sq) * a_j;
        let block = riesz_block_radial_derivatives(d, (j) as f64, r, max_order);
        for (k, v) in block.into_iter().enumerate() {
            total[k].add(a_j_dd * v);
        }
    }

    let sign_a = if a.is_multiple_of(2) { 1.0 } else { -1.0 };
    for ell in 1..=b {
        let n_ell = a + b - ell;
        let binom = binomial_f64(a + b - ell - 1, b - ell);
        let b_ell = sign_a * binom * kappa_sq.powi(-(n_ell as i32));
        let n_ell_f = n_ell as f64;
        let b_ell_prime = -(2.0 * n_ell_f / kappa) * b_ell;
        let b_ell_dd = (2.0 * n_ell_f * (2.0 * n_ell_f + 1.0) / kappa_sq) * b_ell;

        let ell_f = ell as f64;
        let m_ell = matern_block_radial_derivatives(d, ell, kappa, r, max_order);
        let m_ell_p1 = matern_block_radial_derivatives(d, ell + 1, kappa, r, max_order);
        let m_ell_p2 = matern_block_radial_derivatives(d, ell + 2, kappa, r, max_order);

        // ∂_κ M_ℓ = -2 ℓ κ · M_{ℓ+1}
        // ∂²_κ M_ℓ = -2 ℓ · M_{ℓ+1} + 4 ℓ (ℓ+1) κ² · M_{ℓ+2}
        let cross_factor = -4.0 * ell_f * kappa; // 2 · ∂_κ M_ℓ / B_ℓ' coefficient
        let m_dd_a = -2.0 * ell_f; // first half of ∂²κ M_ℓ
        let m_dd_b = 4.0 * ell_f * (ell_f + 1.0) * kappa_sq; // second half
        for k in 0..=max_order {
            total[k].add(
                b_ell_dd * m_ell[k]
                    + b_ell_prime * cross_factor * m_ell_p1[k]
                    + b_ell * (m_dd_a * m_ell_p1[k] + m_dd_b * m_ell_p2[k]),
            );
        }
    }

    total.iter().map(|acc| acc.sum()).collect()
}

/// Value `g_q` and its partial derivatives w.r.t. the invariants
/// `(R, s_1, s_2, u_1, u_2)` at fixed radial-derivative table `fr`.
///
/// `fr[k]` must equal `f^{(k)}(R; κ)` for `k = 0, …, 2q + 1`.
/// The extra `f^{(2q+1)}` is needed because `g_R` differentiates
/// each `f^{(k)}` term once.
///
/// Returns `(g, g_R, g_s1, g_s2, g_u1, g_u2)`.
pub(crate) fn radial_g_q_partials(
    q: usize,
    big_r: f64,
    s1: f64,
    s2: f64,
    u1: f64,
    u2: f64,
    fr: &[f64],
) -> (f64, f64, f64, f64, f64, f64) {
    let r = big_r;
    let r2 = r * r;
    let r3 = r2 * r;
    let r4 = r2 * r2;
    let r5 = r4 * r;
    let r6 = r4 * r2;
    let r7 = r6 * r;
    let r8 = r4 * r4;

    match q {
        0 => {
            // g_0 = f(R)
            let g = fr[0];
            let g_r = fr[1];
            (g, g_r, 0.0, 0.0, 0.0, 0.0)
        }
        1 => {
            // g_1 = -[f''·u_1/R² + f'·(s_1/R - u_1/R³)]
            //     = -f''·u_1/R² - f'·s_1/R + f'·u_1/R³
            let g = -(fr[2] * u1 / r2 + fr[1] * (s1 / r - u1 / r3));
            // ∂g/∂R: differentiate every R-power and every f^{(k)}(R).
            //   d/dR[-f''·u1/R²] = -f'''·u1/R² + 2 f''·u1/R³
            //   d/dR[-f'·s1/R]   = -f''·s1/R + f'·s1/R²
            //   d/dR[ f'·u1/R³]  =  f''·u1/R³ - 3 f'·u1/R⁴
            let g_r = -fr[3] * u1 / r2 + 2.0 * fr[2] * u1 / r3 - fr[2] * s1 / r
                + fr[1] * s1 / r2
                + fr[2] * u1 / r3
                - 3.0 * fr[1] * u1 / r4;
            // Combine the two u1/R³ terms:
            //   2 f''·u1/R³ + f''·u1/R³ = 3 f''·u1/R³.
            // Final form (kept above as raw sum for clarity; algebra:
            //   g_R = -f'''·u1/R² + 3 f''·u1/R³ - f''·s1/R + f'·s1/R² - 3 f'·u1/R⁴)
            let g_s1 = -fr[1] / r;
            let g_u1 = -fr[2] / r2 + fr[1] / r3;
            (g, g_r, g_s1, 0.0, g_u1, 0.0)
        }
        2 => {
            // g_2 = u_1²·F1 + s_1·u_1·F2 + s_1²·F3 + u_2·F4 + s_2·F5
            let f1 = fr[4] / r4 - 6.0 * fr[3] / r5 + 15.0 * fr[2] / r6 - 15.0 * fr[1] / r7;
            let f2 = 2.0 * fr[3] / r3 - 6.0 * fr[2] / r4 + 6.0 * fr[1] / r5;
            let f3 = fr[2] / r2 - fr[1] / r3;
            let f4 = 4.0 * fr[3] / r3 - 12.0 * fr[2] / r4 + 12.0 * fr[1] / r5;
            let f5 = 2.0 * fr[2] / r2 - 2.0 * fr[1] / r3;
            let g = u1 * u1 * f1 + s1 * u1 * f2 + s1 * s1 * f3 + u2 * f4 + s2 * f5;

            // F_i derivatives in R (chain through f^{(k)}(R) and through 1/R^e).
            //   dF1/dR = f⁽⁵⁾/R⁴ - 10 f''''/R⁵ + 45 f'''/R⁶ - 105 f''/R⁷ + 105 f'/R⁸
            //   dF2/dR = 2 f''''/R³ - 12 f'''/R⁴ + 30 f''/R⁵ - 30 f'/R⁶
            //   dF3/dR = f'''/R² - 3 f''/R³ + 3 f'/R⁴
            //   dF4/dR = 4 f''''/R³ - 24 f'''/R⁴ + 60 f''/R⁵ - 60 f'/R⁶
            //   dF5/dR = 2 f'''/R² - 6 f''/R³ + 6 f'/R⁴
            let df1 = fr[5] / r4 - 10.0 * fr[4] / r5 + 45.0 * fr[3] / r6 - 105.0 * fr[2] / r7
                + 105.0 * fr[1] / r8;
            let df2 = 2.0 * fr[4] / r3 - 12.0 * fr[3] / r4 + 30.0 * fr[2] / r5 - 30.0 * fr[1] / r6;
            let df3 = fr[3] / r2 - 3.0 * fr[2] / r3 + 3.0 * fr[1] / r4;
            let df4 = 4.0 * fr[4] / r3 - 24.0 * fr[3] / r4 + 60.0 * fr[2] / r5 - 60.0 * fr[1] / r6;
            let df5 = 2.0 * fr[3] / r2 - 6.0 * fr[2] / r3 + 6.0 * fr[1] / r4;

            let g_r = u1 * u1 * df1 + s1 * u1 * df2 + s1 * s1 * df3 + u2 * df4 + s2 * df5;

            // ∂g/∂s_1 = u_1·F2 + 2 s_1·F3
            // ∂g/∂s_2 = F5
            // ∂g/∂u_1 = 2 u_1·F1 + s_1·F2
            // ∂g/∂u_2 = F4
            let g_s1 = u1 * f2 + 2.0 * s1 * f3;
            let g_s2 = f5;
            let g_u1 = 2.0 * u1 * f1 + s1 * f2;
            let g_u2 = f4;
            (g, g_r, g_s1, g_s2, g_u1, g_u2)
        }
        // SAFETY: `q` is the spatial-derivative order of the radial
        // Duchon kernel, which the type-level callers (q=0,1,2 only via
        // `RadialKernel::value`/`gradient`/`laplacian`) restrict to
        // `{0, 1, 2}`; reaching this wildcard means an unsupported `q`
        // was forwarded by an internal entry point that should never
        // expose it.
        // SAFETY: `q` is statically restricted to `{0, 1, 2}` by the
        // public radial-kernel API (value/gradient/laplacian); this
        // arm is unreachable for in-contract callers.
        _ => panic!("radial_g_q_partials requires q in {{0, 1, 2}}: q={q}"),
    }
}

/// Hessian of `g_q` w.r.t. the invariants `(R, s_1, s_2, u_1, u_2)` at
/// fixed radial-derivative table `fr`. Required for the second-order
/// chain rule into `∂²_{η_k η_l} g_q`.
///
/// `fr[k]` must equal `f^{(k)}(R; κ)` for `k = 0, …, 2q + 2`, since
/// `g_{RR}` differentiates `g_R` once more in R (which already had
/// `f^{(2q+1)}`).
///
/// Returned tuple is the upper-triangle of the symmetric 5×5 Hessian:
///   `(g_RR,
///     g_R_s1, g_R_s2, g_R_u1, g_R_u2,
///     g_s1s1, g_s1s2, g_s1u1, g_s1u2,
///     g_s2s2, g_s2u1, g_s2u2,
///     g_u1u1, g_u1u2,
///     g_u2u2)`
/// All cross terms not involving `R` vanish for q ≤ 2 except
/// `g_s1s1 = 2 F3`, `g_s1u1 = F2`, `g_u1u1 = 2 F1` at q = 2.
pub(crate) fn radial_g_q_hessian(
    q: usize,
    big_r: f64,
    s1: f64,
    s2: f64,
    u1: f64,
    u2: f64,
    fr: &[f64],
) -> (
    f64, // g_RR
    f64, // g_R_s1
    f64, // g_R_s2
    f64, // g_R_u1
    f64, // g_R_u2
    f64, // g_s1s1
    f64, // g_s1s2
    f64, // g_s1u1
    f64, // g_s1u2
    f64, // g_s2s2
    f64, // g_s2u1
    f64, // g_s2u2
    f64, // g_u1u1
    f64, // g_u1u2
    f64, // g_u2u2
) {
    let r = big_r;
    let r2 = r * r;
    let r3 = r2 * r;
    let r4 = r2 * r2;
    let r5 = r4 * r;
    let r6 = r4 * r2;
    let r7 = r6 * r;
    let r8 = r4 * r4;
    let r9 = r8 * r;

    match q {
        0 => {
            // g = f(R). Only non-zero second partial is g_RR = f''.
            let g_rr = fr[2];
            (
                g_rr, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
            )
        }
        1 => {
            // g_R   = -f''' u_1/R² + 3 f'' u_1/R³ - f'' s_1/R + f' s_1/R² - 3 f' u_1/R⁴
            // g_RR  = ∂g_R/∂R, expanded with chain through f^{(k)}(R) and 1/R^e:
            //   = -f'''' u1/R² + 5 f''' u1/R³ - 12 f'' u1/R⁴
            //     - f''' s1/R + 2 f'' s1/R² - 2 f' s1/R³
            //     + 12 f' u1/R⁵
            let g_rr =
                -fr[4] * u1 / r2 + 5.0 * fr[3] * u1 / r3 - 12.0 * fr[2] * u1 / r4 - fr[3] * s1 / r
                    + 2.0 * fr[2] * s1 / r2
                    - 2.0 * fr[1] * s1 / r3
                    + 12.0 * fr[1] * u1 / r5;
            // g_R_s1 = ∂g_R/∂s1 = -f''/R + f'/R²
            let g_r_s1 = -fr[2] / r + fr[1] / r2;
            // g_R_u1 = ∂g_R/∂u1 = -f'''/R² + 3 f''/R³ - 3 f'/R⁴
            let g_r_u1 = -fr[3] / r2 + 3.0 * fr[2] / r3 - 3.0 * fr[1] / r4;
            // q=1 is linear in s_1, u_1; cross/diagonal partials vanish.
            (
                g_rr, g_r_s1, 0.0, g_r_u1, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
            )
        }
        2 => {
            // F_i, dF_i/dR, d²F_i/dR² blocks (manually expanded; see
            // `radial_g_q_partials` for F_i and dF_i/dR derivations).
            // F1, F2, F3 are used by g_{s1, s1}, g_{s1, u1}, g_{u1, u1}
            // (the only nonzero pure-invariant Hessian entries for q ≤ 2);
            // F4, F5 do not appear in any second partial of g (g is linear
            // in u_2 and s_2) so they are not needed here.
            let f1 = fr[4] / r4 - 6.0 * fr[3] / r5 + 15.0 * fr[2] / r6 - 15.0 * fr[1] / r7;
            let f2 = 2.0 * fr[3] / r3 - 6.0 * fr[2] / r4 + 6.0 * fr[1] / r5;
            let f3 = fr[2] / r2 - fr[1] / r3;

            let df1 = fr[5] / r4 - 10.0 * fr[4] / r5 + 45.0 * fr[3] / r6 - 105.0 * fr[2] / r7
                + 105.0 * fr[1] / r8;
            let df2 = 2.0 * fr[4] / r3 - 12.0 * fr[3] / r4 + 30.0 * fr[2] / r5 - 30.0 * fr[1] / r6;
            let df3 = fr[3] / r2 - 3.0 * fr[2] / r3 + 3.0 * fr[1] / r4;
            let df4 = 4.0 * fr[4] / r3 - 24.0 * fr[3] / r4 + 60.0 * fr[2] / r5 - 60.0 * fr[1] / r6;
            let df5 = 2.0 * fr[3] / r2 - 6.0 * fr[2] / r3 + 6.0 * fr[1] / r4;

            // d²F_i/dR² (derived in task #24 phase 3 notes):
            //   d²F1/dR² = f⁽⁶⁾/R⁴ - 14 f⁽⁵⁾/R⁵ + 95 f⁽⁴⁾/R⁶ - 375 f'''/R⁷ + 840 f''/R⁸ - 840 f'/R⁹
            //   d²F2/dR² = 2 f⁽⁵⁾/R³ - 18 f⁽⁴⁾/R⁴ + 78 f'''/R⁵ - 180 f''/R⁶ + 180 f'/R⁷
            //   d²F3/dR² = f⁽⁴⁾/R² - 5 f'''/R³ + 12 f''/R⁴ - 12 f'/R⁵
            //   d²F4/dR² = 4 f⁽⁵⁾/R³ - 36 f⁽⁴⁾/R⁴ + 156 f'''/R⁵ - 360 f''/R⁶ + 360 f'/R⁷
            //   d²F5/dR² = 2 f⁽⁴⁾/R² - 10 f'''/R³ + 24 f''/R⁴ - 24 f'/R⁵
            let d2f1 = fr[6] / r4 - 14.0 * fr[5] / r5 + 95.0 * fr[4] / r6 - 375.0 * fr[3] / r7
                + 840.0 * fr[2] / r8
                - 840.0 * fr[1] / r9;
            let d2f2 = 2.0 * fr[5] / r3 - 18.0 * fr[4] / r4 + 78.0 * fr[3] / r5
                - 180.0 * fr[2] / r6
                + 180.0 * fr[1] / r7;
            let d2f3 = fr[4] / r2 - 5.0 * fr[3] / r3 + 12.0 * fr[2] / r4 - 12.0 * fr[1] / r5;
            let d2f4 = 4.0 * fr[5] / r3 - 36.0 * fr[4] / r4 + 156.0 * fr[3] / r5
                - 360.0 * fr[2] / r6
                + 360.0 * fr[1] / r7;
            let d2f5 = 2.0 * fr[4] / r2 - 10.0 * fr[3] / r3 + 24.0 * fr[2] / r4 - 24.0 * fr[1] / r5;

            // g_RR = u_1²·d²F1 + s_1·u_1·d²F2 + s_1²·d²F3 + u_2·d²F4 + s_2·d²F5
            let g_rr = u1 * u1 * d2f1 + s1 * u1 * d2f2 + s1 * s1 * d2f3 + u2 * d2f4 + s2 * d2f5;

            // Mixed R-X partials: ∂(∂g/∂X)/∂R = derivative of the X-coefficient
            // of g_R, i.e., differentiate the corresponding F (or its dF) entries.
            //   g_R_s1 = ∂g_R/∂s1 = u_1·dF2 + 2 s_1·dF3
            //   g_R_s2 = ∂g_R/∂s2 = dF5
            //   g_R_u1 = ∂g_R/∂u1 = 2 u_1·dF1 + s_1·dF2
            //   g_R_u2 = ∂g_R/∂u2 = dF4
            let g_r_s1 = u1 * df2 + 2.0 * s1 * df3;
            let g_r_s2 = df5;
            let g_r_u1 = 2.0 * u1 * df1 + s1 * df2;
            let g_r_u2 = df4;

            // Pure-invariant Hessian (q=2 quadratic form):
            //   g = u_1²·F1 + s_1 u_1·F2 + s_1²·F3 + u_2·F4 + s_2·F5
            // ⇒ g_{s1, s1} = 2 F3
            //   g_{s1, u1} = F2
            //   g_{u1, u1} = 2 F1
            // All other invariant-pair Hessian entries vanish.
            let g_s1s1 = 2.0 * f3;
            let g_s1u1 = f2;
            let g_u1u1 = 2.0 * f1;

            (
                g_rr, g_r_s1, g_r_s2, g_r_u1, g_r_u2, g_s1s1, 0.0, g_s1u1, 0.0, 0.0, 0.0, 0.0,
                g_u1u1, 0.0, 0.0,
            )
        }
        // SAFETY: companion to `radial_g_q_partials` above — `q` is
        // statically restricted to `{0, 1, 2}` by the public radial
        // kernel API (value/gradient/laplacian only). An unsupported `q`
        // here means an internal helper forwarded an out-of-contract
        // value.
        // SAFETY: companion to `radial_g_q_partials` — `q` is
        // statically restricted to `{0, 1, 2}` by the public radial
        // kernel API; this arm is unreachable for in-contract callers.
        _ => panic!("radial_g_q_hessian requires q in {{0, 1, 2}}: q={q}"),
    }
}

pub(crate) fn aniso_invariants_eta_jacobian_with_powers(
    eta: &[f64],
    r: &[f64],
    powers: &AnisoMetricPowers,
) -> (
    f64,
    f64,
    f64,
    f64,
    f64,
    Vec<f64>,
    Vec<f64>,
    Vec<f64>,
    Vec<f64>,
    Vec<f64>,
) {
    let d = eta.len();
    powers.assert_dim(d);
    let (big_r, s1, s2, u1, u2) = aniso_invariants_with_powers(powers, r);
    let mut dr_de = vec![0.0_f64; d];
    let mut ds1_de = vec![0.0_f64; d];
    let mut ds2_de = vec![0.0_f64; d];
    let mut du1_de = vec![0.0_f64; d];
    let mut du2_de = vec![0.0_f64; d];
    for l in 0..d {
        let b_l = powers.b[l];
        let b_l_sq = powers.b2[l];
        let b_l_cu = powers.b3[l];
        let r_l_sq = r[l] * r[l];
        ds1_de[l] = -2.0 * b_l;
        ds2_de[l] = -4.0 * b_l_sq;
        dr_de[l] = if big_r > 0.0 {
            -b_l * r_l_sq / big_r
        } else {
            0.0
        };
        du1_de[l] = -4.0 * b_l_sq * r_l_sq;
        du2_de[l] = -6.0 * b_l_cu * r_l_sq;
    }
    (big_r, s1, s2, u1, u2, dr_de, ds1_de, ds2_de, du1_de, du2_de)
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
pub fn pair_block_radial_with_j_second_derivatives(
    q: usize,
    m: usize,
    s: usize,
    kappa: f64,
    eta: &[f64],
    r: &[f64],
) -> PairBlockBundle {
    let powers = AnisoMetricPowers::new(eta);
    pair_block_radial_with_j_second_derivatives_with_powers(q, m, s, kappa, eta, &powers, r)
}

pub(crate) fn pair_block_radial_with_j_second_derivatives_with_powers(
    q: usize,
    m: usize,
    s: usize,
    kappa: f64,
    eta: &[f64],
    powers: &AnisoMetricPowers,
    r: &[f64],
) -> PairBlockBundle {
    assert_eq!(
        eta.len(),
        r.len(),
        "pair_block_radial_with_j_second_derivatives_with_powers: eta and r dimension mismatch"
    );
    assert!(
        !r.is_empty(),
        "pair_block_radial_with_j_second_derivatives_with_powers: empty input"
    );
    assert!(
        q <= 2,
        "pair_block_radial_with_j_second_derivatives_with_powers: q must be in {{0,1,2}}"
    );
    let d = eta.len();
    powers.assert_dim(d);
    let (big_r_check, _, _, _, _) = aniso_invariants_with_powers(powers, r);
    let analytic_first_ok = big_r_check > 0.0;
    if big_r_check == 0.0
        && let Some(bundle) = analytic_self_pair_bundle(q, m, s, kappa, eta)
    {
        return bundle;
    }
    assert!(
        analytic_first_ok,
        "pair_block_radial_with_j_second_derivatives: zero lag has no finite analytic self-pair for q={q} d={d} m={m} s={s}"
    );

    // First derivatives ∂_{η_l} (J · g_q) and ∂_κ (J · g_q).
    let mut d_eta = vec![0.0_f64; d];
    let mut d2_eta = vec![vec![0.0_f64; d]; d];
    let mut d2_eta_kappa = vec![0.0_f64; d];

    // Analytic d_eta and d_kappa via chain rule on (R, s_1, s_2, u_1, u_2).
    // Need radial derivatives up to order 2q + 2 because the η-Hessian
    // uses g_RR (q=2 → f^{(6)}). The same table is reused for value and
    // first η derivatives.
    let max_order_h = (2 * q + 2).min(6);
    let (big_r, s1, s2, u1, u2, dr_de, ds1_de, ds2_de, du1_de, du2_de) =
        aniso_invariants_eta_jacobian_with_powers(eta, r, powers);
    let fr = radial_derivatives_of_isotropic_duchon(d, m, (s) as f64, kappa, big_r, max_order_h);
    let (g, g_r, g_s1, g_s2, g_u1, g_u2) = radial_g_q_partials(q, big_r, s1, s2, u1, u2, &fr);
    let big_j = eta.iter().sum::<f64>().exp();
    let value = big_j * g;

    // ∂_η_l (J · g) = J · (g + ∂_η_l g)
    //   ∂_η_l g = g_R · ∂R/∂η_l + g_s1 · ∂s1/∂η_l + g_s2 · ∂s2/∂η_l
    //           + g_u1 · ∂u1/∂η_l + g_u2 · ∂u2/∂η_l
    for l in 0..d {
        let bare_d_eta_g = g_r * dr_de[l]
            + g_s1 * ds1_de[l]
            + g_s2 * ds2_de[l]
            + g_u1 * du1_de[l]
            + g_u2 * du2_de[l];
        d_eta[l] = big_j * (g + bare_d_eta_g);
    }

    // ∂_κ. Analytic via chain rule on F^{(k)}: only the radial-derivative
    // table depends on κ, the invariants (R, s_1, s_2, u_1, u_2) do not.
    // ∂_κ (J · g_q) = J · g_q evaluated with fr replaced by ∂_κ fr.
    let dfr = if s != 0 && kappa != 0.0 {
        let max_order = 2 * q + 1;
        Some(radial_derivatives_of_isotropic_duchon_kappa_partial(
            d, m, s, kappa, big_r, max_order,
        ))
    } else {
        None
    };
    let d_kappa = if let Some(dfr) = &dfr {
        // The same g_q expansion in (R, s1, s2, u1, u2) but with ∂_κ F^{(k)}
        // in place of F^{(k)}; the (R, ..., u2) factors are κ-independent.
        // Because g_q is linear in each F^{(k)} for q ∈ {0, 1, 2}, the
        // partial helper applied to ∂_κ F gives ∂_κ g_q.
        let (dg, _, _, _, _, _) = radial_g_q_partials(q, big_r, s1, s2, u1, u2, dfr);
        big_j * dg
    } else {
        0.0
    };

    // ∂²_κ. Analytic via the second κ-partial of the radial derivative
    // table. The invariants (R, s_1, s_2, u_1, u_2) are κ-independent, so
    // ∂²_κ (J · g_q) = J · g_q evaluated with fr replaced by ∂²_κ fr.
    // Linearity of g_q in each f^{(k)} for q ∈ {0, 1, 2} lets us reuse
    // `radial_g_q_partials` directly on the second-κ-partial table.
    let d2_kappa = if s != 0 && kappa != 0.0 {
        let max_order = 2 * q + 1;
        let ddfr =
            radial_derivatives_of_isotropic_duchon_kappa_partial2(d, m, s, kappa, big_r, max_order);
        let (ddg, _, _, _, _, _) = radial_g_q_partials(q, big_r, s1, s2, u1, u2, &ddfr);
        big_j * ddg
    } else {
        0.0
    };

    // Second η-η Hessian. Analytic chain rule:
    //   ∂²_{η_k η_l} (J · g)
    //     = J · (g + ∂_{η_k} g + ∂_{η_l} g + ∂²_{η_k η_l} g)
    // where
    //   ∂²_{η_k η_l} g = Σ_{X, Y} g_{X,Y} · (∂_{η_k} Y) · (∂_{η_l} X)
    //                   + Σ_X g_X · (∂²_{η_k η_l} X).
    //
    // The second-derivative tensor of the invariants is diagonal in
    // (k, l) for s_1, s_2, u_1, u_2 (each invariant is a single sum
    // over k whose summand depends only on b_k, r_k):
    //   ∂²s_1/∂η_k ∂η_l = 4 b_l δ_{kl}
    //   ∂²s_2/∂η_k ∂η_l = 16 b_l² δ_{kl}
    //   ∂²u_1/∂η_k ∂η_l = 16 b_l² r_l² δ_{kl}
    //   ∂²u_2/∂η_k ∂η_l = 36 b_l³ r_l² δ_{kl}
    // R is built from R² = Σ b_k r_k² so
    //   ∂²R/∂η_k ∂η_l = 2 b_l r_l² δ_{kl} / R − b_k b_l r_k² r_l² / R³.
    let (
        g_rr,
        g_r_s1,
        g_r_s2,
        g_r_u1,
        g_r_u2,
        g_s1s1,
        _g_s1s2,
        g_s1u1,
        _g_s1u2,
        _g_s2s2,
        _g_s2u1,
        _g_s2u2,
        g_u1u1,
        _g_u1u2,
        _g_u2u2,
    ) = radial_g_q_hessian(q, big_r, s1, s2, u1, u2, &fr);
    // Per-axis ∂_{η_l} g (recompute; cheap).
    let bare_d_eta_g: Vec<f64> = (0..d)
        .map(|l| {
            g_r * dr_de[l]
                + g_s1 * ds1_de[l]
                + g_s2 * ds2_de[l]
                + g_u1 * du1_de[l]
                + g_u2 * du2_de[l]
        })
        .collect();
    for k in 0..d {
        for l in 0..d {
            // Mixed-partial term: Σ_{X,Y} g_{X,Y} (∂η_k Y) (∂η_l X).
            // Symmetric form using only nonzero Hessian entries.
            // R-X cross terms (X ∈ {R, s1, s2, u1, u2}, summed both ways):
            let dr_k = dr_de[k];
            let dr_l = dr_de[l];
            let ds1_k = ds1_de[k];
            let ds1_l = ds1_de[l];
            let ds2_k = ds2_de[k];
            let ds2_l = ds2_de[l];
            let du1_k = du1_de[k];
            let du1_l = du1_de[l];
            let du2_k = du2_de[k];
            let du2_l = du2_de[l];
            let mut hess_term = g_rr * dr_k * dr_l
                + g_r_s1 * (dr_k * ds1_l + ds1_k * dr_l)
                + g_r_s2 * (dr_k * ds2_l + ds2_k * dr_l)
                + g_r_u1 * (dr_k * du1_l + du1_k * dr_l)
                + g_r_u2 * (dr_k * du2_l + du2_k * dr_l);
            // Pure-invariant Hessian (only s1s1, s1u1, u1u1 nonzero for q ≤ 2):
            hess_term += g_s1s1 * ds1_k * ds1_l
                + g_s1u1 * (ds1_k * du1_l + du1_k * ds1_l)
                + g_u1u1 * du1_k * du1_l;

            // Second-derivative-of-invariant term: Σ_X g_X · ∂²_{η_k η_l} X.
            // Only diagonal pieces (k == l) for s1, s2, u1, u2; R has both
            // a diagonal δ_{kl} piece and an off-diagonal -b_k b_l r_k² r_l² / R³ piece.
            let kron = if k == l { 1.0 } else { 0.0 };
            let b_l_v = powers.b[l];
            let b_l_sq_v = powers.b2[l];
            let b_l_cu_v = powers.b3[l];
            let r_l_sq_v = r[l] * r[l];
            let b_k_v = powers.b[k];
            let r_k_sq_v = r[k] * r[k];
            let d2s1 = if k == l { 4.0 * b_l_v } else { 0.0 };
            let d2s2 = if k == l { 16.0 * b_l_sq_v } else { 0.0 };
            let d2u1 = if k == l {
                16.0 * b_l_sq_v * r_l_sq_v
            } else {
                0.0
            };
            let d2u2 = if k == l {
                36.0 * b_l_cu_v * r_l_sq_v
            } else {
                0.0
            };
            let d2r = {
                let r_inv = if big_r > 0.0 { 1.0 / big_r } else { 0.0 };
                let r_inv_cu = r_inv * r_inv * r_inv;
                kron * 2.0 * b_l_v * r_l_sq_v * r_inv
                    - b_k_v * b_l_v * r_k_sq_v * r_l_sq_v * r_inv_cu
            };
            let inv_term = g_r * d2r + g_s1 * d2s1 + g_s2 * d2s2 + g_u1 * d2u1 + g_u2 * d2u2;

            let bare_d2 = hess_term + inv_term;
            d2_eta[k][l] = big_j * (g + bare_d_eta_g[k] + bare_d_eta_g[l] + bare_d2);
        }
    }

    // Mixed ∂²_{η_l, κ}. Analytic via chain rule: differentiate ∂_κ g (which
    // is the same g_q expansion with fr → ∂_κ fr) w.r.t. η_l. Since the
    // invariants (R, s_1, ..., u_2) are κ-independent and g_q is linear
    // in F^{(k)}, we apply `radial_g_q_partials` to ∂_κ fr to obtain
    // (∂_κ g, ∂_κ g_R, ∂_κ g_s1, ..., ∂_κ g_u2), then chain through η_l:
    //   ∂²_{η_l κ} (J · g) = J · (∂_κ g + ∂_{η_l} ∂_κ g)
    //   ∂_{η_l} ∂_κ g = (∂_κ g_R) · ∂R/∂η_l
    //                  + (∂_κ g_s1) · ∂s1/∂η_l + (∂_κ g_s2) · ∂s2/∂η_l
    //                  + (∂_κ g_u1) · ∂u1/∂η_l + (∂_κ g_u2) · ∂u2/∂η_l.
    if let Some(dfr) = &dfr {
        let (dg, dg_r, dg_s1, dg_s2, dg_u1, dg_u2) =
            radial_g_q_partials(q, big_r, s1, s2, u1, u2, dfr);
        for l in 0..d {
            let bare_cross = dg_r * dr_de[l]
                + dg_s1 * ds1_de[l]
                + dg_s2 * ds2_de[l]
                + dg_u1 * du1_de[l]
                + dg_u2 * du2_de[l];
            d2_eta_kappa[l] = big_j * (dg + bare_cross);
        }
    }

    PairBlockBundle {
        value,
        d_eta,
        d_kappa,
        d2_eta,
        d2_eta_kappa,
        d2_kappa,
    }
}

/// Bare-kernel first derivative ∂g_q/∂η_k. The production bundle carries
/// derivatives of `J · g_q`; this unwraps the `J` contribution.
pub fn psi_first_derivative(
    q: usize,
    m: usize,
    s: usize,
    kappa: f64,
    eta: &[f64],
    r: &[f64],
    k: usize,
) -> f64 {
    assert!(
        k < eta.len(),
        "psi_first_derivative: axis index out of range"
    );
    let bundle = pair_block_radial_with_j_second_derivatives(q, m, s, kappa, eta, r);
    let big_j = eta.iter().sum::<f64>().exp();
    (bundle.d_eta[k] - bundle.value) / big_j
}

/// Bare-kernel second derivative ∂²g_q/∂η_k∂η_l.
pub fn psi_second_derivative(
    q: usize,
    m: usize,
    s: usize,
    kappa: f64,
    eta: &[f64],
    r: &[f64],
    k: usize,
    l: usize,
) -> f64 {
    assert!(
        k < eta.len() && l < eta.len(),
        "psi_second_derivative: axis index out of range"
    );
    let bundle = pair_block_radial_with_j_second_derivatives(q, m, s, kappa, eta, r);
    let big_j = eta.iter().sum::<f64>().exp();
    (bundle.d2_eta[k][l] - bundle.d_eta[k] - bundle.d_eta[l] + bundle.value) / big_j
}

/// Bare-kernel first derivative ∂g_q/∂κ.
pub fn kappa_first_derivative(
    q: usize,
    m: usize,
    s: usize,
    kappa: f64,
    eta: &[f64],
    r: &[f64],
) -> f64 {
    let bundle = pair_block_radial_with_j_second_derivatives(q, m, s, kappa, eta, r);
    bundle.d_kappa / eta.iter().sum::<f64>().exp()
}

/// Bare-kernel second derivative ∂²g_q/∂κ².
pub fn kappa_second_derivative(
    q: usize,
    m: usize,
    s: usize,
    kappa: f64,
    eta: &[f64],
    r: &[f64],
) -> f64 {
    let bundle = pair_block_radial_with_j_second_derivatives(q, m, s, kappa, eta, r);
    bundle.d2_kappa / eta.iter().sum::<f64>().exp()
}

/// Bare-kernel mixed derivative ∂²g_q/∂η_k∂κ.
pub fn psi_kappa_mixed_derivative(
    q: usize,
    m: usize,
    s: usize,
    kappa: f64,
    eta: &[f64],
    r: &[f64],
    k: usize,
) -> f64 {
    assert!(
        k < eta.len(),
        "psi_kappa_mixed_derivative: axis index out of range"
    );
    let bundle = pair_block_radial_with_j_second_derivatives(q, m, s, kappa, eta, r);
    let big_j = eta.iter().sum::<f64>().exp();
    (bundle.d2_eta_kappa[k] - bundle.d_kappa) / big_j
}

#[cfg(test)]
mod tests {
    use super::{bessel_k, bessel_k_half_integer, gamma_fn};

    /// I_ν(x) = (x/2)^ν Σ_{k=0}^∞ (x²/4)^k / (k! Γ(ν+k+1)).
    pub(crate) fn bessel_i_series(nu: f64, x: f64) -> f64 {
        let half_x = 0.5 * x;
        let half_x_sq = half_x * half_x;
        let prefix = (nu * half_x.ln()).exp();
        let mut term = 1.0 / gamma_fn(nu + 1.0);
        let mut sum = term;
        for k in 1..200 {
            term *= half_x_sq / (k as f64 * (nu + k as f64));
            sum += term;
            if term.abs() < 1e-18 * sum.abs() {
                break;
            }
        }
        prefix * sum
    }

    pub(crate) fn assert_relative_close(actual: f64, expected: f64, rel_tol: f64) {
        let scale = expected.abs();
        let diff = (actual - expected).abs();
        assert!(
            diff <= rel_tol * scale,
            "actual={actual} expected={expected} diff={diff} rel_tol={rel_tol}"
        );
    }

    #[test]
    pub(crate) fn bessel_k_matches_classic_k0_k1_values() {
        assert_relative_close(bessel_k(0.0, 1.0), 0.421_024_438_240_708_34, 1e-12);
        assert_relative_close(bessel_k(1.0, 1.0), 0.601_907_230_197_234_6, 1e-12);
    }

    #[test]
    pub(crate) fn bessel_k_large_order_at_moderate_x_matches_reference() {
        assert_relative_close(bessel_k(10.0, 3.0), 2_459.6, 1e-3);
    }

    #[test]
    pub(crate) fn bessel_k_half_integer_formula_and_nearby_orders_are_continuous() {
        for x in [0.5, 1.5, 3.0, 10.0] {
            let closed_form = bessel_k_half_integer(2, x);
            assert_relative_close(bessel_k(2.5, x), closed_form, 1e-14);
            assert_relative_close(bessel_k(2.5 - 1e-9, x), closed_form, 1e-6);
            assert_relative_close(bessel_k(2.5 + 1e-9, x), closed_form, 1e-6);
        }
    }

    #[test]
    pub(crate) fn bessel_k_satisfies_order_recurrence() {
        for nu in [0.3, 1.7, 6.2, 11.4] {
            for x in [0.5, 1.9, 2.1, 3.0, 8.0, 25.0] {
                let k_next = bessel_k(nu + 1.0, x);
                let residual = k_next - bessel_k(nu - 1.0, x) - (2.0 * nu / x) * bessel_k(nu, x);
                assert!(
                    residual.abs() <= 1e-10 * k_next.abs(),
                    "nu={nu} x={x} residual={residual} k_next={k_next}"
                );
            }
        }
    }

    #[test]
    pub(crate) fn bessel_i_k_wronskian_holds_for_small_x() {
        for nu in [0.0, 0.3, 1.7, 6.2] {
            for x in [0.5, 1.0, 1.9, 2.0] {
                let lhs = bessel_i_series(nu, x) * bessel_k(nu + 1.0, x)
                    + bessel_i_series(nu + 1.0, x) * bessel_k(nu, x);
                let expected = 1.0 / x;
                assert_relative_close(lhs, expected, 1e-10);
            }
        }
    }

    #[test]
    pub(crate) fn bessel_k_is_continuous_across_x_equals_two_dispatch() {
        for nu in [0.3, 4.6] {
            let left = bessel_k(nu, 2.0 - 1e-9);
            let right = bessel_k(nu, 2.0 + 1e-9);
            let diff = (left - right).abs();
            let scale = left.abs().max(right.abs()).max(1.0);
            assert!(
                diff <= 1e-8 * scale,
                "nu={nu} left={left} right={right} diff={diff}"
            );
        }
    }
}
