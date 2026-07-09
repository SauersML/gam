//! Shared Pólya–Gamma `PG(1, c)` sampler core — single source of the Devroye
//! algorithm's math and constants for every CPU/GPU host path.
//!
//! The exact `PG(1, c)` draw (Polson, Scott & Windle 2013; Devroye 1986) is
//! used in two places with two different randomness sources:
//!
//! * `inference::polya_gamma::PolyaGamma` — production Gibbs sampler driven by
//!   any [`rand::Rng`], with cached `Exp(1)` / `N(0,1)` distribution objects.
//! * `gpu::polya_gamma` — a host oracle driven by a bit-exact `XorwowState`
//!   that reproduces the device kernel's RNG byte stream, plus the embedded
//!   CUDA source that runs the same arithmetic on-device.
//!
//! Before this module the Devroye helpers (exponential tail mass, truncated
//! inverse-Gaussian small/large-`z` branches, the alternating-series tail
//! coefficient) and their constants were carried independently in each place,
//! so a formula or constant could silently drift between the CPU posterior
//! path and the GPU validation/device paths (issue #414).
//!
//! Here the math lives once, parameterised over a [`PgRng`] randomness source
//! so the same code runs against either driver. The CUDA device source is
//! *generated* from the same [`constants`] via [`render_cuda_constants`], so
//! the embedded kernel and the host code cannot diverge on a numeric literal.

use std::f64::consts::{FRAC_2_PI, PI};

/// Shared numeric constants for the `PG(1, c)` Devroye sampler. The CUDA
/// device source is rendered from these (see [`render_cuda_constants`]) so a
/// device literal can never drift from the host value.
pub mod constants {
    use std::f64::consts::PI;

    /// `π²`.
    pub const PI_SQ: f64 = PI * PI;

    /// `sqrt(2 / π)`.
    ///
    /// The `InverseGamma(α = 1/2, β = 2k²)` PDF coefficient is
    /// `(β^α / Γ(α)) = sqrt(2k²) / sqrt(π) = k · sqrt(2) / sqrt(π) = k · sqrt(2 / π)`,
    /// so the n-th series term in the small-`x` arm is
    /// `2 · InverseGamma(α = 1/2, β = 2k²) PDF = 2 · k · sqrt(2 / π) · x^{-3/2} · exp(-2k²/x)`.
    /// We fold the leading `2k` into `coeff = 2k · sqrt(2 / π)` at use sites.
    pub const SQRT_2_OVER_SQRT_PI: f64 = 0.797_884_560_802_865_4;

    /// `sqrt(π / 2)` — the standard-normal-scale factor inside the
    /// exponential-tail-mass calculation. Precomputed to avoid `sqrt` calls
    /// per PG draw setup.
    pub const SQRT_PI_OVER_2: f64 = 1.253_314_137_315_500_1;
}

use constants::{PI_SQ, SQRT_2_OVER_SQRT_PI, SQRT_PI_OVER_2};

/// Randomness source for the Devroye `PG(1, c)` sampler.
///
/// The sampler core consumes exactly three primitives — a uniform on `(0, 1]`,
/// a standard exponential, and a standard normal — and is otherwise pure. Each
/// caller supplies an adapter over its own RNG (a `rand::Rng` with cached
/// distributions, or a bit-exact `XorwowState`) so the math runs unchanged.
pub trait PgRng {
    /// Uniform variate. Callers must keep this on `(0, 1]` (strictly positive)
    /// so the inverse-CDF exponential never sees `log(0)`.
    fn next_unit(&mut self) -> f64;

    /// Standard exponential `Exp(1)` variate.
    fn next_exp(&mut self) -> f64;

    /// Standard normal `N(0, 1)` variate.
    fn next_norm(&mut self) -> f64;
}

/// Standard-normal CDF `Φ(x)`.
///
/// Evaluates `½·erfc(-x/√2)` through `libm::erfc` — the SunOS msun
/// implementation, which is accurate to within a ULP across the entire real
/// line. The same `½·erfc(-x/√2)` identity backs the device
/// `std_normal_cdf` JIT under `gpu::polya_gamma` (CUDA's `erfcf`/`erfc` is
/// itself derived from the msun implementation), so CPU and GPU posterior
/// callers see bit-identical tail masses.
///
/// `statrs::distribution::Normal::cdf` and `statrs::function::erf::erfc`
/// share a rational-approximation core with a `~10⁻¹¹` precision floor in
/// the bulk; routing the saddle-point tail-mass evaluator
/// `exponential_tail_mass` through that floor would have spoiled
/// `Φ(η ≈ ±1)` digits the GPU oracle considers correct and broken the
/// GPU/CPU parity gate.
#[inline]
pub fn std_normal_cdf(x: f64) -> f64 {
    let inv_sqrt2 = 1.0 / std::f64::consts::SQRT_2;
    0.5 * libm::erfc(-x * inv_sqrt2)
}

/// Largest `base·(2/π) + tilt` for which the folded direct evaluation of the
/// tail-mass terms stays inside binary64: beyond it `base · exp(base·2/π) ·
/// exp(tilt)` overflows to `+∞` while `Φ(lower)` underflows to `0`, producing
/// `∞ · 0 = NaN` from a function whose contract is a probability. The
/// crossover sits near half-tilt ≈ 42 (`|c| ≈ 84`), safely below the ≈ 46
/// overflow point, so the direct path — and its bit-exact agreement with the
/// CUDA kernel — is preserved everywhere the parity gate exercises.
const TAIL_MASS_DIRECT_MAX_LOG: f64 = 600.0;

/// Stable `ln Φ(x)`. In the bulk, `erfc` is exact to a ULP and the direct
/// logarithm is fine; once `erfc` underflows (x ≲ −38) switch to the leading
/// Mills-ratio asymptotic `ln Φ(x) ≈ −x²/2 − ln(−x) − ½·ln(2π)`.
#[inline]
fn log_std_normal_cdf(x: f64) -> f64 {
    let inv_sqrt2 = 1.0 / std::f64::consts::SQRT_2;
    let erfc_val = libm::erfc(-x * inv_sqrt2);
    if erfc_val > 0.0 {
        erfc_val.ln() - std::f64::consts::LN_2
    } else {
        const HALF_LOG_2PI: f64 = 0.918_938_533_204_672_7;
        -0.5 * x * x - (-x).ln() - HALF_LOG_2PI
    }
}

/// Exponential-tail acceptance mass for the proposal mixture (PSW 2013 §2;
/// Devroye 1986). Returns the probability of proposing from the exponential
/// right-tail arm rather than the truncated inverse-Gaussian arm.
///
/// `tilt` is the half-tilt `|c|/2`. In the bulk the factors are folded
/// directly as `base · exp(base·2/π) · exp(∓tilt) · Φ(·)`, saving two
/// transcendentals per draw setup. For extreme tilt that folding overflows
/// (`∞ · 0 = NaN` around `|c| ≈ 95`), so each term is assembled in log space
/// as `ln(base) + base·(2/π) ∓ tilt + ln Φ(·)` and exponentiated once —
/// the two regimes evaluate the same expression, only regrouped.
#[inline]
pub fn exponential_tail_mass(tilt: f64) -> f64 {
    let base = 0.125 * PI_SQ + 0.5 * tilt * tilt;
    let upper = SQRT_PI_OVER_2 * (FRAC_2_PI * tilt - 1.0);
    let lower = -(SQRT_PI_OVER_2 * (FRAC_2_PI * tilt + 1.0));
    let log_growth = base * FRAC_2_PI;
    let exp_terms = if log_growth + tilt <= TAIL_MASS_DIRECT_MAX_LOG {
        let base_factor = base * log_growth.exp();
        let p_upper = base_factor * (-tilt).exp() * std_normal_cdf(upper);
        let p_lower = base_factor * tilt.exp() * std_normal_cdf(lower);
        (4.0 / PI) * (p_upper + p_lower)
    } else {
        let log_base = base.ln();
        let log_p_upper = log_base + log_growth - tilt + log_std_normal_cdf(upper);
        let log_p_lower = log_base + log_growth + tilt + log_std_normal_cdf(lower);
        (4.0 / PI) * (log_p_upper.exp() + log_p_lower.exp())
    };
    1.0 / (1.0 + exp_terms)
}

/// `n`-th Devroye alternating-series coefficient `a_n(x)` of `J*(1, 0)`, with
/// `k = n + 1/2`. Left branch (`x ≤ 2/π`):
///   `a_n(x) = 2k · √(2/π) · x^{-3/2} · exp(-2k²/x)`.
/// Right branch (`x > 2/π`):
///   `a_n(x) = π · k · exp(-k² π² x / 2)`.
///
/// The right-tail coefficient carries the full factor `π · k`; the historical
/// `(π/2) · exp(...)` form dropped the `2k` and only coincidentally agreed at
/// `n = 0` (where `k = 1/2`).
#[inline]
pub fn series_coefficient(n: usize, x: f64) -> f64 {
    if x <= 0.0 {
        return 0.0;
    }
    let k0 = n as f64 + 0.5;
    let k_sq = k0 * k0;
    if x <= FRAC_2_PI {
        let coeff = 2.0 * k0 * SQRT_2_OVER_SQRT_PI;
        let inv_x = 1.0 / x;
        // x^{-3/2} = inv_x · sqrt(inv_x) — avoids a `powf`.
        coeff * inv_x * inv_x.sqrt() * (-2.0 * k_sq * inv_x).exp()
    } else {
        PI * k0 * (-0.5 * k_sq * PI_SQ * x).exp()
    }
}

/// Draw from the truncated inverse-Gaussian proposal `IG(z, 1)` restricted to
/// `(0, trunc]`, switching between the small-`z` and large-`z` Devroye branches
/// at `z = 2/π`.
#[inline]
pub fn sample_trunc_inv_gauss<R: PgRng + ?Sized>(rng: &mut R, z: f64, trunc: f64) -> f64 {
    let z = z.abs();
    if FRAC_2_PI > z {
        sample_small_z(rng, z, trunc)
    } else {
        sample_large_z(rng, 1.0 / z, trunc)
    }
}

#[inline]
fn sample_small_z<R: PgRng + ?Sized>(rng: &mut R, z: f64, trunc: f64) -> f64 {
    let mut accept = 0.0;
    let mut sample = 0.0;
    while accept < rng.next_unit() {
        let exp_sample = loop {
            let e1 = rng.next_exp();
            let e2 = rng.next_exp();
            if e1 * e1 <= 2.0 * e2 / trunc {
                break e1;
            }
        };
        sample = 1.0 + exp_sample * trunc;
        sample = trunc / (sample * sample);
        accept = (-0.5 * z * z * sample).exp();
    }
    sample
}

#[inline]
fn sample_large_z<R: PgRng + ?Sized>(rng: &mut R, mean: f64, trunc: f64) -> f64 {
    let mut sample = f64::INFINITY;
    while sample > trunc {
        let n = rng.next_norm();
        let n_sq = n * n;
        let half_mean = 0.5 * mean;
        let mn_sq = mean * n_sq;
        let disc = (4.0 * mn_sq + mn_sq * mn_sq).sqrt();
        sample = mean + half_mean * mn_sq - half_mean * disc;
        if rng.next_unit() > mean / (mean + sample) {
            sample = mean * mean / sample;
        }
    }
    sample
}

/// Draw a single `PG(1, c)` variate via Devroye's exact algorithm.
///
/// `tilt` is the raw tilt `c = ψ`; sign is irrelevant (the sampler uses `|c|`).
/// All randomness is drawn from `rng` via the three [`PgRng`] primitives, in a
/// fixed order so a deterministic `rng` (e.g. a seeded `XorwowState`) produces
/// a reproducible draw stream independent of which caller invokes the core.
pub fn draw_pg1<R: PgRng + ?Sized>(rng: &mut R, tilt: f64) -> f64 {
    let half_tilt = tilt.abs() * 0.5;
    let half_tilt_sq = half_tilt * half_tilt;
    let scale_factor = 0.125 * PI_SQ + 0.5 * half_tilt_sq;
    let exp_mass = exponential_tail_mass(half_tilt);

    loop {
        let u = rng.next_unit();
        let proposal = if u < exp_mass {
            FRAC_2_PI + rng.next_exp() / scale_factor
        } else {
            sample_trunc_inv_gauss(rng, half_tilt, FRAC_2_PI)
        };

        let mut series_sum = series_coefficient(0, proposal);
        let threshold = rng.next_unit() * series_sum;
        let mut idx = 0;

        loop {
            idx += 1;
            let term = series_coefficient(idx, proposal);
            if idx % 2 == 1 {
                series_sum -= term;
                if threshold <= series_sum {
                    return 0.25 * proposal;
                }
            } else {
                series_sum += term;
                if threshold >= series_sum {
                    break;
                }
            }
        }
    }
}

// ────────────────────────────────────────────────────────────────────────
// CUDA constant rendering — device source is generated from `constants`
// ────────────────────────────────────────────────────────────────────────

/// Render the `#define` block of `PG(1, c)` constants for the embedded CUDA
/// source. The device kernel `#include`s these definitions textually, so every
/// numeric literal the device uses for the Devroye math originates from the
/// Rust [`constants`] above — there is no second hand-typed copy to drift.
///
/// `{:.20e}` prints enough significant digits that the parsed `double` round-
/// trips to the exact host `f64`, so host and device share bit-identical
/// constants modulo NVRTC's literal parser (IEEE-correct to <1 ULP).
pub fn render_cuda_constants() -> String {
    format!(
        "#define PG_FRAC_2_PI       ({:.20e})\n\
         #define PG_PI              ({:.20e})\n\
         #define PG_PI_SQ           ({:.20e})\n\
         #define PG_SQRT_2_OVER_PI  ({:.20e})\n\
         #define PG_SQRT_PI_OVER_2  ({:.20e})\n",
        FRAC_2_PI, PI, PI_SQ, SQRT_2_OVER_SQRT_PI, SQRT_PI_OVER_2,
    )
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Hand-computed reference for `a_n(x)` independent of the production
    /// implementation, to lock the series coefficient against algebra drift.
    fn reference_a_n(n: usize, x: f64) -> f64 {
        let k = n as f64 + 0.5;
        let k_sq = k * k;
        let frac_2_pi = 2.0 / PI;
        if x <= frac_2_pi {
            let sqrt_2_over_pi = (2.0 / PI).sqrt();
            2.0 * k * sqrt_2_over_pi * x.powf(-1.5) * (-2.0 * k_sq / x).exp()
        } else {
            PI * k * (-0.5 * k_sq * PI * PI * x).exp()
        }
    }

    // -----------------------------------------------------------------------
    // exponential_tail_mass
    // -----------------------------------------------------------------------

    #[test]
    fn exponential_tail_mass_is_in_unit_interval() {
        for &tilt in &[0.0_f64, 0.5, 1.0, 2.0, 5.0, 10.0] {
            let p = exponential_tail_mass(tilt);
            assert!(
                p > 0.0 && p < 1.0,
                "exponential_tail_mass({tilt}) = {p} is outside (0,1)"
            );
        }
    }

    #[test]
    fn exponential_tail_mass_decreases_with_tilt() {
        // Larger |c| concentrates PG(1,c) mass away from the exponential
        // right-tail proposal arm, so the acceptance mass for that arm decreases.
        let tilts = [0.0_f64, 1.0, 2.0, 5.0, 10.0];
        let masses: Vec<f64> = tilts.iter().map(|&t| exponential_tail_mass(t)).collect();
        for w in masses.windows(2) {
            assert!(
                w[1] < w[0],
                "exponential_tail_mass should decrease with tilt, but {:.6} ≥ {:.6}",
                w[1],
                w[0]
            );
        }
    }

    #[test]
    fn exponential_tail_mass_extreme_tilt_is_a_probability_not_nan() {
        // Around |c| ≈ 95 (half-tilt ≈ 47.5) the direct folding forms
        // `∞ · 0 = NaN`; the log-space regrouping must keep the result a
        // genuine probability, monotone decreasing, all the way out.
        let mut prev = exponential_tail_mass(40.0);
        for &tilt in &[45.0_f64, 47.5, 60.0, 100.0, 500.0] {
            let p = exponential_tail_mass(tilt);
            assert!(
                p.is_finite() && (0.0..=1.0).contains(&p),
                "exponential_tail_mass({tilt}) = {p} is not a probability"
            );
            assert!(
                p <= prev,
                "tail mass must keep decreasing with tilt: p({tilt}) = {p} > {prev}"
            );
            prev = p;
        }
    }

    #[test]
    fn tail_mass_log_and_direct_regimes_agree_at_the_crossover() {
        // Just below the direct-evaluation ceiling both groupings are exact;
        // evaluating the log-space assembly by hand at a direct-regime tilt
        // must reproduce the production value to rounding.
        for &tilt in &[5.0_f64, 20.0, 35.0] {
            let base = 0.125 * PI_SQ + 0.5 * tilt * tilt;
            let upper = SQRT_PI_OVER_2 * (FRAC_2_PI * tilt - 1.0);
            let lower = -(SQRT_PI_OVER_2 * (FRAC_2_PI * tilt + 1.0));
            let log_growth = base * FRAC_2_PI;
            let hand_log = {
                let lp_u = base.ln() + log_growth - tilt + (std_normal_cdf(upper)).ln();
                let lp_l = base.ln() + log_growth + tilt + (std_normal_cdf(lower)).ln();
                1.0 / (1.0 + (4.0 / PI) * (lp_u.exp() + lp_l.exp()))
            };
            let prod = exponential_tail_mass(tilt);
            let rel = (prod - hand_log).abs() / prod.max(1e-300);
            assert!(
                rel < 1e-10,
                "regimes disagree at tilt {tilt}: direct {prod:.17e} vs log {hand_log:.17e}"
            );
        }
    }

    #[test]
    fn std_normal_cdf_symmetry() {
        for &x in &[0.5_f64, 1.0, 1.5, 2.0, 3.0] {
            let lo = std_normal_cdf(-x);
            let hi = std_normal_cdf(x);
            assert!(
                (lo + hi - 1.0).abs() < 1e-14,
                "Φ({x}) + Φ(-{x}) must equal 1.0; got {lo:.15} + {hi:.15}",
            );
        }
    }

    // -----------------------------------------------------------------------
    // series_coefficient
    // -----------------------------------------------------------------------

    #[test]
    fn series_coefficient_matches_reference() {
        for &x in &[0.1_f64, 0.5, 1.0, 2.0] {
            for n in 0..5 {
                let got = series_coefficient(n, x);
                let want = reference_a_n(n, x);
                let rel = (got - want).abs() / want.abs().max(1.0);
                assert!(
                    rel < 1e-14,
                    "a_n mismatch at n={n}, x={x}: got {got:.17e}, want {want:.17e}, rel={rel:.3e}",
                );
            }
        }
    }

    #[test]
    fn std_normal_cdf_matches_known_values() {
        // Φ(0) = 1/2, Φ(±∞) = 1/0, and a couple of tabulated points.
        assert!((std_normal_cdf(0.0) - 0.5).abs() < 1e-15);
        assert!((std_normal_cdf(1.0) - 0.841_344_746_068_542_9).abs() < 1e-12);
        assert!((std_normal_cdf(-1.0) - 0.158_655_253_931_457_1).abs() < 1e-12);
        assert!(std_normal_cdf(40.0) > 1.0 - 1e-15);
        assert!(std_normal_cdf(-40.0) < 1e-15);
    }

    #[test]
    fn rendered_cuda_constants_roundtrip_to_host() {
        // Every constant printed into the device `#define` block must parse
        // back to the exact host `f64`, proving the device sees the same bits.
        let src = render_cuda_constants();
        let parse = |name: &str| -> f64 {
            let line = src
                .lines()
                .find(|l| l.contains(name))
                .unwrap_or_else(|| panic!("missing #define {name}"));
            let inner = line
                .split_once('(')
                .and_then(|(_, rest)| rest.split_once(')'))
                .map(|(num, _)| num.trim())
                .expect("malformed #define");
            inner.parse::<f64>().expect("parse f64")
        };
        assert_eq!(parse("PG_FRAC_2_PI").to_bits(), FRAC_2_PI.to_bits());
        assert_eq!(parse("PG_PI").to_bits(), PI.to_bits());
        assert_eq!(parse("PG_PI_SQ").to_bits(), PI_SQ.to_bits());
        assert_eq!(
            parse("PG_SQRT_2_OVER_PI").to_bits(),
            SQRT_2_OVER_SQRT_PI.to_bits()
        );
        assert_eq!(
            parse("PG_SQRT_PI_OVER_2").to_bits(),
            SQRT_PI_OVER_2.to_bits()
        );
    }
}
