//! The hybrid Duchon–Matérn radial profile, built once per `(p, s, d)` and
//! certified against its own reference integral.
//!
//! ## The one function behind every hybrid kernel value
//!
//! The hybrid kernel with spectrum `|ω|^{-2p} (κ² + |ω|²)^{-s}` in `d`
//! dimensions has the single-integral form (gam#1424)
//!
//! ```text
//! φ(r; κ) = pref · ∫₀¹ (1−w)^{p−1} w^{s−1} · 2 (r/(2κ√w))^b K_b(κ r √w) dw,
//! b = p + s − d/2,   pref = (4π)^{-d/2} / (Γ(p) Γ(s)).
//! ```
//!
//! Substituting `ρ = κ r` moves every `κ` out of the integrand:
//!
//! ```text
//! φ(r; κ) = pref · κ^{-2b} · G(κ r),
//! G(ρ)    = ∫₀¹ (1−w)^{p−1} w^{s−1} · 2 (ρ/(2√w))^b K_b(ρ √w) dw,
//! ```
//!
//! so the whole family of length scales, and every radial derivative the
//! operator penalties consume (`φ^{(m)}(r) = pref κ^{m−2b} G^{(m)}(κ r)`), is
//! ONE univariate function per `(p, s, d)` and its first four derivatives.
//! This module represents `G, G′, …, G⁗` once per process and answers every
//! evaluation from that representation.
//!
//! ## Why a fixed generic quadrature per point was wrong
//!
//! The previous evaluator applied a 64-node Gauss–Legendre rule on
//! `w ∈ [0, 1]` to every `(row, center)` pair. Measured against an adaptive
//! reference (2026-09-04): for half-integer `b` the integrand carries the
//! endpoint factor `w^{s−1−b}` with a fractional exponent (`w^{-1/2}` for
//! `d = 5, p = 2, s = 2`), on which the rule converges only algebraically —
//! the kernel VALUE was 1.0 % off at every distance (0.7 % for `d = 3, p = 1,
//! s = 2`). For large `κ r` the integrand is a peak of width `~1/ρ²` at
//! `w → 0` that the rule cannot resolve: the relative error was `2e-5` at
//! `ρ = 30`, `1e-2` at `ρ = 100`, `7e-1` at `ρ = 300` and `100 %` at
//! `ρ = 1000`, while the true kernel decays only algebraically there
//! (`G(ρ) → 2^{d−2p} Γ(s−b) Γ(s) · ρ^{-(d−2p)}`, the polyharmonic tail that
//! IS the model at long range). It also cost 64 Bessel evaluations per kernel
//! value and `64 · (3 + 9 + 27 + 81)` per operator jet, per pair, on a design
//! that is streamed (re-evaluated on every product).
//!
//! ## Reference integral
//!
//! In `v = √w` the profile reads
//!
//! ```text
//! G^{(m)}(ρ) = 2^{2−b} ∫₀¹ v^{d−2p−1+m} (1−v²)^{p−1} T_m(ρ v) dv,
//! T_m(z)     = ∂_z^m [ z^b K_b(z) ],
//! ```
//!
//! (`∂_ρ = v ∂_z` at fixed `v`, and `2 (ρ/(2√w))^b K_b = 2^{1−b} v^{-2b} z^b
//! K_b(z)`), with an INTEGER power `d − 2p − 1 + m ≥ 0` of `v` at the left
//! endpoint. `T_m` is a short term list `Σ c · z^a · K_{b+j}(z)` obtained by
//! differentiating `z^a K_ν(z)` symbolically; the Bessel orders `b−4 … b+4`
//! come from two seeds and the upward recurrence, which is stable for `K`. The
//! integral is taken by the tanh–sinh (double-exponential) rule, split at
//! `v* = min(1, T/ρ)` with `T = ln(2/ε) + 3a` for the largest endpoint power
//! `a = d − 2p + 3`, so the `z^a e^{-z}` peak (`z = ρ v`) at large
//! `ρ` always owns a panel of its own scale (the remainder past `v*` is
//! integrated too, not argued away); the rule halves its step until two
//! successive levels agree to [`REFERENCE_RTOL`] of the absolute sum, and
//! refuses if they never do. The double-exponential clustering at `v = 0`
//! resolves the algebraic endpoint and the `z^{2b} ln z` terms of
//! integer-order `K_b` alike; a panel whose levels do not agree is bisected,
//! its halves judged on the parent's scale.
//!
//! ## Representation and certificate
//!
//! For each `m` the scaled channel `S_m(u) = G^{(m)}(e^u) · (1 + e^{2u})^{α_m/2}`,
//! `α_m = d − 2p + m`, is interpolated by Chebyshev panels in `u = ln ρ`,
//! shared by the five channels. The envelope removes the algebraic tail, so
//! every channel tends to a constant at large `ρ`. Panels are bisected until,
//! for every channel, BOTH
//! 1. the last two Chebyshev coefficients are below
//!    [`chebyshev_tail_tolerance`] of the largest — the geometric-decay
//!    certificate of an interpoland that is analytic in `u` (`K_b(e^u)`'s
//!    nearest singularity sits at `Im u = ±π`), and
//! 2. for the (positive) value channel, the panel's values span at most a
//!    factor `e` — so its per-panel absolute bound is a RELATIVE bound at
//!    every point of the panel, not only at its largest. A derivative channel
//!    crosses zero, where a relative bound is not a meaningful demand; its
//!    certificate is the absolute bound on the panel's scale.
//!
//! Outside the covered range the profile is its own closed-form limit: `G(0)
//! = Γ(b) B(s−b, p)` below [`rho_value_floor`] (the deviation is `O(ρ²)`,
//! below `ε` there), and the last panel's value times the exact envelope above
//! [`rho_ceiling`], which is the polyharmonic law with an `O(ρ_hi^{-2})`
//! relative remainder that is `ε` at `ρ_hi`. The derivative channels are
//! covered down to [`rho_derivative_floor`], two decades under the collision
//! floor every operator consumer applies.
//!
//! [`DuchonRadialProfile::resolution`] states the absolute error the
//! certificate guarantees at a point, so a consumer can test against the
//! profile's own bar instead of an invented one.

use super::*;
use gam_linalg::utils::KahanSum;
use std::collections::HashMap;
use std::sync::{Arc, Mutex, OnceLock};

/// Chebyshev coefficients per panel. A panel narrower than one e-fold whose
/// interpoland is analytic in a strip of half-width `π` converges faster than
/// `4^{-k}`, so 32 coefficients reach `f64` resolution with margin; wider
/// panels bisect until they do.
const PANEL_ORDER: usize = 32;

/// Highest radial derivative order the profile carries (`φ⁗` feeds `t_rr`).
pub(crate) const MAX_DERIVATIVE_ORDER: usize = 4;
const CHANNELS: usize = MAX_DERIVATIVE_ORDER + 1;
const ORDER_SHIFTS: usize = 2 * MAX_DERIVATIVE_ORDER + 1;

/// Relative agreement of two successive tanh–sinh levels, against the sum of
/// the absolute contributions. A converged level's own rounding is bounded by
/// `n · ε · Σ|terms|` for its 100–300 nodes, and the term evaluations feeding
/// it carry up to two lost digits at the series/recurrence crossover of
/// [`TermEvaluator`]; 256ε covers both.
pub(crate) const REFERENCE_RTOL: f64 = 256.0 * f64::EPSILON;

/// `z` below which an integer-order `z^b K_b(z)` and its derivatives are
/// taken from the ascending series (exact rational coefficients, no
/// cancellation) and above which from the Bessel recurrence term lists. At
/// this point each form loses at most two digits: the series' logarithmic
/// and regular parts cancel by a factor `~10` (`z^n I_n(z) ln(z/2)` against
/// `z^n K_n(z)`), and the recurrence's largest term `z^{b−4} K_{b+4}(z)`
/// exceeds `T_4(z)` by a factor `~70` for `b = 1`.
const SERIES_CROSSOVER_Z: f64 = 2.5;

/// Terms of the ascending Bessel series kept below the crossover: at
/// `z = 2.5` the `k`-th term `(z²/4)^k / (k! (n+k)!)` is under `ε` of the
/// first for `k ≥ 14`.
const SERIES_TERMS: usize = 20;

/// Euler–Mascheroni constant, `ψ(1) = −γ`.
const EULER_MASCHERONI: f64 = 0.577_215_664_901_532_9;

/// Bisection depth of the adaptive reference: `2^6` slivers of a panel that
/// is itself already the peak's own width leave nothing for a bisection to
/// resolve, so a failure past it is reported, not halved again.
const REFERENCE_MAX_BISECTIONS: u32 = 6;

/// One tanh–sinh panel's estimate and the numbers its convergence test saw.
struct PanelEstimate {
    values: [f64; CHANNELS],
    abs_sums: [f64; CHANNELS],
    last_delta: [f64; CHANNELS],
    converged: bool,
}

/// Ceiling on the last two Chebyshev coefficients relative to the largest: the
/// sampled values carry [`REFERENCE_RTOL`], and a 32-point cosine transform of
/// values with that noise cannot resolve a tail below a few multiples of it.
pub(crate) fn chebyshev_tail_tolerance() -> f64 {
    4.0 * REFERENCE_RTOL
}

/// Largest ratio `max|S| / min|S|` a sign-definite panel may span, so that the
/// certified absolute bound is at most `e` times a relative one anywhere in it.
const PANEL_DYNAMIC_RANGE: f64 = std::f64::consts::E;

/// tanh–sinh abscissa cut: at `|t| = 3` the node sits `5e-14` of the panel
/// width from its end and its weight is below the sum's resolution. Levels
/// halve the step from `1/8` down to `1/256`.
const TANH_SINH_T_MAX: f64 = 3.0;
const TANH_SINH_FIRST_LEVEL: u32 = 3;
const TANH_SINH_LAST_LEVEL: u32 = 8;

/// Below this `ρ` the deviation `G(ρ) − G(0)` is under `ε · G(0)`, so the
/// closed-form origin value is the profile. The deviation's leading power is
/// `ρ^{min(2, d−2p)}`: the analytic `ρ²` term, or the polyharmonic `ρ^{d−2p}`
/// branch when that is lower — a `d − 2p = 1` kernel has a linear cusp at
/// the origin, and its floor is `ε`, not `√ε`.
pub(crate) fn rho_value_floor(d: usize, p: usize) -> f64 {
    let leading = (d - 2 * p).min(2) as f64;
    0.1 * f64::EPSILON.powf(1.0 / leading)
}

/// The derivative channels' lower edge: two decades under the collision floor
/// every operator consumer applies (`DUCHON_DERIVATIVE_R_FLOOR_REL · ℓ`, i.e.
/// `ρ = 1e-5`), so that floor sits inside the certified range with margin.
pub(crate) fn rho_derivative_floor() -> f64 {
    DUCHON_DERIVATIVE_R_FLOOR_REL * 1e-2
}

/// The profile's upper edge. Past it the scaled channels are their limits up
/// to a relative `O(ρ^{-2})` remainder, which is `ε` at `ρ = 1/√ε`.
pub(crate) fn rho_ceiling() -> f64 {
    1.0 / f64::EPSILON.sqrt()
}

/// `z^b` for integer or half-integer `b` and `z > 0`.
fn pow_b(z: f64, b: f64) -> f64 {
    let two_b = (2.0 * b).round() as i32;
    if two_b % 2 == 0 {
        z.powi(two_b / 2)
    } else {
        z.powi(two_b / 2) * z.sqrt()
    }
}

/// One Chebyshev panel of the scaled channels on `u ∈ [u_lo, u_hi]`.
#[derive(Clone, Debug)]
struct Panel {
    u_lo: f64,
    u_hi: f64,
    coeff: [[f64; PANEL_ORDER]; CHANNELS],
    /// Largest `|S_m|` sampled on the panel, per channel: the scale the tail
    /// certificate is stated against.
    scale: [f64; CHANNELS],
}

impl Panel {
    /// Clenshaw evaluation of channel `m` at `u` (inside the panel).
    fn eval(&self, m: usize, u: f64) -> f64 {
        let x = (2.0 * u - (self.u_lo + self.u_hi)) / (self.u_hi - self.u_lo);
        let x2 = 2.0 * x;
        let coeff = &self.coeff[m];
        let mut b1 = 0.0_f64;
        let mut b2 = 0.0_f64;
        for &c in coeff.iter().skip(1).rev() {
            let b0 = c + x2 * b1 - b2;
            b2 = b1;
            b1 = b0;
        }
        0.5 * coeff[0] + x * b1 - b2
    }
}

/// The derivative-order term lists `T_m(z) = Σ_{k,j} coef[m][k][j] · z^{b−k} ·
/// K_{|b+j−4|}(z)`: `k` = number of power-lowering steps, `j − 4` = order
/// shift, produced by `∂_z [c z^a K_ν] = c a z^{a−1} K_ν − (c/2) z^a (K_{ν−1}
/// + K_{ν+1})`. After `m` steps the shifts stay within `±m`, so the index
/// arithmetic never leaves the table.
#[derive(Clone, Debug)]
struct TermTable {
    coef: [[[f64; ORDER_SHIFTS]; CHANNELS]; CHANNELS],
}

impl TermTable {
    fn new(b: f64) -> Self {
        let mut coef = [[[0.0_f64; ORDER_SHIFTS]; CHANNELS]; CHANNELS];
        coef[0][0][MAX_DERIVATIVE_ORDER] = 1.0;
        for m in 0..MAX_DERIVATIVE_ORDER {
            for k in 0..=m {
                let a = b - k as f64;
                for j in 1..ORDER_SHIFTS - 1 {
                    let c = coef[m][k][j];
                    if c == 0.0 {
                        continue;
                    }
                    if a != 0.0 {
                        coef[m + 1][k + 1][j] += c * a;
                    }
                    coef[m + 1][k][j - 1] -= 0.5 * c;
                    coef[m + 1][k][j + 1] -= 0.5 * c;
                }
            }
        }
        Self { coef }
    }

    /// `T_0(z) … T_4(z)` from `z`, `z^b` and the Bessel ladder
    /// `k[j] = K_{|b + j − 4|}(z)`.
    fn evaluate(&self, z: f64, z_pow_b: f64, k: &[f64; ORDER_SHIFTS]) -> [f64; CHANNELS] {
        let inv_z = 1.0 / z;
        let mut z_pow = [0.0_f64; CHANNELS];
        z_pow[0] = z_pow_b;
        for k_idx in 1..CHANNELS {
            z_pow[k_idx] = z_pow[k_idx - 1] * inv_z;
        }
        let mut out = [0.0_f64; CHANNELS];
        for (m, slot) in out.iter_mut().enumerate() {
            let mut acc = KahanSum::default();
            for (k_idx, row) in self.coef[m].iter().enumerate().take(m + 1) {
                for (j, &c) in row.iter().enumerate() {
                    if c != 0.0 {
                        acc.add(c * z_pow[k_idx] * k[j]);
                    }
                }
            }
            *slot = acc.sum();
        }
        out
    }
}

/// `K_{|b + j − 4|}(z)` for `j = 0..9`, from two seeds and the upward
/// recurrence `K_{ν+1} = K_{ν−1} + (2ν/z) K_ν` (stable for `K`), for integer
/// or half-integer `b > 0`.
fn bessel_k_ladder(b: f64, z: f64) -> [f64; ORDER_SHIFTS] {
    let two_b = (2.0 * b).round() as i64;
    let half_integer = two_b % 2 != 0;
    let top = (b - MAX_DERIVATIVE_ORDER as f64)
        .abs()
        .max((b + MAX_DERIVATIVE_ORDER as f64).abs());
    let count = if half_integer {
        (top - 0.5).round() as usize + 1
    } else {
        top.round() as usize + 1
    };
    let mut ladder = vec![0.0_f64; count];
    if half_integer {
        ladder[0] = bessel_k_half_integer_order(0, z);
        if count > 1 {
            ladder[1] = ladder[0] * (1.0 + 1.0 / z);
        }
        for idx in 2..count {
            let nu = (idx - 1) as f64 + 0.5;
            ladder[idx] = ladder[idx - 2] + 2.0 * nu * ladder[idx - 1] / z;
        }
    } else {
        ladder[0] = bessel_k_integer_order(0, z);
        if count > 1 {
            ladder[1] = bessel_k_integer_order(1, z);
        }
        for idx in 2..count {
            let nu = (idx - 1) as f64;
            ladder[idx] = ladder[idx - 2] + 2.0 * nu * ladder[idx - 1] / z;
        }
    }
    let mut out = [0.0_f64; ORDER_SHIFTS];
    for (j, slot) in out.iter_mut().enumerate() {
        let order = (b + j as f64 - MAX_DERIVATIVE_ORDER as f64).abs();
        let idx = if half_integer {
            (order - 0.5).round() as usize
        } else {
            order.round() as usize
        };
        *slot = ladder[idx];
    }
    out
}

/// tanh–sinh nodes of one refinement level on `(-1, 1)`: the abscissae NEW to
/// that level (every abscissa at the first level, the odd multiples of the
/// step afterwards), each as `(offset from the nearer endpoint in units of the
/// half-width, endpoint is the left one, weight including the step)`. The
/// offset `1 − |x| = 2 / (e^{2|y|} + 1)` is formed directly, so a node
/// `1e-14` from an endpoint keeps its full relative precision — that is what
/// resolves the endpoint power.
fn tanh_sinh_level(level: u32) -> Vec<(f64, bool, f64)> {
    let h = 2.0_f64.powi(-(level as i32));
    let first = level == TANH_SINH_FIRST_LEVEL;
    let max_k = (TANH_SINH_T_MAX / h).floor() as i64;
    let mut nodes = Vec::new();
    let mut k: i64 = if first { 0 } else { 1 };
    while k <= max_k {
        for sign in [-1.0_f64, 1.0] {
            if k == 0 && sign > 0.0 {
                continue;
            }
            let t = sign * k as f64 * h;
            let y = std::f64::consts::FRAC_PI_2 * t.sinh();
            let offset_unit = 2.0 / ((2.0 * y.abs()).exp() + 1.0);
            let weight_unit =
                std::f64::consts::FRAC_PI_2 * t.cosh() / (y.cosh() * y.cosh());
            nodes.push((offset_unit, t < 0.0, weight_unit * h));
        }
        k += if first { 1 } else { 2 };
    }
    nodes
}

fn tanh_sinh_levels() -> &'static [Vec<(f64, bool, f64)>] {
    static LEVELS: OnceLock<Vec<Vec<(f64, bool, f64)>>> = OnceLock::new();
    LEVELS.get_or_init(|| {
        (TANH_SINH_FIRST_LEVEL..=TANH_SINH_LAST_LEVEL)
            .map(tanh_sinh_level)
            .collect()
    })
}

/// A monomial `c · z^q`, optionally times `ln(z/2)`.
#[derive(Clone, Copy, Debug)]
struct Monomial {
    coef: f64,
    power: i32,
    logged: bool,
}

/// Merge like terms (same power, same log flag) and drop the zeros, so a
/// cancellation such as `1 − z − 1` is performed on the exact coefficients
/// and never at evaluation time — where it would leave an absolute `ε` on a
/// value of order `z`.
fn normalize_monomials(mut terms: Vec<Monomial>) -> Vec<Monomial> {
    terms.sort_by(|a, b| (a.logged, a.power).cmp(&(b.logged, b.power)));
    let mut out: Vec<Monomial> = Vec::with_capacity(terms.len());
    for term in terms {
        match out.last_mut() {
            Some(last) if last.logged == term.logged && last.power == term.power => {
                last.coef += term.coef;
            }
            _ => out.push(term),
        }
    }
    out.retain(|term| term.coef != 0.0);
    out
}

/// One derivative of a monomial list: `∂[c z^q] = c q z^{q−1}` and
/// `∂[c z^q ln(z/2)] = c q z^{q−1} ln(z/2) + c z^{q−1}`, merged.
fn differentiate_monomials(terms: &[Monomial]) -> Vec<Monomial> {
    let mut out = Vec::with_capacity(2 * terms.len());
    for &Monomial { coef, power, logged } in terms {
        if power != 0 {
            out.push(Monomial {
                coef: coef * power as f64,
                power: power - 1,
                logged,
            });
        }
        if logged {
            out.push(Monomial {
                coef,
                power: power - 1,
                logged: false,
            });
        }
    }
    normalize_monomials(out)
}

fn evaluate_monomials(terms: &[Monomial], z: f64, ln_half_z: f64) -> f64 {
    let mut plain = KahanSum::default();
    let mut logged = KahanSum::default();
    for &Monomial { coef, power, logged: is_logged } in terms {
        let value = coef * z.powi(power);
        if is_logged {
            logged.add(value);
        } else {
            plain.add(value);
        }
    }
    plain.sum() + ln_half_z * logged.sum()
}

/// `T_m(z) = ∂_z^m [z^b K_b(z)]`, `m = 0..=4`, evaluated without the
/// cancellation that any combination of Bessel values suffers at small `z`
/// once `m > 2b` (the leading `z^{-k-j}` orders of the recurrence terms
/// cancel exactly, leaving `T_m ~ z^{2b−m}` from terms `z^{-(2b+4)}` larger).
///
/// * Half-integer `|b| = n + ½`: `z^b K_{|b|}(z) = √(π/2) e^{-z} Σ_j
///   (n+j)!/(j!(n−j)!) 2^{-j} z^{b−½−j}` — a Laurent polynomial times
///   `e^{-z}`, differentiated exactly, valid at every `z` and for negative
///   `b` (a pure Matérn block, `2(p+s) ≤ d`, has `b ≤ 0`: the kernel is
///   singular at the origin but every `T_m(z)` is finite for `z > 0`).
/// * Integer `|b| = n`: below [`SERIES_CROSSOVER_Z`] the ascending series
///   (times `z^{-2n}` when `b < 0`)
///   `z^n K_n(z) = 2^{n−1} Σ_{k<n} (n−k−1)!/k! (−z²/4)^k + (−1)^{n+1} ln(z/2)
///   z^n I_n(z) + (−1)^n 2^{-n-1} z^{2n} Σ_k [ψ(k+1) + ψ(n+k+1)] (z²/4)^k /
///   (k!(n+k)!)`, differentiated term by term as monomials with and without
///   `ln(z/2)`; above it the Bessel recurrence term lists, where the terms
///   are comparable to the result.
#[derive(Clone, Debug)]
struct TermEvaluator {
    b: f64,
    recurrence: TermTable,
    mode: TermMode,
}

#[derive(Clone, Debug)]
enum TermMode {
    /// `T_m = √(π/2) e^{-z} Σ c z^q`: the Laurent polynomial of
    /// `z^b K_{|b|}(z) / (√(π/2) e^{-z})`, differentiated with
    /// `∂[e^{-z} z^q] = e^{-z} (q z^{q−1} − z^q)`. Valid at every `z` and for
    /// negative `b` (powers then run negative).
    HalfInteger {
        laurent: [Vec<Monomial>; CHANNELS],
    },
    Integer {
        series: [Vec<Monomial>; CHANNELS],
    },
}

impl TermEvaluator {
    fn new(b: f64) -> Self {
        let two_b = (2.0 * b).round() as i64;
        let recurrence = TermTable::new(b);
        let mode = if two_b % 2 != 0 {
            // |b| = n + ½: z^b K_{|b|}(z) = √(π/2) e^{-z} Σ_j (n+j)!/(j!(n−j)!) 2^{-j} z^{b−½−j}.
            let n = ((two_b.abs() - 1) / 2) as usize;
            let base_power = (two_b - 1) / 2; // b − ½ as an integer
            let mut base: Vec<Monomial> = Vec::with_capacity(n + 1);
            for j in 0..=n {
                let num: f64 = (1..=(n + j)).map(|k| k as f64).product();
                let den_j: f64 = (1..=j).map(|k| k as f64).product();
                let den_nj: f64 = (1..=(n - j)).map(|k| k as f64).product();
                base.push(Monomial {
                    coef: num / (den_j * den_nj * 2.0_f64.powi(j as i32)),
                    power: base_power as i32 - j as i32,
                    logged: false,
                });
            }
            let mut laurent: [Vec<Monomial>; CHANNELS] = Default::default();
            laurent[0] = normalize_monomials(base);
            for m in 1..CHANNELS {
                // ∂[e^{-z} Q] = e^{-z} (Q′ − Q), merged so the cancellations
                // happen on the exact coefficients.
                let mut next = differentiate_monomials(&laurent[m - 1]);
                next.extend(laurent[m - 1].iter().map(|t| Monomial {
                    coef: -t.coef,
                    power: t.power,
                    logged: false,
                }));
                laurent[m] = normalize_monomials(next);
            }
            TermMode::HalfInteger { laurent }
        } else {
            // |b| = n: the ascending series of z^n K_n(z); a negative `b`
            // is `z^{-2n}` times it, a shift of every power.
            let n = (two_b.abs() / 2) as usize;
            let power_shift: i32 = if two_b < 0 { -2 * n as i32 } else { 0 };
            let factorial = |k: usize| -> f64 { (1..=k).map(|i| i as f64).product() };
            let digamma_int = |k: usize| -> f64 {
                // ψ(k) = −γ + Σ_{i<k} 1/i for k ≥ 1.
                -EULER_MASCHERONI + (1..k).map(|i| 1.0 / i as f64).sum::<f64>()
            };
            let mut base: Vec<Monomial> = Vec::new();
            for k in 0..n {
                base.push(Monomial {
                    coef: 2.0_f64.powi(n as i32 - 1) * factorial(n - k - 1) / factorial(k)
                        * (-0.25_f64).powi(k as i32),
                    power: 2 * k as i32,
                    logged: false,
                });
            }
            let sign_n = if n % 2 == 0 { 1.0 } else { -1.0 };
            for k in 0..SERIES_TERMS {
                let shared = 0.25_f64.powi(k as i32) / (factorial(k) * factorial(n + k));
                base.push(Monomial {
                    coef: -sign_n * 2.0_f64.powi(-(n as i32)) * shared,
                    power: 2 * (n + k) as i32,
                    logged: true,
                });
                base.push(Monomial {
                    coef: sign_n * 2.0_f64.powi(-(n as i32) - 1)
                        * (digamma_int(k + 1) + digamma_int(n + k + 1))
                        * shared,
                    power: 2 * (n + k) as i32,
                    logged: false,
                });
            }
            for term in &mut base {
                term.power += power_shift;
            }
            let mut series: [Vec<Monomial>; CHANNELS] = Default::default();
            series[0] = normalize_monomials(base);
            for m in 1..CHANNELS {
                series[m] = differentiate_monomials(&series[m - 1]);
            }
            TermMode::Integer { series }
        };
        Self {
            b,
            recurrence,
            mode,
        }
    }

    /// `[T_0, …, T_4](z)` for `z > 0`.
    fn evaluate(&self, z: f64) -> [f64; CHANNELS] {
        match &self.mode {
            TermMode::HalfInteger { laurent } => {
                let scale = std::f64::consts::FRAC_PI_2.sqrt() * (-z).exp();
                std::array::from_fn(|m| scale * evaluate_monomials(&laurent[m], z, 0.0))
            }
            TermMode::Integer { series } => {
                if z <= SERIES_CROSSOVER_Z {
                    let ln_half_z = (0.5 * z).ln();
                    std::array::from_fn(|m| evaluate_monomials(&series[m], z, ln_half_z))
                } else {
                    let ladder = bessel_k_ladder(self.b, z);
                    self.recurrence.evaluate(z, pow_b(z, self.b), &ladder)
                }
            }
        }
    }
}

/// The reference integrand's shape parameters.
#[derive(Clone, Debug)]
struct ProfileShape {
    p: usize,
    s: usize,
    d: usize,
    b: f64,
    terms: TermEvaluator,
}

impl ProfileShape {
    /// `f_m(v)` for the first `channels` orders at once:
    /// `2^{2−b} v^{d−2p−1+m} (1−v²)^{p−1} T_m(ρ v)`.
    fn integrand(&self, rho: f64, v: f64) -> [f64; CHANNELS] {
        let z = rho * v;
        let t = self.terms.evaluate(z);
        let weight = (1.0 - v * v).powi(self.p as i32 - 1) * 2.0_f64.powf(2.0 - self.b);
        let mut v_pow = v.powi(self.d as i32 - 2 * self.p as i32 - 1);
        let mut out = [0.0_f64; CHANNELS];
        for (m, slot) in out.iter_mut().enumerate() {
            *slot = weight * v_pow * t[m];
            v_pow *= v;
        }
        out
    }

    /// tanh–sinh over `[a, b] ⊂ [0, 1]` of the first `channels` orders,
    /// refined until two successive levels agree to [`REFERENCE_RTOL`] of the
    /// absolute sum (or of `scale_floor`, whichever is larger) in every one of
    /// them. Reports whether that happened; the caller bisects otherwise.
    fn integrate_panel(
        &self,
        rho: f64,
        a: f64,
        b: f64,
        channels: usize,
        scale_floor: &[f64; CHANNELS],
    ) -> Result<PanelEstimate, BasisError> {
        let half = 0.5 * (b - a);
        let mut sums: [KahanSum; CHANNELS] = Default::default();
        let mut abs_sums = [0.0_f64; CHANNELS];
        let mut previous: Option<[f64; CHANNELS]> = None;
        let mut last_delta = [f64::NAN; CHANNELS];
        for (level_idx, level) in tanh_sinh_levels().iter().enumerate() {
            if level_idx > 0 {
                // Halving the step halves every earlier node's weight.
                for m in 0..channels {
                    let s = sums[m].sum();
                    sums[m] = KahanSum::default();
                    sums[m].add(0.5 * s);
                    abs_sums[m] *= 0.5;
                }
            }
            for &(offset_unit, left, weight_unit) in level {
                let v = if left {
                    a + half * offset_unit
                } else {
                    b - half * offset_unit
                };
                if v <= 0.0 || v >= 1.0 {
                    continue;
                }
                let f = self.integrand(rho, v);
                let w = half * weight_unit;
                for m in 0..channels {
                    if !f[m].is_finite() {
                        crate::bail_invalid_basis!(
                            "Duchon radial profile reference integrand is not finite at rho={rho:e}, \
                             v={v:e} (p={}, s={}, d={}, derivative order {m})",
                            self.p,
                            self.s,
                            self.d
                        );
                    }
                    let term = w * f[m];
                    sums[m].add(term);
                    abs_sums[m] += term.abs();
                }
            }
            let current: [f64; CHANNELS] = std::array::from_fn(|m| sums[m].sum());
            if let Some(prev) = previous {
                for m in 0..channels {
                    last_delta[m] = (current[m] - prev[m]).abs()
                        / abs_sums[m].max(scale_floor[m]).max(f64::MIN_POSITIVE);
                }
                if (0..channels).all(|m| last_delta[m] <= REFERENCE_RTOL) {
                    return Ok(PanelEstimate {
                        values: current,
                        abs_sums,
                        last_delta,
                        converged: true,
                    });
                }
            }
            previous = Some(current);
        }
        Ok(PanelEstimate {
            values: std::array::from_fn(|m| sums[m].sum()),
            abs_sums,
            last_delta,
            converged: false,
        })
    }

    /// Adaptive tanh–sinh over `[a, b]`: a panel whose levels do not agree is
    /// bisected, and its halves are judged on the parent's absolute scale (so
    /// the criterion stays "the whole integral to [`REFERENCE_RTOL`]", not
    /// "each sliver to its own"). An entire integrand such as `e^{-z}` over
    /// many decay lengths is where the double-exponential rule is slowest —
    /// halving the range is what restores its rate. Refuses past
    /// [`REFERENCE_MAX_BISECTIONS`] with the last level's measured deltas.
    fn integrate_adaptive(
        &self,
        rho: f64,
        a: f64,
        b: f64,
        channels: usize,
        scale_floor: &[f64; CHANNELS],
        depth: u32,
    ) -> Result<[f64; CHANNELS], BasisError> {
        let estimate = self.integrate_panel(rho, a, b, channels, scale_floor)?;
        if estimate.converged {
            return Ok(estimate.values);
        }
        if depth >= REFERENCE_MAX_BISECTIONS {
            let report: Vec<String> = (0..channels)
                .map(|m| {
                    format!(
                        "m={m}: |Δ|/scale={:.2e} at the last level (value {:e}, Σ|terms| {:e})",
                        estimate.last_delta[m], estimate.values[m], estimate.abs_sums[m]
                    )
                })
                .collect();
            crate::bail_invalid_basis!(
                "Duchon radial profile reference integral did not converge at rho={rho:e} on \
                 [{a:e}, {b:e}] after {depth} bisections (p={}, s={}, d={}; bar {:.2e}): {}",
                self.p,
                self.s,
                self.d,
                REFERENCE_RTOL,
                report.join("; ")
            );
        }
        let floor: [f64; CHANNELS] =
            std::array::from_fn(|m| scale_floor[m].max(estimate.abs_sums[m]));
        let mid = 0.5 * (a + b);
        let left = self.integrate_adaptive(rho, a, mid, channels, &floor, depth + 1)?;
        let right = self.integrate_adaptive(rho, mid, b, channels, &floor, depth + 1)?;
        Ok(std::array::from_fn(|m| left[m] + right[m]))
    }

    /// `G^{(m)}(ρ)` for `m < channels` by the reference integral.
    fn reference(&self, rho: f64, channels: usize) -> Result<[f64; CHANNELS], BasisError> {
        // The large-ρ integrand is `z^a e^{-z}` in `z = ρ v` with `a = d − 2p
        // − 1 + m` (the endpoint power): it peaks at `z = a` and has fallen to
        // `ε` of that peak by `z = a + ln(2/ε) + 2a` (`(z/a)^a e^{a−z}` is
        // under `ε` there for every `a ≥ 1`). Splitting there gives the peak a
        // panel of its own width, spanning no more decades than a double
        // resolves (a panel spanning `e^{-73}` converged only to `9e-14` at
        // `ρ = 1/√ε`, twice the bar), while the remainder — integrated too,
        // on the whole integral's scale — is below `ε` of the whole.
        let largest_power = (self.d as i32 - 2 * self.p as i32 - 1 + MAX_DERIVATIVE_ORDER as i32) as f64;
        let cut = (2.0 / f64::EPSILON).ln() + 3.0 * largest_power;
        let v_star = (cut / rho).min(1.0);
        let mut total = self.integrate_adaptive(rho, 0.0, v_star, channels, &[0.0; CHANNELS], 0)?;
        if v_star < 1.0 {
            // The remainder is under `ε` of the peak's contribution; its
            // convergence is judged on the scale of the whole integral, not
            // of its own (astronomically small, exponentially varying) sum.
            let floor: [f64; CHANNELS] = std::array::from_fn(|m| total[m].abs());
            let tail = self.integrate_adaptive(rho, v_star, 1.0, channels, &floor, 0)?;
            for m in 0..channels {
                total[m] += tail[m];
            }
        }
        Ok(total)
    }
}

/// Certified representation of `G, G′, …, G⁗` for one `(p, s, d)`.
#[derive(Clone, Debug)]
pub(crate) struct DuchonRadialProfile {
    shape: ProfileShape,
    /// `G(0) = Γ(b) B(s − b, p)` when `b > 0`; `None` for a kernel singular
    /// at the origin.
    g0: Option<f64>,
    u_value_lo: f64,
    u_lo: f64,
    u_hi: f64,
    /// Value-only panels on `[u_value_lo, u_lo]`.
    low: Vec<Panel>,
    /// All-channel panels on `[u_lo, u_hi]`.
    main: Vec<Panel>,
}

/// `(1 + ρ²)^{α/2}`, the envelope that turns `G^{(m)}` into a bounded channel.
fn envelope(alpha: i32, rho: f64) -> f64 {
    (1.0 + rho * rho).sqrt().powi(alpha)
}

impl DuchonRadialProfile {
    fn alpha(&self, m: usize) -> i32 {
        self.shape.d as i32 - 2 * self.shape.p as i32 + m as i32
    }

    /// Build and certify the profile for `(p, s, d)`.
    pub(crate) fn build(p: usize, s: usize, d: usize) -> Result<Self, BasisError> {
        if !(p >= 1 && s >= 1 && 2 * p < d) {
            // `p = 0` is a bare Matérn block: the Schwinger parametrization
            // that produces the single integral carries `1/Γ(p)`, which is not a
            // number there (the previous evaluator returned exactly `0` for it).
            crate::bail_invalid_basis!(
                "Duchon radial profile requires p ≥ 1, s ≥ 1 and 2p < d; got p={p}, s={s}, d={d}"
            );
        }
        let b = p as f64 + s as f64 - 0.5 * d as f64;
        let shape = ProfileShape {
            p,
            s,
            d,
            b,
            terms: TermEvaluator::new(b),
        };
        // `G(0) = Γ(b) B(s−b, p)` exists only for `b > 0`; a pure Matérn
        // block (`2(p+s) ≤ d`) is singular at the origin and its value channel
        // is answered from the covered range down to the floor.
        let g0 = if b > 0.0 {
            Some(
                gamma_lanczos(b) * gamma_lanczos(s as f64 - b) * gamma_lanczos(p as f64)
                    / gamma_lanczos(s as f64 - b + p as f64),
            )
        } else {
            None
        };
        let u_value_lo = rho_value_floor(d, p).ln();
        let u_lo = rho_derivative_floor().ln();
        let u_hi = rho_ceiling().ln();
        let build_start = std::time::Instant::now();
        let low = build_panels(&shape, 1, u_value_lo, u_lo)?;
        let main = build_panels(&shape, CHANNELS, u_lo, u_hi)?;
        let profile = Self {
            shape,
            g0,
            u_value_lo,
            u_lo,
            u_hi,
            low,
            main,
        };
        profile.spot_check()?;
        let (low_count, main_count) = profile.panel_counts();
        log::info!(
            "[duchon-profile] (p={p}, s={s}, d={d}): {low_count} value-only + {main_count} all-channel \
             panels certified in {:.3}s",
            build_start.elapsed().as_secs_f64()
        );
        Ok(profile)
    }

    /// Off-node certification: one interior point per panel (at 37 % of its
    /// width, never a Chebyshev node), every covered channel against the
    /// reference integral, within [`Self::resolution`]. The coefficient-tail
    /// test is a statement about the interpolant's own convergence; this is
    /// the independent check that the interpolant reproduces the reference
    /// off the grid it was built from.
    fn spot_check(&self) -> Result<(), BasisError> {
        let sets: [(&[Panel], usize); 2] = [(&self.low, 1), (&self.main, CHANNELS)];
        for (set, channels) in sets {
            for panel in set {
                let u = panel.u_lo + 0.37 * (panel.u_hi - panel.u_lo);
                let rho = u.exp();
                let reference = if channels == CHANNELS {
                    self.reference(rho)?
                } else {
                    self.shape.reference(rho, channels)?
                };
                for m in 0..channels {
                    let got = self.derivative(m, rho);
                    let bar = self.resolution(m, rho);
                    if !((got - reference[m]).abs() <= bar) {
                        crate::bail_invalid_basis!(
                            "Duchon radial profile (p={}, s={}, d={}) channel {m} misses its reference at \
                             rho={rho:e}: {got:e} vs {:e}, |Δ|={:e} > {bar:e}",
                            self.shape.p,
                            self.shape.s,
                            self.shape.d,
                            reference[m],
                            (got - reference[m]).abs()
                        );
                    }
                }
            }
        }
        Ok(())
    }

    /// `G(0)`, the closed-form origin value, or an error for a kernel that
    /// is singular there (`2(p+s) ≤ d`).
    pub(crate) fn origin_value(&self) -> Result<f64, BasisError> {
        self.g0.ok_or_else(|| {
            BasisError::InvalidInput(format!(
                "the hybrid Duchon kernel is singular at the origin for 2(p+s) ≤ d (p={}, s={}, d={})",
                self.shape.p, self.shape.s, self.shape.d
            ))
        })
    }

    /// The panel holding `u` for channel `m` and the abscissa to evaluate it
    /// at (clamped into the covered range).
    fn locate(&self, m: usize, u: f64) -> (&Panel, f64) {
        let (set, lo) = if m == 0 && u < self.u_lo {
            (&self.low, self.u_value_lo)
        } else {
            (&self.main, self.u_lo)
        };
        let u_eval = u.clamp(lo, self.u_hi);
        let idx = set
            .partition_point(|panel| panel.u_hi < u_eval)
            .min(set.len() - 1);
        (&set[idx], u_eval)
    }

    /// `G^{(m)}(ρ)` for `ρ > 0`.
    pub(crate) fn derivative(&self, m: usize, rho: f64) -> f64 {
        assert!(
            m <= MAX_DERIVATIVE_ORDER,
            "Duchon radial profile carries derivatives up to order {MAX_DERIVATIVE_ORDER}, asked {m}"
        );
        assert!(
            rho > 0.0 && rho.is_finite(),
            "Duchon radial profile needs a finite positive rho, got {rho}"
        );
        let u = rho.ln();
        if m == 0
            && u < self.u_value_lo
            && let Some(g0) = self.g0
        {
            return g0;
        }
        let (panel, u_eval) = self.locate(m, u);
        panel.eval(m, u_eval) / envelope(self.alpha(m), rho)
    }

    /// `G(ρ)`.
    pub(crate) fn value(&self, rho: f64) -> f64 {
        self.derivative(0, rho)
    }

    /// `[G, G′, G″, G‴, G⁗](ρ)`.
    pub(crate) fn derivatives(&self, rho: f64) -> [f64; CHANNELS] {
        std::array::from_fn(|m| self.derivative(m, rho))
    }

    /// The absolute error the certificate guarantees for `G^{(m)}(ρ)`, on the
    /// panel's scale and mapped through the envelope: the Chebyshev
    /// truncation (a tail of `tol · scale` in the last two coefficients bounds
    /// the neglected remainder by about twice that) plus the reference noise
    /// the interpolant inherits from its samples and the reference carries at
    /// the point compared against (twice [`REFERENCE_RTOL`]).
    pub(crate) fn resolution(&self, m: usize, rho: f64) -> f64 {
        let (panel, _) = self.locate(m, rho.ln());
        2.0 * (chebyshev_tail_tolerance() + REFERENCE_RTOL) * panel.scale[m]
            / envelope(self.alpha(m), rho)
    }

    /// The reference integral itself (all channels), for certification and
    /// tests.
    pub(crate) fn reference(&self, rho: f64) -> Result<[f64; CHANNELS], BasisError> {
        self.shape.reference(rho, CHANNELS)
    }

    /// Panel counts of the value-only and the all-channel sets (a build-size
    /// diagnostic).
    pub(crate) fn panel_counts(&self) -> (usize, usize) {
        (self.low.len(), self.main.len())
    }
}

/// Chebyshev-of-the-first-kind abscissae on `[-1, 1]`, cached.
fn chebyshev_nodes() -> &'static [f64; PANEL_ORDER] {
    static NODES: OnceLock<[f64; PANEL_ORDER]> = OnceLock::new();
    NODES.get_or_init(|| {
        std::array::from_fn(|i| {
            (std::f64::consts::PI * (i as f64 + 0.5) / PANEL_ORDER as f64).cos()
        })
    })
}

/// Coefficients of the interpolant through the samples at [`chebyshev_nodes`].
fn chebyshev_coefficients(values: &[f64; PANEL_ORDER]) -> [f64; PANEL_ORDER] {
    let n = PANEL_ORDER as f64;
    std::array::from_fn(|k| {
        let mut acc = KahanSum::default();
        for (i, &v) in values.iter().enumerate() {
            acc.add(v * (std::f64::consts::PI * k as f64 * (i as f64 + 0.5) / n).cos());
        }
        2.0 * acc.sum() / n
    })
}

enum PanelVerdict {
    Certified(Panel),
    /// Not certified: per channel `(tail / largest coefficient, max / min of a
    /// sign-definite panel or 0)`, the numbers the certificate measured.
    Split([(f64, f64); CHANNELS]),
}

fn sample_panel(
    shape: &ProfileShape,
    channels: usize,
    u_lo: f64,
    u_hi: f64,
) -> Result<PanelVerdict, BasisError> {
    use rayon::prelude::*;
    // The 32 reference integrals of a panel are independent: sample them
    // across the pool (the first bisection rounds have few panels to spread).
    let sampled: Vec<Result<[f64; CHANNELS], BasisError>> = chebyshev_nodes()
        .par_iter()
        .map(|&x| {
            let u = 0.5 * (u_lo + u_hi) + 0.5 * (u_hi - u_lo) * x;
            let rho = u.exp();
            let g = shape.reference(rho, channels)?;
            Ok(std::array::from_fn(|m| {
                let alpha = shape.d as i32 - 2 * shape.p as i32 + m as i32;
                g[m] * envelope(alpha, rho)
            }))
        })
        .collect();
    let mut values = [[0.0_f64; PANEL_ORDER]; CHANNELS];
    for (i, node) in sampled.into_iter().enumerate() {
        let node = node?;
        for m in 0..channels {
            values[m][i] = node[m];
        }
    }
    let mut coeff = [[0.0_f64; PANEL_ORDER]; CHANNELS];
    let mut scale = [0.0_f64; CHANNELS];
    let mut certified = true;
    let mut measured = [(0.0_f64, 0.0_f64); CHANNELS];
    for m in 0..channels {
        let samples = &values[m];
        scale[m] = samples.iter().fold(0.0_f64, |acc, v| acc.max(v.abs()));
        if !scale[m].is_finite() {
            crate::bail_invalid_basis!(
                "Duchon radial profile channel {m} is not finite on u ∈ [{u_lo:.3}, {u_hi:.3}] \
                 (p={}, s={}, d={})",
                shape.p,
                shape.s,
                shape.d
            );
        }
        coeff[m] = chebyshev_coefficients(samples);
        let largest = coeff[m].iter().fold(0.0_f64, |acc, c| acc.max(c.abs()));
        let tail = coeff[m][PANEL_ORDER - 1]
            .abs()
            .max(coeff[m][PANEL_ORDER - 2].abs());
        let tail_ok = tail <= chebyshev_tail_tolerance() * largest;
        // The value channel is positive everywhere, so its per-panel absolute
        // bound is made a relative one by capping the panel's dynamic range.
        // A derivative channel crosses zero, and on the approach to a zero a
        // relative bound is not a meaningful demand: there the certificate is
        // the absolute bound on the panel's scale, which `resolution` reports.
        let sign_definite = m == 0
            && (samples.iter().all(|v| *v > 0.0) || samples.iter().all(|v| *v < 0.0));
        let range = if sign_definite {
            let smallest = samples.iter().fold(f64::INFINITY, |acc, v| acc.min(v.abs()));
            scale[m] / smallest
        } else {
            0.0
        };
        let range_ok = range <= PANEL_DYNAMIC_RANGE;
        measured[m] = (if largest > 0.0 { tail / largest } else { 0.0 }, range);
        certified &= tail_ok && range_ok;
    }
    if certified {
        Ok(PanelVerdict::Certified(Panel {
            u_lo,
            u_hi,
            coeff,
            scale,
        }))
    } else {
        Ok(PanelVerdict::Split(measured))
    }
}

/// Adaptive bisection of `[u_lo, u_hi]` until every panel certifies for the
/// first `channels` channels; the frontier of one round is sampled in
/// parallel.
fn build_panels(
    shape: &ProfileShape,
    channels: usize,
    u_lo: f64,
    u_hi: f64,
) -> Result<Vec<Panel>, BasisError> {
    use rayon::prelude::*;
    // A panel narrower than this has been bisected far past what any analytic
    // interpoland needs (a width under one e-fold already converges faster
    // than `4^{-k}`): the certificate is then failing on something else, which
    // is an error to report with the numbers it measured, not a width to
    // keep halving.
    let min_width = (u_hi - u_lo) * 2.0_f64.powi(-10);
    let mut certified: Vec<Panel> = Vec::new();
    let mut frontier = vec![(u_lo, u_hi)];
    while !frontier.is_empty() {
        let verdicts: Vec<Result<(f64, f64, PanelVerdict), BasisError>> = frontier
            .par_iter()
            .map(|&(a, b)| sample_panel(shape, channels, a, b).map(|v| (a, b, v)))
            .collect();
        let mut next = Vec::new();
        for verdict in verdicts {
            let (a, b, verdict) = verdict?;
            match verdict {
                PanelVerdict::Certified(panel) => certified.push(panel),
                PanelVerdict::Split(measured) => {
                    if b - a < min_width {
                        let report: Vec<String> = measured
                            .iter()
                            .take(channels)
                            .enumerate()
                            .map(|(m, (tail, range))| {
                                format!("m={m}: tail/largest={tail:.2e} range={range:.2e}")
                            })
                            .collect();
                        crate::bail_invalid_basis!(
                            "Duchon radial profile does not certify on u ∈ [{a:.6}, {b:.6}] \
                             (p={}, s={}, d={}; bars tail ≤ {:.2e}, range ≤ {:.3}): {}",
                            shape.p,
                            shape.s,
                            shape.d,
                            chebyshev_tail_tolerance(),
                            PANEL_DYNAMIC_RANGE,
                            report.join("; ")
                        );
                    }
                    let mid = 0.5 * (a + b);
                    next.push((a, mid));
                    next.push((mid, b));
                }
            }
        }
        frontier = next;
    }
    certified.sort_by(|x, y| x.u_lo.total_cmp(&y.u_lo));
    Ok(certified)
}

/// The process-wide profile for `(p, s, d)`, built on first use.
pub(crate) fn duchon_radial_profile(
    p: usize,
    s: usize,
    d: usize,
) -> Result<Arc<DuchonRadialProfile>, BasisError> {
    static PROFILES: OnceLock<Mutex<HashMap<(usize, usize, usize), Arc<DuchonRadialProfile>>>> =
        OnceLock::new();
    let cache = PROFILES.get_or_init(|| Mutex::new(HashMap::new()));
    if let Some(profile) = cache
        .lock()
        .expect("Duchon radial profile cache poisoned")
        .get(&(p, s, d))
    {
        return Ok(Arc::clone(profile));
    }
    let built = Arc::new(DuchonRadialProfile::build(p, s, d)?);
    let mut guard = cache.lock().expect("Duchon radial profile cache poisoned");
    Ok(Arc::clone(guard.entry((p, s, d)).or_insert(built)))
}

#[cfg(test)]
mod tests {
    use super::*;

    /// `(d, p, s)` shapes spanning integer and half-integer `b`, the
    /// production 16-D CTN order, and the low-dimensional cases whose
    /// `w^{-1/2}` endpoint defeated the old rule.
    const SHAPES: [(usize, usize, usize); 13] = [
        (6, 1, 3),
        (6, 2, 2),
        (5, 1, 3),
        (5, 2, 2),
        (3, 1, 2),
        (4, 1, 3),
        (16, 1, 9),
        (10, 2, 5),
        (9, 3, 4),
        // b = ½, and the #979 3-D order-0 power-9 gate (b = 8.5).
        (3, 1, 1),
        (3, 1, 9),
        // Kernels singular at the origin (2(p+s) ≤ d): b = 0 and −2.
        (10, 1, 4),
        (16, 2, 4),
    ];

    /// Log-uniform probe radii across the derivative range, off the panel
    /// nodes.
    fn probe_radii() -> Vec<f64> {
        let lo = rho_derivative_floor().ln();
        let hi = rho_ceiling().ln();
        (0..40)
            .map(|i| (lo + (i as f64 + 0.37) / 40.0 * (hi - lo)).exp())
            .collect()
    }

    #[test]
    fn every_channel_matches_its_reference_integral_within_the_certificate() {
        for &(d, p, s) in &SHAPES {
            let profile = duchon_radial_profile(p, s, d).expect("profile builds");
            let (low, main) = profile.panel_counts();
            assert!(low >= 1 && main >= 4, "(d={d}, p={p}, s={s}) panel counts {low}/{main}");
            for rho in probe_radii() {
                let reference = profile.reference(rho).expect("reference converges");
                for m in 0..CHANNELS {
                    let got = profile.derivative(m, rho);
                    let bar = profile.resolution(m, rho);
                    assert!(
                        (got - reference[m]).abs() <= bar,
                        "(d={d}, p={p}, s={s}) m={m} rho={rho:.3e}: profile {got:.16e} vs reference \
                         {:.16e}, |Δ|={:.3e} > resolution {bar:.3e}",
                        reference[m],
                        (got - reference[m]).abs()
                    );
                }
            }
        }
    }

    #[test]
    fn the_value_channel_below_the_derivative_floor_matches_its_reference() {
        for &(d, p, s) in &SHAPES {
            let profile = duchon_radial_profile(p, s, d).expect("profile builds");
            for rho in [2.0e-9_f64, 1.0e-8, 3.0e-8, 2.0e-7]
                .into_iter()
                .filter(|&rho| rho > rho_value_floor(d, p))
            {
                let reference = profile.reference(rho).expect("reference converges")[0];
                let got = profile.value(rho);
                let bar = profile.resolution(0, rho);
                assert!(
                    (got - reference).abs() <= bar,
                    "(d={d}, p={p}, s={s}) rho={rho:.1e}: {got:.16e} vs {reference:.16e} (bar {bar:.2e})"
                );
            }
        }
    }

    #[test]
    fn the_origin_value_is_the_closed_form_and_the_profile_reaches_it() {
        for &(d, p, s) in SHAPES.iter().filter(|(d, p, s)| 2 * (p + s) > *d) {
            let profile = duchon_radial_profile(p, s, d).expect("profile builds");
            let g0 = profile.origin_value().expect("b > 0 has an origin value");
            let rho = 1.5 * rho_value_floor(d, p);
            let near = profile.value(rho);
            let bar = profile.resolution(0, rho) + 8.0 * f64::EPSILON * g0;
            assert!(
                (near - g0).abs() <= bar,
                "(d={d}, p={p}, s={s}): G(ρ_lo) = {near:.16e} vs G(0) = {g0:.16e} (bar {bar:.2e})"
            );
            let below = profile.value(0.5 * rho_value_floor(d, p));
            assert_eq!(below.to_bits(), g0.to_bits(), "below the value floor the profile is G(0)");
        }
    }

    /// The long-range law is the polyharmonic tail `C ρ^{-(d−2p)}` with
    /// `C = 2^{d−2p} Γ(s−b) Γ(s)` (the `t = ρ v` substitution with
    /// `∫₀^∞ t^{μ−1} K_b(t) dt = 2^{μ−2} Γ((μ−b)/2) Γ((μ+b)/2)`); for `p = 1`
    /// the only remainder is
    /// exponentially small, so at `ρ = 10³` the profile must sit on it.
    #[test]
    fn the_large_radius_profile_is_the_polyharmonic_tail() {
        for &(d, p, s) in SHAPES.iter().filter(|(_, p, _)| *p == 1) {
            let profile = duchon_radial_profile(p, s, d).expect("profile builds");
            let b = p as f64 + s as f64 - 0.5 * d as f64;
            let tail_constant = 2.0_f64.powi(d as i32 - 2 * p as i32)
                * gamma_lanczos(s as f64 - b)
                * gamma_lanczos(s as f64);
            for rho in [1.0e3_f64, 1.0e5, 1.0e9] {
                let law = tail_constant * rho.powi(-(d as i32 - 2 * p as i32));
                let got = profile.value(rho);
                assert!(
                    ((got - law) / law).abs() <= 1e-12,
                    "(d={d}, p={p}, s={s}): G({rho:.0e}) = {got:.16e} vs tail law {law:.16e}"
                );
            }
        }
    }

    /// Independent oracle: `scipy.integrate.quad` (QUADPACK, `epsrel=1e-13`,
    /// split at `v = 40/ρ`) on the same integrand with the derivative term
    /// lists coded separately in Python (`duchon_oracle.py`, 2026-09-04; its
    /// `dw = 2v dv` Jacobian was written as `4v`, so its rows are halved here
    /// — the closed-form `G(0) = Γ(b) B(s−b, p)` and the 64-node rule's own
    /// small-ρ values fix the normalization independently).
    /// Rows are `(d, p, s, ρ, G, G′, G″, G‴, G⁗)`; only rows whose quad error
    /// estimate was under `1e-12` of the value in every channel are kept,
    /// and the bar is ten times that estimate.
    #[test]
    fn the_profile_agrees_with_an_independent_quadpack_oracle() {
        let rows: [(usize, usize, usize, f64, [f64; 5]); 27] = [
            (6, 1, 3, 3.0e-01, [4.6985918744347777e-01, -1.5280509899838315e-01, -1.9816847111553884e-01, 9.2561975226344528e-01, -3.9694341175201573e+00]),
            (6, 1, 3, 3.0e+00, [1.0914697710866623e-01, -6.5216440555166594e-02, 3.9215058819385841e-02, -2.1276924973236151e-02, 6.9381035837886417e-03]),
            (6, 1, 3, 3.0e+01, [3.9506172790671214e-05, -5.2674896620681878e-06, 8.7791490102848121e-07, -1.7558293827270790e-07, 4.0969311059236578e-08]),
            (6, 1, 3, 3.0e+02, [3.9506172839506188e-09, -5.2674897119341561e-11, 8.7791495198902613e-13, -1.7558299039780523e-14, 4.0969364426154538e-16]),
            (6, 2, 2, 1.0e+00, [4.0505129895177810e-01, -1.1956079034782202e-01, 4.1440616094682572e-03, 7.2017547769282639e-02, -1.6811933626324055e-01]),
            (6, 2, 2, 1.0e+01, [3.6810901080102675e-02, -6.7329643588438281e-03, 1.7756624741639996e-03, -5.9526359784749206e-04, 2.3537927072176507e-04]),
            (6, 2, 2, 3.0e+02, [4.4440493827160507e-05, -2.9624362139917695e-07, 2.9620850480109740e-09, -3.9488614540466389e-11, 6.5802652034750789e-13]),
            (6, 2, 2, 1.0e+04, [3.9999996800000020e-08, -7.9999987200000017e-12, 2.3999993600000012e-15, -9.5999961600000003e-19, 4.7999973120000002e-22]),
            (5, 1, 3, 3.0e-01, [5.7728516709470501e-01, -8.2898535984488389e-02, -2.0775229494176811e-01, 3.9871733020347994e-01, -5.3363865834983748e-01]),
            (5, 1, 3, 1.0e+01, [1.4059892518100654e-02, -4.1294515470580041e-03, 1.5713113384848453e-03, -7.1323331693790200e-04, 3.6356456589205617e-04]),
            (5, 1, 3, 1.0e+03, [1.4179630807244130e-08, -4.2538892421732390e-11, 1.7015556968692956e-13, -8.5077784843464781e-16, 5.1046670906078874e-18]),
            (5, 2, 2, 1.0e+00, [1.1021648736861631e+00, -1.2897521047617824e-01, -5.3421239984428652e-02, 1.0165169935384380e-01, -1.1354669441816742e-01]),
            (5, 2, 2, 3.0e+01, [1.1763841854900088e-01, -3.8862691842207011e-03, 2.5558346888487324e-04, -2.5091527780010377e-05, 3.2677338632772836e-06]),
            (5, 2, 2, 3.0e+02, [1.1815833834525396e-02, -3.9382611638342123e-05, 2.6251573282152214e-07, -2.6246905090939952e-09, 3.4988093135899478e-11]),
            (3, 1, 2, 3.0e-01, [1.7495188441799354e+00, -1.4177634563722352e-01, -3.6789047048678736e-01, 6.1508378604185920e-01, -7.6040920150538838e-01]),
            (3, 1, 2, 3.0e+00, [1.0345604321804829e+00, -2.2719310265493881e-01, 6.3216787382709264e-02, -4.3866000134315120e-03, -2.3566293666730211e-02]),
            (3, 1, 2, 1.0e+02, [3.5449077018110321e-02, -3.5449077018110320e-04, 7.0898154036220673e-06, -2.1269446210866188e-07, 8.5077784843464801e-09]),
            (4, 1, 3, 1.0e+00, [8.9873717526205488e-01, -1.7263545188893351e-01, -8.4000874530434450e-02, 1.5512070616520965e-01, -1.1443744630115851e-01]),
            (4, 1, 3, 3.0e+01, [8.8888888881474906e-03, -5.9259259186006819e-04, 5.9259258535687199e-05, -7.9012338533590316e-06, 1.3168717225478964e-06]),
            (8, 1, 4, 3.0e+00, [6.0953872337774703e-02, -4.1594882419161060e-02, 2.7575716872150619e-02, -1.6382052326866139e-02, 6.4381022935026989e-03]),
            (8, 1, 4, 1.0e+02, [7.6800000000000004e-10, -4.6079999999999998e-11, 3.2255999999999995e-12, -2.5804799999999990e-13, 2.3224319999999999e-14]),
            (16, 1, 9, 1.0e+00, [1.1886003289332642e-01, -3.9201561871392100e-02, -1.3883802126352989e-02, 4.1258042065121049e-02, -4.0220630582165504e-02]),
            (16, 1, 9, 1.0e+01, [3.3036650726599696e-04, -2.4741494010306798e-04, 1.8463467561634622e-04, -1.3626363127830295e-04, 9.8500625583362153e-05]),
            (16, 1, 9, 3.0e+02, [9.9443269149350568e-24, -4.6406858936363596e-25, 2.3203429468181794e-26, -1.2375162383030290e-27, 7.0125920170504993e-29]),
            (10, 2, 5, 3.0e+00, [3.6612100958410623e-02, -1.5401471911417455e-02, 4.7637380557147158e-03, 5.0292547448374211e-04, -2.8214782576543706e-03]),
            (10, 2, 5, 1.0e+03, [3.0718156800000003e-15, -1.8430525439999997e-17, 1.2901072896000000e-19, -1.0320592895999999e-21, 9.2882681855999984e-24]),
            (9, 3, 4, 1.0e+01, [2.7040056431742739e-02, -5.8916911171334472e-03, 1.5341464221588009e-03, -4.3208855617272632e-04, 1.1618407984814464e-04]),
        ];
        for &(d, p, s, rho, want) in &rows {
            let profile = duchon_radial_profile(p, s, d).expect("profile builds");
            let got = profile.derivatives(rho);
            for m in 0..CHANNELS {
                let rel = ((got[m] - want[m]) / want[m]).abs();
                assert!(
                    rel <= 1e-11,
                    "(d={d}, p={p}, s={s}) rho={rho}: m={m} profile {:.16e} vs QUADPACK {:.16e} (rel {rel:.2e})",
                    got[m],
                    want[m]
                );
            }
        }
    }

    /// The derivative channels are the derivatives of the value channel: a
    /// central difference of `G^{(m)}` at the step `ρ ε^{1/3}` agrees with
    /// `G^{(m+1)}` to the difference's own truncation-plus-rounding error,
    /// which needs `G^{(m+3)}` and so is stated for `m ≤ 1`.
    #[test]
    fn derivative_channels_are_finite_differences_of_the_lower_channel() {
        for &(d, p, s) in &SHAPES {
            let profile = duchon_radial_profile(p, s, d).expect("profile builds");
            for rho in [0.3_f64, 1.0, 4.0, 20.0, 200.0] {
                for m in 0..2 {
                    let h = rho * f64::EPSILON.cbrt();
                    let fd = (profile.derivative(m, rho + h) - profile.derivative(m, rho - h))
                        / (2.0 * h);
                    let exact = profile.derivative(m + 1, rho);
                    let truncation = h * h / 6.0 * profile.derivative(m + 3, rho).abs();
                    let rounding = 2.0 * profile.resolution(m, rho) / h
                        + 4.0 * f64::EPSILON * profile.derivative(m, rho).abs() / h;
                    let bar = 2.0 * (truncation + rounding) + profile.resolution(m + 1, rho);
                    assert!(
                        (fd - exact).abs() <= bar,
                        "(d={d}, p={p}, s={s}) m={m} rho={rho}: FD {fd:.10e} vs channel {exact:.10e}, \
                         |Δ|={:.3e} > {bar:.3e}",
                        (fd - exact).abs()
                    );
                }
            }
        }
    }

    /// The two evaluation forms of `T_m` agree where they meet: the integer
    /// series against the Bessel recurrence at the crossover, and the
    /// half-integer closed form against the recurrence at a moderate `z`
    /// where the recurrence has no cancellation to speak of.
    #[test]
    fn term_evaluator_forms_agree_where_they_meet() {
        for b in [1.0_f64, 2.0, 3.0] {
            let evaluator = TermEvaluator::new(b);
            let z = SERIES_CROSSOVER_Z;
            let series = evaluator.evaluate(z);
            let ladder = bessel_k_ladder(b, z);
            let recurrence = evaluator.recurrence.evaluate(z, pow_b(z, b), &ladder);
            for m in 0..CHANNELS {
                let scale = recurrence[m].abs().max(1e-300);
                assert!(
                    ((series[m] - recurrence[m]) / scale).abs() <= 1e-11,
                    "b={b} m={m}: series {:.16e} vs recurrence {:.16e}",
                    series[m],
                    recurrence[m]
                );
            }
        }
        for b in [0.5_f64, 1.5, 2.5] {
            let evaluator = TermEvaluator::new(b);
            let z = 6.0;
            let closed = evaluator.evaluate(z);
            let ladder = bessel_k_ladder(b, z);
            let recurrence = evaluator.recurrence.evaluate(z, pow_b(z, b), &ladder);
            for m in 0..CHANNELS {
                let scale = recurrence[m].abs().max(1e-300);
                assert!(
                    ((closed[m] - recurrence[m]) / scale).abs() <= 1e-12,
                    "b={b} m={m}: closed form {:.16e} vs recurrence {:.16e}",
                    closed[m],
                    recurrence[m]
                );
            }
        }
    }

    /// Build size and time per shape, printed so a slow or refused build is
    /// read from the numbers rather than inferred from a timeout.
    #[test]
    fn profile_builds_report_their_panel_counts_and_build_time() {
        for &(d, p, s) in &SHAPES {
            let start = std::time::Instant::now();
            let built = DuchonRadialProfile::build(p, s, d);
            let elapsed = start.elapsed().as_secs_f64();
            match built {
                Ok(profile) => {
                    let (low, main) = profile.panel_counts();
                    eprintln!("[profile-build] (d={d}, p={p}, s={s}): low={low} main={main} in {elapsed:.3}s");
                }
                Err(error) => {
                    eprintln!("[profile-build] (d={d}, p={p}, s={s}): REFUSED after {elapsed:.3}s: {error}");
                    panic!("(d={d}, p={p}, s={s}) must build: {error}");
                }
            }
        }
    }

    #[test]
    fn shapes_outside_the_single_integral_regime_are_refused() {
        assert!(
            DuchonRadialProfile::build(3, 1, 6).is_err(),
            "2p = d is the partial-fraction regime"
        );
        assert!(
            DuchonRadialProfile::build(1, 0, 6).is_err(),
            "s = 0 is pure polyharmonic"
        );
        let singular = duchon_radial_profile(1, 1, 6).expect("2(p + s) ≤ d builds: finite away from the origin");
        assert!(singular.origin_value().is_err(), "but it has no origin value");
        assert!(
            DuchonRadialProfile::build(0, 1, 16).is_err(),
            "p = 0 is a bare Matérn block, outside the single-integral reduction"
        );
    }
}
