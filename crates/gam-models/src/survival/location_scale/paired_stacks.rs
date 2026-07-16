//! Per-link stable *paired* derivative stacks for the survival location-scale
//! family (the scalar foundation of #2342).
//!
//! # What this solves
//!
//! A left-truncated survival row contributes, per derivative order `k`, the
//! *sum* of an entry-survival term and an exit-density (event) or exit-survival
//! (censored) term.  Writing `A(u) := −log S(u)` (the negative log-survival of
//! the residual distribution at index `u`) and `B(u) := log φ(u)` (the log-pdf),
//! the consumer needs, for entry index `u0` and exit index `u1 = u0 + δu`:
//!
//! * event pair sums   `s_kᵉᵛ  := A⁽ᵏ⁾(u0) + B⁽ᵏ⁾(u1)`,   `k = 1..4`
//! * censored pair sums `s_kᶜᵉⁿˢ := A⁽ᵏ⁾(u0) − A⁽ᵏ⁾(u1)`, `k = 1..4`
//!
//! In the far tail the two operands of each sum are each astronomically large
//! (`A′(u0) ≈ u0 ≈ 1e150`, `B′(u1) ≈ −u1`) while their *sum* is the moderate
//! physical answer (`ρ(u0) − δu`, the Mills residual minus the index gap).
//! Adding the rounded operands is catastrophic — the same disease #2335 fixed
//! for the value channel (`stable_exit_entry_log_pairs`, commit `ad3ab650c`),
//! now on the derivative channels.  This module returns the regrouped, stable
//! `s_kᵉᵛ` / `s_kᶜᵉⁿˢ` directly so every retained term is either moderate
//! (the `s_k`) or honestly-huge with no cancellation.
//!
//! # Conventions (mirror the existing per-`u` stacks)
//!
//! `SurvivalExactRowKernel::exact_survival_neglog_derivatives_fourth_rescaled`
//! returns `(log_s, r, dr, ddr, dddr)` with `r = A′`, `dr = A″`, `ddr = A‴`,
//! `dddr = A⁗`; `exact_log_pdf_derivatives_rescaled` returns
//! `(logφ, B′, B″, B‴, B⁗)`.  The moderate-regime branch here is literally the
//! plain sum / difference of those two functions, so it is bitwise-identical to
//! today off the far tail (the "no behaviour change" contract).
//!
//! The results are the **unrescaled** stacks (`deriv_log_scale = 0` semantics).
//! For CLogLog in particular the caller must apply the `exp(−L)` derivative
//! rescale exactly as it does for the per-`u` stacks (see
//! `exact_survival_neglog_derivatives_fourth_rescaled`'s CLogLog arm).

use super::*;

/// Entry index above which the probit far-tail series takes over from the naive
/// stack sum.  Below it the naive branch is accurate (and bitwise-identical to
/// today); the series is only *needed* once `A″(u0) = 1 + ρ′(u0)` starts losing
/// its `−1/u0²` correction to the `+1` (which rounds `dr` to exactly `1.0` near
/// `u ~ 1e150`).  The threshold keys on `u0` (not `max(|u0|,|u1|)`): the
/// monotone contract `u1 ≥ u0` means a large `u0` forces a large `u1`, and a
/// moderate `u0` never cancels against `B′(u1) = −u1` however large `u1` is
/// (an `O(u0)` term cannot cancel an `O(u1)` term unless `u1 ≈ u0`).
const PROBIT_FAR_TAIL_ENTRY: f64 = 1.0e3;

/// Entry index at or below which the Gaussian hazard `r(u0)` underflows to a
/// subnormal / zero.  There `log S(u0) ≈ 0`, so the naive stack differences are
/// already cancellation-free and the consumer's plain-difference fallback is
/// exact — we return `None` and let it take that path.
const PROBIT_LEFT_TAIL_CUTOFF: f64 = -38.0;

/// The regrouped event / censored pair sums for one row.
///
/// `event[k]` is `s_{k+1}ᵉᵛ` and `censored[k]` is `s_{k+1}ᶜᵉⁿˢ` (index 0 → the
/// first derivative order).
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct PairedNeglogStacks {
    pub(crate) event: [f64; 4],
    pub(crate) censored: [f64; 4],
}

/// Stable per-link paired derivative stacks for a left-truncated survival row.
///
/// `u0` is the entry index, `u1` the exit index, and `delta_u = u1 − u0` the
/// Sterbenz-exact channel difference supplied separately by the caller (exactly
/// `0.0` for a shared entry/exit channel — the truncation-instant event case).
/// Returns `None` for links without a closed-form pairing (the consumer falls
/// back to plain stack differences) and for the probit left tail.
pub fn paired_neglog_stacks(
    link: &InverseLink,
    u0: f64,
    u1: f64,
    delta_u: f64,
) -> Option<PairedNeglogStacks> {
    match link {
        InverseLink::Standard(StandardLink::Probit) => probit_paired(link, u0, u1, delta_u),
        InverseLink::Standard(StandardLink::Logit) => logit_paired(link, u0, u1),
        InverseLink::Standard(StandardLink::CLogLog) => Some(cloglog_paired(u0, delta_u)),
        InverseLink::Standard(StandardLink::Identity) => identity_paired(link, u0, u1),
        _ => None,
    }
}

/// `[A′, A″, A‴, A⁗](u)` from the existing survival stack (bitwise-identical to
/// the per-`u` evaluator the consumers already use).
fn surv_derivs(link: &InverseLink, u: f64) -> Option<[f64; 4]> {
    let (_, r, dr, ddr, dddr) =
        SurvivalLocationScaleFamily::exact_survival_neglog_derivatives_fourth_rescaled(link, u, 0.0)
            .ok()?;
    Some([r, dr, ddr, dddr])
}

/// `[B′, B″, B‴, B⁗](u)` from the existing log-pdf stack.
fn pdf_derivs(link: &InverseLink, u: f64) -> Option<[f64; 4]> {
    let (_, d1, d2, d3, d4) =
        SurvivalLocationScaleFamily::exact_log_pdf_derivatives_rescaled(link, u, 0.0).ok()?;
    Some([d1, d2, d3, d4])
}

/// Stable `σ(−u) = 1 / (1 + e^{u})`, formed without the `1 − σ(u)` subtraction
/// that annihilates the tail for large positive `u`.
fn sigmoid_of_neg(u: f64) -> f64 {
    if u >= 0.0 {
        // e^{−u} ≤ 1: no overflow, and the numerator carries the tail directly.
        let z = (-u).exp();
        z / (1.0 + z)
    } else {
        // e^{u} < 1: the reciprocal form is exact and overflow-free.
        1.0 / (1.0 + u.exp())
    }
}

/// The Gaussian Mills residual `ρ(u) = r(u) − u` (hazard minus index) and its
/// first three derivatives, from the asymptotic series valid for large `u`.
///
/// # Derivation
///
/// With `A(u) = −log Φ(−u)` and the upper-tail expansion
/// `Φ(−u) = φ(u)/u · C(u)`, `C(u) = 1 − u⁻² + 3u⁻⁴ − 15u⁻⁶ + 105u⁻⁸ − …`
/// (coefficients `(−1)ⁿ (2n−1)!!`), we have
/// `A(u) = u²/2 + log√(2π) + log u − log C(u)`, hence
/// `A′(u) = r(u) = u + 1/u − C′/C` and `ρ(u) = 1/u − C′/C`.  Expanding
/// `C′/C = 2u⁻³ − 10u⁻⁵ + 74u⁻⁷ − …` gives
///
/// ```text
/// ρ(u)   =  u⁻¹ − 2u⁻³ + 10u⁻⁵ − 74u⁻⁷ + …
/// ρ′(u)  = −u⁻² + 6u⁻⁴ − 50u⁻⁶ + 518u⁻⁸ − …        = A″(u) − 1
/// ρ″(u)  =  2u⁻³ − 24u⁻⁵ + 300u⁻⁷ − 4144u⁻⁹ + …     = A‴(u)
/// ρ‴(u)  = −6u⁻⁴ + 120u⁻⁶ − 2100u⁻⁸ + 37296u⁻¹⁰ − … = A⁗(u)
/// ```
///
/// The `s_4ᵉᵛ` leading/next coefficients (`−6u⁻⁴ + 120u⁻⁶`) are exactly the
/// termwise derivatives of `ρ″` (`d/du[2u⁻³] = −6u⁻⁴`,
/// `d/du[−24u⁻⁵] = 120u⁻⁶`), matching `d⁴(log u)/du⁴ = −6/u⁴` at leading order.
/// Each series is truncated after the `u⁻⁷…u⁻¹⁰` term, whose contribution at the
/// `u ≥ 1e3` crossover is `≲ 1e-27` — far below rounding.
fn mills_residual_derivs(u: f64) -> (f64, f64, f64, f64) {
    let inv = 1.0 / u;
    let p = inv * inv; // 1/u²
    let rho = inv * (1.0 + p * (-2.0 + p * (10.0 + p * (-74.0))));
    let d_rho = p * (-1.0 + p * (6.0 + p * (-50.0 + p * 518.0)));
    let dd_rho = inv * p * (2.0 + p * (-24.0 + p * (300.0 + p * (-4144.0))));
    let ddd_rho = p * p * (-6.0 + p * (120.0 + p * (-2100.0 + p * 37296.0)));
    (rho, d_rho, dd_rho, ddd_rho)
}

fn probit_paired(
    link: &InverseLink,
    u0: f64,
    u1: f64,
    delta_u: f64,
) -> Option<PairedNeglogStacks> {
    if u0 <= PROBIT_LEFT_TAIL_CUTOFF {
        // Left tail: r(u0) underflows, log S(u0) ≈ 0 — the naive difference is
        // already cancellation-free, so defer to the consumer's fallback.
        return None;
    }
    if u0 <= PROBIT_FAR_TAIL_ENTRY {
        // Naive branch: the plain sum / difference of the existing stacks. This
        // is bitwise-identical to the pre-#2342 op order for every moderate row.
        let a0 = surv_derivs(link, u0)?;
        let a1 = surv_derivs(link, u1)?;
        let b1 = pdf_derivs(link, u1)?;
        return Some(PairedNeglogStacks {
            event: [
                a0[0] + b1[0],
                a0[1] + b1[1],
                a0[2] + b1[2],
                a0[3] + b1[3],
            ],
            censored: [
                a0[0] - a1[0],
                a0[1] - a1[1],
                a0[2] - a1[2],
                a0[3] - a1[3],
            ],
        });
    }

    // Far tail (u0 > 1e3 ⇒ u1 ≥ u0 > 1e3). Regroup analytically:
    //   s_1ᵉᵛ  = A′(u0) + B′(u1) = (u0 + ρ(u0)) + (−u1) = ρ(u0) − δu
    //   s_2ᵉᵛ  = A″(u0) + B″(u1) = (1 + ρ′(u0)) + (−1) = ρ′(u0)
    //   s_3ᵉᵛ  = A‴(u0) + 0      = ρ″(u0)
    //   s_4ᵉᵛ  = A⁗(u0) + 0      = ρ‴(u0)
    // and for the censored side, since A⁽ᵏ⁾(u) = (u, 1, 0, 0)ₖ + ρ⁽ᵏ⁻¹…⁾, the
    // O(1) constants (`u`, `1`) cancel in the difference before any rounding:
    //   s_1ᶜᵉⁿˢ = (u0 − u1) + (ρ(u0) − ρ(u1)) = −δu + (ρ(u0) − ρ(u1))
    //   s_kᶜᵉⁿˢ = ρ⁽ᵏ⁻¹⁾(u0) − ρ⁽ᵏ⁻¹⁾(u1)   for k = 2..4.
    let (rho0, d_rho0, dd_rho0, ddd_rho0) = mills_residual_derivs(u0);
    let (rho1, d_rho1, dd_rho1, ddd_rho1) = mills_residual_derivs(u1);
    Some(PairedNeglogStacks {
        event: [rho0 - delta_u, d_rho0, dd_rho0, ddd_rho0],
        censored: [
            (-delta_u) + (rho0 - rho1),
            d_rho0 - d_rho1,
            dd_rho0 - dd_rho1,
            ddd_rho0 - ddd_rho1,
        ],
    })
}

fn logit_paired(link: &InverseLink, u0: f64, u1: f64) -> Option<PairedNeglogStacks> {
    // With μ(u) = σ(u), σ(−u) = 1 − μ(u), and A⁽ᵏ⁾ / B⁽ᵏ⁾ the bounded logistic
    // stacks (`A″ = w`, `A‴ = w(1−2μ)`, `A⁗ = w(1−6w)`, `B″ = −2w`, …), no O(1)
    // constant cancels for k ≥ 2 — both operands simply decay like `w`, so the
    // naive stack sum / difference is globally accurate.  Only s_1 carries the
    // cancellation, cured by a GLOBAL exact regrouping:
    //   s_1ᵉᵛ  = μ(u0) + (1 − 2μ(u1)) = 2σ(−u1) − σ(−u0)
    //   s_1ᶜᵉⁿˢ = μ(u0) − μ(u1)        = σ(−u1) − σ(−u0)
    // (both verified by substituting μ = 1 − σ(−·)).  For a truncation-instant
    // event with u0 = u1 large, the naive `μ(u0) + 1 − 2μ(u1)` rounds to 0 once
    // μ saturates to 1.0, whereas `σ(−u0)` retains the true tail.
    let a0 = surv_derivs(link, u0)?;
    let a1 = surv_derivs(link, u1)?;
    let b1 = pdf_derivs(link, u1)?;
    let s1_event = 2.0 * sigmoid_of_neg(u1) - sigmoid_of_neg(u0);
    let s1_censored = sigmoid_of_neg(u1) - sigmoid_of_neg(u0);
    Some(PairedNeglogStacks {
        event: [s1_event, a0[1] + b1[1], a0[2] + b1[2], a0[3] + b1[3]],
        censored: [s1_censored, a0[1] - a1[1], a0[2] - a1[2], a0[3] - a1[3]],
    })
}

fn cloglog_paired(u0: f64, delta_u: f64) -> PairedNeglogStacks {
    // CLogLog: A⁽ᵏ⁾(u) = e^u for all k ≥ 1; B′(u) = 1 − e^u, B⁽ᵏ≥²⁾ = −e^u.
    //   s_1ᵉᵛ  = e^{u0} + (1 − e^{u1}) = 1 − e^{u0}·expm1(δu)
    //   s_kᵉᵛ  = e^{u0} − e^{u1}       = −e^{u0}·expm1(δu)   (k ≥ 2)
    //   s_kᶜᵉⁿˢ = e^{u0} − e^{u1}       = −e^{u0}·expm1(δu)   (all k)
    // The `expm1` form keeps the exit/entry gap exact and, guarding δu == 0,
    // avoids `inf·0 = NaN` when e^{u0} overflows (a shared-channel event row).
    // These are UNRESCALED (deriv_log_scale = 0); the caller applies the
    // CLogLog `exp(−L)` derivative rescale exactly as for the per-`u` stacks.
    let growth = if delta_u == 0.0 {
        0.0
    } else {
        u0.exp() * delta_u.exp_m1()
    };
    PairedNeglogStacks {
        event: [1.0 - growth, -growth, -growth, -growth],
        censored: [-growth, -growth, -growth, -growth],
    }
}

fn identity_paired(link: &InverseLink, u0: f64, u1: f64) -> Option<PairedNeglogStacks> {
    // Identity: A⁽ᵏ⁾(u) = (k−1)!/(1−u)ᵏ, B ≡ 0.  The domain (u < 1) keeps every
    // stack moderate, so plain sums / differences of the existing stacks are
    // exact.  (`surv_derivs` returns `None` — deferring to the fallback — if the
    // row leaves the identity survival domain.)
    let a0 = surv_derivs(link, u0)?;
    let a1 = surv_derivs(link, u1)?;
    let b1 = pdf_derivs(link, u1)?;
    Some(PairedNeglogStacks {
        event: [
            a0[0] + b1[0],
            a0[1] + b1[1],
            a0[2] + b1[2],
            a0[3] + b1[3],
        ],
        censored: [
            a0[0] - a1[0],
            a0[1] - a1[1],
            a0[2] - a1[2],
            a0[3] - a1[3],
        ],
    })
}

/// Whether the naive coefficient-space contraction
/// `d1_q1·dq_exit + d1_q0·dq_entry` loses precision to catastrophic
/// cancellation for a row with entry index `u0` — i.e. the entry hazard
/// `A′(u0)` is astronomically large and (near-)opposite to the exit term.
///
/// Only links with an unbounded `A′` cancel: probit (`A′ ≈ u0`) and cloglog
/// (`A′ = e^{u0}`). Logistic (`A′ = μ ∈ [0,1]`) and identity (bounded on its
/// domain) never lose an O(1) result to two ~1e300 operands, so their naive
/// contraction stays exact and this returns `false` — the consumer keeps
/// today's op order, preserving the moderate-regime bitwise contract. Cloglog
/// is deliberately **not** enabled here yet (it has no failing far-tail test
/// and cannot be validated this increment); enabling it is a trivial follow-on.
pub(crate) fn paired_contraction_needs_regroup(link: &InverseLink, u0: f64) -> bool {
    matches!(link, InverseLink::Standard(StandardLink::Probit))
        && u0.is_finite()
        && u0 > PROBIT_FAR_TAIL_ENTRY
}

/// The stable, weighted, event-mixed combined index-derivative sums
/// `S_k = w·event_mix(d, s_kᵉᵛ, s_kᶜᵉⁿˢ)` for `k = 1..3` — exactly
/// `d_k_q0 + d_k_q1` but computed cancellation-free via [`paired_neglog_stacks`]
/// instead of adding the two ~1e300 index-derivative channels. `d` is the row
/// event indicator and `w` the combined row weight already folded into the
/// `d_k_q0`/`d_k_q1` arrays (family weight × HT mask). Returns `None` for links
/// without a closed-form pairing (the consumer keeps the naive contraction).
pub(crate) fn weighted_paired_index_sums(
    link: &InverseLink,
    u0: f64,
    u1: f64,
    delta_u: f64,
    d: f64,
    w: f64,
) -> Option<[f64; 3]> {
    let p = paired_neglog_stacks(link, u0, u1, delta_u)?;
    Some([
        w * event_mix(d, p.event[0], p.censored[0]),
        w * event_mix(d, p.event[1], p.censored[1]),
        w * event_mix(d, p.event[2], p.censored[2]),
    ])
}

#[cfg(test)]
mod paired_stack_tests {
    use super::*;

    fn probit() -> InverseLink {
        InverseLink::Standard(StandardLink::Probit)
    }
    fn logit() -> InverseLink {
        InverseLink::Standard(StandardLink::Logit)
    }
    fn cloglog() -> InverseLink {
        InverseLink::Standard(StandardLink::CLogLog)
    }
    fn identity() -> InverseLink {
        InverseLink::Standard(StandardLink::Identity)
    }

    fn assert_close(a: f64, b: f64, rel: f64, abs: f64, ctx: &str) {
        let diff = (a - b).abs();
        let scale = a.abs().max(b.abs());
        assert!(
            diff <= rel * scale + abs,
            "{ctx}: a={a:e} b={b:e} diff={diff:e} rel_budget={:e}",
            rel * scale + abs
        );
    }

    /// Test 1 — moderate-regime agreement with the naive stack sum / difference.
    ///
    /// Probit (naive branch) and Identity are literally that sum, so they must
    /// be *bitwise* identical.  Logistic k ≥ 2 is also the naive stack op, hence
    /// bitwise; its s_1 regrouping and CLogLog's closed forms are algebraically
    /// equal to the naive form and agree to a few ULP (asserted relatively —
    /// the naive logistic-s_1 `μ + 1 − 2μ` form loses ~1e-13 relative to its
    /// own `+1` cancellation even at moderate `u`, so a strict 4-ULP bound is
    /// not attainable through that reference).
    #[test]
    fn paired_matches_naive_stacks_in_moderate_regime() {
        let probit_pts = [
            (-3.0, -2.5),
            (-1.0, 0.0),
            (0.0, 0.0),
            (0.5, 2.0),
            (2.0, 2.0),
            (-5.0, 3.0),
        ];
        let logit_pts = [
            (-4.0, -3.0),
            (-1.0, 1.0),
            (0.0, 0.0),
            (2.0, 2.0),
            (3.0, 5.0),
            (6.0, 6.0),
        ];
        let cloglog_pts = [
            (-2.0, -1.0),
            (-1.0, 0.0),
            (0.0, 0.0),
            (0.5, 1.5),
            (1.0, 1.0),
            (-3.0, 2.0),
        ];
        let identity_pts = [
            (-3.0, -2.0),
            (-1.0, 0.0),
            (0.0, 0.0),
            (0.3, 0.6),
            (0.8, 0.9),
            (-10.0, 0.5),
        ];

        let bitwise_all = |link: &InverseLink, pts: &[(f64, f64)]| {
            for &(u0, u1) in pts {
                let du = u1 - u0;
                let p = paired_neglog_stacks(link, u0, u1, du).unwrap();
                let a0 = surv_derivs(link, u0).unwrap();
                let a1 = surv_derivs(link, u1).unwrap();
                let b1 = pdf_derivs(link, u1).unwrap();
                for k in 0..4 {
                    assert_eq!(
                        p.event[k].to_bits(),
                        (a0[k] + b1[k]).to_bits(),
                        "event[{k}] u0={u0} u1={u1}"
                    );
                    assert_eq!(
                        p.censored[k].to_bits(),
                        (a0[k] - a1[k]).to_bits(),
                        "censored[{k}] u0={u0} u1={u1}"
                    );
                }
            }
        };
        bitwise_all(&probit(), &probit_pts);
        bitwise_all(&identity(), &identity_pts);

        // Logistic: k ≥ 2 bitwise, s_1 regrouping close to the naive form.
        for &(u0, u1) in &logit_pts {
            let du = u1 - u0;
            let p = paired_neglog_stacks(&logit(), u0, u1, du).unwrap();
            let a0 = surv_derivs(&logit(), u0).unwrap();
            let a1 = surv_derivs(&logit(), u1).unwrap();
            let b1 = pdf_derivs(&logit(), u1).unwrap();
            for k in 1..4 {
                assert_eq!(p.event[k].to_bits(), (a0[k] + b1[k]).to_bits());
                assert_eq!(p.censored[k].to_bits(), (a0[k] - a1[k]).to_bits());
            }
            assert_close(p.event[0], a0[0] + b1[0], 1e-12, 1e-300, "logit s1 event");
            assert_close(p.censored[0], a0[0] - a1[0], 1e-12, 1e-300, "logit s1 censored");
        }

        // CLogLog: closed forms vs the naive stack op (algebraically equal).
        for &(u0, u1) in &cloglog_pts {
            let du = u1 - u0;
            let p = paired_neglog_stacks(&cloglog(), u0, u1, du).unwrap();
            let a0 = surv_derivs(&cloglog(), u0).unwrap();
            let a1 = surv_derivs(&cloglog(), u1).unwrap();
            let b1 = pdf_derivs(&cloglog(), u1).unwrap();
            for k in 0..4 {
                assert_close(p.event[k], a0[k] + b1[k], 1e-12, 1e-300, "cloglog event");
                assert_close(p.censored[k], a0[k] - a1[k], 1e-12, 1e-300, "cloglog censored");
            }
        }
    }

    /// Test 2 — probit far-tail truth: the paired sums equal the Mills-series
    /// asymptotics, not the derivative of the old cancelling surface.
    #[test]
    fn probit_far_tail_matches_mills_series() {
        let entries = [1.0e4, 1.0e8, 1.0e50, 1.0e150, 3.66e150];
        let gaps = [0.0, 1.0e-3, 0.48];
        for &u0 in &entries {
            for &du in &gaps {
                let u1 = u0 + du;
                let p = paired_neglog_stacks(&probit(), u0, u1, du).unwrap();

                // s_2ᵉᵛ ≈ −1/u0² (next term +6/u0⁴ ⇒ relative size 6/u0²).
                let ref2 = -1.0 / (u0 * u0);
                assert_close(p.event[1], ref2, 10.0 / (u0 * u0), 0.0, "s2 event");

                // s_3ᵉᵛ ≈ 2/u0³ and s_4ᵉᵛ ≈ −6/u0⁴, where representable.
                // (next-term relative sizes ≈ 12/u0² and 20/u0² respectively).
                let ref3 = 2.0 / (u0 * u0 * u0);
                if ref3 != 0.0 && ref3.is_finite() {
                    assert_close(p.event[2], ref3, 16.0 / (u0 * u0), 0.0, "s3 event");
                }
                let ref4 = -6.0 / (u0 * u0 * u0 * u0);
                if ref4 != 0.0 && ref4.is_finite() {
                    assert_close(p.event[3], ref4, 28.0 / (u0 * u0), 0.0, "s4 event");
                }

                // s_1ᵉᵛ = ρ(u0) − δu.  For δu = 0 the leading term 1/u0 is
                // recoverable; for δu ≠ 0 the moderate `−δu` dominates and the
                // residual `s_1 + δu = ρ(u0)` is bounded by 2/u0.
                if du == 0.0 {
                    assert_close(p.event[0], 1.0 / u0, 3.0 / (u0 * u0), 0.0, "s1 event du=0");
                } else {
                    assert!(
                        (p.event[0] + du).abs() <= 2.0 / u0,
                        "s1 event residual too large: u0={u0} du={du} val={:e}",
                        p.event[0] + du
                    );
                }

                // s_1ᶜᵉⁿˢ ≈ −δu·A″(u0) to first order in δu (A″ = 1 + ρ′).
                if du != 0.0 {
                    let (_, d_rho0, _, _) = mills_residual_derivs(u0);
                    let ref_cens = -du * (1.0 + d_rho0);
                    assert_close(p.censored[0], ref_cens, 1e-10, 0.0, "s1 censored");
                }
            }
        }
    }

    /// Test 3 — continuity across the naive/series crossover at u0 = 1e3.  The
    /// two branches must agree per entry; the value/gradient orders (0,1) are
    /// pinned tightly, the tiny 3rd/4th orders loosely (they sit near the naive
    /// stack's noise floor at this magnitude).
    #[test]
    fn probit_crossover_is_continuous() {
        let thr = PROBIT_FAR_TAIL_ENTRY;
        let off = 1.0e-11; // Δu0 ≈ 2e-8: branch discretization dominates.
        let du = 0.1;
        let lo = thr * (1.0 - off);
        let hi = thr * (1.0 + off);
        let p_lo = paired_neglog_stacks(&probit(), lo, lo + du, du).unwrap();
        let p_hi = paired_neglog_stacks(&probit(), hi, hi + du, du).unwrap();
        for k in 0..2 {
            assert_close(p_lo.event[k], p_hi.event[k], 1e-6, 0.0, "crossover event lo-order");
            assert_close(p_lo.censored[k], p_hi.censored[k], 1e-6, 0.0, "crossover censored lo-order");
        }
        for k in 2..4 {
            assert_close(p_lo.event[k], p_hi.event[k], 1e-1, 1e-300, "crossover event hi-order");
            assert_close(
                p_lo.censored[k],
                p_hi.censored[k],
                1e-1,
                1e-300,
                "crossover censored hi-order",
            );
        }
    }

    /// Test 4 — logistic s_1 regrouping is globally exact where the naive form
    /// collapses.  At a saturated truncation-instant event (u0 = u1 = 40) the
    /// naive `μ + 1 − 2μ` rounds to 0, while the regrouped `σ(−40) > 0` retains
    /// the true tail.
    #[test]
    fn logistic_s1_regrouping_beats_naive_at_saturation() {
        // Moderate agreement with the closed regrouped reference.
        for &(u0, u1) in &[(-2.0, 1.0), (0.0, 0.0), (1.5, 3.0)] {
            let p = paired_neglog_stacks(&logit(), u0, u1, u1 - u0).unwrap();
            let want = 2.0 * sigmoid_of_neg(u1) - sigmoid_of_neg(u0);
            assert_eq!(p.event[0].to_bits(), want.to_bits(), "logit s1 exact form");
        }

        let p = paired_neglog_stacks(&logit(), 40.0, 40.0, 0.0).unwrap();
        let sig = sigmoid_of_neg(40.0);
        assert!(sig > 0.0, "sigma(-40) must be positive: {sig:e}");
        assert_eq!(p.event[0].to_bits(), sig.to_bits());

        // The naive stack form annihilates this row.
        let a0 = surv_derivs(&logit(), 40.0).unwrap();
        let b1 = pdf_derivs(&logit(), 40.0).unwrap();
        let naive = a0[0] + b1[0];
        assert_eq!(naive, 0.0, "naive s1 should saturate to 0, got {naive:e}");
        assert!(p.event[0] > naive, "regrouping must recover a positive s1");
    }

    /// Test 5 — CLogLog δu = 0 guard: no `inf·0 = NaN` when e^{u0} overflows.
    #[test]
    fn cloglog_zero_gap_guard_stays_finite() {
        assert!((800.0_f64).exp().is_infinite(), "fixture must overflow e^u0");
        let p = paired_neglog_stacks(&cloglog(), 800.0, 800.0, 0.0).unwrap();
        assert_eq!(p.event[0], 1.0);
        for k in 1..4 {
            assert_eq!(p.event[k], 0.0, "event[{k}]");
        }
        for k in 0..4 {
            assert_eq!(p.censored[k], 0.0, "censored[{k}]");
        }
        assert!(
            p.event.iter().chain(p.censored.iter()).all(|v| v.is_finite()),
            "all entries finite"
        );

        // A finite non-zero gap still produces the expmm1 growth.
        let q = paired_neglog_stacks(&cloglog(), 1.0, 1.5, 0.5).unwrap();
        let growth = 1.0_f64.exp() * 0.5_f64.exp_m1();
        assert_close(q.event[0], 1.0 - growth, 1e-14, 0.0, "cloglog s1 growth");
        assert_close(q.censored[3], -growth, 1e-14, 0.0, "cloglog cens growth");
    }

    /// The regroup gate fires only in the probit far tail; bounded links and
    /// moderate probit rows keep the naive contraction.
    #[test]
    fn regroup_gate_fires_only_in_probit_far_tail() {
        assert!(paired_contraction_needs_regroup(&probit(), 3.6e150));
        assert!(paired_contraction_needs_regroup(&probit(), 1.0e4));
        assert!(!paired_contraction_needs_regroup(&probit(), 5.0));
        assert!(!paired_contraction_needs_regroup(&probit(), 1.0e3));
        assert!(!paired_contraction_needs_regroup(&logit(), 3.6e150));
        assert!(!paired_contraction_needs_regroup(&cloglog(), 3.6e150));
        assert!(!paired_contraction_needs_regroup(&identity(), 0.9));
    }

    /// `weighted_paired_index_sums[0]` reproduces `w·(A′(u0)+B′(u1))` in the
    /// moderate regime and stays the moderate `w·(ρ(u0)−δu)` in the far tail,
    /// where forming `A′(u0)+B′(u1)` directly collapses to 0.
    #[test]
    fn weighted_paired_index_sums_are_stable() {
        // Moderate event row: S1 equals the naive weighted stack sum exactly.
        let (u0, u1, w) = (0.3_f64, 0.5_f64, 1.5_f64);
        let s = weighted_paired_index_sums(&probit(), u0, u1, u1 - u0, 1.0, w).unwrap();
        let a = surv_derivs(&probit(), u0).unwrap();
        let b = pdf_derivs(&probit(), u1).unwrap();
        assert_close(s[0], w * (a[0] + b[0]), 1e-12, 0.0, "moderate S1 event");

        // Far-tail event row (u0 ≈ u1 ≈ 3.6e150, δu = 0.08): S1 = w·(ρ(u0)−δu)
        // ≈ −w·δu since ρ(u0) ~ 1/u0 ~ 1e-151. Forming A′(u0)+B′(u1) directly is
        // u0 − u1 rounded to exactly 0 — the moderate answer is lost.
        let (u0, delta_u, w) = (3.6e150_f64, 0.08_f64, 1.2_f64);
        let u1 = u0 + delta_u;
        let s = weighted_paired_index_sums(&probit(), u0, u1, delta_u, 1.0, w).unwrap();
        assert_close(s[0], w * (-delta_u), 1e-6, 0.0, "far-tail S1 event");
        let naive = w * (surv_derivs(&probit(), u0).unwrap()[0] + pdf_derivs(&probit(), u1).unwrap()[0]);
        assert!(
            naive.abs() < 1e-3 && (s[0] - w * (-delta_u)).abs() < 1e-6,
            "naive S1 collapses ({naive:e}) while paired recovers {:e}",
            s[0]
        );
    }
}
