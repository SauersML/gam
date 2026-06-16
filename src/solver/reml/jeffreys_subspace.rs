//! Shared under-identified-subspace selector for the universal Jeffreys/Firth
//! robustness machinery.
//!
//! The Jeffreys penalty `Phi = 1/2 log|I(beta)|` is only ever applied to the
//! directions that are identified by NEITHER the data nor a proper prior — the
//! "under-identified span". Penalized smooth directions already carry a proper
//! wiggliness prior (their `S_lambda` curvature), so applying Jeffreys there
//! would double-regularize and bias the smooth fit. This module produces the
//! orthonormal basis `Z_J` of that span for one parameter block.
//!
//! The under-identified span is the FULL identifiable coefficient span of the
//! (post-rank-deficiency-removal) reduced block — `Z_J = I_p` — NOT the penalty
//! null space `ker(S)`. The Jeffreys penalty is self-limiting (its `O(1)` score
//! is dominated by the data's `O(n)` Fisher information), so on a data-identified
//! direction (penalized OR not) its only effect is the `O(1/n)` Firth bias
//! correction; it bites only where the information is near-singular. Using the
//! full span — rather than scoping to `ker(S)` — lets it reach a near-separation
//! on a penalized spline direction too (the residual BMS-probit pathology). The
//! aggregate penalty is consulted only to pick up the block dimension `p`.
//!
//! Both tiers of the robustness machinery consume the SAME `Z_J`:
//!   * Tier A (single-eta GLM via `FirthDenseOperator`) scopes the Fisher
//!     information to `X * Z_J`.
//!   * Tier B (coupled multi-predictor custom-family joint Newton, e.g. BMS)
//!     restricts the joint-Hessian Jeffreys term `Phi_J = 1/2 log|Z_J^T H Z_J|`
//!     to the same span.
//!
//! Everything here is pure linear algebra on the block's penalty matrices.
//! Robustness is the unconditional default; the conditioning gate in
//! [`joint_jeffreys_term`] (self-limiting, returns the exact zero contribution on
//! a well-conditioned fit) is the only "apply where needed" mechanism.

use crate::linalg::faer_ndarray::FaerEigh;
use crate::linalg::lanczos::{SymmetricLanczosOptions, symmetric_lanczos_eigenpairs};
use faer::Side;
use ndarray::{Array1, Array2, ArrayView2};
use std::collections::HashMap;
use std::sync::Arc;

#[inline]
pub(crate) fn norm2_slice(a: &[f64]) -> f64 {
    a.iter().map(|x| x * x).sum::<f64>().sqrt()
}

/// Relative floor on a reduced-information eigenvalue, as a fraction of the
/// dominant (identified) curvature `λ_max`. Negligible on data-identified
/// directions (whose curvature is `O(n) · λ_max`-scale), positive on separating
/// directions, keeping the Jeffreys log-det finite even when the observed
/// information is indefinite at an off-mode trial point.
pub(crate) const REDUCED_INFO_RELATIVE_FLOOR: f64 = 1e-10;

/// Absolute floor for the degenerate case where every reduced eigenvalue is
/// (near) zero, so `λ_max ≈ 0` cannot scale the relative floor.
pub(crate) const REDUCED_INFO_ABSOLUTE_FLOOR: f64 = 1e-12;

/// Upper saturation scale `Λ` of the Jeffreys antiderivative: once a
/// direction is identified at the conditioning gate's own "clearly
/// identified" curvature (`CONDITIONING_GATE_ABSOLUTE_CLEAR` observation-
/// equivalents), additional information carries no additional prior reward.
/// `max(·, floor)` keeps the branch order well-defined in the (extreme-scale)
/// regime where the relative floor exceeds the gate scale; the joins remain
/// C¹ automatically there because the log window simply collapses to empty.
#[inline]
pub(crate) fn jeffreys_cap(floor: f64) -> f64 {
    CONDITIONING_GATE_ABSOLUTE_CLEAR.max(floor)
}

/// Slope `d(λ) = g'(λ)` of the Jeffreys eigenvalue antiderivative `g`, a
/// BOUNDED, MONOTONE, C¹ function of the reduced eigenvalue (gam#979), with
/// `Λ = jeffreys_cap(floor)`:
///
///   * `λ ≥ Λ`:           `g = ln Λ + 1 − Λ/λ`,                `d = Λ/λ²`
///     (TOP saturation: a direction already identified at the gate's
///     clearly-identified scale earns no further prior reward; `g → ln Λ + 1`,
///     `d → 0` as `λ → ∞`);
///   * `floor ≤ λ < Λ`:   `g = ln λ`,                          `d = 1/λ`
///     (the exact Jeffreys log-volume on the under-identified window — the
///     only region where the prior is meant to act);
///   * `0 ≤ λ < floor`:   `g = λ/floor + ln(floor) − 1`,       `d = 1/floor`
///     (the #787 linear continuation — C¹ at `+floor`, preserves the
///     1/floor separation bound);
///   * `λ < 0`:           `g = ln(floor) − 1 + λ/(floor − λ)`, `d = floor/(floor − λ)²`
///     (BOTTOM saturation). C¹ at `0` (`g(0) = ln(floor) − 1`,
///     `d(0) = 1/floor` from both sides), `g → ln(floor) − 2`, `d → 0` as
///     `λ → −∞`.
///
/// WHY BOTH ENDS SATURATE (gam#979; supersedes the gam#814 `ln|λ|` magnitude
/// branch and the unbounded top). An unbounded `g` lets the Φ-augmented inner
/// objective `−ℓ + ½βᵀSβ − Φ` be descended by harvesting `Φ` growth instead
/// of fitting data, and the exact divided-difference `H_Φ` (unlike the old
/// frozen `K²` vec-Gram phantom) lets the joint Newton actually follow that
/// reward. Both directions were measured on the constrained binomial-LS
/// wiggle fixture: the `ln|λ|` bottom rewards driving a reduced eigenvalue
/// negative; the unbounded top rewards driving OBSERVED information up —
/// for probit, observed information grows like `η²` on misclassified
/// saturated observations, so the iterate walked into likelihood saturation
/// (bare `−ℓ + pen` climbing 5.2 → 9.9 over 500 cycles with the augmented
/// Δobj negative every cycle, residual → 3e2, budget exhausted). The Jeffreys
/// term's mandate is BOUNDING under-identified directions; the bounded `g`
/// is the per-eigenvalue, smooth realisation of the same self-limitation the
/// binary conditioning gate states globally: exact `ln λ` inside the
/// under-identified window `[floor, Λ)`, flat outside it. `g` spans the
/// bounded range `(ln floor − 2, ln Λ + 1]`, so `Φ` can never out-pay a
/// data-likelihood term.
#[inline]
pub(crate) fn floored_inverse(lam: f64, floor: f64) -> f64 {
    let cap = jeffreys_cap(floor);
    if lam >= cap {
        cap / (lam * lam)
    } else if lam >= floor {
        1.0 / lam
    } else if lam >= 0.0 {
        1.0 / floor
    } else {
        let denom = floor - lam;
        floor / (denom * denom)
    }
}

/// The Jeffreys eigenvalue antiderivative `g(λ; floor)` whose derivative is
/// exactly [`floored_inverse`] — i.e. `d/dλ g = floored_inverse(λ, floor)`.
///
/// This is the SINGLE source of the per-eigenvalue value branches the
/// `Φ = ½ Σ_i g(λ_i)` accumulation in [`joint_jeffreys_term`] computes inline;
/// it is factored out so the `½ log|H_id|₊`-style **criterion atom**
/// (`JeffreysLogdetAtom`) can emit its VALUE (`½ Σ g`) and its frozen
/// directional DERIVATIVE (`½ Σ floored_inverse(λ) · Ṽ_ii`, the `tr(H_id⁺ Ḣ)`
/// trace) as projections of one spectrum — value and gradient cannot desync
/// because `g` and `g' = floored_inverse` are pinned here (the gam#787/#785
/// value↔gradient-consistency invariant, now structural). The four branches
/// mirror the documentation on [`floored_inverse`] exactly:
///
///   * `λ ≥ Λ`:           `g = ln Λ + 1 − Λ/λ`        (TOP saturation);
///   * `floor ≤ λ < Λ`:   `g = ln λ`                  (exact Jeffreys log-volume);
///   * `0 ≤ λ < floor`:   `g = λ/floor + ln(floor) − 1` (#787 linear continuation);
///   * `λ < 0`:           `g = ln(floor) − 1 + λ/(floor − λ)` (BOTTOM saturation).
#[inline]
pub(crate) fn jeffreys_antiderivative(lam: f64, floor: f64) -> f64 {
    let cap = jeffreys_cap(floor);
    if lam >= cap {
        cap.ln() + 1.0 - cap / lam
    } else if lam >= floor {
        lam.ln()
    } else if lam >= 0.0 {
        lam / floor + floor.ln() - 1.0
    } else {
        floor.ln() - 1.0 + lam / (floor - lam)
    }
}

/// `∂g/∂floor` with `λ` held fixed — the floor-motion term of the Jeffreys
/// VALUE, the exact antiderivative partner of [`floored_inverse_floor_sensitivity`]
/// (which is `∂g'/∂floor`). Pinned to the SAME four branches as
/// [`jeffreys_antiderivative`] so the floor-response gradient cannot drift from
/// the value the way the inline branches once did (gam#826):
///
///   * `λ ≥ Λ`:           `1/floor − 1/λ` when the cap is floor-bound
///     (`Λ = floor`, extreme-scale regime), else `0` (the gate-bound cap does
///     not move with the floor, so the value does not either);
///   * `floor ≤ λ < Λ`:   `0`            (`g = ln λ` is floor-free here);
///   * `0 ≤ λ < floor`:   `1/floor − λ/floor²` (the #787 linear continuation);
///   * `λ < 0`:           `1/floor − λ/(floor − λ)²` (BOTTOM saturation).
#[inline]
pub(crate) fn jeffreys_antiderivative_floor_sensitivity(lam: f64, floor: f64) -> f64 {
    let cap = jeffreys_cap(floor);
    if lam >= cap {
        if cap > CONDITIONING_GATE_ABSOLUTE_CLEAR {
            // Floor-bound cap: g = ln(floor) + 1 − floor/λ ⇒ ∂g/∂floor = 1/floor − 1/λ.
            1.0 / floor - 1.0 / lam
        } else {
            0.0
        }
    } else if lam >= floor {
        0.0
    } else if lam >= 0.0 {
        1.0 / floor - lam / (floor * floor)
    } else {
        let denom = floor - lam;
        1.0 / floor - lam / (denom * denom)
    }
}

/// `d'(λ)` with the floor held fixed: `−2Λ/λ³` on the top saturation,
/// `−1/λ²` in the log window, `0` inside the band (the linear continuation
/// has no curvature in `λ`), `2·floor/(floor − λ)³` on the bottom saturation.
#[inline]
pub(crate) fn floored_inverse_prime(lam: f64, floor: f64) -> f64 {
    let cap = jeffreys_cap(floor);
    if lam >= cap {
        -2.0 * cap / (lam * lam * lam)
    } else if lam >= floor {
        -1.0 / (lam * lam)
    } else if lam >= 0.0 {
        0.0
    } else {
        let denom = floor - lam;
        2.0 * floor / (denom * denom * denom)
    }
}

/// `d''(λ)` with the floor held fixed: `6Λ/λ⁴` on the top saturation, `2/λ³`
/// in the log window, `0` inside the band, `6·floor/(floor − λ)⁴` on the
/// bottom saturation. Needed by the drift path for the confluent-pair limit
/// of the divided-difference kernel motion (`δΨ_ii = d''(λ_i)·λ̇_i`).
#[inline]
pub(crate) fn floored_inverse_second(lam: f64, floor: f64) -> f64 {
    let cap = jeffreys_cap(floor);
    if lam >= cap {
        6.0 * cap / (lam * lam * lam * lam)
    } else if lam >= floor {
        2.0 / (lam * lam * lam)
    } else if lam >= 0.0 {
        0.0
    } else {
        let denom = floor - lam;
        6.0 * floor / (denom * denom * denom * denom)
    }
}

/// `∂d/∂floor` with `λ` held fixed: `1/λ²` on the top saturation when the cap
/// is floor-bound (`Λ = floor`, the extreme-scale regime; `0` otherwise — the
/// gate-bound cap does not move with the floor), `0` in the log window,
/// `−1/floor²` inside the band, `−(floor + λ)/(floor − λ)³` on the bottom
/// saturation. Feeds the floor-motion term of the kernel drift when the floor
/// is in its active relative regime (`floor = REL·λ_max(β)`).
#[inline]
pub(crate) fn floored_inverse_floor_sensitivity(lam: f64, floor: f64) -> f64 {
    let cap = jeffreys_cap(floor);
    if lam >= cap {
        if cap > CONDITIONING_GATE_ABSOLUTE_CLEAR {
            // Floor-bound cap: d = floor/λ².
            1.0 / (lam * lam)
        } else {
            0.0
        }
    } else if lam >= floor {
        0.0
    } else if lam >= 0.0 {
        -1.0 / (floor * floor)
    } else {
        let denom = floor - lam;
        -(floor + lam) / (denom * denom * denom)
    }
}

/// `∂d'/∂floor` with `λ` held fixed: nonzero only where `d'` carries the
/// floor — `−2/λ³` on a floor-bound top saturation (`Λ = floor`),
/// `−2(2·floor + λ)/(floor − λ)⁴` on the bottom saturation, `0` elsewhere.
/// Feeds the confluent-pair floor-motion of the kernel drift.
#[inline]
pub(crate) fn floored_inverse_prime_floor_sensitivity(lam: f64, floor: f64) -> f64 {
    let cap = jeffreys_cap(floor);
    if lam >= cap {
        if cap > CONDITIONING_GATE_ABSOLUTE_CLEAR {
            -2.0 / (lam * lam * lam)
        } else {
            0.0
        }
    } else if lam >= 0.0 {
        0.0
    } else {
        let denom = floor - lam;
        -2.0 * (2.0 * floor + lam) / (denom * denom * denom * denom)
    }
}

/// Daleckii–Krein divided-difference matrix of the floored signed inverse on
/// the reduced spectrum: `Ψ_ij = (d(λ_i) − d(λ_j)) / (λ_i − λ_j)` for
/// well-separated pairs, with the confluent limit `Ψ_ii = d'(λ_i)` on the
/// diagonal and on (near-)tied pairs. This is the exact eigenbasis kernel of
/// the Fréchet derivative of the floored pseudo-inverse `K = V diag(d_i) Vᵀ`:
/// `δK = V (Ψ ∘ (Vᵀ δH_id V)) Vᵀ` — and therefore also the exact curvature
/// kernel of `Φ = ½ Σ g(λ_i)`: dropping only the second-directional-Hessian
/// term `½ tr(K D_ab)` (which carries the vanishing curvature factor of a
/// separating direction, so it is O(1) exactly where the term arms),
///   `∂²Φ[a,b] = ½ Σ_ij Ψ_ij (Ṽ_a)_ij (Ṽ_b)_ij`,  `Ṽ_k = Vᵀ D_k V`.
/// With the saturating negative branch `d` is continuous everywhere, so every
/// divided difference is bounded by `max|d'| ≤ 2/floor`.
pub(crate) fn floored_inverse_divided_differences(evals: &Array1<f64>, floor: f64) -> Array2<f64> {
    let m = evals.len();
    let mut psi = Array2::<f64>::zeros((m, m));
    for i in 0..m {
        for j in 0..m {
            let denom = evals[i] - evals[j];
            psi[[i, j]] = if denom.abs() <= REDUCED_INFO_ABSOLUTE_FLOOR {
                floored_inverse_prime(evals[i], floor)
            } else {
                (floored_inverse(evals[i], floor) - floored_inverse(evals[j], floor)) / denom
            };
        }
    }
    psi
}

/// Conditioning gate. When the reduced information `H_id = Z_J^T H Z_J` is
/// well-conditioned — every direction's curvature is within this relative
/// factor of the dominant `λ_max` — the data identifies the WHOLE span at
/// `O(n)` strength and the self-limiting `O(1)` Jeffreys term is negligible
/// there (its only effect would be the `O(1/n)` Firth bias correction, which is
/// not what this machinery exists to supply). We therefore SKIP the term
/// entirely and return the zero contribution, so a clean/easy fit pays no cost
/// and stays byte-identical to the un-penalized inner Newton. The gate fires
/// only on the OTHER side: an ill-conditioned / near-separating reduced
/// information (`λ_min/λ_max` below this threshold), where the floored log-det
/// curvature below is exactly the `O(1)`-bounding term Firth supplies.
///
/// The threshold sits far from machine precision: at `1e-8` the worst-
/// conditioned direction is still 8 orders of magnitude from the absolute floor
/// (`REDUCED_INFO_RELATIVE_FLOOR = 1e-10`), i.e. comfortably identified rather
/// than separating, so nothing the term would actually bound is gated out.
///
/// NOTE: a relative ratio is SCALE-FREE in `n` — it cannot, on its own, tell a
/// near-separating direction (absolute curvature `O(1)`) from a well-identified
/// one (absolute curvature `O(n)`). At small `n` an absolutely-near-separating
/// direction can still clear this relative gate (if `λ_max` is also small), so it
/// is paired with the ABSOLUTE gate below; the term fires when EITHER gate
/// reports under-identification (see [`conditioning_gate_weight`]).
pub(crate) const CONDITIONING_GATE_RELATIVE: f64 = 1e-8;

/// Absolute-curvature conditioning gate (the `n`-aware half of the gate).
///
/// Separation is an ABSOLUTE statement about curvature, not a relative one: a
/// direction is near-separating when the data place `O(1)` Fisher information on
/// it — a handful of near-boundary observations — REGARDLESS of the sample size
/// `n`. A well-identified direction instead accumulates `O(n)` information (each
/// of `n` observations contributes `O(1)` curvature). The reduced information
/// `H_id = Z_Jᵀ H Z_J` here IS that observed/expected Fisher information (an
/// un-normalised sum over observations, NOT a per-observation average), so its
/// smallest eigenvalue `λ_min` is `O(n)` on an identified direction and `O(1)`
/// on a separating one — the two regimes are separated by an absolute `O(1)`
/// scale that does not move with `n`.
///
/// We therefore ALSO fire the Jeffreys term whenever `λ_min` falls below this
/// absolute scale, independent of the relative ratio. This catches the
/// small-`n` admixture-cline / near-separation regime the relative gate misses,
/// where `λ_max` is itself modest so a near-zero `λ_min` can still satisfy
/// `λ_min/λ_max ≥ 1e-8`.
///
/// THRESHOLD CHOICE. One observation contributes at most `O(1)` curvature to a
/// unit-scale direction (e.g. a binomial Fisher weight `p(1−p) ≤ 1/4`, a
/// Gaussian unit weight `1`), so a direction carrying less than a single
/// observation's worth of information is, by construction, not identified by the
/// data and is the regime Firth exists to stabilise. We set the gate at `1.0`:
/// `λ_min < 1` ⇒ the direction holds under one observation-equivalent of
/// curvature ⇒ treat as near-separating and fire the term. This is conservative
/// (it never fires on a genuinely well-conditioned large-`n` fit, whose
/// `λ_min = O(n) ≫ 1`, so the byte-identical clean-fit guarantee is preserved)
/// while catching absolute near-separation at any `n`. The design is assumed to
/// be on a standardized/O(1)-column scale, which the upstream reduction already
/// enforces; the floor below (`REDUCED_INFO_ABSOLUTE_FLOOR = 1e-12`) keeps the
/// log-det finite once the term fires.
pub(crate) const CONDITIONING_GATE_ABSOLUTE: f64 = 1.0;

/// Upper knot of the SMOOTH absolute conditioning ramp. Below
/// `CONDITIONING_GATE_ABSOLUTE` (one observation-equivalent of curvature) the
/// direction is near-separating and the Jeffreys weight is `1` (full term);
/// above this value it is comfortably identified and the weight is `0` (skip).
/// Between the two knots the weight is a C¹ cubic-smoothstep blend, so the outer
/// LAML objective `Φ(ρ)` is CONTINUOUS as `β̂(ρ)` carries the spectrum across the
/// boundary. A BINARY gate makes `Φ(ρ)` jump there, which the gradient-based
/// outer smoother (BFGS) cannot optimize across — the root cause of the #787
/// "outer smoothing did not converge" regression. `16` ≈ a handful of
/// observation-equivalents: comfortably past "identified by the data".
pub(crate) const CONDITIONING_GATE_ABSOLUTE_CLEAR: f64 = 16.0;

/// Upper knot of the SMOOTH relative conditioning ramp (ramped in `log10` space
/// since conditioning ratios span orders of magnitude). At
/// `λ_min/λ_max ≥ CONDITIONING_GATE_RELATIVE_CLEAR` the relative sub-weight is
/// `0`; at `≤ CONDITIONING_GATE_RELATIVE` it is `1`; smooth in between.
pub(crate) const CONDITIONING_GATE_RELATIVE_CLEAR: f64 = 1e-6;

/// Shared conditioning-gate predicate for the Jeffreys term, evaluated from the
/// reduced-information spectrum (`λ_min`, `λ_max`). Returns `true` when the term
/// should be SKIPPED (zero contribution) because the reduced information is
/// well-conditioned — both relatively (`λ_min/λ_max ≥ CONDITIONING_GATE_RELATIVE`)
/// AND absolutely (`λ_min ≥ CONDITIONING_GATE_ABSOLUTE`). If EITHER test reports
/// under-identification the gate does NOT skip and the floored log-det term
/// fires. Centralised so every call site (objective value, gradient/curvature,
/// and the `H_Φ` directional derivative) uses byte-identical gating — any
/// divergence would reintroduce the value/derivative mismatch the term removes.
///
/// Returns a SMOOTH (C¹) weight in `[0, 1]`: `1` when the reduced information is
/// under-identified (the term is fully active and supplies the `O(1)`-bounding
/// curvature), `0` when comfortably well-conditioned (the term is skipped,
/// preserving the clean-fit guarantee), and a cubic-smoothstep blend across each
/// transition band so `Φ(ρ)`, `∇Φ` and `H_Φ` are continuous in the smoothing
/// parameters. The term fires when EITHER criterion reports under-identification,
/// so the weight is the MAX of the absolute and relative sub-weights.
#[inline]
pub(crate) fn conditioning_gate_weight(lambda_min: f64, lambda_max: f64) -> f64 {
    if lambda_max <= 0.0 {
        // Degenerate / non-positive spectrum: not well-conditioned, fully active.
        return 1.0;
    }
    if !lambda_min.is_finite() {
        return 1.0;
    }
    // `ramp_down(x, under, clear)`: the still-active weight, `1` for `x ≤ under`,
    // `0` for `x ≥ clear`, C¹ cubic smoothstep `1 − (3t² − 2t³)` between.
    #[inline]
    fn ramp_down(x: f64, under: f64, clear: f64) -> f64 {
        if x <= under {
            return 1.0;
        }
        if x >= clear {
            return 0.0;
        }
        let t = (x - under) / (clear - under);
        1.0 - t * t * (3.0 - 2.0 * t)
    }
    let w_abs = ramp_down(
        lambda_min,
        CONDITIONING_GATE_ABSOLUTE,
        CONDITIONING_GATE_ABSOLUTE_CLEAR,
    );
    let ratio = (lambda_min / lambda_max).max(f64::MIN_POSITIVE);
    let w_rel = ramp_down(
        ratio.log10(),
        CONDITIONING_GATE_RELATIVE.log10(),
        CONDITIONING_GATE_RELATIVE_CLEAR.log10(),
    );
    w_abs.max(w_rel)
}

/// Partial derivatives `(∂G/∂λ_min, ∂G/∂λ_max)` of the conditioning gate weight
/// `G = max(w_abs, w_rel)` (see [`conditioning_gate_weight`]).
///
/// The gate scales the Jeffreys curvature the LAML value folds into
/// `½ log|H + S_λ + G·H_Φ_raw|`; because `λ_min, λ_max` move with β through the
/// inner mode response, the gate's own mode-response variation is part of the
/// EXACT outer hypergradient. Dropping it (treating `G` as locally constant in the
/// drift) desyncs the analytic outer gradient from its own value precisely when
/// the gate sits in its smooth transition band — the residual tension-axis
/// mismatch in gam#854, even when no eigenvalue is floored. Returns `(0, 0)` on the
/// saturated / degenerate branches where `G` is locally constant (so the outer
/// drift is byte-unchanged on every fully-active or well-conditioned fit).
pub(crate) fn conditioning_gate_weight_grad(lambda_min: f64, lambda_max: f64) -> (f64, f64) {
    if lambda_max <= 0.0 || !lambda_min.is_finite() {
        // Matches `conditioning_gate_weight`'s constant-`1.0` early returns.
        return (0.0, 0.0);
    }
    // `ramp_down`'s value and derivative: `d/dx [1 − (3t² − 2t³)] = −6 t (1−t) / (clear − under)`
    // on the open band (`under < x < clear`), `0` at/outside both knots (C¹).
    #[inline]
    fn ramp_down_value_and_deriv(x: f64, under: f64, clear: f64) -> (f64, f64) {
        if x <= under {
            return (1.0, 0.0);
        }
        if x >= clear {
            return (0.0, 0.0);
        }
        let span = clear - under;
        let t = (x - under) / span;
        let value = 1.0 - t * t * (3.0 - 2.0 * t);
        let deriv = -6.0 * t * (1.0 - t) / span;
        (value, deriv)
    }
    let (w_abs, dw_abs_dlmin) = ramp_down_value_and_deriv(
        lambda_min,
        CONDITIONING_GATE_ABSOLUTE,
        CONDITIONING_GATE_ABSOLUTE_CLEAR,
    );
    let ratio = (lambda_min / lambda_max).max(f64::MIN_POSITIVE);
    let (w_rel, dw_rel_dlogratio) = ramp_down_value_and_deriv(
        ratio.log10(),
        CONDITIONING_GATE_RELATIVE.log10(),
        CONDITIONING_GATE_RELATIVE_CLEAR.log10(),
    );
    // `G = w_abs.max(w_rel)`: only the active branch varies the max. A tie is a
    // measure-zero kink the smooth band stays away from; resolve it to `w_abs`
    // (consistent, and the dominant branch in the small-`n` absolute regime).
    if w_abs >= w_rel {
        (dw_abs_dlmin, 0.0)
    } else {
        // `∂ log₁₀(λ_min/λ_max)/∂λ_min = 1/(λ_min ln10)`,
        // `∂ log₁₀(λ_min/λ_max)/∂λ_max = −1/(λ_max ln10)`.
        let ln10 = std::f64::consts::LN_10;
        (
            dw_rel_dlogratio / (lambda_min * ln10),
            -dw_rel_dlogratio / (lambda_max * ln10),
        )
    }
}

/// Below this joint dimension the dense reduced eigendecomposition in
/// [`joint_jeffreys_term`] is itself cheap (`O(p³)` with `p` in the tens — e.g.
/// the BMS-probit `p≈51` fit), so the matrix-free pre-check below would only add
/// `O(p·k)` matvecs for no asymptotic win and a (tiny) chance of a conservative
/// false-fall-through. We therefore run the exact path directly for small joint
/// systems and reserve the cheap pre-check for the wide systems whose `O(p³)`
/// eigendecomposition (and the dense `H_id` it needs) is the cost we want to
/// avoid on a well-conditioned fit. This threshold matches the matrix-free joint
/// path's `JOINT_MATRIX_FREE_MIN_DIM_AT_LARGE_N` so the pre-check exists exactly
/// where the dense formation is the regression.
pub const CHEAP_CONDITIONING_PRECHECK_MIN_DIM: usize = 128;

/// Safety factor by which the CONSERVATIVE spectral bounds must clear each
/// conditioning gate before the cheap pre-check is allowed to declare the term
/// skippable. The Lanczos bounds below are already one-sided-conservative
/// (`λ_min` is bounded from BELOW, `λ_max` from ABOVE — see
/// [`cheap_conditioning_bounds`]), so a clearance factor of `1` would already be
/// correct; we demand an extra `8×` margin purely as defense-in-depth against
/// round-off in the bound arithmetic and against a `k`-step subspace that has not
/// yet resolved the extreme Ritz pair. The cost of being wrong is asymmetric:
/// a false SKIP omits the curvature that prevents non-convergence (unacceptable),
/// whereas a false FALL-THROUGH merely pays the exact dense path we would have
/// paid anyway. The margin is therefore set firmly on the safe side.
pub(crate) const CHEAP_PRECHECK_SAFETY_MARGIN: f64 = 8.0;

/// Number of Lanczos steps for the cheap conditioning pre-check. A handful of
/// steps with full reorthogonalization resolves the extreme Ritz pair to far
/// better than the `8×` safety margin on the realistic spectra here (a
/// well-conditioned joint information whose `λ_min` we only need to certify is
/// `≳ 8` and whose ratio we only need to certify is `≳ 8e-8`), while keeping the
/// pre-check at `O(p·k)` matvecs — negligible against the `O(p³)`/dense-`H_id`
/// path it guards. Capped at `p` for tiny systems (which the size gate already
/// routes to the exact path anyway).
pub(crate) const CHEAP_PRECHECK_LANCZOS_STEPS: usize = 12;

/// Relative residual below which an extreme Ritz pair counts as "converged" and
/// its residual-augmented eigenvalue bound may be trusted. Measured against the
/// spectral scale `θ_max` (floored at 1): once `‖H y − θ y‖ ≤ 1e-3·scale`, the
/// extreme eigenvalue is resolved to three digits, far tighter than the `8×`
/// safety margin the skip decision then applies. An unresolved residual returns
/// `None` so the caller falls through to the exact dense path.
pub(crate) const CHEAP_PRECHECK_RITZ_REL_TOL: f64 = 1e-3;

/// Conservative extreme-eigenvalue bounds `(λ_min_lower, λ_max_upper)` for the
/// reduced information `H_id = Z_Jᵀ H Z_J` on the FULL span (`Z_J = I`, so
/// `H_id = H`), computed MATRIX-FREE from a Hessian-vector product `hv` and the
/// dimension `p`, WITHOUT ever forming the dense `H_id` or its eigendecomposition.
///
/// METHOD. `k`-step Lanczos with FULL reorthogonalization builds a symmetric
/// tridiagonal `T_k` from a dense aperiodic start vector. With full reorth the
/// factorization `H Q_k = Q_k T_k + β_k q_{k+1} e_kᵀ` holds exactly, so each Ritz
/// pair `(θ_i, y_i)` has the SHARP residual `‖H(Q_k y_i) − θ_i (Q_k y_i)‖ =
/// β_k·|e_kᵀ y_i| =: res_i` (Saad, *Numerical Methods for Large Eigenvalue
/// Problems*, §6; Parlett, *The Symmetric Eigenvalue Problem*). The residual
/// eigenvalue bound then guarantees a true eigenvalue of `H` within `res_i` of
/// `θ_i`.
///
/// CONSERVATIVE one-sided bounds. The EXTREME Ritz pairs converge FIRST under
/// Lanczos (Kaniel–Paige); the TRUST GATE below requires the extreme residuals to
/// be small relative to the spectral scale `θ_max` before trusting the estimate,
/// which (with full reorth, so spurious/ghost eigenvalues cannot arise) means the
/// Krylov space has resolved BOTH ends of the spectrum — hence `θ_min`/`θ_max`
/// ARE `λ_min`/`λ_max` to within `res_min`/`res_max`. We therefore return:
///   * `λ_min(H) ≥ θ_min − res_min`  (LOWER bound on the smallest eigenvalue),
///   * `λ_max(H) ≤ θ_max + res_max`  (UPPER bound on the largest eigenvalue).
/// These bias the conditioning estimate toward "looks WORSE-conditioned than it
/// is" — the direction that makes a SKIP decision safe: if even these pessimistic
/// bounds clear the gate (with the caller's extra `8×` margin), the true spectrum
/// clears it by more.
///
/// Returns `None` — forcing the caller to fall through to the EXACT dense path —
/// whenever the estimate cannot be trusted: a non-finite/degenerate Krylov space
/// (zero-start collapse) OR an UNCONVERGED extreme Ritz pair (`res_min`/`res_max`
/// not small). The latter is the critical safety valve: if the cheap iteration
/// has not resolved the bottom of the spectrum it NEVER authorises a skip, so a
/// hidden small eigenvalue cannot be missed — the term is then formed exactly.
pub(crate) fn cheap_conditioning_bounds<HvFn>(
    mut hv: HvFn,
    p: usize,
) -> Result<Option<(f64, f64)>, String>
where
    HvFn: FnMut(&Array1<f64>) -> Result<Array1<f64>, String>,
{
    if p == 0 {
        return Ok(None);
    }
    let steps = CHEAP_PRECHECK_LANCZOS_STEPS.min(p);
    // Deterministic dense start vector with a NON-UNIFORM, aperiodic pattern. A
    // plain all-ones vector is orthogonal to any eigenvector whose entries sum to
    // zero — a symmetry an exactly-balanced Fisher information could exhibit,
    // which would hide that eigenvalue from the Krylov space. Seeding each entry
    // from an irrational-rotation sequence (`frac(i·φ) − ½`, φ the golden ratio)
    // gives a deterministic, reproducible (no RNG in the hot path) start that has
    // a non-zero component along every eigenvector for any realistic operator, so
    // the extreme Ritz pairs are resolved and the trust gate below is meaningful.
    let mut q0 = vec![0.0_f64; p];
    let golden = 0.618_033_988_749_894_8_f64; // frac(golden ratio)
    for (i, qi) in q0.iter_mut().enumerate() {
        let frac = ((i as f64 + 1.0) * golden).fract();
        *qi = frac - 0.5;
    }
    let q_norm = norm2_slice(&q0);
    if !(q_norm.is_finite() && q_norm > 0.0) {
        return Ok(None);
    }
    // Single shared matrix-free Lanczos primitive with FULL reorthogonalization,
    // so `H Q_k = Q_k T_k + β_k q_{k+1} e_kᵀ` holds exactly and the per-Ritz-pair
    // residual `β_k·|e_kᵀ y_i|` (read from `residual_norm` and the last row of the
    // Ritz vectors below) is a SHARP eigenvalue bound. A matvec/HVP failure or a
    // non-finite iterate surfaces as `None` (caller falls through to the exact
    // dense path), preserving the conservative "never authorise a skip on an
    // unconverged check" contract.
    let mut hv_failed: Option<String> = None;
    let eigen = match symmetric_lanczos_eigenpairs(
        p,
        &q0,
        SymmetricLanczosOptions {
            max_steps: steps,
            // Lucky-breakdown floor: stop (β_k reported as 0 ⇒ EXACT Ritz
            // spectrum, tight bounds) once the next Lanczos vector is at the
            // machine-precision noise floor of a unit-norm start, before any
            // divide-by-≈0 can pollute the basis. Conservative: a non-breakdown
            // small residual still propagates as a faithful (small) bound below.
            residual_tol: f64::EPSILON,
            local_reorthogonalize: false,
            full_reorthogonalize: true,
        },
        |q, out| {
            let qv = Array1::from_vec(q.to_vec());
            let w = match hv(&qv) {
                Ok(w) => w,
                Err(e) => {
                    hv_failed = Some(e);
                    return Err("cheap_conditioning_bounds: HVP failed".to_string());
                }
            };
            if w.len() != p || w.iter().any(|x| !x.is_finite()) {
                return Err(
                    "cheap_conditioning_bounds: HVP produced non-finite/ill-sized output"
                        .to_string(),
                );
            }
            out.copy_from_slice(w.as_slice().ok_or_else(|| {
                "cheap_conditioning_bounds: HVP output not contiguous".to_string()
            })?);
            Ok(())
        },
    ) {
        Ok(eigen) => eigen,
        Err(_) => {
            // Propagate a genuine HVP error to the caller; treat a Lanczos
            // numerical degeneracy (zero-start collapse, non-finite iterate) as
            // an untrusted estimate and fall through to the exact path.
            if let Some(e) = hv_failed {
                return Err(e);
            }
            return Ok(None);
        }
    };
    let ritz = eigen.eigenvalues;
    let ritz_vecs = eigen.eigenvectors;
    let last_residual_norm = eigen.residual_norm;
    let k = ritz.len();
    if k == 0 {
        return Ok(None);
    }
    // Index of the smallest / largest Ritz value and the eigenvector last
    // components, which give the SHARP per-pair residual `β_k·|e_kᵀ y_i|` ≤ β_k.
    let mut idx_min = 0usize;
    let mut idx_max = 0usize;
    for i in 1..k {
        if ritz[i] < ritz[idx_min] {
            idx_min = i;
        }
        if ritz[i] > ritz[idx_max] {
            idx_max = i;
        }
    }
    let theta_min = ritz[idx_min];
    let theta_max = ritz[idx_max];
    if !theta_min.is_finite() || !theta_max.is_finite() {
        return Ok(None);
    }
    // Sharp Ritz residuals: `‖H y_i − θ_i y_i‖ = β_k·|e_kᵀ y_i|` (β_k =
    // last_residual_norm, the norm of the unnormalised next Lanczos vector). The
    // last row of `ritz_vecs` holds `e_kᵀ y_i` for every Ritz pair.
    let last_row = k - 1;
    let res_min = last_residual_norm * ritz_vecs[[last_row, idx_min]].abs();
    let res_max = last_residual_norm * ritz_vecs[[last_row, idx_max]].abs();
    // TRUST GATE (Krylov near-invariance). With FULL reorthogonalization the
    // Lanczos factorization satisfies `H Q_k = Q_k T_k + β_k q_{k+1} e_kᵀ` exactly,
    // and the EXTREME Ritz pairs converge FIRST (Kaniel–Paige). When the extreme
    // residuals are small relative to the spectral scale θ_max, the Krylov space
    // has resolved the spectrum's ENDS, so `θ_min`/`θ_max` are faithful estimates
    // of `λ_min`/`λ_max` and the residual-augmented one-sided bounds below are
    // sound. When a residual is NOT small (the bottom of the spectrum is
    // unresolved — e.g. a start vector poorly aligned with the extreme
    // eigenspace), we return `None` so the caller CONSERVATIVELY falls through to
    // the exact dense path rather than trusting an unconverged estimate. This is
    // the safety valve: an unconverged cheap check never authorises a skip.
    let scale = theta_max.abs().max(1.0);
    let converged_tol = CHEAP_PRECHECK_RITZ_REL_TOL * scale;
    if res_min > converged_tol || res_max > converged_tol {
        return Ok(None);
    }
    // Conservative one-sided bounds. Subtract/ add the SHARP residual: there is a
    // true eigenvalue of `H` within `res_min` of `θ_min` and within `res_max` of
    // `θ_max` (residual eigenvalue bound), and with the extreme pairs converged
    // these ARE the extreme eigenvalues, so:
    //   λ_min(H) ≥ θ_min − res_min,   λ_max(H) ≤ θ_max + res_max.
    let lambda_min_lb = theta_min - res_min;
    let lambda_max_ub = theta_max + res_max;
    Ok(Some((lambda_min_lb, lambda_max_ub)))
}

/// Cheap MATRIX-FREE pre-check answering "is the Jeffreys term provably
/// SKIPPABLE here?" using only Hessian-vector products against the full-span
/// reduced information `H_id = H` — WITHOUT forming the dense `H_id` or running
/// the `O(p³)` eigendecomposition in [`joint_jeffreys_term`].
///
/// Returns `true` ONLY when the CONSERVATIVE spectral bounds from
/// [`cheap_conditioning_bounds`] clear BOTH conditioning gates with the
/// [`CHEAP_PRECHECK_SAFETY_MARGIN`] safety factor, i.e. the fit is so clearly
/// well-conditioned that the exact gate is certain to skip too. In that case the
/// caller may return the EXACT-ZERO term (byte-identical to the gated-off dense
/// path) without forming anything dense. Returns `false` (the conservative
/// default) whenever the cheap bounds are unresolved, non-positive, or merely
/// close to the gate — the caller then falls through to the exact dense
/// formation + gate, so the term is still computed exactly wherever it might be
/// needed.
///
/// CORRECTNESS. `cheap_conditioning_bounds` returns `λ_min_lb ≤ λ_min(H)` and
/// `λ_max_ub ≥ λ_max(H)`. Hence
///   `λ_min(H) ≥ λ_min_lb` and `λ_min(H)/λ_max(H) ≥ λ_min_lb/λ_max_ub`,
/// so when `λ_min_lb ≥ MARGIN·CONDITIONING_GATE_ABSOLUTE` and
/// `λ_min_lb/λ_max_ub ≥ MARGIN·CONDITIONING_GATE_RELATIVE` the TRUE spectrum
/// satisfies the exact `conditioning_gate_skips` predicate by at least `MARGIN×`
/// — the exact path would skip, so skipping cheaply is byte-identical. The
/// converse cases never skip, preserving exactness where the term bites.
pub fn jeffreys_term_skippable_via_matvec<HvFn>(hv: HvFn, p: usize) -> Result<bool, String>
where
    HvFn: FnMut(&Array1<f64>) -> Result<Array1<f64>, String>,
{
    if p < CHEAP_CONDITIONING_PRECHECK_MIN_DIM {
        // Small systems: the exact dense eigh is already cheap; do not pre-check.
        return Ok(false);
    }
    let (lambda_min_lb, lambda_max_ub) = match cheap_conditioning_bounds(hv, p)? {
        Some(bounds) => bounds,
        None => return Ok(false),
    };
    if !(lambda_min_lb.is_finite() && lambda_max_ub.is_finite()) {
        return Ok(false);
    }
    // The conservative lower bound must itself be positive (and large) — a
    // non-positive `λ_min_lb` cannot certify SPD/well-conditioned, so never skip.
    if lambda_min_lb <= 0.0 || lambda_max_ub <= 0.0 {
        return Ok(false);
    }
    // A full skip now requires the SMOOTH weight to be exactly 0, i.e. the
    // spectrum must clear the UPPER (`*_CLEAR`) knots of both ramps — not merely
    // the lower (firing) thresholds. The conservative bounds must clear those
    // upper knots by the safety margin.
    let absolute_clears =
        lambda_min_lb >= CHEAP_PRECHECK_SAFETY_MARGIN * CONDITIONING_GATE_ABSOLUTE_CLEAR;
    let relative_clears = lambda_min_lb / lambda_max_ub
        >= CHEAP_PRECHECK_SAFETY_MARGIN * CONDITIONING_GATE_RELATIVE_CLEAR;
    Ok(absolute_clears && relative_clears)
}

/// Orthonormal basis of one block's Jeffreys span.
///
/// `columns` is `p x m` with orthonormal columns spanning `ker(S_aggregate)`
/// (the parametric + smooth-null directions). `m == 0` means the block is fully
/// penalized in every direction and gets no Jeffreys term.
#[derive(Debug, Clone)]
pub struct JeffreysSubspace {
    /// `p x m` orthonormal basis of the under-identified span (m <= p).
    pub columns: Array2<f64>,
}

impl JeffreysSubspace {
    /// Dimension `m` of the under-identified span (columns of the basis).
    #[inline]
    pub fn span_dim(&self) -> usize {
        self.columns.ncols()
    }
}

/// Build `Z_J` for a block: the FULL identifiable coefficient span of the
/// (post-rank-deficiency-removal) reduced block design — the entire reduced
/// coefficient space, `Z_J = I_p`.
///
/// PRINCIPLE (why this is the right span, not `ker(S)`). The Jeffreys penalty
/// `Φ = ½ log|I_r(β)|` is SELF-LIMITING: its score is `O(1)` against the data's
/// `O(n)` Fisher information. So on a data-identified direction (penalized OR
/// not) its only effect is the `O(1/n)` Firth bias-reduction correction — it
/// does not bias a genuine smooth fit. It bites ONLY where `I(β)` is
/// near-singular, i.e. a separating direction, supplying the missing
/// `O(1)`-bounding curvature there regardless of whether that direction lives in
/// `ker(S)` (an unpenalized nullspace direction) or `range(S)` (a penalized
/// spline direction). Scoping `Z_J` to `ker(S)` only — the previous behavior —
/// could not reach a near-separation on a penalized spline direction, which is
/// the residual BMS-probit pathology. Using the full identifiable span makes the
/// inner objective coercive with a finite unique minimizer on EVERY direction,
/// without any design surgery and with the optimizer untouched.
///
/// `aggregate_penalty` is `p x p` and PSD (`sum_k S_k`); it is used only to
/// validate squareness and pick up `p`. Rank-softness, if any, is absorbed by
/// the reduced-Fisher Cholesky in [`joint_jeffreys_term`] (which simply omits the
/// `Φ` value contribution for a not-yet-SPD trial point while the step machinery
/// still bounds the coefficient).
pub fn jeffreys_subspace_from_penalty(
    aggregate_penalty: ArrayView2<'_, f64>,
) -> Result<JeffreysSubspace, String> {
    let p = aggregate_penalty.nrows();
    if aggregate_penalty.ncols() != p {
        return Err(format!(
            "jeffreys_subspace: aggregate penalty must be square, got {}x{}",
            aggregate_penalty.nrows(),
            aggregate_penalty.ncols()
        ));
    }
    if p == 0 {
        return Ok(JeffreysSubspace {
            columns: Array2::zeros((0, 0)),
        });
    }
    Ok(JeffreysSubspace {
        columns: Array2::eye(p),
    })
}

/// Tier-B Jeffreys term on the joint under-identified span, computed directly
/// from the coupled joint Hessian `H` (NOT from a single-eta
/// `FirthDenseOperator`). This is the path BMS / survival-marginal-slope /
/// location-scale GAMLSS take: their working curvature block IS the Fisher
/// information at the working point, so the Jeffreys penalty is
/// `Phi_J = 1/2 log|Z_J^T H Z_J|` on the under-identified span `Z_J`.
///
/// Returns `(phi, grad, hphi)`:
///   * `phi`   = `1/2 log|H_id|`, the objective contribution (`H_id = Z_J^T H Z_J`).
///   * `grad`  = the `p`-vector `dPhi/dbeta`, with `grad[k] = 1/2 tr(H_id^{-1} Z_J^T Hdot[e_k] Z_J)`.
///   * `hphi`  = the `p x p` symmetric curvature contribution to the penalized
///               Hessian, the leading Gauss-Newton term
///               `1/2 sum over reduced pairs`. We use the positive-semidefinite
///               Gauss-Newton surrogate `H_Phi = 1/2 * J^T H_id^{-1} J` built
///               from the reduced gradient sensitivities, which supplies the
///               correct O(n) automatic curvature that bounds a near-separating
///               direction to O(1) while keeping `H_pen + H_Phi` SPD.
///
/// `hessian_dir` is a closure returning `Hdot[d] = d/d eps H(beta + eps d)|_0`
/// for a full coefficient-space direction `d` (the exact joint-Hessian
/// directional derivative the inner Newton already exposes). `Z_J` is the
/// `p x m` joint under-identified basis (block-diagonal stack of per-block
/// `Z_J`). When `m == 0` this returns a zero term.
pub fn joint_jeffreys_term<DirFn>(
    h_joint: ArrayView2<'_, f64>,
    z_j: ArrayView2<'_, f64>,
    hessian_dir: DirFn,
) -> Result<(f64, Array1<f64>, Array2<f64>), String>
where
    DirFn: Fn(&Array1<f64>) -> Result<Option<Array2<f64>>, String> + Sync,
{
    let p = h_joint.nrows();
    if h_joint.ncols() != p {
        return Err(format!(
            "joint_jeffreys_term: H must be square, got {}x{}",
            h_joint.nrows(),
            h_joint.ncols()
        ));
    }
    if z_j.nrows() != p {
        return Err(format!(
            "joint_jeffreys_term: Z_J has {} rows, expected {} to match H",
            z_j.nrows(),
            p
        ));
    }
    let m = z_j.ncols();
    if m == 0 {
        return Ok((0.0, Array1::zeros(p), Array2::zeros((p, p))));
    }
    // H_id = Z_J^T H Z_J  (m x m reduced information on the Jeffreys span).
    let hz = h_joint.dot(&z_j);
    let h_id = z_j.t().dot(&hz);
    // Symmetrize defensively (observed-information round-off can break exact
    // symmetry).
    let mut h_id_sym = Array2::<f64>::zeros((m, m));
    for i in 0..m {
        for j in 0..m {
            h_id_sym[[i, j]] = 0.5 * (h_id[[i, j]] + h_id[[j, i]]);
        }
    }
    // FULL-SPAN ROBUSTNESS. With the Jeffreys span equal to the FULL identifiable
    // coefficient space, `H_id` is the (reduced) observed information over every
    // direction. For non-canonical links (e.g. probit) the observed information
    // need NOT be PSD away from the mode, so a plain Cholesky would fail at
    // off-mode trial points and reject every outer seed. The Jeffreys prior is
    // `Φ = ½ log det I(β)` with `I` the EXPECTED (PSD) Fisher information; we
    // realise that here through the symmetric eigendecomposition, flooring each
    // eigenvalue at a tiny absolute ridge so `Φ` is the log-volume of the
    // POSITIVE curvature and the reduced inverse is the floored (pseudo-)inverse.
    // On an identified direction the data's O(n) curvature dwarfs the floor, so
    // the value, gradient and curvature are the exact Jeffreys quantities there;
    // a genuinely separating direction has near-zero curvature, where the floor
    // simply keeps `Φ` finite while the `H_Φ` curvature below grows to bound it.
    let (evals, evecs) = h_id_sym.eigh(Side::Lower).map_err(|e| {
        format!("joint_jeffreys_term: reduced-information eigendecomposition failed: {e}")
    })?;
    let lambda_max = evals.iter().cloned().fold(0.0_f64, f64::max);
    // CONDITIONING GATE ("no cost on easy fits"). The eigendecomposition we just
    // computed gives the full reduced spectrum; the worst-conditioned direction
    // is `λ_min`. We skip the term (zero value, gradient and curvature) only when
    // the reduced information is well-conditioned BOTH relatively
    // (`λ_min/λ_max ≥ CONDITIONING_GATE_RELATIVE`) AND absolutely
    // (`λ_min ≥ CONDITIONING_GATE_ABSOLUTE`, the `n`-aware criterion): every
    // direction is then identified by the data at `O(n)` strength, the
    // self-limiting Jeffreys term is negligible, and a clean/well-conditioned fit
    // stays byte-identical to the un-penalized inner Newton. If EITHER gate
    // reports under-identification — including an absolutely-near-separating
    // direction at small `n` that the scale-free relative ratio alone would miss
    // — we fall through to the floored log-det term below, the `O(1)`-bounding
    // curvature this machinery exists to supply.
    let gate_weight = {
        let lambda_min = evals.iter().cloned().fold(f64::INFINITY, f64::min);
        conditioning_gate_weight(lambda_min, lambda_max)
    };
    if gate_weight == 0.0 {
        return Ok((0.0, Array1::zeros(p), Array2::zeros((p, p))));
    }
    // Absolute floor relative to the dominant identified curvature: negligible on
    // identified directions (O(n)), positive on separating ones.
    let floor = (REDUCED_INFO_RELATIVE_FLOOR * lambda_max).max(REDUCED_INFO_ABSOLUTE_FLOOR);
    // FLOOR β-DEPENDENCE (root cause of gam#826 / the below-floor value↔gradient
    // mismatch). The regularization floor is `max(REL·λ_max, ABS)`; in the active
    // RELATIVE regime it scales with `λ_max(β)`, which is itself a function of β.
    // The below-floor antiderivative is `g(λ; floor) = λ/floor + ln(floor) − 1`, so
    // for a below-floor eigenvalue the TOTAL derivative `dΦ/dβ_k` carries, beyond
    // the eigenvalue term `(1/floor) ∂λ/∂β_k`, the floor-response term
    //   `∂g/∂floor · ∂floor/∂β_k = (1/floor − λ/floor²) · ∂floor/∂β_k`.
    // The earlier gradient differentiated only the eigenvalue and treated the floor
    // as constant, so on any fit with an eigenvalue parked below the floor the
    // analytic gradient did not equal `d/dβ Φ` — the inner joint-Newton KKT residual
    // could not reach zero (a contributor to the coupled location-scale
    // non-convergence). We restore the exact pair below by adding the floor-response
    // term, using `∂λ_max/∂β_k = v_maxᵀ D_k v_max` (first-order eigenvalue
    // perturbation; `D_k = Z_Jᵀ Hdot[e_k] Z_J` is already formed in the gradient
    // loop). When the floor is in the ABSOLUTE regime (`REL·λ_max ≤ ABS`, including
    // a non-positive `λ_max`) the floor is β-independent, so the term is exactly
    // zero and nothing is added — preserving the PSD-fit and indefinite fast paths
    // byte-for-byte. The eigenvalue-perturbation formula is exact only at a SIMPLE
    // dominant eigenvalue; a tied `λ_max` is a measure-zero kink the smooth-gate
    // band keeps away from (and the floor-response term is itself O(λ/floor)-tiny
    // there), so no special-casing is warranted.
    let floor_in_relative_regime =
        lambda_max > 0.0 && REDUCED_INFO_RELATIVE_FLOOR * lambda_max >= REDUCED_INFO_ABSOLUTE_FLOOR;
    // Index of the dominant eigenvalue `λ_max` (the one the relative floor
    // tracks), needed for `∂λ_max/∂β_k = v_maxᵀ D_k v_max = (Ṽ_k)_mm`. Only
    // consulted in the relative regime.
    let lambda_max_idx: Option<usize> = if floor_in_relative_regime {
        let mut idx_max = 0usize;
        for i in 1..m {
            if evals[i] > evals[idx_max] {
                idx_max = i;
            }
        }
        Some(idx_max)
    } else {
        None
    };
    // SINGLE-EMISSION (gam#931). The Jeffreys value and first derivative are
    // emitted by the atom below from one spectrum and one floor. The live call
    // site supplies the same reduced drifts it already needs for curvature; the
    // atom owns the scalar projection (`g`, `g'_λ`, and `g'_floor`) so no inline
    // value/gradient branch can drift from it.
    let value_atom = super::atoms::JeffreysLogdetAtom {
        eigvals: evals.clone(),
        floor,
        gate_weight,
        reduced_drift: HashMap::new(),
        floor_drift: HashMap::new(),
        stratum: super::atoms::StratumFingerprint {
            kept_rank: m,
            min_relative_eigengap: 0.0,
        },
    };
    let phi = super::atoms::CriterionAtom::value(&value_atom);
    // Gradient: grad[k] = ½ tr(K · Z_Jᵀ Hdot[e_k] Z_J) = ½ Σ_i d_i (Ṽ_k)_ii with
    // Ṽ_k = Vᵀ D_k V the reduced derivative rotated into the eigenbasis. For the
    // inner-Newton dense path the Hessian is beta-dependent through the working
    // weights only along coefficient directions; we evaluate Hdot per canonical
    // coefficient axis.
    let mut grad = Array1::<f64>::zeros(p);
    // PARALLEL DIRECTIONAL DERIVATIVES. Each canonical axis `e_k` requires one
    // FULL-DATA directional-derivative pass `Hdot[e_k]` (the n-row inner-Newton
    // exact derivative) — the dominant cost (e.g. ~1.5 s × p=35 ≈ 55 s serial on
    // a biobank fit, deterministic on the cycle where the conditioning gate
    // arms). The p passes are fully independent pure evaluations of
    // `(family, states, e_k)`, so we fan them across the Rayon pool here. The
    // `Sync` bound on `DirFn` makes the evaluator safe to call concurrently;
    // combined with the nested-BLAS guard each pass runs single-threaded faer,
    // so the directions fan across cores with no rayon×BLAS oversubscription.
    //
    // The cheap per-k reduction (D_k = ZᵀHdotZ rotation and atom input writes)
    // stays SERIAL below over the index-ordered results, so the outputs are
    // bit-identical to the original `for k in 0..p` loop. Early-return
    // semantics are preserved exactly: if ANY axis yields `Ok(None)` the family
    // does not expose the exact derivative and the whole term degenerates to
    // `(gate_weight·phi, 0, 0)` (matching the serial first-None behaviour); any
    // `Err` propagates.
    let hdots: Vec<Array2<f64>> = {
        use rayon::iter::{IntoParallelIterator, ParallelIterator};
        let results: Vec<Result<Option<Array2<f64>>, String>> = (0..p)
            .into_par_iter()
            .map(|k| {
                let mut axis = Array1::<f64>::zeros(p);
                axis[k] = 1.0;
                // Mark this whole directional pass as a nested data-parallel
                // region so every faer GEMM it issues pins to `Par::Seq` instead
                // of re-fanning the global Rayon pool against this p-way axis
                // fan-out (the rayon×BLAS oversubscription guard from the
                // nested-BLAS fix). Bit-identical: faer partitions matmul output,
                // never the contracted axis.
                crate::linalg::faer_ndarray::with_nested_parallel(|| hessian_dir(&axis))
            })
            .collect();
        // Resolve in index order so the first anomaly (Err, then None, then a
        // shape mismatch) wins exactly as the original serial loop did.
        let mut hdots = Vec::with_capacity(p);
        for hdot in results {
            let hdot = match hdot? {
                Some(hdot) => hdot,
                None => {
                    // Family does not expose an exact directional derivative; the
                    // Jeffreys gradient/curvature degenerate to zero (objective
                    // still well-defined). This keeps the term safe rather than
                    // wrong.
                    return Ok((phi, Array1::zeros(p), Array2::zeros((p, p))));
                }
            };
            if hdot.nrows() != p || hdot.ncols() != p {
                return Err(format!(
                    "joint_jeffreys_term: Hdot shape {}x{} != {p}x{p}",
                    hdot.nrows(),
                    hdot.ncols()
                ));
            }
            hdots.push(hdot);
        }
        hdots
    };
    let mut reduced_drift: HashMap<usize, Arc<Array2<f64>>> = HashMap::with_capacity(p);
    let mut floor_drift: HashMap<usize, f64> = HashMap::new();
    for (k, hdot) in hdots.into_iter().enumerate() {
        // Reduced derivative D_k = Z_J^T Hdot Z_J (m x m), rotated into the
        // eigenbasis: Ṽ_k = Vᵀ D_k V.
        let hdz = hdot.dot(&z_j);
        let d_k = z_j.t().dot(&hdz);
        let a_k = evecs.t().dot(&d_k).dot(&evecs);
        // FLOOR-RESPONSE term (see the `floor` block above). The atom consumes
        // `floor_dot` beside `Ṽ_k`, so `dΦ/dβ_k` remains the derivative of its
        // own `value()`.
        if let Some(idx_max) = lambda_max_idx {
            let dlambda_max = a_k[[idx_max, idx_max]]; // v_maxᵀ D_k v_max
            floor_drift.insert(k, REDUCED_INFO_RELATIVE_FLOOR * dlambda_max);
        }
        reduced_drift.insert(k, Arc::new(a_k));
    }
    let gradient_atom = super::atoms::JeffreysLogdetAtom {
        eigvals: evals.clone(),
        floor,
        gate_weight,
        reduced_drift,
        floor_drift,
        stratum: super::atoms::StratumFingerprint {
            kept_rank: m,
            min_relative_eigengap: 0.0,
        },
    };
    for k in 0..p {
        let dir = super::atoms::ThetaDirection {
            index: Some(k),
            beta_dot: None,
            h_dot_total: None,
        };
        grad[k] = super::atoms::CriterionAtom::frozen_d1(&gradient_atom, &dir);
    }
    // EXACT Jeffreys curvature on the floored spectrum (gam#979), now emitted
    // by the same atom that emitted `phi` and `grad`. The penalized objective is
    // `−ℓ + ½βᵀSβ − Φ`, so the Newton system needs `−∇²Φ`. By the
    // Daleckii–Krein formula (first-order eigen-perturbation), with
    // `Ṽ_k = Vᵀ D_k V` and `Ψ` the divided differences of `d = g'`:
    //   ∇²Φ[a,b] = ½ Σ_ij Ψ_ij (Ṽ_a)_ij (Ṽ_b)_ij + ½ tr(K D_ab),
    // and `JeffreysLogdetAtom::second_order_curvature` keeps everything except
    // the second-directional-Hessian term `½ tr(K D_ab)` (a genuinely
    // separating direction's `D_ab` carries its vanishing curvature factor, so
    // that remainder is O(1) exactly where the term arms — the trust region owns
    // it). So
    //   H_Φ[a,b] = −½ Σ_ij Ψ_ij (Ṽ_a)_ij (Ṽ_b)_ij,
    // assembled as one BLAS-3 GEMM over the atom's stored rows. On an unfloored
    // PSD spectrum `Ψ_ij = −1/(λ_i λ_j)` and this is the classic PSD log-det
    // Gauss-Newton curvature `½ tr(K D_a K D_b)`.
    //
    // WHY NOT THE vec-GRAM `½⟨vec(K D_a), vec(K D_b)⟩` (the previous surrogate):
    // that object is `½ tr(D_a K² D_b)` — a `K²`-weighted Gram that puts
    // `1/floor² ≈ 1e20` PHANTOM curvature on every floored↔floored eigenpair,
    // whose true curvature (divided difference of two equal `1/floor` slopes)
    // is ZERO. The inner joint-Newton step along any Firth-active direction was
    // shrunk by up to twenty orders of magnitude while the rhs carried the
    // exact `∇Φ`, degrading Newton to a frozen/linear crawl — the gam#979
    // `phantom_multiplier_with_well_conditioned_H` grind (binary marginal-slope
    // slowdown, survival marginal-slope hang) and the spurious model-stationary
    // accepts the outer evaluator then rejects as unresolved KKT mass. The
    // divided-difference curvature is the exact β-derivative of the implemented
    // `∇Φ` (same `g`, same floor, same spectrum — modulo the floor-response and
    // `D_ab` remainders documented above), so the trust-region model matches
    // the objective and Newton recovers its quadratic rate. It is indefinite
    // exactly where `Φ` is (mixed-sign spectrum); the exact Moré–Sorensen
    // trust-region subproblem handles that rigorously.
    let hphi = gradient_atom.second_order_curvature(p)?;
    Ok((phi, grad, hphi))
}

/// Exact second-directional-Hessian completion for the Tier-B joint Jeffreys
/// curvature. [`joint_jeffreys_term`] builds the Daleckii-Krein
/// divided-difference part of `-∇²Φ`; this adds the omitted
/// `-1/2 tr(K · D_ab)` term, where `K` is the same floored reduced inverse and
/// `D_ab = Z_J^T H''[e_a,e_b] Z_J`.
pub fn joint_jeffreys_second_order_completion<Dir2Fn>(
    h_joint: ArrayView2<'_, f64>,
    z_j: ArrayView2<'_, f64>,
    hessian_second_dir: Dir2Fn,
) -> Result<Option<Array2<f64>>, String>
where
    Dir2Fn: Fn(&Array1<f64>, &Array1<f64>) -> Result<Option<Array2<f64>>, String> + Sync,
{
    let p = h_joint.nrows();
    if h_joint.ncols() != p {
        return Err(format!(
            "joint_jeffreys_second_order_completion: H must be square, got {}x{}",
            h_joint.nrows(),
            h_joint.ncols()
        ));
    }
    if z_j.nrows() != p {
        return Err(format!(
            "joint_jeffreys_second_order_completion: Z_J has {} rows, expected {p}",
            z_j.nrows()
        ));
    }
    let m = z_j.ncols();
    if m == 0 {
        return Ok(Some(Array2::zeros((p, p))));
    }

    let hz = h_joint.dot(&z_j);
    let h_id = z_j.t().dot(&hz);
    let mut h_id_sym = Array2::<f64>::zeros((m, m));
    for i in 0..m {
        for j in 0..m {
            h_id_sym[[i, j]] = 0.5 * (h_id[[i, j]] + h_id[[j, i]]);
        }
    }
    let (evals, evecs) = h_id_sym.eigh(Side::Lower).map_err(|e| {
        format!("joint_jeffreys_second_order_completion: reduced-information eigendecomposition failed: {e}")
    })?;
    let lambda_max = evals.iter().cloned().fold(0.0_f64, f64::max);
    let gate_weight = {
        let lambda_min = evals.iter().cloned().fold(f64::INFINITY, f64::min);
        conditioning_gate_weight(lambda_min, lambda_max)
    };
    if gate_weight == 0.0 {
        return Ok(Some(Array2::zeros((p, p))));
    }
    let floor = (REDUCED_INFO_RELATIVE_FLOOR * lambda_max).max(REDUCED_INFO_ABSOLUTE_FLOOR);
    let mut inv_diag = Array1::<f64>::zeros(m);
    for (i, &lam) in evals.iter().enumerate() {
        inv_diag[i] = floored_inverse(lam, floor);
    }
    let mut k_reduced = Array2::<f64>::zeros((m, m));
    for eig in 0..m {
        let weight = inv_diag[eig];
        if weight == 0.0 {
            continue;
        }
        for row in 0..m {
            let wr = weight * evecs[[row, eig]];
            for col in 0..m {
                k_reduced[[row, col]] += wr * evecs[[col, eig]];
            }
        }
    }

    let mut out = Array2::<f64>::zeros((p, p));
    // PARALLEL SECOND-DIRECTIONAL DERIVATIVES. Each upper-triangle pair `(a, b)`
    // needs one FULL-DATA mixed second-directional pass `H''[e_a, e_b]` — the
    // dominant cost (p(p+1)/2 independent passes). They are pure evaluations of
    // `(family, states, e_a, e_b)`, so we fan the pairs across the Rayon pool;
    // the `Sync` bound on `Dir2Fn` plus the nested-BLAS guard keep each pass
    // single-threaded in faer, so the pairs spread across cores without
    // rayon×BLAS oversubscription.
    //
    // The cheap per-pair reduction (D_ab rotation + K-trace) stays serial over
    // the index-ordered results, so the output is bit-identical to the original
    // double `for` loop. Early-return semantics preserved exactly: if ANY pair
    // yields `Ok(None)` the whole completion returns `Ok(None)` (the family does
    // not expose the exact second derivative); any `Err` propagates.
    let pairs: Vec<(usize, usize)> = (0..p).flat_map(|a| (a..p).map(move |b| (a, b))).collect();
    let h2s: Vec<Array2<f64>> = {
        use rayon::iter::{IntoParallelRefIterator, ParallelIterator};
        let results: Vec<Result<Option<Array2<f64>>, String>> = pairs
            .par_iter()
            .map(|&(a, b)| {
                let mut axis_a = Array1::<f64>::zeros(p);
                axis_a[a] = 1.0;
                let mut axis_b = Array1::<f64>::zeros(p);
                axis_b[b] = 1.0;
                // Pin nested faer GEMM to `Par::Seq` (see the value-path note).
                crate::linalg::faer_ndarray::with_nested_parallel(|| {
                    hessian_second_dir(&axis_a, &axis_b)
                })
            })
            .collect();
        // Resolve in pair order so the first anomaly (Err, then None, then a
        // shape mismatch) wins exactly as the original serial double loop did.
        let mut h2s = Vec::with_capacity(pairs.len());
        for (&(a, b), result) in pairs.iter().zip(results.into_iter()) {
            let h2 = match result? {
                Some(h2) => h2,
                None => return Ok(None),
            };
            if h2.dim() != (p, p) {
                return Err(format!(
                    "joint_jeffreys_second_order_completion: H''[{a},{b}] shape {:?} != ({p}, {p})",
                    h2.dim()
                ));
            }
            h2s.push(h2);
        }
        h2s
    };
    for (&(a, b), h2) in pairs.iter().zip(h2s.into_iter()) {
        let h2z = h2.dot(&z_j);
        let d_ab = z_j.t().dot(&h2z);
        let mut trace = 0.0_f64;
        for i in 0..m {
            for j in 0..m {
                trace += k_reduced[[i, j]] * d_ab[[j, i]];
            }
        }
        let value = -0.5 * gate_weight * trace;
        out[[a, b]] = value;
        out[[b, a]] = value;
    }
    Ok(Some(out))
}

/// Explicit (β-frozen) derivative `∂_ρ H_Φ|_β` of the gated joint-Jeffreys
/// curvature along an OUTER hyperparameter `ρ` (e.g. a log-penalty `log λ_m` or a
/// family log-scale `log ε_m`), for the augmented-LAML hypergradient.
///
/// THE GAP THIS CLOSES (gam#854). `H_Φ` is built from the JOINT Hessian
/// `H_joint(β, ρ) = H_data + Σ_m λ_m H_m^pen(β; ε_m)` (value path
/// [`joint_jeffreys_term`]), so it depends on ρ BOTH through β̂ — the mode response,
/// supplied by [`joint_jeffreys_hphi_directional_derivative`] — AND EXPLICITLY
/// through the `λ_m`/`ε_m` that scale and shape the penalty blocks INSIDE
/// `H_joint`. The outer score
///   `½ tr[(H+S_λ+H_Φ)⁻¹ ∂_ρ(H+S_λ+H_Φ)]`
/// therefore needs the explicit term `½ tr[(·)⁻¹ ∂_ρ H_Φ|_β]`; omitting it leaves
/// the analytic hypergradient short on exactly the most-active penalty axis (the
/// residual spatial-adaptive tension-axis miss).
///
/// The arithmetic is IDENTICAL to the mode-response drift, with the perturbation
/// sourced from the explicit ρ-derivatives instead of `Hdot[δ]`/`H²dot[δ,e_a]`:
///   * `pert_h = ∂_ρ H_joint|_β`              perturbs `H_id` and hence `K`,
///   * `pert_hessian_dir(e_a) = ∂_ρ Hdot[e_a]|_β`  perturbs each axis derivative `D_a`,
/// while the BASE `M_a = K D_a` uses `base_hessian_dir` (= the real `Hdot[e_a]` at
/// the current point). Returns the zero matrix when the family lacks the exact
/// derivatives or the conditioning gate skips the term, so a clean fit is
/// byte-unchanged.
pub fn joint_jeffreys_hphi_explicit_param_derivative<BaseFn, PertFn>(
    h_joint: ArrayView2<'_, f64>,
    z_j: ArrayView2<'_, f64>,
    pert_h: &Array2<f64>,
    base_hessian_dir: BaseFn,
    pert_hessian_dir: PertFn,
) -> Result<Array2<f64>, String>
where
    BaseFn: Fn(&Array1<f64>) -> Result<Option<Array2<f64>>, String> + Sync,
    PertFn: Fn(&Array1<f64>) -> Result<Option<Array2<f64>>, String> + Sync,
{
    joint_jeffreys_hphi_perturbation_derivative(
        h_joint,
        z_j,
        base_hessian_dir,
        pert_h,
        pert_hessian_dir,
    )
}

/// β-FIXED PREPARED BASE for the joint-Jeffreys curvature perturbation derivative.
///
/// PERF (the biobank #979 outer-gradient black hole). Every mode-response drift
/// `D_β H_Φ[v_k]` shares the SAME β̂-dependent base: the reduced-information
/// eigendecomposition (`evals`, `evecs`), the floor/gate, the divided-difference
/// kernel `Ψ`, AND — the dominant cost — the `p` per-axis first directional
/// derivatives `Hdot[e_a]` (each an `O(n)` row-stream over n≈348k biobank rows)
/// that form the base `M_a = K D_a` (stored as `a_rows = vec(Ṽ_a)`,
/// `aw_rows = vec(Ψ∘Ṽ_a)`). NONE of these depend on the perturbation direction;
/// the released code recomputed all `p` axis row-streams INSIDE every one of the
/// `k` drift calls, i.e. `k·p` redundant full-data passes per outer gradient
/// eval. Preparing the base ONCE and reusing it across all `k` directions
/// collapses that to a single `p`-axis sweep; each direction then pays only its
/// own genuinely-`δ`-dependent perturbation work (`Hdot[δ]` and the `p` second
/// directional derivatives `H²dot[δ,e_a]`). The arithmetic per direction is
/// byte-identical to [`joint_jeffreys_hphi_perturbation_derivative`].
pub(crate) struct JeffreysHphiDriftBase {
    p: usize,
    m: usize,
    z_j: Array2<f64>,
    evals: Array1<f64>,
    evecs: Array2<f64>,
    floor: f64,
    gate_weight: f64,
    psi: Array2<f64>,
    floor_in_relative_regime: bool,
    idx_min: usize,
    idx_max: usize,
    /// Per-axis rotated base derivative rows `vec(Ṽ_a)` (`p × m·m`).
    a_rows: Array2<f64>,
    /// `vec(Ψ ∘ Ṽ_a)` (`p × m·m`).
    aw_rows: Array2<f64>,
}

impl JeffreysHphiDriftBase {
    /// Prepare the β-fixed base ONCE. Runs the `p` first-directional-derivative
    /// row-streams via `base_hessian_dir`. Returns `Ok(None)` when the term is
    /// gated out (`H_Φ ≡ 0` in a neighborhood ⇒ every drift vanishes) or the
    /// family does not expose the exact first derivative on some axis (the
    /// released per-direction code returned the zero matrix there; preparing
    /// returns `None`, and the caller emits the same zero drift for every
    /// direction).
    pub(crate) fn prepare<BaseFn>(
        h_joint: ArrayView2<'_, f64>,
        z_j: ArrayView2<'_, f64>,
        base_hessian_dir: BaseFn,
    ) -> Result<Option<JeffreysHphiDriftBase>, String>
    where
        BaseFn: Fn(&Array1<f64>) -> Result<Option<Array2<f64>>, String> + Sync,
    {
        let p = h_joint.nrows();
        if h_joint.ncols() != p {
            return Err(format!(
                "JeffreysHphiDriftBase::prepare: H must be square, got {}x{}",
                h_joint.nrows(),
                h_joint.ncols()
            ));
        }
        if z_j.nrows() != p {
            return Err(format!(
                "JeffreysHphiDriftBase::prepare: Z_J has {} rows, expected {p}",
                z_j.nrows()
            ));
        }
        let m = z_j.ncols();
        if m == 0 || p == 0 {
            return Ok(None);
        }
        // Reproduce EXACTLY the value-path reduced information, conditioning gate,
        // and floored pseudo-inverse so the derivative is consistent with the
        // `H_Φ` the objective uses.
        let hz0 = h_joint.dot(&z_j);
        let h_id = z_j.t().dot(&hz0);
        let mut h_id_sym = Array2::<f64>::zeros((m, m));
        for i in 0..m {
            for j in 0..m {
                h_id_sym[[i, j]] = 0.5 * (h_id[[i, j]] + h_id[[j, i]]);
            }
        }
        let (evals, evecs) = h_id_sym.eigh(Side::Lower).map_err(|e| {
            format!("JeffreysHphiDriftBase::prepare: eigendecomposition failed: {e}")
        })?;
        let lambda_max = evals.iter().cloned().fold(0.0_f64, f64::max);
        let lambda_min = evals.iter().cloned().fold(f64::INFINITY, f64::min);
        let gate_weight = conditioning_gate_weight(lambda_min, lambda_max);
        if gate_weight == 0.0 {
            return Ok(None);
        }
        let floor = (REDUCED_INFO_RELATIVE_FLOOR * lambda_max).max(REDUCED_INFO_ABSOLUTE_FLOOR);
        let psi = floored_inverse_divided_differences(&evals, floor);
        let floor_in_relative_regime = lambda_max > 0.0
            && REDUCED_INFO_RELATIVE_FLOOR * lambda_max >= REDUCED_INFO_ABSOLUTE_FLOOR;
        let mut idx_max = 0usize;
        let mut idx_min = 0usize;
        for i in 1..m {
            if evals[i] > evals[idx_max] {
                idx_max = i;
            }
            if evals[i] < evals[idx_min] {
                idx_min = i;
            }
        }
        // The β-FIXED per-axis base: `Ṽ_a = Vᵀ D_a V` and `Ψ ∘ Ṽ_a`, formed from
        // the `p` first directional derivatives `Hdot[e_a]` (the dominant `O(n·p)`
        // cost — done ONCE here, reused for every drift direction).
        //
        // PARALLEL AXIS SWEEP. Each axis `e_a` requires one FULL-DATA
        // directional-derivative row-stream `Hdot[e_a]` (n≈348k biobank rows) — the
        // dominant cost of the whole outer-gradient eval. The `p` passes are
        // independent pure evaluations of `(family, β̂, e_a)`, so we fan them across
        // the Rayon pool exactly as the value-path `joint_jeffreys_term` does; the
        // `Sync` bound on `BaseFn` makes the evaluator safe to call concurrently and
        // the nested-BLAS guard pins each pass's faer GEMM to `Par::Seq` so the
        // axes fan across cores without rayon×BLAS oversubscription. The cheap
        // per-axis reduction (`Ṽ_a` rotation + row writes) stays serial over the
        // index-ordered results, so the prepared base is bit-identical to the
        // original serial loop. First-anomaly semantics preserved: any `Err`
        // propagates and the first `None` (family lacks the exact derivative on
        // some axis) collapses the whole base to `None` (zero drift everywhere).
        let z_owned = z_j.to_owned();
        let hdots: Vec<Array2<f64>> = {
            use rayon::iter::{IntoParallelIterator, ParallelIterator};
            let results: Vec<Result<Option<Array2<f64>>, String>> = (0..p)
                .into_par_iter()
                .map(|a| {
                    let mut axis = Array1::<f64>::zeros(p);
                    axis[a] = 1.0;
                    crate::linalg::faer_ndarray::with_nested_parallel(|| base_hessian_dir(&axis))
                })
                .collect();
            let mut hdots = Vec::with_capacity(p);
            for result in results {
                let hdot = match result? {
                    Some(hd) => hd,
                    None => return Ok(None),
                };
                if hdot.nrows() != p || hdot.ncols() != p {
                    return Err(format!(
                        "JeffreysHphiDriftBase::prepare: Hdot[e_a] shape {}x{} != {p}x{p}",
                        hdot.nrows(),
                        hdot.ncols()
                    ));
                }
                hdots.push(hdot);
            }
            hdots
        };
        let mut a_rows = Array2::<f64>::zeros((p, m * m));
        let mut aw_rows = Array2::<f64>::zeros((p, m * m));
        for (a, hdot_a) in hdots.into_iter().enumerate() {
            let d_a_raw = z_j.t().dot(&hdot_a.dot(&z_j));
            let mut d_a = Array2::<f64>::zeros((m, m));
            for i in 0..m {
                for j in 0..m {
                    d_a[[i, j]] = 0.5 * (d_a_raw[[i, j]] + d_a_raw[[j, i]]);
                }
            }
            let a_a = evecs.t().dot(&d_a).dot(&evecs);
            let mut col = 0usize;
            for i in 0..m {
                for j in 0..m {
                    a_rows[[a, col]] = a_a[[i, j]];
                    aw_rows[[a, col]] = psi[[i, j]] * a_a[[i, j]];
                    col += 1;
                }
            }
        }
        Ok(Some(JeffreysHphiDriftBase {
            p,
            m,
            z_j: z_owned,
            evals,
            evecs,
            floor,
            gate_weight,
            psi,
            floor_in_relative_regime,
            idx_min,
            idx_max,
            a_rows,
            aw_rows,
        }))
    }

    /// Per-direction drift `D[gate·H_Φ_raw]` reusing the prepared base. Only the
    /// `δ`-dependent perturbation is evaluated here: the reduced perturbation
    /// `Ḋ = Z_Jᵀ pert_h Z_J` and, per axis, `∂D_a = Z_Jᵀ pert_hessian_dir(e_a) Z_J`
    /// (the genuinely `δ`-dependent second directional derivatives). Returns
    /// `Ok(None)` (⇒ caller emits the zero drift) when the family does not expose
    /// the exact second derivative on some axis, matching the released semantics.
    pub(crate) fn perturbation_derivative<PertFn>(
        &self,
        pert_h: &Array2<f64>,
        pert_hessian_dir: PertFn,
    ) -> Result<Array2<f64>, String>
    where
        PertFn: Fn(&Array1<f64>) -> Result<Option<Array2<f64>>, String> + Sync,
    {
        let p = self.p;
        // Acquire the `p` second-directional axis matrices `{H²dot[δ,e_a]}` via the
        // per-axis closure (the parallel sweep preserved from before the batched
        // hook). The first `None` ⇒ the family lacks the exact second derivative
        // on some axis ⇒ the whole drift collapses to zero, matching the released
        // singular-hook semantics.
        let pert_hdots: Vec<Array2<f64>> = {
            use rayon::iter::{IntoParallelIterator, ParallelIterator};
            let results: Vec<Result<Option<Array2<f64>>, String>> = (0..p)
                .into_par_iter()
                .map(|a| {
                    let mut axis = Array1::<f64>::zeros(p);
                    axis[a] = 1.0;
                    crate::linalg::faer_ndarray::with_nested_parallel(|| pert_hessian_dir(&axis))
                })
                .collect();
            let mut pert_hdots = Vec::with_capacity(p);
            for result in results {
                match result? {
                    Some(h2) => pert_hdots.push(h2),
                    None => return Ok(Array2::zeros((p, p))),
                }
            }
            pert_hdots
        };
        self.perturbation_derivative_from_axis_matrices(pert_h, pert_hdots)
    }

    /// Batched-axes variant of [`Self::perturbation_derivative`]: the caller
    /// supplies the full all-axes object `{H²dot[δ,e_a]}_{a=0..p}` in ONE shot
    /// (e.g. via the row-kernel BLAS-3
    /// `second_directional_derivative_all_axes_dense_override`), avoiding the `p`
    /// independent full-data second-directional sweeps the per-axis closure runs.
    /// `None` ⇒ the family lacks the exact second derivative ⇒ zero drift, exactly
    /// as the per-axis path's first-`None` collapse. The reduction from the axis
    /// matrices onward is shared with (and bit-identical to) the per-axis path.
    pub(crate) fn perturbation_derivative_batched_axes(
        &self,
        pert_h: &Array2<f64>,
        pert_axis_matrices: Option<Vec<Array2<f64>>>,
    ) -> Result<Array2<f64>, String> {
        let p = self.p;
        let Some(pert_hdots) = pert_axis_matrices else {
            return Ok(Array2::zeros((p, p)));
        };
        if pert_hdots.len() != p {
            return Err(format!(
                "JeffreysHphiDriftBase::perturbation_derivative_batched_axes: got {} axis \
                 matrices, expected {p}",
                pert_hdots.len()
            ));
        }
        self.perturbation_derivative_from_axis_matrices(pert_h, pert_hdots)
    }

    /// Shared reduction core: given the reduced-base eigen-data and the already
    /// acquired per-axis second-directional matrices `{H²dot[δ,e_a]}`, assemble
    /// the gated curvature drift `D[gate·H_Φ_raw]`. Both
    /// [`Self::perturbation_derivative`] (per-axis closure) and
    /// [`Self::perturbation_derivative_batched_axes`] (batched all-axes) feed this,
    /// so the two paths are bit-identical from the axis matrices onward.
    fn perturbation_derivative_from_axis_matrices(
        &self,
        pert_h: &Array2<f64>,
        pert_hdots: Vec<Array2<f64>>,
    ) -> Result<Array2<f64>, String> {
        let p = self.p;
        if pert_h.nrows() != p || pert_h.ncols() != p {
            return Err(format!(
                "JeffreysHphiDriftBase::perturbation_derivative: pert_h shape {}x{} != {p}x{p}",
                pert_h.nrows(),
                pert_h.ncols()
            ));
        }
        let m = self.m;
        let z_j = self.z_j.view();
        let evals = &self.evals;
        let evecs = &self.evecs;
        let floor = self.floor;
        let gate_weight = self.gate_weight;
        let psi = &self.psi;
        let floor_in_relative_regime = self.floor_in_relative_regime;
        let idx_min = self.idx_min;
        let idx_max = self.idx_max;
        let lambda_min = evals[idx_min];
        let lambda_max = evals[idx_max];
        // Ḋ = Z_Jᵀ (∂H_joint) Z_J, the reduced perturbation of the reduced information.
        let dbar_raw = z_j.t().dot(&pert_h.dot(&z_j)); // m x m
        let mut dbar = Array2::<f64>::zeros((m, m));
        for i in 0..m {
            for j in 0..m {
                dbar[[i, j]] = 0.5 * (dbar_raw[[i, j]] + dbar_raw[[j, i]]);
            }
        }

        // EXACT DERIVATIVE OF THE DIVIDED-DIFFERENCE CURVATURE (value↔drift
        // consistency, gam#979). The value path builds
        //   `H_Φ_raw[a,b] = −½ Σ_ij Ψ_ij (Ṽ_a)_ij (Ṽ_b)_ij`,
        // with `Ṽ_k = Vᵀ D_k V` the eigenbasis-rotated reduced derivatives and `Ψ`
        // the Daleckii–Krein divided differences of the floored signed inverse
        // (see `joint_jeffreys_term`). Under a perturbation that moves `H_id` by
        // `Ḋ` and each `D_a` by `∂D_a`, the exact first-order pieces are:
        //   * eigenvalue motion       λ̇_i = (Ḋ̃)_ii,            Ḋ̃ = Vᵀ Ḋ V,
        //   * eigenvector rotation    δV = V C,   C_ij = (Ḋ̃)_ij/(λ_j − λ_i) (i≠j),
        //   * rotated derivatives     δṼ_a = Vᵀ (∂D_a) V + Ṽ_a C − C Ṽ_a,
        //   * kernel motion           δΨ_ij from the chain rule on the divided
        //     difference `Ψ(λ_i, λ_j; floor)` — for separated pairs
        //       δΨ_ij = [(d'(λ_i) − Ψ_ij)·λ̇_i + (Ψ_ij − d'(λ_j))·λ̇_j]/(λ_i − λ_j)
        //               + (∂d_i/∂floor − ∂d_j/∂floor)/(λ_i − λ_j) · δfloor,
        //     and for confluent/diagonal entries `δΨ_ii = d''(λ_i)·λ̇_i` (the floor
        //     branch has `d'' = 0`, and `d'` is floor-independent in both branches
        //     so no diagonal floor-motion arises). The floor-motion term fires only
        //     in the active RELATIVE regime, mirroring the value path's
        //     floor-response: `δfloor = REL · λ̇_{idx_max}`, `∂d_i/∂floor = −1/floor²`
        //     on below-floor entries (0 otherwise).
        // Then
        //   δH_Φ_raw[a,b] = −½ Σ_ij [δΨ_ij (Ṽ_a)_ij (Ṽ_b)_ij
        //                            + Ψ_ij ((δṼ_a)_ij (Ṽ_b)_ij + (Ṽ_a)_ij (δṼ_b)_ij)].
        let dbar_red = evecs.t().dot(&dbar).dot(evecs); // Vᵀ Ḋ V (m × m)
        let dfloor = if floor_in_relative_regime {
            REDUCED_INFO_RELATIVE_FLOOR * dbar_red[[idx_max, idx_max]]
        } else {
            0.0
        };
        // Eigenvector-rotation generator C (skew: C_ij = (Ḋ̃)_ij/(λ_j − λ_i), 0 on
        // the diagonal and on confluent pairs, where first-order rotation within
        // the degenerate subspace cancels out of the symmetric contraction).
        let mut rotation = Array2::<f64>::zeros((m, m));
        // Kernel motion δΨ.
        let mut dpsi = Array2::<f64>::zeros((m, m));
        for i in 0..m {
            for j in 0..m {
                let denom = evals[j] - evals[i];
                if denom.abs() > REDUCED_INFO_ABSOLUTE_FLOOR {
                    rotation[[i, j]] = dbar_red[[i, j]] / denom;
                }
                let gap = evals[i] - evals[j];
                if gap.abs() > REDUCED_INFO_ABSOLUTE_FLOOR {
                    let dp_i = floored_inverse_prime(evals[i], floor);
                    let dp_j = floored_inverse_prime(evals[j], floor);
                    let lam_dot_i = dbar_red[[i, i]];
                    let lam_dot_j = dbar_red[[j, j]];
                    dpsi[[i, j]] =
                        ((dp_i - psi[[i, j]]) * lam_dot_i + (psi[[i, j]] - dp_j) * lam_dot_j) / gap;
                    if dfloor != 0.0 {
                        dpsi[[i, j]] += (floored_inverse_floor_sensitivity(evals[i], floor)
                            - floored_inverse_floor_sensitivity(evals[j], floor))
                            / gap
                            * dfloor;
                    }
                } else {
                    // Confluent/diagonal: Ψ = d'(λ), so δΨ = d''(λ)·λ̇ with the
                    // averaged eigenvalue motion of the (near-)tied pair, plus the
                    // floor motion of d' (nonzero only on the saturating branches).
                    dpsi[[i, j]] = floored_inverse_second(evals[i], floor)
                        * 0.5
                        * (dbar_red[[i, i]] + dbar_red[[j, j]]);
                    if dfloor != 0.0 {
                        dpsi[[i, j]] +=
                            floored_inverse_prime_floor_sensitivity(evals[i], floor) * dfloor;
                    }
                }
            }
        }

        // Per canonical axis e_a: the rotated base Ṽ_a (REUSED from the prepared
        // base — its `Hdot[e_a]` row-stream is NOT recomputed per direction) and its
        // Ψ-weighted partner, plus the perturbed rotation δṼ_a and the δΨ/Ψ-weighted
        // derivative rows, flattened so the final contraction is a pair of m·m inner
        // products per (a,b). `a_rows`/`aw_rows` come from the base; only `da_rows`
        // (vec(δṼ_a)) and `dw_rows` (vec(δΨ∘Ṽ_a + Ψ∘δṼ_a)) are direction-dependent.
        let a_rows = &self.a_rows; // vec(Ṽ_a)
        let aw_rows = &self.aw_rows; // vec(Ψ ∘ Ṽ_a)
        let mut da_rows = Array2::<f64>::zeros((p, m * m)); // vec(δṼ_a)
        let mut dw_rows = Array2::<f64>::zeros((p, m * m)); // vec(δΨ ∘ Ṽ_a + Ψ ∘ δṼ_a)

        // The per-axis second-directional matrices `{H²dot[δ,e_a]}` were acquired
        // by the caller (per-axis closure sweep or batched all-axes override) and
        // validated for count here; each must be the full `p×p` shape.
        for (a, pert_hdot_a) in pert_hdots.iter().enumerate() {
            if pert_hdot_a.nrows() != p || pert_hdot_a.ncols() != p {
                return Err(format!(
                    "JeffreysHphiDriftBase::perturbation_derivative: ∂Hdot[e_{a}] shape {}x{} != {p}x{p}",
                    pert_hdot_a.nrows(),
                    pert_hdot_a.ncols()
                ));
            }
        }
        // Reconstruct Ṽ_a (m × m) from the flattened base row for the rotation terms.
        let mut a_a = Array2::<f64>::zeros((m, m));
        for (a, pert_hdot_a) in pert_hdots.into_iter().enumerate() {
            {
                let mut col = 0usize;
                for i in 0..m {
                    for j in 0..m {
                        a_a[[i, j]] = a_rows[[a, col]];
                        col += 1;
                    }
                }
            }

            let d_a_pert_raw = z_j.t().dot(&pert_hdot_a.dot(&z_j)); // Z_Jᵀ (∂Hdot[e_a]) Z_J
            let mut d_a_pert = Array2::<f64>::zeros((m, m));
            for i in 0..m {
                for j in 0..m {
                    d_a_pert[[i, j]] = 0.5 * (d_a_pert_raw[[i, j]] + d_a_pert_raw[[j, i]]);
                }
            }

            // δṼ_a = Vᵀ (∂D_a) V + Ṽ_a C − C Ṽ_a.
            let da_a =
                evecs.t().dot(&d_a_pert).dot(evecs) + &a_a.dot(&rotation) - &rotation.dot(&a_a);

            let mut col = 0usize;
            for i in 0..m {
                for j in 0..m {
                    da_rows[[a, col]] = da_a[[i, j]];
                    dw_rows[[a, col]] = dpsi[[i, j]] * a_a[[i, j]] + psi[[i, j]] * da_a[[i, j]];
                    col += 1;
                }
            }
        }

        // δH_Φ_raw[a,b] = −½ (⟨vec(δΨ∘Ṽ_a + Ψ∘δṼ_a), vec(Ṽ_b)⟩ + ⟨vec(Ψ∘Ṽ_a), vec(δṼ_b)⟩).
        // Mathematically symmetric in (a, b); assembled symmetrically for exactness.
        let mut out = Array2::<f64>::zeros((p, p));
        for a in 0..p {
            for b in a..p {
                let mut acc = 0.0;
                for col in 0..(m * m) {
                    acc += dw_rows[[a, col]] * a_rows[[b, col]]
                        + aw_rows[[a, col]] * da_rows[[b, col]];
                }
                let value = -0.5 * acc;
                out[[a, b]] = value;
                out[[b, a]] = value;
            }
        }

        // GATE DERIVATIVE (value↔gradient consistency, gam#854). `H_Φ = G(λ_min,λ_max)·H_Φ_raw`,
        // and the gate moves with the perturbation through the reduced eigenvalues, so
        //   `D[G·H_Φ_raw] = (D G)·H_Φ_raw + G·(D H_Φ_raw)`,
        // `D G = G_λmin·δλ_min + G_λmax·δλ_max`, `δλ = vᵀ Ḋ v`. Identically zero on a
        // saturated gate, so fully-active / well-conditioned fits are byte-unchanged.
        let mut result = out * gate_weight;
        let (g_dlmin, g_dlmax) = conditioning_gate_weight_grad(lambda_min, lambda_max);
        if g_dlmin != 0.0 || g_dlmax != 0.0 {
            let extreme_perturbation = |idx: usize| -> f64 {
                let v = evecs.column(idx);
                v.dot(&dbar.dot(&v))
            };
            let d_gate =
                g_dlmin * extreme_perturbation(idx_min) + g_dlmax * extreme_perturbation(idx_max);
            if d_gate != 0.0 {
                // H_Φ_raw = −½ Σ_ij Ψ_ij (Ṽ_a)_ij (Ṽ_b)_ij, matching the value path.
                let hphi_raw = aw_rows.dot(&a_rows.t()).mapv(|x| -0.5 * x);
                result.scaled_add(d_gate, &hphi_raw);
            }
        }
        Ok(result)
    }
}

/// Shared core for the joint-Jeffreys curvature perturbation derivative
/// `D[gate·H_Φ_raw]`, given a perturbation that acts on `H_joint` through `pert_h`
/// (a `p×p` matrix `∂H_joint`) and on each axis derivative `D_a` through
/// `pert_hessian_dir(e_a)` (a `p×p` matrix `∂Hdot[e_a]`). `base_hessian_dir(e_a)`
/// supplies the unperturbed `Hdot[e_a]` that forms the base `M_a = K D_a`.
///
/// Both the mode-response drift (perturbation `Hdot[δ]`, `H²dot[δ,e_a]`) and the
/// explicit ρ-derivative (perturbation `∂_ρ H_joint`, `∂_ρ Hdot[e_a]`) are
/// instances. It reproduces the value path's reduced information, smooth
/// conditioning gate, and floored pseudo-inverse, and differentiates the gate
/// (`(D gate)·H_Φ_raw`) so the result is consistent with the gated `H_Φ` the
/// objective folds into `½ log|H+S_λ+H_Φ|`. With `K = H_id⁻¹` the floored
/// pseudo-inverse, `M_a = K D_a`, `H_Φ_raw[a,b] = ½⟨vec M_a, vec M_b⟩`:
///   * `δK = −K Ḋ K`, `Ḋ = Z_Jᵀ (∂H_joint) Z_J` (exact on the unfloored spectrum,
///     matching the value path; the floored-spectrum divided-difference correction
///     is the same residual the value/gradient pair already carries),
///   * `δM_a = −K Ḋ M_a + K (Z_Jᵀ ∂Hdot[e_a] Z_J)`,
///   * `δH_Φ_raw[a,b] = ½(⟨δM_a, M_b⟩ + ⟨M_a, δM_b⟩)`.
///
/// This is the single-perturbation entry point used by the explicit-ρ-derivative
/// path; it prepares a [`JeffreysHphiDriftBase`] and applies one perturbation.
/// The batched mode-response drift instead prepares the base ONCE and applies it
/// across every direction (see `custom_family_outer_jeffreys_hphi_drift_batched`).
pub(crate) fn joint_jeffreys_hphi_perturbation_derivative<BaseFn, PertFn>(
    h_joint: ArrayView2<'_, f64>,
    z_j: ArrayView2<'_, f64>,
    base_hessian_dir: BaseFn,
    pert_h: &Array2<f64>,
    pert_hessian_dir: PertFn,
) -> Result<Array2<f64>, String>
where
    BaseFn: Fn(&Array1<f64>) -> Result<Option<Array2<f64>>, String> + Sync,
    PertFn: Fn(&Array1<f64>) -> Result<Option<Array2<f64>>, String> + Sync,
{
    let p = h_joint.nrows();
    match JeffreysHphiDriftBase::prepare(h_joint, z_j, base_hessian_dir)? {
        // Gated out / no exact first derivative ⇒ H_Φ ≡ 0 in a neighborhood ⇒
        // its derivative vanishes (byte-identical to the released zero return).
        None => Ok(Array2::zeros((p, p))),
        Some(base) => base.perturbation_derivative(pert_h, pert_hessian_dir),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::array;

    /// Test-only analytic oracle: the per-direction mode-response drift
    /// `D_β H_Φ[δ]` of the Tier-B Jeffreys curvature surrogate. Production now
    /// computes this via the batched H_Φ drift path; this standalone reference
    /// (one `Hdot[δ]` then the perturbation core) backs the FD-vs-analytic check
    /// below. Lives inside `mod tests` because it has no production caller.
    fn joint_jeffreys_hphi_directional_derivative<DirFn, Dir2Fn>(
        h_joint: ArrayView2<'_, f64>,
        z_j: ArrayView2<'_, f64>,
        delta: &Array1<f64>,
        hessian_dir: DirFn,
        hessian_second_dir: Dir2Fn,
    ) -> Result<Array2<f64>, String>
    where
        DirFn: Fn(&Array1<f64>) -> Result<Option<Array2<f64>>, String> + Sync,
        Dir2Fn: Fn(&Array1<f64>, &Array1<f64>) -> Result<Option<Array2<f64>>, String> + Sync,
    {
        let p = h_joint.nrows();
        if delta.len() != p {
            return Err(format!(
                "joint_jeffreys_hphi_directional_derivative: delta has {} entries, expected {p}",
                delta.len()
            ));
        }
        let pert_h = match hessian_dir(delta)? {
            Some(hd) => hd,
            None => return Ok(Array2::zeros((p, p))),
        };
        if pert_h.nrows() != p || pert_h.ncols() != p {
            return Err(format!(
                "joint_jeffreys_hphi_directional_derivative: Hdot[δ] shape {}x{} != {p}x{p}",
                pert_h.nrows(),
                pert_h.ncols()
            ));
        }
        joint_jeffreys_hphi_perturbation_derivative(
            h_joint,
            z_j,
            |axis| hessian_dir(axis),
            &pert_h,
            |axis| hessian_second_dir(delta, axis),
        )
    }

    /// `joint_jeffreys_hphi_explicit_param_derivative` must equal the central
    /// finite difference of the value-path gated curvature `H_Φ` w.r.t. a scalar
    /// outer parameter `s`, on a synthetic β-frozen family where
    /// `H_joint(s) = H0 + s·P` and `Hdot[e_a](s) = G_a + s·Q_a` (so `∂_s H_joint = P`,
    /// `∂_s Hdot[e_a] = Q_a`). H0's smallest reduced eigenvalue sits in the ABSOLUTE
    /// gate transition band (exercising the gate derivative) with no floored
    /// eigenvalue, the regime of the gam#854 tension-axis miss.
    #[test]
    pub(crate) fn explicit_param_derivative_matches_finite_difference() {
        let p = 4usize;
        let z = Array2::<f64>::eye(p);
        let h0 = array![
            [30.0, 1.0, 0.5, 0.2],
            [1.0, 12.0, 0.3, 0.1],
            [0.5, 0.3, 5.0, 0.4],
            [0.2, 0.1, 0.4, 1.5],
        ];
        let pmat = array![
            [2.0, 0.3, 0.1, 0.05],
            [0.3, 1.5, 0.2, 0.1],
            [0.1, 0.2, 1.0, 0.15],
            [0.05, 0.1, 0.15, 0.7],
        ];
        let make_sym = |seed: f64| -> Array2<f64> {
            let mut a = Array2::<f64>::zeros((p, p));
            for i in 0..p {
                for j in 0..p {
                    a[[i, j]] = (seed + 0.37 * (i as f64) - 0.19 * (j as f64)).sin()
                        + 0.5 * ((i + j) as f64 * seed).cos();
                }
            }
            let at = a.t().to_owned();
            (&a + &at).mapv(|v| 0.5 * v)
        };
        let g: Vec<Array2<f64>> = (0..p).map(|a| make_sym(1.0 + a as f64)).collect();
        let q: Vec<Array2<f64>> = (0..p).map(|a| make_sym(7.0 + 2.0 * a as f64)).collect();
        let axis_index = |axis: &Array1<f64>| -> usize {
            axis.iter().position(|&x| x != 0.0).expect("one-hot axis")
        };

        let hphi_at = |s: f64| -> Array2<f64> {
            let h = &h0 + &pmat.mapv(|v| s * v);
            joint_jeffreys_term(h.view(), z.view(), |axis: &Array1<f64>| {
                let a = axis_index(axis);
                Ok(Some(&g[a] + &q[a].mapv(|v| s * v)))
            })
            .expect("value-path H_Φ")
            .2
        };

        let s0 = 0.0_f64;
        let hh = 1e-5;
        let fd = (&hphi_at(s0 + hh) - &hphi_at(s0 - hh)).mapv(|v| v / (2.0 * hh));

        let h_s0 = &h0 + &pmat.mapv(|v| s0 * v);
        let analytic = joint_jeffreys_hphi_explicit_param_derivative(
            h_s0.view(),
            z.view(),
            &pmat,
            |axis: &Array1<f64>| {
                let a = axis_index(axis);
                Ok(Some(&g[a] + &q[a].mapv(|v| s0 * v)))
            },
            |axis: &Array1<f64>| {
                let a = axis_index(axis);
                Ok(Some(q[a].clone()))
            },
        )
        .expect("explicit ∂_s H_Φ");

        let mut max_err = 0.0_f64;
        for i in 0..p {
            for j in 0..p {
                max_err = max_err.max((analytic[[i, j]] - fd[[i, j]]).abs());
            }
        }
        assert!(
            max_err < 1e-5,
            "explicit ∂_s H_Φ mismatch vs FD: max_err={max_err}"
        );
    }

    /// `joint_jeffreys_hphi_directional_derivative` (the mode-response drift
    /// `D_β H_Φ[δ]`) must equal the central finite difference of the value-path gated
    /// curvature `H_Φ` along the coefficient direction `δ`, INCLUDING the below-floor
    /// regime — the regime the earlier `δK = −K Ḋ K` got wrong and that produced the
    /// gam#808 frozen-|g| survival-marginal-slope outer stall.
    ///
    /// Setup: a β-linear joint Hessian `H(β) = H0 + Σ_a β_a A_a` (so `Hdot[e_a] = A_a`
    /// constant and the joint second directional derivative is identically zero). We
    /// FD `H_Φ(t)` along `β = t·δ` with `δ = e_0`, so `pert_h = Hdot[δ] = A_0` and
    /// `pert_hessian_dir(δ, e_a) = 0`. `H0` is engineered with one reduced eigenvalue
    /// BELOW the relative floor (a near-separating direction), exactly where the
    /// floored pseudo-inverse and its moving floor matter.
    #[test]
    pub(crate) fn perturbation_derivative_matches_finite_difference_below_floor() {
        let p = 3usize;
        let z = Array2::<f64>::eye(p);
        // H0: large dominant curvature (λ_max ≈ 5e8 ⇒ relative floor = 1e-10·λ_max
        // ≈ 5e-2) and one tiny eigenvalue (≈1e-4) comfortably BELOW that floor — the
        // separating direction. The margin (floor ≈ 5e-2 vs λ ≈ 1e-4) keeps the small
        // eigenvalue below the floor across the whole FD window, so the floored branch
        // is exercised cleanly without crossing the floor knot.
        let h0 = array![
            [5.0e8, 2.0e3, 1.0e2],
            [2.0e3, 4.0e8, 5.0e1],
            [1.0e2, 5.0e1, 1.0e-4],
        ];
        let make_sym = |seed: f64| -> Array2<f64> {
            let mut a = Array2::<f64>::zeros((p, p));
            for i in 0..p {
                for j in 0..p {
                    a[[i, j]] = (seed + 0.41 * (i as f64) - 0.23 * (j as f64)).sin()
                        + 0.6 * ((i + j) as f64 * seed).cos();
                }
            }
            let at = a.t().to_owned();
            (&a + &at).mapv(|v| 0.5 * v)
        };
        // A_a = ∂H/∂β_a, the per-axis first directional derivative (constant in β).
        let a_mats: Vec<Array2<f64>> = (0..p).map(|a| make_sym(2.3 + 1.7 * a as f64)).collect();
        let axis_index = |axis: &Array1<f64>| -> usize {
            axis.iter().position(|&x| x != 0.0).expect("one-hot axis")
        };
        // β = t·δ with δ = e_0 ⇒ H(t) = H0 + t·A_0, Hdot[e_a] ≡ A_a.
        let hphi_at = |t: f64| -> Array2<f64> {
            let h = &h0 + &a_mats[0].mapv(|v| t * v);
            joint_jeffreys_term(h.view(), z.view(), |axis: &Array1<f64>| {
                Ok(Some(a_mats[axis_index(axis)].clone()))
            })
            .expect("value-path H_Φ")
            .2
        };

        let hh = 1e-5;
        let fd = (&hphi_at(hh) - &hphi_at(-hh)).mapv(|v| v / (2.0 * hh));

        let mut delta = Array1::<f64>::zeros(p);
        delta[0] = 1.0;
        let analytic = joint_jeffreys_hphi_directional_derivative(
            h0.view(),
            z.view(),
            &delta,
            // Hdot[d] = Σ_a d_a A_a (linear in the direction).
            |d: &Array1<f64>| {
                let mut acc = Array2::<f64>::zeros((p, p));
                for a in 0..p {
                    if d[a] != 0.0 {
                        acc.scaled_add(d[a], &a_mats[a]);
                    }
                }
                Ok(Some(acc))
            },
            // H²dot[u, v] = 0 (H is β-linear).
            |_u: &Array1<f64>, _v: &Array1<f64>| Ok(Some(Array2::<f64>::zeros((p, p)))),
        )
        .expect("mode-response drift D_β H_Φ[δ]");

        // Relative tolerance against the FD magnitude: the below-floor entries carry
        // O(1/floor)≈1e10-scale curvature, so an absolute bound is meaningless.
        let mut max_rel = 0.0_f64;
        for i in 0..p {
            for j in 0..p {
                let scale = fd[[i, j]].abs().max(analytic[[i, j]].abs()).max(1.0);
                max_rel = max_rel.max((analytic[[i, j]] - fd[[i, j]]).abs() / scale);
            }
        }
        assert!(
            max_rel < 1e-4,
            "mode-response drift D_β H_Φ[δ] mismatch vs FD (below-floor): max_rel={max_rel}"
        );
    }

    /// Test-only convenience predicate: `true` when the smooth gate weight is exactly
    /// `0` (the term is fully skippable). Non-test code uses `conditioning_gate_weight`
    /// directly so the transition band stays continuous; the cheap matrix-free
    /// pre-check certifies a full skip by clearing the UPPER (`*_CLEAR`) knots.
    pub(crate) fn conditioning_gate_skips(lambda_min: f64, lambda_max: f64) -> bool {
        conditioning_gate_weight(lambda_min, lambda_max) == 0.0
    }

    #[test]
    pub(crate) fn full_span_is_identity_regardless_of_penalty() {
        // The principled cure: Z_J is the FULL identifiable span (the entire
        // reduced block), i.e. the identity, irrespective of the penalty's null
        // space. Jeffreys is self-limiting, so this does not bias identified
        // directions; it only bounds near-separating ones.
        for s in [
            Array2::<f64>::zeros((3, 3)), // pure parametric
            {
                let mut s = Array2::<f64>::zeros((3, 3));
                s[[2, 2]] = 5.0; // rank-deficient (ker dim 2)
                s
            },
            Array2::<f64>::eye(4) * 2.0, // full-rank penalty
        ] {
            let p = s.nrows();
            let z = jeffreys_subspace_from_penalty(s.view()).unwrap();
            assert_eq!(z.span_dim(), p, "full span must equal the block dimension");
            assert_eq!(z.columns.nrows(), p);
            // Identity ⇒ orthonormal columns spanning the whole space.
            let gram = z.columns.t().dot(&z.columns);
            for i in 0..p {
                for j in 0..p {
                    let expected = if i == j { 1.0 } else { 0.0 };
                    assert!((gram[[i, j]] - expected).abs() < 1e-12);
                }
            }
        }
    }

    #[test]
    pub(crate) fn empty_block_yields_empty_span() {
        let s = Array2::<f64>::zeros((0, 0));
        let z = jeffreys_subspace_from_penalty(s.view()).unwrap();
        assert_eq!(z.span_dim(), 0);
    }

    #[test]
    pub(crate) fn joint_jeffreys_term_matches_finite_difference_gradient() {
        // A 2x2 quadratic-form Hessian whose log-determinant has a known
        // gradient. The SECOND direction is scaled by `ill` so the reduced
        // information is ILL-conditioned (`λ_min/λ_max ≈ 8.6e-10`, below the
        // conditioning gate) — this exercises the active Jeffreys path rather
        // than the gate, while both eigenvalues stay comfortably above the
        // floored ridge so `Φ` and `grad` are the exact log-det quantities.
        // H(beta) = diag(exp(beta0), ill*(1+beta1^2)), Z_J = I.
        let p = 2usize;
        let ill = 1e-9_f64;
        let z = Array2::<f64>::eye(p);
        let h_at = |b: &Array1<f64>| -> Array2<f64> {
            let mut h = Array2::<f64>::zeros((p, p));
            h[[0, 0]] = b[0].exp();
            h[[1, 1]] = ill * (1.0 + b[1] * b[1]);
            h
        };
        // Hdot[d] = d/d eps H(beta + eps d): diag(exp(b0) d0, ill*2 b1 d1).
        let beta: Array1<f64> = array![0.3, -0.4];
        let hdir = |d: &Array1<f64>| -> Result<Option<Array2<f64>>, String> {
            let mut hd = Array2::<f64>::zeros((p, p));
            hd[[0, 0]] = beta[0].exp() * d[0];
            hd[[1, 1]] = ill * 2.0 * beta[1] * d[1];
            Ok(Some(hd))
        };
        let h = h_at(&beta);
        let (phi, grad, hphi) = joint_jeffreys_term(h.view(), z.view(), hdir).unwrap();
        // Phi = 1/2 log(exp(b0) * ill*(1 + b1^2)). The reduced-information
        // eigendecomposition resolves a spectrum spanning ~9 orders of magnitude
        // (λ_max ≈ 1.35, λ_min ≈ 1.16e-9), so the small eigenvalue — and hence Φ
        // — carries the eigensolver's relative round-off (~1e-7 abs on a Φ ≈ -10
        // log-volume). That is expected on a deliberately ill-conditioned design
        // exercising the active (un-gated) path; the load-bearing correctness
        // check is the gradient FD below, which is insensitive to the constant
        // `ill` scale.
        let expected_phi = 0.5 * (beta[0].exp() * ill * (1.0 + beta[1] * beta[1])).ln();
        assert!(
            (phi - expected_phi).abs() < 1e-6,
            "phi {phi} vs {expected_phi}"
        );
        // Finite-difference the gradient. Note ∂/∂β of log|H| is scale-free in
        // the constant `ill` factor (it differentiates the log), so the gradient
        // matches the un-scaled form exactly.
        let eps = 1e-6;
        for k in 0..p {
            let mut bp = beta.clone();
            let mut bm = beta.clone();
            bp[k] += eps;
            bm[k] -= eps;
            let hp = h_at(&bp);
            let hm = h_at(&bm);
            let phi_p = 0.5 * (hp[[0, 0]] * hp[[1, 1]]).ln();
            let phi_m = 0.5 * (hm[[0, 0]] * hm[[1, 1]]).ln();
            let fd = (phi_p - phi_m) / (2.0 * eps);
            assert!(
                (grad[k] - fd).abs() < 1e-5,
                "grad[{k}] {} vs fd {fd}",
                grad[k]
            );
        }
        // H_Phi is symmetric PSD.
        for a in 0..p {
            for b in 0..p {
                assert!((hphi[[a, b]] - hphi[[b, a]]).abs() < 1e-12);
            }
        }
        let (evals, _) = hphi.eigh(Side::Lower).unwrap();
        for e in evals.iter() {
            assert!(*e >= -1e-10, "H_Phi must be PSD, got eigenvalue {e}");
        }
    }

    #[test]
    pub(crate) fn joint_jeffreys_term_value_gradient_consistent_below_floor() {
        // Regression for the bernoulli-MS outer-non-convergence stall
        // (gam#787/#785): a separating direction whose reduced-information
        // eigenvalue sits BELOW the floored ridge. The released code computed the
        // value as the CONSTANT `½ ln(floor)` there (derivative 0) while the
        // gradient used the floored inverse `½ (1/floor) ∂λ/∂β` (derivative
        // nonzero), so ∇Φ ≠ d/dβ Φ exactly where Firth arms. The inner KKT
        // residual then floored at that mismatch and the joint-Newton could never
        // certify. The existing above-floor FD test could not catch this (its
        // λ_min stays above the floor). Here the second eigenvalue is genuinely
        // below the floor, so the FD MUST match the analytic gradient only with
        // the C¹ linear continuation of the value below the floor.
        let p = 2usize;
        // λ_max ≈ exp(0.3) ≈ 1.35 ⇒ floor = 1e-10·λ_max ≈ 1.35e-10. With
        // ill = 1e-12 the second eigenvalue λ_min ≈ 1.16e-12 < floor.
        let ill = 1e-12_f64;
        let z = Array2::<f64>::eye(p);
        let h_at = |b: &Array1<f64>| -> Array2<f64> {
            let mut h = Array2::<f64>::zeros((p, p));
            h[[0, 0]] = b[0].exp();
            h[[1, 1]] = ill * (1.0 + b[1] * b[1]);
            h
        };
        let beta: Array1<f64> = array![0.3, -0.4];
        let hdir = |d: &Array1<f64>| -> Result<Option<Array2<f64>>, String> {
            let mut hd = Array2::<f64>::zeros((p, p));
            hd[[0, 0]] = beta[0].exp() * d[0];
            hd[[1, 1]] = ill * 2.0 * beta[1] * d[1];
            Ok(Some(hd))
        };
        let h = h_at(&beta);
        let (_phi, grad, _hphi) = joint_jeffreys_term(h.view(), z.view(), hdir).unwrap();
        // Re-derive the floor exactly as the term does, and finite-difference the
        // value the term ACTUALLY accumulates (C¹ floored-inverse antiderivative):
        //   g(λ) = ln(λ)                    for λ ≥ floor,
        //   g(λ) = λ/floor + ln(floor) − 1  for λ < floor.
        let value_at = |b: &Array1<f64>| -> f64 {
            let hh = h_at(b);
            let lam0 = hh[[0, 0]];
            let lam1 = hh[[1, 1]];
            let lambda_max = lam0.max(lam1);
            let floor = (1e-10_f64 * lambda_max).max(1e-12_f64);
            let g = |lam: f64| -> f64 {
                if lam >= floor {
                    lam.ln()
                } else {
                    lam / floor + floor.ln() - 1.0
                }
            };
            0.5 * (g(lam0) + g(lam1))
        };
        let eps = 1e-7;
        for k in 0..p {
            let mut bp = beta.clone();
            let mut bm = beta.clone();
            bp[k] += eps;
            bm[k] -= eps;
            let fd = (value_at(&bp) - value_at(&bm)) / (2.0 * eps);
            assert!(
                (grad[k] - fd).abs() <= 1e-5 * (1.0 + fd.abs()),
                "below-floor grad[{k}] {} vs fd {fd}; the Jeffreys value must be the \
                 exact antiderivative of the floored-inverse gradient",
                grad[k]
            );
        }
    }

    #[test]
    pub(crate) fn joint_jeffreys_term_indefinite_value_gradient_consistent() {
        // Regression for the survival clustered-PC marginal-slope inner-solve
        // crawl (gam#814). The reduced OBSERVED information `H_id = Z_Jᵀ H Z_J` is
        // NOT PSD away from the mode for a non-canonical link, so it carries a
        // MODERATE NEGATIVE eigenvalue (|λ| ≫ floor). The released code floored on
        // the SIGNED eigenvalue (`1/max(λ, floor)`), pinning that moderate negative
        // to `+1/floor ≈ 1.7e6` — a phantom Firth score that no Newton step could
        // satisfy. The fix floors on `|λ|` and keeps the sign of the inverse
        // (`1/λ`), with the value `½ ln|λ|`. This test exercises that branch: the
        // second eigenvalue is genuinely NEGATIVE and well above the floor in
        // magnitude, so the FD gradient must match the analytic gradient ONLY with
        // the signed `1/λ` inverse and the `½ ln|λ|` value. The existing PSD-only
        // FD test cannot catch a sign error here.
        //
        // H(beta) = diag(exp(beta0), -(1 + beta1^2)), Z_J = I. λ_min < 0 < 1 so the
        // conditioning gate fires fully (weight 1) and the active path runs.
        let p = 2usize;
        let z = Array2::<f64>::eye(p);
        let h_at = |b: &Array1<f64>| -> Array2<f64> {
            let mut h = Array2::<f64>::zeros((p, p));
            h[[0, 0]] = b[0].exp();
            h[[1, 1]] = -(1.0 + b[1] * b[1]);
            h
        };
        let beta: Array1<f64> = array![0.3, -0.4];
        // Hdot[d] = diag(exp(b0) d0, -2 b1 d1).
        let hdir = |d: &Array1<f64>| -> Result<Option<Array2<f64>>, String> {
            let mut hd = Array2::<f64>::zeros((p, p));
            hd[[0, 0]] = beta[0].exp() * d[0];
            hd[[1, 1]] = -2.0 * beta[1] * d[1];
            Ok(Some(hd))
        };
        let h = h_at(&beta);
        let (phi, grad, hphi) = joint_jeffreys_term(h.view(), z.view(), hdir).unwrap();
        // Sanity (the gam#814 guarantee, preserved by the gam#979 saturating
        // branch): the moderate negative direction must NOT carry a phantom
        // 1/floor-scale Firth score. With the original signed floor, |grad|
        // would be ~1/floor ≈ 1.7e9 here; with the saturating branch the
        // negative direction's slope is `floor/(floor − λ)² ≈ 0`.
        assert!(
            grad.iter().all(|g| g.abs() < 1e3),
            "indefinite direction must carry no phantom Firth score; grad={grad:?}"
        );
        // The saturating branch must also not REWARD deeper indefiniteness
        // (the gam#979 runaway): the gradient along the negative-curvature
        // coordinate is essentially zero, not the gam#814 signed `1/λ` that
        // pulled λ further negative.
        assert!(
            grad[1].abs() < 1e-6,
            "saturating branch must be flat on a moderate negative eigenvalue; grad[1]={}",
            grad[1]
        );
        // Φ = ½(ln λ0 + g_sat(λ1)) with the saturating continuation on λ1 < 0:
        // g_sat(λ) = ln(floor) − 1 + λ/(floor − λ). λ_max = e^{b0}, so
        // floor = REL · λ_max here (relative regime).
        let lam0 = beta[0].exp();
        let lam1 = -(1.0 + beta[1] * beta[1]);
        let floor = 1e-10_f64 * lam0;
        let g_sat = |lam: f64, floor: f64| -> f64 {
            if lam >= floor {
                lam.ln()
            } else if lam >= 0.0 {
                lam / floor + floor.ln() - 1.0
            } else {
                floor.ln() - 1.0 + lam / (floor - lam)
            }
        };
        let expected_phi = 0.5 * (lam0.ln() + g_sat(lam1, floor));
        assert!(
            (phi - expected_phi).abs() < 1e-9,
            "phi {phi} vs {expected_phi}"
        );
        // Finite-difference the value the term accumulates (the same three-branch
        // antiderivative, floor moving with λ_max) and compare to the analytic
        // gradient — value/gradient consistency on the mixed-sign spectrum.
        let value_at = |b: &Array1<f64>| -> f64 {
            let hh = h_at(b);
            let lam_max = hh[[0, 0]].max(0.0);
            let fl = (1e-10 * lam_max).max(1e-12);
            0.5 * (g_sat(hh[[0, 0]], fl) + g_sat(hh[[1, 1]], fl))
        };
        let eps = 1e-7;
        for k in 0..p {
            let mut bp = beta.clone();
            let mut bm = beta.clone();
            bp[k] += eps;
            bm[k] -= eps;
            let fd = (value_at(&bp) - value_at(&bm)) / (2.0 * eps);
            assert!(
                (grad[k] - fd).abs() <= 1e-5 * (1.0 + fd.abs()),
                "indefinite grad[{k}] {} vs fd {fd}; value/gradient must share the saturating g",
                grad[k]
            );
        }
        // The exact divided-difference H_Φ is symmetric; on a mixed-sign
        // spectrum it may be indefinite (that is the honest curvature of Φ —
        // the Moré–Sorensen step owns it), so only symmetry is asserted here.
        for a in 0..p {
            for b in 0..p {
                assert!((hphi[[a, b]] - hphi[[b, a]]).abs() < 1e-12);
            }
        }
    }

    #[test]
    pub(crate) fn conditioning_gate_skips_well_conditioned_information() {
        // A WELL-conditioned reduced information (`λ_min/λ_max = 0.5`, far above
        // the gate) must skip the Jeffreys term entirely: zero value, gradient
        // and curvature, so an easy fit pays no cost. The directional-derivative
        // closure here is deliberately NONZERO; the gate must short-circuit
        // before it would otherwise produce a nonzero gradient.
        let p = 2usize;
        let z = Array2::<f64>::eye(p);
        let mut h = Array2::<f64>::zeros((p, p));
        h[[0, 0]] = 200.0;
        h[[1, 1]] = 100.0; // λ_min=100 ≫ 16 (upper knot), ratio 0.5 ⇒ fully skipped
        let hdir = |d: &Array1<f64>| -> Result<Option<Array2<f64>>, String> {
            // Nonzero derivative; would yield a nonzero gradient if not gated.
            let mut hd = Array2::<f64>::zeros((p, p));
            hd[[0, 0]] = 3.0 * d[0];
            hd[[1, 1]] = 5.0 * d[1];
            Ok(Some(hd))
        };
        let (phi, grad, hphi) = joint_jeffreys_term(h.view(), z.view(), hdir).unwrap();
        assert_eq!(phi, 0.0, "well-conditioned ⇒ no Jeffreys value");
        assert!(
            grad.iter().all(|v| *v == 0.0),
            "well-conditioned ⇒ zero grad"
        );
        assert!(
            hphi.iter().all(|v| *v == 0.0),
            "well-conditioned ⇒ zero curvature"
        );
    }

    #[test]
    pub(crate) fn conditioning_gate_fires_only_below_threshold() {
        // Bracket the COMBINED relative+absolute gate. To be SKIPPED a fit must be
        // well-conditioned both relatively (ratio ≥ 1e-8) AND absolutely
        // (λ_min ≥ 1); if EITHER fails the term fires. This pins the "no cost on a
        // genuinely well-conditioned large-n fit, full term on a (relatively OR
        // absolutely) near-separating one" boundary.
        let p = 2usize;
        let z = Array2::<f64>::eye(p);
        let hdir = |d: &Array1<f64>| -> Result<Option<Array2<f64>>, String> {
            let mut hd = Array2::<f64>::zeros((p, p));
            hd[[0, 0]] = d[0];
            hd[[1, 1]] = d[1];
            Ok(Some(hd))
        };
        // λ_max = 1.0 (h[[0,0]]); λ_min = the closure argument (h[[1,1]]).
        let mk = |lmin: f64| {
            let mut h = Array2::<f64>::zeros((p, p));
            h[[0, 0]] = 1.0;
            h[[1, 1]] = lmin;
            h
        };
        // Genuinely well-conditioned (large-n): ratio 0.5 ≥ 1e-8 AND λ_min = 50 ≫ 1
        // ⇒ gated. NOTE the second arg of `mk` is λ_min while `h[[0,0]]` is fixed
        // at 1.0 in the closure above; we override it here to a large λ_max.
        let mut above = mk(50.0);
        above[[0, 0]] = 100.0;
        let (phi_a, grad_a, _) = joint_jeffreys_term(above.view(), z.view(), hdir).unwrap();
        assert_eq!(phi_a, 0.0);
        assert!(grad_a.iter().all(|v| *v == 0.0));
        // Relatively near-separating (ratio < 1e-8, λ_max = 1.0) ⇒ fires.
        let below_rel = mk(CONDITIONING_GATE_RELATIVE * 0.1);
        let (phi_r, _g, hphi_r) = joint_jeffreys_term(below_rel.view(), z.view(), hdir).unwrap();
        assert!(phi_r != 0.0, "relatively near-separating must fire");
        assert!(hphi_r.iter().any(|v| v.abs() > 0.0));
        // ABSOLUTELY near-separating at SMALL n: λ_max = 1.0, λ_min = 0.05 ⇒ ratio
        // 0.05 ≥ 1e-8 (the relative gate alone would WRONGLY skip), but λ_min < 1
        // ⇒ the n-aware ABSOLUTE gate fires the stabilising term. This is exactly
        // the FIX-C small-n admixture-cline regime the relative-only gate missed.
        let below_abs = mk(0.05);
        let (phi_b, _grad_b, hphi_b) =
            joint_jeffreys_term(below_abs.view(), z.view(), hdir).unwrap();
        assert!(
            phi_b != 0.0,
            "absolutely near-separating (small-n) must fire even though the relative ratio clears the gate",
        );
        assert!(
            hphi_b.iter().any(|v| v.abs() > 0.0),
            "absolute-gate firing must produce nonzero bounding curvature",
        );
    }

    #[test]
    pub(crate) fn conditioning_gate_predicate_relative_and_absolute() {
        // Unit coverage of the shared predicate's two-sided logic.
        // Well-conditioned (both gates pass) ⇒ skip.
        assert!(conditioning_gate_skips(50.0, 100.0));
        // Relatively ill-conditioned ⇒ do not skip.
        assert!(!conditioning_gate_skips(
            CONDITIONING_GATE_RELATIVE * 0.1,
            1.0
        ));
        // Absolutely near-separating at small n (ratio fine, λ_min < 1) ⇒ do not skip.
        assert!(!conditioning_gate_skips(0.05, 1.0));
        // SMOOTH boundary: λ_min at the lower (firing) knot is still fully active,
        // and anywhere inside the ramp band is only partially tapered — NOT a full
        // skip. A full skip requires clearing the UPPER (`*_CLEAR`) knot.
        assert!(!conditioning_gate_skips(
            CONDITIONING_GATE_ABSOLUTE,
            CONDITIONING_GATE_ABSOLUTE
        ));
        assert!(!conditioning_gate_skips(4.0, 100.0));
        // Comfortably identified (λ_min past the upper knot, fine ratio) ⇒ skip.
        assert!(conditioning_gate_skips(
            CONDITIONING_GATE_ABSOLUTE_CLEAR,
            CONDITIONING_GATE_ABSOLUTE_CLEAR
        ));
        // Non-positive / non-finite spectra ⇒ never skip (fully active).
        assert!(!conditioning_gate_skips(0.0, 0.0));
        assert!(!conditioning_gate_skips(f64::NAN, 100.0));
    }

    #[test]
    pub(crate) fn conditioning_gate_weight_is_continuous_and_monotone() {
        // The whole point of the smooth gate (#787): the weight is C⁰/C¹ across the
        // absolute transition band [1, 16], so the outer LAML objective does not
        // jump as β̂(ρ) carries λ_min across the boundary. Sweep λ_min upward with a
        // fixed large λ_max (relative sub-weight pinned to 0 throughout) and assert
        // the weight is 1 at/below the lower knot, 0 at/above the upper knot,
        // strictly decreasing inside, and never jumps by more than a small step.
        let lambda_max = 1.0e6; // ratio ≪ knots, so the absolute ramp dominates
        let w = |lmin: f64| conditioning_gate_weight(lmin, lambda_max);
        assert_eq!(w(CONDITIONING_GATE_ABSOLUTE), 1.0);
        assert_eq!(w(0.1), 1.0);
        assert_eq!(w(CONDITIONING_GATE_ABSOLUTE_CLEAR), 0.0);
        assert_eq!(w(100.0), 0.0);
        let mut prev = 1.0;
        let n = 200usize;
        for k in 0..=n {
            let lmin = CONDITIONING_GATE_ABSOLUTE
                + (CONDITIONING_GATE_ABSOLUTE_CLEAR - CONDITIONING_GATE_ABSOLUTE)
                    * (k as f64 / n as f64);
            let cur = w(lmin);
            assert!((0.0..=1.0).contains(&cur));
            assert!(cur <= prev + 1e-12, "weight must be non-increasing");
            assert!(
                (prev - cur).abs() < 0.1,
                "no large jumps across the smooth band (continuity)"
            );
            prev = cur;
        }
    }

    /// Desync-guard (gam#931) for the conditioning-gate value↔gradient pair.
    ///
    /// `conditioning_gate_weight` (value `G`) and `conditioning_gate_weight_grad`
    /// (its analytic partials `(∂G/∂λ_min, ∂G/∂λ_max)`) are written as two
    /// SEPARATE functions, each independently re-spelling the cubic `ramp_down`
    /// smoothstep and the `max(w_abs, w_rel)` branch selection. That is precisely
    /// the split value/derivative code path #931 exists to make non-drifting:
    /// the gradient is consumed live in the Jeffreys outer hypergradient
    /// (`gate_weight` mode-response term, gam#854), so a drift between these two
    /// would silently bias the analytic outer gradient against its own value
    /// exactly where the gate sits in its transition band.
    ///
    /// Central-difference `G` in both arguments and assert the analytic grad
    /// matches, sampling the absolute-active band, the relative-active band, and
    /// the saturated (locally-constant `G`) regimes — staying inside each branch
    /// to avoid the C¹ knots (the `max`-tie and ramp endpoints) where FD straddles
    /// a kink.
    #[test]
    pub(crate) fn conditioning_gate_weight_grad_matches_finite_difference() {
        // Each config picks (λ_min, λ_max) comfortably inside one branch:
        //  - absolute-active: λ_min in (1, 16), λ_max huge so log10(ratio) ≤ -6
        //    (w_rel = 0) ⇒ w_abs dominates and varies, ∂/∂λ_max = 0;
        //  - relative-active: λ_min above the absolute-clear knot (w_abs = 0) with
        //    log10(λ_min/λ_max) inside (-8, -6) ⇒ w_rel dominates and varies in
        //    BOTH arguments;
        //  - saturated: well inside a flat region ⇒ both partials are 0.
        let configs: [(f64, f64); 6] = [
            (8.0, 1.0e9),            // absolute band mid (w_rel = 0)
            (4.0, 1.0e9),            // absolute band lower-mid
            (12.0, 1.0e9),           // absolute band upper-mid
            (100.0, 100.0 / 1.0e-7), // relative band mid (w_abs = 0, ratio = 1e-7)
            (0.05, 1.0e9),           // saturated: w_abs = 1 (λ_min < 1), w_rel = 0
            (1.0e3, 1.0e3 / 1.0e-9), // saturated: ratio = 1e-9 < relative-clear ⇒ w_rel = 1
        ];
        for &(lmin, lmax) in &configs {
            let (g_dlmin, g_dlmax) = conditioning_gate_weight_grad(lmin, lmax);

            // Central difference in λ_min (λ_max fixed), relative step away from knots.
            let hmin = 1e-7 * lmin.abs().max(1e-3);
            let fd_dlmin = (conditioning_gate_weight(lmin + hmin, lmax)
                - conditioning_gate_weight(lmin - hmin, lmax))
                / (2.0 * hmin);
            assert!(
                (fd_dlmin - g_dlmin).abs() <= 1e-4 * g_dlmin.abs().max(1.0),
                "∂G/∂λ_min desync at (λ_min={lmin}, λ_max={lmax}): fd={fd_dlmin} analytic={g_dlmin}"
            );

            // Central difference in λ_max (λ_min fixed).
            let hmax = 1e-7 * lmax.abs().max(1e-3);
            let fd_dlmax = (conditioning_gate_weight(lmin, lmax + hmax)
                - conditioning_gate_weight(lmin, lmax - hmax))
                / (2.0 * hmax);
            assert!(
                (fd_dlmax - g_dlmax).abs() <= 1e-4 * g_dlmax.abs().max(1.0),
                "∂G/∂λ_max desync at (λ_min={lmin}, λ_max={lmax}): fd={fd_dlmax} analytic={g_dlmax}"
            );
        }
    }

    #[test]
    pub(crate) fn empty_span_yields_zero_term() {
        let h = Array2::<f64>::eye(3);
        let z = Array2::<f64>::zeros((3, 0));
        let hdir = |_d: &Array1<f64>| -> Result<Option<Array2<f64>>, String> {
            Ok(Some(Array2::<f64>::zeros((3, 3))))
        };
        let (phi, grad, hphi) = joint_jeffreys_term(h.view(), z.view(), hdir).unwrap();
        assert_eq!(phi, 0.0);
        assert!(grad.iter().all(|v| *v == 0.0));
        assert!(hphi.iter().all(|v| *v == 0.0));
    }

    // ---- cheap matrix-free conditioning pre-check ----

    /// Build a diagonal-matvec closure for a synthetic spectrum (the joint
    /// information on the full span is `H` itself, so a diagonal `H` exercises the
    /// Lanczos bound against a known `[λ_min, λ_max]`).
    pub(crate) fn diag_hv(
        diag: Vec<f64>,
    ) -> impl FnMut(&Array1<f64>) -> Result<Array1<f64>, String> {
        move |v: &Array1<f64>| {
            let mut out = Array1::<f64>::zeros(v.len());
            for (i, &d) in diag.iter().enumerate() {
                out[i] = d * v[i];
            }
            Ok(out)
        }
    }

    #[test]
    pub(crate) fn cheap_precheck_skips_clearly_well_conditioned_large_p() {
        // A wide (p ≥ threshold) well-conditioned spectrum: every eigenvalue in
        // [200, 250], so λ_min = 200 clears the 8x margin on the smooth absolute
        // clear knot (16) and the ratio 0.8 clears the relative margin. The
        // conservative Lanczos bounds must still clear both gates ⇒ skippable.
        // This is the common large-p fast path the pre-check exists to make
        // matrix-free-free of any dense formation.
        let p = 200usize;
        let mut diag = vec![220.0; p];
        diag[0] = 200.0; // λ_min
        diag[1] = 250.0; // λ_max
        let skippable = jeffreys_term_skippable_via_matvec(diag_hv(diag), p).unwrap();
        assert!(
            skippable,
            "clearly well-conditioned wide fit must be skippable"
        );
    }

    #[test]
    pub(crate) fn cheap_precheck_does_not_skip_near_separating() {
        // One near-zero eigenvalue (λ_min = 1e-3) below the absolute gate (1.0):
        // the term is genuinely needed here. The conservative bounds must NOT
        // certify skippable, so the caller falls through to the exact formation.
        let p = 200usize;
        let mut diag = vec![50.0; p];
        diag[7] = 1e-3; // near-separating direction
        let skippable = jeffreys_term_skippable_via_matvec(diag_hv(diag), p).unwrap();
        assert!(
            !skippable,
            "a near-separating direction must NOT be skipped (term is needed)"
        );
    }

    #[test]
    pub(crate) fn cheap_precheck_does_not_skip_below_size_threshold() {
        // Small p: even a perfectly-conditioned spectrum is never pre-checked
        // (the exact dense eigh is already cheap there). Guarantees the small-p
        // BMS-style fits keep running the exact dense path unchanged.
        let p = CHEAP_CONDITIONING_PRECHECK_MIN_DIM - 1;
        let diag = vec![100.0; p];
        let skippable = jeffreys_term_skippable_via_matvec(diag_hv(diag), p).unwrap();
        assert!(
            !skippable,
            "below the size threshold the pre-check never skips"
        );
    }

    #[test]
    pub(crate) fn cheap_precheck_does_not_skip_marginal_absolute() {
        // Absolutely marginal: λ_min = 2.0 clears the bare absolute gate (1.0) but
        // NOT the 8× safety margin (needs ≥ 8). The conservative pre-check must
        // refuse to skip even though the EXACT gate would skip — the asymmetric
        // safety bias (false fall-through is cheap, false skip is fatal).
        let p = 200usize;
        let mut diag = vec![50.0; p];
        diag[3] = 2.0;
        let skippable = jeffreys_term_skippable_via_matvec(diag_hv(diag), p).unwrap();
        assert!(
            !skippable,
            "λ_min within the 8× absolute margin must conservatively fall through"
        );
    }

    #[test]
    pub(crate) fn cheap_precheck_skip_implies_exact_gate_skips() {
        // CONSISTENCY: wherever the cheap pre-check declares skippable, the EXACT
        // conditioning gate on the same spectrum must also skip (return the zero
        // term). This is the byte-identical-on-skip guarantee. Sweep a range of
        // well-conditioned spectra and assert the implication.
        let p = 150usize;
        let z = Array2::<f64>::eye(p);
        let hdir = |_d: &Array1<f64>| -> Result<Option<Array2<f64>>, String> {
            Ok(Some(Array2::<f64>::zeros((p, p))))
        };
        for &lmin in &[10.0_f64, 25.0, 80.0, 200.0] {
            let mut diag = vec![lmin * 4.0; p];
            diag[0] = lmin;
            let cheap_skip = jeffreys_term_skippable_via_matvec(diag_hv(diag.clone()), p).unwrap();
            if cheap_skip {
                // The exact path on the same dense H must produce the zero term.
                let mut h = Array2::<f64>::zeros((p, p));
                for (i, &d) in diag.iter().enumerate() {
                    h[[i, i]] = d;
                }
                let (phi, grad, hphi) = joint_jeffreys_term(h.view(), z.view(), hdir).unwrap();
                assert_eq!(
                    phi, 0.0,
                    "cheap-skip ⇒ exact phi must be zero (byte-identical)"
                );
                assert!(grad.iter().all(|v| *v == 0.0));
                assert!(hphi.iter().all(|v| *v == 0.0));
            }
        }
    }

    #[test]
    pub(crate) fn cheap_precheck_bails_on_nonfinite_matvec() {
        // A matvec that returns non-finite values cannot certify conditioning ⇒
        // the pre-check must return false (never skip on an unresolved estimate).
        let p = 200usize;
        let hv = |v: &Array1<f64>| -> Result<Array1<f64>, String> {
            Ok(Array1::from_elem(v.len(), f64::NAN))
        };
        assert!(!jeffreys_term_skippable_via_matvec(hv, p).unwrap());
    }

    /// Single-emission pin (gam#931) for the Jeffreys eigenvalue function
    /// `g(λ; floor)`: the three canonical functions `joint_jeffreys_term` now
    /// reads from — the value `g = jeffreys_antiderivative`, its λ-slope
    /// `g' = floored_inverse`, and its floor-motion
    /// `∂g/∂floor = jeffreys_antiderivative_floor_sensitivity` — are a
    /// consistent `(g, ∂g/∂λ, ∂g/∂floor)` triple. Because the production loop
    /// now PROJECTS its value, gradient and floor-response off exactly these
    /// functions (no inline copy), an FD agreement here is a structural
    /// guarantee the value and its derivatives cannot drift. We sample one
    /// point from each of the four branches and central-difference both `λ`
    /// (floor fixed) and `floor` (λ fixed), avoiding the C¹ knots.
    #[test]
    pub(crate) fn jeffreys_antiderivative_is_consistent_value_slope_floor_triple() {
        let floor = 1e-3_f64;
        let cap = jeffreys_cap(floor);
        // One sample comfortably inside each branch (away from cap/floor/0 knots).
        // The top branch here is GATE-bound (cap = CLEAR ≫ floor), so its
        // ∂g/∂floor is exactly 0 — exercised below alongside a separate
        // floor-bound-cap point.
        for &lam in &[cap * 4.0, (floor + cap) * 0.5, floor * 0.5, -0.7_f64] {
            // ∂g/∂λ = floored_inverse.
            let hl = 1e-7 * lam.abs().max(1e-3);
            let fd_lam = (jeffreys_antiderivative(lam + hl, floor)
                - jeffreys_antiderivative(lam - hl, floor))
                / (2.0 * hl);
            let dl = floored_inverse(lam, floor);
            assert!(
                (fd_lam - dl).abs() <= 1e-4 * dl.abs().max(1.0),
                "∂g/∂λ desync at λ={lam}: fd={fd_lam} analytic={dl}"
            );
            // ∂g/∂floor = jeffreys_antiderivative_floor_sensitivity.
            let hf = 1e-7 * floor;
            let fd_floor = (jeffreys_antiderivative(lam, floor + hf)
                - jeffreys_antiderivative(lam, floor - hf))
                / (2.0 * hf);
            let df = jeffreys_antiderivative_floor_sensitivity(lam, floor);
            assert!(
                (fd_floor - df).abs() <= 1e-4 * df.abs().max(1.0),
                "∂g/∂floor desync at λ={lam}: fd={fd_floor} analytic={df}"
            );
        }
        // Floor-bound-cap regime (`Λ = floor`, the extreme-scale branch): pick a
        // floor above the gate-clear scale so `jeffreys_cap(floor) = floor` and a
        // λ in the now-active top branch carries a nonzero ∂g/∂floor.
        let big_floor = CONDITIONING_GATE_ABSOLUTE_CLEAR * 10.0;
        assert!((jeffreys_cap(big_floor) - big_floor).abs() < 1e-12);
        let lam_top = big_floor * 3.0;
        let hf = 1e-7 * big_floor;
        let fd_floor = (jeffreys_antiderivative(lam_top, big_floor + hf)
            - jeffreys_antiderivative(lam_top, big_floor - hf))
            / (2.0 * hf);
        let df = jeffreys_antiderivative_floor_sensitivity(lam_top, big_floor);
        assert!(
            df != 0.0 && (fd_floor - df).abs() <= 1e-4 * df.abs().max(1.0),
            "floor-bound-cap ∂g/∂floor desync: fd={fd_floor} analytic={df}"
        );
    }
}
