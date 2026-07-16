//! Asymptote certificate (#2337 §2, Thm 2.1/2.2) — layer 3 of the #2299 rail fix.
//!
//! # The problem this certifies
//!
//! A double-penalty smooth (`s(x)`) gives block-orthogonal smoothing
//! coordinates, and some of them have **no interior optimum**: the objective
//! keeps improving as the coordinate runs to a rail. A *bending* coordinate
//! wants `λ → ∞` (`ρ → +∞`) to kill un-needed wiggle; a *null-space shrinkage*
//! coordinate wants `λ → 0` (`ρ → −∞`) so it does not shrink a real signal
//! living in the penalty null space. Neither coordinate is ever "stationary"
//! in the fixed-tolerance gradient sense — its projected gradient stays above
//! any fixed bound all the way to the rail — so a gradient-only certificate
//! grinds the outer loop to its iteration cap even though the *fitted model*
//! stopped moving long ago. This is the #2299 stall.
//!
//! # The tail law (Thm 2.1), experiment-verified
//!
//! Along such a coordinate the criterion `V(ρ)` obeys an **exact exponential
//! tail law** as it approaches its asymptote. For the upper rail
//! (`ρ → +∞`),
//!
//! ```text
//!     ∂V/∂ρ = −c · e^{−ρ},        c > 0  (the pencil-basis constant),
//! ```
//!
//! and symmetrically for the lower rail (`ρ → −∞`), `∂V/∂ρ = +c · e^{+ρ}`.
//! `exp4_rail.py` reproduces this: `e^{ρ}·∂V/∂ρ` is constant (`≈ −6723`) across
//! ~16 e-folds before dissolving into the finite-difference repro floor.
//!
//! Two consequences are load-bearing:
//!
//! 1. **The pencil constant is observable.** From one iterate,
//!    `ĉ = ∓ e^{±ρ} · ∂V/∂ρ` (sign per rail); on the tail it is a positive
//!    constant, off the tail (still-curved region, or the finite-difference
//!    noise floor) it drifts. A *window* of iterates with `ĉ` constant within a
//!    drift band **confirms the tail** — a single snapshot never does
//!    (`FlatnessWindow`: "snapshots never certify").
//!
//! 2. **The remaining value-gap to the rail is exactly `|∂V/∂ρ|`.** Integrating
//!    the tail law from the current `ρ` to the rail,
//!    `V(ρ) − V(rail) = ∫ ∂V = c·e^{∓ρ} = |∂V/∂ρ|`. So the entire objective
//!    improvement still available by running to the rail equals the current
//!    directional derivative magnitude — a computable, not assumed, quantity.
//!
//! # What the certificate actually gates on — estimand equivalence
//!
//! The value-gap `|∂V/∂ρ|` is *not* compared to the summation rounding floor
//! (Lemma 3.4): at a genuine plateau that floor (`~nε|V|`) is far below the
//! gap, so it would never certify. The floor's role here is narrower — it is
//! the **drift band's noise floor**, the level below which `ĉ` is
//! finite-difference noise and the tail is *not* confirmed (this is exactly why
//! `exp4`'s `ρ ≥ 30` rows, where `e^{ρ}∂V` swings to `−9111`, `+11221`, are
//! correctly rejected).
//!
//! The **certify gate is estimand equivalence**: the *fitted model* at the
//! current `ρ` must already equal the rail-limit fit to within an estimand
//! tolerance. On a confirmed geometric tail the per-step coefficient move
//! `‖Δβ‖` decays with a ratio `q < 1`, so the remaining travel to the limit is
//! the geometric tail sum
//!
//! ```text
//!     ‖β(ρ) − β(rail)‖ ≤ ‖Δβ_last‖ · q / (1 − q),
//! ```
//!
//! bounded from *observed* steps with no separate restricted fit. When that
//! bound is below the estimand tolerance the coordinate is certified
//! **stationary-at-asymptote** even though its gradient never cleared the fixed
//! bound — the loop stops because the answer stopped changing, which is the
//! decision the user actually cares about.
//!
//! This module computes those facts as pure functions with a deterministic
//! window; wiring them into `certify_outer_optimality` (and, per the §2
//! refactor, the survival certify path) is the consuming step.

use std::collections::VecDeque;

use gam_linalg::utils::KahanSum;

/// Default ring-buffer capacity for [`AsymptoteWindow`]. Enough recent iterates
/// for a stable constant-`ĉ` drift test while staying local to the current tail.
pub const DEFAULT_ASYMPTOTE_WINDOW: usize = 12;

/// Minimum confirmed-tail samples before any asymptote verdict is attempted.
/// Two points make `ĉ` trivially "constant" (no drift signal); three is the
/// smallest count at which the drift band carries information.
const MIN_TAIL_SAMPLES: usize = 3;

/// Which rail a coordinate is approaching.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum AsymptoteSide {
    /// `ρ → +∞` (e.g. a bending penalty driving `λ → ∞`). Descent direction
    /// `−∂V/∂ρ > 0`, so `∂V/∂ρ < 0`.
    Upper,
    /// `ρ → −∞` (e.g. a null-space shrinkage penalty driving `λ → 0`). Descent
    /// direction `−∂V/∂ρ < 0`, so `∂V/∂ρ > 0`.
    Lower,
}

impl AsymptoteSide {
    /// The rail a coordinate with directional derivative `grad` is descending
    /// toward, or `None` when `|grad|` is within `interior_grad_tol` (the point
    /// is interior-stationary in this coordinate — an [`super`] Interior case,
    /// not an asymptote).
    pub fn from_gradient(grad: f64, interior_grad_tol: f64) -> Option<Self> {
        if !grad.is_finite() || grad.abs() <= interior_grad_tol {
            None
        } else if grad < 0.0 {
            // V decreasing in ρ ⇒ descent runs ρ upward.
            Some(AsymptoteSide::Upper)
        } else {
            Some(AsymptoteSide::Lower)
        }
    }

    /// The pencil constant `ĉ = ∓ e^{±ρ} · grad` for this rail. Positive on a
    /// genuine tail (`grad = ∓c e^{∓ρ}` ⇒ `ĉ = c > 0`).
    pub fn tail_constant(self, rho: f64, grad: f64) -> f64 {
        match self {
            // ĉ = −e^{+ρ}·grad, grad = −c e^{−ρ} ⇒ ĉ = c.
            AsymptoteSide::Upper => -rho.exp() * grad,
            // ĉ = +e^{−ρ}·grad, grad = +c e^{+ρ} ⇒ ĉ = c.
            AsymptoteSide::Lower => (-rho).exp() * grad,
        }
    }
}

/// One outer iterate's contribution to a single coordinate's tail history.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct AsymptoteSample {
    /// The coordinate's smoothing value `ρ_i` at this iterate.
    pub rho: f64,
    /// The coordinate's directional derivative `∂V/∂ρ_i` at this iterate.
    pub grad: f64,
    /// The accepted coefficient move `‖β_k − β_{k−1}‖` for this step (the
    /// estimand-travel signal). `0.0` if the step did not move coefficients.
    pub coef_step_norm: f64,
}

/// A deterministic fixed-capacity window of one coordinate's recent iterates,
/// used to confirm the exponential tail and bound the remaining estimand travel.
///
/// Determinism: FIFO eviction, ordered iteration, no randomness.
#[derive(Clone, Debug)]
pub struct AsymptoteWindow {
    capacity: usize,
    ring: VecDeque<AsymptoteSample>,
}

impl Default for AsymptoteWindow {
    fn default() -> Self {
        Self::with_capacity(DEFAULT_ASYMPTOTE_WINDOW)
    }
}

impl AsymptoteWindow {
    /// A window with the [`DEFAULT_ASYMPTOTE_WINDOW`] capacity.
    pub fn new() -> Self {
        Self::default()
    }

    /// A window with an explicit ring capacity (promoted to at least one).
    pub fn with_capacity(capacity: usize) -> Self {
        let capacity = capacity.max(1);
        Self {
            capacity,
            ring: VecDeque::with_capacity(capacity),
        }
    }

    /// Record one iterate, evicting the oldest sample if the ring is full.
    pub fn push(&mut self, sample: AsymptoteSample) {
        if self.ring.len() == self.capacity {
            self.ring.pop_front();
        }
        self.ring.push_back(sample);
    }

    /// Retained samples, oldest to newest.
    pub fn samples(&self) -> impl Iterator<Item = &AsymptoteSample> {
        self.ring.iter()
    }

    /// Number of retained samples.
    pub fn len(&self) -> usize {
        self.ring.len()
    }

    /// Whether the ring is empty.
    pub fn is_empty(&self) -> bool {
        self.ring.is_empty()
    }

    /// The most recent sample, or `None` if empty.
    pub fn latest(&self) -> Option<&AsymptoteSample> {
        self.ring.back()
    }
}

/// The tolerances an asymptote assessment is measured against.
#[derive(Clone, Copy, Debug)]
pub struct AsymptoteTolerances {
    /// `|grad|` at or below this is interior-stationary — no asymptote to
    /// certify. (The fixed outer gradient bound.)
    pub interior_grad_tol: f64,
    /// The pencil constant `ĉ` must exceed this to be a *genuine* tail rather
    /// than finite-difference noise (the Lemma 3.4 summation-model floor scaled
    /// to the gradient's regime). Below it the tail is unconfirmed.
    pub tail_noise_floor: f64,
    /// Maximum relative spread of `ĉ` across the window for the tail to be
    /// confirmed constant (the drift band). `(max ĉ − min ĉ)/mean ĉ`.
    pub tail_drift_rel: f64,
    /// The estimand tolerance: the remaining coefficient travel to the rail
    /// must fall at or below this for `stationary-at-asymptote`.
    pub estimand_tol: f64,
}

/// The result of assessing one coordinate against its recent history.
#[derive(Clone, Debug, PartialEq)]
pub enum AsymptoteVerdict {
    /// The coordinate is on a confirmed exponential tail AND the fitted model
    /// already equals the rail-limit fit to within `estimand_tol`. It may be
    /// certified stationary despite a gradient above the fixed bound.
    CertifiedAtAsymptote {
        /// Which rail.
        side: AsymptoteSide,
        /// The observed pencil constant `ĉ` (window mean).
        tail_constant: f64,
        /// The exact remaining value-gap to the rail, `|∂V/∂ρ|`.
        value_gap: f64,
        /// The bound on remaining coefficient travel to the rail limit.
        estimand_travel_bound: f64,
    },
    /// A confirmed tail, but the fitted model is still moving: the remaining
    /// estimand travel exceeds `estimand_tol`. Not stationary yet — the loop
    /// should keep stepping (the tail law guarantees it will get there).
    OnTailNotYetEquivalent {
        /// Which rail.
        side: AsymptoteSide,
        /// The bound on remaining coefficient travel (exceeds `estimand_tol`).
        estimand_travel_bound: f64,
    },
    /// No confirmed tail — either interior-stationary, too few samples, drifting
    /// `ĉ` (still in the curved region), or `ĉ` below the noise floor
    /// (finite-difference-dominated). Never certifies.
    NoAsymptote {
        /// Human-readable reason, for logs.
        reason: String,
    },
}

/// Mean and relative spread `(max − min)/|mean|` of a slice, or `None` if empty
/// or the mean is not usable. Deterministic (Kahan sum, ordered).
fn mean_and_rel_spread(values: &[f64]) -> Option<(f64, f64)> {
    if values.is_empty() {
        return None;
    }
    let mut sum = KahanSum::default();
    let mut lo = f64::INFINITY;
    let mut hi = f64::NEG_INFINITY;
    for &v in values {
        if !v.is_finite() {
            return None;
        }
        sum.add(v);
        lo = lo.min(v);
        hi = hi.max(v);
    }
    let mean = sum.sum() / values.len() as f64;
    if !(mean.abs() > 0.0) {
        return None;
    }
    Some((mean, (hi - lo) / mean.abs()))
}

/// Geometric decay ratio `q` of the accepted coefficient moves `‖Δβ‖` across
/// the window, estimated as the ratio of the newest to the previous nonzero
/// step. Returns `None` if fewer than two positive moves are available. On a
/// confirmed tail `q ∈ (0, 1)`.
fn coef_step_ratio(samples: &[AsymptoteSample]) -> Option<f64> {
    let positives: Vec<f64> = samples
        .iter()
        .map(|s| s.coef_step_norm)
        .filter(|&d| d.is_finite() && d > 0.0)
        .collect();
    if positives.len() < 2 {
        return None;
    }
    let last = positives[positives.len() - 1];
    let prev = positives[positives.len() - 2];
    if !(prev > 0.0) {
        return None;
    }
    Some(last / prev)
}

/// Assess a single coordinate for the asymptote (stationary-at-rail) certificate.
///
/// The window must carry the coordinate's recent `(ρ, ∂V/∂ρ, ‖Δβ‖)` iterates.
/// Reasoning order:
///
/// 1. **Side from the current gradient.** `|grad| ≤ interior_grad_tol` ⇒ no
///    asymptote (interior case).
/// 2. **Confirm the tail.** Compute `ĉ` per retained sample; require
///    `≥ MIN_TAIL_SAMPLES`, all positive, mean above `tail_noise_floor`, and
///    relative spread within `tail_drift_rel`. Any failure ⇒ `NoAsymptote`.
/// 3. **Bound the remaining estimand travel.** From the geometric coefficient
///    ratio `q`, `‖Δβ_last‖ · q/(1−q)`. A non-contracting ratio (`q ≥ 1`) means
///    the estimand is not settling ⇒ `NoAsymptote`.
/// 4. **Decide.** `travel ≤ estimand_tol` ⇒ `CertifiedAtAsymptote`, else
///    `OnTailNotYetEquivalent`.
pub fn assess_coordinate(
    window: &AsymptoteWindow,
    tol: &AsymptoteTolerances,
) -> AsymptoteVerdict {
    let latest = match window.latest() {
        Some(s) => *s,
        None => {
            return AsymptoteVerdict::NoAsymptote {
                reason: "empty window".to_string(),
            }
        }
    };
    let side = match AsymptoteSide::from_gradient(latest.grad, tol.interior_grad_tol) {
        Some(s) => s,
        None => {
            return AsymptoteVerdict::NoAsymptote {
                reason: format!(
                    "interior-stationary in this coordinate: |grad|={:.3e} ≤ tol {:.3e}",
                    latest.grad.abs(),
                    tol.interior_grad_tol
                ),
            }
        }
    };

    let samples: Vec<AsymptoteSample> = window.samples().copied().collect();
    if samples.len() < MIN_TAIL_SAMPLES {
        return AsymptoteVerdict::NoAsymptote {
            reason: format!(
                "too few samples to confirm a tail: {} < {MIN_TAIL_SAMPLES}",
                samples.len()
            ),
        };
    }

    // Every retained sample must be on the SAME rail for a coherent tail; a
    // sign flip in `ĉ` means we are not (yet) on a single exponential.
    let constants: Vec<f64> = samples
        .iter()
        .map(|s| side.tail_constant(s.rho, s.grad))
        .collect();
    if constants.iter().any(|&c| !(c > 0.0)) {
        return AsymptoteVerdict::NoAsymptote {
            reason: "pencil constant ĉ not uniformly positive across the window (not a single tail)"
                .to_string(),
        };
    }
    let (mean_c, spread) = match mean_and_rel_spread(&constants) {
        Some(v) => v,
        None => {
            return AsymptoteVerdict::NoAsymptote {
                reason: "pencil constant ĉ has no usable mean".to_string(),
            }
        }
    };
    if mean_c <= tol.tail_noise_floor {
        return AsymptoteVerdict::NoAsymptote {
            reason: format!(
                "pencil constant ĉ={mean_c:.3e} below the noise floor {:.3e} \
                 (finite-difference-dominated, not a confirmed tail)",
                tol.tail_noise_floor
            ),
        };
    }
    if spread > tol.tail_drift_rel {
        return AsymptoteVerdict::NoAsymptote {
            reason: format!(
                "pencil constant ĉ drifts {spread:.3e} > band {:.3e} (still in the curved region)",
                tol.tail_drift_rel
            ),
        };
    }

    // Tail confirmed. The exact remaining value-gap is |∂V/∂ρ|; the remaining
    // estimand travel is the geometric tail sum of the coefficient moves.
    let value_gap = latest.grad.abs();
    let q = match coef_step_ratio(&samples) {
        Some(q) if q.is_finite() && q >= 0.0 && q < 1.0 => q,
        _ => {
            return AsymptoteVerdict::NoAsymptote {
                reason: "coefficient moves not geometrically contracting (estimand not settling)"
                    .to_string(),
            }
        }
    };
    let last_step = samples
        .iter()
        .rev()
        .map(|s| s.coef_step_norm)
        .find(|&d| d.is_finite() && d > 0.0)
        .unwrap_or(0.0);
    let estimand_travel_bound = last_step * q / (1.0 - q);

    if estimand_travel_bound <= tol.estimand_tol {
        AsymptoteVerdict::CertifiedAtAsymptote {
            side,
            tail_constant: mean_c,
            value_gap,
            estimand_travel_bound,
        }
    } else {
        AsymptoteVerdict::OnTailNotYetEquivalent {
            side,
            estimand_travel_bound,
        }
    }
}

#[cfg(test)]
mod asymptote_certificate_tests {
    use super::*;

    /// Build a window from a pure lower-rail exponential tail with a KNOWN
    /// pencil constant `c`: `ρ_k = ρ0 − k·dρ` (running toward −∞),
    /// `grad_k = c·e^{ρ_k}`, and coefficient moves `‖Δβ‖_k = a·e^{ρ_k}` (they
    /// decay with the same ratio `q = e^{−dρ}` as the tail — the estimand tracks
    /// the value).
    fn lower_tail_window(c: f64, a: f64, rho0: f64, drho: f64, n: usize) -> AsymptoteWindow {
        let mut w = AsymptoteWindow::with_capacity(n);
        for k in 0..n {
            let rho = rho0 - (k as f64) * drho;
            w.push(AsymptoteSample {
                rho,
                grad: c * rho.exp(),
                coef_step_norm: a * rho.exp(),
            });
        }
        w
    }

    fn tol(estimand_tol: f64) -> AsymptoteTolerances {
        AsymptoteTolerances {
            interior_grad_tol: 1.0e-8,
            tail_noise_floor: 1.0e-6,
            tail_drift_rel: 1.0e-3,
            estimand_tol,
        }
    }

    /// The pencil constant `ĉ = e^{−ρ}·grad` is recovered on a pure lower tail,
    /// and the exact remaining value-gap equals `|grad|`.
    #[test]
    fn lower_tail_recovers_constant_and_value_gap() {
        let c = 6723.0;
        let window = lower_tail_window(c, 1.0e-3, -7.0, 0.5, 8);
        // A very loose estimand tol so the verdict is Certified and exposes the
        // recovered constant + value_gap.
        match assess_coordinate(&window, &tol(1.0)) {
            AsymptoteVerdict::CertifiedAtAsymptote {
                side,
                tail_constant,
                value_gap,
                ..
            } => {
                assert_eq!(side, AsymptoteSide::Lower);
                assert!(
                    (tail_constant - c).abs() / c < 1.0e-9,
                    "recovered ĉ={tail_constant} should equal c={c}"
                );
                let latest = window.latest().unwrap();
                assert!(
                    (value_gap - latest.grad.abs()).abs() <= f64::EPSILON * value_gap.max(1.0),
                    "value_gap must be the exact tail integral |grad|"
                );
            }
            other => panic!("expected CertifiedAtAsymptote on a pure tail, got {other:?}"),
        }
    }

    /// A confirmed tail whose remaining estimand travel is below tolerance
    /// certifies; the same tail with a tighter estimand tolerance refuses (but
    /// stays OnTail — it is not spuriously demoted to NoAsymptote).
    #[test]
    fn estimand_gate_decides_certify_vs_on_tail() {
        // Last step ‖Δβ‖ = 1e-3·e^{ρ_last}; with dρ=0.5, q=e^{−0.5}=0.6065,
        // travel bound = last·q/(1−q). Make the numbers concrete at ρ0=-7.
        let window = lower_tail_window(6723.0, 1.0e-3, -7.0, 0.5, 8);
        let last_step = window.latest().unwrap().coef_step_norm; // 1e-3·e^{ρ_last}
        let q = (-0.5_f64).exp();
        let expected_travel = last_step * q / (1.0 - q);

        match assess_coordinate(&window, &tol(expected_travel * 2.0)) {
            AsymptoteVerdict::CertifiedAtAsymptote {
                estimand_travel_bound,
                ..
            } => {
                assert!(
                    (estimand_travel_bound - expected_travel).abs()
                        <= 1.0e-9 * expected_travel.max(1.0),
                    "travel bound {estimand_travel_bound} should match the geometric tail sum \
                     {expected_travel}"
                );
            }
            other => panic!("expected Certified under a loose estimand tol, got {other:?}"),
        }

        match assess_coordinate(&window, &tol(expected_travel * 0.5)) {
            AsymptoteVerdict::OnTailNotYetEquivalent {
                estimand_travel_bound,
                side,
            } => {
                assert_eq!(side, AsymptoteSide::Lower);
                assert!(estimand_travel_bound > expected_travel * 0.5);
            }
            other => panic!("expected OnTailNotYetEquivalent under a tight estimand tol, got {other:?}"),
        }
    }

    /// An upper-rail pure tail is classified `Upper` and certified — the sign
    /// convention `ĉ = −e^{+ρ}·grad` recovers a positive constant.
    #[test]
    fn upper_tail_side_and_constant() {
        let c = 42.0;
        let mut w = AsymptoteWindow::with_capacity(6);
        for k in 0..6 {
            let rho = 8.0 + (k as f64) * 0.5; // running toward +∞
            w.push(AsymptoteSample {
                rho,
                grad: -c * (-rho).exp(), // grad = −c·e^{−ρ} < 0
                coef_step_norm: 1.0e-4 * (-rho).exp(),
            });
        }
        match assess_coordinate(&w, &tol(1.0)) {
            AsymptoteVerdict::CertifiedAtAsymptote {
                side,
                tail_constant,
                ..
            } => {
                assert_eq!(side, AsymptoteSide::Upper);
                assert!((tail_constant - c).abs() / c < 1.0e-9);
            }
            other => panic!("expected Certified Upper tail, got {other:?}"),
        }
    }

    /// The finite-difference noise regime (exp4's `ρ ≥ 30`, where `e^{ρ}∂V`
    /// swings sign and magnitude) is rejected: a drifting `ĉ` is not a
    /// confirmed tail, so the certificate refuses rather than false-certifying.
    #[test]
    fn drifting_constant_is_rejected_as_noise_regime() {
        // ĉ values that swing like exp4's tail floor: 6723, 6731, 9111, ...
        let rhos = [-9.0, -9.5, -10.0, -10.5];
        let cs = [6723.0, 6731.0, 9111.0, 4200.0]; // large relative drift
        let mut w = AsymptoteWindow::with_capacity(4);
        for (rho, c) in rhos.iter().zip(cs.iter()) {
            w.push(AsymptoteSample {
                rho: *rho,
                grad: c * rho.exp(),
                coef_step_norm: 1.0e-3 * rho.exp(),
            });
        }
        match assess_coordinate(&w, &tol(1.0)) {
            AsymptoteVerdict::NoAsymptote { reason } => {
                assert!(reason.contains("drift"), "should reject on drift: {reason}");
            }
            other => panic!("drifting ĉ must not certify, got {other:?}"),
        }
    }

    /// An interior-stationary coordinate (`|grad|` under the fixed bound) is not
    /// an asymptote case and must not be certified as one.
    #[test]
    fn interior_stationary_is_not_an_asymptote() {
        let mut w = AsymptoteWindow::new();
        for k in 0..4 {
            w.push(AsymptoteSample {
                rho: 1.0 + k as f64 * 0.1,
                grad: 1.0e-12, // below interior_grad_tol
                coef_step_norm: 1.0e-9,
            });
        }
        match assess_coordinate(&w, &tol(1.0)) {
            AsymptoteVerdict::NoAsymptote { reason } => {
                assert!(reason.contains("interior-stationary"), "{reason}");
            }
            other => panic!("interior point must not be an asymptote, got {other:?}"),
        }
    }

    /// The exp4_rail.py tail, fed as *verified* `(ρ, ∂V/∂ρ)` literals, is
    /// classified correctly by the full `assess_coordinate` path:
    ///
    /// * On the confirmed-tail rows `ρ ∈ {14,…,24}` the pencil constant
    ///   `ĉ = −e^{ρ}·∂V` sits at `≈ 6723` with spread well under the band, so a
    ///   loose estimand tolerance yields `CertifiedAtAsymptote` (Upper rail).
    /// * On the finite-difference noise floor `ρ ∈ {28,30,32}` the same
    ///   statistic swings (`ĉ ≈ 6577, 9112, −11221` — the last one *negative*),
    ///   so the tail is NOT confirmed and the certificate refuses. This is the
    ///   self-protection against the repro floor: the detector never
    ///   false-certifies FD noise as a tail.
    ///
    /// The coefficient moves are a clean geometric decay (matching exp4's
    /// `‖b−b_inf‖` column, which contracts by ≈ e^{−2} per Δρ=2) so the
    /// estimand machinery reaches the tail verdict rather than short-circuiting.
    #[test]
    fn exp4_verified_tail_certifies_and_noise_floor_refuses() {
        // exp4_rail.py, columns ρ and dV/drho (FD). Upper rail (ρ → +∞).
        let confirmed: [(f64, f64); 6] = [
            (14.0, -5.589576e-03),
            (16.0, -7.565987e-04),
            (18.0, -1.023966e-04),
            (20.0, -1.385843e-05),
            (22.0, -1.874980e-06),
            (24.0, -2.538059e-07),
        ];
        let mut tail = AsymptoteWindow::with_capacity(confirmed.len());
        // ‖b−b_inf‖ from exp4 at these rows, used as the per-step estimand move.
        let coef = [6.196e-05, 8.398e-06, 1.137e-06, 1.539e-07, 2.082e-08, 2.818e-09];
        for (i, (rho, grad)) in confirmed.iter().enumerate() {
            tail.push(AsymptoteSample {
                rho: *rho,
                grad: *grad,
                coef_step_norm: coef[i],
            });
        }
        // The recovered ĉ is the exp4 pencil constant to 3+ figures.
        let constants: Vec<f64> = tail
            .samples()
            .map(|s| AsymptoteSide::Upper.tail_constant(s.rho, s.grad))
            .collect();
        for c in &constants {
            assert!(
                (c - 6723.0).abs() / 6723.0 < 5.0e-3,
                "exp4 pencil constant ĉ={c} should be ≈ 6723"
            );
        }
        match assess_coordinate(&tail, &tol(1.0)) {
            AsymptoteVerdict::CertifiedAtAsymptote { side, tail_constant, .. } => {
                assert_eq!(side, AsymptoteSide::Upper);
                assert!((tail_constant - 6723.0).abs() / 6723.0 < 5.0e-3);
            }
            other => panic!("exp4 confirmed-tail rows must certify, got {other:?}"),
        }

        // The finite-difference noise floor: e^{ρ}·dV = −6577, −9112, +11221
        // (exp4's ρ=28,30,32 rows). The ρ=32 FD gradient even flips sign, so
        // the pencil constant is not uniformly one sign — not a coherent single
        // tail. These gradients are tiny (≤ 5e-9), so we drive the tail logic
        // directly with an interior_grad_tol below them; the point is to prove
        // the tail-confirmation stage itself rejects the noise regime, not that
        // the interior short-circuit happens to catch it.
        let noise: [(f64, f64); 3] = [
            (28.0, -4.547474e-09),
            (30.0, -8.526513e-10),
            (32.0, 1.421085e-10),
        ];
        let mut floor = AsymptoteWindow::with_capacity(noise.len());
        for (rho, grad) in noise.iter() {
            floor.push(AsymptoteSample {
                rho: *rho,
                grad: *grad,
                coef_step_norm: 1.0e-11,
            });
        }
        let noise_tol = AsymptoteTolerances {
            interior_grad_tol: 1.0e-13,
            ..tol(1.0)
        };
        match assess_coordinate(&floor, &noise_tol) {
            AsymptoteVerdict::NoAsymptote { .. } => {}
            other => panic!("the FD noise floor must NOT certify as a tail, got {other:?}"),
        }
    }

    /// Too few samples never certify (a snapshot cannot confirm a tail).
    #[test]
    fn too_few_samples_never_certify() {
        let mut w = AsymptoteWindow::new();
        w.push(AsymptoteSample {
            rho: -7.0,
            grad: 6723.0 * (-7.0_f64).exp(),
            coef_step_norm: 1.0e-3,
        });
        w.push(AsymptoteSample {
            rho: -7.5,
            grad: 6723.0 * (-7.5_f64).exp(),
            coef_step_norm: 6.0e-4,
        });
        match assess_coordinate(&w, &tol(1.0)) {
            AsymptoteVerdict::NoAsymptote { reason } => {
                assert!(reason.contains("too few samples"), "{reason}");
            }
            other => panic!("two samples must not certify, got {other:?}"),
        }
    }
}
