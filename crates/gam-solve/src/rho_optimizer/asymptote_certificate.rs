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
//!    noise floor) it drifts. A *window* of iterates whose `ĉ` settles within
//!    its own rounding bands **confirms the tail** — a single snapshot never
//!    does (`FlatnessWindow`: "snapshots never certify").
//!
//! 2. **The remaining value-gap to the rail is exactly `|∂V/∂ρ|`.** Integrating
//!    the tail law from the current `ρ` to the rail,
//!    `V(ρ) − V(rail) = ∫ ∂V = c·e^{∓ρ} = |∂V/∂ρ|`. So the entire objective
//!    improvement still available by running to the rail equals the current
//!    directional derivative magnitude — a computable, not assumed, quantity.
//!
//! # What the certificate actually gates on
//!
//! Every band is derived at the probe point, never read off a fixture (#3565):
//!
//! * **Gradient band.** Each probed `∂V/∂ρ` carries the rigorous rounding
//!   bound `ε` of its own formation (the Theorem 9 coordinate band of
//!   `outer_coordinate_bands`, charged on the magnitudes of the channels the
//!   component was summed from). A sample with `|∂V/∂ρ| ≤ ε` carries no sign,
//!   let alone a tail; that is the whole "finite-difference floor" beside the
//!   rail, and it scales with the criterion exactly as the gradient does.
//! * **Pencil-constant band.** The same bound mapped through the tail law,
//!   `β = e^{±ρ}·ε`, is the resolution of `ĉ` at that probe.
//! * **Settlement, not a drift band.** Past the leading term the tail law is
//!   `ĉ(ρ) = c + O(e^{∓ρ})`: consecutive `ĉ` differences contract
//!   geometrically toward the rail. `settle_tail_constant` bounds that
//!   contraction ratio from the observed differences and their bands, and sums
//!   the remaining geometric series into an extrapolation radius `R` with
//!   `|c − ĉ_last| ≤ R`. A window whose differences grow toward the rail, or
//!   do not resolvably contract, does not settle.
//! * **Value gap against the criterion's resolution.** Integrating the tail law
//!   from the shipped `ρ̂` to the rail, the objective improvement still
//!   available is at most `(ĉ + R)·e^{∓ρ̂}`. The coordinate is certified
//!   **stationary-at-asymptote** when that gap is within the criterion's
//!   declared statistical resolution (`outer_criterion_resolution`, the same
//!   number every other "the criterion cannot tell these apart" judgement
//!   reads): running to the rail cannot move any reported quantity by more
//!   than the sampling error the inference already carries.
//! * **Estimand settling.** On a confirmed geometric tail the per-step
//!   coefficient move `‖Δβ‖` decays with a ratio `q < 1`, so the remaining
//!   coefficient travel is the geometric tail sum
//!
//! ```text
//!     ‖β(ρ) − β(rail)‖ ≤ ‖Δβ_last‖ · q / (1 − q),
//! ```
//!
//! bounded from *observed* steps with no separate restricted fit. A window
//! whose coefficients do not contract is not on the tail and never certifies.
//!
//! This module computes those facts as pure functions with a deterministic
//! window; `run.rs` places the probes and supplies each sample's band.

use std::collections::VecDeque;

/// Default ring-buffer capacity for [`AsymptoteWindow`]. Enough recent iterates
/// for a settlement test while staying local to the current tail.
pub(crate) const DEFAULT_ASYMPTOTE_WINDOW: usize = 12;

/// Minimum confirmed-tail samples before any asymptote verdict is attempted.
/// Two points give one `ĉ` difference and no contraction signal; three give
/// two differences, the smallest count from which the geometric settlement of
/// the next-order term can be bounded.
pub(crate) const MIN_TAIL_SAMPLES: usize = 3;

/// Which rail a coordinate is approaching.
#[derive(Clone, Copy, Debug, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
pub enum AsymptoteSide {
    /// `ρ → +∞` (e.g. a bending penalty driving `λ → ∞`). Descent direction
    /// `−∂V/∂ρ > 0`, so `∂V/∂ρ < 0`.
    Upper,
    /// `ρ → −∞` (e.g. a null-space shrinkage penalty driving `λ → 0`). Descent
    /// direction `−∂V/∂ρ < 0`, so `∂V/∂ρ > 0`.
    Lower,
}

impl AsymptoteSide {
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

    /// The criterion improvement still available by running from `rho` to this
    /// rail on a tail with pencil constant `constant`: the tail-law integral
    /// `∫ |∂V/∂ρ| = constant·e^{∓ρ}`.
    pub(crate) fn value_gap(self, rho: f64, constant: f64) -> f64 {
        match self {
            AsymptoteSide::Upper => constant * (-rho).exp(),
            AsymptoteSide::Lower => constant * rho.exp(),
        }
    }
}

/// One probe's contribution to a single coordinate's tail history.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct AsymptoteSample {
    /// The coordinate's smoothing value `ρ_i` at this probe.
    pub rho: f64,
    /// The coordinate's directional derivative `∂V/∂ρ_i` at this probe.
    pub grad: f64,
    /// The rigorous rounding bound on `grad`: the forward error of its
    /// formation at this probe (`ε_j` of the coordinate's Theorem 9 band).
    pub grad_band: f64,
    /// The coefficient move `‖β_k − β_{k−1}‖` into this probe from the previous
    /// one (the estimand-travel signal). `0.0` when no move was observed.
    pub coef_step_norm: f64,
}

/// A deterministic fixed-capacity window of one coordinate's probes, ordered
/// from the interior toward the rail, used to confirm the exponential tail and
/// bound the remaining estimand travel.
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
    /// A window with the `DEFAULT_ASYMPTOTE_WINDOW` capacity.
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

    /// Record one probe, evicting the oldest sample if the ring is full.
    pub fn push(&mut self, sample: AsymptoteSample) {
        if self.ring.len() == self.capacity {
            self.ring.pop_front();
        }
        self.ring.push_back(sample);
    }

    /// Retained samples, oldest (most interior) to newest (nearest the rail).
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

/// The result of assessing one coordinate against its probed tail.
#[derive(Clone, Debug, PartialEq)]
pub(crate) enum AsymptoteVerdict {
    /// The coordinate is on a confirmed, settled exponential tail, its
    /// coefficients contract toward the rail limit, and the criterion
    /// improvement still available by running to the rail is within the
    /// criterion's resolution. It may be certified stationary despite a
    /// gradient above the fixed bound.
    CertifiedAtAsymptote {
        /// Which rail.
        side: AsymptoteSide,
        /// The settled pencil constant `ĉ` (the rail-most sample's).
        tail_constant: f64,
        /// The bound `R` with `|c − tail_constant| ≤ R` for the tail's limit
        /// constant `c` ([`settle_tail_constant`]).
        extrapolation_radius: f64,
        /// The rounding bound on the rail-most sample's gradient.
        gradient_band: f64,
        /// The bound on the remaining value gap to the rail from `ρ̂`,
        /// `(tail_constant + R)·e^{∓ρ̂}`.
        value_gap: f64,
        /// The bound on remaining coefficient travel to the rail limit.
        estimand_travel_bound: f64,
    },
    /// A confirmed, settled tail, but running to the rail would still improve
    /// the criterion by more than it resolves: the loop should keep stepping.
    OnTailNotYetEquivalent {
        /// Which rail.
        side: AsymptoteSide,
        /// The settled pencil constant `ĉ`.
        tail_constant: f64,
        /// The settlement radius on `ĉ`.
        extrapolation_radius: f64,
        /// The bound on the remaining value gap (exceeds `value_gap_tol`).
        value_gap: f64,
        /// The criterion resolution the gap was judged against.
        value_gap_tol: f64,
    },
    /// No confirmed tail — too few samples, a gradient inside its own rounding
    /// band, a `ĉ` of the wrong sign, a constant that does not settle, or
    /// coefficients that do not contract. Never certifies.
    NoAsymptote {
        /// Human-readable reason, for logs.
        reason: String,
    },
}

/// A settled pencil constant: the tail's limit constant `c` lies within
/// `radius` of `limit`.
#[derive(Clone, Copy, Debug, PartialEq)]
pub(crate) struct TailSettlement {
    /// The rail-most observed constant.
    pub(crate) limit: f64,
    /// `|c − limit| ≤ radius` for the limit constant `c`.
    pub(crate) radius: f64,
}

/// Bound the limit of a sequence of pencil constants from its own differences.
///
/// `constants[i]` are equally spaced probes ordered from the interior toward
/// the rail, each resolved to `± bands[i]`. Past the leading term the tail law
/// is `ĉ(ρ) = c + a·e^{∓ρ} + …`, so on the tail the successive differences
/// `D_i = ĉ_{i+1} − ĉ_i` contract geometrically toward the rail. Each
/// difference is known to the interval `|D_i| ∈ [lo_i, hi_i]` with
/// `lo_i = max(|D_i| − (β_i + β_{i+1}), 0)` and `hi_i = |D_i| + β_i + β_{i+1}`.
///
/// * A difference that is resolvably LARGER than the one before it
///   (`lo_{i+1} ≥ hi_i > 0`) is growth toward the rail — the noise-dominated
///   band beside it or a curved region — and the window does not settle.
/// * When no difference is resolved at all, the constants agree within their
///   bands and `radius` is the widest observed disagreement plus the rail-most
///   band.
/// * Otherwise the contraction ratio is bounded above by
///   `r = max_i hi_{i+1}/lo_i` over the resolved differences that have a
///   successor. `r < 1` sums the remaining series:
///   `|c − ĉ_last| ≤ hi_last·r/(1 − r) + β_last`. `r ≥ 1` does not settle.
pub(crate) fn settle_tail_constant(
    constants: &[f64],
    bands: &[f64],
) -> Result<TailSettlement, String> {
    let m = constants.len();
    if m < MIN_TAIL_SAMPLES || bands.len() != m {
        return Err(format!(
            "settlement needs at least {MIN_TAIL_SAMPLES} banded constants, got {m} constants \
             and {} bands",
            bands.len()
        ));
    }
    if constants.iter().any(|c| !c.is_finite())
        || bands.iter().any(|b| !(b.is_finite() && *b >= 0.0))
    {
        return Err("pencil constant or its band is not finite".to_string());
    }
    let limit = constants[m - 1];
    let limit_band = bands[m - 1];
    let steps: Vec<(f64, f64)> = (0..m - 1)
        .map(|i| {
            let difference = (constants[i + 1] - constants[i]).abs();
            let band = bands[i] + bands[i + 1];
            ((difference - band).max(0.0), difference + band)
        })
        .collect();
    for (i, pair) in steps.windows(2).enumerate() {
        let (_, hi_here) = pair[0];
        let (lo_next, _) = pair[1];
        if lo_next > 0.0 && lo_next >= hi_here {
            return Err(format!(
                "pencil constant not settling: its step {} toward the rail is at least \
                 {lo_next:.3e}, after a step of at most {hi_here:.3e}",
                i + 1
            ));
        }
    }
    if steps.iter().all(|&(lo, _)| lo == 0.0) {
        let spread = constants
            .iter()
            .zip(bands)
            .map(|(&c, &b)| (c - limit).abs() + b)
            .fold(0.0_f64, f64::max);
        return Ok(TailSettlement {
            limit,
            radius: spread + limit_band,
        });
    }
    let ratio = steps
        .windows(2)
        .filter(|pair| pair[0].0 > 0.0)
        .map(|pair| pair[1].1 / pair[0].0)
        .fold(None, |acc: Option<f64>, r| Some(acc.map_or(r, |q| q.max(r))));
    let Some(ratio) = ratio else {
        return Err(
            "pencil constant not settling: only its rail-most step is resolved, so its \
             contraction is unobserved"
                .to_string(),
        );
    };
    if !(ratio < 1.0) {
        return Err(format!(
            "pencil constant not settling: its steps are not contracting (ratio bound \
             {ratio:.3e} ≥ 1)"
        ));
    }
    let (_, hi_last) = steps[m - 2];
    Ok(TailSettlement {
        limit,
        radius: hi_last * ratio / (1.0 - ratio) + limit_band,
    })
}

/// Geometric decay ratio `q` of the coefficient moves `‖Δβ‖` across the
/// window, estimated as the ratio of the newest to the previous nonzero step.
/// Returns `None` if fewer than two positive moves are available. On a
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
    Some(last / prev)
}

/// Assess a single coordinate for the asymptote (stationary-at-rail)
/// certificate.
///
/// The window carries the coordinate's `(ρ, ∂V/∂ρ, ε, ‖Δβ‖)` probes, ordered
/// from the interior toward `side`'s rail; `rho_hat` is the shipped point and
/// `value_gap_tol` the criterion's resolution. Reasoning order:
///
/// 1. **Enough samples** (`MIN_TAIL_SAMPLES`).
/// 2. **Every sample resolved.** `|∂V/∂ρ| > ε` (a gradient inside its own
///    rounding band has no sign) and `ĉ > 0` for this rail.
/// 3. **The constant settles** ([`settle_tail_constant`]) with its whole
///    interval positive, `ĉ − R > 0`.
/// 4. **The estimand settles.** The coefficient moves contract, `q < 1`, and
///    the remaining travel is `‖Δβ_last‖·q/(1−q)`.
/// 5. **Decide.** The remaining value gap `(ĉ + R)·e^{∓ρ̂}` within
///    `value_gap_tol` ⇒ `CertifiedAtAsymptote`, else `OnTailNotYetEquivalent`.
pub(crate) fn assess_coordinate(
    window: &AsymptoteWindow,
    side: AsymptoteSide,
    rho_hat: f64,
    value_gap_tol: f64,
) -> AsymptoteVerdict {
    let samples: Vec<AsymptoteSample> = window.samples().copied().collect();
    if samples.len() < MIN_TAIL_SAMPLES {
        return AsymptoteVerdict::NoAsymptote {
            reason: format!(
                "too few samples to confirm a tail: {} < {MIN_TAIL_SAMPLES}",
                samples.len()
            ),
        };
    }
    let mut constants = Vec::with_capacity(samples.len());
    let mut bands = Vec::with_capacity(samples.len());
    for s in &samples {
        if !(s.grad.is_finite() && s.grad_band.is_finite() && s.grad_band >= 0.0)
            || s.grad.abs() <= s.grad_band
        {
            return AsymptoteVerdict::NoAsymptote {
                reason: format!(
                    "gradient {:.3e} at ρ={:.3} within its rounding band {:.3e}",
                    s.grad, s.rho, s.grad_band
                ),
            };
        }
        let constant = side.tail_constant(s.rho, s.grad);
        if !(constant > 0.0) {
            return AsymptoteVerdict::NoAsymptote {
                reason: format!(
                    "pencil constant ĉ={constant:.3e} at ρ={:.3} is not positive for the \
                     {side:?} rail (not a single tail)",
                    s.rho
                ),
            };
        }
        constants.push(constant);
        bands.push(side.tail_constant(s.rho, s.grad_band).abs());
    }
    let settlement = match settle_tail_constant(&constants, &bands) {
        Ok(settlement) => settlement,
        Err(reason) => return AsymptoteVerdict::NoAsymptote { reason },
    };
    if !(settlement.limit - settlement.radius > 0.0) {
        return AsymptoteVerdict::NoAsymptote {
            reason: format!(
                "settled pencil constant ĉ={:.3e} ± {:.3e} is not resolvably positive",
                settlement.limit, settlement.radius
            ),
        };
    }
    let q = match coef_step_ratio(&samples) {
        Some(q) if q.is_finite() && q >= 0.0 && q < 1.0 => q,
        _ => {
            return AsymptoteVerdict::NoAsymptote {
                reason: "coefficient moves not geometrically contracting (estimand not settling)"
                    .to_string(),
            };
        }
    };
    let last_step = samples
        .iter()
        .rev()
        .map(|s| s.coef_step_norm)
        .find(|&d| d.is_finite() && d > 0.0)
        .unwrap_or(0.0);
    let estimand_travel_bound = last_step * q / (1.0 - q);
    let value_gap = side.value_gap(rho_hat, settlement.limit + settlement.radius);
    let gradient_band = samples[samples.len() - 1].grad_band;
    if value_gap.is_finite() && value_gap <= value_gap_tol {
        AsymptoteVerdict::CertifiedAtAsymptote {
            side,
            tail_constant: settlement.limit,
            extrapolation_radius: settlement.radius,
            gradient_band,
            value_gap,
            estimand_travel_bound,
        }
    } else {
        AsymptoteVerdict::OnTailNotYetEquivalent {
            side,
            tail_constant: settlement.limit,
            extrapolation_radius: settlement.radius,
            value_gap,
            value_gap_tol,
        }
    }
}

#[cfg(test)]
mod asymptote_certificate_tests {
    use super::*;

    /// Relative rounding bound attached to the synthetic gradients below: the
    /// size of a few dozen `f64` roundings, far below any tail structure.
    const SYNTHETIC_REL_BAND: f64 = 1.0e-12;

    /// A pure lower-rail exponential tail with a KNOWN pencil constant `c`:
    /// `ρ_k = ρ0 − k·dρ` (running toward −∞), `grad_k = c·e^{ρ_k}`, and
    /// coefficient moves `‖Δβ‖_k = a·e^{ρ_k}` (decaying with the tail's own
    /// ratio `q = e^{−dρ}`).
    fn lower_tail_window(c: f64, a: f64, rho0: f64, drho: f64, n: usize) -> AsymptoteWindow {
        let mut w = AsymptoteWindow::with_capacity(n);
        for k in 0..n {
            let rho = rho0 - (k as f64) * drho;
            let grad = c * rho.exp();
            w.push(AsymptoteSample {
                rho,
                grad,
                grad_band: grad.abs() * SYNTHETIC_REL_BAND,
                coef_step_norm: a * rho.exp(),
            });
        }
        w
    }

    /// `ĉ = e^{−ρ}·grad` is recovered on a pure lower tail, and the certified
    /// value gap is the tail integral `|grad|` widened only by the settlement
    /// radius.
    #[test]
    fn lower_tail_recovers_constant_and_value_gap() {
        let c = 6723.0;
        let window = lower_tail_window(c, 1.0e-3, -7.0, 0.5, 8);
        let latest = *window.latest().unwrap();
        match assess_coordinate(&window, AsymptoteSide::Lower, latest.rho, 1.0) {
            AsymptoteVerdict::CertifiedAtAsymptote {
                side,
                tail_constant,
                extrapolation_radius,
                gradient_band,
                value_gap,
                ..
            } => {
                assert_eq!(side, AsymptoteSide::Lower);
                assert!(
                    (tail_constant - c).abs() / c < 1.0e-9,
                    "recovered ĉ={tail_constant} should equal c={c}"
                );
                assert!((tail_constant - c).abs() <= extrapolation_radius);
                assert_eq!(gradient_band, latest.grad_band);
                let g = latest.grad.abs();
                assert!(
                    value_gap >= g && value_gap <= g * (1.0 + 1.0e-9),
                    "value gap {value_gap} must bound the tail integral |grad|={g} tightly"
                );
            }
            other => panic!("expected CertifiedAtAsymptote on a pure tail, got {other:?}"),
        }
    }

    /// The value gap decides certify versus keep-stepping: the same settled
    /// tail certifies against a resolution above its gap and stays
    /// `OnTailNotYetEquivalent` (not demoted to `NoAsymptote`) below it. The
    /// estimand travel bound is the geometric tail sum of the observed moves.
    #[test]
    fn value_gap_gate_decides_certify_vs_on_tail() {
        let window = lower_tail_window(6723.0, 1.0e-3, -7.0, 0.5, 8);
        let latest = *window.latest().unwrap();
        let gap = match assess_coordinate(&window, AsymptoteSide::Lower, latest.rho, f64::MAX) {
            AsymptoteVerdict::CertifiedAtAsymptote { value_gap, .. } => value_gap,
            other => panic!("expected Certified under an unbounded resolution, got {other:?}"),
        };
        let q = (-0.5_f64).exp();
        let expected_travel = latest.coef_step_norm * q / (1.0 - q);

        match assess_coordinate(&window, AsymptoteSide::Lower, latest.rho, 2.0 * gap) {
            AsymptoteVerdict::CertifiedAtAsymptote {
                estimand_travel_bound,
                ..
            } => {
                assert!(
                    (estimand_travel_bound - expected_travel).abs() <= 1.0e-9 * expected_travel,
                    "travel bound {estimand_travel_bound} should match the geometric tail sum \
                     {expected_travel}"
                );
            }
            other => panic!("expected Certified with resolution above the gap, got {other:?}"),
        }

        match assess_coordinate(&window, AsymptoteSide::Lower, latest.rho, 0.5 * gap) {
            AsymptoteVerdict::OnTailNotYetEquivalent {
                side,
                value_gap,
                value_gap_tol,
                ..
            } => {
                assert_eq!(side, AsymptoteSide::Lower);
                assert_eq!(value_gap, gap);
                assert!(value_gap > value_gap_tol);
            }
            other => panic!("expected OnTailNotYetEquivalent below the gap, got {other:?}"),
        }
    }

    /// An upper-rail pure tail is certified on the `Upper` side: the sign
    /// convention `ĉ = −e^{+ρ}·grad` recovers a positive constant.
    #[test]
    fn upper_tail_side_and_constant() {
        let c = 42.0;
        let mut w = AsymptoteWindow::with_capacity(6);
        for k in 0..6 {
            let rho = 8.0 + (k as f64) * 0.5;
            let grad = -c * (-rho).exp();
            w.push(AsymptoteSample {
                rho,
                grad,
                grad_band: grad.abs() * SYNTHETIC_REL_BAND,
                coef_step_norm: 1.0e-4 * (-rho).exp(),
            });
        }
        let rho_hat = w.latest().unwrap().rho;
        match assess_coordinate(&w, AsymptoteSide::Upper, rho_hat, 1.0) {
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

    /// A pencil constant whose steps GROW toward the rail (the noise-dominated
    /// band beside a rail, or a curved region) is not a settled tail.
    #[test]
    fn drifting_constant_is_rejected_as_not_settling() {
        let rhos = [-9.0_f64, -9.5, -10.0, -10.5];
        let cs = [6723.0_f64, 6731.0, 9111.0, 4200.0];
        let mut w = AsymptoteWindow::with_capacity(4);
        for (&rho, &c) in rhos.iter().zip(cs.iter()) {
            let grad = c * rho.exp();
            w.push(AsymptoteSample {
                rho,
                grad,
                grad_band: grad.abs() * SYNTHETIC_REL_BAND,
                coef_step_norm: 1.0e-3 * rho.exp(),
            });
        }
        match assess_coordinate(&w, AsymptoteSide::Lower, -10.5, f64::MAX) {
            AsymptoteVerdict::NoAsymptote { reason } => {
                assert!(reason.contains("settling"), "should reject as unsettled: {reason}");
            }
            other => panic!("drifting ĉ must not certify, got {other:?}"),
        }
    }

    /// A gradient inside its own rounding band carries no sign, so it cannot
    /// witness a tail, whatever its absolute size.
    #[test]
    fn gradient_inside_its_band_is_not_an_asymptote() {
        let mut w = AsymptoteWindow::new();
        for k in 0..4 {
            w.push(AsymptoteSample {
                rho: 1.0 + k as f64 * 0.1,
                grad: 1.0e-12,
                grad_band: 1.0e-10,
                coef_step_norm: 1.0e-9,
            });
        }
        match assess_coordinate(&w, AsymptoteSide::Lower, 1.3, f64::MAX) {
            AsymptoteVerdict::NoAsymptote { reason } => {
                assert!(reason.contains("rounding band"), "{reason}");
            }
            other => panic!("an unresolved gradient must not be an asymptote, got {other:?}"),
        }
    }

    /// The exp4_rail.py tail, fed as verified `(ρ, ∂V/∂ρ)` literals whose
    /// finite-difference gradients are resolved to `5·10⁻⁴` relative:
    ///
    /// * on the confirmed rows `ρ ∈ {14,…,24}` the pencil constant settles at
    ///   `≈ 6723` and the certificate holds with the limit inside its radius;
    /// * on the noise floor `ρ ∈ {28,30,32}` (`ĉ ≈ 6577, 9112, −11221`) the
    ///   certificate refuses.
    #[test]
    fn exp4_verified_tail_certifies_and_noise_floor_refuses() {
        const FD_REL_BAND: f64 = 5.0e-4;
        let confirmed: [(f64, f64); 6] = [
            (14.0, -5.589576e-03),
            (16.0, -7.565987e-04),
            (18.0, -1.023966e-04),
            (20.0, -1.385843e-05),
            (22.0, -1.874980e-06),
            (24.0, -2.538059e-07),
        ];
        let coef = [6.196e-05, 8.398e-06, 1.137e-06, 1.539e-07, 2.082e-08, 2.818e-09];
        let mut tail = AsymptoteWindow::with_capacity(confirmed.len());
        for (&(rho, grad), &step) in confirmed.iter().zip(coef.iter()) {
            tail.push(AsymptoteSample {
                rho,
                grad,
                grad_band: grad.abs() * FD_REL_BAND,
                coef_step_norm: step,
            });
        }
        match assess_coordinate(&tail, AsymptoteSide::Upper, 24.0, 1.0) {
            AsymptoteVerdict::CertifiedAtAsymptote {
                side,
                tail_constant,
                extrapolation_radius,
                ..
            } => {
                assert_eq!(side, AsymptoteSide::Upper);
                assert!(
                    (tail_constant - 6723.0).abs() <= extrapolation_radius,
                    "exp4 pencil constant 6723 must lie within {tail_constant} ± \
                     {extrapolation_radius}"
                );
            }
            other => panic!("exp4 confirmed-tail rows must certify, got {other:?}"),
        }

        let noise: [(f64, f64); 3] = [
            (28.0, -4.547474e-09),
            (30.0, -8.526513e-10),
            (32.0, 1.421085e-10),
        ];
        let mut floor = AsymptoteWindow::with_capacity(noise.len());
        for &(rho, grad) in noise.iter() {
            floor.push(AsymptoteSample {
                rho,
                grad,
                grad_band: grad.abs() * FD_REL_BAND,
                coef_step_norm: 1.0e-11,
            });
        }
        match assess_coordinate(&floor, AsymptoteSide::Upper, 32.0, f64::MAX) {
            AsymptoteVerdict::NoAsymptote { .. } => {}
            other => panic!("the FD noise floor must NOT certify as a tail, got {other:?}"),
        }
    }

    /// Too few samples never certify (a snapshot cannot confirm a tail).
    #[test]
    fn too_few_samples_never_certify() {
        let w = lower_tail_window(6723.0, 1.0e-3, -7.0, 0.5, 2);
        match assess_coordinate(&w, AsymptoteSide::Lower, -7.5, f64::MAX) {
            AsymptoteVerdict::NoAsymptote { reason } => {
                assert!(reason.contains("too few samples"), "{reason}");
            }
            other => panic!("two samples must not certify, got {other:?}"),
        }
    }

    /// The verdict is invariant to the criterion's units: scaling every
    /// gradient, its band and the criterion resolution by `s` scales the
    /// constant, radius and gap by `s` and changes nothing else.
    #[test]
    fn verdict_is_invariant_to_criterion_scale() {
        let base = lower_tail_window(6723.0, 1.0e-3, -7.0, 0.5, 8);
        let rho_hat = base.latest().unwrap().rho;
        let reference = assess_coordinate(&base, AsymptoteSide::Lower, rho_hat, f64::MAX);
        let AsymptoteVerdict::CertifiedAtAsymptote {
            tail_constant: c_ref,
            value_gap: gap_ref,
            ..
        } = reference
        else {
            panic!("reference tail must certify, got {reference:?}");
        };
        for s in [1.0e-9, 1.0, 1.0e6] {
            let mut scaled = AsymptoteWindow::with_capacity(8);
            for sample in base.samples() {
                scaled.push(AsymptoteSample {
                    grad: sample.grad * s,
                    grad_band: sample.grad_band * s,
                    ..*sample
                });
            }
            for (tol_factor, certifies) in [(2.0, true), (0.5, false)] {
                let verdict =
                    assess_coordinate(&scaled, AsymptoteSide::Lower, rho_hat, tol_factor * gap_ref * s);
                match (verdict, certifies) {
                    (AsymptoteVerdict::CertifiedAtAsymptote { tail_constant, .. }, true)
                    | (AsymptoteVerdict::OnTailNotYetEquivalent { tail_constant, .. }, false) => {
                        assert!(
                            (tail_constant / s - c_ref).abs() <= 1.0e-12 * c_ref,
                            "scale {s}: ĉ/s={} vs {c_ref}",
                            tail_constant / s
                        );
                    }
                    (other, _) => panic!("scale {s}, factor {tol_factor}: got {other:?}"),
                }
            }
        }
    }

    /// A tail with a live next-order term, `ĉ(ρ) = c + d·e^{−ρ}` toward the
    /// upper rail, settles to a radius that covers the true limit `c`.
    #[test]
    fn settlement_radius_covers_next_order_limit() {
        let (c, d) = (7.1, 400.0);
        let constants: Vec<f64> = [7.0_f64, 8.0, 9.0]
            .iter()
            .map(|&rho| c + d * (-rho).exp())
            .collect();
        let bands = vec![SYNTHETIC_REL_BAND; constants.len()];
        let settled = settle_tail_constant(&constants, &bands).expect("geometric tail settles");
        assert_eq!(settled.limit, constants[2]);
        assert!(
            (settled.limit - c).abs() <= settled.radius,
            "limit {c} must lie within {} ± {}",
            settled.limit,
            settled.radius
        );
        // The bound is tight: the exact remaining series is d·e^{−9}, and the
        // radius exceeds it only by the band terms.
        assert!(settled.radius <= d * (-9.0_f64).exp() * (1.0 + 1.0e-6));
    }
}
