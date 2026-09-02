//! KL rate-certificate module (#2337 §9 step 7).
//!
//! This module observes the outer smoothing-parameter loop's stream of
//! *accepted-step objective decreases* and answers one question with a
//! certificate rather than a heuristic: **will this loop reach its target
//! tolerance within the remaining iteration budget, and if not, is that
//! because it is converging too slowly or because it is defective?**
//!
//! The distinction matters. A loop that is *provably converging but slow*
//! deserves a `RateCertified` refusal that names the forecast — the caller
//! can raise the budget. A loop whose accepted steps contradict the solver's
//! own monotone-descent contract is *defective*: no budget will save it, and
//! we say so (`KlInconsistent`) from an **exact theorem**, never from a noisy
//! rate fit.
//!
//! # Why a rate can be *named*, not merely assumed
//!
//! The outer criterion this engine certifies (the REML/Laplace objective on
//! the certification tube) is real-analytic there — it is `C^ω`, see #2337
//! Thm 5.3. A real-analytic function satisfies the **Łojasiewicz gradient
//! inequality**
//!
//! ```text
//!     ‖∇V(x)‖ ≥ c · |V(x) − V*|^θ ,   θ ∈ [1/2, 1),
//! ```
//!
//! on a neighborhood of any critical point `x*`, with some exponent `θ` and
//! constant `c > 0`. Existence of `θ` is therefore *guaranteed*; this module
//! only *names* it from the observed decreases. It never assumes convergence
//! — a bad fit yields [`LoopVerdict::InsufficientData`], and a contract
//! violation yields [`LoopVerdict::KlInconsistent`] from a defect theorem.
//!
//! # From the Łojasiewicz exponent to the observable decrease slope
//!
//! Let `e_k = V_k − V*` be the optimality gap (`e_k → 0`, monotone
//! decreasing under a descent method) and `d_k = e_k − e_{k+1} = V_k − V_{k+1}`
//! the accepted-step decrease. Assume **sufficient decrease**
//!
//! ```text
//!     d_k ≥ a · ‖∇V_k‖²                                          (SD)
//! ```
//!
//! (Armijo / trust-region / MM all provide (SD) with some `a > 0`).
//! Combining (SD) with Łojasiewicz gives the scalar recurrence
//!
//! ```text
//!     e_k − e_{k+1} ≥ a c² · e_k^{2θ}.                            (R)
//! ```
//!
//! **Case θ = 1/2.** (R) reads `e_k − e_{k+1} ≥ a c² e_k`, i.e.
//! `e_{k+1} ≤ (1 − a c²) e_k` — *linear* (geometric) convergence,
//! `e_k ≍ r^k` with `r = 1 − a c² ∈ (0,1)`. Since
//! `d_k = e_k(1 − e_{k+1}/e_k) ≍ e_k`, the *decreases are geometric too*:
//! `d_k ≍ r^k`. This is the `Geometric` model.
//!
//! **Case θ ∈ (1/2, 1).** Treat (R) as the continuum ODE
//! `ė = −C e^{2θ}` with `2θ > 1`. Then
//! `d/dk (e^{1−2θ}) = (1−2θ) e^{−2θ} ė = C(2θ−1) > 0`, so
//! `e_k^{1−2θ} ≍ C(2θ−1) k`, giving the sublinear gap
//!
//! ```text
//!     e_k ≍ k^{−1/(2θ−1)}.
//! ```
//!
//! Differentiating, `d_k ≍ −de/dk ≍ k^{−(1/(2θ−1) + 1)} = k^{−s}` with the
//! **observable decrease slope**
//!
//! ```text
//!     s = 1/(2θ−1) + 1 = 2θ/(2θ−1).                              (S)
//! ```
//!
//! Note the valid range: for `θ ∈ (1/2, 1)`, (S) maps to `s ∈ (2, ∞)`
//! (`θ=3/4 ↦ s=3`, `θ=5/6 ↦ s=2.5`, `θ→1 ↦ s→2`, `θ→1/2⁺ ↦ s→∞`).
//! The gap exponent `p = 1/(2θ−1) = s − 1 ∈ (1, ∞)`, so the gap is always
//! summable — the telescoped forecast below is well-defined.
//!
//! # Sign convention for θ̂ — reconciled explicitly
//!
//! We fit `log d_k` linearly against `log k` by least squares and read off
//! the **raw slope** `ŝ_raw`. Because `d_k` *decreases*, `ŝ_raw < 0`, and its
//! magnitude is the `s` of (S): `s = −ŝ_raw`. Inverting (S) for `θ`:
//!
//! ```text
//!     s = 2θ/(2θ−1)  ⟹  s(2θ−1) = 2θ  ⟹  2θ(s−1) = s
//!                    ⟹  θ = s / (2s − 2).                        (I)
//! ```
//!
//! Substituting the *raw* (negative) slope `s = −ŝ_raw` into (I):
//!
//! ```text
//!     θ = (−ŝ_raw) / (2(−ŝ_raw) − 2)
//!       = (−ŝ_raw) / (−2 ŝ_raw − 2)
//!       = ŝ_raw / (2 ŝ_raw + 2).                                 (I')
//! ```
//!
//! So **`θ̂ = ŝ_raw / (2 ŝ_raw + 2)`** with `ŝ_raw` the raw (negative)
//! log-log slope — this is exactly the parametrization the #2337 theory doc
//! records as `θ̂ = ŝ/(2ŝ+2)`. We therefore store the **raw negative slope**
//! in [`RateModel::Power::exponent_s`] and compute `kl_theta` via (I').
//!
//! Sanity: `ŝ_raw = −3` (i.e. `d_k ≍ k^{−3}`, the `f(x)=x⁴` case, `θ=3/4`)
//! gives `θ̂ = −3/(−6+2) = 0.75`. `ŝ_raw = −2.5` (the `f(x)=x⁶` case,
//! `θ=5/6`) gives `θ̂ = −2.5/(−5+2) = 0.8333`. ✓
//!
//! # Forecasts
//!
//! Given a current gap bound `e` and target `tol` (`0 < tol < e`):
//!
//! * **Geometric** (`e_k ≍ e·r^n`): `e·r^N ≤ tol ⟺ N ≥ log(tol/e)/log(r)`,
//!   so `N̂ = log(tol/e) / log(r)` (both logs negative, `N̂ > 0`).
//!
//! * **Power** (`e_k ≍ C k^{−p}`, `p = s − 1`, `s = −exponent_s`): with the
//!   loop currently at iteration `k_now`, `e = C k_now^{−p}` and we need
//!   `C k_target^{−p} ≤ tol`. Dividing, `(k_target/k_now)^{−p} = tol/e`, so
//!   `k_target = k_now (e/tol)^{1/p}` and the *additional* iterations are
//!
//!   ```text
//!       N̂ = k_target − k_now = k_now · ((e/tol)^{1/p} − 1),
//!       p = s − 1 = (−exponent_s) − 1.                           (F)
//!   ```
//!
//!   Convention reconciliation: the #2337 task sketch wrote the power
//!   forecast exponent as `1/(s_pos+1)`. That does not survive the
//!   telescoping derivation — the gap exponent is `p = s − 1` (the gap is one
//!   power *shallower* than the decreases `d_k ≍ k^{−s}`, because
//!   `d_k = −de/dk`), so the correct forecast exponent is `1/(s−1)`, which is
//!   what (F) uses. We flag this as the derivation-correct form.
//!
//! `Grant` iff `N̂ ≤ budget`; otherwise `RateCertified` (a provable-but-slow
//! refusal carrying the forecast). A non-convergent fit (geometric `r ≥ 1`,
//! or power `p ≤ 0`) is treated as an *uninformative* fit —
//! [`LoopVerdict::InsufficientData`], never a defect claim.
//!
//! # Defect theorems (exact — not fits)
//!
//! See [`monotonicity_defect`] and [`energy_budget_defect`]. These are the
//! *only* sources of [`LoopVerdict::KlInconsistent`]: they are proofs, valid
//! independent of any rate model.

use std::collections::VecDeque;

use gam_linalg::utils::KahanSum;

/// Default ring-buffer capacity `W` for [`DecreaseWindow`].
///
/// Chosen to hold enough recent accepted steps for a stable two-parameter
/// log/log-log least-squares fit while staying local to the current basin.
pub const DEFAULT_WINDOW_CAPACITY: usize = 24;

/// One accepted outer step's contribution to the decrease record.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct DecreaseEntry {
    /// Outer iteration index `k` at which this step was accepted. Must be
    /// `≥ 1` to participate in the power (`log k`) fit.
    pub iter_index: u64,
    /// Signed objective decrease `d_k = V_k − V_{k+1}`. Positive on genuine
    /// descent; a non-positive value records an accepted-step *increase* and
    /// is the raw material of the monotonicity defect theorem.
    pub decrease: f64,
    /// Squared step norm `‖x_{k+1} − x_k‖²` for this accepted step. Feeds the
    /// energy-budget accumulator.
    pub step_norm_sq: f64,
}

/// A fixed-capacity deterministic ring buffer of accepted-step decreases.
///
/// The ring holds the most recent `W` entries (for the *local* rate fit),
/// while two compensated (Kahan) accumulators track the *lifetime* totals of
/// decrease and squared step norm — these never evict, so they remain valid
/// inputs to the telescoped energy budget over the whole run.
///
/// Determinism: `VecDeque` preserves insertion order; eviction is strictly
/// FIFO; the Kahan accumulators are updated in push order. No randomness, no
/// unordered iteration.
#[derive(Clone, Debug)]
pub struct DecreaseWindow {
    capacity: usize,
    ring: VecDeque<DecreaseEntry>,
    total_decrease: KahanSum,
    total_step_norm_sq: KahanSum,
    observed_count: u64,
}

impl Default for DecreaseWindow {
    fn default() -> Self {
        Self::with_capacity(DEFAULT_WINDOW_CAPACITY)
    }
}

impl DecreaseWindow {
    /// A window with the [`DEFAULT_WINDOW_CAPACITY`].
    pub fn new() -> Self {
        Self::default()
    }

    /// A window with an explicit ring capacity `W`. A capacity of zero is
    /// promoted to one so the ring can always hold the most recent step.
    pub fn with_capacity(capacity: usize) -> Self {
        let capacity = capacity.max(1);
        Self {
            capacity,
            ring: VecDeque::with_capacity(capacity),
            total_decrease: KahanSum::default(),
            total_step_norm_sq: KahanSum::default(),
            observed_count: 0,
        }
    }

    /// Record one accepted step. `decrease` is signed (`V_k − V_{k+1}`);
    /// `step_norm_sq` is `‖x_{k+1} − x_k‖² ≥ 0`. The lifetime accumulators are
    /// updated first (they see every step), then the ring evicts its oldest
    /// entry if full.
    pub fn push(&mut self, iter_index: u64, decrease: f64, step_norm_sq: f64) {
        self.total_decrease.add(decrease);
        self.total_step_norm_sq.add(step_norm_sq);
        self.observed_count += 1;
        if self.ring.len() == self.capacity {
            self.ring.pop_front();
        }
        self.ring.push_back(DecreaseEntry {
            iter_index,
            decrease,
            step_norm_sq,
        });
    }

    /// The ring's capacity `W`.
    pub fn capacity(&self) -> usize {
        self.capacity
    }

    /// Number of entries currently retained in the ring (`≤ W`).
    pub fn len(&self) -> usize {
        self.ring.len()
    }

    /// Whether the ring is empty.
    pub fn is_empty(&self) -> bool {
        self.ring.is_empty()
    }

    /// The retained entries, oldest to newest.
    pub fn entries(&self) -> impl Iterator<Item = &DecreaseEntry> {
        self.ring.iter()
    }

    /// Lifetime total decrease `Σ_k d_k` (compensated).
    pub fn total_decrease(&self) -> f64 {
        self.total_decrease.sum()
    }

    /// Lifetime total squared step norm `Σ_k ‖x_{k+1} − x_k‖²` (compensated).
    pub fn total_step_norm_sq(&self) -> f64 {
        self.total_step_norm_sq.sum()
    }

    /// Total number of accepted steps ever recorded (includes evicted ones).
    pub fn observed_count(&self) -> u64 {
        self.observed_count
    }

}

/// A fitted decrease-rate model with its log-space residual sum of squares.
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum RateModel {
    /// Geometric decreases `d_k ≍ ratio^k` (Łojasiewicz exponent `θ = 1/2`,
    /// linear convergence of the gap). `ratio = exp(slope of log d_k vs k)`.
    Geometric {
        /// Geometric ratio `r = exp(m)`, `m` = slope of `log d_k` against `k`.
        ratio: f64,
        /// Residual sum of squares of the `log d_k`-vs-`k` fit.
        resid: f64,
    },
    /// Power-law decreases `d_k ≍ k^{exponent_s}` (Łojasiewicz exponent
    /// `θ ∈ (1/2, 1)`, sublinear convergence). `exponent_s` is the **raw
    /// (negative) log-log slope** `ŝ_raw`; `kl_theta = ŝ_raw/(2ŝ_raw+2)` per
    /// (I') in the module docs.
    Power {
        /// Raw log-log slope `ŝ_raw = slope of log d_k vs log k` (negative on
        /// a converging loop; magnitude `s = −exponent_s ∈ (2, ∞)`).
        exponent_s: f64,
        /// Recovered Łojasiewicz exponent `θ̂ = ŝ_raw / (2ŝ_raw + 2)`.
        kl_theta: f64,
        /// Residual sum of squares of the `log d_k`-vs-`log k` fit.
        resid: f64,
    },
}

impl RateModel {
    /// The log-space residual sum of squares of this fit (used for model
    /// selection).
    pub fn resid(&self) -> f64 {
        match self {
            RateModel::Geometric { resid, .. } => *resid,
            RateModel::Power { resid, .. } => *resid,
        }
    }
}

/// The certificate/refusal returned by [`assess`].
#[derive(Clone, Debug, PartialEq)]
pub enum LoopVerdict {
    /// The winning rate model forecasts reaching `tol` within budget.
    Grant {
        /// Forecast additional iterations `N̂` to reach `tol`.
        forecast_iters: f64,
        /// The winning rate model.
        model: RateModel,
    },
    /// Provably converging (a valid rate was named) but the forecast exceeds
    /// the budget — a refusal that carries the evidence so the caller can
    /// raise the budget rather than abandon the loop.
    RateCertified {
        /// Forecast additional iterations `N̂` to reach `tol`.
        forecast_iters: f64,
        /// The winning rate model.
        model: RateModel,
    },
    /// A defect theorem fired: the accepted-step stream contradicts the
    /// loop's own descent contract. Never produced by a mere bad fit.
    KlInconsistent {
        /// Human-readable proof-of-defect explanation.
        reason: String,
    },
    /// Not enough (or too degenerate) data to name a rate. Includes the case
    /// of a fit that names a *non-convergent* model.
    InsufficientData,
}

#[cfg(test)]
mod kl_certificate_tests {
    use super::*;

    /// Push a synthetic geometric decrease sequence `d_k = d0 · r^k` into a
    /// fresh window over iterations `1..=n`.
    fn geometric_window(d0: f64, r: f64, n: u64) -> DecreaseWindow {
        let mut w = DecreaseWindow::new();
        for k in 1..=n {
            let d = d0 * r.powi(k as i32);
            w.push(k, d, d); // step_norm_sq unused by these asserts
        }
        w
    }

    /// (a) A geometric sequence with ratio 0.994 is recovered as Geometric,
    /// with the forecast/Grant/RateCertified behavior mirroring exp3.
    #[test]
    fn geometric_ratio_recovered_and_budget_decides() {
        let r = 0.994_f64;
        let window = geometric_window(1.0, r, 24);

        let model = fit_rate(&window).expect("geometric fit");
        match model {
            RateModel::Geometric { ratio, .. } => {
                assert!(
                    (ratio - r).abs() < 1.0e-3,
                    "recovered ratio {ratio} should be within 1e-3 of {r}"
                );
            }
            other => panic!("expected Geometric, got {other:?}"),
        }

        // Forecast setup mirroring exp3: gap e=1.0, tol chosen so N̂ ≈ 599.
        // N̂ = log(tol/e)/log(r). With r=0.994, log(r)=-6.018e-3; picking
        // tol = exp(599·log r) gives N̂ = 599 exactly.
        let e = 1.0_f64;
        let n_target = 599.0_f64;
        let tol = (n_target * r.ln()).exp(); // ≈ 2.72e-2

        // Grant when the budget clears the forecast.
        match assess(&window, e, tol, 600.0) {
            LoopVerdict::Grant {
                forecast_iters, ..
            } => {
                assert!(
                    (forecast_iters - n_target).abs() < 1.0,
                    "forecast {forecast_iters} should be ≈ {n_target}"
                );
            }
            other => panic!("expected Grant at budget 600, got {other:?}"),
        }

        // RateCertified (provable-but-slow refusal) when the budget is below
        // the forecast.
        match assess(&window, e, tol, 550.0) {
            LoopVerdict::RateCertified {
                forecast_iters, ..
            } => {
                assert!(
                    (forecast_iters - n_target).abs() < 1.0,
                    "refusal forecast {forecast_iters} should be ≈ {n_target}"
                );
            }
            other => panic!("expected RateCertified at budget 550, got {other:?}"),
        }
    }

    /// Deterministic gradient descent on `f(x) = x^m` from `x0 = 1`.
    /// Returns the accepted decreases `d_k = f(x_k) − f(x_{k+1})` for
    /// `k = 0 …` alongside the iteration indices `k+1`.
    fn power_descent_decreases(m: i32, eta: f64, steps: usize) -> Vec<(u64, f64)> {
        let mut x = 1.0_f64;
        let f = |x: f64| x.powi(m);
        let grad = |x: f64| (m as f64) * x.powi(m - 1);
        let mut out = Vec::with_capacity(steps);
        for k in 0..steps {
            let fx = f(x);
            let x_next = x - eta * grad(x);
            let fx_next = f(x_next);
            let d = fx - fx_next;
            out.push(((k as u64) + 1, d));
            x = x_next;
        }
        out
    }

    /// Build a window from a deterministic subsample of a descent sequence:
    /// indices `start, start+step, …` for `count` points (wide log-k leverage
    /// deep in the asymptotic regime).
    fn subsampled_window(seq: &[(u64, f64)], start: usize, step: usize, count: usize) -> DecreaseWindow {
        let mut w = DecreaseWindow::with_capacity(count);
        for j in 0..count {
            let idx = start + j * step;
            let (iter, d) = seq[idx];
            w.push(iter, d, 0.0);
        }
        w
    }

    /// (b) Gradient descent on x⁴ (Łojasiewicz θ = 3/4). The recovered θ̂ must
    /// land in (0.72, 0.78) and the model must be selected as Power.
    #[test]
    fn power_theta_recovered_x4() {
        let seq = power_descent_decreases(4, 0.01, 3200);
        // Subsample deep-asymptotic indices 800, 900, …, 3100 (24 points).
        let window = subsampled_window(&seq, 799, 100, 24);
        let model = fit_rate(&window).expect("power fit x4");
        match model {
            RateModel::Power { kl_theta, .. } => {
                assert!(
                    kl_theta > 0.72 && kl_theta < 0.78,
                    "θ̂ = {kl_theta} should be in (0.72, 0.78) for x⁴ (θ=3/4)"
                );
            }
            other => panic!("expected Power for x⁴ descent, got {other:?}"),
        }
    }

    /// (c) Gradient descent on x⁶ (Łojasiewicz θ = 5/6 ≈ 0.833). θ̂ ∈ (0.80,
    /// 0.87), selected as Power.
    #[test]
    fn power_theta_recovered_x6() {
        let seq = power_descent_decreases(6, 0.01, 3200);
        let window = subsampled_window(&seq, 799, 100, 24);
        let model = fit_rate(&window).expect("power fit x6");
        match model {
            RateModel::Power { kl_theta, .. } => {
                assert!(
                    kl_theta > 0.80 && kl_theta < 0.87,
                    "θ̂ = {kl_theta} should be in (0.80, 0.87) for x⁶ (θ=5/6)"
                );
            }
            other => panic!("expected Power for x⁶ descent, got {other:?}"),
        }
    }

    /// (d) An oscillating sequence (an accepted-step increase) triggers the
    /// monotonicity defect, and `assess` surfaces it as `KlInconsistent`.
    #[test]
    fn oscillation_triggers_monotonicity_defect() {
        let mut window = DecreaseWindow::new();
        window.push(1, 0.10, 0.01);
        window.push(2, 0.05, 0.01);
        window.push(3, -0.02, 0.01); // accepted-step INCREASE
        window.push(4, 0.03, 0.01);

        let reason = monotonicity_defect(&window, 1.0e-9).expect("defect must fire");
        assert!(
            reason.contains("iter 3"),
            "defect should name the offending iter 3: {reason}"
        );

        match assess(&window, 1.0, 1.0e-3, 1.0e6) {
            LoopVerdict::KlInconsistent { reason } => {
                assert!(reason.contains("monotonicity defect"), "{reason}");
            }
            other => panic!("expected KlInconsistent from oscillation, got {other:?}"),
        }

        // A clean monotone window must NOT fire the defect.
        let clean = geometric_window(1.0, 0.9, 10);
        assert!(
            monotonicity_defect(&clean, 1.0e-9).is_none(),
            "monotone window must not report a defect"
        );
    }

    /// (e) A total step-energy exceeding the sufficient-decrease budget
    /// triggers the energy-budget defect; a within-budget total does not.
    #[test]
    fn energy_budget_violation_triggers_defect() {
        // Budget = (V0 − V_lb)/a = (1.0 − 0.0)/1.0 = 1.0; total 10.0 > 1.0.
        let reason = energy_budget_defect(10.0, 1.0, 0.0, 1.0).expect("defect must fire");
        assert!(reason.contains("energy-budget defect"), "{reason}");

        // Within budget: no defect.
        assert!(
            energy_budget_defect(0.5, 1.0, 0.0, 1.0).is_none(),
            "within-budget energy must not report a defect"
        );
        // Hypotheses unmet (a ≤ 0): no defect asserted.
        assert!(
            energy_budget_defect(1.0e9, 1.0, 0.0, 0.0).is_none(),
            "a ≤ 0 is outside the theorem; no defect"
        );
    }

    /// A too-short window yields `InsufficientData`, never a spurious verdict.
    #[test]
    fn insufficient_data_when_window_too_short() {
        let mut window = DecreaseWindow::new();
        window.push(1, 0.1, 0.0);
        window.push(2, 0.05, 0.0);
        assert!(fit_rate(&window).is_none());
        assert_eq!(
            assess(&window, 1.0, 1.0e-3, 1.0e6),
            LoopVerdict::InsufficientData
        );
    }

    /// Lifetime Kahan accumulators track totals across ring eviction.
    #[test]
    fn lifetime_accumulators_survive_eviction() {
        let mut window = DecreaseWindow::with_capacity(2);
        window.push(1, 0.5, 4.0);
        window.push(2, 0.25, 1.0);
        window.push(3, 0.125, 0.25); // evicts iter 1 from the ring
        assert_eq!(window.len(), 2);
        assert_eq!(window.observed_count(), 3);
        assert!((window.total_decrease() - 0.875).abs() < 1.0e-12);
        assert!((window.total_step_norm_sq() - 5.25).abs() < 1.0e-12);
        // Ring retains only the two most recent entries.
        let iters: Vec<u64> = window.entries().map(|e| e.iter_index).collect();
        assert_eq!(iters, vec![2, 3]);
    }
}
