//! #1011 — deterministic two-sided enclosures for a block-SPD log-determinant.
//!
//! For the bordered-arrow evidence at frontier atom counts, the dense border
//! Schur factor is the scaling wall. This module computes CERTIFIED bounds
//! `lower ≤ log|S| − log|D| ≤ upper` from exact moments of the
//! block-preconditioned residual, with no randomness and no estimator
//! variance — an enclosure, refinable until it is tighter than the consuming
//! decision's margin (topology-race Δ, EFS step tolerance, …).
//!
//! Math (derivation on issue #1011): with `D = blockdiag(S_11..S_KK)`,
//! `S_ii = L_i L_iᵀ`, and `E = D^{-1/2}(S − D)D^{-1/2}`:
//! * `I + E = D^{-1/2} S D^{-1/2} ≻ 0` ⇒ every eigenvalue `λ_a > −1`;
//! * `E` has ZERO diagonal blocks ⇒ `tr E = Σ λ_a = 0`;
//! * `p₂ = tr E² = Σ_{i≠j} ‖Ẽ_ij‖_F²` and
//!   `p₃ = tr E³ = Σ_{i≠j≠k≠i} tr(Ẽ_ij Ẽ_jk Ẽ_ki)` are EXACT block
//!   contractions of `Ẽ_ij = L_i⁻¹ S_ij L_j⁻ᵀ` — never forming `E` densely;
//! * a spectral-radius certificate `ρ = min(√p₂, max_i Σ_{j≠i} ‖Ẽ_ij‖_F)`
//!   (block Gershgorin via `‖·‖₂ ≤ ‖·‖_F`), required `< 1`.
//!
//! Per-eigenvalue inequalities, valid for ALL `λ > −1` (alternating-series
//! remainder for `λ ≥ 0`, monotone tail for `λ < 0`):
//! `log(1+λ) ≤ λ − λ²/2 + λ³/3`, and on `[−ρ, ρ]` the cubic remainder obeys
//! `R(λ) ≥ −ρ²λ²/(4(1−ρ))`. Summing with `Σλ = 0`:
//!
//! ```text
//! order 3:  upper = −p₂/2 + p₃/3
//!           lower = upper − ρ²·p₂ / (4(1−ρ))
//! order 2:  upper = −p₂/2 + ρ·p₂/3          (λ³ ≤ ρλ² for λ≥0; λ³<0≤ρλ² else)
//!           lower = −p₂/2 − ρ·p₂ / (3(1−ρ))
//! ```
//!
//! The gap scales as `ρ·p₂` (order 2) / `ρ²·p₂` (order 3): preconditioner
//! quality drives certainty, and absorbing the worst off-diagonal pair into
//! `D` is the refinement step when the gap is too wide. `ρ ≥ 1` is an
//! explicit refusal (`Err`), never a silent fallback.

/// A certified enclosure of `log|S|` for a block-partitioned SPD matrix.
#[derive(Debug, Clone)]
pub struct LogdetEnclosure {
    /// Exact `log|D| = Σ_i log|S_ii|` from the per-block Cholesky factors.
    pub block_diag_logdet: f64,
    /// Certified lower bound on `log|S|` (i.e. `block_diag_logdet + correction_lower`).
    pub lower: f64,
    /// Certified upper bound on `log|S|`.
    pub upper: f64,
    /// The spectral-radius certificate used (`< 1` or this struct would not exist).
    pub rho: f64,
    /// Exact second moment `tr(E²)`.
    pub p2: f64,
    /// Exact third moment `tr(E³)` when the order-3 enclosure was requested.
    pub p3: Option<f64>,
}

impl LogdetEnclosure {
    /// Width of the enclosure — compare against the consuming decision's margin.
    pub fn gap(&self) -> f64 {
        self.upper - self.lower
    }

    /// The enclosure's certified point value when (and only when) the gap is
    /// already below the consuming decision's margin. Below margin a single
    /// `f64` is a lie — the caller must escalate — so this returns the explicit
    /// [`MarginVerdict`] rather than ever fabricating one.
    ///
    /// `decision_margin` is the smallest spread in the consumer's verdict that
    /// matters: the topology-race candidate gap Δ, the EFS step tolerance, an
    /// Armijo slack. A `Decided` value is the enclosure midpoint, which is
    /// within `gap/2 ≤ margin/2` of the truth — tighter than the decision can
    /// resolve, so the verdict is identical to the one the exact logdet would
    /// have produced.
    /// Whether a bare enclosure `gap` is resolved more tightly than a consumer's
    /// `decision_margin` — the predicate behind [`Self::decide_within_margin`],
    /// exposed for consumers that hold only the gap (e.g. the EFS engine, which
    /// receives the cost's enclosure width through `EfsEval`).
    pub fn gap_resolves_margin(gap: f64, decision_margin: f64) -> bool {
        decision_margin.is_finite()
            && decision_margin > 0.0
            && gap.is_finite()
            && gap < decision_margin
    }

    pub fn decide_within_margin(&self, decision_margin: f64) -> MarginVerdict {
        let gap = self.gap();
        if decision_margin.is_finite() && decision_margin > 0.0 && gap < decision_margin {
            MarginVerdict::Decided {
                value: 0.5 * (self.lower + self.upper),
                gap,
                decision_margin,
            }
        } else {
            MarginVerdict::InsufficientMargin {
                gap,
                decision_margin,
            }
        }
    }
}

/// The shared decision-margin contract between an enclosure-valued quantity and
/// its consumer (the topology race, the EFS outer step, the coreset race
/// transfer — all declare a margin and inherit this verdict).
///
/// `Decided` means the enclosure is strictly tighter than the consumer's
/// decision margin, so its midpoint is interchangeable with the exact value for
/// that decision. `InsufficientMargin` is the honesty escalation: the consumer
/// must refine (more moments, pair absorption, a larger coreset) or fall back
/// to the exact dense path — never decide on a point value that the enclosure
/// does not actually pin down.
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum MarginVerdict {
    Decided {
        value: f64,
        gap: f64,
        decision_margin: f64,
    },
    InsufficientMargin {
        gap: f64,
        decision_margin: f64,
    },
}

impl MarginVerdict {

}

