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

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::Array2;

    /// Deterministic block-SPD fixture: strong SPD diagonal blocks, weak
    /// off-diagonal coupling (so the certificate ρ < 1 holds), assembled
    /// densely for the oracle.
    fn fixture(
        k: usize,
        m: usize,
        coupling: f64,
    ) -> (
        Vec<Array2<f64>>,
        Vec<(usize, usize, Array2<f64>)>,
        Array2<f64>,
    ) {
        let dim = k * m;
        let mut dense = Array2::<f64>::zeros((dim, dim));
        let mut diag = Vec::new();
        let mut off = Vec::new();
        for i in 0..k {
            let mut d = Array2::<f64>::zeros((m, m));
            for r in 0..m {
                for c in 0..m {
                    let v = if r == c {
                        3.0 + 0.4 * (i as f64) + 0.2 * (r as f64)
                    } else {
                        0.3 * ((r + 2 * c + i) as f64 * 0.7).sin()
                    };
                    d[[r, c]] = v;
                }
            }
            // Symmetrize and make diagonally dominant ⇒ SPD.
            let mut sym = Array2::<f64>::zeros((m, m));
            for r in 0..m {
                for c in 0..m {
                    sym[[r, c]] = 0.5 * (d[[r, c]] + d[[c, r]]);
                }
                sym[[r, r]] += 1.0;
            }
            for r in 0..m {
                for c in 0..m {
                    dense[[i * m + r, i * m + c]] = sym[[r, c]];
                }
            }
            diag.push(sym);
        }
        for i in 0..k {
            for j in (i + 1)..k {
                let mut b = Array2::<f64>::zeros((m, m));
                for r in 0..m {
                    for c in 0..m {
                        b[[r, c]] =
                            coupling * ((r as f64) - (c as f64) + (i + j) as f64 * 0.31).cos();
                    }
                }
                for r in 0..m {
                    for c in 0..m {
                        dense[[i * m + r, j * m + c]] = b[[r, c]];
                        dense[[j * m + c, i * m + r]] = b[[r, c]];
                    }
                }
                off.push((i, j, b));
            }
        }
        (diag, off, dense)
    }

    fn dense_logdet(s: &Array2<f64>) -> f64 {
        let l = s
            .cholesky(Side::Lower)
            .expect("oracle fixture must be SPD")
            .lower_triangular();
        (0..l.nrows()).map(|d| 2.0 * l[[d, d]].ln()).sum()
    }

    /// Containment: the enclosure must contain the dense truth at both
    /// orders, and the order-3 gap must not exceed the order-2 gap.
    #[test]
    fn enclosure_contains_dense_truth_and_order3_tightens() {
        let (diag, off, dense) = fixture(4, 3, 0.08);
        let truth = dense_logdet(&dense);
        let e2 =
            block_preconditioned_logdet_enclosure(&diag, &off, false).expect("order-2 enclosure");
        let e3 =
            block_preconditioned_logdet_enclosure(&diag, &off, true).expect("order-3 enclosure");
        assert!(
            e2.lower <= truth && truth <= e2.upper,
            "order-2 enclosure [{}, {}] must contain dense log|S| = {}",
            e2.lower,
            e2.upper,
            truth
        );
        assert!(
            e3.lower <= truth && truth <= e3.upper,
            "order-3 enclosure [{}, {}] must contain dense log|S| = {}",
            e3.lower,
            e3.upper,
            truth
        );
        assert!(
            e3.gap() <= e2.gap() + 1e-12,
            "order-3 gap {} must not exceed order-2 gap {}",
            e3.gap(),
            e2.gap()
        );
        // The enclosure is non-vacuous: the correction is genuinely bounded
        // away from the trivial ±∞ and the gap shrinks with ρ²p₂.
        assert!(e3.gap() < 0.5 * e2.gap() + 1e-9 || e2.gap() < 1e-9);
    }

    /// The block-diagonal case is exact: zero coupling ⇒ enclosure collapses
    /// to the exact log|D| at width 0.
    #[test]
    fn zero_coupling_is_exact() {
        let (diag, _off, dense) = fixture(3, 2, 0.0);
        let truth = dense_logdet(&dense);
        let e = block_preconditioned_logdet_enclosure(&diag, &[], true).expect("enclosure");
        assert!((e.lower - truth).abs() < 1e-10 && (e.upper - truth).abs() < 1e-10);
        assert!(e.gap() < 1e-12);
    }

    /// Strong coupling must REFUSE (ρ ≥ 1), never emit a wrong enclosure.
    #[test]
    fn failed_radius_certificate_refuses() {
        let (diag, off, _dense) = fixture(3, 2, 5.0);
        let err = block_preconditioned_logdet_enclosure(&diag, &off, false)
            .expect_err("ρ ≥ 1 must refuse");
        assert!(err.contains("spectral-radius certificate failed"));
    }

    /// The margin verdict refuses to fabricate a point value when the gap
    /// exceeds the declared decision margin, and decides (with a midpoint that
    /// the dense truth brackets) when it is below.
    #[test]
    fn margin_verdict_is_honest_about_the_gap() {
        let (diag, off, dense) = fixture(4, 3, 0.08);
        let truth = dense_logdet(&dense);
        let e = block_preconditioned_logdet_enclosure(&diag, &off, true).expect("enclosure");
        // A margin tighter than the gap must escalate, never decide.
        let tight = e.gap() * 0.5;
        assert!(!e.decide_within_margin(tight).is_decided());
        assert!(e.decide_within_margin(tight).decided_value().is_none());
        // A margin wider than the gap decides, and the midpoint is within
        // gap/2 of the truth — i.e. interchangeable for that decision.
        let loose = e.gap() * 2.0 + 1e-9;
        let verdict = e.decide_within_margin(loose);
        assert!(verdict.is_decided());
        let value = verdict.decided_value().expect("decided");
        assert!((value - truth).abs() <= 0.5 * e.gap() + 1e-12);
    }

    /// Pair absorption preserves the dense `log|S|` it encloses while shrinking
    /// the residual: the absorbed pair drops out of `E`, so the gap can only
    /// tighten, and the enclosure still brackets the truth.
    #[test]
    fn pair_absorption_preserves_truth_and_tightens() {
        let (diag, off, dense) = fixture(4, 3, 0.14);
        let truth = dense_logdet(&dense);
        let before =
            block_preconditioned_logdet_enclosure(&diag, &off, true).expect("pre-absorption");
        let (mdiag, moff) = absorb_strongest_pair(&diag, &off).expect("absorb");
        assert_eq!(
            mdiag.len(),
            diag.len() - 1,
            "one fewer block after absorption"
        );
        let after =
            block_preconditioned_logdet_enclosure(&mdiag, &moff, true).expect("post-absorption");
        assert!(
            after.lower <= truth && truth <= after.upper,
            "absorbed enclosure [{}, {}] must still contain log|S| = {truth}",
            after.lower,
            after.upper
        );
        assert!(
            after.gap() <= before.gap() + 1e-9,
            "absorption must not widen the gap ({} vs {})",
            after.gap(),
            before.gap()
        );
    }

    /// The refinement ladder closes a margin that order-3 alone cannot, by
    /// absorbing pairs — and the decided value brackets the dense truth.
    #[test]
    fn refinement_ladder_closes_margin_via_absorption() {
        let (diag, off, dense) = fixture(5, 2, 0.16);
        let truth = dense_logdet(&dense);
        let order3 =
            block_preconditioned_logdet_enclosure(&diag, &off, true).expect("order-3 enclosure");
        // Choose a margin between the dense-exact (0) and the order-3 gap so the
        // ladder must climb past moments into absorption.
        let margin = order3.gap() * 0.5;
        let (enc, verdict) =
            refine_logdet_enclosure_to_margin(&diag, &off, margin, 8).expect("ladder");
        assert!(
            verdict.is_decided(),
            "ladder must close the margin via absorption"
        );
        assert!(
            enc.lower <= truth && truth <= enc.upper,
            "refined enclosure [{}, {}] must contain log|S| = {truth}",
            enc.lower,
            enc.upper
        );
        assert!(enc.gap() < margin);
    }
}
