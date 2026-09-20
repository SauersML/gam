use super::*;

/// Penalized-loss breakdown for diagnostics.
#[derive(Debug, Clone, Copy)]
pub struct SaeManifoldLoss {
    pub data_fit: f64,
    pub assignment_sparsity: f64,
    pub smoothness: f64,
    pub ard: f64,
    pub criterion_gauge_deflated_directions: usize,
}

impl SaeManifoldLoss {
    pub const fn total(&self) -> f64 {
        self.data_fit + self.assignment_sparsity + self.smoothness + self.ard
    }

    /// Negative penalized loss `−(data_fit + assignment_sparsity + smoothness +
    /// ard)`. Larger is "less penalized loss", so penalized quasi-Laplace wrappers that rank
    /// larger-is-better can sort on it — but this is **not** a REML / marginal
    /// likelihood: it omits the Hessian log-determinant, the Occam log-λ term,
    /// any extra analytic penalties, the co-training fold, and hybrid-collapse
    /// effects. Callers must surface it under
    /// an honest name (`penalized_loss_score`, or `oos_penalized_loss` on the
    /// fixed-decoder OOS path), never `reml_score`.
    pub const fn penalized_loss_score(&self) -> f64 {
        -self.total()
    }

    /// Honest component breakdown of [`Self::total`] — the four penalized-loss
    /// terms this struct actually carries — so a consumer can see exactly what
    /// the score is (and what it is *not*: it is missing the quasi-Laplace pieces
    /// listed on [`Self::penalized_loss_score`]). The values are the raw
    /// (positive) loss contributions; `penalized_loss_score == −Σ` of the first
    /// four.
    pub const fn breakdown(&self) -> SaeManifoldLossBreakdown {
        SaeManifoldLossBreakdown {
            data_fit: self.data_fit,
            assignment_sparsity: self.assignment_sparsity,
            smoothness: self.smoothness,
            ard: self.ard,
            total_penalized_loss: self.total(),
            penalized_loss_score: self.penalized_loss_score(),
            criterion_gauge_deflated_directions: self.criterion_gauge_deflated_directions,
        }
    }
}

/// The penalized objective `penalized_objective_total` with the first-order
/// rounding band of its evaluation (#3243), from
/// `SaeManifoldTerm::penalized_objective_banded`.
///
/// The objective is the loss components plus the extra penalty energies. Its
/// longest accumulation is the data fit over the `n·p` residual cells, and the
/// eight summands are then added in turn, so the computed value lies within
/// `band = γ_(n·p+7)·Σ|summandᵢ|` of the exactly accumulated one and resolves no
/// change below it.
#[derive(Debug, Clone, Copy, PartialEq)]
pub(crate) struct BandedPenalizedObjective {
    pub(crate) value: f64,
    pub(crate) band: f64,
}

impl BandedPenalizedObjective {
    /// A trial that failed to apply or to evaluate: no finite value, so no
    /// comparison accepts it.
    pub(crate) const UNUSABLE: Self = Self {
        value: f64::INFINITY,
        band: 0.0,
    };

    /// Whether `trial`, from this value, is an Armijo decrease the two
    /// evaluations can resolve (#3243).
    ///
    /// The comparison `V̂ − V̂'` of two evaluations carries the error of both,
    /// `B + B'`. The trial must be resolvably below this value, `V̂ − V̂' > B + B'`,
    /// so a committed step lowered the exact objective and not only its rounding.
    /// It must also pass the Armijo test `V̂ − V̂' ≥ sufficient` relaxed by the same
    /// `B + B'` (Berahas, Byrd & Nocedal 2019), so a step whose only shortfall is
    /// rounding is never refused for it. With both bands zero this is the exact
    /// strict Armijo test.
    pub(crate) fn armijo_accepts(&self, trial: &Self, sufficient: f64) -> bool {
        let bands = self.band + trial.band;
        let decrease = self.value - trial.value;
        self.value.is_finite()
            && trial.value.is_finite()
            && decrease > bands
            && decrease >= sufficient - bands
    }
}

/// Honest, fully-itemized view of [`SaeManifoldLoss`] for the model output. It
/// reports the penalized-loss components that the score is actually built from,
/// and is deliberately NOT named or shaped like a REML / evidence breakdown:
/// the Hessian log-determinant, Occam log-λ, extra penalties, co-training fold,
/// and top-k / hybrid-collapse effects are not part of this object (#1231).
#[derive(Debug, Clone, Copy)]
pub struct SaeManifoldLossBreakdown {
    pub data_fit: f64,
    pub assignment_sparsity: f64,
    pub smoothness: f64,
    pub ard: f64,
    /// `data_fit + assignment_sparsity + smoothness + ard`.
    pub total_penalized_loss: f64,
    /// `−total_penalized_loss` (larger = less penalized loss).
    pub penalized_loss_score: f64,
    /// Count of criterion-gauge-deflated directions recorded on the loss.
    pub criterion_gauge_deflated_directions: usize,
}

/// Componentized analytic derivative of the SAE penalized quasi-Laplace criterion with respect to
/// the flat [`SaeManifoldRho`] layout.
///
/// Production objective and certificate paths consume this value object so the
/// criterion value and gradient are assembled from the same converged cache.
#[derive(Debug, Clone)]
pub struct SaeOuterRhoGradientComponents {
    /// Direct (no-envelope) derivative of `loss.total() + extra_penalty_energy`
    /// with respect to log-strength coordinates, excluding the custom factor
    /// logdet and Occam terms, PLUS the realised-rank charge's direct
    /// ρ-differential (`ProductionRankChargeDerivative::direct_rho`, folded in by
    /// `analytic_outer_rho_gradient_components_with_bundle`).
    ///
    /// #2087 — that last summand is why this is NOT the ρ-derivative of
    /// `loss.total() + extra_penalty_energy` alone. The scalar criterion adds
    /// the realised-rank charge to `½log|H|`, so the
    /// charge lives in the quasi-Laplace COMPLEXITY and `SaeManifoldTerm::loss`
    /// cannot see it. An audit that finite-differences `loss.total()` and compares
    /// it to this field is comparing two different quantities and will report the
    /// charge as a data-fit/prior desync; subtract the charge's differential first.
    pub explicit: Array1<f64>,
    /// `0.5 * tr(B^{-1} dB/d rho_j)` for the currently available penalty blocks.
    pub logdet_trace: Array1<f64>,
    /// Derivative contribution of `-occam`.
    pub occam: Array1<f64>,
    /// `−½·Γᵀθ̂_ρ`, the implicit fitted-state response of the custom `log|B|`
    /// term. Inner stationarity removes the corresponding response
    /// of the penalized loss, not this trace term.
    pub third_order_correction: Array1<f64>,
}

impl SaeOuterRhoGradientComponents {
    /// The consumed outer-ρ gradient: `explicit + logdet_trace + occam +
    /// third_order_correction`.
    #[must_use]
    pub fn gradient(&self) -> Array1<f64> {
        &(&(&self.explicit + &self.logdet_trace) + &self.occam) + &self.third_order_correction
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// #1231 — the public score is the NEGATIVE penalized loss of the four loss
    /// components, and the breakdown itemizes exactly those components. It is not
    /// (and must not be presented as) a penalized quasi-Laplace criterion.
    #[test]
    fn penalized_loss_score_is_negative_total_with_breakdown() {
        let loss = SaeManifoldLoss {
            data_fit: 1.5,
            assignment_sparsity: 0.25,
            smoothness: 0.5,
            ard: 0.75,
            criterion_gauge_deflated_directions: 3,
        };
        let total = 1.5 + 0.25 + 0.5 + 0.75;
        assert!((loss.total() - total).abs() < 1e-12);
        assert!((loss.penalized_loss_score() - (-total)).abs() < 1e-12);

        let b = loss.breakdown();
        assert!((b.data_fit - 1.5).abs() < 1e-12);
        assert!((b.assignment_sparsity - 0.25).abs() < 1e-12);
        assert!((b.smoothness - 0.5).abs() < 1e-12);
        assert!((b.ard - 0.75).abs() < 1e-12);
        assert!((b.total_penalized_loss - total).abs() < 1e-12);
        assert!((b.penalized_loss_score - (-total)).abs() < 1e-12);
        // The breakdown's four components must sum to the reported total — the
        // score is fully explained by what the breakdown lists, with no hidden
        // evidence pieces folded into it.
        let summed = b.data_fit + b.assignment_sparsity + b.smoothness + b.ard;
        assert!((summed - b.total_penalized_loss).abs() < 1e-12);
        assert_eq!(b.criterion_gauge_deflated_directions, 3);
    }

    /// #3243 — the acceptance at its edges, with bands `B = B' = 1e-9` around
    /// `V̂ = 1e3`, where opt's former cushion `8ε(1 + |V̂|)` is `1.8e-12`:
    /// - a trial `1.5e-9` lower passes that cushion's Armijo test at a
    ///   sufficient decrease of `1e-9`, but lies inside `B + B'`, resolves
    ///   nothing, and is refused;
    /// - a trial `3e-9` lower is resolved, and is refused at a sufficient
    ///   decrease of `1e-8`, far past it plus the bands;
    /// - the same trial is accepted at a sufficient decrease of `4e-9`, which
    ///   it misses by less than `B + B'`: the exact test refuses it, the relaxed
    ///   one does not;
    /// - an unusable trial, or a non-finite base, is never accepted.
    #[test]
    fn armijo_accepts_only_resolved_decrease_and_relaxes_armijo_by_the_bands_3243() {
        let band = 1.0e-9;
        let base = BandedPenalizedObjective { value: 1.0e3, band };
        let at = |decrease: f64| BandedPenalizedObjective {
            value: base.value - decrease,
            band,
        };
        let former_cushion = 8.0 * f64::EPSILON * (1.0 + base.value);
        let unresolved = at(1.5e-9);
        assert!(
            base.value - unresolved.value >= 1.0e-9 - former_cushion,
            "the former cushion's test accepted this rounding-sized trial"
        );
        assert!(!base.armijo_accepts(&unresolved, 1.0e-9));
        let resolved = at(3.0e-9);
        assert!(!base.armijo_accepts(&resolved, 1.0e-8));
        assert!(base.value - resolved.value < 4.0e-9);
        assert!(base.armijo_accepts(&resolved, 4.0e-9));
        assert!(!base.armijo_accepts(&BandedPenalizedObjective::UNUSABLE, 0.0));
        let non_finite = BandedPenalizedObjective {
            value: f64::NAN,
            band,
        };
        assert!(!non_finite.armijo_accepts(&resolved, 0.0));
        let exact = BandedPenalizedObjective {
            value: 1.0,
            band: 0.0,
        };
        let below = BandedPenalizedObjective {
            value: 1.0 - 1.0e-12,
            band: 0.0,
        };
        assert!(exact.armijo_accepts(&below, 1.0e-12 * 0.5));
        assert!(!exact.armijo_accepts(&exact, 0.0));
    }
}
