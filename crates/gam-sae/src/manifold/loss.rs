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
    /// Direct derivative of `loss.total() + extra_penalty_energy` with respect to
    /// log-strength coordinates, excluding the custom factor logdet and Occam terms.
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
}
