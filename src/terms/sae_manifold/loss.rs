use super::*;

/// Loss breakdown for diagnostics and evidence ranking.
#[derive(Debug, Clone, Copy)]
pub struct SaeManifoldLoss {
    pub data_fit: f64,
    pub assignment_sparsity: f64,
    pub smoothness: f64,
    pub ard: f64,
    pub evidence_gauge_deflated_directions: usize,
}


impl SaeManifoldLoss {
    pub const fn total(&self) -> f64 {
        self.data_fit + self.assignment_sparsity + self.smoothness + self.ard
    }

    /// Laplace/REML wrappers rank larger evidence higher. This local score is
    /// the negative penalized objective, used when a full `RemlState` is not
    /// driving the term yet.
    pub const fn evidence_proxy(&self) -> f64 {
        -self.total()
    }
}


/// Componentized analytic derivative of the SAE REML criterion with respect to
/// the flat [`SaeManifoldRho`] layout.
///
/// This is intentionally only a value object for tests and derivation gates. It
/// is not wired into [`SaeManifoldOuterObjective`] capability planning until the
/// third-order logdet correction is available behind its own oracle.
#[derive(Debug, Clone)]
pub struct SaeOuterRhoGradientComponents {
    /// Direct derivative of `loss.total() + extra_penalty_energy` with respect to
    /// log-strength coordinates, excluding the Hessian logdet and Occam terms.
    pub explicit: Array1<f64>,
    /// `0.5 * tr(H^{-1} dH/d rho_j)` for the currently available penalty blocks.
    pub logdet_trace: Array1<f64>,
    /// Derivative contribution of `-occam`.
    pub occam: Array1<f64>,
    /// Reserved channel for `0.5 * tr(H^{-1} (dH/dtheta * dtheta_hat/d rho_j))`.
    pub third_order_correction: Array1<f64>,
    /// Whether `third_order_correction` is populated from analytic channels.
    pub third_order_correction_available: bool,
}


impl SaeOuterRhoGradientComponents {
    #[must_use]
    pub fn gradient_excluding_unavailable_correction(&self) -> Array1<f64> {
        &(&self.explicit + &self.logdet_trace) + &self.occam
    }

    #[must_use]
    pub fn gradient_with_available_correction(&self) -> Array1<f64> {
        // The name is a contract: callers asking for the corrected gradient
        // must not silently receive the uncorrected one. Zeros-by-omission in
        // the correction channel are exactly the objective↔gradient desync
        // class; fail loudly instead.
        assert!(
            self.third_order_correction_available,
            "gradient_with_available_correction: third-order correction channel \
             is not populated for this fit; use \
             gradient_excluding_unavailable_correction() and account for the \
             missing term explicitly"
        );
        &self.gradient_excluding_unavailable_correction() + &self.third_order_correction
    }
}
