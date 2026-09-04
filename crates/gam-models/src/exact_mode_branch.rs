use crate::custom_family::{CustomFamilyWarmStart, EvalMode};
use ndarray::Array1;

/// The coefficient mode an exact profiled objective is solved from.
///
/// The profiled objective `V(θ) = min_β J(β; θ)` is a function of `θ` only
/// when the inner minimizer is reached from a start that does not depend on
/// which outer trials happened to be evaluated before it. A warm start taken
/// from the last *trial* breaks that: a rejected line-search probe would then
/// choose the basin of the next probe, and the value at one `θ` would depend on
/// the order in which the line search visited its candidates.
///
/// The previous answer was a cold solve at every trial after the first
/// derivative-bearing evaluation. That keeps the value history-free but throws
/// away the one thing the outer walk knows — the certified mode of the iterate
/// it is stepping from — and far from the seed the cold solve is the fragile
/// one (gam#2765: at `ρ = 12` with the Jeffreys term armed, every cold start
/// took one trust-region step into a region where the reduced information is
/// singular and stalled there, so every line-search probe was refused as
/// infeasible and the outer search halted with `|g| = 7.5` while the accepted
/// iterates had been solving fine).
///
/// This branch therefore carries ONE anchor: the certified mode of the current
/// accepted outer iterate. Every evaluation warm-starts from it, and only a
/// derivative-bearing evaluation — which the outer solver requests at an
/// iterate it has accepted, never at a probe — may replace it. Value-only
/// probes read the anchor and never write it, so within one line search every
/// candidate step is solved from the same start and the objective is a
/// function of `θ` and of the accepted iterate, not of the probe order. Before
/// the first derivative-bearing evaluation there is no accepted iterate, so the
/// value-only seed probes carry their converged mode forward.
#[derive(Default)]
pub(crate) struct ExactCoefficientModeBranch {
    /// The mode every evaluation is solved from.
    anchor: Option<CustomFamilyWarmStart>,
    /// Whether `anchor` is the certified mode of an accepted outer iterate
    /// (written by a derivative-bearing evaluation) rather than a seed-time
    /// carry. Once true, seeds and value-only probes can no longer write it.
    anchored_at_iterate: bool,
}

impl ExactCoefficientModeBranch {
    /// The warm-start candidates for one evaluation: the anchor when it is
    /// dimensionally compatible with `rho`, otherwise a cold solve.
    ///
    /// The returned flag is true exactly at the first derivative-bearing
    /// evaluation, the moment the anchor becomes iterate-owned.
    pub(crate) fn candidates(
        &mut self,
        eval_mode: EvalMode,
        rho: &Array1<f64>,
    ) -> (bool, Vec<Option<CustomFamilyWarmStart>>) {
        let first_iterate_evaluation =
            !self.anchored_at_iterate && !matches!(eval_mode, EvalMode::ValueOnly);
        let warm = self
            .anchor
            .as_ref()
            .filter(|warm| warm.compatible_with_rho(rho))
            .cloned();
        match warm {
            Some(warm) => (first_iterate_evaluation, vec![Some(warm)]),
            None => (first_iterate_evaluation, vec![None]),
        }
    }

    /// Install a seed mode from outside the walk (an outer ρ-cache coefficient
    /// seed). Refused once an accepted iterate owns the anchor: a cached seed
    /// from another walk must not displace the mode this walk certified.
    pub(crate) fn install_seed(&mut self, warm_start: CustomFamilyWarmStart) -> bool {
        if self.anchored_at_iterate {
            false
        } else {
            self.anchor = Some(warm_start);
            true
        }
    }

    /// Record the mode an evaluation converged to. A derivative-bearing
    /// evaluation is an accepted iterate and replaces the anchor; a value-only
    /// probe writes only while no iterate has been accepted yet. A mode that
    /// did not converge is never recorded — the evaluation it came from is
    /// refused by the caller, and a refused trial must leave no trace.
    pub(crate) fn record_value(
        &mut self,
        eval_mode: EvalMode,
        warm_start: CustomFamilyWarmStart,
        converged: bool,
    ) {
        if !converged {
            return;
        }
        if !matches!(eval_mode, EvalMode::ValueOnly) {
            self.anchor = Some(warm_start);
            self.anchored_at_iterate = true;
        } else if !self.anchored_at_iterate {
            self.anchor = Some(warm_start);
        }
    }
}
