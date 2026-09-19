use crate::custom_family::{CustomFamilyWarmStart, EvalMode};
use ndarray::Array1;
use std::sync::Arc;
use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering};

/// The two outer-walk events a coefficient-mode branch reads from its driver.
///
/// `accepted_steps` is advanced by the optimizer's accept observer on every outer
/// step it accepts, through `OuterProblem::with_stuck_stall_cold_reeval_signal`
/// (#2613, #2668). `walk_resets` is advanced by the objective's reset, which the
/// driver fires before each seed's walk and before terminal certification. It is
/// `pub` because the `pub` spatial exact-joint drivers take it as a parameter.
#[derive(Clone, Default)]
pub struct OuterWalkSignals {
    pub(crate) accepted_steps: Arc<AtomicUsize>,
    pub(crate) walk_resets: Arc<AtomicUsize>,
}

impl OuterWalkSignals {
    /// `problem` with its accept observer advancing `accepted_steps`, through the
    /// channel the custom-family outer loop subscribes with (#2668). The cold-reeval
    /// pulse that channel also carries is inert here: no exact-joint driver reads it.
    pub(crate) fn subscribe(
        signals: Option<&Self>,
        problem: gam_solve::rho_optimizer::OuterProblem,
    ) -> gam_solve::rho_optimizer::OuterProblem {
        match signals {
            Some(signals) => problem.with_stuck_stall_cold_reeval_signal(
                Arc::new(AtomicBool::new(false)),
                Arc::clone(&signals.accepted_steps),
            ),
            None => problem,
        }
    }

    /// The objective's reset callback: it advances `walk_resets` so the branch
    /// anchors the next walk's starting iterate at once, and drops no memo.
    pub(crate) fn reset_counter<S>(&self) -> impl FnMut(&mut S) + use<S> {
        let walk_resets = Arc::clone(&self.walk_resets);
        move |_: &mut S| {
            walk_resets.fetch_add(1, Ordering::Relaxed);
        }
    }
}

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
/// accepted outer iterate, and every evaluation warm-starts from it. Value-only
/// probes read the anchor and never write it. A derivative-bearing evaluation is
/// not evidence of acceptance either: the Strong-Wolfe line search evaluates the
/// gradient at every trial that clears Armijo (#2613), so a trial allowed to
/// replace the anchor chose the basin of every later probe, and at one outer
/// point two certified modes alternated in the published criterion (#2973). The
/// latest converged derivative-bearing evaluation is therefore held pending, and
/// it becomes the anchor only when the optimizer reports its step accepted — the
/// rule `CustomOuterState` applies to the custom-family outer loop (#2668). The
/// first derivative-bearing evaluation after a walk reset is that walk's starting
/// iterate and anchors at once. Before it there is no accepted iterate, so the
/// value-only seed probes carry their converged mode forward.
///
/// An evaluation at a `θ` the walk has already accepted is solved from that
/// iterate's own certified mode, not from the anchor. After a multi-start the
/// anchor belongs to whichever walk accepted an iterate last, while the terminal
/// certification re-evaluates the winning walk's iterate. Solving that
/// evaluation from another walk's mode is a different inner problem at the
/// certified `θ` (gam#2765: seed 0 certified, a later seed moved the anchor, and
/// the terminal `ValueGradientHessian` evaluation at seed 0's `θ` started from
/// `|β|∞ = 9.46` and diverged onto the singular Jeffreys face). The certified
/// modes are keyed by the bits of the full `θ`, so a ψ move is a different
/// iterate even where `ρ` did not move.
pub(crate) struct ExactCoefficientModeBranch {
    /// The mode every evaluation is solved from.
    anchor: Option<CustomFamilyWarmStart>,
    /// Whether `anchor` is the certified mode of an accepted outer iterate
    /// rather than a seed-time carry. Once true, seeds and value-only probes can
    /// no longer write it.
    anchored_at_iterate: bool,
    /// The certified mode of every accepted outer iterate, keyed by its `θ` bits.
    iterate_modes: Vec<(Vec<u64>, CustomFamilyWarmStart)>,
    /// The driver events this branch folds in.
    signals: OuterWalkSignals,
    /// How many accepted steps have been folded in.
    accepted_steps_adopted: usize,
    /// How many walk resets have been folded in.
    walk_resets_adopted: usize,
    /// Whether the current walk's starting iterate has anchored.
    walk_started: bool,
    /// The latest converged derivative-bearing evaluation since the last accepted
    /// step, held until the optimizer reports that step accepted.
    pending_iterate: Option<(Vec<u64>, CustomFamilyWarmStart)>,
}

fn theta_bits(theta: &Array1<f64>) -> Vec<u64> {
    theta.iter().map(|value| value.to_bits()).collect()
}

impl ExactCoefficientModeBranch {
    pub(crate) fn new(signals: OuterWalkSignals) -> Self {
        let accepted_steps_adopted = signals.accepted_steps.load(Ordering::Relaxed);
        let walk_resets_adopted = signals.walk_resets.load(Ordering::Relaxed);
        Self {
            anchor: None,
            anchored_at_iterate: false,
            iterate_modes: Vec::new(),
            signals,
            accepted_steps_adopted,
            walk_resets_adopted,
            walk_started: false,
            pending_iterate: None,
        }
    }

    /// Fold in the accepted steps and walk resets reported since the previous
    /// evaluation. An accepted step promotes the pending trial, which is the
    /// accepted iterate's own evaluation, because the observer fires after the
    /// line search that produced the step. A reset starts a new walk: its first
    /// derivative-bearing evaluation anchors at once, and a trial the previous
    /// walk never accepted is dropped.
    ///
    /// When both are reported, the step came first: a step is accepted only after
    /// its walk has evaluated, so no step can follow a reset before the next
    /// evaluation. The step is therefore folded in before the reset. A walk that
    /// ends on an accepted step keeps that iterate's certified mode, and the
    /// terminal certification the driver runs after its reset solves the winning
    /// `θ` from it rather than from the iterate before.
    fn adopt_walk_events(&mut self) {
        let accepted = self.signals.accepted_steps.load(Ordering::Relaxed);
        if accepted != self.accepted_steps_adopted {
            self.accepted_steps_adopted = accepted;
            if let Some((key, warm_start)) = self.pending_iterate.take() {
                self.anchor_iterate(key, warm_start);
            }
        }
        let resets = self.signals.walk_resets.load(Ordering::Relaxed);
        if resets != self.walk_resets_adopted {
            self.walk_resets_adopted = resets;
            self.walk_started = false;
            self.pending_iterate = None;
        }
    }

    fn anchor_iterate(&mut self, key: Vec<u64>, warm_start: CustomFamilyWarmStart) {
        self.iterate_modes.retain(|(bits, _)| *bits != key);
        self.iterate_modes.push((key, warm_start.clone()));
        self.anchor = Some(warm_start);
        self.anchored_at_iterate = true;
    }

    /// The warm-start candidates for one evaluation at `theta`: the certified
    /// mode of the accepted iterate at `theta` when there is one, otherwise the
    /// anchor, when it is dimensionally compatible with `rho`; otherwise a cold
    /// solve.
    ///
    /// The returned flag is true exactly at the first derivative-bearing
    /// evaluation, the moment the anchor becomes iterate-owned.
    pub(crate) fn candidates(
        &mut self,
        eval_mode: EvalMode,
        theta: &Array1<f64>,
        rho: &Array1<f64>,
    ) -> (bool, Vec<Option<CustomFamilyWarmStart>>) {
        self.adopt_walk_events();
        let first_iterate_evaluation =
            !self.anchored_at_iterate && !matches!(eval_mode, EvalMode::ValueOnly);
        let key = theta_bits(theta);
        let warm = self
            .iterate_modes
            .iter()
            .find(|(bits, _)| *bits == key)
            .map(|(_, warm)| warm)
            .or(self.anchor.as_ref())
            .filter(|warm| warm.compatible_with_rho(rho))
            .cloned();
        (first_iterate_evaluation, vec![warm])
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

    /// Record the mode an evaluation at `theta` converged to. A
    /// derivative-bearing evaluation that starts a walk anchors at once; any
    /// later one is held pending until the optimizer accepts its step, and the
    /// next trial replaces it if it never is. A value-only probe writes only while
    /// no iterate has been accepted yet. A mode that did not converge is never
    /// recorded — the evaluation it came from is refused by the caller, and a
    /// refused trial must leave no trace.
    pub(crate) fn record_value(
        &mut self,
        eval_mode: EvalMode,
        theta: &Array1<f64>,
        warm_start: CustomFamilyWarmStart,
        converged: bool,
    ) {
        if !converged {
            return;
        }
        self.adopt_walk_events();
        if !matches!(eval_mode, EvalMode::ValueOnly) {
            let key = theta_bits(theta);
            if self.walk_started {
                self.pending_iterate = Some((key, warm_start));
            } else {
                self.anchor_iterate(key, warm_start);
                self.walk_started = true;
            }
        } else if !self.anchored_at_iterate {
            self.anchor = Some(warm_start);
        }
    }
}
