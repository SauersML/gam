//! #2953: a stored checkpoint is priced at full inner fidelity, from the reset state, before it
//! may outrank a certified optimum.
//!
//! The lead's 09-18 ruling: the pricing lifts the search-time inner cap exactly as the finalize
//! installation and the terminal certificate lift it. A checkpoint that is refused there, or
//! whose inner solve does not converge there, is unpriceable, and an unpriceable state cannot
//! defeat a certified one. The pricing's inner start is the reset state every stored value's
//! search started from, not the mode the winner's evaluation left in the objective.

use super::*;
use ndarray::array;
use std::sync::Arc;
use std::sync::atomic::{AtomicBool, AtomicU64, AtomicUsize, Ordering};

/// The cap the search schedule holds when the checkpoint is priced.
const SEARCH_CAP: usize = 3;
const CAPPED_MARKER: &str = "the #2953 pricing fixture refuses a capped inner solve";
const CHECKPOINT: f64 = -3.5;
const WINNER: f64 = 0.0;
/// The checkpoint's value on the mode its search, started from the reset state, converged to.
const RESET_MODE_VALUE: f64 = -9.0;
/// The checkpoint's value on the mode an inner solve warm-started at the winner's mode reaches.
const WINNER_MODE_VALUE: f64 = -1.0;

/// How the fixture's inner solve behaves at the checkpoint.
#[derive(Clone, Copy)]
enum InnerSolve {
    /// Refused under any nonzero cap; converged with the cap lifted.
    RefusedUnderCap,
    /// Reports non-convergence even with the cap lifted.
    UnconvergedUncapped,
    /// Converges to the mode its start lies in: the reset mode from the reset state, the
    /// winner's mode once an evaluation at the winner has left that mode as the start.
    StartDependent,
}

struct PricingFixture {
    feedback: InnerProgressFeedback,
    inner: InnerSolve,
    started_at_winners_mode: bool,
    resets: usize,
    evaluated_caps: Vec<usize>,
}

impl PricingFixture {
    fn new(inner: InnerSolve) -> Self {
        Self {
            feedback: InnerProgressFeedback {
                cap: Arc::new(AtomicUsize::new(SEARCH_CAP)),
                accepted_iter: Arc::new(AtomicUsize::new(0)),
                last_iters: Arc::new(AtomicUsize::new(0)),
                last_converged: Arc::new(AtomicBool::new(true)),
                ift_residual: Arc::new(AtomicU64::new(f64::NAN.to_bits())),
                accept_rho: Arc::new(AtomicU64::new(f64::NAN.to_bits())),
                force_cold: Arc::new(AtomicBool::new(false)),
            },
            inner,
            started_at_winners_mode: false,
            resets: 0,
            evaluated_caps: Vec::new(),
        }
    }

    fn config(&self) -> OuterConfig {
        OuterConfig {
            outer_inner_cap: Some(self.feedback.clone()),
            ..OuterConfig::default()
        }
    }

    fn report_inner(&self, iterations: usize, converged: bool) {
        self.feedback.last_iters.store(iterations, Ordering::Relaxed);
        self.feedback
            .last_converged
            .store(converged, Ordering::Relaxed);
    }

    fn value(&mut self, theta: &Array1<f64>) -> Result<f64, EstimationError> {
        let cap = self.feedback.cap.load(Ordering::Relaxed);
        self.evaluated_caps.push(cap);
        if theta[0] == WINNER {
            self.report_inner(4, true);
            self.started_at_winners_mode = true;
            return Ok(0.0);
        }
        match self.inner {
            InnerSolve::RefusedUnderCap if cap != 0 => {
                self.report_inner(cap, false);
                Err(EstimationError::TrialPointRefused {
                    reason: CAPPED_MARKER.to_string(),
                })
            }
            InnerSolve::RefusedUnderCap => {
                self.report_inner(9, true);
                Ok(RESET_MODE_VALUE)
            }
            InnerSolve::UnconvergedUncapped => {
                self.report_inner(1200, false);
                Ok(RESET_MODE_VALUE)
            }
            InnerSolve::StartDependent => {
                self.report_inner(9, true);
                Ok(if self.started_at_winners_mode {
                    WINNER_MODE_VALUE
                } else {
                    RESET_MODE_VALUE
                })
            }
        }
    }
}

impl OuterObjective for PricingFixture {
    fn capability(&self) -> OuterCapability {
        OuterCapability {
            gradient: Derivative::Analytic,
            hessian: DeclaredHessianForm::Unavailable,
            n_params: 1,
            psi_dim: 0,
            fixed_point_available: false,
            barrier_config: None,
            prefer_gradient_only: true,
            disable_fixed_point: true,
        }
    }
    fn eval_cost(&mut self, theta: &Array1<f64>) -> Result<f64, EstimationError> {
        self.value(theta)
    }
    fn eval(&mut self, theta: &Array1<f64>) -> Result<OuterEval, EstimationError> {
        Ok(OuterEval {
            cost: self.value(theta)?,
            gradient: Array1::<f64>::zeros(theta.len()),
            hessian: HessianValue::Unavailable,
            inner_beta_hint: None,
        })
    }
    fn reset(&mut self) {
        self.resets += 1;
        self.started_at_winners_mode = false;
    }
    fn seed_inner_state(&mut self, beta: &Array1<f64>) -> Result<SeedOutcome, EstimationError> {
        if beta.iter().any(|value| !value.is_finite()) {
            return Err(EstimationError::RemlOptimizationFailed(
                "the #2953 pricing fixture was offered a non-finite inner seed".to_string(),
            ));
        }
        Ok(SeedOutcome::NoSlot)
    }
}

/// A stored value may come from a capped inner solve. With the search-time cap in force the
/// fixture refuses the checkpoint, and with it lifted the checkpoint prices, so it is priced,
/// and then still dominates, only because the pricing lifts the cap. The scheduler's cap is
/// restored afterwards.
#[test]
fn a_checkpoint_refused_only_under_the_search_cap_is_priced_at_full_fidelity_2953() {
    let mut fixture = PricingFixture::new(InnerSolve::RefusedUnderCap);
    let config = fixture.config();
    let price = price_checkpoint(&mut fixture, &config, &array![CHECKPOINT], "pricing #2953")
        .expect("a refusal of the capped solve is not fatal");
    assert_eq!(price, CheckpointPrice::Priced(RESET_MODE_VALUE));
    assert_eq!(
        fixture.evaluated_caps,
        vec![0],
        "the one pricing evaluation must run with the cap lifted"
    );
    assert_eq!(fixture.feedback.cap.load(Ordering::Relaxed), SEARCH_CAP);
    assert_eq!(fixture.resets, 2, "the pricing starts and ends at the reset state");
}

/// A value whose inner solve did not converge with the cap lifted is not a value of the
/// profiled criterion, so the checkpoint is unpriceable and cannot defeat a certified optimum.
#[test]
fn a_checkpoint_whose_inner_solve_does_not_converge_at_full_fidelity_is_unpriceable_2953() {
    let mut fixture = PricingFixture::new(InnerSolve::UnconvergedUncapped);
    let config = fixture.config();
    let price = price_checkpoint(&mut fixture, &config, &array![CHECKPOINT], "pricing #2953")
        .expect("an unconverged inner solve is not fatal");
    assert_eq!(
        price,
        CheckpointPrice::Unpriceable(CheckpointPriceRefusal::InnerUnconverged)
    );
    assert_eq!(fixture.evaluated_caps, vec![0]);
    assert_eq!(fixture.feedback.cap.load(Ordering::Relaxed), SEARCH_CAP);
}

/// The winner's evaluation leaves its mode as the objective's next inner start, and from there
/// the inner solve at the checkpoint reaches a different mode than the reset state reaches. The
/// checkpoint's stored value came from a search that started at the reset state, as the
/// winner's did, so the pricing starts there: it prices the reset mode, not the winner's.
#[test]
fn a_checkpoint_is_priced_from_the_reset_state_not_from_the_winners_mode_2953() {
    let mut fixture = PricingFixture::new(InnerSolve::StartDependent);
    let config = fixture.config();
    fixture
        .eval_cost(&array![WINNER])
        .expect("the fixture prices the winner");
    assert!(fixture.started_at_winners_mode);
    let price = price_checkpoint(&mut fixture, &config, &array![CHECKPOINT], "pricing #2953")
        .expect("the start-dependent fixture always prices");
    assert_eq!(
        price,
        CheckpointPrice::Priced(RESET_MODE_VALUE),
        "a start carried from the winner's mode prices {WINNER_MODE_VALUE}, another mode"
    );
    assert_eq!(fixture.resets, 2);
}
