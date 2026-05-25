//! Continuation / homotopy seed strategy for the outer REML loop.
//!
//! Anneal ρ from an oversmoothing start ρ₀ down to the target ρ*,
//! warm-starting β across steps. Each per-ρ inner solve is exact;
//! continuation only changes *which* ρ the solver is asked about,
//! not *how* it answers.
//!
//! `fit_with_continuation` calls the supplied `OuterObjective` for
//! every step: `seed_inner_state` installs the warm-start β, then
//! `eval_with_order` runs the inner P-IRLS. Returned errors flow
//! through `inner_status::classify_inner_error` so the rollback
//! decision tree matches the structured `InnerFailure` enum that
//! `outer_strategy.rs` already aggregates into `StartupStats`.
//!
//! # Failure rollback (per `InnerFailure` variant)
//!
//! | variant                                                | action                                  |
//! |--------------------------------------------------------|-----------------------------------------|
//! | `CertRefused { RankDeficientHPen }`                    | expand ρ₀ outward; restart path         |
//! | `CertRefused { ActiveSetIncomplete }`                  | propagate — real KKT bug                |
//! | `CertRefused { PhantomMultiplierWithWellConditionedH }`| halve α; on underflow → fail            |
//! | `BudgetExhausted`                                      | halve α                                 |
//! | `TrustRegionFloor`                                     | halve α; on repeat → expand ρ₀          |
//! | `LikelihoodFailure`                                    | halve α (β at ρ_k was just accepted)    |
//! | `Other`                                                | propagate (conservative)                |
//!
//! # Magic-by-default
//!
//! Continuation is unconditional. When ρ₀ ≈ ρ* component-wise (within
//! `RHO_EQUAL_TOL`), the schedule collapses to a single direct step at
//! ρ*. The collapse test is bracketing-based (test, don't predict) —
//! no expensive ‖H_loglik‖ estimate, so the no-op cost is zero on top
//! of the cold eval.

use ndarray::Array1;

use crate::families::custom_family::KktRefusalDiagnosis;
use crate::families::inner_status::{InnerFailure, classify_inner_error};
use crate::solver::estimate::EstimationError;
use crate::solver::outer_strategy::{OuterEval, OuterEvalOrder, OuterObjective};

/// Hard ceiling on the number of ρ steps along a single continuation
/// path. Past this we surface the last `InnerFailure` rather than the
/// schedule's give-up.
pub(crate) const PATH_BUDGET: usize = 64;

/// Hard floor on the geometric step fraction `α`. Shrinking below
/// this counts as "the path is not making progress at this ρ₀".
pub(crate) const ALPHA_FLOOR: f64 = 1.0 / 1024.0;

/// Initial step fraction: halve the remaining distance per step.
pub(crate) const ALPHA_INIT: f64 = 0.5;

/// Multiplier on α after a successful step. Grows toward "one shot to
/// ρ*" on well-behaved paths.
pub(crate) const ALPHA_EXPAND: f64 = 1.5;

/// Multiplier on α after a refused step.
pub(crate) const ALPHA_SHRINK: f64 = 0.5;

/// Initial oversmoothing offset added to ρ* to build ρ₀. log(32) ≈
/// 3.4657 — penalty curvature dominates the likelihood Hessian by ~1.5
/// orders of magnitude in the generic case.
pub(crate) const OVERSMOOTH_OFFSET_INIT: f64 = 3.4657359027997265;

/// Maximum number of ρ₀ outward expansions when the path refuses at
/// ρ₀ itself. Each expansion doubles the offset.
pub(crate) const OVERSMOOTH_RETRY_MAX: usize = 3;

/// Component-wise tolerance below which ρ₀ ≈ ρ* triggers the one-step
/// collapse. 0.5 log-units ≈ factor of √e in λ.
pub(crate) const RHO_EQUAL_TOL: f64 = 0.5;

#[derive(Debug, Clone)]
pub(crate) enum ContinuationFailure {
    PathBudgetExhausted {
        last: InnerFailure,
        steps_taken: usize,
        final_rho: Array1<f64>,
    },
    PathStuck {
        last: InnerFailure,
        rho_zero_offset: f64,
        final_rho: Array1<f64>,
    },
    StructuralPropagate(InnerFailure),
    DomainAtOversmoothedStart(InnerFailure),
}

impl ContinuationFailure {
    pub(crate) fn inner(&self) -> &InnerFailure {
        match self {
            Self::PathBudgetExhausted { last, .. }
            | Self::PathStuck { last, .. }
            | Self::StructuralPropagate(last)
            | Self::DomainAtOversmoothedStart(last) => last,
        }
    }
}

/// Accepted state carried across continuation steps. Stage 2 carries
/// the contractually-available payload (β, ρ, the OuterEval).
/// Active-set + trust-radius warm-start ride on later extensions of
/// the `OuterObjective::seed_inner_state` contract.
#[derive(Debug, Clone)]
pub(crate) struct ContinuationState {
    pub last_rho: Array1<f64>,
    pub last_eval: OuterEval,
    pub last_beta: Array1<f64>,
    pub steps_accepted: usize,
}

#[derive(Debug, Clone, Copy)]
enum FailureAction {
    ShrinkStep,
    ShrinkOrExpand,
    Propagate,
    ExpandRhoZero,
}

fn classify_action(failure: &InnerFailure) -> FailureAction {
    match failure {
        InnerFailure::CertRefused { diagnosis, .. } => match diagnosis {
            KktRefusalDiagnosis::RankDeficientHPen => FailureAction::ExpandRhoZero,
            KktRefusalDiagnosis::ActiveSetIncomplete => FailureAction::Propagate,
            KktRefusalDiagnosis::PhantomMultiplierWithWellConditionedH => {
                FailureAction::ShrinkStep
            }
        },
        InnerFailure::BudgetExhausted { .. } => FailureAction::ShrinkStep,
        InnerFailure::TrustRegionFloor { .. } => FailureAction::ShrinkOrExpand,
        InnerFailure::LikelihoodFailure(_) => FailureAction::ShrinkStep,
        InnerFailure::Other(_) => FailureAction::Propagate,
    }
}

fn build_rho_zero(target: &Array1<f64>, upper: &Array1<f64>, offset: f64) -> Array1<f64> {
    debug_assert_eq!(target.len(), upper.len());
    let mut rho0 = target.clone();
    for i in 0..rho0.len() {
        let candidate = target[i] + offset;
        rho0[i] = candidate.min(upper[i]);
    }
    rho0
}

fn rho_zero_is_target(rho0: &Array1<f64>, target: &Array1<f64>) -> bool {
    debug_assert_eq!(rho0.len(), target.len());
    rho0.iter()
        .zip(target.iter())
        .all(|(a, b)| (a - b).abs() <= RHO_EQUAL_TOL)
}

fn step_toward(rho_k: &Array1<f64>, target: &Array1<f64>, alpha: f64) -> Array1<f64> {
    debug_assert_eq!(rho_k.len(), target.len());
    let mut out = Array1::<f64>::zeros(rho_k.len());
    for i in 0..rho_k.len() {
        out[i] = rho_k[i] + alpha * (target[i] - rho_k[i]);
    }
    out
}

fn reached_target(rho: &Array1<f64>, target: &Array1<f64>) -> bool {
    let tol = RHO_EQUAL_TOL / 8.0;
    rho.iter()
        .zip(target.iter())
        .all(|(a, b)| (a - b).abs() <= tol)
}

fn inner_failure_from(err: EstimationError) -> InnerFailure {
    match err {
        EstimationError::RemlOptimizationFailed(msg) => classify_inner_error(msg),
        other => InnerFailure::Other(other.to_string()),
    }
}

fn eval_step(
    obj: &mut dyn OuterObjective,
    rho: &Array1<f64>,
    beta_seed: &Array1<f64>,
    order: OuterEvalOrder,
) -> Result<OuterEval, InnerFailure> {
    if let Err(e) = obj.seed_inner_state(beta_seed) {
        return Err(inner_failure_from(e));
    }
    obj.eval_with_order(rho, order).map_err(inner_failure_from)
}

pub(crate) type ContinuationResult = Result<ContinuationState, ContinuationFailure>;

/// Run the continuation path from an oversmoothing ρ₀ down to `target`.
/// `initial_beta` seeds the inner solve at ρ₀ (zero vector is fine —
/// ρ₀ is in the strongly-convex regime). `bounds_upper` clamps ρ₀ to
/// the legal box.
pub(crate) fn fit_with_continuation(
    obj: &mut dyn OuterObjective,
    target: &Array1<f64>,
    bounds_upper: &Array1<f64>,
    initial_beta: &Array1<f64>,
    order: OuterEvalOrder,
) -> ContinuationResult {
    if target.len() != bounds_upper.len() {
        return Err(ContinuationFailure::StructuralPropagate(InnerFailure::Other(
            format!(
                "continuation: target len {} != bounds_upper len {}",
                target.len(),
                bounds_upper.len()
            ),
        )));
    }

    let mut offset = OVERSMOOTH_OFFSET_INIT;

    for retry in 0..=OVERSMOOTH_RETRY_MAX {
        match run_path(obj, target, bounds_upper, initial_beta, order, offset) {
            Ok(state) => return Ok(state),
            Err(PathOutcome::ExpandRhoZero(last)) | Err(PathOutcome::Stuck(last)) => {
                if retry == OVERSMOOTH_RETRY_MAX {
                    let final_rho = build_rho_zero(target, bounds_upper, offset);
                    return Err(ContinuationFailure::PathStuck {
                        last,
                        rho_zero_offset: offset,
                        final_rho,
                    });
                }
                offset *= 2.0;
            }
            Err(PathOutcome::PathBudgetExhausted {
                last,
                steps_taken,
                final_rho,
            }) => {
                return Err(ContinuationFailure::PathBudgetExhausted {
                    last,
                    steps_taken,
                    final_rho,
                });
            }
            Err(PathOutcome::Propagate(last)) => {
                return Err(ContinuationFailure::StructuralPropagate(last));
            }
            Err(PathOutcome::DomainAtStart(last)) => {
                if retry == OVERSMOOTH_RETRY_MAX {
                    return Err(ContinuationFailure::DomainAtOversmoothedStart(last));
                }
                offset *= 2.0;
            }
        }
    }

    // Loop above always returns; this is a structural impossibility.
    // Surface the structural error rather than panicking so a future
    // refactor that changes the retry shape can't quietly drop it.
    Err(ContinuationFailure::PathStuck {
        last: InnerFailure::Other("continuation: retry loop ended unexpectedly".into()),
        rho_zero_offset: offset,
        final_rho: build_rho_zero(target, bounds_upper, offset),
    })
}

enum PathOutcome {
    ExpandRhoZero(InnerFailure),
    Stuck(InnerFailure),
    DomainAtStart(InnerFailure),
    Propagate(InnerFailure),
    PathBudgetExhausted {
        last: InnerFailure,
        steps_taken: usize,
        final_rho: Array1<f64>,
    },
}

fn run_path(
    obj: &mut dyn OuterObjective,
    target: &Array1<f64>,
    bounds_upper: &Array1<f64>,
    initial_beta: &Array1<f64>,
    order: OuterEvalOrder,
    offset: f64,
) -> Result<ContinuationState, PathOutcome> {
    let rho0 = build_rho_zero(target, bounds_upper, offset);
    let collapsed = rho_zero_is_target(&rho0, target);
    let rho_first = if collapsed { target.clone() } else { rho0 };

    let mut beta_seed = initial_beta.clone();

    let eval0 = match eval_step(obj, &rho_first, &beta_seed, order) {
        Ok(eval) => eval,
        Err(failure) => {
            return Err(match failure {
                InnerFailure::LikelihoodFailure(_) => PathOutcome::DomainAtStart(failure),
                InnerFailure::CertRefused {
                    diagnosis: KktRefusalDiagnosis::ActiveSetIncomplete,
                    ..
                } => PathOutcome::Propagate(failure),
                // RankDeficientHPen at ρ₀ → definitely expand. Other
                // failures at the most oversmoothed point are also
                // unusual (we're in the strongly-convex regime);
                // treat as "expand ρ₀ and retry" rather than give up.
                _ => PathOutcome::ExpandRhoZero(failure),
            });
        }
    };

    let mut state = ContinuationState {
        last_rho: rho_first,
        last_eval: eval0,
        last_beta: beta_seed.clone(),
        steps_accepted: 1,
    };

    if collapsed || reached_target(&state.last_rho, target) {
        return Ok(state);
    }

    let mut alpha = ALPHA_INIT;
    let mut steps_taken: usize = 1;
    let mut last_failure: Option<InnerFailure> = None;
    let mut consecutive_trust_floor: usize = 0;

    while steps_taken < PATH_BUDGET {
        if reached_target(&state.last_rho, target) {
            return Ok(state);
        }

        let rho_next = step_toward(&state.last_rho, target, alpha);
        // Prefer the previous eval's published inner-β hint over our
        // own carried β. The objective itself knows its converged β at
        // ρ_k; if it surfaces it, that is the best warm-start for ρ_{k+1}.
        beta_seed = state
            .last_eval
            .inner_beta_hint
            .clone()
            .unwrap_or_else(|| state.last_beta.clone());

        match eval_step(obj, &rho_next, &beta_seed, order) {
            Ok(eval) => {
                state.last_rho = rho_next;
                state.last_eval = eval;
                state.last_beta = beta_seed;
                state.steps_accepted += 1;
                steps_taken += 1;
                last_failure = None;
                consecutive_trust_floor = 0;
                alpha = (alpha * ALPHA_EXPAND).min(1.0);
            }
            Err(failure) => {
                last_failure = Some(failure.clone());
                match classify_action(&failure) {
                    FailureAction::Propagate => {
                        return Err(PathOutcome::Propagate(failure));
                    }
                    FailureAction::ExpandRhoZero => {
                        return Err(PathOutcome::ExpandRhoZero(failure));
                    }
                    FailureAction::ShrinkStep => {
                        alpha *= ALPHA_SHRINK;
                        if alpha < ALPHA_FLOOR {
                            return Err(PathOutcome::Stuck(failure));
                        }
                        steps_taken += 1;
                    }
                    FailureAction::ShrinkOrExpand => {
                        consecutive_trust_floor += 1;
                        if consecutive_trust_floor >= 2 {
                            return Err(PathOutcome::ExpandRhoZero(failure));
                        }
                        alpha *= ALPHA_SHRINK;
                        if alpha < ALPHA_FLOOR {
                            return Err(PathOutcome::Stuck(failure));
                        }
                        steps_taken += 1;
                    }
                }
            }
        }
    }

    Err(PathOutcome::PathBudgetExhausted {
        last: last_failure.unwrap_or_else(|| {
            InnerFailure::Other("continuation: budget hit without recorded failure".into())
        }),
        steps_taken,
        final_rho: state.last_rho.clone(),
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn rho_zero_collapses_when_target_at_upper_bound() {
        let target = Array1::from_vec(vec![5.0, 5.0]);
        let upper = Array1::from_vec(vec![5.0, 5.0]);
        let rho0 = build_rho_zero(&target, &upper, OVERSMOOTH_OFFSET_INIT);
        assert_eq!(rho0, target);
        assert!(rho_zero_is_target(&rho0, &target));
    }

    #[test]
    fn rho_zero_offsets_above_target_when_room() {
        let target = Array1::from_vec(vec![0.0, -2.0]);
        let upper = Array1::from_vec(vec![10.0, 10.0]);
        let rho0 = build_rho_zero(&target, &upper, OVERSMOOTH_OFFSET_INIT);
        assert!((rho0[0] - OVERSMOOTH_OFFSET_INIT).abs() < 1e-12);
        assert!((rho0[1] - (-2.0 + OVERSMOOTH_OFFSET_INIT)).abs() < 1e-12);
        assert!(!rho_zero_is_target(&rho0, &target));
    }

    #[test]
    fn step_toward_is_convex_combination() {
        let a = Array1::from_vec(vec![0.0, 0.0]);
        let b = Array1::from_vec(vec![4.0, -8.0]);
        let mid = step_toward(&a, &b, 0.5);
        assert!((mid[0] - 2.0).abs() < 1e-12);
        assert!((mid[1] - (-4.0)).abs() < 1e-12);
        let full = step_toward(&a, &b, 1.0);
        assert!((full[0] - 4.0).abs() < 1e-12);
        assert!((full[1] - (-8.0)).abs() < 1e-12);
    }

    #[test]
    fn classify_action_routes_diagnoses_correctly() {
        let rank_def = InnerFailure::CertRefused {
            diagnosis: KktRefusalDiagnosis::RankDeficientHPen,
            carrying_block: None,
            message: "".into(),
        };
        assert!(matches!(
            classify_action(&rank_def),
            FailureAction::ExpandRhoZero
        ));

        let active_incomp = InnerFailure::CertRefused {
            diagnosis: KktRefusalDiagnosis::ActiveSetIncomplete,
            carrying_block: None,
            message: "".into(),
        };
        assert!(matches!(
            classify_action(&active_incomp),
            FailureAction::Propagate
        ));

        let phantom = InnerFailure::CertRefused {
            diagnosis: KktRefusalDiagnosis::PhantomMultiplierWithWellConditionedH,
            carrying_block: None,
            message: "".into(),
        };
        assert!(matches!(
            classify_action(&phantom),
            FailureAction::ShrinkStep
        ));

        assert!(matches!(
            classify_action(&InnerFailure::BudgetExhausted {
                message: "".into()
            }),
            FailureAction::ShrinkStep
        ));
        assert!(matches!(
            classify_action(&InnerFailure::TrustRegionFloor {
                message: "".into()
            }),
            FailureAction::ShrinkOrExpand
        ));
        assert!(matches!(
            classify_action(&InnerFailure::LikelihoodFailure("".into())),
            FailureAction::ShrinkStep
        ));
        assert!(matches!(
            classify_action(&InnerFailure::Other("".into())),
            FailureAction::Propagate
        ));
    }
}
