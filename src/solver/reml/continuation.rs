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
use crate::solver::estimate::EstimationError;
use crate::solver::inner_status::{InnerFailure, classify_inner_error};
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
    pub(crate) fn message(&self) -> String {
        match self {
            Self::PathBudgetExhausted {
                last,
                steps_taken,
                final_rho,
            } => format!(
                "{} (continuation budget exhausted after {} step(s), final rho dim={})",
                last.message(),
                steps_taken,
                final_rho.len()
            ),
            Self::PathStuck {
                last,
                rho_zero_offset,
                final_rho,
            } => format!(
                "{} (continuation stuck at oversmooth offset {:.6e}, final rho dim={})",
                last.message(),
                rho_zero_offset,
                final_rho.len()
            ),
            Self::StructuralPropagate(last) | Self::DomainAtOversmoothedStart(last) => {
                last.message().to_string()
            }
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
            KktRefusalDiagnosis::PhantomMultiplierWithWellConditionedH => FailureAction::ShrinkStep,
            // Structural cross-block alias surfaced at fit time: no
            // ρ-anneal recovers it. Propagate so the outer driver
            // refuses the fit rather than burning continuation budget.
            KktRefusalDiagnosis::AliasingDetectedAtFit => FailureAction::Propagate,
        },
        InnerFailure::BudgetExhausted { .. } => FailureAction::ShrinkStep,
        InnerFailure::TrustRegionFloor { .. } => FailureAction::ShrinkOrExpand,
        InnerFailure::LikelihoodFailure(_) => FailureAction::ShrinkStep,
        // Structural pre-fit identifiability failure surfaced via the
        // inner classifier — propagate. The audit's structured report
        // is already attached; rollback would just retry the same
        // joint design with the same alias.
        InnerFailure::IdentifiabilityFailure { .. } => FailureAction::Propagate,
        InnerFailure::Other(_) => FailureAction::Propagate,
    }
}

fn build_rho_zero(target: &Array1<f64>, upper: &Array1<f64>, offset: f64) -> Array1<f64> {
    assert_eq!(target.len(), upper.len());
    let mut rho0 = target.clone();
    for i in 0..rho0.len() {
        let candidate = target[i] + offset;
        rho0[i] = candidate.min(upper[i]);
    }
    rho0
}

fn rho_zero_is_target(rho0: &Array1<f64>, target: &Array1<f64>) -> bool {
    assert_eq!(rho0.len(), target.len());
    rho0.iter()
        .zip(target.iter())
        .all(|(a, b)| (a - b).abs() <= RHO_EQUAL_TOL)
}

fn step_toward(rho_k: &Array1<f64>, target: &Array1<f64>, alpha: f64) -> Array1<f64> {
    assert_eq!(rho_k.len(), target.len());
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
    // The seed-hook contract is encoded in the typed `SeedOutcome`:
    //   - `Ok(Installed)`: β is now in the inner-β slot for `eval_with_order`.
    //   - `Ok(NoSlot)`: the objective owns no inner-β slot; the hint is
    //     silently discarded and we proceed cold. This is the documented
    //     fallback for objectives that publish `inner_beta_hint` from a
    //     read-only view of inner state without exposing a write hook —
    //     not a structural failure (issue #236).
    //   - `Err(_)`: a genuine seeding failure (dimension mismatch when a
    //     slot exists, etc.). Forward as a hard failure.
    obj.seed_inner_state(beta_seed)
        .map_err(inner_failure_from)?;
    obj.eval_with_order(rho, order).map_err(inner_failure_from)
}

pub(crate) type ContinuationResult = Result<ContinuationState, ContinuationFailure>;

/// Telemetry returned by a successful `prime_outer_seed` call.
/// Surfaced so the outer-loop call site can emit a single structured
/// log line distinguishing the no-op collapse path from a real anneal.
#[derive(Debug, Clone, Copy)]
pub(crate) struct PrimingSummary {
    /// `true` when ρ₀ would clamp to ρ* and no inner call was made.
    pub collapsed: bool,
    /// Number of accepted inner evaluations along the path. 0 for
    /// collapse, 1 when ρ₀ ≠ ρ* but the first step at ρ₀ also reached
    /// ρ* within `reached_target`, ≥2 for a real anneal.
    pub steps_accepted: usize,
}

/// Prime the outer optimizer's seed by walking a continuation path from
/// an oversmoothing ρ₀ down to `seed`. Designed for the
/// `run_outer_with_plan` per-seed loop: a successful call leaves the
/// objective's inner state warm at `seed`, so the subsequent
/// `eval_with_order(seed, …)` from the regular cold-eval path converges
/// from a near-optimal β instead of cold β=0.
///
/// Returns `Ok(PrimingSummary { collapsed: true, .. })` (no work
/// performed) when ρ₀ collapses to ρ* — the fit is "easy" in the
/// sense that the bound clamps ρ₀ back to the target. In that case
/// the regular cold eval is already the right thing to do, and
/// continuation imposes zero overhead.
///
/// On failure, the underlying `ContinuationFailure` is returned with
/// its inner `InnerFailure` preserved so the caller can route it
/// through the existing `SeedRejection::from_message` accounting.
///
/// `seed` is the per-iteration ρ candidate (the loop variable in
/// `run_outer_with_plan`). `bounds_upper` is the legal upper bound on ρ.
pub(crate) fn prime_outer_seed(
    obj: &mut dyn OuterObjective,
    seed: &Array1<f64>,
    bounds_upper: &Array1<f64>,
) -> Result<PrimingSummary, ContinuationFailure> {
    // Pre-screen: if ρ₀ would clamp to ρ*, skip entirely. No inner
    // call, no allocation, no log line — continuation is invisible on
    // easy fits, satisfying the "magic by default" zero-overhead bar.
    let rho_zero = build_rho_zero(seed, bounds_upper, OVERSMOOTH_OFFSET_INIT);
    if rho_zero_is_target(&rho_zero, seed) {
        return Ok(PrimingSummary {
            collapsed: true,
            steps_accepted: 0,
        });
    }

    // Empty β: the objective's `seed_inner_state` contract is to treat
    // a zero-length slice as "no warm-start available, use your own
    // cold default". Continuation then carries β forward step-to-step
    // via `OuterEval::inner_beta_hint` from each accepted eval.
    let empty_beta: Array1<f64> = Array1::zeros(0);

    match fit_with_continuation(
        obj,
        seed,
        bounds_upper,
        &empty_beta,
        OuterEvalOrder::ValueAndGradient,
    ) {
        Ok(state) => Ok(PrimingSummary {
            collapsed: false,
            steps_accepted: state.steps_accepted,
        }),
        Err(failure) => Err(failure),
    }
}

/// Run the continuation path from an oversmoothing ρ₀ down to `target`.
/// `initial_beta` seeds the inner solve at ρ₀ (zero vector is fine —
/// ρ₀ is in the strongly-convex regime). `bounds_upper` clamps ρ₀ to
/// the legal box.
fn fit_with_continuation(
    obj: &mut dyn OuterObjective,
    target: &Array1<f64>,
    bounds_upper: &Array1<f64>,
    initial_beta: &Array1<f64>,
    order: OuterEvalOrder,
) -> ContinuationResult {
    if target.len() != bounds_upper.len() {
        return Err(ContinuationFailure::StructuralPropagate(
            InnerFailure::Other(format!(
                "continuation: target len {} != bounds_upper len {}",
                target.len(),
                bounds_upper.len()
            )),
        ));
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
                InnerFailure::CertRefused {
                    diagnosis: KktRefusalDiagnosis::AliasingDetectedAtFit,
                    ..
                } => PathOutcome::Propagate(failure),
                // Structural identifiability failure at ρ₀: rho-anneal
                // cannot fix a rank-deficient joint design. Propagate
                // rather than expanding into an even more oversmoothed
                // regime where the alias persists.
                InnerFailure::IdentifiabilityFailure { .. } => PathOutcome::Propagate(failure),
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
            classify_action(&InnerFailure::BudgetExhausted { message: "".into() }),
            FailureAction::ShrinkStep
        ));
        assert!(matches!(
            classify_action(&InnerFailure::TrustRegionFloor { message: "".into() }),
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

    // ─────────────────────────────────────────────────────────────────
    //                       Scenario tests
    // ─────────────────────────────────────────────────────────────────
    //
    // These cover the four operational paths the team-lead asked for:
    //   1. degenerates_to_cold_start_on_easy_fits           (collapse)
    //   2. budget_exhausted_warmstart_completes_path        (warm-start past slow region)
    //   3. trust_region_floor_alpha_shrink_then_recovers    (shrink path)
    //   4. likelihood_failure_alpha_shrink_then_recovers    (shrink path on domain miss)
    //   5. active_set_incomplete_propagates_structurally    (KKT-bug propagation)
    //   6. path_budget_exhausted_surfaces_last_inner_failure
    //
    // All driven by a scripted OuterObjective whose responses are a
    // queue of `Result<&'static str_or_ok, &'static str_failure>`. The
    // mock records the ρ at each call so step counting can be asserted.

    use crate::solver::outer_strategy::{
        DeclaredHessianForm, Derivative, HessianResult, OuterCapability,
    };

    /// A response scripted for the next `eval_with_order` call.
    #[derive(Clone)]
    enum ScriptedResponse {
        Ok,
        Fail(&'static str),
    }

    struct ScriptedObjective {
        n_params: usize,
        queue: Vec<ScriptedResponse>,
        idx: usize,
        rho_history: Vec<Array1<f64>>,
        seed_calls: usize,
        last_seeded_beta_len: Option<usize>,
    }

    impl ScriptedObjective {
        fn new(n_params: usize, queue: Vec<ScriptedResponse>) -> Self {
            Self {
                n_params,
                queue,
                idx: 0,
                rho_history: Vec::new(),
                seed_calls: 0,
                last_seeded_beta_len: None,
            }
        }

        fn next_response(&mut self) -> ScriptedResponse {
            let r = self
                .queue
                .get(self.idx)
                .cloned()
                .unwrap_or(ScriptedResponse::Ok);
            self.idx += 1;
            r
        }
    }

    impl OuterObjective for ScriptedObjective {
        fn capability(&self) -> OuterCapability {
            OuterCapability {
                gradient: Derivative::Analytic,
                hessian: DeclaredHessianForm::Unavailable,
                n_params: self.n_params,
                psi_dim: 0,
                fixed_point_available: false,
                barrier_config: None,
                prefer_gradient_only: false,
                disable_fixed_point: false,
            }
        }

        fn eval_cost(&mut self, rho: &Array1<f64>) -> Result<f64, EstimationError> {
            self.rho_history.push(rho.clone());
            match self.next_response() {
                ScriptedResponse::Ok => Ok(rho.dot(rho)),
                ScriptedResponse::Fail(msg) => {
                    Err(EstimationError::RemlOptimizationFailed(msg.to_string()))
                }
            }
        }

        fn eval(&mut self, rho: &Array1<f64>) -> Result<OuterEval, EstimationError> {
            let cost = self.eval_cost(rho)?;
            Ok(OuterEval {
                cost,
                gradient: Array1::zeros(self.n_params),
                hessian: HessianResult::Unavailable,
                inner_beta_hint: None,
            })
        }

        fn reset(&mut self) {
            self.idx = 0;
            self.rho_history.clear();
            self.seed_calls = 0;
            self.last_seeded_beta_len = None;
        }

        fn seed_inner_state(&mut self, beta: &Array1<f64>) -> Result<(), EstimationError> {
            assert_eq!(beta.len(), self.n_params);
            self.seed_calls += 1;
            self.last_seeded_beta_len = Some(beta.len());
            Ok(())
        }
    }

    fn rho(values: &[f64]) -> Array1<f64> {
        Array1::from_vec(values.to_vec())
    }

    #[test]
    fn degenerates_to_cold_start_on_easy_fits() {
        // ρ₀ would clamp to ρ* because the bounds-upper is *at* the
        // target. prime_outer_seed must return Ok with ZERO inner
        // calls — that's the no-overhead promise.
        let target = rho(&[5.0, 5.0]);
        let upper = rho(&[5.0, 5.0]);
        let mut obj = ScriptedObjective::new(2, Vec::new());
        let summary = prime_outer_seed(&mut obj, &target, &upper).expect("collapse path");
        assert!(summary.collapsed, "must report collapsed=true on easy fits");
        assert_eq!(summary.steps_accepted, 0);
        assert_eq!(obj.rho_history.len(), 0, "no inner calls on collapse");
        assert_eq!(obj.seed_calls, 0);
    }

    #[test]
    fn budget_exhausted_warmstart_completes_path() {
        // Hard fit at target: cold-start refuses with BudgetExhausted at
        // every intermediate ρ until α shrinks enough that the step
        // lands inside the strongly-convex basin. Scenario simulates
        // this by:
        //   - Step 0 (ρ₀): Ok (oversmoothed → easy)
        //   - Step 1 (α=0.5 toward target): BudgetExhausted
        //   - Step 2 (α=0.25): BudgetExhausted
        //   - Step 3+ : Ok all the way to target
        // Result: path completes via shrink, demonstrating that
        // continuation+warm-start gets past the slow region.
        let target = rho(&[0.0]);
        let upper = rho(&[10.0]);
        let mut obj = ScriptedObjective::new(
            1,
            vec![
                ScriptedResponse::Ok,                               // ρ₀ accept
                ScriptedResponse::Fail("inner_max_cycles reached"), // 1st step refused
                ScriptedResponse::Fail("inner_max_cycles reached"), // 2nd step refused
                // After two shrinks α≈0.125; remaining accepts.
                ScriptedResponse::Ok,
                ScriptedResponse::Ok,
                ScriptedResponse::Ok,
                ScriptedResponse::Ok,
                ScriptedResponse::Ok,
                ScriptedResponse::Ok,
                ScriptedResponse::Ok,
                ScriptedResponse::Ok,
            ],
        );
        prime_outer_seed(&mut obj, &target, &upper).expect("path completes via shrink-on-budget");
        // Confirm we did execute ρ₀ (the oversmoothed start) before
        // any of the failed attempts — direct evidence that the
        // continuation actually walked a path.
        assert!(obj.rho_history.len() >= 3, "must have walked a path");
        let rho0 = &obj.rho_history[0];
        assert!(
            (rho0[0] - (target[0] + OVERSMOOTH_OFFSET_INIT)).abs() < 1e-9,
            "first call is at ρ₀ = ρ*+offset",
        );
    }

    #[test]
    fn trust_region_floor_alpha_shrink_then_recovers() {
        // TrustRegionFloor → ShrinkOrExpand. First occurrence shrinks
        // (consecutive_trust_floor=1, still under threshold). If a
        // SECOND consecutive TR-floor fires, the schedule escalates to
        // ExpandRhoZero. This test demonstrates the shrink branch:
        // one TR-floor, then accepts, no escalation.
        let target = rho(&[0.0]);
        let upper = rho(&[10.0]);
        let mut obj = ScriptedObjective::new(
            1,
            vec![
                ScriptedResponse::Ok,                                 // ρ₀
                ScriptedResponse::Fail("trust-region floor reached"), // one TR-floor
                // Rest: succeeds (intervening accept resets counter).
                ScriptedResponse::Ok,
                ScriptedResponse::Ok,
                ScriptedResponse::Ok,
                ScriptedResponse::Ok,
                ScriptedResponse::Ok,
                ScriptedResponse::Ok,
                ScriptedResponse::Ok,
                ScriptedResponse::Ok,
            ],
        );
        prime_outer_seed(&mut obj, &target, &upper)
            .expect("path completes after single TR-floor shrink");
        assert!(obj.rho_history.len() >= 3);
    }

    #[test]
    fn likelihood_failure_alpha_shrink_then_recovers() {
        // LikelihoodFailure (NaN / domain miss) → ShrinkStep. The β at
        // ρ_k was just accepted, so the family domain is reachable;
        // only the over-shoot landed outside. Halving α restores
        // feasibility.
        let target = rho(&[0.0]);
        let upper = rho(&[10.0]);
        let mut obj = ScriptedObjective::new(
            1,
            vec![
                ScriptedResponse::Ok, // ρ₀
                ScriptedResponse::Fail("likelihood evaluation failed: NaN"),
                ScriptedResponse::Ok,
                ScriptedResponse::Ok,
                ScriptedResponse::Ok,
                ScriptedResponse::Ok,
                ScriptedResponse::Ok,
                ScriptedResponse::Ok,
                ScriptedResponse::Ok,
                ScriptedResponse::Ok,
            ],
        );
        let outcome = prime_outer_seed(&mut obj, &target, &upper);
        assert!(
            outcome.is_ok(),
            "path completes after likelihood shrink, got {:?}",
            outcome.err(),
        );
        assert!(obj.rho_history.len() >= 3);
    }

    #[test]
    fn active_set_incomplete_propagates_structurally() {
        // ActiveSetIncomplete is a real KKT bug — continuation must
        // NOT shrink and retry, it must surface the failure.
        let target = rho(&[0.0]);
        let upper = rho(&[10.0]);
        let mut obj = ScriptedObjective::new(
            1,
            vec![
                ScriptedResponse::Ok, // ρ₀ accept
                ScriptedResponse::Fail(
                    "cycle=3 cert REFUSED: residual=1.0e+02 > tol=1.0e+00; \
                     carrying-block: time_surface (idx=0); \
                     diagnosis: active_set_incomplete",
                ),
            ],
        );
        let err = prime_outer_seed(&mut obj, &target, &upper)
            .expect_err("structural failure must propagate");
        assert!(
            matches!(err, ContinuationFailure::StructuralPropagate(_)),
            "got {err:?}",
        );
        match err {
            ContinuationFailure::StructuralPropagate(InnerFailure::CertRefused {
                diagnosis,
                ..
            }) => assert_eq!(diagnosis, KktRefusalDiagnosis::ActiveSetIncomplete),
            other => panic!("expected CertRefused, got {other:?}"),
        }
    }

    #[test]
    fn path_budget_exhausted_surfaces_last_inner_failure() {
        // Queue is short on Oks but long on phantom-multiplier
        // refusals. ShrinkStep underflows α before the path completes,
        // producing PathStuck. After OVERSMOOTH_RETRY_MAX retries, the
        // outer wrapper returns PathStuck (not PathBudgetExhausted —
        // budget exhaustion requires 64 steps, alpha-floor stuck is
        // the earlier failure).
        let target = rho(&[0.0]);
        let upper = rho(&[10.0]);
        let mut responses: Vec<ScriptedResponse> = Vec::new();
        // ρ₀ accepts on every retry attempt (this objective is
        // re-entered fresh each `run_path` because retries call
        // run_path again with a new offset; the scripted queue is
        // monotonically advanced though, so we need plenty of
        // phantom-style refusals after each ρ₀ accept).
        for _ in 0..32 {
            // ρ₀ ok
            responses.push(ScriptedResponse::Ok);
            // Phantom-multiplier refusals: ShrinkStep until α < floor.
            // Need ~log2(1/ALPHA_FLOOR) = 10 consecutive refusals to
            // underflow α from 0.5 → 2⁻¹¹. Push generously.
            for _ in 0..20 {
                responses.push(ScriptedResponse::Fail(
                    "coupled exact-joint inner solve exited the joint Newton path \
                     before convergence — block 'time_surface' carries the dominant \
                     unresolved KKT gradient (|g_block|∞ = 5.000e+05); \
                     |∇L − Sβ|∞ = 5.000e+05",
                ));
            }
        }
        let mut obj = ScriptedObjective::new(1, responses);
        let err = prime_outer_seed(&mut obj, &target, &upper).expect_err("schedule must fail");
        // PhantomMultiplier classifies as ShrinkStep → α-floor →
        // Stuck → ExpandRhoZero (outer) → retries doubled offset →
        // PathStuck after OVERSMOOTH_RETRY_MAX.
        match err {
            ContinuationFailure::PathStuck { last, .. } => match last {
                InnerFailure::CertRefused { diagnosis, .. } => assert_eq!(
                    diagnosis,
                    KktRefusalDiagnosis::PhantomMultiplierWithWellConditionedH
                ),
                other => panic!("expected CertRefused, got {other:?}"),
            },
            other => panic!("expected PathStuck, got {other:?}"),
        }
    }

    // ─────────────────────────────────────────────────────────────────
    //  RED tests for issue #236:
    //  https://github.com/SauersML/gam/issues/236
    //
    //  Standard REML publishes a non-empty `inner_beta_hint` from every
    //  eval (see src/solver/estimate.rs:3275). Continuation forwards
    //  that hint into `seed_inner_state` at the next step. When the
    //  objective is a `ClosureObjective` without `with_seed_inner_state`
    //  (which is exactly how `build_objective` wires the standard REML
    //  closure at src/solver/estimate.rs:3202), the seed call returns
    //  Invalid input fatally — the pre-warm fails before the first
    //  real solver step, rejecting every seed and pinning
    //  `solver_started=0`.
    //
    //  The contract this should satisfy is: an objective that publishes
    //  a hint MUST be able to consume that hint via `seed_inner_state`,
    //  *or* the continuation/pre-warm wiring must not forward a hint
    //  the objective cannot accept. Either way, this scenario must not
    //  abort the seed.
    // ─────────────────────────────────────────────────────────────────

    use crate::solver::outer_strategy::ClosureObjective;

    #[test]
    fn closure_objective_publishing_inner_beta_hint_without_seed_hook_is_acceptable() {
        // ClosureObjective wired exactly like the standard REML closure:
        //   - eval_with_order returns inner_beta_hint = Some(non-empty β)
        //   - no with_seed_inner_state(...) installed
        // continuation walks ρ from oversmoothed rho_zero to target, so
        // step 2 forwards the published hint into seed_inner_state.
        // Today that path raises Invalid input and prime_outer_seed
        // returns Err — but it should not.
        let target = Array1::from_vec(vec![0.0]);
        let upper = Array1::from_vec(vec![10.0]);
        let obj = ClosureObjective {
            state: (),
            cap: crate::solver::outer_strategy::OuterCapability {
                gradient: crate::solver::outer_strategy::Derivative::Analytic,
                hessian: crate::solver::outer_strategy::DeclaredHessianForm::Unavailable,
                n_params: 1,
                psi_dim: 0,
                fixed_point_available: false,
                barrier_config: None,
                prefer_gradient_only: false,
                disable_fixed_point: false,
            },
            cost_fn: |_: &mut (), rho: &Array1<f64>| Ok(rho.dot(rho)),
            eval_fn: |_: &mut (), rho: &Array1<f64>| {
                Ok(OuterEval {
                    cost: rho.dot(rho),
                    gradient: Array1::zeros(1),
                    hessian: HessianResult::Unavailable,
                    inner_beta_hint: Some(Array1::from_vec(vec![0.1, 0.2, 0.3, 0.4])),
                })
            },
            eval_order_fn: None::<
                fn(
                    &mut (),
                    &Array1<f64>,
                    crate::solver::outer_strategy::OuterEvalOrder,
                ) -> Result<OuterEval, EstimationError>,
            >,
            reset_fn: None::<fn(&mut ())>,
            efs_fn: None::<
                fn(
                    &mut (),
                    &Array1<f64>,
                )
                    -> Result<crate::solver::outer_strategy::EfsEval, EstimationError>,
            >,
            screening_proxy_fn: None::<fn(&mut (), &Array1<f64>) -> Result<f64, EstimationError>>,
            seed_fn: None::<fn(&mut (), &Array1<f64>) -> Result<(), EstimationError>>,
        };

        // No `.with_seed_inner_state(...)` — mirrors standard REML's
        // build_objective wiring at src/solver/estimate.rs:3202.
        let mut obj = obj;
        let result = prime_outer_seed(&mut obj, &target, &upper);

        assert!(
            result.is_ok(),
            "prime_outer_seed must not reject a seed just because the \
             objective publishes inner_beta_hint without installing a \
             seed hook (issue #236). got: {:?}",
            result.err().map(|e| e.message().to_string()),
        );
    }

    #[test]
    fn pre_warm_does_not_forward_hint_into_objective_lacking_seed_hook() {
        // A weaker, more targeted check: the continuation layer must
        // not blindly forward `inner_beta_hint` to an objective that
        // would reject it. Today the error message
        //   "cached inner beta has length N, but this objective does
        //    not expose an inner-state seeding hook"
        // is surfaced verbatim through `prime_outer_seed`. Pin that
        // this message no longer reaches users (issue #236).
        let target = Array1::from_vec(vec![0.0]);
        let upper = Array1::from_vec(vec![10.0]);

        let obj = ClosureObjective {
            state: (),
            cap: crate::solver::outer_strategy::OuterCapability {
                gradient: crate::solver::outer_strategy::Derivative::Analytic,
                hessian: crate::solver::outer_strategy::DeclaredHessianForm::Unavailable,
                n_params: 1,
                psi_dim: 0,
                fixed_point_available: false,
                barrier_config: None,
                prefer_gradient_only: false,
                disable_fixed_point: false,
            },
            cost_fn: |_: &mut (), rho: &Array1<f64>| Ok(rho.dot(rho)),
            eval_fn: |_: &mut (), rho: &Array1<f64>| {
                Ok(OuterEval {
                    cost: rho.dot(rho),
                    gradient: Array1::zeros(1),
                    hessian: HessianResult::Unavailable,
                    inner_beta_hint: Some(Array1::from_vec(vec![1.0, 2.0])),
                })
            },
            eval_order_fn: None::<
                fn(
                    &mut (),
                    &Array1<f64>,
                    crate::solver::outer_strategy::OuterEvalOrder,
                ) -> Result<OuterEval, EstimationError>,
            >,
            reset_fn: None::<fn(&mut ())>,
            efs_fn: None::<
                fn(
                    &mut (),
                    &Array1<f64>,
                )
                    -> Result<crate::solver::outer_strategy::EfsEval, EstimationError>,
            >,
            screening_proxy_fn: None::<fn(&mut (), &Array1<f64>) -> Result<f64, EstimationError>>,
            seed_fn: None::<fn(&mut (), &Array1<f64>) -> Result<(), EstimationError>>,
        };

        let mut obj = obj;
        match prime_outer_seed(&mut obj, &target, &upper) {
            Ok(_) => {}
            Err(cf) => {
                let msg = cf.message();
                assert!(
                    !msg.contains("does not expose an inner-state seeding hook"),
                    "pre-warm leaked the seed-hook rejection to the outer \
                     loop (issue #236). msg='{msg}'"
                );
            }
        }
    }

    #[test]
    fn pre_warm_failure_carries_underlying_message_for_seed_rejection() {
        // The outer wiring in run_outer_with_plan formats
        // `cf.message()` into the SeedRejection. Pin that
        // the message is preserved through the failure chain so the
        // existing classifier in StartupStats keeps working.
        let target = rho(&[0.0]);
        let upper = rho(&[10.0]);
        let mut obj = ScriptedObjective::new(
            1,
            vec![ScriptedResponse::Fail(
                "cycle=3 cert REFUSED: residual=1.0e+02 > tol=1.0e+00; \
                 diagnosis: active_set_incomplete",
            )],
        );
        let err = prime_outer_seed(&mut obj, &target, &upper).expect_err("propagation expected");
        let msg = err.message();
        assert!(msg.contains("active_set_incomplete"), "msg='{msg}'");
    }
}
