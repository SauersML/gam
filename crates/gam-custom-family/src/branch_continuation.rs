//! Continuation of the inner coefficient mode along its branch (gam#2973, gam#2366).
//!
//! `V(ρ) = ℓ_p(θ̂(ρ), ρ)` is a function of ρ only once a rule fixes which mode `θ̂(ρ)` names.
//! gam#2366's rule is continuation: `θ̂(ρ)` is the endpoint of the branch followed from the
//! anchor. The fit anchors once, and each outer evaluation then published whichever certified
//! mode its globalized corrector reached from the accepted incumbent's coefficients. Nothing tied
//! that mode to the incumbent's branch, so where the branch ends at a fold the corrector landed
//! on another branch without saying so. On the tilted double well, a walk from the shallow well
//! that accepts a trial past the fold publishes the shallow mode at ρ = 0.9 on the way out and
//! the deep one on the way back (census Slurm 1334963, `branch_walk_probe_2973`).
//!
//! [`continue_branch`] applies the rule inside one evaluation. From an accepted certified mode at
//! `ρ_A` it follows `ρ(t) = ρ_A + t (ρ − ρ_A)`, `t: 0 → 1`. Each sub-step predicts the mode along
//! the IFT tangent `dβ̂/dρ_k = −(H + S_λ)⁻¹ λ_k S_k β̂` of the last certified mode and corrects it
//! with the inner solver. A sub-step is kept only when [`newton_region_contraction`] shows the
//! predictor inside the Newton region of the root the corrector returns; a failed sub-step is
//! halved. Halving has no count: it ends on a certified sub-step or where a sub-step no longer
//! moves `ρ(t)` in floating point. There the branch ends, at a fold
//! ([`BranchContinuationRefusal::FoldReached`]), and its mode drops out of the evaluation's
//! candidates: [`evaluate_on_branch`] publishes the certified rival with the lowest penalized
//! objective, and refuses the trial only when no start certified a mode (gam#3173).
//!
//! Each sub-step costs one inner solve and no pricing. The tangent is formed from the exact-Newton
//! curvature the certified mode's own solve ended on ([`single_block_ift_predictor`]), so a mode
//! needs no derivative-bearing evaluation to be continued from. The corrector solves to the
//! criterion's own accuracy ([`criterion_inner_solve_options`]), and only the published mode is
//! priced, in the caller's evaluation mode, as it is. Taking the tangent from the
//! evaluator's mode responses instead cost two derivative-bearing pricings per value probe on the
//! tilted double well: one to re-derive a seed's tangent (a value-only seed files none) and one
//! per interior sub-step.
//!
//! The test's corrections are Deuflhard's: the Newton correction at the predictor, then the
//! simplified correction at its image with the curvature frozen at the predictor. This slice
//! computes them for single-block solves ([`single_block_simplified_newton_corrections`]).
//! A family whose solve runs the joint Newton path (two coupled blocks or a Hessian-vector
//! workspace), or carries a joint penalty or a Jeffreys term, keeps the per-evaluation rule it had
//! until that path's frozen-curvature re-solve lands.
use super::*;

/// The Newton-region bound on the measured contraction `Θ = ‖Δ¹‖ / ‖Δ⁰‖`.
///
/// Kantorovich, in its affine-covariant form (Deuflhard, *Newton Methods for Nonlinear
/// Problems*): with `ω` the Lipschitz constant of the Jacobian in the corrections' norm and `Δ⁰`
/// the first Newton correction at the predictor, `h₀ = ω‖Δ⁰‖ ≤ ½` guarantees that Newton
/// converges from the predictor to a root that is unique in the ball of radius
/// `‖Δ⁰‖ (1 − √(1 − 2h₀)) / h₀` around it. The simplified correction `Δ̄¹ = −F′(x⁰)⁻¹F(x⁰ + Δ⁰)`,
/// with the Jacobian frozen at the predictor, gives Deuflhard's computational estimate
/// `[ω] = 2‖Δ̄¹‖ / ‖Δ⁰‖²`, so `[h₀] = 2Θ` and `h₀ ≤ ½` reads `Θ ≤ ¼`. The Jacobian must stay frozen:
/// an ordinary second Newton step re-reads the curvature at `x¹`, so a first step that jumped a
/// barrier looks contractive in the basin it landed in (gam#2973 gate 1336821: Θ = 0.18 on the
/// double well's 0.85 → 1.2 fold crossing). `[ω]` bounds `ω` from below, so this is an
/// a-posteriori contraction test, not a rigorous certificate.
pub(crate) const NEWTON_REGION_CONTRACTION_BOUND: f64 = 0.25;

/// The Newton correction and the simplified correction at a predictor, each beside its arithmetic
/// resolution.
#[derive(Clone, Copy, Debug, PartialEq)]
pub(crate) struct NewtonRegionContraction {
    /// `‖Δ⁰‖`, the Newton correction at the predictor.
    pub(crate) first_correction: f64,
    /// `‖Δ̄¹‖`, the simplified correction, with the curvature frozen at the predictor.
    pub(crate) second_correction: f64,
    /// The largest `‖Δ⁰‖` the rounding of its right-hand side and of the iterate it lands on can
    /// produce ([`ExactNewtonBlockUpdater::update_step_with_resolution`]).
    pub(crate) first_resolution: f64,
    /// The same for `‖Δ̄¹‖`.
    pub(crate) second_resolution: f64,
}

impl NewtonRegionContraction {
    /// The measured contraction `Θ = ‖Δ̄¹‖ / ‖Δ⁰‖`, read only where both corrections are resolved.
    ///
    /// A correction at or below its resolution is zero on this arithmetic: the iterate it starts
    /// from is at its root, so Newton has converged there and `Θ = 0`. Above it, `Θ` compares two
    /// resolved corrections. Below it, `Θ` is a ratio of rounding errors: on the event-history
    /// slope surface at `λ = 1.09e9` (gam#2973 comment 5744888545), corrections of `9.2e-11` and
    /// `8.2e-11` against a resolution of `7.1e-9` read `Θ = 0.90`, and the certified mode's own
    /// continuation was refused as a fold. A non-finite correction or resolution measures nothing.
    pub(crate) fn contraction_factor(&self) -> Option<f64> {
        if ![
            self.first_correction,
            self.second_correction,
            self.first_resolution,
            self.second_resolution,
        ]
        .iter()
        .all(|value| value.is_finite())
        {
            return None;
        }
        if self.first_correction <= self.first_resolution
            || self.second_correction <= self.second_resolution
        {
            return Some(0.0);
        }
        Some(self.second_correction / self.first_correction)
    }

    /// Whether the predictor passes the Newton-region contraction test, `Θ ≤ ¼`.
    pub(crate) fn in_newton_region(&self) -> bool {
        self.contraction_factor()
            .is_some_and(|theta| theta <= NEWTON_REGION_CONTRACTION_BOUND)
    }

    /// The radius around the predictor inside which the root the test names lies: the
    /// Kantorovich radius `‖Δ⁰‖ (1 − √(1 − 2[h₀])) / [h₀]` with `[h₀] = 2Θ`, which is `‖Δ⁰‖` at
    /// `Θ = 0` and `2‖Δ⁰‖` at `Θ = ¼`, widened by the two corrections' resolutions, since the
    /// corrections it is measured from are known only to within them. `None` outside the Newton
    /// region.
    pub(crate) fn root_radius(&self) -> Option<f64> {
        if !self.in_newton_region() {
            return None;
        }
        let theta = self.contraction_factor()?;
        let h = 2.0 * theta;
        let kantorovich = if h == 0.0 {
            self.first_correction
        } else {
            self.first_correction * (1.0 - (1.0 - 2.0 * h).sqrt()) / h
        };
        Some(kantorovich + self.first_resolution + self.second_resolution)
    }
}

/// The Newton-region contraction test at one predictor, with the state after its first
/// correction.
pub(crate) struct NewtonRegionTest {
    pub(crate) contraction: NewtonRegionContraction,
    /// The predictor after its Newton correction, `x¹ = x⁰ + Δ⁰`.
    pub(crate) state: ConstrainedWarmStart,
}

/// The Newton-region contraction test of the inner coefficient solve at `predictor`, at the
/// labeled `rho` (gam#2973): the solve's own Newton correction there and Deuflhard's simplified
/// correction after it. This is the one Newton-region test; the #2661 continuation's waypoints
/// call it too.
pub(crate) fn newton_region_contraction<F: CustomFamily + Clone + Send + Sync + 'static>(
    family: &F,
    specs: &[ParameterBlockSpec],
    options: &BlockwiseFitOptions,
    layout: &PenaltyLabelLayout,
    rho: &Array1<f64>,
    predictor: &ConstrainedWarmStart,
) -> Result<NewtonRegionTest, CustomFamilyError> {
    let physical_rho = expand_labeled_log_lambdas(rho, layout)?;
    let per_block = split_log_lambdas(&physical_rho, &layout.penalty_counts)?;
    let labeled_options = labeled_options_for_rho(options, specs, layout, rho)?;
    let corrections = single_block_simplified_newton_corrections(
        family,
        specs,
        &per_block,
        labeled_options.as_ref(),
        &predictor.block_beta,
    )?;
    Ok(NewtonRegionTest {
        contraction: NewtonRegionContraction {
            first_correction: corrections.first_correction,
            second_correction: corrections.second_correction,
            first_resolution: corrections.first_resolution,
            second_resolution: corrections.second_resolution,
        },
        state: ConstrainedWarmStart {
            rho: rho.clone(),
            block_beta: corrections.after_first,
            active_sets: predictor.active_sets.clone(),
            cached_inner: None,
        },
    })
}

/// Euclidean distance between two block coefficient vectors of one shape.
fn coefficient_distance(
    left: &[Array1<f64>],
    right: &[Array1<f64>],
) -> Result<f64, CustomFamilyError> {
    if left.len() != right.len()
        || left
            .iter()
            .zip(right.iter())
            .any(|(first, second)| first.len() != second.len())
    {
        return Err(CustomFamilyError::DimensionMismatch {
            reason: format!(
                "branch continuation: coefficient states of {} and {} blocks differ in shape",
                left.len(),
                right.len()
            ),
        });
    }
    Ok(left
        .iter()
        .zip(right.iter())
        .map(|(first, second)| {
            first
                .iter()
                .zip(second.iter())
                .map(|(a, b)| (a - b) * (a - b))
                .sum::<f64>()
        })
        .sum::<f64>()
        .sqrt())
}

/// Why a branch continuation published no mode.
#[derive(Debug)]
pub(crate) enum BranchContinuationRefusal {
    /// The branch ends between `last_certified_rho` and the next point toward `target_rho`: every
    /// sub-step that still moved `ρ(t)` in floating point failed the Newton-region contraction
    /// test or its corrector. That is a fold, where the branch's mode ceases to exist.
    FoldReached {
        last_certified_rho: Array1<f64>,
        target_rho: Array1<f64>,
        attempts: usize,
        last_failure: Option<String>,
    },
    /// The certified mode carries no exact-Newton curvature to form its IFT tangent from.
    TangentUnavailable { rho: Array1<f64>, reason: String },
    /// An evaluation the continuation needs failed.
    Evaluation(CustomFamilyError),
}

/// The rho a mode-selection or continuation line names, one rendering for both (gam#3173).
pub(crate) fn join_rho(rho: &Array1<f64>) -> String {
    rho.iter()
        .map(|value| format!("{value:.6e}"))
        .collect::<Vec<_>>()
        .join(",")
}

impl std::fmt::Display for BranchContinuationRefusal {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::FoldReached {
                last_certified_rho,
                target_rho,
                attempts,
                last_failure,
            } => write!(
                f,
                "the inner mode's branch ends past rho=[{}] toward rho=[{}]: every sub-step that \
                 still moves rho failed the Newton-region contraction test (a fold, gam#2973; \
                 {attempts} sub-step attempts; last: {})",
                join_rho(last_certified_rho),
                join_rho(target_rho),
                last_failure.as_deref().unwrap_or("none"),
            ),
            Self::TangentUnavailable { rho, reason } => write!(
                f,
                "branch continuation from rho=[{}] has no IFT tangent: {reason}",
                join_rho(rho)
            ),
            Self::Evaluation(error) => write!(f, "{error}"),
        }
    }
}

impl BranchContinuationRefusal {
    /// The refusal as the outer search takes it: a rejected trial point.
    pub(crate) fn into_trial_point(self) -> CustomFamilyError {
        match self {
            Self::Evaluation(error) => error.into_trial_point(),
            other => CustomFamilyError::trial_point(other.to_string()),
        }
    }
}

/// A continued mode, with the sub-steps that reached it. The mode is solved to the criterion's
/// own accuracy and not priced: the evaluation prices the mode it publishes ([`evaluate_on_branch`]).
pub(crate) struct BranchContinuation {
    /// The endpoint corrector's solve.
    pub(crate) inner: BlockwiseInnerResult,
    /// Sub-steps attempted, certified or not.
    pub(crate) attempts: usize,
    /// The measured contraction of every certified sub-step, in path order.
    pub(crate) contractions: Vec<f64>,
}

fn same_point(left: &Array1<f64>, right: &Array1<f64>) -> bool {
    left.len() == right.len()
        && left
            .iter()
            .zip(right.iter())
            .all(|(a, b)| a.to_bits() == b.to_bits())
}

/// The IFT predictor at the labeled `rho_trial` from the certified mode `certified`, formed from
/// the exact-Newton curvature the mode's own solve ended on ([`single_block_ift_predictor`]).
fn branch_predictor<F: CustomFamily + Clone + Send + Sync + 'static>(
    family: &F,
    specs: &[ParameterBlockSpec],
    options: &BlockwiseFitOptions,
    layout: &PenaltyLabelLayout,
    certified: &ConstrainedWarmStart,
    rho_trial: &Array1<f64>,
) -> Result<ConstrainedWarmStart, BranchContinuationRefusal> {
    let curvature = certified
        .cached_inner
        .as_ref()
        .and_then(|cached| cached.terminal_working_sets.as_deref())
        .and_then(|sets| sets.first())
        .filter(|set| matches!(set, BlockWorkingSet::ExactNewton { .. }))
        .ok_or_else(|| BranchContinuationRefusal::TangentUnavailable {
            rho: certified.rho.clone(),
            reason: "the certified mode carries no exact-Newton terminal curvature".to_string(),
        })?;
    let per_block = |rho: &Array1<f64>| {
        expand_labeled_log_lambdas(rho, layout)
            .and_then(|physical| split_log_lambdas(&physical, &layout.penalty_counts))
    };
    let from = per_block(&certified.rho).map_err(BranchContinuationRefusal::Evaluation)?;
    let to = per_block(rho_trial).map_err(BranchContinuationRefusal::Evaluation)?;
    let labeled_options = labeled_options_for_rho(options, specs, layout, &certified.rho)
        .map_err(BranchContinuationRefusal::Evaluation)?;
    let block_beta = single_block_ift_predictor(
        family,
        specs,
        &from,
        &to,
        labeled_options.as_ref(),
        &certified.block_beta,
        curvature,
        certified.active_sets.first().and_then(|set| set.as_deref()),
    )
    .map_err(BranchContinuationRefusal::Evaluation)?;
    Ok(ConstrainedWarmStart {
        rho: rho_trial.clone(),
        block_beta,
        active_sets: certified.active_sets.clone(),
        cached_inner: None,
    })
}

enum SubStep {
    Certified {
        /// The corrector's solve, which an endpoint's evaluation prices as it is.
        inner: BlockwiseInnerResult,
        /// The corrected mode, which an interior sub-step's successor predicts from.
        mode: ConstrainedWarmStart,
        contraction: f64,
    },
    Failed(String),
}

/// Correct one sub-step's predictor, and keep the result only when the predictor passes the
/// Newton-region contraction test and the corrected mode lies inside its root radius.
///
/// `corrector_options` are the criterion's own ([`criterion_inner_solve_options`]), so the
/// endpoint's evaluation prices the corrected mode without solving it again. The endpoint's
/// corrector also forms the determinant artifacts, since its mode is the one the evaluation
/// publishes as the next seed; an interior corrector's mode only seeds the next sub-step.
fn correct_sub_step<F: CustomFamily + Clone + Send + Sync + 'static>(
    family: &F,
    specs: &[ParameterBlockSpec],
    options: &BlockwiseFitOptions,
    corrector_options: &BlockwiseFitOptions,
    layout: &PenaltyLabelLayout,
    rho_trial: &Array1<f64>,
    predictor: &ConstrainedWarmStart,
    endpoint: bool,
) -> SubStep {
    let test = match newton_region_contraction(family, specs, options, layout, rho_trial, predictor)
    {
        Ok(test) => test,
        Err(error) => return SubStep::Failed(format!("the corrector refused: {error}")),
    };
    let (Some(contraction), Some(radius)) = (
        test.contraction.contraction_factor(),
        test.contraction.root_radius(),
    ) else {
        return SubStep::Failed(format!(
            "outside the Newton region at rho=[{}]: Newton correction {:.3e} (resolution {:.3e}), \
             simplified correction {:.3e} (resolution {:.3e}), contraction {}",
            join_rho(rho_trial),
            test.contraction.first_correction,
            test.contraction.first_resolution,
            test.contraction.second_correction,
            test.contraction.second_resolution,
            test.contraction
                .contraction_factor()
                .map_or_else(|| "unmeasured".to_string(), |theta| format!("{theta:.3e}")),
        ));
    };
    let corrector = if endpoint {
        correct_labeled_laplace_mode
    } else {
        correct_labeled_coefficient_mode
    };
    let (inner, mode) = match corrector(
        family,
        specs,
        corrector_options,
        layout,
        rho_trial,
        Some(&test.state),
    ) {
        Ok(corrected) => corrected,
        Err(error) => return SubStep::Failed(format!("the corrector refused: {error}")),
    };
    match coefficient_distance(&predictor.block_beta, &mode.block_beta) {
        Ok(distance) if distance <= radius => SubStep::Certified {
            inner,
            mode,
            contraction,
        },
        Ok(distance) => SubStep::Failed(format!(
            "the corrected mode lies {distance:.3e} from the predictor, outside the root radius \
             {radius:.3e} at rho=[{}]",
            join_rho(rho_trial)
        )),
        Err(error) => SubStep::Failed(error.to_string()),
    }
}

/// Continue the certified mode `start` along its branch to the labeled `rho_target`, to the
/// criterion's own accuracy (gam#2973, gam#2366).
pub(crate) fn continue_branch<F: CustomFamily + Clone + Send + Sync + 'static>(
    family: &F,
    specs: &[ParameterBlockSpec],
    options: &BlockwiseFitOptions,
    layout: &PenaltyLabelLayout,
    start: &ConstrainedWarmStart,
    rho_target: &Array1<f64>,
) -> Result<BranchContinuation, BranchContinuationRefusal> {
    if start.rho.len() != rho_target.len() {
        return Err(BranchContinuationRefusal::TangentUnavailable {
            rho: start.rho.clone(),
            reason: format!(
                "the accepted mode has {} smoothing coordinates and the target {}",
                start.rho.len(),
                rho_target.len()
            ),
        });
    }
    let criterion_options = criterion_inner_solve_options(family, options, 0);
    let corrector_options = criterion_options.as_ref().unwrap_or(options);
    if same_point(&start.rho, rho_target) {
        let (inner, _) = correct_labeled_laplace_mode(
            family,
            specs,
            corrector_options,
            layout,
            rho_target,
            Some(start),
        )
        .map_err(BranchContinuationRefusal::Evaluation)?;
        return Ok(BranchContinuation {
            inner,
            attempts: 0,
            contractions: Vec::new(),
        });
    }
    let rho_start = start.rho.clone();
    let mut certified = start.clone();
    let mut t_certified = 0.0_f64;
    let mut sub_step = 1.0_f64;
    let mut attempts = 0_usize;
    let mut contractions = Vec::new();
    let mut last_failure: Option<String> = None;
    loop {
        let t_trial = (t_certified + sub_step).min(1.0);
        let endpoint = t_trial >= 1.0;
        let rho_trial = if endpoint {
            rho_target.clone()
        } else {
            &rho_start + &((rho_target - &rho_start) * t_trial)
        };
        if t_trial <= t_certified || same_point(&rho_trial, &certified.rho) {
            return Err(BranchContinuationRefusal::FoldReached {
                last_certified_rho: certified.rho.clone(),
                target_rho: rho_target.clone(),
                attempts,
                last_failure,
            });
        }
        attempts += 1;
        let predictor = branch_predictor(family, specs, options, layout, &certified, &rho_trial)?;
        match correct_sub_step(
            family,
            specs,
            options,
            corrector_options,
            layout,
            &rho_trial,
            &predictor,
            endpoint,
        ) {
            SubStep::Certified {
                inner,
                mode,
                contraction,
            } => {
                contractions.push(contraction);
                if endpoint {
                    return Ok(BranchContinuation {
                        inner,
                        attempts,
                        contractions,
                    });
                }
                // An interior mode seeds the next predictor from the curvature its own solve
                // ended on; nothing prices it.
                t_certified = t_trial;
                certified = mode;
            }
            SubStep::Failed(reason) => {
                last_failure = Some(reason);
                // Bisection: the failed sub-step is halved.
                sub_step *= 0.5;
            }
        }
    }
}

/// Whether the continuation covers a family's inner solve: its inner objective may have more than
/// one mode (the predicate the #2366 anchor uses: a β-dependent Hessian, not declared globally
/// convex), and the Newton-region test covers its solve (a single block with exact-Newton
/// curvature, no joint penalty, no Jeffreys term, no Hessian-vector workspace). A joint-Newton
/// family keeps its per-evaluation rule until that path's frozen-curvature re-solve lands
/// (gam#2973's second slice).
fn continuation_covers<F: CustomFamily + Clone + Send + Sync + 'static>(
    family: &F,
    specs: &[ParameterBlockSpec],
    options: &BlockwiseFitOptions,
    layout: &PenaltyLabelLayout,
) -> bool {
    inner_objective_may_have_several_modes(family)
        && single_block_newton_region_probe_applies(family, specs, options)
        && layout.joint_specs.is_empty()
}

/// Whether `seed` is a certified exact-Newton mode a continuation can start from.
fn certified_exact_newton_mode(seed: &ConstrainedWarmStart) -> bool {
    seed.cached_inner.as_ref().is_some_and(|cached| {
        cached.converged
            && cached
                .terminal_working_sets
                .as_deref()
                .is_some_and(|sets| matches!(sets, [BlockWorkingSet::ExactNewton { .. }]))
    })
}

/// Whether an evaluation at `rho` from `seed` continues the seed's branch: the continuation
/// covers the family's solve and the seed is a certified mode at another θ.
fn continues_its_branch<F: CustomFamily + Clone + Send + Sync + 'static>(
    family: &F,
    specs: &[ParameterBlockSpec],
    options: &BlockwiseFitOptions,
    layout: &PenaltyLabelLayout,
    seed: &ConstrainedWarmStart,
    rho: &Array1<f64>,
) -> bool {
    continuation_covers(family, specs, options, layout)
        && certified_exact_newton_mode(seed)
        && seed.rho.len() == rho.len()
        && !same_point(&seed.rho, rho)
}

/// The starts one outer evaluation solves its candidate modes from (gam#3173). `None` is the
/// caller's own starting coefficients.
#[derive(Clone, Copy)]
pub(crate) struct ModeStarts<'a> {
    /// The accepted incumbent's mode.
    pub(crate) incumbent: Option<&'a ConstrainedWarmStart>,
    /// The fit's fixed starts, the same at every evaluation of one fit.
    pub(crate) fixed: &'a [Option<ConstrainedWarmStart>],
}

/// A solved mode certified at the labeled `rho` ([`certify_inner_mode`]). `labeled_options` carry
/// the evaluation's joint penalties.
fn certified_mode<F: CustomFamily + Clone + Send + Sync + 'static>(
    family: &F,
    specs: &[ParameterBlockSpec],
    labeled_options: &BlockwiseFitOptions,
    layout: &PenaltyLabelLayout,
    mut inner: BlockwiseInnerResult,
) -> Result<BlockwiseInnerResult, CustomFamilyError> {
    let rho_dim = layout.penalty_counts.iter().sum::<usize>();
    certify_inner_mode(family, specs, labeled_options, &mut inner, rho_dim, 0)?;
    Ok(inner)
}

/// One outer evaluation at the labeled `rho` under the published-mode rule (gam#3173, gam#2973):
/// a certified mode from each start, and the one the rule selects priced in `eval_mode`
/// ([`select_lowest_penalized`]).
///
/// The incumbent is continued along its branch where the continuation covers the solve
/// ([`continue_branch`]). A branch that ends at its fold before `rho` drops out of the set, and
/// the lowest rival is published: the branch hands over at its fold. Every other start, and an
/// incumbent the continuation does not cover, is solved directly at `rho`, as the seed rules chose
/// it, and a start whose solve is refused at this trial point drops out too. A fixed start whose
/// seed is the incumbent's is the same computation and is not repeated, and one start is its own
/// selection. The evaluation refuses only when no start certified a mode, and then with the
/// incumbent's refusal. Any other error is a failure of the evaluation, not of a start, and is
/// returned as it is.
pub(crate) fn evaluate_on_branch<F: CustomFamily + Clone + Send + Sync + 'static>(
    family: &F,
    specs: &[ParameterBlockSpec],
    options: &BlockwiseFitOptions,
    layout: &PenaltyLabelLayout,
    rho: &Array1<f64>,
    starts: ModeStarts<'_>,
    rho_prior: &gam_problem::RhoPrior,
    eval_mode: EvalMode,
) -> Result<OuterObjectiveEvalResult, CustomFamilyError> {
    let criterion_options = criterion_inner_solve_options(family, options, 0);
    let solve_options = criterion_options.as_ref().unwrap_or(options);
    let labeled_options = labeled_options_for_rho(options, specs, layout, rho)?;
    let direct = |seed: Option<&ConstrainedWarmStart>| {
        correct_labeled_laplace_mode(family, specs, solve_options, layout, rho, seed)
            .and_then(|(inner, _)| {
                certified_mode(family, specs, labeled_options.as_ref(), layout, inner)
            })
    };
    // A start that certifies no mode at this trial point drops out of the set; any other error
    // is the evaluation's.
    let dropped_or_failed = |refusal: CustomFamilyError| {
        if refusal.is_trial_point_infeasible() {
            Ok(refusal)
        } else {
            Err(refusal)
        }
    };
    let incumbent_seed = SeedIdentity::of(starts.incumbent);
    let rivals: Vec<(usize, Option<&ConstrainedWarmStart>)> = starts
        .fixed
        .iter()
        .enumerate()
        .filter(|(_, seed)| SeedIdentity::of(seed.as_ref()) != incumbent_seed)
        .map(|(index, seed)| (index, seed.as_ref()))
        .collect();
    let incumbent = match starts
        .incumbent
        .filter(|seed| continues_its_branch(family, specs, options, layout, seed, rho))
    {
        Some(start) => match continue_branch(family, specs, options, layout, start, rho) {
            Ok(continuation) => {
                log::debug!(
                    "[branch continuation] rho=[{}] from rho=[{}]: {} sub-step attempt(s), \
                     certified contractions [{}]",
                    join_rho(rho),
                    join_rho(&start.rho),
                    continuation.attempts,
                    continuation
                        .contractions
                        .iter()
                        .map(|theta| format!("{theta:.3e}"))
                        .collect::<Vec<_>>()
                        .join(","),
                );
                certified_mode(
                    family,
                    specs,
                    labeled_options.as_ref(),
                    layout,
                    continuation.inner,
                )
                .map_err(dropped_or_failed)
            }
            Err(refusal @ BranchContinuationRefusal::FoldReached { .. }) => {
                log::debug!("[branch continuation] ended: {refusal}");
                Err(Ok(refusal.into_trial_point()))
            }
            Err(refusal) => return Err(refusal.into_trial_point()),
        },
        None => direct(starts.incumbent).map_err(dropped_or_failed),
    };
    // How far the incumbent's own mode sat above the published one, where the published one is
    // another start's (gam#3173). It is the evidence the stratum rule needs to tell a trial that
    // merely wandered to another kept rank from one that is on the branch this run is not.
    let mut incumbent_mode_excess: Option<f64> = None;
    let published = if rivals.is_empty() {
        match incumbent {
            Ok(inner) => inner,
            Err(Ok(refusal) | Err(refusal)) => return Err(refusal),
        }
    } else {
        // The penalty roots at this θ, the same the solves evaluated `½βᵀS_λβ` on (#2954).
        let physical_rho = expand_labeled_log_lambdas(rho, layout)?;
        let per_block = split_log_lambdas(&physical_rho, &layout.penalty_counts)?;
        let roots = BlockPenaltyRoots::new(
            specs,
            &per_block,
            labeled_options.joint_penalties.as_deref(),
        )?;
        let candidate = |start: ModeStart, inner: BlockwiseInnerResult| {
            penalized_objective_at_mode(family, specs, &roots, &inner).map(|penalized_objective| {
                ModeCandidate {
                    start,
                    inner,
                    penalized_objective,
                }
            })
        };
        let mut candidates = Vec::new();
        let mut incumbent_penalized: Option<PenalizedObjective> = None;
        let incumbent_refusal = match incumbent {
            Ok(inner) => {
                let incumbent_candidate = candidate(ModeStart::Incumbent, inner)?;
                incumbent_penalized = Some(incumbent_candidate.penalized_objective);
                candidates.push(incumbent_candidate);
                None
            }
            Err(Ok(refusal)) => Some(refusal),
            Err(Err(error)) => return Err(error),
        };
        for &(index, seed) in &rivals {
            match direct(seed) {
                Ok(inner) => candidates.push(candidate(ModeStart::FixedSeed(index), inner)?),
                Err(refusal) if refusal.is_trial_point_infeasible() => log::debug!(
                    "[mode selection #3173] rho=[{}]: fixed seed {index} certified no mode: \
                     {refusal}",
                    join_rho(rho)
                ),
                Err(error) => return Err(error),
            }
        }
        let Some(selection) = select_lowest_penalized(candidates) else {
            return Err(incumbent_refusal.unwrap_or_else(|| {
                CustomFamilyError::trial_point(format!(
                    "no start certified an inner mode at rho=[{}]",
                    join_rho(rho)
                ))
            }));
        };
        // The incumbent's own mode lost, and by more than comparing the two can round by: at
        // this θ the branch the outer walk carries is not the posterior branch (gam#3173).
        incumbent_mode_excess = incumbent_penalized.and_then(|incumbent| {
            (selection.winner.start != ModeStart::Incumbent
                && selection
                    .winner
                    .penalized_objective
                    .resolvably_below(&incumbent))
            .then(|| incumbent.value - selection.winner.penalized_objective.value)
        });
        log::debug!(
            "[mode selection #3173] rho=[{}]: {} of {} start(s) certified a mode; published the \
             {} mode, f={:.9e}; runner-up gap {}; incumbent {}",
            join_rho(rho),
            selection.certified,
            rivals.len() + 1,
            selection.winner.start,
            selection.winner.penalized_objective.value,
            selection
                .runner_up_gap
                .map_or_else(|| "none".to_string(), |gap| format!("{gap:.3e}")),
            incumbent_refusal.as_ref().map_or_else(
                || match incumbent_mode_excess {
                    Some(excess) => format!("certified, above the published mode by {excess:.3e}"),
                    None => "certified".to_string(),
                },
                |refusal| format!("dropped: {refusal}")
            ),
        );
        selection.winner.inner
    };
    let mut result = outerobjective_from_coefficient_mode_labeled(
        family, specs, options, layout, rho, rho_prior, published, eval_mode,
    )?;
    result.incumbent_mode_excess = incumbent_mode_excess;
    Ok(result)
}
