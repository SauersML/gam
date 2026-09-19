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
//! `ρ_A` it follows `ρ(t) = ρ_A + t (ρ − ρ_A)`, `t: 0 → 1`. Each sub-step predicts the mode from
//! the IFT tangent `dβ̂/dρ_k = −v_k` of the last certified mode and corrects it with the inner
//! solver. A sub-step is kept only when [`newton_region_contraction`] shows the predictor inside
//! the Newton region of the root the corrector returns; a failed sub-step is halved. Halving has
//! no count: it ends on a certified sub-step or where a sub-step no longer moves `ρ(t)` in
//! floating point. There the branch ends, at a fold, and the evaluation is refused as
//! [`BranchContinuationRefusal::FoldReached`], which the outer search takes as a rejected trial.
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

/// The Newton correction and the simplified correction at a predictor.
#[derive(Clone, Copy, Debug, PartialEq)]
pub(crate) struct NewtonRegionContraction {
    /// `‖Δ⁰‖`, the Newton correction at the predictor.
    pub(crate) first_correction: f64,
    /// `‖Δ̄¹‖`, the simplified correction, with the curvature frozen at the predictor.
    pub(crate) second_correction: f64,
}

impl NewtonRegionContraction {
    /// The measured contraction `Θ = ‖Δ̄¹‖ / ‖Δ⁰‖`. A predictor that is already stationary
    /// (`Δ⁰ = Δ̄¹ = 0`) has `Θ = 0`; a non-finite correction measures nothing.
    pub(crate) fn contraction_factor(&self) -> Option<f64> {
        if !(self.first_correction.is_finite() && self.second_correction.is_finite()) {
            return None;
        }
        if self.first_correction == 0.0 {
            return (self.second_correction == 0.0).then_some(0.0);
        }
        Some(self.second_correction / self.first_correction)
    }

    /// Whether the predictor passes the Newton-region contraction test, `Θ ≤ ¼`.
    pub(crate) fn in_newton_region(&self) -> bool {
        self.contraction_factor()
            .is_some_and(|theta| theta <= NEWTON_REGION_CONTRACTION_BOUND)
    }

    /// The Kantorovich radius around the predictor inside which the root the test names lies:
    /// `‖Δ⁰‖ (1 − √(1 − 2[h₀])) / [h₀]` with `[h₀] = 2Θ`, which is `‖Δ⁰‖` at `Θ = 0` and `2‖Δ⁰‖`
    /// at `Θ = ¼`. `None` outside the Newton region.
    pub(crate) fn root_radius(&self) -> Option<f64> {
        if !self.in_newton_region() {
            return None;
        }
        let theta = self.contraction_factor()?;
        let h = 2.0 * theta;
        if h == 0.0 {
            Some(self.first_correction)
        } else {
            Some(self.first_correction * (1.0 - (1.0 - 2.0 * h).sqrt()) / h)
        }
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
        /// Sub-steps whose corrected mode passed the test but whose tangent evaluation refused
        /// the point, so they were halved and never published.
        tangent_refusals: usize,
        last_failure: Option<String>,
    },
    /// The accepted mode's derivative-bearing evaluation formed no IFT tangent of the shape the
    /// continuation needs.
    TangentUnavailable { rho: Array1<f64>, reason: String },
    /// An evaluation the continuation needs failed.
    Evaluation(CustomFamilyError),
}

fn join_rho(rho: &Array1<f64>) -> String {
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
                tangent_refusals,
                last_failure,
            } => write!(
                f,
                "the inner mode's branch ends past rho=[{}] toward rho=[{}]: every sub-step that \
                 still moves rho failed the Newton-region contraction test (a fold, gam#2973; \
                 {attempts} sub-step attempts, {tangent_refusals} halved for a refused tangent; \
                 last: {})",
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

/// A continued evaluation, with the sub-steps that reached it.
pub(crate) struct BranchContinuation {
    pub(crate) eval: OuterObjectiveEvalResult,
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

fn carried_tangent(mode: &ConstrainedWarmStart) -> Option<Arc<Array2<f64>>> {
    mode.cached_inner
        .as_ref()
        .and_then(|cached| cached.rho_mode_responses.clone())
}

/// The certified mode with its IFT tangent. A mode filed by a derivative-bearing evaluation
/// carries the tangent that evaluation formed. Any other certified mode gets one from a
/// derivative-bearing evaluation at its own ρ.
fn tangent_at<F: CustomFamily + Clone + Send + Sync + 'static>(
    family: &F,
    specs: &[ParameterBlockSpec],
    options: &BlockwiseFitOptions,
    layout: &PenaltyLabelLayout,
    rho_prior: &gam_problem::RhoPrior,
    mode: &ConstrainedWarmStart,
) -> Result<(ConstrainedWarmStart, Arc<Array2<f64>>), BranchContinuationRefusal> {
    if let Some(tangent) = carried_tangent(mode) {
        return Ok((mode.clone(), tangent));
    }
    let eval = outerobjectivegradienthessian_labeled(
        family,
        specs,
        options,
        layout,
        &mode.rho,
        Some(mode),
        rho_prior,
        EvalMode::ValueAndGradient,
    )
    .map_err(BranchContinuationRefusal::Evaluation)?;
    if !eval.inner_converged {
        return Err(BranchContinuationRefusal::Evaluation(
            CustomFamilyError::trial_point(format!(
                "branch continuation: the certified mode at rho=[{}] did not re-converge for its \
                 tangent",
                join_rho(&mode.rho)
            )),
        ));
    }
    let tangent = carried_tangent(&eval.warm_start).ok_or_else(|| {
        BranchContinuationRefusal::TangentUnavailable {
            rho: mode.rho.clone(),
            reason: "its derivative-bearing evaluation formed no rho mode responses".to_string(),
        }
    })?;
    Ok((eval.warm_start, tangent))
}

/// The penalty coordinates the evaluator's mode responses are indexed by: the per-block log-λ's
/// the labeled `rho` expands to, then the joint penalties' (gam#1587).
fn penalty_coordinates(
    rho: &Array1<f64>,
    layout: &PenaltyLabelLayout,
) -> Result<Array1<f64>, CustomFamilyError> {
    let physical = expand_labeled_log_lambdas(rho, layout)?;
    Ok(physical
        .iter()
        .copied()
        .chain(layout.joint_log_lambdas(rho))
        .collect())
}

/// The IFT predictor at `rho_trial`: `β̂ − Σ_k Δρ_k v_k`, since `dβ̂/dρ_k = −v_k`.
fn branch_predictor(
    certified: &ConstrainedWarmStart,
    tangent: &Array2<f64>,
    rho_trial: &Array1<f64>,
    layout: &PenaltyLabelLayout,
) -> Result<ConstrainedWarmStart, BranchContinuationRefusal> {
    let from = penalty_coordinates(&certified.rho, layout)
        .map_err(BranchContinuationRefusal::Evaluation)?;
    let to =
        penalty_coordinates(rho_trial, layout).map_err(BranchContinuationRefusal::Evaluation)?;
    let total: usize = certified.block_beta.iter().map(|block| block.len()).sum();
    if tangent.nrows() != total || tangent.ncols() != from.len() {
        return Err(BranchContinuationRefusal::TangentUnavailable {
            rho: certified.rho.clone(),
            reason: format!(
                "its mode responses are {}x{}, and the continuation needs {total}x{}",
                tangent.nrows(),
                tangent.ncols(),
                from.len()
            ),
        });
    }
    let mut beta = Array1::from_iter(
        certified
            .block_beta
            .iter()
            .flat_map(|block| block.iter().copied()),
    );
    for (coordinate, (start, end)) in from.iter().zip(to.iter()).enumerate() {
        let delta = end - start;
        if delta != 0.0 {
            beta.scaled_add(-delta, &tangent.column(coordinate));
        }
    }
    let mut offset = 0;
    let block_beta = certified
        .block_beta
        .iter()
        .map(|block| {
            let piece = beta
                .slice(ndarray::s![offset..offset + block.len()])
                .to_owned();
            offset += block.len();
            piece
        })
        .collect();
    Ok(ConstrainedWarmStart {
        rho: rho_trial.clone(),
        block_beta,
        active_sets: certified.active_sets.clone(),
        cached_inner: None,
    })
}

enum SubStep {
    Certified {
        mode: ConstrainedWarmStart,
        contraction: f64,
    },
    Failed(String),
}

/// Correct one sub-step's predictor, and keep the result only when the predictor passes the
/// Newton-region contraction test and the corrected mode lies inside its root radius.
fn correct_sub_step<F: CustomFamily + Clone + Send + Sync + 'static>(
    family: &F,
    specs: &[ParameterBlockSpec],
    options: &BlockwiseFitOptions,
    layout: &PenaltyLabelLayout,
    rho_trial: &Array1<f64>,
    predictor: &ConstrainedWarmStart,
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
            "outside the Newton region at rho=[{}]: Newton correction {:.3e}, simplified correction \
             {:.3e}, contraction {}",
            join_rho(rho_trial),
            test.contraction.first_correction,
            test.contraction.second_correction,
            test.contraction
                .contraction_factor()
                .map_or_else(|| "unmeasured".to_string(), |theta| format!("{theta:.3e}")),
        ));
    };
    let mode = match correct_labeled_coefficient_mode(
        family,
        specs,
        options,
        layout,
        rho_trial,
        Some(&test.state),
    ) {
        Ok((_, mode)) => mode,
        Err(error) => return SubStep::Failed(format!("the corrector refused: {error}")),
    };
    match coefficient_distance(&predictor.block_beta, &mode.block_beta) {
        Ok(distance) if distance <= radius => SubStep::Certified { mode, contraction },
        Ok(distance) => SubStep::Failed(format!(
            "the corrected mode lies {distance:.3e} from the predictor, outside the root radius \
             {radius:.3e} at rho=[{}]",
            join_rho(rho_trial)
        )),
        Err(error) => SubStep::Failed(error.to_string()),
    }
}

/// Continue the certified mode `start` along its branch to the labeled `rho_target`, then
/// evaluate there in `eval_mode` from the continued mode (gam#2973, gam#2366).
pub(crate) fn continue_branch<F: CustomFamily + Clone + Send + Sync + 'static>(
    family: &F,
    specs: &[ParameterBlockSpec],
    options: &BlockwiseFitOptions,
    layout: &PenaltyLabelLayout,
    rho_prior: &gam_problem::RhoPrior,
    start: &ConstrainedWarmStart,
    rho_target: &Array1<f64>,
    eval_mode: EvalMode,
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
    if same_point(&start.rho, rho_target) {
        let eval = outerobjectivegradienthessian_labeled(
            family,
            specs,
            options,
            layout,
            rho_target,
            Some(start),
            rho_prior,
            eval_mode,
        )
        .map_err(BranchContinuationRefusal::Evaluation)?;
        return Ok(BranchContinuation {
            eval,
            attempts: 0,
            contractions: Vec::new(),
        });
    }
    let rho_start = start.rho.clone();
    let (mut certified, mut tangent) =
        tangent_at(family, specs, options, layout, rho_prior, start)?;
    let mut t_certified = 0.0_f64;
    let mut sub_step = 1.0_f64;
    let mut attempts = 0_usize;
    let mut tangent_refusals = 0_usize;
    let mut contractions = Vec::new();
    let mut last_failure: Option<String> = None;
    loop {
        let t_trial = (t_certified + sub_step).min(1.0);
        let rho_trial = if t_trial >= 1.0 {
            rho_target.clone()
        } else {
            &rho_start + &((rho_target - &rho_start) * t_trial)
        };
        if t_trial <= t_certified || same_point(&rho_trial, &certified.rho) {
            return Err(BranchContinuationRefusal::FoldReached {
                last_certified_rho: certified.rho.clone(),
                target_rho: rho_target.clone(),
                attempts,
                tangent_refusals,
                last_failure,
            });
        }
        attempts += 1;
        let predictor = branch_predictor(&certified, &tangent, &rho_trial, layout)?;
        match correct_sub_step(family, specs, options, layout, &rho_trial, &predictor) {
            SubStep::Certified { mode, contraction } => {
                if t_trial >= 1.0 {
                    contractions.push(contraction);
                    certified = mode;
                    break;
                }
                // The next sub-step predicts from this mode's IFT tangent. A tangent evaluation
                // that refuses the point (the envelope-gradient tripwire near a fold, where the
                // mode response is unresolved) leaves no prediction to continue from, so this
                // sub-step fails like any other and is halved from the last certified mode; it is
                // never published. A tangent of the wrong shape is a contract failure, not a
                // step-size question, and is returned.
                match tangent_at(family, specs, options, layout, rho_prior, &mode) {
                    Ok((next, next_tangent)) => {
                        contractions.push(contraction);
                        t_certified = t_trial;
                        certified = next;
                        tangent = next_tangent;
                    }
                    Err(BranchContinuationRefusal::Evaluation(error))
                        if error.is_trial_point_infeasible() =>
                    {
                        tangent_refusals += 1;
                        last_failure = Some(format!(
                            "the certified mode at rho=[{}] refused its tangent evaluation: {error}",
                            join_rho(&rho_trial)
                        ));
                        sub_step *= 0.5;
                    }
                    Err(refusal) => return Err(refusal),
                }
            }
            SubStep::Failed(reason) => {
                last_failure = Some(reason);
                // Bisection: the failed sub-step is halved.
                sub_step *= 0.5;
            }
        }
    }
    let eval = outerobjectivegradienthessian_labeled(
        family,
        specs,
        options,
        layout,
        rho_target,
        Some(&certified),
        rho_prior,
        eval_mode,
    )
    .map_err(BranchContinuationRefusal::Evaluation)?;
    Ok(BranchContinuation {
        eval,
        attempts,
        contractions,
    })
}

/// Whether an evaluation at `rho` from `seed` continues the seed's branch.
///
/// The family's inner objective may have more than one mode (the predicate the #2366 anchor
/// uses: a β-dependent Hessian, not declared globally convex); its solve is one the
/// Newton-region test covers (a single block with exact-Newton curvature, no joint penalty, no
/// Jeffreys term, no Hessian-vector workspace); and the seed is a certified mode at another θ.
/// A joint-Newton family keeps its per-evaluation rule until that path's frozen-curvature
/// re-solve lands (gam#2973's second slice).
fn continues_its_branch<F: CustomFamily + Clone + Send + Sync + 'static>(
    family: &F,
    specs: &[ParameterBlockSpec],
    options: &BlockwiseFitOptions,
    layout: &PenaltyLabelLayout,
    seed: &ConstrainedWarmStart,
    rho: &Array1<f64>,
) -> bool {
    let exact_newton_block = seed
        .cached_inner
        .as_ref()
        .and_then(|cached| cached.terminal_working_sets.as_deref())
        .is_some_and(|sets| matches!(sets, [BlockWorkingSet::ExactNewton { .. }]));
    family.exact_newton_joint_hessian_beta_dependent()
        && !family.inner_coefficient_objective_is_globally_convex()
        && single_block_newton_region_probe_applies(family, specs, options)
        && layout.joint_specs.is_empty()
        && exact_newton_block
        && seed
            .cached_inner
            .as_ref()
            .is_some_and(|cached| cached.converged)
        && seed.rho.len() == rho.len()
        && !same_point(&seed.rho, rho)
}

/// One outer evaluation at the labeled `rho` from `seed` under gam#2366's selection rule
/// (gam#2973). A seed that is a certified mode at another θ is continued along its branch
/// ([`continue_branch`]). Any other seed is evaluated as the seed rules chose it: no seed, the
/// same θ (a same-ρ reuse), a seed that is no certified mode, a family whose inner objective
/// declares one mode, or a solve the Newton-region test does not cover yet (the joint Newton
/// path). A branch that ends before `rho` refuses the trial point.
pub(crate) fn evaluate_on_branch<F: CustomFamily + Clone + Send + Sync + 'static>(
    family: &F,
    specs: &[ParameterBlockSpec],
    options: &BlockwiseFitOptions,
    layout: &PenaltyLabelLayout,
    rho: &Array1<f64>,
    seed: Option<&ConstrainedWarmStart>,
    rho_prior: &gam_problem::RhoPrior,
    eval_mode: EvalMode,
) -> Result<OuterObjectiveEvalResult, CustomFamilyError> {
    let Some(start) =
        seed.filter(|seed| continues_its_branch(family, specs, options, layout, seed, rho))
    else {
        return outerobjectivegradienthessian_labeled(
            family, specs, options, layout, rho, seed, rho_prior, eval_mode,
        );
    };
    match continue_branch(
        family, specs, options, layout, rho_prior, start, rho, eval_mode,
    ) {
        Ok(continuation) => {
            log::debug!(
                "[branch continuation] rho=[{}] from rho=[{}]: {} sub-step attempt(s), certified \
                 contractions [{}]",
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
            Ok(continuation.eval)
        }
        Err(refusal) => {
            log::debug!("[branch continuation] refused: {refusal}");
            Err(refusal.into_trial_point())
        }
    }
}
