//! The published inner mode at an outer point (gam#3173, gam#2973).
//!
//! `V(θ) = ℓ_p(β̂(θ), θ)` is a function of θ only once a rule names `β̂(θ)`, and the inner
//! objective `f(β; θ) = −ℓ(β) + ½βᵀS_λβ − Φ(β)` of a family with a β-dependent Hessian can have
//! several certified local minima at one θ. The rule: `β̂(θ)` is the certified local minimum with
//! the lowest `f` among the modes reached from the evaluation's starts. The criterion is `f`, not
//! `V`: the Laplace approximation behind `V` is taken at the posterior mode, and the posterior mode
//! is the minimizer of `f`.
//!
//! The starts are the accepted incumbent's mode, continued along its branch where the
//! continuation covers the solve ([`evaluate_on_branch`]), and the fit's fixed starts: the #2661
//! anchored endpoint when that continuation certified, and the caller's seed. The fixed starts'
//! modes at θ are a function of θ. The incumbent's is not: it depends on which iterates the walk
//! accepted before θ. So `V(θ)` is single-valued over the modes the fixed starts reach, and two
//! walks that carry different incumbents to one θ publish the same value there unless an incumbent
//! reaches a mode lower than every fixed start's, which then wins in that walk only.
//!
//! The exact-joint drivers that publish through `evaluate_custom_family_joint_hyper_best_mode_shared`
//! -- the survival marginal-slope, Bernoulli marginal-slope and transformation-normal routes --
//! carry their incumbent in a coefficient-mode branch and solve no continuation, so their starts
//! are completed where the rule is applied (`joint_mode_starts`, beside that evaluator). Their
//! fit's fixed start is the blocks' own seed, which the driver rebuilds at every theta from the
//! fit's coefficient hints. The latent-survival route takes the branch's one start and calls
//! `evaluate_custom_family_joint_hyper_owned`, which solves it alone: routing that driver through
//! the same evaluator is the remaining site.
//!
//! Measured on `default_worker_stack_2967`: at ρ = [0, 0] the #2661 anchored endpoint's branch
//! certifies `f = 2356.81` and the caller's seed's branch `f = 2330.28`. An outer search that
//! carried only the first stayed on it into its fold near ρ = [4.30, 2.24] and ended uncertified
//! after 130 iterations; one that entered on the second certified in 9.
//!
//! A branch that ends at a fold drops out of the set and the lowest rival is published, which is
//! the continuation's handover. An evaluation refuses only when no start certified a mode. Where
//! two branches' `f` cross, the published criterion changes branch and `V` has a kink there; the
//! outer search is handed the selected mode's value, gradient and certificate.
use super::*;

/// A certified mode's penalized objective `f = −ℓ + ½βᵀS_λβ − Φ`, with what its evaluation
/// accumulated, so two modes' values are compared to the rounding the comparison can carry.
#[derive(Clone, Copy, Debug)]
pub(crate) struct PenalizedObjective {
    pub(crate) value: f64,
    pub(crate) likelihood_rows: usize,
    /// The depth of the penalty's root-form accumulation ([`PenaltyValue::depth`]).
    pub(crate) penalty_entries: usize,
    /// What the penalty's root-form evaluation summed ([`PenaltyValue::magnitude`], #2954).
    pub(crate) penalty_accumulation: f64,
    /// The Jeffreys log-determinant's certified rounding; zero without the term.
    pub(crate) jeffreys_roundoff: f64,
}

impl PenalizedObjective {
    /// Whether this value is below `incumbent` by more than comparing the two can round by
    /// ([`ObjectiveAccumulation::between_endpoints`], the bound the inner trust loop forms for the
    /// same comparison). Two starts that certify one mode differ in `f` only by rounding, and a
    /// comparison inside that band would pick between them on noise.
    pub(crate) fn resolvably_below(&self, incumbent: &Self) -> bool {
        let ceiling = ObjectiveAccumulation::between_endpoints(
            self.likelihood_rows.max(incumbent.likelihood_rows),
            self.penalty_entries.max(incumbent.penalty_entries),
            [self.value, incumbent.value],
            [self.penalty_accumulation, incumbent.penalty_accumulation],
            [self.jeffreys_roundoff, incumbent.jeffreys_roundoff],
        )
        .roundoff_ceiling();
        self.value < incumbent.value - ceiling
    }
}

/// The penalized objective the inner solve minimizes, at `inner`'s mode: `f = −ℓ + ½βᵀS_λβ − Φ`,
/// with the Jeffreys term `Φ` (its strength included) where the family requires it. `roots` are
/// the evaluation's penalty roots at its θ (#2954), whose value is the solve's `½βᵀS_λβ`.
pub(crate) fn penalized_objective_at_mode<F: CustomFamily + Clone + Send + Sync + 'static>(
    family: &F,
    specs: &[ParameterBlockSpec],
    roots: &BlockPenaltyRoots,
    inner: &BlockwiseInnerResult,
) -> Result<PenalizedObjective, CustomFamilyError> {
    let jeffreys = if family.joint_jeffreys_term_required() {
        let ranges = block_param_ranges(specs);
        match build_joint_jeffreys_subspace(family, specs, &ranges)? {
            Some(z_joint) => custom_family_joint_jeffreys_value(
                family,
                &inner.block_states,
                specs,
                &ranges,
                &z_joint,
            )?,
            None => JointJeffreysValue::default(),
        }
    } else {
        JointJeffreysValue::default()
    };
    let value = checked_penalizedobjective(
        inner.log_likelihood,
        inner.penalty_value,
        -jeffreys.phi,
        "inner mode selection (gam#3173)",
    )?;
    let penalty = roots.value_of_states(&inner.block_states);
    Ok(PenalizedObjective {
        value,
        likelihood_rows: joint_observation_count(&inner.block_states),
        penalty_entries: penalty.depth,
        penalty_accumulation: penalty.magnitude,
        jeffreys_roundoff: jeffreys.roundoff,
    })
}

/// Where a candidate mode was solved from.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum ModeStart {
    /// The accepted incumbent's mode, continued along its branch where the continuation covers
    /// the solve.
    Incumbent,
    /// The fit's fixed start at this index.
    FixedSeed(usize),
}

impl std::fmt::Display for ModeStart {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Incumbent => write!(f, "incumbent"),
            Self::FixedSeed(index) => write!(f, "fixed seed {index}"),
        }
    }
}

/// A certified mode at the evaluation's θ, with its penalized objective.
pub(crate) struct ModeCandidate {
    pub(crate) start: ModeStart,
    pub(crate) inner: BlockwiseInnerResult,
    pub(crate) penalized_objective: PenalizedObjective,
}

/// The published mode and what the selection saw.
pub(crate) struct ModeSelection {
    pub(crate) winner: ModeCandidate,
    /// How many starts certified a mode.
    pub(crate) certified: usize,
    /// The lowest other certified `f` minus the winner's; `None` with one certified mode.
    pub(crate) runner_up_gap: Option<f64>,
}

/// The index the rule publishes among the certified candidates (`Some`), taken in order: the
/// first, replaced by a later one only when that one's `f` is resolvably below the current
/// choice's ([`PenalizedObjective::resolvably_below`]). So the incumbent, listed first, keeps
/// every tie within rounding. `None` when nothing certified. This is the one selection every
/// published mode goes through (gam#3173).
pub(crate) fn lowest_penalized_index(penalized: &[Option<PenalizedObjective>]) -> Option<usize> {
    let mut choice: Option<(usize, &PenalizedObjective)> = None;
    for (index, candidate) in penalized.iter().enumerate() {
        let Some(candidate) = candidate.as_ref() else {
            continue;
        };
        let replaces = choice.is_none_or(|(_, current)| candidate.resolvably_below(current));
        if replaces {
            choice = Some((index, candidate));
        }
    }
    choice.map(|(index, _)| index)
}

/// The certified candidate the rule publishes ([`lowest_penalized_index`]). `None` when nothing
/// certified.
pub(crate) fn select_lowest_penalized(candidates: Vec<ModeCandidate>) -> Option<ModeSelection> {
    let penalized: Vec<Option<PenalizedObjective>> = candidates
        .iter()
        .map(|candidate| Some(candidate.penalized_objective))
        .collect();
    let winner_index = lowest_penalized_index(&penalized)?;
    let winner_value = candidates[winner_index].penalized_objective.value;
    let runner_up_gap = candidates
        .iter()
        .enumerate()
        .filter(|(index, _)| *index != winner_index)
        .map(|(_, candidate)| candidate.penalized_objective.value - winner_value)
        .reduce(f64::min);
    let certified = candidates.len();
    let winner = candidates.into_iter().nth(winner_index)?;
    Some(ModeSelection {
        winner,
        certified,
        runner_up_gap,
    })
}

/// Whether the family's inner objective may have more than one local minimum: its Hessian depends
/// on β and the family does not certify the objective globally convex. One mode needs one start,
/// and this is the predicate the #2661 anchor and the branch continuation read too.
pub(crate) fn inner_objective_may_have_several_modes<F: CustomFamily + ?Sized>(family: &F) -> bool {
    family.exact_newton_joint_hessian_beta_dependent()
        && !family.inner_coefficient_objective_is_globally_convex()
}

/// The fit's fixed starts (gam#3173): each distinct seed once, in order, and none for a family
/// whose inner objective has one mode.
pub(crate) fn fixed_mode_starts<F: CustomFamily + ?Sized>(
    family: &F,
    starts: impl IntoIterator<Item = Option<ConstrainedWarmStart>>,
) -> Vec<Option<ConstrainedWarmStart>> {
    if !inner_objective_may_have_several_modes(family) {
        return Vec::new();
    }
    let mut distinct: Vec<Option<ConstrainedWarmStart>> = Vec::new();
    for start in starts {
        let identity = SeedIdentity::of(start.as_ref());
        if distinct
            .iter()
            .all(|kept| SeedIdentity::of(kept.as_ref()) != identity)
        {
            distinct.push(start);
        }
    }
    distinct
}
