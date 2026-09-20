//! Integer-rank birth/death selection with a continuously profiled
//! log-hyperparameter vector at every rank (#3920).
//!
//! The caller supplies one oracle. At a genuinely discrete rank `r` and a
//! log-hyperparameter vector `θ` it fits its inner model to convergence and
//! returns the negative log evidence `V_r(θ)`, the analytic gradient
//! `∇_θ V_r(θ)` and the magnitude `s_r(θ) > 0` of the terms that gradient is
//! summed from. Everything outside that fit is owned here:
//!
//! * **Continuous profile.** `min_θ V_r(θ)` at each visited rank is found by
//!   `opt`'s BFGS on the oracle's analytic gradient, with no iteration budget
//!   and without the generic relative-stall exit. Every distinct iterate is
//!   evaluated exactly once: evaluations are cached bit-exactly per rank and
//!   the seed evaluation is handed to the solver as its initial sample.
//! * **Stationarity certificate.** A profile is returned only at an iterate
//!   whose own evaluation satisfies `‖∇_θ V_r‖₂ ≤ √ε · s_r(θ)`: `√ε` is the
//!   relative resolution of a gradient summed from terms of magnitude `s_r`,
//!   so a smaller defect is not resolvable from the oracle's arithmetic.
//!   `opt` resolves its gradient tolerance once, at the seed of a run. When a
//!   run stops on that tolerance at a point whose own scale is smaller, the
//!   descent is continued from that point with the tolerance its own
//!   evaluation defines; every continuation starts strictly below the
//!   previous one. Any other stop without the certificate is a typed
//!   [`RankSelectionError::NonConvergence`] carrying the checkpoint.
//! * **Rank walk.** From the current rank the death (`r − 1`) and birth
//!   (`r + 1`) neighbours inside `[0, max_rank]` are profiled, warm-started
//!   from the current rank's profile. The walk moves to the better neighbour
//!   only when it lowers `V` by more than the rounding of the difference of
//!   two evaluated values, `ε (1 + |V_cur| + |V_best|)`, and stops when
//!   neither does. Each rank is profiled once and the walk strictly
//!   decreases `V`, so it terminates over the finite rank set.

use ndarray::Array1;
use opt::{
    Bfgs, BfgsError, FirstOrderSample, FusedObjective, GradientTolerance, MaxIterations,
    ObjectiveEvalError, Solution,
};

/// Why the oracle is being asked for a rank's evidence.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ProfileTransition {
    /// The first evaluation of the initial rank; nothing to warm-start from.
    Seed,
    /// The first evaluation of `r + 1`, warm-started from rank `r`'s profile.
    Birth,
    /// The first evaluation of `r − 1`, warm-started from rank `r`'s profile.
    Death,
    /// A later iterate of the continuous profile at the same rank.
    Continuous,
}

impl ProfileTransition {
    pub fn as_str(self) -> &'static str {
        match self {
            Self::Seed => "seed",
            Self::Birth => "birth",
            Self::Death => "death",
            Self::Continuous => "continuous",
        }
    }
}

/// One oracle request.
#[derive(Debug, Clone)]
pub struct ProfileQuery {
    /// Unique over the whole selection; the oracle keys its fitted state on it.
    pub evaluation_id: usize,
    pub rank: usize,
    pub log_hyperparameters: Array1<f64>,
    pub transition: ProfileTransition,
    /// The `evaluation_id` whose fitted state seeds this inner fit: for
    /// `Birth`/`Death` the current rank's profile, for `Continuous` the
    /// lowest-evidence evaluation of the same rank so far, `None` for `Seed`.
    pub warm_start: Option<usize>,
}

/// The oracle's converged answer to one [`ProfileQuery`].
#[derive(Debug, Clone)]
pub struct ProfileSample {
    pub value: f64,
    pub gradient: Array1<f64>,
    /// Magnitude of the terms the gradient is summed from; sets the
    /// resolution of the stationarity certificate.
    pub gradient_scale: f64,
}

/// A validated oracle evaluation at one `(rank, θ)`.
#[derive(Debug, Clone)]
pub struct ProfileEvaluation {
    pub evaluation_id: usize,
    pub rank: usize,
    pub log_hyperparameters: Array1<f64>,
    pub value: f64,
    pub gradient: Array1<f64>,
    pub gradient_scale: f64,
}

impl ProfileEvaluation {
    /// `‖∇_θ V_r(θ)‖₂`.
    pub fn stationarity_defect(&self) -> f64 {
        self.gradient.dot(&self.gradient).sqrt()
    }

    /// `√ε · s_r(θ)`: the smallest gradient norm resolvable at this point.
    pub fn stationarity_tolerance(&self) -> f64 {
        f64::EPSILON.sqrt() * self.gradient_scale
    }

    pub fn is_stationary(&self) -> bool {
        self.stationarity_defect() <= self.stationarity_tolerance()
    }
}

/// One evaluated rank move of the walk.
#[derive(Debug, Clone, PartialEq)]
pub struct RankMove {
    pub from_rank: usize,
    pub to_rank: usize,
    /// `V(to) − V(from)` between the two continuous profiles.
    pub value_gap: f64,
    pub accepted: bool,
}

/// The selected rank's certified profile, its two neighbours' certified
/// profiles (`None` where the neighbour is outside `[0, max_rank]`) and the
/// ordered record of every evaluated move.
#[derive(Debug, Clone)]
pub struct RankSelection {
    pub selected: ProfileEvaluation,
    pub death: Option<ProfileEvaluation>,
    pub birth: Option<ProfileEvaluation>,
    pub moves: Vec<RankMove>,
    /// Number of oracle evaluations the whole selection made.
    pub evaluations: usize,
}

/// Why [`select_rank_with_profiled_hyperparameters`] has no selection.
#[derive(Debug)]
pub enum RankSelectionError<E> {
    /// The request itself is malformed.
    InvalidRequest(String),
    /// The oracle failed; its own error is returned unchanged.
    Oracle(E),
    /// The oracle returned a sample that is not a finite jet of the queried
    /// dimension with a positive scale.
    InvalidSample { query: ProfileQuery, reason: String },
    /// A rank's continuous profile stopped without the stationarity
    /// certificate. `evaluation` is the checkpoint it stopped at.
    NonConvergence {
        evaluation: ProfileEvaluation,
        reason: String,
    },
}

impl<E: std::fmt::Display> std::fmt::Display for RankSelectionError<E> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::InvalidRequest(reason) => write!(f, "invalid rank selection request: {reason}"),
            Self::Oracle(error) => write!(f, "profile oracle failed: {error}"),
            Self::InvalidSample { query, reason } => write!(
                f,
                "profile oracle returned an invalid sample for evaluation {} at rank {}: {reason}",
                query.evaluation_id, query.rank
            ),
            Self::NonConvergence { evaluation, reason } => write!(
                f,
                "rank {} continuous profile did not converge ({reason}); stationarity defect {} \
                 exceeds {}",
                evaluation.rank,
                evaluation.stationarity_defect(),
                evaluation.stationarity_tolerance()
            ),
        }
    }
}

impl<E: std::fmt::Debug + std::fmt::Display> std::error::Error for RankSelectionError<E> {}

fn same_point_bits(lhs: &Array1<f64>, rhs: &Array1<f64>) -> bool {
    lhs.len() == rhs.len()
        && lhs
            .iter()
            .zip(rhs.iter())
            .all(|(a, b)| a.to_bits() == b.to_bits())
}

fn lowest(evaluations: &[ProfileEvaluation]) -> Option<&ProfileEvaluation> {
    evaluations
        .iter()
        .min_by(|a, b| a.value.total_cmp(&b.value))
}

/// Issues evaluation ids and validates every oracle answer.
struct Evaluator<'o, O> {
    oracle: &'o mut O,
    next_id: usize,
    dimension: usize,
}

impl<O, E> Evaluator<'_, O>
where
    O: FnMut(&ProfileQuery) -> Result<ProfileSample, E>,
{
    fn evaluate(
        &mut self,
        rank: usize,
        log_hyperparameters: Array1<f64>,
        transition: ProfileTransition,
        warm_start: Option<usize>,
    ) -> Result<ProfileEvaluation, RankSelectionError<E>> {
        let query = ProfileQuery {
            evaluation_id: self.next_id,
            rank,
            log_hyperparameters,
            transition,
            warm_start,
        };
        self.next_id += 1;
        let sample = (self.oracle)(&query).map_err(RankSelectionError::Oracle)?;
        let invalid = |reason: String| RankSelectionError::InvalidSample {
            query: query.clone(),
            reason,
        };
        if sample.gradient.len() != self.dimension {
            return Err(invalid(format!(
                "gradient has length {}, expected {}",
                sample.gradient.len(),
                self.dimension
            )));
        }
        if !sample.value.is_finite() || !sample.gradient.iter().all(|g| g.is_finite()) {
            return Err(invalid("value or gradient is not finite".to_string()));
        }
        if !(sample.gradient_scale.is_finite() && sample.gradient_scale > 0.0) {
            return Err(invalid(format!(
                "gradient_scale must be finite and > 0, got {}",
                sample.gradient_scale
            )));
        }
        Ok(ProfileEvaluation {
            evaluation_id: query.evaluation_id,
            rank,
            log_hyperparameters: query.log_hyperparameters,
            value: sample.value,
            gradient: sample.gradient,
            gradient_scale: sample.gradient_scale,
        })
    }
}

/// How one BFGS run ended, with the evaluation it ended at.
enum RunEnd {
    /// `opt` reported a converged termination.
    Converged(ProfileEvaluation, &'static str),
    /// `opt` stopped without convergence.
    Stopped(ProfileEvaluation, String),
}

/// Minimise `V_rank(θ)` from `seed` and return the certified profile.
fn profile_rank<O, E>(
    evaluator: &mut Evaluator<'_, O>,
    seed: ProfileEvaluation,
) -> Result<ProfileEvaluation, RankSelectionError<E>>
where
    O: FnMut(&ProfileQuery) -> Result<ProfileSample, E>,
{
    let rank = seed.rank;
    let mut evaluations = vec![seed.clone()];
    let mut current = seed;
    let max_iterations =
        MaxIterations::new(usize::MAX).expect("usize::MAX is a valid iteration count");
    loop {
        if current.is_stationary() {
            return Ok(current);
        }
        let mut failure: Option<RankSelectionError<E>> = None;
        let outcome = {
            let objective = FusedObjective::new(
                |x: &Array1<f64>| -> Result<FirstOrderSample, ObjectiveEvalError> {
                    if let Some(hit) = evaluations
                        .iter()
                        .find(|e| same_point_bits(&e.log_hyperparameters, x))
                    {
                        return Ok(FirstOrderSample {
                            value: hit.value,
                            gradient: hit.gradient.clone(),
                        });
                    }
                    let warm_start = lowest(&evaluations).map(|e| e.evaluation_id);
                    match evaluator.evaluate(
                        rank,
                        x.clone(),
                        ProfileTransition::Continuous,
                        warm_start,
                    ) {
                        Ok(evaluation) => {
                            let sample = FirstOrderSample {
                                value: evaluation.value,
                                gradient: evaluation.gradient.clone(),
                            };
                            evaluations.push(evaluation);
                            Ok(sample)
                        }
                        Err(error) => {
                            failure = Some(error);
                            Err(ObjectiveEvalError::fatal("profile oracle failed"))
                        }
                    }
                },
            );
            let mut solver = Bfgs::new(current.log_hyperparameters.clone(), objective)
                .with_initial_sample(
                    current.log_hyperparameters.clone(),
                    FirstOrderSample {
                        value: current.value,
                        gradient: current.gradient.clone(),
                    },
                )
                .with_gradient_tolerance(GradientTolerance::absolute(
                    current.stationarity_tolerance(),
                ))
                .with_max_iterations(max_iterations)
                .without_relative_stall();
            solver.run()
        };
        if let Some(error) = failure {
            return Err(error);
        }
        let at = |solution: &Solution| -> ProfileEvaluation {
            evaluations
                .iter()
                .find(|e| same_point_bits(&e.log_hyperparameters, &solution.final_point))
                .or_else(|| lowest(&evaluations))
                .cloned()
                .expect("the seed evaluation is always recorded")
        };
        let end = match outcome {
            Ok(solution) if solution.status().is_success() => {
                RunEnd::Converged(at(&solution), solution.termination.label())
            }
            Ok(solution) => RunEnd::Stopped(
                at(&solution),
                format!("BFGS stopped: {}", solution.termination.label()),
            ),
            Err(error) => {
                let checkpoint = match &error {
                    BfgsError::LineSearchFailed { last_solution, .. }
                    | BfgsError::MaxIterationsReached { last_solution } => at(last_solution),
                    _ => lowest(&evaluations)
                        .cloned()
                        .expect("the seed evaluation is always recorded"),
                };
                RunEnd::Stopped(checkpoint, error.to_string())
            }
        };
        match end {
            RunEnd::Converged(evaluation, _) | RunEnd::Stopped(evaluation, _)
                if evaluation.is_stationary() =>
            {
                return Ok(evaluation);
            }
            // `opt` met the tolerance resolved at this run's seed, but the
            // point it stopped at resolves a smaller gradient: continue the
            // descent from it under its own tolerance.
            RunEnd::Converged(evaluation, _) if evaluation.value < current.value => {
                current = evaluation;
            }
            RunEnd::Converged(evaluation, label) => {
                return Err(RankSelectionError::NonConvergence {
                    evaluation,
                    reason: format!(
                        "BFGS converged ({label}) without decreasing the evidence and without \
                         the stationarity certificate"
                    ),
                });
            }
            RunEnd::Stopped(evaluation, reason) => {
                return Err(RankSelectionError::NonConvergence { evaluation, reason });
            }
        }
    }
}

/// Select a discrete rank in `[0, max_rank]` by a strict-improvement
/// birth/death walk from `initial_rank`, continuously profiling the
/// log-hyperparameters at every visited rank from `initial_log_hyperparameters`
/// (see the module docs for the certificate and the acceptance rule).
pub fn select_rank_with_profiled_hyperparameters<O, E>(
    initial_rank: usize,
    max_rank: usize,
    initial_log_hyperparameters: Array1<f64>,
    mut oracle: O,
) -> Result<RankSelection, RankSelectionError<E>>
where
    O: FnMut(&ProfileQuery) -> Result<ProfileSample, E>,
{
    if initial_rank > max_rank {
        return Err(RankSelectionError::InvalidRequest(format!(
            "initial_rank must be in [0, {max_rank}], got {initial_rank}"
        )));
    }
    if initial_log_hyperparameters.is_empty() {
        return Err(RankSelectionError::InvalidRequest(
            "at least one log-hyperparameter is required".to_string(),
        ));
    }
    if !initial_log_hyperparameters.iter().all(|v| v.is_finite()) {
        return Err(RankSelectionError::InvalidRequest(
            "initial log-hyperparameters must be finite".to_string(),
        ));
    }
    let mut evaluator = Evaluator {
        oracle: &mut oracle,
        next_id: 0,
        dimension: initial_log_hyperparameters.len(),
    };
    let mut profiles: std::collections::BTreeMap<usize, ProfileEvaluation> =
        std::collections::BTreeMap::new();
    let mut moves = Vec::new();

    let seed = evaluator.evaluate(
        initial_rank,
        initial_log_hyperparameters,
        ProfileTransition::Seed,
        None,
    )?;
    let mut current = profile_rank(&mut evaluator, seed)?;
    profiles.insert(initial_rank, current.clone());

    loop {
        let mut neighbours: Vec<ProfileEvaluation> = Vec::with_capacity(2);
        let candidates = [
            current
                .rank
                .checked_sub(1)
                .map(|rank| (rank, ProfileTransition::Death)),
            (current.rank < max_rank).then_some((current.rank + 1, ProfileTransition::Birth)),
        ];
        for (rank, transition) in candidates.into_iter().flatten() {
            let profile = match profiles.get(&rank) {
                Some(profile) => profile.clone(),
                None => {
                    let seed = evaluator.evaluate(
                        rank,
                        current.log_hyperparameters.clone(),
                        transition,
                        Some(current.evaluation_id),
                    )?;
                    let profile = profile_rank(&mut evaluator, seed)?;
                    profiles.insert(rank, profile.clone());
                    profile
                }
            };
            neighbours.push(profile);
        }
        let best = neighbours
            .iter()
            .min_by(|a, b| a.value.total_cmp(&b.value))
            .cloned();
        let improves = best.as_ref().is_some_and(|best| {
            let roundoff = f64::EPSILON * (1.0 + current.value.abs() + best.value.abs());
            best.value < current.value - roundoff
        });
        for neighbour in &neighbours {
            moves.push(RankMove {
                from_rank: current.rank,
                to_rank: neighbour.rank,
                value_gap: neighbour.value - current.value,
                accepted: improves
                    && best
                        .as_ref()
                        .is_some_and(|best| best.evaluation_id == neighbour.evaluation_id),
            });
        }
        match best {
            Some(best) if improves => current = best,
            _ => {
                let death = current
                    .rank
                    .checked_sub(1)
                    .and_then(|rank| profiles.get(&rank).cloned());
                let birth = profiles.get(&(current.rank + 1)).cloned();
                return Ok(RankSelection {
                    selected: current,
                    death,
                    birth,
                    moves,
                    evaluations: evaluator.next_id,
                });
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::array;

    const TARGET: [f64; 2] = [0.37, -0.61];

    /// `V_r(θ) = ‖θ − θ*‖² + (r − 2)²`, scale `1 + |V|`, recording queries.
    fn quadratic(
        queries: &mut Vec<ProfileQuery>,
    ) -> impl FnMut(&ProfileQuery) -> Result<ProfileSample, String> + '_ {
        move |query| {
            queries.push(query.clone());
            let d0 = query.log_hyperparameters[0] - TARGET[0];
            let d1 = query.log_hyperparameters[1] - TARGET[1];
            let rank_cost = (query.rank as f64 - 2.0).powi(2);
            let value = d0 * d0 + d1 * d1 + rank_cost;
            Ok(ProfileSample {
                value,
                gradient: array![2.0 * d0, 2.0 * d1],
                gradient_scale: 1.0 + value.abs(),
            })
        }
    }

    #[test]
    fn birth_death_walk_selects_the_evidence_minimising_rank_with_certificates() {
        let mut queries = Vec::new();
        let initial = array![0.2_f64.ln(), 0.8_f64.ln()];
        let selection =
            select_rank_with_profiled_hyperparameters(0, 3, initial, quadratic(&mut queries))
                .expect("quadratic profile selects");
        assert_eq!(selection.selected.rank, 2);
        assert!(selection.selected.is_stationary());
        for (got, want) in selection.selected.log_hyperparameters.iter().zip(TARGET) {
            assert!((got - want).abs() <= 1e-7, "{got} vs {want}");
        }
        let death = selection.death.as_ref().expect("rank 1 is feasible");
        let birth = selection.birth.as_ref().expect("rank 3 is feasible");
        assert_eq!((death.rank, birth.rank), (1, 3));
        assert!(death.is_stationary() && birth.is_stationary());
        assert!(death.value > selection.selected.value);
        assert!(birth.value > selection.selected.value);
        // 0 -> 1 -> 2 accepted, 2's neighbours rejected.
        let accepted: Vec<(usize, usize)> = selection
            .moves
            .iter()
            .filter(|m| m.accepted)
            .map(|m| (m.from_rank, m.to_rank))
            .collect();
        assert_eq!(accepted, vec![(0, 1), (1, 2)]);
        assert!(selection.moves.iter().all(|m| m.to_rank <= 3));
        assert_eq!(selection.evaluations, queries.len());
    }

    #[test]
    fn every_distinct_iterate_is_evaluated_exactly_once() {
        let mut queries = Vec::new();
        select_rank_with_profiled_hyperparameters(
            2,
            3,
            array![0.0, 0.0],
            quadratic(&mut queries),
        )
        .expect("quadratic profile selects");
        for (i, a) in queries.iter().enumerate() {
            for b in &queries[i + 1..] {
                assert!(
                    a.rank != b.rank
                        || !same_point_bits(&a.log_hyperparameters, &b.log_hyperparameters),
                    "rank {} point {:?} evaluated twice",
                    a.rank,
                    a.log_hyperparameters
                );
            }
        }
        assert_eq!(queries[0].transition, ProfileTransition::Seed);
        assert_eq!(queries[0].warm_start, None);
        for query in &queries[1..] {
            assert!(query.warm_start.is_some());
            assert!(query.warm_start.unwrap() < query.evaluation_id);
        }
    }

    #[test]
    fn a_rank_boundary_neighbour_is_structurally_infeasible() {
        let mut queries = Vec::new();
        // max_rank = 2 caps the walk at the optimum: no birth neighbour.
        let selection = select_rank_with_profiled_hyperparameters(
            2,
            2,
            array![0.0, 0.0],
            quadratic(&mut queries),
        )
        .expect("quadratic profile selects");
        assert_eq!(selection.selected.rank, 2);
        assert!(selection.birth.is_none());
        assert_eq!(selection.death.as_ref().map(|d| d.rank), Some(1));
    }

    #[test]
    fn an_oracle_error_is_returned_unchanged() {
        let result = select_rank_with_profiled_hyperparameters(
            0,
            3,
            array![0.0, 0.0],
            |query: &ProfileQuery| -> Result<ProfileSample, String> {
                if query.rank == 1 {
                    Err("inner fit at rank 1 failed".to_string())
                } else {
                    Ok(ProfileSample {
                        value: query.rank as f64,
                        gradient: array![0.0, 0.0],
                        gradient_scale: 1.0,
                    })
                }
            },
        );
        match result {
            Err(RankSelectionError::Oracle(message)) => {
                assert_eq!(message, "inner fit at rank 1 failed");
            }
            other => panic!("expected the oracle error, got {other:?}"),
        }
    }

    #[test]
    fn a_non_finite_sample_is_rejected() {
        let result = select_rank_with_profiled_hyperparameters(
            0,
            1,
            array![0.0],
            |_: &ProfileQuery| -> Result<ProfileSample, String> {
                Ok(ProfileSample {
                    value: f64::NAN,
                    gradient: array![0.0],
                    gradient_scale: 1.0,
                })
            },
        );
        assert!(matches!(
            result,
            Err(RankSelectionError::InvalidSample { .. })
        ));
    }
}
