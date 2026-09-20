// Child module of `rho_optimizer::run` (see the `#[path]` declaration there), so
// it can read the certified carrier's result for keep-best.

//! Parallel full multistart (gnomon#2359).
//!
//! A seed's start value is an upper bound on the minimum of the basin it drains
//! into, never a lower bound, so no comparison of start values can prove a seed
//! dominated. With no valid lower bound on a basin, every declared start is
//! searched. Each seed is its own complete, certified outer run on a lane: a task
//! of the pool the caller runs on (the process worker pool, or a caller's own
//! pool), with no more lanes than that pool has workers. A search starts only once
//! the memory governor grants its predicted working set (SPEC 10); a lane refused
//! while a search is live retires, and the next search to finish respawns it. Only certified runs compete:
//! the lowest certified value wins, values within the criterion's rounding
//! envelope of each other tie, and a tie goes to the lower seed index, so the
//! winner depends neither on which run finished first nor on how many ran at once.
//!
//! A finished run's payload (for a custom family, its terminal inner mode and
//! every O(n) buffer that carries) is kept only while it can still be the one
//! published, and it keeps its search's grant on the ledger until it is dropped
//! (#3238).

use super::*;

/// One parallel multistart: the seeds, each run's outcome, the keep-best winner
/// among the certified runs, the one payload a caller publishes, and how they
/// ran.
pub struct MultistartOutcome<R> {
    pub seeds: Vec<Array1<f64>>,
    /// Every seed's outcome, in seed order.
    pub outcomes: Vec<Result<CertifiedOuterResult, EstimationError>>,
    pub winner: Option<usize>,
    /// The caller's payload from the winner's run or, when no seed certified,
    /// from the first seed's run. No other payload outlives the multistart: each
    /// was dropped as soon as the finished outcomes showed it could not be this
    /// one (#3238).
    pub payload: R,
    /// Lanes the seeds ran on.
    pub lanes: usize,
    /// The most searches the memory governor had live at once.
    pub most_live: usize,
}

impl<R> MultistartOutcome<R> {
    /// The refusal when no seed certified: every seed's start, in seed order, with
    /// its certificate verdict (the typed refusal it ended on, by variant name, and
    /// its message), so no uncertified run is ever published in place of a
    /// certified one.
    pub fn refusal(&self, context: &str) -> EstimationError {
        let outcomes: Vec<String> = self
            .seeds
            .iter()
            .zip(&self.outcomes)
            .enumerate()
            .map(|(index, (seed, outcome))| match outcome {
                Ok(certified) => format!(
                    "seed {index} rho={:?}: certified value={:.9e}",
                    seed.to_vec(),
                    certified.final_value()
                ),
                Err(error) => format!(
                    "seed {index} rho={:?}: not certified [{}]: {error}",
                    seed.to_vec(),
                    error.variant_name()
                ),
            })
            .collect();
        EstimationError::RemlOptimizationFailed(format!(
            "{context}: no multistart seed certified an outer optimum ({} seeds searched): {}",
            self.seeds.len(),
            outcomes.join("; ")
        ))
    }
}

/// Admission of seed runs against the memory governor: a run starts only once the
/// governor has granted its predicted working set, and with no run live it starts
/// anyway, since it is then the serial run. It never fails and never takes a
/// smaller working set.
///
/// Admission never blocks. The lanes are pool tasks, and a worker that waits on a
/// live run can be the very worker running that run beneath it (it stole the
/// waiting lane while joining inside the run). A lane refused while a run is live
/// therefore retires instead, and the next run to finish respawns every retired
/// lane.
///
/// A finished run whose payload can still be published keeps its grant: the
/// payload is state its search built, inside the working set the grant charged,
/// so the ledger carries it until [`FinishedSeeds`] drops it (#3238).
struct LaneAdmission<'a> {
    governor: &'a gam_runtime::resource::MemoryGovernor,
    working_set_bytes: usize,
    serial_available_bytes: u64,
    lanes: std::sync::Mutex<LaneCount>,
    /// The most runs that were live at once.
    most_live: AtomicUsize,
}

#[derive(Default)]
struct LaneCount {
    live: usize,
    retired: usize,
}

impl LaneAdmission<'_> {
    fn lanes(&self) -> std::sync::MutexGuard<'_, LaneCount> {
        self.lanes
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner)
    }

    /// The lane budget of an admitted run, or `None` when the lane retires.
    fn admit(
        &self,
        context: &str,
    ) -> Option<std::sync::Arc<gam_runtime::resource::SearchLaneBudget>> {
        let mut lanes = self.lanes();
        let grant = match self.governor.try_reserve(self.working_set_bytes, context) {
            Ok(grant) => Some(grant),
            Err(_) if lanes.live == 0 => None,
            Err(_) => {
                lanes.retired += 1;
                return None;
            }
        };
        lanes.live += 1;
        self.most_live.fetch_max(lanes.live, Ordering::Relaxed);
        Some(std::sync::Arc::new(
            gam_runtime::resource::SearchLaneBudget::new(self.serial_available_bytes, grant),
        ))
    }

    /// End an admitted run; returns how many retired lanes to respawn. `retired`
    /// holds the payloads the run's outcome just ruled out, its own among them
    /// unless it can still be published, each with the lane whose grant charged
    /// it: they are freed, then their grants return to the ledger.
    fn release<R>(&self, retired: Vec<RetiredSeed<R>>) -> usize {
        let lanes_to_free: Vec<_> = retired
            .into_iter()
            .map(|(payload, lane)| {
                drop(payload);
                lane
            })
            .collect();
        let mut lanes = self.lanes();
        for lane in lanes_to_free {
            lane.release_grant();
        }
        lanes.live -= 1;
        std::mem::take(&mut lanes.retired)
    }
}

/// A payload a finished run no longer needs, with the lane whose grant charged
/// it (`None` for a run that panicked and left none).
type RetiredSeed<R> = (
    Option<R>,
    std::sync::Arc<gam_runtime::resource::SearchLaneBudget>,
);

/// What one seed's run ended with: its outcome and wall time, or its panic.
type SeedRun = std::thread::Result<(Result<CertifiedOuterResult, EstimationError>, f64)>;

/// A finished seed run: how it ended and, while its payload can still be the
/// one published, that payload with the lane whose grant charges it.
struct FinishedSeed<R> {
    ran: SeedRun,
    held: Option<(R, std::sync::Arc<gam_runtime::resource::SearchLaneBudget>)>,
}

/// The finished seed runs of one multistart, in seed order (#3238).
///
/// Filing a run drops, at once, every payload the outcomes finished so far rule
/// out of publication ([`publishable_payloads`]). Holding every payload until the
/// last seed finished kept S − c of them (S seeds, c lanes) outside the working
/// set the admission charged: a custom family's `CustomOuterState`, whose terminal
/// inner mode owns per-block linear predictors and working sets of n rows each.
struct FinishedSeeds<R> {
    seeds: std::sync::Mutex<Vec<Option<FinishedSeed<R>>>>,
}

impl<R> FinishedSeeds<R> {
    fn new(count: usize) -> Self {
        Self {
            seeds: std::sync::Mutex::new((0..count).map(|_| None).collect()),
        }
    }

    /// File seed `index`'s run, made on `lane`, and return every payload that can
    /// no longer be published, for [`LaneAdmission::release`] to free.
    fn file(
        &self,
        index: usize,
        ran: std::thread::Result<(Result<CertifiedOuterResult, EstimationError>, R, f64)>,
        lane: &std::sync::Arc<gam_runtime::resource::SearchLaneBudget>,
    ) -> Vec<RetiredSeed<R>> {
        let mut seeds = self
            .seeds
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner);
        let mut retired = Vec::new();
        seeds[index] = Some(match ran {
            Ok((outcome, payload, seconds)) => FinishedSeed {
                ran: Ok((outcome, seconds)),
                held: Some((payload, std::sync::Arc::clone(lane))),
            },
            Err(panic) => {
                retired.push((None, std::sync::Arc::clone(lane)));
                FinishedSeed {
                    ran: Err(panic),
                    held: None,
                }
            }
        });
        let values: Vec<Option<Option<f64>>> = seeds
            .iter()
            .map(|seed| seed.as_ref().map(|seed| certified_value(&seed.ran)))
            .collect();
        for (seed, publishable) in seeds.iter_mut().zip(publishable_payloads(&values)) {
            if !publishable
                && let Some((payload, lane)) = seed.as_mut().and_then(|seed| seed.held.take())
            {
                retired.push((Some(payload), lane));
            }
        }
        retired
    }

    fn into_inner(self) -> Vec<Option<FinishedSeed<R>>> {
        self.seeds
            .into_inner()
            .unwrap_or_else(std::sync::PoisonError::into_inner)
    }
}

/// The value keep-best compares for a finished run, or `None` when the run did not
/// certify (a panic included).
fn certified_value(ran: &SeedRun) -> Option<f64> {
    match ran {
        Ok((Ok(certified), _)) => Some(certified.final_value()),
        _ => None,
    }
}

/// Whether each seed's payload can still be the one a multistart publishes, from
/// the runs finished so far: `values[i]` is `None` while seed `i` runs, and
/// `Some(v)` once it has finished, with `v` its certified value or `None` when it
/// did not certify (#3238).
///
/// The published payload is the winner's, or the first seed's when no seed
/// certifies. Keep-best ([`multistart_winner`]) walks the seeds in order, and a run
/// it displaces never becomes the incumbent again, so a certified run cannot win
/// once
/// - every seed before it has finished and the walk up to it leaves another
///   incumbent, or
/// - a certified run after it displaces it: the walk compares the two only if it
///   is still the incumbent there, and it has already lost otherwise.
///
/// A run that did not certify never wins, and the first seed's payload is needed
/// only until some seed certifies, since a winner then exists. The winner at the
/// end is never ruled out: the full walk keeps it from its own step on.
fn publishable_payloads(values: &[Option<Option<f64>>]) -> Vec<bool> {
    let finished_prefix = values.iter().take_while(|value| value.is_some()).count();
    let prefix_incumbent = keep_best(
        values[..finished_prefix]
            .iter()
            .map(|&value| value.flatten()),
    );
    let any_certified = values.iter().any(|value| matches!(value, Some(Some(_))));
    values
        .iter()
        .enumerate()
        .map(|(index, value)| match value {
            None => true,
            Some(None) => index == 0 && !any_certified,
            Some(Some(own)) => {
                let lost_its_step = index < finished_prefix && prefix_incumbent != Some(index);
                let displaced_after = values[index + 1..]
                    .iter()
                    .any(|later| matches!(later, Some(Some(later)) if displaces(*own, *later)));
                !lost_its_step && !displaced_after
            }
        })
        .collect()
}

/// The shared state of one multistart's lanes.
struct SeedLanes<'a, R, Run> {
    context: &'a str,
    admission: LaneAdmission<'a>,
    problems: Vec<std::sync::Mutex<Option<OuterProblem>>>,
    finished: FinishedSeeds<R>,
    next_seed: AtomicUsize,
    run_seed: &'a Run,
    /// Whether the multistart's caller stood at top level: every seed run stands
    /// where it did, so it parallelises exactly as a lone search there would.
    top_level: bool,
}

impl<R, Run> SeedLanes<'_, R, Run>
where
    R: Send,
    Run: Fn(
            usize,
            OuterProblem,
            std::sync::Arc<gam_runtime::resource::SearchLaneBudget>,
        ) -> (Result<CertifiedOuterResult, EstimationError>, R)
        + Sync,
{
    /// A lane: runs seeds one after another, each once the governor has admitted
    /// it, until none is left or admission retires the lane.
    fn run<'s>(&'s self, scope: &rayon::Scope<'s>) {
        while self.next_seed.load(Ordering::Relaxed) < self.problems.len() {
            let Some(budget) = self.admission.admit(self.context) else {
                return;
            };
            let index = self.next_seed.fetch_add(1, Ordering::Relaxed);
            let problem = self.problems.get(index).and_then(|slot| {
                slot.lock()
                    .unwrap_or_else(std::sync::PoisonError::into_inner)
                    .take()
            });
            let ran = problem.is_some();
            let retired = if let Some(problem) = problem {
                let seed_started = std::time::Instant::now();
                let outcome = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
                    gam_runtime::parallel::with_top_level(self.top_level, || {
                        gam_linalg::faer_ndarray::with_faer_sequential(|| {
                            (self.run_seed)(index, problem, std::sync::Arc::clone(&budget))
                        })
                    })
                }))
                .map(|(outcome, payload)| (outcome, payload, seed_started.elapsed().as_secs_f64()));
                self.finished.file(index, outcome, &budget)
            } else {
                vec![(None, std::sync::Arc::clone(&budget))]
            };
            for _ in 0..self.admission.release(retired) {
                scope.spawn(move |scope| self.run(scope));
            }
            if !ran {
                return;
            }
        }
    }
}

impl OuterProblem {
    /// The starts a multistart searches, in seed order: `leading` (a joined warm
    /// start) when given, then this problem's own derived start
    /// ([`outer_start_point`](crate::rho_optimizer::run_plan::outer_start_point)),
    /// then one start per level of `additional_levels`, every coordinate at that
    /// level. Each is projected into the search box and exact duplicates are
    /// dropped, so a level beyond a face lands on that face.
    pub fn multistart_seeds(
        &self,
        leading: Option<Array1<f64>>,
        additional_levels: &[f64],
    ) -> Result<Vec<Array1<f64>>, EstimationError> {
        let config = self.config();
        let model_domain_bounds = outer_model_domain_bounds_template(&config, self.n_params);
        let bounds = outer_search_bounds_template(&config, self.n_params);
        let own = crate::rho_optimizer::run_plan::outer_start_point(
            &config,
            self.n_params,
            &model_domain_bounds,
        )?;
        let mut seeds: Vec<Array1<f64>> = Vec::with_capacity(2 + additional_levels.len());
        for start in leading.into_iter().chain(std::iter::once(own)).chain(
            additional_levels
                .iter()
                .map(|&level| Array1::from_elem(self.n_params, level)),
        ) {
            let projected = project_to_bounds(&start, Some(&bounds));
            if !seeds.contains(&projected) {
                seeds.push(projected);
            }
        }
        Ok(seeds)
    }

    /// Search every start of [`Self::multistart_seeds`], each by `run_seed`, and
    /// pick the winner among the certified runs.
    ///
    /// `run_seed(index, problem, lane)` receives this problem started at that one
    /// seed (`with_initial_rho`) and without a cache session, which belongs to
    /// one search, and the memory lane the search runs on. It builds its own
    /// objective, runs it (normally through [`OuterProblem::run_certified`]) and
    /// returns the outcome with whatever state the caller needs from the winner.
    /// Only the winner's state is published, or the first seed's when no seed
    /// certifies; every other run's is dropped as soon as the finished outcomes
    /// rule it out, and until then it holds its search's grant.
    ///
    /// `serial_available_bytes` is the available memory the caller read once before
    /// launch, and `working_set_bytes` its prediction of one search's working set
    /// with its caches at their serial size on that availability (SPEC 10). A
    /// search starts only once the memory governor grants the working set, and its
    /// memory-dependent choices read the lane: that availability and its own pins. So every
    /// search takes the path it takes running alone, and the winner does not
    /// depend on how many ran at once.
    ///
    /// A warm start ([`OuterProblem::with_warm_start`]) on the parent's own inputs
    /// is one run first, which accepts the parent's point where it stands or
    /// declines; the outcome is then that one run. On other inputs the point is
    /// one more seed.
    pub fn run_certified_multistart<R, Run>(
        &self,
        additional_levels: &[f64],
        context: &str,
        serial_available_bytes: u64,
        working_set_bytes: usize,
        run_seed: Run,
    ) -> Result<MultistartOutcome<R>, EstimationError>
    where
        R: Send,
        Run: Fn(
                usize,
                OuterProblem,
                std::sync::Arc<gam_runtime::resource::SearchLaneBudget>,
            ) -> (Result<CertifiedOuterResult, EstimationError>, R)
            + Sync,
    {
        self.run_certified_multistart_on(
            gam_runtime::resource::MemoryGovernor::global(),
            additional_levels,
            context,
            serial_available_bytes,
            working_set_bytes,
            run_seed,
        )
    }

    /// [`Self::run_certified_multistart`] against `governor`'s ledger.
    pub(crate) fn run_certified_multistart_on<R, Run>(
        &self,
        governor: &gam_runtime::resource::MemoryGovernor,
        additional_levels: &[f64],
        context: &str,
        serial_available_bytes: u64,
        working_set_bytes: usize,
        run_seed: Run,
    ) -> Result<MultistartOutcome<R>, EstimationError>
    where
        R: Send,
        Run: Fn(
                usize,
                OuterProblem,
                std::sync::Arc<gam_runtime::resource::SearchLaneBudget>,
            ) -> (Result<CertifiedOuterResult, EstimationError>, R)
            + Sync,
    {
        // A warm start (gam#3002), if its point has this search's dimension.
        let warm_start = self.warm_start_for_this_search().cloned();
        // On the parent's own inputs the point is offered as a prior certificate
        // first, as one run of its own: that run accepts the point where it stands
        // or declines without searching (`resume_only`), and a decline leaves the
        // multistart to run exactly as it runs cold.
        if let Some(warm_start) = warm_start
            .as_ref()
            .filter(|warm_start| warm_start.same_inputs)
        {
            let mut resume = self.clone().with_initial_rho(warm_start.theta.clone());
            resume.cache_session = None;
            resume.cache_mirror_sessions.clear();
            resume.resume_only = true;
            let attempt = Self::run_seed_problems(
                governor,
                context,
                serial_available_bytes,
                working_set_bytes,
                vec![warm_start.theta.clone()],
                vec![resume],
                &run_seed,
            )?;
            if attempt.winner.is_some() {
                return Ok(attempt);
            }
            warm_start.record(gam_model_api::WarmStartOutcome::NotUsed(
                super::RESUME_DECLINED,
            ));
        }
        // On other inputs the point joins this argmin as one more start, so the
        // winner is taken over a superset of the cold starts and its V is at most
        // the cold winner's, within the tie envelope. It is projected into this
        // fit's search box and deduplicated like every other start.
        let cold = self.without_warm_start();
        let joined = warm_start
            .as_ref()
            .filter(|warm_start| !warm_start.same_inputs);
        let seeds = cold.multistart_seeds(
            joined.map(|warm_start| warm_start.theta.clone()),
            additional_levels,
        )?;
        if let Some(warm_start) = joined {
            warm_start.record(gam_model_api::WarmStartOutcome::JoinedMultistart);
        }
        // One problem per seed, started there, without the cache session, which belongs
        // to one search. The joined point's inner mode seeds only its own run.
        let problems = seeds
            .iter()
            .map(|seed| {
                let mut problem = cold.clone().with_initial_rho(seed.clone());
                problem.cache_session = None;
                problem.cache_mirror_sessions.clear();
                problem.warm_start =
                    joined
                        .filter(|warm_start| &warm_start.theta == seed)
                        .map(|warm_start| super::BoundInnerSeed {
                            theta: warm_start.theta.clone(),
                            beta: warm_start.beta.clone(),
                        });
                problem
            })
            .collect();
        Self::run_seed_problems(
            governor,
            context,
            serial_available_bytes,
            working_set_bytes,
            seeds,
            problems,
            &run_seed,
        )
    }

    /// Run `problems[i]`, the search from `seeds[i]`, by `run_seed` on lanes the
    /// memory governor admits, and pick the keep-best winner among the certified
    /// runs.
    fn run_seed_problems<R, Run>(
        governor: &gam_runtime::resource::MemoryGovernor,
        context: &str,
        serial_available_bytes: u64,
        working_set_bytes: usize,
        seeds: Vec<Array1<f64>>,
        problems: Vec<OuterProblem>,
        run_seed: &Run,
    ) -> Result<MultistartOutcome<R>, EstimationError>
    where
        R: Send,
        Run: Fn(
                usize,
                OuterProblem,
                std::sync::Arc<gam_runtime::resource::SearchLaneBudget>,
            ) -> (Result<CertifiedOuterResult, EstimationError>, R)
            + Sync,
    {
        let top_level = gam_runtime::parallel::at_top_level();
        let lanes = SeedLanes {
            context,
            admission: LaneAdmission {
                governor,
                working_set_bytes,
                serial_available_bytes,
                lanes: std::sync::Mutex::new(LaneCount::default()),
                most_live: AtomicUsize::new(0),
            },
            problems: problems
                .into_iter()
                .map(|problem| std::sync::Mutex::new(Some(problem)))
                .collect(),
            finished: FinishedSeeds::new(seeds.len()),
            next_seed: AtomicUsize::new(0),
            run_seed,
            top_level,
        };
        // The lanes are tasks of the pool the caller runs on: the process pool, or a
        // caller's own pool (gnomon's calibrate pool, for one), whose workers and
        // stacks then run every seed. No more lanes than that pool has workers, so
        // the multistart never runs more threads than the caller gave it.
        let started = std::time::Instant::now();
        let concurrency = gam_runtime::parallel::install(|| {
            let concurrency = seeds.len().min(rayon::current_num_threads().max(1));
            log::debug!(
                "[OUTER] {context}: multistart searches all {} starts on {concurrency} \
                 lanes ({working_set_bytes} bytes predicted per search, {} remaining in the \
                 memory budget, {serial_available_bytes} bytes available before launch)",
                seeds.len(),
                governor.remaining_bytes(),
            );
            gam_runtime::parallel::fan_out(|| {
                rayon::scope(|scope| {
                    for _ in 0..concurrency {
                        scope.spawn(|scope| lanes.run(scope));
                    }
                });
            });
            concurrency
        });
        let SeedLanes {
            admission,
            finished,
            ..
        } = lanes;
        let most_live = admission.most_live.load(Ordering::Relaxed);
        let mut outcomes = Vec::with_capacity(seeds.len());
        let mut held = Vec::with_capacity(seeds.len());
        for (index, seed) in finished.into_inner().into_iter().enumerate() {
            let seed = seed.ok_or_else(|| {
                EstimationError::RemlOptimizationFailed(format!(
                    "{context}: multistart seed {index} produced no outcome"
                ))
            })?;
            let (outcome, seconds) = seed
                .ran
                .unwrap_or_else(|panic| std::panic::resume_unwind(panic));
            match &outcome {
                Ok(certified) => log::debug!(
                    "[OUTER] {context}: multistart seed {index} rho={:?} certified value={:?} at \
                     rho={:?} after {} iterations in {seconds:.3}s",
                    seeds[index].to_vec(),
                    certified.final_value(),
                    certified.rho().to_vec(),
                    certified.iterations(),
                ),
                Err(error) => log::debug!(
                    "[OUTER] {context}: multistart seed {index} rho={:?} did not certify in \
                     {seconds:.3}s: {error}",
                    seeds[index].to_vec(),
                ),
            }
            outcomes.push(outcome);
            held.push(seed.held);
        }
        let winner = multistart_winner(&outcomes);
        // Every other payload was dropped when the outcomes ruled it out.
        let published = winner.unwrap_or(0);
        let (payload, published_lane) = held.swap_remove(published).ok_or_else(|| {
            EstimationError::RemlOptimizationFailed(format!(
                "{context}: multistart seed {published}'s payload was dropped before publication"
            ))
        })?;
        // From here the payload is the caller's, as a lone search's state is.
        published_lane.release_grant();
        match winner {
            Some(index) => log::debug!(
                "[OUTER] {context}: multistart winner is seed {index} of {} (value={:.9e}) \
                 after {:.3}s",
                outcomes.len(),
                outcomes[index]
                    .as_ref()
                    .map(CertifiedOuterResult::final_value)
                    .unwrap_or(f64::NAN),
                started.elapsed().as_secs_f64(),
            ),
            None => log::debug!(
                "[OUTER] {context}: no multistart seed certified ({} runs, {:.3}s)",
                outcomes.len(),
                started.elapsed().as_secs_f64(),
            ),
        }
        Ok(MultistartOutcome {
            seeds,
            outcomes,
            winner,
            payload,
            lanes: concurrency,
            most_live,
        })
    }
}

/// Keep-best over the certified runs in seed order. A run that did not certify
/// never competes.
pub(crate) fn multistart_winner(
    outcomes: &[Result<CertifiedOuterResult, EstimationError>],
) -> Option<usize> {
    keep_best(
        outcomes
            .iter()
            .map(|outcome| outcome.as_ref().ok().map(CertifiedOuterResult::final_value)),
    )
}

/// Keep-best over certified values in seed order (`None`: that run did not
/// certify): the first certified value is the incumbent, and each later one
/// replaces it when it [`displaces`] it.
fn keep_best(values: impl IntoIterator<Item = Option<f64>>) -> Option<usize> {
    let mut winner: Option<(usize, f64)> = None;
    for (index, value) in values.into_iter().enumerate() {
        let Some(value) = value else {
            continue;
        };
        if winner.is_none_or(|(_, incumbent)| displaces(incumbent, value)) {
            winner = Some((index, value));
        }
    }
    winner.map(|(index, _)| index)
}

/// Whether a run certified at `value` displaces the incumbent certified at
/// `incumbent`: only when it is lower by more than the rounding envelope of the
/// two values (`outer_value_agreement_bound`), so a tie keeps the lower seed index.
fn displaces(incumbent: f64, value: f64) -> bool {
    incumbent - value > outer_value_agreement_bound(incumbent, value)
}

#[cfg(test)]
#[path = "multistart_tests.rs"]
mod multistart_tests;
