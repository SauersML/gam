// Child module of `rho_optimizer::run` (see the `#[path]` declaration there), so
// it can read the certified carrier's result for keep-best.

//! Parallel full multistart (gnomon#2359).
//!
//! A seed's start value is an upper bound on the minimum of the basin it drains
//! into, never a lower bound, so no comparison of start values can prove a seed
//! dominated. With no valid lower bound on a basin, every declared start is
//! searched to completion unless a certified quorum releases it (see below).
//! Each seed is its own complete, certified outer run on a lane: a task
//! of the pool the caller runs on (the process worker pool, or a caller's own
//! pool), with no more lanes than that pool has workers. A search starts only once
//! the memory governor grants its predicted working set (SPEC 10); a lane refused
//! while a search is live retires, and the next search to finish respawns it. Only certified runs compete:
//! the lowest certified value wins, values within the criterion's rounding
//! envelope of each other tie, and a tie goes to the lower seed index, so among a
//! FIXED set of certified runs the winner depends neither on which run finished
//! first nor on how many ran at once.
//!
//! Once a second certified run confirms the optimum keep-best would publish, a
//! seed still searching whose own local model no longer forecasts a value below
//! it is released at its next evaluation ([`SeedQuorum`], #3325), so one slow seed
//! no longer gates the fit after the others agree. A released seed never
//! certifies, so the SET of certified runs — and with it the published run — is
//! timing-dependent whenever a route declares an observation count. What the
//! release test establishes is a property of the seed's own quadratic model over
//! the feasible box, not of the objective, so a released seed is one the
//! optimizer's own model says has nothing left to find, not one proved unable to
//! find it. Determinism of the winner is therefore guaranteed only on a route
//! with no observation count, where nothing is released.
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

/// The certified quorum one multistart's seeds share, and the floor below which a
/// still-searching seed must reach to stay in the search (#3325).
///
/// Keep-best publishes the run with the lowest certified value. Once a second
/// certified run reaches the very optimum the current keep-best run certified,
/// that optimum is confirmed, and a seed still searching can change what is
/// published only by certifying below it by more than the certificate's own
/// tolerance. A seed whose every evaluation so far, and whose local model's
/// minimum, stay above that floor is released: its next evaluation is refused, so
/// its search ends within one evaluation instead of gating the fit's wall time.
///
/// "The same optimum" is judged at the certificate's tolerance
/// `max(τ − b, b)` ([`DecrementTolerance`]), with `τ = 1/(2n)` the statistical
/// resolution ([`OuterProblemSize::statistical_resolution`]) and `b` the rounding
/// envelope of the two certified values ([`outer_value_agreement_bound`]): the
/// values differ by at most that, and the displacement `δ` between the two
/// certified points has `½|δᵀHδ|` at most that under both certificates' analytic
/// outer Hessians. The second condition is what makes the two runs one optimum
/// rather than two basins that happen to share a value: `½δᵀHδ ≤ τ` bounds every
/// smooth functional's change along `δ` by `√(2τ)` standard errors, the resolution
/// the certificate itself stops at. A certificate without an analytic Hessian
/// never joins a quorum.
///
/// A route that declares no observation count has no `τ`, and its multistart
/// never releases a seed early; only there is the published run timing-independent.
/// Elsewhere, which seeds are released depends on when the quorum forms, and a
/// released seed never certifies, so the published run is the keep-best of a
/// timing-dependent SET of certified runs. The release test bounds the seed's own
/// quadratic model over the feasible box, which is not a bound on the objective
/// there, so a seed released before it left its current basin could in principle
/// have certified below the floor. This is a deliberate trade of that possibility
/// for the wall time of a search the optimizer's own model says has nothing left,
/// not a proof that the released seed was dominated.
///
/// [`DecrementTolerance`]: crate::rho_optimizer::decrement_bands::DecrementTolerance
pub(super) struct SeedQuorum {
    /// `τ = 1/(2n)`.
    tau: f64,
    /// The confirmed keep-best value less the tolerance, as `f64` bits; NaN until
    /// a quorum forms.
    floor: std::sync::atomic::AtomicU64,
    /// Every run certified so far, in seed order.
    certified: std::sync::Mutex<Vec<QuorumMember>>,
    /// Signalled under `certified` when the floor is first set.
    formed: std::sync::Condvar,
}

/// A certified run as the quorum judges it.
struct QuorumMember {
    index: usize,
    value: f64,
    rho: Array1<f64>,
    hessian: Option<Array2<f64>>,
}

impl SeedQuorum {
    fn new(tau: f64) -> Self {
        Self {
            tau,
            floor: std::sync::atomic::AtomicU64::new(f64::NAN.to_bits()),
            certified: std::sync::Mutex::new(Vec::new()),
            formed: std::sync::Condvar::new(),
        }
    }

    /// The floor a still-searching seed must reach below, once a quorum formed.
    fn floor(&self) -> Option<f64> {
        let floor = f64::from_bits(self.floor.load(Ordering::Acquire));
        (!floor.is_nan()).then_some(floor)
    }

    /// File seed `index`'s certified run, and confirm the keep-best run's optimum
    /// once another certified run reaches it.
    fn record(&self, index: usize, certified: &CertifiedOuterResult, context: &str) {
        let mut members = self
            .certified
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner);
        let at = members.partition_point(|member| member.index < index);
        members.insert(
            at,
            QuorumMember {
                index,
                value: certified.final_value(),
                rho: certified.rho().clone(),
                hessian: certified.final_hessian().cloned(),
            },
        );
        let Some(best) = keep_best(members.iter().map(|member| Some(member.value))) else {
            return;
        };
        let incumbent = &members[best];
        let Some((confirming, tolerance)) = members.iter().enumerate().find_map(|(j, member)| {
            (j != best)
                .then(|| same_optimum(incumbent, member, self.tau))
                .flatten()
                .map(|tolerance| (member.index, tolerance))
        }) else {
            return;
        };
        let floor = incumbent.value - tolerance;
        let previous = f64::from_bits(self.floor.swap(floor.to_bits(), Ordering::AcqRel));
        self.formed.notify_all();
        if previous.is_nan() || previous != floor {
            log::debug!(
                "[OUTER] {context}: multistart quorum: seeds {} and {confirming} certified one \
                 optimum at value={:.9e} (tolerance {tolerance:.3e}); seeds still searching are \
                 released unless their evidence reaches below {floor:.9e}",
                incumbent.index,
                incumbent.value,
            );
        }
    }
}

/// The certificate's tolerance when `a` and `b` certified one optimum, else `None`
/// ([`SeedQuorum`]).
fn same_optimum(a: &QuorumMember, b: &QuorumMember, tau: f64) -> Option<f64> {
    let tolerance = crate::rho_optimizer::decrement_bands::DecrementTolerance {
        tau_stat: tau,
        band_f: outer_value_agreement_bound(a.value, b.value),
    }
    .value();
    if !((a.value - b.value).abs() <= tolerance) || a.rho.len() != b.rho.len() {
        return None;
    }
    let delta = &b.rho - &a.rho;
    if !delta.iter().all(|d| d.is_finite()) {
        return None;
    }
    for hessian in [a.hessian.as_ref()?, b.hessian.as_ref()?] {
        if hessian.dim() != (delta.len(), delta.len()) || !hessian.iter().all(|h| h.is_finite())
        {
            return None;
        }
        if !(0.5 * delta.dot(&hessian.dot(&delta)).abs() <= tolerance) {
            return None;
        }
    }
    Some(tolerance)
}

/// One seed's membership in its multistart's [`SeedQuorum`], carried by the seed's
/// [`OuterProblem`] into [`OuterProblem::run`].
#[derive(Clone)]
pub(super) struct SeedReleaseHandle {
    quorum: Arc<SeedQuorum>,
    seed: usize,
}

impl SeedReleaseHandle {
    /// `inner` behind this seed's release guard, searching the feasible box
    /// `domain`.
    pub(super) fn guard<'a>(
        &'a self,
        inner: &'a mut dyn OuterObjective,
        domain: (Array1<f64>, Array1<f64>),
        context: &str,
    ) -> ReleasableSeed<'a> {
        ReleasableSeed {
            inner,
            handle: self,
            domain,
            context: context.to_string(),
            best_value: f64::INFINITY,
            incumbent: None,
            latest: None,
            released: None,
        }
    }
}

/// A derivative-bearing evaluation the release test reads: its point, value,
/// gradient and Hessian.
type LocalModel = (Array1<f64>, f64, Array1<f64>, HessianValue);

/// A multistart seed's objective behind its release guard ([`SeedQuorum`]).
///
/// It delegates every call. Before each evaluation it refuses, with a fatal
/// evaluation error, once a quorum has formed and this seed's evidence stays
/// above the quorum's floor: its lowest evaluated value, and a lower bound on
/// the minimum over the whole feasible box of the local quadratic model at both
/// its incumbent (the lowest derivative-bearing evaluation) and its latest
/// derivative-bearing evaluation ([`local_model_floor`]). A model is judged only
/// on an analytic Hessian; without one it bounds nothing, so a seed without one
/// is never released, and neither is a seed that has not evaluated with
/// derivatives yet.
pub(super) struct ReleasableSeed<'a> {
    inner: &'a mut dyn OuterObjective,
    handle: &'a SeedReleaseHandle,
    /// The feasible box the seed searches.
    domain: (Array1<f64>, Array1<f64>),
    context: String,
    best_value: f64,
    incumbent: Option<LocalModel>,
    latest: Option<LocalModel>,
    /// Why the seed was released, once it was.
    released: Option<String>,
}

impl ReleasableSeed<'_> {
    /// Refuse the next evaluation once this seed is released.
    fn admit(&mut self) -> Result<(), EstimationError> {
        if self.released.is_none() {
            let Some(floor) = self.handle.quorum.floor() else {
                return Ok(());
            };
            if !(self.best_value >= floor) {
                return Ok(());
            }
            let (Some(incumbent), Some(latest)) = (&self.incumbent, &self.latest) else {
                return Ok(());
            };
            let forecast = local_model_floor(incumbent, &self.domain)
                .min(local_model_floor(latest, &self.domain));
            if !(forecast >= floor) {
                return Ok(());
            }
            let note = format!(
                "multistart seed {} released: a certified quorum confirmed the optimum keep-best \
                 publishes, and neither this seed's evaluations nor its own quadratic model over \
                 the feasible box reaches below {floor:.9e} (lowest evaluated \
                 value {:.9e}, |g|={:.3e}, local-model minimum {forecast:.9e})",
                self.handle.seed,
                self.best_value,
                latest.2.dot(&latest.2).sqrt(),
            );
            log::info!("[OUTER] {}: {note}", self.context);
            self.released = Some(note);
        }
        let note = self.released.clone().unwrap_or_default();
        Err(EstimationError::fatal_objective_evaluation(
            self.context.clone(),
            ::opt::ObjectiveEvalError::fatal(note),
        ))
    }

    fn note_value(&mut self, cost: f64) {
        if cost.is_finite() {
            self.best_value = self.best_value.min(cost);
        }
    }

    fn note_derivatives(&mut self, rho: &Array1<f64>, eval: &OuterEval) {
        self.note_value(eval.cost);
        if !eval.cost.is_finite() {
            return;
        }
        let model = (
            rho.clone(),
            eval.cost,
            eval.gradient.clone(),
            eval.hessian.clone(),
        );
        if self
            .incumbent
            .as_ref()
            .is_none_or(|incumbent| eval.cost <= incumbent.1)
        {
            self.incumbent = Some(model.clone());
        }
        self.latest = Some(model);
    }
}

/// A lower bound on the minimum of the local quadratic model
/// `q(δ) = V + gᵀδ + ½δᵀHδ` over the feasible box, `ρ + δ ∈ [lower, upper]`, or
/// `−∞` when the Hessian is not an analytic, finite matrix.
///
/// Two bounds hold for every `δ` in the box, and the larger is returned:
/// - `δᵀHδ ≥ λ_min|δ|²` makes `q` at least `V + Σᵢ (gᵢδᵢ + ½λ_min δᵢ²)`, a
///   separable function whose minimum over the box is taken coordinate by
///   coordinate. It holds for indefinite curvature too, where the box alone
///   limits the decrease.
/// - On a positive-definite Hessian, `q` is at least its unconstrained minimum
///   `V − ½gᵀH⁻¹g`.
fn local_model_floor(
    (rho, value, gradient, hessian): &LocalModel,
    (lower, upper): &(Array1<f64>, Array1<f64>),
) -> f64 {
    use gam_linalg::faer_ndarray::FaerEigh;
    let Ok(Some(hessian)) = hessian.materialize_dense() else {
        return f64::NEG_INFINITY;
    };
    let dim = gradient.len();
    if hessian.dim() != (dim, dim)
        || rho.len() != dim
        || lower.len() != dim
        || upper.len() != dim
        || !gradient.iter().chain(rho).all(|x| x.is_finite())
    {
        return f64::NEG_INFINITY;
    }
    let Ok((values, vectors)) = hessian.eigh(faer::Side::Lower) else {
        return f64::NEG_INFINITY;
    };
    let Some(lambda_min) = values.iter().copied().reduce(f64::min) else {
        return *value;
    };
    if !values.iter().all(|lambda| lambda.is_finite()) {
        return f64::NEG_INFINITY;
    }
    let mut boxed = 0.0;
    for i in 0..dim {
        // The step to each face, clamped to 0 so a point on a face reads the
        // face as the step's one limit there.
        let (below, above) = ((lower[i] - rho[i]).min(0.0), (upper[i] - rho[i]).max(0.0));
        if !(below.is_finite() && above.is_finite()) {
            return f64::NEG_INFINITY;
        }
        let along = |step: f64| gradient[i] * step + 0.5 * lambda_min * step * step;
        boxed += if lambda_min > 0.0 {
            along((-gradient[i] / lambda_min).clamp(below, above))
        } else {
            along(below).min(along(above))
        };
    }
    let mut bound = value + boxed;
    if lambda_min > 0.0 {
        let decrease: f64 = values
            .iter()
            .enumerate()
            .map(|(k, &lambda)| {
                let along = vectors.column(k).dot(gradient);
                along * along / lambda
            })
            .sum();
        bound = bound.max(value - 0.5 * decrease);
    }
    bound
}

impl OuterObjective for ReleasableSeed<'_> {
    fn capability(&self) -> OuterCapability {
        self.inner.capability()
    }

    fn eval_cost(&mut self, rho: &Array1<f64>) -> Result<f64, EstimationError> {
        self.admit()?;
        let cost = self.inner.eval_cost(rho)?;
        self.note_value(cost);
        Ok(cost)
    }

    fn eval(&mut self, rho: &Array1<f64>) -> Result<OuterEval, EstimationError> {
        self.admit()?;
        let eval = self.inner.eval(rho)?;
        self.note_derivatives(rho, &eval);
        Ok(eval)
    }

    fn eval_with_order(
        &mut self,
        rho: &Array1<f64>,
        order: OuterEvalOrder,
    ) -> Result<OuterEval, EstimationError> {
        self.admit()?;
        let eval = self.inner.eval_with_order(rho, order)?;
        match order {
            // A value-only evaluation's gradient slot is a placeholder.
            OuterEvalOrder::Value => self.note_value(eval.cost),
            OuterEvalOrder::ValueAndGradient | OuterEvalOrder::ValueGradientHessian => {
                self.note_derivatives(rho, &eval)
            }
        }
        Ok(eval)
    }

    fn eval_efs(&mut self, rho: &Array1<f64>) -> Result<EfsEval, EstimationError> {
        self.admit()?;
        let eval = self.inner.eval_efs(rho)?;
        self.note_value(eval.cost);
        Ok(eval)
    }

    fn eval_fixed_point_certificate(
        &mut self,
        rho: &Array1<f64>,
    ) -> Result<FixedPointCertificateEval, EstimationError> {
        self.admit()?;
        let eval = self.inner.eval_fixed_point_certificate(rho)?;
        self.note_value(eval.cost);
        Ok(eval)
    }

    fn rail_face_limit(
        &mut self,
        rho: &Array1<f64>,
        face: &[usize],
    ) -> Result<RailFaceLimitOutcome, EstimationError> {
        self.inner.rail_face_limit(rho, face)
    }

    fn criterion_invariant_directions(&mut self, theta: &Array1<f64>) -> Option<Array2<f64>> {
        self.inner.criterion_invariant_directions(theta)
    }

    fn criterion_rank(&self) -> Option<crate::rho_optimizer::objective::CriterionRank> {
        self.inner.criterion_rank()
    }

    fn reset(&mut self) {
        self.inner.reset();
    }

    fn owns_terminal_coefficient_mode(&self) -> bool {
        self.inner.owns_terminal_coefficient_mode()
    }

    fn begin_exact_polish(&mut self) -> bool {
        self.inner.begin_exact_polish()
    }

    fn seed_inner_state(&mut self, beta: &Array1<f64>) -> Result<SeedOutcome, EstimationError> {
        self.inner.seed_inner_state(beta)
    }

    fn outer_domain_upper_bound(&self) -> Result<Option<Array1<f64>>, EstimationError> {
        self.inner.outer_domain_upper_bound()
    }

    fn outer_domain_lower_bound(&self) -> Result<Option<Array1<f64>>, EstimationError> {
        self.inner.outer_domain_lower_bound()
    }

    fn outer_device_admission(&self) -> Option<gam_gpu::policy::RemlOuterAdmission> {
        self.inner.outer_device_admission()
    }

    fn reactive_domain_scalar_contract(
        &self,
    ) -> Result<Option<crate::continuation_path::ContinuationScalarContract>, EstimationError> {
        self.inner.reactive_domain_scalar_contract()
    }

    fn install_reactive_domain_scalar_state(
        &mut self,
        state: &crate::continuation_path::ContinuationScalarState,
    ) -> Result<(), EstimationError> {
        self.inner.install_reactive_domain_scalar_state(state)
    }

    fn begin_reactive_domain_waypoint(&mut self) -> Result<(), EstimationError> {
        self.inner.begin_reactive_domain_waypoint()
    }

    fn commit_reactive_domain_waypoint(
        &mut self,
        rho: &Array1<f64>,
    ) -> Result<(), EstimationError> {
        self.inner.commit_reactive_domain_waypoint(rho)
    }

    fn rollback_reactive_domain_waypoint(&mut self) -> Result<(), EstimationError> {
        self.inner.rollback_reactive_domain_waypoint()
    }

    fn curvature_homotopy_entry(
        &mut self,
        rho: &Array1<f64>,
    ) -> Option<Result<bool, EstimationError>> {
        self.inner.curvature_homotopy_entry(rho)
    }

    fn accept_seed_without_outer_iterations(
        &mut self,
        rho: &Array1<f64>,
    ) -> Result<Option<f64>, EstimationError> {
        self.inner.accept_seed_without_outer_iterations(rho)
    }

    fn terminal_eval_order(&self) -> Option<OuterEvalOrder> {
        self.inner.terminal_eval_order()
    }

    fn finalize_outer_result(
        &mut self,
        rho: &Array1<f64>,
        plan: &OuterPlan,
    ) -> Result<(), EstimationError> {
        self.inner.finalize_outer_result(rho, plan)
    }
}

/// The shared state of one multistart's lanes.
struct SeedLanes<'a, R, Run> {
    context: &'a str,
    admission: LaneAdmission<'a>,
    problems: Vec<std::sync::Mutex<Option<OuterProblem>>>,
    finished: FinishedSeeds<R>,
    next_seed: AtomicUsize,
    run_seed: &'a Run,
    /// The quorum this multistart's certified runs file into, and whose floor
    /// releases a seed still searching ([`SeedQuorum`]); `None` when no quorum
    /// can form.
    quorum: Option<Arc<SeedQuorum>>,
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
            let retired = if let Some(mut problem) = problem {
                problem.seed_release = self.quorum.as_ref().map(|quorum| SeedReleaseHandle {
                    quorum: Arc::clone(quorum),
                    seed: index,
                });
                let seed_started = std::time::Instant::now();
                let outcome = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
                    gam_runtime::parallel::with_top_level(self.top_level, || {
                        gam_linalg::faer_ndarray::with_faer_sequential(|| {
                            (self.run_seed)(index, problem, std::sync::Arc::clone(&budget))
                        })
                    })
                }))
                .map(|(outcome, payload)| (outcome, payload, seed_started.elapsed().as_secs_f64()));
                if let (Some(quorum), Ok((Ok(certified), _, _))) = (&self.quorum, &outcome) {
                    quorum.record(index, certified, self.context);
                }
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
        // A quorum needs two certified runs, and a route that declares no
        // observation count has no resolution to judge one optimum at.
        let quorum = problems
            .first()
            .and_then(|problem| problem.problem_size.statistical_resolution())
            .filter(|_| seeds.len() > 1)
            .map(|tau| Arc::new(SeedQuorum::new(tau)));
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
            quorum,
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

#[cfg(test)]
mod test_support {
    use super::*;

    impl SeedQuorum {
        /// Block until a quorum has formed, or `timeout` elapses; the floor if one
        /// exists by then.
        pub(crate) fn wait_floor(&self, timeout: std::time::Duration) -> Option<f64> {
            let deadline = std::time::Instant::now() + timeout;
            let mut members = self
                .certified
                .lock()
                .unwrap_or_else(std::sync::PoisonError::into_inner);
            loop {
                if let Some(floor) = self.floor() {
                    return Some(floor);
                }
                let now = std::time::Instant::now();
                if now >= deadline {
                    return None;
                }
                members = self
                    .formed
                    .wait_timeout(members, deadline - now)
                    .unwrap_or_else(std::sync::PoisonError::into_inner)
                    .0;
            }
        }
    }
}
