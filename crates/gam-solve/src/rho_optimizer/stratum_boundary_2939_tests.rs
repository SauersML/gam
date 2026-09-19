// Child module of `run_plan::run_plan_tests` (see the `#[path]` declaration
// there): #2939, a first-order search that meets a kept-rank boundary of the
// face log-determinant (#2765).

use super::*;
use ndarray::array;

// The criterion prices `½·log μ` for an eigenvalue `μ` of the face operator while the
// rank rule keeps it. Along ρ₀ that eigenvalue shrinks, `μ = b − ρ₀ + c`, and the rule
// drops it where `μ` reaches its cutoff `c`, at `ρ₀ = b`. On the kept side the
// criterion falls toward `½·log c` as ρ₀ approaches `b`, and on the other side the
// dropped eigenvalue prices nothing. A smooth part `q = ½(ρ₀ − m₀)² + ½(ρ₁ − 1)²` carries
// the minimum, and `across_shift` offsets the criterion across the boundary. The
// kept-rank side has no stationary point: a search that starts there descends into
// its rank boundary, which is #2939's seed 0 in miniature.
const STRATUM_OFFSET: f64 = 3.0e4;
const STRATUM_BOUNDARY: f64 = 2.0;
const KEPT_RANK_INSIDE: usize = 56;
const KEPT_RANK_ACROSS: usize = 55;

#[derive(Clone, Copy, Debug)]
struct StratumFixture {
    cutoff: f64,
    center0: f64,
    across_shift: f64,
}

impl StratumFixture {
    fn rank(&self, rho: &Array1<f64>) -> usize {
        if rho[0] < STRATUM_BOUNDARY {
            KEPT_RANK_INSIDE
        } else {
            KEPT_RANK_ACROSS
        }
    }

    fn eval(&self, rho: &Array1<f64>) -> OuterEval {
        let d0 = rho[0] - self.center0;
        let d1 = rho[1] - 1.0;
        let smooth = 0.5 * d0 * d0 + 0.5 * d1 * d1;
        let (cost, g0) = if rho[0] < STRATUM_BOUNDARY {
            let mu = STRATUM_BOUNDARY - rho[0] + self.cutoff;
            (STRATUM_OFFSET + smooth + 0.5 * mu.ln(), d0 - 0.5 / mu)
        } else {
            (STRATUM_OFFSET + smooth + self.across_shift, d0)
        };
        OuterEval {
            cost,
            gradient: array![g0, d1],
            hessian: HessianValue::Unavailable,
            inner_beta_hint: None,
        }
    }
}

/// One request the stratum objective served, in the order it was served.
struct StratumRequest {
    order: OuterEvalOrder,
    rho: Array1<f64>,
    rank: usize,
    cost: f64,
}

/// The fixture as an objective that publishes the kept rank of its latest evaluation
/// and records every request, so a test reads what the search asked for.
struct RecordingStratum {
    fixture: StratumFixture,
    requests: Vec<StratumRequest>,
    last_rank: Option<usize>,
}

impl RecordingStratum {
    fn serve(&mut self, rho: &Array1<f64>, order: OuterEvalOrder) -> OuterEval {
        let eval = self.fixture.eval(rho);
        let rank = self.fixture.rank(rho);
        self.last_rank = Some(rank);
        self.requests.push(StratumRequest {
            order,
            rho: rho.clone(),
            rank,
            cost: eval.cost,
        });
        eval
    }
}

impl OuterObjective for RecordingStratum {
    fn capability(&self) -> OuterCapability {
        OuterCapability {
            gradient: Derivative::Analytic,
            hessian: DeclaredHessianForm::Unavailable,
            n_params: 2,
            psi_dim: 0,
            fixed_point_available: false,
            barrier_config: None,
            prefer_gradient_only: true,
            disable_fixed_point: true,
        }
    }
    fn eval_cost(&mut self, rho: &Array1<f64>) -> Result<f64, EstimationError> {
        Ok(self.serve(rho, OuterEvalOrder::Value).cost)
    }
    fn eval(&mut self, rho: &Array1<f64>) -> Result<OuterEval, EstimationError> {
        Ok(self.serve(rho, OuterEvalOrder::ValueAndGradient))
    }
    fn eval_with_order(
        &mut self,
        rho: &Array1<f64>,
        order: OuterEvalOrder,
    ) -> Result<OuterEval, EstimationError> {
        Ok(self.serve(rho, order))
    }
    fn criterion_rank(&self) -> Option<usize> {
        self.last_rank
    }
    fn reset(&mut self) {}
    fn seed_inner_state(&mut self, beta: &Array1<f64>) -> Result<SeedOutcome, EstimationError> {
        if beta.iter().any(|value| !value.is_finite()) {
            return Err(EstimationError::RemlOptimizationFailed(format!(
                "the recording stratum was offered a non-finite inner seed of length {}",
                beta.len()
            )));
        }
        Ok(SeedOutcome::NoSlot)
    }
}

/// The gradient-only plan #2939's fit ran, at production's default tolerance and the
/// criterion's declared scale.
fn stratum_problem() -> OuterProblem {
    OuterProblem::new(2)
        .with_gradient(Derivative::Analytic)
        .with_hessian(DeclaredHessianForm::Unavailable)
        .with_prefer_gradient_only(true)
        .with_tolerance(OuterConfig::default().tolerance)
        .with_objective_scale(Some(STRATUM_OFFSET))
        .with_bounds(Array1::from_elem(2, -20.0), Array1::from_elem(2, 20.0))
        .with_initial_rho(array![0.0, 0.0])
        .with_screen_initial_rho(false)
        .with_seed_config(gam_problem::SeedConfig {
            max_seeds: 1,
            seed_budget: 1,
            ..Default::default()
        })
}

fn run_stratum(
    fixture: StratumFixture,
    context: &str,
) -> (Result<OuterResult, EstimationError>, Vec<StratumRequest>) {
    let problem = stratum_problem();
    let mut obj = RecordingStratum {
        fixture,
        requests: Vec::new(),
        last_rank: None,
    };
    let outcome = problem.run(&mut obj, context);
    let across = obj
        .requests
        .iter()
        .filter(|request| request.rank == KEPT_RANK_ACROSS)
        .count();
    let outcome_line = match &outcome {
        Ok(result) => format!(
            "Ok rho={:?} f-V0={:.9e} iterations={}",
            result.rho.to_vec(),
            result.final_value - STRATUM_OFFSET,
            result.iterations,
        ),
        Err(err) => format!("Err {err}"),
    };
    eprintln!(
        "[#2939 stratum] {context}: fixture={fixture:?} requests={} across={across} \
         outcome={outcome_line}",
        obj.requests.len(),
    );
    for (index, request) in obj.requests.iter().enumerate() {
        eprintln!(
            "[#2939 stratum] {context}: #{index} {:?} rho=({:.9}, {:.9}) rank={} f-V0={:.9e}",
            request.order,
            request.rho[0],
            request.rho[1],
            request.rank,
            request.cost - STRATUM_OFFSET,
        );
    }
    (outcome, obj.requests)
}

#[test]
fn a_search_pinned_at_its_rank_boundary_crosses_after_one_window_2939() {
    // Every rank-55 trial the rank-56 search proposes is higher than the rank-56 criterion
    // just inside the boundary, and the first ones are lower than the search's incumbent
    // at the time. On main at d2f73efe17 the seed escaped window after window, pinned
    // at the boundary with |Pg| growing from 3.6 to 8.7, and the fit refused after 116
    // requests (job 1201363).
    let fixture = StratumFixture {
        cutoff: 0.05,
        center0: 4.0,
        across_shift: 3.0,
    };
    let (outcome, requests) = run_stratum(fixture, "stratum wall above #2939");
    let result = outcome.expect("the rank-55 minimum certifies once the pinned rank-56 search crosses");
    assert!(
        result
            .criterion_certificate
            .as_ref()
            .is_some_and(|cert| cert.certifies()),
        "the fit must ship a certified point"
    );
    assert_eq!(
        fixture.rank(&result.rho),
        KEPT_RANK_ACROSS,
        "the certified point lies on the rank the search crossed to: {:?}",
        result.rho,
    );
    // A rank-55 trial is refused at value order, before any derivative is asked for, so
    // the first derivative request on rank 55 is the crossing's re-evaluation.
    let crossing = requests
        .iter()
        .position(|request| {
            request.rank == KEPT_RANK_ACROSS
                && matches!(request.order, OuterEvalOrder::ValueAndGradient)
        })
        .expect("the seed crosses to rank 55");
    let refused_before_crossing = requests[..crossing]
        .iter()
        .filter(|request| request.rank == KEPT_RANK_ACROSS)
        .count();
    assert_eq!(
        refused_before_crossing, COST_STALL_WINDOW,
        "the pinned rank-56 search must stop at its first window of refused trials, not \
         escape it"
    );
}

#[test]
fn a_rank_boundary_halt_publishes_typed_evidence_and_never_certifies_2939() {
    // Rank 55 is higher than rank 56 everywhere the search probes it, so the halted
    // incumbent is below every refused trial and nothing crosses.
    let fixture = StratumFixture {
        cutoff: 0.05,
        center0: 4.0,
        across_shift: 12.0,
    };
    let (outcome, requests) = run_stratum(fixture, "stratum wall far above #2939");
    assert!(
        requests
            .iter()
            .all(|request| request.rank == KEPT_RANK_INSIDE
                || matches!(request.order, OuterEvalOrder::Value)),
        "fixture premise: no seed crosses, so every rank-55 request is a refused value trial"
    );
    let error = outcome
        .expect_err("an incumbent pinned at its rank boundary outside the band must not certify");
    let message = error.to_string();
    assert!(
        message.contains(&format!(
            "rank_boundary=[kept_rank={KEPT_RANK_INSIDE}, refused_trials={COST_STALL_WINDOW}, band="
        )),
        "the refusal must carry the typed rank-boundary evidence: {message}"
    );
    assert!(
        message.contains("claimed_converged=false"),
        "a rank-boundary halt must never read as a converged claim: {message}"
    );
}

/// A guard holding a non-stationary incumbent, the state a search pinned against its
/// rank boundary reaches: its projected gradient is far outside the band.
fn pinned_guard() -> (CostStallGuard, Arc<Mutex<Option<CostStallExit>>>, Array1<f64>) {
    let exit: Arc<Mutex<Option<CostStallExit>>> = Arc::new(Mutex::new(None));
    let config = stratum_problem().config();
    let mut guard = CostStallGuard::new(1.0e-6, COST_STALL_WINDOW, &config, exit.clone());
    let incumbent = array![1.99965, 0.4549];
    let fixture = StratumFixture {
        cutoff: 0.05,
        center0: 4.0,
        across_shift: 3.0,
    };
    let eval = fixture.eval(&incumbent);
    let residual = eval.gradient.dot(&eval.gradient).sqrt();
    assert!(
        residual > guard.stationarity_band(eval.cost),
        "fixture premise: the pinned incumbent's |g| = {residual:.3e} must be outside the band \
         {:.3e}",
        guard.stationarity_band(eval.cost),
    );
    guard.observe_seed(&incumbent, eval.cost, residual);
    (guard, exit, incumbent)
}

#[test]
fn a_window_of_rank_refusals_halts_at_the_incumbent_without_an_escape_2939() {
    let (mut guard, exit, incumbent) = pinned_guard();
    let refused = array![2.011981138, 0.457077141];
    for trial in 1..COST_STALL_WINDOW {
        assert!(
            matches!(
                guard.observe_off_stratum(&refused, KEPT_RANK_INSIDE),
                CostStallVerdict::Continue
            ),
            "trial {trial} of a {COST_STALL_WINDOW}-trial window must not decide the stall"
        );
    }
    let verdict = guard.observe_off_stratum(&refused, KEPT_RANK_INSIDE);
    assert!(
        matches!(verdict, CostStallVerdict::FlatValleyStall { .. }),
        "a window whose every trial left the run's kept rank must halt at the incumbent, not \
         reopen the window: got {:?}",
        std::mem::discriminant(&verdict),
    );
    assert_eq!(guard.stuck_escapes, 0, "a rank-refusal window grants no escape");
    let published = exit
        .lock()
        .expect("exit cell")
        .clone()
        .expect("the halt publishes the incumbent");
    assert_eq!(published.rho, incumbent, "the halt publishes the incumbent it was pinned at");
    assert!(!published.converged, "a pinned incumbent outside the band is not converged");
    let evidence = published
        .rank_boundary
        .expect("a rank-boundary halt publishes its typed evidence");
    assert_eq!(
        evidence.kept_rank, KEPT_RANK_INSIDE,
        "the evidence names the rank the search searched"
    );
    assert_eq!(
        evidence.refused_trials, COST_STALL_WINDOW,
        "every trial in the filled window was a rank refusal"
    );
    assert!(
        published.grad_norm > evidence.band,
        "the incumbent's |Pg| = {:.3e} must be outside the reported band {:.3e}",
        published.grad_norm,
        evidence.band,
    );
}

#[test]
fn a_window_holding_an_infeasible_probe_keeps_the_1426_escape_2939() {
    let (mut guard, exit, incumbent) = pinned_guard();
    let refused = array![2.011981138, 0.457077141];
    assert!(matches!(
        guard.observe_infeasible(&refused),
        CostStallVerdict::Continue
    ));
    for trial in 2..COST_STALL_WINDOW {
        assert!(
            matches!(
                guard.observe_off_stratum(&refused, KEPT_RANK_INSIDE),
                CostStallVerdict::Continue
            ),
            "trial {trial} of a {COST_STALL_WINDOW}-trial window must not decide the stall"
        );
    }
    let verdict = guard.observe_off_stratum(&refused, KEPT_RANK_INSIDE);
    assert!(
        matches!(verdict, CostStallVerdict::StuckKeepDescending { .. }),
        "a window that also holds a probe that failed to evaluate keeps the #1426 escape: got \
         {:?}",
        std::mem::discriminant(&verdict),
    );
    assert_eq!(guard.stuck_escapes, 1, "the mixed window is granted its escape");
    assert!(
        exit.lock().expect("exit cell").as_ref().is_none_or(|published| published.rho
            == incumbent
            && published.rank_boundary.is_none()),
        "an escape publishes no other point and no rank-boundary evidence"
    );
}
