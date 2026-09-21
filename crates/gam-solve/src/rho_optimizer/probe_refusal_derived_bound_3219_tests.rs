// Child module of `run_plan::run_plan_tests` (see the `#[path]` declaration
// there): a first-order run whose line-search probes are refused ends on the
// refused step's own linear model, not on a count of refusals (#3219). Scope
// comes from the parent via `use super::*`.
//
// The defect these pin. The bridge aborted a seed as fatal after 150
// consecutive refused cost probes with no accepted gradient step, or after 25
// when the seed came through `with_initial_sample`. Neither count is something
// the search knows. The seed's gradient and resolution already decide the run:
// a refused probe at `x` stalls iff its model decrease `p = −g_bᵀ(x − x_b)` is at
// most the seed's resolution `R_b` (#3018). A backtracking search that shrinks
// `α` by `β` along `d` therefore stalls first at probe index
// `k* = ⌈log_{1/β}(α₀·|g_bᵀd| / R_b)⌉`.

use super::*;
use ndarray::array;

/// The seed's value. The line search moves along `d = −g = +1` from `ρ = 0`.
const V0_3219: f64 = 10.0;

/// The seed's resolution, `R_b = 2⁻⁴⁰`. With `α₀ = 1`, `β = ½` and `|g_bᵀd| = 1`,
/// the first stalled probe is at `k* = 40`, past both of the old counts' 25.
const LOG2_ALPHA0_SLOPE_OVER_RESOLUTION_3219: i32 = 40;

fn resolution_3219() -> f64 {
    2.0_f64.powi(-LOG2_ALPHA0_SLOPE_OVER_RESOLUTION_3219)
}

/// The backtracking contraction factor `β`.
const BETA_3219: f64 = 0.5;

/// Probe index `k` of the backtracking search from `α₀ = 1`.
fn probe_3219(k: i32) -> f64 {
    BETA_3219.powi(k)
}

/// The derived bound `k* = ⌈log_{1/β}(α₀·|g_bᵀd| / R_b)⌉`: the index of the
/// first refused probe whose model decrease is within the seed's resolution.
fn derived_first_stall_3219() -> usize {
    let alpha0_slope = 1.0;
    ((alpha0_slope / resolution_3219()).ln() / (1.0 / BETA_3219).ln()).ceil() as usize
}

/// Drive the first-order bridge's cost probes along the backtracking sequence
/// from a seed installed as its incumbent, the way the BFGS route installs it:
/// no gradient evaluated by the bridge itself (`last_value_grad_rho = None`) and
/// no accepted step. `feasible_from` is the first probe index whose value lane
/// evaluates; every earlier probe is refused. Returns each probe's outcome,
/// stopping at the first fatal one, and what the guard published.
fn drive_refused_probes_3219(
    feasible_from: Option<i32>,
    probes: i32,
) -> (Vec<Result<f64, ObjectiveEvalError>>, Option<CostStallExit>) {
    let first_feasible = feasible_from.map_or(f64::NEG_INFINITY, probe_3219);
    let problem = OuterProblem::new(1).with_gradient(Derivative::Analytic);
    let mut obj = problem.build_objective_with_eval_order(
        (),
        |_: &mut (), _: &Array1<f64>| Ok(V0_3219),
        |_: &mut (), _: &Array1<f64>| {
            Err(EstimationError::InvalidInput(
                "legacy eager eval should not run".to_string(),
            ))
        },
        move |_: &mut (), theta: &Array1<f64>, _: OuterEvalOrder| {
            if theta[0] <= first_feasible {
                Ok(OuterEval {
                    cost: V0_3219 - theta[0],
                    gradient: array![-1.0],
                    hessian: HessianValue::Unavailable,
                    inner_beta_hint: None,
                })
            } else {
                Err(EstimationError::TrialPointRefused {
                    reason: "planted gam#3219 infeasible neighbourhood".to_string(),
                })
            }
        },
        None::<fn(&mut ())>,
        None::<fn(&mut (), &Array1<f64>) -> Result<EfsEval, EstimationError>>,
    );
    let config = claim_band_config(1.0e-3);
    let exit: Arc<Mutex<Option<CostStallExit>>> = Arc::new(Mutex::new(None));
    let mut guard = CostStallGuard::new(&config, exit.clone());
    guard.observe_seed(&array![0.0], V0_3219, resolution_3219(), 1.0);
    let mut bridge = OuterFirstOrderBridge {
        obj: &mut obj,
        layout: OuterThetaLayout::new(1, 0),
        outer_inner_cap: None,
        first_order_evals: 0,
        g_norm_initial: None,
        last_g_norm: None,
        last_value_grad_rho: None,
        value_probe_cache: Vec::new(),
        cost_stall: Some(guard),
        cost_stall_bounds: Some((array![-30.0], array![30.0])),
        accepted_steps: Arc::default(),
        pending_first_order: Vec::new(),
        incumbent: Some(OuterIncumbent {
            rho: array![0.0],
            cost: V0_3219,
            gradient: array![-1.0],
        }),
        stratum_rank: None,
        stratum_probe: None,
    };
    let mut outcomes = Vec::new();
    for k in 0..probes {
        let outcome = ZerothOrderObjective::eval_cost(&mut bridge, &array![probe_3219(k)]);
        let fatal = matches!(&outcome, Err(err) if !err.is_recoverable());
        outcomes.push(outcome);
        if fatal {
            break;
        }
    }
    let published = exit.lock().expect("exit cell").take();
    (outcomes, published)
}

/// A seed whose every probe is refused stops on the derived rule. The first
/// stalled probe, at `k*`, finds the seed non-stationary (`|g| = 1`) and is
/// granted its first escape. The next stalls again with no descent bought in
/// between, so the progress licence stops the run there (#2817), at the seed,
/// unconverged, for the certificate and the reseed ladder to judge. Before #3219
/// the count of 25 aborted the seed as fatal after probe 24, long before its
/// model decrease reached its resolution.
#[test]
fn a_seed_whose_every_probe_is_refused_stops_within_the_derived_bound_3219() {
    let k_star = derived_first_stall_3219();
    assert_eq!(k_star, LOG2_ALPHA0_SLOPE_OVER_RESOLUTION_3219 as usize);
    let (outcomes, published) = drive_refused_probes_3219(None, 4 * k_star as i32);

    let stop = outcomes.len() - 1;
    assert_eq!(
        stop,
        k_star + 1,
        "the run must stop at the probe after the first stall k* = {k_star}, got {} probes: \
         last = {:?}",
        outcomes.len(),
        outcomes.last(),
    );
    assert!(
        outcomes[..stop]
            .iter()
            .all(|outcome| matches!(outcome, Err(err) if err.is_recoverable())),
        "every probe before the stop is a recoverable refusal the line search shortens"
    );
    match &outcomes[stop] {
        Err(err) => assert_eq!(
            (err.is_fatal(), err.message()),
            (true, COST_STALL_CONVERGED_SENTINEL),
            "the run ends through the cost-stall guard, not a refusal count"
        ),
        Ok(cost) => panic!("the stopping probe was refused, got a value {cost}"),
    }
    let published = published.expect("the guard publishes the seed it stopped at");
    assert_eq!(published.rho, array![0.0], "{published:?}");
    assert_eq!(published.value, V0_3219, "{published:?}");
    assert!(
        !published.converged,
        "a non-stationary seed is never claimed converged: {published:?}"
    );
    // Probes k* and k* + 1 are both refused within the seed's resolution, so the
    // stop carries two proofs that the domain ends within an unresolvable step of
    // the seed. The escape granted between them does not clear the first (#3400).
    assert_eq!(published.wall_refusals, 2, "{published:?}");
}

/// A seed whose probes recover before the derived bound is not aborted: the
/// refusals before it reach no verdict, however many there are, and the first
/// feasible probe is handed to the line search. Thirty refusals is past the old
/// count of 25 and short of `k* = 40`.
#[test]
fn a_seed_whose_probes_recover_before_the_derived_bound_is_not_aborted_3219() {
    let recover_at = 30;
    assert!((recover_at as usize) < derived_first_stall_3219());
    let (outcomes, published) = drive_refused_probes_3219(Some(recover_at), recover_at + 1);

    assert_eq!(outcomes.len(), recover_at as usize + 1, "{:?}", outcomes.last());
    assert!(
        outcomes[..recover_at as usize]
            .iter()
            .all(|outcome| matches!(outcome, Err(err) if err.is_recoverable())),
        "the refusals before the recovery stay recoverable: none aborts the seed"
    );
    let recovered = match &outcomes[recover_at as usize] {
        Ok(cost) => *cost,
        Err(err) => panic!("the first feasible probe must reach the line search, got {err}"),
    };
    assert_eq!(recovered, V0_3219 - probe_3219(recover_at));
    let published = published.expect("the seed is published up front");
    assert_eq!(published.rho, array![0.0], "a refusal displaced the seed: {published:?}");
    // Every refusal before the recovery was a step whose model promised more than
    // the seed's resolution, so none is evidence of a domain wall (#3400).
    assert_eq!(published.wall_refusals, 0, "{published:?}");
}
