// Child module of `run_plan::run_plan_tests` (see the `#[path]` declaration
// there): the dense-ARC route's adjudication of a deferred cost stall by the
// certificate's own second-order test (#2817), and the trajectory census
// (#2735). Scope comes from the parent via `use super::*`.
//
// The defect these pin. `opt::Arc` stops on an absolute projected-gradient
// band; the certificate that judges the result accepts on the Newton decrement
// `½·gᵀH⁻¹g` against the criterion's resolution. Those are two standards in two
// different units, and on a flat REML valley the band is the far stricter one —
// measured 29× apart on a gaussian n=50 000, p=93, K=11 fit, where all six runs
// burned their 200-iteration budget for a last-100 improvement of `4e-4` in a
// criterion of `5.3e4`. The cost-stall guard already sees the flatline, and on
// this route it DEFERS, "because only ARC owns the synchronized reduced-Hessian
// certificate" — and then nothing ever ran that certificate. This is the
// adjudication the deferral was waiting for.

use super::*;
use ndarray::array;

/// The criterion resolution floor these fixtures hand the bridge — the same
/// quantity `outer_rel_cost_floor` returns at the default outer tolerance
/// (`COST_STALL_REL_TOL_FLOOR`), written out so a fixture states the threshold
/// it brackets instead of importing it and asserting a tautology.
const FLOOR_2817: f64 = 1.0e-7;

/// The criterion value the flat fixtures sit at. A REML score of `O(1e3)` is
/// the ordinary scale of the fits this defect was measured on, and it makes the
/// resolution `FLOOR_2817·(1 + |V|)` a number worth writing down: `1.001e-4`.
const COST_2817: f64 = 1.0e3;

/// The resolution the adjudication is judged against — the certificate's own
/// `objective_tol` at [`COST_2817`]: `1.001e-4`.
const RESOLUTION_2817: f64 = FLOOR_2817 * (1.0 + COST_2817);

/// The residual the paired fixtures sit at.
///
/// Chosen to satisfy three things at once, which is what makes the pair sharp:
/// its Newton decrement `½‖g‖²/(1 + √ε) = 8.45e-5` is INSIDE the criterion's
/// resolution `1.001e-4` (so the certificate accepts the point); it is three
/// orders ABOVE the default absolute outer band `1e-5` (so the solver's own
/// stopping test never reaches it, which is the whole defect); and it is above
/// [`COST_STALL_PROJECTED_GRAD_FLOOR`] (so the guard does not read the point as
/// KKT-stationary-at-bound, which would fill its window from the gradient
/// rather than from the criterion and make the descending control vacuous).
const STOP_GRAD_2817: f64 = 1.3e-2;

/// The wide box the interior fixtures use: no coordinate is rail-adjacent, so
/// the projection is the identity and the fixture is about curvature alone.
fn wide_box_2817(dim: usize) -> (Array1<f64>, Array1<f64>) {
    (Array1::from_elem(dim, -30.0), Array1::from_elem(dim, 30.0))
}

/// Drive `samples.len()` ARC oracle evaluations at `point`, one per entry, and
/// report each evaluation's outcome plus whatever reached the shared exit cell.
///
/// The schedule is the whole fixture: a CONSTANT cost is a criterion that has
/// stopped moving (the guard's window fills and the stall is adjudicated), a
/// DECREASING one is a search still making progress (the window never fills and
/// nothing is adjudicated). Evaluation stops at the first error.
fn drive_arc_oracle_2817(
    point: Array1<f64>,
    samples: Vec<(f64, Array1<f64>)>,
    hessian: Array2<f64>,
    bounds: (Array1<f64>, Array1<f64>),
    floor: Option<f64>,
) -> (Vec<Result<f64, String>>, Option<CostStallExit>) {
    let table = Arc::new(samples.clone());
    let calls = Arc::new(AtomicUsize::new(0));
    let problem = OuterProblem::new(point.len())
        .with_gradient(Derivative::Analytic)
        .with_hessian(DeclaredHessianForm::Either);
    let scripted_hessian = hessian.clone();
    let cost_table = Arc::clone(&table);
    let mut obj = problem.build_objective_with_eval_order(
        (),
        move |_: &mut (), _: &Array1<f64>| Ok(cost_table[0].0),
        |_: &mut (), _: &Array1<f64>| {
            Err(EstimationError::InvalidInput(
                "legacy eager eval should not run".to_string(),
            ))
        },
        move |_: &mut (), _: &Array1<f64>, order: OuterEvalOrder| {
            let idx = calls.fetch_add(1, Ordering::Relaxed);
            let (cost, gradient) = table[idx.min(table.len() - 1)].clone();
            Ok(OuterEval {
                cost,
                gradient,
                hessian: match order {
                    OuterEvalOrder::ValueGradientHessian => {
                        HessianValue::Dense(scripted_hessian.clone())
                    }
                    _ => HessianValue::Unavailable,
                },
                inner_beta_hint: None,
            })
        },
        None::<fn(&mut ())>,
        None::<fn(&mut (), &Array1<f64>) -> Result<EfsEval, EstimationError>>,
    );
    let exit: Arc<Mutex<Option<CostStallExit>>> = Arc::new(Mutex::new(None));
    let guard = CostStallGuard::new(
        FLOOR_2817,
        ARC_COST_STALL_WINDOW,
        COST_STALL_PROJECTED_GRAD_FLOOR,
        exit.clone(),
    );
    let mut bridge = OuterSecondOrderBridge {
        obj: &mut obj,
        layout: OuterThetaLayout::new(point.len(), 0),
        hessian_source: HessianSource::Analytic,
        materialize_operator_max_dim: OUTER_HVP_MATERIALIZE_MAX_DIM,
        eval_count: 0,
        outer_inner_cap: None,
        g_norm_initial: None,
        last_g_norm: None,
        last_value_grad_rho: None,
        cost_stall: Some(guard),
        cost_stall_bounds: Some(bounds),
        curvature_stationary_floor: floor,
    };
    let mut outcomes = Vec::new();
    for _ in 0..samples.len() {
        match SecondOrderObjective::eval_hessian(&mut bridge, &point) {
            Ok(sample) => outcomes.push(Ok(sample.value)),
            Err(err) => {
                outcomes.push(Err(err.into_message()));
                break;
            }
        }
    }
    let published = exit.lock().expect("exit cell").take();
    (outcomes, published)
}

/// A criterion that has stopped moving, held at the same value for a full
/// window.
fn flatlined_2817(gradient: Array1<f64>, count: usize) -> Vec<(f64, Array1<f64>)> {
    (0..count).map(|_| (COST_2817, gradient.clone())).collect()
}

/// A criterion still buying real decrease every step: each entry improves on
/// the last by far more than the guard's relative floor.
fn descending_2817(gradient: Array1<f64>, count: usize) -> Vec<(f64, Array1<f64>)> {
    (0..count)
        .map(|i| (COST_2817 - (i as f64), gradient.clone()))
        .collect()
}

/// The adjudication fires, and the point it stops at carries a gradient two
/// orders ABOVE the absolute band the solver was being driven to.
///
/// [`STOP_GRAD_2817`] against unit curvature is a Newton decrement of `8.45e-5`,
/// inside the criterion's resolution of `1.001e-4`: no step from here can move
/// the criterion by as much as it can be resolved. The default absolute outer
/// band is `1e-5`, so before this repair the search kept going — toward a
/// threshold the certificate it was heading for never applies.
#[test]
fn a_flatlined_arc_stall_is_adjudicated_by_the_certificates_own_test_2817() {
    let (outcomes, published) = drive_arc_oracle_2817(
        array![0.5],
        flatlined_2817(array![STOP_GRAD_2817], ARC_COST_STALL_WINDOW + 3),
        array![[1.0]],
        wide_box_2817(1),
        Some(FLOOR_2817),
    );
    let message = outcomes
        .last()
        .expect("the schedule runs at least once")
        .clone()
        .expect_err("a flatlined stall the certificate accepts must halt ARC");
    assert_eq!(
        message, ARC_CURVATURE_STATIONARY_SENTINEL,
        "the halt must be the curvature-stationary sentinel, not an objective failure"
    );
    assert!(
        outcomes.len() >= ARC_COST_STALL_WINDOW,
        "the halt must wait for the guard's window to fill, not fire on the first \
         evaluation: it took {} evaluations",
        outcomes.len()
    );
    let published = published.expect("the halt must publish the point it stopped at");
    assert!(
        published.converged,
        "a point whose decrement is inside the criterion's resolution under a PSD \
         reduced Hessian is a convergence, not a checkpoint"
    );
    assert_eq!(published.rho, array![0.5]);
    assert_eq!(published.value, COST_2817);
    assert!(
        (published.grad_norm - STOP_GRAD_2817).abs() < 1.0e-15,
        "the published residual is the rail-projected gradient norm at the stop: {}",
        published.grad_norm
    );
    assert!(
        published.grad_norm > 1.0e-5,
        "the fixture is only meaningful while the stop sits ABOVE the absolute \
         gradient band: |Pg|={} vs 1e-5",
        published.grad_norm
    );
}

/// NEGATIVE CONTROL ON THE GATE, and it is the SAME fixture as the one above
/// with one thing changed. Same point, same curvature, same gradient, same
/// decrement — and a criterion still buying a whole unit of decrease per step
/// instead of standing still. Nothing is adjudicated, because nothing stalled.
///
/// This is what keeps the rung from becoming a first-choice stop: a search
/// making real progress is left alone even when its local decrement is already
/// inside the criterion's resolution.
#[test]
fn a_still_descending_search_is_never_adjudicated_2817() {
    let (outcomes, published) = drive_arc_oracle_2817(
        array![0.5],
        descending_2817(array![STOP_GRAD_2817], ARC_COST_STALL_WINDOW + 3),
        array![[1.0]],
        wide_box_2817(1),
        Some(FLOOR_2817),
    );
    assert!(
        outcomes.iter().all(|o| o.is_ok()),
        "a descending search must never be halted: {outcomes:?}"
    );
    assert!(
        published.is_none_or(|exit| !exit.converged),
        "a descending search must never be published as converged"
    );
}

/// NEGATIVE CONTROL ON THE TEST ITSELF. A flatlined criterion whose residual
/// carries real available descent is not adjudicated converged.
///
/// `|g| = 1.0` against unit curvature is a decrement of `0.5`, `5000×` the
/// criterion's resolution: a Newton step from here really does buy that much,
/// so the stall is a stall and not an optimum.
#[test]
fn a_stall_with_real_available_descent_is_not_certified_2817() {
    let (outcomes, published) = drive_arc_oracle_2817(
        array![0.5],
        flatlined_2817(array![1.0], ARC_COST_STALL_WINDOW + 3),
        array![[1.0]],
        wide_box_2817(1),
        Some(FLOOR_2817),
    );
    assert!(
        outcomes.iter().all(|o| o.is_ok()),
        "a stall whose Newton step still buys 5000x the criterion's resolution \
         must be handed back to ARC: {outcomes:?}"
    );
    assert!(
        published.is_none_or(|exit| !exit.converged),
        "such a stall must never be published as converged"
    );
}

/// The bracket: the adjudication is decided by the criterion's resolution and
/// by nothing else.
///
/// With unit curvature the decrement is `½‖g‖²/(1 + √ε)`, so the gradient at
/// which the rung flips is `‖g‖* = √(2·resolution·(1 + √ε))`. Straddling it by
/// 1% must flip the verdict — which is what makes "the tolerance is the
/// certificate's" a measurement rather than a comment.
#[test]
fn the_adjudication_threshold_is_the_criterion_resolution_2817() {
    let shift = f64::EPSILON.sqrt();
    let critical = (2.0 * RESOLUTION_2817 * (1.0 + shift)).sqrt();
    let (inside, _) = drive_arc_oracle_2817(
        array![0.5],
        flatlined_2817(array![critical * 0.99], ARC_COST_STALL_WINDOW + 3),
        array![[1.0]],
        wide_box_2817(1),
        Some(FLOOR_2817),
    );
    assert_eq!(
        inside.last().expect("ran").clone().err().as_deref(),
        Some(ARC_CURVATURE_STATIONARY_SENTINEL),
        "a decrement 1% inside the criterion's resolution must end the stall"
    );
    let (outside, _) = drive_arc_oracle_2817(
        array![0.5],
        flatlined_2817(array![critical * 1.01], ARC_COST_STALL_WINDOW + 3),
        array![[1.0]],
        wide_box_2817(1),
        Some(FLOOR_2817),
    );
    assert!(
        outside.iter().all(|o| o.is_ok()),
        "a decrement 1% outside the criterion's resolution must not end the stall"
    );
}

/// A STRICT SADDLE IS NEVER A STOP. This is the failure a bare gradient band
/// admits and the decrement rung refuses.
///
/// `‖g‖ = 1e-6` clears any gradient band ever configured on this route, and a
/// search halted there ships a point with a descent direction still available —
/// the #2748 stage-1 signature (`hessian_psd=NO` at a point the band had
/// already accepted). Both halves of the rung reject it independently: the
/// reduced Hessian is indefinite, and the shifted Cholesky behind
/// `newton_predicted_decrease` has no positive factor to build a decrement out
/// of.
#[test]
fn a_strict_saddle_is_never_adjudicated_stationary_2817() {
    let hessian = array![[1.0, 0.0], [0.0, -1.0]];
    let gradient = array![1.0e-6, 1.0e-6];
    assert!(
        newton_predicted_decrease(&hessian, &gradient).is_none(),
        "an indefinite Hessian yields no decrement at all"
    );
    let (lower, upper) = wide_box_2817(2);
    assert_eq!(
        reduced_hessian_psd_at_point(&array![0.5, 0.5], &gradient, &hessian, Some((&lower, &upper))),
        Some(false),
        "the reduced-Hessian gate must see the negative eigenvalue"
    );
    let (outcomes, published) = drive_arc_oracle_2817(
        array![0.5, 0.5],
        flatlined_2817(gradient, ARC_COST_STALL_WINDOW + 3),
        hessian,
        wide_box_2817(2),
        Some(FLOOR_2817),
    );
    assert!(
        outcomes.iter().all(|o| o.is_ok()),
        "a strict saddle must reach ARC so it can exploit the negative curvature, \
         never be converted into a convergence: {outcomes:?}"
    );
    assert!(
        published.is_none_or(|exit| !exit.converged),
        "a strict saddle must never be published as converged"
    );
}

/// A residual lying along a NEAR-FLAT direction is real descent, and the rung
/// refuses it — which a gradient threshold cannot do.
///
/// `‖g‖ = 1e-5` sits exactly at the default absolute band, so a gradient-only
/// test would stop here. The curvature in that direction is `1e-9`, so the
/// decrement is `≈ 3.1e-3`, thirty times the criterion's resolution, and a
/// Newton step really does buy that much.
#[test]
fn a_residual_along_a_flat_direction_keeps_the_search_moving_2817() {
    let hessian = array![[1.0, 0.0], [0.0, 1.0e-9]];
    let gradient = array![0.0, 1.0e-5];
    let decrement =
        newton_predicted_decrease(&hessian, &gradient).expect("a PSD Hessian yields a decrement");
    assert!(
        decrement > RESOLUTION_2817,
        "the fixture requires the flat direction to inflate the decrement past the \
         resolution: {decrement:.3e} vs {RESOLUTION_2817:.3e}"
    );
    let (outcomes, _) = drive_arc_oracle_2817(
        array![0.5, 0.5],
        flatlined_2817(gradient, ARC_COST_STALL_WINDOW + 3),
        hessian,
        wide_box_2817(2),
        Some(FLOOR_2817),
    );
    assert!(
        outcomes.iter().all(|o| o.is_ok()),
        "a residual whose Newton step is worth more than the criterion's resolution \
         must keep the search running even at |g| = 1e-5: {outcomes:?}"
    );
}

/// A coordinate pinned at its bound with an OUTWARD pull is KKT-stationary, and
/// the adjudication reads the projected residual exactly as the certificate
/// does.
///
/// The raw gradient is `+1` forever — no band it is measured against can ever
/// be cleared — but its descent step exits the box, so the projection zeroes it
/// and the decrement is zero.
#[test]
fn a_bound_pinned_outward_pull_is_adjudicated_stationary_2817() {
    let (outcomes, published) = drive_arc_oracle_2817(
        array![-30.0],
        flatlined_2817(array![1.0], ARC_COST_STALL_WINDOW + 3),
        array![[1.0]],
        wide_box_2817(1),
        Some(FLOOR_2817),
    );
    assert_eq!(
        outcomes.last().expect("ran").clone().err().as_deref(),
        Some(ARC_CURVATURE_STATIONARY_SENTINEL),
        "a bound-pinned outward pull has no feasible descent and must end the stall"
    );
    let published = published.expect("the halt publishes its point");
    assert!(published.converged);
    assert_eq!(
        published.grad_norm, 0.0,
        "the published residual is the PROJECTED gradient, which is zero here"
    );
}

/// POSITIVE CONTROL ON THE WIRING. The floor is what performs the
/// adjudication: the same stall on a route that declares no criterion
/// resolution behaves exactly as it did before this change.
#[test]
fn a_route_that_declares_no_resolution_is_unchanged_2817() {
    let (outcomes, published) = drive_arc_oracle_2817(
        array![0.5],
        flatlined_2817(array![STOP_GRAD_2817], ARC_COST_STALL_WINDOW + 3),
        array![[1.0]],
        wide_box_2817(1),
        None,
    );
    assert!(
        outcomes.iter().all(|o| o.is_ok()),
        "with no declared resolution every sample reaches the solver: {outcomes:?}"
    );
    assert!(published.is_none_or(|exit| !exit.converged));
}

// ─── the trajectory census (#2735) ───────────────────────────────────────────

fn step_2817(iter: usize, step_norm: f64, radius: f64, actual: f64) -> StepInfo {
    StepInfo {
        iter,
        step_norm,
        predicted_decrease: actual,
        actual_decrease: actual,
        trust_radius: Some(radius),
    }
}

/// A CRAWL and a THRASH end the same way — on the iteration budget — and the
/// two numbers a budget-exhausted run reports (`final_value`, `‖g‖`) cannot
/// tell them apart. The census can, and this is the fixture that says so.
#[test]
fn the_census_separates_a_crawl_from_a_thrash_2735() {
    let crawl = OuterStepCensus::default();
    for iter in 0..40 {
        // Every step accepted, every one pinned to a radius that never grows.
        crawl.observe(&step_2817(iter, 0.125, 0.125, 0.2), true);
    }
    let crawl_line = crawl.describe().expect("40 observed steps describe");
    assert!(
        crawl_line.contains("accepted=40")
            && crawl_line.contains("rejected=0")
            && crawl_line.contains("boundary_limited=40/40"),
        "a crawl must read as all-accepted and boundary-limited: {crawl_line}"
    );
    assert!(
        crawl_line.contains("radius=[1.250e-1, 1.250e-1]"),
        "a crawl's radius never moves: {crawl_line}"
    );

    let thrash = OuterStepCensus::default();
    let mut radius = 1.0;
    for iter in 0..40 {
        // Three rejections per acceptance, the radius quartering each time.
        for _ in 0..3 {
            thrash.observe(&step_2817(iter, radius, radius, f64::NAN), false);
            radius *= 0.25;
        }
        thrash.observe(&step_2817(iter, 0.5 * radius, radius, 1.0e-6), true);
    }
    let thrash_line = thrash.describe().expect("observed steps describe");
    assert!(
        thrash_line.contains("accepted=40") && thrash_line.contains("rejected=120"),
        "a thrash must report its rejections: {thrash_line}"
    );
    assert!(
        thrash_line.contains("boundary_limited=0/40"),
        "a thrashing walk's accepted steps are interior to the collapsed region: {thrash_line}"
    );
    assert_ne!(
        crawl_line, thrash_line,
        "the whole point of the census is that these two are different lines"
    );
}

/// A run that took no step describes nothing, so the summary says "no step
/// observed" rather than printing a row of zeros that looks like a measurement.
#[test]
fn a_census_with_no_observed_step_describes_nothing_2735() {
    assert!(OuterStepCensus::default().describe().is_none());
}
