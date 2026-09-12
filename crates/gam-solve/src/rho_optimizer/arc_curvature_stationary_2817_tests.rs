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
/// nothing is adjudicated). Evaluation stops at the first error. Value probes
/// answer with the schedule's first cost: a criterion that is flat everywhere.
fn drive_arc_oracle_2817(
    point: Array1<f64>,
    samples: Vec<(f64, Array1<f64>)>,
    hessian: Array2<f64>,
    bounds: (Array1<f64>, Array1<f64>),
    floor: Option<f64>,
) -> (Vec<Result<f64, String>>, Option<CostStallExit>) {
    let flat = samples[0].0;
    drive_arc_oracle_valued_2817(point, samples, hessian, bounds, floor, move |_| flat)
}

/// [`drive_arc_oracle_2817`] with the criterion's value probe supplied. The
/// strict-saddle adjudication (#1082) steps along the reported negative
/// eigenvector through `eval_cost`, so `value` decides whether the criterion
/// confirms that claim or contradicts it.
fn drive_arc_oracle_valued_2817(
    point: Array1<f64>,
    samples: Vec<(f64, Array1<f64>)>,
    hessian: Array2<f64>,
    bounds: (Array1<f64>, Array1<f64>),
    floor: Option<f64>,
    mut value: impl FnMut(&Array1<f64>) -> f64,
) -> (Vec<Result<f64, String>>, Option<CostStallExit>) {
    let table = Arc::new(samples.clone());
    let calls = Arc::new(AtomicUsize::new(0));
    let problem = OuterProblem::new(point.len())
        .with_gradient(Derivative::Analytic)
        .with_hessian(DeclaredHessianForm::Either);
    let scripted_hessian = hessian.clone();
    let mut obj = problem.build_objective_with_eval_order(
        (),
        move |_: &mut (), theta: &Array1<f64>| Ok(value(theta)),
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

/// The quadratic model a scripted sample claims, as the criterion itself:
/// `V(θ) = COST_2817 + g·d + ½·dᵀHd` with `d = θ − center`. Along a negative
/// eigenvector of `H` it really falls, so the strict-saddle adjudication (#1082)
/// confirms the claim instead of contradicting it.
fn quadratic_criterion_1082(
    theta: &Array1<f64>,
    center: &Array1<f64>,
    gradient: &Array1<f64>,
    hessian: &Array2<f64>,
) -> f64 {
    let d = theta - center;
    COST_2817 + gradient.dot(&d) + 0.5 * d.dot(&hessian.dot(&d))
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
///
/// The criterion really has the saddle: along the negative eigenvector it falls
/// by `½α²`, so the strict-saddle adjudication (#1082) confirms the descent and
/// the escape stands. A criterion that does not move along it contradicts the
/// claim instead, which is the stop
/// `a_strict_saddle_claim_the_criterion_contradicts_stops_at_the_incumbent_1082`
/// pins.
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
        reduced_hessian_psd_at_point(&array![0.5, 0.5], &gradient, &hessian, Some((&lower, &upper)), 0.0),
        Some(false),
        "the reduced-Hessian gate must see the negative eigenvalue"
    );
    let center = array![0.5, 0.5];
    let (claimed_gradient, claimed_hessian) = (gradient.clone(), hessian.clone());
    let (outcomes, published) = drive_arc_oracle_valued_2817(
        center.clone(),
        flatlined_2817(gradient, ARC_COST_STALL_WINDOW + 3),
        hessian,
        wide_box_2817(2),
        Some(FLOOR_2817),
        move |theta| quadratic_criterion_1082(theta, &center, &claimed_gradient, &claimed_hessian),
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

/// #1082 — A STRICT-SADDLE CLAIM THE CRITERION CONTRADICTS IS A STOP, at the
/// incumbent, exactly as the terminal certificate accepts it (#2612).
///
/// Same point, gradient and Hessian as the saddle fixture above; only the
/// criterion changes. It does not move along the reported negative eigenvector,
/// so no trial of the adjudication ladder lowers it, the claim is
/// `Contradicted`, and with `|Pg| = 1.4e-6` inside the solver band
/// ([`COST_STALL_PROJECTED_GRAD_FLOOR`] here) the bridge stops ARC where the
/// certificate would accept.
#[test]
fn a_strict_saddle_claim_the_criterion_contradicts_stops_at_the_incumbent_1082() {
    let (outcomes, published) = drive_arc_oracle_valued_2817(
        array![0.5, 0.5],
        flatlined_2817(array![1.0e-6, 1.0e-6], ARC_COST_STALL_WINDOW + 3),
        array![[1.0, 0.0], [0.0, -1.0]],
        wide_box_2817(2),
        Some(FLOOR_2817),
        |_| COST_2817,
    );
    assert_eq!(
        outcomes.last().expect("ran").clone().err().as_deref(),
        Some(ARC_CURVATURE_STATIONARY_SENTINEL),
        "a strict-saddle claim the criterion contradicts, inside the solver band, must end \
         the stall: {outcomes:?}"
    );
    assert!(
        outcomes.len() >= ARC_COST_STALL_WINDOW,
        "the adjudication waits for the guard's window to fill: it took {} evaluations",
        outcomes.len()
    );
    let published = published.expect("the stop publishes the incumbent");
    assert!(published.converged);
    assert_eq!(published.rho, array![0.5, 0.5]);
    assert_eq!(published.value, COST_2817);
}

/// NEGATIVE CONTROL ON THE BAND. The same contradicted claim at a residual ABOVE
/// the solver band is not a stop: that band is the rung the stop answers to, so
/// the escape stands and ARC keeps every sample.
#[test]
fn a_contradicted_strict_saddle_outside_the_solver_band_keeps_the_search_moving_1082() {
    let (outcomes, published) = drive_arc_oracle_valued_2817(
        array![0.5, 0.5],
        flatlined_2817(array![5.0e-3, 5.0e-3], ARC_COST_STALL_WINDOW + 3),
        array![[1.0, 0.0], [0.0, -1.0]],
        wide_box_2817(2),
        Some(FLOOR_2817),
        |_| COST_2817,
    );
    assert!(
        outcomes.iter().all(|o| o.is_ok()),
        "a residual above the solver band must reach ARC whatever the adjudication says: \
         {outcomes:?}"
    );
    assert!(
        published.is_none_or(|exit| !exit.converged),
        "such a stall must never be published as converged"
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

/// The decrement is taken at the resolution its definiteness verdict was taken at.
///
/// A Hessian that factors at the arithmetic shift keeps its historical decrement
/// at any resolution, so a near-flat positive direction still reads as descent. A
/// negative eigenvalue below the criterion's curvature resolution has no factor at
/// the arithmetic shift and a decrement at the resolution shift. A resolvable
/// saddle factors at neither, and a zero resolution is the historical function.
#[test]
fn the_decrement_travels_with_the_resolution_its_verdict_was_taken_at_2817() {
    let resolution = 2.0 * RESOLUTION_2817;
    let near_flat = array![[1.0, 0.0], [0.0, 1.0e-9]];
    let flat_residual = array![0.0, 1.0e-5];
    assert_eq!(
        crate::rho_optimizer::run::newton_predicted_decrease_at_resolution(
            &near_flat,
            &flat_residual,
            resolution,
        ),
        newton_predicted_decrease(&near_flat, &flat_residual),
        "a Hessian that factors at the arithmetic shift keeps its historical decrement"
    );
    let sub_resolution = array![[1.0, 0.0], [0.0, -1.0e-6]];
    let stiff_residual = array![1.0e-3, 0.0];
    assert!(
        newton_predicted_decrease(&sub_resolution, &stiff_residual).is_none(),
        "the fixture needs the arithmetic shift alone to find no factor"
    );
    assert!(
        crate::rho_optimizer::run::newton_predicted_decrease_at_resolution(
            &sub_resolution,
            &stiff_residual,
            0.0,
        )
        .is_none(),
        "a zero resolution is the historical function"
    );
    let at_resolution = crate::rho_optimizer::run::newton_predicted_decrease_at_resolution(
        &sub_resolution,
        &stiff_residual,
        resolution,
    )
    .expect("a sub-resolution negative eigenvalue factors at the resolution shift");
    let expected = 0.5 * 1.0e-6 / (1.0 + resolution);
    assert!(
        (at_resolution - expected).abs() <= 1.0e-12 * expected,
        "the decrement is ½·gᵀ(H + resolution·I)⁻¹g: got {at_resolution:.6e}, expected {expected:.6e}"
    );
    let saddle = array![[1.0, 0.0], [0.0, -1.0]];
    assert!(
        crate::rho_optimizer::run::newton_predicted_decrease_at_resolution(
            &saddle,
            &stiff_residual,
            resolution,
        )
        .is_none(),
        "a resolvable negative eigenvalue factors at neither shift"
    );
}

/// The stop fires where its own verdict accepts (#2817 with #1082).
///
/// `λ = −1e-6` is resolvable by the arithmetic shift `√ε = 1.49e-8` but not by the
/// criterion's curvature resolution `2·FLOOR_2817·(1 + |V|) = 2.002e-4`, so the
/// bridge judges this reduced Hessian PSD. The residual [`STOP_GRAD_2817`] lies
/// along the stiff direction, a Newton decrement of `8.45e-5`, inside the
/// criterion's resolution `1.001e-4`, exactly as in the unit-curvature fixture
/// above. Before the decrement travelled with the verdict, the shifted Cholesky
/// found no factor, this exit returned nothing, and the guard kept escaping a
/// stall its own verdict called stationary.
#[test]
fn a_sub_resolution_negative_eigenvalue_does_not_block_the_stationary_stop_2817() {
    let hessian = array![[1.0, 0.0], [0.0, -1.0e-6]];
    let gradient = array![STOP_GRAD_2817, 0.0];
    let (lower, upper) = wide_box_2817(2);
    assert_eq!(
        reduced_hessian_psd_at_point(
            &array![0.5, 0.5],
            &gradient,
            &hessian,
            Some((&lower, &upper)),
            2.0 * RESOLUTION_2817,
        ),
        Some(true),
        "the fixture needs the negative eigenvalue below the criterion's curvature resolution"
    );
    assert!(
        newton_predicted_decrease(&hessian, &gradient).is_none(),
        "the fixture needs the arithmetic shift alone to find no factor"
    );
    let (outcomes, published) = drive_arc_oracle_2817(
        array![0.5, 0.5],
        flatlined_2817(gradient, ARC_COST_STALL_WINDOW + 3),
        hessian,
        wide_box_2817(2),
        Some(FLOOR_2817),
    );
    assert_eq!(
        outcomes.last().expect("ran").clone().err().as_deref(),
        Some(ARC_CURVATURE_STATIONARY_SENTINEL),
        "a flatlined stall whose decrement is inside the resolution, under a reduced \
         Hessian PSD at that resolution, must end the stall: {outcomes:?}"
    );
    let published = published.expect("the halt publishes its point");
    assert!(published.converged);
}

/// NEGATIVE CONTROL: the same sub-resolution negative eigenvalue, with the
/// residual lying ALONG it. Its decrement at the resolution shift is
/// `½·(1.3e-2)²/(2.002e-4 − 1e-6) ≈ 0.42`, four thousand times the criterion's
/// resolution, so the search keeps moving. Taking the decrement at the larger
/// shift errs toward continuing, never toward stopping.
#[test]
fn a_residual_along_a_sub_resolution_negative_direction_keeps_the_search_moving_2817() {
    let hessian = array![[1.0, 0.0], [0.0, -1.0e-6]];
    let gradient = array![0.0, STOP_GRAD_2817];
    let (outcomes, published) = drive_arc_oracle_2817(
        array![0.5, 0.5],
        flatlined_2817(gradient, ARC_COST_STALL_WINDOW + 3),
        hessian,
        wide_box_2817(2),
        Some(FLOOR_2817),
    );
    assert!(
        outcomes.iter().all(|o| o.is_ok()),
        "a residual whose Newton step along a sub-resolution negative direction buys \
         thousands of resolutions must keep the search running: {outcomes:?}"
    );
    assert!(
        published.is_none_or(|exit| !exit.converged),
        "such a stall must never be published as converged"
    );
}

// ─── the budget retry's iteration ledger ─────────────────────────────────────

/// The ARC budget retry must report every attempt's iterations, not only the
/// last attempt's.
///
/// `OuterResult.iterations` is "total outer iterations across all solver
/// restarts", and a fit-level `outer_iterations < max_iter` is the only evidence
/// a caller has that no attempt ran out of budget. The retry dropped each
/// exhausted attempt's count, so a search that burned its budget and was then
/// continued read as one short run.
///
/// Same quartic ladder as
/// `run_nonconverged_arc_returns_typed_checkpoint_after_budget_retry_ladder`:
/// every attempt exhausts `max_iter = 1` and both retries fire, so the ladder
/// spends more iterations than any one attempt can report. The call goes
/// through `run_outer_uncertified`, the layer that owns the retry, so the
/// certification resume in `run_outer` (which adds its own counts) cannot
/// supply the total.
#[test]
fn arc_budget_retry_reports_every_attempts_iterations_2817() {
    const OFFSET: f64 = 0.5;
    const SCALE: f64 = 1.0e6;
    const MAX_ITER: usize = 1;
    let mut seed_config = gam_problem::SeedConfig::default();
    seed_config.seed_budget = 1;
    seed_config.risk_profile = gam_problem::SeedRiskProfile::Gaussian;
    let problem = OuterProblem::new(1)
        .with_gradient(Derivative::Analytic)
        .with_hessian(DeclaredHessianForm::Either)
        .with_seed_config(seed_config)
        .with_initial_rho(array![5.0])
        .with_max_iter(MAX_ITER);
    let mut obj = problem.build_objective(
        (),
        |_: &mut (), theta: &Array1<f64>| Ok(SCALE * (theta[0] - OFFSET).powi(4)),
        |_: &mut (), theta: &Array1<f64>| {
            let d = theta[0] - OFFSET;
            Ok(OuterEval {
                cost: SCALE * d.powi(4),
                gradient: array![SCALE * 4.0 * d.powi(3)],
                hessian: HessianValue::Dense(array![[SCALE * 12.0 * d.powi(2)]]),
                inner_beta_hint: None,
            })
        },
        None::<fn(&mut ())>,
        None::<fn(&mut (), &Array1<f64>) -> Result<EfsEval, EstimationError>>,
    );
    let checkpoint = super::super::super::run::run_outer_uncertified(
        &mut obj,
        &problem.config(),
        "arc budget retry iterations #2817",
    )
    .expect("an exhausted ARC ladder hands its best finite checkpoint to the certificate");
    eprintln!(
        "[#2817 arc retry iterations] iterations={} solver_converged={} rho={:?}",
        checkpoint.iterations,
        checkpoint.solver_claimed_convergence(),
        checkpoint.rho,
    );
    assert!(
        !checkpoint.solver_claimed_convergence(),
        "fixture precondition: no attempt reaches the quartic optimum at {OFFSET} within \
         max_iter = {MAX_ITER}"
    );
    assert!(
        checkpoint.iterations > MAX_ITER,
        "the returned checkpoint reports {} iteration(s), a count one exhausted attempt with \
         max_iter = {MAX_ITER} could spend alone, although the budget retry ran more than one \
         attempt",
        checkpoint.iterations
    );
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
