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
//
// One stalled step reaches a verdict (#3018): a step stalls when neither its
// measured decrease nor its model's predicted decrease is resolvable against the
// two values' resolutions. These scripted objectives publish no evidence, so each
// value's resolution is the criterion's resolution [`RESOLUTION_2817`] the guard
// is built with.

use super::*;
use ndarray::array;
use opt::OperatorObjective;

/// The criterion value the flat fixtures sit at. A REML score of `O(1e3)` is
/// the ordinary scale of the fits this defect was measured on. The resolution
/// is absolute, so this value only places the fixtures; it does not scale any
/// threshold.
const COST_2817: f64 = 1.0e3;

/// The criterion resolution these fixtures hand the bridge and its guard: the
/// statistical resolution `τ_stat = 1/(2n)` of an `n = 5000` fit, `1e-4`,
/// written out so a fixture states the threshold it brackets instead of
/// importing it and asserting a tautology.
const RESOLUTION_2817: f64 = 1.0e-4;

/// The certificate band these fixtures' guards judge claims by (#2817). The guard
/// reads its band from the run's configuration; this one is the absolute
/// tolerance, so the band is the same at every value.
const CLAIM_BAND_2817: f64 = 1.0e-3;

/// Evaluations a fixture needs for its first stall to be judged: the first
/// sample becomes the incumbent and the second is the stalled step. Each one is
/// settled as an accepted iterate right after it is evaluated (#3017), so the
/// second evaluation reports the verdict.
const FIRST_STALL_2817: usize = 2;

/// Evaluations a fixture needs for a second stall to be judged.
const SECOND_STALL_2817: usize = FIRST_STALL_2817 + 1;

fn claim_band_config_2817(band: f64) -> OuterConfig {
    OuterConfig {
        tolerance: band,
        ..OuterConfig::default()
    }
}

/// The residual the paired fixtures sit at.
///
/// Chosen to satisfy three things at once, which is what makes the pair sharp:
/// its Newton decrement `½‖g‖²/(1 + √ε) = 8.45e-5` is INSIDE the criterion's
/// resolution `1e-4` (so the certificate accepts the point); it is three
/// orders ABOVE the default absolute outer band `1e-5` (so the solver's own
/// stopping test never reaches it, which is the whole defect); and it is above
/// [`CLAIM_BAND_2817`] (so the guard does not read the point as
/// KKT-stationary-at-bound, which would stall it from the gradient rather than
/// from the criterion and make the descending control vacuous).
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
/// stopped moving (the step stalls and is adjudicated), a DECREASING one is a
/// search still making progress (nothing stalls and nothing is adjudicated).
/// `opt` accepts every scripted trial. Evaluation stops at the first error.
/// Value probes answer with the schedule's first cost: a criterion that is flat
/// everywhere.
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
    value: impl FnMut(&Array1<f64>) -> f64,
) -> (Vec<Result<f64, String>>, Option<CostStallExit>) {
    let points = vec![point; samples.len()];
    drive_arc_oracle_at_points_2817(points, samples, hessian, bounds, floor, value)
}

/// [`drive_arc_oracle_valued_2817`] with evaluation `i` taken at `points[i]`
/// instead of at one fixed point, so a fixture can say which trials it
/// evaluated.
fn drive_arc_oracle_at_points_2817(
    points: Vec<Array1<f64>>,
    samples: Vec<(f64, Array1<f64>)>,
    hessian: Array2<f64>,
    bounds: (Array1<f64>, Array1<f64>),
    floor: Option<f64>,
    value: impl FnMut(&Array1<f64>) -> f64,
) -> (Vec<Result<f64, String>>, Option<CostStallExit>) {
    drive_arc_oracle_publishing_2817(points, samples, hessian, bounds, floor, value, None)
}

/// [`drive_arc_oracle_at_points_2817`] on a route that takes the certificate's
/// Newton-decrement verdict (#2954): when `verdict` is given, the bridge judges
/// under its configuration and every evaluation publishes its evidence to the
/// armed capture, exactly as a REML evaluator publishes its own.
fn drive_arc_oracle_publishing_2817(
    points: Vec<Array1<f64>>,
    samples: Vec<(f64, Array1<f64>)>,
    hessian: Array2<f64>,
    bounds: (Array1<f64>, Array1<f64>),
    floor: Option<f64>,
    mut value: impl FnMut(&Array1<f64>) -> f64,
    verdict: Option<(&OuterConfig, crate::estimate::outer_eval_capture::CertificateEvidence)>,
) -> (Vec<Result<f64, String>>, Option<CostStallExit>) {
    let (verdict_config, published_evidence) = verdict.unzip();
    assert_eq!(points.len(), samples.len(), "one point per scripted sample");
    let point = points[0].clone();
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
            if let Some(evidence) = published_evidence.as_ref() {
                use crate::estimate::outer_eval_capture as capture;
                capture::record_certificate_parts(&evidence.parts);
                if let Some(criterion) = evidence.criterion {
                    capture::record_certificate_criterion(criterion);
                }
                if let Some(factor) = evidence.inner_factor {
                    capture::record_certificate_inner_factor(factor);
                }
                if let Some(charge) = evidence.inner_residual {
                    capture::record_certificate_inner_residual(charge);
                }
            }
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
        RESOLUTION_2817,
        &claim_band_config_2817(CLAIM_BAND_2817),
        exit.clone(),
    );
    let ledger: Arc<AcceptedStepLedger> = Arc::default();
    let mut bridge = OuterSecondOrderBridge {
        obj: &mut obj,
        layout: OuterThetaLayout::new(point.len(), 0),
        hessian_source: HessianSource::Analytic,
        eval_count: 0,
        outer_inner_cap: None,
        g_norm_initial: None,
        last_g_norm: None,
        last_value_grad_rho: None,
        cost_stall: Some(guard),
        cost_stall_bounds: Some(bounds),
        curvature_stationary_resolution: floor,
        accepted_trials: AcceptedTrialGate::new(Arc::clone(&ledger)),
        decrement_verdict_config: verdict_config,
    };
    // Every scripted evaluation is an iterate ARC accepted, so each one is
    // reported accepted and settled before the next (#3017); a stop that
    // settling reaches is that evaluation's outcome.
    let mut outcomes = Vec::new();
    for (iter, trial) in points.iter().enumerate() {
        let outcome = SecondOrderObjective::eval_hessian(&mut bridge, trial)
            .and_then(|sample| {
                report_accepted_trial_3017(&ledger, iter);
                bridge.settle_pending_trial().map_or(Ok(sample.value), Err)
            });
        match outcome {
            Ok(value) => outcomes.push(Ok(value)),
            Err(err) => {
                outcomes.push(Err(err.into_message()));
                break;
            }
        }
    }
    let published = exit.lock().expect("exit cell").take();
    (outcomes, published)
}

/// A criterion that has stopped moving, held at the same value.
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
/// the last by far more than the guard's resolution.
fn descending_2817(gradient: Array1<f64>, count: usize) -> Vec<(f64, Array1<f64>)> {
    (0..count)
        .map(|i| (COST_2817 - (i as f64), gradient.clone()))
        .collect()
}

/// The adjudication fires, and the point it stops at carries a gradient two
/// orders ABOVE the absolute band the solver was being driven to.
///
/// [`STOP_GRAD_2817`] against unit curvature is a Newton decrement of `8.45e-5`,
/// inside the criterion's resolution of `1e-4`: no step from here can move
/// the criterion by as much as it can be resolved. The default absolute outer
/// band is `1e-5`, so before this repair the search kept going — toward a
/// threshold the certificate it was heading for never applies.
#[test]
fn a_flatlined_arc_stall_is_adjudicated_by_the_certificates_own_test_2817() {
    let (outcomes, published) = drive_arc_oracle_2817(
        array![0.5],
        flatlined_2817(array![STOP_GRAD_2817], SECOND_STALL_2817),
        array![[1.0]],
        wide_box_2817(1),
        Some(RESOLUTION_2817),
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
    assert_eq!(
        outcomes.len(),
        FIRST_STALL_2817,
        "the halt comes at the first stalled step, not at the incumbent's own \
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

/// The wall the escape's adjudication exists for (#2817). A flat valley whose
/// incumbent keeps creeping by less than the criterion's resolution, while its
/// residual keeps contracting, is licensed at every stall by that contraction.
/// So a stall above the band that escapes WITHOUT adjudication runs until the
/// iteration count ends it: the 200-iteration wall measured on the gaussian
/// n=50 000 fit. Every stall above the band escapes, and the escape is
/// adjudicated by the certificate's own decrement test, so this one stops at its
/// first stall, at a point the certificate accepts.
#[test]
fn a_creeping_flat_valley_stops_at_its_first_window_by_adjudication_2817() {
    const WALL_2817: usize = 200;
    // Each evaluation improves by 1e-9, far under the resolution `1e-4`, so
    // every step stalls while the incumbent still moves.
    const CREEP_2817: f64 = 1.0e-9;
    // The residual contracts by 1e-4 of itself per evaluation, so each stall's
    // incumbent carries a smaller gradient than the last one did.
    const CONTRACTION_2817: f64 = 1.0e-4;
    let schedule: Vec<(f64, Array1<f64>)> = (0..WALL_2817)
        .map(|index| {
            let step = index as f64;
            (
                COST_2817 - CREEP_2817 * step,
                array![STOP_GRAD_2817 * (1.0 - CONTRACTION_2817 * step)],
            )
        })
        .collect();
    let (outcomes, published) = drive_arc_oracle_2817(
        array![0.5],
        schedule,
        array![[1.0]],
        wide_box_2817(1),
        Some(RESOLUTION_2817),
    );
    assert_eq!(
        outcomes.last().expect("ran").clone().err().as_deref(),
        Some(ARC_CURVATURE_STATIONARY_SENTINEL),
        "a creeping flat valley the certificate accepts must stop by adjudication, not run \
         the wall: {} evaluation(s)",
        outcomes.len()
    );
    assert_eq!(
        outcomes.len(),
        FIRST_STALL_2817,
        "the stop must come at the first stalled step: {} evaluation(s)",
        outcomes.len()
    );
    assert!(
        published.is_some_and(|exit| exit.converged),
        "the adjudicated stop publishes its point as converged"
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
        descending_2817(array![STOP_GRAD_2817], 2 * SECOND_STALL_2817),
        array![[1.0]],
        wide_box_2817(1),
        Some(RESOLUTION_2817),
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
        flatlined_2817(array![1.0], FIRST_STALL_2817),
        array![[1.0]],
        wide_box_2817(1),
        Some(RESOLUTION_2817),
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
        flatlined_2817(array![critical * 0.99], FIRST_STALL_2817),
        array![[1.0]],
        wide_box_2817(1),
        Some(RESOLUTION_2817),
    );
    assert_eq!(
        inside.last().expect("ran").clone().err().as_deref(),
        Some(ARC_CURVATURE_STATIONARY_SENTINEL),
        "a decrement 1% inside the criterion's resolution must end the stall"
    );
    let (outside, _) = drive_arc_oracle_2817(
        array![0.5],
        flatlined_2817(array![critical * 1.01], FIRST_STALL_2817),
        array![[1.0]],
        wide_box_2817(1),
        Some(RESOLUTION_2817),
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
        flatlined_2817(gradient, FIRST_STALL_2817),
        hessian,
        wide_box_2817(2),
        Some(RESOLUTION_2817),
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
/// ([`CLAIM_BAND_2817`] here) the bridge stops ARC where the
/// certificate would accept.
#[test]
fn a_strict_saddle_claim_the_criterion_contradicts_stops_at_the_incumbent_1082() {
    let (outcomes, published) = drive_arc_oracle_valued_2817(
        array![0.5, 0.5],
        flatlined_2817(array![1.0e-6, 1.0e-6], SECOND_STALL_2817),
        array![[1.0, 0.0], [0.0, -1.0]],
        wide_box_2817(2),
        Some(RESOLUTION_2817),
        |_| COST_2817,
    );
    assert_eq!(
        outcomes.last().expect("ran").clone().err().as_deref(),
        Some(ARC_CURVATURE_STATIONARY_SENTINEL),
        "a strict-saddle claim the criterion contradicts, inside the solver band, must end \
         the stall: {outcomes:?}"
    );
    assert_eq!(
        outcomes.len(),
        FIRST_STALL_2817,
        "the adjudication waits for the first stalled step: it took {} evaluations",
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
        flatlined_2817(array![5.0e-3, 5.0e-3], FIRST_STALL_2817),
        array![[1.0, 0.0], [0.0, -1.0]],
        wide_box_2817(2),
        Some(RESOLUTION_2817),
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
        flatlined_2817(gradient, FIRST_STALL_2817),
        hessian,
        wide_box_2817(2),
        Some(RESOLUTION_2817),
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
        flatlined_2817(array![1.0], SECOND_STALL_2817),
        array![[1.0]],
        wide_box_2817(1),
        Some(RESOLUTION_2817),
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

/// POSITIVE CONTROL ON THE WIRING. The resolution is what performs the
/// adjudication: the same stall on a route that declares no criterion
/// resolution is handed back to ARC.
#[test]
fn a_route_that_declares_no_resolution_is_unchanged_2817() {
    let (outcomes, published) = drive_arc_oracle_2817(
        array![0.5],
        flatlined_2817(array![STOP_GRAD_2817], FIRST_STALL_2817),
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

// ─── the stop decides on the certificate's verdict (#2954) ──────────────────

/// The size the certifying verdict fixture declares, as every REML route does:
/// its statistical resolution is `τ_stat = 1/(2n) = 5e-4`.
const VERDICT_ROWS_2954: usize = 1_000;
/// The size the refusing verdict fixture declares, `τ_stat = 5e-6`.
const REFUSING_ROWS_2954: usize = 100_000;
const VERDICT_COEFFICIENTS_2954: usize = 10;

/// [`claim_band_config_2817`] on a route that declares `n_obs` rows, so the
/// certificate decides stationarity on the Newton-decrement verdict (#2954).
fn sized_claim_band_config_2954(n_obs: usize) -> OuterConfig {
    OuterConfig {
        problem_size: crate::rho_optimizer::OuterProblemSize {
            n_obs: Some(n_obs),
            p_coefficients: Some(VERDICT_COEFFICIENTS_2954),
        },
        ..claim_band_config_2817(CLAIM_BAND_2817)
    }
}

/// What a one-coordinate REML evaluation at residual `gradient` publishes: the
/// whole entry in the penalty channel, an exact inner mode, and, when given, the
/// criterion's channels with the inner factor `log|H_β|` was read from.
fn published_evidence_2954(
    gradient: f64,
    criterion: Option<(
        crate::estimate::outer_eval_capture::CertificateCriterion,
        crate::estimate::outer_eval_capture::InnerFactorCondition,
    )>,
) -> crate::estimate::outer_eval_capture::CertificateEvidence {
    use crate::estimate::outer_eval_capture as capture;
    capture::CertificateEvidence {
        parts: vec![capture::RhoGradientParts {
            index: 0,
            lambda: 0.5f64.exp(),
            block_quadratic: 0.0,
            rank: 1,
            dim: 1,
            fixed_beta: gradient,
            logdet_h: 0.0,
            frozen_logdet_h: 0.0,
            mode_response_logdet_h: 0.0,
            logdet_s: 0.0,
            total: gradient,
        }],
        criterion: criterion.map(|(criterion, _)| criterion),
        inner_factor: criterion.map(|(_, factor)| factor),
        inner_residual: Some(capture::InnerResidualCharge {
            energy: 0.0,
            source: capture::InnerResidualSource::InnerGradient,
        }),
    }
}

/// The flatlined unit-curvature stall of
/// [`a_flatlined_arc_stall_is_adjudicated_by_the_certificates_own_test_2817`],
/// on a route that takes the certificate's verdict under `config`.
fn drive_flat_stall_with_verdict_2954(
    gradient: f64,
    config: &OuterConfig,
    evidence: crate::estimate::outer_eval_capture::CertificateEvidence,
) -> (Vec<Result<f64, String>>, Option<CostStallExit>) {
    let samples = flatlined_2817(array![gradient], SECOND_STALL_2817 + 1);
    drive_arc_oracle_publishing_2817(
        vec![array![0.5]; samples.len()],
        samples,
        array![[1.0]],
        wide_box_2817(1),
        Some(RESOLUTION_2817),
        |_| COST_2817,
        Some((config, evidence)),
    )
}

/// The loop does not stop where its certificate refuses (#2954).
///
/// [`STOP_GRAD_2817`] against unit curvature is a model decrease `½g² = 8.45e-5`,
/// inside the resolution `1e-4`, so the curvature-resolvability rung
/// halts ARC there (the control half, on a route that takes no verdict). A route
/// that declares its size is certified on the verdict instead: at `1e5` rows its
/// statistical resolution is `τ_stat = 1/(2n) = 5e-6`, and a criterion of `V =
/// 1e3` channelled through the penalty alone rounds far inside it, so the
/// decrement `g² = 1.69e-4` is resolvable and the certificate refuses the point by
/// name. The abalone Poisson fit (UCI, 5-fold CV fold 0) was this split: `|V| =
/// 4.4e4` carries a λ-independent `Σ log y!`, so the old rung's tolerance grew
/// with a constant no decision depends on. Every seed stopped at `|Pg| ≈ 1e-4`,
/// the screening certificate refused each on `DecrementAboveTolerance`, and the
/// fit was minted only by the polish after all three seeds had run.
#[test]
fn the_online_stop_declines_a_point_the_certificates_verdict_refuses_2954() {
    let (control, _) = drive_arc_oracle_2817(
        array![0.5],
        flatlined_2817(array![STOP_GRAD_2817], SECOND_STALL_2817 + 1),
        array![[1.0]],
        wide_box_2817(1),
        Some(RESOLUTION_2817),
    );
    assert_eq!(
        control.last().expect("ran").clone().err().as_deref(),
        Some(ARC_CURVATURE_STATIONARY_SENTINEL),
        "control: without the verdict the curvature-resolvability rung halts this stall"
    );

    let config = sized_claim_band_config_2954(REFUSING_ROWS_2954);
    let evidence = published_evidence_2954(STOP_GRAD_2817, None);
    let decision = crate::rho_optimizer::decrement_bands::outer_decrement_verdict(
        &config,
        &array![[1.0]],
        &array![STOP_GRAD_2817],
        &[],
        COST_2817,
        &evidence,
    )
    .expect("the fixture publishes everything the verdict needs");
    assert!(
        matches!(decision.verdict, opt::DecrementVerdict::DecrementAboveTolerance(_)),
        "fixture precondition: the certificate refuses this point on its verdict: {:?}",
        decision.verdict
    );

    let (outcomes, published) = drive_flat_stall_with_verdict_2954(STOP_GRAD_2817, &config, evidence);
    assert!(
        outcomes
            .iter()
            .all(|outcome| outcome.as_ref().err().map(String::as_str)
                != Some(ARC_CURVATURE_STATIONARY_SENTINEL)),
        "a point the certificate's verdict refuses must never be the loop's stationary stop: \
         {outcomes:?}"
    );
    assert!(
        published.is_none_or(|exit| !exit.converged),
        "such a point must never be published as converged"
    );
}

/// The residual of the certifying fixture: above the claim band, inside the
/// verdict's tolerance.
const CERTIFIED_GRADIENT_2954: f64 = 5.0e-3;

/// A `1e3`-row route and the evidence of a point its verdict certifies at
/// [`CERTIFIED_GRADIENT_2954`] against unit curvature, with both preconditions
/// checked.
fn certifying_verdict_2954() -> (
    OuterConfig,
    crate::estimate::outer_eval_capture::CertificateEvidence,
) {
    let config = sized_claim_band_config_2954(VERDICT_ROWS_2954);
    let criterion = crate::estimate::outer_eval_capture::CertificateCriterion {
        cost: COST_2817,
        fixed_beta: COST_2817 - 1.0,
        logdet_h: 1.0,
        logdet_s: 0.0,
        kkt: 0.0,
        inner_residual_energy: Some(0.0),
    };
    let factor = crate::estimate::outer_eval_capture::InnerFactorCondition {
        logdet_forward_error: 1.0e-4,
    };
    let evidence = published_evidence_2954(CERTIFIED_GRADIENT_2954, Some((criterion, factor)));
    let decision = crate::rho_optimizer::decrement_bands::outer_decrement_verdict(
        &config,
        &array![[1.0]],
        &array![CERTIFIED_GRADIENT_2954],
        &[],
        COST_2817,
        &evidence,
    )
    .expect("the fixture publishes everything the verdict needs");
    assert!(
        decision.verdict.is_certified(),
        "fixture precondition: the certificate accepts this point on its verdict: {:?}",
        decision.verdict
    );
    assert!(
        CERTIFIED_GRADIENT_2954 > CLAIM_BAND_2817,
        "the stop must not come from the claim band"
    );
    (config, evidence)
}

/// POSITIVE CONTROL: the verdict stops the loop where it certifies.
///
/// The criterion's `½·log|H_β|` channel comes from a factor whose forward error
/// is `1e-4`, which charges `5e-5` to the objective band, leaving the decrement
/// `τ_stat − 5e-5 ≈ 4.5e-4` of the statistical resolution `τ_stat = 5e-4` at
/// `1e3` rows. `|g| = 5e-3` is a decrement of `2.5e-5`, inside that tolerance, so
/// the certificate accepts the point and the loop halts there, above the claim
/// band the guard reads as KKT-stationary.
#[test]
fn the_online_stop_halts_where_the_certificates_verdict_certifies_2954() {
    let (config, evidence) = certifying_verdict_2954();
    let (outcomes, published) =
        drive_flat_stall_with_verdict_2954(CERTIFIED_GRADIENT_2954, &config, evidence);
    assert_eq!(
        outcomes.last().expect("ran").clone().err().as_deref(),
        Some(ARC_CURVATURE_STATIONARY_SENTINEL),
        "a flatlined stall the certificate's verdict certifies must halt ARC: {outcomes:?}"
    );
    let published = published.expect("the halt publishes its point");
    assert!(published.converged);
    assert_eq!(published.value, COST_2817);
}

/// The verdict is taken at every evaluated point, not only once a stall window
/// has filled.
///
/// A criterion still falling by a unit each step never stalls (#3018), yet the
/// first point it reaches is one its certificate accepts on the verdict. Walking
/// on buys nothing the certificate can resolve, and on a smoothing parameter
/// penalized out of the fit that walk is linear: the REML gradient decays as
/// `e^{−ρ}` and each ARC step is one e-fold. Gated behind the window, the
/// Gaussian `s(x)` fit at n = 10⁴ certified at ρ ≈ 19.6 and stopped at ρ ≈ 22.7,
/// about five evaluations per seed for nothing.
#[test]
fn a_descending_search_stops_at_the_first_point_its_verdict_certifies_2954() {
    let (config, evidence) = certifying_verdict_2954();
    let samples = descending_2817(array![CERTIFIED_GRADIENT_2954], SECOND_STALL_2817 + 3);
    let (outcomes, published) = drive_arc_oracle_publishing_2817(
        vec![array![0.5]; samples.len()],
        samples,
        array![[1.0]],
        wide_box_2817(1),
        Some(RESOLUTION_2817),
        |_| COST_2817,
        Some((&config, evidence)),
    );
    assert_eq!(
        outcomes,
        vec![Err(ARC_CURVATURE_STATIONARY_SENTINEL.to_string())],
        "the first evaluated point the verdict certifies must halt ARC, with no stall"
    );
    let published = published.expect("the halt publishes its point");
    assert!(published.converged);
    assert_eq!(published.value, COST_2817);
}

/// A caller's required `|Pg|` (#2568) holds the online stop until it is met
/// (#3311, #3429).
///
/// 6ac3a704e7 took the Newton-decrement verdict at every evaluation and stopped
/// at the first point it certified, whatever the caller required, so the exact
/// block Gaussian REML fit returned a point short of its `|Pg| <= 1e-8`. #3429
/// declines such a point in the online stop. This pins the boundary.
///
/// The descending search above halts at its first point, where the verdict
/// certifies the residual [`CERTIFIED_GRADIENT_2954`]. With a requirement one ulp
/// below that residual, the same search keeps moving, because the caller asked
/// for a tighter point than the verdict needs. The exact block Gaussian REML fit
/// is such a caller: its backward pass needs the optimum's zero gradient. With a
/// requirement exactly at the residual, the requirement is met, and the search
/// halts at the same first point (the control).
#[test]
fn a_caller_requirement_holds_the_online_stop_until_it_is_met_3311() {
    let (config, evidence) = certifying_verdict_2954();
    let run = |required: f64| {
        let config = OuterConfig {
            required_projected_gradient_norm: Some(required),
            ..config.clone()
        };
        let samples = descending_2817(array![CERTIFIED_GRADIENT_2954], SECOND_STALL_2817 + 3);
        drive_arc_oracle_publishing_2817(
            vec![array![0.5]; samples.len()],
            samples,
            array![[1.0]],
            wide_box_2817(1),
            Some(RESOLUTION_2817),
            |_| COST_2817,
            Some((&config, evidence.clone())),
        )
    };

    let (held, published) = run(CERTIFIED_GRADIENT_2954.next_down());
    assert!(
        held.iter().all(|outcome| outcome.is_ok()),
        "a requirement tighter than the certified residual must keep the search moving: {held:?}"
    );
    assert!(published.is_none_or(|exit| !exit.converged));

    let (halted, published) = run(CERTIFIED_GRADIENT_2954);
    assert_eq!(
        halted,
        vec![Err(ARC_CURVATURE_STATIONARY_SENTINEL.to_string())],
        "a requirement the residual meets must not hold the stop"
    );
    assert!(published.expect("the halt publishes its point").converged);
}

/// CONTROL: the curvature-resolvability rung still waits for a stalled
/// criterion. The same descending search on a route that takes no verdict, whose
/// every point is inside the rung's tolerance, is never halted on it.
#[test]
fn where_no_verdict_is_taken_a_descending_search_is_not_halted_2954() {
    let config = claim_band_config_2817(CLAIM_BAND_2817);
    let evidence = published_evidence_2954(STOP_GRAD_2817, None);
    let samples = descending_2817(array![STOP_GRAD_2817], SECOND_STALL_2817 + 3);
    let (outcomes, published) = drive_arc_oracle_publishing_2817(
        vec![array![0.5]; samples.len()],
        samples,
        array![[1.0]],
        wide_box_2817(1),
        Some(RESOLUTION_2817),
        |_| COST_2817,
        Some((&config, evidence)),
    );
    assert!(
        outcomes.iter().all(|o| o.is_ok()),
        "a descending search the verdict does not decide must never be halted: {outcomes:?}"
    );
    assert!(published.is_none_or(|exit| !exit.converged));
}

/// Where no verdict is taken the certificate's `else` branch, the
/// curvature-resolvability rung, decides, and so does the loop: the same
/// evidence on a route that declares no size halts where it always did.
#[test]
fn where_no_verdict_is_taken_the_curvature_rung_still_decides_2954() {
    let config = claim_band_config_2817(CLAIM_BAND_2817);
    let evidence = published_evidence_2954(STOP_GRAD_2817, None);
    assert_eq!(
        crate::rho_optimizer::decrement_bands::outer_decrement_verdict(
            &config,
            &array![[1.0]],
            &array![STOP_GRAD_2817],
            &[],
            COST_2817,
            &evidence,
        )
        .err(),
        Some(crate::rho_optimizer::decrement_bands::DecrementVerdictNotTaken::NoProblemSize),
        "fixture precondition: a route with no declared size takes no verdict"
    );
    let (outcomes, published) = drive_flat_stall_with_verdict_2954(STOP_GRAD_2817, &config, evidence);
    assert_eq!(
        outcomes.last().expect("ran").clone().err().as_deref(),
        Some(ARC_CURVATURE_STATIONARY_SENTINEL),
        "with no verdict taken the curvature-resolvability rung halts this stall: {outcomes:?}"
    );
    assert!(published.is_some_and(|exit| exit.converged));
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
/// criterion's curvature resolution `2·RESOLUTION_2817 = 2e-4`, so the
/// bridge judges this reduced Hessian PSD. The residual [`STOP_GRAD_2817`] lies
/// along the stiff direction, a Newton decrement of `8.45e-5`, inside the
/// criterion's resolution `1e-4`, exactly as in the unit-curvature fixture
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
        flatlined_2817(gradient, SECOND_STALL_2817),
        hessian,
        wide_box_2817(2),
        Some(RESOLUTION_2817),
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
/// `½·(1.3e-2)²/(2e-4 − 1e-6) ≈ 0.42`, four thousand times the criterion's
/// resolution, so the search keeps moving. Taking the decrement at the larger
/// shift errs toward continuing, never toward stopping.
#[test]
fn a_residual_along_a_sub_resolution_negative_direction_keeps_the_search_moving_2817() {
    let hessian = array![[1.0, 0.0], [0.0, -1.0e-6]];
    let gradient = array![0.0, STOP_GRAD_2817];
    let (outcomes, published) = drive_arc_oracle_2817(
        array![0.5, 0.5],
        flatlined_2817(gradient, FIRST_STALL_2817),
        hessian,
        wide_box_2817(2),
        Some(RESOLUTION_2817),
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

// ─── an unprogressing stall stops ────────────────────────────────────────────

/// Drive a flat stall at `gradient` through two stalled steps and report what the
/// bridge did, so the two stop fixtures below share one schedule.
fn drive_two_flat_stalls_2817(gradient: f64) -> (Vec<Result<f64, String>>, Option<CostStallExit>) {
    drive_arc_oracle_2817(
        array![0.5],
        flatlined_2817(array![gradient], SECOND_STALL_2817 + 1),
        array![[1.0]],
        wide_box_2817(1),
        Some(RESOLUTION_2817),
    )
}

/// The stop both fixtures below must produce: nothing halts at the first stall,
/// and the second, having bought nothing since, stops the run on the
/// unprogressing sentinel at the incumbent, without a convergence claim.
fn assert_stops_at_the_second_stall_2817(
    outcomes: &[Result<f64, String>],
    published: Option<CostStallExit>,
) {
    let stop = SECOND_STALL_2817 - 1;
    assert_eq!(
        outcomes.len(),
        stop + 1,
        "the run must stop exactly at its second stall: {outcomes:?}"
    );
    assert!(
        outcomes[..stop].iter().all(|outcome| outcome.is_ok()),
        "nothing may stop before the second stall: {outcomes:?}"
    );
    let message = outcomes[stop]
        .clone()
        .expect_err("the second unprogressing stall must stop the run");
    assert_eq!(
        message, ARC_UNPROGRESSING_STALL_SENTINEL,
        "the stop must be the unprogressing-stall sentinel, not an objective failure"
    );
    let exit = published.expect("the stop must publish the incumbent it stopped at");
    assert!(!exit.converged, "an unprogressing stop makes no convergence claim");
    assert_eq!(exit.rho, array![0.5]);
    assert_eq!(exit.value, COST_2817);
}

/// A flat stall above the guard's first-order band that the certificate does not
/// accept stops when a second stall has bought nothing (#2817).
///
/// `|g| = 1` is above [`CLAIM_BAND_2817`], and its Newton decrement `0.5` is 5000×
/// the criterion's resolution, so the certificate's own rung refuses it. That is
/// the split this issue's sleepstudy fit fell into: ARC kept spending windows at
/// |Pg| = 8.6e-2 against a bound of 1.6e-3 until its 200-iteration count ran out.
/// The first stall is licensed, since nothing about what continuing buys has been
/// measured yet. The second bought no resolved descent and did not contract the
/// residual, so the run stops.
#[test]
fn a_deferred_stall_that_buys_nothing_stops_at_its_second_window_2817() {
    let (outcomes, published) = drive_two_flat_stalls_2817(1.0);
    assert_stops_at_the_second_stall_2817(&outcomes, published);
}

/// The same stop for a larger residual, where the first stall grants a
/// stuck-stall escape. The escape bought nothing either, and the second stall
/// stops the run on the same licence.
#[test]
fn an_escape_that_buys_nothing_stops_at_its_second_window_2817() {
    let (outcomes, published) = drive_two_flat_stalls_2817(2.0);
    assert_stops_at_the_second_stall_2817(&outcomes, published);
}

/// The scripted costs of a stall, a unit of descent, and a second stall: what the
/// two resolved-descent controls below evaluate.
fn stall_descend_stall_2817() -> Vec<(f64, Array1<f64>)> {
    [COST_2817, COST_2817, COST_2817 - 1.0, COST_2817 - 1.0]
        .into_iter()
        .map(|cost| (cost, array![1.0]))
        .collect()
}

/// NEGATIVE CONTROL: the same residual, and a criterion that bought a whole unit
/// of decrease between the two stalls. Resolved descent licenses the second
/// stall, so nothing stops.
#[test]
fn a_stall_that_bought_resolved_descent_between_windows_keeps_moving_2817() {
    let (outcomes, _) = drive_arc_oracle_2817(
        array![0.5],
        stall_descend_stall_2817(),
        array![[1.0]],
        wide_box_2817(1),
        Some(RESOLUTION_2817),
    );
    assert!(
        outcomes.iter().all(|outcome| outcome.is_ok()),
        "a unit of descent, 10000 resolutions, must license the next stall: {outcomes:?}"
    );
}

/// NEGATIVE CONTROL: no resolved descent between the stalls, but the incumbent's
/// projected gradient halved. The search is buying stationarity, so the second
/// stall is licensed.
///
/// Each step improves by `1e-6`, below the resolution `1e-4`, so every step
/// stalls while the incumbent still moves and carries the gradient of the point
/// that set it.
#[test]
fn a_stall_whose_residual_contracted_between_windows_keeps_moving_2817() {
    let schedule: Vec<(f64, Array1<f64>)> = (0..SECOND_STALL_2817)
        .map(|index| {
            let gradient = if index < 2 { 1.0 } else { 0.5 };
            (COST_2817 - 1.0e-6 * index as f64, array![gradient])
        })
        .collect();
    let (outcomes, _) = drive_arc_oracle_2817(
        array![0.5],
        schedule,
        array![[1.0]],
        wide_box_2817(1),
        Some(RESOLUTION_2817),
    );
    assert!(
        outcomes.iter().all(|outcome| outcome.is_ok()),
        "a stall that halved the incumbent's residual must license the next one: \
         {outcomes:?}"
    );
}

/// Drive `samples.len()` evaluations of the matrix-free route's operator oracle
/// at `point`, one per entry, and report each outcome plus whatever the bridge
/// published to its unprogressing-stop slot. `opt` accepts every scripted trial.
/// Evaluation stops at the first error.
fn drive_operator_oracle_2817(
    point: Array1<f64>,
    samples: Vec<(f64, Array1<f64>)>,
) -> (Vec<Result<f64, String>>, Option<CostStallExit>) {
    let table = Arc::new(samples.clone());
    let calls = Arc::new(AtomicUsize::new(0));
    let problem = OuterProblem::new(point.len())
        .with_gradient(Derivative::Analytic)
        .with_hessian(DeclaredHessianForm::Either);
    let flat = samples[0].0;
    let mut obj = problem.build_objective_with_eval_order(
        (),
        move |_: &mut (), _: &Array1<f64>| Ok(flat),
        |_: &mut (), _: &Array1<f64>| {
            Err(EstimationError::InvalidInput(
                "legacy eager eval should not run".to_string(),
            ))
        },
        move |_: &mut (), theta: &Array1<f64>, _: OuterEvalOrder| {
            let idx = calls.fetch_add(1, Ordering::Relaxed);
            let (cost, gradient) = table[idx.min(table.len() - 1)].clone();
            Ok(OuterEval {
                cost,
                gradient,
                hessian: HessianValue::Dense(Array2::eye(theta.len())),
                inner_beta_hint: None,
            })
        },
        None::<fn(&mut ())>,
        None::<fn(&mut (), &Array1<f64>) -> Result<EfsEval, EstimationError>>,
    );
    let stop: Arc<Mutex<Option<CostStallExit>>> = Arc::new(Mutex::new(None));
    let guard = CostStallGuard::new(
        RESOLUTION_2817,
        &claim_band_config_2817(CLAIM_BAND_2817),
        Arc::new(Mutex::new(None)),
    );
    let ledger: Arc<AcceptedStepLedger> = Arc::default();
    let mut bridge = OuterOperatorBridge {
        obj: &mut obj,
        layout: OuterThetaLayout::new(point.len(), 0),
        outer_inner_cap: None,
        eval_count: 0,
        g_norm_initial: None,
        last_g_norm: None,
        last_value_grad_rho: None,
        cost_stall: Some(guard),
        cost_stall_bounds: Some(wide_box_2817(point.len())),
        unprogressing_stop: Arc::clone(&stop),
        accepted_trials: AcceptedTrialGate::new(Arc::clone(&ledger)),
    };
    let mut outcomes = Vec::new();
    for iter in 0..samples.len() {
        let outcome = OperatorObjective::eval_value_grad_op(&mut bridge, &point)
            .and_then(|sample| {
                report_accepted_trial_3017(&ledger, iter);
                bridge.settle_pending_trial().map_or(Ok(sample.value), Err)
            });
        match outcome {
            Ok(value) => outcomes.push(Ok(value)),
            Err(err) => {
                outcomes.push(Err(err.into_message()));
                break;
            }
        }
    }
    let published = stop.lock().expect("stop cell").take();
    (outcomes, published)
}

/// The matrix-free route stops the same way (#2817). opt's matrix-free trust
/// region has no stall stop of its own, so this flat stall, whose residual is
/// above the guard's first-order band and whose Newton decrement is 5000× the
/// criterion's resolution, used to end only when the iteration count ran out.
#[test]
fn an_operator_route_stall_that_buys_nothing_stops_at_its_second_window_2817() {
    let (outcomes, published) = drive_operator_oracle_2817(
        array![0.5],
        flatlined_2817(array![1.0], SECOND_STALL_2817 + 1),
    );
    assert_stops_at_the_second_stall_2817(&outcomes, published);
}

/// NEGATIVE CONTROL on the matrix-free route: resolved descent between the two
/// stalls licenses the second one.
#[test]
fn an_operator_route_stall_that_bought_resolved_descent_keeps_moving_2817() {
    let (outcomes, published) =
        drive_operator_oracle_2817(array![0.5], stall_descend_stall_2817());
    assert!(
        outcomes.iter().all(|outcome| outcome.is_ok()),
        "a unit of descent, 10000 resolutions, must license the next stall: {outcomes:?}"
    );
    assert!(published.is_none(), "a licensed run publishes no stop");
}

/// A stall at a certified strict saddle is granted its escape, and a proven
/// replay of that escape stops the run non-converged (#2817, #2668 row 30).
///
/// `H = diag(1, −1)`, so λ_min = −1 is far outside the criterion's curvature
/// resolution `2·1e-7·(1 + 1e3) ≈ 2e-4`, and the bridge calls the incumbent a
/// strict saddle. `|g| = 2` sits above the band, and the decrement exit refuses a
/// saddle, so neither ends the run. The first stall grants the saddle escape and
/// the run keeps going. The fixture evaluates one point, so the next stall is on
/// the same trial from a bit-identical incumbent: continuing provably replays the
/// same procedure, and no licence reopens a proven replay. The saddle licence
/// used to continue past that cut on the premise that ARC's regularization
/// ceiling ends a saddle it cannot exploit. opt at the pinned rev has no such exit
/// on its pre-evaluation failure paths, and parity1561's curved-acceleration
/// Weibull fit spun there until a test timeout. A real ARC run at a saddle moves
/// along the negative curvature, so its incumbent changes and the saddle licence
/// still applies (row 30).
#[test]
fn a_strict_saddle_stall_escapes_then_stops_on_its_proven_replay_2817() {
    let (outcomes, published) = drive_arc_oracle_valued_2817(
        array![0.5, 0.5],
        flatlined_2817(array![2.0, 0.0], SECOND_STALL_2817 + 1),
        array![[1.0, 0.0], [0.0, -1.0]],
        wide_box_2817(2),
        Some(RESOLUTION_2817),
        |_| COST_2817,
    );
    let replay = SECOND_STALL_2817 - 1;
    assert!(
        outcomes.len() > replay && outcomes[..replay].iter().all(|outcome| outcome.is_ok()),
        "the escape at a strict saddle must keep the search running through its first \
         stall: {outcomes:?}"
    );
    assert_eq!(
        outcomes.len(),
        replay + 1,
        "the proven replay must stop the run at the evaluation that judges it: {outcomes:?}"
    );
    assert_eq!(
        outcomes.last().expect("ran").clone().err().as_deref(),
        Some(ARC_UNPROGRESSING_STALL_SENTINEL),
        "the stop at a proven replay is the unprogressing-stall sentinel: {outcomes:?}"
    );
    assert!(
        published.is_some_and(|exit| !exit.converged),
        "the stop must publish its incumbent, and a strict saddle is never converged"
    );
}

/// A strict-saddle escape followed by a stall on a NEW trial is not a replay, even
/// though the incumbent did not move; the same trial from the same incumbent is.
///
/// The incumbent alone cannot tell a replay from a search that is still exploring
/// a saddle: ARC's regularization changes between trials and so do the points it
/// proposes. The gaussian pure-noise fit with twenty `k = 10` smooths at
/// `n = 100` stopped non-converged on exactly that misreading, at an incumbent of
/// 21.85 whose next window had evaluated 105 → 40 → 22.3 on fresh points. Here the
/// second stall is on a point the first never saw, so it earns a second escape;
/// the third is on the second's point again from the same incumbent, and that is
/// the replay the guard cuts.
#[test]
fn a_strict_saddle_window_on_new_trials_escapes_again_until_it_replays() {
    let seed = array![0.5, 0.5];
    let offset = |step: usize| array![0.5 + 0.01 * step as f64, 0.5];
    let points: Vec<Array1<f64>> = vec![seed, offset(1), offset(2), offset(2)];
    let (outcomes, published) = drive_arc_oracle_at_points_2817(
        points.clone(),
        flatlined_2817(array![2.0, 0.0], points.len()),
        array![[1.0, 0.0], [0.0, -1.0]],
        wide_box_2817(2),
        Some(RESOLUTION_2817),
        |_| COST_2817,
    );
    let replay = points.len() - 1;
    assert!(
        outcomes[..replay].iter().all(|outcome| outcome.is_ok()),
        "a stall on a trial the previous one never saw must earn another escape: \
         {outcomes:?}"
    );
    assert_eq!(
        outcomes.len(),
        replay + 1,
        "a stall on the previous stall's trial from the same incumbent is a proven replay \
         and must stop the run at the evaluation that judges it: {outcomes:?}"
    );
    assert_eq!(
        outcomes.last().expect("ran").clone().err().as_deref(),
        Some(ARC_UNPROGRESSING_STALL_SENTINEL),
        "the stop at a proven replay is the unprogressing-stall sentinel: {outcomes:?}"
    );
    assert!(
        published.is_some_and(|exit| !exit.converged),
        "the stop must publish its incumbent, and a strict saddle is never converged"
    );
}

// ─── #3018: the evaluation's own band decides resolved progress ──────────────

/// A scripted criterion has no inner solve, so its mode is exact: it charges a
/// zero inner-residual energy wherever a REML evaluator publishes its own.
const EXACT_INNER_MODE_3018: crate::estimate::outer_eval_capture::InnerResidualCharge =
    crate::estimate::outer_eval_capture::InnerResidualCharge {
        energy: 0.0,
        source: crate::estimate::outer_eval_capture::InnerResidualSource::InnerGradient,
    };

/// A crawl buying `1e-9` per step at `|g| = 1`: a ten-thousandth of the
/// criterion's resolution `RESOLUTION_2817 = 1e-4`, and ten thousand times the rounding band `γ₁·|V| ≈ 1.1e-13` the evaluation carries.
/// The residual sits far above the band and its decrement `0.5` far above the
/// resolution, so neither the band nor the adjudication can stop it.
fn crawl_3018(count: usize) -> Vec<(f64, Array1<f64>)> {
    (0..count)
        .map(|index| (COST_2817 - 1.0e-9 * index as f64, array![1.0]))
        .collect()
}

/// Drive the ARC bridge over `schedule` at one point, `opt` accepting every
/// trial, on a route that declares its problem size. With `publishes_band`, each
/// evaluation publishes an exact inner mode, so its value's resolution is its
/// rounding band ([`sample_resolution`]); without it, the evaluation publishes
/// nothing and the resolution is the criterion's, [`RESOLUTION_2817`].
fn drive_arc_crawl_3018(
    schedule: Vec<(f64, Array1<f64>)>,
    publishes_band: bool,
) -> (Vec<Result<f64, String>>, Option<CostStallExit>) {
    let point = array![0.5];
    let table = Arc::new(schedule.clone());
    let calls = Arc::new(AtomicUsize::new(0));
    let problem = OuterProblem::new(1)
        .with_gradient(Derivative::Analytic)
        .with_hessian(DeclaredHessianForm::Either);
    let flat = schedule[0].0;
    let mut obj = problem.build_objective_with_eval_order(
        (),
        move |_: &mut (), _: &Array1<f64>| Ok(flat),
        |_: &mut (), _: &Array1<f64>| {
            Err(EstimationError::InvalidInput(
                "legacy eager eval should not run".to_string(),
            ))
        },
        move |_: &mut (), _: &Array1<f64>, order: OuterEvalOrder| {
            let idx = calls.fetch_add(1, Ordering::Relaxed);
            let (cost, gradient) = table[idx.min(table.len() - 1)].clone();
            if publishes_band {
                crate::estimate::outer_eval_capture::record_certificate_inner_residual(
                    EXACT_INNER_MODE_3018,
                );
            }
            Ok(OuterEval {
                cost,
                gradient,
                hessian: match order {
                    OuterEvalOrder::ValueGradientHessian => HessianValue::Dense(array![[1.0]]),
                    _ => HessianValue::Unavailable,
                },
                inner_beta_hint: None,
            })
        },
        None::<fn(&mut ())>,
        None::<fn(&mut (), &Array1<f64>) -> Result<EfsEval, EstimationError>>,
    );
    let config = OuterConfig {
        problem_size: OuterProblemSize {
            n_obs: Some(100),
            p_coefficients: Some(4),
        },
        ..claim_band_config_2817(CLAIM_BAND_2817)
    };
    let exit: Arc<Mutex<Option<CostStallExit>>> = Arc::new(Mutex::new(None));
    let guard = CostStallGuard::new(RESOLUTION_2817, &config, exit.clone());
    let ledger: Arc<AcceptedStepLedger> = Arc::default();
    let mut bridge = OuterSecondOrderBridge {
        obj: &mut obj,
        layout: OuterThetaLayout::new(1, 0),
        hessian_source: HessianSource::Analytic,
        eval_count: 0,
        outer_inner_cap: None,
        g_norm_initial: None,
        last_g_norm: None,
        last_value_grad_rho: None,
        cost_stall: Some(guard),
        cost_stall_bounds: Some(wide_box_2817(1)),
        curvature_stationary_resolution: Some(RESOLUTION_2817),
        accepted_trials: AcceptedTrialGate::new(Arc::clone(&ledger)),
        decrement_verdict_config: None,
    };
    let mut outcomes = Vec::new();
    for iter in 0..schedule.len() {
        let outcome = SecondOrderObjective::eval_hessian(&mut bridge, &point).and_then(|sample| {
            report_accepted_trial_3017(&ledger, iter);
            bridge.settle_pending_trial().map_or(Ok(sample.value), Err)
        });
        match outcome {
            Ok(value) => outcomes.push(Ok(value)),
            Err(err) => {
                outcomes.push(Err(err.into_message()));
                break;
            }
        }
    }
    let published = exit.lock().expect("exit cell").take();
    (outcomes, published)
}

/// gam#3018: a step is progress when its decrease is resolvable against the two
/// values' evaluation bands, and a crawl whose every step the arithmetic resolves
/// is a search still descending. The guard used to count a decrease below the
/// relative floor `rel_tol·(1 + |V|)` as no improvement, so this crawl stopped as
/// unprogressing at its second window while every step bought a certified
/// decrease.
///
/// Where the evaluation publishes its band, `1e-9` per step is ten thousand
/// resolutions and the run never stalls. The CONTROL is the same crawl on an
/// evaluation that publishes nothing, whose resolution is the criterion's
/// `RESOLUTION_2817 = 1e-4`: there `1e-9` resolves nothing, the second stall bought
/// nothing, and the run stops. The two arms differ only in the band, which is
/// what makes the band the thing that decides.
#[test]
fn a_crawl_the_evaluations_band_resolves_is_never_stalled_3018() {
    let (resolved, published) = drive_arc_crawl_3018(crawl_3018(4 * SECOND_STALL_2817), true);
    assert!(
        resolved.iter().all(|outcome| outcome.is_ok()),
        "every step of the crawl buys a decrease its band resolves, so none may stall: \
         {resolved:?}"
    );
    assert!(
        published.is_some_and(|exit| !exit.converged),
        "a running crawl publishes only its best-so-far snapshot, never a convergence"
    );

    let (asserted, stopped) = drive_arc_crawl_3018(crawl_3018(4 * SECOND_STALL_2817), false);
    assert_eq!(
        asserted.len(),
        SECOND_STALL_2817,
        "under the asserted resolution the same crawl stalls twice and stops: {asserted:?}"
    );
    assert_eq!(
        asserted.last().expect("ran").clone().err().as_deref(),
        Some(ARC_UNPROGRESSING_STALL_SENTINEL),
        "the control stops on the unprogressing sentinel: {asserted:?}"
    );
    assert!(stopped.is_some_and(|exit| !exit.converged));
}

// ─── an unprogressing fixed-point walk stops ─────────────────────────────────

/// A fixed-point walk caught in a limit cycle stops at its first evaluation
/// that buys nothing (#2817, #3176).
///
/// The map alternates between two points, so its step norm is the constant
/// distance between them and it never improves on the first value. The first
/// evaluation sets the incumbent; the second neither improves on it beyond the
/// two values' rounding nor contracts the step, so the walk stops there.
/// Before, a cycling walk ran two 6-evaluation windows first, and before that
/// until its iteration count ran out.
#[test]
fn a_limit_cycling_fixed_point_walk_stops_at_its_first_unprogressing_evaluation_2817() {
    let mut progress = FixedPointProgress::new();
    assert!(
        !progress.observe(COST_2817, 0.5),
        "the first evaluation sets the incumbent"
    );
    assert!(
        progress.observe(COST_2817 + 1.0, 0.5),
        "the cycle's second point bought neither improvement nor contraction"
    );
}

/// NEGATIVE CONTROL: no evaluation improves the incumbent, but every step is
/// shorter than the one before it. The map is contracting, so the walk keeps
/// walking.
///
/// Once the step stops contracting, a value one ulp below the incumbent does
/// not carry the walk: at `|V| = 1e3` one ulp is `1.1e-13`, inside the two
/// values' rounding `γ₁|V_best| + γ₁|V| ≈ 2.2e-13`, so it is no resolved
/// improvement and the walk stops.
#[test]
fn a_fixed_point_walk_whose_step_contracted_keeps_walking_2817() {
    let mut progress = FixedPointProgress::new();
    for index in 0..40 {
        let step_norm = 0.5 * 0.5_f64.powi(index);
        let stopped = progress.observe(COST_2817, step_norm);
        assert!(
            !stopped,
            "evaluation {index}: a walk whose step contracts must keep walking"
        );
    }
    let one_ulp_below = f64::from_bits(COST_2817.to_bits() - 1);
    assert!(one_ulp_below < COST_2817);
    assert!(
        progress.observe(one_ulp_below, 0.5),
        "once the step stops contracting with no resolved improvement, the walk stops"
    );
}

/// NEGATIVE CONTROL: every evaluation improves the incumbent by a whole unit
/// at a constant step, so the walk keeps walking.
#[test]
fn a_fixed_point_walk_that_bought_resolved_improvement_keeps_walking_2817() {
    let mut progress = FixedPointProgress::new();
    for index in 0..40 {
        let stopped = progress.observe(COST_2817 - f64::from(index), 0.5);
        assert!(
            !stopped,
            "evaluation {index}: a walk that bought a resolved improvement must keep walking"
        );
    }
}

/// An improvement smaller than the criterion's statistical resolution `τ`, but
/// far beyond its values' rounding, is progress (#3176).
///
/// `τ = 1/(2n)` is the certificate's decrement tolerance: how much decrease may
/// be left at a certified point. It is not the arithmetic error of one value.
/// At `n = 80`, `τ = 6.25e-3`; a walk at `V ≈ 100` improving by `1e-3` per
/// evaluation at a constant step is still descending by about `10¹¹` times its
/// values' rounding. Charging each value `τ` stopped it at its second
/// evaluation (`1e-3 < 2τ` and no contraction); charging each value its own
/// rounding `γ₁|V|` keeps it walking.
#[test]
fn a_fixed_point_walk_improving_below_tau_but_above_rounding_keeps_walking_3176() {
    let tau = 0.5 / 80.0;
    let improvement = 1.0e-3;
    assert!(improvement < 2.0 * tau, "the separating case improves by less than 2τ");
    let mut progress = FixedPointProgress::new();
    for index in 0..40 {
        let stopped = progress.observe(100.0 - improvement * f64::from(index), 0.5);
        assert!(
            !stopped,
            "evaluation {index}: a walk improving by {improvement:e} per step, above its \
             values' rounding, must keep walking"
        );
    }
}

// ─── an exhausted budget refuses ─────────────────────────────────────────────

/// An exhausted ARC budget is a refusal that carries its iteration ledger, not a
/// retry (#2817).
///
/// `run_outer_uncertified` used to rerun an exhausted ARC plan from its last
/// iterate, up to twice, which only continued a search that had not converged
/// (SPEC rule 21) under an arbitrary retry count (rule 23). With the automatic
/// fallback ladder disabled there is one plan attempt, so what the runner hands
/// to the certificate must be exactly what one pass of that plan produced: the
/// same checkpoint, with the same iterations. A retry continues from that
/// checkpoint and moves it.
///
/// Every seed of the quartic `SCALE·(θ − OFFSET)⁴` exhausts `max_iter = 1`, since
/// no generated candidate sits at ½, so the plan pass returns `Exhausted`.
#[test]
fn an_exhausted_arc_budget_refuses_instead_of_retrying_2817() {
    const OFFSET: f64 = 0.5;
    const SCALE: f64 = 1.0e6;
    let problem = OuterProblem::new(1)
        .with_gradient(Derivative::Analytic)
        .with_hessian(DeclaredHessianForm::Either)
        .with_initial_rho(array![5.0])
        .with_max_iter(1)
        .with_fallback_policy(FallbackPolicy::Disabled);
    let cost = |_: &mut (), theta: &Array1<f64>| -> Result<f64, EstimationError> {
        Ok(SCALE * (theta[0] - OFFSET).powi(4))
    };
    let eval = |_: &mut (), theta: &Array1<f64>| -> Result<OuterEval, EstimationError> {
        let d = theta[0] - OFFSET;
        Ok(OuterEval {
            cost: SCALE * d.powi(4),
            gradient: array![SCALE * 4.0 * d.powi(3)],
            hessian: HessianValue::Dense(array![[SCALE * 12.0 * d.powi(2)]]),
            inner_beta_hint: None,
        })
    };
    let config = problem.config();
    let context = "exhausted ARC budget #2817";

    let mut plan_obj = problem.build_objective(
        (),
        cost,
        eval,
        None::<fn(&mut ())>,
        None::<fn(&mut (), &Array1<f64>) -> Result<EfsEval, EstimationError>>,
    );
    let capability = super::super::super::capability::primary_capability_for_config(
        plan_obj.capability(),
        &config,
        context,
    );
    let the_plan = plan(&capability);
    assert_eq!(
        the_plan.solver,
        Solver::Arc,
        "fixture precondition: the quartic plans ARC"
    );
    let one_pass =
        match run_outer_with_plan(&mut plan_obj, &config, context, &capability, &the_plan, true)
            .expect("one plan pass over the quartic returns an outcome")
        {
            PlanRunOutcome::Exhausted(checkpoint) => checkpoint,
            PlanRunOutcome::Converged(result) => panic!(
                "fixture precondition: no seed reaches the optimum at {OFFSET} within one \
                 iteration, but the pass converged at rho={:?}",
                result.rho
            ),
            PlanRunOutcome::FirstOrderFallbackRequested(request) => {
                panic!("an ARC pass requested a first-order fallback: {}", request.reason())
            }
            PlanRunOutcome::FixedPointContinuationRequested(request) => {
                panic!("an ARC pass requested a fixed-point continuation: {}", request.refusal)
            }
            PlanRunOutcome::DominatedPlateau(dominated) => panic!(
                "fixture precondition: one pass over the quartic returns its exhausted \
                 checkpoint, but it declined a dominated certified winner at cost {:.6e}",
                dominated.plateau.final_value
            ),
        };

    let mut runner_obj = problem.build_objective(
        (),
        cost,
        eval,
        None::<fn(&mut ())>,
        None::<fn(&mut (), &Array1<f64>) -> Result<EfsEval, EstimationError>>,
    );
    let refused =
        super::super::super::run::run_outer_uncertified(&mut runner_obj, &config, context)
            .expect("an exhausted ARC budget hands its checkpoint to the certificate");
    eprintln!(
        "[#2817 exhausted budget] one pass: rho={:?} value={:e} iterations={}; runner: \
         rho={:?} value={:e} iterations={}",
        one_pass.rho,
        one_pass.final_value,
        one_pass.iterations,
        refused.rho,
        refused.final_value,
        refused.iterations,
    );
    assert!(
        !refused.solver_claimed_convergence(),
        "an exhausted budget makes no convergence claim"
    );
    assert_eq!(
        refused.rho, one_pass.rho,
        "the runner moved the checkpoint one plan pass produced, so it continued the search"
    );
    assert_eq!(
        refused.final_value.to_bits(),
        one_pass.final_value.to_bits(),
        "the runner's checkpoint value differs from the one plan pass"
    );
    assert_eq!(
        refused.iterations, one_pass.iterations,
        "the runner spent iterations beyond the one plan pass"
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
        regularization: None,
        line_search_step: None,
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

/// The criterion value the no-size fixture sits at, large enough that its own
/// representation band `γ₁·|V| ≈ 1.1e-5` puts the flipping gradient above the
/// claim band [`CLAIM_BAND_2817`], so the stall is the guard's cost verdict and not
/// its stationary claim.
const COST_3286: f64 = 1.0e11;

/// A ROUTE THAT DECLARES NO SIZE STILL STOPS, AT ITS ARITHMETIC RESOLUTION
/// (#3286). With no observation count the criterion has no statistical slack,
/// `τ_stat = 0`, and the online stop decides at the band of the value it
/// evaluated: with no evidence published, `γ₁·|V|`. Reading the bare `0` there
/// switched the stop off, and a no-size ARC search ran on to its reject floor
/// (#3286's `100·eʳ − 1e5·ρ`). The decrement flips the verdict where it crosses
/// that band, 1% either side, as it crosses `τ_stat` on a sized route
/// (`the_adjudication_threshold_is_the_criterion_resolution_2817`).
#[test]
fn a_route_that_declares_no_size_stops_at_its_arithmetic_resolution_3286() {
    let band = crate::rho_optimizer::decrement_bands::value_representation_band(COST_3286);
    let critical = (2.0 * band * (1.0 + f64::EPSILON.sqrt())).sqrt();
    assert!(
        critical > CLAIM_BAND_2817,
        "fixture premise: the flipping gradient {critical:.3e} must sit above the claim band"
    );
    let flat = |gradient: f64| {
        (0..FIRST_STALL_2817)
            .map(|_| (COST_3286, array![gradient]))
            .collect::<Vec<_>>()
    };
    let (inside, _) = drive_arc_oracle_2817(
        array![0.5],
        flat(critical * 0.99),
        array![[1.0]],
        wide_box_2817(1),
        Some(0.0),
    );
    assert_eq!(
        inside.last().expect("ran").clone().err().as_deref(),
        Some(ARC_CURVATURE_STATIONARY_SENTINEL),
        "a decrement 1% inside the value's own band {band:.3e} must end the stall on a route \
         with no declared size"
    );
    let (outside, _) = drive_arc_oracle_2817(
        array![0.5],
        flat(critical * 1.01),
        array![[1.0]],
        wide_box_2817(1),
        Some(0.0),
    );
    assert!(
        outside.iter().all(|o| o.is_ok()),
        "a decrement 1% outside the value's own band must not end the stall: {outside:?}"
    );
}

/// ONE RESOLUTION, BOTH ARMS (#3286, #3192). The online stop decides the decrease
/// left at `outer_resolution(τ, b) = max(τ − b, b)`, with `b` the band the
/// evaluated value's own evidence forms: `τ − b` while the band is small against
/// `τ`, and `b` itself once it exceeds `τ/2`, where the arithmetic cannot resolve
/// the statistical standard. The same fixture with the band on either side of
/// `τ/2` flips its stop where the decrement crosses each arm, 1% either side.
#[test]
fn the_online_stop_decides_on_both_arms_of_the_one_resolution_3286() {
    use crate::estimate::outer_eval_capture as capture;
    use crate::rho_optimizer::decrement_bands::{outer_resolution, value_representation_band};
    let config = claim_band_config_2817(CLAIM_BAND_2817);
    let stop = |inner_residual: f64, gradient: f64| {
        let evidence = capture::CertificateEvidence {
            inner_residual: Some(capture::InnerResidualCharge {
                energy: inner_residual,
                source: capture::InnerResidualSource::InnerGradient,
            }),
            ..capture::CertificateEvidence::default()
        };
        let samples = flatlined_2817(array![gradient], FIRST_STALL_2817);
        let (outcomes, _) = drive_arc_oracle_publishing_2817(
            vec![array![0.5]; samples.len()],
            samples,
            array![[1.0]],
            wide_box_2817(1),
            Some(RESOLUTION_2817),
            |_| COST_2817,
            Some((&config, evidence)),
        );
        outcomes.last().expect("ran").clone().err().as_deref()
            == Some(ARC_CURVATURE_STATIONARY_SENTINEL)
    };
    for (arm, inner_residual) in [("τ − b", 1.0e-6), ("b", 0.8 * RESOLUTION_2817)] {
        let band = value_representation_band(COST_2817) + inner_residual;
        let resolution = outer_resolution(RESOLUTION_2817, band);
        let expected = if band < 0.5 * RESOLUTION_2817 {
            RESOLUTION_2817 - band
        } else {
            band
        };
        assert_eq!(
            resolution, expected,
            "fixture premise: the {arm} arm decides"
        );
        let critical = (2.0 * resolution * (1.0 + f64::EPSILON.sqrt())).sqrt();
        assert!(
            stop(inner_residual, critical * 0.99),
            "a decrement 1% inside the {arm} arm {resolution:.3e} must end the stall"
        );
        assert!(
            !stop(inner_residual, critical * 1.01),
            "a decrement 1% outside the {arm} arm {resolution:.3e} must not end the stall"
        );
    }
}
