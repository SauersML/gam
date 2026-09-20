use super::*;
use crate::rho_optimizer::rail_face::RailFaceLimit;
use crate::rho_optimizer::zero_smoothing_face::{ZeroSmoothingFace, ZeroSmoothingFaceOutcome};
use ndarray::array;

/// The problem size every fixture below declares: `n = 100` observations and
/// `p = 5` coefficients. It fixes the criterion's statistical resolution
/// `1/(2n) = 5e-3` ([`outer_criterion_resolution`]) and the formation count
/// `m = n + p²` the probes' rounding bands are charged at
/// ([`outer_coordinate_bands`]).
fn test_config() -> OuterConfig {
    OuterConfig {
        problem_size: OuterProblemSize {
            n_obs: Some(100),
            p_coefficients: Some(5),
        },
        ..OuterConfig::default()
    }
}

/// Publish coordinate 0's gradient parts the way the REML assembly does beside
/// a rail: the component `g` is the small difference of an `O(scale)` penalty
/// channel and an `O(scale)` log-determinant channel, so its rounding bound is
/// set by `scale`, not by `|g|`. The `e^{±ρ}` amplification of that bound in
/// the pencil constant is what makes the probes nearest a rail unresolved.
fn publish_cancelling_parts(rho: f64, g: f64, scale: f64) {
    crate::estimate::outer_eval_capture::record_certificate_parts(&[
        crate::estimate::outer_eval_capture::RhoGradientParts {
            index: 0,
            lambda: rho.exp(),
            block_quadratic: 0.0,
            rank: 1,
            dim: 1,
            fixed_beta: 0.5 * scale,
            logdet_h: g - 0.5 * scale,
            frozen_logdet_h: 0.0,
            mode_response_logdet_h: 0.0,
            logdet_s: 0.0,
            total: g,
        },
    ]);
}

/// A proven λ=∞ face certifies on the proof alone, whatever the depth of the
/// shipped rail.
///
/// The fixture's criterion IS the face law: `V(ρ) = 2e^{−ρ}`, which is
/// `½tr((λ·diag(2,3))⁻¹·diag(4,6))` with `V_∞ = 0`. At `ρ̂ = 12` the KKT
/// statistic is positive and the shipped fit is `≈ 5e−6` from the limit fit,
/// so the face is proven and reached. The retired value probe refused it
/// anyway: its "ideal" pull-back `½(ln(tol/gap) + ρ̂) = 0.14` e-folds fell
/// under its one-e-fold floor ("no room inside the box to falsify the face
/// law"), and the rail went to the finite-difference tail ladder. The proof
/// must mint without spending a criterion evaluation: the value still
/// available on the face, `2e^{−12} ≈ 1.2e−5`, is far below the criterion's
/// resolution `1/(2n) = 5e−3`.
#[test]
fn proven_face_certifies_a_shallow_rail_without_a_value_probe() {
    let rho_hat = 12.0_f64;
    let limit = RailFaceLimit {
        face: vec![0],
        face_rho: vec![rho_hat],
        first_order_form: Array2::from_diag(&array![4.0, 6.0]),
        released_penalties: vec![Array2::from_diag(&array![2.0, 3.0])],
        released_score: array![1.0, 2.0],
        form_error_bound: 0.0,
        limit_beta: Array1::zeros(0),
        limit_dispersion: 1.0,
        released_curvature_drift: None,
    };
    let problem = OuterProblem::new(1).with_gradient(Derivative::Analytic);
    let mut obj = problem
        .build_objective(
            Vec::<f64>::new(),
            |seen: &mut Vec<f64>, rho: &Array1<f64>| {
                seen.push(rho[0]);
                Ok(2.0 * (-rho[0]).exp())
            },
            |seen: &mut Vec<f64>, rho: &Array1<f64>| {
                seen.push(rho[0]);
                Ok(OuterEval {
                    cost: 2.0 * (-rho[0]).exp(),
                    gradient: array![-2.0 * (-rho[0]).exp()],
                    hessian: HessianValue::Unavailable,
                    inner_beta_hint: None,
                })
            },
            None::<fn(&mut Vec<f64>)>,
            None::<fn(&mut Vec<f64>, &Array1<f64>) -> Result<EfsEval, EstimationError>>,
        )
        .with_rail_face_limit(move |_: &mut Vec<f64>, _: &Array1<f64>, face: &[usize]| {
            assert_eq!(face, &[0]);
            Ok(RailFaceLimitOutcome::Available(Box::new(limit.clone())))
        });
    let rho = array![rho_hat];
    // The rail's whole pull is the blocked KKT multiplier, so the projection
    // zeroed it, exactly as it does for the railed rows of the tests below.
    let gradient = array![0.0];
    let hessian = array![[2.0 * (-rho_hat).exp()]];
    let bounds = (array![-30.0], array![30.0]);
    let config = test_config();
    let inputs = AsymptoteRailInputs {
        rho: &rho,
        projected_gradient: &gradient,
        railed: &[0],
        layout: OuterThetaLayout::new(1, 0),
        hessian: &hessian,
        bounds: &bounds,
        stationarity_bound: StationarityBound::from_ladder(1.0e-6, StationarityBoundSource::SolverBand),
        objective_tol: outer_criterion_resolution(&config),
        context: "proven face on a shallow rail",
        config: &config,
    };

    let (_, _, rails) = try_certify_asymptote_rail(&mut obj, &inputs)
        .expect("certification must not error")
        .expect("a proven, reached λ=∞ face must certify");
    assert_eq!(rails.len(), 1);
    assert_eq!(rails[0].index, 0);
    assert_eq!(rails[0].side, AsymptoteSide::Upper);
    assert!(
        matches!(rails[0].evidence, RailTailEvidence::AnalyticFaceProof { .. }),
        "the rail must carry the face proof, not a measured tail: {:?}",
        rails[0].evidence
    );
    assert!(
        rails[0].evidence.admits(rails[0].tail_constant),
        "the proof's own well-formedness rule must admit its tail constant: {:?}",
        rails[0]
    );
    assert!(
        (rails[0].tail_constant - 2.0).abs() <= 1.0e-12,
        "the face law's tail constant is ½tr(diag(2,3)⁻¹diag(4,6)) = 2, got {}",
        rails[0].tail_constant
    );
    assert!(
        obj.state.is_empty(),
        "the proof spends no criterion evaluation, but the criterion was evaluated at {:?}",
        obj.state
    );
}

/// A one-coordinate objective whose criterion IS a covered zero-smoothing
/// law, `V(ρ) = c′·e^{ρ}`, recording every criterion evaluation it is asked
/// for. `law` installs the zero-smoothing hook; `None` leaves the objective
/// without one.
fn zero_smoothing_objective(
    slope: f64,
    law: Option<ZeroSmoothingFace>,
) -> impl OuterObjective + HasEvaluationLog {
    let problem = OuterProblem::new(1).with_gradient(Derivative::Analytic);
    let obj = problem.build_objective(
        Vec::<f64>::new(),
        move |seen: &mut Vec<f64>, rho: &Array1<f64>| {
            seen.push(rho[0]);
            Ok(slope * rho[0].exp())
        },
        move |seen: &mut Vec<f64>, rho: &Array1<f64>| {
            seen.push(rho[0]);
            Ok(OuterEval {
                cost: slope * rho[0].exp(),
                gradient: array![slope * rho[0].exp()],
                hessian: HessianValue::Unavailable,
                inner_beta_hint: None,
            })
        },
        None::<fn(&mut Vec<f64>)>,
        None::<fn(&mut Vec<f64>, &Array1<f64>) -> Result<EfsEval, EstimationError>>,
    );
    match law {
        Some(law) => obj.with_zero_smoothing_face(
            move |_: &mut Vec<f64>, _: &Array1<f64>, face: &[usize]| {
                assert_eq!(face, &[0]);
                Ok(ZeroSmoothingFaceOutcome::Available(Box::new(law.clone())))
            },
        ),
        None => obj,
    }
}

trait HasEvaluationLog {
    fn evaluations(&self) -> &[f64];
}

impl<Fc, Fe, Fr, Fefs, Feo, Fseed> HasEvaluationLog
    for ClosureObjective<Vec<f64>, Fc, Fe, Fr, Fefs, Feo, Fseed>
{
    fn evaluations(&self) -> &[f64] {
        &self.state
    }
}

fn lower_rail_inputs<'a>(
    rho: &'a Array1<f64>,
    gradient: &'a Array1<f64>,
    hessian: &'a Array2<f64>,
    bounds: &'a (Array1<f64>, Array1<f64>),
    config: &'a OuterConfig,
) -> AsymptoteRailInputs<'a> {
    AsymptoteRailInputs {
        rho,
        projected_gradient: gradient,
        railed: &[0],
        layout: OuterThetaLayout::new(1, 0),
        hessian,
        bounds,
        stationarity_bound: StationarityBound::from_ladder(1.0e-6, StationarityBoundSource::SolverBand),
        objective_tol: outer_criterion_resolution(config),
        context: "zero-smoothing rail",
        config,
    }
}

/// A proven λ=0 face certifies on its law alone (#2348 Inc 5, lower face).
///
/// The criterion is the covered first-order law `V = c′·λ` with `c′ = 3`,
/// railed at `ρ̂ = −12`. The zero-smoothing face was previously declined by
/// the analytic route ("covers λ→∞ only") and handed to the measured-tail
/// ladder, which never minted a lower rail across the regression suite. The
/// law must now mint a LOWER rail whose pencil constant is the slope itself,
/// carrying the covered-zero-smoothing proof, without spending a criterion
/// evaluation.
#[test]
fn proven_zero_smoothing_face_certifies_a_lower_rail_without_probing() {
    let rho_hat = -12.0_f64;
    let slope = 3.0_f64;
    let law = ZeroSmoothingFace {
        face: vec![0],
        face_rho: vec![rho_hat],
        slopes: vec![slope],
        slope_bands: vec![1.0e-12],
        limit_beta: Array1::zeros(0),
        limit_dispersion: 1.0,
        estimand_travel: 0.0,
    };
    let mut obj = zero_smoothing_objective(slope, Some(law));
    let rho = array![rho_hat];
    let gradient = array![0.0];
    let hessian = array![[slope * rho_hat.exp()]];
    let bounds = (array![-30.0], array![30.0]);
    let config = test_config();
    let inputs = lower_rail_inputs(&rho, &gradient, &hessian, &bounds, &config);

    let (_, _, rails) = try_certify_asymptote_rail(&mut obj, &inputs)
        .expect("certification must not error")
        .expect("a proven, reached λ=0 face must certify");
    assert_eq!(rails.len(), 1);
    assert_eq!(rails[0].side, AsymptoteSide::Lower);
    assert!(
        matches!(
            rails[0].evidence,
            RailTailEvidence::AnalyticFaceProof {
                route: FacePositivityRoute::CoveredZeroSmoothing,
                ..
            }
        ),
        "the rail must carry the zero-smoothing proof: {:?}",
        rails[0].evidence
    );
    assert!(rails[0].evidence.admits(rails[0].tail_constant));
    assert_eq!(rails[0].tail_constant, slope);
    let expected_gap = slope * rho_hat.exp();
    assert!(
        (rails[0].value_gap - expected_gap).abs() <= 1.0e-15 * expected_gap,
        "the value gap is c′·λ = {expected_gap:.6e}, got {:.6e}",
        rails[0].value_gap
    );
    assert!(
        obj.evaluations().is_empty(),
        "the proof spends no criterion evaluation, but the criterion was evaluated at {:?}",
        obj.evaluations()
    );
}

/// A zero-smoothing rail with no law is refused outright, without probing:
/// a measured λ → 0 tail cannot turn a barrier, an exact fit, or a criterion
/// outside the closed form into a minimizer, and it never minted a rail.
#[test]
fn zero_smoothing_rail_without_a_law_is_refused_without_probing() {
    let rho_hat = -12.0_f64;
    let mut obj = zero_smoothing_objective(3.0, None);
    let rho = array![rho_hat];
    let gradient = array![0.0];
    let hessian = array![[3.0 * rho_hat.exp()]];
    let bounds = (array![-30.0], array![30.0]);
    let config = test_config();
    let inputs = lower_rail_inputs(&rho, &gradient, &hessian, &bounds, &config);

    let refusal = try_certify_asymptote_rail(&mut obj, &inputs)
        .expect("certification must not error")
        .expect_err("a lower rail with no law has nothing to certify it");
    assert!(
        refusal.contains("zero-smoothing"),
        "the refusal must name the missing law: {refusal}"
    );
    assert!(
        obj.evaluations().is_empty(),
        "no tail may be probed at the zero-smoothing end, but the criterion was evaluated at {:?}",
        obj.evaluations()
    );
}

/// Evaluation counter shared between a fixture's gradient closure and the test.
fn eval_counter() -> (
    std::sync::Arc<std::sync::atomic::AtomicUsize>,
    std::sync::Arc<std::sync::atomic::AtomicUsize>,
) {
    let count = std::sync::Arc::new(std::sync::atomic::AtomicUsize::new(0));
    let in_eval = std::sync::Arc::clone(&count);
    (count, in_eval)
}

/// Build a one-coordinate UPPER-rail tail-law objective at criterion scale
/// `scale`: at ρ its gradient is `−scale·c_eff(ρ)·e^{−ρ}` with
/// `c_eff = c + drift_amp·ρ` (so `ĉ = −e^{ρ}·grad = scale·c_eff`), its value
/// carries the additive constant `offset`, and its published inner β is
/// `a·e^{−ρ}` (so consecutive-probe `‖Δβ‖` contracts by `e^{−1}`).
/// `drift_amp ≠ 0` models a pencil constant that keeps moving by the same
/// amount every e-fold: no tail law, so nothing may settle.
fn upper_tail_objective(
    c: f64,
    a: f64,
    drift_amp: f64,
    scale: f64,
    offset: f64,
    evaluations: std::sync::Arc<std::sync::atomic::AtomicUsize>,
) -> impl OuterObjective {
    let problem = OuterProblem::new(1).with_gradient(Derivative::Analytic);
    let value = move |r: f64| offset + scale * ((c + drift_amp * r) * (-r).exp()).abs();
    problem.build_objective(
        (),
        move |_: &mut (), rho: &Array1<f64>| Ok(value(rho[0])),
        move |_: &mut (), rho: &Array1<f64>| {
            evaluations.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
            let r = rho[0];
            let g = -scale * (c + drift_amp * r) * (-r).exp();
            publish_cancelling_parts(r, g, scale);
            Ok(OuterEval {
                cost: value(r),
                gradient: array![g],
                hessian: HessianValue::Unavailable,
                inner_beta_hint: Some(array![a * (-r).exp()]),
            })
        },
        None::<fn(&mut ())>,
        None::<fn(&mut (), &Array1<f64>) -> Result<EfsEval, EstimationError>>,
    )
}

/// An exact upper-rail exponential tail is certified from the first three
/// probes: every pencil constant is `c` to rounding, so the settlement radius
/// is the derived rounding band of the constants (`e^{ρ}·ε`, `≈ 0.05` at one
/// e-fold from the rail) and the certificate carries it as its evidence.
#[test]
fn asymptote_rail_mints_on_exact_tail_law() {
    let config = test_config();
    let (count, in_eval) = eval_counter();
    let mut obj = upper_tail_objective(6723.0, 1.0, 0.0, 1.0, 0.0, in_eval);
    let rho = array![29.9];
    let rail = build_and_assess_rail_coordinate(
        &mut obj,
        &rho,
        0,
        AsymptoteSide::Upper,
        &config,
        outer_criterion_resolution(&config),
        (-30.0, 30.0),
    )
    .expect("probing the tail-law objective must not error")
    .expect("an exact exponential tail must certify a rail");
    assert_eq!(rail.index, 0);
    assert_eq!(rail.side, AsymptoteSide::Upper);
    assert!(
        (rail.tail_constant - 6723.0).abs() / 6723.0 < 1.0e-6,
        "recovered ĉ={} should equal c=6723",
        rail.tail_constant,
    );
    let RailTailEvidence::ProbedTail {
        gradient_band,
        extrapolation_radius,
    } = rail.evidence
    else {
        panic!("a measured tail must carry probed evidence: {:?}", rail.evidence);
    };
    assert!(gradient_band > 0.0 && gradient_band.is_finite());
    assert!(
        (rail.tail_constant - 6723.0).abs() <= extrapolation_radius,
        "the settlement radius {extrapolation_radius} must cover the true constant"
    );
    assert!(
        extrapolation_radius < 1.0,
        "on an exact tail the radius is the constants' rounding band alone, got \
         {extrapolation_radius}"
    );
    assert!(rail.evidence.admits(rail.tail_constant));
    assert!(rail.value_gap.is_finite() && rail.value_gap >= 0.0);
    assert!(rail.value_gap <= outer_criterion_resolution(&config));
    assert!(rail.estimand_travel_bound.is_finite() && rail.estimand_travel_bound >= 0.0);
    assert_eq!(
        count.load(std::sync::atomic::Ordering::Relaxed),
        3,
        "three resolved probes that settle are a complete certificate"
    );
}

/// A pencil constant that moves by the same amount every e-fold never
/// certifies: its steps do not contract, so no window on the ladder settles,
/// down to the midpoint of the box where the ladder ends.
#[test]
fn asymptote_rail_refuses_on_drifting_constant() {
    let config = test_config();
    let (count, in_eval) = eval_counter();
    let mut obj = upper_tail_objective(6723.0, 1.0, 3000.0, 1.0, 0.0, in_eval);
    let rho = array![29.9];
    let verdict = build_and_assess_rail_coordinate(
        &mut obj,
        &rho,
        0,
        AsymptoteSide::Upper,
        &config,
        outer_criterion_resolution(&config),
        (-30.0, 30.0),
    )
    .expect("probing must not error");
    let reason = verdict.expect_err("a drifting ĉ must not certify a tail");
    assert!(
        reason.contains("no settled tail window"),
        "the decline must name the settlement, got: {reason}"
    );
    assert_eq!(
        count.load(std::sync::atomic::Ordering::Relaxed),
        rail_probe_ladder(29.9, AsymptoteSide::Upper, (-30.0, 30.0))
            .expect("finite box")
            .len(),
        "an unsettled tail walks the whole ladder to the box midpoint"
    );
}

/// #2358: a finite smoothing box can expose only a few e-folds of the
/// leading-order tail. For
///
/// `V(ρ) = c·e⁻ρ + (d/2)·e⁻²ρ`,
///
/// the pencil constant is `ĉ(ρ) = c + d·e⁻ρ`: the asymptotic law plus its
/// first vanishing correction. The retired drift band read the correction as
/// drift and refused. The settlement reads it as what it is: consecutive
/// constants differ by `d(e^{−ρ} − e^{−ρ−1})`, contracting by exactly `e^{−1}`
/// toward the rail, and summing that series bounds the distance from the
/// rail-most constant to `c`. The first three unit probes from ρ̂ = 10
/// (ρ = 9, 8, 7) certify, and the radius covers the true limit `c = 7.1`.
#[test]
fn tail_probe_resolves_narrow_regular_band_before_finite_box_2358() {
    let (c, d, a) = (7.1_f64, 400.0_f64, 1.0e-3_f64);
    let config = test_config();
    let (count, in_eval) = eval_counter();
    let problem = OuterProblem::new(1).with_gradient(Derivative::Analytic);
    let mut obj = problem.build_objective(
        (),
        move |_: &mut (), rho: &Array1<f64>| {
            let tau = (-rho[0]).exp();
            Ok(c * tau + 0.5 * d * tau * tau)
        },
        move |_: &mut (), rho: &Array1<f64>| {
            in_eval.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
            let tau = (-rho[0]).exp();
            let g = -c * tau - d * tau * tau;
            publish_cancelling_parts(rho[0], g, 1.0);
            Ok(OuterEval {
                cost: c * tau + 0.5 * d * tau * tau,
                gradient: array![g],
                hessian: HessianValue::Unavailable,
                inner_beta_hint: Some(array![a * tau]),
            })
        },
        None::<fn(&mut ())>,
        None::<fn(&mut (), &Array1<f64>) -> Result<EfsEval, EstimationError>>,
    );
    let rho = array![10.0];
    let rail = build_and_assess_rail_coordinate(
        &mut obj,
        &rho,
        0,
        AsymptoteSide::Upper,
        &config,
        outer_criterion_resolution(&config),
        (-10.0, 10.5),
    )
    .expect("probing must not error")
    .unwrap_or_else(|reason| panic!("the regular tail must settle and certify: {reason}"));
    let RailTailEvidence::ProbedTail {
        extrapolation_radius,
        ..
    } = rail.evidence
    else {
        panic!("a measured tail must carry probed evidence: {:?}", rail.evidence);
    };
    assert!(
        (rail.tail_constant - c).abs() <= extrapolation_radius,
        "the settlement ĉ={} ± {extrapolation_radius} must cover the true constant {c}",
        rail.tail_constant
    );
    assert!(
        extrapolation_radius < 0.1,
        "the radius is the remaining correction d·e^{{−9}} ≈ 0.049, got {extrapolation_radius}"
    );
    assert!(rail.evidence.admits(rail.tail_constant));
    assert_eq!(count.load(std::sync::atomic::Ordering::Relaxed), 3);
}

/// A one-coordinate objective `V(ρ) = sign·c·e^{−ρ}` whose gradient closure
/// counts its evaluations and publishes its parts.
fn signed_upper_tail_objective(
    signed_c: f64,
    evaluations: std::sync::Arc<std::sync::atomic::AtomicUsize>,
) -> impl OuterObjective {
    let problem = OuterProblem::new(1).with_gradient(Derivative::Analytic);
    problem.build_objective(
        (),
        move |_: &mut (), rho: &Array1<f64>| Ok(signed_c * (-rho[0]).exp()),
        move |_: &mut (), rho: &Array1<f64>| {
            evaluations.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
            let g = -signed_c * (-rho[0]).exp();
            publish_cancelling_parts(rho[0], g, 1.0);
            Ok(OuterEval {
                cost: signed_c * (-rho[0]).exp(),
                gradient: array![g],
                hessian: HessianValue::Unavailable,
                inner_beta_hint: Some(array![(-rho[0]).exp()]),
            })
        },
        None::<fn(&mut ())>,
        None::<fn(&mut (), &Array1<f64>) -> Result<EfsEval, EstimationError>>,
    )
}

/// #2392 wrong-rail pull-back FIRES: a coordinate sitting at the UPPER bound
/// whose resolved probes carry a POSITIVE gradient (`∂V/∂ρ > 0`, so the
/// pencil constant `ĉ = −e^{ρ}·g < 0` — descent points INWARD, away from the
/// bound) was driven to the wrong rail. `detect_wrong_rail_pullback` returns
/// the deepest probe of the deciding run as the interior reseed target.
#[test]
fn wrong_rail_pullback_fires_on_inward_descent_2392() {
    // V(ρ) = −c·e^{−ρ} ⇒ ∂V/∂ρ = +c·e^{−ρ} > 0: the descent runs ρ DOWN, away
    // from the upper rail, and ĉ_upper = −e^{ρ}·(c·e^{−ρ}) = −c < 0 uniformly.
    let config = test_config();
    let (count, in_eval) = eval_counter();
    let mut obj = signed_upper_tail_objective(-6723.0, in_eval);
    let rho = array![29.9];
    let target = detect_wrong_rail_pullback(
        &mut obj,
        &rho,
        0,
        AsymptoteSide::Upper,
        &config,
        (-30.0, 30.0),
    )
    .expect("probing the wrong-rail objective must not error")
    .expect("an inward-descent rail must publish a pull-back target");
    assert!(
        (target - 26.9).abs() <= 1.0e-12,
        "the reseed is the deciding run's deepest probe ρ̂ − 3 = 26.9, got {target}",
    );
    assert_eq!(
        count.load(std::sync::atomic::Ordering::Relaxed),
        3,
        "three resolved rows whose constants settle away from zero are a complete \
         wrong-rail proof"
    );
}

/// #2392 wrong-rail pull-back does NOT fire on a GENUINE upper-rail tail:
/// `∂V/∂ρ < 0` ⇒ `ĉ > 0` ⇒ descent runs TOWARD the bound (a real λ→∞ optimum),
/// which must never be pulled off its rail.
#[test]
fn wrong_rail_pullback_refuses_a_genuine_upper_tail_2392() {
    let config = test_config();
    let (count, in_eval) = eval_counter();
    let mut obj = signed_upper_tail_objective(6723.0, in_eval);
    let rho = array![29.9];
    let verdict = detect_wrong_rail_pullback(
        &mut obj,
        &rho,
        0,
        AsymptoteSide::Upper,
        &config,
        (-30.0, 30.0),
    )
    .expect("probing must not error");
    assert!(
        verdict.is_none(),
        "a genuine λ→∞ tail (ĉ>0) must not be pulled off its rail, got {verdict:?}",
    );
    assert_eq!(
        count.load(std::sync::atomic::Ordering::Relaxed),
        3,
        "the first settled run proves a genuine rail; probing further interior \
         points cannot change that local fact"
    );
}

/// #2349: the interior-PSD gate must judge curvature above the
/// gradient-residue noise floor. Fixture = the measured multinomial
/// checkpoint shape: excluded tail candidates {0}, interior coordinate 1
/// gradient-stationary (|g| = 1.0228e-3) with the corrupted tie-signature
/// diagonal H₁₁ = −1.0216e-3 ≈ −|g₁| (the #2298 trace-pair residue — the
/// entire measured 6×6 spectrum was PSD except this one sub-resolution
/// entry). The raw gate refuses on the residue; the floored gate
/// certifies; a GENUINE interior saddle (λ_min = −0.5 against the same
/// tiny gradient) still refuses under the floor.
#[test]
fn interior_psd_gate_floors_tail_residue_but_keeps_genuine_saddles_2349() {
    let hessian = array![[0.2828, 0.0004], [0.0004, -1.0216e-3]];
    let gradient = array![-1.057, -1.0228e-3];
    let excluded = [0usize];
    assert_eq!(
        certificate_hessian_is_psd_off_railed(&hessian, &excluded, None),
        Some(false),
        "raw gate must see the corrupted sub-resolution entry as indefinite"
    );
    assert_eq!(
        certificate_hessian_is_psd_off_railed_above_gradient_floor(
            &hessian, &excluded, &gradient, None
        ),
        Some(true),
        "the gradient floor must absorb the O(|g|) trace-pair residue"
    );
    let saddle = array![[0.2828, 0.0004], [0.0004, -0.5]];
    assert_eq!(
        certificate_hessian_is_psd_off_railed_above_gradient_floor(
            &saddle, &excluded, &gradient, None
        ),
        Some(false),
        "a genuine interior saddle dwarfs the bound-scale floor and refuses"
    );
}

/// THE GUARANTEE, not an instance of it. Weyl bounds the floored spectrum by
/// `λ_min(H) + min|g| ≤ λ_min(H + diag|g|) ≤ λ_min(H) + max|g|`, so the floor
/// can absorb AT MOST `max_k |g_k|` over the JUDGED coordinates. Therefore
/// any interior spectrum whose most negative direction exceeds that floor
/// must still refuse — for every such spectrum, not merely for the one
/// saddle that happened to be measured.
///
/// Swept over curvatures spanning six orders and gradients spanning four,
/// including the pair where they are within a factor of two of each other
/// (the regime the floor exists to serve, and the only place the verdict is
/// genuinely close). The excluded coordinate carries a deliberately huge
/// gradient: the sub-block is extracted AFTER flooring, so it must never
/// reach the floor.
#[test]
fn gradient_floor_absorbs_at_most_max_interior_gradient_weyl_bound() {
    let excluded = [0usize];
    for &lambda_min in &[-5.0e-1, -1.5e-2, -1.0e-3, -1.0e-5, -1.0e-7] {
        for &g_interior in &[1.0e-7, 1.0e-5, 1.0e-3, 1.0e-2] {
            // The railed coordinate's gradient is four orders above every
            // interior one; if it ever entered the floor the sweep would
            // certify everything.
            let gradient = array![-1.4017, g_interior];
            let hessian = array![[0.2828, 0.0004], [0.0004, lambda_min]];
            let floored = certificate_hessian_is_psd_off_railed_above_gradient_floor(
                &hessian, &excluded, &gradient, None,
            );
            if g_interior < lambda_min.abs() {
                assert_eq!(
                    floored,
                    Some(false),
                    "Weyl: max|g_int|={g_interior:.1e} < |λ_min|={:.1e} means the floored \
                     sub-block is still indefinite, so the gate MUST refuse",
                    lambda_min.abs()
                );
            }
            // And the recorded clearance must report the same verdict
            // against the same floor, so the certificate's evidence and its
            // gate can never disagree.
            let clearance =
                interior_curvature_floor_clearance(&hessian, &excluded, &gradient, None)
                    .expect("a finite 1×1 interior sub-block has a clearance");
            assert_eq!(
                clearance.gradient_floor, g_interior,
                "the floor must be the largest JUDGED gradient — the excluded \
                 coordinate's 1.4017 must never enter it"
            );
            assert!(
                (clearance.interior_min_eigenvalue - lambda_min).abs()
                    <= 1.0e-12 * lambda_min.abs(),
                "recorded λ_min {} should be the interior sub-block's own {lambda_min}",
                clearance.interior_min_eigenvalue
            );
            assert_eq!(
                Some(clearance.cleared),
                floored,
                "the recorded verdict and the gate must be the same judgment"
            );
        }
    }
}

/// #2388: the tail-probe ladder must never step the probed coordinate
/// outside its own box interval. Past a box bound the ρ-gradient assembly
/// reports the #197 frozen-axis projection — a literal `0.0` — so an
/// out-of-box probe fabricates a hard-zero tail row (`1.531e0 → 0.000e0` in
/// one e-fold in the #2388 evidence). The ladder stops at the midpoint of the
/// coordinate's box, so every probe is strictly inside it and on this rail's
/// half, and the in-domain rows alone confirm the exact tail.
#[test]
fn tail_probe_ladder_never_leaves_the_coordinate_box_2388() {
    let c = 6723.0_f64;
    let box_lower = 12.0_f64;
    let config = test_config();
    let probed = std::sync::Arc::new(std::sync::Mutex::new(Vec::<f64>::new()));
    let probed_in_eval = std::sync::Arc::clone(&probed);
    let problem = OuterProblem::new(1).with_gradient(Derivative::Analytic);
    let mut obj = problem.build_objective(
        (),
        move |_: &mut (), rho: &Array1<f64>| Ok((c * (-rho[0]).exp()).abs()),
        move |_: &mut (), rho: &Array1<f64>| {
            let r = rho[0];
            probed_in_eval.lock().expect("probe log").push(r);
            // Below the box the assembly's frozen-axis convention reports a
            // fabricated zero gradient — exactly the #2388 evidence shape.
            let grad = if r <= box_lower + 1.0e-8 {
                0.0
            } else {
                -c * (-r).exp()
            };
            publish_cancelling_parts(r, grad, 1.0);
            Ok(OuterEval {
                cost: (c * (-r).exp()).abs(),
                gradient: array![grad],
                hessian: HessianValue::Unavailable,
                inner_beta_hint: Some(array![(-r).exp()]),
            })
        },
        None::<fn(&mut ())>,
        None::<fn(&mut (), &Array1<f64>) -> Result<EfsEval, EstimationError>>,
    );
    let rho = array![29.9];
    let rail = build_and_assess_rail_coordinate(
        &mut obj,
        &rho,
        0,
        AsymptoteSide::Upper,
        &config,
        outer_criterion_resolution(&config),
        (box_lower, 30.0),
    )
    .expect("probing must not error")
    .expect("the in-domain rows alone must certify the exact tail");
    assert!(
        (rail.tail_constant - c).abs() / c < 1.0e-6,
        "recovered ĉ={} should equal c={c}",
        rail.tail_constant,
    );
    let midpoint = 0.5 * (box_lower + 30.0);
    let seen = probed.lock().expect("probe log").clone();
    assert!(
        !seen.is_empty() && seen.iter().all(|&r| r > box_lower && r > midpoint),
        "no probe may leave the upper rail's half of the box ({midpoint}, 30): {seen:?}",
    );
}

/// The ladder itself: unit e-folds back from the rail, strictly inside the
/// box, ending at the box midpoint where the other rail becomes the nearer
/// one, and absent when there is no finite box around the coordinate.
#[test]
fn rail_probe_ladder_steps_unit_e_folds_to_the_box_midpoint() {
    let upper = rail_probe_ladder(29.9, AsymptoteSide::Upper, (12.0, 30.0))
        .expect("a finite box containing ρ has a ladder");
    let expected_upper: Vec<f64> = (1_u32..=8).map(|j| 29.9 - f64::from(j)).collect();
    assert_eq!(upper.len(), expected_upper.len(), "ladder {upper:?}");
    for (got, want) in upper.iter().zip(&expected_upper) {
        assert!((got - want).abs() <= 1.0e-12, "ladder {upper:?}");
    }
    assert!(upper.iter().all(|&r| r > 21.0 && r < 30.0));

    let lower = rail_probe_ladder(-9.5, AsymptoteSide::Lower, (-10.0, 4.0))
        .expect("a finite box containing ρ has a ladder");
    let expected_lower: Vec<f64> = (1_u32..=6).map(|j| -9.5 + f64::from(j)).collect();
    assert_eq!(lower.len(), expected_lower.len(), "ladder {lower:?}");
    for (got, want) in lower.iter().zip(&expected_lower) {
        assert!((got - want).abs() <= 1.0e-12, "ladder {lower:?}");
    }
    assert!(lower.iter().all(|&r| r > -10.0 && r < -3.0));

    assert!(rail_probe_ladder(29.9, AsymptoteSide::Upper, (f64::NEG_INFINITY, 30.0)).is_none());
    assert!(rail_probe_ladder(31.0, AsymptoteSide::Upper, (-30.0, 30.0)).is_none());
    assert_eq!(
        rail_probe_ladder(29.9, AsymptoteSide::Upper, (29.0, 30.0)),
        Some(Vec::new()),
        "a box narrower than one e-fold past its midpoint has no probe"
    );
}

/// #3565: the verdict is a property of the tail, not of the criterion's
/// units. The probes' rounding bands are charged on the magnitudes of the
/// parts each gradient was summed from, so multiplying the criterion by `s`
/// multiplies every constant, band and radius by `s` and changes nothing
/// else; an additive constant in `V` changes nothing at all. The retired
/// absolute noise floor on `ĉ` refused the same tail at one scale and
/// certified it at another.
#[test]
fn rail_certificate_is_invariant_to_criterion_units_3565() {
    let config = test_config();
    let certify = |scale: f64, offset: f64| {
        let (count, in_eval) = eval_counter();
        let mut obj = upper_tail_objective(6723.0, 1.0, 0.0, scale, offset, in_eval);
        let rail = build_and_assess_rail_coordinate(
            &mut obj,
            &array![29.9],
            0,
            AsymptoteSide::Upper,
            &config,
            outer_criterion_resolution(&config),
            (-30.0, 30.0),
        )
        .expect("probing must not error")
        .unwrap_or_else(|reason| panic!("scale {scale}: the exact tail must certify: {reason}"));
        let RailTailEvidence::ProbedTail {
            extrapolation_radius,
            ..
        } = rail.evidence
        else {
            panic!("a measured tail must carry probed evidence: {:?}", rail.evidence);
        };
        (
            rail.tail_constant / scale,
            extrapolation_radius / scale,
            count.load(std::sync::atomic::Ordering::Relaxed),
        )
    };
    let (reference_constant, reference_radius, reference_evals) = certify(1.0, 0.0);
    for (scale, offset) in [(1.0e-9, 0.0), (1.0e3, 0.0), (1.0, 1.0e6)] {
        let (constant, radius, evals) = certify(scale, offset);
        assert!(
            (constant - reference_constant).abs() <= 1.0e-12 * reference_constant,
            "scale {scale}, offset {offset}: ĉ/s = {constant} vs {reference_constant}"
        );
        assert!(
            (radius - reference_radius).abs() <= 1.0e-8 * reference_radius,
            "scale {scale}, offset {offset}: R/s = {radius} vs {reference_radius}"
        );
        assert_eq!(evals, reference_evals, "scale {scale}, offset {offset}");
    }
}

/// Build a two-coordinate objective: coordinate 0 follows the upper-rail tail
/// law; coordinate 1 (interior) is gradient-flat.
fn upper_tail_with_interior(c: f64, a: f64) -> impl OuterObjective {
    let problem = OuterProblem::new(2).with_gradient(Derivative::Analytic);
    problem.build_objective(
        (),
        move |_: &mut (), rho: &Array1<f64>| Ok((c * (-rho[0]).exp()).abs()),
        move |_: &mut (), rho: &Array1<f64>| {
            let r = rho[0];
            publish_cancelling_parts(r, -c * (-r).exp(), 1.0);
            Ok(OuterEval {
                cost: (c * (-r).exp()).abs(),
                gradient: array![-c * (-r).exp(), 0.0],
                hessian: HessianValue::Unavailable,
                inner_beta_hint: Some(array![a * (-r).exp(), 0.0]),
            })
        },
        None::<fn(&mut ())>,
        None::<fn(&mut (), &Array1<f64>) -> Result<EfsEval, EstimationError>>,
    )
}

/// The interior-PSD gate is load-bearing: with a positive-definite interior
/// sub-block the confirmed tail mints, but a genuinely indefinite interior
/// curvature refuses the rail certificate even though the tail is clean.
#[test]
fn asymptote_rail_requires_psd_interior_sub_block() {
    let rho = array![29.9, 0.0];
    let projected = array![0.0, 0.0];
    let bounds = (array![-30.0, -30.0], array![30.0, 30.0]);
    let railed = [0usize];
    let config = test_config();

    let mut obj = upper_tail_with_interior(6723.0, 1.0);
    let hessian_psd = array![[1.0, 0.0], [0.0, 2.0]];
    let inputs_psd = AsymptoteRailInputs {
        rho: &rho,
        projected_gradient: &projected,
        railed: &railed,
        layout: OuterThetaLayout::new(2, 0),
        hessian: &hessian_psd,
        bounds: &bounds,
        stationarity_bound: StationarityBound::from_ladder(1.0e-6, StationarityBoundSource::SolverBand),
        objective_tol: 1.0e-5,
        context: "asymptote-rail psd test",
        config: &config,
    };
    let minted = try_certify_asymptote_rail(&mut obj, &inputs_psd)
        .expect("certification must not error");
    let (interior_norm, effective_bound, rails) =
        minted.expect("PSD interior + confirmed tail must mint");
    assert!(interior_norm <= 1.0e-6);
    assert!(
        effective_bound.value() >= interior_norm,
        "the admitting bound must cover the interior norm"
    );
    assert_eq!(rails.len(), 1);
    assert_eq!(rails[0].index, 0);

    let hessian_indefinite = array![[1.0, 0.0], [0.0, -2.0]];
    let inputs_indefinite = AsymptoteRailInputs {
        hessian: &hessian_indefinite,
        ..inputs_psd
    };
    let refused = try_certify_asymptote_rail(&mut obj, &inputs_indefinite)
        .expect("certification must not error");
    assert!(
        refused.is_err(),
        "indefinite interior curvature must refuse the rail certificate, got {refused:?}",
    );
    let reason = refused.unwrap_err();
    assert!(
        reason.contains("not PSD") || reason.contains("interior"),
        "the decline must name the refusing gate, got: {reason}"
    );
}

/// #2453: what authorizes the asymptote certificate is the coordinate
/// being `log λ`, not the numbers looking exponential.
///
/// Holding the fixture bit-identical — same objective, same ρ, same
/// gradient, same PSD Hessian, same box, same railed set, the same
/// textbook-clean `ĉ·e^{−ρ}` tail that the test above mints on — and
/// flipping ONE declared fact, that the coordinate carries a sectional
/// curvature rather than a log-smoothing parameter, must refuse. The
/// alternative is a certificate that reports `value_gap = ĉ·e^{−κ}` as
/// "the exact remaining criterion value-gap" for a quantity that is not
/// in an exponent of anything.
#[test]
fn asymptote_rail_refuses_a_psi_coordinate_with_a_perfect_tail() {
    let rho = array![29.9, 0.0];
    let projected = array![0.0, 0.0];
    let bounds = (array![-30.0, -30.0], array![30.0, 30.0]);
    let railed = [0usize];
    let hessian = array![[1.0, 0.0], [0.0, 2.0]];
    let config = test_config();

    let mut obj = upper_tail_with_interior(6723.0, 1.0);
    let as_psi = AsymptoteRailInputs {
        rho: &rho,
        projected_gradient: &projected,
        railed: &railed,
        // Both slots declared ψ: rho_dim = 0, so coordinate 0 is a
        // design-moving quantity whose box endpoint is attainable.
        layout: OuterThetaLayout::new(2, 2),
        hessian: &hessian,
        bounds: &bounds,
        stationarity_bound: StationarityBound::from_ladder(
            1.0e-6,
            StationarityBoundSource::SolverBand,
        ),
        objective_tol: 1.0e-5,
        context: "asymptote-rail psi-identity test",
        config: &config,
    };
    let refused =
        try_certify_asymptote_rail(&mut obj, &as_psi).expect("certification must not error");
    let reason = refused.expect_err(
        "a psi coordinate must not be certified at an asymptote it has no law for",
    );
    assert!(
        reason.contains("parameterizes log λ"),
        "the decline must name the coordinate's identity as the refusing gate, got: {reason}"
    );

    // The control: the identical numbers under a log-λ declaration still
    // mint, so the refusal above is the identity and nothing else.
    let as_rho = AsymptoteRailInputs {
        layout: OuterThetaLayout::new(2, 0),
        ..as_psi
    };
    let minted = try_certify_asymptote_rail(&mut obj, &as_rho)
        .expect("certification must not error")
        .expect("the same tail under a log-λ declaration must mint");
    assert_eq!(minted.2.len(), 1);
    assert_eq!(minted.2[0].index, 0);
}
