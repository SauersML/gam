//! #2954: where curvature is in hand and the route declares its size, the
//! certificate's stationarity standard is the Newton decrement against rounding
//! bands, and certification does not get easier as the number of rows grows.

use super::*;
use crate::rho_optimizer::decrement_bands::DecrementVerdictNotTaken;
use ndarray::array;

/// gnomon's `--reml-convergence-tolerance` default.
const OUTER_TOL_2954: f64 = 1.0e-3;
const COEFFICIENTS_2954: usize = 10;
const ROWS_2954: [usize; 3] = [2_000, 20_000, 200_000];
/// A scripted criterion has no inner solve, so its mode is exact: it charges a
/// zero inner-residual energy wherever a REML evaluator publishes its own.
const EXACT_INNER_MODE_2954: crate::estimate::outer_eval_capture::InnerResidualCharge =
    crate::estimate::outer_eval_capture::InnerResidualCharge {
        energy: 0.0,
        source: crate::estimate::outer_eval_capture::InnerResidualSource::InnerGradient,
    };

/// `V(ρ) = n·(0.6 + ½ρ²)`, a criterion summed over `n` rows with curvature `n`,
/// certified at `ρ = theta` with the declared scale `n` every REML route sets.
///
/// When `publishes_parts`, each evaluation publishes the gradient parts a REML
/// evaluator would: a same-sign criterion whose whole entry is the penalty
/// channel, so the summed channel magnitude is `|n·ρ|`.
fn certify_row_summed_quadratic_2954(
    n_obs: usize,
    theta: f64,
    publishes_parts: bool,
    criterion_falls: bool,
) -> (
    Result<OuterCriterionCertificate, EstimationError>,
    Array1<f64>,
) {
    let n = n_obs as f64;
    let config = OuterConfig {
        tolerance: OUTER_TOL_2954,
        objective_scale: Some(n),
        rho_uncertainty_problem_size: crate::rho_uncertainty::RhoUncertaintyProblemSize {
            n_obs: Some(n_obs),
            p_coefficients: Some(COEFFICIENTS_2954),
        },
        ..OuterConfig::default()
    };
    // Without `criterion_falls` the value is flat while the gradient and Hessian
    // still describe the quadratic, so no Newton step lowers the criterion.
    let cost = move |rho: &Array1<f64>| {
        if criterion_falls {
            n * (0.6 + 0.5 * rho[0] * rho[0])
        } else {
            n * 0.6
        }
    };
    let mut obj = OuterProblem::new(1)
        .with_gradient(Derivative::Analytic)
        .with_hessian(DeclaredHessianForm::Dense)
        .build_objective(
            (),
            move |_: &mut (), rho: &Array1<f64>| Ok(cost(rho)),
            move |_: &mut (), rho: &Array1<f64>| {
                if publishes_parts {
                    crate::estimate::outer_eval_capture::record_certificate_parts(&[
                        crate::estimate::outer_eval_capture::RhoGradientParts {
                            index: 0,
                            lambda: rho[0].exp(),
                            block_quadratic: 0.0,
                            rank: 1,
                            dim: 1,
                            fixed_beta: n * rho[0],
                            logdet_h: 0.0,
                            frozen_logdet_h: 0.0,
                            mode_response_logdet_h: 0.0,
                            logdet_s: 0.0,
                            total: n * rho[0],
                        },
                    ]);
                    crate::estimate::outer_eval_capture::record_certificate_inner_residual(
                        EXACT_INNER_MODE_2954,
                    );
                }
                Ok(OuterEval {
                    cost: cost(rho),
                    gradient: array![n * rho[0]],
                    hessian: HessianValue::Dense(array![[n]]),
                    inner_beta_hint: None,
                })
            },
            None::<fn(&mut ())>,
            None::<fn(&mut (), &Array1<f64>) -> Result<EfsEval, EstimationError>>,
        );
    let mut result = OuterResult::new(
        array![theta],
        cost(&array![theta]),
        1,
        true,
        OuterPlan {
            solver: Solver::Bfgs,
            hessian_source: HessianSource::BfgsApprox,
        },
    );
    let outcome = certify_outer_optimality(&mut obj, &config, "newton-decrement-2954", &mut result);
    (outcome, result.rho)
}

/// `ρ = 5e-4` sits inside the n-anchored band at every size, `|Pg| = 5e-4·n ≤
/// 1e-3·(1 + |V|)`. The decrease a Newton step still buys there is `½·n·(5e-4)²
/// = 1.25e-7·n`, from 2.5e-4 to 2.5e-2, far above the objective's rounding band,
/// so the point is not stationary at 2,000, 20,000 or 200,000 rows and is never
/// published as it stands. At every size `λ = √n·5e-4` (0.022, 0.071, 0.224) is
/// inside the quadratic region `λ ≤ 1/4`: the mint takes one Newton step, which
/// on this exact quadratic reaches `ρ = 0`, and certifies that point on the
/// decrement rung with the step recorded.
#[test]
fn a_resolvable_decrement_is_not_published_at_any_size_the_scaled_band_admitted_2954() {
    let theta = 5.0e-4;
    for n_obs in ROWS_2954 {
        let n = n_obs as f64;
        let scaled = OuterConfig {
            tolerance: OUTER_TOL_2954,
            objective_scale: Some(n),
            ..OuterConfig::default()
        };
        let band =
            outer_stationarity_band_and_rung_at(&scaled, n * (0.6 + 0.5 * theta * theta)).bound;
        assert!(
            n * theta <= band,
            "control: the n-anchored band {band:.3e} must admit |Pg|={:.3e} at n={n_obs}",
            n * theta,
        );
        let (outcome, published) = certify_row_summed_quadratic_2954(n_obs, theta, true, true);
        let certificate =
            outcome.expect("the Newton polish reaches the optimum, and that point certifies");
        assert_eq!(
            certificate.stationarity.rung().label,
            "newton-decrement",
            "n={n_obs}"
        );
        assert_eq!(
            published[0].to_bits(),
            0.0_f64.to_bits(),
            "n={n_obs}: the published point must be the Newton step's, not ρ = {theta:e}",
        );
        let polish = certificate
            .newton_polish
            .expect("the certificate records the Newton step it took");
        assert_eq!(polish.decreases.len(), 1, "n={n_obs}");
        assert!(polish.lambda_sq_before > 0.0, "n={n_obs}");
        assert_eq!(
            polish.lambda_sq_after.to_bits(),
            0.0_f64.to_bits(),
            "n={n_obs}"
        );
    }
}

/// At 2,000,000 rows the same `ρ = 5e-4` is still admitted by the n-anchored band
/// (`|Pg| = 1000 ≤ 1e-3·(1 + |V|) ≈ 1200`), but `λ = √n·5e-4 ≈ 0.71` is outside
/// the quadratic region `λ ≤ 1/4`. No step budget follows from quadratic
/// convergence, and the mint is refused by name instead of certified: the
/// premature search stop gam#2980 resumes from.
#[test]
fn a_mint_outside_the_quadratic_region_is_refused_by_name_2954() {
    let n_obs = 2_000_000;
    let theta = 5.0e-4;
    let n = n_obs as f64;
    let scaled = OuterConfig {
        tolerance: OUTER_TOL_2954,
        objective_scale: Some(n),
        ..OuterConfig::default()
    };
    let band = outer_stationarity_band_and_rung_at(&scaled, n * (0.6 + 0.5 * theta * theta)).bound;
    assert!(
        n * theta <= band,
        "control: the n-anchored band {band:.3e} must admit |Pg|={:.3e}",
        n * theta,
    );
    let (outcome, published) = certify_row_summed_quadratic_2954(n_obs, theta, true, true);
    let message = outcome
        .expect_err("outside the quadratic region no step budget follows")
        .to_string();
    assert!(
        message.contains("Newton-decrement above tolerance after polish")
            && message.contains("after 0 of 0"),
        "{message}",
    );
    assert_eq!(published[0].to_bits(), theta.to_bits());
}

/// A criterion that does not fall along the Newton step its own gradient and
/// Hessian describe: the step is taken, lowers nothing, and the mint is refused
/// by name rather than certified or handed to the first-order ladder.
#[test]
fn a_polish_that_lowers_nothing_is_refused_by_name_2954() {
    let (outcome, published) = certify_row_summed_quadratic_2954(2_000, 5.0e-4, true, false);
    let message = outcome
        .expect_err("a resolvable decrement the criterion never realises must refuse")
        .to_string();
    assert!(
        message.contains("Newton-decrement above tolerance after polish")
            && message.contains("after 0 of"),
        "{message}",
    );
    assert_eq!(
        published[0].to_bits(),
        5.0e-4_f64.to_bits(),
        "the refused checkpoint is the judged point, not the declined step",
    );
}

/// Near the optimum the decrement `½·n·(1e-9)²` is below the objective's
/// rounding band at every size, and the certificate names the rung that
/// admitted it.
#[test]
fn a_decrement_inside_the_rounding_band_certifies_on_its_own_rung_2954() {
    for n_obs in ROWS_2954 {
        let certificate = certify_row_summed_quadratic_2954(n_obs, 1.0e-9, true, true)
            .0
            .expect("a decrement below the rounding band must certify");
        assert!(certificate.certifies(), "n={n_obs}");
        let rung = certificate.stationarity.rung();
        assert_eq!(rung.label, "newton-decrement", "n={n_obs}");
        assert!(rung.derived_standard, "n={n_obs}");
    }
}

/// An evaluation with curvature in hand but no published gradient parts has no
/// term magnitudes to charge, so no decrement verdict is taken: the first-order
/// ladder decides, and the certificate's rung says which standard that was.
#[test]
fn an_evaluation_without_parts_takes_no_decrement_verdict_2954() {
    for n_obs in ROWS_2954 {
        let certificate = certify_row_summed_quadratic_2954(n_obs, 1.0e-9, false, true)
            .0
            .expect("the first-order ladder certifies a near-optimum");
        let label = &certificate.stationarity.rung().label;
        assert!(
            !label.starts_with("newton-decrement"),
            "n={n_obs}: a verdict was taken without parts: {label}",
        );
    }
}

/// A gradient component summed from two channels that cancel, `+1e6` and
/// `−1e6 + 1e-9`. Charged on the assembled `|g| = 1e-9` its rounding band is
/// about `1e-22`, and the decrement `5e-19` certifies against `band_f = u·|V| ≈
/// 1.1e-16`. Charged on the channels it was summed from, its band is about
/// `2.4e-7`, so `band_λ² ≈ 4.9e-16` reaches the objective band and the verdict
/// cannot certify.
#[test]
fn a_cancelling_gradient_is_charged_on_its_channels_not_its_sum_2954() {
    let n_obs = 1_000;
    let config = OuterConfig {
        rho_uncertainty_problem_size: crate::rho_uncertainty::RhoUncertaintyProblemSize {
            n_obs: Some(n_obs),
            p_coefficients: Some(COEFFICIENTS_2954),
        },
        ..OuterConfig::default()
    };
    let hessian = array![[1.0]];
    let gradient = array![1.0e-9];
    let cost = 1.0;
    let fixed_beta = 1.0e6;
    let logdet_h = -1.0e6 + 1.0e-9;
    let parts = [crate::estimate::outer_eval_capture::RhoGradientParts {
        index: 0,
        lambda: 1.0,
        block_quadratic: 0.0,
        rank: 1,
        dim: 1,
        fixed_beta,
        logdet_h,
        frozen_logdet_h: logdet_h,
        mode_response_logdet_h: 0.0,
        logdet_s: 0.0,
        total: fixed_beta + logdet_h,
    }];
    let growth =
        gam_linalg::roundoff::accumulation_growth(n_obs + COEFFICIENTS_2954 * COEFFICIENTS_2954);
    let assembled = opt::DecrementBands {
        objective: gam_linalg::roundoff::accumulation_growth(1) * cost,
        gradient: gradient.mapv(|component: f64| growth * component.abs()),
        hessian: growth,
    };
    assert!(
        opt::newton_decrement_verdict(&hessian, &gradient, None, &assembled).is_certified(),
        "control: charged on the assembled |g| the decrement certifies",
    );
    let verdict = crate::rho_optimizer::decrement_bands::outer_decrement_verdict(
        &config,
        &hessian,
        &gradient,
        &[],
        cost,
        &crate::estimate::outer_eval_capture::CertificateEvidence {
            parts: parts.to_vec(),
            criterion: None,
            inner_factor: None,
            inner_residual: Some(EXACT_INNER_MODE_2954),
        },
    )
    .expect("the parts cover the coordinate")
    .verdict;
    assert!(
        !verdict.is_certified(),
        "charged on its channels the cancelling component must not certify: {verdict:?}",
    );
}

/// Certify a one-coordinate scripted criterion `V(ρ) = n·f(ρ)` at `ρ = theta`
/// inside the box `[lower, upper]`, with the declared size every REML route
/// sets and, when `limit_faces` is given, which of the two faces the route
/// derived from the term's limit model. Each evaluation publishes the gradient parts a REML evaluator would
/// (the whole entry in the penalty channel) and, when `inner_residual_energy`
/// is given, the criterion channels with that inner-mode energy.
fn certify_scripted_2954(
    n_obs: usize,
    theta: f64,
    (lower, upper): (f64, f64),
    limit_faces: Option<(bool, bool)>,
    f: fn(f64) -> [f64; 3],
    inner_residual_energy: Option<f64>,
) -> (
    Result<OuterCriterionCertificate, EstimationError>,
    Array1<f64>,
) {
    let n = n_obs as f64;
    let config = OuterConfig {
        tolerance: OUTER_TOL_2954,
        objective_scale: Some(n),
        rho_uncertainty_problem_size: crate::rho_uncertainty::RhoUncertaintyProblemSize {
            n_obs: Some(n_obs),
            p_coefficients: Some(COEFFICIENTS_2954),
        },
        model_domain_bounds: Some((array![lower], array![upper])),
        model_domain_limit_faces: limit_faces.map(|(lower, upper)| (vec![lower], vec![upper])),
        ..OuterConfig::default()
    };
    let publish = move |rho: f64| {
        let [value, gradient, _] = f(rho);
        crate::estimate::outer_eval_capture::record_certificate_parts(&[
            crate::estimate::outer_eval_capture::RhoGradientParts {
                index: 0,
                lambda: rho.exp(),
                block_quadratic: 0.0,
                rank: 1,
                dim: 1,
                fixed_beta: n * gradient,
                logdet_h: 0.0,
                frozen_logdet_h: 0.0,
                mode_response_logdet_h: 0.0,
                logdet_s: 0.0,
                total: n * gradient,
            },
        ]);
        crate::estimate::outer_eval_capture::record_certificate_criterion(
            crate::estimate::outer_eval_capture::CertificateCriterion {
                cost: n * value,
                fixed_beta: n * value,
                logdet_h: 0.0,
                logdet_s: 0.0,
                kkt: 0.0,
                inner_residual_energy,
            },
        );
        crate::estimate::outer_eval_capture::record_certificate_inner_residual(
            crate::estimate::outer_eval_capture::InnerResidualCharge {
                energy: inner_residual_energy.unwrap_or(0.0),
                ..EXACT_INNER_MODE_2954
            },
        );
    };
    let mut obj = OuterProblem::new(1)
        .with_gradient(Derivative::Analytic)
        .with_hessian(DeclaredHessianForm::Dense)
        .build_objective(
            (),
            move |_: &mut (), rho: &Array1<f64>| Ok(n * f(rho[0])[0]),
            move |_: &mut (), rho: &Array1<f64>| {
                publish(rho[0]);
                let [value, gradient, curvature] = f(rho[0]);
                Ok(OuterEval {
                    cost: n * value,
                    gradient: array![n * gradient],
                    hessian: HessianValue::Dense(array![[n * curvature]]),
                    inner_beta_hint: None,
                })
            },
            None::<fn(&mut ())>,
            None::<fn(&mut (), &Array1<f64>) -> Result<EfsEval, EstimationError>>,
        );
    let mut result = OuterResult::new(
        array![theta],
        n * f(theta)[0],
        1,
        true,
        OuterPlan {
            solver: Solver::Bfgs,
            hessian_source: HessianSource::BfgsApprox,
        },
    );
    let outcome = certify_outer_optimality(&mut obj, &config, "newton-decrement-2954", &mut result);
    (outcome, result.rho)
}

/// The #2954 census's exponential tail, `V = n·(0.6 + a·e^(−ρ))` with `n·a·e^(−5) = 1e-4`.
const TAIL_2954: fn(f64) -> [f64; 3] = |rho| {
    let a = 1.0e-4 * 5.0_f64.exp() / 2_000.0;
    let tail = a * (-rho).exp();
    [0.6 + tail, -tail, tail]
};

/// [`TAIL_2954`]'s infimum is at `ρ → ∞`, and the route declares both bounds the
/// term's limit model. From `ρ = 5`, where `λ̂² = n·a·e^(−5) = 1e-4`, each Newton step
/// moves ρ by exactly one and `λ̂²` contracts by `e^(−1)`, so the two steps quadratic
/// convergence allows against the channel band `γ_(n+p²)·|V| ≈ 2.8e-10` leave
/// `½λ̂² ≈ 6.8e-6`, four orders above it. The coordinate carrying that step heads to
/// its bound `ρ = 20`, and railing it there lowers the criterion by about `1.35e-5`, so the mint rails it and
/// certifies the railed point on the decrement rung rather than refusing it.
#[test]
fn an_exponential_tail_is_railed_at_its_bound_and_certified_there_2954() {
    let (outcome, published) = certify_scripted_2954(
        2_000,
        5.0,
        (-20.0, 20.0),
        Some((true, true)),
        TAIL_2954,
        None,
    );
    let certificate = outcome.expect("the railed point certifies");
    assert_eq!(certificate.stationarity.rung().label, "newton-decrement");
    assert_eq!(
        published[0].to_bits(),
        20.0_f64.to_bits(),
        "published at the bound"
    );
    let polish = certificate
        .newton_polish
        .expect("the certificate records the polish and its rail");
    assert_eq!(polish.decreases.len(), 2, "{polish:?}");
    // Each decrease is a difference of two values rounded to `u·|V|`, so the ratio
    // carries at most `ε·|V|·(1/ΔV₁ + 1/ΔV₂)` of relative error, `|V| = 0.6·n`.
    let (first, second) = (polish.decreases[0], polish.decreases[1]);
    let ratio = second / first;
    let rounding = f64::EPSILON * 0.6 * 2_000.0 * (1.0 / first + 1.0 / second);
    assert!(
        (ratio / (-1.0_f64).exp() - 1.0).abs() <= rounding,
        "one e-fold per Newton step: ΔV ratio {ratio:.9e}, rounding {rounding:.3e}",
    );
    assert_eq!(polish.rails.len(), 1, "{polish:?}");
    let rail = &polish.rails[0];
    assert_eq!((rail.index, rail.to.to_bits()), (0, 20.0_f64.to_bits()));
    assert_eq!(rail.steps_before, 2);
    assert!(rail.decrease > 1.0e-7, "{rail:?}");
    assert_eq!(rail.face, crate::model_types::RailFaceKind::LimitModel);
    assert!(
        certificate.railed_facts.iter().any(|fact| {
            fact.index == 0 && fact.face == crate::model_types::RailFaceKind::LimitModel
        }),
        "the certificate records the face kind: {:?}",
        certificate.railed_facts,
    );
}

/// A coordinate railed at its upper bound whose gradient points back into the
/// box, `V = n·(0.6 + ½(ρ − ρ*)²)` handed over at `ρ = 20` with `ρ* = 20 − 1e-3`,
/// fails its bound's KKT condition: its inward descent `n·1e-3` is far above its
/// band. It is released from the face, polished to `ρ*` in one Newton step, and
/// certified there, not certified where it was handed over.
#[test]
fn a_railed_coordinate_with_resolvable_inward_descent_is_released_2954() {
    const INWARD_2954: fn(f64) -> [f64; 3] = |rho| {
        let offset = rho - (20.0 - 1.0e-3);
        [0.6 + 0.5 * offset * offset, offset, 1.0]
    };
    let (outcome, published) = certify_scripted_2954(
        2_000,
        20.0,
        (-20.0, 20.0),
        Some((true, true)),
        INWARD_2954,
        None,
    );
    let certificate = outcome.expect("the released coordinate is polished to its optimum");
    assert_eq!(certificate.stationarity.rung().label, "newton-decrement");
    assert!(
        (published[0] - (20.0 - 1.0e-3)).abs() <= 1e-12,
        "published at ρ* = {:.12e}, not at the bound",
        published[0],
    );
    let polish = certificate
        .newton_polish
        .expect("the released coordinate takes a Newton step");
    assert_eq!(polish.decreases.len(), 1, "{polish:?}");
    assert!(polish.rails.is_empty(), "{polish:?}");
}

/// `V = n·(0.6 + ½ρ²)` at `ρ = 3.16e-6`, `n = 2000`: `½λ̂² ≈ 1e-8` is thirty-five times
/// the channel band `γ_(n+p²)·|V| ≈ 2.8e-10`, so a criterion evaluated to rounding
/// accuracy polishes it. When the evaluation reports an inner KKT residual energy
/// `½rᵀH_β⁻¹r = 1e-7`, its value is uncertain by that much, the decrement is
/// inside the band, and the point certifies at that floor where it stands.
#[test]
fn the_inner_residual_energy_is_charged_to_the_objective_band_2954() {
    const QUADRATIC_2954: fn(f64) -> [f64; 3] = |rho| [0.6 + 0.5 * rho * rho, rho, 1.0];
    let theta = 3.16e-6;
    let (rounding_only, polished) =
        certify_scripted_2954(2_000, theta, (-20.0, 20.0), None, QUADRATIC_2954, None);
    let rounding_only = rounding_only.expect("the Newton polish reaches the optimum");
    assert_eq!(
        rounding_only
            .newton_polish
            .map(|polish| polish.decreases.len()),
        Some(1),
        "control: against rounding alone the decrement is resolvable",
    );
    assert!(
        polished[0].abs() <= 1e-6 * theta,
        "control: the step reaches the optimum, ρ = {:.3e}",
        polished[0],
    );
    let (at_floor, published) = certify_scripted_2954(
        2_000,
        theta,
        (-20.0, 20.0),
        None,
        QUADRATIC_2954,
        Some(1.0e-7),
    );
    let at_floor = at_floor.expect("inside the inner-residual band the point certifies");
    assert_eq!(at_floor.stationarity.rung().label, "newton-decrement");
    assert!(
        at_floor.newton_polish.is_none(),
        "{:?}",
        at_floor.newton_polish
    );
    assert_eq!(published[0].to_bits(), theta.to_bits());
}

/// The same exponential tail when the route declares its upper bound `ρ = 20` a
/// representability literal instead of the term's derived limit model (#2627).
/// The two Newton steps leave the same resolvable decrement and the step heads
/// to that bound, but box-KKT certifies nothing about the data at a literal
/// face: the mint does not rail the coordinate there, and refuses by the
/// `representability-face` rung with the checkpoint where the polish stopped.
#[test]
fn a_tail_heading_to_a_representability_face_is_refused_by_type_2954() {
    let (outcome, published) = certify_scripted_2954(
        2_000,
        5.0,
        (-20.0, 20.0),
        Some((true, false)),
        TAIL_2954,
        None,
    );
    let error = outcome.expect_err("a literal face must not certify a rail");
    let message = error.to_string();
    assert!(
        message.contains("Newton-decrement above tolerance after polish")
            && message.contains("representability face"),
        "{message}",
    );
    match &error {
        EstimationError::RemlDidNotConverge {
            stationarity_standard,
            ..
        } => assert_eq!(
            stationarity_standard.rung().map(|rung| rung.label),
            Some("representability-face"),
            "{message}",
        ),
        other => panic!("expected a typed outer refusal, got {other:?}"),
    }
    // Each Newton step is `−g/H` through a Cholesky factor, one e-fold up to a
    // few roundings.
    assert!(
        (published[0] - 7.0).abs() <= 8.0 * f64::EPSILON * 7.0,
        "the checkpoint is where the two Newton steps stopped, not the literal bound: {}",
        published[0],
    );
}

/// The same tail toward `ρ → −∞`, `V = n·(0.6 + a·e^ρ)` from `ρ = −5`, on a term
/// whose unpenalized fit its data do not identify, so the route declares the
/// lower bound a representability face (`rho_domain::unpenalized_fit_is_identified`).
/// λ → 0 is then no limit model: the mint does not rail there and refuses by the
/// `representability-face` rung.
#[test]
fn a_tail_toward_an_unidentified_unpenalized_fit_is_refused_by_type_2954() {
    const LOWER_TAIL_2954: fn(f64) -> [f64; 3] = |rho| {
        let a = 1.0e-4 * 5.0_f64.exp() / 2_000.0;
        let tail = a * rho.exp();
        [0.6 + tail, tail, tail]
    };
    let (outcome, published) = certify_scripted_2954(
        2_000,
        -5.0,
        (-20.0, 20.0),
        Some((false, true)),
        LOWER_TAIL_2954,
        None,
    );
    let error = outcome.expect_err("an unidentified unpenalized limit must not certify a rail");
    let message = error.to_string();
    match &error {
        EstimationError::RemlDidNotConverge {
            stationarity_standard,
            ..
        } => assert_eq!(
            stationarity_standard.rung().map(|rung| rung.label),
            Some("representability-face"),
            "{message}",
        ),
        other => panic!("expected a typed outer refusal, got {other:?}"),
    }
    assert!(
        (published[0] + 7.0).abs() <= 8.0 * f64::EPSILON * 7.0,
        "the checkpoint is where the two Newton steps stopped: {}",
        published[0],
    );
}

/// The objective band is the error the evaluated `V` carries, by term (#2954). A
/// criterion `V = 1` summed from channels `+1e6` and `−1e6 + 1` rounds like its
/// channels, `γ_(n+p²)·2e6 ≈ 2.4e-7`, not like its sum, so `½λ̂² = 1e-8` is inside
/// the channel band and outside `γ_1·|V|`. A `log|H_β|` read from a factor whose
/// own first-order forward error is `δ_logdet = 1.2e-5` carries `½·δ_logdet = 6e-6`
/// into the `½·log|H_β|` channel, so `½λ̂² = 1e-6` is inside the factor band and
/// outside the channels'. A `log|H_β|` channel from a factor that derives no
/// forward error takes no verdict. Nor does a band past the objective resolution
/// the certificate asserts.
#[test]
fn the_objective_band_charges_the_channels_and_the_inner_factor_2954() {
    let config = OuterConfig {
        tolerance: OUTER_TOL_2954,
        rho_uncertainty_problem_size: crate::rho_uncertainty::RhoUncertaintyProblemSize {
            n_obs: Some(1_000),
            p_coefficients: Some(COEFFICIENTS_2954),
        },
        ..OuterConfig::default()
    };
    type Factor = Option<crate::estimate::outer_eval_capture::InnerFactorCondition>;
    let verdict = |half_lambda_sq: f64, channels: Option<(f64, f64)>, factor: Factor| {
        let gradient = array![(2.0 * half_lambda_sq).sqrt()];
        let evidence = crate::estimate::outer_eval_capture::CertificateEvidence {
            parts: vec![crate::estimate::outer_eval_capture::RhoGradientParts {
                index: 0,
                lambda: 1.0,
                block_quadratic: 0.0,
                rank: 1,
                dim: 1,
                fixed_beta: gradient[0],
                logdet_h: 0.0,
                frozen_logdet_h: 0.0,
                mode_response_logdet_h: 0.0,
                logdet_s: 0.0,
                total: gradient[0],
            }],
            criterion: channels.map(|(fixed_beta, logdet_h)| {
                crate::estimate::outer_eval_capture::CertificateCriterion {
                    cost: fixed_beta + logdet_h,
                    fixed_beta,
                    logdet_h,
                    logdet_s: 0.0,
                    kkt: 0.0,
                    inner_residual_energy: None,
                }
            }),
            inner_factor: factor,
            inner_residual: Some(EXACT_INNER_MODE_2954),
        };
        crate::rho_optimizer::decrement_bands::outer_decrement_verdict(
            &config,
            &array![[1.0]],
            &gradient,
            &[],
            1.0,
            &evidence,
        )
    };
    // A factor whose own forward error is negligible, so only the channels decide.
    let exact = Some(crate::estimate::outer_eval_capture::InnerFactorCondition {
        logdet_forward_error: 0.0,
    });
    let cancelling = Some((1.0e6, -1.0e6 + 1.0));
    let control = verdict(1.0e-8, None, None).expect("the parts cover the coordinate");
    assert!(
        !control.verdict.is_certified(),
        "control: against γ_1·|V| the decrement is resolvable",
    );
    let charged = verdict(1.0e-8, cancelling, exact).expect("the parts cover the coordinate");
    assert!(charged.verdict.is_certified(), "{charged:?}");
    assert!(charged.objective_band.channels > 1.0e-7, "{charged:?}");
    let plain = Some((0.5, 0.5));
    let factor = crate::estimate::outer_eval_capture::InnerFactorCondition {
        logdet_forward_error: 1.2e-5,
    };
    let channels_only = verdict(1.0e-6, plain, exact).expect("the parts cover the coordinate");
    assert!(
        !channels_only.verdict.is_certified(),
        "control: against the channels alone the decrement is resolvable",
    );
    let conditioned = verdict(1.0e-6, plain, Some(factor)).expect("the parts cover the coordinate");
    assert!(conditioned.verdict.is_certified(), "{conditioned:?}");
    assert!(
        conditioned.objective_band.factor > 1.0e-6,
        "{conditioned:?}"
    );
    // A nonzero `log|H_β|` channel whose factor derives no forward error takes no
    // verdict, rather than being charged nothing for it.
    assert_eq!(
        verdict(1.0e-6, plain, None).err(),
        Some(DecrementVerdictNotTaken::NoLogdetForwardError)
    );
    // A band past the objective resolution `τ = rel_cost_floor·(1 + |V|) = 2e-5` the
    // certificate asserts would read any decrease as noise: no verdict is taken, and
    // nothing is certified.
    let vacuous = crate::estimate::outer_eval_capture::InnerFactorCondition {
        logdet_forward_error: 1.0,
    };
    let refused =
        verdict(1.0e-6, plain, Some(vacuous)).expect_err("a vacuous band takes no verdict");
    let DecrementVerdictNotTaken::ObjectiveNotResolvable { band_f, tau } = refused else {
        panic!("a vacuous band is refused as unresolvable: {refused}");
    };
    assert!(band_f > tau, "{refused}");
    assert!(
        refused
            .to_string()
            .starts_with("objective not resolvable: band_f "),
        "{refused}"
    );
}

/// A tail two penalties carry together, as the margins of one tensor term do:
/// `V = n·(0.6 + a·(e^(−ρ₀) + e^(−ρ₁)) + b·(ρ₀ − ρ₁)²)` from `ρ₀ = ρ₁ = 5`.
/// Each Newton step moves both by one e-fold, and railing either alone at
/// `ρ = 20` costs `n·b·15²`, so the leader's face is refused and the pair's is
/// taken: both are railed at their limit-model bounds together, and the railed
/// point certifies.
#[test]
fn a_coupled_tail_is_railed_as_one_face_2954() {
    let n_obs = 2_000;
    let n = n_obs as f64;
    let a = 0.5e-4 * 5.0_f64.exp() / n;
    let b = 1.0e-3;
    let value = move |rho: &Array1<f64>| {
        let gap = rho[0] - rho[1];
        n * (0.6 + a * ((-rho[0]).exp() + (-rho[1]).exp()) + b * gap * gap)
    };
    let config = OuterConfig {
        tolerance: OUTER_TOL_2954,
        objective_scale: Some(n),
        rho_uncertainty_problem_size: crate::rho_uncertainty::RhoUncertaintyProblemSize {
            n_obs: Some(n_obs),
            p_coefficients: Some(COEFFICIENTS_2954),
        },
        model_domain_bounds: Some((array![-20.0, -20.0], array![20.0, 20.0])),
        model_domain_limit_faces: Some((vec![true, true], vec![true, true])),
        ..OuterConfig::default()
    };
    let mut obj = OuterProblem::new(2)
        .with_gradient(Derivative::Analytic)
        .with_hessian(DeclaredHessianForm::Dense)
        .build_objective(
            (),
            move |_: &mut (), rho: &Array1<f64>| Ok(value(rho)),
            move |_: &mut (), rho: &Array1<f64>| {
                let gap = rho[0] - rho[1];
                let tails = [a * (-rho[0]).exp(), a * (-rho[1]).exp()];
                let gradient = array![
                    n * (-tails[0] + 2.0 * b * gap),
                    n * (-tails[1] - 2.0 * b * gap)
                ];
                let hessian = array![
                    [n * (tails[0] + 2.0 * b), -2.0 * n * b],
                    [-2.0 * n * b, n * (tails[1] + 2.0 * b)],
                ];
                let parts: Vec<_> = (0..2)
                    .map(|k| crate::estimate::outer_eval_capture::RhoGradientParts {
                        index: k,
                        lambda: rho[k].exp(),
                        block_quadratic: 0.0,
                        rank: 1,
                        dim: 1,
                        fixed_beta: gradient[k],
                        logdet_h: 0.0,
                        frozen_logdet_h: 0.0,
                        mode_response_logdet_h: 0.0,
                        logdet_s: 0.0,
                        total: gradient[k],
                    })
                    .collect();
                crate::estimate::outer_eval_capture::record_certificate_parts(&parts);
                crate::estimate::outer_eval_capture::record_certificate_inner_residual(
                    EXACT_INNER_MODE_2954,
                );
                Ok(OuterEval {
                    cost: value(rho),
                    gradient,
                    hessian: HessianValue::Dense(hessian),
                    inner_beta_hint: None,
                })
            },
            None::<fn(&mut ())>,
            None::<fn(&mut (), &Array1<f64>) -> Result<EfsEval, EstimationError>>,
        );
    let start = array![5.0, 5.0];
    let mut result = OuterResult::new(
        start.clone(),
        value(&start),
        1,
        true,
        OuterPlan {
            solver: Solver::Bfgs,
            hessian_source: HessianSource::BfgsApprox,
        },
    );
    let certificate =
        certify_outer_optimality(&mut obj, &config, "newton-decrement-2954", &mut result)
            .expect("the pair railed together certifies");
    assert_eq!(certificate.stationarity.rung().label, "newton-decrement");
    assert_eq!(
        result.rho.to_vec(),
        vec![20.0, 20.0],
        "both railed at their bounds"
    );
    let polish = certificate
        .newton_polish
        .expect("the polish records its rails");
    let mut railed: Vec<usize> = polish.rails.iter().map(|rail| rail.index).collect();
    railed.sort_unstable();
    assert_eq!(railed, vec![0, 1], "{polish:?}");
    assert!(
        result.final_value < value(&start),
        "the rail lowers the criterion"
    );
}

/// The tensor_clamped census stop: three penalties of one term ride out together
/// (`ρ₂ = ρ₃ = ρ₄ = 15.7`, coupled so that railing any alone costs `n·b·Δ²`) while
/// two interior coordinates sit at `ρ₀ = ρ₁ = −10.4`, a hair below their optimum,
/// so their steps head outward too with a negligible share. Railing the leader
/// alone is refused, and railing every outgoing coordinate kills the interior
/// terms; projected Newton's path along `p` carries the three tail coordinates to
/// their bounds together while the interior coordinates move by `t·p ≈ 0`. The
/// first to arrive is railed there; the other two arrive within the step's own
/// solve error of the face, where the decrement certifies them free, and 0 and 1
/// stay free near their optimum. "On the face" is the resolution every rail is
/// accepted by: railing the arrivals too would lower `V` by no more than `band_f`,
/// here `γ_1·|V|` because the scripted criterion publishes no channels.
#[test]
fn projected_newton_path_rails_the_tail_and_leaves_the_interior_free_2954() {
    let n_obs = 2_000;
    let n = n_obs as f64;
    let a = 1.0e-4 / 3.0 * 15.7_f64.exp() / n;
    let (b, c, optimum) = (1.0e-3, 1.0e-3, -10.4 + 1.0e-5);
    let value = move |rho: &Array1<f64>| {
        let tail: f64 = (2..5).map(|k| a * (-rho[k]).exp()).sum();
        let coupling = (rho[2] - rho[3]).powi(2) + (rho[3] - rho[4]).powi(2);
        let interior = (rho[0] - optimum).powi(2) + (rho[1] - optimum).powi(2);
        n * (0.6 + tail + b * coupling + c * interior)
    };
    let config = OuterConfig {
        tolerance: OUTER_TOL_2954,
        objective_scale: Some(n),
        rho_uncertainty_problem_size: crate::rho_uncertainty::RhoUncertaintyProblemSize {
            n_obs: Some(n_obs),
            p_coefficients: Some(COEFFICIENTS_2954),
        },
        model_domain_bounds: Some((Array1::from_elem(5, -20.0), Array1::from_elem(5, 20.0))),
        model_domain_limit_faces: Some((vec![true; 5], vec![true; 5])),
        ..OuterConfig::default()
    };
    let mut obj = OuterProblem::new(5)
        .with_gradient(Derivative::Analytic)
        .with_hessian(DeclaredHessianForm::Dense)
        .build_objective(
            (),
            move |_: &mut (), rho: &Array1<f64>| Ok(value(rho)),
            move |_: &mut (), rho: &Array1<f64>| {
                let mut gradient = Array1::<f64>::zeros(5);
                let mut hessian = Array2::<f64>::zeros((5, 5));
                for k in 0..2 {
                    gradient[k] = n * 2.0 * c * (rho[k] - optimum);
                    hessian[[k, k]] = n * 2.0 * c;
                }
                for k in 2..5 {
                    let tail = a * (-rho[k]).exp();
                    gradient[k] = -n * tail;
                    hessian[[k, k]] = n * tail;
                }
                for (i, j) in [(2, 3), (3, 4)] {
                    let pull = n * 2.0 * b * (rho[i] - rho[j]);
                    gradient[i] += pull;
                    gradient[j] -= pull;
                    hessian[[i, i]] += n * 2.0 * b;
                    hessian[[j, j]] += n * 2.0 * b;
                    hessian[[i, j]] -= n * 2.0 * b;
                    hessian[[j, i]] -= n * 2.0 * b;
                }
                let parts: Vec<_> = (0..5)
                    .map(|k| crate::estimate::outer_eval_capture::RhoGradientParts {
                        index: k,
                        lambda: rho[k].exp(),
                        block_quadratic: 0.0,
                        rank: 1,
                        dim: 1,
                        fixed_beta: gradient[k],
                        logdet_h: 0.0,
                        frozen_logdet_h: 0.0,
                        mode_response_logdet_h: 0.0,
                        logdet_s: 0.0,
                        total: gradient[k],
                    })
                    .collect();
                crate::estimate::outer_eval_capture::record_certificate_parts(&parts);
                crate::estimate::outer_eval_capture::record_certificate_inner_residual(
                    EXACT_INNER_MODE_2954,
                );
                Ok(OuterEval {
                    cost: value(rho),
                    gradient,
                    hessian: HessianValue::Dense(hessian),
                    inner_beta_hint: None,
                })
            },
            None::<fn(&mut ())>,
            None::<fn(&mut (), &Array1<f64>) -> Result<EfsEval, EstimationError>>,
        );
    let start = array![-10.4, -10.4, 15.7, 15.7, 15.7];
    let mut result = OuterResult::new(
        start.clone(),
        value(&start),
        1,
        true,
        OuterPlan {
            solver: Solver::Bfgs,
            hessian_source: HessianSource::BfgsApprox,
        },
    );
    let certificate =
        certify_outer_optimality(&mut obj, &config, "newton-decrement-2954", &mut result)
            .expect("the tail railed along the path certifies");
    assert_eq!(certificate.stationarity.rung().label, "newton-decrement");
    let polish = certificate
        .newton_polish
        .expect("the polish records its rails");
    assert!(
        !polish.rails.is_empty()
            && polish.rails.iter().all(|rail| {
                (2..5).contains(&rail.index)
                    && rail.to == 20.0
                    && rail.face == crate::model_types::RailFaceKind::LimitModel
            }),
        "only tail coordinates are railed, at their limit-model bounds: {polish:?}"
    );
    let mut on_face = result.rho.clone();
    for k in 2..5 {
        on_face[k] = 20.0;
    }
    let band_f = gam_linalg::roundoff::accumulation_growth(1) * result.final_value.abs();
    assert!(
        result.final_value - value(&on_face) <= band_f,
        "the tail ends on its face to the criterion's resolution: railing it lowers V by \
         {:.3e} against band_f {band_f:.3e} at {:?}",
        result.final_value - value(&on_face),
        result.rho,
    );
    for k in 0..2 {
        assert!(
            (result.rho[k] - optimum).abs() < 1.0e-3,
            "interior coordinate {k} stays free near its optimum: {:?}",
            result.rho,
        );
    }
    assert!(
        result.final_value < value(&start),
        "the rail lowers the criterion"
    );
}

/// Certify a one-coordinate scripted criterion `V(ρ) = n·f(ρ)` at `ρ = theta`
/// through an evaluator with two routes to the same value, as the Gaussian REML
/// state has (#2954): a gradient-bearing evaluation returns `V`, and so does a
/// value-only probe at a point the gradient-bearing route already evaluated (the
/// outer-evaluation cache, which holds `theta` because the search's last
/// iterate was evaluated there), but a value-only probe anywhere else reads
/// `V + value_only_offset`. Like that cache, a repeated gradient-bearing
/// evaluation at a point is answered without publishing its evidence again.
/// The points a two-route objective has evaluated: by any route (its value
/// cache), and by the gradient-bearing route (its published evidence).
type TwoRouteCache2954 = (
    std::collections::HashSet<u64>,
    std::collections::HashSet<u64>,
);

fn certify_two_route_2954(
    theta: f64,
    f: impl Fn(f64) -> [f64; 3] + Copy + 'static,
    value_only_offset: f64,
) -> (
    Result<OuterCriterionCertificate, EstimationError>,
    OuterResult,
) {
    certify_two_route_walk_2954(theta, f, value_only_offset, false).0
}

/// [`certify_two_route_2954`] for one walk that may start at a point its
/// gradient-bearing route already evaluated unarmed (`cached_at_theta`), so the
/// certificate's first evaluation there publishes nothing, and that reports the
/// points that route evaluated, in order.
fn certify_two_route_walk_2954(
    theta: f64,
    f: impl Fn(f64) -> [f64; 3] + Copy + 'static,
    value_only_offset: f64,
    cached_at_theta: bool,
) -> (
    (
        Result<OuterCriterionCertificate, EstimationError>,
        OuterResult,
    ),
    Vec<f64>,
) {
    let evaluated = std::sync::Arc::new(std::sync::Mutex::new(Vec::<f64>::new()));
    let trace = std::sync::Arc::clone(&evaluated);
    let n_obs = 2_000;
    let n = n_obs as f64;
    let config = OuterConfig {
        tolerance: OUTER_TOL_2954,
        objective_scale: Some(n),
        rho_uncertainty_problem_size: crate::rho_uncertainty::RhoUncertaintyProblemSize {
            n_obs: Some(n_obs),
            p_coefficients: Some(COEFFICIENTS_2954),
        },
        ..OuterConfig::default()
    };
    let mut obj = OuterProblem::new(1)
        .with_gradient(Derivative::Analytic)
        .with_hessian(DeclaredHessianForm::Dense)
        .build_objective(
            (
                std::collections::HashSet::from([theta.to_bits()]),
                if cached_at_theta {
                    std::collections::HashSet::from([theta.to_bits()])
                } else {
                    std::collections::HashSet::new()
                },
            ),
            move |(valued, _): &mut TwoRouteCache2954, rho: &Array1<f64>| {
                let value = n * f(rho[0])[0];
                Ok(if valued.contains(&rho[0].to_bits()) {
                    value
                } else {
                    value + value_only_offset
                })
            },
            move |(valued, published): &mut TwoRouteCache2954, rho: &Array1<f64>| {
                valued.insert(rho[0].to_bits());
                trace
                    .lock()
                    .expect("the trace lock is never poisoned")
                    .push(rho[0]);
                let [value, gradient, curvature] = f(rho[0]);
                let evaluation = OuterEval {
                    cost: n * value,
                    gradient: array![n * gradient],
                    hessian: HessianValue::Dense(array![[n * curvature]]),
                    inner_beta_hint: None,
                };
                if !published.insert(rho[0].to_bits()) {
                    return Ok(evaluation);
                }
                crate::estimate::outer_eval_capture::record_certificate_parts(&[
                    crate::estimate::outer_eval_capture::RhoGradientParts {
                        index: 0,
                        lambda: rho[0].exp(),
                        block_quadratic: 0.0,
                        rank: 1,
                        dim: 1,
                        fixed_beta: n * gradient,
                        logdet_h: 0.0,
                        frozen_logdet_h: 0.0,
                        mode_response_logdet_h: 0.0,
                        logdet_s: 0.0,
                        total: n * gradient,
                    },
                ]);
                crate::estimate::outer_eval_capture::record_certificate_inner_residual(
                    EXACT_INNER_MODE_2954,
                );
                Ok(evaluation)
            },
            None::<fn(&mut TwoRouteCache2954)>,
            None::<fn(&mut TwoRouteCache2954, &Array1<f64>) -> Result<EfsEval, EstimationError>>,
        );
    let mut result = OuterResult::new(
        array![theta],
        n * f(theta)[0],
        1,
        true,
        OuterPlan {
            solver: Solver::Bfgs,
            hessian_source: HessianSource::BfgsApprox,
        },
    );
    let outcome = certify_outer_optimality(&mut obj, &config, "newton-decrement-2954", &mut result);
    let evaluated = evaluated
        .lock()
        .expect("the trace lock is never poisoned")
        .clone();
    ((outcome, result), evaluated)
}

/// A polish step is kept only when the evaluation order the certificate judged
/// the current point at says it lowers `V` by more than `band_f` (#2954).
///
/// `V = n·(0.6 + s·√(1 + ρ²))` from `ρ = 1.5` with `n·s = 0.01`: `λ₀ ≈ 0.20` is
/// inside the quadratic region, but the Newton step `−ρ(1 + ρ²) = −4.875`
/// overshoots to `ρ = −3.375` and raises `V` by `0.0172`. A value-only probe that
/// reads `0.02` low there would call that a decrease. Judged at the certificate's
/// own order it is a rise: the step is not kept, the judged point stays where it
/// is, and the mint refuses by name.
///
/// Conversely, on `V = n·(0.6 + ½ρ²)` from `ρ = 5e-4` a value-only probe that
/// reads `1.0` high would refuse the step that reaches the optimum. Judged at the
/// certificate's order it lowers `V` by the full-order decrease the record
/// carries, and the point it reaches is published with its full-order value. The
/// certificate's own evaluation there is answered from the cache the trial
/// filled, so it decides on the evidence the trial published.
#[test]
fn a_polish_step_is_judged_at_the_certificates_own_evaluation_order_2954() {
    const OVERSHOOT_2954: fn(f64) -> [f64; 3] = |rho| {
        let s = 0.01 / 2_000.0;
        let root = (1.0 + rho * rho).sqrt();
        [0.6 + s * root, s * rho / root, s / (root * root * root)]
    };
    let (overshoot, judged) = certify_two_route_2954(1.5, OVERSHOOT_2954, -0.02);
    let refusal = overshoot.expect_err("a step that raises the judged criterion is never kept");
    assert!(
        refusal
            .to_string()
            .contains("Newton-decrement above tolerance after polish"),
        "{refusal}"
    );
    assert_eq!(judged.rho.to_vec(), vec![1.5], "the judged point stays put");

    const QUADRATIC_2954: fn(f64) -> [f64; 3] = |rho| [0.6 + 0.5 * rho * rho, rho, 1.0];
    let (reaching, reached) = certify_two_route_2954(5.0e-4, QUADRATIC_2954, 1.0);
    let certificate = reaching.expect("the full-order decrease is kept and the optimum certifies");
    assert_eq!(certificate.stationarity.rung().label, "newton-decrement");
    let polish = certificate
        .newton_polish
        .expect("the polish records its step");
    let full_order = |rho: f64| 2_000.0 * QUADRATIC_2954(rho)[0];
    assert_eq!(polish.decreases.len(), 1, "{polish:?}");
    assert_eq!(
        polish.decreases[0],
        full_order(5.0e-4) - full_order(reached.rho[0]),
        "the recorded decrease is the full-order one",
    );
    assert_eq!(
        reached.final_value.to_bits(),
        full_order(reached.rho[0]).to_bits(),
        "the published value is the full-order one",
    );
}

/// An evaluation that forms no residual for its inner mode carries an error the
/// objective band cannot charge, so no verdict is taken there (#2954); one that
/// forms it is charged its `½·rᵀH_β⁻¹r`.
#[test]
fn an_inner_mode_without_a_residual_takes_no_decrement_verdict_2954() {
    let config = OuterConfig {
        tolerance: OUTER_TOL_2954,
        objective_scale: Some(2_000.0),
        rho_uncertainty_problem_size: crate::rho_uncertainty::RhoUncertaintyProblemSize {
            n_obs: Some(2_000),
            p_coefficients: Some(COEFFICIENTS_2954),
        },
        ..OuterConfig::default()
    };
    let evidence = |inner_residual| crate::estimate::outer_eval_capture::CertificateEvidence {
        parts: vec![crate::estimate::outer_eval_capture::RhoGradientParts {
            index: 0,
            lambda: 1.0,
            block_quadratic: 0.0,
            rank: 1,
            dim: 1,
            fixed_beta: 1.0e-6,
            logdet_h: 0.0,
            frozen_logdet_h: 0.0,
            mode_response_logdet_h: 0.0,
            logdet_s: 0.0,
            total: 1.0e-6,
        }],
        criterion: None,
        inner_factor: None,
        inner_residual,
    };
    let verdict = |inner_residual| {
        crate::rho_optimizer::decrement_bands::outer_decrement_verdict(
            &config,
            &array![[1.0]],
            &array![1.0e-6],
            &[],
            1.0,
            &evidence(inner_residual),
        )
    };
    assert_eq!(
        verdict(None).err(),
        Some(DecrementVerdictNotTaken::NoInnerResidual)
    );
    let charged = verdict(Some(
        crate::estimate::outer_eval_capture::InnerResidualCharge {
            energy: 1.0e-9,
            source: crate::estimate::outer_eval_capture::InnerResidualSource::NormalEquations,
        },
    ))
    .expect("a formed residual is charged");
    assert_eq!(charged.objective_band.inner_residual, 1.0e-9);
}

/// Two optimizer walks that visit the same point each certify from their own
/// evidence (#2954). Walk A polishes from `ρ = 5e-4` toward the optimum, where a
/// bump makes its full-order criterion rise, so its trial is evaluated and
/// refused. Walk B, a different objective, starts at exactly that trial point,
/// which its gradient-bearing route already evaluated unarmed, so its
/// certificate's evaluation there publishes nothing and B has no evidence to
/// decide the decrement on. The evidence A's trial published at those bits
/// belongs to A's walk and is dropped with it: B decides the same way whether or
/// not A ran first.
#[test]
fn two_walks_at_one_point_certify_from_their_own_evidence_2954() {
    let bumped = |rho: f64| {
        let bump = if rho.abs() < 1.0e-4 { 1.0 } else { 0.0 };
        [0.6 + 0.5 * rho * rho + bump, rho, 1.0]
    };
    let ((walk_a, _), evaluated) = certify_two_route_walk_2954(5.0e-4, bumped, 0.0, false);
    assert!(
        walk_a.is_err(),
        "walk A refuses its rising trial: {walk_a:?}"
    );
    let trial = *evaluated
        .iter()
        .rev()
        .find(|rho| rho.abs() < 1.0e-4)
        .expect("walk A evaluated its trial at full order");
    let centred = move |rho: f64| [0.6 + 0.5 * (rho - trial) * (rho - trial), rho - trial, 1.0];
    fn walk_b(trial: f64, centred: impl Fn(f64) -> [f64; 3] + Copy + 'static) -> String {
        match certify_two_route_walk_2954(trial, centred, 0.0, true).0.0 {
            Ok(certificate) => format!("certified on {}", certificate.stationarity.rung().label),
            Err(error) => format!("refused: {error}"),
        }
    }
    let after_a = walk_b(trial, centred);
    let alone = std::thread::spawn(move || walk_b(trial, centred))
        .join()
        .expect("walk B runs on its own thread");
    assert_eq!(after_a, alone);
}
