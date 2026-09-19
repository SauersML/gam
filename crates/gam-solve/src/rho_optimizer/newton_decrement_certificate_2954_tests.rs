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
/// The statistical resolution `τ_stat = 1/(2n)` of a criterion summed over the
/// 2,000 rows most scripted criteria below declare: the log-likelihood change
/// one observation's worth of information can resolve.
const TAU_STAT_2954: f64 = 0.5 / 2_000.0;
/// A scripted criterion has no inner solve, so its mode is exact: it charges a
/// zero inner-residual energy wherever a REML evaluator publishes its own.
const EXACT_INNER_MODE_2954: crate::estimate::outer_eval_capture::InnerResidualCharge =
    crate::estimate::outer_eval_capture::InnerResidualCharge {
        energy: 0.0,
        source: crate::estimate::outer_eval_capture::InnerResidualSource::InnerGradient,
    };

/// The certificate band #2954 removed, `τ·(1 + |V|)` at the judged point: a
/// criterion summed over `n` rows made it grow with `n`. Stated here so the
/// controls below can show a point that band admitted.
fn removed_n_anchored_band_2954(cost: f64) -> f64 {
    OUTER_TOL_2954 * (1.0 + cost.abs())
}

/// `V(ρ) = n·(0.6 + ½ρ²)`, a criterion summed over `n` rows with curvature `n`,
/// certified at `ρ = theta` with the problem size every REML route declares.
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
        problem_size: crate::rho_optimizer::OuterProblemSize {
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

/// `ρ = 5e-4` sits inside the removed n-anchored band at every size, `|Pg| = 5e-4·n ≤
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
        let band = removed_n_anchored_band_2954(n * (0.6 + 0.5 * theta * theta));
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

/// At 2,000,000 rows the same `ρ = 5e-4` is still admitted by the removed n-anchored band
/// (`|Pg| = 1000 ≤ 1e-3·(1 + |V|) ≈ 1200`), and `λ = √n·5e-4 ≈ 0.71` is outside
/// the quadratic region `λ ≤ 1/4`. The polish no longer predicts a step budget from
/// that region, which here was zero steps and a refusal by name: it takes the damped
/// Newton step, which on this exact quadratic reaches `ρ = 0`, and certifies there.
#[test]
fn a_mint_outside_the_quadratic_region_takes_its_newton_step_3012() {
    let n_obs = 2_000_000;
    let theta = 5.0e-4;
    let n = n_obs as f64;
    let band = removed_n_anchored_band_2954(n * (0.6 + 0.5 * theta * theta));
    assert!(
        n * theta <= band,
        "control: the n-anchored band {band:.3e} must admit |Pg|={:.3e}",
        n * theta,
    );
    assert!(
        (n.sqrt() * theta) > 0.25,
        "control: λ = √n·ρ must sit outside the quadratic region",
    );
    let (outcome, published) = certify_row_summed_quadratic_2954(n_obs, theta, true, true);
    let certificate =
        outcome.expect("outside the quadratic region the Newton step is still taken");
    assert_eq!(certificate.stationarity.rung().label, "newton-decrement");
    // The step `ρ − (n·ρ)/n` reaches the optimum up to the rounding of `ρ` itself.
    assert!(
        published[0].abs() <= 2.0 * f64::EPSILON * theta,
        "published at ρ = {:.3e}, not at the optimum",
        published[0],
    );
    let polish = certificate
        .newton_polish
        .expect("the certificate records the Newton step it took");
    assert_eq!(polish.decreases.len(), 1, "{polish:?}");
    assert!(!polish.settled, "{polish:?}");
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
            && message.contains("after 0 Newton step(s)"),
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
/// `−1e6 + 1e-9`, along a direction of curvature `1e-13`. Charged on the
/// assembled `|g| = 1e-9` its rounding band is about `1e-22`, and the decrement
/// `λ̂² = 1e-5` certifies against `τ_stat = 1/(2n) = 5e-4`. Charged on the
/// channels it was summed from, its band is about `2.4e-7`, so `band_λ²`, at
/// least `2λ̂·band_g/√H ≈ 4.9e-3`, passes the tolerance and the verdict cannot
/// certify.
#[test]
fn a_cancelling_gradient_is_charged_on_its_channels_not_its_sum_2954() {
    let n_obs = 1_000;
    let config = OuterConfig {
        problem_size: crate::rho_optimizer::OuterProblemSize {
            n_obs: Some(n_obs),
            p_coefficients: Some(COEFFICIENTS_2954),
        },
        ..OuterConfig::default()
    };
    let curvature = 1.0e-13;
    let hessian = array![[curvature]];
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
        objective: crate::rho_optimizer::decrement_bands::DecrementTolerance {
            tau_stat: 0.5 / n_obs as f64,
            band_f: gam_linalg::roundoff::accumulation_growth(1) * cost,
        }
        .value(),
        gradient: gradient.mapv(|component: f64| growth * component.abs()),
        hessian: growth * curvature,
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
        problem_size: crate::rho_optimizer::OuterProblemSize {
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

/// The #2954 census's exponential tail, `V = n·(0.6 + a·e^(−ρ))` with
/// `n·a·e^(−5) = 3e-2`, `120·τ_stat` at `n = 2000`.
const TAIL_2954: fn(f64) -> [f64; 3] = |rho| {
    let a = 3.0e-2 * 5.0_f64.exp() / 2_000.0;
    let tail = a * (-rho).exp();
    [0.6 + tail, -tail, tail]
};

/// [`TAIL_2954`]'s infimum is at `ρ → ∞`, and the route declares both bounds the
/// term's limit model. From `ρ = 5`, where `λ̂² = n·a·e^(−5) = 3e-2`, the Newton
/// step moves ρ by exactly one and `λ̂²` contracts by `e^(−1)`: `λ₊ ≈ 0.105` against
/// Newton's quadratic rate `2λ² = 6e-2`, so the decrement contracts at a linear
/// rate, the tail's signature, with `λ̂² ≈ 1.1e-2` still forty times the
/// statistical resolution `τ_stat = 1/(2n) = 2.5e-4`. The coordinate carrying that step heads to
/// its bound `ρ = 20`, and railing it there lowers the criterion by about `1.1e-2`,
/// the whole decrease `λ̂² = V − V∞` left, so the mint rails it and certifies the
/// railed point on the decrement rung rather than refusing it.
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
    assert_eq!(polish.decreases.len(), 1, "{polish:?}");
    // The Newton step lowers `V` by `λ̂²·(1 − e^(−1))`, a difference of two values
    // rounded to `u·|V|`, `|V| = 0.6·n`.
    let rounding = 2.0 * f64::EPSILON * 0.6 * 2_000.0;
    assert!(
        (polish.decreases[0] - polish.lambda_sq_before * (1.0 - (-1.0_f64).exp())).abs()
            <= rounding,
        "one e-fold per Newton step: {polish:?}",
    );
    assert_eq!(polish.rails.len(), 1, "{polish:?}");
    let rail = &polish.rails[0];
    assert_eq!((rail.index, rail.to.to_bits()), (0, 20.0_f64.to_bits()));
    assert_eq!(rail.steps_before, 1);
    assert!(rail.decrease > TAU_STAT_2954, "{rail:?}");
    assert_eq!(rail.face, crate::model_types::RailFaceKind::LimitModel);
    assert!(
        certificate.railed_facts.iter().any(|fact| {
            fact.index == 0 && fact.face == crate::model_types::RailFaceKind::LimitModel
        }),
        "the certificate records the face kind: {:?}",
        certificate.railed_facts,
    );
}

/// `V = n·(0.6 + s·(e^ρ − 1 − ρ))` with `n·s = 1e-2`: an interior optimum at
/// `ρ = 0` whose third derivative equals its second, so `½V‴/V″^(3/2) =
/// 1/(2√(n·s)) = 5` and Newton converges quadratically at five times the unit
/// self-concordant rate, as a LAML criterion does along a smooth whose penalty
/// barely binds. From `ρ = 1.5` (`λ̂² ≈ 2.7e-2`) the decrement contracts at
/// five times Newton's unit rate, slower than the measured test `λ₊ ≤ 2λ²` admits,
/// and there is no limit-model face to rail. The decrement is still contracting and
/// each step lowers the criterion by more than the tolerance, so the polish keeps
/// taking Newton steps (#3012), and the third is inside `τ_stat = 2.5e-4`: the
/// mint certifies the optimum instead of refusing a walk that is converging.
#[test]
fn a_criterion_at_a_steep_quadratic_rate_is_polished_to_its_optimum_2954() {
    const STEEP_2954: fn(f64) -> [f64; 3] = |rho| {
        let s = 1.0e-2 / 2_000.0;
        [
            0.6 + s * (rho.exp() - 1.0 - rho),
            s * (rho.exp() - 1.0),
            s * rho.exp(),
        ]
    };
    let (outcome, published) =
        certify_scripted_2954(2_000, 1.5, (-20.0, 20.0), None, STEEP_2954, None);
    let certificate = outcome.expect("Newton at the measured rate reaches the optimum");
    assert_eq!(certificate.stationarity.rung().label, "newton-decrement");
    let polish = certificate
        .newton_polish
        .expect("the certificate records the polish");
    assert_eq!(polish.decreases.len(), 3, "{polish:?}");
    assert!(polish.rails.is_empty(), "{polish:?}");
    assert!(polish.lambda_sq_after <= TAU_STAT_2954, "{polish:?}");
    assert!(
        published[0].abs() <= 5.0e-2,
        "published at ρ = {:.3e}",
        published[0]
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

/// `V = n·(0.6 + ½ρ²)` at `ρ = 3.16e-4`, `n = 2000`: `λ̂² ≈ 2e-4` is inside the
/// statistical resolution `τ_stat = 1/(2n) = 2.5e-4` less the channel band
/// `γ_(n+p²)·|V| ≈ 2.8e-10`, so a criterion evaluated to rounding accuracy
/// certifies where it stands. When the evaluation reports an inner KKT residual
/// energy `½rᵀH_β⁻¹r = 1e-4`, its value is uncertain by that much, the tolerance
/// left to the decrement is `τ_stat − 1e-4 ≈ 1.5e-4`, and the point is not
/// certified where it stands: the polish takes the step that reaches the optimum.
#[test]
fn the_inner_residual_energy_is_charged_to_the_objective_band_2954() {
    const QUADRATIC_2954: fn(f64) -> [f64; 3] = |rho| [0.6 + 0.5 * rho * rho, rho, 1.0];
    let theta = 3.16e-4;
    let energy = 1.0e-4;
    let lambda_sq = 2_000.0 * theta * theta;
    assert!(
        lambda_sq < TAU_STAT_2954 && lambda_sq > TAU_STAT_2954 - energy,
        "control: λ̂²={lambda_sq:.3e} sits between τ_stat − E and τ_stat",
    );
    let (rounding_only, standing) =
        certify_scripted_2954(2_000, theta, (-20.0, 20.0), None, QUADRATIC_2954, None);
    let rounding_only = rounding_only.expect("against rounding alone the point certifies");
    assert_eq!(rounding_only.stationarity.rung().label, "newton-decrement");
    assert!(
        rounding_only.newton_polish.is_none(),
        "control: against rounding alone the decrement is inside the tolerance: {:?}",
        rounding_only.newton_polish,
    );
    assert_eq!(standing[0].to_bits(), theta.to_bits());
    let (charged, polished) = certify_scripted_2954(
        2_000,
        theta,
        (-20.0, 20.0),
        None,
        QUADRATIC_2954,
        Some(energy),
    );
    let charged = charged.expect("the polish reaches the optimum");
    assert_eq!(charged.stationarity.rung().label, "newton-decrement");
    assert_eq!(
        charged.newton_polish.map(|polish| polish.decreases.len()),
        Some(1),
        "the charged residual leaves the decrement above its tolerance",
    );
    assert!(
        polished[0].abs() <= 1e-6 * theta,
        "the step reaches the optimum, ρ = {:.3e}",
        polished[0],
    );
}

/// `V = n·(0.6 + ½ρ²)`, `n = 2000`, handed over where `λ̂² = n·ρ² = 1.5·tol`,
/// `tol = τ_stat − band_f` the decrease the verdict may leave to the minimum.
/// The decrement bounds the decrease left, `V − V* = ½λ̂²` here, only up to the
/// factor two a general self-concordant criterion needs, so `λ̂² > tol` leaves a
/// decrease the statistics may resolve and the point is not certified where it
/// stands, although `½λ̂² = 0.75·tol` would have certified it (#3012). The full
/// step's own model decrease is inside the tolerance, so no step could show a
/// resolvable decrease: the polish takes it as a settling step, which may not
/// raise `V` by more than `tol` and whose point must certify, and it reaches
/// the optimum.
#[test]
fn a_decrement_whose_full_step_is_unresolvable_takes_a_settling_step_3012() {
    const QUADRATIC_3012: fn(f64) -> [f64; 3] = |rho| [0.6 + 0.5 * rho * rho, rho, 1.0];
    let n_obs = 2_000;
    let n = n_obs as f64;
    // The criterion channel is the whole value, `n·f(ρ)`, charged at the
    // formation count of `n` rows and `p²` coefficient pairs.
    let band_f = gam_linalg::roundoff::accumulation_growth(n_obs + COEFFICIENTS_2954 * COEFFICIENTS_2954)
        * n
        * 0.6;
    let tol = crate::rho_optimizer::decrement_bands::DecrementTolerance {
        tau_stat: TAU_STAT_2954,
        band_f,
    }
    .value();
    let theta = (1.5 * tol / n).sqrt();
    let lambda_sq = n * theta * theta;
    assert!(
        0.5 * lambda_sq <= tol && lambda_sq > tol,
        "control: ½λ̂²={:.3e} ≤ tol={tol:.3e} < λ̂²={lambda_sq:.3e}",
        0.5 * lambda_sq,
    );
    let (outcome, published) =
        certify_scripted_2954(n_obs, theta, (-20.0, 20.0), None, QUADRATIC_3012, None);
    let certificate = outcome.expect("the settling step's point certifies");
    assert_eq!(certificate.stationarity.rung().label, "newton-decrement");
    let polish = certificate
        .newton_polish
        .expect("a point whose λ̂² exceeds tol is not certified where it stands");
    assert_eq!(polish.decreases.len(), 1, "{polish:?}");
    assert!(polish.settled, "the step is recorded as a settling step: {polish:?}");
    assert!(
        (polish.lambda_sq_before / lambda_sq - 1.0).abs() <= 1.0e-12,
        "{polish:?}"
    );
    assert!(polish.lambda_sq_after <= tol, "{polish:?}");
    assert!(
        published[0].abs() <= 4.0 * f64::EPSILON * theta,
        "published at ρ = {:.3e}, not at the optimum",
        published[0],
    );
}

/// The same exponential tail when the route declares its upper bound `ρ = 20` a
/// representability literal instead of the term's derived limit model (#2627).
/// Box-KKT certifies nothing about the data at a literal face, so the mint never
/// rails the coordinate there. Along the tail, though, `λ̂² = V − V∞` is the whole
/// decrease left, and although it contracts slower than Newton's quadratic rate the
/// polish keeps following it inside the box while `λ̂²` contracts (#3012). It certifies
/// on the decrement rung once that decrease is inside `τ_stat`, short of the face.
#[test]
fn a_tail_heading_to_a_representability_face_is_certified_inside_the_box_3012() {
    let (outcome, published) = certify_scripted_2954(
        2_000,
        5.0,
        (-20.0, 20.0),
        Some((true, false)),
        TAIL_2954,
        None,
    );
    let certificate = outcome.expect("the tail is followed inside the box and certified");
    assert_eq!(certificate.stationarity.rung().label, "newton-decrement");
    let polish = certificate
        .newton_polish
        .expect("the certificate records the polish");
    assert!(
        polish.rails.is_empty(),
        "never railed at a literal face: {polish:?}"
    );
    assert!(
        polish.decreases.len() > 1,
        "the polish must continue past a step that contracts at a linear rate: {polish:?}",
    );
    assert!(
        published[0] > 7.0 && published[0] < 20.0,
        "certified inside the box, past the first linear-rate step: {}",
        published[0],
    );
    // Along the tail the decrement is the decrease left to the infimum,
    // `λ̂² = n·a·e^(−ρ) = V(ρ) − V∞`, at the certified point too.
    let left = 2_000.0 * (TAIL_2954(published[0])[0] - 0.6);
    assert!(
        (polish.lambda_sq_after - left).abs() <= 64.0 * f64::EPSILON * 2_000.0 * 0.6,
        "λ̂² {:.9e} against the decrease left {left:.9e}",
        polish.lambda_sq_after,
    );
}

/// When the literal face is closer than the band, the tail cannot be followed to
/// a certifiable point inside the box: every Newton step that would move the
/// coordinate onto the face, into the rail margin within which the certificate
/// reads it railed there, is halved (#3012). The halved steps stop buying a
/// resolvable decrease short of that margin, and the mint refuses by the
/// `representability-face` rung with the checkpoint inside the box.
#[test]
fn a_tail_that_needs_a_representability_face_is_refused_by_type_3012() {
    let (outcome, published) = certify_scripted_2954(
        2_000,
        5.0,
        (-20.0, 10.0),
        Some((true, false)),
        TAIL_2954,
        None,
    );
    let error = outcome.expect_err("a tail that needs the literal face must not certify");
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
    assert!(
        published[0] > 8.5 && published[0] < 9.5,
        "the checkpoint stays short of the literal face's rail margin: {}",
        published[0],
    );
}

/// The same tail toward `ρ → −∞`, `V = n·(0.6 + a·e^ρ)` from `ρ = −5`, on a term
/// whose unpenalized fit its data do not identify, so the route declares the
/// lower bound a representability face (`rho_domain::unpenalized_fit_is_identified`).
/// λ → 0 is then no limit model, and the mint never rails there; the polish
/// follows the tail inside the box and certifies it once the decrease left is
/// below the tolerance (#3012).
#[test]
fn a_tail_toward_an_unidentified_unpenalized_fit_is_certified_inside_the_box_3012() {
    const LOWER_TAIL_2954: fn(f64) -> [f64; 3] = |rho| {
        let a = 3.0e-2 * 5.0_f64.exp() / 2_000.0;
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
    let certificate = outcome.expect("the tail is followed inside the box and certified");
    assert_eq!(certificate.stationarity.rung().label, "newton-decrement");
    let polish = certificate
        .newton_polish
        .expect("the certificate records the polish");
    assert!(
        polish.rails.is_empty(),
        "never railed at a literal face: {polish:?}"
    );
    assert!(
        published[0] < -7.0 && published[0] > -20.0,
        "certified inside the box: {}",
        published[0],
    );
}

/// The objective band is the error the evaluated `V` carries, by term (#2954),
/// and the verdict certifies the decrease left against the statistical
/// resolution `τ_stat = 1/(2n) = 5e-4` less that band (C3). A criterion `V = 1`
/// summed from channels `+6e8` and `−6e8 + 1` rounds like its channels,
/// `γ_(n+p²)·1.2e9 ≈ 1.5e-4`, not like its sum, so `λ̂² = 4e-4` certifies against
/// `τ_stat − γ_1·|V|` and not against `τ_stat` less the channel band. Channels
/// `±1.4e9` round past `τ_stat/2`: the arithmetic, not the statistics, limits the
/// verdict, which then certifies against the band itself. A `log|H_β|` read from
/// a factor whose own first-order forward error is `δ_logdet = 4e-4` carries
/// `½·δ_logdet = 2e-4` into the `½·log|H_β|` channel, so `λ̂² = 4e-4` certifies on
/// the channels alone and not once the factor is charged. A `log|H_β|` channel
/// from a factor that derives no forward error takes no verdict. Nor does a band
/// past `τ_stat`, which would read any statistically resolvable decrease as noise.
#[test]
fn the_objective_band_charges_the_channels_and_the_inner_factor_2954() {
    let config = OuterConfig {
        tolerance: OUTER_TOL_2954,
        problem_size: crate::rho_optimizer::OuterProblemSize {
            n_obs: Some(1_000),
            p_coefficients: Some(COEFFICIENTS_2954),
        },
        ..OuterConfig::default()
    };
    type Factor = Option<crate::estimate::outer_eval_capture::InnerFactorCondition>;
    let verdict = |lambda_sq: f64, channels: Option<(f64, f64)>, factor: Factor| {
        let gradient = array![lambda_sq.sqrt()];
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
    let tau_stat = 0.5 / 1_000.0;
    let control = verdict(4.0e-4, None, None).expect("the parts cover the coordinate");
    assert!(
        control.verdict.is_certified(),
        "control: against τ_stat − γ_1·|V| the decrement certifies: {control:?}",
    );
    assert_eq!(control.tolerance.tau_stat, tau_stat);
    let cancelling = Some((6.0e8, -6.0e8 + 1.0));
    let charged = verdict(4.0e-4, cancelling, exact).expect("the parts cover the coordinate");
    assert!(!charged.verdict.is_certified(), "{charged:?}");
    assert!(charged.objective_band.channels > 1.0e-4, "{charged:?}");
    assert!(!charged.tolerance.arithmetic_limited(), "{charged:?}");
    // Channels whose band passes `τ_stat/2` leave the arithmetic, not the
    // statistics, to limit the verdict: it certifies against the band itself.
    let limited = verdict(3.0e-4, Some((1.4e9, -1.4e9 + 1.0)), exact)
        .expect("the parts cover the coordinate");
    assert!(limited.tolerance.arithmetic_limited(), "{limited:?}");
    assert_eq!(
        limited.tolerance.value(),
        limited.objective_band.total(),
        "{limited:?}"
    );
    assert!(limited.verdict.is_certified(), "{limited:?}");
    let plain = Some((0.5, 0.5));
    let factor = crate::estimate::outer_eval_capture::InnerFactorCondition {
        logdet_forward_error: 4.0e-4,
    };
    let channels_only = verdict(4.0e-4, plain, exact).expect("the parts cover the coordinate");
    assert!(
        channels_only.verdict.is_certified(),
        "control: against the channels alone the decrement certifies: {channels_only:?}",
    );
    let conditioned = verdict(4.0e-4, plain, Some(factor)).expect("the parts cover the coordinate");
    assert!(!conditioned.verdict.is_certified(), "{conditioned:?}");
    assert!(
        conditioned.objective_band.factor > 1.0e-4,
        "{conditioned:?}"
    );
    // A nonzero `log|H_β|` channel whose factor derives no forward error takes no
    // verdict, rather than being charged nothing for it.
    assert_eq!(
        verdict(4.0e-4, plain, None).err(),
        Some(DecrementVerdictNotTaken::NoLogdetForwardError)
    );
    // A band past `τ_stat` would read any statistically resolvable decrease as
    // noise: no verdict is taken, and nothing is certified.
    let vacuous = crate::estimate::outer_eval_capture::InnerFactorCondition {
        logdet_forward_error: 1.0,
    };
    let refused =
        verdict(4.0e-4, plain, Some(vacuous)).expect_err("a vacuous band takes no verdict");
    let DecrementVerdictNotTaken::ObjectiveNotResolvable { band_f, tau_stat } = refused else {
        panic!("a vacuous band is refused as unresolvable: {refused}");
    };
    assert!(band_f > tau_stat, "{refused}");
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
    let a = 1.5e-2 * 5.0_f64.exp() / n;
    let b = 1.0e-3;
    let value = move |rho: &Array1<f64>| {
        let gap = rho[0] - rho[1];
        n * (0.6 + a * ((-rho[0]).exp() + (-rho[1]).exp()) + b * gap * gap)
    };
    let config = OuterConfig {
        tolerance: OUTER_TOL_2954,
        problem_size: crate::rho_optimizer::OuterProblemSize {
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
    // The leader is railed at its bound, and the path carries its partner there
    // up to the rounding of the step.
    assert!(
        result
            .rho
            .iter()
            .all(|rho| (20.0 - rho).abs() <= 64.0 * f64::EPSILON * 20.0),
        "both at their bounds: {:?}",
        result.rho
    );
    let polish = certificate
        .newton_polish
        .expect("the polish records its rails");
    assert!(!polish.rails.is_empty(), "{polish:?}");
    // The pair's face is taken along projected Newton's path: the leader is railed
    // at its bound, and the path carries its partner to the same face with it.
    let mut railed: Vec<usize> = certificate
        .railed_facts
        .iter()
        .filter(|fact| fact.face == crate::model_types::RailFaceKind::LimitModel)
        .map(|fact| fact.index)
        .collect();
    railed.sort_unstable();
    assert_eq!(railed, vec![0, 1], "{:?}", certificate.railed_facts);
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
/// accepted by: railing the arrivals too would lower `V` by no more than the
/// tolerance `τ_stat − band_f`, with `band_f = γ_1·|V|` because the scripted
/// criterion publishes no channels.
#[test]
fn projected_newton_path_rails_the_tail_and_leaves_the_interior_free_2954() {
    let n_obs = 2_000;
    let n = n_obs as f64;
    let a = 3.0e-2 / 3.0 * 15.7_f64.exp() / n;
    let (b, c, optimum) = (1.0e-3, 1.0e-3, -10.4 + 1.0e-5);
    let value = move |rho: &Array1<f64>| {
        let tail: f64 = (2..5).map(|k| a * (-rho[k]).exp()).sum();
        let coupling = (rho[2] - rho[3]).powi(2) + (rho[3] - rho[4]).powi(2);
        let interior = (rho[0] - optimum).powi(2) + (rho[1] - optimum).powi(2);
        n * (0.6 + tail + b * coupling + c * interior)
    };
    let config = OuterConfig {
        tolerance: OUTER_TOL_2954,
        problem_size: crate::rho_optimizer::OuterProblemSize {
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
    let tol = crate::rho_optimizer::decrement_bands::DecrementTolerance {
        tau_stat: TAU_STAT_2954,
        band_f: gam_linalg::roundoff::accumulation_growth(1) * result.final_value.abs(),
    }
    .value();
    assert!(
        result.final_value - value(&on_face) <= tol,
        "the tail ends on its face to the criterion's resolution: railing it lowers V by \
         {:.3e} against the tolerance {tol:.3e} at {:?}",
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
        problem_size: crate::rho_optimizer::OuterProblemSize {
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
/// the current point at says it lowers `V` by more than the tolerance (#2954).
///
/// `V = n·(0.6 + s·√(1 + ρ²))` from `ρ = 1.5` with `n·s = 0.01`: `λ₀ ≈ 0.20` is
/// inside the quadratic region, but the Newton step `−ρ(1 + ρ²) = −4.875`
/// overshoots to `ρ = −3.375` and raises `V` by `0.0172`. A value-only probe that
/// reads `0.02` low there would call that a decrease. Judged at the certificate's
/// own order it is a rise: the step is not kept, the judged point stays where it
/// is, and the mint refuses by name.
///
/// Conversely, on `V = n·(0.6 + ½ρ²)` from `ρ = 2e-3` a value-only probe that
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
    // The full Newton step `−ρ·(1 + ρ²)` overshoots from 1.5 to −3.375 and raises
    // the full-order criterion, while the value-only route reads it 0.02 lower, a
    // decrease. Judged at the certificate's order it is not kept: the damped half
    // step is (#3012), and the walk certifies the minimum.
    let (overshoot, judged) = certify_two_route_2954(1.5, OVERSHOOT_2954, -0.02);
    let certificate =
        overshoot.expect("the damped walk reaches the minimum and certifies it (#3012)");
    let polish = certificate
        .newton_polish
        .expect("the polish records its steps");
    let overshoot_full_order = |rho: f64| 2_000.0 * OVERSHOOT_2954(rho)[0];
    let half = 1.5 - 0.5 * 1.5 * (1.0 + 1.5 * 1.5);
    assert_eq!(
        polish.decreases[0],
        overshoot_full_order(1.5) - overshoot_full_order(half),
        "the first kept step is the half step, judged at full order: {polish:?}",
    );
    // Near its minimum `λ̂² = n·s·ρ²·√(1 + ρ²) ≥ n·s·ρ²`, so the certified
    // decrement bounds the distance left.
    assert!(polish.lambda_sq_after <= TAU_STAT_2954, "{polish:?}");
    assert!(
        judged.rho[0].abs() <= (polish.lambda_sq_after / 0.01).sqrt(),
        "published {:?}",
        judged.rho
    );

    const QUADRATIC_2954: fn(f64) -> [f64; 3] = |rho| [0.6 + 0.5 * rho * rho, rho, 1.0];
    let (reaching, reached) = certify_two_route_2954(2.0e-3, QUADRATIC_2954, 1.0);
    let certificate = reaching.expect("the full-order decrease is kept and the optimum certifies");
    assert_eq!(certificate.stationarity.rung().label, "newton-decrement");
    let polish = certificate
        .newton_polish
        .expect("the polish records its step");
    let full_order = |rho: f64| 2_000.0 * QUADRATIC_2954(rho)[0];
    assert_eq!(polish.decreases.len(), 1, "{polish:?}");
    assert_eq!(
        polish.decreases[0],
        full_order(2.0e-3) - full_order(reached.rho[0]),
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
        problem_size: crate::rho_optimizer::OuterProblemSize {
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

/// An exponential tail with its minimum inside the box, like the exact-fit
/// `y ~ s(x)` of #3012: `V = n·(0.6 + a·e^(−ρ) + b·e^(ρ−20))` with `b = a·e^(−6)`,
/// so `V` is least at `ρ* = 13` and rises again toward the bound.
const INTERIOR_TAIL_3012: fn(f64) -> [f64; 3] = |rho| {
    let a = 3.0e-2 * 10.0_f64.exp() / 2_000.0;
    let b = a * (-6.0_f64).exp();
    let tail = a * (-rho).exp();
    let rise = b * (rho - 20.0).exp();
    [0.6 + tail + rise, rise - tail, tail + rise]
};

/// From `ρ = 10`, where `λ̂² ≈ 3e-2`, [`INTERIOR_TAIL_3012`] contracts `λ̂²` by about `e^(−1)` per Newton
/// step, a linear rate `λ₊ > 2λ²`, until the rising term takes over near `ρ* = 13`.
/// Railing the coordinate at its bound `ρ = 20` raises the criterion, so the rail
/// is declined (#3012). The polish continues while the decrement contracts and each
/// step lowers the criterion by more than the tolerance, and certifies the interior
/// minimum on the decrement rung.
#[test]
fn a_linearly_converging_polish_certifies_its_interior_minimum_3012() {
    let (outcome, published) = certify_scripted_2954(
        2_000,
        10.0,
        (-20.0, 20.0),
        Some((true, true)),
        INTERIOR_TAIL_3012,
        None,
    );
    let certificate = outcome.expect("the polish reaches the interior minimum and certifies it");
    assert_eq!(certificate.stationarity.rung().label, "newton-decrement");
    let polish = certificate
        .newton_polish
        .expect("the certificate records the polish");
    assert!(
        polish.decreases.len() > 1,
        "the polish must continue past a step that contracts at a linear rate: {polish:?}",
    );
    assert!(polish.rails.is_empty(), "{polish:?}");
    // Near `ρ*` the criterion is quadratic with `V''(ρ*) = 2n·a·e^(−13)`, so the
    // certified point is one Newton step `√(λ̂²/V'')` from the minimum, up to the
    // curvature's variation over that step (a factor two covers it) and the
    // rounding of `ρ` itself.
    let a = 3.0e-2 * 10.0_f64.exp() / 2_000.0;
    let curvature = 2.0 * 2_000.0 * a * (-13.0_f64).exp();
    let reach = 2.0 * (polish.lambda_sq_after / curvature).sqrt() + 64.0 * f64::EPSILON * 13.0;
    assert!(
        (published[0] - 13.0).abs() <= reach,
        "published {:.9} against the minimum 13 within {reach:.3e}",
        published[0],
    );
}

/// `V = n·(0.6 + s·ln cosh ρ)` handed over at `ρ = 1.5`, where the full Newton
/// step `−sinh ρ·cosh ρ` overshoots to `ρ ≈ −3.5` and raises the criterion. The
/// polish damps it (#3012): the half step lowers the criterion by more than
/// the tolerance, and the walk then converges to `ρ = 0` and certifies there.
#[test]
fn a_full_newton_step_that_raises_the_criterion_is_damped_3012() {
    const SCALE: f64 = 5.0e-4;
    let (outcome, published) = certify_scripted_2954(
        2_000,
        1.5,
        (-20.0, 20.0),
        Some((true, true)),
        |rho| {
            [
                0.6 + SCALE * rho.cosh().ln(),
                SCALE * rho.tanh(),
                SCALE / (rho.cosh() * rho.cosh()),
            ]
        },
        None,
    );
    let certificate = outcome.expect("the damped walk reaches the minimum and certifies it");
    assert_eq!(certificate.stationarity.rung().label, "newton-decrement");
    let polish = certificate
        .newton_polish
        .expect("the certificate records the polish");
    let half = 1.5 - 0.5 * 1.5_f64.sinh() * 1.5_f64.cosh();
    let expected = 2_000.0 * SCALE * (1.5_f64.cosh().ln() - half.cosh().ln());
    let first = polish.decreases[0];
    assert!(
        (first - expected).abs() <= 64.0 * f64::EPSILON * 2_000.0 * 0.6,
        "the first step is the half Newton step: decrease {first:.9e} against {expected:.9e}",
    );
    // `λ̂² = n·s·sinh²ρ ≥ n·s·ρ²`, so the certified decrement bounds the distance
    // left to the minimum.
    assert!(polish.lambda_sq_after <= TAU_STAT_2954, "{polish:?}");
    assert!(
        published[0].abs() <= (polish.lambda_sq_after / (2_000.0 * SCALE)).sqrt(),
        "published {:.3e}",
        published[0]
    );
}

/// A scripted criterion whose Newton step lowers it but lands where the
/// decrement is larger: at `ρ = 1` the gradient and curvature are `c` (so
/// `λ̂² = n·c`), and at `ρ = 0` the gradient doubles (`λ̂² = 4n·c`). The polish
/// does not move along a Newton sequence that stopped contracting (#3012): it
/// refuses by name.
#[test]
fn a_decrement_that_stops_contracting_ends_the_polish_3012() {
    const C: f64 = 1.0e-5;
    let (outcome, _) = certify_scripted_2954(
        2_000,
        1.0,
        (-20.0, 20.0),
        None,
        |rho| {
            if rho > 0.5 {
                [0.6 + C * rho, C, C]
            } else {
                [0.6 + C * rho, 2.0 * C, C]
            }
        },
        None,
    );
    let message = outcome
        .expect_err("a decrement that grew after the step must not certify")
        .to_string();
    assert!(
        message.contains("Newton-decrement above tolerance after polish")
            && message.contains("the Newton decrement stopped contracting"),
        "{message}",
    );
}

/// A criterion whose value does not fall along the Newton step its own gradient
/// and Hessian describe, on a route whose bounds are both limit models: the
/// backtrack halves the step down to where the quadratic model's own decrease
/// reaches the tolerance and finds no resolved decrease, no limit face lowers the
/// criterion either, and the mint refuses by the typed
/// `newton-backtrack-unresolved` rung (#3012), with the checkpoint where it was
/// judged.
#[test]
fn a_backtrack_without_a_resolved_decrease_is_a_typed_refusal_3012() {
    let theta = 5.0e-4;
    let (outcome, published) = certify_scripted_2954(
        2_000,
        theta,
        (-20.0, 20.0),
        Some((true, true)),
        |rho| [0.6, rho, 1.0],
        None,
    );
    let error = outcome.expect_err("a decrease the criterion never delivers must not certify");
    let message = error.to_string();
    assert!(
        message.contains("Newton-decrement above tolerance after polish")
            && message.contains("no step along the Newton step lowers the criterion"),
        "{message}",
    );
    match &error {
        EstimationError::RemlDidNotConverge {
            stationarity_standard,
            ..
        } => assert_eq!(
            stationarity_standard.rung().map(|rung| rung.label),
            Some("newton-backtrack-unresolved"),
            "{message}",
        ),
        other => panic!("expected a typed outer refusal, got {other:?}"),
    }
    assert_eq!(published[0].to_bits(), theta.to_bits());
}

/// The decrement tolerance is the statistical resolution less the objective
/// band while the arithmetic resolves more than half of it, and the band itself
/// once it does not (C3): continuous at `band_f = τ_stat/2`, never above
/// `τ_stat`, and, while the statistics limit it, leaving `tol + band_f = τ_stat`
/// so a certified decrement plus the error in `V` stays inside one observation's
/// worth of log-likelihood.
#[test]
fn the_decrement_tolerance_is_the_statistical_resolution_less_the_band_c3() {
    use crate::rho_optimizer::decrement_bands::DecrementTolerance;
    let tau_stat = 0.5 / 2_000.0;
    let tolerance = |band_f: f64| DecrementTolerance { tau_stat, band_f };
    assert_eq!(tolerance(0.0).value(), tau_stat);
    assert!(!tolerance(0.0).arithmetic_limited());
    for fraction in [0.0, 0.1, 0.25, 0.49] {
        let band_f = fraction * tau_stat;
        let tol = tolerance(band_f);
        assert!(!tol.arithmetic_limited(), "{tol:?}");
        assert!(
            (tol.value() + band_f - tau_stat).abs() <= f64::EPSILON * tau_stat,
            "{tol:?}"
        );
    }
    for fraction in [0.51, 0.75, 1.0] {
        let band_f = fraction * tau_stat;
        let tol = tolerance(band_f);
        assert!(tol.arithmetic_limited(), "{tol:?}");
        assert_eq!(tol.value(), band_f, "{tol:?}");
    }
    let half = 0.5 * tau_stat;
    assert_eq!(tolerance(half).value(), half);
    let below = tolerance(half * (1.0 - f64::EPSILON)).value();
    let above = tolerance(half * (1.0 + f64::EPSILON)).value();
    assert!(
        (below - half).abs() <= 2.0 * f64::EPSILON * half
            && (above - half).abs() <= 2.0 * f64::EPSILON * half,
        "continuous at τ_stat/2: {below:e}, {above:e}",
    );
    for fraction in [0.0, 0.3, 0.5, 0.8, 1.0] {
        assert!(tolerance(fraction * tau_stat).value() <= tau_stat);
    }
}

/// Rescaling `y` by `c` shifts a Gaussian criterion by the constant `n·log c`,
/// and an additive constant changes no decision a likelihood supports. The
/// verdict's tolerance is the statistical resolution `τ_stat = 1/(2n)`, not a
/// fraction of `|V|`: at `n = 2000` and `n = 20000`, a decrement of `0.9·τ_stat`
/// certifies and one of `1.1·τ_stat` does not, whether `V` is `10` or `1e6`.
#[test]
fn the_decrement_verdict_is_invariant_to_an_additive_cost_shift_c3() {
    for n_obs in [2_000, 20_000] {
        let config = OuterConfig {
            tolerance: OUTER_TOL_2954,
            problem_size: crate::rho_optimizer::OuterProblemSize {
                n_obs: Some(n_obs),
                p_coefficients: Some(COEFFICIENTS_2954),
            },
            ..OuterConfig::default()
        };
        let tau_stat = 0.5 / n_obs as f64;
        let certified = |lambda_sq: f64, cost: f64| {
            let gradient = array![lambda_sq.sqrt()];
            crate::rho_optimizer::decrement_bands::outer_decrement_verdict(
                &config,
                &array![[1.0]],
                &gradient,
                &[],
                cost,
                &crate::estimate::outer_eval_capture::CertificateEvidence {
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
                    criterion: None,
                    inner_factor: None,
                    inner_residual: Some(EXACT_INNER_MODE_2954),
                },
            )
            .expect("the parts cover the coordinate")
            .verdict
            .is_certified()
        };
        for cost in [10.0, 1.0e6] {
            assert!(
                certified(0.9 * tau_stat, cost),
                "n={n_obs}, V={cost:e}: 0.9·τ_stat must certify",
            );
            assert!(
                !certified(1.1 * tau_stat, cost),
                "n={n_obs}, V={cost:e}: 1.1·τ_stat must not certify",
            );
        }
    }
}
