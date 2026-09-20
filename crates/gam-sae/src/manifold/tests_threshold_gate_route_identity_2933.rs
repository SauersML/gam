#![cfg(test)]
//! #2933 F03 — a ThresholdGate fit's outer hypergradient must be the derivative
//! of the criterion its route ranks, on every route that ranks one.
//!
//! Both production value routes price the exact observed information `½log|A|`.
//! The dense direct-logdet route used to exclude `ThresholdGate` from its exact-A
//! derivative channels, so with no evidence bundle it contracted the majorizer
//! `B`'s selected inverse and differentiated `∂B` while the value ranked `A`.
//! The gap `½·d/dρ log|I + B⁻¹ΔC|` is not a tolerance question.
//!
//! The arbiter is the complete criterion at the stationary root of every endpoint's
//! inner problem. Production's inner acceptance certifies the penalized OBJECTIVE
//! (`½λ²/scale` at the stall band), which resolves the root itself only to about
//! `√(tol/μ)` along a weak direction, and `½log|A|` is first order in the root. On
//! this fixture that leaves the priced criterion trajectory-dependent: job 1113848
//! re-priced the centre from the fixture start and moved the cost by −5.4e-4 and
//! +1.08e-3, and its differences did not converge in `h`. A warm endpoint a small step
//! away starts inside the band and never moves, so its difference is the fixed-θ
//! partial derivative instead. Neither is the derivative the gradient claims.
//!
//! So every root, centre and endpoints, is reconverged by production and then driven
//! to its roundoff floor by undamped exact-A Newton steps on production's arrow system
//! (`exact_a_evidence_system`), the step the terminal polish takes. Each route prices
//! the criterion at that frozen state (`inner_max_iter = 0`). Every endpoint holds the
//! central root's collapse-prevention gates (#2933 F05) and must keep its deflation
//! stratum, and the differences must agree across three steps before their Richardson
//! value is compared.
//!
//! The control prices the same endpoints with the central root held, which is the
//! fixed-θ partial and omits the implicit response. The arbiter has to reject it by a
//! material margin, or it could not tell a gradient that drops a channel from a
//! complete one.

use super::tests_sparse_curvature_operator_2500::threshold_gate_tiny_fixture;
use super::*;
use ndarray::{Array1, Array2};

const INNER_MAX_ITER: usize = 40;
const LEARNING_RATE: f64 = 0.4;
const RIDGE: f64 = 1.0e-6;
/// An undamped Newton step contracts quadratically near the root; the polish ends
/// when `‖g‖` stops falling, and this only bounds a loop that cannot.
const POLISH_MAX_STEPS: usize = 32;
/// Central-difference steps, each half the previous one, so `D(h) = D + c·h² + O(h⁴)`
/// gives two Richardson values from successive halvings.
const STEPS: [f64; 3] = [4.0e-3, 2.0e-3, 1.0e-3];
/// Relative budget shared by the analytic-vs-oracle gap and the oracle's own
/// disagreement across steps (the #2933 F05 derivative test's budget).
const BUDGET: f64 = 1.0e-4;

fn outer_objective(
    term: SaeManifoldTerm,
    target: &Array2<f64>,
    rho: &SaeManifoldRho,
    inner_max_iter: usize,
) -> SaeManifoldOuterObjective {
    SaeManifoldOuterObjective::new(
        term,
        target.clone(),
        None,
        rho.clone(),
        inner_max_iter,
        LEARNING_RATE,
        RIDGE,
        RIDGE,
    )
}

/// Drive `term` to the stationary root of its penalized objective at `rho` with
/// undamped exact-A Newton steps, keeping a step only while it lowers `‖g‖`.
/// Returns `‖g‖` at entry and at the polished state.
fn polish_root(
    term: &mut SaeManifoldTerm,
    target: &Array2<f64>,
    rho: &SaeManifoldRho,
) -> Result<(f64, f64), String> {
    let options = term.evidence_factor_options();
    let mut majorizer = term.assemble_arrow_schur(target.view(), rho, None)?;
    let entry = SaeManifoldTerm::system_grad_norm_sq(&majorizer).sqrt();
    let mut current = entry;
    for _ in 0..POLISH_MAX_STEPS {
        let exact = term.exact_a_evidence_system(target.view(), rho, &majorizer, 1.0)?;
        let (delta_t, delta_beta, _) =
            gam_solve::arrow_schur::solve_with_lm_escalation_inner(&exact, 0.0, 0.0, &options)
                .map_err(|err| format!("exact-A Newton solve: {err:?}"))?;
        let snapshot = term.snapshot_mutable_state();
        term.apply_newton_step(delta_t.view(), delta_beta.view(), 1.0)?;
        let trial = term.assemble_arrow_schur(target.view(), rho, None)?;
        let trial_norm = SaeManifoldTerm::system_grad_norm_sq(&trial).sqrt();
        if !(trial_norm < current) {
            term.restore_mutable_state(&snapshot)
                .map_err(|err| err.to_string())?;
            break;
        }
        current = trial_norm;
        majorizer = trial;
    }
    Ok((entry, current))
}

#[test]
fn threshold_gate_route_gradients_differentiate_the_reconverged_criterion_2933() {
    let routes = [("dense", true), ("streaming", false)];
    let mut failures = Vec::new();
    let mut compared = 0usize;
    let mut worst_control = 0.0_f64;
    for straddle in [false, true] {
        let (term, target, rho) = threshold_gate_tiny_fixture(straddle);
        let mut production = outer_objective(term, &target, &rho, INNER_MAX_ITER);
        let layout = production.baseline_rho.clone();
        let flat = layout.flat_coordinates();
        let centre_rho = layout
            .from_flat(flat.view())
            .expect("#2933 F03: the objective owns its typed rho layout");
        if let Err(err) = production.evaluate_outer_criterion_route(&centre_rho, true, false) {
            failures.push(format!("straddle={straddle}: the centre refused: {err}"));
            continue;
        }
        // The set the central root adopted; every priced state holds it.
        let gates = production.term.collapse_prevention_gates();
        let declared = |state: &SaeManifoldTerm| -> SaeManifoldTerm {
            let mut state = state.clone();
            state.declare_collapse_prevention_gates(&gates);
            state
        };
        let polished = |start: &SaeManifoldTerm,
                        point: &Array1<f64>,
                        reconverge: bool|
         -> Result<(SaeManifoldTerm, f64, f64), String> {
            let rho_at = layout.from_flat(point.view())?;
            let mut state = declared(start);
            if reconverge {
                let mut endpoint = outer_objective(state, &target, &rho, INNER_MAX_ITER);
                endpoint
                    .evaluate_outer_criterion_route(&rho_at, true, false)
                    .map_err(|err| format!("production reconvergence refused: {err}"))?;
                state = declared(&endpoint.term);
            }
            let (entry, after) = polish_root(&mut state, &target, &rho_at)?;
            Ok((state, entry, after))
        };
        let stratum = |state: &SaeManifoldTerm, point: &Array1<f64>| -> Result<usize, String> {
            let rho_at = layout.from_flat(point.view())?;
            declared(state)
                .penalized_quasi_laplace_criterion_with_cache(
                    target.view(),
                    &rho_at,
                    None,
                    0,
                    LEARNING_RATE,
                    RIDGE,
                    RIDGE,
                )
                .map(|(_, loss, _)| loss.criterion_gauge_deflated_directions)
                .map_err(|err| err.to_string())
        };
        let price_value =
            |state: &SaeManifoldTerm, point: &Array1<f64>, direct: bool| -> Result<f64, String> {
                let mut frozen = outer_objective(declared(state), &target, &rho, 0);
                let rho_at = layout.from_flat(point.view())?;
                frozen
                    .evaluate_outer_criterion_route(&rho_at, direct, false)
                    .map(|evaluation| evaluation.cost)
                    .map_err(|err| err.to_string())
            };
        let price_with_gradient = |state: &SaeManifoldTerm,
                                   direct: bool|
         -> Result<(f64, Array1<f64>), String> {
            let mut frozen = outer_objective(declared(state), &target, &rho, 0);
            let evaluation = frozen
                .evaluate_outer_criterion_route(&centre_rho, direct, false)
                .map_err(|err| format!("value refused: {err}"))?;
            let gradient = frozen
                .analytic_gradient_for_outer_evaluation(&centre_rho, &evaluation)
                .map_err(|err| format!("gradient refused: {err}"))?;
            Ok((evaluation.cost, gradient))
        };

        let (centre, centre_entry, centre_after) = match polished(&production.term, &flat, false) {
            Ok(polished) => polished,
            Err(err) => {
                failures.push(format!("straddle={straddle}: the centre polish refused: {err}"));
                continue;
            }
        };
        let centre_stratum = match stratum(&centre, &flat) {
            Ok(count) => count,
            Err(err) => {
                failures.push(format!("straddle={straddle}: the centre stratum refused: {err}"));
                continue;
            }
        };
        println!(
            "[#2933 F03] straddle={straddle} centre root ‖g‖ {centre_entry:.3e} → \
             {centre_after:.3e} after the exact-A polish; deflated directions {centre_stratum}"
        );
        let mut centre_prices = Vec::with_capacity(routes.len());
        for &(route, direct) in &routes {
            match price_with_gradient(&centre, direct) {
                Ok((cost, gradient)) => {
                    println!(
                        "[#2933 F03] straddle={straddle} route={route} cost={cost:.12e} \
                         gradient={:?}",
                        gradient.to_vec()
                    );
                    centre_prices.push(Some(gradient));
                }
                Err(err) => {
                    failures.push(format!("straddle={straddle} route={route}: {err}"));
                    centre_prices.push(None);
                }
            }
        }

        for coordinate in 0..flat.len() {
            let mut differences = vec![[None::<f64>; STEPS.len()]; routes.len()];
            let mut controls = vec![[None::<f64>; STEPS.len()]; routes.len()];
            for (index, &step) in STEPS.iter().enumerate() {
                let mut plus = flat.clone();
                plus[coordinate] += step;
                let mut minus = flat.clone();
                minus[coordinate] -= step;
                let mut roots = Vec::with_capacity(2);
                for point in [&plus, &minus] {
                    let root = polished(&centre, point, true).and_then(|(state, _, after)| {
                        stratum(&state, point).map(|count| (state, after, count))
                    });
                    match root {
                        Ok((state, after, count)) if count == centre_stratum => {
                            roots.push((state, after));
                        }
                        Ok((_, _, count)) => failures.push(format!(
                            "straddle={straddle} coord={coordinate} h={step:.1e}: the endpoint \
                             deflates {count} directions against the centre's {centre_stratum}, \
                             so the difference would straddle a stratum"
                        )),
                        Err(err) => failures.push(format!(
                            "straddle={straddle} coord={coordinate} h={step:.1e}: endpoint \
                             refused: {err}"
                        )),
                    }
                }
                let [(up_state, up_norm), (down_state, down_norm)] = roots.as_slice() else {
                    continue;
                };
                for (slot, &(route, direct)) in routes.iter().enumerate() {
                    let reconverged = (
                        price_value(up_state, &plus, direct),
                        price_value(down_state, &minus, direct),
                    );
                    let held = (
                        price_value(&centre, &plus, direct),
                        price_value(&centre, &minus, direct),
                    );
                    match (reconverged, held) {
                        ((Ok(up), Ok(down)), (Ok(held_up), Ok(held_down))) => {
                            let fd = (up - down) / (2.0 * step);
                            let held_fd = (held_up - held_down) / (2.0 * step);
                            differences[slot][index] = Some(fd);
                            controls[slot][index] = Some(held_fd);
                            println!(
                                "[#2933 F03] straddle={straddle} route={route} coord={coordinate} \
                                 h={step:.1e} fd={fd:.10e} held_root_fd={held_fd:.10e} endpoint \
                                 ‖g‖ {up_norm:.2e}/{down_norm:.2e}"
                            );
                        }
                        ((Err(err), _), _)
                        | ((_, Err(err)), _)
                        | (_, (Err(err), _))
                        | (_, (_, Err(err))) => failures.push(format!(
                            "straddle={straddle} route={route} coord={coordinate} h={step:.1e}: \
                             pricing refused: {err}"
                        )),
                    }
                }
            }
            for (slot, &(route, _)) in routes.iter().enumerate() {
                let Some(gradient) = centre_prices[slot].as_ref() else {
                    continue;
                };
                let analytic = gradient[coordinate];
                let ([Some(d1), Some(d2), Some(d3)], [_, Some(c2), Some(c3)]) =
                    (differences[slot], controls[slot])
                else {
                    failures.push(format!(
                        "straddle={straddle} route={route} coord={coordinate}: a difference is \
                         missing"
                    ));
                    continue;
                };
                let coarse = (4.0 * d2 - d1) / 3.0;
                let fine = (4.0 * d3 - d2) / 3.0;
                let held = (4.0 * c3 - c2) / 3.0;
                let scale = 1.0 + analytic.abs().max(fine.abs());
                let oracle_error = (fine - coarse).abs() / scale;
                let gap = (analytic - fine).abs() / scale;
                let control_gap = (analytic - held).abs() / scale;
                worst_control = worst_control.max(control_gap);
                println!(
                    "[#2933 F03] straddle={straddle} route={route} coord={coordinate} \
                     analytic={analytic:.10e} richardson={fine:.10e} oracle_error={oracle_error:.3e} \
                     gap={gap:.3e} held_root_gap={control_gap:.3e}"
                );
                if oracle_error > BUDGET {
                    failures.push(format!(
                        "straddle={straddle} route={route} coord={coordinate}: the reconverged \
                         criterion is not h-convergent (Richardson {coarse:.10e} vs {fine:.10e}, \
                         relative {oracle_error:.3e})"
                    ));
                } else if gap + oracle_error > BUDGET {
                    failures.push(format!(
                        "straddle={straddle} route={route} coord={coordinate}: analytic \
                         {analytic:.10e} vs reconverged Richardson {fine:.10e} (relative gap \
                         {gap:.3e}, oracle error {oracle_error:.3e})"
                    ));
                }
                compared += 1;
            }
        }
    }
    println!(
        "[#2933 F03] compared {compared} coordinates; {} failures; worst held-root gap \
         {worst_control:.3e}",
        failures.len()
    );
    assert!(
        failures.is_empty(),
        "#2933 F03: a ThresholdGate route returned a hypergradient that is not the \
         derivative of the criterion it ranks:\n{}",
        failures.join("\n")
    );
    assert!(
        worst_control > 10.0 * BUDGET,
        "#2933 F03 control: the held-root difference departs from the analytic gradient by \
         only {worst_control:.3e}, so at this fixture the implicit response is not material \
         and the arbiter cannot tell a gradient that drops a channel from a complete one"
    );
}

/// #2933 F03 — the outer-gradient assembler accepts only the two routes that
/// differentiate the `½log|A|` value: the dense exact-A route (the evaluation's spectral
/// block, no bundle, no matrix-free system) and the streaming exact-A route (an
/// `ExactObservedInformation` bundle together with its system). Every other pairing
/// would contract `B` channels, or two operators' inverses, against an A-valued score,
/// and has to be refused before any channel is produced. #2267 — a dense route without
/// its block would decompose `A` again for each consumer, so it is refused too. The legal
/// dense pairing on the same converged state is the control: the refusals are not a
/// blanket failure.
#[test]
fn threshold_gate_gradient_refuses_every_pairing_that_is_not_an_exact_a_route_2933() {
    let (mut term, target, rho) = threshold_gate_tiny_fixture(false);
    let (_, loss, cache) = term
        .penalized_quasi_laplace_criterion_with_cache(
            target.view(),
            &rho,
            None,
            INNER_MAX_ITER,
            LEARNING_RATE,
            RIDGE,
            RIDGE,
        )
        .expect("#2933 F03: the fixture prices a dense criterion at its own rho");
    let system = term
        .assemble_arrow_schur(target.view(), &rho, None)
        .expect("#2933 F03: the converged state assembles its majorizer system");
    let geometry = term
        .materialize_dense_exact_a_geometry(&rho, target.view(), &cache)
        .expect("#2267: the converged state's exact-A spectral block");
    let bundle = |operator: EvidenceOperator| BundleEvidenceGeometry {
        operator,
        cache: &cache,
        probes: &[],
        sinv: &[],
    };
    let assemble = |evidence: Option<BundleEvidenceGeometry<'_>>,
                    matrix_free_system: Option<&ArrowSchurSystem>,
                    dense_geometry: Option<&DenseExactAGeometry>| {
        term.analytic_outer_rho_gradient_components_with_bundle(
            target.view(),
            &rho,
            &loss,
            &cache,
            evidence,
            matrix_free_system,
            dense_geometry,
        )
    };
    let control = assemble(None, None, Some(&geometry));
    assert!(
        control.is_ok(),
        "#2933 F03 control: the dense exact-A pairing must assemble a gradient: {:?}",
        control.err()
    );
    let illegal = [
        (
            "system without its bundle",
            assemble(None, Some(&system), Some(&geometry)),
        ),
        (
            "majorizer bundle with its system",
            assemble(Some(bundle(EvidenceOperator::Majorizer)), Some(&system), None),
        ),
        (
            "exact-A bundle without its system",
            assemble(Some(bundle(EvidenceOperator::ExactObservedInformation)), None, None),
        ),
        (
            "dense route without its spectral block",
            assemble(None, None, None),
        ),
        (
            "exact-A bundle and system with a dense spectral block",
            assemble(
                Some(bundle(EvidenceOperator::ExactObservedInformation)),
                Some(&system),
                Some(&geometry),
            ),
        ),
    ];
    let mut failures = Vec::new();
    for (pairing, outcome) in illegal {
        match outcome {
            Err(err) if err.to_string().contains("pairs evidence operator") => {
                println!("[#2933 F03] {pairing}: refused: {err}");
            }
            Err(err) => failures.push(format!("{pairing}: failed for another reason: {err}")),
            Ok(_) => failures.push(format!("{pairing}: assembled a gradient")),
        }
    }
    assert!(
        failures.is_empty(),
        "#2933 F03: a derivative route that does not differentiate the A-valued criterion \
         was not refused:\n{}",
        failures.join("\n")
    );
}
