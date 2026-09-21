use super::*;
use ndarray::array;

/// #2735: the face law's pulled-back falsification point is a trial point, and a
/// trial the criterion refuses is a statement about that point. The falsification
/// must DECLINE with the refusal's reason — never end the fit through `?` — and it
/// must still restore the certified point before judging.
#[test]
fn face_law_falsification_declines_at_a_refused_pulled_back_point_2735() {
    const PLANTED: &str = "planted refusal at the pulled-back face point";
    let rho_hat = 30.0_f64;
    let limit = RailFaceLimit {
        face: vec![0],
        face_rho: vec![rho_hat],
        first_order_form: Array2::from_diag(&array![4.0, 6.0]),
        released_penalties: vec![Array2::from_diag(&array![2.0, 3.0])],
        released_score: array![1.0, 2.0],
        form_conditioning: 1.0,
        limit_beta: Array1::zeros(0),
        limit_dispersion: 1.0,
        released_curvature_drift: None,
    };
    let proof = match certify_rail_face(&limit) {
        RailFaceVerdict::Certified(proof) => proof,
        other => panic!("the diagonal positive-definite face must certify, got {other:?}"),
    };
    let problem = OuterProblem::new(1).with_gradient(Derivative::Analytic);
    let mut obj = problem.build_objective(
        Vec::<f64>::new(),
        move |seen: &mut Vec<f64>, rho: &Array1<f64>| {
            seen.push(rho[0]);
            if rho[0] < rho_hat - 0.5 {
                Err(EstimationError::TrialPointRefused {
                    reason: PLANTED.to_string(),
                })
            } else {
                Ok(1.0)
            }
        },
        |_: &mut Vec<f64>, rho: &Array1<f64>| {
            Ok(OuterEval {
                cost: 1.0,
                gradient: Array1::zeros(rho.len()),
                hessian: HessianValue::Unavailable,
                inner_beta_hint: None,
            })
        },
        None::<fn(&mut Vec<f64>)>,
        None::<fn(&mut Vec<f64>, &Array1<f64>) -> Result<EfsEval, EstimationError>>,
    );
    let rho = array![rho_hat];
    let gradient = array![0.0];
    let hessian = array![[1.0]];
    let bounds = (array![0.0], array![40.0]);
    let inputs = AsymptoteRailInputs {
        rho: &rho,
        projected_gradient: &gradient,
        railed: &[0],
        layout: OuterThetaLayout::new(1, 0),
        hessian: &hessian,
        bounds: &bounds,
        terminal_beta: None,
        stationarity_bound: StationarityBound::from_ladder(1.0e-3, StationarityBoundSource::SolverBand),
        objective_tol: 1.0e-10,
        context: "face law falsification at a refused point",
        native_coordinate_order: None,
    };

    let verdict = falsify_face_law(&mut obj, &inputs, &limit, &proof)
        .expect("a refused pulled-back point must decline, not end the fit");
    match verdict {
        Err(reason) => assert!(
            reason.contains(PLANTED),
            "the decline must carry the refusal's reason, got {reason:?}"
        ),
        Ok(()) => panic!("a refused falsification point cannot falsify the face law"),
    }
    assert!(
        obj.state.iter().any(|&r| r < rho_hat - 0.5),
        "the pulled-back point must have been evaluated: {:?}",
        obj.state
    );
    assert_eq!(
        obj.state.last().copied(),
        Some(rho_hat),
        "the certified point must be restored before the decline: {:?}",
        obj.state
    );
}

/// Build a one-coordinate UPPER-rail tail-law objective: at ρ its gradient is
/// `−c·e^{−ρ}` (so `ĉ = −e^{ρ}·grad = c` is constant) and its published inner
/// β is `a·e^{−ρ}` (so consecutive-probe `‖Δβ‖` contracts geometrically).
/// `drift_amp` ramps `ĉ` with ρ to model the finite-difference noise regime
/// (a non-constant pencil constant that no drift band can confirm).
fn upper_tail_objective(c: f64, a: f64, drift_amp: f64) -> impl OuterObjective {
    let problem = OuterProblem::new(1).with_gradient(Derivative::Analytic);
    problem.build_objective(
        (),
        move |_: &mut (), rho: &Array1<f64>| {
            let r = rho[0];
            let c_eff = c + drift_amp * r;
            Ok((c_eff * (-r).exp()).abs())
        },
        move |_: &mut (), rho: &Array1<f64>| {
            let r = rho[0];
            let c_eff = c + drift_amp * r;
            Ok(OuterEval {
                cost: (c_eff * (-r).exp()).abs(),
                gradient: array![-c_eff * (-r).exp()],
                hessian: HessianValue::Unavailable,
                inner_beta_hint: Some(array![a * (-r).exp()]),
            })
        },
        None::<fn(&mut ())>,
        None::<fn(&mut (), &Array1<f64>) -> Result<EfsEval, EstimationError>>,
    )
}

/// An exact upper-rail exponential tail is certified: the reconstructed
/// pencil constant is `c`, and the value-gap / estimand-travel are finite.
#[test]
fn asymptote_rail_mints_on_exact_tail_law() {
    let mut obj = upper_tail_objective(6723.0, 1.0, 0.0);
    let rho = array![29.9];
    let tol = AsymptoteTolerances::exp4_rail_bands(1.0e-2);
    let rail = build_and_assess_rail_coordinate(
        &mut obj,
        &rho,
        0,
        AsymptoteSide::Upper,
        &tol,
        (f64::NEG_INFINITY, f64::INFINITY),
        0,
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
    assert!(rail.value_gap.is_finite() && rail.value_gap >= 0.0);
    assert!(rail.estimand_travel_bound.is_finite() && rail.estimand_travel_bound >= 0.0);
}

/// A drifting pencil constant (finite-difference noise regime) never
/// certifies: no finite-difference-clean run of the required length exists.
#[test]
fn asymptote_rail_refuses_on_drifting_constant() {
    let mut obj = upper_tail_objective(6723.0, 1.0, 3000.0);
    let rho = array![29.9];
    let tol = AsymptoteTolerances::exp4_rail_bands(1.0e-2);
    let verdict = build_and_assess_rail_coordinate(
        &mut obj,
        &rho,
        0,
        AsymptoteSide::Upper,
        &tol,
        (f64::NEG_INFINITY, f64::INFINITY),
        0,
    )
    .expect("probing must not error");
    assert!(
        verdict.is_err(),
        "a drifting ĉ must not certify a tail, got {verdict:?}",
    );
}

/// #2358: a finite smoothing box can expose fewer than three whole
/// e-folds of the leading-order tail even though the compactified
/// criterion is already regular there. For
///
/// `V(ρ) = c·e⁻ρ + (d/2)·e⁻²ρ`,
///
/// the pencil constant is `ĉ(ρ) = c + d·e⁻ρ`: the ordinary asymptotic law
/// plus its first vanishing correction. Unit probes from ρ=10 step into
/// enough correction curvature that no three-row clean run exists; the
/// equal-spaced half-e-fold fallback resolves the local tail without
/// changing the drift, estimand, sign, or noise gates.
#[test]
fn tail_probe_resolves_narrow_regular_band_before_finite_box_2358() {
    let (c, d, a) = (7.1_f64, 400.0_f64, 1.0e-3_f64);
    let problem = OuterProblem::new(1).with_gradient(Derivative::Analytic);
    let mut obj = problem.build_objective(
        (),
        move |_: &mut (), rho: &Array1<f64>| {
            let tau = (-rho[0]).exp();
            Ok(c * tau + 0.5 * d * tau * tau)
        },
        move |_: &mut (), rho: &Array1<f64>| {
            let tau = (-rho[0]).exp();
            Ok(OuterEval {
                cost: c * tau + 0.5 * d * tau * tau,
                gradient: array![-c * tau - d * tau * tau],
                hessian: HessianValue::Unavailable,
                inner_beta_hint: Some(array![a * tau]),
            })
        },
        None::<fn(&mut ())>,
        None::<fn(&mut (), &Array1<f64>) -> Result<EfsEval, EstimationError>>,
    );
    let rho = array![10.0];
    let mut tol = AsymptoteTolerances::exp4_rail_bands(1.0e-2);
    tol.tail_drift_rel = TAIL_SNAP_DRIFT_REL;

    let (coarse, coarse_rows) = probe_tail_window_at_resolution(
        &mut obj,
        &rho,
        0,
        AsymptoteSide::Upper,
        &tol,
        (-10.0, 10.5),
        (1.0, ASYMPTOTE_PROBE_COUNT),
    )
    .expect("coarse probing must not error");
    assert!(
        coarse.is_none(),
        "unit probes must not manufacture a pure tail through the curved band: {coarse_rows}"
    );

    let (window, rows) = probe_tail_window(
        &mut obj,
        &rho,
        0,
        AsymptoteSide::Upper,
        &tol,
        (-10.0, 10.5),
    )
    .expect("multi-resolution probing must not error");
    let window = window.unwrap_or_else(|| {
        panic!("the half-e-fold fallback must resolve the regular tail: {rows}")
    });
    assert!(
        matches!(
            assess_coordinate(&window, &tol),
            AsymptoteVerdict::CertifiedAtAsymptote { .. }
        ),
        "the resolved tail must pass the unchanged asymptote gates"
    );
}

/// #2358: pre-snap interior stationarity cannot authorize direct
/// certification at the rail. The second coordinate is stationary at the
/// checkpoint but its mode depends on the tail coordinate, so moving the
/// first coordinate changes the second coordinate's optimum. A confirmed
/// snap must therefore publish a re-optimization waypoint even though the
/// pre-snap interior gradient is exactly zero.
#[test]
fn coupled_coordinate_stationary_before_snap_still_reseeds_2358() {
    let c = 10.0_f64;
    let problem = OuterProblem::new(2).with_gradient(Derivative::Analytic);
    let mut obj = problem.build_objective(
        (),
        move |_: &mut (), rho: &Array1<f64>| {
            let tau = (-rho[0]).exp();
            let coupled = rho[1] - tau;
            Ok(c * tau + 0.5 * coupled * coupled)
        },
        move |_: &mut (), rho: &Array1<f64>| {
            let tau = (-rho[0]).exp();
            Ok(OuterEval {
                cost: c * tau + 0.5 * (rho[1] - tau).powi(2),
                gradient: array![
                    -c * tau + (rho[1] - tau) * tau,
                    rho[1] - tau
                ],
                hessian: HessianValue::Unavailable,
                inner_beta_hint: Some(array![1.0e-3 * tau]),
            })
        },
        None::<fn(&mut ())>,
        None::<fn(&mut (), &Array1<f64>) -> Result<EfsEval, EstimationError>>,
    );
    let rho = array![8.0, (-8.0_f64).exp()];
    let tail_gradient = -c * (-rho[0]).exp();
    let gradient = array![tail_gradient, 0.0];
    let hessian = array![[tail_gradient.abs(), 0.0], [0.0, 1.0]];
    let bounds = (Array1::from_elem(2, -12.0), Array1::from_elem(2, 12.0));

    let outcome = try_tail_snap_to_rail(
        &mut obj,
        &AsymptoteRailInputs {
            rho: &rho,
            projected_gradient: &gradient,
            railed: &[],
            layout: OuterThetaLayout::new(2, 0),
            hessian: &hessian,
            bounds: &bounds,
            terminal_beta: None,
            stationarity_bound: StationarityBound::from_ladder(1.0e-3, StationarityBoundSource::SolverBand),
            objective_tol: 1.0e-8,
            context: "coupled pre-snap stationarity guard",
            native_coordinate_order: None,
        },
    )
    .expect("tail snap must not error");
    match outcome {
        TailSnapOutcome::ConfirmedNeedsReseed(snapped) => {
            assert_eq!(snapped, array![12.0, (-8.0_f64).exp()]);
        }
        other => panic!(
            "a coupled coordinate stationary only before the snap must reoptimize, got {other:?}"
        ),
    }
}

/// #2392 wrong-rail pull-back FIRES: a coordinate sitting at the UPPER bound
/// whose clean-band probes carry a POSITIVE gradient (`∂V/∂ρ > 0`, so the
/// pencil constant `ĉ = −e^{ρ}·g < 0` — descent points INWARD, away from the
/// bound) was driven to the wrong rail. `detect_wrong_rail_pullback` returns
/// an interior reseed target strictly below the coordinate's current ρ.
#[test]
fn wrong_rail_pullback_fires_on_inward_descent_2392() {
    // V(ρ) = −c·e^{−ρ} ⇒ ∂V/∂ρ = +c·e^{−ρ} > 0: the descent runs ρ DOWN, away
    // from the upper rail, and ĉ_upper = −e^{ρ}·(c·e^{−ρ}) = −c < 0 uniformly.
    let c = 6723.0;
    let evaluation_count = std::sync::Arc::new(std::sync::atomic::AtomicUsize::new(0));
    let count_during_eval = std::sync::Arc::clone(&evaluation_count);
    let problem = OuterProblem::new(1).with_gradient(Derivative::Analytic);
    let mut obj = problem.build_objective(
        (),
        move |_: &mut (), rho: &Array1<f64>| Ok(-c * (-rho[0]).exp()),
        move |_: &mut (), rho: &Array1<f64>| {
            count_during_eval.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
            Ok(OuterEval {
                cost: -c * (-rho[0]).exp(),
                gradient: array![c * (-rho[0]).exp()],
                hessian: HessianValue::Unavailable,
                inner_beta_hint: Some(array![(-rho[0]).exp()]),
            })
        },
        None::<fn(&mut ())>,
        None::<fn(&mut (), &Array1<f64>) -> Result<EfsEval, EstimationError>>,
    );
    let rho = array![29.9];
    let tol = AsymptoteTolerances::exp4_rail_bands(1.0e-2);
    let target = detect_wrong_rail_pullback(
        &mut obj,
        &rho,
        0,
        AsymptoteSide::Upper,
        &tol,
        (-30.0, 30.0),
    )
    .expect("probing the wrong-rail objective must not error")
    .expect("an inward-descent rail must publish a pull-back target");
    assert!(
        target < rho[0] && target.is_finite(),
        "the reseed must move the coordinate INWARD (ρ down), got {target}",
    );
    assert_eq!(
        evaluation_count.load(std::sync::atomic::Ordering::Relaxed),
        5,
        "two near-rail rows are below the gradient floor; the next three \
         clean rows are already a complete wrong-rail proof"
    );
}

/// #2392 wrong-rail pull-back does NOT fire on a GENUINE upper-rail tail:
/// `∂V/∂ρ < 0` ⇒ `ĉ > 0` ⇒ descent runs TOWARD the bound (a real λ→∞ optimum),
/// which must never be pulled off its rail.
#[test]
fn wrong_rail_pullback_refuses_a_genuine_upper_tail_2392() {
    let c = 6723.0;
    let evaluation_count = std::sync::Arc::new(std::sync::atomic::AtomicUsize::new(0));
    let count_during_eval = std::sync::Arc::clone(&evaluation_count);
    let problem = OuterProblem::new(1).with_gradient(Derivative::Analytic);
    let mut obj = problem.build_objective(
        (),
        move |_: &mut (), rho: &Array1<f64>| Ok(c * (-rho[0]).exp()),
        move |_: &mut (), rho: &Array1<f64>| {
            count_during_eval.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
            Ok(OuterEval {
                cost: c * (-rho[0]).exp(),
                gradient: array![-c * (-rho[0]).exp()],
                hessian: HessianValue::Unavailable,
                inner_beta_hint: Some(array![(-rho[0]).exp()]),
            })
        },
        None::<fn(&mut ())>,
        None::<fn(&mut (), &Array1<f64>) -> Result<EfsEval, EstimationError>>,
    );
    let rho = array![29.9];
    let tol = AsymptoteTolerances::exp4_rail_bands(1.0e-2);
    let verdict = detect_wrong_rail_pullback(
        &mut obj,
        &rho,
        0,
        AsymptoteSide::Upper,
        &tol,
        (-30.0, 30.0),
    )
    .expect("probing must not error");
    assert!(
        verdict.is_none(),
        "a genuine λ→∞ tail (ĉ>0) must not be pulled off its rail, got {verdict:?}",
    );
    assert_eq!(
        evaluation_count.load(std::sync::atomic::Ordering::Relaxed),
        5,
        "the first three finite-difference-clean rows prove a genuine rail; \
         probing thirteen more interior points cannot change that local fact"
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

/// Joint-face objective `V(ρ) = c·e^{−(ρ₀+ρ₁)/2}` — the algebraic skeleton
/// of an OVERLAPPING-penalty λ→∞ face (the coalesced pseudo-logdet's
/// shared range space couples the two coordinates, so each marginal
/// gradient `g_k = −(c/2)e^{−(ρ₀+ρ₁)/2}` decays in the JOINT coordinate
/// only). Along either single coordinate the pencil constant
/// `ĉ_k = |g_k|e^{ρ_k}` sweeps a factor `e^{1/2}` per e-fold — far outside
/// any drift band — while along the face direction it is exactly constant.
fn joint_face_objective(c: f64, a: f64) -> impl OuterObjective {
    let problem = OuterProblem::new(2).with_gradient(Derivative::Analytic);
    problem.build_objective(
        (),
        move |_: &mut (), rho: &Array1<f64>| Ok(c * (-(rho[0] + rho[1]) / 2.0).exp()),
        move |_: &mut (), rho: &Array1<f64>| {
            let v = c * (-(rho[0] + rho[1]) / 2.0).exp();
            Ok(OuterEval {
                cost: v,
                gradient: array![-0.5 * v, -0.5 * v],
                hessian: HessianValue::Unavailable,
                inner_beta_hint: Some(array![a * (-(rho[0] + rho[1]) / 2.0).exp()]),
            })
        },
        None::<fn(&mut ())>,
        None::<fn(&mut (), &Array1<f64>) -> Result<EfsEval, EstimationError>>,
    )
}

/// #2349 round 7: the joint multi-coordinate rail face. The marginal
/// single-coordinate tail law honestly fails on an overlapping-penalty
/// face (measured on the multinomial checkpoint: ĉ₀ swept 8 orders of
/// magnitude), so tail-snap must fall back to the joint face direction,
/// certify the one-dimensional joint law there, and snap the whole face.
#[test]
fn joint_face_tail_certifies_where_single_coordinate_law_drifts_2349() {
    let c = 1.2 * (7.5_f64).exp();
    let rho = array![8.0, 7.0];
    let tol = {
        let mut t = AsymptoteTolerances::exp4_rail_bands(1.0e-2);
        t.tail_drift_rel = TAIL_SNAP_DRIFT_REL;
        t
    };
    let bounds = (Array1::from_elem(2, -12.0), Array1::from_elem(2, 12.0));

    // The marginal law must genuinely fail first — otherwise this test
    // would pass vacuously with the joint path never exercised.
    let mut obj = joint_face_objective(c, 1.0e-9);
    let (single_window, _) = probe_tail_window(
        &mut obj,
        &rho,
        0,
        AsymptoteSide::Upper,
        &tol,
        (bounds.0[0], bounds.1[0]),
    )
    .expect("single-coordinate probing must not error");
    assert!(
        single_window.is_none(),
        "the marginal pencil constant drifts e^(1/2) per e-fold and must not \
         produce a finite-difference-clean run"
    );

    // The joint window recovers the exact face constant.
    let (joint_window, _) = probe_joint_tail_window(
        &mut obj,
        &rho,
        &[(0, AsymptoteSide::Upper), (1, AsymptoteSide::Upper)],
        &tol,
        (&bounds.0, &bounds.1),
    )
    .expect("joint probing must not error");
    let window = joint_window.expect("the joint face law is exactly exponential");
    match assess_coordinate(&window, &tol) {
        AsymptoteVerdict::CertifiedAtAsymptote { tail_constant, .. } => {
            assert!(
                (tail_constant - c).abs() / c < 1.0e-6,
                "joint pencil constant must recover c: got {tail_constant}, want {c}"
            );
        }
        other => panic!("joint face must certify, got {other:?}"),
    }

    // End-to-end: tail-snap declines the marginal law, falls back to the
    // joint face, and snaps BOTH coordinates to their upper rails.
    let g = -0.5 * c * (-7.5_f64).exp();
    let gradient = array![g, g];
    let hessian = array![[g.abs(), 0.0], [0.0, g.abs()]];
    let outcome = try_tail_snap_to_rail(
        &mut obj,
        &AsymptoteRailInputs {
            rho: &rho,
            projected_gradient: &gradient,
            railed: &[],
            layout: OuterThetaLayout::new(2, 0),
            hessian: &hessian,
            bounds: &bounds,
            terminal_beta: None,
            stationarity_bound: StationarityBound::from_ladder(1.0e-3, StationarityBoundSource::SolverBand),
            objective_tol: 1.0e-8,
            context: "joint-face guard test",
            native_coordinate_order: None,
        },
    )
    .expect("tail snap must not error");
    match outcome {
        TailSnapOutcome::ConfirmedNeedsReseed(snapped) => {
            assert_eq!(
                snapped,
                array![12.0, 12.0],
                "both face coordinates must snap to their upper rails"
            );
        }
        other => panic!(
            "the joint face must confirm and publish a reseed waypoint, got {other:?}"
        ),
    }
}

/// #2349 e2e shape: the joint pencil-constant run confirms (measured on
/// the multinomial checkpoint: ĉ settling 62.7 → … → 34.2) while the
/// coefficient steps in the retained deep-interior rows are NOT yet
/// geometrically contracting — the crawl was cut mid-travel. A confirmed
/// law with an unsettled estimand must SNAP the face for re-optimization
/// (the single-coordinate `OnTailNotYetEquivalent` semantics), not
/// decline.
#[test]
fn joint_face_with_unsettled_estimand_snaps_for_reoptimization_2349() {
    let c = 1.2 * (7.5_f64).exp();
    // Non-contracting coefficient hints: constant per-probe steps (β moves
    // linearly in r), so coef_step_ratio has q = 1 and the estimand gate
    // refuses while the pencil constant is exact.
    let problem = OuterProblem::new(2).with_gradient(Derivative::Analytic);
    let mut obj = problem.build_objective(
        (),
        move |_: &mut (), rho: &Array1<f64>| Ok(c * (-(rho[0] + rho[1]) / 2.0).exp()),
        move |_: &mut (), rho: &Array1<f64>| {
            let v = c * (-(rho[0] + rho[1]) / 2.0).exp();
            Ok(OuterEval {
                cost: v,
                gradient: array![-0.5 * v, -0.5 * v],
                hessian: HessianValue::Unavailable,
                inner_beta_hint: Some(array![0.1 * (rho[0] + rho[1])]),
            })
        },
        None::<fn(&mut ())>,
        None::<fn(&mut (), &Array1<f64>) -> Result<EfsEval, EstimationError>>,
    );
    let rho = array![8.0, 7.0];
    let g = -0.5 * c * (-7.5_f64).exp();
    let gradient = array![g, g];
    let hessian = array![[g.abs(), 0.0], [0.0, g.abs()]];
    let bounds = (Array1::from_elem(2, -12.0), Array1::from_elem(2, 12.0));
    let outcome = try_tail_snap_to_rail(
        &mut obj,
        &AsymptoteRailInputs {
            rho: &rho,
            projected_gradient: &gradient,
            railed: &[],
            layout: OuterThetaLayout::new(2, 0),
            hessian: &hessian,
            bounds: &bounds,
            terminal_beta: None,
            stationarity_bound: StationarityBound::from_ladder(1.0e-3, StationarityBoundSource::SolverBand),
            objective_tol: 1.0e-8,
            context: "joint-face unsettled-estimand guard test",
            native_coordinate_order: None,
        },
    )
    .expect("tail snap must not error");
    match outcome {
        TailSnapOutcome::ConfirmedNeedsReseed(snapped) => {
            assert_eq!(
                snapped,
                array![12.0, 12.0],
                "a confirmed joint law with unsettled estimand must snap the face"
            );
        }
        other => panic!("expected a face snap, got {other:?}"),
    }
}

/// A pair of super-bound coordinates that do NOT share a face (independent
/// laws with strongly drifting joint section) must still decline — the
/// joint fallback cannot manufacture a certificate where no joint law
/// holds.
#[test]
fn joint_face_fallback_refuses_a_non_face_2349() {
    // V = c₀·e^{−ρ₀} + drift·ρ₀·e^{−ρ₀} + c₁·e^{−2ρ₁}: coordinate 0's own
    // law is corrupted by the drift term and coordinate 1 decays at a
    // DIFFERENT exponential rate, so neither the marginals nor the joint
    // direction carry a constant pencil.
    let problem = OuterProblem::new(2).with_gradient(Derivative::Analytic);
    let (c0, drift, c1) = (3.0e3, 2.0e3, 5.0e2);
    let mut obj = problem.build_objective(
        (),
        move |_: &mut (), rho: &Array1<f64>| {
            Ok((c0 + drift * rho[0]) * (-rho[0]).exp() + c1 * (-2.0 * rho[1]).exp())
        },
        move |_: &mut (), rho: &Array1<f64>| {
            let e0 = (-rho[0]).exp();
            let e1 = (-2.0 * rho[1]).exp();
            Ok(OuterEval {
                cost: (c0 + drift * rho[0]) * e0 + c1 * e1,
                gradient: array![
                    drift * e0 - (c0 + drift * rho[0]) * e0,
                    -2.0 * c1 * e1
                ],
                hessian: HessianValue::Unavailable,
                inner_beta_hint: Some(array![1.0e-9 * e0]),
            })
        },
        None::<fn(&mut ())>,
        None::<fn(&mut (), &Array1<f64>) -> Result<EfsEval, EstimationError>>,
    );
    let rho = array![8.0, 7.0];
    let g0 = drift * (-8.0_f64).exp() - (c0 + drift * 8.0) * (-8.0_f64).exp();
    let g1 = -2.0 * c1 * (-14.0_f64).exp();
    let gradient = array![g0, g1];
    let hessian = array![[g0.abs(), 0.0], [0.0, g1.abs()]];
    let bounds = (Array1::from_elem(2, -12.0), Array1::from_elem(2, 12.0));
    let outcome = try_tail_snap_to_rail(
        &mut obj,
        &AsymptoteRailInputs {
            rho: &rho,
            projected_gradient: &gradient,
            railed: &[],
            layout: OuterThetaLayout::new(2, 0),
            hessian: &hessian,
            bounds: &bounds,
            terminal_beta: None,
            stationarity_bound: StationarityBound::from_ladder(1.0e-9, StationarityBoundSource::SolverBand),
            objective_tol: 1.0e-8,
            context: "non-face guard test",
            native_coordinate_order: None,
        },
    )
    .expect("tail snap must not error");
    match outcome {
        TailSnapOutcome::Declined(reason) => {
            assert!(
                reason.contains("joint 2-coordinate face"),
                "the decline must carry the joint-face evidence, got: {reason}"
            );
        }
        other => panic!("a non-face must decline, got {other:?}"),
    }
}

/// #2388: the tail-probe ladder must never step the probed coordinate
/// outside its own box interval. Past a box bound the ρ-gradient assembly
/// reports the #197 frozen-axis projection — a literal `0.0` — so an
/// out-of-box probe fabricates a hard-zero tail row (`1.531e0 → 0.000e0` in
/// one e-fold in the #2388 evidence) that the drift band can never confirm,
/// and the fit refuses. The ladder must stop at the last strictly-in-domain
/// probe, and the in-domain rows alone must still confirm an exact tail.
#[test]
fn tail_probe_ladder_never_leaves_the_coordinate_box_2388() {
    let c = 6723.0_f64;
    // Deep enough that the in-domain ladder keeps a healthy-gradient run
    // (probes at |g| below the interior floor are rightly judged unclean),
    // shallow enough that the 18-probe ladder would cross it without the
    // domain clip.
    let box_lower = 12.0_f64;
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
    let tol = AsymptoteTolerances::exp4_rail_bands(1.0e-2);
    let rail = build_and_assess_rail_coordinate(
        &mut obj,
        &rho,
        0,
        AsymptoteSide::Upper,
        &tol,
        (box_lower, 30.0),
        0,
    )
    .expect("probing must not error")
    .expect("the in-domain rows alone must certify the exact tail");
    assert!(
        (rail.tail_constant - c).abs() / c < 1.0e-6,
        "recovered ĉ={} should equal c={c}",
        rail.tail_constant,
    );
    let seen = probed.lock().expect("probe log").clone();
    assert!(
        !seen.is_empty() && seen.iter().all(|&r| r > box_lower),
        "no probe may leave the λ-selection domain (lower bound {box_lower}): {seen:?}",
    );
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

    let mut obj = upper_tail_with_interior(6723.0, 1.0);
    let hessian_psd = array![[1.0, 0.0], [0.0, 2.0]];
    let inputs_psd = AsymptoteRailInputs {
        rho: &rho,
        projected_gradient: &projected,
        railed: &railed,
        layout: OuterThetaLayout::new(2, 0),
        hessian: &hessian_psd,
        bounds: &bounds,
        terminal_beta: None,
        stationarity_bound: StationarityBound::from_ladder(1.0e-6, StationarityBoundSource::SolverBand),
        objective_tol: 1.0e-5,
        context: "asymptote-rail psd test",
        native_coordinate_order: None,
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
        terminal_beta: None,
        stationarity_bound: StationarityBound::from_ladder(
            1.0e-6,
            StationarityBoundSource::SolverBand,
        ),
        objective_tol: 1.0e-5,
        context: "asymptote-rail psi-identity test",
        native_coordinate_order: None,
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

/// #2453: the same identity gate on the snap path. Tail-snap MOVES the
/// coordinate to its bound and republishes that as the search's next
/// point, so applying it to a ψ coordinate does not just mislabel an
/// optimum — it relocates the fit to one.
#[test]
fn tail_snap_refuses_a_psi_coordinate() {
    let c = 6723.0_f64;
    let mut obj = upper_tail_with_interior(c, 1.0);
    let rho = array![8.0_f64, 0.0];
    let tail_gradient = -c * (-rho[0]).exp();
    let gradient = array![tail_gradient, 0.0];
    let hessian = array![[tail_gradient.abs(), 0.0], [0.0, 1.0]];
    let bounds = (Array1::from_elem(2, -12.0), Array1::from_elem(2, 12.0));

    let outcome = try_tail_snap_to_rail(
        &mut obj,
        &AsymptoteRailInputs {
            rho: &rho,
            projected_gradient: &gradient,
            railed: &[],
            layout: OuterThetaLayout::new(2, 2),
            hessian: &hessian,
            bounds: &bounds,
            terminal_beta: None,
            stationarity_bound: StationarityBound::from_ladder(
                1.0e-3,
                StationarityBoundSource::SolverBand,
            ),
            objective_tol: 1.0e-8,
            context: "tail-snap psi-identity test",
            native_coordinate_order: None,
        },
    )
    .expect("tail snap must not error");
    match outcome {
        TailSnapOutcome::Declined(reason) => assert!(
            reason.contains("psi coordinate"),
            "the decline must name the coordinate's identity, got: {reason}"
        ),
        other => panic!("a psi coordinate must never be snapped to its bound, got {other:?}"),
    }
}
