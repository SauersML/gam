// Child module of `run_plan::run_plan_tests` (see the `#[path]` declaration
// there): #2425 — the certificate face is the whole theta box, not the lambda
// block — and #2596 — one stationarity standard for screening and mint. Scope
// comes from the parent via `use super::*`; the split is purely physical.

use super::*;
use ndarray::array;

// ─── #2425 — the certificate face must be the whole θ box, not the λ block ───
//
// A joint spatial-κ search optimizes θ = [ρ …, ψ …] over ONE box, and
// `project_gradient_vector` already projects every coordinate against its own
// bound. `certificate_railed_lambdas` scans only `0..layout.rho_dim()`, and
// `rho_dim()` is `n_params − psi_dim` — so a ψ coordinate pinned on its
// data-derived κ window was invisible to every certificate decision that takes
// an active face: the off-railed PSD test, the interior curvature floor, the
// asymptote-rail mint, tail-snap, the #2155 saddle escape, and the #2392
// active-set reduction.
//
// The fixture is the shape of the `gam-models` Matérn refusals: two interior ρ
// coordinates that are exactly stationary, and one ψ coordinate driven onto the
// upper box bound whose curvature there is saturated NEGATIVE. The full Hessian
// is diag(1, 1, −1) and indefinite; the feasible tangent subspace at the active
// bound is {ρ₀, ρ₁}, where it is diag(1, 1) and positive definite. Second-order
// admissibility at a box-constrained optimum is a statement about that tangent
// subspace, so the certificate must judge the reduced block — which it could
// not, because the ψ coordinate never reached `certificate_railed`.
fn psi_rail_cost(theta: &Array1<f64>) -> f64 {
    // ½ρ₀² + ½ρ₁² − 0.3ψ − ½ψ²: the ψ arm is concave and pushed outward, so the
    // search rails it at +rho_bound and the curvature there is −1.
    0.5 * theta[0] * theta[0] + 0.5 * theta[1] * theta[1] - 0.3 * theta[2]
        - 0.5 * theta[2] * theta[2]
}

fn psi_rail_eval(theta: &Array1<f64>) -> OuterEval {
    OuterEval {
        cost: psi_rail_cost(theta),
        gradient: array![theta[0], theta[1], -0.3 - theta[2]],
        hessian: HessianValue::Dense(array![
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, 0.0, -1.0],
        ]),
        inner_beta_hint: None,
    }
}

#[test]
fn railed_psi_coordinate_is_on_the_certificate_face_not_only_the_lambda_report_2425() {
    // The two sets differ exactly as designed: the report stays λ-scoped, the
    // face covers the whole θ box. `rho_dim = n_params − psi_dim = 2`, so index
    // 2 is a ψ coordinate and can only appear in the θ-wide scan.
    let config = OuterConfig::default();
    let theta = array![0.0, 0.0, config.rho_bound];
    assert_eq!(
        certificate_railed_lambdas(&theta, 2, &config),
        Vec::<usize>::new(),
        "the λ-block report must stay λ-scoped: index 2 is a ψ coordinate"
    );
    assert_eq!(
        certificate_railed_coordinates(&theta, &config),
        vec![2],
        "the θ-wide face must see the ψ coordinate pinned on the box"
    );

    // And the face is load-bearing: the same Hessian reads indefinite against
    // the λ-only report and positive definite against the true active face.
    let hessian = array![[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, -1.0]];
    assert_eq!(
        certificate_hessian_is_psd_off_railed(&hessian, &[], None),
        Some(false),
        "control: judged against an empty face the saturated ψ row makes the \
         full Hessian indefinite — this is what the λ-only scan produced"
    );
    assert_eq!(
        certificate_hessian_is_psd_off_railed(&hessian, &[2], None),
        Some(true),
        "judged on the feasible tangent subspace at the active ψ bound, the \
         curvature is admissible"
    );
}

#[test]
fn joint_rho_psi_optimum_certifies_when_only_the_psi_coordinate_rails_2425() {
    // End-to-end: a point that is genuinely a constrained minimum — interior
    // gradient exactly zero, ψ on its bound with the outward pull the projector
    // discards — must certify. Before the face was widened this refused with
    // `hessian_psd=NO` and `railed=[]`, which is the `gam-models` Matérn
    // signature and #979's "the projector removed 99.2% of |g| but nothing is
    // listed as railed".
    let config = OuterConfig::default();
    let problem = OuterProblem::new(3)
        .with_psi_dim(1)
        .with_gradient(Derivative::Analytic)
        .with_hessian(DeclaredHessianForm::Dense)
        .with_initial_rho(array![0.0, 0.0, config.rho_bound])
        .with_screen_initial_rho(false)
        .with_seed_config(gam_problem::SeedConfig {
            max_seeds: 1,
            seed_budget: 1,
            ..Default::default()
        });
    let mut obj = problem.build_objective(
        (),
        |_: &mut (), theta: &Array1<f64>| Ok(psi_rail_cost(theta)),
        |_: &mut (), theta: &Array1<f64>| Ok(psi_rail_eval(theta)),
        None::<fn(&mut ())>,
        None::<fn(&mut (), &Array1<f64>) -> Result<EfsEval, EstimationError>>,
    );
    let result = audit_stationary_point(
        &mut obj,
        array![0.0, 0.0, config.rho_bound],
        "railed-psi certificate face #2425",
    )
    .unwrap_or_else(|rejection| {
        panic!(
            "a constrained minimum whose only active bound is a psi coordinate must \
             certify; refused with: {}",
            rejection.source
        )
    });
    let cert = result
        .criterion_certificate
        .as_ref()
        .expect("a certified point records its certificate");
    assert!(
        cert.certifies(),
        "certificate must accept the constrained minimum: {}",
        cert.summary()
    );
    assert_eq!(
        cert.curvature_verdict(),
        crate::model_types::result_types::CurvatureAdmissibility::Admissible,
        "curvature is MEASURED admissible on the feasible tangent subspace, not merely          unevaluated (#2578): {}",
        cert.summary()
    );
    assert_eq!(
        cert.lambdas_railed,
        Vec::<usize>::new(),
        "no SMOOTHING coordinate railed here, so the λ report stays empty: {:?}",
        cert.lambdas_railed
    );
}

// ─── #2624 — the certificate's EVIDENCE must cover the same face ──────────
//
// `certificate_railed_coordinates` widened every certificate DECISION to the
// whole theta box (#2425 above). The `railed_facts` evidence field — the
// `#N theta=… box=[…] margin=…` detail a refusal prints — was still built from
// the lambda-only index set, so on an exact-joint spatial route the one
// coordinate whose rail status decides the fit was the one coordinate the
// refusal could not print: `railed=[]` with no detail is indistinguishable from
// nothing being railed at all. On #2624's arms psi carries the ENTIRE outer
// gradient, and that thread nearly attributed the resulting `railed=[]` to a
// gradient defect.
//
// `lambdas_railed` is the stable lambda-scoped REPORT and is asserted unchanged
// here: this widens the evidence, it does not relabel the report.
#[test]
fn railed_psi_coordinate_is_in_the_certificates_evidence_not_only_on_its_face_2624() {
    let config = OuterConfig::default();
    let problem = OuterProblem::new(3)
        .with_psi_dim(1)
        .with_gradient(Derivative::Analytic)
        .with_hessian(DeclaredHessianForm::Dense)
        .with_initial_rho(array![0.0, 0.0, config.rho_bound])
        .with_screen_initial_rho(false)
        .with_seed_config(gam_problem::SeedConfig {
            max_seeds: 1,
            seed_budget: 1,
            ..Default::default()
        });
    let mut obj = problem.build_objective(
        (),
        |_: &mut (), theta: &Array1<f64>| Ok(psi_rail_cost(theta)),
        |_: &mut (), theta: &Array1<f64>| Ok(psi_rail_eval(theta)),
        None::<fn(&mut ())>,
        None::<fn(&mut (), &Array1<f64>) -> Result<EfsEval, EstimationError>>,
    );
    let result = audit_stationary_point(
        &mut obj,
        array![0.0, 0.0, config.rho_bound],
        "railed-psi certificate evidence #2624",
    )
    .unwrap_or_else(|rejection| {
        panic!(
            "fixture precondition: this constrained minimum must certify before \
             its evidence field can be read; refused with: {}",
            rejection.source
        )
    });
    let cert = result
        .criterion_certificate
        .as_ref()
        .expect("a certified point records its certificate");

    // The report is unchanged and stays lambda-scoped.
    assert_eq!(
        cert.lambdas_railed,
        Vec::<usize>::new(),
        "no SMOOTHING coordinate rails here, so the lambda report stays empty: {:?}",
        cert.lambdas_railed
    );
    // Precondition: index 2 really is railed on the face the decisions use. If
    // this fails the fixture stopped exercising the thing under test, and the
    // assertion below would pass or fail for an unrelated reason.
    assert_eq!(
        certificate_railed_coordinates(&result.rho, &config),
        vec![2],
        "fixture precondition: the psi coordinate must be on the theta-wide face"
    );

    let psi_fact = cert
        .railed_facts
        .iter()
        .find(|fact| fact.index == 2)
        .unwrap_or_else(|| {
            panic!(
                "the psi coordinate is the ONLY active bound at this optimum, so \
                 the certificate's evidence must carry it; railed_facts={:?} \
                 summary={}",
                cert.railed_facts,
                cert.summary()
            )
        });
    assert!(
        (psi_fact.theta - config.rho_bound).abs() < 1e-9,
        "the evidence must report the psi coordinate at the value it was judged \
         at: theta={} bound={}",
        psi_fact.theta,
        config.rho_bound
    );
    assert!(
        psi_fact.lower < psi_fact.upper,
        "a fact is evidence only if it carries the interval it was judged \
         against, got [{}, {}]",
        psi_fact.lower,
        psi_fact.upper
    );
    assert!(
        cert.summary().contains("#2 theta="),
        "the rendered summary must SHOW the psi coordinate's interval — a reader \
         who sees only `railed=[]` is told the opposite of what was decided: {}",
        cert.summary()
    );
}

// ─── #2596: one stationarity standard for screening and mint ──────────────

/// Build a one-coordinate flat-valley objective `c + ½·curvature·(ρ − ρ*)²`
/// with an analytic Hessian, and certify the point `ρ* + offset` at `fidelity`.
///
/// `curvature` sets the Newton decrement at a given gradient: a LARGE curvature
/// makes `½gᵀH⁻¹g` tiny, which is precisely the regime the
/// curvature-resolvability rung exists to certify — a residual gradient that
/// buys no resolvable objective decrease.
fn certify_flat_valley_point_2596(
    curvature: f64,
    offset: f64,
    fidelity: CertificationFidelity,
    order_log: Arc<Mutex<Vec<OuterEvalOrder>>>,
) -> Result<OuterCriterionCertificate, EstimationError> {
    let problem = OuterProblem::new(1)
        .with_gradient(Derivative::Analytic)
        .with_hessian(DeclaredHessianForm::Dense);
    let mut obj = problem.build_objective_with_eval_order(
        (),
        move |_: &mut (), rho: &Array1<f64>| Ok(1.0 + 0.5 * curvature * rho[0] * rho[0]),
        |_: &mut (), _: &Array1<f64>| {
            Err(EstimationError::InvalidInput(
                "this objective is driven through the derivative-order seam".to_string(),
            ))
        },
        move |_: &mut (), rho: &Array1<f64>, order: OuterEvalOrder| {
            order_log.lock().expect("order log").push(order);
            Ok(OuterEval {
                cost: 1.0 + 0.5 * curvature * rho[0] * rho[0],
                gradient: array![curvature * rho[0]],
                hessian: match order {
                    OuterEvalOrder::ValueGradientHessian => {
                        HessianValue::Dense(array![[curvature]])
                    }
                    OuterEvalOrder::Value | OuterEvalOrder::ValueAndGradient => {
                        HessianValue::Unavailable
                    }
                },
                inner_beta_hint: None,
            })
        },
        None::<fn(&mut ())>,
        None::<fn(&mut (), &Array1<f64>) -> Result<EfsEval, EstimationError>>,
    );
    let mut result = OuterResult::new(
        array![offset],
        1.0 + 0.5 * curvature * offset * offset,
        1,
        true,
        OuterPlan {
            solver: Solver::Bfgs,
            hessian_source: HessianSource::BfgsApprox,
        },
    );
    certify_outer_optimality_with_fidelity(
        &mut obj,
        &OuterConfig::default(),
        "screening-vs-mint-2596",
        &mut result,
        fidelity,
    )
}

/// #2596 — the screening pass must never refuse what the mint would certify.
///
/// The stationarity bound is a LADDER, and its curvature-resolvability rung
/// (`‖Pg‖·√(objective_tol / ½gᵀH⁻¹g)`) is owned by the analytic outer Hessian.
/// `Screening` reserves the order-four ladder for the mint (#2359) — a
/// reservation argued on the CURVATURE CONJUNCT, where a missing verdict is
/// permissive. But the same Hessian owns a rung of the BOUND, where a missing
/// verdict is not permissive at all: it reverts the pass to the un-widened
/// solver band. Screening therefore applied a strictly TIGHTER standard than
/// the audit that mints, and the multi-start DISCARDS a screening refusal —
/// on #2596 discarding the correct interior optimum and publishing the ρ-box
/// corner (which certifies with `|Pg| ≡ 0` by construction) at 26× the cost.
///
/// The invariant, swept across four decades of residual gradient: whatever the
/// mint certifies, screening certifies. Any future re-tiering of the ladder by
/// fidelity fails here.
#[test]
fn screening_never_refuses_what_mint_certifies_2596() {
    // Large curvature ⇒ a tiny Newton decrement at every gradient below, so
    // each of these points is stationary at the resolution the criterion can be
    // optimized even where its raw |Pg| exceeds the solver band.
    let curvature = 1.0e4;
    for &offset in &[0.0, 1.0e-12, 1.0e-10, 1.0e-9, 1.0e-8, 1.0e-7] {
        let mint = certify_flat_valley_point_2596(
            curvature,
            offset,
            CertificationFidelity::Mint,
            Arc::new(Mutex::new(Vec::new())),
        );
        let screening = certify_flat_valley_point_2596(
            curvature,
            offset,
            CertificationFidelity::Screening,
            Arc::new(Mutex::new(Vec::new())),
        );
        let mint_certifies = mint.as_ref().is_ok_and(OuterCriterionCertificate::certifies);
        let screening_certifies = screening
            .as_ref()
            .is_ok_and(OuterCriterionCertificate::certifies);
        assert!(
            !mint_certifies || screening_certifies,
            "offset={offset:e}: the mint certified but screening refused — the cheap \
             per-candidate filter is applying a tighter standard than the audit that \
             mints, so the multi-start discards optima the engine itself calls \
             stationary (#2596). mint={:?} screening={:?}",
            mint.as_ref().map(OuterCriterionCertificate::summary),
            screening.as_ref().map(|c| c.summary()),
        );
    }
}

/// #2596 companion — the point the fixture actually died on.
///
/// A residual `|Pg|` a small multiple above the solver band, on curvature that
/// makes the Newton decrement orders below the objective tolerance, is
/// stationary: no model-resolvable step lowers the criterion. Screening must
/// say so. Before the fix this refused, and the refusal is what left the
/// vacuously-certified ρ-box corner as the only survivor.
#[test]
fn screening_certifies_a_flat_valley_gradient_above_the_solver_band_2596() {
    let curvature = 1.0e4;
    let offset = 1.0e-8;
    let orders = Arc::new(Mutex::new(Vec::new()));
    let certificate = certify_flat_valley_point_2596(
        curvature,
        offset,
        CertificationFidelity::Screening,
        Arc::clone(&orders),
    )
    .unwrap_or_else(|error| {
        panic!("screening must certify a curvature-resolvable flat-valley point: {error}")
    });
    assert!(
        certificate.certifies(),
        "screening certificate must accept the point: {}",
        certificate.summary()
    );
    let orders = orders.lock().expect("order log");
    assert!(
        orders.contains(&OuterEvalOrder::ValueGradientHessian),
        "the escalation is what supplies the curvature-resolvability rung, so a \
         would-refuse screening pass must have spent order four: {orders:?}"
    );
}

/// #2596 — and it must stay free for every healthy fit.
///
/// The escalation is gated on the un-widened bound ALREADY refusing. A point
/// whose projected gradient clears the solver band must reach the same verdict
/// on first-order evidence alone, spending no order-four evaluation — which is
/// what keeps #2359's "order four exactly once, at the mint" true in practice.
#[test]
fn screening_at_a_clean_optimum_spends_no_order_four_2596() {
    let orders = Arc::new(Mutex::new(Vec::new()));
    let certificate = certify_flat_valley_point_2596(
        1.0e4,
        0.0,
        CertificationFidelity::Screening,
        Arc::clone(&orders),
    )
    .expect("an exactly-stationary point certifies on first-order evidence");
    assert!(certificate.certifies(), "{}", certificate.summary());
    let orders = orders.lock().expect("order log");
    assert!(
        !orders.contains(&OuterEvalOrder::ValueGradientHessian),
        "a candidate that clears its first-order band must pay nothing extra: {orders:?}"
    );
}
