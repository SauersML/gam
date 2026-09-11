//! End-to-end check of the SAE-manifold **steering primitive with output
//! dosimetry** (`gam::inference::steering::steer_delta`).
//!
//! # The planted oracle
//!
//! We plant ONE circle atom whose decoder maps the unit circle into a
//! 2-dimensional output as an exact radius-`R` circle:
//!
//! ```text
//! Φ(t) = [1, sin(2π t), cos(2π t)]          (PeriodicHarmonicEvaluator, num_basis = 3)
//! B    = [[0, 0], [0, R], [R, 0]]            (3 × 2 decoder)
//! g(t) = Φ(t) B = R · [cos(2π t), sin(2π t)]
//! ```
//!
//! and a **synthetic quadratic readout** whose output-Fisher information is the
//! identity `F = I₂`. For a quadratic readout the KL between the steered and
//! unsteered output is the chord quadratic form
//! `KL = ½ · ‖g(t_to) − g(t_from)‖²_F` (amplitude 1). With the planted circle
//! and `F = I₂`:
//!
//! ```text
//! ‖g(t_to) − g(t_from)‖² = R² (2 − 2 cos(2π Δ)),   Δ = t_to − t_from
//! ⇒ KL_analytic(Δ) = R² (1 − cos(2π Δ))            ← the closed-form oracle
//! ```
//!
//! The canonical dosimetry is the endpoint quadratic form itself, so the
//! reported `predicted_nats` must match `KL_analytic` for both small and large
//! moves. The separate `validity_radius` still flags where the local
//! initial-tangent approximation stops matching that exact chord dose.
//!
//! Off-manifold component: `δ` is a chord of the decoder circle, so it lies in
//! the local tangent line at `t_from` up to curvature; the reported
//! `off_manifold_norm` is `~0` for a small step and grows with the arc.
//!
//! Fixed construction, no clock, no RNG.

use std::sync::Arc;

use ndarray::{Array1, Array2};

use gam::inference::row_metric::{MetricProvenance, RowMetric};
use gam::inference::steering::{
    AppliedDoseObservation, AppliedDoseProbe, SteerPlan, TargetDoseConfig, TargetDoseError,
    TargetDoseRequest, steer_delta, steer_to_target_nats,
};
use gam::terms::latent::{LatentCoordValues, LatentIdMode, LatentManifold};
use gam::terms::{
    sae::manifold::PeriodicHarmonicEvaluator, sae::manifold::SaeAssignment,
    sae::manifold::SaeAtomBasisKind, sae::manifold::SaeManifoldAtom,
    sae::manifold::SaeManifoldTerm,
};

const R: f64 = 1.3;
const TWO_PI: f64 = 2.0 * std::f64::consts::PI;

/// The closed-form KL of the quadratic readout for a latent move of `delta`
/// (amplitude 1, `F = I₂`): `R² (1 − cos(2π Δ))`.
fn analytic_kl(delta: f64) -> f64 {
    R * R * (1.0 - (TWO_PI * delta).cos())
}

/// Build the single planted circle atom + a fitted term with a single active
/// row at the planted coordinate `t0`, plus the identity output-Fisher metric.
fn planted_circle(t0: f64) -> (SaeManifoldTerm, RowMetric) {
    let p = 2usize; // output dimension
    let m = 3usize; // basis size: [1, sin, cos]
    let d = 1usize; // latent dim
    let n = 1usize; // one stored row (the active row)

    // Stored basis at the planted coordinate via the same evaluator the steering
    // primitive will call at arbitrary t — keeps the stored basis consistent.
    let evaluator = Arc::new(PeriodicHarmonicEvaluator::new(m).expect("evaluator"));
    let coords = Array2::from_shape_vec((n, d), vec![t0]).expect("coords");
    let (phi, jet) = {
        use gam::terms::sae::manifold::SaeBasisEvaluator;
        evaluator.evaluate(coords.view()).expect("evaluate")
    };
    assert_eq!(phi.dim(), (n, m));
    assert_eq!(jet.dim(), (n, m, d));

    // Decoder B (3 × 2): out[0] = R·cos = R·Φ[2], out[1] = R·sin = R·Φ[1].
    let mut decoder = Array2::<f64>::zeros((m, p));
    decoder[[2, 0]] = R; // cos column drives output 0
    decoder[[1, 1]] = R; // sin column drives output 1

    // No roughness penalty (m × m zeros ⇒ operator order 0, no reweighting).
    let smooth = Array2::<f64>::zeros((m, m));

    let atom = SaeManifoldAtom::new_with_provided_function_gram(
        "circle",
        SaeAtomBasisKind::Periodic,
        d,
        phi,
        jet,
        decoder,
        smooth,
    )
    .expect("atom")
    .with_basis_evaluator(evaluator);

    // Single-atom assignment, active on the one row (logit 0, softmax of one
    // atom ⇒ mass 1).
    let logits = Array2::<f64>::zeros((n, 1));
    let coord_values = LatentCoordValues::from_matrix_with_manifold(
        coords.view(),
        LatentIdMode::None,
        LatentManifold::Circle { period: 1.0 },
    );
    let assignment = SaeAssignment::new(logits, vec![coord_values], 1.0).expect("assignment");
    let term = SaeManifoldTerm::new(vec![atom], assignment).expect("term");

    // Identity output-Fisher metric: U_n = I₂ (rank 2). Factors are stored
    // row-major as U_n[i, k] = u[n, i * rank + k]; the identity is u[0] = [1,0,0,1].
    let rank = p;
    let u = Array2::from_shape_vec((n, p * rank), vec![1.0, 0.0, 0.0, 1.0]).expect("u");
    // The planted readout's Fisher IS the identity -- exact and complete, not a
    // truncation -- but `RowMetric` defaults every factored output-Fisher to
    // `UncertifiedApproximation` on purpose ("exactness must be asserted by the
    // producer, never inferred"). The `probe = None` dose path requires an
    // ExactFull kind and otherwise fails closed with
    // `FactorNeedsAppliedDoseProbe`, which is exactly the three failures here:
    // they are the only three target-dose calls that pass no probe.
    let metric = RowMetric::output_fisher(Arc::new(u), p, rank)
        .expect("metric")
        .with_fisher_factor_kind(gam_problem::FisherFactorKind::ExactFull)
        .expect("exact-full certification");

    (term, metric)
}

#[test]
fn predicted_nats_match_analytic_kl_within_validity_radius() {
    let t0 = 0.0;
    let (term, metric) = planted_circle(t0);

    // A SMALL latent step: the chord and the arc agree, so the path-energy dose
    // must match the closed-form KL tightly.
    let delta = 0.01_f64;
    let plan = steer_delta(&term, &metric, 0, 0, 1.0, &[t0], &[t0 + delta]).expect("plan");

    assert_eq!(plan.atom, 0);
    assert_eq!(plan.atom_name, "circle");
    assert_eq!(
        plan.metric_provenance,
        MetricProvenance::OutputFisher { rank: 2 }
    );
    assert!(
        (plan.amplitude - 1.0).abs() < 1e-12,
        "amplitude {}",
        plan.amplitude
    );

    let nats = plan.predicted_nats.expect("behavioral dose available");
    let kl = analytic_kl(delta);
    println!("small step: predicted_nats={nats:.8e} analytic_kl={kl:.8e}");
    let rel = (nats - kl).abs() / kl;
    assert!(
        rel < 1e-3,
        "predicted nats {nats:.8e} must match analytic KL {kl:.8e} within 0.1% for a small step (rel={rel:.3e})"
    );

    // The off-manifold component is ~0: the chord of a tiny arc lies on the
    // tangent line. Scale tolerance to the move size.
    let move_norm = plan.delta.iter().map(|&v| v * v).sum::<f64>().sqrt();
    println!(
        "off_manifold_norm={:.3e} move_norm={move_norm:.3e}",
        plan.off_manifold_norm
    );
    assert!(
        plan.off_manifold_norm < 1e-3 * move_norm,
        "off-manifold residual {:.3e} must be ~0 vs move {move_norm:.3e}",
        plan.off_manifold_norm
    );

    // The validity radius is reported and, for a step this small, covers the
    // whole move (linearization trusted to t_to).
    let vr = plan.validity_radius.expect("validity radius available");
    println!("validity_radius={vr:.5e} full_move={delta:.5e}");
    assert!(
        (vr - delta).abs() < 1e-9,
        "tiny step must be fully within validity radius: vr={vr:.5e}, move={delta:.5e}"
    );
}

#[test]
fn predicted_nats_remain_endpoint_kl_beyond_validity_radius() {
    let t0 = 0.0;
    let (term, metric) = planted_circle(t0);

    // A LARGE latent step (a quarter turn): the arc has curved far from the
    // initial tangent, but the canonical prediction still prices the exact
    // applied endpoint chord. The validity radius must fall strictly inside the
    // requested move.
    let delta = 0.25_f64; // quarter circle
    let plan = steer_delta(&term, &metric, 0, 0, 1.0, &[t0], &[t0 + delta]).expect("plan");

    let nats = plan.predicted_nats.expect("dose");
    let kl = analytic_kl(delta); // R²(1 − cos(π/2)) = R²
    println!("large step: predicted_nats={nats:.6} analytic_kl={kl:.6}");

    // The prediction is exactly the endpoint Fisher quadratic form, hence it
    // agrees with the patched-forward quadratic oracle at any chord length.
    assert!(
        (nats - kl).abs() / kl < 1e-12,
        "endpoint dose {nats:.6} must match the closed-form chord KL {kl:.6}"
    );

    // The validity radius must flag the breakdown: strictly inside the full move.
    let vr = plan.validity_radius.expect("vr");
    println!("validity_radius={vr:.5} full_move={delta:.5}");
    assert!(
        vr < delta - 1e-9,
        "validity radius {vr:.5} must fall strictly inside the over-long move {delta:.5}"
    );
    assert!(vr > 0.0, "validity radius must be positive, got {vr}");
}

#[test]
fn euclidean_metric_yields_geometry_but_no_dose() {
    // A Euclidean (no-behavior) metric: the activation-space delta and the
    // off-manifold guard are still produced, but the behavioral dose and
    // validity radius are *not available* (None), not zero.
    let t0 = 0.1;
    let (term, _fisher) = planted_circle(t0);
    let p = term.output_dim();
    let euclid = RowMetric::euclidean(term.n_obs(), p).expect("euclidean");

    let delta = 0.05;
    let plan = steer_delta(&term, &euclid, 0, 0, 1.0, &[t0], &[t0 + delta]).expect("plan");

    assert_eq!(plan.metric_provenance, MetricProvenance::Euclidean);
    assert!(
        plan.predicted_nats.is_none(),
        "no behavioral axis ⇒ dose unavailable"
    );
    assert!(
        plan.validity_radius.is_none(),
        "no dose ⇒ no validity radius"
    );

    // The geometry is still there: a nonzero on-manifold move and a ~0
    // off-manifold residual.
    let move_norm = plan.delta.iter().map(|&v| v * v).sum::<f64>().sqrt();
    assert!(
        move_norm > 0.0,
        "the activation-space delta must be nonzero"
    );
    assert!(
        plan.off_manifold_norm < 1e-3 * move_norm,
        "off-manifold residual {:.3e} must be ~0 vs move {move_norm:.3e}",
        plan.off_manifold_norm
    );
}

#[test]
fn delta_is_the_activation_space_chord() {
    // The delta must be exactly the planted chord a·(g(t_to) − g(t_from)).
    let t0 = 0.0;
    let (term, metric) = planted_circle(t0);
    let delta = 0.05;
    let plan = steer_delta(&term, &metric, 0, 0, 1.0, &[t0], &[t0 + delta]).expect("plan");

    // g(t) = R·[cos(2πt), sin(2πt)]; chord at amplitude 1.
    let g_from = [R * (TWO_PI * t0).cos(), R * (TWO_PI * t0).sin()];
    let g_to = [
        R * (TWO_PI * (t0 + delta)).cos(),
        R * (TWO_PI * (t0 + delta)).sin(),
    ];
    let expect0 = g_to[0] - g_from[0];
    let expect1 = g_to[1] - g_from[1];
    println!(
        "delta=[{:.6}, {:.6}] expected=[{expect0:.6}, {expect1:.6}]",
        plan.delta[0], plan.delta[1]
    );
    assert!((plan.delta[0] - expect0).abs() < 1e-9);
    assert!((plan.delta[1] - expect1).abs() < 1e-9);

    // Endpoint chord quad-form ½‖δ‖²_F equals the analytic KL exactly here.
    let chord_kl = 0.5 * (plan.delta[0] * plan.delta[0] + plan.delta[1] * plan.delta[1]);
    assert!(
        (chord_kl - analytic_kl(delta)).abs() < 1e-9,
        "chord quad-form {chord_kl:.8} must equal analytic KL {:.8}",
        analytic_kl(delta)
    );
    // Confirm the behavioral axis is live and the metric provenance is Fisher.
    assert!(plan.predicted_nats.is_some());
    assert!(matches!(
        metric.provenance(),
        MetricProvenance::OutputFisher { .. }
    ));
}

/// Build the SAME planted circle oracle as [`planted_circle`], except the
/// installed decoder lives in a Tier-0 STANDARDIZED internal frame:
/// `B_int[:, c] = B_raw[:, c] / σ_c`, exactly the frame
/// [`SaeManifoldTerm::set_tier0_scale`] documents ("the fit runs on
/// `(Z − μ)/σ`... every reconstruction lifts back `x̂ = μ + σ ⊙ x̂_internal`").
/// `scale` is `σ`; the term additionally carries `tier0_scale = Some(σ)` so a
/// consumer that un-scales correctly recovers the exact same raw-frame
/// `g(t) = R·[cos(2πt), sin(2πt)]` oracle as the unscaled fixture.
fn planted_circle_tier0_scaled(t0: f64, scale: [f64; 2]) -> (SaeManifoldTerm, RowMetric) {
    let p = 2usize;
    let m = 3usize;
    let d = 1usize;
    let n = 1usize;

    let evaluator = Arc::new(PeriodicHarmonicEvaluator::new(m).expect("evaluator"));
    let coords = Array2::from_shape_vec((n, d), vec![t0]).expect("coords");
    let (phi, jet) = {
        use gam::terms::sae::manifold::SaeBasisEvaluator;
        evaluator.evaluate(coords.view()).expect("evaluate")
    };
    assert_eq!(phi.dim(), (n, m));
    assert_eq!(jet.dim(), (n, m, d));

    // The RAW-frame decoder is identical to `planted_circle`'s; the INSTALLED
    // decoder is that raw decoder divided column-wise by `scale`, matching the
    // `B_int = B_raw / σ` internal-frame contract exactly.
    let mut decoder = Array2::<f64>::zeros((m, p));
    decoder[[2, 0]] = R / scale[0]; // cos column drives output 0
    decoder[[1, 1]] = R / scale[1]; // sin column drives output 1

    let smooth = Array2::<f64>::zeros((m, m));

    let atom = SaeManifoldAtom::new_with_provided_function_gram(
        "circle",
        SaeAtomBasisKind::Periodic,
        d,
        phi,
        jet,
        decoder,
        smooth,
    )
    .expect("atom")
    .with_basis_evaluator(evaluator);

    let logits = Array2::<f64>::zeros((n, 1));
    let coord_values = LatentCoordValues::from_matrix_with_manifold(
        coords.view(),
        LatentIdMode::None,
        LatentManifold::Circle { period: 1.0 },
    );
    let assignment = SaeAssignment::new(logits, vec![coord_values], 1.0).expect("assignment");
    let mut term = SaeManifoldTerm::new(vec![atom], assignment).expect("term");
    term.set_tier0_scale(Array1::from_vec(scale.to_vec()))
        .expect("tier0 scale");

    // The row output-Fisher metric is ALWAYS built from raw activation-space
    // probes (never from the decoder), so it is IDENTICAL to the unscaled
    // fixture's — an identity metric here, exactly like `planted_circle`.
    let rank = p;
    let u = Array2::from_shape_vec((n, p * rank), vec![1.0, 0.0, 0.0, 1.0]).expect("u");
    // The planted readout's Fisher IS the identity -- exact and complete, not a
    // truncation -- but `RowMetric` defaults every factored output-Fisher to
    // `UncertifiedApproximation` on purpose ("exactness must be asserted by the
    // producer, never inferred"). The `probe = None` dose path requires an
    // ExactFull kind and otherwise fails closed with
    // `FactorNeedsAppliedDoseProbe`, which is exactly the three failures here:
    // they are the only three target-dose calls that pass no probe.
    let metric = RowMetric::output_fisher(Arc::new(u), p, rank)
        .expect("metric")
        .with_fisher_factor_kind(gam_problem::FisherFactorKind::ExactFull)
        .expect("exact-full certification");

    (term, metric)
}

/// #2249 Tier-0-frame regression pin (`ace3b9af3`).
///
/// `steer_delta`/`decode_tangents_at`/`atom_behavior_isometry` used to read
/// the fitted decoder directly, which — under Tier-0 standardization/
/// equilibration (the default for real-activation fits) — lives in the
/// internal frame `B_int[:, c] = B_raw[:, c] / σ_c`, while the row output-
/// Fisher metric `M = UUᵀ` is always built from raw activation-space probes.
/// Every dose `0.5·δᵀMδ` was therefore priced with a per-column `1/σ_c`
/// mis-scale on `δ`, a live calibration confound stacked on top of the
/// rank-truncation/out-of-radius pooling artifact separately diagnosed on
/// #2249.
///
/// This test plants the SAME closed-form circle oracle as
/// [`predicted_nats_match_analytic_kl_within_validity_radius`], but with the
/// decoder installed in a Tier-0-scaled internal frame (`σ = [2.0, 0.5]`,
/// deliberately asymmetric so a missed per-column correction cannot cancel).
/// With the fix engaged, `steer_delta` un-scales the decoded chord/tangents
/// back to raw units before they meet the metric, so `predicted_nats` must
/// recover the EXACT SAME `analytic_kl` oracle the unscaled fixture matches,
/// to the same `rel < 1e-3` tolerance. Reverting the `steering.rs` un-scaling
/// (i.e. dropping the `if let Some(scale) = scale { ... }` correction in
/// `decode_at`/`decode_tangents_at`) makes this test fail deterministically:
/// the un-corrected chord at `t0=0, Δ=0.01` has component `c` off by the fixed
/// factor `1/σ_c`, so its reported `predicted_nats` is a specific, computable
/// wrong number, not merely a looser match to `analytic_kl` — `σ = [2.0, 0.5]`
/// is deliberately asymmetric so the two components' errors cannot cancel
/// each other in the quadratic form. No external data, no GPU: fixed
/// construction, no clock, no RNG, matching the rest of this file.
#[test]
fn predicted_nats_survive_tier0_frame_rescale_2249() {
    let t0 = 0.0;
    let scale = [2.0_f64, 0.5_f64];
    let (term, metric) = planted_circle_tier0_scaled(t0, scale);

    assert_eq!(
        term.tier0_scale().map(|s| s.to_vec()),
        Some(scale.to_vec()),
        "tier0 scale must round-trip through the setter"
    );

    // Same small step as the unscaled fixture's within-radius test.
    let delta = 0.01_f64;
    let plan = steer_delta(&term, &metric, 0, 0, 1.0, &[t0], &[t0 + delta]).expect("plan");

    // The decoded chord must land back in RAW units: identical to the
    // unscaled fixture's `delta_is_the_activation_space_chord` expectation,
    // regardless of the internal Tier-0 scale.
    let g_from = [R * (TWO_PI * t0).cos(), R * (TWO_PI * t0).sin()];
    let g_to = [
        R * (TWO_PI * (t0 + delta)).cos(),
        R * (TWO_PI * (t0 + delta)).sin(),
    ];
    let expect0 = g_to[0] - g_from[0];
    let expect1 = g_to[1] - g_from[1];
    println!(
        "tier0-scaled delta=[{:.6}, {:.6}] expected(raw)=[{expect0:.6}, {expect1:.6}]",
        plan.delta[0], plan.delta[1]
    );
    assert!(
        (plan.delta[0] - expect0).abs() < 1e-9,
        "un-scaled chord[0] {:.8} must match raw-frame expectation {expect0:.8}",
        plan.delta[0]
    );
    assert!(
        (plan.delta[1] - expect1).abs() < 1e-9,
        "un-scaled chord[1] {:.8} must match raw-frame expectation {expect1:.8}",
        plan.delta[1]
    );

    let nats = plan.predicted_nats.expect("behavioral dose available");
    let kl = analytic_kl(delta);
    println!("tier0-scaled small step: predicted_nats={nats:.8e} analytic_kl={kl:.8e}");
    let rel = (nats - kl).abs() / kl;
    assert!(
        rel < 1e-3,
        "predicted nats {nats:.8e} must match analytic KL {kl:.8e} within 0.1% under Tier-0 \
         rescaling (rel={rel:.3e}) — a failure here means the raw-frame un-scaling \
         (#2249, ace3b9af3) is not engaged or not correct"
    );

    // The validity radius must also be reported in RAW units (a σ-mis-scaled
    // tangent would report a systematically wrong radius even when the
    // endpoint dose above happened to be checked at a different δ).
    let vr = plan.validity_radius.expect("validity radius available");
    println!("tier0-scaled validity_radius={vr:.5e} full_move={delta:.5e}");
    assert!(
        (vr - delta).abs() < 1e-9,
        "tiny step must be fully within validity radius under Tier-0 rescaling: \
         vr={vr:.5e}, move={delta:.5e}"
    );
}

/// The dose rounding bar on the planted circle: an exact-factor solve resolves the
/// displacement to its representation limit, so its dose can differ from the
/// target only by a few ulps of the endpoint quadratic's operand scale `R²`.
fn dose_rounding_bar() -> f64 {
    8.0 * f64::EPSILON * R * R
}

/// The analytic displacement along `+1` whose planted-circle dose is `q*`:
/// `R²(1 − cos 2πs) = q*` on the rising half of the circle.
fn analytic_displacement(target: f64) -> f64 {
    (1.0 - target / (R * R)).acos() / TWO_PI
}

/// The displacement bar implied by [`dose_rounding_bar`] through the analytic
/// slope `2πR² sin 2πs*`, plus the representation of `s*` itself.
fn displacement_rounding_bar(expected_s: f64) -> f64 {
    dose_rounding_bar() / (TWO_PI * R * R * (TWO_PI * expected_s).sin())
        + 4.0 * f64::EPSILON * expected_s
}

/// gh#2263 target-dose API with no model in the loop (`probe = None`): the solve
/// moves the row's coordinate along its chart until the exact resident dose equals
/// the target, and returns the landing coordinate. Along `+1` the planted circle's
/// dose at the row's own gate (1 here) is `R²(1 − cos 2πs)`, so the displacement has
/// the closed form `s* = acos(1 − q*/R²)/2π`, and the move must be bit for bit the
/// chord `steer_delta` writes to `t0 + s*`.
#[test]
fn target_dose_exact_factor_lands_the_dose_on_the_chart() {
    let t0 = 0.0;
    let (term, metric) = planted_circle(t0);
    let target = 0.37 * analytic_kl(0.02);

    let plan = steer_to_target_nats(
        &term,
        &metric,
        TargetDoseRequest {
            atom_k: 0,
            metric_row: 0,
            t_from: &[t0],
            direction: &[1.0],
            target_nats: target,
            config: TargetDoseConfig::default(),
        },
        None,
    )
    .expect("target-dose plan");

    let landed = plan.steer.predicted_nats.expect("predicted dose");
    let expected_s = analytic_displacement(target);
    let expected_seed = (2.0 * target).sqrt() / (TWO_PI * R);
    println!(
        "exact-factor target {target:.6e}: landed {landed:.6e}, displacement {:.12e} vs \
         analytic {expected_s:.12e}, seed {:.12e} vs {expected_seed:.12e}",
        plan.displacement, plan.seed_displacement
    );
    assert!(
        (landed - target).abs() <= dose_rounding_bar(),
        "the exact-factor solve must land the dose to its rounding bar: {landed} vs {target} \
         (bar {:.3e})",
        dose_rounding_bar()
    );
    assert!(
        (plan.displacement - expected_s).abs() <= displacement_rounding_bar(expected_s),
        "displacement {} must be the analytic crossing {expected_s} (bar {:.3e})",
        plan.displacement,
        displacement_rounding_bar(expected_s)
    );
    assert!(
        (plan.seed_displacement - expected_seed).abs() <= 8.0 * f64::EPSILON * expected_seed,
        "seed displacement {} must be sqrt(2 q*)/(2πR) = {expected_seed}",
        plan.seed_displacement
    );
    assert_eq!(
        plan.steer.amplitude, 1.0,
        "the move is written at the row's own gate"
    );
    assert_eq!(
        plan.steer.t_to,
        vec![t0 + plan.displacement],
        "t_to is the solved landing coordinate"
    );
    let chord = steer_delta(&term, &metric, 0, 0, 1.0, &[t0], &plan.steer.t_to).expect("chord");
    assert_eq!(
        plan.steer.delta, chord.delta,
        "the move must be the on-chart chord to the solved t_to"
    );
    assert!(plan.applied_probe.is_none());
    assert_eq!(plan.iterations, 0);
    assert!(plan.readout_kl_radius.is_none());
    assert!(
        plan.steer.validity_radius.is_some(),
        "chart radius must be reported"
    );
}

/// gh#2263 target-dose API — with an EXACT-quadratic patched forward the closed
/// loop lands the requested dose to `1e-9` and stamps the readout-KL radius, since
/// every probe matches its directional dose. The first-order seed is not exact on
/// a circle (its dose falls short by about `q*/(6R²)`), so the correction is a real
/// secant solve over the displacement rather than a confirmation of the seed.
#[test]
fn target_dose_exact_probe_lands_to_a_tight_tolerance() {
    let t0 = 0.0;
    let (term, metric) = planted_circle(t0);
    let target = 0.5 * analytic_kl(0.02);
    let tol_rel = 1.0e-9;

    // Exact-quadratic probe: measured KL == the endpoint Fisher quadratic (the
    // planted readout IS a quadratic, so the second-order dose is the true KL).
    let mut probe = |plan: &SteerPlan| -> Result<AppliedDoseObservation, String> {
        let exact = plan.predicted_nats.expect("dose");
        Ok(AppliedDoseObservation {
            effective_delta: plan.delta.clone(),
            exact_directional_nats: exact,
            measured_nats: exact,
            certified_attainable_upper_nats: None,
        })
    };
    let plan = steer_to_target_nats(
        &term,
        &metric,
        TargetDoseRequest {
            atom_k: 0,
            metric_row: 0,
            t_from: &[t0],
            direction: &[1.0],
            target_nats: target,
            config: TargetDoseConfig {
                tol_rel,
                ..TargetDoseConfig::default()
            },
        },
        Some(&mut probe as &mut AppliedDoseProbe<'_>),
    )
    .expect("target-dose plan");

    let observation = plan.applied_probe.as_ref().expect("applied probe");
    let measured = observation.measured_nats;
    println!(
        "exact probe: {} probes, measured {measured:.12e} vs target {target:.12e}, \
         displacement {:.6e} vs seed {:.6e}",
        plan.iterations, plan.displacement, plan.seed_displacement
    );
    assert_eq!(observation.effective_delta, plan.steer.delta);
    assert!(
        (measured - target).abs() / target <= tol_rel,
        "measured KL {measured} must equal target {target} to {tol_rel:e}"
    );
    assert!(
        plan.iterations <= TargetDoseConfig::default().max_iter,
        "an exact probe must land within the probe budget; took {}",
        plan.iterations
    );
    assert_eq!(
        plan.steer.amplitude, 1.0,
        "the move is written at the row's own gate"
    );
    // Every probe matched its directional dose, so the readout-KL radius covers the
    // solved displacement.
    let rr = plan.readout_kl_radius.expect("readout radius established");
    assert!(
        rr >= plan.displacement,
        "readout radius {rr} must cover the solved displacement {}",
        plan.displacement
    );
}

/// gh#2263 target-dose API — with a SATURATING patched forward (true KL bounded
/// while the quadratic grows) the first-order seed under-delivers KL, and the
/// secant loop moves the coordinate FURTHER onto the measured curve. The seed sits
/// past the readout-KL radius (measured departs from the quadratic there), so the
/// radius is reported as unestablished — exactly the diagnostic #2249 asks for.
#[test]
fn target_dose_saturating_probe_secant_corrects_upward() {
    let t0 = 0.0;
    let (term, metric) = planted_circle(t0);
    let target = 0.5_f64; // nats, below the saturation ceiling K = 1.0 and the peak 2R²

    // Saturating monotone KL: measured = K·(1 − exp(−quad/K)), ≈ quad for small
    // quad, bounded by K.
    let k_sat = 1.0_f64;
    let mut probe = |plan: &SteerPlan| -> Result<AppliedDoseObservation, String> {
        let exact = plan.predicted_nats.expect("exact local dose");
        Ok(AppliedDoseObservation {
            effective_delta: plan.delta.clone(),
            exact_directional_nats: exact,
            measured_nats: k_sat * (1.0 - (-exact / k_sat).exp()),
            certified_attainable_upper_nats: Some(k_sat),
        })
    };
    let plan = steer_to_target_nats(
        &term,
        &metric,
        TargetDoseRequest {
            atom_k: 0,
            metric_row: 0,
            t_from: &[t0],
            direction: &[1.0],
            target_nats: target,
            config: TargetDoseConfig::default(),
        },
        Some(&mut probe as &mut AppliedDoseProbe<'_>),
    )
    .expect("target-dose plan");

    let measured = plan
        .applied_probe
        .as_ref()
        .expect("applied probe")
        .measured_nats;
    println!(
        "saturating probe: {} probes, measured {measured:.6e}, displacement {:.6e} vs seed {:.6e}",
        plan.iterations, plan.displacement, plan.seed_displacement
    );
    assert!(
        (measured - target).abs() / target <= TargetDoseConfig::default().tol_rel,
        "corrected measured KL {measured} must reach target {target}"
    );
    // Saturation ⇒ the quadratic over-predicts, so the realized move is longer than
    // the first-order seed (had to move further to realize the same measured KL).
    assert!(
        plan.displacement > plan.seed_displacement,
        "saturating readout needs a LONGER move than the quadratic seed: {} vs {}",
        plan.displacement,
        plan.seed_displacement
    );
    assert_eq!(
        plan.steer.amplitude, 1.0,
        "the correction moves the coordinate, never the amplitude"
    );
    assert!(
        plan.iterations >= 2,
        "correction must take at least one secant step"
    );
    // At the seed the exact dose (≈ 0.476) already departs from the saturating
    // measured (≈ 0.379, 20% > 10% readout tol), so no probed displacement was in
    // readout tolerance ⇒ the readout-KL radius is honestly unestablished here.
    assert!(
        plan.readout_kl_radius.is_none(),
        "seed is past the readout-KL radius; radius must be reported unestablished"
    );
}

#[test]
fn target_dose_plateau_is_an_explicit_unreachable_error() {
    let t0 = 0.0;
    let (term, metric) = planted_circle(t0);
    let target = 0.5_f64;
    let mut probe = |plan: &SteerPlan| -> Result<AppliedDoseObservation, String> {
        Ok(AppliedDoseObservation {
            effective_delta: plan.delta.clone(),
            exact_directional_nats: plan.predicted_nats.expect("exact local dose"),
            measured_nats: 0.1,
            certified_attainable_upper_nats: Some(0.1),
        })
    };
    let error = steer_to_target_nats(
        &term,
        &metric,
        TargetDoseRequest {
            atom_k: 0,
            metric_row: 0,
            t_from: &[t0],
            direction: &[1.0],
            target_nats: target,
            config: TargetDoseConfig::default(),
        },
        Some(&mut probe),
    )
    .expect_err("a certified global envelope below the target is unreachable");
    let TargetDoseError::UnreachableTarget {
        certified_attainable_upper_nats,
        ..
    } = error
    else {
        panic!("expected a certified unreachable-target error");
    };
    assert_eq!(certified_attainable_upper_nats, 0.1);
}

#[test]
fn target_dose_apparent_plateau_without_certificate_is_only_unbracketed() {
    let t0 = 0.0;
    let (term, metric) = planted_circle(t0);
    let target = 0.5_f64;
    let mut probe = |plan: &SteerPlan| -> Result<AppliedDoseObservation, String> {
        Ok(AppliedDoseObservation {
            effective_delta: plan.delta.clone(),
            exact_directional_nats: plan.predicted_nats.expect("exact local dose"),
            measured_nats: 0.1,
            certified_attainable_upper_nats: None,
        })
    };
    let error = steer_to_target_nats(
        &term,
        &metric,
        TargetDoseRequest {
            atom_k: 0,
            metric_row: 0,
            t_from: &[t0],
            direction: &[1.0],
            target_nats: target,
            config: TargetDoseConfig {
                max_iter: 3,
                ..TargetDoseConfig::default()
            },
        },
        Some(&mut probe),
    )
    .expect_err("three equal samples cannot certify a global attainable envelope");
    assert!(matches!(
        error,
        TargetDoseError::UnbracketedTarget {
            probes: 3,
            max_measured_nats: 0.1,
            ..
        }
    ));
}

/// gh#2263: the solve never moves past the chart. On the planted circle the extent
/// along `+1` is one period; with an uncertified plateau and probes to spare the
/// expansion reaches exactly that end, observes it once, and refuses with the
/// typed `ChartExtentExhausted` rather than clamping or wrapping into a second turn.
#[test]
fn target_dose_chart_end_is_a_typed_refusal_never_a_clamp() {
    let t0 = 0.0;
    let (term, metric) = planted_circle(t0);
    let target = 0.5_f64;
    let mut probed: Vec<f64> = Vec::new();
    let mut probe = |plan: &SteerPlan| -> Result<AppliedDoseObservation, String> {
        probed.push(plan.t_to[0]);
        Ok(AppliedDoseObservation {
            effective_delta: plan.delta.clone(),
            exact_directional_nats: plan.predicted_nats.expect("exact local dose"),
            measured_nats: 0.1,
            certified_attainable_upper_nats: None,
        })
    };
    let error = steer_to_target_nats(
        &term,
        &metric,
        TargetDoseRequest {
            atom_k: 0,
            metric_row: 0,
            t_from: &[t0],
            direction: &[1.0],
            target_nats: target,
            config: TargetDoseConfig::default(),
        },
        Some(&mut probe),
    )
    .expect_err("a plateau below the target along the whole circle must be refused");
    println!("chart end: probed t_to {probed:?}; {error}");
    let TargetDoseError::ChartExtentExhausted {
        extent,
        max_observed_nats,
        ..
    } = error
    else {
        panic!("expected ChartExtentExhausted");
    };
    assert_eq!(extent, 1.0, "the circle's extent along +1 is one period");
    assert_eq!(max_observed_nats, 0.1);
    assert!(
        probed.len() < TargetDoseConfig::default().max_iter,
        "the refusal must come from the chart's end, not from the probe budget: {} probes",
        probed.len()
    );
    assert_eq!(
        probed.last().copied(),
        Some(0.0),
        "the last observation must be the chart's end, one full turn from t0"
    );
}

#[test]
fn target_dose_expansion_continues_through_a_local_decrease() {
    let t0 = 0.0;
    let (term, metric) = planted_circle(t0);
    let target = 0.5_f64;
    // First-order seed on the planted circle: q(s) ≈ ½ (2πR)² s².
    let seed = (target / (0.5 * (TWO_PI * R).powi(2))).sqrt();
    let mut probe = |plan: &SteerPlan| -> Result<AppliedDoseObservation, String> {
        let displacement = plan.t_to[0] - t0;
        let measured_nats = if displacement < 1.5 * seed {
            0.20
        } else if displacement < 3.0 * seed {
            0.19
        } else {
            target
        };
        Ok(AppliedDoseObservation {
            effective_delta: plan.delta.clone(),
            exact_directional_nats: plan.predicted_nats.expect("exact local dose"),
            measured_nats,
            certified_attainable_upper_nats: None,
        })
    };
    let plan = steer_to_target_nats(
        &term,
        &metric,
        TargetDoseRequest {
            atom_k: 0,
            metric_row: 0,
            t_from: &[t0],
            direction: &[1.0],
            target_nats: target,
            config: TargetDoseConfig {
                tol_rel: 0.0,
                max_iter: 3,
                ..TargetDoseConfig::default()
            },
        },
        Some(&mut probe),
    )
    .expect("a local decrease cannot terminate ordered bracket expansion");
    assert_eq!(plan.iterations, 3);
    assert_eq!(
        plan.applied_probe.expect("final exact probe").measured_nats,
        target
    );
}

#[test]
fn target_dose_probe_exhaustion_never_returns_an_unconverged_plan() {
    let t0 = 0.0;
    let (term, metric) = planted_circle(t0);
    let target = 0.5_f64;
    let mut probe = |plan: &SteerPlan| -> Result<AppliedDoseObservation, String> {
        let exact = plan.predicted_nats.expect("exact local dose");
        Ok(AppliedDoseObservation {
            effective_delta: plan.delta.clone(),
            exact_directional_nats: exact,
            measured_nats: 0.5 * exact,
            certified_attainable_upper_nats: None,
        })
    };
    let error = steer_to_target_nats(
        &term,
        &metric,
        TargetDoseRequest {
            atom_k: 0,
            metric_row: 0,
            t_from: &[t0],
            direction: &[1.0],
            target_nats: target,
            config: TargetDoseConfig {
                max_iter: 1,
                ..TargetDoseConfig::default()
            },
        },
        Some(&mut probe),
    )
    .expect_err("one below-target probe cannot certify a target-dose plan");
    assert!(matches!(
        error,
        TargetDoseError::UnbracketedTarget { probes: 1, .. }
    ));
}

/// gh#2263: a relative tolerance is a fraction, not an arbitrary non-negative
/// scalar. Values at or above one can certify zero realized KL for a positive
/// target, while NaN/infinity make every comparison meaningless. Zero remains
/// a valid exact-agreement request.
#[test]
fn target_dose_relative_tolerances_have_fractional_finite_domain() {
    let t0 = 0.0;
    let (term, metric) = planted_circle(t0);
    let target = analytic_kl(0.02);

    steer_to_target_nats(
        &term,
        &metric,
        TargetDoseRequest {
            atom_k: 0,
            metric_row: 0,
            t_from: &[t0],
            direction: &[1.0],
            target_nats: target,
            config: TargetDoseConfig {
                tol_rel: 0.0,
                readout_tol_rel: 0.0,
                ..TargetDoseConfig::default()
            },
        },
        None,
    )
    .expect("zero requests exact agreement and is a valid relative tolerance");

    for invalid in [1.0, f64::INFINITY, f64::NAN] {
        for config in [
            TargetDoseConfig {
                tol_rel: invalid,
                ..TargetDoseConfig::default()
            },
            TargetDoseConfig {
                readout_tol_rel: invalid,
                ..TargetDoseConfig::default()
            },
        ] {
            let error = steer_to_target_nats(
                &term,
                &metric,
                TargetDoseRequest {
                    atom_k: 0,
                    metric_row: 0,
                    t_from: &[t0],
                    direction: &[1.0],
                    target_nats: target,
                    config,
                },
                None,
            )
            .expect_err("a non-fractional/non-finite tolerance must fail closed");
            assert!(matches!(error, TargetDoseError::InvalidRequest(_)));
        }
    }
}

/// gh#2263: the request names a direction to move along, not a target coordinate.
/// A zero, non-finite, or wrong-length direction has nothing to move along and must
/// fail closed before any move is written, and the direction's length carries no
/// meaning: only where it points is used.
#[test]
fn target_dose_direction_must_be_a_finite_nonzero_chart_vector() {
    let t0 = 0.0;
    let (term, metric) = planted_circle(t0);
    let target = analytic_kl(0.02);
    let request = |direction: &'static [f64]| TargetDoseRequest {
        atom_k: 0,
        metric_row: 0,
        t_from: &[0.0],
        direction,
        target_nats: target,
        config: TargetDoseConfig::default(),
    };
    for direction in [&[0.0][..], &[f64::NAN][..], &[1.0, 0.0][..]] {
        let error = steer_to_target_nats(&term, &metric, request(direction), None)
            .expect_err("an unusable direction must fail closed");
        assert!(
            matches!(error, TargetDoseError::InvalidRequest(_)),
            "direction {direction:?}: {error}"
        );
    }
    let unit = steer_to_target_nats(&term, &metric, request(&[1.0]), None).expect("unit direction");
    let scaled =
        steer_to_target_nats(&term, &metric, request(&[7.5]), None).expect("scaled direction");
    assert_eq!(
        unit.steer.delta, scaled.steer.delta,
        "the solve must use only the direction, not its length"
    );
    assert_eq!(unit.steer.t_to, scaled.steer.t_to);
    assert_eq!(unit.steer.t_from, vec![t0]);
}

#[test]
fn applied_dose_probe_payload_fails_closed() {
    let t0 = 0.0;
    let (term, metric) = planted_circle(t0);
    let target = analytic_kl(0.02);
    let cases = [
        AppliedDoseObservation {
            effective_delta: Array1::zeros(1),
            exact_directional_nats: target,
            measured_nats: target,
            certified_attainable_upper_nats: None,
        },
        AppliedDoseObservation {
            effective_delta: Array1::from(vec![f64::NAN, 0.0]),
            exact_directional_nats: target,
            measured_nats: target,
            certified_attainable_upper_nats: None,
        },
        AppliedDoseObservation {
            effective_delta: Array1::zeros(2),
            exact_directional_nats: f64::NAN,
            measured_nats: target,
            certified_attainable_upper_nats: None,
        },
        AppliedDoseObservation {
            effective_delta: Array1::zeros(2),
            exact_directional_nats: target,
            measured_nats: -1.0,
            certified_attainable_upper_nats: None,
        },
        AppliedDoseObservation {
            effective_delta: Array1::zeros(2),
            exact_directional_nats: target,
            measured_nats: target,
            certified_attainable_upper_nats: Some(f64::NAN),
        },
        AppliedDoseObservation {
            effective_delta: Array1::zeros(2),
            exact_directional_nats: target,
            measured_nats: target,
            certified_attainable_upper_nats: Some(0.5 * target),
        },
    ];
    for invalid in cases {
        let mut probe =
            |_: &SteerPlan| -> Result<AppliedDoseObservation, String> { Ok(invalid.clone()) };
        let error = steer_to_target_nats(
            &term,
            &metric,
            TargetDoseRequest {
                atom_k: 0,
                metric_row: 0,
                t_from: &[t0],
                direction: &[1.0],
                target_nats: target,
                config: TargetDoseConfig::default(),
            },
            Some(&mut probe),
        )
        .expect_err("invalid applied-dose observations must fail closed");
        assert!(matches!(error, TargetDoseError::Probe(_)));
    }
}

/// gh#2263 items 1+3 — the contract change itself. The retired API scaled the
/// chord `a · (g(t_to) − g(t_from))` to reach a dose and returned `t_to` unchanged,
/// so a large dose wrote the row off the circle: a `+1` month request at amplitude
/// 16 re-encoded past `+3` months and saturated. A target dose now moves the
/// coordinate along the chart at the row's own gate, `t_to` is solved, and a dose
/// the chart cannot produce is refused.
///
/// * **Positive control**: `4×` the dose of a `+1/12` chord is on the circle. It is
///   landed to the dose rounding bar at the analytic displacement
///   `acos(1 − q*/R²)/2π` (about `+2.08` twelfths), with the row's own amplitude
///   and the exact chord to the solved `t_to`.
/// * **The saturating case**: `256×` that dose, `+1` month written at amplitude 16,
///   exceeds the circle's largest dose `2R²`. It is refused with
///   `ChartExtentExhausted` after observing that peak, instead of returning a plan
///   clamped onto the circle or an off-circle chord. An extreme finite dose is
///   refused the same way rather than becoming an unbounded move.
#[test]
fn target_dose_moves_the_coordinate_and_refuses_what_the_chart_cannot_produce_2263() {
    let t0 = 0.0;
    let (term, metric) = planted_circle(t0);
    let month = 1.0 / 12.0;
    let unit_month = analytic_kl(month);
    let peak = 2.0 * R * R;
    let request = |target_nats: f64| TargetDoseRequest {
        atom_k: 0,
        metric_row: 0,
        t_from: &[0.0],
        direction: &[1.0],
        target_nats,
        config: TargetDoseConfig::default(),
    };

    let reachable = 4.0 * unit_month;
    let plan = steer_to_target_nats(&term, &metric, request(reachable), None)
        .expect("an on-chart dose must be landed");
    let expected_s = analytic_displacement(reachable);
    let landed = plan.steer.predicted_nats.expect("dose");
    let saturating = 256.0 * unit_month;
    let saturating_error = steer_to_target_nats(&term, &metric, request(saturating), None)
        .expect_err("a dose above the circle's largest dose must be refused");
    let extreme_error = steer_to_target_nats(&term, &metric, request(f64::MAX), None)
        .expect_err("an extreme finite dose must be refused");
    println!(
        "[#2263 contract] 4x +1-month dose {reachable:.6e}: landed {landed:.6e} at {:.6} \
         twelfths (analytic {:.6}); 256x dose {saturating:.6e} vs peak {peak:.6e}: \
         {saturating_error}; f64::MAX: {extreme_error}",
        12.0 * plan.displacement,
        12.0 * expected_s
    );

    assert!(
        (landed - reachable).abs() <= dose_rounding_bar(),
        "positive control: the on-chart dose must be landed to its rounding bar: {landed} vs \
         {reachable}"
    );
    assert!(
        (plan.displacement - expected_s).abs() <= displacement_rounding_bar(expected_s),
        "positive control: displacement {} must be the analytic crossing {expected_s}",
        plan.displacement
    );
    assert!(
        plan.displacement > month,
        "a larger dose must move the coordinate further than +1 month: {} twelfths",
        12.0 * plan.displacement
    );
    assert_eq!(
        plan.steer.amplitude, 1.0,
        "the dose must be realized at the row's own gate, never by scaling the chord"
    );
    let chord = steer_delta(&term, &metric, 0, 0, 1.0, &[t0], &plan.steer.t_to).expect("chord");
    assert_eq!(
        plan.steer.delta, chord.delta,
        "the move must be the on-chart chord to the solved t_to"
    );

    assert!(
        saturating > peak,
        "precondition: the saturating request must exceed the circle's largest dose"
    );
    for (name, error) in [
        ("256x +1-month dose", saturating_error),
        ("f64::MAX", extreme_error),
    ] {
        let TargetDoseError::ChartExtentExhausted {
            extent,
            max_observed_nats,
            ..
        } = error
        else {
            panic!("{name}: expected ChartExtentExhausted");
        };
        assert_eq!(
            extent, 1.0,
            "{name}: the circle's extent along +1 is one period"
        );
        assert!(
            (max_observed_nats - peak).abs() <= dose_rounding_bar(),
            "{name}: the refusal must have observed the circle's peak dose {peak}; observed \
             {max_observed_nats}"
        );
    }
}
