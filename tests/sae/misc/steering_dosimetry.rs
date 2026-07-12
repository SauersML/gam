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
    TargetDoseConfig, TargetDoseError, TargetDoseRequest, steer_delta, steer_to_target_nats,
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
    let metric = RowMetric::output_fisher(Arc::new(u), p, rank).expect("metric");

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
    let metric = RowMetric::output_fisher(Arc::new(u), p, rank).expect("metric");

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

/// gh#2263 target-dose API — the closed-form seed `a0 = sqrt(2 q*/(dgᵀ M dg))`
/// hits the requested dose EXACTLY on the planted quadratic readout, with no
/// model in the loop (`probe = None`). The planted circle metric is `F = I₂` at
/// `rank = p`, so `predicted_nats(a) = ½‖a·dg‖²` is exactly quadratic in `a`;
/// the seed identity `½ a0² dgᵀM dg = q*` therefore holds to machine precision at
/// every dose (and a-fortiori at infinitesimal dose). This is the pure math +
/// plumbing surface; `measured_nats=None` makes its quadratic-only validation
/// explicit while the returned `steer` carries the exact delta to apply.
#[test]
fn target_dose_closed_form_seed_hits_dose_exactly() {
    let t0 = 0.0;
    let (term, metric) = planted_circle(t0);
    let delta = 0.02_f64;

    // Unit-amplitude dose sets the scale; ask for a fraction of it.
    let unit = steer_delta(&term, &metric, 0, 0, 1.0, &[t0], &[t0 + delta]).expect("unit");
    let unit_nats = unit.predicted_nats.expect("unit dose");
    let target = 0.37 * unit_nats;

    let plan = steer_to_target_nats(
        &term,
        &metric,
        TargetDoseRequest {
            atom_k: 0,
            metric_row: 0,
            t_from: &[t0],
            t_to: &[t0 + delta],
            target_nats: target,
            config: TargetDoseConfig::default(),
        },
        None,
    )
    .expect("target-dose plan");

    // Closed-form identity: ½ a0² dgᵀM dg == q* exactly.
    assert!(
        (plan.steer.predicted_nats.expect("predicted dose") - target).abs() / target < 1e-12,
        "closed-form seed dose {} must equal target {target} (rel={:.3e})",
        plan.steer.predicted_nats.expect("predicted dose"),
        (plan.steer.predicted_nats.expect("predicted dose") - target).abs() / target
    );
    assert!(
        (plan.steer.amplitude - plan.seed_amplitude).abs() < 1e-15,
        "with no probe the amplitude is the closed-form seed"
    );
    // The target-dose result is itself the exact applied move; no second steer
    // call is necessary to recover its activation delta.
    let applied_nats = plan.steer.predicted_nats.expect("applied dose");
    assert!(
        (applied_nats - target).abs() / target < 1e-12,
        "the atomic applied plan must land the dose: {applied_nats} vs {target}"
    );
    assert!(plan.measured_nats.is_none());
    assert_eq!(plan.iterations, 0);
    assert!(plan.readout_kl_radius.is_none());
    assert!(
        plan.steer.validity_radius.is_some(),
        "chart radius must be reported"
    );
}

/// gh#2263 target-dose API — with an EXACT-quadratic patched forward the seed is
/// already right, so the closed-loop correction confirms it in a single probe and
/// stamps the readout-KL radius (the quadratic matches the measured KL there).
#[test]
fn target_dose_exact_probe_converges_in_one_step() {
    let t0 = 0.0;
    let (term, metric) = planted_circle(t0);
    let delta = 0.02_f64;
    let unit = steer_delta(&term, &metric, 0, 0, 1.0, &[t0], &[t0 + delta]).expect("unit");
    let unit_nats = unit.predicted_nats.expect("unit dose");
    let target = 0.5 * unit_nats;

    // Exact-quadratic probe: measured KL == the endpoint Fisher quadratic (the
    // planted readout IS a quadratic, so the second-order dose is the true KL).
    let mut probe = |a: f64| -> Result<f64, String> {
        let p = steer_delta(&term, &metric, 0, 0, a, &[t0], &[t0 + delta])?;
        Ok(p.predicted_nats.expect("dose"))
    };
    let plan = steer_to_target_nats(
        &term,
        &metric,
        TargetDoseRequest {
            atom_k: 0,
            metric_row: 0,
            t_from: &[t0],
            t_to: &[t0 + delta],
            target_nats: target,
            config: TargetDoseConfig::default(),
        },
        Some(&mut probe as &mut dyn FnMut(f64) -> Result<f64, String>),
    )
    .expect("target-dose plan");

    assert_eq!(
        plan.iterations, 1,
        "the seed already hits the target exactly"
    );
    let measured = plan.measured_nats.expect("measured");
    assert!(
        (measured - target).abs() / target < 1e-9,
        "measured KL {measured} must equal target {target}"
    );
    // The probe matched the quadratic at the seed ⇒ that amplitude is inside the
    // readout-KL radius.
    let rr = plan.readout_kl_radius.expect("readout radius established");
    assert!(
        (rr - plan.seed_amplitude).abs() < 1e-9,
        "readout radius {rr} must be the (in-tolerance) seed amplitude {}",
        plan.seed_amplitude
    );
}

/// gh#2263 target-dose API — with a SATURATING patched forward (true KL bounded
/// while the quadratic grows) the closed-form seed under-delivers KL, and the
/// secant loop corrects UPWARD onto the measured curve. The seed sits past the
/// readout-KL radius (measured departs from the quadratic there), so the radius is
/// reported as unestablished — exactly the diagnostic #2249 asks for.
#[test]
fn target_dose_saturating_probe_secant_corrects_upward() {
    let t0 = 0.0;
    let (term, metric) = planted_circle(t0);
    let delta = 0.05_f64;
    let unit = steer_delta(&term, &metric, 0, 0, 1.0, &[t0], &[t0 + delta]).expect("unit");
    let unit_nats = unit.predicted_nats.expect("unit dose");
    let target = 0.5_f64; // nats, below the saturation ceiling K = 1.0

    // Saturating monotone KL: measured = K·(1 − exp(−quad/K)), ≈ quad for small
    // quad, bounded by K. quad(a) = a²·unit_nats.
    let k_sat = 1.0_f64;
    let mut probe = |a: f64| -> Result<f64, String> {
        let quad = a * a * unit_nats;
        Ok(k_sat * (1.0 - (-quad / k_sat).exp()))
    };
    let plan = steer_to_target_nats(
        &term,
        &metric,
        TargetDoseRequest {
            atom_k: 0,
            metric_row: 0,
            t_from: &[t0],
            t_to: &[t0 + delta],
            target_nats: target,
            config: TargetDoseConfig::default(),
        },
        Some(&mut probe as &mut dyn FnMut(f64) -> Result<f64, String>),
    )
    .expect("target-dose plan");

    let measured = plan.measured_nats.expect("measured");
    assert!(
        (measured - target).abs() / target <= TargetDoseConfig::default().tol_rel,
        "corrected measured KL {measured} must reach target {target}"
    );
    // Saturation ⇒ the quadratic over-predicts, so the true amplitude exceeds the
    // closed-form seed (had to push harder to realize the same measured KL).
    assert!(
        plan.steer.amplitude > plan.seed_amplitude,
        "saturating readout needs MORE amplitude than the quadratic seed: {} vs {}",
        plan.steer.amplitude,
        plan.seed_amplitude
    );
    assert!(
        plan.iterations >= 2,
        "correction must take at least one secant step"
    );
    // The seed dose (quad = target = 0.5) already departs from the saturating
    // measured (0.5 vs 0.39, 22% > 10% readout tol), so no probed amplitude was in
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
    let delta = 0.05_f64;
    let target = 0.5_f64;
    let mut probe = |_amplitude: f64| -> Result<f64, String> { Ok(0.1) };
    let error = steer_to_target_nats(
        &term,
        &metric,
        TargetDoseRequest {
            atom_k: 0,
            metric_row: 0,
            t_from: &[t0],
            t_to: &[t0 + delta],
            target_nats: target,
            config: TargetDoseConfig::default(),
        },
        Some(&mut probe),
    )
    .expect_err("a measured plateau below the target is unreachable");
    assert!(matches!(error, TargetDoseError::UnreachableTarget { .. }));
}

#[test]
fn target_dose_probe_exhaustion_never_returns_an_unconverged_plan() {
    let t0 = 0.0;
    let (term, metric) = planted_circle(t0);
    let delta = 0.05_f64;
    let unit = steer_delta(&term, &metric, 0, 0, 1.0, &[t0], &[t0 + delta]).expect("unit");
    let unit_nats = unit.predicted_nats.expect("unit dose");
    let target = 0.5_f64;
    let mut probe = |amplitude: f64| -> Result<f64, String> {
        Ok(0.5 * amplitude * amplitude * unit_nats)
    };
    let error = steer_to_target_nats(
        &term,
        &metric,
        TargetDoseRequest {
            atom_k: 0,
            metric_row: 0,
            t_from: &[t0],
            t_to: &[t0 + delta],
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
        TargetDoseError::ProbeBudgetExhausted { probes: 1, .. }
    ));
}
