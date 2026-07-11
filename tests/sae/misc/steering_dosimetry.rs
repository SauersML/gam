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

use ndarray::Array2;

use gam::inference::row_metric::{MetricProvenance, RowMetric};
use gam::inference::steering::steer_delta;
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
