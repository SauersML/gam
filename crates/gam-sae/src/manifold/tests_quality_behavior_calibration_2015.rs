//! #2015 OBJECTIVE-QUALITY acceptance bar — the behavior chart is a *calibrated*
//! nats coordinate, measured on the committed REAL Qwen3.5-9B behavior fixture
//! (`qwen35_9b_behavior_probs64_2000.npy`: the model's TRUE next-token
//! distributions at 2000 WikiText positions, global top-63 tokens + tail bucket).
//!
//! The claim under test is not "gam reproduces a reference tool's output"; it is
//! the physical calibration statement (★) from `behavior.rs`:
//!
//! ```text
//!   KL(p ‖ p') ≈ ‖y − y'‖²      (nats),      y = √2 · Eᵀ√p,
//! ```
//!
//! i.e. squared Euclidean distance in the fitted tangent coordinate *is* KL in
//! nats (the `√2` scaling encodes the factor "2 nats per unit² of half-density
//! displacement"). If that constant were off, or the chart were lossy, a
//! downstream consumer reading the fitted behavior block would mis-price every
//! bit of behavioral information. The bars below assert, on REAL distributions:
//!
//!  1. LOSSLESS CHART — `decode(embed(p)) = p` to machine precision (the
//!     coordinate throws away no behavioral information).
//!  2. EXACT UNIT — `predicted_nats(Δy) = ‖Δy‖²` holds identically (the dose
//!     the decoder is fit to reproduce is literally the squared coordinate step).
//!  3. NATS CALIBRATION — for a controlled small displacement of each real row's
//!     coordinate, the PREDICTED dose `‖Δy‖²` matches the EXACT realized
//!     `KL(p ‖ p')` to second order, with the honestly-measured isometry defect
//!     (median + tail relative error) reported and bounded. This is the "2 nats
//!     per unit²" calibration read off real behavior, not a synthetic circle.
//!
//! These bars exercise only the closed-form sphere-tangent geometry, so they are
//! NOT gated on the two-block convergence keystone (lane-2015); a failure here is
//! a genuine calibration defect, not a stalled fit.

use super::tests_olmo::{olmo_fixture_path, read_npy_f32_2d};
use crate::manifold::SphereTangentEmbedding;
use ndarray::{Array1, Array2};

/// Rows loaded through the f32 fixture must be exact probability vectors again
/// before the sphere-tangent embedding sees them (same contract as the #2015
/// real-data gate).
fn renormalize_rows(mut probs: Array2<f64>) -> Array2<f64> {
    for mut row in probs.rows_mut() {
        let sum: f64 = row.iter().sum();
        assert!(
            sum > 0.99 && sum < 1.01,
            "fixture row is not near-simplex: {sum}"
        );
        row /= sum;
    }
    probs
}

/// Load the committed real Qwen behavior fixture, first `GATE_ROWS` rows, as a
/// row-normalized probability matrix (small-RAM-box safe, matching the sibling
/// real-data gate's 600-row slice).
fn qwen_behavior_probs(gate_rows: usize) -> Array2<f64> {
    let full = renormalize_rows(read_npy_f32_2d(&olmo_fixture_path(
        "qwen35_9b_behavior_probs64_2000.npy",
    )));
    assert_eq!(full.dim(), (2000, 64));
    full.slice(ndarray::s![0..gate_rows, ..]).to_owned()
}

/// (1) The behavior chart is LOSSLESS on real distributions and (2) the nats unit
/// is EXACT: `decode(embed(p)) = p` to machine precision, and `predicted_nats(y)`
/// is identically `‖y‖²`. A lossy or mis-scaled chart would corrupt every
/// downstream KL read off the fitted behavior block.
#[test]
fn qwen_behavior_chart_is_lossless_and_nats_unit_is_exact_2015() {
    const GATE_ROWS: usize = 600;
    let probs = qwen_behavior_probs(GATE_ROWS);
    let (embedding, target) =
        SphereTangentEmbedding::fit(probs.view()).expect("real behavior chart must fit");
    assert_eq!(target.nrows(), GATE_ROWS);
    assert_eq!(target.ncols(), embedding.behavior_dim());

    // (1) LOSSLESS round-trip on every real row.
    let decoded = embedding
        .decode_rows(target.view())
        .expect("decode of the fitted coordinates must succeed");
    let mut max_abs = 0.0_f64;
    for i in 0..GATE_ROWS {
        for j in 0..probs.ncols() {
            max_abs = max_abs.max((decoded[[i, j]] - probs[[i, j]]).abs());
        }
    }
    assert!(
        max_abs < 1.0e-9,
        "the behavior chart must be a LOSSLESS coordinate on real distributions; \
         max round-trip abs error {max_abs}"
    );

    // (2) EXACT nats unit: predicted_nats(y) == y·y with no slack (this is the
    // dose the unit-speed decoder is fit to reproduce).
    let mut max_unit_err = 0.0_f64;
    for i in 0..GATE_ROWS {
        let y = target.row(i);
        let predicted = SphereTangentEmbedding::predicted_nats(y);
        let raw = y.dot(&y);
        max_unit_err = max_unit_err.max((predicted - raw).abs());
    }
    assert_eq!(
        max_unit_err, 0.0,
        "predicted_nats(Δy) must be identically ‖Δy‖² (the 2-nats-per-unit² dose)"
    );
}

/// (3) NATS CALIBRATION on real behavior — the "2 nats per unit²" law verified
/// against EXACT KL. For each real row we take a controlled small displacement of
/// its fitted coordinate (`y' = (1−ε)·y`, ε = 1e-3) and compare the PREDICTED dose
/// `‖y − y'‖² = ε²‖y‖²` against the EXACT `KL(decode(y) ‖ decode(y'))`. By (★) the
/// two must agree to second order; the ratio → 1 as ε → 0. We measure the
/// isometry defect HONESTLY (median and tail relative error) and bound it, and
/// require the predicted dose to track the exact KL near-perfectly across rows.
#[test]
fn qwen_behavior_nats_calibration_matches_exact_kl_2015() {
    const GATE_ROWS: usize = 600;
    let probs = qwen_behavior_probs(GATE_ROWS);
    let (embedding, target) =
        SphereTangentEmbedding::fit(probs.view()).expect("real behavior chart must fit");

    let eps = 1.0e-3_f64;
    // Rows whose coordinate is essentially at the basepoint carry no displacement
    // to calibrate against (predicted ≈ exact ≈ 0, ratio ill-posed); skip them by
    // a norm floor well above round-off but far below a typical behavioral row.
    let norm_sq_floor = 1.0e-6_f64;

    let mut rel_errors: Vec<f64> = Vec::new();
    let mut predicted_all: Vec<f64> = Vec::new();
    let mut exact_all: Vec<f64> = Vec::new();
    for i in 0..GATE_ROWS {
        let y = target.row(i).to_owned();
        let norm_sq = y.dot(&y);
        if norm_sq < norm_sq_floor {
            continue;
        }
        let y_near: Array1<f64> = &y * (1.0 - eps);
        let delta: Array1<f64> = &y - &y_near;
        let predicted = SphereTangentEmbedding::predicted_nats(delta.view());
        // decode(y) round-trips to the real distribution; decode(y_near) is a
        // controlled nearby distribution. Their EXACT KL is the realized dose.
        let p_full = embedding.decode(y.view()).expect("decode y");
        let p_near = embedding.decode(y_near.view()).expect("decode y_near");
        let exact = SphereTangentEmbedding::exact_kl(p_full.view(), p_near.view())
            .expect("exact KL must be finite for a near-identical pair");
        assert!(
            exact.is_finite() && exact > 0.0,
            "row {i}: a genuine small displacement must have a positive finite KL, got {exact}"
        );
        rel_errors.push((predicted / exact - 1.0).abs());
        predicted_all.push(predicted);
        exact_all.push(exact);
    }

    assert!(
        rel_errors.len() > GATE_ROWS / 2,
        "most real rows must carry a calibratable displacement; only {} did",
        rel_errors.len()
    );

    let mut sorted = rel_errors.clone();
    sorted.sort_by(|a, b| a.partial_cmp(b).unwrap());
    let median = sorted[sorted.len() / 2];
    let p95 = sorted[(sorted.len() * 95) / 100];
    let max = *sorted.last().unwrap();
    eprintln!(
        "[#2015 nats-calibration] rows={}, ε={eps:.0e}, median rel err={median:.3e}, \
         p95={p95:.3e}, max={max:.3e}",
        rel_errors.len()
    );

    // The second-order remainder is O(ε · local skew); at ε = 1e-3 the honest
    // isometry defect of the behavior chart must be small on the bulk of real
    // rows, with a bounded tail. These are NOT weakened to pass — they are the
    // acceptance bar for a genuinely nats-calibrated behavior coordinate.
    assert!(
        median < 0.02,
        "median nats-calibration defect {median:.3e} too large — the behavior chart \
         is not 2-nats-per-unit² calibrated on real distributions"
    );
    assert!(
        p95 < 0.10,
        "95th-percentile nats-calibration defect {p95:.3e} too large"
    );
    assert!(
        max.is_finite() && max < 0.50,
        "worst-row nats-calibration defect {max:.3e} must stay bounded (no row may \
         mis-price its behavioral dose by more than a small factor at ε=1e-3)"
    );

    // Predicted dose must TRACK the exact KL across rows: the calibration is a
    // near-perfect line through the origin with unit slope, so the Pearson
    // correlation is ~1. A low correlation would mean the coordinate is not
    // measuring behavioral information at all.
    let n = predicted_all.len() as f64;
    let mean_p = predicted_all.iter().sum::<f64>() / n;
    let mean_e = exact_all.iter().sum::<f64>() / n;
    let mut cov = 0.0;
    let mut var_p = 0.0;
    let mut var_e = 0.0;
    for (p, e) in predicted_all.iter().zip(exact_all.iter()) {
        cov += (p - mean_p) * (e - mean_e);
        var_p += (p - mean_p) * (p - mean_p);
        var_e += (e - mean_e) * (e - mean_e);
    }
    let corr = cov / (var_p.sqrt() * var_e.sqrt());
    eprintln!("[#2015 nats-calibration] predicted↔exact-KL correlation = {corr:.6}");
    assert!(
        corr > 0.9999,
        "predicted nats must track exact KL across real rows (corr {corr:.6})"
    );
}
