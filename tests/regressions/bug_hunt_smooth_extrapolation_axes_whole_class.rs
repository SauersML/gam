//! Regression guard for #806 from two angles the committed `s(x)` test does not
//! cover: a **tensor** smooth `te(x, z)` (a multi-dimensional, different basis
//! code path) and a **radial Matérn** smooth (whose correct out-of-hull
//! behaviour is *non-monotone* — it reverts to the GP mean, not a linear
//! extension, so a naive flat-clamp and a correct extrapolation differ in a way
//! a slope test alone would miss).
//!
//! Both go through `FittedModel::axis_clip_to_training_ranges`
//! (`src/inference/model.rs`). Before the #806 fix the clip clamped the smooth's
//! input column to the training range, freezing predictions at the boundary
//! fitted value; the fix exempts axes whose basis extrapolates boundedly so the
//! `FittedModel` predict pipeline matches the raw `build_term_collection_design`
//! path.

use gam::test_support::cli_harness::fit_then_predict_gaussian;

/// `te(x, z)` on a tilted plane `y = 0.5 + 1.25·x + 0.75·z`, training both axes
/// over `[0, 2]`. Predicting at `x` beyond the hull (with `z` held in range) must
/// keep moving off the boundary slope along the `x` tensor margin, not freeze.
#[test]
fn tensor_te_smooth_extrapolates_along_a_margin_instead_of_flat_clamping() {
    let dir = tempfile::tempdir().expect("tempdir");
    let train = dir.path().join("train.csv");
    let pred_in = dir.path().join("pred.csv");
    let model = dir.path().join("model.json");
    let out = dir.path().join("out.csv");

    {
        let mut w = csv::Writer::from_path(&train).expect("train csv");
        w.write_record(["x", "z", "y"]).unwrap();
        // A 16x16 grid over [0,2]^2 so the tensor basis is well-identified.
        let n = 16usize;
        for i in 0..n {
            for j in 0..n {
                let x = 2.0 * i as f64 / (n - 1) as f64;
                let z = 2.0 * j as f64 / (n - 1) as f64;
                let y = 0.5 + 1.25 * x + 0.75 * z;
                w.write_record([format!("{x:.10}"), format!("{z:.10}"), format!("{y:.10}")])
                    .unwrap();
            }
        }
        w.flush().unwrap();
    }
    // z held at the in-range mid-point; x swept from in-hull to far outside.
    let probes: [f64; 4] = [1.0, 2.0, 4.0, 6.0];
    {
        let mut w = csv::Writer::from_path(&pred_in).expect("pred csv");
        w.write_record(["x", "z", "y"]).unwrap();
        for &x in &probes {
            w.write_record([format!("{x:.10}"), "1.0".to_string(), "0.0".to_string()])
                .unwrap();
        }
        w.flush().unwrap();
    }

    let preds = fit_then_predict_gaussian(&train, "y ~ te(x, z)", &model, &pred_in, &out);
    let at = |x: f64| preds[probes.iter().position(|&p| (p - x).abs() < 1e-9).unwrap()];

    let in_hull_slope = at(2.0) - at(1.0); // ≈ 1.25 along x
    assert!(
        (in_hull_slope - 1.25).abs() < 0.2,
        "te in-hull slope along x not recovered: {in_hull_slope:.4} (expected ≈ 1.25); \
         extrapolation check would be moot"
    );
    let boundary = at(2.0);
    for &x in probes.iter().filter(|&&x| x > 2.0) {
        let moved = at(x) - boundary;
        let need = 0.5 * in_hull_slope * (x - 2.0);
        assert!(
            moved >= need,
            "te(x, z) did not extrapolate along the x margin: pred(x={x})={:.4} moved only \
             {moved:+.4} off the boundary {boundary:.4} (need ≥ {need:.4}) — looks flat-clamped",
            at(x)
        );
    }
}

/// `matern(x)` on the noise-free line `y = 0.5 + 1.25·x`, `x ∈ [0, 2]`. A Matérn
/// kernel decays with distance, so far outside the hull the fit reverts toward
/// its mean (≈ the grand mean 1.75 here) — it does **not** stay pinned at the
/// boundary fitted value 3.0. The flat-clamp bug produced exactly the boundary
/// value; the correct extrapolation moves away from it while staying bounded.
#[test]
fn radial_matern_reverts_to_mean_instead_of_freezing_at_the_boundary() {
    let dir = tempfile::tempdir().expect("tempdir");
    let train = dir.path().join("train.csv");
    let pred_in = dir.path().join("pred.csv");
    let model = dir.path().join("model.json");
    let out = dir.path().join("out.csv");

    {
        let mut w = csv::Writer::from_path(&train).expect("train csv");
        w.write_record(["x", "y"]).unwrap();
        let n = 200usize;
        for i in 0..n {
            let x = 2.0 * i as f64 / (n - 1) as f64;
            w.write_record([format!("{x:.12}"), format!("{:.12}", 0.5 + 1.25 * x)])
                .unwrap();
        }
        w.flush().unwrap();
    }
    let probes: [f64; 3] = [1.0, 2.0, 8.0];
    {
        let mut w = csv::Writer::from_path(&pred_in).expect("pred csv");
        w.write_record(["x", "y"]).unwrap();
        for &x in &probes {
            w.write_record([format!("{x:.12}"), "0.0".to_string()])
                .unwrap();
        }
        w.flush().unwrap();
    }

    let preds = fit_then_predict_gaussian(&train, "y ~ matern(x)", &model, &pred_in, &out);
    let at = |x: f64| preds[probes.iter().position(|&p| (p - x).abs() < 1e-9).unwrap()];

    let boundary = at(2.0); // ≈ 3.0
    let far = at(8.0);
    // Moved meaningfully off the frozen boundary value (flat-clamp would give 0)…
    assert!(
        (far - boundary).abs() > 0.5,
        "matern(x) froze at the boundary: pred(8)={far:.4} vs boundary pred(2)={boundary:.4} \
         (|diff|={:.4} ≤ 0.5) — looks flat-clamped, not reverting to the kernel mean",
        (far - boundary).abs()
    );
    // …and is bounded (a Matérn fit reverts toward its mean far from the data; it
    // must not blow up). The in-hull data lives in [0.5, 3.0]; the mean is ≈ 1.75.
    assert!(
        far.abs() < 10.0 && (far - 1.75).abs() < 1.0,
        "matern(x) far extrapolation not a bounded revert-to-mean: pred(8)={far:.4} \
         (expected ≈ grand mean 1.75, |·| < 10)"
    );
}
