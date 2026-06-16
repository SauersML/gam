//! Sanity check: BC clamped/anchored fits must approximate the data well
//! in the interior. The variants tests verify they don't crash, but a
//! degenerate fit that returns ~constant predictions everywhere would also
//! "succeed". We check that RMSE against truth in the interior is within
//! a small multiple of the unconstrained fit's RMSE.

use csv::StringRecord;
use gam::matrix::LinearOperator;
use gam::smooth::build_term_collection_design;
use gam::{
    FitConfig, FitResult, encode_recordswith_inferred_schema, fit_from_formula, init_parallelism,
};
use ndarray::Array2;
use rand::SeedableRng;
use rand::rngs::StdRng;
use rand_distr::{Distribution, Normal, Uniform};

fn truth(x: f64) -> f64 {
    // A smooth interior function that does NOT vanish at endpoints — so a
    // wrong anchored fit pinning to 0 at the boundary would show non-zero
    // bias near the interior.
    (2.0 * std::f64::consts::PI * x).sin() + 0.3
}

fn make_data(n: usize, sigma: f64, seed: u64) -> gam::data::EncodedDataset {
    let mut rng = StdRng::seed_from_u64(seed);
    let u = Uniform::new(0.0, 1.0).expect("uniform");
    let noise = Normal::new(0.0, sigma).expect("normal");
    let mut x: Vec<f64> = (0..n).map(|_| u.sample(&mut rng)).collect();
    x.sort_by(|a, b| a.partial_cmp(b).unwrap());
    let y: Vec<f64> = x
        .iter()
        .map(|t| truth(*t) + noise.sample(&mut rng))
        .collect();
    let headers = ["x", "y"].into_iter().map(String::from).collect();
    let rows: Vec<StringRecord> = x
        .iter()
        .zip(y.iter())
        .map(|(a, b)| StringRecord::from(vec![a.to_string(), b.to_string()]))
        .collect();
    encode_recordswith_inferred_schema(headers, rows).expect("encode")
}

fn rmse_against_truth(formula: &str, eval_xs: &[f64]) -> f64 {
    let data = make_data(300, 0.05, 7);
    let cfg = FitConfig {
        family: Some("gaussian".to_string()),
        ..FitConfig::default()
    };
    let result = fit_from_formula(formula, &data, &cfg)
        .unwrap_or_else(|e| panic!("fit failed for `{formula}`: {e}"));
    let FitResult::Standard(fit) = result else {
        panic!("expected standard fit for `{formula}`")
    };
    let n = eval_xs.len();
    let mut m = Array2::<f64>::zeros((n, 2));
    for (i, &x) in eval_xs.iter().enumerate() {
        m[[i, 0]] = x;
        m[[i, 1]] = 0.0;
    }
    let design = build_term_collection_design(m.view(), &fit.resolvedspec)
        .unwrap_or_else(|e| panic!("predict design failed for `{formula}`: {e:?}"));
    let pred = design.design.apply(&fit.fit.beta);
    let mut sumsq = 0.0_f64;
    for (i, &x) in eval_xs.iter().enumerate() {
        let d = pred[i] - truth(x);
        sumsq += d * d;
    }
    (sumsq / n as f64).sqrt()
}

#[test]
fn bc_clamped_fit_tracks_interior_well() {
    init_parallelism();
    // Interior probe grid (avoid the boundary).
    let eval_xs: Vec<f64> = (0..101).map(|i| 0.10 + 0.80 * (i as f64) / 100.0).collect();
    let rmse_free = rmse_against_truth("y ~ s(x, k=12)", &eval_xs);
    let rmse_clamped_both = rmse_against_truth("y ~ s(x, k=12, bc=clamped)", &eval_xs);
    let rmse_clamped_left = rmse_against_truth("y ~ s(x, k=12, bc_left=clamped)", &eval_xs);
    let rmse_clamped_right = rmse_against_truth("y ~ s(x, k=12, bc_right=clamped)", &eval_xs);
    eprintln!(
        "[bc-quality] free={rmse_free:.4} clamped_both={rmse_clamped_both:.4} \
         clamped_left={rmse_clamped_left:.4} clamped_right={rmse_clamped_right:.4}",
    );
    // Free has σ=0.05 → expected RMSE ≲ 0.07. Clamped variants should be
    // within 5× of free in the interior — any worse is a sign that the BC
    // is corrupting the fit far from the boundary.
    let budget = (5.0 * rmse_free).max(0.10);
    for (label, r) in [
        ("clamped_both", rmse_clamped_both),
        ("clamped_left", rmse_clamped_left),
        ("clamped_right", rmse_clamped_right),
    ] {
        assert!(
            r <= budget,
            "BC {label} interior RMSE {r:.4} > budget {budget:.4} (free {rmse_free:.4})",
        );
    }
}

#[test]
fn bc_anchored_zero_fit_tracks_interior_with_known_bias() {
    init_parallelism();
    // Anchored=0 forces f(boundary) = 0. Truth has f(0)=0.3 and f(1)=0.3,
    // so the anchored fit MUST have visible bias near the boundary but
    // should recover well in the interior away from the pinned ends.
    let interior_xs: Vec<f64> = (0..51).map(|i| 0.20 + 0.60 * (i as f64) / 50.0).collect();
    let rmse_free = rmse_against_truth("y ~ s(x, k=12)", &interior_xs);
    let rmse_anchored_both = rmse_against_truth("y ~ s(x, k=12, bc=anchored)", &interior_xs);
    eprintln!("[bc-anchored-quality] free={rmse_free:.4} anchored_both={rmse_anchored_both:.4}",);
    // Even with 0.2 of fixed bias at the boundary, away from the boundary
    // the spline should recover the true curve. Budget = 8× free (anchored
    // costs basis flexibility at both ends but interior should still fit).
    let budget = (8.0 * rmse_free).max(0.15);
    assert!(
        rmse_anchored_both <= budget,
        "BC anchored_both interior RMSE {rmse_anchored_both:.4} > budget {budget:.4} (free {rmse_free:.4})",
    );
}

#[test]
fn bc_anchored_zero_pins_smooth_to_constant_intercept_at_basis_boundaries() {
    init_parallelism();
    // bc=anchored pins the SMOOTH to zero at the *basis* knot boundaries
    // (knots[degree] and knots[knots.len()-degree-1]), which equal the
    // min/max of the training data — NOT the user-passed [0, 1] interval.
    // To verify the pin we probe at the actual training data extremes,
    // not at 0/1 which may be slightly outside basis support.
    //
    // The smooth being zero at both knot boundaries plus a shared intercept
    // means total predictions at those two probe points must agree to high
    // precision. If they differ by more than O(1e-6), the BC constraint
    // is not actually being enforced — the original failing version of
    // this test probed at x=0/x=1 (extrapolation breaks the pin), giving
    // a misleading 0.013 diff that looked like a bug but was just
    // extrapolation outside the BC range.
    let data = make_data(300, 0.05, 7);
    // Find training data extremes.
    let values = data.values.column(0);
    let x_min = values.iter().cloned().fold(f64::INFINITY, f64::min);
    let x_max = values.iter().cloned().fold(f64::NEG_INFINITY, f64::max);

    let cfg = FitConfig {
        family: Some("gaussian".to_string()),
        ..FitConfig::default()
    };
    let result =
        fit_from_formula("y ~ s(x, k=12, bc=anchored)", &data, &cfg).expect("anchored fit ok");
    let FitResult::Standard(fit) = result else {
        panic!("expected standard fit")
    };
    let mut m = Array2::<f64>::zeros((2, 2));
    m[[0, 0]] = x_min;
    m[[0, 1]] = 0.0;
    m[[1, 0]] = x_max;
    m[[1, 1]] = 0.0;
    let design =
        build_term_collection_design(m.view(), &fit.resolvedspec).expect("predict design ok");
    let pred = design.design.apply(&fit.fit.beta);
    eprintln!(
        "[bc-anchored-pin] f(x_min={x_min:.6})={:.6} f(x_max={x_max:.6})={:.6}",
        pred[0], pred[1]
    );
    let diff = (pred[0] - pred[1]).abs();
    assert!(
        diff < 1e-6,
        "BC anchored both must yield f(x_min)==f(x_max) at basis boundaries (smooth pinned to zero plus shared intercept): {:.6} vs {:.6} diff={:.3e}",
        pred[0],
        pred[1],
        diff,
    );
}
