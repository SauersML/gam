//! Regression: the fitted cyclic smooth must lie inside the data envelope.
//!
//! Data lives on a tilted 3D unit circle with Gaussian noise σ = 0.07. The
//! generating ground truth has range ~1.0 in each coordinate, so any fit that
//! stays sane should produce predictions within a generous 5σ ≈ 0.35 envelope
//! around the truth. We assert that condition.
//!
//! Several smooth families are run side-by-side. As of this test's introduction
//! `duchon(ct, st, centers=80)` returns predictions that wander outside the
//! envelope (REML for Duchon's three operator penalties — mass / tension /
//! stiffness — fails to converge: indefinite LAML Hessian, ARC stalls at 200
//! iters with `|g| ≈ 9.7`, and the fit silently uses the degenerate ρ from the
//! pseudo-inverse fallback). `thinplate(ct, st)` and `matern(ct, st)` on the
//! same data stay within tolerance.

use csv::StringRecord;
use gam::matrix::LinearOperator;
use gam::{
    FitConfig, FitResult, encode_recordswith_inferred_schema, fit_from_formula,
    init_parallelism,
};
use ndarray::Array2;
use rand::SeedableRng;
use rand::rngs::StdRng;
use rand_distr::{Distribution, Normal, Uniform};

const TAU: f64 = std::f64::consts::TAU;

/// Generate noisy 3D circle data parameterized by theta ∈ [0, 2π).
/// Returns `(theta, clean[n,3], noisy[n,3])`.
fn make_noisy_circle(n: usize, seed: u64) -> (Vec<f64>, Array2<f64>, Array2<f64>) {
    let mut rng = StdRng::seed_from_u64(seed);
    let utheta = Uniform::new(0.0, TAU).expect("uniform");
    let noise = Normal::new(0.0, 0.07).expect("normal");
    let tilt = 30.0_f64.to_radians();
    let (st, ct) = tilt.sin_cos();

    let theta: Vec<f64> = (0..n).map(|_| utheta.sample(&mut rng)).collect();

    let mut clean = Array2::<f64>::zeros((n, 3));
    let mut noisy = Array2::<f64>::zeros((n, 3));
    for (i, &t) in theta.iter().enumerate() {
        let cx = t.cos();
        let cy = t.sin();
        // Rotate (cx, cy, 0) by `tilt` about x: y' = cy*cos − 0*sin, z' = cy*sin + 0*cos
        let x = cx;
        let y = cy * ct;
        let z = cy * st;
        clean[[i, 0]] = x;
        clean[[i, 1]] = y;
        clean[[i, 2]] = z;
        noisy[[i, 0]] = x + noise.sample(&mut rng);
        noisy[[i, 1]] = y + noise.sample(&mut rng);
        noisy[[i, 2]] = z + noise.sample(&mut rng);
    }
    (theta, clean, noisy)
}

/// Build a dataset with columns [ct, st, y].
fn build_dataset(theta: &[f64], y_col: &[f64]) -> gam::data::EncodedDataset {
    let headers = ["ct", "st", "y"]
        .into_iter()
        .map(String::from)
        .collect::<Vec<_>>();
    let rows = theta
        .iter()
        .zip(y_col.iter())
        .map(|(t, y)| {
            let ct = t.cos();
            let st = t.sin();
            StringRecord::from(vec![ct.to_string(), st.to_string(), y.to_string()])
        })
        .collect::<Vec<_>>();
    encode_recordswith_inferred_schema(headers, rows).expect("encode dataset")
}

/// Fit `y ~ <formula_body>` and return the absolute residual on the *training*
/// fitted values vs the clean ground truth for a single coordinate. We use the
/// training-time fitted values (X · β) rather than rebuilding a prediction
/// design at new points, because the bug we're catching shows up identically
/// at the training points: the LAML optimizer parks at a bad ρ and the
/// resulting β over- or under-fits visibly even at the training thetas.
fn max_residual_against_truth(theta: &[f64], y_noisy: &[f64], y_truth: &[f64],
                              formula_body: &str) -> f64 {
    let data = build_dataset(theta, y_noisy);
    let cfg = FitConfig {
        family: Some("gaussian".to_string()),
        ..FitConfig::default()
    };
    let formula = format!("y ~ {}", formula_body);
    let result = fit_from_formula(&formula, &data, &cfg).expect("fit succeeded");
    let FitResult::Standard(fit) = result else {
        panic!("expected standard Gaussian fit");
    };
    let beta = &fit.fit.beta;
    let design = &fit.design.design;
    assert_eq!(design.ncols(), beta.len(),
               "design width != beta length for `{}`", formula_body);
    let fitted = design.apply(beta);
    // Compare fitted values to the clean truth (not the noisy y) — measures
    // how close the smooth's mean is to the underlying signal.
    let mut max_abs = 0.0_f64;
    for (yhat, yt) in fitted.iter().zip(y_truth.iter()) {
        let r = (yhat - yt).abs();
        if r > max_abs {
            max_abs = r;
        }
    }
    max_abs
}

#[test]
fn cyclic_duchon_centers_80_fit_stays_within_envelope() {
    // 5σ envelope on the noise — a sane fit can never need more than this.
    const TOL: f64 = 0.35;
    init_parallelism();

    let (theta, clean, noisy) = make_noisy_circle(260, 17);
    // Use the z coordinate (most diagnostic in our investigation)
    let y_noisy: Vec<f64> = noisy.column(2).to_vec();
    let y_truth: Vec<f64> = clean.column(2).to_vec();

    let cases: &[(&str, &str)] = &[
        ("thinplate",      "thinplate(ct, st)"),
        ("matern",         "matern(ct, st)"),
        ("duchon default", "duchon(ct, st)"),
        ("duchon 80",      "duchon(ct, st, centers=80)"),
    ];

    let mut violations = Vec::new();
    for (label, body) in cases {
        let m = max_residual_against_truth(&theta, &y_noisy, &y_truth, body);
        eprintln!("[envelope] {label:20s} max|fitted - truth| = {m:.4}");
        if m > TOL {
            violations.push(format!("{label} ({body}): {m:.4} > {TOL}"));
        }
    }
    assert!(
        violations.is_empty(),
        "fit wandered outside the {:.2}σ envelope:\n  - {}",
        TOL / 0.07,
        violations.join("\n  - "),
    );
}
