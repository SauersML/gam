// Issue #1266 — Half A (a supported fit is not distorted by `select = TRUE`),
// re-derived for #2668 group C.
//
// On `y = 2 + 3x + N(0, 0.15)` the truth lives entirely in the bending
// penalty's null space `{1, x}`. The DEFAULT double penalty adds a null-space
// ridge whose smoothing parameter REML must switch OFF here (the null space is
// supported), so the double-penalty fit must reproduce the single-penalty fit
// and the true line — never the #1266 failure, where a prior cap on the ridge
// coordinate kept the linear part shrunk and REML answered with spurious
// wiggle in its place (EDF 5–10 on a straight line).
//
// WHAT THIS TEST NO LONGER CLAIMS, AND WHY (measured 2026-09-04). The previous
// version asserted `mean edf ≤ 2.35` over five seeds as "the mgcv linear-data
// EDF target (~2.10)". On the identical five data sets, with the identical
// basis (a cubic B-spline with an integrated squared second-derivative penalty
// is mgcv `bs="bs", m=c(3,2)`; `k=20`, `select=TRUE`), mgcv's own REML optimum
// is 2.0023 / 3.4224 / 2.5749 / 3.6280 / 2.0004 edf (mean 2.726) against
// gam's 2.00002 / 3.4221 / 2.5744 / 3.6280 / 1.99998, and a scan of mgcv's
// criterion along the bending coordinate shows the λ = ∞ face is 1.25 / 0.13 /
// 3.70 nats WORSE than the interior optimum on seeds 1–3. REML keeps a little
// spurious wiggle on those noise draws; that is a property of the criterion,
// shared exactly with the reference, not an engine defect. The old bar was a
// number the reference does not produce on this data.
//
// What IS true on every draw, and what #1266 was actually about:
//   * the fitted line is the true line to within its own uncertainty;
//   * the double-penalty fit and the single-penalty fit are the same function
//     to within the fit's posterior standard deviation at every design point —
//     the null-space ridge, switched off, changes nothing a user can see.

use csv::StringRecord;
use gam::linalg::matrix::DesignMatrix;
use gam::{
    FitConfig, FitResult, encode_recordswith_inferred_schema, fit_from_formula, init_parallelism,
};
use ndarray::Array2;
use rand::SeedableRng;
use rand::rngs::StdRng;
use rand_distr::{Distribution, Normal};

const NOISE_SD: f64 = 0.15;
const TRUE_INTERCEPT: f64 = 2.0;
const TRUE_SLOPE: f64 = 3.0;

fn linear_dataset(seed: u64, n: usize) -> (gam::data::EncodedDataset, Vec<f64>) {
    let mut rng = StdRng::seed_from_u64(seed);
    let noise = Normal::new(0.0, NOISE_SD).expect("normal");
    let mut xs = Vec::with_capacity(n);
    let rows: Vec<StringRecord> = (0..n)
        .map(|i| {
            let x = i as f64 / (n.saturating_sub(1).max(1)) as f64;
            let y = TRUE_INTERCEPT + TRUE_SLOPE * x + noise.sample(&mut rng);
            xs.push(x);
            StringRecord::from(vec![x.to_string(), y.to_string()])
        })
        .collect();
    let data = encode_recordswith_inferred_schema(
        ["x", "y"].into_iter().map(String::from).collect(),
        rows,
    )
    .expect("encode");
    (data, xs)
}

/// The fitted function at every design row, with its posterior standard
/// deviation, on whichever route the fit took (the single-penalty single-smooth
/// Gaussian fit is the spline scan; the double-penalty fit is the dense
/// two-ρ standard route).
struct FittedCurve {
    mean: Vec<f64>,
    sd: Vec<f64>,
    edf: f64,
    lambdas: Vec<f64>,
    route: &'static str,
}

fn fitted_curve(fit: &FitResult, xs: &[f64]) -> FittedCurve {
    match fit {
        FitResult::Standard(std_fit) => {
            let x: Array2<f64> = match &std_fit.design.design {
                DesignMatrix::Dense(m) => m.to_dense(),
                other => panic!("expected a dense design, got {other:?}"),
            };
            let mean = x.dot(&std_fit.fit.beta).to_vec();
            let cov = std_fit
                .fit
                .covariance_conditional
                .as_ref()
                .expect("a default fit publishes its conditional covariance");
            let sd = (0..x.nrows())
                .map(|i| {
                    let xi = x.row(i).to_owned();
                    xi.dot(&cov.dot(&xi)).sqrt()
                })
                .collect();
            FittedCurve {
                mean,
                sd,
                edf: std_fit
                    .fit
                    .inference
                    .as_ref()
                    .expect("default fit must compute inference")
                    .edf_total,
                lambdas: std_fit.fit.lambdas.to_vec(),
                route: "standard",
            }
        }
        FitResult::SplineScan(scan) => {
            let mut mean = Vec::with_capacity(xs.len());
            let mut sd = Vec::with_capacity(xs.len());
            for &x in xs {
                // `predict` returns the exact posterior (mean, VARIANCE).
                let (m, var) = scan.predict(x).expect("spline scan predicts on its own support");
                mean.push(m);
                sd.push(var.sqrt());
            }
            FittedCurve {
                mean,
                sd,
                edf: scan.edf(),
                lambdas: vec![scan.lambda()],
                route: "spline_scan",
            }
        }
        _ => panic!("expected a standard Gaussian or spline-scan fit"),
    }
}

#[test]
fn bspline_double_penalty_does_not_distort_a_supported_linear_fit() {
    init_parallelism();
    let cfg = FitConfig {
        family: Some("gaussian".to_string()),
        ..FitConfig::default()
    };
    let n = 800usize;
    let mut report: Vec<String> = Vec::new();
    let mut failures: Vec<String> = Vec::new();
    for seed in 0..5u64 {
        let (data, xs) = linear_dataset(seed, n);
        let on = fit_from_formula("y ~ s(x, k=20, bs=ps, double_penalty=True)", &data, &cfg)
            .expect("double-penalty fit ok");
        let off = fit_from_formula("y ~ s(x, k=20, bs=ps, double_penalty=False)", &data, &cfg)
            .expect("single-penalty fit ok");
        let on_curve = fitted_curve(&on, &xs);
        let off_curve = fitted_curve(&off, &xs);

        // (1) The true line, to within the fit's own uncertainty: at every
        // design row the fitted mean is within four posterior standard
        // deviations of the truth (the same bar, 4 sd of the row's own
        // posterior, at every row).
        let mut worst_truth_z = 0.0_f64;
        for (i, &x) in xs.iter().enumerate() {
            let truth = TRUE_INTERCEPT + TRUE_SLOPE * x;
            worst_truth_z = worst_truth_z.max((on_curve.mean[i] - truth).abs() / on_curve.sd[i]);
        }
        // (2) The double-penalty fit is the single-penalty fit: the two curves
        // differ by less than one posterior standard deviation at every row.
        let mut worst_pair_z = 0.0_f64;
        for i in 0..n {
            worst_pair_z =
                worst_pair_z.max((on_curve.mean[i] - off_curve.mean[i]).abs() / on_curve.sd[i]);
        }
        report.push(format!(
            "seed {seed}: double_penalty=true ({}) edf={:.6} lambdas={:?}; double_penalty=false ({}) \
             edf={:.6} lambdas={:?}; max |fit - truth|/sd = {worst_truth_z:.3}; \
             max |double - single|/sd = {worst_pair_z:.3e}",
            on_curve.route,
            on_curve.edf,
            on_curve.lambdas,
            off_curve.route,
            off_curve.edf,
            off_curve.lambdas
        ));
        if !(worst_truth_z <= 4.0) {
            failures.push(format!(
                "seed {seed}: the double-penalty fit leaves the true line by \
                 {worst_truth_z:.3} posterior sd at some design row"
            ));
        }
        if !(worst_pair_z <= 1.0) {
            failures.push(format!(
                "seed {seed}: the double-penalty fit differs from the single-penalty fit by \
                 {worst_pair_z:.3} posterior sd at some design row — the null-space ridge \
                 distorted a supported linear fit (#1266)"
            ));
        }
    }
    println!("{}", report.join("\n"));
    assert!(
        failures.is_empty(),
        "B-spline double penalty distorted a supported linear fit:\n{}\nper-seed report:\n{}",
        failures.join("\n"),
        report.join("\n")
    );
}
