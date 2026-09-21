//! #4024 — opt-in `knot_placement="quantile"` must not turn a fit that
//! certifies under uniform knots into an outer-REML refusal.
//!
//! PR #3078 measured the convex / concave shape-constrained smooths, the
//! binomial `s(x)` and the global smooth plus a sum-to-zero factor smooth
//! refusing with `DominatedCertifiedPlateau` once knots were placed at data
//! quantiles. These are well-posed models; a refusal on any of them is an
//! optimizer defect (a certified optimum that an evaluated state beats, whose
//! continuation then fails to certify), so every (seed, placement) pair must
//! produce a fit.

use csv::StringRecord;
use gam::{
    FitConfig, FitResult, encode_recordswith_inferred_schema, fit_from_formula, init_parallelism,
};
use rand::SeedableRng;
use rand::rngs::StdRng;
use rand_distr::{Bernoulli, Distribution, Normal, Uniform};

const PLACEMENTS: [&str; 2] = ["uniform", "quantile"];

fn numeric_column(values: &[f64]) -> Vec<String> {
    values.iter().map(f64::to_string).collect()
}

fn fit_outcome(
    formula: &str,
    family: &str,
    headers: &[&str],
    columns: &[Vec<String>],
) -> Result<(), String> {
    let headers: Vec<String> = headers.iter().map(|name| (*name).to_string()).collect();
    let n = columns[0].len();
    let rows: Vec<StringRecord> = (0..n)
        .map(|i| StringRecord::from(columns.iter().map(|c| c[i].clone()).collect::<Vec<_>>()))
        .collect();
    let ds = encode_recordswith_inferred_schema(headers, rows).expect("encode dataset");
    let cfg = FitConfig {
        family: Some(family.to_string()),
        ..FitConfig::default()
    };
    match fit_from_formula(formula, &ds, &cfg) {
        Ok(FitResult::Standard(fit)) => {
            if fit.fit.beta.iter().all(|v| v.is_finite()) {
                Ok(())
            } else {
                Err("non-finite beta".to_string())
            }
        }
        Ok(_) => Err("not a standard fit".to_string()),
        Err(e) => Err(e.to_string()),
    }
}

fn gaussian_quadratic(seed: u64, sign: f64) -> (Vec<f64>, Vec<f64>) {
    let mut rng = StdRng::seed_from_u64(seed);
    let ux = Uniform::new(0.0, 1.0).expect("uniform");
    let noise = Normal::new(0.0, 0.05).expect("normal");
    let mut x: Vec<f64> = (0..400).map(|_| ux.sample(&mut rng)).collect();
    x.sort_by(|a, b| a.partial_cmp(b).unwrap());
    let y = x
        .iter()
        .map(|xi| sign * (xi - 0.5) * (xi - 0.5) + noise.sample(&mut rng))
        .collect();
    (x, y)
}

fn binomial_logistic(seed: u64) -> (Vec<f64>, Vec<f64>) {
    let mut rng = StdRng::seed_from_u64(seed);
    let ux = Uniform::new(0.0, 1.0).expect("uniform");
    let x: Vec<f64> = (0..2000).map(|_| ux.sample(&mut rng)).collect();
    let y = x
        .iter()
        .map(|xi| {
            let p = 1.0 / (1.0 + (-(-0.5 + 2.0 * xi)).exp());
            if Bernoulli::new(p).expect("bernoulli").sample(&mut rng) {
                1.0
            } else {
                0.0
            }
        })
        .collect();
    (x, y)
}

/// A global trend plus a per-level deviation: `2x + d_g·sin(2πx) + N(0, 0.2²)`
/// over three balanced levels, the design `s(x) + s(x, g, bs=sz)` fits. The
/// level deviations sum to zero over the levels, which is the constraint the
/// sum-to-zero factor smooth carries, so the truth is inside the model.
fn global_plus_factor_deviation(seed: u64) -> (Vec<f64>, Vec<String>, Vec<f64>) {
    const LEVELS: [&str; 3] = ["A", "B", "C"];
    const DEVIATIONS: [f64; 3] = [0.5, -0.5, 0.0];
    let mut rng = StdRng::seed_from_u64(seed);
    let ux = Uniform::new(0.0, 1.0).expect("uniform");
    let ug = Uniform::new(0usize, LEVELS.len()).expect("uniform level");
    let noise = Normal::new(0.0, 0.2).expect("normal");
    let (mut x, mut g, mut y) = (Vec::new(), Vec::new(), Vec::new());
    for _ in 0..600 {
        let xi = ux.sample(&mut rng);
        let level = ug.sample(&mut rng);
        let deviation = DEVIATIONS[level] * (2.0 * std::f64::consts::PI * xi).sin();
        y.push(2.0 * xi + deviation + noise.sample(&mut rng));
        g.push(LEVELS[level].to_string());
        x.push(xi);
    }
    (x, g, y)
}

fn assert_all_certify(label: &str, failures: Vec<String>, total: usize) {
    assert!(
        failures.is_empty(),
        "#4024 {label}: {} of {total} fits refused to certify:\n{}",
        failures.len(),
        failures.join("\n\n")
    );
}

fn shape_case(shape: &str, sign: f64) {
    init_parallelism();
    let mut failures = Vec::new();
    for seed in 0..4 {
        let (x, y) = gaussian_quadratic(seed, sign);
        for placement in PLACEMENTS {
            let formula = format!("y ~ s(x, shape=\"{shape}\", knot_placement=\"{placement}\")");
            if let Err(e) = fit_outcome(
                &formula,
                "gaussian",
                &["x", "y"],
                &[numeric_column(&x), numeric_column(&y)],
            ) {
                failures.push(format!("seed {seed} {placement}: {e}"));
            }
        }
    }
    assert_all_certify(shape, failures, 8);
}

#[test]
fn convex_smooth_certifies_under_both_knot_placements_4024() {
    shape_case("convex", 1.0);
}

#[test]
fn concave_smooth_certifies_under_both_knot_placements_4024() {
    shape_case("concave", -1.0);
}

#[test]
fn binomial_smooth_certifies_under_both_knot_placements_4024() {
    init_parallelism();
    let mut failures = Vec::new();
    for seed in 0..4 {
        let (x, y) = binomial_logistic(seed);
        for placement in PLACEMENTS {
            let formula = format!("y ~ s(x, knot_placement=\"{placement}\")");
            if let Err(e) = fit_outcome(
                &formula,
                "binomial",
                &["x", "y"],
                &[numeric_column(&x), numeric_column(&y)],
            ) {
                failures.push(format!("seed {seed} {placement}: {e}"));
            }
        }
    }
    assert_all_certify("binomial s(x)", failures, 8);
}

#[test]
fn global_plus_factor_smooth_certifies_under_both_knot_placements_4024() {
    init_parallelism();
    let mut failures = Vec::new();
    for seed in 0..4 {
        let (x, g, y) = global_plus_factor_deviation(seed);
        for placement in PLACEMENTS {
            let formula = format!(
                "y ~ s(x, knot_placement=\"{placement}\") + \
                 s(x, g, bs=sz, knot_placement=\"{placement}\")"
            );
            if let Err(e) = fit_outcome(
                &formula,
                "gaussian",
                &["x", "g", "y"],
                &[numeric_column(&x), g.clone(), numeric_column(&y)],
            ) {
                failures.push(format!("seed {seed} {placement}: {e}"));
            }
        }
    }
    assert_all_certify("s(x) + s(x, g, bs=sz)", failures, 8);
}
