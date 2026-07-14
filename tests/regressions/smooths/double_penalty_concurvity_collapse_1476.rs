// Issue #1476 — concurvity-triggered double-penalty null-space collapse.
//
// A textbook additive model `y ~ s(x1) + s(x2)` (gaussian, DEFAULT double
// penalty / mgcv `select = TRUE`) on two MODERATELY CORRELATED covariates
// (corr ≈ 0.9) where BOTH smooths carry genuine signal must keep BOTH smooths
// alive. On `main` it did not: REML drove ONE smooth's null-space ridge to
// `λ_nullspace ≈ 1e13`, collapsing that smooth to a flat line (`EDF ≈ 0`) and
// over-crediting the other, while mgcv splits the fit sensibly and recovers
// both partial effects.
//
// Mechanism. Each smooth's `DoublePenaltyNullspace` block (`Z Zᵀ`) shrinks
// that smooth's OWN linear (null-space) component. When `x1, x2` are
// near-collinear their two linear directions overlap, so the joint REML
// objective is essentially FLAT along the "transfer the shared linear signal
// between the two smooths" ridge. The well-determined Gaussian path left those
// null-space coordinates fully `Flat` (no prior curvature), so REML could not
// certify an interior stationary point and one `λ_nullspace` railed to the ρ
// bound — annihilating a SIGNAL-BEARING direction. The fix walls the
// null-space coordinate with the penalized-complexity select-out prior in the
// well-determined regime too: its strictly-positive curvature pins the
// coordinate at a finite stationary ρ, retaining the supported collinear null
// space while still selecting out a genuinely-unsupported one (#1266).
//
// This is a two-smooth dense fit, so it routes through the dense multi-rho path
// that owns BOTH terms' penalties jointly, exercised here through the public
// `fit_from_formula` API. Each term's EDF is read exactly the way the model
// summary does (`per_term_edf` over the term's coefficient block and penalty
// cursor).

use csv::StringRecord;
use gam::{
    FitConfig, FitResult, encode_recordswith_inferred_schema, fit_from_formula, init_parallelism,
};
use rand::SeedableRng;
use rand::rngs::StdRng;
use rand_distr::{Distribution, Normal, Uniform};

/// `n=400`, `x1 ~ U(0,1)`, `x2 = ρ·x1 + sqrt(1-ρ²)·U(0,1)` rescaled to `[0,1]`
/// so `corr(x1, x2) ≈ 0.9`. Both partial effects are real and DISTINCT:
/// `f1 = sin(2π x1)` (wiggly), `f2 = x2²` (a clear non-flat curve). Neither
/// smooth is the irrelevant covariate of #1266 — both must keep real EDF.
fn correlated_smooths_dataset(seed: u64, n: usize) -> gam::data::EncodedDataset {
    let mut rng = StdRng::seed_from_u64(seed);
    let unit = Uniform::new(0.0_f64, 1.0).expect("uniform");
    let noise = Normal::new(0.0, 0.3).expect("normal");
    // Mixing weight giving Pearson corr ≈ 0.9 between x1 and the latent mix.
    let rho = 0.9_f64;
    let comp = (1.0 - rho * rho).sqrt();
    let rows: Vec<StringRecord> = (0..n)
        .map(|_| {
            let x1: f64 = unit.sample(&mut rng);
            let e: f64 = unit.sample(&mut rng);
            // Correlated latent; both marginals are ~U(0,1)-supported, the
            // moderate-concurvity + default-basis regime #1476 targets.
            let x2: f64 = (rho * x1 + comp * e).clamp(0.0, 1.0);
            let f1 = (2.0 * std::f64::consts::PI * x1).sin();
            let f2 = x2 * x2;
            let y = f1 + f2 + noise.sample(&mut rng);
            StringRecord::from(vec![x1.to_string(), x2.to_string(), y.to_string()])
        })
        .collect();
    encode_recordswith_inferred_schema(
        ["x1", "x2", "y"].into_iter().map(String::from).collect(),
        rows,
    )
    .expect("encode")
}

/// Per-term EDF for the smooth whose name contains `needle`, computed exactly as
/// the model summary does: walk random-effect ranges then smooth terms,
/// advancing the penalty cursor by each term's active penalty count, and call
/// `per_term_edf(coeff_range, penalty_cursor, k)` on the matching term.
fn smooth_term_edf(fit: &FitResult, needle: &str) -> f64 {
    let FitResult::Standard(std_fit) = fit else {
        panic!("expected a standard Gaussian fit for a two-smooth dense model");
    };
    let design = &std_fit.design;
    let unified = &std_fit.fit;
    let mut penalty_cursor = 0usize;
    for (_name, _range) in &design.random_effect_ranges {
        penalty_cursor += 1;
    }
    for term in &design.smooth.terms {
        let k = term.active_penalties.len();
        if term.name.contains(needle) {
            return unified.per_term_edf(term.coeff_range.clone(), penalty_cursor, k);
        }
        penalty_cursor += k;
    }
    panic!(
        "no smooth term whose name contains {needle:?}; terms = {:?}",
        design
            .smooth
            .terms
            .iter()
            .map(|t| t.name.clone())
            .collect::<Vec<_>>()
    );
}

/// Largest fitted smoothing parameter — the railed `λ_nullspace ≈ 1e13` of the
/// collapsed smooth shows up here on the broken build.
fn max_lambda(fit: &FitResult) -> f64 {
    let FitResult::Standard(std_fit) = fit else {
        panic!("expected a standard Gaussian fit");
    };
    std_fit.fit.lambdas.iter().copied().fold(0.0_f64, f64::max)
}

#[test]
fn default_double_penalty_keeps_both_correlated_smooths_alive() {
    init_parallelism();

    let cfg = FitConfig {
        family: Some("gaussian".to_string()),
        ..FitConfig::default()
    };

    let mut x1_edf: Vec<f64> = Vec::new();
    let mut x2_edf: Vec<f64> = Vec::new();
    let mut min_term_edf: Vec<f64> = Vec::new();
    let mut max_lambdas: Vec<f64> = Vec::new();
    for seed in 11u64..21 {
        let data = correlated_smooths_dataset(seed, 400);
        // DEFAULT smooths: no `bs=`/`double_penalty=` => mgcv `select = TRUE`
        // (double penalty true) on the default B-spline basis — exactly what a
        // user writing `s(x1)+s(x2)` gets.
        let fit = fit_from_formula("y ~ s(x1) + s(x2)", &data, &cfg).expect("fit ok");
        let e1 = smooth_term_edf(&fit, "x1");
        let e2 = smooth_term_edf(&fit, "x2");
        x1_edf.push(e1);
        x2_edf.push(e2);
        min_term_edf.push(e1.min(e2));
        max_lambdas.push(max_lambda(&fit));
    }

    let mean_x1 = x1_edf.iter().sum::<f64>() / x1_edf.len() as f64;
    let mean_x2 = x2_edf.iter().sum::<f64>() / x2_edf.len() as f64;
    let worst_min = min_term_edf.iter().copied().fold(f64::INFINITY, f64::min);
    let worst_lambda = max_lambdas.iter().copied().fold(0.0_f64, f64::max);

    // Primary #1476 gate, un-weakened: under moderate concurvity NEITHER
    // genuinely-supported smooth may collapse. On the broken build one smooth's
    // EDF dropped to ≈ 0.000 on several seeds; both must stay clearly above the
    // 1-EDF select-out floor on EVERY seed.
    assert!(
        worst_min > 1.0,
        "a genuinely-supported smooth collapsed under concurvity (#1476): \
         worst per-seed min(edf[s(x1)], edf[s(x2)]) = {worst_min:.6} (must be > 1.0); \
         mean edf[s(x1)] = {mean_x1:.4}, mean edf[s(x2)] = {mean_x2:.4}; \
         x1 = {x1_edf:?}, x2 = {x2_edf:?}"
    );

    // Both partial effects carry real curvature, so on average each smooth must
    // recover meaningfully more than its ~2-EDF linear null space.
    assert!(
        mean_x1 > 2.0 && mean_x2 > 2.0,
        "a correlated smooth under-recovered its real partial effect (#1476): \
         mean edf[s(x1)] = {mean_x1:.4}, mean edf[s(x2)] = {mean_x2:.4} (each must be > 2.0); \
         x1 = {x1_edf:?}, x2 = {x2_edf:?}"
    );

    // The mechanism guard: no smoothing parameter may rail to the
    // null-space-collapse runaway (`λ_nullspace ≈ 1e13`). The penalized-complexity
    // wall pins it at a finite stationary point.
    assert!(
        worst_lambda < 1e10,
        "a null-space smoothing parameter railed to the collapse bound (#1476): \
         worst fitted λ = {worst_lambda:.3e} (must be < 1e10); lambdas-max = {max_lambdas:?}"
    );
}
