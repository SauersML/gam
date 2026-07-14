// Issue #1266 — Half B (irrelevant-covariate shrinkage).
//
// Half A (`bug_hunt_double_penalty_inflates_edf_instead_of_shrinking.rs`)
// proves the DEFAULT double penalty (mgcv `select = TRUE`) does not INFLATE the
// EDF of an `s(x)` fit on linear data. That alone does not prove the second
// (null-space) smoothing parameter is actually LIVE: a fix that merely neutered
// the extra penalty (a single-lambda fold) would also pass Half A while
// silently disabling `select = TRUE`.
//
// Half B is the positive direction the reopened issue demands: on a model
// `y ~ s(x) + s(z)` where the response depends only on `x`, the DEFAULT
// double-penalty smooth on the genuinely-irrelevant covariate `z` must be
// driven toward `EDF -> 0` (mgcv `select = TRUE` term-selection), NOT merely
// "not inflated". An unsupported smooth has a constant + linear null space
// (~2 EDF under a single wiggliness penalty); the live null-space coordinate
// must shrink the term WELL BELOW that floor.
//
// This is a multi-smooth (`smooth_terms.len() == 2`) model, so it is correctly
// excluded from the single-smooth `spline_scan` / residual-cascade fast paths
// (`smooth_terms.len() != 1`) and routes to the dense two-rho path that owns
// BOTH penalties jointly. This test exercises that dense path directly through
// the public `fit_from_formula` API and reads each term's EDF exactly the way
// the model summary does (`UnifiedFitResult::per_term_edf` over the term's
// coefficient block and its penalty cursor).

use csv::StringRecord;
use gam::{
    FitConfig, FitResult, encode_recordswith_inferred_schema, fit_from_formula, init_parallelism,
};
use rand::SeedableRng;
use rand::rngs::StdRng;
use rand_distr::{Distribution, Normal, Uniform};

/// `y = sin(6x) + N(0, 0.3)` with `x, z ~ U(0,1)` independent. `z` carries no
/// signal whatsoever, so a `select = TRUE` smooth on it must shrink out.
fn irrelevant_covariate_dataset(seed: u64, n: usize) -> gam::data::EncodedDataset {
    let mut rng = StdRng::seed_from_u64(seed);
    let unit = Uniform::new(0.0_f64, 1.0).expect("uniform");
    let noise = Normal::new(0.0, 0.3).expect("normal");
    let rows: Vec<StringRecord> = (0..n)
        .map(|_| {
            let x: f64 = unit.sample(&mut rng);
            let z: f64 = unit.sample(&mut rng);
            let y = (6.0_f64 * x).sin() + noise.sample(&mut rng);
            StringRecord::from(vec![x.to_string(), z.to_string(), y.to_string()])
        })
        .collect();
    encode_recordswith_inferred_schema(
        ["x", "z", "y"].into_iter().map(String::from).collect(),
        rows,
    )
    .expect("encode")
}

/// Per-term EDF for the smooth whose name contains `needle`, computed exactly as
/// the model summary does: walk the random-effect ranges then the smooth terms,
/// advancing the penalty cursor by each term's active penalty count, and call
/// `per_term_edf(coeff_range, penalty_cursor, k)` on the matching term.
fn smooth_term_edf(fit: &FitResult, needle: &str) -> f64 {
    let FitResult::Standard(std_fit) = fit else {
        panic!("expected a standard Gaussian fit for a two-smooth dense model");
    };
    let design = &std_fit.design;
    let unified = &std_fit.fit;
    let mut penalty_cursor = 0usize;
    // Random-effect smooths consume one penalty coordinate each (none here, but
    // mirror the summary's cursor bookkeeping for fidelity).
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

#[test]
fn default_double_penalty_shrinks_irrelevant_covariate_edf_below_one() {
    init_parallelism();

    let cfg = FitConfig {
        family: Some("gaussian".to_string()),
        ..FitConfig::default()
    };

    let mut z_edf: Vec<f64> = Vec::new();
    let mut x_edf: Vec<f64> = Vec::new();
    for seed in 200u64..205 {
        let data = irrelevant_covariate_dataset(seed, 800);
        // DEFAULT smooths: `s(x)` with no `bs=`/`double_penalty=` => mgcv
        // `select = TRUE` (double_penalty true) on the default B-spline basis.
        let fit = fit_from_formula("y ~ s(x) + s(z)", &data, &cfg).expect("fit ok");
        z_edf.push(smooth_term_edf(&fit, "z"));
        x_edf.push(smooth_term_edf(&fit, "x"));
    }

    let mean_z = z_edf.iter().sum::<f64>() / z_edf.len() as f64;
    let mean_x = x_edf.iter().sum::<f64>() / x_edf.len() as f64;

    // Sanity: the SUPPORTED smooth `s(x)` must NOT shrink out — sin(6x) is
    // genuinely wiggly, so its EDF should clearly exceed the 2-d null space.
    assert!(
        mean_x > 2.5,
        "supported smooth s(x) failed to recover the sin(6x) signal: \
         mean x edf={mean_x:.6}, values={x_edf:?}"
    );

    // The reopened-#1266 bar, un-weakened: the irrelevant covariate's default
    // double-penalty smooth must shrink WELL BELOW its ~2-EDF null-space floor.
    assert!(
        mean_z < 1.0,
        "default double penalty failed to shrink the irrelevant covariate s(z) \
         (mgcv select=TRUE): mean z edf={mean_z:.6} (must be < 1.0), \
         values={z_edf:?}; supported mean x edf={mean_x:.6}, x values={x_edf:?}"
    );
}

/// `y = 2 + 3·x1 + N(0, 0.3)` with `x1, x2 ~ U(0,1)` independent. `x1` carries a
/// GENUINE strong linear trend (signal in its `{1, x1}` penalty NULL space);
/// `x2` is pure noise (its null space is UNSUPPORTED). Returns the dataset plus
/// the realized `x1` column for a slope measurement.
fn supported_linear_plus_irrelevant_dataset(
    seed: u64,
    n: usize,
) -> (gam::data::EncodedDataset, Vec<f64>) {
    let mut rng = StdRng::seed_from_u64(seed);
    let unit = Uniform::new(0.0_f64, 1.0).expect("uniform");
    let noise = Normal::new(0.0, 0.3).expect("normal");
    let mut x1s = Vec::with_capacity(n);
    let rows: Vec<StringRecord> = (0..n)
        .map(|_| {
            let x1: f64 = unit.sample(&mut rng);
            let x2: f64 = unit.sample(&mut rng);
            x1s.push(x1);
            let y = 2.0 + 3.0 * x1 + noise.sample(&mut rng);
            StringRecord::from(vec![x1.to_string(), x2.to_string(), y.to_string()])
        })
        .collect();
    let ds = encode_recordswith_inferred_schema(
        ["x1", "x2", "y"].into_iter().map(String::from).collect(),
        rows,
    )
    .expect("encode");
    (ds, x1s)
}

/// Slope of the fitted mean `μ = X β̂` w.r.t. a covariate via `cov(x, μ)/var(x)`.
fn fitted_mean_slope(fit: &FitResult, xs: &[f64]) -> f64 {
    let FitResult::Standard(std_fit) = fit else {
        panic!("expected a standard Gaussian fit for a two-smooth dense model");
    };
    let mu = std_fit.design.design.dot(&std_fit.fit.beta);
    let n = xs.len() as f64;
    let xbar = xs.iter().sum::<f64>() / n;
    let mbar = mu.iter().sum::<f64>() / n;
    let (mut sxy, mut sxx) = (0.0, 0.0);
    for (xi, mi) in xs.iter().zip(mu.iter()) {
        sxy += (xi - xbar) * (mi - mbar);
        sxx += (xi - xbar) * (xi - xbar);
    }
    sxy / sxx
}

/// #1266 DISCRIMINATOR (the crux: the null-space shrink-out must be SELECTIVE).
///
/// Half B proves an unsupported term shrinks out when the SUPPORTED term is
/// wiggly (no supported null space to protect). The cheap-and-wrong way to pass
/// it is to over-smooth EVERY double-penalty null space — which would ANNIHILATE
/// a genuinely-supported LINEAR trend (the #1371 failure, but now beside a second
/// smooth). This test puts a supported linear-null-space term and an unsupported
/// term in the SAME fit and asserts the fix distinguishes them by the DATA:
///   * `s(x1)` on `y = 2 + 3·x1`: the slope lives in the term's `{1, x1}` NULL
///     space and is STRONGLY supported — it must be RETAINED (recovered slope
///     ≈ 3), never shrunk to 0.
///   * `s(x2)` on pure noise: its null space is UNSUPPORTED — the null-space
///     ridge must select it OUT, dropping x2's EDF BELOW its single-penalty
///     (`double_penalty = False`) value, which has no null-space ridge and so
///     leaves the linear component un-penalized (mgcv `select = TRUE`).
///
/// Only a pure-REML (data-dependent) selection satisfies both: the symmetric
/// degeneracy prior alone leaves x2 under-shrunk, while a one-sided "always
/// over-smooth" rule would kill x1's slope. The pure-REML shrink-out escape
/// passes because pure REML descends toward shrink-out for x2's null space but
/// strictly opposes it for x1's.
///
/// NOTE the assertion is "double penalty shrinks x2 BELOW the single penalty",
/// NOT "x2 EDF → 0": with a purely-LINEAR supported signal the unsupported
/// term's *bending* (wiggliness) coordinate — a SEPARATE single-penalty
/// selection that this issue does not touch and that is identical for both
/// double- and single-penalty fits — keeps a few EDF of spurious wiggle on this
/// regime's noise draws (single-penalty x2 EDF is itself 1–6.6 here). Comparing
/// double vs single cancels that shared bending baseline and isolates exactly the
/// null-space ridge's #1266 contribution.
#[test]
fn default_double_penalty_keeps_supported_slope_while_shrinking_unsupported() {
    init_parallelism();

    let cfg = FitConfig {
        family: Some("gaussian".to_string()),
        ..FitConfig::default()
    };

    let mut slopes: Vec<f64> = Vec::new();
    let mut x2_double: Vec<f64> = Vec::new();
    let mut x2_single: Vec<f64> = Vec::new();
    for seed in 300u64..305 {
        let (data, x1s) = supported_linear_plus_irrelevant_dataset(seed, 800);
        // DEFAULT (double penalty, mgcv select=TRUE): the null-space ridge is live.
        let fit = fit_from_formula("y ~ s(x1) + s(x2)", &data, &cfg).expect("fit ok");
        slopes.push(fitted_mean_slope(&fit, &x1s));
        x2_double.push(smooth_term_edf(&fit, "x2"));
        // SINGLE penalty: no null-space ridge, so x2's linear component is
        // un-penalized — the floor the double penalty must shrink below.
        let fit_sp = fit_from_formula(
            "y ~ s(x1, double_penalty=False) + s(x2, double_penalty=False)",
            &data,
            &cfg,
        )
        .expect("single-penalty fit ok");
        x2_single.push(smooth_term_edf(&fit_sp, "x2"));
    }

    let mean_slope = slopes.iter().sum::<f64>() / slopes.len() as f64;
    let mean_double = x2_double.iter().sum::<f64>() / x2_double.len() as f64;
    let mean_single = x2_single.iter().sum::<f64>() / x2_single.len() as f64;

    // SUPPORTED null space RETAINED: the genuine slope=3 trend survives (the
    // #1371 guard, now beside a second smooth — the shrink-out must NOT
    // annihilate a real linear effect to chase the unsupported one).
    assert!(
        (mean_slope - 3.0).abs() < 0.4,
        "the supported linear trend on x1 was not retained: recovered mean slope \
         {mean_slope:.4} (truth 3.0), per-seed={slopes:?} — the null-space \
         shrink-out wrongly annihilated a SUPPORTED null space (#1371 dual)"
    );

    // The double penalty must NEVER inflate the unsupported term above the
    // single penalty (the literal #1266 contract).
    assert!(
        mean_double <= mean_single + 1e-9,
        "default double penalty INFLATED the irrelevant covariate s(x2) above its \
         single-penalty EDF (the #1266 contract violation): double mean \
         {mean_double:.6} > single mean {mean_single:.6}; double per-seed \
         {x2_double:?}, single per-seed {x2_single:?}"
    );

    // SELECT=TRUE shrinks it BELOW the single-penalty floor by a real margin:
    // the live null-space ridge genuinely selects the unsupported linear
    // component out, it is not a no-op. (Deterministic seeds 300..305: double
    // mean ≈ 2.21 vs single mean ≈ 3.65 — a ≈1.4 EDF gap.)
    assert!(
        mean_single - mean_double > 0.5,
        "default double penalty did NOT select the irrelevant covariate s(x2) \
         below its single-penalty null-space floor (the null-space ridge is \
         inert): double mean {mean_double:.6}, single mean {mean_single:.6}; \
         double per-seed {x2_double:?}, single per-seed {x2_single:?}"
    );
}
