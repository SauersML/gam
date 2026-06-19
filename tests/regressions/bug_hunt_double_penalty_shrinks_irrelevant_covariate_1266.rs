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
    let unit = Uniform::new(0.0, 1.0).expect("uniform");
    let noise = Normal::new(0.0, 0.3).expect("normal");
    let rows: Vec<StringRecord> = (0..n)
        .map(|_| {
            let x = unit.sample(&mut rng);
            let z = unit.sample(&mut rng);
            let y = (6.0 * x).sin() + noise.sample(&mut rng);
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
/// advancing the penalty cursor by each term's local penalty count, and call
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
        let k = term.penalties_local.len();
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
