// Issue #1266 — Half B (irrelevant-covariate selection), re-derived for #2668.
//
// Half A (`double_penalty_inflates_edf_instead_of_shrinking.rs`) proves the
// DEFAULT double penalty (mgcv `select = TRUE`) does not distort a supported
// fit. That alone does not prove the second (null-space) smoothing parameter is
// LIVE: a fix that merely neutered the extra penalty would also pass Half A
// while silently disabling `select = TRUE`.
//
// Half B is the positive direction: on `y ~ s(x) + s(z)` where the response
// depends only on `x`, REML must be free to switch the irrelevant `s(z)` OFF —
// both of its coordinates (bending and null-space) must be able to reach the
// λ = ∞ face when the criterion is lowest there.
//
// WHAT THIS TEST DOES NOT CLAIM, AND WHY (#2668 group C, measured 2026-09-04).
// The previous version asserted `mean z edf < 1.0` over five seeds against an
// "mgcv select=TRUE" reference. On the identical data, with the identical basis
// family (gam's default `bs=ps` is a cubic B-spline with an integrated squared
// second-derivative penalty = mgcv `bs="bs", m=c(3,2)`), mgcv's own REML
// optimum for `s(z)` was 0.45 / 1.84 / 1.01 / 1.76 / 0.62 edf (mean 1.14) on
// seeds 200–204, and a scan of mgcv's REML along z's bending coordinate showed
// the λ = ∞ face is 0.13–0.45 nats WORSE than the interior optimum on four of
// the five seeds. The "0.31 edf" that motivated the bar is mgcv with its
// default thin-plate basis on the same seeds — a different penalty spectrum,
// not a different estimator. Under the null, REML's variance-component estimate
// is positive with substantial probability per coordinate, so a term's REML
// optimum keeps a little wiggle on a large fraction of pure-noise draws; that
// is a property of REML, not of this engine. (The old helper also indexed the
// per-term edf with the block-LOCAL `coeff_range`, folding the intercept into
// `s(x)` and `s(x)`'s last column into `s(z)` — the production summary offsets
// by `smooth_start`; this test reads the summary rows instead.)
//
// The exact, reference-free statement of "term selection works" is therefore:
//
//   the shipped fit is NEVER BEATEN BY DELETING THE TERM.
//
// Deleting `s(z)` is the λ = ∞ face of both of its coordinates, and on that
// face the criterion equals the reduced model's criterion at the same `s(x)`
// smoothing (the divergent log-determinants cancel exactly; the reduced fit
// re-optimises `s(x)` and so can only be lower). So for every seed
//
//   reml(y ~ s(x) + s(z))  ≤  reml(y ~ s(x))  +  band,
//
// with `band` the gap a certified stop can leave on an exponential tail (the
// terminal gradient, see `criterion_band`) plus the criterion's arithmetic
// resolution √ε·(1 + |V|). A violation means a strictly better term-deletion face existed and the
// optimizer stopped short of it — exactly the #1266 failure (a prior cap or a
// stalled tail keeping λ finite). Equality within the band means the fit IS on
// the deletion face; strict inequality means REML genuinely prefers a finite
// smoothing for `s(z)` on that draw. Both regimes must occur across the seed
// sweep, or the comparison has not been exercised (a null result needs its
// non-vacuity control).
//
// This is a multi-smooth (`smooth_terms.len() == 2`) model, so it routes to
// the dense multi-ρ path that owns BOTH penalties jointly; the reduced model
// `y ~ s(x)` takes whatever route a single default smooth takes, and both
// report the same profiled criterion (`reml_score`, one constant convention).

use csv::StringRecord;
use gam::{
    FitConfig, FitResult, encode_recordswith_inferred_schema, fit_from_formula, init_parallelism,
};
use gam_solve::estimate::smooth_term_summary_rows;
use rand::SeedableRng;
use rand::rngs::StdRng;
use rand_distr::{Distribution, Normal, Uniform};

/// `y = sin(6x) + N(0, 0.3)` with `x, z ~ U(0,1)` independent. `z` carries no
/// signal whatsoever.
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

fn standard(fit: &FitResult) -> &gam::StandardFitResult {
    let FitResult::Standard(std_fit) = fit else {
        panic!("expected a standard Gaussian fit");
    };
    std_fit
}

/// Per-term EDF exactly as the model summary reports it: the influence-matrix
/// trace over the term's GLOBAL coefficient block (the summary applies the
/// `smooth_start` offset; a block-local `coeff_range` is not a column index).
fn smooth_term_edf(fit: &FitResult, needle: &str) -> f64 {
    let std_fit = standard(fit);
    let rows = smooth_term_summary_rows(&std_fit.design, &std_fit.resolvedspec, &std_fit.fit, None);
    rows.iter()
        .find(|row| row.name.contains(needle))
        .map(|row| row.edf)
        .unwrap_or_else(|| {
            panic!(
                "no smooth term whose name contains {needle:?}; terms = {:?}",
                rows.iter().map(|r| r.name.clone()).collect::<Vec<_>>()
            )
        })
}

fn reml_score(fit: &FitResult) -> f64 {
    standard(fit)
        .fit
        .reml_score()
        .expect("a Gaussian REML fit reports its criterion")
}

fn outer_gradient_norm(fit: &FitResult) -> f64 {
    standard(fit)
        .fit
        .outer_gradient_norm
        .expect("a certified outer fit reports its terminal projected gradient norm")
}

/// How far a certified fit's criterion can sit above its own limit.
///
/// A coordinate stopped on its λ→∞ tail obeys `V(ρ) − V_∞ = c·e^{−ρ} = |∂V/∂ρ|`
/// exactly, so the criterion gap left by a stop with terminal gradient `g` is
/// at most `Σ_j |g_j| ≤ √d·‖g‖` over the `d` outer coordinates; the reduced
/// fit contributes its own residual the same way. Below that sits the
/// criterion's arithmetic resolution √ε·(1 + |V|) — the same species the outer
/// engine floors its gradient band at.
fn criterion_band(full: &FitResult, reduced: &FitResult, v_full: f64, v_reduced: f64) -> f64 {
    let d = standard(full)
        .fit
        .log_lambdas
        .len()
        .max(standard(reduced).fit.log_lambdas.len());
    (d as f64).sqrt() * (outer_gradient_norm(full) + outer_gradient_norm(reduced))
        + f64::EPSILON.sqrt() * (1.0 + v_full.abs().max(v_reduced.abs()))
}

#[test]
fn default_double_penalty_is_never_beaten_by_deleting_the_irrelevant_covariate_1266() {
    init_parallelism();

    let cfg = FitConfig {
        family: Some("gaussian".to_string()),
        ..FitConfig::default()
    };

    let mut at_face: Vec<u64> = Vec::new();
    let mut interior: Vec<u64> = Vec::new();
    let mut beaten: Vec<String> = Vec::new();
    let mut x_edf: Vec<f64> = Vec::new();
    for seed in 200u64..220 {
        let data = irrelevant_covariate_dataset(seed, 800);
        // DEFAULT smooths: `s(x)` with no `bs=`/`double_penalty=` => mgcv
        // `select = TRUE` (double_penalty true) on the default B-spline basis.
        let full = fit_from_formula("y ~ s(x) + s(z)", &data, &cfg).expect("full fit ok");
        let reduced = fit_from_formula("y ~ s(x)", &data, &cfg).expect("reduced fit ok");
        let v_full = reml_score(&full);
        let v_reduced = reml_score(&reduced);
        let band = criterion_band(&full, &reduced, v_full, v_reduced);
        let gap = v_full - v_reduced;
        x_edf.push(smooth_term_edf(&full, "x"));
        let z_edf = smooth_term_edf(&full, "z");
        let rho = standard(&full).fit.log_lambdas.to_vec();
        if gap > band {
            beaten.push(format!(
                "seed {seed}: reml(full)={v_full:.9} > reml(y ~ s(x))={v_reduced:.9} by {gap:.3e} \
                 (band {band:.1e}); z edf={z_edf:.6}, fitted rho=[x bend, x null, z bend, z null]={rho:?}"
            ));
        } else if gap < -band {
            interior.push(seed);
        } else {
            at_face.push(seed);
        }
    }

    // The supported smooth `s(x)` must NOT shrink out — sin(6x) is genuinely
    // wiggly, so its EDF clearly exceeds the 2-d null space on every draw.
    let mean_x = x_edf.iter().sum::<f64>() / x_edf.len() as f64;
    assert!(
        mean_x > 2.5,
        "supported smooth s(x) failed to recover the sin(6x) signal: mean x edf={mean_x:.6}, \
         values={x_edf:?}"
    );

    // The #1266 contract, exact: no seed's fit is beaten by deleting s(z).
    assert!(
        beaten.is_empty(),
        "default double penalty left s(z) with a finite smoothing where switching the term \
         OFF has a lower REML (the deletion face was reachable and not reached):\n{}",
        beaten.join("\n")
    );

    // Non-vacuity: the sweep must contain BOTH regimes, or the comparison above
    // proved nothing about the face. Seeds on the deletion face show the
    // λ = ∞ face is representable and reached; interior seeds show the bound is
    // not trivially tight.
    assert!(
        !at_face.is_empty(),
        "no seed in 200..220 switched s(z) off (fit on the deletion face within the \
         criterion's resolution); interior seeds={interior:?}. The face is either \
         unreachable or the criterion never prefers it — either way this test \
         measured nothing about term selection"
    );
    assert!(
        !interior.is_empty(),
        "every seed in 200..220 sits exactly on the deletion face (at-face={at_face:?}); \
         the bound was never a comparison between two distinct optima"
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
    let std_fit = standard(fit);
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
/// The test above proves an unsupported term is switched off whenever the
/// criterion prefers that. The cheap-and-wrong way to pass it is to over-smooth
/// EVERY double-penalty null space — which would ANNIHILATE a genuinely-supported
/// LINEAR trend (the #1371 failure, but now beside a second smooth). This test
/// puts a supported linear-null-space term and an unsupported term in the SAME
/// fit and asserts the fix distinguishes them by the DATA:
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
/// over-smooth" rule would kill x1's slope.
///
/// NOTE the assertion is "double penalty shrinks x2 BELOW the single penalty",
/// NOT "x2 EDF → 0": with a purely-LINEAR supported signal the unsupported
/// term's *bending* (wiggliness) coordinate — a SEPARATE single-penalty
/// selection that this issue does not touch and that is identical for both
/// double- and single-penalty fits — keeps a few EDF of spurious wiggle on this
/// regime's noise draws. Comparing double vs single cancels that shared bending
/// baseline and isolates exactly the null-space ridge's #1266 contribution.
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
    // component out, it is not a no-op.
    assert!(
        mean_single - mean_double > 0.5,
        "default double penalty did NOT select the irrelevant covariate s(x2) \
         below its single-penalty null-space floor (the null-space ridge is \
         inert): double mean {mean_double:.6}, single mean {mean_single:.6}; \
         double per-seed {x2_double:?}, single per-seed {x2_single:?}"
    );
}
