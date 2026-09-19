//! Standing type-I size and power gate for the p-value of a default `s()`
//! smooth, one test per response family (pyGAM audit, lanes
//! `pv-wald-families` and `pv-random-effects`).
//!
//! The default smooth is double-penalized: a wiggliness penalty plus a ridge on
//! its polynomial null space, so the penalties cover every coefficient
//! direction and "no effect" is every variance component at zero — on the
//! boundary of the parameter space. Its summary row is the variance-component
//! score test (`gam_terms::inference::variance_component_test`) scored against
//! its exact finite-sample null law, not the Wood (2013) rank-truncated Wald
//! test the row used to carry. That Wald test was computed from coefficients
//! REML had shrunk from the same data: about half of all null fits shrank the
//! term to `edf ≈ 0` and reported `p ≈ 1`, a point mass that made the test
//! conservative at every level (the bench in
//! `bench/pvalue_calibration/pv-random-effects/` has the before/after tables).
//!
//! The family changes three inputs of the test: the IRLS weights inside the
//! projection and the score variance, the scale (profiled, or fixed), and the
//! reference law (a weighted `χ²₁` sum, ratioed against the residual `χ²` when
//! the scale is estimated). A wrong weight, a wrong scale predicate or a
//! residual df read off the wrong fit is invisible to a Gaussian-only gate.
//!
//! Audit: `y ~ s(x1) + s(x2)` with a real `s(x1)` and a TRUE-NULL `s(x2)`
//! (`x2` is drawn independently of `y`), `n = 200`, 200 seeded replications
//! per family. The p-value read is the production summary row — the shared
//! `smooth_term_summary_rows` walk the CLI and Python summaries use. At
//! `α ∈ {0.10, 0.05, 0.01}` the empirical size must stay within
//! `α ± 2·MCSE(α)`, `MCSE(α) = √(α(1 − α)/m)`, over the `m` fits that
//! converged, and the whole null p-value sample must pass a two-sided
//! Kolmogorov–Smirnov test against `U(0, 1)` at level 0.01. The reference law
//! is exact and continuous, so an undersized (conservative) test fails exactly
//! like an oversized one.
//!
//! A power control fits a real `s(x2)` and requires the test to find it.

use csv::StringRecord;
use gam::{
    FitConfig, FitResult, encode_recordswith_inferred_schema, fit_from_formula, init_parallelism,
};
use gam_solve::estimate::smooth_term_summary_rows;
use rand::SeedableRng;
use rand::rngs::StdRng;
use rand_distr::{Beta, Distribution, Gamma, Normal, Poisson, Uniform};
use rayon::iter::{IntoParallelIterator, ParallelIterator};
use std::f64::consts::PI;

const N_OBS: usize = 200;
const N_REPLICATIONS: u64 = 200;
const ALPHAS: [f64; 3] = [0.10, 0.05, 0.01];
/// Level of the two-sided Kolmogorov–Smirnov uniformity test.
const KS_LEVEL: f64 = 0.01;
const SEED: u64 = 0x5A17_3051_0000;
/// A fit the outer optimizer refuses to certify returns an error, not a
/// p-value. Those replications are reported, never counted as rejections or
/// non-rejections; more than this share of them would leave the size estimate
/// resting on a selected subsample.
const MAX_FAILED_FIT_SHARE: f64 = 0.05;
const FORMULA: &str = "y ~ s(x1) + s(x2)";
const NULL_TERM: &str = "x2";

#[derive(Clone, Copy, Debug)]
enum Family {
    Gaussian,
    Poisson,
    Binomial,
    Gamma,
    NegativeBinomial,
    Tweedie,
    Beta,
}

impl Family {
    fn config_name(self) -> &'static str {
        match self {
            Self::Gaussian => "gaussian",
            Self::Poisson => "poisson",
            Self::Binomial => "binomial",
            Self::Gamma => "gamma",
            Self::NegativeBinomial => "negative-binomial",
            // The variance power is fixed, never estimated (SPEC), and must
            // be named: p = 1.5, matching the simulated response.
            Self::Tweedie => "tweedie(p=1.5)",
            Self::Beta => "beta",
        }
    }

    fn index(self) -> u64 {
        self as u64
    }

    /// One response draw at linear predictor `f`.
    fn draw(self, eta: f64, rng: &mut StdRng) -> f64 {
        match self {
            Self::Gaussian => eta + Normal::new(0.0, 0.5).expect("normal").sample(rng),
            Self::Poisson => Poisson::new((0.5 + 0.5 * eta).exp())
                .expect("poisson rate")
                .sample(rng),
            Self::Binomial => {
                let p = 1.0 / (1.0 + (-eta).exp());
                if Uniform::new(0.0, 1.0).expect("uniform").sample(rng) < p {
                    1.0
                } else {
                    0.0
                }
            }
            Self::Gamma => {
                let mu = (1.0 + 0.5 * eta).exp();
                let shape = 3.0;
                Gamma::new(shape, mu / shape).expect("gamma").sample(rng)
            }
            Self::NegativeBinomial => {
                // Poisson–gamma mixture with size θ = 2.
                let mu = (1.0 + 0.5 * eta).exp();
                let theta = 2.0;
                let rate = Gamma::new(theta, mu / theta).expect("gamma").sample(rng);
                Poisson::new(rate.max(1e-12)).expect("poisson").sample(rng)
            }
            Self::Tweedie => {
                // Compound Poisson–gamma with p = 1.5, φ = 1.
                let mu = (0.5 + 0.5 * eta).exp();
                let (p, phi) = (1.5_f64, 1.0_f64);
                let lambda = mu.powf(2.0 - p) / (phi * (2.0 - p));
                let alpha = (2.0 - p) / (p - 1.0);
                let scale = phi * (p - 1.0) * mu.powf(p - 1.0);
                let count = Poisson::new(lambda).expect("poisson").sample(rng);
                if count > 0.0 {
                    Gamma::new(alpha * count, scale).expect("gamma").sample(rng)
                } else {
                    0.0
                }
            }
            Self::Beta => {
                let mu = 1.0 / (1.0 + (-0.5 * eta).exp());
                let phi = 10.0;
                Beta::new(mu * phi, (1.0 - mu) * phi)
                    .expect("beta")
                    .sample(rng)
                    .clamp(1e-6, 1.0 - 1e-6)
            }
        }
    }
}

struct SmoothRow {
    p_value: f64,
    edf: f64,
    ref_df: f64,
}

/// `y` at `sin(2πx1) + effect·sin(2πx2)`; `effect = 0` is the null.
fn dataset(family: Family, rep: u64, effect: f64) -> gam::data::EncodedDataset {
    let mut rng = StdRng::seed_from_u64(SEED + 1_000_000 * family.index() + rep);
    let unit = Uniform::new(0.0_f64, 1.0).expect("uniform");
    let rows: Vec<StringRecord> = (0..N_OBS)
        .map(|_| {
            let x1 = unit.sample(&mut rng);
            let x2 = unit.sample(&mut rng);
            let y = family.draw(
                (2.0 * PI * x1).sin() + effect * (2.0 * PI * x2).sin(),
                &mut rng,
            );
            StringRecord::from(vec![
                format!("{x1:.17e}"),
                format!("{x2:.17e}"),
                format!("{y:.17e}"),
            ])
        })
        .collect();
    encode_recordswith_inferred_schema(
        ["x1", "x2", "y"].into_iter().map(String::from).collect(),
        rows,
    )
    .expect("encode dataset")
}

/// The `s(x2)` summary row, or the fit error that stopped it.
fn tested_row(family: Family, rep: u64, effect: f64) -> Result<SmoothRow, String> {
    let data = dataset(family, rep, effect);
    let config = FitConfig {
        family: Some(family.config_name().to_string()),
        ..FitConfig::default()
    };
    let result = fit_from_formula(FORMULA, &data, &config).map_err(|e| format!("{e:?}"))?;
    let FitResult::Standard(fit) = result else {
        panic!("{family:?} rep {rep}: expected a standard fit");
    };
    let rows = smooth_term_summary_rows(
        &fit.design,
        &fit.resolvedspec,
        &fit.fit,
        fit.fit.weighted_gram(),
    );
    let row = rows
        .iter()
        .find(|row| row.name.contains(NULL_TERM))
        .unwrap_or_else(|| panic!("{family:?} rep {rep}: no summary row for s({NULL_TERM})"));
    let p_value = row.pvalue.unwrap_or_else(|| {
        panic!(
            "{family:?} rep {rep}: s({NULL_TERM}) reported no p-value (edf {}, ref_df {}, {:?})",
            row.edf, row.ref_df, row.pvalue_unavailable
        )
    });
    Ok(SmoothRow {
        p_value,
        edf: row.edf,
        ref_df: row.ref_df,
    })
}

fn assert_null_size_within_monte_carlo_error(family: Family) {
    // The fits run on rayon workers, which need the wide worker stack.
    init_parallelism();
    let outcomes: Vec<(u64, Result<SmoothRow, String>)> = (0..N_REPLICATIONS)
        .into_par_iter()
        .map(|rep| (rep, tested_row(family, rep, 0.0)))
        .collect();

    let mut rows = Vec::new();
    let mut failed_fits = Vec::new();
    for (rep, outcome) in outcomes {
        match outcome {
            Ok(row) => rows.push((rep, row)),
            Err(reason) => failed_fits.push(format!("rep {rep}: {reason}")),
        }
    }
    assert!(
        (failed_fits.len() as f64) <= MAX_FAILED_FIT_SHARE * N_REPLICATIONS as f64,
        "{family:?}: {} of {N_REPLICATIONS} null fits failed, first: {}",
        failed_fits.len(),
        failed_fits[0]
    );

    for (rep, row) in &rows {
        assert!(
            row.p_value.is_finite() && (0.0..=1.0).contains(&row.p_value),
            "{family:?} rep {rep}: p-value out of range: {}",
            row.p_value
        );
        // The effective reference df `(Σμ)²/Σμ²` of a weighted χ² sum is at
        // least one, whatever edf REML chose.
        assert!(
            row.ref_df.is_finite() && row.ref_df >= 1.0,
            "{family:?} rep {rep}: ref_df {} undefined or below one at edf {}",
            row.ref_df,
            row.edf
        );
    }

    let m = rows.len() as f64;
    let p_values: Vec<f64> = rows.iter().map(|(_, row)| row.p_value).collect();
    let mut miscalibrated = Vec::new();
    let mut report = Vec::new();
    for &alpha in &ALPHAS {
        let rejections = p_values.iter().filter(|&&p| p <= alpha).count();
        let size = rejections as f64 / m;
        let half_width = 2.0 * (alpha * (1.0 - alpha) / m).sqrt();
        report.push(format!("α={alpha}: size {size:.4} (α ± {half_width:.4})"));
        if (size - alpha).abs() > half_width {
            miscalibrated.push(format!(
                "α={alpha}: {rejections}/{} rejections, size {size:.4} outside α ± 2·MCSE",
                p_values.len()
            ));
        }
    }
    let (ks_distance, ks_p_value) = ks_uniform(&p_values);
    report.push(format!("KS D {ks_distance:.4}, p {ks_p_value:.4}"));
    if ks_p_value < KS_LEVEL {
        miscalibrated.push(format!(
            "null p-values are not U(0, 1): two-sided KS D = {ks_distance:.4}, \
             p = {ks_p_value:.4} < {KS_LEVEL}"
        ));
    }
    eprintln!(
        "{family:?}: {} usable fits, {} failed; {}",
        rows.len(),
        failed_fits.len(),
        report.join("; ")
    );
    assert!(
        miscalibrated.is_empty(),
        "{family:?}: the smooth-term p-value is miscalibrated under a true-null \
         s({NULL_TERM}):\n{}",
        miscalibrated.join("\n")
    );
}

/// Two-sided one-sample Kolmogorov–Smirnov test of `values` against `U(0, 1)`:
/// the distance `D = sup |F_m − F|` and its asymptotic p-value
/// `Q(√m·D)`, `Q(λ) = 2 Σ_{k≥1} (−1)^{k−1} e^{−2k²λ²}`.
fn ks_uniform(values: &[f64]) -> (f64, f64) {
    let mut sorted = values.to_vec();
    sorted.sort_by(f64::total_cmp);
    let m = sorted.len() as f64;
    let distance = sorted
        .iter()
        .enumerate()
        .map(|(i, &p)| ((i as f64 + 1.0) / m - p).max(p - i as f64 / m))
        .fold(0.0_f64, f64::max);
    let lambda = m.sqrt() * distance;
    // The alternating series converges geometrically; stop once a term no
    // longer changes the sum in double precision.
    let mut sum = 0.0;
    let mut k = 1.0_f64;
    loop {
        let term = (-2.0 * k * k * lambda * lambda).exp();
        let signed = if (k as u64) % 2 == 1 { term } else { -term };
        if sum + signed == sum {
            break;
        }
        sum += signed;
        k += 1.0;
    }
    (distance, (2.0 * sum).clamp(0.0, 1.0))
}

#[test]
fn gaussian_null_smooth_size_is_within_monte_carlo_error() {
    assert_null_size_within_monte_carlo_error(Family::Gaussian);
}

#[test]
fn poisson_null_smooth_size_is_within_monte_carlo_error() {
    assert_null_size_within_monte_carlo_error(Family::Poisson);
}

#[test]
fn binomial_null_smooth_size_is_within_monte_carlo_error() {
    assert_null_size_within_monte_carlo_error(Family::Binomial);
}

#[test]
fn gamma_null_smooth_size_is_within_monte_carlo_error() {
    assert_null_size_within_monte_carlo_error(Family::Gamma);
}

#[test]
fn negative_binomial_null_smooth_size_is_within_monte_carlo_error() {
    assert_null_size_within_monte_carlo_error(Family::NegativeBinomial);
}

#[test]
fn tweedie_null_smooth_size_is_within_monte_carlo_error() {
    assert_null_size_within_monte_carlo_error(Family::Tweedie);
}

#[test]
fn beta_null_smooth_size_is_within_monte_carlo_error() {
    assert_null_size_within_monte_carlo_error(Family::Beta);
}

/// Replications of the power control.
const POWER_REPLICATIONS: u64 = 40;

/// Under a real `s(x2) = 0.5·sin(2πx2)` the test must reject at `α = 0.01` in
/// most fits: a gate that only checked size would pass a test that never
/// rejects.
#[test]
fn gaussian_real_smooth_effect_is_detected() {
    init_parallelism();
    let p_values: Vec<f64> = (0..POWER_REPLICATIONS)
        .into_par_iter()
        .map(|rep| {
            tested_row(Family::Gaussian, rep, 0.5)
                .unwrap_or_else(|reason| panic!("power rep {rep}: fit failed: {reason}"))
                .p_value
        })
        .collect();
    let rejected = p_values.iter().filter(|&&p| p <= 0.01).count();
    eprintln!("Gaussian power at α = 0.01: {rejected}/{POWER_REPLICATIONS}");
    assert!(
        rejected as f64 >= 0.9 * POWER_REPLICATIONS as f64,
        "a real s({NULL_TERM}) effect was found at α = 0.01 in only \
         {rejected}/{POWER_REPLICATIONS} fits: {p_values:?}"
    );
}
