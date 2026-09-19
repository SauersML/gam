//! Standing type-I calibration gate for the single-response smooth-term
//! p-value, one test per response family (pyGAM audit, lane
//! `pv-wald-families`).
//!
//! The summary p-value is the variance-component score test of
//! `gam_terms::inference::smooth_score_test`, read off the fit's penalized
//! Hessian and weighted Gram. The family changes three of its inputs: the IRLS
//! weights inside `X'WX`, the dispersion the score is standardized by (profiled,
//! Pearson-refreshed, or fixed), and the reference law (a weighted `χ²₁` sum,
//! over an independent `χ²_ρ/ρ` when the scale is estimated). A wrong weight, a
//! wrong scale predicate, or a residual df read off the wrong fit is invisible
//! to a Gaussian-only gate.
//!
//! Audit: `y ~ s(x1) + s(x2)` with a real `s(x1)` and a TRUE-NULL `s(x2)`
//! (`x2` is drawn independently of `y`), `n = 200`, 200 seeded replications
//! per family (the 500-replication acceptance run is the bench; this is its
//! standing CI-sized gate). The p-value read is the production summary row —
//! the shared `smooth_term_summary_rows` walk, the same call
//! `saved_model_summary` makes for CLI and Python.
//!
//! The gate is two-sided, because a valid p-value has `P(p ≤ a) = a` at every
//! level: a conservative test fails it exactly as a liberal one does. Over the
//! `m` fits that converged, the empirical size at `α ∈ {0.10, 0.05, 0.01}` must
//! lie within `3·MCSE(α)`, `MCSE(α) = √(α(1 − α)/m)`, of `α` on both sides, and
//! the Kolmogorov-Smirnov distance to `U(0, 1)` must be below its asymptotic
//! 0.1% critical value `√(−½·ln(0.0005))/√m`. The score statistic does not
//! depend on the null term's own smoothing parameter, so a term REML shrinks
//! flat puts no point mass near `p = 1`.

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

    /// One response draw at signal `f1 = sin(2πx1)`; `x2` never enters.
    fn draw(self, f1: f64, rng: &mut StdRng) -> f64 {
        match self {
            Self::Gaussian => f1 + Normal::new(0.0, 0.5).expect("normal").sample(rng),
            Self::Poisson => Poisson::new((0.5 + 0.5 * f1).exp())
                .expect("poisson rate")
                .sample(rng),
            Self::Binomial => {
                let p = 1.0 / (1.0 + (-f1).exp());
                if Uniform::new(0.0, 1.0).expect("uniform").sample(rng) < p {
                    1.0
                } else {
                    0.0
                }
            }
            Self::Gamma => {
                let mu = (1.0 + 0.5 * f1).exp();
                let shape = 3.0;
                Gamma::new(shape, mu / shape).expect("gamma").sample(rng)
            }
            Self::NegativeBinomial => {
                // Poisson–gamma mixture with size θ = 2.
                let mu = (1.0 + 0.5 * f1).exp();
                let theta = 2.0;
                let rate = Gamma::new(theta, mu / theta).expect("gamma").sample(rng);
                Poisson::new(rate.max(1e-12)).expect("poisson").sample(rng)
            }
            Self::Tweedie => {
                // Compound Poisson–gamma with p = 1.5, φ = 1.
                let mu = (0.5 + 0.5 * f1).exp();
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
                let mu = 1.0 / (1.0 + (-0.5 * f1).exp());
                let phi = 10.0;
                Beta::new(mu * phi, (1.0 - mu) * phi)
                    .expect("beta")
                    .sample(rng)
                    .clamp(1e-6, 1.0 - 1e-6)
            }
        }
    }
}

struct NullRow {
    p_value: f64,
    edf: f64,
    ref_df: f64,
}

fn null_dataset(family: Family, rep: u64) -> gam::data::EncodedDataset {
    let mut rng = StdRng::seed_from_u64(SEED + 1_000_000 * family.index() + rep);
    let unit = Uniform::new(0.0_f64, 1.0).expect("uniform");
    let rows: Vec<StringRecord> = (0..N_OBS)
        .map(|_| {
            let x1 = unit.sample(&mut rng);
            let x2 = unit.sample(&mut rng);
            let y = family.draw((2.0 * PI * x1).sin(), &mut rng);
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
    .expect("encode null dataset")
}

/// The null term's summary row, or the fit error that stopped it.
fn null_row(family: Family, rep: u64) -> Result<NullRow, String> {
    let data = null_dataset(family, rep);
    let config = FitConfig {
        family: Some(family.config_name().to_string()),
        ..FitConfig::default()
    };
    let result = fit_from_formula(FORMULA, &data, &config).map_err(|e| format!("{e:?}"))?;
    let FitResult::Standard(fit) = result else {
        panic!("{family:?} rep {rep}: expected a standard fit");
    };
    let rows = smooth_term_summary_rows(&fit.design, &fit.resolvedspec, &fit.fit);
    let row = rows
        .iter()
        .find(|row| row.name.contains(NULL_TERM))
        .unwrap_or_else(|| panic!("{family:?} rep {rep}: no summary row for s({NULL_TERM})"));
    let p_value = row.pvalue.unwrap_or_else(|| {
        panic!(
            "{family:?} rep {rep}: the null smooth reported no p-value (edf {}, ref_df {})",
            row.edf, row.ref_df
        )
    });
    Ok(NullRow {
        p_value,
        edf: row.edf,
        ref_df: row.ref_df,
    })
}

fn assert_null_size_within_monte_carlo_error(family: Family) {
    // The fits run on rayon workers, which need the wide worker stack.
    init_parallelism();
    let outcomes: Vec<(u64, Result<NullRow, String>)> = (0..N_REPLICATIONS)
        .into_par_iter()
        .map(|rep| (rep, null_row(family, rep)))
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
        // The reference df `(Σw)²/Σw²` of the score's weighted χ² law is
        // defined, and at least one, for every edf — including a term shrunk
        // below one effective degree of freedom.
        assert!(
            row.ref_df.is_finite() && row.ref_df >= 1.0,
            "{family:?} rep {rep}: ref_df {} undefined or below one at edf {}",
            row.ref_df,
            row.edf
        );
    }

    let m = rows.len() as f64;
    let mut miscalibrated = Vec::new();
    let mut report = Vec::new();
    for &alpha in &ALPHAS {
        let rejections = rows.iter().filter(|(_, r)| r.p_value <= alpha).count();
        let size = rejections as f64 / m;
        let tolerance = 3.0 * (alpha * (1.0 - alpha) / m).sqrt();
        report.push(format!("α={alpha}: size {size:.4} (α ± {tolerance:.4})"));
        if (size - alpha).abs() > tolerance {
            miscalibrated.push(format!(
                "α={alpha}: {rejections}/{} rejections, size {size:.4} outside α ± 3·MCSE = \
                 [{:.4}, {:.4}]",
                rows.len(),
                alpha - tolerance,
                alpha + tolerance
            ));
        }
    }
    let mut sorted: Vec<f64> = rows.iter().map(|(_, r)| r.p_value).collect();
    sorted.sort_by(f64::total_cmp);
    let ks_distance = sorted
        .iter()
        .enumerate()
        .map(|(i, &p)| (p - i as f64 / m).max((i + 1) as f64 / m - p))
        .fold(0.0_f64, f64::max);
    let ks_critical = (-0.5 * (0.001_f64 / 2.0).ln()).sqrt() / m.sqrt();
    report.push(format!("KS D {ks_distance:.4} (critical {ks_critical:.4})"));
    if ks_distance > ks_critical {
        miscalibrated.push(format!(
            "KS distance {ks_distance:.4} to U(0, 1) exceeds the 0.1% critical value \
             {ks_critical:.4}"
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
        "{family:?}: the smooth-term p-value of a true-null s({NULL_TERM}) is not U(0, 1):\n{}",
        miscalibrated.join("\n")
    );
}

#[test]
fn gaussian_null_smooth_wald_size_is_within_monte_carlo_error() {
    assert_null_size_within_monte_carlo_error(Family::Gaussian);
}

#[test]
fn poisson_null_smooth_wald_size_is_within_monte_carlo_error() {
    assert_null_size_within_monte_carlo_error(Family::Poisson);
}

#[test]
fn binomial_null_smooth_wald_size_is_within_monte_carlo_error() {
    assert_null_size_within_monte_carlo_error(Family::Binomial);
}

#[test]
fn gamma_null_smooth_wald_size_is_within_monte_carlo_error() {
    assert_null_size_within_monte_carlo_error(Family::Gamma);
}

#[test]
fn negative_binomial_null_smooth_wald_size_is_within_monte_carlo_error() {
    assert_null_size_within_monte_carlo_error(Family::NegativeBinomial);
}

#[test]
fn tweedie_null_smooth_wald_size_is_within_monte_carlo_error() {
    assert_null_size_within_monte_carlo_error(Family::Tweedie);
}

#[test]
fn beta_null_smooth_wald_size_is_within_monte_carlo_error() {
    assert_null_size_within_monte_carlo_error(Family::Beta);
}
