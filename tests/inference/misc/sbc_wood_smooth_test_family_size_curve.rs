//! Standing type-I size gate for the single-response Wood (2013) smooth-term
//! Wald p-value, one test per response family (pyGAM audit, lane
//! `pv-wald-families`).
//!
//! The registry's `wood_smooth_test_pvalue` target was audited on Gaussian
//! data only. The same `wood_smooth_test` primitive is reached by every
//! standard family, and the family changes three of its inputs: the IRLS
//! weights inside the whitening Gram `X'WX`, the covariance `Vb` (whose scale
//! is profiled, Pearson-refreshed, or fixed), and the reference distribution
//! (`F(ref_df, n − edf)` when the scale is estimated, `χ²(ref_df)` when it is
//! not). A wrong weight, a wrong scale predicate, or a residual df read off the
//! wrong fit is invisible to a Gaussian-only gate.
//!
//! Audit: `y ~ s(x1) + s(x2)` with a real `s(x1)` and a TRUE-NULL `s(x2)`
//! (`x2` is drawn independently of `y`), `n = 200`, 200 seeded replications
//! per family (the 500-replication acceptance run, at n = 60, 200 and 2000, is
//! the bench; this is its standing CI-sized gate). The p-value read is the
//! production summary row — the shared
//! `smooth_term_summary_rows` walk with the fit's exact weighted Gram, the same
//! call `saved_model_summary` makes for CLI and Python. At
//! `α ∈ {0.10, 0.05, 0.01}` the empirical size must not exceed
//! `α + 2·MCSE(α)`, `MCSE(α) = √(α(1 − α)/m)`, over the `m` fits that
//! converged. An undersized (conservative) test passes. The measured sizes and
//! the Monte-Carlo study behind them are in
//! `bench/pvalue_calibration/pv-wald-families/`.
//!
//! What this gate does NOT assert is uniformity of the p-value below 0.5. On
//! a null term REML shrinks the fit to the boundary, and conditional on that
//! λ̂ the statistic is not a χ²: in the one-direction case `T = max(z² − 1, 0)`
//! against a `χ²₁` reference, which is conservative at every level and
//! front-loaded below 0.5 (the README derives it). mgcv's `testStat` refers a
//! sub-unit-edf term to the same `χ²₁`, so it shares the shape.

use csv::StringRecord;
use gam::{
    FitConfig, FitResult, encode_recordswith_inferred_schema, fit_from_formula, init_parallelism,
};
use gam_solve::estimate::{SummaryBlockOffset, smooth_term_summary_rows};
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
    let rows = smooth_term_summary_rows(
        &fit.design,
        &fit.fit,
        fit.fit.weighted_gram(),
        SummaryBlockOffset::default(),
    );
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
        // The reference df is defined, and at least one, for every edf —
        // including a term shrunk below one effective degree of freedom.
        assert!(
            row.ref_df.is_finite() && row.ref_df >= 1.0,
            "{family:?} rep {rep}: ref_df {} undefined or below one at edf {}",
            row.ref_df,
            row.edf
        );
        // A term REML switched off carries no evidence against the null: the
        // statistic vanishes with the shrunk coefficients, so the p-value
        // must sit in the upper half, never near a rejection.
        if row.edf < 0.01 {
            assert!(
                row.p_value > 0.5,
                "{family:?} rep {rep}: edf {} term reported p = {}",
                row.edf,
                row.p_value
            );
        }
    }

    let m = rows.len() as f64;
    let mut oversized = Vec::new();
    let mut report = Vec::new();
    for &alpha in &ALPHAS {
        let rejections = rows.iter().filter(|(_, r)| r.p_value <= alpha).count();
        let size = rejections as f64 / m;
        let bound = alpha + 2.0 * (alpha * (1.0 - alpha) / m).sqrt();
        report.push(format!("α={alpha}: size {size:.4} (bound {bound:.4})"));
        if size > bound {
            oversized.push(format!(
                "α={alpha}: {rejections}/{} rejections, size {size:.4} > α + 2·MCSE = {bound:.4}",
                rows.len()
            ));
        }
    }
    eprintln!(
        "{family:?}: {} usable fits, {} failed; {}",
        rows.len(),
        failed_fits.len(),
        report.join("; ")
    );
    assert!(
        oversized.is_empty(),
        "{family:?}: the Wood smooth-term Wald test rejects a true-null s({NULL_TERM}) too \
         often (anti-conservative):\n{}",
        oversized.join("\n")
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
