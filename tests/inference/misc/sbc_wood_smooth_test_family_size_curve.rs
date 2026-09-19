//! Standing null-calibration gate for the single-response smooth-term Wald
//! p-value, one test per response family (pyGAM audit, lane
//! `pv-wald-families`).
//!
//! The same smooth-term Wald primitive is reached by every standard family,
//! and the family changes three of its inputs: the IRLS weights inside the
//! penalized Hessian, the covariance `Vb` (whose scale is profiled,
//! Pearson-refreshed, or fixed), and the reference law. A wrong weight, a
//! wrong scale predicate, or a residual df read off the wrong fit is invisible
//! to a Gaussian-only gate.
//!
//! Audit: `y ~ s(x1) + s(x2)` with a real `s(x1)` and a TRUE-NULL `s(x2)`
//! (`x2` is drawn independently of `y`), `n = 200`, 200 seeded replications
//! per family. The p-value read is the production summary row — the shared
//! `smooth_term_summary_rows` walk, the same call `saved_model_summary` makes
//! for CLI and Python.
//!
//! Under the null the p-value must be U(0, 1) over the whole range, so the
//! gate is two-sided. A conservative p-value (a pile near one, a size below
//! nominal) fails exactly as an anti-conservative one does:
//!
//! * at `α ∈ {0.10, 0.05, 0.01}` the empirical size over the `m` fits that
//!   converged is within `3·MCSE(α)` of `α`, `MCSE(α) = √(α(1 − α)/m)`;
//! * the Kolmogorov distance of the p-values from U(0, 1) is below the
//!   Dvoretzky–Kiefer–Wolfowitz radius `√(ln(2/δ)/(2m))` at the same
//!   three-sigma level `δ = 2Φ(−3)`.
//!
//! A null term REML shrinks onto its penalty null space is in every family's
//! sample; its p-value is held to the same uniform law, with no exemption.
//! The measured laws are in `bench/pvalue_calibration/pv-wald-families/`.

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
/// Width of every acceptance band, in standard errors.
const BAND_SIGMAS: f64 = 3.0;
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

fn assert_null_p_value_is_uniform(family: Family) {
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
        // `ref_df` is the null mean `Σ_k 1/(1 + e_k)` of the whitened
        // statistic at `λ̂`, so a term REML shrank onto its null space
        // honestly reports about its edf, below one. The p-value does not
        // read it: its law is the λ̂-selection replay, checked below.
        assert!(
            row.ref_df.is_finite() && row.ref_df >= 0.0,
            "{family:?} rep {rep}: ref_df {} undefined or negative at edf {}",
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
        let band = BAND_SIGMAS * (alpha * (1.0 - alpha) / m).sqrt();
        report.push(format!("α={alpha}: size {size:.4} (α ± {band:.4})"));
        if (size - alpha).abs() > band {
            miscalibrated.push(format!(
                "α={alpha}: {rejections}/{} rejections, size {size:.4} outside α ± 3·MCSE = \
                 [{:.4}, {:.4}]",
                rows.len(),
                alpha - band,
                alpha + band
            ));
        }
    }
    let mut p_values: Vec<f64> = rows.iter().map(|(_, r)| r.p_value).collect();
    let distance = kolmogorov_distance_from_uniform(&mut p_values);
    let radius = dkw_radius(rows.len());
    report.push(format!("KS D {distance:.4} (DKW radius {radius:.4})"));
    if distance > radius {
        let above = p_values.iter().filter(|&&p| p > 0.99).count();
        miscalibrated.push(format!(
            "Kolmogorov distance {distance:.4} from U(0,1) exceeds the DKW radius {radius:.4}; \
             {above}/{} p-values are above 0.99",
            rows.len()
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
        "{family:?}: the smooth-term Wald p-value of a true-null s({NULL_TERM}) is not \
         U(0,1):\n{}",
        miscalibrated.join("\n")
    );
}

/// `sup_u |F̂(u) − u|` of the sample against U(0, 1).
fn kolmogorov_distance_from_uniform(values: &mut [f64]) -> f64 {
    values.sort_by(f64::total_cmp);
    let m = values.len() as f64;
    values
        .iter()
        .enumerate()
        .map(|(i, &u)| ((i + 1) as f64 / m - u).max(u - i as f64 / m))
        .fold(0.0, f64::max)
}

/// The Dvoretzky–Kiefer–Wolfowitz radius (Massart's constant) exceeded with
/// probability at most `δ = 2Φ(−BAND_SIGMAS)`, the level of the size bands.
fn dkw_radius(m: usize) -> f64 {
    let delta = libm::erfc(BAND_SIGMAS / std::f64::consts::SQRT_2);
    ((2.0 / delta).ln() / (2.0 * m as f64)).sqrt()
}

#[test]
fn gaussian_null_smooth_wald_p_value_is_uniform() {
    assert_null_p_value_is_uniform(Family::Gaussian);
}

#[test]
fn poisson_null_smooth_wald_p_value_is_uniform() {
    assert_null_p_value_is_uniform(Family::Poisson);
}

#[test]
fn binomial_null_smooth_wald_p_value_is_uniform() {
    assert_null_p_value_is_uniform(Family::Binomial);
}

#[test]
fn gamma_null_smooth_wald_p_value_is_uniform() {
    assert_null_p_value_is_uniform(Family::Gamma);
}

#[test]
fn negative_binomial_null_smooth_wald_p_value_is_uniform() {
    assert_null_p_value_is_uniform(Family::NegativeBinomial);
}

#[test]
fn tweedie_null_smooth_wald_p_value_is_uniform() {
    assert_null_p_value_is_uniform(Family::Tweedie);
}

#[test]
fn beta_null_smooth_wald_p_value_is_uniform() {
    assert_null_p_value_is_uniform(Family::Beta);
}
