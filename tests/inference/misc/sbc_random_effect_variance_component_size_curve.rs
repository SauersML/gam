//! Standing type-I size gate for the random-effect variance-component p-value
//! (pyGAM audit, lane `pv-random-effects`).
//!
//! A `group(g)` block is a ridge-penalized one-hot block whose variance
//! component `σ²_b` is ZERO under the null of no group effect — on the
//! boundary of its parameter space, where a Wald or likelihood-ratio statistic
//! referred to `χ²` is not calibrated. The summary row used to carry no
//! p-value at all. It now carries the Lin (1997) / Wood (2013) score test of
//! `σ²_b = 0` scored against its exact finite-sample null law (a weighted sum
//! of `χ²₁`, ratioed against the residual `χ²` when the scale is estimated);
//! `gam_terms::inference::random_effect_test` derives it.
//!
//! Audit: `y ~ s(x1) + group(g)` with a real `s(x1)`, `L = 20` UNBALANCED
//! levels (level shares proportional to the squares of a uniform draw, every
//! level seen) and no group effect, `n = 200`, 200 seeded replications per
//! family. The p-value read is the production summary row — the shared
//! `smooth_term_summary_rows` walk the CLI and Python summaries use. At
//! `α ∈ {0.10, 0.05, 0.01}` the empirical size must stay within
//! `α ± 2·MCSE(α)`, `MCSE(α) = √(α(1 − α)/m)`, and the whole null p-value
//! sample must pass a two-sided Kolmogorov–Smirnov test against `U(0, 1)` at
//! level 0.01: the reference law is exact and continuous (no atom at `p = 1`),
//! so an undersized or conservative test is as much a defect as an oversized
//! one. The 500-rep
//! acceptance run over 5/20/200 levels, balanced and unbalanced, is the bench
//! in `bench/pvalue_calibration/pv-random-effects/`; this is its CI-sized gate.
//!
//! A power control fits the same design with a real group effect and requires
//! the test to find it.

use csv::StringRecord;
use gam::{
    FitConfig, FitResult, encode_recordswith_inferred_schema, fit_from_formula, init_parallelism,
};
use gam_solve::estimate::smooth_term_summary_rows;
use rand::SeedableRng;
use rand::rngs::StdRng;
use rand_distr::{Distribution, Normal, Poisson, Uniform};
use rayon::iter::{IntoParallelIterator, ParallelIterator};
use std::f64::consts::PI;

const N_OBS: usize = 200;
const N_LEVELS: usize = 20;
const N_REPLICATIONS: u64 = 200;
const ALPHAS: [f64; 3] = [0.10, 0.05, 0.01];
/// Level of the two-sided Kolmogorov–Smirnov uniformity test.
const KS_LEVEL: f64 = 0.01;
const SEED: u64 = 0x2E_5EED_0000;
const FORMULA: &str = "y ~ s(x1) + group(g)";
const GROUP_TERM: &str = "g";

#[derive(Clone, Copy, Debug)]
enum Family {
    Gaussian,
    Binomial,
    Poisson,
}

impl Family {
    fn config_name(self) -> &'static str {
        match self {
            Self::Gaussian => "gaussian",
            Self::Binomial => "binomial",
            Self::Poisson => "poisson",
        }
    }

    fn index(self) -> u64 {
        self as u64
    }

    fn draw(self, eta: f64, rng: &mut StdRng) -> f64 {
        match self {
            Self::Gaussian => eta + Normal::new(0.0, 0.5).expect("normal").sample(rng),
            Self::Binomial => {
                let p = 1.0 / (1.0 + (-eta).exp());
                if Uniform::new(0.0, 1.0).expect("uniform").sample(rng) < p {
                    1.0
                } else {
                    0.0
                }
            }
            Self::Poisson => Poisson::new((0.3 + 0.5 * eta).exp())
                .expect("poisson rate")
                .sample(rng),
        }
    }
}

fn dataset(family: Family, rep: u64, group_sd: f64) -> gam::data::EncodedDataset {
    let mut rng = StdRng::seed_from_u64(SEED + 1_000_000 * family.index() + rep);
    let unit = Uniform::new(0.0_f64, 1.0).expect("uniform");
    let effects: Vec<f64> = if group_sd > 0.0 {
        let normal = Normal::new(0.0, group_sd).expect("normal");
        (0..N_LEVELS).map(|_| normal.sample(&mut rng)).collect()
    } else {
        vec![0.0; N_LEVELS]
    };
    let rows: Vec<StringRecord> = (0..N_OBS)
        .map(|row| {
            let x1 = unit.sample(&mut rng);
            let share = unit.sample(&mut rng);
            let g = if row < N_LEVELS {
                row
            } else {
                ((share * share * N_LEVELS as f64) as usize).min(N_LEVELS - 1)
            };
            let y = family.draw((2.0 * PI * x1).sin() + effects[g], &mut rng);
            StringRecord::from(vec![format!("{x1:.17e}"), format!("{g}"), format!("{y:.17e}")])
        })
        .collect();
    encode_recordswith_inferred_schema(
        ["x1", "g", "y"].into_iter().map(String::from).collect(),
        rows,
    )
    .expect("encode dataset")
}

/// The group term's summary p-value.
fn group_p_value(family: Family, rep: u64, group_sd: f64) -> f64 {
    let data = dataset(family, rep, group_sd);
    let config = FitConfig {
        family: Some(family.config_name().to_string()),
        ..FitConfig::default()
    };
    let result = fit_from_formula(FORMULA, &data, &config)
        .unwrap_or_else(|e| panic!("{family:?} rep {rep}: fit failed: {e:?}"));
    let FitResult::Standard(fit) = result else {
        panic!("{family:?} rep {rep}: expected a standard fit");
    };
    let rows = smooth_term_summary_rows(&fit.design, &fit.resolvedspec, &fit.fit);
    let row = rows
        .iter()
        .find(|row| row.name == GROUP_TERM)
        .unwrap_or_else(|| panic!("{family:?} rep {rep}: no summary row for {GROUP_TERM}"));
    let p_value = row.pvalue.unwrap_or_else(|| {
        panic!(
            "{family:?} rep {rep}: the random effect reported no p-value ({:?})",
            row.pvalue_unavailable.map(|reason| reason.label())
        )
    });
    assert!(
        p_value.is_finite() && (0.0..=1.0).contains(&p_value),
        "{family:?} rep {rep}: p-value out of range: {p_value}"
    );
    assert!(
        row.ref_df.is_finite() && row.ref_df > 0.0 && row.ref_df <= (N_LEVELS as f64),
        "{family:?} rep {rep}: effective df {} outside (0, {N_LEVELS}]",
        row.ref_df
    );
    p_value
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

fn assert_null_size_within_monte_carlo_error(family: Family) {
    init_parallelism();
    let p_values: Vec<f64> = (0..N_REPLICATIONS)
        .into_par_iter()
        .map(|rep| group_p_value(family, rep, 0.0))
        .collect();
    let m = p_values.len() as f64;
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
    eprintln!("{family:?}: {}", report.join("; "));
    assert!(
        miscalibrated.is_empty(),
        "{family:?}: the random-effect variance-component test is miscalibrated under a \
         true-null group effect:\n{}",
        miscalibrated.join("\n")
    );
}

#[test]
fn gaussian_null_random_effect_size_is_within_monte_carlo_error() {
    assert_null_size_within_monte_carlo_error(Family::Gaussian);
}

#[test]
fn binomial_null_random_effect_size_is_within_monte_carlo_error() {
    assert_null_size_within_monte_carlo_error(Family::Binomial);
}

#[test]
fn poisson_null_random_effect_size_is_within_monte_carlo_error() {
    assert_null_size_within_monte_carlo_error(Family::Poisson);
}

/// Under a real group effect (sd 1) the rejection rate at `α = 0.05` over
/// `N_POWER_REPLICATIONS` seeded fits must clear the null band `α + 2·MCSE(α)`:
/// the test has power beyond its size.
#[test]
fn a_real_group_effect_is_detected() {
    const N_POWER_REPLICATIONS: u64 = 40;
    const ALPHA: f64 = 0.05;
    init_parallelism();
    let m = N_POWER_REPLICATIONS as f64;
    let null_band = ALPHA + 2.0 * (ALPHA * (1.0 - ALPHA) / m).sqrt();
    for family in [Family::Gaussian, Family::Binomial, Family::Poisson] {
        let rejections = (0..N_POWER_REPLICATIONS)
            .into_par_iter()
            .filter(|&rep| group_p_value(family, rep, 1.0) <= ALPHA)
            .count();
        let power = rejections as f64 / m;
        eprintln!("{family:?}: power at α={ALPHA} is {power:.3} (null band {null_band:.3})");
        assert!(
            power > null_band,
            "{family:?}: a group effect with sd 1 was rejected in {rejections}/{N_POWER_REPLICATIONS} \
             fits at α={ALPHA}, not above the null band {null_band:.3}"
        );
    }
}
