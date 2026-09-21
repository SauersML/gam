//! #3573: the p-value of a linear term under its default null-recovery ridge.
//!
//! Every parametric linear term carries a REML-selected `LinearTermRidge`, so
//! under `H₀: β = 0` REML sends λ to its null rail in a large fraction of
//! samples. The summary row used to divide the shrunk `β̂` by the equally
//! shrunk standard error, so the Wald p-value collapsed to `p ≈ 1` there (the
//! committed baseline had 69/100 Gaussian null reps above 0.999, KS D 0.69).
//! A ridged slope is the one-column variance component, so its row now
//! reports the variance-component score test the `group()` rows use.
//!
//! Two gates, both on the production summary row
//! (`parametric_term_summary_rows`, the walk the CLI and Python read):
//!
//! - Exactness. For a Gaussian fit the one-column score test of `β = 0` with
//!   an estimated scale is the partial `F` test of the unpenalized regression
//!   on the same design, computed here independently by orthogonalization.
//! - Size. `y ~ s(x1) + x2` with `x2` independent of `y`, 200 seeded
//!   replications per family: at `α ∈ {0.10, 0.05, 0.01}` the size stays in
//!   `α ± 2·MCSE(α)` and the null p-values pass a two-sided KS test against
//!   `U(0, 1)` at level 0.01 — conservative is as much a defect as liberal.
//!   A power control requires a real slope to be found.

use csv::StringRecord;
use gam::{
    FitConfig, FitResult, encode_recordswith_inferred_schema, fit_from_formula, init_parallelism,
};
use gam_math::probability::student_t_two_sided_probability;
use gam_solve::estimate::{ParametricTest, SummaryBlockOffset, parametric_term_summary_rows};
use gam_terms::inference::random_effect_test::RandomEffectTestOutcome;
use ndarray::{Array1, Array2, Axis};
use rand::SeedableRng;
use rand::rngs::StdRng;
use rand_distr::{Distribution, Normal, Poisson, Uniform};
use rayon::iter::{IntoParallelIterator, ParallelIterator};
use std::f64::consts::PI;

const N_OBS: usize = 200;
const N_REPLICATIONS: u64 = 200;
const ALPHAS: [f64; 3] = [0.10, 0.05, 0.01];
/// Level of the two-sided Kolmogorov–Smirnov uniformity test.
const KS_LEVEL: f64 = 0.01;
const SEED: u64 = 0x35_73_0000;
const FORMULA: &str = "y ~ s(x1) + x2";
const LINEAR_TERM: &str = "x2";

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

/// The seeded draws `(x1, x2, y)` of one replication.
fn draws(family: Family, rep: u64, slope: f64) -> Vec<[f64; 3]> {
    let mut rng = StdRng::seed_from_u64(SEED + 1_000_000 * family as u64 + rep);
    let unit = Uniform::new(0.0_f64, 1.0).expect("uniform");
    let normal = Normal::new(0.0, 1.0).expect("normal");
    (0..N_OBS)
        .map(|_| {
            let x1 = unit.sample(&mut rng);
            let x2 = normal.sample(&mut rng);
            let y = family.draw((2.0 * PI * x1).sin() + slope * x2, &mut rng);
            [x1, x2, y]
        })
        .collect()
}

fn dataset(rows: &[[f64; 3]]) -> gam::data::EncodedDataset {
    // `{:e}` prints the shortest string that parses back to the same `f64`,
    // so the fit sees exactly the draws the reference computation uses.
    let records = rows
        .iter()
        .map(|row| StringRecord::from(row.iter().map(|v| format!("{v:e}")).collect::<Vec<_>>()))
        .collect();
    encode_recordswith_inferred_schema(
        ["x1", "x2", "y"].into_iter().map(String::from).collect(),
        records,
    )
    .expect("encode dataset")
}

fn fit(family: Family, rep: u64, slope: f64) -> gam::StandardFitResult {
    let config = FitConfig {
        family: Some(family.config_name().to_string()),
        ..FitConfig::default()
    };
    match fit_from_formula(FORMULA, &dataset(&draws(family, rep, slope)), &config)
        .unwrap_or_else(|e| panic!("{family:?} rep {rep}: fit failed: {e:?}"))
    {
        FitResult::Standard(fit) => fit,
        _ => panic!("{family:?} rep {rep}: expected a standard fit"),
    }
}

/// The ridged term's summary row: its score statistic and p-value.
fn linear_row(family: Family, fitted: &gam::StandardFitResult, rep: u64) -> (f64, f64) {
    let rows = parametric_term_summary_rows(
        &fitted.design,
        &fitted.resolvedspec,
        &fitted.fit,
        SummaryBlockOffset::default(),
    );
    let row = rows
        .iter()
        .find(|row| row.name == LINEAR_TERM)
        .unwrap_or_else(|| panic!("{family:?} rep {rep}: no summary row for {LINEAR_TERM}"));
    assert_eq!(
        row.test,
        ParametricTest::VarianceComponentScore,
        "{family:?} rep {rep}: the default linear term must carry the null-recovery ridge \
         and so the score test"
    );
    let p_value = row.pvalue.unwrap_or_else(|| {
        panic!(
            "{family:?} rep {rep}: the ridged linear term reported no p-value ({:?})",
            row.pvalue_unavailable.map(|reason| reason.label())
        )
    });
    assert!(
        p_value.is_finite() && (0.0..=1.0).contains(&p_value),
        "{family:?} rep {rep}: p-value out of range: {p_value}"
    );
    let statistic = row
        .statistic
        .unwrap_or_else(|| panic!("{family:?} rep {rep}: the score row carries no statistic"));
    (statistic, p_value)
}

/// Orthonormalize `column` against the orthonormal columns of `basis` by
/// modified Gram–Schmidt with one reorthogonalization pass ("twice is
/// enough", Giraud et al. 2005), which keeps the result orthogonal to
/// rounding.
fn orthogonalize(basis: &[Array1<f64>], column: &Array1<f64>) -> Array1<f64> {
    let mut v = column.clone();
    for _ in 0..2 {
        for q in basis {
            let coefficient = q.dot(&v);
            v.scaled_add(-coefficient, q);
        }
    }
    v
}

/// The unpenalized partial `F` test of column `tested` of `x`: `(t, ν)` with
/// `t = x̃ᵀy/(‖x̃‖·σ̂)`, `x̃` the column's residual on the other columns and
/// `σ̂² = RSS/ν` from the full least-squares fit.
fn partial_t(x: &Array2<f64>, y: &Array1<f64>, tested: usize) -> (f64, f64) {
    let mut basis: Vec<Array1<f64>> = Vec::new();
    // A column the earlier ones already span leaves only rounding behind; its
    // residual is below `√n·ε` of its own norm and adds no direction.
    let span_floor = (x.nrows() as f64).sqrt() * f64::EPSILON;
    for (j, column) in x.axis_iter(Axis(1)).enumerate() {
        if j == tested {
            continue;
        }
        let column = column.to_owned();
        let v = orthogonalize(&basis, &column);
        let norm = v.dot(&v).sqrt();
        if norm > span_floor * column.dot(&column).sqrt() {
            basis.push(v / norm);
        }
    }
    let x_tilde = orthogonalize(&basis, &x.column(tested).to_owned());
    let x_tilde_norm = x_tilde.dot(&x_tilde).sqrt();
    basis.push(&x_tilde / x_tilde_norm);
    let residual = orthogonalize(&basis, y);
    let nu = (x.nrows() - basis.len()) as f64;
    let sigma = (residual.dot(&residual) / nu).sqrt();
    (x_tilde.dot(y) / (x_tilde_norm * sigma), nu)
}

#[test]
fn gaussian_ridged_linear_score_test_is_the_unpenalized_partial_f_test() {
    init_parallelism();
    for rep in 0..5 {
        for slope in [0.0, 0.15] {
            let fitted = fit(Family::Gaussian, rep, slope);
            let (statistic, p_value) = linear_row(Family::Gaussian, &fitted, rep);
            let x = fitted.design.design.to_dense();
            let y: Array1<f64> = draws(Family::Gaussian, rep, slope)
                .iter()
                .map(|row| row[2])
                .collect();
            let range = fitted
                .design
                .linear_ranges
                .iter()
                .find(|(name, _)| name == LINEAR_TERM)
                .map(|(_, range)| range.clone())
                .expect("x2 coefficient range");
            let (t_reference, nu) = partial_t(&x, &y, range.start);
            let p_reference = student_t_two_sided_probability(t_reference, nu);
            let record = fitted
                .fit
                .artifacts
                .random_effect_tests
                .iter()
                .find(|record| record.term == LINEAR_TERM)
                .expect("recorded score test");
            let RandomEffectTestOutcome::Tested(test) = &record.outcome else {
                panic!("rep {rep}: the score test was not computed: {:?}", record.outcome);
            };
            eprintln!(
                "#3573 rep {rep} slope {slope}: score t {statistic:.10} vs partial t \
                 {t_reference:.10} (ν {nu} vs {:?}); p {p_value:.10e} vs {p_reference:.10e}",
                test.residual_df
            );
            assert_eq!(test.residual_df, Some(nu), "rep {rep}: residual df");
            // Both sides are backward-stable orthogonal projections of the same
            // columns; the design's spline block is conditioned well inside
            // 1e6, so the two agree to κ·ε ≈ 1e-10 relative. 1e-8 leaves two
            // orders of margin and still rejects any approximation.
            let relative = (statistic - t_reference).abs() / t_reference.abs().max(1.0);
            assert!(
                relative <= 1e-8,
                "rep {rep} slope {slope}: score root {statistic} is not the partial t \
                 {t_reference} (relative {relative:.3e})"
            );
            // A relative error δ in t moves the two-sided t tail by at most
            // (1 + t²)·δ relatively; the tail evaluation adds its own bound.
            let tail_tolerance =
                test.p_value_relative_error + (1.0 + t_reference * t_reference) * 1e-8;
            assert!(
                (p_value - p_reference).abs() <= tail_tolerance * p_reference,
                "rep {rep} slope {slope}: p-value {p_value:e} is not the partial F p-value \
                 {p_reference:e}"
            );
        }
    }
}

/// Two-sided one-sample Kolmogorov–Smirnov test of `values` against `U(0, 1)`:
/// the distance `D = sup |F_m − F|` and its asymptotic p-value `Q(√m·D)`.
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
        .map(|rep| linear_row(family, &fit(family, rep, 0.0), rep).1)
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
    let near_one = p_values.iter().filter(|&&p| p > 0.999).count();
    let (ks_distance, ks_p_value) = ks_uniform(&p_values);
    report.push(format!("p > 0.999: {near_one}; KS D {ks_distance:.4}, p {ks_p_value:.4}"));
    if ks_p_value < KS_LEVEL {
        miscalibrated.push(format!(
            "null p-values are not U(0, 1): two-sided KS D = {ks_distance:.4}, \
             p = {ks_p_value:.4} < {KS_LEVEL}"
        ));
    }
    eprintln!("#3573 {family:?}: {}", report.join("; "));
    assert!(
        miscalibrated.is_empty(),
        "{family:?}: the ridged linear term's p-value is miscalibrated under a true-null \
         slope:\n{}",
        miscalibrated.join("\n")
    );
}

#[test]
fn gaussian_null_ridged_linear_term_size_is_within_monte_carlo_error() {
    assert_null_size_within_monte_carlo_error(Family::Gaussian);
}

#[test]
fn binomial_null_ridged_linear_term_size_is_within_monte_carlo_error() {
    assert_null_size_within_monte_carlo_error(Family::Binomial);
}

#[test]
fn poisson_null_ridged_linear_term_size_is_within_monte_carlo_error() {
    assert_null_size_within_monte_carlo_error(Family::Poisson);
}

/// Under a real slope the rejection rate at `α = 0.05` over
/// `N_POWER_REPLICATIONS` seeded fits must clear the null band
/// `α + 2·MCSE(α)`.
#[test]
fn a_real_ridged_linear_slope_is_detected() {
    const N_POWER_REPLICATIONS: u64 = 40;
    const ALPHA: f64 = 0.05;
    const SLOPE: f64 = 0.3;
    init_parallelism();
    let m = N_POWER_REPLICATIONS as f64;
    let null_band = ALPHA + 2.0 * (ALPHA * (1.0 - ALPHA) / m).sqrt();
    for family in [Family::Gaussian, Family::Binomial, Family::Poisson] {
        let rejections = (0..N_POWER_REPLICATIONS)
            .into_par_iter()
            .filter(|&rep| linear_row(family, &fit(family, rep, SLOPE), rep).1 <= ALPHA)
            .count();
        let power = rejections as f64 / m;
        eprintln!("#3573 {family:?}: power at α={ALPHA} is {power:.3} (null band {null_band:.3})");
        assert!(
            power > null_band,
            "{family:?}: a slope of {SLOPE} was rejected in {rejections}/{N_POWER_REPLICATIONS} \
             fits at α={ALPHA}, not above the null band {null_band:.3}"
        );
    }
}
